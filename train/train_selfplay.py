#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""From-random retrain (design rule): warm-start from the current generation's net on a RECENT WINDOW
of self-play rows (default: the last 500k), no data from any other engine, no hand-coded knowledge.

Self-play chunk format (zero selfplay, append-only): sp_rec.bin (136 B records), sp_pol.bin (96 x {int16 policy
index, float16 prob} = the Gumbel completed-Q policy), sp_res.bin (int8 result, RY view), sp_q.bin (float32 root Q,
mover-team view), sp_ply.bin (int16 plies to end), sp_mat.bin (int8 final material balance, RY view; optional).

  python train_selfplay.py --init runs/rand/net_gen3.pt --data runs/rand/gen1 --data runs/rand/gen2 --data runs/rand/gen3
      --out runs/rand/train_gen4 --steps 2000 --batch 1024 --window 500000

Losses (mover-team view): policy = KL(target || net); value = soft cross-entropy on the WDL head whose target has
expectation  value_mix * game_result + (1 - value_mix) * root_search_value  (0.5/0.5 mix), same scalar as a
smooth-L1 target for the score head; moves-left smooth-L1 on plies-left/100; auxiliary material head smooth-L1 on
final material/20. Best-holdout selection by the NEWEST 5% of the window (KL + 0.5 * WDL), exported via export_best.py.
--overfit-rows N: the tiny-overfit sanity test (train and evaluate on the same N rows; the loss must go near zero).
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ZeroNet, net_for, in_planes_of, CELLS
from encoder import GpuEncoder, REC

POL = np.dtype([("i", "<i2"), ("p", "<f2")])
HIST_POS = 7
HIST_ROW = HIST_POS * REC + 1          # input history: 7 x 136 history bytes, newest first, then nhist
AUX = 26                               # sp_aux.bin, byte 0 = king-capture + in_check bits, 1..25 = a 196-bit mask


class SelfplaySet:
    def __init__(self, dirs):
        self.rec, self.pol, self.res, self.q, self.ply, self.mat = [], [], [], [], [], []
        self.hist = []   # input history: sp_hist.bin, 7 x 136 bytes + 1 nhist per row; absent in pre-step-d dirs
        self.ok_per_dir = []   # data gate: False for rows the uniform stand-in produced
        self.aux = []          # sp_aux.bin, absent in pre-order-3 gen dirs, which stay readable
        for d in dirs:
            n = os.path.getsize(os.path.join(d, "sp_rec.bin")) // REC
            if n == 0:
                continue
            self.rec.append(np.memmap(os.path.join(d, "sp_rec.bin"), dtype=np.uint8, mode="r", shape=(n, REC)))
            self.pol.append(np.memmap(os.path.join(d, "sp_pol.bin"), dtype=POL, mode="r", shape=(n, 96)))
            self.res.append(np.memmap(os.path.join(d, "sp_res.bin"), dtype=np.int8, mode="r", shape=(n,)))
            self.q.append(np.memmap(os.path.join(d, "sp_q.bin"), dtype=np.float32, mode="r", shape=(n,)))
            # DATA GATE (2026-10-01): the uniform stand-in evaluator returns value 0 for every position, so a row it
            # produced has a root Q of EXACTLY 0. A real net does not: measured 0 of 225,825 rows in gen29, against
            # 51,466 of 109,160 in gen30 after a 210-plane binary was pointed at a 41-plane net and fell back silently.
            # Entropy cannot do this job - clean gen29 has 7,759 rows above 2.0 nats legitimately - but |q| < 1e-6 is
            # exact. The engine now refuses to run at all in that situation; this gate covers data already on disk.
            okd = np.abs(np.asarray(self.q[-1])) >= 1e-6
            if not okd.all():
                print(f"DATA GATE: {d} drops {int((~okd).sum()):,} of {n:,} rows with |root Q| < 1e-6 "
                      f"(uniform stand-in signature)", flush=True)
            self.ok_per_dir.append(okd)
            self.ply.append(np.memmap(os.path.join(d, "sp_ply.bin"), dtype=np.int16, mode="r", shape=(n,)))
            mp = os.path.join(d, "sp_mat.bin")
            self.mat.append(np.memmap(mp, dtype=np.int8, mode="r", shape=(n,)) if os.path.exists(mp) and os.path.getsize(mp) >= n else None)
            ap = os.path.join(d, "sp_aux.bin")
            # EXACT size only. sp_aux.bin started mid-generation in gen31, so a ">=" test would one day
            # pass on a file whose row 0 is not the dir's row 0 and attach every label to the WRONG
            # position - the silent-misalignment class that poisoned gen30 this morning.
            aux_ok = os.path.exists(ap) and os.path.getsize(ap) == n * AUX
            if os.path.exists(ap) and not aux_ok:
                print(f"DATA GATE: {d} sp_aux.bin is {os.path.getsize(ap) // AUX:,} rows against {n:,} records "
                      f"- PARTIAL, so the aux labels for this dir are ignored", flush=True)
            self.aux.append(np.memmap(ap, dtype=np.uint8, mode="r", shape=(n, AUX)) if aux_ok else None)
            hp = os.path.join(d, "sp_hist.bin")
            hist_ok = os.path.exists(hp) and os.path.getsize(hp) == n * HIST_ROW
            if os.path.exists(hp) and not hist_ok:
                print(f"DATA GATE: {d} sp_hist.bin is {os.path.getsize(hp) // HIST_ROW:,} rows against {n:,} records "
                      f"- PARTIAL, so history for this dir is zeroed", flush=True)
            self.hist.append(np.memmap(hp, dtype=np.uint8, mode="r", shape=(n, HIST_ROW)) if hist_ok else None)
        self.sizes = [len(r) for r in self.rec]
        self.offsets = np.cumsum([0] + self.sizes)
        self.n = int(self.offsets[-1])
        self.ok = np.concatenate(self.ok_per_dir) if self.ok_per_dir else np.ones(0, bool)

    def valid(self, lo, hi):
        """Row indices in [lo, hi) that the data gate accepts. The gate must SELECT rows, not just
        count them, or it is decoration."""
        v = np.flatnonzero(self.ok[lo:hi]) + lo
        return v if v.size else np.arange(lo, hi)   # never hand back an empty training set

    def batch(self, idx):
        idx = np.sort(idx)
        recs, pols, ress, qs, plys, mats, hists = [], [], [], [], [], [], []
        auxs, auxok = [], []        # ; auxok marks rows whose dir actually HAS sp_aux.bin
        for k in range(len(self.rec)):
            sel = idx[(idx >= self.offsets[k]) & (idx < self.offsets[k + 1])] - self.offsets[k]
            if sel.size == 0:
                continue
            recs.append(np.asarray(self.rec[k][sel])); pols.append(np.asarray(self.pol[k][sel]))
            ress.append(np.asarray(self.res[k][sel])); qs.append(np.asarray(self.q[k][sel])); plys.append(np.asarray(self.ply[k][sel]))
            mats.append(np.asarray(self.mat[k][sel]) if self.mat[k] is not None else np.zeros(sel.size, np.int8))
            hists.append(np.asarray(self.hist[k][sel]) if self.hist[k] is not None
                         else np.zeros((sel.size, HIST_ROW), np.uint8))
            auxs.append(np.asarray(self.aux[k][sel]) if self.aux[k] is not None
                        else np.zeros((sel.size, AUX), np.uint8))
            auxok.append(np.full(sel.size, self.aux[k] is not None, bool))
        rec = np.concatenate(recs); pol = np.concatenate(pols); res = np.concatenate(ress).astype(np.float32)
        q = np.concatenate(qs).astype(np.float32); ply = np.concatenate(plys).astype(np.float32); mat = np.concatenate(mats).astype(np.float32)
        stm = rec[:, 0].astype(np.int64)
        sign = np.where(stm % 2 == 0, 1.0, -1.0).astype(np.float32)   # RY view -> mover-team view
        hb = np.concatenate(hists)
        ab = np.concatenate(auxs); aok = np.concatenate(auxok)
        # targets, unpacked here so the training loop never touches bit layout:
        #   kcap [B,4] = "a legal move of the mover captures seat k's king"  (byte 0, bits 0..3)
        #   hang [B,196] = "our piece on this mover-frame cell is attacked and not defended" (bytes 1..25)
        kcap = ((ab[:, 0:1] >> np.arange(4, dtype=np.uint8)[None, :]) & 1).astype(np.float32)
        hang = np.unpackbits(ab[:, 1:26], axis=1, bitorder="little")[:, :CELLS].astype(np.float32)
        return (rec, pol["i"].astype(np.int64), pol["p"].astype(np.float32), res * sign, np.clip(q, -1, 1),
                ply / 100.0, mat * sign / 20.0,
                hb[:, :HIST_POS * REC].reshape(-1, HIST_POS, REC), hb[:, HIST_POS * REC],
                kcap, hang, aok.astype(np.float32))


def policy_kl(logits, idx, prob):
    """KL(target || softmax(logits)) over the stored slots (idx == -1 masked)."""
    mask = idx >= 0
    safe = idx.clamp(min=0)
    lp = F.log_softmax(logits.float(), dim=1).gather(1, safe)
    p = prob * mask
    p = p / p.sum(1, keepdim=True).clamp(min=1e-6)
    return -(p * lp).sum(1).mean()


def wdl_target(z, q, mix):
    """Soft [W, D, L] target with expectation (W - L) = mix * z + (1 - mix) * q (mover-team view)."""
    zt = torch.stack([(z > 0).float(), (z == 0).float(), (z < 0).float()], 1)
    qt = torch.stack([q.clamp(min=0), 1 - q.abs(), (-q).clamp(min=0)], 1)
    return mix * zt + (1 - mix) * qt


def soft_ce(logits, target):
    return -(target * F.log_softmax(logits.float(), dim=1)).sum(1).mean()


def infer_arch(init):
    if init.endswith(".pt"):
        sd = torch.jit.load(init, map_location="cpu").state_dict()
        blocks = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
        return sd, blocks, int(sd["stem.0.weight"].shape[0])
    ck = torch.load(init, map_location="cpu", weights_only=False)
    return ck["net"], int(ck["blocks"]), int(ck["filters"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="warm-start weights: a .pt (TorchScript) or .ckpt; the file is logged")
    ap.add_argument("--data", action="append", required=True, help="self-play chunk dir (repeatable, OLDEST first)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--holdout-frac", type=float, default=0.05)
    ap.add_argument("--window", type=int, default=500_000, help="train on the LAST N rows of the data (0 = all)")
    ap.add_argument("--value-mix", type=float, default=0.5, help="value target = mix * result + (1 - mix) * root search value")
    ap.add_argument("--overfit-rows", type=int, default=0, help="sanity test: train AND evaluate on the same first N rows")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--history", action="store_true", help="input history: feed the sp_hist.bin planes (210 inputs instead of 41)")
    ap.add_argument("--aux-weight", type=float, default=0.15, help="weight on the tactical auxiliary heads (the range 0.1-0.25)")
    a = ap.parse_args()
    a.lr_explicit = any(x.startswith("--lr") for x in sys.argv[1:])   # only an explicit --lr overrides the KataGo per-sample lr
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.backends.cudnn.benchmark = True
    os.makedirs(a.out, exist_ok=True)
    sp = SelfplaySet(a.data)
    lo = max(0, sp.n - a.window) if a.window > 0 else 0
    n_w = sp.n - lo
    if a.overfit_rows > 0:
        train_lo, train_hi, hold_lo, hold_hi = lo, min(sp.n, lo + a.overfit_rows), lo, min(sp.n, lo + a.overfit_rows)
    else:
        hold_n = max(min(1024, n_w // 4), int(n_w * a.holdout_frac))
        train_lo, train_hi, hold_lo, hold_hi = lo, sp.n - hold_n, sp.n - hold_n, sp.n
    train_idx = sp.valid(train_lo, train_hi)
    hold_idx = sp.valid(hold_lo, hold_hi)
    if train_idx.size != train_hi - train_lo or hold_idx.size != hold_hi - hold_lo:
        print(f"DATA GATE: train {train_hi - train_lo:,} -> {train_idx.size:,} rows, "
              f"holdout {hold_hi - hold_lo:,} -> {hold_idx.size:,} rows", flush=True)
    print(f"self-play rows {sp.n:,}; window {n_w:,} (rows {lo:,}..{sp.n:,}); train {train_hi - train_lo:,}, holdout {hold_hi - hold_lo:,} newest"
          + (" [OVERFIT TEST: holdout == train]" if a.overfit_rows else ""), flush=True)
    genc = GpuEncoder(dev)
    sd, blocks, filters = infer_arch(a.init)
    net = net_for(sd, blocks, filters)
    net.load_state_dict({k: v.contiguous() for k, v in sd.items()})
    net = net.to(dev).to(memory_format=torch.channels_last).train()
    print(f"warm start from {a.init} ({blocks}x{filters}, {sum(p.numel() for p in net.parameters()):,} params)", flush=True)
    # : KataGo's optimizer, copied.
    # [P S2] SGD with momentum 0.9, L2 3e-5, and a PER-SAMPLE learning rate of 6e-5 (2e-5 for the first 5M samples
    # of a FRESH net; we always warm-start, so 6e-5). The code's lr is per-BATCH, so the conversion is
    #     lr_batch = lr_sample x batch = 6e-5 x 128 = 7.68e-3
    # KataGo holds that lr flat and drops it by hand later in a run, so the cosine decay is gone; the 100-step
    # warmup stays because SGD at this lr needs it.
    KATAGO_LR_PER_SAMPLE = 6e-5
    lr_batch = a.lr if a.lr_explicit else KATAGO_LR_PER_SAMPLE * a.batch
    opt = torch.optim.SGD(net.parameters(), lr=lr_batch, momentum=0.9, weight_decay=3e-5, nesterov=False)
    print(f"optimizer: SGD momentum 0.9, L2 3e-5, lr {lr_batch:.3e} "
          f"(= {KATAGO_LR_PER_SAMPLE:.0e} per sample x batch {a.batch}) [KataGo paper S2]", flush=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 100))   # flat after warmup, as KataGo does
    scaler = torch.amp.GradScaler("cuda")
    log = open(os.path.join(a.out, "train_log.jsonl"), "a")
    best = float("inf"); t0 = time.time(); run = None

    def losses(o, pi_t, pp_t, z_t, q_t, ply_t, mat_t):
        kl = policy_kl(o["policy"], pi_t, pp_t)
        wt = wdl_target(z_t, q_t, a.value_mix)
        wdl = soft_ce(o["wdl"], wt)
        vt = a.value_mix * z_t + (1 - a.value_mix) * q_t
        sc = F.smooth_l1_loss(o["score"].float(), vt, beta=0.25)
        ml = F.smooth_l1_loss(o["moves_left"].float() / 100.0, ply_t, beta=0.25)
        mt = F.smooth_l1_loss(o["mat"].float(), mat_t, beta=0.25)
        return kl, wdl, sc, ml, mt

    def to_dev(arrs):
        return [torch.from_numpy(x).to(dev) for x in arrs]

    def evaluate():
        net.eval(); tot = {"kl": 0.0, "wdl": 0.0, "vacc": 0.0, "vn": 0, "n": 0}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for start in range(0, hold_idx.size, a.batch):
                idx = hold_idx[start:start + a.batch]
                rec, pi, pp, z, q, ply, mat, hst, nhs, kcap, hang, auxok = sp.batch(idx)
                x = genc(torch.from_numpy(rec).to(dev),
                         *( (torch.from_numpy(hst).to(dev), torch.from_numpy(nhs).to(dev)) if a.history else () ))
                o = net.forward_train(x.to(memory_format=torch.channels_last))
                pi_t, pp_t, z_t, q_t, ply_t, mat_t = to_dev([pi, pp, z, q, ply, mat])
                kl, wdl, _, _, _ = losses(o, pi_t, pp_t, z_t, q_t, ply_t, mat_t)
                tot["kl"] += kl.item() * len(idx); tot["wdl"] += wdl.item() * len(idx); tot["n"] += len(idx)
                pw = torch.softmax(o["wdl"].float(), 1); ev = pw[:, 0] - pw[:, 2]
                dec = z_t != 0
                tot["vacc"] += ((ev > 0) == (z_t > 0))[dec].float().sum().item(); tot["vn"] += int(dec.sum().item())
        net.train(); n = max(1, tot["n"])
        return {"sp_policy_kl": tot["kl"] / n, "sp_wdl_loss": tot["wdl"] / n, "sp_val_acc": tot["vacc"] / max(1, tot["vn"]),
                "sp_holdout": tot["kl"] / n + 0.5 * tot["wdl"] / n}

    for step in range(1, a.steps + 1):
        idx = train_idx[rng.integers(0, train_idx.size, a.batch)]
        rec, pi, pp, z, q, ply, mat, hst, nhs, kcap, hang, auxok = sp.batch(idx)
        x = genc(torch.from_numpy(rec).to(dev),
                 *( (torch.from_numpy(hst).to(dev), torch.from_numpy(nhs).to(dev)) if a.history else () ))
        pi_t, pp_t, z_t, q_t, ply_t, mat_t = to_dev([pi, pp, z, q, ply, mat])
        with torch.autocast("cuda", dtype=torch.float16):
            o = net.forward_train(x.to(memory_format=torch.channels_last))
            kl, wdl, sc, ml, mt = losses(o, pi_t, pp_t, z_t, q_t, ply_t, mat_t)
            loss = kl + 0.5 * wdl + 0.5 * sc + 0.1 * ml + 0.1 * mt
            # auxiliary heads. Masked per ROW by auxok, because gen dirs written before 
            # have no sp_aux.bin and their zeros are ABSENT labels, not negative ones - training on those
            # would teach the net that nothing is ever hanging. aux weight 0.15 (the range 0.1-0.25).
            aux_loss = None
            if "kcap" in o:
                m = torch.from_numpy(auxok).to(dev)
                den = m.sum().clamp(min=1.0)
                kc = (F.binary_cross_entropy_with_logits(o["kcap"].float(),
                        torch.from_numpy(kcap).to(dev), reduction="none").mean(1) * m).sum() / den
                hg = (F.binary_cross_entropy_with_logits(o["hang"].float(),
                        torch.from_numpy(hang).to(dev), reduction="none").mean(1) * m).sum() / den
                aux_loss = kc + hg
                loss = loss + a.aux_weight * aux_loss
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); scaler.step(opt); scaler.update(); sched.step()
        run = float(loss) if run is None else 0.98 * run + 0.02 * float(loss)
        if step % 50 == 0:
            auxs = f" aux {float(aux_loss):.3f}" if aux_loss is not None else ""
            print(f"step {step:5d} loss {run:.4f} (kl {float(kl):.3f} wdl {float(wdl):.3f} mat {float(mt):.3f}{auxs}) {(step * a.batch) / (time.time() - t0):.0f} rows/s lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            m = evaluate(); m.update({"step": step, "train_loss": run})
            log.write(json.dumps(m) + "\n"); log.flush()
            print(f"EVAL step {step}: sp_policy_kl {m['sp_policy_kl']:.4f} sp_wdl {m['sp_wdl_loss']:.4f} sp_val_acc {m['sp_val_acc']:.3f} sp_holdout {m['sp_holdout']:.4f} train {run:.4f}", flush=True)
            ck = {"net": net.state_dict(), "step": step, "blocks": blocks, "filters": filters}
            torch.save(ck, os.path.join(a.out, "latest.ckpt"))
            if m["sp_holdout"] < best:
                best = m["sp_holdout"]
                torch.save(ck, os.path.join(a.out, "best.ckpt"))
                subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_best.py"),
                                os.path.join(a.out, "best.ckpt"), os.path.join(a.out, "best.pt")], check=False)
                print(f"BEST step {step}: sp_holdout {best:.4f} -> best.pt", flush=True)
    print(f"RESULT: done (best sp_holdout {best:.4f}, init {a.init}, {blocks}x{filters}, window {n_w:,} rows)", flush=True)


if __name__ == "__main__":
    main()
