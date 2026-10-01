#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""helper: build a TensorRT fp16 .plan for a net, so the C++ backend only has to DESERIALISE one.

ONNX parsing and engine building stay in Python because that path is already proven (tools/trt_bench.py) and
because building takes 13-16 s - something a self-play process should not redo on every start. The plan is
written next to the net as <net>.plan with a sidecar <net>.plan.meta recording the net's mtime and size, so a
stale plan is detected rather than silently used.

TensorRT 11 is strongly typed: there is no FP16 builder flag, so the ONNX itself must be fp16.

  python3 tools/trt_build_plan.py --net runs/rand/net_gen32_nohist.pt --batches 512 256 128 64 32 16 8 4 2 1
"""
import argparse, hashlib, json, os, sys, time
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[512])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    plan_path = a.out or (a.net + ".plan")
    st = os.stat(a.net)

    dev = torch.device("cuda")
    ts = torch.jit.load(a.net, map_location=dev).eval()
    planes = ts.state_dict()["stem.0.weight"].shape[1]
    bmax, bmin = max(a.batches), min(a.batches)
    onnx_path = plan_path + ".onnx"
    x = torch.zeros(bmax, planes, 14, 14, device=dev).half()
    torch.onnx.export(ts.half(), (x,), onnx_path, input_names=["planes"],
                      output_names=["policy", "wdl", "moves_left"], opset_version=17, dynamo=False,
                      dynamic_axes={"planes": {0: "b"}, "policy": {0: "b"}, "wdl": {0: "b"}, "moves_left": {0: "b"}})

    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network()
    parser = trt.OnnxParser(net, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("onnx-parse:", parser.get_error(i))
            return 1
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    prof = builder.create_optimization_profile()
    prof.set_shape("planes", (bmin, planes, 14, 14), (bmax, planes, 14, 14), (bmax, planes, 14, 14))
    cfg.add_optimization_profile(prof)
    t0 = time.perf_counter()
    plan = builder.build_serialized_network(net, cfg)
    if plan is None:
        print("engine build FAILED")
        return 1
    blob = bytes(plan)
    tmp = plan_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, plan_path)
    meta = {"net": os.path.abspath(a.net), "net_mtime": st.st_mtime, "net_size": st.st_size,
            "planes": int(planes), "batch_min": bmin, "batch_max": bmax,
            "trt": trt.__version__, "sha256": hashlib.sha256(blob).hexdigest()[:16],
            "built_s": round(time.perf_counter() - t0, 1)}
    with open(plan_path + ".meta", "w") as f:
        json.dump(meta, f, indent=1)
    os.unlink(onnx_path)
    print(f"wrote {plan_path}: {len(blob):,} bytes, {planes} planes, batch {bmin}..{bmax}, "
          f"built in {meta['built_s']}s, trt {meta['trt']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
