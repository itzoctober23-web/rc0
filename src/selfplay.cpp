// SPDX-License-Identifier: GPL-3.0-or-later
// Multi-game batched self-play  and its throughput measurement.
//
//   zero selfplay --net F.pt --games N --threads T --playouts P [--out DIR] [--seconds S]
//
// T game threads, one shared GPU batcher, one NN cache. Each finished game appends to
// DIR (append-only, one lock): sp_rec.bin (136-byte position records), sp_pol.bin
// (96 x {int16 policy index, float16 prob} visit distribution), sp_res.bin (int8 result,
// RY view), sp_q.bin (float32 root Q, mover-team view), sp_ply.bin (int16 plies to end).
// From-random line (design rule): playout-cap randomization (fullPct of moves at `playouts`, recorded;
// the rest at fastPlayouts, not recorded), Dirichlet root noise + temperature-1 sampling for the opening plies,
// adjudication of clearly decided games by the search value (verifyPct of them played out to measure the error),
// sp_mat.bin (int8 final material balance, RY view) as an auxiliary target, and per-run policy entropy /
// value accuracy / average game length in the progress line. Games hitting the ply cap are written as draws.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <random>
#include <algorithm>
#include <cmath>
#include <string>
#include <thread>
#include <vector>
#include "batcher.h"
#include "mcts.h"
#include "movegen.h"
#include "nncache.h"
#include "uci4.h"
#include "selfplay.h"

namespace zero {
using namespace quad;


namespace {
struct Rec { unsigned char rec[136]; int16_t polIdx[96]; uint16_t polP[96]; float q; int ply; int pliesLeft;
             unsigned char hist[HIST_POS * REC_BYTES]; unsigned char nhist;
             unsigned char aux[AUX_BYTES]; };   // input history: the HIST_POS positions before this one, newest first; rules-only aux labels



uint16_t f16(float f) {  // float -> IEEE half (round to nearest), enough for probabilities
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000; int exp = int((x >> 23) & 0xFF) - 127 + 15; uint32_t man = x & 0x7FFFFF;
    if (exp <= 0) return uint16_t(sign);
    if (exp >= 31) return uint16_t(sign | 0x7C00);
    return uint16_t(sign | (uint32_t(exp) << 10) | (man >> 13));
}

struct Writer {
    std::mutex mu;
    FILE *fgames = nullptr, *fr = nullptr, *fp = nullptr, *fres = nullptr, *fq = nullptr, *fply = nullptr, *fmat = nullptr, *fhist = nullptr, *faux = nullptr;
    bool open(const std::string& dir) {
        auto op = [&](const char* n) { return std::fopen((dir + "/" + n).c_str(), "ab"); };
        fr = op("sp_rec.bin"); fp = op("sp_pol.bin"); fres = op("sp_res.bin"); fq = op("sp_q.bin"); fply = op("sp_ply.bin"); fmat = op("sp_mat.bin");
        fhist = op("sp_hist.bin");   // input history side file
        faux = op("sp_aux.bin");     // side file, same backward-compatible pattern
        fgames = std::fopen((dir + "/games.txt").c_str(), "a");   // one line per game, in row order (see validate.cpp)
        return fgames && fr && fp && fres && fq && fply && fmat && fhist && faux;
    }
    void write_game(const std::vector<Rec>& recs, int8_t result, int8_t mat, const std::string& gameLine) {
        std::lock_guard<std::mutex> lk(mu);
        std::fputs(gameLine.c_str(), fgames); std::fputc('\n', fgames); std::fflush(fgames);
        int n = int(recs.size());
        for (int i = 0; i < n; i++) {
            std::fwrite(recs[i].rec, 1, 136, fr);
            for (int j = 0; j < 96; j++) { std::fwrite(&recs[i].polIdx[j], 2, 1, fp); std::fwrite(&recs[i].polP[j], 2, 1, fp); }
            std::fwrite(&result, 1, 1, fres);
            std::fwrite(&recs[i].q, 4, 1, fq);
            int16_t left = int16_t(recs[i].pliesLeft); std::fwrite(&left, 2, 1, fply);
            std::fwrite(&mat, 1, 1, fmat);
            std::fwrite(recs[i].hist, 1, HIST_POS * REC_BYTES, fhist);
            std::fwrite(&recs[i].nhist, 1, 1, fhist);
            std::fwrite(recs[i].aux, 1, AUX_BYTES, faux);
        }
        std::fflush(fr); std::fflush(fp); std::fflush(fres); std::fflush(fq); std::fflush(fply); std::fflush(fmat); std::fflush(fhist); std::fflush(faux);
    }
    void close() { for (FILE* f : {fgames, fr, fp, fres, fq, fply, fmat, fhist, faux}) if (f) std::fclose(f); }
};
}  // namespace

int selfplay_main(const SelfplayOpts& o) {
    std::unique_ptr<Evaluator> net = o.net.empty() ? nullptr : make_evaluator(o.net, o.maxBatch);
    // a requested --net that cannot be loaded is FATAL. 2026-10-01 08:47: a 210-plane (ZERO_HISTORY) binary was
    // pointed at the 41-plane net_gen30.pt, libtorch raised "expected input[1,210,14,14] to have 41 channels",
    // and this fell back to the uniform stand-in and generated 51,466 rows of near-uniform policy targets
    // (root Q exactly 0.0, policy entropy 3.50 against 0.45) at 182,000 evals/s. The stand-in is only ever
    // legitimate when NO net was asked for.
    if (!net && !o.net.empty()) {
        std::printf("FATAL selfplay: --net %s was requested but could not be loaded; refusing to generate with the "
                    "uniform stand-in (it would write uniform policy targets and a zero root Q).\n", o.net.c_str());
        std::fflush(stdout);
        return 2;
    }
    if (!net) { net = std::make_unique<UniformEvaluator>(); std::printf("info string WARNING no net: uniform stand-in\n"); }
    CachedEvaluator cache(*net, o.cacheLog2);
    SharedBatcher batcher(cache, o.batchTarget, o.maxBatch, o.flushMs);
    Writer writer;
    bool writing = !o.out.empty();
    if (writing && !writer.open(o.out)) { std::printf("cannot open %s\n", o.out.c_str()); return 1; }
    std::atomic<int> gamesStarted{0}, gamesDone{0}, capped{0}, ry{0}, bg{0}, draws{0};
    std::atomic<int> adjTotal{0}, adjVerified{0}, adjWrong{0}, adjUnresolved{0};
    std::atomic<uint64_t> positions{0}, recorded{0}, plies{0}, valN{0}, valCorrect{0};
    std::atomic<double> entSum{0.0}; std::atomic<uint64_t> entN{0};
    std::atomic<bool> stopFlag{false};
    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    std::vector<std::thread> threads;
    for (int t = 0; t < o.threads; t++) {
        threads.emplace_back([&, t] {
            BatchedEvaluator ev(batcher);
            std::mt19937_64 rng(o.seed * 7919 + t);
            while (!stopFlag.load()) {
                int g = gamesStarted.fetch_add(1);
                if (g >= o.games) break;
                SearchParams sp;
                sp.rootNoise = o.noise && o.gumbel <= 0;
                sp.seed = o.seed * 100003 + g;
                sp.gatherK = 16;
                sp.forcedPlayouts = o.forcedPlayouts;   // , self-play only
                Position pos;
                if (int(rng() % 100) < o.c960pct) pos.init_startpos960(rng()); else pos.init_startpos();
                const std::string startFen = pos.fen4();   // game record (games.txt): start, every move, how it ended
                std::string moveList;
                int nrand = 0;
                int rp = int(rng() % (o.randomPlies + 1));
                for (int i = 0; i < rp; i++) {
                    Move list[MAX_MOVES]; int n = generate_legal(pos, list);
                    bool over = false; game_result(pos, over);
                    if (over || n == 0) break;
                    Move rm = list[rng() % n];
                    moveList += (moveList.empty() ? "" : " ") + Position::move_str(rm);
                    pos.do_move(rm);
                    nrand++;
                }
                Mcts mcts(ev, sp);
                mcts.set_root(pos);
                std::vector<Rec> recs;
                unsigned char ring[HIST_POS * REC_BYTES] = {0};   // input history: positions before the current one, newest first
                int nring = 0;
                GameResult gr = ONGOING;
                int ply = 0;
                int adjStreak = 0, adjSign = 0, adjResult = 0; bool adjudicated = false, verifying = false;
                const char* endReason = "stopped";   // natural | capped | adjudicated | stopped (never written)
                while (!stopFlag.load()) {
                    bool over = false;
                    gr = game_result(pos, over);
                    if (over) { endReason = "natural"; break; }
                    if (ply >= o.maxPlies) { gr = ONGOING; endReason = "capped"; break; }
                    bool full = int(rng() % 100) < o.fullPct;
                    int po = full ? o.playouts : o.fastPlayouts;
                    mcts.set_root_noise(o.noise && ply < o.noisePlies);
                    mcts.set_root_history(ring, nring);
                    std::vector<float> target;
                    Move m;
                    if (o.gumbel > 0) m = mcts.gumbel_search(po, o.gumbel, target);
                    else { mcts.run(po); m = mcts.select_move(ply < o.tempPlies ? 1.0f : 0.0f); }
                    RootStats rs = mcts.root_stats();
                    if (m == MOVE_NONE) { endReason = "error"; break; }
                    // "self-play training targets for proven positions use the proven value".
                    // Where the RULES settle the position, the value target is the proof, not the search's
                    // estimate of it - and the adjudicator below should trust a proof completely.
                    const int rootPv = mcts.root_proven();
                    if (rootPv != 0) rs.rootQ = mcts.root_proven_value();
                    if (o.gumbel > 0 && ply < o.tempPlies && target.size() == rs.moves.size() && !target.empty()) {
                        // opening temperature: sample the move from the improved policy instead of taking its argmax
                        float u = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng), acc = 0.0f;
                        for (size_t i = 0; i < target.size(); i++) { acc += target[i]; if (u <= acc) { m = rs.moves[i]; break; } }
                    }
                    if (writing && full) {
                        Rec r; fill_record(pos, r.rec); fill_aux(pos, r.aux);
                        std::memcpy(r.hist, ring, sizeof(r.hist)); r.nhist = (unsigned char)nring;
                        // policy target: Gumbel completed-Q policy if available, else visit fractions
                        std::vector<std::pair<float, size_t>> order;
                        uint64_t tot = 0; for (auto v : rs.visits) tot += v;
                        double H = 0.0;
                        for (size_t i = 0; i < rs.moves.size(); i++) {
                            float pr = target.size() == rs.moves.size() ? target[i] : (tot ? float(rs.visits[i]) / float(tot) : 0.0f);
                            order.push_back({pr, i});
                            if (pr > 1e-9f) H -= double(pr) * std::log(double(pr));
                        }
                        std::sort(order.begin(), order.end(), [](auto& a, auto& b) { return a.first > b.first; });
                        for (int j = 0; j < 96; j++) {
                            if (j < int(order.size())) {
                                r.polIdx[j] = int16_t(policy_index(pos, rs.moves[order[j].second]));
                                r.polP[j] = f16(order[j].first);
                            } else { r.polIdx[j] = -1; r.polP[j] = 0; }
                        }
                        r.q = rs.rootQ; r.ply = ply;
                        recs.push_back(r);
                        { double e = entSum.load(); while (!entSum.compare_exchange_weak(e, e + H)) {} }
                        entN++;
                    }
                    // adjudication by the search value (RY view), sign-consistent for adjPlies plies after adjMinPly
                    float qry = team_of(pos.side_to_move()) == 0 ? rs.rootQ : -rs.rootQ;
                    int sgn = qry >= o.adjThr ? 1 : (qry <= -o.adjThr ? -1 : 0);
                    if (sgn != 0 && sgn == adjSign) adjStreak++; else { adjSign = sgn; adjStreak = sgn ? 1 : 0; }
                    if (!adjudicated && ply >= o.adjMinPly && adjStreak >= o.adjPlies) {
                        adjudicated = true; adjResult = sgn; adjTotal++;
                        if (int(rng() % 100) < o.verifyPct) verifying = true;
                        else { gr = sgn > 0 ? TEAM_RY_WINS : TEAM_BG_WINS; endReason = "adjudicated"; break; }
                    }
                    // push the position we just left onto the ring (newest first) before advancing
                    if (HIST_POS > 1) std::memmove(ring + REC_BYTES, ring, size_t(HIST_POS - 1) * REC_BYTES);
                    fill_record(pos, ring);
                    if (nring < HIST_POS) nring++;
                    moveList += (moveList.empty() ? "" : " ") + Position::move_str(m);
                    pos.do_move(m);
                    mcts.apply_move(m);
                    ply++;
                    positions++;
                }
                plies += ply;
                int8_t result = gr == TEAM_RY_WINS ? 1 : gr == TEAM_BG_WINS ? -1 : 0;
                if (gr == ONGOING) { capped++; draws++; }           // ply cap: written as a draw
                else if (gr == TEAM_RY_WINS) ry++; else if (gr == TEAM_BG_WINS) bg++; else draws++;
                if (verifying) { if (gr == ONGOING) adjUnresolved++; else { adjVerified++; if (result != adjResult) adjWrong++; } }
                // A game cut off by --seconds or a stop has no honest result, so it is not written at all.
                const bool complete = std::strcmp(endReason, "stopped") != 0 && std::strcmp(endReason, "error") != 0;
                if (writing && complete) {
                    for (Rec& r : recs) {
                        r.pliesLeft = ply - r.ply;
                        float qry = team_of(Color(r.rec[0])) == 0 ? r.q : -r.q;
                        if (result != 0) { valN++; if ((qry > 0) == (result > 0)) valCorrect++; }   // decisive rows only (draws made it read 22% at gen 1)
                    }
                    std::string rows;
                    for (const Rec& r : recs) rows += (rows.empty() ? "" : ",") + std::to_string(r.ply);
                    std::string line = startFen + "\t" + std::to_string(nrand) + "\t" + std::to_string(o.maxPlies) + "\t" +
                                       endReason + "\t" + std::to_string(int(result)) + "\t" + moveList + "\t" + rows;
                    writer.write_game(recs, result, material_ry(pos), line);
                    recorded += recs.size();
                }
                gamesDone++;
                if (o.seconds > 0 && elapsed() >= o.seconds) stopFlag = true;
            }
        });
    }
    // progress line every 10 s
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        double el = elapsed();
        std::printf("[selfplay] %.0fs games %d/%d (capped %d) positions %llu (%.0f/s) recorded %llu (%.0f/s) evals %llu (%.0f/s, avg batch %.1f) cache hit %.1f%% | "
                    "adj %d verified %d wrong %d unresolved %d | entropy %.3f valacc %.1f%% avgplies %.1f\n",
                    el, gamesDone.load(), o.games, capped.load(), (unsigned long long)positions.load(), positions.load() / el,
                    (unsigned long long)recorded.load(), recorded.load() / el,
                    (unsigned long long)batcher.items(), batcher.items() / el, batcher.avgBatch(),
                    100.0 * cache.hits() / std::max<uint64_t>(1, cache.hits() + cache.misses()),
                    adjTotal.load(), adjVerified.load(), adjWrong.load(), adjUnresolved.load(),
                    entSum.load() / std::max<uint64_t>(1, entN.load()), 100.0 * valCorrect.load() / std::max<uint64_t>(1, valN.load()),
                    gamesDone.load() ? double(plies.load()) / gamesDone.load() : 0.0);
        std::fflush(stdout);
        if (gamesDone.load() >= o.games || stopFlag.load()) break;
    }
    stopFlag = true;
    for (auto& th : threads) th.join();
    batcher.stop();
    if (writing) writer.close();
    double el = elapsed();
    std::printf("RESULT selfplay: %d games in %.0f s (%.2f games/s), RY %d BG %d draw %d capped %d, %llu positions (%.0f/s), recorded %llu, avg %.1f plies, "
                "adj %d verified %d wrong %d unresolved %d, entropy %.3f valacc %.1f%%, evals %llu (%.0f/s), avg batch %.1f, cache hit %.1f%%, threads %d playouts %d/%d full %d%% (%s)\n",
                gamesDone.load(), el, gamesDone.load() / el, ry.load(), bg.load(), draws.load(), capped.load(),
                (unsigned long long)positions.load(), positions.load() / el, (unsigned long long)recorded.load(), gamesDone.load() ? double(plies.load()) / gamesDone.load() : 0.0,
                adjTotal.load(), adjVerified.load(), adjWrong.load(), adjUnresolved.load(),
                entSum.load() / std::max<uint64_t>(1, entN.load()), 100.0 * valCorrect.load() / std::max<uint64_t>(1, valN.load()),
                (unsigned long long)batcher.items(), batcher.items() / el, batcher.avgBatch(),
                100.0 * cache.hits() / std::max<uint64_t>(1, cache.hits() + cache.misses()), o.threads, o.fastPlayouts, o.playouts, o.fullPct, net->name());
    return 0;
}

}  // namespace zero
