// SPDX-License-Identifier: GPL-3.0-or-later
// Rc0 — entry point.
//   zero                      UCI4 loop (uniform stand-in until a net is set)
//   zero --net file.pt        UCI4 with a TorchScript net
//   zero selfplay-test N P    play N games with the stand-in evaluator at P playouts
//   zero bench [net] [P]      playouts/s of the search (+ net if given)
//   zero selfplay --net F --games N --threads T --playouts P [--out DIR]  batched multi-game self-play
//   zero tables               rotation/geometry tables as JSON
//   zero validate DIR [-v] [--games-only]   replay every game in a self-play directory and check every training row
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "geometry.h"
#include "mcts.h"
#include "movegen.h"
#include "position.h"
#include "uci4.h"
#include "nncache.h"

using namespace quad;
using namespace zero;

#ifndef ZERO_HAVE_TORCH
namespace zero {
std::unique_ptr<Evaluator> make_torch_evaluator(const std::string&, int) { return nullptr; }
}
#endif
#ifndef ZERO_HAVE_TRT
namespace zero {
std::unique_ptr<Evaluator> make_trt_evaluator(const std::string&, int) { return nullptr; }
}
#endif
#ifndef ZERO_HAVE_ONNX
namespace zero {
std::unique_ptr<Evaluator> make_onnx_evaluator(const std::string&, int) { return nullptr; }
}
#endif

namespace zero {
// Backend selection lives here because only this target knows which backends were built.
//   a .onnx path        -> ONNX Runtime
//   otherwise           -> TensorRT (if a matching .plan exists), then LibTorch, then the .onnx sibling
// ZERO_BACKEND=trt|torch|onnx forces one backend and refuses to fall back, so a measurement can never
// silently test a different backend than the one it names.
std::unique_ptr<Evaluator> make_evaluator(const std::string& path, int maxBatch) {
    const char* w = std::getenv("ZERO_BACKEND");
    const std::string want = w ? w : "";
    const bool isOnnx = path.size() > 5 && path.substr(path.size() - 5) == ".onnx";
    if (want == "trt" || want == "torch" || want == "onnx") {
        auto e = want == "trt" ? make_trt_evaluator(path, maxBatch)
               : want == "torch" ? make_torch_evaluator(path, maxBatch) : make_onnx_evaluator(path, maxBatch);
        if (!e) std::printf("FATAL ZERO_BACKEND=%s but that backend could not load %s\n", want.c_str(), path.c_str());
        return e;
    }
    if (isOnnx) return make_onnx_evaluator(path, maxBatch);
    if (auto e = make_trt_evaluator(path, maxBatch)) return e;
    if (auto e = make_torch_evaluator(path, maxBatch)) return e;
    return make_onnx_evaluator(path, maxBatch);
}
}  // namespace zero
#include "selfplay.h"

static void init_all() {
    geo::init();
    Position::init_zobrist();
}

static int selfplay_test(int games, int playouts) {
    RandomEvaluator ev(12345);
    SearchParams sp;
    sp.rootNoise = true;
    int wins[3] = {0, 0, 0};  // RY, BG, draw
    long totalPlies = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < games; g++) {
        Position pos;
        pos.init_startpos();
        sp.seed = 1000 + g;
        Mcts mcts(ev, sp);
        mcts.set_root(pos);
        int ply = 0;
        GameResult gr = ONGOING;
        while (true) {
            bool over = false;
            gr = game_result(pos, over);
            if (over) break;
            if (ply >= 600) { gr = DRAW_RESULT; break; }
            mcts.run(playouts);
            Move m = mcts.select_move(ply < 30 ? 1.0f : 0.0f);
            if (m == MOVE_NONE) { std::printf("NO MOVE at ply %d\n%s\n", ply, pos.pretty().c_str()); return 1; }
            // cross-check the move is legal per the rules library's move parser
            if (pos.parse_move(Position::move_str(m)) != m) {
                std::printf("ILLEGAL move %s at ply %d\n%s\n", Position::move_str(m).c_str(), ply, pos.pretty().c_str());
                return 1;
            }
            pos.do_move(m);
            mcts.apply_move(m);
            ply++;
        }
        totalPlies += ply;
        wins[gr == TEAM_RY_WINS ? 0 : gr == TEAM_BG_WINS ? 1 : 2]++;
        std::printf("game %d: %s in %d plies (tree nodes %llu)\n", g + 1,
                    gr == TEAM_RY_WINS ? "RY wins" : gr == TEAM_BG_WINS ? "BG wins" : "draw", ply,
                    (unsigned long long)mcts.nodes());
        std::fflush(stdout);
    }
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("RESULT selfplay-test: %d games, RY %d BG %d draw %d, avg %.1f plies, %.1f s total, %.0f playouts/s (stand-in evaluator)\n",
                games, wins[0], wins[1], wins[2], double(totalPlies) / games, s, double(totalPlies) * playouts / s);
    return 0;
}

static int bench(const std::string& net, int playouts) {
    std::unique_ptr<Evaluator> raw;
    if (!net.empty()) raw = make_evaluator(net, 256);
    if (!raw) raw = std::make_unique<UniformEvaluator>();
    CachedEvaluator cached(*raw, 18);
    SearchParams sp;
    {   // warm-up (CUDA context, cuDNN autotune) so the first row is not a cold start
        Position pos; pos.init_startpos(); Mcts w(*raw, sp); w.set_root(pos); w.run(2000);
    }
    // A/B alternated (load-immune comparison): raw vs NN-cached evaluator, same K
    for (int K : {16, 64, 256}) {
        for (int rep = 0; rep < 2; rep++) {
            for (int useCache = 0; useCache < 2; useCache++) {
                sp.gatherK = K;
                cached.clear();
                Position pos;
                pos.init_startpos();
                Mcts mcts(useCache ? static_cast<Evaluator&>(cached) : *raw, sp);
                mcts.set_root(pos);
                auto t0 = std::chrono::steady_clock::now();
                mcts.run(playouts);
                double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                const auto& st = mcts.stats();
                std::printf("bench K=%-3d cache=%d rep=%d: %d playouts in %.2f s = %.0f/s, depth %d, batches %llu avg %.1f, collisions %llu, terminals %llu, cache hit %.1f%% (%s)\n",
                            K, useCache, rep, playouts, s, playouts / s, mcts.max_depth(), (unsigned long long)st.batches,
                            st.batches ? double(st.leaves) / st.batches : 0.0, (unsigned long long)st.collisions, (unsigned long long)st.terminals,
                            useCache ? 100.0 * cached.hits() / std::max<uint64_t>(1, cached.hits() + cached.misses()) : 0.0, raw->name());
            }
        }
    }
    return 0;
}

// tables: dump the rotation/role/geometry tables as JSON for the Python trainer.
static int tables() {
    std::printf("{\"SEAT_INV\": [");
    for (int c = 0; c < COLOR_NB; c++) { std::printf("%s[", c ? "," : ""); for (int s = 0; s < SQUARE_NB; s++) std::printf("%s%d", s ? "," : "", geo::SEAT_INV[c][s]); std::printf("]"); }
    std::printf("], \"ROLE_UNDER\": [");
    for (int m = 0; m < COLOR_NB; m++) { std::printf("%s[", m ? "," : ""); for (int c = 0; c < COLOR_NB; c++) std::printf("%s%d", c ? "," : "", int(geo::ROLE_UNDER[m][c])); std::printf("]"); }
    std::printf("], \"FILE_OF\": [");
    for (int s = 0; s < SQUARE_NB; s++) std::printf("%s%d", s ? "," : "", geo::FILE_OF[s]);
    std::printf("], \"RANK_OF\": [");
    for (int s = 0; s < SQUARE_NB; s++) std::printf("%s%d", s ? "," : "", geo::RANK_OF[s]);
    std::printf("], \"N_PLANES\": %d, \"POLICY_SIZE\": %d, \"RULE50_PLIES\": %d, \"ENCODING_VERSION\": %d}\n", N_PLANES, POLICY_SIZE, rules.fiftyMovePlies, ENCODING_VERSION);
    return 0;
}


// cross-check the aux labels against an INDEPENDENT derivation from the rules. fill_aux uses
// Position::attacked_by_team; this uses generate_legal + game_result, a different code path, and the two
// must agree - the mover can end the game exactly when some ENEMY king is flagged as attacked.
static int aux_check(int games) {
    std::mt19937_64 rng(12345);
    long pos_n = 0, mismatch_a = 0, bad_cell = 0, bits_b = 0, rows_with_b = 0;
    long mism_end_nocap = 0, mism_cap_noend = 0, perturbed = 0;
    for (int g = 0; g < games; g++) {
        Position p; p.init_startpos();
        for (int ply = 0; ply < 400; ply++) {
            bool over = false; game_result(p, over); if (over) break;
            Move list[MAX_MOVES];
            int n = generate_legal(p, list); if (n == 0) break;
            unsigned char aux[zero::AUX_BYTES];
            // fill_aux const_casts and calls generate_legal, so PROVE it leaves the position untouched -
            // if it did not, every recorded position would silently perturb the game that follows.
            const uint64_t key_before = p.key();
            const std::string fen_before = p.fen4();
            zero::fill_aux(p, aux);
            if (p.key() != key_before || p.fen4() != fen_before) perturbed++;
            pos_n++;
            bool canEnd = false;
            int myTeam = team_of(p.side_to_move());
            for (int i = 0; i < n && !canEnd; i++) {
                p.do_move(list[i]);
                bool o2 = false; GameResult g2 = game_result(p, o2);
                if (o2 && g2 != DRAW_RESULT && ((g2 == TEAM_RY_WINS) ? 0 : 1) == myTeam) canEnd = true;
                p.undo_move(list[i]);
            }
            bool enemyKingFlagged = false;
            for (int c = 0; c < COLOR_NB; c++)
                if ((aux[0] >> c) & 1u) enemyKingFlagged = true;   // bits 0..3 are legal captures BY the mover
            // Split the disagreement by DIRECTION so the cause is measured, not assumed.
            if (canEnd && !enemyKingFlagged) mism_end_nocap++;      // game ends with no king capture available
            if (!canEnd && enemyKingFlagged) mism_cap_noend++;      // a king can be captured but the game does not end
            if (canEnd != enemyKingFlagged) mismatch_a++;
            int nb = 0;
            for (int cell = 0; cell < zero::CELLS; cell++) {
                if (!((aux[1 + cell / 8] >> (cell % 8)) & 1u)) continue;
                nb++;
                bool found = false;
                for (int sq = 0; sq < SQUARE_NB && !found; sq++) {
                    Piece pc = p.piece_on(sq);
                    if (pc == NO_PIECE || team_of(color_of(pc)) != myTeam) continue;
                    if (zero::mover_cell_of(p, sq) == cell) found = true;
                }
                if (!found) bad_cell++;
            }
            bits_b += nb; if (nb) rows_with_b++;
            p.do_move(list[rng() % n]);
        }
    }
    double den = double(pos_n ? pos_n : 1);
    std::printf("aux-check: %ld positions | (a) mismatches vs generate_legal: %ld | (b) bits on a cell with no own piece: %ld"
                " | (b) mean bits/row %.2f, rows with any %.1f%%\n",
                pos_n, mismatch_a, bad_cell, double(bits_b) / den, 100.0 * double(rows_with_b) / den);
    std::printf("  split: game ends with NO king capture available: %ld | a king IS capturable but the game does not end: %ld\n",
                mism_end_nocap, mism_cap_noend);
    std::printf("  fill_aux left the position PERTURBED on %ld of %ld positions (must be 0)\n", perturbed, pos_n);
    return (bad_cell || perturbed) ? 1 : 0;
}

int main(int argc, char** argv) {
    init_all();
    EngineOptions opts;
    if (argc >= 2 && std::string(argv[1]) == "selfplay-test")
        return selfplay_test(argc >= 3 ? std::atoi(argv[2]) : 2, argc >= 4 ? std::atoi(argv[3]) : 64);
    if (argc >= 2 && std::string(argv[1]) == "aux-check") return aux_check(argc >= 3 ? std::atoi(argv[2]) : 20);
    if (argc >= 4 && std::string(argv[1]) == "encode-probe") {
        // input historyverification: dump (record, history, nhist, planes) for N random positions so the PYTHON
        // encoder can be diffed against this one. GpuEncoder is a reimplementation of encode(); the 01:05
        // repetition-plane bug came from exactly that, so it gets an equality test, not a code review.
        int n = std::atoi(argv[2]);
        FILE* f = std::fopen(argv[3], "wb");
        if (!f) { std::printf("cannot write %s\n", argv[3]); return 1; }
        std::mt19937_64 rng(20261001);
        std::vector<float> planes(INPUT_FLOATS);
        int written = 0;
        while (written < n) {
            Position p; p.init_startpos();
            unsigned char ring[HIST_POS * REC_BYTES] = {0};
            int nring = 0;
            int depth = int(rng() % 40);
            for (int i = 0; i < depth; i++) {
                bool over = false; game_result(p, over); if (over) break;
                LegalSet ls = legal_with_index(p);
                if (ls.moves.empty()) break;
                if (HIST_POS > 1) std::memmove(ring + REC_BYTES, ring, size_t(HIST_POS - 1) * REC_BYTES);
                fill_record(p, ring);
                if (nring < HIST_POS) nring++;
                p.do_move(ls.moves[rng() % ls.moves.size()]);
            }
            bool over = false; game_result(p, over); if (over) continue;
            unsigned char rec[REC_BYTES]; fill_record(p, rec);
            std::fill(planes.begin(), planes.end(), 0.0f);
            encode(p, ring, nring, planes.data());
            unsigned char nh = (unsigned char)nring;
            std::fwrite(rec, 1, REC_BYTES, f);
            std::fwrite(ring, 1, HIST_POS * REC_BYTES, f);
            std::fwrite(&nh, 1, 1, f);
            std::fwrite(planes.data(), 4, planes.size(), f);
            written++;
        }
        std::fclose(f);
        std::printf("encode-probe: %d samples -> %s (%d planes, %d cells, %zu bytes/sample)\n",
                    written, argv[3], N_PLANES, CELLS, size_t(REC_BYTES + HIST_POS * REC_BYTES + 1 + INPUT_FLOATS * 4));
        return 0;
    }
    if (argc >= 3 && std::string(argv[1]) == "validate") {   // replay-check a self-play directory (distributed uploads)
        bool verbose = false, gamesOnly = false;
        for (int i = 3; i < argc; i++) { verbose |= std::string(argv[i]) == "-v"; gamesOnly |= std::string(argv[i]) == "--games-only"; }
        return validate_main(argv[2], verbose, gamesOnly);
    }
    if (argc >= 2 && std::string(argv[1]) == "tables") return tables();
    if (argc >= 3 && std::string(argv[1]) == "encode-dump") {  // fen4 -> "plane cell value" lines + legal policy indices
        Position p; if (!p.set_fen4(argv[2])) { std::printf("bad fen\n"); return 1; }
        std::vector<float> e(INPUT_FLOATS); encode(p, e.data());
        for (int i = 0; i < INPUT_FLOATS; i++) if (e[i] != 0.0f) std::printf("%d %d %.4f\n", i / CELLS, i % CELLS, e[i]);
        LegalSet ls = legal_with_index(p);
        std::printf("policy");
        for (size_t i = 0; i < ls.moves.size(); i++) std::printf(" %s:%d", Position::move_str(ls.moves[i]).c_str(), ls.idx[i]);
        std::printf("\n");
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "selfplay") {
        SelfplayOpts o;
        for (int i = 2; i + 1 < argc; i += 2) {
            std::string k = argv[i], v = argv[i + 1];
            if (k == "--net") o.net = v; else if (k == "--out") o.out = v;
            else if (k == "--games") o.games = std::atoi(v.c_str()); else if (k == "--threads") o.threads = std::atoi(v.c_str());
            else if (k == "--playouts") o.playouts = std::atoi(v.c_str()); else if (k == "--max-plies") o.maxPlies = std::atoi(v.c_str());
            else if (k == "--temp-plies") o.tempPlies = std::atoi(v.c_str()); else if (k == "--random-plies") o.randomPlies = std::atoi(v.c_str());
            else if (k == "--c960") o.c960pct = std::atoi(v.c_str()); else if (k == "--batch-target") o.batchTarget = std::atoi(v.c_str());
            else if (k == "--max-batch") o.maxBatch = std::atoi(v.c_str()); else if (k == "--cache-log2") o.cacheLog2 = std::atoi(v.c_str());
            else if (k == "--flush-ms") o.flushMs = std::atof(v.c_str()); else if (k == "--seconds") o.seconds = std::atof(v.c_str());
            else if (k == "--seed") o.seed = std::strtoull(v.c_str(), nullptr, 10); else if (k == "--noise") o.noise = std::atoi(v.c_str()) != 0;
            else if (k == "--gumbel") o.gumbel = std::atoi(v.c_str());
            else if (k == "--forced-playouts") o.forcedPlayouts = std::atof(v.c_str());
            else if (k == "--fast-playouts") o.fastPlayouts = std::atoi(v.c_str()); else if (k == "--full-pct") o.fullPct = std::atoi(v.c_str());
            else if (k == "--noise-plies") o.noisePlies = std::atoi(v.c_str()); else if (k == "--adj-thr") o.adjThr = float(std::atof(v.c_str()));
            else if (k == "--adj-plies") o.adjPlies = std::atoi(v.c_str()); else if (k == "--adj-min-ply") o.adjMinPly = std::atoi(v.c_str());
            else if (k == "--verify-pct") o.verifyPct = std::atoi(v.c_str());
        }
        return selfplay_main(o);
    }
    if (argc >= 2 && std::string(argv[1]) == "bench")
        return bench(argc >= 3 ? argv[2] : "", argc >= 4 ? std::atoi(argv[3]) : 2000);
    for (int i = 1; i + 1 < argc; i++)
        if (std::string(argv[i]) == "--net") opts.netPath = argv[i + 1];
        else if (std::string(argv[i]) == "--policy") opts.policy = argv[i + 1];   // random | greedy (ladder bots, no net needed)
    uci4_loop(opts);
    return 0;
}
