// SPDX-License-Identifier: GPL-3.0-or-later
// UCI4 loop per docs/PROTOCOL.md. The search runs on its own thread so that
// `go infinite` + `stop` (the harness ponders off-turn this way) and `stop`
// during a timed search both produce a `bestmove` when the GUI asks for it.
// Set ZERO_LOG=<file> to append every line in and out (protocol debugging).
#include "uci4.h"
#include <cstring>
#include <algorithm>
#include <random>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include "movegen.h"
#include "nncache.h"

namespace zero {
using namespace quad;

static FILE* g_log = nullptr;
static void logline(const char* dir, const std::string& s) {
    if (!g_log) return;
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    std::fprintf(g_log, "%lld %s %s\n", (long long)now, dir, s.c_str());
    std::fflush(g_log);
}
static void out(const std::string& s) {
    std::fputs(s.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
    logline(">", s);
}

struct PosSpec {
    std::string base;          // "startpos" or the fen4 string
    std::vector<Move> moves;
    // input history: the HIST_POS positions before the current one, newest first - the same ring self-play keeps,
    // so the ladder and any match play feed the net the SAME inputs the training data was recorded with.
    unsigned char ring[HIST_POS * REC_BYTES] = {0};
    int nring = 0;
};
static void set_position(Position& pos, std::istringstream& is, PosSpec& spec) {
    std::string tok;
    is >> tok;
    spec.moves.clear();
    if (tok == "startpos") {
        pos.init_startpos();
        spec.base = "startpos";
        is >> tok;  // "moves" or nothing
    } else if (tok == "fen4" || tok == "fen") {
        std::string fen;
        is >> fen;
        if (!pos.set_fen4(fen)) { out("info string bad fen4"); pos.init_startpos(); fen = "startpos"; }
        spec.base = fen;
        is >> tok;
    }
    spec.nring = 0;
    std::memset(spec.ring, 0, sizeof(spec.ring));
    if (tok == "moves")
        while (is >> tok) {
            Move m = pos.parse_move(tok);
            if (m == MOVE_NONE) { out("info string illegal move " + tok); break; }
            // push the position we are about to leave, newest first (identical to selfplay.cpp's ring)
            if (HIST_POS > 1) std::memmove(spec.ring + REC_BYTES, spec.ring, size_t(HIST_POS - 1) * REC_BYTES);
            fill_record(pos, spec.ring);
            if (spec.nring < HIST_POS) spec.nring++;
            pos.do_move(m);
            spec.moves.push_back(m);
        }
}

struct GoLimits {
    int64_t time[COLOR_NB] = {-1, -1, -1, -1}, inc[COLOR_NB] = {0, 0, 0, 0}, delay[COLOR_NB] = {0, 0, 0, 0};
    int64_t movetime = -1;
    int64_t nodes = 0;
    int depth = 0;
    bool infinite = false;
};

static int64_t budget_ms(const GoLimits& g, Color us, int overhead) {
    if (g.infinite) return -1;
    if (g.movetime > 0) return std::max<int64_t>(1, g.movetime - overhead);
    if (g.time[us] < 0) return -1;
    // a 30th of the clock plus most of the increment; the chess.com delay is
    // free time, spend nearly all of it.
    int64_t t = g.time[us] / 30 + g.inc[us] * 3 / 4 + g.delay[us] * 9 / 10 - overhead;
    return std::max<int64_t>(5, t);
}

namespace {
struct Searcher {
    std::thread th;
    std::atomic<bool> stop{false};
    std::atomic<bool> running{false};
    void join() {
        if (th.joinable()) th.join();
        running = false;
    }
    void halt() {
        stop = true;
        join();
    }
};
}  // namespace

static std::mt19937_64& bot_rng() { static std::mt19937_64 r(std::chrono::steady_clock::now().time_since_epoch().count()); return r; }

void uci4_loop(EngineOptions opts) {
    if (const char* lp = std::getenv("ZERO_LOG")) g_log = std::fopen(lp, "a");
    Position pos;
    pos.init_startpos();
    std::unique_ptr<Evaluator> ev;
    if (!opts.netPath.empty()) ev = make_evaluator(opts.netPath, 256);
    // Same rule as selfplay_main: a requested net that will not load is FATAL, never a silent
    // downgrade. A ladder game played by the stand-in would be logged as a real measurement.
    if (!ev && !opts.netPath.empty()) {
        out("info string FATAL --net was requested but could not be loaded; refusing to play with the stand-in");
        std::fflush(stdout);
        std::exit(2);   // uci4_loop returns void; the harness must SEE the failure, not a stand-in game
    }
    if (!ev) {
        ev = std::make_unique<UniformEvaluator>();
        out("info string WARNING no net loaded: uniform stand-in evaluator");
    }
    Searcher s;
    // Tree reuse: the tree persists across `position`/`go` as long as the new position
    // extends the tree root's move list (the harness ponders off-turn with `go infinite`,
    // so every real search inherits the ponder tree's visits for free).
    std::unique_ptr<Mcts> tree;
    std::unique_ptr<CachedEvaluator> cache = std::make_unique<CachedEvaluator>(*ev, 18);
    PosSpec treeSpec, posSpec;
    posSpec.base = "startpos";
    auto reset_tree = [&]() { tree.reset(); treeSpec = PosSpec(); };
    std::string line, tok;
    while (std::getline(std::cin, line)) {
        logline("<", line);
        std::istringstream is(line);
        tok.clear();
        is >> tok;
        if (tok == "uci4" || tok == "uci") {
            out("id name Rc0 0.1");
            out("id author the Rc0 authors");
            out("option name Net type string default <empty>");
            out("option name Playouts type spin default 0 min 0 max 100000000");
            out("option name MoveOverhead type spin default 50 min 0 max 5000");
            out("option name CPuct type string default 1.75");
            out("option name FPU type string default 0.33");
            out("option name PolicyTemp type string default 1");
            out("option name GatherK type spin default 16 min 1 max 256");
            out("option name Temperature type string default 0");
            out(tok == "uci4" ? "uci4ok" : "uciok");
        } else if (tok == "isready") {
            out("readyok");
        } else if (tok == "stop") {
            s.halt();
        } else if (tok == "quit") {
            s.halt();
            break;
        } else if (tok == "ucinewgame") {
            s.halt();
            cache->clear();
            pos.init_startpos();
            posSpec = PosSpec(); posSpec.base = "startpos";
            reset_tree();
        } else if (tok == "setoption") {
            s.halt();
            std::string name, value, w;
            is >> w;  // name
            while (is >> w && w != "value") name += (name.empty() ? "" : " ") + w;
            std::getline(is, value);
            if (!value.empty() && value[0] == ' ') value.erase(0, 1);
            if (name == "Net") {
                auto e = make_evaluator(value, 256);
                if (e) { ev = std::move(e); opts.netPath = value; cache = std::make_unique<CachedEvaluator>(*ev, 18); reset_tree(); out("info string loaded " + value); }
                else out("info string failed to load " + value);
            } else if (name == "Playouts") opts.playoutsCap = std::atoi(value.c_str());
            else if (name == "MoveOverhead") opts.moveOverheadMs = std::atoi(value.c_str());
            else if (name == "CPuct") { opts.sp.cpuct = float(std::atof(value.c_str())); reset_tree(); }
            else if (name == "FPU") { opts.sp.fpuReduction = float(std::atof(value.c_str())); reset_tree(); }
            else if (name == "PolicyTemp") { opts.sp.policyTemp = float(std::atof(value.c_str())); reset_tree(); }
            else if (name == "GatherK") { opts.sp.gatherK = std::atoi(value.c_str()); reset_tree(); }
            else if (name == "Temperature") opts.temperature = float(std::atof(value.c_str()));
            // Threads / Hash / anything else: accepted silently
        } else if (tok == "position") {
            s.halt();
            set_position(pos, is, posSpec);
        } else if (tok == "recinfo") {
            // input historyseam test: checksum of fill_record() for the CURRENT position
            unsigned char r[REC_BYTES]; fill_record(pos, r);
            unsigned long long h = 1469598103934665603ULL;
            for (int i = 0; i < REC_BYTES; i++) { h ^= r[i]; h *= 1099511628211ULL; }
            out("recinfo " + std::to_string(h));
        } else if (tok == "histinfo") {
            // input historyseam test: nhist and a checksum per ring slot, newest first. Lets the self-play ring and
            // the UCI ring be compared exactly instead of by "they share the code" (flagged unverified 05:26).
            std::string r = "histinfo " + std::to_string(posSpec.nring);
            for (int k = 0; k < posSpec.nring; k++) {
                unsigned long long h = 1469598103934665603ULL;
                for (int i = 0; i < REC_BYTES; i++) { h ^= posSpec.ring[k * REC_BYTES + i]; h *= 1099511628211ULL; }
                r += " " + std::to_string(h);
            }
            out(r);
        } else if (tok == "legalmoves") {
            // arbiter rules-oracle commands, used by GUIs and match runners as a rules referee
            s.halt();
            Move list[MAX_MOVES];
            int n = generate_legal(pos, list);
            std::string r = "legalmoves";
            for (int i = 0; i < n; i++) r += " " + Position::move_str(list[i]);
            out(r);
        } else if (tok == "tacticsprobe") {
            // Rules-only tactics oracle for the order-1 gate . One call per position
            // instead of O(moves^2) UCI round trips, and the rules live in ONE place - the engine -
            // so the suite cannot disagree with what self-play and the ladder see.
            //   win    = our moves that end the game with OUR team winning
            //   unsafe = our moves after which the NEXT mover has such a move available
            s.halt();
            Move list[MAX_MOVES];
            int n = generate_legal(pos, list);
            int myTeam = team_of(pos.side_to_move());
            std::string win = "win", unsafe = "unsafe";
            for (int i = 0; i < n; i++) {
                pos.do_move(list[i]);
                bool over = false;
                GameResult gr = game_result(pos, over);
                if (over) {
                    if (gr != DRAW_RESULT && ((gr == TEAM_RY_WINS) ? 0 : 1) == myTeam)
                        win += " " + Position::move_str(list[i]);
                } else {
                    Move rep[MAX_MOVES];
                    int m = generate_legal(pos, rep);
                    int oppTeam = team_of(pos.side_to_move());
                    for (int j = 0; j < m; j++) {
                        pos.do_move(rep[j]);
                        bool o2 = false;
                        GameResult g2 = game_result(pos, o2);
                        pos.undo_move(rep[j]);
                        if (o2 && g2 != DRAW_RESULT && ((g2 == TEAM_RY_WINS) ? 0 : 1) == oppTeam) {
                            unsafe += " " + Position::move_str(list[i]);
                            break;
                        }
                    }
                }
                pos.undo_move(list[i]);
            }
            out("tacticsprobe " + win + " " + unsafe);
        } else if (tok == "status") {
            s.halt();
            bool over = false;
            GameResult r = game_result(pos, over);
            static const char* names[] = {"ongoing", "ry_wins", "bg_wins", "draw"};
            out(std::string("status ") + names[r] + " check " + (pos.stm_in_check() ? "1" : "0"));
        } else if (tok == "d") {
            out(pos.pretty() + "\nfen4 " + pos.fen4());
        } else if (tok == "go" && !opts.policy.empty()) {   // ladder bots : no search
            s.halt();
            std::string rest; std::getline(is, rest);
            const bool infinite = rest.find("infinite") != std::string::npos;
            Move list[MAX_MOVES];
            int n = generate_legal(pos, list);
            bool over = false; game_result(pos, over);
            Move best = MOVE_NONE;
            if (!over && n > 0) {
                if (opts.policy == "greedy") {
                    static const int val[] = {1, 3, 5, 5, 9, 1000};   // by type_of: P N B R Q K
                    int bestScore = -1; int nb = 0;
                    for (int i = 0; i < n; i++) {
                        Piece v = pos.piece_on(to_sq(list[i]));
                        int sc = (v != NO_PIECE && flag_of(list[i]) != MF_EN_PASSANT) ? val[std::min(int(type_of(v)), 5)] : 0;
                        if (flag_of(list[i]) == MF_EN_PASSANT) sc = 1;
                        if (sc > bestScore) { bestScore = sc; nb = 0; }
                        if (sc == bestScore && (bot_rng()() % ++nb) == 0) best = list[i];
                    }
                } else best = list[bot_rng()() % n];
            }
            if (infinite) { s.stop = false; s.running = true; s.th = std::thread([&s, best]() { while (!s.stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(2)); out(best == MOVE_NONE ? "bestmove 0000" : "bestmove " + Position::move_str(best)); }); }
            else out(best == MOVE_NONE ? "bestmove 0000" : "bestmove " + Position::move_str(best));
        } else if (tok == "go") {
            s.halt();
            GoLimits g;
            while (is >> tok) {
                auto num = [&]() { int64_t v = 0; is >> v; return v; };
                if (tok == "movetime") g.movetime = num();
                else if (tok == "nodes") g.nodes = num();
                else if (tok == "depth") g.depth = int(num());
                else if (tok == "infinite") g.infinite = true;
                else if (tok.size() == 5 && tok.substr(1) == "time") g.time[Color(std::string("rbyg").find(tok[0]))] = num();
                else if (tok.size() == 4 && tok.substr(1) == "inc") g.inc[Color(std::string("rbyg").find(tok[0]))] = num();
                else if (tok.size() == 6 && tok.substr(1) == "delay") g.delay[Color(std::string("rbyg").find(tok[0]))] = num();
            }
            int64_t ms = budget_ms(g, pos.side_to_move(), opts.moveOverheadMs);
            if (ms < 0 && !g.infinite && g.nodes == 0 && opts.playoutsCap == 0)
                ms = g.depth > 0 ? 200LL * g.depth : 1000;  // bare `go` / `go depth d`: a bounded search
            const bool infinite = g.infinite;
            int64_t maxPlayouts = g.nodes > 0 ? g.nodes : (opts.playoutsCap > 0 ? opts.playoutsCap : (1LL << 30));
            // reuse the tree if the new position extends the tree's root by 0+ moves
            bool reuse = tree && treeSpec.base == posSpec.base && posSpec.moves.size() >= treeSpec.moves.size() &&
                         std::equal(treeSpec.moves.begin(), treeSpec.moves.end(), posSpec.moves.begin());
            if (reuse) {
                for (size_t i = treeSpec.moves.size(); i < posSpec.moves.size(); i++) tree->apply_move(posSpec.moves[i]);
            } else {
                tree = std::make_unique<Mcts>(*cache, opts.sp);
                tree->set_root(pos);
            }
            tree->set_root_history(posSpec.ring, posSpec.nring);   // input history: same history self-play recorded
            treeSpec = posSpec;
            Mcts* treep = tree.get();
            uint64_t inherited = treep->playouts();
            float temperature = opts.temperature;
            s.stop = false;
            s.running = true;
            s.th = std::thread([&s, treep, inherited, temperature, ms, maxPlayouts, infinite]() {
                Mcts& mcts = *treep;
                auto t0 = std::chrono::steady_clock::now();
                auto elapsed = [&]() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count(); };
                mcts.run_until([&]() { return s.stop.load() || (ms >= 0 && elapsed() >= ms); },
                               int(std::min<int64_t>(inherited + maxPlayouts, 1LL << 30)));
                if (infinite) {  // per UCI: bestmove only after stop
                    while (!s.stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
                RootStats rs = mcts.root_stats();
                Move best = mcts.select_move(temperature);
                int64_t el = std::max<int64_t>(1, elapsed());
                if (best == MOVE_NONE) {
                    char b2[128];
                    std::snprintf(b2, sizeof b2, "info string no move: playouts %llu rootmoves %zu terminal %d", (unsigned long long)rs.nodes, rs.moves.size(), int(mcts.root_terminal()));
                    out(b2);
                    out("bestmove 0000");
                    return;
                }
                float q = 0.0f;
                for (size_t i = 0; i < rs.moves.size(); i++) if (rs.moves[i] == best) q = rs.q[i];
                char buf[256];
                std::snprintf(buf, sizeof buf, "info depth %d score cp %d nodes %llu nps %llu time %lld string inherited %llu pv %s", rs.maxDepth,
                              value_to_cp(q), (unsigned long long)rs.nodes, (unsigned long long)((rs.nodes - inherited) * 1000 / el),
                              (long long)el, (unsigned long long)inherited, Position::move_str(best).c_str());
                out(buf);
                out("bestmove " + Position::move_str(best));
            });
        }
    }
    s.halt();
    if (g_log) std::fclose(g_log);
}

}  // namespace zero
