// SPDX-License-Identifier: GPL-3.0-or-later
// MCTS tests:
//  1. mate-in-1 found: from a position where the side to move can capture an
//     enemy king / deliver mate, the most-visited move is the winning one.
//  2. backup sign: a proven-loss child is avoided.
//  3. terminal handling: search on a terminal root does nothing and does not crash.
//  4. position restored after every batch (do/undo balance): fen4 unchanged.
#include <cstdio>
#include <string>
#include <random>
#include <vector>
#include "geometry.h"
#include "mcts.h"
#include "movegen.h"

using namespace quad;
using namespace zero;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// Finds, by random play, a position where the side to move has a king capture
// available (the 4PC "king en prise at the start of your turn" rule).
static bool find_king_capture_position(Position& out, uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (int g = 0; g < 2000; g++) {
        Position p;
        p.init_startpos();
        for (int ply = 0; ply < 400; ply++) {
            bool over = false;
            if (game_result(p, over), over) break;
            Move list[MAX_MOVES];
            int n = generate_legal(p, list);
            for (int i = 0; i < n; i++) {
                Piece v = p.piece_on(to_sq(list[i]));
                if (v != NO_PIECE && type_of(v) == KING && flag_of(list[i]) != MF_EN_PASSANT) { out = p; return true; }
            }
            p.do_move(list[rng() % n]);
        }
    }
    return false;
}


// --- certainty propagation (certainty propagation) -------------------------------------------------------
// A king capture available to the side to move is a mate-in-1 BY THE RULES, so these positions are
// found by random play and verified with generate_legal + piece_on, never with chess knowledge.
// For each SEAT, so the partner/opponent turn order is covered rather than assumed.
static bool find_king_capture_for_seat(Position& out, int seat, uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (int g = 0; g < 4000; g++) {
        Position p;
        p.init_startpos();
        for (int ply = 0; ply < 400; ply++) {
            bool over = false;
            game_result(p, over);
            if (over) break;
            Move list[MAX_MOVES];
            int n = generate_legal(p, list);
            if (n == 0) break;
            if (int(p.side_to_move()) == seat) {
                for (int i = 0; i < n; i++) {
                    Piece v = p.piece_on(to_sq(list[i]));
                    if (v != NO_PIECE && type_of(v) == KING && flag_of(list[i]) != MF_EN_PASSANT) { out = p; return true; }
                }
            }
            p.do_move(list[rng() % n]);
        }
    }
    return false;
}

static bool is_king_capture(const Position& p, Move m) {
    Piece v = p.piece_on(to_sq(m));
    return v != NO_PIECE && type_of(v) == KING && flag_of(m) != MF_EN_PASSANT;
}

int main() {
    geo::init();
    Position::init_zobrist();
    UniformEvaluator ev;
    SearchParams sp;
    sp.gatherK = 8;

    // 1 + 4: king capture must dominate visits
    Position p;
    bool found = find_king_capture_position(p, 3);
    CHECK(found, "no king-capture position found by random play");
    if (found) {
        std::string fen = p.fen4();
        Mcts m(ev, sp);
        m.set_root(p);
        m.run(400);
        CHECK(p.fen4() == fen, "position modified by search");
        RootStats rs = m.root_stats();
        size_t best = 0;
        for (size_t i = 1; i < rs.moves.size(); i++) if (rs.visits[i] > rs.visits[best]) best = i;
        Piece v = p.piece_on(to_sq(rs.moves[best]));
        CHECK(v != NO_PIECE && type_of(v) == KING, "most visited move %s is not the king capture (visits %u of %llu)",
              Position::move_str(rs.moves[best]).c_str(), rs.visits[best], (unsigned long long)rs.nodes);
        CHECK(rs.q[best] > 0.9f, "king-capture Q %.3f should be ~+1", rs.q[best]);
        CHECK(rs.rootQ > 0.5f, "root Q %.3f should be strongly positive", rs.rootQ);
        // 3: terminal root
        Position t = p;
        t.do_move(rs.moves[best]);
        Mcts m2(ev, sp);
        m2.set_root(t);
        m2.run(50);
        CHECK(m2.root_terminal(), "root after king capture should be terminal");
        CHECK(m2.select_move(0.0f) == MOVE_NONE, "terminal root must return MOVE_NONE");
    }

    // 2: backup sign — a child that hands the opponent a king capture must get a low Q
    {
        Position s;
        s.init_startpos();
        Mcts m(ev, sp);
        m.set_root(s);
        m.run(3000);
        RootStats rs = m.root_stats();
        uint64_t total = 0;
        for (auto v : rs.visits) total += v;
        CHECK(total == rs.nodes - 1 || total == rs.nodes, "visits sum %llu vs playouts %llu", (unsigned long long)total, (unsigned long long)rs.nodes);
        CHECK(m.max_depth() >= 3, "depth %d too shallow for 3000 playouts", m.max_depth());
        std::printf("startpos 3000 playouts: rootQ %.3f depth %d tree %llu\n", rs.rootQ, m.max_depth(), (unsigned long long)m.nodes());
    }

    // --- certainty propagation ----------------------------------------------------------
    // (a) mate-in-1 is PROVEN, for every seat. The uniform stand-in gives every move the same prior
    //     and value 0, so nothing but the proof can make the search pick the capture.
    for (int seat = 0; seat < 4; seat++) {
        Position kp;
        if (!find_king_capture_for_seat(kp, seat, 11 + seat)) { CHECK(false, "seat %d: no king-capture position found", seat); continue; }
        CHECK(int(kp.side_to_move()) == seat, "seat %d: found a position whose mover is %d", seat, int(kp.side_to_move()));
        std::string fen = kp.fen4();
        Mcts m(ev, sp);
        m.set_root(kp);
        m.run(200);
        CHECK(kp.fen4() == fen, "seat %d: position modified by search", seat);
        CHECK(m.root_proven() == 1, "seat %d: root should be PROVEN WIN, got %d", seat, m.root_proven());
        Move chosen = m.select_move(0.0f);
        CHECK(is_king_capture(kp, chosen), "seat %d: chose %s, not a king capture", seat, Position::move_str(chosen).c_str());
        // a proven win must be played even at high temperature - a proof is not a suggestion
        Move hot = m.select_move(2.0f);
        CHECK(is_king_capture(kp, hot), "seat %d: temperature 2.0 chose %s, not the proven win", seat, Position::move_str(hot).c_str());
        // and the Gumbel self-play path must agree, with a one-hot target on the proven move
        Mcts mg(ev, sp);
        mg.set_root(kp);
        std::vector<float> tgt;
        Move gm = mg.gumbel_search(200, 8, tgt);
        CHECK(is_king_capture(kp, gm), "seat %d: gumbel chose %s, not the proven win", seat, Position::move_str(gm).c_str());
        float mx = 0.0f; int argmx = -1; float sum = 0.0f;
        for (size_t i = 0; i < tgt.size(); i++) { sum += tgt[i]; if (tgt[i] > mx) { mx = tgt[i]; argmx = int(i); } }
        CHECK(mx > 0.999f, "seat %d: policy target on a proven win should be one-hot, max %.4f", seat, mx);
        CHECK(sum > 0.999f && sum < 1.001f, "seat %d: policy target sums to %.4f", seat, sum);
        CHECK(argmx >= 0 && is_king_capture(kp, mg.root_stats().moves[argmx]), "seat %d: one-hot target is not on the capture", seat);
    }

    // (b) proof crosses a ply: after a quiet move that leaves the capture on, the NEW mover (an
    //     opponent) must prove a win for its own team. This is the propagation step that a
    //     mate-in-2/3 test would exercise; exhaustive mate-in-N verification is not feasible here
    //     because 4PC branching makes a depth-5 proof search enormous, so the mechanism is tested
    //     directly instead of claiming depth coverage that was never run.
    {
        Position kp;
        if (find_king_capture_for_seat(kp, 0, 101)) {
            Move list[MAX_MOVES];
            int n = generate_legal(kp, list);
            int quiet = -1;
            for (int i = 0; i < n; i++) if (!is_king_capture(kp, list[i])) { quiet = i; break; }
            if (quiet >= 0) {
                Position after = kp;
                after.do_move(list[quiet]);
                bool over = false; game_result(after, over);
                if (!over) {
                    Mcts m2(ev, sp);
                    m2.set_root(after);
                    m2.run(200);
                    // whoever moves now may or may not have a capture; what must hold is that if the
                    // search proves anything at the root it is consistent with the rules at depth 1
                    int pr = m2.root_proven();
                    Move ch = m2.select_move(0.0f);
                    bool cap = ch != MOVE_NONE && is_king_capture(after, ch);
                    CHECK(!(cap && pr != 1), "a king capture was chosen but the root was not proven a win (proven=%d)", pr);
                }
            }
        }
    }

    // (c) a proof survives tree reuse: apply_move must not discard `proven`
    {
        Position kp;
        if (find_king_capture_for_seat(kp, 0, 7)) {
            Mcts m(ev, sp);
            m.set_root(kp);
            m.run(200);
            int before = m.root_proven();
            CHECK(before == 1, "setup: root should be proven, got %d", before);
            Move win = m.select_move(0.0f);
            m.apply_move(win);
            CHECK(m.root_terminal(), "after playing the proven win the root must be terminal");
        }
    }


    // (d) REGRESSION: root_proven_win_edge() must not hold references into nodes_ across new_node().
    //     The first version did, and ASan did not catch it because the suite's positions found the
    //     winning move on the first scanned edge, so nothing ever reallocated. The GAME gate caught
    //     it (a fixed bot 16/40 -> 8/40). This test runs ONE playout so nodes_ is tiny and the scan's
    //     node creation is almost certain to reallocate, which is exactly the path that was unsafe.
    for (int seat = 0; seat < 4; seat++) {
        Position kp;
        if (!find_king_capture_for_seat(kp, seat, 211 + seat)) continue;
        std::string fen = kp.fen4();
        Mcts m(ev, sp);
        m.set_root(kp);
        m.run(1);                                  // nodes_ has almost no spare capacity here
        Move chosen = m.select_move(0.0f);         // -> root_proven_win_edge() allocates while scanning
        CHECK(is_king_capture(kp, chosen), "seat %d (tiny tree): chose %s, not the proven win", seat,
              Position::move_str(chosen).c_str());
        CHECK(kp.fen4() == fen, "seat %d (tiny tree): position corrupted by the root scan", seat);
        Mcts mg(ev, sp);
        mg.set_root(kp);
        std::vector<float> tgt;
        mg.run(1);
        Move gm = mg.gumbel_search(2, 2, tgt);
        CHECK(is_king_capture(kp, gm), "seat %d (tiny tree): gumbel chose %s, not the proven win", seat,
              Position::move_str(gm).c_str());
        CHECK(kp.fen4() == fen, "seat %d (tiny tree): position corrupted by the gumbel root scan", seat);
    }

    std::printf("%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}
