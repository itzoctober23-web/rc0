// SPDX-License-Identifier: GPL-3.0-or-later
// Encoding tests:
//  1. role/rotation tables: every seat's king lands on h1 in the mover frame
//  2. policy_index is a bijection over the legal moves of 20k random positions
//  3. rotation invariance: the start position encodes identically for all four
//     seats (each seat to move at its own first ply of a symmetric game)
//  4. every input plane index in range; piece plane count == piece count
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <vector>
#include "encoding.h"
#include "geometry.h"
#include "movegen.h"

using namespace quad;
using namespace zero;

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

int main() {
    geo::init();
    Position::init_zobrist();
    Position pos;
    pos.init_startpos();

    // 1. rotation tables
    for (int c = 0; c < COLOR_NB; c++) {
        int k = pos.king_sq(Color(c));
        int ck = canon_sq(Color(c), k);
        CHECK(geo::sq_name(ck) == "h1", "seat %c king canonicalises to %s", COLOR_CHAR[c], geo::sq_name(ck).c_str());
        CHECK(geo::ROLE_UNDER[c][c] == RED, "role of self != 0");
        CHECK(geo::ROLE_UNDER[c][int(partner_of(Color(c)))] == YELLOW, "role of partner != 2");
        CHECK(geo::ROLE_UNDER[c][int(next_color(Color(c)))] == BLUE, "role of next != 1");
    }

    // 3. rotation invariance of the start position across seats: build a
    // symmetric game (each seat plays the rotated copy of Red's move) and
    // compare the tensor at each seat's turn.
    {
        std::vector<float> ref(INPUT_FLOATS), cur(INPUT_FLOATS);
        Position p;
        p.init_startpos();
        encode(p, ref.data());
        // Red plays h2h3 -> rotate for each seat via SEAT_MAP (Red frame -> seat frame)
        Move red = p.parse_move("h2h3");
        CHECK(red != MOVE_NONE, "h2h3 illegal?");
        int f = from_sq(red), t = to_sq(red);
        for (int c = 0; c < COLOR_NB; c++) {
            if (c == 0) {
                encode(p, cur.data());
                float diff = 0;
                for (int i = 0; i < INPUT_FLOATS; i++) diff += std::abs(cur[i] - ref[i]);
                CHECK(diff == 0, "encoding not deterministic");
            }
            Move m = make_move(geo::SEAT_MAP[c][f], geo::SEAT_MAP[c][t]);
            Move parsed = p.parse_move(Position::move_str(m));
            CHECK(parsed != MOVE_NONE, "rotated move %s illegal for seat %c", Position::move_str(m).c_str(), COLOR_CHAR[c]);
            p.do_move(parsed);
        }
        // Rotation invariance: the position is now fully symmetric (each seat made the
        // same rotated move). Re-parse it with each seat as the side to move (FEN4 turn
        // letter swapped): every seat's mover-relative tensor must be identical except
        // the first-team plane.
        std::string fen = p.fen4();
        std::vector<std::vector<float>> encs;
        for (int c = 0; c < COLOR_NB; c++) {
            std::string f2 = fen;
            f2[0] = COLOR_CHAR[c];
            Position q;
            CHECK(q.set_fen4(f2), "fen4 with turn %c rejected", COLOR_CHAR[c]);
            std::vector<float> e(INPUT_FLOATS);
            encode(q, e.data());
            int mask = 0;
            for (int i = 0; i < CELLS; i++) mask += e[PL_MASK * CELLS + i] > 0.5f;
            CHECK(mask == 160, "mask count %d", mask);
            int h3 = geo::SQ[7][2];
            CHECK(e[0 * CELLS + cell_of(h3)] == 1.0f, "seat %c: own advanced pawn not at h3 in mover frame", COLOR_CHAR[c]);
            encs.push_back(e);
        }
        for (int c = 1; c < COLOR_NB; c++) {
            int d = 0;
            for (int i = 0; i < PL_FIRSTTEAM * CELLS; i++) d += encs[c][i] != encs[0][i];
            CHECK(d == 0, "seat %c planes differ from Red's by %d cells (rotation invariance broken)", COLOR_CHAR[c], d);
            CHECK(encs[c][PL_FIRSTTEAM * CELLS] == (c % 2 == 0 ? 1.0f : 0.0f), "first-team plane seat %c", COLOR_CHAR[c]);
        }
        // policy indices must be rotation-invariant too: the legal-move index SET is identical per seat
        std::vector<std::set<int>> sets;
        for (int c = 0; c < COLOR_NB; c++) {
            std::string f2 = fen; f2[0] = COLOR_CHAR[c];
            Position q; q.set_fen4(f2);
            LegalSet ls = legal_with_index(q);
            sets.emplace_back(ls.idx.begin(), ls.idx.end());
            CHECK(sets.back().size() == ls.idx.size(), "duplicate policy index for seat %c", COLOR_CHAR[c]);
        }
        for (int c = 1; c < COLOR_NB; c++) CHECK(sets[c] == sets[0], "policy index set differs for seat %c", COLOR_CHAR[c]);
    }

    // 2. bijection over random games
    std::mt19937_64 rng(7);
    long positions = 0, moves = 0, promos = 0, under = 0, castles = 0, eps = 0;
    for (int g = 0; g < 400 && fails < 20; g++) {
        Position p;
        if (g % 2) p.init_startpos960(rng()); else p.init_startpos();
        for (int ply = 0; ply < 300; ply++) {
            bool over = false;
            if (game_result(p, over), over) break;
            LegalSet ls = legal_with_index(p);
            std::set<int> seen;
            for (size_t i = 0; i < ls.moves.size(); i++) {
                Move m = ls.moves[i];
                int ix = ls.idx[i];
                CHECK(ix >= 0 && ix < POLICY_SIZE, "index out of range for %s", Position::move_str(m).c_str());
                CHECK(!seen.count(ix), "duplicate policy index %d for %s (%s)", ix, Position::move_str(m).c_str(), p.fen4().c_str());
                seen.insert(ix);
                if (is_promotion(m)) { promos++; if (promo_of(m) != QUEEN) under++; }
                if (flag_of(m) == MF_CASTLE_K || flag_of(m) == MF_CASTLE_Q) castles++;
                if (flag_of(m) == MF_EN_PASSANT) eps++;
            }
            moves += ls.moves.size();
            positions++;
            std::vector<float> e(INPUT_FLOATS);
            encode(p, e.data());
            int cnt = 0, pieces = 0;
            for (int i = 0; i < 24 * CELLS; i++) cnt += e[i] > 0.5f;
            for (int sq = 0; sq < SQUARE_NB; sq++) pieces += p.piece_on(sq) != NO_PIECE;
            CHECK(cnt == pieces, "piece plane count %d != %d pieces", cnt, pieces);
            // prefer captures/promotions sometimes so promotions occur
            Move m = ls.moves[rng() % ls.moves.size()];
            for (size_t i = 0; i < ls.moves.size(); i++)
                if (is_promotion(ls.moves[i]) && rng() % 2) { m = ls.moves[i]; break; }
            p.do_move(m);
        }
    }
    std::printf("bijection: %ld positions, %ld moves (%.1f avg), promos %ld (under %ld), castles %ld, ep %ld\n",
                positions, moves, double(moves) / positions, promos, under, castles, eps);
    CHECK(promos > 0 && under > 0 && castles > 0, "coverage: need promotions, underpromotions and castling in the sample");

    // 5. input history: adding history must not disturb planes 0..PL_HIST-1, must light exactly the
    //    expected history planes, and must set PL_HISTVALID only on a FULL history.
    //    Only meaningful in a -DZERO_HISTORY build; without it the history planes do not exist and reading
    //    them would run off the end of the 41-plane buffer (which is what this guard caught at 06:22).
#ifdef ZERO_HISTORY
    {
        auto fill_rec = [](const Position& p, unsigned char* r) {   // same layout as selfplay.cpp fill_record
            std::memset(r, 0xFF, REC_BYTES);
            r[0] = (unsigned char)p.side_to_move();
            int np = 0;
            for (int sq = 0; sq < SQUARE_NB && np < 64; sq++) {
                Piece pc = p.piece_on(sq);
                if (pc == NO_PIECE) continue;
                r[8 + 2 * np] = (unsigned char)pc; r[9 + 2 * np] = (unsigned char)sq; np++;
            }
            r[3] = (unsigned char)np;
        };
        Position p; p.init_startpos();
        std::vector<unsigned char> recs(HIST_POS * REC_BYTES, 0);
        std::mt19937_64 rng(11);
        std::vector<Position> past;
        for (int i = 0; i < HIST_POS; i++) {
            past.push_back(p);
            LegalSet ls = legal_with_index(p);
            if (ls.moves.empty()) break;
            p.do_move(ls.moves[rng() % ls.moves.size()]);
        }
        for (int i = 0; i < (int)past.size(); i++)   // newest first
            fill_rec(past[past.size() - 1 - i], &recs[i * REC_BYTES]);

        std::vector<float> a(INPUT_FLOATS, 0.0f), b(INPUT_FLOATS, 0.0f), c0(INPUT_FLOATS, 0.0f);
        encode(p, a.data());                                   // no history
        encode(p, recs.data(), (int)past.size(), b.data());    // full history
        encode(p, recs.data(), 1, c0.data());                  // one position only

        for (int i = 0; i < PL_HIST * CELLS; i++)
            CHECK(a[i] == b[i], "history changed base plane float %d (%g vs %g)", i, a[i], b[i]);
        int tail = 0;
        for (int i = PL_HIST * CELLS; i < INPUT_FLOATS; i++) if (a[i] != 0.0f) tail++;
        CHECK(tail == 0, "no-history encode left %d non-zero floats above plane %d", tail, PL_HIST);
        CHECK(b[PL_HISTVALID * CELLS] == 1.0f, "PL_HISTVALID not set on a full history");
        CHECK(c0[PL_HISTVALID * CELLS] == 0.0f, "PL_HISTVALID set on a PARTIAL history");
        int hset = 0;
        for (int h = 0; h < HIST_POS; h++)
            for (int i = 0; i < 24 * CELLS; i++) if (b[(PL_HIST + h * 24) * CELLS + i] != 0.0f) hset++;
        CHECK(hset == 64 * (int)past.size(), "history piece count %d != %d", hset, 64 * (int)past.size());
        int c1 = 0;
        for (int i = 0; i < 24 * CELLS; i++) if (c0[PL_HIST * CELLS + i] != 0.0f) c1++;
        CHECK(c1 == 64, "one-position history set %d planes-cells, expected 64", c1);
        std::printf("history: %d base floats unchanged, %d history cells on a %d-position history, HISTVALID full=%g partial=%g\n",
                    PL_HIST * CELLS, hset, (int)past.size(), b[PL_HISTVALID * CELLS], c0[PL_HISTVALID * CELLS]);
    }
#else
    std::printf("history: skipped (41-plane build; rebuild with -DZERO_HISTORY to exercise the history planes)\n");
#endif
    std::printf("%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}
