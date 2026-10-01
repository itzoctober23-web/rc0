// Rc0 rules self-tests: perft from the start position, FEN4 round trips, and make/unmake
// consistency over random games (key, FEN and move list restored exactly).
// SPDX-License-Identifier: GPL-3.0-or-later
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

#include "geometry.h"
#include "movegen.h"
#include "position.h"

using namespace quad;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); failures++; } } while (0)

int main(int argc, char** argv) {
    geo::init();
    Position::init_zobrist();

    // Perft from the standard start (cross-checked against an independent implementation).
    const uint64_t expected[] = {1, 20, 395, 7800, 152050, 3452310, 77430383};
    int maxDepth = argc > 1 ? std::atoi(argv[1]) : 5;  // pass 6 for the full table
    for (int d = 1; d <= maxDepth && d < 7; ++d) {
        Position p;
        p.init_startpos();
        uint64_t n = perft(p, d);
        std::printf("perft(%d) = %llu\n", d, (unsigned long long)n);
        CHECK(n == expected[d], "perft(%d) = %llu, expected %llu", d, (unsigned long long)n, (unsigned long long)expected[d]);
    }

    // Random games: FEN4 round trip, make/unmake restores everything, parse_move(move_str(m)) == m.
    std::mt19937_64 rng(2024);
    long plies = 0;
    for (int g = 0; g < 200; ++g) {
        Position p;
        if (g % 2) p.init_startpos960(rng()); else p.init_startpos();
        for (int ply = 0; ply < 300; ++ply) {
            bool over = false;
            game_result(p, over);
            if (over) break;
            Move list[MAX_MOVES];
            int n = generate_legal(p, list);
            if (n == 0) break;
            const std::string fen = p.fen4();
            const Key key = p.key();
            Position q;
            CHECK(q.set_fen4(fen) && q.fen4() == fen, "FEN4 round trip failed: %s", fen.c_str());
            for (int i = 0; i < n; ++i) {
                CHECK(p.parse_move(Position::move_str(list[i])) == list[i], "parse_move(%s) mismatch", Position::move_str(list[i]).c_str());
                p.do_move(list[i]);
                p.undo_move(list[i]);
                CHECK(p.key() == key && p.fen4() == fen, "do/undo of %s changed the position", Position::move_str(list[i]).c_str());
            }
            p.do_move(list[rng() % n]);
            ++plies;
            if (failures > 10) return 1;
        }
    }
    std::printf("random games: %ld plies checked\n", plies);
    std::printf(failures ? "rules: %d FAILURES\n" : "rules: all tests passed\n", failures);
    return failures ? 1 : 0;
}
