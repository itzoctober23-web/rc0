// Rc0 - board geometry for 4-player chess: lookup tables built once at startup.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "types.h"

namespace quad {
namespace geo {

// Files a..n (0..13), ranks 1..14 (0..13). SQ[f][r] is the square index, or SQ_NONE
// in a removed corner. Squares are numbered rank by rank from rank 1, file a first.
extern int SQ[FILE_NB][RANK_NB];
extern int FILE_OF[SQUARE_NB];
extern int RANK_OF[SQUARE_NB];

// Ray directions: 0..3 orthogonal (N, E, S, W), 4..7 diagonal (NE, SE, SW, NW).
// RAY[s][d] lists the squares from s outward in direction d, SQ_NONE-terminated.
constexpr int DIR_NB = 8;
extern int16_t RAY[SQUARE_NB][DIR_NB][14];
extern int16_t KNIGHT_TO[SQUARE_NB][9];
extern int16_t KING_TO[SQUARE_NB][9];

// Pawns. Red advances up the ranks, Yellow down, Blue right along the files, Green left.
extern int PAWN_PUSH[COLOR_NB][SQUARE_NB];               // one step forward, or SQ_NONE
extern int16_t PAWN_ATTACK[COLOR_NB][SQUARE_NB][3];      // capture targets
extern int16_t PAWN_ATTACK_FROM[COLOR_NB][SQUARE_NB][3]; // where a c-pawn attacks s from
extern int RELATIVE_RANK[COLOR_NB][SQUARE_NB];           // 0 = own back edge, 13 = far edge
extern bool PAWN_START[COLOR_NB][SQUARE_NB];             // may double-push (relative rank 1)
extern int PROMO_DIST[COLOR_NB][SQUARE_NB];              // pushes left to the promotion line (relative rank 10)

// Squares on one line (rank, file or diagonal), excluding a == b.
extern uint8_t ALIGNED[SQUARE_NB][SQUARE_NB];

// The four armies are rotations of one another. SEAT_MAP[c][s] rotates a square from
// Red's frame into seat c's frame; SEAT_INV[c][s] rotates it back to Red's frame.
//   Blue: (f, r) -> (r, 13 - f)   Yellow: (f, r) -> (13 - f, 13 - r)   Green: (f, r) -> (13 - r, f)
extern int SEAT_MAP[COLOR_NB][SQUARE_NB];
extern int SEAT_INV[COLOR_NB][SQUARE_NB];
// ROLE_UNDER[m][c]: the seat color c's army occupies after rotating mover m to Red's seat.
extern Color ROLE_UNDER[COLOR_NB][COLOR_NB];

// Castling per seat and side (0 = kingside, 1 = queenside). For Red: king h1; kingside
// rook k1 (king to j1, rook to i1); queenside rook d1 (king to f1, rook to g1).
struct CastleInfo {
    int kFrom, kTo, rFrom, rTo;
    int empty[4];  // must be empty, SQ_NONE-terminated
    int safe[4];   // king passes or lands on these; must not be attacked (kFrom checked separately)
};
extern CastleInfo CASTLE[COLOR_NB][2];
constexpr int castle_bit(Color c, int side) { return 1 << (int(c) * 2 + side); }

void init();
std::string sq_name(int sq);                        // "a4", "n11"
int parse_sq(const char* s, int* len = nullptr);    // SQ_NONE if not a playable square

} // namespace geo
} // namespace quad
