// Rc0 - move generation and game-end detection for 4-player chess (Teams).
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "position.h"

namespace quad {

// All moves for the side to move that obey piece movement, ignoring whether the
// mover's own king is left attacked. Returns the count written to out (>= MAX_MOVES room).
int generate_pseudo(const Position& pos, Move* out);
// Pseudo-legal captures (including en passant) and promotions only.
int generate_captures(const Position& pos, Move* out);
// Fully legal moves. Capturing an enemy king is always legal (it ends the game).
int generate_legal(Position& pos, Move* out);

// Plays m and reports whether the mover's king is safe afterwards.
bool legal_after(Position& pos, Move m);

// Cheap filter: can m possibly expose the mover's own king? If not, a pseudo-legal
// move is legal without playing it.
inline bool needs_verify(const Position& pos, Move m, bool inCheck) {
    if (inCheck || flag_of(m) == MF_EN_PASSANT) return true;
    int ksq = pos.king_sq(pos.side_to_move());
    if (ksq == SQ_NONE) return true;
    int from = from_sq(m);
    return from == ksq || geo::ALIGNED[ksq][from];
}

// Leaf count of the legal move tree; root = print the per-move split.
uint64_t perft(Position& pos, int depth, bool root = false);

// Terminal check. gameOver is set when the game has ended; the return value says how.
GameResult game_result(Position& pos, bool& gameOver);

} // namespace quad
