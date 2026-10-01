// SPDX-License-Identifier: GPL-3.0-or-later
// Rc0 — input-plane and policy encoding (version 1, mover-relative).
//
// The board is rotated so the side to move always sits in Red's seat
// (geo::SEAT_INV[stm]); pieces are ordered by ROLE (mover, next enemy, partner,
// previous enemy) via geo::ROLE_UNDER. Rotating a position by a full seat and
// advancing the mover therefore yields a byte-identical tensor: the net is
// seat-agnostic by construction. Reflection is NOT a symmetry (K/Q swap,
// castling) and must never be used for augmentation.
//
// v1 has NO history planes (single positions).
#pragma once
#include <cstdint>
#include <vector>
#include "position.h"

namespace zero {

constexpr int BOARD = 14;
constexpr int CELLS = BOARD * BOARD;  // 196, dead corners included (masked)

// ---- input planes ----
constexpr int PL_PIECES = 0;      // 24: role*6 + type
constexpr int PL_CASTLE = 24;     // 8:  role*2 + side (0 = kingside)
constexpr int PL_EP = 32;         // 4:  per role, the en-passant target square
constexpr int PL_RULE50 = 36;     // 1:  rule50 / 200, constant plane
constexpr int PL_REP = 37;        // 2:  repeated once / twice, constant planes
constexpr int PL_MASK = 39;       // 1:  playable-square mask
constexpr int PL_FIRSTTEAM = 40;  // 1:  1 if the mover's team moves first (RY)
// --- input history: the piece planes of the previous HIST_POS positions,
// all rendered in the CURRENT mover's frame so they line up spatially with the live board. Current + 7 = 8
// positions, as in Lc0. PL_HISTVALID is 1 only when a FULL history was supplied, so the net can tell
// "no history recorded" (old rows, game start) from "the board really was empty there".
constexpr int HIST_POS = 7;
constexpr int REC_BYTES = 136;                               // one self-play record, as fill_record() writes it
// The history is COLLECTED and WRITTEN (sp_hist.bin) either way, so the replay window fills with
// history-bearing rows from now on; -DZERO_HISTORY is what feeds it to the net. Without it this is the
// 41-plane engine and every existing net still loads.
#ifdef ZERO_HISTORY
constexpr int PL_HIST = 41;                                  // 7 x 24 = 168
constexpr int PL_HISTVALID = PL_HIST + HIST_POS * 24;        // 209
constexpr int N_PLANES = PL_HISTVALID + 1;                   // 210
#else
constexpr int PL_HIST = 41;
constexpr int PL_HISTVALID = 41;
constexpr int N_PLANES = 41;
#endif
constexpr int INPUT_FLOATS = N_PLANES * CELLS;

// ---- policy planes (mover frame, indexed by FROM square) ----
constexpr int SLIDE_MAX = 13;
constexpr int N_SLIDE = 8 * SLIDE_MAX;       // 104: dir*13 + (dist-1)
constexpr int N_KNIGHT = 8;                  // 8
constexpr int N_UNDERPROMO = 3 * 3;          // 9: piece(N,B,R)*3 + dir(fwd,capL,capR)
constexpr int POLICY_PLANES = N_SLIDE + N_KNIGHT + N_UNDERPROMO;  // 121
constexpr int POLICY_SIZE = POLICY_PLANES * CELLS;                // 23716
constexpr int ENCODING_VERSION = 1;

// Mover-frame square (0..159 in the rules library's dense indexing, after rotation).
inline int canon_sq(quad::Color stm, int sq) { return quad::geo::SEAT_INV[stm][sq]; }
// Cell index (rank*14 + file) of a mover-frame square.
inline int cell_of(int csq) { return quad::geo::RANK_OF[csq] * BOARD + quad::geo::FILE_OF[csq]; }

// Fill out[INPUT_FLOATS] (zeroed first). Position is not modified.
void encode(const quad::Position& pos, float* out);
// hist: nhist records of REC_BYTES, NEWEST FIRST (hist[0] is the position one ply ago). nhist may be 0..HIST_POS.
void encode(const quad::Position& pos, const unsigned char* hist, int nhist, float* out);

// labels computed EXACTLY from the rules, written
// to a backward-compatible sp_aux.bin side file. These are TRAINING TARGETS ONLY - the search never
// reads them, which is what keeps the no-hand-coded-knowledge rule intact.
//   byte 0, bits 0..3 : a LEGAL move available to the side to move captures seat k's king - i.e. literally
//                       "can be captured on the next move". NOT in_check: attacked_by_team is geometric and
//                       disagrees with the legal-move derivation on 3.4% of positions (366 of 10,795
//                       measured), because an attacker can be pinned. The order asks for capturable.
//   byte 0, bits 4..7 : seat k's king is geometrically attacked by the opposing team (Position::in_check),
//                       kept alongside because it is free and is a different, weaker signal.
//   bytes 1..25       : a 196-bit mask over the MOVER-FRAME cells (the same cell_of(canon_sq(stm, sq))
//                       indexing the input planes use) of the mover team's own pieces that are attacked
//                       by the opposing team and NOT defended by the mover's team.
constexpr int AUX_BYTES = 26;
void fill_aux(const quad::Position& pos, unsigned char* out);
// the same square -> plane-cell mapping the encoder uses, exported so a TEST can line labels up with
// planes instead of reimplementing canon_sq/cell_of and drifting from them
int mover_cell_of(const quad::Position& pos, int sq);
// The 136-byte self-play record. ONE writer, used by self-play, the MCTS history ring and the tests.
void fill_record(const quad::Position& pos, unsigned char* r);

// Policy index of a move for the side to move in pos; -1 if the move has no slot
// (cannot happen for a legal move — asserted in tests).
int policy_index(const quad::Position& pos, quad::Move m);

// Convenience: legal moves and their policy indices.
struct LegalSet {
    std::vector<quad::Move> moves;
    std::vector<int> idx;
};
LegalSet legal_with_index(quad::Position& pos);

// Final material balance, Red+Yellow minus Blue+Green, pawn units (P1 N3 B5 R5 Q9), clamped to int8
// (an auxiliary training target; the validator recomputes it).
int8_t material_ry(const quad::Position& p);

}  // namespace zero
