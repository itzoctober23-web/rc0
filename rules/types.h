// Rc0 (Revenant Chess Zero) - 4-player chess, Teams variant.
// Core value types shared by the rules library and the engine.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <string>

namespace quad {

// Seats in turn order, clockwise from the bottom: Red, Blue, Yellow, Green.
// Teams are the opposite seats: Red+Yellow (team 0) and Blue+Green (team 1),
// so the team to move alternates every ply.
enum Color : int { RED = 0, BLUE = 1, YELLOW = 2, GREEN = 3, COLOR_NB = 4 };

constexpr int team_of(Color c) { return int(c) & 1; }
constexpr Color next_color(Color c) { return Color((int(c) + 1) % 4); }
constexpr Color prev_color(Color c) { return Color((int(c) + 3) % 4); }
constexpr Color partner_of(Color c) { return Color((int(c) + 2) % 4); }
constexpr bool same_team(Color a, Color b) { return team_of(a) == team_of(b); }

constexpr char COLOR_CHAR[COLOR_NB] = {'R', 'B', 'Y', 'G'};

enum PieceType : int { PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING, PIECE_TYPE_NB = 6 };
constexpr char PT_CHAR[PIECE_TYPE_NB] = {'P', 'N', 'B', 'R', 'Q', 'K'};

// A piece is color * 6 + type (0..23). 24 means "empty".
enum Piece : uint8_t { NO_PIECE = 24, PIECE_NB = 24 };
constexpr Piece make_piece(Color c, PieceType t) { return Piece(int(c) * 6 + int(t)); }
constexpr Color color_of(Piece p) { return Color(int(p) / 6); }
constexpr PieceType type_of(Piece p) { return PieceType(int(p) % 6); }

// The board is 14x14 with the four 3x3 corners removed: 160 playable squares,
// numbered 0..159 rank by rank from the bottom (see geometry.h).
constexpr int FILE_NB = 14;
constexpr int RANK_NB = 14;
constexpr int SQUARE_NB = 160;
constexpr int SQ_NONE = -1;

// Moves are packed into 32 bits: from (bits 0-7), to (8-15), promotion piece type
// (16-18, PAWN = no promotion) and a flag (20-23).
enum MoveFlag : int { MF_NONE = 0, MF_DOUBLE_PUSH = 1, MF_EN_PASSANT = 2, MF_CASTLE_K = 3, MF_CASTLE_Q = 4 };
using Move = uint32_t;
constexpr Move MOVE_NONE = 0;
constexpr Move MOVE_NULL = 0xFFFFFFFFu;

constexpr Move make_move(int from, int to, MoveFlag flag = MF_NONE, PieceType promo = PAWN) {
    return Move(from) | (Move(to) << 8) | (Move(promo) << 16) | (Move(flag) << 20);
}
constexpr int from_sq(Move m) { return int(m & 0xFF); }
constexpr int to_sq(Move m) { return int((m >> 8) & 0xFF); }
constexpr PieceType promo_of(Move m) { return PieceType((m >> 16) & 0x7); }
constexpr MoveFlag flag_of(Move m) { return MoveFlag((m >> 20) & 0xF); }
constexpr bool is_promotion(Move m) { return promo_of(m) != PAWN; }

constexpr int MAX_MOVES = 512;
using Key = uint64_t;

} // namespace quad
