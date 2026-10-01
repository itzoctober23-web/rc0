// Rc0 - game state for 4-player chess (Teams): board, move making, check detection, FEN4.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <string>
#include <vector>

#include "geometry.h"
#include "types.h"

namespace quad {

struct Rules {
    int fiftyMovePlies = 200;  // 50 moves by each of the four players
    enum Stalemate { STALEMATE_DRAW, STALEMATE_SIDE_LOSES, STALEMATE_SIDE_WINS };
    Stalemate stalemate = STALEMATE_DRAW;
};
extern Rules rules;

enum GameResult : int { ONGOING = 0, TEAM_RY_WINS, TEAM_BG_WINS, DRAW_RESULT };

class Position {
public:
    static void init_zobrist();

    void init_startpos();
    // Shuffled back rank (the same for all four seats), castling off. For opening variety.
    void init_startpos960(uint64_t seed);
    bool set_fen4(const std::string& fen);
    std::string fen4() const;
    std::string pretty() const;

    Color side_to_move() const { return stm_; }
    Piece piece_on(int sq) const { return board_[sq]; }
    bool empty_sq(int sq) const { return board_[sq] == NO_PIECE; }
    int king_sq(Color c) const { return king_[c]; }
    int count(Color c, PieceType t) const { return count_[c][t]; }
    Key key() const { return top().key; }
    int game_ply() const { return ply_; }
    int rule50_plies() const { return top().rule50; }
    uint8_t castling_rights() const { return top().castling; }
    int ep_sq(Color c) const { return top().epSq[c]; }      // square a c-pawn just skipped, or SQ_NONE
    int ep_pawn(Color c) const { return top().epPawn[c]; }  // where that pawn now stands

    void do_move(Move m);
    void undo_move(Move m);

    // Is sq attacked by any piece of `team` (both partner armies)?
    bool attacked_by_team(int sq, int team) const;
    bool in_check(Color c) const { return king_[c] != SQ_NONE && attacked_by_team(king_[c], team_of(c) ^ 1); }
    bool stm_in_check() const { return in_check(stm_); }

    bool is_draw_by_rule50() const { return top().rule50 >= rules.fiftyMovePlies; }
    int repetition_count() const;  // occurrences of the current position since the last capture/pawn move

    Move parse_move(const std::string& s);  // "h2h3", "j10j11q", castling as the king's 2-square move
    static std::string move_str(Move m);

private:
    struct State {
        Key key = 0;
        int rule50 = 0;
        uint8_t castling = 0;
        int epSq[COLOR_NB] = {SQ_NONE, SQ_NONE, SQ_NONE, SQ_NONE};
        int epPawn[COLOR_NB] = {SQ_NONE, SQ_NONE, SQ_NONE, SQ_NONE};
        Piece captured = NO_PIECE;
        int capturedSq = SQ_NONE;
    };
    const State& top() const { return states_.back(); }
    State& top() { return states_.back(); }

    void reset();
    void add(int sq, Piece p);
    void remove(int sq);
    void relocate(int from, int to);
    void finish_setup();

    Piece board_[SQUARE_NB];
    int king_[COLOR_NB];
    int count_[COLOR_NB][PIECE_TYPE_NB];
    Color stm_ = RED;
    int ply_ = 0;
    std::vector<State> states_;
    std::vector<Key> keys_;  // keys of earlier positions, for repetition counting
};

} // namespace quad
