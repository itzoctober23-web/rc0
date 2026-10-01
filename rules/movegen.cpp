// Rc0 - move generation and game-end detection for 4-player chess (Teams).
// SPDX-License-Identifier: GPL-3.0-or-later
#include "movegen.h"

#include <cstdio>

namespace quad {

namespace {

constexpr PieceType PROMO_PIECES[4] = {QUEEN, ROOK, BISHOP, KNIGHT};

// A target square is open to `us` if it is empty or holds an enemy piece.
// Own and partner pieces block.
inline bool open_for(const Position& pos, int to, int ourTeam) {
    Piece p = pos.piece_on(to);
    return p == NO_PIECE || team_of(color_of(p)) != ourTeam;
}

inline Move* add_pawn_move(Move* out, Color us, int from, int to, bool capturesOnly) {
    if (geo::PROMO_DIST[us][from] == 1) {
        for (PieceType pt : PROMO_PIECES) *out++ = make_move(from, to, MF_NONE, pt);
    } else if (!capturesOnly) {
        *out++ = make_move(from, to);
    }
    return out;
}

// Shared generator. capturesOnly keeps captures, en passant and promotions.
int generate(const Position& pos, Move* const start, bool capturesOnly) {
    Move* out = start;
    const Color us = pos.side_to_move();
    const int team = team_of(us);

    for (int from = 0; from < SQUARE_NB; ++from) {
        Piece pc = pos.piece_on(from);
        if (pc == NO_PIECE || color_of(pc) != us) continue;
        PieceType pt = type_of(pc);

        if (pt == PAWN) {
            // Diagonal captures of enemy pieces.
            for (const int16_t* a = geo::PAWN_ATTACK[us][from]; *a != SQ_NONE; ++a) {
                Piece t = pos.piece_on(*a);
                if (t != NO_PIECE && team_of(color_of(t)) != team)
                    out = add_pawn_move(out, us, from, *a, false);
            }
            // En passant against any enemy whose pawn just double-pushed and still stands there.
            for (int v = 0; v < COLOR_NB; ++v) {
                if (team_of(Color(v)) == team) continue;
                int ep = pos.ep_sq(Color(v));
                int victim = pos.ep_pawn(Color(v));
                if (ep == SQ_NONE || victim == SQ_NONE) continue;
                if (!pos.empty_sq(ep) || pos.piece_on(victim) != make_piece(Color(v), PAWN)) continue;
                for (const int16_t* a = geo::PAWN_ATTACK[us][from]; *a != SQ_NONE; ++a)
                    if (*a == ep) *out++ = make_move(from, ep, MF_EN_PASSANT);
            }
            // Pushes.
            int one = geo::PAWN_PUSH[us][from];
            if (one != SQ_NONE && pos.empty_sq(one)) {
                out = add_pawn_move(out, us, from, one, capturesOnly);
                if (!capturesOnly && geo::PAWN_START[us][from]) {
                    int two = geo::PAWN_PUSH[us][one];
                    if (two != SQ_NONE && pos.empty_sq(two))
                        *out++ = make_move(from, two, MF_DOUBLE_PUSH);
                }
            }
            continue;
        }

        if (pt == KNIGHT || pt == KING) {
            const int16_t* to = (pt == KNIGHT) ? geo::KNIGHT_TO[from] : geo::KING_TO[from];
            for (; *to != SQ_NONE; ++to) {
                if (!open_for(pos, *to, team)) continue;
                if (capturesOnly && pos.empty_sq(*to)) continue;
                *out++ = make_move(from, *to);
            }
            continue;
        }

        // Sliders: rook directions 0..3, bishop 4..7, queen all eight.
        int d0 = (pt == BISHOP) ? 4 : 0;
        int d1 = (pt == ROOK) ? 4 : 8;
        for (int d = d0; d < d1; ++d) {
            for (const int16_t* r = geo::RAY[from][d]; *r != SQ_NONE; ++r) {
                Piece t = pos.piece_on(*r);
                if (t == NO_PIECE) {
                    if (!capturesOnly) *out++ = make_move(from, *r);
                    continue;
                }
                if (team_of(color_of(t)) != team) *out++ = make_move(from, *r);
                break;
            }
        }
    }

    // Castling: rights kept, path empty, king not in check and not crossing or landing
    // on an attacked square.
    if (!capturesOnly) {
        const int enemy = team ^ 1;
        for (int side = 0; side < 2; ++side) {
            if (!(pos.castling_rights() & geo::castle_bit(us, side))) continue;
            const geo::CastleInfo& ci = geo::CASTLE[us][side];
            if (pos.piece_on(ci.kFrom) != make_piece(us, KING)) continue;
            if (pos.piece_on(ci.rFrom) != make_piece(us, ROOK)) continue;
            bool ok = true;
            for (const int* e = ci.empty; ok && *e != SQ_NONE; ++e) ok = pos.empty_sq(*e);
            if (!ok || pos.attacked_by_team(ci.kFrom, enemy)) continue;
            for (const int* s = ci.safe; ok && *s != SQ_NONE; ++s) ok = !pos.attacked_by_team(*s, enemy);
            if (ok) *out++ = make_move(ci.kFrom, ci.kTo, side ? MF_CASTLE_Q : MF_CASTLE_K);
        }
    }
    return int(out - start);
}

}  // namespace

int generate_pseudo(const Position& pos, Move* out) { return generate(pos, out, false); }
int generate_captures(const Position& pos, Move* out) { return generate(pos, out, true); }

bool legal_after(Position& pos, Move m) {
    Color us = pos.side_to_move();
    pos.do_move(m);
    bool ok = !pos.in_check(us);
    pos.undo_move(m);
    return ok;
}

int generate_legal(Position& pos, Move* out) {
    Move pseudo[MAX_MOVES];
    int n = generate_pseudo(pos, pseudo);
    const Color us = pos.side_to_move();
    const bool inCheck = pos.in_check(us);
    int k = 0;
    for (int i = 0; i < n; ++i) {
        Move m = pseudo[i];
        Piece t = pos.piece_on(to_sq(m));
        bool takesKing = t != NO_PIECE && type_of(t) == KING && team_of(color_of(t)) != team_of(us);
        if (takesKing || !needs_verify(pos, m, inCheck) || legal_after(pos, m)) out[k++] = m;
    }
    return k;
}

uint64_t perft(Position& pos, int depth, bool root) {
    if (depth <= 0) return 1;
    Move moves[MAX_MOVES];
    int n = generate_legal(pos, moves);
    if (depth == 1 && !root) return uint64_t(n);
    uint64_t total = 0;
    for (int i = 0; i < n; ++i) {
        pos.do_move(moves[i]);
        uint64_t c = perft(pos, depth - 1, false);
        pos.undo_move(moves[i]);
        if (root) std::printf("%s: %llu\n", Position::move_str(moves[i]).c_str(), (unsigned long long)c);
        total += c;
    }
    return total;
}

GameResult game_result(Position& pos, bool& gameOver) {
    gameOver = true;
    // A captured king loses the game for that king's team.
    for (int c = 0; c < COLOR_NB; ++c)
        if (pos.count(Color(c), KING) == 0)
            return team_of(Color(c)) == 0 ? TEAM_BG_WINS : TEAM_RY_WINS;

    const Color us = pos.side_to_move();
    const GameResult usWin = team_of(us) == 0 ? TEAM_RY_WINS : TEAM_BG_WINS;
    const GameResult usLose = team_of(us) == 0 ? TEAM_BG_WINS : TEAM_RY_WINS;

    Move moves[MAX_MOVES];
    if (generate_legal(pos, moves) == 0) {
        if (pos.in_check(us)) return usLose;  // checkmate: decided when the mated side is to move
        switch (rules.stalemate) {
            case Rules::STALEMATE_SIDE_LOSES: return usLose;
            case Rules::STALEMATE_SIDE_WINS: return usWin;
            default: return DRAW_RESULT;
        }
    }
    if (pos.is_draw_by_rule50() || pos.repetition_count() >= 3) return DRAW_RESULT;

    gameOver = false;
    return ONGOING;
}

}  // namespace quad
