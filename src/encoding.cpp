// SPDX-License-Identifier: GPL-3.0-or-later
#include "encoding.h"
#include <algorithm>
#include <cstring>
#include "geometry.h"
#include "movegen.h"

namespace zero {
using namespace quad;

// Slide directions in the mover frame: E, W, N, S, NE, NW, SE, SW (RASA order).
static const int SDF[8] = {1, -1, 0, 0, 1, -1, 1, -1};
static const int SDR[8] = {0, 0, 1, -1, 1, 1, -1, -1};
static const int KDF[8] = {1, 2, 2, 1, -1, -2, -2, -1};
static const int KDR[8] = {2, 1, -1, -2, -2, -1, 1, 2};

void encode(const Position& pos, float* out) { encode(pos, nullptr, 0, out); }

void encode(const Position& pos, const unsigned char* hist, int nhist, float* out) {
    std::memset(out, 0, sizeof(float) * INPUT_FLOATS);
    const Color stm = pos.side_to_move();
    // pieces
    for (int sq = 0; sq < SQUARE_NB; sq++) {
        Piece p = pos.piece_on(sq);
        if (p == NO_PIECE) continue;
        int role = int(geo::ROLE_UNDER[stm][color_of(p)]);
        int plane = PL_PIECES + role * 6 + int(type_of(p));
        out[plane * CELLS + cell_of(canon_sq(stm, sq))] = 1.0f;
    }
    // mask
    for (int sq = 0; sq < SQUARE_NB; sq++) out[PL_MASK * CELLS + cell_of(sq)] = 1.0f;
    // castling rights, per role, constant planes
    uint8_t cr = pos.castling_rights();
    for (int c = 0; c < COLOR_NB; c++) {
        int role = int(geo::ROLE_UNDER[stm][c]);
        for (int side = 0; side < 2; side++)
            if (cr & geo::castle_bit(Color(c), side)) {
                float* pl = out + (PL_CASTLE + role * 2 + side) * CELLS;
                for (int i = 0; i < CELLS; i++) pl[i] = 1.0f;
            }
    }
    // en passant targets, per role
    for (int c = 0; c < COLOR_NB; c++) {
        int ep = pos.ep_sq(Color(c));
        if (ep == SQ_NONE) continue;
        int role = int(geo::ROLE_UNDER[stm][c]);
        out[(PL_EP + role) * CELLS + cell_of(canon_sq(stm, ep))] = 1.0f;
    }
    // rule50, repetition, first-team flag: constant planes
    float r50 = float(pos.rule50_plies()) / float(rules.fiftyMovePlies);
    if (r50 > 1.0f) r50 = 1.0f;
    // REPETITION PLANES ARE DELIBERATELY LEFT AT ZERO (2026-09-30 21:52, rule 8/9 encoding bug).
    // The 136-byte self-play record carries no repetition count, so the python trainer
    // (train/train_supervised.py GpuEncoder, used by train_selfplay.py) can never set planes 37/38:
    // they are identically 0 in EVERY training sample. Setting them here made the net see an input
    // pattern at play time (self-play AND the UCI engine, both via mcts.cpp -> encode()) that it had
    // never seen in training, exactly in repeated positions. Zeroing them here makes play match
    // training. Proper fix later: carry rep in the record and set it on both sides.
    int rep = 1;  // was pos.repetition_count(); see above
    float ft = team_of(stm) == 0 ? 1.0f : 0.0f;
    for (int i = 0; i < CELLS; i++) {
        out[PL_RULE50 * CELLS + i] = r50;
        if (rep >= 2) out[(PL_REP + 0) * CELLS + i] = 1.0f;
        if (rep >= 3) out[(PL_REP + 1) * CELLS + i] = 1.0f;
        out[PL_FIRSTTEAM * CELLS + i] = ft;
    }
    // history planes (input history): each previous position's pieces, in the CURRENT mover's frame
#ifdef ZERO_HISTORY
    if (hist && nhist > 0) {
        const int nh = nhist < HIST_POS ? nhist : HIST_POS;
        for (int h = 0; h < nh; h++) {
            const unsigned char* r = hist + h * REC_BYTES;
            int np = int(r[3]);
            if (np > 64) np = 64;
            for (int i = 0; i < np; i++) {
                Piece pc = Piece(r[8 + 2 * i]);
                int sq = int(r[9 + 2 * i]);
                if (pc == NO_PIECE || sq < 0 || sq >= SQUARE_NB) continue;
                int role = int(geo::ROLE_UNDER[stm][color_of(pc)]);
                out[(PL_HIST + h * 24 + role * 6 + int(type_of(pc))) * CELLS + cell_of(canon_sq(stm, sq))] = 1.0f;
            }
        }
        if (nh >= HIST_POS)
            for (int i = 0; i < CELLS; i++) out[PL_HISTVALID * CELLS + i] = 1.0f;
    }
#else
    (void)hist; (void)nhist;
#endif
}

int mover_cell_of(const Position& p, int sq) { return cell_of(canon_sq(p.side_to_move(), sq)); }

void fill_aux(const Position& p, unsigned char* out) {
    std::memset(out, 0, AUX_BYTES);
    // (a) bits 0..3: a LEGAL move of the side to move captures seat c's king. This is the order's
    // "can any king be captured on the next move", and it is NOT the same as in_check - a geometric
    // attacker can be pinned, which is why the two disagreed on 3.4% of positions when measured.
    {
        Position& q = const_cast<Position&>(p);   // do_move/undo_move restore exactly; p is logically const
        quad::Move list[quad::MAX_MOVES];
        int n = quad::generate_legal(q, list);
        for (int i = 0; i < n; i++) {
            int to = int(to_sq(list[i]));
            Piece vic = p.piece_on(to);
            if (vic == NO_PIECE || type_of(vic) != KING) continue;
            if (flag_of(list[i]) == MF_EN_PASSANT) continue;
            out[0] |= (unsigned char)(1u << int(color_of(vic)));
        }
    }
    // bits 4..7: the weaker geometric signal, free to compute
    for (int c = 0; c < COLOR_NB; c++)
        if (p.in_check(Color(c))) out[0] |= (unsigned char)(1u << (4 + c));
    // (b) our pieces attacked and not defended. Only the cells that actually hold one of our pieces are
    // queried - about 32 squares, not all 196 - so this costs ~64 attack queries a row instead of 392.
    const Color stm = p.side_to_move();
    const int us = team_of(stm), them = us ^ 1;
    for (int sq = 0; sq < SQUARE_NB; sq++) {
        Piece pc = p.piece_on(sq);
        if (pc == NO_PIECE) continue;
        if (team_of(color_of(pc)) != us) continue;
        if (!p.attacked_by_team(sq, them)) continue;
        if (p.attacked_by_team(sq, us)) continue;          // defended, so not the label we want
        int cell = mover_cell_of(p, sq);
        if (cell < 0 || cell >= CELLS) continue;
        out[1 + cell / 8] |= (unsigned char)(1u << (cell % 8));
    }
}

void fill_record(const Position& p, unsigned char* r) {   // moved here 2026-10-01 05:06 so MCTS can reuse it
    std::memset(r, 0xFF, 136);
    r[0] = (unsigned char)p.side_to_move();
    r[1] = p.castling_rights();
    int r50 = p.rule50_plies(); r[2] = (unsigned char)(r50 > 255 ? 255 : r50);
    for (int c = 0; c < COLOR_NB; c++) { int e = p.ep_sq(Color(c)); r[4 + c] = (unsigned char)(e == SQ_NONE ? 0 : e + 1); }
    int np = 0;
    for (int sq = 0; sq < SQUARE_NB && np < 64; sq++) {
        Piece pc = p.piece_on(sq);
        if (pc == NO_PIECE) continue;
        r[8 + 2 * np] = (unsigned char)pc; r[9 + 2 * np] = (unsigned char)sq; np++;
    }
    r[3] = (unsigned char)np;
}

int policy_index(const Position& pos, Move m) {
    const Color stm = pos.side_to_move();
    int from = canon_sq(stm, from_sq(m)), to = canon_sq(stm, to_sq(m));
    int df = geo::FILE_OF[to] - geo::FILE_OF[from];
    int dr = geo::RANK_OF[to] - geo::RANK_OF[from];
    int plane = -1;
    PieceType promo = promo_of(m);
    if (is_promotion(m) && promo != QUEEN) {
        // mover pawns push toward +rank in the mover frame; dr == 1 always
        int piece = promo == KNIGHT ? 0 : promo == BISHOP ? 1 : 2;
        int dir = df == 0 ? 0 : df < 0 ? 1 : 2;
        plane = N_SLIDE + N_KNIGHT + piece * 3 + dir;
    } else {
        int adf = df < 0 ? -df : df, adr = dr < 0 ? -dr : dr;
        if ((adf == 1 && adr == 2) || (adf == 2 && adr == 1)) {
            for (int k = 0; k < 8; k++)
                if (KDF[k] == df && KDR[k] == dr) { plane = N_SLIDE + k; break; }
        } else if (df == 0 || dr == 0 || adf == adr) {
            int sdf = (df > 0) - (df < 0), sdr = (dr > 0) - (dr < 0);
            int dist = adf > adr ? adf : adr;
            for (int d = 0; d < 8; d++)
                if (SDF[d] == sdf && SDR[d] == sdr) { plane = d * SLIDE_MAX + (dist - 1); break; }
        }
    }
    if (plane < 0) return -1;
    return plane * CELLS + cell_of(from);
}

LegalSet legal_with_index(Position& pos) {
    LegalSet ls;
    Move list[MAX_MOVES];
    int n = generate_legal(pos, list);
    ls.moves.assign(list, list + n);
    ls.idx.resize(n);
    for (int i = 0; i < n; i++) ls.idx[i] = policy_index(pos, list[i]);
    return ls;
}

// final material balance, RY minus BG, pawn units (P1 N3 B5 R5 Q9; kings excluded), clamped to int8
int8_t material_ry(const quad::Position& p) {
    static const int val[] = {1, 3, 5, 5, 9, 0};
    int m = 0;
    for (int sq = 0; sq < quad::SQUARE_NB; sq++) {
        quad::Piece pc = p.piece_on(sq);
        if (pc == quad::NO_PIECE) continue;
        int v = val[int(quad::type_of(pc))];
        m += (quad::team_of(quad::color_of(pc)) == 0) ? v : -v;
    }
    return int8_t(std::max(-127, std::min(127, m)));
}

}  // namespace zero
