// Rc0 - board geometry tables.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "geometry.h"

#include <cstdlib>
#include <stdexcept>

namespace quad {
namespace geo {

int SQ[FILE_NB][RANK_NB];
int FILE_OF[SQUARE_NB];
int RANK_OF[SQUARE_NB];
int16_t RAY[SQUARE_NB][DIR_NB][14];
int16_t KNIGHT_TO[SQUARE_NB][9];
int16_t KING_TO[SQUARE_NB][9];
int PAWN_PUSH[COLOR_NB][SQUARE_NB];
int16_t PAWN_ATTACK[COLOR_NB][SQUARE_NB][3];
int16_t PAWN_ATTACK_FROM[COLOR_NB][SQUARE_NB][3];
int RELATIVE_RANK[COLOR_NB][SQUARE_NB];
bool PAWN_START[COLOR_NB][SQUARE_NB];
int PROMO_DIST[COLOR_NB][SQUARE_NB];
uint8_t ALIGNED[SQUARE_NB][SQUARE_NB];
int SEAT_MAP[COLOR_NB][SQUARE_NB];
int SEAT_INV[COLOR_NB][SQUARE_NB];
Color ROLE_UNDER[COLOR_NB][COLOR_NB];
CastleInfo CASTLE[COLOR_NB][2];

namespace {

struct FR { int f, r; };

bool playable(int f, int r) {
    if (f < 0 || f >= FILE_NB || r < 0 || r >= RANK_NB) return false;
    const bool edgeFile = f <= 2 || f >= 11;
    const bool edgeRank = r <= 2 || r >= 11;
    return !(edgeFile && edgeRank);  // the four 3x3 corners are not part of the board
}

int at(int f, int r) { return playable(f, r) ? SQ[f][r] : SQ_NONE; }

// Rotate a coordinate from Red's frame into seat c's frame, and back.
FR to_seat(Color c, FR p) {
    switch (c) {
    case BLUE:   return {p.r, 13 - p.f};
    case YELLOW: return {13 - p.f, 13 - p.r};
    case GREEN:  return {13 - p.r, p.f};
    default:     return p;
    }
}
FR from_seat(Color c, FR p) {
    switch (c) {
    case BLUE:   return {13 - p.r, p.f};
    case YELLOW: return {13 - p.f, 13 - p.r};
    case GREEN:  return {p.r, 13 - p.f};
    default:     return p;
    }
}

// Forward direction of each seat's pawns, as (df, dr).
constexpr FR FORWARD[COLOR_NB] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
// Ray directions N, E, S, W, NE, SE, SW, NW.
constexpr FR DIRS[DIR_NB] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}, {1, 1}, {1, -1}, {-1, -1}, {-1, 1}};
constexpr FR KNIGHT_JUMPS[8] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};

template <size_t N>
void terminate_list(int16_t (&list)[N], int count) { list[count] = SQ_NONE; }

}  // namespace

void init() {
    int next = 0;
    for (int r = 0; r < RANK_NB; r++)
        for (int f = 0; f < FILE_NB; f++)
            SQ[f][r] = playable(f, r) ? next++ : SQ_NONE;
    if (next != SQUARE_NB) throw std::logic_error("geometry: expected 160 playable squares");
    for (int f = 0; f < FILE_NB; f++)
        for (int r = 0; r < RANK_NB; r++)
            if (SQ[f][r] != SQ_NONE) { FILE_OF[SQ[f][r]] = f; RANK_OF[SQ[f][r]] = r; }

    for (int s = 0; s < SQUARE_NB; s++) {
        const int f = FILE_OF[s], r = RANK_OF[s];

        for (int d = 0; d < DIR_NB; d++) {
            int n = 0;
            for (int k = 1;; k++) {
                int t = at(f + DIRS[d].f * k, r + DIRS[d].r * k);
                if (t == SQ_NONE) break;
                RAY[s][d][n++] = int16_t(t);
            }
            RAY[s][d][n] = SQ_NONE;
        }

        int n = 0;
        for (const FR& j : KNIGHT_JUMPS)
            if (int t = at(f + j.f, r + j.r); t != SQ_NONE) KNIGHT_TO[s][n++] = int16_t(t);
        terminate_list(KNIGHT_TO[s], n);

        n = 0;
        for (const FR& d : DIRS)
            if (int t = at(f + d.f, r + d.r); t != SQ_NONE) KING_TO[s][n++] = int16_t(t);
        terminate_list(KING_TO[s], n);

        for (int c = 0; c < COLOR_NB; c++) {
            const FR fw = FORWARD[c];
            const FR side = {fw.r != 0 ? 1 : 0, fw.f != 0 ? 1 : 0};  // perpendicular to forward
            PAWN_PUSH[c][s] = at(f + fw.f, r + fw.r);
            int na = 0, nb = 0;
            for (int sgn : {-1, 1}) {
                int ta = at(f + fw.f + sgn * side.f, r + fw.r + sgn * side.r);
                if (ta != SQ_NONE) PAWN_ATTACK[c][s][na++] = int16_t(ta);
                int tb = at(f - fw.f + sgn * side.f, r - fw.r + sgn * side.r);
                if (tb != SQ_NONE) PAWN_ATTACK_FROM[c][s][nb++] = int16_t(tb);
            }
            terminate_list(PAWN_ATTACK[c][s], na);
            terminate_list(PAWN_ATTACK_FROM[c][s], nb);

            // Distance travelled from the seat's own back edge.
            const int rel = from_seat(Color(c), {f, r}).r;
            RELATIVE_RANK[c][s] = rel;
            PAWN_START[c][s] = rel == 1;
            PROMO_DIST[c][s] = rel >= 10 ? 0 : 10 - rel;
        }

        for (int c = 0; c < COLOR_NB; c++) {
            const FR a = to_seat(Color(c), {f, r});
            const FR b = from_seat(Color(c), {f, r});
            SEAT_MAP[c][s] = at(a.f, a.r);
            SEAT_INV[c][s] = at(b.f, b.r);
            if (SEAT_MAP[c][s] == SQ_NONE || SEAT_INV[c][s] == SQ_NONE)
                throw std::logic_error("geometry: seat rotation left the board");
        }
    }

    for (int a = 0; a < SQUARE_NB; a++)
        for (int b = 0; b < SQUARE_NB; b++) {
            const int df = std::abs(FILE_OF[a] - FILE_OF[b]);
            const int dr = std::abs(RANK_OF[a] - RANK_OF[b]);
            ALIGNED[a][b] = uint8_t(a != b && (df == 0 || dr == 0 || df == dr));
        }

    // Seat roles: rotate each army's king home by the mover's inverse rotation and see whose
    // home it lands on. Every home king square is Red's h1 rotated into that seat.
    const int redKing = SQ[7][0];
    int home[COLOR_NB];
    for (int c = 0; c < COLOR_NB; c++) home[c] = SEAT_MAP[c][redKing];
    for (int m = 0; m < COLOR_NB; m++)
        for (int c = 0; c < COLOR_NB; c++) {
            const int lands = SEAT_INV[m][home[c]];
            int role = -1;
            for (int t = 0; t < COLOR_NB; t++)
                if (home[t] == lands) role = t;
            if (role < 0) throw std::logic_error("geometry: seat role not found");
            ROLE_UNDER[m][c] = Color(role);
        }

    // Castling, written for Red on rank 1 and rotated to the other seats.
    struct Red { int kTo, rFrom, rTo; int empty[4]; int safe[4]; };
    const Red red[2] = {
        {SQ[9][0], SQ[10][0], SQ[8][0], {SQ[8][0], SQ[9][0], SQ_NONE, SQ_NONE}, {SQ[8][0], SQ[9][0], SQ_NONE, SQ_NONE}},
        {SQ[5][0], SQ[3][0], SQ[6][0], {SQ[4][0], SQ[5][0], SQ[6][0], SQ_NONE}, {SQ[6][0], SQ[5][0], SQ_NONE, SQ_NONE}},
    };
    for (int c = 0; c < COLOR_NB; c++)
        for (int side = 0; side < 2; side++) {
            const Red& w = red[side];
            CastleInfo& ci = CASTLE[c][side];
            ci.kFrom = SEAT_MAP[c][redKing];
            ci.kTo = SEAT_MAP[c][w.kTo];
            ci.rFrom = SEAT_MAP[c][w.rFrom];
            ci.rTo = SEAT_MAP[c][w.rTo];
            for (int i = 0; i < 4; i++) {
                ci.empty[i] = w.empty[i] == SQ_NONE ? SQ_NONE : SEAT_MAP[c][w.empty[i]];
                ci.safe[i] = w.safe[i] == SQ_NONE ? SQ_NONE : SEAT_MAP[c][w.safe[i]];
            }
        }
}

std::string sq_name(int sq) {
    if (sq < 0 || sq >= SQUARE_NB) return "??";
    return std::string(1, char('a' + FILE_OF[sq])) + std::to_string(RANK_OF[sq] + 1);
}

int parse_sq(const char* s, int* len) {
    if (!s || s[0] < 'a' || s[0] > 'n' || s[1] < '0' || s[1] > '9') return SQ_NONE;
    int rank = s[1] - '0', used = 2;
    if (s[2] >= '0' && s[2] <= '9') { rank = rank * 10 + (s[2] - '0'); used = 3; }
    const int sq = at(s[0] - 'a', rank - 1);
    if (sq != SQ_NONE && len) *len = used;
    return sq;
}

}  // namespace geo
}  // namespace quad
