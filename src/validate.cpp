// SPDX-License-Identifier: GPL-3.0-or-later
// rc0 validate DIR: replay every game of a self-play directory through the rules and check every
// training row against the replay. This is what lets a server accept games from strangers: a chunk
// passes only if every move is legal, every row is the position it claims to be, every policy target
// covers legal moves only and sums to 1, and every result is the one the rules (or the adjudicator) gave.
//
// games.txt, one line per game, tab-separated, in the same order as the rows in the sp_*.bin files:
//   start FEN4 | random opening plies | ply cap | end (natural|capped|adjudicated) | result (RY view: 1, 0, -1)
//   | all moves, space-separated (opening plies first) | searched-ply index of each recorded row, comma-separated
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "encoding.h"
#include "movegen.h"
#include "selfplay.h"

namespace zero {
using namespace quad;

namespace {

constexpr int POL_BYTES = 96 * 4;
constexpr int HIST_BYTES = HIST_POS * REC_BYTES + 1;

float half_to_float(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, man = h & 0x3FF, x;
    if (exp == 0) x = sign;                                   // the writer flushes subnormals to zero
    else if (exp == 31) x = sign | 0x7F800000u | (man << 13);
    else x = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float f; std::memcpy(&f, &x, 4); return f;
}

struct File {
    FILE* f = nullptr;
    long rows = 0;
    int width = 0;
    bool open(const std::string& path, int w) {
        width = w;
        f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        std::fseek(f, 0, SEEK_END);
        long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (n % w) return false;
        rows = n / w;
        return true;
    }
    bool read(void* dst) { return std::fread(dst, 1, size_t(width), f) == size_t(width); }
    ~File() { if (f) std::fclose(f); }
};

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream is(s);
    while (std::getline(is, cur, sep)) out.push_back(cur);
    if (!s.empty() && s.back() == sep) out.push_back("");
    return out;
}

}  // namespace

int validate_main(const std::string& dir, bool verbose, bool gamesOnly) {
    auto fail = [&](long game, const std::string& why) {
        std::printf("INVALID game %ld: %s\n", game, why.c_str());
        std::printf("{\"valid\": false, \"game\": %ld, \"reason\": \"%s\"}\n", game, why.c_str());
        return 1;
    };
    std::ifstream games(dir + "/games.txt");
    if (!games) return fail(-1, "no games.txt");
    File frec, fpol, fres, fq, fply, fmat, fhist, faux;
    if (!gamesOnly && (!frec.open(dir + "/sp_rec.bin", REC_BYTES) || !fpol.open(dir + "/sp_pol.bin", POL_BYTES) ||
        !fres.open(dir + "/sp_res.bin", 1) || !fq.open(dir + "/sp_q.bin", 4) || !fply.open(dir + "/sp_ply.bin", 2) ||
        !fmat.open(dir + "/sp_mat.bin", 1) || !fhist.open(dir + "/sp_hist.bin", HIST_BYTES) ||
        !faux.open(dir + "/sp_aux.bin", AUX_BYTES)))
        return fail(-1, "missing or truncated sp_*.bin file");
    const long nrows = gamesOnly ? 0 : frec.rows;
    if (!gamesOnly)
        for (File* f : {&fpol, &fres, &fq, &fply, &fmat, &fhist, &faux})
            if (f->rows != nrows) return fail(-1, "sp_*.bin files disagree on the row count");

    long g = 0, rowsSeen = 0, ry = 0, bg = 0, draws = 0, plies = 0;
    std::string line;
    while (std::getline(games, line)) {
        if (line.empty()) continue;
        if (line.size() > 200000) return fail(g, "game line too long");
        auto f = split(line, '\t');
        if (f.size() != 7) return fail(g, "expected 7 tab-separated fields");
        Position pos;
        if (!pos.set_fen4(f[0])) return fail(g, "bad start FEN4");
        // a game must start from the standard position or a shuffled back rank (same for all seats)
        { Position sp; sp.init_startpos(); if (pos.fen4() != sp.fen4() && pos.castling_rights() != 0) return fail(g, "unexpected start position"); }
        const int nrand = std::atoi(f[1].c_str()), cap = std::atoi(f[2].c_str()), result = std::atoi(f[4].c_str());
        const std::string end = f[3];
        if (end != "natural" && end != "capped" && end != "adjudicated") return fail(g, "unknown end reason " + end);
        if (gamesOnly && end == "adjudicated") return fail(g, "match games are played to the end");
        if (result < -1 || result > 1) return fail(g, "bad result");
        std::vector<std::string> moves = f[5].empty() ? std::vector<std::string>{} : split(f[5], ' ');
        if (nrand < 0 || nrand > int(moves.size())) return fail(g, "bad opening ply count");
        std::set<int> recPlies;
        std::vector<int> recOrder;
        if (!f[6].empty())
            for (auto& t : split(f[6], ',')) { int k = std::atoi(t.c_str()); recOrder.push_back(k); recPlies.insert(k); }
        for (size_t i = 1; i < recOrder.size(); i++) if (recOrder[i] <= recOrder[i - 1]) return fail(g, "row plies not increasing");
        const int searched = int(moves.size()) - nrand;
        if (!recOrder.empty() && (recOrder.front() < 0 || recOrder.back() >= searched)) return fail(g, "row ply out of range");
        if (rowsSeen + long(recOrder.size()) > nrows) return fail(g, "more rows claimed than stored");

        unsigned char ring[HIST_POS * REC_BYTES] = {0};
        int nring = 0;
        std::vector<int> rowPly;
        size_t nextRec = 0;
        for (int i = 0; i < int(moves.size()); i++) {
            bool over = false;
            game_result(pos, over);
            if (over) return fail(g, "move " + std::to_string(i) + " played after the game ended");
            const int k = i - nrand;   // searched-ply index (negative during the random opening)
            if (k >= 0 && nextRec < recOrder.size() && recOrder[nextRec] == k) {
                unsigned char rec[REC_BYTES], want[REC_BYTES], aux[AUX_BYTES], wantAux[AUX_BYTES], hist[HIST_BYTES];
                unsigned char pol[POL_BYTES];
                frec.read(rec); faux.read(aux); fhist.read(hist); fpol.read(pol);
                fill_record(pos, want);
                fill_aux(pos, wantAux);
                if (std::memcmp(rec, want, REC_BYTES)) return fail(g, "row at ply " + std::to_string(k) + " is not the replayed position");
                if (std::memcmp(aux, wantAux, AUX_BYTES)) return fail(g, "aux labels at ply " + std::to_string(k) + " do not match the rules");
                if (std::memcmp(hist, ring, HIST_POS * REC_BYTES) || hist[HIST_POS * REC_BYTES] != nring)
                    return fail(g, "history at ply " + std::to_string(k) + " does not match the replay");
                LegalSet ls = legal_with_index(pos);
                std::set<int> legal(ls.idx.begin(), ls.idx.end());
                std::set<int> seen;
                double sum = 0.0;
                bool tail = false;
                for (int j = 0; j < 96; j++) {
                    int16_t idx; uint16_t p;
                    std::memcpy(&idx, pol + 4 * j, 2); std::memcpy(&p, pol + 4 * j + 2, 2);
                    float pr = half_to_float(p);
                    if (idx < 0) { tail = true; if (p != 0) return fail(g, "probability on an empty policy slot"); continue; }
                    if (tail) return fail(g, "policy entry after the end marker");
                    if (!legal.count(idx)) return fail(g, "policy target on an illegal move at ply " + std::to_string(k));
                    if (!seen.insert(idx).second) return fail(g, "duplicate policy entry");
                    if (!(pr >= 0.0f && pr <= 1.0f)) return fail(g, "policy probability out of range");
                    sum += pr;
                }
                // 96 slots hold every move with real probability; fp16 rounding and the cut-off tail leave a little slack
                if (sum < 0.95 || sum > 1.03) return fail(g, "policy target does not sum to 1 at ply " + std::to_string(k));
                rowPly.push_back(k);
                nextRec++;
            }
            if (k >= 0) {   // self-play's history ring holds only searched positions
                if (HIST_POS > 1) std::memmove(ring + REC_BYTES, ring, size_t(HIST_POS - 1) * REC_BYTES);
                fill_record(pos, ring);
                if (nring < HIST_POS) nring++;
            }
            Move m = pos.parse_move(moves[i]);
            if (m == MOVE_NONE) return fail(g, "illegal move " + moves[i] + " at ply " + std::to_string(i));
            pos.do_move(m);
        }
        if (nextRec != recOrder.size()) return fail(g, "recorded ply not reached");
        bool over = false;
        GameResult gr = game_result(pos, over);
        if (end == "natural") {
            if (!over) return fail(g, "game marked finished but the rules say it is not over");
            int want = gr == TEAM_RY_WINS ? 1 : gr == TEAM_BG_WINS ? -1 : 0;
            if (want != result) return fail(g, "result does not match the rules");
        } else if (end == "capped") {
            if (over || searched != cap || result != 0) return fail(g, "bad ply-cap game");
        } else {   // adjudicated: decided by the search value, so the rules can only check it was still undecided
            if (over || result == 0) return fail(g, "bad adjudicated game");
        }
        // per-row game outcome targets
        const int8_t mat = material_ry(pos);
        for (size_t r = 0; r < recOrder.size(); r++) {
            int8_t res, m8; float q; int16_t left;
            fres.read(&res); fq.read(&q); fply.read(&left); fmat.read(&m8);
            if (res != result) return fail(g, "row result differs from the game result");
            if (left != searched - rowPly[r]) return fail(g, "row plies-left is wrong");
            if (m8 != mat) return fail(g, "row material target is wrong");
            if (!std::isfinite(q) || std::fabs(q) > 1.0001f) return fail(g, "row value out of range");
        }
        rowsSeen += long(recOrder.size());
        plies += long(moves.size());
        if (result > 0) ry++; else if (result < 0) bg++; else draws++;
        if (verbose) std::printf("game %ld ok: %zu moves, %zu rows, %s %d\n", g, moves.size(), recOrder.size(), end.c_str(), result);
        g++;
    }
    if (rowsSeen != nrows) return fail(g, "rows left over that belong to no game");
    std::printf("VALID %ld games, %ld rows\n", g, rowsSeen);
    std::printf("{\"valid\": true, \"games\": %ld, \"rows\": %ld, \"plies\": %ld, \"ry\": %ld, \"bg\": %ld, \"draw\": %ld}\n",
                g, rowsSeen, plies, ry, bg, draws);
    return 0;
}

}  // namespace zero
