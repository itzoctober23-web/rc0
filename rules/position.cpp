// Rc0 - game state for 4-player chess (Teams).
// SPDX-License-Identifier: GPL-3.0-or-later
#include "position.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "movegen.h"

namespace quad {

Rules rules;

// ------------------------------------------------------------------- hashing ---
namespace {

Key Z_PIECE[PIECE_NB][SQUARE_NB];
Key Z_STM[COLOR_NB];
Key Z_CASTLE_BIT[8];
Key Z_EP[COLOR_NB][SQUARE_NB];
uint8_t CASTLE_KEEP[SQUARE_NB];  // castling bits that survive a move touching this square

Key castle_key(uint8_t rights) {
    Key k = 0;
    for (int b = 0; b < 8; b++)
        if (rights & (1 << b)) k ^= Z_CASTLE_BIT[b];
    return k;
}

uint64_t next_random(uint64_t& state) {  // SplitMix64
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

const char COLOR_LOWER[COLOR_NB] = {'r', 'b', 'y', 'g'};

int color_index(char ch) {
    for (int c = 0; c < COLOR_NB; c++)
        if (COLOR_LOWER[c] == char(std::tolower((unsigned char)ch))) return c;
    return -1;
}
int type_index(char ch) {
    for (int t = 0; t < PIECE_TYPE_NB; t++)
        if (PT_CHAR[t] == char(std::toupper((unsigned char)ch))) return t;
    return -1;
}

std::vector<std::string> split(const std::string& s, char sep, bool respectBrackets) {
    std::vector<std::string> out(1);
    int depth = 0;
    for (char ch : s) {
        if (respectBrackets && (ch == '{' || ch == '(')) depth++;
        if (respectBrackets && (ch == '}' || ch == ')')) depth--;
        if (ch == sep && depth == 0) out.emplace_back();
        else out.back().push_back(ch);
    }
    return out;
}

}  // namespace

void Position::init_zobrist() {
    uint64_t seed = 0x52633020524556ULL;  // "Rc0 REV"
    for (auto& row : Z_PIECE)
        for (Key& k : row) k = next_random(seed);
    for (Key& k : Z_STM) k = next_random(seed);
    for (Key& k : Z_CASTLE_BIT) k = next_random(seed);
    for (auto& row : Z_EP)
        for (Key& k : row) k = next_random(seed);

    std::memset(CASTLE_KEEP, 0xFF, sizeof(CASTLE_KEEP));
    for (int c = 0; c < COLOR_NB; c++)
        for (int side = 0; side < 2; side++) {
            const geo::CastleInfo& ci = geo::CASTLE[c][side];
            CASTLE_KEEP[ci.kFrom] &= uint8_t(~(geo::castle_bit(Color(c), 0) | geo::castle_bit(Color(c), 1)));
            CASTLE_KEEP[ci.rFrom] &= uint8_t(~geo::castle_bit(Color(c), side));
        }
}

// ------------------------------------------------------------------- board edits ---
void Position::reset() {
    std::fill(std::begin(board_), std::end(board_), NO_PIECE);
    std::fill(std::begin(king_), std::end(king_), SQ_NONE);
    std::memset(count_, 0, sizeof(count_));
    stm_ = RED;
    ply_ = 0;
    states_.assign(1, State());
    states_.reserve(256);
    keys_.clear();
}

void Position::add(int sq, Piece p) {
    board_[sq] = p;
    count_[color_of(p)][type_of(p)]++;
    if (type_of(p) == KING) king_[color_of(p)] = sq;
}

void Position::remove(int sq) {
    const Piece p = board_[sq];
    board_[sq] = NO_PIECE;
    count_[color_of(p)][type_of(p)]--;
    if (type_of(p) == KING) king_[color_of(p)] = SQ_NONE;
}

void Position::relocate(int from, int to) {
    const Piece p = board_[from];
    board_[from] = NO_PIECE;
    board_[to] = p;
    if (type_of(p) == KING) king_[color_of(p)] = to;
}

void Position::finish_setup() {
    State& s = top();
    Key k = Z_STM[stm_] ^ castle_key(s.castling);
    for (int sq = 0; sq < SQUARE_NB; sq++)
        if (board_[sq] != NO_PIECE) k ^= Z_PIECE[board_[sq]][sq];
    for (int c = 0; c < COLOR_NB; c++)
        if (s.epSq[c] != SQ_NONE) k ^= Z_EP[c][s.epSq[c]];
    s.key = k;
}

// ------------------------------------------------------------------- start positions ---
namespace {
const PieceType BACK_RANK[8] = {ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK};
}

void Position::init_startpos() {
    reset();
    for (int c = 0; c < COLOR_NB; c++)
        for (int i = 0; i < 8; i++) {  // Red: d1..k1 = R N B Q K B N R, pawns on rank 2; others rotated
            add(geo::SEAT_MAP[c][geo::SQ[3 + i][0]], make_piece(Color(c), BACK_RANK[i]));
            add(geo::SEAT_MAP[c][geo::SQ[3 + i][1]], make_piece(Color(c), PAWN));
        }
    top().castling = 0xFF;
    finish_setup();
}

void Position::init_startpos960(uint64_t seed) {
    reset();
    PieceType back[8];
    std::copy(std::begin(BACK_RANK), std::end(BACK_RANK), back);
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;  // xorshift64, Fisher-Yates
    for (int i = 7; i > 0; i--) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        std::swap(back[i], back[x % uint64_t(i + 1)]);
    }
    for (int c = 0; c < COLOR_NB; c++)
        for (int i = 0; i < 8; i++) {
            add(geo::SEAT_MAP[c][geo::SQ[3 + i][0]], make_piece(Color(c), back[i]));
            add(geo::SEAT_MAP[c][geo::SQ[3 + i][1]], make_piece(Color(c), PAWN));
        }
    top().castling = 0;
    finish_setup();
}

// ------------------------------------------------------------------- attacks ---
bool Position::attacked_by_team(int sq, int team) const {
    if (sq == SQ_NONE) return false;
    for (int c = team; c < COLOR_NB; c += 2)
        for (const int16_t* p = geo::PAWN_ATTACK_FROM[c][sq]; *p != SQ_NONE; ++p)
            if (board_[*p] == make_piece(Color(c), PAWN)) return true;
    auto ours = [&](Piece p, PieceType t) { return p != NO_PIECE && type_of(p) == t && team_of(color_of(p)) == team; };
    for (const int16_t* p = geo::KNIGHT_TO[sq]; *p != SQ_NONE; ++p)
        if (ours(board_[*p], KNIGHT)) return true;
    for (const int16_t* p = geo::KING_TO[sq]; *p != SQ_NONE; ++p)
        if (ours(board_[*p], KING)) return true;
    for (int d = 0; d < geo::DIR_NB; d++) {
        const PieceType slider = d < 4 ? ROOK : BISHOP;
        for (const int16_t* p = geo::RAY[sq][d]; *p != SQ_NONE; ++p) {
            const Piece pc = board_[*p];
            if (pc == NO_PIECE) continue;
            if (ours(pc, QUEEN) || ours(pc, slider)) return true;
            break;
        }
    }
    return false;
}

// ------------------------------------------------------------------- make / unmake ---
void Position::do_move(Move m) {
    keys_.push_back(top().key);
    states_.push_back(top());
    State& s = top();
    s.captured = NO_PIECE;
    s.capturedSq = SQ_NONE;
    s.rule50++;

    const Color us = stm_;
    Key k = s.key;

    // A double-push en-passant chance lasts until that player moves again.
    if (s.epSq[us] != SQ_NONE) {
        k ^= Z_EP[us][s.epSq[us]];
        s.epSq[us] = s.epPawn[us] = SQ_NONE;
    }

    const int from = from_sq(m), to = to_sq(m);
    const MoveFlag flag = flag_of(m);

    if (flag == MF_CASTLE_K || flag == MF_CASTLE_Q) {
        const geo::CastleInfo& ci = geo::CASTLE[us][flag == MF_CASTLE_Q];
        const Piece king = board_[ci.kFrom], rook = board_[ci.rFrom];
        relocate(ci.kFrom, ci.kTo);
        relocate(ci.rFrom, ci.rTo);
        k ^= Z_PIECE[king][ci.kFrom] ^ Z_PIECE[king][ci.kTo] ^ Z_PIECE[rook][ci.rFrom] ^ Z_PIECE[rook][ci.rTo];
    } else {
        const Piece mover = board_[from];
        int capSq = to;
        if (flag == MF_EN_PASSANT) {
            for (int c = 0; c < COLOR_NB; c++)
                if (!same_team(Color(c), us) && s.epSq[c] == to) {
                    capSq = s.epPawn[c];
                    k ^= Z_EP[c][s.epSq[c]];
                    s.epSq[c] = s.epPawn[c] = SQ_NONE;
                    break;
                }
        }
        const Piece victim = board_[capSq];
        if (victim != NO_PIECE) {
            // Taking a pawn that could still be captured en passant ends that chance too.
            const Color vc = color_of(victim);
            if (flag != MF_EN_PASSANT && type_of(victim) == PAWN && s.epPawn[vc] == capSq) {
                k ^= Z_EP[vc][s.epSq[vc]];
                s.epSq[vc] = s.epPawn[vc] = SQ_NONE;
            }
            remove(capSq);
            k ^= Z_PIECE[victim][capSq];
            s.captured = victim;
            s.capturedSq = capSq;
            s.rule50 = 0;
        }
        relocate(from, to);
        k ^= Z_PIECE[mover][from] ^ Z_PIECE[mover][to];
        if (type_of(mover) == PAWN) {
            s.rule50 = 0;
            if (flag == MF_DOUBLE_PUSH) {
                s.epSq[us] = geo::PAWN_PUSH[us][from];
                s.epPawn[us] = to;
                k ^= Z_EP[us][s.epSq[us]];
            } else if (is_promotion(m)) {
                const Piece promoted = make_piece(us, promo_of(m));
                remove(to);
                add(to, promoted);
                k ^= Z_PIECE[mover][to] ^ Z_PIECE[promoted][to];
            }
        }
    }

    const uint8_t rights = s.castling & CASTLE_KEEP[from] & CASTLE_KEEP[to];
    if (rights != s.castling) {
        k ^= castle_key(s.castling) ^ castle_key(rights);
        s.castling = rights;
    }

    k ^= Z_STM[us] ^ Z_STM[next_color(us)];
    stm_ = next_color(us);
    ply_++;
    s.key = k;
}

void Position::undo_move(Move m) {
    const State& s = top();
    const Color us = prev_color(stm_);
    const int from = from_sq(m), to = to_sq(m);
    const MoveFlag flag = flag_of(m);

    if (flag == MF_CASTLE_K || flag == MF_CASTLE_Q) {
        const geo::CastleInfo& ci = geo::CASTLE[us][flag == MF_CASTLE_Q];
        relocate(ci.rTo, ci.rFrom);
        relocate(ci.kTo, ci.kFrom);
    } else {
        if (is_promotion(m)) {
            remove(to);
            add(to, make_piece(us, PAWN));
        }
        relocate(to, from);
        if (s.captured != NO_PIECE) add(s.capturedSq, s.captured);
    }
    states_.pop_back();
    keys_.pop_back();
    stm_ = us;
    ply_--;
}

int Position::repetition_count() const {
    const int n = int(keys_.size());
    const int window = std::min(top().rule50, n);
    int seen = 1;
    for (int i = 1; i <= window; i++)
        if (keys_[n - i] == top().key) seen++;
    return seen;
}

// ------------------------------------------------------------------- FEN4 ---
// chess.com-style FEN4: "R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-<board>". Fields: side to move,
// eliminated flags, kingside and queenside castling per seat, points, half-move clock, an
// optional {'enPassant':(...)} block, then 14 rows from rank 14 to rank 1 ("x" = corner).
std::string Position::fen4() const {
    const State& s = top();
    std::ostringstream o;
    o << COLOR_CHAR[stm_] << "-0,0,0,0-";
    for (int side = 0; side < 2; side++) {
        for (int c = 0; c < COLOR_NB; c++)
            o << ((s.castling & geo::castle_bit(Color(c), side)) ? 1 : 0) << (c < 3 ? "," : "");
        o << "-";
    }
    o << "0,0,0,0-" << s.rule50 << "-";

    if (std::any_of(std::begin(s.epSq), std::end(s.epSq), [](int e) { return e != SQ_NONE; })) {
        o << "{'enPassant':(";
        for (int c = 0; c < COLOR_NB; c++) {
            o << "'";
            if (s.epSq[c] != SQ_NONE) o << geo::sq_name(s.epSq[c]) << ":" << geo::sq_name(s.epPawn[c]);
            o << "'" << (c < 3 ? "," : "");
        }
        o << ")}-";
    }

    for (int r = RANK_NB - 1; r >= 0; r--) {
        std::vector<std::string> cells;  // one token per run / corner / piece
        int run = 0;
        for (int f = 0; f < FILE_NB; f++) {
            const int sq = geo::SQ[f][r];
            const bool emptyCell = sq != SQ_NONE && board_[sq] == NO_PIECE;
            if (emptyCell) { run++; continue; }
            if (run) { cells.push_back(std::to_string(run)); run = 0; }
            if (sq == SQ_NONE) cells.push_back("x");
            else cells.push_back(std::string(1, COLOR_LOWER[color_of(board_[sq])]) + PT_CHAR[type_of(board_[sq])]);
        }
        if (run) cells.push_back(std::to_string(run));
        for (size_t i = 0; i < cells.size(); i++) o << (i ? "," : "") << cells[i];
        if (r > 0) o << "/";
    }
    return o.str();
}

bool Position::set_fen4(const std::string& fen) {
    const std::vector<std::string> fields = split(fen, '-', true);
    if (fields.size() < 2 || fields[0].empty()) return false;
    reset();

    switch (std::toupper((unsigned char)fields[0][0])) {
    case 'R': stm_ = RED; break;
    case 'B': stm_ = BLUE; break;
    case 'Y': stm_ = YELLOW; break;
    case 'G': stm_ = GREEN; break;
    default: return false;
    }

    State& s = top();
    uint8_t castling = 0;
    for (size_t i = 1; i + 1 < fields.size(); i++) {
        const std::string& fd = fields[i];
        if (fd.empty()) continue;
        if (i == 2 || i == 3) {
            const std::vector<std::string> bits = split(fd, ',', false);
            for (int c = 0; c < COLOR_NB && c < int(bits.size()); c++)
                if (!bits[c].empty() && bits[c][0] == '1') castling |= uint8_t(geo::castle_bit(Color(c), i == 2 ? 0 : 1));
        } else if (fd[0] == '{') {
            const size_t key = fd.find("enPassant");
            const size_t open = key == std::string::npos ? key : fd.find('(', key);
            const size_t close = open == std::string::npos ? open : fd.find(')', open);
            if (close == std::string::npos) continue;
            const std::vector<std::string> per = split(fd.substr(open + 1, close - open - 1), ',', false);
            for (int c = 0; c < COLOR_NB && c < int(per.size()); c++) {
                std::string e;
                for (char ch : per[c])
                    if (ch != '\'' && ch != ' ') e.push_back(ch);
                const std::vector<std::string> two = split(e, ':', false);
                if (e.empty() || two.size() != 2) continue;
                const int skipped = geo::parse_sq(two[0].c_str()), pawn = geo::parse_sq(two[1].c_str());
                if (skipped != SQ_NONE && pawn != SQ_NONE) { s.epSq[c] = skipped; s.epPawn[c] = pawn; }
            }
        } else if (i >= 5 && fd.size() <= 4 && std::all_of(fd.begin(), fd.end(), [](char ch) { return std::isdigit((unsigned char)ch); })) {
            s.rule50 = std::atoi(fd.c_str());   // at most 4 digits: no overflow from hostile input
        }
    }

    const std::vector<std::string> rows = split(fields.back(), '/', false);
    if (int(rows.size()) != RANK_NB) return false;
    for (int i = 0; i < RANK_NB; i++) {
        const int r = RANK_NB - 1 - i;
        int f = 0;
        for (const std::string& raw : split(rows[i], ',', false)) {
            std::string tok;
            for (char ch : raw)
                if (!std::isspace((unsigned char)ch)) tok.push_back(ch);
            if (tok.empty()) continue;
            if (tok == "x" || tok == "X") { if (++f > FILE_NB) return false; continue; }
            if (std::isdigit((unsigned char)tok[0])) {   // a run of empty squares: 1..14, digits only
                if (tok.size() > 2 || !std::all_of(tok.begin(), tok.end(), [](char ch) { return std::isdigit((unsigned char)ch); }))
                    return false;
                const int run = std::atoi(tok.c_str());
                if (run < 1 || f + run > FILE_NB) return false;
                f += run;
                continue;
            }
            if (tok.size() < 2) continue;
            const int c = color_index(tok[0]), t = type_index(tok[1]);
            if (c < 0 || t < 0 || f >= FILE_NB || geo::SQ[f][r] == SQ_NONE) return false;
            add(geo::SQ[f][r], make_piece(Color(c), PieceType(t)));
            f++;
        }
        if (f != FILE_NB) return false;
    }
    for (int c = 0; c < COLOR_NB; c++)
        if (king_[c] == SQ_NONE || count_[c][KING] != 1) return false;   // exactly one king per army

    // Drop castling rights whose king or rook is not on its home square.
    for (int c = 0; c < COLOR_NB; c++)
        for (int side = 0; side < 2; side++) {
            const geo::CastleInfo& ci = geo::CASTLE[c][side];
            if (board_[ci.kFrom] != make_piece(Color(c), KING) || board_[ci.rFrom] != make_piece(Color(c), ROOK))
                castling &= uint8_t(~geo::castle_bit(Color(c), side));
        }
    s.castling = castling;
    finish_setup();
    return true;
}

// ------------------------------------------------------------------- move text ---
std::string Position::move_str(Move m) {
    if (m == MOVE_NONE) return "0000";
    std::string s = geo::sq_name(from_sq(m)) + geo::sq_name(to_sq(m));
    if (is_promotion(m)) s += char(std::tolower((unsigned char)PT_CHAR[promo_of(m)]));
    return s;
}

Move Position::parse_move(const std::string& text) {
    int lenFrom = 0, lenTo = 0;
    const int from = geo::parse_sq(text.c_str(), &lenFrom);
    if (from == SQ_NONE) return MOVE_NONE;
    const int to = geo::parse_sq(text.c_str() + lenFrom, &lenTo);
    if (to == SQ_NONE) return MOVE_NONE;
    PieceType promo = PAWN;
    size_t i = size_t(lenFrom + lenTo);
    if (i < text.size() && text[i] == '=') i++;
    if (i < text.size()) {
        const int t = type_index(text[i]);
        if (t > PAWN && t < KING) promo = PieceType(t);
    }
    Move list[MAX_MOVES];
    const int n = generate_legal(*this, list);
    for (int j = 0; j < n; j++)
        if (from_sq(list[j]) == from && to_sq(list[j]) == to && promo_of(list[j]) == promo) return list[j];
    return MOVE_NONE;
}

std::string Position::pretty() const {
    std::ostringstream o;
    for (int r = RANK_NB - 1; r >= 0; r--) {
        o << (r + 1 < 10 ? " " : "") << r + 1 << " ";
        for (int f = 0; f < FILE_NB; f++) {
            const int sq = geo::SQ[f][r];
            if (sq == SQ_NONE) o << "   ";
            else if (board_[sq] == NO_PIECE) o << " . ";
            else o << COLOR_LOWER[color_of(board_[sq])] << PT_CHAR[type_of(board_[sq])] << " ";
        }
        o << "\n";
    }
    o << "   ";
    for (int f = 0; f < FILE_NB; f++) o << " " << char('a' + f) << " ";
    o << "\n" << COLOR_CHAR[stm_] << " to move\n";
    return o.str();
}

}  // namespace quad
