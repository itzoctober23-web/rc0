// SPDX-License-Identifier: GPL-3.0-or-later
// libFuzzer target: untrusted FEN4 and move strings (both reach the server through uploads).
//   clang++ -fsanitize=fuzzer,address,undefined ...   (see tests/fuzz/run.sh)
#include <cstddef>
#include <cstdint>
#include <string>

#include "geometry.h"
#include "movegen.h"
#include "position.h"

using namespace quad;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    static bool init = [] { geo::init(); Position::init_zobrist(); return true; }();
    (void)init;
    std::string in(reinterpret_cast<const char*>(data), size);
    size_t tab = in.find('\t');
    std::string fen = in.substr(0, tab), moves = tab == std::string::npos ? "" : in.substr(tab + 1);
    Position p;
    if (!p.set_fen4(fen)) return 0;
    (void)p.fen4();
    (void)p.pretty();
    Move list[MAX_MOVES];
    int n = generate_legal(p, list);
    for (int i = 0; i < n; i++) { p.do_move(list[i]); p.undo_move(list[i]); }
    bool over = false;
    game_result(p, over);
    // play the move list (whatever it is) through the parser, as the validator does
    size_t pos = 0;
    for (int k = 0; k < 64 && pos < moves.size(); k++) {
        size_t sp = moves.find(' ', pos);
        std::string m = moves.substr(pos, sp == std::string::npos ? std::string::npos : sp - pos);
        pos = sp == std::string::npos ? moves.size() : sp + 1;
        game_result(p, over);
        if (over) break;
        Move mv = p.parse_move(m);
        if (mv == MOVE_NONE) break;
        p.do_move(mv);
    }
    return 0;
}
