// SPDX-License-Identifier: GPL-3.0-or-later
// libFuzzer target: `rc0 validate` on hostile uploads. One input = one upload directory, packed as
// [u8 flags] then records of [u8 file index][u32 length][bytes] (tests/fuzz/pack.py builds seeds from real output).
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>

#include "geometry.h"
#include "position.h"
#include "selfplay.h"

using namespace quad;

static const char* NAMES[] = {"games.txt", "sp_rec.bin", "sp_pol.bin", "sp_res.bin", "sp_q.bin",
                              "sp_ply.bin", "sp_mat.bin", "sp_hist.bin", "sp_aux.bin"};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    static std::string dir = [] {
        geo::init();
        Position::init_zobrist();
        const char* t = std::getenv("FUZZ_DIR");
        std::string d = std::string(t ? t : ".") + "/fuzz_validate_" + std::to_string(getpid());
        std::filesystem::create_directories(d);
        if (!std::freopen("/dev/null", "w", stdout)) std::abort();   // the validator's verdict lines
        return d;
    }();
    for (const char* n : NAMES) std::remove((dir + "/" + n).c_str());
    bool gamesOnly = size > 0 && (data[0] & 1);
    size_t i = 1;
    while (i + 5 <= size) {
        uint8_t idx = data[i] % 9;
        uint32_t len;
        std::memcpy(&len, data + i + 1, 4);
        i += 5;
        if (len > size - i) len = uint32_t(size - i);
        FILE* f = std::fopen((dir + "/" + NAMES[idx]).c_str(), "ab");
        std::fwrite(data + i, 1, len, f);
        std::fclose(f);
        i += len;
    }
    zero::validate_main(dir, false, gamesOnly);
    return 0;
}
