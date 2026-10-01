// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
#include <string>
namespace zero {
struct SelfplayOpts {
    std::string net, out;
    int games = 100, threads = 32, playouts = 128, maxPlies = 600, tempPlies = 20, randomPlies = 8, c960pct = 20;
    int batchTarget = 128, maxBatch = 256, cacheLog2 = 18;
    double flushMs = 2.0, seconds = 0.0;
    uint64_t seed = 1;
    bool noise = true;
    int noisePlies = 20;         // Dirichlet root noise for the first N plies (the spec: noise + temperature for 20 plies)
    int gumbel = 16;             // >0: Gumbel root search with Top-m = gumbel (policy target = completed-Q policy); 0 = PUCT + Dirichlet
    int fastPlayouts = 16;       // playout-cap randomization (KataGo): most moves are cheap and NOT recorded ...
    int fullPct = 25;            // ... this % of moves get the full `playouts` budget and ARE recorded as training rows
    float adjThr = 0.95f;        // adjudicate when |root value| (RY view, sign-consistent) >= thr for adjPlies plies, after adjMinPly
    int adjPlies = 8, adjMinPly = 40;
    int verifyPct = 10;          // this % of adjudicated games are played out anyway, to measure how often adjudication is wrong
    float forcedPlayouts = 0.0f; // the 2.0 in KataGo's n_forced = sqrt(2 * P * N); 0 = off. SELF-PLAY ONLY
};
int selfplay_main(const SelfplayOpts& o);
}
