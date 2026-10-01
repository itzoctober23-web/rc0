// SPDX-License-Identifier: GPL-3.0-or-later
// PUCT Monte-Carlo tree search for 4PC Teams.
//
// Values are always "from the perspective of the TEAM to move at that node".
// Backup flips sign by explicit team lookup (same_team), never by ply parity.
// Terminals (any king captured, checkmate of the side to move, stalemate,
// rule-50, threefold) are detected BEFORE expansion via quad::game_result.
// No hand-coded chess knowledge inside the tree (design rule): the only proven
// values are TRUE terminals from game_result, like Lc0. The 1-ply king-safety pre-resolution
// and the root mate-in-1 filter of the warm-start line were removed.
// Leaves are gathered K at a time with virtual loss and evaluated as a batch.
#pragma once
#include <cstdint>
#include <memory>
#include <random>
#include <vector>
#include "evaluator.h"
#include "position.h"

namespace zero {

struct SearchParams {
    float cpuct = 1.75f;
    float cpuctBase = 38739.0f;   // Lc0-style log growth
    float cpuctFactor = 3.894f;
    float fpuReduction = 0.33f;   // FPU = parentQ - fpu * sqrt(sum visited priors)
    float fpuRootReduction = 0.0f;
    float policyTemp = 1.0f;      // softmax temperature applied to priors (1 = as returned)
    int gatherK = 16;             // leaves per batch
    float virtualLoss = 1.0f;
    bool rootNoise = false;       // Dirichlet at root (self-play)
    // , SELF-PLAY ONLY. n_forced(i) = sqrt(2 * P_i * N): any root move
    // whose visit count is below that is topped up after Sequential Halving, so a move with real prior
    // mass is never left on the vmix fallback. Default 0 = off, so the UCI engine and the ladder are
    // untouched unless asked.
    float forcedPlayouts = 0.0f;  // the 2.0 in sqrt(2 * P * N); 0 disables
    float dirichletAlpha = 0.3f;  // scaled by 64/legal-moves inside
    float dirichletEps = 0.25f;
    uint64_t seed = 0;
};

struct Edge {
    quad::Move move;
    float prior;
    uint32_t child;  // 0 = not yet created
};

struct Node {
    uint32_t n = 0;          // completed visits
    uint32_t vl = 0;         // virtual loss in flight
    float w = 0.0f;          // sum of values, team-to-move-at-node perspective
    float mlSum = 0.0f;      // moves-left sum
    quad::Color stm;
    bool expanded = false;   // priors filled, children selectable
    bool claimed = false;    // gathered into a batch, awaiting evaluation
    bool terminal = false;
    float terminalValue = 0.0f;  // if terminal
    // CERTAINTY PROPAGATION . A terminal node is a PROVEN result from the
    // rules, not a net estimate, and proof propagates: any child proven a win for the team to move here
    // makes this node a proven win; all children proven losses make it a proven loss. Sign convention is
    // the same as w/terminalValue - the team to move AT THIS NODE - and the frame change between parent
    // and child always goes through same_team(), never ply parity.
    enum : int8_t { PV_NONE = 0, PV_WIN = 1, PV_LOSS = -1, PV_DRAW = 2 };
    int8_t proven = PV_NONE;
    bool certain() const { return terminal || proven != PV_NONE; }
    float certain_value() const {
        if (terminal) return terminalValue;
        return proven == PV_WIN ? 1.0f : proven == PV_LOSS ? -1.0f : 0.0f;
    }
    std::vector<Edge> edges;
    // input history: the 136-byte record of the position AT this node, written once (when the node is first
    // reached as a leaf) so a deeper leaf can read its ancestors' positions without re-deriving them.
    unsigned char rec[REC_BYTES] = {0};
    bool hasRec = false;
    float q() const { return n ? w / float(n) : 0.0f; }
};

struct RootStats {
    std::vector<quad::Move> moves;
    std::vector<uint32_t> visits;
    std::vector<float> q;         // child value from the ROOT team's perspective
    std::vector<float> prior;
    float rootQ = 0.0f;
    uint64_t nodes = 0;
    int maxDepth = 0;
};

class Mcts {
public:
    Mcts(Evaluator& ev, const SearchParams& p);
    // Set the root position (copied; the game history in pos is kept so
    // repetition detection sees it).
    void set_root(const quad::Position& pos);
    // Run `playouts` more simulations (each = one leaf evaluation, terminals count).
    void run(int playouts);
    // Run until stop_fn() returns true or maxPlayouts reached; checks every batch.
    template <class F> void run_until(F stop_fn, int maxPlayouts) {
        while (visits_done_ < uint64_t(maxPlayouts) && !stop_fn()) step();
    }
    RootStats root_stats() const;
    // Visit-count move selection with temperature (0 = argmax).
    quad::Move select_move(float temperature);
    // Gumbel AlphaZero root search (Danihelka et al. 2022): Gumbel-Top-m over the root priors,
    // Sequential Halving over `playouts` simulations, PUCT inside the tree. Returns the chosen
    // move and fills `policyTarget` (aligned with root_stats().moves) with the completed-Q
    // improved policy  softmax(logit + sigma(completedQ)), the self-play training target.
    quad::Move gumbel_search(int playouts, int m, std::vector<float>& policyTarget);
    // Advance the root by a move, keeping the subtree (tree reuse).
    void apply_move(quad::Move m);
    uint64_t nodes() const { return nodes_.size(); }
    struct Stats { uint64_t batches = 0, leaves = 0, collisions = 0, terminals = 0; };
    const Stats& stats() const { return stats_; }
    int max_depth() const { return maxDepth_; }
    // certainty propagation: PV_NONE / PV_WIN / PV_LOSS / PV_DRAW for the root, for the self-play
    // value target (certainty propagation: "training targets for proven positions use the proven value").
    int root_proven() const { return nodes_.empty() ? 0 : int(nodes_[root_].proven); }
    float root_proven_value() const { return nodes_[root_].certain_value(); }
    uint64_t playouts() const { return visits_done_; }
    bool root_terminal() const { return nodes_[root_].terminal; }
    void set_root_noise(bool on) { p_.rootNoise = on; }   // self-play: Dirichlet on the root priors for the opening plies
    // input history: the positions BEFORE the root, newest first, as fill_record() writes them. n may be 0..HIST_POS.
    // Self-play passes the game's own ring; the UCI engine derives it from "position ... moves".
    void set_root_history(const unsigned char* h, int n);

private:
    void step();  // one gather-K batch
    struct PathItem { uint32_t node; int edge; };
    // Select from root to a leaf, applying virtual loss; returns leaf node id and
    // leaves pos_ AT the leaf (caller undoes path.size() moves).
    uint32_t select(std::vector<PathItem>& path);
    void expand_terminal_or_prepare(uint32_t nid, const std::vector<PathItem>& path, EvalItem& item);
    void backup(const std::vector<PathItem>& path, uint32_t leaf, float leafValue, float ml);
    float child_q_for_parent(const Node& parent, const Node& child) const;
    // child's proven state expressed in the PARENT's frame (PV_NONE if the child proves nothing)
    int8_t proven_for_parent(const Node& parent, const Node& child) const;
    // recompute `proven` up the path after a backup; stops at the first node that is still unknown
    void propagate_proven(const std::vector<PathItem>& path);
    // One-ply RULES scan of the root's moves for an immediate proven win, creating and marking the
    // child so propagation sees it. Needed because Gumbel-Top-m samples only m root moves, so a
    // mate-in-1 outside the sample was never visited and never proved (measured: 3 of 4 seats).
    // Only terminal detection is used, which is the one thing the no-hand-coded-knowledge rule allows to be hard-coded.
    int root_proven_win_edge();
    uint32_t new_node(quad::Color stm);
    float cpuct_for(const Node& n) const;
    void add_root_noise();
    void ensure_root_expanded();
    void run_forced(int edge, int visits);   // simulations that all start through root edge `edge`
    int forcedRootEdge_ = -1;

    unsigned char rootHist_[HIST_POS * REC_BYTES] = {0};
    int rootHistN_ = 0;

    Evaluator& ev_;
    SearchParams p_;
    std::vector<Node> nodes_;
    std::vector<std::vector<PathItem>> collisions_;
    Stats stats_;  // paths whose virtual loss is released after the batch
    uint32_t root_ = 0;
    quad::Position pos_;
    std::mt19937_64 rng_;
    uint64_t visits_done_ = 0;
    int maxDepth_ = 0;
};

// value <-> centipawn display mapping (Lc0-style)
inline int value_to_cp(float v) {
    if (v > 0.9999f) v = 0.9999f;
    if (v < -0.9999f) v = -0.9999f;
    return int(290.68f * std::tan(1.5637f * v));
}

}  // namespace zero
