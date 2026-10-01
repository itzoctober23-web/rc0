// SPDX-License-Identifier: GPL-3.0-or-later
#include "mcts.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include "movegen.h"

namespace zero {
using namespace quad;

Mcts::Mcts(Evaluator& ev, const SearchParams& p)
    : ev_(ev), p_(p), rng_(p.seed ? p.seed : 0x9E3779B97F4A7C15ULL) {
    nodes_.reserve(1 << 16);
}

uint32_t Mcts::new_node(Color stm) {
    nodes_.emplace_back();
    nodes_.back().stm = stm;
    return uint32_t(nodes_.size() - 1);
}

void Mcts::set_root(const Position& pos) {
    nodes_.clear();
    nodes_.emplace_back();  // index 0 = unused sentinel so child==0 means "none"
    pos_ = pos;
    root_ = new_node(pos_.side_to_move());
    fill_record(pos_, nodes_[root_].rec);   // input history: the root's own position
    nodes_[root_].hasRec = true;
    visits_done_ = 0;
    maxDepth_ = 0;
}

void Mcts::set_root_history(const unsigned char* h, int n) {
    rootHistN_ = n < 0 ? 0 : (n > HIST_POS ? HIST_POS : n);
    if (rootHistN_ && h) std::memcpy(rootHist_, h, size_t(rootHistN_) * REC_BYTES);
}

float Mcts::cpuct_for(const Node& n) const {
    return p_.cpuct + p_.cpuctFactor * std::log((float(n.n) + p_.cpuctBase) / p_.cpuctBase);
}

// Q of `child` as seen by `parent`'s team (explicit team lookup, never ply parity).
int8_t Mcts::proven_for_parent(const Node& parent, const Node& child) const {
    if (child.proven == Node::PV_NONE) return Node::PV_NONE;
    if (child.proven == Node::PV_DRAW) return Node::PV_DRAW;          // a draw is a draw in every frame
    return same_team(parent.stm, child.stm) ? child.proven : int8_t(-child.proven);
}

int Mcts::root_proven_win_edge() {
    // NO references into nodes_ held across new_node(): nodes_ is a vector and new_node() can
    // REALLOCATE it. My first version kept `Node& r` and `Edge& e` across the call, so the
    // pos_.undo_move(e.move) afterwards read freed memory and could undo a garbage move, corrupting
    // the position for the rest of the search. The codebase already had the pattern I ignored -
    // mcts.cpp's step() re-fetches with "expansion may have reallocated nodes_". A a fixed bot rung that
    // came back 16/40 against 8/40 (z=1.95) is what sent me looking; it is NOT a measurement of the
    // bug's cost, because that rung turned out to have huge variance - the same binary on the same
    // openings ranged 0.350-0.450, and on a different opening seed gave 0.100. The fix stands on the
    // undefined behaviour itself, not on that number.
    for (size_t i = 0; i < nodes_[root_].edges.size(); i++) {
        const uint32_t existing = nodes_[root_].edges[i].child;
        if (existing) {   // already in the tree: use what is known, do not re-walk the rules
            if (proven_for_parent(nodes_[root_], nodes_[existing]) == Node::PV_WIN) return int(i);
            continue;
        }
        const quad::Move mv = nodes_[root_].edges[i].move;   // BY VALUE, before anything can reallocate
        pos_.do_move(mv);
        bool over = false;
        GameResult gr = game_result(pos_, over);
        if (!over) { pos_.undo_move(mv); continue; }
        const quad::Color cstm = pos_.side_to_move();
        const float tv = (gr == DRAW_RESULT) ? 0.0f
                       : (team_of(cstm) == ((gr == TEAM_RY_WINS) ? 0 : 1)) ? 1.0f : -1.0f;
        pos_.undo_move(mv);                                  // undo BEFORE allocating, so no reference is live
        uint32_t cid = new_node(cstm);
        {
            Node& c = nodes_[cid];
            c.terminal = true;
            c.expanded = true;
            c.terminalValue = tv;
            c.proven = tv > 0.5f ? Node::PV_WIN : tv < -0.5f ? Node::PV_LOSS : Node::PV_DRAW;
        }
        nodes_[root_].edges[i].child = cid;
        if (proven_for_parent(nodes_[root_], nodes_[cid]) == Node::PV_WIN) {
            nodes_[root_].proven = Node::PV_WIN;
            return int(i);
        }
    }
    return -1;
}

void Mcts::propagate_proven(const std::vector<PathItem>& path) {
    for (int i = int(path.size()) - 1; i >= 0; i--) {
        Node& node = nodes_[path[i].node];
        if (node.proven != Node::PV_NONE || node.edges.empty()) break;  // already settled, or nothing to prove from
        bool allProven = true, anyDraw = false, anyWin = false;
        for (const Edge& e : node.edges) {
            if (!e.child) { allProven = false; continue; }
            int8_t pv = proven_for_parent(node, nodes_[e.child]);
            if (pv == Node::PV_WIN) { anyWin = true; break; }
            if (pv == Node::PV_NONE) allProven = false;
            else if (pv == Node::PV_DRAW) anyDraw = true;
        }
        if (anyWin) node.proven = Node::PV_WIN;
        else if (allProven) node.proven = anyDraw ? Node::PV_DRAW : Node::PV_LOSS;
        else break;   // unknown here, so no ancestor can newly become proven THROUGH this path
    }
}

float Mcts::child_q_for_parent(const Node& parent, const Node& child) const {
    float sign = same_team(parent.stm, child.stm) ? 1.0f : -1.0f;
    float n = float(child.n) + float(child.vl);
    if (n <= 0.0f) return 0.0f;
    // virtual loss: each in-flight visit counts as a loss for the parent
    float w = sign * child.w - float(child.vl) * p_.virtualLoss;
    return w / n;
}

uint32_t Mcts::select(std::vector<PathItem>& path) {
    path.clear();
    uint32_t nid = root_;
    int depth = 0;
    while (true) {
        Node& node = nodes_[nid];
        node.vl++;
        // A certain node is a leaf - except the ROOT. Returning the root here made every playout after
        // the proof a no-op that only bumped the root's own counters: the king-capture test went to
        // "visits 1 of 401" because the tree was starved. At the root we keep descending, and the
        // proven-win shortcut below sends every playout down the winning line, which costs no eval.
        if (node.terminal || (nid != root_ && node.proven != Node::PV_NONE) || !node.expanded) return nid;
        // PUCT over edges
        float sumP = 0.0f;
        for (const Edge& e : node.edges)
            if (e.child && (nodes_[e.child].n + nodes_[e.child].vl) > 0) sumP += e.prior;
        float fpuRed = (nid == root_) ? p_.fpuRootReduction : p_.fpuReduction;
        float nTot = float(node.n) + float(node.vl - 1);
        float parentQ = node.n ? node.w / float(node.n) : 0.0f;
        float fpu = parentQ - fpuRed * std::sqrt(sumP);
        float c = cpuct_for(node) * std::sqrt(std::max(nTot, 1.0f));
        int best = -1;
        float bestScore = -1e30f;
        // CERTAINTY: a child proven to win for the team to move here ends the choice - play it, do not
        // spend playouts comparing it with estimates. And never enter a proven-loss child while a child
        // that is not yet proven lost exists.
        int provenWin = -1, nonLosing = 0;
        for (size_t i = 0; i < node.edges.size(); i++) {
            const Edge& e = node.edges[i];
            if (!e.child) { nonLosing++; continue; }
            int8_t pv = proven_for_parent(node, nodes_[e.child]);
            if (pv == Node::PV_WIN) { provenWin = int(i); break; }
            if (pv != Node::PV_LOSS) nonLosing++;
        }
        if (nid == root_ && forcedRootEdge_ >= 0 && forcedRootEdge_ < int(node.edges.size())) best = forcedRootEdge_;
        else if (provenWin >= 0) best = provenWin;
        else
        for (size_t i = 0; i < node.edges.size(); i++) {
            const Edge& e = node.edges[i];
            if (nonLosing > 0 && e.child && proven_for_parent(node, nodes_[e.child]) == Node::PV_LOSS) continue;
            float q, nch;
            if (e.child && (nodes_[e.child].n + nodes_[e.child].vl) > 0) {
                q = child_q_for_parent(node, nodes_[e.child]);
                nch = float(nodes_[e.child].n + nodes_[e.child].vl);
            } else {
                q = fpu;
                nch = 0.0f;
            }
            float s = q + c * e.prior / (1.0f + nch);
            if (s > bestScore) { bestScore = s; best = int(i); }
        }
        Edge& e = node.edges[best];
        pos_.do_move(e.move);
        if (!e.child) e.child = new_node(pos_.side_to_move());
        path.push_back({nid, best});
        nid = e.child;
        depth++;
        if (depth > maxDepth_) maxDepth_ = depth;
    }
}

// Mark terminal if the game is over at the leaf; otherwise fill the eval item
// (planes + legal slots) so the batch can evaluate it.
void Mcts::expand_terminal_or_prepare(uint32_t nid, const std::vector<PathItem>& path, EvalItem& item) {
    Node& node = nodes_[nid];
    bool over = false;
    GameResult gr = game_result(pos_, over);
    if (over) {
        node.terminal = true;
        node.expanded = true;
        if (gr == DRAW_RESULT) node.terminalValue = 0.0f;
        else {
            int winner = (gr == TEAM_RY_WINS) ? 0 : 1;
            node.terminalValue = (team_of(node.stm) == winner) ? 1.0f : -1.0f;
        }
        // the rules decided this, so it is PROVEN, not estimated
        node.proven = node.terminalValue > 0.5f ? Node::PV_WIN
                    : node.terminalValue < -0.5f ? Node::PV_LOSS : Node::PV_DRAW;
        return;
    }
    item.key = pos_.key() ^ (uint64_t(pos_.side_to_move()) * 0x9E3779B97F4A7C15ULL);
    item.planes.resize(INPUT_FLOATS);
    // input history: history = the ancestors on this path (newest first), then the pre-root game history.
    // Every ancestor has a record because every node is first reached as a leaf and recorded here.
    fill_record(pos_, node.rec);
    node.hasRec = true;
    unsigned char hist[HIST_POS * REC_BYTES];
    int nh = 0;
    for (int i = int(path.size()) - 1; i >= 0 && nh < HIST_POS; i--) {
        const Node& an = nodes_[path[i].node];
        if (!an.hasRec) break;
        std::memcpy(hist + nh * REC_BYTES, an.rec, REC_BYTES);
        nh++;
    }
    for (int i = 0; i < rootHistN_ && nh < HIST_POS; i++, nh++)
        std::memcpy(hist + nh * REC_BYTES, rootHist_ + i * REC_BYTES, REC_BYTES);
    encode(pos_, hist, nh, item.planes.data());
    LegalSet ls = legal_with_index(pos_);
    item.policyIdx = ls.idx;
    node.edges.clear();
    node.edges.reserve(ls.moves.size());
    for (Move m : ls.moves) node.edges.push_back({m, 0.0f, 0});
}

void Mcts::backup(const std::vector<PathItem>& path, uint32_t leaf, float leafValue, float ml) {
    Node& ln = nodes_[leaf];
    ln.n++; ln.vl--; ln.w += leafValue; ln.mlSum += ml;
    Color leafStm = ln.stm;
    for (int i = int(path.size()) - 1; i >= 0; i--) {
        Node& node = nodes_[path[i].node];
        float v = same_team(node.stm, leafStm) ? leafValue : -leafValue;
        node.n++; node.vl--; node.w += v; node.mlSum += ml + float(path.size() - i);
    }
}

void Mcts::add_root_noise() {
    Node& r = nodes_[root_];
    if (r.edges.empty()) return;
    float alpha = p_.dirichletAlpha * 64.0f / float(r.edges.size());
    std::gamma_distribution<float> g(alpha, 1.0f);
    std::vector<float> d(r.edges.size());
    float s = 0;
    for (float& x : d) { x = g(rng_); s += x; }
    for (size_t i = 0; i < d.size(); i++)
        r.edges[i].prior = (1.0f - p_.dirichletEps) * r.edges[i].prior + p_.dirichletEps * d[i] / s;
}

void Mcts::step() {
    std::vector<std::vector<PathItem>> paths;
    std::vector<uint32_t> leaves;
    std::vector<std::unique_ptr<EvalItem>> items;
    std::vector<EvalItem*> batch;
    std::vector<uint32_t> batchLeaf;
    std::vector<std::vector<PathItem>> batchPath;
    int K = std::max(1, p_.gatherK);
    int collisions = 0;
    collisions_.clear();
    for (int k = 0; k < K; k++) {
        std::vector<PathItem> path;
        uint32_t leaf = select(path);
        Node& ln = nodes_[leaf];
        if (ln.certain()) {
            float v = ln.certain_value();
            for (int i = int(path.size()) - 1; i >= 0; i--) pos_.undo_move(nodes_[path[i].node].edges[path[i].edge].move);
            backup(path, leaf, v, 0.0f);
            propagate_proven(path);
            visits_done_++;
            continue;
        }
        if (ln.claimed) {
            // collision: a leaf already claimed by this batch (not yet evaluated).
            // Leave the virtual loss in place so the next selection is steered
            // elsewhere, and keep gathering up to a bounded number of collisions.
            for (int i = int(path.size()) - 1; i >= 0; i--) pos_.undo_move(nodes_[path[i].node].edges[path[i].edge].move);
            collisions_.push_back(path);
            stats_.collisions++;
            if (++collisions >= K) break;
            continue;
        }
        auto item = std::make_unique<EvalItem>();
        expand_terminal_or_prepare(leaf, path, *item);
        Node& ln2 = nodes_[leaf];  // re-fetch: expansion may have reallocated nodes_
        for (int i = int(path.size()) - 1; i >= 0; i--) pos_.undo_move(nodes_[path[i].node].edges[path[i].edge].move);
        if (ln2.terminal) {
            backup(path, leaf, ln2.terminalValue, 0.0f);
            propagate_proven(path);
            visits_done_++;
            stats_.terminals++;
            continue;
        }
        ln2.claimed = true;  // priors arrive after the batch eval
        batch.push_back(item.get());
        batchLeaf.push_back(leaf);
        batchPath.push_back(std::move(path));
        items.push_back(std::move(item));
    }
    auto release_collisions = [&]() {
        for (auto& path : collisions_) {
            if (path.empty()) { nodes_[root_].vl--; continue; }
            uint32_t leaf = nodes_[path.back().node].edges[path.back().edge].child;
            nodes_[leaf].vl--;
            for (auto& it : path) nodes_[it.node].vl--;
        }
        collisions_.clear();
    };
    if (batch.empty()) { release_collisions(); return; }
    ev_.evaluate(batch);
    stats_.batches++;
    stats_.leaves += batch.size();
    release_collisions();
    for (size_t b = 0; b < batch.size(); b++) {
        Node& ln = nodes_[batchLeaf[b]];
        const EvalItem& it = *batch[b];
        // priors with optional temperature
        float s = 0.0f;
        for (size_t i = 0; i < ln.edges.size(); i++) {
            float pr = it.priors[i];
            if (p_.policyTemp != 1.0f) pr = std::pow(std::max(pr, 1e-12f), 1.0f / p_.policyTemp);
            ln.edges[i].prior = pr;
            s += pr;
        }
        if (s > 0) for (Edge& e : ln.edges) e.prior /= s;
        ln.expanded = true;
        ln.claimed = false;
        if (batchLeaf[b] == root_ && p_.rootNoise) add_root_noise();
        backup(batchPath[b], batchLeaf[b], it.value, it.movesLeft);
        propagate_proven(batchPath[b]);
        visits_done_++;
    }
}

void Mcts::run(int playouts) {
    uint64_t target = visits_done_ + uint64_t(playouts);
    while (visits_done_ < target) {
        uint64_t before = visits_done_;
        step();
        if (visits_done_ == before) break;  // root terminal or nothing to do
        if (nodes_[root_].terminal) break;
    }
}

void Mcts::ensure_root_expanded() {
    int guard = 0;
    while (!nodes_[root_].expanded && !nodes_[root_].terminal && guard++ < 64) step();
}

void Mcts::run_forced(int edge, int visits) {
    Node& r = nodes_[root_];
    if (edge < 0 || edge >= int(r.edges.size())) return;
    forcedRootEdge_ = edge;
    uint32_t target = (r.edges[edge].child ? nodes_[r.edges[edge].child].n : 0) + uint32_t(visits);
    int savedK = p_.gatherK;
    int guard = 0;
    while (guard++ < visits * 4 + 8) {
        uint32_t c = nodes_[root_].edges[edge].child;
        uint32_t have = c ? nodes_[c].n : 0;
        if (have >= target) break;
        if (c && nodes_[c].terminal) break;
        p_.gatherK = std::max(1, std::min(savedK, int(target - have)));  // never overshoot the halving budget
        uint64_t before = visits_done_;
        step();
        if (visits_done_ == before) break;
    }
    p_.gatherK = savedK;
    forcedRootEdge_ = -1;
}

Move Mcts::gumbel_search(int playouts, int m, std::vector<float>& policyTarget) {
    policyTarget.clear();
    ensure_root_expanded();
    Node& r0 = nodes_[root_];
    size_t E = r0.edges.size();
    if (E == 0 || r0.terminal) return MOVE_NONE;
    if (E == 1) { policyTarget.assign(1, 1.0f); return r0.edges[0].move; }
    std::vector<float> logit(E), g(E);
    std::uniform_real_distribution<float> u(1e-7f, 1.0f - 1e-7f);
    for (size_t i = 0; i < E; i++) {
        logit[i] = std::log(std::max(r0.edges[i].prior, 1e-9f));
        g[i] = -std::log(-std::log(u(rng_)));
    }
    auto qroot = [&](size_t i) -> float {
        const Node& r = nodes_[root_];
        uint32_t c = r.edges[i].child;
        return (c && nodes_[c].n) ? child_q_for_parent(r, nodes_[c]) : 0.0f;
    };
    auto visited = [&](size_t i) { uint32_t c = nodes_[root_].edges[i].child; return c && nodes_[c].n > 0; };
    uint32_t sigmaMaxN = 0;        // 0 = use the live max; set before the target so forced top-ups cannot inflate the scale
    auto sigma = [&](float q) {
        uint32_t maxN = sigmaMaxN;
        if (!maxN)
            for (size_t i = 0; i < E; i++) { uint32_t c = nodes_[root_].edges[i].child; if (c) maxN = std::max(maxN, nodes_[c].n); }
        return (50.0f + float(maxN)) * 1.0f * q;   // c_visit = 50, c_scale = 1 (paper defaults)
    };
    // Gumbel-Top-m
    std::vector<size_t> order(E);
    for (size_t i = 0; i < E; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return logit[a] + g[a] > logit[b] + g[b]; });
    int mm = std::min<int>(m, int(E));
    std::vector<size_t> rem(order.begin(), order.begin() + mm);
    // Sequential Halving
    int phases = std::max(1, int(std::ceil(std::log2(double(mm)))));
    int budget = std::max(playouts, mm);
    while (rem.size() > 1) {
        int per = std::max(1, budget / (phases * int(rem.size())));
        for (size_t a : rem) run_forced(int(a), per);
        std::sort(rem.begin(), rem.end(), [&](size_t a, size_t b) {
            return g[a] + logit[a] + sigma(qroot(a)) > g[b] + logit[b] + sigma(qroot(b));
        });
        rem.resize((rem.size() + 1) / 2);
        if (rem.size() == 1) break;
    }
    size_t best = rem[0];
    // , KataGo forced playouts, adapted to this search and the adaptation stated rather than hidden.
    // KataGo applies n_forced DURING a PUCT descent because PUCT under-visits low-prior moves. This search
    // does not use PUCT at the root at all: Sequential Halving hands every surviving Gumbel-Top-m child the
    // SAME budget, so "under-visited by PUCT" has no meaning here. What forced playouts are FOR - never
    // leaving a move with real prior mass unexamined - still applies, so n_forced is applied as a top-up
    // after halving, over ALL root moves including those Top-m eliminated.
    //   The policy-target pruning half of the order is UNNECESSARY HERE, and not skipped out of laziness:
    // KataGo prunes because its target is the VISIT DISTRIBUTION, so a forced visit directly teaches the net
    // to like that move. This target is completed-Q, softmax(logit + sigma(completedQ)) - it never reads a
    // visit count, so a forced visit can only improve a Q ESTIMATE, which is the point. The one place visits
    // do leak in is sigma's scale, which uses max child visits; so sigma is computed from the maxN captured
    // BEFORE the top-up. That is the faithful analogue of pruning.
    uint32_t maxNbeforeForced = 0;
    for (size_t i = 0; i < E; i++) { uint32_t c = nodes_[root_].edges[i].child; if (c) maxNbeforeForced = std::max(maxNbeforeForced, nodes_[c].n); }
    if (p_.forcedPlayouts > 0.0f) {
        uint32_t N = 0;
        for (size_t i = 0; i < E; i++) { uint32_t c = nodes_[root_].edges[i].child; if (c) N += nodes_[c].n; }
        for (size_t i = 0; i < E; i++) {
            float pr = nodes_[root_].edges[i].prior;
            int nf = int(std::sqrt(double(p_.forcedPlayouts) * double(pr) * double(N)));
            if (nf <= 0) continue;
            uint32_t c = nodes_[root_].edges[i].child;
            int have = int(c ? nodes_[c].n : 0);
            if (have < nf) run_forced(int(i), nf - have);
        }
    }
    // completed-Q improved policy: unvisited actions get the root value estimate
    float vmix = nodes_[root_].q();
    sigmaMaxN = maxNbeforeForced;   // pruning analogue: the scale is the pre-top-up one
    policyTarget.resize(E);
    float mx = -1e30f;
    for (size_t i = 0; i < E; i++) { policyTarget[i] = logit[i] + sigma(visited(i) ? qroot(i) : vmix); mx = std::max(mx, policyTarget[i]); }
    float sum = 0.0f;
    for (float& z : policyTarget) { z = std::exp(z - mx); sum += z; }
    for (float& z : policyTarget) z /= sum;
    // CERTAINTY at the root: if a move is PROVEN to win for the team to move, play it and make the
    // training target one-hot on it. The rules settled this position, so a completed-Q target that
    // spread mass elsewhere would be teaching the net away from a proof.
    {
        int pw = root_proven_win_edge();
        if (pw >= 0) {
            std::fill(policyTarget.begin(), policyTarget.end(), 0.0f);
            if (size_t(pw) < policyTarget.size()) policyTarget[pw] = 1.0f;
            return nodes_[root_].edges[pw].move;
        }
    }
    return nodes_[root_].edges[best].move;
}

RootStats Mcts::root_stats() const {
    RootStats rs;
    const Node& r = nodes_[root_];
    rs.rootQ = r.q();
    rs.nodes = visits_done_;
    rs.maxDepth = maxDepth_;
    for (const Edge& e : r.edges) {
        rs.moves.push_back(e.move);
        rs.prior.push_back(e.prior);
        if (e.child) {
            const Node& c = nodes_[e.child];
            rs.visits.push_back(c.n);
            rs.q.push_back(c.n ? (same_team(r.stm, c.stm) ? c.q() : -c.q()) : 0.0f);
        } else {
            rs.visits.push_back(0);
            rs.q.push_back(0.0f);
        }
    }
    return rs;
}

Move Mcts::select_move(float temperature) {
    RootStats rs = root_stats();
    if (rs.moves.empty()) return MOVE_NONE;
    // CERTAINTY: a proven win is played at once, whatever the visit counts say, and whatever the
    // temperature is - randomising away from a proof would be strictly worse play.
    {
        int pw = root_proven_win_edge();
        if (pw >= 0) return nodes_[root_].edges[pw].move;
    }
    if (temperature <= 0.0f) {
        size_t best = 0;
        for (size_t i = 1; i < rs.moves.size(); i++)
            if (rs.visits[i] > rs.visits[best] ||
                (rs.visits[i] == rs.visits[best] && rs.q[i] > rs.q[best])) best = i;
        return rs.moves[best];
    }
    std::vector<double> w(rs.moves.size());
    double s = 0;
    for (size_t i = 0; i < w.size(); i++) { w[i] = std::pow(double(rs.visits[i]), 1.0 / temperature); s += w[i]; }
    if (s <= 0) return rs.moves[0];
    std::uniform_real_distribution<double> u(0.0, s);
    double x = u(rng_);
    for (size_t i = 0; i < w.size(); i++) { x -= w[i]; if (x <= 0) return rs.moves[i]; }
    return rs.moves.back();
}

void Mcts::apply_move(Move m) {
    Node& r = nodes_[root_];
    uint32_t child = 0;
    for (const Edge& e : r.edges) if (e.move == m) { child = e.child; break; }
    pos_.do_move(m);
    maxDepth_ = 0;
    if (!child) {
        nodes_.clear();
        nodes_.emplace_back();
        root_ = new_node(pos_.side_to_move());
        visits_done_ = 0;
        return;
    }
    if (nodes_[child].terminal && nodes_[child].edges.empty()) {  // pre-resolved child became the root
        bool over = false;
        game_result(pos_, over);
        if (!over) { nodes_.clear(); nodes_.emplace_back(); root_ = new_node(pos_.side_to_move()); visits_done_ = 0; return; }
    }
    // Re-root on the chosen child and COMPACT: copy the live subtree into a fresh
    // pool so memory does not grow across a game (the rest of the old tree dies).
    std::vector<Node> fresh;
    fresh.reserve(nodes_[child].n + 64);
    fresh.emplace_back();  // sentinel
    fresh.push_back(std::move(nodes_[child]));
    std::vector<uint32_t> stack{1};
    while (!stack.empty()) {
        uint32_t nn = stack.back();
        stack.pop_back();
        for (size_t i = 0; i < fresh[nn].edges.size(); i++) {
            uint32_t oc = fresh[nn].edges[i].child;
            if (!oc) continue;
            fresh.push_back(std::move(nodes_[oc]));
            uint32_t idx = uint32_t(fresh.size() - 1);
            fresh[nn].edges[i].child = idx;
            stack.push_back(idx);
        }
    }
    nodes_.swap(fresh);
    root_ = 1;
    visits_done_ = nodes_[root_].n;
}

}  // namespace zero
