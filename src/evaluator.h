// SPDX-License-Identifier: GPL-3.0-or-later
// Network interface. The search hands over a batch of encoded positions with
// their legal policy slots; the backend returns a prior per legal move
// (softmax over the legal slots only) and a value in [-1, 1] from the
// perspective of the TEAM to move at that position.
#pragma once
#include <cmath>
#include <random>
#include <vector>
#include "encoding.h"

namespace zero {

struct EvalItem {
    uint64_t key = 0;            // position hash (Zobrist ^ side to move) for the NN cache
    std::vector<float> planes;   // INPUT_FLOATS
    std::vector<int> policyIdx;  // one slot per legal move
    // outputs
    std::vector<float> priors;   // same length as policyIdx, sums to 1
    float value = 0.0f;          // team-to-move perspective
    float movesLeft = 0.0f;
};

class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual void evaluate(std::vector<EvalItem*>& batch) = 0;
    virtual const char* name() const = 0;
};

// Uniform priors, zero value: the Stage-0 stand-in (FLAGGED: not a net).
class UniformEvaluator : public Evaluator {
public:
    void evaluate(std::vector<EvalItem*>& batch) override {
        for (EvalItem* it : batch) {
            size_t n = it->policyIdx.size();
            it->priors.assign(n, n ? 1.0f / float(n) : 0.0f);
            it->value = 0.0f;
        }
    }
    const char* name() const override { return "uniform (stand-in, no net)"; }
};

// Random priors + random value, seeded: exercises non-uniform paths in tests.
class RandomEvaluator : public Evaluator {
public:
    explicit RandomEvaluator(uint64_t seed) : rng_(seed) {}
    void evaluate(std::vector<EvalItem*>& batch) override {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        for (EvalItem* it : batch) {
            size_t n = it->policyIdx.size();
            it->priors.resize(n);
            float s = 0;
            for (size_t i = 0; i < n; i++) { it->priors[i] = 0.05f + u(rng_); s += it->priors[i]; }
            for (size_t i = 0; i < n; i++) it->priors[i] /= s;
            it->value = 2.0f * u(rng_) - 1.0f;
        }
    }
    const char* name() const override { return "random (stand-in, no net)"; }
private:
    std::mt19937_64 rng_;
};

}  // namespace zero
