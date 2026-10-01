// SPDX-License-Identifier: GPL-3.0-or-later
// NN cache (Lc0-style): a direct-mapped table keyed by position hash. Transpositions
// and re-searched positions skip the GPU. Priors are stored in legal-move order, which
// is deterministic for a given position, so they line up with the caller's edges.
#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>
#include "evaluator.h"

namespace zero {

class CachedEvaluator : public Evaluator {
public:
    CachedEvaluator(Evaluator& inner, int log2Entries = 18)
        : inner_(inner), mask_((1u << log2Entries) - 1), table_(1u << log2Entries) {}

    void evaluate(std::vector<EvalItem*>& batch) override {
        std::vector<EvalItem*> miss;
        miss.reserve(batch.size());
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (EvalItem* it : batch) {
                Entry& e = table_[it->key & mask_];
                if (e.valid && e.key == it->key && e.priors.size() == it->policyIdx.size()) {
                    it->priors = e.priors;
                    it->value = e.value;
                    it->movesLeft = e.movesLeft;
                    hits_++;
                } else {
                    miss.push_back(it);
                }
            }
        }
        misses_ += miss.size();
        if (!miss.empty()) {
            inner_.evaluate(miss);
            std::lock_guard<std::mutex> lk(mu_);
            for (EvalItem* it : miss) {
                Entry& e = table_[it->key & mask_];
                e.key = it->key;
                e.priors = it->priors;
                e.value = it->value;
                e.movesLeft = it->movesLeft;
                e.valid = true;
            }
        }
    }
    const char* name() const override { return inner_.name(); }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        for (Entry& e : table_) e.valid = false;
        hits_ = misses_ = 0;
    }

private:
    struct Entry {
        uint64_t key = 0;
        std::vector<float> priors;
        float value = 0.0f, movesLeft = 0.0f;
        bool valid = false;
    };
    Evaluator& inner_;
    uint64_t mask_;
    std::vector<Entry> table_;
    std::mutex mu_;
    std::atomic<uint64_t> hits_{0}, misses_{0};
};

}  // namespace zero
