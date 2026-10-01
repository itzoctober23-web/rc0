// SPDX-License-Identifier: GPL-3.0-or-later
// Cross-game batcher: many game threads each own an Mcts and hand their gathered leaves
// to one GPU thread, which evaluates them together (Lc0's self-play design). A game
// thread blocks until its items are done. Batches fire at `target` items or after
// `flushMs` of waiting, whichever first.
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include "evaluator.h"

namespace zero {

class SharedBatcher {
public:
    SharedBatcher(Evaluator& inner, int target, int maxBatch, double flushMs)
        : inner_(inner), target_(target), maxBatch_(maxBatch), flushMs_(flushMs) {
        gpu_ = std::thread([this] { loop(); });
    }
    ~SharedBatcher() { stop(); }
    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (stopping_) return;
            stopping_ = true;
        }
        cvGpu_.notify_all();
        if (gpu_.joinable()) gpu_.join();
    }

    // Called from a game thread with that game's batch; returns when all are evaluated.
    void submit(std::vector<EvalItem*>& items) {
        if (items.empty()) return;
        std::atomic<int> pending{int(items.size())};
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (EvalItem* it : items) queue_.push_back({it, &pending});
        }
        cvGpu_.notify_one();
        std::unique_lock<std::mutex> lk(mu_);
        cvDone_.wait(lk, [&] { return pending.load() == 0; });
    }

    uint64_t batches() const { return batches_; }
    uint64_t items() const { return items_; }
    double avgBatch() const { return batches_ ? double(items_) / double(batches_) : 0.0; }

private:
    struct Req { EvalItem* item; std::atomic<int>* pending; };
    void loop() {
        std::vector<Req> take;
        std::vector<EvalItem*> batch;
        while (true) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                if (queue_.empty()) cvGpu_.wait(lk, [&] { return stopping_ || !queue_.empty(); });
                if (stopping_ && queue_.empty()) return;
                if (int(queue_.size()) < target_)
                    cvGpu_.wait_for(lk, std::chrono::duration<double, std::milli>(flushMs_),
                                    [&] { return stopping_ || int(queue_.size()) >= target_; });
                size_t n = std::min(queue_.size(), size_t(maxBatch_));
                take.assign(queue_.begin(), queue_.begin() + n);
                queue_.erase(queue_.begin(), queue_.begin() + n);
            }
            batch.clear();
            for (Req& r : take) batch.push_back(r.item);
            inner_.evaluate(batch);
            batches_++;
            items_ += batch.size();
            {
                std::lock_guard<std::mutex> lk(mu_);
                for (Req& r : take) r.pending->fetch_sub(1);
            }
            cvDone_.notify_all();
        }
    }

    Evaluator& inner_;
    int target_, maxBatch_;
    double flushMs_;
    std::vector<Req> queue_;
    std::mutex mu_;
    std::condition_variable cvGpu_, cvDone_;
    bool stopping_ = false;
    std::thread gpu_;
    std::atomic<uint64_t> batches_{0}, items_{0};
};

// Per-game proxy: looks like an Evaluator to the Mcts, forwards to the shared batcher.
class BatchedEvaluator : public Evaluator {
public:
    explicit BatchedEvaluator(SharedBatcher& b) : b_(b) {}
    void evaluate(std::vector<EvalItem*>& batch) override { b_.submit(batch); }
    const char* name() const override { return "shared batcher"; }
private:
    SharedBatcher& b_;
};

}  // namespace zero
