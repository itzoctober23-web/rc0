// SPDX-License-Identifier: GPL-3.0-or-later
// LibTorch backend: a TorchScript module with
//   forward(x[B,41,14,14]) -> (policy_logits[B,23716], wdl_logits[B,3], moves_left[B,1])
// Priors = softmax over the LEGAL slots only; value = P(win) - P(loss), team-to-move view.
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/api/module.h>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include "uci4.h"

namespace zero {

class TorchEvaluator : public Evaluator {
public:
    TorchEvaluator(torch::jit::Module m, torch::Device dev, int maxBatch, bool prepared = false)
        : m_(std::move(m)), dev_(dev), maxBatch_(maxBatch) {
        half_ = dev_.is_cuda();
        if (!prepared) {
            m_.eval();
            if (half_) m_.to(torch::kHalf);
            m_.to(dev_);
        }
        // kHalf on the HOST, not float32. The net already runs in kHalf, so the values the net sees are
        // bit-identical - the float->half conversion merely moves off the .to(device, kHalf) call, which
        // could not use the pinned async path and therefore SERIALISED with compute. Measured 2026-10-01 at
        // 210 planes, as a fraction of compute-only: float32 + host cast 82.9%, half on host 92.1%.
        in_ = torch::empty({maxBatch_, N_PLANES, BOARD, BOARD},
                           torch::TensorOptions().dtype(half_ ? torch::kHalf : torch::kFloat32).pinned_memory(dev_.is_cuda()));
        in_.zero_();
        // Batches are padded to power-of-two sizes so only ~9 shapes ever reach cuDNN, and
        // every one of them is run once here: otherwise the first search of a game spends
        // its whole time budget in per-shape algorithm selection (measured: 1 playout in 300 ms).
        // Each shape THREE times: the TorchScript profiling executor profiles the first run(s) and
        // optimises (NVRTC-compiles fused kernels) on a later one; with a single warm-up call that
        // optimisation landed inside the first real search (measured 2026-09-30: a 6x64 net answered
        // `go movetime 300` from a cold process with 0 playouts).
        torch::NoGradGuard ng;
        for (int pass = 0; pass < 3; pass++)
            for (int b = 1; b <= maxBatch_; b *= 2) {
                torch::Tensor x = in_.narrow(0, 0, b).to(dev_)
                                     .contiguous(torch::MemoryFormat::ChannelsLast);   // 1.26x measured 06:17
                auto out = m_.forward({x}).toTuple();
                (void)torch::softmax(out->elements()[1].toTensor().to(torch::kFloat32), 1).cpu();
            }
        if (dev_.is_cuda()) torch::cuda::synchronize();
    }
    static int padded(int n) { int b = 1; while (b < n) b *= 2; return b; }
    const char* name() const override { return half_ ? "torchscript cuda fp16" : "torchscript cpu fp32"; }

    void evaluate(std::vector<EvalItem*>& batch) override {
        torch::NoGradGuard ng;
        for (size_t start = 0; start < batch.size(); start += size_t(maxBatch_)) {
            size_t n = std::min(batch.size() - start, size_t(maxBatch_));
            int64_t maxLegal = 1;
            if (half_) {   // convert while filling: -march=native turns this into vcvtps2ph
                at::Half* dst = in_.data_ptr<at::Half>();
                for (size_t i = 0; i < n; i++) {
                    const float* src = batch[start + i]->planes.data();
                    at::Half* d = dst + i * INPUT_FLOATS;
                    for (int k = 0; k < INPUT_FLOATS; k++) d[k] = static_cast<at::Half>(src[k]);
                    maxLegal = std::max<int64_t>(maxLegal, int64_t(batch[start + i]->policyIdx.size()));
                }
            } else {
                float* dst = in_.data_ptr<float>();
                for (size_t i = 0; i < n; i++) {
                    std::memcpy(dst + i * INPUT_FLOATS, batch[start + i]->planes.data(), sizeof(float) * INPUT_FLOATS);
                    maxLegal = std::max<int64_t>(maxLegal, int64_t(batch[start + i]->policyIdx.size()));
                }
            }
            int64_t nb = std::min(padded(int(n)), maxBatch_);
            // legal-move slots, padded with slot 0 and masked: the gather + softmax run on the
            // GPU so only n x maxLegal floats come back instead of n x 23716.
            torch::Tensor idx = torch::zeros({int64_t(n), maxLegal}, torch::kInt64);
            torch::Tensor mask = torch::zeros({int64_t(n), maxLegal}, torch::kBool);
            auto ia = idx.accessor<int64_t, 2>();
            auto ma = mask.accessor<bool, 2>();
            for (size_t i = 0; i < n; i++) {
                const auto& pi = batch[start + i]->policyIdx;
                for (size_t j = 0; j < pi.size(); j++) { ia[i][j] = pi[j]; ma[i][j] = true; }
            }
            torch::Tensor x = in_.narrow(0, 0, nb).to(dev_, /*non_blocking=*/true)
                                      .contiguous(torch::MemoryFormat::ChannelsLast);   // 1.26x measured 06:17
            auto out = m_.forward({x}).toTuple();
            torch::Tensor pol = out->elements()[0].toTensor().narrow(0, 0, int64_t(n)).to(torch::kFloat32);
            torch::Tensor gi = idx.to(dev_, /*non_blocking=*/true);
            torch::Tensor gm = mask.to(dev_, /*non_blocking=*/true);
            torch::Tensor logits = torch::gather(pol, 1, gi).masked_fill(~gm, -1e9f);
            // ONE device-to-host transfer instead of three. Each .cpu() is a synchronisation point with
            // nothing queued behind it, and the measured gap was exactly that: at 19,132 evals/s on full
            // batches of 512 the generator does 37.4 batches/s = 26.8 ms of wall per batch while nvidia-smi
            // reads 66-70%, i.e. a pipeline bubble rather than a shortage of work. Concatenating on the GPU
            // leaves one sync per batch.
            torch::Tensor priG = torch::softmax(logits, 1);
            torch::Tensor wdlG = torch::softmax(out->elements()[1].toTensor().narrow(0, 0, int64_t(n)).to(torch::kFloat32), 1);
            torch::Tensor mlG  = out->elements()[2].toTensor().narrow(0, 0, int64_t(n)).to(torch::kFloat32).reshape({int64_t(n), 1});
            torch::Tensor allH = torch::cat({priG, wdlG, mlG}, 1).contiguous().cpu();   // [n, maxLegal + 3 + 1]
            auto aa = allH.accessor<float, 2>();
            const int64_t WOFF = maxLegal, MOFF = maxLegal + 3;
            for (size_t i = 0; i < n; i++) {
                EvalItem& it = *batch[start + i];
                size_t k = it.policyIdx.size();
                it.priors.resize(k);
                for (size_t j = 0; j < k; j++) it.priors[j] = aa[i][int64_t(j)];
                it.value = aa[i][WOFF + 0] - aa[i][WOFF + 2];
                it.movesLeft = aa[i][MOFF];
            }
        }
    }

private:
    torch::jit::Module m_;
    torch::Device dev_;
    int maxBatch_;
    bool half_;
    torch::Tensor in_;
};

std::unique_ptr<Evaluator> make_torch_evaluator(const std::string& path, int maxBatch) {
    try {
        torch::jit::Module m = torch::jit::load(path, torch::kCPU);
        torch::Device dev = torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
        if (const char* opt = std::getenv("ZERO_TORCH_OPT"); opt && *opt == '1') {
            // freeze + optimize_for_inference: folds BN, drops training-only graph parts (A/B benched)
            m.eval();
            if (dev.is_cuda()) m.to(torch::kHalf);
            m.to(dev);
            torch::jit::Module fm = torch::jit::freeze(m);
            m = torch::jit::optimize_for_inference(fm);
            return std::make_unique<TorchEvaluator>(std::move(m), dev, maxBatch, /*prepared=*/true);
        }
        return std::make_unique<TorchEvaluator>(std::move(m), dev, maxBatch);
    } catch (const std::exception& e) {
        std::printf("info string torch load failed: %s\n", e.what());
        return nullptr;
    }
}

}  // namespace zero
