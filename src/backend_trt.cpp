// SPDX-License-Identifier: GPL-3.0-or-later
// TensorRT backend for self-play inference, alongside backend_torch.cpp.
//
// Why this exists: TensorRT fp16 measured 1.655x LibTorch on 10x128 and 2.029x on 15x192 (BENCH.md), and the
// 15x192 net is choosing only clears its recorded/hr floor on TensorRT. There is no NvInfer.h on this box
// and sudo needs a password, so the headers come from github.com/NVIDIA/TensorRT main (11.3.0.99, an exact
// match to the installed wheel) in zero/third_party/tensorrt/include, linked against the wheel's own
// libnvinfer.so.11 with an RPATH - no system install.
//
// The .plan is built OUT OF PROCESS by tools/trt_build_plan.py (ONNX parse + engine build take 13-16 s and
// should not run on every self-play start). This file only deserialises, and it REFUSES a plan whose .meta
// does not match the net's current mtime/size, so a stale plan cannot be silently used.
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <vector>
#include "uci4.h"

namespace zero {
namespace {

class TrtLogger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::printf("info string TRT %s\n", msg);
    }
};
TrtLogger g_trtLogger;

// the .meta sidecar tools/trt_build_plan.py writes; a mismatch means the plan is for a different net
bool plan_matches_net(const std::string& planPath, const std::string& netPath, int& planesOut, int& batchMaxOut) {
    std::ifstream f(planPath + ".meta");
    if (!f) return false;
    std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    struct stat st{};
    if (::stat(netPath.c_str(), &st) != 0) return false;
    auto num = [&](const char* key, double& out) {
        auto p = all.find(std::string("\"") + key + "\"");
        if (p == std::string::npos) return false;
        p = all.find(':', p);
        if (p == std::string::npos) return false;
        out = std::strtod(all.c_str() + p + 1, nullptr);
        return true;
    };
    double mt = 0, sz = 0, pl = 0, bmax = 0;
    if (!num("net_mtime", mt) || !num("net_size", sz) || !num("planes", pl) || !num("batch_max", bmax)) return false;
    if (static_cast<long long>(sz) != static_cast<long long>(st.st_size)) return false;
    // mtime is a float in the json; compare to the second
    if (static_cast<long long>(mt) != static_cast<long long>(st.st_mtime)) return false;
    planesOut = static_cast<int>(pl);
    batchMaxOut = static_cast<int>(bmax);
    return true;
}

class TrtEvaluator : public Evaluator {
public:
    TrtEvaluator(std::unique_ptr<nvinfer1::IRuntime> rt, std::unique_ptr<nvinfer1::ICudaEngine> eng,
                 std::unique_ptr<nvinfer1::IExecutionContext> ctx, int planes, int maxBatch)
        : rt_(std::move(rt)), eng_(std::move(eng)), ctx_(std::move(ctx)), planes_(planes), maxBatch_(maxBatch) {
        // _Float16, not __fp16: __fp16 is an ARM type and GCC rejects it in C++ on x86-64. _Float16 is the
        // portable spelling and GCC 16 supports it here.
        cudaStreamCreate(&stream_);
        const size_t inElems = size_t(maxBatch_) * planes_ * CELLS;
        cudaMallocHost(&hIn_, inElems * 2);                       // fp16 host staging, pinned
        cudaMalloc(&dIn_, inElems * 2);
        cudaMallocHost(&hPol_, size_t(maxBatch_) * POLICY_SIZE * 2);
        cudaMalloc(&dPol_, size_t(maxBatch_) * POLICY_SIZE * 2);
        cudaMallocHost(&hWdl_, size_t(maxBatch_) * 3 * 2);
        cudaMalloc(&dWdl_, size_t(maxBatch_) * 3 * 2);
        cudaMallocHost(&hMl_, size_t(maxBatch_) * 2);
        cudaMalloc(&dMl_, size_t(maxBatch_) * 2);
        ok_ = hIn_ && dIn_ && hPol_ && dPol_ && hWdl_ && dWdl_ && hMl_ && dMl_;
        if (!ok_) std::printf("info string TRT: buffer allocation failed\n");
    }
    ~TrtEvaluator() override {
        for (void* p : {hIn_, hPol_, hWdl_, hMl_}) if (p) cudaFreeHost(p);
        for (void* p : {dIn_, dPol_, dWdl_, dMl_}) if (p) cudaFree(p);
        if (stream_) cudaStreamDestroy(stream_);
    }
    bool ok() const { return ok_; }
    const char* name() const override { return "tensorrt fp16"; }

    void evaluate(std::vector<EvalItem*>& batch) override {
        for (size_t start = 0; start < batch.size(); start += size_t(maxBatch_)) {
            const size_t n = std::min(batch.size() - start, size_t(maxBatch_));
            // fp16 host staging. The planes are 0/1 apart from PL_RULE50, and the net runs in fp16 anyway, so
            // converting here is the same arithmetic the LibTorch path does - it just happens before the copy.
            // PARALLEL fill. Measured: at 31,094 evals/s the wall time is 16.5 ms a batch with the GPU 82.3%
            // busy, i.e. ~2.9 ms idle, and the batcher is already 511.7 of 512 full - so the gap is this serial
            // conversion of 4,114,432 floats, not a shortage of leaves. Rule 6 prescribes "add parallel games /
            // raise batch", but batch is maxed and more threads would only deepen a queue that is already full.
            auto* hin = static_cast<_Float16*>(hIn_);
#pragma omp parallel for schedule(static)
            for (long i = 0; i < long(n); i++) {
                const float* src = batch[start + size_t(i)]->planes.data();
                _Float16* dst = hin + size_t(i) * size_t(INPUT_FLOATS);
                for (int k = 0; k < INPUT_FLOATS; k++) dst[k] = static_cast<_Float16>(src[k]);
            }
            cudaMemcpyAsync(dIn_, hIn_, n * size_t(INPUT_FLOATS) * 2, cudaMemcpyHostToDevice, stream_);
            if (!set_shapes(int(n))) return;
            if (!ctx_->enqueueV3(stream_)) { std::printf("info string TRT: enqueueV3 failed\n"); return; }
            cudaMemcpyAsync(hPol_, dPol_, n * size_t(POLICY_SIZE) * 2, cudaMemcpyDeviceToHost, stream_);
            cudaMemcpyAsync(hWdl_, dWdl_, n * 3 * 2, cudaMemcpyDeviceToHost, stream_);
            cudaMemcpyAsync(hMl_, dMl_, n * 2, cudaMemcpyDeviceToHost, stream_);
            cudaStreamSynchronize(stream_);
            // The LibTorch path gathers the legal slots ON the GPU so only n x maxLegal floats come back.
            // Doing that here would need a CUDA kernel of our own, so this first version brings the whole
            // policy row back (n x 23716 fp16 = 24 MB at batch 512) and gathers on the host: about 1.2 ms at
            // 20 GB/s against ~23 ms of compute. If a measurement says that matters, a kernel replaces it -
            // but the number comes first.
            const auto* pol = static_cast<const _Float16*>(hPol_);
            const auto* wdl = static_cast<const _Float16*>(hWdl_);
            const auto* ml  = static_cast<const _Float16*>(hMl_);
            for (size_t i = 0; i < n; i++) {
                EvalItem& it = *batch[start + i];
                const size_t k = it.policyIdx.size();
                it.priors.resize(k);
                const _Float16* row = pol + i * size_t(POLICY_SIZE);
                float mx = -1e30f;
                for (size_t j = 0; j < k; j++) { float v = float(row[it.policyIdx[j]]); it.priors[j] = v; if (v > mx) mx = v; }
                float sum = 0.0f;
                for (size_t j = 0; j < k; j++) { it.priors[j] = std::exp(it.priors[j] - mx); sum += it.priors[j]; }
                if (sum > 0.0f) for (size_t j = 0; j < k; j++) it.priors[j] /= sum;
                // wdl comes out as logits, exactly as the LibTorch path receives them
                float w0 = float(wdl[i * 3 + 0]), w1 = float(wdl[i * 3 + 1]), w2 = float(wdl[i * 3 + 2]);
                float m = std::max(w0, std::max(w1, w2));
                float e0 = std::exp(w0 - m), e1 = std::exp(w1 - m), e2 = std::exp(w2 - m);
                float z = e0 + e1 + e2;
                it.value = z > 0.0f ? (e0 - e2) / z : 0.0f;
                it.movesLeft = float(ml[i]);
            }
        }
    }

private:
    bool set_shapes(int n) {
        for (int i = 0; i < eng_->getNbIOTensors(); i++) {
            const char* nm = eng_->getIOTensorName(i);
            const bool isIn = eng_->getTensorIOMode(nm) == nvinfer1::TensorIOMode::kINPUT;
            if (isIn) {
                auto d = eng_->getTensorShape(nm);
                d.d[0] = n;
                if (!ctx_->setInputShape(nm, d)) { std::printf("info string TRT: setInputShape failed\n"); return false; }
                ctx_->setTensorAddress(nm, dIn_);
            } else {
                void* dst = std::strcmp(nm, "wdl") == 0 ? dWdl_ : std::strcmp(nm, "moves_left") == 0 ? dMl_ : dPol_;
                ctx_->setTensorAddress(nm, dst);
            }
        }
        return true;
    }
    std::unique_ptr<nvinfer1::IRuntime> rt_;
    std::unique_ptr<nvinfer1::ICudaEngine> eng_;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
    int planes_, maxBatch_;
    bool ok_ = false;
    cudaStream_t stream_ = nullptr;
    void *hIn_ = nullptr, *dIn_ = nullptr, *hPol_ = nullptr, *dPol_ = nullptr;
    void *hWdl_ = nullptr, *dWdl_ = nullptr, *hMl_ = nullptr, *dMl_ = nullptr;
};

}  // namespace

// Returns nullptr when there is no usable plan - the caller then falls back to LibTorch, which is the
// behaviour asks for ("keep LibTorch as the fallback path behind a flag").
std::unique_ptr<Evaluator> make_trt_evaluator(const std::string& netPath, int maxBatch) {
    const std::string planPath = netPath + ".plan";
    int planes = 0, planBatchMax = 0;
    if (!plan_matches_net(planPath, netPath, planes, planBatchMax)) {
        std::printf("info string TRT: no plan matching %s (run tools/trt_build_plan.py); using LibTorch\n",
                    netPath.c_str());
        return nullptr;
    }
    if (planes != N_PLANES) {
        std::printf("info string TRT: plan has %d planes, this build expects %d; using LibTorch\n", planes, N_PLANES);
        return nullptr;
    }
    std::ifstream f(planPath, std::ios::binary);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (blob.empty()) { std::printf("info string TRT: plan unreadable; using LibTorch\n"); return nullptr; }
    std::unique_ptr<nvinfer1::IRuntime> rt{nvinfer1::createInferRuntime(g_trtLogger)};
    if (!rt) return nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> eng{rt->deserializeCudaEngine(blob.data(), blob.size())};
    if (!eng) { std::printf("info string TRT: deserialize failed; using LibTorch\n"); return nullptr; }
    std::unique_ptr<nvinfer1::IExecutionContext> ctx{eng->createExecutionContext()};
    if (!ctx) return nullptr;
    std::printf("info string TRT: loaded %s (%zu bytes, %d planes, batch<=%d)\n", planPath.c_str(), blob.size(),
                planes, planBatchMax);
    return std::make_unique<TrtEvaluator>(std::move(rt), std::move(eng), std::move(ctx), planes,
                                          std::min(maxBatch, planBatchMax));
}

}  // namespace zero
