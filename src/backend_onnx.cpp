// SPDX-License-Identifier: GPL-3.0-or-later
// ONNX Runtime backend: the portable one. Prebuilt ONNX Runtime packages exist for Windows, Linux and macOS,
// so this is what the release downloads use. Execution providers, in the order tried when RC0_ONNX_EP is
// unset or "auto": CUDA (NVIDIA), DirectML (any DirectX 12 GPU on Windows), CPU.
//   RC0_ONNX_EP=cuda | dml | cpu   forces one (and fails rather than silently falling back to the CPU)
// The model is the .onnx export of the network (train/export_best.py writes it next to the .pt):
//   input "planes" float32 [B,41,14,14] -> "policy" [B,23716], "wdl" [B,3], "moves_left" [B,1]
#include <onnxruntime_cxx_api.h>
#ifdef RC0_ORT_DML
#include <dml_provider_factory.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "uci4.h"

namespace zero {

namespace {

Ort::Env& ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "rc0");
    return env;
}

#ifdef _WIN32
std::wstring widen(const std::string& s) { return std::wstring(s.begin(), s.end()); }
#endif

bool has_provider(const char* name) {
    for (const std::string& p : Ort::GetAvailableProviders())
        if (p == name) return true;
    return false;
}

}  // namespace

class OnnxEvaluator : public Evaluator {
public:
    OnnxEvaluator(Ort::Session s, std::string label, int maxBatch)
        : session_(std::move(s)), label_(std::move(label)), maxBatch_(maxBatch),
          mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        input_.resize(size_t(maxBatch_) * INPUT_FLOATS);
    }
    const char* name() const override { return label_.c_str(); }

    void evaluate(std::vector<EvalItem*>& batch) override {
        static const char* inNames[] = {"planes"};
        static const char* outNames[] = {"policy", "wdl", "moves_left"};
        for (size_t start = 0; start < batch.size(); start += size_t(maxBatch_)) {
            const size_t n = std::min(batch.size() - start, size_t(maxBatch_));
            for (size_t i = 0; i < n; i++)
                std::memcpy(input_.data() + i * INPUT_FLOATS, batch[start + i]->planes.data(), sizeof(float) * INPUT_FLOATS);
            const int64_t shape[4] = {int64_t(n), N_PLANES, BOARD, BOARD};
            Ort::Value in = Ort::Value::CreateTensor<float>(mem_, input_.data(), n * INPUT_FLOATS, shape, 4);
            auto out = session_.Run(Ort::RunOptions{nullptr}, inNames, &in, 1, outNames, 3);
            const float* pol = out[0].GetTensorData<float>();
            const float* wdl = out[1].GetTensorData<float>();
            const float* ml = out[2].GetTensorData<float>();
            for (size_t i = 0; i < n; i++) {
                EvalItem& it = *batch[start + i];
                const float* p = pol + i * POLICY_SIZE;
                const size_t k = it.policyIdx.size();
                it.priors.resize(k);
                float mx = -1e30f;
                for (size_t j = 0; j < k; j++) mx = std::max(mx, p[it.policyIdx[j]]);
                float sum = 0.0f;
                for (size_t j = 0; j < k; j++) { it.priors[j] = std::exp(p[it.policyIdx[j]] - mx); sum += it.priors[j]; }
                for (size_t j = 0; j < k; j++) it.priors[j] /= sum;
                const float* w = wdl + i * 3;
                const float wm = std::max(w[0], std::max(w[1], w[2]));
                const float e0 = std::exp(w[0] - wm), e1 = std::exp(w[1] - wm), e2 = std::exp(w[2] - wm);
                it.value = (e0 - e2) / (e0 + e1 + e2);
                it.movesLeft = ml[i];
            }
        }
    }

private:
    Ort::Session session_;
    std::string label_;
    int maxBatch_;
    Ort::MemoryInfo mem_;
    std::vector<float> input_;
};

// path: the .onnx file, or a .pt whose .onnx sibling exists.
std::unique_ptr<Evaluator> make_onnx_evaluator(const std::string& pathIn, int maxBatch) {
    std::string path = pathIn;
    if (path.size() > 3 && path.substr(path.size() - 3) == ".pt") path = path.substr(0, path.size() - 3) + ".onnx";
    if (!std::ifstream(path).good()) return nullptr;
    const char* want = std::getenv("RC0_ONNX_EP");
    const std::string ep = want && *want ? want : "auto";
    std::vector<std::string> order = ep == "auto" ? std::vector<std::string>{"cuda", "dml", "cpu"} : std::vector<std::string>{ep};
    for (const std::string& e : order) {
        try {
            Ort::SessionOptions so;
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            std::string label;
            if (e == "cuda") {
                if (!has_provider("CUDAExecutionProvider")) continue;
                OrtCUDAProviderOptions cuda{};
                cuda.device_id = 0;
                so.AppendExecutionProvider_CUDA(cuda);
                label = "onnxruntime cuda";
            } else if (e == "dml") {
#ifdef RC0_ORT_DML
                if (!has_provider("DmlExecutionProvider")) continue;
                so.DisableMemPattern();
                so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(so, 0));
                label = "onnxruntime directml";
#else
                continue;
#endif
            } else if (e == "cpu") {
                label = "onnxruntime cpu";
            } else {
                std::printf("info string unknown RC0_ONNX_EP=%s (use cuda, dml, cpu or auto)\n", e.c_str());
                return nullptr;
            }
#ifdef _WIN32
            Ort::Session s(ort_env(), widen(path).c_str(), so);
#else
            Ort::Session s(ort_env(), path.c_str(), so);
#endif
            auto ev = std::make_unique<OnnxEvaluator>(std::move(s), label, maxBatch);
            // one warm-up batch so provider initialisation is not charged to the first real search
            EvalItem warm;
            warm.planes.assign(INPUT_FLOATS, 0.0f);
            warm.policyIdx = {0};
            std::vector<EvalItem*> b{&warm};
            ev->evaluate(b);
            std::printf("info string network %s on %s\n", path.c_str(), label.c_str());
            return ev;
        } catch (const std::exception& ex) {
            std::printf("info string onnxruntime %s failed: %s\n", e.c_str(), ex.what());
        }
    }
    return nullptr;
}

}  // namespace zero
