// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <memory>
#include "evaluator.h"
#include "mcts.h"

namespace zero {
struct EngineOptions {
    std::string netPath;       // network file (.pt or .onnx); empty = uniform stand-in
    int playoutsCap = 0;       // 0 = unlimited (use time)
    int moveOverheadMs = 50;
    float temperature = 0.0f;
    std::string policy;        // "" = MCTS; "random" = uniform random legal move; "greedy" = 1-ply material-greedy (ladder rungs)
    SearchParams sp;
};
// Factory implemented by the backend unit (returns nullptr if unavailable).
std::unique_ptr<Evaluator> make_torch_evaluator(const std::string& path, int maxBatch);
// returns nullptr when there is no usable .plan for this net, or when the TRT backend was not built -
// the caller then falls back to LibTorch, which is the fallback-behind-a-flag asks for.
std::unique_ptr<Evaluator> make_trt_evaluator(const std::string& path, int maxBatch);
// ONNX Runtime (the portable backend used by the release downloads); nullptr when not built or no .onnx file.
std::unique_ptr<Evaluator> make_onnx_evaluator(const std::string& path, int maxBatch);
// Try TensorRT first, then LibTorch. ZERO_BACKEND=torch forces LibTorch; ZERO_BACKEND=trt refuses to fall back,
// so a gate can prove it is really measuring TRT rather than silently testing LibTorch twice.
std::unique_ptr<Evaluator> make_evaluator(const std::string& path, int maxBatch);
void uci4_loop(EngineOptions opts);
}  // namespace zero
