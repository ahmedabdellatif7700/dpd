// The transformer DPD in plain C++ on the CPU, float32: docs/dpd/METHODOLOGY.md, operation by
// operation. The twin's own reference: the CUDA kernels are checked against it, and it against
// the PyTorch reference's golden files.
#pragma once

#include <cstddef>
#include <vector>

#include "dpd/weights.h"

namespace dpd {

// The intermediate values of one window (the golden files' probes)
struct Trace {
    std::vector<float> embed;               // [T][d]
    std::vector<std::vector<float>> layer;  // [L][T][d], after each encoder layer
    std::vector<float> final_norm;          // [d], the centre token after the final norm
    float out[2] = {0, 0};                  // before the scale
};

// One window: win[T][2] (I/Q, the drive's units) -> out[2] (I/Q). trace optional.
void forward_window(const Weights& w, const float* win, float* out, Trace* trace = nullptr);

// A whole sequence, as the reference's predistort(): x[n][2] -> y[n][2], output n from inputs
// n - T/2 .. n + T/2 - 1, zeros beyond the sequence's ends.
void predistort(const Weights& w, const float* x, std::size_t n, float* y);

// The window of output sample i of a sequence x[n][2] (zero-padded), into win[T][2].
void window_at(const Config& c, const float* x, std::size_t n, std::size_t i, float* win);

}  // namespace dpd
