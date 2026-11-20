// The transformer DPD in plain C++ on the CPU, float32: docs/dpd/METHODOLOGY.md, operation by
// operation. The twin's own reference: the CUDA kernels (GpuModel, gpu_model.h) are checked
// against it, and it against the PyTorch reference's golden files. CpuModel has GpuModel's
// interface, so the two are used the same way.
#pragma once

#include <cstddef>
#include <vector>

#include "dpd/weights.h"

namespace dpd {

class CpuModel {
public:
    explicit CpuModel(const Weights& w);

    // x[n][2] -> y[n][2] (I/Q, the drive's units), the frame alone (zeros beyond its ends):
    // what the reference's predistort() does.
    void predistort(const float* x, float* y, int n) const;

    // The intermediates of the given output samples, in the golden file's layout, one probe after
    // the other: embed [T][d], layer0 .. layer<L-1> [T][d], final_norm [d], out [2] (before scale).
    // Throws if a probe lies outside 0 .. n-1.
    std::vector<float> trace(const float* x, int n, const std::vector<int>& probes) const;
    std::size_t trace_floats_per_probe() const;

private:
    Weights w_;
};

// --- the one-window computation underneath CpuModel ---

// The intermediate values of one window
struct Trace {
    std::vector<float> embed;               // [T][d]
    std::vector<std::vector<float>> layer;  // [L][T][d], after each encoder layer
    std::vector<float> final_norm;          // [d], the centre token after the final norm
    float out[2] = {0, 0};                  // before the scale
};

// One window: win[T][2] (I/Q, the drive's units) -> out[2] (I/Q). trace optional.
void forward_window(const Weights& w, const float* win, float* out, Trace* trace = nullptr);

// The window of output sample i of a sequence x[n][2] (zero-padded), into win[T][2].
void window_at(const Config& c, const float* x, std::size_t n, std::size_t i, float* win);

}  // namespace dpd
