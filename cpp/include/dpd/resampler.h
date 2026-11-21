// The resampler between the pipeline's rate and the DPD's: BASEBAND at 50 samples per symbol <->
// the model at 8 (docs/testbed/jetson/STAGE.md, decision 2b; the reference: dpd/resample.py).
//
//   y[m] = up * sum_k x[k] h[c + m*down - k*up],   c = (taps - 1) / 2,
//
// x zero outside 0 .. n-1: output sample m sits at the time of input sample m*down/up (no delay).
// 50 -> 8 is up 4, down 25; 8 -> 50 swaps them; one low-pass h at the intermediate 80 GS/s serves
// both. CpuResampler and GpuResampler implement it, with the same interface.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace dpd {

struct ResamplerFilter {
    int up = 0, down = 0;  // 50 -> 8: 4, 25 (8 -> 50: 25, 4)
    std::vector<float> h;  // the low-pass, odd length, symmetric
};

// Loads a resampler file (dpd/export.py, export_resampler): `resampler.ratio` [2], `resampler.h`
// [taps]; throws std::runtime_error if it is not one.
ResamplerFilter load_resampler(const std::string& path);

// the output samples of an n-sample sequence resampled by up/down: ceil(n * up / down)
int resampled_length(int n, int up, int down);

}  // namespace dpd
