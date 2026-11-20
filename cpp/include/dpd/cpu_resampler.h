// The resampler (resampler.h) in plain C++ on the CPU, float32: the twin's own reference for
// GpuResampler, which has the same interface.
#pragma once

#include "dpd/resampler.h"

namespace dpd {

class CpuResampler {
public:
    explicit CpuResampler(const ResamplerFilter& f);

    // 50 -> 8 samples per symbol: x[n][2] -> y[dpd_length(n)][2] (I/Q), the sequence alone
    void to_dpd(const float* x, int n, float* y) const;
    // 8 -> 50: x[n][2] -> y[pipeline_length(n)][2]
    void to_pipeline(const float* x, int n, float* y) const;

    int dpd_length(int n) const { return resampled_length(n, f_.up, f_.down); }
    int pipeline_length(int n) const { return resampled_length(n, f_.down, f_.up); }

private:
    void run(const float* x, int n, int up, int down, float* y) const;
    ResamplerFilter f_;
};

}  // namespace dpd
