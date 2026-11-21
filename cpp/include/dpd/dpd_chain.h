// The DPD at the pipeline's rate: 50 -> 8 samples per symbol, the model, 8 -> 50 (decision 2b),
// one segment taken alone. Written once over the shared interface of the model classes
// (CpuModel, GpuModel: predistort) and the resampler classes (CpuResampler, GpuResampler: to_dpd,
// to_pipeline), so the same code runs on the CPU or the GPU. The reference: dpd/resample.py,
// predistort_pipeline with frame=None.
#pragma once

#include <algorithm>
#include <vector>

namespace dpd {

template <typename Model, typename Resampler>
class DpdChain {
public:
    DpdChain(const Model& model, const Resampler& resampler) : model_(model), resampler_(resampler) {}

    // x[n][2] -> y[n][2] (I/Q at 50 samples per symbol, the model's units)
    void run(const float* x, int n, float* y) const {
        const int n8 = resampler_.dpd_length(n), n50 = resampler_.pipeline_length(n8);
        a_.resize(2 * static_cast<std::size_t>(n8));
        b_.resize(2 * static_cast<std::size_t>(n8));
        c_.resize(2 * static_cast<std::size_t>(n50));
        resampler_.to_dpd(x, n, a_.data());
        model_.predistort(a_.data(), b_.data(), n8);
        resampler_.to_pipeline(b_.data(), n8, c_.data());
        std::copy(c_.begin(), c_.begin() + 2 * static_cast<std::ptrdiff_t>(n), y);  // n50 >= n
    }

private:
    const Model& model_;
    const Resampler& resampler_;
    mutable std::vector<float> a_, b_, c_;
};

}  // namespace dpd
