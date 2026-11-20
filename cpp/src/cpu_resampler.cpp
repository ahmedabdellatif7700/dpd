#include "dpd/cpu_resampler.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "dpd/tensor_file.h"

namespace dpd {

ResamplerFilter load_resampler(const std::string& path) {
    const auto t = read_tensor_file(path);
    const Tensor& r = find(t, "resampler.ratio");
    const Tensor& h = find(t, "resampler.h");
    if (r.size() != 2 || h.dims.size() != 1 || h.size() % 2 == 0)
        throw std::runtime_error(path + ": not a resampler file (ratio [2], an odd number of taps)");
    ResamplerFilter f{static_cast<int>(r.data[0]), static_cast<int>(r.data[1]), h.data};
    if (f.up <= 0 || f.down <= 0) throw std::runtime_error(path + ": bad resampling ratio");
    return f;
}

int resampled_length(int n, int up, int down) {
    return static_cast<int>((static_cast<long long>(n) * up + down - 1) / down);
}

CpuResampler::CpuResampler(const ResamplerFilter& f) : f_(f) {}

void CpuResampler::to_dpd(const float* x, int n, float* y) const { run(x, n, f_.up, f_.down, y); }
void CpuResampler::to_pipeline(const float* x, int n, float* y) const { run(x, n, f_.down, f_.up, y); }

// y[m] = up * sum_k x[k] h[c + m*down - k*up], the k for which the tap exists
void CpuResampler::run(const float* x, int n, int up, int down, float* y) const {
    const long long L = static_cast<long long>(f_.h.size()), c = (L - 1) / 2;
    const int n_out = resampled_length(n, up, down);
    for (int m = 0; m < n_out; ++m) {
        const long long t = c + static_cast<long long>(m) * down;           // the tap index of x[0]
        const long long k_lo = std::max(0LL, t - (L - 1) <= 0 ? 0 : (t - (L - 1) + up - 1) / up);
        const long long k_hi = std::min(static_cast<long long>(n) - 1, t / up);
        float re = 0.0f, im = 0.0f;
        for (long long k = k_lo; k <= k_hi; ++k) {
            const float w = f_.h[static_cast<std::size_t>(t - k * up)];
            re += w * x[2 * k];
            im += w * x[2 * k + 1];
        }
        y[2 * m] = static_cast<float>(up) * re;
        y[2 * m + 1] = static_cast<float>(up) * im;
    }
}

}  // namespace dpd
