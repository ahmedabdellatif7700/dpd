// The resampler on the GPU (resampler.h): one thread per output sample, the formula as CpuResampler
// computes it. The taps (503 floats) are read by every thread and stay in the cache.
#include "dpd/gpu_resampler.h"

#include <cuda_runtime.h>

#include "cuda_util.h"

namespace dpd {

namespace {

// y[m] = up * sum_k x[k] h[c + m*down - k*up], the k for which the tap exists
__global__ void resample_kernel(const float2* __restrict__ x, int n, const float* __restrict__ h, int taps, int up,
                                int down, float2* __restrict__ y, int n_out) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= n_out) return;
    const long long L = taps, c = (L - 1) / 2;
    const long long t = c + static_cast<long long>(m) * down;  // the tap index of x[0]
    const long long k_lo = t - (L - 1) <= 0 ? 0 : (t - (L - 1) + up - 1) / up;
    const long long k_hi = min(static_cast<long long>(n) - 1, t / up);
    float re = 0.0f, im = 0.0f;
    for (long long k = k_lo; k <= k_hi; ++k) {
        const float w = h[t - k * up];
        re += w * x[k].x;
        im += w * x[k].y;
    }
    y[m] = make_float2(static_cast<float>(up) * re, static_cast<float>(up) * im);
}

}  // namespace

GpuResampler::GpuResampler(const ResamplerFilter& f)
    : taps_(static_cast<int>(f.h.size())), up_(f.up), down_(f.down) {
    h_ = cuda::device_alloc<float>(f.h.size(), "cudaMalloc resampler taps");
    cuda::check(cudaMemcpy(h_.get(), f.h.data(), f.h.size() * sizeof(float), cudaMemcpyHostToDevice),
                "upload resampler taps");
}

void GpuResampler::run(const float* d_x, int n, int up, int down, float* d_y, void* stream) const {
    const int n_out = resampled_length(n, up, down);
    if (n <= 0 || n_out <= 0) return;
    constexpr int threads = 128;
    resample_kernel<<<(n_out + threads - 1) / threads, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const float2*>(d_x), n, h_.get(), taps_, up, down, reinterpret_cast<float2*>(d_y), n_out);
    cuda::check(cudaGetLastError(), "resample_kernel launch");
}

void GpuResampler::to_dpd_device(const float* d_x, int n, float* d_y, void* stream) const {
    run(d_x, n, up_, down_, d_y, stream);
}
void GpuResampler::to_pipeline_device(const float* d_x, int n, float* d_y, void* stream) const {
    run(d_x, n, down_, up_, d_y, stream);
}

void GpuResampler::run_host(const float* x, int n, int up, int down, float* y) const {
    const int n_out = resampled_length(n, up, down);
    if (n <= 0) return;
    if (n > x_n_) {
        x_.reset();
        x_n_ = 0;
        x_ = cuda::device_alloc<float>(2 * static_cast<std::size_t>(n), "cudaMalloc x");
        x_n_ = n;
    }
    if (n_out > y_n_) {
        y_.reset();
        y_n_ = 0;
        y_ = cuda::device_alloc<float>(2 * static_cast<std::size_t>(n_out), "cudaMalloc y");
        y_n_ = n_out;
    }
    cuda::check(cudaMemcpy(x_.get(), x, 2 * sizeof(float) * n, cudaMemcpyHostToDevice), "copy x");
    run(x_.get(), n, up, down, y_.get(), nullptr);
    cuda::check(cudaDeviceSynchronize(), "resample_kernel");  // a fault inside the kernel is reported here
    cuda::check(cudaMemcpy(y, y_.get(), 2 * sizeof(float) * n_out, cudaMemcpyDeviceToHost), "copy y");
}

void GpuResampler::to_dpd(const float* x, int n, float* y) const { run_host(x, n, up_, down_, y); }
void GpuResampler::to_pipeline(const float* x, int n, float* y) const { run_host(x, n, down_, up_, y); }

}  // namespace dpd
