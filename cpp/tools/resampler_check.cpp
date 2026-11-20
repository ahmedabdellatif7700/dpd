// resampler_check: the resampler (CpuResampler, GpuResampler) against the reference's golden file
// (dpd/export.py, export_resampler_golden), and its time per frame on the GPU.
//
//   resampler_check <resampler.dpdt> <golden.dpdt> [--tol 1e-5] [--reps 200]
//
// 50 -> 8 of the golden frame against `down`, and 8 -> 50 of `down` against `up`; errors as the
// other checks: the largest absolute difference over the RMS of the reference. Exit 0 if all are
// within --tol.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "dpd/cpu_resampler.h"
#include "dpd/gpu_resampler.h"
#include "dpd/tensor_file.h"
#include "model_check.h"

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// one resampler (CPU or GPU) against the golden down and up; true if within tol
template <typename R>
bool against_golden(const char* tag, const R& r, const dpd::Tensor& in, const dpd::Tensor& down, const dpd::Tensor& up,
                    double tol) {
    const int n = static_cast<int>(in.dims.at(0)), n8 = r.dpd_length(n);
    if (n8 != static_cast<int>(down.dims.at(0)) || r.pipeline_length(n8) < n)
        throw std::runtime_error("the golden file's lengths do not match the resampler");
    std::vector<float> y8(2 * static_cast<std::size_t>(n8)), y50(2 * static_cast<std::size_t>(r.pipeline_length(n8)));
    r.to_dpd(in.data.data(), n, y8.data());
    r.to_pipeline(down.data.data(), n8, y50.data());   // from the golden `down`: each direction alone
    const double e8 = dpd::check::rel_err(y8.data(), down.data.data(), y8.size());
    const double e50 = dpd::check::rel_err(y50.data(), up.data.data(), up.data.size());
    std::printf("resampler_check: %s: 50 -> 8 error %.2e, 8 -> 50 error %.2e\n", tag, e8, e50);
    return e8 <= tol && e50 <= tol;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: resampler_check <resampler.dpdt> <golden.dpdt> [--tol 1e-5] [--reps 200]\n");
        return 2;
    }
    double tol = 1e-5;
    int reps = 200;
    for (int i = 3; i + 1 < argc; ++i) {
        if (!std::strcmp(argv[i], "--tol")) tol = std::atof(argv[i + 1]);
        if (!std::strcmp(argv[i], "--reps")) reps = std::atoi(argv[i + 1]);
    }

    try {
        const dpd::ResamplerFilter f = dpd::load_resampler(argv[1]);
        const auto g = dpd::read_tensor_file(argv[2]);
        const auto &in = dpd::find(g, "input"), &down = dpd::find(g, "down"), &up = dpd::find(g, "up");
        std::printf("resampler_check: %s: %zu taps, 50 -> 8 = %d/%d; frame of %u samples\n", argv[1], f.h.size(), f.up,
                    f.down, in.dims.at(0));

        const dpd::CpuResampler cpu(f);
        const dpd::GpuResampler gpu(f);
        bool ok = against_golden("CpuResampler", cpu, in, down, up, tol);
        ok = against_golden("GpuResampler", gpu, in, down, up, tol) && ok;

        // the time per frame on the GPU, each direction, the frame already there
        const int n = static_cast<int>(in.dims.at(0)), n8 = gpu.dpd_length(n);
        float *d50 = nullptr, *d8 = nullptr, *d50b = nullptr;
        check(cudaMalloc(&d50, 2 * sizeof(float) * n), "cudaMalloc");
        check(cudaMalloc(&d8, 2 * sizeof(float) * n8), "cudaMalloc");
        check(cudaMalloc(&d50b, 2 * sizeof(float) * gpu.pipeline_length(n8)), "cudaMalloc");
        check(cudaMemcpy(d50, in.data.data(), 2 * sizeof(float) * n, cudaMemcpyHostToDevice), "copy");
        cudaEvent_t t0, t1, t2;
        check(cudaEventCreate(&t0), "cudaEventCreate");
        check(cudaEventCreate(&t1), "cudaEventCreate");
        check(cudaEventCreate(&t2), "cudaEventCreate");
        for (int i = 0; i < 10; ++i) {  // warm-up
            gpu.to_dpd_device(d50, n, d8);
            gpu.to_pipeline_device(d8, n8, d50b);
        }
        float ms_down = 0, ms_up = 0;
        for (int i = 0; i < reps; ++i) {
            check(cudaEventRecord(t0), "cudaEventRecord");
            gpu.to_dpd_device(d50, n, d8);
            check(cudaEventRecord(t1), "cudaEventRecord");
            gpu.to_pipeline_device(d8, n8, d50b);
            check(cudaEventRecord(t2), "cudaEventRecord");
            check(cudaEventSynchronize(t2), "resample_kernel");
            float a = 0, b = 0;
            check(cudaEventElapsedTime(&a, t0, t1), "cudaEventElapsedTime");
            check(cudaEventElapsedTime(&b, t1, t2), "cudaEventElapsedTime");
            ms_down += a;
            ms_up += b;
        }
        std::printf("resampler_check: GPU per %d-sample frame: 50 -> 8 %.3f ms, 8 -> 50 %.3f ms (%d runs)\n", n,
                    ms_down / reps, ms_up / reps, reps);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaEventDestroy(t2);
        cudaFree(d50);
        cudaFree(d8);
        cudaFree(d50b);

        std::printf("resampler_check: tolerance %.0e: %s\n", tol, ok ? "OK" : "FAILED");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "resampler_check: %s\n", e.what());
        return 1;
    }
}
