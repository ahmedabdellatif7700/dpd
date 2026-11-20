// cuda_check: the fused CUDA kernel (GpuModel) against the PyTorch reference's golden file and
// against the C++ model (CpuModel). The twin of golden_check; both use the same check
// (model_check.h).
//
//   cuda_check <weights.dpdt> <golden.dpdt> [--tol 1e-5] [--reps 200]
//
// Exit 0 if every error is within --tol. Then a first timing (part 5 measures it fully): the
// kernel alone on one frame already on the GPU, with CUDA events, over --reps runs.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

#include "dpd/cpu_model.h"
#include "dpd/gpu_model.h"
#include "model_check.h"

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: cuda_check <weights.dpdt> <golden.dpdt> [--tol 1e-5] [--reps 200]\n");
        return 2;
    }
    double tol = 1e-5;
    int reps = 200;
    for (int i = 3; i + 1 < argc; ++i) {
        if (!std::strcmp(argv[i], "--tol")) tol = std::atof(argv[i + 1]);
        if (!std::strcmp(argv[i], "--reps")) reps = std::atoi(argv[i + 1]);
    }

    try {
        const dpd::Weights w = dpd::load_weights(argv[1]);
        const dpd::check::Golden g(argv[2]);
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
        std::printf("cuda_check: %s on \"%s\" (CC %d.%d, %d SMs)\n", argv[1], prop.name, prop.major, prop.minor,
                    prop.multiProcessorCount);

        const dpd::GpuModel gpu(w);
        std::vector<float> y_gpu, y_cpu(2 * static_cast<std::size_t>(g.n));
        bool ok = dpd::check::against_golden("cuda_check", gpu, w.cfg, g, tol, y_gpu);

        // and against the twin's own reference, the C++ model
        dpd::CpuModel(w).predistort(g.input->data.data(), y_cpu.data(), g.n);
        const double e_cpu = dpd::check::rel_err(y_gpu.data(), y_cpu.data(), y_gpu.size());
        ok = ok && e_cpu <= tol;
        std::printf("cuda_check: output error %.2e against the C++ model (CpuModel)\n", e_cpu);

        // a first timing: the kernel alone, the frame already on the GPU
        const int n = g.n;
        float *d_x = nullptr, *d_y = nullptr;
        check(cudaMalloc(&d_x, 2 * sizeof(float) * n), "cudaMalloc");
        check(cudaMalloc(&d_y, 2 * sizeof(float) * n), "cudaMalloc");
        check(cudaMemcpy(d_x, g.input->data.data(), 2 * sizeof(float) * n, cudaMemcpyHostToDevice), "copy");
        for (int i = 0; i < 10; ++i) gpu.forward(d_x, d_y, n);  // warm-up
        cudaEvent_t t0, t1;
        check(cudaEventCreate(&t0), "cudaEventCreate");
        check(cudaEventCreate(&t1), "cudaEventCreate");
        check(cudaEventRecord(t0), "cudaEventRecord");
        for (int i = 0; i < reps; ++i) gpu.forward(d_x, d_y, n);
        check(cudaEventRecord(t1), "cudaEventRecord");
        check(cudaEventSynchronize(t1), "timing");
        float ms = 0;
        check(cudaEventElapsedTime(&ms, t0, t1), "cudaEventElapsedTime");
        const double per_frame = ms / reps;
        std::printf("cuda_check: kernel: %.3f ms per %d-sample frame (%d runs), %.0f samples/s\n", per_frame, n, reps,
                    n / (per_frame * 1e-3));
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaFree(d_x);
        cudaFree(d_y);

        std::printf("cuda_check: tolerance %.0e: %s\n", tol, ok ? "OK" : "FAILED");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "cuda_check: %s\n", e.what());
        return 1;
    }
}
