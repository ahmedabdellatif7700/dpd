// dpd_bench: the twin's latency and throughput per frame on the GPU, as the stage will run it.
//
//   dpd_bench <weights.dpdt> [--frame 6400] [--frames 1000] [--period-ms 0] [--memory pinned|pageable]
//
// Each frame: copy to the GPU, the kernel, copy back, wait (one CUDA stream). The latency of a
// frame is the wall time of those four, taken on the host (steady_clock); the kernel's own time
// is taken with CUDA events. --period-ms paces the frames as they arrive in the pipeline (a frame
// starts every period; 0: back to back). Reports the median, 99th percentile and largest
// latency, and the throughput. The data is a fixed pseudo-random drive (the timing does not
// depend on it).
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "dpd/gpu_model.h"
#include "dpd/weights.h"

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    const std::size_t i = std::min(v.size() - 1, static_cast<std::size_t>(p / 100.0 * static_cast<double>(v.size())));
    return v[i];
}

}  // namespace

int main(int argc, char** argv) {
    int frame = 6400, frames = 1000;
    double period_ms = 0;
    std::string memory = "pinned";
    const char* weights = nullptr;
    for (int i = 1; i < argc; ++i) {
        const bool more = i + 1 < argc;
        if (!std::strcmp(argv[i], "--frame") && more) frame = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--frames") && more) frames = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--period-ms") && more) period_ms = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--memory") && more) memory = argv[++i];
        else weights = argv[i];
    }
    if (!weights || frame <= 0 || frames <= 0 || (memory != "pinned" && memory != "pageable")) {
        std::fprintf(stderr,
                     "usage: dpd_bench <weights.dpdt> [--frame 6400] [--frames 1000] [--period-ms 0] "
                     "[--memory pinned|pageable]\n");
        return 2;
    }

    try {
        const dpd::GpuModel gpu(dpd::load_weights(weights));
        const std::size_t bytes = 2 * sizeof(float) * static_cast<std::size_t>(frame);

        // host buffers: pinned (page-locked, cudaMallocHost) or ordinary
        float *x = nullptr, *y = nullptr;
        std::vector<float> x_pageable, y_pageable;
        if (memory == "pinned") {
            check(cudaMallocHost(&x, bytes), "cudaMallocHost");
            check(cudaMallocHost(&y, bytes), "cudaMallocHost");
        } else {
            x_pageable.resize(2 * static_cast<std::size_t>(frame));
            y_pageable.resize(2 * static_cast<std::size_t>(frame));
            x = x_pageable.data();
            y = y_pageable.data();
        }
        std::mt19937 rng(2017);
        std::normal_distribution<float> g(0.0f, 0.371f / 1.41421356f);
        for (int i = 0; i < 2 * frame; ++i) x[i] = g(rng);

        float *d_x = nullptr, *d_y = nullptr;
        check(cudaMalloc(&d_x, bytes), "cudaMalloc");
        check(cudaMalloc(&d_y, bytes), "cudaMalloc");
        cudaStream_t stream;
        check(cudaStreamCreate(&stream), "cudaStreamCreate");
        cudaEvent_t k0, k1;
        check(cudaEventCreate(&k0), "cudaEventCreate");
        check(cudaEventCreate(&k1), "cudaEventCreate");

        auto one_frame = [&](float* kernel_ms) {
            check(cudaMemcpyAsync(d_x, x, bytes, cudaMemcpyHostToDevice, stream), "copy in");
            check(cudaEventRecord(k0, stream), "cudaEventRecord");
            gpu.forward(d_x, d_y, frame, stream);
            check(cudaEventRecord(k1, stream), "cudaEventRecord");
            check(cudaMemcpyAsync(y, d_y, bytes, cudaMemcpyDeviceToHost, stream), "copy out");
            check(cudaStreamSynchronize(stream), "dpd_forward");
            if (kernel_ms) check(cudaEventElapsedTime(kernel_ms, k0, k1), "cudaEventElapsedTime");
        };
        for (int i = 0; i < 20; ++i) one_frame(nullptr);  // warm-up

        using clock = std::chrono::steady_clock;
        std::vector<double> lat_ms(frames), ker_ms(frames);
        const auto t_start = clock::now();
        for (int i = 0; i < frames; ++i) {
            if (period_ms > 0)
                std::this_thread::sleep_until(t_start + std::chrono::duration<double, std::milli>(i * period_ms));
            const auto t0 = clock::now();
            float km = 0;
            one_frame(&km);
            lat_ms[i] = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
            ker_ms[i] = km;
        }
        const double total_s = std::chrono::duration<double>(clock::now() - t_start).count();

        double busy = 0;
        for (double v : lat_ms) busy += v;
        std::printf("dpd_bench: frame %d samples, %d frames, %s host memory, %s\n", frame, frames, memory.c_str(),
                    period_ms > 0 ? ("paced every " + std::to_string(period_ms) + " ms").c_str() : "back to back");
        std::printf("dpd_bench: latency (copy in, kernel, copy out) ms: median %.3f, p99 %.3f, max %.3f\n",
                    percentile(lat_ms, 50), percentile(lat_ms, 99), percentile(lat_ms, 100));
        std::printf("dpd_bench: kernel alone ms: median %.3f, p99 %.3f, max %.3f\n", percentile(ker_ms, 50),
                    percentile(ker_ms, 99), percentile(ker_ms, 100));
        std::printf("dpd_bench: copies and waiting, median %.3f ms\n", percentile(lat_ms, 50) - percentile(ker_ms, 50));
        std::printf("dpd_bench: throughput while busy %.0f samples/s; over the run %.0f samples/s\n",
                    frames * static_cast<double>(frame) / (busy * 1e-3), frames * static_cast<double>(frame) / total_s);

        cudaEventDestroy(k0);
        cudaEventDestroy(k1);
        cudaStreamDestroy(stream);
        cudaFree(d_x);
        cudaFree(d_y);
        if (memory == "pinned") {
            cudaFreeHost(x);
            cudaFreeHost(y);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "dpd_bench: %s\n", e.what());
        return 1;
    }
}
