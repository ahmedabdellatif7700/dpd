// The transformer DPD on the GPU: one fused CUDA kernel per frame (docs/twin/README.md).
// The same computation as CpuModel (cpu_model.h, docs/dpd/METHODOLOGY.md), float32, with the same
// interface; this header has no CUDA types, so plain C++ code can use it.
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "dpd/device_buffer.h"
#include "dpd/weights.h"

namespace dpd {

namespace detail {
struct KernelParams;  // the kernel's pointers into the weights (gpu_model.cu)
}  // namespace detail

class GpuModel {
public:
    // Uploads the weights (transposed for the kernel). Throws std::runtime_error if the model is
    // not the one the kernel is built for (window 32, width 32, 4 heads, 2 layers, FF 64) or on a
    // CUDA error.
    explicit GpuModel(const Weights& w);
    ~GpuModel();
    GpuModel(const GpuModel&) = delete;
    GpuModel& operator=(const GpuModel&) = delete;

    // x[n][2] -> y[n][2] (I/Q, the drive's units), the frame alone (zeros beyond its ends):
    // what the reference's predistort() does. Host in, host out, synchronous.
    void predistort(const float* x, float* y, int n) const;

    // The intermediates of the given output samples, in the golden file's layout, one probe after
    // the other: embed [T][d], layer0 .. layer<L-1> [T][d], final_norm [d], out [2] (before scale).
    // Throws if a probe lies outside 0 .. n-1.
    std::vector<float> trace(const float* x, int n, const std::vector<int>& probes) const;
    std::size_t trace_floats_per_probe() const;

    // GPU only: device in, device out, x[n][2] -> y[n][2], asynchronous on stream (a
    // cudaStream_t, nullptr for the default stream). What the stage calls.
    void forward(const float* d_x, float* d_y, int n, void* stream = nullptr) const;

private:
    void ensure_io(int n) const;
    DeviceBuffer<float> weights_;                    // all tensors, one buffer
    std::unique_ptr<detail::KernelParams> params_;   // the kernel's pointers into it
    // working buffers of predistort() and trace(), grown as needed
    mutable DeviceBuffer<float> x_, y_;
    mutable int io_n_ = 0;
};

}  // namespace dpd
