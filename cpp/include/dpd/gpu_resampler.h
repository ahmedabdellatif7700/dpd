// The resampler (resampler.h) on the GPU: one CUDA thread per output sample. The same computation
// as CpuResampler, float32, with the same interface; plus device-memory versions for the stage.
#pragma once

#include "dpd/device_buffer.h"
#include "dpd/resampler.h"

namespace dpd {

class GpuResampler {
public:
    explicit GpuResampler(const ResamplerFilter& f);  // uploads the taps

    // host in, host out, synchronous: as CpuResampler
    void to_dpd(const float* x, int n, float* y) const;
    void to_pipeline(const float* x, int n, float* y) const;

    int dpd_length(int n) const { return resampled_length(n, up_, down_); }
    int pipeline_length(int n) const { return resampled_length(n, down_, up_); }

    // GPU only: device in, device out, asynchronous on stream (a cudaStream_t, nullptr: default)
    void to_dpd_device(const float* d_x, int n, float* d_y, void* stream = nullptr) const;
    void to_pipeline_device(const float* d_x, int n, float* d_y, void* stream = nullptr) const;

private:
    void run(const float* d_x, int n, int up, int down, float* d_y, void* stream) const;
    void run_host(const float* x, int n, int up, int down, float* y) const;
    DeviceBuffer<float> h_;
    int taps_ = 0, up_ = 0, down_ = 0;
    mutable DeviceBuffer<float> x_, y_;  // working buffers of the host versions, grown as needed
    mutable int x_n_ = 0, y_n_ = 0;
};

}  // namespace dpd
