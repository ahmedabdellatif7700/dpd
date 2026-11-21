// Internal helpers of the CUDA sources (not part of the library's interface).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#include "dpd/device_buffer.h"

namespace dpd::cuda {

// throws std::runtime_error naming the call on a CUDA error
inline void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// count elements of GPU memory, freed by the DeviceBuffer
template <typename T>
DeviceBuffer<T> device_alloc(std::size_t count, const char* what) {
    T* p = nullptr;
    check(cudaMalloc(&p, count * sizeof(T)), what);
    return DeviceBuffer<T>(p);
}

}  // namespace dpd::cuda
