// GPU memory owned by a unique_ptr that calls cudaFree, so nothing leaks or is freed twice, also when
// an exception is thrown. No CUDA types here: plain C++ code can hold one.
#pragma once

#include <memory>

namespace dpd {

namespace detail {
struct CudaFree {
    void operator()(void* p) const noexcept;  // cudaFree (gpu_model.cu)
};
}  // namespace detail

template <typename T>
using DeviceBuffer = std::unique_ptr<T, detail::CudaFree>;

}  // namespace dpd
