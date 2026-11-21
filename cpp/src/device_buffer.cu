// DeviceBuffer's deleter (device_buffer.h)
#include "dpd/device_buffer.h"

#include <cuda_runtime.h>

namespace dpd::detail {

void CudaFree::operator()(void* p) const noexcept { cudaFree(p); }

}  // namespace dpd::detail
