// gpu_check: is the Jetson's GPU usable from the dpd stage's container?
// Prints the device, runs a kernel (y = a*x + y on 6400 complex samples, one BASEBAND frame) and
// checks it against the CPU. Exit 0 if all is well. Replaced as the container's command by the
// dpd stage in step 3; kept as the image's self-test.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#define CHECK(call)                                                                       \
    do {                                                                                  \
        cudaError_t e = (call);                                                           \
        if (e != cudaSuccess) {                                                           \
            std::fprintf(stderr, "gpu_check: %s: %s\n", #call, cudaGetErrorString(e));    \
            return 1;                                                                     \
        }                                                                                 \
    } while (0)

__global__ void axpy(int n, float a, const float2* x, float2* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = make_float2(a * x[i].x + y[i].x, a * x[i].y + y[i].y);
}

int main() {
    int rt = 0, drv = 0, n_dev = 0;
    CHECK(cudaRuntimeGetVersion(&rt));
    CHECK(cudaDriverGetVersion(&drv));
    CHECK(cudaGetDeviceCount(&n_dev));
    cudaDeviceProp p{};
    CHECK(cudaGetDeviceProperties(&p, 0));
    std::printf("gpu_check: CUDA runtime %d.%d, driver %d.%d, %d device(s)\n", rt / 1000, rt % 1000 / 10,
                drv / 1000, drv % 1000 / 10, n_dev);
    std::printf("gpu_check: device 0 \"%s\", compute capability %d.%d, %d SMs, %.1f GB, integrated %d\n", p.name,
                p.major, p.minor, p.multiProcessorCount, p.totalGlobalMem / 1e9, p.integrated);

    const int n = 6400;                              // one BASEBAND frame (128 symbols x 50)
    std::vector<float2> x(n), y(n), out(n);
    for (int i = 0; i < n; ++i) {
        x[i] = make_float2(std::sin(0.01f * i), std::cos(0.02f * i));
        y[i] = make_float2(0.5f, -0.25f);
    }
    float2 *dx, *dy;
    CHECK(cudaMalloc(&dx, n * sizeof(float2)));
    CHECK(cudaMalloc(&dy, n * sizeof(float2)));
    CHECK(cudaMemcpy(dx, x.data(), n * sizeof(float2), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dy, y.data(), n * sizeof(float2), cudaMemcpyHostToDevice));
    axpy<<<(n + 255) / 256, 256>>>(n, 2.0f, dx, dy);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(out.data(), dy, n * sizeof(float2), cudaMemcpyDeviceToHost));
    cudaFree(dx);
    cudaFree(dy);
    double err = 0;
    for (int i = 0; i < n; ++i)
        err = std::fmax(err, std::fabs(out[i].x - (2.0f * x[i].x + y[i].x)) + std::fabs(out[i].y - (2.0f * x[i].y + y[i].y)));
    std::printf("gpu_check: kernel on %d samples, max error %.3g: %s\n", n, err, err < 1e-6 ? "OK" : "FAILED");
    return err < 1e-6 ? 0 : 1;
}
