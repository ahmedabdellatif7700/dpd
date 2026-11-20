// golden_check: the C++ model on the CPU (CpuModel) against the PyTorch reference's golden file.
// The twin of cuda_check (GpuModel); both use the same check (model_check.h).
//
//   golden_check <weights.dpdt> <golden.dpdt> [--tol 1e-5]
//
// Exit 0 if every error is within --tol. Also times the frame on one CPU core.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

#include "dpd/cpu_model.h"
#include "model_check.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: golden_check <weights.dpdt> <golden.dpdt> [--tol 1e-5]\n");
        return 2;
    }
    double tol = 1e-5;
    for (int i = 3; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], "--tol")) tol = std::atof(argv[i + 1]);

    try {
        const dpd::Weights w = dpd::load_weights(argv[1]);
        const dpd::check::Golden g(argv[2]);
        const auto& c = w.cfg;
        std::printf("golden_check: %s: window %d, width %d, %d heads, %d layers, FF %d, %zu parameters, scale %.6g\n",
                    argv[1], c.window, c.d_model, c.n_heads, c.n_layers, c.d_ff, w.n_params(), w.scale);

        const dpd::CpuModel cpu(w);
        std::vector<float> y;
        const auto t0 = std::chrono::steady_clock::now();
        bool ok = dpd::check::against_golden("golden_check", cpu, c, g, tol, y);
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("golden_check: %.3f s for the frame and the probes on one CPU core (%.0f samples/s)\n", dt,
                    g.n / dt);
        std::printf("golden_check: tolerance %.0e: %s\n", tol, ok ? "OK" : "FAILED");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "golden_check: %s\n", e.what());
        return 1;
    }
}
