// golden_check: the C++ model (cpu_model) against the PyTorch reference's golden file.
//
//   golden_check <weights.dpdt> <golden.dpdt> [--tol 1e-5]
//
// Runs the golden frame through the model, compares the whole output and, for each probed sample,
// every intermediate (embedding, each layer, final norm, output). Each error is the largest
// absolute difference over the RMS of the reference's values. Exit 0 if all are within --tol.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "dpd/cpu_model.h"
#include "dpd/tensor_file.h"
#include "dpd/weights.h"

namespace {

// max |a - b| / rms(b)
double rel_err(const float* a, const float* b, std::size_t n) {
    double m = 0, p = 0;
    for (std::size_t i = 0; i < n; ++i) {
        m = std::fmax(m, std::fabs(static_cast<double>(a[i]) - b[i]));
        p += static_cast<double>(b[i]) * b[i];
    }
    return p > 0 ? m / std::sqrt(p / static_cast<double>(n)) : m;
}

}  // namespace

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
        const auto g = dpd::read_tensor_file(argv[2]);
        const auto& in = dpd::find(g, "input");
        const auto& ref = dpd::find(g, "output");
        const auto& idx = dpd::find(g, "probe_index");
        const std::size_t n = in.dims.at(0);
        const auto& c = w.cfg;
        std::printf("golden_check: %s: window %d, width %d, %d heads, %d layers, FF %d, %zu parameters, scale %.6g\n",
                    argv[1], c.window, c.d_model, c.n_heads, c.n_layers, c.d_ff, w.n_params(), w.scale);

        std::vector<float> y(2 * n);
        const auto t0 = std::chrono::steady_clock::now();
        dpd::predistort(w, in.data.data(), n, y.data());
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double e_out = rel_err(y.data(), ref.data.data(), y.size());
        bool ok = e_out <= tol;
        std::printf("golden_check: frame of %zu samples: output error %.2e  (%.3f s on one CPU core, %.0f samples/s)\n",
                    n, e_out, dt, n / dt);

        // the probes, stage by stage
        const std::size_t T = c.window, d = c.d_model;
        std::vector<float> win(2 * T);
        double worst[8] = {0};  // embed, layer0.., final_norm, out
        const int n_stages = c.n_layers + 3;
        for (std::size_t p = 0; p < idx.size(); ++p) {
            const auto i = static_cast<std::size_t>(idx.data[p]);
            dpd::Trace tr;
            float out[2];
            dpd::window_at(c, in.data.data(), n, i, win.data());
            dpd::forward_window(w, win.data(), out, &tr);
            std::vector<double> e;
            e.push_back(rel_err(tr.embed.data(), &dpd::find(g, "probe.embed").data[p * T * d], T * d));
            for (int l = 0; l < c.n_layers; ++l)
                e.push_back(rel_err(tr.layer[l].data(),
                                    &dpd::find(g, "probe.layer" + std::to_string(l)).data[p * T * d], T * d));
            e.push_back(rel_err(tr.final_norm.data(), &dpd::find(g, "probe.final_norm").data[p * d], d));
            e.push_back(rel_err(tr.out, &dpd::find(g, "probe.out").data[p * 2], 2));
            std::printf("golden_check: probe %5zu:", i);
            for (int s = 0; s < n_stages; ++s) {
                std::printf(" %.1e", e[s]);
                worst[s] = std::fmax(worst[s], e[s]);
                ok = ok && e[s] <= tol;
            }
            std::printf("\n");
        }
        std::printf("golden_check: worst per stage (embed");
        for (int l = 0; l < c.n_layers; ++l) std::printf(", layer%d", l);
        std::printf(", final_norm, out):");
        for (int s = 0; s < n_stages; ++s) std::printf(" %.1e", worst[s]);
        std::printf("\ngolden_check: tolerance %.0e: %s\n", tol, ok ? "OK" : "FAILED");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "golden_check: %s\n", e.what());
        return 1;
    }
}
