// The check shared by golden_check (CpuModel) and cuda_check (GpuModel): a model against the
// reference's golden file (docs/dpd/FORMAT.md). The whole frame's output, then every probed
// intermediate (embedding, each layer, final norm, output before the scale). An error is the
// largest absolute difference over the RMS of the reference's values.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "dpd/tensor_file.h"
#include "dpd/weights.h"

namespace dpd::check {

// max |a - b| / rms(b)
inline double rel_err(const float* a, const float* b, std::size_t n) {
    double m = 0, p = 0;
    for (std::size_t i = 0; i < n; ++i) {
        m = std::fmax(m, std::fabs(static_cast<double>(a[i]) - b[i]));
        p += static_cast<double>(b[i]) * b[i];
    }
    return p > 0 ? m / std::sqrt(p / static_cast<double>(n)) : m;
}

struct Golden {
    std::vector<Tensor> t;
    const Tensor *input, *output;
    std::vector<int> probes;
    int n;

    explicit Golden(const std::string& path) : t(read_tensor_file(path)) {
        input = &find(t, "input");
        output = &find(t, "output");
        for (float v : find(t, "probe_index").data) probes.push_back(static_cast<int>(v));
        n = static_cast<int>(input->dims.at(0));
    }
};

// Runs the model on the golden frame into y, compares the output and the probes, prints the
// errors with the given tag; true if all are within tol.
template <typename Model>
bool against_golden(const char* tag, const Model& model, const Config& c, const Golden& g, double tol,
                    std::vector<float>& y) {
    y.assign(2 * static_cast<std::size_t>(g.n), 0.0f);
    model.predistort(g.input->data.data(), y.data(), g.n);
    const double e_out = rel_err(y.data(), g.output->data.data(), y.size());
    bool ok = e_out <= tol;
    std::printf("%s: frame of %d samples: output error %.2e against the reference\n", tag, g.n, e_out);

    const auto tr = model.trace(g.input->data.data(), g.n, g.probes);
    const std::size_t T = c.window, d = c.d_model, per = model.trace_floats_per_probe();
    std::vector<std::string> names{"embed"};
    for (int l = 0; l < c.n_layers; ++l) names.push_back("layer" + std::to_string(l));
    std::vector<double> worst(names.size() + 2, 0.0);
    for (std::size_t p = 0; p < g.probes.size(); ++p) {
        const float* t = &tr[p * per];
        std::vector<double> e;
        for (std::size_t s = 0; s < names.size(); ++s)
            e.push_back(rel_err(t + s * T * d, &find(g.t, "probe." + names[s]).data[p * T * d], T * d));
        e.push_back(rel_err(t + names.size() * T * d, &find(g.t, "probe.final_norm").data[p * d], d));
        e.push_back(rel_err(t + names.size() * T * d + d, &find(g.t, "probe.out").data[p * 2], 2));
        std::printf("%s: probe %5d:", tag, g.probes[p]);
        for (std::size_t s = 0; s < e.size(); ++s) {
            std::printf(" %.1e", e[s]);
            worst[s] = std::max(worst[s], e[s]);
            ok = ok && e[s] <= tol;
        }
        std::printf("\n");
    }
    std::printf("%s: worst per stage (", tag);
    for (const auto& s : names) std::printf("%s, ", s.c_str());
    std::printf("final_norm, out):");
    for (double v : worst) std::printf(" %.1e", v);
    std::printf("\n");
    return ok;
}

}  // namespace dpd::check
