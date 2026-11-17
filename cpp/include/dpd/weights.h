// The transformer DPD's configuration and weights, from a DPDT weights file
// (docs/dpd/FORMAT.md; names and shapes as in docs/dpd/METHODOLOGY.md section 2).
#pragma once

#include <string>
#include <vector>

namespace dpd {

struct Config {
    int window = 0;    // T, the tokens of one window
    int d_model = 0;   // d
    int n_heads = 0;   // h
    int n_layers = 0;  // L
    int d_ff = 0;      // f
    int centre() const { return window / 2; }
    int d_head() const { return d_model / n_heads; }
};

// One encoder layer (encoder.layers.<l>.*); matrices are row-major [out][in]
struct Layer {
    std::vector<float> norm1_w, norm1_b;  // [d]
    std::vector<float> in_w, in_b;        // [3d][d], [3d]: rows q, then k, then v
    std::vector<float> out_w, out_b;      // [d][d], [d]
    std::vector<float> norm2_w, norm2_b;  // [d]
    std::vector<float> ff1_w, ff1_b;      // [f][d], [f]
    std::vector<float> ff2_w, ff2_b;      // [d][f], [d]
};

struct Weights {
    Config cfg;
    float scale = 1.0f;
    std::vector<float> pos;                // [T][d]
    std::vector<float> embed_w, embed_b;   // [d][2], [d]
    std::vector<Layer> layers;             // [L]
    std::vector<float> norm_w, norm_b;     // [d]
    std::vector<float> head_w, head_b;     // [2][d], [2]

    std::size_t n_params() const;
};

// Loads and checks a weights file: every tensor present with the shape the config implies, no
// tensor left over. Throws std::runtime_error naming the tensor otherwise.
Weights load_weights(const std::string& path);

}  // namespace dpd
