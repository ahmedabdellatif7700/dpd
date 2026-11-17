// docs/dpd/METHODOLOGY.md section 2, in the same order and float32 throughout.
#include "dpd/cpu_model.h"

#include <algorithm>
#include <cmath>

namespace dpd {

namespace {

// y[o] = b[o] + sum_i W[o][i] x[i], for one token
void linear(const float* W, const float* b, const float* x, int in, int out, float* y) {
    for (int o = 0; o < out; ++o) {
        float acc = b[o];
        const float* row = W + static_cast<std::size_t>(o) * in;
        for (int i = 0; i < in; ++i) acc += row[i] * x[i];
        y[o] = acc;
    }
}

// layer norm over d features: biased variance, eps 1e-5
void layer_norm(const float* v, const float* w, const float* b, int d, float* y) {
    float mu = 0;
    for (int i = 0; i < d; ++i) mu += v[i];
    mu /= static_cast<float>(d);
    float var = 0;
    for (int i = 0; i < d; ++i) var += (v[i] - mu) * (v[i] - mu);
    var /= static_cast<float>(d);
    const float r = 1.0f / std::sqrt(var + 1e-5f);
    for (int i = 0; i < d; ++i) y[i] = (v[i] - mu) * r * w[i] + b[i];
}

// GELU, the exact form: 0.5 u (1 + erf(u / sqrt 2))
float gelu(float u) { return 0.5f * u * (1.0f + std::erf(u * 0.70710678118654752f)); }

}  // namespace

void window_at(const Config& c, const float* x, std::size_t n, std::size_t i, float* win) {
    const long T = c.window, ctr = c.centre();
    for (long t = 0; t < T; ++t) {
        const long k = static_cast<long>(i) - ctr + t;
        const bool in = k >= 0 && k < static_cast<long>(n);
        win[2 * t] = in ? x[2 * k] : 0.0f;
        win[2 * t + 1] = in ? x[2 * k + 1] : 0.0f;
    }
}

void forward_window(const Weights& w, const float* win, float* out, Trace* trace) {
    const Config& c = w.cfg;
    const int T = c.window, d = c.d_model, h = c.n_heads, dh = c.d_head(), f = c.d_ff, ctr = c.centre();
    const float inv_s = 1.0f / w.scale;
    const float att_scale = 1.0f / std::sqrt(static_cast<float>(dh));

    std::vector<float> z(2 * T), e(T * d), a(T * d), qkv(T * 3 * d), o(T * d), tmp(std::max(d, f)), hid(f),
        S(T);

    // input scale, embedding and position
    for (int t = 0; t < 2 * T; ++t) z[t] = win[t] * inv_s;
    for (int t = 0; t < T; ++t) {
        linear(w.embed_w.data(), w.embed_b.data(), &z[2 * t], 2, d, &e[t * d]);
        for (int i = 0; i < d; ++i) e[t * d + i] += w.pos[t * d + i];
    }
    if (trace) trace->embed = e;

    for (int l = 0; l < c.n_layers; ++l) {
        const Layer& L = w.layers[l];
        // attention block, pre-norm
        for (int t = 0; t < T; ++t) {
            layer_norm(&e[t * d], L.norm1_w.data(), L.norm1_b.data(), d, &a[t * d]);
            linear(L.in_w.data(), L.in_b.data(), &a[t * d], d, 3 * d, &qkv[t * 3 * d]);
        }
        for (int j = 0; j < h; ++j) {
            const int q0 = j * dh, k0 = d + j * dh, v0 = 2 * d + j * dh;
            for (int t = 0; t < T; ++t) {
                float m = -INFINITY;
                for (int u = 0; u < T; ++u) {
                    float s = 0;
                    for (int i = 0; i < dh; ++i) s += qkv[t * 3 * d + q0 + i] * qkv[u * 3 * d + k0 + i];
                    S[u] = s * att_scale;
                    m = std::max(m, S[u]);
                }
                float sum = 0;
                for (int u = 0; u < T; ++u) sum += (S[u] = std::exp(S[u] - m));
                for (int i = 0; i < dh; ++i) {
                    float acc = 0;
                    for (int u = 0; u < T; ++u) acc += S[u] * qkv[u * 3 * d + v0 + i];
                    o[t * d + q0 + i] = acc / sum;
                }
            }
        }
        for (int t = 0; t < T; ++t) {
            linear(L.out_w.data(), L.out_b.data(), &o[t * d], d, d, tmp.data());
            for (int i = 0; i < d; ++i) e[t * d + i] += tmp[i];
        }
        // feed-forward block, pre-norm
        for (int t = 0; t < T; ++t) {
            layer_norm(&e[t * d], L.norm2_w.data(), L.norm2_b.data(), d, &a[t * d]);
            linear(L.ff1_w.data(), L.ff1_b.data(), &a[t * d], d, f, hid.data());
            for (int i = 0; i < f; ++i) hid[i] = gelu(hid[i]);
            linear(L.ff2_w.data(), L.ff2_b.data(), hid.data(), f, d, tmp.data());
            for (int i = 0; i < d; ++i) e[t * d + i] += tmp[i];
        }
        if (trace) trace->layer.push_back(e);
    }

    // the centre token: final norm, head, the skip, the scale
    std::vector<float> n(d);
    layer_norm(&e[ctr * d], w.norm_w.data(), w.norm_b.data(), d, n.data());
    float y[2];
    linear(w.head_w.data(), w.head_b.data(), n.data(), d, 2, y);
    y[0] += z[2 * ctr];
    y[1] += z[2 * ctr + 1];
    if (trace) {
        trace->final_norm = n;
        trace->out[0] = y[0];
        trace->out[1] = y[1];
    }
    out[0] = y[0] * w.scale;
    out[1] = y[1] * w.scale;
}

void predistort(const Weights& w, const float* x, std::size_t n, float* y) {
    std::vector<float> win(2 * w.cfg.window);
    for (std::size_t i = 0; i < n; ++i) {
        window_at(w.cfg, x, n, i, win.data());
        forward_window(w, win.data(), &y[2 * i]);
    }
}

}  // namespace dpd
