#include "dpd/weights.h"

#include <set>
#include <stdexcept>

#include "dpd/tensor_file.h"

namespace dpd {

namespace {

class Taker {
public:
    explicit Taker(const std::vector<Tensor>& t) : t_(t) {}

    std::vector<float> take(const std::string& name, std::vector<uint32_t> dims) {
        const Tensor& x = find(t_, name);
        if (x.dims != dims) {
            Tensor want{name, dims, {}};
            throw std::runtime_error("tensor '" + name + "' has shape " + x.shape() + ", expected " + want.shape());
        }
        used_.insert(name);
        return x.data;
    }
    void check_all_used() const {
        for (const auto& x : t_)
            if (!used_.count(x.name)) throw std::runtime_error("unexpected tensor '" + x.name + "'");
    }

private:
    const std::vector<Tensor>& t_;
    std::set<std::string> used_;
};

int as_int(float v, const char* what) {
    const int i = static_cast<int>(v);
    if (static_cast<float>(i) != v || i <= 0) throw std::runtime_error(std::string("bad config value for ") + what);
    return i;
}

}  // namespace

std::size_t Weights::n_params() const {
    std::size_t n = pos.size() + embed_w.size() + embed_b.size() + norm_w.size() + norm_b.size() +
                    head_w.size() + head_b.size();
    for (const auto& l : layers)
        n += l.norm1_w.size() + l.norm1_b.size() + l.in_w.size() + l.in_b.size() + l.out_w.size() +
             l.out_b.size() + l.norm2_w.size() + l.norm2_b.size() + l.ff1_w.size() + l.ff1_b.size() +
             l.ff2_w.size() + l.ff2_b.size();
    return n;
}

Weights load_weights(const std::string& path) {
    const auto tensors = read_tensor_file(path);
    Taker t(tensors);
    Weights w;

    const auto c = t.take("config", {5});
    w.cfg = {as_int(c[0], "window"), as_int(c[1], "d_model"), as_int(c[2], "n_heads"), as_int(c[3], "n_layers"),
             as_int(c[4], "d_ff")};
    if (w.cfg.d_model % w.cfg.n_heads != 0) throw std::runtime_error("d_model is not a multiple of n_heads");
    w.scale = t.take("scale", {1})[0];

    const auto T = static_cast<uint32_t>(w.cfg.window), d = static_cast<uint32_t>(w.cfg.d_model),
               f = static_cast<uint32_t>(w.cfg.d_ff);
    w.pos = t.take("pos", {T, d});
    w.embed_w = t.take("embed.weight", {d, 2});
    w.embed_b = t.take("embed.bias", {d});
    w.layers.resize(w.cfg.n_layers);
    for (int i = 0; i < w.cfg.n_layers; ++i) {
        const std::string p = "encoder.layers." + std::to_string(i) + ".";
        Layer& l = w.layers[i];
        l.in_w = t.take(p + "self_attn.in_proj_weight", {3 * d, d});
        l.in_b = t.take(p + "self_attn.in_proj_bias", {3 * d});
        l.out_w = t.take(p + "self_attn.out_proj.weight", {d, d});
        l.out_b = t.take(p + "self_attn.out_proj.bias", {d});
        l.ff1_w = t.take(p + "linear1.weight", {f, d});
        l.ff1_b = t.take(p + "linear1.bias", {f});
        l.ff2_w = t.take(p + "linear2.weight", {d, f});
        l.ff2_b = t.take(p + "linear2.bias", {d});
        l.norm1_w = t.take(p + "norm1.weight", {d});
        l.norm1_b = t.take(p + "norm1.bias", {d});
        l.norm2_w = t.take(p + "norm2.weight", {d});
        l.norm2_b = t.take(p + "norm2.bias", {d});
    }
    w.norm_w = t.take("norm.weight", {d});
    w.norm_b = t.take("norm.bias", {d});
    w.head_w = t.take("head.weight", {2, d});
    w.head_b = t.take("head.bias", {2});
    t.check_all_used();
    return w;
}

}  // namespace dpd
