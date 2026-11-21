// The dpd stage (dpd_stage.h): docs/testbed/jetson/STAGE.md, decisions 1a, 2b and 3b.
#include "dpd/dpd_stage.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "dpd/cpu_model.h"
#include "dpd/cpu_resampler.h"
#include "dpd/dpd_chain.h"
#include "dpd/gpu_model.h"
#include "dpd/gpu_resampler.h"
#include "dpd/resampler.h"
#include "dpd/weights.h"

namespace dpd {

using nlohmann::json;

// the model and the resampler on one device, behind one call: x[n][2] -> y[n][2] at 50 per symbol
struct DpdStage::Engine {
    std::unique_ptr<CpuModel> cpu_model;
    std::unique_ptr<CpuResampler> cpu_resampler;
    std::unique_ptr<DpdChain<CpuModel, CpuResampler>> cpu_chain;
    std::unique_ptr<GpuModel> gpu_model;
    std::unique_ptr<GpuResampler> gpu_resampler;
    std::unique_ptr<DpdChain<GpuModel, GpuResampler>> gpu_chain;

    Engine(const Weights& w, const ResamplerFilter& f, const std::string& device) {
        if (device == "cuda") {
            gpu_model = std::make_unique<GpuModel>(w);
            gpu_resampler = std::make_unique<GpuResampler>(f);
            gpu_chain = std::make_unique<DpdChain<GpuModel, GpuResampler>>(*gpu_model, *gpu_resampler);
        } else if (device == "cpu") {
            cpu_model = std::make_unique<CpuModel>(w);
            cpu_resampler = std::make_unique<CpuResampler>(f);
            cpu_chain = std::make_unique<DpdChain<CpuModel, CpuResampler>>(*cpu_model, *cpu_resampler);
        } else {
            throw std::runtime_error("dpd_bb: AROF_DPD_DEVICE must be cuda or cpu, is " + device);
        }
    }
    void run(const float* x, int n, float* y) const {
        if (gpu_chain) gpu_chain->run(x, n, y);
        else cpu_chain->run(x, n, y);
    }
};

namespace {

// sample i of a BASEBAND frame (complex128 or complex64), I or Q
double value(const arof::Frame& f, std::size_t i) {
    if (f.dtype() == arof::DType::c128) {
        double v;
        std::memcpy(&v, f.payload.data() + 8 * i, 8);
        return v;
    }
    float v;
    std::memcpy(&v, f.payload.data() + 4 * i, 4);
    return v;
}

}  // namespace

DpdStage::DpdStage(std::string models_dir, std::string device)
    : models_dir_(std::move(models_dir)), device_(std::move(device)) {}

DpdStage::~DpdStage() = default;

void DpdStage::configure(const json& config, json& entry) {
    const json none = json::object();
    const json& d = config.contains("dpd") && config["dpd"].contains("bb") ? config["dpd"]["bb"] : none;
    const std::string mode = d.value("mode", "bypass");
    if (mode != "bypass" && mode != "dpd")
        throw std::runtime_error("dpd_bb: dpd.bb.mode must be \"bypass\" or \"dpd\", is \"" + mode + "\"");
    entry["mode"] = mode;
    const bool was_on = dpd_on_;
    dpd_on_ = mode == "dpd";
    held_.clear();
    if (!dpd_on_) return;

    // what co_optics reads to compute kappa (link_stages.cpp, LinkStage::read_config)
    const json& st = config.at("settings");
    const long sps = std::lround(st.at("tx").at("fs").get<double>() / st.at("tx").at("rs").get<double>());
    if (sps != 50)
        throw std::runtime_error("dpd_bb: the DPD's resampler is built for 50 samples per symbol; the settings give " +
                                 std::to_string(sps));
    vpp_ = st.at("link").at("vpp_mv").get<double>() * 1e-3;
    drive_scale_ = 1.0;  // default_link's, unless the settings override stages.iq.drive_scale
    if (st.contains("stages") && st["stages"].contains("iq") && st["stages"]["iq"].contains("drive_scale"))
        drive_scale_ = st["stages"]["iq"]["drive_scale"].get<double>();
    block_samples_ = config.at("block").at("block_symbols").get<std::size_t>() * static_cast<std::size_t>(sps);

    const std::string name = d.value("model", "");
    if (name.empty()) throw std::runtime_error("dpd_bb: dpd.bb.model must name the weights file (in " + models_dir_ + ")");
    if (!engine_ || name != model_name_) {
        engine_ = std::make_unique<Engine>(load_weights(models_dir_ + "/" + name + ".dpdt"),
                                           load_resampler(models_dir_ + "/resampler.dpdt"), device_);
        model_name_ = name;
    }
    if (!was_on || !stream_) {  // a new stream; a live change keeps the running one
        stream_ = std::make_unique<FrameStream>([this](const float* x, int n, float* y) {
            // the model acts on the drive in volts: DPD(kappa s) / kappa
            const float k = static_cast<float>(kappa_);
            std::vector<float> v(x, x + 2 * static_cast<std::size_t>(n));
            for (float& a : v) a *= k;
            engine_->run(v.data(), n, y);
            for (int i = 0; i < 2 * n; ++i) y[i] /= k;
        });
        base_ = -1;
        pushed_ = 0;
    }
    entry["model"] = name;
    entry["device"] = device_;
    entry["dpd_sps"] = 8;
    entry["hold_samples"] = FrameStream::kHold;
}

void DpdStage::push(const arof::Frame& in, std::vector<float>& y, int64_t& first) {
    const int n = static_cast<int>(in.h.n_samples);
    const auto at = static_cast<int64_t>(in.h.first_sample);
    if (base_ < 0) base_ = at;
    if (at != base_ + pushed_)
        throw std::runtime_error("dpd_bb: a BASEBAND frame at sample " + std::to_string(at) + ", expected " +
                                 std::to_string(base_ + pushed_) + ": the drive must be contiguous");
    std::vector<float> x(2 * static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(value(in, i));
    first = stream_->push(x.data(), n, y);
    pushed_ += n;
}

arof::Frame DpdStage::output(const arof::Frame& like, int64_t first, const std::vector<float>& y) const {
    arof::Frame o;
    o.h.signal = static_cast<uint8_t>(arof::Signal::baseband);
    o.h.dtype = like.h.dtype;
    o.h.words_per_sample = 1;
    o.h.n_samples = static_cast<uint32_t>(y.size() / 2);
    o.h.first_sample = static_cast<uint64_t>(base_ + first);
    if (like.dtype() == arof::DType::c128) {
        o.payload.resize(8 * y.size());
        for (std::size_t i = 0; i < y.size(); ++i) {
            const double v = y[i];
            std::memcpy(o.payload.data() + 8 * i, &v, 8);
        }
    } else {
        o.payload.resize(4 * y.size());
        std::memcpy(o.payload.data(), y.data(), 4 * y.size());
    }
    return o;
}

void DpdStage::process(const arof::Frame& in, std::vector<arof::Frame>& out) {
    if (!dpd_on_) {  // bypass: the frame as it came
        out.push_back(in);
        return;
    }
    if (!accepts(in.dtype()) || in.h.words_per_sample != 1)
        throw std::runtime_error(std::string("dpd_bb: BASEBAND in ") + arof::dtype_name(in.dtype()) + " with " +
                                 std::to_string(in.h.words_per_sample) +
                                 " values per sample, which it cannot take (complex128 or complex64, 1 per sample)");

    if (in.h.flags & arof::kCalib) {
        held_.push_back(in);
        if (!(in.h.flags & arof::kCalibEnd)) return;
        // kappa over the block, the group's first block_samples_ samples, as co_optics takes it
        double smin = std::numeric_limits<double>::infinity(), smax = -smin;
        std::size_t seen = 0;
        for (const arof::Frame& f : held_)
            for (std::size_t i = 0; i < f.h.n_samples && seen < block_samples_; ++i, ++seen) {
                smin = std::min(smin, value(f, 2 * i));
                smax = std::max(smax, value(f, 2 * i));
            }
        if (seen != block_samples_)
            throw std::logic_error("dpd_bb: the calibration block is not the size the CONFIG frame gives");
        kappa_ = drive_scale_ * vpp_ / (smax - smin);
        have_kappa_ = new_kappa_ = true;
        // the group through the stream, and out whole: its last frame takes what is held
        for (std::size_t j = 0; j < held_.size(); ++j) {
            std::vector<float> y;
            int64_t first = 0;
            push(held_[j], y, first);  // first: where its output starts, also when it emitted nothing
            if (j + 1 == held_.size()) stream_->flush(y);  // continues where the push stopped
            out.push_back(output(held_[j], first, y));
        }
        held_.clear();
        return;
    }
    if (!have_kappa_) throw std::logic_error("dpd_bb: BASEBAND before the first calibration group");
    std::vector<float> y;
    int64_t first = 0;
    push(in, y, first);
    out.push_back(output(in, first, y));
}

bool DpdStage::take_calibration(json& constants) {
    if (!new_kappa_) return false;
    new_kappa_ = false;
    constants = {{"kappa", kappa_}};
    return true;
}

}  // namespace dpd
