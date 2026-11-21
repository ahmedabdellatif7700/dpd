// The dpd stage: the pipeline's dpd_bb on the Jetson (docs/testbed/jetson/STAGE.md), on the Pi's
// stage library (cpp/arof: the contract, the connections, the tap, the timing log are its).
//
// It consumes BASEBAND (the RRC waveform s at 50 samples per symbol) and produces BASEBAND; every
// other frame (REF, CONFIG, END) passes through. Its CONFIG section, top level "dpd": {"bb": {...}}:
//   "mode": "bypass"   every frame forwarded untouched, byte for byte (the default)
//   "mode": "dpd"      the drive predistorted:
//       - holds each calibration group (CALIB .. CALIB_END), computes the drive scale
//         kappa = drive_scale * vpp / (max Re s - min Re s) over its block, co_optics's formula, on
//         the undistorted signal, and writes it into its line of the "pipeline" section (decision 1a);
//       - predistorts the drive in volts: s -> DPD(kappa s) / kappa, so that co_optics, multiplying
//         by the same kappa, drives the IQ modulator with DPD(kappa s), as the MATLAB link does;
//       - DPD = 50 -> 8 samples per symbol, the model, 8 -> 50 (decision 2b), through a stream that
//         holds back 250 samples per frame (decision 3b); the calibration group leaves whole;
//       - "model": the weights file's name in AROF_MODELS_DIR (default /models), without .dpdt;
//         the resampler is AROF_MODELS_DIR/resampler.dpdt; AROF_DPD_DEVICE: cuda (default) or cpu.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "dpd/frame_stream.h"
#include "stage/stage.h"

namespace dpd {

class DpdStage : public arof::Stage {
public:
    DpdStage(std::string models_dir, std::string device);
    ~DpdStage() override;

    arof::Signal in_signal() const override { return arof::Signal::baseband; }
    arof::Signal out_signal() const override { return arof::Signal::baseband; }
    // complex128 and complex64 as they come (bypass forwards them untouched; dpd keeps the dtype)
    bool accepts(arof::DType t) const override { return t == arof::DType::c128 || t == arof::DType::c64; }
    void configure(const nlohmann::json& config, nlohmann::json& entry) override;
    void process(const arof::Frame& in, std::vector<arof::Frame>& out) override;
    bool calibrates() const override { return dpd_on_; }
    bool take_calibration(nlohmann::json& constants) override;

    double kappa() const { return kappa_; }

private:
    struct Engine;                             // the model and the resampler on the CPU or the GPU
    arof::Frame output(const arof::Frame& like, int64_t first, const std::vector<float>& y) const;
    void push(const arof::Frame& in, std::vector<float>& y, int64_t& first);

    std::string models_dir_, device_, model_name_;
    std::unique_ptr<Engine> engine_;
    std::unique_ptr<FrameStream> stream_;
    bool dpd_on_ = false;
    // co_optics's inputs to kappa (testbed/rpi/src/stages/link_stages.cpp, LinkStage::read_config)
    double vpp_ = 0, drive_scale_ = 1;
    std::size_t block_samples_ = 0;
    double kappa_ = 0;
    bool have_kappa_ = false, new_kappa_ = false;
    std::vector<arof::Frame> held_;            // the calibration group
    int64_t base_ = -1;                        // the stream's sample 0 as first_sample
    int64_t pushed_ = 0;                       // samples pushed into the stream
};

}  // namespace dpd
