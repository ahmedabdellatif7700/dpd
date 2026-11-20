// The frame between the stages of the pipeline (docs/testbed/rpi/PIPELINE.md, Section 2): version 2
// of stream/frame.h. The first 36 bytes keep the layout of version 1 (its reserved word now holds
// dtype and signal), and capture_time_ns is appended: a 44-byte header, little-endian, then
//   data frames:    n_samples * words_per_sample values of dtype, sample by sample
//   CONFIG frames:  n_samples bytes of JSON text, words_per_sample = 0
// seq and capture_time_ns are set by the source and carried unchanged by every stage. The frame to
// the KV260 stays version 1 (stream/frame.h).
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "stream/frame.h"

namespace arof {

constexpr uint16_t kFrameVersion2 = 2;
constexpr uint16_t kCalib = 1 << 4;             // the frame belongs to a calibration block
constexpr uint16_t kCalibEnd = 1 << 5;          // the last frame of a calibration block (its tail)

enum class DType : uint8_t { q8_24 = 0, c128 = 1, c64 = 2, json = 3 };
enum class Signal : uint8_t {
    none = 0,           // CONFIG
    ref = 1,            // transmitted symbols: d and its Gray label, per symbol
    baseband = 2,       // RRC waveform before the IQ modulator
    optical = 3,        // the optical components, n = -n_harm..n_harm
    rf = 4,             // mmW complex envelope
    rx_baseband = 5,    // after the UE's mixer, low-pass and ADCs
    rx_symbols = 6,     // received symbols
    samples = 7,        // engine samples (Q8.24)
};

#pragma pack(push, 1)
struct FrameHeader2 {
    uint32_t magic = kFrameMagic;
    uint16_t version = kFrameVersion2;
    uint16_t flags = 0;
    uint32_t seq = 0;               // the source's frame number, carried unchanged
    uint64_t first_sample = 0;      // index of the first sample, in the units of the signal
    uint32_t n_samples = 0;         // samples (may be 0) or payload bytes (CONFIG)
    uint16_t words_per_sample = 0;  // values per sample; 0 for CONFIG
    uint8_t dtype = 0;              // DType
    uint8_t signal = 0;             // Signal
    uint32_t config_id = 0;
    uint32_t seed = 0;
    uint64_t capture_time_ns = 0;   // CLOCK_REALTIME when the source made the frame, carried unchanged
};
#pragma pack(pop)
static_assert(sizeof(FrameHeader2) == 44, "frame header version 2 is 44 bytes");

inline size_t value_bytes(DType t) {
    switch (t) {
        case DType::q8_24: return 4;
        case DType::c128: return 16;
        case DType::c64: return 8;
        case DType::json: return 1;
    }
    throw std::runtime_error("frame: unknown dtype");
}

inline size_t payload_bytes(const FrameHeader2& h) {
    if (h.flags & kConfig) return h.n_samples;
    return static_cast<size_t>(h.n_samples) * h.words_per_sample * value_bytes(static_cast<DType>(h.dtype));
}

inline const char* signal_name(Signal s) {
    static const char* const names[] = {"CONFIG", "REF", "BASEBAND", "OPTICAL", "RF", "RX_BASEBAND", "RX_SYMBOLS", "SAMPLES"};
    const auto i = static_cast<unsigned>(s);
    return i < 8 ? names[i] : "?";
}

inline const char* dtype_name(DType t) {
    static const char* const names[] = {"q8.24", "c128", "c64", "json"};
    const auto i = static_cast<unsigned>(t);
    return i < 4 ? names[i] : "?";
}

struct Frame {
    FrameHeader2 h;
    std::vector<unsigned char> payload;

    bool is_config() const { return h.flags & kConfig; }
    bool is_end() const { return h.flags & kEnd; }
    Signal signal() const { return static_cast<Signal>(h.signal); }
    DType dtype() const { return static_cast<DType>(h.dtype); }
    std::string text() const { return std::string(payload.begin(), payload.end()); }
};

// a CONFIG frame carrying JSON text
inline Frame config_frame(const std::string& json_text, uint32_t seq, uint32_t config_id, uint32_t seed,
                          uint64_t capture_time_ns) {
    Frame f;
    f.h.flags = kConfig;
    f.h.seq = seq;
    f.h.n_samples = static_cast<uint32_t>(json_text.size());
    f.h.dtype = static_cast<uint8_t>(DType::json);
    f.h.config_id = config_id;
    f.h.seed = seed;
    f.h.capture_time_ns = capture_time_ns;
    f.payload.assign(json_text.begin(), json_text.end());
    return f;
}

}  // namespace arof
