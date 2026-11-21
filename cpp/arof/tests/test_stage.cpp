// The frame between the stages and the stage library (docs/testbed/rpi/PIPELINE.md, Sections 2, 3,
// 5, 7): the layout of version 2 next to version 1, a stage between a test source and a test sink
// on the loopback (forwarding, one frame in one frame out, the handshake, the END), the tap in its
// two modes, the timing log, the start-up retry.
#include <unistd.h>

#include <chrono>
#include <complex>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <thread>

#include "check.h"
#include "stage/stage.h"

using namespace arof;
using nlohmann::json;
using cd = std::complex<double>;

namespace {

uint16_t free_port() {
    Listener l(Endpoint{"127.0.0.1", 0});
    return l.port();
}

std::string url(uint16_t port) { return "tcp://127.0.0.1:" + std::to_string(port); }

Frame data_frame(Signal s, uint32_t seq, uint64_t first, const std::vector<cd>& v, uint16_t words = 1) {
    Frame f;
    f.h.seq = seq;
    f.h.first_sample = first;
    f.h.n_samples = static_cast<uint32_t>(v.size() / words);
    f.h.words_per_sample = words;
    f.h.dtype = static_cast<uint8_t>(DType::c128);
    f.h.signal = static_cast<uint8_t>(s);
    f.h.config_id = 1;
    f.h.seed = 2017;
    f.h.capture_time_ns = 1000000 + seq;
    f.payload.resize(v.size() * sizeof(cd));
    std::memcpy(f.payload.data(), v.data(), f.payload.size());
    return f;
}

Frame end_frame(uint32_t seq) {
    Frame f;
    f.h.flags = kEnd;
    f.h.seq = seq;
    f.h.config_id = 1;
    return f;
}

std::vector<cd> ramp(size_t n, double a) {
    std::vector<cd> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = cd(a + static_cast<double>(i), -a * static_cast<double>(i));
    return v;
}

bool same(const Frame& a, const Frame& b) {
    return std::memcmp(&a.h, &b.h, sizeof a.h) == 0 && a.payload == b.payload;
}

// a stage on the loopback between this test's source (which sends `in`) and its sink
struct Harness {
    std::vector<Frame> out, tapped;
    std::string error;
    StageEnv env;

    void run(Stage& stage, const std::vector<Frame>& in, const std::string& name = "stage_under_test",
             bool with_tap = false, const std::string& log_dir = "") {
        const uint16_t p_in = free_port(), p_out = free_port(), p_tap = free_port();
        env.name = name;
        env.in = url(p_in);
        env.out = url(p_out);
        env.tap = with_tap ? url(p_tap) : "";
        env.log_dir = log_dir;
        env.connect_timeout_s = 5;
        Listener sink(Endpoint{"127.0.0.1", p_out});
        std::unique_ptr<Listener> tap_sink;
        if (with_tap) tap_sink = std::make_unique<Listener>(Endpoint{"127.0.0.1", p_tap});
        std::thread st([&] {
            try {
                run_filter(stage, env);
            } catch (const std::exception& e) {
                error = e.what();
            }
        });
        std::thread tt;
        if (with_tap)
            tt = std::thread([&] {
                Socket s = tap_sink->accept();
                Frame f;
                while (s.read_frame(f)) tapped.push_back(f);
            });
        std::thread src([&] {
            try {
                Socket s = connect_retry(Endpoint{"127.0.0.1", p_in}, 5);
                for (const Frame& f : in) s.write_frame(f);
            } catch (const std::exception&) {
            }
        });
        {
            Socket s = sink.accept();
            Frame f;
            try {
                while (s.read_frame(f)) {           // a consumer closes after the END frame
                    out.push_back(f);
                    if (f.is_end()) break;
                }
            } catch (const std::exception&) {
            }
        }
        src.join();
        st.join();
        if (tt.joinable()) {
            if (!error.empty() && tapped.empty()) {         // the stage never connected its tap
                Socket wake = connect_retry(Endpoint{"127.0.0.1", p_tap}, 1);
            }
            tt.join();
        }
    }
};

Frame config(const json& settings, uint32_t seq = 0) { return config_frame(settings.dump(), seq, 1, 2017, 1000000 + seq); }

json source_settings(Signal out) {
    return {{"seed", 2017}, {"pipeline", json::array({{{"stage", "source"}, {"out", signal_name(out)}}})}};
}

// consumes BASEBAND, produces RF: keeps every second sample and holds the first frame back
class HalfStage : public Stage {
public:
    Signal in_signal() const override { return Signal::baseband; }
    Signal out_signal() const override { return Signal::rf; }
    void process(const Frame& in, std::vector<Frame>& outs) override {
        outs.emplace_back();
        Frame& out = outs.back();
        out.h = in.h;
        out.h.signal = static_cast<uint8_t>(Signal::rf);
        if (first_) {                                       // nothing comes out for the first frame
            first_ = false;
            out.h.n_samples = 0;
            return;
        }
        const size_t n = in.h.n_samples / 2;
        out.h.n_samples = static_cast<uint32_t>(n);
        out.h.first_sample = in.h.first_sample / 2;
        out.payload.resize(n * sizeof(cd));
        for (size_t i = 0; i < n; ++i) std::memcpy(out.payload.data() + i * sizeof(cd), in.payload.data() + 2 * i * sizeof(cd), sizeof(cd));
    }

private:
    bool first_ = true;
};

class RfStage : public Stage {
public:
    Signal in_signal() const override { return Signal::rf; }
    Signal out_signal() const override { return Signal::rf; }
    void process(const Frame& in, std::vector<Frame>& out) override { out.push_back(in); }
};

}  // namespace

TEST(version_2_keeps_the_layout_of_version_1) {
    CHECK(sizeof(FrameHeader) == 36);
    CHECK(sizeof(FrameHeader2) == 44);
    CHECK(offsetof(FrameHeader2, magic) == offsetof(FrameHeader, magic));
    CHECK(offsetof(FrameHeader2, version) == offsetof(FrameHeader, version));
    CHECK(offsetof(FrameHeader2, flags) == offsetof(FrameHeader, flags));
    CHECK(offsetof(FrameHeader2, seq) == offsetof(FrameHeader, seq));
    CHECK(offsetof(FrameHeader2, first_sample) == offsetof(FrameHeader, first_sample));
    CHECK(offsetof(FrameHeader2, n_samples) == offsetof(FrameHeader, n_samples));
    CHECK(offsetof(FrameHeader2, words_per_sample) == offsetof(FrameHeader, words_per_sample));
    CHECK(offsetof(FrameHeader2, dtype) == offsetof(FrameHeader, reserved));        // the old reserved word
    CHECK(offsetof(FrameHeader2, signal) == offsetof(FrameHeader, reserved) + 1);
    CHECK(offsetof(FrameHeader2, config_id) == offsetof(FrameHeader, config_id));
    CHECK(offsetof(FrameHeader2, seed) == offsetof(FrameHeader, seed));
    CHECK(offsetof(FrameHeader2, capture_time_ns) == 36);
    CHECK(kCalib == 16);
    CHECK((kCalib & (kRecalibrated | kTest | kEnd | kConfig)) == 0);
}

TEST(version_1_header_bytes_are_unchanged) {
    // the frame to the KV260: the bytes of a known header, as stream_rx reads them
    FrameHeader h;
    h.flags = kTest;
    h.seq = 7;
    h.first_sample = 0x0102030405060708ull;
    h.n_samples = 5;
    h.words_per_sample = 16;
    h.config_id = 3;
    h.seed = 2017;
    unsigned char b[36];
    write_header(h, b);
    const unsigned char want[36] = {0x41, 0x6F, 0x52, 0x46, 1, 0, 2, 0, 7, 0, 0, 0, 8, 7, 6, 5, 4, 3, 2, 1,
                                    5, 0, 0, 0, 16, 0, 0, 0, 3, 0, 0, 0, 0xE1, 0x07, 0, 0};
    CHECK(std::memcmp(b, want, 36) == 0);
}

TEST(payload_sizes) {
    FrameHeader2 h;
    h.n_samples = 6400;
    h.words_per_sample = 7;
    h.dtype = static_cast<uint8_t>(DType::c128);
    CHECK(payload_bytes(h) == 6400u * 7 * 16);
    h.dtype = static_cast<uint8_t>(DType::c64);
    CHECK(payload_bytes(h) == 6400u * 7 * 8);
    h.dtype = static_cast<uint8_t>(DType::q8_24);
    h.words_per_sample = 16;
    CHECK(payload_bytes(h) == 6400u * 16 * 4);
    h.flags = kConfig;
    h.n_samples = 123;
    CHECK(payload_bytes(h) == 123);
}

TEST(complex_float32_and_float64) {
    Frame f = data_frame(Signal::rf, 1, 0, {cd(1.5, -2.25), cd(0.1, 1e-3)});
    convert_complex(f, DType::c64);
    CHECK(f.dtype() == DType::c64);
    CHECK(f.payload.size() == 2 * 8);
    convert_complex(f, DType::c128);
    CHECK(f.payload.size() == 2 * 16);
    cd v[2];
    std::memcpy(v, f.payload.data(), sizeof v);
    CHECK(v[0] == cd(1.5, -2.25));                                      // exact in float32
    CHECK(v[1] == cd(static_cast<double>(0.1f), static_cast<double>(1e-3f)));
}

TEST(bypass_forwards_every_byte) {
    std::vector<Frame> in = {config(source_settings(Signal::baseband)),
                             data_frame(Signal::ref, 1, 0, ramp(256, 1), 2),
                             data_frame(Signal::baseband, 2, 0, ramp(6400, 0.5)),
                             data_frame(Signal::baseband, 3, 6400, {}),             // an empty frame
                             end_frame(4)};
    in[2].h.flags = kCalib;
    BypassStage s;
    Harness h;
    h.run(s, in, "dpd_bb");
    CHECK(h.error.empty());
    CHECK(h.out.size() == in.size());
    if (h.out.size() != in.size()) return;
    for (size_t i = 1; i < in.size(); ++i) CHECK(same(h.out[i], in[i]));
    // the CONFIG frame: same seq and time, the stage's line added to the pipeline
    CHECK(h.out[0].is_config());
    CHECK(h.out[0].h.seq == in[0].h.seq && h.out[0].h.capture_time_ns == in[0].h.capture_time_ns);
    const json j = json::parse(h.out[0].text());
    CHECK(j["seed"] == 2017);
    CHECK(j["pipeline"].size() == 2);
    CHECK(j["pipeline"][1]["stage"] == "dpd_bb");
    CHECK(j["pipeline"][1]["mode"] == "bypass");
    CHECK(j["pipeline"][1]["in"] == "BASEBAND");
    CHECK(j["pipeline"][1]["out"] == "BASEBAND");
}

TEST(one_frame_in_one_frame_out) {
    const std::vector<Frame> in = {config(source_settings(Signal::baseband)),
                                   data_frame(Signal::ref, 1, 0, ramp(256, 1), 2),
                                   data_frame(Signal::baseband, 2, 0, ramp(6400, 0.5)),
                                   data_frame(Signal::ref, 3, 128, ramp(256, 2), 2),
                                   data_frame(Signal::baseband, 4, 6400, ramp(6400, 0.25)),
                                   end_frame(5)};
    HalfStage s;
    Harness h;
    h.run(s, in);
    CHECK(h.error.empty());
    CHECK(h.out.size() == in.size());
    if (h.out.size() != in.size()) return;
    for (size_t i = 0; i < in.size(); ++i) {
        CHECK(h.out[i].h.seq == in[i].h.seq);
        CHECK(h.out[i].h.capture_time_ns == in[i].h.capture_time_ns);
        CHECK(h.out[i].h.config_id == in[i].h.config_id);
    }
    CHECK(same(h.out[1], in[1]));                           // REF: not this stage's signal, forwarded
    CHECK(same(h.out[3], in[3]));
    CHECK(h.out[2].signal() == Signal::rf && h.out[2].h.n_samples == 0);    // held back: an empty frame
    CHECK(h.out[4].signal() == Signal::rf && h.out[4].h.n_samples == 3200);
    CHECK(h.out[4].h.first_sample == 3200);
    cd v;
    std::memcpy(&v, h.out[4].payload.data() + sizeof(cd), sizeof v);
    CHECK(v == cd(0.25 + 2, -0.25 * 2));
    CHECK(same(h.out[5], in[5]));
    CHECK(json::parse(h.out[0].text())["pipeline"][1]["out"] == "RF");
}

TEST(a_pipeline_in_the_wrong_order_stops_at_the_handshake) {
    RfStage s;                                              // consumes RF, placed after a BASEBAND stage
    Harness h;
    h.run(s, {config(source_settings(Signal::baseband)), end_frame(1)}, "pa");
    CHECK(h.error.find("pa consumes RF") != std::string::npos);
    CHECK(h.error.find("source") != std::string::npos);
    CHECK(h.error.find("BASEBAND") != std::string::npos);
    CHECK(h.out.empty());
}

TEST(a_stream_must_open_with_config_and_close_with_end) {
    BypassStage s;
    Harness h1;
    h1.run(s, {data_frame(Signal::baseband, 0, 0, ramp(8, 1))});
    CHECK(h1.error.find("must be a CONFIG frame") != std::string::npos);
    Harness h2;
    h2.run(s, {config(source_settings(Signal::baseband)), data_frame(Signal::baseband, 1, 0, ramp(8, 1))});
    CHECK(h2.error.find("without an END frame") != std::string::npos);
}

TEST(lossless_tap_gets_every_output_frame) {
    json st = source_settings(Signal::baseband);
    st["taps"] = {{"stage_under_test", "lossless"}};
    std::vector<Frame> in = {config(st)};
    for (uint32_t k = 1; k <= 200; ++k) {
        in.push_back(data_frame(Signal::ref, 2 * k - 1, 128 * k, ramp(256, k), 2));
        in.push_back(data_frame(Signal::baseband, 2 * k, 6400 * k, ramp(6400, k)));
    }
    in.push_back(end_frame(401));
    HalfStage s;
    Harness h;
    h.run(s, in, "stage_under_test", true);
    CHECK(h.error.empty());
    // the tap carries the CONFIG frame, the stage's own output (RF) and the END, not the REF frames
    CHECK(h.tapped.size() == 1 + 200 + 1);
    if (h.tapped.size() != 202) return;
    CHECK(h.tapped.front().is_config());
    CHECK(h.tapped.back().is_end());
    size_t k = 1;
    for (const Frame& f : h.out)
        if (!f.is_config() && !f.is_end() && f.signal() == Signal::rf) {
            CHECK(same(h.tapped[k], f));
            ++k;
        }
    CHECK(k == 201);
}

TEST(live_tap_never_slows_the_chain) {
    // a consumer that accepts and never reads: the frames queue, then drop, and send() returns
    Listener stuck(Endpoint{"127.0.0.1", 0});
    Tap tap;
    Socket peer;
    std::thread acc([&] { peer = stuck.accept(); });
    tap.open(Endpoint{"127.0.0.1", stuck.port()}, Tap::Mode::live, 5);
    acc.join();
    const Frame f = data_frame(Signal::rf, 1, 0, ramp(65536, 1));           // 1 MB
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 400; ++i) tap.send(f);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(s < 5.0);
    CHECK(tap.dropped() > 0);
    CHECK(tap.dropped() + tap.sent() <= 400);
    peer.close();                                           // lets the sender thread end
}

TEST(tap_switched_on_without_an_endpoint_is_an_error) {
    json st = source_settings(Signal::baseband);
    st["taps"] = {{"stage_under_test", true}};
    BypassStage s;
    Harness h;
    h.run(s, {config(st), end_frame(1)});
    CHECK(h.error.find("AROF_TAP is not set") != std::string::npos);
}

TEST(timing_log_has_one_line_per_frame) {
    char dir[] = "/tmp/arof_stage_log_XXXXXX";
    CHECK(::mkdtemp(dir) != nullptr);
    json st = source_settings(Signal::baseband);
    st["log"] = {{"timing", true}};
    std::vector<Frame> in = {config(st), data_frame(Signal::baseband, 1, 0, ramp(6400, 1)),
                             data_frame(Signal::baseband, 2, 6400, ramp(6400, 2)), end_frame(3)};
    in[1].h.flags = kCalib;
    HalfStage s;
    Harness h;
    h.run(s, in, "pa", false, dir);
    CHECK(h.error.empty());
    std::ifstream f(std::string(dir) + "/pa.csv");
    std::vector<std::string> lines;
    for (std::string l; std::getline(f, l);) lines.push_back(l);
    CHECK(lines.size() == 1 + in.size());
    if (lines.size() != 1 + in.size()) return;
    CHECK(lines[0] == "seq,signal,flags,config_id,n_in,n_out,capture_time_ns,t_recv_ns,t_send_ns,process_ns");
    CHECK(lines[1].rfind("0,CONFIG,8,1,", 0) == 0);                 // the CONFIG frame, flags 8
    CHECK(lines[2].rfind("1,BASEBAND,16,1,6400,0,1000001,", 0) == 0);   // CALIB, held back: 0 samples out
    CHECK(lines[3].rfind("2,BASEBAND,0,1,6400,3200,1000002,", 0) == 0);
    CHECK(lines[4].rfind("3,CONFIG,4,1,0,0,", 0) == 0);             // END (signal 0)
    // t_send >= t_recv
    std::stringstream ss(lines[3]);
    std::vector<std::string> c;
    for (std::string x; std::getline(ss, x, ',');) c.push_back(x);
    CHECK(c.size() == 10);
    if (c.size() == 10) CHECK(std::stoull(c[8]) >= std::stoull(c[7]));
}

TEST(output_connection_retries_at_start_up) {
    const uint16_t p = free_port();
    std::thread late([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(600));
        Listener l(Endpoint{"127.0.0.1", p});
        Socket s = l.accept();
    });
    Socket s = connect_retry(Endpoint{"127.0.0.1", p}, 5);          // the listener comes up later
    CHECK(s.valid());
    late.join();
    std::string err;
    try {
        connect_retry(Endpoint{"127.0.0.1", free_port()}, 0.5);
    } catch (const std::exception& e) {
        err = e.what();
    }
    CHECK(err.find("could not connect to tcp://127.0.0.1:") != std::string::npos);
    CHECK(parse_endpoint("tcp://wireless:7008").host == "wireless");
    CHECK(parse_endpoint("tcp://0.0.0.0:7007").port == 7007);
}

TEST(the_chain_ends_from_its_last_stage) {
    // after the END frame a stage waits for the stage after it to close; it then returns
    BypassStage s;
    Harness h;
    const auto t0 = std::chrono::steady_clock::now();
    h.run(s, {config(source_settings(Signal::baseband)), end_frame(1)});
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(h.error.empty());
    CHECK(h.out.size() == 2 && h.out.back().is_end());
    CHECK(secs < 5);                                    // not the 30 s timeout: the sink closed after END
}

TEST_MAIN()
