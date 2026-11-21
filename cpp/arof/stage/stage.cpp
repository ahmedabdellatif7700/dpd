#include "stage/stage.h"

#include <complex>
#include <csignal>
#include <unistd.h>
#include <cstdlib>
#include <stdexcept>

namespace arof {

using nlohmann::json;

namespace {

std::string env_or(const char* name, const std::string& fallback = "") {
    const char* v = std::getenv(name);
    return v && *v ? std::string(v) : fallback;
}

}  // namespace

StageEnv StageEnv::from_environment() {
    StageEnv e;
    e.name = env_or("AROF_STAGE");
    e.in = env_or("AROF_IN");
    e.out = env_or("AROF_OUT");
    e.tap = env_or("AROF_TAP");
    e.log_dir = env_or("AROF_LOG_DIR");
    e.connect_timeout_s = std::stod(env_or("AROF_CONNECT_TIMEOUT_S", "60"));
    return e;
}

void convert_complex(Frame& f, DType to) {
    const DType from = f.dtype();
    if (from == to || f.is_config()) return;
    const size_t n = static_cast<size_t>(f.h.n_samples) * f.h.words_per_sample;
    if (from == DType::c128 && to == DType::c64) {
        std::vector<unsigned char> p(n * 8);
        for (size_t i = 0; i < 2 * n; ++i) {
            double v;
            std::memcpy(&v, f.payload.data() + 8 * i, 8);
            const float w = static_cast<float>(v);
            std::memcpy(p.data() + 4 * i, &w, 4);
        }
        f.payload.swap(p);
    } else if (from == DType::c64 && to == DType::c128) {
        std::vector<unsigned char> p(n * 16);
        for (size_t i = 0; i < 2 * n; ++i) {
            float v;
            std::memcpy(&v, f.payload.data() + 4 * i, 4);
            const double w = v;
            std::memcpy(p.data() + 8 * i, &w, 8);
        }
        f.payload.swap(p);
    } else {
        throw std::runtime_error(std::string("cannot convert ") + dtype_name(from) + " to " + dtype_name(to));
    }
    f.h.dtype = static_cast<uint8_t>(to);
}

// ---- the order of the stop

namespace {
volatile std::sig_atomic_t g_end_sent = 0;
extern "C" void on_sigterm(int) { ::_exit(g_end_sent ? 0 : 143); }
}  // namespace

void install_stop_handler() {
    std::signal(SIGTERM, on_sigterm);
    std::signal(SIGINT, on_sigterm);
}

void end_sent() { g_end_sent = 1; }

// ---- Tap

Tap::~Tap() {
    try {
        close();
    } catch (...) {
    }
}

void Tap::open(const Endpoint& e, Mode mode, double connect_timeout_s) {
    close();
    if (mode == Mode::off) return;
    sock_ = connect_retry(e, connect_timeout_s);
    mode_ = mode;
    stop_ = false;
    failed_ = false;
    thread_ = std::thread([this] { run(); });
}

void Tap::run() {
    for (;;) {
        Frame f;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) return;             // stop, and nothing left to send
            f = std::move(queue_.front());
            queue_.pop_front();
        }
        cv_.notify_all();
        try {
            sock_.write_frame(f);
            ++sent_;
        } catch (const std::exception&) {
            failed_ = true;
            std::lock_guard<std::mutex> lk(m_);
            queue_.clear();
            cv_.notify_all();
            return;
        }
    }
}

void Tap::send(const Frame& f) {
    if (mode_ == Mode::off) return;
    if (failed_) {
        if (mode_ == Mode::lossless) throw std::runtime_error("tap: the consumer of a lossless tap is gone");
        ++dropped_;
        return;
    }
    std::unique_lock<std::mutex> lk(m_);
    if (queue_.size() >= kCapacity) {
        if (mode_ == Mode::live) {
            ++dropped_;
            return;
        }
        cv_.wait(lk, [this] { return queue_.size() < kCapacity || failed_; });
        if (failed_) throw std::runtime_error("tap: the consumer of a lossless tap is gone");
    }
    queue_.push_back(f);
    lk.unlock();
    cv_.notify_all();
}

void Tap::close() {
    if (thread_.joinable()) {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }
    sock_.close();
    mode_ = Mode::off;
}

// ---- TimingLog

TimingLog::~TimingLog() { close(); }

void TimingLog::open(const std::string& dir, const std::string& stage) {
    close();
    const std::string path = dir + "/" + stage + ".csv";
    f_ = std::fopen(path.c_str(), "w");
    if (!f_) throw std::runtime_error("timing log: cannot write " + path);
    std::fprintf(f_, "seq,signal,flags,config_id,n_in,n_out,capture_time_ns,t_recv_ns,t_send_ns,process_ns\n");
}

void TimingLog::line(const FrameHeader2& in, uint32_t n_out, uint64_t t_recv_ns, uint64_t t_send_ns, uint64_t process_ns) {
    if (!f_) return;
    std::fprintf(f_, "%u,%s,%u,%u,%u,%u,%llu,%llu,%llu,%llu\n", in.seq, signal_name(static_cast<Signal>(in.signal)),
                 static_cast<unsigned>(in.flags), in.config_id, in.n_samples, n_out,
                 static_cast<unsigned long long>(in.capture_time_ns), static_cast<unsigned long long>(t_recv_ns),
                 static_cast<unsigned long long>(t_send_ns), static_cast<unsigned long long>(process_ns));
}

void TimingLog::close() {
    if (f_) std::fclose(f_);
    f_ = nullptr;
}

// ---- settings switches

Tap::Mode tap_mode(const json& settings, const std::string& stage) {
    if (!settings.contains("taps") || !settings["taps"].contains(stage)) return Tap::Mode::off;
    const json& v = settings["taps"][stage];
    if (v.is_boolean()) return v.get<bool>() ? Tap::Mode::live : Tap::Mode::off;
    if (v.is_string() && v.get<std::string>() == "lossless") return Tap::Mode::lossless;
    if (v.is_string() && v.get<std::string>() == "live") return Tap::Mode::live;
    throw std::runtime_error("settings: taps." + stage + " must be false, true or \"lossless\"");
}

bool timing_on(const json& settings) {
    return settings.contains("log") && settings["log"].value("timing", false);
}

json control_request(const Endpoint& e, const json& cmd, double timeout_s) {
    Socket s = connect_retry(e, timeout_s);
    s.write_line(cmd.dump());
    std::string line;
    if (!s.read_line(line)) throw std::runtime_error("the control connection closed without a reply");
    return json::parse(line);
}

// ---- the rules of a stage on a stream of frames

void StageRunner::push(Frame&& f, uint64_t t_recv_ns, std::vector<Out>& ready) {
    if (ended_) throw std::runtime_error(name_ + ": a frame after the END frame");
    if (!handshaken_ && !f.is_config())
        throw std::runtime_error(name_ + ": the first frame of a connection must be a CONFIG frame");
    slots_.emplace_back();
    Slot& slot = slots_.back();
    slot.out.in = f.h;
    slot.out.t_recv_ns = t_recv_ns;

    if (f.is_config()) {
        json cfg = json::parse(f.text());
        config_ = cfg;
        json& pipeline = cfg["pipeline"];
        if (!pipeline.is_array()) pipeline = json::array();
        // the stage before must produce what this one consumes
        upstream_ = Signal::none;
        std::string before = "(none)";
        if (!pipeline.empty()) {
            before = pipeline.back().value("stage", "?");
            const std::string s = pipeline.back().value("out", "");
            for (unsigned i = 0; i < 8; ++i)
                if (s == signal_name(static_cast<Signal>(i))) upstream_ = static_cast<Signal>(i);
        }
        if (stage_.in_signal() != Signal::none && stage_.in_signal() != upstream_)
            throw std::runtime_error(name_ + " consumes " + signal_name(stage_.in_signal()) + ", but the stage before it, " +
                                     before + ", produces " + signal_name(upstream_));
        my_out_ = stage_.out_signal() == Signal::none ? upstream_ : stage_.out_signal();
        json entry = {{"stage", name_},
                      {"in", signal_name(stage_.in_signal() == Signal::none ? upstream_ : stage_.in_signal())},
                      {"out", signal_name(my_out_)}};
        stage_.configure(cfg, entry);
        pipeline.push_back(entry);
        handshaken_ = true;
        slot.out.is_output = true;                          // the tap's consumer needs the settings too
        slot.json = std::move(cfg);
        slot.config = true;
        if (!stage_.calibrates()) {
            slot.out.frame = config_frame(slot.json.dump(), f.h.seq, f.h.config_id, f.h.seed, f.h.capture_time_ns);
            slot.out.frame.h.flags = f.h.flags;
            slot.filled = true;
        }
    } else if (!f.is_end() && (stage_.in_signal() == Signal::none || f.signal() == stage_.in_signal())) {
        if (f.dtype() != DType::q8_24 && !stage_.accepts(f.dtype())) {
            const DType to = stage_.accepts(DType::c128) ? DType::c128 : DType::c64;
            if (!warned_dtype_) {
                std::fprintf(stderr, "%s: converting its input from %s to %s\n", name_.c_str(), dtype_name(f.dtype()), dtype_name(to));
                warned_dtype_ = true;
            }
            convert_complex(f, to);
        }
        slot.consumed = true;
        produced_.clear();
        const uint64_t t0 = monotonic_ns();
        stage_.process(f, produced_);
        slot.out.process_ns = monotonic_ns() - t0;
        // the constants of a calibration, if this input completed it: into the waiting CONFIG frame
        json constants;
        if (stage_.calibrates() && stage_.take_calibration(constants)) {
            bool found = false;
            for (Slot& s : slots_)
                if (s.config && !s.filled) {
                    s.json["pipeline"].back()["calibration"] = constants;
                    s.out.frame = config_frame(s.json.dump(), s.out.in.seq, s.out.in.config_id, s.out.in.seed, s.out.in.capture_time_ns);
                    s.out.frame.h.flags = s.out.in.flags;
                    s.filled = true;
                    found = true;
                    break;
                }
            if (!found) throw std::logic_error(name_ + ": calibration constants without a CONFIG frame waiting for them");
        }
        // the outputs answer the consumed inputs in their order
        auto it = slots_.begin();
        for (Frame& o : produced_) {
            while (it != slots_.end() && !(it->consumed && !it->filled)) ++it;
            if (it == slots_.end()) throw std::logic_error(name_ + ": more output frames than input frames");
            const FrameHeader2& in = it->out.in;
            o.h.magic = kFrameMagic;
            o.h.version = kFrameVersion2;
            o.h.flags = in.flags;
            o.h.seq = in.seq;
            o.h.config_id = in.config_id;
            o.h.seed = in.seed;
            o.h.capture_time_ns = in.capture_time_ns;
            if (o.payload.size() != payload_bytes(o.h)) throw std::logic_error(name_ + ": an output frame's payload does not match its header");
            it->out.is_output = o.signal() == my_out_;
            it->out.frame = std::move(o);
            it->filled = true;
        }
    } else {
        if (!f.is_end()) stage_.observe(f);
        slot.out.is_output = f.is_end();
        if (f.is_end()) {
            ended_ = true;
            // a stream that ends inside a calibration: what was held leaves empty
            for (Slot& s : slots_)
                if (!s.filled && &s != &slot) {
                    if (s.config) {
                        s.out.frame = config_frame(s.json.dump(), s.out.in.seq, s.out.in.config_id, s.out.in.seed, s.out.in.capture_time_ns);
                        s.out.frame.h.flags = s.out.in.flags;
                    } else {
                        s.out.frame.h = s.out.in;
                        s.out.frame.h.signal = static_cast<uint8_t>(my_out_);
                        s.out.frame.h.n_samples = 0;
                        s.out.frame.payload.clear();
                    }
                    s.filled = true;
                }
        }
        slot.out.frame = std::move(f);                      // not this stage's signal, or END: forwarded as it is
        slot.filled = true;
    }
    flush(ready);
}

void StageRunner::flush(std::vector<Out>& ready) {
    while (!slots_.empty() && slots_.front().filled) {
        ready.push_back(std::move(slots_.front().out));
        slots_.pop_front();
    }
}

// ---- the loop of a stage between two others

uint64_t run_filter(Stage& stage, const StageEnv& env) {
    if (env.name.empty() || env.in.empty() || env.out.empty())
        throw std::runtime_error("a stage needs AROF_STAGE, AROF_IN and AROF_OUT");
    Listener listener(parse_endpoint(env.in));              // listen first, so the stage before can connect
    Socket out = connect_retry(parse_endpoint(env.out), env.connect_timeout_s);
    Socket in = listener.accept();

    install_stop_handler();
    StageRunner runner(stage, env.name);
    Tap tap;
    TimingLog log;
    uint64_t frames = 0;
    std::vector<StageRunner::Out> ready;
    while (!runner.ended()) {
        Frame f;
        if (!in.read_frame(f)) throw std::runtime_error(env.name + ": the input closed without an END frame");
        const bool is_config = f.is_config();
        ready.clear();
        runner.push(std::move(f), realtime_ns(), ready);
        if (is_config) {                                    // the switches of this stage
            const json& cfg = runner.config();
            const Tap::Mode mode = tap_mode(cfg, env.name);
            if (mode != tap.mode()) {
                if (mode != Tap::Mode::off && env.tap.empty())
                    throw std::runtime_error(env.name + ": the settings turn its tap on, but AROF_TAP is not set");
                if (mode == Tap::Mode::off) tap.close();
                else tap.open(parse_endpoint(env.tap), mode, env.connect_timeout_s);
            }
            if (timing_on(cfg) && !log.on()) {
                if (env.log_dir.empty()) throw std::runtime_error(env.name + ": log.timing is on, but AROF_LOG_DIR is not set");
                log.open(env.log_dir, env.name);
            }
        }
        for (StageRunner::Out& o : ready) {
            out.write_frame(o.frame);
            const uint64_t t_send = realtime_ns();
            if (o.is_output) tap.send(o.frame);
            log.line(o.in, o.frame.h.n_samples, o.t_recv_ns, t_send, o.process_ns);
            ++frames;
        }
    }
    tap.close();
    log.close();
    end_sent();
    out.wait_closed(30);                                // the stages after this one end first
    return frames;
}

}  // namespace arof
