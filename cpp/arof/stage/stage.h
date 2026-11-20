// The stage contract of the pipeline (PIPELINE.md, Section 3). A stage consumes the frames of one
// signal, emits exactly one frame for each (same seq, capture_time_ns, config_id and flags), and
// forwards every other frame unchanged. run_filter() does the rest: the connections from the
// environment, the CONFIG handshake, forwarding, the tap, the timing log, the END of the stream.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "stage/frame2.h"
#include "stage/net.h"

namespace arof {

// the endpoints and switches a container gets from compose.yaml
struct StageEnv {
    std::string name;                   // AROF_STAGE
    std::string in;                     // AROF_IN: where it listens (tcp://0.0.0.0:port)
    std::string out;                    // AROF_OUT: where it sends
    std::string tap;                    // AROF_TAP: where the tap sends, if the settings turn it on
    std::string log_dir;                // AROF_LOG_DIR
    double connect_timeout_s = 60;      // AROF_CONNECT_TIMEOUT_S
    static StageEnv from_environment();
};

class Stage {
public:
    virtual ~Stage() = default;
    // the signal it consumes and the one it produces; Signal::none: whatever the stage before it
    // produces (a pass-through)
    virtual Signal in_signal() const = 0;
    virtual Signal out_signal() const = 0;
    // a complex dtype the stage does not accept is converted at its input
    virtual bool accepts(DType t) const { return t == DType::c128; }
    // called for every CONFIG frame with its JSON; entry is this stage's line of the "pipeline"
    // section, to which it may add
    virtual void configure(const nlohmann::json& config, nlohmann::json& entry) {
        (void)config;
        (void)entry;
    }
    // One frame out for every frame in, in order. The stage appends to out the output frames that
    // this input made ready: usually one; none while it holds the frames of a calibration block,
    // and then all of them at once. It sets signal, dtype, words_per_sample, n_samples,
    // first_sample and the payload; seq, capture_time_ns, config_id, seed and flags are copied by
    // the runner from the input frame each output belongs to.
    virtual void process(const Frame& in, std::vector<Frame>& out) = 0;
    // a frame of another signal, forwarded as it is (the receiver reads the REF symbols this way)
    virtual void observe(const Frame& f) { (void)f; }
    // A stage that takes constants from the calibration block returns true: the CONFIG frame is
    // then held with the frames behind it until take_calibration() gives the constants, which go
    // into the stage's line of the "pipeline" section (PIPELINE.md, Section 4).
    virtual bool calibrates() const { return false; }
    virtual bool take_calibration(nlohmann::json& constants) {
        (void)constants;
        return false;
    }
};

// forwards every frame untouched: the DPD boundaries until a DPD takes their place
class BypassStage : public Stage {
public:
    Signal in_signal() const override { return Signal::none; }
    Signal out_signal() const override { return Signal::none; }
    void configure(const nlohmann::json&, nlohmann::json& entry) override { entry["mode"] = "bypass"; }
    void process(const Frame& in, std::vector<Frame>& out) override { out.push_back(in); }
};

// complex float64 <-> complex float32, in place; other dtypes are left as they are
void convert_complex(Frame& f, DType to);

// a copy of a stage's output for another consumer (PIPELINE.md, Section 5). live: a frame that
// cannot be queued is dropped and counted, the chain is never slowed; lossless: it waits.
class Tap {
public:
    enum class Mode { off, live, lossless };
    Tap() = default;
    ~Tap();
    void open(const Endpoint& e, Mode mode, double connect_timeout_s);
    void send(const Frame& f);          // throws in lossless mode if the consumer is gone
    void close();                       // sends what is queued, then closes
    Mode mode() const { return mode_; }
    uint64_t dropped() const { return dropped_; }
    uint64_t sent() const { return sent_; }

private:
    void run();
    Mode mode_ = Mode::off;
    Socket sock_;
    std::thread thread_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<Frame> queue_;
    bool stop_ = false;
    std::atomic<bool> failed_{false};
    std::atomic<uint64_t> dropped_{0}, sent_{0};
    static constexpr size_t kCapacity = 64;
};

// one CSV line per frame (PIPELINE.md, Section 7)
class TimingLog {
public:
    ~TimingLog();
    void open(const std::string& dir, const std::string& stage);
    bool on() const { return f_ != nullptr; }
    void line(const FrameHeader2& in, uint32_t n_out, uint64_t t_recv_ns, uint64_t t_send_ns, uint64_t process_ns);
    void close();

private:
    std::FILE* f_ = nullptr;
};

// The rules of a stage on a stream of frames, without the sockets: the CONFIG handshake, what the
// stage consumes and what is forwarded, the order of the output. push() takes the next input
// frame and appends the frames that are ready to leave, in the order of their inputs.
class StageRunner {
public:
    StageRunner(Stage& stage, std::string name) : stage_(stage), name_(std::move(name)) {}
    struct Out {
        Frame frame;
        FrameHeader2 in;            // the header of the input frame it answers
        uint64_t t_recv_ns = 0;
        uint64_t process_ns = 0;    // time in process() for this input
        bool is_output = false;     // a frame of the stage's own output signal (or CONFIG, END): for the tap
    };
    void push(Frame&& f, uint64_t t_recv_ns, std::vector<Out>& ready);
    const nlohmann::json& config() const { return config_; }       // the last CONFIG, as received
    bool ended() const { return ended_; }
    Signal out_signal() const { return my_out_; }

private:
    struct Slot {
        Out out;
        bool filled = false;
        bool consumed = false;      // waits for the stage's output
        bool config = false;        // a CONFIG frame waiting for the calibration constants
        nlohmann::json json;        // the CONFIG's JSON, this stage's entry last
    };
    void flush(std::vector<Out>& ready);
    Stage& stage_;
    std::string name_;
    std::deque<Slot> slots_;
    nlohmann::json config_;
    Signal upstream_ = Signal::none, my_out_ = Signal::none;
    bool handshaken_ = false, warned_dtype_ = false, ended_ = false;
    std::vector<Frame> produced_;
};

// a stage between two others: listens on env.in, connects to env.out, runs until the END frame.
// Returns the number of frames it emitted; throws on any error (the pipeline then stops).
uint64_t run_filter(Stage& stage, const StageEnv& env);

// one JSON line to a control endpoint, one back (the streamer's relay to the source)
nlohmann::json control_request(const Endpoint& e, const nlohmann::json& cmd, double timeout_s = 5);

// The order of the stop (PIPELINE.md, Section 3.5): a stage that has sent the END frame waits until
// the stage after it has closed the connection, so the chain ends from its last stage backwards
// and every stage reads the END first. After end_sent(), a SIGTERM (compose stopping the others
// when the first one exits) ends the process with 0; before it, with 143, a failure.
void install_stop_handler();
void end_sent();

// helpers shared with the first and the last stage
Tap::Mode tap_mode(const nlohmann::json& settings, const std::string& stage);
bool timing_on(const nlohmann::json& settings);

}  // namespace arof
