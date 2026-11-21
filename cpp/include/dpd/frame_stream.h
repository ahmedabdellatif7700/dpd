// The stage's frame-by-frame processing of the BASEBAND stream with a holdback (decision 3b): the
// C++ counterpart of dpd/stream.py, which is its specification.
//
// The resamplers and the model's window reach about 220 samples (at 50 per symbol) into the next
// frame and 226 into the previous one. The stream keeps the last `hold` samples of what it has
// until the next frame brings the look-ahead, and `context` samples behind as the past. Each push
// returns the samples it can complete, from the index `done`: the signal is not delayed, only packed
// differently. hold, context and the frames are multiples of 25 (the grid the two rates share), so
// the stream gives the drive's whole transform, except within reach of its start and of a flush.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace dpd {

class FrameStream {
public:
    // transform: a segment x[n][2] (I/Q, at 50 samples per symbol) -> y[n][2], the segment alone
    using Transform = std::function<void(const float* x, int n, float* y)>;
    static constexpr int kHold = 250, kContext = 250, kGrid = 25;

    explicit FrameStream(Transform transform, int hold = kHold, int context = kContext);

    // the next frame, x[n][2], n a multiple of 25; appends to y the samples completed, returns the
    // index (from the stream's start) of the first of them
    int64_t push(const float* x, int n, std::vector<float>& y);
    // everything held, its last samples with zeros after them (a calibration group leaves whole);
    // the past is kept, the stream goes on
    int64_t flush(std::vector<float>& y);

    int64_t done() const { return done_; }        // samples emitted so far
    int64_t held() const { return buf_start_ + static_cast<int64_t>(buf_.size() / 2) - done_; }

private:
    int64_t emit(int64_t end, std::vector<float>& y);
    Transform transform_;
    int hold_, context_;
    std::vector<float> buf_, work_;  // the samples from buf_start_ on; the transform's output
    int64_t buf_start_ = 0, done_ = 0;
};

}  // namespace dpd
