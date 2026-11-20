// dpd/stream.py's FrameStream, line for line.
#include "dpd/frame_stream.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace dpd {

FrameStream::FrameStream(Transform transform, int hold, int context)
    : transform_(std::move(transform)), hold_(hold), context_(context) {
    if (hold % kGrid || context % kGrid || hold < 0 || context < 0)
        throw std::invalid_argument("FrameStream: hold and context must be multiples of 25");
}

int64_t FrameStream::emit(int64_t end, std::vector<float>& y) {
    const int n = static_cast<int>(buf_.size() / 2);
    work_.resize(buf_.size());
    transform_(buf_.data(), n, work_.data());
    const auto from = static_cast<std::size_t>(2 * (done_ - buf_start_)), to = static_cast<std::size_t>(2 * (end - buf_start_));
    y.insert(y.end(), work_.begin() + static_cast<std::ptrdiff_t>(from), work_.begin() + static_cast<std::ptrdiff_t>(to));
    const int64_t first = done_;
    done_ = end;
    const int64_t keep_from = std::max(buf_start_, done_ - context_);
    buf_.erase(buf_.begin(), buf_.begin() + static_cast<std::ptrdiff_t>(2 * (keep_from - buf_start_)));
    buf_start_ = keep_from;
    return first;
}

int64_t FrameStream::push(const float* x, int n, std::vector<float>& y) {
    if (n % kGrid) throw std::invalid_argument("FrameStream: a frame must be a multiple of 25 samples, is " + std::to_string(n));
    buf_.insert(buf_.end(), x, x + 2 * static_cast<std::size_t>(n));
    const int64_t end = buf_start_ + static_cast<int64_t>(buf_.size() / 2) - hold_;
    if (end <= done_) return done_;
    return emit(end, y);
}

int64_t FrameStream::flush(std::vector<float>& y) {
    return emit(buf_start_ + static_cast<int64_t>(buf_.size() / 2), y);
}

}  // namespace dpd
