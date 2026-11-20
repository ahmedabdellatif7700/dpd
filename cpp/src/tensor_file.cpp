#include "dpd/tensor_file.h"

#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace dpd {

namespace {

constexpr char kMagic[4] = {'D', 'P', 'D', 'T'};
constexpr uint32_t kVersion = 1;

// little-endian reads from the file's bytes; the Jetson (aarch64) is little-endian too
class Reader {
public:
    Reader(const std::vector<char>& b, const std::string& path) : b_(b), path_(path) {}

    template <typename T>
    T get() {
        T v;
        need(sizeof(T));
        std::memcpy(&v, b_.data() + p_, sizeof(T));
        p_ += sizeof(T);
        return v;
    }
    void bytes(void* dst, std::size_t n) {
        need(n);
        std::memcpy(dst, b_.data() + p_, n);
        p_ += n;
    }
    bool done() const { return p_ == b_.size(); }
    std::size_t left() const { return b_.size() - p_; }

private:
    void need(std::size_t n) const {
        if (b_.size() - p_ < n) throw std::runtime_error(path_ + ": truncated DPDT file");
    }
    const std::vector<char>& b_;
    const std::string& path_;
    std::size_t p_ = 0;
};

}  // namespace

std::string Tensor::shape() const {
    std::string s = "[";
    for (std::size_t i = 0; i < dims.size(); ++i) s += (i ? ", " : "") + std::to_string(dims[i]);
    return s + "]";
}

std::vector<Tensor> read_tensor_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(path + ": cannot open");
    const std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Reader r(b, path);

    char magic[4];
    r.bytes(magic, 4);
    if (std::memcmp(magic, kMagic, 4) != 0) throw std::runtime_error(path + ": not a DPDT file");
    const auto version = r.get<uint32_t>();
    if (version != kVersion)
        throw std::runtime_error(path + ": DPDT version " + std::to_string(version) + ", expected 1");
    const auto count = r.get<uint32_t>();

    std::vector<Tensor> out(count);
    for (auto& t : out) {
        t.name.resize(r.get<uint16_t>());
        r.bytes(t.name.data(), t.name.size());
        t.dims.resize(r.get<uint8_t>());
        std::size_t n = 1;
        for (auto& d : t.dims) n *= (d = r.get<uint32_t>());
        t.data.resize(n);
        r.bytes(t.data.data(), n * sizeof(float));
    }
    if (!r.done()) throw std::runtime_error(path + ": " + std::to_string(r.left()) + " trailing bytes");
    return out;
}

void write_tensor_file(const std::string& path, const std::vector<Tensor>& tensors) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(path + ": cannot write");
    auto put = [&](const void* p, std::size_t n) { f.write(static_cast<const char*>(p), static_cast<std::streamsize>(n)); };
    const auto count = static_cast<uint32_t>(tensors.size());
    put(kMagic, 4);
    put(&kVersion, 4);
    put(&count, 4);
    for (const auto& t : tensors) {
        std::size_t n = 1;
        for (auto d : t.dims) n *= d;
        if (n != t.data.size()) throw std::runtime_error("tensor '" + t.name + "': dims do not match its data");
        const auto name_len = static_cast<uint16_t>(t.name.size());
        const auto ndim = static_cast<uint8_t>(t.dims.size());
        put(&name_len, 2);
        put(t.name.data(), t.name.size());
        put(&ndim, 1);
        put(t.dims.data(), 4 * t.dims.size());
        put(t.data.data(), 4 * t.data.size());
    }
    if (!f) throw std::runtime_error(path + ": write failed");
}

const Tensor& find(const std::vector<Tensor>& tensors, std::string_view name) {
    for (const auto& t : tensors)
        if (t.name == name) return t;
    throw std::runtime_error("no tensor '" + std::string(name) + "'");
}

}  // namespace dpd
