// The DPDT file: a list of named float32 tensors (docs/dpd/FORMAT.md), written by the reference
// (dpd/export.py) and read here.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace dpd {

struct Tensor {
    std::string name;
    std::vector<uint32_t> dims;
    std::vector<float> data;  // row-major

    std::size_t size() const { return data.size(); }
    std::string shape() const;  // "[8, 32, 32]"
};

// Reads a DPDT file; throws std::runtime_error on a wrong magic, version, truncation or
// trailing bytes. The tensors come in the file's order.
std::vector<Tensor> read_tensor_file(const std::string& path);

// Writes tensors to a DPDT file (the format dpd/export.py reads); throws if it cannot.
void write_tensor_file(const std::string& path, const std::vector<Tensor>& tensors);

// The tensor called name; throws if it is missing.
const Tensor& find(const std::vector<Tensor>& tensors, std::string_view name);

}  // namespace dpd
