// dpd_apply: predistort a drive with the twin, file in, file out.
//
//   dpd_apply [--device cuda|cpu] <weights.dpdt> <in.dpdt> <out.dpdt>
//
// in.dpdt holds a tensor "input" [n][2] (I/Q, the drive in the units the model was trained in);
// out.dpdt gets "output" [n][2], the sequence taken whole (zeros beyond its ends), as the
// reference's predistort() does. The bridge between the twin and the Python checks
// (scripts/twin_check.py): numpy writes the drive, this tool predistorts it, numpy measures.
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "dpd/cpu_model.h"
#include "dpd/gpu_model.h"
#include "dpd/tensor_file.h"
#include "dpd/weights.h"

int main(int argc, char** argv) {
    std::string device = "cuda";
    std::vector<const char*> pos;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--device") && i + 1 < argc)
            device = argv[++i];
        else
            pos.push_back(argv[i]);
    }
    if (pos.size() != 3 || (device != "cuda" && device != "cpu")) {
        std::fprintf(stderr, "usage: dpd_apply [--device cuda|cpu] <weights.dpdt> <in.dpdt> <out.dpdt>\n");
        return 2;
    }

    try {
        const dpd::Weights w = dpd::load_weights(pos[0]);
        const auto in_file = dpd::read_tensor_file(pos[1]);
        const dpd::Tensor& in = dpd::find(in_file, "input");
        if (in.dims.size() != 2 || in.dims[1] != 2) throw std::runtime_error("'input' must be [n][2], is " + in.shape());
        const int n = static_cast<int>(in.dims[0]);

        dpd::Tensor out{"output", in.dims, std::vector<float>(in.data.size())};
        if (device == "cuda")
            dpd::GpuModel(w).predistort(in.data.data(), out.data.data(), n);
        else
            dpd::CpuModel(w).predistort(in.data.data(), out.data.data(), n);
        dpd::write_tensor_file(pos[2], {out});
        std::printf("dpd_apply: %d samples on the %s -> %s\n", n, device == "cuda" ? "GPU" : "CPU", pos[2]);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "dpd_apply: %s\n", e.what());
        return 1;
    }
}
