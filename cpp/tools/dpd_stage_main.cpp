// dpd_stage: the program of the dpd_bb container (docs/testbed/jetson/STAGE.md). The stage
// contract is the Pi's stage library's (cpp/arof, run_filter): it listens on AROF_IN, connects to
// AROF_OUT, and runs until the END frame; any error stops it with a message and exit code 1.
//
//   AROF_STAGE=dpd_bb AROF_IN=tcp://0.0.0.0:7000 AROF_OUT=tcp://<co_optics>:7000 dpd_stage
//   AROF_MODELS_DIR  the weights (default /models), AROF_DPD_DEVICE  cuda (default) or cpu
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "dpd/dpd_stage.h"
#include "stage/stage.h"

namespace {

std::string env_or(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return v && *v ? v : fallback;
}

}  // namespace

int main() {
    try {
        const arof::StageEnv env = arof::StageEnv::from_environment();
        dpd::DpdStage stage(env_or("AROF_MODELS_DIR", "/models"), env_or("AROF_DPD_DEVICE", "cuda"));
        const uint64_t frames = arof::run_filter(stage, env);
        std::fprintf(stderr, "%s: %llu frames, END\n", env.name.c_str(), static_cast<unsigned long long>(frames));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "dpd_stage: %s\n", e.what());
        return 1;
    }
}
