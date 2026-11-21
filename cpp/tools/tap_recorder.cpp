// tap_recorder: records a tap's frames on the Jetson (docs/testbed/jetson/HOOKUP.md): the PA's output
// (the pa stage's tap, y) and the drive the dpd stage sent (its own tap, x_d), the two recordings of
// indirect learning, matched by seq. Listens on AROF_IN, writes every frame as it came (frame v2,
// header and payload, back to back) to AROF_SINK_FILE until the END frame, then exits 0. The same
// file format as the Pi's pa_tap sink; read with dpd/frame2.py (read_frames).
//
//   AROF_STAGE=tap_pa AROF_IN=tcp://0.0.0.0:7107 AROF_SINK_FILE=/records/pa_tap.bin tap_recorder
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

#include "stage/net.h"

int main() {
    const char* in = std::getenv("AROF_IN");
    const char* file = std::getenv("AROF_SINK_FILE");
    const char* name_env = std::getenv("AROF_STAGE");
    const std::string name = name_env && *name_env ? name_env : "tap_recorder";
    try {
        if (!in || !*in || !file || !*file) throw std::runtime_error("needs AROF_IN and AROF_SINK_FILE");
        arof::Listener listener(arof::parse_endpoint(in));
        std::FILE* out = std::fopen(file, "wb");
        if (!out) throw std::runtime_error(std::string("cannot write ") + file);
        arof::Socket s = listener.accept();
        uint64_t frames = 0, bytes = 0;
        arof::Frame f;
        bool end = false;
        while (!end && s.read_frame(f)) {
            if (std::fwrite(&f.h, sizeof f.h, 1, out) != 1 ||
                (!f.payload.empty() && std::fwrite(f.payload.data(), f.payload.size(), 1, out) != 1))
                throw std::runtime_error(std::string("write failed: ") + file);
            ++frames;
            bytes += sizeof f.h + f.payload.size();
            end = f.is_end();
        }
        std::fclose(out);
        std::fprintf(stderr, "%s: %llu frames, %llu bytes to %s%s\n", name.c_str(), static_cast<unsigned long long>(frames),
                     static_cast<unsigned long long>(bytes), file, end ? ", END" : " (the tap closed without END)");
        return end ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: %s\n", name.c_str(), e.what());
        return 1;
    }
}
