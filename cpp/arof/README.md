# cpp/arof: the Pi's stage library, copied

The stage library of the Raspberry Pi's link pipeline, copied **verbatim** so that the `dpd` stage
follows the stage contract exactly as the Pi's ten other stages do (frame v2, the CONFIG handshake,
one frame out per frame in, forwarding, holding during a calibration, the tap, the timing log, the
order of the stop; on-chip-training `docs/testbed/rpi/PIPELINE.md`). Namespace `arof`; the include
paths are the Pi's (`"stage/stage.h"`, `"stream/frame.h"`), with `cpp/arof` as the include root.

Do not edit these files here: a change belongs in on-chip-training, then is copied again. The `dpd`
stage builds on them (`cpp/src/`, `cpp/tools/`).

## Source

on-chip-training, branch `dpd`, commit `14a72ff` (2026-11-17), `testbed/rpi/`:

| Here | There | SHA-256 |
|---|---|---|
| `stage/frame2.h` | `src/stage/frame2.h` | `0954e659fdc1cd6a247b341245248017459d5f7c2306cde1f30bed948ad4cbc1` |
| `stage/net.h` | `src/stage/net.h` | `e67c679e91c788b46011addbf604492712d7633e5aa3ebb60d60b798910fef02` |
| `stage/net.cpp` | `src/stage/net.cpp` | `94ca2ab9e980bc1c29627e71dad46dba04ea0c98a2a225e55752ce7b74784e7b` |
| `stage/stage.h` | `src/stage/stage.h` | `513a4d6c252e1cee9bb5de0930faccad8b56afb02643ae9bcda3627a615c7347` |
| `stage/stage.cpp` | `src/stage/stage.cpp` | `f0ba4e623a39741c3aa75636c31b7d32a1199f00f57545a40b3c6e3c9c2b197c` |
| `stream/frame.h` | `src/stream/frame.h` (version 1, which `frame2.h` builds on) | `bd3ecc57a7f84860b8aad34282a88ddb89f414aae04d07f26b572822c36c1a62` |
| `tests/test_stage.cpp` | `tests/test_stage.cpp` (the library's own test) | `b7b9b45567e0663fc69b7cc433555c9327b6fe8e5e05f465024db26393ed2e9f` |
| `tests/check.h` | `tests/check.h` | `fb353086295fcc221ed80988e0dd91563cead7bf9052337cb9ea1c098666d47d` |

To check the copy against the source (both repositories side by side):

```bash
cd cpp/arof && sha256sum stage/* stream/* tests/*.h tests/*.cpp       # against the table
diff -r stage ../../../on-chip-training/testbed/rpi/src/stage           # no output: identical
```

Built as the Pi builds it (its `CMakeLists.txt`): the library `arof_pipe` with nlohmann/json
(from the system, or a pinned download of v3.11.3) and threads. It also compiles clean here as
C++20 with `-Wpedantic`, and its test passes on the Jetson (`make -C cpp test`, `stage_library`).
