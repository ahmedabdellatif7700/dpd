# dpd — digital predistortion stage for the A-RoF testbed

Digital predistortion (DPD) for the 60 GHz analog radio-over-fiber link of `on-chip-training`:
16QAM at 0.4 GBd, 15 km SSMF, a GaN power amplifier (PA) at the remote antenna unit, no equalizer
at the receiver. The DPD sits at the central office and predistorts the baseband drive of the IQ
modulator so that the chain up to the PA's output is linear.

On the testbed the DPD is **one stage of the link pipeline**: the `dpd` stage, running on a Jetson
Orin Nano's GPU in place of the Raspberry Pi's pass-through `dpd_bb` service (DPD place A). It
follows the same stage contract as the Pi's ten other stages, which stay on the Pi.

## Overview

| Part | Role | Runs in | Where |
|---|---|---|---|
| Reference DPD | defines the model, trains it by indirect learning (ILA), produces the expected outputs; called from MATLAB on the PC | conda env `machine-learning` (PyTorch, CPU or GPU) | `dpd/`, `tests/`, `scripts/` |
| C++/CUDA twin | the same model, inference only, hand-written CUDA kernels, the reference's weights | Docker, NVIDIA runtime | `cpp/` |
| The `dpd` stage | the twin inside the pipeline: frame v2, the stage contract, the PA tap | Docker, NVIDIA runtime | `cpp/`, `docker/` |

The twin is checked against the reference: same input, same output within float32 tolerance.

Documentation: **[docs/](docs/README.md)** — how to run ([howto/](docs/howto/README.md)), the
reference ([dpd/](docs/dpd/README.md)) and every equation it computes
([METHODOLOGY](docs/dpd/METHODOLOGY.md)), the Jetson and the image ([testbed/jetson/](docs/testbed/jetson/README.md)), the twin ([twin/](docs/twin/README.md)).

## Layout

```
dpd/                the reference DPD (Python package); dpd/testing/: its test fixtures (test PA,
                    signals, measures), used by the tests and the checks only
scripts/            export_model.py: the weights and golden files for the twin
checks/             the checks behind the results (step1_check, twin_check, resample_check, timing)
tests/              the unit tests (pytest)
results/            step reports; trained models (results/models/: *.pt for PyTorch, *.dpdt for the
                    twin, golden/*.dpdt to check it)
cpp/                the C++/CUDA twin and the stage (CMake): src/, include/, tools/ (the programs
                    that run: dpd_stage, tap_recorder), checks/ (the twin's checks), arof/ (the
                    Pi's stage library, copied)
docker/             the stage's image and compose service
docs/               documentation
environment.yml     the conda env
```

This repository is to be merged into `on-chip-training`: `dpd/` into `matlab/dpd/py/`, the stage
into `testbed/jetson/`. It holds no paths into that repository; what was copied from it is listed
under [Provenance](#provenance).

## Quick start (Jetson)

```bash
conda activate machine-learning
python -m pytest -q                                  # unit tests, ~100 s
python checks/step1_check.py --device cuda          # ILA on the test PA, ~6 min (run in tmux)
make -C cpp test                                     # the C++ model against the reference (golden files)
make -C docker check                                 # build the image, GPU self-test in the container
```

Details: [docs/howto/jetson.md](docs/howto/jetson.md).

## Steps

| # | Step | Output | Status |
|---|---|---|---|
| 1 | Reference transformer DPD in PyTorch; ILA on a test PA; NMSE and EVM with and without the DPD | `dpd/`, `tests/`, `checks/step1_check.py`, `results/step1.txt` | done |
| — | Container base: the stage's image (CUDA 13.2, sm_87) and the C++ build | `docker/`, `cpp/` | done |
| 2 | The transformer as hand-written CUDA kernels on the Jetson's GPU (float32): checked against the reference; throughput at the BASEBAND frame size | `dpd/export.py`, `cpp/` | done (branch `cuda-kernels`, to merge after review) |
| 3 | The `dpd` stage: frame v2 and the stage contract (a copy of the Pi's stage library), `bypass` and `dpd` modes, the PA tap; tested with a Python feeder and sink; the hookup to the Pi proposed | `dpd/resample.py`, `dpd/stream.py`, `cpp/`, `docker/`, `tests/` | in progress (branch `dpd-stage`): the resampler, the stage library, the stage, the hookup proposed |

## Decisions for the stage (2026-11-20)

Facts, options and measurements: [docs/testbed/jetson/STAGE.md](docs/testbed/jetson/STAGE.md).

- **Drive scale κ: 1a.** `dpd_bb` holds the calibration block, computes κ from the undistorted
  signal with `co_optics`'s formula and writes it into its `pipeline` line; `co_optics` uses it
  (a small change on the Pi, prepared as a patch for the user, not applied). Without it κ would fall to 0.65 of
  its value (−3.7 to −3.9 dB) on the predistorted block.
- **Rate: 2b.** The model runs at 8 samples per symbol; the stage resamples 50 → 8 → 50, with one
  resampler in the reference and the twin, checked like the model.
- **Frame boundaries: 3b** (first 3a; changed after the resampler's measurement). The stage holds
  back the last 250 samples of each frame, as `fiber` does, so that the filters and the window see
  the next frame: equal to the drive taken whole (-47.79 dB), where cutting each frame alone costs
  12.5 dB (-35.25 dB). The calibration group is flushed whole; `co_optics` locates the block by
  `first_sample` (in the same patch as κ).

Still open: **the PA with memory** (ILA stops at −16 dB NMSE when the PA's memory option is on, off
in the nominal link); left to the model screens.

## Record

### Step 1 — reference transformer DPD (2026-11-17)

A transformer DPD (window of 32 complex samples, 2 pre-norm encoder layers, width 32, 4 heads;
**18 338 parameters, 657 k MAC per sample**), trained by ILA on a Python port of the link's PA
(`rf_amp.m`, the nominal 3GPP GaN PA, driven with data plus pilot tone at the link's levels).
Training on the GPU: 3 ILA passes, **~170 s**. Test sequence (seed 2017), `results/step1.txt`:

| PA | NMSE, dB | EVM, waveform, % | EVM, symbols, % |
|---|---|---|---|
| nominal | -14.86 → **-48.45** | 17.56 → **0.22** | 17.29 → **0.20** |
| nominal, memory on | -10.64 → -16.19 | 28.61 → 9.89 | 27.36 → 7.96 |

- Without DPD the test chain's EVM, 17.6 %, agrees with the MATLAB link's PA alone (17.3 %): a
  check of the port and of the signal levels.
- The nominal PA is linearized to the training floor, far inside the 3GPP 16QAM limit of 12.5 %.
  The DPD raises the PA input's peak-to-average ratio from 3.3 to 5.4 dB.
- With the PA's memory on, the post-inverse fits to -41 dB but the predistorter reaches only -16 dB:
  the known limit of ILA once nonlinearity and memory mix. Open (see above).
- A first run showed a floor of -35 dB that came from the measure, not the DPD: removing the pilot
  tone as the output's mean also removes the data's own mean. The measures now fit a constant, as
  the link's EVM definition does.

Model, ILA, test PA and measures: [docs/dpd/](docs/dpd/README.md). Environment: Python 3.12, PyTorch
2.11.0+cu130 (NVIDIA's JetPack 7 wheel), CUDA available on the Orin. 19 unit tests pass.

### Container base (2026-11-17)

The stage's image, `arof-dpd:local`: built on `nvidia/cuda:13.2.0-devel` (CMake, Ninja, sm_87),
run on `nvidia/cuda:13.2.0-runtime`, with the NVIDIA runtime; compose service `dpd_bb`. Its
self-test, `gpu_check`, finds the Orin (CC 8.7, 8 SMs, CUDA runtime and driver 13.2) from inside
the container and runs a kernel on one BASEBAND frame (6400 samples) with error 0. Design and
the templates it follows: [docs/testbed/jetson/](docs/testbed/jetson/README.md).

### Step 2, part 1 — export for the twin (2026-11-17)

The reference writes what the C++/CUDA twin reads, in one plain format of named float32 tensors
([FORMAT](docs/dpd/FORMAT.md)): the weights (`results/models/*.dpdt`, 74.5 kB, 33 tensors) and a
golden file per model (`results/models/golden/*.dpdt`, 202 kB): a 6400-sample BASEBAND-sized frame,
its output, and the intermediate values (after the embedding, each layer, the final norm) of 8
probed samples at the frame's ends and middle. The weights load back into the reference with
outputs identical bit for bit. 24 unit tests pass.

### Step 2, part 2 — the model in plain C++ (2026-11-17)

`cpp/`: the DPDT reader, the weights (every shape checked), and the forward pass in plain C++,
float32, transcribed from [METHODOLOGY](docs/dpd/METHODOLOGY.md): the twin's own reference for the
CUDA kernels. Against the golden files of both models (`make -C cpp test`, ctest; the same in the
image): output error 1.0e-6 and 9.2e-7, every probed intermediate within 1.4e-6 (relative to the
RMS; float32 rounding, which does not grow through the layers). 1 200 samples/s on one CPU core.
Details: [docs/twin/](docs/twin/README.md).

### Step 2, part 3 — the fused CUDA kernel (2026-11-20)

`GpuModel` (`cpp/src/gpu_model.cu`): the transformer as one CUDA kernel per frame, one 128-thread
block per output sample, the window and every intermediate in shared memory, a warp per token (the
width, 32, is a warp's). `CpuModel` now has the same interface (`predistort`, `trace`), and both are
checked by the same code (`cpp/checks/model_check.h`). Against the golden files: output within 8.0e-7
and 9.2e-7 of the reference, every intermediate within 1.7e-6, and within 7.6e-7 of `CpuModel`; the
same in the image. Speed, the kernel alone: **45.5 ms per 6400-sample frame, 141 000 samples/s**,
117 times one CPU core; enough for the 40 000 samples/s of 5 000 symbols/s at 8 samples per symbol,
not for the 250 000 at 50. Details: [docs/twin/](docs/twin/README.md).

### Step 2, part 3 (continued) — the kernel optimized (2026-11-20)

Four changes, each measured and each giving bit-identical outputs: the last layer for the centre
token only (45.5 → 27.0 ms per frame), 8 tokens per warp at once (→ 25.5 ms), no shared-memory bank
conflict in the attention (→ 20.4 ms), inputs read four at a time (→ **16.65 ms per 6400-sample
frame, 384 000 samples/s**). 2.73 times the first version; the 250 000 samples/s that 5 000
symbols/s need at 50 samples per symbol are met with 54 % to spare. Details:
[docs/twin/](docs/twin/README.md).

### Step 2, part 4 — the twin through the test PA (2026-11-20)

`checks/twin_check.py`: the step 1 validation and test drives predistorted by the reference and by
the twin (`cpp/bin/dpd_apply`, GPU and CPU), then through the test PA. The twin's outputs are within
8.8e-7 of the reference's, and its NMSE and EVM equal the reference's to 9e-7 (dB, %), for both models:
nominal PA -48.446 dB and 0.205 % on the test sequence, as the reference. The step 1 drive and
measures moved into `dpd/testing/evaluate.py`, shared by both checks (step 1's numbers unchanged). 27 unit
tests pass, among them the twin on a short drive. Details: [docs/twin/](docs/twin/README.md).

### Step 2, part 5 — latency and throughput per frame (2026-11-20)

`cpp/tools/dpd_bench` times each frame as the stage will run it: copy in, kernel, copy out
(`checks/step2_timing.sh`, `results/step2_timing.txt`). Paced at the pipeline's rate (a 128-symbol
frame every 25.6 ms), the latency is **17.3 ms median, 17.6 ms at most at 50 samples per symbol**
(6400-sample frames, the GPU busy 67 %), and **8.9 ms median at 8** (1024-sample frames): both keep
up. The copies cost 0.06 to 0.13 ms (shared memory). At 8 samples per symbol the light load lets
the driver drop the GPU clock from 918 to 306 MHz, tripling the kernel's time (2.71 → 8.05 ms);
pinning the clock is a system setting, left unchanged. Details: [docs/twin/](docs/twin/README.md).

Step 2 is complete: the transformer runs as hand-written CUDA kernels on the Jetson's GPU, equal to
the PyTorch reference to 1e-6, and fast enough for the pipeline at either rate.

### Step 3, part 1 — the resampler (2026-11-20)

`dpd/resample.py`: 50 ↔ 8 samples per symbol (4/25 and 25/4), one Kaiser low-pass at 80 GS/s
(503 taps, 0.002 dB ripple to 1.2 GHz, 80 dB from 2.0 GHz), one explicit formula, equal to
`scipy.signal.upfirdn` to 1e-15, aligned with the waveform made at 8 (−68.7 dB), round trip
−70.8 dB. Through the test PA, the DPD at 50 samples per symbol through the resampler loses 0.7 dB
against the reference at 8 (−47.79 against −48.45 dB, EVM 0.205 % both) on the drive whole, but
**12.5 dB with each frame alone** (−35.25 dB): the filters ring at every cut. Decision 3 therefore
changed from 3a to 3b (the user, 2026-11-20): `dpd/stream.py` holds back 250 samples per frame and
gives the whole drive's transform to 1e-6. The twin's `CpuResampler` and `GpuResampler` match the
reference within 8.8e-7 and take 0.065 ms per frame on the GPU. 36 unit tests and 5 ctest checks
pass. `results/step3_resampler.txt`; [STAGE.md](docs/testbed/jetson/STAGE.md).

### Step 3, part 2 — the stage library (2026-11-20)

`cpp/arof/`: the Pi's stage library copied verbatim from on-chip-training `14a72ff`
(`testbed/rpi/src/stage/`, the version 1 frame header it builds on, and its test), with the source
and the checksum of every file in `cpp/arof/README.md`; `diff` against the source shows no
difference. Built as the Pi builds it (nlohmann/json from the system or a pinned download; apt in
the image); it compiles clean here as C++20 with `-Wpedantic`, and its own test passes on the Jetson
and in the image. The `dpd` stage is built on it next.

### Step 3, part 3 — the `dpd` stage (2026-11-20)

`cpp/bin/dpd_stage` on the copied stage library: `bypass` forwards every frame untouched; `dpd`
holds each calibration group, computes κ from its undistorted block with co_optics's formula, writes
it into its `pipeline` line, and predistorts `s → DPD(κ·s)/κ` (50 → 8, the model, 8 → 50) through a
250-sample holdback (`FrameStream`, the C++ counterpart of `dpd/stream.py`), the group leaving whole.
Tested between a Python feeder and sink standing in for the Pi (`tests/test_dpd_stage.py`, frame v2
in `dpd/frame2.py`), on the host and in the image: bypass byte for byte; in dpd mode κ bit for bit
co_optics's, the BASEBAND within 9.9e-7 (CUDA) and 1.2e-6 (CPU) of the reference applied offline,
one frame out per frame in with seq, flags and order kept; a wrong pipeline stops it with a message
naming both stages. The compose service now runs it (port 7000). 40 Python tests and 6 ctest checks
pass. Details: [STAGE.md](docs/testbed/jetson/STAGE.md).

### Step 3, part 4 — the hookup, proposed (2026-11-20)

[HOOKUP.md](docs/testbed/jetson/HOOKUP.md): the Pi (192.168.91.80) and the Jetson (192.168.91.89) on
one wired subnet; the Pi's `source` sends to the Jetson's stage on port 7000, which sends to
`co_optics` on the Pi's port 7002; `dpd_bb` on the Pi is not started. On the Pi, nothing is changed
from here: a compose override (`patches/on-chip-training/compose.jetson.yaml`, checked with
`docker compose config`) and the `co_optics` patch (`0001-co_optics-kappa-from-dpd_bb.patch`:
κ from the stage before it, the block located by `first_sample`; bit for bit as before in bypass;
checked to apply, not compiled). The ILA data go to the Jetson through taps: the PA's output and the
stage's own output, recorded by `tap_recorder` (tested: the stage's tap recorded byte for byte).
Clocks: chrony with the Pi serving the Jetson (the Jetson now follows the public pool with 19 ms root
distance, too coarse for the latency logs), for the user to set up. The bring-up: bypass through the
Jetson (the board's bytes unchanged), a recording, training on it, then dpd mode.

## Provenance

Copied or ported from `on-chip-training`, branch `dpd`:

| Here | From | What |
|---|---|---|
| `dpd/testing/pa.py` | `matlab/link/rf_amp.m` | the PA model, ported line for line |
| `dpd/testing/pa.py` | `matlab/link/default_link_odsb.m`, `matlab/core/impairment_sweeps_odsb.m` (`joint_nominal`) | the PA's parameters; the nominal case (saturation 1.5 a0, AM/PM on) |
| `dpd/testing/pa.py` | `matlab/results/pa_operating_point.txt` | the PA input's levels: pilot tone 1.104 a0, data 0.371 a0 RMS |
| `dpd/testing/signals.py` | `matlab/link/default_tx.m`, `transmit_receive_odsb.m` | 16QAM, RRC roll-off 0.75, span 16; `rcosdesign` and `upfirdn` ported |
| `dpd/testing/metrics.py` | `matlab/paper_rewrite/validation/main_pa_operating_point.m` | the EVM of the PA's output (gain and constant fitted, in-band) |
| `dpd/frame2.py` | `testbed/rpi/src/stage/frame2.h`, `testbed/rpi/tests/test_pipeline_e2e.py` (`tap_frames`) | frame version 2: the layout, the header struct |
| `dpd/models.py`, `dpd/ila.py` | `matlab/dpd/README.md` | the DPD plan: the transformer's shape, ILA, the DPD rate, the seeds |
| `docker/`, `cpp/` | `docs/testbed/rpi/PIPELINE.md` | the stage contract, the service name `dpd_bb`, `restart: "no"` |
| `cpp/arof/` | `testbed/rpi/src/stage/`, `src/stream/frame.h`, `tests/test_stage.cpp`, `tests/check.h` (commit `14a72ff`) | the stage library, verbatim |
| `cpp/CMakeLists.txt` (nlohmann/json) | `testbed/rpi/CMakeLists.txt` | the dependency lookup and the pinned download |
