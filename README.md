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
([METHODOLOGY](docs/dpd/METHODOLOGY.md)), the Jetson and the image ([testbed/jetson/](docs/testbed/jetson/README.md)).

## Layout

```
dpd/                the reference DPD (Python package)
tests/              its unit tests (pytest)
scripts/            checks that produce results/ (step1_check.py)
results/            step reports; trained models (results/models/: *.pt for PyTorch, *.dpdt for the
                    twin, golden/*.dpdt to check it)
cpp/                the C++/CUDA twin and the stage (CMake)
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
python -m pytest -q                                  # unit tests, ~80 s
python scripts/step1_check.py --device cuda          # ILA on the test PA, ~6 min (run in tmux)
make -C docker check                                 # build the image, GPU self-test in the container
```

Details: [docs/howto/jetson.md](docs/howto/jetson.md).

## Steps

| # | Step | Output | Status |
|---|---|---|---|
| 1 | Reference transformer DPD in PyTorch; ILA on a test PA; NMSE and EVM with and without the DPD | `dpd/`, `tests/`, `scripts/step1_check.py`, `results/step1.txt` | done |
| — | Container base: the stage's image (CUDA 13.2, sm_87) and the C++ build | `docker/`, `cpp/` | done |
| 2 | The transformer as hand-written CUDA kernels on the Jetson's GPU (float32): checked against the reference; throughput at the BASEBAND frame size | `dpd/export.py`, `cpp/` | in progress (branch `cuda-kernels`): part 1 of 5 done |
| 3 | The `dpd` stage: frame v2 and the stage contract (a copy of the Pi's stage library), `bypass` and `dpd` modes, the PA tap; tested with a Python feeder and sink; the hookup to the Pi proposed | `cpp/`, `docker/`, `tests/` | |

## Open decisions

- **Drive scale κ.** `co_optics` computes κ from the calibration block it receives; with a DPD in
  front, that block is predistorted, whereas the MATLAB DPD acts on `κ·s` with κ from the
  undistorted signal. A proposal for how `dpd_bb` and `co_optics` agree on κ comes with step 3.
- **The DPD's rate.** The reference model runs at 8 samples per symbol (3.2 GS/s, the rate of the
  DPD plan); the pipeline's BASEBAND is at 50 (20 GS/s). Either the stage resamples around the
  model, or the model is trained at 50. To settle before step 3.
- **Frame boundaries.** The model's window reaches 16 samples back and 15 ahead. The reference and
  the golden file take a frame alone (zeros beyond its ends); in the pipeline, consecutive
  BASEBAND frames are one continuous drive, so the stage must keep the previous frame's last 16
  samples and wait for the next frame's first 15 (a 15-sample delay, one frame of look-ahead).
  How that fits the one-frame-in, one-frame-out rule comes with step 3.
- **The PA with memory.** ILA stops at -16 dB NMSE when the PA's memory option is on (off in the
  nominal link); left to the model screens.

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

Plan of the rest of step 2: (2) the model in plain C++ on the CPU, against the golden files;
(3) the CUDA kernels, one fused kernel per frame first, then optimized; (4) the check against the
reference, and the PA-output NMSE and EVM with the twin; (5) the throughput and latency per frame.

## Provenance

Copied or ported from `on-chip-training`, branch `dpd`:

| Here | From | What |
|---|---|---|
| `dpd/pa.py` | `matlab/link/rf_amp.m` | the PA model, ported line for line |
| `dpd/pa.py` | `matlab/link/default_link_odsb.m`, `matlab/core/impairment_sweeps_odsb.m` (`joint_nominal`) | the PA's parameters; the nominal case (saturation 1.5 a0, AM/PM on) |
| `dpd/pa.py` | `matlab/results/pa_operating_point.txt` | the PA input's levels: pilot tone 1.104 a0, data 0.371 a0 RMS |
| `dpd/signals.py` | `matlab/link/default_tx.m`, `transmit_receive_odsb.m` | 16QAM, RRC roll-off 0.75, span 16; `rcosdesign` and `upfirdn` ported |
| `dpd/metrics.py` | `matlab/paper_rewrite/validation/main_pa_operating_point.m` | the EVM of the PA's output (gain and constant fitted, in-band) |
| `dpd/models.py`, `dpd/ila.py` | `matlab/dpd/README.md` | the DPD plan: the transformer's shape, ILA, the DPD rate, the seeds |
| `docker/`, `cpp/` | `docs/testbed/rpi/PIPELINE.md` | the stage contract, the service name `dpd_bb`, `restart: "no"` |
