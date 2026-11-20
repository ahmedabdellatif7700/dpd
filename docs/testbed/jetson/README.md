# testbed/jetson

The Jetson Orin Nano runs **only the DPD**: the `dpd` stage of the link pipeline, in place of the
Raspberry Pi's pass-through `dpd_bb` service (DPD place A, the baseband drive before `co_optics`).
The Pi's ten other stages and the PA tap's sink stay on the Pi; none of the Pi's stages or its
`arof_stage` image is built or run here. The pipeline itself is documented in on-chip-training,
`docs/testbed/rpi/PIPELINE.md`.

Two environments on the Jetson, one per implementation:

| | Reference DPD | C++/CUDA twin and the stage |
|---|---|---|
| Code | `dpd/` (Python, PyTorch) | `cpp/` (C++, hand-written CUDA kernels) |
| Runs in | conda env `machine-learning` | Docker image `arof-dpd:local`, NVIDIA runtime |
| Does | trains (ILA), exports the weights, gives the expected outputs | inference only, as the pipeline's stage |

The stage's framework is a copy of the Pi's C++ stage library (frame v2, handshake, forwarding,
taps, logging), so the stage contract behaves exactly as on the other stages (step 3). The `dpd/`
package does not depend on Docker or on the Jetson: on the PC, MATLAB calls it on the CPU.

## The device

Checked 2026-11-17.

| | |
|---|---|
| Board | Jetson Orin Nano 8 GB: 6 × Cortex-A78AE (1.73 GHz), GPU `Orin` (Ampere, compute capability 8.7, 8 SMs), 7.4 GB RAM shared by CPU and GPU |
| L4T / JetPack | R39.2.1 (JetPack 7 line), Ubuntu 24.04.5, kernel 6.8.12-tegra |
| CUDA, cuDNN, TensorRT | 13.2 (`nvcc` 13.2.86), 9.20, 10.16.2 |
| Docker | 29.9.3; runtimes `nvidia` and `runc`, default `runc` (the GPU needs `--runtime nvidia`) |

## The conda env `machine-learning`

Miniconda (aarch64) in `~/miniconda3`. Packages from conda-forge; PyTorch from NVIDIA's Jetson
wheel index, whose Python (3.12) sets the env's.

```bash
conda env create -f environment.yml
# equivalent, by hand:
conda create -n machine-learning --override-channels -c conda-forge python=3.12 numpy scipy matplotlib pytest
conda run -n machine-learning pip install torch==2.11.0 \
    --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple --extra-index-url https://pypi.org/simple
```

| Package | Version | Source |
|---|---|---|
| Python | 3.12.16 | conda-forge |
| PyTorch | 2.11.0+cu130 | Jetson AI Lab index `sbsa/cu130` (JetPack 7 is SBSA: one aarch64 CUDA build for Thor and Orin) |
| CUDA runtime, cuDNN | 13.0, 9.19 | pip `nvidia-*` wheels pulled by torch; run by the host's 13.2 driver |
| numpy, scipy, matplotlib, pytest | 2.5.5, 1.18.1, 3.11.2, 9.2.0 | conda-forge |

`torch.cuda.is_available()` is true. The wheel ships machine code for sm_80, 90, 100, 110 and 120
but not sm_87, and PyTorch warns about it; sm_80 code runs on the 8.7 GPU (binary compatibility
within a major version), and the CPU-against-CUDA test passes. NVIDIA's JetPack 6 indexes have native
sm_87 builds, but for CUDA 12 and L4T R36.

## The stage's image

`docker/Dockerfile`, built from the repository root:

| Stage | Base | Content |
|---|---|---|
| build | `nvidia/cuda:13.2.0-devel-ubuntu24.04` | CMake + Ninja + ccache; `cpp/` compiled for sm_87 (native code, no JIT at start-up) |
| run | `nvidia/cuda:13.2.0-runtime-ubuntu24.04` | the binaries only, in `/opt/arof-dpd` (image 5.8 GB, against 10.4 GB for the devel base) |

CUDA 13.2 matches the host's driver; a newer CUDA in the image would need the forward-compatibility
libraries. The image moves those libraries' directories aside: the NVIDIA runtime supplies the
host's driver, and NVIDIA Container Toolkit 1.19.1's CDI `enable-cuda-compat` hook crashes on them.

`docker/compose.yaml` defines one service, `dpd_bb` — the name of the service it replaces, so its
timing log is `dpd_bb.csv` — with `runtime: nvidia` and `restart: "no"` (the stage contract: a
stage never restarts, since it could not resume its stream). The trained weights,
`results/models/`, are mounted read-only at `/models`. Until step 3 the service runs `gpu_check`
(`cpp/tools/gpu_check.cu`), the image's self-test: it reports the CUDA runtime, the driver and the
device, runs a kernel on one BASEBAND frame (6400 complex samples) and checks it against the CPU.

```text
$ make -C docker check
gpu_check: CUDA runtime 13.2, driver 13.2, 1 device(s)
gpu_check: device 0 "Orin", compute capability 8.7, 8 SMs, 7.9 GB, integrated 1
gpu_check: kernel on 6400 samples, max error 0: OK
```

### Templates

The image and the build follow two existing projects: `jetson-llama-cuda`, a CUDA Docker project
already run on this Jetson, and a C++ CMake project.

| Template convention | Here |
|---|---|
| base `nvidia/cuda:13.2.0-devel-ubuntu24.04`; the compat directories moved aside | `docker/Dockerfile` |
| CMake + Ninja, `CMAKE_CUDA_ARCHITECTURES=87`, BuildKit cache mounts for the build and ccache | `docker/Dockerfile`, `cpp/CMakeLists.txt` |
| compose with `runtime: nvidia`, `NVIDIA_VISIBLE_DEVICES`, models mounted read-only | `docker/compose.yaml` |
| a self-documenting `Makefile` (`help`), device targets and SSH targets from the laptop | `docker/Makefile` |
| `-Wall -Wextra -Wpedantic`; Debug `-g -O0`, Release `-O3`; `src/`, `include/`, `bin/`; `make`, `run`, `debug`, `clean` | `cpp/CMakeLists.txt`, `cpp/Makefile` |

Departures from the templates:

| Departure | Reason |
|---|---|
| `restart: "no"` instead of `unless-stopped` | the stage contract forbids restarts |
| a separate runtime stage | the stage runs without compiler and toolkit; a smaller image |
| C++20 instead of C++23 | the newest standard nvcc 13.2 accepts for `.cu` files |
| no CUDA linking stub | the stage uses the CUDA runtime API only |
| no healthcheck yet | the stage's readiness is its TCP handshake (step 3) |

### Images considered

NVIDIA's L4T images (`l4t-pytorch`, `l4t-jetpack`, `l4t-ml`) end with JetPack 6; from JetPack 7,
Jetson uses NVIDIA's SBSA images. For a stage with PyTorch inside, the NGC PyTorch container would
fit as follows (arm64 variants, registry read 2026-11-17). A C++/CUDA stage needs only the CUDA
images above.

| `nvcr.io/nvidia/pytorch:` | CUDA | TensorRT | PyTorch | Runs on the 13.2 driver |
|---|---|---|---|---|
| 26.02-py3-igpu | 12.9 | 10.11 | 2.11 | no: built for JetPack 6 (L4T R36) |
| 26.03-py3 | 13.2.0 | 10.16.0 | 2.11 | yes |
| 26.05-py3 | 13.2.1 | 10.16.1 | 2.12 | yes (the newest that does) |
| 26.06-py3 and later | 13.3–13.4 | 11.x | 2.13–2.15 | no |

## The stage (step 3)

The three decisions it is built on (κ, the DPD's rate, frame boundaries), with the facts, the
options and the choice made (2026-11-20): [STAGE.md](STAGE.md).

## Running

On the Jetson and from the laptop: [docs/howto/jetson.md](../../howto/jetson.md).
