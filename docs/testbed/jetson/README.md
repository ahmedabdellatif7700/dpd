# testbed/jetson

The Jetson Orin Nano of the testbed runs **only the DPD**: one stage of the link pipeline
(on-chip-training `docs/testbed/rpi/PIPELINE.md`), the `dpd` stage, in place of the pass-through
`dpd_bb` service (DPD place A, the baseband drive before `co_optics`). The other ten stages and the
PA tap's sink stay on the Raspberry Pi 4; nothing of the Pi's (`arof_stage`, its image, its stages)
is built or run here.

- Development (the model, training, export, timing; steps 1-2) runs in the conda env
  `machine-learning`, as on the PC.
- **The stage on the hardware is C++ with hand-written CUDA kernels**, in a Docker container with
  the NVIDIA runtime (`cpp/`, `docker/`; user's decision, 2026-11-17). The Python/PyTorch DPD
  (`dpd/`) is the **reference** and the training tool: it trains the weights (ILA) that the C++
  stage loads, and the C++ output is checked against it. The stage's framework is a copy of the
  Pi's C++ stage library (frame v2, handshake, forwarding, taps, logging), so the contract behaves
  as on the other ten stages.
- The `dpd/` package does not depend on Docker: on the PC, MATLAB calls it in its own conda env,
  CPU only, through `matlab/dpd/py/bridge.py`.

## The device (checked 2026-11-17)

| | |
|---|---|
| Board | Jetson Orin Nano, GPU `Orin`, compute capability 8.7, 6 CPU cores, 7.4 GB RAM shared with the GPU |
| L4T / JetPack | R39.2.1 (`/etc/nv_tegra_release`), the JetPack 7 line; Ubuntu 24.04.5, kernel 6.8.12-tegra |
| CUDA / cuDNN / TensorRT (host) | 13.2 (`nvcc` 13.2.86), cuDNN 9.20, TensorRT 10.16.2 |
| Docker | 29.9.3, runtimes `nvidia` and `runc` (default `runc`: pass `--runtime nvidia`) |

## The conda env `machine-learning`

Miniconda (aarch64) in `~/miniconda3`; the env from conda-forge, PyTorch from NVIDIA's Jetson wheels:

```
conda env create -f environment.yml          # or, by hand:
conda create -n machine-learning --override-channels -c conda-forge python=3.12 numpy scipy matplotlib pytest
conda run -n machine-learning pip install torch==2.11.0 \
    --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple --extra-index-url https://pypi.org/simple
```

| Package | Version | From |
|---|---|---|
| Python | 3.12.16 | conda-forge (3.12: the Python of the JetPack 7 wheels) |
| PyTorch | 2.11.0+cu130 | Jetson AI Lab index `sbsa/cu130` (JetPack 7 is SBSA: one aarch64 CUDA build for Thor and Orin) |
| CUDA runtime, cuDNN | 13.0 (pip `nvidia-*` wheels), cuDNN 9.19 | pulled by the torch wheel; the host's 13.2 driver runs them |
| numpy, scipy, matplotlib, pytest | 2.5.5, 1.18.1, 3.11.2, 9.2.0 | conda-forge |

`torch.cuda.is_available()` is true. The wheel carries SASS for sm_80/90/100/110/120 and none for
sm_87; PyTorch warns that 8.7 is not in its list, but sm_80 binaries run on the 8.7 GPU (CUDA's
binary compatibility within a major version): a matmul matches the CPU to float32 rounding, and the
DPD's CPU/CUDA test (`tests/test_models.py::test_cpu_vs_cuda`) passes. The JetPack 6 indexes
(`jp6/cu126..cu129`) have native sm_87 builds but for CUDA 12 and L4T R36, not this JetPack.

## The stage's image

The image and the C++ build follow the user's two templates (2026-11-17): `jetson-llama-cuda`, a
CUDA Docker project already run on this Jetson, and a C++ project (CMake + Ninja).

| From the template | Here |
|---|---|
| base `nvidia/cuda:13.2.0-devel-ubuntu24.04` | the build stage (`docker/Dockerfile`) |
| compat directories moved aside (Container Toolkit 1.19.1's CDI `enable-cuda-compat` crash; the NVIDIA runtime supplies the host's driver) | the same, in both stages |
| CMake + Ninja, `CMAKE_CUDA_ARCHITECTURES=87`, BuildKit cache mounts for the build and ccache | the same (`cpp/CMakeLists.txt`) |
| compose: `runtime: nvidia`, `NVIDIA_VISIBLE_DEVICES`, models mounted read-only | the same; the models are `results/models/` (trained by the Python reference) |
| a `Makefile` with `help`, Jetson targets and SSH targets from the laptop (`JETSON=orinnano8s`) | `docker/Makefile` |
| C++: warnings `-Wall -Wextra -Wpedantic`, Debug `-g -O0`, Release `-O3`, `src/`, `include/`, binaries in `bin/`, `make` / `make run` / `make debug` / `make clean` | `cpp/CMakeLists.txt`, `cpp/Makefile` |

Departures, each for a reason:

- **`restart: "no"`** instead of `unless-stopped`: the stage contract (PIPELINE.md 3.5) forbids a
  restart, which could not resume the stream.
- **A runtime stage**: the binaries are copied into `nvidia/cuda:13.2.0-runtime-ubuntu24.04`, so
  the stage runs without the compiler and toolkit (image `arof-dpd:local`, 5.8 GB, against 10.4 GB
  for the devel base).
- **C++20**, not C++23: the newest standard nvcc 13.2 takes for `.cu` files.
- No CUDA linking stub: the stage uses the CUDA runtime API only (no `libcuda` at link time).
- No healthcheck yet: the stage's readiness is its TCP handshake (step 3).

CUDA 13.2 is the host driver's version: an image with a newer CUDA would need the forward
compatibility libraries.

`docker/compose.yaml` has one service, `dpd_bb` (the name of the pass-through it replaces, so its
log is `dpd_bb.csv`). Until step 3 its command is `gpu_check` (`cpp/src/gpu_check.cu`), the
image's self-test: it prints the CUDA runtime and driver and the device, runs a kernel on one
BASEBAND frame (6400 complex samples) and checks it against the CPU.

```
make -C docker help            # the targets
make -C docker check           # build the image, run gpu_check in it with the NVIDIA runtime
#   gpu_check: CUDA runtime 13.2, driver 13.2, 1 device(s)
#   gpu_check: device 0 "Orin", compute capability 8.7, 8 SMs, 7.9 GB, integrated 1
#   gpu_check: kernel on 6400 samples, max error 0: OK
make -C docker jetson-check    # the same from the laptop, over SSH
make -C cpp run                # the same binary built on the host (nvcc 13.2, cmake, ninja)
```

**Other images considered.** NVIDIA's L4T images (`l4t-pytorch`, `l4t-jetpack`, `l4t-ml`) stop
at JetPack 6; from JetPack 7 Jetson uses the SBSA images. With PyTorch in the stage, the NGC
container would have fit as follows (arm64, read from the registry on 2026-11-17); it is not
needed for a C++/CUDA stage:

| `nvcr.io/nvidia/pytorch:` | CUDA | TensorRT | PyTorch | Fits the 13.2 driver |
|---|---|---|---|---|
| 26.02-py3-igpu | 12.9 | 10.11 (JetPack 6) | 2.11 | no: built for L4T R36 |
| 26.03-py3 | 13.2.0 | 10.16.0 | 2.11 | yes |
| 26.05-py3 | 13.2.1 | 10.16.1 | 2.12 | yes, the newest such |
| 26.06-py3 and later | 13.3 - 13.4 | 11.x | 2.13 - 2.15 | no |

## Running

```
conda activate machine-learning
cd ~/dpd
python -m pytest -q                          # the unit tests (CPU, and CUDA if present)
tmux new -s step1 'python scripts/step1_check.py --device cuda'   # step 1's check, results/step1.txt
```

Long jobs go in a detached tmux session (the SSH connection drops).
