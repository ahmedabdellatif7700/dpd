# dpd: the DPD stage of the A-RoF testbed, on the Jetson Orin Nano

The digital predistorter (DPD) of the 60 GHz A-RoF link (16QAM, 0.4 GBd, 15 km SSMF, a GaN PA at
the remote antenna unit), at the central office: it predistorts the baseband drive of the IQ
modulator, so that the chain up to the PA's output becomes linear; no equalizer at the receiver.
The end result is **one stage of the link pipeline**, the `dpd` stage, in place of the Pi's
pass-through `dpd_bb` service (DPD place A), with the same stage contract as the other ten stages;
only its image (the Jetson, CUDA) and its host differ.

This repository will be merged into `on-chip-training` (branch `dpd`): the Python DPD into
`matlab/dpd/py/`, the stage into `testbed/jetson/`. It uses no paths into that repository; what is
copied from it says where from. The paper's plan is `matlab/dpd/README.md` there.

| Part | What | Where |
|---|---|---|
| Reference DPD | PyTorch: the models, ILA training, a test PA (port of `rf_amp.m`); numpy at the boundary, so MATLAB calls it through `bridge.py` on the PC (CPU) | `dpd/`, `tests/`, `scripts/` |
| Hardware DPD | C++ with hand-written CUDA kernels, the weights trained by the reference; frame v2 and the stage contract from a copy of the Pi's stage library | `cpp/` (steps 2-3) |
| Container | the stage's image: CUDA 13.2 build and runtime, NVIDIA runtime, compose service `dpd_bb` | `docker/` |
| Docs | the Jetson, the env, the image, how to run | [docs/testbed/jetson/README.md](docs/testbed/jetson/README.md) |

Development runs in the conda env `machine-learning` (Miniconda, [environment.yml](environment.yml));
the stage runs in Docker. Long jobs in a detached tmux session.

## Steps

| # | Step | Output | Status |
|---|---|---|---|
| 1 | Reference transformer DPD in PyTorch, ILA, a test PA; NMSE/EVM with and without the DPD | `dpd/`, `tests/`, `scripts/step1_check.py`, `results/step1.txt` | done |
| 2 | The transformer in C++/CUDA (hand-written kernels) on the Jetson's GPU, float32: against the PyTorch reference, throughput at the BASEBAND frame size | `cpp/` | |
| 3 | The `dpd` stage container: frame v2, the stage contract (a copy of the Pi's stage library), `bypass` and `dpd` modes, the PA tap; tested with a Python feeder and sink; the hookup to the Pi proposed (not done) | `cpp/`, `docker/`, `tests/` | the image's base done |

Open, to bring to the user: **κ**, the drive scale. `co_optics` computes it from the calibration
block it receives, which with a DPD in front is the predistorted one; the MATLAB DPD acts on `κ·s`
with κ from the undistorted signal. A proposal comes with step 3.

## Record of the steps

### Step 1 (2026-11-17): the reference transformer DPD

**Environment.** Miniconda in `~/miniconda3`, env `machine-learning`: Python 3.12.16, PyTorch
2.11.0+cu130 (NVIDIA's JetPack 7 wheel, Jetson AI Lab index `sbsa/cu130`), numpy 2.5.5, scipy
1.18.1; `torch.cuda.is_available()` true on the Orin (CC 8.7; the wheel's sm_80 code runs there).

**Model** (`dpd/models.py`, `TransformerDPD`). A window of 32 complex drive samples as I/Q pairs
(output sample n sees n-16 .. n+15, zeros beyond the sequence's ends, no delay); a linear embedding
to width 32 plus a learned position embedding; a transformer encoder of 2 pre-norm layers, 4 heads,
feed-forward 64, GELU; a layer norm and a linear head on the centre token, added to the centre
sample (a skip; the head starts at zero, so an untrained model is the identity). Inputs are divided
by a stored scale (the RMS of the first training input), so the model works in volts or in a0.

- **18 338 parameters**; **657 k multiply-accumulates per output sample** (1.31 MFLOP; the whole
  window is encoded for every sample: Q/K/V and output projections, scores, feed-forward; softmax,
  GELU and norms not counted).
- `DPD` wraps it for numpy: `predistort(x)` (complex in, complex out, any length, CPU or CUDA,
  float32 inside), `predistort_reim(re, im)` (MATLAB's two float64 vectors), `fit(u, t)`,
  `save`/`load`.

**ILA** (`dpd/ila.py`). A post-inverse g, warm-started from the current DPD, is trained so that
g(y/G) ≈ x_d (x_d the drive that went through the link, y the PA's output with the tone removed,
G the complex LS gain of y on x without DPD), then copied in front as the DPD; the first pass starts
from no DPD. `ila_step` takes one recording (the form of the Pi's data: BASEBAND and the lossless PA
tap); `ila` runs passes against a callable link.

**Test PA** (`dpd/pa.py`, a fixture, not a link stage). `rf_amp.m` ported line for line (AGC, Rapp
AM/AM, Saleh AM/PM, the memory polynomial; MATLAB's `round` and `eps` kept), with the nominal link's
3GPP GaN PA: asat 1.5 a0, smoothness 1.94, AM/PM α 0.51, β 1.93; memory (0.6 ns, linear
[-0.08 0.03], cubic [-0.05 0.02]) off as in the nominal link, on in a second case. The test chain
drives it with the data plus the pilot tone at the levels the link gives the PA (tone 1.104 a0, data
RMS 0.371 a0, from `matlab/results/pa_operating_point.txt`) and removes the tone as the output's
mean. Checked (`tests/test_pa.py`) against values computed by hand from the .m file's formulas:

| Input | Expected (by hand) |
|---|---|
| \|A\| = asat = 1.5, AM/AM only | 1.5 / 2^(1/3.88) = 1.2546026733935693 |
| A = 1, P_RX 2 dB below the reference (AGC) | A_in = 1.5848931924611134, out 1.287725872296698 |
| 0.6·e^{j0.3}, AM/AM and AM/PM | \|out\| 0.5956588580691601, phase +0.062347188264058676 rad |
| memory at 3.2 GS/s (D = round(1.92) = 2), u = [0.5 0 0 0 0.8j] | y[2] = -0.04625, y[4] = 0.0175 + 0.8j |
| delay of 2.5 samples | D = 3 (MATLAB's round; numpy's would give 2) |

**The check** (`scripts/step1_check.py`, GPU). 16QAM RRC (roll-off 0.75, span 16) at 8 samples per
symbol (3.2 GS/s, the DPD rate of `matlab/dpd/README.md`); training sequence 4096 symbols (seed 1),
validation and test 2048 (seeds 4242 and 2017); ILA 3 passes × 30 epochs, batch 512, Adam 2e-3 with
a cosine schedule. Measures (`dpd/metrics.py`), against the undistorted drive, each with a fitted
complex gain and constant (the tone is removed as a constant): NMSE over the whole band; the EVM of
the waveform in-band as `main_pa_operating_point.m` defines it; the EVM of the matched-filtered
symbols (the receiver up to its decisions). Test sequence (`results/step1.txt`):

| PA | Training (GPU) | NMSE, dB | EVM waveform, % | EVM symbols, % |
|---|---|---|---|---|
| nominal (Rapp + AM/PM) | 169 s | -14.86 → **-48.45** | 17.56 → **0.22** | 17.29 → **0.20** |
| nominal + memory | 167 s | -10.64 → -16.19 | 28.61 → 9.89 | 27.36 → 7.96 |

The validation sequence gives the same to 0.1. Without DPD the test chain's EVM (17.6 %) matches
the MATLAB link's PA alone at 1.5 a0 (17.3 %, `pa_operating_point.txt`), a check of the port and the
levels. The DPD raises the PA input's peak-to-average ratio from 3.3 to 5.4 dB (the expansion the
compression needs).

- **Nominal PA**: linearized to the float32 and training floor (post-inverse NMSE -56.6, -57.3,
  -58.1 dB over the passes), far inside the 3GPP 12.5 % EVM.
- **With memory**: the DPD halves the error but stops at -16 dB, although the post-inverse fits to
  -41 dB: the classic ILA gap with memory (a post-inverse is not a pre-inverse once nonlinearity and
  memory mix; the cubic memory term on data plus tone also makes an image, x*, term). Memory is off
  in the nominal link (an impairment-study option); this case is left open for the model screens.
- A first run showed a floor of -35 dB NMSE and 1.7 % symbol EVM that was the measure's, not the
  DPD's: the tone's removal as the mean takes the data's own small mean with it. The measures now fit
  a constant, as `main_pa_operating_point.m`'s EVM does.

Tests: `python -m pytest -q`, 18 passed (78 s): the PA port, the model (identity at start, dtypes and
the bridge's form, the window alignment and span, save/load, size, CPU against CUDA on a 6400-sample
frame), a short ILA run that must improve NMSE by 6 dB and the EVM by 40 %.

### The container's base (2026-11-17, ahead of steps 2-3)

`docker/` and `cpp/`, after the user's two templates (a Jetson CUDA Docker project and a C++ CMake
project; details and departures in [docs/testbed/jetson/README.md](docs/testbed/jetson/README.md)):
`nvidia/cuda:13.2.0` devel to build (CMake + Ninja, sm_87, ccache), runtime to run, the compat
workaround, `runtime: nvidia`, the compose service `dpd_bb` with `restart: "no"`, a Makefile with
Jetson and SSH targets. `make -C docker check` builds the image (`arof-dpd:local`, 5.8 GB) and runs
its self-test `gpu_check` in it: CUDA runtime and driver 13.2, the Orin (CC 8.7, 8 SMs) found, a
native sm_87 kernel on one BASEBAND frame (6400 complex samples) with error 0. The stage itself
comes in step 3, the kernels of the transformer in step 2.
