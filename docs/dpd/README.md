# The reference DPD

`dpd/`, a Python package (PyTorch). It defines the model, trains it, and gives the outputs the
C++/CUDA twin must reproduce; on the PC, MATLAB calls it. numpy at the boundary: complex arrays in
and out, float32 inside, on the CPU or CUDA. Every equation and constant it computes:
[METHODOLOGY](METHODOLOGY.md).

| Module | Content |
|---|---|
| `models.py` | `TransformerDPD`, the network; `DPD`, its numpy wrapper: `predistort(x)`, `predistort_reim(re, im)` (MATLAB's form: two float64 vectors), `fit(u, t)`, `save` / `load`, `describe` |
| `ila.py` | indirect learning: `target_gain`, `ila_step` (one recording), `ila` (passes against a callable link) |
| `pa.py` | the test PA: `rf_amp.m` ported line for line, with the nominal link's parameters; `pa_chain` adds and removes the pilot tone. A test fixture, not a link stage |
| `signals.py` | 16QAM through the RRC filter, and the matched filter |
| `metrics.py` | NMSE, the EVM of the waveform (in-band), the EVM of the symbols |

## The model

A window of 32 complex drive samples — 16 before the output sample, the sample, 15 after — as I/Q
pairs; a linear embedding to width 32 plus a learned position embedding; 2 pre-norm transformer
encoder layers (4 heads, feed-forward 64, GELU); a layer norm and a linear head on the centre token,
added to the centre sample. The head starts at zero, so an untrained model is the identity. Inputs
are divided by a stored scale, so the model works in any unit.

| | |
|---|---|
| Parameters | 18 338 |
| Cost | 657 472 MAC per output sample (1.31 MFLOP); 403 520 with the last layer for the centre token only |
| Delay | none: output sample n is aligned with input sample n |

```python
from dpd.models import DPD
d = DPD.load("results/models/transformer_nominal.pt")      # CUDA if present, else CPU
x_dpd = d.predistort(x)                                     # complex numpy in, same length out
```

## Training: indirect learning

`ila(dpd, link, x, passes=3, epochs=30)`. Pass 1 runs without DPD. Each pass records the PA's
output `y` for the drive `x_d` actually sent, trains a copy of the DPD so that `g(y/G) ≈ x_d`, and
copies it in front; `G` is the complex gain of the link without DPD. On the testbed, `link` becomes
a run of the pipeline: `x_d` from the BASEBAND frames the stage sends, `y` from the PA's lossless
tap, matched by `seq`.

## The test PA

The nominal link's PA: Rapp AM/AM (saturation 1.5 a0, smoothness 1.94) and Saleh AM/PM (α 0.51,
β 1.93), the fit of the 3GPP TR 38.803 GaN PA; the memory polynomial (0.6 ns, linear
[-0.08, 0.03], cubic [-0.05, 0.02]) is off, as in the nominal link, and on in a second case. The
chain drives it with the data (0.371 a0 RMS) plus the pilot tone (1.104 a0) — the levels the link
gives the PA — and removes the tone from the output as its mean.

The port is checked against values computed by hand from the `.m` file's formulas
(`tests/test_pa.py`, agreement to 1e-12):

| Input | Expected |
|---|---|
| \|A\| = 1.5 (= saturation), AM/AM only | 1.5 / 2^(1/3.88) = 1.2546026733935693 |
| A = 1, received power 2 dB below the reference (AGC) | input 1.5848931924611134, output 1.287725872296698 |
| 0.6 e^{j0.3}, AM/AM and AM/PM | magnitude 0.5956588580691601, phase +0.062347188264058676 rad |
| memory at 3.2 GS/s (delay round(1.92) = 2 samples), u = [0.5, 0, 0, 0, 0.8j] | y[2] = -0.04625, y[4] = 0.0175 + 0.8j |
| memory delay of 2.5 samples | 3 samples (MATLAB's `round`; numpy's would give 2) |

Without DPD, the chain's EVM is 17.6 %; the MATLAB link's PA alone gives 17.3 % at the same
operating point.

## Results

Step 1, `scripts/step1_check.py` on the GPU: 16QAM at 8 samples per symbol, 4096 training symbols
(seed 1), 2048 validation (seed 4242) and test (seed 2017) symbols; 3 ILA passes of 30 epochs,
~170 s. Test sequence (`results/step1.txt`; validation agrees to 0.1):

| PA | NMSE, dB | EVM, waveform, % | EVM, symbols, % |
|---|---|---|---|
| nominal | -14.86 → -48.45 | 17.56 → 0.22 | 17.29 → 0.20 |
| nominal, memory on | -10.64 → -16.19 | 28.61 → 9.89 | 27.36 → 7.96 |

## Tests

`python -m pytest -q`, 19 tests, ~80 s:

| File | Checks |
|---|---|
| `test_pa.py` | the PA port against the hand-computed values; the parameters; MATLAB's rounding |
| `test_metrics.py` | a gain and a constant are not errors; a known error is measured as such |
| `test_models.py` | identity at start; dtypes and MATLAB's form; window alignment and span; save/load; size; CPU against CUDA on a 6400-sample frame |
| `test_methodology.py` | [METHODOLOGY](METHODOLOGY.md), written again in numpy, against the model (1e-5) |
| `test_ila.py` | a short ILA run improves NMSE by 6 dB and the EVM by 40 % |
