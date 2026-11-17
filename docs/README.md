# dpd: documentation

The digital predistorter (DPD) of the 60-GHz analog radio-over-fiber link of on-chip-training (one
laser, an optical double-sideband carrier, 16QAM at 0.4 GBd on the upper sideband, 15 km SSMF, a
GaN PA at the remote antenna unit; no equalizer at the receiver). The DPD sits at the central office
and predistorts the baseband drive of the IQ modulator so that the chain up to the PA's output is
linear. On the testbed it is one stage of the Pi's pipeline, `dpd_bb`, running on a Jetson Orin
Nano's GPU.

```
 reference DPD (PyTorch)  ══  C++/CUDA twin  ──  dpd stage (Docker)  ◄──►  Pi pipeline (10 stages)
 dpd/, conda env              cpp/                docker/                   on-chip-training testbed/rpi/
 trains (ILA), MATLAB         inference only      frame v2, dpd_bb          source ─► dpd ─► co_optics ...
 ══ : same outputs, within float32 tolerance                                PA tap ─► (training data)
```

Two implementations of one model. The **reference** (Python, PyTorch) defines the model, trains it
by indirect learning, runs on the PC for MATLAB (CPU) and on the Jetson in the conda env (GPU). The
**twin** (C++ with hand-written CUDA kernels, in Docker) only runs inference, with the reference's
weights, as the pipeline's stage. All documentation is here; each code folder keeps a short README
that points here.

## Reading order

1. **This page**: the project, the chain of checks.
2. **[howto/](howto/)**: what to run on each device: [jetson](howto/jetson.md), [pc](howto/pc.md).
3. **The pieces**:

| Piece | Docs |
|---|---|
| The reference DPD (PyTorch) | [dpd/](dpd/README.md), [METHODOLOGY](dpd/METHODOLOGY.md) (every equation and constant, the page the twin implements) |
| The Jetson, its env and the stage's image | [testbed/jetson/](testbed/jetson/README.md) |
| The plan and the record of the steps | [../README.md](../README.md) |

The pipeline the stage plugs into is documented in on-chip-training: `docs/testbed/rpi/PIPELINE.md`
(stages, frame v2, the stage contract, calibration, taps, logging).

## The chain of checks

| Level | Against | Result |
|---|---|---|
| test PA (`dpd/pa.py`) | values computed by hand from `rf_amp.m` | equal to 1e-12 (`tests/test_pa.py`) |
| test chain, no DPD | the MATLAB link's PA alone at 1.5 a0 (`pa_operating_point.txt`) | EVM 17.6 % against 17.3 % |
| [METHODOLOGY](dpd/METHODOLOGY.md) in numpy | the PyTorch model | equal to 1e-5 (`tests/test_methodology.py`) |
| reference on CUDA | the reference on the CPU, one 6400-sample frame | equal to 1e-4 (`tests/test_models.py`) |
| reference DPD by ILA | no DPD, nominal PA, test sequence | NMSE -14.9 → -48.5 dB, EVM 17.3 → 0.20 % (step 1) |
| the image | the CPU | a kernel on one 6400-sample frame, error 0 (`gpu_check`) |
| C++/CUDA twin | the reference | step 2 |
| the `dpd` stage | the stage contract, the reference offline | step 3 |

## Not done yet

The twin (step 2), the stage (step 3), the hookup to the Pi; κ, the drive scale, between `dpd_bb`
and `co_optics`; the DPD's rate (the model at 8 samples per symbol, the pipeline's BASEBAND at 50);
the PA with memory (ILA stops at -16 dB); the memory polynomial and the 14-12-2 network as DPDs.
