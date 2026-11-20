# The `dpd` stage (step 3): the three decisions

Status: **decided by the user on 2026-11-20: 1a, 2b and 3a**, the recommendations below. The change
on the Pi that 1a needs (`co_optics` reading κ) is prepared as a patch for the user; nothing in
on-chip-training is changed from this repository. The stage takes the place of the Pi's pass-through `dpd_bb`
(on-chip-training `docs/testbed/rpi/PIPELINE.md`): it receives the `source`'s BASEBAND frames (the
RRC waveform `s`, 50 samples per symbol at 20 GS/s, 6400 samples per 128-symbol frame, complex128),
predistorts them with the twin (`GpuModel`) and sends them to `co_optics`; every other frame (CONFIG,
REF, END) passes unchanged. Three things must be settled first. Each section gives the facts (from
the Pi's code, cited), the options, measurements where they decide something, and a recommendation.

## Decision 1: the drive scale κ

**What κ is.** The factor that turns the waveform `s` into the IQ modulator's drive in volts:
`κ = drive_scale · vpp / (max Re s − min Re s)`, so that the in-phase rail swings `vpp` peak to peak
(0.5 V at the nominal operating point, paper 1 §III-A). It sets the operating point of the IQ
modulator and, through it, of the PA.

**Where it is computed now.** `co_optics` holds the calibration block (the frames flagged `CALIB`
up to `CALIB_END`), takes the extrema of `Re s` over its first `block_symbols · sps` samples,
computes κ and the field power `field_norm`, and writes both into its own line of the CONFIG frame's
`pipeline` list (`"calibration": {"kappa", "field_norm"}`; `testbed/rpi/src/stages/link_stages.cpp`,
`CoOptics::process` and `take_calibration`). `fiber` then finds `field_norm` by searching the
`pipeline` list (`FiberStage::configure`).

**The problem.** With the DPD in front, `co_optics` would see the *predistorted* block. The DPD
expands the peaks the PA compresses, so the span of `Re s` grows and κ shrinks. Measured with the
step 1 models on the step 1 drives: **κ would fall to 0.64-0.65 of its value (−3.7 to −3.9 dB)**,
scaling the whole drive down by almost 4 dB: the link would run at another operating point, where
the DPD was not trained, and most of the predistortion would be scaled away. The MATLAB link avoids
it by taking κ from the undistorted signal (`transmit_receive_odsb.m`, line 37); the pipeline must
do the same.

| | Option | Pi change | For | Against |
|---|---|---|---|---|
| **1a** | `dpd_bb` calibrates: it holds the CONFIG frame and the calibration block (the library's `calibrates()`, as `co_optics` does), computes κ from the **undistorted** block with `co_optics`'s formula, and writes it into its own `pipeline` line (`"calibration": {"kappa": κ}`). `co_optics` uses a κ found in an earlier `pipeline` line instead of its own; it still computes `field_norm` from the block it receives, the predistorted field actually sent | small, in `co_optics` only: read κ from the `pipeline` list, as `fiber` already reads `field_norm` | κ exactly the MATLAB link's; uses the existing mechanism; recomputed at every calibration (live changes); the whole chain (field power, ADC ranges, receiver gains) is calibrated on the predistorted signal it will carry | a change on the Pi; `dpd_bb` holds the block like `co_optics` (memory: 2.5 M samples, 40 MB) |
| 1b | `co_optics` takes a fixed κ from the settings (`"dpd": {"bb": {"kappa": …}}`), read from a bypass run's calibration report | small, in `co_optics` | `dpd_bb` stays simple | a manual step; κ depends on the block's extrema, so on the seed: a fixed value is only approximately the run's |
| 1c | `dpd_bb` passes the calibration block through undistorted and predistorts only the data after it | none | no change on the Pi | the receiver and the attenuator are calibrated on a signal the link then no longer carries: `field_norm` from the undistorted field, `ue_frontend`'s ADC ranges and `ue_dsp`'s gains and timing from a PA output that is compressed, while the data is linearized |

**Recommendation: 1a. Decided: 1a.** It is the MATLAB link's behaviour, it uses the mechanism the pipeline
already has, and it keeps the chain's calibration consistent with what it carries. In `bypass`
mode `dpd_bb` writes no κ, so `co_optics` computes its own as today and the pipeline stays bit for
bit the no-DPD baseline. A check: with the DPD in place but set to the identity, the κ that `dpd_bb`
writes must equal the one `co_optics` would compute. It needs your go-ahead for the change on the
Pi.

## Decision 2: the rate the DPD runs at

**Facts.** The BASEBAND frames are at 50 samples per symbol (20 GS/s), and `co_optics` expects them
there. The DPD plan's default is 8 samples per symbol (3.2 GS/s, a realistic DAC rate; 4 and 50 to
be checked on validation; on-chip-training `matlab/dpd/README.md`), and says the DPD's output is
interpolated back to the simulation rate; the step 1 models are trained at 8. The model's window is
32 samples: 4 symbols at 8 samples per symbol, 0.64 of a symbol at 50.

| | Option | GPU (part 5, paced at 25.6 ms a frame) | For | Against |
|---|---|---|---|---|
| 2a | the model at **50** samples per symbol, no resampling | 17.3 ms median, 17.6 max per frame; busy 67 % | simplest stage: frames in, frames out | the models must be retrained at 50; a 32-sample window sees 0.64 symbol, too short for memory (PA memory, filters) spanning symbols, and a window long enough (~200 samples) costs 6 to 40 times more per sample (the projections grow with the window, the attention with its square), and 6 times more samples per symbol on top, beyond the GPU; not the plan's rate |
| **2b** | the model at **8**: the stage resamples 50 → 8, predistorts, resamples 8 → 50 | 8.9 ms median per frame (the GPU at 306 MHz under this light load), plus the resampling | the plan's rate and the step 1 models; a window of 4 symbols; a third of the GPU's time; the predistorted band (±1.6 GHz) inside the RAU's 3 GHz band-pass | two resampling filters (50/8 = 25/4, polyphase), in Python and in the twin, checked like the model; they add their own look-ahead (Decision 3) and a small error, to measure |

**Recommendation: 2b**, the plan's rate. **Decided: 2b.** The resampler must be the same in the reference (which
MATLAB also calls) and in the twin, so it is written once in `dpd/`, exported and checked against
golden files like the model. The training data from the Pi (BASEBAND and the PA tap, at 50) are
resampled to 8 the same way.

## Decision 3: frame boundaries

**Facts.** The window of output sample `n` spans `n − 16 … n + 15`: near the ends of a frame it
reaches into the previous and the next frame. The contract is one frame out per frame in, in order,
with `first_sample` saying which samples a frame holds; a stage may delay its output, as `fiber`
does (its output frames carry its own running sample count, `FiberStage::process`). The source adds
a tail to each calibration block to bring held-back samples out (`fiber`'s holdback + guard + 2
symbols). With 2b the resampling filters add their own reach into the neighbouring frames.

**Measured** (the PyTorch reference, the step 1 test drive at 8 samples per symbol cut into
1024-sample frames, the same 128 symbols as a BASEBAND frame):

| | Nominal PA: NMSE, EVM symbols | PA with memory: NMSE, EVM symbols |
|---|---|---|
| the drive whole (the reference) | -48.45 dB, 0.205 % | -16.19 dB, 7.958 % |
| each frame alone, zeros at both ends | -48.42 dB, 0.206 % | -16.10 dB, 7.949 % |
| the previous frame's last 16 samples kept, zeros ahead | -48.43 dB, 0.206 % | -16.19 dB, 7.970 % |

At the model, the frame edges cost at most 0.1 dB: these models lean on the centre sample.

| | Option | For | Against |
|---|---|---|---|
| **3a** | each frame **alone**, zeros beyond its ends (as the golden files and `predistort` do) | no state, no delay: output frame = input frame's samples, the latency is the frame's processing only; the calibration block and `first_sample` unchanged down the chain | the edges of every frame predistorted with a truncated window: measured negligible at the model; to measure again with the resampler (2b), whose filters ring at a cut |
| 3b | a **continuous** stream with a delay: keep the previous frame's tail, wait for the next frame's head; output frame k holds the samples shifted by the look-ahead (`first_sample` says so, as `fiber`) | exact: equal to the reference on the uncut drive | up to one frame (25.6 ms) more latency; state across frames; the shift travels down the chain: `co_optics` takes the calibration block as the first `n_block` samples of the `CALIB` frames, which would then start a few samples early, and `ue_dsp` must absorb the delay in its timing (to verify on the Pi); the last held samples have no frame to leave in at the END |
| 3c | the previous frame's tail kept, zeros ahead | no delay; the past exact | state across frames, for no measured gain over 3a |

**Recommendation: 3a**, the simplest. **Decided: 3a**, at a measured cost of at most 0.1 dB with these models;
to be measured again once the resampler (2b) is in, and with the models trained on the Pi's recordings.
3b stays the exact fallback if the edges cost more then.

## What follows from the decisions

1a, 2b and 3a give a stage that, for each BASEBAND frame: resamples it to 8 samples per symbol,
predistorts it on the GPU, resamples it back to 50, and sends it on, with the frame's own samples
and header; holds the calibration block once per calibration to compute κ from it, undistorted,
and writes κ into its `pipeline` line; forwards CONFIG, REF and END. The one change on the Pi:
`co_optics` reads κ from that line when it is there.
