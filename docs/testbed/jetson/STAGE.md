# The `dpd` stage (step 3): the three decisions

Status: **decided by the user on 2026-11-20: 1a, 2b and 3b.** 3a was chosen first; the
resampler's measurement (below) showed that cutting each frame costs 12.5 dB once the resampling
filters are in, and the user changed it to 3b. The change
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

**Recommendation: 3a**, the simplest, at a measured cost of at most 0.1 dB with these models;
to be measured again once the resampler (2b) is in.

**Measured again with the resampler** (`checks/resample_check.py`, `results/step3_resampler.txt`;
the drive at 50 samples per symbol, 50 → 8, the model, 8 → 50, the test PA at 20 GS/s, nominal PA):

| | NMSE | EVM symbols |
|---|---|---|
| reference at 8 samples per symbol (step 1) | -48.45 dB | 0.205 % |
| through the resampler, the drive whole | -47.79 dB | 0.205 % |
| 3a: each 6400-sample frame alone | **-35.25 dB** | 0.544 % |
| 3c: 300 samples of the previous frame kept, zeros ahead | -36.18 dB | 0.409 % |
| 3b: 300 samples kept behind, 300 waited for ahead | **-47.79 dB** | **0.205 %** |

The resampling filters reach about 63 samples (at 50 per symbol) into each neighbouring frame and
ring at a cut; the model's window adds 94 (15 at 8). Only the look-ahead recovers the drive whole.

**Decided: 3b**, as a holdback like `fiber`'s:

- the stage keeps the last D samples of each BASEBAND frame (D = 250, 5 symbols, above the
  look-ahead of the two filters and the window, ~220) and the samples behind them as context;
  each output frame carries the samples it can complete, `first_sample` saying which: the signal
  is not delayed, only packed differently. The frame's processing is unchanged; its last D samples
  wait for the next frame (25.6 ms at most);
- at `CALIB_END` the stage flushes what it holds (it holds the whole calibration group anyway for
  κ, decision 1a), so the group leaves complete and the source's tail needs no change. The group
  then starts D samples before the block: `ue_frontend` and `ue_dsp` locate the block by
  `first_sample` already (`link_stages.cpp`, lines 447-451 and 545-547); `co_optics` takes the
  first `n_block` samples of the group (line 228), so the 1a patch also makes it locate the block
  by `first_sample`;
- at END the last D samples of the stream have no frame to leave in and are dropped, as `fiber`'s
  holdback is.

## What follows from the decisions

1a, 2b and 3b give a stage that, for each BASEBAND frame: appends it to what it holds, resamples to
8 samples per symbol, predistorts on the GPU and resamples back to 50 the samples it can complete,
and sends those on, keeping the last 250; holds the calibration group once per calibration, computes
κ from its undistorted block, writes κ into its `pipeline` line, and flushes the group whole; and
forwards CONFIG, REF and END. The change on the Pi, one patch to `co_optics`: use the κ of that line
when it is there, and locate the block by `first_sample`.

## The stage as built

`cpp/bin/dpd_stage` (`cpp/src/dpd_stage.cpp`, on the copied stage library `cpp/arof/`), the command
of the `dpd_bb` service (`docker/compose.yaml`). Its section of the settings file, carried to it in
the CONFIG frame as the Pi's source carries `"taps"` and `"log"`:

```json
"dpd": {"bb": {"mode": "dpd", "model": "transformer_nominal"}}
```

| | `"mode": "bypass"` (the default) | `"mode": "dpd"` |
|---|---|---|
| BASEBAND | forwarded untouched, byte for byte | predistorted: `s → DPD(κ·s)/κ`, DPD = 50 → 8, the model, 8 → 50, through the 250-sample holdback (`FrameStream`, `dpd/stream.py`'s counterpart) |
| calibration group | passes | held; κ from its block (co_optics's formula, on the undistorted `s`); leaves whole, κ in the stage's `pipeline` line (`"calibration": {"kappa": …}`) |
| CONFIG | the stage's line `{"stage": "dpd_bb", "in"/"out": "BASEBAND", "mode": "bypass"}` | also `model`, `device`, `dpd_sps` 8, `hold_samples` 250, and the calibration |
| REF, END | forwarded | forwarded |
| dtype | as it comes | complex128 or complex64 in, the same out; float32 inside |

Environment: `AROF_STAGE`, `AROF_IN`, `AROF_OUT`, `AROF_TAP`, `AROF_LOG_DIR`,
`AROF_CONNECT_TIMEOUT_S` as every stage (the library's); `AROF_MODELS_DIR` (default `/models`: the
weights `<model>.dpdt` and `resampler.dpdt`) and `AROF_DPD_DEVICE` (`cuda`, default, or `cpu`). It
stops with a message and exit code 1 on: a stage before it that does not produce BASEBAND, a dtype it
cannot take, a non-contiguous drive, a calibration block of another size than the CONFIG frame's,
settings not at 50 samples per symbol, a missing model.

**Tested without the Pi** (`tests/test_dpd_stage.py`): a feeder sends what the Pi's source sends (a
CONFIG frame, the calibration block of 552 symbols in 128-symbol frames with their REF frames, a
30-symbol tail, 3 data frames, END), a sink records what comes out, the stage runs between them over
TCP, on the host or in the image (`DPD_STAGE_DOCKER=1`):

| Test | Result |
|---|---|
| bypass | every frame byte for byte as it came; the CONFIG frame gains the stage's line, nothing else |
| dpd, κ | in the stage's line, bit for bit co_optics's formula on the undistorted block (1.2822809823405896) |
| dpd, BASEBAND | equal to the reference applied offline (PyTorch model, resampler, `dpd/stream.py`): within 9.9e-7 on CUDA, 1.2e-6 on the CPU |
| dpd, the frames | one out per frame in (20), in order, seq and flags kept; REF and END unchanged; the drive contiguous by `first_sample`; the calibration group leaves whole (its last frame: the 30-symbol tail and the 250 held samples) |
| a wrong pipeline | exit code 1, `dpd_bb consumes BASEBAND, but the stage before it, source, produces OPTICAL` |
| the timing log | `dpd_bb.csv`, one line per frame, the library's columns |
