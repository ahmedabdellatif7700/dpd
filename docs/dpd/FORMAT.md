# The export format (DPDT)

How the reference hands the model to the C++/CUDA twin: the weights, and golden files that check
the twin layer by layer. Written by `dpd/export.py` (`scripts/export_model.py`), read by the twin.

## The file

A list of named float32 tensors, little-endian:

| Field | Type | Value |
|---|---|---|
| magic | 4 bytes | `DPDT` |
| version | u32 | 1 |
| count | u32 | number of tensors |
| *per tensor:* name length | u16 | |
| name | ASCII, no terminator | |
| ndim | u8 | |
| dims | u32 × ndim | |
| data | float32 × the product of dims | row-major (C order) |

Nothing follows the last tensor; a reader rejects a wrong magic, an unknown version and trailing
bytes. Integers are stored as float32 (exact below 2^24).

## The weights file

`results/models/<name>.dpdt` (74.5 kB for the transformer):

| Tensor | Shape | |
|---|---|---|
| `config` | [5] | window, d_model, n_heads, n_layers, d_ff: [32, 32, 4, 2, 64] |
| `scale` | [1] | the input scale `s` |
| `pos`, `embed.weight`, … `head.bias` | as in the model | the model's `state_dict`, in its order, with its names and shapes ([METHODOLOGY](METHODOLOGY.md) §2) |

A weights file loads back into the reference (`dpd.export.load_weights`) with outputs identical
bit for bit to the original model's.

## The golden file

`results/models/golden/<name>.dpdt` (202 kB): one BASEBAND-sized frame of 6400 samples through the
reference on the CPU, in float32. The frame is the step 1 test drive (seed 2017, 0.371 a0 RMS,
8 samples per symbol), taken alone: zeros beyond its two ends, as `predistort` does.

| Tensor | Shape | |
|---|---|---|
| `input` | [6400, 2] | the drive, I/Q |
| `output` | [6400, 2] | the reference's output, I/Q |
| `probe_index` | [8] | the probed output samples: 0, 1, 15, 16, 3200, 6384, 6385, 6399 |
| `probe.embed` | [8, 32, 32] | after the embedding and position (all 32 tokens) |
| `probe.layer0`, `probe.layer1` | [8, 32, 32] | after each encoder layer |
| `probe.final_norm` | [8, 32] | the centre token after the final layer norm |
| `probe.out` | [8, 2] | the output before the scale: `out × s` = `output` at the probe |

The probes cover both ends of the frame (windows that reach past it), the first and last samples
whose window lies inside it (16 and 6384), the next ones, and the middle. When the twin's output
differs, the first probe that differs shows the stage at fault.

## The resampler's files

`results/models/resampler.dpdt` (2 kB): `resampler.ratio` [2] = 4, 25 (50 → 8; 8 → 50 swaps them)
and `resampler.h` [503], the low-pass at 80 GS/s (`dpd/resample.py`). Its golden file,
`results/models/golden/resampler.dpdt`: `input` [6400, 2], a BASEBAND frame at 50 samples per
symbol; `down` [1024, 2], its 50 → 8; `up` [6400, 2], 8 → 50 of `down`; computed in float64 from the
float32 input.
