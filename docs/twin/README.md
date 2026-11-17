# The C++/CUDA twin

`cpp/`: the reference DPD again, in C++, for inference on the Jetson — the CPU model now, the CUDA
kernels next (step 2), the pipeline's `dpd` stage around them (step 3). It reads the reference's
exported weights and is checked against its golden files ([FORMAT](../dpd/FORMAT.md)); what it
computes is [METHODOLOGY](../dpd/METHODOLOGY.md), operation by operation.

## Layout

| File | Content |
|---|---|
| `include/dpd/tensor_file.h`, `src/tensor_file.cpp` | the DPDT reader: rejects a wrong magic or version, truncation, trailing bytes |
| `include/dpd/weights.h`, `src/weights.cpp` | `Config`, `Weights`, `load_weights`: every tensor of the model, each shape checked against the config, no tensor left over |
| `include/dpd/cpu_model.h`, `src/cpu_model.cpp` | the forward pass in plain C++, float32: `forward_window` (one window, with an optional trace of the intermediates), `predistort` (a sequence, zero-padded like the reference), `window_at` |
| `tools/golden_check.cpp` | the model against a golden file: the whole frame and every probed intermediate |
| `tools/gpu_check.cu` | the image's GPU self-test |
| `CMakeLists.txt`, `Makefile` | the library `dpd_core`, the tools, the ctest checks |

## The CPU model

A direct transcription of METHODOLOGY section 2, one window at a time: the input scale, the
embedding and position, each pre-norm layer (layer norm, Q/K/V, per head the scaled scores, the
softmax with its maximum subtracted, the weighted sum; the output projection and its residual;
layer norm, feed-forward with the exact GELU, residual), then the centre token's norm, head, skip
and scale. float32 throughout, accumulation in float32, no SIMD or threads: it is the twin's own
reference, which the CUDA kernels will be checked against, and it is kept simple to read.

## The check against the reference

`golden_check <weights> <golden>` runs the golden frame (6400 samples) and compares the output and
each probe's intermediates; an error is the largest absolute difference over the RMS of the
reference's values. `make -C cpp test` runs it for both exported models (ctest).

| Model | Output | Embedding | Layer 0 | Layer 1 | Final norm | Before scale |
|---|---|---|---|---|---|---|
| `transformer_nominal` | 1.0e-6 | 2.0e-7 | 9.2e-7 | 8.6e-7 | 4.9e-7 | 2.9e-7 |
| `transformer_nominal_memory` | 9.2e-7 | 2.1e-7 | 8.6e-7 | 1.4e-6 | 7.3e-7 | 4.1e-7 |

(worst over the 8 probes; tolerance 1e-5). The errors are float32 rounding, from a different order
of summation than PyTorch's: about 8 units in the last place of float32 (1.2e-7), and they do not grow
through the layers. The same numbers come out of the image (`arof-dpd:local`, with
`results/models/` mounted at `/models`). A mismatched model and golden file, a truncated file and a
file without weights are each rejected.

Speed, for the record only: 1 200 samples/s on one Cortex-A78AE core (a 6400-sample frame in
5.3 s), the whole window encoded for each sample. The CUDA kernels are the fast path.
