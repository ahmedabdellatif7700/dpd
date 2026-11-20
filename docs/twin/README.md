# The C++/CUDA twin

`cpp/`: the reference DPD again, in C++, for inference on the Jetson: a CPU model and a CUDA model
with the same interface, and, in step 3, the pipeline's `dpd` stage around the CUDA one. Both read
the reference's exported weights and are checked against its golden files
([FORMAT](../dpd/FORMAT.md)); what they compute is [METHODOLOGY](../dpd/METHODOLOGY.md), operation
by operation.

The weights files are named after the test PA they were trained on, not after a different model:
`transformer_nominal` is the transformer trained by ILA on the nominal link's PA,
`transformer_nominal_memory` the same transformer trained on that PA with its memory option on.

## Layout

| File | Content |
|---|---|
| `include/dpd/tensor_file.h`, `src/tensor_file.cpp` | the DPDT reader: rejects a wrong magic or version, truncation, trailing bytes |
| `include/dpd/weights.h`, `src/weights.cpp` | `Config`, `Weights`, `load_weights`: every tensor of the model, each shape checked against the config, no tensor left over |
| `include/dpd/cpu_model.h`, `src/cpu_model.cpp` | `CpuModel`, the model in plain C++ on the CPU; underneath it `forward_window` (one window, with an optional trace) and `window_at` |
| `include/dpd/gpu_model.h`, `src/gpu_model.cu` | `GpuModel`, the model as one fused CUDA kernel |
| `tools/model_check.h` | the check both tools share: a model against a golden file |
| `tools/golden_check.cpp` | `CpuModel` against a golden file, and its speed on one core |
| `tools/cuda_check.cpp` | `GpuModel` against a golden file and against `CpuModel`, and the kernel's speed |
| `tools/gpu_check.cu` | the image's GPU self-test |
| `CMakeLists.txt`, `Makefile` | the libraries `dpd_core` (CPU) and `dpd_gpu` (CUDA), the tools, the ctest checks |

## The two models

`CpuModel` and `GpuModel` have the same interface, so the checks, and later the stage, use them the
same way:

```cpp
const dpd::Weights w = dpd::load_weights("results/models/transformer_nominal.dpdt");
dpd::CpuModel cpu(w);                      dpd::GpuModel gpu(w);
cpu.predistort(x, y, n);                   gpu.predistort(x, y, n);    // host in, host out: x[n][2] -> y[n][2]
cpu.trace(x, n, probes);                   gpu.trace(x, n, probes);    // intermediates, golden-file layout
                                           gpu.forward(d_x, d_y, n, stream);   // GPU buffers, asynchronous
```

`predistort` treats the frame alone (zeros beyond its ends), as the reference does. `trace` returns,
per probed sample, the embedding, each layer's output, the final norm and the output before the
scale, and rejects a probe outside the frame. `forward` exists only on the GPU side: it takes
buffers already in GPU memory and returns at once; it is what the stage will call.

### CpuModel

A direct transcription of METHODOLOGY section 2, one window at a time: the input divided by the
scale, the embedding and position, each pre-norm layer (layer norm, Q/K/V, per head the scaled
scores, the softmax with its maximum subtracted, the weighted sum; the output projection and its
residual; layer norm, feed-forward with the exact GELU, residual), then the centre token's norm,
head, skip and scale. float32 throughout, no SIMD or threads: it is the twin's own reference, kept
simple to read, and the CUDA kernel is checked against it.

### GpuModel: one fused kernel per frame

One kernel launch computes a whole frame. Each output sample gets one thread block of 128 threads
(4 warps), which loads the sample's window and runs the whole model on it in shared memory
(23.8 kB per block); only the sample's I and Q go back to GPU memory.

- **A warp per token.** The model width is 32, a warp's width: a warp works on one token at a time,
  lane i holding feature i. Layer norms, the softmax and the head reduce across the warp with
  shuffles; the 4 warps share the 32 tokens.
- **Per layer, two phases**, separated by a barrier: (1) layer norm and Q/K/V of every token; (2)
  per token, the attention over all tokens' keys and values, the output projection and residual,
  then the feed-forward block and residual. Phase 2 needs every token's K and V, hence the barrier;
  everything else in it stays within the warp.
- **Weights.** All 18 338 parameters (73 kB) in one GPU buffer, shared by every block. The matrices
  are stored transposed, [in][out], so that the 32 lanes read 32 neighbouring words.
- **Trace.** For the probed samples only, the kernel also writes the intermediates, in the golden
  file's layout; in normal runs the trace pointer is null and costs nothing.
- **Errors.** Every CUDA call is checked and throws with the call's name. `predistort` and `trace`
  synchronize after the kernel, so a fault inside it is reported as `dpd_forward`, not at a later
  copy. GPU memory is owned by `DeviceBuffer` (a `unique_ptr` that calls `cudaFree`), so nothing
  leaks or is freed twice, also when an exception is thrown.
- **One configuration.** The kernel is written for this model (window 32, width 32, 4 heads,
  2 layers, FF 64); `GpuModel` refuses any other configuration with an error naming it.

This first version favours clarity over speed: every block encodes its whole window, although
neighbouring windows share 31 of their 32 samples, and the last layer is computed for all tokens
where only the centre one is read. Those are the optimizations that follow.

## The checks against the reference

`make -C cpp test` runs `golden_check` and `cuda_check` on both exported models (ctest). Errors are
the largest absolute difference over the RMS of the reference's values, worst over the 8 probes;
tolerance 1e-5.

| Model | Implementation | Output | Embedding | Layer 0 | Layer 1 | Final norm | Before scale |
|---|---|---|---|---|---|---|---|
| `transformer_nominal` | `CpuModel` | 1.0e-6 | 0 | 9.2e-7 | 8.6e-7 | 4.9e-7 | 2.9e-7 |
| | `GpuModel` | 8.0e-7 | 0 | 6.9e-7 | 6.7e-7 | 4.9e-7 | 2.9e-7 |
| `transformer_nominal_memory` | `CpuModel` | 8.4e-7 | 0 | 8.4e-7 | 1.9e-6 | 7.3e-7 | 4.1e-7 |
| | `GpuModel` | 9.2e-7 | 0 | 8.1e-7 | 1.7e-6 | 8.0e-7 | 5.9e-7 |

`GpuModel` against `CpuModel`: 6.0e-7 and 7.6e-7. The errors are float32 rounding, from different
orders of summation: a few units in the last place of float32 (1.2e-7), and they do not grow
through the layers. The same numbers come out of the image (`arof-dpd:local`, with
`results/models/` mounted at `/models`). A mismatched model and golden file, a truncated file and a
file without weights are each rejected.

## Speed

One 6400-sample frame, the kernel alone with the frame already on the GPU (CUDA events, 200 runs):

| Implementation | Per frame | Samples/s |
|---|---|---|
| `CpuModel`, one Cortex-A78AE core | 5.3 s | 1 200 |
| `GpuModel`, first version | 45.5 ms | 141 000 |

The pipeline streams 5 000 symbols/s (the testbed of paper 1), so the DPD must keep up with
40 000 samples/s at 8 samples per symbol, or 250 000 at the BASEBAND frame's 50: the first version
meets the first, not the second. The timing with the copies and per-frame latency is part 5.
