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
| `tools/dpd_bench.cpp` | the latency and throughput per frame on the GPU, as the stage will run it (`scripts/step2_timing.sh`) |
| `tools/dpd_apply.cpp` | predistorts a drive file on the GPU or the CPU: the bridge to the Python checks |
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

- **Optimizations**, each kept only if it was measured faster and gave bit-identical outputs:
  - *the last layer for the centre token only*: past its Q/K/V, the last layer computes only token
    16, the one the output reads (in trace mode it still computes all, since the golden file
    compares them);
  - *8 tokens per warp at once*: each warp owns tokens w, w+4, …, w+28 and runs every matrix
    product on the 8 together, one accumulator per token in registers, so each weight is loaded
    once for 8 tokens;
  - *no bank conflict in the attention*: the rows of `qkv` are padded to 97 floats, so the 32 lanes
    that read one key each fall in 32 different shared-memory banks (with 96, all hit one bank);
  - *inputs read four at a time*: the matrix products read their inputs with one `float4` load
    instead of four, the summation order unchanged.

  Resources: 96 registers per thread, 31.1 kB of shared memory per block, no spills.

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

## Through the test PA

`scripts/twin_check.py` closes the loop: the drive of step 1 (validation seed 4242 and test seed
2017, 2048 symbols, 16 505 samples each) is predistorted by the reference (PyTorch) and by the twin
(`dpd_apply`, on the GPU and on the CPU), and each goes through the test PA the model was trained
for. Numpy writes the drive to a DPDT file, `dpd_apply` predistorts it, numpy reads the output and
measures it (`dpd/evaluate.py`, the same code as step 1), so there are no bindings to debug.

| Model | Sequence | NMSE, dB: none → PyTorch → twin | EVM symbols, %: none → PyTorch → twin |
|---|---|---|---|
| `transformer_nominal` | validation | -14.843 → -48.831 → -48.831 | 17.277 → 0.198 → 0.198 |
| | test | -14.860 → -48.446 → -48.446 | 17.294 → 0.205 → 0.205 |
| `transformer_nominal_memory` | validation | -10.606 → -16.117 → -16.117 | 27.386 → 7.868 → 7.868 |
| | test | -10.644 → -16.192 → -16.192 | 27.356 → 7.958 → 7.958 |

The twin's outputs differ from the reference's by at most 8.8e-7 (over the RMS), and its NMSE and
EVM by at most 9e-7 (dB, percentage points), on the GPU and the CPU alike (`results/step2_twin.txt`;
tolerances 1e-5 and 0.01). The twin linearizes the PA exactly as the reference does.
`tests/test_twin.py` repeats the comparison on a short drive within `pytest`, and is skipped where
the C++ tools are not built.

## Speed

One 6400-sample frame, the kernel alone with the frame already on the GPU (CUDA events, 200 to
1 000 runs, repeats within 0.04 %), the Jetson in its default 25 W mode (GPU clock up to 918 MHz,
set by the driver; not pinned):

| Implementation | Per frame | Samples/s | Gain |
|---|---|---|---|
| `CpuModel`, one Cortex-A78AE core | 5.3 s | 1 200 | |
| `GpuModel`, first version | 45.5 ms | 141 000 | |
| + last layer for the centre token only | 27.0 ms | 237 000 | 1.68 |
| + 8 tokens per warp at once | 25.5 ms | 250 500 | 1.06 |
| + no bank conflict in the attention | 20.4 ms | 313 000 | 1.25 |
| + inputs read four at a time | **16.65 ms** | **384 000** | 1.23 |

2.73 times the first version, 320 times one CPU core. Tried and dropped: the softmax sums in a
register instead of a small array (no measurable gain). The profiler (Nsight Compute) needs access
to the GPU's performance counters, which this user does not have; the limits were found by
reasoning and measurement instead.

The pipeline streams 5 000 symbols/s (the testbed of paper 1), so the DPD must keep up with
40 000 samples/s at 8 samples per symbol, or 250 000 at the BASEBAND frame's 50: the kernel now
meets both, with 54 % to spare at 50.

### Per frame, as the stage will run it

`dpd_bench` times what the stage does with each frame: copy it to the GPU, run the kernel, copy the
result back, wait (one CUDA stream); 1 000 frames per run, wall time on the host
(`scripts/step2_timing.sh`, `results/step2_timing.txt`). Two frame sizes: the BASEBAND frame at 50
samples per symbol (6400) and the same 128 symbols at 8 (1024), should the stage resample. Back to
back, and paced at the pipeline's arrival rate: a 128-symbol frame every 25.6 ms at 5 000 symbols/s.

| Frame | Frames | Latency, median / p99 / max | Kernel alone, median | GPU busy |
|---|---|---|---|---|
| 6400 | back to back | 16.8 / 17.1 / 17.4 ms | 16.67 ms | |
| 6400 | **paced every 25.6 ms** | **17.3 / 17.5 / 17.6 ms** | 17.13 ms | 67 % |
| 1024 | back to back | 2.8 / 3.0 / 3.2 ms | 2.71 ms | |
| 1024 | **paced every 25.6 ms** | **8.9 / 9.1 / 9.2 ms** | 8.05 ms | 35 % |

- **Both rates keep up** with the pipeline: the largest latency, 17.6 ms at 50 samples per symbol,
  stays 8 ms inside the 25.6 ms between frames. Back to back, the GPU could take 380 000 samples/s
  (6400) or 366 000 (1024).
- **The copies cost 0.06 to 0.13 ms** per frame: the Jetson's CPU and GPU share their memory.
  Pinned and pageable host memory perform alike (pageable showed one outlier of 20 ms in 1 000
  frames; the stage uses pinned).
- **The GPU's clock follows its load.** The driver sets it (25 W mode, 306 to 918 MHz): at 6400
  samples a frame the GPU is busy 67 % of the time and runs at 918 MHz; at 1024 it is busy 35 % and
  drops to 306 MHz, which makes the kernel three times slower (8.05 ms against 2.71 ms; the clock
  read during the runs). Pinning the clock at 918 MHz would bring the 8-samples-per-symbol latency
  to about 2.8 ms; that is a system setting of the Jetson, left as it is.
- Not in these figures: the stage's own frame handling, and the 15 samples of look-ahead the window
  needs from the next frame (step 3).
