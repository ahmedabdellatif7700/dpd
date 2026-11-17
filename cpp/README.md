# cpp

The C++/CUDA twin of the reference DPD: the transformer as hand-written CUDA kernels (inference
only, the reference's weights) and the `dpd` stage around it. Today: the build and the image's GPU
self-test (`src/gpu_check.cu`). `make`, `make run`, `make debug`, `make clean` on the host;
`make -C ../docker check` in the image.

Docs: [docs/testbed/jetson/](../docs/testbed/jetson/README.md), what it must compute:
[docs/dpd/METHODOLOGY.md](../docs/dpd/METHODOLOGY.md), how to run: [docs/howto/jetson.md](../docs/howto/jetson.md).
