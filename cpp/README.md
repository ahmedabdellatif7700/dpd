# cpp

The C++/CUDA twin of the reference DPD (inference only, the reference's weights) and, in step 3,
the `dpd` stage around it. Today: the DPDT reader, the weights, the model in plain C++ on the CPU
(`src/`, `include/dpd/`), its check against the reference's golden files and the image's GPU
self-test (`tools/`). `make`, `make test`, `make run`, `make debug`, `make clean` on the host;
`make -C ../docker check` in the image.

Docs: [docs/twin/](../docs/twin/README.md), what it computes:
[docs/dpd/METHODOLOGY.md](../docs/dpd/METHODOLOGY.md), the files it reads:
[docs/dpd/FORMAT.md](../docs/dpd/FORMAT.md), how to run: [docs/howto/jetson.md](../docs/howto/jetson.md).
