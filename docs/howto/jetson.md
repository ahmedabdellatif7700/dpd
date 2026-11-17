# The Jetson Orin Nano

`ssh orinnano8s`, then from `~/dpd`. Long jobs in a detached tmux session (the SSH connection drops):
`tmux new -s <name> '<command>'`, `tmux attach -t <name>`.

## The reference DPD, in the conda env

```bash
git pull
conda activate machine-learning        # ~/miniconda3; created from environment.yml
python -m pytest -q                     # the unit tests, ~80 s (CPU, and CUDA where present)
tmux new -s step1 'python scripts/step1_check.py --device cuda'
                                        # ILA on the test PA, ~6 min on the GPU:
                                        # results/step1.txt, step1.json, results/models/*.pt
```

`step1_check.py` takes `--device cpu|cuda`, `--passes`, `--epochs`, `--n-train`, `--n-eval`. A new
env, if needed: `conda env create -f environment.yml` (PyTorch from NVIDIA's JetPack 7 wheel index,
see [testbed/jetson](../testbed/jetson/README.md)).

## The stage's image

```bash
make -C docker help                     # the targets
make -C docker build                    # the image arof-dpd:local (compiles cpp/ for sm_87)
make -C docker check                    # build, then the GPU self-test in the container
make -C docker shell                    # a shell in the image, GPU attached
make -C cpp run                         # the same C++/CUDA build on the host (nvcc 13.2, no Docker)
```

`make -C docker check` should print the CUDA runtime and driver 13.2, the device `Orin` (8.7) and
`kernel on 6400 samples, max error 0: OK`. The container runs with the NVIDIA runtime
(`runtime: nvidia` in `docker/compose.yaml`); without it the GPU is not visible.

From the laptop, the same over SSH: `make -C docker jetson-build`, `make -C docker jetson-check`
(`JETSON=orinnano8s`, `JETSON_DIR=dpd` by default).
