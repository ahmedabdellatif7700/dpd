# The PC

The reference DPD on the CPU, called from MATLAB, which runs the link
(`matlab/link/transmit_receive_odsb.m` in on-chip-training). Not yet run on the PC: this page says
how it is meant to work, and is checked when the DPD is first used from MATLAB.

## The env

The PC's conda env `machine-learning` (on-chip-training `matlab/dpd/README.md`, step 1: Python
3.10, numpy 2.2.6, torch 2.13 CPU) has what `dpd/` needs: numpy and torch. The package uses nothing
newer than Python 3.10 and no CUDA unless `device="cuda"` is asked for; `pytest` for the tests.

```bash
conda activate machine-learning
cd ~/dpd && python -m pytest -q          # the CUDA test is skipped without a GPU
```

## From MATLAB

MATLAB loads Python out of process on that env, as `matlab/dpd/dpd_bridge.m` does, and puts the
folder that holds the `dpd` package on Python's path (this repository today; `matlab/dpd/py/` after
the merge). Complex signals cross as two float64 vectors:

```matlab
pyenv('Version', '.../envs/machine-learning/bin/python', 'ExecutionMode', 'OutOfProcess');
insert(py.sys.path, int32(0), '<folder holding dpd/>');
m = py.importlib.import_module('dpd.models');
d = m.DPD.load('<repo>/results/models/transformer_nominal.pt', 'cpu');
out = d.predistort_reim(real(drive), imag(drive));        % a Python tuple (re, im)
drive_dpd = double(out{1}) + 1j*double(out{2});
```

`predistort_reim` returns float64 vectors of the input's length, aligned with it (no delay). The
model works at the rate it was trained at (8 samples per symbol for the step 1 models): the link's
drive at 50 samples per symbol is resampled around it (to be settled with the DPD's rate).
