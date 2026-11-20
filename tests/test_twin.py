"""The C++/CUDA twin (cpp/bin/dpd_apply) against the PyTorch reference on a short drive, on the GPU
and the CPU. Skipped when the C++ tools are not built (e.g. on the PC); the full check through the
test PA is scripts/twin_check.py."""
import os
import subprocess

import numpy as np
import pytest

from dpd.evaluate import drive
from dpd.export import read_tensors, write_tensors
from dpd.models import DPD, to_iq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "cpp", "bin", "dpd_apply")
MODEL = os.path.join(ROOT, "results", "models", "transformer_nominal")

pytestmark = pytest.mark.skipif(not os.path.exists(BIN), reason="cpp/bin/dpd_apply not built (make -C cpp)")


def run_twin(device, x, tmp_path):
    fin, fout = tmp_path / "in.dpdt", tmp_path / "out.dpdt"
    write_tensors(fin, [("input", to_iq(x))])
    subprocess.run([BIN, "--device", device, MODEL + ".dpdt", str(fin), str(fout)], check=True, capture_output=True)
    y = read_tensors(fout)["output"]
    return y[:, 0].astype(np.float64) + 1j * y[:, 1].astype(np.float64)


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_twin_equals_reference(device, tmp_path):
    x, _ = drive(256, 2017)
    ref = DPD.load(MODEL + ".pt", "cpu").predistort(x)
    y = run_twin(device, x, tmp_path)
    assert y.shape == ref.shape
    assert np.max(np.abs(y - ref)) / np.sqrt(np.mean(np.abs(ref) ** 2)) < 1e-5


def test_twin_rejects_bad_input(tmp_path):
    fin = tmp_path / "in.dpdt"
    write_tensors(fin, [("input", np.zeros((4, 3)))])
    r = subprocess.run([BIN, MODEL + ".dpdt", str(fin), str(tmp_path / "out.dpdt")], capture_output=True, text=True)
    assert r.returncode == 1 and "'input' must be [n][2]" in r.stderr
