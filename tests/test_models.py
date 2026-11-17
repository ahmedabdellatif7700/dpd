"""The transformer DPD: shapes, the numpy boundary, the window alignment, CPU against CUDA."""
import numpy as np
import pytest
import torch

from dpd.models import DPD, TransformerDPD, windows

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


def sig(n=1000, seed=1):
    r = np.random.default_rng(seed)
    return r.standard_normal(n) + 1j * r.standard_normal(n)


def test_untrained_is_identity():
    x = sig()
    d = DPD(TransformerDPD(), "cpu")
    y = d.predistort(x)
    assert y.dtype == np.complex128 and y.shape == x.shape
    assert np.max(np.abs(y - x)) < 1e-5            # float32 inside


def test_dtypes_and_bridge_form():
    x = sig(100)
    d = DPD(TransformerDPD(), "cpu")
    assert d.predistort(x.astype(np.complex64)).dtype == np.complex64
    re, im = d.predistort_reim(x.real, x.imag)
    assert re.dtype == np.float64 and im.dtype == np.float64
    assert np.allclose(re + 1j * im, d.predistort(x))
    assert d.predistort(np.zeros(0, np.complex128)).size == 0


def test_window_alignment():
    # output n sees inputs n-16 .. n+15 (window 32, centre 16), zeros beyond the ends
    xr = torch.arange(1, 6, dtype=torch.float32)[:, None].repeat(1, 2)
    w = windows(xr, 4, 2)
    assert w.shape == (5, 4, 2)
    assert w[0, :, 0].tolist() == [0, 0, 1, 2]
    assert w[4, :, 0].tolist() == [3, 4, 5, 0]


def test_causality_span():
    # a change at sample k moves outputs k-15 .. k+16 only
    m = TransformerDPD()
    torch.nn.init.normal_(m.head.weight, std=0.1)
    d = DPD(m, "cpu")
    x = sig(200)
    x2 = x.copy()
    x2[100] += 1
    diff = np.nonzero(np.abs(d.predistort(x2) - d.predistort(x)) > 1e-7)[0]
    assert diff.min() >= 100 - 15 and diff.max() <= 100 + 16


def test_save_load(tmp_path):
    m = TransformerDPD()
    torch.nn.init.normal_(m.head.weight, std=0.1)
    m.scale.fill_(0.37)
    d = DPD(m, "cpu")
    d.save(tmp_path / "m.pt")
    e = DPD.load(tmp_path / "m.pt", "cpu")
    x = sig(300)
    assert np.array_equal(d.predistort(x), e.predistort(x))


def test_size():
    d = DPD(TransformerDPD(), "cpu")
    ops = d.model.ops_per_sample()
    assert 15000 < d.n_params() < 30000
    assert ops["flops"] == 2 * ops["macs"]


@cuda
def test_cpu_vs_cuda():
    m = TransformerDPD()
    torch.nn.init.normal_(m.head.weight, std=0.1)
    m.scale.fill_(0.37)
    x = 0.4 * sig(6400)
    a = DPD(m, "cpu").predistort(x)
    b = DPD(TransformerDPD(), "cuda")
    b.model.load_state_dict(m.state_dict())
    b = b.predistort(x)
    assert np.max(np.abs(a - b)) / np.sqrt(np.mean(np.abs(a) ** 2)) < 1e-4
