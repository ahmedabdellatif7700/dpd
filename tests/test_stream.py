"""The stage's stream (dpd/stream.py, decision 3b): frames in, completed samples out; equal to the
drive's whole transform away from the stream's start and from a flush."""
import numpy as np
import pytest

from dpd import resample as R
from dpd.models import DPD, TransformerDPD
from dpd.stream import HOLD, FrameStream


def model():
    import torch
    torch.manual_seed(2)
    m = TransformerDPD()
    with torch.no_grad():
        torch.nn.init.normal_(m.head.weight, std=0.1)   # not the identity: the window counts
        m.scale.fill_(0.37)
    return DPD(m, "cpu")


def drive(n, seed=3):
    r = np.random.default_rng(seed)
    return 0.3 * (r.standard_normal(n) + 1j * r.standard_normal(n))


def test_stream_equals_the_whole_transform():
    dpd, h = model(), R.lowpass()
    transform = lambda v: R.to_pipeline(dpd.predistort(R.to_dpd(v, h)), h)  # noqa: E731
    x = drive(5 * 6400)
    whole = transform(x)[:x.size]
    st, got, firsts = FrameStream(transform), [], []
    for k in range(5):
        first, y = st.push(x[k * 6400:(k + 1) * 6400])
        firsts.append(first)
        got.append(y)
    y = np.concatenate(got)
    assert firsts == [0, 6400 - HOLD, 2 * 6400 - HOLD, 3 * 6400 - HOLD, 4 * 6400 - HOLD]
    assert y.size == x.size - HOLD                      # the last HOLD samples wait for a next frame
    err = np.max(np.abs(y - whole[:y.size])) / np.sqrt(np.mean(np.abs(whole) ** 2))
    assert err < 1e-6                                   # equal, float32 rounding of the model aside


def test_flush_and_continue():
    transform = lambda v: R.to_pipeline(R.to_dpd(v))  # noqa: E731  (the resamplers only)
    x = drive(4 * 6400, 4)
    st = FrameStream(transform)
    a = st.push(x[:6400])
    b = st.push(x[6400:12800])
    c = st.flush()                                      # a calibration group leaves whole
    d = st.push(x[12800:19200])                         # the stream goes on, its past kept
    assert (a[0], b[0], c[0], d[0]) == (0, 6400 - HOLD, 12800 - HOLD, 12800)
    assert c[1].size == HOLD and d[1].size == 6400 - HOLD
    whole = transform(x)[:x.size]
    rel = lambda u, v: np.max(np.abs(u - v)) / np.sqrt(np.mean(np.abs(v) ** 2))  # noqa: E731
    assert rel(b[1], whole[b[0]:b[0] + b[1].size]) < 1e-12
    # after the flush, the past is kept: only the flushed samples near the cut differ
    assert rel(d[1][300:], whole[d[0] + 300:d[0] + d[1].size]) < 1e-12


def test_frames_must_be_on_the_grid():
    st = FrameStream(lambda v: v)
    with pytest.raises(ValueError):
        st.push(np.zeros(6401))
