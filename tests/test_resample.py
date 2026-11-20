"""The resampler 50 <-> 8 samples per symbol (dpd/resample.py): the formula, the filter, the alignment,
the round trip. The full check through the test PA is scripts/resample_check.py."""
import numpy as np
from scipy.signal import freqz, upfirdn

from dpd import resample as R
from dpd import signals


def test_filter():
    h = R.lowpass()
    assert h.size == 503 and h.size % 2 == 1
    assert np.allclose(h, h[::-1])                      # symmetric: no phase distortion
    assert abs(h.sum() - 1) < 1e-12                     # unit gain at DC
    w, H = freqz(h, worN=2 ** 15, fs=R.FS_HIGH)
    assert np.all(np.abs(20 * np.log10(np.abs(H[w <= 1.2e9]))) < 0.01)   # passband within 0.01 dB
    assert np.all(20 * np.log10(np.abs(H[w >= 2.0e9])) < -79)          # stopband below -79 dB


def test_lengths():
    assert R.out_length(6400, R.UP, R.DOWN) == 1024    # a BASEBAND frame
    assert R.out_length(1024, R.DOWN, R.UP) == 6400
    for n in (1, 7, 25, 500, 6401):
        n8 = R.out_length(n, R.UP, R.DOWN)
        assert R.out_length(n8, R.DOWN, R.UP) >= n      # the round trip covers the input


def test_formula_against_upfirdn():
    # an independent implementation: scipy's upfirdn at the high rate, delay c removed
    h = R.lowpass()
    c = (h.size - 1) // 2
    r = np.random.default_rng(1)
    x = r.standard_normal(300) + 1j * r.standard_normal(300)
    for up, down in ((R.UP, R.DOWN), (R.DOWN, R.UP)):
        ref = upfirdn(h * up, x, up, down=1)[c::down][:R.out_length(x.size, up, down)]
        assert np.max(np.abs(R.resample(x, up, down, h) - ref)) < 1e-12


def test_alignment_with_the_waveform_made_at_8():
    # the same symbols made at 50 and at 8 samples per symbol: 50 -> 8 gives the second, no delay
    # (the RRC has unit energy, so the waveform at 50 is sqrt(8/50) of the one at 8)
    s50, _ = signals.baseband(256, sps=50, seed=5)
    s8, _ = signals.baseband(256, sps=8, seed=5)
    s8r = R.to_dpd(s50)
    assert s8r.size == s8.size
    mid = slice(200, s8.size - 200)
    e = np.sqrt(50 / 8) * s8r[mid] - s8[mid]
    assert 10 * np.log10(np.sum(np.abs(e) ** 2) / np.sum(np.abs(s8[mid]) ** 2)) < -60


def test_round_trip():
    s50, _ = signals.baseband(256, sps=50, seed=6)
    rt = R.to_pipeline(R.to_dpd(s50))[:s50.size]
    mid = slice(1000, s50.size - 1000)
    e = rt[mid] - s50[mid]
    assert 10 * np.log10(np.sum(np.abs(e) ** 2) / np.sum(np.abs(s50[mid]) ** 2)) < -65
