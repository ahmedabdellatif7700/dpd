"""The measures: a gain and a constant are not errors; a known error is measured as such."""
import numpy as np

from dpd.testing import metrics, signals

FS = signals.RS * signals.N_DPD


def test_gain_and_constant_are_not_errors():
    s, d = signals.baseband(1024, seed=3)
    y = (0.8 - 0.3j) * s + (0.1 + 0.05j)
    assert metrics.nmse_db(y, s) < -250
    assert metrics.evm_waveform(y, s, FS) < 1e-12
    assert metrics.evm_symbols(y, d) < 5e-4          # the RRC pair's own floor (span 16): 0.03 %


def test_known_error():
    s, d = signals.baseband(4096, seed=4)
    e = np.random.default_rng(5).standard_normal(s.size) * 0.1 * np.sqrt(np.mean(np.abs(s) ** 2) / 2)
    y = s + e + 1j * e[::-1]                          # white, 20 dB below the signal
    assert abs(metrics.nmse_db(y, s) + 20) < 0.3
