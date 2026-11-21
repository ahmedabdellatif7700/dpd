"""The test drive and the measures at the PA's output, shared by the checks (checks/step1_check.py,
checks/twin_check.py), so that the PyTorch reference and the C++/CUDA twin are measured the same
way.

The drive: 16QAM RRC at N_DPD = 8 samples per symbol (3.2 GS/s), scaled to the data's RMS at the PA
input (0.371 a0). The link: the test PA (dpd/testing/pa.py) driven with the drive plus the pilot tone, its
output with the tone removed.
"""
import numpy as np

from . import metrics, pa, signals

FS = signals.RS * signals.N_DPD


def drive(n_sym, seed):
    """the drive x of n_sym symbols and the symbols d"""
    s, d = signals.baseband(n_sym, seed=seed)
    return s * pa.DATA_RMS_A0 / np.sqrt(np.mean(np.abs(s) ** 2)), d


def link(params):
    """the test chain with the PA's parameters (pa.pa_params), as a callable drive -> output"""
    def f(x):
        return pa.pa_chain(x, params, fs=FS)
    return f


def measures(link_fn, x, d, x_dpd=None):
    """NMSE and EVM at the PA's output against the drive x, and the PA input's peak-to-average ratio,
    with the predistorted drive x_dpd (None: no DPD)"""
    xd = x if x_dpd is None else x_dpd
    y = link_fn(xd)
    u = xd + pa.TONE_A0
    return {"nmse_db": metrics.nmse_db(y, x), "evm_waveform_pct": 100 * metrics.evm_waveform(y, x, FS),
            "evm_symbols_pct": 100 * metrics.evm_symbols(y, d),
            "papr_in_db": float(10 * np.log10(np.max(np.abs(u) ** 2) / np.mean(np.abs(u) ** 2)))}
