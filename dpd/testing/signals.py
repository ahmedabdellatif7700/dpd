"""Test signals: 16QAM through an RRC filter, as the link's transmitter makes them.

From on-chip-training, branch `dpd`: `matlab/link/default_tx.m` (16QAM, 0.4 GBd, RRC roll-off 0.75,
span 16 symbols, unit average power) and `matlab/link/transmit_receive_odsb.m` (`qammod`,
`upfirdn(d, rrc, sps)`). `rrc` ports MATLAB's `rcosdesign(beta, span, sps, 'sqrt')`. The labels'
Gray map is per axis and need not be MATLAB's: only the symbols matter for the DPD's checks.
"""
import numpy as np

RS = 0.4e9          # symbol rate
ROLLOFF = 0.75
SPAN = 16           # RRC span, symbols
N_DPD = 8           # samples per symbol at the DPD (matlab/dpd/README.md: 3.2 GS/s by default)

_LEVELS = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)   # 16QAM, unit average power
_GRAY = np.array([0, 1, 3, 2])                                # 2 bits -> level index


def rrc(beta=ROLLOFF, span=SPAN, sps=N_DPD):
    """rcosdesign(beta, span, sps, 'sqrt'): unit energy, span*sps + 1 taps"""
    delay = span * sps / 2
    t = (np.arange(span * sps + 1) - delay) / sps
    b = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti == 0:
            b[i] = -1 / (np.pi * sps) * (np.pi * (beta - 1) - 4 * beta)
        elif abs(abs(4 * beta * ti) - 1) < np.sqrt(np.finfo(float).eps):
            b[i] = 1 / (2 * np.pi * sps) * (np.pi * (beta + 1) * np.sin(np.pi * (beta + 1) / (4 * beta))
                                            - 4 * beta * np.sin(np.pi * (beta - 1) / (4 * beta))
                                            + np.pi * (beta - 1) * np.cos(np.pi * (beta - 1) / (4 * beta)))
        else:
            b[i] = (-4 * beta / sps * (np.cos((1 + beta) * np.pi * ti) + np.sin((1 - beta) * np.pi * ti) / (4 * beta * ti))
                    / (np.pi * ((4 * beta * ti) ** 2 - 1)))
    return b / np.sqrt(np.sum(b ** 2))


def qam16(n, rng):
    """n 16QAM symbols (unit average power) and their labels 0..15"""
    idx = rng.integers(0, 16, n)
    d = _LEVELS[_GRAY[idx >> 2]] + 1j * _LEVELS[_GRAY[idx & 3]]
    return d, idx


def baseband(n_sym, sps=N_DPD, seed=0, span=SPAN):
    """the RRC waveform s = upfirdn(d, rrc, sps) of n_sym symbols, and the symbols d.
    Symbol k sits at sample k*sps + span*sps/2 of s."""
    d, _ = qam16(n_sym, np.random.default_rng(seed))
    up = np.zeros(n_sym * sps, dtype=np.complex128)
    up[::sps] = d
    s = np.convolve(up, rrc(sps=sps, span=span))[: (n_sym - 1) * sps + 1 + span * sps]
    return s, d


def matched(y, n_sym, sps=N_DPD, span=SPAN):
    """matched filter and symbol-rate sampling of a waveform made by baseband(): n_sym samples"""
    z = np.convolve(y, rrc(sps=sps, span=span))
    return z[span * sps: span * sps + n_sym * sps: sps]
