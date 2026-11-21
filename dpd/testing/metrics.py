"""Measures at the PA's output, against the undistorted drive x.

All three fit a complex gain and a constant: the pilot tone is removed from the PA's output as a
constant (its mean, as the ILA recording and the receiver do), which also takes the data's own small
mean with it; neither is an error of the chain.

- NMSE: the whole band, after the gain and constant fitted by least squares.
- EVM of the waveform: matlab/paper_rewrite/validation/main_pa_operating_point.m's definition (the
  data at the PA's output, in-band |f| <= Rs(1+beta)/2, fitted by a complex gain on the data and a
  constant; the residual over the fitted data), with the undistorted drive as the reference.
- EVM of the symbols: matched filter, symbol-rate samples, one complex LS gain and constant (the
  receiver up to its decisions, no equalizer), edge symbols dropped.
"""
import numpy as np

from . import signals


def _fit(y, x):
    """least squares y ~ c x + k: returns c x + k and c"""
    A = np.stack([x, np.ones_like(x)], axis=1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ c, c[0]


def nmse_db(y, x):
    """10 log10(|y - (c x + k)|^2 / |c x|^2), c and k fitted by least squares"""
    y, x = np.asarray(y, dtype=np.complex128), np.asarray(x, dtype=np.complex128)
    fit, c = _fit(y, x)
    return float(10 * np.log10(np.sum(np.abs(y - fit) ** 2) / np.sum(np.abs(c * x) ** 2)))


def inband(v, fs, bw):
    """keep |f| <= bw/2 (FFT mask)"""
    f = np.fft.fftfreq(v.size, 1 / fs)
    return np.fft.ifft(np.fft.fft(v) * (np.abs(f) <= bw / 2))


def evm_waveform(y, x, fs, rs=signals.RS, rolloff=signals.ROLLOFF):
    """EVM (fraction) of the data at the PA's output, in-band, against the drive x"""
    bw = rs * (1 + rolloff)
    yb, xb = inband(np.asarray(y), fs, bw), inband(np.asarray(x), fs, bw)
    fit, c = _fit(yb, xb)
    return float(np.sqrt(np.mean(np.abs(yb - fit) ** 2) / np.mean(np.abs(c * xb) ** 2)))


def evm_symbols(y, d, sps=signals.N_DPD, edge=signals.SPAN):
    """EVM (fraction) of the matched-filtered symbols against the sent ones d, LS gain and constant"""
    z = signals.matched(np.asarray(y), d.size, sps)[edge:-edge]
    dd = d[edge:-edge]
    fit, c = _fit(z, dd)
    return float(np.sqrt(np.mean(np.abs(z - fit) ** 2) / np.mean(np.abs(c * dd) ** 2)))
