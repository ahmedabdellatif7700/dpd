"""The resampler between the pipeline's rate and the DPD's (docs/testbed/jetson/STAGE.md, decision
2b): BASEBAND at 50 samples per symbol (20 GS/s) <-> the model at 8 (3.2 GS/s).

50 -> 8 is a rational ratio of 4/25: up by 4, filter, down by 25; 8 -> 50 is up by 25, filter, down
by 4. Both pass through the same intermediate rate, 80 GS/s, so one low-pass filter `h` serves both:
a Kaiser-windowed sinc at 80 GS/s that passes up to 1.2 GHz (the RRC band, ±0.35 GHz, and the DPD's
spectral growth) and stops from 2.0 GHz with 80 dB of rejection (503 taps, cut-off 1.6 GHz, the
Nyquist frequency of 3.2 GS/s).

One formula, written so that the C++/CUDA twin implements it as it stands:

    y[m] = up * sum_k x[k] h[c + m*down - k*up],    c = (len(h) - 1) / 2,

with x zero outside 0 .. n-1 and h zero outside its taps: output sample m sits at the time of input
sample m*down/up (no delay), and the sequence is taken alone, as the model takes a frame (decision
3a). The output has ceil(n * up / down) samples: a 6400-sample frame gives 1024 and back.
"""
import math

import numpy as np
from scipy.signal import firwin, kaiserord

SPS_PIPELINE = 50          # the BASEBAND frames
SPS_DPD = 8                # the model (signals.N_DPD)
UP, DOWN = 4, 25           # 50 -> 8 = 4/25 (and 8 -> 50 = 25/4)
FS_HIGH = 80e9             # the intermediate rate: 20 GS/s x 4 = 3.2 GS/s x 25
CUTOFF = 1.6e9             # -6 dB, the Nyquist frequency of 3.2 GS/s
TRANSITION = 0.8e9         # pass up to 1.2 GHz, stop from 2.0 GHz
ATTENUATION_DB = 80


def lowpass():
    """the filter h at 80 GS/s: Kaiser window for ATTENUATION_DB over TRANSITION (503 taps, beta
    7.857), unit gain at DC, symmetric, odd length"""
    numtaps, beta = kaiserord(ATTENUATION_DB, TRANSITION / (FS_HIGH / 2))
    numtaps += 1 - numtaps % 2
    return firwin(numtaps, CUTOFF, window=("kaiser", beta), fs=FS_HIGH)


def out_length(n, up, down):
    """the output samples of an n-sample sequence: those at times m*down/up < n"""
    return math.ceil(n * up / down)


def resample(x, up, down, h=None):
    """y[m] = up * sum_k x[k] h[c + m*down - k*up] (the module's formula), x taken alone"""
    h = lowpass() if h is None else np.asarray(h)
    x = np.asarray(x)
    L, c = h.size, (h.size - 1) // 2
    n_out = out_length(x.size, up, down)
    y = np.zeros(n_out, dtype=np.result_type(x, h))
    for m in range(n_out):
        t = c + m * down                                      # the filter index of x[0]
        k_lo = max(0, math.ceil((t - (L - 1)) / up))          # c + m*down - k*up <= L-1
        k_hi = min(x.size - 1, t // up)                       # c + m*down - k*up >= 0
        if k_hi < k_lo:
            continue
        k = np.arange(k_lo, k_hi + 1)
        y[m] = up * np.dot(x[k], h[t - k * up])
    return y


def to_dpd(x50, h=None):
    """50 -> 8 samples per symbol"""
    return resample(x50, UP, DOWN, h)


def to_pipeline(x8, h=None):
    """8 -> 50 samples per symbol"""
    return resample(x8, DOWN, UP, h)


def predistort_pipeline(dpd, x50, frame=6400, h=None):
    """what the stage does to the BASEBAND drive: each frame alone (decision 3a), 50 -> 8, the
    model, 8 -> 50. frame=None takes the drive whole. Returns the predistorted drive at 50."""
    h = lowpass() if h is None else h
    x50 = np.asarray(x50)
    step = x50.size if frame is None else frame
    out = np.empty(x50.size, dtype=np.complex128)
    for s in range(0, x50.size, step):
        f = x50[s:s + step]
        out[s:s + f.size] = to_pipeline(dpd.predistort(to_dpd(f, h)), h)[:f.size]
    return out
