"""Test PA: a port of the link's PA, for the unit tests and the step 1 check only (not a link stage).

Ported from on-chip-training, branch `dpd`, `matlab/link/rf_amp.m` (AGC, Rapp AM/AM, Saleh AM/PM,
memory polynomial), line for line. The parameters are those of the nominal link of paper 1:
`matlab/link/default_link_odsb.m` (the 3GPP TR 38.803 28 GHz GaN PA fit: Rapp smoothness 1.94,
AM/PM alpha 0.51, beta 1.93; memory delay 0.6 ns, linear [-0.08 0.03], cubic [-0.05 0.02]) with the
nominal case of `matlab/core/impairment_sweeps_odsb.m` (`joint_nominal`: asat = 1.5 a0, AM/PM on;
memory off, it is an impairment-study option only).

The test chain (`pa_chain`) drives the PA with the data plus the pilot tone, at the levels the link
gives the PA (`matlab/results/pa_operating_point.txt`: tone 1.104 a0, data RMS 0.371 a0, 10.2 % of
the power) and removes the tone from the output as its mean, as the ILA recording does.
"""
import copy

import numpy as np

# matlab/link/default_link_odsb.m (link.rf) with the nominal case's asat and AM/PM (see above)
GAN_3GPP = {
    "enable": True, "asat": 1.5, "smooth": 1.94,
    "memory": {"enable": False, "delay_s": 0.6e-9, "linear": [-0.08, 0.03], "cubic": [-0.05, 0.02]},
    "ampm": {"enable": True, "alpha": 0.51, "beta": 1.93},
}

TONE_A0 = 1.104       # pilot tone at the PA input, / a0 (matlab/results/pa_operating_point.txt)
DATA_RMS_A0 = 0.371   # data RMS at the PA input, / a0

_EPS = np.finfo(np.float64).eps   # MATLAB's eps


def pa_params(**changes):
    """a copy of GAN_3GPP with top-level or nested changes, e.g. pa_params(memory={'enable': True})"""
    p = copy.deepcopy(GAN_3GPP)
    for k, v in changes.items():
        if isinstance(v, dict):
            p[k].update(v)
        else:
            p[k] = v
    return p


def _matlab_round(v):
    """MATLAB's round: halves away from zero (numpy rounds them to even)"""
    return int(np.sign(v) * np.floor(abs(v) + 0.5))


def rf_amp(A, p=None, a0=1.0, rop_dbm=-10.0, rop_ref_dbm=-10.0, fs=None):
    """rf_amp.m: AGC, then the PA. A: complex envelope (numpy). p: the PA's parameters (GAN_3GPP).
    a0: the reference amplitude (ref.a0). fs: sample rate, needed only for the memory option."""
    p = GAN_3GPP if p is None else p
    A = np.asarray(A, dtype=np.complex128).ravel()
    A = A / 10 ** ((rop_dbm - rop_ref_dbm) / 10)
    A_in = A
    if p["enable"]:
        r = np.abs(A)
        asat = p["asat"] * a0
        s = p["smooth"]
        A = A / np.maximum(r, _EPS) * (r / (1 + (r / asat) ** (2 * s)) ** (1 / (2 * s)))
    if "ampm" in p and p["ampm"]["enable"]:
        r2 = (np.abs(A_in) / (p["asat"] * a0)) ** 2
        A = A * np.exp(1j * p["ampm"]["alpha"] * r2 / (1 + p["ampm"]["beta"] * r2))
    if "memory" in p and p["memory"]["enable"]:
        if fs is None:
            raise ValueError("rf_amp: the memory option needs fs")
        D = _matlab_round(p["memory"]["delay_s"] * fs)
        u = A_in / a0
        for m, (lin, cub) in enumerate(zip(p["memory"]["linear"], p["memory"]["cubic"]), start=1):
            k = min(m * D, u.size)
            um = np.concatenate([np.zeros(k, dtype=np.complex128), u[:u.size - k]])
            A = A + a0 * (lin * um + cub * um * np.abs(um) ** 2)
    return A


def pa_chain(x, p=None, tone=TONE_A0, fs=None):
    """the test chain of the ILA check: drive x (the data at the PA input, / a0) plus the pilot tone
    -> PA -> output with the tone removed as its mean (/ a0). Memoryless in x apart from p's memory."""
    y = rf_amp(np.asarray(x) + tone, p, a0=1.0, fs=fs)
    return y - y.mean()
