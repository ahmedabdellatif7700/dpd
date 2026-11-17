"""Indirect learning (ILA) of a DPD (matlab/dpd/README.md, "How it learns").

One pass: the drive x is predistorted by the current DPD into x_d, which goes through the link; y
is the PA's output (aligned, the pilot tone removed). A post-inverse g is trained so that
g(y/G) ~ x_d, G the target linear gain, and copied in front of the link as the new DPD. The first
pass starts with no DPD (x_d = x); each later pass is recorded with the previous DPD in place, and
g is warm-started from it.

`ila_step` takes one recording (x_d, y): the form the Pi's data will have (the drive from BASEBAND,
the PA's output from its lossless tap). `ila` runs passes against a callable `link` (here the test
PA of dpd/pa.py); with the testbed, each pass is a run.
"""
import copy

import numpy as np

from .models import DPD


def target_gain(x, y):
    """G: the complex LS gain of y on x, with no DPD (the DPD keeps the output's mean power and
    phase and makes it linear)"""
    return complex(np.vdot(x, y) / np.vdot(x, x))


def ila_step(dpd, x_d, y, G, **fit_kw):
    """train a post-inverse g (a copy of dpd, warm start) on g(y/G) ~ x_d, copy it into dpd.
    Returns fit()'s history."""
    g = DPD(copy.deepcopy(dpd.model), dpd.device)
    g.model._scaled = getattr(dpd.model, "_scaled", False)
    hist = g.fit(np.asarray(y) / G, np.asarray(x_d), **fit_kw)
    dpd.copy_from(g)
    return hist


def ila(dpd, link, x, passes=3, G=None, log=None, **fit_kw):
    """ILA passes against link (callable: drive -> PA output, tone removed). x: the training drive.
    Returns G and the per-pass histories."""
    if G is None:
        G = target_gain(x, link(x))
    out = []
    for k in range(passes):
        x_d = x if k == 0 else dpd.predistort(x)
        y = link(x_d)
        h = ila_step(dpd, x_d, y, G, **fit_kw)
        out.append(h)
        if log:
            log(f"ILA pass {k + 1}/{passes}: post-inverse NMSE {h['nmse_db'][-1]:.2f} dB, {h['time_s']:.1f} s")
    return G, out
