"""Export a trained reference DPD for the C++/CUDA twin (step 2, part 1).

Writes, next to the model:
  <name>.dpdt          the weights (docs/dpd/FORMAT.md)
  golden/<name>.dpdt   one BASEBAND-sized frame (6400 samples) through the reference on the CPU,
                       with the intermediate values of a few probed samples

The frame is the test drive of step 1 (seed 2017, data RMS 0.371 a0, 8 samples per symbol), cut
to 6400 samples. The probes cover both ends of the frame (where the window reaches past it,
zero-padded), the first and last samples whose window is all inside, and the middle.

    python scripts/export_model.py [results/models/transformer_nominal.pt ...]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dpd.testing import pa, signals                                  # noqa: E402
from dpd.export import (export_golden, export_resampler, export_resampler_golden, export_weights,  # noqa: E402
                        load_weights, read_tensors)
from dpd.models import DPD                                   # noqa: E402

FRAME = 6400          # samples in one BASEBAND frame (128 symbols x 50)


def golden_frame():
    s, _ = signals.baseband(FRAME // signals.N_DPD + 2 * signals.SPAN, seed=2017)
    x = s * pa.DATA_RMS_A0 / np.sqrt(np.mean(np.abs(s) ** 2))
    return x[:FRAME].astype(np.complex64)


def probes(window=32):
    c = window // 2
    return [0, 1, c - 1, c, FRAME // 2, FRAME - window + c, FRAME - window + c + 1, FRAME - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=["results/models/transformer_nominal.pt",
                                                   "results/models/transformer_nominal_memory.pt"])
    a = ap.parse_args()
    x = golden_frame()
    for path in a.models:
        base = os.path.splitext(path)[0]
        name = os.path.basename(base)
        dpd = DPD.load(path, "cpu")
        w = base + ".dpdt"
        export_weights(dpd, w)
        # the round trip: the exported weights give the same outputs, bit for bit
        back = load_weights(w, "cpu")
        y_ref = dpd.predistort(x)
        assert np.array_equal(back.predistort(x), y_ref), "weights round trip differs"
        gdir = os.path.join(os.path.dirname(path), "golden")
        os.makedirs(gdir, exist_ok=True)
        g = os.path.join(gdir, name + ".dpdt")
        y = export_golden(dpd, x, probes(dpd.model.window), g)
        assert np.array_equal(y, y_ref), "golden output differs from predistort"
        t = read_tensors(g)
        print(f"{name}: {w} ({os.path.getsize(w)} B, {len(read_tensors(w))} tensors), "
              f"{g} ({os.path.getsize(g)} B: {', '.join(f'{k} {list(v.shape)}' for k, v in t.items())})")
    export_resampler_files(os.path.dirname(a.models[0]) or ".")


def export_resampler_files(models_dir):
    """the resampler 50 <-> 8 (dpd/resample.py) and its golden frame: the BASEBAND drive at 50
    samples per symbol, a 6400-sample frame from the middle of the step 1 test drive"""
    w = os.path.join(models_dir, "resampler.dpdt")
    g = os.path.join(models_dir, "golden", "resampler.dpdt")
    export_resampler(w)
    s, _ = signals.baseband(256, sps=50, seed=2017)
    x = s * pa.DATA_RMS_A0 / np.sqrt(np.mean(np.abs(s) ** 2))
    export_resampler_golden(x[3200:3200 + FRAME], g)
    t = read_tensors(g)
    print(f"resampler: {w} ({os.path.getsize(w)} B), {g} ({', '.join(f'{k} {list(v.shape)}' for k, v in t.items())})")


if __name__ == "__main__":
    main()
