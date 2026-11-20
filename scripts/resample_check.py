"""Step 3: the resampler 50 <-> 8 samples per symbol (dpd/resample.py), and the DPD through it.

1. The filter: passband ripple and stopband rejection.
2. The formula against scipy.signal.upfirdn (an independent implementation).
3. Alignment: the BASEBAND waveform made at 50 samples per symbol, resampled to 8, against the same
   symbols made at 8 (the models' training rate), with the RRC's gain 1/sqrt(sps) fitted.
4. The round trip 50 -> 8 -> 50, on the whole drive and with each 6400-sample frame alone
   (decision 3a): the resampler's own error, and what cutting it at the frames costs.
5. The DPD as the stage runs it: the drive at 50 samples per symbol, each frame alone, 50 -> 8,
   the model, 8 -> 50, then the test PA at 20 GS/s; against the reference at 8 (step 1) and the
   drive taken whole. Test sequence (seed 2017), 2048 symbols.
Writes results/step3_resampler.txt and .json.

    python scripts/resample_check.py
"""
import json
import os
import sys

import numpy as np
from scipy.signal import freqz, upfirdn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dpd import metrics, pa, resample as R, signals          # noqa: E402
from dpd.evaluate import drive, link, measures               # noqa: E402
from dpd.models import DPD                                   # noqa: E402

FS50 = signals.RS * R.SPS_PIPELINE
FRAME = 128 * R.SPS_PIPELINE      # 6400


def nmse_db(a, ref):
    return float(10 * np.log10(np.sum(np.abs(a - ref) ** 2) / np.sum(np.abs(ref) ** 2)))


def drive50(n_sym, seed):
    """the drive at 50 samples per symbol, scaled to the data's RMS at the PA input (as at 8)"""
    s, d = signals.baseband(n_sym, sps=R.SPS_PIPELINE, seed=seed)
    return s * pa.DATA_RMS_A0 / np.sqrt(np.mean(np.abs(s) ** 2)), d


def measures50(link_fn, x, d, x_dpd=None):
    xd = x if x_dpd is None else x_dpd
    y = link_fn(xd)
    return {"nmse_db": metrics.nmse_db(y, x), "evm_waveform_pct": 100 * metrics.evm_waveform(y, x, FS50),
            "evm_symbols_pct": 100 * metrics.evm_symbols(y, d, sps=R.SPS_PIPELINE)}


def main():
    rep, lines = {}, ["Step 3: the resampler 50 <-> 8 samples per symbol (dpd/resample.py) and the DPD through it", ""]
    h = R.lowpass()

    # 1. the filter
    w, H = freqz(h, worN=2 ** 16, fs=R.FS_HIGH)
    pb, sb = np.abs(H[w <= R.CUTOFF - R.TRANSITION / 2]), np.abs(H[w >= R.CUTOFF + R.TRANSITION / 2])
    rep["filter"] = {"taps": int(h.size), "passband_ripple_db": float(20 * np.log10(pb.max() / pb.min())),
                     "stopband_db": float(20 * np.log10(sb.max()))}
    lines.append(f"1. filter: {h.size} taps at 80 GS/s, passband (to 1.2 GHz) ripple {rep['filter']['passband_ripple_db']:.4f} dB, "
                 f"stopband (from 2.0 GHz) {rep['filter']['stopband_db']:.1f} dB")

    # 2. the formula against upfirdn
    r = np.random.default_rng(0)
    x = r.standard_normal(700) + 1j * r.standard_normal(700)
    c = (h.size - 1) // 2
    diffs = []
    for up, down in ((R.UP, R.DOWN), (R.DOWN, R.UP)):
        ref = upfirdn(h * up, x, up, down=1)[c::down][:R.out_length(x.size, up, down)]
        diffs.append(float(np.max(np.abs(R.resample(x, up, down, h) - ref))))
    rep["formula_vs_upfirdn"] = max(diffs)
    lines.append(f"2. the formula against scipy.signal.upfirdn: largest difference {max(diffs):.1e}")

    # 3. alignment
    s50, _ = signals.baseband(512, sps=R.SPS_PIPELINE, seed=3)
    s8, _ = signals.baseband(512, sps=R.SPS_DPD, seed=3)
    s8r = R.to_dpd(s50, h)
    mid = slice(200, s8.size - 200)
    g = np.vdot(s8r[mid], s8[mid]) / np.vdot(s8r[mid], s8r[mid])
    rep["alignment"] = {"gain": abs(g), "error_db": nmse_db(g * s8r[mid], s8[mid])}
    lines.append(f"3. 50 -> 8 against the waveform made at 8: gain {abs(g):.5f} (sqrt(50/8) = 2.5, the RRC's), "
                 f"error {rep['alignment']['error_db']:.1f} dB, no delay")

    # 4. round trip, whole and frame by frame
    x50, d50 = drive50(2048, 2017)
    whole = R.to_pipeline(R.to_dpd(x50, h), h)[:x50.size]
    framed = np.concatenate([R.to_pipeline(R.to_dpd(x50[s:s + FRAME], h), h)[:x50[s:s + FRAME].size]
                             for s in range(0, x50.size, FRAME)])
    edge = 2 * FRAME  # away from the drive's own ends
    rep["round_trip"] = {"whole_db": nmse_db(whole[edge:-edge], x50[edge:-edge]),
                         "frames_alone_db": nmse_db(framed[edge:-edge], x50[edge:-edge])}
    lines.append(f"4. round trip 50 -> 8 -> 50: error {rep['round_trip']['whole_db']:.1f} dB on the whole drive, "
                 f"{rep['round_trip']['frames_alone_db']:.1f} dB with each {FRAME}-sample frame alone")

    # 5. the DPD as the stage runs it, through the test PA
    lines += ["", "5. the DPD through the resampler, then the test PA at 20 GS/s; test drive (seed 2017), 2048 symbols",
              f"   {'PA':12s} {'DPD':44s} {'NMSE dB':>8s} {'EVM wf %':>9s} {'EVM sym %':>9s}"]
    x8, d8 = drive(2048, 2017)
    rep["dpd"] = {}
    for name, params in (("nominal", pa.pa_params()), ("memory", pa.pa_params(memory={"enable": True}))):
        dpd = DPD.load(f"results/models/transformer_nominal{'' if name == 'nominal' else '_memory'}.pt", "cpu")
        l8, l50 = link(params), (lambda v, p=params: pa.pa_chain(v, p, fs=FS50))
        rows = {"none (at 50)": measures50(l50, x50, d50),
                "reference at 8 (step 1)": measures(l8, x8, d8, dpd.predistort(x8)),
                "at 50 through the resampler, drive whole": measures50(l50, x50, d50, R.predistort_pipeline(dpd, x50, None, h)),
                "at 50 through the resampler, frames alone": measures50(l50, x50, d50, R.predistort_pipeline(dpd, x50, FRAME, h))}
        rep["dpd"][name] = rows
        for k, m in rows.items():
            lines.append(f"   {name:12s} {k:44s} {m['nmse_db']:8.2f} {m['evm_waveform_pct']:9.3f} {m['evm_symbols_pct']:9.3f}")
    lines.append("   (the memory PA at 20 GS/s delays by round(0.6 ns x 20 GS/s) = 12 samples = 0.600 ns; the model was "
                 "trained at 3.2 GS/s, where it is 2 samples = 0.625 ns)")
    txt = "\n".join(lines) + "\n"
    with open("results/step3_resampler.txt", "w") as f:
        f.write(txt)
    with open("results/step3_resampler.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(txt)


if __name__ == "__main__":
    main()
