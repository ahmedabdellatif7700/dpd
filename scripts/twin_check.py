"""Step 2, part 4: the C++/CUDA twin through the test PA, against the PyTorch reference.

For each trained model, on the step 1 validation (seed 4242) and test (seed 2017) drives: the
drive predistorted by the reference (PyTorch, CPU) and by the twin (cpp/bin/dpd_apply, on the GPU
and on the CPU), then each through the test PA it was trained for. Reports, per sequence, the
largest difference of the twin's output from the reference's (over the RMS), and the NMSE and EVM
at the PA's output with no DPD, the reference and the twin. The twin passes if its outputs are
within --tol of the reference's and its NMSE and EVM equal the reference's to 0.01 (dB, %).
Writes results/step2_twin.json and results/step2_twin.txt.

    python scripts/twin_check.py [--bin cpp/bin/dpd_apply] [--tol 1e-5]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dpd import pa                                            # noqa: E402
from dpd.evaluate import drive, link, measures                # noqa: E402
from dpd.export import read_tensors, write_tensors            # noqa: E402
from dpd.models import DPD, to_iq                             # noqa: E402

# the trained models and the PA each was trained for (scripts/step1_check.py)
CASES = {"transformer_nominal": pa.pa_params(), "transformer_nominal_memory": pa.pa_params(memory={"enable": True})}
SEQUENCES = {"validation": 4242, "test": 2017}
N_SYM = 2048
METRIC_TOL = 0.01  # dB for NMSE, percentage points for EVM


def twin(binary, weights, x, device, tmp):
    """x (complex) through dpd_apply on device; returns the twin's output (complex)"""
    fin, fout = os.path.join(tmp, "in.dpdt"), os.path.join(tmp, "out.dpdt")
    write_tensors(fin, [("input", to_iq(x))])
    subprocess.run([binary, "--device", device, weights, fin, fout], check=True, capture_output=True, text=True)
    y = read_tensors(fout)["output"]
    return y[:, 0].astype(np.float64) + 1j * y[:, 1].astype(np.float64)


def rel_err(a, b):
    return float(np.max(np.abs(a - b)) / np.sqrt(np.mean(np.abs(b) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="cpp/bin/dpd_apply")
    ap.add_argument("--models", default="results/models")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rep, ok = {"n_sym": N_SYM, "tol": a.tol, "metric_tol": METRIC_TOL, "cases": {}}, True
    with tempfile.TemporaryDirectory() as tmp:
        for name, params in CASES.items():
            ref = DPD.load(os.path.join(a.models, name + ".pt"), "cpu")
            weights = os.path.join(a.models, name + ".dpdt")
            pa_link = link(params)
            rep["cases"][name] = {}
            for seq, seed in SEQUENCES.items():
                x, d = drive(N_SYM, seed)
                y_ref = ref.predistort(x)
                r = {"samples": int(x.size), "no_dpd": measures(pa_link, x, d), "reference": measures(pa_link, x, d, y_ref)}
                for device in ("cuda", "cpu"):
                    y = twin(a.bin, weights, x, device, tmp)
                    e = rel_err(y, y_ref)
                    m = measures(pa_link, x, d, y)
                    dm = max(abs(m[k] - r["reference"][k]) for k in ("nmse_db", "evm_waveform_pct", "evm_symbols_pct"))
                    r[f"twin_{device}"] = {**m, "output_error": e, "metric_diff": dm}
                    ok = ok and e <= a.tol and dm <= METRIC_TOL
                rep["cases"][name][seq] = r
                print(f"{name} {seq}: twin GPU error {r['twin_cuda']['output_error']:.2e}, "
                      f"CPU {r['twin_cpu']['output_error']:.2e}", flush=True)
    rep["ok"] = ok

    lines = [f"Step 2, part 4: the C++/CUDA twin (cpp/bin/dpd_apply) through the test PA, against the PyTorch "
             f"reference; {N_SYM} symbols at 8 samples per symbol per sequence",
             "output error: the largest difference from the reference's output over its RMS; NMSE (dB) and "
             "EVM (%) at the PA's output", ""]
    hdr = f"{'model':28s} {'sequence':10s} {'DPD':10s} {'output error':>12s} {'NMSE':>9s} {'EVM wf':>8s} {'EVM sym':>8s}"
    lines += [hdr, "-" * len(hdr)]
    for name, seqs in rep["cases"].items():
        for seq, r in seqs.items():
            for key, label in (("no_dpd", "none"), ("reference", "PyTorch"), ("twin_cuda", "twin GPU"),
                               ("twin_cpu", "twin CPU")):
                m = r[key]
                err = f"{m['output_error']:.2e}" if "output_error" in m else ""
                lines.append(f"{name:28s} {seq:10s} {label:10s} {err:>12s} {m['nmse_db']:9.3f} "
                             f"{m['evm_waveform_pct']:8.3f} {m['evm_symbols_pct']:8.3f}")
    twins = [r[t] for seqs in rep["cases"].values() for r in seqs.values() for t in ("twin_cuda", "twin_cpu")]
    rep["max_output_error"] = max(t["output_error"] for t in twins)
    rep["max_metric_diff"] = max(t["metric_diff"] for t in twins)
    lines += ["", f"largest output error {rep['max_output_error']:.1e}; largest NMSE/EVM difference from the "
              f"reference {rep['max_metric_diff']:.1e} (dB, %)",
              f"tolerances: output {a.tol:.0e}, NMSE and EVM {METRIC_TOL}: {'OK' if ok else 'FAILED'}"]
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(a.out, "step2_twin.json"), "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(a.out, "step2_twin.txt"), "w") as f:
        f.write(txt)
    print(txt)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
