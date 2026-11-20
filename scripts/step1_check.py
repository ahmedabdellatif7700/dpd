"""Step 1 check: ILA of the transformer DPD on the test PA (dpd/pa.py), and its report.

The drive x: 16QAM RRC at N_DPD = 8 samples per symbol (3.2 GS/s), scaled to the data's RMS at the
PA input (0.371 a0); the PA sees x plus the pilot tone (1.104 a0), its output has the tone removed.
Two PA cases: the nominal link's (Rapp + AM/PM, memoryless) and the same with rf_amp.m's memory
option on. Training sequence seed 1, validation 4242, test 2017 (the seeds of matlab/dpd/README.md).
Writes results/step1.json, results/step1.txt and the trained models (results/models/*.pt).

    python scripts/step1_check.py [--device cuda|cpu] [--passes 3] [--epochs 30]
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dpd import pa, signals                               # noqa: E402
from dpd.evaluate import FS, drive, measures              # noqa: E402
from dpd.evaluate import link as make_link                # noqa: E402
from dpd.ila import ila                                   # noqa: E402
from dpd.models import DPD, TransformerDPD                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-train", type=int, default=4096, help="training symbols")
    ap.add_argument("--n-eval", type=int, default=2048, help="validation and test symbols")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cases", default="nominal,nominal_memory")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "models"), exist_ok=True)
    torch.manual_seed(0)

    x_tr, _ = drive(a.n_train, 1)
    x_va, d_va = drive(a.n_eval, 4242)
    x_te, d_te = drive(a.n_eval, 2017)
    cases = {"nominal": pa.pa_params(), "nominal_memory": pa.pa_params(memory={"enable": True})}
    cases = {k: cases[k] for k in a.cases.split(",")}
    rep = {"env": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                   "device": a.device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "threads": torch.get_num_threads()},
           "setup": {"sps": signals.N_DPD, "fs": FS, "tone_a0": pa.TONE_A0, "data_rms_a0": pa.DATA_RMS_A0,
                     "n_train_samples": int(x_tr.size), "n_eval_samples": int(x_te.size),
                     "passes": a.passes, "epochs": a.epochs, "batch": 512, "lr": 2e-3},
           "cases": {}}
    for name, p in cases.items():
        link = make_link(p)
        dpd = DPD(TransformerDPD(), a.device)
        rep["model"] = json.loads(dpd.describe())
        print(f"== {name}: {dpd.describe()}", flush=True)
        t0 = time.perf_counter()
        G, hist = ila(dpd, link, x_tr, passes=a.passes, epochs=a.epochs, batch=512, lr=2e-3, log=print)
        t_train = time.perf_counter() - t0
        dpd.save(os.path.join(a.out, "models", f"transformer_{name}.pt"))
        r = {"G": [G.real, G.imag], "train_time_s": t_train,
             "pass_post_inverse_nmse_db": [h["nmse_db"][-1] for h in hist],
             "validation": {"no_dpd": measures(link, x_va, d_va), "dpd": measures(link, x_va, d_va, dpd.predistort(x_va))},
             "test": {"no_dpd": measures(link, x_te, d_te), "dpd": measures(link, x_te, d_te, dpd.predistort(x_te))}}
        rep["cases"][name] = r
        print(json.dumps(r["test"], indent=1), flush=True)

    with open(os.path.join(a.out, "step1.json"), "w") as f:
        json.dump(rep, f, indent=1)
    m = rep["model"]
    lines = [f"Step 1: transformer DPD by ILA on the test PA (dpd/pa.py), {a.device} ({rep['env']['gpu'] or 'CPU'}), "
             f"torch {torch.__version__}",
             f"Model: window {m['window']}, width {m['d_model']}, {m['n_heads']} heads, {m['n_layers']} layers, FF {m['d_ff']}: "
             f"{m['params']} parameters, {m['macs'] / 1e3:.1f} k MAC = {m['flops'] / 1e3:.1f} k FLOP per sample",
             f"Drive: 16QAM RRC (0.75) at {signals.N_DPD} sps ({FS / 1e9:.1f} GS/s), data RMS {pa.DATA_RMS_A0} a0 + tone "
             f"{pa.TONE_A0} a0; train {x_tr.size} samples (seed 1), test {x_te.size} (seed 2017); "
             f"ILA {a.passes} passes x {a.epochs} epochs, batch 512, Adam 2e-3 cosine", ""]
    lines.append(f"{'case':16s} {'train s':>8s}   {'NMSE dB':>17s}   {'EVM waveform %':>17s}   {'EVM symbols %':>17s}")
    lines.append(f"{'':16s} {'':>8s}   {'no DPD -> DPD':>17s}   {'no DPD -> DPD':>17s}   {'no DPD -> DPD':>17s}")
    for name, r in rep["cases"].items():
        n, d = r["test"]["no_dpd"], r["test"]["dpd"]
        lines.append(f"{name:16s} {r['train_time_s']:8.1f}   {n['nmse_db']:7.2f} -> {d['nmse_db']:6.2f}   "
                     f"{n['evm_waveform_pct']:7.2f} -> {d['evm_waveform_pct']:6.2f}   "
                     f"{n['evm_symbols_pct']:7.2f} -> {d['evm_symbols_pct']:6.2f}")
    lines.append("")
    lines.append("validation (seed 4242), same columns:")
    for name, r in rep["cases"].items():
        n, d = r["validation"]["no_dpd"], r["validation"]["dpd"]
        lines.append(f"{name:16s} {'':8s}   {n['nmse_db']:7.2f} -> {d['nmse_db']:6.2f}   "
                     f"{n['evm_waveform_pct']:7.2f} -> {d['evm_waveform_pct']:6.2f}   "
                     f"{n['evm_symbols_pct']:7.2f} -> {d['evm_symbols_pct']:6.2f}")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(a.out, "step1.txt"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
