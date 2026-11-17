"""Export of the reference DPD for the C++/CUDA twin: the weights, and golden files to check it.

One file format for both (docs/dpd/FORMAT.md): a list of named float32 tensors, little-endian.

    magic   4 bytes   "DPDT"
    version u32       1
    count   u32       number of tensors
    then, per tensor:
      name_len u16, name (ASCII, name_len bytes)
      ndim     u8,  dims (u32 x ndim)
      data     float32 x prod(dims), row-major (C order)

Weights file (`export_weights`): `config` [5] = window, d_model, n_heads, n_layers, d_ff;
`scale` [1]; then the model's tensors in the order of its state_dict, with its names and shapes
(docs/dpd/METHODOLOGY.md section 2). Integers in `config` are exact in float32.

Golden file (`export_golden`): one BASEBAND-sized frame through the reference, float32:
`input` [N, 2] and `output` [N, 2] (I/Q; the frame alone, zeros beyond its ends, as `predistort`
does), `probe_index` [P] (the probed output samples), and for each the intermediate values
`probe.embed` [P, window, d], `probe.layer<l>` [P, window, d], `probe.final_norm` [P, d],
`probe.out` [P, 2] (before the scale, see METHODOLOGY).
"""
import struct

import numpy as np
import torch

from .models import DPD, MODELS, TransformerDPD, to_iq, windows

MAGIC = b"DPDT"
VERSION = 1
CONFIG_KEYS = ("window", "d_model", "n_heads", "n_layers", "d_ff")


def write_tensors(path, tensors):
    """tensors: list of (name, array); written as float32"""
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<II", VERSION, len(tensors)))
        for name, a in tensors:
            a = np.ascontiguousarray(np.asarray(a, dtype="<f4"))
            nb = name.encode("ascii")
            f.write(struct.pack("<H", len(nb)) + nb + struct.pack("<B", a.ndim))
            f.write(struct.pack(f"<{a.ndim}I", *a.shape))
            f.write(a.tobytes())


def read_tensors(path):
    """-> dict name -> float32 array, in the file's order"""
    with open(path, "rb") as f:
        b = f.read()
    if b[:4] != MAGIC:
        raise ValueError(f"{path}: not a DPDT file")
    version, count = struct.unpack_from("<II", b, 4)
    if version != VERSION:
        raise ValueError(f"{path}: version {version}, expected {VERSION}")
    p, out = 12, {}
    for _ in range(count):
        (n,) = struct.unpack_from("<H", b, p)
        name = b[p + 2:p + 2 + n].decode("ascii")
        p += 2 + n
        (ndim,) = struct.unpack_from("<B", b, p)
        dims = struct.unpack_from(f"<{ndim}I", b, p + 1)
        p += 1 + 4 * ndim
        size = int(np.prod(dims)) if ndim else 1
        out[name] = np.frombuffer(b, "<f4", size, p).reshape(dims).copy()
        p += 4 * size
    if p != len(b):
        raise ValueError(f"{path}: {len(b) - p} trailing bytes")
    return out


def export_weights(dpd, path):
    m = dpd.model
    if m.kind != "transformer":
        raise ValueError(f"export: model kind {m.kind!r} not supported")
    sd = m.state_dict()
    tensors = [("config", [m.config[k] for k in CONFIG_KEYS]), ("scale", [float(sd["scale"])])]
    tensors += [(k, v.detach().cpu().numpy()) for k, v in sd.items() if k != "scale"]
    write_tensors(path, tensors)


def load_weights(path, device=None):
    """a weights file back into a DPD (the round trip of the export)"""
    t = read_tensors(path)
    cfg = {k: int(v) for k, v in zip(CONFIG_KEYS, t.pop("config"))}
    model = MODELS["transformer"](**cfg)
    sd = {"scale": torch.tensor(float(t.pop("scale")[0]))}
    sd.update({k: torch.from_numpy(v) for k, v in t.items()})
    model.load_state_dict(sd)
    model._scaled = True
    return DPD(model, device)


@torch.no_grad()
def export_golden(dpd, x, probes, path):
    """x: one frame (complex); probes: output sample indices whose intermediates are kept.
    Runs on dpd's device; returns the output (complex64)."""
    m = dpd.model.eval()
    xr = torch.from_numpy(to_iq(x)).to(dpd.device)
    W = windows(xr, m.window, m.centre)
    out = torch.cat([m(W[i:i + dpd.chunk]) for i in range(0, W.shape[0], dpd.chunk)])
    probes = np.asarray(probes, dtype=np.int64)
    tr = {}
    m(W[torch.from_numpy(probes).to(dpd.device)], trace=tr)
    tensors = [("input", to_iq(x)), ("output", out.cpu().numpy()), ("probe_index", probes.astype(np.float32))]
    tensors += [(f"probe.{k}", v.cpu().numpy()) for k, v in tr.items()]
    write_tensors(path, tensors)
    o = out.cpu().numpy()
    return (o[:, 0] + 1j * o[:, 1]).astype(np.complex64)


__all__ = ["write_tensors", "read_tensors", "export_weights", "load_weights", "export_golden", "TransformerDPD"]
