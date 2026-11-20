"""The export for the C++/CUDA twin: the file format, the weights' round trip, the golden probes."""
import struct

import numpy as np
import pytest
import torch

from dpd.export import export_golden, export_weights, load_weights, read_tensors, write_tensors
from dpd.models import DPD, TransformerDPD


def trained_like(seed=3):
    torch.manual_seed(seed)
    m = TransformerDPD()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1 * torch.randn_like(p))
        m.scale.fill_(0.37)
    return DPD(m, "cpu")


def sig(n, seed=4):
    r = np.random.default_rng(seed)
    return (0.4 * (r.standard_normal(n) + 1j * r.standard_normal(n))).astype(np.complex64)


def test_format_bytes(tmp_path):
    p = tmp_path / "t.dpdt"
    write_tensors(p, [("ab", np.array([[1.0, 2.0, 3.0]]))])
    b = p.read_bytes()
    # magic, version 1, one tensor; name "ab"; 2 dims [1, 3]; three float32
    assert b == (b"DPDT" + struct.pack("<II", 1, 1) + struct.pack("<H", 2) + b"ab" + struct.pack("<B", 2)
                 + struct.pack("<2I", 1, 3) + struct.pack("<3f", 1, 2, 3))
    t = read_tensors(p)
    assert list(t) == ["ab"] and t["ab"].dtype == np.float32 and t["ab"].shape == (1, 3)


def test_format_rejects(tmp_path):
    p = tmp_path / "bad.dpdt"
    p.write_bytes(b"XXXX" + bytes(8))
    with pytest.raises(ValueError):
        read_tensors(p)
    write_tensors(p, [("a", [1.0])])
    p.write_bytes(p.read_bytes() + b"\0")
    with pytest.raises(ValueError):
        read_tensors(p)


def test_weights_round_trip(tmp_path):
    d = trained_like()
    export_weights(d, tmp_path / "w.dpdt")
    t = read_tensors(tmp_path / "w.dpdt")
    assert list(t)[:4] == ["config", "scale", "pos", "embed.weight"]
    assert t["config"].tolist() == [32, 32, 4, 2, 64] and t["scale"][0] == np.float32(0.37)
    assert sum(v.size for k, v in t.items() if k not in ("config", "scale")) == d.n_params()
    x = sig(500)
    assert np.array_equal(load_weights(tmp_path / "w.dpdt", "cpu").predistort(x), d.predistort(x))


def test_trace_is_the_forward():
    d = trained_like()
    w = torch.randn(5, 32, 2)
    tr = {}
    out = d.model(w, trace=tr)
    assert torch.equal(out, d.model(w))
    assert torch.allclose(tr["out"] * d.model.scale, out)
    assert set(tr) == {"embed", "layer0", "layer1", "final_norm", "out"}
    assert tr["layer1"].shape == (5, 32, 32) and tr["final_norm"].shape == (5, 32)


def test_golden(tmp_path):
    d = trained_like()
    x = sig(6400)
    probes = [0, 16, 3200, 6399]
    y = export_golden(d, x, probes, tmp_path / "g.dpdt")
    assert np.array_equal(y, d.predistort(x))
    g = read_tensors(tmp_path / "g.dpdt")
    assert g["input"].shape == (6400, 2) and g["output"].shape == (6400, 2)
    assert g["probe_index"].tolist() == probes
    # the probes' outputs are the frame's outputs at those samples
    assert np.allclose(g["probe.out"] * 0.37, g["output"][probes], atol=1e-6)
    # probe 0's window starts 16 zeros before the frame: its first 16 embeddings are pos + bias only
    emb0 = g["probe.embed"][0]
    base = d.model.pos.detach().numpy() + d.model.embed.bias.detach().numpy()
    assert np.allclose(emb0[:16], base[:16], atol=1e-6)
