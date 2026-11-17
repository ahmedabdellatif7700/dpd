"""docs/dpd/METHODOLOGY.md section 2, written again in plain numpy (float64), against the PyTorch
model: the page describes exactly what the model computes (the C++/CUDA twin implements the page)."""
import math

import numpy as np
import torch

from dpd.models import DPD, TransformerDPD


def ln(v, w, b, eps=1e-5):
    mu = v.mean(-1, keepdims=True)
    var = ((v - mu) ** 2).mean(-1, keepdims=True)
    return (v - mu) / np.sqrt(var + eps) * w + b


def gelu(u):
    return 0.5 * u * (1 + np.vectorize(math.erf)(u / math.sqrt(2)))


def methodology(x, P, T=32, d=32, h=4, L=2):
    """the page's equations; P: the state_dict as float64 numpy"""
    c, dh = T // 2, d // h
    s = float(P["scale"])
    xp = np.concatenate([np.zeros(c, complex), x, np.zeros(T - 1 - c, complex)])
    out = np.empty(x.size, complex)
    for n in range(x.size):
        w = xp[n:n + T]
        z = np.stack([w.real, w.imag], 1) / s                                 # T x 2
        e = z @ P["embed.weight"].T + P["embed.bias"] + P["pos"]              # T x d
        for l in range(L):
            p = f"encoder.layers.{l}."
            a = ln(e, P[p + "norm1.weight"], P[p + "norm1.bias"])
            qkv = a @ P[p + "self_attn.in_proj_weight"].T + P[p + "self_attn.in_proj_bias"]
            q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
            o = np.empty((T, d))
            for j in range(h):
                cs = slice(j * dh, (j + 1) * dh)
                S = q[:, cs] @ k[:, cs].T / math.sqrt(dh)
                Pm = np.exp(S - S.max(1, keepdims=True))
                Pm /= Pm.sum(1, keepdims=True)
                o[:, cs] = Pm @ v[:, cs]
            e = e + o @ P[p + "self_attn.out_proj.weight"].T + P[p + "self_attn.out_proj.bias"]
            g = ln(e, P[p + "norm2.weight"], P[p + "norm2.bias"])
            e = e + gelu(g @ P[p + "linear1.weight"].T + P[p + "linear1.bias"]) @ P[p + "linear2.weight"].T \
                + P[p + "linear2.bias"]
        y = z[c] + ln(e[c], P["norm.weight"], P["norm.bias"]) @ P["head.weight"].T + P["head.bias"]
        out[n] = s * (y[0] + 1j * y[1])
    return out


def test_page_equals_model():
    torch.manual_seed(7)
    m = TransformerDPD()
    with torch.no_grad():                    # weights far from the identity, so every part counts
        for p in m.parameters():
            p.add_(0.2 * torch.randn_like(p))
        m.scale.fill_(0.37)
    r = np.random.default_rng(8)
    x = 0.4 * (r.standard_normal(80) + 1j * r.standard_normal(80))
    P = {k: v.double().numpy() for k, v in m.state_dict().items()}
    ref = methodology(x, P)
    got = DPD(m, "cpu").predistort(x)
    assert np.max(np.abs(got - ref)) / np.sqrt(np.mean(np.abs(ref) ** 2)) < 1e-5
    assert np.max(np.abs(ref - x)) > 0.1     # and the network did change the drive
