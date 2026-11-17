"""DPD models (PyTorch) with a numpy boundary.

`TransformerDPD` is the network: a window of `window` complex drive samples, as I/Q pairs, through
a small transformer encoder; out comes the centre sample's I and Q. The centre sample is also passed
straight through (a skip), and the output head starts at zero, so an untrained model is the
identity: the first ILA pass starts from "no DPD".

`DPD` wraps a model for use from numpy (and so from MATLAB through bridge.py, which passes complex
signals as two float64 vectors):

    dpd = DPD(TransformerDPD())                 # or DPD.load(path)
    x_dpd = dpd.predistort(x)                   # complex numpy in, complex numpy out
    re, im = dpd.predistort_reim(re, im)        # the bridge's form
    dpd.fit(u, t)                               # train so that model(u) ~ t (the ILA post-inverse)

Window alignment: output sample n is computed from input samples n - c .. n - c + window - 1,
c = window // 2, with zeros beyond the ends of the sequence; there is no delay.
"""
import json
import time

import numpy as np
import torch
from torch import nn


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class TransformerDPD(nn.Module):
    """window of I/Q pairs [B, window, 2] -> centre sample's I/Q [B, 2]"""

    kind = "transformer"

    def __init__(self, window=32, d_model=32, n_heads=4, n_layers=2, d_ff=64):
        super().__init__()
        self.config = {"window": window, "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "d_ff": d_ff}
        self.window, self.centre = window, window // 2
        self.embed = nn.Linear(2, d_model)
        self.pos = nn.Parameter(0.02 * torch.randn(window, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # input scale (the RMS of the training input), so that the model works in any units
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, w):
        z = w / self.scale
        h = self.encoder(self.embed(z) + self.pos)
        out = z[:, self.centre] + self.head(self.norm(h[:, self.centre]))
        return out * self.scale

    def ops_per_sample(self):
        """multiply-accumulates per output sample (the whole window is encoded for each sample;
        softmax, GELU and the layer norms not counted), and FLOPs = 2 MACs"""
        c = self.config
        T, d, f, L = c["window"], c["d_model"], c["d_ff"], c["n_layers"]
        per_layer = (3 * T * d * d      # Q, K, V projections
                     + 2 * T * T * d    # scores and the weighted sum
                     + T * d * d        # output projection
                     + 2 * T * d * f)   # feed-forward
        macs = T * 2 * d + L * per_layer + d * 2
        return {"macs": macs, "flops": 2 * macs}


MODELS = {"transformer": TransformerDPD}


def windows(xr, window, centre):
    """[N, 2] -> [N, window, 2], zero-padded at both ends (a view-free gather; xr on any device)"""
    n = xr.shape[0]
    pad = torch.zeros(window, 2, dtype=xr.dtype, device=xr.device)
    xp = torch.cat([pad[:centre], xr, pad[: window - 1 - centre]])
    return xp.unfold(0, window, 1).transpose(1, 2)[:n]


def to_iq(x):
    x = np.asarray(x).ravel()
    return np.stack([x.real, x.imag], axis=1).astype(np.float32)


def from_iq(v, dtype=np.complex128):
    return (v[:, 0].astype(np.float64) + 1j * v[:, 1].astype(np.float64)).astype(dtype)


class DPD:
    """a DPD model behind a numpy interface; float32 inside, on the CPU or CUDA"""

    def __init__(self, model=None, device=None, chunk=16384):
        self.device = torch.device(device or default_device())
        self.model = (model or TransformerDPD()).to(self.device)
        self.chunk = chunk

    # ---- inference ----
    @torch.no_grad()
    def predistort(self, x):
        """x (complex numpy, the drive) -> x_dpd (complex numpy, same length; complex64 in ->
        complex64 out, otherwise complex128)"""
        x = np.asarray(x)
        out_dtype = np.complex64 if x.dtype == np.complex64 else np.complex128
        if x.size == 0:
            return np.zeros(0, dtype=out_dtype)
        self.model.eval()
        m = self.model
        xr = torch.from_numpy(to_iq(x)).to(self.device)
        W = windows(xr, m.window, m.centre)
        out = torch.cat([m(W[i:i + self.chunk]) for i in range(0, W.shape[0], self.chunk)])
        return from_iq(out.cpu().numpy(), out_dtype)

    def predistort_reim(self, re, im):
        """for MATLAB: two float64 vectors in, two float64 vectors out"""
        z = self.predistort(np.asarray(re, dtype=np.float64).ravel() + 1j * np.asarray(im, dtype=np.float64).ravel())
        return np.ascontiguousarray(z.real), np.ascontiguousarray(z.imag)

    __call__ = predistort

    # ---- training ----
    def fit(self, u, t, epochs=40, batch=512, lr=2e-3, weight_decay=0.0, seed=0, set_scale=None, log=None):
        """train the model so that model(u) ~ t, both complex numpy of one length (ILA: u = y/G,
        t = the drive that made y). Adam, cosine learning rate, MSE on I/Q. The input scale is set
        from u on the first fit (set_scale=None) and kept after. Returns the loss history (NMSE, dB)
        and the time."""
        m = self.model
        if set_scale or (set_scale is None and not getattr(m, "_scaled", False)):
            m.scale.fill_(float(np.sqrt(np.mean(np.abs(u) ** 2))))
            m._scaled = True
        g = torch.Generator(device="cpu").manual_seed(seed)
        ur = torch.from_numpy(to_iq(u)).to(self.device)
        tr = torch.from_numpy(to_iq(t)).to(self.device)
        W = windows(ur, m.window, m.centre)
        n = W.shape[0]
        steps = epochs * max(1, n // batch)
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        p_t = float((tr ** 2).sum(1).mean())
        hist = []
        m.train()
        t0 = time.perf_counter()
        for ep in range(epochs):
            perm = torch.randperm(n, generator=g).to(self.device)
            tot = 0.0
            for i in range(0, n - batch + 1, batch):
                idx = perm[i:i + batch]
                loss = ((m(W[idx]) - tr[idx]) ** 2).sum(1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss) * batch
            hist.append(10 * np.log10(tot / (n - n % batch) / p_t))
            if log:
                log(f"  epoch {ep + 1:3d}/{epochs}  NMSE {hist[-1]:7.2f} dB")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        m.eval()
        return {"nmse_db": hist, "time_s": time.perf_counter() - t0, "steps": steps}

    # ---- bookkeeping ----
    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def copy_from(self, other):
        """copy another DPD's weights (the ILA's post-inverse -> predistorter)"""
        self.model.load_state_dict(other.model.state_dict())
        self.model._scaled = getattr(other.model, "_scaled", False)

    def save(self, path):
        torch.save({"kind": self.model.kind, "config": self.model.config,
                    "state": {k: v.cpu() for k, v in self.model.state_dict().items()}}, path)

    @classmethod
    def load(cls, path, device=None):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        model = MODELS[ck["kind"]](**ck["config"])
        model.load_state_dict(ck["state"])
        model._scaled = True
        return cls(model, device)

    def describe(self):
        return json.dumps({"kind": self.model.kind, **self.model.config, "params": self.n_params(),
                           **self.model.ops_per_sample(), "device": str(self.device)})
