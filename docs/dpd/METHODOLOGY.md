# The reference DPD, equation by equation

Every operation of the reference (`dpd/`), in the order it is computed, with the constants that
fix it. The C++/CUDA twin (`cpp/`, step 2) implements this page; `tests/test_methodology.py`
implements it again in plain numpy and checks it against the PyTorch model, so this page and the
model cannot drift apart.

Notation: vectors are rows; `W x + b` for a `nn.Linear` with weight `W` (out × in) and bias `b`;
`d = 32` the model width, `T = 32` the window, `c = T/2 = 16` the centre, `h = 4` heads of
`d_h = d/h = 8`, `f = 64` the feed-forward width, `L = 2` layers. All arithmetic in float32.

## 1. The window

The drive `x[n]` (complex) becomes I/Q pairs `[Re x[n], Im x[n]]`. Output sample `n` is computed
from the window

```
w_t = x[n - c + t],   t = 0 .. T-1        (x[k] = 0 for k < 0 or k >= N)
```

so the window holds 16 samples before `n`, `n` itself at `t = c`, and 15 after. No delay: output
`n` is aligned with input `n`. Each output sample's window is encoded on its own (no state between
samples or frames).

## 2. The network, for one window

Tensors as named in the model's `state_dict` (the export of step 2 keeps these names).

**Input scale.** `z_t = w_t / s`, `s` = `scale` (a scalar, the RMS of the first training input).

**Embedding.** For each token `t`:

```
e_t = W_embed z_t + b_embed + pos[t]          embed.weight (32×2), embed.bias (32), pos (32×32)
```

**Encoder layer `l`** (`encoder.layers.l.*`, pre-norm: `norm_first=True`), applied to all 32 tokens:

```
a_t   = LN1(e_t)                                          norm1.weight, norm1.bias
[q_t | k_t | v_t] = W_in a_t + b_in                       self_attn.in_proj_weight (96×32), in_proj_bias (96)
                                                          rows 0-31 -> q, 32-63 -> k, 64-95 -> v
for each head j = 0..3 (columns 8j .. 8j+7 of q, k, v):
    S_tu  = (q_t,j · k_u,j) / sqrt(8)                     t, u = 0..31, no mask
    P_tu  = exp(S_tu - max_u S_tu) / sum_u exp(S_tu - max_u S_tu)     softmax over u
    o_t,j = sum_u P_tu v_u,j
o_t   = concat_j o_t,j                                    (head j in columns 8j .. 8j+7)
e_t   = e_t + W_out o_t + b_out                           self_attn.out_proj.weight (32×32), .bias
g_t   = LN2(e_t)                                          norm2.weight, norm2.bias
e_t   = e_t + W_2 GELU(W_1 g_t + b_1) + b_2               linear1 (64×32), linear2 (32×64)
```

**Output.** Only the centre token is read:

```
y = z_c + W_head LN(e_c) + b_head                         norm.weight, norm.bias; head.weight (2×32), head.bias (2)
x_dpd[n] = s * (y_0 + j y_1)
```

The skip `z_c`: the network learns the correction to the identity; `head` starts at zero, so an
untrained model passes the drive through.

**Layer norm**, over the 32 features of one token, `eps = 1e-5`, biased variance:

```
mu = mean(v),  var = mean((v - mu)^2),  LN(v) = (v - mu) / sqrt(var + 1e-5) * weight + bias
```

**GELU**, the exact form (PyTorch's default, not the tanh approximation):

```
GELU(u) = 0.5 u (1 + erf(u / sqrt(2)))
```

No dropout (0), no final norm of the encoder (`encoder.norm` is None).

## 3. Size and cost

Parameters: `pos` 1024, `embed` 96, each layer 8544 (in-projection 3168, out-projection 1056,
`linear1` 2112, `linear2` 2080, two norms 128), `norm` 64, `head` 66: **18 338**, plus the scale.

Multiply-accumulates per output sample, the whole window encoded (`TransformerDPD.ops_per_sample`):

| Part | MAC | For T = 32, d = 32, f = 64 |
|---|---|---|
| embedding | T·2·d | 2 048 |
| Q, K, V, per layer | 3·T·d² | 98 304 |
| scores and weighted sum, per layer | 2·T²·d | 65 536 |
| output projection, per layer | T·d² | 32 768 |
| feed-forward, per layer | 2·T·d·f | 131 072 |
| head | 2·d | 64 |
| **total, 2 layers** | | **657 472** (1.31 MFLOP) |

Layer norms, softmax and GELU are not counted. Two savings are exact and left to the twin:

- the last layer needs K and V of all 32 tokens but its query, attention, output projection and
  feed-forward only for the centre token: 73 728 MAC instead of 327 680 (77 % of that layer, 39 % of
  the total), 403 520 MAC per sample in all;
- the embeddings of neighbouring windows are the same samples at shifted positions.

## 4. ILA (indirect learning)

`dpd/ila.py`. The link takes the drive `x_d` (with the DPD in place) to the PA's output `y`
(aligned, the pilot tone removed as its mean). The target gain `G` is fixed once, with no DPD:

```
G = <x, y> / <x, x>          (complex LS gain, <a, b> = sum conj(a) b)
```

Each pass trains a post-inverse `g` (a copy of the current DPD, warm start) on

```
minimize  mean_n |g(y/G)[n] - x_d[n]|^2      (the sum over I and Q of the squared error)
```

then copies `g`'s weights into the DPD. Pass 1 has no DPD (`x_d = x`); 3 passes. The input scale
`s` is set once, from `y/G` of pass 1. Training: Adam, learning rate 2e-3 with a cosine decay to 0
over all the steps, batches of 512 windows drawn at random (a fixed seed), 30 epochs a pass.

## 5. The test PA

`dpd/pa.py`, a port of on-chip-training `matlab/link/rf_amp.m`, line for line, `a0 = 1`:

```
A_in = A / 10^((rop - rop_ref)/10)                                   AGC (rop = rop_ref in the chain)
A    = A_in / max(|A_in|, eps) * |A_in| / (1 + (|A_in|/asat)^(2p))^(1/(2p))          Rapp AM/AM
A    = A * exp(j alpha r2 / (1 + beta r2)),  r2 = (|A_in|/asat)^2                     Saleh AM/PM
A    = A + sum_m (lin_m u_m + cub_m u_m |u_m|^2),  u_m = A_in delayed by m*D          memory (option)
D    = round(delay_s * fs)          MATLAB's round: halves away from zero; eps = 2^-52
```

The nominal link's PA: `asat = 1.5`, `p = 1.94`, `alpha = 0.51`, `beta = 1.93`; memory off
(`delay_s = 0.6 ns`, `lin = [-0.08, 0.03]`, `cub = [-0.05, 0.02]` when on). The chain: the PA's input
is `x + 1.104` (the data at 0.371 a0 RMS plus the pilot tone), its output minus its mean.

## 6. The measures

`dpd/metrics.py`, each against the undistorted drive `x`, each fitting a complex gain `c` and a
constant `k` by least squares (the tone's removal as a mean also takes the data's own mean):

```
NMSE    = 10 log10( sum|y - c x - k|^2 / sum|c x|^2 )                     the whole band
EVM_wf  = sqrt( mean|y_b - c x_b - k|^2 / mean|c x_b|^2 )                 in-band, |f| <= Rs (1 + 0.75) / 2
EVM_sym = sqrt( mean|z - c d - k|^2 / mean|c d|^2 )                       z: matched filter, symbol rate
```

`y_b`, `x_b`: an FFT mask to the band (as `main_pa_operating_point.m`). `z`: the RRC matched filter
(roll-off 0.75, span 16), one sample per symbol, 16 symbols dropped at each end; `d` the sent symbols.
