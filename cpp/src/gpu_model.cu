// The transformer DPD as one fused CUDA kernel (docs/twin/README.md, docs/dpd/METHODOLOGY.md).
//
// One thread block per output sample, 4 warps. The model width is 32, a warp's width: a warp
// works on one token at a time, lane i holding feature i, so the layer norms and the softmax
// reduce with warp shuffles. The window and every intermediate stay in shared memory; only the
// sample's I/Q goes back to global memory. The weights are shared by all blocks (73 kB, cached);
// the matrices are stored transposed, [in][out], so the 32 lanes read 32 neighbouring words.
//
// Per layer, two phases separated by __syncthreads(): (1) layer norm and Q/K/V of every token;
// (2) per token: attention over all tokens' K and V, the output projection and its residual, then
// the feed-forward block and its residual (which needs only the token itself). Each warp owns 8
// tokens and runs every matrix product on the 8 together, so each weight is loaded once for 8.
// In the last layer only the centre token is computed past Q/K/V (the output reads only it).
#include "dpd/gpu_model.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "cuda_util.h"

namespace dpd {

namespace {

constexpr int T = 32;   // window
constexpr int D = 32;   // width (= warp size)
constexpr int H = 4;    // heads
constexpr int DH = 8;   // per head
constexpr int F = 64;   // feed-forward
constexpr int L = 2;    // layers
constexpr int C = 16;   // centre
constexpr int WARPS = 4;
constexpr int THREADS = 32 * WARPS;
constexpr int TPW = T / WARPS;  // tokens per warp: warp w owns tokens w, w + 4, ..., w + 28
constexpr unsigned FULL = 0xffffffffu;
// one probe's trace: embed, each layer [T][D], final_norm [D], out [2] (the golden file's layout)
constexpr int TRACE_FLOATS = (L + 1) * T * D + D + 2;

static_assert(D == 32, "the kernel maps the model width to the warp");
static_assert(H * DH == D, "heads x per-head width = width");

using cuda::check;
using cuda::device_alloc;

}  // namespace

// the kernel's arguments: pointers into the weights buffer (matrices transposed, [in][out])
namespace detail {
struct LayerParams {
    const float *n1w, *n1b, *inT, *inb, *outT, *outb, *n2w, *n2b, *ff1T, *ff1b, *ff2T, *ff2b;
};
struct KernelParams {
    float scale;
    const float *pos, *embT, *embb;
    LayerParams layer[L];
    const float *nw, *nb, *headw, *headb;
};
}  // namespace detail

namespace {

using detail::KernelParams;
using detail::LayerParams;

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
    return v;
}
// layer norm of one token, one feature per lane: biased variance, eps 1e-5
__device__ __forceinline__ float layer_norm(float v, float w, float b) {
    const float mu = warp_sum(v) * (1.0f / D);
    const float d = v - mu;
    const float var = warp_sum(d * d) * (1.0f / D);
    return d / sqrtf(var + 1e-5f) * w + b;
}
// x[i .. i+3] in one shared-memory load (x 16-byte aligned, i a multiple of 4)
__device__ __forceinline__ float4 load4(const float* x, int i) { return *reinterpret_cast<const float4*>(x + i); }

__device__ __forceinline__ float gelu(float u) { return 0.5f * u * (1.0f + erff(u * 0.70710678118654752f)); }

__global__ void __launch_bounds__(THREADS) dpd_forward(KernelParams m, const float2* __restrict__ x,
                                                        float2* __restrict__ y, int n, const int* __restrict__ probes,
                                                        int n_probes, float* __restrict__ trace) {
    __shared__ float z[T][2];        // the window, scaled
    __shared__ float e[T][D];                   // the token stream (residual)
    __shared__ __align__(16) float a[T][D];     // a norm's output, then the attention's (read as float4)
    // q | k | v of each token; one float of padding per row, so that the 32 lanes reading one
    // column each (the attention scores read k of token u = lane) fall in 32 different banks
    __shared__ float qkv[T][3 * D + 1];
    __shared__ float p[WARPS][H][T];      // the softmax numerators of a warp's current token
    __shared__ __align__(16) float hid[WARPS][TPW][F];  // the feed-forward hidden layer of a warp's tokens (float4)

    const int s = blockIdx.x;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const float att_scale = 1.0f / sqrtf(static_cast<float>(DH));

    // is this sample probed? (the trace path only; trace is null in normal runs)
    float* tr = nullptr;
    if (trace)
        for (int i = 0; i < n_probes; ++i)
            if (probes[i] == s) tr = trace + static_cast<size_t>(i) * TRACE_FLOATS;

    // the window, zero-padded beyond the frame, divided by the scale
    if (threadIdx.x < T) {
        const int k = s - C + threadIdx.x;
        const float2 v = (k >= 0 && k < n) ? x[k] : make_float2(0.0f, 0.0f);
        z[threadIdx.x][0] = v.x / m.scale;
        z[threadIdx.x][1] = v.y / m.scale;
    }
    __syncthreads();

    // embedding and position
    for (int t = warp; t < T; t += WARPS) {
        float acc = m.embb[lane];
        acc += m.embT[lane] * z[t][0];
        acc += m.embT[D + lane] * z[t][1];
        e[t][lane] = acc + m.pos[t * D + lane];
    }
    __syncthreads();
    if (tr)
        for (int i = threadIdx.x; i < T * D; i += THREADS) tr[i] = (&e[0][0])[i];

    // Each warp works on its own TPW = 8 tokens (t = warp + WARPS * k) together: every weight is
    // loaded once and applied to all of them, with one accumulator per token in registers. The
    // sums are those of the per-token loop, in the same order.
    for (int l = 0; l < L; ++l) {
        const LayerParams& P = m.layer[l];
        // the output reads only the centre token: in the last layer, only it needs attention and
        // the feed-forward block (every token still needs its K and V). The trace keeps all.
        const bool centre_only = (l == L - 1) && !trace;
        bool on[TPW];  // the warp's tokens that phase 2 computes
#pragma unroll
        for (int k = 0; k < TPW; ++k) on[k] = !centre_only || warp + WARPS * k == C;

        // phase 1: norm 1, then Q/K/V of the warp's tokens (all of them: attention needs every K, V)
        for (int k = 0; k < TPW; ++k) {
            const int t = warp + WARPS * k;
            a[t][lane] = layer_norm(e[t][lane], P.n1w[lane], P.n1b[lane]);
        }
        __syncwarp();
        {
            float q[TPW], kk[TPW], v[TPW];
#pragma unroll
            for (int k = 0; k < TPW; ++k) {
                q[k] = P.inb[lane];
                kk[k] = P.inb[D + lane];
                v[k] = P.inb[2 * D + lane];
            }
            for (int i = 0; i < D; i += 4) {
                float wq[4], wk[4], wv[4];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const float* w = P.inT + (i + c) * 3 * D;
                    wq[c] = w[lane];
                    wk[c] = w[D + lane];
                    wv[c] = w[2 * D + lane];
                }
#pragma unroll
                for (int k = 0; k < TPW; ++k) {
                    const float4 a4 = load4(a[warp + WARPS * k], i);
                    const float ai[4] = {a4.x, a4.y, a4.z, a4.w};
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        q[k] += wq[c] * ai[c];
                        kk[k] += wk[c] * ai[c];
                        v[k] += wv[c] * ai[c];
                    }
                }
            }
#pragma unroll
            for (int k = 0; k < TPW; ++k) {
                const int t = warp + WARPS * k;
                qkv[t][lane] = q[k];
                qkv[t][D + lane] = kk[k];
                qkv[t][2 * D + lane] = v[k];
            }
        }
        __syncthreads();  // phase 2 reads every token's K and V

        // phase 2a: attention of each of the warp's tokens (lane = u for the scores), into a[t]
        for (int k = 0; k < TPW; ++k) {
            if (!on[k]) continue;
            const int t = warp + WARPS * k;
            float sums[H];
            for (int j = 0; j < H; ++j) {
                float sc = 0.0f;
                for (int i = 0; i < DH; ++i) sc += qkv[t][j * DH + i] * qkv[lane][D + j * DH + i];
                sc *= att_scale;
                const float ex = expf(sc - warp_max(sc));
                sums[j] = warp_sum(ex);
                p[warp][j][lane] = ex;
            }
            __syncwarp();
            // weighted sum of the values: lane c = 8j + i, head j
            const int j = lane / DH;
            float acc = 0.0f;
            for (int u = 0; u < T; ++u) acc += p[warp][j][u] * qkv[u][2 * D + lane];
            __syncwarp();  // p is rewritten for the next token
            a[t][lane] = acc / sums[j];
        }
        __syncwarp();

        // phase 2b: output projection and residual
        {
            float r[TPW];
#pragma unroll
            for (int k = 0; k < TPW; ++k) r[k] = P.outb[lane];
            for (int i = 0; i < D; i += 4) {
                float w[4];
#pragma unroll
                for (int c = 0; c < 4; ++c) w[c] = P.outT[(i + c) * D + lane];
#pragma unroll
                for (int k = 0; k < TPW; ++k)
                    if (on[k]) {
                        const float4 a4 = load4(a[warp + WARPS * k], i);
                        r[k] += w[0] * a4.x;
                        r[k] += w[1] * a4.y;
                        r[k] += w[2] * a4.z;
                        r[k] += w[3] * a4.w;
                    }
            }
            __syncwarp();  // every lane has read a before norm 2 rewrites it
#pragma unroll
            for (int k = 0; k < TPW; ++k)
                if (on[k]) {
                    const int t = warp + WARPS * k;
                    e[t][lane] += r[k];
                    a[t][lane] = layer_norm(e[t][lane], P.n2w[lane], P.n2b[lane]);  // phase 2c: norm 2
                }
        }
        __syncwarp();

        // phase 2d: feed-forward, first layer with GELU (lane computes hidden units lane, lane + 32)
        {
            float h0[TPW], h1[TPW];
#pragma unroll
            for (int k = 0; k < TPW; ++k) {
                h0[k] = P.ff1b[lane];
                h1[k] = P.ff1b[lane + 32];
            }
            for (int i = 0; i < D; i += 4) {
                float w0[4], w1[4];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    w0[c] = P.ff1T[(i + c) * F + lane];
                    w1[c] = P.ff1T[(i + c) * F + lane + 32];
                }
#pragma unroll
                for (int k = 0; k < TPW; ++k)
                    if (on[k]) {
                        const float4 g4 = load4(a[warp + WARPS * k], i);
                        const float gi[4] = {g4.x, g4.y, g4.z, g4.w};
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            h0[k] += w0[c] * gi[c];
                            h1[k] += w1[c] * gi[c];
                        }
                    }
            }
#pragma unroll
            for (int k = 0; k < TPW; ++k)
                if (on[k]) {
                    hid[warp][k][lane] = gelu(h0[k]);
                    hid[warp][k][lane + 32] = gelu(h1[k]);
                }
        }
        __syncwarp();

        // phase 2e: feed-forward, second layer, and residual
        {
            float r[TPW];
#pragma unroll
            for (int k = 0; k < TPW; ++k) r[k] = P.ff2b[lane];
            for (int i = 0; i < F; i += 4) {
                float w[4];
#pragma unroll
                for (int c = 0; c < 4; ++c) w[c] = P.ff2T[(i + c) * D + lane];
#pragma unroll
                for (int k = 0; k < TPW; ++k)
                    if (on[k]) {
                        const float4 h4 = load4(hid[warp][k], i);
                        r[k] += w[0] * h4.x;
                        r[k] += w[1] * h4.y;
                        r[k] += w[2] * h4.z;
                        r[k] += w[3] * h4.w;
                    }
            }
#pragma unroll
            for (int k = 0; k < TPW; ++k)
                if (on[k]) e[warp + WARPS * k][lane] += r[k];
        }
        __syncthreads();  // the next layer's norm reads e; its Q/K/V overwrite qkv
        if (tr)
            for (int i = threadIdx.x; i < T * D; i += THREADS) tr[(l + 1) * T * D + i] = (&e[0][0])[i];
    }

    // the centre token: final norm, head, the skip, the scale
    if (warp == 0) {
        const float v = layer_norm(e[C][lane], m.nw[lane], m.nb[lane]);
        const float y0 = warp_sum(m.headw[lane] * v);
        const float y1 = warp_sum(m.headw[D + lane] * v);
        if (lane == 0) {
            const float o0 = m.headb[0] + y0 + z[C][0];
            const float o1 = m.headb[1] + y1 + z[C][1];
            y[s] = make_float2(o0 * m.scale, o1 * m.scale);
            if (tr) {
                tr[(L + 1) * T * D + D] = o0;
                tr[(L + 1) * T * D + D + 1] = o1;
            }
        }
        if (tr) tr[(L + 1) * T * D + lane] = v;
    }
}

// [rows][cols] -> [cols][rows] (the kernel reads matrices as [in][out])
std::vector<float> transpose(const std::vector<float>& a, int rows, int cols) {
    std::vector<float> t(a.size());
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) t[static_cast<size_t>(c) * rows + r] = a[static_cast<size_t>(r) * cols + c];
    return t;
}

}  // namespace

std::size_t GpuModel::trace_floats_per_probe() const { return TRACE_FLOATS; }

GpuModel::GpuModel(const Weights& w) {
    const Config& c = w.cfg;
    if (c.window != T || c.d_model != D || c.n_heads != H || c.n_layers != L || c.d_ff != F)
        throw std::runtime_error("GpuModel: the kernel is built for window 32, width 32, 4 heads, 2 layers, FF 64; got " +
                                 std::to_string(c.window) + ", " + std::to_string(c.d_model) + ", " +
                                 std::to_string(c.n_heads) + ", " + std::to_string(c.n_layers) + ", " +
                                 std::to_string(c.d_ff));

    // pack every tensor into one buffer (matrices transposed), remembering the offsets
    std::vector<float> buf;
    std::vector<size_t> off;
    auto put = [&](const std::vector<float>& v) {
        off.push_back(buf.size());
        buf.insert(buf.end(), v.begin(), v.end());
    };
    put(w.pos);
    put(transpose(w.embed_w, D, 2));
    put(w.embed_b);
    for (const auto& l : w.layers) {
        put(l.norm1_w);
        put(l.norm1_b);
        put(transpose(l.in_w, 3 * D, D));
        put(l.in_b);
        put(transpose(l.out_w, D, D));
        put(l.out_b);
        put(l.norm2_w);
        put(l.norm2_b);
        put(transpose(l.ff1_w, F, D));
        put(l.ff1_b);
        put(transpose(l.ff2_w, D, F));
        put(l.ff2_b);
    }
    put(w.norm_w);
    put(w.norm_b);
    put(w.head_w);
    put(w.head_b);

    weights_ = device_alloc<float>(buf.size(), "cudaMalloc weights");
    check(cudaMemcpy(weights_.get(), buf.data(), buf.size() * sizeof(float), cudaMemcpyHostToDevice), "upload weights");

    params_ = std::make_unique<KernelParams>();
    KernelParams& m = *params_;
    std::size_t k = 0;
    auto at = [&]() { return weights_.get() + off[k++]; };
    m.scale = w.scale;
    m.pos = at();
    m.embT = at();
    m.embb = at();
    for (int i = 0; i < L; ++i) {
        LayerParams& p = m.layer[i];
        p.n1w = at(); p.n1b = at(); p.inT = at(); p.inb = at(); p.outT = at(); p.outb = at();
        p.n2w = at(); p.n2b = at(); p.ff1T = at(); p.ff1b = at(); p.ff2T = at(); p.ff2b = at();
    }
    m.nw = at();
    m.nb = at();
    m.headw = at();
    m.headb = at();
}

GpuModel::~GpuModel() = default;

void GpuModel::forward(const float* d_x, float* d_y, int n, void* stream) const {
    if (n <= 0) return;
    dpd_forward<<<n, THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        *params_, reinterpret_cast<const float2*>(d_x), reinterpret_cast<float2*>(d_y), n, nullptr, 0, nullptr);
    check(cudaGetLastError(), "dpd_forward launch");
}

void GpuModel::ensure_io(int n) const {
    if (n <= io_n_) return;
    io_n_ = 0;
    x_.reset();
    y_.reset();
    x_ = device_alloc<float>(2 * static_cast<std::size_t>(n), "cudaMalloc x");
    y_ = device_alloc<float>(2 * static_cast<std::size_t>(n), "cudaMalloc y");
    io_n_ = n;
}

void GpuModel::predistort(const float* x, float* y, int n) const {
    if (n <= 0) return;
    ensure_io(n);
    check(cudaMemcpy(x_.get(), x, 2 * sizeof(float) * n, cudaMemcpyHostToDevice), "copy x");
    forward(x_.get(), y_.get(), n);
    check(cudaDeviceSynchronize(), "dpd_forward");  // a fault inside the kernel is reported here
    check(cudaMemcpy(y, y_.get(), 2 * sizeof(float) * n, cudaMemcpyDeviceToHost), "copy y");
}

std::vector<float> GpuModel::trace(const float* x, int n, const std::vector<int>& probes) const {
    for (int p : probes)
        if (p < 0 || p >= n)
            throw std::runtime_error("trace: probe " + std::to_string(p) + " outside the frame of " + std::to_string(n));
    if (probes.empty()) return {};
    ensure_io(n);
    const std::size_t per = trace_floats_per_probe();
    auto d_probes = device_alloc<int>(probes.size(), "cudaMalloc probes");
    auto d_trace = device_alloc<float>(probes.size() * per, "cudaMalloc trace");
    check(cudaMemcpy(d_probes.get(), probes.data(), probes.size() * sizeof(int), cudaMemcpyHostToDevice), "copy probes");
    check(cudaMemcpy(x_.get(), x, 2 * sizeof(float) * n, cudaMemcpyHostToDevice), "copy x");
    dpd_forward<<<n, THREADS>>>(*params_, reinterpret_cast<const float2*>(x_.get()), reinterpret_cast<float2*>(y_.get()),
                                n, d_probes.get(), static_cast<int>(probes.size()), d_trace.get());
    check(cudaGetLastError(), "dpd_forward (trace) launch");
    check(cudaDeviceSynchronize(), "dpd_forward (trace)");
    std::vector<float> out(probes.size() * per);
    check(cudaMemcpy(out.data(), d_trace.get(), out.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy trace");
    return out;
}

}  // namespace dpd
