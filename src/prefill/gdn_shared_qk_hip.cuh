// Opt-in gfx11 prep: exact FP32 KK/QK shared across the three value-head waves.
__global__ void __launch_bounds__(32 * GW_VPK) gdn_wmma_prep_shared_one_kernel(const float* __restrict__ h, const float* __restrict__ gate,
                                                                 const float* __restrict__ beta, _Float16* __restrict__ tp,
                                                                 int64_t T) {
    __shared__ float sa[GW_VPK][GW_CK * GW_CK], sg[GW_VPK][GW_CK], sb[GW_VPK][GW_CK];
    __shared__ float common_kk[GW_CK * GW_CK], common_qk[GW_CK * GW_CK];
    const int qh = blockIdx.x, ch = blockIdx.y, w = threadIdx.x >> 5, l = threadIdx.x & 31, c = l & 15, hi = l >> 4;
    const int vh = qh + w * HK;
    const int64_t t0 = (int64_t) ch * GW_CK;
    gw_f8 kk = {}, qk = {};   // k_t.k_i and q_t.k_i (lane: column i = c, rows t = 2v + hi)
    if (w == 0) {
        const bool ok = t0 + c < T;   // past the end: zero rows
        const float* ht = h + (ok ? t0 + c : 0) * C;
#pragma unroll
        for (int j = 0; j < S / 16; ++j) {
            gw_h16 ka, qa;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const gw_f4 k4 = *reinterpret_cast<const gw_f4*>(ht + HK * S + qh * S + 16 * j + 4 * i);
                gw_f4 q4 = {};
                if (w == 0)
                    q4 = *reinterpret_cast<const gw_f4*>(ht + qh * S + 16 * j + 4 * i);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    ka[4 * i + e] = (_Float16) (ok ? k4[e] : 0.0f);
                    qa[4 * i + e] = (_Float16) (ok ? q4[e] : 0.0f);
                }
            }
            if (w == 0) kk = gw_mma(ka, ka, kk);
            if (w == 0) qk = gw_mma(qa, ka, qk);
        }
    }
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const int at = (2 * v + hi) * GW_CK + c;
        if (w == 0) common_kk[at] = kk[v];
        if (w == 0) common_qk[at] = qk[v];
    }
    if (l < GW_CK) {   // past the end: gate 0, beta 0
        const bool ok = t0 + l < T;
        float g = ok ? gate[(t0 + l) * HV + vh] : 0.0f;
#pragma unroll
        for (int o = 1; o < GW_CK; o <<= 1) {
            const float u = __shfl_up(g, o, GW_CK);
            if (l >= o) g += u;
        }
        sg[w][l] = g;
        sb[w][l] = ok ? beta[(t0 + l) * HV + vh] : 0.0f;
    }
    gw_lds_barrier();   // producer matrices and each wave's gate/beta are ready
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const int at = (2 * v + hi) * GW_CK + c;
        kk[v] = common_kk[at];
        qk[v] = common_qk[at];
    }
    _Float16* out = tp + ((int64_t) ch * HV + vh) * (2 * GW_CK * GW_CK);
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const int t = 2 * v + hi;
        sa[w][t * GW_CK + c] = c < t ? sb[w][t] * __expf(sg[w][t] - sg[w][c]) * kk[v] : 0.0f;
        out[GW_CK * GW_CK + t * GW_CK + c] = (_Float16) (c <= t ? __expf(sg[w][t] - sg[w][c]) * qk[v] : 0.0f);
    }
    gw_lds_wait();
    if (l < GW_CK) {   // lane c: column c of T = (I + A)^-1 by forward substitution
        float tc[GW_CK];
#pragma unroll
        for (int t = 0; t < GW_CK; ++t) {
            float v = t == c ? 1.0f : 0.0f;
#pragma unroll
            for (int k = 0; k < t; ++k) v = fmaf(-sa[w][t * GW_CK + k], tc[k], v);
            tc[t] = v;
            out[t * GW_CK + c] = (_Float16) v;
        }
    }
}
