// HIP gfx1151 opt-in: identical FAST attention with original gate and coalesced FP16 stores.
// Included inside qsa_prompt_attn.cu's anonymous namespace after SmemW/original WMMA kernel.
template <bool SPLIT>
__global__ void __launch_bounds__(THREADS) prompt_attn_wmma_gated_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                   const int32_t* __restrict__ ids,
                                                                   const int32_t* __restrict__ steps, int n_kv_heads,
                                                                   int page_size, float scale_log2,
                                                                   float* __restrict__ attn, int cap, const float* __restrict__ qfull, uint16_t* __restrict__ out16, int ld16) {
#if STRATA_PA_WMMA
    __shared__ SmemW S;
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    qfull += (size_t) qi * n_head * HD * 2 + (size_t) kvh * G * HD * 2;
    out16 += (size_t) qi * ld16 + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int col = lane & 15, half = lane >> 4;   // fragment column / k half
    const int dim0 = warp * 64;

    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    // q A fragments of this wave's 64 dims: row = col (heads 12..15 zero), k = dim0 + kk*16 + PA_KOFF(half) + i
    wh8 qh[4], ql[4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        float x[PA_KL];
        if (col < G) {
            const float4* src = reinterpret_cast<const float4*>(q + (size_t) col * HD + dim0 + kk * 16 + PA_KOFF(half));
#pragma unroll
            for (int j = 0; j < PA_KL / 4; ++j) {
                const float4 a = src[j];
                x[4 * j] = a.x; x[4 * j + 1] = a.y; x[4 * j + 2] = a.z; x[4 * j + 3] = a.w;
            }
        } else {
#pragma unroll
            for (int i = 0; i < PA_KL; ++i) x[i] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < PA_KL; ++i) {
            const float v = x[i] * qup;
            const _Float16 hi = (_Float16) v;
            qh[kk][i] = hi;
            ql[kk][i] = (_Float16) (v - (float) hi);
        }
    }

    wf8 acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int n_chunks = (n + WCH - 1) / WCH;
    // chunk ci's cell for this lane (lane = cell index): its pool row, or -1 (past the selection, or a block the KV
    // streaming left non-resident: masked as in the decode kernel)
    auto row_of = [&](int ci) -> long long {
        const int c0 = ci * WCH;
        long long r = -1;
        if (c0 + lane < n) {
            const int cell = __ldg(ids + c0 + lane);
            const long long page = (long long) __ldg(p.page_table + cell / page_size);
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        return r;
    };
    // a chunk's global reads: this wave's V slice (64 bytes of each row) and V scale, and the K fragments (8 or 16
    // consecutive codes per lane) and K scales of its two 16-cell tiles
    struct Chunk {
        long long row;
        uint4 v[4];
        float vsc, ksc[2];
        uint32_t kx[2][4][PA_KL / 4];
    };
    auto fetch = [&](long long row, Chunk& d) {
        d.row = row;
        d.v[0] = d.v[1] = d.v[2] = d.v[3] = make_uint4(0, 0, 0, 0);
        d.vsc = 0.0f;
        if (row >= 0) {
            const uint4* src = reinterpret_cast<const uint4*>(p.v_q + row * HD + dim0);
            d.v[0] = __ldg(src); d.v[1] = __ldg(src + 1); d.v[2] = __ldg(src + 2); d.v[3] = __ldg(src + 3);
            d.vsc = __half2float(__ushort_as_half(__ldg(p.v_scale + row * (HD / KV_Q8_GROUP) + warp)));
        }
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const long long rr = __shfl(row, nt * 16 + col);
            d.ksc[nt] = 0.0f;
            if (rr >= 0) d.ksc[nt] = __half2float(__ushort_as_half(__ldg(p.k_scale + rr * (HD / KV_Q8_GROUP) + warp)));
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
                for (int i = 0; i < PA_KL / 4; ++i) d.kx[nt][kk][i] = 0;
                if (rr >= 0) {
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(p.k_q + rr * HD + dim0 + kk * 16 + PA_KOFF(half));
#if STRATA_PA_GFX11
                    const uint4 v = __ldg(reinterpret_cast<const uint4*>(src));
                    d.kx[nt][kk][0] = v.x; d.kx[nt][kk][1] = v.y; d.kx[nt][kk][2] = v.z; d.kx[nt][kk][3] = v.w;
#else
                    const uint2 v = __ldg(reinterpret_cast<const uint2*>(src));
                    d.kx[nt][kk][0] = v.x; d.kx[nt][kk][1] = v.y;
#endif
                }
            }
        }
    };
    auto chunk = [&](const Chunk& cd) {
        // stage this wave's V slice and scale
        {
            uint4* dst = reinterpret_cast<uint4*>(&S.v[warp][lane][0]);
            dst[0] = cd.v[0]; dst[1] = cd.v[1]; dst[2] = cd.v[2]; dst[3] = cd.v[3];
            S.vs[warp][lane] = cd.vsc;
            if (warp == 0) S.valid[lane] = cd.row >= 0;
        }
        // q.k over this wave's 64 dims for two 16-cell tiles, times the cell's K scale for this group
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            wf8 s = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const wh8 b = i8_to_h(cd.kx[nt][kk]);
                s = wmma_f16(qh[kk], b, s);
                if constexpr (SPLIT) s = wmma_f16(ql[kk], b, s);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) S.part[warp][PA_ROW(i, half)][nt * 16 + col] = s[i] * cd.ksc[nt];
        }
        __syncthreads();
        // online softmax over the four groups' sum (fixed order): row t/8, 4 cells per thread
        {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                // past the selection, or a masked cell (non-resident page, as in the decode kernel): no weight
                x[j] = S.valid[c] ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * qdown
                                  : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.p[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        // p.v over this wave's 64 dims: 4 tiles of 16 dims, 2 k-steps of 16 cells
        {
            float vmax = S.vs[warp][lane];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            wf8 tmp[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) tmp[j] = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int cb = ks * 16 + PA_KOFF(half);   // this lane's PA_KL cells
                wh8 ah, al;
#pragma unroll
                for (int i = 0; i < PA_KL; ++i) {
                    const float pv = S.p[col][cb + i] * (S.vs[warp][cb + i] * vup);
                    const _Float16 hi = (_Float16) pv;
                    ah[i] = hi;
                    al[i] = (_Float16) (pv - (float) hi);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int d = j * 16 + col;
                    wh8 b;
#pragma unroll
                    for (int i = 0; i < PA_KL; ++i) b[i] = (_Float16) (int) (int8_t) S.v[warp][cb + i][d];
                    tmp[j] = wmma_f16(ah, b, tmp[j]);
                    if constexpr (SPLIT) tmp[j] = wmma_f16(al, b, tmp[j]);
                }
            }
            float a[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = S.alpha[PA_ROW(i, half)];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[j][i] = fmaf(acc[j][i], a[i], tmp[j][i] * vdown);
        }
        // no barrier here: `v`/`vs` are this wave's own, and `p`/`alpha`/`valid` are rewritten only after the next
        // chunk's first barrier (which every wave reaches after its p.v)
    };
    // all of a chunk's reads are issued before its compute (they were interleaved with the q.k WMMAs and the V
    // staging): 32K context, 2,048 queries, gfx1151, STRATA_PA_FAST=1: 12.49 -> 11.27 ms; the same bits
    for (int ci = 0; ci < n_chunks; ++ci) {
        Chunk d;
        fetch(row_of(ci), d);
        chunk(d);
    }
    // Each wave's V staging bytes are dead after its last p.v computation.
    static_assert(sizeof(S.v[0]) >= 16 * 64 * sizeof(uint16_t));
    uint16_t* gate_tile = reinterpret_cast<uint16_t*>(&S.v[warp][0][0]);
    float inv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float l = S.lsum[PA_ROW(i, half)];
        inv[i] = l > 0.0f ? 1.0f / l : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = PA_ROW(i, half);
            if (row < G) {
                const int d = dim0 + j * 16 + col;
                float a = acc[j][i] * inv[i];
                // Match the separate FP32 output before sigmoid multiplication and FP16 narrowing.
                asm volatile("" : "+v"(a));
                const float gate = 1.0f / (1.0f + expf(-qfull[(size_t) row * HD * 2 + HD + d]));
                gate_tile[row * 64 + j * 16 + col] = __half_as_ushort(__float2half_rn(a * gate));
            }
        }
    __builtin_amdgcn_wave_barrier();
#pragma unroll
    for (int phase = 0; phase < 3; ++phase) {
        const int flat = (phase * 32 + lane) * 8;
        const int row = flat / 64, d = flat % 64;
        *reinterpret_cast<uint4*>(out16 + (size_t) row * HD + dim0 + d) =
            *reinterpret_cast<const uint4*>(gate_tile + flat);
    }
#else
    __builtin_trap();
#endif
}
