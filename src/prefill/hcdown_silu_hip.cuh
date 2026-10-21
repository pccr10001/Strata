#pragma once
// Original HC down/inject WMMA order, with the existing BF16 SiLU epilogue.
__device__ __forceinline__ uint16_t hcdown_silu_bf(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}
template<bool TailGuard = false, int GroupM = GM>
__global__ void __launch_bounds__(256) kernel_hcdown_silu(const uint16_t* __restrict__ X, int ldx,
                                                     const uint16_t* __restrict__ Wd, const uint16_t* __restrict__ Wi,
                                                     int nd, int ni, float* __restrict__ lo, float* __restrict__ inj,
                                                     int M, int K, uint16_t* lo16) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int BN = 128, WN = 32, TN = 2, BK64 = 64, LDK64 = BK64 + 8;
    __shared__ __align__(16) uint16_t sA[BM][LDK64];
    __shared__ __align__(16) uint16_t sB[BN][LDK64];
    const int N = nd + ni;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wm = wave & 1, wn = wave >> 1;
    const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN, b = blockIdx.x;
    const int group = b / (GroupM * num_n), first_m = group * GroupM, gsize = min(GroupM, num_m - first_m);
    const int m0 = (first_m + (b % (GroupM * num_n)) % gsize) * BM, n0 = ((b % (GroupM * num_n)) / gsize) * BN;
    const int sr = tid >> 3, sq = tid & 7;
    auto ldA = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(X + (size_t) min(m0 + r, M - 1) * ldx + k0 + 8 * sq);
    };
    auto ldB = [&](int k0, int r) -> uint4 {
        const int n = min(n0 + r, N - 1);
        const uint16_t* row = n < nd ? Wd + (size_t) n * K : Wi + (size_t) (n - nd) * K;
        return *reinterpret_cast<const uint4*>(row + k0 + 8 * sq);
    };
    uint4 ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3;
    auto load = [&](int k0) {
        ra0 = ldA(k0, sr); ra1 = ldA(k0, sr + 32); ra2 = ldA(k0, sr + 64); ra3 = ldA(k0, sr + 96);
        rb0 = ldB(k0, sr); rb1 = ldB(k0, sr + 32); rb2 = ldB(k0, sr + 64); rb3 = ldB(k0, sr + 96);
    };
    auto store = [&]() {
        *reinterpret_cast<uint4*>(&sA[sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA[sr + 32][8 * sq]) = ra1;
        *reinterpret_cast<uint4*>(&sA[sr + 64][8 * sq]) = ra2; *reinterpret_cast<uint4*>(&sA[sr + 96][8 * sq]) = ra3;
        *reinterpret_cast<uint4*>(&sB[sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB[sr + 32][8 * sq]) = rb1;
        *reinterpret_cast<uint4*>(&sB[sr + 64][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB[sr + 96][8 * sq]) = rb3;
    };
    f8 acc[4][TN];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = f8{0, 0, 0, 0, 0, 0, 0, 0};
    load(0);
    store();
    __syncthreads();
    const int ar = 64 * wm + l16, br = WN * wn + l16;
    for (int k0 = 0; k0 < K; k0 += BK64) {
        const bool more = k0 + BK64 < K;
        if (more) load(k0 + BK64);
#pragma unroll
        for (int ks = 0; ks < BK64; ks += 16) {
            // Every lane in a wave owns the same output tile; barriers stay outside.
            if (!TailGuard || n0 + WN * wn < N) {
                const hb16 a0 = bfrag(&sA[ar][ks]), a1 = bfrag(&sA[ar + 16][ks]),
                           a2 = bfrag(&sA[ar + 32][ks]), a3 = bfrag(&sA[ar + 48][ks]);
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    if (!TailGuard || n0 + WN * wn + 16 * j < N) {
                        const hb16 b = bfrag(&sB[br + 16 * j][ks]);
                        acc[0][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, b, acc[0][j]);
                        acc[1][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, b, acc[1][j]);
                        acc[2][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a2, b, acc[2][j]);
                        acc[3][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a3, b, acc[3][j]);
                    }
                }
            }
        }
        if (more) {
            __syncthreads();
            store();
            __syncthreads();
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n = n0 + WN * wn + 16 * j + l16;
            if (n >= N) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = m0 + 64 * wm + 16 * i + 2 * e + hi;
                if (m < M) {
                    if (n < nd) {
                        const float x = acc[i][j][e] / 4.0f;
                        const float v = x / (1.0f + __expf(-x));
                        lo16[(size_t) m * nd + n] = hcdown_silu_bf(v);
                    }
                    else inj[(size_t) m * ni + (n - nd)] = acc[i][j][e];
                }
            }
        }
#endif
}
