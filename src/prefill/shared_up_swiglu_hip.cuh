#pragma once
// Opt-in shared expert UP projection: unchanged BK32 accumulation, fused SwiGLU store.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
namespace shared_up_swiglu {
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
constexpr int BM = 128, BK = 32, LDK = BK + 8, GM = 8;
__device__ __forceinline__ h16 frag(const _Float16* p) {
    const uint4 a = *reinterpret_cast<const uint4*>(p), b = *reinterpret_cast<const uint4*>(p + 8);
    const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(h16, w);
}
template <int BN, int WN, bool Fused>
__global__ void __launch_bounds__(256) kernel(const _Float16* __restrict__ X, const _Float16* __restrict__ W,
                                              float* __restrict__ Y, int M, int N, int K, int ldy, int accumulate,
                                              int ldx, int ldw, const float* G, uint16_t* H) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int TN = WN / 16, NB = BN / 64;
    __shared__ __align__(16) _Float16 sA[2][BM][LDK];
    __shared__ __align__(16) _Float16 sB[2][BN][LDK];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wm = wave & 1, wn = wave >> 1;
    const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN, b = blockIdx.x;
    const int group = b / (GM * num_n), first_m = group * GM, gsize = min(GM, num_m - first_m);
    const int m0 = (first_m + (b % (GM * num_n)) % gsize) * BM, n0 = ((b % (GM * num_n)) / gsize) * BN;
    const int sr = tid >> 2, sq = tid & 3;
    auto ldA = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(X + (size_t) min(m0 + r, M - 1) * ldx + k0 + 8 * sq);
    };
    auto ldB = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(W + (size_t) min(n0 + r, N - 1) * ldw + k0 + 8 * sq);
    };
    f8 acc[4][TN];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < TN; ++j) acc[i][j] = f8{0, 0, 0, 0, 0, 0, 0, 0};
    uint4 ra0 = ldA(0, sr), ra1 = ldA(0, sr + 64), rb0 = ldB(0, sr), rb1 = ldB(0, sr + 64), rb2 = rb0, rb3 = rb0;
    if constexpr (NB > 2) { rb2 = ldB(0, sr + 128); rb3 = ldB(0, sr + 192); }
    int buf = 0;
    *reinterpret_cast<uint4*>(&sA[0][sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA[0][sr + 64][8 * sq]) = ra1;
    *reinterpret_cast<uint4*>(&sB[0][sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB[0][sr + 64][8 * sq]) = rb1;
    if constexpr (NB > 2) { *reinterpret_cast<uint4*>(&sB[0][sr + 128][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB[0][sr + 192][8 * sq]) = rb3; }
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;
        if (more) {
            ra0 = ldA(k0 + BK, sr); ra1 = ldA(k0 + BK, sr + 64); rb0 = ldB(k0 + BK, sr); rb1 = ldB(k0 + BK, sr + 64);
            if constexpr (NB > 2) { rb2 = ldB(k0 + BK, sr + 128); rb3 = ldB(k0 + BK, sr + 192); }
        }
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            h16 a[4], bb[TN];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = frag(&sA[buf][64 * wm + 16 * i + l16][ks]);
#pragma unroll
            for (int j = 0; j < TN; ++j) bb[j] = frag(&sB[buf][WN * wn + 16 * j + l16][ks]);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], bb[j], acc[i][j]);
        }
        if (more) {
            const int nb = buf ^ 1;
            *reinterpret_cast<uint4*>(&sA[nb][sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA[nb][sr + 64][8 * sq]) = ra1;
            *reinterpret_cast<uint4*>(&sB[nb][sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB[nb][sr + 64][8 * sq]) = rb1;
            if constexpr (NB > 2) { *reinterpret_cast<uint4*>(&sB[nb][sr + 128][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB[nb][sr + 192][8 * sq]) = rb3; }
            __syncthreads();
            buf = nb;
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
                    if constexpr (Fused) {
                        float up = acc[i][j][e];
                        __asm__ volatile("" : "+v"(up));
                        const float gate = G[(size_t)m*N+n];
                        const float v = gate / (1.0f + __expf(-gate)) * up;
                        H[(size_t)m*N+n] = __half_as_ushort(__float2half_rn(isnan(v) ? v : fminf(fmaxf(v,-65504.0f),65504.0f)));
                    } else {
                        float* y = Y + (size_t) m * ldy + n;
                        *y = accumulate ? *y + acc[i][j][e] : acc[i][j][e];
                    }
                }
            }
        }
#endif
}

}
