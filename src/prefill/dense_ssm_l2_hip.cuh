// Opt-in dense projection + convolution + Q/K L2, gfx11.
namespace ssm_l2 {
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
constexpr int BM=128, GM=8;
__device__ __forceinline__ h16 frag(const _Float16* p) {
    const uint4 a=*reinterpret_cast<const uint4*>(p),b=*reinterpret_cast<const uint4*>(p+8);
    const uint32_t w[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
    return __builtin_bit_cast(h16,w);
}
__device__ __forceinline__ float conv4(float4 w,float a,float b,float c,float d) {
    const float s=__fmaf_rn(d,w.w,__fmaf_rn(c,w.z,__fmaf_rn(a,w.x,__fmul_rn(b,w.y))));
    return s/(1.0f+__expf(-s));
}
__global__ void __launch_bounds__(256) kernel_fused(const _Float16* __restrict__ X, const _Float16* __restrict__ W,
                                                   float* __restrict__ Y, int M, int N, int K, int ldy, int accumulate,
                                                   int ldx, int ldw, const float* __restrict__ conv_w, float* __restrict__ h, float eps) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int BN = 256, WN = 64, BK64 = 64, LDK64 = BK64 + 8;
    __shared__ __align__(16) _Float16 sA[BM][LDK64];
    __shared__ __align__(16) _Float16 sB[BN][LDK64];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wm = wave & 1, wn = wave >> 1;
    const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN, b = blockIdx.x;
    const int group = b / (GM * num_n), first_m = group * GM, gsize = min(GM, num_m - first_m);
    const int m0 = (first_m + (b % (GM * num_n)) % gsize) * BM, n0 = ((b % (GM * num_n)) / gsize) * BN;
    const int sr = tid >> 3, sq = tid & 7;            // 32 rows per pass, 8 x 8 halves per row
    auto ldA = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(X + (size_t) min(m0 + r, M - 1) * ldx + k0 + 8 * sq);
    };
    auto ldB = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(W + (size_t) min(n0 + r, N - 1) * ldw + k0 + 8 * sq);
    };
    uint4 ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7;
    auto load = [&](int k0) {
        ra0 = ldA(k0, sr); ra1 = ldA(k0, sr + 32); ra2 = ldA(k0, sr + 64); ra3 = ldA(k0, sr + 96);
        rb0 = ldB(k0, sr); rb1 = ldB(k0, sr + 32); rb2 = ldB(k0, sr + 64); rb3 = ldB(k0, sr + 96);
        rb4 = ldB(k0, sr + 128); rb5 = ldB(k0, sr + 160); rb6 = ldB(k0, sr + 192); rb7 = ldB(k0, sr + 224);
    };
    auto store = [&]() {
        *reinterpret_cast<uint4*>(&sA[sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA[sr + 32][8 * sq]) = ra1;
        *reinterpret_cast<uint4*>(&sA[sr + 64][8 * sq]) = ra2; *reinterpret_cast<uint4*>(&sA[sr + 96][8 * sq]) = ra3;
        *reinterpret_cast<uint4*>(&sB[sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB[sr + 32][8 * sq]) = rb1;
        *reinterpret_cast<uint4*>(&sB[sr + 64][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB[sr + 96][8 * sq]) = rb3;
        *reinterpret_cast<uint4*>(&sB[sr + 128][8 * sq]) = rb4; *reinterpret_cast<uint4*>(&sB[sr + 160][8 * sq]) = rb5;
        *reinterpret_cast<uint4*>(&sB[sr + 192][8 * sq]) = rb6; *reinterpret_cast<uint4*>(&sB[sr + 224][8 * sq]) = rb7;
    };
    f8 acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = f8{0, 0, 0, 0, 0, 0, 0, 0};
    load(0);
    store();
    __syncthreads();
    const int ar = 64 * wm + l16, br = WN * wn + l16;
    for (int k0 = 0; k0 < K; k0 += BK64) {
        const bool more = k0 + BK64 < K;
        if (more) load(k0 + BK64);
#pragma unroll
        for (int ks = 0; ks < BK64; ks += 16) {
            const h16 a0 = frag(&sA[ar][ks]), a1 = frag(&sA[ar + 16][ks]), a2 = frag(&sA[ar + 32][ks]),
                      a3 = frag(&sA[ar + 48][ks]);
            const h16 b0 = frag(&sB[br][ks]), b1 = frag(&sB[br + 16][ks]), b2 = frag(&sB[br + 32][ks]),
                      b3 = frag(&sB[br + 48][ks]);
#define PFG_ROW(i, a)                                                                                                 \
    acc[i][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b0, acc[i][0]);                                         \
    acc[i][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b1, acc[i][1]);                                         \
    acc[i][2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b2, acc[i][2]);                                         \
    acc[i][3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b3, acc[i][3]);
            PFG_ROW(0, a0) PFG_ROW(1, a1) PFG_ROW(2, a2) PFG_ROW(3, a3)
#undef PFG_ROW
        }
        if (more) {
            __syncthreads();   // every wave is done reading the tile
            store();
            __syncthreads();
        }
    }

    // All GEMM readers finish before the weight LDS becomes a 32-token transpose.
    static_assert(sizeof(sB) >= 8*32*36*sizeof(float));
    __syncthreads();
    if(n0<4096) {
        // A whole 128-channel head must be present before its L2 reduction.
        // Reuse weight LDS for one token half at a time, with bank padding.
        constexpr int stride=260;
        static_assert(sizeof(sB)>=32*stride*sizeof(float));
        float* raw=reinterpret_cast<float*>(sB);
#pragma unroll
        for(int selected_wm=0;selected_wm<2;++selected_wm) {
#pragma unroll
            for(int ip=0;ip<2;++ip) {
                if(wm==selected_wm) {
#pragma unroll
                    for(int g=0;g<2;++g)
#pragma unroll
                        for(int j=0;j<4;++j)
#pragma unroll
                            for(int e=0;e<8;++e)
                                raw[(16*g+2*e+hi)*stride+wn*64+j*16+l16]=acc[ip*2+g][j][e];
                }
                __syncthreads();
                // Preserve just the rows needed by the separate boundary fixup/history.
#pragma unroll
                for(int pass=0;pass<32;++pass) {
                    const int flat=pass*256+tid;
                    const int tl=flat/256,c=flat%256;
                    const int m=m0+64*selected_wm+32*ip+tl;
                    if(m<M && (tl<3 || tl>=29 || m+3>=M))
                        Y[size_t(m)*ldy+n0+c]=raw[tl*stride+c];
                }
                // One wave owns a complete head. Each lane carries one channel
                // from each of the original four 32-channel reduction groups.
#pragma unroll
                for(int round=0;round<8;++round) {
                    const int tl=3+round*4+(wave>>1);
                    const int m=m0+64*selected_wm+32*ip+tl;
                    if(tl<32 && m<M) {
                        float value[4],sum=0.0f;
#pragma unroll
                        for(int g=0;g<4;++g) {
                            const int c=(wave&1)*128+g*32+lane;
                            const float4 w=*reinterpret_cast<const float4*>(conv_w+(n0+c)*4);
                            const float v=conv4(w,raw[(tl-3)*stride+c],raw[(tl-2)*stride+c],
                                                raw[(tl-1)*stride+c],raw[tl*stride+c]);
                            value[g]=v;
                            float sq=__fmaf_rn(v,v,__shfl_xor(__fmul_rn(v,v),16));
#pragma unroll
                            for(int o=8;o>0;o>>=1)sq=__fadd_rn(sq,__shfl_xor(sq,o));
                            // The original CTA norm stores only lane0's partial.
                            sq=__shfl(sq,0);
                            if(g==0)sum=sq;else sum=__fadd_rn(sum,sq);
                        }
                        const float inverse=rsqrtf(sum+eps);
#pragma unroll
                        for(int g=0;g<4;++g) {
                            const int c=(wave&1)*128+g*32+lane;
                            h[size_t(m)*N+n0+c]=value[g]*inverse;
                        }
                    }
                }
                __syncthreads();
            }
        }
        return;
    }
    float* ep = reinterpret_cast<float*>(sB) + wave*32*36;
#pragma unroll
    for(int ip=0;ip<2;++ip) {
#pragma unroll
        for(int jp=0;jp<2;++jp) {
#pragma unroll
            for(int g=0;g<2;++g)
#pragma unroll
                for(int q=0;q<2;++q)
#pragma unroll
                    for(int e=0;e<8;++e)
                        ep[(16*g+2*e+hi)*36+q*16+l16] = acc[ip*2+g][jp*2+q][e];
            __builtin_amdgcn_wave_barrier();
#pragma unroll
            for(int phase=0;phase<8;++phase) {
                const int flat=(phase*32+lane)*4;
                const int tl=flat/32;
                const int m=m0+64*wm+32*ip+tl;
                const int n=n0+WN*wn+32*jp+flat%32;
                if(m<M && n+4<=N) {
                    const float4 v=*reinterpret_cast<const float4*>(ep+tl*36+flat%32);
                    if(tl<3 || tl>=29 || m+3>=M)
                        *reinterpret_cast<float4*>(Y+size_t(m)*ldy+n)=v;
                    if(tl>=3) {
                        const float4 a=*reinterpret_cast<const float4*>(ep+(tl-3)*36+flat%32);
                        const float4 b=*reinterpret_cast<const float4*>(ep+(tl-2)*36+flat%32);
                        const float4 c=*reinterpret_cast<const float4*>(ep+(tl-1)*36+flat%32);
                        const float4 w0=*reinterpret_cast<const float4*>(conv_w+(n+0)*4);
                        const float4 w1=*reinterpret_cast<const float4*>(conv_w+(n+1)*4);
                        const float4 w2=*reinterpret_cast<const float4*>(conv_w+(n+2)*4);
                        const float4 w3=*reinterpret_cast<const float4*>(conv_w+(n+3)*4);
                        *reinterpret_cast<float4*>(h+size_t(m)*N+n)=make_float4(
                            conv4(w0,a.x,b.x,c.x,v.x),conv4(w1,a.y,b.y,c.y,v.y),
                            conv4(w2,a.z,b.z,c.z,v.z),conv4(w3,a.w,b.w,c.w,v.w));
                    }
                }
            }
            __builtin_amdgcn_wave_barrier();
        }
    }
#endif
}

__global__ void boundary(const float* hist,const float* qkv,const float* w,float* h,int T,float eps) {
    const int c=blockIdx.x*128+threadIdx.x,t=blockIdx.y*32+blockIdx.z;
    if(t>=T)return;
    float v[4];
#pragma unroll
    for(int i=0;i<4;++i) {const int src=t-3+i;v[i]=src<0?hist[c*3+src+3]:qkv[size_t(src)*10240+c];}
    float value=conv4(*reinterpret_cast<const float4*>(w+c*4),v[0],v[1],v[2],v[3]);
    if(c<4096) {
        float sq=__fmaf_rn(value,value,__shfl_xor(__fmul_rn(value,value),16));
#pragma unroll
        for(int o=8;o>0;o>>=1)sq=__fadd_rn(sq,__shfl_xor(sq,o));
        __shared__ float part[4];
        if((threadIdx.x&31)==0)part[threadIdx.x>>5]=sq;
        __syncthreads();
        const float sum=__fadd_rn(__fadd_rn(__fadd_rn(part[0],part[1]),part[2]),part[3]);
        value*=rsqrtf(sum+eps);
    }
    h[size_t(t)*10240+c]=value;
}
} // namespace ssm_l2
