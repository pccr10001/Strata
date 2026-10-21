// Included inside verify_kernels.cu's anonymous namespace on HIP only.
__global__ void __launch_bounds__(S * RG) gdn_t1_commit_early_kernel(
    float* __restrict__ state, const float* __restrict__ hbuf, int C,
    const float* __restrict__ gate, const float* __restrict__ beta,
    const float* __restrict__ z, const float* __restrict__ gamma, float eps,
    float* __restrict__ y, int h_k, int h_v, const int32_t* __restrict__ n_keep,
    GdnQ81* __restrict__ xq) {
    // The host selects n_tok=1 and a non-null keep pointer; valid counts are 0/1.
    if (*n_keep <= 0) return;
    __shared__ float sk[2][S], sq[2][S];
    __shared__ float red_kv[RG][S], red_o[RG][S];
    __shared__ float wsum[S / 32];
    const int head = blockIdx.x;
    const int col = threadIdx.x;
    const int rg = threadIdx.y;
    const int qh = head % h_k;
    const int qk = S * h_k;
    if (rg == 0) sk[0][col] = hbuf[qk + qh * S + col];
    else if (rg == 1) sq[0][col] = hbuf[qh * S + col];
    const float gam = (rg == 0) ? gamma[col] : 0.0f;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    __syncthreads();
    const float g = __expf(gate[head]);
    float kv = 0.0f;
#pragma unroll
    for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[0][rg * RPG + r], kv);
    red_kv[rg][col] = kv;
    __syncthreads();
    const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
    const float delta = (hbuf[2 * qk + head * S + col] - g * kv_col) * beta[head];
    float o = 0.0f;
    // Preserve the pointer bits while preventing reuse of all 32 load addresses.
    uintptr_t store_address_base = reinterpret_cast<uintptr_t>(base);
#if defined(__HIPCC__)
    asm volatile("" : "+v"(store_address_base));
#endif
    float* dst = reinterpret_cast<float*>(store_address_base);
#pragma unroll
    for (int r = 0; r < RPG; ++r) {
        s[r] = fmaf(g, s[r], sk[0][rg * RPG + r] * delta);
        o = fmaf(s[r], sq[0][rg * RPG + r], o);
        *dst = s[r];
        dst += row_stride;
    }
    red_o[rg][col] = o;
    __syncthreads();
    float oc = 0.0f, sq_part = 0.0f;
    if (rg == 0) {
        oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float) S);
        sq_part = oc * oc;
    }
    for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
    if (rg == 0 && (col & 31) == 0) wsum[col >> 5] = sq_part;
    __syncthreads();
    if (rg == 0) {
        const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
        const float scale = rsqrtf(ss / (float) S + eps);
        const float zz = z[head * S + col];
        const float yv = oc * scale * gam * (1.0f / (1.0f + __expf(-zz)));
        y[head * S + col] = yv;
        gdn_q8_1_store(xq, (size_t) head * S + col, yv);
    }
}
