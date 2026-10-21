// HIP opt-in query norm/RoPE fusion; included inside the existing anonymous namespace.
template<bool TAB>
__global__ void split_q_norm_rope_kernel(const float* __restrict__ qf, const float* __restrict__ w,
                                        float* __restrict__ q, float eps, int64_t pos0,
                                        float theta_scale, float freq_scale, float corr_low,
                                        float corr_high, float ext_factor, float mscale,
                                        const int32_t* __restrict__ mtab, strata::kernels::RopeTab rt) {
    __shared__ float sh[32];
    __shared__ float normalized_rotary[64];
    const int64_t row = blockIdx.x;
    const int col = threadIdx.x;
    const float value = qf[row * 512 + col];
    float ss = 0.0f;
    ss += value * value;
    // The original runtime row loop rounds the square before the shuffle sum.
#if defined(__HIPCC__)
    asm volatile("" : "+v"(ss));
#else
    asm volatile("" : "+f"(ss));
#endif
    const float s = rsqrtf(block_sum(ss, sh) / 256.0f + eps);
    __syncthreads();
    const float normalized = s * value * w[col];
    if (col < 64) normalized_rotary[col] = normalized;
    else q[row * 256 + col] = normalized;
    __syncthreads();
    if (col < 32) {
      const int pair = col;
      const int64_t t = row / 24;
      float c, sn;
      if (!(TAB && strata::kernels::rope_tab_cs(rt,
               strata::kernels::mrope_pos(mtab, (int)(pos0+t), pair), pair, c, sn))) {
        const float theta_extrap = (float)strata::kernels::mrope_pos(mtab, (int)(pos0+t), pair)
                                   * powf(theta_scale, (float)pair);
        strata::kernels::rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high,
                                           ext_factor, mscale, pair, c, sn);
      }
      const float a = normalized_rotary[pair], b = normalized_rotary[pair+32];
      q[row*256+pair] = a*c-b*sn;
      q[row*256+pair+32] = a*sn+b*c;
    }
}
