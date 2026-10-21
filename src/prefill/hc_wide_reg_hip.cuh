// Adapted from gufo DenseF16GEMMKernel: BF16 HC projection with a fused Strata mix.
// Keeps normalization and BF16/FP16 rounding in Strata order.
/*
MIT License

Copyright (c) 2026 gufo contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
namespace hc_wide_reg {
typedef __bf16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
 return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a,b,c);
#else
 __builtin_trap(); return c;
#endif
}
__device__ __forceinline__ v8f WmmaBf16(v16h a, v16h b, v8f c) { return Wmma(a,b,c); }
__device__ __forceinline__ void CodesToHalves(uint32_t codes, __half2 magic, __half2 scale, __half2 bias, __half2& lo, __half2& hi) {
#if defined(__HIP_DEVICE_COMPILE__)
 const uint32_t p0 = __builtin_amdgcn_perm(codes, 0x64646464U, 0x01050004U);
 const uint32_t p1 = __builtin_amdgcn_perm(codes, 0x64646464U, 0x03070206U);
 lo=__hfma2(__hadd2(__builtin_bit_cast(__half2,p0),magic),scale,bias);
 hi=__hfma2(__hadd2(__builtin_bit_cast(__half2,p1),magic),scale,bias);
#endif
}
template<int BM, int BN, int BK, int WM, int WN, int kRowGroup = 1,
         bool kStrided = false>
__launch_bounds__(256) __global__ void upmix(
    const void* __restrict__ w, const __half* __restrict__ x,
    float* __restrict__ y, const float* R, const float* rs, const float* norm, uint16_t* mixed16, uint16_t* mixed_h, std::size_t batch, std::size_t m, std::size_t k,
    std::size_t input_stride = 0, std::size_t output_stride = 0) {
  constexpr bool kHcMix = true, kSsmConv = false, kHalfWeights = false, kBf16 = false;
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  static_assert(!(kBf16 && kHalfWeights));
  // 16-bit weight rows staged as they are: F16, or BF16 with BF16
  // activations.
  constexpr bool kRawWeights = true;
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;
  // Weight and activation K blocks staged per thread; a unit past the
  // stage's block count is idle.
  constexpr int kAUnits = BM * BK;
  constexpr int kBUnits = BN * BK;
  constexpr int kAPer = (kAUnits + 255) / 256;
  constexpr int kBPer = (kBUnits + 255) / 256;

  // One K block of one row or token is four 16-byte chunks; the chunks of
  // nearby rows are permuted so a fragment read (one row per lane, 64-byte
  // stride) covers all bank groups.
  constexpr int kLdsChunks = BK * (BM + BN) * 4;
  static_assert(kLdsChunks * 16 >= 8 * 512 * 4, "epilogue transposes 16 KB");
  constexpr int kTransposeChunks = 8 * 16 * 36 * sizeof(float) / sizeof(uint4);
  constexpr int kLdsStorage =
      kLdsChunks < kTransposeChunks ? kTransposeChunks : kLdsChunks;
  __shared__ __attribute__((aligned(16))) uint4 s_lds[kLdsStorage];
  auto* s_a = reinterpret_cast<uint4(*)[BM][4]>(s_lds);
  auto* s_b = reinterpret_cast<uint4(*)[BN][4]>(s_lds + (BK * BM * 4));
  const auto swizzle = [](int row, int c) { return c ^ ((row >> 1) & 3); };

  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const auto* w_bytes = static_cast<const std::uint8_t*>(w);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;
  // Group row tiles for the narrow HC up projection. The tile grid's row
  // count is divisible by kRowGroup; every dot product keeps its K order.
  const unsigned row_group = blockIdx.y / kRowGroup;
  const unsigned within = (blockIdx.y % kRowGroup) * gridDim.x + blockIdx.x;
  const int r_block = (row_group * kRowGroup + within % kRowGroup) * BM;
  const int t_block = (within / kRowGroup) * BN;
  // BF16 replaces a library GEMM whose 32-token tile g starts its K loop
  // (g % 4) * 4 blocks in and wraps. A block takes the four tiles of one
  // rotation class, one per token wave, so every dot product keeps the
  // library's order.
  const int bf16_class = kBf16 ? static_cast<int>(blockIdx.x % 4) : 0;
  const int k_rotation = kBf16 ? (bf16_class * 4) % num_kb : 0;
  const auto token_of = [&](int local) {
    if constexpr (kBf16) {
      static_assert(BN == 128 && WN == 4 && kRowGroup == 1);
      return (static_cast<int>(blockIdx.x / 4) * 512) + (bf16_class * 32) +
             ((local / 32) * 128) + (local % 32);
    } else {
      return t_block + local;
    }
  };
  const auto rotate = [&](int kb) {
    const int r = kb + k_rotation;
    return r >= num_kb ? r - num_kb : r;
  };

  // Weight fetch unit p of a thread: row (p * 256 + tid) / BK, K block
  // (p * 256 + tid) % BK of the stage; rows past m read the last row with
  // a zero scale.
  const std::uint8_t* a_ptr[kAPer];
  bool a_live[kAPer];
#pragma unroll
  for (int p = 0; p < kAPer; ++p) {
    const int idx = (p * 256) + tid;
    const int r = r_block + (idx / BK);
    a_live[p] = idx < kAUnits && r < m_i;
    // Interleave the four gate streams within the tile, without repacking
    // weights. Each group of four rows produces one mixed hidden element.
    const int weight_row = kHcMix ? (r % 4) * (m_i / 4) + r / 4 : r;
    a_ptr[p] = w_bytes +
               static_cast<std::size_t>(a_live[p] ? weight_row : (m_i - 1)) *
                   static_cast<std::size_t>(num_kb) * (kRawWeights ? 64 : 34);
  }
  const __half* b_ptr[kBPer];
#pragma unroll
  for (int p = 0; p < kBPer; ++p) {
    const int idx = (p * 256) + tid;
    const int t = token_of(idx / BK);
    b_ptr[p] =
        idx < kBUnits && t < static_cast<int>(batch)
            ? x + (static_cast<std::size_t>(t) * (kStrided ? input_stride : k))
            : nullptr;
  }
  uint4 a_codes[kAPer][2];
  uint4 a_half[kRawWeights ? kAPer : 1][4];
  std::uint32_t a_d[kAPer];
  uint4 b_data0[kBPer],b_data1[kBPer],b_data2[kBPer],b_data3[kBPer];

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const bool live = a_live[p] && kb < num_kb;
      if constexpr (kRawWeights) {
        const auto* src = reinterpret_cast<const uint4*>(
            a_ptr[p] + static_cast<std::size_t>(rotate(kb)) * 64);
#pragma unroll
        for (int c = 0; c < 4; ++c)
          a_half[p][c] = live ? src[c] : make_uint4(0u, 0u, 0u, 0u);
      } else {
        const std::uint8_t* blk =
            a_ptr[p] +
            (static_cast<std::size_t>(live ? kb : (num_kb - 1)) * 34);
        a_d[p] = live ? *reinterpret_cast<const std::uint16_t*>(blk) : 0U;
        __builtin_memcpy(&a_codes[p][0], blk + 2, 16);
        __builtin_memcpy(&a_codes[p][1], blk + 18, 16);
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      if (b_ptr[p] != nullptr && kb < num_kb) {
        const auto* src =
            reinterpret_cast<const uint4*>(b_ptr[p] + (rotate(kb) * 32));
        b_data0[p] = src[0];
        b_data1[p] = src[1];
        b_data2[p] = src[2];
        b_data3[p] = src[3];
      } else {
        b_data0[p] = make_uint4(0u,0u,0u,0u);
        b_data1[p] = make_uint4(0u,0u,0u,0u);
        b_data2[p] = make_uint4(0u,0u,0u,0u);
        b_data3[p] = make_uint4(0u,0u,0u,0u);
      }
    }
  };

  const __half2 magic = __floats2half2_rn(-1152.0F, -1152.0F);
  const __half2 zero2 = __floats2half2_rn(0.0F, 0.0F);
  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kAUnits) {
        break;
      }
      const int row = idx / BK;
      const int kk = idx % BK;
      if constexpr (kRawWeights) {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          s_a[kk][row][swizzle(row, c)] = a_half[p][c];
      } else {
        // Q8_0 codes are signed; flipping the sign bit carries q + 128, which
        // the 1152 magic takes back out.
        const std::uint32_t words[8] = {
            a_codes[p][0].x, a_codes[p][0].y, a_codes[p][0].z, a_codes[p][0].w,
            a_codes[p][1].x, a_codes[p][1].y, a_codes[p][1].z, a_codes[p][1].w};
        const __half2 scale2 = __half2half2(
            __builtin_bit_cast(__half, static_cast<std::uint16_t>(a_d[p])));
        __half2 h[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          CodesToHalves(words[i] ^ 0x80808080U, magic, scale2, zero2, h[2 * i],
                        h[2 * i + 1]);
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          uint4 v;
          __builtin_memcpy(&v, &h[4 * c], 16);
          s_a[kk][row][swizzle(row, c)] = v;
        }
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kBUnits) {
        break;
      }
      const int t = idx / BK;
      const int kk = idx % BK;
      s_b[kk][t][swizzle(t,0)] = b_data0[p];
      s_b[kk][t][swizzle(t,1)] = b_data1[p];
      s_b[kk][t][swizzle(t,2)] = b_data2[p];
      s_b[kk][t][swizzle(t,3)] = b_data3[p];
    }
  };

  v8f acc[kWaveRowTiles][kWaveTokTiles];
  // Separate K16 chains reduce FP32 accumulation error for the sensitive
  // unquantized router/gate projections, with the same order in every chunk.
  v8f acc_high[kHalfWeights ? kWaveRowTiles : 1]
              [kHalfWeights ? kWaveTokTiles : 1]{};
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      acc[i][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  // Keep full tiles free of per-K bounds branches. The wide SSM paths can
  // omit whole 16-token subtiles in their final partial tile while retaining
  // the original WMMA order for every live output. One launch preserves
  // weight reuse between neighboring token tiles.
  const auto accumulate = [&]<bool partial>() {
    fetch_stage(0);
    for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
      commit_stage();
      __syncthreads();
      if (kb0 + BK < num_kb) {
        fetch_stage(kb0 + BK);
      }
      // Keeps the next stage's loads ahead of the matrix work: scheduled
      // freely, the compiler sinks them below it and the commit waits on
      // the full memory latency.
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int kb = 0; kb < BK; ++kb) {
        v16h a_lo[kWaveRowTiles];
        v16h a_hi[kWaveRowTiles];
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          const int row = (((wave_row * kWaveRowTiles) + i) * 16) + sub_lane;
          uint4 c[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            c[q] = s_a[kb][row][swizzle(row, q)];
          }
          __builtin_memcpy(&a_lo[i], &c[0], 32);
          __builtin_memcpy(&a_hi[i], &c[2], 32);
          __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int j = 0; j < kWaveTokTiles; ++j) {
          if constexpr (partial) {
            if (token_of(((wave_tok * kWaveTokTiles) + j) * 16) >= batch)
              continue;
          }
          const int t = (((wave_tok * kWaveTokTiles) + j) * 16) + sub_lane;
          uint4 c[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            c[q] = s_b[kb][t][swizzle(t, q)];
          }
          v16h b_lo;
          v16h b_hi;
          __builtin_memcpy(&b_lo, &c[0], 32);
          __builtin_memcpy(&b_hi, &c[2], 32);
            if constexpr (kBf16) {
              // One K16 chain in K order, as the library GEMM it replaces.
              acc[i][j] = WmmaBf16(a_lo[i], b_lo, acc[i][j]);
              acc[i][j] = WmmaBf16(a_hi[i], b_hi, acc[i][j]);
            } else if constexpr (kHalfWeights) {
              acc[i][j] = Wmma(a_lo[i], b_lo, acc[i][j]);
              acc_high[i][j] = Wmma(a_hi[i], b_hi, acc_high[i][j]);
            } else {
              acc[i][j] = Wmma(a_lo[i], b_lo, acc[i][j]);
              acc[i][j] = Wmma(a_hi[i], b_hi, acc[i][j]);
            }
          }
        }
      }
      __syncthreads();
    }
  };
  if constexpr (kSsmConv || kStrided) {
    if (t_block + BN <= batch)
      accumulate.template operator()<false>();
    else
      accumulate.template operator()<true>();
  } else {
    accumulate.template operator()<false>();
  }

  if constexpr (kHalfWeights) {
#pragma unroll
    for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j)
        acc[i][j] += acc_high[i][j];
    }
  }
  constexpr int hidden_tile=BM/4, gate_stride=hidden_tile+1, plane=16*gate_stride;
  static_assert(BM%16==0);
  static_assert(4*plane*sizeof(float)<=sizeof(s_lds));
  float* gates=reinterpret_cast<float*>(s_lds);
#pragma unroll
  for(int j=0;j<kWaveTokTiles;++j) {
#pragma unroll
    for(int group=0;group<WN;++group) {
      if(wave_tok==group) {
#pragma unroll
        for(int i=0;i<kWaveRowTiles;++i) {
#pragma unroll
          for(int l=0;l<8;++l) {
            int row=(wave_row*kWaveRowTiles+i)*16+2*l+half_id;
            gates[(row%4)*plane+sub_lane*gate_stride+row/4]=acc[i][j][l];
          }
        }
      }
      __syncthreads();
      for (int out=tid;out<16*(hidden_tile/4);out+=256) {
      int token_local=out/(hidden_tile/4), hlocal=(out%(hidden_tile/4))*4;
      size_t token=t_block+(group*kWaveTokTiles+j)*16+token_local;
      int h=r_block/4+hlocal;
      if(token_local<16 && token<batch) {
        float4 value={0.f,0.f,0.f,0.f};
#pragma unroll
        for(int c=0;c<4;++c) {
          int off=c*2560+h;
          float4 x=*reinterpret_cast<const float4*>(R+token*10240+off);
          float4 w=*reinterpret_cast<const float4*>(norm+off);
          const float* g=gates+c*plane+token_local*gate_stride+hlocal;
          float scale=rs[token*4+c];
          value.x=fmaf(x.x*scale*w.x,sigm(g[0]),value.x);
          value.y=fmaf(x.y*scale*w.y,sigm(g[1]),value.y);
          value.z=fmaf(x.z*scale*w.z,sigm(g[2]),value.z);
          value.w=fmaf(x.w*scale*w.w,sigm(g[3]),value.w);
        }
        value.x*=0.25f; value.y*=0.25f; value.z*=0.25f; value.w*=0.25f;
        if(y) *reinterpret_cast<float4*>(y+token*2560+h)=value;
        if(mixed16) {
          uint16_t halves[4]={act16(value.x),act16(value.y),act16(value.z),act16(value.w)};
          *reinterpret_cast<uint2*>(mixed16+token*2560+h)=__builtin_bit_cast(uint2,halves);
        }
        if(mixed_h) {
          uint16_t halves[4]={hf(value.x),hf(value.y),hf(value.z),hf(value.w)};
          *reinterpret_cast<uint2*>(mixed_h+token*(output_stride ? output_stride : 2560)+h)=__builtin_bit_cast(uint2,halves);
        }
      }
      }
      __syncthreads();
    }
  }
}
} // namespace hc_wide_reg
