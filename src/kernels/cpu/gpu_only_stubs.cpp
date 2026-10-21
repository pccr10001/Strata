// ARM GPU-only build: retain x86 SIMD symbols, but trap if they are called.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include <cstdio>
#include <cstdlib>
namespace strata::kernels::cpu {
namespace { [[noreturn]] void unavailable(const char* name) {
    std::fprintf(stderr, "GPU-only ARM build reached CPU SIMD kernel: %s\n", name);
    std::abort();
} }
void act_quant_q8_1(const float* x, int n, ActQ& a) { unavailable("act_quant_q8_1"); }
void s2_expert_vnni(const uint8_t* blob, const float* x, float* out, ExpertScratch& ws) { unavailable("s2_expert_vnni"); }
void s2_expert_vnni_q(const uint8_t* blob, const ActQ& a1, float* out, ExpertScratch& ws) { unavailable("s2_expert_vnni_q"); }
void s2_expert_gu_rows(const uint8_t* blob, const ActQ& a1, float* ff, int r0, int r1) { unavailable("s2_expert_gu_rows"); }
void s2_expert_down_rows(const uint8_t* blob, const ActQ& a2, float* out, int r0, int r1) { unavailable("s2_expert_down_rows"); }
void s2_expert_gu_rows_multi(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* ff, int r0,
                             int r1) { unavailable("s2_expert_gu_rows_multi"); }
void s2_expert_down_rows_multi(const uint8_t* blob, const ActQ* const* a2, int n_tokens, float* const* out, int r0,
                               int r1) { unavailable("s2_expert_down_rows_multi"); }
void s2_expert_vnni_multi(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* out,
                          ExpertScratchMulti& ws) { unavailable("s2_expert_vnni_multi"); }
void q2_0_gguf_rows_multi(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                          float* const* out, int r0, int r1) { unavailable("q2_0_gguf_rows_multi"); }
void q2_0_gguf_rows_multi_avx2(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1) { unavailable("q2_0_gguf_rows_multi_avx2"); }
bool q2_bitplane_enabled() { return false; }
void q2_0_gguf_rows_multi_avx2_legacy(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                                      float* const* out, int r0, int r1) { unavailable("q2_0_gguf_rows_multi_avx2_legacy"); }
void q2_0_gguf_rows_multi_avx2_v(bool vnni, const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a,
                                 int nt, float* const* out, int r0, int r1) { unavailable("q2_0_gguf_rows_multi_avx2_v"); }
void act_quant_q8_1_avx2(const float* x, int n, ActQ& a) { unavailable("act_quant_q8_1_avx2"); }
void q2_0_gguf_rows_multi_avx1(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1) { unavailable("q2_0_gguf_rows_multi_avx1"); }
void q2_0_gguf_rows_multi_avx1_legacy(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                                      float* const* out, int r0, int r1) { unavailable("q2_0_gguf_rows_multi_avx1_legacy"); }
void act_quant_q8_1_avx1(const float* x, int n, ActQ& a) { unavailable("act_quant_q8_1_avx1"); }
void s2_expert_scalar(const uint8_t* blob, const float* x, float* out, bool quant_acts) { unavailable("s2_expert_scalar"); }
void iq512_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1) { unavailable("iq512_gu_rows"); }
void iq512_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1) { unavailable("iq512_rows"); }
void iq256_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1) { unavailable("iq256_gu_rows"); }
void iq256_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1) { unavailable("iq256_rows"); }
int iq256_variant() noexcept { return 0; }
int iq256_variants() noexcept { return 0; }
void iq256_gu_rows_v(int variant, int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                     const void* const* act, int nt, float* const* ff, int r0, int r1) { unavailable("iq256_gu_rows_v"); }
void iq256_rows_v(int variant, int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act,
                  int nt, float* const* out, int r0, int r1) { unavailable("iq256_rows_v"); }
void q8k_quant_avx2(const float* x, void* y, int64_t n) { unavailable("q8k_quant_avx2"); }
void iq4nl256_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                        float* const* out, int r0, int r1) { unavailable("iq4nl256_down_rows"); }
void iq4nl256_down_rows_v(int variant, const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                          float* const* out, int r0, int r1) { unavailable("iq4nl256_down_rows_v"); }
bool iq128_supported(int ggml_type) noexcept { return false; }
void iq128_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1) { unavailable("iq128_gu_rows"); }
bool iq128_down_supported(int ggml_type) noexcept { return false; }
void iq128_down_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                     float* const* out, int r0, int r1) { unavailable("iq128_down_rows"); }
void q8k_quant_avx1(const float* x, void* y, int64_t n) { unavailable("q8k_quant_avx1"); }
void kq256_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1) { unavailable("kq256_gu_rows"); }
void kq256_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1) { unavailable("kq256_rows"); }
void bf16_rows_dot(const uint16_t* w, int rows, int cols, const float* x, float* out) { unavailable("bf16_rows_dot"); }
void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) { unavailable("bf16_rows_dot_multi"); }
void bf16_rows_dot_multi_avx1(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) { unavailable("bf16_rows_dot_multi_avx1"); }
}
