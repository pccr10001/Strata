#include "strata/prefill/kernels.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void env(const char* name, const char* value) {
#ifdef _WIN32
    const int status = _putenv_s(name, value ? value : "");
#else
    const int status = value ? setenv(name, value, 1) : unsetenv(name);
#endif
    if (status) throw std::runtime_error("cannot set test environment");
}
struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t bytes) { ck(cudaMalloc(&p, bytes)); }
    ~Buffer() { cudaFree(p); }
    template<class T> T* as() { return static_cast<T*>(p); }
};
static void compare(Buffer& a, Buffer& b, size_t bytes) {
    std::vector<unsigned char> x(bytes), y(bytes);
    ck(cudaMemcpy(x.data(), a.p, bytes, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(y.data(), b.p, bytes, cudaMemcpyDeviceToHost));
    if (x != y) throw std::runtime_error("GDN staging differs from 8-token staging");
}
int main(int argc, char** argv) {
    try {
        const bool flat_norm = argc > 1 && std::strcmp(argv[1], "--flat-norm") == 0;
        env("STRATA_GDN_STAGE_TOKENS", "8");
        env("STRATA_GDN_CHUNKED", "0");
        env("STRATA_GDN_KEYHEAD", "1");
        env("STRATA_GDN_PIPELINE", "1");
        env("STRATA_GDN_REC_HEADS", nullptr);
        cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::mt19937 rng(309);
        for (int T : {1, 9, 16, 17, 127, 1025, 8192, 32768}) {
            const size_t state_bytes = 128ull * 48 * 128 * 4, y_bytes = size_t(T) * 48 * 128 * 4;
            Buffer state(state_bytes), ref_state(state_bytes), h(size_t(T) * 10240 * 4);
            Buffer gate(size_t(T) * 48 * 4), beta(size_t(T) * 48 * 4), z(y_bytes), gamma(128 * 4);
            Buffer y(y_bytes), ref_y(y_bytes), y16(y_bytes / 2), ref_y16(y_bytes / 2);
            auto fill = [&](Buffer& buf, size_t count, float scale, float bias) {
                std::vector<float> v(count);
                for (auto& f : v) f = (int(rng() % 2001) - 1000) * scale + bias;
                ck(cudaMemcpy(buf.p, v.data(), count * 4, cudaMemcpyHostToDevice));
            };
            fill(state, state_bytes / 4, .00001f, 0);
            ck(cudaMemcpy(ref_state.p, state.p, state_bytes, cudaMemcpyDeviceToDevice));
            fill(h, size_t(T) * 10240, .0001f, 0);
            fill(gate, size_t(T) * 48, .00005f, -.1f);
            fill(beta, size_t(T) * 48, .0001f, .2f);
            fill(z, y_bytes / 4, .001f, 0);
            fill(gamma, 128, .0001f, 1);
            env("STRATA_GDN_NORM_FLAT", "0");
            strata::prefill::gdn_recurrence_variant(0, ref_state.as<float>(), h.as<float>(), gate.as<float>(),
                beta.as<float>(), z.as<float>(), gamma.as<float>(), 1e-6f, ref_y.as<float>(), ref_y16.as<uint16_t>(), T, stream);
            env("STRATA_GDN_NORM_FLAT", flat_norm ? "1" : "0");
            strata::prefill::gdn_recurrence_variant(flat_norm ? 0 : 6, state.as<float>(), h.as<float>(), gate.as<float>(),
                beta.as<float>(), z.as<float>(), gamma.as<float>(), 1e-6f, y.as<float>(), y16.as<uint16_t>(), T, stream);
            ck(cudaStreamSynchronize(stream));
            compare(state, ref_state, state_bytes); compare(y, ref_y, y_bytes); compare(y16, ref_y16, y_bytes / 2);
            std::printf("T %d: state, FP32 output and FP16 image bitwise identical\n", T);
        }
        ck(cudaStreamDestroy(stream)); return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
