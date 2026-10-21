#include "strata/prefill/kernels.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
constexpr int N = 2560, D = N * 4, K = 10, guard = 16;
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
template<class T> struct Buffer {
    T* data = nullptr;
    size_t count;
    explicit Buffer(size_t n) : count(n) { check(hipMalloc(&data, n * sizeof(T))); }
    explicit Buffer(const std::vector<T>& v) : Buffer(v.size()) {
        check(hipMemcpy(data, v.data(), count * sizeof(T), hipMemcpyHostToDevice));
    }
    ~Buffer() { (void) hipFree(data); }
    std::vector<T> read() const {
        std::vector<T> v(count);
        check(hipMemcpy(v.data(), data, count * sizeof(T), hipMemcpyDeviceToHost));
        return v;
    }
};
template<class T> void equal(const Buffer<T>& a, const Buffer<T>& b, const char* name) {
    const auto x = a.read(), y = b.read();
    if (std::memcmp(x.data(), y.data(), x.size() * sizeof(T))) {
        size_t i = 0;
        while (i < x.size() && !std::memcmp(&x[i], &y[i], sizeof(T))) ++i;
        std::fprintf(stderr, "%s first mismatch %zu\n", name, i);
        throw std::runtime_error(name);
    }
}
float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

void run(int tokens, bool padded, bool low, bool f16image, bool timing) {
    const int ld = D + (padded ? 64 : 0), inj_ld = padded ? 8 : 4;
    std::mt19937 rng(498);
    auto floats = [&](size_t n) {
        std::vector<float> v(n);
        for (auto& x : v) x = (int(rng() % 2001) - 1000) * .001f;
        return v;
    };
    std::vector<uint16_t> dm(size_t(tokens) * K * N);
    for (auto& x : dm) x = uint16_t((rng() & 0x83ffu) | 0x3800u);
    std::vector<int32_t> slots(size_t(tokens) * K);
    std::iota(slots.begin(), slots.end(), 0);
    std::shuffle(slots.begin(), slots.end(), rng);
    const auto initial = floats(size_t(tokens) * D);
    Buffer<uint16_t> ddm(dm);
    Buffer<int32_t> ds(slots);
    Buffer<float> dw(floats(size_t(tokens) * K)), shared(floats(size_t(tokens) * N)), sg(floats(tokens));
    Buffer<float> inject(floats(size_t(tokens) * inj_ld)), norm(floats(D));
    Buffer<float> original(initial), ref_R(initial), got_R(initial), bo(size_t(tokens) * N);
    Buffer<float> ref_rs(size_t(tokens) * 4 + 2 * guard), got_rs(ref_rs.count);
    Buffer<uint16_t> ref_x(size_t(tokens) * ld + 2 * guard), got_x(ref_x.count);
    Buffer<uint16_t> ref_lo(ref_x.count), got_lo(ref_x.count);
    auto fill_guard = [&](auto& b) { check(hipMemset(b.data, 0xa5, b.count * sizeof(*b.data))); };
    fill_guard(ref_rs); fill_guard(got_rs); fill_guard(ref_x); fill_guard(got_x);
    fill_guard(ref_lo); fill_guard(got_lo);
    hipStream_t stream;
    check(hipStreamCreate(&stream));
    auto launch = [&](bool fused) {
        if (fused)
            strata::prefill::moe_combine_half_write_norm(ddm.data, ds.data, dw.data, shared.data, sg.data,
                got_R.data, inject.data, inj_ld, norm.data, 1e-6f, got_rs.data + guard,
                got_x.data + guard, tokens, stream, low ? got_lo.data + guard : nullptr, ld, f16image);
        else {
            strata::prefill::moe_combine_half(ddm.data, ds.data, dw.data, shared.data, sg.data, bo.data, tokens, stream);
            strata::prefill::gr_write_norm_rs(ref_R.data, bo.data, inject.data, inj_ld, norm.data,
                1e-6f, ref_rs.data + guard, ref_x.data + guard, tokens, stream,
                low ? ref_lo.data + guard : nullptr, ld, f16image);
        }
    };
    launch(false); launch(true); check(hipStreamSynchronize(stream));
    equal(ref_R, got_R, "R"); equal(ref_rs, got_rs, "rs");
    equal(ref_x, got_x, "xn16/padding/guards"); equal(ref_lo, got_lo, "low/padding/guards");
    std::printf("PASS T=%d padded=%d low=%d f16image=%d: R/rs/images/guards byte-exact\n",
                tokens, padded, low, f16image);
    if (timing) {
        std::vector<float> times[2];
        hipEvent_t begin, end;
        check(hipEventCreate(&begin)); check(hipEventCreate(&end));
        for (int trial = 0; trial < 23; ++trial)
            for (int pos = 0; pos < 2; ++pos) {
                const int fused = pos ^ (trial & 1);
                check(hipMemcpyAsync(fused ? got_R.data : ref_R.data, original.data,
                                     original.count * sizeof(float), hipMemcpyDeviceToDevice, stream));
                check(hipEventRecord(begin, stream)); launch(fused);
                check(hipEventRecord(end, stream)); check(hipEventSynchronize(end));
                float ms;
                check(hipEventElapsedTime(&ms, begin, end));
                if (trial >= 3) times[fused].push_back(ms);
            }
        std::printf("BENCH T=%d baseline_ms=%.6f fused_ms=%.6f speedup=%.6f\n",
                    tokens, median(times[0]), median(times[1]), median(times[0]) / median(times[1]));
        check(hipEventDestroy(begin)); check(hipEventDestroy(end));
    }
    check(hipStreamDestroy(stream));
}
}

int main(int argc, char** argv) {
    try {
        if (argc > 2) return 2;
        if (argc > 1) {
            char* end = nullptr;
            const long parsed = std::strtol(argv[1], &end, 10);
            if (end == argv[1] || *end || parsed < 1 || parsed > 32768) return 2;
            const int tokens = static_cast<int>(parsed);
            run(tokens, true, false, false, true);
        } else {
            for (int tokens : {1, 7, 129})
                for (bool padded : {false, true})
                    for (bool low : {false, true})
                        for (bool image : {false, true}) run(tokens, padded, low, image, false);
        }
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
}
