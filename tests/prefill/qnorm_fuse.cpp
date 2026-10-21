#include "strata/prefill/kernels.hpp"
#ifdef STRATA_USE_HIP
#include <hip/hip_runtime.h>
#define GPU(name) hip##name
#else
#include <cuda_runtime.h>
#define GPU(name) cuda##name
#endif
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
constexpr int guard = 16;
void check(GPU(Error_t) status) {
    if (status != GPU(Success)) throw std::runtime_error(GPU(GetErrorString)(status));
}
struct Buffer {
    float* data = nullptr;
    size_t count;
    explicit Buffer(const std::vector<float>& values) : count(values.size()) {
        check(GPU(Malloc)(&data, count * sizeof(float)));
        check(GPU(Memcpy)(data, values.data(), count * sizeof(float), GPU(MemcpyHostToDevice)));
    }
    ~Buffer() { (void) GPU(Free)(data); }
    std::vector<float> read() const {
        std::vector<float> values(count);
        check(GPU(Memcpy)(values.data(), data, count * sizeof(float), GPU(MemcpyDeviceToHost)));
        return values;
    }
};
void equal(const std::vector<float>& a, const std::vector<float>& b, const char* name) {
    if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float))) {
        size_t i = 0;
        while (i < a.size() && !std::memcmp(&a[i], &b[i], sizeof(float))) ++i;
        std::fprintf(stderr, "%s mismatch at %zu: %.9g / %.9g\n", name, i, a[i], b[i]);
        throw std::runtime_error(name);
    }
}
void run(int tokens, int pattern, float eps, bool timing = false) {
    std::mt19937 random(671 + pattern);
    const size_t rows = size_t(tokens) * 24;
    std::vector<float> input(rows * 512 + 2 * guard, -917.0f), weights(256);
    for (size_t i = 0; i < rows * 512; ++i) {
        float value = (int(random() % 20001) - 10000) * 0.0001f;
        if (pattern == 1) value = i % 2 ? -0.0f : 0.0f;
        if (pattern == 2) value *= i % 4 < 2 ? 1e-15f : 1e15f;
        input[guard + i] = value;
    }
    for (auto& value : weights) value = (int(random() % 2001) - 1000) * 0.001f;
    const std::vector<float> initial(rows * 256 + 2 * guard, -819.0f);
    Buffer source(input), weight(weights), ref(initial), fused(initial);
    if (tokens > 0) {
        strata::prefill::split_q(source.data + guard, ref.data + guard, tokens, nullptr);
        strata::prefill::rms_rows(ref.data + guard, weight.data, rows, 256, 256, eps, nullptr);
    }
    strata::prefill::split_q_norm(source.data + guard, weight.data, fused.data + guard, tokens, eps, nullptr);
    check(GPU(DeviceSynchronize)());
    equal(ref.read(), fused.read(), "normalized query/guards");
    equal(input, source.read(), "full query/gate input/guards");
    const auto output = fused.read();
    for (int i = 0; i < guard; ++i)
        if (std::memcmp(&output[i], &initial[i], sizeof(float)) ||
            std::memcmp(&output[output.size() - 1 - i], &initial[initial.size() - 1 - i], sizeof(float)))
            throw std::runtime_error("output guard overwritten");
    std::printf("PASS T=%d pattern=%d eps=%g: query byte-exact, gates/input/guards unchanged\n", tokens, pattern, eps);
    if (timing) {
        GPU(Event_t) begin, end;
        check(GPU(EventCreate)(&begin)); check(GPU(EventCreate)(&end));
        std::vector<float> times[2];
        for (int trial = 0; trial < 23; ++trial)
            for (int position = 0; position < 2; ++position) {
                const int use_fused = position ^ (trial & 1);
                check(GPU(EventRecord)(begin));
                if (use_fused)
                    strata::prefill::split_q_norm(source.data + guard, weight.data, fused.data + guard, tokens, eps, nullptr);
                else {
                    strata::prefill::split_q(source.data + guard, fused.data + guard, tokens, nullptr);
                    strata::prefill::rms_rows(fused.data + guard, weight.data, rows, 256, 256, eps, nullptr);
                }
                check(GPU(EventRecord)(end)); check(GPU(EventSynchronize)(end));
                float ms = 0;
                check(GPU(EventElapsedTime)(&ms, begin, end));
                if (trial >= 3) {
                    times[use_fused].push_back(ms);
                    std::printf("TIMING trial=%d fused=%d ms=%.6f\n", trial - 3, use_fused, ms);
                }
            }
        for (auto& values : times) std::sort(values.begin(), values.end());
        std::printf("BENCH T=%d baseline_ms=%.6f fused_ms=%.6f speedup=%.6f\n", tokens,
                    times[0][10], times[1][10], times[0][10] / times[1][10]);
        equal(ref.read(), fused.read(), "query/guards after timing");
        check(GPU(EventDestroy)(begin)); check(GPU(EventDestroy)(end));
    }
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 1) {
            if (argc != 3 || std::strcmp(argv[1], "--bench")) return 2;
            char* end = nullptr;
            const long tokens = std::strtol(argv[2], &end, 10);
            if (end == argv[2] || *end || tokens < 1 || tokens > 32768) return 2;
            run(static_cast<int>(tokens), 0, 1e-6f, true);
            return 0;
        }
        for (int tokens : {0, 1, 7, 33, 129})
            for (int pattern : {0, 1, 2})
                for (float eps : {1e-6f, 1e-5f}) run(tokens, pattern, eps);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
