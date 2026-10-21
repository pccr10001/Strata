#include "strata/prefill/kernels.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
template<class T> struct Device {
    T* data = nullptr;
    explicit Device(const std::vector<T>& values) {
        check(hipMalloc(reinterpret_cast<void**>(&data), values.size() * sizeof(T)));
        check(hipMemcpy(data, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    ~Device() { hipFree(data); }
    std::vector<T> read(size_t size) const {
        std::vector<T> values(size);
        check(hipMemcpy(values.data(), data, size * sizeof(T), hipMemcpyDeviceToHost));
        return values;
    }
};
}

int main() {
    try {
        constexpr int N = 2560, K = 10, guard = 16;
        std::mt19937 rng(8192);
        std::uniform_real_distribution<float> sample(-2.0f, 2.0f);
        for (int tokens : {1, 7, 129}) {
            std::vector<uint16_t> half(size_t(tokens) * K * N);
            std::vector<float> rounded(half.size());
            for (size_t i = 0; i < half.size(); ++i) {
                const __half value = __float2half(sample(rng));
                std::memcpy(&half[i], &value, sizeof(value));
                rounded[i] = __half2float(value);
            }
            std::vector<int32_t> slots(tokens * K);
            std::iota(slots.begin(), slots.end(), 0);
            std::shuffle(slots.begin(), slots.end(), rng);
            std::vector<float> weights(tokens * K), shared(tokens * N), gates(tokens);
            for (auto& value : weights) value = sample(rng);
            for (auto& value : shared) value = sample(rng);
            for (auto& value : gates) value = sample(rng);
            std::vector<float> output(tokens * N + 2 * guard, 123456.0f);
            Device<uint16_t> dh(half);
            Device<float> df(rounded), dw(weights), ds(shared), dg(gates), ref(output), candidate(output);
            Device<int32_t> di(slots);
            strata::prefill::moe_combine(df.data, di.data, dw.data, ds.data, dg.data, ref.data + guard, tokens, nullptr);
            strata::prefill::moe_combine_half(dh.data, di.data, dw.data, ds.data, dg.data, candidate.data + guard,
                                              tokens, nullptr);
            check(hipDeviceSynchronize());
            const auto expected = ref.read(output.size()), actual = candidate.read(output.size());
            if (std::memcmp(expected.data(), actual.data(), output.size() * sizeof(float)) != 0)
                throw std::runtime_error("half combine differs from the same rounded inputs in float combine");
            for (int i = 0; i < guard; ++i)
                if (actual[i] != 123456.0f || actual[actual.size() - 1 - i] != 123456.0f)
                    throw std::runtime_error("output guard was overwritten");
            std::printf("half combine: %d tokens, permuted slots, byte-exact output and intact guards\n", tokens);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
