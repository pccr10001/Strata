#pragma once
#include <atomic>
#if defined(__aarch64__)
inline void strata_spin_pause() { asm volatile("yield" ::: "memory"); }
inline void strata_store_fence() { asm volatile("dmb oshst" ::: "memory"); }
inline void strata_load_fence() { asm volatile("dmb oshld" ::: "memory"); }
#else
#include <immintrin.h>
inline void strata_spin_pause() { _mm_pause(); }
inline void strata_store_fence() { _mm_sfence(); }
inline void strata_load_fence() { std::atomic_thread_fence(std::memory_order_acquire); }
#endif

inline void strata_system_fence() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
#if defined(__aarch64__)
    // The GPU participates in the outer shareable domain.
    asm volatile("dmb osh" ::: "memory");
#endif
}
