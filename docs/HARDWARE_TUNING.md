# Hardware tuning: gb10

This branch is based on upstream v0.2.1.4 (`9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4`).
The upstream engine version is 0.1.46.3. All new runtime paths are opt-in.
The original model weights, BF16 hyper-connection weights, int8 KV cache and
native 262144-token context/RoPE are retained in the model runs below.

## Changes

- Opt-in ARM64 CUDA build, portable fences/spin loops and explicit CPU expert traps.
- Q4_K/Q5_1 pointer-table MMQ instances read resident experts without gathering their weights.
- 16-token GDN staging and flat output normalization.
- GDN convolution/L2 fusion with explicit SM121 rounding order (`STRATA_GDN_CONVL2=2`).
- Opt-in Linux UMA available-memory query for prefill planning.

## Build and enable

Use CUDA 13 with SM121 support and a compatible ggml source tree as described in
[DETAILS.md](DETAILS.md):

```sh
cmake -S . -B build-arm -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121 -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_GPU_ONLY_ARM=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_MMQ_TAB=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_GGML_DIR=/path/to/compatible/llama.cpp
cmake --build build-arm --target strata -j6
. tools/tuning/gb10.env
```

Keep **`libstrata_kq.so` beside the actual `strata` executable**, including when copying
or archiving it. The loader resolves the executable directory, not the working
directory. A missing library silently selects the slower FP16 expert path. For Q4
validate both the process's loaded library maps and the engine log reporting
`48 of 48 layers` for resident prompt MMQ. Q2 and IQ types use the statically
linked MMQ instances, so they do not need to map the K-quant shared library.

ARM GPU-only mode requires every expert resident, the graph hit path and zero CPU
prefill share. CPU expert execution is unsupported and traps if reached. Use the
model's own profile geometry: 24576 experts for the full family, 12288 for Coder,
plus space for MMQ guards. Native-pack `--expert-cache N` is a byte budget of
`N * align256(largest_blob)`, rather than a promise of N resident experts. Direct
MMQ adds 256 bytes per expert slot. For uniform blobs use at least
`ceil(experts * align256(blob + 256) / align256(blob))`; a budget suitable for Q4
can otherwise leave two Q2_0 experts outside the cache. Check the actual filled
count in the engine log. This is not a general ARM CPU backend.

Run from the root after sourcing the preset; update conflicting server config `env`
values. HC requantization stays disabled. The register-only convolution candidate
was excluded because corrected tests with the K-quant library found no stable gain.

## Validation

Linux aarch64, NVIDIA GB10 / Blackwell SM121, CUDA 13.0.88.
Passed GDN staging and flat-normalization state/output bitwise comparisons through
32768 tokens, and pointer-table MMQ parity (890 comparisons, zero failures).
HIP and SYCL builds were not available on this machine.

### Model run (2026-10-21)

CYBER-FROST-3.8 Q4 (111170978048-byte GGUF), existing BF16-compatible dense pack,
all 24576 experts resident, `--prefill 32768`, `--kv int8`, `--max-context 262144`,
`--mtp-max-t 1 --suffix-draft 0 --lookup-chain 0`. No weight requantization.

| Run | Prefill tokens/s | Target decode tokens/s |
| --- | ---: | ---: |
| Fresh 16603-token prompt, trial 1 | 1782.1 | 27.1 |
| Trial 2 | 1857.9 | 27.3 |
| Trial 3 | 1852.4 | 27.5 |
| Median | 1852.4 | 27.3 |

Each trial generates 256 tokens at temperature 0, seed 42; cached tokens and draft
counts are zero. Complete choices and usage match the preserved pre-update K-quant
build on all three fixed prompts. This is an upgrade compatibility check, not a
same-day speedup claim against stock upstream.

Native-context fresh/repeat/continuation prompts are 254723/254723/254801 tokens.
Fresh prefill is 1633.2 tokens/s; prefix hits are 0/254718/254718 tokens. All recall
the initial nonce; complete choices and usage match the preserved build. The exact
executable and K-quant shared library were checked in the running process, with
48/48 layers using direct MMQ.

The earlier ~1980 tokens/s result was not reproduced in this corrected run. A
register-only candidate showed no stable improvement with the required K-quant
library, so it is excluded. No 90%-of-theoretical-throughput claim is made.

Reproduce with `tools/tuning/bench_local_prefill.py --lines 512 --trials 3
--max-tokens 256 --expect-no-drafts --label gb10-direct-controlled-original-precision`
and `bench_context_continuation.py --lines 7000 --nonce STRATA-CONTEXT-A9731`, supplying
`--config`, `--output`, and the isolated localhost `--url`. Restart before fresh runs.
No external cache-server changes or machine-specific server configuration are included.
