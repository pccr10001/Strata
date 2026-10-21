# Hardware tuning: gfx1151

This branch is based on upstream v0.2.1.4 (`9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4`).
The upstream engine version is 0.1.46.3. All new runtime paths are opt-in.
The original model weights, BF16 hyper-connection weights, int8 KV cache and
native 262144-token context/RoPE are retained in the model runs below.

## Changes

- gfx1151 WMMA expert tiles and FP16 intermediate writes, including 192-row gate/up and down tiles.
- Padded HC output written directly for the next GEMM; skip the unused FP32 image and packing copy.
- Fused HC down/SiLU, shared up/SwiGLU, dense SSM/convolution, shared GDN Q/K,
  query norm/RoPE and prompt attention gating on eligible shapes.
- One-token GDN state commit avoids extra state copies during target-only decode.
- hipBLASLt solution table guarded by architecture and library version.

## Build and enable

Use the ROCm setup described in [AMD_HIP.md](AMD_HIP.md), then build from the repository root:

```sh
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build-hip --target strata -j6
. tools/tuning/gfx1151.env
```

Run the engine or server from this root after sourcing the preset. A server config's
`env` values override the parent environment: remove old tuning values or update them.
The relative hipBLASLt table path assumes this working directory. The table targets
hipBLASLt 100500; other versions use the library's normal selection.

The preset was selected for a resident Q4_K/Q5_1/Q8_0 Qwen3.8-family model. Format,
shape and device checks retain the existing paths for other models. That is not a
speed or output-equivalence guarantee for unmeasured formats. Fused reductions can
change floating-point rounding without changing the model's quantization.

## Validation

Linux, Ryzen AI Max+ 395 / gfx1151, ROCm HIP compiler 24, hipBLASLt 100500.
Passed half-combine, combine-write, QNorm, upmix, native fused-IQ parity, and WMMA
parity (3520 cases; 26 unsupported shapes skipped). Byte-exact claims apply only
to the individual test assertions; upmix and fused-IQ use numerical tolerances.
CUDA and SYCL builds were not available on this machine.

### Model run (2026-10-21)

CYBER-FROST-3.8 Q4 (111170978048-byte GGUF), existing BF16-compatible dense pack,
all 24576 experts resident, `--prefill 32768`, `--kv int8`, `--max-context 262144`,
`--mtp-max-t 1 --suffix-draft 0 --lookup-chain 0`. No weight requantization.

| Build | Median prefill tokens/s | Median target decode tokens/s |
| --- | ---: | ---: |
| Preserved pre-update optimized build, same day | 1290.2 | 25.9 |
| v0.2.1.4 with selected optimizations | 1295.4 | 25.9 |

Each build runs three fresh 16602-token prompts and generates 256 tokens per trial
at temperature 0, seed 42. Cached tokens and draft counts are zero. Complete
choices and usage match between builds for all three fixed prompts. This is an
upgrade compatibility check, not a speedup claim against stock upstream.

The previously measured ~1844 tokens/s was not reproduced: sampled power is
53.45-55.27 W in the current session. The SMU power limit could not be read without
sudo, so the cause is not confirmed. Do not compare the old 120 W session with this
session as evidence of a software regression. No 90%-of-theoretical claim is made.

Reproduce with `tools/tuning/bench_local_prefill.py --lines 512 --trials 3
--max-tokens 256 --expect-no-drafts --label pure-target-matched16k`, supplying
`--config`, `--output`, and the isolated localhost `--url`. Restart before fresh runs.

Native-context fresh/repeat/continuation prompts are 254723/254723/254801 tokens.
Fresh prefill is 1097.5 tokens/s; prefix hits are 0/254718/254718 tokens. All recall
the initial nonce, and complete choices and usage match the preserved pre-update
build. Run `tools/tuning/bench_context_continuation.py` with `--lines 7000
--nonce STRATA-CONTEXT-A9731`, plus `--config`, `--output`, and `--url`.

No external cache-server changes or machine-specific server configuration are included.
