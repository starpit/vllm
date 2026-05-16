# Standalone pod smoke tests for ferrite-TK ops

Per-op `.cu` harnesses that exercise `ferrite::ops::<op>::{loader,
consumer,launcher,storer}` without going through ferrite's codegen.
Each one defines its own `FerriteConfig`, inlines the 4-role
dispatch the way codegen does, loads random bf16 inputs, and
compares the device output against a CPU reference.

Purpose: isolate "is the op's `.cuh` numerically correct?" from
"is codegen wiring correct?". A failure here is an op bug; codegen
failures get caught by building `ferrite-cuda-builder` and running
the generated variant.

## Build / run (pod `nick`, CUDA 12.9, H100 sm_90a)

Sync the csrc tree to the pod first:

```
oc rsync vllm-rs/crates/ferrite-kernels/csrc/ \
  nick:/home/nickm/vllm-mega/vllm-rs/crates/ferrite-kernels/csrc/
```

Then on pod:

```
nvcc -O3 -std=c++20 \
  -gencode arch=compute_90a,code=sm_90a \
  --extended-lambda --expt-relaxed-constexpr -DKITTENS_HOPPER \
  -I /home/nickm/vllm-mega/vllm-rs/third_party/thunderkittens/include \
  -I /home/nickm/vllm-mega/vllm-rs/crates/ferrite-kernels/csrc/tk \
  /home/nickm/vllm-mega/vllm-rs/crates/ferrite-kernels/csrc/smoke/ferrite_gemv_smoke.cu \
  -o /tmp/ferrite_gemv_smoke -lcuda
/tmp/ferrite_gemv_smoke
```

Expected output: `ok: gemv_bf16 matches CPU reference within
tolerance`, exit 0.

## gencode note

`sm_90a` (not plain `sm_90`) is required — TK's `setmaxnreg.inc/
dec` PTX lives in the `a` extension arch. Same constraint as
`ferrite-cuda-builder/build.rs`.

## Harnesses

- `ferrite_gemv_smoke.cu` — gemv_bf16, N=64, K=2048 (llama-3.2-1B
  `hidden_dim`). Validates the default Phase 2 config shape.
- `ferrite_gemv_smoke_k8192.cu` — gemv_bf16, N=32, K=8192
  (llama-3.2-1B `intermediate_dim`). Validates the op against
  the wider matmul shape that up/gate/down proj use, with
  PAGE_SIZE=16384.
- `ferrite_gemm_smoke.cu` — Phase 3f gemm_bf16, M=8 tokens,
  N=64, K=2048. Extends gemv_bf16's single-row shape to a 2D
  grid (`dim3(N, M)`) — one CTA per (token, output_col) — and
  validates the m≥2 prefill matmul path against a CPU reference.
- `ferrite_pool_abi_smoke.cu` — Phase 3e pool-ABI exit gate. Two
  rms_norm ops back-to-back, driven through the same
  `act_ptrs[]` / `weight_ptrs[w * NUM_LAYERS + l]` pointer-of-
  pointers shape `emit_cu_variant` emits. Exercises multi-op
  base_stage assignment, multi-accessor weight indexing, and the
  staged-pointer-array launch path against a CPU reference.
- `ferrite_fused_add_rms_norm_smoke.cu` — Phase 3f part 2a
  `fused_add_rms_norm` op. Two in-place slots (delta + residual)
  and one weight accessor, NUM_PAGES=3. Residual_out is bf16 of
  `delta + residual`; delta_out is bf16 of `rms_norm(residual_out)
  * weight`. Validates the dual-output storer path (two
  `tma::store_async` calls in one op) and the simultaneous
  per-page `page_done` arrival pattern the consumer issues.
- `ferrite_embed_smoke.cu` — Phase 3f part 2e-ii `embed` op.
  HIDDEN_DIM=2048 (Llama-3.2-1B), VOCAB_SIZE=128,
  NUM_TOKENS=4. Grid `dim3(NUM_TOKENS)`; each CTA gathers one row
  from a bf16 `[VOCAB_SIZE, HIDDEN_DIM]` embedding table into the
  output at offset `input_ids[blockIdx.x] * HIDDEN_DIM`. NUM_PAGES
  stays at 2 to match gemv's layout (embed needs only 1 page but
  there's no win to trimming below the substrate's minimum).
  Validates the pure-gather no-math consumer path — the only
  source of error is the bf16 round-trip, which both sides share,
  so the device output bit-matches the CPU reference (max_abs = 0
  expected, tolerance 1e-3 as epsilon-slop).
- `ferrite_silu_upgate_smoke.cu` — Phase 3f part 2g-ii
  `silu_upgate` op. HIDDEN_DIM=2048, INTERMEDIATE_DIM=8192
  (Llama-3.2-1B decode MLP shape), NCW=4. Grid
  `dim3(INTERMEDIATE_DIM)`; each CTA owns one output row, loading
  the activation `x` plus the gate row and up row of a packed
  `[2*INTERMEDIATE_DIM, HIDDEN_DIM]` weight buffer. NUM_PAGES=3
  (act + gate_w + up_w). Consumer does interleaved fp32 gate/up
  dot products with a single consumer-scoped `bar.sync` (id 13)
  publishing both partials, warp 0 lane 0 fuses `silu(gate) * up`
  and packs the bf16 output. Inputs scaled so gate/up partial
  sums cross silu's inflection band; the CPU fp32 reference
  round-trips through bf16 before compare, so the passing run
  matches the kernel to bf16 resolution (max_abs = 0 at scale 0.2,
  TOL=0.05 as drift-absorbing margin).
- `ferrite_fused_qkv_rope_cache_smoke.cu` — Phase 3f part 2b-ii
  `fused_qkv_rope_cache` op (NeoX / non-biased). Llama-3.2-1B
  decode dims (HIDDEN_DIM=2048, HEAD_DIM=64, NUM_Q_HEADS=32,
  NUM_KV_HEADS=8). Grid `dim3(HEAD_DIM/2, NUM_HEADS_TOTAL)` = (32,
  48); each CTA produces one rope pair inside one head.
  NUM_PAGES=4 (act, two weight rows, cos_sin). Validates packed
  QKV GEMM + NeoX RoPE on Q/K + vLLM NHD paged KV-cache write at
  `slot_mapping[0]`. Passes extra pointer families (cos_sin_cache,
  positions, slot_mapping, key_cache, value_cache) as positional
  kernel args — the pool-ABI extension for these families lands in
  Phase 3f part 2b-iii (codegen dispatch).

Future ops add a harness here as they land.
