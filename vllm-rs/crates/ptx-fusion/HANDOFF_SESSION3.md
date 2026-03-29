# Handoff — Session 3

## The goal

Ferrite fuses CUDA kernels at compile time via escape analysis on PTX. The current
milestone is fusing `rms_norm` into the CUTLASS GEMM prologue so that normalization
and matrix multiply happen in a single kernel launch, eliminating the GMEM round-trip
for the normalized intermediate.

The full pipeline target (Phase 5G) is:
```
rms_norm → GEMM_gate_up → SiLU+mul → GEMM_down → residual_add
```
All in one launch. The current milestone is the first piece: `rms_norm → GEMM`.

When this works in production (`vllm serve` produces correct text on Qwen2.5-0.5B),
the architecture extends to the full pipeline. The `compile!` macro, `PointwiseComputation`,
and `replace_a_loads_with_inline_fn` infrastructure are all designed for this.

## Where we are

**The fused kernel is correct.** 103 unit tests + 11 GPU tests, all 0.00e0, covering
every production dimension (M=1 through M=128, N=1152, N=9728), real model weights,
real embedding data, production launch path, non-default CUDA streams, and
CachingAllocator block reuse.

**Production doesn't work.** When the fused kernel is wired into `llama.rs`, `vllm serve`
produces deterministic garbage (`"ereço八大以来..."` — same every time). The gap between
"all tests pass" and "production fails" is unsolved. `llama.rs` is reverted to the
working standard path (separate `fused_add_rms_norm_inplace` + CUTLASS GEMM).

## What the working path does (llama.rs today)

Per transformer layer, for layers 1-23 (layer 0 has no residual):
```
1. fused_add_rms_norm_inplace(hidden_states, residual, norm_weight, eps)
   → residual = residual + hidden_states
   → hidden_states = rms_norm(residual) * norm_weight
2. qkv = forward_ferrite(hidden_states)   // CUTLASS GEMM via ferrite.gemm
3. attn_output = attention(qkv, ...)
4. fused_add_rms_norm_inplace(attn_output, residual, norm_weight2, eps)
5. gate_up = forward_ferrite(attn_output)  // CUTLASS GEMM
6. activated = silu_and_mul(gate_up)
7. mlp_output = down_proj(activated)
```

This produces correct text. The ferrite CUTLASS GEMMs (flat-param, perimeter-replaced)
match cuBLAS at 0.00e0.

## What the fused path should do

Replace steps 1+2 with a single call:
```
1. add_inplace(residual, hidden_states)  // residual += hidden_states
2. qkv = launch_fused_norm_gemm(FUSED_NORM_GEMM, residual, qkv_weight, norm_weight, eps)
   // internally: rms_norm(residual) * norm_weight → GEMM → qkv
```

Same for steps 4+5 (MLP norm+GEMM).

## What was done in this session

### Three bugs fixed

**1. bf16 entry extraction.** `from_ptx()` extracted the f32 rms_norm variant. Added
`entry_hint: Option<&str>` parameter, wired through `ferrite.toml` (`entry = "bfloat16"`),
`compile!`, and `pipeline_fuse!`.

**2. PTX transplant.** Replaced ~200 lines of hand-written prologue PTX with a mechanical
transplant of the actual bf16 rms_norm kernel code. Register rename (`%f17` → `%f_rms_17`),
label rename, SMEM symbol rename. Every computational instruction (FMA, shuffle, SMEM reduce,
rsqrt) comes verbatim from the extracted kernel. Only plumbing is new (param loads, row loop,
m_tile computation).

**3. Partial tile bounds.** Row offset used `m_tile * nrows` (clamped) instead of
`m_tile * tile_m` (constant). M=65 failed, M=64 passed — the first M spanning two tiles.

### Test progression (test9a-k)

Each adds one element of production realism beyond test8:

| Test | What it adds | Result |
|------|-------------|--------|
| test9a | M=1,2,3,7,15,32,63,64,65,128 | 0.00e0 |
| test9b | Real QKV dims (N=1152), gate_up (N=9728) | 0.00e0 |
| test9c | `launch_fused_norm_gemm` (production launch) | 0.00e0 |
| test9d | GPU `add_inplace` + fused launch | 0.00e0 |
| test9e | 24-layer chain, GPU add + production launch | 0.00e0 (22 layers) |
| test9i | Non-default CUDA stream | 0.00e0 |
| test9j | Real model embedding data | 0.00e0 |
| test9k | CachingAllocator sentinel reuse | no read-before-write |

### Production wiring attempts (all failed, all reverted)

Three different approaches tried in `llama.rs`, all producing identical garbage:
1. `add_inplace` + `launch_fused_norm_gemm` for both QKV and MLP
2. `add_inplace` + `launch_fused_norm_gemm` for QKV only (MLP stays standard)
3. `fused_add_rms_norm_inplace` (standard add+norm) + `launch_fused_norm_gemm` (fused GEMM reading from residual)

In-production debug (running both standard and fused, comparing on GPU) showed diffs of
10-200. This is because `forward_ferrite` selects the 64x64x32 tile config for M=1
(`ferrite.gemm.select(m=1)` returns `configs[0]` = 64x64x32), while `FUSED_NORM_GEMM`
is compiled for 64x128x32. Different tile configs → different bf16 rounding. This is
expected and NOT the bug.

## The unsolved gap

Every test element has been validated in isolation and in combination:
- Kernel correctness: 0.00e0 at all dims, all data, all launch paths
- Thread count: 128 vs 256 produces identical results for hidden=896
- Partial tiles: bounded correctly, tested at M=1 through M=128
- Streams: works on non-default streams
- Allocator: no read-before-write on reused blocks
- Real data: actual embedding values, not synthetic

Something about the `llama.rs` runtime context breaks it. Candidates:
- How `OwnedTensor` deref/drop interacts with kernel launches on `device.compute_stream`
- Async scheduling overlap (the engine overlaps CPU scheduling with GPU execution)
- Some subtle difference in how `device.caching` allocator state evolves across layers

## What to do next

**Write a test that calls `forward_transformer_block` directly.** Load the model,
construct a GpuDevice, embed tokens, call the actual function with fused path enabled.
Compare output against standard path. This eliminates ALL gaps between test and production.
If it passes, the bug is in the engine layer. If it fails, add per-step debug inside the
function to find where divergence first appears.

**Do NOT touch llama.rs until you have a failing test.**

## Rules (non-negotiable)

- **NEVER write PTX by hand** when extracted code is available — use the transplant approach
- **NEVER build special-case macros** — extend `compile!`
- **NEVER claim "proven"** without `vllm serve` producing correct text
- **NEVER dismiss divergence** — any non-zero diff must be investigated to root cause
- **Tests ARE the product** — add realism until a test fails, fix the failing test, repeat
- **No shortcuts** — every step on the critical path to the full pipeline, not hacks for the milestone

## Key files

| File | What |
|------|------|
| `pipeline_compile.rs:506-1020` | Transplanted prologue builder |
| `pipeline_compile.rs:580-640` | `rename_ptx_regs`, `is_transplant_skip` |
| `pipeline.rs:114-170` | `from_ptx` with entry_hint |
| `compile.rs:21-28` | ManifestKernel.entry_hint |
| `lib.rs:1627-1630` | compile! passes entry_hint |
| `lib.rs:2108-2122` | pipeline_fuse! optional 4th arg |
| `ferrite.toml:31-32` | rms_norm entry = "bfloat16" |
| `ferrite_gemm_real_weights.rs:2131+` | test9a through test9k |
| `llama.rs:1004-1106` | ferrite forward path (standard, NOT fused) |
| `ferrite.rs:375-462` | `launch_fused_norm_gemm` |
| `FERRITE.md` | Architecture, roadmap, all proven capabilities |
