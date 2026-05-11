# MetalKittens Plan

Analogous to ThunderKittens (CUDA): a composable MSL primitive library that
exposes Apple Silicon MMA hardware at the simdgroup level, enabling fused
kernels that keep intermediates in registers/threadgroup memory.

## Motivation

Current state:
- `fused_affine_qkv_rope_cache.metal` EXISTS but is UNTESTED JUNK — scalar
  per-thread inner product, never validated, not wired into dispatch. Discard it.
- The standalone `quantized_qmv.metal` IS correct and tested — faithful MLX
  port of `qmv_fast_impl` / `qmv_quad_impl` with simdgroup MMA + `simd_sum`.
- There is no working fused QKV kernel. The 4-dispatch gap is fully open.

Root cause: no primitive layer. The existing `qmv_fast_impl` cooperative inner
product is buried in `quantized_qmv.metal` and can't be reused by a fused kernel.

MetalKittens extracts that primitive into a shared header so fused kernels can
compose it without reinventing or degrading it.

## What MetalKittens is

An MSL header (`metal_kittens.h`) with:

```
rt<T, Rows, Cols>   — register tile over simdgroup_matrix_storage
st<T, Rows, Cols>   — threadgroup memory tile
rv<T, Elems>        — register vector (per-simdgroup accumulator)

load(st, device_ptr, stride)    — cooperative load into threadgroup mem
store(device_ptr, st, stride)   — cooperative store
mma(rv, rt_a, rt_b)             — simdgroup_multiply_accumulate
dot_affine_int4(rv, x_st, w_packed, scales, biases, gs)  — the hot path
apply(rv, fn)                   — pointwise on register vector
mk_sync()                       — threadgroup_barrier(mem_threadgroup)
```

The SIMD group (32 threads) is the unit of work. A threadgroup is a group of
SIMD groups cooperating on a tile.

## What we are NOT doing

- No async tile copy (Metal has no cp.async equivalent; simdgroup_event not
  public API).
- No producer/consumer warp specialization (same reason).
- No new dispatch mechanism. ICB and the bucket system are unchanged.
- No new Rust crate. This is a pure MSL header that existing shaders include.

## Phases

### P0: Build metal_kittens.h (simdgroup GEMV primitive)

The single most important primitive: cooperative int4 GEMV for a single output
row, matching what MLX's `affine_qmv_quad_impl` does.

MLX structure (from quantized.h lines ~692-800):
- 32 threads per simdgroup, each thread handles `values_per_thread = K/SIMD_SIZE`
  K elements
- Each thread loads its slice of the input vector into registers via
  `load_vector<T, U, values_per_thread, bits>`
- Each thread accumulates a partial dot product against its K-slice of the
  weight row
- Simdgroup reduce: `result = simd_sum(acc)` — hardware SIMD reduction

MK wraps this as:

```metal
// Cooperative int4 dot product: 32 threads → one float output element
// x_thread: caller's slice of the input vector (values_per_thread elements)
// w_packed: pointer to this row's packed nibbles
// scales/biases: pointer to this row's scale/bias pairs
template<int GROUP_SIZE, int VALUES_PER_THREAD>
inline float mk_affine_dot(
    thread float* x_thread,          // pre-loaded by caller via load_vector
    const device uint* w_packed,     // row start
    const device T_scale* scales,
    const device T_scale* biases,
    ushort simd_lane_id)
```

The caller wraps multiple `mk_affine_dot` calls (one per output head/dim) in a
threadgroup that has SIMD_SIZE * num_output_rows threads.

### P1: Validate — port affine_qmv_quad to MK

Rewrite `quantized_qmv.metal`'s `affine_qmv_quad` entry point using
`mk_affine_dot`. Output must be bit-identical to the current MLX port. Measure
perf: must match or exceed current.

This is purely a shader refactor. No Rust changes. No dispatch changes.
Gate: passes all existing cpu_golden tests + timing within 2% of current qmv.

### P2: Write fused_affine_qkv_rope_cache.metal from scratch using MK

Discard the untested existing file. Write from scratch using `mk_affine_dot`.
Reference: the working `qmv_fast_impl` in `quantized_qmv.metal` for the inner
product; `fused_qkv_rope_cache.metal`'s RoPE + cache-write epilogue for the
Q/K/V band logic.

Dispatch shape changes:
- Current: `(M, NUM_Q + 2*NUM_KV, 1)` × `(HEAD_DIM, 1, 1)` — 1 thread/dim
- MK: `(M, NUM_Q + 2*NUM_KV, 1)` × `(SIMD_SIZE * ceil(HEAD_DIM/TG_ROWS), 1, 1)`
  — 32 threads cooperating per output row, multiple rows per threadgroup

The RoPE + cache-write epilogue is unchanged — it operates on the accumulated
output in registers after the dot product, before writing to device memory.

Gate: coherent output on Llama-3.2-1B-4bit + 3B-4bit, timing ≤ standalone
3×AffineQmv + RopeAppend dispatches.

### P3: Wire into Rust dispatch (lowering.rs + pipelines.rs)

The shader exists and is correct. Wire it in:

1. `ferrite-forward/src/interpreter/metal/lowering.rs` — add fusion pattern:
   detect `[AffineQmv(Q), AffineQmv(K), AffineQmv(V), RopeAppend]` in the DAG
   → emit single `LoweredCommand::FusedAffineQkvRopeCache`.

2. `ferrite-forward/src/interpreter/metal/pipelines.rs` — add
   `KernelId::FusedAffineQkvRopeCache` arm in `lower_one` with the 8
   function constants.

3. `instruction_executor/fused.rs` — add recorder that binds the 10 buffers.

Gate: Llama-3.2-1B-4bit decode tok/s ≥ current (no regression). Ideally
matches MLX within 5% (from 87% → ≥95% of MLX).

### P4: RMSNorm fusion

Fuse pre-RMSNorm into the QKV kernel. The RMSNorm output
([M, hidden_size]) fits in threadgroup memory at 3072 * 2 = 6KB for 3B
(threadgroup limit 32KB on M1, 64KB on M4 — headroom fine).

Pattern in DAG: `[RmsNorm, FusedAffineQkvRopeCache]` → single dispatch.

The reduction (mean of squares) already uses threadgroup memory internally in
`rmsnorm.metal`. In the fused version, the normalized + scaled vector lives in
`st<bf16, 1, HIDDEN>` and feeds directly into the Q/K/V dot products.

Bandwidth saved: 2 × HIDDEN × sizeof(T) per layer (one write + one read of
the hidden state eliminated). For 3B: 2 × 3072 × 2 = 12KB × 28 layers ≈
336KB/forward — small but worth having for free.

## File layout

```
shaders/
  metal_kittens.h                    ← NEW (P0)
  quantized_qmv.metal                ← refactored to use MK (P1, optional)
  fused_affine_qkv_rope_cache.metal  ← REWRITTEN with mk_affine_dot (P2)
  fused_qkv_rope_cache.metal         ← BF16 variant rewritten (P4)
```

No new shaders. No new Rust crates.

## What does NOT change

- ICB / bucket dispatch architecture
- Cost sweep / profile system
- SpecializedPipelineCache
- Everything outside ferrite-metal-kernels and the lowering/pipelines glue

## Perf target

P0–P3 together: Llama-3.2-3B-4bit decode ≥ 50 tok/s (from 45.7), closing the
gap from 87% → ~95% of MLX. The remaining ~5% is expected to be the matmul
(attention + output projection) and long-range ops not covered by this fusion.

## Decision log

- `inherit_pipeline_state=false` not needed: each MK fused kernel is a single
  pipeline state. The multi-pipeline-range split in ICB was a workaround for
  per-pipeline grouping; a fused kernel eliminates the split.
- No `simdgroup_event`: blocked (not public API). Async double-buffering is
  out of scope for this plan.
- No new dispatch mechanism: the ICB argument-caching benefit (70-80µs/forward)
  is preserved exactly as-is.
