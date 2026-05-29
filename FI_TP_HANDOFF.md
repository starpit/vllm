# FlashInfer at TP>1 — investigation handoff

Branch: `worktree-rm-old-cuda` · Tip: `5e9f8108c9`
Pod: `nick3` (2× L40S sm_89) · Path: `/root/rm-old-cuda/vllm-rs/`

## What's broken

FlashInfer's `BatchPagedAttentionPersistent` kernel returns rc=-1 on
L40S sm_89 at TP=2 decode geometry for Llama-3.2-1B. Symptoms:

1. **`fi_run` rc=-1** — the kernel launch returns non-`cudaSuccess`.
   Lives at the `flash_attn::run` host wrapper inside the
   ferrite-cuda-builder-emitted shim
   (`flashinfer_shim.cu.j2:579-589`).
2. **CUDA context poisoned** — once `fi_run` errors, every subsequent
   kernel on that context (cutlass split-K, FA2 memset, plain H2D
   memcpy) errors with `CUDA_ERROR_ILLEGAL_ADDRESS`. This is the
   `H2D u32: CUDA_ERROR_ILLEGAL_ADDRESS` symptom that surfaces in
   the user's curl request.
3. **`fi_plan_new` rc=-1 inside graph capture** — `cudaMalloc` inside
   `cuStreamBeginCapture` is forbidden, so the FI plan's float_ws_d /
   int_ws_d allocations fail when the first FI call lands inside a
   captured stream. Workaround: ensure `fi_plan_new` runs OUTSIDE
   capture (warmup forward with `SuppressNcclGuard` does this).

## Current workaround (committed)

The two callers of `flashinfer_attention` in
`ferrite-forward/src/instr.rs` (the `FlashInferAttentionDecode` and
`FlashInferAttentionPrefill` arms of `Instruction::eval`) gate on
`ctx.fwd.tp_group.is_some()`: when a TP group is attached, skip FI
and fall through to the existing FA2 path that's already wired in
the same arm. TP=1 (no `tp_group`) keeps the FI fast path. Set
`FERRITE_USE_FLASHINFER=1` to override the gate for testing.

This costs throughput at TP>1 for long-sk decode bf16 (FI's
strongest workload), but it's the only way to keep TP=1 fast while
keeping TP=2 from crashing.

## What we know

### Geometry
At TP=2 for Llama-3.2-1B (per-rank, after sharding):
- `num_qo_heads = 16` (= 32 / 2)
- `num_kv_heads = 4`  (= 8 / 2)
- `head_dim = 64`
- `page_size = 16`
- GQA ratio (qo/kv) = 4 (same as TP=1)

At TP=1 for the same model:
- `num_qo_heads = 32`, `num_kv_heads = 8` — same kernel template
  (`BatchPagedAttentionPersistent<CTA_TILE_Q_1=128, CTA_TILE_Q_2=16,
  HEAD_DIM_QK=64, HEAD_DIM_VO=64, MaskMode::kCausal, …>`) succeeds.

### What does NOT differ between TP=1 and TP=2
- Compiled kernel binary (the FI dispatch tuple is keyed by
  `cfg = (dtype, head_dim, use_logits_soft_cap)` — identical at both).
- `target_num_clusters` (= `num_sm`) — both ranks see the same SM count.
- `float_ws_bytes` / `int_ws_bytes` — derived from `(num_sm, head_dim,
  num_kv_heads)`; differs because of `num_kv_heads`. **Worth probing.**
- `kv_indices` and `seqused_k` shapes — driven by batch size, not TP.

### What DOES differ
- `num_qo_heads` and `num_kv_heads` (sharded by TP).
- The plan-time `target_num_clusters` may interact with per-rank head
  counts inside `TwoStageHolisticPlanWithNumSm` — that's the planner
  failure surface.
- Per-rank input addresses are different (each rank's GpuTensor
  pointers come from its own private allocator pool).

## Empirical observations

`fi_plan_new` itself **succeeds** at TP=2 when called outside graph
capture (warmup builds the plan cleanly). The plan struct is
allocated, `TwoStageHolisticPlanWithNumSm` returns `cudaSuccess`,
`PersistentParams` are populated, handle returned non-null. So the
**planner accepts the geometry**.

Failure is at `fi_run` time — the actual kernel dispatch
`flashinfer::BatchPagedAttentionPersistent<…>(plan->params_1,
plan->params_2, plan->num_blks_x, plan->num_blks_y, stream)`
returns non-`cudaSuccess`. The shim collapses this to rc=-1 without
preserving `cudaGetErrorString(st)`.

## Investigation directions

### 1. Recover the actual cuda error string

Patch `flashinfer_shim.cu.j2:579-589` to return `cudaGetErrorString(st)`
or at least convert the `cudaError_t` value to its named code (e.g.
`cudaErrorIllegalAddress = 700`, `cudaErrorLaunchOutOfResources = 701`,
`cudaErrorInvalidConfiguration = 9`). Right now we squash all errors
to -1 and lose the discriminator.

```diff
-extern "C" int32_t fi_run_{{ sym_suffix }}(void* handle, cudaStream_t stream) {
-  ...
-  return st == cudaSuccess ? 0 : -1;
-}
+extern "C" int32_t fi_run_{{ sym_suffix }}(void* handle, cudaStream_t stream) {
+  ...
+  if (st != cudaSuccess) {
+    std::fprintf(stderr,
+      "fi_run_{{ sym_suffix }}: kernel launch failed: %s (cudaError=%d)\n",
+      cudaGetErrorString(st), (int)st);
+  }
+  return st == cudaSuccess ? 0 : -(int32_t)st;
+}
```

That alone tells us if it's a launch-time resource limit, an invalid
config (block dim out of range), or post-launch illegal address.

### 2. Probe `num_blks_x` / `num_blks_y`

Print `plan->num_blks_x`, `plan->num_blks_y`, and
`plan_info.len_kv_chunk_offset`-style fields right after `plan_new`
succeeds. Compare TP=1 vs TP=2 values. A degenerate launch grid
(e.g. one dim is 0 or > 65535) is a likely culprit given the
per-rank `num_qo_heads = 16` may produce a smaller/larger block count
than the planner is configured for.

### 3. Check `int_ws_bytes` / `float_ws_bytes` sizing

`flashinfer::workspace_bytes(num_sm, head_dim, num_kv_heads)` returns
the buffer sizes the kernel reads from. At TP=2 `num_kv_heads = 4`
shrinks the int_ws scratch — if the kernel was compiled assuming a
larger workspace and indexes past the actual allocation, that's
exactly the failure shape.

Cross-reference `workspace_bytes()` (in
`ferrite-kernels/src/flashinfer.rs`) against
`TwoStageHolisticPlanWithNumSm`'s actual `int_ws` indexing inside the
FI source.

### 4. Verify the kernel binary against L40S sm_89

cudaforge compiles FI shims per-target. Confirm the kernel was
emitted with `-arch=sm_89` (Ada) and not `-arch=sm_80` (Ampere) or
`-arch=sm_90` (Hopper). Mismatch produces silent JIT failures or
register-pressure errors at runtime on L40S.

```bash
oc rsh nick3 cuobjdump --list-text \
  /root/.cache/cudaforge/vllm-cuda/<fi-shim>.so | head
```

### 5. Compare against the known-good handoff state

`PIECEWISE_HANDOFF.md` claims TP=2 eager decode produced coherent
text on this same branch + pod. Either:
- Pod state changed (driver / cudaforge cache regenerated against
  different toolkit), OR
- The handoff author tested at a different commit.

Check `git log -- vllm-rs/crates/ferrite-cuda-builder/templates/`
and the cudaforge cache mtimes:

```bash
oc rsh nick3 ls -la /root/.cache/cudaforge/vllm-cuda/ | sort -k6
```

If the cache was rebuilt recently, force-regenerate and bisect.

### 6. Try a non-Persistent kernel variant

FI has a non-persistent paged attention path. `BatchPagedAttention*`
non-persistent vs `BatchPagedAttentionPersistent`. If the persistent
variant has stricter sm_89 requirements, the non-persistent may
work. Wire it up as a fallback inside `flashinfer_attention` when
`fi_run` returns specific cuda errors.

## Files to read

- `vllm-rs/crates/ferrite-cuda-builder/templates/flashinfer_shim.cu.j2`
  — the host shim. `fi_plan_*_new` (line 348) and `fi_run_*` (line 579)
  are where errors get squashed.
- `vllm-rs/crates/ferrite-kernels/src/flashinfer.rs` —
  `FlashInferPlanCache` (line 312) + `dispatch_for(cfg)` (selects the
  compiled tuple). `workspace_bytes` lives near the top.
- `vllm-rs/crates/ferrite-kernels/src/attention_helpers.rs:805-911` —
  the `flashinfer_attention` host wrapper. The current short-circuit
  to None lives at the top of this fn.
- `vllm-rs/crates/ferrite-forward/src/instr.rs:2326-2397` —
  `Instruction::FlashInferAttentionDecode::eval`. The fallback to
  `attention_decode_from_cache` (FA2) when FI returns None.

## Reproducer

```bash
# On nick3, with FI re-enabled:
cd /root/rm-old-cuda/vllm-rs
FERRITE_MODELS=llama-3.2-1b cargo build --release \
    -p vllm-cli --features cuda,nccl
ps -ef | grep "release/vllm" | grep -v grep | \
    awk '{print $2}' | xargs -r kill -9 ; sleep 3
CUDA_LAUNCH_BLOCKING=1 FERRITE_USE_FLASHINFER=1 \
    ./target/release/vllm serve \
    --model unsloth/Llama-3.2-1B-Instruct \
    --tensor-parallel-size 2 --port 18200 \
    --max-num-batched-tokens 1024 --enforce-eager 2>&1 | head -100
```

Expected failure (current state):

```
ERROR fi_run returned nonzero — falling back to FA2 rc=-1
```

After the FA2 fallback path also fails (because the cuda context is
already poisoned by the failed FI launch), the curl request returns
`H2D u32: CUDA_ERROR_ILLEGAL_ADDRESS`.

## What's NOT this bug

- The piecewise capture+replay path itself is fine. Once FI is
  short-circuited (current default), TP=2 piecewise decode produces
  coherent text. The committed code in `5e9f8108c9` is correct.
- `cutlass_gemm_splitk` returning -3 for `M=4 N=2048 K=1024` is a
  *consequence* of the poisoned context, not an independent bug —
  it stops happening once FI is bypassed.
- NCCL inside CUDA graphs failing with `ILLEGAL_ADDRESS` is the
  motivation for piecewise itself, not this FI issue. Different
  symptom path.

## When this is fixed

Remove the `tp_group.is_some()` gate from both
`Instruction::FlashInferAttentionDecode::eval` and
`Instruction::FlashInferAttentionPrefill::eval` in
`crates/ferrite-forward/src/instr.rs`. Keep the
`fi_run rc != 0 → return None` fallback inside `flashinfer_attention`
— that's defensive against future regressions and costs nothing on
the happy path. Drop the `FERRITE_USE_FLASHINFER` env var.
