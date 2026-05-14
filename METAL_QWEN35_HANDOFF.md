# Metal Qwen3-MoE / Qwen3-Next port — handoff

**Branch:** `worktree-fm-qwen35` (off `main`)
**Commit:** these two commits on top of `702db8e16`:
- one main commit landing kernels + the I::SharedFusedMoe lowering arm
- one droppable debug commit adding `FERRITE_TRY_LOAD_DEBUG` instrumentation in `crates/ferrite-forward/src/lib.rs::try_load`. Drop after debugging the fingerprint miss below.

**Validation target:** `mlx-community/Qwen3-30B-A3B-4bit` (HF cache, 4 shards, ≈ 16 GB on disk).

**Hardware target:** Apple Silicon (verified on a 24 GiB M-class Mac). Per [machine memory cap doc] the engine sizes the KV cache against 24 GiB minus model weights; the 30B-A3B-4bit checkpoint plus 235 920-token KV pool fits with ~16 s load time.

**Status:** code compiles clean on `cargo check --bin vllm -F metal`; new kernel parity tests all pass on this machine. **End-to-end NOT verified** — `vllm chat` on the 30B-A3B-4bit checkpoint still falls back to the MLX backend with `ferrite-metal has no variant for arch Qwen3MoeForCausalLM`. The compiled-in variant `qwen3_moe / qwen3-30b-a3b-instruct-mlx-affine-b4-g64` is present (`./target/release/vllm ferrite info qwen3 moe` lists it), so `try_load`'s fingerprint sniff is the next thing to debug. Reproduce + diagnosis steps below.

---

## TL;DR

1. **Two new Metal kernels** (`slice_trailing_cols_u32`, `top_k_renormalize`) plus six already-on-the-branch kernels from prior sessions (`softmax`, `argpartition`, `take_along_axis`, `row_gather`, `moe_weighted_sum`, `affine_gather_qmv[_fast]`).
2. **`I::SharedFusedMoe` lowering arm** in `crates/ferrite-forward/src/interpreter/metal/lowering.rs::lower_one`, decomposing one MoE block into 11 `LoweredCommand`s.
3. **GEMM dispatch path extended** to accept `Binding::MoeScratch` outputs (Issue A from the prior `project_metal_moe_router_kernels` memory).
4. **`ferrite-models` `metal` feature now forwards to `ferrite-model-qwen3-moe`**, and `ferrite-model-qwen3-moe/configs/quantizations.json` now lists `mlx-affine-b4-g64` so the macro synthesizes a 4 bit variant.
5. **Open bug**: at runtime the `qwen3-30b-a3b-instruct-mlx-affine-b4-g64` variant's `fingerprint_matches` rejects the live checkpoint; engine falls back to MLX. The debug instrumentation lands together so you can `FERRITE_TRY_LOAD_DEBUG=1 ./target/release/vllm chat ...` and see which step bails.

---

## Working environment

```bash
# Worktree
cd /Users/moosevan/git/vllm/.claude/worktrees/fm-qwen35   # or wherever you cloned it
git checkout worktree-fm-qwen35

# Build (release, Metal feature; from vllm-rs/)
cd vllm-rs
cargo build --release --bin vllm -F metal
# Debug-profile sanity check (fast):
cargo check --bin vllm -F metal

# Run the new kernel parity tests
cargo test -p ferrite-metal-kernels --test slice_trailing_cols_u32_test --test top_k_renormalize_test \
    --test softmax_test --test argpartition_test --test take_along_axis_test \
    --test row_gather_test --test moe_weighted_sum_test --test affine_gather_qmv_test
```

The Metal toolchain (`xcrun -find metal` / `xcrun -find metallib`) must be present. `cargo build` invokes `crates/ferrite-metal-kernels/build.rs`, which compiles every `shaders/*.metal` file to a per-library `.metallib` under `OUT_DIR` (so cold startup skips the MSL→AIR pass).

The 30B model is fetched on demand:
```bash
./target/release/vllm pull mlx-community/Qwen3-30B-A3B-4bit
```

`huggingface-cli` is not required; `vllm pull` resolves through the HF Hub directly. The local cache lands at `~/.cache/huggingface/hub/models--mlx-community--Qwen3-30B-A3B-4bit/`.

---

## What landed (commit 1 — main work)

### A. New Metal kernels (this session)

Two new shaders + dispatchers + parity tests. The other six kernels listed earlier in this section were already on the branch from prior sessions but are included in commit 1 because they were still uncommitted at session start.

**`shaders/slice_trailing_cols_u32.metal` + `src/slice_trailing_cols_u32.rs`**:
```
dst[n, k] = src[n, src_cols - dst_cols + k]
```
Materializes the `[..., -k:]` slice MLX expresses as a strided view at `qwen3_moe.py:131`. Single 2D dispatch grid `(top_k, batch)`. Tests:
- `slice_trailing_cols_qwen3_moe_router_shape` — N=8 src_cols=128 dst_cols=8 (matches Qwen3-MoE router).
- `slice_trailing_cols_dst_equals_src` — identity case (`dst_cols == src_cols`).

**`shaders/top_k_renormalize.metal` + `src/top_k_renormalize.rs`**:
```
scores[n, k] /= sum(scores[n, :])
```
In-place row L1 renormalize. Implements `norm_topk_prob` from `qwen3_moe.py:134`. `TKR_TOP_K` is a `function_constant(0)` so the row loop unrolls. One thread per row.

Both kernels register through the standard pipeline:
- `crates/ferrite-metal-kernels/src/lib.rs` — `pub mod slice_trailing_cols_u32; pub mod top_k_renormalize;`
- `crates/ferrite-metal-kernels/src/shader_cache.rs` — adds both to the `(name, bytes)` table and the `library_for` prefix dispatch.
- `crates/ferrite-metal-kernels/src/specialized_pipeline_cache.rs` — same.

### B. Already-uncommitted kernels from prior sessions

These shaders/dispatchers/tests pre-existed on the worktree as `??` files at session start. The handoff memory `project_metal_moe_router_kernels` covers them in detail. Summary:

| File group | Purpose | MLX provenance |
|---|---|---|
| `softmax.metal` + `src/softmax.rs` | row-wise softmax (precise variant) for router probs | `mlx/.../softmax.metal` |
| `argpartition.metal` + `src/argpartition.rs` | single-block argsort, dtype × bn ∈ {32,64,128} | `mlx/.../sort/sort.cpp` |
| `take_along_axis.metal` + `src/take_along_axis.rs` | 2D contiguous gather, axis=-1 | `mlx/.../indexing/gather_axis.h` |
| `row_gather.metal` + `src/row_gather.rs` | `out[m,d] = src[idx[m]/divisor, d]` (SwitchGLU `_gather_sort` / `_scatter_unsort`) | `mlx-lm/.../switch_layers.py:12` |
| `moe_weighted_sum.metal` + `src/moe_weighted_sum.rs` | fused `(expert_out * scores).sum(axis=-2)` | `qwen3_moe.py:137` |
| `quantized_qmv.metal::affine_gather_qmv[_fast]` (extension to existing file) | MoE per-expert matvec, transpose=true | `mlx/.../quantized.h:1900` |

`MetalSwitchGluMoeWeights` (in `crates/ferrite-kernels/src/layers_moe.rs`) loads the on-disk 4 bit affine SwitchGLU weights for `{prefix}.gate`, `{prefix}.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}`, and the optional Qwen3-Next shared expert.

### C. `I::SharedFusedMoe` lowering arm

In `crates/ferrite-forward/src/interpreter/metal/lowering.rs::lower_one`, between the `I::ScalarMul` and `I::Reshape` arms. Decomposes one MoE block into 11 `LoweredCommand`s. Pattern:

```
 step  kernel                     in                          out                       notes
 a     Gemm (router)              x [N, hidden]              MoeScratch.router         m=N, n=num_experts, k=hidden
 b     Softmax (precise)          MoeScratch.router          MoeScratch.router         in-place; axis_size via Binding::Inline(2)
 c     ArgPartitionTopK           MoeScratch.router          MoeScratch.argpart        full sort u32 [N, num_experts]
 d     SliceTrailingColsU32       MoeScratch.argpart         MoeScratch.indices        [N, top_k] u32
 e     TakeAlongAxis              MoeScratch.router + idx    MoeScratch.scores         [N, top_k] T_act
 f     TopKRenormalize            MoeScratch.scores          MoeScratch.scores         in-place sum-normalize
 g     AffineGatherQmv (gate)     x + indices                MoeScratch.router         reuses region_router after step b is consumed
 h     AffineGatherQmv (up)       x + indices                MoeScratch.up
 i     SiluMul                    .router (gate) + .up       MoeScratch.up             in-place
 j     AffineGatherQmv (down)     .up + indices              MoeScratch.router         **top_k=1 (no row broadcast — each (token, slot) has its own activation)**
 k     MoeWeightedSum             .router + .scores          ArenaSlot[out_slot]       Σ_k(expert[n,k,d] * scores[n,k])
```

**Critical detail on step (j)**: `affine_gather_qmv` indexes `x` by `row / top_k` (it expects `x` to be `[N, IN]` and broadcasts to `[N*top_k, IN]`). For `down_proj` each `(token, slot)` row has its own activation in `region_up`, so the arm passes `top_k=1` at `buffer(6)` and the kernel's divider collapses to identity. `rhs_indices` (binding 4) is still the full `region_indices` flat layout `[N*top_k]`.

MoE scratch is laid out as five 256-byte-aligned regions:
```
[ argpart u32 | router T_act | up T_act | indices u32 | scores T_act ]
```
`region_router` is sized to the max usage across the kernel chain (router probs → gate_out → expert_out=down_out): `max(N*num_experts*e, N*top_k*moe_inter*e, N*top_k*hidden*e)`.

Helpers added at the bottom of `lowering.rs`:
- `align_256(x: u32) -> u32`
- `softmax_tg_size(axis_size: u32) -> u32` (mirrors `ferrite_metal_kernels::softmax::softmax_tg_size`)
- `arg_sort_bn(axis_size: u32) -> u32` (mirrors `argpartition::arg_sort_bn`; panics for axis_size > 512 — multi-block sort not ported)
- `pick_affine_gather_qmv(dtype, scale, gs, k_in)` → `(KernelId, &'static str)`, returns `_fast` variant when `k_in % 512 == 0`.

The arm `assert!`s that `m.shared_expert_gate.is_none()` — Qwen3-Next's shared-expert tail is **not yet wired**. Qwen3-MoE has no shared expert (config `shared_expert_intermediate_size: 0` everywhere), so this is fine for the Phase F target but blocks Qwen3-Next.

### D. GEMM dispatch + MoE scratch threading

`crates/ferrite-forward/src/interpreter/metal/worker.rs`:
- `resolve_gemm_buffers` gained a `moe_scratch: Option<&Buffer>` parameter and now passes it through to `resolve_bindings` (so the router GEMM in step (a) can bind its output via `Binding::MoeScratch`).
- Both the f16 MPS path (`BucketStep::Gemm` via `MPSMatrixMultiplication`) and the bf16 path (`gemm_bf16_specialized` direct ICB binding) honor `BoundBuffer { buffer, offset }` natively — no further plumbing needed.

`crates/ferrite-forward/src/interpreter/metal/lowered.rs` (already on the branch but listed for completeness):
- `KernelId::SharedFusedMoe`-relevant variants added: `Softmax`, `ArgPartitionTopK`, `TakeAlongAxis`, `AffineGatherQmvFast`, `AffineGatherQmv`, `RowGather`, `MoeWeightedSum`, `SliceTrailingColsU32`, `TopKRenormalize`.
- `Binding::MoeScratch { binding_index, byte_offset }` — region inside the worker's shared MoE scratch buffer.
- `Binding::Inline { binding_index, value: u32 }` — for `constant int& [[buffer(N)]]` style scalar params under ICB (the worker allocates a tiny 4-byte MTLBuffer at bake time).
- `WeightBundleKind::SharedFusedMoe(WtFn<W, ferrite_kernels::layers_moe::SharedFusedMoELayer>)`.
- 17 new `WeightTensor` variants for the MoE bundle (`MoeRouterGate`, `MoeExpert{Gate,Up,Down}{W,S,B}`, `MoeShared{Gate,Up,Down}{W,S,B}`, `MoeSharedExpertGate`).
- `LoweredMetalTape::moe_scratch_bytes: u32`.

### E. Registration / feature wiring

- `crates/ferrite-models/Cargo.toml`: `metal` feature now forwards to `ferrite-model-qwen3-moe?/metal` (previously qwen3-moe was cuda-only at the workspace umbrella level).
- `crates/ferrite-model-qwen3-moe/configs/quantizations.json`: now lists `mlx-affine-b4-g64` so the macro synthesizes a 4 bit variant. Before this change `quantizations.json` was `[]` and the only compiled variant was the BF16 unquantized `qwen3-30b-a3b-instruct`, which can't match the live 4 bit checkpoint's fingerprint.

After these two edits, `./target/release/vllm ferrite info qwen3 moe` lists:
```
══ qwen3_moe / qwen3-30b-a3b-instruct · tp=1 ══
══ qwen3_moe / qwen3-30b-a3b-instruct-mlx-affine-b4-g64 · tp=1 ══
══ qwen3_moe / qwen3-moe-1-layer · tp=1 ══
══ qwen3_moe / qwen3-moe-1-layer-mlx-affine-b4-g64 · tp=1 ══
```
The `-mlx-affine-b4-g64` variant is the one the engine should pick at runtime.

---

## What landed (commit 2 — debug instrumentation)

`crates/ferrite-forward/src/lib.rs::try_load` gains an `FERRITE_TRY_LOAD_DEBUG=1` env-gated print of:
- the `arch_hint` and requested `tp_world_size`
- every `FerriteArchRegistration` in the inventory (`arch_name`, `hf_arches`, `tp_world_size`, match-or-not)
- per-claimant walk result (`Ok(Some(weights))` / `Err(_)` / `Ok(None) — no variant claimed`)

This is a **droppable** commit — remove after the fingerprint root-cause is identified.

---

## Open bug: try_load returns `Ok(None)` for `Qwen3MoeForCausalLM`

### Symptoms

```
$ RUST_LOG=info ./target/release/vllm chat mlx-community/Qwen3-30B-A3B-4bit --device metal --quick "hi"
...
INFO FerriteWorker(metal): parsed hf_config in 38.75µs (arch = Qwen3MoeForCausalLM)
INFO Loading 4 shards in parallel
INFO ferrite-metal has no variant for arch `Qwen3MoeForCausalLM` — falling back to MLX backend
INFO Using MLX backend (Apple Silicon GPU)
```

The engine reaches `ferrite-metal try_load`, gets back `Ok(None)`, and the init code (`crates/vllm-serve/src/init.rs:278`) treats that as `ArchNotSupported(arch)` and falls back to MLX. The MLX run then loads the checkpoint successfully — which proves the model files themselves are intact and the issue is on the ferrite side.

### What we know

`vllm ferrite info qwen3 moe` shows the synthesized `qwen3-30b-a3b-instruct-mlx-affine-b4-g64` variant is compiled in (and its DSL contains the `SharedFusedMoe` op the new lowering arm handles). So one of two things happens at runtime:

1. The macro-generated `inventory::submit!` block doesn't actually fire under `--features metal` for this crate. Unlikely — the `#[cfg(any(feature = "cuda", feature = "metal"))]` gate at `crates/ferrite-forward-macro/src/lib.rs:1628` is the same one Qwen3 / Llama use successfully.
2. The submission fires but every variant's `fingerprint_matches(gw, hf)` returns `false`. **Most likely.**

The model's tensor manifest (from `model.safetensors.index.json`) is consistent with the `mlx-affine-b4-g64` variant signature:
- `model.embed_tokens.weight` packed u32 (alongside `.scales` and `.biases` siblings)
- per-layer `self_attn.q_proj.{weight,scales,biases}` triples
- per-layer `mlp.gate.{weight,scales,biases}` — **router gate is quantized too** (see "Likely real-world bug #2" below)
- per-layer `mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}` triples

Spot-check: hidden_size=2048, bits=4 ⇒ pack_factor=8 ⇒ embed_tokens.weight expected shape `[151936, 256]`. That's what the `mlx-affine-b4-g64` variant's fingerprint should require (see `crates/ferrite-forward-macro/src/codegen.rs:2102-2110`).

### Reproduce on a fresh machine

```bash
git checkout worktree-fm-qwen35
cd vllm-rs
cargo build --release --bin vllm -F metal
./target/release/vllm pull mlx-community/Qwen3-30B-A3B-4bit   # ≈ 16 GB
FERRITE_TRY_LOAD_DEBUG=1 RUST_LOG=info ./target/release/vllm chat \
    mlx-community/Qwen3-30B-A3B-4bit --device metal --quick "hi" 2>&1 | head -200
```

The `FERRITE_TRY_LOAD_DEBUG` output will show:
- whether a registration with `arch_name = "qwen3_moe"` is actually in the inventory.
- whether the dispatcher walked it (`match=true`).
- the per-claimant verdict (`Ok(None)` means the inner `Weights::load` walked every variant and none's `fingerprint_matches` returned true).

If you see `match=false` for every qwen3_moe registration ⇒ the inventory submit didn't fire (look at `crates/ferrite-forward-macro/src/lib.rs:1623` cfg gating). If you see `match=true` ⇒ the per-variant `fingerprint_matches` is the culprit.

### Diagnosis next step

Once you can confirm we got into `Weights::load` but no `fingerprint_matches` returned `true`, instrument that next. Quickest path:

1. Add an `eprintln!` per early-return inside the generated `fingerprint_matches` body at `crates/ferrite-forward-macro/src/codegen.rs:2119`, gated on `FERRITE_FINGERPRINT_DEBUG`. Each early-return is captioned in source — print the caption + the relevant tensor name / shape we just checked.
2. Rebuild release, rerun. The first false-returning check tells you which gate to fix.

Likely suspects, in priority order:
- **Embed shape check (line 2123)**: variant expects `shape == [vocab_lit, embed_hidden_lit]`. For the affine variant `embed_hidden_lit = hidden_size / pack_factor = 2048 / 8 = 256`. Verify the on-disk `model.embed_tokens.weight` shape (probably `[151936, 256]` but the safetensors `dtype` may be `U32` rather than the elem dtype the manifest expects).
- **`positive_candidate_tensor_refs` (line 2130)**: variant requires one of `model.layers.{l}.self_attn.q_proj.weight` for `l` in `{0, mid, last, ...}`. The live model has these.
- **`one_past_tensor` / `opposite_tensor` (lines 2133, 2136)**: `one_past_tensor = "model.layers.48.self_attn.q_proj.weight"` (with `last_layer = num_hidden_layers = 48`); should be absent. `opposite_tensor = "model.layers.0.self_attn.q_proj.qweight"`; should also be absent (the affine variant uses `.weight`).
- **`fp8_marker_tensor_refs` (line 2154)**: rejects when `model.layers.{l}.self_attn.q_proj.weight_scale` is present at any of the sampled layers. The live model has no `.weight_scale` — should be fine.

### Likely real-world bug #2 (after the fingerprint passes)

Even once the fingerprint is fixed, `SharedFusedMoELayer::load` (metal impl, `crates/ferrite-kernels/src/layers_moe.rs:919`) loads `{prefix}.gate.weight` as **dense** T_act. The on-disk checkpoint stores the router gate as 4 bit affine: `mlp.gate.{weight,scales,biases}`. The router GEMM in the lowering arm (step a) is dispatched through the dense MPS / `gemm_bf16_specialized` path and will see u32-packed weights, producing nonsense logits.

Fix options when you get there:
1. Dequant the router gate at load time (one-shot CPU/GPU dequant into a dense T_act buffer). Cheap; gate is `[num_experts=128, hidden=2048] = 256 KB` after dequant.
2. Replace the router GEMM with an `affine_qmv` dispatch. Requires the existing dense-GEMM-output → arena slot routing in step (a) to be extended to MoE-scratch outputs for `KernelId::AffineQmv` (analogous to the `KernelId::Gemm` change in commit 1).

Option 1 is the smaller change; do it first to keep the lowering arm shape stable.

---

## Reference docs / context this depends on

These are the only files outside the worktree you need to read to make progress. **Do not** try to recreate them; they are existing reference material in this repo or the user's environment.

### Inside this repo

- `crates/ferrite-forward/src/interpreter/metal/lowering.rs` — the file you just edited. The arm itself is at `lower_one()` `I::SharedFusedMoe(...)`; the helpers are at the bottom of the file. The existing arms above are good shape examples (look at `I::AffineQmm`, `I::FusedGateUpSiluMul`, `I::SiluMul`).
- `crates/ferrite-forward/src/interpreter/metal/lowered.rs` — `KernelId`, `Binding`, `WeightBundleKind`, `WeightTensor`, `LoweredCommand`, `LoweredMetalTape`. All vocabulary the lowering pass and worker speak.
- `crates/ferrite-forward/src/interpreter/metal/worker.rs` — `bake_bucket`, `resolve_bindings`, `resolve_gemm_buffers`, `resolve_weight`. Resolves the lowered tape against the live `MetalAllocator` arenas at worker init time.
- `crates/ferrite-kernels/src/layers_moe.rs` — `SharedFusedMoELayer` (cuda + metal `load` impls), `MetalSwitchGluMoeWeights`. Lines 877-1080 are the metal-only `load` (4 bit MLX-affine SwitchGLU loader).
- `crates/ferrite-forward-macro/src/impl_lib.rs:16585-16916` — `SharedFusedMoeRefImpl` (claims the `OpKind::Moe` tile and emits `Instruction::SharedFusedMoe`). Registered in the Metal starter library at line 2113.
- `crates/ferrite-forward-macro/src/codegen.rs:1500-2170` — the macro's `fingerprint_matches` generator. Read this if you're debugging the open bug.
- `crates/ferrite-metal-kernels/shaders/quantized_qmv.metal:939-1110` — `affine_gather_qmv[_fast]` shaders. Critical: row→token mapping uses `row / top_k`, which is why step (j) of the lowering arm passes `top_k=1` for `down_proj`.
- `crates/ferrite-model-qwen3-moe/src/lib.rs` — the `#[forward]` DSL body. The macro expands this into the compiled-in `Weights::load` + `fingerprint_matches` for every (variant, tp) tuple.
- `crates/ferrite-model-qwen3-moe/configs/*.json` — variant configs. `qwen3-30b-a3b-instruct.json` is the 30B-A3B base, `qwen3-moe-1-layer.json` is a tiny test variant. The `mlx-affine-b4-g64` quantization is applied at macro-expansion time per `quantizations.json`.

### Outside the repo (MLX reference — porting source-of-truth)

These are the Python reference implementations the kernels are faithful ports of. The user's machine has `~/git/mlx-lm` and `~/git/mlx` cloned locally. On a fresh machine, clone both to the same paths:

```bash
git clone https://github.com/ml-explore/mlx-lm.git ~/git/mlx-lm
git clone https://github.com/ml-explore/mlx.git ~/git/mlx
```

Key files (line numbers as of mlx-lm 0.X / mlx 0.X — verify against the file before quoting):
- `~/git/mlx-lm/mlx_lm/models/qwen3_moe.py:110` — `Qwen3MoeSparseMoeBlock.__call__`. This is the body the lowering arm replicates.
- `~/git/mlx-lm/mlx_lm/models/qwen3_next.py:308` — `Qwen3NextSparseMoeBlock.__call__`. Same plus a sigmoid-gated shared expert (not yet wired in the lowering arm — `assert!` blocks it).
- `~/git/mlx-lm/mlx_lm/models/switch_layers.py` — `SwitchGLU`, `QuantizedSwitchLinear`, `_gather_sort` / `_scatter_unsort`. The MLX module the arm decomposes.
- `~/git/mlx/mlx/backend/metal/kernels/quantized.h:1900` — `affine_gather_qmv_fast` (MLX-side row-gather decode kernel; the ferrite-metal port is in `quantized_qmv.metal:966`).
- `~/git/mlx/mlx/backend/metal/kernels/softmax.h:11` — `softmax_single_row` (precise variant). Port at `shaders/softmax.metal`.
- `~/git/mlx/mlx/backend/metal/kernels/sort/sort.cpp:15` — `single_block_sort`. Port at `shaders/argpartition.metal`.
- `~/git/mlx/mlx/backend/metal/kernels/indexing/gather_axis.h:6` — `gather_axis`. Port at `shaders/take_along_axis.metal`.

The contract this branch operates under is "port the MLX kernel verbatim, no rewriting"; if you find yourself reinventing one of these, stop and copy the MLX source instead.

---

## What's still ahead (after this branch validates Qwen3-MoE)

The original goal is Qwen3-Next (`Qwen3NextForCausalLM`). Phase F validates **Qwen3-MoE** first because it shares the SwitchGLU MoE block (same lowering arm) but has none of the GDN / gated-attention complications. After Qwen3-MoE is coherent, the open work for Qwen3-Next is:

1. **Shared-expert tail** in the `I::SharedFusedMoe` arm — drop the `shared_expert_gate.is_none()` assert and emit five more `LoweredCommand`s for `(affine_qmm gate, affine_qmm up, silu_mul, affine_qmm down, sigmoid(shared_gate_logit) * shared_mlp_out, residual_add)`. The Cuda-side reference is `vllm-cuda/src/model/qwen3_next.rs::Qwen3NextSparseMoeBlock`.
2. **Gated attention Metal variant** — output gate (sigmoid * o_proj_input), partial RoPE (`partial_rotary_factor`), per-head Q/K RMSNorm. MLX reference at `qwen3_next.py:158`.
3. **GDN (gated delta network) Metal kernel** + `GdnStatePool`. MLX reference at `~/git/mlx-lm/mlx_lm/models/gated_delta.py::_make_gated_delta_kernel` (~80 LOC `mx.fast.metal_kernel`, copy verbatim).
4. **Per-layer dispatch** (`layer_types[i] ∈ {full_attention, linear_attention}`) — needs macro-level support, not just a lowering arm.

These are tracked in the worktree's prior session memory but are explicitly out of scope for this commit. Don't start on them until Qwen3-MoE produces coherent vllm chat output on this branch.

---

## Smoke procedure once the open bug is fixed

```bash
# 1. Confirm compiled variant is recognized:
./target/release/vllm ferrite info qwen3 moe | head -5
# Expect to see: qwen3_moe / qwen3-30b-a3b-instruct-mlx-affine-b4-g64

# 2. Run end-to-end:
RUST_LOG=info ./target/release/vllm chat \
    mlx-community/Qwen3-30B-A3B-4bit --device metal --quick "What is 2+2?"
# Expect: log line "Using ferrite-metal backend (Apple Silicon GPU)" (NOT "falling back to MLX")
# Expect: a coherent textual answer to "What is 2+2?" (≈ "2 + 2 = 4." or a CoT-style explanation)

# 3. If output is garbled, suspect router_gate (likely real-world bug #2 above):
#    re-check the SharedFusedMoELayer::load metal impl is dequantizing mlp.gate.{scales,biases}
#    into a dense T_act buffer, not passing the packed u32 to the GEMM.

# 4. Only after coherent output, this branch is shippable. Per the project's
#    "don't commit half-wired work" rule, that's when commit 1's diff is
#    considered validated. Drop commit 2 (debug instrumentation) before opening
#    a PR.
```

---

## Quick file index for the changeset

```
crates/ferrite-forward/src/interpreter/metal/lowered.rs          # +KernelId variants, +Binding::{MoeScratch,Inline}, +SharedFusedMoe bundle, +MoE WeightTensor, +moe_scratch_bytes
crates/ferrite-forward/src/interpreter/metal/lowering.rs         # +I::SharedFusedMoe arm + helpers (align_256, softmax_tg_size, arg_sort_bn, pick_affine_gather_qmv)
crates/ferrite-forward/src/interpreter/metal/pool.rs             # pass weights to lower_pair
crates/ferrite-forward/src/interpreter/metal/worker.rs           # +moe_scratch buffer, +Binding::{MoeScratch,Inline} resolvers, +MoE WeightTensor arms, resolve_gemm_buffers accepts moe_scratch
crates/ferrite-forward/src/lib.rs                                # commit 2: FERRITE_TRY_LOAD_DEBUG eprintln
crates/ferrite-kernels/src/layers_moe.rs                         # +MetalSwitchGluMoeWeights, +metal SharedFusedMoELayer::load
crates/ferrite-metal-kernels/shaders/quantized_qmv.metal         # +affine_gather_qmv[_fast]_<dt>_s_f16_gs_<gs>_b_4 instantiations
crates/ferrite-metal-kernels/shaders/{softmax,argpartition,take_along_axis,row_gather,moe_weighted_sum,slice_trailing_cols_u32,top_k_renormalize}.metal
crates/ferrite-metal-kernels/src/lib.rs                          # +mod registrations
crates/ferrite-metal-kernels/src/quantized.rs                    # +MetalAffineGatherQmv
crates/ferrite-metal-kernels/src/shader_cache.rs                 # +library registration + library_for prefix dispatch
crates/ferrite-metal-kernels/src/specialized_pipeline_cache.rs   # +library registration
crates/ferrite-metal-kernels/src/{softmax,argpartition,take_along_axis,row_gather,moe_weighted_sum,slice_trailing_cols_u32,top_k_renormalize}.rs
crates/ferrite-metal-kernels/tests/{softmax,argpartition,take_along_axis,row_gather,moe_weighted_sum,slice_trailing_cols_u32,top_k_renormalize,affine_gather_qmv}_test.rs
crates/ferrite-model-qwen3-moe/configs/quantizations.json        # +mlx-affine-b4-g64
crates/ferrite-models/Cargo.toml                                 # metal = [..., "ferrite-model-qwen3-moe?/metal"]
```

End of handoff.
