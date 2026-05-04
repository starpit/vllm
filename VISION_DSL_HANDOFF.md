# Vision DSL — Phase G handoff

**Branch:** `feat/rust`, worktree `.claude/worktrees/multimodal`.
**Goal:** lift the vision tower into a `#[vision_forward]` DSL body, mirroring how text decoders are described. Each MM arch ends up with two backbones — text (`#[forward]`) and vision (`#[vision_forward]`) — composed at runtime via the existing `MmEmbedSplice` op. Supersedes the MULTIMODAL_HANDOFF.md "Phase G — Gemma3-MM / SigLIP" framing, which assumed a non-DSL `ferrite-vision` shared module.

**Why now:** with 3+ VL/MM arches imminent (Qwen2-VL + Qwen2.5-VL + Gemma3-MM/SigLIP, internvl/llava/pixtral on the horizon), the imperative ~1k-line per-arch `vision.rs` (931 + 989 lines today) is the wrong cut. Hybrid DSL gives per-block fusion, const-prop, and reduces each new tower to ~30 lines of math. The first port is the expensive one; each subsequent arch is the payoff.

## Phases

**G.1 — `ferrite-vision` host-glue crate. (DONE.)** Lifted from both VL crates: `VisionConfig` (common geometric fields) with `build_rope_cos_sin_bf16` + `patches_from_normalized_chw` as methods, plus free-fn `build_cu_seqlens_i32`, `pad_linear_k_to_mult8` (Q2.5-VL cuBLAS K=3420 fix), `TraceDump`, byte-slice helpers. Both VL crates re-export `pub use ferrite_vision::VisionConfig` and now carry only their arch-specific extras (Qwen2.5-VL: `VisionExtras { intermediate_size, window_size }`). Behavior-preserving — no DSL changes.

**G.2 — New OpKinds in `ferrite-forward-macro`. (DONE.)** Landed `7f0dabec3`: four new variants in the `OpKind` enum (`classified.rs:67`) — `VarlenAttention`, `VisionRope`, `QuickGelu`, `GeluErf`. Existing `Gelu` stays tanh-form. Each variant has a shape signature in `shape.rs` (`sig_varlen_attention`, `sig_vision_rope`, plus arms on `sig_unary_elementwise` for the two GELUs) and a corresponding `weight_arg_ranks` row. `Stmt::AssignTuple` grew a third arm for `(q, k) = vision_rope(...)`. Six unit tests in `shape::tests` lock the sigs. **Deliberately deferred:** `from_name` arms — they land in lockstep with G.4 Impls + G.5 DSL bodies so parse ⟺ codegen stays total (no parse-then-reject). Until then the new OpKinds are produced only via direct `apply_signature` calls (i.e., from tests).

**G.3 — `#[vision_forward]` attr macro. (DONE.)** Both attribute entries call a shared `compile_common(args, carrier, mode: CompileMode)`; the mode gates three local overrides — DSL prelude (`Prelude::Decoder` vs `Prelude::Vision` threaded through `classify::classify_with`), tp fanout (decoder fans `{1, 2, 4, 8}` at nccl-enabled, vision pinned to `[1]`), and the post-FUF lowering passes (`insert_all_reduces` / `insert_lm_head_allgather` / `insert_mm_splices` skipped for vision). `ExternKind` grew six vision variants (`Pixels`/`CuSeqlens`/`Cos`/`Sin`/`GridThw`/`MaxSeqlen`) with a `from_name_for(name, prelude)` that keeps the two extern sets disjoint — `pixels` is unrecognized in a decoder body and `input_ids` in a vision body. `extern_shape` got arms for the new variants (`Pixels`/`Cos`/`Sin` shapes anchor on `vision_in_features` / `vision_rope_half_dim` bounds populated per-arch in G.5; the rest are opaque). 4 new classify unit tests pin the routing; llama-2-7b text decoder still emits 484 tiles · 228 waves byte-equivalently. Workload axis reuse: vision keeps the existing `workloads = [...]` slot and reads it as **batched-total flat L** (not per-image) — same axis as text-side `num_tokens`.

**G.4 — Implementations for the new OpKinds.** One Impl per OpKind, claiming the obvious tile shape, calling existing kernels: `VarlenAttention` → `flash_attn_contiguous` with `cos_sin_cache_ptr=null`; `VisionRope` → `vision_rope_apply`; `QuickGelu` → `quick_gelu_inplace`; `GeluErf` → `gelu_erf_inplace`. Calibrate cost CSV at the workload sizes the vision backbones will actually produce (one image, varlen L derived from `[224², 448², 672², streamlit-screenshot]`).

**G.5 — Port Qwen2-VL.** Rewrite `ferrite-model-qwen2-vl/src/vision.rs` so the encoder is a `#[vision_forward] fn qwen2_vl()` body — patch_embed (`gemm` + synthesized `reshape`), 32-block loop, ln_q + merger reshape + merger MLP all inside the DSL. Imperative `VisionWeights::forward` deletes ~200 lines (block loop + merger MLP path); the orchestrator wrapper retains pixels packing + `cu_seqlens` / cos-sin building (now via `ferrite-vision`) + a single call into the generated `qwen2_vl_vision_forward(...)`. **Verification:** the three bug reproducers (`test_qwen2_vl_bug{1,2,3}_*`) plus the byte-equivalence smoke (224² red circle + streamlit screenshot) all pass.

**G.6 — Port Qwen2.5-VL.** Same shape as G.5. Two block flavors (full vs windowed varlen) — pick after seeing G.5's emitted code whether that's two `#[vision_forward]` bodies, one body with a runtime-conditioned varlen-attn op, or a `fullatt_block_indexes`-driven `if` inside the body. K=3420 cuBLAS pad trick lives in `ferrite-vision` (load-time weight pad + runtime concat-with-pad before `silu_and_mul_fused`). Same e2e tests; same byte-equivalence smoke.

**G.7 — SigLIP / Gemma3-MM.** Third arch lands as a ~30-line `#[vision_forward]` body in a new `ferrite-model-gemma3-mm` crate. **If it's longer than ~50 lines, the DSL is missing an op — fix the DSL, not the arch.** This phase is the test that we got the cut right.

## Non-goals

- No TP for vision in v1. Replicated weights stay; no AllReduce/AllGather extension.
- No new pyo3 / vllm-cuda / cuda_worker surface. Touch budget on those crates: 0 lines.
- Existing imperative `vision.rs` files don't survive G.5/G.6 — they're rewritten, not paralleled. (Per the "no parallel impls" feedback.)
- `ferrite-vision` is host-glue ONLY. GPU compute lives in OpKinds. If a helper does `kernels::*` work it belongs as an OpKind, not a shared function.
- No attempt to merge text and vision into one body. They run at different times in the request lifecycle, with different KV semantics — keep them as peer compiled forwards.

## Open questions to resolve before G.3

1. **Workload bucketing key.** RESOLVED: bucket on **batched-total flat L** (the concatenated varlen length the kernel actually launches over), not per-image. Mirrors text-side `num_tokens` semantics — total across the batch, not per-sequence. Default sweep `workloads = [256, 1024, 4096, 16384]` covers single-image 224² (~256 patches), 448² (~1024), 672² (~2304), and ~4-image streamlit-class (~16384). The vision macro reuses the existing `workloads` attribute slot — no new key. Per-image `max_seqlen` is a kernel input, not a solver axis.
2. **Reshape DSL surface.** RESOLVED FOR G.3: deferred. `OpKind::Reshape` stays synthesized-only through G.3. The vision body has two explicit reshapes (`[L, H, D] → [L, H*D]` post-attention, `[L, embed] → [L/S², embed*S²]` pre-merger-MLP); the first is recoverable by per-head shape inference (same shape as text's QK-norm path), the second requires bound-arithmetic in the reshape target dims (`l_out = num_tokens / merge_factor`, `merge_hidden = embed_dim * merge_factor`). G.3's stub vision body avoids the merger reshape; the DSL surface for it lands in G.5 when we actually port qwen2-vl.
3. **Tuple-returning ops in the DSL.** RESOLVED: 2-target works. `vision_rope_arity_and_output_shape` test path validates `Stmt::AssignTuple` for 2-target via the `vision_rope` shape signature; 3-target is exercised by `MlaSplit`. No further code change needed.

## Where to start

G.1 + G.2 + G.3 done. Next: **G.4** — Impls for the four new vision OpKinds (`VarlenAttention` / `VisionRope` / `QuickGelu` / `GeluErf`) plus the `OpKind::from_name` arms that make them DSL-writable. The two land lockstep so parse ⟺ codegen stays total. After G.4, G.5 ports qwen2-vl's `vision.rs` into a `#[vision_forward]` body and exercises the whole stack against the e2e suite.

**G.4 entry notes** (for the next session):
- `OpKind::from_name` arms today are total-but-decoder-only. G.4 lands four new arms (`varlen_attention`, `vision_rope`, `quick_gelu`, `gelu_erf`) — naming convention follows `OpKind::as_str`. No prelude gating needed: the names are unambiguous and unused on the decoder side. Adding them in `OpKind::from_name` exposes them to BOTH preludes' classifiers, but only vision bodies have any reason to write them.
- Each new OpKind needs at least one `Impl` claiming its tile shape so the solver doesn't error with `UnclaimedTile`. Existing kernel surfaces map directly: `VarlenAttention → flash_attn_contiguous(cos_sin_cache_ptr=null)`, `VisionRope → vision_rope_apply`, `QuickGelu → quick_gelu_inplace`, `GeluErf → gelu_erf_inplace`. The cost CSV may need calibration at vision-shape workloads (varlen L = 256 / 1024 / 4096 / 16384) before peer Impls win on cost — see `feedback_calibrate_before_new_impl`.
- G.5's qwen2-vl port also needs new bounds in vision config (`vision_in_features = c*t*p*p`, `vision_rope_half_dim = head_dim/2`). The `extern_shape` arms reference these; until vision config files exist they won't resolve. G.4 does NOT need them — Impl matchers run AFTER shape inference and the existing decoder configs don't trigger the vision arms.

## Verification at every phase

Per the "integration test per phase" feedback rule: each phase ships with the existing e2e suite (`test_qwen2_vl_bug{1,2,3}_*` + tp=2 sanity) green. No phase commits with red e2e tests. The byte-equivalence smoke against the streamlit-screenshot prompt is the canary — if it stops naming "vLLM Chat Assistant" we lost the math somewhere.
