# Vision DSL — Phase G handoff

**Branch:** `feat/rust`, worktree `.claude/worktrees/multimodal`.
**Goal:** lift the vision tower into a `#[vision_forward]` DSL body, mirroring how text decoders are described. Each MM arch ends up with two backbones — text (`#[forward]`) and vision (`#[vision_forward]`) — composed at runtime via the existing `MmEmbedSplice` op. Supersedes the MULTIMODAL_HANDOFF.md "Phase G — Gemma3-MM / SigLIP" framing, which assumed a non-DSL `ferrite-vision` shared module.

**Why now:** with 3+ VL/MM arches imminent (Qwen2-VL + Qwen2.5-VL + Gemma3-MM/SigLIP, internvl/llava/pixtral on the horizon), the imperative ~1k-line per-arch `vision.rs` (931 + 989 lines today) is the wrong cut. Hybrid DSL gives per-block fusion, const-prop, and reduces each new tower to ~30 lines of math. The first port is the expensive one; each subsequent arch is the payoff.

## Phases

**G.1 — `ferrite-vision` host-glue crate. (DONE.)** Lifted from both VL crates: `VisionConfig` (common geometric fields) with `build_rope_cos_sin_bf16` + `patches_from_normalized_chw` as methods, plus free-fn `build_cu_seqlens_i32`, `pad_linear_k_to_mult8` (Q2.5-VL cuBLAS K=3420 fix), `TraceDump`, byte-slice helpers. Both VL crates re-export `pub use ferrite_vision::VisionConfig` and now carry only their arch-specific extras (Qwen2.5-VL: `VisionExtras { intermediate_size, window_size }`). Behavior-preserving — no DSL changes.

**G.2 — New OpKinds in `ferrite-forward-macro`. (DONE.)** Landed `7f0dabec3`: four new variants in the `OpKind` enum (`classified.rs:67`) — `VarlenAttention`, `VisionRope`, `QuickGelu`, `GeluErf`. Existing `Gelu` stays tanh-form. Each variant has a shape signature in `shape.rs` (`sig_varlen_attention`, `sig_vision_rope`, plus arms on `sig_unary_elementwise` for the two GELUs) and a corresponding `weight_arg_ranks` row. `Stmt::AssignTuple` grew a third arm for `(q, k) = vision_rope(...)`. Six unit tests in `shape::tests` lock the sigs. **Deliberately deferred:** `from_name` arms — they land in lockstep with G.4 Impls + G.5 DSL bodies so parse ⟺ codegen stays total (no parse-then-reject). Until then the new OpKinds are produced only via direct `apply_signature` calls (i.e., from tests).

**G.3 — `#[vision_forward]` attr macro.** Sibling to `#[forward]` in `ferrite-forward-macro`, sharing the FUF / solver / codegen pipeline downstream. Only differences: prelude defines vision externs (`pixels`, `cu_seqlens`, `cos`, `sin`, `grid_thw`, `max_seqlen`) instead of decoder externs (`positions`, `rotary`, `kv_cache`, `block_table`); workload-bucketing key is varlen total-L (candidate buckets `[256, 1024, 4096, 16384]`, ceil-bucketed on per-image post-merger token count) instead of decode-iter count; no AllReduce/AllGather lowering pass for v1 (vision stays replicated). Same `Forward` trait, same `LAUNCHER_TABLE` shape, no parallel codegen path.

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

1. **Workload bucketing key.** Vision is one-shot prefill, so the text-side decode-iter grid doesn't apply. Best candidate: ceil-bucket on `max_seqlen` (per-image post-merger token count, ~256 / 1024 / 4096 / 16384 covering 224² → 672² and streamlit-class images). Decide whether the bucket key is per-image or batched-total.
2. **Reshape DSL surface.** `OpKind::Reshape` exists but isn't writable from DSL today (`from_name` lacks the arm; reshape is synthesized by shape inference). Patch_embed and merger both need explicit reshapes — either add `reshape(...)` to `from_name` or extend shape inference to cover the vision patterns.
3. **Tuple-returning ops in the DSL.** Vision uses `(q, k, v) = qkv_split(qkv)` and `(q, k) = vision_rope(q, k, cos, sin)`. `MlaSplit` already covers tuple returns; verify the same `Stmt::AssignTuple` path handles 2-target and 3-target uniformly.

## Where to start

G.1 + G.2 done. Next: G.3 (the `#[vision_forward]` attr macro). G.4 follows. G.5 is the integration test for the whole stack — every preceding phase converges there.

**G.3 entry notes** (for the next session):
- The OpKinds exist and are total over `apply_signature` / `weight_arg_ranks` / `as_str` (G.2). What's missing on the DSL surface is `from_name` arms — those should land WITH G.3's `#[vision_forward]` body parser in lockstep, not standalone.
- The three open questions above (workload bucketing key, Reshape DSL surface, tuple-return verification) want resolution before the macro skeleton is more than a stub. Tuple-return for `vision_rope` is already validated by the `vision_rope_arity_and_output_shape` test path — the AssignTuple machinery handles 2-target uniformly.
- `compile()` in `lib.rs` is large; the prudent shape for G.3 is to factor a `compile_common()` and have `vision_forward` reuse most of it, with three local overrides: extern set, workload bucketing key, and the AllReduce/AllGather/MmEmbedSplice lowering passes (skipped). Avoid copy-pasting `compile()`.

## Verification at every phase

Per the "integration test per phase" feedback rule: each phase ships with the existing e2e suite (`test_qwen2_vl_bug{1,2,3}_*` + tp=2 sanity) green. No phase commits with red e2e tests. The byte-equivalence smoke against the streamlit-screenshot prompt is the canary — if it stops naming "vLLM Chat Assistant" we lost the math somewhere.
