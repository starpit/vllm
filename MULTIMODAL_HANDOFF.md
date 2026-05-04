# Multimodal (Qwen2-VL / Qwen2.5-VL) — handoff

**Branch:** `feat/rust`, worktree `.claude/worktrees/multimodal`.
**Target arches:** `Qwen2VLForConditionalGeneration`, `Qwen2_5_VLForConditionalGeneration`. Text decoder = `ferrite-model-qwen2` (shared).
**Status:** **Qwen2-VL-2B and Qwen2.5-VL-3B both e2e-coherent at tp=1.** Real-image chat works on both (Qwen2.5-VL-3B verified on 224×224 red circle and the streamlit screenshot — "vLLM Chat Assistant" identified correctly; Qwen2-VL-2B also coherent at tp=2). Each VL arch has its own crate and registers via `inventory::submit!`. Shape-derived VisionConfigs — 7B/72B (Qwen2-VL) and 7B/72B (Qwen2.5-VL) *should* load without code edits, unverified.

## What remains

1. **Verify 7B / 72B (both Qwen2-VL and Qwen2.5-VL) load + decode coherently.** Shape-derived VisionConfigs should make this no-code-edit `vllm chat` runs. Failure mode would be an unanticipated tensor name in either loader's shape probe (`GpuWeights::tensor_shape_any`).

2. **Qwen2.5-VL TP=2.** Single-rank works; TP=2 inventory rows are emitted (`#[cfg(feature = "nccl")]`) but not exercised. Same replicated-per-rank ViT + post-AllReduce `MmEmbedSplice` op handles it for free at the FUF level — likely just a verify step.

3. **First-20-token byte-match vs Python vLLM (Qwen2-VL).** Currently diverges at token 3 on a synthetic gradient input — both responses are coherent and correctly identify the gradient. Three-way diff an earlier session **clears the encoder** (ferrite-vs-pyvllm cosine ≥ pyvllm-vs-HF at every block; patch_embed bit-exact; merger 0.993 vs 0.971). Residual = sub-bf16-eps logit drift flipping argmax. Not a wiring bug — only chase if byte-exact is a hard requirement.

4. **Phase G — vision DSL + Gemma3-MM / SigLIP.** See `VISION_DSL_HANDOFF.md` for the full plan. Short version: each MM arch grows a second backbone alongside its `#[forward]` text body — a `#[vision_forward]` DSL function for the encoder — composed at runtime by the existing `MmEmbedSplice` op. Sequencing: G.1 extracts a `ferrite-vision` host-glue crate; G.2 adds `VarlenAttention` / `VisionRope` / `QuickGelu` / `GeluErf` OpKinds; G.3 adds the `#[vision_forward]` attr macro (same FUF/solver/codegen pipeline downstream); G.5/G.6 port Qwen2-VL and Qwen2.5-VL; G.7 lands SigLIP/Gemma3-MM as the ~30-line third arch that proves the cut. Non-ferrite touch budget = 0 (per Phase F's precedent). Earlier framing of Phase G as an imperative `ferrite-vision` shared module is superseded.

## How to verify nothing regressed

Single-rank:
```bash
cargo build -p vllm-cli --features cuda --release
cargo test --release -p vllm-e2e --features e2e,cuda \
    --test e_qwen2_vl -- --ignored --test-threads=1
```

TP=2 (requires 2 CUDA GPUs + NCCL; nick3 is 2× L40S):
```bash
cargo test --release -p vllm-e2e --features e2e,cuda,nccl \
    --test e_qwen2_vl test_cuda_tp2 -- --ignored --test-threads=1
```

The three bug reproducers each fail without their fix:
- `test_qwen2_vl_bug1_prefill_graph_skips_encoder` — 224×224 red circle, asserts response mentions red/circle. Pre-fix: model hallucinates content unrelated to the image.
- `test_qwen2_vl_bug2_two_distinguishable_images` — 224×224 red-circle then 224×224 blue-square. Same prompt template, identical 64-image-pad runs → identical block layout. Asserts response B mentions blue/square and does not describe a red circle.
- `test_qwen2_vl_bug3_same_image_twice_cached_prefix` — `docs/assets/deployment/streamlit-chat.png` twice (~1681 tokens, block-aligned). Asserts second response ≥ 8 completion tokens (pre-fix: 1, immediate `<|im_end|>`).

TP=2 sanity suite (`test_cuda_tp2_qwen2_vl_*`): text-only, single image (red circle), two images (red→blue). Guards that the post-Embed splice op ran on the reduced embedding (pre-fix the mm_embeds were summed × tp and decode was fluent nonsense).

Fixtures: `vllm-rs/crates/vllm-e2e/tests/fixtures/{red_circle_224,blue_square_224}.png` + `docs/assets/deployment/streamlit-chat.png`.

Smoke: spin up `vllm serve <model> --enforce-eager --port 17777` and POST `/v1/chat/completions` with an `image_url` data-URI (`vllm chat` has no `--image` flag yet — the e2e tests use the OpenAI-compatible serving path). Both arches should return:
- 224×224 red-circle PNG → mentions "red" and "circle".
- `docs/assets/deployment/streamlit-chat.png` → identifies it as a chat interface (Qwen2.5-VL-3B reliably names "vLLM Chat Assistant").
- text-only "What is the capital of France?" → "The capital of France is Paris."

Build `vllm-cli` with `FERRITE_MODELS=qwen2-vl-2b,qwen2.5-vl-3b` (or omit to compile every variant). The `qwen2.5-vl-3b` stem in the env var matches the `configs/qwen2.5-vl-3b.json` file basename — the dot is significant (`qwen2-5-vl` will not match).

Numerical-golden harness (only needed if you suspect a vision-encoder regression). The `tests/` dir didn't move with the split; harness lives at:
- `crates/ferrite-model-qwen2/tests/golden_gen_qwen2_vl_vision.py` — dumps HF intermediates on a deterministic 224×224 input. Run via `~/vllm/.venv`.
- `crates/ferrite-model-qwen2/tests/diff_qwen2_vl_vision.py` — bf16-tolerant diff against `FERRITE_VIT_DUMP_DIR`.
- `ferrite-model-qwen2-vl::vision::TraceDump` — env-gated D2H of per-stage encoder tensors. (Move the `tests/` files alongside this crate when convenient — they only test the vision side.)

## Where to find things

| What | Where |
|---|---|
| Worker MM dispatch (encoder run, MRoPE 2D, MM-bearing graph gates) | `vllm-executor/src/cuda_worker.rs` — `run_mm_vision_forward`, `build_per_req_mm_seq_info`, `build_mrope_positions_2d`, `mm_data_buffers`, the `use_prefill_graph` gate at ~8600 + the symmetric decode-graph gate at ~7729 |
| Splice site (vision embeds → token positions) | `ferrite-forward/src/instr.rs` — `Instruction::Embed::eval` |
| Vision tower, weights, registration | `ferrite-model-qwen2-vl/src/vision.rs` (Qwen2-VL) and `ferrite-model-qwen2-5-vl/src/vision.rs` (Qwen2.5-VL) — `VisionWeights`, `VisionConfig` (shape-derived), `vision_forward`, `MultimodalForward` impl, `inventory::submit!` rows at tp ∈ {1,2,4,8}. Both crates pair with `ferrite-model-qwen2` (shared text decoder); `configs/qwen2-vl-2b.json` and `configs/qwen2.5-vl-3b.json` over there register the text forward for each arch string. |
| Umbrella linker keepalives | `ferrite-models/src/lib.rs` — `extern crate ferrite_model_<arch> as _keep_<arch>` and `pub use ferrite_model_<arch> as <arch>` for **every** per-arch crate. Without the `_keep_*` line the linker GCs `inventory::submit!` rows; symptom is "MM loader matches the arch but vision_forward never runs", model replies "I can't see any image". |
| Vision-only kernels | `ferrite-kernels/src/{rotary,kernels}.rs` — `vision_rope_apply`, `quick_gelu_inplace`, `gelu_erf_inplace`; `embedding_gather` is reused for the Qwen2.5-VL window permute / reverse-permute (no new kernel). |
| Per-image-bytes block hash (Bug 2 fix) | `SimpleBlockTracker::hash_all_blocks` |
| Image preprocessor (bicubic + smart_resize) | `vllm-model::image::preprocess_qwen2_vl` |
| Tokenizer special-token patch | `Tokenizer::register_additional_special_tokens` + `init.rs::patch_additional_special_tokens` |
| Per-image placeholder expansion | `expand_image_placeholders_per_image` (chat path) + `engine.rs` count = `(h/28)·(w/28)` for qwen2_vl |
| Post-Embed splice op (Phase F) | `ferrite-forward-macro/src/{classified,impl_lib,tp_lowering}.rs` — `OpKind::MmEmbedSplice`, `MmEmbedSpliceImpl`, `insert_mm_splices`; `ferrite-forward/src/instr.rs` — `Instruction::SpliceMmEmbeds::eval` |
| TP=2 MM init wiring | `vllm-serve/src/init.rs::initialize_stack_tp` — MM-config block mirrors `initialize_stack`'s |

## What landed (terse)

Phases A–E from the original plan are done. **A** generalized text-side RoPE kernels with optional `mrope_section`. **B** added `MultimodalForward` as a sibling trait (no `unimplemented!` defaults), extended `Weights` with `Option<VisionWeights>`, taught the loader the `visual.*` weight names. **C** added the three genuinely-new kernels (`vision_rope_apply`, `quick_gelu_inplace`, `gelu_erf_inplace`); the rest were compositions of existing primitives, folded into D. **D** built the `qwen2_vl_vision` DSL forward. **E** plumbed the splice (`ForwardCtx.mm_embeds` + `embed_patches`, `Instruction::Embed::eval` D2D copy), the executor seam (encoder run, 2D MRoPE positions, mm_data lifecycle), the chat-template additional_special_tokens patch, per-image placeholder counts, bicubic resize, and `preprocessor_config.json::{min,max}_pixels`.

After E shipped, three integration bugs surfaced and were fixed:
- **Bug 1** — captured prefill graph replays skipped vision encoder + Embed splice. Fix: gate `use_prefill_graph` on `!req_has_mm` (forces eager path for MM reqs).
- **Bug 2** — block hash matched across different images at same dims (image_pad token IDs identical → same block hash → cache hit on wrong KV). Fix: fold per-image hash into every block whose tokens overlap that image's placeholder range.
- **Bug 3** — cached-prefix MM-bearing requests skipped `run_mm_vision_forward` (`tokens_before > 0`) and the trailing new prompt token's MRoPE positions fell back to 1D, disagreeing with the cached KV → instant `<|im_end|>`. Fix: a CPU companion (`embed_patch_grids`) runs unconditionally; `build_per_req_mm_seq_info` walks every MM req regardless of `tokens_before`; `build_mrope_positions_2d` walks the full seq advancing the cursor through cached patches.

VisionConfig was then generalized to derive from weight shapes (no more 2B constants).

**Crate split (post-Phase F).** Qwen2-VL's vision tower moved out of `ferrite-model-qwen2` into its own `ferrite-model-qwen2-vl` crate. The text decoder stays shared in `ferrite-model-qwen2` (same `qwen2()` forward for plain Qwen2/Qwen2.5/Qwen2-VL/Qwen2.5-VL — config-only differences). The split aligns with the workspace convention (one crate per arch with distinct math) and clears the path for `ferrite-model-qwen2-5-vl` and `ferrite-model-gemma3-mm` to land as siblings without expanding `ferrite-model-qwen2`. Move was purely organizational — vision.rs has no internal dependencies on the qwen2 text decoder code (splice is runtime via Embed/MmEmbedSplice ops, not Rust imports).

**Phase F — TP for VL (this commit).** Four changes:
1. **Post-Embed splice op.** New `OpKind::MmEmbedSplice` + `Instruction::SpliceMmEmbeds(slot)` + `MmEmbedSpliceImpl` (mirrors `AllReduceImpl`: single-tile in-place, output aliases input slot). `tp_lowering::insert_mm_splices` walks the FUF after `insert_all_reduces` and appends one splice op per Embed; at tp>1 it ends up reading the AllReduce output (which `insert_all_reduces` already chained onto the ShardDim0 Embed), at tp=1 it reads the Embed directly. Unconditional at every tp — runtime no-op when `ForwardCtx::embed_patches` is empty. This replaces the pre-refactor inline splice inside `Instruction::Embed::eval`, which ran before the vocab-parallel AllReduce and so got summed × tp (fluent-nonsense at tp=2).
2. **`backbone_output_for` / `last_node_id` skip splice tails.** `insert_mm_splices` pushes new FufNodes at array-tail, which would otherwise displace the lm_head Gemm / AllGather as `fuf.nodes.last()`. New `last_non_splice_node` helper walks the tail skipping `OpKind::MmEmbedSplice` so both callers recover the real terminal.
3. **Vision registrations at tp ∈ {1, 2, 4, 8}.** `vision.rs` emits one `inventory::submit!` per tp (≥2 gated on `nccl`). Each rank loads the full `visual.*` weights (replicated) and runs `vision_forward` independently — mm_embeds are identical per-rank, the post-AllReduce splice overwrites patch rows on every rank, and the subsequent text-decoder forward sees consistent hidden states.
4. **`initialize_stack_tp` MM wiring.** The tp-path was missing `set_multimodal_config` / `set_mm_model_type` / `set_mm_image_processor_pixel_limits` and the `patch_additional_special_tokens` call. Without them the engine dropped image content silently at the chat-template level (image_url → text pass-through). Block copied from the single-rank `initialize_stack`.

**Qwen2.5-VL crate (this commit).** Sibling to `ferrite-model-qwen2-vl`. Vision math deltas vs Qwen2-VL: block norms (norm1/norm2) and merger.ln_q switched from LayerNorm (weight + bias) to RMSNorm (weight only); block MLP swapped `fc1 → QuickGELU → fc2` for SwiGLU `down_proj(silu(gate_proj(x)) · up_proj(x))` with explicit `intermediate_size=3420` (not embed×ratio); 28-of-32 blocks run windowed varlen attention bucketed by 112-px windows (the 4 in `fullatt_block_indexes=[7,15,23,31]` use full image-frame attention); tokens + cos/sin are gather-permuted into window order on entry and unpermuted post-merger via `embedding_gather` (no new kernel). Patch flatten and merger MLP shape unchanged. Family-wide constants (`num_heads=16`, `window_size=112`, `fullatt_block_indexes`, `norm_eps=1e-6`) baked into the loader; everything else (`embed_dim`, `depth`, `intermediate_size`, `d_model`, etc.) derived from weight shapes. Inventory rows at tp ∈ {1,2,4,8}.

**Q2.5-VL gotcha — cuBLAS K=3420.** Qwen2.5-VL's vision MLP picks a deliberately not-power-of-two intermediate (3420 = 4×3×5×3×19; mod 4 but not mod 8). cuBLAS BF16 GEMM on K=3420 fails on every cublasLt algo and the cublasGemmEx fallback errors with `CUBLAS_STATUS_INTERNAL_ERROR` — every other GEMM in the tower has K mod 8. Fix: at load time pad `down_proj.weight` from `[E, 3420]` to `[E, 3424]` with 4 zero columns (`pad_linear_k_to_mult8` in `vision.rs`); at runtime concatenate `gate_proj(x)` and `up_proj(x)` into a `[L, 2·3424]` zero-init buffer before `silu_and_mul_fused(_, 3424)` so the trailing 4 cols of each half are zero (`silu(0)·0 = 0` keeps the padded slots zero through the next GEMM's K=3424 contraction). All math identical to the unpadded reference. The packing trade-off vs `LinearLayer::load_dense_concat` is 2 extra D2D copies per layer (per-row, total ~60 µs added on a 256-token vision batch) — the alternative would be a custom 2-tensor silu+mul kernel.

## Non-goals (do not attempt under this handoff)

- Adding any new pyo3 surface for MM. Vision runs in ferrite or it doesn't run.
- Calling Python vLLM at runtime. Python is the numerical golden only.
- Adding any arch arm to a god-switch in `vllm-model/src/gguf.rs` or in `cuda_worker.rs::match arch`. Per-arch state lives on emitted Weights, accessed through `MultimodalForward`.
- Widening the non-ferrite touch budget: `vllm-pyo3 = 0`, `vllm-cuda = 0`, `vllm-mlx = 0`, `vllm-model = 0`, `vllm-engine = 0`, `vllm-common = 0`, Python `vllm/ = 0`, `vllm-executor = ~10 lines in the Self::Ferrite arm`. If a phase looks like it needs more, the design is wrong — stop and revisit.
