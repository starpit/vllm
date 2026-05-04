# Multimodal (Qwen2-VL) — handoff

**Branch:** `feat/rust`, worktree `.claude/worktrees/multimodal`.
**Target arch:** `Qwen2VLForConditionalGeneration`. Text decoder = `ferrite-model-qwen2`.
**Status:** **2B is e2e-coherent at tp=1 AND tp=2.** Real-image chat works (red circle, blue square, streamlit screenshot). Bugs 1, 2, 3 fixed at tp=1; Phase F TP=2 landed with replicated-per-rank ViT + post-AllReduce splice op. VisionConfig derives from weight shapes — 7B/72B *should* load through the same registration without code edits, but unverified.

## What remains

1. **Verify 7B / 72B load + decode coherently.** Shape-derived VisionConfig should make this a no-code-edit `vllm chat` run. Failure mode would be an unanticipated tensor name in the loader's shape probe (`GpuWeights::tensor_shape_any`).

2. **First-20-token byte-match vs Python vLLM.** Currently diverges at token 3 on a synthetic gradient input — both responses are coherent and correctly identify the gradient. Three-way diff an earlier session **clears the encoder** (ferrite-vs-pyvllm cosine ≥ pyvllm-vs-HF at every block; patch_embed bit-exact; merger 0.993 vs 0.971). Residual = sub-bf16-eps logit drift flipping argmax. Not a wiring bug — only chase if byte-exact is a hard requirement.

3. **Qwen2.5-VL** (own crate, `ferrite-model-qwen2-5-vl`, depending on `ferrite-model-qwen2` for text). Vision tower has different math from Qwen2-VL — window attention with periodic full-attention, RMSNorm in vision blocks, revised 2D RoPE, plus FPS/time embeddings for video. New kernel(s) likely required for windowed varlen attention.

4. **Phase G — Gemma3-MM / SigLIP** (own crate, `ferrite-model-gemma3-mm`). First SigLIP integration — second concrete consumer that justifies factoring shared ViT building blocks (LN/varlen-attn/MLP/projector wrappers) out of qwen2-vl into a `ferrite-vision` shared module. Plug-in surface is `MultimodalForward` + `inventory::submit!`. Non-ferrite touch budget = 0 (per Phase F's precedent).

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

Smoke: `vllm chat` on `Qwen/Qwen2-VL-2B-Instruct` with `--image=docs/assets/deployment/streamlit-chat.png "what does the page say?"` should return a coherent description of the streamlit chat page. Text-only `"What is the capital of France?"` should return "The capital of France is Paris."

Numerical-golden harness (only needed if you suspect a vision-encoder regression). The `tests/` dir didn't move with the split; harness lives at:
- `crates/ferrite-model-qwen2/tests/golden_gen_qwen2_vl_vision.py` — dumps HF intermediates on a deterministic 224×224 input. Run via `~/vllm/.venv`.
- `crates/ferrite-model-qwen2/tests/diff_qwen2_vl_vision.py` — bf16-tolerant diff against `FERRITE_VIT_DUMP_DIR`.
- `ferrite-model-qwen2-vl::vision::TraceDump` — env-gated D2H of per-stage encoder tensors. (Move the `tests/` files alongside this crate when convenient — they only test the vision side.)

## Where to find things

| What | Where |
|---|---|
| Worker MM dispatch (encoder run, MRoPE 2D, MM-bearing graph gates) | `vllm-executor/src/cuda_worker.rs` — `run_mm_vision_forward`, `build_per_req_mm_seq_info`, `build_mrope_positions_2d`, `mm_data_buffers`, the `use_prefill_graph` gate at ~8600 + the symmetric decode-graph gate at ~7729 |
| Splice site (vision embeds → token positions) | `ferrite-forward/src/instr.rs` — `Instruction::Embed::eval` |
| Vision tower, weights, registration | `ferrite-model-qwen2-vl/src/vision.rs` — `VisionWeights`, `VisionConfig` (shape-derived), `vision_forward`, `MultimodalForward` impl, `inventory::submit!` rows at tp ∈ {1,2,4,8}. Crate is paired with `ferrite-model-qwen2` (shared text decoder); `configs/qwen2-vl-2b.json` over there registers the text forward for the same arch string. |
| Vision-only kernels | `ferrite-kernels/src/{rotary,kernels}.rs` — `vision_rope_apply`, `quick_gelu_inplace`, `gelu_erf_inplace` |
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

## Non-goals (do not attempt under this handoff)

- Adding any new pyo3 surface for MM. Vision runs in ferrite or it doesn't run.
- Calling Python vLLM at runtime. Python is the numerical golden only.
- Adding any arch arm to a god-switch in `vllm-model/src/gguf.rs` or in `cuda_worker.rs::match arch`. Per-arch state lives on emitted Weights, accessed through `MultimodalForward`.
- Widening the non-ferrite touch budget: `vllm-pyo3 = 0`, `vllm-cuda = 0`, `vllm-mlx = 0`, `vllm-model = 0`, `vllm-engine = 0`, `vllm-common = 0`, Python `vllm/ = 0`, `vllm-executor = ~10 lines in the Self::Ferrite arm`. If a phase looks like it needs more, the design is wrong — stop and revisit.
