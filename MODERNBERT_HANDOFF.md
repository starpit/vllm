# ModernBERT in ferrite — handoff

Branch: `feat/rust` · worktree: `.claude/worktrees/modernbert` · tip
**1g GREEN** — `vllm serve answerdotai/ModernBERT-base --runner pooling
--pooling-strategy cls` agrees with HF `AutoModel(...)`-CLS at
cosine ≥ 0.9998 across multiple prompts.

## STATE 2026-05-04 (evening)

Phase 1f wired the dispatch but exposed FIVE blockers when the server
was actually exercised. All five landed in 1g (one of them — the
load-time race — is a generic bug that fixed every dense F32-on-disk
checkpoint, not just ModernBERT):

1. **Fingerprint check was llama-hardcoded.** `emit_fingerprint_check`
   sniffed `model.embed_tokens.weight` + `model.layers.{N-1}.self_attn.q_proj.weight`,
   misses ModernBERT (`model.embeddings.tok_embeddings.weight` +
   `model.layers.{N-1}.attn.Wqkv.weight` packed parent). Generalized:
   embed-path comes from the manifest's `[vocab_size, hidden_size]`
   entry; `fp_leaf` checks packed_splits children for `*q_proj`, then
   `q_a_proj`/`q_proj` entries, then `attn.q_proj`. `try_load` now
   matches and loads.
2. **`x * 1.0` consumed its source.** `ScalarMulImpl` always declared
   `consumes_input_tiles = [src]`, but ModernBERT's layer-0 identity
   path uses `normed = hidden_states * 1.0` while `hidden_states` is
   ALSO consumed by the residual add later in the iteration.
   `take_owned` emptied slot 0; the residual add panicked. Added a
   unity-passthrough mode: when scale == 1.0, `output_alias` returns
   `Some(src)`, `consumes_input_tiles` is empty, and `fan_out` returns
   no instructions. Codegen collapses output to source's slot.
3. **Embedding tokenization didn't add specials.**
   `tokenize_embed_text` called `tokenizer.encode(text, false)` —
   without [CLS]/[SEP] the encoder's pooled output is wrong.
   Defaulted to `true` (matches Python vLLM).
4. **Pooling strategy "all" was unrecognized.** The `--pooling-strategy
   all` CLI flag silently fell through `cuda_worker`'s match to the
   auto branch (which returned Last). Added the `"all"` arm. Useful
   for per-token bisection.
5. **GpuWeights::take's slow path raced on the shared cast buffer.**
   ROOT CAUSE of the numerical divergence. Disk weights are F32;
   target is BF16; `maybe_cast_cpu` casts F32→BF16 into a SINGLE
   shared pinned buffer (`cast_pinned`), then `take` issues
   `memcpy_htod_async` from that buffer. The next `take` overwrites
   `cast_pinned` with a new tensor's cast BEFORE the previous async
   memcpy drains, so every queued copy reads whichever cast was
   written last — distinct GPU pointers end up with the same wrong
   payload. Discovered via per-instruction debug dump (added under
   `FERRITE_DEBUG_DUMP_PATH`) showing `mlp_norm[0]`'s GPU pointer
   contained `mlp_norm[16]`'s data. The fast/precast path already
   syncs before freeing its per-tensor pinned buffer; the slow path
   needed the same. Fix: `stream_synchronize` after the async memcpy
   in `take`/`take_into`/`get` whenever `data` came from
   `cast_pinned`. Cosine vs HF AutoModel went from **-0.31 → 0.999944**
   on "Hello world" (and ≥ 0.9998 across 4 different-length prompts).

This last fix is repo-wide load correctness, not modernbert-specific:
ANY safetensors with F32-on-disk weights (most HF base checkpoints)
was getting subtly wrong weight data on GPU. Decoder fleet probably
masked it through robust attention saturating noise across 22+ layers
+ greedy sampling; ModernBERT's CLS-on-encoder pooling is a sensitive
witness that surfaced it.

CommandR canary stays at 9/10 waves through every step.

## Cold-start gotchas

- `cargo test -p ferrite-forward` has a PRE-EXISTING test compilation failure
  (`gemma2_end_to_end.rs` / `phase7_end_to_end.rs` need `ferrite_gguf` which
  isn't in the default test feature set). Confirmed via `git stash`; not
  caused by 1a/1b/1c. Run `cargo test -p ferrite-forward-macro --lib` for
  the relevant test surface (219 pass).
- `Instruction::MeanSubRmsNormBiasAdd::eval` uses `bias.expect(...)` —
  follows existing `FusedGemmBias::dense_bias().is_some()` precedent. The
  matcher-enforced invariant is sound (Impl only fires when DSL has a
  downstream `BiasAdd`, which forces the loader to populate the bias). If
  the "no panics in compiler" memory rule needs stricter enforcement, the
  refactor is to introduce a `LayerNormWithBias` struct in
  `ferrite-kernels::layers` with non-Optional bias, hooked into a new
  `FieldLoad::LayerNormWithBias` arm. Deferred decision; not blocking
  ModernBERT bringup.
- Build commands per memory: `FERRITE_MODELS=command-r-1-layer cargo build
  -p ferrite-model-commandr --features cuda` for canary; same with
  `-p ferrite-model-modernbert` once 1e lands.

## What landed

Two new fusion-only OpKinds (no singleton Impls; matched only inside fusions):

- `OpKind::Mean` (`mean(x)`) — `classified.rs`
- `OpKind::Sub` (`sub(x, y)`) — `classified.rs`

One new fusion Impl per pattern; old LayerNorm-flavored entries deleted, not
parallel:

- `MeanSubRmsNormImpl` claims `(Mean, Sub, RmsNorm)` → emits
  `Instruction::MeanSubRmsNorm` → `kernels::cohere_layer_norm`
- `CutlassFusedMeanSubRmsNormGemmImpl` claims `(Mean, Sub, RmsNorm, Gemm)` →
  emits `Instruction::CutlassFusedMeanSubRmsNormGemm` →
  `cohere_layer_norm + cutlass_gemm`

Retired (deleted): `OpKind::LayerNorm`, `LayerNormRefImpl`,
`CutlassFusedLayerNormGemmImpl`, `Instruction::LayerNorm`,
`Instruction::CutlassFusedLayerNormGemm`, `FieldLoad::CohereLayerNorm`,
`load_layered_cohere_layer_norm`, `is_cohere_layer_norm` codegen branch,
`layer_norm_eps` reader. `rms_norm_eps` now reads either `rms_norm_eps` or
`layer_norm_eps` from JSON (same role under both names).

Commandr DSL migrated: `layer_norm(x, w)` → `mu = mean(x); centered = sub(x,
mu); rmsnorm(centered, w);` at the 2 LN sites in
`ferrite-model-commandr/src/lib.rs`.

## Verification

- 210 macro tests pass (5 LayerNorm-flavored tests retargeted to the
  4-tile pattern; never-deleted).
- Sibling fleet — 201 model variants compile clean across qwen2/mistral/
  llama/gemma2/gemma3/qwen3/phi3/granite/mixtral, wave counts unchanged.
- CommandR wave count matches pre-migration: 9/10 (bf16/ggml). Cutlass
  class restored after adding `CutlassFusedMeanSubRmsNormGemmImpl`.
- Golden 8/8 PASS — `command_r_1l` end-to-end; earliest divergence
  position 13, well past framework threshold of 10 (`assertions.rs:319-402`).

## Why fusion-only opcodes

Following the `Silu`/`Mul` precedent. Standalone Mean or Sub authoring →
`UnclaimedTile` (compiler-author bug). They exist purely as DSL math
primitives that fusion Impls match against. The `cohere_layer_norm` kernel
is unchanged — only the structural claim graph changes.

## Why CutlassFusedMeanSubRmsNormGemmImpl was needed

The old `CutlassFusedLayerNormGemmImpl` seeded on `OpKind::LayerNorm`. After
migration the FUF carries no LayerNorm tiles, so the cutlass norm-gemm
fusion would stop firing → wave count regression. The peer 4-tile claim
preserves the pick. End state has one Impl per pattern (the old one is
deleted, not duplicated).

## Phase 1 — ModernBERT itself

Audit (2026-05-04) confirmed prerequisites + gaps. `OpKind::Gelu` and
`OpKind::BiasAdd` already exist; gemma3's `rotary_local` second-cache
mechanism is reusable as-is. Per-layer specialization for layer-0 identity is
expressible as `if layer < 1` (`BoolPred::Less` exists). Hand-written
ModernBERT is currently wired via `CudaModel::ModernBert` (cuda_worker.rs:170)
— the anti-pattern P1 removes.

Sub-steps, each its own commit, in execution order:

1. **1a — sig_attention arity flex. ✅ DONE.** `shape.rs:556` now accepts
   3 (encoder) OR 5 (decoder); other arities still error. Two regression
   tests added (`encoder_attention_3arg_passes_shape_inference`,
   `attention_arity_4_rejected`). Existing 5-arg unification path unchanged;
   212 macro tests pass.
2. **1b — EncoderAttentionImpl. ✅ DONE.** New `Instruction::EncoderAttention(q,k,v,out)`
   variant + `EncoderAttentionImpl`. Eval calls `flash_attn_contiguous(is_causal=false,
   softcap=0, window=-1, null cos_sin)` — `cu_seqlens_q` was already on
   `ForwardCtx`, no plumbing needed. Routing predicate
   `attention_has_kv_cache_extern` gates all 6 decoder Impls
   (Attention/Sliding × {ViaCache,PrefillContiguous} + FI Decode/Prefill)
   to require the 5-arg form, and the new encoder Impl to require 3-arg.
   Three regression tests: encoder-claims-3-arg, decoder-claims-5-arg,
   starter_library-registers-EncoderAttentionImpl. CommandR canary
   wave count unchanged (9/10).
3. **1c — MeanSubRmsNormBiasAddImpl. ✅ DONE.** 4-tile fusion `(Mean, Sub,
   RmsNorm, BiasAdd)` → `Instruction::MeanSubRmsNormBiasAdd` →
   `kernels::layer_norm_bias`. New `FieldLoad::LayerNorm(prefix, eps)` arm in
   codegen + `load_layered_layer_norm` helper; the rmsnorm's accessor types
   as `LayerNorm` (not `RmsNorm`), so the loader pulls `<prefix>.weight` AND
   `<prefix>.bias` together via `LayerNorm::load`. DSL bias-weight ref is
   structural-only — same trick `FusedGemmBiasImpl` plays with `LinearLayer`.
   Solver claim-size-DESC routes 4-tile over 3-tile when a `BiasAdd` is
   downstream; trio-only sites (CommandR) keep firing the trio. Four
   regression tests; CommandR canary unchanged at 9/10 waves.
4. **1d — Encoder terminator in codegen. ✅ DONE.** `backbone_output_for`
   replaced by `backbone_layout(fuf, program) -> BackboneLayout`
   (Decoder { backbone_out } | Encoder). Decoder path is bit-identical;
   encoder path skips the LM_HEAD slice (lowering uses
   `skip_subgraph=None` and emits an empty `lm_head` slice + matching
   `LM_HEAD_M_<wp>` static), and emits a `forward_backbone` that
   delegates to `forward` (no DtoD memcpy — the encoder's terminal slot
   IS the backbone output). `forward()` works unchanged: `run(backbone,
   empty_lm_head, ..., terminal_slot)` does `take_owned(terminal_slot)`.
   Four regression tests pin the classifier (decoder, encoder,
   tp>1-AllGather-walk-past, non-lm_head-Gemm-is-Encoder); CommandR
   canary unchanged at 9/10 waves; llama/qwen2 sample variants
   compile with unchanged wave counts.
5. **1e — Author ferrite-model-modernbert. ✅ DONE.** New
   `crates/ferrite-model-modernbert` crate registered behind feature
   `arch-modernbert`. DSL is dual-rotary encoder: every Nth layer
   uses the global `rotary` cache, the rest use `rotary_local` (the
   `rope_local_base_freq` second cache, gemma3-style). GeGLU MLP via
   `gelu(gemm(...)) * gemm(...)`. CohereLayerNorm everywhere — the
   trio (`mean, sub, rmsnorm`) without `bias_add` because
   ModernBERT-base ships `norm_bias: false`. The 4-tile
   `MeanSubRmsNormBiasAdd` (1c) stays available for future variants
   that flip `norm_bias: true`. Layer-0 identity attn pre-norm
   expressed as `if layer < 1 { normed = hidden_states * 1.0 }
   else { normed = rmsnorm(sub(x, mean(x)), attn_norm[layer]) }` —
   the `* 1.0` passthrough satisfies the merge-carry rule (both
   branches must bind `normed`) without forcing a layer-0 weight
   load that doesn't exist on disk. Encoder 3-arg
   `attention(q, k, v)` claims via `EncoderAttentionImpl` (1b);
   `rope_append` still threads through `kv_cache[layer]` because
   the 5-arg sig is shape-load-bearing — those K/V writes are the
   per-step storage path the encoder simply never reads back. The
   `attn.Wqkv` and `mlp.Wi` packed parents on disk get carved into
   virtual `attn.{q,k,v}_proj` / `mlp.{gate,up}_proj` row-slices via
   `__packed_splits__`. Encoder backbone terminator: the final
   `final_norm` rmsnorm is the last tile (no `lm_head` Gemm), which
   makes `backbone_layout` (1d) classify the body as Encoder.
   `encoder_attention` added to the FA2 prefix list in
   `ferrite-forward-macro/src/lib.rs` so the kernel-class summary
   accepts it. modernbert-base compiles at 423 tiles / 201 waves;
   1-layer test variant at 24 tiles / 12 waves; CommandR canary
   unchanged at 9/10 waves; full sibling fleet (commandr, qwen2/3,
   llama, gemma2/3, phi3, granite, mistral, mixtral, deepseek-v2/v3)
   compiles clean.
6. **1f — Dispatch rewire. ✅ DONE.** Both `ModernBertModel` and
   `ModernBertForMaskedLM` now route through `CudaModel::Ferrite` —
   added both names to `architectures` in modernbert-base{,-1-layer}.json
   so `ferrite_forward::try_load` matches either. Hidden-state return
   already exposed: `CudaModel::hidden_states()` ferrite arm calls
   `weights.forward_backbone(&ctx, ...)`, and 1d's encoder layout
   makes that semantically identical to `forward()` for modernbert.
   `CudaModel::ModernBert` variant + `vllm-cuda/src/model/modernbert.rs`
   + `modernbert_config_from_hf` deleted. Encoder detection moved
   from `matches!(m, ModernBert(_))` to a `CudaModel::is_ferrite_encoder()`
   helper that tests `m.weights.arch_name() == "modernbert"`; both
   `is_encoder` profiling-skip and CUDA-graph-disable use it. Forward
   logit path (`Self::Ferrite`) now panics for encoders with the
   same "use hidden_states()" message. modernbert-base unchanged at
   423/201; 1-layer 24/12; CommandR canary 9/10; full sibling fleet
   compiles clean. The MaskedLM head shim noted in the cold-start
   gotchas is deferred — for embedding/pooling consumers (the
   primary use case), backbone + pooler is sufficient.
7. **1g — Verify. ✅ DONE.** Five fixes landed (commits
   `0d7075b14`, `e7cf27ac2`, and the load-race fix). Cosine vs HF
   AutoModel-CLS = 0.999944 on "Hello world", ≥ 0.9998 across 4
   different-length prompts. ModernBERT-base now serves coherent
   embeddings end-to-end via `vllm serve --runner pooling`. Reference golden script:

   ```python
   from transformers import AutoTokenizer, AutoModel
   import torch
   tok = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
   m = AutoModel.from_pretrained('answerdotai/ModernBERT-base',
                                 dtype=torch.bfloat16).cuda().eval()
   inp = tok('Hello world', return_tensors='pt').to('cuda')
   with torch.no_grad():
       out = m(**inp).last_hidden_state
   hf = torch.nn.functional.normalize(out[0,0,:].float(), p=2, dim=0)
   ```
   Rust path: `vllm serve answerdotai/ModernBERT-base --runner pooling
   --pooling-strategy cls --port 8765`, then POST to `/v1/embeddings`.
   Python vLLM doesn't support `ModernBertForMaskedLM` directly
   (only `ModernBertModel`, and that loader rejects MaskedLM head
   weights). HF AutoModel is the reference instead.

   Reproducer (now that pooling-strategy "all" is wired):

   ```bash
   ./target/release/vllm serve answerdotai/ModernBERT-base \
     --runner pooling --pooling-strategy all --port 8765
   curl -sS -X POST http://localhost:8765/v1/embeddings \
     -H 'Content-Type: application/json' \
     -d '{"input":"Hello world","model":"answerdotai/ModernBERT-base"}'
   ```
   Returns 4 rows × 768 floats (one per token, all L2-normed).
   Compare against HF `AutoModel(...).last_hidden_state[0, i, :]`
   normalized; expected cosine ≈ 1.0, observed ≈ -0.3 across the
   board. Earlier sessions of /tmp/rust_emb_all.json + the python
   matrix in this PR's debug log show the consistent-flip pattern.

   Open hypotheses to bisect (priority order, all need per-layer
   hidden-state capture from a Rust-side debug hook):

   - **Rope direction / position semantics.** Encoder positions are
     [0..N) without padding; verify `ctx.fwd.positions` carries
     these for the encoder path. If positions are zero / ones for
     all tokens, every layer's Q/K rotates identically and CLS
     position becomes meaningless.
   - **Wqkv per-head packing INSIDE q.** HF's
     `qkv.view(bs, sl, 3, num_heads, head_dim)` puts q's per-head
     dim AFTER the 3-split. The packed_splits row carve gives us
     q = Wqkv[0..768], but the per-head INSIDE that 768 might be
     `[head_dim, num_heads]`-major in HF and `[num_heads, head_dim]`
     -major in our codegen, OR vice versa. flash-attn expects
     `[num_heads, head_dim]`-major.
   - **Final norm.** Confirm last RmsNorm output is what's pooled
     (vs the residual stream pre-final-norm). HF returns
     `last_hidden_state` post-final-norm; the `forward_backbone`
     terminal slot must match.
   - **EncoderAttention's softmax scale.** ATTN_SCALE = 0.125 =
     1/sqrt(64). Correct for head_dim=64.
   - **mean/var accumulator dtype.** cohere_layer_norm in BF16 vs
     HF LayerNorm in BF16 may have different reduction precision.
     Unlikely to flip cosine sign but could matter.

Each sub-step touches one component and lands independently. CommandR is the
canary for any non-modernbert-affecting change (1a, 1c, 1d).

## Cold-start gotchas (1f)

- `local_attention: 128` (window-masked bidirectional on every non-
  global layer) is NOT yet enforced kernel-side. Today every encoder
  layer runs full bidirectional FA2 via `EncoderAttentionImpl`. A
  windowed encoder Impl is the missing piece for golden parity
  on local layers; defer to 1g if dispatch alone in 1f gets
  global-only checkpoints to coherent output.
- `rope_append` writes per-step K/V into `kv_cache[layer]` even
  on the encoder side. 1f's executor must allocate a KV-cache pool
  sized for the encoder's max-token window (or a no-op-shaped
  scratch buffer), even though no `attention(..., kv_cache, ...)`
  tile reads them back. The simplest path is to reuse the existing
  `KvCachePool` allocation from the decoder fleet at the
  modernbert-input max sequence length.
- `tie_word_embeddings: true` on ModernBERT-base means
  `decoder.weight` shares storage with `embeddings.tok_embeddings`.
  The MaskedLM head (`head.dense`, `head.norm`, tied
  `decoder.weight + decoder.bias`) lives OUTSIDE the `#[forward]`
  body — 1f wires it as a separate post-backbone shim. The
  ferrite forward returns `[num_tokens, hidden_size]` from
  `final_norm`; the dispatch layer applies head.

## Cold-start gotchas (1e — historical)

- `backbone_layout` matches `lm_head` by the FIRST segment of the dotted
  weight path (`program.weights.path(id)[0] == "lm_head"`). ModernBERT's
  encoder DSL must END on a non-Gemm tile (e.g. the final `bias_add` or
  the encoder-output `add`) — otherwise a `gemm(x, w)` whose first path
  segment is `lm_head` will route to decoder mode. If a future encoder
  arch wraps its head in a Gemm with a non-lm_head weight name,
  `backbone_layout` already classifies that as Encoder (regression test
  `backbone_layout_non_lm_head_gemm_is_encoder` pins this).
- `forward_backbone` in encoder mode delegates to `forward` and returns
  ownership via `take_owned(terminal_slot)`. The encoder terminal slot
  must be Owned at the end of the backbone (typical for Add / RmsNorm /
  BiasAdd outputs). If the encoder DSL ends on a tile whose Impl emits a
  `View`-aliased output, `take_owned` will panic — the matcher /
  `output_alias` declaration on that Impl is the fix, not the
  forward-shim path.
