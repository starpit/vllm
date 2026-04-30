# GGML/GGUF — open gaps

State as of commit `96e733ccb` on `worktree-ff-gguf`. End-to-end GGUF
inference through ferrite-forward is working at tp=1 across every
supported arch. This doc tracks the remaining holes.

## Quick map

```
loader path:
  cuda_worker::load_model
    → HfModelConfig::from_path                   (vllm-model/src/weight.rs)
    → GpuWeights::from_path                      (ferrite-cuda-core/src/weights.rs)
        ├─ safetensors → existing from_dir
        └─ .gguf       → inventory-registered ferrite-kernels loader
                         GgufGpuWeights::load    (ferrite-kernels/src/ggml.rs)
                         load_gguf_into_weights  (ferrite-kernels/src/ggml.rs)
    → ferrite_forward::try_load                  (ferrite-forward/src/lib.rs)

solver Impls (impl_lib.rs, all gated on StorageFormat::Ggml):
  GgmlGemmImpl                       singleton matmul (decode + prefill)
  GgmlFusedGateUpSiluMulImpl         SwiGLU MLP
  GgmlFusedGateUpGeluMulImpl         GeGLU MLP (gemma)
  GgmlFusedQkvRopeCacheImpl          QKV + RoPE decode (incl. interleaved)
  GgmlFusedQkvRopePrefillImpl        QKV + RoPE prefill (non-interleaved)

runtime opcodes (ferrite-forward/src/instr.rs):
  Instruction::GgmlGemm
  Instruction::GgmlFusedGateUpSiluMul
  Instruction::GgmlFusedGateUpGeluMul
  Instruction::GgmlFusedQkvRopeCache       — interleaved flag selects kernel
  Instruction::GgmlFusedQkvRopePrefill

per-arch ggml preset wired:
  llama, mistral, qwen2 (incl. 2.5), qwen3, gemma2, gemma3, granite,
  commandr
```

## Open gaps, ordered by user-visible impact

### 1. TP fingerprint mismatch at tp>1 — BLOCKER for multi-GPU GGUF

**What's wrong.** The per-tensor block-aligned slicing pass is in
place inside `GgufGpuWeights::load` (ferrite-kernels/src/ggml.rs:1308–
1493) and produces a `GpuWeights` whose `quantized` and `gguf_dense`
maps store **per-rank shapes**. The codegen-emitted
`fingerprint_matches` (ferrite-forward-macro/src/codegen.rs:1135)
checks the embedding shape against full-model literals
(`vocab_lit × hidden_lit`). At tp>1 the GGUF tensors are sized
`[vocab/tp, hidden]` (ShardDim0) so the fingerprint rejects and
ferrite returns `Ok(None)` → load fails with "no ferrite ggml
variant claims arch X".

**Fix.** When `quantization.method = Ggml` and `tp_world_size > 1`,
divide `vocab_lit` by `tp_world_size` in the fingerprint. The
`tp_world_size` is already in scope at the codegen call site (it's
threaded through `forward!` fanout). About 10 lines in
`emit_fingerprint_check`.

**Verify.** Once the fingerprint passes, the runtime kernel calls
should already work (per-rank q_proj has shape `[out/tp, in]`,
activations are `[tokens, hidden]`, GgmlGemmImpl matmul is shape-
correct; o_proj/down_proj kernel reads per-rank in_features and the
solver's existing `tp_lowering::insert_all_reduces` already inserts
AllReduce after row-parallel gemms). Test at tp=2 with NCCL on a
two-GPU box.

**Refuse-at-load coverage.** Loader already errors when
`(in_features / tp) % block_size != 0` for ShardDim1 tensors
(ggml.rs:1431). Q4_K block size is 256; for Llama-3.2-1B at tp=8,
in_features=2048, per-rank=256 → exact fit. tp=4 → 512 → fits. tp=2
→ 1024 → fits. Larger tp on smaller models would refuse.

### 2. Llama-2 SentencePiece tokenizer — affects Llama-2 / Mistral-1 GGUFs

**What's wrong.** `gguf_tokenizer` in vllm-model/src/gguf.rs:147
returns `Ok(None)` when `tokenizer.ggml.model != "gpt2"`. The Llama-2
family ships `model = "llama"` (SentencePiece-BPE). When the GGUF
has no sibling `tokenizer.json`, tokenizer loading fails.

**Fix.** Add a SentencePiece-BPE construction path. Differs from gpt2
BPE in three places:
- Tokens use `▁` (U+2581) as space prefix instead of `Ġ`.
- Pre-tokenizer is Metaspace, not ByteLevel.
- Decoder is Metaspace.
- Scores from `tokenizer.ggml.scores` matter (Unigram-style). For
  vanilla SP-BPE the merges array drives BPE; scores are advisory.

Rough sketch:

```rust
if model == "llama" {
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()?;
    let mut tok = Tokenizer::new(bpe);
    tok.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Always)));
    tok.with_decoder(Some(MetaspaceDecoder::new('▁', PrependScheme::Always)));
    tok.add_special_tokens(...);
    return Ok(Some(tok));
}
```

The HF `tokenizers` crate has both `Metaspace` types; verify
construction args match the crate's current API. Add tokens with
type=USER_DEFINED (4) and CONTROL (3) — see existing gpt2 path.

**Verify.** Same byte-equality test against an official
`tokenizer.json` for a Llama-2 model on plain text + chat prompts.

### 3. Output quality at small Q4_K_M — diagnostic, not necessarily a bug

**Symptom.** Llama-3.2-1B and -3B at Q4_K_M produce coherent-but-weak
chat output (e.g. `"The 2023"` then EOS) on simple prompts. Same
output as the retired hand-written GGUF path produced — confirmed by
direct comparison earlier in the worktree's history.

**Likely causes (not fixed):**
- Q4_K_M dequant accuracy at this scale is genuinely weak.
- Sampling: chat is greedy by default at temperature=1. The
  `vllm chat` default may not match what gives best output.
- Chat template + first-token bias on tiny instruct models.

**Verify it's not a ferrite bug.** Run a 7B+ model through the same
path. If output is coherent, this is a small-model artifact, not a
ferrite issue. Use bartowski's Llama-3.1-8B-Instruct-Q4_K_M.gguf or
similar.

### 4. No fused interleaved-RoPE prefill kernel — affects Cohere/commandr prefill

**What's wrong.** `GgmlFusedQkvRopePrefillImpl::matches`
(impl_lib.rs:12011) explicitly rejects `RopeAppendInterleaved` so
commandr prefill falls through to standalone `GgmlGemm` ×3 +
`RopeAppendRefImpl`. Functionally correct, but slower.

**Fix.** Either:
- Add a `fused_qkv_interleaved_rope` (non-cache) kernel to
  ferrite-kernels/csrc/ and dispatch on `interleaved` in the prefill
  handler, OR
- Accept the standalone fallback as the long-term answer for
  commandr prefill (it's the only arch using interleaved RoPE; not
  worth a kernel).

Standalone fallback path is fine for now.

### 5. TQ1_0 / TQ2_0 ternary quants — not in dtype enum

**What's wrong.** `GgmlDType` (ferrite-cuda-core/src/ggml_quant.rs:18)
doesn't list TQ1_0 (tag 34) or TQ2_0 (tag 35). `from_u32` returns
`None` for these tags so loading any GGUF that uses them errors with
"unsupported GGUF dtype".

**Fix.** Two parts:
- Add enum variants + `from_u32` / `type_size` / `block_size` arms
  (mechanical; type sizes are documented in llama.cpp's
  `ggml-quants.h`).
- Write CUDA kernels: `dequantize_block_tq1_0_*`,
  `dequantize_block_tq2_0_*`, `mul_mat_vec_tq*_q8_1`. No reference
  exists in ferrite-kernels/csrc today; port from llama.cpp's
  `ggml-cuda/`.

Lower priority — TQ types are uncommon in distributed GGUFs.

### 6. IQ types in MoE forward — affects IQ-quantized MoE GGUFs

**What's wrong.** `ggml_moe_forward` (ferrite-kernels/src/ggml.rs:1114
ish) dispatches only Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_0 / Q4_0
/ Q4_1 / Q5_0 / Q5_1. IQ types panic with "unsupported dtype for
ggml_moe_forward". MoE arches today are DeepSeek and Qwen3-MoE; an
IQ-quantized GGUF of either would fail.

**Fix.** Add `launch_indexed_moe_forward_iq{4_nl,4_xs,1_s,1_m,
2_xxs,2_xs,2_s,3_s}_q8_1` FFI bindings + dispatch arms. The kernels
need to be ported from llama.cpp; non-trivial.

Lower priority.

### 7. lm_head / embedding stays dequantized at load — memory cost

**What's wrong.** GGUF ships `output.weight` (lm_head) typically as
Q6_K. The loader dequantizes it to BF16 at load (ggml.rs:1474–1490)
to keep the dense lm_head Linear path. For Llama-3.2-1B:
`lm_head = [128256, 2048]`. Q6_K is ~13.4 bits/element, BF16 is 16
bits — modest delta. For Llama-3-70B:
`lm_head = [128256, 8192]` → ~210 MiB stays as Q6_K savings, but
~840 MiB lives as dequantized BF16 vs the ~640 MiB it would take
quantized.

**Fix.** Route lm_head through `LinearLayer::Ggml` via
`take_quantized_linear` + `GgmlGemmImpl`. Two flips:
- `storage_format_for_weight` at quantization.rs:725 currently
  forces lm_head to `Dense` for Ggml (matches the eager dequant in
  the loader). Remove that special case.
- Loader at ggml.rs:1474 currently dequantizes when
  `is_lm_head`. Drop `is_lm_head` from that branch so quantized
  lm_head stays as `GgmlStorage`.

Saves ~25% lm_head memory on big models. Verify with an A/B that
output stays byte-equivalent.

### 8. RopeAppendInterleaved + Ggml deferral asymmetry

**What's wrong.** `rope_append_has_fused_qkv_upstream`
(impl_lib.rs:5891) defers to fused for interleaved + Ggml at decode
(Cache fused matches), and falls through to `RopeAppendRefImpl` at
prefill (Prefill fused rejects interleaved). The current
implementation has a hard `return false` for the
RopeAppendInterleaved+Ggml combination at the bottom of the helper
(impl_lib.rs:5928), which means **both** decode and prefill fall
back to standalone for commandr. The fused Cache claim still wins
at decode because its claim covers more tiles, but the deferral
gate is overly broad.

**Fix.** Tighten the gate to check the workload constraint —
defer to fused only if a fused Impl will *actually* match at this
workload. Out of scope until commandr prefill matters more; the
current behavior is correct.

## Crate-level layering (for future work)

```
   vllm-executor/cuda_worker     — thin dispatcher, format-agnostic
        ↓
   vllm-serve/init               — chat template + tokenizer wiring
        ↓
   vllm-model                    — config + tokenizer + GGUF parser
        ↓
   ferrite-forward(-macro)       — solver, codegen, Impls
        ↓
   ferrite-kernels               — CUDA kernels + GGUF loader
        ↓
   ferrite-cuda-core             — types, GpuWeights, inventory seam
```

`ferrite-cuda-core::gguf_loader` is the cross-crate seam:
`ferrite-kernels` submits `GgufLoaderRegistration` via
`inventory::submit!`; `GpuWeights::from_gguf_file` calls through it.
Adding a future format (or refactoring the GGUF loader) preserves
this layering — don't put format-specific code in `cuda_worker`.

## Verification commands

```bash
# Per-arch compile (no GPU needed):
cargo build -p vllm-cli --features cuda --release

# Format check, lint:
cargo fmt -p ferrite-cuda-core -p ferrite-kernels -p ferrite-forward-macro -p ferrite-forward -p vllm-model -p vllm-serve -p vllm-executor
cargo clippy -p ferrite-cuda-core -p ferrite-kernels -p ferrite-forward-macro -p ferrite-forward -p vllm-model -p vllm-serve -p vllm-executor --features cuda --no-deps -- -D warnings

# Inference smoke test (requires GPU + a Llama-3.2 GGUF):
target/release/vllm batch \
  -m /path/to/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  -i /tmp/chat_test.jsonl -o /tmp/out.jsonl
```

## Memory / soft constraints to honor

- Ferrite owns weight loading. Don't add format-specific logic to
  cuda_worker — extend `GpuWeights::from_path` and the inventory-
  registered loader instead. cuda_worker stays a dispatcher.
- For new fused Impl families, clone the closest existing pattern
  (Marlin / Bnb4 / Fp8 / Ggml). Each family runs ~150–250 lines of
  matches/required_weights/opcode_shape/fan_out.
- Test against real models. Synthetic tests don't catch tokenizer
  / chat-template / RoPE bugs. The byte-identical-vs-tokenizer.json
  comparison harness lives in the commit history (`tok_compare.rs`
  example) if you need to revive it.
