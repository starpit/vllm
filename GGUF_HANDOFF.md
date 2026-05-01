# Ferrite GGUF integration — handoff

Branch: `worktree-ff-gguf`. Status as of 2026-05-01.

## Headline

**Llama-3.x Q4_K_M GGUFs run end-to-end through ferrite-forward and produce coherent output** at tp=1. Verified on `unsloth/Llama-3.2-3B-Instruct-GGUF`, `unsloth/Llama-3.2-1B-Instruct-GGUF`, `bartowski/Llama-3.2-3B-Instruct-GGUF`. Other archs (Qwen2/3, Gemma2/3, CommandR, Mistral, Phi3, DeepSeekV2/3, Granite) compile but are unverified — Qwen2 specifically is **known-broken** end-to-end (model loads, output is gibberish even through `FERRITE_USE_REFERENCE=1` reference forward).

## What landed (2026-04-30 → 2026-05-01)

The earlier sections of this doc covered the foundation: data-type relocation, `GpuWeights` GGUF backing maps, `load_gguf_into_weights` entry point, `GgmlGemmImpl` solver Impl, codegen FieldLoad arm, `ggml.json` preset. All that is still in place. The 2026-05-01 session closed the gap from "compiles" to "produces correct output":

### Five correctness bugs fixed

1. **`FERRITE_MODELS=<base>` did not include overlay variants.** `set.contains(stem)` only matched bare bases — every `<base>-ggml`, `<base>-awq-gemm`, etc. got filtered out. Now uses prefix-with-dash matching: `FERRITE_MODELS=llama-3.2-3b` keeps the dense base AND all overlays. (`config.rs`)

2. **`load_dense_concat` couldn't see `gguf_dense`.** `tensor_info` + `take_into` only check the safetensors mmap map. FP16/F32 GGUFs land all weights in `gguf_dense`. Added `load_gguf_dense_concat` helper that uses `take()` (which already checks all three backings) + per-row D2D copy. (`layers.rs`)

3. **GGUF `<arch>.context_length` and rope_scaling fields disagree with HF JSON.** Qwen2.5-0.5B GGUF reports 8192, JSON reports 32768; unsloth Llama-3 GGUFs omit rope_scaling entirely. Fingerprint rejected on `max_pos_check`. Now suppresses those hf-config-disambiguation hints when `gw.is_gguf()` — the compile-time variant's baked rope/max_pos values stay authoritative. (`cuda_worker.rs`)

4. **Tokenizer reconstruction added a leading space on every word.** `ByteLevelPre::default()` has `add_prefix_space=true, use_regex=true`, so reconstructed tokenizers tokenized "user" as `Ġuser` (token 1196) instead of "user" (token 872). Now uses `ByteLevelPre::new(false, false, false)` everywhere with a per-arch regex Split (`llama-bpe`, `qwen2`, gpt2 default). Decoder also fixed (`ByteLevelDecoder::new(false, false, false)`). Tokens now match HF byte-for-byte for Llama-3 and Qwen2 GGUFs. (`gguf.rs`)

5. **Q/K row permutation: GGML uses interleaved pairs, HF uses split halves.** llama.cpp's `convert_hf_to_gguf.py` pre-permutes q_proj/k_proj rows for archs that derive from `LlamaModel` (LlamaModel.modify_tensors calls `permute()`). ferrite's RoPE follows HF split-halves, so we now un-permute on load. The permutation is row-only, so quant-block rows can be moved bytewise. **Gated to `general.architecture == "llama"`** — Qwen2/3, Gemma, Cohere, Phi3, DeepSeek don't permute and would silently break if we un-permuted them.
   Verified vs safetensors: bartowski Llama-3.2-3B Q4_K_M q_proj un-permute → 0.07 maxdiff (Q4_K noise level), raw → 1.3 (broken). (`ggml.rs`)

### Bonus — bias dtype mismatch in GgmlLinear

`GgmlLinear::forward` produces F32 output and called `bias_add_inplace(out_f32, bias)` where bias was cast to model_dtype (BF16) at GGUF load time. `bias_add_inplace` reinterprets bias bytes as `out.dtype()` → silent garbage. Now casts bias to F32 inside the forward when dtypes disagree. (`layers.rs`)

### UX

- **Hard error** when GGUF source loads but no ferrite-forward variant fingerprint matched. Previously fell through silently to the dense-load path which fails deep with `weight not found`. New error names the arch + tp_world and points at `FERRITE_MODELS` and `quantizations.json`. (`cuda_worker.rs`)

## How to use

```sh
cd vllm-rs
cargo build -p vllm-cli --features cuda --release
./target/release/vllm chat unsloth/Llama-3.2-3B-Instruct-GGUF \
    --enforce-eager --quick "What is 2+2?" --max-tokens 16 --temperature 0.0
# expected: "2 + 2 = 4"
```

`FERRITE_MODELS=<base-stem>` narrows codegen during dev iteration; the prefix matches all overlays for that base, so `FERRITE_MODELS=llama-3.2-3b` builds the dense `llama-3.2-3b`, the `-ggml`, the `-awq-gemm`, etc.

## Open work

### Blockers (correctness)

1. **Qwen2 GGUF garbage**. Bartowski Qwen2.5-0.5B Q4_K_M produces gibberish even through `FERRITE_USE_REFERENCE=1` (which routes every Ggml linear through dequant+cuBLAS, bypassing the MMVQ kernel entirely). Bytes match HF safetensors at the weight level (raw maxdiff = 0.047 = Q5_0 quant noise) and tokenizer matches HF byte-for-byte. So the bug is downstream of weight loading and tokenization but specific to Qwen2 — likely in some kernel path that doesn't exercise on Llama (qwen2 has biased q/k/v but my bias-cast fix is in place; rope_theta is compile-time-baked and matches; rms_norm_eps matches). **Next debugging step**: dump hidden states after layer 0 for both safetensors and GGUF and compare; or compare logits at the first decode position.

2. **Audit other archs.** The qk-permute is gated to `arch == "llama"` so Qwen/Gemma/CommandR/Phi3/DeepSeek shouldn't be silently broken by it, but I haven't end-to-end-verified any of them. Mistral GGUFs in the wild typically tag as `arch == "llama"` (same convert class) — should work but untested.

### Plumbing

3. **TP > 1.** Un-permute respects head boundaries by construction (verified `(rows_per_rank) % head_dim == 0` for Llama-3.2-3B at tp=2: 1536/128=12 heads/rank). Untested. Llama-3.2-1B has 32 q heads / 8 kv heads — also clean tp=2.

4. **Llama-2 SentencePiece tokenizer.** `tokenizer.ggml.model = "llama"` (older L2 GGUFs) returns `Ok(None)` from `gguf_tokenizer`; falls back to sibling `tokenizer.json`. Most older L2 GGUFs ship without a sibling tokenizer.json so they fail. Add a SentencePiece reconstruction path.

5. **Integration test.** No CI test exercises the GGUF→ferrite path. Add at least a smoke test for Llama-3.2-1B-Q4_K_M end-to-end via `vllm chat`.

6. **3D MoE expert tensors.** `gguf_format` reverses dim order for 1-D and 2-D tensors via `dimensions.reverse()` (it always reverses); for 3-D MoE experts I haven't verified that the resulting layout matches HF's `[num_experts, out_dim, in_dim]` convention.

## Files touched (this session — uncommitted)

```
vllm-rs/Cargo.lock                                     (vllm-examples added vllm-model + anyhow dev-deps)
vllm-rs/crates/ferrite-forward-macro/src/config.rs     FERRITE_MODELS prefix-match
vllm-rs/crates/ferrite-kernels/src/ggml.rs             qk un-permute, arch gate, qk_meta extraction
vllm-rs/crates/ferrite-kernels/src/layers.rs           load_gguf_dense_concat, GgmlConcat reference path,
                                                       GgmlLinear bias dtype cast
vllm-rs/crates/vllm-executor/src/cuda_worker.rs        suppress hf-fp hints for GGUF, GGUF-no-match hard error
vllm-rs/crates/vllm-model/src/gguf.rs                  per-arch tokenizer regex table, decoder fix
vllm-rs/crates/vllm-examples/Cargo.toml                + gguf_tok_dump example
vllm-rs/crates/vllm-examples/examples/gguf_tok_dump.rs NEW (diagnostic)
```

## Diagnostic scripts kept in /tmp

- `/tmp/embed_compare.py` — compare embed_tokens between HF safetensors and GGUF
- `/tmp/weight_compare_llama.py` — same for layer-0 weights, with Q4_K_M dequant
- `/tmp/unpermute_test.py` — verify q/k row permutation hypothesis on Llama
- `/tmp/qwen_bartowski.py` — same for Qwen (confirmed Qwen does NOT permute)
- `/tmp/tok_compare.py` / `/tmp/qwen_tok_compare.py` — HF tokenizer reference

## Pre-2026-05-01 history (foundation)

Historical sections preserved at the bottom of this doc for context — they describe the multi-day foundation work (data-type relocation, FieldLoad arm, GgmlGemmImpl scaffolding) that this session built on. The headline above is the user-facing state.
