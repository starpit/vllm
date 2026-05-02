# Ferrite GGUF integration — handoff

Branch: `worktree-ff-gguf`. Tip: `d4519563d` (rebased onto
`ff-interpreter` 2026-05-02; `ferrite-model-deepseek-v3-flat`
landed in the workspace via the rebase).

## Hit the ground running

```
# Build (CPU-runnable; CUDA needed for actual inference).
cargo build --manifest-path vllm-rs/Cargo.toml -p vllm-cli --features cuda --release

# Smoke test (GPU). Greedy "Paris" sanity on a verified-coherent fixture:
timeout 90 ./vllm-rs/target/release/vllm chat \
  --model ~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/b69aef112e9f895e6f98d7ae0949f72ff09aa401/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --max-tokens 30 --temperature 0 --prompt "What is the capital of France?"
# Expected: "The capital of France is Paris."

# CPU-only verification — locks down tokenization on every supported arch:
cargo test --manifest-path vllm-rs/Cargo.toml -p ferrite-gguf --release -- --include-ignored
# Expected: 19 tests pass; integration test compares ids against HF reference
# for Llama-3.2-1B/3B, Qwen2.5-0.5B, Qwen3-0.6B, Granite-3.1-2B,
# Mistral-7B-v0.3, Phi-3.5-mini, Gemma-2-2B, Gemma-3-1B.

# Narrow build to one arch when iterating (minutes → seconds):
FERRITE_MODELS=llama-3.2-1b-instruct cargo build --manifest-path vllm-rs/Cargo.toml -p vllm-cli --features cuda --release
```

GGUFs already cached locally for Llama-3.2-1B/3B, Qwen2.5-0.5B,
Qwen3-0.6B, Granite-3.1-2B, Mistral-7B-v0.3, Phi-3.5-mini,
Gemma-2-2B, Gemma-3-1B, CommandR-35B-writer-v2 (IQ1_S + Q2_K).
Paths in `crates/ferrite-gguf/src/lib.rs::gguf_tokenizer_matches_hf_reference`
test fixtures.

## Status

GGUF-as-a-format is now fully owned by `ferrite-gguf` (parser,
`GgufFile`, tokenizer reconstruction, `gguf_model_config`,
`gguf_to_hf_name`). Per-arch GGUF specifics are declarative — each
arch's `configs/quantizations.json` lists `"ggml"` (string form for
defaults) or `{"ggml": {...}}` (qk_permute, gguf_arch override,
tensor_renames, metadata_u32/f32, metadata_defaults,
llama3_rope_scaling_inference). The macro reads it and forwards every
field as static data into a `ferrite_gguf::register!{...}` block.
There are NO arch-keyed god-switches anywhere; no GGUF code in
`vllm-model`; no GGUF code in any model crate's `lib.rs`.

Verified at tp=1: Llama-3.2-1B/3B, Qwen2.5-0.5B, Qwen3-0.6B,
Granite-3.1-2B, Mistral-7B-Instruct-v0.3, Gemma-2-2B,
Gemma-3-1B — tokenization round-trips against `LlamaTokenizerFast` /
`MistralTokenizerFast` / `GemmaTokenizerFast` on the standard
prompt. The SentencePiece path (`tokenizer.ggml.model = "llama"`)
synthesizes BPE merges from vocab+scores via a direct port of HF's
`generate_merges` — see `build_sentencepiece_tokenizer`.

## Phases landed

* **Pre-refactor correctness fixes** — the five bugs that closed
  Llama / Qwen output at tp=1: FERRITE_MODELS prefix match, GGUF
  dense-concat, hf-fp suppression for GGUF, ByteLevel-pre defaults,
  qk un-permute. Plus Qwen-bias rename gap (`09a6694cf`). See commits
  `74a50c23d` ← `09a6694cf`.
* **Mistral arch alias** (`12341e7f6`) — `try_load` falls through on
  Ok(None); Mistral declares `extra_hf_arches = ["LlamaForCausalLM"]`
  so Mistral GGUFs (which report `general.architecture = "llama"`)
  route through the Llama spec then land on Mistral's fingerprint.
* **P1: GGUF format relocation** (`71869363e`) — `GgufFile`,
  `gguf_format`, tokenizer, chat-template, the (transitional)
  god-switches all moved out of `vllm-model` into the new
  `ferrite-gguf` crate. `HfModelConfig::from_path` no longer
  GGUF-aware; cuda_worker handles the .gguf branch via ferrite-gguf.
* **P2: declarative spec** (`92df20431`) — `GgufArchSpec` is pure
  data; `register!` macro is the only construction site;
  `ferrite-forward-macro/src/config.rs::load_gguf_spec` parses the
  ggml entry; the macro emits the registration. Every arch-keyed
  `match` arm in `gguf_model_config` / `gguf_to_hf_name` /
  `ferrite-kernels::ggml.rs` deleted in favor of registry lookup +
  declarative data.
* **CommandR + dual-claim load fix.** Two pieces:
  (a) `gguf_arch: "command-r"` override on the commandr spec so
  the GGUF tag (with hyphen) routes to the `commandr` Rust ident.
  (b) `take_quantized_linear` is now non-destructive and
  `load_dense_concat_or_ggml` no longer frees per-prefix storages
  after byte-pack. Required because commandR's interleaved RoPE
  causes `GgmlFusedQkvRopePrefillImpl` to reject at prefill, so
  codegen emits BOTH a singleton GgmlGemm accessor (per-prefill q/k/v)
  AND a fused-QKV accessor (decode); both load paths now succeed.
  Cost: each per-prefix QKV/gate-up buffer stays alongside its
  byte-packed twin. Negligible on Llama-1B (~32 MB), bounded by
  arch size. Plus a `rope.scaling.type = "none"` filter in
  `gguf_model_config` so commandR's HF-no-scaling fingerprint
  matches (llama.cpp emits the literal string `"none"`).

## Open follow-ups

* **Audit other archs end-to-end.** With P2 landed, each arch's
  GGUF support is a small `quantizations.json` edit. Granite +
  Gemma2 + Gemma3 + CommandR + Phi3 specs landed (Phi3 covers
  Phi-3-mini-4k; Phi-3.5/Phi-4 LongRoPE variants need separate
  follow-up).
* **DeepSeek V2 / V3 GGUF.** Three-piece job, not the "bespoke MLA
  dim derivation" the original handoff implied. MLA dims fingerprint-
  match cleanly through the existing JSON-baked variants
  (`deepseek-v2-lite`, `deepseek-v3-tiny`, …); the code-side work
  is elsewhere:
  1. **Add `quantizations.json` to both crates.** Today neither
     ferrite-model-deepseek-v2 nor -v3 has one, so the macro emits
     no `register!` and `find_spec("deepseek2")` fails. Pure data:
     `{"ggml": {"gguf_arch": "deepseek2"}}` plus any `metadata_u32`
     reads we want to surface (`{arch}.attention.kv_lora_rank`,
     `{arch}.attention.q_lora_rank`, etc.). The MLA tensor renames
     (`attn_q_a / q_b / kv_a_mqa / kv_b / kv_a_norm`) and MoE
     renames (`ffn_gate_inp`, `ffn_*_exps`, `ffn_*_shexp`,
     `exp_probs_b.bias`) are already in `default_gguf_to_hf_name`
     — no overrides needed.
  2. **Yarn `rope_scaling` reconstruction.** Code, ~15 lines.
     `gguf_model_config` currently builds `extra["rope_scaling"]`
     with `factor` + `original_max_position_embeddings` and
     special-cases `llama3` for its low/high-freq factors. Yarn
     needs the parallel treatment: read `{arch}.rope.scaling.{
     beta_fast, beta_slow, mscale, mscale_all_dim}` and stamp
     them into the same nested object. Without this, the
     compile-baked `rope_scaling_hash` for V2/V3 won't match the
     runtime config and the fingerprint dispatcher rejects.
     Declarative `metadata_f32` writes flat into `extra`, not
     into nested objects, so this stays in code.
  3. **MoE fused-3D expert tensor loader audit.** GGUF stores
     experts as a single 3D `ffn_*_exps.weight` (shape
     `[num_experts, n, k]`). Default rename emits
     `mlp.experts.fused_*`. Need to confirm the DeepSeek-V3
     forward arch's MoE loader consumes that fused storage
     directly — `ggml_moe_forward` (`ferrite-kernels/src/ggml.rs`)
     already has `launch_indexed_moe_forward_q*_q8_1` per-quant
     paths so the kernel side is likely fine; the question is
     whether the load-time path produces the storage shape
     those kernels expect. Empirical: drop in the spec, run, see
     where it errors.

  Q-path coverage in the project (`v3-flat` is in this worktree's
  workspace post-rebase):

  | Q path        | V2 MoE (softmax, scale=1)         | V3 MoE (sigmoid + noaux_tc, scale=2.5)  |
  | ------------- | --------------------------------- | --------------------------------------- |
  | flat (no LoRA) | `ferrite-model-deepseek-v2`  ✓  | `ferrite-model-deepseek-v3-flat` ✓ (Moonlight, Kimi K2 family) |
  | LoRA (q_a/q_b) | (DeepSeek-V2 non-Lite — n/a)    | `ferrite-model-deepseek-v3`  ✓          |

  Suggested order of attack:
  1. Land step 1 (quantizations.json) on all three deepseek crates.
     Try a tiny fixture — `bzantium/tiny-deepseek-v3` ships
     safetensors only (no GGUF) so for end-to-end you need a
     real GGUF. Smallest known: `unsloth/DeepSeek-V2-Lite-GGUF`
     `Q4_K_M` (~10 GB, fits L4); for V3-flat use
     `unsloth/Moonlight-16B-A3B-Instruct-GGUF` (BF16 16 GB or
     Q4_K_M ~9 GB, fits L4).
  2. Run the smoke-test command above against the GGUF. Expected
     failure mode if step 2 (yarn rope_scaling) isn't done:
     fingerprint dispatch rejects (`hf.rope_scaling_hash` mismatch).
     Add the yarn case in `crates/ferrite-gguf/src/lib.rs::
     gguf_model_config` next to the existing llama3 special case.
  3. If load proceeds and ggml_moe_forward fires, MoE storage
     shape is fine. If it errors at the experts-load step, audit
     `ferrite-kernels::layers` MoE loader vs. the rename's
     `mlp.experts.fused_*` target.
* **CommandR forward correctness on a bigger GPU.** Reproducer:
  ```
  ./vllm-rs/target/release/vllm chat \
    --model ~/.cache/huggingface/hub/models--mradermacher--command-r-35b-writer-v2-i1-GGUF/snapshots/dcb373142e0c8db72dbce68e964db438f4a95fc2/command-r-35b-writer-v2.i1-IQ1_S.gguf \
    --max-model-len 256 --max-tokens 30 --temperature 0 --prompt "What is the capital of France?"
  ```
  Today on an L4 this runs through prefill (kernels dispatch via
  `[ggml dispatch] first matmul via IQ1_S`/`IQ2_XXS`) and exits 0
  having generated only token 0 (PAD) repeatedly — also under
  `FERRITE_USE_REFERENCE=1` (dequant+cuBLAS), so it's not a
  quantized-matmul bug. Q2_K (~13 GB weights + 4 GB embed dequant
  + KV cache) OOMs on 24 GB. On a >40 GB GPU, run Q2_K with the
  same command; if it produces coherent text it's an IQ1_S
  quality cliff, otherwise it's a real arch bug in commandR's
  GGUF forward path (CohereLayerNorm / RopeAppendInterleaved on
  singletons / parallel attn+MLP residual / logit_scale × 0.0625).
* **TP > 1 untested for GGUF.** Un-permute respects head boundaries
  by construction (see comment in `ferrite-kernels/src/ggml.rs`).
  Reproducer on a 2-GPU box, any verified-coherent fixture:
  ```
  ./vllm-rs/target/release/vllm chat \
    --model ~/.cache/huggingface/hub/models--unsloth--Llama-3.2-3B-Instruct-GGUF/.../Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --tensor-parallel-size 2 --max-tokens 30 --temperature 0 \
    --prompt "What is the capital of France?"
  ```
  Expected: same coherent output as tp=1. Anything else is a
  TP-side bug — most likely shard-aware un-permute in the
  `qk_permuted` path of `ferrite-kernels/src/ggml.rs::
  load_gguf_into_weights`.
* **Integration test.** Tokenization is locked down by
  `cargo test -p ferrite-gguf -- --include-ignored` —
  `gguf_tokenizer_matches_hf_reference` covers Llama-3.2-1B/3B,
  Qwen2.5-0.5B, Qwen3-0.6B, Granite-3.1-2B, Mistral-7B-v0.3,
  Phi-3.5-mini, Gemma-2-2B, Gemma-3-1B against HF reference ids.
  Forward-pass / inference-coherence integration tests still
  unwritten (would need a GPU CI runner).

## Where to look first

1. `crates/ferrite-gguf/src/spec.rs` — `GgufArchSpec` data shape
   and the `register!` macro.
2. `crates/ferrite-forward-macro/src/config.rs::load_gguf_spec` —
   the JSON parser the macro feeds from.
3. `crates/ferrite-model-llama/configs/quantizations.json` —
   reference structured ggml entry (`qk_permute`,
   `llama3_rope_scaling_inference`).
4. `crates/ferrite-gguf/src/spec.rs::apply_metadata` — the runtime
   side that consumes the declarative metadata reads.
