# Ferrite GGUF integration — handoff

Branch: `worktree-ff-gguf`. Tip: `92df20431` (2026-05-01 evening).

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

Verified at tp=1: Llama-3.2-1B/3B (unsloth + bartowski), Qwen2.5-0.5B
(bartowski), Qwen3-0.6B (unsloth), Granite-3.1-2B (bartowski Q4_K_M
+ Q8_0) — all coherent at greedy. Gemma-2-2B / Gemma-3-1B GGUFs load
+ dispatch but decode empty (SentencePiece blocker, see below).

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

## Open follow-ups

* **Audit other archs end-to-end.** With P2 landed, each arch's
  GGUF support is a small `quantizations.json` edit. Granite +
  Gemma2 + Gemma3 specs landed; the concrete remaining JSON edits
  are: Phi3 fused qkv/gate-up (also needs LongRoPE for Phi-3.5/Phi-4
  variants — out of current arch scope); CommandR (no small GGUF
  available — 35B+ is the smallest published, deferred); DeepSeek
  V2/V3 MLA + MoE (V2/V3's bespoke MLA dim derivation isn't
  expressible as pure data and needs follow-up).
* **TP > 1 untested for GGUF** (task #8). Un-permute respects head
  boundaries by construction (see comment in `ggml.rs`); just needs
  a real run.
* **Llama-2 SentencePiece tokenizer.** `tokenizer.ggml.model =
  "llama"` GGUFs (Mistral, Llama-2, Gemma) fall through to sibling
  tokenizer.json. Bartowski Mistral / Gemma repos ship none, so
  output decodes to empty even though the model loads + dispatches
  correctly. Same blocker for any SentencePiece arch.
* **Integration test.** No CI test exercises the GGUF→ferrite path.

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
