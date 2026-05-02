# Ferrite GGUF integration — handoff

Branch: `worktree-ff-gguf`. Tip: see `git log -1`. Last big landing:
the Gemma `norm_weight_offset` fix (rmsnorm garbage-output bug, see
"Phases landed" below). Rebased onto `ff-interpreter` 2026-05-02.

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

**Inference-coherent at tp=1** (greedy "Paris" smoke through `vllm
chat`): Llama-3.2-1B/3B, Qwen2.5-0.5B, Qwen3-0.6B, Granite-3.1-2B,
Mistral-7B-Instruct-v0.3, **Gemma-2-2B**, **Gemma-3-1B**.

**Tokenizer-only verified** (CPU round-trip, NO inference run):
Phi-3.5-mini. The SentencePiece path (`tokenizer.ggml.model =
"llama"`) synthesizes BPE merges from vocab+scores via a direct port
of HF's `generate_merges` — see `build_sentencepiece_tokenizer`.

**Known broken / out of scope here:** Phi-3.5-mini inference (loader
asks for `mlp.gate_proj.weight` while the Phi3 spec renames to fused
`gate_up_proj` — LongRoPE follow-up); CommandR-35B (decodes only PAD
on L4, needs >40 GB GPU to disambiguate IQ1_S quality cliff vs real
arch bug).

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
  *Superseded* — see "GGUF dispatch via gguf_archs" below.
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
* **GGUF dispatch via `gguf_archs`.** GGUF tags are family-level
  (`"deepseek2"` covers V2 / V3-LoRA / V3-flat; `"llama"` covers Llama
  + Mistral); the previous `extra_hf_arches` hack had each non-canonical
  crate manually claim the canonical owner's HF class string so dispatch
  fanned out. New design: `FerriteArchRegistration` gains a
  `gguf_archs: &[&str]` field, `try_load` filters by
  `(hf_arches ∪ gguf_archs).contains(arch_hint)`, and `gguf_model_config`
  stamps the GGUF tag itself (`"deepseek2"`) into `architectures[0]`
  instead of an HF class string. Per-crate `quantizations.json` now
  carries an explicit `gguf_arch` (used both for the registration's
  `gguf_archs` and — when this crate is the canonical owner — for
  `ferrite_gguf::register!`). Non-canonical claimants set
  `register_spec: false` so a single deterministic spec covers each
  gguf_arch (find_spec collisions gone). `extra_hf_arches` removed
  from the `#[forward]` macro entirely; Mistral / V3 / V3-flat opt in
  via `register_spec: false`. Verified Llama + Mistral GGUFs still
  coherent post-refactor; DeepSeek-V2-Lite + Moonlight GGUFs now
  fingerprint-match correctly via the new path (V2 forward catches
  V2-Lite, V3-flat forward catches Moonlight).
* **Gemma `norm_weight_offset` (rmsnorm garbage-output bug).**
  llama.cpp's converter pre-bakes `+1` into every Gemma rmsnorm
  weight (incl. QK and final norms) so a vanilla `rmsnorm(x, w)`
  matches HF's `(1+w)*x`. Ferrite's DSL writes `rmsnorm(x, weight +
  1.0)` and `ScalarOffsetRmsNormImpl` adds `+1` at kernel time —
  resulting in `(2+w_orig)*x` and ~2× scaled activations every
  norm. Pre-fix output: `"1.1.1.4. in tartalomajánló…"` /
  `"ia theks. on and கீض sweep…"`. Symmetric across Gemma2 + Gemma3,
  reproduces under `FERRITE_USE_REFERENCE=1` (so not a quant kernel
  bug). Fix: declarative `norm_weight_offset: f32` field on
  `GgufArchSpec` (default 0.0), plumbed through `quantizations.json`
  → macro → register; subtracted from each F32 norm element in the
  GGUF F32→model-dtype load path so canonical raw `w` lands on GPU.
  Set to `1.0` for Gemma2 + Gemma3 specs. Verified coherent post-fix
  on both fixtures.

## Open follow-ups

* **Audit other archs end-to-end.** With P2 landed, each arch's
  GGUF support is a small `quantizations.json` edit. Granite +
  Gemma2 + Gemma3 + CommandR + Phi3 specs landed (Phi3 covers
  Phi-3-mini-4k; Phi-3.5/Phi-4 LongRoPE variants need separate
  follow-up — see Phi-3.5-mini bullet below).
* **Phi-3.5-mini GGUF load error.** Reproducer:
  ```
  ./vllm-rs/target/release/vllm chat \
    --model ~/.cache/huggingface/hub/models--bartowski--Phi-3.5-mini-instruct-GGUF/snapshots/6d70da17e749a471ccb62ade694486011a75cda3/Phi-3.5-mini-instruct-Q4_K_M.gguf \
    --max-tokens 5 --prompt "Hi"
  ```
  Today: `weight not found: model.layers.0.mlp.gate_proj.weight`.
  The Phi3 spec renames `ffn_up.weight → mlp.gate_up_proj.weight`
  (fused) and the Phi3 forward DSL expects fused gate_up_proj — but
  fingerprint dispatch is landing on a non-Phi3 variant whose loader
  asks for split `mlp.gate_proj`. Two suspects:
  (a) Phi-3.5-mini's LongRoPE config differs from any compiled
  Phi3 variant (Phi3 only has `phi3-mini-4k`-style entries today),
  so dispatch falls through to a Llama-shaped variant that wants
  split MLP. (b) The LongRoPE `rope_scaling_hash` doesn't match.
  Triage: print which variant `Weights::load` returns at runtime,
  and add a `phi-3.5-mini-128k.json` config with longrope scaling.
* **DeepSeek V2 / V3 GGUF — MoE loader is the only remaining blocker.**
  Steps 1 (quantizations.json on all three deepseek crates) and 2
  (yarn rope_scaling reconstruction in `gguf_model_config`) landed
  with the `gguf_archs` dispatch refactor (see Phases above).
  V2-Lite Q4_K_M and Moonlight-16B-A3B-Instruct Q4_K_M both
  download cleanly, both fingerprint-match the right forward arch
  (V2 catches V2-Lite; V3-flat catches Moonlight), both fail at the
  same point: `DeepSeekV2MoELayer::load` asks for
  `model.layers.{N}.mlp.experts.{e}.gate_proj.weight` per expert,
  which is the safetensors layout. GGUF ships fused 3D
  `mlp.experts.fused_{gate,up,down}_exps.weight`. The hand-written
  `vllm-cuda::deepseek_v2::load_gguf` (lines 1620–1707) is the
  reference: takes the three fused tensors, interleaves gate+up
  into a `[num_experts, 2*inter, hidden]` quantized w1, uses the
  fused down as w2, builds a `GgmlFusedMoELayer`. Need either
  (a) a separate `DeepSeekV2GgmlMoELayer` accessor type + Impl
  variant that the codegen picks when expert storage is `Ggml`, or
  (b) `DeepSeekV2MoELayer.moe` becomes an enum (`Bf16(FusedMoELayer)
  | Ggml(GgmlFusedMoELayer)`) and `load` branches on storage. Same
  for `shared_gate_up` / `shared_down` (currently dense `Linear`).
  (a) is more invasive (DP solver + Impl library + codegen) but
  cleaner; (b) is contained but couples bf16/quant in one type.
  Reproducer:
  ```
  ./vllm-rs/target/release/vllm chat \
    --model ~/.cache/huggingface/hub/models--mradermacher--DeepSeek-V2-Lite-GGUF/snapshots/0f37fdf276e8094747457f0ae4d40f2e8d2521f9/DeepSeek-V2-Lite.Q4_K_M.gguf \
    --max-tokens 5 --temperature 0 --prompt "Hi"
  # Errors: weight not found: model.layers.1.mlp.experts.0.gate_proj.weight
  ```

  Q-path coverage in the project (`v3-flat` is in this worktree's
  workspace post-rebase):

  | Q path        | V2 MoE (softmax, scale=1)         | V3 MoE (sigmoid + noaux_tc, scale=2.5)  |
  | ------------- | --------------------------------- | --------------------------------------- |
  | flat (no LoRA) | `ferrite-model-deepseek-v2`  ✓  | `ferrite-model-deepseek-v3-flat` ✓ (Moonlight, Kimi K2 family) |
  | LoRA (q_a/q_b) | (DeepSeek-V2 non-Lite — n/a)    | `ferrite-model-deepseek-v3`  ✓          |

  Test fixtures cached:
  - `mradermacher/DeepSeek-V2-Lite-GGUF` `Q4_K_M` (~10 GB) — V2 path
  - `mmnga/Moonlight-16B-A3B-Instruct-gguf` `Q4_K_M` (~10.5 GB) — V3-flat path
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
* **Integration tests.**
  - **Tokenization** — `gguf_tokenizer_matches_hf_reference` covers
    Llama-3.2-1B/3B, Qwen2.5-0.5B, Qwen3-0.6B, Granite-3.1-2B,
    Mistral-7B-v0.3, Phi-3.5-mini, Gemma-2-2B, Gemma-3-1B against
    HF reference ids.
  - **End-to-end inference** — `gguf_inference_smoke` shells out
    to the built `vllm` binary and asserts greedy "Paris" coherent
    on Llama-3.2-1B + Mistral-7B-v0.3 GGUFs. Locks down the load +
    dispatch + forward pipeline; needs a built binary + GPU. Run:
    ```
    cargo build --manifest-path vllm-rs/Cargo.toml -p vllm-cli \
        --features cuda --release
    cargo test --manifest-path vllm-rs/Cargo.toml -p ferrite-gguf \
        --release -- --ignored gguf_inference_smoke
    ```
    DeepSeek-V2-Lite + Moonlight fixtures intentionally omitted
    until the fused-3D MoE expert loader lands (handoff step 3) —
    adding them today would assert "Paris" against an error msg
    and flap.

## Where to look first

1. `crates/ferrite-gguf/src/spec.rs` — `GgufArchSpec` data shape
   and the `register!` macro. Includes `norm_weight_offset`.
2. `crates/ferrite-forward-macro/src/config.rs::load_gguf_spec` —
   the JSON parser the macro feeds from.
3. `crates/ferrite-model-gemma3/configs/quantizations.json` —
   reference for `norm_weight_offset` + `tensor_renames`.
4. `crates/ferrite-model-llama/configs/quantizations.json` —
   reference for `qk_permute` + `llama3_rope_scaling_inference`.
5. `crates/ferrite-gguf/src/spec.rs::apply_metadata` — the runtime
   side that consumes the declarative metadata reads.
6. `crates/ferrite-kernels/src/ggml.rs::load_gguf_into_weights` —
   GGUF load path; `norm_baked_offset` lookup + subtraction lives
   in the F32 → model-dtype norm-weight branch.

## Reproducing the smoke sweep

`/tmp/gguf_smoke.sh` is the script run at handoff: 9 cached fixtures
(Llama-3.2-1B/3B, Qwen2.5-0.5B, Qwen3-0.6B, Granite-3.1-2B,
Mistral-7B-v0.3, Gemma-2-2B, Gemma-3-1B + Phi-3.5-mini known-broken,
CommandR known-broken-on-L4). Each runs greedy "Paris" with a 90 s
(180 s for CommandR) timeout. Per-fixture log at
`/tmp/smoke_<name>.log`.
