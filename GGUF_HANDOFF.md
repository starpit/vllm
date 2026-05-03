# Ferrite GGUF integration — handoff

Branch: `worktree-ff-gguf`. Tip: see `git log -1`. Last big landings:
DeepSeek V2-Lite GGUF coherent end-to-end (`c105fea17`, fixes a
row-vs-column scale bug in `GgmlFusedMoELayer.forward`'s topk-weight
application) and the GGML MoE accessor + dispatch (`d1191c03a`).
Rebased onto `ff-interpreter` 2026-05-02.

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
Mistral-7B-Instruct-v0.3, Gemma-2-2B, Gemma-3-1B, **DeepSeek-V2-Lite**,
**Phi-3.5-mini**. The SentencePiece path (`tokenizer.ggml.model =
"llama"`) synthesizes BPE merges from vocab+scores via a direct port
of HF's `generate_merges` — see `build_sentencepiece_tokenizer`.

**Known broken / out of scope here:** CommandR-35B (decodes only PAD
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
* **Phi-3.5-mini GGUF — quantized packed-split.** Phi3's GGUF ships
  fused `attn_qkv.weight` and `ffn_up.weight` (the "ffn_up" name is
  llama.cpp's tag for the fused gate_up_proj — there's no separate
  `ffn_gate.weight`). The Phi3 spec renames these to the fused HF
  names (`self_attn.qkv_proj.weight`, `mlp.gate_up_proj.weight`) so
  that the manifest-driven `__packed_splits__` prelude can synthesize
  per-slice virtual entries before any field load. But the splitter
  (`GpuWeights::synthesize_packed_row_split_sizes`) only walked the
  dense `tensors` map; for GGUF the fused parents land in the
  `quantized` map (block-quantized `GgmlStorage`) and the call
  returned Ok(false), leaving the per-slice names absent → first
  field read failed with `weight not found:
  model.layers.0.mlp.gate_proj.weight`. Fix: extended
  `synthesize_packed_row_split_sizes` to recognize a quantized parent
  and carve N children out of the same GPU buffer at row-aligned byte
  offsets. Each child `GgmlStorage` is a view into the parent's
  allocation (no doubling, no extra H2D, no dequant); the parent
  allocation is leaked-by-design exactly as before (model weights
  live for the model's lifetime). Block-alignment is asserted —
  `ncols % dtype.block_size() == 0` is true for every quant we ship
  but a future format would error rather than silently produce torn
  blocks. Verified: Phi-3.5-mini Q4_K_M now greedy-coherent ("The
  capital of France is Paris."); Llama-3.2-1B / Mistral-7B /
  V2-Lite / Gemma-3-1B unchanged. `gguf_inference_smoke` integration
  test now covers Phi-3.5-mini and DeepSeek-V2-Lite alongside
  Llama-3.2-1B + Mistral-7B-v0.3 — 4 fixtures, all green serial.

## Open follow-ups

* **Audit other archs end-to-end.** With P2 landed, each arch's
  GGUF support is a small `quantizations.json` edit. Granite +
  Gemma2 + Gemma3 + CommandR + Phi3 specs landed; Phi-3.5-mini now
  inference-coherent (see "Phi-3.5-mini GGUF — quantized packed-split"
  in Phases landed below).
* **DeepSeek V2 / V3 GGUF — DONE.** V2-Lite Q4_K_M coherent end-to-end
  through the ferrite path. Two fixes landed: (1) the GGML MoE accessor
  + dispatch (see "Phases landed" → "DeepSeek V2 GGML MoE wiring") and
  (2) a row-vs-column-scale primitive bug in `GgmlFusedMoELayer.forward`:
  the topk-weight scaling used `kernels::broadcast_mul_inplace`, whose
  underlying CUDA kernel does `out[r,c] *= scale[c]` (column-wise) but
  was being called expecting `out[r,:] *= scale[r]` (row-wise). The
  kernel read `hidden=2048` scale elements when only `num_tokens*topk=48`
  existed → out-of-bounds read on topk_weights → garbage scales →
  ±Inf/NaN in `moe_out` → propagates into every downstream MoE layer's
  gate matmul → all-NaN logits → token 0 PAD output. Replaced with
  `kernels::fp8_post_scale_multiply` (genuinely per-row), adding a F32
  variant of the row-scale kernel since the existing one only had BF16/F16.
  Verified: V2-Lite reproducer below now outputs "Paris." Moonlight
  (V3-flat) GGUF loads but `vllm chat` fails on missing chat template
  in mradermacher's GGUF — separate issue (chat template extraction
  from GGUF metadata, out of scope here).

  Reproducer (now coherent):
  ```
  ./vllm-rs/target/release/vllm chat \
    --model ~/.cache/huggingface/hub/models--mradermacher--DeepSeek-V2-Lite-GGUF/snapshots/0f37fdf276e8094747457f0ae4d40f2e8d2521f9/DeepSeek-V2-Lite.Q4_K_M.gguf \
    --max-tokens 30 --temperature 0 --prompt "What is the capital of France?"
  # Outputs: " Paris.\n\nUser: ..."
  ```

* **(former blocker, now resolved) DeepSeek V2 / V3 GGUF — MoE loader DONE; forward gives garbage.**
  Loader (a)-shape landed: separate `DeepSeekV2GgmlMoELayer` accessor +
  `Instruction::DeepSeekMoeGgml` + `DeepSeekGgmlMoeImpl` claim on
  `is_ggml_moe(fuf, tile)` + `FieldLoad::DeepSeekV2GgmlMoe` codegen arm.
  `GgmlFusedMoELayer.gate` reverted back to `Linear` (the GGUF loader's
  dense path already converts the F32 router gate to model dtype, so a
  plain dense Linear is the right field type — see
  `ferrite-kernels::ggml::load_gguf_into_weights` lines 1635–1679).
  Loader interleaves `fused_{gate,up}_exps` byte slabs into a single
  quantized `w1 = [E, 2*inter, hidden]`, frees originals via
  `unrecord_alloc + driver::mem_free`, and re-`record_alloc`s the new
  buffer. Shared experts: `gate_proj + up_proj` byte-concat into a fused
  `GgmlLinear[2*shared_inter, hidden]`; `down_proj` lives as its own
  `GgmlLinear`. Gate-up dtype must match (interleaved into shared w1);
  `down` may use a higher-precision quant (typical Q4_K_M ships
  gate/up=Q4_K, down=Q8_0). V2-Lite Q4_K_M loads cleanly through this
  path now — no missing-weight errors.

  **But inference output is `!!!!!!!!!!`** (token 0 / PAD repeated) on
  V2-Lite Q4_K_M, both with and without `FERRITE_USE_REFERENCE=1` (so
  not a quantized-matmul kernel bug). Reproducer:
  ```
  ./vllm-rs/target/release/vllm chat \
    --model ~/.cache/huggingface/hub/models--mradermacher--DeepSeek-V2-Lite-GGUF/snapshots/0f37fdf276e8094747457f0ae4d40f2e8d2521f9/DeepSeek-V2-Lite.Q4_K_M.gguf \
    --max-tokens 30 --temperature 0 --prompt "What is the capital of France?"
  # Today: "!!!!!!!!!!"
  # Expected: "The capital of France is Paris."
  ```

  Per `feedback_v2lite_was_verified.md`, the underlying ferrite V2
  MLA / attention / norms / sampling were verified end-to-end through
  the BF16 safetensors path *before* the GGUF work — so the bug is in
  the GGUF-specific surface, not in the ferrite V2 forward.

  **What's been ruled out:**
  - YaRN params: V2-Lite GGUF (mradermacher's) ships only
    `rope.scaling.{type,factor,original_context_length}`, omitting
    `mscale`, `mscale_all_dim`, `beta_fast`, `beta_slow`. The
    `metadata_defaults` mechanism was extended to support dotted keys
    that splice into nested config objects (`rope_scaling.mscale`),
    and V2's `quantizations.json` declares the four V2-typical
    YaRN defaults. Confirmed via `FERRITE_GGUF_TRACE` that
    `rope_scaling` post-defaults has all six fields. **However**:
    ferrite bakes YaRN at *compile time* from the per-variant
    config (`ferrite-forward-macro/src/config.rs::extract_rope_scaling`,
    line 829), reading the merged JSON which is the BASE
    `deepseek-v2-lite.json` (already complete). So the runtime
    `metadata_defaults` is wasted for the ferrite path — only
    helps the vllm-cuda hand-written V2 path which is not on
    today's GGUF dispatch route. Defaults left in place as future
    defense.

  **Open suspects** (next session triage):
  (1) `DeepSeekV2GgmlMoELayer::load_gguf` byte interleave / concat
  arithmetic — most plausible. The hand-written reference is
  `vllm-cuda::deepseek_v2::load_gguf` lines 1620–1764; my mirror
  is in `ferrite-kernels::layers_moe::DeepSeekV2GgmlMoELayer`.
  Compare expert-slab byte offsets carefully.
  (2) Shared-expert gate+up concat order. Hand-written and mine
  both put gate first then up. Confirm `silu_and_mul_fused`
  expects [gate | up] — yes (gate in first half, up in second).
  (3) `down_exps` direct reuse as `w2`. Storage `nrows = E*hidden,
  ncols = inter` — is this what `indexed_moe_forward` expects for
  `w2` slicing per expert? Check `ggml.rs::ggml_moe_forward`
  + the kernel.
  (4) Layer-0 dense path: V2-Lite has `first_k_dense_replace=1`;
  layer 0 uses `mlp.gate_proj/up_proj/down_proj` via the standard
  ferrite GEMM Impls. Should be solid (other archs share this
  path) but worth verifying tensors load correctly.

  Triage path: dump hidden-state norms after embedding, after layer
  0, after layer 1's attention, after layer 1's MoE → compare
  against Python vLLM ground truth for the same prompt. No V2-Lite
  safetensors cached locally (~30 GB; only 2.8 GB free on /).
  Reproducer at top of bullet. Q-path coverage table below still
  holds; Moonlight V3-flat awaits the same fix.

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
    on Llama-3.2-1B, Mistral-7B-v0.3, DeepSeek-V2-Lite, and
    Phi-3.5-mini GGUFs. Locks down the load + dispatch + forward
    pipeline; needs a built binary + GPU. Moonlight (V3-flat) is
    held back pending GGUF chat-template extraction. Run:
    ```
    cargo build --manifest-path vllm-rs/Cargo.toml -p vllm-cli \
        --features cuda --release
    cargo test --manifest-path vllm-rs/Cargo.toml -p ferrite-gguf \
        --release -- --ignored gguf_inference_smoke
    ```
    Run with `--test-threads=1` if mixing with other GPU tests; the
    four fixtures together transiently use up to ~12 GB GPU.

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
