# Model architectures

Each subdirectory here is one model architecture that ferrite-forward
can compile — `llama/`, `qwen2/`, `gemma2/`, `granite/`, etc. The
compiler reads these files at macro-expansion time (via
`#[forward(models_dir, target, workloads)]` in
`vllm-rs/crates/ferrite-models/src/<arch>.rs`).

## Files per architecture

```
<arch>/
  <size1>.json       per-model hyperparameters (HF config.json verbatim)
  <size2>.json
  ...
  weights.json       per-arch weight shape formulas (shared across sizes)
```

- **`<size>.json`** — the upstream HuggingFace `config.json` for one
  model (e.g. Qwen3-0.6B). Contains hyperparameters the compiler uses
  as shape bounds: `hidden_size`, `num_attention_heads`, `head_dim`,
  `intermediate_size`, `num_hidden_layers`, `vocab_size`, etc. Commit
  verbatim — don't edit by hand. Different model sizes under the same
  architecture get one file each.

- **`weights.json`** — shape of every weight that the compiler's
  dataflow-driven shape inference cannot pin (or would pin
  incorrectly). Expressed as products of bound names from the
  config.json vocabulary. Example for Qwen3, where `q_norm` / `k_norm`
  are per-head RMS norms of shape `[head_dim]`:

  ```json
  {
    "self_attn.q_norm": ["head_dim"],
    "self_attn.k_norm": ["head_dim"]
  }
  ```

  Only weights whose shape can't be derived from op-sig dataflow +
  `<size>.json` bounds belong here. For most Llama / Qwen2 / Granite
  entries, dataflow pins everything and `weights.json` is empty or
  absent.

## Recipe for a new model architecture

### 1. Create the subdirectory

```
mkdir model_architectures/<arch>
```

### 2. Commit one config.json per supported size

Download the upstream HuggingFace `config.json` for each model size
the arch supports and commit it as `<size>.json`:

```
curl -L https://huggingface.co/<org>/<model-size>/resolve/main/config.json \
    > model_architectures/<arch>/<size>.json
```

The filename is free-form but should match the HF model ID (e.g.
`qwen3-0.6b.json`). The compiler reads the `architectures` field of
each config to sanity-check it matches the DSL body's arch name.

### 3. Generate weights.json

Run the probe script against any one model size under this arch — the
shape formulas don't change across sizes, so one probe suffices:

```
scripts/probe-weights <arch>/<size>
```

The script:
- Fetches the safetensors header from HF Hub (no full weight download).
- Reads the tensor names and shapes from the header's JSON blob.
- Cross-references numeric dims against `<size>.json` to rewrite them
  as bound-name products (`128` → `"head_dim"` when
  `config.head_dim == 128`; `5504` → `"intermediate_size"`; etc.).
- Drops weights whose shape dataflow already pins correctly (the
  compiler will infer those).
- Writes `model_architectures/<arch>/weights.json`.

Review the output before committing — if the script couldn't rewrite
a dim as a known bound (e.g. because of a non-HF convention in the
config), it leaves the literal integer and flags it for review.

### 4. Write the DSL body

Add `vllm-rs/crates/ferrite-models/src/<arch>.rs` with:

```rust
use ferrite_forward::forward;

#[forward(
    target = "../../../target_profiles/l4_sm89.json",
    workloads = [1, 8, 64, 512, 4096],
)]
fn <arch>() {
    // math of the forward pass, in bound-name shapes
}
```

The DSL body expresses the forward pass as pure math — `embed`,
`rmsnorm`, `gemm`, `rope_append`, `attention`, `silu`, `add`, `mul`,
etc. Every tensor is a local; weights are referenced by dotted path
(e.g. `self_attn.q_proj[layer]`). Loop over `num_hidden_layers` using
a bound name from config.json. See `ferrite-models/src/llama.rs` as
the canonical reference.

### 5. Wire the Weights struct into `cuda_worker.rs`

In `vllm-rs/crates/vllm-executor/src/cuda_worker.rs`, add a
`CudaModel::<Arch>Ferrite` variant and accessor/forward match arms
paralleling `LlamaFerrite`. The `Weights::load` method the macro
generates will consume safetensors and populate the WeightBundle
accessors declared by the solver's chosen Impls.

### 6. Commit a correctness golden

Add `testdata/<arch>_<size>.json` generated from Python vLLM on the
same prompts the existing goldens use. Add a
`test_cuda_correctness_<arch>_<size>` in
`crates/vllm-e2e/tests/e_correctness.rs`. Verify the test passes on
CUDA before claiming the arch is landed:

```
cargo test -p vllm-e2e --features e2e,cuda --release \
    --test e_correctness test_cuda_correctness_<arch>_<size> \
    -- --ignored --test-threads=1
```

## What NOT to do

- **Don't** hand-edit `<size>.json` — keep it verbatim with upstream.
- **Don't** add entries to `weights.json` for weights dataflow already
  pins correctly. The file is for exceptions only (per-head norms,
  latent projections, etc.) — every entry is a claim that dataflow
  can't derive this one.
- **Don't** put cross-arch weight shapes in
  `ferrite-forward-macro/src/weight_conventions.rs`. That table is
  reserved for truly universal conventions (e.g. `embed_tokens`,
  `lm_head`) — everything arch-specific lives in the arch's
  `weights.json`.
