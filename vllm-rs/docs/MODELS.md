# Model architectures

Each model architecture ferrite-forward can compile lives in its own
crate: `crates/ferrite-model-<arch>/` (`ferrite-model-llama`,
`ferrite-model-qwen3`, `ferrite-model-gemma3`, etc.). The umbrella
`ferrite-models` crate composes them all under per-arch cargo
features (`arch-llama`, `arch-qwen3`, ...) with a default of
`all-arches`.

## Layout of a per-arch crate

```
crates/ferrite-model-<arch>/
  Cargo.toml
  src/lib.rs              # one #[forward(...)] body — the math
  configs/
    <size1>.json          # per-model hyperparameters (HF config.json verbatim)
    <size2>.json
    ...
    weights.json          # per-arch weight shape formulas (shared across sizes)
    quantizations.json    # optional — preset names this arch supports
    <size>-<preset>.overrides.json  # optional — per-(size, preset) drift
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

- **`quantizations.json`** (optional) — flat list of preset names
  this arch supports. Each preset cross-multiplies with every dense
  base in `configs/` to produce one compiled variant per
  `(size, preset)` pair. The preset definitions themselves live in
  the shared `ferrite-quantizations` crate (see below); this file
  just declares which ones apply to this arch.

  ```json
  { "quantizations": ["awq-gemm", "gptq-sym", "fp8-block-128x128"] }
  ```

- **`<size>-<preset>.overrides.json`** (optional) — per-HF-repo
  drift. Some HF quantization checkpoints diverge from their dense
  base in non-quantization fields (e.g. TinyLlama-GPTQ has
  `vocab_size=32003` vs. the dense base's `32000`). The overlay
  deep-merges last on top of the dense base + preset.

## Shared quantization presets

Preset definitions are JSON fragments that deep-merge onto each dense
base. They live once in `crates/ferrite-quantizations/presets/`:

```
crates/ferrite-quantizations/presets/
  awq-gemm.json
  bnb-nf4-dq.json
  ct-int4-sym.json
  fp8-block-128x128.json
  fp8-dynamic-per-tensor.json
  fp8-static-per-tensor.json
  gptq-sym.json
  gptq-sym-desc_act.json
```

Per-arch crates do not depend on `ferrite-quantizations` at the Rust
level — the `#[forward]` macro discovers `presets/` by walking up
from the per-arch crate's `configs/` to the workspace root. Adding a
new preset means dropping a JSON in `presets/` and listing it in the
relevant arch's `quantizations.json`.

## Recipe for a new model architecture

### 1. Create the per-arch crate

```
mkdir -p crates/ferrite-model-<arch>/{src,configs}
```

`Cargo.toml` mirrors any of the existing `ferrite-model-*` crates:

```toml
[package]
name = "ferrite-model-<arch>"
version.workspace = true
edition.workspace = true
license.workspace = true
description = "Ferrite <arch> architecture (one #[forward] body)"

[features]
default = []
cuda = ["ferrite-cuda-core/cuda", "ferrite-kernels/cuda", "ferrite-forward/cuda"]

[dependencies]
ferrite-cuda-core = { path = "../ferrite-cuda-core" }
ferrite-kernels = { path = "../ferrite-kernels" }
ferrite-forward = { path = "../ferrite-forward" }
anyhow = { workspace = true }
tracing = { workspace = true }
```

Add the crate to the workspace `members` in `vllm-rs/Cargo.toml`.

### 2. Wire it into the umbrella

Edit `crates/ferrite-models/Cargo.toml` to add the optional
dependency, the `arch-<name>` feature, and the `cuda` passthrough,
mirroring the existing arches. Add the crate to `all-arches`. Edit
`crates/ferrite-models/src/lib.rs` to add the corresponding
`#[cfg(feature = "arch-<name>")] extern crate ... as _keep_<name>;`
and `pub use ... as <name>;` lines.

### 3. Commit one config.json per supported size

Download the upstream HuggingFace `config.json` for each model size:

```
curl -L https://huggingface.co/<org>/<model-size>/resolve/main/config.json \
    > crates/ferrite-model-<arch>/configs/<size>.json
```

The filename is free-form but should match the HF model ID (e.g.
`qwen3-0.6b.json`). The compiler reads the `architectures` field of
each config to sanity-check it matches the DSL body's arch name.

### 4. Generate weights.json

Run the probe binary against any one model size — the shape formulas
don't change across sizes, so one probe suffices:

```
cargo run -p ferrite-forward --bin probe-weights --features probe -- \
    --arch <arch> <hf-org>/<size> [<hf-org>/<size> ...]
```

The probe walks up to the workspace root, resolves
`crates/ferrite-model-<arch>/configs/` (with `_` → `-` on the arch
name), fetches the safetensors header, rewrites numeric dims as
bound-name products from the config.json vocabulary, and writes
`weights.json` next to the size configs. The crate must already
exist (steps 1-2). Review the output before committing — literal
integers it couldn't rewrite as known bounds are flagged for
review.

### 5. Write the DSL body

Add `crates/ferrite-model-<arch>/src/lib.rs`:

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
a bound name from config.json. See `ferrite-model-llama/src/lib.rs`
as the canonical reference.

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

No edits to `vllm-executor/src/cuda_worker.rs` are needed — the
arch auto-registers via `inventory::submit!` from the umbrella.

## Dev iteration: `FERRITE_MODELS`

The `#[forward]` macro is the bulk of per-arch compile time. By
default it processes every config in `configs/` (e.g. all 96 llama
base + preset variants). Set `FERRITE_MODELS` to a comma-separated
list of stems to restrict it to those:

```
FERRITE_MODELS=llama-3.2-3b cargo build -p ferrite-model-llama --features cuda
# → ~2s instead of ~1m40s, only llama-3.2-3b's solver runs
```

Stems match base file names AND synthesized `<base>-<preset>`
variants:

```
FERRITE_MODELS=llama-3.2-3b,tinyllama-1.1b-gptq-sym cargo build ...
```

Unset → no filter, all models compile (the default for production
builds and CI). Each per-arch crate has a `build.rs` that emits
`cargo:rerun-if-env-changed=FERRITE_MODELS`, so cargo's incremental
cache invalidates on toggle — no manual `cargo clean` needed.

Pair with `--features` to also skip whole arches:

```
FERRITE_MODELS=llama-3.2-3b cargo build -p ferrite-models \
    --no-default-features --features cuda,arch-llama
```

This is a developer knob, not a deployment mechanism — for ad-hoc
"I'm only iterating on this one model right now." For deployments
that should ship a fixed subset, gate at the per-arch feature level
and ship the full `configs/` directory.

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
- **Don't** edit `vllm-executor/src/cuda_worker.rs` for a new arch.
  Auto-registration covers it; arch-specific state lives on the
  emitted `Weights` type.
