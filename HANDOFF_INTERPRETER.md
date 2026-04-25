# ff-interpreter — handoff

> Branch `ff-interpreter`. Read this whole file before changing
> anything. ff4 stencil rewrite is dead. The seam swap + a first
> round of codegen-size compression + downstream-build verification
> are in. **Goldens still hit a CUDA runtime failure on Llama 3.2
> 3B — see §5 (CURRENT BLOCKER)**, that's the next thing to chase.

## Where we are

Eight commits on top of the seam-swap checkpoint:

1. **`Vec<T>` Weights compression** (3313f8fdd) — layered accessors
   collapse from N per-layer fields + 40-arm `match layer` accessor
   methods + N per-layer load lets into one `Vec<T>` field, one
   `&self.<base>.get_unchecked(layer as usize)` accessor, one
   `(0..N).map(|layer| { format!(..., layer); …::load(…) }).collect()`
   Vec-build per layered base. `pub struct Weights`: 217→30 lines
   on commandr (-86%); `impl Weights { … }`: 578→240 (-58%);
   `pub fn load_with`: 1380→278 (-80%).
2. **Drop redundant `let _ = layer`** (58ba9b2cc) — the fn-level
   `#[allow(unused_variables)]` already suppresses warnings; the
   fn-top redundant `let layer = __layer;` setup is also gone. Per-
   arm `let layer: u32 = __layer;` shadow stays — load-bearing for
   `Op::Loop` iteration semantics.
3. **Per-bucket fn shim + manual Op Clone** (5275d036a) — per-bucket
   forward fns collapse to 1-line shims passing the bucket's static-
   slice idents + slot indices to a single `__forward_inner` /
   `__forward_backbone_inner` helper emitted once per arch module.
   `forward_backbone_m_<N>`: 30→11 lines per bucket. `Op` enum's
   `derive(Clone)` replaced with manual `*self` to skip the verbose
   `AssertParamIsClone` expansion.
4. **Shared arm-prelude lift** (2b35dca2a) —
   `extract_arch_wide_constants` records dropped `(fname, ty,
   value)` tuples in a sibling `extracted_prelude` side map on
   `ArchOpcodes`; `emit_interpreter` partitions by string-form key.
   Tuples present in 2+ variants lift to ONE fn-scope `let` at the
   top of `__dispatch_one`; per-variant residuals stay at the top
   of the arm. On commandr this lifts `cos_sin_fn`, `interleaved`,
   `biased`.
5. **Drop type annotations on extracted prelude lets** (8d96ea8d5)
   — Rust infers the type from the value, and arm-body call sites
   only need the binding's NAME. Collapses 4-line
   `for<'a> fn(&'a Weights, u32) -> &'a LinearLayer` types to one
   line per `weight_fn` / `cos_sin_fn` declaration.
6. **Trim verbose macro doc comments** (f84b116c8) — multi-paragraph
   `///` comments shortened to 1–2 line summaries (Op enum: 17→2,
   load_with: 8→2, fingerprint: 7→2, dispatcher: 8→1, etc.).
   Detailed rationale stays in macro source via `//` comments
   (prettyplease drops those from cargo expand).
7. **Three downstream-build bug fixes** (579d0644d) — surfaced when
   `cargo build -p ferrite-model-*` was run for the first time
   since the seam swap. All 11 model crates now compile clean. See
   §3 below for the bugs.

`cargo expand -p ferrite-model-commandr --lib --features cuda`:
**4244 → 2339 lines (-45% beyond the seam-swap baseline,
~ -90% from the original ~23k pre-pivot baseline).**
Llama: **263k → ~108k lines (-59%)**, with `pub fn load_with`
dropping from 118400 to 9842 lines (-92%) — the Vec compression's
single biggest win.

Macro test count: **177 pass / 23 fail** (same 23 pre-existing
failures from before the size pass). All 11 model crates build
clean: llama, qwen2, qwen3, gemma2, gemma3, mistral, phi3,
granite, commandr, deepseek-v2, deepseek-v3.

The user's stretch target is <1000 lines for commandr. Cumulative
progress is in good shape but not at goal. The two remaining big
levers (cross-canonical `__dispatch_one` dedup, per-arm `Ctx`
struct refactor) are wholesale refactors — see "What's left" §1+§2
below.

## Original (pre-compression) state

Last commit before the size pass: **seam swap + coloring + Op::Loop
checkpoint**. The interpreter pivot's *core mechanism* is done:

- All 49 Impls go through `opcode_shape` / `fan_out` /
  `interpreter_arm`. `emit_call` / `EmitMode` / `EmitCtx` /
  `FragmentLibrary` are deleted.
- Per-arch enum + per-arch `__interpret` driver, dispatching via a
  `__dispatch_one(op, __layer, …)` helper. `Op::Loop(count,
  body_len)` is honored by the driver — it re-runs the next
  body_len ops `count` times, threading the iteration index to the
  arms as `__layer`.
- Linear-scan register allocation (`colored_slot_map`) packs slot
  indices so layer L's intermediates reuse the same colors as
  layer L+1's. Combined with `extract_arch_wide_constants`
  (drops fields that are byte-identical across all instances of a
  variant) + `apply_loop_compression` (generic loop detection),
  layer bodies collapse from 40× rows to one Op::Loop + one
  iteration's body.
- `forward = backbone + lm_head`. ONE backbone slice per bucket,
  ONE lm_head row. forward / forward_backbone share the backbone.
- Aliases ride at the start of the slice as `Op::Alias(dst, src)`
  rows. No per-bucket-fn alias prelude.
- Free emission dropped (coloring makes it redundant — the next
  write to a reused slot drops the old OwnedTensor automatically).
- 19 invariant tests in `interpreter_codegen::tests`.
- Fp8 family migrated via new `Fp8AnyLinear { Std(Fp8Linear),
  Block(Fp8BlockLinear) }` wrapper in `ferrite-kernels::layers`.

**`cargo expand -p ferrite-model-commandr --lib --features cuda`
went from 22,883 → 4,196 lines.** Macro crate builds in ~5s.

## What's left

### 1. Cross-canonical `__dispatch_one` / `Op` / `Weights` dedup

Current breakdown for commandr (cargo expand, 2417 total lines):

| Section | Lines | Notes |
|---|---|---|
| `fn __dispatch_one` | 810 | per-canonical, ~25 arms × ~16 lines/arm |
| `(blank lines / use / misc)` | 322 | doc comments + module wrappers + m_<N> consts |
| `pub fn load_with` | 278 | bounds-baked literals — must stay per-canonical |
| `impl Weights { … }` | 240 | accessor methods (Vec-compressed) |
| `static BACKBONE_M_<N>` | 111 | static slice rows |
| `__forward_inner` helpers | 100 | per-arch shared (already lifted) |
| `pub enum Op` | 78 | per-arch enum |
| (others smaller) | 478 | |

Commandr ships TWO canonicals (c4ai_command_r_v01 + command_r_1_layer)
because `dedup_signature` includes bounds (40 layers vs 1 layer).
Both modules emit a near-byte-identical `__dispatch_one`, `Op` enum,
`__interpret`, accessor methods, and forward helpers — just with
different bounds-baked literals in `load_with` + the static slices.

To get under 1000 lines, the bounds-INDEPENDENT parts need to lift
above the per-model `pub mod` boundary. Concretely:

- `Weights` struct + accessor methods are TYPE-IDENTICAL across
  same-arch canonicals (layered fields are `Vec<T>` regardless of
  N). One Weights definition could be shared via `pub type Weights
  = super::__shared::Weights;` — same trick as the existing shim
  mechanism.
- `Op` enum + `__dispatch_one` + `__interpret` + `__forward_inner`
  helpers reference Weights but otherwise contain no bounds-baked
  literals (the kernel-call bodies use `ctx.<...>` runtime values,
  not baked literals — except for one outlier per Impl, see below).
- `load_with` + `fingerprint_matches` + per-bucket statics +
  per-bucket forward shims STAY per-canonical (they're where bounds
  bake into literals).

The wrinkle: a few Impls bake a constant from model bounds into
their `interpreter_arm` body. E.g., `AttentionViaCacheImpl` bakes
`scale = 1/sqrt(head_dim)` as a literal `0.088388346f32` (commandr,
head_dim=128). Since the literal differs between two canonicals
with different head_dim, their `__dispatch_one` bodies aren't
byte-identical and can't be shared.

Path to fix: convert these baked constants into `OpcodeShape`
fields. `extract_arch_wide_constants` then lifts them as fn-scope
lets in `__dispatch_one`. Two canonicals with different `head_dim`
get different fn-scope `let scale = 0.0884f32;` rows, but the
arm body is byte-identical. They can share `__dispatch_one`.

Refactor scope:
- Audit every `Impl::interpreter_arm` for baked literals derived
  from model bounds/scalars. Add corresponding `OpcodeShape`
  fields. (Memory rule `feedback_no_piecemeal_codegen_migration`:
  do all Impls in one wholesale commit.)
- Add a new shim mode: `WeightsEmitMode::SharedWithCanonical` that
  emits `pub type Weights = super::<canonical>::Weights;` AND
  `pub use super::<canonical>::{Op, __dispatch_one, __interpret,
  __forward_inner, __forward_backbone_inner};` while keeping the
  variant's own `load_with` + `fingerprint_matches` + per-bucket
  statics.
- New canonicalization tier in `compute_canonical_variants`:
  partition by (impls picked + tie + rope_scaling_kind) only —
  drop bounds + scalars from this signature. Variants in the same
  partition share the bounds-independent items.

Estimated savings: commandr ~600 lines (one __dispatch_one + Op +
helpers instead of two), llama ~25k lines (57 canonicals → some
smaller number, depending on how many distinct (impl_set,
tie, rope_kind) triples exist).

### 2. Per-arm body compression via a Ctx struct (user's idea)

Each arm body is ~10-15 lines:
```
let __out = unsafe {
    let __view = ::ferrite_forward::tile_ref(__tiles, in_slot).as_view(__tiles);
    let __w = (weight_fn)(wm, layer);
    ::ferrite_kernels::kernels::xyz(*__view, __w.weight, ..., device.compute_stream)
};
__tiles[out_slot as usize] = Some(::ferrite_forward::TileEntry::Owned(__out));
```

User suggested in this session: pack the per-call state into a
`Ctx` struct with helper methods (`ctx.view(slot)`,
`ctx.write_owned(slot, t)`). Each arm body shrinks to:
```
let __view = ctx.view(in_slot);
let __w = (weight_fn)(ctx.wm, layer);
ctx.write_owned(out_slot, ::ferrite_kernels::kernels::xyz(*__view, __w.weight, ..., ctx.stream()));
```

Roughly 3-4 lines per arm × 25 arms × N canonicals = 75-100 lines
saved per canonical. Combined with #1, this would bring commandr
under 1000.

Refactor scope: similar to #1 (touch every Impl's
`interpreter_arm`). Land wholesale, commit-by-commit forbidden.

### 3. Past-but-completed: downstream-build bug fixes

Three bugs surfaced when `cargo build -p ferrite-model-*` was run
end-to-end for the first time since the seam swap. All three were
pre-existing (from commit `692d24603`); none came from this
session's compression work. Fixed in commit `579d0644d`:

- **`ReshapeRefImpl` body**: `*ndim as usize` was dereferencing a
  u8 value (`ndim: u8`). Whether the field was destructured by-
  value from the variant or extracted as a `let ndim: u8 = 2u8;`
  prelude, `*ndim` was always invalid. Fixed to `ndim as usize`.
  Affected: gemma3, qwen3, deepseek-v2, deepseek-v3 (every arch
  that emits `Op::Reshape`).
- **`TensorView` lifetime**: `as_view<'a>(&'a self, tiles: &'a
  [TileEntry]) -> TensorView<'a>` tied the returned view's
  PhantomData to `tiles`'s borrow. `TensorView` is a `Copy`
  wrapper around a raw GPU pointer — the lifetime is purely a
  marker, not a real borrow. But the borrow checker treated
  `__tiles` as immutably borrowed for the view's full scope, so
  arms that produced `Reshaped` slots from view metadata
  (`RopeAppendRefImpl`, `FusedQkvQkNormRopeCacheImpl`, …) hit
  E0502 the moment they tried to write back into `__tiles`.
  Relaxed to return `TensorView<'static>`; safety story unchanged
  (`as_view` is already `unsafe`, the drop-pass invariant keeps
  the upstream `OwnedTensor` alive).
- **`AccessorGroup` contiguity check**: my `group_accessors_by_base`
  panicked on layered families that didn't fill `0..N`
  contiguously, e.g. DeepSeek-V2's `moe` accessor (layers 1..N —
  layer 0 is dense FFN) and `mlp_down_proj` (layer 0 only —
  layers 1..N use `moe`). Replaced the `layered: bool` flag with
  an `AccessorGroupKind` enum:
  - `Unindexed` — single field, no layer arg.
  - `LayeredContiguous` — `Vec<T>`, slice-index accessor (only
    fires for layered groups starting at layer 0 with no gaps).
  - `LayeredSparse` — per-layer fields + match-arm accessor
    (legacy fallback for non-zero-start or has-gaps families).
  Also dropped the `entries.len() == num_hidden_layers`
  assertion — a contiguous-from-zero group can be a partial
  cover (e.g. `mlp_down_proj` is layer 0 only).

### 4. End-to-end build verification

**Done** (commit `579d0644d`). `cargo build -p ferrite-model-*
--features cuda` passes for all 11 arches: llama, qwen2, qwen3,
gemma2, gemma3, mistral, phi3, granite, commandr, deepseek-v2,
deepseek-v3.

### 5. CURRENT BLOCKER — Llama 3.2 3B runtime failure

`vllm chat unsloth/Llama3.2-3B-Instruct --prompt "why is the sky
blue" --enforce-eager` fails. Symptom:

```
Error: engine step failed: executor error: worker execution failed:
async D2H token ids: CUDA driver error: CUDA_ERROR_ILLEGAL_ADDRESS
```

With `CUDA_LAUNCH_BLOCKING=1` the actual panic surfaces:

```
panicked at crates/ferrite-cuda-core/src/cublas.rs:532:13:
cublasGemmEx failed for GEMM [M=41, K=8192, N=5120] BF16 status=13 trans=true
   1: …CublasHandle::gemm_ex
   2: …CublasHandle::run_matmul_with_fallback
   3: …CublasHandle::run_gemm
   4: …LinearLayer::forward
   5: ferrite_model_llama::llama_3_2_3b::__dispatch_one
   6: …forward_inner → forward → FerriteWeights::forward
   7: vllm_executor::cuda_worker::CudaModel::forward
```

Status 13 = `CUBLAS_STATUS_EXECUTION_FAILED`. M=41 is the
chat-templated prompt length. The shape `[41, 8192] @
[8192, 5120]ᵀ` doesn't match Llama 3.2 3B's documented dims
(hidden=3072, intermediate=8192, qkv=q+k+v=24·128+8·128+8·128=5120) —
so K=8192 is unexpected unless this `LinearLayer::forward` call
is the QKV proj loaded with the wrong size. Possible causes:

- `Vec<T>` weights compression loaded the wrong layer's weights —
  the `(0..N).map(|layer| { format!("model.layers.{}.…", layer);
  …::load(…) }).collect()` body is new in this branch. If the
  format-template + `safetensors_prefix` agreement is off-by-one,
  layer L's slot would hold layer L+1's weights or vice-versa.
- `as_view`'s relaxed lifetime (now `'static`) lets a stale view
  survive an `__tiles[X]` overwrite somewhere. Color reuse +
  `set_owned` in the same arm sequence might expose this where
  the tighter lifetime would have caught it.
- Op::Loop's `__layer` shadow is wrong. If a row in the loop body
  destructures a `layer` field that should be `__layer`-driven
  but isn't — e.g. an `Impl::interpreter_arm` body that
  references `layer` AFTER the `let layer: u32 = __layer;`
  shadow but ALSO has its own destructure-shadow inside an inner
  scope — the iteration counter would be ignored.
- Coloring puts two live tensors on the same slot. If a kernel
  reads its input AFTER the runtime starts writing its output,
  the new write would clobber the read. Most kernels read
  fully-then-write but some may not.

Reproduce / triage steps:
1. `cargo expand -p ferrite-model-llama --lib --features cuda
   > /tmp/llama.rs` and find the `llama_3_2_3b` module's
   `__dispatch_one`. Inspect the `Op::Gemm` / `Op::CutlassGemv`
   arm bodies for `K=8192, N=5120` — that should NOT be a Llama
   3.2 3B GEMM. If it appears, the Vec accessor lookup is wrong.
2. Diff `Weights::load_with` for `llama_3_2_3b` between this
   branch and `ferrite-forward` (the pre-pivot branch) — focus
   on the `(0..N).map(|layer| …)` Vec builds. The format-string
   templates are computed by `layer_templated_prefix_expr` (in
   `crates/ferrite-forward-macro/src/codegen.rs`); ensure
   `model.layers.{}.<joined>` substitutes the closure's
   `layer: u32` and not some captured loop variable.
3. Add `print-weight-layout` instrumentation: at the top of
   `LinearLayer::forward`, log `(in.shape, weight.shape)` to a
   trace stream. Run `vllm chat` and compare against the
   hand-written `vllm-cuda` reference's log for the same prompt
   step.
4. Bisect: revert this branch's commits one at a time and re-run
   `vllm chat`. The Vec compression commit (`3313f8fdd`) is the
   most likely culprit; the per-bucket helper commit
   (`5275d036a`) and the `as_view` relaxation (`579d0644d`)
   are the next-likeliest.

The interpreter design changes runtime behavior in subtle ways
and any of these is a plausible source:

- Slot reuse via coloring. If any kernel reads its input AFTER
  starting to write its output (atypical), color sharing could
  expose a UAF. Currently every Impl's interpreter_arm reads the
  view before calling the kernel; the kernel queues its work then
  the OwnedTensor is dropped. Should be safe but test exercises
  this empirically.
- Op::Loop's iteration counter passed via `__layer` rather than
  per-row literal. If any arm body reads the layer index from a
  destructured field after the let-rebind shadows it, the layer
  used for kv_cache index / weight lookup would be wrong. The
  shadowing pattern is in `emit_interpreter` (post-destructure
  `let layer: u32 = __layer;`). Verify by reading the expanded arm
  bodies — none should reference the destructured `layer` field
  before the let.
- Aliases as `Op::Alias` rows execute IN slice order. Check that
  the alias prelude in the expanded output runs before any Op
  that reads the aliased slot. (Should be fine — aliases are
  prepended in `lower_bucket`.)
- Vec compression (new on this branch) — `Weights::input_layernorm`
  is now `Vec<RmsNorm>` indexed by `layer as usize`. The
  `(0..N).map` builder runs at load time; off-by-one in the
  format-string template would mis-load layer weights. Verify
  `safetensors_prefix(program, id, Some(L))` for L=0..N-1
  produces the same on-disk paths the runtime reads.

### 6. Goldens

After §5 is fixed, run:

```
cargo test --release --test e_correctness -p vllm-e2e \
    --features e2e,cuda -- --ignored --test-threads=1
```

Llama subset first; match must be exact. Then Qwen2 / Qwen3 /
Mistral / Phi3 / Gemma2 / Gemma3 / Granite / CommandR / DeepSeek-V2
/ DeepSeek-V3.

### 7. Open warnings / cleanup

- `_protected: &HashSet<(TileId, u8)>` unused arg in `lower_bucket`
  signature. Either remove (callers update) or keep as
  documentation. Decide.
- The pre-existing `layers_moe.rs:1434` error
  (`FusedMoELayer { … }` missing fields). Not from our changes;
  should be resolved separately on `main` or by another contributor.
- Three pre-existing macro-crate test failures
  (`load_real_llama_configs`, `load_real_qwen2_configs`,
  `add_rmsnorm_pairs_claimed_as_fused_subgraph`) unchanged. Not in
  scope.

### 8. Optional follow-ups

- **Arena-backed `__tiles`.** With coloring, slot count is small
  (~10–20 for commandr) and bounded per arch. Replace `Vec<Option<
  TileEntry>>` with `[Option<TileEntry>; N_SLOTS_<ARCH>]`
  stack-allocated per per-bucket fn. Eliminates heap allocation
  per forward call.
- **Skip arch-wide-extraction's hash-string comparison.** Currently
  `to_string()` per field-value to detect identical literal tokens.
  Could pre-hash with `std::collections::hash_map::DefaultHasher`
  once per OpInstance and compare hashes for the constant-extraction
  pass too. (Already done for `detect_repeating_run`.)
- **`__dispatch_one` static slimming.** ~430 lines for commandr's
  ~25 variants × ~15 lines per arm. Each arm body has a let-rebind
  for arch-wide constants extracted from rows. Could combine
  rebinds when multiple variants share an extracted value
  (e.g., `cos_sin_fn` baked into both QkvRopeCache and
  QkvRopePrefill). Marginal gain; not worth doing before #1.

## Reference points

- `crates/ferrite-forward-macro/src/interpreter_codegen.rs`:
  - `colored_slot_map` — linear-scan register allocator.
  - `extract_arch_wide_constants` — drops fields with constant
    values across instances.
  - `detect_repeating_run` + `apply_loop_compression` — generic
    loop detection + Op::Loop emission. Caller passes the
    iter-index field name (`"layer"` for transformers).
  - `emit_enum` + `emit_interpreter` — per-arch enum + driver.
  - The arm-body wrapper `let layer: u32 = __layer; … #body` is the
    bridge from Op::Loop's iteration counter to arm bodies.
- `crates/ferrite-forward-macro/src/codegen.rs::emit_model`:
  - The seam. Lowers each canonical bucket once
    (skip terminal subgraph), computes terminal row separately,
    runs extraction + loop-compression passes, emits BACKBONE +
    LM_HEAD slices, emits per-bucket fns.
  - `emit_weights_struct` + `emit_weights_accessor_methods` — the
    targets of the next refactor (#1 above).
- `crates/ferrite-forward/src/tile_table.rs`:
  - `TileEntry::{Owned, View, Reshaped}`, `tile_ref`,
    `take_owned`, `view`. The interpreter's runtime data model.
- `crates/ferrite-kernels/src/layers.rs::Fp8AnyLinear`:
  - The unblocker for the Fp8 migration. Wrapper enum dispatching
    between `Fp8Linear` and `Fp8BlockLinear` at runtime.

## Pre-commit checklist (when you next commit)

Re-read this file. Verify:

- [ ] `cargo build -p ferrite-forward-macro` clean.
- [ ] `cargo test -p ferrite-forward-macro --lib` passes
      (3 pre-existing failures don't count, see above).
- [ ] `cargo build -p ferrite-models --features cuda` clean.
- [ ] llama golden subset green; sibling-arch goldens green.
- [ ] `cargo fmt` clean on touched crates.
- [ ] No universal opcode enum / registry / string lookup. (Per-arch
      `Op` is the only enum; `Alias` / `Free` / `Loop` are
      universal codegen-issued, not Impl-issued.)
- [ ] No `_` arm or `unsafe { unreachable_unchecked() }` in any
      emitted match arm. (`Op::Loop(_,_) => unreachable_unchecked()`
      inside `__dispatch_one` is intentional — Loop never reaches
      dispatch, only the driver.)
- [ ] No megakernel wire-format types in ferrite-forward.
- [ ] No new tests deleted; new code lands with new
      invariant tests where applicable.

If any box is unchecked, **stop and wait** — do not commit.
