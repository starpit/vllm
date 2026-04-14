# ferrite-forward handoff

Paired with `PLAN.md`. PLAN describes what to build; this describes what's
built, what decisions were made along the way, and the landmines a fresh
context needs to know before writing more code.

Read PLAN first. Then read this.

## Status — what's done vs. pending

Phases committed on branch `worktree-ferrite-forward` off `2492e3c71`.

| Phase | Done | Commit subject |
|------:|:----:|:---------------|
| 0 | ✅ | scaffolding + model architecture configs |
| 1 | ✅ | parse DSL body into AST |
| 2 | ✅ | classify free variables |
| 3 | ✅ | read model configs from directory |
| 4 | ✅ | shape inference |
| 5 | ✅ | build CFG from classified program |
| 6 | ✅ | unroll CFG into numeric FUF |
| 7 | ✅ | solver — pick impl per tile (MVP) |
| 7+ | ✅ | anchor MLP weights via HF convention table |
| 8 | ✅ | BSP schedule via topological layering |
| 9 | ⬜ | codegen Rust (specialized forward fns) |
| 10 | ⬜ | codegen CUDA (.cu + manifest in $OUT_DIR) |
| 11 | ⬜ | real Llama inference via vllm chat |
| 12 | ⬜ | Gemma2 acid test |

Test count at handoff: **58 unit tests + 3 integration tests passing**.
Zero clippy warnings on `-p ferrite-forward -p ferrite-forward-macro`.

## How to verify current state

```bash
cd /home/moosevan/vllm/.claude/worktrees/ferrite-forward/vllm-rs
cargo test -p ferrite-forward -p ferrite-forward-macro
cargo fmt   -p ferrite-forward -p ferrite-forward-macro --check
cargo clippy -p ferrite-forward -p ferrite-forward-macro --lib --tests -- -D warnings
```

All three should exit 0. If they don't, something drifted — investigate
before adding anything.

## Repository layout

```
ferrite-forward/                                 (worktree root)
├── PLAN.md                                      the plan
├── HANDOFF.md                                   this file
├── model_architectures/                         arch-level per-model data
│   ├── llama/                                     9 Llama configs
│   └── qwen2/                                     11 Qwen2/2.5 configs
├── target_profiles/                             hardware metadata
│   ├── l4_sm89.json
│   └── h100_sm90.json
└── vllm-rs/crates/
    ├── ferrite-forward/                         consumer crate (re-exports macro)
    │   ├── src/lib.rs
    │   └── tests/                               end-to-end integration tests
    └── ferrite-forward-macro/                   proc-macro + all compiler logic
        └── src/
            ├── lib.rs                           #[proc_macro_attribute] fn forward
            ├── ast.rs                           raw parse tree
            ├── parse.rs                         syn → Ast
            ├── classified.rs                    classified IR + tables
            ├── classify.rs                      Ast → classified::Program
            ├── config.rs                        model config.json loader
            ├── weight_conventions.rs            HF weight-name → shape table
            ├── shape.rs                         Dim, Shape, Solver, infer
            ├── cfg.rs                           classified::Program → Cfg
            ├── fuf.rs                           Cfg → Fuf (numeric tile graph)
            ├── target.rs                        TargetProfile loader
            ├── impl_lib.rs                      Implementation + starter_library
            ├── solver.rs                        Fuf + lib → Assignment
            └── schedule.rs                      Assignment → BSP Schedule
```

## Key types — quick reference

Every type here lives in `ferrite-forward-macro/src/`. Knowing the map
saves a lot of grep.

- `ast::{Ast, Stmt, Expr, BoundExpr}` — raw parse tree, syn::Ident preserved verbatim.
- `classified::{Program, Stmt, Expr, Bound, LocalId, WeightId, ExternKind, OpKind}` —
  free-var categories classified, names interned, IDs numeric. `LocalTable` and
  `WeightTable` are the side maps from numeric id → debug ident / path.
- `shape::{Dim, Shape, Solver, DimVar, Inferred, ShapeError}` — shapes over symbolic
  bounds with union-find unification. `Inferred` is the output: `locals: HashMap<LocalId, Shape>` and `weights: HashMap<WeightId, Shape>`.
- `config::{ModelParams, load_dir, load_file}` — `ModelParams { name, source_stem, source_path, bounds: BTreeMap<String,u64> }`.
- `target::{TargetProfile, load_dir, load_file}` — hardware metadata.
- `cfg::{Cfg, Block, Instr, Terminator, BlockId}` — per-model CFG with concrete u64 trip counts.
- `fuf::{Fuf, FufNode, FufInput, TileId}` — flat numeric tile graph. `FufInput = Tile{id, slot} | Weight{id, index} | Extern{kind, index}`.
- `impl_lib::{Implementation, ImplementationLibrary, ImplId, CostCtx, starter_library}`.
- `solver::{solve, Assignment, SolveError}` — `Assignment { tile_to_impl, predicted_us }`.
- `schedule::{schedule, Schedule, Step, find_intra_step_dep_violation}`.

## Decisions made along the way

These aren't in the original PLAN but shouldn't be rediscovered.

### DSL syntax

- **Attribute macro** `#[forward] fn llama() { body }`, not function-like
  `forward! { body }`. Chosen so rustfmt formats the body like any Rust fn.
- **No `let`.** Every binding is `name = expr;` or `(a, b, c) = expr;`.
  The parser rejects `let` with a pointed error. Rationale: SSA under the hood,
  `let` was ceremony without semantics.
- **Only `*` is admitted as a binary op** (used by `gate * up`). Everything else
  is a parse error.
- **Loop bounds** are either integer literals or bare identifiers (no arithmetic
  in the range position). Bound resolution happens at CFG-build time via
  per-model `ModelParams.bounds`.

### Models and targets are directory-based, not enumerated

- `#[forward]` on an arch fn implicitly reads `model_architectures/<arch-name>/*.json`
  to discover models. File stem = generated identifier (normalized to valid Rust
  ident — dashes/dots → underscores; leading digit → `m_` prefix).
- Target profiles live in a parallel `target_profiles/` directory at the repo
  root. Same discovery pattern.
- **Not yet wired into the macro**: the macro currently parses + classifies the
  body and emits an empty fn. Phases 9+ plug the real compilation in, at which
  point the macro will need to resolve these directory paths relative to
  `CARGO_MANIFEST_DIR`.

### Shape inference

- Dim vocabulary: `Lit(u64) | Bound(String) | Mul(Vec<Dim>) | Var(DimVar)`.
  `Bound` names come from the **HuggingFace config.json vocabulary**
  (`hidden_size`, `num_attention_heads`, `head_dim`, …). Using these names in op
  signatures is transformer-math, not arch-specific — it's the same vocabulary
  across every HF decoder-only LLM.
- **Unresolved Vars are preserved, not errored on.** `Solver::close_dim`
  returns `Ok(Dim::Var(v))` if v has no binding. Callers decide what to do —
  cost functions return `None`; `solver.rs` treats that as infinite cost and
  picks some other impl; codegen will concretize via weight manifest or equivalent.
- **Op signatures anchor dims via transformer-math conventions**:
  - embed: output's last dim = `hidden_size`; table is `[vocab_size, hidden_size]`
  - rope_append: q.last = `num_attention_heads * head_dim`; k,v.last = `num_key_value_heads * head_dim`
  - attention: same as rope_append
  - rmsnorm: shape-preserving, weight = `[input.last]`
  - gemm: `[.., K] × [K, N] → [.., N]`, no naming of N (let downstream pin it)
  - add: shape unify operand-by-operand
  - silu: shape-preserving
- **Weight shapes that dataflow doesn't pin** (notably MLP weights — `intermediate_size`
  isn't a transformer-math anchor, it's a naming convention) get anchored by
  `weight_conventions::standard_shape()`. This table covers every weight in a
  standard HF transformer. Arches with non-standard weights (MoE routers, vision
  patch embeddings) would need a per-arch `weights.json` override — not implemented
  yet because none of Llama/Qwen2/Gemma2 need it.

### FUF structure

- One `FufNode` per DSL op. No `GemmQ/K/V` variants — all gemms have
  `OpKind::Gemm`; their role is defined by their inputs and outputs, not by a
  kind variant.
- `rope_append` has **three output slots**. The three target LocalIds all map
  to the same TileId with different slots (0/1/2). Consumers read via
  `FufInput::Tile { id, slot }`.
- **Loop-carried locals** are threaded via `Stmt::For.loop_carry: Vec<(LocalId, LocalId)>`
  (classifier-emitted, CFG-carrying, unroller-applied). Fixes the bug where
  iteration N+1's body reads would return iteration 0's tile. See classifier's
  scope-diff logic; PLAN doesn't mention this, it surfaced during Phase 6.

### Solver is MVP

- Phase 7 picks the cheapest candidate per tile independently. With one impl per
  op, that's trivially correct. When the library grows multi-impls-per-op with
  real `claim_mask` contention, upgrade to DP over (position, claim_state).
  Writing DP now would be theatre.
- Cost functions return `None` when shapes have unresolved Vars. Solver treats
  that as infinite cost. For the standard body, the convention table closes
  everything and costs are real.
- Solve time at NL=16 is well under the 100ms budget (PLAN's requirement).

### Schedule is topological layering

- Not "step-merge from a 1-per-step starting point" as PLAN described — the
  output is equivalent either way, but topological layering is simpler and
  matches the natural DAG semantics. q/k/v gemms land in one step because they
  all read `normed` and don't depend on each other.
- **Further merging** (packing across layers using claim_masks) is left for when
  the library grows impls that actually interact. For the starter library with
  `claim_mask=0` everywhere, layering is optimal.

## Known punt points for phases 9–12

These are things I intentionally deferred. Phase 9+ must handle them.

1. **`Mul` expression (`gate * up`) is currently emitted as `OpKind::Add` tile.**
   `fuf.rs` has a `TODO(phase 9)` comment at the relevant call site. The DSL has
   `*` for elementwise multiply but `OpKind` doesn't yet have a `Mul` variant —
   fix by adding `OpKind::Mul` + a shape signature + (for Phase 10) a CUDA
   kernel + (for Phase 4) an entry in the ops list. Until fixed, codegen would
   emit an `add` where a `mul` should be. **Do this before Phase 10.**
2. **Runtime dims stay symbolic.** `num_tokens` (batch × sequence) is a `Dim::Bound`
   that never resolves at compile time. Cost functions accept a `num_tokens` in
   the bounds map (tests pass `num_tokens: 1` for decode). Codegen has to emit
   code that reads the actual runtime value from the activation tensor — same
   as any existing forward.
3. **Attribute args unimplemented.** `#[forward]` currently ignores its args.
   Phase 9 should parse `#[forward(targets = ["..."])]` and `#[forward(dir = "...")]`.
4. **Tracked paths unregistered.** The config and target loaders use
   `std::fs::read` but don't call `proc_macro::tracked_path::path`. When the
   loader is wired into the macro (Phase 9), add the tracking so cargo
   rebuilds on config edits.
5. **`weights.json` deferred.** Not needed until we meet an arch the HF
   convention table doesn't cover. If Gemma2's new ops or weights require it,
   add this before Phase 12.

## Remaining phases — what they need to read

Before writing any code for 9–12, the next context should do these reads
(do not skip — writing blind is the exact failure mode PLAN.md § 5 is about):

### Phase 9 — Rust codegen

Read:
- `vllm-rs/crates/ferrite-models/src/` — understand the current signature
  shape (probably `pub fn forward(model: &Model, input_ids: &Tensor, ...) -> Tensor`).
  Figure out the exact arg list + return type the generated code has to match.
- `vllm-rs/crates/ferrite-models/src/llama.rs` or equivalent — see how the existing
  `forward!` macro emits code. Our output needs to plug into the same call sites.
- `vllm-rs/crates/ferrite-cuda-core/` or wherever the Tensor / FFI types live.
  Generated code will call into extern "C" functions.

Output:
- For each `model` in `ModelParams`: a `pub fn <model_name>_forward(...)` that
  walks the schedule, for each step invokes the chosen Implementation's runtime
  entry point. Weight refs become `model.layers[i].<path>`; extern params
  become typed fn args; local bindings become Rust `let` bindings (the one
  place `let` is appropriate — it's generated Rust, not DSL).

### Phase 10 — CUDA codegen

Read:
- `vllm-rs/crates/ferrite-cuda-builder/` — what manifest shape does its build.rs
  consume? Where does it expect `.cu` files? What linker directives?
- `vllm-rs/crates/ferrite-kernels/` — existing .cu sources, to see what functions
  already exist and their signatures. Reuse existing kernels; don't rewrite them.

Output:
- For each `(tile, impl)` that needs fresh CUDA: emit the .cu to $OUT_DIR and
  add an entry to a manifest JSON. ferrite-cuda-builder (or whatever the current
  equivalent is) picks it up.
- For tiles whose chosen Implementation references an already-existing kernel,
  emit only the extern "C" Rust FFI binding.

### Phase 11 — Llama inference

Read:
- `vllm-rs/crates/vllm-cli/` — find the `vllm chat` entrypoint.
- How does vllm-cli get a forward fn today? Replace that path for Llama.
- Per `feedback_no_run_chat` memory: `timeout ... vllm chat ...`. Exits cleanly
  on coherent output; loops on garbage.

Acid test: `timeout 60 vllm chat --model=<path-to-a-small-llama>` produces
coherent text and exits. Latency parity or better vs the legacy path.

### Phase 12 — Gemma2

Gemma2-specific differences (do not encode these in the compiler — they all
live in the DSL body + new ops + new configs):

- Alternating **sliding window** vs global attention per layer. Needs a
  `sliding_attention` op (or parameterize `attention`), plus an arch-level
  signal per layer. Likely a layer-index-keyed branch in the body.
- **Pre-AND-post attention norms**, plus pre-AND-post FFN norms. Structurally
  different body shape.
- **Approximate GELU** instead of SiLU. Add `OpKind::Gelu` + a shape signature
  (elementwise, shape-preserving) + a kernel.
- **Query scaling** (fixed scalar multiplier on q before attention). Either a
  new op or a parameter on `attention`.
- **Logit soft-capping** (tanh-based clamp on the final logits). `OpKind::SoftCap`
  + shape sig + kernel.

Acid test (PLAN.md Phase 12): the diff that adds Gemma2 touches only
`ferrite-models/gemma2.rs`, `model_architectures/gemma2/*.json`, and new kernel
implementations. **Zero changes to `ferrite-forward` or `ferrite-forward-macro`.**
If a framework change creeps in, the rewrite failed its premise.

## Invariants to keep firing every commit

The PLAN's anti-bullshit checklist. Literally copy-paste into commit-prep:

- [ ] No string that names a transformer concept appears below the parser.
- [ ] No `BTreeMap<String, _>` keyed on a DSL identifier.
- [ ] No type with `.layer: u16`, `.num_layers: u16`, `weight_name: String`,
      `loop_bounds: BTreeMap<...>`, or `LoopPhase`.
- [ ] No `match tile_kind` or `if weight_name.contains(...)` anywhere.
- [ ] The phase's test runs on a realistic config (NL ≥ 16), not a synthetic stub.
- [ ] The test observes the claimed property, not just compile-success.

## Memory (from prior sessions — do not re-discover)

Already in the user's `~/.claude/projects/-home-moosevan-vllm/memory/`:

- `feedback_wire_up_new_code.md` — name algorithms honestly; wire replacements at
  every call site in the same commit.
- `feedback_show_dont_tell.md` — structural claims need real-input tests.
- `feedback_trace_real_input.md` — trace real inputs before claiming a path works.
- `feedback_integration_test_per_phase.md` — realistic N, not synthetic.
- `feedback_worktree_ops.md` — never checkout/stash in a worktree.
- `feedback_no_run_chat.md` — `timeout` + `vllm chat` for correctness.
- `feedback_build_flags.md` — `cargo build -p vllm-cli --features cuda --release`;
  not `cargo check --workspace`.

## Communication back to the user

The user (Nick) has been in pain on this codebase. Spend fewer words, ship more
code. When a design question surfaces, describe the trade-off in 3–5 sentences
and pick one — don't dither. When something breaks, investigate the root cause
rather than patching locally. Do not rename concepts to avoid removing them.
Do not add `#[allow(dead_code)]` or fallbacks without an explicit "consumed in
phase N" comment. Do not declare something done without an integration test
that observes the claim on a real input.
