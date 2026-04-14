# ferrite-forward — parallel compiler path

A clean reimplementation of the `forward!` pipeline. Separate crate tree,
zero reuse of the existing `ferrite-solver` IR or DSL parser.

**Reuse default is NO.** The existing `ferrite-solver` tree has
transformer-domain concepts leaked into its types (`TileKind::GemmQ/K/V`,
`weight_name: String` on tiles, `classify_gemm` substring matching,
`BTreeMap<String, _>` on the CFG, etc.). Any code that *consumes* those
types is contaminated even if its own algorithm is clean. The solver, the
wavefront scheduler, the cost model, the implementation library — none are
presumed reusable. Reuse is an explicit exception that has to clear a
concrete bar (see Phase 7/8).

Things outside the compiler are fine to reuse: the CUDA kernel source
files themselves, `ferrite-kernel-builder`, the tensor runtime, the KV
cache manager. Those don't know about the IR.

## Invariants (every phase must preserve these)

1. **Below the parser, no transformer-domain concepts.** No "layer," no
   "iter," no "phase," no `NL`-as-a-string, no `hidden_states` special case,
   no `GemmQ/K/V` sub-kinds, no weight-name substring matching. AST may
   carry symbolic bound names verbatim from the DSL; everything below the
   CFG is either numeric or a small enum.
2. **FUF is numbers only.** A FUF node is `{ id: u32, op: OpKind,
   inputs: Vec<Input>, shape: Shape }` with `Input = Tile(u32) |
   Weight(u32) | ExternParam(ExternKind)`. No strings, no identifiers,
   no substring matches.
3. **Three categories of free vars classified at parse time.**
   - non-weight params: `input_ids`, `positions`, `rotary`, `block_table`,
     `kv_cache` — fixed enum, same across all models.
   - weight refs: `self_attn.q_proj[layer]`, `lm_head`, etc. — resolved
     per-model via the config.
   - local bindings: everything the DSL introduces with `name = ...`.
   `hidden_states` is a local binding, not a special case. If a DSL reads
   it before writing it, that is a parse-time error — not papered over
   with a sentinel.
4. **One `#[forward]` per architecture.** The macro is attribute-form,
   attached to an empty `fn <arch>()` carrier, so rustfmt formats the body
   like any normal Rust fn.
5. **Realistic integration tests at each phase.** Tests run on real
   Llama/Qwen2 config.json inputs, not synthetic `<NL=3>` stubs. A phase
   isn't done until there's a test that observes the claimed property on a
   real input.

## Crates

- `ferrite-forward-macro/` — `proc-macro = true`. Entry point:
  `#[proc_macro_attribute] fn forward(args, item) -> TokenStream`.
- `ferrite-forward/` — consumer-facing crate. Re-exports the macro, holds
  any runtime types the generated code depends on.
- `model_architectures/` — at repo root. `llama/*.json`, `qwen2/*.json`.

## Phases

Each phase ends with a buildable, commit-able, test-passing state. No phase
may introduce a shortcut that a later phase has to undo.

### Phase 0 — scaffolding + configs

- Create `vllm-rs/crates/ferrite-forward-macro/` (proc-macro crate) and
  `vllm-rs/crates/ferrite-forward/` (consumer crate).
- The attribute macro is a stub: accepts `#[forward]`, reads and discards
  the fn's body, emits an empty fn. No DSL parsing yet.
- Create `model_architectures/llama/` and `model_architectures/qwen2/`.
  Download canonical configs via `curl`:
  - Llama (from unsloth mirrors — ungated):
    - llama-2-7b, llama-2-13b, llama-2-70b
    - llama-3-8b, llama-3-70b
    - llama-3.1-8b, llama-3.1-70b, llama-3.1-405b
    - llama-3.2-1b, llama-3.2-3b
  - Qwen2 (official Qwen repos — ungated):
    - qwen2-0.5b, qwen2-1.5b, qwen2-7b, qwen2-72b
    - qwen2.5-0.5b, qwen2.5-1.5b, qwen2.5-3b, qwen2.5-7b,
      qwen2.5-14b, qwen2.5-32b, qwen2.5-72b
  File stem is the generated identifier, so lowercase/hyphenated.
- **Test:** `#[forward] fn llama() {}` compiles into an empty fn.
- **Commit:** `ferrite-forward: scaffolding + model architecture configs`.

### Phase 1 — parse DSL body into AST

- Define the AST: `Program { statements }`, `Statement::{Assign, ForLoop}`,
  `Expr::{Call, Var, VarAt, FieldAccess, BinOp, Literal}`.
- Parse via `syn`: walk the carrier fn's body, recognize:
  - assignment statement `name = expr;`
  - tuple-destructuring assignment `(a, b, c) = expr;`
  - for-loop `for v in 0..<bound> { ... }` with the bound being an
    identifier (symbolic).
  - dotted field access (`self_attn.q_proj`) and indexing (`[layer]`).
- No classification yet; the AST mirrors source structure.
- **Tests:**
  - Parse a realistic Llama body (the one we've sketched). Assert top-level
    statement count, loop trip-count expression is `Ident("num_hidden_layers")`.
  - Parse a Qwen2 body (slight arch differences: KV head count, norm
    placement — whatever the Qwen2 arch actually is; I'll look up the
    reference forward in the HF source during this phase).
- **Commit:** `ferrite-forward: parse DSL body into AST`.

### Phase 2 — classify free variables

- Walk the AST, build three tables:
  - `ExternParams`: match against a fixed enum (`input_ids`, `positions`,
    `rotary`, `block_table`, `kv_cache`). Anything else in read position
    that isn't a local binding or a weight ref is an error.
  - `Weights`: any reference like `foo.bar[idx]` or bare `foo` that's read
    but never written.
  - `Locals`: anything written by an assignment statement (SSA; a rewrite
    shadows the prior binding).
- Classification is a separate pass; AST nodes get annotated with
  `VarClass` enum.
- **Test:** classify realistic Llama body. Assert `hidden_states` is a
  local binding (written by two `hidden_states = add(...)` statements),
  `self_attn.q_proj` is a weight ref, `input_ids` is an extern param.
- **Commit:** `ferrite-forward: classify DSL free variables`.

### Phase 3 — read config.json, resolve per-model bounds

- Directory walker: given `model_architectures/<arch>/`, enumerate
  `*.json`. For each, parse into `ModelParams { name, bounds }` where
  `bounds: BTreeMap<String, u64>` maps field names like
  `num_hidden_layers` → `16`.
- Register every file read via `proc_macro::tracked_path::path`, plus
  the directory itself, so cargo rebuilds on add/edit/remove.
- **Test:** load all 10 Llama configs; assert extracted bounds match the
  known published values (e.g. `llama-3.2-1b.num_hidden_layers == 16`).
- **Commit:** `ferrite-forward: read model configs from directory`.

### Phase 4 — shape inference

- Shape language: `Shape = Vec<Expr>` where `Expr` is arithmetic over
  bound names and literals (`[HD, NAH * HDM]` etc.).
- Per-op shape signature: each DSL op knows its input/output shapes in
  terms of its operand shapes. Examples:
  - `gemm(x: [.., K], w) → [.., N]` implies `w: [K, N]`
  - `rmsnorm(x: [.., H], w) → [.., H]` implies `w: [H]`
  - `embed(ids, table) → [.., H]` implies `table: [V, H]`
  - `rope_append((q, k, v), positions, rotary, kv_cache) → (q', k', v')`
    with shape-preserving semantics
  - `attention(q, k, v, kv_cache, block_table) → [.., H']`
- Walk the classified AST, propagate shapes from extern-param inputs
  outward. Weight refs get their shape inferred from usage context.
- **Test:** for realistic Llama body, inferred shape for `self_attn.q_proj`
  is `[hidden_size, num_attention_heads * head_dim]` — match published
  value.
- **Commit:** `ferrite-forward: infer weight shapes from DSL usage`.

### Phase 5 — build CFG from AST

- CFG with basic blocks, terminators `Jump`, `Return`, `LoopHeader { ivar,
  start: u64, end: u64, body, exit }`. Bounds are *integers* because the
  caller (Phase 6) substitutes per-model params.
- No `BTreeMap<String, usize>`. No symbol table at this level — lookups
  happened at parse time via Phase 3.
- **Test:** build a CFG from a classified AST + a chosen set of concrete
  bounds; assert number of blocks, loop header start/end values.
- **Commit:** `ferrite-forward: build CFG from classified AST`.

### Phase 6 — unroll CFG into FUF

- Flat graph: `Fuf { nodes: Vec<FufNode> }`. `FufNode { id: u32, op:
  OpKind, inputs: Vec<FufInput>, shape: Shape }`. `FufInput = Tile(u32) |
  Weight(u32) | ExternParam(ExternKind)`. `OpKind` is one variant per DSL
  op (`Gemm`, `RmsNorm`, `Embed`, `Rope`, `Attention`, `Silu`, `Add`,
  `Mul`). No `GemmQ/K/V` sub-kinds.
- Unroller substitutes loop-variable references with concrete integers.
  All weight refs resolve to numeric weight-slot IDs (by hashing/indexing
  into the schema derived in Phase 4).
- **Tests:**
  - `build_fuf` of the realistic Llama-3.2-1B spec has `1 + 15*16 + 2 =
    243` tiles (or whatever the real per-iteration count is once we've
    settled it — the point is: `tile_count = f(N)` and we assert against
    a realistic N).
  - No node anywhere in the FUF references a `String` or an identifier.
    (Compiler enforces this structurally, but also a runtime assertion
    during dev.)
- **Commit:** `ferrite-forward: unroll CFG into numeric FUF`.

### Phase 7 — solver

- **Default: write fresh.** The existing DP solver (`ferrite-solver::
  lowering::solver::dp`) consumes a `Problem` built from the contaminated
  `TileGraph`, with tile-kind-specialized cost lookups and an
  `ImplementationLibrary` keyed on `TileKind::GemmQ/K/V/...`. The
  algorithm is clean; the interface is not. Writing a fresh DP over our
  FUF is simpler than building a bidirectional adapter that stays honest.
- **Reuse bar** (must ALL hold to reuse):
  1. Zero references to any `TileKind` variant beyond a single unified
     `Gemm`.
  2. No `String`-keyed maps in the solver's input or output.
  3. No weight-name inspection anywhere in the cost path.
  4. The adapter from our FUF to the solver's `Problem` is ≤ 50 LOC and
     has no `match tile_kind` or `if weight_name.contains(...)`.
  5. The existing tests that exercise the DP solver continue to pass
     after the adapter-only change — so we can be confident the
     algorithm works as intended.
- If any of those fail: write fresh. Do not ship a contaminated adapter
  and promise to clean it up later.
- The solver's input is the FUF + an implementation library + a target
  profile. Output is an assignment of tiles → impls + a predicted cost.
- **Test:** realistic Llama body + starter implementation library
  produces an assignment with no unassigned tiles. Solve time < 100 ms
  on Llama-3.1-8B's unrolled graph.
- **Commit:** `ferrite-forward: wire (or write) solver`.

### Phase 8 — schedule (step-merge)

- Given the solver's per-tile impl choices plus its preliminary
  one-per-step schedule, produce the final BSP schedule: a list of
  steps, each holding one or more `(tile, impl)` pairs.
- Merge rules:
  - two tiles can share a step if their impls' claim-states don't
    collide AND they satisfy cooperative-exclusive constraints AND
    no edge between them would skip a step boundary in violation of
    dependency order.
  - only merge when it reduces predicted cost (handoff elimination ≥
    serialization overhead of packing).
- Output: `Schedule { steps: Vec<Step> }`, `Step = Vec<(TileId, ImplId)>`.
- **Default: write fresh.** Same reasoning as the solver — the existing
  merge pass consumes contaminated types.
- **Test:** for realistic Llama body, the merged schedule has strictly
  fewer steps than the solver's preliminary one-per-step schedule, and
  cost goes down or stays equal (never regresses).
- **Commit:** `ferrite-forward: BSP schedule via step-merge`.

### Phase 9 — codegen: Rust glue

- For each model in `models/`, emit:
  - one `pub fn <model_name>_forward(...)` specialized to its bounds
  - call into the compiled CUDA kernels via `extern "C"` bindings
- Weight refs resolve to per-arch runtime paths (`model.layers[i].self_attn.
  q_proj`), driven by a small per-arch table in the macro.
- Non-weight params become typed fn arguments.
- **Test:** `cargo check` on a downstream crate that invokes the
  macro-generated forward compiles.
- **Commit:** `ferrite-forward: emit specialized Rust forward fns`.

### Phase 10 — codegen: CUDA

- Macro writes `.cu` files to `$OUT_DIR` plus a manifest JSON.
- `ferrite-kernel-builder`'s `build.rs` picks up the manifest and compiles
  the `.cu` files, emitting linker directives.
- **Test:** full build succeeds, linker resolves the generated symbols.
- **Commit:** `ferrite-forward: emit CUDA and wire into kernel builder`.

### Phase 11 — real inference (Llama)

- Swap the model crate (`ferrite-models::llama`) to use
  `#[forward]` instead of the legacy `forward!()`.
- `vllm chat --model=<small-llama>` produces coherent output (correctness
  check per `feedback_no_run_chat`).
- Latency parity or better vs. legacy path.
- **Commit:** `ferrite-models: switch Llama to #[forward]`.

### Phase 12 — Gemma2 (the acid test)

This is the whole reason for the rewrite. If the compiler is genuinely
generic, adding Gemma2 is:

1. Write the Gemma2 body DSL in a new `#[forward] fn gemma2()` block.
2. Drop Gemma2 configs into `model_architectures/gemma2/`.
3. Build. Inference works.

Zero framework changes. No new parser arms, no new CFG quirks, no new
solver branches, no new codegen special cases. If any of those become
necessary, the rewrite failed its premise and we have to go back.

Gemma2 stresses the framework because it differs from Llama in ways that
have historically broken generic compilers:

- **Sliding window attention** alternating per layer. Every odd/even
  layer uses a different attention span. If the DSL has to grow a new
  `sliding_attention` op — fine, generic extension. If the CFG or FUF
  has to grow per-layer metadata to track "is this layer windowed" —
  failed.
- **Double norms** (post-attention AND post-FFN layernorm). Structurally
  different body. The parser/CFG/unroll/codegen should not care.
- **Approximate GELU** instead of SiLU. New DSL op `gelu`. Adding a DSL
  op should be: one `OpKind` variant + one shape signature + one kernel
  implementation. Nothing else.
- **Query scaling**. Possibly a new op or possibly parameterized
  attention. Either way, it goes in the DSL body, not the compiler.
- **Logit soft-capping**. New op `soft_cap`. Pure post-processing.

**Success criteria** (must ALL hold):
1. The diff that adds Gemma2 touches ONLY:
   - a new `gemma2.rs` file in `ferrite-models` with the DSL body,
   - new `.json` files in `model_architectures/gemma2/`,
   - kernel implementations for any *new* DSL ops (`gelu`, `soft_cap`,
     `sliding_attention`) — data, not control flow.
2. No file in `ferrite-forward-macro/` or `ferrite-forward/` is modified
   by the Gemma2 diff.
3. `vllm chat --model=gemma-2-2b` produces coherent output.
4. The Gemma2 body is visually comparable in size/complexity to the Llama
   body. No explosion of special cases to work around the generic layer.

If any criterion fails, the compiler is not generic and the rewrite has
not achieved its goal. We fix it by making the offending pass generic,
not by carving a Gemma2-specific escape hatch.

- **Commit:** `ferrite-models: add Gemma2 via #[forward]`.

## Explicit non-goals for this plan

- Rewriting the CUDA kernel *sources* (reused — they don't know about
  the IR).
- Rewriting ferrite-kernel-builder (reused — it just compiles `.cu` files).
- Rewriting the runtime (tensor allocation, KV cache mgmt — unchanged).
- Back-compat with the legacy `forward!` — both live until Phase 11;
  after that, delete the old one in a separate commit.

## Explicit NON-reuse

- The existing `ferrite-solver` crate, wholesale. Its types are
  contaminated; consuming them means importing the contamination. Any
  reuse has to pass the Phase 7/8 bar.
- The existing `ImplementationLibrary` type as-is. Its keys are
  `TileKind::GemmQ/K/V/...` variants. We need a library keyed on a single
  `Gemm` kind with its dimensions stored as `Shape`, not as a variant
  selector. We may port the *data* (FLOP counts, microbenchmark numbers)
  but not the container type.
- The existing `forward!` proc-macro parser. Writing a fresh `syn`-based
  parser against the AST we design in Phase 1 is cheaper than untangling
  what's there.
- Any type with a `.layer: u16`, `.num_layers: u16`, `weight_name: String`,
  `loop_bounds: BTreeMap<...>`, or `LoopPhase` field. If importing a
  function would require importing a type with one of these fields, the
  function doesn't cross the boundary.

## Anti-bullshit checklist (run before committing any phase)

- [ ] Does the diff introduce any string that names a transformer concept
      below the parser? (`"hidden_states"`, `"q_proj"`, `"NL"`, etc.) If
      yes — why? Move it back up.
- [ ] Does any pass take a `BTreeMap<String, _>` keyed on a DSL identifier?
      If yes — the info should have been resolved at parse time.
- [ ] Does any type carry a `layer: u16` or `num_layers: u16` field? If
      yes — that's the old IR leaking back. Delete.
- [ ] Does any code path special-case a specific DSL identifier? If yes —
      replace with a structural predicate.
- [ ] Does the phase's test run on a realistic config (N ≥ 16 layers),
      not a synthetic `NL=2` stub?
- [ ] Does a new test actually observe the phase's claimed property, or
      just assert that compile succeeds?
