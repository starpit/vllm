# ferrite-forward — the forward-pass compiler

> **If you are a fresh Claude reading this: read this whole file
> before you read any code or propose any work. Every section here
> exists because a prior Claude got the framing wrong and burned
> hours of user time on bullshit. Specifically, do not:**
>
> - **Think of codegen as "deciding" launch mode.** It doesn't.
>   HostCallable/DeviceCallable is a tag on each Implementation in
>   the library. Codegen emits a loop nest and invokes each Impl
>   the way its tag says. Codegen does not pick.
> - **Think of the solver as picking "launch mode."** The solver
>   picks Impls. An Impl's launch kind is an attribute like cost.
>   The solver is just a cost-minimising matcher — algorithm is
>   the DP ported from
>   `vllm-rs/crates/ferrite-solver/src/lowering/solver/dp.rs`
>   onto our clean FUF/Implementation types.
> - **Think of the scheduler as "packing fused spans" or "deciding
>   launch boundaries."** It turns the SFUF into a wavefront loop.
>   That's it. One job.
> - **Treat greedy as an MVP of DP.** It isn't — it's a different
>   algorithm that can't solve the problem. Old ferrite already
>   had the DP; port it.
> - **Treat `vllm-cuda/src/model/*.rs` as in-scope for replacement.**
>   It is not. vllm-cuda is untouched. The compiler replaces only
>   `ferrite-macros` + `ferrite-solver`; its output plugs in at the
>   same call site the old ferrite's output plugged in at.
> - **Treat the prior ferrite as non-working.** It worked — it ran
>   Llama and Qwen2 correctly. Its sin was llama-specific
>   hardcoding (booleans per layer, weight-name substring checks,
>   GemmQ/K/V tile variants). We replace the framework, not the
>   functionality.

Takes a DSL describing a forward pass. Compiles to Rust + CUDA that
vllm invokes for each forward step. Replaces the prior ferrite
compiler (`ferrite-macros` + `ferrite-solver`) with a real compiler
framework instead of llama-specific hardcoded behavior sprinkled
through the codegen.

The prior ferrite compiler produced working output — it ran Llama
and Qwen2 correctly. What made it wrong: the compilation path was
not a compilation path. It had booleans tripping per-layer behavior,
weight-name substring checks, special-cased tile kinds — so the
moment you tried to extend it to Gemma2 (or anything that didn't
match the assumptions baked in for Llama), the framework collapsed.

The new compiler produces output that plugs in at the exact same
call site as the old one. It does not change how vllm-cuda invokes
forwards, does not replace any vllm-cuda code, does not touch any
`.cu` sources. Its only intersection with vllm-cuda is that
vllm-cuda calls the ferrite-generated fn per forward step, the same
way it calls the prior ferrite's generated fn today.

One DSL body per architecture. Many models per architecture (one
JSON per model). Many workload buckets per model. The solver picks
one kernel Implementation per FUF tile × workload bucket. Some
kernels are HostCallable (extern "C", run on CPU), some are
DeviceCallable (`__device__` body, run on GPU); that tag is set by
the kernel author on the Implementation and the solver sees it like
any other attribute. Some kernels claim multiple FUF tiles at once
(fused kernels) — they still go through the solver as a single
Implementation choice.

Codegen's job is mechanical: walk the loop of waves the scheduler
produced and emit a loop nest that invokes each wave's kernels. Pure
host loops → pure Rust. Pure device loops → one CUDA kernel doing
the whole loop on-GPU. Mixed → Rust outer loop with host waves as
calls and device waves as megakernel launches.

## Compiler flow — read this before touching anything

The compiler has exactly three passes below the parser. Each pass
has ONE job. The output of each pass is a distinct IR with a name.
DO NOT conflate the jobs. DO NOT move work between passes.

```
  #[forward] fn llama() { DSL body }
          │
          ▼
  ┌──────────────────────────────────────────────────────────┐
  │ front end   parse → classify → shape-infer               │
  │             → build CFG → unroll                          │
  └─────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
                      ┌──────────┐
                      │   FUF    │   Fully Unrolled Forward.
                      │          │   An unrolled DAG of numeric
                      │          │   tile nodes (loops over
                      │          │   num_hidden_layers etc. are
                      │          │   expanded into concrete
                      │          │   tiles). One node per DSL op
                      │          │   occurrence. Zero strings,
                      │          │   zero loops, zero transformer
                      │          │   domain concepts below here.
                      └────┬─────┘
                           │
                           ▼
  ┌──────────────────────────────────────────────────────────┐
  │                       SOLVER                              │
  │                                                           │
  │  ONE JOB: match kernels (Implementations from the         │
  │  library) to FUF nodes, cheapest total cost for the       │
  │  given target profile.                                    │
  │                                                           │
  │  A kernel may claim MULTIPLE FUF nodes (a fused           │
  │  kernel covers several ops). The solver's output is       │
  │  therefore smaller than its input: tiles collapse into    │
  │  subgraphs, one subgraph per kernel instance.             │
  │                                                           │
  │  DP algorithm, polynomial, ported from old ferrite        │
  │  (see ferrite-solver/src/lowering/solver/dp.rs).          │
  │                                                           │
  │  NOT the solver's job: deciding launch mode (that's a     │
  │  tag the kernel author sets on each Impl; the solver      │
  │  reads it like any other attribute), loop structure,      │
  │  wave grouping, codegen. Don't put those here.            │
  └─────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
                      ┌──────────┐
                      │   SFUF   │   Solved FUF.
                      │          │   The FUF with every tile
                      │          │   assigned to a subgraph and
                      │          │   every subgraph assigned one
                      │          │   Impl from the library.
                      │          │   When Impls claim a single
                      │          │   tile each (the common case),
                      │          │   SFUF has the same node count
                      │          │   as FUF. When a fused Impl
                      │          │   claims multiple tiles, the
                      │          │   SFUF is smaller — those
                      │          │   tiles share one subgraph.
                      └────┬─────┘
                           │
                           ▼
  ┌──────────────────────────────────────────────────────────┐
  │                      SCHEDULER                            │
  │                                                           │
  │  ONE JOB: turn the SFUF back into a loop. BSP/wavefront   │
  │  is fine — subgraphs with no dependence between them      │
  │  share a wave; dependents go in later waves.              │
  │                                                           │
  │  NOT the scheduler's job: kernel selection (done),        │
  │  launch mode (codegen), emission (codegen). Don't put     │
  │  that here either.                                        │
  └─────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
                      ┌──────────┐
                      │   LOOP   │   Ordered sequence of waves.
                      │          │   A wave (== BSP superstep)
                      │          │   is a set of subgraphs with
                      │          │   no dependence on each other;
                      │          │   they're mutually concurrent.
                      │          │   Waves execute in sequence.
                      └────┬─────┘
                           │
                           ▼
  ┌──────────────────────────────────────────────────────────┐
  │                       CODEGEN                             │
  │                                                           │
  │  ONE JOB: emit a loop nest that walks the LOOP's waves    │
  │  and invokes each subgraph's Impl.                        │
  │                                                           │
  │  Every Impl in the library is tagged HostCallable or      │
  │  DeviceCallable — that's the kernel author's declaration, │
  │  not a codegen decision. Codegen reads the tag and emits  │
  │  accordingly:                                             │
  │                                                           │
  │    - A wave of all HostCallables    → Rust calls in       │
  │                                       sequence.           │
  │    - A wave of all DeviceCallables  → one __global__      │
  │                                       megakernel that     │
  │                                       runs the wave on    │
  │                                       the GPU.            │
  │    - 100% Host loop                 → pure Rust loop      │
  │                                       iterating waves.    │
  │    - 100% Device loop               → one CUDA kernel     │
  │                                       implementing the    │
  │                                       whole loop on-GPU.  │
  │    - Mixed                          → Rust outer loop;    │
  │                                       host waves emit as  │
  │                                       calls, device waves │
  │                                       emit as megakernel  │
  │                                       launches.           │
  │                                                           │
  │  Codegen does NOT pick kernels, does NOT decide fusion,   │
  │  does NOT schedule. The solver already picked the kernels │
  │  (a "fused" kernel is just a DeviceCallable Impl that     │
  │  claimed multiple FUF tiles at solve time). The scheduler │
  │  already produced the loop. Codegen's job is mechanical:  │
  │  emit the loop nest.                                      │
  │                                                           │
  │  Output: per model, one pub fn <model>_forward dispatched │
  │  on num_tokens; plus one .cu per emitted megakernel.      │
  └─────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
                   linked into vllm;
                   called at the same call site the prior
                   ferrite's output was called at.
```

### The IRs have names. Use them.

- **FUF** = Fully Unrolled Forward. Output of the front end. Input
  to the solver.
- **SFUF** = Solved FUF. Output of the solver. Input to the
  scheduler. FUF with each tile bound to a subgraph and each
  subgraph bound to one Impl.
- **LOOP** = ordered wave list. Output of the scheduler. Input to
  codegen.
- **wave** = BSP superstep. A set of mutually independent
  subgraphs. Waves are executed in sequence; subgraphs within a
  wave are concurrent.

Calling the solver's output "Assignment" or the scheduler's output
"Schedule" is fine as a Rust type name, but in prose use the IR
name (FUF/SFUF/LOOP) so it's obvious which pass produced it.

### Per-architecture vs per-model vs per-workload

The front end runs once per architecture (one DSL body).

The solver, scheduler, and codegen run per (model × workload
bucket). Models fan out over `model_architectures/<arch>/*.json`;
workload buckets fan out over the `num_tokens` range. Codegen
coalesces adjacent-equal assignments into match arms (`1..=8 =>
<one impl set>`, etc.) so the emitted fn isn't one arm per M.

## What gets replaced

- `ferrite-macros/` — the `forward!{}` proc-macro. Dead when the
  last `forward!{}` site has migrated to `#[forward]`.
- `ferrite-solver/` — the contaminated IR + codegen the old macro
  drove. Dead when `ferrite-macros` is dead. See "Explicit
  NON-reuse" below for why we don't reuse any of its types.
- `ferrite-models/src/*.rs` — the DSL-bearing sites only. The files
  stay but switch from `forward!{}` to `#[forward]`.

What stays untouched:

- `vllm-cuda` entirely. Kernel sources, tensor runtime, KV cache,
  allocator, the call site that invokes the ferrite-generated fn —
  all unchanged. The only edit to vllm-cuda would be if the new
  generated fn's signature diverges from the old one's, and that
  edit should be obvious and local.
- Everything above vllm-cuda (engine, server, CLI, bench).

Adding a new model = drop a JSON into `model_architectures/<arch>/`.
Adding a new architecture = DSL body + configs + kernels for any
new ops. No compiler edits.

## HostCallable vs DeviceCallable is a library tag

Each `Implementation` in the library has a launch-kind tag set by
the kernel author:

- **HostCallable** — invoked via `extern "C"` from CPU. Paged
  attention, flash attention, cutlass gemm, anything that needs
  host-side orchestration or that's already compiled as a
  host-launch kernel.
- **DeviceCallable** — has a `__device__` body that runs on the
  GPU. Can be composed with other DeviceCallable kernels into a
  single GPU kernel (a "megakernel"). Elementwise ops, rmsnorm,
  silu, add, rope, small reductions.

The tag is data. The solver reads it when matching Impls to FUF
tiles; the cost model may favor one tag over another based on the
surrounding context. Codegen reads it to emit the right kind of
call (Rust extern call vs. `__global__` launch).

The compiler's code contains no hardcoded list of which ops are
Host and which are Device — it's whatever the library says.

## Status

Foundation done (do not redo):

- Parse DSL → classified program (extern / weight / local).
- Shape inference with HF weight-name convention anchoring.
- CFG with integer trip counts; unroll to numeric FUF.
- Greedy per-tile solver, sweeps workload buckets, real costs for
  every op (weight shapes threaded from Inferred; first-None is
  fatal; workload dimension preserved).
- Topological-layering scheduler.
- 62 unit tests + 3 integration tests passing, fmt + clippy clean.

**Still to do — this is where the real work is:**

- **Replace** the greedy solver with the DP ported from
  `ferrite-solver/src/lowering/solver/dp.rs`. The greedy picker
  is not the algorithm; it's a placeholder written before the user
  clarified the DP was already proven on Llama + Qwen2. Port it.
- Rework the scheduler to emit a real LOOP of waves (today's
  topological layering is close, but its output is `Vec<Step>` —
  update the type and the downstream consumer accordingly).
- Codegen does not exist yet. Emit `pub fn <model>_forward`
  containing a `match num_tokens` dispatch; each arm walks the
  LOOP emitting Rust calls for host waves and megakernel launches
  for device waves.
- Add DeviceCallable Impls to the library (elementwise tail, rope,
  rmsnorm) so the solver has real Host/Device alternatives.
- Migrate `ferrite-models/src/llama.rs`, then `qwen2.rs`, then the
  rest. Each is `forward!{}` → `#[forward]`.
- Validate with `vllm chat` / `vllm bench` / `vllm serve`.

Everything before codegen is plumbing. The compiler emits an empty
fn today.

## Remaining work

Ordered by dependency, not phase number.

### 1. DP solver + scheduler + Host-only codegen: Llama

Three things in parallel because they can't ship independently:

- **Solver**: port the DP from
  `vllm-rs/crates/ferrite-solver/src/lowering/solver/dp.rs` onto
  our FUF + Implementation types. Output is the SFUF. Kill the
  current greedy picker.
- **Scheduler**: walk the SFUF topologically, produce a LOOP of
  waves.
- **Codegen**: emit `pub fn <model>_forward(...)` per model
  containing a `match num_tokens { … }` dispatch. Each arm walks
  the LOOP and emits Rust calls for each subgraph's Impl (all
  HostCallable at this stage — no DeviceCallable Impls exist yet).

`ferrite-models/src/llama.rs` switches from `forward!{}` to
`#[forward]`. Nothing in vllm-cuda changes.

Done when `timeout 60 vllm chat --model=<small-llama>` produces
coherent output and `vllm bench` hits parity with the old ferrite
path.

### 2. Qwen2 through the same pipeline

Add Qwen2's bias-add as a HostCallable Impl in the library (a new
op, one new Impl — not a compiler edit).
`ferrite-models/src/qwen2.rs` switches to `#[forward]`. Same
pipeline, same codegen, different body and library.

### 3. DeviceCallable Impls + megakernel emission in codegen

Add DeviceCallable Implementations to the library for the ops where
on-GPU composition pays (rmsnorm, silu, add, rope, elementwise
bias-add — the memory-bound tail around gemm/attention). Each has a
`__device__` body in a `.cu`.

Codegen's output shape now varies per wave: a wave whose subgraphs
all map to DeviceCallable Impls emits as one `__global__` (the
megakernel for that wave). A pure-device LOOP emits as one CUDA
kernel implementing the whole loop on-GPU. Mixed LOOPs emit as
Rust outer loop + host-wave calls + device-wave megakernel
launches. Same codegen pass; what it emits is driven by the Impl
tags in the LOOP it was handed.

The solver now has real alternatives per tile (HostCallable vs
DeviceCallable for the ops that have both). Cost model favors
DeviceCallable when neighbors in the same wave are also
DeviceCallable, because the launch-overhead amortises.

Done when `vllm bench` shows measurable throughput gain vs.
pure-Host on the same model.

### 4. Migrate remaining architectures

One arch at a time, each is a DSL body + configs + any new-op
kernels. `vllm chat` / `serve` / `bench` stay green throughout.
Order roughly by DSL complexity (smallest first, genericity
stress-tests later):

- llama, qwen2 (covered in 1–2)
- gemma2 (the genericity acid test — see 5)
- mixtral (MoE — see 6)
- qwen2_moe
- deepseek_v2 (MLA — see 7)

### 5. Gemma2 acid test — compiler genericity

Gemma2 stresses the compiler where every prior "generic" approach
broke: sliding window attention alternating by layer, pre+post norms
on both attention and FFN, approximate GELU, query scaling, logit
soft-capping.

The diff that adds Gemma2 must touch **only**:
- a new `ferrite-models/src/gemma2.rs` with the DSL body,
- `model_architectures/gemma2/*.json`,
- new kernel impls for new ops (`gelu`, `soft_cap`,
  `sliding_attention`, `query_scale`).

**Zero lines** in `ferrite-forward-macro` or `ferrite-forward`. If
the compiler needs surgery to express Gemma2, the design is wrong —
fix the compiler's generality, not the Gemma2 diff.

### 6. MoE acid test — control-flow genericity

Mixtral / Qwen2-MoE need top-k routing inside the forward. The DSL
today has no construct for "for each of top-k selected experts, run
this sub-body." Adding one is a generic language extension, not a
MoE-specific branch. When we write it, it must also express any
future conditional dispatch pattern (conditional compute, early exit,
speculative branches).

### 7. MLA acid test — attention-variant genericity

DeepSeek-V2's Multi-Latent Attention is a different attention op,
not a compiler change. A new `attention_mla` op + new kernel +
Shape signature. If anything below the op signature has to learn
"MLA," the compiler is leaking.

### 8. Delete the legacy

When the last `forward!{}` site is gone: delete `ferrite-solver`,
`ferrite-macros`, and any support code only those two required.
vllm-cuda is untouched.

## Invariants

Every step above preserves these. When an invariant breaks, the break
is the bug, not the invariant.

1. **Below the parser, no transformer-domain concepts.** No "layer,"
   no "iter," no "phase," no `NL`-as-a-string, no `hidden_states`
   special case, no `GemmQ/K/V` sub-kinds, no weight-name substring
   matching. AST may carry symbolic bound names verbatim from the
   DSL; everything below the CFG is either numeric or a small enum.

2. **FUF is numbers only.** `FufNode { id: u32, op: OpKind, inputs:
   Vec<Input>, shape: Shape }` with `Input = Tile(u32) | Weight(u32)
   | Extern(ExternKind)`. No strings, no identifiers.

3. **Three categories of free vars, classified at parse time.**
   Extern params (fixed enum: `input_ids`, `positions`, `rotary`,
   `block_table`, `kv_cache`), weight refs (HF paths), locals. No
   sentinels. `hidden_states` is a local binding, period.

4. **One `#[forward]` per architecture.** Attribute-form, attached
   to an empty `fn <arch>()` so rustfmt formats the body.

5. **Launch mode is a library attribute, not a compiler attribute.**
   A new kernel with a new launch mode goes in the library. The
   compiler reads launch mode off the Implementation; it does not
   infer it from DSL syntax or op kind.

6. **Realistic integration tests at every step.** Run on real configs
   at realistic N (≥16 layers). Compile-success is not a test.
   The test must observe the claimed property on real input.

7. **No silent fallbacks.** An impl's cost function returning `None`
   is a hard error. An unknown op is a hard error. A missing shape
   is a hard error. A candidate that can't apply at a given M opts
   out via an explicit `applicability_fn`, not by returning `None`.
   The class of bug this prevents: gemm costs silently dropped
   because weight shapes weren't threaded, the workload dimension
   silently collapsed because a caller passed one M and nothing
   complained. Both happened in the current greedy solver before
   being fixed; neither would have been caught by compile-success
   tests alone.

## Explicit NON-reuse

- `ferrite-solver` wholesale. Its types are contaminated
  (`TileKind::GemmQ/K/V/...`, `weight_name: String`, BTreeMaps on
  DSL idents). Consuming them means importing the contamination.
  **Exception**: the DP algorithm in
  `ferrite-solver/src/lowering/solver/dp.rs` is explicitly ported
  — the algorithm is clean, the types it operated over are not.
  Re-implement against our FUF + Implementation types; do NOT
  import the old `Problem`, `TileGraph`, or `Assignment`.
- `ferrite-macros::forward!` as a parser. Fresh `syn`-based parse
  against the AST we designed is cheaper than untangling.
- Any type with `.layer: u16`, `.num_layers: u16`, `weight_name:
  String`, `loop_bounds: BTreeMap<...>`, `LoopPhase`, or a
  `TileKind::GemmQ/K/V/...`-style variant. Those are the exact
  llama-specific hardcodings that made the prior ferrite collapse
  under Gemma2. Importing a function that requires one of those
  imports the contamination.

## Non-goals

- Backwards compat with `ferrite_macros::forward!{}`. Gone when the
  last site migrates.
- Touching vllm-cuda. The compiler's output plugs in at vllm-cuda's
  existing call site for ferrite-generated forwards.
- Runtime solver invocation. Everything is compile-time.
- Rewriting existing CUDA kernel sources. They stay and are linked
  as Host-launch Implementation entries.
- Rewriting the tensor runtime, KV cache manager, or page allocator.
  Orthogonal.
