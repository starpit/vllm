# Megakernel reset plan — SUPERSEDED

> **SUPERSEDED by `FERRITE_TK_PLAN.md` (2026-05-04).** This plan's
> core bet — reuse vendor `cross-gpu-llama` op bodies + vendor
> `megakernels/include/*` substrate with parameterization patches —
> failed: the substrate dragged in `state_t::instruction()` /
> `icode` / `controller_loop` (the tape/KVM machinery we wanted to
> leave behind), vendor `pgl<..., true, true, ...>` multicast init
> errored on single-device TP=1 runtime, and a TK upstream bump to
> 2.0 cascaded into ~46 vendor call sites across 8 `.cu` files. The
> vendored `third_party/megakernels/` tree was deleted in its
> entirety when this plan was superseded; every `#include` or
> template reference called out below is dead.
>
> Kept only as historical record. **Do NOT act on anything described
> below.** The active plan is `FERRITE_TK_PLAN.md` — ferrite owns
> its substrate and op bodies, built directly on TK 2.0 primitives,
> four warp-role walkers (loader / launcher / storer / consumer), no
> vendor megakernels in the `#include` path, no tape, no `icode`.

---

> **ORIGINAL PREAMBLE (historical):**
>
> **READONLY** — this document is frozen. Do not modify without
> explicit user consultation. Progress, findings, and in-flight work
> live in `MEGAKERNEL_RESET_PROGRESS.md` (append-only). If the plan
> needs to change, ask the user first.

Status: proposed, not yet started. Supersedes the prior patch-the-vendor
strategy that was producing garbage output after ~1 week of work.

## Core decision

Stop calling the vendor's tape-driven `mk` interpreter with a
`Globals<Llama-70B-TP=8>` instantiation and patching around its
hardcoded assumptions. Instead: **per KVM-eligible variant, codegen our
own `.cu` file** that

- instantiates vendor `Globals<our model's values>` with real dims
- walks Ferrite's lowered schedule and emits straight-line C++ for
  straight-line sections, `for (...)` for `Loop` ops, nested if nested
- calls TK op bodies (`qkv_rope_append`, `attention_decode`,
  `matmul_pipeline`, `silu_and_mul`, `lm_head`, etc.) directly — no
  opcode, no dispatch, no tape
- emits its own KVM substrate init (pages, semaphores, warp roles)
  either by `#include`-ing the vendor substrate or by codegen-ing it

**There is no interpreter.** The codegen'd `.cu` is the kernel. The
schedule is baked in at compile time.

## What dies with the tape/interpreter path

Since the old strategy was "build a tape, H2D it, vendor interpreter
plays it back," ripping the interpreter rips all of that:

- `ferrite-forward-macro/src/interpreter/kvm.rs` — the ~1.4K-line
  encoder: `encode_op`, `encode_bucket`, `variant_kvm_eligible`, the
  role-aware opcode dispatch, the `[[i32; 32]]` tape-row layout, the
  `OP_BARRIER_INC` double-emit logic. Gone.
- Opcode constants in sync with vendor's `OPCODE_*`
  (`OP_ATTN_NORM=1`, `OP_O_PROJ_RESIDUAL=5`, `OP_MLP_NORM=6`,
  `OP_DOWN_PROJ_RESIDUAL=9`, `OP_LM_HEAD_NORM=10`, `OP_BARRIER_INC`).
  Gone — codegen calls TK ops directly, not via opcodes.
- `ferrite-forward/src/interpreter/kvm.rs` — mostly gone. Tape H2D
  path, `d2d_stack_raw` stacking-for-tape-dispatcher, the `mk` extern
  launcher signature. The small amount that survives is renamed/
  repurposed (see "What stays").
- The vendor's tape-driven `mk` entry point in
  `third_party/megakernels/cross-gpu-llama/`. Codegen replaces it.

## What stays (codegen re-uses, interpreter-free)

- **Step 1 TK solver Impls** — the solver still picks TK for the
  eligible variants the same way.
- **Vendored TK + `cross-gpu-llama` op bodies**
  (`qkv_rope_append.cu`, `attention_decode.cu`, `matmul_pipeline.cuh`,
  `gate_silu.cu`, `up_matmul.cu`, `matmul_adds.cu`, `lm_head.cu`).
  Codegen calls these directly.
- **Vendor parameterization patches** already on this worktree: every
  `LLAMA_*` `#define` `#ifndef`-guarded, `globals_t::num_devices` off
  `LLAMA_NUM_DEVICES`, `gl_as_pgl<GL>` shim for TK 1.x vs 2.x,
  storer/assert generalizations for `num_kv_heads/num_devices > 1`.
  All still correct because codegen instantiates `Globals<...>` over
  these same defines.
- **TK substrate** that op bodies assume exists in shared memory:
  pages, semaphores, warp-role split, Category B allocations
  (NUM_CONSUMER_WARPS=8, NUM_PAGES=6, PAGE_SIZE=32768, etc.).
  Codegen emits the init + role dispatch, doesn't rewrite it.
- **NVCC build pipeline** (`ferrite-cuda-builder/build.rs`): scans the
  cudaforge cache, compiles `.cu` → `libmegakernels.a`, linked into
  `vllm-cuda`. Codegen just feeds it different `.cu` inputs.
- **emit_model hookup** (`codegen.rs` under `FERRITE_KVM=1`): per
  canonical, walk the lowered schedule, invoke codegen, write `.cu` to
  cache, emit Rust extern decl + wrapper. Shape preserved; body swapped.
- **Rust-side `launch()` extern wrapper** in `ferrite-forward`: the
  signature changes (no tape arg — instead shape/weight/rope pointers),
  but the "extract by-name weights, build arg pack, call per-canonical
  extern C fn" plumbing survives.
- **By-name weight classification** (in today's encoder): walks bucket
  for OpInstances whose weight accessor path contains
  `input_layernorm`, `self_attn_o_proj`, `mlp_down_proj`, `lm_head`,
  etc. Stays — but emits `T* w_attn_norm = layer.input_layernorm.weight;`
  C++ instead of opcode-tagged tape rows.

## Audit findings (already done)

Vendor `Globals` template in `third_party/megakernels/cross-gpu-llama/
llama.cuh:154-175` is already well-parameterized on all model dims
(head_dim, hidden_dim, intermediate_dim, num_attention_heads,
num_kv_heads, matmul_out_block_size, matmul_batch_block_size,
num_hidden_layers, kv_page_size, sm_count). Derived constexprs
(num_output_blocks, num_generated_heads_per_col, Q_COLS, KV_COLS)
recompute correctly.

Literals in the .cu files classified:

- **Category A** — already parametric via `Globals::` or `head_dim`
  typedefs. Most of the code. Fine.
- **Category B** — H100 infrastructure (NUM_CONSUMER_WARPS=8, NUM_PAGES
  =6, PAGE_SIZE=32768, DYNAMIC_SEMAPHORES=128, SCRATCH_BYTES,
  104/208 register allocations). Tied to H100 SRAM budget, asserted.
  Not model-specific, don't touch.
- **Category C** — `256` hardcoded as `matmul_out_block_size` in tile
  types across ~15 sites in 6 files. The real offenders. Sites:
  - `matmul_pipeline.cuh:23` — `b_st = st_bf<256, PIPELINE_K_DIM>`
    (weight tile — this is what was corrupting K/V writes)
  - `matmul_pipeline.cuh:49-50` — `get_output_tile` return type
  - `matmul_pipeline.cuh:176, 183` — `matmul_loop` register tile
  - `llama.cuh:192, 194` — `weights_t`, `weights_big_indim_t` TMA descriptors
  - `llama.cuh:214, 223, 224, 226` — activation / down / silu / logits gl descriptors
  - `qkv_rope_append.cu:232-233` — `matmul_rt`, `matmul_st`
  - `gate_silu.cu:63, 74`
  - `up_matmul.cu:16, 18, 92, 102`
  - `matmul_adds.cu:11, 62, 72`
  - `lm_head.cu:57, 67`
  All mechanical: replace `256` with `Globals::matmul_out_block_size`
  (or a file-local typedef pulling from it).
- **Category D** — `64`s in `activations_t`, `matmul_pipeline.cuh` etc.
  Vendor comment at `llama.cuh:197-200` states these are "vendor-tuned
  tile shapes (kv-block / pipeline-depth specific) that aren't
  head_dim-derived." Believable — `PIPELINE_K_DIM = 64`. Leave alone
  unless Phase 0 says otherwise.

## Plan

### Phase 0 — verification (before committing)

Two cheap checks that would blow up scope if wrong:

- Fork the .cu files, do the Category C replacement, try to compile
  with `Globals<...Llama-3.2-1B...>`. Compile errors tell us which
  Category D `64`s are actually head_dim-tied.
- Read Ferrite's lowered schedule for Llama-3.2-1B m=8. Confirm it's
  really pre + `Loop(16, per_layer_body)` + post, not something weirder.

Exit: both verified, or we know the specific extra work needed.

### Phase 1 — parameterize the vendor .cu files in place

Edit `third_party/megakernels/cross-gpu-llama/` (our vendored directory)
to replace Category C hardcodes with `Globals::matmul_out_block_size`.
Leave B and D alone unless Phase 0 says otherwise.

Exit: files compile as a drop-in replacement with
`matmul_out_block_size=256` — i.e., only parameterized, Llama-70B
behavior unchanged.

### Phase 2 — codegen the kernel

Rewrite `ferrite-forward-macro/src/interpreter/kvm.rs` (keeping the
weight-classification logic, ripping the encoder) to emit a `.cu` file
per KVM-eligible variant that

- `#include`s parameterized TK op headers
- declares `Globals<our_values>` with the model's real dims
- emits a `__global__` / `__launch_bounds__` kernel that:
  - initializes the KVM substrate (semaphore init, page allocator init,
    warp-role dispatch) — ideally via `#include` of the vendor substrate
    so we inherit future fixes
  - walks Ferrite's lowered schedule:
    - straight-line calls to TK ops for straight-line sections
    - `for (...)` around `Loop` bodies, with induction-variable
      substitution into body coords
    - nested loops if nested
- exports a plain `extern "C"` launcher that Rust calls with shape /
  weight / rope / kv / io pointers

The Rust-side `launch()` in `ferrite-forward/src/interpreter/kvm.rs`
loses its tape argument; everything else (weight extraction by-name,
arg pack, per-canonical extern call) stays.

Exit: codegen'd .cu compiles for Llama-3.2-1B m=8 and launches without
crashing.

### Phase 3 — correctness

- Single-request, single-token decode: output matches host-interpreter
  (`instr::run`) reference
- Multi-request batched decode: matches
- Multi-token prefill: matches

Exit: coherent output, matches host-interpreter reference within bf16
tolerance.

(No "Phase 4 keep tape path for diff." The host interpreter
`instr::run` is already the correctness baseline; the tape path was a
half-broken intermediate, not a reference.)

## Explicit non-goals

- No changes to TK op bodies (aside from Category C typedef fixes).
- No TK substrate rewrite (pages/semaphores/warps).
- No per-op standalone launches.
- No giving up cross-op pipelining — the codegen'd kernel is one
  launch, preserves producer/consumer page handoff.
- **No interpreter, no tape, no opcode dispatch.**

## Biggest risk

Phase 2 kernel codegen. Emitting the KVM substrate init correctly
(pages, semaphores, warp roles) is non-trivial. Mitigation: `#include`
the vendor substrate and generate only the schedule walker / op-call
sequence. Fall back to full substrate codegen only if the vendor
substrate also has undiscovered hardcodes.

## Rough sequencing

- Phase 0: hours
- Phase 1: ~1 day, mechanical
- Phase 2: the real work. Days. Iteration expected.
- Phase 3: hours to days depending on what breaks.
