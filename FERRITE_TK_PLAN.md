# Ferrite-TK megakernel codegen plan

> **READONLY** — this document is frozen. Do not modify without
> explicit user consultation. Progress, findings, and in-flight work
> live in `FERRITE_TK_PROGRESS.md` (append-only). If the plan needs
> to change, ask the user first.

Status: proposed, not yet started. Supersedes
`MEGAKERNEL_RESET_PLAN.md`, which tried to reuse the vendor
`cross-gpu-llama` op bodies and vendor `megakernels/` substrate;
that path kept dragging in vendor assumptions, opcode dispatch, and
version-coupled APIs. This plan drops the vendor megakernels
substrate entirely and builds on ThunderKittens 2.0 primitives only.

## Core decision

Ferrite's proc-macro emits one `.cu` file per KVM-eligible variant.
Each file contains:

- A ferrite-owned `Globals<model_dims...>` template holding only the
  pointers + shape scalars the kernel actually reads. No instruction
  tape, no `Bar` pgl, no vendor-shaped `globals_t`.
- A ferrite-owned `SharedState<Config>` struct declaring the exact
  page count, scratch bytes, and semaphore set this variant needs —
  nothing more.
- A single `__global__` kernel body that dispatches to four warp-role
  walkers, each a straight-line C++ walk of ferrite's lowered
  schedule. Ops are called by name, with arguments inline, at
  compile time. Zero runtime dispatch.

**There is no opcode, no tape, no `state_t::instruction()`, no
controller warp, no `icode` switch.** The schedule is baked in at
compile time by ferrite's codegen.

The ops themselves (rms_norm, gemv, rope_append, attention_partial,
silu_upgate, lm_head) live in a ferrite-owned header-only library
written directly against TK 2.0 primitives (`tma::*_async`, `wgmma`,
`sv_bf`/`st_bf`/`rt_bf`, `semaphore`, `wait`/`arrive`). None of the
vendor `cross-gpu-llama/*.cu` or `megakernels/include/*` files are
included or reused.

## Warp layout — four roles

Each warp in the kernel has exactly one of four roles, picked at
kernel entry by `warpid()`:

- **consumer** — the register-resident math. `NUM_CONSUMER_WARPS`
  of these, high register allocation (`CONSUMER_REGISTERS`). Does
  MMA accumulation, softmax, rmsnorm reduction, silu, etc.
- **loader** — issues `tma::load_async` into shared-memory pages;
  signals `page_ready`. Low register allocation
  (`NON_CONSUMER_REGISTERS`).
- **launcher** — issues `wgmma.mma_async` / Blackwell
  `tcgen05.mma` on behalf of consumer warpgroups; on Blackwell also
  owns tensor-memory allocation. Low register allocation. On Hopper
  this role may be fused into `consumer` for simplicity; the first
  cut folds it in and revisits later if register pressure demands
  separation.
- **storer** — issues `tma::store_async` out of shared pages to
  gmem; waits on completion semaphores. Low register allocation.

Ferrite owns all the knobs: `NUM_CONSUMER_WARPS`,
`CONSUMER_REGISTERS`, `NON_CONSUMER_REGISTERS`, `NUM_PAGES`,
`PAGE_SIZE`, `SCRATCH_BYTES`, `INSTRUCTION_PIPE_STAGES`. These live
in a per-variant `ferrite_config.cuh` emitted alongside the kernel
so every knob is visible and tunable per schedule.

**Why no controller warp:** mk-v2 / throughput use the controller
warp to fetch the next instruction, decode its opcode, and dispatch
to the right op's worker. Ferrite has no runtime tape — the
schedule is baked at compile time, so there is nothing to fetch and
nothing to dispatch. Walker bodies call op functions by name
directly.

## Cross-op pipelining (INSTRUCTION_PIPE_STAGES-style overlap)

Straight-line emission of role bodies gives us per-op pipelining
for free (via TK primitives' ping-pong tile loading) but misses the
load-for-op-N+1-while-computing-op-N overlap that throughput and
mk-v2 get from their instruction-pipeline-stages indirection.

**Ferrite achieves cross-op overlap by emitting the loader walker
N stages ahead of the consumer walker.** Concretely:

- `consumer_body` walks schedule ops `[0, 1, 2, 3, ..., K-1]`.
- `loader_body` walks the same schedule but shifted: it emits the
  load work for op 0 first, then op 1, then op 2 — while the
  consumer walker is still on op 0.
- Shared-memory pages are the handoff buffer. Each op instance's
  loader call writes to a specific `stage` slot; consumer reads
  from that same slot. After consumer finishes, the page is
  released (via `arrive(page_done[stage])`) and the loader walker
  may reuse it for a later op.
- `INSTRUCTION_PIPE_STAGES` is the pipeline depth. 2 is the
  baseline (op N consumer overlaps with op N+1 load); 4 may be
  worth exploring for decode paths with cheap ops.
- **Ferrite-side page allocator.** Codegen tracks which pages are
  live at each schedule point and emits explicit `wait(page_done)`
  / `arrive(page_done)` at reuse points. No runtime page manager,
  no `pid_order[]`, no controller dispatch.

## Subtile wavefront across SMs

Some ops (large-reduction attention, lm_head against the full
vocab) need more work than one SM can do in a single pass. Ferrite
splits these across SMs using gmem-resident barriers — plain CUDA,
no TK `pgl` needed.

- A ferrite-owned `barriers` tensor in gmem sized
  `(num_layers, num_ops_with_cross_sm_dep, num_batch_blocks,
  num_tiles)`. Init'd to zeros.
- Producer SMs `atomicAdd(&barriers[...], 1)` on completion.
- Consumer SMs poll for expected count. `__threadfence_system()` /
  `__threadfence()` as needed for ordering.
- No multi-GPU multicast. No `pgl<>`. No cuMulticastCreate. TP=1
  means one GPU; the barrier pool is plain gmem.
- Exact barrier shape is picked by ferrite's schedule walker based
  on which ops actually have cross-SM fan-in/out.

## What stays, what dies

### Stays
- TK 2.0 vendored headers (`third_party/thunderkittens/include/`) —
  upstream commit `4b0aa30d...`. Primitives only: tile types, TMA
  wrappers, MMA wrappers, semaphores, warpgroup utilities.
- Ferrite lowered schedule — the output of step 1 (solver), step 2
  (lowering), and loop compression. Unchanged by this plan.
- `ferrite-forward-macro`'s proc-macro hook that runs on
  `FERRITE_KVM=1` and writes `.cu` files to
  `~/.cache/cudaforge/megakernels/`. Entry point stays; body of
  what it emits is replaced.
- `ferrite-cuda-builder/build.rs` pipeline that scans the cudaforge
  cache and nvcc-compiles the `.cu` files into `libmegakernels.a`.
  Unchanged — it just feeds different `.cu` inputs.
- Rust-side `KvmLaunchArgs` + `launch()` plumbing in
  `ferrite-forward/src/interpreter/kvm.rs`. Shape of the struct
  changes (fewer fields — no tape, no Bar, no per-device pgl
  pointers to replicate), but the extern-C ABI + fn-pointer
  dispatch stays.
- The existing schedule-walker logic in
  `ferrite-forward-macro/src/interpreter/kvm.rs` and
  `interpreter/variant_cpp.rs` — loop compression, per-op emission,
  `layer_override` substitution for `Loop` bodies. All reusable;
  only the *per-op text it emits* changes.

### Dies
- **`third_party/megakernels/cross-gpu-llama/*`** — vendor op
  bodies. `batched_rms_norm.cu`, `qkv_rope_append.cu`,
  `attention_prefill.cu`, `attention_decode.cu`, `matmul_adds.cu`,
  `matmul_pipeline.cuh`, `gate_silu.cu`, `up_matmul.cu`,
  `lm_head.cu`, `inc_barriers.cu`, `all_device_barrier.cu`,
  `llama.cuh`. All replaced by ferrite-owned TK kernels. The
  directory may be kept as a reference (AUDIT.md) but is NOT in
  the `#include` path.
- **`third_party/megakernels/include/*`** — vendor substrate
  (`megakernel.cuh`, `controller.cuh`, `loader.cuh`, `storer.cuh`,
  `launcher.cuh`, `consumer.cuh`, `config.cuh`, `util.cuh`). All
  replaced by ferrite substrate headers. Not in `#include` path.
- **`globals_t<>` / `llama_70b_globals`** typedef — replaced by
  ferrite-owned `Globals<>`.
- **The `_mc_enabled` + `num_devices` parameterization patches**
  applied to vendor `llama.cuh` during the MEGAKERNEL_RESET work —
  vendor `llama.cuh` is no longer included, so these patches are
  moot.
- **The TK-2.0-port-of-vendor-op-bodies work** — unneeded; we
  write ferrite-owned TK-2.0-native code from the start, no
  porting of 46 vendor call sites.
- **The `instruction_t` / `icode` / `dispatch_instruction` /
  `state_t::instruction()` apparatus** — no tape, so nothing to
  fetch/decode/dispatch.

## Reference (used, not reused)

- **mk-v2-llama's `csrc/itypes/llama1b/*.cuh`** — treat as idiom
  source for TK-2.0 patterns: how to set up TMA descriptors for
  bf16 weight tiles, how to do warpgroup MMA for bs=1 decode, the
  attention softmax+scale+store pattern. Copy the TK-primitive
  sequences; drop the `state_t` / `instruction` / `pipeline_
  specifics` wrappers. Do NOT `#include` these files — copy the
  idioms into ferrite-owned headers.
- **mk-v2's `csrc/workers.cuh`** — idiom source for producer/
  consumer page-ready/page-done semaphore patterns. Same copy-
  idiom-not-code policy.

## Ferrite substrate library (new code under ferrite ownership)

Header-only, lives under `crates/ferrite-kernels/csrc/tk/`
(exact path TBD during Phase 1 but it must be ferrite-owned, not
under `third_party/`). One header per concern:

- `ferrite_config.cuh` (per-variant, emitted alongside each
  `.cu`) — `NUM_PAGES`, `PAGE_SIZE`, `NUM_CONSUMER_WARPS`,
  `CONSUMER_REGISTERS`, `NON_CONSUMER_REGISTERS`,
  `INSTRUCTION_PIPE_STAGES`, `SCRATCH_BYTES`.
- `ferrite_globals.cuh` — `template<model dims...> struct
  Globals { /* pointers + shape scalars only */ };`.
- `ferrite_substrate.cuh` — `template<Config> struct SharedState {
  pages, page_ready, page_done, scratch, ... }` + init helpers.
- `ferrite_barrier.cuh` — gmem barrier atomicAdd+poll helpers.
- `ferrite_warp_roles.cuh` — role dispatch at kernel entry (the
  `if (warpid() < NUM_CONSUMER_WARPS) ... else switch
  (warpgroup::warpid()) { ... }` macro).
- `ferrite_kernels/<op>.cuh` — per-op header. Each defines four
  functions in the op's namespace:
  - `<op>::consumer(Globals&, SharedState&, int stage, <args>)`
  - `<op>::loader  (Globals&, SharedState&, int stage, <args>)`
  - `<op>::launcher(Globals&, SharedState&, int stage, <args>)`
  - `<op>::storer  (Globals&, SharedState&, int stage, <args>)`
  (launcher may be empty on Hopper first cut.)

## Ops to build (derived from ferrite's schedule)

The op set is not locked in by this plan; it's whatever ferrite's
lowered schedule actually emits after the solver + lowering + loop
compression. Based on today's `ferrite-model-llama` shape and the
existing `interpreter/variant_cpp.rs` mappings, the Phase-2/3 set
is expected to be roughly:

- `rms_norm` — RMSNorm (standalone — first op to land, simplest).
- `gemv_bf16` — bs=1 decode matmul (matvec).
- `gemm_bf16` — multi-token prefill path matmul.
- `rms_qkv_rope_append` — fused input_layernorm + QKV gemm + rope
  + KV cache append.
- `attention_partial` — split-SM decode attention (partial softmax
  + partial output).
- `attention_reduction` — cross-SM reduction merging partial
  outputs into final attention.
- `o_proj_residual` — o_proj gemm + residual add.
- `silu_upgate` — fused post_attention_layernorm + up_proj +
  gate_proj + silu multiply.
- `down_proj_residual` — down_proj + residual add.
- `lm_head` — fused final norm + lm_head gemv.

(Prefill variants may need a separate `attention_prefill` op; TBD
when we look at the m>1 lowered schedule.)

## Phases

### Phase 0 — tear out vendor megakernels substrate

- Strip `#include "llama.cuh"`, `#include "*.cu"` op bodies,
  `#include "config.cuh"`, `#include "util.cuh"` from
  `emit_prelude` in `ferrite-forward-macro/src/interpreter/
  kvm.rs`.
- Remove the `using Config = llama_config;` / `using Globals =
  llama_70b_globals;` lines; Config/Globals will come from
  ferrite substrate headers in Phase 1.
- Stub the kernel body to a bare `__global__ void
  ferrite_<variant>_kernel(Globals g) {}` that does nothing.
- Verify the stub compiles against TK 2.0 alone
  (`#include "kittens.cuh"` only) on pod with nvcc.

Exit: empty `__global__` emitted per variant, compiles with TK
2.0 headers only, no vendor megakernels dependency.

### Phase 1 — ferrite substrate skeleton

- Write the six substrate headers under ferrite ownership.
- Codegen `emit_cu_variant` composes:
  1. `#include "kittens.cuh"`
  2. `#include "ferrite_config.cuh"` (emitted inline per variant)
  3. `#include "ferrite_globals.cuh"` + variant's `Globals<>`
  4. `#include "ferrite_substrate.cuh"`
  5. Kernel body: init semaphores, role-dispatch to four empty
     walker functions (`loader_body`, `launcher_body`,
     `storer_body`, `consumer_body`), then exit.
- Codegen extern-C launcher body populates `Globals<>` from
  `KvmLaunchArgs` and launches the kernel.
- Verify: kernel launches, returns `cudaSuccess`, produces no
  output (walkers are empty).

Exit: empty 4-role kernel launches without crashing.

### Phase 2 — first op: `rms_norm`

- Write `ferrite_kernels/rms_norm.cuh` — four role functions, no
  fusion, simplest possible shape.
- Update `interpreter/variant_cpp.rs` (or whatever schedule-walker
  emitter) so the ferrite `RmsNorm` op emits calls to
  `rms_norm::{consumer,loader,launcher,storer}` in the four
  walker bodies.
- Single-op test harness: populate `KvmLaunchArgs` with real
  bf16 input + weight, call the launcher, compare output against
  ferrite host-interpreter reference (within bf16 tolerance).
- Exit gate is numeric match on a minimal input — no
  pipelining, no cross-SM split.

Exit: ferrite-codegen'd RmsNorm matches host-interpreter
reference (bf16 tolerance). Proves the codegen shape, substrate,
and role dispatch all work.

### Phase 3 — remaining ops

For each op in the schedule's op set, in order of complexity:

1. `gemv_bf16` (then `gemm_bf16` for prefill).
2. `rms_qkv_rope_append`.
3. `attention_partial` + `attention_reduction`.
4. `o_proj_residual`, `down_proj_residual`.
5. `silu_upgate`.
6. `lm_head`.

Each op: write the four role functions, register the op→function
mapping in `variant_cpp.rs`, single-op numeric test vs host
interpreter. No cross-op pipelining yet — walkers remain
sequential.

Exit: full llama-3.2-1B m=8 decode produces correct output
matching host-interpreter reference (bf16 tolerance) end-to-end.

### Phase 4 — cross-op pipelining

- Ferrite codegen emits the loader walker N stages ahead of the
  consumer walker; `INSTRUCTION_PIPE_STAGES = 2` baseline.
- Codegen tracks page liveness across schedule points; emits
  `wait(page_done[...])` / `arrive(page_done[...])` at reuse
  points.
- Benchmark: nsys timeline shows consumer_warp compute and
  loader_warp TMA overlap across op boundaries.

Exit: cross-op TMA-compute overlap visible in nsys;
decode-latency improvement vs Phase 3 baseline.

### Phase 5 — subtile wavefront + perf

- Ops with cross-SM fan-in (attention_reduction, lm_head) split
  their tile space across SMs; ferrite codegen emits
  per-SM schedule slicing.
- Ferrite-owned gmem barrier init + wait helpers used at sync
  points.
- Benchmark vs Python vLLM reference on llama-3.2-1B m=8 decode
  latency.

Exit: latency matches or beats Python vLLM on llama-3.2-1B m=8
decode.

## Non-goals (out of scope for this plan)

- **Multi-GPU / TP>1.** First pass is TP=1. Barrier pool is plain
  gmem; no TK `pgl<>`, no NCCL, no multicast. TP>1 is a follow-up
  plan.
- **Blackwell.** Target Hopper (sm_90a) first. Launcher/consumer
  role split, `tcgen05.mma`, tensor-memory allocation, CLC
  scheduling — all Blackwell-era concerns we defer.
- **CLC / global work queue.** Per-SM block-indexed work
  (`blockIdx.x`-addressed schedule slice). Simple.
- **Reuse of any vendor megakernels op body or substrate file.**
  Ferrite-owned TK code only. Vendor `megakernels/` may stay in
  the tree as reference but not in the `#include` path.
- **Quantized variants in Phase 2-3.** Baseline is bf16 weights +
  bf16 activations. AWQ / BNB / GPTQ / FP8 variants are
  follow-up work; the proc-macro already emits per-variant .cu
  files so adding them later doesn't require substrate changes.

## Biggest risks

- **Hopper `wgmma.mma_async` synchronization patterns.** Fence /
  commit / wait ordering is easy to get wrong and failures are
  silent (garbage output). Mitigation: crib the exact pattern
  from mk-v2-llama `llama1b/matvec_pipeline.cuh`, reimplement in
  ferrite-owned code, test each op in isolation against host
  interpreter before composing.
- **Register budget.** `NUM_CONSUMER_WARPS * CONSUMER_REGISTERS +
  3 * NON_CONSUMER_REGISTERS` must fit. Starting estimate: 8 *
  192 + 3 * 64 = 1728, within H100's 64K/thread × 32-thread warp
  budget. Phase 2 validation on a real op confirms; knobs in
  `ferrite_config.cuh` are easy to revisit.
- **Cross-op page reuse correctness.** Semaphore bugs are silent.
  Mitigation: Phase 3 tests each op in isolation (no reuse),
  Phase 4 adds reuse only after single-op correctness is
  established and adds isolation tests for the pipelining itself.
- **TK-2.0 primitive surface instability.** Upstream TK occasionally
  renames / reshapes primitives. Mitigation: pin TK to a specific
  commit in `third_party/thunderkittens/VENDOR.md`, revendor
  deliberately on a bump.

## Rough sequencing

- Phase 0: hours
- Phase 1: 1–2 days
- Phase 2: 2–3 days (first op end-to-end shakes out substrate
  bugs; subsequent ops go faster)
- Phase 3: ~1 week (7–9 ops × ~1 day each)
- Phase 4: 2–3 days
- Phase 5: days to weeks depending on performance chase

## Precedent / why this supersedes the reset plan

`MEGAKERNEL_RESET_PLAN.md` bet that reusing vendor
`cross-gpu-llama` op bodies + vendor megakernels substrate with
parameterization patches would be faster than rewriting. The bet
lost:

- Parameterization patches compounded (every model dim needed
  `#ifndef`-guard audit + `num_devices` template threading).
- Vendor op bodies were written against pre-TK-2.0 pgl /
  `tma::*_async` APIs; a TK bump cascaded into ~46 vendor call
  sites across 8 `.cu` files.
- The substrate brought along `instruction_t` / `icode` /
  `controller_loop` / `dispatch_instruction` — the exact tape/KVM
  machinery we were trying to escape. Even with the tape not used,
  the substrate's `state_t::instruction()` accessor forced every
  op body to read from a fake instruction struct we'd have to
  populate per call.
- Vendor's `llama.cuh` `num_devices = 8` hardcode and `pgl`
  multicast-init in the constructor were incompatible with TP=1
  single-GPU runtime without additional patches.

This plan cuts the knot: ferrite owns its substrate, its op
bodies, and its constants. TK 2.0 is a clean dependency because
we only touch its primitive surface (tile types, TMA, MMA, sync)
— surfaces that upstream treats as stable. The four-warp-role
codegen pattern preserves every megakernel benefit that matters
(single launch, intra-op pipelining, cross-op pipelining, subtile
wavefront) without inheriting the vendor pump's runtime dispatch.
