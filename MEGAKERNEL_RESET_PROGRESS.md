# Megakernel reset — progress log — SUPERSEDED

> **SUPERSEDED by `FERRITE_TK_PROGRESS.md` (2026-05-04).** The plan
> this log tracks (`MEGAKERNEL_RESET_PLAN.md`) failed for reasons
> summarized in the banner at the top of that file. The log is
> preserved as historical record of what was tried, what broke, and
> why. **Do NOT pattern-match off this log as a source of current
> design.** Every Phase 2 artifact described below — the vendor
> `Globals<>` brace-init launcher, the `_bar_data[ND]` pgl data-ptr
> arrays, the `<op>::{consumer,loader,storer}::run(g, kvms)` walker
> emissions, the `_mc_enabled` vendor parameterization patches —
> was removed or gutted when this plan was superseded, alongside
> the vendored `third_party/megakernels/` tree.

---

> **ORIGINAL PREAMBLE (historical):**
>
> Append-only. Newest entries at the bottom. Each entry: date, phase,
> what happened, what's next.
>
> Companion to `MEGAKERNEL_RESET_PLAN.md` (readonly).

---

## 2026-05-03 — worktree established

**Branch**: `worktree-ff-mega-codegen` off `origin/ff-interpreter`
(commit `766966778` — "ff-gguf: tp>1 coherent across Llama / Qwen2/3 /
Mistral / Phi-3 / Gemma-3").

**Prior work location**: `worktree-ff-interpreter-mega` (archaeology
branch, untouched). 31 commits of tape/KVM-based strategy end-to-end
decode on H100 but output-correctness unverified, ~1344 lines of
uncommitted prior-session Rust on top. Abandoned by user decision
("keep nothing") in favor of clean restart.

**Carried over from archaeology**:
- `MEGAKERNEL_RESET_PLAN.md` — readonly plan (as of this session).
- `vllm-rs/third_party/megakernels/AUDIT.md` — numerical constant
  audit (line numbers reference the old branch's post-rename
  `ferrite/` tree; serves as reference, not authoritative map of
  current tree).

**Vendor drop (clean, no patches)**:
- TK from `~/git/ThunderKittens` @ `cce72c2f5c71c3ab812f27f96d6289e412baed60`
  → `vllm-rs/third_party/thunderkittens/` (excluded `.claude/` and
  `tests/batch-vm/llama_sm89/`; provenance in `thunderkittens/VENDOR.md`).
- Megakernels from `~/git/Megakernels` @ `91eaff262c2b473cfdcb135f5f2abefbe2835fe9`
  → `vllm-rs/third_party/megakernels/{cross-gpu-llama,include,megakernels}/`
  (vendor-original dir name preserved; provenance in `megakernels/VENDOR.md`).
  `cross-gpu-llama` is the "Throughput Llama" demo per upstream README.

**Commit status**: None. Vendor drop is untracked.

**Next**: Phase 1 — Category C substitutions on `cross-gpu-llama/`
tile types (bare `256` → `Globals::matmul_out_block_size`). See plan.

## 2026-05-03 — Phase 1 complete (Category C substitution)

Ran perl in-place replacements across `vllm-rs/third_party/megakernels/
cross-gpu-llama/`. 28 runtime literal-256 sites replaced with
`Globals::matmul_out_block_size` (inside op `.cu` templates which take
`<Config, Globals>`) or bare `matmul_out_block_size` (inside
`globals_t` struct scope in `llama.cuh`).

Substitutions (perl regex):
- `st_bf<256, ...>`, `st_bf<..., 256>`, `st_bf<16, 256>`,
  `rt_bf<16, 256>`, `rt_fl<16, 256>`, `st_fl<16, 256>` → with
  `Globals::matmul_out_block_size` replacing the `256`.
- `tt<float, 128, 256>` (Blackwell tensor-memory allocate) →
  `tt<float, 128, Globals::matmul_out_block_size>`.
- `half_consumer::groupid() * 256` → `* Globals::matmul_out_block_size`.
- `qkv_rope_append.cu` rope_arrived_sem: `sizeof(float) * 128 * 256 /
  WARP_THREADS` → `sizeof(float) * Globals::matmul_batch_block_size *
  2 * Globals::head_dim / WARP_THREADS`.

Files touched:
- `llama.cuh` (6 sites in weights_t / activations_parallel_t /
  activations_big_indim_t / logits_t)
- `matmul_pipeline.cuh` (5 sites: b_st, get_output_tile ×2,
  matmul_loop return, local `out`)
- `qkv_rope_append.cu` (3 sites: matmul_rt/matmul_st typedefs + rope
  size expr)
- `gate_silu.cu` (4 sites: rt_fl out_fl/gate_buf, tt<...>, *256)
- `up_matmul.cu` (5 sites: silu_tile Blackwell/Hopper, rt_fl
  out_fl/silu_fl, tt<...>, *256)
- `matmul_adds.cu` (5 sites: output_tile, rt_bf out, rt_fl
  matmul_out, tt<...>, *256)
- `lm_head.cu` (4 sites: rt_bf out, rt_fl matmul_out, tt<...>, *256)

Verification: `grep -n 256 ... | grep -v // | grep -v #define` returns
zero sites. All remaining `256`s are in comments or the `#ifndef`
default macros (`LLAMA_MATMUL_OUT_BLOCK_SIZE`, `LLAMA_MATMUL_BATCH_BLOCK_SIZE`).

**Exit criterion met**: the tree compiles as a drop-in replacement
when `matmul_out_block_size=256` (vendor's default config); 70B
behavior unchanged. Pod-side compile check deferred until we have
a build target that exercises this file.

**Commit status**: No commits yet. Vendor drop + Phase 1 edits are
all unstaged/untracked in the working tree.

**Next**: Phase 2 — codegen. Rewrite (or rather: write from scratch,
since we're not carrying the old `kvm.rs`) a Ferrite proc-macro
module that walks the lowered schedule and emits a per-variant `.cu`
file. Responsibilities:
- `#include` parameterized TK op headers.
- Emit `#define FERRITE_*` overrides (num_layers, hidden_dim, ...)
  before including `llama.cuh`.
- Declare a `Globals<...>` typedef bound to the model's dims.
- Emit `__global__` entry via `mk<Config, Globals, OPS_LIST>` (the
  TK substrate handles the warp-role dispatch and instruction pump).
- Supply a `static constexpr` tape / schedule the `mk` pump walks —
  OR write our own `mk`-replacement that bakes the schedule into
  straight-line C++ (per plan's preference).

Plan's biggest risk is Phase 2. TBD which `mk` strategy to use:
reuse vendor's `mk` with a static instruction list, vs emit straight-
line C++. Vendor demo uses a Python-host-built instruction list
streamed to the GPU; we want schedule baked in at compile time.
Needs research on the `mk` substrate's API before committing.

## 2026-05-03 — Phase 2a: substrate / pump boundary identified

Read vendor `third_party/megakernels/include/{megakernel.cuh,config.cuh,
util.cuh,controller/controller.cuh}`. Confirmed the boundary the plan
anticipates ("#include the vendor substrate, generate only the schedule
walker / op-call sequence").

### Substrate — keep via `#include` / re-use

These are architecturally neutral and bound only to the TK model of
shared-memory pages + semaphores + instruction buffer:

- `kittens.cuh` — TK primitives (tiles, MMA, TMA, semaphores, warp
  groups).
- `include/config.cuh` — `default_config` (pipeline-stage count, widths,
  page size, register caps). Per-model config (`ferrite_config`) in
  `cross-gpu-llama/llama.cuh` derives from the same pattern.
- `include/util.cuh` — `state<config>` (binds SRAM-resident
  `instruction_state_t[PIPELINE_STAGES]`, `instruction_arrived/finished`
  semaphores, `instruction_fetch_ready` semaphore, pages, timing
  helpers). Accessors `state::instruction()`, `::timing()`,
  `::pid_order()`, `::scratch()` index the ring by `instruction_ring`.
  The `state<config>` type is needed as-is because op bodies reference
  its members directly (pages, scratch, semaphores(), etc.).
- `include/util.cuh::instruction_state_t<config>` — 128-byte-aligned
  per-instruction SRAM chunk containing the instruction i32 buffer,
  timing buffer, pid_order, dynamic semaphores, scratch.
- `include/util.cuh::page<config>` — shared-mem page struct.

### Pump — dies; codegen emits its replacement

These are the interpreter. All must go per the plan:

- `include/megakernel.cuh::mk_internal` — the kernel entry that
  allocates shared-mem layouts, initializes ring-based semaphores,
  and dispatches to role main_loops.
- `include/megakernel.cuh::mk` / `mk_cutlass` — the `__global__`
  wrapper.
- `include/controller/controller.cuh::main_loop` — the instruction
  fetch / `atomicAdd(&g.global_instruction_index,1)` / opcode dispatch
  loop.
- `include/loader.cuh`, `storer.cuh`, `consumer.cuh`, `launcher.cuh`
  — per-role instruction pumps that wait on `instruction_arrived`,
  dispatch by opcode, signal `instruction_finished`.
- `include/controller/instruction_fetch.cuh` — reads instruction rows
  from `g.instructions` into the ring.
- `include/controller/semaphore_constructor.cuh`,
  `page_allocator.cuh` — opcode-keyed dispatch that calls each op's
  `controller::init_semaphores(s)` / `controller::release_lid(...)`.
- `dispatch_op<..., ops...>` template — the opcode→op jump table.

### Op-body coupling

The vendor `.cu` op bodies (`qkv_rope_append`, `attention_decode`,
`gate_silu`, `up_matmul`, `matmul_adds`, `lm_head`,
`batched_rms_norm`, `inc_barriers`, `all_device_barrier`,
`attention_prefill`) each expose an inner struct with five statics:

```cpp
struct parsed_instruction { ... };            // reads from kvms.instruction()
struct controller { release_lid, init_semaphores };
struct loader   { run(const Globals&, state<config>&); };
struct storer   { run(const Globals&, state<config>&); };
struct consumer { run(const Globals&, state<config>&); };
struct launcher { run(const Globals&, state<config>&); };
```

They assume the interpreter has populated `kvms.instruction()[]` with
their arg pack (layer, local_row, local_col, row, col, …) before
invocation.

### Codegen path (minimal deviation from op bodies)

Per plan: we keep the op bodies unchanged. Our `__global__` replaces
`mk_internal` + the four `main_loop`s. For each scheduled op in the
lowered schedule:

1. **Populate `state.instruction()[]`** — compile-time constants go
   in slots 1..N (layer, row, col, etc.); slot 0 (vendor opcode
   field) is irrelevant since we don't dispatch by opcode.
2. **Call op's `controller::init_semaphores(s)`** — sets up per-op
   semaphores in `state.semaphores()[]`.
3. **`__syncthreads()` / `everyone::sync()`** — publish
   semaphore init to all warps.
4. **Warp-role dispatch** — `if (warpid() < NUM_CONSUMER_WARPS) {
   op::consumer::run(g, s); } else switch (warpgroup::warpid()) {
   case 0: op::loader::run(...); case 1: op::storer::run(...);
   case 2: op::launcher::run(...); }`. The `case 3: controller`
   path has nothing to do — the controller warp already ran
   `init_semaphores` on its solo path before the sync.
5. **Sync** — wait for the op to complete before the next op's args
   clobber the instruction buffer.

For Loop ops in the schedule: wrap steps 1-5 in `for (uint32_t
__layer = 0; __layer < ITERS; ++__layer) { ... }`, substituting
`__layer + baseline` for each row's `layer` field at step 1.

**No tape**: there is no `g.instructions` global tensor referenced.
**No dispatch**: each op call is a direct C++ call to the op's
statics; no `dispatch_op<..., ops...>` template, no opcode switch.
**No interpreter**: no `controller::main_loop`, no
`instruction_arrived/finished` semaphores, no ring rotation. Ring
stays at 0 for the whole kernel life (op bodies access
`kvms.instruction_ring` indirectly via accessors; we leave it set
to 0 in state).

### Open questions for Phase 2b (not plan-blocking — will resolve
during implementation)

1. **instruction_arrived / instruction_finished** — Op bodies don't
   reference these directly (they're read by the pump only). We can
   omit them from our `state` construction.
2. **`instruction_fetch_ready`** — Referenced by op `loader::run`
   implementations (e.g. `warp::arrive(s.instruction_fetch_ready,
   NUM_CONSUMER_WARPS)` at start of loader to release the
   controller). With no controller main_loop waiting, this arrive
   is a no-op; semaphore still needs to exist so `s.instruction_fetch_ready`
   is a valid ref. Init to 0 and ignore arrives.
3. **`page_finished` ring semaphores** — Op bodies call
   `s.finish_page(pid, NUM_CONSUMER_WARPS)` / `s.wait_page_ready(pid)`
   using the ring-indexed `page_finished[pid][ring_bit]` semaphore.
   Without the interpreter's between-op page rotation, we need to
   re-init these between ops. Simplest: use ring_bit=0 always,
   re-init each page's single semaphore at start of each op.
4. **Timing records** — `s.loader_record(LOAD_EVENT)` etc. write to
   `kvms.timing()`. Harmless if the `timing_t` buffer exists in
   SRAM but never gets flushed to global. Can skip the flush path.

### Next (Phase 2b)

Minimum viable codegen: a Rust proc-macro helper that emits a `.cu`
with the above kernel shape around a single hardcoded
`batched_rms_norm` op call. Goal: compile it, prove the substrate
pattern works without `mk`. No lowered-schedule walking yet.

Module location: `vllm-rs/crates/ferrite-forward-macro/src/
interpreter/kvm.rs` (plan-specified path).

## 2026-05-03 — Phase 2b: minimum-viable codegen module landed

Created `vllm-rs/crates/ferrite-forward-macro/src/interpreter/kvm.rs`
(+ `interpreter/mod.rs` + `mod interpreter;` in `lib.rs`).

Module exports:
- `KvmDims` — per-variant model dims (num_layers, hidden_dim, head_dim,
  etc.). Field `matmul_out_block_size: u32` is constrained to `2 *
  head_dim` by vendor assert. `KvmDims::llama_3_2_1b_hopper_tp1()`
  returns the canonical Phase 2b test dim set.
- `emit_cu_phase2b(variant_name, dims) -> String` — emits a complete
  `.cu` source for a single hardcoded `attn_norm` op call.

Emitted `.cu` structure (verified via unit tests):

1. **FERRITE_* defines** (per-variant, before any vendor header).
2. **LLAMA_* → FERRITE_* bridge defines** — vendored tree still uses
   LLAMA_* names post-Phase-1; aliases let FERRITE_* overrides reach
   vendor `#ifndef` guards.
3. **Substrate-only includes**: `kittens.cuh`, `config.cuh`,
   `util.cuh`. Deliberately NOT `megakernel.cuh` (that's the
   interpreter pump). Op bodies still come in via their `.cu`
   includes (`batched_rms_norm.cu`, `qkv_rope_append.cu`, etc.).
4. **Kernel body**: replaces `mk_internal`:
   - Inline SRAM setup (pages, semaphores, instruction_state ring).
     Ring fixed at 0; `instruction_arrived/finished` allocated but
     never signalled (no pump).
   - Same semaphore-init block as `mk_internal`.
   - `kittens::everyone::sync(15)` to publish init.
   - Compile-time constant store into `kvms.instruction()[1..N]`.
   - Controller warp (wid == NUM_CONSUMER_WARPS+3) calls
     `Op::controller::init_semaphores(g, kvms)`.
   - `kittens::everyone::sync(15)`.
   - Warp-role dispatch: consumers → `Op::consumer::run`; others
     → loader/storer/launcher by warpgroup::warpid() 0/1/2.
   - Final `kittens::everyone::sync(15)`.

Tests (both pass):
- `phase2b_emits_nonempty_cu` — asserts emitted `.cu` contains the
  kernel-function name, FERRITE_* defines at the right values, op
  dispatch for all four roles, and does NOT contain the banned
  constructs: `g.instructions`, `::mk<`, `dispatch_op`, `OPS_LIST`.
- `matmul_out_block_size_is_2x_head_dim` — pins the
  `2*head_dim == matmul_out_block_size` invariant in Rust so it
  can't drift from the vendor `static_assert`.

### Not yet wired

- Schedule walking (Phase 2c). Current emit is hardcoded to one
  `attn_norm` op call at layer=0, item_idx=0; doesn't consume a
  `LoweredBucket`.
- Per-op argument mapping (Phase 2c). The mapping
  `OpInstance variant → which kvms.instruction()[] slots` is per-op
  and needs a lookup table (pattern taken from archaeology branch
  but rebuilt fresh).
- Weight extraction / launch-side Rust glue (Phase 2d). Extern
  launcher that Rust calls with weight/rope/kv/io pointers, weight
  by-name classification.
- Pod-side nvcc compile of the emitted `.cu`. Deferred — first
  prove schedule walking works locally, then validate on H100.

### Commits

- `80bd16920` — Phase 2a substrate/pump analysis.
- (pending) — Phase 2b codegen module.

## 2026-05-04 — Phase 2c: bucket walker + variant framework

Added `interpreter/variant_cpp.rs` with per-variant C++ emitters +
`parse_u32_literal` helper. Extended `interpreter/kvm.rs` with:

- `emit_bucket_body(instances, shapes_by_name) -> String` — walks
  the post-loop-compression `Vec<OpInstance>` in order. On `Loop`
  sentinel: emits `for (uint32_t __layer = 0; __layer < iters;
  ++__layer) { ... }` wrapping the next `body_len` instances,
  substituting `__layer + <baseline>` for each body row's `layer`
  field (baseline extracted from the instance's own layer
  field_value, preserving the vendor-fused-boundary layer-N vs
  layer-N+1 distinction).
- `emit_one_instance_with_layer_override(inst, shapes, Option<&str>)`
  — emits one op's C++ block. Resolves `layer_expr` per the
  substitution rule above, delegates to `variant_cpp::emit_op_block`.
  Unmapped variants produce `// TODO(phase2c): variant <name> …`.

Variant mappings so far:
- `Free` / `Alias` → empty (universal sentinels, no C++ needed).
- `Loop` → handled directly by walker, never routed to
  `emit_op_block`.
- `RmsNorm` → `attn_norm<Config, Globals>` (Phase 2c hardcoded; role
  dispatch to `attn_norm` / `mlp_norm` / `lm_head_norm` based on
  weight accessor path deferred to Phase 2d).
- All other variants → `TODO(phase2c)` placeholder.

Tests (all 10 pass):
- `variant_cpp::parse_u32_literal_handles_suffix_variants` — both
  `u32_suffixed` and `u32_unsuffixed` literal forms parse back.
- `variant_cpp::rms_norm_emits_warp_role_dispatch` — emitted block
  has all four warp-role invocations + expected instruction-slot
  populates.
- `variant_cpp::unmapped_variant_returns_none` — FusedQkvRopeCache
  and CutlassGemmAdd return None (caller generates TODO comment).
- `variant_cpp::free_and_alias_emit_empty`.
- `kvm::walker_emits_straight_line_for_nonloop_bucket` — two
  consecutive RmsNorm instances → two op blocks, no `for`.
- `kvm::walker_wraps_loop_body_with_for_and_layer_substitution` —
  `[RmsNorm(L=0), Loop(14,1), RmsNorm(L=1 baseline), RmsNorm(L=15)]`
  → one `for (...; __layer < 14u; ...)`, body substitutes
  `__layer + 1u`, prefix/suffix keep their own layer literals, 3
  total op blocks.
- `kvm::walker_emits_todo_for_unmapped_variant` — FusedQkvRopeCache
  emits TODO.
- `kvm::walker_emits_nothing_for_free_and_alias` — Free emits
  empty.

### Not yet wired

- Role dispatch for RmsNorm / CutlassGemmAdd by weight-accessor
  path inspection.
- The 10+ remaining KVM-eligible variants (FusedQkvRopeCache,
  AttentionViaCache, FusedGateUpSiluMul, CutlassGemm, etc.).
- `emit_cu_variant(variant_name, dims, backbone, lm_head, shapes)`
  top-level function that composes prelude + SRAM setup + bucket
  bodies. Currently `emit_cu_phase2b` still emits a hardcoded
  single-op body.
- `emit_model` hookup so the walker actually runs during
  proc-macro expansion for real canonicals.

Each of these is a Phase 2d-e item.

### Commits

- `80bd16920` — Phase 2a.
- `3ee0d4463` — Phase 2b.
- (pending) — Phase 2c.

## 2026-05-04 — Phase 2d: RmsNorm role dispatch + per-SM batch partitioning

Extended `variant_cpp::emit_rms_norm` to inspect the `weight_fn`
OpInstance field (position resolved by field-name lookup, not
index, so future shape revisions don't silently break) and route
to the correct vendor op template:

- `Weights::input_layernorm` → `attn_norm<Config, Globals>`
- `Weights::post_attention_layernorm` → `mlp_norm<Config, Globals>`
- `Weights::model_norm` / `final_norm` → `lm_head_norm<Config, Globals>`
- Anything else → `panic!` with the observed token string. A new
  RmsNorm role is a vendor-side problem (new op template needed);
  silent defaulting would be a correctness footgun.

`rms_norm_vendor_op(weight_fn_tokens: &str) -> &'static str` is
exported for downstream uses (e.g. future generic emit paths that
want the op name without the C++ block).

### Per-SM batch partitioning

Plan-silent question: when a codegen'd kernel runs, how is work
split across the SM grid? Vendor's `mk` uses a global atomic
instruction queue; we have no queue. Picked the minimal plan-
consistent scheme: each SM processes one batch block identified
by `blockIdx.x`. RmsNorm ops populate:
- `kvms.instruction()[1] = layer_expr` (per walker: literal or
  `__layer + baseline`)
- `kvms.instruction()[2] = 1` (num_items)
- `kvms.instruction()[3] = blockIdx.x` (local_batch_indices[0])

The kernel entry is responsible for clamping `blockIdx.x <
num_batch_blocks` before dispatching to the schedule body. That
clamp is a Phase 2e concern (wiring `emit_cu_variant`); the per-op
emitter just assumes it's already happened.

### Tests (12/12 pass)

New:
- `rms_norm_role_dispatch_by_weight_fn` — asserts each of the
  three roles emits its correct vendor op and none of the others.
- `rms_norm_unknown_weight_fn_panics` — `should_panic` on a
  fabricated weight accessor, locking in the no-silent-default
  rule.

Updated fixtures to match the real `RmsNormRefImpl::opcode_shape`
(in_slot, out_slot, layer, weight_fn) — the Phase 2c test
fixtures had used stub fields that no longer resemble what the
lowering produces.

### Commits

- `8fbf26019` — Phase 2c bucket walker.
- (pending) — Phase 2d role dispatch + blockIdx partitioning.

## 2026-05-04 — Phase 2e: emit_cu_variant top-level composer

Added `emit_cu_variant(variant_name, dims, backbone, lm_head,
shapes)` that composes a complete `.cu` source by calling:

1. `emit_prelude` — FERRITE_* defines, LLAMA_* bridge, substrate-
   only `#include`s, vendor op body includes.
2. `emit_kernel_open` — `extern "C" __launch_bounds__` signature.
3. `emit_sram_and_semaphore_init` — shared-mem layout, state<Config>
   construction, full init_semaphore block. Factored into a
   `const SRAM_AND_INIT_BLOCK: &str` so the init sequence stays
   byte-identical across entry points.
4. `emit_batch_block_clamp` — `if (blockIdx.x >= num_batch_blocks)
   return;` so an over-launched grid is safe.
5. `emit_bucket_body(backbone, shapes)` — backbone walker (with
   Loop handling).
6. `emit_bucket_body(lm_head, shapes)` — lm_head walker.
7. `emit_kernel_close` — terminal `everyone::sync(15)` + `}`.

The old hardcoded `emit_cu_phase2b` is kept but `#[deprecated]`ed
so its Phase 2b test keeps catching regressions in the
SRAM/semaphore-init code without blocking the new variant path.

### Test (13/13 pass total)

`emit_cu_variant_composes_prelude_backbone_and_lm_head` asserts:
- All FERRITE_* defines at expected values (NUM_LAYERS=16,
  HEAD_DIM=64, MATMUL_OUT_BLOCK_SIZE=128).
- Kernel signature present with correct variant name.
- SRAM/init block present.
- Batch clamp present.
- Backbone and lm_head section markers emitted with their
  counted op totals.
- Three distinct norm specializations emitted
  (`attn_norm`, `mlp_norm`, `lm_head_norm`) proving role dispatch
  works end-to-end through a composed emission.
- One `for (uint32_t __layer` loop for the 14-iter backbone body.
- No banned constructs (`g.instructions`, `::mk<`, `dispatch_op`,
  `OPS_LIST`).

### Status vs plan Phase 2 exit criterion

Plan: *"codegen'd .cu compiles for Llama-3.2-1B m=8 and launches
without crashing."*

Gap:
- Need to wire `emit_cu_variant` into `codegen.rs::emit_model` so
  it actually runs during proc-macro expansion.
- Need to write the `.cu` to the cudaforge cache (or equivalent).
- Need Rust-side extern launcher function that Rust calls.
- Need the remaining ~10 OpInstance variant → vendor op mappings
  (currently only RmsNorm is mapped; all others emit `TODO(phase2c)`
  which would fail to compile at nvcc time).
- Need pod-side nvcc compile check.

Phase 2f onward will fill each gap.

### Commits

- `cecf7e56b` — Phase 2d (RmsNorm role dispatch + blockIdx partition).
- (pending) — Phase 2e (emit_cu_variant composer).

## 2026-05-04 — Phase 2f: remaining variant emitters

Extended `variant_cpp.rs` with five new emitters covering all core
KVM-eligible variants that appear in Llama's lowered schedule:

- `FusedQkvRopeCache` → `qkv_rope_append<Config, Globals>`. Inner
  loop over `Globals::Q_COLS + Globals::KV_COLS` (Q heads first,
  then K/V heads interleaved). Populates `[1]=layer`,
  `[2]=blockIdx.x (local_row)`, `[3]=__col (local_col)`,
  `[4]=blockIdx.x (row)`, `[5]=__col (col)`.

- `FusedGateUpSiluMul` → two vendor ops in sequence: `gate_silu` +
  `up_matmul`. Each inner-looped over `Globals::intermediate_dim /
  Globals::matmul_out_block_size / Globals::num_devices` output
  blocks.

- `CutlassGemmAdd` → role dispatch via `gemm_add_vendor_op(weight_fn)`:
  `self_attn_o_proj` → `o_proj<Config, Globals>`,
  `mlp_down_proj` → `downproj<Config, Globals>`,
  anything else → panic. Inner loop over
  `Globals::num_output_blocks` (hidden_dim / matmul_out_block_size).

- `AttentionViaCache` → `attention_decode<Config, Globals>`. Single
  sequence per SM (`[2]=num_seqs*2=2`, `[3]=blockIdx.x=global_seq_idx`,
  `[4]=__kvh`). Inner loop over `Globals::num_kv_heads /
  Globals::num_devices` kv heads per device.

- `CutlassFusedAddRmsNormGemm` → two vendor ops in sequence:
  `lm_head_norm` (one norm per SM batch block) + `lm_head` (inner
  loop over `g.logits.cols() / Globals::matmul_out_block_size`
  vocab blocks).

Helpers factored:
- `warp_role_dispatch_block() -> &'static str` — the consumer /
  loader / storer / launcher switch pattern that every emitter
  appends after its instruction populate. Keeps each emitter tight.
- `field_str(field_names, field_values, name, variant)` — resolve
  an OpInstance field by name with a caller-naming panic on miss.

### Tests (20/20 pass)

New:
- `qkv_rope_emits_q_plus_kv_col_loop`
- `gate_up_silu_emits_dual_ops_with_inner_loop`
- `gemm_add_role_dispatch_by_weight_fn`
- `gemm_add_unknown_weight_fn_panics` (`should_panic`)
- `attention_decode_emits_kv_head_loop`
- `lm_head_emits_norm_then_gemm_with_vocab_loop`
- `all_variants_no_banned_constructs` — every Phase 2f emitter
  verified to NOT contain `g.instructions[`, `dispatch_op`,
  `OPS_LIST`, or `::mk<`.

Updated:
- `unmapped_variant_returns_none` — swapped
  `FusedQkvRopeCache`/`CutlassGemmAdd` (now mapped) for
  `CutlassFusedGemmBias` / `LayerNorm` / `__Unmigrated`.
- `walker_emits_todo_for_unmapped_variant` — same swap
  (`LayerNorm` now the unmapped proxy).

### Still unmapped (Phase 2g+ if they show up in a bucket)

- `CutlassGemm`, `Gemm`, `Cublas`, `CutlassGemv` — generic matmul
  Impls. Solver should prefer TK-tier Impls on sm90+ so these
  should not appear in KVM-eligible buckets; if they do, the walker
  emits TODO.
- `Embed`, `Reshape` — embed/reshape have no vendor megakernel op.
  They should not appear in a KVM-eligible bucket either.
- `Add`, `AllGather`, `AllReduce`, `FusedAddRmsNorm`, `RopeAppend`,
  `TanhSoftCap`, `ScalarMul`, MLA variants — either host-only or
  architectures the vendor megakernel does not implement.
- Quantized variants (Marlin, Bnb4, Fp8) — dense BF16 only.

Plan phase 2 exit criterion ("codegen'd .cu compiles for
Llama-3.2-1B m=8 and launches without crashing") still requires
2g/2h/2i.

### Commits

- `36d7a999f` — Phase 2e emit_cu_variant composer.
- (pending) — Phase 2f remaining variant emitters.

## 2026-05-04 — Phase 2g: emit_model hookup + cudaforge cache write

Wired the codegen pipeline end-to-end on the Ferrite side.

### interpreter/kvm.rs additions

- `KvmDims::from_bounds(bounds, num_devices, sm_count) -> Result<Self, String>`
  — extracts each dim from a `ModelParams::bounds` map (HF
  config.json field names) and validates four vendor invariants
  up front (panicking with a specific message if any is unmet):
  1. `head_dim % 32 == 0` (qkv_rope_append apply_rope_inplace).
  2. `matmul_out_block_size == 2 * head_dim` (pinned in dims).
  3. `num_kv_heads % num_devices == 0`.
  4. `kv_col_start (= num_attention_heads / 2 / num_devices)` is even
     (vendor qkv_rope_append storer alignment).
  5. `hidden_dim % (PIPELINE_K_DIM * INPUT_PIPELINE_STAGES) == 0`
     and same for `intermediate_dim / num_devices`.
- `megakernel_cache_dir() -> PathBuf` — `$XDG_CACHE_HOME/cudaforge/
  megakernels/` (or `$HOME/.cache/...` as fallback). Shared by
  proc-macro (writes) and ferrite-cuda-builder's build.rs (reads).
- `write_cu_to_cache(canonical_name, source) -> io::Result<PathBuf>`
  — creates dir + writes `ferrite_<canonical>.cu`, returns full path.

### codegen.rs hook

After `apply_loop_compression` runs on every canonical's backbone +
lm_head, `emit_model` checks `FERRITE_KVM=1` env var and, if set,
calls a local `emit_kvm_artifacts_inline(model, canonical_lowered,
arch_opcodes, tp_world_size)`.

For each `(WorkloadPoint, CanonicalLowered)`:
1. Build canonical name `<model_stem>_m_<num_tokens>_sk_<sk_bucket>`
   with non-alphanumeric chars mapped to `_`.
2. Call `emit_cu_variant(name, dims, backbone, lm_head, shapes)`.
3. Write to cache via `write_cu_to_cache`.
4. `eprintln!` status (bytes + op counts) so the macro-expansion
   log shows what landed.

If `KvmDims::from_bounds` fails (model not megakernel-eligible),
log and skip — host interpreter path takes over. No hard failure.

Gated on `FERRITE_KVM=1` env var so default builds don't write
anything. The emit_model path is otherwise unchanged.

### Not yet wired (Phase 2h)

- Rust-side `extern "C"` declarations for each canonical's
  generated kernel so Rust can dispatch calls.
- Per-canonical launch wrapper that pulls weight pointers + shape
  + rope pointers out of the Rust-side `Weights` struct and
  marshals them into the kernel arg pack.
- `ferrite-cuda-builder/build.rs` glue that nvcc-compiles the
  written `.cu` into `libmegakernels.a` and links it into
  `vllm-cuda`.

### Test status

20/20 Phase 2 unit tests still pass.
6 pre-existing failures in `config::tests` / `solver::tests` /
`impl_lib::tests` — unrelated to my changes (verified by stashing
my work and re-running; same 6 fail on stash-popped baseline too).
Upstream issue; out of scope for Phase 2g.

### Commits

- `c03ad803e` — Phase 2f remaining variant emitters.
- (pending) — Phase 2g emit_model hookup + cache write.

## 2026-05-04 — Phase 2h-1: extern-C launcher scaffold

Phase 2h ("Rust launch() runtime plumbing") split into sub-phases
since the scope is substantial:

- **2h-1 (this commit)**: C++ arg-struct + `extern "C"` launcher
  declaration with stub body. ABI boundary exists, symbol is
  linkable, no Globals construction yet.
- 2h-2: fill launcher body (construct `Globals<>` from args,
  launch the kernel on a stream).
- 2h-3: Rust-side extern decl + `pub fn launch(...)` wrapper that
  extracts pointers from the Weights struct + IO state and calls
  the launcher.

### emit_launcher_scaffold

Appended to `emit_cu_variant` output after the kernel body. Emits:

1. `struct ferrite_<name>_args { ... }` — C struct with every
   Globals field as `void*` + four scalars (`attn_scale`,
   `rms_norm_eps`, `num_pages`, `batch_size`, `num_prefill_tokens`).
   Grouped by role in the struct for readability: model weights,
   paged KV cache, rotary tables, activation buffers, scheduling
   vectors, barrier array, vm-side fields, scalars.
2. `extern "C" int ferrite_<name>_launch(const ferrite_<name>_args*
   args, cudaStream_t stream)` — Phase 2h-1 stub returns -1
   (ENOSYS). Args/stream unused for now. Signature is final for
   Rust to bind against.

Comment inside the stub body lists exactly what Phase 2h-2 must
fill in (Globals default-construct + field-by-field `gl<>`
construction from `args->*` + kernel launch with grid =
`sm_count`).

### Tests (23/23 pass)

New:
- `emit_cu_variant_includes_launcher_scaffold` — asserts the
  generated .cu contains the arg-struct, launcher decl, a
  sampling of pointer fields (qkv_weights, lm_head_weights,
  k_cache, rope_cos, hidden_states), and the `return -1` stub.
- `kvm_dims_from_bounds_llama_3_2_1b` — builds a bounds map
  matching Llama-3.2-1B config.json, asserts derived
  `matmul_out_block_size == 2 * head_dim == 128`.
- `kvm_dims_from_bounds_rejects_non_multiple_head_dim` — a
  `head_dim=48` (violates vendor's `% 32 == 0` assert) returns a
  typed error; no silent default.

### Commits

- `597bc70b9` — Phase 2g emit_model hookup + cache write.
- `6fde97dd5` — Phase 2h-1 launcher scaffold.

---

## 2026-05-04 — Phase 2h-3 — Rust extern ABI mirror

**Commit**: `b6016404d`

Rust side of the Phase 2h-1 C ABI, in new module
`ferrite-forward/src/interpreter/kvm.rs` (with `mod.rs` gating
on `feature = "cuda"` to match the rest of the crate).

### What's emitted

- `KvmLaunchArgs` — `#[repr(C)]` struct one-for-one with
  `emit_launcher_scaffold`'s emitted `ferrite_<variant>_args`.
  34 `*mut c_void` + 2 `f32` + 3 `i32`. Field order and types
  locked in lockstep with the C side.
- `KvmLaunchFn = unsafe extern "C" fn(*const KvmLaunchArgs,
  CUstream) -> i32` — the fn-pointer type every per-variant
  `extern "C"` declaration will resolve to.
- `launch(launch_fn, args, stream) -> i32` — one-line wrapper
  so the call site through any emitted extern funnels through
  one place.
- `KvmLaunchArgs::zeroed()` — nulls + zeros for incremental
  field population.

### Why fn-pointer indirection (not a hard-coded extern)

Each canonical gets a different `ferrite_<variant>_launch`
symbol name (from `{source_stem}_m_{num_tokens}_sk_{sk_bucket}`).
The proc-macro will emit one `extern "C" { fn
ferrite_<variant>_launch(...) -> i32; }` declaration per
canonical at the remaining Phase 2h sub-step, then call
`launch(ferrite_<variant>_launch, &args, stream)`. This lib
stays variant-agnostic; the generated code handles name
plumbing.

### Verification

- Pod `nick` (H100): `cargo check -p ferrite-forward --features
  cuda` compiles clean. Pre-existing warnings only; no new
  diagnostics from the added module.
- macOS: `cargo test -p ferrite-forward-macro interpreter::` —
  23/23 pass unchanged (Phase 2h-3 only adds files to
  `ferrite-forward`, proc-macro surface untouched).
- Module-local tests (`kvm_launch_args_size_matches_c_layout`,
  `kvm_launch_args_zeroed_is_all_null_zero`) ride on the
  `ferrite-forward` cuda-gated test binary; the pod build
  currently fails to link `ferrite-forward` tests due to a
  pre-existing `launch_dequantize_block_q4_0_f32` link error
  in `ferrite-kernels`, orthogonal to this change. Lib-level
  compile proves ABI is well-formed.

### Next (Phase 2h — final sub-step)

Proc-macro side: per canonical, emit

```rust
extern "C" {
    fn ferrite_<variant>_launch(
        args: *const ::ferrite_forward::interpreter::kvm::KvmLaunchArgs,
        stream: ::ferrite_cuda_core::CUstream,
    ) -> i32;
}
```

plus a call site that builds `KvmLaunchArgs` from the live
`Weights` + `ForwardCtx` and invokes it. Only then is Phase 2h
fully callable — today there's no caller wiring, just both
sides of an otherwise-unused ABI.

### Phase 2i follow-on (pod validation)

Filling in `emit_launcher_scaffold`'s stub body (default-
construct `Globals`, per-field `gl<>` init from `args->*`,
kernel launch with `grid=Globals::sm_count`, `block=Config::
NUM_THREADS`, `dynamic_smem=Config::DYNAMIC_SHARED_MEMORY`).
Requires nvcc feedback to get `pgl<>`/`gl<>` templating right.

---

## 2026-05-04 — Phase 2i — codegen'd .cu compiles on Hopper

**Commit**: `5db546071`

Attempted `nvcc -c` on the first emitted variant
(`ferrite_llama_3_2_1b_m_8_sk_128.cu`) against the vendored
thunderkittens + megakernels/cross-gpu-llama includes on pod
`nick` (H100, nvcc 12.9).

### Two defects found & fixed

1. `using Config = ferrite_config;` — the vendor config
   struct is `llama_config` (from
   `cross-gpu-llama/llama.cuh`), no such name as
   `ferrite_config`. Renamed the codegen's Config typedef +
   the "ferrite_config / globals_t<> / llama_70b_globals"
   comment.

2. Outside-a-Loop `layer_expr` emission used
   `TokenStream::to_string()`, which preserves Rust's `u32`
   literal suffix (`0u32`, `15u32`). nvcc: "extra text after
   expected end of number". Strip via `parse_u32_literal`
   before format!'ing into C++.

### Compile result

```
nvcc -c -o /tmp/ferrite_llama.o \
  /home/nickm/.cache/cudaforge/megakernels/ferrite_llama_3_2_1b_m_8_sk_128.cu \
  -I<tk> -I<mk> -I<cross-gpu-llama> \
  -DKITTENS_HOPPER -gencode=arch=compute_90a,code=sm_90a \
  -std=c++20 --expt-extended-lambda --expt-relaxed-constexpr -x cu
```

Output: 258KB object file, single ptxas performance note
(`wgmma.mma_async serialized due to pipeline crossing function
boundary` — a property of non-inlined TK op bodies, not a
correctness issue). Every previously-failing compile stage
(identifier lookup, template instantiation, PTX codegen,
ptxas) now completes clean.

### Exit criterion status

From plan: *"codegen'd .cu compiles for Llama-3.2-1B m=8 and
launches without crashing."*

- ✅ Compiles.
- ⬜ Launches — blocked on: (a) Phase 2h-2 launcher body —
  the scaffold still `return -1`s, so it cannot actually
  launch anything; (b) Rust glue that builds `KvmLaunchArgs`
  from live `Weights` + `ForwardCtx` and calls `launch()`.

### Tests

- macOS: 23/23 macro-side interpreter tests pass unchanged.
- Pod: no new unit tests — Phase 2i validation is the nvcc
  compile itself.


