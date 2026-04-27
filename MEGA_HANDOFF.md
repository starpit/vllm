# mega — handoff

> Worktree `ff-interpreter-mega`, branch `feat/rust`. This file is
> the source of truth for the megakernel work. Read it before
> changing anything. The host-interpreter pivot
> (`HANDOFF_INTERPRETER.md`) is the prerequisite and is fold-ready;
> mega builds on top of its `Instruction<W>` IR + `static
> BACKBONE_M_<N>: &[Instruction<W>]` slices without modification.

## Where we are

Phase 1 work order steps 1–5 done + DC siblings partially landed
+ step 8 trace half + cost model now picks DC on Hopper / host
on Ada (principled, not push-order). Tip
`3f700439f`. Nine commits on `feat/rust` past the host-pivot
baseline:

- `f7249aaae` — vendor `Megakernels` (throughput@`91eaff262`) +
  `ThunderKittens` (`0b55588d2`) under `vllm-rs/third_party/`.
  Megakernels LICENSE restored from upstream `main`'s tip; see
  `third_party/VENDOR.md`.
- `ef265b2a8` — `MegakernelFit::{None, Primitive, Kvm}` enum +
  defaulted `Implementation::megakernel_fit()` →
  `MegakernelFit::None`; `interpreter_codegen.rs` renamed to
  `interpreters/host.rs` with sibling stubs `prim_mega.rs` +
  `kvm_mega.rs`; `interpreters::pick_interpreter` selector +
  `TargetProfile::{prim_mega_compatible, kvm_compatible}` stubs.
- `e9c489c6e` — persistent `__global__` skeleton at
  `vllm-cuda/csrc/megakernel/prim_mega.cu`. Cooperative grid sync
  between phases (forced by existing `dc_*` early-return
  pattern); pointer table side-channel for 64-bit ptrs; arms
  wired to `dc_rms_norm` / `dc_fused_add_rms_norm` /
  `dc_fused_qkv_rope_cache` / `dc_silu_and_mul` / `dc_gemv` from
  the existing `megakernel_ops.cuh`. Build infra extension in
  `ferrite-cuda-builder/build.rs::build_megakernels` to scan the
  tree-resident dir alongside `~/.cache/cudaforge/megakernels/`.
- `a2924e765` — CUTLASS DC pattern proven for one tile. On-device
  Params construction via `CUTLASS_HOST_DEVICE` constructors
  (`gemm.h:99–135`, `threadblock_swizzle.h:64`); zero
  host-side `prepare_*_params` shim needed for non-splitK
  configs (split_k_slices=1 sidesteps the workspace branch in
  `device::Gemm::initialize`). Header at
  `vllm-cuda/csrc/dc_cutlass.cuh`.
- `9a66f06b3` — wholesale CUTLASS DC fan-out via X-macro list
  (`cutlass_gemm_configs.cuh::CUTLASS_DC_GEMM_LIST`) expanded
  three times in prim_mega.cu (typedefs + `CutlassConfig` enum +
  `run_cutlass_gemm` switch arms). 18 workhorse tiles —
  `64x64`, `64x128`, `128x64`, `128x128`, `128x256`, `256x64`,
  `256x128` at stages `s2/s3/s4`. Deep-pipeline (s5+), 32x*
  small-M, K64, W8, swizzle, silu, GEMV, splitK, sm90 not yet
  fanned out (each needs its own typedef macro shape).
- `cd9fd897c` — FI DC template at
  `vllm-cuda/csrc/dc_flashinfer.cuh`. `dc_persistent_attn<
  Runner1, Runner2, Reduction, Params>` mirrors vendor's
  `PersistentKernelTemplate` body
  (`flashinfer/attention/persistent_template.cuh:60–97`).
  Verifies: `Runner1::Run + Runner2::Run +
  cg::this_grid().sync() + Reduction::Run` are all
  `static __device__ __forceinline__` on
  `persistent.cuh:181 + 488`, directly callable from inside
  another `__global__`. `build_megakernels` now pulls the
  FlashInfer headers via the same `with_git_dependency` pin
  (`08ab45d67`) the standalone shim build uses.
- `cf918daec` + `9b4168ed4` — DC siblings landed via thin-
  delegation pattern: `DcRmsNormImpl`, `DcFusedAddRmsNormImpl`,
  `DcFusedQkvRopeCacheImpl`, `DcCutlassGemmImpl × 11 tiles` (the
  `CUTLASS_TILE_ZOO ∩ CUTLASS_DC_GEMM_LIST` intersection — see
  `dc_cutlass_zoo_is_subset_of_host_zoo` test for the invariant).
  Each delegates `matches` / `cost_us` / `fan_out` /
  `opcode_shape` / etc. to a fresh host-counterpart instance and
  overrides only the four mega-relevant methods.
- `1f4bf3192` — step 8 trace half: per-workload `pick_interpreter`
  decision printed alongside the solve summary (commandr today
  reports `Host` across every M because DC sibling coverage is
  too thin for the all-or-nothing tier semantics to flip a whole
  canonical).
- `3f700439f` — solver now picks DC siblings on Hopper, host on
  Ada, **for principled reasons** (not library push-order).
  Reverse-engineered from `worktree-ferrite-mega/f5918269f`:
  - `prim_mega_compatible()` tightened from `>= 80` to `>= 90`.
    Ada's `cg::this_grid().sync()` is ~100us (gmem-atomic spin),
    Hopper's is ~15us (L2 synchronizer + TMA fences). Ada loses
    by construction.
  - `launch_overhead_us(DeviceCallable, _) = 0us` (was
    `InKernelGridSync`). Wave-level mega-launch cost amortizes
    over N picks → ~0.5us per pick at N=10, round to zero. The
    older branch never charged grid-sync per-pick anywhere.
  - Solver effect: host pays +5us/pick, DC pays +0us/pick. DC
    wins where feasible; feasibility gates on `>= 90`.
  - Locked by `cost::tests::solver_picks_dc_siblings_on_h100`
    (asserts ≥1 DC sibling picked on H100 LLAMA_BODY at M=1) +
    `predicted_us_includes_launch_overhead` (asserts zero DC
    picks on L4).

The host interpreter at the parent commit (`8a45b39d6` or later,
after the seam swap + slot-metadata fix) is the platform.

`libmegakernels.a` builds clean (~2.0 MB at 18 CUTLASS tiles +
ferrite DC ops; FlashInfer not yet instantiated). Includes
`prim_mega_llama_kernel` + `prim_mega_llama_launch` exported
symbols.

The pivot already staked the abstractions we need:

- `crates/ferrite-forward-macro/src/impl_lib.rs:135` — `enum
  LaunchKind { HostCallback, RegularLaunch, CooperativeLaunch,
  DeviceCallable }`. Every one of today's 49 Impls returns
  `HostCallback`. Zero `DeviceCallable` in tree.
- `crates/ferrite-forward-macro/src/impl_lib.rs:459` — `fn
  launch_kind(&self) -> LaunchKind` on the `Implementation`
  trait.
- `crates/ferrite-forward-macro/src/cost.rs:17` — comment:
  "consumers (DeviceCallable impls + megakernel emission)".
- `crates/ferrite-forward-macro/src/concurrency.rs:38–44, 119–123`
  — Rule 4 stub for "DeviceCallable in same persistent kernel".
- `Handoff` cost table already values mega-shaped handoffs:
  `KernelBoundary=5us`, `Mbarrier=0.1us`, `SyncThreads=0.5us`.
  Solver naturally prefers cheaper handoffs once Impls are
  mega-fit; no new cost knob needed for Phase 1.

## Locked design

### Two phases, one IR

The host's universal `Instruction<W>` enum + per-bucket `static
BACKBONE_M_<N>` slices are the input to **both** mega backends.
No second IR. No new Instruction variants required for Phase 1.
Phase 2 may add output-tile-bound variants (see §Phase 2).

**Phase 1 — primitive megakernel.** The host interpreter ported
to CUDA. One persistent `__global__` per arch; body is a `switch`
over `[i32; 32]` opcodes; each arm calls a `__device__` fn that
wraps cutlass / flashinfer / TK. No warp specialization, no page
virtual memory, no controller/loader/storer split. Per-op
handoffs are `__syncthreads()` or grid sync. Probably flat-or-
slightly-worse than host on perf — its job is to **prove the
infra** end-to-end with the smallest possible surface.

**Phase 2 — KVM megakernel.** Vendored `~/Megakernels` template
instantiated per arch. Warp specialization (controller / loader /
consumers / storer / launcher), instruction pipelining, page
virtual memory, per-SM instruction tape (see
`include/controller/instruction_fetch.cuh:22,32` —
`get_worker_id()` indexes the second dim of the
`[1, NUM_SMS, ROWS_PER_SM, 32]` instruction tensor). Where the
perf actually comes from. Builds on Phase 1's DeviceCallable
Impls; adds KVM-specific authoring (`release_lid`, semaphore
choreography, output-tile-granular `fan_out`).

### Solver picks first; interpreter follows

Solver behavior is unchanged — it picks the cheapest feasible
Impl per claim by cost. The interpreter is selected **post-solve**
from the picked set:

```text
fn pick_interpreter(picked_impls, profile) -> Interpreter {
    if profile.kvm_compatible()
        && picked_impls.all(|i| i.megakernel_fit() == Kvm) {
        Interpreter::KvmMega
    } else if profile.prim_mega_compatible()
        && picked_impls.all(|i| i.megakernel_fit() >= Primitive) {
        Interpreter::PrimMega
    } else {
        Interpreter::Host
    }
}
```

Both axes must hold: every picked Impl fits the tier, AND the
target supports the tier. If TK kernels are slow on a workload,
solver picks cutlass HostCallback, mega isn't available, host
runs — no force-fit, no perf regression.

### One new Impl method, no other trait surface changes

```rust
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MegakernelFit { None, Primitive, Kvm }

impl Implementation {
    fn megakernel_fit(&self) -> MegakernelFit { MegakernelFit::None }
}
```

`Kvm > Primitive > None` (a KvmFit Impl is implicitly Primitive-
fit). Default `None`; new DeviceCallable wrappings return
`Primitive`; new TK / KVM-template Impls return `Kvm`.

**No** `mega_opcode`. **No** `MegaOp` struct. **No** mega-specific
methods on `Implementation`. The encoder is one match per emitter
file, in the macro. The variant-shape information needed for
encoding comes from the `OpInstance` fields `fan_out` already
emits.

### Three emitters, sibling files

```
crates/ferrite-forward-macro/src/interpreters/
    host.rs       (today's interpreter_codegen.rs body, renamed)
    prim_mega.rs  (Phase 1 emitter)
    kvm_mega.rs   (Phase 2 emitter)
    mod.rs
```

Each consumes the same lowered buckets `colored_slot_map` /
`apply_loop_compression` / `collect_boundary_inputs` already
produce. Shared utilities stay in their current location until
the second consumer reaches for them, then extract.

### Per-emitter encoder match

Each mega emitter contains exactly one exhaustive `match` over
`Instruction<W>` variants → `[i32; 32]`. Variants without an arm
mark the canonical mega-ineligible **at codegen time** (the
emitter just doesn't emit the `MEGA_PROGRAM_M_<N>` static for that
canonical). No `_` catch-all. No runtime "refused" returns.

### Loops: unroll in Phase 1, vendor a LOOP opcode later if needed

KVM's controller is a flat `for kvms.instruction_index = 0..num_iters`
loop over the per-SM tape (`include/controller/controller.cuh:24-29`).
No provision for repeating a range. So `Instruction::Loop { body,
count }` — the row the host's `apply_loop_compression` emits —
has no native KVM analog.

**Phase 1: unroll at encode time.** Mega emitters expand each
`Loop` row back out into `count` copies of `body` before
serializing to the per-SM tape. Program memory: a worst-case
decode bucket on Llama-70B is ~80 layers × ~10 ops ≈ 800 rows ×
32 ints × 4 bytes ≈ 100 KB per SM tape — trivial in device
memory.

**Phase 2 (optional): vendor a LOOP opcode.** Modify vendored
controller to support a backwards-jump opcode that adjusts
`instruction_index`. Keeps the tape compact, lets multiple
layers share one program region, and may help i-cache pressure.
Only worth doing if profiles show program-memory or fetch
overhead. Not on the critical path.

The host emitter's `apply_loop_compression` stays unchanged —
the host runtime *does* benefit from the compressed form
(`__layer + baseline` shadowing, per-row baselines). Mega
emitters consume the compressed form and re-expand on encoding.

### No mega types in `ferrite-forward` runtime

`ferrite-forward` continues to host the universal IR + host
`eval`. All mega artifacts (encoder, scheduler, launcher,
per-arch `globals` struct, vendored .cu) are macro-emitted +
vendor code. Nothing megakernel-shaped leaks into the runtime
crate.

### Vendor in-tree

Copy `~/Megakernels` and `~/ThunderKittens` under
`vllm-rs/third_party/{megakernels,thunderkittens}/` with their
LICENSE files and snapshotted upstream commit hashes. Treat as
ours to tweak. Build path runs them through `vllm-cuda`'s build
system (or a dedicated `vllm-mega-cuda` crate — TBD on first
build).

## Phase 1 — work order

The original PoC-pair-then-widen framing (steps 4 + 10) was
collapsed into a single wholesale audit per the no-piecemeal
rule (`feedback_no_piecemeal_codegen_migration`). The list below
reflects the actual landed sequence + remaining gaps.

✅ 1. Vendor Megakernels + ThunderKittens. `f7249aaae`.
✅ 2. `MegakernelFit` enum + `megakernel_fit()` trait method
   defaulting to `None`. `ef265b2a8`.
✅ 3. `interpreters/{host,prim_mega,kvm_mega}.rs` rename +
   `Interpreter` enum + `pick_interpreter` selector +
   `TargetProfile::{prim_mega_compatible, kvm_compatible}`
   stubs (currently both return `false`). `ef265b2a8`.
✅ 5. **Primitive megakernel `.cu`** at
   `vllm-cuda/csrc/megakernel/prim_mega.cu` —
   `prim_mega_llama_kernel` + `prim_mega_llama_launch` (cooperative
   launch). Arms: rms_norm / fused_add_rms_norm / qkv_rope_cache
   / silu_and_mul / gemv / cutlass_gemm. `e9c489c6e`.
🟡 4 + 10. **DC sibling audit + wholesale wrapping.** Audit
   correction landed: all CUTLASS host launchers + all FlashInfer
   `fi_run_*` configs ARE wholesale-DC-able. The earlier
   "0 eligible" finding was wrong on FlashInfer (missed the
   `static __device__ Run` methods on
   `persistent.cuh:181 + 488`).
   Done:
   - CUTLASS DC pattern (`dc_cutlass.cuh`) — `a2924e765`.
   - 18 standard tile fan-out via X-macro
     (`cutlass_gemm_configs.cuh`) — `9a66f06b3`.
   - FI DC template (`dc_flashinfer.cuh`) — `cd9fd897c`.
   Remaining:
   - CUTLASS deep-pipeline (s5+) variants, 32x* small-M,
     K64 / W8 / swizzle / silu / GEMV / splitK / splitK_K64,
     sm90 GemmUniversalAdapter. Each needs its own typedef macro
     shape; the GEMV case especially is its own beast (see the
     `cutlass_gemv_launch` shape in cutlass_standalone_gemm.cu
     line 521–574 — uses `gemm::kernel::Gemv` not
     `gemm::device::Gemm`).
   - FI DC arm in prim_mega.cu + the host-side params marshaling
     (copy `plan->params_1` / `params_2` from
     `flashinfer_shim.cu.j2`'s `FlashInferPlan` to a device-
     resident buffer; encoder emits `OP_FI_PAGED_ATTN` row with
     pt[] indices for the two Params blobs + smem offset).

⏳ 6. Encoder match in `prim_mega.rs`. Exhaustive over
   `Instruction<W>` → `[i32; 32]`; no `_` arm; per-bucket
   `MEGA_PROGRAM_M_<N>` static slice emission with `Loop`
   unrolled at encode time. Slot conventions match prim_mega.cu
   (see the per-op docstrings on `run_*` fns).
⏳ 7. Per-arch globals struct emit + launcher fn. Macro emits a
   per-arch globals carrying weight ptrs / kv-cache ptrs / output
   ptr / instruction tape ptr; launcher fn hosts the
   `prim_mega_llama_launch` extern call (already declared in
   `ferrite-kernels/src/megakernel.rs`).
🟡 8. **Wire `pick_interpreter` into codegen post-solve dispatch.**
   Trace half landed (`1f4bf3192`): `lib.rs` emits per-workload
   `interp: M=…→Host|PrimMega|KvmMega` alongside the solve
   summary. With the cost model now correctly biased
   (`3f700439f` — DC siblings strictly cheaper than host on
   sm_90+, gated out on Ada), the trace lights up DC seeds
   internally even though the canonical-level interpreter
   selection still reports `Host` everywhere because of sparse
   DC sibling coverage (every backbone has at least one non-DC
   pick → all-or-nothing `pick_interpreter` falls through).
   Remaining: emit the actual runtime path (the `if PrimMega
   then …` branch) once the encoder + launcher (steps 6 + 7)
   land and there's a real mega program to call.
🟡 — **New Impls in impl_lib.rs returning `DeviceCallable +
   Primitive` fit.** Sibling for every CUTLASS launcher we have
   a DC sibling for, every ferrite-owned DC op, every FI config.
   Done (thin-delegation pattern — host-counterpart instance
   answers `matches` / `cost_us` / `fan_out` / `opcode_shape`,
   sibling overrides only `launch_kind`, `megakernel_fit`,
   `target_compatible`, mega-internal handoffs):
   - `DcRmsNormImpl` (host: `RmsNormRefImpl`) — `cf918daec`
   - `DcFusedAddRmsNormImpl` (host: `FusedAddRmsNormImpl`)
   - `DcFusedQkvRopeCacheImpl` (host: `FusedQkvRopeCacheImpl`)
   - `DcCutlassGemmImpl` × 11 tiles via `CUTLASS_DC_TILE_ZOO`
     — intersection of `CUTLASS_TILE_ZOO` ∩
     `CUTLASS_DC_GEMM_LIST` (the host CSV-calibrated zoo and the
     C++ X-macro list); a `dc_cutlass_zoo_is_subset_of_host_zoo`
     test pins the invariant so cost-lookup never misses the CSV.
   `prim_mega_compatible()` lives at `compute_capability >= 90`
   (Hopper+). The earlier `>= 80` was speculative — Ada / Ampere
   `cg::this_grid().sync()` runs ~100us/call (gmem-atomic spin
   counters), so prim-mega's per-op handoff blows out the host's
   ~5us `KernelBoundary` by 20× and host wins by construction on
   those targets. Hopper has hw-accelerated grid sync (~15us, L2-
   synchronizer + TMA fences) — that's where the older
   `worktree-ferrite-mega` branch's "100% megakernel" demo
   actually shipped. Locked by
   `cost::tests::solver_picks_dc_siblings_on_h100` (asserts at
   least one DC sibling picked on H100 LLAMA_BODY at M=1) and
   `predicted_us_includes_launch_overhead` (asserts zero DC picks
   on L4).
   Remaining: DC siblings for `FusedQkvRopePrefillImpl`,
   `CutlassGemvImpl`, silu_mul, FlashInfer attention configs;
   plus the `s2` deep-pipeline / 32×* / W8 / sm90 CUTLASS DC
   tile fan-out (Rust + C++ X-macro both need it).
⏳ 9. End-to-end: `vllm chat unsloth/Llama-3.2-3B-Instruct
   --enforce-eager` on H100 → solver picks DeviceCallable Impls
   wherever a DC sibling exists (host pays +5us/pick, DC pays 0us
   per pick — wave-level overhead is amortized) → selector picks
   `PrimMega` for any canonical where every pick is `>=`
   Primitive-fit. Today coverage is too sparse for the all-or-
   nothing tier semantics to flip whole canonicals (commandr +
   Llama still report `Host` even on H100), so step 9 is gated on
   widening DC siblings to cover every op in at least one arch's
   forward.

## Next session — what to pick up

The cost model is principled now (`3f700439f`), so any of the
remaining Phase-1 steps unblocks something concrete. In rough
order of leverage / effort:

1. **Re-evaluate whether PrimMega is worth shipping at all.**
   The current architecture (whole-forward persistent kernel via
   opcode-table dispatch in `prim_mega.cu`) is mine, not
   precedent — the older `worktree-ferrite-mega` branch's
   `cuda_codegen.rs::generate_megakernel` emitted **one
   cooperative kernel per layer**, with kernel boundaries
   between layers. Per-layer vs whole-forward is a wash in the
   all-DC regime (this branch's tier semantics) but per-layer
   stays sane under partial coverage and avoids occupancy
   ceilings. Three options on the table:
   - Keep PrimMega, redo it per-layer (the working precedent).
   - Drop PrimMega entirely; KvmMega is the perf path on
     Hopper anyway (vendored TK templates + mbarrier). The cost
     model already correctly excludes PrimMega on Ada via the
     sm_90 gate, so PrimMega's only constituency is "Hopper
     without TK template authoring" — debatable.
   - Keep PrimMega as a forced-debug verification path
     (`FERRITE_FORCE_PRIM_MEGA=1`) for byte-equality testing
     against host before KvmMega lands.
   Decision unblocks step 6 (encoder shape depends on whether
   we're emitting a tape vs inlining per layer).

2. **DC sibling coverage widening.** Today's coverage flips
   ~10% of seeds on H100; the all-or-nothing tier semantics
   needs ~100% to actually pick PrimMega for a canonical. The
   biggest gaps:
   - `DcCutlassGemvImpl` (M=1 GEMV — its own kernel shape via
     `gemm::kernel::Gemv`)
   - `DcFusedQkvRopePrefillImpl`
   - DC siblings for `FusedGateUpSiluMul` (the load-bearing MLP
     fusion — covers every layer)
   - FlashInfer attention DC arm in `prim_mega.cu` + Params
     marshaling (decode + prefill).
   Each is mechanical (delegate to host counterpart, override 4
   methods, add the .cu arm), but FI Params marshaling is a real
   piece of work. Until coverage hits 100%, the trace stays
   `Host`-everywhere even on H100.

3. **Wave-level launch counter for cost.rs.** Match the older
   branch's `cost.rs:90–112`: walk schedule, count `1 launch`
   per mega wave + `1 launch` per non-mega host pick, multiply
   by `launch_overhead_us(HostCallback, profile)` once.
   Improves `predicted_us` accuracy without changing picks
   (DC's per-pick zero stays zero). Small commit.

4. **Encoder match in `prim_mega.rs` (step 6).** Blocked on
   decision (1) — encoder shape depends on architecture choice.
   Most of the test surface for the encoder match is in place
   (the `Instruction<W>` IR is stable, slot conventions are
   pinned by `prim_mega.cu`'s per-op docstrings).

Don't let any of these become a yak-shave. (1) is the most
load-bearing question; resolving it points the rest.

## Phase 2 — work order (later)

Authoring sequence per arch:

1. Define output-tile-granular `Instruction<W>` variants matching
   vendor's existing op shapes
   (`QKV_MatMulRopeAppend(layer, batch_start, qkv_block_idx)`,
   etc.). These coexist with today's whole-kernel-call variants;
   they're picked when the solver chooses a KvmFit Impl. Host
   `eval` arms for them either panic (mega-only) or loop over
   tiles (preserves universal-IR principle but adds host-side
   work). Default to mega-only — same canonical compiled
   per-eligibility is already handled by the host/prim_mega/
   kvm_mega split.
2. Author KvmFit Impls — output-tile-granular `fan_out`,
   `release_lid` order, TK-shaped consumer/loader/storer
   templates on the .cu side, opcode-pack-position fixed in
   per-arch `mk<config, globals_arch, NoOp, ops...>`
   instantiation.
3. `interpreters/kvm_mega.rs`: encoder match for the new
   variants, scheduler (round-robin SM assignment first; smarter
   later), per-arch globals + `mk` instantiation, launcher.
4. PyVM differential testing using vendor's `python_vm.py` —
   diff the megakernel output against the PyVM trace per-row,
   per-instruction-stage. Vendor's pattern.
5. End-to-end: `vllm chat` Llama on H100, byte-equal to host /
   prim_mega.

## Known-broken: vendor's TP=8 hardcoding

The vendored cross-GPU Llama path is hard-wired to TP=8 and will
need de-hardcoding before KvmMega ships on anything other than an
8-GPU node. Concrete sites in
`vllm-rs/third_party/megakernels/`:

- `demos/cross-gpu-llama/llama.cuh:111` —
  `constexpr static int num_devices = 8;` on the Globals struct
  (compile-time constant, threads through every kernel that takes
  `Globals`).
- `demos/cross-gpu-llama/qkv_rope_append.cu:188, 287` — two
  `static_assert(Globals::num_devices == 8, "Fix this function.")`
  markers; vendor flagged the algorithmic dependency on
  `num_devices == 8` themselves.
- Python harness defaults: `megakernels/scripts/tp_generate.py:48`,
  `tp_generate_pyvm.py:23`, `tp_diff_test.py:47`,
  `demos/tp_throughput/{bench_cpp_scheduling.py:17,
  test_cpp_scheduling.py:38,144}`.

Resolution path (Phase 2 work, not Phase 1):

1. Make `num_devices` a `KvmMega` per-arch template parameter
   threaded through `globals_arch` rather than baked at the demo
   level. Ferrite emits the instantiation per the active TP
   degree it already tracks (the host pivot's TP plan in
   `project_tp_design_notes.md`).
2. Audit and rewrite the two `qkv_rope_append.cu` arms the vendor
   `static_assert`'d as 8-only. These almost certainly encode a
   head-shard / shuffle pattern that's hand-unrolled for 8.
3. Update `python_vm.py` + the TP scripts to take `num_devices`
   from the same source so PyVM differential testing keeps working
   at TP≠8.

PrimMega is unaffected — the cross-GPU demo paths are only entered
once we instantiate the KVM template.

## Cost-metric refinements (post-Phase 2)

Once the basics work, the solver cost model needs to grow to
properly value mega benefits:

- Persistent-SM cache reuse across instructions (vendor's design
  point — currently invisible to ferrite's CSV-driven cost).
- Cluster-block + DSMEM advantages (sm≥90).
- Wave-level launch counting. Today `launch_overhead_us` zeros
  out per-pick overhead for `DeviceCallable` (the older
  `worktree-ferrite-mega` branch did the same), so a chain of N
  DC ops contributes zero launch overhead even though it's one
  cooperative `cudaLaunchCooperativeKernel` (~5us). The
  approximation is fine for picks (~0.5us per pick at N=10) but
  under-counts `predicted_us`. Match the older branch's
  `cost.rs:90–112`: walk the schedule, count `1 launch` per mega
  wave + `1 launch` per non-mega host pick, multiply by
  `launch_overhead_us` once. Comes online when the per-wave
  scheduler hook (`Wave::is_megakernel`) lands.
- Per-op `Handoff::InKernelGridSync` charge inside a mega wave.
  100us/call on Ada, 15us on Hopper (target-aware in
  `Handoff::cost_us`). Currently uncharged anywhere — fine on
  Hopper where 9 syncs/layer × 32 layers ≈ 4.3ms is small
  relative to the work, less fine if PrimMega ever returns to
  Ada via a per-wave restructure that keeps grid-sync chains
  short.
- KvmFit DC ops should pay [`Handoff::Mbarrier`] (~0.1us) rather
  than the prim-mega grid-sync when running inside a KVM
  megakernel. The wave-level term should split on
  `MegakernelFit` once KvmFit impls appear.
- Per-SM tape length imbalance penalty (idle SMs at end of
  bucket).

What already lands: per-launch overhead is summed into both the
solver's DP candidate cost AND `loop_cost_us` aggregation.
HostCallback / RegularLaunch / CooperativeLaunch each pay one
`KernelBoundary` per pick (~5us — target-agnostic). DeviceCallable
pays zero per pick (wave-level overhead amortized — see
"Wave-level launch counting" above). `Handoff::InKernelGridSync` is
target-aware (100us on Ada, 15us on Hopper) for future consumers
but isn't yet summed anywhere. The combination makes DC strictly
cheaper than host at every seed where DC is feasible; feasibility
gates on `prim_mega_compatible() >= 90`, so on Ada DC is filtered
out at `target_compatible()` and on Hopper it's picked. Verified
by `cost::tests::solver_picks_dc_siblings_on_h100` and
`predicted_us_includes_launch_overhead`.

Don't bolt the rest on before Phase 2 lands.

## Things-that-must-not-happen

- **No** `mega_opcode` / `MegaOp` struct on `Implementation`.
  The encoder is one match per emitter file. Variant payload
  comes from `OpInstance` fields `fan_out` already emits.
- **No** mega wire types or encoder logic in `ferrite-forward`
  runtime. Macro-emitted launcher + vendored .cu only.
- **No** universal opcode registry on the Rust side. Per-arch
  `ops...` pack on the .cu side closes the opcode space
  template-side.
- **No** `_` arm in any encoder match. Missing arms = canonical
  ineligible at codegen, not runtime.
- **No** "compile for mega" target flag forcing solver picks.
  Solver picks by cost; interpreter selection is post-hoc.
- **No** runtime fallback ("try mega, fall back if it fails").
  Eligibility is decided once at startup per canonical.
- **No** piecemeal Impl migration. DeviceCallable wrapping audit
  is wholesale per kernel: a kernel either has both wrappings or
  the HostCallback one only. (Mirrors host pivot's wholesale
  rule.)
- **No** TK-only assumption. CUTLASS device-side and FlashInfer
  device-callable are equally valid sources of DeviceCallable
  Impls. Pick whatever's easiest to wrap first.
- **No** authoring `.cu` templates for ops Phase 1 already covers
  via DeviceCallable cutlass / flashinfer wrappings. Phase 2
  only adds template authoring where output-tile granularity
  buys real perf.

## Reference points

### Vendor

- `~/Megakernels/include/megakernel.cuh` — template entrypoint,
  `mk<config, globals, ops...>` `__global__` and warp dispatch.
- `~/Megakernels/include/config.cuh` — `INSTRUCTION_WIDTH = 32`,
  `NUM_CONSUMER_WARPS = 16`, page size, semaphore count, etc.
- `~/Megakernels/include/controller/instruction_fetch.cuh` —
  per-SM tape indexing via `get_worker_id()`. Termination
  signal: `instruction[0] == -1`.
- `~/Megakernels/megakernels/demos/throughput/instructions.py` —
  vendor's existing Llama op set
  (`PreAttnLayerNorm`, `QKV_MatMulRopeAppend`, `AttentionDecode`,
  `O_ProjResidual`, `PreMLP_Norm`, `GateSilu`, `UpMatMul`,
  `DownProjResidual`, `PreLMHeadRMS`, `LM_Head`).
- `~/Megakernels/megakernels/python_vm.py` — PyVM ground-truth
  reference for differential testing.
- `~/Megakernels/megakernels/scheduler.py` — vendor's per-SM
  assignment logic; the Rust scheduler will mirror it.
- `~/ThunderKittens/kernels/{attention,gemm}/` — H100/B200/B300
  TK kernels. Ampere unsupported as of TK 2.0 — the sm≥90 floor
  is TK's, not arbitrary.

### Tree

- `crates/ferrite-forward/src/instr.rs` — `Instruction<W>` IR.
  Encoder reads from here; nothing mega-shaped lands here.
- `crates/ferrite-forward-macro/src/impl_lib.rs:135` —
  `LaunchKind` (existing).
- `crates/ferrite-forward-macro/src/impl_lib.rs:459` —
  `launch_kind()` trait method (existing).
- `crates/ferrite-forward-macro/src/concurrency.rs:38-44,
  119-123` — Rule 4 stub. Activated by Phase 2.
- `crates/ferrite-forward-macro/src/interpreter_codegen.rs` —
  host emitter, sibling reference for prim_mega + kvm_mega.
- `crates/ferrite-forward-macro/src/target.rs` — `TargetProfile`
  including `compute_capability`. Add `kvm_compatible(&self)`
  and `prim_mega_compatible(&self)` methods here.

## Pre-commit checklist

Re-read this file. Verify:

- [ ] `cargo build -p ferrite-forward-macro` clean.
- [ ] `cargo test -p ferrite-forward-macro --lib` green
      (host-interpreter test count baseline + any new mega
      tests).
- [ ] `cargo build -p ferrite-models --features cuda` clean.
- [ ] `cargo fmt` clean and `cargo clippy --all-targets
      -D warnings` clean on every touched crate.
- [ ] If a Phase 1 commit: `vllm chat
      unsloth/Llama-3.2-3B-Instruct --prompt "why is the sky
      blue" --enforce-eager` produces coherent output AND
      `pick_interpreter` returns `PrimMega` for at least one
      bucket of the canonical. Stderr-trace the selection.
- [ ] No new methods on `Implementation` beyond
      `megakernel_fit()`.
- [ ] No `_` arm or `unsafe { unreachable_unchecked() }` in any
      mega encoder match.
- [ ] No mega wire types in `ferrite-forward`.
- [ ] If you touched the host emitter while doing mega work:
      every host pre-commit check from `HANDOFF_INTERPRETER.md`
      §Pre-commit also passes (curated golden subset, no
      regression).
- [ ] Vendor sources in `vllm-rs/third_party/{megakernels,
      thunderkittens}/` carry their LICENSE + a `VENDOR.md` with
      upstream commit hash + import date.
