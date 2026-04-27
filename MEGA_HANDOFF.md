# mega — handoff

> Worktree `ff-interpreter-mega`, branch `feat/rust`. This file is
> the source of truth for the megakernel work. Read it before
> changing anything. The host-interpreter pivot
> (`HANDOFF_INTERPRETER.md`) is the prerequisite and is fold-ready;
> mega builds on top of its `Instruction<W>` IR + `static
> BACKBONE_M_<N>: &[Instruction<W>]` slices without modification.

## Where we are

Phase 1 work order steps 1–5 done + DC siblings partially landed.
Six commits on `feat/rust` past the host-pivot baseline:

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
⏳ 8. Wire `pick_interpreter` into codegen post-solve dispatch.
   In codegen.rs, after the solver picks per canonical, call
   `interpreters::pick_interpreter(&picked, &profile)` and emit
   the appropriate runtime path. Stderr-trace the decision.
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
   `prim_mega_compatible()` bumped from stub-`false` to
   `compute_capability >= 80` so the siblings are solver-feasible
   on Ada / Hopper.
   Remaining: DC siblings for `FusedQkvRopePrefillImpl`,
   `CutlassGemvImpl`, silu_mul, FlashInfer attention configs;
   plus the `s2` deep-pipeline / 32×* / W8 / sm90 CUTLASS DC
   tile fan-out (Rust + C++ X-macro both need it).
⏳ 9. End-to-end: `vllm chat unsloth/Llama-3.2-3B-Instruct
   --enforce-eager` → solver picks DeviceCallable Impls (because
   tiebreak prefers them at equal cost) → selector picks
   `PrimMega` → coherent output, byte-equal to host run.

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
- Reduced launch-overhead amortization at very small M (today's
  `Handoff::KernelBoundary=5us` is a placeholder).
- Per-SM tape length imbalance penalty (idle SMs at end of
  bucket).

Don't bolt these on before Phase 2 lands.

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
