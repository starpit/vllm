# Ferrite Fusion Synthesis Plan (Metal first, CUDA-portable)

This is the architectural plan for **compiler-driven megakernel synthesis**
using a backend-generic atom abstraction over per-backend primitive
libraries (MetalKittens on Metal, ThunderKittens on CUDA). Replaces the
hand-coded fusion pattern (P3/P5/P6) with a new compiler pass that walks
the DAG, identifies fuseable atom chains, and emits per-backend kernels
by stitching primitive calls at macro-expansion time.

**Naming discipline.** Backend-generic structs / traits / passes use
neutral names (`Atom`, `FusePass`, `SynthesisOutput`, …). Per-backend
shims and primitive libraries are named after their backend
(`MetalAtomAdapter`, `metal_kittens.h`; eventually
`CudaAtomAdapter`, `thunderkittens.cuh`). The fuse pass itself, the
atom trait, the `Instruction::SynthesizedKernel` variant, the
symbol-naming logic, and the FUF walk are all backend-neutral so the
CUDA path is a plug-in of:
- A `ThunderKittens` primitive library.
- An `emit_cuda_body` impl per atom.
- A nvcc-driven build.rs hook.

## What exists today

`ferrite-forward-macro` is a real compiler. Pipeline (per `lib.rs`):

```
parse → classify → shape-infer → CFG → unroll (FUF, the DAG)
  → solve → schedule → codegen → emit Instruction<W> stream
```

- **FUF**: tile-level DAG. Each `FufNode` is one op (`Gemm`, `RmsNorm`,
  `RopeAppend`, …).
- **Solver**: per-bucket SFUF view; assigns one `Implementation` per
  claim. Claims can span multiple tiles (existing pattern for
  `FusedAddRmsNormImpl` claims 2 tiles, `FusedQkvRopeCacheImpl` claims 4).
- **Codegen**: emits `Instruction<W>` per claim via the Impl's `fan_out`.
  Each variant of `Instruction` maps to ONE hand-written shader via the
  metal `lowering` pass.

The current "fusion" pattern (P3/P5/P6) bolted a hand-written kernel +
hand-written matcher per fusion shape. That violates
`feedback_no_handcoded_fusion`. The right architecture is for the macro
to **synthesize** the kernel from atomic primitives.

## What this plan adds

A new compiler pass — **fuse-into-megakernels** — between solver and codegen:

```
parse → classify → shape-infer → CFG → unroll (FUF)
  → solve → fuse-into-megakernels (NEW)
  → schedule → codegen
```

The pass walks per-bucket SFUF claim assignments, groups adjacent
**atoms** whose data flow stays in registers / threadgroup memory, and
emits a single Metal source file per group. Generated kernels are
compiled by `build.rs` like any hand-written shader.

## Atom abstraction

A new trait sits beside `Implementation`. **Backend-generic surface
with per-backend emit hooks.**

```rust
// crates/ferrite-forward-macro/src/atom.rs  — backend-neutral location
trait Atom {
    fn kind(&self) -> AtomKind;
    fn signature(&self) -> AtomSignature;       // in/out channels
    fn dispatch_shape(&self, &AtomCtx) -> DispatchShape;
    fn fuseability(&self) -> Fuseability;       // None / WithSameDispatch / Standalone

    // Per-backend emit hooks. Each returns the right primitive-call
    // snippet in the right shader language for that backend. Defaults
    // to `None` so backends opt in incrementally.
    fn emit_metal_body(&self, &AtomCtx) -> Option<TokenStream> { None }
    fn emit_cuda_body (&self, &AtomCtx) -> Option<TokenStream> { None }
}
```

Atom emission stays in primitive-library terms. `emit_metal_body`
returns code that calls `mk_qmv_fast` / `mk_tg_rmsnorm_scale` /
`mk_silu`. `emit_cuda_body` returns code that calls `tk_qmv_fast` /
`tk_rmsnorm_scale` / `tk_silu`. Each is in its target's shader
language but logically parallel.

Each existing `Implementation` either:
- Provides an `Atom` view alongside its `fan_out` (preferred —
  decoder body ops), or
- Stays monolithic, declaring itself non-fuseable (attention, MPS gemm,
  cuBLAS / cutlass calls).

`MetalAtomAdapter` / `CudaAtomAdapter` are the per-backend shim layers
that bind atom emissions to the existing per-Impl machinery. The
adapters themselves are thin — the per-Impl atom data lives on the
generic `Atom` impl.

## Fuse pass

Backend-neutral. Lives in `crates/ferrite-forward-macro/src/fuse_pass.rs`.

```rust
// Parameterized on target so the atomicity-wall set and the emit-body
// choice are per-backend, but the FUF walk + grouping logic is shared.
pub fn fuse_into_megakernels(
    target: Backend,
    sfufs: &WorkloadAssignments,
    fuf: &Fuf,
) -> SynthesisOutput {
    // For each per-bucket SFUF:
    //   1. Walk claims in topo order.
    //   2. Build maximal groups of consecutive atom-tagged claims that:
    //      a. Share dispatch_shape (TG/CTA dims, threads).
    //      b. Communicate only via register/SMem (or threadgroup-mem
    //         on Metal) channels.
    //      c. Don't cross an `atomicity_walls(target)` boundary.
    //   3. Each group → one SynthesizedKernel emission unit.
    //   4. Non-grouped claims fall through to today's per-Instruction
    //      emission (attention kernels, MPS gemm / cuBLAS, etc.).
}
```

### Atomicity walls (backend-dependent)

```rust
fn atomicity_walls(target: Backend) -> AtomKindSet { … }
```

Things that cannot fuse with neighbors on **either** backend:
- AllReduce / AllGather — host-coordinated.
- `Embed` gather — sparse pattern.

Things that wall on **Metal**:
- `AttentionViaCache` / `AttentionPrefillSdpaPaged` — different
  dispatch shape (per-`(batch, head)`), reads device-resident paged
  KV. No public Apple `simdgroup_event` / async tile copy → can't
  pipeline tile-block work across the attention boundary.
- MPS GEMM — opaque.

Things that wall on **CUDA** (initial expectation):
- Attention — same shape mismatch story.
- cuBLAS / cuBLASLt GEMM — opaque.

Things that **do not wall** on CUDA but **do** on Metal:
- Tile-block pipelined fusion across attention via TK persistent
  kernels + `cp.async` / TMA. On CUDA the fuse pass can in principle
  fold attention into a per-layer megakernel; on Metal it can't until
  Apple ships the missing primitives.

Per-layer body partitioning (Metal, initial target):
- **Pre-attention chunk**: `FusedAddRmsNorm + Q + K + V + RopeAppend +
  KV cache write` → 1 synthesized kernel.
- **Attention**: stays its own kernel.
- **Post-attention chunk**: `o_proj + FusedAddRmsNorm + gate + up +
  silu + mul + down` → 1 synthesized kernel.

Per-layer kernel count drops from ~10 to 3 on Metal. CUDA eventually
collapses further with TK persistent megakernels.

## Kernel emission

For each fuse group, the **backend-neutral** pieces:

1. **Symbol name**: deterministic hash of `(atom sequence, dtype,
   group_size, shape constants)`, **plus a backend tag prefix**
   (`mk_` / `tk_`). Structurally identical groups across layers share
   one symbol within a backend.
2. **Channel binding**: producer atom's `out` channel becomes consumer
   atom's `in` channel — variables in the synthesized kernel scope.
   The fuse pass threads channels in the AtomCtx passed to each
   `emit_*_body` call.
3. **Function constants** for shape parameters (HIDDEN, NUM_Q_HEADS,
   HEAD_DIM, etc.). Same constant indices across backends so the
   binding layer is shared.

**Backend-specific** emission:

```
Metal:                          CUDA:
OUT_DIR/synthesized_kernels/    OUT_DIR/synthesized_kernels/
  <symbol>.metal                  <symbol>.cu
  #include "metal_kittens.h"      #include "thunderkittens.cuh"
  [[kernel]] void <symbol>(...)   __global__ void <symbol>(...)
  body = mk_*-calls               body = tk_*-calls
```

Same atom sequence on both sides — the emit function is what diverges.

## Build integration

- `ferrite-metal-kernels/build.rs` already globs `shaders/*.metal`. Add
  glob for `<OUT_DIR>/synthesized_kernels/*.metal`.
- Future `ferrite-cuda-kernels/build.rs`: add glob for
  `<OUT_DIR>/synthesized_kernels/*.cu` driven through `nvcc`.
- Generated `.metallib` / `.cubin` blobs register through the same
  `shader_cache` / pipeline-cache machinery (auto-discovered from
  `OUT_DIR`) on each backend.

## Instruction shape

A new backend-neutral Instruction variant:

```rust
Instruction::SynthesizedKernel {
    symbol_id: SymbolId,                    // unique per generated kernel
    bindings: &'static [BindingSpec<W>],    // arena slots, weights, runtime
    dispatch: DispatchShape,
    constants: &'static [ConstantValue],
}
```

`SymbolId` resolves to the right backend symbol at lowering time
(`mk_<...>` for Metal, `tk_<...>` for CUDA — same fuse pass picks
which based on `target`).

Today's per-fusion `Instruction` variants (`FusedQkvRopeCache`,
`FusedAddRmsNorm`, …) get retired as their hand-written kernels are
replaced by synthesized equivalents. Eventually `Instruction` has very
few non-`SynthesizedKernel` variants — just the atomicity-wall
kernels.

## Cross-backend overlap audit

| component                              | location                                    | backend-generic? |
|---|---|---|
| `Atom` trait + `AtomKind` enum         | `ferrite-forward-macro/src/atom.rs`         | ✅ |
| `AtomSignature`, `AtomCtx`, channels   | `ferrite-forward-macro/src/atom.rs`         | ✅ |
| `fuse_into_megakernels` pass           | `ferrite-forward-macro/src/fuse_pass.rs`    | ✅ |
| `atomicity_walls(target)`              | `ferrite-forward-macro/src/fuse_pass.rs`    | ✅ (parameterized) |
| `SynthesisOutput` + symbol-naming      | `ferrite-forward-macro/src/fuse_pass.rs`    | ✅ |
| Per-Impl `Atom` impls (RmsNorm, etc.)  | beside each `Implementation` impl           | ✅ surface, ⚠️ emit body per backend |
| `Instruction::SynthesizedKernel`       | `ferrite-forward/src/instr.rs`              | ✅ |
| Lowering arm for SynthesizedKernel     | `ferrite-forward/src/interpreter/*/lowering.rs` | per-backend (Metal first, CUDA later) |
| `metal_kittens.h` primitive library    | `ferrite-metal-kernels/shaders/`            | Metal-only |
| `thunderkittens.cuh` primitive library | `ferrite-cuda-kernels/include/` (future)    | CUDA-only |
| `emit_metal_body` per atom             | beside the per-Impl atom decl               | Metal-only |
| `emit_cuda_body` per atom              | beside the per-Impl atom decl               | CUDA-only |
| build.rs synthesized-kernel glob       | `ferrite-metal-kernels/build.rs`            | per-backend (mirror in cuda) |
| `MetalAtomAdapter` / `CudaAtomAdapter` | per-backend metal/ / cuda/                  | per-backend (thin wiring) |

**Rule of thumb for naming.** If a struct / fn ever calls
`emit_metal_body` directly OR references `metal_kittens.h`, it's
allowed a `Metal*` prefix. If it works against the generic `Atom` trait
and never names a backend, it's neutrally named — even if it currently
only has Metal callers.

Hand-written shaders in `ferrite-metal-kernels/shaders/*.metal` stay
Metal-named (they ARE Metal-only). The synthesis pass and atom
abstraction don't.

## MK primitive inventory

### Already in `metal_kittens.h` (✅):
- `mk_get_pack_factor`, `mk_get_bytes_per_pack` — int4 pack helpers.
- `mk_load_vector` (device + threadgroup overloads) — strided load
  with bit-4 pre-shift packing.
- `mk_qdot` — quantized dot-product (per-thread slice).
- `mk_qmv_fast` — cooperative simdgroup int4 GEMV (4 outputs/simdgroup).
- `mk_qmv_fast_to_smem` — same + smem writeback.
- `mk_sync` — TG barrier.
- `mk_tg_sum`, `mk_tg_rmsnorm_scale` — TG-wide reductions.

### Missing — decode atoms (small, ~50-100 LoC each):
- `mk_silu(x: float) -> float` — pull existing inline out into header.
- `mk_rope_pair(x0, x1, cos, sin) -> (x0', x1')` — pair rotation.
- `mk_paged_kv_write_row(...)` — store rotated K / unrotated V to paged
  cache.
- `mk_residual_add_load_sumsq(...)` — combined load(residual + delta) +
  accumulate-sumsq + write residual_new to TG memory.
- `mk_apply_rms_scale(x, scale, weight) -> x'` — per-element rmsnorm
  output.

### Missing — prefill atoms (bigger, ~200-500 LoC each):
- `mk_load_tile<T, R, C>(dst_simdgroup_matrix, src_device, stride)` —
  cooperative load of a `simdgroup_matrix_storage<T, R, C>` tile.
- `mk_store_tile<T, R, C>(dst_device, src_simdgroup_matrix, stride)`.
- `mk_mma<...>(c_tile, a_tile, b_tile)` — wrap
  `simdgroup_multiply_accumulate`.
- `mk_qload_tile<gs, bits>(dst_simdgroup_matrix<bf16, 8, 8>, packed_dev,
  scales_dev, biases_dev, row_base, k_base)` — int4 → bf16 dequant +
  cooperative tile load.

### Missing — attention atom (ceiling work):
- `mk_simd_online_softmax(scores, m_prev, l_prev) -> (probs, m_new,
  l_new)` — online-softmax accumulator step.
- `mk_kv_paged_block_load_tile<R, C>(cache_dev, block_table, t,
  kv_head, k, dst_tile)` — paged KV tile load.

With these, attention itself becomes an atom group that can fuse with
its surrounding o_proj / etc.

## Phased delivery

### Phase 0 — Revert (user has authorized)

Drop P3 / P5 / P6 commits — they cement the hand-coded fusion pattern
this plan replaces. MK foundation (`70c879cdf`) + plan commits stay.

### Phase 1 — Decode atoms in MK

Add the 5 missing decode primitives to `metal_kittens.h`. No
compiler / behavior changes. Just primitive additions, verified by
the synthesized kernels in later phases.

### Phase 2 — Atom trait + adapters

Define backend-neutral `Atom` trait in
`crates/ferrite-forward-macro/src/atom.rs`. Add atom views for
existing decoder Impls:
- `MetalRmsNormImpl` → atom emits `mk_tg_rmsnorm_scale + mk_apply_rms_scale`.
- `MetalAffineQmmImpl` (decode branch) → atom emits `mk_qmv_fast`.
- `MetalRopeAppendImpl` → atom emits `mk_rope_pair + mk_paged_kv_write_row`.
- `MetalSiluMulImpl` → atom emits `mk_silu + multiply`.
- `MetalAddImpl` → atom emits residual add inline.

(The per-Impl wrappers stay `Metal*Impl` because they ARE Metal-only
adapters; the `Atom` they expose is in the neutral trait.)

`emit_cuda_body` defaults to `None` on every atom for now. CUDA
adapters land in a later phase without touching this layer.

No fuse pass yet — just the atom layer. Existing per-Instruction
codegen still runs. Verify it stays equivalent to today.

### Phase 3 — Minimal fuse pass: pre-attention chunk only

Implement the fuse-into-megakernels pass. Initially fuses ONLY the
(AddRmsNorm + QKV + RoPE + cache) chain. Other claims fall through
to today's emission.

Emits one synthesized kernel per layer's pre-attention chunk. Validate
end-to-end on Llama-3.2-{1B,3B}-Instruct-4bit. No tok/s regression vs
the today's unfused-decoder baseline (M4 AND M1 Max).

### Phase 4 — Post-attention chunk

Extend fuse pass to the (o_proj + AddRmsNorm + gate + up + silu + mul
+ down) chain.

### Phase 5 — Prefill MK primitives

Add `mk_mma`, `mk_load_tile`, `mk_store_tile`, `mk_qload_tile`.

**M4+ NAX path (open TODO).** On Apple Family 9+ (M4 / A18 Pro+,
macOS 26.2+ runtime + arch_gen ≥ 17 — refined gate per
INT4_PARITY_PROBES.md), `mk_mma` should select `MetalPerformancePrimitives.matmul2d` +
`cooperative_tensor` (the hardware MMA path MLX uses on M4). On
M1–M3 fall back to the simdgroup-scalar tile-MMA implementation —
same dual-path strategy MLX uses. Largest single perf lever on M4
(accounts for most of the ~16% gap to `mlx_lm.generate`). See
`feedback_mpp_confirmed` for the verified MPP `matmul2d` +
`cooperative_tensor` surface. Coordinates with INT4_PARITY_PLAN.md
P7 (the NAX qmm_t / qmm_n integration).

### Phase 6 — Prefill fuse pass

Extend fuse pass to prefill claims. Synthesizes qmm_t-based
(AddRmsNorm + QKV + RoPE) and (o_proj + AddRmsNorm + gate+up + silu +
mul + down) kernels using the tile primitives.

### Phase 7 — Retire hand-written shaders

Drop `shaders/fused_*.metal` once synthesized equivalents pass
parity. Cleaner codebase.

### Phase 8 — Attention atom (eventual, Metal)

Add the attention-specific primitives so attention itself can be
expressed as an atom group. On Metal, fold attention with its
o_proj epilogue when possible. Per-layer kernel count 3 → 2 on
Metal.

### Phase 9+ — CUDA portability (separate track)

Once the Metal path is solid:
1. Author `thunderkittens.cuh` equivalents of the MK primitive
   shelf (`tk_qmv_fast`, `tk_tg_rmsnorm_scale`, `tk_silu`,
   `tk_rope_pair`, `tk_mma`, `tk_load_tile`, `tk_store_tile`,
   `tk_qload_tile`).
2. Add `emit_cuda_body` implementations on each `Atom` impl. Same
   atom logic; emits CUDA-side primitive calls.
3. Add `<OUT_DIR>/synthesized_kernels/*.cu` glob to the CUDA build
   crate (`ferrite-cuda-kernels` or equivalent) driven through
   `nvcc`.
4. Add a `CudaAtomAdapter` shim mirroring `MetalAtomAdapter`.
5. Wire the CUDA lowering arm for `Instruction::SynthesizedKernel`.
6. Loosen atomicity walls on CUDA where TK persistent megakernels
   make attention fuseable with surrounding work.

The fuse pass itself doesn't change. Per-backend additions land
behind the existing `cuda` / `metal` cfg flags.

## Validation invariants

Each phase requires:
- Coherent output on `vllm chat --device metal` for the int4 Llama
  models.
- No tok/s regression on **both** M4 and M1 Max.
- Generated `.metal` files reviewable in `OUT_DIR/synthesized_kernels/`.
- `vllm ferrite info` shows the expected dispatch structure.

## Orthogonal runtime TODO: MTL4 command-encoding migration

The synthesis pipeline is independent of the command-encoding model
used to dispatch the synthesized kernels. Today the pool encodes via
MTLCommandBuffer + a per-step ICB on a Serial encoder — a hack to
serialize Apple's `ConcurrentDispatch`-only compute-ICB API.

`objc2-metal 0.3.2` exposes `MTL4CommandBuffer`, `MTL4CommandQueue`,
`MTL4CommandAllocator`, `MTL4ComputeCommandEncoder`, and
`MTL4ArgumentTable`. Migration would:

- Replace ICB-on-Serial-encoder with native MTL4 compute sequencing
  (compute dependencies are first-class in MTL4, no Concurrent-only
  workaround).
- Replace ~18 `setBuffer` calls per dispatch with one `setArgumentTable`
  bind built once at warmup. Real CPU encoder savings on top of the
  ICB savings already shipped.
- Potentially fix the open M1 Max `MTLCommandBufferStatus(5)` failure
  — the ICB hack is the most likely M1 trigger and MTL4 sidesteps
  the whole `ConcurrentDispatch` constraint.

**OS-gated (macOS 15+ / 26+); no hardware exclusivity** — works on
M1 Max through M4. Composes with the M4+ NAX path: MTL4 argument
tables are also the preferred binding model for MPP-backed kernels.

Tracked alongside INT4_PARITY_PLAN.md "open TODOs" section.
Scope-A experiment: side-by-side on the m=1..2 decode bucket, decision
point after measured A/B vs the MTL3 path. ≥5% win → full migration.

## Build-speed notes (for iteration)

Macro is heavy. For decode-chain iteration, restrict to the target
model:

```
FERRITE_MODELS=llama-3.2-3b,llama-3.2-3b-mlx-affine-b4-g64 \
  cargo build --release -Fmetal --bin vllm
```

Drops build time from ~45s → ~20s.

## Risks / open questions

- **Channel typing**: atoms need to agree on whether their I/O lives in
  registers or threadgroup memory. May need explicit channel kinds.
- **Per-chip tuning**: the synthesized kernel still has hardcoded
  threadgroup-shape choices baked in. Per-chip variants would either
  require chip-gated codegen branches OR runtime function-constant
  parameterization of the synthesized shaders.
- **Symbol explosion**: per-(arch, bucket, dtype) emission risks
  many generated kernels. Mitigated by structural hashing — only
  unique atom-sequences get unique symbols, even across layers.
- **MMA tile shapes**: Apple GPU supports 8×8 bf16 tiles; need to
  verify all int4 prefill shapes tile-decompose cleanly (HEAD_DIM=64,
  128, plus odd sizes from group_size constraints).
- **MK_ROWS_PER_SIMDGROUP=4 vs other tunings**: current `mk_qmv_fast`
  hardcodes 4 rows × 2 simdgroups per TG. The synthesis pass might
  need an `mk_qmv_fast_<rows, simdgroups>` family so the pass can
  pick tile dims per shape / per chip.

## Cross-backend portability summary

**Backend-generic (~80% of fusion infra):**
- The fuse pass itself (FUF walk, subgraph grouping, dispatch-shape
  compatibility check, channel binding, symbol naming, atomicity
  wall framework).
- The `Atom` trait, `AtomKind`, `AtomSignature`, `AtomCtx`,
  `Fuseability`.
- Atom adapters for each `Implementation` (same per-Impl shape,
  declares atom kind + signature once, emits per-backend bodies).
- `Instruction::SynthesizedKernel` variant.

**Per-backend (~20%):**
- `metal_kittens.h` vs `thunderkittens.cuh` primitive libraries.
- `emit_metal_body` vs `emit_cuda_body` per atom (different shader
  language, but identical logical body shape).
- `build.rs` integration (`xcrun metal` vs `nvcc`).
- Tile-shape decisions (Apple 8×8 bf16 vs NVIDIA 16×8 / 16×16 /
  wgmma) — drives `dispatch_shape` per chip.
- Atomicity wall set (CUDA can fuse across attention via TK
  persistent kernels + `cp.async` / TMA; Metal can't — no public
  `simdgroup_event` / async tile copy).

**Naming discipline:**
- Anything in `ferrite-forward-macro/src/atom.rs` or
  `fuse_pass.rs` — backend-neutral name (no `Metal*` prefix).
- Anything that references a specific shader-language primitive or
  toolchain — backend-prefixed.
- Per-Impl atom decls stay alongside the existing `MetalFooImpl`
  shim files (the shims themselves stay `Metal*` since they bind
  to Metal-specific worker / pipeline machinery).

## Why MLX's lazy-graph fusion isn't a direct path

MLX evaluates ops lazily and the framework's compiler applies fusions
at `mx.eval()` time. We can't lean on that path because:
- MLX is its own runtime and we'd be embedding their evaluator,
  giving up Rust-owned compute / ferrite-forward's static codegen
  guarantees per `feedback_ferrite_thesis`.
- The structural choice (one big macro-time emission vs runtime graph
  evaluation) is the design point — ferrite is the macro-time
  pre-baked path.

Our equivalent: build the same kind of fusion compiler at macro
expansion time. That's what this plan describes.
