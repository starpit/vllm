# Stencil IR — status & handoff

Dated 2026-04-19. Companion to `STENCIL_IR_DESIGN.md` (vocabulary freeze) and `STENCIL_IR_SKETCH.md` (struct shapes). This doc is the "what landed, what's next, where to look" layer; the other two stay untouched.

## Read order for a fresh session

1. **This doc** — current state + next steps. Start here.
2. `STENCIL_IR_DESIGN.md` — vocabulary (3 roles, 5 dep kinds, 3 clarifications). Frozen. Only needed when the work touches vocabulary.
3. `STENCIL_IR_SKETCH.md` — §11 struct sketch that preceded the crate. Largely realised; see divergences at the bottom of this doc. Only needed for IR-type archaeology.

## The goal — don't get this wrong

The target is **one persistent `__global__` per model forward on SM90+**, in the HazyResearch Megakernels sense: 1 CTA per SM, 20 warps = 5 warpgroups (loader / consumer×3 / storer), compile-time instruction sequence (no runtime VM — we codegen the topo order), `g.Bar` between regions, cross-region `smem` reuse. Regions stay distinct (Gemma3 alternates Attn(W=∞) / Attn(W=4096) per-layer — that's *why* Region exists); the megakernel is a statically scheduled composition of them, not a fused single stencil.

**On SM89 and earlier**, the megakernel path is not the runtime path. The solver naturally picks conventional per-op impls; the stencil IR happens to lower cleanly there too as a side-effect, and we emit SM90 `.cu` files at build time *for inspection*, but the plan is not to run megakernel on SM89.

### Anti-patterns this session burned cycles on

- Don't read "Region" as "per-impl kernel wrapper"; it's a parametric chunk of computation that the megakernel composes.
- Don't chase the old 3a/3b SM89 smoke-kernel path — that's per-region-kernel scaffolding, off-critical-path. Keep for reference, don't extend.
- Don't write scalar single-thread placeholder helpers as if they were progress; either the prelude helper does real work or it `__trap()`s honestly.
- Don't conflate "megakernel" with kernel fusion. It's static scheduling, not region-body fusion.
- Don't declare ambient identifiers (`Q_frag`, `smem_q`, …) in the emitter output — they come from the prelude.

## Where we are

**Every real-model FUF (Llama/Gemma2/Gemma3/Qwen2/Qwen3/Mistral/Granite/CommandR, full-precision + marlin + bnb4 + gptq variants) now lowers to a complete SM90 megakernel source file AND compiles cleanly via nvcc.** Llama-3-8B: 227 regions / 290 control edges / ~14k lines → 187 KB object file. Qwen3-0.6B: 339 regions / 450 edges / ~16k lines → 227 KB. Gemma-3-12B: 676 regions / 675 edges / ~32k lines → 409 KB. Gemma-3-27B → 516 KB. Every build writes `/tmp/ferrite-stencil/<variant>-sm90.cu`; `nvcc -arch=sm_89 -I csrc -c <file>.cu` produces a linkable object.

The pipeline runs FUF → Stencil IR → Megakernel → emitted source → nvcc → object file, end-to-end, on real models. Helpers are trap-bodied placeholders (`__trap()`) — running the object would abort on the device; real PTX lowering lands one helper at a time.

## First 5 minutes (verify the claim)

From the worktree root (`vllm-rs/`):

```bash
# 1. Tests all green (40 lib + 7 integration on stencil; 14 on lowering).
cargo test -p ferrite-stencil --lib | tail -3
cargo test -p ferrite-stencil --tests | tail -3
cargo test -p ferrite-forward-macro --lib lower_to_stencil | tail -3

# 2. Regenerate the megakernel .cu files for a model crate; watch the
#    telemetry line — "N/N regions on sm89_fa2 · 0 skipped · mega <L>L/<E>e · <path>".
touch crates/ferrite-model-llama/src/lib.rs
cargo check -p ferrite-model-llama 2>&1 | grep "ferrite stencil · llama-3-8b "

# 3. Compile the generated megakernel with nvcc. Expect a ~190KB .o,
#    warnings OK (default-ctor __device__ attr), no errors.
/usr/local/cuda-12.9/bin/nvcc -arch=sm_89 \
    -I crates/ferrite-stencil/csrc \
    -c /tmp/ferrite-stencil/llama-3-8b-sm90.cu \
    -o /tmp/llama-3-8b.o
ls -l /tmp/llama-3-8b.o
```

If any of these fails before your first edit, stop and investigate — don't start adding features on top of a broken foundation.

## What an emitted kernel looks like

Peek at `/tmp/ferrite-stencil/llama-3-8b-sm90.cu`. Structural shape:

```cuda
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "ferrite_stencil_prelude.cuh"
typedef __nv_bfloat16 bf16;
__device__ uint32_t gbar_counter;

__global__ void mega_kernel(
    uint32_t num_token_tiles, uint32_t num_head_tiles, ...,
    const bf16* __restrict__ A_gmem, const bf16* __restrict__ B_gmem, bf16* __restrict__ C_gmem,
    const bf16* __restrict__ Embed_gmem, ... /* 18 pointers total */
) {
  uint32_t wg = threadIdx.x / 128u;
  // ═══ region 0 (rmsnorm) pipeline_depth=0 ═══
  for (uint32_t token_tile = 0; token_tile < num_token_tiles; ++token_tile) { ... }
  // ─── inter-region barrier: region 0 → region 1 (Barrier) ───
  gbar_sync(&gbar_counter);
  // ═══ region 1 (qkv_rope) pipeline_depth=3 ═══
  for (uint32_t token_tile = 0; ...) for (uint32_t head_tile = 0; ...) {
    for (uint32_t k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
      { uint32_t slot = (head_tile + 3) % 3;
        if (wg == LOADER_WG) { tma_load_2d(smem_wqkv[slot], ...); mbarrier_arrive(&bar_smem_wqkv[slot]); } }
      ...
      { if (wg == CONSUMER_WG) { wgmma_fence(); wgmma_mma_async(QKV_frag, smem_x[slot], smem_wqkv[slot]); ... } }
    }
    ...
  }
  // ═══ region 2 (fa2_prefill) pipeline_depth=3 ═══
  ...
}  // end mega_kernel

extern "C" cudaError_t launch_mega_kernel(cudaStream_t stream, ...) {
    uint32_t zero = 0;
    cudaMemcpyToSymbolAsync(gbar_counter, &zero, ...);
    int sm_count = 0; cudaDeviceGetAttribute(&sm_count, ...);
    mega_kernel<<<dim3((unsigned)sm_count), dim3(640u), 0, stream>>>(...);
    return cudaGetLastError();
}
```

The placeholders (`tma_load_2d`, `wgmma_mma_async`, `mbarrier_arrive`, `gbar_sync`, `StencilFrag`, `smem_wqkv[]`, `bar_smem_*`, `CONSUMER_WG`, …) all resolve through `csrc/ferrite_stencil_prelude.cuh`. Their bodies are `__trap()` — the object links but aborts on launch. Real PTX goes in the prelude, one helper at a time; the emitter stays put.

## What's landed (continuing from step 3b)

Ten commits on `worktree-ff2` extending the Stencil IR toward the goal of "FUF → emitted persistent SM90+ megakernel":

| Commit | Layer | Notes |
|---|---|---|
| `53ac38e9c` | **emit_megakernel shell** — `(Megakernel, ArchMap) → String`, one `__global__`, topo-sorted regions, warpgroup dispatch, inter-region barriers via `ArchMap::barrier`. | 5 new tests; stencil suite at 19 green. |
| `80531c069` | First non-attention region templates — `gemm_region`, `rmsnorm_region`, `residual_add_region`. | 4 tests; 23 green. |
| `c1eeef6f2` | `lower_impl` arms for non-attention impls (`fused_gemm_bias`, `fused_add_rms_norm`, `add_ref`, marlin/bnb4 GEMM variants). `LowerHints` grows `gemm_*_tile`, `hidden_dim`, `token_tile`; `hidden_dim` flows from `bounds["hidden_size"]`. | 4 tests; lowering suite 11 green. |
| `c8fa9cd62` | `lower_assignment` populates `Megakernel.control` from the FUF's inter-subgraph dep graph. Region ids = position in `regions` for stable references. | 2 tests; 13 green. |
| `5d02141f9` | Wire `emit_megakernel` into the macro drive. Every forward! expansion invokes the emitter on the lowered Megakernel against SM90 `ArchMap` and writes `/tmp/ferrite-stencil/<variant>-sm90.cu`; telemetry reports lines + control-edge count. First real-model evidence the pipeline connects end-to-end. |
| `f65fefa03` | `qkv_rope_region` template + 6 impl arms (prefill/cache/qk_norm × marlin/bnb4). | Skip counts drop ~30–80 per model. |
| `d57bc7274` | `gate_up_silu_mul_region` template + 6 impl arms. `LowerHints.intermediate_dim` from `bounds["intermediate_size"]`. | Skip counts drop another ~30–80. |
| `8b857c4aa` | `unary_inplace_region` (parametric on op tag), sliding-window support on paged decode, reference variants routed through existing templates. Telemetry grows `[impl,…]` summary of skipped impl names. | Llama-3-8B hits 1 skipped. |
| `340bf771b` | `embed_region` (gather-only, no Compute), `flashinfer_attention_decode` routed to paged decode, `derive_control_edges` transitively traverses skipped (reshape_ref) subgraphs so dep chains stay intact. | 0 skipped on every supported model; reshape_ref stays skipped but the closure preserves A → B edges through it. |
| `42474b51c` | Full `emit_ops` intrinsic expansions for all 31 non-attention tags (pipelined/preamble loads, gemm accumulate, generic stores, cache stores, rmsnorm_compute, elementwise_add, apply_rope, silu_mul_fuse, unary ops, embed gather). | 38 lib + 7 integration stencil tests green. Emitted kernels render real TMA/wgmma/cp.async/mma.sync bodies, no `tag();` stubs. |
| `7e6dcf8b9` | Gmem pointer plumbing: `emit_ops::gmem_refs` per-tag access table + union into signature. `const bf16* __restrict__` for read-only tensors, mutable for anything read+written (e.g. Q_gmem, written by qkv_rope and read by attention). `__device__ uint32_t gbar_counter;` emitted above the kernel. | Llama-3-8B signature: 8 scalars + 18 gmem pointers. |
| `ec41b8fc4` | Persistent-CTA launcher emitted alongside the kernel. `extern "C" launch_mega_kernel(stream, scalars..., gmem_ptrs...)` zeroes `gbar_counter`, queries `multiProcessorCount`, launches with `grid=#SMs`, `block=640` (20 warps). Same param list as the kernel — no marshaling divergence. |
| `c3fee7188` | **Emitted megakernel compiles cleanly via nvcc.** New `csrc/ferrite_stencil_prelude.cuh` (220 lines) providing typedefs (bf16, Mbarrier, StencilFrag), warpgroup-id constants, smem/fragment placeholders, and templated `__trap()`-bodied helpers for every intrinsic (tma_load_2d, cp_async_128, wgmma_mma_async, mma_sync_accumulate, row_max, silu, frag_mul, apply_rope, …). Emitter adds `#include` preamble, IndexedScalar bounds flow as `const uint32_t*`, fixed a few C++23-incompatible subscript patterns. **Verified: `nvcc -arch=sm_89 -I csrc -c <variant>.cu` produces linkable objects for llama-3-8b (187 KB), qwen3-0.6b (227 KB), gemma-3-12b-it (409 KB), gemma-3-27b-it (516 KB), c4ai-command-r (214 KB), gemma2-2b (173 KB), and others.** Placeholders trap at runtime — real PTX lowers one helper at a time. |

Cumulative: **ferrite-stencil 40 lib + 7 integration green · forward-macro lowering 14 green · nvcc-compilable `.o` for every supported model variant.**

## Pipeline that now exists

```
Fuf + solver::Assignment + impl_lib::ImplementationLibrary
    → ferrite_forward_macro::lower_to_stencil::lower_assignment(..., &LowerHints)
    → ferrite_stencil::Megakernel { regions, control }
    → ferrite_stencil::emit_megakernel(&mega, &arch)
    → one persistent __global__ (text), 1 CTA per SM, warpgroup-partitioned
       on SM90, inter-region barriers from ArchMap::barrier
```

Each region still schedules through `schedule_wavefront`; `emit_megakernel` calls the scheduler internally, topo-sorts regions by `Megakernel.control`, emits the union of entry scalars as kernel params, and inlines each region's preamble + serial-axis loop (if any) + epilogue. Per-node bodies expand through `emit_ops::expand` to TMA + wgmma (SM90) or cp.async + mma.sync (SM89) — no stubs remain for any tag any current template emits.

## Key file pointers

- Crate root: `vllm-rs/crates/ferrite-stencil/`
  - `src/ir.rs` — core types + `validate()`
  - `src/template.rs` — region templates: `attn_region`, `attn_region_paged_decode`, `gemm_region`, `rmsnorm_region`, `residual_add_region`, `qkv_rope_region`, `gate_up_silu_mul_region`, `unary_inplace_region`, `embed_region`
  - `src/arch.rs` — `ArchMap`, `HardwareUnit`, `BarrierPrim`, `sm90_fa2`, `sm89_fa2`
  - `src/schedule.rs` — arch-neutral primitives
  - `src/wavefront.rs` — preamble/body/epilogue scheduler
  - `src/emit_mega.rs` — **the megakernel emitter** (main entry: `emit_megakernel`)
  - `src/emit.rs` — per-region sketch emitter (used by snapshot tests; `emit_mega` is the real target)
  - `src/emit_ops.rs` — per-tag intrinsic expansion table (31 non-attention + 7 attention tags populated) + `gmem_refs` for signature-time pointer plumbing
  - `src/print.rs` — round-trip printer (used by tests, not by emitter)
  - `csrc/ferrite_stencil_prelude.cuh` — **the megakernel prelude**. `__trap()`-bodied placeholders for every helper the emitter references; typedefs (bf16, Mbarrier, StencilFrag), warpgroup-id constants, ambient smem/fragment state. This is what turns the emitted `.cu` into a linkable object.
  - `csrc/stencil_prelude_sm89.cuh` — older per-region-kernel prelude (unused by `emit_mega`; scaffolding from the 3b path)
  - `csrc/stencil_smoke_sm89.cu` — 3b hand-written smoke kernel (off-critical-path; kept for reference)
- Launch wrappers + GPU tests (off-critical-path): `vllm-rs/crates/ferrite-stencil-kernels/` — 3b launcher. Not wired to `emit_megakernel`.
- Lowering: `vllm-rs/crates/ferrite-forward-macro/src/lower_to_stencil.rs`
  - `lower_assignment` / `lower_assignment_partial` with transitive-closure control-edge derivation
  - `lower_impl` match covers every `impl_lib` Impl name we currently see; new impls fail to the `UnsupportedImpl` branch with a crisp error
- Macro drive: `vllm-rs/crates/ferrite-forward-macro/src/lib.rs` — parallel-pass telemetry + `emit_megakernel` output written to `/tmp/ferrite-stencil/<variant>-sm90.cu`

## What's left to reach the finish line

The design doc's finish line is *efficient megakernel execution from the FUF* — comm/compute overlap, cross-subtile parallelism, real SM90 utilization, one launch per forward. From today's state:

### The first next step

**Pick one: fix per-region smem/fragment collision (item 2) OR replace one trap-bodied helper with real PTX (item 1).** Both are concrete. Item 2 is the bigger correctness win — until it's fixed, multiple GEMM regions will all try to write the same `C_frag` / `smem_a` / `smem_b`. Item 1 is the smaller mechanical cut — pick the leftmost helper from the list below, write the real PTX, run `nvcc -c` to confirm it still compiles. Repeat.

Don't try to do both at once, and don't re-plan the whole arc before starting. Commit the smallest meaningful slice, watch tests + nvcc stay green, move.

### All remaining items

1. **Real PTX bodies in the prelude.** Every helper in `ferrite_stencil_prelude.cuh` is a `__trap()` placeholder. Replacing them with real PTX is the bulk of the remaining work — one helper at a time, keeping the prelude as stable ABI between emitter and compilable source. Order of impact (leftmost first):
   - `cp_async_128` + `cp_async_commit_group` + `cp_async_wait_group` (SM89 path, unblocks L4 testing)
   - `mma_sync_accumulate` (SM89 mma.sync m16n8k16)
   - `stg_128` / `tma_store_2d` (writeback)
   - `tma_load_2d` + `mbarrier_arrive` / `mbarrier_wait` (SM90 TMA path)
   - `wgmma_mma_async` + fence/commit/wait (SM90 compute)
   - `gbar_sync` (cross-CTA atomic counter + spin-wait)
   - Fragment helpers: `row_max`, `row_sum`, `exp2f_frag`, `warp_reduce_sum_of_squares`, `silu`, `rope_rotate`
2. **Fragment type + smem allocation design.** Placeholders treat every fragment as opaque `StencilFrag`; real lowering needs concrete mma fragments (register layouts, accumulator types), plus a region-local smem-allocation pass so multiple GEMM regions don't collide on `smem_a` / `smem_b` / `C_frag`. Currently every region shares the same file-scope placeholders — correctness-breaking, not just cosmetic.
3. **Axis-name threading through `ExpandCtx`.** Today expansions hard-code axis names (`q_tile`, `head_group`, …) regardless of the region they're emitted inside; paged decode's `b` axis falls through to a file-scope `b=0` placeholder. Passing axis names per-region at emit time fixes this without touching the vocabulary.
4. **FUF-level tensor naming.** The signature uses tag-level names (`Q_gmem`, `A_gmem`, …) deduped globally — multiple GEMM regions share one `A_gmem` pointer, which is wrong at runtime (each layer has its own weights). Needs to walk back from region Node → FufOpRef → FUF inputs/outputs to get per-region unique names, then dedupe on actual tensor identity.
5. **Tile calibration + real shape walking.** `LowerHints` tile fields are `Default`-valued; `num_q_heads` / `num_kv_heads` default to 1 for qkv_rope. Pull from per-Impl calibrated sizes and `bounds`.
6. **SM89 solver policy.** Document that on SM89 targets the solver picks conventional per-op impls; megakernel path is SM90+. Today we emit SM90 source for inspection regardless — fine as telemetry, but the runtime branch should split.
7. **Ad-hoc H100 run.** First proof the kernel launches. Needs (1) far enough along that `__trap()` isn't hit immediately. Correctness vs Python vLLM comes after.

Items 1–4 are the remaining substantive work before a first runtime trial. 5–7 are follow-on.

## Common commands

```bash
# Run the full pipeline on a model crate (regenerates all its .cu files).
touch crates/ferrite-model-<arch>/src/lib.rs
cargo check -p ferrite-model-<arch>  2>&1 | grep "ferrite stencil"
# Archs: llama, gemma2, gemma3, qwen2, qwen3, mistral, granite, commandr

# Recompile one emitted megakernel. -arch=sm_89 is fine even for sm90_fa2
# output — the prelude traps are arch-neutral, the compile is what matters.
/usr/local/cuda-12.9/bin/nvcc -arch=sm_89 \
    -I crates/ferrite-stencil/csrc \
    -c /tmp/ferrite-stencil/<variant>-sm90.cu \
    -o /tmp/<variant>.o

# Stencil crate tests, lowering tests, integration tests, clippy.
cargo test -p ferrite-stencil --lib
cargo test -p ferrite-stencil --tests
cargo test -p ferrite-forward-macro --lib lower_to_stencil
cargo fmt -p ferrite-stencil -p ferrite-forward-macro
cargo clippy -p ferrite-stencil     --lib -- -D warnings
cargo clippy -p ferrite-forward-macro --lib -- -D warnings
```

Never mix `cargo` commands from outside `vllm-rs/` — `cargo` from the worktree root won't find the workspace manifest.

## Sketch → reality divergences worth knowing

- `AxisId` is `u16` (sketch said `SmallVec<(AxisId, i32)>`, left unspecified). Works.
- `RegionTemplate` wasn't implemented as a type; templates are free functions (`attn_region`, `gemm_region`, …). Collapsed into plain functions because the `build: fn(&TemplateArgs)` indirection added no information when there's one call site per template.
- `HardwareUnit::Warpgroup` uses `role_name` + `num_wg` rather than `first_warp` + `count`. Matches the design doc's "reclaim controller wg, 5 wg = 20 warps" comment rather than pinning exact warp indices — those are codegen-time decisions, not mapping-time.
- `ArchMap.role: fn(Role, &Region) -> HardwareUnit` is a plain `fn`, no trait object. Two arches, both written by hand, no dynamic dispatch needed.
- **`lower_impl` dispatches by `Impl::name()` string**, not by structural properties of the FUF. Works for the current impl library but will want a trait-based dispatch if the Impl count grows a lot.
- **`emit_megakernel` produces `String`**, not a `TokenStream`. Because the whole point is that the emitter output goes into a `.cu` file the launcher reads, not into the macro's expansion token stream. The old per-region `emit_kernel_sketch` had the same shape; we kept it.
- **Step 3b's `stencil_smoke_sm89.cu` is off the critical path.** It validates the earlier per-region per-SM89-kernel approach. The real emitter target is `emit_megakernel` producing one `__global__` for the whole forward; SM89 per-region codegen is not the runtime path (per design: on SM89, the solver picks conventional impls and doesn't engage the megakernel emitter at all).

## Memory pointer

`~/.claude/projects/-home-moosevan-vllm/memory/project_ferrite_stencil.md` tracks this status for future sessions via auto-memory; if you update the status here, skim that file too.
