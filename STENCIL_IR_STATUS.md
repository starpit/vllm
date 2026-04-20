# Stencil IR — status & handoff

Dated 2026-04-20. Companion to `STENCIL_IR_DESIGN.md` (vocabulary freeze) and `STENCIL_IR_SKETCH.md` (struct shapes). This doc is the "what landed, what's next, where to look" layer; the other two stay untouched.

## THE PLAN (load-bearing — re-read before touching anything)

**The compiler emits CUDA that uses ThunderKittens primitives directly.** Static wavefront schedule from the stencil IR. No VM. No Megakernels-style op structs (controller / loader / launcher / consumer / storer). No runtime instruction fetch. Just a straight-line `__global__` that calls `kittens::tma::load_async` / `kittens::warpgroup::mma` / `kittens::warp::store` / `kittens::semaphore` in the order the stencil's wavefront scheduler produces.

**Reference for how-to-use-kittens**: `~/Megakernels/demos/cross-gpu-llama/` (throughput branch of `~/Megakernels`). Their op-struct bodies show how to compose kittens primitives for real matmul-shape LLM ops. We port the shape, *not* the VM scaffolding.

**Goal**: `vllm chat --model=<small-llama>` produces coherent output on H100 through this emit path.

**Build flags** (copy into `build.rs` when the integration lands):

```
nvcc -DKITTENS_HOPPER -gencode=arch=compute_90a,code=sm_90a \
     -std=c++20 --extended-lambda --expt-relaxed-constexpr -O3 \
     -I ~/ThunderKittens/include
```

### Wrong turns to not re-take

- **Writing raw PTX in the stencil prelude** (`cp.async.ca` / hand-rolled `tma_load_2d`). Today's `emit_mega.rs` is that direction. It's the *prior* direction, kept as reference; `emit_kittens.rs` replaces it. When every region lowers cleanly through emit_kittens and vllm chat runs end-to-end, `emit_mega.rs` + `csrc/ferrite_stencil_prelude.cuh` get deleted.
- **sm89 cp.async helpers for the real runtime path**. Runtime target is sm_90a. sm89 lowering is off-critical-path per design §6.
- **Inflated estimates from hand-writing every primitive**. Kittens gives us TMA, WGMMA, stmatrix, mbarriers, fragment math for free. We don't reinvent them.
- **Option-shopping in mid-commit**. When a wrinkle surfaces, pick one interpretation consistent with the plan and keep going. Don't pause to redesign.

## Read order for a fresh session

1. **This doc** — current state + next steps. Start here.
2. `STENCIL_IR_DESIGN.md` — vocabulary (3 roles, 5 dep kinds, 3 clarifications). Frozen. Only needed when the work touches vocabulary.
3. `STENCIL_IR_SKETCH.md` — §11 struct sketch that preceded the crate. Largely realised; see divergences at the bottom of this doc. Only needed for IR-type archaeology.

## The goal — don't get this wrong

The target is **one persistent `__global__` per model forward on SM90a+**, in the HazyResearch Megakernels sense: 1 CTA per SM, 20 warps = 5 warpgroups (loader / consumer×3 / storer), compile-time instruction sequence (no runtime VM — we codegen the topo order), `g.Bar` between regions, cross-region `smem` reuse. Regions stay distinct (Gemma3 alternates Attn(W=∞) / Attn(W=4096) per-layer — that's *why* Region exists); the megakernel is a statically scheduled composition of them, not a fused single stencil.

**On SM89 and earlier**, the megakernel path is not the runtime path. The solver naturally picks conventional per-op impls; the stencil IR happens to lower cleanly there too as a side-effect, and we emit SM90 `.cu` files at build time *for inspection*, but the plan is not to run megakernel on SM89. The local dev box is an L4 (SM89); the runtime target is H100 / SM90a (user has access elsewhere).

### Anti-patterns this session burned cycles on

- Don't read "Region" as "per-impl kernel wrapper"; it's a parametric chunk of computation that the megakernel composes.
- Don't chase the old 3a/3b SM89 smoke-kernel path — that's per-region-kernel scaffolding, off-critical-path. Keep for reference, don't extend.
- Don't write scalar single-thread placeholder helpers as if they were progress; either the prelude helper does real work or it `__trap()`s honestly.
- Don't conflate "megakernel" with kernel fusion. It's static scheduling, not region-body fusion.
- Don't declare ambient identifiers (`Q_frag`, `smem_q`, …) in the emitter output — they come from region-local scopes now (item 2), not from prelude file-scope placeholders.
- Don't try to land real PTX for a variadic helper (`cp_async_128`, `tma_load_2d`, `wgmma_mma_async`, `stg_128`) without first picking a concrete `StencilFrag` type and passing byte counts / smem descriptors from the emitter. The variadic signatures today absorb anything; they trap honestly. The honest next step is a shape commitment, not another trap-to-real-PTX tweak.
- Don't target `-arch=sm_90` for the megakernel. Use `-gencode=arch=compute_90a,code=sm_90a` (or just `-arch=sm_90a`). `wgmma.fence` / `wgmma.commit_group` / `wgmma.wait_group` / TMA are all sm_90a-only.

## Where we are (kittens path)

**`emit_kittens.rs` has 6 of 9 region templates ported to a kittens-based `.cu` emit.** Each emits:
- A `globals_t` struct with `kittens::gl<>` members for every gmem tensor, all runtime dims (`-1`), compile-time tile shapes baked into the attached TMA descriptor type.
- A `__global__ __launch_bounds__(…)` kernel that takes `const __grid_constant__ globals_t g`, declares `__shared__` kittens tiles/vecs/mbarriers, and calls `kittens::tma::load_async` / `warp::load` / `warpgroup::mma_AB` / `warpgroup::store` / `tma::store_async` in static order.
- An `extern "C" cudaError_t launch_<region>(...)` host wrapper that constructs `kittens::gl` from raw pointers + dims and launches.

Region coverage:

| Region | Status | Kernel primitives |
|---|---|---|
| `rmsnorm` | ✓ commit `f91bf20d7` | `warp::load/mul/sum/store`, `sv_bf<D>` |
| `residual_add` | ✓ commit `cec725bdb` | `warp::load/add/store`, `sv_bf<D>` |
| `unary_inplace` | ✓ commit `cec725bdb` | `warp::load/mul/store` (scalar_mul placeholder) |
| `embed` | ✓ commit `cec725bdb` | `warp::load/store` with token_ids gather |
| `gemm` | ✓ commit `960a064ea` | `warpgroup::mma_AB` on `rt_fl<16, TN>`, `tma::load_async` with `semaphore`, `warpgroup::store` + `tma::store_async` |
| `gate_up_silu_mul` | ✓ commit `7594c09ed` | two parallel `warpgroup::mma_AB`, SiLU via `copy/mul/exp/add/div` on `rt_fl`, TN=64 for shared-mem budget |
| `qkv_rope` | not ported | matmul + RoPE rotation + KV cache append — complex, 3 outputs from 1 X |
| `attn_region` (FA2 prefill) | not ported | tiled attention + online softmax (hardest) |
| `attn_region_paged_decode` | not ported | M=1 decode attention + paged-KV gather |

Every ported region nvcc-compiles cleanly on sm_90a with the build flags above; object sizes 14 KB (header only) → 160 KB (4 mem-bound regions together). GEMM + gate_up_silu_mul emit real `wgmma.mma_async` + `tma::load_async`/`store_async` PTX.

Legacy `emit_mega.rs` path (hand-rolled prelude helpers, sm_89 cp.async.ca, `bf16[BYTES/2]` smem decls, `gmem + (q_tile * 16384u + …)` address expressions): still present, still nvcc-compiles for all 8 model variants, but scheduled for deletion once kittens path covers every region and vllm chat runs.

## Next three things (in order)

1. **Port `qkv_rope_region`.** Three parallel WGMMAs (Wq, Wk, Wv) sharing X. RoPE rotation on Q and K via `kittens::warp::mul/add` on register tiles with cos/sin loaded from `g.rope_cos` / `g.rope_sin`. V stored direct; K stored to paged KV cache via `AxisDivGather` on `block_table`. See `~/Megakernels/demos/cross-gpu-llama/qkv_rope_append.cu` for the reference pattern.
2. **Port `attn_region` + `attn_region_paged_decode`.** FA2 online softmax using kittens register-tile primitives (`row_max`, `row_sum`, `exp2` in `base_ops`). Tiled over kv_tile with pipelined loads. Reference: `~/Megakernels/demos/cross-gpu-llama/attention_prefill.cu` + `attention_decode.cu`.
3. **Integration (task-8 / task-9 / task-10).** `build.rs` in `vllm-cuda` or a new `ferrite-kittens-builder` crate nvcc's the per-model emitted `.cu` into a `.a` with deterministic symbol names; Rust FFI bindings for each `launch_<region>`; cuda_worker routes sm90a forwards through megakernel dispatch instead of the HostCallable path.

Do NOT return to the raw-PTX `emit_mega.rs` path, CUtensorMap plumbing by hand, or stride-fix commits against the legacy emit — those are dead-ends we already verified don't get us closer to the goal.

## Legacy (`emit_mega.rs`) status

**Every real-model FUF (Llama/Gemma2/Gemma3/Qwen2/Qwen3/Mistral/Granite/CommandR, full-precision + marlin + bnb4 + gptq variants) now lowers to a complete SM90a megakernel source file AND compiles cleanly via nvcc on both `-arch=sm_89` and `-gencode=arch=compute_90a,code=sm_90a`.** Llama-3-8B: 227 regions / 290 control edges / ~18 k lines → 595 KB (sm_89) / 446 KB (sm_90a). Qwen3-0.6B → 878 KB / 663 KB. Gemma-3-12B → 1.7 MB / 1.3 MB. Every build writes `/tmp/ferrite-stencil/<variant>-sm90.cu`.

**Phase A — correctness shape — complete.** Per-region scoped smem/frag/mbarriers (item 2), axis names threaded through `ExpandCtx` (item 3), and per-layer FUF tensor identities in the kernel signature (item 4). Llama-3-8B's signature carries ~230 per-tensor pointers (`w{id}_{layer}`, `t{tile}_{slot}`, `x_{kind}_{index}`) instead of 18 canonical names — each layer's Wqkv/Wo/Wgate/Wup/RMSnorm is its own parameter.

**Phase B — byte-count plumbing + templated call sites landed.** Every region emits region-scope `constexpr uint32_t SMEM_{LOCAL}_BYTES = …u;` constants from `Region.tile_consts` (populated by each template from its tile params), and smem declarations reference them: `__shared__ bf16 smem_q[SMEM_Q_BYTES / 2]` instead of the opaque `__shared__ StencilFrag smem_q`. Every smem-targeted `cp_async_128 / tma_load_2d / tma_store_2d / stg_128 / stmatrix_smem` call now carries the matching `<SMEM_*_BYTES>` template arg. Helper bodies still trap — the prelude signatures got the non-type template parameter (`uint32_t BYTES = 0`) but the bodies are unchanged. What's not yet bridged is the *address* half: call sites still pass raw axis indices (`q_tile, head_group`), and a real cp.async needs a concrete `gmem + tile_offset` pointer. That refactor is what task 3 below demands.

**Phase B — real PTX — first wave landed.** 8 fixed-signature helpers compile to real PTX (guarded by `__CUDA_ARCH__`):

| Helper | PTX | Arch |
|---|---|---|
| `cp_async_commit_group` | `cp.async.commit_group` | SM80+ |
| `cp_async_wait_group(N)` | `cp.async.wait_group N` | SM80+ |
| `wgmma_fence` | `wgmma.fence.sync.aligned` | SM90a+ |
| `wgmma_commit_group` | `wgmma.commit_group.sync.aligned` | SM90a+ |
| `wgmma_wait_group<N>` | `wgmma.wait_group.sync.aligned N` | SM90a+ |
| `gbar_sync(counter)` | atomicAdd spin + `__syncthreads` | any (SM60+) |
| `cluster_sync` | `barrier.cluster.sync.aligned` | SM90+ |
| `mbarrier_wait()` no-arg | `__syncthreads` | any |

`gbar_sync` is the HazyResearch-style global barrier — no hardware grid-sync, just atomic-counter spin-wait. Thread 0 of each CTA elects, increments, spins, and re-enters via `__syncthreads`.

**Still trapping** (these are the real work):
- Data movement: `cp_async_128<BYTES>`, `tma_load_2d<BYTES>`, `tma_store_2d<BYTES>`, `stg_128<BYTES>` — byte counts are threaded; what's missing is the *address* — call sites need to emit `gmem + tile_offset_expr` (walked from `LoadAddr.terms`) so the helper body copies the right tile, not the whole tensor.
- Tensor-core compute: `wgmma_mma_async`, `mma_sync_accumulate` — need accumulator fragments + smem descriptors.
- Smem store: `stmatrix_smem` — needs smem address + fragment layout.
- Pointer-form mbarriers: `mbarrier_wait(bar*)`, `mbarrier_arrive(bar*)` — need phase-bit plumbing per barrier.
- Named semaphores: `sem_wait(name, depth)` — needs compile-time name hash.
- Fragment primitives: `row_max`, `row_sum`, `exp2f_frag`, `frag_mul`, `frag_add`, `silu`, `rope_rotate`, `warp_reduce_sum_of_squares` — need concrete `StencilFrag` type.

Running the object today: the kernel launches, enters region 0, hits a trap on the first data-movement instruction. Progress is measured in how far past `region 0` we get before trap.

## First 5 minutes (verify the claim)

From the worktree root (`vllm-rs/`):

```bash
# 1. Tests all green (59 lib + 7 integration on stencil; 17 on lowering).
cargo test -p ferrite-stencil --lib | tail -3
cargo test -p ferrite-stencil --tests | tail -3
cargo test -p ferrite-forward-macro --lib lower_to_stencil | tail -3

# 2. Regenerate the megakernel .cu for a model crate; watch the
#    telemetry — "N/N regions on sm89_fa2 · 0 skipped · mega <L>L/<E>e · <path>".
touch crates/ferrite-model-llama/src/lib.rs
cargo check -p ferrite-model-llama 2>&1 | grep "ferrite stencil · llama-3-8b "

# 3. Compile with nvcc. Both archs must succeed; sm_90a is the runtime target.
/usr/local/cuda-12.9/bin/nvcc -arch=sm_89 \
    -I crates/ferrite-stencil/csrc \
    -c /tmp/ferrite-stencil/llama-3-8b-sm90.cu \
    -o /tmp/llama-3-8b-sm89.o
/usr/local/cuda-12.9/bin/nvcc -gencode=arch=compute_90a,code=sm_90a \
    -I crates/ferrite-stencil/csrc \
    -c /tmp/ferrite-stencil/llama-3-8b-sm90.cu \
    -o /tmp/llama-3-8b-sm90a.o
ls -l /tmp/llama-3-8b-sm89.o /tmp/llama-3-8b-sm90a.o
```

If any of these fails before your first edit, stop and investigate — don't start adding features on top of a broken foundation.

**Useful debug env var**: `FERRITE_STENCIL_4B_TRACE=1 cargo check -p ferrite-model-llama` prints per-subgraph impl + FUF external inputs + resolved gmem bindings — one line per region. Turn on when a new Impl's bindings look wrong.

## What an emitted kernel looks like

Peek at `/tmp/ferrite-stencil/llama-3-8b-sm90.cu`. Structural shape (after Phase A):

```cuda
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "ferrite_stencil_prelude.cuh"
typedef __nv_bfloat16 bf16;
__device__ uint32_t gbar_counter;

__global__ void mega_kernel(
    uint32_t num_token_tiles, uint32_t num_head_tiles, /* ... */,
    const bf16* __restrict__ w0,              // embed weight
    bf16* __restrict__ t0_0,                  // embed output
    const bf16* __restrict__ w1_0,            // layer 0 rmsnorm weight
    const bf16* __restrict__ w2_0,            // layer 0 Wqkv
    const bf16* __restrict__ w5_0,            // layer 0 Wo
    bf16* __restrict__ t5_0, t5_1, t5_2,      // layer 0 Q, K, V outputs
    bf16* __restrict__ t6_0,                  // layer 0 attn output
    /* ... 220 more pointers, one per FUF tensor identity ... */
    bf16* __restrict__ x_kv_cache_0,          // layer 0 kv cache
    const uint32_t* __restrict__ x_input_ids_0
) {
  uint32_t wg = threadIdx.x / 128u;
  // ═══ region 0 (embed) pipeline_depth=0 ═══
  {
    constexpr uint32_t SMEM_OUT_BYTES = 8192u;
    __shared__ bf16 smem_out[SMEM_OUT_BYTES / 2];
    StencilFrag Embed_frag;
    __shared__ Mbarrier bar_embed;
    __shared__ Mbarrier bar_Y_gmem_ready;
    for (uint32_t token_tile = 0; token_tile < num_token_tiles; ++token_tile) {
      // preamble: cp_async_128(Embed_frag, w0 + row * hidden_stride);  (still bare — no smem_embed yet)
      // body: stmatrix_smem<SMEM_OUT_BYTES>(smem_out, Embed_frag);
      //       tma_store_2d<SMEM_OUT_BYTES>(t0_0, smem_out, token_tile, 0u);
    }
  }  // end region 0
  gbar_sync(&gbar_counter);
  // ═══ region 1 (rmsnorm) pipeline_depth=0 ═══
  {
    constexpr uint32_t SMEM_X_BYTES = 8192u;
    constexpr uint32_t SMEM_W_BYTES = 8192u;  // weight is 1-D [hidden_dim]
    constexpr uint32_t SMEM_OUT_BYTES = 8192u;
    __shared__ bf16 smem_x[SMEM_X_BYTES / 2];
    __shared__ bf16 smem_w[SMEM_W_BYTES / 2];
    __shared__ bf16 smem_out[SMEM_OUT_BYTES / 2];
    StencilFrag Y_frag;
    // tma_load_2d<SMEM_X_BYTES>(smem_x, t0_0, token_tile);
    // tma_load_2d<SMEM_W_BYTES>(smem_w, w1_0, token_tile);
    // …rmsnorm_compute…
    // stmatrix_smem<SMEM_OUT_BYTES>(smem_out, Y_frag);
    // tma_store_2d<SMEM_OUT_BYTES>(t1_0, smem_out, token_tile, 0u);
  }  // end region 1
  // ... 225 more regions ...
}  // end mega_kernel

extern "C" cudaError_t launch_mega_kernel(cudaStream_t, /* same param list */) {
    uint32_t zero = 0;
    cudaMemcpyToSymbolAsync(gbar_counter, &zero, /* ... */);
    int sm_count;  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    mega_kernel<<<dim3(sm_count), dim3(640), 0, stream>>>(/* ... */);
    return cudaGetLastError();
}
```

The cp.async / wgmma fence/commit/wait calls are real PTX now. The data-movement placeholders (`tma_load_2d<BYTES>`, `cp_async_128<BYTES>`, `wgmma_mma_async`, `stg_128<BYTES>`, `stmatrix_smem<BYTES>`) and the pointer-form mbarriers still trap — but the byte-count template arg is threaded, so the next commit only needs to (a) compute a concrete `gmem + offset` at each call site by walking `LoadAddr.terms`, and (b) emit the real cp.async loop in the prelude. Running the object still aborts on the first data-movement trap; the bookkeeping around it (gbar_sync between regions, wgmma fences in attention) is real.

## What's landed

| Commit | What | Tests |
|---|---|---|
| `c3fee7188` | Prelude header — emitted megakernel compiles via nvcc | 40 lib + 7 integration + 14 lowering |
| `ec41b8fc4` | Persistent-CTA launcher + `extern "C" launch_mega_kernel` | — |
| `7e6dcf8b9` | Gmem pointer plumbing in signature | — |
| `d3605f86f` | STENCIL_IR_STATUS.md: megakernel now nvcc-compilable | — |
| `051931d45` | STENCIL_IR_STATUS.md: hand-off refresh | — |
| `e6186c69c` | **Item 2** — per-region smem/frag/mbarrier scoping | +9 → 49 lib |
| `62a9f71e8` | **Item 3** — axis-name threading through `ExpandCtx` | +4 → 53 lib |
| `85f51b403` | **Item 4a** — `Region.gmem_bindings` + `ExpandCtx::gmem` resolver | +6 → 59 lib |
| `344505683` | **Item 4b** — `lower_impl` stamps FUF tensor identities | +3 → 17 lowering |
| `a9a1a8b59` | **Item 1 first wave** — real PTX for 8 fixed-sig sync helpers | — |
| `32e0690a0` | **Item 1 second wave — byte plumbing** — `Region.tile_consts` + `SMEM_{LOCAL}_BYTES` constexpr + `bf16[BYTES / 2]` smem decls | 59 lib |
| `843498fc9` | **Item 1 second wave — templated call sites** — `cp_async_128<BYTES>(…)` etc. at every smem-targeted call; prelude helpers grow `<uint32_t BYTES = 0>` template param | 59 lib |

Cumulative at HEAD: **ferrite-stencil 59 lib + 7 integration · forward-macro lowering 17 · sm_89 + sm_90a nvcc-compilable objects for every supported model variant · smem decls are concrete `bf16[BYTES/2]` arrays · data-movement call sites carry `<BYTES>` template args.**

## Pipeline that now exists

```
Fuf + solver::Assignment + impl_lib::ImplementationLibrary
    → ferrite_forward_macro::lower_to_stencil::lower_assignment(..., &LowerHints)
    │     (stamp_gmem_bindings walks FUF tiles → per-region canonical→identity map)
    → ferrite_stencil::Megakernel { regions, control }
    │     (each Region carries gmem_bindings + domain + nodes + edges)
    → ferrite_stencil::emit_megakernel(&mega, &arch)
    │     (signature dedupes on identities; region bodies resolve canonical → identity
    │      via ExpandCtx::gmem; region-local smem/frag/mbarrier scoped in `{}`)
    → one persistent __global__ (text) + extern "C" launcher
    │     (per-variant `.cu` at /tmp/ferrite-stencil/)
    → nvcc -arch={sm_89 | sm_90a} -c
    →   linkable `.o` with real gbar_sync / cp_async fences / wgmma fences,
        trap-bodied data-movement + compute placeholders
```

Each region still schedules through `schedule_wavefront`; `emit_megakernel` calls the scheduler internally, topo-sorts regions by `Megakernel.control`, emits the union of entry scalars + per-identity gmem pointers as kernel params, and inlines each region's preamble + serial-axis loop (if any) + epilogue. Per-node bodies expand through `emit_ops::expand` with the region's axis names + gmem bindings threaded.

## Key file pointers

- Crate root: `vllm-rs/crates/ferrite-stencil/`
  - `src/ir.rs` — core types + `validate()`; `Region.gmem_bindings` added by 4a.
  - `src/template.rs` — region templates: `attn_region`, `attn_region_paged_decode`, `gemm_region`, `rmsnorm_region`, `residual_add_region`, `qkv_rope_region`, `gate_up_silu_mul_region`, `unary_inplace_region`, `embed_region`. Each constructs a Region with empty `gmem_bindings`; lowering fills them in.
  - `src/arch.rs` — `ArchMap`, `HardwareUnit`, `BarrierPrim`, `sm90_fa2`, `sm89_fa2`.
  - `src/schedule.rs` — arch-neutral primitives.
  - `src/wavefront.rs` — preamble/body/epilogue scheduler.
  - `src/emit_mega.rs` — **the megakernel emitter** (main entry: `emit_megakernel`). Items 2+3+4a: `collect_locals`, `write_region_locals`, `resolve_gmem`, `parallel_axis_names` / `serial_axis_name` threaded into `write_step`.
  - `src/emit.rs` — per-region sketch emitter (used by snapshot tests; `emit_mega` is the real target).
  - `src/emit_ops.rs` — per-tag intrinsic expansion table. `ExpandCtx` now carries `parallel_axes`, `serial_axis`, `gmem_bindings` with `.par(i)` / `.ser()` / `.gmem(canonical)` accessors. `local_refs(tag)` is the item-2 data source for region-local declarations.
  - `src/print.rs` — round-trip printer (used by tests, not the runtime emitter).
  - `csrc/ferrite_stencil_prelude.cuh` — **the megakernel prelude**. Real PTX for the 8 fixed-sig helpers landed in a9a1a8b59; load/store/mma bodies still `__trap()`. Types (`bf16`, `StencilFrag`, `Mbarrier`) + warpgroup macros + ambient scalars (eps / hidden_dim / token_ids — item 4 backlog).
  - `csrc/stencil_prelude_sm89.cuh` — older per-region-kernel prelude (unused by `emit_mega`; scaffolding from the 3b path).
  - `csrc/stencil_smoke_sm89.cu` — 3b hand-written smoke kernel (off-critical-path; kept for reference).
- Launch wrappers + GPU tests (off-critical-path): `vllm-rs/crates/ferrite-stencil-kernels/` — 3b launcher. Not wired to `emit_megakernel`.
- Lowering: `vllm-rs/crates/ferrite-forward-macro/src/lower_to_stencil.rs`
  - `lower_assignment` / `lower_assignment_partial` with transitive-closure control-edge derivation.
  - `lower_impl` match covers every `impl_lib` Impl name we currently see.
  - `stamp_gmem_bindings` + `bindings_for_impl` walk external FUF inputs and mint `w{id}_{layer}` / `t{tile}_{slot}` / `x_{kind}_{index}` identities. Multi-tile subgraphs pick the max tile id for output identities (fused_qkv_rope_cache = 4 tiles, fused_add_rms_norm = 2 — all covered).
  - `FERRITE_STENCIL_4B_TRACE=1` prints per-subgraph resolution.
- Macro drive: `vllm-rs/crates/ferrite-forward-macro/src/lib.rs` — parallel-pass telemetry + `emit_megakernel` output written to `/tmp/ferrite-stencil/<variant>-sm90.cu`.

## What's left to reach the finish line

The design doc's finish line is *efficient megakernel execution from the FUF* — comm/compute overlap, cross-subtile parallelism, real SM90a utilization, one launch per forward. From today's state:

### The first next step

**Thread a concrete gmem tile offset into every data-movement call site**, so the prelude helpers can drop the trap body for real cp.async / TMA / stg PTX. Byte-count plumbing is already done (commits `32e0690a0` + `843498fc9`); what remains is the *address* half.

The shape:
- Each `Node` with `role ∈ {Load, Store}` carries a `LoadAddr` with `SmallVec<AddrTerm>`. `AddrTerm::AxisStride { axis, stride }` is the common case — the tile offset in source elements is `Σ axis * stride` over every term.
- Add `fn addr_expr(region: &Region, node: &Node) -> String` in `emit_mega.rs` (or a new `emit_addr.rs`) that walks `node.addr.as_ref().expect(...)` and renders `"q_tile * 16384u + head_group * 16384u"` style expressions. `AxisModStride` adds `% modulus`; `AxisDivGather` renders `{table}[axis / divisor]` using the SmemLookup's source name.
- Rewrite every data-movement expansion in `emit_ops.rs` to emit `cp_async_128<SMEM_Q_BYTES>(smem_q, Q_gmem + {addr_expr})` instead of `cp_async_128<SMEM_Q_BYTES>(smem_q, Q_gmem, q_tile, head_group)`. The axes go away from the call; they live inside the expression.
- Prelude helpers drop axes from their signature: `template <uint32_t BYTES> __device__ void cp_async_128(bf16* smem, const bf16* gmem)`. Body loops `BYTES / 16` cp.async.ca.shared.global [smem + tid*16 + i*16*128], [gmem + tid*16 + i*16*128], 16 issues per thread (tid = threadIdx.x % 128).

Concretely one commit drops the trap for `cp_async_128`, a second does `stg_128`, a third does TMA (which needs an extra `CUtensorMap*` descriptor threaded as a kernel param). Each commit is mechanical once the address-expression refactor lands.

Expected LOC: 200-300 for the address refactor + 100 per real-PTX helper. Test churn is moderate — every snapshot assertion that currently includes axes in the call site needs updating to the `gmem + offset` form.

Edge cases to watch:
- **Cache stores**: `generic_cache_store` (store_k_cache / store_v_cache) does a block_table gather. `AddrTerm::AxisDivGather` covers it, but the gather-table name needs to be a kernel param, not a file-scope `__device__` placeholder. See item 6 below — ambient-scalar plumbing lands alongside this naturally.
- **Embedding**: `embed_gather` uses `token_ids[row]` as an indirection. The load targets `Embed_frag` (a register, not a smem local — per the current template), so `cp_async_128` is semantically wrong there. Fix: add a `smem_embed` staging buffer to `embed_region`'s `tile_consts` + `local_refs("load_embed_row")`, and lower `embed_gather` through the normal smem staging path.

### All remaining items

1. **Concrete fragment type + signature extensions + real PTX for load/store/compute.** See "first next step" above. Unblocks items 2-4.
2. **Fragment-arithmetic helpers.** `row_max`, `row_sum`, `exp2f_frag`, `frag_mul`, `frag_add`, `silu`, `rope_rotate`, `warp_reduce_sum_of_squares`. Straight-line math once `StencilFrag` is concrete — `exp2f_frag` is a per-lane `exp2f`, `row_max` is a warp-shuffle reduction.
3. **Named semaphores.** `sem_wait(name, depth)` — hash the string literal at compile time into a slot index, use a single `__shared__` counter array. Used by the scheduler for inter-warpgroup handshakes within a region.
4. **Pointer-form mbarriers.** Once the emitter tracks phase bits per barrier, lower to Hopper `mbarrier.try_wait.shared::cluster.b64` / `mbarrier.arrive.shared::cluster.b64`.
5. **Tile calibration + real shape walking.** `LowerHints` tile fields are `Default`-valued; `num_q_heads` / `num_kv_heads` default to 1 for qkv_rope. Pull from per-Impl calibrated sizes and `bounds`.
6. **Ambient scalar plumbing.** `eps`, `cap`, `rcp_cap`, `hidden_dim`, `blocks_per_tile`, `block_table`, `token_ids`, `hidden_stride` still live as file-scope `__device__` placeholders. Item 4b named them in the comments but didn't move them; item 4 backlog. Route through kernel params.
7. **SM89 solver policy.** Document that on SM89 targets the solver picks conventional per-op impls; megakernel path is SM90a+. Today we emit SM90 source for inspection regardless — fine as telemetry, but the runtime branch should split.
8. **Runtime wiring (Phase C).** Compile the emitted `.cu` into a static library (probably via `ferrite-cuda-builder`'s build.rs with per-model-variant symbol names), add a Rust FFI binding to `launch_mega_kernel`, route the per-arch forward through it on SM90a. Gate on compute capability.
9. **Ad-hoc H100 run (Phase D).** First proof the kernel launches. Needs (1) far enough along that the data-movement traps don't hit immediately, or (2) pointer-stub args. Correctness vs Python vLLM comes after.

Items 1-4 are the remaining substantive work before a first runtime trial. 5-9 are follow-on.

### If you want to start with something smaller

Phase B tasks that don't need the concrete fragment type:

- **Item 6 (ambient scalar plumbing)**: route `eps`, `cap`, `hidden_dim`, etc. through `mega_kernel`'s param list. Requires emitter to collect them and the macro drive to pass them. Small, mechanical, unblocks per-model configurability.
- **Item 5 (tile calibration)**: `num_q_heads` / `num_kv_heads` come from `bounds` today for the attention region but not for qkv_rope. Mechanical fix in `lower_impl`.
- **SM89 solver split (item 7)**: just a documentation + runtime-path cleanup.

## Common commands

```bash
# Run the full pipeline on a model crate (regenerates all its .cu files).
touch crates/ferrite-model-<arch>/src/lib.rs
cargo check -p ferrite-model-<arch>  2>&1 | grep "ferrite stencil"
# Archs: llama, gemma2, gemma3, qwen2, qwen3, mistral, granite, commandr

# With binding-resolution trace:
FERRITE_STENCIL_4B_TRACE=1 cargo check -p ferrite-model-llama 2>&1 | grep "^\[4b\]"

# Recompile one emitted megakernel. Both arches must succeed.
/usr/local/cuda-12.9/bin/nvcc -arch=sm_89 \
    -I crates/ferrite-stencil/csrc \
    -c /tmp/ferrite-stencil/<variant>-sm90.cu \
    -o /tmp/<variant>-sm89.o
/usr/local/cuda-12.9/bin/nvcc -gencode=arch=compute_90a,code=sm_90a \
    -I crates/ferrite-stencil/csrc \
    -c /tmp/ferrite-stencil/<variant>-sm90.cu \
    -o /tmp/<variant>-sm90a.o

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
- **`stamp_gmem_bindings` uses a per-Impl match statement** (same pattern). Multi-tile subgraphs resolve external inputs by dropping intra-subgraph tile refs; output identities use the max tile id in the subgraph. This works for every Impl we emit today but assumes FUF unroll assigns TileIds in topo order.
- **`emit_megakernel` produces `String`**, not a `TokenStream`. Because the whole point is that the emitter output goes into a `.cu` file the launcher reads, not into the macro's expansion token stream.
- **`StencilFrag` is opaque** — a 4-byte struct with assignment/arithmetic overloads that return itself unchanged. That's why the data-movement and compute helpers still trap: no concrete frag means no concrete byte count or register layout. The next commit replaces this.
- **Step 3b's `stencil_smoke_sm89.cu` is off the critical path.** It validates the earlier per-region per-SM89-kernel approach. The real emitter target is `emit_megakernel` producing one `__global__` for the whole forward; SM89 per-region codegen is not the runtime path (per design: on SM89, the solver picks conventional impls and doesn't engage the megakernel emitter at all).

## Memory pointer

`~/.claude/projects/-home-moosevan-vllm/memory/project_ferrite_stencil.md` tracks this status for future sessions via auto-memory; if you update the status here, skim that file too.
