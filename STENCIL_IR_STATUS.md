# Stencil IR — status & handoff

Dated 2026-04-19. Companion to `STENCIL_IR_DESIGN.md` (vocabulary freeze) and `STENCIL_IR_SKETCH.md` (struct shapes). This doc is the "what landed, what's next, where to look" layer; the other two stay untouched.

## Read order for a fresh session

1. `STENCIL_IR_DESIGN.md` — vocabulary (3 roles, 5 dep kinds, 3 clarifications). Frozen.
2. `STENCIL_IR_SKETCH.md` — §11 struct sketch that preceded the crate. Largely realised; see divergences at the bottom of this doc.
3. This doc — current state + next steps.

## Where we are

**Every real-model FUF (Llama/Gemma2/Gemma3/Qwen2/Qwen3/Mistral/Granite/CommandR, full-precision + marlin + bnb4 + gptq variants) now lowers to a complete SM90 megakernel source file.** Llama-3-8B: 227 regions / 290 control edges / ~6.6k lines of generated CUDA. Qwen3-0.6B: 339 regions / 450 edges / ~8.2k lines. Gemma-3-12B: 676 regions / 675 edges / ~15.3k lines. Every build writes per-variant `.cu` into `/tmp/ferrite-stencil/<variant>-sm90.cu` for inspection.

The pipeline runs end-to-end at compile time, with real content (no stubs, no pseudocode fallbacks) for every op the solver currently picks across those model crates.

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

Cumulative: **ferrite-stencil 38 lib + 7 integration green · forward-macro lowering 14 green.**

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
  - `src/emit_ops.rs` — per-tag intrinsic expansion table (31 non-attention + 7 attention tags populated)
  - `src/print.rs` — round-trip printer (used by tests, not by emitter)
  - `csrc/stencil_prelude_sm89.cuh` — SM89 prelude helpers (unused by `emit_mega` today; scaffolding from the earlier per-region 3b path)
  - `csrc/stencil_smoke_sm89.cu` — 3b hand-written smoke kernel (off-critical-path; kept for reference)
- Launch wrappers + GPU tests (off-critical-path): `vllm-rs/crates/ferrite-stencil-kernels/` — 3b launcher. Not wired to `emit_megakernel`.
- Lowering: `vllm-rs/crates/ferrite-forward-macro/src/lower_to_stencil.rs`
  - `lower_assignment` / `lower_assignment_partial` with transitive-closure control-edge derivation
  - `lower_impl` match covers every `impl_lib` Impl name we currently see; new impls fail to the `UnsupportedImpl` branch with a crisp error
- Macro drive: `vllm-rs/crates/ferrite-forward-macro/src/lib.rs` — parallel-pass telemetry + `emit_megakernel` output written to `/tmp/ferrite-stencil/<variant>-sm90.cu`

## What's left to reach the finish line

The design doc's finish line is *efficient megakernel execution from the FUF* — comm/compute overlap, cross-subtile parallelism, real SM90 utilization, one launch per forward. To get there from here:

1. **Real intrinsic lowering.** `emit_ops` emits pseudocode (`wgmma_mma_async`, `tma_load_2d`, `cp_async_128`, `mma_sync_accumulate`, …). These need to become actual PTX + mbarrier/TMA descriptor setup that `nvcc` can compile. Biggest chunk. Probably a prelude header (similar in spirit to the 3b `stencil_prelude_sm89.cuh` but rewritten for the SM90 path, since that's the target).
2. **gmem pointer plumbing in the kernel signature.** Today the signature ends with `/* gmem pointer plumbing: TODO */`. Need to walk every region's `FufOpRef` back to the FUF inputs/outputs and emit one `const bf16*` / `bf16*` per unique tensor, deduped by name. Impacts the launcher.
3. **Launcher glue** — a persistent-CTA runtime wrapper that calls the emitted `mega_kernel(scalars, ptrs)` exactly once per forward, with grid sized to `#SMs` and block sized to `20 warps = 640 threads`. Plus `g.Bar` storage alloc + zero.
4. **SM89 solver policy** — on SM89 targets the solver should naturally pick conventional per-op impls; the megakernel path is SM90+. Document and lock in — today we always emit SM90 source for inspection, which is fine as telemetry but shouldn't be the runtime path on SM89.
5. **Tile calibration** — `LowerHints.gemm_{m,n,k}_tile` / `token_tile` / `inter_tile` are `Default` values. Pull them from per-Impl calibrated sizes once those live on the `Implementation` trait (and route `fused_gate_up_silu_mul`'s tile sizes separately from `fused_gemm_bias`'s).
6. **Real shape walking** — `num_q_heads` and `num_kv_heads` passed to `qkv_rope_region` currently default to 1; the template shape is unaffected but the resource mapping at emit time wants the real head counts. Plumb from `bounds`.
7. **Compile + ad-hoc H100 run.** First proof the emitted source compiles under `nvcc` for SM90 and launches without crashing. Correctness vs hand-written comes after. (No H100 in L4 CI; this is an ad-hoc run the user triggers.)

Items 1 + 2 + 3 are the blocking set for a first runtime trial. 4–7 are follow-on.

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
