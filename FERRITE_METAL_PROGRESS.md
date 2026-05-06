# Ferrite Metal Port - Progress Tracker

## Overview
Port ferrite's compile-time DSL → kernel compilation from CUDA to Metal for Apple Silicon.

**Branch:** `ferrite-metal` (stems from `ff-interpreter`)  
**Worktree:** `/Users/nickm/git/vllm/.claude/worktrees/ferrite-metal`

## Phase Status

### Phase 1: Foundation ✅ COMPLETE
**Goal:** Set up Metal-specific crates and device profiles  
**Duration:** 1 week  
**Status:** ✅ All items complete, tests passing

- ✅ Create `ferrite-metal-targets` crate with M1/M2/M3/M4 profiles
- ✅ Create `ferrite-metal-kernels` crate with Metal-rs bindings
- ✅ Create `ferrite-metal-impl-lib` crate with MetalImplementation trait
- ✅ Implement device detection and capability queries
- ✅ Write first Metal shader (RMSNorm) and verify compilation
- ✅ All crates compile, unit tests pass

### Phase 2: Solver Integration ✅ COMPLETE
**Goal:** Integrate Metal implementations into ferrite's solver  
**Duration:** 1-2 weeks  
**Status:** ✅ All items complete, tests passing, measured costs integrated

- ✅ 2.1: Extend TargetProfile to support both CUDA and Metal backends
- ✅ 2.2: Create Metal implementation bridge (MetalRmsNormImpl)
- ✅ 2.3: Register Metal implementations in starter_library()
- ✅ 2.4: Add solver tests for Metal implementation selection
- ✅ 2.5: Create ferrite-metal-cost-sweep crate for microbenchmarking
- ✅ 2.6: Run benchmarks on M1 Max hardware and populate cost tables

**Key Achievement:** Phase 2 fully complete with measured cost data from real M1 Max hardware. Solver now uses empirical measurements instead of analytical estimates for Metal targets.

### Phase 3: Runtime Infrastructure ✅ COMPLETE
**Goal:** Metal runtime infrastructure (streams, allocators, device management)  
**Duration:** 2-3 weeks  
**Status:** ✅ ALL ITEMS COMPLETE - Full Metal execution pipeline verified

**Note**: No Metal-specific codegen needed - solver already emits backend-agnostic `Instruction<W>` lists

- ✅ 3.1: ICB Architecture Decision
  - Decided on multi-launch ICB approach (not single persistent kernel)
  - ICB pre-records all dispatches at init time
  - Single `execute_icb()` call per forward pass
  - Architecture follows optimization suggestions (§4, §9) - ICB for dispatch amortization

- ✅ 3.2: Implement MetalStream and command buffer management
- ✅ 3.3: Port memory allocator to Metal's MTLBuffer
- ✅ 3.4: Metal-specific error handling
- ✅ 3.5: Integration tests with real Metal execution

### Phase 4: Kernel Library & Implementation Registration ✅ COMPLETE
**Goal:** Create Metal kernels + Implementation trait impls + ICB recording infrastructure  
**Duration:** 3-4 weeks  
**Status:** ✅ ALL ITEMS COMPLETE

- [x] 4.1: Attention kernels (basic, paged, multi-head, optimized) ✅
- [x] 4.2: Fused kernels (Add+RMSNorm, SwiGLU) + GEMM (MPS) ✅
- [x] 4.3: Activation functions (SiLU, GELU variants, FatReLU) ✅
- [x] 4.4: AWQ quantization (4-bit dequantization + GEMM integration) ✅
- [x] 4.5: Implementation trait impls for solver registration ✅
  - 18 critical path operations implemented and registered
  - Modular structure in `src/metal/` directory
  - All tests passing (78 total)
- [x] 4.6: ICB recording infrastructure ✅
  - `RecordingContext` with ICB management
  - `record_compute_dispatch()` for ICB command recording
  - `inheritPipelineState=true` for Apple Silicon compatibility
  - Helper functions: `dispatch_1d()`, `dispatch_2d()`
  - Module structure: `rmsnorm`, `gemm`, `attention`, `fused`

### Phase 5: Worker Pool + Lowering + Specialized Pipelines 🔄 IN PROGRESS (5.A–5.C done)
**Goal:** End-to-end Metal forward via the worker-pool architecture finalized in `FERRITE_METAL_ARCHITECTURE.md` (2026-05-06).
**Status:** 5.A, 5.B, 5.C complete; 5.D–5.G + 5.6 remaining.

**Architecture pivot (2026-05-06).** The earlier 5.1–5.5 plan (`MetalExecutor` walks the tape and calls `record_to_icb()` methods on `Instruction<W>`) was replaced. Reasons:
- A single `MetalExecutor` per model has no concurrency story (spec-decode draft+verify, prefill/decode overlap, multi-stream serving need bounded concurrency without N-fold weight duplication).
- Hanging `record_to_icb()` off `Instruction<W>` mixes Metal-specific concerns (pipeline ids, dispatch shapes, slot resolution) into the shared frontend enum.
- The `inheritBuffers=true` slot↔encoder-binding scheme in `FERRITE_METAL_ICB_INHERIT_BUFFERS.md` doesn't scale past Metal's 31-binding limit.

The pivot:
1. Explicit lowering pass — `From<&[Instruction<W>]> for LoweredMetalTape` translates one bucket's tape into a buffer-pointer-free lowered tape (pipeline keys, dispatch shape, scalar constants, slot ids only). Computed once per `(model, bucket)`; shared across workers via `Arc`.
2. `MetalWorkerPool` — growable from 1 to `max_workers = floor((device_total - weights - misc) / per_worker_arena)`. Each worker owns one arena + one fully-baked ICB per bucket. `pool.checkout()` blocks under contention; `forward()` is `bind_inputs → executeCommandsInBuffer → checkin`.
3. Function-constant pipeline specialization — every `(model variant, bucket, kernel)` gets its own `MTLComputePipelineState` with `M`, `hidden_size`, `num_heads`, `head_dim`, `eps`, `rope_theta`, etc. baked in via `MTLFunctionConstantValues`. Eliminates the runtime `constants` buffer; unlocks loop-unrolling / bounds-check elimination on hand-rolled kernels at small buckets.

#### Phase 5.A: Lowering pass ✅ COMPLETE (2026-05-06)
Pure-CPU translation `&[Instruction<W>] → LoweredMetalTape<W>`. Statically unrolls `Loop`. Drops metadata-only instructions (`Reshape`/`Alias`/`Free`). Covers TinyLlama-1.1B critical path: `Embed`, `RmsNorm`, `FusedAddRmsNorm`, `Gemm`, `FusedGateUpSiluMul`, `RopeAppend`, `AttentionViaCache`, `AttentionPrefillContiguous`, `Add`, `ScalarMul`. Other variants surface as `LoweringError::UnsupportedVariant { index, variant_type }`.

To unblock compilation on Apple Silicon (no nvcc), this phase also extended feature flagging:
- `ferrite-kernels/Cargo.toml`: `cudarc` is now `optional = true`, gated on `cuda` feature. Layer struct *types* (`RmsNorm`, `LinearLayer`, `Embedding`, `MarlinLinear`, `Bnb4bitLinear`, `Fp8AnyLinear`, `LayerNorm`, `Linear`, the MoE structs) compile without `cuda`; their cudarc-using `impl` blocks and helper fns are individually `#[cfg(feature = "cuda")]`-gated.
- `ferrite-vision`: now an optional dep of `ferrite-forward`, pulled in only via `cuda` feature.
- `ferrite-forward/src/instr.rs`: removed file-level `#![cfg(feature = "cuda")]`. The `Instruction<W>` enum, `CanonicalParams` trait, `WtFn`/`CosSinFn` aliases compile under either `cuda` or `metal`. `eval`/`run`/`run_backbone`/`InterpreterCtx`/`debug_dump`/helpers are individually cuda-gated.
- `ferrite-forward/Cargo.toml`: new `metal = []` feature.

This gating is acknowledged technical debt — when parallel Metal weight types (or generic `Linear<B>`-style backend params) land, the `#[cfg(feature = "cuda")] impl …` annotations lift in one pass.

**Files:**
- `crates/ferrite-forward/src/interpreter/metal/lowered.rs` — `LoweredMetalTape<W>`, `LoweredCommand<W>`, `KernelId`, `DispatchShape`, `Binding<W>` (ArenaSlot / Weight / Runtime), `WeightBundleKind<W>`, `WeightTensor`, `RuntimeBindingKind`, `LoweringError`.
- `crates/ferrite-forward/src/interpreter/metal/lowering.rs` — `pub fn lower<W: CanonicalParams>(instructions, bucket_m, num_arena_slots) -> Result<LoweredMetalTape<W>, LoweringError>`.
- `crates/ferrite-forward/src/interpreter/metal/mod.rs` — re-exports.
- Deleted: `interpreter/metal.rs` (Err-stub `MetalExecutor` skeleton), `interpreter/metal_tests.rs` (synthetic-tape `MockRmsNormWeight` tests, per `feedback_no_reinvent_testing.md`).

**Verification:** `cargo check -p ferrite-forward --no-default-features --features metal` ✓ on darwin without CUDA toolkit.

#### Phase 5.B: Function-constant pipeline cache ✅ COMPLETE (2026-05-06)
Rewrote hand-rolled MSL shaders to declare layer-independent params (`hidden_size`, `eps`, `num_heads`, `head_dim`, `intermediate_size`, …) as `[[function_constant(N)]]`. Built `SpecializedPipelineCache` keyed on `(library, kernel, function-constant bag)`; pipelines constructed via `MTLFunctionConstantValues`. Removes the runtime `constants` buffer and saves a binding slot.

**Sub-status:**
- ✅ 5.B.1 — Audited every MSL shader for runtime-constant uses; mapped each to a `CanonicalParams` field, bucket-derived value, or per-instruction extra (eps / scale).
- ✅ 5.B.2 — `SpecializedPipelineCache` + `SpecializedPipelines` glue layer (function-constant index assignments per `KernelId`). 3 device-bound + 5 CPU-only tests pass.
- ✅ 5.B.3 — `rmsnorm_f16_specialized` + `fused_add_rmsnorm_f16_specialized` symbols added; legacy non-specialized symbols kept in place so Phase 4.6 ICB recorders keep compiling.
- ✅ 5.B.4 — `fused_gate_up_silu_mul_f16_specialized` added. Attention kernels deferred to 5.C (single-Q-token kernel shape was wrong for the lowering's multi-Q-token dispatch).
- ✅ 5.B.5 — Device-bound smoke test `rmsnorm_pipeline_builds_and_caches` builds 6 pipelines at buckets {1, 8} for {RmsNorm, FusedAddRmsNorm, FusedGateUpSiluMul} on TinyLlama-1.1B params; asserts cache hit on repeat.

#### Phase 5.C: `MetalWorker` ✅ COMPLETE (2026-05-06)
Allocates the per-worker arena (one buffer per colored slot, sized for the max bucket). Walks the lowered tape, resolves `Binding::ArenaSlot` against `arena[slot]` and `Binding::Weight` against `MetalModelMeta`, records one ICB per bucket using the specialized pipelines from 5.B. Bucket plan is a `Vec<BucketStep>` where each step is either a same-pipeline ICB run or an MPS GEMM dispatch.

**Sub-status:**
- ✅ 5.C scaffolding (commit `ed2862483`) — `MetalModelMeta<W>`, `RuntimeBindings`, `MetalWorker<W>`, `BucketBaking`. Per-bucket ICB recording with `inheritPipelineState=true`; segment partitioning so a single `executeCommandsInBuffer` call only runs commands sharing one pipeline.
- ✅ 5.C.4 (commit `1beee7599`) — `attention_via_cache_f16_specialized` (paged decode) and `attention_prefill_contiguous_f16_specialized` (causal multi-Q-token, contiguous Q/K/V) MSL kernels. Function-constant indices: `0..3` shared (`HEAD_DIM`/`NUM_Q_HEADS`/`NUM_KV_HEADS`/`ATTN_SCALE`), `4..5` decode-only (`BLOCK_SIZE`, `MAX_BLOCKS_PER_SEQ`), `6` prefill-only (`PREFILL_TILE_Q`); disjoint to avoid an MSL same-index/different-name collision in one compilation unit. `KernelExtras` extended with `block_size`, `max_blocks_per_seq`, `prefill_tile_q`. Reference logic is correct but unoptimized — FlashAttention-style blocking deferred to 5.6; capped at `MAX_SHARED_LOGITS = 4096` floats per threadgroup.
- ✅ 5.C.5 (commit `6209efbaa`) — `KernelId::Gemm` routed via MPS, interleaved with ICB segments. Lowering carries M/N/K through `LoweredCommand::gemm_dims: Option<GemmDims>`. Worker plan changed from `Vec<ExecSegment>` to `Vec<BucketStep>` (`Icb { pipeline, range }` | `Gemm { a, b, c, m, n, k }`). `run_bucket(&CommandBufferRef)` manages compute encoder lifecycle: adjacent ICB steps reuse the encoder, a GEMM step ends it and the next ICB step opens a fresh one. New low-level helper `gemm::encode_gemm_into_command_buffer` (no `MetalStream`, caller owns commit). `WorkerError::OpaqueGemmNotYetRouted` removed.

**Tests at 5.C close:** 18 `ferrite-forward` unit tests + 3 `specialized_pipeline_cache` tests pass. Includes `worker_builds_and_segments_coalesce`, `worker_records_attention_via_cache`, `attention_pipelines_build_and_cache`, `worker_routes_gemm_step`, `worker_interleaves_gemm_with_icb`.

#### Phase 5.D: `MetalWorkerPool` 🔜 PLANNED
Growable, capped, semaphore-bounded checkout/checkin. RAII guard. `max_workers` derived from device memory at construction time.

#### Phase 5.E: `forward()` 🔜 PLANNED
Pick bucket from `num_tokens`, checkout worker, bind input/position buffers, `encoder.executeCommandsInBuffer(this bucket's ICB)`, checkin.

#### Phase 5.F: Macro emission 🔜 PLANNED
`#[forward]` emits `MetalWorkerPool::for_<model>()` constructor alongside CUDA's `try_load`. Lowering happens at constructor time.

#### Phase 5.G: Correctness wiring 🔜 PLANNED
Hook into existing `cpu_golden::*` per-op references and `vllm-e2e` golden framework — same path CUDA uses. No bespoke Metal-only test scaffolding (per `feedback_no_reinvent_testing.md`).

### Phase 5.6: TinyLlama-1.1B golden 🔜 PLANNED
Pass the existing TinyLlama-1.1B golden under `--features metal` on M1+. Profile the function-constant specialization win at small buckets vs. an unspecialized control build.

See `FERRITE_METAL_ARCHITECTURE.md` for the source-of-truth design and `FERRITE_METAL_PHASE5_PLAN.md` for prior context (Phase 5.6+ steps still valid; 5.1–5.5 superseded).

### Phase 6: Production Readiness 🔜 PLANNED
**Goal:** Polish and prepare for production use  
**Duration:** 1-2 weeks  
**Status:** Not started

- [ ] 6.1: Add comprehensive error messages and diagnostics
- [ ] 6.2: Write user documentation for Metal backend
- [ ] 6.3: Create CI/CD pipeline for Metal builds
- [ ] 6.4: Performance regression testing
- [ ] 6.5: Performance comparison vs MLX baseline
- [ ] 6.6: Final code review and merge to main

## Timeline
- **Total Duration:** 8-12 weeks
- **Start Date:** 2024-12-XX
- **Phase 1 Complete:** 2024-12-XX ✅
- **Phase 2 Complete:** 2026-05-05 ✅
- **Phase 3 Complete:** 2026-05-05 ✅
- **Phase 4 Complete:** 2026-05-06 ✅ (including 4.6 ICB infrastructure)
- **Phase 5 Started:** 2026-05-06 (initial skeleton; superseded by architecture pivot same day)
- **Phase 5.A Complete:** 2026-05-06 ✅ (lowering pass + feature-flag refactor)
- **Phase 5.B Complete:** 2026-05-06 ✅ (function-constant pipelines for rmsnorm/fused_add_rmsnorm/silu)
- **Phase 5.C Complete:** 2026-05-06 ✅ (worker scaffolding + attention rewrite + MPS GEMM routing)
- **Target Completion:** 2025-03-XX

## Test Results Summary
```
Unit Tests (ferrite-metal-kernels lib):     14 passed
Integration Tests:                           7 passed
Attention Tests (basic):                     2 passed, 1 ignored
Attention Tests (paged):                     2 passed
Attention Tests (multi-head):                2 passed
Attention Tests (optimized):                 2 passed
Fused Kernels Tests:                         6 passed
GEMM Tests:                                  2 passed
Activation Tests:                            4 passed
AWQ Tests:                                   7 passed
RoPE Tests:                                  4 passed
Reshape Tests:                               2 passed
Embed Tests:                                 4 passed
Mul Tests:                                   4 passed
BiasAdd Tests:                               4 passed
TanhSoftCap Tests:                           4 passed
Sub Tests:                                   4 passed
Metal Codegen Tests:                         4 passed
Total:                                      78 passed, 1 ignored
```

## Key Decisions
1. **Multi-launch architecture:** Using ICB (Indirect Command Buffers) instead of single persistent kernel
2. **Cost model strategy:** Hybrid analytical + measured (memory bandwidth for memory-bound ops)
3. **Frontend reuse:** 80% of ferrite's frontend pipeline unchanged, backend-specific solver/schedule/codegen
4. **Backend abstraction:** TargetProfile extended to support both CUDA and Metal via Backend enum
5. **Hardware benchmarking:** Running on M1 Max (32 cores, 400 GB/s) provides realistic cost data
6. **Directory structure:** Following CUDA convention with `profiles/cost_*.csv` layout
7. **Command buffer management:** MetalStream abstraction provides CUDA-stream-like semantics for Metal
8. **Memory management:** Buffer pooling with size-based bucketing reduces allocation overhead
9. **Synchronization:** Proper wait_for_completion with last_committed tracking
10. **Attention strategy:** Start with basic single-head, incrementally add features (paging, GQA, etc.)
11. **Fused kernels:** Prioritize memory-bandwidth optimizations (residual+norm, SwiGLU) per FERRITE_METAL_OPTIMIZATION_SUGGESTIONS.md
12. **Modular implementation structure:** Separate files per kernel category in `src/metal/` for maintainability
13. **Runtime interpreter:** Phase 5 walks instruction tape at init time, calls Phase 4.6's `record_*()` methods to populate ICB

## Recent Progress (2026-05-06)

### Phase 4.6: ICB Recording Infrastructure - ✅ COMPLETE

**Achievement: ICB Recording Infrastructure** ✅

Created complete ICB recording infrastructure in `ferrite-metal-kernels/src/instruction_executor/`:

**Core Infrastructure:**
- `RecordingContext` - ICB management with `inheritPipelineState=true`
- `record_compute_dispatch()` - Records compute commands into ICB
- Helper functions: `dispatch_1d()`, `dispatch_2d()`
- Module structure: `rmsnorm`, `gemm`, `attention`, `fused`

**Key Technical Details:**
- Uses `inheritPipelineState=true` for Apple Silicon compatibility
- Pipeline state set on encoder, NOT on ICB commands
- All ICB commands share pipeline state from encoder
- Avoids crashes on Apple Silicon

### Phase 5: Architecture pivot + Phase 5.A landing - ✅ (2026-05-06)

The earlier 5.1–5.2 `MetalExecutor` skeleton was deleted as part of the pivot. Replaced with:
- `crates/ferrite-forward/src/interpreter/metal/{mod,lowered,lowering}.rs` — pure-CPU `lower()` pass with TinyLlama critical-path coverage.
- Feature-flag refactor across `ferrite-kernels`, `ferrite-vision`, `ferrite-forward` so `Instruction<W>` compiles under either `cuda` or `metal` (Apple Silicon builds no longer need nvcc).

Architecture decisions captured in `FERRITE_METAL_ARCHITECTURE.md` (now the source of truth). `FERRITE_METAL_ICB_INHERIT_BUFFERS.md` superseded; `FERRITE_METAL_PHASE5_PLAN.md` 5.1–5.5 superseded but 5.6+ steps still valid.

### Phase 5.B + 5.C landed - ✅ (2026-05-06)

Specialized pipelines for the hand-rolled-kernel set (rmsnorm / fused-add-rmsnorm / fused gate-up SiLU) shipped as 5.B; the worker, multi-Q-token attention rewrite, and MPS GEMM routing shipped as 5.C in three commits the same day:
- `ed2862483` — 5.C scaffolding (`MetalWorker`, `MetalModelMeta`, `RuntimeBindings`, segmented exec plan).
- `1beee7599` — 5.C.4 (`attention_via_cache_f16_specialized` + `attention_prefill_contiguous_f16_specialized`, function-constant specialized; `KernelExtras` extended with `block_size`/`max_blocks_per_seq`/`prefill_tile_q`).
- `6209efbaa` — 5.C.5 (MPS GEMM interleaved with ICB segments; `BucketBaking::steps: Vec<BucketStep>` (Icb | Gemm); `run_bucket(&CommandBufferRef)` manages compute encoder lifecycle across GEMM boundaries; new `gemm::encode_gemm_into_command_buffer` helper; `WorkerError::OpaqueGemmNotYetRouted` removed).

**Next Steps:**
1. Phase 5.D: `MetalWorkerPool` (growable, capped, semaphore-bounded checkout/checkin; `max_workers = floor((device_total - weights - misc) / per_worker_arena)`)
2. Phase 5.E: `forward()` (bucket pick → checkout → bind inputs → `run_bucket` → checkin → commit cmdbuf)
3. Phase 5.F: `#[forward]` macro emits `MetalWorkerPool::for_<model>()` alongside CUDA's `try_load`
4. Phase 5.G: Wire `cpu_golden` per-op + `vllm-e2e` end-to-end (no bespoke Metal-only test scaffolding)
5. Phase 5.6: TinyLlama-1.1B golden under `--features metal` on M1+; profile function-constant specialization win

## Notes
- Phase 1-4: ✅ COMPLETE - All foundation work done (including 4.6 ICB infrastructure)
- Phase 5.A–5.C: ✅ COMPLETE - Lowering, function-constant cache, worker (incl. attention + GEMM)
- Phase 5.D–G + 5.6: 🔜 PLANNED - Pool, forward, macro emission, e2e wiring, golden
- 78 Phase 1-4 tests + 18 ferrite-forward unit tests + 3 cache tests passing
- `cargo check -p ferrite-forward --no-default-features --features metal` ✓ on darwin
- Feature gates in `layers.rs`/`layers_moe.rs` are scaffolding — revert when parallel Metal weight types land
- Multi-Q-token attention kernels are correctness-first reference impls; FlashAttention-style blocking + perf tuning is Phase 5.6 work