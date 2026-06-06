# SubtileTape → TkTape: TK 2.0 primitive decomposition design

Source: workflow `wbi32wl0g` (27 agents, 2.3M tokens, ~13 min). Inventoried 751 TK 2.0 primitives across `third_party/thunderkittens/include/`. Mapped 14 architectural SubOps to primitive sequences cited by header+line. This document is the design to implement; nothing here is code yet.

## Hard rules (INVIOLABLE)

1. Every Instr maps to ONE TK 2.0 primitive call. Plan §1 line 64-65 + `feedback_tk_player_one_call_per_arm`.
2. Every emitted `kittens::*` substring traces to a real header in `third_party/thunderkittens/include/`. No `kittens::ops::*` placeholders.  `feedback_tk_2_0_only`.
3. Every `kittens::*` in `tk_player.rs` lives in the `tk20::*` Rust sub-module.  `feedback_dogfood_tk20_rust`.
4. Each `tk_player.rs` match arm is ≤5 lines, ONE TK 2.0 call.  `feedback_tk_player_one_call_per_arm`.
5. Invariants (K-equality, softmax phase, page lifetime) are typed witnesses, not runtime checks.  `feedback_compile_time_or_garbage`.

## Per-SubOp Instr counts

| SubOp | Primitive Instrs | New variants |
|---|---:|---|
| MatmulTile | 8 | TmaExpect, TmaLoadTile, MbarrierWait, InitRtZero, WgmmaFenceAcc, WgmmaMmaAB_SmemSmem, WgmmaAsyncWait, StoreRegTileToShmem |
| RmsNorm | 10 | LoadVecGmemToSmem, ShTileMul (square), ShTileRowSum, ShVecMulScalar, ShVecAddScalar, LoadVecSmemToReg, RegVecRsqrt, StoreRegVecToShmem, ShTileMulRow, ShTileMulCol |
| SiluMul | 5 | ShTileMulScalar, ShTileExp, ShTileAddScalar, ShTileDiv, ShTileMul |
| Elementwise::Silu | 6 | LoadShmemToReg, RegTileNeg, RegTileExp, RegTileAddScalar, RegTileDiv, StoreRegTileToShmem |
| Elementwise::Mul | 1 | ShTileMul |
| Elementwise::Add | 1 | ShTileAdd |
| SumReduce | 1 | ShTileAdd (chained) |
| RopeRotateNeoX | 14 | LoadVecGmemToSmem ×2, LoadShmemToReg ×2, LoadVecSmemToReg ×2, RegTileMulCol ×4, RegTileSub, RegTileAdd, StoreRegTileToShmem ×2 |
| RopeRotateInterleaved | 9 | LoadShmemToReg ×3, RegTileApply (RopeRotatePair), RegTileMul ×2, RegTileAdd, StoreRegTileToShmem |
| RopeAppend | 14 | TmaLoadVec ×2, MbarrierWait, LoadShmemToReg, LoadVecSmemToReg ×2, RegTileMulCol ×2, RegTileApply (RopeRotateHalf), RegTileAdd, StoreRegTileToShmem, TmaStoreTile ×2, TmaStoreCommit |
| AttnDecode_Init | 3 | InitRvNegInfty, InitRvZero, InitRtZero |
| AttnDecode_Qkt | 11 | WgmmaFenceAcc, WgmmaMmaABt_SmemSmem, WgmmaAsyncWait, RegTileMulScalar, RegTileRowMaxAcc, RegTileSubRow, RegTileExp2, RegVecSub, RegVecExp2, RegVecMul, RegTileRowSumAcc |
| AttnDecode_Sv | 1 | WgmmaMmaAB_RegSmem |
| AttnDecode_Finalise | 2 | RegTileDivRow, StoreRegTileToShmem |

Total deduplicated new variants: ~46. Total Instrs in a Llama-3.2-1B forward (per layer × 16 layers + 33 norms + lm_head): **thousands**. The pipeline emits one TK 2.0 call per Instr.

## New typed-witness types (per `feedback_compile_time_or_garbage`)

| Type | Purpose | Why now |
|---|---|---|
| `RegTileId<ROWS, COLS, T, Layout>` (sealed, NonZeroU16 + phantom) | SSA register-tile handle. Const-generic shape/dtype/layout threaded through tk20:: bindings. | Almost every new Instr (Silu, RoPE*, AttnDecode_*, MatmulTile epilogue) consumes/produces register tiles. K-equality and row-vec-layout-equality become rustc errors instead of runtime asserts. |
| `RegVecId<LEN, T, RvLayout>` (sealed) | Sealed handle for register-vector accumulators (m_i, l_i, cos_rv, sin_rv, alpha). Layout (`ortho_l` vs `align_l`) is part of the type. | rt mul_col / sub_row / div_row layout requirements are discharged at the type level. Catches Hopper warp-shuffle layout mismatches. |
| `SoftmaxRowMaxAcc<P: MaxPhase>` / `SoftmaxRowSumAcc<P: SumPhase>` | Typestate threading the online-softmax accumulator through (Init → Qkt iter_0 → Qkt iter_1 → … → Sv → Finalise). | Phase-ordering invariant — you cannot row_max_acc against an uninitialized accumulator — is a typestate, not a debug_assert (`feedback_asserts_must_be_dead_code`). |
| `MbarrierPhase(u8)` (sealed; only via `Mbarrier::next_phase`) | Const-tracked phase bit for mbarrier wait. PhantomData ties to `MbarId` so the wait Instr proves it waits on the SAME mbarrier the matching expect/load_async armed. | MatmulTile and RopeAppend use TMA-load-then-wait; pairing an expect against the wrong mbar/phase deadlocks. |
| `ScalarF32(f32)` newtype | Inlined immediate scalar carried by *AddScalar / *MulScalar Instrs. Codegen emits a literal in the TK 2.0 call. | SiluMul (1.0), AttnDecode_Qkt (scale=log2(e)/sqrt(d)), RmsNorm (eps, inv_k) all known at lowering time. Avoids a runtime kernel-arg lookup. |
| `ApplyLambdaKind { RopeRotateHalf, RopeRotatePair, … }` | Sealed enum of supported `rt::apply` lambdas. Each variant maps to a named `__device__` functor struct in `ferrite_codegen_runtime/include` — NOT inline anonymous lambdas. | `tk_player.rs` has zero `format!()` of CUDA closure bodies (per `feedback_dogfood_tk20_rust`). New lambdas: enum variant + functor struct. |
| `AccPolicy { Reset, Accumulate }` / `FencePolicy { External, Internal }` | Sealed booleans converted to template params on mma_AB / mma_ABt. | MatmulTile, AttnDecode_Qkt, AttnDecode_Sv all parameterize fence/accumulate. Sealed enum + tape-level pairing proof catches missing-fence Hopper corruption. |
| `GroupWidth<const N: usize>` (sealed N ∈ {1, 4}) | Const-generic width binding from `WarpRole`. | Emitting mma_AB under a per-warp role is a deadlock; encoding N as const-generic makes that combination a compile error. |
| `TileTypeId` (phantom-typed, exposes const ROWS, COLS, T) | Sealed shape+dtype carrier used by `TmaExpect` to pre-compute the expected transaction byte count at codegen. | `TmaExpect` at `ops/group/util/tma.cuh:29` takes the tile type as a template parameter, not a runtime byte count. |
| `Coord4 = [u32; 4]` newtype + `CoordExpr { Const(Coord4), RuntimeRow(ScalarSlotId, …) }` | Typed TMA coordinate carrier. RuntimeRow handles position-dependent rows (RopeAppend cos/sin row = p). | RopeAppend lowering cannot produce a TmaLoadVec at runtime row without CoordExpr::RuntimeRow. |

## `lower_subtile_tape_to_tk_tape` signature (target)

```rust
pub fn lower_subtile_tape_to_tk_tape(
    tape: &SubtileTape,
    layout: &WarpRoleLayout,
    pages: &PagePool,
    mbarriers: &MbarrierTable,
    softmax_state: Option<&SoftmaxAccumulatorBinding>,
) -> Result<TkTape, LoweringError>;

// where
//   WarpRoleLayout: const map WarpRole -> GroupWidth<N>; consumed at
//                   lowering to choose tk20::group_for.
//   PagePool: SSA arena for SmemTileId / SvId / RegTileId / RegVecId;
//             allocates fresh ids per Instr def-site, proves last-use
//             before reuse.
//   MbarrierTable: typed (MbarId, MbarrierPhase) pairing; expects
//                  matching expect/load/wait sequences in tape order.
//   SoftmaxAccumulatorBinding: optional. When the SubOp chain contains
//                              AttnDecode_*, this carries the
//                              SoftmaxRowMaxAcc / SoftmaxRowSumAcc
//                              typestate cursors so the loop-carried
//                              phase is threaded across (Init → Qkt
//                              iter → Sv → Finalise) without escaping
//                              into ad-hoc RegVecId aliasing.
//
// LoweringError: ShapeMismatch, KEqualityViolation,
//                MbarrierPhaseMismatch, GroupWidthMismatch,
//                MissingFencePairing, RegisterTileLifetime,
//                UnsupportedApplyLambdaKind, RuntimeScalarRequired.
//                No catch-all "Other" arm.
```

## Implementation order

1. **Sealed-handle infrastructure**: `RegTileId`, `RegVecId`, `ScalarF32`, `AccPolicy`, `FencePolicy`, `GroupWidth<N>`, `TileTypeId`, `Coord4`. Foundational.
2. **`Elementwise::Mul`** (1 Instr, `ShTileMul`). Smallest end-to-end test of the lowerer with no register-tile lifetime risk.
3. **`Elementwise::Add`** + **`SumReduce`** (1 Instr each). Reuses (2) plumbing; proves dst-aliasing rules in PagePool.
4. **`SiluMul`** (5 Instrs, all shared-tile). No register-tile lifetime questions yet. Exercises `ScalarF32` plumbing.
5. **`RmsNorm`** (10 Instrs, shared+register-vec). Adds `RegVecId` + the rsqrt detour. First mixed shmem/reg lowering.
6. **`Elementwise::Silu`** (6 Instrs, register-resident). First all-register chain; stress-tests `RegTileId` SSA.
7. **`RopeRotateNeoX`** (14 Instrs). Larger register-tile set (8 live RtIds). No `apply`, no TMA — gates on (1)+(6) only.
8. **`RopeRotateInterleaved`** (9 Instrs). First user of `ApplyLambdaKind` (RopeRotatePair). Locks in the named-functor pattern for `apply`.
9. **`MatmulTile`** (8 Instrs). First TMA + WGMMA path. Foundation for AttnDecode_Qkt.
10. **`RopeAppend`** (14 Instrs). Builds on (8) for Apply and (9) for TMA; resolves `Coord4::RuntimeRow` + kv-slot coord plumbing.
11. **`AttnDecode_Init`** (3 Instrs). Introduces `SoftmaxRowMaxAcc` / `SoftmaxRowSumAcc` typestate construction.
12. **`AttnDecode_Qkt`** (11 Instrs). First user of typestate transitions, accumulating row_max/row_sum overloads, `WgmmaMmaABt_SmemSmem`.
13. **`AttnDecode_Sv`** (1 Instr + the prerequisite `RegTileCopyConvert` from Open Question 5). Unblocks the full decode path.
14. **`AttnDecode_Finalise`** (2 Instrs). Wraps the loop with `RegTileDivRow` + `StoreRegTileToShmem`.

## Resolved design decisions

1. **Lifetime model: SSA**. Every Instr that produces a `RegTileId` / `RegVecId` mints a fresh id; the lowerer walks the tape forward proving last-use. Llama compute is acyclic per-op; arena+Drop adds an Instr type for no flexibility win.
2. **rsqrt for register-vec**: use `kittens::group<N>::unary_op<kittens::base_ops::rsqrt, RvType>(dst, src)` from `ops/group/register/vec/maps.cuh:16` (generic unary-op template) + `common/base_ops.cuh:218` (the rsqrt op struct). No shared-vec rsqrt exists — RmsNorm permanently routes its rsqrt step through register-vec. The Instr is `RegVecUnaryRsqrt { src: RegVecId, dst: RegVecId, role: WarpRole }`.
3. **`ApplyLambdaKind`: named `__device__` functor structs** in a new `ferrite_codegen_runtime/include/tk_apply_kinds.cuh` header we ship alongside the codegen. Each `ApplyLambdaKind` variant maps 1:1 to a struct with a `static __device__ inline T op(const T& src, const T& partner)` method. `tk_player.rs` emits `kittens::group<N>::apply<ApplyKind::RopeRotatePair>(dst, src, partner_smem)`. No inline closure bodies in the player.
4. **WGMMA m==1: pad `act_smem` to 4 tile rows in `PagePool`**. One row of padding waste vs a parallel warp-scope mma Instr variant — padding keeps a single MMA path through MatmulTile and AttnDecode_Qkt.
5. **P_block dtype on AttnDecode_Sv**: add `RegTileCopyConvert { src: RegTileId, dst: RegTileId, role: WarpRole }` Instr (47th variant) mapping to `kittens::group<N>::copy` in `ops/group/register/tile/conversions.cuh`. Inserted between Qkt and Sv to convert fp32 P → bf16. Non-negotiable: mma_AB requires A.T == B.T.
6. **`CoordExpr::RuntimeRow`**: yes. Required by RopeAppend (cos/sin row = position p). The same `decode_position` kernel-arg slot RopeRotate already mints serves as the runtime row.
7. **`block_table` indirect lookup**: orchestrator (host) precomputes the kv-slot Coord4 per token before launch and passes it as a kernel arg. The TMA descriptor stays static (one descriptor per cache layer); the Coord4 read at the call site comes from a kernel-arg slot. No non-TK `ScalarLoadFromGmem` Instr — that would violate inviolable rule 2.
8. **`SoftmaxAccumulatorBinding`** lives in `lower_subtile_tape_to_tk_tape`'s `LoweringState` across the OpenLoop body. Per-AttnDecode-loop binding is a lowering concern, not a SubtileTape field. The `SoftmaxRowMaxAcc<P>` / `SoftmaxRowSumAcc<P>` typestate cursors are owned by `LoweringState::softmax_cursor: Option<SoftmaxAccumulatorBinding>` set on `SubOp::AttnDecode` entry, threaded through OpenLoop body, consumed at CloseLoop's `Finalise`-deferred drain.
9. **`ScalarF32` only**. Add `ScalarSlotId` when something actually needs runtime-variable scalars; YAGNI.

The 47-variant Instr set including `RegVecUnaryRsqrt` (generic unary-op pattern) and `RegTileCopyConvert` (fp32→bf16 for AttnDecode_Sv) is now closed. Implementation can proceed.

## Status (2026-06-06)

- **Substrate**: clean. Every invented `kittens::ops::*` reference nuked. 67 unit + 12 doctests green. `lower_compute` panics on any architectural SubOp until expansion lands (no fake `.cu` can escape).
- **Cache**: `~/.cache/cudaforge/megakernels/tk_decode_full_*.cu` deleted.
- **Next**: implementation step 1 (sealed-handle infrastructure) per the order above.
