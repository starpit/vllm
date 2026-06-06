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

## Blocking open questions (need answers before steps 5+)

1. **`RegTileId` / `RegVecId` lifetime model**: SSA-with-last-use vs arena-named with explicit `Drop`. Blocks ALL register-resident SubOps (Silu, RoPE*, AttnDecode_*). Recommendation: SSA per Instr-output, lowerer proves liveness; mandatory before lowering AttnDecode_Init.
2. **rsqrt over a shared vector in TK 2.0**: header inventory only lists rsqrt as rt/rv unary (`common/base_ops.cuh:219`). Direct grep needed before lowering RmsNorm. If no sv-rsqrt exists, RmsNorm permanently splits step 6 into LoadVecSmemToReg + RegVecRsqrt + StoreRegVecToShmem (already in the decomposition).
3. **`ApplyLambdaKind` closure-body emission**: need named `__device__` functor structs in `ferrite_codegen_runtime` so `tk_player.rs` never inlines `format!("[](...){...}")`. Blocks `RopeRotateInterleaved` + `RopeAppend`.
4. **WGMMA decode-path m == 1**: A.M_dim must equal 4*TILE_ROW_DIM at `warpgroup.cuh:199`. Blocks AttnDecode_Qkt and MatmulTile decode. Decision: pad act_smem to 4 tile rows in `PagePool` (preferred) vs route m==1 through warp-scope `mma_AB` at `warp.cuh:583` (forces a new Instr variant).
5. **P_block dtype on AttnDecode_Sv**: mma_AB requires `A.T == B.T` (bf16). Softmax produces fp32 P; need a separate `RegTileCopyConvert` Instr (mapping to `ops/group/register/tile/conversions.cuh`) inserted between Qkt and Sv. **Add as a 47th variant before lowering AttnDecode_Sv.**
6. **`Coord4 / CoordExpr::RuntimeRow`** support: required by RopeAppend (cos/sin row = p). Without it, RopeAppend lowering cannot produce a TmaLoadVec at runtime row. Blocks RopeAppend.
7. **`block_table` indirect lookup for kv-slot coord in RopeAppend**: TK 2.0 has no primitive for indirect-coord TMA. Either (a) host precomputes kv_slot_coord per token and passes it as a Coord4 array baked into the descriptor, or (b) we add a non-TK `ScalarLoadFromGmem` Instr (which would be NOT a TK 2.0 primitive — violates inviolable rule 2). **Must be resolved with the orchestrator before RopeAppend goes live.**
8. **Softmax accumulator binding across SubOp boundaries**: `SoftmaxRowMaxAcc<P>` / `SoftmaxRowSumAcc<P>` typestate must survive `lower_dag_to_tape`'s chain split. Confirm `SubtileTape` exposes a chain-state slot we can attach the typestate cursor to; if not, add `SoftmaxAccumulatorBinding` to `SubtileTape` as a sealed field.
9. **Inlined `ScalarF32` vs `ScalarSlotId`**: locking in ScalarF32-only blocks any future runtime-variable scalar (e.g. dynamic ALiBi slope). Low risk, but record the decision.

## Status (2026-06-06)

- **Substrate**: clean. Every invented `kittens::ops::*` reference nuked. 67 unit + 12 doctests green. `lower_compute` panics on any architectural SubOp until expansion lands (no fake `.cu` can escape).
- **Cache**: `~/.cache/cudaforge/megakernels/tk_decode_full_*.cu` deleted.
- **Next**: implementation step 1 (sealed-handle infrastructure) per the order above.
