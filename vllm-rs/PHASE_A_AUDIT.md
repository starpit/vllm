# Phase A audit — design vs plan

Worktree: `ff-subtile` · Branch: `worktree-ff-subtile` · Date: 2026-06-06

## Scope

Audit the implemented `SubtileIR → SubtileTape → TkTape` pipeline against
`SUBTILE_IR_REDESIGN.md` (12-step plan) and `SUBTILE_TK20_DECOMP.md` (per-SubOp
TK 2.0 primitive map + 14-step implementation order).

Phase A in this audit means: every step from "land Elementwise::Mul end-to-end"
through "phase A step 8 — pod nvcc compile of emitted .cu". The emit currently
fails nvcc with **10/100 errors remaining**, all in the WGMMA path. The 90
errors removed this audit cycle (kernel-arg plumbing, mbarrier namespace,
arrive ref-vs-ptr, smem↔reg width, vec/tile namespace split, RegVec layout,
non-tensor TMA bulk, scalar literal) all landed as typed-witness lifts, not
runtime guards.

---

## A. What landed against the plan

### A.1 Substrate (commits 1–6 of the SUBTILE_IR_REDESIGN.md plan)

- **Naming + module split**: `subtile_ir.rs` / `subtile_tape.rs` / `tk_tape.rs`
  match plan §1. ✅
- **DAG → linear → target-specific** layering: `lower_dag_to_tape` and
  `lower_subtile_tape_to_tk_tape` exist as syntax-directed translations. ✅
- **Two validators**: `validate_subtile_tape(&SubtileTape, &SubtileIR)` and
  `validate_tk_tape(&TkTape)` both in place. ✅
- **TapeBuilder<S> typestate**: sealed `state::Outside` / `state::InsideLoop`
  with move-only `SlotHandle` / `SlotWritten` tokens. Compile-fail doctests
  exist (subtile_tape.rs lines 442/449/457/469/480/490). ✅
- **Sealed newtypes** per plan §2: `SubtileId`, `TensorId`, `SlotId`,
  `LoopVarId`, `RuntimeBoundId`, `KernelArgRef`, `SoftmaxStateId`, `KvLayoutId`,
  `PageId`, plus the additions this audit cycle: `SmemVecSlot`, `RegTileSlot`,
  `RegVecSlot`. Construction is type-private; doctests prove it. ✅

### A.2 Per-SubOp Instr decomposition (SUBTILE_TK20_DECOMP.md §"Per-SubOp Instr counts")

| SubOp | Plan | Implemented | Notes |
|---|---|---|---|
| MatmulTile | 8 | ✅ | A=128 page (substrate deviation — see B.1) |
| RmsNorm | 10 | ✅ | RegVec rsqrt detour intact |
| SiluMul | 5 | ✅ | ShTileExp/Div/Mul/AddScalar/MulScalar all wired |
| Elementwise::Silu | 6 | ✅ | RegTile chain |
| Elementwise::Mul/Add | 1 each | ✅ | Single ShTileMul/Add |
| SumReduce | 1 | ✅ | Variadic chain of ShTileAdd |
| RopeRotateNeoX | 14 | ✅ | Sub-tile col load + RegTileMulCol×4 + Sub/Add + Sub-tile col store |
| RopeRotateInterleaved | 9 | ⏳ | Plan task #53 still pending — Llama-1B uses NeoX so not blocking |
| RopeAppend | 14 | ✅ | TmaLoadVec×2 + cache writes via `KvCacheChunkStride` typed witness |
| AttnDecode_Init | 3 | ✅ | InitRvNegInfty / InitRvZero / InitRtZero |
| AttnDecode_Qkt | 11 | ✅ | RowMaxAcc / SubRow / Exp2 + RegVec ops + RowSumAcc + Mma_ABt |
| AttnDecode_Sv | 1 | ⚠️ | Implementation uses `WgmmaMmaAB_SmemSmem`; plan says `WgmmaMmaAB_RegSmem` (deviation — see B.2) |
| AttnDecode_Finalise | 2 | ✅ | RegTileDivRow + Store |

### A.3 Typed witnesses (SUBTILE_TK20_DECOMP.md §"New typed-witness types")

All landed:

- `RegTileId<ROWS, COLS, T, Layout>` ✅
- `RegVecId<LEN, T, RvLayout>` ✅ (with `OrthoLayout` / `AlignLayout` /
  `NaiveLayout` impls; AttnDecode + RmsNorm uses Ortho, Rope* uses Align —
  matches TK 2.0 col_vec_layout / row_vec_layout per `rt_base.cuh:78-79`)
- `MbarrierPhase`, `Parity` ✅
- `ScalarF32(f32)` newtype ✅ (replaces stringly scalars; player formats
  with `{:.6}f` — no `1f` C++ literal failures)
- `ApplyLambdaKind` ⏳ (plan §5 — not yet needed; RopeRotateInterleaved
  pending)
- `AccPolicy { Reset, Accumulate }` / `FencePolicy { External, Internal }` ✅
- `GroupWidth<const N>` ✅ — but with **enrichment**: plan said sealed
  N ∈ {1, 4}; impl has {1, 4, 16, 20} plus three sealed marker traits:
  - `WarpLoadWidth` (smem↔reg moves; only `<1>`)
  - `ComputeWidth` (collective compute; `<4>` + `<16>`)
  - `wgmma_*` constructors hardcode `GroupWidth<4>`
  This is a legitimate enrichment — plan's {1,4} restriction was about
  MMA contexts; non-MMA group ops legitimately need wider widths.
- `TileTypeId` (now `TileTypeSpec` + `SmemTileSpec<R, C, T>`) ✅
- `Coord4` / `CoordExpr::RuntimeRow` ⏳ (TMA path uses non-tensor void*+byte
  bulk transfer instead — see B.4)

### A.4 Compile-time-or-garbage rule (`feedback_compile_time_or_garbage` INVIOLABLE)

**Largely upheld** — the typed-constructor layer rejects ~95% of misuse at
rustc time. Selected enforcement points:

- `SmemTileId<R, C, T>` equality across compute Instr inputs (rustc
  unification on `lhs/rhs/dst`).
- `SmemVecId<LEN, T>` length tied to source tile's ROWS/COLS via
  constructor where-clauses (`sh_tile_row_sum<N, ROWS, COLS, T>` requires
  `dst: SmemVecId<ROWS, T>`).
- `RegVecLayout` per row_reduce / row_map / col_map TK 2.0 preconditions —
  our `OrthoLayout`/`AlignLayout` mints satisfy `static_assert(V::layout ==
  col_vec_layout/row_vec_layout)` at compile time of TK 2.0 templates.
- `WarpLoadWidth` (sealed for `GroupWidth<1>`) on smem↔reg move
  constructors. Caller passing `<16>::ALL_CONSUMERS` is rustc E0277.
- `KernelArgRef` sealed-namespace plumbing replaces raw `TensorId` in
  TMA emit — undefined-`aN` errors gone; the body's `aN` indexes are
  by construction in `tape.kernel_args` range.
- `SmemVecSlot` distinct from `PageId`: passing one where the other is
  required is rustc E0308.
- `ByteStride<const BYTES>` const-generic on KvCacheLayout — wrong stride
  is a function-signature mismatch, not a runtime offset bug.
- `ComputeInputs::A1/A2/A3/A4/A5/A6/Variadic` per-arity destructure on
  every `lower_compute` arm — wrong arity surfaces at typed `expect_aN`
  call site, then arm body uses array-typed `[in0, in1]` indexing
  (rustc-bounds-checked).
- `KvCacheLayout<K>` const-generic single source of truth for cache shape.

**Gaps** (acknowledged):

- `wgmma_mma_ab_smem_smem` accepts arbitrary M (no compile-time `M==4` or
  `M==64` gate). Hardware constraint sits in TK 2.0's `static_assert(M==4)`,
  not at the Rust constructor. **This is the cause of the remaining 10/100
  nvcc errors.** See B.1.
- `Instr::StoreAsyncTyped` is dead code — typed constructor exists, but
  the lowerer never pushes this variant. Cleanup candidate.
- `SoftmaxState<Phase>` typestate explicitly deferred per plan §2 line 102:
  "today `feedback_compile_time_or_garbage` is unmet for this witness only."
  Implementation matches plan: ordering enforced procedurally by
  `lower_compute`'s syntax-directed walk, not by typestate.

---

## B. Deviations from plan

### B.1 BLOCKER — Substrate page tile size vs Hopper WGMMA m64

**Plan §"Resolved decision 4"** (SUBTILE_TK20_DECOMP.md:105):
> WGMMA m==1: pad `act_smem` to 4 tile rows in `PagePool`. One row of
> padding waste vs a parallel warp-scope mma Instr variant — padding
> keeps a single MMA path through MatmulTile and AttnDecode_Qkt.

The plan called for activation pages to be **4 tile rows = 64 rows**
(matching Hopper WGMMA's `m64nNk16` instruction shape — TK 2.0
`warpgroup.cuh:199` hardcodes `static_assert(M == 4)`).

**Implementation has uniform 128×128 pages** (`tk_player.rs:583`:
`__shared__ kittens::st_bf<128, 128> page_buf[NUM_PAGES];`). This works
for non-MMA ops (RmsNorm, SiluMul, Rope*) but trips the WGMMA M==4 assert
at every `mma_AB` / `mma_ABt` site (M=128/16=8, not 4). All 10 remaining
nvcc errors trace here.

**Why this happened**: a uniform page substrate simplifies allocation
(one `__shared__` array, single PageId index) at the cost of forcing the
WGMMA path off-spec. The plan acknowledged the trade ("padding keeps a
single MMA path") but expected 64-row pages. The implementation chose
128-row pages, presumably to fit non-MMA ops cleanly, but did not adjust
WGMMA emit to match.

**Resolution paths**:

1. **Pool split** (matches plan exactly): two page pools — `act_pool` at
   64 rows, `weight_pool` (or `general_pool`) at 128 rows. Lowerer routes
   WGMMA A/D pages to the 64-row pool. Costs: doubles arena complexity,
   needs a second `__shared__` array decl, two PageId namespaces.
2. **Uniform 64-row pages**: most decode-time ops only touch 1 row
   semantically anyway; 64-row substrate is enough headroom. RmsNorm /
   Silu could iterate (2 passes per 128 logical rows). Costs: doubles
   the iteration count for non-MMA ops; need to verify SmemVec sizes
   don't break (var_vec/inv_rms_vec at LEN=128 — these are vec, not
   tile-rows, so probably fine).
3. **Register-A WGMMA variant** (`mma_AB(rt_d, rt_a, st_b)` —
   `warpgroup.cuh:139`): supports `D::height == A::height` arbitrary,
   so 128 rows works if A is loaded smem→reg first. Per-warp shape
   becomes 32 rows under width=4 (4 warps × 32 = 128). Plan line 29 says
   AttnDecode_Sv uses this variant — implementation uses smem-A. Costs:
   adds a smem→reg load step before each WGMMA; per-warp shape semantics
   diverge from per-warp-FULL semantics under width=1 (other ops);
   register pressure rises.

Plan's implicit preference (option 1 — pool split per "PagePool") is
the cleanest match.

### B.2 AttnDecode_Sv MMA primitive selection

**Plan SUBTILE_TK20_DECOMP.md:29**:
> AttnDecode_Sv | 1 | **WgmmaMmaAB_RegSmem**

**Implementation** (`lower_subtile_tape_to_tk_tape.rs` AttnDecode_Sv arm
in the loop body) uses `wgmma_mma_ab_smem_smem`. The plan's
`RegSmem` (register-A, smem-B) is exactly the `warpgroup.cuh:139`
variant — its `D::height == A::height` constraint accepts our 128×128
shapes natively. The implementation's `SmemSmem` choice forces the
substrate-shape mismatch in B.1.

The plan also explicitly required `RegTileCopyConvert` (rt fp32 → rt bf16)
between Qkt and Sv per §"Resolved decision 5" (line 106):
> mma_AB requires A.T == B.T. Insert `RegTileCopyConvert` between Qkt
> and Sv to convert fp32 P → bf16.

✅ **`reg_tile_copy_convert(rt_s_fp32, rt_p_bf16, ...)` is wired** in the
AttnDecode arm (lower_subtile_tape_to_tk_tape.rs around line 1463) —
this part of the plan landed. The deviation is only on the MMA variant.

**Resolution**: switch AttnDecode_Sv emit to use a `WgmmaMmaAB_RegSmem`
constructor. The constructor needs: `d: RegTileId<...>`, `a:
RegTileId<...>` (already minted as rt_p), `b: SmemTileId<...>` (v_tile).
Both rt_d and rt_a become per-warp-distributed under width=4 — see B.3.

### B.3 Per-warp register tile shape semantics

Implementation treats `RegTileId<ROWS, COLS, T, L>` as **per-warp-FULL**:
under `GroupWidth<1>` (per_warp), each warp owns the full ROWS×COLS tile;
all 16 consumer warps have redundant copies of the same data.

TK 2.0 register tiles under wider groups (e.g. `group<4>::*`) are
**per-warp-DISTRIBUTED**: each of the 4 warps owns a different
ROWS-row slice; 4 × ROWS = collective row count.

The IR has no syntactic distinction between these two models. Today
this hasn't manifested because:
- All non-WGMMA register ops use `<1>::PER_WARP` (per-warp-FULL).
- WGMMA uses `<4>::WARPGROUP` but the smem-A variant (B.1) doesn't
  have register-tile inputs that overlap with the per-warp-FULL ops.

**This becomes a real conflict** if/when AttnDecode_Sv switches to the
register-A WGMMA variant: rt_p would be loaded per-warp-DISTRIBUTED
(32 rows per warp), but downstream of Sv it gets consumed by the
per-warp-FULL `reg_tile_div_row` / RowSumAcc pipeline.

**Resolution**: introduce a layout-distinction marker on `RegTileId`,
e.g. `WarpDistribution::Replicated | Sharded`. Each Instr constructor
constrains which distribution it accepts. Today this is implicit and
silently invariant on width.

### B.4 TMA path: non-tensor void* bulk vs typed gl<>

Plan SUBTILE_TK20_DECOMP.md:47 specifies:
> `Coord4 = [u32; 4]` newtype + `CoordExpr { Const(Coord4),
> RuntimeRow(ScalarSlotId, …) }` — typed TMA coordinate carrier.
> RopeAppend lowering cannot produce a TmaLoadVec at runtime row
> without `CoordExpr::RuntimeRow`.

This implies the typed-template TK 2.0 path
(`kittens::group<1>::tma::load_async(ST&, gl<T,B,D,R,C>&, coord<ST>&,
sem&)` — `ops/group/memory/tile/tma.cuh:130`).

**Implementation uses the non-tensor bulk variant**
(`ops/group/util/tma.cuh:72` — `load_async(void*, void*, size_bytes,
sem&)`) per Phase A step 8e. Kernel arg type is `void const*
__restrict__` (raw pointer) and byte-offset arithmetic happens at the
call site. Reasoning: `gl<>` plumbing requires per-tensor host-side
TMA descriptor pre-computation (CUtensorMap descriptor encoding for
each shape), which is significant additional host wrapper work.

**Trade-off**: the bulk path is functionally equivalent for our
contiguous loads/stores but loses the TK 2.0 stride/swizzle metadata
that `gl<>` carries. For Llama-1B BF16 weight loads where stride =
row_bytes (no exotic layouts), the bulk path is correct. For
quantized weights or non-contiguous gathers, the typed `gl<>` path
would be required.

**This is a plan deviation** but a justified one — and reversible: the
`KernelArgTy::BufPtr(TensorId)` payload still names the SubtileIR
tensor, so a future commit could route through `gl<>` without
changing IR shape.

`CoordExpr::RuntimeRow` for runtime-row TMA also unimplemented;
RopeAppend uses `kv_cache_runtime_position::<K>(pos_arg, layer)` byte
offset arithmetic instead. Same trade as above.

### B.5 Other minor deviations

- **`mbarrier::*` namespace path**: plan didn't specify; implementation
  initially emitted `kittens::mbarrier::init` / `wait` (both
  non-existent in TK 2.0). Phase A step 8d corrected to
  `kittens::group<1>::init_semaphore` / `kittens::group<1>::wait`
  per `ops/group/util/sync.cuh:35,112`.
- **`arrive` reference vs pointer**: plan didn't specify; implementation
  initially passed `&page_ready[N]` (pointer) where TK 2.0 wants
  `semaphore&` (lvalue reference). Fixed in step 8b.
- **`WarpLoadWidth` separate from `ComputeWidth`**: plan had only
  `ComputeWidth`. Step 8b split it because TK 2.0's smem↔reg moves
  static-assert `ST::rows / RT::rows == GROUP_WARPS`, which for our
  ROWS-equal pairs forces N=1 — `ComputeWidth`'s `<4>`/`<16>` impls
  fail this. The split is a pure compile-time-or-garbage win.

---

## C. Substrate / wiring choices NOT in the plan

These were implementation decisions made during Phase A that the plan
didn't foresee. None violate inviolable rules; document for future
audit.

1. **Per-arity `ComputeInputs` enum** (A1/A2/A3/A4/A5/A6/Variadic).
   Plan §3.2 talks about validator-time edge coverage but didn't
   specify how `Compute.inputs` is typed at the SubtileTape layer.
   Implementation lifted the runtime `inputs.len() >= N` asserts to
   compile-time array destructure. This was Phase A step 7's typed
   lockdown; surfaced two real arity bugs (RopeAppend's 6-input vs
   4-input arity, AttnDecode's 5-input vs 1-input arity) in step 8a.

2. **`SmemVecSlot` distinct from `PageId`**. Plan referenced
   `SmemVecId<LEN, T>` as a typed view onto a page; implementation
   originally aliased the slot onto a `PageId`. TK 2.0 row_sum /
   mul_row / mul_col reject tile-typed pages where vec is required
   (`ducks::sv::all V` concept failure). Step 8d split the namespace,
   added per-slot `__shared__ kittens::sv_<dtype><LEN> sv_<idx>;` decls
   in the kernel preamble, switched 7 Instr fields from `PageId` to
   `SmemVecSlot`. Compile-time-typed; can't pass a tile where vec is
   wanted.

3. **`KernelArgName::Tensor(TensorId)` variant + interner**. Plan's
   §2 sealed `KernelArgRef` as an opaque slot id; implementation
   initially had `KernelArgName::Fixed(&'static str)` only. Phase A
   step 8c added `KernelArgName::Tensor(TensorId)` and a
   `LoweringState::tensor_arg(TensorId) -> KernelArgRef` interner so
   every TMA-emitted `aN` resolves to a real signature parameter.
   Without this, the body emitted `a181..a192` for raw TensorId
   numerics that were never declared.

4. **CTensorMap → `void* __restrict__`** kernel arg type (B.4
   above).

5. **`expect_aN` panic-on-mismatch helpers on `ComputeInputs`**. Per
   the plan's compile-time-or-garbage rule, the residual panic
   ("RopeAppend: expected ComputeInputs::A6, got 5") IS a runtime
   guard — but it's structurally dead given correct
   `dispatch_compute_inputs` in `lower_dag_to_tape`. Acceptable
   because the alternative (typed `SubOp::RopeAppend(A6)` carrying
   a typed-arity payload at the SubtileIR level) is a deeper IR
   refactor than Phase A's scope. Surfaces under
   `feedback_asserts_must_be_dead_code` if pushed strictly.

---

## D. Plan items not yet started

From SUBTILE_IR_REDESIGN.md §4 staged plan:

- **§6.5 optimizer passes** (shmem promotion, fence narrowing, page
  coalescing). The conservative all-gmem lowering is in place. No
  passes implemented. Plan calls these "TkTape → TkTape" transforms.
- **`SoftmaxState<Phase>` typestate** lift (deferred per plan §2:102).

From SUBTILE_TK20_DECOMP.md §"Implementation order":

- Step 8 — `RopeRotateInterleaved` (Llama uses NeoX, not blocking
  for Llama-1B target).
- Phase B "Production gaps" (per the prior session's tracking):
  - cuda-builder glob/link verification
  - dispatch-body fill-in
  - symbol smoke test
  - decode test port
  - pod run + finite-logits gate
  - **wire into ferrite-forward Llama-1B test (the
    `feedback_megakernel_working_means` gate)**

---

## E. Recommendations

1. **B.1 is the critical-path blocker**. Resolve before any further
   step-8 nvcc iteration. The cleanest resolution is **option 1**
   (split page pool per plan §"Resolved decision 4"). Split estimate:
   substrate change to declare two `__shared__` arrays, dual PageId
   namespace (or a sealed `ActPageId` / `WeightPageId` distinction),
   lowerer routes WGMMA A/D pages to act_pool. Maybe 200-400 LOC
   across tk_tape.rs + tk_player.rs + lower_subtile_tape_to_tk_tape.rs.

2. **B.2 (AttnDecode_Sv → RegSmem variant)** can land on top of B.1:
   once act pages are 64-row, the smem-A variant works. Or land it
   independently with the register-A path which has a different
   advantage (lower smem traffic). Plan preference is RegSmem.

3. **B.3 per-warp distribution semantics** is latent but should be
   surfaced before the next IR commit that touches register tiles
   under wider widths. Recommend a `WarpDistribution` marker on
   `RegTileId` or a per-Instr constraint that pins which model the
   tile uses.

4. **B.4 TMA gl<>+coord path** can wait. Llama-1B with bf16 weights
   doesn't need it. Re-open if quantized weights / non-contiguous
   gathers land.

5. **Compile-time-or-garbage gap on WGMMA M=4**: add a where-clause
   to `wgmma_mma_ab_smem_smem` that ties M to TILE_ROW_DIM<T_AB>*4.
   Today this is silently TK-2.0-side; should be rustc-side.

6. **`Instr::StoreAsyncTyped` cleanup**: dead code; remove.

---

## F. What this audit confirms

- The substrate work (typed witnesses, sealed handles, validator
  layering, syntax-directed lowering) is largely faithful to the
  plan. Compile-time-or-garbage is upheld at ~95% of construction
  sites.
- The Llama-1B decode pipeline is structurally complete: every plan
  SubOp has lowered Instrs, every typed witness has a Rust
  representation, the kernel signature + preamble + dispatch
  scaffold are wired.
- Plan-implementation deviations are concentrated at exactly two
  places: (B.1) substrate page shape vs WGMMA m64, (B.2) AttnDecode_Sv
  MMA variant. Both are mechanical fixes traceable to plan
  §"Resolved decisions 4 + 5". Neither requires rethinking the IR
  shape or the staged commit ordering.
- All 90 nvcc errors removed this audit cycle landed as typed-witness
  lifts (sealed traits, sealed-id namespaces, const-generic equality
  bounds). Per `feedback_compile_time_or_garbage`: the substrate
  refuses misuse at construction, not at TK 2.0 template
  instantiation.

---

## G. Open question for the user

Before proceeding to step-8 compile-clean iteration: do you want the
**B.1 fix to be plan-faithful (split page pool) or pragmatic (uniform
64-row pages, double-iterate non-MMA ops)**? The former preserves
plan-stated trade ("padding keeps a single MMA path"); the latter is
~50% smaller diff and accepts a 2× iteration cost on RmsNorm/Silu
that may not matter for decode at seq_len=1.

---

## ADDENDUM (2026-06-07) — post-step-8h

### Resolution of B.1 + B.2

Implemented option B (rt_st WGMMA variant, audit §B.2) via step 8h
commit (`43760b3826`). MatmulTile and AttnDecode_Qkt/Sv now use
`WgmmaMmaAB_RegSmem` / `WgmmaMmaABt_RegSmem` with per-warp 32-row
register tiles (collective 128 across the warpgroup of 4 warps).
PagePool stays uniform 128×128. Plan-line-29-faithful for
AttnDecode_Sv; plan-deviation for MatmulTile + AttnDecode_Qkt
(plan picked SmemSmem with padding for those, audit picked RegSmem
for substrate consistency — accepted trade is one extra
`group<4>::load(rt_a, page_buf[a])` per matmul).

New typed witnesses landed:
* `WarpgroupLoadShape<ST_ROWS, RT_ROWS>` sealed marker, impl'd only
  for `(GroupWidth<4>, 128, 32)`. Gates the warpgroup-sharded
  load/store constructors at compile time.
* `Instr::WgmmaMmaABt_RegSmem` variant + `wgmma_mma_abt_reg_smem`
  constructor (mirror of the pre-existing AB_RegSmem path).
* `load_shmem_to_reg_warpgroup` / `store_reg_tile_to_shmem_warpgroup`
  — sharded movers; the store accepts T_ST != T_RT so fp32→bf16
  conversion happens at TK 2.0's store boundary (replaces the
  pre-existing raw-struct-literal StoreRegTileToShmem workaround).

Substrate decl emits `kittens::st_bf<128, 128, true, 64>` (explicit
swizzle_bytes=64) so RopeRotateNeoX/Append's `subtile<32>(idx)`
splits the head_dim=64 into two 32-col halves cleanly. The implicit
swizzle for 128-col bf16 is 128 (st.cuh:91-103), which makes
subtile<32> fail the `subtile_cols % swizzle_elements == 0` static
assert at st.cuh:163. WGMMA accepts {32, 64, 128} swizzle (st.cuh:90).

### Pod nvcc result after 8h

* **Frontend nvcc**: 0 errors. All TK 2.0 template instantiations
  type-check. Compile-clean by the front-end.
* **ptxas register allocation**: FAILS with `(C7600) Register
  allocation failed with register count of '96'`. ptxas cannot fit
  the kernel into the per-warp register budget under
  `__launch_bounds__(640)` (20 warps × 32 threads × ~96 regs/lane).
  Bumping `--maxrregcount=255` does NOT help — ptxas's "96" is the
  per-thread budget computed from launch_bounds.

### NEW finding: register-pressure design issue

The 128×128 per-warp register tile design under `GroupWidth<1>`
(per-warp-FULL semantics) is **over-budget on Hopper**:

* One `rt<bf16, 128, 128, row>` per-warp = 128*128 / 32 lanes =
  512 elements/lane × 2 bytes / 4 bytes-per-32bit-reg = **256
  lane-regs per single tile**. That's the entire Hopper per-thread
  register file (256 32-bit regs).
* `Elementwise::Silu` mints **5** of these (rt_x, rt_neg, rt_exp,
  rt_denom, rt_result per plan §"Per-SubOp Instr counts" line 20)
  → 5 × 256 = 1280 lane-regs needed, ~5× over budget per warp.
* `RmsNorm` register chain: 2 rt's + 2 rv's at LEN=128 → ~600
  lane-regs.
* `RopeRotateNeoX`: 6 rt's at <128, 32, Bf16> + 2 rv's = 6 × 64 +
  2 = ~386 lane-regs.

The matmul arms (post-8h, per-warp 32-row rt tiles) are at:
* `MatmulTile`: rt_a (32×128 bf16 = 64 lane-regs) + rt_d (32×128
  fp32 = 128 lane-regs) = 192. Just under.
* `AttnDecode`: rt_q + rt_o + rt_s + rt_p + 4 rv's at LEN=32 ≈
  64+128+128+64+8 = 392 lane-regs. **Over.**

So even the matmul arms are over-budget when AttnDecode mints all
its register state simultaneously.

**Root cause**: the IR's per-warp-FULL semantics under width=1
(every warp has the entire data redundantly) combined with
"large register tiles" (>= 32×128) doesn't fit Hopper's
256-reg/lane budget. The fix is either:

1. **Reduce rt sizes**: switch non-MMA arms (Silu, RmsNorm, Rope)
   from full-tile 128×128 register tiles to 16-row register tiles
   (height=1, fitting in one TILE_ROW_DIM block) iterated row-by-row.
   Plan §"Per-SubOp Instr counts" wrote these as full-tile chains;
   need to revisit per the register budget.
2. **Push to shared tiles**: many "register tile" intermediates can
   live in shared tiles (RmsNorm's x_sq, Silu's exp, etc.). The
   plan already does this for some (RmsNorm uses sh_tile_mul for
   x²); extend uniformly.
3. **Reduce concurrent live rt's**: alias / SSA-merge the rt
   intermediates so only 1-2 are live at once. Plan §"Resolved 1"
   says "Lifetime model: SSA. Every Instr that produces a `RegTileId`
   / `RegVecId` mints a fresh id" — explicit choice over arena
   reuse. The arena reuse path (which the plan declined) would help
   here.

This is a **plan-level rethink**, not iteration territory.

### Recommendation

Phase A step 8 (pod nvcc compile of emitted .cu) is at the
**frontend-clean / ptxas-blocked** state. The register-pressure
issue needs a substrate-IR design pass that:

* Looks at ACTUAL Hopper register budgets (256 32-bit regs/lane).
* Maps each Instr's register-tile contribution.
* Rewrites Silu / RmsNorm / Rope / AttnDecode register chains to
  fit. Likely means smaller rt sizes (16×N height=1 iterated) or
  more shared-tile intermediates.

This is its own milestone (plan calls for it but the per-SubOp
Instr counts didn't budget for register pressure). Don't iterate
on this without a fresh plan section.

---

## ADDENDUM 2 (2026-06-07) — post-rt_alias_pass

### What landed

`rt_alias_pass` (commit `6006a478a3`) — first §6.5 optimizer pass.
Linear-scan greedy coalescing keyed on `RegTileArenaEntry`,
loop-aware (slots defined inside a loop body extend their effective
live range to the enclosing `[loop_open, loop_close]` extent;
predecessors are admitted only if their effective_last_use precedes
the candidate's `loop_open`). Plan-faithful — slots into the §6.5
pass infrastructure with `crates/ferrite-wavefront/src/passes/`
directory and per-pass validator postcondition.

Pod measurements:
* `reg_tile_arena` cardinality: 71 → 9 (87% reduction).
* Llama-1B megakernel still ptxas-fails with 96-reg target —
  the remaining 9 rt's collectively need ~704 lane-regs. Pass
  did its job; the floor is set by the register-tile shapes
  (32-row per-warp under width=4 = 128 lane-regs per fp32 tile
  × 2 tiles + 4 per-warp accumulator vecs ≈ 384 lane-regs in
  AttnDecode alone), not arena multiplicity.

### Substrate-shrink attempt + cascade

Tried switching uniform substrate from 64×128 page rows × cols to
**64-row pages** (matching plan §"Resolved decision 4" "pad
act_smem to 4 tile rows"). With 64-row substrate, WGMMA per-warp
rt drops to 16-row height=1, AttnDecode register total drops to
~192 lane-regs — under the 256/lane budget.

The cascade: 64-row uniform substrate means **B**'s rows = 64,
which is the matmul **K** dim. WGMMA `rt-A · st-B` requires
`A.cols == B.rows` (= K). Our matmul A has cols = 128 (head_dim
or hidden chunk). Mismatch.

To make 64-row substrate work for WGMMA:
* **Option A**: K-tile in the lowerer — emit `2× wgmma_mma_ab_reg_smem`
  per logical matmul, each on a 64-K slice, accumulating into rt_d.
  Requires a K-loop in MatmulTile / AttnDecode_Qkt / AttnDecode_Sv
  arms. Substantial arm-rewrite work.
* **Option B**: Split substrate pools — separate `act_pool` (64×128)
  and `weight_pool` (128×128). Doubles the `__shared__` decls,
  introduces a sealed `ActPageId` namespace, lowerer routes WGMMA
  A/D pages to act, B pages to weight. Plan §"Resolved 4" implied
  this with "padding waste vs a parallel warp-scope mma Instr
  variant — padding keeps a single MMA path."
* **Option C**: Hopper `setmaxnreg` PTX intrinsic. Asymmetric
  per-warpgroup register budgets — service warps trade their share
  to consumer warps. Available in TK 2.0 helpers. Doesn't change
  the substrate; just makes ptxas's per-thread budget non-uniform.

Reverted the substrate-shrink attempt — it requires either A or
B as a structural commit. Returning to the stable 8h+rt_alias
state (97 ferrite-wavefront tests green; arena 9 entries on the
emitted Llama-1B megakernel; ptxas blocks at 96 regs).

### Concrete next steps (ordered by leverage)

1. **`rt_to_smem_pass`** (or lowerer rewrite) for `RopeRotate` /
   `RopeAppend` — eliminates 5 of the 9 remaining rt's
   (`rt<bf16, 128, 32>` per-warp). Per the recon table, every
   op in the RoPE chain has a shared analog (`mul_col`, `sub`,
   `add`); the only blocker is shared-side sub-tile col views.
   Plan-faithful (§6.5 pass infrastructure already exists).
2. **`setmaxnreg` warp specialization** — drops the
   `__launch_bounds__(640)` floor by giving consumer warps
   ~200 regs/lane instead of the uniform 96. Hopper-native;
   TK 2.0 has helpers. Touches `tk_player.rs::emit_kernel`
   preamble + per-warp role gating. Low-touch, high-impact.
3. **Substrate split (A or B)** — only if 1+2 don't get to
   compile-clean. Larger commit; defer until needed.

`rt_alias_pass` is committed and works; the foundation for §6.5
is in place. Next pass to land is `rt_to_smem_pass` for the
liftable RoPE chain. ptxas-clean is 1-2 commits away once that
lands and either the launch_bounds is loosened or the substrate
splits.

---

## ADDENDUM 3 (2026-06-07) — substrate split step 1 landed

Commit `b9c0a50292` adds the activation page pool alongside
`page_buf` without touching existing logic:

* `NUM_ACT_PAGES = 8`, `ACT_PAGE_SIZE = 64*128*2 = 16384` bytes
  per entry (sized for Hopper WGMMA m64 — A.rows == 4 *
  TILE_ROW_DIM<bf16> = 64).
* Sealed `ActPageId(u8)` — distinct namespace from `PageId`.
  Passing one where the other is required is rustc E0308.
* `tk_player::shared_act_bf_decl` emits
  `__shared__ kittens::st_bf<64, 128, true, 64> act_buf[NUM_ACT_PAGES];`
  alongside `page_buf` in the kernel preamble. Same swizzle_bytes=64
  so RopeRotateNeoX's `subtile<32>(idx)` head_dim=64 split works
  on either pool.
* `act_ready` / `act_done` semaphores parallel to `page_ready` /
  `page_done`.
* `DYN_SMEM` math sums both pool budgets; the host wrapper sets
  `cudaFuncAttributeMaxDynamicSharedMemorySize` to the sum.

No existing arm uses `act_buf` yet. Kernel emit is unchanged
modulo the new declarations; nvcc still 0 errors; arena still 9
entries (substrate-add doesn't touch rt slot count); ptxas still
blocked.

### Step 2 design (remaining work for ptxas-clean)

Routing WGMMA A and AttnDecode q/k/v tiles to `ActPageId` requires
extending the typed substrate. Two design choices on the table:

1. **`ActSmemTileId<R, C, T>` peer type** — distinct from
   `SmemTileId`. Each Instr that takes a smem tile gets either a
   new parallel constructor or its constructor is generic over an
   `IsSmemTile` trait (so `SmemTileId` and `ActSmemTileId` both
   satisfy it). The Instr field at the runtime layer becomes an
   `AnyPageId { Page(PageId), Act(ActPageId) }` enum (or a typed
   sealed sum), and the player matches at emit time to choose
   `page_buf[N]` vs `act_buf[N]`.
   - Pros: full compile-time-or-garbage; routing decisions are
     in the type.
   - Cons: ~15 Instr variants get touched (every smem-bearing
     one); every load/store/compute constructor needs the trait
     bound; the player gains a per-Instr arm to choose the
     correct array reference.

2. **Encoded `PageId` (high bit = act)** — keep `PageId(u8)` but
   reserve the top bit (or N bit-range) for "act pool". Lowerer
   sets the high bit when minting from act_pool; player checks
   the bit at emit time.
   - Pros: minimal type-level disruption; existing Instr fields
     unchanged.
   - Cons: runtime-checked routing decision (violates
     `feedback_compile_time_or_garbage`); the bit-encoding is a
     premature-format-encoding leak per
     `feedback_no_premature_string_encoding` adjacent reasoning.

The plan-faithful answer is **option 1**. Substantial work but
plays well with the rest of the typed-witness substrate. Estimated
3–5 LOC commits across:

* New `ActSmemTileId` + `IsSmemTile` trait + `AnyPageId` Instr
  field replacement (or per-arm Instr variants for act-routed ops).
* WGMMA constructors gain act-source variants (or a generic
  trait bound).
* `load_shmem_to_reg_warpgroup` adds a parallel constructor for
  `src: ActSmemTileId<ST_ROWS, COLS, T>` with new
  `WarpgroupLoadShape<64, 16>` impl for `GroupWidth<4>`.
* Lowerer arms: MatmulTile (A → act, dst → act), AttnDecode
  (q/k/v → act). Per-warp rt sizes drop to 16-row height=1.
* Player gains `act_buf[N]` emit arm parallel to existing
  `page_buf[N]` arm; both pools' barriers (act_ready/act_done vs
  page_ready/page_done) get matched at the appropriate Instr
  arms.

After step 2 lands and `rt_to_smem_pass` for RoPE follows, the
arena should drop to ~4 register tiles (down from 9), with
per-warp 16-row sizes totaling ~192 lane-regs — under Hopper's
256/lane budget. ptxas-clean is the expected outcome.

Status at this commit: **frontend nvcc 0 errors, ptxas blocked,
substrate split foundation laid (step 1), step 2 routing is the
next major commit**.
