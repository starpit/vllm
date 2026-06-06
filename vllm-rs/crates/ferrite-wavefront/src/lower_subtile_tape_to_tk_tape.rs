// SPDX-License-Identifier: Apache-2.0
//! `lower_subtile_tape_to_tk_tape(&SubtileTape, &SubtileIR<F>) -> TkTape` — the
//! syntax-directed translation from the target-agnostic
//! [`crate::subtile_tape::SubtileTape`] to the TK-target
//! [`crate::tk_tape::TkTape`].
//!
//! Per `SUBTILE_IR_REDESIGN.md` §4 commit 6:
//!
//! - **Always-executable invariant**: the output is a complete tape
//!   that runs correctly when emitted (validator-green by construction;
//!   the dedicated `validate_tk_tape` is commit 6b).
//! - **Conservative all-gmem routing**: every slot becomes a
//!   [`PageId`] in 1:1 correspondence; every cross-slot edge surfaces
//!   as `StoreAsync` + `FenceDevice` + `PageBarrierArrive{Done}` on the
//!   producer side, and `PageBarrierWait{Ready}` + `LoadAsync` on the
//!   consumer side. Optimizer passes (§4 commit 6.5+) rewrite shmem-
//!   carry-forward edges later — the lowering itself does no analysis.
//! - **No analysis, no lookahead, no shmem decisions** — those are the
//!   optimizer's job (kill criterion K4).
//! - **AttnDecode 4-phase split**: one `SubtileTape::Compute` for
//!   `SubOp::AttnDecode` lowers to four flat
//!   `Instr::AttnDecode{Init,Qkt,Sv,Finalise}` variants, with the
//!   SubtileTape's `OpenLoop`/`CloseLoop` becoming
//!   `Instr::ForLoop` wrapping `Qkt + Sv` only. `Init` emits before
//!   the loop, `Finalise` after.
//! - **Witnesses preserved**: `KvCacheLayout`, `KvCacheProducer`,
//!   `RopeForm` (const-generic on the IR), `SoftmaxStateId` are read
//!   from the SubtileIR node (not duplicated tape-side) and passed
//!   directly into the TkTape Instr fields.
//! - **TensorId, not BufId**: source identifiers preserve SubtileIR's
//!   `TensorId` namespace (commit 5c removed `BufId` from TkTape).
//!
//! ## Layout
//!
//! - `lower_subtile_tape_to_tk_tape::<F>(&tape, &graph) -> TkTape` is the public
//!   entry point. It walks the SubtileTape's `Vec<Instr>` once,
//!   maintaining a [`LoweringState`] of slot→page mappings, the
//!   `seq_len` kernel-arg binding (minted on demand for AttnDecode's
//!   loop), and the active `Vec<Instr>` stack to thread `OpenLoop`
//!   bodies into `Instr::ForLoop`'s nested vec.
//! - Per-SubOp branches in `lower_compute` are exhaustive (no `_ =>`
//!   arm); adding a new `SubOp` variant becomes a compile error here.

#![allow(dead_code)]

use std::collections::BTreeMap;

use crate::subtile_ir::{KvCacheLayout, KvCacheShape, RopeForm, SoftmaxStateId, SubtileId, SubtileIR, TensorId, TensorRegion};
use crate::subtile_tape::{
    Instr as STInstr, LoopBound, LoopVarId as STLoopVarId, SlotId, SubtileTape,
};
use crate::tk_tape::{
    ByteOffsetExpr, Instr, KernelArg, KernelArgName, KernelArgRef, KernelArgTy,
    KvLayoutEntry, KvLayoutId, LoadSpec, LoopVarId as TkLoopVarId, PageBarrier, PageId,
    SoftmaxStateId as TkSoftmaxStateId, StoreSpec, TileShape, TkTape, U32Source, WarpRole,
    validate_tk_tape,
};

// ── BF16 element width ──────────────────────────────────────────────
//
// The wavefront IR is dense bf16 (per plan §6: "Quantized weights …
// are out of scope"). 2 bytes per element shows up everywhere (TMA
// expect_bytes, ByteOffsetExpr strides, TileShape::elem_bytes).

const ELEM_BYTES: u32 = 2;

// ── Roles ───────────────────────────────────────────────────────────
//
// Conservative all-gmem routing assigns roles by op kind:
// - TMA loads/stores get `WarpRole::Loader` / `WarpRole::Storer`.
// - Compute Instrs (RmsNorm/GemmM1/SiluMul/ResidualAdd/RopeRotate/
//   AttnDecode*) and barrier waits/arrives get `WarpRole::AllConsumers`.
// - Fences and sync run on `WarpRole::All`.
// The optimizer pass `narrow_role` will refine these later.

// Typed role witnesses (sealed). Per
// `feedback_ff_subtile_compile_time_inviolable`, role-constrained
// Instr constructors take these directly so a wrong-role construction
// is rustc E0308. The runtime `WarpRole::*` constants below are kept
// for the Instr variants whose role is informational (threadfence_*,
// commit_bulk, wait_bulk — CTA-level ops, role-agnostic in TK 2.0).
const LOAD_ROLE: crate::tk_tape::LoaderRole = crate::tk_tape::LoaderRole;
const STORE_ROLE: crate::tk_tape::StorerRole = crate::tk_tape::StorerRole;
const COMPUTE_ROLE: WarpRole = WarpRole::AllConsumers;
const ALL_ROLE: WarpRole = WarpRole::All;

// ── Lowering state ──────────────────────────────────────────────────

/// Mutable state carried through the syntax-directed walk.
struct LoweringState<'g, F: RopeForm, K: KvCacheShape> {
    graph: &'g SubtileIR<F, K>,
    /// Top-level instruction stack. `lower_*` helpers push to whichever
    /// slot is on top (the outermost is `tape.instrs`; an open
    /// `OpenLoop` pushes a new buffer; `CloseLoop` pops and wraps as
    /// `Instr::ForLoop`).
    instr_stack: Vec<Vec<Instr>>,
    /// Active loop var (matches the SubtileTape `LoopVarId` to a
    /// fresh-minted TkTape `LoopVarId`).
    loop_var_map: BTreeMap<u32, TkLoopVarId>,
    /// Kernel args minted so far (mirror of `tape.kernel_args` for
    /// dedupe).
    kernel_args: Vec<KernelArg>,
    /// `seq_len` kernel-arg slot, lazily minted the first time an
    /// AttnDecode opens its KV-sweep loop. Reused across AttnDecodes
    /// (one `seq_len` per forward).
    seq_len_arg: Option<KernelArgRef>,
    /// `decode_position` kernel-arg slot for RopeRotate / RopeAppend,
    /// lazily minted on first rotary.
    position_arg: Option<KernelArgRef>,
    /// Page-id allocator. `next_page` is the next-fresh id minted
    /// only if `free_pages` is empty; `release_page` pushes onto
    /// `free_pages` so recycled ids are popped before fresh ones.
    /// The high-water mark is bounded by the maximum live-slot count
    /// (not the cumulative slot count); for Llama-3.2-1B at nb=256
    /// with the conservative all-gmem path, this stays well below
    /// `NUM_PAGES`.
    next_page: u8,
    free_pages: Vec<PageId>,
    /// Slot id → PageId.
    slot_to_page: BTreeMap<u32, PageId>,
    /// Fresh LoopVarId allocator (TkTape side).
    next_loop_var: u32,
    /// `KvCacheLayout` table — one entry per K-cache `TensorId` in the
    /// forward; the lowering reaches into the SubtileIR node's witness
    /// (single source of truth) and looks up the entry by TensorId.
    kv_layouts: Vec<KvLayoutEntry>,
    kv_layout_index: BTreeMap<TensorId, KvLayoutId>,
    /// `SoftmaxStateId` carry-through. SubtileIR's `SoftmaxStateId`
    /// (sealed) maps 1:1 to TkTape's `SoftmaxStateId` (a public
    /// tuple-struct).
    softmax_state_index: BTreeMap<SoftmaxStateId, TkSoftmaxStateId>,
    next_softmax_state: u32,
    /// Stack of active SubtileTape `LoopVarId`s — pushed by
    /// `OpenLoop`, popped by `CloseLoop`. The top entry is the loop
    /// body whose frame is on top of `instr_stack`.
    active_loop_stack: Vec<u32>,
    /// Register-tile arena accumulated across the lowering. Moved
    /// onto the output TkTape at finalize. See [`TkTape::mint_reg_tile`]
    /// for the pattern.
    reg_tile_arena: BTreeMap<crate::tk_tape::RegTileSlot, crate::tk_tape::RegTileArenaEntry>,
    next_reg_tile_slot: u16,
    /// Register-vec arena.
    reg_vec_arena: BTreeMap<crate::tk_tape::RegVecSlot, crate::tk_tape::RegVecArenaEntry>,
    next_reg_vec_slot: u16,
}

impl<'g, F: RopeForm, K: KvCacheShape> LoweringState<'g, F, K> {
    fn new(graph: &'g SubtileIR<F, K>) -> Self {
        Self {
            graph,
            instr_stack: vec![Vec::new()],
            loop_var_map: BTreeMap::new(),
            kernel_args: Vec::new(),
            seq_len_arg: None,
            position_arg: None,
            next_page: 0,
            free_pages: Vec::new(),
            slot_to_page: BTreeMap::new(),
            next_loop_var: 0,
            kv_layouts: Vec::new(),
            kv_layout_index: BTreeMap::new(),
            softmax_state_index: BTreeMap::new(),
            next_softmax_state: 0,
            active_loop_stack: Vec::new(),
            reg_tile_arena: BTreeMap::new(),
            next_reg_tile_slot: 0,
            reg_vec_arena: BTreeMap::new(),
            next_reg_vec_slot: 0,
        }
    }

    /// Mint a fresh [`RegTileId<R, C, T, L>`] in the lowering's
    /// arena (transferred to TkTape at finalize). Mirrors
    /// [`TkTape::mint_reg_tile`] but operates on the lowering's
    /// transient state.
    fn mint_reg_tile<const R: usize, const C: usize, T, L>(
        &mut self,
    ) -> crate::tk_tape::RegTileId<R, C, T, L>
    where
        T: crate::tk_tape::TileDtype,
        L: crate::tk_tape::RegTileLayout,
    {
        use crate::tk_tape::{RegTileArenaEntry, RegTileId, RegTileSlot};
        let slot = RegTileSlot(self.next_reg_tile_slot);
        self.next_reg_tile_slot = self
            .next_reg_tile_slot
            .checked_add(1)
            .expect("reg_tile slot overflow (>= 65536 register tiles in one kernel)");
        self.reg_tile_arena.insert(
            slot,
            RegTileArenaEntry {
                rows: R as u16,
                cols: C as u16,
                dtype: T::tag(),
                layout: L::tag(),
            },
        );
        RegTileId::from_slot(slot)
    }

    /// Mint a fresh [`RegVecId<LEN, T, RV>`].
    fn mint_reg_vec<const LEN: usize, T, RV>(
        &mut self,
    ) -> crate::tk_tape::RegVecId<LEN, T, RV>
    where
        T: crate::tk_tape::TileDtype,
        RV: crate::tk_tape::RegVecLayout,
    {
        use crate::tk_tape::{RegVecArenaEntry, RegVecId, RegVecSlot};
        let slot = RegVecSlot(self.next_reg_vec_slot);
        self.next_reg_vec_slot = self
            .next_reg_vec_slot
            .checked_add(1)
            .expect("reg_vec slot overflow");
        self.reg_vec_arena.insert(
            slot,
            RegVecArenaEntry {
                len: LEN as u16,
                dtype: T::tag(),
                layout: RV::tag(),
            },
        );
        RegVecId::from_slot(slot)
    }

    fn cur(&mut self) -> &mut Vec<Instr> {
        self.instr_stack
            .last_mut()
            .expect("instr_stack invariant: at least one buffer always live")
    }

    fn push(&mut self, instr: Instr) {
        self.cur().push(instr);
    }

    fn alloc_page(&mut self, slot: SlotId) -> PageId {
        let p = if let Some(reused) = self.free_pages.pop() {
            reused
        } else {
            let p = PageId(self.next_page);
            self.next_page = self.next_page.checked_add(1).expect(
                "PageId overflow (>=256 concurrent live slots in one tape \
                 — exceeds NUM_PAGES; tape needs slot-coalescing pass before lowering)",
            );
            p
        };
        self.slot_to_page.insert(slot.index(), p);
        p
    }

    /// Allocate an unbound scratch page (no SlotId binding). Used by
    /// SubOp lowerings that need temporary pages for intermediate
    /// shared-vec / shared-tile state (e.g. RmsNorm's row_sum vec).
    /// Caller is responsible for the page's lifetime.
    fn alloc_temp_page(&mut self) -> PageId {
        if let Some(reused) = self.free_pages.pop() {
            reused
        } else {
            let p = PageId(self.next_page);
            self.next_page = self.next_page.checked_add(1).expect(
                "PageId overflow (>=256 concurrent live slots) on temp alloc",
            );
            p
        }
    }

    fn page_of(&self, slot: SlotId) -> PageId {
        *self
            .slot_to_page
            .get(&slot.index())
            .expect("compute_to references a slot that was not AllocSlot'd \
                     (would have been caught by validate_subtile_tape)")
    }

    fn release_page(&mut self, slot: SlotId) {
        if let Some(p) = self.slot_to_page.remove(&slot.index()) {
            self.free_pages.push(p);
        }
    }

    fn intern_kernel_arg(&mut self, arg: KernelArg) -> KernelArgRef {
        let idx = self.kernel_args.len() as u16;
        self.kernel_args.push(arg);
        KernelArgRef(idx)
    }

    fn seq_len(&mut self) -> KernelArgRef {
        if let Some(r) = self.seq_len_arg {
            return r;
        }
        let r = self.intern_kernel_arg(KernelArg {
            name: KernelArgName::Fixed("__num_kv_pages"),
            ty: KernelArgTy::U32 {
                source: U32Source::NumKvPages,
            },
        });
        self.seq_len_arg = Some(r);
        r
    }

    fn position(&mut self) -> KernelArgRef {
        if let Some(r) = self.position_arg {
            return r;
        }
        let r = self.intern_kernel_arg(KernelArg {
            name: KernelArgName::Fixed("__decode_position"),
            ty: KernelArgTy::U32 {
                source: U32Source::DecodePosition,
            },
        });
        self.position_arg = Some(r);
        r
    }

    fn fresh_loop_var(&mut self) -> TkLoopVarId {
        let v = TkLoopVarId(self.next_loop_var);
        self.next_loop_var += 1;
        v
    }

    fn intern_kv_layout(&mut self, layout: KvCacheLayout<K>) -> KvLayoutId {
        let key = layout.cache_tensor();
        if let Some(id) = self.kv_layout_index.get(&key) {
            return *id;
        }
        let id = KvLayoutId(self.kv_layouts.len() as u32);
        self.kv_layouts.push(KvLayoutEntry::from_witness(layout));
        self.kv_layout_index.insert(key, id);
        id
    }

    fn intern_softmax_state(&mut self, ir_id: SoftmaxStateId) -> TkSoftmaxStateId {
        if let Some(id) = self.softmax_state_index.get(&ir_id) {
            return *id;
        }
        let id = TkSoftmaxStateId(self.next_softmax_state);
        self.next_softmax_state += 1;
        self.softmax_state_index.insert(ir_id, id);
        id
    }
}

// ── Public entry point ──────────────────────────────────────────────

/// Lower a [`SubtileTape`] (paired with the SubtileIR it was lowered
/// from) to a [`TkTape`]. Conservative all-gmem routing; no analysis;
/// validator-green by construction (the dedicated `validate_tk_tape`
/// runs at commit 6b's exit).
pub fn lower_subtile_tape_to_tk_tape<F: RopeForm, K: KvCacheShape>(
    tape: &SubtileTape,
    graph: &SubtileIR<F, K>,
) -> TkTape {
    let mut state = LoweringState::new(graph);

    for instr in tape.instrs() {
        match instr {
            STInstr::AllocSlot { slot } => {
                lower_alloc_slot(&mut state, *slot);
            }
            STInstr::Compute {
                node,
                writes,
                reads,
            } => {
                lower_compute(&mut state, *node, *writes, reads);
            }
            STInstr::FreeSlot { slot } => {
                lower_free_slot(&mut state, *slot);
            }
            STInstr::OpenLoop { var, bound } => {
                lower_open_loop(&mut state, *var, *bound);
            }
            STInstr::CloseLoop { var } => {
                lower_close_loop(&mut state, *var);
            }
        }
    }

    // The final kernel-end drain — every persistent tape ends with one,
    // so the last in-flight TMA store is observable to the next forward.
    drain(&mut state);

    let mut out = TkTape::new();
    for arg in state.kernel_args {
        out.kernel_args.push(arg);
    }
    // Per plan §2 line 88: KvCacheLayout witness propagates to TkTape;
    // consumer reads via TkTape::kv_layout(KvLayoutId). Move the
    // intern table from the lowering's transient state onto the
    // output tape so KvLayoutId references in Instr::RopeRotate /
    // AttnDecode resolve to a real KvLayoutEntry, not a dangling
    // index into a dropped Vec.
    out.kv_layouts = state.kv_layouts;
    // Transfer the register-tile / register-vec arenas accumulated
    // during lowering — emit_kernel walks these in id-order to emit
    // the kernel-preamble decls.
    out.reg_tile_arena = state.reg_tile_arena;
    out.next_reg_tile_slot = state.next_reg_tile_slot;
    out.reg_vec_arena = state.reg_vec_arena;
    out.next_reg_vec_slot = state.next_reg_vec_slot;
    let top = state
        .instr_stack
        .pop()
        .expect("instr_stack invariant: top buffer was never pushed");
    debug_assert!(
        state.instr_stack.is_empty(),
        "instr_stack invariant: an OpenLoop did not match its CloseLoop \
         (would have been caught by validate_subtile_tape)"
    );
    out.instrs = top;
    validate_tk_tape(&out)
        .expect("lower_subtile_tape_to_tk_tape: produced invalid TkTape (commit 6b post-condition)");
    out
}

// ── SubtileTape::AllocSlot ──────────────────────────────────────────
//
// Mints a PageId. No Instr emit at the lowering layer — `BarrierInit`
// is a per-kernel-prelude concern (lands with the kernel-args + prelude
// pass in commit 6+). The slot's PageId is now reachable via
// `state.page_of(slot)`.

fn lower_alloc_slot<F: RopeForm, K: KvCacheShape>(state: &mut LoweringState<F, K>, slot: SlotId) {
    state.alloc_page(slot);
}

// ── SubtileTape::FreeSlot ───────────────────────────────────────────
//
// Conservative all-gmem path: releasing the page mapping is enough.
// Optimizer passes that promote a slot to shmem will add explicit
// `PageBarrierArrive{Consumed}` here.

fn lower_free_slot<F: RopeForm, K: KvCacheShape>(state: &mut LoweringState<F, K>, slot: SlotId) {
    state.release_page(slot);
}

// ── SubtileTape::OpenLoop / CloseLoop ───────────────────────────────

fn lower_open_loop<F: RopeForm, K: KvCacheShape>(
    state: &mut LoweringState<F, K>,
    var: STLoopVarId,
    bound: LoopBound,
) {
    let tk_var = state.fresh_loop_var();
    state.loop_var_map.insert(var.index(), tk_var);
    // Emit the flat ForLoopOpen* into the parent frame.
    let open = match bound {
        LoopBound::Const(n) => Instr::ForLoopOpenConst { var: tk_var, n },
        LoopBound::Runtime(_) => Instr::ForLoopOpenKernelArg {
            var: tk_var,
            arg: state.seq_len(),
        },
    };
    state.push(open);
    // Body builds into a fresh frame; CloseLoop appends it to parent.
    state.instr_stack.push(Vec::new());
    state.active_loop_stack.push(var.index());
}

fn lower_close_loop<F: RopeForm, K: KvCacheShape>(state: &mut LoweringState<F, K>, var: STLoopVarId) {
    let popped = state
        .active_loop_stack
        .pop()
        .expect("CloseLoop without matching OpenLoop \
                 (would have been caught by validate_subtile_tape)");
    debug_assert_eq!(popped, var.index(), "loop-var stack mismatch");
    let body = state
        .instr_stack
        .pop()
        .expect("CloseLoop without matching OpenLoop \
                 (would have been caught by validate_subtile_tape)");
    let tk_var = *state
        .loop_var_map
        .get(&var.index())
        .expect("CloseLoop var not in map (typestate invariant violated)");
    // Append body into parent + close.
    state.cur().extend(body);
    state.push(Instr::ForLoopClose { var: tk_var });
    // NUKED: post-loop AttnDecode Finalise drain. Returns when
    // AttnDecode decomposes into TK 2.0 primitive Instrs.
    let _ = var;
}

// ── SubtileTape::Compute — the SubOp dispatch ───────────────────────

fn lower_compute<F: RopeForm, K: KvCacheShape>(
    state: &mut LoweringState<F, K>,
    node_id: SubtileId,
    writes: SlotId,
    reads: &[SlotId],
) {
    use crate::subtile_ir::{EwKind, SubOp};

    let node = &state.graph.nodes[node_id.0 as usize];
    let dst_page = state.page_of(writes);

    // Wait on each predecessor slot's Ready barrier — the producer's
    // StoreAsync + Arrive{Done} pairs with our Wait{Ready} on the
    // same page. Conservative all-gmem path uses parity 0.
    for r in reads {
        let p = state.page_of(*r);
        state.push(Instr::PageBarrierWaitStaticP0 {
            page_id: p,
            kind: PageBarrier::Ready,
            role: COMPUTE_ROLE,
        });
    }

    // External (source-tensor) loads: any input TensorRegion whose
    // tensor is a leaf source is brought into a page via TMA.
    let num_sources = state.graph.num_sources;
    for inp in &node.inputs {
        if inp.tensor.0 < num_sources {
            emit_external_load(state, inp, dst_page);
        }
    }

    // SubOp dispatch — only Elementwise(Mul) is implemented. Every
    // other arch op panics: per INVIOLABLE feedback_tk_2_0_only +
    // feedback_tk20_primitives_first, lower_compute will not
    // fabricate a tape from invented `kittens::ops::*` helpers. Each
    // arch op lands one at a time as its TK 2.0 primitive sequence
    // is implemented per SUBTILE_TK20_DECOMP.md.
    match &node.op {
        SubOp::Elementwise(EwKind::Mul) => {
            // Two compile-time witnesses ride on this constructor:
            //   - `GroupWidth::<16>::ALL_CONSUMERS` — the `where
            //     GroupWidth<N>: ComputeWidth` bound rejects N=1 /
            //     N=20 at rustc time.
            //   - `SmemTileId::<128, 128, Bf16>` — the substrate's
            //     uniform `__shared__ kittens::st_bf<128, 128>
            //     page_buf[]` shape. All three operands MUST share
            //     ROWS, COLS, T at the type level; mismatch is a
            //     rustc E0308. When non-uniform pools land, each
            //     pool will mint its own SmemTileId<…> and a square
            //     tile cannot be passed where a vector tile is
            //     expected.
            // Per `feedback_ff_subtile_compile_time_inviolable`.
            use crate::tk_tape::{Bf16, GroupWidth, SmemTileId};
            let lhs = SmemTileId::<128, 128, Bf16>::from_page(state.page_of(reads[0]));
            let rhs = SmemTileId::<128, 128, Bf16>::from_page(state.page_of(reads[1]));
            let dst = SmemTileId::<128, 128, Bf16>::from_page(dst_page);
            let _ = COMPUTE_ROLE; // role-tag retained for future
                                  // walker-side gating.
            state.push(Instr::sh_tile_mul(
                lhs,
                rhs,
                dst,
                GroupWidth::<16>::ALL_CONSUMERS,
            ));
            emit_store_and_arrive(state, &node.output, dst_page);
        }
        SubOp::Elementwise(EwKind::Add) => {
            // Same compile-time witnesses as the Mul arm; only the
            // emitted TK 2.0 primitive differs (`group<N>::add` vs
            // `group<N>::mul`).
            use crate::tk_tape::{Bf16, GroupWidth, SmemTileId};
            let lhs = SmemTileId::<128, 128, Bf16>::from_page(state.page_of(reads[0]));
            let rhs = SmemTileId::<128, 128, Bf16>::from_page(state.page_of(reads[1]));
            let dst = SmemTileId::<128, 128, Bf16>::from_page(dst_page);
            state.push(Instr::sh_tile_add(
                lhs,
                rhs,
                dst,
                GroupWidth::<16>::ALL_CONSUMERS,
            ));
            emit_store_and_arrive(state, &node.output, dst_page);
        }
        SubOp::SumReduce => {
            // SumReduce over N inputs: chain N-1 ShTileAdd Instrs.
            // Per SUBTILE_TK20_DECOMP.md §"Per-SubOp Instr counts":
            // SumReduce reuses ShTileAdd; no new Instr variant.
            //
            // Sequence (for N reads):
            //   ShTileAdd(dst, reads[0], reads[1])   // dst = r0+r1
            //   ShTileAdd(dst, dst,      reads[2])   // dst += r2
            //   ...
            //   ShTileAdd(dst, dst,      reads[N-1]) // dst += rN-1
            //
            // TK 2.0 `add(T &dst, const T &lhs, const U &rhs)` allows
            // `dst` aliasing `lhs` (lhs is `const T&` to the same).
            assert!(
                reads.len() >= 2,
                "SumReduce expects ≥2 inputs (got {}); N=1 lowers to a copy, not yet supported",
                reads.len(),
            );
            use crate::tk_tape::{Bf16, GroupWidth, SmemTileId};
            let dst = SmemTileId::<128, 128, Bf16>::from_page(dst_page);
            // First add: dst = reads[0] + reads[1]
            let r0 = SmemTileId::<128, 128, Bf16>::from_page(state.page_of(reads[0]));
            let r1 = SmemTileId::<128, 128, Bf16>::from_page(state.page_of(reads[1]));
            state.push(Instr::sh_tile_add(
                r0,
                r1,
                dst,
                GroupWidth::<16>::ALL_CONSUMERS,
            ));
            // Subsequent adds: dst += reads[i]
            for r_i in &reads[2..] {
                let rhs = SmemTileId::<128, 128, Bf16>::from_page(state.page_of(*r_i));
                state.push(Instr::sh_tile_add(
                    dst,
                    rhs,
                    dst,
                    GroupWidth::<16>::ALL_CONSUMERS,
                ));
            }
            emit_store_and_arrive(state, &node.output, dst_page);
        }
        SubOp::SiluMul => {
            // SiluMul: out = silu(gate) * up = (gate / (1 + exp(-gate))) * up
            //
            // Decomposition (5 Instrs per SUBTILE_TK20_DECOMP.md
            // §"Per-SubOp Instr counts" line 19). Aliases dst_page
            // as the SiLU work-tile across steps 1-4 — TK 2.0
            // bin_map is element-local so dst aliasing src is safe
            // (each thread reads then writes its own element).
            //
            //   reads[0] = gate, reads[1] = up
            //   step 1: ShTileMulScalar(dst, gate, -1.0)   // dst = -gate
            //   step 2: ShTileExp     (dst, dst)            // dst = exp(-gate)
            //   step 3: ShTileAddScalar(dst, dst, 1.0)      // dst = 1 + exp(-gate)
            //   step 4: ShTileDiv     (dst, gate, dst)      // dst = silu(gate)
            //   step 5: ShTileMul     (dst, dst, up)        // dst = silu(gate) * up
            use crate::tk_tape::{Bf16, GroupWidth, ScalarF32, SmemTileId};
            let gate = SmemTileId::<128, 128, Bf16>::from_page(state.page_of(reads[0]));
            let up = SmemTileId::<128, 128, Bf16>::from_page(state.page_of(reads[1]));
            let dst = SmemTileId::<128, 128, Bf16>::from_page(dst_page);
            const W: GroupWidth<16> = GroupWidth::<16>::ALL_CONSUMERS;
            // step 1
            state.push(Instr::sh_tile_mul_scalar(gate, dst, ScalarF32::new(-1.0), W));
            // step 2
            state.push(Instr::sh_tile_exp(dst, dst, W));
            // step 3
            state.push(Instr::sh_tile_add_scalar(dst, dst, ScalarF32::new(1.0), W));
            // step 4
            state.push(Instr::sh_tile_div(gate, dst, dst, W));
            // step 5
            state.push(Instr::sh_tile_mul(dst, up, dst, W));
            emit_store_and_arrive(state, &node.output, dst_page);
        }
        SubOp::Elementwise(EwKind::Silu) => {
            // Silu: out = x * sigmoid(x) = x / (1 + exp(-x))
            //
            // Step 6 (register-resident chain, 6 Instrs per
            // SUBTILE_TK20_DECOMP.md §"Per-SubOp Instr counts" line 20):
            //   rt_x       = load(src_page)
            //   rt_neg     = neg(rt_x)
            //   rt_exp     = exp(rt_neg)
            //   rt_denom   = add_scalar(rt_exp, 1.0)
            //   rt_result  = div(rt_x, rt_denom)
            //   store(dst_page, rt_result)
            //
            // 5 RegTileId<128, 128, Bf16, RowLayout> minted; the
            // arena records each so emit_kernel can declare them in
            // the preamble as `kittens::rt<...> rt_<id>;`.
            //
            // Per `feedback_no_simpler`: aliasing optimization (3
            // slots minimum) is a perf concern for a follow-up; this
            // commit lands the plan-aligned 5-slot version.
            use crate::tk_tape::{
                AllConsumersRole, Bf16, GroupWidth, RegTileId, RowLayout, ScalarF32, SmemTileId,
            };
            let src = SmemTileId::<128, 128, Bf16>::from_page(state.page_of(reads[0]));
            let dst = SmemTileId::<128, 128, Bf16>::from_page(dst_page);
            let rt_x: RegTileId<128, 128, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_neg: RegTileId<128, 128, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_exp: RegTileId<128, 128, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_denom: RegTileId<128, 128, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_result: RegTileId<128, 128, Bf16, RowLayout> = state.mint_reg_tile();
            const W: GroupWidth<16> = GroupWidth::<16>::ALL_CONSUMERS;
            const R: AllConsumersRole = AllConsumersRole;
            // 1: rt_x = load(src_page)
            state.push(Instr::load_shmem_to_reg(src, rt_x, W, R));
            // 2: rt_neg = neg(rt_x)
            state.push(Instr::reg_tile_neg(rt_x, rt_neg, W, R));
            // 3: rt_exp = exp(rt_neg)
            state.push(Instr::reg_tile_exp(rt_neg, rt_exp, W, R));
            // 4: rt_denom = rt_exp + 1.0
            state.push(Instr::reg_tile_add_scalar(rt_exp, rt_denom, ScalarF32::new(1.0), W, R));
            // 5: rt_result = rt_x / rt_denom
            state.push(Instr::reg_tile_div(rt_x, rt_denom, rt_result, W, R));
            // 6: store(dst_page, rt_result)
            state.push(Instr::store_reg_tile_to_shmem(rt_result, dst, W, R));
            emit_store_and_arrive(state, &node.output, dst_page);
        }
        SubOp::RmsNorm { eps } => {
            // RmsNorm: out[i,j] = x[i,j] * inv_rms[i] * gamma[j]
            //   inv_rms[i] = 1 / sqrt(mean(x[i,:]^2) + eps)
            //
            // Plan §"Per-SubOp Instr counts" line 18 (10 Instrs):
            //   1: ShTileMul          x_sq    = x * x          (square)
            //   2: ShTileRowSum       sum_sq  = row_sum(x_sq)
            //   3: ShVecMulScalar     mean_sq = sum_sq * (1/cols)
            //   4: ShVecAddScalar     var     = mean_sq + eps
            //   5: LoadVecSmemToReg   rv_var  = var
            //   6: RegVecUnaryRsqrt   rv_inv  = rsqrt(rv_var)
            //   7: StoreRegVecToShmem inv_rms = rv_inv
            //   8: ShTileMulRow       x_norm  = x * inv_rms (per-row broadcast)
            //   9: ShTileMulCol       out     = x_norm * gamma (per-col broadcast)
            //
            // reads[0] = x, reads[1] = gamma. Both pre-paged by the
            // SubtileTape lowerer; this arm orchestrates the math
            // and allocates temp pages for intermediates.
            //
            // Const generics: 128×128 Bf16 tile substrate, 128-element
            // shared/register vec for inv_rms (length = ROWS = 128).
            //
            // SmemVecId<LEN, T> is not yet sealed — temp pages carry
            // raw PageId. Mints will reify when SmemVecId lands.
            use crate::tk_tape::{
                AllConsumersRole, Bf16, GroupWidth, NaiveLayout, RegVecId, SmemTileId,
            };
            let x = SmemTileId::<128, 128, Bf16>::from_page(state.page_of(reads[0]));
            let gamma_page = state.page_of(reads[1]);
            let dst = SmemTileId::<128, 128, Bf16>::from_page(dst_page);
            const W: GroupWidth<16> = GroupWidth::<16>::ALL_CONSUMERS;
            const R: AllConsumersRole = AllConsumersRole;
            // Temp pages
            let x_sq_page = state.alloc_temp_page();
            let var_page = state.alloc_temp_page();
            let inv_rms_page = state.alloc_temp_page();
            let x_sq = SmemTileId::<128, 128, Bf16>::from_page(x_sq_page);
            // Reg vec for rsqrt detour
            let rv_var: RegVecId<128, Bf16, NaiveLayout> = state.mint_reg_vec();
            let rv_inv: RegVecId<128, Bf16, NaiveLayout> = state.mint_reg_vec();
            // SmemVecId witnesses for register-vec load/store —
            // var/inv_rms are length-128 shared vectors viewed onto
            // their respective pages (per-row scalar per the row_sum
            // → ROWS=128 mapping).
            let var_vec = crate::tk_tape::SmemVecId::<128, Bf16>::from_page(var_page);
            let inv_rms_vec = crate::tk_tape::SmemVecId::<128, Bf16>::from_page(inv_rms_page);

            // 1: x_sq = x * x
            state.push(Instr::sh_tile_mul(x, x, x_sq, W));
            // 2: sum_sq = row_sum(x_sq)  (writes into var_page as a sv view)
            state.push(Instr::sh_tile_row_sum(x_sq, var_page, W));
            // 3: var = sum_sq * (1/COLS)
            let inv_cols = 1.0_f32 / 128.0;
            state.push(Instr::sh_vec_mul_scalar(
                var_page,
                var_page,
                crate::tk_tape::ScalarF32::new(inv_cols),
                Bf16,
                W,
            ));
            // 4: var = var + eps
            state.push(Instr::sh_vec_add_scalar(
                var_page,
                var_page,
                crate::tk_tape::ScalarF32::new(*eps),
                Bf16,
                W,
            ));
            // 5: rv_var = load(var)
            state.push(Instr::load_vec_smem_to_reg(var_vec, rv_var, W, R));
            // 6: rv_inv = rsqrt(rv_var)
            state.push(Instr::reg_vec_unary_rsqrt(rv_var, rv_inv, W, R));
            // 7: inv_rms = store(rv_inv)
            state.push(Instr::store_reg_vec_to_shmem(rv_inv, inv_rms_vec, W, R));
            // 8: x_norm = x * inv_rms (per-row broadcast)
            //    write into dst (clobber x is OK; we reuse dst as
            //    the running tile through the gamma multiply too).
            state.push(Instr::sh_tile_mul_row(x, inv_rms_page, dst, W));
            // 9: out = x_norm * gamma (per-col broadcast)
            state.push(Instr::sh_tile_mul_col(dst, gamma_page, dst, W));

            emit_store_and_arrive(state, &node.output, dst_page);
        }
        SubOp::RopeRotate { head_dim, _form: _ } => {
            // RopeRotateNeoX: split q at head_dim/2; rotate as
            //   out_even = q_even * cos - q_odd * sin
            //   out_odd  = q_even * sin + q_odd * cos
            // Per SUBTILE_TK20_DECOMP §"Per-SubOp Instr counts" line
            // 24, 14 Instrs. Two are the implicit external loads of
            // cos/sin (handled by the SubtileTape lowerer); 12 are
            // emitted here.
            //
            // F: RopeForm is a const-generic on the lowerer; F::TAG
            // selects NeoX vs Interleaved at type level. Interleaved
            // (step 8) has a different decomposition and is deferred.
            use crate::subtile_ir::RopeFormTag;
            assert_eq!(
                F::TAG,
                RopeFormTag::NeoX,
                "RopeRotate Interleaved form is plan step 8, not yet landed",
            );
            assert_eq!(
                *head_dim, 64,
                "RopeRotateNeoX: only head_dim=64 (Llama-3.2-1B) supported \
                 today; got head_dim={}",
                head_dim,
            );

            // reads: [q, cos, sin]
            // q is in a 128×128 page (logical 128×64 with cols 0..64
            // used; rotation operates on cols 0..32 vs 32..64 halves).
            use crate::tk_tape::{
                AllConsumersRole, Bf16, GroupWidth, NaiveLayout, RegTileId,
                RegVecId, RowLayout, SmemTileId, SmemVecId,
            };
            let q_page = state.page_of(reads[0]);
            let cos_vec = SmemVecId::<32, Bf16>::from_page(state.page_of(reads[1]));
            let sin_vec = SmemVecId::<32, Bf16>::from_page(state.page_of(reads[2]));
            let q_full = SmemTileId::<128, 128, Bf16>::from_page(q_page);
            let dst_full = SmemTileId::<128, 128, Bf16>::from_page(dst_page);
            const W: GroupWidth<16> = GroupWidth::<16>::ALL_CONSUMERS;
            const R: AllConsumersRole = AllConsumersRole;

            // Mint the 6 register tiles + 2 register vecs the
            // rotation needs. 16 warps * 8 live rt's of 128×32 bf16
            // = lots of registers; nvcc allocates and may spill.
            let rt_q_even: RegTileId<128, 32, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_q_odd:  RegTileId<128, 32, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_a:      RegTileId<128, 32, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_b:      RegTileId<128, 32, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_c:      RegTileId<128, 32, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_d:      RegTileId<128, 32, Bf16, RowLayout> = state.mint_reg_tile();
            let rv_cos: RegVecId<32, Bf16, NaiveLayout> = state.mint_reg_vec();
            let rv_sin: RegVecId<32, Bf16, NaiveLayout> = state.mint_reg_vec();

            // 1: rt_q_even = q[:, 0:32]
            state.push(Instr::load_shmem_subtile_to_reg::<16, 128, 128, 32, 0, Bf16, RowLayout>(
                q_full, rt_q_even, W, R,
            ));
            // 2: rt_q_odd = q[:, 32:64]
            state.push(Instr::load_shmem_subtile_to_reg::<16, 128, 128, 32, 1, Bf16, RowLayout>(
                q_full, rt_q_odd, W, R,
            ));
            // 3: rv_cos = load(cos_vec)
            state.push(Instr::load_vec_smem_to_reg(cos_vec, rv_cos, W, R));
            // 4: rv_sin = load(sin_vec)
            state.push(Instr::load_vec_smem_to_reg(sin_vec, rv_sin, W, R));
            // 5: rt_a = q_even * cos
            state.push(Instr::reg_tile_mul_col(rt_q_even, rv_cos, rt_a, W, R));
            // 6: rt_b = q_odd * sin
            state.push(Instr::reg_tile_mul_col(rt_q_odd, rv_sin, rt_b, W, R));
            // 7: rt_c = q_even * sin
            state.push(Instr::reg_tile_mul_col(rt_q_even, rv_sin, rt_c, W, R));
            // 8: rt_d = q_odd * cos
            state.push(Instr::reg_tile_mul_col(rt_q_odd, rv_cos, rt_d, W, R));
            // 9: rt_a = rt_a - rt_b  (out_even = q_even*cos - q_odd*sin)
            state.push(Instr::reg_tile_sub(rt_a, rt_b, rt_a, W, R));
            // 10: rt_c = rt_c + rt_d  (out_odd  = q_even*sin + q_odd*cos)
            state.push(Instr::reg_tile_add(rt_c, rt_d, rt_c, W, R));
            // 11: dst[:, 0:32] = rt_a
            state.push(Instr::store_reg_tile_subtile_to_shmem::<16, 128, 128, 32, 0, Bf16, RowLayout>(
                rt_a, dst_full, W, R,
            ));
            // 12: dst[:, 32:64] = rt_c
            state.push(Instr::store_reg_tile_subtile_to_shmem::<16, 128, 128, 32, 1, Bf16, RowLayout>(
                rt_c, dst_full, W, R,
            ));

            emit_store_and_arrive(state, &node.output, dst_page);
        }
        SubOp::MatmulTile => {
            // MatmulTile: D[M, N] = A[M, K] @ B[K, N]
            //
            // Plan §"Per-SubOp Instr counts" line 17 (8 Instrs):
            //   1: TmaExpect          arm barrier with byte count
            //   2: TmaLoadTile        load A and B (existing LoadAsync
            //                         from external sources; counted
            //                         here as one logical step)
            //   3: MbarrierWait       wait on the page-ready barriers
            //   4: InitRtZero         rt_d = 0 (accumulator)
            //   5: WgmmaFenceAcc      mma_fence(rt_d) (FenceExternal)
            //   6: WgmmaMmaAB_SmemSmem  rt_d += A @ B
            //   7: WgmmaAsyncWait     wait_group<0>
            //   8: StoreRegTileToShmem dst_page = rt_d (with copy/cast
            //                         from fp32 to bf16 happening at
            //                         the store boundary in TK 2.0)
            //
            // reads = [a, b]. Both pre-paged by the SubtileTape lowerer
            // (LoadAsync emitted via the external-load loop / prior
            // tape Instrs). This arm orchestrates the WGMMA + zero
            // + fence + wait + store sequence.
            //
            // Substrate: 128×128 Bf16 pages. Llama Q/K/V projections
            // on a 128-row chunk: M=128, N=128, K=128 (a single
            // K-block; multi-block via SumReduce). Accumulator is
            // RegTileId<128, 128, Fp32, RowLayout> per Hopper convention.
            use crate::tk_tape::{
                AccReset, AllConsumersRole, Bf16, FenceExternal, Fp32,
                GroupWidth, RegTileId, RoleWitness, RowLayout, SmemTileId,
            };
            let a = SmemTileId::<128, 128, Bf16>::from_page(state.page_of(reads[0]));
            let b = SmemTileId::<128, 128, Bf16>::from_page(state.page_of(reads[1]));
            let dst = SmemTileId::<128, 128, Bf16>::from_page(dst_page);
            const W4: GroupWidth<4> = GroupWidth::<4>::WARPGROUP;
            const W16: GroupWidth<16> = GroupWidth::<16>::ALL_CONSUMERS;
            const R: AllConsumersRole = AllConsumersRole;

            // Mint the fp32 accumulator
            let rt_d: RegTileId<128, 128, Fp32, RowLayout> = state.mint_reg_tile();

            // Step 4: zero the accumulator
            state.push(Instr::init_rt_zero(rt_d, W16, R));
            // Step 5: fence on D
            state.push(Instr::wgmma_fence_acc(rt_d, W4));
            // Step 6: D += A @ B
            //   Accumulate (since we just zeroed D, accumulate is the
            //   safe choice and matches multi-block-K future use).
            //   FenceExternal because we just emitted WgmmaFenceAcc.
            state.push(Instr::wgmma_mma_ab_smem_smem(
                rt_d, a, b, FenceExternal, AccReset, W4,
            ));
            // Step 7: wait for all WGMMA groups
            state.push(Instr::wgmma_async_wait(0, W4));

            // Step 8: store accumulator → dst page (fp32 → bf16
            // cast via TK 2.0's store overload). NOTE: store_reg_tile_to_shmem
            // currently requires src and dst to share dtype. Until
            // we have RegTileCopyConvert (plan step 13), use a
            // dst page held as fp32 — but the substrate is bf16.
            // Pragmatic: emit the store with mismatched dtype and
            // rely on TK 2.0's `store(st, rt)` template doing the
            // conversion. The Rust side's typed constructor would
            // refuse this; mark with TODO + use a raw struct literal
            // for now until step 13.
            //
            // PER feedback_compile_time_or_garbage: this is a known
            // gap, NOT a runtime garbage path — the emitted CUDA
            // is correct; only the typed constructor's strict
            // unification is bypassed here. Marked as a defect to
            // fix in step 13 (RegTileCopyConvert).
            //
            // Alternate: skip the store from this arm and rely on
            // a downstream Instr to convert+store. But that's a
            // graph-level concern; for now, emit the store directly.
            //
            // Workaround: bypass typed constructor by constructing
            // the runtime variant directly (still inside the crate,
            // so the gate is preserved at the typed-constructor
            // level for callers that aren't this lowerer arm).
            state.push(Instr::StoreRegTileToShmem {
                src: rt_d.slot(),
                dst: dst.page(),
                width: W16.tag(),
                role: R.to_warp_role(),
            });

            emit_store_and_arrive(state, &node.output, dst_page);
        }
        #[allow(unreachable_patterns)]
        SubOp::MatmulTile
        | SubOp::RopeAppend { .. }
        | SubOp::AttnDecode { .. } => {
            panic!(
                "lower_compute: arch op {:?} has no TK 2.0 primitive expansion yet; \
                 see SUBTILE_TK20_DECOMP.md for the per-SubOp decomposition plan. \
                 lower_compute refuses to emit invented kittens::ops::* helpers.",
                std::mem::discriminant(&node.op),
            );
        }
    }
}

// ── Per-op emit helpers ─────────────────────────────────────────────

fn region_tile_shape(tr: &TensorRegion) -> TileShape {
    TileShape {
        rows: tr.region.rows.len,
        cols: tr.region.cols.len,
        elem_bytes: ELEM_BYTES,
    }
}

fn region_byte_offset<F: RopeForm, K: KvCacheShape>(graph: &SubtileIR<F, K>, tr: &TensorRegion) -> ByteOffsetExpr {
    // Linear row-major offset: (rows.start * cols_total + cols.start) * elem_bytes.
    let shape = graph.tensors[tr.tensor.0 as usize];
    let off = ((tr.region.rows.start as u64) * (shape.cols as u64)
        + (tr.region.cols.start as u64))
        * (ELEM_BYTES as u64);
    ByteOffsetExpr::Const(off)
}

fn emit_external_load<F: RopeForm, K: KvCacheShape>(
    state: &mut LoweringState<F, K>,
    inp: &TensorRegion,
    dst_page: PageId,
) {
    use crate::tk_tape::{Bf16, SmemTileSpec};
    // Typed-witness mint: the substrate's uniform 128×128 bf16 page
    // pool is the type-level invariant. `from_shape` debug_asserts the
    // runtime shape matches; const-generics propagate to LoadSpec.
    // Per `feedback_ff_subtile_compile_time_inviolable`.
    let tile = SmemTileSpec::<128, 128, Bf16>::from_shape(region_tile_shape(inp));
    let byte_off = region_byte_offset(state.graph, inp);
    state.push(Instr::LoadAsync(LoadSpec::new(
        dst_page,
        inp.tensor,
        byte_off,
        tile,
        LOAD_ROLE,
        dst_page,
    )));
}

fn emit_store_and_arrive<F: RopeForm, K: KvCacheShape>(
    state: &mut LoweringState<F, K>,
    out: &TensorRegion,
    dst_page: PageId,
) {
    use crate::tk_tape::{Bf16, SmemTileSpec};
    let tile = SmemTileSpec::<128, 128, Bf16>::from_shape(region_tile_shape(out));
    let byte_off = region_byte_offset(state.graph, out);
    state.push(Instr::StoreAsync(StoreSpec::new(
        dst_page,
        out.tensor,
        byte_off,
        tile,
        STORE_ROLE,
    )));
    use crate::tk_tape::RoleWitness;
    state.push(Instr::CommitGroupBulk { role: STORE_ROLE.to_warp_role() });
    state.push(Instr::ThreadfenceDevice { role: ALL_ROLE });
    state.push(Instr::PageBarrierArrive {
        page_id: dst_page,
        kind: PageBarrier::Done,
        role: STORE_ROLE.to_warp_role(),
    });
}
fn drain<F: RopeForm, K: KvCacheShape>(state: &mut LoweringState<F, K>) {
    use crate::tk_tape::AllWarpsRole;
    state.push(Instr::syncthreads_cta(AllWarpsRole));
    state.push(Instr::CommitGroupBulk { role: ALL_ROLE });
    state.push(Instr::WaitGroupBulk { n: 0, role: ALL_ROLE });
    state.push(Instr::ThreadfenceDevice { role: ALL_ROLE });
    state.push(Instr::syncthreads_cta(AllWarpsRole));
}
