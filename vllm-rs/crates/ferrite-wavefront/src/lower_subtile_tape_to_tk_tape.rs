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
    ByteOffset, Instr, KernelArg, KernelArgName, KernelArgRef, KernelArgTy,
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

const LOAD_ROLE: WarpRole = WarpRole::Loader;
const STORE_ROLE: WarpRole = WarpRole::Storer;
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
        }
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
    _writes: SlotId,
    _reads: &[SlotId],
) {
    // The architectural compute lowering — emit_matmul_tile,
    // emit_sum_reduce, emit_elementwise, emit_silu_mul, emit_rmsnorm,
    // emit_rope_rotate, emit_rope_append, emit_attn_decode — was
    // nuked because it composed invented `kittens::ops::*` calls.
    //
    // Per plan §1 line 64-65 + INVIOLABLE feedback_tk_2_0_only /
    // feedback_tk20_primitives_first, every Compute Instr must be a
    // real TK 2.0 primitive call from `third_party/thunderkittens/
    // include/`. Each architectural SubOp must decompose into a
    // SEQUENCE of primitive Instrs (e.g. RmsNorm = warp::row_squared
    // + shared_tile::row_sum + thread::rsqrt + warp::mul_row + ...)
    // emitted by this lowering. None of that expansion exists yet;
    // it lands alongside per-arch-op work.
    //
    // Until then, lower_compute refuses to fabricate a tape.
    let node = &state.graph.nodes[node_id.0 as usize];
    panic!(
        "lower_compute: arch op {:?} has no TK 2.0 primitive expansion yet; \
         see SUBTILE_TAPE_HANDOFF.md — emit_* helpers were nuked alongside the \
         invented kittens::ops::* calls they used to produce",
        std::mem::discriminant(&node.op),
    );
}

// ── Per-op emit helpers ─────────────────────────────────────────────

fn region_tile_shape(tr: &TensorRegion) -> TileShape {
    TileShape {
        rows: tr.region.rows.len,
        cols: tr.region.cols.len,
        elem_bytes: ELEM_BYTES,
    }
}

fn region_byte_offset<F: RopeForm, K: KvCacheShape>(graph: &SubtileIR<F, K>, tr: &TensorRegion) -> ByteOffset {
    // Linear row-major offset: (rows.start * cols_total + cols.start) * elem_bytes.
    let shape = graph.tensors[tr.tensor.0 as usize];
    let off = ((tr.region.rows.start as u64) * (shape.cols as u64)
        + (tr.region.cols.start as u64))
        * (ELEM_BYTES as u64);
    ByteOffset::from_const(off)
}

fn emit_external_load<F: RopeForm, K: KvCacheShape>(
    state: &mut LoweringState<F, K>,
    inp: &TensorRegion,
    dst_page: PageId,
) {
    let tile = region_tile_shape(inp);
    let byte_off = region_byte_offset(state.graph, inp);
    state.push(Instr::LoadAsync(LoadSpec {
        dst_page,
        src_tensor: inp.tensor,
        byte_off,
        tile,
        role: LOAD_ROLE,
        barrier_page: dst_page,
    }));
}

fn emit_store_and_arrive<F: RopeForm, K: KvCacheShape>(
    state: &mut LoweringState<F, K>,
    out: &TensorRegion,
    dst_page: PageId,
) {
    let tile = region_tile_shape(out);
    let byte_off = region_byte_offset(state.graph, out);
    state.push(Instr::StoreAsync(StoreSpec {
        src_page: dst_page,
        dst_tensor: out.tensor,
        byte_off,
        tile,
        role: STORE_ROLE,
    }));
    state.push(Instr::CommitGroupBulk { role: STORE_ROLE });
    state.push(Instr::ThreadfenceDevice { role: ALL_ROLE });
    state.push(Instr::PageBarrierArrive {
        page_id: dst_page,
        kind: PageBarrier::Done,
        role: STORE_ROLE,
    });
}
fn drain<F: RopeForm, K: KvCacheShape>(state: &mut LoweringState<F, K>) {
    state.push(Instr::SyncthreadsCta { role: ALL_ROLE });
    state.push(Instr::CommitGroupBulk { role: ALL_ROLE });
    state.push(Instr::WaitGroupBulk { n: 0, role: ALL_ROLE });
    state.push(Instr::ThreadfenceDevice { role: ALL_ROLE });
    state.push(Instr::SyncthreadsCta { role: ALL_ROLE });
}
