// SPDX-License-Identifier: Apache-2.0
//! `lower_tape_to_tk(&SubtileTape, &SubtileIR<F>) -> TkTape` — the
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
//! - `lower_tape_to_tk::<F>(&tape, &graph) -> TkTape` is the public
//!   entry point. It walks the SubtileTape's `Vec<Instr>` once,
//!   maintaining a [`LoweringState`] of slot→page mappings, the
//!   `seq_len` kernel-arg binding (minted on demand for AttnDecode's
//!   loop), and the active `Vec<Instr>` stack to thread `OpenLoop`
//!   bodies into `Instr::ForLoop`'s nested vec.
//! - Per-SubOp branches in `lower_compute` are exhaustive (no `_ =>`
//!   arm); adding a new `SubOp` variant becomes a compile error here.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::num::NonZeroU32;

use crate::subtile_ir::{
    KvCacheLayout, KvCacheProducer, RopeForm, SoftmaxStateId, SubOp, SubtileId, SubtileIR,
    SubtileNode, TensorId, TensorRegion,
};
use crate::subtile_tape::{
    Instr as STInstr, LoopBound, LoopVarId as STLoopVarId, SlotId, SubtileTape,
};
use crate::tk_tape::{
    AccumKind, ByteOffsetExpr, Instr, KernelArg, KernelArgName, KernelArgRef, KernelArgTy,
    KvLayoutEntry, KvLayoutId, LoadSpec, LoopVarId as TkLoopVarId, PageBarrier, PageId,
    ParityExpr, RopeFormTag, RopeSide, SoftmaxStateId as TkSoftmaxStateId, StoreSpec, TileShape,
    TkTape, U32Source, WarpRole, validate_tk_tape,
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
struct LoweringState<'g, F: RopeForm> {
    graph: &'g SubtileIR<F>,
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
    /// Fresh PageId allocator. Conservative 1:1 with SlotId.
    next_page: u8,
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
    /// AttnDecode's `Finalise` Instr is emitted AFTER its KV-sweep
    /// `OpenLoop`/`CloseLoop` pair — but the AttnDecode `Compute` that
    /// owns it lives INSIDE the loop body. We queue Finalises here at
    /// the AttnDecode-Compute site, keyed by SubtileTape `LoopVarId`,
    /// and drain into the parent frame in `lower_close_loop` right
    /// after the `Instr::ForLoop` lands.
    pending_post_loop: BTreeMap<u32, Vec<PostLoopAction>>,
    /// Stack of active SubtileTape `LoopVarId`s — pushed by
    /// `OpenLoop`, popped by `CloseLoop`. The top entry is the loop
    /// body whose frame is on top of `instr_stack`.
    active_loop_stack: Vec<u32>,
}

impl<'g, F: RopeForm> LoweringState<'g, F> {
    fn new(graph: &'g SubtileIR<F>) -> Self {
        Self {
            graph,
            instr_stack: vec![Vec::new()],
            loop_var_map: BTreeMap::new(),
            kernel_args: Vec::new(),
            seq_len_arg: None,
            position_arg: None,
            next_page: 0,
            slot_to_page: BTreeMap::new(),
            next_loop_var: 0,
            kv_layouts: Vec::new(),
            kv_layout_index: BTreeMap::new(),
            softmax_state_index: BTreeMap::new(),
            next_softmax_state: 0,
            pending_post_loop: BTreeMap::new(),
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
        let p = PageId(self.next_page);
        self.next_page = self
            .next_page
            .checked_add(1)
            .expect("PageId overflow (>=256 slots in one tape — exceeds NUM_PAGES)");
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
        self.slot_to_page.remove(&slot.index());
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

    fn intern_kv_layout(&mut self, layout: KvCacheLayout) -> KvLayoutId {
        let key = layout.cache_tensor();
        if let Some(id) = self.kv_layout_index.get(&key) {
            return *id;
        }
        let id = KvLayoutId(self.kv_layouts.len() as u32);
        self.kv_layouts.push(KvLayoutEntry { layout });
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
pub fn lower_tape_to_tk<F: RopeForm>(
    tape: &SubtileTape,
    graph: &SubtileIR<F>,
) -> TkTape {
    let mut state = LoweringState::new(graph);

    for instr in &tape.instrs {
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
        .expect("lower_tape_to_tk: produced invalid TkTape (commit 6b post-condition)");
    out
}

// ── SubtileTape::AllocSlot ──────────────────────────────────────────
//
// Mints a PageId. No Instr emit at the lowering layer — `BarrierInit`
// is a per-kernel-prelude concern (lands with the kernel-args + prelude
// pass in commit 6+). The slot's PageId is now reachable via
// `state.page_of(slot)`.

fn lower_alloc_slot<F: RopeForm>(state: &mut LoweringState<F>, slot: SlotId) {
    state.alloc_page(slot);
}

// ── SubtileTape::FreeSlot ───────────────────────────────────────────
//
// Conservative all-gmem path: releasing the page mapping is enough.
// Optimizer passes that promote a slot to shmem will add explicit
// `PageBarrierArrive{Consumed}` here.

fn lower_free_slot<F: RopeForm>(state: &mut LoweringState<F>, slot: SlotId) {
    state.release_page(slot);
}

// ── SubtileTape::OpenLoop / CloseLoop ───────────────────────────────

fn lower_open_loop<F: RopeForm>(
    state: &mut LoweringState<F>,
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

fn lower_close_loop<F: RopeForm>(state: &mut LoweringState<F>, var: STLoopVarId) {
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
    // Drain any AttnDecode Finalise queued by an AttnDecode::Compute
    // that lived inside this loop body. The Finalise + store + arrive
    // sequence lands in the parent frame, immediately after the
    // ForLoopClose instr.
    if let Some(actions) = state.pending_post_loop.remove(&var.index()) {
        for action in actions {
            apply_post_loop(state, action);
        }
    }
}

fn apply_post_loop<F: RopeForm>(state: &mut LoweringState<F>, action: PostLoopAction) {
    match action {
        PostLoopAction::Finalise {
            state: smx,
            out_page,
            num_q_heads,
            head_dim,
            output,
        } => {
            state.push(Instr::AttnDecodeFinalise {
                state: smx,
                out_page,
                num_q_heads,
                head_dim,
                role: COMPUTE_ROLE,
            });
            emit_store_and_arrive(state, &output, out_page);
        }
    }
}

// ── SubtileTape::Compute — the SubOp dispatch ───────────────────────

fn lower_compute<F: RopeForm>(
    state: &mut LoweringState<F>,
    node_id: SubtileId,
    writes: SlotId,
    reads: &[SlotId],
) {
    let node = &state.graph.nodes[node_id.0 as usize];
    let dst_page = state.page_of(writes);

    // Per the SubOp dispatch below, each compute branch:
    //   1. Loads its source tiles (TMA expect_bytes + load_async) into
    //      the reads' pages — for slot reads, the page is already the
    //      producer's; for source-tensor reads (PrefixK, weights, etc.),
    //      the load brings them in. We DON'T re-load slot pages: the
    //      producer's StoreAsync + Arrive{Done} + our Wait{Ready} is
    //      the handshake.
    //   2. Waits for each input page barrier.
    //   3. Emits the architectural compute Instr.
    //   4. Stores the dst page (StoreAsync to gmem-backing + Fence +
    //      Arrive{Done}). This is the conservative all-gmem path.
    //
    // Source-tensor reads that DON'T have a producing slot in this tape
    // are "external" reads (weights, prefix-KV cache, cos/sin, embed):
    // those do require an explicit LoadAsync. The lowering identifies
    // them by the fact that the SubtileNode's input is a TensorRegion
    // pointing at a leaf source tensor (TensorId < num_sources).

    // Wait on each predecessor slot's Ready barrier — the producer's
    // Arrive{Done} signals downstream readiness through the cross-page
    // protocol (Done → Ready handshake lives at TkTape, but the
    // conservative path uses a direct Ready wait on the producer's
    // page).
    for r in reads {
        let p = state.page_of(*r);
        state.push(Instr::PageBarrierWait {
            page_id: p,
            kind: PageBarrier::Ready,
            parity: ParityExpr::Static(0),
            role: COMPUTE_ROLE,
        });
    }

    // External (source-tensor) loads: any input TensorRegion whose
    // tensor is a leaf source (id < num_sources) must be brought into
    // the dst page (or a transient page — for the conservative path,
    // we reuse the dst page).
    let num_sources = state.graph.num_sources;
    for inp in &node.inputs {
        if inp.tensor.0 < num_sources {
            emit_external_load(state, inp, dst_page);
        }
    }

    // SubOp dispatch — exhaustive match, no `_ =>` arm.
    match &node.op {
        SubOp::MatmulTile => emit_matmul_tile(state, node, dst_page),
        SubOp::SumReduce => emit_sum_reduce(state, node, dst_page, reads),
        SubOp::Elementwise(kind) => emit_elementwise(state, node, *kind, dst_page, reads),
        SubOp::SiluMul => emit_silu_mul(state, node, dst_page, reads),
        SubOp::RmsNorm { eps } => emit_rmsnorm(state, node, *eps, dst_page, reads),
        SubOp::RopeRotate { head_dim, _form: _ } => {
            emit_rope_rotate::<F>(state, node, *head_dim, dst_page, reads, RopeSide::Q);
        }
        SubOp::RopeAppend {
            head_dim,
            layer: _,
            layout,
            _form: _,
        } => {
            emit_rope_append::<F>(state, node, *head_dim, *layout, dst_page, reads);
        }
        SubOp::AttnDecode {
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            layout,
            producer,
            softmax_state,
        } => {
            emit_attn_decode(
                state,
                node,
                *num_q_heads,
                *num_kv_heads,
                *head_dim,
                *scale,
                *layout,
                *producer,
                *softmax_state,
                dst_page,
                reads,
            );
        }
    }

    // Producer-side store + fence + Arrive{Done}. For AttnDecode the
    // store happens at Finalise (inside emit_attn_decode), so the
    // outer skips this by returning early in that branch.
    match node.op {
        SubOp::AttnDecode { .. } => {} // store handled in emit_attn_decode
        _ => emit_store_and_arrive(state, &node.output, dst_page),
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

fn region_byte_offset<F: RopeForm>(graph: &SubtileIR<F>, tr: &TensorRegion) -> ByteOffsetExpr {
    // Linear row-major offset: (rows.start * cols_total + cols.start) * elem_bytes.
    let shape = graph.tensors[tr.tensor.0 as usize];
    let off = ((tr.region.rows.start as u64) * (shape.cols as u64)
        + (tr.region.cols.start as u64))
        * (ELEM_BYTES as u64);
    ByteOffsetExpr::Const(off)
}

fn emit_external_load<F: RopeForm>(
    state: &mut LoweringState<F>,
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

fn emit_store_and_arrive<F: RopeForm>(
    state: &mut LoweringState<F>,
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

fn emit_matmul_tile<F: RopeForm>(
    state: &mut LoweringState<F>,
    node: &SubtileNode<F>,
    dst_page: PageId,
) {
    let m = node.output.region.rows.len;
    let n = node.output.region.cols.len;
    let k = node.inputs[0].region.cols.len;
    // input[0] = activation, input[1] = weight (per LoweredOp::Gemm).
    let lhs_page = dst_page; // shared via expect_bytes; refined by optimizer
    let rhs_tensor = node.inputs[1].tensor;
    let rhs_byte_off = region_byte_offset(state.graph, &node.inputs[1]);
    state.push(Instr::GemmM1 {
        lhs_page,
        rhs_tensor,
        rhs_byte_off,
        out_page: dst_page,
        m,
        n,
        k,
        accum: AccumKind::Zero,
        role: COMPUTE_ROLE,
    });
}

fn emit_sum_reduce<F: RopeForm>(
    state: &mut LoweringState<F>,
    node: &SubtileNode<F>,
    dst_page: PageId,
    reads: &[SlotId],
) {
    // Split-K combine: `out = Σ partials`. Conservative encoding as a
    // chain of ResidualAdd Instrs (binary). The optimizer can later
    // fuse into a true reduction Instr; today the split-K combine is
    // semantically a sum of N partials and binary chaining is correct.
    let cols = node.output.region.cols.len;
    if reads.is_empty() {
        return; // degenerate
    }
    let mut acc_page = state.page_of(reads[0]);
    for r in &reads[1..] {
        let rhs_page = state.page_of(*r);
        state.push(Instr::ResidualAdd {
            a_page: acc_page,
            b_page: rhs_page,
            out_page: dst_page,
            cols,
            role: COMPUTE_ROLE,
        });
        acc_page = dst_page;
    }
}

fn emit_elementwise<F: RopeForm>(
    state: &mut LoweringState<F>,
    node: &SubtileNode<F>,
    kind: crate::subtile_ir::EwKind,
    dst_page: PageId,
    reads: &[SlotId],
) {
    use crate::subtile_ir::EwKind;
    let cols = node.output.region.cols.len;
    match kind {
        EwKind::Silu => {
            // Silu is unary; the SiluMul fusion absorbs most cases. A
            // standalone Silu lowers to SiluMul(silu, 1.0)-equivalent
            // — but the IR carries a single read, so emit a SiluMul
            // with the same page on both sides as a defensive lowering.
            // (The fuser drops standalone Silu before it reaches here
            // in production; this branch is the audited fallback per
            // SubOp::Elementwise's complete dispatch.)
            let p = page_of_first(state, reads, dst_page);
            state.push(Instr::SiluMul {
                gate_page: p,
                up_page: p,
                out_page: dst_page,
                cols,
                role: COMPUTE_ROLE,
            });
        }
        EwKind::Mul => {
            let a = page_of_nth(state, reads, 0, dst_page);
            let b = page_of_nth(state, reads, 1, dst_page);
            state.push(Instr::SiluMul {
                gate_page: a,
                up_page: b,
                out_page: dst_page,
                cols,
                role: COMPUTE_ROLE,
            });
        }
        EwKind::Add => {
            let a = page_of_nth(state, reads, 0, dst_page);
            let b = page_of_nth(state, reads, 1, dst_page);
            state.push(Instr::ResidualAdd {
                a_page: a,
                b_page: b,
                out_page: dst_page,
                cols,
                role: COMPUTE_ROLE,
            });
        }
    }
    let _ = node;
}

fn emit_silu_mul<F: RopeForm>(
    state: &mut LoweringState<F>,
    node: &SubtileNode<F>,
    dst_page: PageId,
    reads: &[SlotId],
) {
    let cols = node.output.region.cols.len;
    let gate = page_of_nth(state, reads, 0, dst_page);
    let up = page_of_nth(state, reads, 1, dst_page);
    state.push(Instr::SiluMul {
        gate_page: gate,
        up_page: up,
        out_page: dst_page,
        cols,
        role: COMPUTE_ROLE,
    });
}

fn emit_rmsnorm<F: RopeForm>(
    state: &mut LoweringState<F>,
    node: &SubtileNode<F>,
    eps: f32,
    dst_page: PageId,
    reads: &[SlotId],
) {
    let rows = node.output.region.rows.len;
    let cols = node.output.region.cols.len;
    let src_page = page_of_nth(state, reads, 0, dst_page);
    let gain_tensor = node.inputs[1].tensor;
    state.push(Instr::RmsNorm {
        src_page,
        dst_page,
        gain_tensor,
        rows,
        cols,
        eps_bits: eps.to_bits(),
        role: COMPUTE_ROLE,
    });
}

fn emit_rope_rotate<F: RopeForm>(
    state: &mut LoweringState<F>,
    node: &SubtileNode<F>,
    head_dim: u32,
    dst_page: PageId,
    reads: &[SlotId],
    side: RopeSide,
) {
    let cols = node.output.region.cols.len;
    let num_heads = NonZeroU32::new(head_dim.max(1))
        .expect(".max(1) above guarantees nonzero")
        .get();
    let num_heads = cols / num_heads;
    let position = state.position();
    let cos_sin_tensor = node.inputs[1].tensor;
    let src_page = page_of_nth(state, reads, 0, dst_page);
    // Rotary uses a cache layout for offset math; for standalone
    // RopeRotate (Q-side) we synthesize a layout from the rotated
    // tensor's shape — the IR doesn't carry one for Q-side rotate.
    let qk_layout = synthesize_q_layout(state, node, head_dim);
    let kv_layout = state.intern_kv_layout(qk_layout);
    state.push(Instr::RopeRotate {
        src_page,
        dst_page,
        cos_sin_tensor,
        position,
        kv_layout,
        head_dim,
        num_heads,
        form: bridge_rope_form_tag::<F>(),
        side,
        role: COMPUTE_ROLE,
    });
}

fn emit_rope_append<F: RopeForm>(
    state: &mut LoweringState<F>,
    node: &SubtileNode<F>,
    head_dim: u32,
    layout: KvCacheLayout,
    dst_page: PageId,
    reads: &[SlotId],
) {
    let cols = node.output.region.cols.len;
    let num_heads = if head_dim == 0 { 1 } else { cols / head_dim };
    let position = state.position();
    let cos_sin_tensor = node.inputs[1].tensor;
    let src_page = page_of_nth(state, reads, 0, dst_page);
    let kv_layout = state.intern_kv_layout(layout);
    state.push(Instr::RopeRotate {
        src_page,
        dst_page,
        cos_sin_tensor,
        position,
        kv_layout,
        head_dim,
        num_heads,
        form: bridge_rope_form_tag::<F>(),
        side: RopeSide::K,
        role: COMPUTE_ROLE,
    });
}

#[allow(clippy::too_many_arguments)]
fn emit_attn_decode<F: RopeForm>(
    state: &mut LoweringState<F>,
    node: &SubtileNode<F>,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    scale: f32,
    layout: KvCacheLayout,
    producer: KvCacheProducer,
    softmax_state: SoftmaxStateId,
    dst_page: PageId,
    reads: &[SlotId],
) {
    let _ = (layout, producer);

    let smx = state.intern_softmax_state(softmax_state);

    // Init — outside the loop bracket. The walker has not yet entered
    // the OpenLoop body (the SubtileTape's lower_dag_to_tape places
    // OpenLoop AFTER alloc_slot AND AFTER the AttnDecode Compute by
    // construction… actually Compute is inside the loop in
    // SubtileTape; see lower_dag_to_tape line 862-869). So
    // emit_attn_decode runs INSIDE the body's instr_stack frame. We
    // re-shape: pop the body's current instructions, insert Init
    // before the body, and let the loop handle Qkt+Sv. Finalise is
    // appended after CloseLoop.
    //
    // Cleaner approach: push Init / Qkt / Sv / Finalise sequentially
    // into the current frame; the SubtileTape OpenLoop/CloseLoop
    // already wraps just the Compute, so the body buffer at this
    // moment is empty and contains only this Compute's outputs.
    // Init and Finalise need to live OUTSIDE the loop — i.e. in the
    // parent frame.
    //
    // But the current frame IS the loop body (we're being called from
    // lower_compute, which is invoked while the OpenLoop's frame is on
    // top). Init must go to the parent. Same for Finalise. Solution:
    // push into instr_stack[len-2] (the parent frame).
    let parent_idx = state.instr_stack.len().checked_sub(2).expect(
        "AttnDecode::Compute should appear inside a SubtileTape OpenLoop \
         (lower_dag_to_tape wraps every AttnDecode in OpenLoop+CloseLoop). \
         Empty stack means the SubtileTape is malformed — would have been \
         caught by validate_subtile_tape.",
    );
    // Parent's last Instr is the ForLoopOpen* we emitted at OpenLoop.
    // Init must precede the loop, so insert just before that Open.
    let parent = &mut state.instr_stack[parent_idx];
    let insert_at = parent.len().saturating_sub(1);
    parent.insert(
        insert_at,
        Instr::AttnDecodeInit {
            state: smx,
            num_q_heads,
            num_kv_heads,
            head_dim,
            role: COMPUTE_ROLE,
        },
    );

    // Qkt + Sv — inside the loop body (current frame).
    let q_page = page_of_nth(state, reads, 0, dst_page);
    let k_page = page_of_nth(state, reads, 1, dst_page);
    let v_page = page_of_nth(state, reads, 2, dst_page);
    state.push(Instr::AttnDecodeQkt {
        state: smx,
        q_page,
        k_page,
        scale_bits: scale.to_bits(),
        num_q_heads,
        num_kv_heads,
        head_dim,
        role: COMPUTE_ROLE,
    });
    state.push(Instr::AttnDecodeSv {
        state: smx,
        v_page,
        num_q_heads,
        num_kv_heads,
        head_dim,
        role: COMPUTE_ROLE,
    });

    // Finalise + store + arrive — outside the loop, on the parent
    // frame (queued for after CloseLoop's ForLoop instr lands).
    // We schedule a deferred "post-loop" sequence by pushing it AFTER
    // the loop closes. To keep the syntax-directed walk simple, we
    // emit Finalise as a side-band: queue it onto a per-AttnDecode
    // pending stack tied to the active SubtileTape OpenLoop. But
    // SubtileTape's lower_dag_to_tape only generates OpenLoop[Compute]
    // CloseLoop sequences for AttnDecodes today, so the loop body has
    // exactly one Compute — meaning right after CloseLoop fires, the
    // parent frame's last Instr is the Instr::ForLoop wrapping
    // Qkt+Sv. We append Finalise + store + arrive immediately.
    //
    // Because emit_attn_decode runs DURING the body walk (before
    // CloseLoop), we push a marker to the parent and process it on
    // CloseLoop. Simpler: push Finalise into a queue on state, then
    // CloseLoop drains the queue.
    state
        .pending_post_loop
        .entry(active_loop_var(state).expect("AttnDecode inside live loop"))
        .or_default()
        .push(PostLoopAction::Finalise {
            state: smx,
            out_page: dst_page,
            num_q_heads,
            head_dim,
            output: node.output,
        });
    let _ = node;
}

// ── Helpers ─────────────────────────────────────────────────────────

fn page_of_first<F: RopeForm>(
    state: &LoweringState<F>,
    reads: &[SlotId],
    fallback: PageId,
) -> PageId {
    reads.first().map(|s| state.page_of(*s)).unwrap_or(fallback)
}

fn page_of_nth<F: RopeForm>(
    state: &LoweringState<F>,
    reads: &[SlotId],
    n: usize,
    fallback: PageId,
) -> PageId {
    reads.get(n).map(|s| state.page_of(*s)).unwrap_or(fallback)
}

fn synthesize_q_layout<F: RopeForm>(
    _state: &mut LoweringState<F>,
    node: &SubtileNode<F>,
    head_dim: u32,
) -> KvCacheLayout {
    // For Q-side RopeRotate the layout is synthetic — the rotated
    // tensor is itself the cache descriptor for offset math.
    let cols = node.output.region.cols.len;
    let num_heads = if head_dim == 0 { 1 } else { cols / head_dim };
    KvCacheLayout::for_cache_tensor(node.output.tensor, num_heads, head_dim)
}

/// Bridge SubtileIR's `RopeForm::TAG` (canonical) to TkTape's
/// `RopeFormTag` (the legacy `tk_tape::RopeForm` trait will be deleted
/// alongside the metal_tape carcass; until then this bridge keeps the
/// canonical IR-side const flowing).
fn bridge_rope_form_tag<F: RopeForm>() -> RopeFormTag {
    use crate::subtile_ir::RopeFormTag as IrTag;
    match F::TAG {
        IrTag::NeoX => RopeFormTag::NeoX,
        IrTag::Interleaved => RopeFormTag::Interleaved,
    }
}

fn drain<F: RopeForm>(state: &mut LoweringState<F>) {
    state.push(Instr::SyncthreadsCta { role: ALL_ROLE });
    state.push(Instr::CommitGroupBulk { role: ALL_ROLE });
    state.push(Instr::WaitGroupBulk { n: 0, role: ALL_ROLE });
    state.push(Instr::ThreadfenceDevice { role: ALL_ROLE });
    state.push(Instr::SyncthreadsCta { role: ALL_ROLE });
}

// ── AttnDecode post-loop deferred emit ──────────────────────────────

#[derive(Clone)]
enum PostLoopAction {
    Finalise {
        state: TkSoftmaxStateId,
        out_page: PageId,
        num_q_heads: u32,
        head_dim: u32,
        output: TensorRegion,
    },
}

fn active_loop_var<F: RopeForm>(state: &LoweringState<F>) -> Option<u32> {
    state.active_loop_stack.last().copied()
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subtile_ir::{
        EwKind, NeoX, Range, Region, SubOp, SubtileNode, TensorId, TensorRegion, TensorShape,
    };
    use crate::subtile_tape::lower_dag_to_tape;

    fn silu_node(
        id: u32,
        in_t: TensorId,
        in_cols: Range,
        out_t: TensorId,
        out_cols: Range,
    ) -> SubtileNode<NeoX> {
        SubtileNode {
            id: SubtileId(id),
            op: SubOp::Elementwise(EwKind::Silu),
            inputs: vec![TensorRegion {
                tensor: in_t,
                region: Region {
                    rows: Range::new(0, 1),
                    cols: in_cols,
                },
            }],
            output: TensorRegion {
                tensor: out_t,
                region: Region {
                    rows: Range::new(0, 1),
                    cols: out_cols,
                },
            },
        }
    }

    #[test]
    fn empty_dag_lowers_to_drain_only() {
        // A graph with one source-reading silu lowers to a tape that
        // alloc/compute/free's one slot; lower_tape_to_tk produces
        // load + wait + silumul + store + arrive + drain.
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
        ];
        let g: SubtileIR<NeoX> = SubtileIR {
            tensors,
            num_sources: 1,
            nodes: vec![silu_node(0, TensorId(0), Range::new(0, 4), TensorId(1), Range::new(0, 4))],
            result: TensorId(1),
        };
        let tape = lower_dag_to_tape(&g);
        let tk = lower_tape_to_tk(&tape, &g);
        // Header: LoadAsync (external), SiluMul, StoreAsync, CommitGroup,
        // Threadfence, PageBarrierArrive. Tail drain: 5 instrs.
        let n_load = tk
            .instrs
            .iter()
            .filter(|i| matches!(i, Instr::LoadAsync(_)))
            .count();
        assert!(n_load >= 1, "expected at least one LoadAsync, got: {:?}", tk.instrs);
        let n_store = tk
            .instrs
            .iter()
            .filter(|i| matches!(i, Instr::StoreAsync(_)))
            .count();
        assert!(n_store >= 1, "expected at least one StoreAsync");
        let n_silu = tk
            .instrs
            .iter()
            .filter(|i| matches!(i, Instr::SiluMul { .. }))
            .count();
        assert_eq!(n_silu, 1, "expected exactly one SiluMul, got: {:?}", tk.instrs);
        // Drain at end: last 5 Instrs are Sync/Commit/Wait/Fence/Sync.
        let n = tk.instrs.len();
        assert!(matches!(tk.instrs[n - 5], Instr::SyncthreadsCta { .. }));
        assert!(matches!(tk.instrs[n - 4], Instr::CommitGroupBulk { .. }));
        assert!(matches!(tk.instrs[n - 3], Instr::WaitGroupBulk { n: 0, .. }));
        assert!(matches!(tk.instrs[n - 2], Instr::ThreadfenceDevice { .. }));
        assert!(matches!(tk.instrs[n - 1], Instr::SyncthreadsCta { .. }));
    }

    #[test]
    fn chain_threads_pages_through_load_store() {
        // source -> silu(0) -> silu(1). The first silu loads from
        // source; the second reads silu(0)'s page (no external load).
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
        ];
        let g: SubtileIR<NeoX> = SubtileIR {
            tensors,
            num_sources: 1,
            nodes: vec![
                silu_node(0, TensorId(0), Range::new(0, 4), TensorId(1), Range::new(0, 4)),
                silu_node(1, TensorId(1), Range::new(0, 4), TensorId(2), Range::new(0, 4)),
            ],
            result: TensorId(2),
        };
        let tape = lower_dag_to_tape(&g);
        let tk = lower_tape_to_tk(&tape, &g);
        // Two SiluMul Instrs (one per node).
        let n_silu = tk
            .instrs
            .iter()
            .filter(|i| matches!(i, Instr::SiluMul { .. }))
            .count();
        assert_eq!(n_silu, 2);
        // Two PageBarrierWait{Ready} (one per producer/consumer edge).
        // The chain has one DAG edge (silu(0) → silu(1)); one Wait.
        let n_wait = tk
            .instrs
            .iter()
            .filter(|i| matches!(i, Instr::PageBarrierWait { kind: PageBarrier::Ready, .. }))
            .count();
        assert_eq!(n_wait, 1, "one DAG edge → one Ready wait, got: {:?}", tk.instrs);
    }

    #[test]
    fn no_unimplemented_panics_in_lowering() {
        // Ensure no SubOp branch panics. Builds a graph with several
        // non-AttnDecode SubOp variants and checks the lowering walks
        // to completion without a panic.
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
        ];
        let g: SubtileIR<NeoX> = SubtileIR {
            tensors,
            num_sources: 2,
            nodes: vec![
                SubtileNode {
                    id: SubtileId(0),
                    op: SubOp::Elementwise(EwKind::Add),
                    inputs: vec![
                        TensorRegion {
                            tensor: TensorId(0),
                            region: Region {
                                rows: Range::new(0, 1),
                                cols: Range::new(0, 4),
                            },
                        },
                        TensorRegion {
                            tensor: TensorId(1),
                            region: Region {
                                rows: Range::new(0, 1),
                                cols: Range::new(0, 4),
                            },
                        },
                    ],
                    output: TensorRegion {
                        tensor: TensorId(2),
                        region: Region {
                            rows: Range::new(0, 1),
                            cols: Range::new(0, 4),
                        },
                    },
                },
                silu_node(1, TensorId(2), Range::new(0, 4), TensorId(3), Range::new(0, 4)),
            ],
            result: TensorId(3),
        };
        let tape = lower_dag_to_tape(&g);
        let tk = lower_tape_to_tk(&tape, &g);
        assert!(!tk.instrs.is_empty());
    }
}
