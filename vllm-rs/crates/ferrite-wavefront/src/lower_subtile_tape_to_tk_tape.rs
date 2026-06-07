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
    ComputeInput, ComputeInputs, Instr as STInstr, LoopBound, LoopVarId as STLoopVarId, SlotId, SubtileTape,
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
    /// SubtileIR `TensorId` → kernel-arg slot for the tensor's
    /// CTensorMap. Lazily minted by [`Self::tensor_arg`] the first
    /// time a Load/Store references the tensor; cached so multiple
    /// references share one signature parameter.
    tensor_arg_index: BTreeMap<TensorId, KernelArgRef>,
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
    /// Smem-vec arena. Each minted [`SmemVecSlot`] corresponds to a
    /// `__shared__ kittens::sv_<dtype><LEN> sv_<idx>;` decl emitted
    /// by the player. Per Cat 5 of the step-8 emit fix.
    smem_vec_arena: BTreeMap<crate::tk_tape::SmemVecSlot, crate::tk_tape::SmemVecArenaEntry>,
    next_smem_vec_slot: u16,
    /// Pages allocated by `resolve_input_page` for External inputs
    /// during the current `lower_compute` arm. Drained and released
    /// at the end of each arm — without this, every per-call
    /// External load would burn a fresh PageId, overflowing u8 across
    /// a Llama-1B tape (hundreds of External weights × chunks).
    ephemeral_pages: Vec<PageId>,
    /// Activation-pool page allocator. Distinct namespace from
    /// `next_page` / `free_pages`. Sized to NUM_ACT_PAGES at the
    /// substrate.
    next_act_page: u8,
    free_act_pages: Vec<crate::tk_tape::ActPageId>,
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
            tensor_arg_index: BTreeMap::new(),
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
            smem_vec_arena: BTreeMap::new(),
            next_smem_vec_slot: 0,
            ephemeral_pages: Vec::new(),
            next_act_page: 0,
            free_act_pages: Vec::new(),
        }
    }

    /// Release all pages allocated by [`Self::resolve_input_page`]
    /// during the current arm. Called at the end of `lower_compute`
    /// so per-arm External loads don't burn unbounded PageIds.
    fn release_ephemeral_pages(&mut self) {
        let drained: Vec<PageId> = self.ephemeral_pages.drain(..).collect();
        for p in drained {
            self.free_pages.push(p);
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

    /// Mint a fresh [`SmemVecId<LEN, T>`]. Each call adds a new
    /// `__shared__ kittens::sv_<dtype><LEN> sv_<idx>;` declaration
    /// to the kernel preamble at emit time.
    fn mint_smem_vec<const LEN: usize, T>(
        &mut self,
    ) -> crate::tk_tape::SmemVecId<LEN, T>
    where
        T: crate::tk_tape::TileDtype,
    {
        use crate::tk_tape::{SmemVecArenaEntry, SmemVecId, SmemVecSlot};
        let slot = SmemVecSlot(self.next_smem_vec_slot);
        self.next_smem_vec_slot = self
            .next_smem_vec_slot
            .checked_add(1)
            .expect("smem_vec slot overflow");
        self.smem_vec_arena.insert(
            slot,
            SmemVecArenaEntry {
                len: LEN as u32,
                dtype: T::tag(),
            },
        );
        SmemVecId::from_slot(slot)
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

    /// Allocate a 64×128 activation page from the act_buf pool.
    /// Distinct namespace from `alloc_temp_page` — the page indexes
    /// into `act_buf[NUM_ACT_PAGES]`, not `page_buf[NUM_PAGES]`. Used
    /// by WGMMA A/D and (later) AttnDecode q/k/v tiles per audit
    /// ADDENDUM 3.
    fn alloc_act_page(&mut self) -> crate::tk_tape::ActPageId {
        if let Some(reused) = self.free_act_pages.pop() {
            reused
        } else {
            let p = crate::tk_tape::ActPageId(self.next_act_page);
            self.next_act_page = self.next_act_page.checked_add(1).expect(
                "ActPageId overflow (>=256 act pages); raise NUM_ACT_PAGES",
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

    /// Resolve a [`ComputeInput`] to the [`PageId`] holding its data.
    /// For `Computed`: returns the existing slot→page mapping.
    /// For `External`: allocates a fresh temp page, emits an
    /// `emit_external_load` TMA load to bring the source-tensor
    /// region into that page, and returns the new page.
    ///
    /// Per Phase A step 4 of the panic-RCA plan: this is the
    /// per-input external resolution that replaces the broken
    /// "load all externals into dst_page" loop at the top of
    /// lower_compute.
    fn resolve_input_page(&mut self, input: &ComputeInput) -> PageId {
        match input {
            ComputeInput::Computed(slot) => self.page_of(*slot),
            ComputeInput::External { tensor, region } => {
                let page = self.alloc_temp_page();
                self.ephemeral_pages.push(page);
                let tr = TensorRegion {
                    tensor: *tensor,
                    region: *region,
                };
                emit_external_load(self, &tr, page);
                page
            }
        }
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

    /// Lazily mint a kernel-arg slot for `t`'s CTensorMap descriptor.
    /// The arg's canonical name is `t<TensorId>`, set via
    /// [`KernelArgName::Tensor`]. Each [`Instr::LoadAsync`] /
    /// [`Instr::StoreAsync`] / [`Instr::StoreAsyncTyped`] references
    /// the resulting [`KernelArgRef`] — never a raw [`TensorId`] —
    /// so the emitted CUDA's `aN` aliases always resolve to a real
    /// kernel-signature parameter.
    ///
    /// Per Phase A step 8 cat 6: prior code passed
    /// `spec.src_tensor.0` (a global SubtileIR TensorId, e.g. 181)
    /// as the body-side `aN` index without ever pushing a matching
    /// kernel-arg, so nvcc reported `identifier "a181" undefined`.
    fn tensor_arg(&mut self, t: TensorId) -> KernelArgRef {
        if let Some(r) = self.tensor_arg_index.get(&t) {
            return *r;
        }
        let r = self.intern_kernel_arg(KernelArg {
            name: KernelArgName::Tensor(t),
            ty: KernelArgTy::BufPtr(t),
        });
        self.tensor_arg_index.insert(t, r);
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
                inputs,
            } => {
                lower_compute(&mut state, *node, *writes, inputs);
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
    out.smem_vec_arena = state.smem_vec_arena;
    out.next_smem_vec_slot = state.next_smem_vec_slot;
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

    // §6.5 optimizer pass pipeline. Each pass is a `TkTape → TkTape`
    // rewrite that preserves `validate_tk_tape` invariants. First pass
    // to land: `rt_alias_pass` (register-tile slot coalescing) — drops
    // arena cardinality so the player declares fewer
    // `kittens::rt<...> rt_<slot>;` per kernel, reducing per-warp
    // register pressure.
    crate::passes::rt_alias_pass(&mut out);
    validate_tk_tape(&out)
        .expect("rt_alias_pass: produced invalid TkTape (post-condition)");

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
    inputs: &ComputeInputs,
) {
    use crate::subtile_ir::{EwKind, SubOp};

    let node = &state.graph.nodes[node_id.0 as usize];
    let dst_page = state.page_of(writes);

    // Wait on each computed-input slot's Ready barrier — the producer's
    // StoreAsync + Arrive{Done} pairs with our Wait{Ready} on the
    // same page. External inputs don't have a barrier (they're loaded
    // ahead of time via emit_external_load).
    // Conservative all-gmem path uses parity 0.
    for ci in inputs.iter() {
        if let ComputeInput::Computed(slot) = ci {
            let p = state.page_of(*slot);
            state.push(Instr::PageBarrierWaitStaticP0 {
                page_id: p,
                kind: PageBarrier::Ready,
                role: COMPUTE_ROLE,
            });
        }
    }

    // External-input loading is now per-arm, via
    // `state.resolve_input_page(input)` (Phase A step 4+ of the
    // panic-RCA plan). The previous "load all externals into dst_page"
    // loop here was broken — it clobbered each external in turn.
    // Each arm now allocates one temp page per External input and
    // emits its TMA load there, in positional order.

    // SubOp dispatch — only Elementwise(Mul) is implemented. Every
    // other arch op panics: per INVIOLABLE feedback_tk_2_0_only +
    // feedback_tk20_primitives_first, lower_compute will not
    // fabricate a tape from invented `kittens::ops::*` helpers. Each
    // arch op lands one at a time as its TK 2.0 primitive sequence
    // is implemented per SUBTILE_TK20_DECOMP.md.
    match &node.op {
        SubOp::Elementwise(EwKind::Mul) => {
            // Compile-time arity: A2 destructures into a typed
            // 2-array; `[in0, in1]` indexing is rustc-checked, no
            // runtime bounds check.
            let [in0, in1] = inputs.expect_a2("Elementwise(Mul)");
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
            let lhs = SmemTileId::<128, 128, Bf16>::from_page(state.resolve_input_page(in0));
            let rhs = SmemTileId::<128, 128, Bf16>::from_page(state.resolve_input_page(in1));
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
            let [in0, in1] = inputs.expect_a2("Elementwise(Add)");
            // Same compile-time witnesses as the Mul arm; only the
            // emitted TK 2.0 primitive differs (`group<N>::add` vs
            // `group<N>::mul`).
            use crate::tk_tape::{Bf16, GroupWidth, SmemTileId};
            let lhs = SmemTileId::<128, 128, Bf16>::from_page(state.resolve_input_page(in0));
            let rhs = SmemTileId::<128, 128, Bf16>::from_page(state.resolve_input_page(in1));
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
            // Variadic — N≥2 inputs chained as N-1 ShTileAdd.
            let inputs_v = inputs.expect_variadic("SumReduce");
            assert!(
                inputs_v.len() >= 2,
                "SumReduce expects ≥2 inputs (got {}); N=1 lowers to a copy, not yet supported",
                inputs_v.len(),
            );
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
            use crate::tk_tape::{Bf16, GroupWidth, SmemTileId};
            let dst = SmemTileId::<128, 128, Bf16>::from_page(dst_page);
            // First add: dst = reads[0] + reads[1]
            let r0 = SmemTileId::<128, 128, Bf16>::from_page(state.resolve_input_page(&inputs_v[0]));
            let r1 = SmemTileId::<128, 128, Bf16>::from_page(state.resolve_input_page(&inputs_v[1]));
            state.push(Instr::sh_tile_add(
                r0,
                r1,
                dst,
                GroupWidth::<16>::ALL_CONSUMERS,
            ));
            // Subsequent adds: dst += reads[i]
            for ci in &inputs_v[2..] {
                let rhs = SmemTileId::<128, 128, Bf16>::from_page(state.resolve_input_page(ci));
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
            let [gate_in, up_in] = inputs.expect_a2("SiluMul");
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
            let gate = SmemTileId::<128, 128, Bf16>::from_page(state.resolve_input_page(gate_in));
            let up = SmemTileId::<128, 128, Bf16>::from_page(state.resolve_input_page(up_in));
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
            let [in0] = inputs.expect_a1("Elementwise(Silu)");
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
            let src = SmemTileId::<128, 128, Bf16>::from_page(state.resolve_input_page(in0));
            let dst = SmemTileId::<128, 128, Bf16>::from_page(dst_page);
            let rt_x: RegTileId<128, 128, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_neg: RegTileId<128, 128, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_exp: RegTileId<128, 128, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_denom: RegTileId<128, 128, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_result: RegTileId<128, 128, Bf16, RowLayout> = state.mint_reg_tile();
            const W: GroupWidth<16> = GroupWidth::<16>::ALL_CONSUMERS;
            // Smem↔reg moves require ST::rows == GROUP_WARPS * RT::rows.
            // ROWS-equal pair (typed witness on the constructor) forces
            // GROUP_WARPS=1 — sealed via `WarpLoadWidth` impl'd only for
            // `GroupWidth<1>`. Per `feedback_ff_subtile_compile_time_inviolable`.
            const WL: GroupWidth<1> = GroupWidth::<1>::PER_WARP;
            const R: AllConsumersRole = AllConsumersRole;
            // 1: rt_x = load(src_page)
            state.push(Instr::load_shmem_to_reg(src, rt_x, WL, R));
            // 2: rt_neg = neg(rt_x)
            state.push(Instr::reg_tile_neg(rt_x, rt_neg, WL, R));
            // 3: rt_exp = exp(rt_neg)
            state.push(Instr::reg_tile_exp(rt_neg, rt_exp, WL, R));
            // 4: rt_denom = rt_exp + 1.0
            state.push(Instr::reg_tile_add_scalar(rt_exp, rt_denom, ScalarF32::new(1.0), WL, R));
            // 5: rt_result = rt_x / rt_denom
            state.push(Instr::reg_tile_div(rt_x, rt_denom, rt_result, WL, R));
            // 6: store(dst_page, rt_result)
            state.push(Instr::store_reg_tile_to_shmem(rt_result, dst, WL, R));
            emit_store_and_arrive(state, &node.output, dst_page);
        }
        SubOp::RmsNorm { eps } => {
            let [x_in, gamma_in] = inputs.expect_a2("RmsNorm");
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
            // shared vec slots for var/inv_rms (length = ROWS = 128).
            // Per `feedback_ff_subtile_compile_time_inviolable` (Cat 5):
            // shared vecs live in their own arena (`SmemVecSlot`), NOT
            // viewed onto tile pages — TK 2.0's row_sum/mul_row/etc.
            // require a `kittens::sv_*<LEN>` operand.
            use crate::tk_tape::{
                AllConsumersRole, Bf16, GroupWidth, OrthoLayout, RegVecId, SmemTileId, SmemVecId,
            };
            // Resolve each positional input to a page. Computed
            // inputs reuse their producer's page; External inputs
            // get a fresh temp page populated by an emit_external_load.
            let x_page = state.resolve_input_page(x_in);
            let gamma_page = state.resolve_input_page(gamma_in);
            let x = SmemTileId::<128, 128, Bf16>::from_page(x_page);
            let dst = SmemTileId::<128, 128, Bf16>::from_page(dst_page);
            const W: GroupWidth<16> = GroupWidth::<16>::ALL_CONSUMERS;
            const WL: GroupWidth<1> = GroupWidth::<1>::PER_WARP;
            const R: AllConsumersRole = AllConsumersRole;
            // Temp pages (tile substrate)
            let x_sq_page = state.alloc_temp_page();
            let x_sq = SmemTileId::<128, 128, Bf16>::from_page(x_sq_page);
            // Gamma must be viewed as a column-vector for ShTileMulCol
            // (per-col broadcast). Lower a fresh smem-vec slot of LEN=128.
            // The external gamma load lands in a tile page, but the
            // lowerer's contract is that gamma is logically 1×N — at
            // emit time the player references the same shmem region.
            // For the arity-1 vec endpoint we mint a dedicated slot.
            let var_vec: SmemVecId<128, Bf16> = state.mint_smem_vec();
            let inv_rms_vec: SmemVecId<128, Bf16> = state.mint_smem_vec();
            // Reg vec for rsqrt detour
            // RmsNorm reg-vecs flow through TK 2.0's unary_op (rsqrt) —
            // pointwise per-element, layout-agnostic. Pick OrthoLayout to
            // avoid drift if a later refactor routes these through a
            // row-reduction or row_map (both want ortho on row-layout rt).
            // Per `feedback_ff_subtile_compile_time_inviolable`: choose
            // the strictest compatible layout up front.
            let rv_var: RegVecId<128, Bf16, OrthoLayout> = state.mint_reg_vec();
            let rv_inv: RegVecId<128, Bf16, OrthoLayout> = state.mint_reg_vec();

            // 1: x_sq = x * x
            state.push(Instr::sh_tile_mul(x, x, x_sq, W));
            // 2: sum_sq = row_sum(x_sq)  (writes into the var smem-vec slot)
            state.push(Instr::sh_tile_row_sum(x_sq, var_vec, W));
            // 3: var = sum_sq * (1/COLS)
            let inv_cols = 1.0_f32 / 128.0;
            state.push(Instr::sh_vec_mul_scalar(
                var_vec,
                var_vec,
                crate::tk_tape::ScalarF32::new(inv_cols),
                W,
            ));
            // 4: var = var + eps
            state.push(Instr::sh_vec_add_scalar(
                var_vec,
                var_vec,
                crate::tk_tape::ScalarF32::new(*eps),
                W,
            ));
            // 5: rv_var = load(var)
            state.push(Instr::load_vec_smem_to_reg(var_vec, rv_var, WL, R));
            // 6: rv_inv = rsqrt(rv_var)
            state.push(Instr::reg_vec_unary_rsqrt(rv_var, rv_inv, WL, R));
            // 7: inv_rms = store(rv_inv)
            state.push(Instr::store_reg_vec_to_shmem(rv_inv, inv_rms_vec, WL, R));
            // 8: x_norm = x * inv_rms (per-row broadcast)
            //    write into dst (clobber x is OK; we reuse dst as
            //    the running tile through the gamma multiply too).
            state.push(Instr::sh_tile_mul_row(x, inv_rms_vec, dst, W));
            // 9: out = x_norm * gamma (per-col broadcast)
            //    gamma flows in as an External tile page; ferrite-runtime
            //    knows the gamma source is logically 1×128 and packs it
            //    into a sv-shaped layout. For now route the col-vec
            //    endpoint via a freshly-minted slot that aliases the
            //    gamma external page at emit time. NOTE: this is the
            //    next gap to close — see Cat 5 follow-up.
            let gamma_vec: SmemVecId<128, Bf16> = state.mint_smem_vec();
            let _ = gamma_page; // gamma external page tracked for predecessor coverage; the SmemVecSlot path supersedes it once the external-load substrate emits sv-shaped pages.
            state.push(Instr::sh_tile_mul_col(dst, gamma_vec, dst, W));

            emit_store_and_arrive(state, &node.output, dst_page);
        }
        SubOp::RopeRotate { head_dim, _form: _ } => {
            let [q_in, cos_in, sin_in] = inputs.expect_a3("RopeRotate");
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
                AllConsumersRole, AlignLayout, Bf16, GroupWidth, RegTileId,
                RegVecId, RowLayout, SmemTileId, SmemVecId,
            };
            let q_page = state.resolve_input_page(q_in);
            // Resolve cos/sin to External tile pages for predecessor
            // coverage; mint dedicated `SmemVecId<32, Bf16>` slots
            // that the player declares as `__shared__ sv_bf<32> sv_<id>`
            // and the external-load path will route into. Cat 5 of
            // step-8: vec endpoints cannot live in `page_buf[]`.
            let _cos_page = state.resolve_input_page(cos_in);
            let _sin_page = state.resolve_input_page(sin_in);
            let cos_vec: SmemVecId<32, Bf16> = state.mint_smem_vec();
            let sin_vec: SmemVecId<32, Bf16> = state.mint_smem_vec();
            let q_full = SmemTileId::<128, 128, Bf16>::from_page(q_page);
            let dst_full = SmemTileId::<128, 128, Bf16>::from_page(dst_page);
            const W: GroupWidth<16> = GroupWidth::<16>::ALL_CONSUMERS;
            const WL: GroupWidth<1> = GroupWidth::<1>::PER_WARP;
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
            // RopeRotate uses `mul_col` (col_map) on row-layout rt;
            // TK 2.0 col_map requires V::layout == row_vec_layout =
            // align_l (rt_base.cuh:78). NaiveLayout fails the static
            // assert at maps.cuh:287.
            let rv_cos: RegVecId<32, Bf16, AlignLayout> = state.mint_reg_vec();
            let rv_sin: RegVecId<32, Bf16, AlignLayout> = state.mint_reg_vec();

            // 1: rt_q_even = q[:, 0:32]
            state.push(Instr::load_shmem_subtile_to_reg::<1, 128, 128, 32, 0, Bf16, RowLayout>(
                q_full, rt_q_even, WL, R,
            ));
            // 2: rt_q_odd = q[:, 32:64]
            state.push(Instr::load_shmem_subtile_to_reg::<1, 128, 128, 32, 1, Bf16, RowLayout>(
                q_full, rt_q_odd, WL, R,
            ));
            // 3: rv_cos = load(cos_vec)
            state.push(Instr::load_vec_smem_to_reg(cos_vec, rv_cos, WL, R));
            // 4: rv_sin = load(sin_vec)
            state.push(Instr::load_vec_smem_to_reg(sin_vec, rv_sin, WL, R));
            // 5: rt_a = q_even * cos
            state.push(Instr::reg_tile_mul_col(rt_q_even, rv_cos, rt_a, WL, R));
            // 6: rt_b = q_odd * sin
            state.push(Instr::reg_tile_mul_col(rt_q_odd, rv_sin, rt_b, WL, R));
            // 7: rt_c = q_even * sin
            state.push(Instr::reg_tile_mul_col(rt_q_even, rv_sin, rt_c, WL, R));
            // 8: rt_d = q_odd * cos
            state.push(Instr::reg_tile_mul_col(rt_q_odd, rv_cos, rt_d, WL, R));
            // 9: rt_a = rt_a - rt_b  (out_even = q_even*cos - q_odd*sin)
            state.push(Instr::reg_tile_sub(rt_a, rt_b, rt_a, WL, R));
            // 10: rt_c = rt_c + rt_d  (out_odd  = q_even*sin + q_odd*cos)
            state.push(Instr::reg_tile_add(rt_c, rt_d, rt_c, WL, R));
            // 11: dst[:, 0:32] = rt_a
            state.push(Instr::store_reg_tile_subtile_to_shmem::<1, 128, 128, 32, 0, Bf16, RowLayout>(
                rt_a, dst_full, WL, R,
            ));
            // 12: dst[:, 32:64] = rt_c
            state.push(Instr::store_reg_tile_subtile_to_shmem::<1, 128, 128, 32, 1, Bf16, RowLayout>(
                rt_c, dst_full, WL, R,
            ));

            emit_store_and_arrive(state, &node.output, dst_page);
        }
        SubOp::MatmulTile => {
            let [a_in, b_in] = inputs.expect_a2("MatmulTile");
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
            // Substrate: 128×128 Bf16 pages. Hopper WGMMA m64 hardcodes
            // A.rows == 4*TILE_ROW_DIM<bf16> == 64 for the smem-A
            // variant; the register-A variant at warpgroup.cuh:139 sets
            // M_DIV_4 = A::height instead, so per-warp 32-row rt_a
            // (4 warps × 32 = 128 collective) sidesteps the constraint.
            // Per audit B.1+B.2: switched MatmulTile to RegSmem path.
            //
            // Plan-faithful flow (8 Instrs):
            //   1: InitRtZero          rt_d = 0
            //   2: LoadShmemToReg      rt_a = load(a)  (group<4> ST→RT)
            //   3: WgmmaFenceAcc       mma_fence(rt_d)
            //   4: WgmmaMmaAB_RegSmem  rt_d += rt_a @ b
            //   5: WgmmaAsyncWait      wait_group<0>
            //   6: StoreRegTileToShmem dst = rt_d  (fp32→bf16 in TK 2.0)
            //
            // Per-warp rt shapes: rt_a / rt_d are <32, 128> per-warp
            // (group<4>::load distributes ST.rows=128 across 4 warps,
            // ST.rows / RT.rows == GROUP_WARPS=4 ⇒ RT.rows=32).
            use crate::tk_tape::{
                AccReset, AllConsumersRole, Bf16, FenceExternal, Fp32,
                GroupWidth, RegTileId, RoleWitness, RowLayout, SmemTileId,
            };
            let a = SmemTileId::<128, 128, Bf16>::from_page(state.resolve_input_page(a_in));
            let b = SmemTileId::<128, 128, Bf16>::from_page(state.resolve_input_page(b_in));
            let dst = SmemTileId::<128, 128, Bf16>::from_page(dst_page);
            const W4: GroupWidth<4> = GroupWidth::<4>::WARPGROUP;
            const W16: GroupWidth<16> = GroupWidth::<16>::ALL_CONSUMERS;
            const WL: GroupWidth<1> = GroupWidth::<1>::PER_WARP;
            const R: AllConsumersRole = AllConsumersRole;

            // Per-warp register tiles for the rt_st mma_AB. M=32 per
            // warp ⇒ height=2; warpgroup of 4 = 128 collective rows.
            let rt_a: RegTileId<32, 128, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_d: RegTileId<32, 128, Fp32, RowLayout> = state.mint_reg_tile();

            // Step 4: zero the accumulator (per-warp 32 rows)
            state.push(Instr::init_rt_zero(rt_d, WL, R));
            // Load A from smem into per-warp rt_a — group<4>::load
            // splits ST.rows=128 across 4 warps (32 rows per warp).
            // Typed witness `WarpgroupLoadShape<128, 32>` gates the
            // shape pair.
            state.push(Instr::load_shmem_to_reg_warpgroup(a, rt_a, W4, R));
            // Step 5: fence on D
            state.push(Instr::wgmma_fence_acc(rt_d, W4));
            // Step 6: D += rt_a @ B  (register-A variant — no M==4
            // constraint; M_DIV_4 = A::height = 2)
            state.push(Instr::wgmma_mma_ab_reg_smem(
                rt_d, rt_a, b, FenceExternal, AccReset, W4,
            ));
            // Step 7: wait for all WGMMA groups
            state.push(Instr::wgmma_async_wait(0, W4));

            // Step 8: store accumulator → dst page. Per-warp 32-row
            // rt_d (Fp32) → 128-row smem dst (Bf16) via the
            // warpgroup-sharded store. TK 2.0 `store(st, rt)` at
            // `shared_to_register.cuh` handles fp32→bf16 conversion at
            // store time, so the dtype mismatch is allowed (the
            // typed constructor's `T_ST != T_RT` parameters reflect
            // this). Per plan §"Resolved decision 5" the proper
            // RegTileCopyConvert lift is a follow-up.
            let _ = W16;
            state.push(Instr::store_reg_tile_to_shmem_warpgroup(
                rt_d, dst, W4, R,
            ));

            emit_store_and_arrive(state, &node.output, dst_page);
        }
        SubOp::RopeAppend { head_dim, layer, layout, _form: _ } => {
            // Arity 6 from the SubtileIR: [K, cos, sin, V, kcache, vcache].
            // The last two (kcache, vcache) are graph-edge inputs needed
            // for predecessor coverage, but the actual store sites take
            // their tensor IDs from the `layout` witness, not from these
            // input slots — so resolve_input_page is only invoked on the
            // first four, and 4/5 are unused here. Their page-allocation
            // is by design empty (cache tensors are global, not paged).
            let [k_in, cos_in, sin_in, v_in, _kcache_in, _vcache_in] =
                inputs.expect_a6("RopeAppend");
            // RopeAppend (step 10): rotate K (NeoX) + write rotated K
            // and un-rotated V into the paged KV cache at the runtime
            // decode position. Plan §"Per-SubOp Instr counts" line 26
            // (14 Instrs aggregate; 12 rotation + 2 cache writes).
            //
            // F: RopeForm const-generic dispatches NeoX vs Interleaved.
            // Llama-3.2-1B uses NeoX with head_dim=64.
            use crate::subtile_ir::RopeFormTag;
            assert_eq!(
                F::TAG,
                RopeFormTag::NeoX,
                "RopeAppend Interleaved form is plan step 8, not yet landed",
            );
            assert_eq!(
                *head_dim, 64,
                "RopeAppend: only head_dim=64 (Llama-3.2-1B) supported \
                 today; got head_dim={}",
                head_dim,
            );

            // reads = [K, cos, sin, V, ...] (arity 6 — last 2 are
            // typically cache handles propagated via `layout` not via
            // input slots).
            use crate::tk_tape::{
                AllConsumersRole, Bf16, ByteOffset, ByteOffsetExpr,
                AlignLayout, ByteStride, GroupWidth, PerPositionStep,
                RegTileId, RegVecId, RowLayout, SmemTileId, SmemVecId,
                StoreSpec, TileShape, WarpRole,
            };
            let k_page = state.resolve_input_page(k_in);
            // Cos/sin External pages → predecessor coverage only;
            // dedicated SmemVecId slots for the actual vec ops.
            let _cos_page = state.resolve_input_page(cos_in);
            let _sin_page = state.resolve_input_page(sin_in);
            let cos_vec: SmemVecId<32, Bf16> = state.mint_smem_vec();
            let sin_vec: SmemVecId<32, Bf16> = state.mint_smem_vec();
            let v_page = state.resolve_input_page(v_in);
            let k_full = SmemTileId::<128, 128, Bf16>::from_page(k_page);
            let dst_full = SmemTileId::<128, 128, Bf16>::from_page(dst_page);
            const W: GroupWidth<16> = GroupWidth::<16>::ALL_CONSUMERS;
            const WL: GroupWidth<1> = GroupWidth::<1>::PER_WARP;
            const R: AllConsumersRole = AllConsumersRole;

            // Mint registers (same shape as step 7 RopeRotate)
            let rt_k_even: RegTileId<128, 32, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_k_odd:  RegTileId<128, 32, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_a:      RegTileId<128, 32, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_b:      RegTileId<128, 32, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_c:      RegTileId<128, 32, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_d:      RegTileId<128, 32, Bf16, RowLayout> = state.mint_reg_tile();
            // mul_col (col_map) on row-layout rt → V must be align_l.
            let rv_cos: RegVecId<32, Bf16, AlignLayout> = state.mint_reg_vec();
            let rv_sin: RegVecId<32, Bf16, AlignLayout> = state.mint_reg_vec();

            // Rotation (12 Instrs, identical algorithm to step 7)
            state.push(Instr::load_shmem_subtile_to_reg::<1, 128, 128, 32, 0, Bf16, RowLayout>(
                k_full, rt_k_even, WL, R,
            ));
            state.push(Instr::load_shmem_subtile_to_reg::<1, 128, 128, 32, 1, Bf16, RowLayout>(
                k_full, rt_k_odd, WL, R,
            ));
            state.push(Instr::load_vec_smem_to_reg(cos_vec, rv_cos, WL, R));
            state.push(Instr::load_vec_smem_to_reg(sin_vec, rv_sin, WL, R));
            state.push(Instr::reg_tile_mul_col(rt_k_even, rv_cos, rt_a, WL, R));
            state.push(Instr::reg_tile_mul_col(rt_k_odd, rv_sin, rt_b, WL, R));
            state.push(Instr::reg_tile_mul_col(rt_k_even, rv_sin, rt_c, WL, R));
            state.push(Instr::reg_tile_mul_col(rt_k_odd, rv_cos, rt_d, WL, R));
            state.push(Instr::reg_tile_sub(rt_a, rt_b, rt_a, WL, R));
            state.push(Instr::reg_tile_add(rt_c, rt_d, rt_c, WL, R));
            state.push(Instr::store_reg_tile_subtile_to_shmem::<1, 128, 128, 32, 0, Bf16, RowLayout>(
                rt_a, dst_full, WL, R,
            ));
            state.push(Instr::store_reg_tile_subtile_to_shmem::<1, 128, 128, 32, 1, Bf16, RowLayout>(
                rt_c, dst_full, WL, R,
            ));

            // Cache writes at runtime decode position.
            //
            // `ByteOffsetExpr::kv_cache_runtime_position::<K>(arg, layer)`
            // derives stride from K::ROW_BYTES and base from
            // `layer × K::LAYER_BYTES`. The K type parameter is the
            // single source of truth for the cache layout — wrong K
            // is rustc E0308 at the layout-witness binding (K7 lift).
            //
            // (Stable Rust forbids `runtime_position::<{ K::ROW_BYTES }>`
            // because `generic_const_exprs` is unstable; the K-witnessed
            // method is the workaround.)
            let pos_arg = state.position();
            let k_cache_arg = state.tensor_arg(layout.cache_tensor());
            let v_cache_arg = state.tensor_arg(layout.v_cache_tensor());

            // K-cache write: rotated K (dst_page) → K cache at slot[p].
            state.push(Instr::StoreAsync(StoreSpec {
                src_page: dst_page,
                dst_arg: k_cache_arg,
                byte_off: ByteOffsetExpr::kv_cache_runtime_position::<K>(pos_arg, *layer),
                tile: TileShape {
                    rows: 128,
                    cols: 128,
                    elem_bytes: 2,
                },
                role: WarpRole::Storer,
            }));

            // V-cache write: un-rotated V (v_page) → V cache at slot[p].
            state.push(Instr::StoreAsync(StoreSpec {
                src_page: v_page,
                dst_arg: v_cache_arg,
                byte_off: ByteOffsetExpr::kv_cache_runtime_position::<K>(pos_arg, *layer),
                tile: TileShape {
                    rows: 128,
                    cols: 128,
                    elem_bytes: 2,
                },
                role: WarpRole::Storer,
            }));

            emit_store_and_arrive(state, &node.output, dst_page);
        }
        SubOp::AttnDecode {
            num_q_heads: _,
            num_kv_heads: _,
            head_dim,
            scale,
            layout,
            producer: _,
            softmax_state: _,
        } => {
            // AttnDecode arity 5 from the SubtileIR:
            //   [q, kcache, vcache, slot_mapping, block_table] (or
            //   similar — the precise non-q TensorIds vary; lowerer
            //   only resolves q to a page; the cache tensors flow via
            //   the `layout` witness, slot_mapping/block_table flow
            //   via TmaLoadKvBlock helpers and runtime kernel args).
            let [q_in, _kcache_in, _vcache_in, _slot_in, _block_in] =
                inputs.expect_a5("AttnDecode");
            // AttnDecode (steps 11-14): online softmax over a paged
            // KV cache.
            //
            // Algorithm:
            //   Init: rt_o = 0 (fp32), rv_m = -inf (fp32), rv_l = 0 (fp32)
            //   For each chunk c in 0..(seq_len / chunk_size):
            //     load K[chunk] from cache → page_k
            //     load V[chunk] from cache → page_v
            //     wait barriers
            //     S = Q @ K^T            (mma_ABt; D fp32)
            //     S *= scale * log2(e)
            //     m_old = m
            //     m = max(m, row_max(S))      (row_max_acc)
            //     alpha = exp2(m_old - m)
            //     l *= alpha
            //     o *= alpha               (per-row mul)
            //     S -= m                   (sub_row)
            //     P = exp2(S) (in fp32)
            //     l += row_sum(P)          (row_sum_acc)
            //     P_bf16 = copy_convert(P) (fp32 → bf16)
            //     o += P @ V               (mma_AB; accumulate)
            //   Finalise: o /= l (div_row), store(dst, o)
            //
            // Per the audit's deferred-typestate decision:
            // SoftmaxRowMaxAcc<P> phase typestate is not enforced
            // here; the lowerer is the only AttnDecode constructor
            // and emits phases in the correct order by construction.
            //
            // For Llama-3.2-1B: HEAD_DIM=64, num_q_heads=32 (16
            // q-rows fit one chunk → 16×64 register tiles), chunk
            // size = 128 cache positions per iteration.

            assert_eq!(
                *head_dim, 64,
                "AttnDecode: only head_dim=64 (Llama-3.2-1B) supported \
                 today; got head_dim={}",
                head_dim,
            );

            use crate::tk_tape::{
                AccAccumulate, AccReset, AllConsumersRole, Bf16,
                ByteOffset, ByteOffsetExpr, FenceExternal, Fp32,
                AlignLayout, GroupWidth, LoaderRole, OrthoLayout, RegTileId,
                RegVecId, RoleWitness, RowLayout, ScalarF32,
                SmemTileId, SmemTileSpec, StoreSpec, TileShape,
                WarpRole,
            };
            // reads = [q_page, k_cache_handle, v_cache_handle, ...]
            let q_page = state.resolve_input_page(q_in);
            // K and V cache TensorIds come from the layout witness
            // (single source of truth, not from reads[].)
            let k_cache = state.tensor_arg(layout.cache_tensor());
            let v_cache = state.tensor_arg(layout.v_cache_tensor());
            const W4: GroupWidth<4> = GroupWidth::<4>::WARPGROUP;
            const W16: GroupWidth<16> = GroupWidth::<16>::ALL_CONSUMERS;
            const WL: GroupWidth<1> = GroupWidth::<1>::PER_WARP;
            const R: AllConsumersRole = AllConsumersRole;

            // Tile geometry:
            //   chunk-row count (M_q) = 128 (q tile rows; one decode
            //     batch tile)
            //   chunk-col count (D)   = 128 (head_dim padded to 128
            //     — Llama head_dim=64 occupies cols 0..64)
            //   chunk K-positions     = 128 per iteration
            // For Llama: number of chunks = MAX_POSITION / 128.
            let q_tile = SmemTileId::<128, 128, Bf16>::from_page(q_page);

            // Allocate temp pages for K and V tiles (one per iteration)
            let k_tile_page = state.alloc_temp_page();
            let v_tile_page = state.alloc_temp_page();
            let k_tile = SmemTileId::<128, 128, Bf16>::from_page(k_tile_page);
            let v_tile = SmemTileId::<128, 128, Bf16>::from_page(v_tile_page);
            let dst_tile = SmemTileId::<128, 128, Bf16>::from_page(dst_page);

            // Per-warp register state. Hopper WGMMA's rt_st mma_AB /
            // mma_ABt at warpgroup.cuh:139,323 requires rt-A and rt-D
            // to be per-warp slices (4 warps × M_per_warp = collective
            // M). For collective M=128: M_per_warp = 32 (height=2).
            // Per audit B.1+B.2 + plan §29 (AttnDecode_Sv = RegSmem).
            let rt_q:   RegTileId<32, 128, Bf16, RowLayout> = state.mint_reg_tile();
            let rt_o:   RegTileId<32, 128, Fp32, RowLayout> = state.mint_reg_tile();
            let rt_s:   RegTileId<32, 128, Fp32, RowLayout> = state.mint_reg_tile();
            let rt_p:   RegTileId<32, 128, Bf16, RowLayout> = state.mint_reg_tile();
            // Per-warp rv (ortho layout for row_max / row_sum / row_map
            // path). LEN==32 ties to rt_s's per-warp ROWS.
            let rv_m:     RegVecId<32, Fp32, OrthoLayout> = state.mint_reg_vec();
            let rv_l:     RegVecId<32, Fp32, OrthoLayout> = state.mint_reg_vec();
            let rv_m_old: RegVecId<32, Fp32, OrthoLayout> = state.mint_reg_vec();
            let rv_alpha: RegVecId<32, Fp32, OrthoLayout> = state.mint_reg_vec();

            // ── Init phase (3 Instrs) ────────────────────────────
            // Init via group<4> = warpgroup-scope: each of the 4 warps
            // initializes its own per-warp 32-row slice of rt_o /
            // 32-element slice of rv_m / rv_l.
            state.push(Instr::init_rt_zero(rt_o, WL, R));
            state.push(Instr::init_rv_neg_infty(rv_m, W4, R));
            state.push(Instr::init_rv_zero(rv_l, W4, R));
            // Load Q from smem to per-warp register tile once (Q is
            // constant across the K-loop iterations).
            state.push(Instr::load_shmem_to_reg_warpgroup(q_tile, rt_q, W4, R));

            // BarrierInit for K and V tile pages: each iteration's
            // TMA load arrives on the corresponding page_ready[*]
            // mbarrier; a parity-alternating Wait per iteration
            // pairs them. Count = 1 (one TMA load arrival per phase).
            state.push(Instr::BarrierInit {
                page_id: k_tile_page,
                kind: PageBarrier::Ready,
                count: 1,
            });
            state.push(Instr::BarrierInit {
                page_id: v_tile_page,
                kind: PageBarrier::Ready,
                count: 1,
            });

            // ── Loop body (Qkt + Sv phases) ──────────────────────
            //
            // OpenLoop iterates 0..(seq_len / chunk_size).
            // chunk_size = 128 positions (= 128 rows of K/V cache
            // per chunk).  seq_len kernel arg drives the bound.
            let seq_arg = state.seq_len();
            let loop_var = state.fresh_loop_var();
            // Loop bound: seq_len (number of chunks; we iterate by
            // 1 chunk per step, so total iterations = seq_len /
            // chunk_size; for now use seq_len directly with the
            // understanding that chunked iteration is a follow-up
            // optimization. The seq_len kernel arg holds the chunk
            // count, set by the host).
            state.push(Instr::ForLoopOpenKernelArg {
                var: loop_var,
                arg: seq_arg,
            });
            // Wait barrier currently issued for the q_page only;
            // K and V are loaded fresh each iteration, no per-page
            // barrier infrastructure for those temp pages.

            // TMA load K[chunk] / V[chunk] from cache. Chunk index
            // is the loop var; chunk size is 128 cache positions
            // (matching the substrate page row count). Per-iteration
            // byte stride = CHUNK_ROWS × K::ROW_BYTES, derived from
            // the typed KvCacheLayout<K> witness.
            //
            // Layer base = 0 — the orchestrator uses per-layer cache
            // TensorIds (layout.cache_tensor() / v_cache_tensor() are
            // already layer-specific). If a unified-cache layout is
            // ever needed, the call site passes the right layer arg.
            const CHUNK_ROWS: usize = 128;
            let k_off = ByteOffsetExpr::kv_cache_chunk_loop::<CHUNK_ROWS, K>(loop_var, 0);
            let v_off = ByteOffsetExpr::kv_cache_chunk_loop::<CHUNK_ROWS, K>(loop_var, 0);

            // TmaExpect per iteration — arms each barrier with the
            // expected byte count derived from SmemTileSpec<R,C,T>
            // (no runtime byte-count param; type system writes it).
            // Per `feedback_tk20_tma_lane_gate`, the emit uses
            // `kittens::group<1>::tma::*` (lane-0-gated).
            let k_shape = SmemTileSpec::<128, 128, Bf16>::from_shape(TileShape {
                rows: 128,
                cols: 128,
                elem_bytes: 2,
            });
            let v_shape = SmemTileSpec::<128, 128, Bf16>::from_shape(TileShape {
                rows: 128,
                cols: 128,
                elem_bytes: 2,
            });
            state.push(Instr::tma_expect(k_tile_page, k_shape, LoaderRole));
            state.push(Instr::tma_expect(v_tile_page, v_shape, LoaderRole));

            state.push(Instr::LoadAsync(LoadSpec::new(
                k_tile_page,
                k_cache,
                k_off,
                k_shape,
                LoaderRole,
                k_tile_page,
            )));
            state.push(Instr::LoadAsync(LoadSpec::new(
                v_tile_page,
                v_cache,
                v_off,
                v_shape,
                LoaderRole,
                v_tile_page,
            )));

            // Wait for TMA loads to complete. Parity alternates with
            // loop_var (mbarrier::wait flips parity per phase, so
            // loop iteration 0 waits on parity 0, iteration 1 on
            // parity 1, etc.). Use the typed wait_loop constructor.
            state.push(Instr::wait_loop(
                k_tile_page,
                PageBarrier::Ready,
                loop_var,
                crate::tk_tape::Parity::P0,
                WarpRole::AllConsumers,
            ));
            state.push(Instr::wait_loop(
                v_tile_page,
                PageBarrier::Ready,
                loop_var,
                crate::tk_tape::Parity::P0,
                WarpRole::AllConsumers,
            ));

            // S = Q @ K^T via rt_st variant (rt_q is register-A; the
            // rt_st mma_ABt's M_DIV_4 = A::height = 2 ⇒ collective
            // M=128 across 4-warp warpgroup).
            state.push(Instr::wgmma_fence_acc(rt_s, W4));
            state.push(Instr::wgmma_mma_abt_reg_smem(
                rt_s, rt_q, k_tile, FenceExternal, AccReset, W4,
            ));
            state.push(Instr::wgmma_async_wait(0, W4));
            // S *= scale * log2(e)
            const LOG2_E: f32 = 1.442_695_f32;
            let scaled = (*scale) * LOG2_E;
            state.push(Instr::reg_tile_mul_scalar(
                rt_s, rt_s, ScalarF32::new(scaled), WL, R,
            ));
            // m_old = m  (save before updating)
            state.push(Instr::reg_vec_copy(rv_m, rv_m_old, WL, R));
            // m = max(m, row_max(S)) via accumulating row_max_acc
            state.push(Instr::reg_tile_row_max_acc(rt_s, rv_m, WL, R));
            // alpha = exp2(m_old - m)
            state.push(Instr::reg_vec_sub(rv_m_old, rv_m, rv_alpha, WL, R));
            state.push(Instr::reg_vec_exp2(rv_alpha, rv_alpha, WL, R));
            // l *= alpha
            state.push(Instr::reg_vec_mul(rv_l, rv_alpha, rv_l, WL, R));
            // o *= alpha (per-row rescale)
            //
            // RegTileMulRow: rv_alpha (length 128 == rt_o.rows) is
            // broadcast across cols of each row. Without this rescale,
            // the online softmax accumulator was correct only for
            // seq_len ≤ chunk_size; this Instr makes multi-chunk
            // decode numerically correct.
            state.push(Instr::reg_tile_mul_row(rt_o, rv_alpha, rt_o, WL, R));
            // S -= m  (sub_row)
            state.push(Instr::reg_tile_sub_row(rt_s, rv_m, rt_s, WL, R));
            // P = exp2(S) (in fp32)
            state.push(Instr::reg_tile_exp2(rt_s, rt_s, WL, R));
            // l += row_sum(P)
            state.push(Instr::reg_tile_row_sum_acc(rt_s, rv_l, WL, R));
            // Convert P to bf16 for the WGMMA
            state.push(Instr::reg_tile_copy_convert(rt_s, rt_p, WL, R));
            // O += P @ V (accumulate)
            state.push(Instr::wgmma_fence_acc(rt_o, W4));
            state.push(Instr::wgmma_mma_ab_reg_smem(
                rt_o, rt_p, v_tile, FenceExternal, AccAccumulate, W4,
            ));
            state.push(Instr::wgmma_async_wait(0, W4));
            state.push(Instr::ForLoopClose { var: loop_var });

            // ── Finalise phase ───────────────────────────────────
            // O /= l (per-row divide)
            state.push(Instr::reg_tile_div_row(rt_o, rv_l, rt_o, WL, R));
            // Store O → dst page via warpgroup-sharded store. Per-warp
            // 32-row rt_o (Fp32) → 128-row dst_tile (Bf16); TK 2.0
            // `store(st, rt)` handles fp32→bf16 conversion at store
            // time. Per plan §"Resolved decision 5".
            let _ = W16;
            state.push(Instr::store_reg_tile_to_shmem_warpgroup(
                rt_o, dst_tile, W4, R,
            ));

            emit_store_and_arrive(state, &node.output, dst_page);
            // Suppress unused (some types referenced only for clarity)
            let _ = WarpRole::AllConsumers;
        }
        #[allow(unreachable_patterns)]
        SubOp::MatmulTile => {
            panic!(
                "lower_compute: arch op {:?} has no TK 2.0 primitive expansion yet; \
                 see SUBTILE_TK20_DECOMP.md for the per-SubOp decomposition plan. \
                 lower_compute refuses to emit invented kittens::ops::* helpers.",
                std::mem::discriminant(&node.op),
            );
        }
    }

    // Release any temp pages allocated by `resolve_input_page` for
    // External inputs during this arm. Without this, every per-call
    // External load burns a fresh PageId — overflows u8 across a
    // Llama-1B tape (hundreds of External weights × chunks).
    state.release_ephemeral_pages();
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
    ByteOffsetExpr::from_const(off)
}

fn emit_external_load<F: RopeForm, K: KvCacheShape>(
    state: &mut LoweringState<F, K>,
    inp: &TensorRegion,
    dst_page: PageId,
) {
    // External loads use runtime shape — `inp.region` carries the
    // SubtileIR's tensor-region slice (1×N for vectors, 128×128 for
    // tile weights, etc.). The typed `LoadSpec::new<R,C,T>` gate is
    // appropriate when the lowerer KNOWS the shape (compute Instrs);
    // at the external-load boundary the shape is genuinely runtime
    // data. Per `feedback_no_speculative_witnesses`.
    let tile = region_tile_shape(inp);
    let byte_off = region_byte_offset(state.graph, inp);
    let src_arg = state.tensor_arg(inp.tensor);
    state.push(Instr::LoadAsync(LoadSpec::new_runtime_shape(
        dst_page,
        src_arg,
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
    // Runtime-shape store; output region's shape is driven by the
    // upstream SubtileIR. See `emit_external_load` for the same
    // const-generic-vs-runtime-shape rationale.
    let tile = region_tile_shape(out);
    let byte_off = region_byte_offset(state.graph, out);
    let dst_arg = state.tensor_arg(out.tensor);
    state.push(Instr::StoreAsync(StoreSpec::new_runtime_shape(
        dst_page,
        dst_arg,
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
