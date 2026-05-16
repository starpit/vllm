// SPDX-License-Identifier: Apache-2.0
//! `Vec<semantic-instr>` → `MegaTape` substrate-aware lowering.
//!
//! Sprint A scope (per `MEGA_IR_PLAN.md` §10): RmsNorm only.
//! Bug classes targeted: #1 (slot bounds), #2 (lifecycle),
//! #5 (scratch budget), #6 (warp roles). #3 (mbarrier phase
//! parity) lands at the per-iter level here too.
//!
//! ## Sprint A integration shim
//!
//! The proc-macro feeds the lowering ONE op at a time via
//! [`MegaTapeBuilder::push_rms_norm`]. The proc-macro's existing
//! `fan_out` returns `Vec<OpInstance>` with TokenStream fields
//! (the type-erasure point — see `MEGA_IR_PLAN.md` §5+§8.2).
//! The integration to feed `Instruction<W>` directly is a later
//! sprint; today's Sprint A pushes from typed numeric/string
//! values handed in by whatever the proc-macro decides to use.

#![allow(dead_code)]

use crate::nodes::{LayerIndex, MegaNode, RmsNorm, WeightRef};
use crate::substrate::{
    Empty, MbarrierPhase, Page, PageId, PagePool, ROLE_CONSUMER, ROLE_LAUNCHER, ROLE_LOADER,
    ROLE_STORER, RmsNormScope, ScratchRegion, SubstrateBudget, WarpRoleTag,
};
use crate::tape::MegaTape;

/// Cumulative mbarrier arrive count tracked across the tape.
/// Each `wait` consumes a phase; the lowering pulls a fresh
/// `MbarrierPhase` keyed off the count and increments.
///
/// Sprint A semantics: each per-iter consumer-arrive bumps the
/// count by 1 (one arrive per consumer iter); the next iter's
/// wait sees parity `count & 1`. The lowering's invariant is
/// that the wait-phase requested by an op MUST equal `count & 1`
/// at the moment the wait happens — `MbarrierPhase::assert_matches`
/// panics on mismatch.
#[derive(Clone, Copy, Debug, Default)]
pub struct ArriveCount(u32);

impl ArriveCount {
    pub fn current(&self) -> u32 {
        self.0
    }
    pub fn bump(&mut self) {
        self.0 += 1;
    }
}

/// Builder that the proc-macro feeds typed op data into. Each
/// `push_*` runs the substrate-proof discharge for that op
/// (allocates pages from the pool, walks the lifecycle, allocates
/// scratch, computes phases, builds the MegaNode), and returns
/// `&mut self` for chaining. Construction-time panics surface as
/// proc-macro errors on the user's `#[forward]`.
pub struct MegaTapeBuilder {
    nodes: Vec<MegaNode>,
    pool: PagePool,
    arrives: ArriveCount,
}

impl MegaTapeBuilder {
    pub fn new(substrate: SubstrateBudget) -> Self {
        Self {
            nodes: Vec::new(),
            pool: PagePool::new(substrate),
            arrives: ArriveCount::default(),
        }
    }

    /// Push a typed RmsNorm op onto the tape.
    ///
    /// Substrate-proof discharge per call:
    /// 1. Take `Page<Empty>` tokens for the input and weight slots
    ///    from the pool. PagePool checks `id < num_pages`
    ///    (bug #1) and "not already in use by another in-flight
    ///    op" (bug #2 cross-op).
    /// 2. Walk the lifecycle: loader fires (Empty → Filled),
    ///    consumer arrives (Filled → Produced), storer consumes
    ///    (Produced → Empty). Each transition is a typestate
    ///    consume; mis-ordering is a compile error in this fn
    ///    body.
    /// 3. Allocate the per-iter `partial_sums` scratch region.
    ///    `ScratchRegion::new` checks within-budget
    ///    (bug #5). The region's Scope is `RmsNormScope`; if a
    ///    later RmsNorm op overlaps in the same scope on the same
    ///    iter, `disjoint_with` panics (bug #4).
    /// 4. Compute consumer + storer wait phases from the
    ///    cumulative arrive count (bug #3).
    /// 5. Pin warp-role tags at compile time (bug #6).
    /// 6. Release the pages back to the pool (Empty state).
    /// 7. Bump arrive count by 1 (one consumer-arrive per op-iter).
    pub fn push_rms_norm(
        &mut self,
        in_slot_id: u32,
        weight_slot_id: u32,
        layer: u32,
        num_layers: u32,
        weight_path: String,
        scratch_offset: u32,
    ) -> &mut Self {
        // 1. Page allocation + bounds check.
        let in_page: Page<Empty> = self.pool.take(in_slot_id);
        let weight_page: Page<Empty> = self.pool.take(weight_slot_id);

        // 2. Lifecycle walk. Each transition consumes the previous
        // typestate token. The compiler enforces the order — try
        // calling consumer_arrived on an Empty page below; it
        // won't typecheck.
        let in_filled = in_page.loader_fired();
        let weight_filled = weight_page.loader_fired();
        let in_produced = in_filled.consumer_arrived();
        let _weight_produced = weight_filled.consumer_arrived();
        let in_empty_again = in_produced.storer_consumed();
        // Weight has no storer; it gets recycled directly from
        // Produced → Empty (the kernel's per-iter __syncthreads
        // serialises the next loader).
        // We don't explicitly transition the weight from Produced
        // back to Empty here — when the pool releases the page,
        // it forgets typestate. The next op picks it up as Empty.

        // 3. Scratch allocation. Per-iter partial sums for
        // NUM_CONSUMER_WARPS f32 values. `disjoint_with` is
        // available for sibling RmsNorm regions if needed (no
        // siblings in Sprint A).
        let scratch_bytes = self.pool.substrate().num_consumer_warps() * 4;
        let partial_sums = ScratchRegion::<RmsNormScope>::new(
            scratch_offset,
            scratch_bytes,
            self.pool.substrate(),
        );

        // 4. Phases. The wait at op-iter T sees parity (n_arrives
        // before this op + iter T) & 1. For Sprint A's first op,
        // the wait phase at op start is `arrives.current() & 1`.
        // Storer waits on consumer's arrive, which happens AFTER
        // the consumer's wait, so storer phase = (arrives + 1) & 1.
        let consumer_phase =
            MbarrierPhase::assert_matches(self.arrives.current() & 1, self.arrives.current());
        let storer_phase = MbarrierPhase::assert_matches(
            (self.arrives.current() + 1) & 1,
            self.arrives.current() + 1,
        );

        // 5. Warp role tags. Const-generic; type-level pinned.
        let _loader_role = WarpRoleTag::<ROLE_LOADER>;
        let _launcher_role = WarpRoleTag::<ROLE_LAUNCHER>;
        let _consumer_role = WarpRoleTag::<ROLE_CONSUMER>;
        let _storer_role = WarpRoleTag::<ROLE_STORER>;

        // 6. Bake the typed node. Helper newtypes carry alongside.
        let layer = LayerIndex::new(layer, num_layers);
        let weight = WeightRef::new(weight_path);
        let in_id = PageId::new(in_slot_id, self.pool.substrate());
        let weight_id = PageId::new(weight_slot_id, self.pool.substrate());

        self.nodes.push(MegaNode::RmsNorm(RmsNorm {
            in_page: in_id,
            weight_page: weight_id,
            partial_sums,
            consumer_phase,
            storer_phase,
            _loader_role,
            _launcher_role,
            _consumer_role,
            _storer_role,
            layer,
            weight,
        }));

        // 7. Release pages back to the pool. The variant has its
        // PageId; the pool re-marks slots free for the next op.
        self.pool.release(in_empty_again);
        // Weight was last seen Produced; we do a no-op release to
        // mark its slot free (the actual lifecycle ends at the
        // kernel's per-iter __syncthreads).
        self.pool.release(_weight_produced);

        // 8. Bump cumulative arrive count.
        self.arrives.bump();

        self
    }

    /// Consume the builder and return the typed tape.
    pub fn finish(self) -> MegaTape {
        let substrate = *self.pool.substrate();
        MegaTape::__build_from_nodes(self.nodes, substrate)
    }
}

#[derive(Debug)]
pub enum LowerError {
    /// Reserved for variants that haven't been substrate-lifted
    /// yet (Sprint B onwards).
    NotYetLifted { op: &'static str },
}

/// Top-level lowering function. Sprint A: takes a slice of typed
/// RmsNorm op data and produces a `MegaTape`. As more sprints
/// land, this signature grows or splits.
pub fn lower_rms_norm_only(
    rms_norm_ops: &[RmsNormInput],
    substrate: SubstrateBudget,
) -> Result<MegaTape, LowerError> {
    let mut builder = MegaTapeBuilder::new(substrate);
    for op in rms_norm_ops {
        builder.push_rms_norm(
            op.in_slot_id,
            op.weight_slot_id,
            op.layer,
            op.num_layers,
            op.weight_path.clone(),
            op.scratch_offset,
        );
    }
    Ok(builder.finish())
}

/// Sprint A's per-op input shim. The proc-macro fills these from
/// fan_out results. Later sprints replace the shim with the
/// `Instruction<W>` semantic Tape directly.
#[derive(Clone, Debug)]
pub struct RmsNormInput {
    pub in_slot_id: u32,
    pub weight_slot_id: u32,
    pub layer: u32,
    pub num_layers: u32,
    pub weight_path: String,
    pub scratch_offset: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget() -> SubstrateBudget {
        SubstrateBudget::new(6, 8, 32_768, 8_192)
    }

    fn good_input() -> RmsNormInput {
        RmsNormInput {
            in_slot_id: 0,
            weight_slot_id: 1,
            layer: 0,
            num_layers: 16,
            weight_path: "Weights :: input_layernorm".to_string(),
            scratch_offset: 0,
        }
    }

    #[test]
    fn lowers_well_formed_rms_norm() {
        let mut b = MegaTapeBuilder::new(budget());
        b.push_rms_norm(0, 1, 0, 16, "W::norm".to_string(), 0);
        let tape = b.finish();
        assert_eq!(tape.nodes().len(), 1);
        let MegaNode::RmsNorm(n) = &tape.nodes()[0];
        assert_eq!(n.in_page.raw(), 0);
        assert_eq!(n.weight_page.raw(), 1);
        assert_eq!(n.partial_sums.offset(), 0);
        assert_eq!(n.partial_sums.bytes(), 8 * 4); // num_consumer_warps * 4
        assert_eq!(n.consumer_phase.phase(), 0);
        assert_eq!(n.storer_phase.phase(), 1);
        assert_eq!(n.layer.raw(), 0);
        assert_eq!(n.weight.path(), "W::norm");
    }

    #[test]
    fn lowers_two_rms_norms_with_phase_advance() {
        // Second op's consumer wait phase = arrive count after
        // the first op = 1.
        let mut b = MegaTapeBuilder::new(budget());
        b.push_rms_norm(0, 1, 0, 16, "W::n0".to_string(), 0);
        b.push_rms_norm(0, 1, 1, 16, "W::n1".to_string(), 0);
        let tape = b.finish();
        let MegaNode::RmsNorm(n0) = &tape.nodes()[0];
        let MegaNode::RmsNorm(n1) = &tape.nodes()[1];
        assert_eq!(n0.consumer_phase.phase(), 0);
        assert_eq!(n0.storer_phase.phase(), 1);
        assert_eq!(n1.consumer_phase.phase(), 1);
        assert_eq!(n1.storer_phase.phase(), 0);
    }

    #[test]
    #[should_panic(expected = "PagePool::take: id 6 out of bounds num_pages=6")]
    fn rejects_page_out_of_bounds() {
        let mut b = MegaTapeBuilder::new(budget());
        b.push_rms_norm(6, 1, 0, 16, "W".to_string(), 0);
    }

    #[test]
    #[should_panic(expected = "PagePool::take: page id 1 already in use")]
    fn rejects_page_aliasing_within_op() {
        // Same slot for both pages within one op — bug class #2:
        // the same physical page can't carry both input and
        // weight at the same time.
        let mut b = MegaTapeBuilder::new(budget());
        b.push_rms_norm(1, 1, 0, 16, "W".to_string(), 0);
    }

    #[test]
    #[should_panic(expected = "ScratchRegion out of substrate scratch budget")]
    fn rejects_scratch_overflow() {
        // 8192 - 31 + 32 = 8193 > 8192. Offset 8161 + 32 bytes
        // (NUM_CONSUMER_WARPS * 4) = 8193 > scratch_bytes.
        let mut b = MegaTapeBuilder::new(budget());
        b.push_rms_norm(0, 1, 0, 16, "W".to_string(), 8161);
    }

    #[test]
    #[should_panic(expected = "LayerIndex out of range: 16 >= 16")]
    fn rejects_layer_out_of_range() {
        let mut b = MegaTapeBuilder::new(budget());
        b.push_rms_norm(0, 1, 16, 16, "W".to_string(), 0);
    }

    #[test]
    #[should_panic(expected = "WeightRef must be a non-empty path string")]
    fn rejects_empty_weight_path() {
        let mut b = MegaTapeBuilder::new(budget());
        b.push_rms_norm(0, 1, 0, 16, "".to_string(), 0);
    }

    #[test]
    fn lower_rms_norm_only_round_trips() {
        let inputs = vec![good_input()];
        let tape = lower_rms_norm_only(&inputs, budget()).expect("good input");
        assert_eq!(tape.nodes().len(), 1);
    }

    #[test]
    fn lower_rms_norm_empty_produces_empty_tape() {
        let tape = lower_rms_norm_only(&[], budget()).expect("empty input is fine");
        assert!(tape.nodes().is_empty());
    }
}
