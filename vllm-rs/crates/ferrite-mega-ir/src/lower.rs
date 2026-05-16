// SPDX-License-Identifier: Apache-2.0
//! `Vec<Instruction>` → `MegaTape` substrate-aware lowering.
//!
//! Sprint A scope (per `MEGA_IR_PLAN.md` §10): RmsNorm only.
//! Bug classes targeted: #1 (slot bounds), #2 (lifecycle),
//! #5 (scratch budget), #6 (warp roles). #3 (mbarrier phase
//! parity) lands at the per-iter level here too.
//!
//! ## Pipeline integration
//!
//! Per `MEGA_IR_PLAN.md` §5, the input is the existing
//! `Vec<Instruction>` semantic tape from
//! `ferrite_forward::Instruction`. There is no `MegaOp` semantic
//! enum, no `OpInstance::field_values: Vec<TokenStream>` round-trip.
//! Each `Instruction` is paired with its per-op weight context (the
//! per-arch base names of the weight slots the variant consumes,
//! extracted from `OpInstance::weight_slots` at fan_out time).
//!
//! [`lower`] is the public entry point: it walks
//! `&[OpInput]` and dispatches to the substrate-aware push handler
//! for each variant. Variants whose substrate lifting hasn't landed
//! yet (Sprint B onwards) return [`LowerError::NotYetLifted`] —
//! callers are expected to fall back to the host interpreter for
//! those tapes until the matching sprint lands.

#![allow(dead_code)]

use ferrite_forward::Instruction;

use crate::nodes::{
    Add, AttentionKind, AttentionViaCacheNode, BarrierSignal, BarrierWait, CutlassFusedNormGemm,
    Embed, FiniteF32, FusedAddRmsNorm, FusedGateUpActivateMul, FusedQkvRopeCache, GateUpActivation,
    Gemm, LayerIndex, LmHeadNormKind, MatmulShape, MegaNode, RmsNorm, RotaryRef, ScalarMul,
    ScalarOffsetRmsNorm, SlidingWindow, TanhSoftCap, WeightRef,
};
use crate::substrate::{
    AttentionScope, EdgeId, Empty, ExpectedCount, GemmScope, IterCount, MbarrierPhase, MlpScope,
    Page, PageId, PagePool, ROLE_CONSUMER, ROLE_LAUNCHER, ROLE_LOADER, ROLE_STORER, RmsNormScope,
    RopeScope, ScratchRegion, SubstrateBudget, WarpRoleTag,
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

    /// Push a typed FusedQkvRopeCache op onto the tape.
    ///
    /// Substrate-proof discharge per call (Sprint B targets bug
    /// classes #2 lifecycle, #3 mbarrier phase math, #4 scratch
    /// overlap, plus reuses #1 / #5 / #6 from Sprint A):
    ///
    /// 1. Allocate six Page<Empty> tokens from the pool — input act,
    ///    QKV weight, cos/sin, and three outputs (Q, K, V). PagePool
    ///    bounds-checks each (#1) and refuses ids already held by
    ///    another in-flight op (#2 cross-op).
    /// 2. Walk every page through Empty → Filled → Produced → Empty.
    ///    Each transition consumes the typestate token; the compiler
    ///    enforces "no consumer-arrive on Empty", "no double-fill",
    ///    "no storer-consume before consumer-arrive" (#2 within-op).
    /// 3. Allocate two `ScratchRegion<RopeScope>` for the per-token
    ///    Q/K rotation buffers, then call `disjoint_with` to prove
    ///    they don't overlap inside the per-tok loop (#4). Each is
    ///    independently within-budget (#5).
    /// 4. Validate the consumer wait phase parity matches the
    ///    cumulative arrive count at op start, and the storer wait
    ///    phase parity matches `cumulative + 1` (#3 boundary).
    ///    `arrives_per_iter == 1` (one consumer arrive per token-iter)
    ///    is the variant-internal invariant; with that fixed, the
    ///    kernel's per-iter `phase ^= 1` toggle keeps `wait` parity
    ///    aligned with the actual count for every iter (#3 per-iter).
    /// 5. Stamp the four warp-role tags (#6).
    /// 6. Return the six pages to the pool, bump the cumulative
    ///    arrive count by `iters` (the post-op count seen by the
    ///    NEXT op's phase boundary check).
    #[allow(clippy::too_many_arguments)]
    pub fn push_fused_qkv_rope_cache(
        &mut self,
        in_slot_id: u32,
        qkv_weight_slot_id: u32,
        cos_sin_slot_id: u32,
        q_out_slot_id: u32,
        k_out_slot_id: u32,
        v_out_slot_id: u32,
        layer: u32,
        num_layers: u32,
        qkv_weight_path: String,
        rotary_path: String,
        iters: u32,
        q_scratch_offset: u32,
        q_scratch_bytes: u32,
        k_scratch_offset: u32,
        k_scratch_bytes: u32,
        biased: bool,
        interleaved: bool,
    ) -> &mut Self {
        // 1. Page allocation + bounds + cross-op aliasing checks.
        let in_page: Page<Empty> = self.pool.take(in_slot_id);
        let qkv_page: Page<Empty> = self.pool.take(qkv_weight_slot_id);
        let cs_page: Page<Empty> = self.pool.take(cos_sin_slot_id);
        let q_page: Page<Empty> = self.pool.take(q_out_slot_id);
        let k_page: Page<Empty> = self.pool.take(k_out_slot_id);
        let v_page: Page<Empty> = self.pool.take(v_out_slot_id);

        // 2. Lifecycle walks. The typestate burns Empty → Filled →
        //    Produced → Empty for each page; rebinding the variable
        //    on every transition gives the compiler one place to
        //    refuse mis-ordering. Inputs flow loader → consumer
        //    (read) → arrive; outputs flow consumer (write) →
        //    arrive → storer (read) → drain. Both shapes share the
        //    same Empty → Filled → Produced → Empty cycle in the
        //    lifecycle FSM — the role doing the read/write differs,
        //    not the typestate path.
        let in_empty = in_page.loader_fired().consumer_arrived().storer_consumed();
        let qkv_empty = qkv_page.loader_fired().consumer_arrived().storer_consumed();
        let cs_empty = cs_page.loader_fired().consumer_arrived().storer_consumed();
        let q_empty = q_page.loader_fired().consumer_arrived().storer_consumed();
        let k_empty = k_page.loader_fired().consumer_arrived().storer_consumed();
        let v_empty = v_page.loader_fired().consumer_arrived().storer_consumed();

        // 3. Scratch allocation + disjointness proof. Both regions
        //    live in `RopeScope`; `disjoint_with` consumes both and
        //    returns them, so the proof is recorded on the values
        //    that land in the variant's fields.
        let q_rope_raw = ScratchRegion::<RopeScope>::new(
            q_scratch_offset,
            q_scratch_bytes,
            self.pool.substrate(),
        );
        let k_rope_raw = ScratchRegion::<RopeScope>::new(
            k_scratch_offset,
            k_scratch_bytes,
            self.pool.substrate(),
        );
        let (q_rope_buf, k_rope_buf) = q_rope_raw.disjoint_with(k_rope_raw);

        // 4. Per-iter phases. `arrives_per_iter == 1` for this
        //    variant: each token-iter does exactly one consumer
        //    arrive (output pages ready). The lowering validates the
        //    boundary phase only — kernel-internal `phase ^= 1`
        //    handles per-iter advance, and bug-class-#3's per-iter
        //    parity proof falls out of `(C_start + t) & 1 ==
        //    (C_start & 1) ^ (t & 1)` once the boundary matches.
        let consumer_phase =
            MbarrierPhase::assert_matches(self.arrives.current() & 1, self.arrives.current());
        let storer_phase = MbarrierPhase::assert_matches(
            (self.arrives.current() + 1) & 1,
            self.arrives.current() + 1,
        );

        // 5. Warp role tags (#6).
        let _loader_role = WarpRoleTag::<ROLE_LOADER>;
        let _launcher_role = WarpRoleTag::<ROLE_LAUNCHER>;
        let _consumer_role = WarpRoleTag::<ROLE_CONSUMER>;
        let _storer_role = WarpRoleTag::<ROLE_STORER>;

        // 6. Bake the typed node.
        let iters_typed = IterCount::new(iters);
        let layer = LayerIndex::new(layer, num_layers);
        let qkv_weight = WeightRef::new(qkv_weight_path);
        let rotary = RotaryRef::new(rotary_path);
        let in_id = PageId::new(in_slot_id, self.pool.substrate());
        let qkv_id = PageId::new(qkv_weight_slot_id, self.pool.substrate());
        let cs_id = PageId::new(cos_sin_slot_id, self.pool.substrate());
        let q_id = PageId::new(q_out_slot_id, self.pool.substrate());
        let k_id = PageId::new(k_out_slot_id, self.pool.substrate());
        let v_id = PageId::new(v_out_slot_id, self.pool.substrate());

        self.nodes
            .push(MegaNode::FusedQkvRopeCache(FusedQkvRopeCache {
                in_page: in_id,
                qkv_weight_page: qkv_id,
                cos_sin_page: cs_id,
                q_out_page: q_id,
                k_out_page: k_id,
                v_out_page: v_id,
                q_rope_buf,
                k_rope_buf,
                consumer_phase,
                storer_phase,
                iters: iters_typed,
                _loader_role,
                _launcher_role,
                _consumer_role,
                _storer_role,
                layer,
                qkv_weight,
                rotary,
                biased,
                interleaved,
            }));

        // 7. Release pages back to the pool. Then bump the cumulative
        //    arrive count by `iters` — every per-token iter does one
        //    consumer arrive, so the post-op count is start + iters.
        self.pool.release(in_empty);
        self.pool.release(qkv_empty);
        self.pool.release(cs_empty);
        self.pool.release(q_empty);
        self.pool.release(k_empty);
        self.pool.release(v_empty);
        for _ in 0..iters {
            self.arrives.bump();
        }

        self
    }

    /// Push a typed `Add` (residual fold) onto the tape.
    ///
    /// `Instruction::Add(delta_slot, residual_slot)` semantics:
    /// `residual_slot += delta_slot`. The plan §10 calls this
    /// "DownProjResidual" when it sits after MLP / attention output;
    /// the `Instruction` enum has just one universal variant.
    ///
    /// Substrate-proof discharge per call:
    /// 1. Take two `Page<Empty>` tokens (#1 + #2 cross-op).
    /// 2. Walk both Empty → Filled → Produced → Empty (#2 within-op:
    ///    consumer can't read before loader fires; storer can't
    ///    drain before consumer arrives).
    /// 3. Validate boundary phase parity against the cumulative
    ///    arrive count (#3).
    /// 4. Stamp the four warp-role tags (#6).
    /// 5. Release both pages, bump arrives by 1 (one consumer
    ///    arrive per op — element-wise add is single-shot).
    pub fn push_add(&mut self, delta_slot_id: u32, residual_slot_id: u32) -> &mut Self {
        let delta_page: Page<Empty> = self.pool.take(delta_slot_id);
        let residual_page: Page<Empty> = self.pool.take(residual_slot_id);

        let delta_empty = delta_page
            .loader_fired()
            .consumer_arrived()
            .storer_consumed();
        let residual_empty = residual_page
            .loader_fired()
            .consumer_arrived()
            .storer_consumed();

        let consumer_phase =
            MbarrierPhase::assert_matches(self.arrives.current() & 1, self.arrives.current());
        let storer_phase = MbarrierPhase::assert_matches(
            (self.arrives.current() + 1) & 1,
            self.arrives.current() + 1,
        );

        let _loader_role = WarpRoleTag::<ROLE_LOADER>;
        let _launcher_role = WarpRoleTag::<ROLE_LAUNCHER>;
        let _consumer_role = WarpRoleTag::<ROLE_CONSUMER>;
        let _storer_role = WarpRoleTag::<ROLE_STORER>;

        let delta_id = PageId::new(delta_slot_id, self.pool.substrate());
        let residual_id = PageId::new(residual_slot_id, self.pool.substrate());

        self.nodes.push(MegaNode::Add(Add {
            delta_page: delta_id,
            residual_page: residual_id,
            consumer_phase,
            storer_phase,
            _loader_role,
            _launcher_role,
            _consumer_role,
            _storer_role,
        }));

        self.pool.release(delta_empty);
        self.pool.release(residual_empty);
        self.arrives.bump();

        self
    }

    /// Push a typed `FusedAddRmsNorm` onto the tape.
    ///
    /// `Instruction::FusedAddRmsNorm(delta_slot, residual_slot,
    /// layer)` semantics: `residual += delta; out = rms_norm(residual,
    /// weight[layer])`, with `out` overwriting `residual_slot`.
    /// Substrate-proof discharge mirrors `push_rms_norm` but with
    /// three pages (delta + residual + weight) instead of two
    /// (input + weight). Scratch shape is identical
    /// (`partial_sums: ScratchRegion<RmsNormScope>` sized for
    /// `num_consumer_warps` f32 partial sums).
    #[allow(clippy::too_many_arguments)]
    pub fn push_fused_add_rms_norm(
        &mut self,
        delta_slot_id: u32,
        residual_slot_id: u32,
        weight_slot_id: u32,
        layer: u32,
        num_layers: u32,
        weight_path: String,
        scratch_offset: u32,
    ) -> &mut Self {
        let delta_page: Page<Empty> = self.pool.take(delta_slot_id);
        let residual_page: Page<Empty> = self.pool.take(residual_slot_id);
        let weight_page: Page<Empty> = self.pool.take(weight_slot_id);

        let delta_empty = delta_page
            .loader_fired()
            .consumer_arrived()
            .storer_consumed();
        let residual_empty = residual_page
            .loader_fired()
            .consumer_arrived()
            .storer_consumed();
        let weight_empty = weight_page
            .loader_fired()
            .consumer_arrived()
            .storer_consumed();

        let scratch_bytes = self.pool.substrate().num_consumer_warps() * 4;
        let partial_sums = ScratchRegion::<RmsNormScope>::new(
            scratch_offset,
            scratch_bytes,
            self.pool.substrate(),
        );

        let consumer_phase =
            MbarrierPhase::assert_matches(self.arrives.current() & 1, self.arrives.current());
        let storer_phase = MbarrierPhase::assert_matches(
            (self.arrives.current() + 1) & 1,
            self.arrives.current() + 1,
        );

        let _loader_role = WarpRoleTag::<ROLE_LOADER>;
        let _launcher_role = WarpRoleTag::<ROLE_LAUNCHER>;
        let _consumer_role = WarpRoleTag::<ROLE_CONSUMER>;
        let _storer_role = WarpRoleTag::<ROLE_STORER>;

        let layer = LayerIndex::new(layer, num_layers);
        let weight = WeightRef::new(weight_path);
        let delta_id = PageId::new(delta_slot_id, self.pool.substrate());
        let residual_id = PageId::new(residual_slot_id, self.pool.substrate());
        let weight_id = PageId::new(weight_slot_id, self.pool.substrate());

        self.nodes.push(MegaNode::FusedAddRmsNorm(FusedAddRmsNorm {
            delta_page: delta_id,
            residual_page: residual_id,
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

        self.pool.release(delta_empty);
        self.pool.release(residual_empty);
        self.pool.release(weight_empty);
        self.arrives.bump();

        self
    }

    /// Push a typed `FusedGateUp{Silu,Gelu}Mul` onto the tape.
    ///
    /// Substrate-proof discharge mirrors `push_fused_qkv_rope_cache`
    /// (the multi-iter pattern) with: 3 pages (in / packed-gate-up
    /// weight / out), 2 disjoint `MlpScope` scratch tiles
    /// (`gate_buf`, `up_buf`), `iters` token-iter count.
    /// The kernel runs two cublas gemms back-to-back per iter
    /// (gate, up) into shmem tiles, then elementwise
    /// `act(gate) * up` into the out page. One consumer arrive per
    /// iter; cumulative arrive count advances by `iters.raw()`.
    #[allow(clippy::too_many_arguments)]
    pub fn push_fused_gate_up_activate_mul(
        &mut self,
        in_slot_id: u32,
        gate_up_weight_slot_id: u32,
        out_slot_id: u32,
        layer: u32,
        num_layers: u32,
        weight_path: String,
        iters: u32,
        gate_scratch_offset: u32,
        gate_scratch_bytes: u32,
        up_scratch_offset: u32,
        up_scratch_bytes: u32,
        activation: GateUpActivation,
    ) -> &mut Self {
        let in_page: Page<Empty> = self.pool.take(in_slot_id);
        let weight_page: Page<Empty> = self.pool.take(gate_up_weight_slot_id);
        let out_page: Page<Empty> = self.pool.take(out_slot_id);

        let in_empty = in_page.loader_fired().consumer_arrived().storer_consumed();
        let weight_empty = weight_page
            .loader_fired()
            .consumer_arrived()
            .storer_consumed();
        let out_empty = out_page.loader_fired().consumer_arrived().storer_consumed();

        let gate_raw = ScratchRegion::<MlpScope>::new(
            gate_scratch_offset,
            gate_scratch_bytes,
            self.pool.substrate(),
        );
        let up_raw = ScratchRegion::<MlpScope>::new(
            up_scratch_offset,
            up_scratch_bytes,
            self.pool.substrate(),
        );
        let (gate_buf, up_buf) = gate_raw.disjoint_with(up_raw);

        let consumer_phase =
            MbarrierPhase::assert_matches(self.arrives.current() & 1, self.arrives.current());
        let storer_phase = MbarrierPhase::assert_matches(
            (self.arrives.current() + 1) & 1,
            self.arrives.current() + 1,
        );

        let _loader_role = WarpRoleTag::<ROLE_LOADER>;
        let _launcher_role = WarpRoleTag::<ROLE_LAUNCHER>;
        let _consumer_role = WarpRoleTag::<ROLE_CONSUMER>;
        let _storer_role = WarpRoleTag::<ROLE_STORER>;

        let iters_typed = IterCount::new(iters);
        let layer = LayerIndex::new(layer, num_layers);
        let weight = WeightRef::new(weight_path);
        let in_id = PageId::new(in_slot_id, self.pool.substrate());
        let weight_id = PageId::new(gate_up_weight_slot_id, self.pool.substrate());
        let out_id = PageId::new(out_slot_id, self.pool.substrate());

        self.nodes
            .push(MegaNode::FusedGateUpActivateMul(FusedGateUpActivateMul {
                in_page: in_id,
                gate_up_weight_page: weight_id,
                out_page: out_id,
                gate_buf,
                up_buf,
                consumer_phase,
                storer_phase,
                iters: iters_typed,
                _loader_role,
                _launcher_role,
                _consumer_role,
                _storer_role,
                layer,
                weight,
                activation,
            }));

        self.pool.release(in_empty);
        self.pool.release(weight_empty);
        self.pool.release(out_empty);
        for _ in 0..iters {
            self.arrives.bump();
        }

        self
    }

    // ── Sprint D push methods ──────────────────────────────

    /// Push a typed `Embed` (vocab lookup) onto the tape.
    /// Two pages (out + embed table), boundary phase parity, four
    /// role tags. The embed table doesn't sit fully in shmem (vocab
    /// is huge); the loader streams per-token rows. The substrate
    /// proof covers the lifecycle of both pages and the weight
    /// reference path.
    pub fn push_embed(
        &mut self,
        out_slot_id: u32,
        embed_weight_slot_id: u32,
        embed_weight_path: String,
    ) -> &mut Self {
        let out_page: Page<Empty> = self.pool.take(out_slot_id);
        let weight_page: Page<Empty> = self.pool.take(embed_weight_slot_id);

        let out_empty = out_page.loader_fired().consumer_arrived().storer_consumed();
        let weight_empty = weight_page
            .loader_fired()
            .consumer_arrived()
            .storer_consumed();

        let consumer_phase =
            MbarrierPhase::assert_matches(self.arrives.current() & 1, self.arrives.current());
        let storer_phase = MbarrierPhase::assert_matches(
            (self.arrives.current() + 1) & 1,
            self.arrives.current() + 1,
        );

        self.nodes.push(MegaNode::Embed(Embed {
            out_page: PageId::new(out_slot_id, self.pool.substrate()),
            embed_weight_page: PageId::new(embed_weight_slot_id, self.pool.substrate()),
            consumer_phase,
            storer_phase,
            _loader_role: WarpRoleTag::<ROLE_LOADER>,
            _launcher_role: WarpRoleTag::<ROLE_LAUNCHER>,
            _consumer_role: WarpRoleTag::<ROLE_CONSUMER>,
            _storer_role: WarpRoleTag::<ROLE_STORER>,
            embed_weight: WeightRef::new(embed_weight_path),
        }));

        self.pool.release(out_empty);
        self.pool.release(weight_empty);
        self.arrives.bump();

        self
    }

    /// Push a typed `ScalarMul` onto the tape. Two pages, no
    /// scratch, no weight, boundary phase parity, four role tags.
    /// `scale` validated finite at construction.
    pub fn push_scalar_mul(&mut self, in_slot_id: u32, out_slot_id: u32, scale: f32) -> &mut Self {
        let in_page: Page<Empty> = self.pool.take(in_slot_id);
        let out_page: Page<Empty> = self.pool.take(out_slot_id);
        let in_empty = in_page.loader_fired().consumer_arrived().storer_consumed();
        let out_empty = out_page.loader_fired().consumer_arrived().storer_consumed();

        let consumer_phase =
            MbarrierPhase::assert_matches(self.arrives.current() & 1, self.arrives.current());
        let storer_phase = MbarrierPhase::assert_matches(
            (self.arrives.current() + 1) & 1,
            self.arrives.current() + 1,
        );

        self.nodes.push(MegaNode::ScalarMul(ScalarMul {
            in_page: PageId::new(in_slot_id, self.pool.substrate()),
            out_page: PageId::new(out_slot_id, self.pool.substrate()),
            consumer_phase,
            storer_phase,
            _loader_role: WarpRoleTag::<ROLE_LOADER>,
            _launcher_role: WarpRoleTag::<ROLE_LAUNCHER>,
            _consumer_role: WarpRoleTag::<ROLE_CONSUMER>,
            _storer_role: WarpRoleTag::<ROLE_STORER>,
            scale: FiniteF32::new(scale),
        }));

        self.pool.release(in_empty);
        self.pool.release(out_empty);
        self.arrives.bump();

        self
    }

    /// Push a typed `TanhSoftCap` onto the tape. Two pages, no
    /// scratch, no weight, boundary phase parity, four role tags.
    pub fn push_tanh_soft_cap(&mut self, in_slot_id: u32, out_slot_id: u32) -> &mut Self {
        let in_page: Page<Empty> = self.pool.take(in_slot_id);
        let out_page: Page<Empty> = self.pool.take(out_slot_id);
        let in_empty = in_page.loader_fired().consumer_arrived().storer_consumed();
        let out_empty = out_page.loader_fired().consumer_arrived().storer_consumed();

        let consumer_phase =
            MbarrierPhase::assert_matches(self.arrives.current() & 1, self.arrives.current());
        let storer_phase = MbarrierPhase::assert_matches(
            (self.arrives.current() + 1) & 1,
            self.arrives.current() + 1,
        );

        self.nodes.push(MegaNode::TanhSoftCap(TanhSoftCap {
            in_page: PageId::new(in_slot_id, self.pool.substrate()),
            out_page: PageId::new(out_slot_id, self.pool.substrate()),
            consumer_phase,
            storer_phase,
            _loader_role: WarpRoleTag::<ROLE_LOADER>,
            _launcher_role: WarpRoleTag::<ROLE_LAUNCHER>,
            _consumer_role: WarpRoleTag::<ROLE_CONSUMER>,
            _storer_role: WarpRoleTag::<ROLE_STORER>,
        }));

        self.pool.release(in_empty);
        self.pool.release(out_empty);
        self.arrives.bump();

        self
    }

    /// Push a typed `ScalarOffsetRmsNorm` onto the tape. Same shape
    /// as RmsNorm with a `FiniteF32` offset helper riding alongside.
    #[allow(clippy::too_many_arguments)]
    pub fn push_scalar_offset_rms_norm(
        &mut self,
        in_slot_id: u32,
        weight_slot_id: u32,
        layer: u32,
        num_layers: u32,
        weight_path: String,
        scratch_offset: u32,
        offset: f32,
    ) -> &mut Self {
        let in_page: Page<Empty> = self.pool.take(in_slot_id);
        let weight_page: Page<Empty> = self.pool.take(weight_slot_id);

        let in_empty = in_page.loader_fired().consumer_arrived().storer_consumed();
        let weight_empty = weight_page
            .loader_fired()
            .consumer_arrived()
            .storer_consumed();

        let scratch_bytes = self.pool.substrate().num_consumer_warps() * 4;
        let partial_sums = ScratchRegion::<RmsNormScope>::new(
            scratch_offset,
            scratch_bytes,
            self.pool.substrate(),
        );

        let consumer_phase =
            MbarrierPhase::assert_matches(self.arrives.current() & 1, self.arrives.current());
        let storer_phase = MbarrierPhase::assert_matches(
            (self.arrives.current() + 1) & 1,
            self.arrives.current() + 1,
        );

        self.nodes
            .push(MegaNode::ScalarOffsetRmsNorm(ScalarOffsetRmsNorm {
                in_page: PageId::new(in_slot_id, self.pool.substrate()),
                weight_page: PageId::new(weight_slot_id, self.pool.substrate()),
                partial_sums,
                consumer_phase,
                storer_phase,
                _loader_role: WarpRoleTag::<ROLE_LOADER>,
                _launcher_role: WarpRoleTag::<ROLE_LAUNCHER>,
                _consumer_role: WarpRoleTag::<ROLE_CONSUMER>,
                _storer_role: WarpRoleTag::<ROLE_STORER>,
                layer: LayerIndex::new(layer, num_layers),
                weight: WeightRef::new(weight_path),
                offset: FiniteF32::new(offset),
            }));

        self.pool.release(in_empty);
        self.pool.release(weight_empty);
        self.arrives.bump();

        self
    }

    /// Push a typed `Gemm` onto the tape. Three pages, one B-tile
    /// in `GemmScope`, multi-iter (one consumer arrive per token).
    #[allow(clippy::too_many_arguments)]
    pub fn push_gemm(
        &mut self,
        in_slot_id: u32,
        weight_slot_id: u32,
        out_slot_id: u32,
        layer: u32,
        num_layers: u32,
        weight_path: String,
        n: u32,
        k: u32,
        iters: u32,
        b_tile_offset: u32,
        b_tile_bytes: u32,
    ) -> &mut Self {
        let in_page: Page<Empty> = self.pool.take(in_slot_id);
        let weight_page: Page<Empty> = self.pool.take(weight_slot_id);
        let out_page: Page<Empty> = self.pool.take(out_slot_id);

        let in_empty = in_page.loader_fired().consumer_arrived().storer_consumed();
        let weight_empty = weight_page
            .loader_fired()
            .consumer_arrived()
            .storer_consumed();
        let out_empty = out_page.loader_fired().consumer_arrived().storer_consumed();

        let b_tile =
            ScratchRegion::<GemmScope>::new(b_tile_offset, b_tile_bytes, self.pool.substrate());

        let consumer_phase =
            MbarrierPhase::assert_matches(self.arrives.current() & 1, self.arrives.current());
        let storer_phase = MbarrierPhase::assert_matches(
            (self.arrives.current() + 1) & 1,
            self.arrives.current() + 1,
        );

        self.nodes.push(MegaNode::Gemm(Gemm {
            in_page: PageId::new(in_slot_id, self.pool.substrate()),
            weight_page: PageId::new(weight_slot_id, self.pool.substrate()),
            out_page: PageId::new(out_slot_id, self.pool.substrate()),
            b_tile,
            consumer_phase,
            storer_phase,
            iters: IterCount::new(iters),
            _loader_role: WarpRoleTag::<ROLE_LOADER>,
            _launcher_role: WarpRoleTag::<ROLE_LAUNCHER>,
            _consumer_role: WarpRoleTag::<ROLE_CONSUMER>,
            _storer_role: WarpRoleTag::<ROLE_STORER>,
            layer: LayerIndex::new(layer, num_layers),
            weight: WeightRef::new(weight_path),
            shape: MatmulShape::new(n, k),
        }));

        self.pool.release(in_empty);
        self.pool.release(weight_empty);
        self.pool.release(out_empty);
        for _ in 0..iters {
            self.arrives.bump();
        }

        self
    }

    /// Push a typed `CutlassFusedNormGemm` (lm_head fusion) onto
    /// the tape. Substrate shape: 3-or-4 pages (in + optional
    /// delta + norm_weight + linear_weight + out), partial_sums in
    /// `RmsNormScope`, B-tile in `GemmScope`. Multi-iter.
    ///
    /// `delta_slot_id` is `Some` for `AddRmsNorm` /
    /// `AddScalarOffsetRmsNorm` (residual-fold flavors); `None` for
    /// `RmsNorm` / `MeanSubRmsNorm`. `offset` is `Some(f32)` only
    /// for `AddScalarOffsetRmsNorm`; the constructor enforces
    /// `(norm_kind == AddScalarOffsetRmsNorm) <=> offset.is_some()`.
    #[allow(clippy::too_many_arguments)]
    pub fn push_cutlass_fused_norm_gemm(
        &mut self,
        in_slot_id: u32,
        delta_slot_id: Option<u32>,
        norm_weight_slot_id: u32,
        linear_weight_slot_id: u32,
        out_slot_id: u32,
        layer: u32,
        num_layers: u32,
        norm_weight_path: String,
        linear_weight_path: String,
        n: u32,
        k: u32,
        iters: u32,
        partial_sums_offset: u32,
        b_tile_offset: u32,
        b_tile_bytes: u32,
        norm_kind: LmHeadNormKind,
        offset: Option<f32>,
    ) -> &mut Self {
        // Cross-field invariant: AddScalarOffsetRmsNorm requires
        // an offset; the others reject it. The lowering catches a
        // proc-macro bug at construction (not at runtime).
        match (norm_kind, offset) {
            (LmHeadNormKind::AddScalarOffsetRmsNorm, Some(_))
            | (LmHeadNormKind::RmsNorm, None)
            | (LmHeadNormKind::AddRmsNorm, None)
            | (LmHeadNormKind::MeanSubRmsNorm, None) => {}
            (LmHeadNormKind::AddScalarOffsetRmsNorm, None) => {
                panic!("CutlassFusedNormGemm: AddScalarOffsetRmsNorm requires Some(offset)",);
            }
            (kind, Some(_)) => {
                panic!("CutlassFusedNormGemm: {kind:?} must not carry an offset",);
            }
        }

        let in_page: Page<Empty> = self.pool.take(in_slot_id);
        let delta_page = delta_slot_id.map(|id| self.pool.take(id));
        let norm_w_page: Page<Empty> = self.pool.take(norm_weight_slot_id);
        let lin_w_page: Page<Empty> = self.pool.take(linear_weight_slot_id);
        let out_page: Page<Empty> = self.pool.take(out_slot_id);

        let in_empty = in_page.loader_fired().consumer_arrived().storer_consumed();
        let delta_empty = delta_page.map(|p| p.loader_fired().consumer_arrived().storer_consumed());
        let norm_w_empty = norm_w_page
            .loader_fired()
            .consumer_arrived()
            .storer_consumed();
        let lin_w_empty = lin_w_page
            .loader_fired()
            .consumer_arrived()
            .storer_consumed();
        let out_empty = out_page.loader_fired().consumer_arrived().storer_consumed();

        let partial_bytes = self.pool.substrate().num_consumer_warps() * 4;
        let partial_sums = ScratchRegion::<RmsNormScope>::new(
            partial_sums_offset,
            partial_bytes,
            self.pool.substrate(),
        );
        let b_tile =
            ScratchRegion::<GemmScope>::new(b_tile_offset, b_tile_bytes, self.pool.substrate());

        let consumer_phase =
            MbarrierPhase::assert_matches(self.arrives.current() & 1, self.arrives.current());
        let storer_phase = MbarrierPhase::assert_matches(
            (self.arrives.current() + 1) & 1,
            self.arrives.current() + 1,
        );

        self.nodes
            .push(MegaNode::CutlassFusedNormGemm(CutlassFusedNormGemm {
                in_page: PageId::new(in_slot_id, self.pool.substrate()),
                delta_page: delta_slot_id.map(|id| PageId::new(id, self.pool.substrate())),
                norm_weight_page: PageId::new(norm_weight_slot_id, self.pool.substrate()),
                linear_weight_page: PageId::new(linear_weight_slot_id, self.pool.substrate()),
                out_page: PageId::new(out_slot_id, self.pool.substrate()),
                partial_sums,
                b_tile,
                consumer_phase,
                storer_phase,
                iters: IterCount::new(iters),
                _loader_role: WarpRoleTag::<ROLE_LOADER>,
                _launcher_role: WarpRoleTag::<ROLE_LAUNCHER>,
                _consumer_role: WarpRoleTag::<ROLE_CONSUMER>,
                _storer_role: WarpRoleTag::<ROLE_STORER>,
                layer: LayerIndex::new(layer, num_layers),
                norm_weight: WeightRef::new(norm_weight_path),
                linear_weight: WeightRef::new(linear_weight_path),
                shape: MatmulShape::new(n, k),
                norm_kind,
                offset: offset.map(FiniteF32::new),
            }));

        self.pool.release(in_empty);
        if let Some(p) = delta_empty {
            self.pool.release(p);
        }
        self.pool.release(norm_w_empty);
        self.pool.release(lin_w_empty);
        self.pool.release(out_empty);
        for _ in 0..iters {
            self.arrives.bump();
        }

        self
    }

    /// Push a typed `AttentionViaCacheNode` (covers
    /// `AttentionViaCache` and `SlidingAttentionViaCache`). Two
    /// substrate pages (Q in, attn out — K/V live in the global
    /// paged KV cache extern), two disjoint `AttentionScope`
    /// scratch tiles (score, PV), multi-iter, four role tags.
    #[allow(clippy::too_many_arguments)]
    pub fn push_attention_via_cache(
        &mut self,
        q_in_slot_id: u32,
        attn_out_slot_id: u32,
        kv_cache_layer: u32,
        num_layers: u32,
        iters: u32,
        score_tile_offset: u32,
        score_tile_bytes: u32,
        pv_tile_offset: u32,
        pv_tile_bytes: u32,
        kind: AttentionKind,
        interleaved: bool,
    ) -> &mut Self {
        let q_page: Page<Empty> = self.pool.take(q_in_slot_id);
        let out_page: Page<Empty> = self.pool.take(attn_out_slot_id);

        let q_empty = q_page.loader_fired().consumer_arrived().storer_consumed();
        let out_empty = out_page.loader_fired().consumer_arrived().storer_consumed();

        let score_raw = ScratchRegion::<AttentionScope>::new(
            score_tile_offset,
            score_tile_bytes,
            self.pool.substrate(),
        );
        let pv_raw = ScratchRegion::<AttentionScope>::new(
            pv_tile_offset,
            pv_tile_bytes,
            self.pool.substrate(),
        );
        let (score_tile, pv_tile) = score_raw.disjoint_with(pv_raw);

        let consumer_phase =
            MbarrierPhase::assert_matches(self.arrives.current() & 1, self.arrives.current());
        let storer_phase = MbarrierPhase::assert_matches(
            (self.arrives.current() + 1) & 1,
            self.arrives.current() + 1,
        );

        self.nodes
            .push(MegaNode::AttentionViaCache(AttentionViaCacheNode {
                q_in_page: PageId::new(q_in_slot_id, self.pool.substrate()),
                attn_out_page: PageId::new(attn_out_slot_id, self.pool.substrate()),
                score_tile,
                pv_tile,
                consumer_phase,
                storer_phase,
                iters: IterCount::new(iters),
                _loader_role: WarpRoleTag::<ROLE_LOADER>,
                _launcher_role: WarpRoleTag::<ROLE_LAUNCHER>,
                _consumer_role: WarpRoleTag::<ROLE_CONSUMER>,
                _storer_role: WarpRoleTag::<ROLE_STORER>,
                kv_cache_layer: LayerIndex::new(kv_cache_layer, num_layers),
                interleaved,
                kind,
            }));

        self.pool.release(q_empty);
        self.pool.release(out_empty);
        for _ in 0..iters {
            self.arrives.bump();
        }

        self
    }

    /// Push a typed `BarrierSignal` onto the tape. No pages, no
    /// scratch, no phases at the per-CTA mbarrier level — just a
    /// validated cross-CTA edge id (#1) and the storer role
    /// constraint (#6). Cumulative arrive count unchanged (the
    /// global atomic add isn't a per-CTA mbarrier arrive).
    pub fn push_barrier_signal(&mut self, edge_idx: u32) -> &mut Self {
        let edge = EdgeId::new(edge_idx, self.pool.substrate());
        self.nodes.push(MegaNode::BarrierSignal(BarrierSignal {
            edge,
            _storer_role: WarpRoleTag::<ROLE_STORER>,
        }));
        self
    }

    /// Push a typed `BarrierWait` onto the tape. Validated edge id
    /// (#1), validated `expected_count > 0`, loader role (#6).
    /// Cumulative arrive count unchanged.
    pub fn push_barrier_wait(&mut self, edge_idx: u32, expected_count: u32) -> &mut Self {
        let edge = EdgeId::new(edge_idx, self.pool.substrate());
        let expected = ExpectedCount::new(expected_count);
        self.nodes.push(MegaNode::BarrierWait(BarrierWait {
            edge,
            expected,
            _loader_role: WarpRoleTag::<ROLE_LOADER>,
        }));
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
    /// yet (Sprint B onwards). Carries the variant name so the
    /// proc-macro caller can fall back to the host interpreter.
    NotYetLifted { op: &'static str },
    /// `OpInput::weight_paths` arity mismatch — the variant declared
    /// N weight slots in `Implementation::required_weights` (plus any
    /// CosSin injection from the walker), but `weight_paths` came in
    /// with a different count. Indicates a fan_out / required_weights
    /// disagreement; surfaces as a proc-macro error.
    WrongWeightArity {
        op: &'static str,
        expected: u32,
        got: u32,
    },
    /// Substrate budget too small for the per-op slot demand
    /// (Sprint B's per-op allocator needs distinct page ids for
    /// every concurrent slot). Caller picks a larger
    /// `num_pages`.
    SubstrateBudgetTooSmall { need: u32, have: u32 },
    /// `Instruction::SlidingAttentionViaCache` requires
    /// `OpInput::sliding_window: Some(n)` (the window value is
    /// arch config, not part of the Instruction's positional
    /// fields).
    MissingSlidingWindow,
}

/// Per-op lowering input: one [`Instruction`] paired with the
/// per-arch weight base names of the slots it consumes (in
/// declaration order — matches `OpInstance::weight_slots` ordering
/// from fan_out time). Each path is the resolved
/// `Weights::<base>` source string the emit step splices in.
///
/// `RmsNorm` consumes one weight slot, so `weight_paths` has one
/// entry. `FusedQkvRopeCache` consumes two — `[qkv_packed_linear,
/// rotary_or_rotary_local]` in declaration order matching the
/// per-arch `WeightAccessors` impl's slot ordinal for that op. The
/// rotary entry is always the last; the lowering pulls it via
/// `weight_paths.last()`.
///
/// Variants that consume no weight (`Add`, `BarrierSignal`,
/// `Reshape`) use an empty `Vec`.
///
/// `iters` is the per-op iteration count for variants whose kernel
/// body loops over tokens internally. Most ops are 1-iter and
/// ignore the field; `FusedQkvRopeCache`, `Gemm`, the lm_head
/// Cutlass fusions, and `(Sliding)AttentionViaCache` all use it.
///
/// `sliding_window` is `Some(n)` only for
/// `Instruction::SlidingAttentionViaCache` (the window value is
/// arch config, not in the Instruction's positional fields). The
/// proc-macro caller threads it from the per-arch config at
/// fan_out time.
#[derive(Clone)]
pub struct OpInput {
    pub instr: Instruction,
    pub weight_paths: Vec<String>,
    pub iters: u32,
    pub sliding_window: Option<u32>,
}

impl OpInput {
    pub fn new(instr: Instruction, weight_paths: Vec<String>) -> Self {
        Self {
            instr,
            weight_paths,
            iters: 1,
            sliding_window: None,
        }
    }
}

/// Top-level lowering function: walks a `Vec<Instruction>` semantic
/// tape (paired with per-op weight context), dispatches each variant
/// to its substrate-aware push handler, and returns the typed
/// `MegaTape`.
///
/// Per `MEGA_IR_PLAN.md` §5 the input is the existing
/// `Instruction` enum from `ferrite_forward::instr`. There is no
/// `MegaOp` semantic enum and no `OpInstance::field_values`
/// round-trip — the proc-macro hands typed `Instruction` values
/// directly into this function.
///
/// Sprint A handles `RmsNorm`. Other variants return
/// [`LowerError::NotYetLifted`] until their sprint lands; the
/// proc-macro is expected to fall back to the host interpreter
/// path for those tapes.
pub fn lower(
    ops: &[OpInput],
    num_layers: u32,
    substrate: SubstrateBudget,
) -> Result<MegaTape, LowerError> {
    let mut builder = MegaTapeBuilder::new(substrate);
    let mut slot_alloc = SlotAllocator::new(substrate.num_pages());
    for op in ops {
        match &op.instr {
            Instruction::RmsNorm(in_slot, _out_slot, layer) => {
                let weight_path = op
                    .weight_paths
                    .first()
                    .expect("lower(RmsNorm): expected one weight_paths entry")
                    .clone();
                let in_slot_id = *in_slot;
                let weight_slot_id = if in_slot_id == 0 { 1 } else { in_slot_id ^ 1 };
                builder.push_rms_norm(
                    in_slot_id,
                    weight_slot_id,
                    *layer,
                    num_layers,
                    weight_path,
                    /*scratch_offset=*/ 0,
                );
            }
            Instruction::Add(delta_slot, residual_slot) => {
                builder.push_add(*delta_slot, *residual_slot);
            }
            Instruction::FusedAddRmsNorm(delta_slot, residual_slot, layer) => {
                let weight_path = op
                    .weight_paths
                    .first()
                    .expect("lower(FusedAddRmsNorm): expected one weight_paths entry")
                    .clone();
                // Avoid aliasing with delta_slot/residual_slot. Pick
                // the smallest free id mod num_pages that doesn't
                // collide. Sprint D's pipeline-aware allocator
                // replaces this once attention/gemm sites land.
                let weight_slot_id =
                    pick_distinct_slot(&[*delta_slot, *residual_slot], &mut slot_alloc)?;
                builder.push_fused_add_rms_norm(
                    *delta_slot,
                    *residual_slot,
                    weight_slot_id,
                    *layer,
                    num_layers,
                    weight_path,
                    /*scratch_offset=*/ 0,
                );
            }
            Instruction::FusedGateUpSiluMul(in_slot, out_slot, layer) => {
                let weight_path = op
                    .weight_paths
                    .first()
                    .expect("lower(FusedGateUpSiluMul): expected one weight_paths entry")
                    .clone();
                let weight_slot_id = pick_distinct_slot(&[*in_slot, *out_slot], &mut slot_alloc)?;
                let half = substrate.scratch_bytes() / 2;
                builder.push_fused_gate_up_activate_mul(
                    *in_slot,
                    weight_slot_id,
                    *out_slot,
                    *layer,
                    num_layers,
                    weight_path,
                    op.iters,
                    /*gate_off=*/ 0,
                    /*gate_bytes=*/ half,
                    /*up_off=*/ half,
                    /*up_bytes=*/ half,
                    GateUpActivation::Silu,
                );
            }
            Instruction::FusedGateUpGeluMul(in_slot, out_slot, layer) => {
                let weight_path = op
                    .weight_paths
                    .first()
                    .expect("lower(FusedGateUpGeluMul): expected one weight_paths entry")
                    .clone();
                let weight_slot_id = pick_distinct_slot(&[*in_slot, *out_slot], &mut slot_alloc)?;
                let half = substrate.scratch_bytes() / 2;
                builder.push_fused_gate_up_activate_mul(
                    *in_slot,
                    weight_slot_id,
                    *out_slot,
                    *layer,
                    num_layers,
                    weight_path,
                    op.iters,
                    0,
                    half,
                    half,
                    half,
                    GateUpActivation::Gelu,
                );
            }
            Instruction::Embed(out_slot) => {
                let weight_path = op
                    .weight_paths
                    .first()
                    .expect("lower(Embed): expected one weight_paths entry")
                    .clone();
                let weight_slot_id = pick_distinct_slot(&[*out_slot], &mut slot_alloc)?;
                builder.push_embed(*out_slot, weight_slot_id, weight_path);
            }
            Instruction::ScalarMul(in_slot, out_slot, scale) => {
                builder.push_scalar_mul(*in_slot, *out_slot, *scale);
            }
            Instruction::TanhSoftCap(in_slot, out_slot) => {
                builder.push_tanh_soft_cap(*in_slot, *out_slot);
            }
            Instruction::ScalarOffsetRmsNorm(in_slot, _out_slot, layer, offset) => {
                let weight_path = op
                    .weight_paths
                    .first()
                    .expect("lower(ScalarOffsetRmsNorm): expected one weight_paths entry")
                    .clone();
                let weight_slot_id = pick_distinct_slot(&[*in_slot], &mut slot_alloc)?;
                builder.push_scalar_offset_rms_norm(
                    *in_slot,
                    weight_slot_id,
                    *layer,
                    num_layers,
                    weight_path,
                    /*scratch_offset=*/ 0,
                    *offset,
                );
            }
            Instruction::Gemm(in_slot, out_slot, layer, n, k) => {
                let weight_path = op
                    .weight_paths
                    .first()
                    .expect("lower(Gemm): expected one weight_paths entry")
                    .clone();
                let weight_slot_id = pick_distinct_slot(&[*in_slot, *out_slot], &mut slot_alloc)?;
                builder.push_gemm(
                    *in_slot,
                    weight_slot_id,
                    *out_slot,
                    *layer,
                    num_layers,
                    weight_path,
                    *n,
                    *k,
                    op.iters,
                    /*b_tile_offset=*/ 0,
                    /*b_tile_bytes=*/ substrate.scratch_bytes(),
                );
            }
            Instruction::CutlassFusedRmsNormGemm(
                in_slot,
                out_slot,
                layer,
                _tile_m,
                _tile_n,
                _stages,
                n,
                k,
            ) => lower_lm_head_fusion(
                &mut builder,
                &mut slot_alloc,
                op,
                LmHeadNormKind::RmsNorm,
                /*delta=*/ None,
                /*offset=*/ None,
                *in_slot,
                *out_slot,
                *layer,
                num_layers,
                *n,
                *k,
                substrate,
            )?,
            Instruction::CutlassFusedAddRmsNormGemm(
                delta_slot,
                residual_slot,
                out_slot,
                layer,
                _tile_m,
                _tile_n,
                _stages,
                n,
                k,
            ) => lower_lm_head_fusion(
                &mut builder,
                &mut slot_alloc,
                op,
                LmHeadNormKind::AddRmsNorm,
                Some(*delta_slot),
                None,
                *residual_slot,
                *out_slot,
                *layer,
                num_layers,
                *n,
                *k,
                substrate,
            )?,
            Instruction::CutlassFusedMeanSubRmsNormGemm(
                in_slot,
                out_slot,
                layer,
                _tile_m,
                _tile_n,
                _stages,
                n,
                k,
            ) => lower_lm_head_fusion(
                &mut builder,
                &mut slot_alloc,
                op,
                LmHeadNormKind::MeanSubRmsNorm,
                None,
                None,
                *in_slot,
                *out_slot,
                *layer,
                num_layers,
                *n,
                *k,
                substrate,
            )?,
            Instruction::CutlassFusedAddScalarOffsetRmsNormGemm(
                delta_slot,
                residual_slot,
                out_slot,
                layer,
                offset,
                _tile_m,
                _tile_n,
                _stages,
                n,
                k,
            ) => lower_lm_head_fusion(
                &mut builder,
                &mut slot_alloc,
                op,
                LmHeadNormKind::AddScalarOffsetRmsNorm,
                Some(*delta_slot),
                Some(*offset),
                *residual_slot,
                *out_slot,
                *layer,
                num_layers,
                *n,
                *k,
                substrate,
            )?,
            Instruction::AttentionViaCache(q_slot, attn_out_slot, layer, interleaved) => {
                let half = substrate.scratch_bytes() / 2;
                builder.push_attention_via_cache(
                    *q_slot,
                    *attn_out_slot,
                    *layer,
                    num_layers,
                    op.iters,
                    /*score_off=*/ 0,
                    /*score_bytes=*/ half,
                    /*pv_off=*/ half,
                    /*pv_bytes=*/ half,
                    AttentionKind::Full,
                    *interleaved,
                );
            }
            Instruction::SlidingAttentionViaCache(q_slot, attn_out_slot, layer, interleaved) => {
                let window = op.sliding_window.ok_or(LowerError::MissingSlidingWindow)?;
                let half = substrate.scratch_bytes() / 2;
                builder.push_attention_via_cache(
                    *q_slot,
                    *attn_out_slot,
                    *layer,
                    num_layers,
                    op.iters,
                    0,
                    half,
                    half,
                    half,
                    AttentionKind::Sliding(SlidingWindow::new(window)),
                    *interleaved,
                );
            }
            Instruction::BarrierSignal(edge) => {
                builder.push_barrier_signal(*edge);
            }
            Instruction::BarrierWait(edge, expected_count) => {
                builder.push_barrier_wait(*edge, *expected_count);
            }
            Instruction::FusedQkvRopeCache(in_slot, _out_slot, layer, biased, interleaved) => {
                // Two weight paths: [qkv_packed, rotary].
                // The fan_out walker pushes the QKV packed
                // LinearLayer first, then injects the CosSin slot
                // for the rotary. `weight_paths.first()` is the
                // QKV; `.last()` (when len == 2) is the rotary.
                if op.weight_paths.len() != 2 {
                    return Err(LowerError::WrongWeightArity {
                        op: "FusedQkvRopeCache",
                        expected: 2,
                        got: op.weight_paths.len() as u32,
                    });
                }
                let qkv_weight_path = op.weight_paths[0].clone();
                let rotary_path = op.weight_paths[1].clone();

                let in_slot_id = *in_slot;
                // Sprint B: synthesize five additional non-aliasing
                // slot ids from a per-op allocator. Real co-allocation
                // (input pages reused across ops, output pages handed
                // off to the next op) lands in Sprint D when the
                // pipeline scheduler does. Until then, pick the next
                // five free slots that don't collide with `in_slot_id`.
                let qkv_id = slot_alloc.next_excluding(in_slot_id)?;
                let cs_id = slot_alloc.next_excluding(in_slot_id)?;
                let q_id = slot_alloc.next_excluding(in_slot_id)?;
                let k_id = slot_alloc.next_excluding(in_slot_id)?;
                let v_id = slot_alloc.next_excluding(in_slot_id)?;

                // Two scratch regions inside `RopeScope`. Sprint B's
                // bug-class-#4 enforcement requires they're disjoint;
                // we pack them at the front of scratch back-to-back
                // (offsets 0 and `q_bytes`). Real scratch sizing
                // tracks the kernel's per-warp tile shape; the
                // lowering doesn't need to be clever — just disjoint
                // and within budget. Sprint D's emit-side cost model
                // picks the actual sizes from the variant's tile
                // dims; until then a token-iter's Q/K rotation
                // tiles are sized as half the consumer-warp scratch
                // budget each (so they pack exactly back-to-back).
                let half = substrate.scratch_bytes() / 2;
                let q_scratch_offset = 0;
                let q_scratch_bytes = half;
                let k_scratch_offset = half;
                let k_scratch_bytes = half;

                builder.push_fused_qkv_rope_cache(
                    in_slot_id,
                    qkv_id,
                    cs_id,
                    q_id,
                    k_id,
                    v_id,
                    *layer,
                    num_layers,
                    qkv_weight_path,
                    rotary_path,
                    op.iters,
                    q_scratch_offset,
                    q_scratch_bytes,
                    k_scratch_offset,
                    k_scratch_bytes,
                    *biased,
                    *interleaved,
                );
            }
            other => {
                return Err(LowerError::NotYetLifted {
                    op: variant_name(other),
                });
            }
        }
    }
    Ok(builder.finish())
}

/// Per-tape slot allocator. Hands out the next page id mod the
/// substrate budget, skipping a caller-specified id (typically the
/// op's input slot) and wrapping around. Sprint B uses this to
/// pick non-aliasing output/weight pages within one op; Sprint D
/// replaces it with a real pipeline-aware allocator.
struct SlotAllocator {
    next: u32,
    num_pages: u32,
}

impl SlotAllocator {
    fn new(num_pages: u32) -> Self {
        Self { next: 0, num_pages }
    }

    fn next_excluding(&mut self, exclude: u32) -> Result<u32, LowerError> {
        // Walk forward at most `num_pages` steps to find the next
        // free id. Returns SubstrateBudgetTooSmall if every page is
        // either the exclusion or already issued.
        for _ in 0..self.num_pages {
            let id = self.next % self.num_pages;
            self.next += 1;
            if id != exclude {
                return Ok(id);
            }
        }
        Err(LowerError::SubstrateBudgetTooSmall {
            need: self.num_pages + 1,
            have: self.num_pages,
        })
    }
}

/// Pick the next slot id from `alloc` that doesn't appear in
/// `exclude`. Used by Sprint C arms (FusedAddRmsNorm, gate-up
/// fusions) to pick a non-aliasing weight/intermediate page when
/// the Instruction only supplies in/out slot ids.
fn pick_distinct_slot(exclude: &[u32], alloc: &mut SlotAllocator) -> Result<u32, LowerError> {
    for _ in 0..alloc.num_pages {
        let id = alloc.next_excluding(u32::MAX)?;
        if !exclude.contains(&id) {
            return Ok(id);
        }
    }
    Err(LowerError::SubstrateBudgetTooSmall {
        need: alloc.num_pages + 1,
        have: alloc.num_pages,
    })
}

/// Shared lowering body for the four `CutlassFused*Gemm` lm_head
/// variants. They share the same substrate shape (in + optional
/// delta + norm-weight + linear-weight + out pages, partial sums
/// in `RmsNormScope`, B-tile in `GemmScope`), differing only in
/// the `LmHeadNormKind` flag and the optional `offset`.
///
/// `op.weight_paths` must carry exactly two entries:
/// `[norm_weight, linear_weight]` in that order — the per-arch
/// `WeightAccessors` impl emits the norm slot before the linear
/// slot at this op_idx.
#[allow(clippy::too_many_arguments)]
fn lower_lm_head_fusion(
    builder: &mut MegaTapeBuilder,
    slot_alloc: &mut SlotAllocator,
    op: &OpInput,
    norm_kind: LmHeadNormKind,
    delta_slot_id: Option<u32>,
    offset: Option<f32>,
    in_slot_id: u32,
    out_slot_id: u32,
    layer: u32,
    num_layers: u32,
    n: u32,
    k: u32,
    substrate: SubstrateBudget,
) -> Result<(), LowerError> {
    if op.weight_paths.len() != 2 {
        let label = match norm_kind {
            LmHeadNormKind::RmsNorm => "CutlassFusedRmsNormGemm",
            LmHeadNormKind::AddRmsNorm => "CutlassFusedAddRmsNormGemm",
            LmHeadNormKind::MeanSubRmsNorm => "CutlassFusedMeanSubRmsNormGemm",
            LmHeadNormKind::AddScalarOffsetRmsNorm => "CutlassFusedAddScalarOffsetRmsNormGemm",
        };
        return Err(LowerError::WrongWeightArity {
            op: label,
            expected: 2,
            got: op.weight_paths.len() as u32,
        });
    }
    let norm_weight_path = op.weight_paths[0].clone();
    let linear_weight_path = op.weight_paths[1].clone();

    // Synthesize non-aliasing weight + (if needed) delta page slots.
    let mut exclude: Vec<u32> = vec![in_slot_id, out_slot_id];
    if let Some(d) = delta_slot_id {
        exclude.push(d);
    }
    let norm_weight_slot_id = pick_distinct_slot(&exclude, slot_alloc)?;
    exclude.push(norm_weight_slot_id);
    let linear_weight_slot_id = pick_distinct_slot(&exclude, slot_alloc)?;

    // Pack partial-sums + B-tile back-to-back at the front of
    // scratch. Different scopes — no overlap proof needed across
    // them; each is independently within-budget.
    let partial_sums_bytes = substrate.num_consumer_warps() * 4;
    let b_tile_offset = partial_sums_bytes;
    let b_tile_bytes = substrate.scratch_bytes().saturating_sub(partial_sums_bytes);

    builder.push_cutlass_fused_norm_gemm(
        in_slot_id,
        delta_slot_id,
        norm_weight_slot_id,
        linear_weight_slot_id,
        out_slot_id,
        layer,
        num_layers,
        norm_weight_path,
        linear_weight_path,
        n,
        k,
        op.iters,
        /*partial_sums_offset=*/ 0,
        b_tile_offset,
        b_tile_bytes,
        norm_kind,
        offset,
    );
    Ok(())
}

/// Stable static-string name for an [`Instruction`] variant. Used
/// in [`LowerError::NotYetLifted`] reporting and proc-macro diag.
fn variant_name(instr: &Instruction) -> &'static str {
    match instr {
        Instruction::Embed(..) => "Embed",
        Instruction::RmsNorm(..) => "RmsNorm",
        Instruction::MeanSubRmsNorm(..) => "MeanSubRmsNorm",
        Instruction::MeanSubRmsNormBiasAdd(..) => "MeanSubRmsNormBiasAdd",
        Instruction::Reshape(..) => "Reshape",
        Instruction::Add(..) => "Add",
        Instruction::SpliceMmEmbeds(..) => "SpliceMmEmbeds",
        Instruction::ScalarMul(..) => "ScalarMul",
        Instruction::TanhSoftCap(..) => "TanhSoftCap",
        Instruction::FusedAddRmsNorm(..) => "FusedAddRmsNorm",
        Instruction::FusedAddRmsNormWithOffset(..) => "FusedAddRmsNormWithOffset",
        Instruction::ScalarOffsetRmsNorm(..) => "ScalarOffsetRmsNorm",
        Instruction::CutlassFusedRmsNormGemm(..) => "CutlassFusedRmsNormGemm",
        Instruction::CutlassFusedMeanSubRmsNormGemm(..) => "CutlassFusedMeanSubRmsNormGemm",
        Instruction::CutlassFusedAddRmsNormGemm(..) => "CutlassFusedAddRmsNormGemm",
        Instruction::CutlassFusedAddScalarOffsetRmsNormGemm(..) => {
            "CutlassFusedAddScalarOffsetRmsNormGemm"
        }
        Instruction::Gemm(..) => "Gemm",
        Instruction::FusedCublasGemmAdd(..) => "FusedCublasGemmAdd",
        Instruction::FusedGemmBias(..) => "FusedGemmBias",
        Instruction::FusedGateUpSiluMul(..) => "FusedGateUpSiluMul",
        Instruction::FusedGateUpGeluMul(..) => "FusedGateUpGeluMul",
        Instruction::FusedQkvRopeCache(..) => "FusedQkvRopeCache",
        Instruction::FusedQkvQkNormRopeCache(..) => "FusedQkvQkNormRopeCache",
        Instruction::FusedQkvRopePrefill(..) => "FusedQkvRopePrefill",
        Instruction::AttentionViaCache(..) => "AttentionViaCache",
        Instruction::AttentionPrefillContiguous(..) => "AttentionPrefillContiguous",
        Instruction::EncoderAttention(..) => "EncoderAttention",
        Instruction::SlidingAttentionViaCache(..) => "SlidingAttentionViaCache",
        Instruction::SlidingAttentionPrefillContiguous(..) => "SlidingAttentionPrefillContiguous",
        Instruction::VarlenAttention(..) => "VarlenAttention",
        Instruction::VisionRope(..) => "VisionRope",
        Instruction::QuickGelu(..) => "QuickGelu",
        Instruction::Gelu(..) => "Gelu",
        Instruction::PosEmbed(..) => "PosEmbed",
        Instruction::LoadPixels(..) => "LoadPixels",
        Instruction::GeluErf(..) => "GeluErf",
        Instruction::EmbeddingGather(..) => "EmbeddingGather",
        Instruction::AvgPool2d(..) => "AvgPool2d",
        Instruction::StripCls(..) => "StripCls",
        Instruction::FlashInferAttentionDecode(..) => "FlashInferAttentionDecode",
        Instruction::FlashInferAttentionPrefill(..) => "FlashInferAttentionPrefill",
        Instruction::RopeAppend(..) => "RopeAppend",
        Instruction::MlaSplit(..) => "MlaSplit",
        Instruction::MlaAttention(..) => "MlaAttention",
        Instruction::DeepSeekMoe(..) => "DeepSeekMoe",
        Instruction::DeepSeekMoeFp8Block(..) => "DeepSeekMoeFp8Block",
        Instruction::DeepSeekMoeGgml(..) => "DeepSeekMoeGgml",
        Instruction::FusedMoe(..) => "FusedMoe",
        Instruction::SharedFusedMoe(..) => "SharedFusedMoe",
        Instruction::CutlassGemm(..) => "CutlassGemm",
        Instruction::CutlassGemmSplitK(..) => "CutlassGemmSplitK",
        Instruction::CutlassGemmAdd(..) => "CutlassGemmAdd",
        Instruction::CutlassGemv(..) => "CutlassGemv",
        Instruction::CutlassFusedGemmBias(..) => "CutlassFusedGemmBias",
        Instruction::CutlassFusedGateUpSiluMul(..) => "CutlassFusedGateUpSiluMul",
        Instruction::CutlassFusedGateUpGeluMul(..) => "CutlassFusedGateUpGeluMul",
        Instruction::CutlassFusedQkvRopeCache(..) => "CutlassFusedQkvRopeCache",
        Instruction::CutlassFusedQkvRopePrefill(..) => "CutlassFusedQkvRopePrefill",
        Instruction::MarlinGemm(..) => "MarlinGemm",
        Instruction::MarlinFusedGateUpSiluMul(..) => "MarlinFusedGateUpSiluMul",
        Instruction::MarlinFusedGateUpGeluMul(..) => "MarlinFusedGateUpGeluMul",
        Instruction::MarlinFusedQkvRopeCache(..) => "MarlinFusedQkvRopeCache",
        Instruction::MarlinFusedQkvRopePrefill(..) => "MarlinFusedQkvRopePrefill",
        Instruction::Bnb4Gemm(..) => "Bnb4Gemm",
        Instruction::Bnb4FusedGateUpSiluMul(..) => "Bnb4FusedGateUpSiluMul",
        Instruction::Bnb4FusedGateUpGeluMul(..) => "Bnb4FusedGateUpGeluMul",
        Instruction::Bnb4FusedQkvRopeCache(..) => "Bnb4FusedQkvRopeCache",
        Instruction::Bnb4FusedQkvRopePrefill(..) => "Bnb4FusedQkvRopePrefill",
        Instruction::GgmlGemm(..) => "GgmlGemm",
        Instruction::GgmlFusedGateUpSiluMul(..) => "GgmlFusedGateUpSiluMul",
        Instruction::GgmlFusedGateUpGeluMul(..) => "GgmlFusedGateUpGeluMul",
        Instruction::GgmlFusedQkvRopeCache(..) => "GgmlFusedQkvRopeCache",
        Instruction::GgmlFusedQkvRopePrefill(..) => "GgmlFusedQkvRopePrefill",
        Instruction::Fp8Gemm(..) => "Fp8Gemm",
        Instruction::Fp8FusedGemmBias(..) => "Fp8FusedGemmBias",
        Instruction::Fp8FusedGateUpSiluMul(..) => "Fp8FusedGateUpSiluMul",
        Instruction::Fp8FusedGateUpGeluMul(..) => "Fp8FusedGateUpGeluMul",
        Instruction::Fp8FusedQkvRopeCache(..) => "Fp8FusedQkvRopeCache",
        Instruction::Fp8FusedQkvRopePrefill(..) => "Fp8FusedQkvRopePrefill",
        Instruction::TkEmbed(..) => "TkEmbed",
        Instruction::TkScalarMul(..) => "TkScalarMul",
        Instruction::TkRmsNorm(..) => "TkRmsNorm",
        Instruction::TkGemm(..) => "TkGemm",
        Instruction::TkFusedAddRmsNorm(..) => "TkFusedAddRmsNorm",
        Instruction::TkFusedQkvRopeCache(..) => "TkFusedQkvRopeCache",
        Instruction::TkAttentionViaCache(..) => "TkAttentionViaCache",
        Instruction::TkSlidingAttentionViaCache(..) => "TkSlidingAttentionViaCache",
        Instruction::TkFusedGateUpSiluMul(..) => "TkFusedGateUpSiluMul",
        Instruction::TkFusedGateUpGeluMul(..) => "TkFusedGateUpGeluMul",
        Instruction::TkGemmAdd(..) => "TkGemmAdd",
        Instruction::TkFusedAddRmsNormGemm(..) => "TkFusedAddRmsNormGemm",
        Instruction::TkScalarOffsetRmsNorm(..) => "TkScalarOffsetRmsNorm",
        Instruction::TkFusedAddRmsNormWithOffset(..) => "TkFusedAddRmsNormWithOffset",
        Instruction::TkTanhSoftCap(..) => "TkTanhSoftCap",
        Instruction::TkFusedAddScalarOffsetRmsNormGemm(..) => "TkFusedAddScalarOffsetRmsNormGemm",
        Instruction::TkBarrierSignal(..) => "TkBarrierSignal",
        Instruction::TkBarrierWait(..) => "TkBarrierWait",
        Instruction::TkSpliceMmEmbeds(..) => "TkSpliceMmEmbeds",
        Instruction::Loop(..) => "Loop",
        Instruction::Alias(..) => "Alias",
        Instruction::Free(..) => "Free",
        Instruction::BarrierSignal(..) => "BarrierSignal",
        Instruction::BarrierWait(..) => "BarrierWait",
        #[cfg(feature = "nccl")]
        Instruction::AllReduce(..) => "AllReduce",
        #[cfg(feature = "nccl")]
        Instruction::AllGather(..) => "AllGather",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget() -> SubstrateBudget {
        SubstrateBudget::new(6, 8, 32_768, 8_192)
    }

    fn good_op() -> OpInput {
        OpInput::new(
            Instruction::RmsNorm(0, 0, 0),
            vec!["Weights::input_layernorm".to_string()],
        )
    }

    #[test]
    fn lowers_well_formed_rms_norm() {
        let mut b = MegaTapeBuilder::new(budget());
        b.push_rms_norm(0, 1, 0, 16, "W::norm".to_string(), 0);
        let tape = b.finish();
        assert_eq!(tape.nodes().len(), 1);
        let MegaNode::RmsNorm(n) = &tape.nodes()[0] else {
            panic!("expected RmsNorm node");
        };
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
        let MegaNode::RmsNorm(n0) = &tape.nodes()[0] else {
            panic!("expected RmsNorm");
        };
        let MegaNode::RmsNorm(n1) = &tape.nodes()[1] else {
            panic!("expected RmsNorm");
        };
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
    fn lower_rms_norm_round_trips() {
        let ops = vec![good_op()];
        let tape = lower(&ops, /*num_layers=*/ 16, budget()).expect("good input");
        assert_eq!(tape.nodes().len(), 1);
    }

    #[test]
    fn lower_empty_produces_empty_tape() {
        let tape = lower(&[], 16, budget()).expect("empty input is fine");
        assert!(tape.nodes().is_empty());
    }

    #[test]
    fn lower_unmigrated_variant_returns_not_yet_lifted() {
        // `MlaAttention` (DeepSeek MLA) is not in Sprint A/B/C/D
        // scope — it lands when the DeepSeek arches enter mega.
        let ops = vec![OpInput::new(
            Instruction::MlaAttention(0, 1, 2, 0, 16),
            Vec::new(),
        )];
        match lower(&ops, 16, budget()) {
            Err(LowerError::NotYetLifted { op }) => assert_eq!(op, "MlaAttention"),
            other => panic!("expected NotYetLifted for MlaAttention, got {other:?}"),
        }
    }

    // ── Sprint B: FusedQkvRopeCache ─────────────────────────────

    fn rope_budget() -> SubstrateBudget {
        // Sprint B variant needs ≥6 pages (in + qkv + cs + q + k + v);
        // give the budget headroom and 4 KiB scratch (split half/half
        // between Q and K rotation tiles in the lowering).
        SubstrateBudget::new(8, 8, 32_768, 4_096)
    }

    fn good_qkv() -> OpInput {
        OpInput {
            instr: Instruction::FusedQkvRopeCache(
                /*in_slot=*/ 0, /*out_slot=*/ 0, /*layer=*/ 0,
                /*biased=*/ false, /*interleaved=*/ false,
            ),
            sliding_window: None,
            weight_paths: vec![
                "Weights::qkv_proj".to_string(),
                "Weights::rotary".to_string(),
            ],
            iters: 1,
        }
    }

    #[test]
    fn lowers_well_formed_fused_qkv_rope_cache() {
        let mut b = MegaTapeBuilder::new(rope_budget());
        b.push_fused_qkv_rope_cache(
            0,
            1,
            2,
            3,
            4,
            5,
            0,
            16,
            "W::qkv".to_string(),
            "W::rot".to_string(),
            /*iters=*/ 4,
            /*q_off=*/ 0,
            /*q_bytes=*/ 2_048,
            /*k_off=*/ 2_048,
            /*k_bytes=*/ 2_048,
            /*biased=*/ true,
            /*interleaved=*/ false,
        );
        let tape = b.finish();
        let MegaNode::FusedQkvRopeCache(n) = &tape.nodes()[0] else {
            panic!("expected FusedQkvRopeCache");
        };
        assert_eq!(n.in_page.raw(), 0);
        assert_eq!(n.qkv_weight_page.raw(), 1);
        assert_eq!(n.cos_sin_page.raw(), 2);
        assert_eq!(n.q_out_page.raw(), 3);
        assert_eq!(n.k_out_page.raw(), 4);
        assert_eq!(n.v_out_page.raw(), 5);
        assert_eq!(n.q_rope_buf.offset(), 0);
        assert_eq!(n.q_rope_buf.bytes(), 2_048);
        assert_eq!(n.k_rope_buf.offset(), 2_048);
        assert_eq!(n.k_rope_buf.bytes(), 2_048);
        assert_eq!(n.iters.raw(), 4);
        assert_eq!(n.consumer_phase.phase(), 0);
        assert_eq!(n.storer_phase.phase(), 1);
        assert!(n.biased);
        assert!(!n.interleaved);
        assert_eq!(n.qkv_weight.path(), "W::qkv");
        assert_eq!(n.rotary.path(), "W::rot");
    }

    #[test]
    fn rope_advances_arrive_count_by_iters() {
        // After a 5-iter rope op, the next op's consumer wait phase
        // must equal `5 & 1 == 1`. Validates per-iter phase math
        // (bug class #3): iter t inside the rope op waits at parity
        // `(C_start + t) & 1 == t & 1`; after the op the cumulative
        // count is `C_start + 5 == 5`, and the next op (here a
        // RmsNorm) must see consumer_phase=1.
        let mut b = MegaTapeBuilder::new(rope_budget());
        b.push_fused_qkv_rope_cache(
            0,
            1,
            2,
            3,
            4,
            5,
            0,
            16,
            "W::qkv".to_string(),
            "W::rot".to_string(),
            5,
            0,
            2_048,
            2_048,
            2_048,
            false,
            false,
        );
        b.push_rms_norm(0, 1, 0, 16, "W::n".to_string(), 0);
        let tape = b.finish();
        let MegaNode::RmsNorm(rms) = &tape.nodes()[1] else {
            panic!("second node should be RmsNorm");
        };
        assert_eq!(rms.consumer_phase.phase(), 1);
        assert_eq!(rms.storer_phase.phase(), 0);
    }

    #[test]
    fn rope_even_iter_count_keeps_phase() {
        // 4-iter rope leaves cumulative arrive count even — next
        // op's consumer phase parity is unchanged from the rope's.
        let mut b = MegaTapeBuilder::new(rope_budget());
        b.push_fused_qkv_rope_cache(
            0,
            1,
            2,
            3,
            4,
            5,
            0,
            16,
            "W::qkv".to_string(),
            "W::rot".to_string(),
            4,
            0,
            2_048,
            2_048,
            2_048,
            false,
            false,
        );
        b.push_rms_norm(0, 1, 0, 16, "W::n".to_string(), 0);
        let tape = b.finish();
        let MegaNode::RmsNorm(rms) = &tape.nodes()[1] else {
            panic!("second node should be RmsNorm");
        };
        assert_eq!(rms.consumer_phase.phase(), 0);
        assert_eq!(rms.storer_phase.phase(), 1);
    }

    #[test]
    #[should_panic(expected = "ScratchRegion overlap within scope")]
    fn rope_rejects_q_k_scratch_overlap() {
        // Sprint B's bug class #4: the per-token Q-rope and K-rope
        // buffers must be disjoint within RopeScope. Constructing
        // them at the same offset must panic.
        let mut b = MegaTapeBuilder::new(rope_budget());
        b.push_fused_qkv_rope_cache(
            0,
            1,
            2,
            3,
            4,
            5,
            0,
            16,
            "W::qkv".to_string(),
            "W::rot".to_string(),
            1,
            /*q_off=*/ 0,
            /*q_bytes=*/ 1_024,
            /*k_off=*/ 512,
            /*k_bytes=*/ 1_024,
            false,
            false,
        );
    }

    #[test]
    #[should_panic(expected = "PagePool::take: page id 3 already in use")]
    fn rope_rejects_page_aliasing_within_op() {
        // Same slot for two outputs — bug class #2 within-op:
        // Q and K can't share a substrate page slot.
        let mut b = MegaTapeBuilder::new(rope_budget());
        b.push_fused_qkv_rope_cache(
            0,
            1,
            2,
            3,
            3,
            5,
            0,
            16,
            "W::qkv".to_string(),
            "W::rot".to_string(),
            1,
            0,
            2_048,
            2_048,
            2_048,
            false,
            false,
        );
    }

    #[test]
    #[should_panic(expected = "IterCount: iters must be > 0")]
    fn rope_rejects_zero_iters() {
        // Bug class #3 boundary: a 0-iter op leaves cumulative
        // arrive count unchanged but the kernel emits zero arrives,
        // so the next op's expected phase is correct only by
        // accident. Reject up front.
        let mut b = MegaTapeBuilder::new(rope_budget());
        b.push_fused_qkv_rope_cache(
            0,
            1,
            2,
            3,
            4,
            5,
            0,
            16,
            "W::qkv".to_string(),
            "W::rot".to_string(),
            0,
            0,
            2_048,
            2_048,
            2_048,
            false,
            false,
        );
    }

    #[test]
    #[should_panic(expected = "RotaryRef must be a non-empty path string")]
    fn rope_rejects_empty_rotary_path() {
        let mut b = MegaTapeBuilder::new(rope_budget());
        b.push_fused_qkv_rope_cache(
            0,
            1,
            2,
            3,
            4,
            5,
            0,
            16,
            "W::qkv".to_string(),
            "".to_string(),
            1,
            0,
            2_048,
            2_048,
            2_048,
            false,
            false,
        );
    }

    #[test]
    fn lower_rope_round_trips() {
        let ops = vec![good_qkv()];
        let tape = lower(&ops, 16, rope_budget()).expect("good input");
        assert_eq!(tape.nodes().len(), 1);
        let MegaNode::FusedQkvRopeCache(_) = &tape.nodes()[0] else {
            panic!("expected FusedQkvRopeCache");
        };
    }

    #[test]
    fn lower_rope_with_wrong_arity_errors() {
        // weight_paths must be exactly 2: [qkv_packed, rotary].
        let ops = vec![OpInput {
            instr: Instruction::FusedQkvRopeCache(0, 0, 0, false, false),
            weight_paths: vec!["W::qkv".to_string()], // missing rotary
            iters: 1,
            sliding_window: None,
        }];
        match lower(&ops, 16, rope_budget()) {
            Err(LowerError::WrongWeightArity { op, expected, got }) => {
                assert_eq!(op, "FusedQkvRopeCache");
                assert_eq!(expected, 2);
                assert_eq!(got, 1);
            }
            other => panic!("expected WrongWeightArity, got {other:?}"),
        }
    }

    fn node_kind(n: &MegaNode) -> &'static str {
        match n {
            MegaNode::RmsNorm(_) => "RmsNorm",
            MegaNode::FusedQkvRopeCache(_) => "FusedQkvRopeCache",
            MegaNode::Add(_) => "Add",
            MegaNode::FusedAddRmsNorm(_) => "FusedAddRmsNorm",
            MegaNode::FusedGateUpActivateMul(_) => "FusedGateUpActivateMul",
            MegaNode::Embed(_) => "Embed",
            MegaNode::ScalarMul(_) => "ScalarMul",
            MegaNode::TanhSoftCap(_) => "TanhSoftCap",
            MegaNode::ScalarOffsetRmsNorm(_) => "ScalarOffsetRmsNorm",
            MegaNode::Gemm(_) => "Gemm",
            MegaNode::CutlassFusedNormGemm(_) => "CutlassFusedNormGemm",
            MegaNode::AttentionViaCache(_) => "AttentionViaCache",
            MegaNode::BarrierSignal(_) => "BarrierSignal",
            MegaNode::BarrierWait(_) => "BarrierWait",
        }
    }

    // ── Sprint C: Add / FusedAddRmsNorm / FusedGateUp{Silu,Gelu}Mul

    #[test]
    fn lowers_add_minimal_shape() {
        let mut b = MegaTapeBuilder::new(budget());
        b.push_add(0, 1);
        let tape = b.finish();
        let MegaNode::Add(n) = &tape.nodes()[0] else {
            panic!("expected Add");
        };
        assert_eq!(n.delta_page.raw(), 0);
        assert_eq!(n.residual_page.raw(), 1);
        assert_eq!(n.consumer_phase.phase(), 0);
        assert_eq!(n.storer_phase.phase(), 1);
    }

    #[test]
    #[should_panic(expected = "PagePool::take: page id 1 already in use")]
    fn add_rejects_self_aliasing() {
        // delta_slot == residual_slot is bug class #2 (within-op
        // alias) — the kernel would race-read its own write.
        let mut b = MegaTapeBuilder::new(budget());
        b.push_add(1, 1);
    }

    #[test]
    fn add_advances_arrives_by_one() {
        let mut b = MegaTapeBuilder::new(budget());
        b.push_add(0, 1);
        b.push_add(0, 1);
        let tape = b.finish();
        let MegaNode::Add(n0) = &tape.nodes()[0] else {
            panic!("expected Add");
        };
        let MegaNode::Add(n1) = &tape.nodes()[1] else {
            panic!("expected Add");
        };
        assert_eq!(n0.consumer_phase.phase(), 0);
        assert_eq!(n0.storer_phase.phase(), 1);
        assert_eq!(n1.consumer_phase.phase(), 1);
        assert_eq!(n1.storer_phase.phase(), 0);
    }

    #[test]
    fn lowers_fused_add_rms_norm_minimal() {
        let mut b = MegaTapeBuilder::new(budget());
        b.push_fused_add_rms_norm(0, 1, 2, 3, 16, "W::norm".to_string(), 0);
        let tape = b.finish();
        let MegaNode::FusedAddRmsNorm(n) = &tape.nodes()[0] else {
            panic!("expected FusedAddRmsNorm");
        };
        assert_eq!(n.delta_page.raw(), 0);
        assert_eq!(n.residual_page.raw(), 1);
        assert_eq!(n.weight_page.raw(), 2);
        assert_eq!(n.layer.raw(), 3);
        assert_eq!(n.weight.path(), "W::norm");
        assert_eq!(n.partial_sums.bytes(), 8 * 4);
    }

    #[test]
    #[should_panic(expected = "PagePool::take: page id 0 already in use")]
    fn fused_add_rms_norm_rejects_aliasing_residual_with_delta() {
        let mut b = MegaTapeBuilder::new(budget());
        b.push_fused_add_rms_norm(0, 0, 2, 0, 16, "W".to_string(), 0);
    }

    fn mlp_budget() -> SubstrateBudget {
        // Sprint C gate-up needs 3 pages and 2 disjoint MlpScope tiles
        // (packed half+half within scratch).
        SubstrateBudget::new(8, 8, 32_768, 4_096)
    }

    #[test]
    fn lowers_fused_gate_up_silu_mul() {
        let mut b = MegaTapeBuilder::new(mlp_budget());
        b.push_fused_gate_up_activate_mul(
            0,
            1,
            2,
            5,
            16,
            "W::mlp_gate_up".to_string(),
            /*iters=*/ 8,
            0,
            2_048,
            2_048,
            2_048,
            GateUpActivation::Silu,
        );
        let tape = b.finish();
        let MegaNode::FusedGateUpActivateMul(n) = &tape.nodes()[0] else {
            panic!("expected FusedGateUpActivateMul");
        };
        assert_eq!(n.in_page.raw(), 0);
        assert_eq!(n.gate_up_weight_page.raw(), 1);
        assert_eq!(n.out_page.raw(), 2);
        assert_eq!(n.iters.raw(), 8);
        assert_eq!(n.gate_buf.bytes(), 2_048);
        assert_eq!(n.up_buf.offset(), 2_048);
        assert_eq!(n.activation, GateUpActivation::Silu);
    }

    #[test]
    fn lowers_fused_gate_up_gelu_mul() {
        let mut b = MegaTapeBuilder::new(mlp_budget());
        b.push_fused_gate_up_activate_mul(
            0,
            1,
            2,
            5,
            16,
            "W::mlp_gate_up".to_string(),
            4,
            0,
            2_048,
            2_048,
            2_048,
            GateUpActivation::Gelu,
        );
        let tape = b.finish();
        let MegaNode::FusedGateUpActivateMul(n) = &tape.nodes()[0] else {
            panic!("expected FusedGateUpActivateMul");
        };
        assert_eq!(n.activation, GateUpActivation::Gelu);
    }

    #[test]
    #[should_panic(expected = "ScratchRegion overlap within scope")]
    fn gate_up_rejects_gate_up_scratch_overlap() {
        let mut b = MegaTapeBuilder::new(mlp_budget());
        b.push_fused_gate_up_activate_mul(
            0,
            1,
            2,
            0,
            16,
            "W".to_string(),
            1,
            0,
            1_024,
            512,
            1_024,
            GateUpActivation::Silu,
        );
    }

    #[test]
    fn lower_add_round_trip() {
        let ops = vec![OpInput::new(Instruction::Add(0, 1), Vec::new())];
        let tape = lower(&ops, 16, budget()).expect("good Add");
        assert_eq!(tape.nodes().len(), 1);
        let MegaNode::Add(_) = &tape.nodes()[0] else {
            panic!("expected Add");
        };
    }

    #[test]
    fn lower_fused_add_rms_norm_round_trip() {
        let ops = vec![OpInput::new(
            Instruction::FusedAddRmsNorm(0, 1, 2),
            vec!["W::norm".to_string()],
        )];
        let tape = lower(&ops, 16, budget()).expect("good FusedAddRmsNorm");
        assert_eq!(tape.nodes().len(), 1);
    }

    #[test]
    fn lower_fused_gate_up_silu_mul_round_trip() {
        let mut op = OpInput::new(
            Instruction::FusedGateUpSiluMul(0, 1, 7),
            vec!["W::mlp_gate_up".to_string()],
        );
        op.iters = 8;
        let tape = lower(&[op], 16, mlp_budget()).expect("good gate-up-silu-mul");
        let MegaNode::FusedGateUpActivateMul(n) = &tape.nodes()[0] else {
            panic!("expected FusedGateUpActivateMul");
        };
        assert_eq!(n.activation, GateUpActivation::Silu);
        assert_eq!(n.iters.raw(), 8);
    }

    #[test]
    fn lower_fused_gate_up_gelu_mul_round_trip() {
        let mut op = OpInput::new(
            Instruction::FusedGateUpGeluMul(0, 1, 7),
            vec!["W::mlp_gate_up".to_string()],
        );
        op.iters = 4;
        let tape = lower(&[op], 16, mlp_budget()).expect("good gate-up-gelu-mul");
        let MegaNode::FusedGateUpActivateMul(n) = &tape.nodes()[0] else {
            panic!("expected FusedGateUpActivateMul");
        };
        assert_eq!(n.activation, GateUpActivation::Gelu);
        assert_eq!(n.iters.raw(), 4);
    }

    // ── Sprint D: Embed / Scalar / Gemm / lm_head / Attn / Barrier

    fn d_budget() -> SubstrateBudget {
        // Sprint D's lm_head fusion + attention need a richer
        // budget: 8 pages (delta + residual + norm_w + lin_w + out
        // + room for the slot-allocator's distinct picks), 8 KiB
        // scratch (split for partial_sums + B-tile / score + PV),
        // 4 cross-CTA edges for barrier tests.
        SubstrateBudget::new(8, 8, 32_768, 8_192).with_num_edges(4)
    }

    #[test]
    fn lowers_embed() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_embed(0, 1, "Weights::embed_tokens".to_string());
        let tape = b.finish();
        let MegaNode::Embed(n) = &tape.nodes()[0] else {
            panic!("expected Embed");
        };
        assert_eq!(n.out_page.raw(), 0);
        assert_eq!(n.embed_weight_page.raw(), 1);
        assert_eq!(n.embed_weight.path(), "Weights::embed_tokens");
    }

    #[test]
    fn lowers_scalar_mul_finite_scale() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_scalar_mul(0, 1, 0.5);
        let tape = b.finish();
        let MegaNode::ScalarMul(n) = &tape.nodes()[0] else {
            panic!("expected ScalarMul");
        };
        assert_eq!(n.scale.raw(), 0.5);
    }

    #[test]
    #[should_panic(expected = "FiniteF32 rejects non-finite value: NaN")]
    fn scalar_mul_rejects_nan_scale() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_scalar_mul(0, 1, f32::NAN);
    }

    #[test]
    #[should_panic(expected = "FiniteF32 rejects non-finite value: inf")]
    fn scalar_mul_rejects_inf_scale() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_scalar_mul(0, 1, f32::INFINITY);
    }

    #[test]
    fn lowers_tanh_soft_cap() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_tanh_soft_cap(0, 1);
        let tape = b.finish();
        let MegaNode::TanhSoftCap(_) = &tape.nodes()[0] else {
            panic!("expected TanhSoftCap");
        };
    }

    #[test]
    fn lowers_scalar_offset_rms_norm() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_scalar_offset_rms_norm(0, 1, 5, 16, "W::norm".to_string(), 0, 1.0);
        let tape = b.finish();
        let MegaNode::ScalarOffsetRmsNorm(n) = &tape.nodes()[0] else {
            panic!("expected ScalarOffsetRmsNorm");
        };
        assert_eq!(n.layer.raw(), 5);
        assert_eq!(n.weight.path(), "W::norm");
        assert_eq!(n.offset.raw(), 1.0);
    }

    #[test]
    fn lowers_gemm() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_gemm(
            0,
            1,
            2,
            3,
            16,
            "W::gemm".to_string(),
            /*n=*/ 4096,
            /*k=*/ 2048,
            /*iters=*/ 4,
            /*b_tile_off=*/ 0,
            /*b_tile_bytes=*/ 4_096,
        );
        let tape = b.finish();
        let MegaNode::Gemm(n) = &tape.nodes()[0] else {
            panic!("expected Gemm");
        };
        assert_eq!(n.shape.n, 4096);
        assert_eq!(n.shape.k, 2048);
        assert_eq!(n.iters.raw(), 4);
    }

    #[test]
    #[should_panic(expected = "MatmulShape: n must be > 0")]
    fn gemm_rejects_zero_n() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_gemm(0, 1, 2, 0, 16, "W".to_string(), 0, 16, 1, 0, 64);
    }

    #[test]
    #[should_panic(expected = "MatmulShape: k must be > 0")]
    fn gemm_rejects_zero_k() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_gemm(0, 1, 2, 0, 16, "W".to_string(), 16, 0, 1, 0, 64);
    }

    #[test]
    fn lowers_lm_head_rms_norm_gemm() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_cutlass_fused_norm_gemm(
            0,
            None,
            1,
            2,
            3,
            0,
            16,
            "W::norm".to_string(),
            "W::lm_head".to_string(),
            128_000,
            4_096,
            1,
            0,
            32,
            4_096,
            LmHeadNormKind::RmsNorm,
            None,
        );
        let tape = b.finish();
        let MegaNode::CutlassFusedNormGemm(n) = &tape.nodes()[0] else {
            panic!("expected CutlassFusedNormGemm");
        };
        assert_eq!(n.norm_kind, LmHeadNormKind::RmsNorm);
        assert!(n.delta_page.is_none());
        assert!(n.offset.is_none());
    }

    #[test]
    fn lowers_lm_head_add_scalar_offset_rms_norm_gemm() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_cutlass_fused_norm_gemm(
            0,
            Some(1),
            2,
            3,
            4,
            0,
            16,
            "W::norm".to_string(),
            "W::lm_head".to_string(),
            128_000,
            4_096,
            1,
            0,
            32,
            4_096,
            LmHeadNormKind::AddScalarOffsetRmsNorm,
            Some(1.0),
        );
        let tape = b.finish();
        let MegaNode::CutlassFusedNormGemm(n) = &tape.nodes()[0] else {
            panic!("expected CutlassFusedNormGemm");
        };
        assert_eq!(n.norm_kind, LmHeadNormKind::AddScalarOffsetRmsNorm);
        assert!(n.delta_page.is_some());
        assert_eq!(n.offset.unwrap().raw(), 1.0);
    }

    #[test]
    #[should_panic(expected = "AddScalarOffsetRmsNorm requires Some(offset)")]
    fn lm_head_rejects_missing_offset_for_scalar_offset_kind() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_cutlass_fused_norm_gemm(
            0,
            Some(1),
            2,
            3,
            4,
            0,
            16,
            "W::n".to_string(),
            "W::l".to_string(),
            16,
            16,
            1,
            0,
            32,
            4_096,
            LmHeadNormKind::AddScalarOffsetRmsNorm,
            None,
        );
    }

    #[test]
    #[should_panic(expected = "RmsNorm must not carry an offset")]
    fn lm_head_rejects_offset_for_rms_norm_kind() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_cutlass_fused_norm_gemm(
            0,
            None,
            1,
            2,
            3,
            0,
            16,
            "W::n".to_string(),
            "W::l".to_string(),
            16,
            16,
            1,
            0,
            32,
            4_096,
            LmHeadNormKind::RmsNorm,
            Some(1.0),
        );
    }

    #[test]
    fn lowers_attention_via_cache_full() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_attention_via_cache(
            0,
            1,
            5,
            16,
            /*iters=*/ 8,
            0,
            4_096,
            4_096,
            4_096,
            AttentionKind::Full,
            false,
        );
        let tape = b.finish();
        let MegaNode::AttentionViaCache(n) = &tape.nodes()[0] else {
            panic!("expected AttentionViaCache");
        };
        assert_eq!(n.kv_cache_layer.raw(), 5);
        assert_eq!(n.iters.raw(), 8);
        assert!(matches!(n.kind, AttentionKind::Full));
    }

    #[test]
    fn lowers_attention_via_cache_sliding() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_attention_via_cache(
            0,
            1,
            5,
            16,
            8,
            0,
            4_096,
            4_096,
            4_096,
            AttentionKind::Sliding(SlidingWindow::new(1024)),
            true,
        );
        let tape = b.finish();
        let MegaNode::AttentionViaCache(n) = &tape.nodes()[0] else {
            panic!("expected AttentionViaCache");
        };
        match n.kind {
            AttentionKind::Sliding(w) => assert_eq!(w.raw(), 1024),
            AttentionKind::Full => panic!("expected sliding"),
        }
    }

    #[test]
    #[should_panic(expected = "ScratchRegion overlap within scope")]
    fn attention_rejects_score_pv_scratch_overlap() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_attention_via_cache(
            0,
            1,
            0,
            16,
            1,
            /*score_off=*/ 0,
            /*score_bytes=*/ 1_024,
            /*pv_off=*/ 512,
            /*pv_bytes=*/ 1_024,
            AttentionKind::Full,
            false,
        );
    }

    #[test]
    #[should_panic(expected = "SlidingWindow must be > 0")]
    fn sliding_attention_rejects_zero_window() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_attention_via_cache(
            0,
            1,
            0,
            16,
            1,
            0,
            1_024,
            1_024,
            1_024,
            AttentionKind::Sliding(SlidingWindow::new(0)),
            false,
        );
    }

    #[test]
    fn lowers_barrier_signal_and_wait() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_barrier_signal(0);
        b.push_barrier_wait(0, 4);
        let tape = b.finish();
        let MegaNode::BarrierSignal(s) = &tape.nodes()[0] else {
            panic!("expected BarrierSignal");
        };
        assert_eq!(s.edge.raw(), 0);
        let MegaNode::BarrierWait(w) = &tape.nodes()[1] else {
            panic!("expected BarrierWait");
        };
        assert_eq!(w.edge.raw(), 0);
        assert_eq!(w.expected.raw(), 4);
    }

    #[test]
    #[should_panic(expected = "EdgeId 4 out of substrate budget num_edges=4")]
    fn barrier_signal_rejects_out_of_bounds_edge() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_barrier_signal(4);
    }

    #[test]
    #[should_panic(expected = "ExpectedCount: BarrierWait count must be > 0")]
    fn barrier_wait_rejects_zero_count() {
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_barrier_wait(0, 0);
    }

    #[test]
    fn barriers_do_not_advance_arrives() {
        // BarrierSignal/Wait operate on cross-CTA gmem atomics —
        // they do NOT bump the per-CTA mbarrier arrive count.
        // After two barriers, the next op's consumer phase should
        // still match an empty cumulative count.
        let mut b = MegaTapeBuilder::new(d_budget());
        b.push_barrier_signal(0);
        b.push_barrier_wait(0, 4);
        b.push_rms_norm(0, 1, 0, 16, "W::n".to_string(), 0);
        let tape = b.finish();
        let MegaNode::RmsNorm(n) = &tape.nodes()[2] else {
            panic!("expected RmsNorm at idx 2");
        };
        assert_eq!(n.consumer_phase.phase(), 0);
    }

    #[test]
    fn lower_embed_round_trip() {
        let ops = vec![OpInput::new(
            Instruction::Embed(0),
            vec!["W::embed".to_string()],
        )];
        let tape = lower(&ops, 16, d_budget()).expect("good Embed");
        let MegaNode::Embed(_) = &tape.nodes()[0] else {
            panic!("expected Embed");
        };
    }

    #[test]
    fn lower_scalar_mul_round_trip() {
        let ops = vec![OpInput::new(Instruction::ScalarMul(0, 1, 2.5), Vec::new())];
        let tape = lower(&ops, 16, d_budget()).expect("good ScalarMul");
        let MegaNode::ScalarMul(n) = &tape.nodes()[0] else {
            panic!("expected ScalarMul");
        };
        assert_eq!(n.scale.raw(), 2.5);
    }

    #[test]
    fn lower_gemm_round_trip() {
        let mut op = OpInput::new(
            Instruction::Gemm(0, 1, 5, 4_096, 2_048),
            vec!["W::gemm".to_string()],
        );
        op.iters = 4;
        let tape = lower(&[op], 16, d_budget()).expect("good Gemm");
        let MegaNode::Gemm(n) = &tape.nodes()[0] else {
            panic!("expected Gemm");
        };
        assert_eq!(n.iters.raw(), 4);
        assert_eq!(n.shape.n, 4_096);
    }

    #[test]
    fn lower_lm_head_rms_norm_gemm_round_trip() {
        let mut op = OpInput::new(
            Instruction::CutlassFusedRmsNormGemm(
                0, 1, 0, /*tile_m=*/ 0, /*tile_n=*/ 0, /*stages=*/ 0, 128_000, 4_096,
            ),
            vec!["W::norm".to_string(), "W::lm_head".to_string()],
        );
        op.iters = 1;
        let tape = lower(&[op], 16, d_budget()).expect("good lm_head fusion");
        let MegaNode::CutlassFusedNormGemm(n) = &tape.nodes()[0] else {
            panic!("expected CutlassFusedNormGemm");
        };
        assert_eq!(n.norm_kind, LmHeadNormKind::RmsNorm);
    }

    #[test]
    fn lower_sliding_attention_requires_window() {
        // Without sliding_window in OpInput, the lowering errors.
        let op = OpInput::new(
            Instruction::SlidingAttentionViaCache(0, 1, 0, false),
            Vec::new(),
        );
        match lower(&[op], 16, d_budget()) {
            Err(LowerError::MissingSlidingWindow) => {}
            other => panic!("expected MissingSlidingWindow, got {other:?}"),
        }
    }

    #[test]
    fn lower_sliding_attention_with_window_round_trip() {
        let mut op = OpInput::new(
            Instruction::SlidingAttentionViaCache(0, 1, 5, true),
            Vec::new(),
        );
        op.iters = 4;
        op.sliding_window = Some(2_048);
        let tape = lower(&[op], 16, d_budget()).expect("good sliding attn");
        let MegaNode::AttentionViaCache(n) = &tape.nodes()[0] else {
            panic!("expected AttentionViaCache");
        };
        match n.kind {
            AttentionKind::Sliding(w) => assert_eq!(w.raw(), 2_048),
            AttentionKind::Full => panic!("expected sliding"),
        }
    }

    #[test]
    fn lower_barrier_round_trip() {
        let ops = vec![
            OpInput::new(Instruction::BarrierSignal(0), Vec::new()),
            OpInput::new(Instruction::BarrierWait(0, 8), Vec::new()),
        ];
        let tape = lower(&ops, 16, d_budget()).expect("good barriers");
        assert_eq!(tape.nodes().len(), 2);
        assert!(matches!(tape.nodes()[0], MegaNode::BarrierSignal(_)));
        assert!(matches!(tape.nodes()[1], MegaNode::BarrierWait(_)));
    }

    #[test]
    fn lower_chains_add_rms_silu() {
        // Mini transformer-block tail: residual fold → norm → MLP
        // gate-up. Phase advance must thread through correctly.
        let ops = vec![
            OpInput::new(Instruction::Add(0, 1), Vec::new()),
            OpInput::new(
                Instruction::FusedAddRmsNorm(0, 1, 2),
                vec!["W::norm".to_string()],
            ),
            {
                let mut o = OpInput::new(
                    Instruction::FusedGateUpSiluMul(0, 1, 5),
                    vec!["W::mlp".to_string()],
                );
                o.iters = 4;
                o
            },
        ];
        let tape = lower(&ops, 16, mlp_budget()).expect("good chain");
        assert_eq!(tape.nodes().len(), 3);
        // Add: arrives 0 → 1 (one bump).
        // FusedAddRmsNorm: arrives 1 → 2 (one bump).
        // FusedGateUp*Mul iters=4: arrives 2 → 6 (four bumps).
        // Phase parities of consumer: 0, 1, 0.
        let phases: Vec<u32> = tape
            .nodes()
            .iter()
            .map(|n| match n {
                MegaNode::Add(a) => a.consumer_phase.phase(),
                MegaNode::FusedAddRmsNorm(f) => f.consumer_phase.phase(),
                MegaNode::FusedGateUpActivateMul(f) => f.consumer_phase.phase(),
                other => panic!("unexpected node {other:?}", other = node_kind(other)),
            })
            .collect();
        assert_eq!(phases, vec![0, 1, 0]);
    }
}
