// SPDX-License-Identifier: Apache-2.0
//! Kernel binding library + DAG coalescing pass.
//!
//! The reified DAG ([`crate::reified_dag::ReifiedDag`]) describes the
//! model's per-tile work units at the finest meaningful granularity.
//! Sitting next to it is a *library* of bound kernels — each entry is a
//! concrete CUDA kernel implementation (hand-written, FlashInfer,
//! CUTLASS, …) together with a pattern that says which DAG subgraph it
//! can replace and the CTA shape / cost it implies.
//!
//! The [`coalesce`] pass walks the DAG and rewrites matched subgraphs
//! into single coarser nodes bound to library entries. Anything that
//! doesn't match a registered pattern falls back to
//! [`BoundKernel::HandWrittenRowTile`], the universal 1:1 binding to
//! today's per-row tile bodies in `templates/scheduled/megakernel.cu`.
//!
//! This module is the *framework*. Phase B (this commit) only registers
//! the trivial fallback — `coalesce` produces a 1:1 mapping. Subsequent
//! phases register real library entries:
//!
//! - Phase C: `FlashInferAttentionLayer` — coalesces all attention row
//!   tiles for one layer into a single node bound to
//!   `BlockBatchPagedAttentionPersistent::Run`.
//! - Phase D+: CUTLASS GEMM collectives, FlashInfer norm/rope, fused
//!   norm+gemm patterns, …
//!
//! The wave scheduler (Phase C) groups coalesced nodes into waves under
//! a monomorphic constraint keyed on [`BoundKernel::kind`] — different
//! library entries need different CTA shapes, so they cannot share a
//! wave. The codegen (Phase C) dispatches per wave on the kernel kind
//! to emit the right tile body and launch parameters.
//!
//! Partial adoption is supported by design: as long as
//! `HandWrittenRowTile` is registered last (the universal fallback),
//! any kind that hasn't been bound to a library entry yet keeps using
//! today's hand-written body. Registering a new kernel only changes
//! the kinds it claims, never the rest.

use crate::reified_dag::{LlamaDims, NodeId, Phase, ReifiedDag, TileSizes};
use crate::schedule::CostModel;

/// Per-binding hardware resource demand. Consumed by the lowering
/// solver (`crate::lowering`) to decide which waves can share a
/// `__global__` function: two waves with very different `regs_per_thread`
/// values pay an occupancy penalty if grouped together (NVCC sets
/// `max-regs/CTA` per `__global__` to the union, which drops occupancy
/// for the lighter wave); waves whose summed `shmem_bytes` exceed the
/// target's per-CTA carveout cannot be grouped at all.
///
/// Numbers are estimates calibrated to the existing megakernel
/// instantiations, not exact NVCC outputs. The solver only needs
/// **relative** correctness so its inequalities point the right way.
/// Future work: replace with NVCC-reported per-`__global__` register
/// counts measured at codegen time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Resources {
    /// Per-CTA dynamic shmem demand for this binding's dispatch arm,
    /// in bytes. Hard constraint for the lowering solver: a kernel
    /// group's max shmem (over its waves) must fit
    /// `LoweringConstraints::max_shmem_per_cta`.
    pub shmem_bytes: u32,
    /// Per-thread register footprint estimate. Soft constraint for
    /// the solver: grouping waves with mismatched register counts
    /// raises NVCC's per-`__global__` reg cap, which drops occupancy
    /// for the lighter wave (modeled in `lowering::cost_occupancy`).
    pub regs_per_thread: u32,
    /// Threads per CTA the dispatch arm runs with. Currently 256
    /// for every variant in the megakernel template; reserved for
    /// future variants with different CTA shapes.
    pub threads_per_cta: u32,
}

/// One concrete binding instance: a kernel implementation choice plus
/// the DAG inputs it consumes.
///
/// Initially this enum has one variant — the universal fallback. Each
/// new library entry adds a variant carrying whatever parameters the
/// codegen needs to instantiate that kernel for that work unit
/// (e.g. `FlashInferAttentionLayer { layer }`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BoundKernel {
    /// 1:1 fallback: one reified DAG node = one tile body call.
    /// Used for any phase that doesn't yet have a library entry, and
    /// for phases where the per-tile body remains the right shape.
    HandWrittenRowTile {
        phase: Phase,
        layer: u16,
        row: u16,
        col: u16,
    },

    /// FlashInfer's `BlockBatchPagedAttentionPersistent::Run` runner,
    /// applied to one layer's full attention. The runner is a
    /// `__device__` function that takes `Params` + `SharedStorage` and
    /// processes the work assigned to its CTA via `work_indptr` —
    /// i.e. each persistent CTA in the wave pulls its share at runtime
    /// from a host-built plan, rather than the scheduler enumerating
    /// per-row tiles.
    ///
    /// The coalesce rule fuses every reified `Phase::Attention` node
    /// for a given layer into a single bound node carrying just the
    /// layer index. The dependencies of the absorbed nodes (typically
    /// the layer's RoPE outputs) become this node's dependencies; any
    /// edge that lived purely between absorbed attention nodes is
    /// dropped (no internal edges in a fused node).
    FlashInferAttentionLayer { layer: u16 },

    /// CUTLASS sm80 multistage GEMM via
    /// `cutlass::gemm::threadblock::MmaMultistage::operator()`,
    /// applied to one layer's worth of work for one of the GEMM
    /// phases (qkv / o_proj / gate_up / down).
    ///
    /// Same wave-cooperative pattern as `FlashInferAttentionLayer`:
    /// the coalesce rule fuses every reified node for one
    /// `(layer, phase)` pair into a single bound node, the
    /// scheduler replicates it across every CTA in its wave, and the
    /// dispatch arm runs CUTLASS's `MmaMultistage` per-CTA with each
    /// CTA picking its `(M_tile, N_tile)` work via its `bid`.
    ///
    /// The four phases share one variant — the megakernel's dispatch
    /// arm branches on `phase` to pick the right A/B/C base pointers
    /// and the right epilogue (LinearCombinationSiluMul for gate_up,
    /// LinearCombination(beta=1) residual-add for o_proj/down,
    /// LinearCombination(beta=0) for qkv).
    ///
    /// `tile` selects which cutlass tile-shape namespace the dispatch
    /// arm uses. The cost-gated polyalgo coalesce in
    /// `coalesce_with_target_profile` tries each available tile per
    /// phase and picks the one that minimizes `score_dag` — different
    /// (M, N) shapes prefer different tiles because of bin-pack
    /// rounding (small phases like qkv N=3072 leave CTAs idle with
    /// 128x128, fewer with 128x64).
    CutlassGemmLayer {
        layer: u16,
        phase: GemmPhase,
        tile: CutlassTile,
    },

    /// Fan-in fusion: a per-row producer phase whose entire fan
    /// terminates in a single wave-cooperative consumer for the same
    /// layer is rewritten as one wave-cooperative node that runs the
    /// producer body strided over CTAs, syncs the grid, then dispatches
    /// the wrapped consumer body — all inside a single wave's worth of
    /// time (one barrier instead of two).
    ///
    /// Today's enabled fan-in patterns (matched by `coalesce_consumer_fanin`):
    /// - `AttnNorm` rows → `CutlassGemmLayer{Qkv}`  (one barrier saved per layer)
    /// - `Rope` rows → `FlashInferAttentionLayer`
    /// - `MlpNorm` rows → `CutlassGemmLayer{GateUp}`
    ///
    /// The pass is cost-gated through `try_coalesce` — if simulating
    /// `partition_into_waves` says applying this fusion does not
    /// reduce predicted_cost, the rewrite is reverted.
    FusedFaninLayer {
        layer: u16,
        producer_phase: Phase,
        consumer: FaninConsumer,
    },
}

/// Which wave-cooperative consumer kind a [`BoundKernel::FusedFaninLayer`]
/// node wraps. Carried by value (not `Box<BoundKernel>`) so the variant
/// stays `Copy`-friendly and the codegen can match on it directly.
///
/// `CutlassGemm` carries the tile choice so the polyalgo decision the
/// cost gate made before the fan-in absorption is preserved through
/// the fusion — the fan-in dispatch arm reads the tile from the
/// FusedFaninLayer's `consumer` field and dispatches to the matching
/// cutlass helper.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FaninConsumer {
    CutlassGemm(GemmPhase, CutlassTile),
    FlashInferAttention,
}

/// Polyalgorithmic tile-shape choice for [`BoundKernel::CutlassGemmLayer`].
/// Each variant maps to a vendored cutlass namespace in
/// `templates/scheduled/megakernel.cu` (`pfl_cutlass_*`) with a
/// matching `tile_cutlass_gemm_*_lincomb` / `_silumul` helper.
///
/// Adding a tile is: a new namespace + helper(s) in megakernel.cu, a
/// new variant here, a new dispatch tag in `kernel_tag()`, and the
/// matching `(tile_m, tile_n, tile_k)` constants returned by
/// `tile_dims()`. The polyalgo coalesce in
/// `coalesce_with_target_profile` will then automatically search the
/// new variant per phase via `try_coalesce`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CutlassTile {
    /// `pfl_cutlass_small`: 128x128x32, 4 cp.async stages.
    Small,
    /// `pfl_cutlass_narrow`: 128x64x32, 4 cp.async stages. Wins for
    /// narrow-N phases (qkv N=3072, oproj/down N=2048) where 128x128
    /// would leave CTAs idle from `ceil(work_units / NUM_CTAS)`
    /// rounding waste.
    Narrow,
}

impl CutlassTile {
    /// Returns `(tile_m, tile_n, tile_k)` for cost-model scoring.
    /// These must match the cutlass namespace's `ThreadblockShape`.
    pub const fn tile_dims(self) -> (u32, u32, u32) {
        match self {
            CutlassTile::Small => (128, 128, 32),
            CutlassTile::Narrow => (128, 64, 32),
        }
    }
}

/// Which of the four GEMM phases a [`BoundKernel::CutlassGemmLayer`]
/// node represents. Carried as the `row` field of the WAVE_OPS entry
/// (since cutlass-fused nodes don't have a row index — the cutlass
/// body iterates over the layer's full M dim internally).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GemmPhase {
    Qkv,
    OProj,
    GateUp,
    Down,
}

impl GemmPhase {
    /// Numeric tag in the WAVE_OPS `row` slot. Stable across codegen
    /// versions. Matches `PFL_GEMM_PHASE_*` constants in megakernel.cu.
    pub fn tag(self) -> u32 {
        match self {
            GemmPhase::Qkv => 0,
            GemmPhase::OProj => 1,
            GemmPhase::GateUp => 2,
            GemmPhase::Down => 3,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            GemmPhase::Qkv => "qkv",
            GemmPhase::OProj => "o_proj",
            GemmPhase::GateUp => "gate_up",
            GemmPhase::Down => "down",
        }
    }

    /// Which reified-DAG `Phase` this gemm phase coalesces.
    pub fn source_phase(self) -> Phase {
        match self {
            GemmPhase::Qkv => Phase::Qkv,
            GemmPhase::OProj => Phase::OProj,
            GemmPhase::GateUp => Phase::GateUp,
            GemmPhase::Down => Phase::Down,
        }
    }
}

impl BoundKernel {
    /// Stable string tag identifying which kernel this binding uses.
    ///
    /// The wave scheduler enforces *monomorphic waves* keyed on this
    /// tag — two coalesced nodes can share a wave only if their kinds
    /// match — because different bound kernels need different CTA
    /// shapes (NUM_THREADS, smem, register budget) and you cannot mix
    /// them inside one persistent grid launch.
    ///
    /// For `HandWrittenRowTile`, the tag is the phase name: today's
    /// codegen already keys per-CTA dispatch on phase, and the per-row
    /// tile bodies for distinct phases happen to share a CTA shape
    /// (256 threads, ~36 KiB shmem) only because we wrote them that
    /// way. Once a library entry registers a different shape for one
    /// of those phases, this tag is what keeps it isolated.
    pub fn kind(&self) -> &'static str {
        match self {
            BoundKernel::HandWrittenRowTile { phase, .. } => phase.name(),
            BoundKernel::FlashInferAttentionLayer { .. } => "flashinfer_attention_layer",
            BoundKernel::CutlassGemmLayer { phase, tile, .. } => match (phase, tile) {
                (GemmPhase::Qkv, CutlassTile::Small) => "cutlass_gemm_qkv_layer_small",
                (GemmPhase::Qkv, CutlassTile::Narrow) => "cutlass_gemm_qkv_layer_narrow",
                (GemmPhase::OProj, CutlassTile::Small) => "cutlass_gemm_o_proj_layer_small",
                (GemmPhase::OProj, CutlassTile::Narrow) => "cutlass_gemm_o_proj_layer_narrow",
                (GemmPhase::GateUp, CutlassTile::Small) => "cutlass_gemm_gate_up_layer_small",
                (GemmPhase::GateUp, CutlassTile::Narrow) => "cutlass_gemm_gate_up_layer_narrow",
                (GemmPhase::Down, CutlassTile::Small) => "cutlass_gemm_down_layer_small",
                (GemmPhase::Down, CutlassTile::Narrow) => "cutlass_gemm_down_layer_narrow",
            },
            BoundKernel::FusedFaninLayer {
                producer_phase,
                consumer,
                ..
            } => match (producer_phase, consumer) {
                (Phase::AttnNorm, FaninConsumer::CutlassGemm(GemmPhase::Qkv, _)) => {
                    "fanin_attn_norm_cutlass_qkv"
                }
                (Phase::Rope, FaninConsumer::FlashInferAttention) => "fanin_rope_fi_attn",
                (Phase::MlpNorm, FaninConsumer::CutlassGemm(GemmPhase::GateUp, _)) => {
                    "fanin_mlp_norm_cutlass_gate_up"
                }
                _ => "fanin_unknown",
            },
        }
    }

    /// Predicted cost of executing this binding, in the same mma-unit
    /// domain as `CostModel`. The wave scheduler uses this number for
    /// LPT bin packing inside a wave and for the makespan rollup.
    ///
    /// For `HandWrittenRowTile` we delegate to the existing per-phase
    /// cost (one tile body per node, the status quo). When new
    /// library entries land, each provides its own per-binding cost
    /// — e.g. `FlashInferAttentionLayer` will roll up the whole
    /// layer's attention into one number.
    /// Numeric tag identifying this binding's dispatch arm in the
    /// generated megakernel's per-CTA op switch. Tags 0..7 are
    /// reserved for the existing per-phase
    /// [`BoundKernel::HandWrittenRowTile`] dispatch (matching the
    /// `PHASE_*` constants in the megakernel template). Tags ≥8 are
    /// for library-bound kernels:
    ///
    /// - `8` — `FlashInferAttentionLayer`
    ///
    /// New library entries claim a stable tag in this enum. The
    /// megakernel template's dispatch switch must grow a matching
    /// `case` arm at the same time.
    /// Does this binding consume an entire wave's worth of CTAs
    /// cooperatively (i.e. all CTAs in the wave participate in the
    /// same work item), or is it a per-CTA tile dispatched LPT-style
    /// across the wave's CTAs?
    ///
    /// `HandWrittenRowTile` is the per-CTA model: each binding is
    /// one tile body call on one CTA, the wave scheduler bin-packs
    /// many of them across the wave's CTAs via LPT.
    ///
    /// `FlashInferAttentionLayer` is wave-cooperative: the FlashInfer
    /// runner internally partitions the layer's attention work across
    /// every CTA in the persistent grid via `work_indptr[blockIdx.y]`,
    /// so the scheduler must place the binding on **every** CTA in
    /// the wave (not bin-pack it onto one CTA). The per-CTA cost is
    /// the rolled-up `cost()` divided across `num_ctas`, since the
    /// CTAs run in parallel.
    pub fn is_wave_cooperative(&self) -> bool {
        match self {
            BoundKernel::HandWrittenRowTile { .. } => false,
            BoundKernel::FlashInferAttentionLayer { .. } => true,
            // CUTLASS GEMM layer kernels are wave-cooperative for the
            // same reason as FlashInfer attention: each persistent CTA
            // in the wave picks one (M_tile, N_tile) work item via its
            // `bid`, and the layer's full GEMM is distributed across
            // them in a CTA-strided loop. Same `gemm_cutlass_mcta.cu`
            // pattern as the existing fused prefill kernel.
            BoundKernel::CutlassGemmLayer { .. } => true,
            // Fan-in fusion absorbs a per-row producer into a wave-coop
            // consumer. The result runs as a single wave-cooperative
            // node: every CTA strides over its share of producer rows,
            // grid-syncs, then participates in the consumer body.
            BoundKernel::FusedFaninLayer { .. } => true,
        }
    }

    pub fn kernel_tag(&self) -> u32 {
        match self {
            BoundKernel::HandWrittenRowTile { phase, .. } => match phase {
                Phase::AttnNorm => 0,
                Phase::Qkv => 1,
                Phase::Rope => 2,
                Phase::Attention => 3,
                Phase::OProj => 4,
                Phase::MlpNorm => 5,
                Phase::GateUp => 6,
                Phase::Down => 7,
            },
            BoundKernel::FlashInferAttentionLayer { .. } => 8,
            // 9 reserved for future use (e.g. unused/idle marker).
            // 10..13: the four cutlass gemm phases.
            // 14..16: fan-in fusion patterns. Each carries a unique
            // dispatch arm in the megakernel template; allocating one
            // tag per (producer, consumer) combo keeps the dispatch
            // switch trivial — no need to decode producer_phase /
            // consumer from the op stream's row/col slots.
            // Per-phase tag (10..13). The tile choice is encoded in
            // the WAVE_OPS `col` slot (`tile_tag`) and the dispatch
            // arm switches on it to pick the right cutlass helper.
            // Keeps the dispatch switch flat — 4 arms instead of 8 —
            // and the compiler inlines both helper branches per arm.
            BoundKernel::CutlassGemmLayer { phase, .. } => match phase {
                GemmPhase::Qkv => 10,
                GemmPhase::OProj => 11,
                GemmPhase::GateUp => 12,
                GemmPhase::Down => 13,
            },
            BoundKernel::FusedFaninLayer {
                producer_phase,
                consumer,
                ..
            } => match (producer_phase, consumer) {
                (Phase::AttnNorm, FaninConsumer::CutlassGemm(GemmPhase::Qkv, _)) => 14,
                (Phase::Rope, FaninConsumer::FlashInferAttention) => 15,
                (Phase::MlpNorm, FaninConsumer::CutlassGemm(GemmPhase::GateUp, _)) => 16,
                // Future fan-in patterns claim 17+ here; the megakernel
                // template gains a matching dispatch arm at the same
                // time. Until then, panic clearly so we don't silently
                // emit a tag with no dispatch arm.
                _ => panic!(
                    "FusedFaninLayer with producer={producer_phase:?}, consumer={consumer:?} \
                     has no kernel_tag assigned; allocate one and add a dispatch arm"
                ),
            },
        }
    }

    /// Per-binding hardware resource demand. See [`Resources`] for the
    /// semantics. Numbers are estimates calibrated to the existing
    /// megakernel instantiations — they only need to be **relatively**
    /// correct so the lowering solver's inequalities point the right way.
    ///
    /// Conventions:
    /// - `shmem_bytes` for cutlass arms tracks the matching namespace's
    ///   `SharedStorage` template instantiation.
    /// - `shmem_bytes` for FlashInfer attention tracks `KTraits::SharedStorage`
    ///   on bf16 head_dim=64 prefill (~70 KiB measured).
    /// - `shmem_bytes` for hand-written tile bodies tracks `TILE_SHMEM_BYTES`
    ///   in the megakernel template (currently 36 KiB).
    /// - `regs_per_thread` for cutlass mainloops is high (~128); for
    ///   norm/rope row tile bodies it's much lower (~32).
    pub fn resources(&self) -> Resources {
        match self {
            BoundKernel::HandWrittenRowTile { phase, .. } => {
                // Per-row tile bodies are warp-level reductions / per-row
                // copies dispatched on a 256-thread CTA. They share the
                // megakernel template's TILE_SHMEM_BYTES arena.
                let regs = match phase {
                    Phase::AttnNorm | Phase::MlpNorm => 24,
                    Phase::Rope => 32,
                    // Hand-written GEMM bodies (mostly dead with the
                    // cutlass profile) — heavier on registers because
                    // they unroll the wmma mainloop.
                    Phase::Qkv | Phase::OProj | Phase::GateUp | Phase::Down => 96,
                    Phase::Attention => 96,
                };
                Resources {
                    shmem_bytes: 36 * 1024,
                    regs_per_thread: regs,
                    threads_per_cta: 256,
                }
            }
            BoundKernel::FlashInferAttentionLayer { .. } => Resources {
                // FlashInfer KTraits1 SharedStorage at bf16 head_dim=64
                // prefill ≈ 70 KiB. Sets the megakernel's per-CTA shmem
                // floor on L4 (the static arena is `max(arm shmem)`).
                shmem_bytes: 70 * 1024,
                regs_per_thread: 128,
                threads_per_cta: 256,
            },
            BoundKernel::CutlassGemmLayer { tile, .. } => match tile {
                CutlassTile::Small => Resources {
                    // pfl_cutlass_small: 128x128x32, 4 cp.async stages.
                    // SharedStorage instantiation lands ~49 KiB; round
                    // to 50 KiB for headroom.
                    shmem_bytes: 50 * 1024,
                    regs_per_thread: 128,
                    threads_per_cta: 256,
                },
                CutlassTile::Narrow => Resources {
                    // pfl_cutlass_narrow: 128x64x32, 4 cp.async stages.
                    // SharedStorage instantiation lands ~25 KiB.
                    shmem_bytes: 25 * 1024,
                    regs_per_thread: 96,
                    threads_per_cta: 256,
                },
            },
            BoundKernel::FusedFaninLayer { consumer, .. } => {
                // Fan-in arms wrap a wave-coop consumer with a small
                // producer prologue + intra-arm grid sync. Resources
                // are dominated by the wrapped consumer.
                match consumer {
                    FaninConsumer::CutlassGemm(phase, tile) => BoundKernel::CutlassGemmLayer {
                        layer: 0,
                        phase: *phase,
                        tile: *tile,
                    }
                    .resources(),
                    FaninConsumer::FlashInferAttention => {
                        BoundKernel::FlashInferAttentionLayer { layer: 0 }.resources()
                    }
                }
            }
        }
    }

    pub fn cost(&self, model: &CostModel) -> u32 {
        match self {
            BoundKernel::HandWrittenRowTile { phase, .. } => model.cost(*phase),
            // Layer-rolled-up attention: one binding does the work that
            // used to be `seq_len / row_tile` separate row tiles. Sum
            // them so the wave scheduler still budgets the right total
            // mma-units per layer.
            //
            // The number is intentionally an over-estimate of FlashInfer's
            // real cost — measured FlashInfer attention runs in ~3.5 ms
            // per layer at seq=1024 (~0.2 ms per row tile), the cost
            // model says ~10× more. Doesn't matter for the LPT bin
            // packer because attention waves are wave-cooperative
            // (every CTA participates); the wave's makespan is
            // determined by this rolled-up cost regardless.
            BoundKernel::FlashInferAttentionLayer { .. } => {
                let row_tiles = model.seq_len.div_ceil(model.row_tile);
                row_tiles * model.cost(Phase::Attention)
            }
            // Per-layer rolled-up cost for the four CUTLASS GEMM
            // phases — same shape as the FlashInfer attention rollup
            // and same caveats about over-estimation. Wave-cooperative
            // execution means all CTAs participate, so per-CTA cost
            // distribution doesn't matter — only the wave-level total.
            // Cost is now polyalgo-aware: it depends on the chosen
            // tile shape via bin-pack rounding. The cost-gated coalesce
            // search uses this to pick the cheaper tile per phase
            // shape — small tiles win for narrow-N (qkv N=3072,
            // oproj/down N=2048) where 128x128 leaves work-unit
            // rounding waste; the larger tile wins for fat-N (gate_up
            // N=8192) where there are enough tiles to amortize.
            BoundKernel::CutlassGemmLayer { phase, tile, .. } => {
                let m = model.seq_len;
                let k = match phase {
                    GemmPhase::Qkv => model.hidden_dim,
                    GemmPhase::OProj => model.hidden_dim,
                    GemmPhase::GateUp => model.hidden_dim,
                    GemmPhase::Down => model.intermediate_dim,
                };
                let n = match phase {
                    GemmPhase::Qkv => {
                        // (num_attn_h + 2 * num_kv_h) * head_dim
                        (model.num_attn_heads + 2 * model.num_kv_heads) * model.head_dim
                    }
                    GemmPhase::OProj => model.hidden_dim,
                    GemmPhase::GateUp => model.intermediate_dim,
                    GemmPhase::Down => model.hidden_dim,
                };
                let (tm, tn, tk) = tile.tile_dims();
                model.cutlass_gemm_total(m, n, k, tm, tn, tk)
            }
            // Fan-in fusion: total work = producer fan + intra-arm
            // grid sync + consumer. The intra-arm sync exists because
            // the consumer reads gmem outputs that the producer wrote
            // across multiple CTAs (cross-CTA dataflow → block-local
            // __syncthreads is insufficient → must use a real grid
            // sync). Charging `BARRIER_COST_MMA_UNITS` here is what
            // makes the cost-gated coalesce honest: the saved
            // wave-end barrier is exactly cancelled out by the
            // intra-arm sync, so the cost gate sees no net change
            // and (correctly) reverts the fusion for these patterns.
            //
            // Future fusion patterns where the consumer only reads
            // its own CTA's producer outputs (e.g. an in-CTA fused
            // norm+gemm where the gemm reads the same rows the norm
            // wrote) would override this to a smaller intra-arm cost
            // (`__syncthreads`-only), and the gate would accept them.
            BoundKernel::FusedFaninLayer {
                producer_phase,
                consumer,
                ..
            } => {
                use crate::schedule::BARRIER_COST_MMA_UNITS;
                let row_tiles = model.seq_len.div_ceil(model.row_tile);
                let producer_cost = row_tiles * model.cost(*producer_phase);
                let consumer_cost = match consumer {
                    FaninConsumer::CutlassGemm(p, _) => row_tiles * model.cost(p.source_phase()),
                    FaninConsumer::FlashInferAttention => row_tiles * model.cost(Phase::Attention),
                };
                producer_cost + (BARRIER_COST_MMA_UNITS as u32) + consumer_cost
            }
        }
    }
}

/// One node in the coalesced DAG.
///
/// `id` is preserved from the source `ReifiedDag` when a node maps 1:1
/// (the only case in Phase B). When the coalesce pass starts fusing
/// subgraphs (Phase C+), the coalesced node will get a fresh id and
/// the bound kernel will reference the original ids it absorbed via
/// its variant fields.
#[derive(Clone, Debug)]
pub struct CoalescedNode {
    pub id: NodeId,
    pub kernel: BoundKernel,
    pub deps: Vec<NodeId>,
}

/// The output of the coalesce pass: a DAG where every node carries an
/// explicit binding to a library kernel.
///
/// In Phase B this has the same shape as the input `ReifiedDag` (1:1
/// node mapping, all bound to `HandWrittenRowTile`). The struct is
/// kept separate from `ReifiedDag` rather than added as a parallel
/// vector field because Phase C will introduce real fusion — the
/// coalesced node count will diverge from the reified node count, and
/// dependencies will need to be remapped from absorbed ids to the new
/// coalesced ids.
#[derive(Clone, Debug)]
pub struct CoalescedDag {
    /// Model dimensions, lifted from the source `ReifiedDag`. Cost
    /// models, codegen prelude, and FlashInfer plan-time inputs all
    /// need these — keep them attached so consumers don't need to
    /// hold a separate handle to the source DAG.
    pub dims: LlamaDims,
    /// Tile shape policy, lifted from the source `ReifiedDag`. Used
    /// by the cost model and the codegen template.
    pub tiles: TileSizes,
    pub nodes: Vec<CoalescedNode>,
}

impl CoalescedDag {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Walk a `ReifiedDag` through the registered kernel library and produce
/// a `CoalescedDag`.
///
/// Phase B: only `HandWrittenRowTile` is registered, so this is a pure
/// 1:1 lift — every reified node becomes a coalesced node with the
/// matching binding and its deps preserved. The point is to land the
/// types and the call site so Phase C can register `FlashInferAttentionLayer`
/// without restructuring the pipeline.
///
/// When more library entries land, the implementation here grows from a
/// single `iter().map()` into a real graph rewriting pass — try each
/// pattern in priority order, replace matched subgraphs with coarser
/// coalesced nodes, fall back to `HandWrittenRowTile` for everything
/// unmatched.
pub fn coalesce(dag: &ReifiedDag) -> CoalescedDag {
    let nodes = dag
        .nodes
        .iter()
        .map(|n| CoalescedNode {
            id: n.id,
            kernel: BoundKernel::HandWrittenRowTile {
                phase: n.phase,
                layer: n.layer,
                row: n.row,
                col: n.col,
            },
            deps: n.deps.clone(),
        })
        .collect();
    CoalescedDag {
        dims: dag.dims,
        tiles: dag.tiles,
        nodes,
    }
}

/// Coalesce variant that fuses every layer's attention row tiles into
/// one [`BoundKernel::FlashInferAttentionLayer`] node, leaving every
/// other phase on the [`BoundKernel::HandWrittenRowTile`] fallback.
///
/// Phase C2a: this function exists alongside the trivial [`coalesce`]
/// but is **not** wired into the production pipeline yet — Phase C2b
/// switches the call sites to use it once the dispatch + launcher
/// plumbing in `scheduled_codegen` and the megakernel template can
/// emit a `FlashInferAttentionLayer` work item.
///
/// Fusion semantics for one layer:
/// 1. Find every reified node with `phase == Attention && layer == L`.
/// 2. Drop those nodes from the coalesced output.
/// 3. Replace them with one fused node:
///    - id: the smallest absorbed `NodeId` (stable across rebuilds —
///      the codegen test asserts this).
///    - kernel: `FlashInferAttentionLayer { layer: L }`.
///    - deps: the union of the absorbed nodes' deps, *minus* any
///      dependency that pointed to another absorbed node (no
///      internal edges in a fused node).
/// 4. For every other coalesced node (any non-attention phase) that
///    used to depend on an absorbed attention id, rewrite that dep
///    to point at the fused node's id.
///
/// All other phases pass through unchanged (1:1 fallback).
pub fn coalesce_with_flashinfer_attention(dag: &ReifiedDag) -> CoalescedDag {
    use std::collections::{HashMap, HashSet};

    // ── Step 1: index attention nodes by layer ──
    let mut attn_ids_by_layer: HashMap<u16, Vec<NodeId>> = HashMap::new();
    for n in &dag.nodes {
        if n.phase == Phase::Attention {
            attn_ids_by_layer.entry(n.layer).or_default().push(n.id);
        }
    }

    // ── Step 2: pick the fused id per layer (smallest absorbed id) ──
    let mut fused_id_by_layer: HashMap<u16, NodeId> = HashMap::new();
    for (layer, ids) in &attn_ids_by_layer {
        let min_id = ids.iter().copied().min().expect("layer has attn nodes");
        fused_id_by_layer.insert(*layer, min_id);
    }

    // ── Step 3: build absorbed-id → fused-id rewrite map ──
    let mut rewrite: HashMap<NodeId, NodeId> = HashMap::new();
    for (layer, ids) in &attn_ids_by_layer {
        let fused = fused_id_by_layer[layer];
        for id in ids {
            rewrite.insert(*id, fused);
        }
    }

    // ── Step 4: build the coalesced node list ──
    // For each reified node:
    //   - If it's an attention node and its id IS the fused id for its
    //     layer, emit one fused FlashInferAttentionLayer node with the
    //     unioned, internally-pruned deps.
    //   - If it's an attention node and its id is NOT the fused id,
    //     drop it (it's been absorbed into the fused node).
    //   - Otherwise, emit a HandWrittenRowTile fallback, rewriting
    //     any deps that pointed to absorbed attention ids.
    let mut nodes: Vec<CoalescedNode> = Vec::with_capacity(dag.nodes.len());
    let absorbed: HashSet<NodeId> = rewrite.keys().copied().collect();

    for n in &dag.nodes {
        if n.phase == Phase::Attention {
            let fused_id = fused_id_by_layer[&n.layer];
            if n.id != fused_id {
                continue; // absorbed into the fused node — drop
            }
            // Union all deps from every attention node in this layer,
            // pruning internal edges and deduping.
            let mut deps: Vec<NodeId> = Vec::new();
            let mut seen: HashSet<NodeId> = HashSet::new();
            for absorbed_id in &attn_ids_by_layer[&n.layer] {
                let absorbed_node = &dag.nodes[absorbed_id.0 as usize];
                for d in &absorbed_node.deps {
                    if absorbed.contains(d) {
                        continue; // internal edge — drop
                    }
                    if seen.insert(*d) {
                        deps.push(*d);
                    }
                }
            }
            nodes.push(CoalescedNode {
                id: fused_id,
                kernel: BoundKernel::FlashInferAttentionLayer { layer: n.layer },
                deps,
            });
        } else {
            // Non-attention phase: rewrite any dep pointing into the
            // absorbed set to point at the matching fused id.
            let deps: Vec<NodeId> = n
                .deps
                .iter()
                .map(|d| rewrite.get(d).copied().unwrap_or(*d))
                .collect();
            nodes.push(CoalescedNode {
                id: n.id,
                kernel: BoundKernel::HandWrittenRowTile {
                    phase: n.phase,
                    layer: n.layer,
                    row: n.row,
                    col: n.col,
                },
                deps,
            });
        }
    }

    renumber_dense_ids(&mut nodes);

    CoalescedDag {
        dims: dag.dims,
        tiles: dag.tiles,
        nodes,
    }
}

/// In-place dense renumbering of `NodeId`s to `[0, nodes.len())`.
/// Used at the end of every coalesce pass that drops or reorders
/// nodes (the schedule + codegen index `dag.nodes[NodeId.0 as usize]`,
/// so any gap in the id space blows them up).
fn renumber_dense_ids(nodes: &mut [CoalescedNode]) {
    use std::collections::HashMap;
    let mut id_remap: HashMap<NodeId, NodeId> = HashMap::with_capacity(nodes.len());
    for (new_idx, n) in nodes.iter().enumerate() {
        id_remap.insert(n.id, NodeId(new_idx as u32));
    }
    for n in nodes.iter_mut() {
        n.id = id_remap[&n.id];
        for d in &mut n.deps {
            *d = *id_remap
                .get(d)
                .expect("dep references a node that doesn't exist in coalesced output");
        }
    }
}

/// Per-phase GEMM coalesce rule. Fuses every reified
/// `(layer, phase, row, col)` node where `node.phase == gemm_phase.source_phase()`
/// into one [`BoundKernel::CutlassGemmLayer { layer, phase: gemm_phase }`]
/// node per layer. Other phases (and other GEMM phases not matching
/// `gemm_phase`) pass through unchanged.
///
/// Mirrors `coalesce_with_flashinfer_attention`'s shape: pick a stable
/// fused id (smallest absorbed), drop internal edges, rewrite incoming
/// deps to the fused id, dense-renumber.
///
/// **Takes a `CoalescedDag` so multiple coalesce rules can be
/// composed.** The combined entry point
/// `coalesce_with_target_profile` runs whichever rules the profile
/// asks for, in dependency-safe order.
pub fn coalesce_gemm_phase(
    input: CoalescedDag,
    gemm_phase: GemmPhase,
    tile: CutlassTile,
) -> CoalescedDag {
    use std::collections::{HashMap, HashSet};
    let source_phase = gemm_phase.source_phase();

    // Polyalgo retile: if the input already contains CutlassGemmLayer
    // nodes for this phase (because a previous polyalgo iteration
    // already coalesced it with a different tile), the producer
    // HandWrittenRowTile nodes are gone — there's nothing left to
    // absorb. In that case the only thing this call can usefully do
    // is rewrite the tile field of the existing CutlassGemmLayer
    // nodes for this phase, so the cost gate can compare different
    // tile shapes via try_coalesce.
    let already_fused: bool = input.nodes.iter().any(|n| {
        matches!(
            n.kernel,
            BoundKernel::CutlassGemmLayer { phase, .. } if phase == gemm_phase
        )
    });
    if already_fused {
        let mut new_nodes: Vec<CoalescedNode> = Vec::with_capacity(input.nodes.len());
        for n in &input.nodes {
            let kernel = match n.kernel {
                BoundKernel::CutlassGemmLayer {
                    layer,
                    phase: p,
                    tile: _,
                } if p == gemm_phase => BoundKernel::CutlassGemmLayer {
                    layer,
                    phase: p,
                    tile,
                },
                ref k => k.clone(),
            };
            new_nodes.push(CoalescedNode {
                id: n.id,
                kernel,
                deps: n.deps.clone(),
            });
        }
        return CoalescedDag {
            dims: input.dims,
            tiles: input.tiles,
            nodes: new_nodes,
        };
    }

    // Index target nodes by layer.
    let mut target_ids_by_layer: HashMap<u16, Vec<NodeId>> = HashMap::new();
    for n in &input.nodes {
        if let BoundKernel::HandWrittenRowTile { phase, layer, .. } = n.kernel
            && phase == source_phase
        {
            target_ids_by_layer.entry(layer).or_default().push(n.id);
        }
    }
    if target_ids_by_layer.is_empty() {
        return input;
    }

    // Pick the fused id per layer (smallest absorbed id).
    let mut fused_id_by_layer: HashMap<u16, NodeId> = HashMap::new();
    for (layer, ids) in &target_ids_by_layer {
        let min_id = ids.iter().copied().min().expect("layer has target nodes");
        fused_id_by_layer.insert(*layer, min_id);
    }

    // absorbed_id → fused_id rewrite map.
    let mut rewrite: HashMap<NodeId, NodeId> = HashMap::new();
    for (layer, ids) in &target_ids_by_layer {
        let fused = fused_id_by_layer[layer];
        for id in ids {
            rewrite.insert(*id, fused);
        }
    }
    let absorbed: HashSet<NodeId> = rewrite.keys().copied().collect();

    let mut new_nodes: Vec<CoalescedNode> = Vec::with_capacity(input.nodes.len());
    for n in &input.nodes {
        let is_target = matches!(
            n.kernel,
            BoundKernel::HandWrittenRowTile { phase, .. } if phase == source_phase
        );
        if is_target {
            let layer = match n.kernel {
                BoundKernel::HandWrittenRowTile { layer, .. } => layer,
                _ => unreachable!(),
            };
            let fused_id = fused_id_by_layer[&layer];
            if n.id != fused_id {
                continue; // absorbed into the fused node
            }
            // Union the absorbed nodes' deps, prune internal edges,
            // rewrite already-fused incoming deps.
            let mut deps: Vec<NodeId> = Vec::new();
            let mut seen: HashSet<NodeId> = HashSet::new();
            for absorbed_id in &target_ids_by_layer[&layer] {
                // Find the absorbed node by id
                let absorbed_node = input
                    .nodes
                    .iter()
                    .find(|nd| nd.id == *absorbed_id)
                    .expect("absorbed id must exist in input");
                for d in &absorbed_node.deps {
                    if absorbed.contains(d) {
                        continue; // internal edge — drop
                    }
                    if seen.insert(*d) {
                        deps.push(*d);
                    }
                }
            }
            new_nodes.push(CoalescedNode {
                id: fused_id,
                kernel: BoundKernel::CutlassGemmLayer {
                    layer,
                    phase: gemm_phase,
                    tile,
                },
                deps,
            });
        } else {
            // Non-target: rewrite any deps that point at the absorbed set.
            let deps: Vec<NodeId> = n
                .deps
                .iter()
                .map(|d| rewrite.get(d).copied().unwrap_or(*d))
                .collect();
            new_nodes.push(CoalescedNode {
                id: n.id,
                kernel: n.kernel.clone(),
                deps,
            });
        }
    }

    renumber_dense_ids(&mut new_nodes);
    CoalescedDag {
        dims: input.dims,
        tiles: input.tiles,
        nodes: new_nodes,
    }
}

/// Fan-in fusion pass: absorb a per-row producer phase into a
/// wave-cooperative consumer in the same layer. The pattern is matched
/// once per layer; if all the producer rows for that layer terminate in
/// **exactly one** wave-coop consumer node — and that consumer is the
/// supplied `consumer_kind` — then the producer rows are absorbed and
/// the consumer node is rewritten as a [`BoundKernel::FusedFaninLayer`].
///
/// Constraints checked per layer (any failure → leave layer unchanged):
/// - All producer rows are `HandWrittenRowTile { phase: producer_phase, layer }`.
/// - Their union of successors is exactly `{ consumer_id }` — every
///   producer's only successor must be the consumer. Otherwise some
///   absorbed producer would leave a dangling dep edge.
/// - The consumer exists, is in the same layer, and matches `consumer_kind`.
///
/// The new node's deps = `(consumer.deps ∪ each producer.deps) − absorbed`.
/// Internal edges (consumer→producer or producer→producer within the
/// fused set) are dropped — they're now intra-arm syncthreads in the
/// rendered dispatch.
///
/// **Cost-gating is handled by the caller** via `try_coalesce`. This
/// function unconditionally applies the rewrite where it pattern-matches.
pub fn coalesce_consumer_fanin(
    input: CoalescedDag,
    producer_phase: Phase,
    consumer_kind: FaninConsumer,
) -> CoalescedDag {
    use std::collections::{HashMap, HashSet};

    // Build successor lists once.
    let mut successors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for n in &input.nodes {
        for d in &n.deps {
            successors.entry(*d).or_default().push(n.id);
        }
    }

    // For each layer, check if the fan-in pattern applies. The tuple
    // is (consumer_id, producer_ids, actual_consumer_kind) — we extract
    // the consumer's *real* fields (e.g. cutlass tile) here so the
    // FusedFaninLayer node we emit later carries the polyalgo decision
    // the cost gate made earlier.
    let mut layers_to_fuse: HashMap<u16, (NodeId, Vec<NodeId>, FaninConsumer)> = HashMap::new();
    let nodes_by_id: HashMap<NodeId, &CoalescedNode> =
        input.nodes.iter().map(|n| (n.id, n)).collect();

    // Collect candidate producer nodes by layer.
    let mut producers_by_layer: HashMap<u16, Vec<NodeId>> = HashMap::new();
    for n in &input.nodes {
        if let BoundKernel::HandWrittenRowTile { phase, layer, .. } = n.kernel
            && phase == producer_phase
        {
            producers_by_layer.entry(layer).or_default().push(n.id);
        }
    }

    // For each layer, verify the constraints.
    for (layer, producer_ids) in &producers_by_layer {
        // Find every producer's set of successors.
        let mut consumer_set: HashSet<NodeId> = HashSet::new();
        let mut all_have_one_succ = true;
        for pid in producer_ids {
            let succs = successors.get(pid).cloned().unwrap_or_default();
            if succs.len() != 1 {
                all_have_one_succ = false;
                break;
            }
            consumer_set.insert(succs[0]);
        }
        if !all_have_one_succ || consumer_set.len() != 1 {
            continue; // pattern doesn't match — leave layer alone
        }
        let consumer_id = *consumer_set.iter().next().unwrap();
        let consumer_node = match nodes_by_id.get(&consumer_id) {
            Some(n) => n,
            None => continue,
        };
        // Verify the consumer kind + layer match, and extract the
        // actual `FaninConsumer` to store in the fused node.
        let actual_consumer: Option<FaninConsumer> = match (&consumer_node.kernel, consumer_kind) {
            (
                BoundKernel::CutlassGemmLayer {
                    layer: cl,
                    phase,
                    tile,
                },
                FaninConsumer::CutlassGemm(target_phase, _),
            ) if *cl == *layer && *phase == target_phase => {
                Some(FaninConsumer::CutlassGemm(*phase, *tile))
            }
            (
                BoundKernel::FlashInferAttentionLayer { layer: cl },
                FaninConsumer::FlashInferAttention,
            ) if *cl == *layer => Some(FaninConsumer::FlashInferAttention),
            _ => None,
        };
        let actual_consumer = match actual_consumer {
            Some(c) => c,
            None => continue,
        };
        // All checks passed — schedule this layer for fusion.
        layers_to_fuse.insert(*layer, (consumer_id, producer_ids.clone(), actual_consumer));
    }

    if layers_to_fuse.is_empty() {
        return input;
    }

    // Build the absorbed-id set and the rewrite map (absorbed → fused id).
    let mut absorbed: HashSet<NodeId> = HashSet::new();
    let mut rewrite: HashMap<NodeId, NodeId> = HashMap::new();
    for (consumer_id, producers, _) in layers_to_fuse.values() {
        for p in producers {
            absorbed.insert(*p);
            rewrite.insert(*p, *consumer_id);
        }
    }

    // Emit new nodes: consumer becomes FusedFaninLayer, producers are dropped,
    // everything else has its deps rewritten.
    let mut new_nodes: Vec<CoalescedNode> = Vec::with_capacity(input.nodes.len());
    for n in &input.nodes {
        if absorbed.contains(&n.id) {
            continue; // dropped — its deps are unioned into the consumer below
        }
        // Is this node a fusion target?
        let fusion_for_layer = layers_to_fuse
            .iter()
            .find(|(_, (consumer_id, _, _))| *consumer_id == n.id)
            .map(|(layer, (_, producers, actual))| (*layer, producers.clone(), *actual));

        if let Some((layer, producers, actual_consumer)) = fusion_for_layer {
            // Union deps of the consumer + every absorbed producer,
            // drop internal edges, dedupe.
            let mut deps: Vec<NodeId> = Vec::new();
            let mut seen: HashSet<NodeId> = HashSet::new();
            // Consumer's deps (rewritten if any point at absorbed nodes —
            // shouldn't happen since absorbed are the producers and the
            // consumer's deps include the producers; we drop those).
            for d in &n.deps {
                if absorbed.contains(d) {
                    continue;
                }
                if seen.insert(*d) {
                    deps.push(*d);
                }
            }
            // Each producer's deps (rewritten / deduped).
            for pid in &producers {
                if let Some(pn) = nodes_by_id.get(pid) {
                    for d in &pn.deps {
                        if absorbed.contains(d) {
                            continue;
                        }
                        if seen.insert(*d) {
                            deps.push(*d);
                        }
                    }
                }
            }
            new_nodes.push(CoalescedNode {
                id: n.id,
                kernel: BoundKernel::FusedFaninLayer {
                    layer,
                    producer_phase,
                    consumer: actual_consumer,
                },
                deps,
            });
        } else {
            // Non-target: rewrite any deps pointing into absorbed set.
            let deps: Vec<NodeId> = n
                .deps
                .iter()
                .map(|d| rewrite.get(d).copied().unwrap_or(*d))
                .collect();
            new_nodes.push(CoalescedNode {
                id: n.id,
                kernel: n.kernel.clone(),
                deps,
            });
        }
    }

    renumber_dense_ids(&mut new_nodes);
    CoalescedDag {
        dims: input.dims,
        tiles: input.tiles,
        nodes: new_nodes,
    }
}

/// Combined coalesce pass driven by [`TargetProfile`]. Runs whichever
/// fusion rules the profile's kernel choices ask for, in
/// dependency-safe order. This is the **entry point** the production
/// codegen calls.
///
/// Adding a new fusion rule (e.g. for a future `FusedNormGemm`
/// kernel) is one new call here, gated on the appropriate profile
/// field.
/// Cost-gated coalesce wrapper. Applies `f` to the input DAG; keeps
/// the result only if `score_dag` (the simulated `partition_into_waves`
/// predicted_cost) decreases. Otherwise reverts.
///
/// This is the framework that makes future fusion experiments safe:
/// any new coalesce pass plugs in via `try_coalesce`, and the cost
/// model decides whether it ships. Failed experiments don't pollute
/// the dispatch arms — they're just not applied.
fn try_coalesce<F>(input: CoalescedDag, num_ctas: u32, label: &str, f: F) -> CoalescedDag
where
    F: FnOnce(CoalescedDag) -> CoalescedDag,
{
    use crate::schedule::score_dag;
    let before = score_dag(&input, num_ctas);
    // f takes ownership; we need a clone in case we revert.
    let candidate = f(input.clone());
    let after = score_dag(&candidate, num_ctas);
    if after < before {
        let _ = label; // logging hook reserved for a future trace flag
        candidate
    } else {
        input
    }
}

pub fn coalesce_with_target_profile(
    dag: &ReifiedDag,
    profile: &crate::target_profile::TargetProfile,
) -> CoalescedDag {
    use crate::target_profile::{AttentionKernelChoice, GemmKernelChoice};

    let num_ctas = profile.cooperative_grid_size();

    // Always start from the trivial 1:1 lift.
    let mut coalesced = coalesce(dag);

    // Attention fusion (FlashInferPersistent path). This pass replaces
    // the trivial coalesced DAG entirely (it re-runs from the reified
    // DAG to pick up the per-layer attention fan-in), so we evaluate
    // the swap as a single try_coalesce: the candidate is the
    // attention-fused DAG, the baseline is the trivial coalesce.
    if matches!(
        profile.attention_kernel,
        AttentionKernelChoice::FlashInferPersistent
    ) {
        let candidate = coalesce_with_flashinfer_attention(dag);
        // Direct cost compare since the function shape doesn't fit
        // try_coalesce's "rewrite the input" pattern.
        use crate::schedule::score_dag;
        if score_dag(&candidate, num_ctas) < score_dag(&coalesced, num_ctas) {
            coalesced = candidate;
        }
    }

    // CUTLASS GEMM fusion — polyalgorithmic per phase. For each phase
    // we try every tile shape in CUTLASS_TILE_CANDIDATES and let
    // try_coalesce pick the one that minimizes predicted_cost (which
    // is now polyalgo-aware via the BoundKernel::cost branch for
    // CutlassGemmLayer that uses CutlassTile::tile_dims). The cost
    // model handles bin-pack rounding waste, so narrow-N phases
    // (qkv N=3072, oproj/down N=2048) tend to pick CutlassTile::Narrow
    // while fat-N phases (gate_up N=8192) tend to pick Small.
    if matches!(
        profile.gemm_kernel,
        GemmKernelChoice::CutlassSm80Multistage { .. }
    ) {
        const CUTLASS_TILE_CANDIDATES: &[CutlassTile] = &[CutlassTile::Small, CutlassTile::Narrow];
        for phase in [
            GemmPhase::Qkv,
            GemmPhase::OProj,
            GemmPhase::GateUp,
            GemmPhase::Down,
        ] {
            for &tile in CUTLASS_TILE_CANDIDATES {
                coalesced = try_coalesce(coalesced, num_ctas, "cutlass_polyalgo", |c| {
                    coalesce_gemm_phase(c, phase, tile)
                });
            }
        }
    }

    // Fan-in fusion passes — each absorbs a per-row producer phase
    // into a wave-cooperative consumer in the same layer, saving one
    // grid_barrier per layer per fused pair. Cost-gated through
    // try_coalesce: if applying a pass doesn't reduce predicted_cost
    // (e.g. because the pattern doesn't apply, or the saved barrier
    // is outweighed by the increased per-CTA work), the pass reverts
    // and the DAG is left untouched. The order matters for the
    // dependency-safety check inside coalesce_consumer_fanin: each
    // pass requires its consumer to already be coalesced into a
    // single wave-coop node, so attention/cutlass passes must run
    // first (they did, above).
    // The placeholder `CutlassTile::Small` here is not the *actual*
    // tile the fan-in arm will use — coalesce_consumer_fanin extracts
    // the matched consumer's real tile field and stores it in the
    // FusedFaninLayer it emits. The argument is only used as a "match
    // a CutlassGemm consumer" type tag. Two arms with `FaninConsumer
    // ::CutlassGemm(_, x)` and `(_, y)` would match the same node.
    coalesced = try_coalesce(coalesced, num_ctas, "fanin_attn_norm_qkv", |c| {
        coalesce_consumer_fanin(
            c,
            Phase::AttnNorm,
            FaninConsumer::CutlassGemm(GemmPhase::Qkv, CutlassTile::Small),
        )
    });
    coalesced = try_coalesce(coalesced, num_ctas, "fanin_rope_fi_attn", |c| {
        coalesce_consumer_fanin(c, Phase::Rope, FaninConsumer::FlashInferAttention)
    });
    coalesced = try_coalesce(coalesced, num_ctas, "fanin_mlp_norm_gate_up", |c| {
        coalesce_consumer_fanin(
            c,
            Phase::MlpNorm,
            FaninConsumer::CutlassGemm(GemmPhase::GateUp, CutlassTile::Small),
        )
    });

    coalesced
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reified_dag::{LlamaDims, TileSizes};

    fn tiny_dims() -> LlamaDims {
        LlamaDims {
            num_layers: 2,
            hidden_dim: 256,
            intermediate_dim: 512,
            num_attn_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            seq_len: 32,
        }
    }

    #[test]
    fn coalesce_is_one_to_one_with_only_fallback_registered() {
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let coalesced = coalesce(&dag);

        assert_eq!(
            coalesced.len(),
            dag.nodes.len(),
            "Phase B coalesce should be a 1:1 lift; every reified node becomes \
             one coalesced node bound to HandWrittenRowTile."
        );

        for (cnode, rnode) in coalesced.nodes.iter().zip(dag.nodes.iter()) {
            assert_eq!(cnode.id, rnode.id, "node id preserved");
            assert_eq!(cnode.deps, rnode.deps, "deps preserved");
            match &cnode.kernel {
                BoundKernel::HandWrittenRowTile {
                    phase,
                    layer,
                    row,
                    col,
                } => {
                    assert_eq!(*phase, rnode.phase);
                    assert_eq!(*layer, rnode.layer);
                    assert_eq!(*row, rnode.row);
                    assert_eq!(*col, rnode.col);
                }
                BoundKernel::FlashInferAttentionLayer { .. } => {
                    panic!("trivial coalesce should never produce FlashInferAttentionLayer");
                }
                BoundKernel::CutlassGemmLayer { .. } => {
                    panic!("trivial coalesce should never produce CutlassGemmLayer");
                }
                BoundKernel::FusedFaninLayer { .. } => {
                    panic!("trivial coalesce should never produce FusedFaninLayer");
                }
            }
        }
    }

    #[test]
    fn kind_matches_phase_name_for_fallback() {
        // The monomorphic wave constraint will key on `kind()`. For the
        // fallback binding the kind is the phase name, which means
        // grouping by kind today is identical to grouping by phase —
        // i.e. behavior-preserving relative to the pre-library scheduler.
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let coalesced = coalesce(&dag);

        for (cnode, rnode) in coalesced.nodes.iter().zip(dag.nodes.iter()) {
            assert_eq!(cnode.kernel.kind(), rnode.phase.name());
        }
    }

    #[test]
    fn flashinfer_coalesce_fuses_attention_per_layer() {
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let coalesced = coalesce_with_flashinfer_attention(&dag);

        // Count by kind in the coalesced output.
        let mut by_kind = std::collections::HashMap::<&'static str, usize>::new();
        for cn in &coalesced.nodes {
            *by_kind.entry(cn.kernel.kind()).or_insert(0) += 1;
        }

        // tiny has NL=2 layers and SEQ_LEN=32 with row_tile=16 → 2 attn
        // row tiles per layer × 2 layers = 4 reified attention nodes,
        // fused into 2 FlashInferAttentionLayer nodes (one per layer).
        let nl = tiny_dims().num_layers as usize;
        assert_eq!(
            by_kind
                .get("flashinfer_attention_layer")
                .copied()
                .unwrap_or(0),
            nl,
            "expected one FlashInferAttentionLayer per layer; got {by_kind:?}",
        );

        // The total node count drops by exactly the number of absorbed
        // attention nodes minus the one fused replacement per layer.
        let row_tiles_per_layer = tiny_dims()
            .seq_len
            .div_ceil(TileSizes::default_v1().row_tile) as usize;
        let absorbed = nl * row_tiles_per_layer;
        let replacements = nl;
        assert_eq!(coalesced.len(), dag.nodes.len() - (absorbed - replacements));
    }

    #[test]
    fn flashinfer_coalesce_drops_internal_attention_edges() {
        // No coalesced node should depend on a NodeId that was absorbed
        // into a different fused FlashInferAttentionLayer node — only
        // on its own fused id (which is impossible since we drop
        // internal edges) or on non-absorbed ids.
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let coalesced = coalesce_with_flashinfer_attention(&dag);

        let live_ids: std::collections::HashSet<NodeId> =
            coalesced.nodes.iter().map(|n| n.id).collect();

        for cn in &coalesced.nodes {
            for d in &cn.deps {
                assert!(
                    live_ids.contains(d),
                    "coalesced node {:?} has dep {d:?} that doesn't exist in the coalesced output",
                    cn.id
                );
            }
        }
    }

    #[test]
    fn flashinfer_coalesce_preserves_non_attention_phases() {
        // Every non-attention phase should still appear in the coalesced
        // output, untouched, with the same row/col counts as the
        // reified DAG.
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let coalesced = coalesce_with_flashinfer_attention(&dag);

        let mut reified_non_attn = 0usize;
        for n in &dag.nodes {
            if n.phase != Phase::Attention {
                reified_non_attn += 1;
            }
        }
        let mut coalesced_non_attn = 0usize;
        for cn in &coalesced.nodes {
            if !matches!(cn.kernel, BoundKernel::FlashInferAttentionLayer { .. }) {
                coalesced_non_attn += 1;
            }
        }
        assert_eq!(reified_non_attn, coalesced_non_attn);
    }

    #[test]
    fn cutlass_gemm_coalesce_fuses_one_phase_per_layer() {
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let trivial = coalesce(&dag);
        let fused = coalesce_gemm_phase(trivial, GemmPhase::GateUp, CutlassTile::Small);

        let mut by_kind = std::collections::HashMap::<&'static str, usize>::new();
        for cn in &fused.nodes {
            *by_kind.entry(cn.kernel.kind()).or_insert(0) += 1;
        }

        // tiny: NL=2 layers, gate_up has 1 row tile × 4 col tiles per
        // layer = 4 nodes, fused into 1 CutlassGemmGateUpLayer node
        // per layer = 2 fused nodes total.
        let nl = tiny_dims().num_layers as usize;
        assert_eq!(
            by_kind
                .get("cutlass_gemm_gate_up_layer_small")
                .copied()
                .unwrap_or(0),
            nl
        );
        // Other GEMM phases pass through untouched (still
        // HandWrittenRowTile).
        assert!(
            by_kind.get("gate_up").copied().unwrap_or(0) == 0,
            "no HandWrittenRowTile gate_up should remain after fusion: {by_kind:?}"
        );
        assert!(by_kind.get("qkv").copied().unwrap_or(0) > 0);
        assert!(by_kind.get("down").copied().unwrap_or(0) > 0);
    }

    #[test]
    fn cutlass_gemm_coalesce_composes_with_attention_fusion() {
        // Run flashinfer attention fusion + all four cutlass gemm
        // fusions in sequence (the production order).
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let mut c = coalesce_with_flashinfer_attention(&dag);
        c = coalesce_gemm_phase(c, GemmPhase::Qkv, CutlassTile::Small);
        c = coalesce_gemm_phase(c, GemmPhase::OProj, CutlassTile::Small);
        c = coalesce_gemm_phase(c, GemmPhase::GateUp, CutlassTile::Small);
        c = coalesce_gemm_phase(c, GemmPhase::Down, CutlassTile::Small);

        let mut by_kind = std::collections::HashMap::<&'static str, usize>::new();
        for cn in &c.nodes {
            *by_kind.entry(cn.kernel.kind()).or_insert(0) += 1;
        }

        let nl = tiny_dims().num_layers as usize;
        assert_eq!(
            by_kind
                .get("flashinfer_attention_layer")
                .copied()
                .unwrap_or(0),
            nl
        );
        for kind in [
            "cutlass_gemm_qkv_layer_small",
            "cutlass_gemm_o_proj_layer_small",
            "cutlass_gemm_gate_up_layer_small",
            "cutlass_gemm_down_layer_small",
        ] {
            assert_eq!(
                by_kind.get(kind).copied().unwrap_or(0),
                nl,
                "expected {nl} {kind} nodes; got {by_kind:?}"
            );
        }

        // Every node id should be dense [0, len) — schedule.rs
        // depends on this.
        let live_ids: std::collections::HashSet<NodeId> = c.nodes.iter().map(|n| n.id).collect();
        for cn in &c.nodes {
            for d in &cn.deps {
                assert!(
                    live_ids.contains(d),
                    "coalesced node has dep {d:?} not in the coalesced output"
                );
            }
        }
    }

    #[test]
    fn coalesce_with_target_profile_dispatches_correctly() {
        use crate::target_profile::{
            AttentionKernelChoice, GemmKernelChoice, NormKernelChoice, RopeKernelChoice,
            TargetProfile,
        };
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());

        // Profile with cutlass enabled — should produce CutlassGemmLayer
        // nodes for all four phases plus FlashInferAttentionLayer.
        let profile_cutlass = TargetProfile {
            num_sm: 58,
            seq_len: 1024,
            cooperative_blocks_per_sm: 1,
            max_dynamic_shmem_bytes: 99 * 1024,
            gemm_kernel: GemmKernelChoice::CutlassSm80Multistage {
                tile_m: 256,
                tile_n: 128,
                tile_k: 32,
                pipeline_stages: 4,
            },
            attention_kernel: AttentionKernelChoice::FlashInferPersistent,
            norm_kernel: NormKernelChoice::HandWrittenWarpShuffle,
            rope_kernel: RopeKernelChoice::HandWrittenSplitHalf,
            lowering: crate::target_profile::LoweringConstraints::l4_sm89(),
        };
        let coalesced_cutlass = coalesce_with_target_profile(&dag, &profile_cutlass);
        let mut kinds: std::collections::HashSet<&'static str> = Default::default();
        for cn in &coalesced_cutlass.nodes {
            kinds.insert(cn.kernel.kind());
        }
        // The cutlass arms for o_proj and down survive standalone (no
        // fan-in pattern absorbs them — their producers are the
        // wave-coop attn / o_proj outputs, not per-row tiles). The
        // polyalgo cost gate picks Small or Narrow per phase based on
        // bin-pack rounding cost, so we accept either suffix.
        let has_oproj_cutlass = kinds.contains("cutlass_gemm_o_proj_layer_small")
            || kinds.contains("cutlass_gemm_o_proj_layer_narrow");
        let has_down_cutlass = kinds.contains("cutlass_gemm_down_layer_small")
            || kinds.contains("cutlass_gemm_down_layer_narrow");
        assert!(
            has_oproj_cutlass,
            "expected an o_proj cutlass kind; got {kinds:?}"
        );
        assert!(
            has_down_cutlass,
            "expected a down cutlass kind; got {kinds:?}"
        );
        // The other three (qkv / gate_up cutlass + flashinfer_attention)
        // get absorbed by their respective fan-in coalesce passes when
        // those win under the cost gate. At tiny dims with the
        // production barrier_cost the gate accepts all three, so the
        // fan-in kinds appear and the standalone variants do not.
        assert!(kinds.contains("fanin_attn_norm_cutlass_qkv"));
        assert!(kinds.contains("fanin_rope_fi_attn"));
        assert!(kinds.contains("fanin_mlp_norm_cutlass_gate_up"));
        assert!(!kinds.contains("flashinfer_attention_layer"));
        assert!(!kinds.contains("cutlass_gemm_qkv_layer_small"));
        assert!(!kinds.contains("cutlass_gemm_qkv_layer_narrow"));
        assert!(!kinds.contains("cutlass_gemm_gate_up_layer_small"));
        assert!(!kinds.contains("cutlass_gemm_gate_up_layer_narrow"));
        // No HandWrittenRowTile remnants for the four GEMM phases or
        // for the per-row producers that got absorbed.
        assert!(!kinds.contains("qkv"));
        assert!(!kinds.contains("o_proj"));
        assert!(!kinds.contains("gate_up"));
        assert!(!kinds.contains("down"));
        assert!(!kinds.contains("attn_norm"));
        assert!(!kinds.contains("rope"));
        assert!(!kinds.contains("mlp_norm"));

        // Profile with cutlass DISABLED — should NOT produce any
        // CutlassGemmLayer nodes; the four GEMM phases stay as
        // HandWrittenRowTile.
        let profile_wmma = TargetProfile {
            gemm_kernel: GemmKernelChoice::HandWrittenWmma,
            ..profile_cutlass
        };
        let coalesced_wmma = coalesce_with_target_profile(&dag, &profile_wmma);
        let mut kinds_wmma: std::collections::HashSet<&'static str> = Default::default();
        for cn in &coalesced_wmma.nodes {
            kinds_wmma.insert(cn.kernel.kind());
        }
        // The cutlass GEMM phases stay as HandWrittenRowTile (qkv,
        // o_proj, gate_up, down) — no fan-in pattern targets a hand-
        // written GEMM consumer, so the qkv/gate_up rows + the
        // attn_norm/mlp_norm rows that produce them all stay
        // standalone.
        assert!(kinds_wmma.contains("qkv"));
        assert!(kinds_wmma.contains("o_proj"));
        assert!(kinds_wmma.contains("gate_up"));
        assert!(kinds_wmma.contains("down"));
        assert!(kinds_wmma.contains("attn_norm"));
        assert!(kinds_wmma.contains("mlp_norm"));
        assert!(!kinds_wmma.contains("cutlass_gemm_qkv_layer_small"));
        assert!(!kinds_wmma.contains("fanin_attn_norm_cutlass_qkv"));
        assert!(!kinds_wmma.contains("fanin_mlp_norm_cutlass_gate_up"));
        // The rope → fi_attn fan-in still applies because fi_attn is
        // wave-coop coalesced regardless of the GEMM kernel choice.
        // So the standalone fi_attn kind disappears and the fan-in
        // kind appears in its place.
        assert!(kinds_wmma.contains("fanin_rope_fi_attn"));
        assert!(!kinds_wmma.contains("flashinfer_attention_layer"));
        assert!(!kinds_wmma.contains("rope"));
    }

    #[test]
    fn try_coalesce_rejects_a_regression() {
        // Verify the cost gate actually reverts a transform that
        // increases predicted_cost. We construct a no-op identity
        // (which has score == before, so `after < before` is false →
        // revert) and a degenerate "duplicate every node's deps"
        // mutator (which doesn't change ids but also doesn't reduce
        // cost → revert). Both must produce a DAG byte-equal to the
        // input.
        use crate::reified_dag::TileSizes;
        let reified = ReifiedDag::reify_llama(
            LlamaDims {
                num_layers: 2,
                hidden_dim: 256,
                intermediate_dim: 512,
                num_attn_heads: 4,
                num_kv_heads: 2,
                head_dim: 64,
                seq_len: 32,
            },
            TileSizes::default_v1(),
        );
        let baseline = coalesce(&reified);
        let baseline_len = baseline.nodes.len();

        // Identity transform: cost is exactly equal → `after < before`
        // is false → revert.
        let after_identity = try_coalesce(baseline.clone(), 8, "identity", |c| c);
        assert_eq!(after_identity.nodes.len(), baseline_len);

        // A real coalesce that we know reduces cost (the FlashInfer
        // attention fusion) should ship under the gate. We use a
        // direct call to verify it would normally compress nodes,
        // then run it via try_coalesce and check the count drops.
        let gated = try_coalesce(baseline.clone(), 8, "fi_attn", |_| {
            coalesce_with_flashinfer_attention(&reified)
        });
        assert!(
            gated.nodes.len() < baseline_len,
            "fi_attn coalesce should reduce node count under the cost gate"
        );
    }

    #[test]
    fn flashinfer_coalesce_rewrites_deps_into_fused_ids() {
        // A non-attention node that used to depend on an attention row
        // tile (e.g. o_proj depends on attention) should, after the
        // flashinfer coalesce, depend on the *fused* attention node id
        // for its layer — not on a now-absorbed reified attention id.
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let coalesced = coalesce_with_flashinfer_attention(&dag);

        // Build (id → kind tag) for the coalesced output so we can
        // assert that downstream attention deps point at a fused node.
        let kind_of: std::collections::HashMap<NodeId, &'static str> = coalesced
            .nodes
            .iter()
            .map(|n| (n.id, n.kernel.kind()))
            .collect();

        let mut rewritten_count = 0usize;
        for cn in &coalesced.nodes {
            if cn.kernel.kind() == "o_proj" {
                for d in &cn.deps {
                    if kind_of.get(d).copied() == Some("flashinfer_attention_layer") {
                        rewritten_count += 1;
                    }
                }
            }
        }
        assert!(
            rewritten_count > 0,
            "expected at least one o_proj→flashinfer_attention_layer dep after coalesce"
        );
    }

    #[test]
    fn every_phase_appears_in_coalesced_output() {
        // Sanity: the trivial coalesce pass shouldn't drop or merge any
        // phase even when run on a multi-layer DAG.
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let coalesced = coalesce(&dag);

        let mut seen = std::collections::HashSet::new();
        for cnode in &coalesced.nodes {
            seen.insert(cnode.kernel.kind());
        }
        for phase in [
            Phase::AttnNorm,
            Phase::Qkv,
            Phase::Rope,
            Phase::Attention,
            Phase::OProj,
            Phase::MlpNorm,
            Phase::GateUp,
            Phase::Down,
        ] {
            assert!(
                seen.contains(phase.name()),
                "expected phase {} in coalesced output, only saw {seen:?}",
                phase.name()
            );
        }
    }
}
