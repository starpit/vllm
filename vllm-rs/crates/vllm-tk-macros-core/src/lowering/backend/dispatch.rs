// SPDX-License-Identifier: Apache-2.0
//! Flattens an [`ExecutionPlan`] into an ordered dispatch sequence.
//!
//! The solver produces an [`Assignment`] with abstract
//! `SubgraphId`/`ImplId`/`ScheduleSlot` mappings. The dispatch
//! module resolves these into a flat, step-ordered list of
//! [`DispatchEntry`]s that the codegen or runtime can walk
//! sequentially.

use std::collections::BTreeMap;

use crate::lowering::assignment::{CompilationUnitId, SubgraphId};
use crate::lowering::implementation::{ImplId, LaunchKind};
use crate::lowering::library::ImplementationLibrary;
use crate::lowering::solver::ExecutionPlan;
use crate::lowering::tile_graph::{TileGraph, TileKind};

/// Which family of FFI dispatch to use for this entry.
///
/// The codegen uses this to emit the right FFI call without
/// string-matching on impl names. Each variant maps to exactly
/// one FFI calling convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImplDispatchKind {
    /// `cublasGemmEx` — params: (M, N, K, alpha, beta, weight, act, out).
    CublasGemm,
    /// `cublasGemmEx` with fused bias epilogue (`cublas.gemm_bias`).
    /// Claims {Gemm{phase}, BiasAdd}. Codegen dispatches through
    /// `LinearLayer::forward`, which internally picks the bias path
    /// when `bias.is_some()`.
    CublasGemmWithBias,
    /// Standalone per-column bias broadcast add on a GEMM output
    /// buffer. Claims {BiasAdd} only — pairs with a bias-less GEMM
    /// impl on the preceding `Gemm{phase}` tile.
    BiasAdd,
    /// `cutlass_gemm_{M}x{N}_s{stages}_launch` — same params + tile size + stages.
    CutlassGemm {
        tile_m: u32,
        tile_n: u32,
        stages: u32,
    },
    /// `cutlass_gemv_launch` — M=1 specialization (SIMT GEMV).
    CutlassGemv,
    /// `cutlass_norm_gemm` — fused norm→GEMM prologue, no separate norm launch.
    CutlassNormGemm { tile_m: u32, tile_n: u32 },
    /// `rms_norm_bf16`.
    RmsNorm,
    /// Decode: `fused_qkv_rope_cache_bf16` — 3-tile fused (split + Q rope + cache write).
    FusedQkvRopeCache,
    /// Prefill: `split_qkv` + `rotary_embedding_inplace` (both Q and K) + `write_kv_cache`.
    PrefillRopeCache,
    /// `rotary_embedding_bf16`.
    RotaryEmbedding,
    /// `silu_and_mul_fused_bf16`.
    SiluAndMul,
    /// Decode: FlashInfer `attention_decode_from_cache`.
    FlashInferAttention,
    /// Prefill: FlashInfer `attention_standard` with explicit Q, K, V.
    FlashInferStandard,
    /// Decode attention inside persistent kernel (FlashInfer `BlockPersistentRunner::Run()`).
    TkAttentionDecode,
    /// Prefill attention inside persistent kernel (FlashInfer `BlockPersistentRunner::Run()`).
    TkAttentionPrefill,
    /// Free passthrough — no FFI call needed (e.g. QkvSplit, KvCacheWrite, ResidualAdd
    /// when folded into an upstream beta=1 epilogue).
    Noop,
    /// Fused QKV GEMM — one GEMM with concatenated [q|k|v] weight.
    /// Claims {GemmQ, GemmK, GemmV} as a 3-tile subgraph.
    FusedQkvGemm,
    /// Fused QKV GEMM + bias — one `gemm_bias` with concatenated
    /// weight + bias. Claims {GemmQ, BiasAdd, GemmK, BiasAdd, GemmV, BiasAdd}.
    FusedQkvGemmWithBias,
    /// Fused gate+up GEMM — one GEMM with concatenated [gate|up] weight.
    /// Claims {GemmGate, GemmUp} as a 2-tile subgraph.
    FusedGateUpGemm,
    /// Embedding-table gather — `kernels::embedding_gather`.
    /// Always pre-loop (runs once per forward, before the
    /// per-layer match dispatch).
    Embed,
    /// CUTLASS Gate GEMM + SiLU + Mul epilogue fusion.
    /// Claims {GemmGate, GateUpConcat, SiluMul} as a 3-tile subgraph.
    /// The CUTLASS epilogue computes `silu(accumulator) * up_output`,
    /// where `up_output` is loaded from GMEM via VisitorAuxLoad.
    /// Saves two kernel launches + two GMEM round-trips.
    CutlassGemmSiluMul {
        tile_m: u32,
        tile_n: u32,
        stages: u32,
    },
}

/// Which GEMM phase this entry operates on (determines buffer
/// pointers and weight offsets).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GemmPhase {
    Q,
    K,
    V,
    OProj,
    Gate,
    Up,
    Down,
    /// Final projection `[seq, vocab] = [seq, hidden] @ [hidden, vocab]`.
    /// Runs once after all decoder layers; weight is `model.lm_head`.
    LmHead,
}

/// One entry in the dispatch sequence — fully resolved, ready for
/// codegen or runtime dispatch.
#[derive(Clone, Debug)]
pub struct DispatchEntry {
    /// Which subgraph this entry dispatches (for debug/tracing).
    pub subgraph: SubgraphId,
    /// The implementation id in the library.
    pub impl_id: ImplId,
    /// Stable implementation name (for debug/tracing and the runtime
    /// interpreter fallback path).
    pub impl_name: &'static str,
    /// Typed dispatch kind — the codegen uses this, not the name.
    pub kind: ImplDispatchKind,
    /// Transformer layer index (0-based).
    pub layer: u16,
    /// Schedule step (for debug; entries are already sorted by step).
    pub step: u32,
    /// Launch kind (determines standalone vs megakernel body).
    pub launch_kind: LaunchKind,
    /// Whether this GEMM folds a residual add (beta=1).
    pub fused_residual: bool,
    /// For GEMM entries: which phase (determines N, K, buffer pointers).
    pub gemm_phase: Option<GemmPhase>,
    /// For RmsNorm: whether this is the attention norm (true) or MLP
    /// norm (false) within the layer.
    pub is_attn_norm: Option<bool>,
    /// Compilation unit this entry belongs to. Entries sharing a
    /// `CompilationUnitId` with `LaunchKind::DeviceCallable` are
    /// fused into one `__global__` kernel by the megakernel codegen.
    pub compilation_unit: CompilationUnitId,
}

/// Ordered dispatch sequence built from one [`ExecutionPlan`].
///
/// Entries are sorted by `(step, subgraph_id)`. Walking them
/// sequentially respects all data dependencies (the solver
/// guaranteed `DependencyOrder` on the step assignment).
#[derive(Clone, Debug)]
pub struct DispatchSequence {
    pub entries: Vec<DispatchEntry>,
}

impl DispatchSequence {
    /// Build a dispatch sequence from a solved plan.
    ///
    /// Resolves every subgraph's impl to a typed `ImplDispatchKind`,
    /// determines the GEMM phase and norm position, and sorts by
    /// step order.
    pub fn from_plan(
        plan: &ExecutionPlan,
        library: &ImplementationLibrary,
        tile_graph: &TileGraph,
    ) -> Self {
        let assignment = &plan.assignment;

        // Collect subgraphs sorted by (step, subgraph_id).
        let mut scheduled: Vec<(u32, CompilationUnitId, SubgraphId)> = assignment
            .schedule
            .iter()
            .map(|(sg, slot)| (slot.step, slot.unit, *sg))
            .collect();
        scheduled.sort();

        // Track which RmsNorm tiles we've seen per layer to distinguish
        // attn_norm (first) from mlp_norm (second).
        let mut norm_count_per_layer: BTreeMap<u16, u32> = BTreeMap::new();

        let mut entries = Vec::with_capacity(scheduled.len());
        for (step, _unit, sg) in &scheduled {
            let impl_id = assignment.impls[sg];
            let imp = library.get(impl_id);
            let imp_name = imp.name();
            let launch_kind = imp.launch_kind();
            let compilation_unit = assignment.schedule[sg].unit;

            let claimed = assignment.tiles_in_subgraph(*sg);
            let layer = claimed
                .iter()
                .map(|t| tile_graph.nodes[t.0 as usize].layer)
                .next()
                .unwrap_or(0);

            let (kind, gemm_phase, fused_residual, is_attn_norm) = classify_impl(
                imp_name,
                &claimed,
                tile_graph,
                layer,
                &mut norm_count_per_layer,
            );

            entries.push(DispatchEntry {
                subgraph: *sg,
                impl_id,
                impl_name: imp_name,
                kind,
                layer,
                step: *step,
                launch_kind,
                fused_residual,
                gemm_phase,
                is_attn_norm,
                compilation_unit,
            });
        }

        DispatchSequence { entries }
    }

    /// Number of distinct schedule steps.
    pub fn num_steps(&self) -> u32 {
        self.entries.last().map(|e| e.step + 1).unwrap_or(0)
    }

    /// All entries for a given layer (useful for per-layer dispatch
    /// in the runtime path).
    pub fn entries_for_layer(&self, layer: u16) -> impl Iterator<Item = &DispatchEntry> {
        self.entries.iter().filter(move |e| e.layer == layer)
    }

    /// Whether every entry is a `HostCallback` launch (multi-launch
    /// plan, typical on sm_89).
    pub fn is_all_host_callback(&self) -> bool {
        self.entries
            .iter()
            .all(|e| e.launch_kind == LaunchKind::HostCallback)
    }

    /// Distinct compilation units referenced by entries.
    pub fn num_launches(&self) -> usize {
        // Noop entries don't count as launches.
        self.entries
            .iter()
            .filter(|e| e.kind != ImplDispatchKind::Noop)
            .count()
    }

    /// Compilation units in step order, deduplicated. Each unit
    /// appears once at the position of its first entry.
    pub fn units_in_order(&self) -> Vec<CompilationUnitId> {
        let mut seen = std::collections::BTreeSet::new();
        let mut units = Vec::new();
        for e in &self.entries {
            if e.kind != ImplDispatchKind::Noop && seen.insert(e.compilation_unit) {
                units.push(e.compilation_unit);
            }
        }
        units
    }

    /// All entries belonging to a given compilation unit, in step order.
    pub fn entries_for_unit(
        &self,
        unit: CompilationUnitId,
    ) -> impl Iterator<Item = &DispatchEntry> {
        self.entries.iter().filter(move |e| e.compilation_unit == unit)
    }

    /// Whether a compilation unit contains a single HostCallback entry
    /// (standalone launch) or multiple DeviceCallable entries (grouped
    /// mini-megakernel).
    pub fn unit_is_standalone(&self, unit: CompilationUnitId) -> bool {
        self.entries_for_unit(unit)
            .all(|e| e.launch_kind == LaunchKind::HostCallback)
    }

    /// Group entries by [`CompilationUnitId`], preserving step order
    /// within each group. Returns `(unit_id, entries)` pairs sorted
    /// by the first step in each unit.
    pub fn units(&self) -> Vec<(CompilationUnitId, Vec<&DispatchEntry>)> {
        let mut map: BTreeMap<CompilationUnitId, Vec<&DispatchEntry>> = BTreeMap::new();
        for entry in &self.entries {
            map.entry(entry.compilation_unit)
                .or_default()
                .push(entry);
        }
        let mut units: Vec<_> = map.into_iter().collect();
        // Sort by earliest step in each unit.
        units.sort_by_key(|(_, entries)| entries.first().map(|e| e.step).unwrap_or(0));
        units
    }

    /// Distinct non-Noop compilation units (actual kernel launches).
    pub fn distinct_launch_units(&self) -> std::collections::HashSet<CompilationUnitId> {
        self.entries
            .iter()
            .filter(|e| e.kind != ImplDispatchKind::Noop)
            .map(|e| e.compilation_unit)
            .collect()
    }
}

/// Classify an impl name + claimed tiles into a typed dispatch kind.
fn classify_impl(
    imp_name: &str,
    claimed: &[crate::lowering::tile_graph::TileId],
    tile_graph: &TileGraph,
    layer: u16,
    norm_count: &mut BTreeMap<u16, u32>,
) -> (ImplDispatchKind, Option<GemmPhase>, bool, Option<bool>) {
    // Strip `dc_` prefix from DeviceCallable wrapper names so the
    // classifier sees the underlying impl name.
    let imp_name = imp_name.strip_prefix("dc_").unwrap_or(imp_name);

    // Helper: does this claim include a ResidualAdd tile?
    let has_residual = claimed
        .iter()
        .any(|t| tile_graph.nodes[t.0 as usize].kind == TileKind::ResidualAdd);

    // Helper: map a GEMM TileKind to its GemmPhase.
    let kind_to_phase = |k: TileKind| -> Option<GemmPhase> {
        match k {
            TileKind::GemmQ => Some(GemmPhase::Q),
            TileKind::GemmK => Some(GemmPhase::K),
            TileKind::GemmV => Some(GemmPhase::V),
            TileKind::GemmOProj => Some(GemmPhase::OProj),
            TileKind::GemmGate => Some(GemmPhase::Gate),
            TileKind::GemmUp => Some(GemmPhase::Up),
            TileKind::GemmDown => Some(GemmPhase::Down),
            TileKind::GemmLmHead => Some(GemmPhase::LmHead),
            _ => None,
        }
    };

    // Extract GEMM phase from the claimed tiles, or (for standalone
    // BiasAdd impls that only claim the BiasAdd tile) from the BiasAdd
    // tile's upstream GEMM dep. Without this, a standalone bias_add
    // dispatch entry wouldn't know which buffer to target.
    let gemm_phase = claimed
        .iter()
        .find_map(|t| kind_to_phase(tile_graph.nodes[t.0 as usize].kind))
        .or_else(|| {
            claimed.iter().find_map(|t| {
                let node = &tile_graph.nodes[t.0 as usize];
                if node.kind != TileKind::BiasAdd {
                    return None;
                }
                node.deps
                    .iter()
                    .find_map(|d| kind_to_phase(tile_graph.nodes[d.0 as usize].kind))
            })
        });

    // Classify by impl name prefix → typed dispatch kind.
    // Detect fused QKV: claimed tiles include all of {GemmQ, GemmK, GemmV}.
    let has_q = claimed
        .iter()
        .any(|t| tile_graph.nodes[t.0 as usize].kind == TileKind::GemmQ);
    let has_k = claimed
        .iter()
        .any(|t| tile_graph.nodes[t.0 as usize].kind == TileKind::GemmK);
    let has_v = claimed
        .iter()
        .any(|t| tile_graph.nodes[t.0 as usize].kind == TileKind::GemmV);
    let is_fused_qkv = has_q && has_k && has_v;
    let has_bias_add = claimed
        .iter()
        .any(|t| tile_graph.nodes[t.0 as usize].kind == TileKind::BiasAdd);
    let is_fused_qkv_with_bias = is_fused_qkv && has_bias_add;

    // Detect fused gate+up: claimed tiles include both {GemmGate, GemmUp}.
    let has_gate = claimed
        .iter()
        .any(|t| tile_graph.nodes[t.0 as usize].kind == TileKind::GemmGate);
    let has_up = claimed
        .iter()
        .any(|t| tile_graph.nodes[t.0 as usize].kind == TileKind::GemmUp);
    let is_fused_gate_up = has_gate && has_up;

    let kind = if is_fused_qkv_with_bias {
        ImplDispatchKind::FusedQkvGemmWithBias
    } else if is_fused_qkv {
        ImplDispatchKind::FusedQkvGemm
    } else if is_fused_gate_up {
        ImplDispatchKind::FusedGateUpGemm
    } else if imp_name.ends_with("_with_bias") {
        // cublas_gemm_ex_<phase>_with_bias — fused {Gemm, BiasAdd}.
        // Must come before the generic cublas_gemm_ex prefix check.
        ImplDispatchKind::CublasGemmWithBias
    } else if imp_name.starts_with("cublas_gemm_ex") {
        ImplDispatchKind::CublasGemm
    } else if imp_name == "standalone_bias_add" {
        ImplDispatchKind::BiasAdd
    } else if imp_name.starts_with("cutlass_norm_") {
        let (tm, tn, _stages) = parse_cutlass_config(imp_name);
        ImplDispatchKind::CutlassNormGemm {
            tile_m: tm,
            tile_n: tn,
        }
    } else if imp_name.starts_with("cutlass_gate_silu_mul_") {
        let (tm, tn, stages) = parse_cutlass_config(imp_name);
        ImplDispatchKind::CutlassGemmSiluMul {
            tile_m: tm,
            tile_n: tn,
            stages,
        }
    } else if imp_name.starts_with("cutlass_gemv_") {
        ImplDispatchKind::CutlassGemv
    } else if imp_name.starts_with("cutlass_") {
        let (tm, tn, stages) = parse_cutlass_config(imp_name);
        ImplDispatchKind::CutlassGemm {
            tile_m: tm,
            tile_n: tn,
            stages,
        }
    } else if imp_name == "vllm_rs_rms_norm" {
        let count = norm_count.entry(layer).or_insert(0);
        let is_attn = *count == 0;
        *count += 1;
        return (ImplDispatchKind::RmsNorm, None, false, Some(is_attn));
    } else if imp_name == "vllm_rs_fused_qkv_rope_cache" {
        ImplDispatchKind::FusedQkvRopeCache
    } else if imp_name == "vllm_rs_prefill_rope_cache" {
        ImplDispatchKind::PrefillRopeCache
    } else if imp_name == "vllm_rs_rotary_embedding" {
        ImplDispatchKind::RotaryEmbedding
    } else if imp_name == "vllm_rs_silu_and_mul_fused" {
        ImplDispatchKind::SiluAndMul
    } else if imp_name == "flashinfer_standalone_fa2" {
        ImplDispatchKind::FlashInferAttention
    } else if imp_name == "flashinfer_standard_fa2" {
        ImplDispatchKind::FlashInferStandard
    } else if imp_name == "tk_attention_decode" {
        ImplDispatchKind::TkAttentionDecode
    } else if imp_name == "tk_attention_prefill" {
        ImplDispatchKind::TkAttentionPrefill
    } else if imp_name == "qkv_split_free"
        || imp_name == "kv_cache_write"
        || imp_name == "residual_add"
    {
        ImplDispatchKind::Noop
    } else if imp_name == "embedding_gather" {
        ImplDispatchKind::Embed
    } else {
        panic!(
            "DispatchSequence::classify_impl: unknown impl name {:?}",
            imp_name
        );
    };

    (kind, gemm_phase, has_residual, None)
}

/// Parse CUTLASS tile sizes from impl name.
/// E.g. "cutlass_qkv_64x64" → (64, 64), "cutlass_down_128x128_res" → (128, 128).
/// Parse tile dims + stages from impl name.
/// Format: `cutlass_{phase}_{M}x{N}_s{stages}` (e.g. `cutlass_qkv_128x128_s4`).
/// Falls back to 128×128 s4 for legacy names.
fn parse_cutlass_config(name: &str) -> (u32, u32, u32) {
    // Find {M}x{N} pattern
    let (tm, tn) = name
        .split('_')
        .find_map(|seg| {
            let (m, n) = seg.split_once('x')?;
            Some((m.parse::<u32>().ok()?, n.parse::<u32>().ok()?))
        })
        .unwrap_or((128, 128));
    // Find s{stages} pattern
    let stages = name
        .split('_')
        .find_map(|seg| seg.strip_prefix('s')?.parse::<u32>().ok())
        .unwrap_or(4);
    (tm, tn, stages)
}

/// Compact plan family summary for debugging / test visualization.
///
/// For each (seq_len, plan) in a `PlanFamily`, builds the
/// `DispatchSequence` and returns a multi-line ASCII summary
/// showing the step order, impl names, and per-step costs.
pub fn format_plan_family(
    family: &crate::lowering::solver::PlanFamily,
    library: &ImplementationLibrary,
    tile_graph: &TileGraph,
) -> String {
    let mut out = String::new();
    for (seq, plan) in family.iter() {
        let ds = DispatchSequence::from_plan(plan, library, tile_graph);
        let distinct_units = ds.distinct_launch_units();
        out.push_str(&format!(
            "\n=== seq={seq} ({} ops, {} kernel launches, {:.0}µs predicted) ===\n",
            ds.num_launches(),
            distinct_units.len(),
            plan.predicted_us,
        ));
        for entry in &ds.entries {
            if entry.kind == ImplDispatchKind::Noop {
                continue;
            }
            let layer_label = if entry.layer == u16::MAX {
                "pre".to_string()
            } else if entry.layer as u32 == tile_graph.num_layers as u32 {
                "post".to_string()
            } else {
                format!("layer {}", entry.layer)
            };
            out.push_str(&format!(
                "  step {:2}  U{:02}  {:<8}  {:?}  {}\n",
                entry.step, entry.compilation_unit.0, layer_label, entry.kind, entry.impl_name,
            ));
        }
    }
    out
}
