//! Pipeline stage descriptors and tile perimeters extracted from PTX.
//!
//! Each kernel is analyzed into a `PipelineStage` that captures its tiled
//! structure: loops, carry registers, pattern classification, and finalization.
//!
//! The `TilePerimeter` describes what flows in and out of each **tile iteration**
//! within a kernel's mainloop. This is the execution interface for the tile
//! pipeline: the compiler steps tiles through stages, passing carries forward.

use crate::fuse_cp_async::{CpAsyncClass, classify_cp_async_loads, identify_a_matrix_param};
use crate::parser::{
    AsyncCopyPort, CarryRegister, CarryRole, KernelProtocol, LoopDescriptor, PtxParser,
    TileIndexMap, analyze_carries, detect_loops, extract_tile_index_map,
};

/// How a reduction is performed within the kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReduceMethod {
    /// Warp-level shuffle only (e.g., small reductions).
    WarpShuffle,
    /// Block-level via shared memory + barrier.
    SharedMem,
    /// Warp shuffle followed by shared memory broadcast (typical for rms_norm).
    WarpShuffleAndSmem,
}

/// The computational pattern of a pipeline stage.
#[derive(Debug, Clone)]
pub enum StagePattern {
    /// Elementwise: no loops with accumulators, no MMA, no reduction.
    Pointwise,

    /// Reduction: loops that accumulate a scalar, then finalize via shuffle/SMEM.
    Reduction {
        /// Accumulator register names (e.g., ["%f83"]).
        accumulators: Vec<String>,
        /// How the reduction is performed.
        reduce_method: ReduceMethod,
    },

    /// Tiled GEMM: software-pipelined with cp.async + MMA instructions.
    TiledGemm {
        /// A-matrix cp.async load sites.
        a_loads: Vec<AsyncCopyPort>,
        /// B-matrix cp.async load sites.
        b_loads: Vec<AsyncCopyPort>,
        /// MMA accumulator registers.
        mma_accumulators: Vec<String>,
        /// Pipeline depth (number of SMEM buffer stages, e.g., 3).
        pipeline_depth: usize,
    },
}

/// Code that runs after the last loop iteration (epilogue, reduction finalize).
#[derive(Debug, Clone)]
pub struct FinalizationBlock {
    /// Line range in the source PTX (start, end) inclusive.
    pub line_range: (usize, usize),
    /// Carry registers consumed from the preceding loop.
    pub consumed_carries: Vec<String>,
}

/// Decomposition of a Reduction stage into three phases.
///
/// A reduction like rms_norm decomposes into:
/// 1. **Accumulate**: loops that accumulate a scalar (e.g., sum-of-squares)
/// 2. **Finalize**: post-accumulate code (warp shuffle, SMEM reduce, rsqrt → inv_rms)
/// 3. **Emit**: loops that produce per-element output using the finalized scalar
///
/// For fusion with a downstream GEMM:
/// - Accumulate + Finalize become the GEMM prologue
/// - Emit's per-element formula becomes a PointwiseComputation injected at cp.async sites
#[derive(Debug, Clone)]
pub struct ReductionDecomposition {
    /// Loops that perform accumulation (carry = accumulator register).
    pub accumulate_loops: Vec<LoopDescriptor>,
    /// Line range of the finalization code (shuffle, reduce, rsqrt).
    pub finalize_range: (usize, usize),
    /// Loops that emit per-element output using the finalized value.
    pub emit_loops: Vec<LoopDescriptor>,
    /// The finalized value register (e.g., "%f12" for inv_rms).
    /// Detected as the register loaded from SMEM after the reduction barrier.
    pub finalized_value_reg: String,
    /// The per-element formula extracted from the emit loop.
    /// Each entry is a PTX instruction from the emit loop body.
    pub emit_body_lines: Vec<String>,
}

// ===== Tile Perimeter =====

/// What kind of stage this is in the tile pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageKind {
    /// Reduction along a dimension (e.g., rms_norm along H).
    /// Produces a scalar per row, consumed by downstream stages.
    Reduction,
    /// Tiled GEMM (CUTLASS). The K-loop iterates over tiles,
    /// loading A and B via cp.async, computing MMA, accumulating.
    TiledGemm,
    /// Pointwise operation (e.g., SiLU+mul). No accumulation,
    /// applied inline at the downstream GEMM's A-load sites.
    Pointwise,
}

/// The tile shape of a stage (M-tile, N-tile, K-tile dimensions).
#[derive(Debug, Clone)]
pub struct TileShape {
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
}

/// A data port in the tile perimeter: one input or output per tile iteration.
#[derive(Debug, Clone)]
pub struct TilePort {
    /// Human-readable name (e.g., "A_tile", "B_tile", "inv_rms").
    pub name: String,
    /// Element type (e.g., "bf16", "f32").
    pub elem_type: String,
    /// How the data is accessed.
    pub access: TilePortAccess,
}

/// How a tile port's data is accessed.
#[derive(Debug, Clone)]
pub enum TilePortAccess {
    /// Loaded from GMEM via cp.async (CUTLASS A/B loads).
    CpAsync { sites: Vec<AsyncCopyPort> },
    /// Loaded from GMEM via explicit ld.global (norm input, weight).
    GlobalLoad,
    /// Scalar broadcast from SMEM (e.g., inv_rms per row).
    SmemScalar,
    /// Per-element formula applied inline (not materialized).
    /// The formula is the emit body from a reduction decomposition.
    InlineFormula { emit_body: Vec<String> },
    /// Written to GMEM (epilogue stores).
    GlobalStore,
}

/// State carried across tile iterations (or across rows within one stage).
#[derive(Debug, Clone)]
pub struct TileCarry {
    /// Human-readable name (e.g., "sum_sq", "mma_accum").
    pub name: String,
    /// The PTX registers holding this carry.
    pub registers: Vec<String>,
    /// Where the carry lives.
    pub storage: CarryStorage,
}

/// Where carry state is stored.
#[derive(Debug, Clone)]
pub enum CarryStorage {
    /// In registers (MMA accumulators, induction vars).
    Registers,
    /// In shared memory (reduction partial sums, inv_rms array).
    Smem,
}

/// Post-iteration finalization (runs after all K-tiles or all H-elements).
#[derive(Debug, Clone)]
pub struct TileFinalization {
    /// The PTX lines of the finalization code (extracted, not hand-written).
    pub lines: Vec<String>,
    /// Line range in the original PTX source.
    pub line_range: (usize, usize),
    /// Register holding the finalized value (e.g., inv_rms).
    pub result_reg: String,
}

/// The tile-level perimeter of a pipeline stage.
///
/// Describes what flows in and out of each tile iteration within a kernel's
/// mainloop. This is the execution interface — the pipeline compiler steps
/// tiles through stages, and the TilePerimeter defines the contract at each
/// stage boundary.
///
/// For a GEMM: one tile iteration = one K-step of the mainloop.
/// For a reduction: one tile iteration = the full reduction for one M-tile.
/// For pointwise: one tile iteration = one element (or vector of elements).
#[derive(Debug, Clone)]
pub struct TilePerimeter {
    /// Stage name (from PipelineStage).
    pub name: String,
    /// What kind of computation this stage performs.
    pub kind: StageKind,
    /// The tile dimensions.
    pub tile_shape: TileShape,
    /// How (ctaid.x, ctaid.y) map to (m_tile, n_tile).
    /// Extracted from the GEMM's PTX. `None` for non-GEMM stages
    /// (they inherit the tile index from their downstream GEMM).
    pub tile_index: Option<TileIndexMap>,
    /// Data flowing into each tile iteration.
    pub inputs: Vec<TilePort>,
    /// Data flowing out of each tile iteration.
    pub outputs: Vec<TilePort>,
    /// State carried across iterations (accumulators, pointers, etc.).
    pub carries: Vec<TileCarry>,
    /// Post-iteration finalization code.
    pub finalization: Option<TileFinalization>,
}

/// An edge connecting two stages in a fusible segment.
#[derive(Debug, Clone)]
pub enum StageEdge {
    /// Reduction output consumed at next GEMM's A-loads.
    /// The reduction's inv_rms goes to SMEM, and its per-element formula
    /// inlines at the GEMM's A-load sites.
    ReductionToGemm,
    /// GEMM output materialized to GMEM, next stage reads from GMEM.
    /// Required when dimensions change (e.g., gate_up → down).
    GmemMaterialization,
    /// Pointwise formula injected at next GEMM's A-loads.
    /// The formula is applied inline, no materialization.
    PointwiseToGemm,
}

/// A fusible segment: a chain of stages connected by edges.
///
/// Compiles into a single kernel launch. The tile executor steps tiles
/// through the stages, using carries to pass state and edges to define
/// how stages connect.
#[derive(Debug, Clone)]
pub struct FusibleSegment {
    pub stages: Vec<TilePerimeter>,
    pub edges: Vec<StageEdge>,
}

/// A single stage in a fused pipeline, extracted entirely from PTX analysis.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Stage name (user-provided, e.g., "norm", "gemm").
    pub name: String,
    /// The computational pattern.
    pub pattern: StagePattern,
    /// Detected loops.
    pub loops: Vec<LoopDescriptor>,
    /// Registers that carry state across loop iterations.
    pub carries: Vec<CarryRegister>,
    /// Post-loop finalization code (reduction finalize, epilogue).
    pub finalization: Option<FinalizationBlock>,
    /// The kernel's escape perimeter (from PtxParser::parse).
    pub protocol: KernelProtocol,
    /// The raw PTX source lines.
    pub source_lines: Vec<String>,
}

impl TilePerimeter {
    /// Extract a TilePerimeter from a PipelineStage.
    ///
    /// Dispatches to the appropriate extractor based on the stage pattern.
    pub fn from_stage(stage: &PipelineStage) -> Result<Self, String> {
        match &stage.pattern {
            StagePattern::TiledGemm {
                a_loads,
                b_loads,
                mma_accumulators,
                pipeline_depth,
            } => Self::from_gemm(stage, a_loads, b_loads, mma_accumulators, *pipeline_depth),
            StagePattern::Reduction {
                accumulators,
                reduce_method,
            } => Self::from_reduction(stage, accumulators, reduce_method),
            StagePattern::Pointwise => Self::from_pointwise(stage),
        }
    }

    /// Extract TilePerimeter from a TiledGemm stage (CUTLASS).
    fn from_gemm(
        stage: &PipelineStage,
        a_loads: &[AsyncCopyPort],
        b_loads: &[AsyncCopyPort],
        mma_accumulators: &[String],
        pipeline_depth: usize,
    ) -> Result<Self, String> {
        // Extract tile shape from the GEMM's mangled entry name
        let (tile_m, tile_n, tile_k) = extract_gemm_shape(&stage.protocol.name)?;

        // Extract tile index from PTX
        let lines: Vec<&str> = stage.source_lines.iter().map(|s| s.as_str()).collect();
        let tile_index = extract_tile_index_map(&lines);

        // Inputs: A-tile and B-tile via cp.async
        let inputs = vec![
            TilePort {
                name: "A_tile".into(),
                elem_type: "bf16".into(),
                access: TilePortAccess::CpAsync {
                    sites: a_loads.to_vec(),
                },
            },
            TilePort {
                name: "B_tile".into(),
                elem_type: "bf16".into(),
                access: TilePortAccess::CpAsync {
                    sites: b_loads.to_vec(),
                },
            },
        ];

        // Outputs: epilogue stores to GMEM
        let outputs = vec![TilePort {
            name: "D_tile".into(),
            elem_type: "bf16".into(),
            access: TilePortAccess::GlobalStore,
        }];

        // Carries: MMA accumulators + pipeline buffer state
        let mut carries = vec![TileCarry {
            name: "mma_accum".into(),
            registers: mma_accumulators.to_vec(),
            storage: CarryStorage::Registers,
        }];

        // Add induction vars and tile pointers from the stage's carries
        let k_pointers: Vec<String> = stage
            .carries
            .iter()
            .filter(|c| c.role == CarryRole::TilePointer)
            .map(|c| c.register.clone())
            .collect();
        if !k_pointers.is_empty() {
            carries.push(TileCarry {
                name: "k_pointer".into(),
                registers: k_pointers,
                storage: CarryStorage::Registers,
            });
        }

        let buffer_regs: Vec<String> = stage
            .carries
            .iter()
            .filter(|c| c.role == CarryRole::BufferState)
            .map(|c| c.register.clone())
            .collect();
        if !buffer_regs.is_empty() {
            carries.push(TileCarry {
                name: format!("buffer_state_{}stage", pipeline_depth),
                registers: buffer_regs,
                storage: CarryStorage::Registers,
            });
        }

        Ok(TilePerimeter {
            name: stage.name.clone(),
            kind: StageKind::TiledGemm,
            tile_shape: TileShape {
                tile_m,
                tile_n,
                tile_k,
            },
            tile_index,
            inputs,
            outputs,
            carries,
            finalization: None, // GEMM epilogue is handled separately
        })
    }

    /// Extract TilePerimeter from a Reduction stage (rms_norm).
    fn from_reduction(
        stage: &PipelineStage,
        accumulators: &[String],
        _reduce_method: &ReduceMethod,
    ) -> Result<Self, String> {
        let decomp = stage
            .decompose_reduction()
            .ok_or("reduction stage does not decompose")?;

        // For a reduction, tile_shape is (tile_m, 1, H) — reduces full hidden dim
        // tile_m is inherited from the downstream GEMM, so we use 0 as placeholder
        let tile_shape = TileShape {
            tile_m: 0, // inherited from downstream GEMM
            tile_n: 1,
            tile_k: 0, // full hidden dimension (runtime)
        };

        // Inputs: the tensors being reduced
        let inputs = vec![
            TilePort {
                name: "input".into(),
                elem_type: "bf16".into(),
                access: TilePortAccess::GlobalLoad,
            },
            TilePort {
                name: "weight".into(),
                elem_type: "bf16".into(),
                access: TilePortAccess::GlobalLoad,
            },
        ];

        // Outputs: inv_rms scalar (to SMEM) + per-element formula (inline)
        let outputs = vec![
            TilePort {
                name: "inv_rms".into(),
                elem_type: "f32".into(),
                access: TilePortAccess::SmemScalar,
            },
            TilePort {
                name: "normalized".into(),
                elem_type: "bf16".into(),
                access: TilePortAccess::InlineFormula {
                    emit_body: decomp.emit_body_lines.clone(),
                },
            },
        ];

        // Carries: sum_sq accumulator
        let carries = vec![TileCarry {
            name: "sum_sq".into(),
            registers: accumulators.to_vec(),
            storage: CarryStorage::Registers,
        }];

        // Finalization: the warp shuffle + SMEM reduce + rsqrt → inv_rms
        let finalization = if !decomp.finalized_value_reg.is_empty() {
            Some(TileFinalization {
                lines: stage.source_lines[decomp.finalize_range.0
                    ..=decomp.finalize_range.1.min(stage.source_lines.len() - 1)]
                    .iter()
                    .map(|l| l.trim().to_string())
                    .collect(),
                line_range: decomp.finalize_range,
                result_reg: decomp.finalized_value_reg.clone(),
            })
        } else {
            None
        };

        Ok(TilePerimeter {
            name: stage.name.clone(),
            kind: StageKind::Reduction,
            tile_shape,
            tile_index: None, // inherited from downstream GEMM
            inputs,
            outputs,
            carries,
            finalization,
        })
    }

    /// Extract TilePerimeter from a Pointwise stage (SiLU+mul).
    fn from_pointwise(stage: &PipelineStage) -> Result<Self, String> {
        // Pointwise: no accumulation, no carries, tile shape inherited
        let tile_shape = TileShape {
            tile_m: 0, // inherited
            tile_n: 0,
            tile_k: 0,
        };

        // Inputs from GMEM
        let inputs = vec![TilePort {
            name: "input".into(),
            elem_type: "bf16".into(),
            access: TilePortAccess::GlobalLoad,
        }];

        // Output is an inline formula (applied at downstream GEMM's A-loads)
        let outputs = vec![TilePort {
            name: "activated".into(),
            elem_type: "bf16".into(),
            access: TilePortAccess::InlineFormula {
                emit_body: vec![], // filled in when composing with GEMM
            },
        }];

        Ok(TilePerimeter {
            name: stage.name.clone(),
            kind: StageKind::Pointwise,
            tile_shape,
            tile_index: None,
            inputs,
            outputs,
            carries: vec![],
            finalization: None,
        })
    }
}

/// Extract (tile_m, tile_n, tile_k) from a CUTLASS mangled entry name.
fn extract_gemm_shape(name: &str) -> Result<(u32, u32, u32), String> {
    let marker = "GemmShapeILi";
    let idx = name.find(marker).ok_or_else(|| {
        format!(
            "no GemmShapeILi in entry name: {}",
            &name[..name.len().min(80)]
        )
    })?;
    let rest = &name[idx + marker.len()..];
    let mut dims = Vec::new();
    let mut cur = rest;
    for _ in 0..3 {
        let end = cur.find('E').ok_or("malformed GemmShape")?;
        dims.push(
            cur[..end]
                .parse::<u32>()
                .map_err(|e| format!("bad dim: {e}"))?,
        );
        cur = &cur[end + 1..];
        if cur.starts_with("Li") {
            cur = &cur[2..];
        }
    }
    Ok((dims[0], dims[1], dims[2]))
}

impl PipelineStage {
    /// Extract a pipeline stage descriptor from PTX source.
    ///
    /// Combines:
    /// 1. `PtxParser::parse()` for the escape perimeter
    /// 2. `detect_loops()` for loop structure
    /// 3. `analyze_carries()` for carry registers per loop
    /// 4. Pattern classification based on the above
    /// 5. Finalization block detection
    pub fn from_ptx(name: &str, source: &str, entry_hint: Option<&str>) -> Result<Self, String> {
        // If an entry hint is provided, extract that specific entry first,
        // then parse the single-entry result. This ensures we analyze the
        // correct variant (e.g., bf16 instead of f32 for multi-entry PTX).
        let pre_extracted;
        let effective_source = if let Some(hint) = entry_hint {
            let raw_lines: Vec<&str> = source.lines().collect();
            let entry_count = raw_lines
                .iter()
                .filter(|l| l.contains(".entry") && l.contains('('))
                .count();
            if entry_count > 1 {
                pre_extracted = crate::extract::extract_entry(source, hint)
                    .map_err(|e| format!("entry hint '{hint}': {e}"))?;
                &pre_extracted
            } else {
                source
            }
        } else {
            source
        };

        let protocol = PtxParser::parse(effective_source)?;

        // For multi-entry PTX (when no hint was provided), extract the first entry
        // so loop/carry analysis only sees one kernel.
        let raw_lines: Vec<&str> = effective_source.lines().collect();
        let entry_count = raw_lines
            .iter()
            .filter(|l| l.contains(".entry") && l.contains('('))
            .count();
        let extracted;
        let lines: Vec<&str> = if entry_count > 1 {
            let substr = if protocol.name.len() > 20 {
                &protocol.name[..20]
            } else {
                &protocol.name
            };
            extracted = crate::extract::extract_entry(effective_source, substr)
                .map_err(|e| format!("multi-entry: {e}"))?;
            extracted.lines().collect()
        } else {
            raw_lines
        };

        let loops = detect_loops(&lines);

        // Collect carries from all loops
        let mut all_carries: Vec<CarryRegister> = Vec::new();
        for lp in &loops {
            let loop_carries = analyze_carries(&lines, lp);
            for carry in loop_carries {
                if !all_carries.iter().any(|c| c.register == carry.register) {
                    all_carries.push(carry);
                }
            }
        }

        // Classify the pattern
        let pattern = classify_pattern(&protocol, &lines, &loops, &all_carries)?;

        // Detect finalization block
        let finalization = detect_finalization(&lines, &loops, &all_carries);

        Ok(PipelineStage {
            name: name.to_string(),
            pattern,
            loops,
            carries: all_carries,
            finalization,
            protocol,
            source_lines: lines.iter().map(|l| l.to_string()).collect(),
        })
    }

    /// Decompose a Reduction stage into accumulate / finalize / emit phases.
    ///
    /// Returns `None` if the stage is not a Reduction.
    pub fn decompose_reduction(&self) -> Option<ReductionDecomposition> {
        match &self.pattern {
            StagePattern::Reduction { accumulators, .. } => decompose_reduction_impl(
                &self.source_lines,
                &self.loops,
                accumulators,
                &self.protocol,
            ),
            _ => None,
        }
    }
}

/// Decompose a reduction kernel into its three phases.
fn decompose_reduction_impl(
    source_lines: &[String],
    loops: &[LoopDescriptor],
    accumulators: &[String],
    protocol: &KernelProtocol,
) -> Option<ReductionDecomposition> {
    let lines: Vec<&str> = source_lines.iter().map(|s| s.as_str()).collect();

    // Classify loops as accumulate vs emit:
    // - Accumulate loops: contain accumulator carries (fma/add.f32 self-modifying)
    // - Emit loops: contain st.global (output stores)
    let mut accumulate_loops = Vec::new();
    let mut emit_loops = Vec::new();

    for lp in loops {
        let body_has_accumulator = {
            let carries = analyze_carries(&lines, lp);
            carries.iter().any(|c| accumulators.contains(&c.register))
        };
        let body_has_store = (lp.body_range.0..=lp.body_range.1)
            .any(|i| i < lines.len() && lines[i].contains("st.global"));

        if body_has_accumulator {
            accumulate_loops.push(lp.clone());
        } else if body_has_store {
            emit_loops.push(lp.clone());
        }
    }

    if accumulate_loops.is_empty() {
        return None;
    }

    // Finalize range: between last accumulate loop back-edge and first emit loop header
    // (or end of kernel if no emit loops)
    let last_accum_end = accumulate_loops
        .iter()
        .map(|l| l.backedge_line)
        .max()
        .unwrap_or(0);
    let first_emit_start = emit_loops.iter().map(|l| l.header_line).min();

    let finalize_start = last_accum_end + 1;
    let finalize_end = match first_emit_start {
        Some(emit_line) => emit_line.saturating_sub(1),
        None => lines.len().saturating_sub(1),
    };

    // Detect the finalized value register: look for ld.shared.f32 in the finalize block,
    // which loads the broadcast result of the reduction.
    let mut finalized_value_reg = String::new();
    for i in finalize_start..=finalize_end.min(lines.len() - 1) {
        let trimmed = lines[i].trim();
        if trimmed.contains("ld.shared.f32") || trimmed.contains("ld.shared.b32") {
            // Extract dest register
            let parts: Vec<&str> = trimmed
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 2 {
                let reg = parts[1].trim_end_matches(',');
                if reg.starts_with('%') {
                    finalized_value_reg = reg.to_string();
                    // Take the LAST ld.shared (the one after the barrier)
                }
            }
        }
    }

    // Extract emit loop body lines (the per-element computation)
    let mut emit_body_lines = Vec::new();
    if let Some(emit_lp) = emit_loops.first() {
        for i in emit_lp.header_line..=emit_lp.backedge_line {
            if i < lines.len() {
                let trimmed = lines[i].trim();
                // Skip labels, loop control (setp for loop condition, bra)
                if trimmed.is_empty()
                    || trimmed.starts_with('$')
                    || trimmed.starts_with("//")
                    || (trimmed.contains("bra") && trimmed.contains(&emit_lp.header_label))
                {
                    continue;
                }
                emit_body_lines.push(trimmed.to_string());
            }
        }
    }

    Some(ReductionDecomposition {
        accumulate_loops,
        finalize_range: (finalize_start, finalize_end),
        emit_loops,
        finalized_value_reg,
        emit_body_lines,
    })
}

/// Classify the kernel's computational pattern from its extracted metadata.
fn classify_pattern(
    protocol: &KernelProtocol,
    lines: &[&str],
    loops: &[LoopDescriptor],
    carries: &[CarryRegister],
) -> Result<StagePattern, String> {
    // TiledGemm: has MMA instructions + async loads
    if protocol.has_mma && !protocol.async_loads.is_empty() {
        let reg_to_param = PtxParser::trace_param_registers_pub(lines, &protocol.params);

        // Classify cp.async into A-matrix vs B-matrix loads
        let a_param_name = identify_a_matrix_param(lines, &reg_to_param, "").unwrap_or_default();

        let a_addr_regs: Vec<String> = reg_to_param
            .iter()
            .filter(|(_, p)| **p == a_param_name)
            .map(|(r, _)| r.clone())
            .collect();

        let classifications = classify_cp_async_loads(lines, &a_addr_regs);

        let mut a_loads = Vec::new();
        let mut b_loads = Vec::new();
        for port in &protocol.async_loads {
            // protocol.async_loads uses 1-indexed lines; classify uses 0-indexed
            let line_0 = port.line.saturating_sub(1);
            match classifications.get(&line_0) {
                Some(CpAsyncClass::AMatrix) => a_loads.push(port.clone()),
                Some(CpAsyncClass::BMatrix) => b_loads.push(port.clone()),
                _ => b_loads.push(port.clone()), // unknown → treat as B (preserve)
            }
        }

        let mma_accumulators: Vec<String> = carries
            .iter()
            .filter(|c| c.role == CarryRole::MmaAccumulator)
            .map(|c| c.register.clone())
            .collect();

        // Pipeline depth: detected from buffer state carries (selp-based rotation)
        let buffer_states = carries
            .iter()
            .filter(|c| c.role == CarryRole::BufferState)
            .count();
        let pipeline_depth = if buffer_states > 0 { 3 } else { 1 };

        return Ok(StagePattern::TiledGemm {
            a_loads,
            b_loads,
            mma_accumulators,
            pipeline_depth,
        });
    }

    // Reduction: has loops with accumulator carries + shfl or SMEM reduction pattern
    let has_accumulator = carries.iter().any(|c| c.role == CarryRole::Accumulator);
    if has_accumulator && !loops.is_empty() {
        let has_shfl = lines.iter().any(|l| l.contains("shfl.sync"));
        let has_smem_reduce =
            protocol.smem_stores > 0 && !protocol.barriers.is_empty() && protocol.smem_loads > 0;

        let reduce_method = match (has_shfl, has_smem_reduce) {
            (true, true) => ReduceMethod::WarpShuffleAndSmem,
            (true, false) => ReduceMethod::WarpShuffle,
            (false, true) => ReduceMethod::SharedMem,
            (false, false) => ReduceMethod::WarpShuffle, // fallback
        };

        let accumulators: Vec<String> = carries
            .iter()
            .filter(|c| c.role == CarryRole::Accumulator)
            .map(|c| c.register.clone())
            .collect();

        return Ok(StagePattern::Reduction {
            accumulators,
            reduce_method,
        });
    }

    // Default: Pointwise
    Ok(StagePattern::Pointwise)
}

/// Detect the finalization block: code between the last loop's back-edge and
/// either the next loop header or `ret`/`exit`.
fn detect_finalization(
    lines: &[&str],
    loops: &[LoopDescriptor],
    carries: &[CarryRegister],
) -> Option<FinalizationBlock> {
    if loops.is_empty() {
        return None;
    }

    // Find the last loop (by back-edge line)
    let last_loop = loops.iter().max_by_key(|l| l.backedge_line)?;
    let start = last_loop.backedge_line + 1;

    // Find the end: next `ret`, `exit`, or end of source
    let mut end = lines.len().saturating_sub(1);
    for i in start..lines.len() {
        let trimmed = lines[i].trim();
        if trimmed == "ret;" || trimmed == "exit;" || trimmed == "}" {
            end = i;
            break;
        }
    }

    if start >= end {
        return None;
    }

    // The finalization consumes the carries from the last loop
    let consumed: Vec<String> = carries
        .iter()
        .filter(|c| c.role == CarryRole::Accumulator)
        .map(|c| c.register.clone())
        .collect();

    Some(FinalizationBlock {
        line_range: (start, end),
        consumed_carries: consumed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_extract_rms_norm() {
        let ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let stage = PipelineStage::from_ptx("rms_norm", ptx, None).expect("extraction failed");

        // Should be classified as Reduction
        match &stage.pattern {
            StagePattern::Reduction {
                accumulators,
                reduce_method,
            } => {
                assert!(
                    !accumulators.is_empty(),
                    "should have at least one accumulator"
                );
                assert_eq!(
                    *reduce_method,
                    ReduceMethod::WarpShuffleAndSmem,
                    "rms_norm uses warp shuffle + SMEM broadcast"
                );
            }
            other => panic!("expected Reduction, got {:?}", other),
        }

        // Should have loops
        assert!(
            !stage.loops.is_empty(),
            "rms_norm should have detected loops"
        );

        // Should have carries
        assert!(
            !stage.carries.is_empty(),
            "rms_norm should have carry registers"
        );

        // Should have a finalization block
        assert!(
            stage.finalization.is_some(),
            "rms_norm should have a finalization block"
        );
    }

    #[test]
    fn stage_extract_cutlass_gemm() {
        let ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.ptx");
        let stage = PipelineStage::from_ptx("gemm", ptx, None).expect("extraction failed");

        match &stage.pattern {
            StagePattern::TiledGemm {
                a_loads,
                b_loads,
                mma_accumulators,
                pipeline_depth,
            } => {
                assert!(!a_loads.is_empty(), "should have A-matrix loads");
                assert!(!b_loads.is_empty(), "should have B-matrix loads");
                assert!(!mma_accumulators.is_empty(), "should have MMA accumulators");
                assert!(
                    *pipeline_depth >= 2,
                    "CUTLASS should have multi-stage pipeline, got {}",
                    pipeline_depth
                );
            }
            other => panic!("expected TiledGemm, got {:?}", other),
        }
    }

    #[test]
    fn stage_extract_silu_mul() {
        let ptx = include_str!("../../ptx-fusion/kernels/vllm_silu_mul.ptx");
        let stage = PipelineStage::from_ptx("silu_mul", ptx, None).expect("extraction failed");

        // silu_mul has loops (it's vectorized) but they shouldn't have
        // accumulator carries — the loop just processes elements independently.
        // It should be classified as Pointwise or at least not as Reduction/TiledGemm.
        match &stage.pattern {
            StagePattern::Pointwise => {} // ideal
            StagePattern::Reduction { .. } => {
                // silu_mul does have a loop with an induction var, but if it
                // somehow gets classified as reduction, that's a bug we'll fix.
                // For now, let's see what happens.
                panic!("silu_mul should not be Reduction");
            }
            StagePattern::TiledGemm { .. } => {
                panic!("silu_mul should not be TiledGemm");
            }
        }
    }

    #[test]
    fn stage_extract_scale() {
        let ptx = include_str!("../../ptx-fusion/kernels/scale.ptx");
        let stage = PipelineStage::from_ptx("scale", ptx, None).expect("extraction failed");

        match &stage.pattern {
            StagePattern::Pointwise => {} // correct
            other => panic!("scale should be Pointwise, got {:?}", other),
        }
    }

    #[test]
    fn decompose_rms_norm_reduction() {
        let ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let stage = PipelineStage::from_ptx("rms_norm", ptx, None).expect("extraction failed");

        let decomp = stage
            .decompose_reduction()
            .expect("rms_norm should decompose as a reduction");

        // Should have accumulate loops (sum-of-squares)
        assert!(
            !decomp.accumulate_loops.is_empty(),
            "should have accumulate loops"
        );

        // Should have emit loops (output = input * inv_rms * weight)
        assert!(!decomp.emit_loops.is_empty(), "should have emit loops");

        // Finalize range should be between accumulate and emit
        assert!(
            decomp.finalize_range.0 < decomp.finalize_range.1,
            "finalize range should be non-empty: {:?}",
            decomp.finalize_range
        );
        let last_accum = decomp.accumulate_loops.last().unwrap().backedge_line;
        let first_emit = decomp.emit_loops.first().unwrap().header_line;
        assert!(
            decomp.finalize_range.0 > last_accum,
            "finalize should start after last accumulate loop"
        );
        assert!(
            decomp.finalize_range.1 < first_emit,
            "finalize should end before first emit loop"
        );

        // Should have detected the finalized value register (inv_rms loaded from SMEM)
        assert!(
            decomp.finalized_value_reg.starts_with('%'),
            "should detect finalized value register, got: {:?}",
            decomp.finalized_value_reg
        );

        // Emit body should contain mul.f32 instructions (input * inv_rms, result * weight)
        let has_mul = decomp.emit_body_lines.iter().any(|l| l.contains("mul.f32"));
        assert!(
            has_mul,
            "emit body should contain mul.f32 for normalization"
        );

        // Emit body should contain st.global (output stores)
        let has_store = decomp
            .emit_body_lines
            .iter()
            .any(|l| l.contains("st.global"));
        assert!(
            has_store,
            "emit body should contain st.global output stores"
        );

        // Emit body should contain ld.global (input + weight loads)
        let ld_count = decomp
            .emit_body_lines
            .iter()
            .filter(|l| l.contains("ld.global"))
            .count();
        assert!(
            ld_count >= 2,
            "emit body should load input and weight from GMEM, got {} loads",
            ld_count
        );
    }

    #[test]
    fn dump_rms_norm_decomposition() {
        let ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let stage = PipelineStage::from_ptx("rms_norm", ptx, None).expect("extraction failed");
        let decomp = stage.decompose_reduction().expect("decompose failed");

        eprintln!(
            "=== Accumulate loops: {} ===",
            decomp.accumulate_loops.len()
        );
        for (i, lp) in decomp.accumulate_loops.iter().enumerate() {
            eprintln!(
                "  loop {i}: lines {}..{} label={}",
                lp.header_line, lp.backedge_line, lp.header_label
            );
            for line_idx in lp.header_line..=lp.backedge_line {
                if line_idx < stage.source_lines.len() {
                    eprintln!(
                        "    {:4}: {}",
                        line_idx,
                        stage.source_lines[line_idx].trim()
                    );
                }
            }
        }

        eprintln!(
            "=== Finalize range: {}..{} ===",
            decomp.finalize_range.0, decomp.finalize_range.1
        );
        for i in decomp.finalize_range.0..=decomp.finalize_range.1.min(stage.source_lines.len() - 1)
        {
            eprintln!("  {:4}: {}", i, stage.source_lines[i].trim());
        }

        eprintln!(
            "=== Finalized value register: {} ===",
            decomp.finalized_value_reg
        );

        eprintln!("=== Emit body: {} lines ===", decomp.emit_body_lines.len());
        for (i, line) in decomp.emit_body_lines.iter().enumerate() {
            eprintln!("  {i:3}: {line}");
        }
    }

    // ── TilePerimeter extraction tests ──

    #[test]
    fn tile_perimeter_from_cutlass_gemm() {
        let ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx");
        let stage = PipelineStage::from_ptx("gemm", ptx, None).expect("parse");
        let perim = TilePerimeter::from_stage(&stage).expect("extract perimeter");

        assert_eq!(perim.kind, StageKind::TiledGemm);
        assert_eq!(perim.tile_shape.tile_m, 64);
        assert_eq!(perim.tile_shape.tile_n, 128);
        assert_eq!(perim.tile_shape.tile_k, 32);

        // Should have tile index extracted
        assert!(perim.tile_index.is_some(), "GEMM should have tile index");
        let ti = perim.tile_index.as_ref().unwrap();
        assert!(!ti.m_tile_reg.is_empty());
        assert!(!ti.n_tile_reg.is_empty());

        // Should have A and B inputs
        assert_eq!(perim.inputs.len(), 2);
        assert_eq!(perim.inputs[0].name, "A_tile");
        assert_eq!(perim.inputs[1].name, "B_tile");
        match &perim.inputs[0].access {
            TilePortAccess::CpAsync { sites } => {
                assert!(!sites.is_empty(), "should have A-load cp.async sites");
            }
            other => panic!("A_tile should be CpAsync, got {:?}", other),
        }

        // Should have output
        assert_eq!(perim.outputs.len(), 1);
        assert_eq!(perim.outputs[0].name, "D_tile");

        // Should have MMA accumulators as carries
        assert!(!perim.carries.is_empty());
        let mma_carry = perim.carries.iter().find(|c| c.name == "mma_accum");
        assert!(mma_carry.is_some(), "should have MMA accumulator carry");
        assert!(
            !mma_carry.unwrap().registers.is_empty(),
            "MMA carry should have registers"
        );

        eprintln!("GEMM TilePerimeter:");
        eprintln!(
            "  tile: {}x{}x{}",
            perim.tile_shape.tile_m, perim.tile_shape.tile_n, perim.tile_shape.tile_k
        );
        eprintln!("  inputs: {}", perim.inputs.len());
        eprintln!("  outputs: {}", perim.outputs.len());
        eprintln!("  carries: {}", perim.carries.len());
        for c in &perim.carries {
            eprintln!("    {}: {} regs", c.name, c.registers.len());
        }
    }

    #[test]
    fn tile_perimeter_from_rms_norm() {
        let ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let stage = PipelineStage::from_ptx("rms_norm", ptx, None).expect("parse");
        let perim = TilePerimeter::from_stage(&stage).expect("extract perimeter");

        assert_eq!(perim.kind, StageKind::Reduction);
        assert!(perim.tile_index.is_none(), "reduction inherits tile index");

        // Should have input and weight ports
        assert!(perim.inputs.len() >= 2, "need input + weight");

        // Should have inv_rms scalar output and inline formula output
        assert_eq!(perim.outputs.len(), 2);
        let inv_rms_port = perim.outputs.iter().find(|p| p.name == "inv_rms");
        assert!(inv_rms_port.is_some(), "should have inv_rms output");
        let formula_port = perim.outputs.iter().find(|p| p.name == "normalized");
        assert!(formula_port.is_some(), "should have normalized output");
        match &formula_port.unwrap().access {
            TilePortAccess::InlineFormula { emit_body } => {
                assert!(!emit_body.is_empty(), "emit body should have instructions");
                let has_mul = emit_body.iter().any(|l| l.contains("mul.f32"));
                assert!(has_mul, "emit body should contain mul.f32");
            }
            other => panic!("normalized should be InlineFormula, got {:?}", other),
        }

        // Should have sum_sq carry
        assert!(!perim.carries.is_empty());
        assert_eq!(perim.carries[0].name, "sum_sq");

        // Should have finalization (warp shuffle → rsqrt)
        assert!(perim.finalization.is_some(), "should have finalization");
        let fin = perim.finalization.as_ref().unwrap();
        assert!(!fin.result_reg.is_empty(), "should have result register");
        let fin_text = fin.lines.join("\n");
        assert!(
            fin_text.contains("rsqrt"),
            "finalization should contain rsqrt"
        );

        eprintln!("Reduction TilePerimeter:");
        eprintln!(
            "  carries: {} ({} regs)",
            perim.carries[0].name,
            perim.carries[0].registers.len()
        );
        eprintln!(
            "  finalization: {} lines, result={}",
            fin.lines.len(),
            fin.result_reg
        );
        eprintln!(
            "  emit body: {} lines",
            match &formula_port.unwrap().access {
                TilePortAccess::InlineFormula { emit_body } => emit_body.len(),
                _ => 0,
            }
        );
    }

    #[test]
    fn tile_perimeter_from_silu_mul() {
        let ptx = include_str!("../../ptx-fusion/kernels/vllm_silu_mul.ptx");
        let stage = PipelineStage::from_ptx("silu_mul", ptx, None).expect("parse");
        let perim = TilePerimeter::from_stage(&stage).expect("extract perimeter");

        assert_eq!(perim.kind, StageKind::Pointwise);
        assert!(perim.tile_index.is_none());
        assert!(perim.carries.is_empty(), "pointwise has no carries");
        assert!(
            perim.finalization.is_none(),
            "pointwise has no finalization"
        );
    }

    #[test]
    fn tile_perimeter_all_cutlass_configs() {
        let configs = [
            (
                "64x64x32",
                include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.ptx"),
                64,
                64,
                32,
            ),
            (
                "64x128x32",
                include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx"),
                64,
                128,
                32,
            ),
            (
                "128x128x32",
                include_str!("../../ptx-fusion/kernels/cutlass_bf16_128x128x32_sm89.ptx"),
                128,
                128,
                32,
            ),
            (
                "128x128x64",
                include_str!("../../ptx-fusion/kernels/cutlass_bf16_128x128x64_sm89.ptx"),
                128,
                128,
                64,
            ),
        ];
        for (label, ptx, exp_m, exp_n, exp_k) in configs {
            let stage = PipelineStage::from_ptx(label, ptx, None)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            let perim =
                TilePerimeter::from_stage(&stage).unwrap_or_else(|e| panic!("{label}: {e}"));

            assert_eq!(perim.tile_shape.tile_m, exp_m, "{label} tile_m");
            assert_eq!(perim.tile_shape.tile_n, exp_n, "{label} tile_n");
            assert_eq!(perim.tile_shape.tile_k, exp_k, "{label} tile_k");
            assert!(perim.tile_index.is_some(), "{label} should have tile index");
            assert!(!perim.carries.is_empty(), "{label} should have carries");

            eprintln!(
                "{label}: {}x{}x{}, {} carries, {} A-loads",
                perim.tile_shape.tile_m,
                perim.tile_shape.tile_n,
                perim.tile_shape.tile_k,
                perim.carries.len(),
                match &perim.inputs[0].access {
                    TilePortAccess::CpAsync { sites } => sites.len(),
                    _ => 0,
                }
            );
        }
    }

    #[test]
    fn decompose_pointwise_returns_none() {
        let ptx = include_str!("../../ptx-fusion/kernels/scale.ptx");
        let stage = PipelineStage::from_ptx("scale", ptx, None).expect("extraction failed");
        assert!(
            stage.decompose_reduction().is_none(),
            "pointwise stage should not decompose as reduction"
        );
    }

    #[test]
    fn classify_fused_add_rms_norm() {
        let ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let stage = PipelineStage::from_ptx(
            "fused_add_rms_norm",
            ptx,
            Some("fused_add_rms_norm_kernelI13__nv_bfloat16"),
        )
        .expect("extraction failed");

        eprintln!("pattern: {:?}", stage.pattern);
        eprintln!("params: {} total", stage.protocol.params.len());
        for (i, p) in stage.protocol.params.iter().enumerate() {
            eprintln!("  param_{i}: {} ({})", p.name, p.ptx_type);
        }
        eprintln!("loops: {}", stage.loops.len());
        eprintln!("carries: {}", stage.carries.len());
        for c in &stage.carries {
            eprintln!("  carry: {} ({:?})", c.register, c.role);
        }

        match &stage.pattern {
            StagePattern::Reduction { .. } => {}
            other => panic!("fused_add_rms_norm should be Reduction, got {:?}", other),
        }
    }

    #[test]
    fn decompose_fused_add_rms_norm() {
        let ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let stage = PipelineStage::from_ptx(
            "fused_add_rms_norm",
            ptx,
            Some("fused_add_rms_norm_kernelI13__nv_bfloat16"),
        )
        .expect("extraction failed");

        let decomp = stage
            .decompose_reduction()
            .expect("fused_add_rms_norm should decompose as a reduction");

        eprintln!(
            "=== Accumulate loops: {} ===",
            decomp.accumulate_loops.len()
        );
        for (i, lp) in decomp.accumulate_loops.iter().enumerate() {
            eprintln!(
                "  loop {i}: lines {}..{} label={}",
                lp.header_line, lp.backedge_line, lp.header_label
            );
        }

        eprintln!(
            "=== Finalize range: {}..{} ===",
            decomp.finalize_range.0, decomp.finalize_range.1
        );
        eprintln!(
            "=== Finalized value register: {} ===",
            decomp.finalized_value_reg
        );
        eprintln!("=== Emit loops: {} ===", decomp.emit_loops.len());
        eprintln!("=== Emit body: {} lines ===", decomp.emit_body_lines.len());
        for (i, line) in decomp.emit_body_lines.iter().enumerate() {
            eprintln!("  {i:3}: {line}");
        }

        // Accumulation loop should contain st.global (writeback to residual)
        let accum_lp = &decomp.accumulate_loops[0];
        let has_store = (accum_lp.body_range.0..=accum_lp.body_range.1)
            .any(|i| i < stage.source_lines.len() && stage.source_lines[i].contains("st.global"));
        eprintln!("  accum loop has st.global (writeback): {has_store}");

        assert!(
            !decomp.accumulate_loops.is_empty(),
            "should have accumulate loops"
        );
        assert!(
            !decomp.finalized_value_reg.is_empty(),
            "should detect finalized value register"
        );
        assert!(
            !decomp.emit_loops.is_empty(),
            "should have emit loops (normalize pass)"
        );
    }
}
