//! Pipeline stage descriptors extracted from PTX.
//!
//! Each kernel is analyzed into a `PipelineStage` that captures its tiled
//! structure: loops, carry registers, pattern classification, and finalization.
//! The pipeline compiler (future) composes N stages into a single fused kernel.

use crate::fuse_cp_async::{CpAsyncClass, classify_cp_async_loads, identify_a_matrix_param};
use crate::parser::{
    AsyncCopyPort, CarryRegister, CarryRole, KernelProtocol, LoopDescriptor, PtxParser,
    analyze_carries, detect_loops,
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
