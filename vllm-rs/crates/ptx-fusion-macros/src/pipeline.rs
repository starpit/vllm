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
    pub fn from_ptx(name: &str, source: &str) -> Result<Self, String> {
        let protocol = PtxParser::parse(source)?;

        // For multi-entry PTX, extract the first entry so loop/carry analysis
        // only sees one kernel, not all entries.
        let raw_lines: Vec<&str> = source.lines().collect();
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
            extracted = crate::extract::extract_entry(source, substr)
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
            source_lines: source.lines().map(|l| l.to_string()).collect(),
        })
    }
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
        let stage = PipelineStage::from_ptx("rms_norm", ptx).expect("extraction failed");

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
        let stage = PipelineStage::from_ptx("gemm", ptx).expect("extraction failed");

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
        let stage = PipelineStage::from_ptx("silu_mul", ptx).expect("extraction failed");

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
        let stage = PipelineStage::from_ptx("scale", ptx).expect("extraction failed");

        match &stage.pattern {
            StagePattern::Pointwise => {} // correct
            other => panic!("scale should be Pointwise, got {:?}", other),
        }
    }
}
