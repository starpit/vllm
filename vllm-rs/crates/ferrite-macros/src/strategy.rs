use crate::ops::{Op, OpGraph};

/// A kernel in the fusion plan.
#[derive(Debug, Clone)]
pub enum KernelKind {
    /// Standalone RMSNorm kernel.
    RmsNorm {
        /// OpGraph node index.
        node_idx: usize,
    },
    /// Standalone GEMM kernel.
    Gemm {
        /// OpGraph node index.
        node_idx: usize,
    },
    /// Standalone SiLU kernel.
    Silu {
        /// OpGraph node index.
        node_idx: usize,
    },
    /// Fused GEMM + SiLU epilogue (SiLU applied to accumulators before store).
    GemmSilu {
        /// OpGraph node index for the GEMM.
        gemm_idx: usize,
        /// OpGraph node index for the SiLU.
        silu_idx: usize,
    },
    /// Fused RMSNorm -> GEMM -> SiLU megakernel.
    /// ONE kernel, ONE launch. Norm factors computed in phase 1, GEMM K-loop
    /// uses NormalizedLoader for A tiles, SiLU applied as epilogue on accumulators.
    RmsNormGemmSilu {
        /// OpGraph node index for the RMSNorm.
        norm_idx: usize,
        /// OpGraph node index for the GEMM.
        gemm_idx: usize,
        /// OpGraph node index for the SiLU.
        silu_idx: usize,
    },
    /// Fused RMSNorm -> GEMM megakernel (no SiLU).
    RmsNormGemm {
        /// OpGraph node index for the RMSNorm.
        norm_idx: usize,
        /// OpGraph node index for the GEMM.
        gemm_idx: usize,
    },
}

/// The fusion plan: a sequence of kernels to launch.
#[derive(Debug, Clone)]
pub struct FusionPlan {
    pub kernels: Vec<KernelKind>,
}

/// Evaluate which operations can be fused based on the megakernel strategy.
///
/// The key insight: Ferrite's whole purpose is the megakernel. We fuse
/// as aggressively as possible into ONE kernel, ONE launch.
///
/// Fusion rules (in priority order):
/// 1. RmsNorm -> GEMM -> SiLU = single fused megakernel (RmsNormGemmSilu)
/// 2. RmsNorm -> GEMM = fused megakernel (RmsNormGemm)
/// 3. GEMM -> SiLU = fused epilogue (GemmSilu)
/// 4. Anything left = standalone kernel
pub fn evaluate_strategy(graph: &OpGraph) -> FusionPlan {
    let mut kernels = Vec::new();
    let mut consumed = vec![false; graph.nodes.len()];

    // Pass 1: Look for RmsNorm -> GEMM -> SiLU triple fusion
    for i in 0..graph.nodes.len() {
        if consumed[i] {
            continue;
        }
        if let Op::RmsNorm { .. } = &graph.nodes[i].op {
            if let Some(ref norm_result) = graph.nodes[i].result_name {
                // Find GEMM consuming this norm output as its A input
                for j in (i + 1)..graph.nodes.len() {
                    if consumed[j] {
                        continue;
                    }
                    if let Op::Gemm { a, .. } = &graph.nodes[j].op {
                        if a == norm_result {
                            // Found RmsNorm -> GEMM. Now look for SiLU after GEMM.
                            if let Some(ref gemm_result) = graph.nodes[j].result_name {
                                let mut found_silu = false;
                                for k in (j + 1)..graph.nodes.len() {
                                    if consumed[k] {
                                        continue;
                                    }
                                    if let Op::Silu { input } = &graph.nodes[k].op {
                                        if input == gemm_result {
                                            // Triple fusion!
                                            kernels.push(KernelKind::RmsNormGemmSilu {
                                                norm_idx: i,
                                                gemm_idx: j,
                                                silu_idx: k,
                                            });
                                            consumed[i] = true;
                                            consumed[j] = true;
                                            consumed[k] = true;
                                            found_silu = true;
                                            break;
                                        }
                                    }
                                    break; // only check immediately following node
                                }
                                if !found_silu && !consumed[i] {
                                    // RmsNorm -> GEMM without SiLU
                                    kernels.push(KernelKind::RmsNormGemm {
                                        norm_idx: i,
                                        gemm_idx: j,
                                    });
                                    consumed[i] = true;
                                    consumed[j] = true;
                                }
                            } else if !consumed[i] {
                                // GEMM has no result name (terminal), fuse norm->gemm
                                kernels.push(KernelKind::RmsNormGemm {
                                    norm_idx: i,
                                    gemm_idx: j,
                                });
                                consumed[i] = true;
                                consumed[j] = true;
                            }
                            break;
                        }
                    }
                    break; // only check immediately following node
                }
            }
        }
    }

    // Pass 2: Look for GEMM -> SiLU pairs (without preceding RmsNorm)
    for i in 0..graph.nodes.len() {
        if consumed[i] {
            continue;
        }
        if let Op::Gemm { .. } = &graph.nodes[i].op {
            if let Some(ref gemm_result) = graph.nodes[i].result_name {
                for j in (i + 1)..graph.nodes.len() {
                    if consumed[j] {
                        continue;
                    }
                    if let Op::Silu { input } = &graph.nodes[j].op {
                        if input == gemm_result {
                            kernels.push(KernelKind::GemmSilu {
                                gemm_idx: i,
                                silu_idx: j,
                            });
                            consumed[i] = true;
                            consumed[j] = true;
                            break;
                        }
                    }
                    break;
                }
            }
        }
    }

    // Pass 3: Emit remaining ops as standalone kernels
    for i in 0..graph.nodes.len() {
        if consumed[i] {
            continue;
        }
        let kind = match &graph.nodes[i].op {
            Op::RmsNorm { .. } => KernelKind::RmsNorm { node_idx: i },
            Op::Gemm { .. } => KernelKind::Gemm { node_idx: i },
            Op::Silu { .. } => KernelKind::Silu { node_idx: i },
        };
        kernels.push(kind);
    }

    FusionPlan { kernels }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{OpGraph, OpNode};

    fn make_graph(nodes: Vec<(Option<&str>, Op)>) -> OpGraph {
        let mut graph = OpGraph::new();
        for (i, (name, op)) in nodes.into_iter().enumerate() {
            graph.nodes.push(OpNode {
                result_name: name.map(|s| s.to_string()),
                op,
                index: i,
            });
        }
        graph
    }

    #[test]
    fn test_rmsnorm_gemm_silu_fuses_into_one_kernel() {
        let graph = make_graph(vec![
            (
                Some("n"),
                Op::RmsNorm {
                    input: "x".into(),
                    weight: "w".into(),
                },
            ),
            (
                Some("g"),
                Op::Gemm {
                    a: "n".into(),
                    b: "w2".into(),
                },
            ),
            (None, Op::Silu { input: "g".into() }),
        ]);

        let plan = evaluate_strategy(&graph);

        assert_eq!(plan.kernels.len(), 1, "Must produce exactly ONE kernel");
        match &plan.kernels[0] {
            KernelKind::RmsNormGemmSilu {
                norm_idx,
                gemm_idx,
                silu_idx,
            } => {
                assert_eq!(*norm_idx, 0);
                assert_eq!(*gemm_idx, 1);
                assert_eq!(*silu_idx, 2);
            }
            other => panic!(
                "Expected RmsNormGemmSilu, got {:?}",
                match other {
                    KernelKind::RmsNorm { .. } => "RmsNorm",
                    KernelKind::Gemm { .. } => "Gemm",
                    KernelKind::Silu { .. } => "Silu",
                    KernelKind::GemmSilu { .. } => "GemmSilu",
                    KernelKind::RmsNormGemm { .. } => "RmsNormGemm",
                    _ => "Unknown",
                }
            ),
        }
    }

    #[test]
    fn test_gemm_silu_fuses_without_norm() {
        let graph = make_graph(vec![
            (
                Some("g"),
                Op::Gemm {
                    a: "x".into(),
                    b: "w".into(),
                },
            ),
            (None, Op::Silu { input: "g".into() }),
        ]);

        let plan = evaluate_strategy(&graph);

        assert_eq!(plan.kernels.len(), 1);
        assert!(matches!(&plan.kernels[0], KernelKind::GemmSilu { .. }));
    }

    #[test]
    fn test_rmsnorm_gemm_fuses_without_silu() {
        let graph = make_graph(vec![
            (
                Some("n"),
                Op::RmsNorm {
                    input: "x".into(),
                    weight: "w".into(),
                },
            ),
            (
                None,
                Op::Gemm {
                    a: "n".into(),
                    b: "w2".into(),
                },
            ),
        ]);

        let plan = evaluate_strategy(&graph);

        assert_eq!(plan.kernels.len(), 1);
        assert!(matches!(&plan.kernels[0], KernelKind::RmsNormGemm { .. }));
    }

    #[test]
    fn test_standalone_ops_not_fused() {
        let graph = make_graph(vec![(
            None,
            Op::Gemm {
                a: "x".into(),
                b: "w".into(),
            },
        )]);

        let plan = evaluate_strategy(&graph);

        assert_eq!(plan.kernels.len(), 1);
        assert!(matches!(&plan.kernels[0], KernelKind::Gemm { .. }));
    }
}
