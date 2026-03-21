use crate::ops::{InputPort, NodeId, OpClass, OpGraph};

/// A single stage in the fusion plan — one GEMM with optional prologue/epilogue transforms.
///
/// Each stage maps to one pipeline invocation:
/// ```text
/// Pipeline<CpAsyncCopy, CpAsyncCopy, transform_atom, MMA, epilogue_atom>
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage {
    /// The GEMM node for this stage.
    pub gemm: NodeId,
    /// Optional elementwise op on the A input (e.g., RmsNorm normalizes fragments
    /// between ldmatrix and MMA).
    pub prologue_transform: Option<NodeId>,
    /// Optional elementwise op on the output (e.g., SiLU applied to accumulators
    /// before store).
    pub epilogue_transform: Option<NodeId>,
}

/// The fusion plan: a sequence of stages to execute.
///
/// Single-stage plans produce one kernel launch.
/// Multi-stage plans produce multiple kernel launches with intermediates through
/// global memory (L2 cache locality).
#[derive(Debug, Clone)]
pub struct FusionPlan {
    pub stages: Vec<Stage>,
}

/// Evaluate the fusion strategy by walking edges in the operation DAG.
///
/// For each GEMM node, this classifies its input and output edges:
///
/// 1. **Prologue transform**: If the GEMM's Primary (A) input comes from an
///    Elementwise op, that op becomes a prologue transform (TransformAtom).
///
/// 2. **Epilogue transform**: If the GEMM's output feeds into an Elementwise op
///    (and that op has no other Matmul consumer), that op becomes an epilogue
///    transform (EpilogueAtom).
///
/// 3. **GEMM chain**: If the GEMM's output feeds into another GEMM, the
///    intermediate goes through global memory, creating a new stage.
///
/// This is fully general — any new OpKind just needs its OpClass, and the
/// strategy engine handles it automatically.
pub fn evaluate_strategy(graph: &OpGraph) -> FusionPlan {
    let mut stages = Vec::new();
    let mut consumed = vec![false; graph.nodes.len()];

    // Process nodes in topological order. For each GEMM, build a Stage.
    for node in &graph.nodes {
        if consumed[node.id] {
            continue;
        }

        if node.class != OpClass::Matmul {
            continue;
        }

        // This is a GEMM node. Build a stage around it.
        let gemm_id = node.id;

        // --- Check for prologue transform ---
        // If the GEMM's Primary input comes from an Elementwise op, fuse it.
        let prologue = graph
            .input_node(gemm_id, InputPort::Primary)
            .filter(|&src_id| {
                !consumed[src_id]
                    && graph.nodes[src_id].class == OpClass::Elementwise
                    // Only fuse if this elementwise op's sole consumer is this GEMM.
                    // If it feeds multiple consumers, it must remain standalone.
                    && graph.consumers_of(src_id).len() == 1
            });

        // --- Check for epilogue transform ---
        // If this GEMM has exactly one consumer and it's an Elementwise op, fuse it.
        let consumers = graph.consumers_of(gemm_id);
        let epilogue = if consumers.len() == 1 {
            let consumer_id = consumers[0];
            if !consumed[consumer_id] && graph.nodes[consumer_id].class == OpClass::Elementwise {
                Some(consumer_id)
            } else {
                None
            }
        } else {
            None
        };

        // Mark consumed
        consumed[gemm_id] = true;
        if let Some(p) = prologue {
            consumed[p] = true;
        }
        if let Some(e) = epilogue {
            consumed[e] = true;
        }

        stages.push(Stage {
            gemm: gemm_id,
            prologue_transform: prologue,
            epilogue_transform: epilogue,
        });
    }

    // Any remaining unconsumed nodes that are standalone (no GEMM absorbed them)
    // get their own degenerate stages. For now, we only handle the case where
    // all ops are part of GEMM stages — standalone elementwise-only kernels
    // are not yet supported at the codegen level.
    //
    // In practice, every meaningful Ferrite pipeline has at least one GEMM,
    // so this path is for future extensibility.

    FusionPlan { stages }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{Edge, InputPort, OpGraph, OpKind, OpNode, PARAM, ParamInfo};

    /// Helper to build an OpGraph from a sequence of node descriptors.
    fn build_graph(descriptors: Vec<NodeDesc>) -> OpGraph {
        let mut graph = OpGraph::new();
        // Add dummy params for anything sourced from PARAM
        graph.params.push(ParamInfo {
            name: "x".to_string(),
            ty: "DevicePtr".to_string(),
        });
        graph.params.push(ParamInfo {
            name: "w".to_string(),
            ty: "DevicePtr".to_string(),
        });
        graph.params.push(ParamInfo {
            name: "w2".to_string(),
            ty: "DevicePtr".to_string(),
        });
        graph.params.push(ParamInfo {
            name: "w3".to_string(),
            ty: "DevicePtr".to_string(),
        });

        for (idx, desc) in descriptors.iter().enumerate() {
            let kind = desc.kind;
            let class = kind.class();
            let inputs = desc
                .inputs
                .iter()
                .map(|(src, port, name)| Edge {
                    src: *src,
                    port: *port,
                    src_name: name.to_string(),
                })
                .collect();
            graph.nodes.push(OpNode {
                id: idx,
                kind,
                class,
                inputs,
                result_name: desc.result_name.map(|s| s.to_string()),
            });
        }

        if !graph.nodes.is_empty() {
            graph.output = graph.nodes.len() - 1;
        }

        graph
    }

    struct NodeDesc {
        kind: OpKind,
        result_name: Option<&'static str>,
        inputs: Vec<(NodeId, InputPort, &'static str)>,
    }

    // ── Existing tests (must still pass) ──

    #[test]
    fn test_rmsnorm_gemm_silu_fuses_into_one_stage() {
        // let n = rmsnorm(x, w);
        // let g = gemm(n, w2);
        // silu(g)
        let graph = build_graph(vec![
            NodeDesc {
                kind: OpKind::RmsNorm,
                result_name: Some("n"),
                inputs: vec![
                    (PARAM, InputPort::Primary, "x"),
                    (PARAM, InputPort::Weight, "w"),
                ],
            },
            NodeDesc {
                kind: OpKind::Gemm,
                result_name: Some("g"),
                inputs: vec![
                    (0, InputPort::Primary, "n"),
                    (PARAM, InputPort::Weight, "w2"),
                ],
            },
            NodeDesc {
                kind: OpKind::Silu,
                result_name: None,
                inputs: vec![(1, InputPort::Primary, "g")],
            },
        ]);

        let plan = evaluate_strategy(&graph);

        assert_eq!(plan.stages.len(), 1, "Must produce exactly ONE stage");
        let stage = &plan.stages[0];
        assert_eq!(stage.gemm, 1);
        assert_eq!(
            stage.prologue_transform,
            Some(0),
            "RmsNorm should be prologue"
        );
        assert_eq!(stage.epilogue_transform, Some(2), "SiLU should be epilogue");
    }

    #[test]
    fn test_gemm_silu_fuses_without_norm() {
        // let g = gemm(x, w);
        // silu(g)
        let graph = build_graph(vec![
            NodeDesc {
                kind: OpKind::Gemm,
                result_name: Some("g"),
                inputs: vec![
                    (PARAM, InputPort::Primary, "x"),
                    (PARAM, InputPort::Weight, "w"),
                ],
            },
            NodeDesc {
                kind: OpKind::Silu,
                result_name: None,
                inputs: vec![(0, InputPort::Primary, "g")],
            },
        ]);

        let plan = evaluate_strategy(&graph);

        assert_eq!(plan.stages.len(), 1);
        let stage = &plan.stages[0];
        assert_eq!(stage.gemm, 0);
        assert_eq!(stage.prologue_transform, None);
        assert_eq!(stage.epilogue_transform, Some(1));
    }

    #[test]
    fn test_rmsnorm_gemm_fuses_without_silu() {
        // let n = rmsnorm(x, w);
        // gemm(n, w2)
        let graph = build_graph(vec![
            NodeDesc {
                kind: OpKind::RmsNorm,
                result_name: Some("n"),
                inputs: vec![
                    (PARAM, InputPort::Primary, "x"),
                    (PARAM, InputPort::Weight, "w"),
                ],
            },
            NodeDesc {
                kind: OpKind::Gemm,
                result_name: None,
                inputs: vec![
                    (0, InputPort::Primary, "n"),
                    (PARAM, InputPort::Weight, "w2"),
                ],
            },
        ]);

        let plan = evaluate_strategy(&graph);

        assert_eq!(plan.stages.len(), 1);
        let stage = &plan.stages[0];
        assert_eq!(stage.gemm, 1);
        assert_eq!(stage.prologue_transform, Some(0));
        assert_eq!(stage.epilogue_transform, None);
    }

    #[test]
    fn test_standalone_gemm() {
        // gemm(x, w)
        let graph = build_graph(vec![NodeDesc {
            kind: OpKind::Gemm,
            result_name: None,
            inputs: vec![
                (PARAM, InputPort::Primary, "x"),
                (PARAM, InputPort::Weight, "w"),
            ],
        }]);

        let plan = evaluate_strategy(&graph);

        assert_eq!(plan.stages.len(), 1);
        let stage = &plan.stages[0];
        assert_eq!(stage.gemm, 0);
        assert_eq!(stage.prologue_transform, None);
        assert_eq!(stage.epilogue_transform, None);
    }

    // ── New tests for new patterns ──

    #[test]
    fn test_rmsnorm_gemm_silu_gemm_two_stages() {
        // MLP block: rmsnorm → gemm → silu → gemm
        // let n = rmsnorm(x, w);
        // let g = gemm(n, w2);
        // let h = silu(g);
        // gemm(h, w3)
        let graph = build_graph(vec![
            NodeDesc {
                kind: OpKind::RmsNorm,
                result_name: Some("n"),
                inputs: vec![
                    (PARAM, InputPort::Primary, "x"),
                    (PARAM, InputPort::Weight, "w"),
                ],
            },
            NodeDesc {
                kind: OpKind::Gemm,
                result_name: Some("g"),
                inputs: vec![
                    (0, InputPort::Primary, "n"),
                    (PARAM, InputPort::Weight, "w2"),
                ],
            },
            NodeDesc {
                kind: OpKind::Silu,
                result_name: Some("h"),
                inputs: vec![(1, InputPort::Primary, "g")],
            },
            NodeDesc {
                kind: OpKind::Gemm,
                result_name: None,
                inputs: vec![
                    (2, InputPort::Primary, "h"),
                    (PARAM, InputPort::Weight, "w3"),
                ],
            },
        ]);

        let plan = evaluate_strategy(&graph);

        assert_eq!(plan.stages.len(), 2, "MLP block should produce two stages");

        // Stage 0: Transform(RmsNorm) + GEMM + Epilogue(SiLU)
        assert_eq!(plan.stages[0].gemm, 1);
        assert_eq!(plan.stages[0].prologue_transform, Some(0));
        assert_eq!(plan.stages[0].epilogue_transform, Some(2));

        // Stage 1: standalone GEMM (h is intermediate through global memory)
        assert_eq!(plan.stages[1].gemm, 3);
        assert_eq!(plan.stages[1].prologue_transform, None);
        assert_eq!(plan.stages[1].epilogue_transform, None);
    }

    #[test]
    fn test_gemm_gemm_chain() {
        // gemm → gemm chain (intermediate through global/L2)
        // let g = gemm(x, w);
        // gemm(g, w2)
        let graph = build_graph(vec![
            NodeDesc {
                kind: OpKind::Gemm,
                result_name: Some("g"),
                inputs: vec![
                    (PARAM, InputPort::Primary, "x"),
                    (PARAM, InputPort::Weight, "w"),
                ],
            },
            NodeDesc {
                kind: OpKind::Gemm,
                result_name: None,
                inputs: vec![
                    (0, InputPort::Primary, "g"),
                    (PARAM, InputPort::Weight, "w2"),
                ],
            },
        ]);

        let plan = evaluate_strategy(&graph);

        assert_eq!(plan.stages.len(), 2, "GEMM chain should produce two stages");

        // Stage 0: standalone GEMM
        assert_eq!(plan.stages[0].gemm, 0);
        assert_eq!(plan.stages[0].prologue_transform, None);
        assert_eq!(plan.stages[0].epilogue_transform, None);

        // Stage 1: standalone GEMM (consumes output of stage 0 through global memory)
        assert_eq!(plan.stages[1].gemm, 1);
        assert_eq!(plan.stages[1].prologue_transform, None);
        assert_eq!(plan.stages[1].epilogue_transform, None);
    }

    #[test]
    fn test_gemm_silu_gemm_two_stages() {
        // gemm → silu → gemm (epilogue fuses with first GEMM)
        // let g = gemm(x, w);
        // let h = silu(g);
        // gemm(h, w2)
        let graph = build_graph(vec![
            NodeDesc {
                kind: OpKind::Gemm,
                result_name: Some("g"),
                inputs: vec![
                    (PARAM, InputPort::Primary, "x"),
                    (PARAM, InputPort::Weight, "w"),
                ],
            },
            NodeDesc {
                kind: OpKind::Silu,
                result_name: Some("h"),
                inputs: vec![(0, InputPort::Primary, "g")],
            },
            NodeDesc {
                kind: OpKind::Gemm,
                result_name: None,
                inputs: vec![
                    (1, InputPort::Primary, "h"),
                    (PARAM, InputPort::Weight, "w2"),
                ],
            },
        ]);

        let plan = evaluate_strategy(&graph);

        assert_eq!(plan.stages.len(), 2);

        // Stage 0: GEMM + Epilogue(SiLU)
        assert_eq!(plan.stages[0].gemm, 0);
        assert_eq!(plan.stages[0].prologue_transform, None);
        assert_eq!(plan.stages[0].epilogue_transform, Some(1));

        // Stage 1: standalone GEMM
        assert_eq!(plan.stages[1].gemm, 2);
        assert_eq!(plan.stages[1].prologue_transform, None);
        assert_eq!(plan.stages[1].epilogue_transform, None);
    }
}
