/// Unique identifier for a node in the operation graph.
pub type NodeId = usize;

/// Sentinel value representing a function parameter (not produced by any op).
pub const PARAM: NodeId = usize::MAX;

/// The kind of operation a node performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    /// Elementwise normalization: y = rmsnorm(x, w)
    RmsNorm,
    /// Matrix multiplication: y = x @ w
    Gemm,
    /// Elementwise activation: y = silu(x)
    Silu,
    /// Elementwise activation: y = gelu(x)  [fast sigmoid approximation]
    Gelu,
    /// Elementwise addition: y = x + residual
    ResidualAdd,
    /// Fused SiLU × multiply: y = silu(gate) * up
    /// Used with wide GEMM (pre-concatenated [w_gate; w_up] weights).
    /// The wide GEMM output is split at midpoint, SiLU applied to first half,
    /// then multiplied by second half.
    SiluMul,
    // Future: RotaryEmbed, Attention, Quantize, etc.
}

impl OpKind {
    pub fn name(&self) -> &'static str {
        match self {
            OpKind::RmsNorm => "rmsnorm",
            OpKind::Gemm => "gemm",
            OpKind::Silu => "silu",
            OpKind::Gelu => "gelu",
            OpKind::ResidualAdd => "residual_add",
            OpKind::SiluMul => "silu_mul",
        }
    }
}

/// Classification of how an op relates to the compute graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpClass {
    /// Operates element-by-element (RmsNorm, SiLU, GELU, etc.)
    Elementwise,
    /// Matrix multiplication (GEMM)
    Matmul,
    // Future: Reduction, Attention, etc.
}

impl OpKind {
    /// Return the structural class of this op kind.
    pub fn class(&self) -> OpClass {
        match self {
            OpKind::RmsNorm => OpClass::Elementwise,
            OpKind::Silu => OpClass::Elementwise,
            OpKind::Gelu => OpClass::Elementwise,
            OpKind::ResidualAdd => OpClass::Elementwise,
            OpKind::SiluMul => OpClass::Elementwise,
            OpKind::Gemm => OpClass::Matmul,
        }
    }
}

/// Which input port an edge connects to on the destination node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputPort {
    /// The main data input (x for elementwise, A for GEMM).
    Primary,
    /// Weight/parameter input (w for norm, B for GEMM).
    Weight,
}

/// A directed edge in the operation graph.
#[derive(Debug, Clone)]
pub struct Edge {
    /// Source node (or `PARAM` for function parameters).
    pub src: NodeId,
    /// Which input port on the destination node this edge connects to.
    pub port: InputPort,
    /// Name of the source variable (for debugging/error messages).
    pub src_name: String,
}

/// A node in the operation DAG.
#[derive(Debug, Clone)]
pub struct OpNode {
    /// Unique identifier.
    pub id: NodeId,
    /// What operation this node performs.
    pub kind: OpKind,
    /// Structural classification.
    pub class: OpClass,
    /// Incoming edges (one per input).
    pub inputs: Vec<Edge>,
    /// The variable name this result is bound to (for debugging/error messages).
    pub result_name: Option<String>,
}

/// Information about a function parameter.
#[derive(Debug, Clone)]
pub struct ParamInfo {
    /// Parameter name.
    pub name: String,
    /// Type as a string (e.g., "DevicePtr", "u32").
    pub ty: String,
}

/// A directed acyclic graph of operations parsed from the function body.
#[derive(Debug, Clone)]
pub struct OpGraph {
    /// Nodes in topological order.
    pub nodes: Vec<OpNode>,
    /// Function parameters.
    pub params: Vec<ParamInfo>,
    /// Which node produces the final output (last node).
    pub output: NodeId,
}

impl OpGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            params: Vec::new(),
            output: 0,
        }
    }

    /// Find the node that produces a given variable name.
    pub fn producer_of(&self, name: &str) -> Option<NodeId> {
        self.nodes
            .iter()
            .position(|n| n.result_name.as_deref() == Some(name))
    }

    /// Check if `name` is a function parameter.
    pub fn is_param(&self, name: &str) -> bool {
        self.params.iter().any(|p| p.name == name)
    }

    /// Return all nodes that consume the output of `node_id`.
    pub fn consumers_of(&self, node_id: NodeId) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|n| n.inputs.iter().any(|e| e.src == node_id))
            .map(|n| n.id)
            .collect()
    }

    /// Return the node that feeds into `node_id` on the given port, if any real node (not PARAM).
    pub fn input_node(&self, node_id: NodeId, port: InputPort) -> Option<NodeId> {
        self.nodes[node_id]
            .inputs
            .iter()
            .find(|e| e.port == port)
            .and_then(|e| if e.src == PARAM { None } else { Some(e.src) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_mlp_graph() -> OpGraph {
        // rmsnorm(x, w) → gemm(n, w2) → silu(g) → gemm(h, w3)
        let mut g = OpGraph::new();
        g.params.push(ParamInfo {
            name: "x".into(),
            ty: "DevicePtr".into(),
        });
        g.params.push(ParamInfo {
            name: "w".into(),
            ty: "DevicePtr".into(),
        });
        g.params.push(ParamInfo {
            name: "w2".into(),
            ty: "DevicePtr".into(),
        });
        g.params.push(ParamInfo {
            name: "w3".into(),
            ty: "DevicePtr".into(),
        });

        // node 0: n = rmsnorm(x, w)
        g.nodes.push(OpNode {
            id: 0,
            kind: OpKind::RmsNorm,
            class: OpClass::Elementwise,
            inputs: vec![
                Edge {
                    src: PARAM,
                    port: InputPort::Primary,
                    src_name: "param".into(),
                },
                Edge {
                    src: PARAM,
                    port: InputPort::Weight,
                    src_name: "param".into(),
                },
            ],
            result_name: Some("n".into()),
        });
        // node 1: g = gemm(n, w2)
        g.nodes.push(OpNode {
            id: 1,
            kind: OpKind::Gemm,
            class: OpClass::Matmul,
            inputs: vec![
                Edge {
                    src: 0,
                    port: InputPort::Primary,
                    src_name: "n".into(),
                },
                Edge {
                    src: PARAM,
                    port: InputPort::Weight,
                    src_name: "param".into(),
                },
            ],
            result_name: Some("g".into()),
        });
        // node 2: h = silu(g)
        g.nodes.push(OpNode {
            id: 2,
            kind: OpKind::Silu,
            class: OpClass::Elementwise,
            inputs: vec![Edge {
                src: 1,
                port: InputPort::Primary,
                src_name: "g".into(),
            }],
            result_name: Some("h".into()),
        });
        // node 3: gemm(h, w3)
        g.nodes.push(OpNode {
            id: 3,
            kind: OpKind::Gemm,
            class: OpClass::Matmul,
            inputs: vec![
                Edge {
                    src: 2,
                    port: InputPort::Primary,
                    src_name: "h".into(),
                },
                Edge {
                    src: PARAM,
                    port: InputPort::Weight,
                    src_name: "param".into(),
                },
            ],
            result_name: None,
        });
        g.output = 3;
        g
    }

    #[test]
    fn test_dag_edge_connectivity() {
        let g = build_mlp_graph();
        // rmsnorm has no producer (inputs from PARAM)
        assert_eq!(g.input_node(0, InputPort::Primary), None);
        // gemm1 gets A from rmsnorm (node 0)
        assert_eq!(g.input_node(1, InputPort::Primary), Some(0));
        // gemm1 gets B from PARAM
        assert_eq!(g.input_node(1, InputPort::Weight), None);
        // silu gets input from gemm1 (node 1)
        assert_eq!(g.input_node(2, InputPort::Primary), Some(1));
        // gemm2 gets A from silu (node 2)
        assert_eq!(g.input_node(3, InputPort::Primary), Some(2));
    }

    #[test]
    fn test_dag_consumers() {
        let g = build_mlp_graph();
        // rmsnorm is consumed by gemm1
        assert_eq!(g.consumers_of(0), vec![1]);
        // gemm1 is consumed by silu
        assert_eq!(g.consumers_of(1), vec![2]);
        // silu is consumed by gemm2
        assert_eq!(g.consumers_of(2), vec![3]);
        // gemm2 has no consumers (it's the output)
        assert_eq!(g.consumers_of(3), vec![]);
    }

    #[test]
    fn test_dag_producer_of() {
        let g = build_mlp_graph();
        assert_eq!(g.producer_of("n"), Some(0));
        assert_eq!(g.producer_of("g"), Some(1));
        assert_eq!(g.producer_of("h"), Some(2));
        assert_eq!(g.producer_of("nonexistent"), None);
    }

    #[test]
    fn test_dag_is_param() {
        let g = build_mlp_graph();
        assert!(g.is_param("x"));
        assert!(g.is_param("w2"));
        assert!(!g.is_param("n")); // n is a node result, not a param
    }

    #[test]
    fn test_op_class_classification() {
        assert_eq!(OpKind::RmsNorm.class(), OpClass::Elementwise);
        assert_eq!(OpKind::Silu.class(), OpClass::Elementwise);
        assert_eq!(OpKind::Gelu.class(), OpClass::Elementwise);
        assert_eq!(OpKind::ResidualAdd.class(), OpClass::Elementwise);
        assert_eq!(OpKind::Gemm.class(), OpClass::Matmul);
    }

    #[test]
    fn test_dag_output_is_last_node() {
        let g = build_mlp_graph();
        assert_eq!(g.output, 3);
        assert_eq!(g.nodes[g.output].kind, OpKind::Gemm);
    }

    #[test]
    fn test_dag_gelu_edges() {
        // gemm(x, w) → gelu(g)
        let mut g = OpGraph::new();
        g.params.push(ParamInfo {
            name: "x".into(),
            ty: "DevicePtr".into(),
        });
        g.params.push(ParamInfo {
            name: "w".into(),
            ty: "DevicePtr".into(),
        });
        g.nodes.push(OpNode {
            id: 0,
            kind: OpKind::Gemm,
            class: OpClass::Matmul,
            inputs: vec![
                Edge {
                    src: PARAM,
                    port: InputPort::Primary,
                    src_name: "x".into(),
                },
                Edge {
                    src: PARAM,
                    port: InputPort::Weight,
                    src_name: "w".into(),
                },
            ],
            result_name: Some("g".into()),
        });
        g.nodes.push(OpNode {
            id: 1,
            kind: OpKind::Gelu,
            class: OpClass::Elementwise,
            inputs: vec![Edge {
                src: 0,
                port: InputPort::Primary,
                src_name: "g".into(),
            }],
            result_name: None,
        });
        g.output = 1;

        // GELU gets input from GEMM
        assert_eq!(g.input_node(1, InputPort::Primary), Some(0));
        // GEMM is consumed by GELU
        assert_eq!(g.consumers_of(0), vec![1]);
        // GELU has no consumers
        assert_eq!(g.consumers_of(1), vec![]);
    }

    #[test]
    fn test_dag_residual_add_edges() {
        // residual_add(x, r) — two inputs, both from params
        let mut g = OpGraph::new();
        g.params.push(ParamInfo {
            name: "x".into(),
            ty: "DevicePtr".into(),
        });
        g.params.push(ParamInfo {
            name: "r".into(),
            ty: "DevicePtr".into(),
        });
        g.nodes.push(OpNode {
            id: 0,
            kind: OpKind::ResidualAdd,
            class: OpClass::Elementwise,
            inputs: vec![
                Edge {
                    src: PARAM,
                    port: InputPort::Primary,
                    src_name: "x".into(),
                },
                Edge {
                    src: PARAM,
                    port: InputPort::Weight,
                    src_name: "r".into(),
                },
            ],
            result_name: None,
        });
        g.output = 0;

        // Both inputs come from PARAM
        assert_eq!(g.input_node(0, InputPort::Primary), None);
        assert_eq!(g.input_node(0, InputPort::Weight), None);
        assert_eq!(g.nodes[0].kind, OpKind::ResidualAdd);
    }
}
