/// An individual operation in the graph.
#[derive(Debug, Clone)]
pub enum Op {
    /// RMSNorm: `rmsnorm(input, weight)`
    RmsNorm { input: String, weight: String },
    /// Matrix multiply: `gemm(a, b)`
    Gemm { a: String, b: String },
    /// SiLU activation: `silu(x)`
    Silu { input: String },
}

impl Op {
    pub fn name(&self) -> &'static str {
        match self {
            Op::RmsNorm { .. } => "rmsnorm",
            Op::Gemm { .. } => "gemm",
            Op::Silu { .. } => "silu",
        }
    }
}

/// A node in the operation graph.
#[derive(Debug, Clone)]
pub struct OpNode {
    /// The variable name this result is bound to (if any).
    /// `None` for the final trailing expression.
    pub result_name: Option<String>,
    /// The operation.
    pub op: Op,
    /// Index in the graph's node list.
    pub index: usize,
}

/// A directed acyclic graph of operations parsed from the function body.
#[derive(Debug, Clone)]
pub struct OpGraph {
    pub nodes: Vec<OpNode>,
    /// The function parameters (name, type as string).
    pub params: Vec<(String, String)>,
}

impl OpGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            params: Vec::new(),
        }
    }

    /// Find the node that produces a given variable name.
    pub fn producer_of(&self, name: &str) -> Option<usize> {
        self.nodes
            .iter()
            .position(|n| n.result_name.as_deref() == Some(name))
    }
}
