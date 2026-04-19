// SPDX-License-Identifier: Apache-2.0
//! Stencil IR core types. See design doc §1-3 for the frozen
//! vocabulary (3 roles, 5 dep kinds, 3 clarifications).

use smallvec::SmallVec;

pub type AxisId = u16;
pub type NodeId = u16;
pub type RegionId = u16;
pub type ScalarId = u16;

#[derive(Debug, Clone)]
pub struct Domain {
    pub axes: Vec<Axis>,
    pub predicates: Vec<Predicate>,
}

#[derive(Debug, Clone)]
pub struct Axis {
    pub id: AxisId,
    pub name: &'static str,
    pub bound: Bound,
}

#[derive(Debug, Clone)]
pub enum Bound {
    Const(u32),
    RegionEntryScalar(ScalarId),
    IndexedScalar(ScalarId, AxisId),
    Unbounded,
}

#[derive(Debug, Clone)]
pub struct Predicate {
    pub coeffs: SmallVec<[(AxisId, i32); 2]>,
    pub offset: AffineOffset,
    pub op: CmpOp,
}

#[derive(Debug, Clone)]
pub enum AffineOffset {
    Const(i32),
    RegionEntry(ScalarId),
    Indexed(ScalarId, AxisId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Le,
    Lt,
    Ge,
    Gt,
    Eq,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Load,
    Compute,
    Store,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepKind {
    Raw,
    Pipeline,
    AtomicReduce,
    Barrier,
    DataDependent,
}

#[derive(Debug, Clone, Default)]
pub struct DepVector(pub SmallVec<[(AxisId, i32); 2]>);

/// Back-reference into the FUF op DAG. For v1 this is a lightweight
/// tag. Resolution against the real FUF graph is deferred until the
/// FUF→Stencil lowering pass lands.
#[derive(Debug, Clone, Copy)]
pub struct FufOpRef {
    pub tag: &'static str,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub role: Role,
    pub op: FufOpRef,
    pub addr: Option<LoadAddr>,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub src: NodeId,
    pub dst: NodeId,
    pub kind: DepKind,
    pub vector: DepVector,
}

#[derive(Debug, Clone)]
pub struct LoadAddr {
    pub terms: SmallVec<[AddrTerm; 4]>,
}

#[derive(Debug, Clone)]
pub enum AddrTerm {
    RegionEntryConst(ScalarId),
    AxisStride {
        axis: AxisId,
        stride: StrideExpr,
    },
    AxisModStride {
        axis: AxisId,
        modulus: u32,
        stride: StrideExpr,
    },
    AxisDivGather {
        axis: AxisId,
        divisor: u32,
        table: SmemLookup,
        stride: StrideExpr,
    },
}

#[derive(Debug, Clone)]
pub enum StrideExpr {
    Const(u64),
    RegionEntry(ScalarId),
}

#[derive(Debug, Clone)]
pub struct SmemLookup {
    /// Symbolic name for v1; resolved against FUF/runtime when
    /// lowering is wired up.
    pub source: &'static str,
}

#[derive(Debug, Clone)]
pub struct ScalarBinding {
    pub id: ScalarId,
    pub name: &'static str,
}

#[derive(Debug, Clone)]
pub struct Region {
    pub id: RegionId,
    pub name: &'static str,
    pub domain: Domain,
    pub entry_scalars: Vec<ScalarBinding>,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    /// Maps canonical gmem names (from `emit_ops::gmem_refs`) to
    /// per-region unique FUF-derived identities. Populated by the
    /// lowering pass after walking tile inputs/outputs — e.g., an
    /// attention Region in layer 5 binds `"Q_gmem"` to something like
    /// `"t123_0"` (tile 123, slot 0 = Q output of the previous
    /// qkv_rope). Two regions that share an upstream tile share the
    /// identity; every other case gets a distinct pointer in the
    /// kernel signature.
    ///
    /// Empty = legacy behavior: the emitter uses canonical names
    /// directly. Pre-item-4 regions (all templates today) leave this
    /// empty; item 4b populates it. STENCIL_IR_STATUS.md item 4.
    pub gmem_bindings: Vec<(&'static str, &'static str)>,
}

#[derive(Debug, Clone)]
pub struct ControlEdge {
    pub src: RegionId,
    pub dst: RegionId,
    pub kind: DepKind,
}

#[derive(Debug, Clone)]
pub struct Megakernel {
    pub regions: Vec<Region>,
    pub control: Vec<ControlEdge>,
}

impl Region {
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id as usize]
    }

    pub fn axis(&self, id: AxisId) -> &Axis {
        self.domain
            .axes
            .iter()
            .find(|a| a.id == id)
            .expect("unknown axis id")
    }

    pub fn scalar(&self, id: ScalarId) -> &ScalarBinding {
        self.entry_scalars
            .iter()
            .find(|s| s.id == id)
            .expect("unknown scalar id")
    }
}

/// Validate structural invariants the lowering pass will rely on.
/// Keep this small and honest; it catches drafting mistakes in
/// region templates, not runtime conditions.
pub fn validate(region: &Region) -> Result<(), String> {
    // Every ControlEdge DepKind is one of {Barrier, DataDependent}
    // (checked at the Megakernel level, not here).
    for e in &region.edges {
        if e.src as usize >= region.nodes.len() {
            return Err(format!("edge src {} out of range", e.src));
        }
        if e.dst as usize >= region.nodes.len() {
            return Err(format!("edge dst {} out of range", e.dst));
        }
        // AtomicReduce / Barrier / DataDependent are region-boundary
        // concerns. Inside a region we expect Raw or Pipeline.
        match e.kind {
            DepKind::Raw | DepKind::Pipeline => {}
            other => {
                return Err(format!(
                    "edge {}→{} uses {:?}; region-internal edges must be Raw or Pipeline",
                    e.src, e.dst, other
                ));
            }
        }
    }
    // A Load/Store must have an address; a Compute must not.
    for n in &region.nodes {
        match (n.role, &n.addr) {
            (Role::Load, None) | (Role::Store, None) => {
                return Err(format!("node {} ({:?}) needs an address", n.id, n.role));
            }
            (Role::Compute, Some(_)) => {
                return Err(format!("node {} (Compute) must not have an address", n.id));
            }
            _ => {}
        }
    }
    Ok(())
}
