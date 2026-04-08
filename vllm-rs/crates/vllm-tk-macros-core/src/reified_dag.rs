// SPDX-License-Identifier: Apache-2.0
//! Fully reified per-tile DAG for the megakernel forward pass.
//!
//! The `dag::ModelDag` is the AST level: ~8 ops inside a layer loop. That's
//! "what the user wrote." For real scheduling we need the *expanded* graph
//! where every node is a single (layer, phase, row_tile, col_tile) work unit
//! and every edge is an explicit data dependency on another work unit.
//!
//! For 1B LLaMA at seq=1024 with row_tile=16 this expansion produces ~100K
//! nodes and ~150K edges across all 16 layers. Because the parameterization
//! is fixed at compile time, we can build the entire forward pass DAG up
//! front and feed it to a static scheduler — something cuBLAS structurally
//! cannot do (it picks shapes from a runtime heuristic table).
//!
//! Phase 1 scope: just the data structures + a llama-specific reifier. No
//! scheduler, no codegen. Deliverables are node/edge counts, critical path
//! length, and a `.dot` dump for inspection.

use std::collections::HashMap;
use std::fmt::Write as _;

/// Phase tag for a reified node. Llama-specific for now; future model
/// families will add their own variants or this becomes generic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Phase {
    AttnNorm,
    Qkv,
    Rope,
    Attention,
    OProj,
    MlpNorm,
    GateUp,
    Down,
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Phase::AttnNorm => "attn_norm",
            Phase::Qkv => "qkv",
            Phase::Rope => "rope",
            Phase::Attention => "attention",
            Phase::OProj => "o_proj",
            Phase::MlpNorm => "mlp_norm",
            Phase::GateUp => "gate_up",
            Phase::Down => "down",
        }
    }

    /// Does this phase have a column tile dimension, or is it row-only?
    pub fn has_col_tiles(self) -> bool {
        matches!(
            self,
            Phase::Qkv | Phase::OProj | Phase::GateUp | Phase::Down
        )
    }
}

/// Unique id for a reified node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// One reified work unit: a (layer, phase, row_tile, col_tile) tile.
#[derive(Clone, Debug)]
pub struct Node {
    pub id: NodeId,
    pub layer: u16,
    pub phase: Phase,
    /// Row tile index (in row_tile units, not absolute row count).
    pub row: u16,
    /// Col tile index, or 0 for row-only phases.
    pub col: u16,
    /// Explicit data dependencies (NodeIds this node must wait on).
    pub deps: Vec<NodeId>,
}

/// Tile-size policy. All sizes in element units (not bytes).
#[derive(Clone, Copy, Debug)]
pub struct TileSizes {
    /// Common row tile (M dimension granularity) for all GEMM and norm phases.
    pub row_tile: u32,
    /// Col tile for QKV phase (output N dimension is HD + 2*HKD).
    pub qkv_col_tile: u32,
    /// Col tile for o_proj (output N = HD).
    pub o_col_tile: u32,
    /// Col tile for gate_up (output N = ID).
    pub gate_up_col_tile: u32,
    /// Col tile for down (output N = HD).
    pub down_col_tile: u32,
}

impl TileSizes {
    /// Default starting point: 16-row tiles, ~128-col tiles for the GEMMs.
    /// Tuned for inspection more than performance — the scheduler will
    /// eventually search this space.
    pub fn default_v1() -> Self {
        Self {
            row_tile: 16,
            qkv_col_tile: 128,
            o_col_tile: 128,
            gate_up_col_tile: 128,
            down_col_tile: 128,
        }
    }
}

/// Llama model dimensions extracted from the AST DAG params.
#[derive(Clone, Copy, Debug)]
pub struct LlamaDims {
    pub num_layers: u32,
    pub hidden_dim: u32,       // HD
    pub intermediate_dim: u32, // ID
    pub num_attn_heads: u32,   // NAH
    pub num_kv_heads: u32,     // NKH
    pub head_dim: u32,         // HDM
    /// Sequence length to reify for. NOT a model dim — chosen per-shape.
    pub seq_len: u32,
}

impl LlamaDims {
    pub fn from_params(params: &HashMap<String, usize>, seq_len: u32) -> Result<Self, String> {
        let get = |k: &str| -> Result<u32, String> {
            params
                .get(k)
                .copied()
                .map(|v| v as u32)
                .ok_or_else(|| format!("missing param {k}"))
        };
        Ok(Self {
            num_layers: get("NL")?,
            hidden_dim: get("HD")?,
            intermediate_dim: get("ID")?,
            num_attn_heads: get("NAH")?,
            num_kv_heads: get("NKH")?,
            head_dim: get("HDM")?,
            seq_len,
        })
    }

    /// QKV output N dimension: (NAH + 2*NKH) * HDM.
    pub fn qkv_out_dim(&self) -> u32 {
        (self.num_attn_heads + 2 * self.num_kv_heads) * self.head_dim
    }
}

/// Per-phase tile counts derived from `LlamaDims` + `TileSizes`.
#[derive(Clone, Copy, Debug)]
pub struct PhaseTileCounts {
    pub row_tiles: u32,
    pub qkv_col_tiles: u32,
    pub o_col_tiles: u32,
    pub gate_up_col_tiles: u32,
    pub down_col_tiles: u32,
}

impl PhaseTileCounts {
    pub fn compute(dims: &LlamaDims, tiles: &TileSizes) -> Self {
        let row_tiles = div_ceil_u32(dims.seq_len, tiles.row_tile);
        Self {
            row_tiles,
            qkv_col_tiles: div_ceil_u32(dims.qkv_out_dim(), tiles.qkv_col_tile),
            o_col_tiles: div_ceil_u32(dims.hidden_dim, tiles.o_col_tile),
            gate_up_col_tiles: div_ceil_u32(dims.intermediate_dim, tiles.gate_up_col_tile),
            down_col_tiles: div_ceil_u32(dims.hidden_dim, tiles.down_col_tile),
        }
    }
}

fn div_ceil_u32(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

/// The reified DAG.
#[derive(Clone, Debug)]
pub struct ReifiedDag {
    pub dims: LlamaDims,
    pub tiles: TileSizes,
    pub counts: PhaseTileCounts,
    pub nodes: Vec<Node>,
    /// Index lookup: (layer, phase, row, col) → NodeId.
    /// For row-only phases, col is always 0.
    index: HashMap<(u16, Phase, u16, u16), NodeId>,
}

impl ReifiedDag {
    /// Build the full reified forward-pass DAG for a llama-shaped model.
    ///
    /// This is intentionally llama-specific for Phase 1 — it bakes in the
    /// 8-phase layer pattern. A general AST→reified pass comes later when
    /// we extend to other model families.
    pub fn reify_llama(dims: LlamaDims, tiles: TileSizes) -> Self {
        let counts = PhaseTileCounts::compute(&dims, &tiles);
        let mut dag = Self {
            dims,
            tiles,
            counts,
            nodes: Vec::new(),
            index: HashMap::new(),
        };

        for layer in 0..dims.num_layers as u16 {
            dag.emit_layer(layer);
        }
        dag
    }

    fn emit_layer(&mut self, layer: u16) {
        let counts = self.counts;
        let row_tiles = counts.row_tiles as u16;

        // ── attn_norm: one tile per row, fan-in from prev layer's down OR input ──
        for r in 0..row_tiles {
            let mut deps = Vec::new();
            if layer > 0 {
                // Need full row of previous layer's down output (residual updated).
                for c in 0..counts.down_col_tiles as u16 {
                    deps.push(self.lookup(layer - 1, Phase::Down, r, c));
                }
            }
            self.add(layer, Phase::AttnNorm, r, 0, deps);
        }

        // ── qkv: GEMM with M=row_tile, K=HD (no col fan-in upstream), N=qkv_dim ──
        for r in 0..row_tiles {
            let norm_id = self.lookup(layer, Phase::AttnNorm, r, 0);
            for c in 0..counts.qkv_col_tiles as u16 {
                self.add(layer, Phase::Qkv, r, c, vec![norm_id]);
            }
        }

        // ── rope: per-row, fan-in from all qkv col tiles in that row ──
        for r in 0..row_tiles {
            let mut deps = Vec::with_capacity(counts.qkv_col_tiles as usize);
            for c in 0..counts.qkv_col_tiles as u16 {
                deps.push(self.lookup(layer, Phase::Qkv, r, c));
            }
            self.add(layer, Phase::Rope, r, 0, deps);
        }

        // ── attention: per-row Q tile, GLOBAL fan-in across all K/V rows ──
        // This is the only phase whose dependency reaches outside its own row.
        for r in 0..row_tiles {
            let mut deps = Vec::with_capacity(row_tiles as usize);
            for r2 in 0..row_tiles {
                deps.push(self.lookup(layer, Phase::Rope, r2, 0));
            }
            self.add(layer, Phase::Attention, r, 0, deps);
        }

        // ── o_proj + residual: GEMM-add, M=row_tile, K=HD, N=HD ──
        for r in 0..row_tiles {
            let attn_id = self.lookup(layer, Phase::Attention, r, 0);
            for c in 0..counts.o_col_tiles as u16 {
                // Note: residual add fuses the prior hidden_states value, which
                // is the same row from the previous layer's down output. That
                // dependency is implicit via how the kernel reads the shared
                // gmem buffer; we don't model it as a dep edge here because
                // o_proj writes to that same buffer (in-place residual).
                self.add(layer, Phase::OProj, r, c, vec![attn_id]);
            }
        }

        // ── mlp_norm: per-row, fan-in from all o_proj col tiles in that row ──
        for r in 0..row_tiles {
            let mut deps = Vec::with_capacity(counts.o_col_tiles as usize);
            for c in 0..counts.o_col_tiles as u16 {
                deps.push(self.lookup(layer, Phase::OProj, r, c));
            }
            self.add(layer, Phase::MlpNorm, r, 0, deps);
        }

        // ── gate_up: GEMM, M=row_tile, K=HD, N=ID, single dep on mlp_norm row ──
        for r in 0..row_tiles {
            let norm_id = self.lookup(layer, Phase::MlpNorm, r, 0);
            for c in 0..counts.gate_up_col_tiles as u16 {
                self.add(layer, Phase::GateUp, r, c, vec![norm_id]);
            }
        }

        // ── down + residual: GEMM-add, M=row_tile, K=ID, N=HD ──
        // A operand (silu_out) fans in across all gate_up col tiles in this row.
        for r in 0..row_tiles {
            let mut row_deps = Vec::with_capacity(counts.gate_up_col_tiles as usize);
            for c in 0..counts.gate_up_col_tiles as u16 {
                row_deps.push(self.lookup(layer, Phase::GateUp, r, c));
            }
            for c in 0..counts.down_col_tiles as u16 {
                self.add(layer, Phase::Down, r, c, row_deps.clone());
            }
        }
    }

    fn add(&mut self, layer: u16, phase: Phase, row: u16, col: u16, deps: Vec<NodeId>) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node {
            id,
            layer,
            phase,
            row,
            col,
            deps,
        });
        self.index.insert((layer, phase, row, col), id);
        id
    }

    fn lookup(&self, layer: u16, phase: Phase, row: u16, col: u16) -> NodeId {
        *self
            .index
            .get(&(layer, phase, row, col))
            .unwrap_or_else(|| {
                panic!(
                    "missing node lookup: layer={layer} phase={:?} row={row} col={col}",
                    phase
                )
            })
    }

    /// Total number of edges (sum of all node deps).
    pub fn edge_count(&self) -> usize {
        self.nodes.iter().map(|n| n.deps.len()).sum()
    }

    /// Compute longest path length (in node count) from any source to any sink.
    /// This is the lower bound on the number of sequential steps in any valid
    /// schedule — i.e. the critical-path depth in nodes.
    pub fn critical_path_depth(&self) -> u32 {
        let n = self.nodes.len();
        let mut depth = vec![0u32; n];
        // Nodes are emitted in topological order (each layer in-order, deps
        // always point to earlier-emitted nodes), so a single forward sweep
        // computes the depths.
        for i in 0..n {
            let max_dep_depth = self.nodes[i]
                .deps
                .iter()
                .map(|d| depth[d.0 as usize])
                .max()
                .unwrap_or(0);
            depth[i] = max_dep_depth + 1;
        }
        depth.into_iter().max().unwrap_or(0)
    }

    /// Histogram of nodes per phase (across all layers).
    pub fn phase_histogram(&self) -> Vec<(Phase, usize)> {
        let mut counts: HashMap<Phase, usize> = HashMap::new();
        for n in &self.nodes {
            *counts.entry(n.phase).or_insert(0) += 1;
        }
        let mut out: Vec<_> = counts.into_iter().collect();
        out.sort_by_key(|(p, _)| *p);
        out
    }

    /// Dump a graphviz dot representation. For large graphs this is huge —
    /// `max_layers` clips the dump to the first N layers for tractability.
    pub fn to_dot(&self, max_layers: u16) -> String {
        let mut out = String::new();
        writeln!(out, "digraph reified {{").unwrap();
        writeln!(out, "  rankdir=LR;").unwrap();
        writeln!(out, "  node [shape=box, fontsize=8];").unwrap();
        for n in &self.nodes {
            if n.layer >= max_layers {
                continue;
            }
            let label = if n.phase.has_col_tiles() {
                format!("L{}.{}\\n[r{},c{}]", n.layer, n.phase.name(), n.row, n.col)
            } else {
                format!("L{}.{}\\n[r{}]", n.layer, n.phase.name(), n.row)
            };
            writeln!(out, "  n{} [label=\"{}\"];", n.id.0, label).unwrap();
        }
        for n in &self.nodes {
            if n.layer >= max_layers {
                continue;
            }
            for d in &n.deps {
                let dn = &self.nodes[d.0 as usize];
                if dn.layer >= max_layers {
                    continue;
                }
                writeln!(out, "  n{} -> n{};", d.0, n.id.0).unwrap();
            }
        }
        writeln!(out, "}}").unwrap();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama_1b_dims(seq_len: u32) -> LlamaDims {
        // Matches the test fixture used elsewhere in the repo.
        LlamaDims {
            num_layers: 16,
            hidden_dim: 2048,
            intermediate_dim: 8192,
            num_attn_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            seq_len,
        }
    }

    #[test]
    fn reify_1b_seq1024_node_counts() {
        let dims = llama_1b_dims(1024);
        let tiles = TileSizes::default_v1();
        let dag = ReifiedDag::reify_llama(dims, tiles);

        let counts = dag.counts;
        // 1024 / 16 = 64 row tiles per layer.
        assert_eq!(counts.row_tiles, 64);
        // qkv_dim = (32 + 16) * 64 = 3072 → 3072 / 128 = 24 col tiles.
        assert_eq!(counts.qkv_col_tiles, 24);
        // o: HD=2048 / 128 = 16.
        assert_eq!(counts.o_col_tiles, 16);
        // gate_up: ID=8192 / 128 = 64.
        assert_eq!(counts.gate_up_col_tiles, 64);
        // down: HD=2048 / 128 = 16.
        assert_eq!(counts.down_col_tiles, 16);

        // Per-layer phase node counts:
        //   attn_norm: 64
        //   qkv:       64*24 = 1536
        //   rope:      64
        //   attn:      64
        //   o_proj:    64*16 = 1024
        //   mlp_norm:  64
        //   gate_up:   64*64 = 4096
        //   down:      64*16 = 1024
        // Total per layer: 7936. × 16 layers = 126_976.
        let expected_per_layer = 64 + 1536 + 64 + 64 + 1024 + 64 + 4096 + 1024;
        assert_eq!(expected_per_layer, 7936);
        assert_eq!(dag.nodes.len(), 7936 * 16);

        // Phase histogram sanity.
        let hist = dag.phase_histogram();
        for (phase, count) in &hist {
            let per_layer = match phase {
                Phase::AttnNorm => 64,
                Phase::Qkv => 1536,
                Phase::Rope => 64,
                Phase::Attention => 64,
                Phase::OProj => 1024,
                Phase::MlpNorm => 64,
                Phase::GateUp => 4096,
                Phase::Down => 1024,
            };
            assert_eq!(*count, per_layer * 16, "phase {:?}", phase);
        }
    }

    #[test]
    fn reify_invariants() {
        // Properties any well-formed reified DAG must satisfy:
        //   1. Every dep points to an earlier-emitted node (topological order).
        //   2. Only layer-0 attn_norm nodes are sources (no incoming deps).
        //   3. Every NodeId is unique and matches its slot index.
        let dims = llama_1b_dims(1024);
        let dag = ReifiedDag::reify_llama(dims, TileSizes::default_v1());
        for (i, n) in dag.nodes.iter().enumerate() {
            assert_eq!(n.id.0 as usize, i, "node id mismatch at slot {i}");
            for d in &n.deps {
                assert!(
                    (d.0 as usize) < i,
                    "node {i} ({:?}) has forward dep on {}",
                    n.phase,
                    d.0
                );
            }
            let is_source = n.deps.is_empty();
            if is_source {
                assert_eq!(n.layer, 0, "non-layer-0 source node {:?}", n);
                assert_eq!(n.phase, Phase::AttnNorm);
            }
        }
    }

    #[test]
    fn dot_dump_first_layer() {
        // Smoke test the .dot dump for layer 0 only — full graph is huge but
        // one layer renders cleanly enough to eyeball.
        let dims = llama_1b_dims(64); // shrink seq for a small dot
        let dag = ReifiedDag::reify_llama(dims, TileSizes::default_v1());
        let dot = dag.to_dot(1);
        assert!(dot.starts_with("digraph reified"));
        assert!(dot.contains("L0.attn_norm"));
        assert!(dot.contains("L0.attention"));
        // Optional: dump to /tmp for human inspection when running locally.
        // std::fs::write("/tmp/reified_layer0.dot", &dot).ok();
    }

    #[test]
    fn reify_1b_seq1024_edge_counts_and_critical_path() {
        let dims = llama_1b_dims(1024);
        let tiles = TileSizes::default_v1();
        let dag = ReifiedDag::reify_llama(dims, tiles);

        // Per-layer edges:
        //   attn_norm:  layer>0 → 64*16 (down col tiles); layer=0 → 0
        //   qkv:        1536*1 = 1536
        //   rope:       64*24 = 1536
        //   attn:       64*64 = 4096
        //   o_proj:     1024*1 = 1024
        //   mlp_norm:   64*16 = 1024
        //   gate_up:    4096*1 = 4096
        //   down:       1024*64 = 65536
        // Per-layer (without cross-layer attn_norm fan-in): 78_848
        // Layer 0: 78_848 + 0
        // Layers 1..16: 78_848 + 1024 each
        let per_layer_intra = 1536 + 1536 + 4096 + 1024 + 1024 + 4096 + 65536;
        assert_eq!(per_layer_intra, 78_848);
        let cross_layer = 64 * 16; // 1024
        let expected = per_layer_intra * 16 + cross_layer * 15;
        assert_eq!(dag.edge_count(), expected);

        // Critical path: walk one row's longest dependency chain.
        // Per layer the chain depth is roughly:
        //   attn_norm (1) → qkv (1) → rope (1) → attn (1) → o_proj (1)
        //     → mlp_norm (1) → gate_up (1) → down (1) = 8 nodes/layer
        // Across 16 layers: ~128 deep.
        let cp = dag.critical_path_depth();
        assert_eq!(cp, 8 * 16);
    }
}
