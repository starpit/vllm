// SPDX-License-Identifier: Apache-2.0
//! Region templates — parametric stencil fragments that the lowering
//! pass instantiates per subgraph. Each template declares its own
//! iteration domain, entry scalars, nodes, and edges in terms of the
//! frozen vocabulary (3 roles, 5 dep kinds); the megakernel emitter
//! stitches instantiated templates together via `Megakernel.control`.
//!
//! Templates live here (not in the lowering pass) so they stay
//! arch-neutral — per-arch lowering happens via `ArchMap` at emit
//! time, not at template construction.

use smallvec::smallvec;

use crate::ir::{
    AddrTerm, AffineOffset, Axis, AxisId, Bound, CmpOp, DepKind, DepVector, Domain, Edge, FufOpRef,
    LoadAddr, Node, NodeId, Predicate, Region, Role, ScalarBinding, ScalarId, SmemLookup,
    StrideExpr,
};

#[derive(Debug, Clone, Copy)]
pub enum Window {
    Finite(u32),
    Infinite,
}

#[derive(Debug, Clone, Copy)]
pub struct AttnParams {
    pub window: Window,
    pub head_dim: u32,
    pub tile_q: u32,
    pub tile_k: u32,
    pub num_head_groups: u32,
    pub pipe: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct PagedDecodeParams {
    pub head_dim: u32,
    pub tile_k: u32,
    pub num_head_groups: u32,
    pub pipe: u32,
    /// Page size in tokens; gather lookup resolves page-base addresses
    /// per `kv_tile / blocks_per_tile`.
    pub tokens_per_page: u32,
    /// `Window::Finite(W)` adds a sliding-window predicate; `Infinite`
    /// emits none. Matches the same mechanism attn_region uses.
    pub window: Window,
}

// Axis ids — fixed for the Attn template.
const Q_TILE: AxisId = 0;
const KV_TILE: AxisId = 1;
const HEAD_GROUP: AxisId = 2;

// Region-entry scalar ids.
const NUM_Q_TILES: ScalarId = 0;
const NUM_KV_TILES: ScalarId = 1;
const WINDOW_TILES: ScalarId = 2;

// Node ids.
const N_LOAD_Q: NodeId = 0;
const N_LOAD_K: NodeId = 1;
const N_LOAD_V: NodeId = 2;
const N_QK: NodeId = 3;
const N_SM: NodeId = 4;
const N_PV: NodeId = 5;
const N_STORE: NodeId = 6;

/// FA2 prefill / Gemma3 local attention instantiation of `Attn(W)`.
pub fn attn_region(p: &AttnParams) -> Region {
    let mut predicates = vec![Predicate {
        // causal (tile granularity): kv_tile ≤ q_tile
        coeffs: smallvec![(KV_TILE, 1), (Q_TILE, -1)],
        offset: AffineOffset::Const(0),
        op: CmpOp::Le,
    }];

    // Window predicate only when finite. For W=∞ const-prop would drop
    // it; we just don't emit it, so the lowering pass doesn't carry
    // dead predicates around.
    if let Window::Finite(_) = p.window {
        predicates.push(Predicate {
            // q_tile − kv_tile ≤ window_in_tiles
            coeffs: smallvec![(Q_TILE, 1), (KV_TILE, -1)],
            offset: AffineOffset::RegionEntry(WINDOW_TILES),
            op: CmpOp::Le,
        });
    }

    let head_dim_stride = p.head_dim as u64;

    let nodes = vec![
        Node {
            id: N_LOAD_Q,
            role: Role::Load,
            op: FufOpRef { tag: "load_q_tile" },
            addr: Some(LoadAddr {
                terms: smallvec![
                    AddrTerm::AxisStride {
                        axis: Q_TILE,
                        stride: StrideExpr::Const(p.tile_q as u64 * head_dim_stride),
                    },
                    AddrTerm::AxisStride {
                        axis: HEAD_GROUP,
                        stride: StrideExpr::Const(head_dim_stride),
                    },
                ],
            }),
        },
        Node {
            id: N_LOAD_K,
            role: Role::Load,
            op: FufOpRef { tag: "load_k_tile" },
            addr: Some(LoadAddr {
                terms: smallvec![
                    AddrTerm::AxisStride {
                        axis: KV_TILE,
                        stride: StrideExpr::Const(p.tile_k as u64 * head_dim_stride),
                    },
                    AddrTerm::AxisStride {
                        axis: HEAD_GROUP,
                        stride: StrideExpr::Const(head_dim_stride),
                    },
                ],
            }),
        },
        Node {
            id: N_LOAD_V,
            role: Role::Load,
            op: FufOpRef { tag: "load_v_tile" },
            addr: Some(LoadAddr {
                terms: smallvec![
                    AddrTerm::AxisStride {
                        axis: KV_TILE,
                        stride: StrideExpr::Const(p.tile_k as u64 * head_dim_stride),
                    },
                    AddrTerm::AxisStride {
                        axis: HEAD_GROUP,
                        stride: StrideExpr::Const(head_dim_stride),
                    },
                ],
            }),
        },
        Node {
            id: N_QK,
            role: Role::Compute,
            op: FufOpRef { tag: "qk_matmul" },
            addr: None,
        },
        Node {
            id: N_SM,
            role: Role::Compute,
            op: FufOpRef {
                tag: "softmax_update",
            },
            addr: None,
        },
        Node {
            id: N_PV,
            role: Role::Compute,
            op: FufOpRef { tag: "pv_matmul" },
            addr: None,
        },
        Node {
            id: N_STORE,
            role: Role::Store,
            op: FufOpRef {
                tag: "store_o_tile",
            },
            addr: Some(LoadAddr {
                // Store address has no kv_tile term — the kv_tile axis
                // is reduced for Store. (Lowering rule §4-1 of sketch.)
                terms: smallvec![
                    AddrTerm::AxisStride {
                        axis: Q_TILE,
                        stride: StrideExpr::Const(p.tile_q as u64 * head_dim_stride),
                    },
                    AddrTerm::AxisStride {
                        axis: HEAD_GROUP,
                        stride: StrideExpr::Const(head_dim_stride),
                    },
                ],
            }),
        },
    ];

    let p_depth = p.pipe as i32;

    let edges = vec![
        // Pipeline: load K[k] overlaps QK[k-P]
        Edge {
            src: N_LOAD_K,
            dst: N_QK,
            kind: DepKind::Pipeline,
            vector: DepVector(smallvec![(KV_TILE, -p_depth)]),
        },
        Edge {
            src: N_LOAD_V,
            dst: N_PV,
            kind: DepKind::Pipeline,
            vector: DepVector(smallvec![(KV_TILE, -p_depth)]),
        },
        // Raw: softmax running-state chain within a q_tile
        Edge {
            src: N_SM,
            dst: N_SM,
            kind: DepKind::Raw,
            vector: DepVector(smallvec![(KV_TILE, -1)]),
        },
        // Raw same-iteration: QK → softmax → PV → store
        Edge {
            src: N_QK,
            dst: N_SM,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
        Edge {
            src: N_SM,
            dst: N_PV,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
        Edge {
            src: N_PV,
            dst: N_STORE,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
        // load_q feeds QK same-iteration.
        Edge {
            src: N_LOAD_Q,
            dst: N_QK,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
    ];

    let mut entry_scalars = vec![
        ScalarBinding {
            id: NUM_Q_TILES,
            name: "num_q_tiles",
        },
        ScalarBinding {
            id: NUM_KV_TILES,
            name: "num_kv_tiles",
        },
    ];
    if let Window::Finite(_) = p.window {
        entry_scalars.push(ScalarBinding {
            id: WINDOW_TILES,
            name: "window_in_tiles",
        });
    }

    Region {
        id: 0,
        name: "fa2_prefill",
        domain: Domain {
            axes: vec![
                Axis {
                    id: Q_TILE,
                    name: "q_tile",
                    bound: Bound::RegionEntryScalar(NUM_Q_TILES),
                },
                Axis {
                    id: KV_TILE,
                    name: "kv_tile",
                    bound: Bound::RegionEntryScalar(NUM_KV_TILES),
                },
                Axis {
                    id: HEAD_GROUP,
                    name: "head_group",
                    bound: Bound::Const(p.num_head_groups),
                },
            ],
            predicates,
        },
        entry_scalars,
        nodes,
        edges,
        gmem_bindings: Vec::new(),
    }
}

// ─── Paged-KV decode instantiation ──────────────────────────────
//
// Same Attn shape (nodes + edges), different axes + load addresses:
//   - q_tile replaced by `b` (batch), tile_q=1 baked in (decode M=1)
//   - causal predicate dropped (M=1 trivially satisfies it)
//   - per-sequence KV length via Bound::IndexedScalar
//   - K/V addresses gather through block_table via AxisDivGather
// Exercises the gather + indexed-scalar machinery clarified in
// design §4.1, §4.2. Zero new IR types.

const B_AXIS: AxisId = 0;

// Decode-specific scalar ids (separate namespace from prefill).
const NUM_BATCHES: ScalarId = 0;
const NUM_KV_TILES_PER_B: ScalarId = 1;

pub fn attn_region_paged_decode(p: &PagedDecodeParams) -> Region {
    let head_dim_stride = p.head_dim as u64;
    let page_stride = (p.tokens_per_page as u64) * head_dim_stride;
    let tile_in_page_stride = (p.tile_k as u64) * head_dim_stride;
    let blocks_per_tile = p.tokens_per_page / p.tile_k.max(1);

    let kv_gather = |tag: &'static str| -> Node {
        Node {
            id: if tag == "load_k_tile" {
                N_LOAD_K
            } else {
                N_LOAD_V
            },
            role: Role::Load,
            op: FufOpRef { tag },
            addr: Some(LoadAddr {
                terms: smallvec![
                    AddrTerm::AxisDivGather {
                        axis: KV_TILE,
                        divisor: blocks_per_tile.max(1),
                        table: SmemLookup {
                            source: "block_table[b]",
                        },
                        stride: StrideExpr::Const(page_stride),
                    },
                    AddrTerm::AxisModStride {
                        axis: KV_TILE,
                        modulus: blocks_per_tile.max(1),
                        stride: StrideExpr::Const(tile_in_page_stride),
                    },
                    AddrTerm::AxisStride {
                        axis: HEAD_GROUP,
                        stride: StrideExpr::Const(head_dim_stride),
                    },
                ],
            }),
        }
    };

    let nodes = vec![
        Node {
            id: N_LOAD_Q,
            role: Role::Load,
            op: FufOpRef { tag: "load_q_tile" },
            addr: Some(LoadAddr {
                terms: smallvec![
                    AddrTerm::AxisStride {
                        axis: B_AXIS,
                        stride: StrideExpr::Const(head_dim_stride),
                    },
                    AddrTerm::AxisStride {
                        axis: HEAD_GROUP,
                        stride: StrideExpr::Const(head_dim_stride),
                    },
                ],
            }),
        },
        kv_gather("load_k_tile"),
        kv_gather("load_v_tile"),
        Node {
            id: N_QK,
            role: Role::Compute,
            op: FufOpRef { tag: "qk_matmul" },
            addr: None,
        },
        Node {
            id: N_SM,
            role: Role::Compute,
            op: FufOpRef {
                tag: "softmax_update",
            },
            addr: None,
        },
        Node {
            id: N_PV,
            role: Role::Compute,
            op: FufOpRef { tag: "pv_matmul" },
            addr: None,
        },
        Node {
            id: N_STORE,
            role: Role::Store,
            op: FufOpRef {
                tag: "store_o_tile",
            },
            addr: Some(LoadAddr {
                terms: smallvec![
                    AddrTerm::AxisStride {
                        axis: B_AXIS,
                        stride: StrideExpr::Const(head_dim_stride),
                    },
                    AddrTerm::AxisStride {
                        axis: HEAD_GROUP,
                        stride: StrideExpr::Const(head_dim_stride),
                    },
                ],
            }),
        },
    ];

    let p_depth = p.pipe as i32;
    let edges = vec![
        Edge {
            src: N_LOAD_K,
            dst: N_QK,
            kind: DepKind::Pipeline,
            vector: DepVector(smallvec![(KV_TILE, -p_depth)]),
        },
        Edge {
            src: N_LOAD_V,
            dst: N_PV,
            kind: DepKind::Pipeline,
            vector: DepVector(smallvec![(KV_TILE, -p_depth)]),
        },
        Edge {
            src: N_SM,
            dst: N_SM,
            kind: DepKind::Raw,
            vector: DepVector(smallvec![(KV_TILE, -1)]),
        },
        Edge {
            src: N_QK,
            dst: N_SM,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
        Edge {
            src: N_SM,
            dst: N_PV,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
        Edge {
            src: N_PV,
            dst: N_STORE,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
        Edge {
            src: N_LOAD_Q,
            dst: N_QK,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
    ];

    // Sliding-window predicate — kv_tile ≤ window_in_tiles — mirrors
    // the prefill attn_region's finite-window handling (design §4.2).
    // For `Window::Infinite` we emit nothing; const-prop at the
    // lowering layer would drop it anyway.
    let mut predicates: Vec<Predicate> = Vec::new();
    let mut entry_scalars = vec![
        ScalarBinding {
            id: NUM_BATCHES,
            name: "num_batches",
        },
        ScalarBinding {
            id: NUM_KV_TILES_PER_B,
            name: "num_kv_tiles_per_b",
        },
    ];
    if let Window::Finite(_) = p.window {
        const WINDOW_TILES_DECODE: ScalarId = 2;
        predicates.push(Predicate {
            coeffs: smallvec![(KV_TILE, 1)],
            offset: AffineOffset::RegionEntry(WINDOW_TILES_DECODE),
            op: CmpOp::Le,
        });
        entry_scalars.push(ScalarBinding {
            id: WINDOW_TILES_DECODE,
            name: "window_in_tiles",
        });
    }

    Region {
        id: 1,
        name: "paged_decode",
        domain: Domain {
            axes: vec![
                Axis {
                    id: B_AXIS,
                    name: "b",
                    bound: Bound::RegionEntryScalar(NUM_BATCHES),
                },
                Axis {
                    id: KV_TILE,
                    name: "kv_tile",
                    // Per-sequence bound: axis range depends on batch.
                    bound: Bound::IndexedScalar(NUM_KV_TILES_PER_B, B_AXIS),
                },
                Axis {
                    id: HEAD_GROUP,
                    name: "head_group",
                    bound: Bound::Const(p.num_head_groups),
                },
            ],
            predicates,
        },
        entry_scalars,
        nodes,
        edges,
        gmem_bindings: Vec::new(),
    }
}

// ─── GEMM ───────────────────────────────────────────────────────
//
// Row-major tiled matmul: C[M,N] += A[M,K] · B[K,N].
//
// Domain: (m_tile, n_tile) parallel, k_tile serial (reduction).
// Load A/B tiles pipelined over k; Compute accumulates across k;
// Store fires at the end of the k loop, outside it. This is the
// workhorse template for projections (qkv, o, gate, up, down,
// lm_head) once lowering routes them here.

#[derive(Debug, Clone, Copy)]
pub struct GemmParams {
    pub m_tile: u32,
    pub n_tile: u32,
    pub k_tile: u32,
    pub pipe: u32,
}

pub fn gemm_region(p: &GemmParams) -> Region {
    const M: AxisId = 0;
    const N: AxisId = 1;
    const K: AxisId = 2;
    const NUM_M: ScalarId = 0;
    const NUM_N: ScalarId = 1;
    const NUM_K: ScalarId = 2;
    const N_LOAD_A: NodeId = 0;
    const N_LOAD_B: NodeId = 1;
    const N_GEMM: NodeId = 2;
    const N_STORE: NodeId = 3;

    let m_stride = p.m_tile as u64;
    let n_stride = p.n_tile as u64;
    let k_stride = p.k_tile as u64;

    let nodes = vec![
        Node {
            id: N_LOAD_A,
            role: Role::Load,
            op: FufOpRef { tag: "load_a_tile" },
            addr: Some(LoadAddr {
                terms: smallvec![
                    AddrTerm::AxisStride {
                        axis: M,
                        stride: StrideExpr::Const(m_stride),
                    },
                    AddrTerm::AxisStride {
                        axis: K,
                        stride: StrideExpr::Const(k_stride),
                    },
                ],
            }),
        },
        Node {
            id: N_LOAD_B,
            role: Role::Load,
            op: FufOpRef { tag: "load_b_tile" },
            addr: Some(LoadAddr {
                terms: smallvec![
                    AddrTerm::AxisStride {
                        axis: K,
                        stride: StrideExpr::Const(k_stride),
                    },
                    AddrTerm::AxisStride {
                        axis: N,
                        stride: StrideExpr::Const(n_stride),
                    },
                ],
            }),
        },
        Node {
            id: N_GEMM,
            role: Role::Compute,
            op: FufOpRef {
                tag: "gemm_accumulate",
            },
            addr: None,
        },
        Node {
            id: N_STORE,
            role: Role::Store,
            op: FufOpRef {
                tag: "store_c_tile",
            },
            addr: Some(LoadAddr {
                // K is reduced; Store addr omits it.
                terms: smallvec![
                    AddrTerm::AxisStride {
                        axis: M,
                        stride: StrideExpr::Const(m_stride),
                    },
                    AddrTerm::AxisStride {
                        axis: N,
                        stride: StrideExpr::Const(n_stride),
                    },
                ],
            }),
        },
    ];

    let p_depth = p.pipe as i32;
    let edges = vec![
        Edge {
            src: N_LOAD_A,
            dst: N_GEMM,
            kind: DepKind::Pipeline,
            vector: DepVector(smallvec![(K, -p_depth)]),
        },
        Edge {
            src: N_LOAD_B,
            dst: N_GEMM,
            kind: DepKind::Pipeline,
            vector: DepVector(smallvec![(K, -p_depth)]),
        },
        // Accumulation chain across k.
        Edge {
            src: N_GEMM,
            dst: N_GEMM,
            kind: DepKind::Raw,
            vector: DepVector(smallvec![(K, -1)]),
        },
        Edge {
            src: N_GEMM,
            dst: N_STORE,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
    ];

    Region {
        id: 0,
        name: "gemm",
        domain: Domain {
            axes: vec![
                Axis {
                    id: M,
                    name: "m_tile",
                    bound: Bound::RegionEntryScalar(NUM_M),
                },
                Axis {
                    id: N,
                    name: "n_tile",
                    bound: Bound::RegionEntryScalar(NUM_N),
                },
                Axis {
                    id: K,
                    name: "k_tile",
                    bound: Bound::RegionEntryScalar(NUM_K),
                },
            ],
            predicates: vec![],
        },
        entry_scalars: vec![
            ScalarBinding {
                id: NUM_M,
                name: "num_m_tiles",
            },
            ScalarBinding {
                id: NUM_N,
                name: "num_n_tiles",
            },
            ScalarBinding {
                id: NUM_K,
                name: "num_k_tiles",
            },
        ],
        nodes,
        edges,
        gmem_bindings: Vec::new(),
    }
}

// ─── RMSNorm ────────────────────────────────────────────────────
//
// Per-row normalization: y = x * weight / rms(x). Each CTA owns a
// token tile; the hidden-dim reduction happens inside the Compute
// node (warp-wide reduction via the intrinsic expansion). No serial
// axis — this region is straight-line.

#[derive(Debug, Clone, Copy)]
pub struct RmsNormParams {
    pub hidden_dim: u32,
    pub token_tile: u32,
}

pub fn rmsnorm_region(p: &RmsNormParams) -> Region {
    const TOKEN: AxisId = 0;
    const NUM_TOKEN_TILES: ScalarId = 0;
    const N_LOAD_X: NodeId = 0;
    const N_LOAD_W: NodeId = 1;
    const N_COMPUTE: NodeId = 2;
    const N_STORE_Y: NodeId = 3;

    let tile_stride = (p.hidden_dim as u64) * (p.token_tile as u64);

    let nodes = vec![
        Node {
            id: N_LOAD_X,
            role: Role::Load,
            op: FufOpRef { tag: "load_x_row" },
            addr: Some(LoadAddr {
                terms: smallvec![AddrTerm::AxisStride {
                    axis: TOKEN,
                    stride: StrideExpr::Const(tile_stride),
                }],
            }),
        },
        Node {
            id: N_LOAD_W,
            role: Role::Load,
            op: FufOpRef { tag: "load_weight" },
            // Weight is invariant across token_tile — no axis terms.
            // The scheduler hoists this to preamble naturally.
            addr: Some(LoadAddr { terms: smallvec![] }),
        },
        Node {
            id: N_COMPUTE,
            role: Role::Compute,
            op: FufOpRef {
                tag: "rmsnorm_compute",
            },
            addr: None,
        },
        Node {
            id: N_STORE_Y,
            role: Role::Store,
            op: FufOpRef { tag: "store_y_row" },
            addr: Some(LoadAddr {
                terms: smallvec![AddrTerm::AxisStride {
                    axis: TOKEN,
                    stride: StrideExpr::Const(tile_stride),
                }],
            }),
        },
    ];

    let edges = vec![
        Edge {
            src: N_LOAD_X,
            dst: N_COMPUTE,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
        Edge {
            src: N_LOAD_W,
            dst: N_COMPUTE,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
        Edge {
            src: N_COMPUTE,
            dst: N_STORE_Y,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
    ];

    Region {
        id: 0,
        name: "rmsnorm",
        domain: Domain {
            axes: vec![Axis {
                id: TOKEN,
                name: "token_tile",
                bound: Bound::RegionEntryScalar(NUM_TOKEN_TILES),
            }],
            predicates: vec![],
        },
        entry_scalars: vec![ScalarBinding {
            id: NUM_TOKEN_TILES,
            name: "num_token_tiles",
        }],
        nodes,
        edges,
        gmem_bindings: Vec::new(),
    }
}

// ─── Elementwise residual add ───────────────────────────────────
//
// y = a + b, element-wise, per token-tile. Two loads, one compute,
// one store, all straight-line. Exercises the minimum region shape:
// one parallel axis, no serial axis, no pipeline depth.

#[derive(Debug, Clone, Copy)]
pub struct ResidualAddParams {
    pub hidden_dim: u32,
    pub token_tile: u32,
}

pub fn residual_add_region(p: &ResidualAddParams) -> Region {
    const TOKEN: AxisId = 0;
    const NUM_TOKEN_TILES: ScalarId = 0;
    const N_LOAD_A: NodeId = 0;
    const N_LOAD_B: NodeId = 1;
    const N_ADD: NodeId = 2;
    const N_STORE: NodeId = 3;

    let tile_stride = (p.hidden_dim as u64) * (p.token_tile as u64);
    let row_addr = || LoadAddr {
        terms: smallvec![AddrTerm::AxisStride {
            axis: TOKEN,
            stride: StrideExpr::Const(tile_stride),
        }],
    };

    let nodes = vec![
        Node {
            id: N_LOAD_A,
            role: Role::Load,
            op: FufOpRef { tag: "load_a_row" },
            addr: Some(row_addr()),
        },
        Node {
            id: N_LOAD_B,
            role: Role::Load,
            op: FufOpRef { tag: "load_b_row" },
            addr: Some(row_addr()),
        },
        Node {
            id: N_ADD,
            role: Role::Compute,
            op: FufOpRef {
                tag: "elementwise_add",
            },
            addr: None,
        },
        Node {
            id: N_STORE,
            role: Role::Store,
            op: FufOpRef {
                tag: "store_sum_row",
            },
            addr: Some(row_addr()),
        },
    ];

    let edges = vec![
        Edge {
            src: N_LOAD_A,
            dst: N_ADD,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
        Edge {
            src: N_LOAD_B,
            dst: N_ADD,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
        Edge {
            src: N_ADD,
            dst: N_STORE,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
    ];

    Region {
        id: 0,
        name: "residual_add",
        domain: Domain {
            axes: vec![Axis {
                id: TOKEN,
                name: "token_tile",
                bound: Bound::RegionEntryScalar(NUM_TOKEN_TILES),
            }],
            predicates: vec![],
        },
        entry_scalars: vec![ScalarBinding {
            id: NUM_TOKEN_TILES,
            name: "num_token_tiles",
        }],
        nodes,
        edges,
        gmem_bindings: Vec::new(),
    }
}

// ─── Embedding lookup ──────────────────────────────────────────
//
// Gather from embed_table[vocab, hidden] indexed by token_ids.
// Expressed as one Load (gather via AxisDivGather with divisor=1 —
// the table term is `token_ids[token_tile]`) directly flowing into
// one Store. No Compute node: embedding lookup is pure data movement.

#[derive(Debug, Clone, Copy)]
pub struct EmbedParams {
    pub hidden_dim: u32,
    pub token_tile: u32,
}

pub fn embed_region(p: &EmbedParams) -> Region {
    const TOKEN: AxisId = 0;
    const NUM_TOKEN_TILES: ScalarId = 0;
    const N_LOAD: NodeId = 0;
    const N_STORE: NodeId = 1;

    let hidden_stride = p.hidden_dim as u64;
    let tile_stride = hidden_stride * (p.token_tile as u64);

    let nodes = vec![
        Node {
            id: N_LOAD,
            role: Role::Load,
            op: FufOpRef {
                tag: "load_embed_row",
            },
            addr: Some(LoadAddr {
                terms: smallvec![AddrTerm::AxisDivGather {
                    axis: TOKEN,
                    divisor: 1,
                    table: SmemLookup {
                        source: "token_ids",
                    },
                    stride: StrideExpr::Const(hidden_stride),
                }],
            }),
        },
        Node {
            id: N_STORE,
            role: Role::Store,
            op: FufOpRef {
                tag: "store_embed_row",
            },
            addr: Some(LoadAddr {
                terms: smallvec![AddrTerm::AxisStride {
                    axis: TOKEN,
                    stride: StrideExpr::Const(tile_stride),
                }],
            }),
        },
    ];
    let edges = vec![Edge {
        src: N_LOAD,
        dst: N_STORE,
        kind: DepKind::Raw,
        vector: DepVector::default(),
    }];

    Region {
        id: 0,
        name: "embed",
        domain: Domain {
            axes: vec![Axis {
                id: TOKEN,
                name: "token_tile",
                bound: Bound::RegionEntryScalar(NUM_TOKEN_TILES),
            }],
            predicates: vec![],
        },
        entry_scalars: vec![ScalarBinding {
            id: NUM_TOKEN_TILES,
            name: "num_token_tiles",
        }],
        nodes,
        edges,
        gmem_bindings: Vec::new(),
    }
}

// ─── Unary in-place element-wise ──────────────────────────────
//
// Smallest possible region: one Load, one Compute, one Store; a
// single parallel axis over token tiles. Parameterized on the
// compute-op tag so a single template serves scalar_mul_inplace,
// tanh_softcap_inplace, and other tag-only variants of the same
// shape. The activation-vs-multiply choice lives at emit_ops, not
// here.

#[derive(Debug, Clone, Copy)]
pub struct UnaryInplaceParams {
    pub hidden_dim: u32,
    pub token_tile: u32,
    /// FUF op tag that identifies the compute. Flows into the
    /// emitted node and picks its intrinsic expansion in emit_ops.
    pub op_tag: &'static str,
}

pub fn unary_inplace_region(p: &UnaryInplaceParams) -> Region {
    const TOKEN: AxisId = 0;
    const NUM_TOKEN_TILES: ScalarId = 0;
    const N_LOAD: NodeId = 0;
    const N_COMPUTE: NodeId = 1;
    const N_STORE: NodeId = 2;

    let tile_stride = (p.hidden_dim as u64) * (p.token_tile as u64);
    let row_addr = || LoadAddr {
        terms: smallvec![AddrTerm::AxisStride {
            axis: TOKEN,
            stride: StrideExpr::Const(tile_stride),
        }],
    };

    let nodes = vec![
        Node {
            id: N_LOAD,
            role: Role::Load,
            op: FufOpRef { tag: "load_x_row" },
            addr: Some(row_addr()),
        },
        Node {
            id: N_COMPUTE,
            role: Role::Compute,
            op: FufOpRef { tag: p.op_tag },
            addr: None,
        },
        Node {
            id: N_STORE,
            role: Role::Store,
            op: FufOpRef { tag: "store_y_row" },
            addr: Some(row_addr()),
        },
    ];
    let edges = vec![
        Edge {
            src: N_LOAD,
            dst: N_COMPUTE,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
        Edge {
            src: N_COMPUTE,
            dst: N_STORE,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
    ];

    Region {
        id: 0,
        name: "unary_inplace",
        domain: Domain {
            axes: vec![Axis {
                id: TOKEN,
                name: "token_tile",
                bound: Bound::RegionEntryScalar(NUM_TOKEN_TILES),
            }],
            predicates: vec![],
        },
        entry_scalars: vec![ScalarBinding {
            id: NUM_TOKEN_TILES,
            name: "num_token_tiles",
        }],
        nodes,
        edges,
        gmem_bindings: Vec::new(),
    }
}

// ─── QKV + RoPE ────────────────────────────────────────────────
//
// Fused op covering the qkv projection + RoPE rotation + (optional)
// KV cache write at the entry of every attention block. Shape:
// hidden_x[token, hidden] → q[token, num_q, head_dim],
// k/v[token, num_kv, head_dim] or a KV cache append.
//
// Domain: (token_tile, head_tile) parallel, k_tile serial (reduction
// over hidden for the projection GEMMs). Three Load streams (x,
// Wqkv, rope_coef) feed the Compute; Stores emit the three outputs.
// The RoPE rotation is a Compute step after the projection — the
// template carries it as a separate node so the emitter can see
// where the post-process fires; the exact intrinsic sequence lives
// in emit_ops per arch.
//
// Quantized variants (marlin_/bnb4_) share this stencil; they only
// differ in the Load_w address expansion, which is emit_ops's job.

#[derive(Debug, Clone, Copy)]
pub struct QkvRopeParams {
    pub hidden_dim: u32,
    pub head_dim: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub token_tile: u32,
    pub k_tile: u32,
    pub pipe: u32,
    /// When true, the template emits K/V stores as writes into the
    /// paged KV cache (via an indexed gather on block_table); when
    /// false, they are direct stores. Runtime-visible difference only
    /// at emit_ops expansion; the region shape is identical.
    pub writes_kv_cache: bool,
}

pub fn qkv_rope_region(p: &QkvRopeParams) -> Region {
    const TOKEN: AxisId = 0;
    const HEAD: AxisId = 1;
    const K: AxisId = 2;
    const NUM_TOKEN_TILES: ScalarId = 0;
    const NUM_HEAD_TILES: ScalarId = 1;
    const NUM_K_TILES: ScalarId = 2;
    const N_LOAD_X: NodeId = 0;
    const N_LOAD_W: NodeId = 1;
    const N_QKV_GEMM: NodeId = 2;
    const N_LOAD_ROPE: NodeId = 3;
    const N_APPLY_ROPE: NodeId = 4;
    const N_STORE_Q: NodeId = 5;
    const N_STORE_K: NodeId = 6;
    const N_STORE_V: NodeId = 7;

    let token_stride = (p.hidden_dim as u64) * (p.token_tile as u64);
    let head_stride = p.head_dim as u64;
    let k_stride = p.k_tile as u64;

    let store_tag_k = if p.writes_kv_cache {
        "store_k_cache"
    } else {
        "store_k_row"
    };
    let store_tag_v = if p.writes_kv_cache {
        "store_v_cache"
    } else {
        "store_v_row"
    };

    let nodes = vec![
        Node {
            id: N_LOAD_X,
            role: Role::Load,
            op: FufOpRef { tag: "load_x_row" },
            addr: Some(LoadAddr {
                terms: smallvec![
                    AddrTerm::AxisStride {
                        axis: TOKEN,
                        stride: StrideExpr::Const(token_stride),
                    },
                    AddrTerm::AxisStride {
                        axis: K,
                        stride: StrideExpr::Const(k_stride),
                    },
                ],
            }),
        },
        Node {
            id: N_LOAD_W,
            role: Role::Load,
            op: FufOpRef {
                tag: "load_wqkv_tile",
            },
            addr: Some(LoadAddr {
                terms: smallvec![
                    AddrTerm::AxisStride {
                        axis: HEAD,
                        stride: StrideExpr::Const(head_stride),
                    },
                    AddrTerm::AxisStride {
                        axis: K,
                        stride: StrideExpr::Const(k_stride),
                    },
                ],
            }),
        },
        Node {
            id: N_QKV_GEMM,
            role: Role::Compute,
            op: FufOpRef { tag: "qkv_matmul" },
            addr: None,
        },
        Node {
            id: N_LOAD_ROPE,
            role: Role::Load,
            op: FufOpRef {
                tag: "load_rope_coef",
            },
            // Cos/sin tables are per-token (same across heads). Live
            // in the preamble; no serial-axis dep.
            addr: Some(LoadAddr {
                terms: smallvec![AddrTerm::AxisStride {
                    axis: TOKEN,
                    stride: StrideExpr::Const((p.head_dim as u64) / 2),
                }],
            }),
        },
        Node {
            id: N_APPLY_ROPE,
            role: Role::Compute,
            op: FufOpRef { tag: "apply_rope" },
            addr: None,
        },
        Node {
            id: N_STORE_Q,
            role: Role::Store,
            op: FufOpRef { tag: "store_q_row" },
            addr: Some(LoadAddr {
                // Per-head output; K is reduced out.
                terms: smallvec![
                    AddrTerm::AxisStride {
                        axis: TOKEN,
                        stride: StrideExpr::Const(token_stride),
                    },
                    AddrTerm::AxisStride {
                        axis: HEAD,
                        stride: StrideExpr::Const(head_stride),
                    },
                ],
            }),
        },
        Node {
            id: N_STORE_K,
            role: Role::Store,
            op: FufOpRef { tag: store_tag_k },
            addr: Some(LoadAddr {
                terms: smallvec![
                    AddrTerm::AxisStride {
                        axis: TOKEN,
                        stride: StrideExpr::Const(token_stride),
                    },
                    AddrTerm::AxisStride {
                        axis: HEAD,
                        stride: StrideExpr::Const(head_stride),
                    },
                ],
            }),
        },
        Node {
            id: N_STORE_V,
            role: Role::Store,
            op: FufOpRef { tag: store_tag_v },
            addr: Some(LoadAddr {
                terms: smallvec![
                    AddrTerm::AxisStride {
                        axis: TOKEN,
                        stride: StrideExpr::Const(token_stride),
                    },
                    AddrTerm::AxisStride {
                        axis: HEAD,
                        stride: StrideExpr::Const(head_stride),
                    },
                ],
            }),
        },
    ];

    let p_depth = p.pipe as i32;
    let edges = vec![
        // Pipeline loads on K into the projection GEMM.
        Edge {
            src: N_LOAD_X,
            dst: N_QKV_GEMM,
            kind: DepKind::Pipeline,
            vector: DepVector(smallvec![(K, -p_depth)]),
        },
        Edge {
            src: N_LOAD_W,
            dst: N_QKV_GEMM,
            kind: DepKind::Pipeline,
            vector: DepVector(smallvec![(K, -p_depth)]),
        },
        // Accumulation chain on K.
        Edge {
            src: N_QKV_GEMM,
            dst: N_QKV_GEMM,
            kind: DepKind::Raw,
            vector: DepVector(smallvec![(K, -1)]),
        },
        // rope_coef feeds apply_rope same-iter.
        Edge {
            src: N_LOAD_ROPE,
            dst: N_APPLY_ROPE,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
        // Projection result feeds RoPE.
        Edge {
            src: N_QKV_GEMM,
            dst: N_APPLY_ROPE,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
        // Stores pull from the rotated fragment.
        Edge {
            src: N_APPLY_ROPE,
            dst: N_STORE_Q,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
        Edge {
            src: N_APPLY_ROPE,
            dst: N_STORE_K,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
        // V doesn't get RoPE'd; pull it straight from the projection.
        Edge {
            src: N_QKV_GEMM,
            dst: N_STORE_V,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
    ];

    // Suppress unused-warnings for params that matter at emit time
    // but not in the template (num_q_heads, num_kv_heads inform the
    // intrinsic expansion's per-head layout, not the stencil shape).
    let _ = (p.num_q_heads, p.num_kv_heads);

    Region {
        id: 0,
        name: "qkv_rope",
        domain: Domain {
            axes: vec![
                Axis {
                    id: TOKEN,
                    name: "token_tile",
                    bound: Bound::RegionEntryScalar(NUM_TOKEN_TILES),
                },
                Axis {
                    id: HEAD,
                    name: "head_tile",
                    bound: Bound::RegionEntryScalar(NUM_HEAD_TILES),
                },
                Axis {
                    id: K,
                    name: "k_tile",
                    bound: Bound::RegionEntryScalar(NUM_K_TILES),
                },
            ],
            predicates: vec![],
        },
        entry_scalars: vec![
            ScalarBinding {
                id: NUM_TOKEN_TILES,
                name: "num_token_tiles",
            },
            ScalarBinding {
                id: NUM_HEAD_TILES,
                name: "num_head_tiles",
            },
            ScalarBinding {
                id: NUM_K_TILES,
                name: "num_k_tiles",
            },
        ],
        nodes,
        edges,
        gmem_bindings: Vec::new(),
    }
}

// ─── Gate + Up + SiLU/GeLU + Mul (MLP input) ───────────────────
//
// Fused op at the MLP's entry:
//   gate = Wgate · x
//   up   = Wup   · x
//   out  = activation(gate) * up
// where activation is silu or gelu — identical stencil shape, the
// intrinsic expansion picks the activation.
//
// Two parallel projection chains (gate_gemm, up_gemm) share the x
// input and meet at the silu_mul Compute; a single Store emits the
// fused output. This is where the design doc's "multiple Compute
// nodes feeding a single Store through Raw same-iter edges" pattern
// gets its stress test.

#[derive(Debug, Clone, Copy)]
pub struct GateUpSiluMulParams {
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub token_tile: u32,
    pub inter_tile: u32,
    pub k_tile: u32,
    pub pipe: u32,
}

pub fn gate_up_silu_mul_region(p: &GateUpSiluMulParams) -> Region {
    const TOKEN: AxisId = 0;
    const INTER: AxisId = 1;
    const K: AxisId = 2;
    const NUM_TOKEN_TILES: ScalarId = 0;
    const NUM_INTER_TILES: ScalarId = 1;
    const NUM_K_TILES: ScalarId = 2;
    const N_LOAD_X: NodeId = 0;
    const N_LOAD_WG: NodeId = 1;
    const N_LOAD_WU: NodeId = 2;
    const N_GATE_GEMM: NodeId = 3;
    const N_UP_GEMM: NodeId = 4;
    const N_SILU_MUL: NodeId = 5;
    const N_STORE: NodeId = 6;

    let token_stride = (p.hidden_dim as u64) * (p.token_tile as u64);
    let inter_stride = (p.intermediate_dim as u64) * (p.inter_tile as u64);
    let k_stride = p.k_tile as u64;

    // All three loads share the same axis shape: x is [token, k],
    // Wgate/Wup are [inter, k]. Separate functions to keep addresses
    // explicit in the template for the reader.
    let x_addr = LoadAddr {
        terms: smallvec![
            AddrTerm::AxisStride {
                axis: TOKEN,
                stride: StrideExpr::Const(token_stride),
            },
            AddrTerm::AxisStride {
                axis: K,
                stride: StrideExpr::Const(k_stride),
            },
        ],
    };
    let w_addr = || LoadAddr {
        terms: smallvec![
            AddrTerm::AxisStride {
                axis: INTER,
                stride: StrideExpr::Const(inter_stride),
            },
            AddrTerm::AxisStride {
                axis: K,
                stride: StrideExpr::Const(k_stride),
            },
        ],
    };

    let nodes = vec![
        Node {
            id: N_LOAD_X,
            role: Role::Load,
            op: FufOpRef { tag: "load_x_row" },
            addr: Some(x_addr),
        },
        Node {
            id: N_LOAD_WG,
            role: Role::Load,
            op: FufOpRef {
                tag: "load_wgate_tile",
            },
            addr: Some(w_addr()),
        },
        Node {
            id: N_LOAD_WU,
            role: Role::Load,
            op: FufOpRef {
                tag: "load_wup_tile",
            },
            addr: Some(w_addr()),
        },
        Node {
            id: N_GATE_GEMM,
            role: Role::Compute,
            op: FufOpRef {
                tag: "gate_gemm_accumulate",
            },
            addr: None,
        },
        Node {
            id: N_UP_GEMM,
            role: Role::Compute,
            op: FufOpRef {
                tag: "up_gemm_accumulate",
            },
            addr: None,
        },
        Node {
            id: N_SILU_MUL,
            role: Role::Compute,
            op: FufOpRef {
                tag: "silu_mul_fuse",
            },
            addr: None,
        },
        Node {
            id: N_STORE,
            role: Role::Store,
            op: FufOpRef {
                tag: "store_inter_tile",
            },
            addr: Some(LoadAddr {
                // K is reduced; output addr uses token and inter.
                terms: smallvec![
                    AddrTerm::AxisStride {
                        axis: TOKEN,
                        stride: StrideExpr::Const(token_stride),
                    },
                    AddrTerm::AxisStride {
                        axis: INTER,
                        stride: StrideExpr::Const(inter_stride),
                    },
                ],
            }),
        },
    ];

    let p_depth = p.pipe as i32;
    let edges = vec![
        // Pipeline: all three loads feed their respective GEMMs on
        // k-lag P. load_x splits to both GEMMs.
        Edge {
            src: N_LOAD_X,
            dst: N_GATE_GEMM,
            kind: DepKind::Pipeline,
            vector: DepVector(smallvec![(K, -p_depth)]),
        },
        Edge {
            src: N_LOAD_X,
            dst: N_UP_GEMM,
            kind: DepKind::Pipeline,
            vector: DepVector(smallvec![(K, -p_depth)]),
        },
        Edge {
            src: N_LOAD_WG,
            dst: N_GATE_GEMM,
            kind: DepKind::Pipeline,
            vector: DepVector(smallvec![(K, -p_depth)]),
        },
        Edge {
            src: N_LOAD_WU,
            dst: N_UP_GEMM,
            kind: DepKind::Pipeline,
            vector: DepVector(smallvec![(K, -p_depth)]),
        },
        // Accumulation chains (one per GEMM).
        Edge {
            src: N_GATE_GEMM,
            dst: N_GATE_GEMM,
            kind: DepKind::Raw,
            vector: DepVector(smallvec![(K, -1)]),
        },
        Edge {
            src: N_UP_GEMM,
            dst: N_UP_GEMM,
            kind: DepKind::Raw,
            vector: DepVector(smallvec![(K, -1)]),
        },
        // Both projections feed the fuse node.
        Edge {
            src: N_GATE_GEMM,
            dst: N_SILU_MUL,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
        Edge {
            src: N_UP_GEMM,
            dst: N_SILU_MUL,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
        // Fuse output to store.
        Edge {
            src: N_SILU_MUL,
            dst: N_STORE,
            kind: DepKind::Raw,
            vector: DepVector::default(),
        },
    ];

    Region {
        id: 0,
        name: "gate_up_silu_mul",
        domain: Domain {
            axes: vec![
                Axis {
                    id: TOKEN,
                    name: "token_tile",
                    bound: Bound::RegionEntryScalar(NUM_TOKEN_TILES),
                },
                Axis {
                    id: INTER,
                    name: "inter_tile",
                    bound: Bound::RegionEntryScalar(NUM_INTER_TILES),
                },
                Axis {
                    id: K,
                    name: "k_tile",
                    bound: Bound::RegionEntryScalar(NUM_K_TILES),
                },
            ],
            predicates: vec![],
        },
        entry_scalars: vec![
            ScalarBinding {
                id: NUM_TOKEN_TILES,
                name: "num_token_tiles",
            },
            ScalarBinding {
                id: NUM_INTER_TILES,
                name: "num_inter_tiles",
            },
            ScalarBinding {
                id: NUM_K_TILES,
                name: "num_k_tiles",
            },
        ],
        nodes,
        edges,
        gmem_bindings: Vec::new(),
    }
}

#[cfg(test)]
mod new_template_tests {
    use super::*;
    use crate::arch::{sm89_fa2, sm90_fa2};
    use crate::ir;
    use crate::schedule::{classify_axes, region_pipeline_depth, topo_order_within_iter};
    use crate::wavefront::schedule_wavefront;

    #[test]
    fn gemm_region_validates_and_schedules() {
        let r = gemm_region(&GemmParams {
            m_tile: 128,
            n_tile: 128,
            k_tile: 32,
            pipe: 3,
        });
        ir::validate(&r).expect("gemm region validates");
        assert_eq!(r.nodes.len(), 4);
        assert_eq!(r.edges.len(), 4);
        assert_eq!(r.domain.axes.len(), 3);

        // K is the serial (reduction) axis; M and N are parallel.
        let classes = classify_axes(&r);
        assert_eq!(classes.len(), 3);
        assert_eq!(region_pipeline_depth(&r), 3);
        let topo = topo_order_within_iter(&r);
        assert_eq!(topo.len(), r.nodes.len());

        // Schedule: preamble empty (all Loads mention serial K),
        // body has 2 Pipeline loads + 1 Compute, epilogue has Store.
        let sched = schedule_wavefront(&r, &sm90_fa2()).unwrap();
        assert_eq!(sched.preamble.len(), 0);
        assert_eq!(sched.body.len(), 3, "2 loads + 1 compute in body");
        assert_eq!(sched.epilogue.len(), 1, "store at the end");
        assert!(sched.serial_axis.is_some());
        assert_eq!(sched.pipeline_depth, 3);
    }

    #[test]
    fn rmsnorm_region_validates_and_schedules_straight_line() {
        let r = rmsnorm_region(&RmsNormParams {
            hidden_dim: 4096,
            token_tile: 64,
        });
        ir::validate(&r).expect("rmsnorm region validates");
        assert_eq!(r.nodes.len(), 4);
        assert_eq!(r.domain.axes.len(), 1);
        assert_eq!(region_pipeline_depth(&r), 0, "no Pipeline edges");

        // Straight-line: no serial axis, preamble has loads, body has
        // compute, epilogue has store.
        let sched = schedule_wavefront(&r, &sm89_fa2()).unwrap();
        assert!(sched.serial_axis.is_none());
        assert_eq!(sched.preamble.len(), 2, "load_x + load_weight");
        assert_eq!(sched.body.len(), 1, "rmsnorm_compute");
        assert_eq!(sched.epilogue.len(), 1, "store_y_row");
    }

    #[test]
    fn embed_region_validates_as_load_to_store_no_compute() {
        let r = embed_region(&EmbedParams {
            hidden_dim: 4096,
            token_tile: 64,
        });
        ir::validate(&r).expect("embed region validates");
        // No Compute node: gather-only region.
        assert_eq!(r.nodes.len(), 2);
        assert!(r.nodes.iter().all(|n| !matches!(n.role, Role::Compute)));
        let sched = schedule_wavefront(&r, &sm89_fa2()).unwrap();
        assert!(sched.serial_axis.is_none());
        assert_eq!(sched.preamble.len(), 1, "load_embed_row in preamble");
        assert_eq!(sched.body.len(), 0);
        assert_eq!(sched.epilogue.len(), 1, "store_embed_row in epilogue");
    }

    #[test]
    fn unary_inplace_region_validates_and_carries_op_tag() {
        let r = unary_inplace_region(&UnaryInplaceParams {
            hidden_dim: 4096,
            token_tile: 64,
            op_tag: "scalar_mul",
        });
        ir::validate(&r).expect("unary_inplace validates");
        assert_eq!(r.nodes.len(), 3);
        // The parameterized op tag flows through to the Compute node.
        assert_eq!(r.nodes[1].op.tag, "scalar_mul");
        let sched = schedule_wavefront(&r, &sm89_fa2()).unwrap();
        assert!(sched.serial_axis.is_none());
        assert_eq!(sched.preamble.len(), 1);
        assert_eq!(sched.body.len(), 1);
        assert_eq!(sched.epilogue.len(), 1);
    }

    #[test]
    fn paged_decode_with_finite_window_emits_predicate() {
        let r = attn_region_paged_decode(&PagedDecodeParams {
            head_dim: 128,
            tile_k: 64,
            num_head_groups: 8,
            pipe: 3,
            tokens_per_page: 256,
            window: Window::Finite(4096),
        });
        ir::validate(&r).expect("sliding paged decode validates");
        assert_eq!(
            r.domain.predicates.len(),
            1,
            "sliding window adds one predicate"
        );
        assert!(
            r.entry_scalars.iter().any(|s| s.name == "window_in_tiles"),
            "window_in_tiles scalar present under Finite window"
        );
    }

    #[test]
    fn gate_up_silu_mul_region_validates_and_schedules() {
        let r = gate_up_silu_mul_region(&GateUpSiluMulParams {
            hidden_dim: 4096,
            intermediate_dim: 14336,
            token_tile: 64,
            inter_tile: 128,
            k_tile: 32,
            pipe: 3,
        });
        ir::validate(&r).expect("gate_up_silu_mul validates");
        assert_eq!(
            r.nodes.len(),
            7,
            "load_x, load_wg, load_wu, gate_gemm, up_gemm, silu_mul, store"
        );
        assert_eq!(r.domain.axes.len(), 3);
        assert_eq!(region_pipeline_depth(&r), 3);
        let sched = schedule_wavefront(&r, &sm90_fa2()).unwrap();
        assert!(sched.serial_axis.is_some());
        assert_eq!(sched.preamble.len(), 0);
        assert_eq!(
            sched.body.len(),
            6,
            "3 loads + 2 gemms + 1 silu_mul inside serial k-loop"
        );
        assert_eq!(sched.epilogue.len(), 1, "store_inter_tile");
    }

    #[test]
    fn qkv_rope_region_validates_and_schedules() {
        let r = qkv_rope_region(&QkvRopeParams {
            hidden_dim: 4096,
            head_dim: 128,
            num_q_heads: 32,
            num_kv_heads: 8,
            token_tile: 64,
            k_tile: 32,
            pipe: 3,
            writes_kv_cache: true,
        });
        ir::validate(&r).expect("qkv_rope region validates");
        assert_eq!(r.nodes.len(), 8, "x, wqkv, gemm, rope_coef, rope, q, k, v");
        assert_eq!(r.domain.axes.len(), 3);
        assert_eq!(region_pipeline_depth(&r), 3);

        let sched = schedule_wavefront(&r, &sm90_fa2()).unwrap();
        assert!(sched.serial_axis.is_some(), "K reduces serially");
        // load_rope_coef doesn't mention the serial axis → preamble.
        assert_eq!(sched.preamble.len(), 1);
        // Body: load_x + load_wqkv (pipeline) + qkv_matmul + apply_rope.
        assert_eq!(sched.body.len(), 4);
        // Three Stores (q, k, v) in the epilogue.
        assert_eq!(sched.epilogue.len(), 3);
    }

    #[test]
    fn qkv_rope_without_cache_uses_plain_store_tags() {
        let r_cache = qkv_rope_region(&QkvRopeParams {
            hidden_dim: 4096,
            head_dim: 128,
            num_q_heads: 32,
            num_kv_heads: 8,
            token_tile: 64,
            k_tile: 32,
            pipe: 3,
            writes_kv_cache: true,
        });
        let r_plain = qkv_rope_region(&QkvRopeParams {
            writes_kv_cache: false,
            ..QkvRopeParams {
                hidden_dim: 4096,
                head_dim: 128,
                num_q_heads: 32,
                num_kv_heads: 8,
                token_tile: 64,
                k_tile: 32,
                pipe: 3,
                writes_kv_cache: true,
            }
        });
        let cache_tags: Vec<&str> = r_cache.nodes.iter().map(|n| n.op.tag).collect();
        let plain_tags: Vec<&str> = r_plain.nodes.iter().map(|n| n.op.tag).collect();
        assert!(cache_tags.contains(&"store_k_cache"));
        assert!(cache_tags.contains(&"store_v_cache"));
        assert!(plain_tags.contains(&"store_k_row"));
        assert!(plain_tags.contains(&"store_v_row"));
    }

    #[test]
    fn residual_add_region_validates_and_schedules_straight_line() {
        let r = residual_add_region(&ResidualAddParams {
            hidden_dim: 4096,
            token_tile: 64,
        });
        ir::validate(&r).expect("residual_add region validates");
        assert_eq!(r.nodes.len(), 4);
        let sched = schedule_wavefront(&r, &sm89_fa2()).unwrap();
        assert!(sched.serial_axis.is_none());
        assert_eq!(sched.preamble.len(), 2);
        assert_eq!(sched.body.len(), 1);
        assert_eq!(sched.epilogue.len(), 1);
    }

    #[test]
    fn heterogeneous_megakernel_composes_attn_gemm_rmsnorm_add() {
        // Compose one of each template into a single Megakernel,
        // wire them with Barrier ControlEdges, and make sure the
        // emitter produces a single __global__ that references each
        // region's body. Acts as the end-to-end sanity check that the
        // new templates feed the megakernel emitter cleanly.
        use crate::emit_mega::emit_megakernel;
        use crate::ir::{ControlEdge, DepKind, Megakernel};

        let mut rnorm = rmsnorm_region(&RmsNormParams {
            hidden_dim: 4096,
            token_tile: 64,
        });
        rnorm.id = 0;
        let mut rgemm = gemm_region(&GemmParams {
            m_tile: 128,
            n_tile: 128,
            k_tile: 32,
            pipe: 3,
        });
        rgemm.id = 1;
        let mut rattn = attn_region(&AttnParams {
            window: Window::Infinite,
            head_dim: 128,
            tile_q: 128,
            tile_k: 64,
            num_head_groups: 8,
            pipe: 3,
        });
        rattn.id = 2;
        let mut radd = residual_add_region(&ResidualAddParams {
            hidden_dim: 4096,
            token_tile: 64,
        });
        radd.id = 3;

        let mk = Megakernel {
            regions: vec![rnorm, rgemm, rattn, radd],
            control: vec![
                ControlEdge {
                    src: 0,
                    dst: 1,
                    kind: DepKind::Barrier,
                },
                ControlEdge {
                    src: 1,
                    dst: 2,
                    kind: DepKind::Barrier,
                },
                ControlEdge {
                    src: 2,
                    dst: 3,
                    kind: DepKind::Barrier,
                },
            ],
        };

        let src = emit_megakernel(&mk, &sm90_fa2()).expect("emit succeeds");

        // Single kernel.
        assert_eq!(src.matches("__global__ void ").count(), 1);
        // All four region bodies present.
        assert!(src.contains("region 0 (rmsnorm)"));
        assert!(src.contains("region 1 (gemm)"));
        assert!(src.contains("region 2 (fa2_prefill)"));
        assert!(src.contains("region 3 (residual_add)"));
        // Three inter-region barriers between them.
        assert_eq!(
            src.matches("inter-region barrier").count(),
            3,
            "one barrier per control edge"
        );
        // Entry scalar union: rmsnorm + gemm + attn + add contribute
        // distinct names; they all show up in the kernel signature.
        assert!(src.contains("uint32_t num_token_tiles"));
        assert!(src.contains("uint32_t num_m_tiles"));
        assert!(src.contains("uint32_t num_q_tiles"));
    }
}
