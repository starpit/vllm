// SPDX-License-Identifier: Apache-2.0
//! Region templates. For v1 there is one: `Attn(W)`. It covers
//! FA2 prefill (`W = ∞`), Gemma3 local attention (`W = finite`),
//! and paged-KV decode (same template, different load-address
//! instantiation and domain axes — see `attn_region_decode`).

use smallvec::smallvec;

use crate::ir::{
    AddrTerm, AffineOffset, Axis, AxisId, Bound, CmpOp, DepKind, DepVector, Domain, Edge, FufOpRef,
    LoadAddr, Node, NodeId, Predicate, Region, Role, ScalarBinding, ScalarId, StrideExpr,
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
    }
}
