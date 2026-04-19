// SPDX-License-Identifier: Apache-2.0
//! Address-expression emitter.
//!
//! Walks a `LoadAddr` (the `SmallVec<AddrTerm>` on each Load/Store
//! node) and renders a C++ offset expression — e.g.
//! `q_tile * 16384u + head_group * 128u` — that the data-movement
//! call sites add to the gmem base pointer:
//!
//! ```c
//! tma_load_2d<SMEM_Q_BYTES>(smem_q, Q_gmem + (q_tile * 16384u + head_group * 128u));
//! ```
//!
//! This replaces the pre-refactor call shape where emit_ops handed
//! the axes to the helper (`tma_load_2d(smem_q, Q_gmem, q_tile,
//! head_group)`) and the variadic trap body ignored them. With a
//! concrete offset the helper can do the actual per-thread cp.async
//! / TMA load without knowing the tensor layout.
//!
//! Pipelined loads: the serial axis is shifted by the step's
//! `iter_offset` at render time. A K-axis load with iter_offset=+3
//! renders `(kv_tile + 3) * tile_bytes + head_group * head_dim` — the
//! load reads from iteration `kv_tile + pipeline_depth`, stores into
//! the slot the consumer will drain at iteration `kv_tile +
//! pipeline_depth - pipeline_depth = kv_tile`.

use crate::ir::{AddrTerm, AxisId, LoadAddr, Region, StrideExpr};

/// Render the C++ offset expression for `addr`, in source-tensor
/// elements. The caller adds this to the gmem base pointer —
/// `gmem + {offset_expr}` — so any pointer-arithmetic scale factor
/// belongs in the element count, not here.
///
/// `iter_offset` shifts the *serial* axis only: pipelined loads at
/// iter_offset=+P read `serial_axis + P` iterations ahead. Parallel
/// axes always render with their raw name; `iter_offset` has no
/// effect on them.
///
/// An empty `addr.terms` renders as `"0u"` — the single-point load
/// case used for constants like rmsnorm's 1-D weight row.
pub fn render(
    region: &Region,
    addr: &LoadAddr,
    iter_offset: i32,
    serial_axis: Option<AxisId>,
) -> String {
    if addr.terms.is_empty() {
        return "0u".to_string();
    }
    let parts: Vec<String> = addr
        .terms
        .iter()
        .map(|t| render_term(region, t, iter_offset, serial_axis))
        .collect();
    parts.join(" + ")
}

fn render_term(
    region: &Region,
    term: &AddrTerm,
    iter_offset: i32,
    serial_axis: Option<AxisId>,
) -> String {
    match term {
        AddrTerm::RegionEntryConst(sid) => region.scalar(*sid).name.to_string(),
        AddrTerm::AxisStride { axis, stride } => {
            let a = axis_ref(region, *axis, iter_offset, serial_axis);
            format!("{} * {}", a, render_stride(region, stride))
        }
        AddrTerm::AxisModStride {
            axis,
            modulus,
            stride,
        } => {
            let a = axis_ref(region, *axis, iter_offset, serial_axis);
            format!("({} % {}u) * {}", a, modulus, render_stride(region, stride))
        }
        AddrTerm::AxisDivGather {
            axis,
            divisor,
            table,
            stride,
        } => {
            let a = axis_ref(region, *axis, iter_offset, serial_axis);
            // Gather table is a host-set indirection (e.g. block_table
            // for paged-KV decode). Emitted as a global lookup; the
            // kernel-param plumbing that makes it a real argument is
            // the item-6 (ambient scalar) work.
            format!(
                "{}[{} / {}u] * {}",
                table.source,
                a,
                divisor,
                render_stride(region, stride)
            )
        }
    }
}

fn axis_ref(
    region: &Region,
    axis: AxisId,
    iter_offset: i32,
    serial_axis: Option<AxisId>,
) -> String {
    let name = region.axis(axis).name;
    match (serial_axis, iter_offset) {
        (Some(sa), off) if sa == axis && off != 0 => format!("({} + {})", name, off),
        _ => name.to_string(),
    }
}

fn render_stride(region: &Region, s: &StrideExpr) -> String {
    match s {
        StrideExpr::Const(n) => format!("{}u", n),
        StrideExpr::RegionEntry(sid) => region.scalar(*sid).name.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        Axis, AxisId, Bound, Domain, Edge, FufOpRef, LoadAddr, Node, NodeId, Region, Role,
        ScalarBinding, ScalarId, SmemLookup,
    };
    use smallvec::smallvec;

    fn region_with_axes() -> Region {
        Region {
            id: 0,
            name: "test",
            domain: Domain {
                axes: vec![
                    Axis {
                        id: 0,
                        name: "m_tile",
                        bound: Bound::Const(8),
                    },
                    Axis {
                        id: 1,
                        name: "n_tile",
                        bound: Bound::Const(4),
                    },
                    Axis {
                        id: 2,
                        name: "k_tile",
                        bound: Bound::Const(16),
                    },
                ],
                predicates: vec![],
            },
            entry_scalars: vec![ScalarBinding {
                id: 0,
                name: "row_stride",
            }],
            nodes: vec![],
            edges: vec![],
            gmem_bindings: vec![],
            tile_consts: vec![],
        }
    }

    #[test]
    fn empty_address_is_zero() {
        let r = region_with_axes();
        let addr = LoadAddr { terms: smallvec![] };
        assert_eq!(render(&r, &addr, 0, None), "0u");
    }

    #[test]
    fn single_axis_stride_const() {
        let r = region_with_axes();
        let addr = LoadAddr {
            terms: smallvec![AddrTerm::AxisStride {
                axis: 0,
                stride: StrideExpr::Const(16384),
            }],
        };
        assert_eq!(render(&r, &addr, 0, None), "m_tile * 16384u");
    }

    #[test]
    fn two_axes_sum_separated_by_plus() {
        let r = region_with_axes();
        let addr = LoadAddr {
            terms: smallvec![
                AddrTerm::AxisStride {
                    axis: 0,
                    stride: StrideExpr::Const(16384),
                },
                AddrTerm::AxisStride {
                    axis: 1,
                    stride: StrideExpr::Const(128),
                },
            ],
        };
        assert_eq!(
            render(&r, &addr, 0, None),
            "m_tile * 16384u + n_tile * 128u"
        );
    }

    #[test]
    fn serial_axis_shifts_with_iter_offset() {
        let r = region_with_axes();
        let addr = LoadAddr {
            terms: smallvec![
                AddrTerm::AxisStride {
                    axis: 2, // k_tile
                    stride: StrideExpr::Const(16384),
                },
                AddrTerm::AxisStride {
                    axis: 1, // n_tile — parallel, stays unshifted
                    stride: StrideExpr::Const(128),
                },
            ],
        };
        // iter_offset=+3 on the serial axis (k_tile) — the pipeline
        // depth. n_tile is parallel and doesn't shift.
        assert_eq!(
            render(&r, &addr, 3, Some(2)),
            "(k_tile + 3) * 16384u + n_tile * 128u"
        );
    }

    #[test]
    fn iter_offset_does_not_shift_non_serial_axes() {
        let r = region_with_axes();
        let addr = LoadAddr {
            terms: smallvec![AddrTerm::AxisStride {
                axis: 0, // m_tile, parallel
                stride: StrideExpr::Const(16384),
            }],
        };
        // serial_axis = 2 (k_tile), but this term uses axis 0 —
        // shouldn't shift.
        assert_eq!(render(&r, &addr, 3, Some(2)), "m_tile * 16384u");
    }

    #[test]
    fn region_entry_stride_renders_as_scalar_name() {
        let r = region_with_axes();
        let addr = LoadAddr {
            terms: smallvec![AddrTerm::AxisStride {
                axis: 0,
                stride: StrideExpr::RegionEntry(0),
            }],
        };
        assert_eq!(render(&r, &addr, 0, None), "m_tile * row_stride");
    }

    #[test]
    fn axis_div_gather_renders_table_lookup() {
        let r = region_with_axes();
        let addr = LoadAddr {
            terms: smallvec![AddrTerm::AxisDivGather {
                axis: 2,
                divisor: 16,
                table: SmemLookup {
                    source: "block_table",
                },
                stride: StrideExpr::Const(16384),
            }],
        };
        assert_eq!(
            render(&r, &addr, 0, None),
            "block_table[k_tile / 16u] * 16384u"
        );
    }

    #[test]
    fn axis_div_gather_serial_axis_shifts_with_iter_offset() {
        let r = region_with_axes();
        let addr = LoadAddr {
            terms: smallvec![AddrTerm::AxisDivGather {
                axis: 2,
                divisor: 16,
                table: SmemLookup {
                    source: "block_table",
                },
                stride: StrideExpr::Const(16384),
            }],
        };
        // The shifted serial axis flows *through* the gather's div:
        // `block_table[(k_tile + 3) / 16u] * 16384u`.
        assert_eq!(
            render(&r, &addr, 3, Some(2)),
            "block_table[(k_tile + 3) / 16u] * 16384u"
        );
    }

    #[test]
    fn axis_mod_stride_renders_modulus_expression() {
        let r = region_with_axes();
        let addr = LoadAddr {
            terms: smallvec![AddrTerm::AxisModStride {
                axis: 2,
                modulus: 4,
                stride: StrideExpr::Const(16384),
            }],
        };
        assert_eq!(render(&r, &addr, 0, None), "(k_tile % 4u) * 16384u");
    }

    // `_` unused warning silence on the explicit AxisId type import.
    const _: AxisId = 0;
    const _: NodeId = 0;
    const _: ScalarId = 0;

    #[test]
    fn unused_imports_silencer() {
        // This keeps the explicit type imports live for the test
        // module so clippy doesn't flag them as unused when iterated
        // manually; the real tests above use them indirectly via the
        // `Region` / `LoadAddr` / `AddrTerm` constructors.
        let _: Option<Edge> = None;
        let _: Option<Node> = None;
        let _ = Role::Load;
        let _ = FufOpRef { tag: "noop" };
    }
}
