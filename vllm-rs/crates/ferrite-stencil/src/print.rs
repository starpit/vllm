// SPDX-License-Identifier: Apache-2.0
//! Pretty-print a Region under an ArchMap. Purpose: make the
//! round-trip from template → mapping human-readable so tests can
//! snapshot-diff it before any codegen exists.

use std::fmt::Write;

use crate::arch::ArchMap;
use crate::ir::{AffineOffset, Bound, CmpOp, Region, Role};

pub fn print_region(region: &Region, arch: &ArchMap) -> String {
    let mut out = String::new();
    writeln!(out, "region {} (arch={})", region.name, arch.name).unwrap();

    writeln!(out, "  domain:").unwrap();
    for a in &region.domain.axes {
        writeln!(out, "    {} : {}", a.name, fmt_bound(region, &a.bound)).unwrap();
    }
    if !region.domain.predicates.is_empty() {
        writeln!(out, "  predicates:").unwrap();
        for p in &region.domain.predicates {
            let lhs = p
                .coeffs
                .iter()
                .map(|(a, c)| format!("{}·{}", c, region.axis(*a).name))
                .collect::<Vec<_>>()
                .join(" + ");
            writeln!(
                out,
                "    {} {} {}",
                lhs,
                fmt_cmp(p.op),
                fmt_offset(region, &p.offset)
            )
            .unwrap();
        }
    }

    writeln!(out, "  entry_scalars: {}", region.entry_scalars.len()).unwrap();
    for s in &region.entry_scalars {
        writeln!(out, "    {} (id={})", s.name, s.id).unwrap();
    }

    writeln!(out, "  nodes:").unwrap();
    for n in &region.nodes {
        let hw = (arch.role)(n.role, region);
        writeln!(
            out,
            "    n{} {:?}({}) -> {}",
            n.id,
            n.role,
            n.op.tag,
            fmt_hw(hw)
        )
        .unwrap();
    }

    writeln!(out, "  edges:").unwrap();
    for e in &region.edges {
        let prim = (arch.barrier)(e.kind);
        let vec = if e.vector.0.is_empty() {
            "0".to_string()
        } else {
            e.vector
                .0
                .iter()
                .map(|(a, d)| format!("{}{:+}", region.axis(*a).name, d))
                .collect::<Vec<_>>()
                .join(",")
        };
        writeln!(
            out,
            "    n{} -> n{} [{:?} {:?} vec={}]",
            e.src, e.dst, e.kind, prim, vec
        )
        .unwrap();
    }

    writeln!(out, "  pipe_depth: {}", (arch.pipe_depth)(region)).unwrap();
    out
}

fn fmt_bound(r: &Region, b: &Bound) -> String {
    match b {
        Bound::Const(n) => format!("0..{}", n),
        Bound::RegionEntryScalar(s) => format!("0..{}", r.scalar(*s).name),
        Bound::IndexedScalar(s, a) => {
            format!("0..{}[{}]", r.scalar(*s).name, r.axis(*a).name)
        }
        Bound::Unbounded => "0..∞".to_string(),
    }
}

fn fmt_offset(r: &Region, o: &AffineOffset) -> String {
    match o {
        AffineOffset::Const(n) => n.to_string(),
        AffineOffset::RegionEntry(s) => r.scalar(*s).name.to_string(),
        AffineOffset::Indexed(s, a) => {
            format!("{}[{}]", r.scalar(*s).name, r.axis(*a).name)
        }
    }
}

fn fmt_cmp(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Le => "≤",
        CmpOp::Lt => "<",
        CmpOp::Ge => "≥",
        CmpOp::Gt => ">",
        CmpOp::Eq => "=",
    }
}

fn fmt_hw(u: crate::arch::HardwareUnit) -> String {
    use crate::arch::HardwareUnit::*;
    match u {
        Warpgroup { role_name, num_wg } => format!("wg[{}×{}]", role_name, num_wg),
        AllWarps => "all_warps".to_string(),
    }
}

#[allow(dead_code)]
fn _role_is_total(_: Role) {}
