// SPDX-License-Identifier: Apache-2.0
//! Arch-neutral analyses over a Region. These are the primitives
//! the wavefront scheduler + emitter consume. None of this is a
//! full schedule yet — it's the fact base the schedule will rest
//! on, exposed and tested so the scheduler can't quietly disagree
//! with the IR about what axes mean.
//!
//! - `classify_axes` splits a region's iteration space into axes
//!   iterated serially (any edge carries a nonzero dep on them) vs
//!   axes distributed in parallel (no edge touches them).
//! - `region_pipeline_depth` reads the maximum |component| any
//!   Pipeline edge declares on any axis. The arch mapping is
//!   responsible for ensuring its chosen `pipe_depth` does not
//!   exceed this — otherwise the schedule would need dep vectors
//!   longer than the template emits.
//! - `topo_order_within_iter` orders nodes by same-iteration Raw
//!   edges (zero dep vector). Cross-iteration edges are satisfied
//!   by barriers, not by intra-iter ordering.

use crate::ir::{AxisId, DepKind, NodeId, Region};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisKind {
    /// No edge carries a nonzero dep on this axis → the axis is
    /// distributed across CTAs; values are independent.
    Parallel,
    /// Some edge carries a nonzero dep on this axis → the axis is
    /// iterated serially within a CTA; order matters.
    Serial,
}

pub fn classify_axes(region: &Region) -> Vec<(AxisId, AxisKind)> {
    region
        .domain
        .axes
        .iter()
        .map(|a| {
            let has_dep = region
                .edges
                .iter()
                .any(|e| e.vector.0.iter().any(|(ax, d)| *ax == a.id && *d != 0));
            let kind = if has_dep {
                AxisKind::Serial
            } else {
                AxisKind::Parallel
            };
            (a.id, kind)
        })
        .collect()
}

pub fn region_pipeline_depth(region: &Region) -> u32 {
    region
        .edges
        .iter()
        .filter(|e| e.kind == DepKind::Pipeline)
        .flat_map(|e| e.vector.0.iter().map(|(_, d)| d.unsigned_abs()))
        .max()
        .unwrap_or(0)
}

/// Kahn's algorithm on the subgraph of Raw edges with zero dep
/// vector. Returns NodeIds in one valid topo order. Panics if a
/// cycle exists among same-iter Raw edges — that would indicate an
/// IR construction bug (cross-iter cycles are fine; they resolve
/// through the serial axis).
pub fn topo_order_within_iter(region: &Region) -> Vec<NodeId> {
    let n = region.nodes.len();
    let mut in_deg = vec![0u32; n];
    let mut adj: Vec<Vec<NodeId>> = vec![Vec::new(); n];

    for e in &region.edges {
        if e.kind != DepKind::Raw {
            continue;
        }
        let is_same_iter = e.vector.0.iter().all(|(_, d)| *d == 0);
        if !is_same_iter {
            continue;
        }
        adj[e.src as usize].push(e.dst);
        in_deg[e.dst as usize] += 1;
    }

    let mut ready: Vec<NodeId> = (0..n as NodeId)
        .filter(|i| in_deg[*i as usize] == 0)
        .collect();
    // Stable order by NodeId gives deterministic output.
    ready.sort_unstable();

    let mut out = Vec::with_capacity(n);
    while let Some(i) = ready.pop() {
        out.push(i);
        let mut fresh = Vec::new();
        for &j in &adj[i as usize] {
            in_deg[j as usize] -= 1;
            if in_deg[j as usize] == 0 {
                fresh.push(j);
            }
        }
        fresh.sort_unstable();
        // Push so that lower ids come out later (pop from end).
        for j in fresh.into_iter().rev() {
            ready.push(j);
        }
    }

    assert_eq!(
        out.len(),
        n,
        "cycle in same-iter Raw subgraph — IR construction bug"
    );
    out
}
