// SPDX-License-Identifier: Apache-2.0
//! Megakernel emitter.
//!
//! Takes a `Megakernel` (a DAG of `Region`s connected by
//! `ControlEdge`s) and an `ArchMap`, produces ONE persistent
//! `__global__` that composes every region in topological order of
//! `Megakernel.control`. Warpgroup partition (SM90+) lives outside
//! per-region code — the whole kernel sees one `wg = threadIdx.x /
//! 128u` dispatch at entry. Inter-region synchronization goes through
//! `ArchMap::barrier` — `Barrier` dep kind lowers to `g.Bar` on SM90
//! and SM89 alike (per design §3).
//!
//! This is the real emitter target: a single `__global__` per model
//! forward, 1 CTA per SM, compile-time instruction sequence (no
//! runtime VM — the region order is baked into the emission by topo
//! sort). Per-region codegen inside the shell reuses the `emit_ops`
//! expansion table so intrinsic-level vocabulary stays in one place.
//!
//! This commit lands the shell; region templates for non-attention
//! ops and real `Megakernel.control` population from the FUF land in
//! follow-up commits.

use std::collections::{BTreeMap, VecDeque};
use std::fmt::Write;

use crate::arch::{ArchMap, BarrierPrim, HardwareUnit};
use crate::emit_ops::{
    ExpandCtx, GmemAccess, LocalKind, expand as expand_op, gmem_refs, local_refs,
};
use crate::ir::{Axis, Bound, DepKind, Megakernel, Region, RegionId, Role};
use crate::wavefront::{Schedule, ScheduleError, Step, schedule_wavefront};

#[derive(Debug)]
pub enum EmitError {
    RegionSchedule {
        region: RegionId,
        error: ScheduleError,
    },
    ControlCycle,
    UnknownRegionInControl(RegionId),
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RegionSchedule { region, error } => {
                write!(f, "scheduling region {}: {}", region, error)
            }
            Self::ControlCycle => write!(f, "Megakernel.control has a cycle"),
            Self::UnknownRegionInControl(r) => {
                write!(f, "control edge references unknown region {}", r)
            }
        }
    }
}

impl std::error::Error for EmitError {}

/// Emit a persistent-megakernel `__global__` composing every region in
/// `mega` in topo order of `mega.control`.
///
/// Structure:
///   - signature (union of region entry scalars, pointer plumbing TBD)
///   - warpgroup dispatch (from `arch.role`)
///   - per-region preamble/body/epilogue, inlined, in topo order
///   - inter-region barriers lowered through `arch.barrier`
pub fn emit_megakernel(mega: &Megakernel, arch: &ArchMap) -> Result<String, EmitError> {
    let schedules = mega
        .regions
        .iter()
        .map(|r| {
            schedule_wavefront(r, arch).map_err(|e| EmitError::RegionSchedule {
                region: r.id,
                error: e,
            })
        })
        .collect::<Result<Vec<Schedule>, _>>()?;

    let order = topo_regions(mega)?;
    let params = collect_params(mega);

    let mut out = String::new();
    write_header(&mut out, mega, arch);
    write_signature(&mut out, &params);
    write_wg_dispatch(&mut out, arch);

    for (i, rid) in order.iter().enumerate() {
        if i > 0 {
            write_interregion_barrier(&mut out, mega, order[i - 1], *rid, arch);
        }
        let idx = mega
            .regions
            .iter()
            .position(|r| r.id == *rid)
            .expect("topo_regions returned unknown id");
        write_region(&mut out, &mega.regions[idx], &schedules[idx], arch);
    }

    writeln!(out, "}}  // end mega_kernel").unwrap();
    write_launcher(&mut out, &params);
    Ok(out)
}

/// Kernel param summary: both signature emission and the host
/// launcher consume this so they can't drift out of sync.
struct ParamSet {
    /// Deduped scalars. `indexed=true` means some region uses the
    /// scalar via `Bound::IndexedScalar` — the param must be a
    /// pointer type (one entry per index), not a plain `uint32_t`.
    scalars: Vec<(&'static str, bool)>,
    /// Deduped gmem tensor params: (name, read-access-kind). Sorted by
    /// name for stable output across compiler runs.
    gmem: Vec<(&'static str, GmemAccess)>,
}

fn collect_params(mega: &Megakernel) -> ParamSet {
    // First pass: identify which scalar names are consumed via
    // `Bound::IndexedScalar` in any region. Those need pointer type.
    let mut indexed: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();
    for r in &mega.regions {
        for a in &r.domain.axes {
            if let Bound::IndexedScalar(sid, _) = a.bound
                && let Some(name) = r.entry_scalars.iter().find(|s| s.id == sid).map(|s| s.name)
            {
                indexed.insert(name);
            }
        }
        // Also scan address terms / predicates — conservative for now.
        let _ = |_: &Axis| (); // silence unused import warning path
    }

    let mut scalar_seen: std::collections::BTreeSet<&'static str> =
        std::collections::BTreeSet::new();
    let mut scalars: Vec<(&'static str, bool)> = Vec::new();
    for r in &mega.regions {
        for s in &r.entry_scalars {
            if scalar_seen.insert(s.name) {
                scalars.push((s.name, indexed.contains(s.name)));
            }
        }
    }

    // Resolve each (canonical) gmem ref through the region's bindings
    // so per-layer identities (populated by item 4b) become distinct
    // params. Pre-4b regions carry empty bindings, and the canonical
    // name flows through unchanged — signature matches legacy output.
    let mut gmem_access: BTreeMap<&'static str, GmemAccess> = BTreeMap::new();
    for r in &mega.regions {
        for n in &r.nodes {
            for &(name, access) in gmem_refs(n.op.tag) {
                let identity = resolve_gmem(&r.gmem_bindings, name);
                gmem_access
                    .entry(identity)
                    .and_modify(|e| *e = e.union(access))
                    .or_insert(access);
            }
        }
    }
    let gmem: Vec<(&'static str, GmemAccess)> = gmem_access.into_iter().collect();

    ParamSet { scalars, gmem }
}

/// Look up `canonical` in `bindings`; return the identity when
/// bound, else `canonical` unchanged. Mirror of
/// `ExpandCtx::gmem` — the signature emitter needs the same
/// substitution so kernel params match what the expansion bodies
/// reference.
fn resolve_gmem(
    bindings: &[(&'static str, &'static str)],
    canonical: &'static str,
) -> &'static str {
    for (c, id) in bindings {
        if *c == canonical {
            return id;
        }
    }
    canonical
}

/// Kahn's algorithm over `Megakernel.control`. When `control` is
/// empty, regions fire in declaration order. A cycle → `ControlCycle`.
fn topo_regions(mega: &Megakernel) -> Result<Vec<RegionId>, EmitError> {
    let mut indeg: BTreeMap<RegionId, usize> =
        mega.regions.iter().map(|r| (r.id, 0usize)).collect();
    let mut out_edges: BTreeMap<RegionId, Vec<RegionId>> = BTreeMap::new();
    for ce in &mega.control {
        if !indeg.contains_key(&ce.src) {
            return Err(EmitError::UnknownRegionInControl(ce.src));
        }
        if !indeg.contains_key(&ce.dst) {
            return Err(EmitError::UnknownRegionInControl(ce.dst));
        }
        *indeg.get_mut(&ce.dst).unwrap() += 1;
        out_edges.entry(ce.src).or_default().push(ce.dst);
    }
    // Seed the queue in region-declaration order so empty-control
    // output matches the user's written sequence of regions.
    let mut queue: VecDeque<RegionId> = mega
        .regions
        .iter()
        .filter(|r| indeg[&r.id] == 0)
        .map(|r| r.id)
        .collect();
    let mut order: Vec<RegionId> = Vec::with_capacity(mega.regions.len());
    while let Some(rid) = queue.pop_front() {
        order.push(rid);
        if let Some(dsts) = out_edges.get(&rid) {
            for &dst in dsts {
                let d = indeg.get_mut(&dst).unwrap();
                *d -= 1;
                if *d == 0 {
                    queue.push_back(dst);
                }
            }
        }
    }
    if order.len() != mega.regions.len() {
        return Err(EmitError::ControlCycle);
    }
    Ok(order)
}

fn write_header(out: &mut String, mega: &Megakernel, arch: &ArchMap) {
    writeln!(
        out,
        "// persistent megakernel — arch={} regions={}",
        arch.name,
        mega.regions.len()
    )
    .unwrap();
    writeln!(
        out,
        "// One __global__, one launch, 1 CTA per SM. Regions run in topo order"
    )
    .unwrap();
    writeln!(
        out,
        "// of Megakernel.control; inter-region edges lower to ArchMap::barrier."
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#include <cuda_runtime.h>").unwrap();
    writeln!(out, "#include <cuda_bf16.h>").unwrap();
    writeln!(out, "#include <cstdint>").unwrap();
    writeln!(
        out,
        "// The helpers + ambient state below resolve through the stencil prelude:"
    )
    .unwrap();
    writeln!(out, "#include \"ferrite_stencil_prelude.cuh\"").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "typedef __nv_bfloat16 bf16;").unwrap();
    // Global g.Bar counter used by inter-region Barrier edges. Host
    // zeros it once per launch. `gbar_sync(&gbar_counter)` arrives +
    // spin-waits for the cross-CTA fence (Megakernel-style — see
    // STENCIL_IR_DESIGN.md §5).
    writeln!(out, "__device__ uint32_t gbar_counter;").unwrap();
}

fn write_signature(out: &mut String, params: &ParamSet) {
    writeln!(out, "__global__ void mega_kernel(").unwrap();
    let total = params.scalars.len() + params.gmem.len();
    let comma = |i: usize| if i + 1 < total { "," } else { "" };
    let mut idx = 0usize;
    for (name, indexed) in &params.scalars {
        let ty = if *indexed {
            "const uint32_t* __restrict__"
        } else {
            "uint32_t"
        };
        writeln!(out, "    {} {}{}", ty, name, comma(idx)).unwrap();
        idx += 1;
    }
    for (name, access) in &params.gmem {
        let qual = gmem_qualifier(*access);
        writeln!(out, "    {} {}{}", qual, name, comma(idx)).unwrap();
        idx += 1;
    }
    writeln!(out, ") {{").unwrap();
}

fn gmem_qualifier(access: GmemAccess) -> &'static str {
    match access {
        GmemAccess::Read => "const bf16* __restrict__",
        GmemAccess::Write | GmemAccess::ReadWrite => "bf16* __restrict__",
    }
}

/// Host-side launcher — zeroes `gbar_counter`, queries SM count, and
/// kicks off the megakernel with `grid = #SMs`, `block = 640` (20
/// warps = persistent-CTA layout per design §5). Signature mirrors
/// the kernel's: same scalar + gmem pointer list. Wrapping in
/// `extern "C"` so Rust FFI can call it without name mangling.
fn write_launcher(out: &mut String, params: &ParamSet) {
    writeln!(out).unwrap();
    writeln!(
        out,
        "// ── Host launcher ──────────────────────────────────────"
    )
    .unwrap();
    writeln!(out, "extern \"C\" cudaError_t launch_mega_kernel(").unwrap();
    writeln!(out, "    cudaStream_t stream,").unwrap();
    let total = params.scalars.len() + params.gmem.len();
    let comma = |i: usize| if i + 1 < total { "," } else { "" };
    let mut idx = 0usize;
    for (name, indexed) in &params.scalars {
        let ty = if *indexed {
            "const uint32_t* __restrict__"
        } else {
            "uint32_t"
        };
        writeln!(out, "    {} {}{}", ty, name, comma(idx)).unwrap();
        idx += 1;
    }
    for (name, access) in &params.gmem {
        let qual = gmem_qualifier(*access);
        writeln!(out, "    {} {}{}", qual, name, comma(idx)).unwrap();
        idx += 1;
    }
    writeln!(out, ") {{").unwrap();
    writeln!(
        out,
        "    // Zero g.Bar once per launch; regions arrive + spin-wait."
    )
    .unwrap();
    writeln!(out, "    uint32_t zero = 0;").unwrap();
    writeln!(
        out,
        "    cudaError_t err = cudaMemcpyToSymbolAsync(gbar_counter, &zero,"
    )
    .unwrap();
    writeln!(
        out,
        "        sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream);"
    )
    .unwrap();
    writeln!(out, "    if (err != cudaSuccess) return err;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // 1 CTA per SM × 20 warps × 32 threads = 640.").unwrap();
    writeln!(out, "    int sm_count = 0;").unwrap();
    writeln!(
        out,
        "    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);"
    )
    .unwrap();
    writeln!(out, "    dim3 grid((unsigned)sm_count);").unwrap();
    writeln!(out, "    dim3 block(640u);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    mega_kernel<<<grid, block, 0, stream>>>(").unwrap();
    let mut idx = 0usize;
    for (name, _) in &params.scalars {
        writeln!(out, "        {}{}", name, comma(idx)).unwrap();
        idx += 1;
    }
    for (name, _) in &params.gmem {
        writeln!(out, "        {}{}", name, comma(idx)).unwrap();
        idx += 1;
    }
    writeln!(out, "    );").unwrap();
    writeln!(out, "    return cudaGetLastError();").unwrap();
    writeln!(out, "}}").unwrap();
}

fn write_wg_dispatch(out: &mut String, arch: &ArchMap) {
    // Probe the ArchMap for each role's hardware unit. For the arches
    // we have today the mapping is region-independent; we pass a
    // throwaway empty region to satisfy the `fn(Role, &Region)` shape.
    let dummy = Region {
        id: 0,
        name: "_dispatch_probe",
        domain: crate::ir::Domain {
            axes: vec![],
            predicates: vec![],
        },
        entry_scalars: vec![],
        nodes: vec![],
        edges: vec![],
        gmem_bindings: Vec::new(),
        tile_consts: Vec::new(),
    };
    let units = [
        (Role::Load, (arch.role)(Role::Load, &dummy)),
        (Role::Compute, (arch.role)(Role::Compute, &dummy)),
        (Role::Store, (arch.role)(Role::Store, &dummy)),
    ];
    let any_wg = units
        .iter()
        .any(|(_, u)| matches!(u, HardwareUnit::Warpgroup { .. }));
    if !any_wg {
        writeln!(
            out,
            "  // role mapping: AllWarps (no warpgroup specialization)"
        )
        .unwrap();
        return;
    }
    writeln!(out, "  // role → warpgroup mapping:").unwrap();
    for (role, unit) in &units {
        match unit {
            HardwareUnit::Warpgroup { role_name, num_wg } => {
                writeln!(
                    out,
                    "  //   {:?} → {} ({} warpgroups)",
                    role, role_name, num_wg
                )
                .unwrap();
            }
            HardwareUnit::AllWarps => {
                writeln!(out, "  //   {:?} → AllWarps", role).unwrap();
            }
        }
    }
    writeln!(out, "  uint32_t wg = threadIdx.x / 128u;").unwrap();
}

fn write_interregion_barrier(
    out: &mut String,
    mega: &Megakernel,
    src: RegionId,
    dst: RegionId,
    arch: &ArchMap,
) {
    // Explicit edge wins; absence defaults to `Barrier` so topo-linked
    // regions never run without a fence between them.
    let ce = mega.control.iter().find(|e| e.src == src && e.dst == dst);
    let kind = ce.map(|e| e.kind).unwrap_or(DepKind::Barrier);
    let prim = (arch.barrier)(kind);
    writeln!(
        out,
        "  // ─── inter-region barrier: region {} → region {} ({:?}) ───",
        src, dst, kind
    )
    .unwrap();
    writeln!(out, "  {}", barrier_stmt(&prim)).unwrap();
}

fn barrier_stmt(p: &BarrierPrim) -> String {
    match p {
        BarrierPrim::Mbarrier => "mbarrier_wait();".to_string(),
        BarrierPrim::NamedSem { name, depth } => {
            format!("sem_wait(\"{}\", depth={});", name, depth)
        }
        BarrierPrim::CpAsyncGroup { depth } => {
            format!("cp_async_wait_group({});", depth.saturating_sub(1))
        }
        BarrierPrim::Gbar => "gbar_sync(&gbar_counter);".to_string(),
        BarrierPrim::Cluster => "cluster_sync();".to_string(),
        BarrierPrim::HostRedispatch => "return;  // host redispatch".to_string(),
    }
}

fn write_region(out: &mut String, region: &Region, sched: &Schedule, arch: &ArchMap) {
    writeln!(
        out,
        "  // ═══ region {} ({}) pipeline_depth={} ═══",
        region.id, region.name, sched.pipeline_depth
    )
    .unwrap();

    // Open a region-scoped block so region-local declarations (smem
    // buffers, register fragments, mbarriers, FA2's online-softmax
    // scalars) don't stomp on sibling regions. Matches
    // STENCIL_IR_STATUS.md item 2: prior revisions declared everything
    // file-scope via the prelude, so two GEMM regions would share one
    // `C_frag` and one `smem_a` / `smem_b`.
    let mut indent = String::from("  ");
    writeln!(out, "{}{{", indent).unwrap();
    indent.push_str("  ");
    // Region-scope compile-time tile sizes — emitted before the smem
    // decls reference them (`__shared__ bf16 smem_q[SMEM_Q_BYTES / 2];`).
    // Same constants feed `cp_async_128<BYTES>` / `tma_load_2d<BYTES>` /
    // `stg_128<BYTES>` template args at the data-movement call sites.
    write_region_tile_consts(out, &indent, &region.tile_consts);
    let locals = collect_locals(region);
    write_region_locals(out, &indent, &locals);

    // Resolve this region's axis names once — expansions thread them
    // through ExpandCtx (item 3) so hard-coded `q_tile` / `kv_tile`
    // don't drop out into file-scope placeholders for templates that
    // use different axis names (paged decode uses `b` for the batch
    // axis, not `q_tile`).
    let parallel_axis_names: Vec<&'static str> = sched
        .parallel_axes
        .iter()
        .map(|id| region.axis(*id).name)
        .collect();
    let serial_axis_name: Option<&'static str> = sched.serial_axis.map(|id| region.axis(id).name);

    // Parallel axes become for-loops *inside* the persistent CTA: the
    // host-side per-SM scheduler assigns which (q_tile, head_group) this
    // SM owns, but from the emitter's POV the kernel iterates its full
    // domain. Cross-CTA distribution is a wrapper concern, not this
    // commit's — what matters here is that every domain point runs.
    for ax in &sched.parallel_axes {
        let axis = region.axis(*ax);
        writeln!(
            out,
            "{}for (uint32_t {} = 0; {} < {}; ++{}) {{",
            indent,
            axis.name,
            axis.name,
            fmt_bound(region, &axis.bound),
            axis.name
        )
        .unwrap();
        indent.push_str("  ");
    }

    writeln!(out, "{}// preamble", indent).unwrap();
    for step in &sched.preamble {
        write_step(
            out,
            region,
            arch,
            step,
            &indent,
            sched.pipeline_depth,
            &parallel_axis_names,
            serial_axis_name,
            sched.serial_axis,
        );
    }

    if let Some(serial) = sched.serial_axis {
        let ax = region.axis(serial);
        writeln!(
            out,
            "{}for (uint32_t {} = 0; {} < {}; ++{}) {{",
            indent,
            ax.name,
            ax.name,
            fmt_bound(region, &ax.bound),
            ax.name
        )
        .unwrap();
        let mut body_indent = indent.clone();
        body_indent.push_str("  ");
        for step in &sched.body {
            write_step(
                out,
                region,
                arch,
                step,
                &body_indent,
                sched.pipeline_depth,
                &parallel_axis_names,
                serial_axis_name,
                sched.serial_axis,
            );
        }
        writeln!(out, "{}}}  // end {}", indent, ax.name).unwrap();
    } else {
        for step in &sched.body {
            write_step(
                out,
                region,
                arch,
                step,
                &indent,
                sched.pipeline_depth,
                &parallel_axis_names,
                serial_axis_name,
                sched.serial_axis,
            );
        }
    }

    writeln!(out, "{}// epilogue", indent).unwrap();
    for step in &sched.epilogue {
        write_step(
            out,
            region,
            arch,
            step,
            &indent,
            sched.pipeline_depth,
            &parallel_axis_names,
            serial_axis_name,
            sched.serial_axis,
        );
    }

    // Close parallel-axis loops in reverse order.
    for ax in sched.parallel_axes.iter().rev() {
        let axis = region.axis(*ax);
        indent.truncate(indent.len() - 2);
        writeln!(out, "{}}}  // end {}", indent, axis.name).unwrap();
    }

    // Close the region scope block.
    indent.truncate(indent.len() - 2);
    writeln!(out, "{}}}  // end region {}", indent, region.id).unwrap();
}

/// Walk the region's nodes, union their `local_refs` by name, and
/// return a deterministically-ordered list for emission. `SmemPlain`
/// widens to `SmemRing` (and the same for mbarriers) when both forms
/// appear under the same identifier in one region — the ring is a
/// superset, plain loads decay into slot 0.
fn collect_locals(region: &Region) -> Vec<(&'static str, LocalKind)> {
    let mut map: BTreeMap<&'static str, LocalKind> = BTreeMap::new();
    for n in &region.nodes {
        for lr in local_refs(n.op.tag) {
            map.entry(lr.name)
                .and_modify(|e| *e = e.widen(lr.kind))
                .or_insert(lr.kind);
        }
    }
    // Group by kind (smem → frag → mbarrier → float) for readable
    // emission, then by name for determinism across compiler runs.
    let mut v: Vec<(&'static str, LocalKind)> = map.into_iter().collect();
    v.sort_by_key(|(name, kind)| (local_kind_order(*kind), *name));
    v
}

fn local_kind_order(k: LocalKind) -> u8 {
    match k {
        LocalKind::SmemPlain | LocalKind::SmemRing => 0,
        LocalKind::Frag | LocalKind::FragQkv => 1,
        LocalKind::MbarrierPlain | LocalKind::MbarrierRing => 2,
        LocalKind::FloatInit(_) => 3,
    }
}

fn write_region_locals(out: &mut String, indent: &str, locals: &[(&'static str, LocalKind)]) {
    if locals.is_empty() {
        return;
    }
    writeln!(out, "{}// region-local declarations", indent).unwrap();
    for (name, kind) in locals {
        writeln!(out, "{}{}", indent, kind.decl(name)).unwrap();
    }
}

/// Emit `constexpr uint32_t {SYMBOL} = {bytes}u;` lines for each
/// `(local_name, bytes)` in `tile_consts`. Symbols match
/// `emit_ops::bytes_symbol(local_name)`. Keeps smem decls and
/// data-movement call sites referring to the same compile-time
/// constant, so a future change to a tile size only needs to update
/// the template that populated `tile_consts`.
fn write_region_tile_consts(out: &mut String, indent: &str, tile_consts: &[(&'static str, u32)]) {
    if tile_consts.is_empty() {
        return;
    }
    writeln!(out, "{}// region-local tile byte counts", indent).unwrap();
    for (name, bytes) in tile_consts {
        writeln!(
            out,
            "{}constexpr uint32_t {} = {}u;",
            indent,
            crate::emit_ops::bytes_symbol(name),
            bytes,
        )
        .unwrap();
    }
}

#[allow(clippy::too_many_arguments)]
fn write_step(
    out: &mut String,
    region: &Region,
    arch: &ArchMap,
    step: &Step,
    indent: &str,
    pipeline_depth: u32,
    parallel_axes: &[&'static str],
    serial_axis: Option<&'static str>,
    serial_axis_id: Option<crate::ir::AxisId>,
) {
    let node = region.node(step.node);
    for bp in &step.barriers_before {
        writeln!(out, "{}{}", indent, barrier_stmt(bp)).unwrap();
    }
    // Pre-render the tile offset expression for Load/Store nodes so
    // data-movement expansions can emit `gmem + ({node_addr})`
    // instead of the pre-refactor `gmem, row, col` form. Compute
    // nodes (no `addr`) pass `None` through — expansions that don't
    // touch gmem never look at `ctx.node_addr`.
    let node_addr = node
        .addr
        .as_ref()
        .map(|addr| crate::emit_addr::render(region, addr, step.iter_offset, serial_axis_id));
    let ctx = ExpandCtx {
        arch_name: arch.name,
        iter_offset: step.iter_offset,
        pipeline_depth,
        parallel_axes,
        serial_axis,
        gmem_bindings: &region.gmem_bindings,
        node_addr: node_addr.as_deref(),
    };
    if let Some(body) = expand_op(node.op.tag, &ctx) {
        for line in body.lines() {
            writeln!(out, "{}{}", indent, line).unwrap();
        }
        return;
    }
    if step.iter_offset != 0 {
        writeln!(
            out,
            "{}{}(/* iter + {} */);  // {:?}",
            indent, node.op.tag, step.iter_offset, node.role,
        )
        .unwrap();
    } else {
        writeln!(out, "{}{}();  // {:?}", indent, node.op.tag, node.role).unwrap();
    }
}

fn fmt_bound(region: &Region, b: &Bound) -> String {
    match b {
        Bound::Const(n) => n.to_string(),
        Bound::RegionEntryScalar(s) => region.scalar(*s).name.to_string(),
        Bound::IndexedScalar(s, a) => {
            format!("{}[{}]", region.scalar(*s).name, region.axis(*a).name)
        }
        Bound::Unbounded => "/* unbounded */ 0".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{sm89_fa2, sm90_fa2};
    use crate::ir::ControlEdge;
    use crate::template::{AttnParams, Window, attn_region};

    fn two_region_mega() -> Megakernel {
        // Two attention regions (Infinite + Finite windows) composed
        // with a single Barrier ControlEdge. Synthetic content — the
        // point is to exercise megakernel composition, not model a
        // real forward yet.
        let mut r0 = attn_region(&AttnParams {
            window: Window::Infinite,
            head_dim: 128,
            tile_q: 128,
            tile_k: 64,
            num_head_groups: 8,
            pipe: 3,
        });
        r0.id = 0;
        let mut r1 = attn_region(&AttnParams {
            window: Window::Finite(4096),
            head_dim: 128,
            tile_q: 128,
            tile_k: 64,
            num_head_groups: 8,
            pipe: 3,
        });
        r1.id = 1;
        Megakernel {
            regions: vec![r0, r1],
            control: vec![ControlEdge {
                src: 0,
                dst: 1,
                kind: DepKind::Barrier,
            }],
        }
    }

    #[test]
    fn one_global_with_warpgroup_dispatch_and_composition_on_sm90() {
        let mk = two_region_mega();
        let src = emit_megakernel(&mk, &sm90_fa2()).expect("emit succeeds");

        // Exactly one __global__ function: the whole megakernel.
        assert_eq!(src.matches("__global__ void ").count(), 1);
        assert!(src.contains("void mega_kernel("));

        // Warpgroup partition at kernel entry, not per-region.
        assert!(src.contains("uint32_t wg = threadIdx.x / 128u;"));
        assert!(src.contains("Load → loader (1 warpgroups)"));
        assert!(src.contains("Compute → consumer (3 warpgroups)"));
        assert!(src.contains("Store → storer (1 warpgroups)"));

        // Both regions composed into the same __global__.
        assert!(src.contains("region 0 (attn_prefill)"));
        assert!(src.contains("region 1 (attn_prefill)"));

        // Inter-region barrier lowered from the ControlEdge.
        assert!(src.contains("inter-region barrier: region 0 → region 1 (Barrier)"));
        assert!(src.contains("gbar_sync(&gbar_counter);"));

        // Per-region body is real: load_q expansion from emit_ops,
        // pipeline-source K/V slot rotation, mbarrier on raw edges.
        // Address-expression refactor (task 5): Q gmem is offset
        // by the rendered `LoadAddr`, not handed row/col axes.
        assert!(src.contains(
            "tma_load_2d<SMEM_Q_BYTES>(smem_q, Q_gmem + (q_tile * 16384u + head_group * 128u));"
        ));
        // K-tile load is pipelined (+3 on the serial axis) AND sits on
        // the serial `kv_tile` axis, so the address shifts.
        assert!(src.contains(
            "tma_load_2d<SMEM_K_BYTES>(smem_k[slot], K_gmem + ((kv_tile + 3) * 8192u + head_group * 128u));"
        ));
        assert!(src.contains("uint32_t slot = (kv_tile + 3) % 3;"));

        // Region 0 is Infinite so it has no `window_in_tiles` scalar;
        // region 1 is Finite so it does — the signature unions them.
        assert!(src.contains("uint32_t num_q_tiles"));
        assert!(src.contains("uint32_t num_kv_tiles"));
        assert!(src.contains("uint32_t window_in_tiles"));

        // Gmem pointer plumbing: attention tags touch Q/K/V (read) +
        // O (write). Signature should carry all four with proper const
        // qualifiers.
        assert!(src.contains("const bf16* __restrict__ Q_gmem"));
        assert!(src.contains("const bf16* __restrict__ K_gmem"));
        assert!(src.contains("const bf16* __restrict__ V_gmem"));
        assert!(src.contains("bf16* __restrict__ O_gmem"));
        // Global g.Bar counter is declared above the kernel.
        assert!(src.contains("__device__ uint32_t gbar_counter;"));
    }

    #[test]
    fn launcher_mirrors_kernel_signature() {
        let mk = two_region_mega();
        let src = emit_megakernel(&mk, &sm90_fa2()).unwrap();
        // Launcher present under extern "C" for FFI.
        assert!(src.contains("extern \"C\" cudaError_t launch_mega_kernel("));
        // Zeroes g.Bar counter once per launch.
        assert!(src.contains("cudaMemcpyToSymbolAsync(gbar_counter"));
        // 1 CTA per SM × 20 warps = 640 threads.
        assert!(src.contains("cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount"));
        assert!(src.contains("dim3 grid((unsigned)sm_count);"));
        assert!(src.contains("dim3 block(640u);"));
        // Same param list, forwarded to the kernel.
        assert!(src.contains("mega_kernel<<<grid, block, 0, stream>>>("));
        assert!(src.contains("    num_q_tiles"));
        assert!(src.contains("    Q_gmem"));
    }

    #[test]
    fn gmem_access_union_promotes_to_mutable() {
        // A tag set that both reads and writes Q_gmem (attention reads
        // load_q_tile, qkv_rope writes store_q_row) must drop the
        // const qualifier — both stencils agree on the name so dedupe
        // hoists to a single mutable ptr.
        use crate::template::{QkvRopeParams, qkv_rope_region};

        let mut r_attn = attn_region(&AttnParams {
            window: Window::Infinite,
            head_dim: 128,
            tile_q: 128,
            tile_k: 64,
            num_head_groups: 8,
            pipe: 3,
        });
        r_attn.id = 0;
        let mut r_qkv = qkv_rope_region(&QkvRopeParams {
            hidden_dim: 4096,
            head_dim: 128,
            num_q_heads: 32,
            num_kv_heads: 8,
            token_tile: 64,
            k_tile: 32,
            pipe: 3,
            writes_kv_cache: false,
        });
        r_qkv.id = 1;
        let mk = Megakernel {
            regions: vec![r_attn, r_qkv],
            control: vec![ControlEdge {
                src: 1,
                dst: 0,
                kind: DepKind::Barrier,
            }],
        };
        let src = emit_megakernel(&mk, &sm90_fa2()).unwrap();
        // Q_gmem is Read by attention, Write by qkv_rope → ReadWrite,
        // so no `const`.
        assert!(
            src.contains("bf16* __restrict__ Q_gmem"),
            "Q_gmem must be mutable when read + written",
        );
        assert!(
            !src.contains("const bf16* __restrict__ Q_gmem"),
            "Q_gmem must not carry the const qualifier",
        );
    }

    #[test]
    fn gmem_bindings_populate_per_region_identities_in_signature() {
        // Two GEMM regions, each bound to a distinct identity for
        // (A, B, C). Pre-4b this is what item 4 needs for real models:
        // layer 3's Wqkv and layer 7's Wqkv become separate params.
        // Unbound regions still show canonical names — mixed behavior
        // is deliberately supported during the 4a→4b transition.
        use crate::template::{GemmParams, gemm_region};
        let params = GemmParams {
            m_tile: 128,
            n_tile: 128,
            k_tile: 32,
            pipe: 3,
        };
        let mut r0 = gemm_region(&params);
        r0.id = 0;
        r0.gmem_bindings = vec![
            ("A_gmem", "layer_3_X"),
            ("B_gmem", "layer_3_Wqkv"),
            ("C_gmem", "layer_3_QKV"),
        ];
        let mut r1 = gemm_region(&params);
        r1.id = 1;
        r1.gmem_bindings = vec![
            ("A_gmem", "layer_7_X"),
            ("B_gmem", "layer_7_Wqkv"),
            ("C_gmem", "layer_7_QKV"),
        ];
        let mk = Megakernel {
            regions: vec![r0, r1],
            control: vec![ControlEdge {
                src: 0,
                dst: 1,
                kind: DepKind::Barrier,
            }],
        };
        let src = emit_megakernel(&mk, &sm90_fa2()).unwrap();
        // Six distinct gmem pointers in the signature, not two.
        assert!(src.contains("const bf16* __restrict__ layer_3_X"));
        assert!(src.contains("const bf16* __restrict__ layer_3_Wqkv"));
        assert!(src.contains("bf16* __restrict__ layer_3_QKV"));
        assert!(src.contains("const bf16* __restrict__ layer_7_X"));
        assert!(src.contains("const bf16* __restrict__ layer_7_Wqkv"));
        assert!(src.contains("bf16* __restrict__ layer_7_QKV"));
        // No canonical pointer types appear in the signature when
        // every region is bound. (Mbarrier names like
        // `bar_C_gmem_ready` still contain the canonical tag — those
        // are region-local handshake objects, not gmem pointers; item
        // 4 doesn't rename them.)
        assert!(!src.contains("__restrict__ A_gmem"));
        assert!(!src.contains("__restrict__ B_gmem"));
        assert!(!src.contains("__restrict__ C_gmem"));
    }

    #[test]
    fn empty_gmem_bindings_produce_canonical_signature() {
        // Pre-4b: every template leaves `gmem_bindings` empty; the
        // emitter must produce the same signature it did before item
        // 4a landed. This is the backward-compat guarantee that lets
        // the lowering pass populate bindings one subgraph at a time.
        use crate::template::{GemmParams, gemm_region};
        let params = GemmParams {
            m_tile: 128,
            n_tile: 128,
            k_tile: 32,
            pipe: 3,
        };
        let mut r0 = gemm_region(&params);
        r0.id = 0;
        assert!(r0.gmem_bindings.is_empty());
        let mk = Megakernel {
            regions: vec![r0],
            control: vec![],
        };
        let src = emit_megakernel(&mk, &sm90_fa2()).unwrap();
        assert!(src.contains("const bf16* __restrict__ A_gmem"));
        assert!(src.contains("const bf16* __restrict__ B_gmem"));
        assert!(src.contains("bf16* __restrict__ C_gmem"));
    }

    #[test]
    fn sm89_mapping_drops_warpgroup_dispatch() {
        let mk = two_region_mega();
        let src = emit_megakernel(&mk, &sm89_fa2()).expect("emit succeeds");
        // No warpgroup id on SM89 — everything is AllWarps.
        assert!(!src.contains("threadIdx.x / 128u"));
        assert!(src.contains("AllWarps"));
        // Regions still composed, inter-region barrier still there.
        assert!(src.contains("region 0 (attn_prefill)"));
        assert!(src.contains("region 1 (attn_prefill)"));
        assert!(src.contains("gbar_sync(&gbar_counter);"));
        // SM89 per-region intrinsics: cp.async instead of TMA.
        assert!(src.contains("cp_async_128<SMEM_Q_BYTES>(smem_q, Q_gmem"));
    }

    #[test]
    fn control_cycle_is_an_error() {
        let mut mk = two_region_mega();
        mk.control.push(ControlEdge {
            src: 1,
            dst: 0,
            kind: DepKind::Barrier,
        });
        match emit_megakernel(&mk, &sm90_fa2()) {
            Err(EmitError::ControlCycle) => {}
            other => panic!("expected ControlCycle, got {:?}", other),
        }
    }

    #[test]
    fn empty_control_uses_declaration_order_with_default_barrier() {
        // No explicit ControlEdges → regions run in declaration order;
        // the emitter inserts a conservative Gbar between adjacent
        // regions so emitted code is never silently unsynchronized.
        let mut mk = two_region_mega();
        mk.control.clear();
        let src = emit_megakernel(&mk, &sm90_fa2()).expect("emit succeeds");
        let r0_pos = src.find("region 0 (attn_prefill)").unwrap();
        let r1_pos = src.find("region 1 (attn_prefill)").unwrap();
        assert!(r0_pos < r1_pos, "declaration order preserved");
        assert!(src.contains("gbar_sync(&gbar_counter);"));
    }

    #[test]
    fn unknown_region_in_control_errors() {
        let mut mk = two_region_mega();
        mk.control.push(ControlEdge {
            src: 0,
            dst: 42,
            kind: DepKind::Barrier,
        });
        match emit_megakernel(&mk, &sm90_fa2()) {
            Err(EmitError::UnknownRegionInControl(42)) => {}
            other => panic!("expected UnknownRegionInControl(42), got {:?}", other),
        }
    }

    #[test]
    fn each_region_gets_its_own_scope_block_with_locals() {
        // Two attention regions in one megakernel — each region opens
        // its own `{}` scope that declares its fragments + smem +
        // mbarriers + float accumulators. Previously these were file-
        // scope statics in the prelude and both regions stomped on
        // the same S_frag / O_frag / m / l.
        let mk = two_region_mega();
        let src = emit_megakernel(&mk, &sm90_fa2()).expect("emit succeeds");
        // The scope open+close annotations appear for each region.
        assert!(src.contains("// end region 0"));
        assert!(src.contains("// end region 1"));
        // FA2 fragment declarations appear inside each region block
        // (not at file scope).
        assert_eq!(src.matches("StencilFrag S_frag;").count(), 2);
        assert_eq!(src.matches("StencilFrag O_frag;").count(), 2);
        // Float accumulators with their initializers, one set per
        // region.
        assert_eq!(src.matches("float m = -INFINITY;").count(), 2);
        assert_eq!(src.matches("float l = 0.0f;").count(), 2);
        // Smem staging buffers / rings declared region-locally.
        // Backing storage is bf16[BYTES / 2] now, and each region
        // emits a matching `constexpr uint32_t *_BYTES = …;` line.
        assert!(src.contains("constexpr uint32_t SMEM_Q_BYTES = "));
        assert!(src.contains("constexpr uint32_t SMEM_K_BYTES = "));
        assert!(src.contains("__shared__ bf16 smem_q[SMEM_Q_BYTES / 2];"));
        assert!(src.contains("__shared__ bf16 smem_k[PIPE][SMEM_K_BYTES / 2];"));
        assert!(src.contains("__shared__ bf16 smem_v[PIPE][SMEM_V_BYTES / 2];"));
        assert!(src.contains("__shared__ bf16 smem_o[SMEM_O_BYTES / 2];"));
        // Mbarriers for the kv pipeline ring and the o handshake.
        assert!(src.contains("__shared__ Mbarrier bar_kv[PIPE];"));
        assert!(src.contains("__shared__ Mbarrier bar_kv_consumed[PIPE];"));
        assert!(src.contains("__shared__ Mbarrier bar_o_ready;"));
    }

    #[test]
    fn smem_plain_widens_to_ring_when_compute_indexes_slot() {
        // qkv_rope has `load_x_row` (SmemPlain) and `qkv_matmul`
        // (SmemRing via generic_gemm's `smem_x[slot]`). The union
        // must promote to the ring form so the `[slot]` indexing
        // parses; collect_locals' widening rule is the single place
        // that enforces this.
        use crate::template::{QkvRopeParams, qkv_rope_region};
        let mut r = qkv_rope_region(&QkvRopeParams {
            hidden_dim: 4096,
            head_dim: 128,
            num_q_heads: 32,
            num_kv_heads: 8,
            token_tile: 64,
            k_tile: 32,
            pipe: 3,
            writes_kv_cache: false,
        });
        r.id = 0;
        let mk = Megakernel {
            regions: vec![r],
            control: vec![],
        };
        let src = emit_megakernel(&mk, &sm90_fa2()).unwrap();
        // Promoted to ring (not plain) because qkv_matmul indexes slot.
        assert!(src.contains("__shared__ bf16 smem_x[PIPE][SMEM_X_BYTES / 2];"));
        // QKV trifecta keeps its compound shape.
        assert!(src.contains("struct { StencilFrag q, k, v; } QKV_frag;"));
    }

    #[test]
    fn disjoint_gemm_regions_declare_independent_locals() {
        // Two distinct GEMM regions composed into one megakernel —
        // each must declare its own C_frag / smem_a / smem_b.
        // Previously the file-scope placeholder made them share one
        // accumulator across regions.
        use crate::template::{GemmParams, gemm_region};
        let params = GemmParams {
            m_tile: 128,
            n_tile: 128,
            k_tile: 32,
            pipe: 3,
        };
        let mut r0 = gemm_region(&params);
        r0.id = 0;
        let mut r1 = gemm_region(&params);
        r1.id = 1;
        let mk = Megakernel {
            regions: vec![r0, r1],
            control: vec![ControlEdge {
                src: 0,
                dst: 1,
                kind: DepKind::Barrier,
            }],
        };
        let src = emit_megakernel(&mk, &sm90_fa2()).unwrap();
        // One C_frag per region — not one for the whole kernel.
        assert_eq!(src.matches("StencilFrag C_frag;").count(), 2);
        assert_eq!(
            src.matches("__shared__ bf16 smem_a[PIPE][SMEM_A_BYTES / 2];")
                .count(),
            2
        );
        assert_eq!(
            src.matches("__shared__ bf16 smem_b[PIPE][SMEM_B_BYTES / 2];")
                .count(),
            2
        );
        // Each region also emits its own `constexpr uint32_t
        // SMEM_A_BYTES = …;` line so a megakernel with multiple GEMM
        // regions carries one per region, not a single file-scope copy.
        assert_eq!(src.matches("constexpr uint32_t SMEM_A_BYTES = ").count(), 2);
        assert_eq!(
            src.matches("__shared__ Mbarrier bar_C_gmem_ready;").count(),
            2
        );
    }
}
