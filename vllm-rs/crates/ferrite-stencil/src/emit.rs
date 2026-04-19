// SPDX-License-Identifier: Apache-2.0
//! CUDA emitter — sketch level.
//!
//! Takes a `(Region, Schedule, ArchMap)` and renders a textual
//! representation of the kernel source that the emitter *will*
//! produce once per-op lowering lands. At sketch level the node
//! bodies are stubs (`op: <tag>()`); everything structural —
//! signature, parallel-axis → grid mapping, role dispatch (SM90
//! warpgroups vs SM89 all-warps), pipeline loop header, barrier
//! primitives, `iter_offset = +P` on prefetch loads — is real.
//!
//! Not compilable yet; the point is to make the kernel *shape*
//! reviewable and snapshot-testable before committing to intrinsic
//! emission (TMA, wgmma, cp.async). Next commits replace the stub
//! bodies with real CUDA; this one nails the skeleton.

use std::fmt::Write;

use crate::arch::{ArchMap, BarrierPrim, HardwareUnit};
use crate::emit_ops::{ExpandCtx, expand as expand_op};
use crate::ir::{Bound, Region, Role};
use crate::wavefront::{Schedule, Step};

pub fn emit_kernel_sketch(region: &Region, schedule: &Schedule, arch: &ArchMap) -> String {
    let pipeline_depth = schedule.pipeline_depth;
    let mut out = String::new();
    writeln!(
        out,
        "// sketch-level emission — structure is real, bodies are stubs",
    )
    .unwrap();
    writeln!(
        out,
        "// region={} arch={} pipeline_depth={}",
        region.name, arch.name, schedule.pipeline_depth,
    )
    .unwrap();

    // Kernel signature + launch hint.
    writeln!(out, "__global__ void {}_kernel(", region.name).unwrap();
    for (i, s) in region.entry_scalars.iter().enumerate() {
        let sep = if i + 1 < region.entry_scalars.len() {
            ","
        } else {
            ""
        };
        writeln!(out, "    uint32_t {}{}", s.name, sep).unwrap();
    }
    writeln!(out, ") {{").unwrap();

    // Grid: parallel axes. Serial axis becomes an in-kernel for-loop.
    let grid_axes: Vec<&str> = schedule
        .parallel_axes
        .iter()
        .map(|a| region.axis(*a).name)
        .collect();
    writeln!(out, "  // grid: ({}).", grid_axes.join(", ")).unwrap();
    for (i, a) in schedule.parallel_axes.iter().enumerate() {
        let ax = region.axis(*a);
        writeln!(
            out,
            "  uint32_t {} = blockIdx.{};",
            ax.name,
            match i {
                0 => "x",
                1 => "y",
                2 => "z",
                _ => "??", // >3-D grids are illegal in CUDA; surfaced if the
                           // template ever declares four parallel axes.
            }
        )
        .unwrap();
    }

    // SM90: role → warpgroup id; SM89: all warps run every role.
    emit_role_prelude(&mut out, region, arch);

    // Preamble.
    writeln!(out, "  // ── preamble ────────────────────────────────").unwrap();
    for step in &schedule.preamble {
        emit_step(&mut out, region, arch, step, "  ", pipeline_depth);
    }

    // Body: serial loop or straight-line.
    if let Some(serial) = schedule.serial_axis {
        let ax = region.axis(serial);
        writeln!(
            out,
            "  // ── body (serial axis = {}) ─────────────",
            ax.name
        )
        .unwrap();
        writeln!(
            out,
            "  for (uint32_t {} = 0; {} < {}; ++{}) {{",
            ax.name,
            ax.name,
            fmt_bound(region, &ax.bound),
            ax.name
        )
        .unwrap();
        for step in &schedule.body {
            emit_step(&mut out, region, arch, step, "    ", pipeline_depth);
        }
        writeln!(out, "  }}").unwrap();
    } else {
        writeln!(out, "  // ── body (no serial axis) ───────────────").unwrap();
        for step in &schedule.body {
            emit_step(&mut out, region, arch, step, "  ", pipeline_depth);
        }
    }

    // Epilogue.
    writeln!(out, "  // ── epilogue ────────────────────────────────").unwrap();
    for step in &schedule.epilogue {
        emit_step(&mut out, region, arch, step, "  ", pipeline_depth);
    }

    writeln!(out, "}}").unwrap();
    out
}

fn emit_role_prelude(out: &mut String, region: &Region, arch: &ArchMap) {
    let mut roles = [Role::Load, Role::Compute, Role::Store];
    roles.sort_by_key(|r| match r {
        Role::Load => 0,
        Role::Compute => 1,
        Role::Store => 2,
    });
    let any_warpgroup = roles
        .iter()
        .any(|r| matches!((arch.role)(*r, region), HardwareUnit::Warpgroup { .. }));
    if !any_warpgroup {
        writeln!(
            out,
            "  // role mapping: AllWarps (no warpgroup specialization)",
        )
        .unwrap();
        return;
    }
    writeln!(out, "  uint32_t wg = threadIdx.x / 128u;").unwrap();
    writeln!(out, "  // role mapping (SM90-class):").unwrap();
    for r in &roles {
        match (arch.role)(*r, region) {
            HardwareUnit::Warpgroup { role_name, num_wg } => {
                writeln!(
                    out,
                    "  //   {:?} → {} ({} warpgroups)",
                    r, role_name, num_wg,
                )
                .unwrap();
            }
            HardwareUnit::AllWarps => {
                writeln!(out, "  //   {:?} → AllWarps", r).unwrap();
            }
        }
    }
}

fn emit_step(
    out: &mut String,
    region: &Region,
    arch: &ArchMap,
    step: &Step,
    indent: &str,
    pipeline_depth: u32,
) {
    let node = region.node(step.node);
    // Fence-before: render each barrier primitive as a CUDA-ish stub.
    for bp in &step.barriers_before {
        writeln!(out, "{}{}", indent, fmt_barrier(bp)).unwrap();
    }
    let expand_ctx = ExpandCtx {
        arch_name: arch.name,
        iter_offset: step.iter_offset,
        pipeline_depth,
    };
    if let Some(body) = expand_op(node.op.tag, &expand_ctx) {
        // Indent every line of the expansion.
        for line in body.lines() {
            writeln!(out, "{}{}", indent, line).unwrap();
        }
        return;
    }
    // Fallback stub: tag is not in the expansion table yet.
    if step.iter_offset != 0 {
        writeln!(
            out,
            "{}{}(/* iter + {} */);  // {:?} (pipeline source)",
            indent, node.op.tag, step.iter_offset, node.role,
        )
        .unwrap();
    } else {
        writeln!(out, "{}{}();  // {:?}", indent, node.op.tag, node.role).unwrap();
    }
}

fn fmt_barrier(bp: &BarrierPrim) -> String {
    match bp {
        BarrierPrim::Mbarrier => "mbarrier_wait();".to_string(),
        BarrierPrim::NamedSem { name, depth } => {
            format!("sem_wait(\"{}\", depth={});", name, depth)
        }
        BarrierPrim::CpAsyncGroup { depth } => {
            format!("cp_async_wait_group({});", depth.saturating_sub(1))
        }
        BarrierPrim::Gbar => "grid.sync();".to_string(),
        BarrierPrim::Cluster => "cluster_sync();".to_string(),
        BarrierPrim::HostRedispatch => "return;  // host redispatch".to_string(),
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
    use crate::template::{AttnParams, Window, attn_region};
    use crate::wavefront::schedule_wavefront;

    fn fa2_prefill_region() -> Region {
        attn_region(&AttnParams {
            window: Window::Infinite,
            head_dim: 128,
            tile_q: 128,
            tile_k: 64,
            num_head_groups: 8,
            pipe: 3,
        })
    }

    #[test]
    fn sm90_sketch_has_warpgroup_roles_and_pipeline_loop() {
        let region = fa2_prefill_region();
        let arch = sm90_fa2();
        let sched = schedule_wavefront(&region, &arch).unwrap();
        let src = emit_kernel_sketch(&region, &sched, &arch);

        assert!(src.contains("__global__ void fa2_prefill_kernel("));
        assert!(src.contains("wg = threadIdx.x / 128u"));
        assert!(src.contains("Load → loader (1 warpgroups)"));
        assert!(src.contains("Compute → consumer (3 warpgroups)"));
        assert!(src.contains("Store → storer (1 warpgroups)"));
        // Serial loop header.
        assert!(src.contains("for (uint32_t kv_tile = 0;"));
        // Preamble now expands load_q_tile into a TMA + mbarrier
        // arrive guarded on the loader warpgroup (see emit_ops).
        assert!(src.contains("tma_load_2d(smem_q, Q_gmem"));
        assert!(src.contains("if (wg == LOADER_WG)"));
        // Pipeline-source loads tagged with +P.
        assert!(src.contains("load_k_tile(/* iter + 3 */);"));
        assert!(src.contains("load_v_tile(/* iter + 3 */);"));
        // Barriers: pipeline edges → NamedSem, raw edges → Mbarrier.
        assert!(src.contains("sem_wait(\"kv_arrived\", depth=3);"));
        assert!(src.contains("mbarrier_wait();"));
        // Epilogue store.
        assert!(src.contains("store_o_tile();  // Store"));
    }

    #[test]
    fn sm89_sketch_is_all_warps_with_cp_async() {
        let region = fa2_prefill_region();
        let arch = sm89_fa2();
        let sched = schedule_wavefront(&region, &arch).unwrap();
        let src = emit_kernel_sketch(&region, &sched, &arch);

        assert!(src.contains("AllWarps (no warpgroup specialization)"));
        // SM89 pipeline lowers to cp.async groups.
        assert!(src.contains("cp_async_wait_group(1);"));
        // No warpgroup-id variable.
        assert!(!src.contains("wg = threadIdx.x"));
        // Same structural bones regardless of arch.
        assert!(src.contains("for (uint32_t kv_tile = 0;"));
        // SM89 load_q_tile expands to cp.async.
        assert!(src.contains("cp_async_128(smem_q, Q_gmem"));
        // store_o_tile still stubs (not yet in the expansion table).
        assert!(src.contains("store_o_tile();  // Store"));
    }

    #[test]
    fn paged_decode_sketch_has_no_serial_loop_when_single_kv_axis() {
        // Decode region uses kv_tile as its serial axis too, so this
        // exercises the same serial-loop path; the purpose here is
        // to make sure the alternate template round-trips through
        // emission without panicking or mangling axis names.
        use crate::template::{PagedDecodeParams, attn_region_paged_decode};

        let region = attn_region_paged_decode(&PagedDecodeParams {
            head_dim: 128,
            tile_k: 64,
            num_head_groups: 8,
            pipe: 3,
            tokens_per_page: 256,
        });
        let arch = sm90_fa2();
        let sched = schedule_wavefront(&region, &arch).unwrap();
        let src = emit_kernel_sketch(&region, &sched, &arch);
        assert!(src.contains("__global__ void paged_decode_kernel("));
        // Decode's parallel axis is `b` (batch).
        assert!(src.contains("blockIdx.x"));
    }
}
