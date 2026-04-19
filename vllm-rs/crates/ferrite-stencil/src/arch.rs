// SPDX-License-Identifier: Apache-2.0
//! Per-arch mapping table. The stencil IR is arch-neutral; mapping
//! decides which hardware unit each role runs on, what primitive
//! each dep kind lowers to, and the pipeline depth.
//!
//! SM90 dedicates warpgroups to roles (TMA producer/consumer/storer)
//! and uses mbarriers + named semaphores. SM89 inlines everything
//! across all warps and uses `cp.async` groups. Same region IR.

use crate::ir::{DepKind, Region, Role};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareUnit {
    /// SM90-class: one or more warpgroups (4 warps each) dedicated
    /// to a role. 20-warp persistent layout is 5 warpgroups:
    /// `consumer` (3 wg) + `loader` (1 wg) + `storer` (1 wg); the
    /// Megakernel controller wg is reclaimed because we codegen
    /// straight-line dispatch instead of a runtime VM.
    Warpgroup { role_name: &'static str, num_wg: u8 },
    /// SM89-class: role runs on all warps of the CTA (no warpgroup
    /// specialization).
    AllWarps,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BarrierPrim {
    Mbarrier,
    NamedSem { name: &'static str, depth: u32 },
    CpAsyncGroup { depth: u32 },
    Gbar,
    Cluster,
    HostRedispatch,
}

pub struct ArchMap {
    pub name: &'static str,
    pub role: fn(Role, &Region) -> HardwareUnit,
    pub barrier: fn(DepKind) -> BarrierPrim,
    pub pipe_depth: fn(&Region) -> u32,
}

pub fn sm90_fa2() -> ArchMap {
    ArchMap {
        name: "sm90_fa2",
        role: sm90_role,
        barrier: sm90_barrier,
        pipe_depth: |_| 3,
    }
}

pub fn sm89_fa2() -> ArchMap {
    ArchMap {
        name: "sm89_fa2",
        role: sm89_role,
        barrier: sm89_barrier,
        pipe_depth: |_| 2,
    }
}

fn sm90_role(role: Role, _: &Region) -> HardwareUnit {
    match role {
        Role::Load => HardwareUnit::Warpgroup {
            role_name: "loader",
            num_wg: 1,
        },
        Role::Compute => HardwareUnit::Warpgroup {
            role_name: "consumer",
            num_wg: 3,
        },
        Role::Store => HardwareUnit::Warpgroup {
            role_name: "storer",
            num_wg: 1,
        },
    }
}

fn sm90_barrier(k: DepKind) -> BarrierPrim {
    match k {
        DepKind::Raw => BarrierPrim::Mbarrier,
        DepKind::Pipeline => BarrierPrim::NamedSem {
            name: "kv_arrived",
            depth: 3,
        },
        DepKind::AtomicReduce | DepKind::Barrier => BarrierPrim::Gbar,
        DepKind::DataDependent => BarrierPrim::HostRedispatch,
    }
}

fn sm89_role(_: Role, _: &Region) -> HardwareUnit {
    HardwareUnit::AllWarps
}

fn sm89_barrier(k: DepKind) -> BarrierPrim {
    match k {
        DepKind::Raw => BarrierPrim::Mbarrier,
        DepKind::Pipeline => BarrierPrim::CpAsyncGroup { depth: 2 },
        DepKind::AtomicReduce | DepKind::Barrier => BarrierPrim::Gbar,
        DepKind::DataDependent => BarrierPrim::HostRedispatch,
    }
}
