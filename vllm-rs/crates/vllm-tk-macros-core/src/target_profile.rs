// SPDX-License-Identifier: Apache-2.0
//! Per-target hardware profile.
//!
//! Captures the device-specific values that show up in the schedule
//! shape, the codegen, and the FlashInfer plan: SM count, how many
//! megakernel CTAs fit per SM under cooperative-launch residency
//! constraints, the per-block dynamic shmem ceiling, etc.
//!
//! Why this exists: hardcoding values like "L4 has 58 SMs" or "use
//! 116 cooperative blocks" inside the codegen, the schedule, the
//! launcher template, **and** the FlashInfer planner shim is exactly
//! the kind of magic-number sprawl that bit rots. A single
//! `TargetProfile` value flows through the pipeline, each consumer
//! reads the field it needs, and adding a new arch (sm_90, sm_100)
//! is a single new constructor.
//!
//! ## Cooperative residency
//!
//! `cooperative_blocks_per_sm` is **not** "what hardware allows" —
//! it's "what THIS megakernel achieves." The megakernel's per-CTA
//! dynamic shmem footprint is dominated by FlashInfer's
//! `KTraits::SharedStorage` (~50-70 KiB on bf16 head_dim=64 prefill).
//! L4's per-SM shmem carveout maxes out around 99 KiB, so two CTAs
//! at 64 KiB each (= 128 KiB) don't fit and we land at one CTA per
//! SM under `cudaLaunchCooperativeKernel`. If a future megakernel
//! variant fits its smem in ≤ 49 KiB per CTA, that variant gets a
//! profile with `cooperative_blocks_per_sm = 2` and 2× the grid.
//!
//! ## Why not query at runtime
//!
//! The schedule + codegen pick a CTA pool size at *codegen time*
//! (it's compiled into the megakernel's `NUM_CTAS` constant and
//! drives the wave bin packing). We need the right number BEFORE
//! the kernel exists, so a runtime query isn't useful — the build
//! script picks the profile based on `compute_cap` from cudaforge.

/// Per-target hardware profile. Single source of truth for
/// device-specific values that flow into the codegen, the
/// scheduler, and the FlashInfer planner shim.
#[derive(Clone, Copy, Debug)]
pub struct TargetProfile {
    /// Streaming multiprocessor count
    /// (`cudaDevAttrMultiProcessorCount`).
    pub num_sm: u32,
    /// How many megakernel CTAs fit per SM under
    /// cooperative-launch residency constraints. Determined by the
    /// megakernel's per-CTA dynamic shmem + register footprint vs
    /// the SM's per-block carveout. With FlashInfer's KTraits1
    /// SharedStorage (~50-70 KiB) this is **1** on L4 — two CTAs
    /// at 64 KiB each can't fit a 99 KiB carveout.
    pub cooperative_blocks_per_sm: u32,
    /// Per-block dynamic shmem ceiling, in bytes — the value passed
    /// to `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`.
    pub max_dynamic_shmem_bytes: u32,
}

impl TargetProfile {
    /// L4 (sm_89). 58 SMs, 99 KiB max dynamic shmem per block.
    /// Megakernel + FlashInfer attention SharedStorage forces
    /// `cooperative_blocks_per_sm = 1`.
    pub const fn l4_sm89() -> Self {
        Self {
            num_sm: 58,
            cooperative_blocks_per_sm: 1,
            max_dynamic_shmem_bytes: 99 * 1024,
        }
    }

    /// Cooperative grid size = `num_sm * cooperative_blocks_per_sm`.
    /// This is the value `cudaLaunchCooperativeKernel` will accept
    /// for the megakernel, and the value that
    /// `BlockBatchPagedAttentionPersistent::Run` will see as
    /// `gridDim.y` (so the `work_indptr[blockIdx.y]` indexing
    /// covers all the planned work).
    pub const fn cooperative_grid_size(&self) -> u32 {
        self.num_sm * self.cooperative_blocks_per_sm
    }
}

impl Default for TargetProfile {
    /// L4 (sm_89) — the only target wired into the build today.
    /// Replace with a per-arch lookup once we add sm_90 / sm_100.
    fn default() -> Self {
        Self::l4_sm89()
    }
}
