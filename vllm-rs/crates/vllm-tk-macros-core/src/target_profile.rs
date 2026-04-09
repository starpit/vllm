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

// ─────────────────────────────────────────────────────────────────────
// Per-kernel-class library choices.
//
// Each enum captures the *class* of kernel implementation an arch
// wants for a given phase. The `TargetProfile` constructor for an
// arch picks the right variants. The codegen reads them and renders
// the corresponding template branch — adding a new arch is one
// constructor + (potentially) new variants in these enums + new
// template branches in the megakernel template, **never** an edit
// to the schedule, the launcher scaffold, or the per-tile dispatch.

/// Which GEMM implementation the megakernel's gate_up / down /
/// qkv / o_proj phases dispatch to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmKernelChoice {
    /// Hand-written wmma + 2-stage cp.async, the body that ships
    /// in `templates/scheduled/megakernel.cu` today. Bounded around
    /// 5-10% peak — only viable as a fallback for new arches before
    /// a CUTLASS path is wired.
    HandWrittenWmma,
    /// CUTLASS sm80 multistage GEMM via
    /// `cutlass::gemm::collective::CollectiveMma<MainloopSm80CpAsync, …>`.
    /// `__device__` callable from inside the megakernel's dispatch
    /// arm. The right path for sm_80 / sm_86 / sm_89.
    CutlassSm80Multistage {
        /// Per-CTA tile shape (M, N, K) for the GEMM. Picked per
        /// arch based on register file size and L1 latency.
        tile_m: u32,
        tile_n: u32,
        tile_k: u32,
        /// Number of cp.async pipeline stages (typically 3-5 on
        /// sm_80; 4 is a good default).
        pipeline_stages: u32,
    },
    /// CUTLASS sm90 warp-specialized TMA SS GEMM. For Hopper/H100.
    /// Not yet implemented; placeholder so the enum is open for
    /// future arches.
    CutlassSm90WarpspecializedSs,
}

/// Which attention implementation the megakernel dispatches to for
/// per-layer attention waves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKernelChoice {
    /// FlashInfer's `BlockBatchPagedAttentionPersistent::Run`. Works
    /// on sm_80+ via the persistent runner pattern. The current
    /// production path on L4.
    FlashInferPersistent,
    /// Hand-written cooperative FA-2 from earlier in this project's
    /// history (Phase 4). Kept as a fallback for arches where
    /// FlashInfer isn't an option.
    HandWrittenCooperativeFa2,
}

/// Which RMSNorm implementation the megakernel dispatches to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKernelChoice {
    /// Hand-written warp-shuffle reduction. Today's body in
    /// `templates/scheduled/megakernel.cu`.
    HandWrittenWarpShuffle,
    /// FlashInfer's `norm.cuh` device functions. The right path
    /// for sm_80+ once we wire it.
    FlashInferNormCuh,
}

/// Which RoPE implementation the megakernel dispatches to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeKernelChoice {
    /// Hand-written warp-0 split-half rotation. Today's body.
    HandWrittenSplitHalf,
    /// FlashInfer's `pos_enc.cuh` device functions.
    FlashInferPosEncCuh,
}

/// Per-target hardware profile. Single source of truth for
/// device-specific values that flow into the codegen, the
/// scheduler, the launcher, and the FlashInfer planner shim.
///
/// Adding a new arch (sm_90, sm_100, …) is **one new constructor**
/// — never a grep-and-replace through the rest of the tree.
#[derive(Clone, Copy, Debug)]
pub struct TargetProfile {
    // ── Hardware shape ─────────────────────────────────────────────
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

    // ── Kernel choices ─────────────────────────────────────────────
    /// Which GEMM implementation gate_up / down / qkv / o_proj
    /// dispatch to. The codegen renders the matching template
    /// branch. Adding a new variant is a new template branch +
    /// new arch constructor — schedule and dispatch are unchanged.
    pub gemm_kernel: GemmKernelChoice,
    /// Which attention implementation the per-layer attention wave
    /// dispatches to.
    pub attention_kernel: AttentionKernelChoice,
    /// Which RMSNorm implementation attn_norm / mlp_norm dispatch to.
    pub norm_kernel: NormKernelChoice,
    /// Which RoPE implementation rope dispatches to.
    pub rope_kernel: RopeKernelChoice,
}

impl TargetProfile {
    /// L4 (sm_89). 58 SMs, 99 KiB max dynamic shmem per block.
    /// Megakernel + FlashInfer attention SharedStorage forces
    /// `cooperative_blocks_per_sm = 1`.
    ///
    /// Kernel choices: FlashInfer attention is wired (Phase C2b);
    /// CUTLASS sm80 multistage is wired in Phase E for the GEMMs;
    /// norm and rope are still hand-written until Phase F/G.
    pub const fn l4_sm89() -> Self {
        Self {
            num_sm: 58,
            cooperative_blocks_per_sm: 1,
            max_dynamic_shmem_bytes: 99 * 1024,

            // Phase E2b: GEMM phases dispatch through the
            // pfl_cutlass / pfl_cutlass_small namespaces vendored
            // into the megakernel template (matching the existing
            // fused prefill kernel's 42 ms baseline). The tile shape
            // and pipeline stages here are informational — the
            // namespaces themselves are hardcoded with the
            // matching values, since CUTLASS template instantiation
            // happens at C++ template-instantiation time, not
            // codegen time.
            gemm_kernel: GemmKernelChoice::CutlassSm80Multistage {
                tile_m: 256,
                tile_n: 128,
                tile_k: 32,
                pipeline_stages: 4,
            },
            attention_kernel: AttentionKernelChoice::FlashInferPersistent,
            norm_kernel: NormKernelChoice::HandWrittenWarpShuffle,
            rope_kernel: RopeKernelChoice::HandWrittenSplitHalf,
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

    /// Float workspace size (bytes) the FlashInfer planner needs.
    /// Holds `partial_o` (per-cluster per-Q-tile fp32 partial
    /// outputs) and `partial_lse`. Sized as a function of the
    /// cooperative grid (clusters), the worst-case `head_dim`, and
    /// some headroom for KV split fan-out — **not** a hardcoded
    /// "big enough" magic number.
    pub fn flashinfer_float_workspace_bytes(&self, head_dim: u32, num_kv_heads: u32) -> usize {
        // Match flashinfer/scheduler.cuh:1170 — `max_num_kv_splits =
        // 4 * num_clusters * (CTA_TILE_Q_SIZES[0] + CTA_TILE_Q_SIZES[1])`
        // = 4 * num_clusters * (128 + 16). The float workspace
        // needs `max_num_kv_splits * 2 * head_dim * num_kv_heads`
        // bytes for `partial_o` (bf16) plus
        // `max_num_kv_splits * 4 * num_kv_heads` bytes for
        // `partial_lse` (fp32). Round up generously for alignment +
        // future headroom.
        let max_num_kv_splits = 4u64 * (self.cooperative_grid_size() as u64) * (128 + 16);
        let partial_o = max_num_kv_splits * 2 * (head_dim as u64) * (num_kv_heads as u64);
        let partial_lse = max_num_kv_splits * 4 * (num_kv_heads as u64);
        let raw = partial_o + partial_lse;
        // 2× headroom for alignment slack inside `AlignedAllocator`
        // and the planner's `aligned_alloc_offset` calls.
        (raw * 2) as usize
    }

    /// Int workspace size (bytes) the FlashInfer planner needs.
    /// Holds 11 indirection arrays per task × 2 tasks plus the
    /// merge_indptr / merge_o_indices / num_qo_len / len_kv_chunk
    /// arrays. Sized off the planner's `max_total_num_works = 65536`
    /// constant and the same `max_num_kv_splits` as above.
    pub fn flashinfer_int_workspace_bytes(&self) -> usize {
        // 11 indirection arrays × 2 tasks × max_total_num_works × i32
        // + 2 × max_num_kv_splits × i32 (merge arrays)
        // + small constant overhead. Round up to next MiB.
        let max_total_num_works: u64 = 65536;
        let max_num_kv_splits = 4u64 * (self.cooperative_grid_size() as u64) * (128 + 16);
        let raw = 11 * 2 * max_total_num_works * 4 + 2 * max_num_kv_splits * 4 + 4096;
        // 2× headroom for the same alignment-slack reason as the
        // float workspace.
        (raw * 2) as usize
    }
}

impl Default for TargetProfile {
    /// L4 (sm_89) — the only target wired into the build today.
    /// Replace with a per-arch lookup once we add sm_90 / sm_100.
    fn default() -> Self {
        Self::l4_sm89()
    }
}
