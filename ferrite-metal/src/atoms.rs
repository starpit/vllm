/// Metal atom traits — the pluggable components of the kernel pipeline.
///
/// Same conceptual interface as CUDA Ferrite's atoms (ferrite-ptx/src/atoms.rs)
/// but emitting MSL instead of PTX.
///
/// These are used at COMPILE TIME by the proc macro. They emit MSL code strings
/// via MslBuilder, not runtime GPU operations.

use crate::msl_builder::MslBuilder;
use crate::config::MetalGemmConfig;

/// How tiles move from device to threadgroup memory.
///
/// apple9 (M3+): simdgroup_event async_copy — hardware DMA
/// apple8 (M1/M2): direct threadgroup loads (async has overhead on older hw)
pub trait MetalCopyAtom {
    /// Emit the tile loading code for one K-iteration.
    /// Includes async copy or direct load, depending on hardware.
    fn emit_tile_load(&self, msl: &mut MslBuilder, config: &MetalGemmConfig);

    /// Emit synchronization after tile loads (threadgroup_barrier or event wait).
    fn emit_sync(&self, msl: &mut MslBuilder, config: &MetalGemmConfig);
}

/// How simdgroup matrix ops consume loaded tiles.
///
/// Metal uses simdgroup_matrix<T, 8, 8> — finer granularity than CUDA's m16n8k16.
/// The inner multiply-accumulate loop operates on 8×8 tiles.
pub trait MetalMmaAtom {
    /// Emit the multiply-accumulate inner loop for one K-step (8 elements).
    /// Loads fragments from threadgroup/device memory, executes simdgroup_multiply.
    fn emit_multiply_accumulate(&self, msl: &mut MslBuilder, config: &MetalGemmConfig);
}

/// How A fragments are transformed between load and MMA.
///
/// This is WHERE FUSION HAPPENS for pre-GEMM operations.
///
/// On Metal, transforms operate on simdgroup_matrix_storage elements.
/// The fragment is in-register after load, before multiply.
///
/// Identity: no transform (standalone GEMM)
/// RmsNorm: multiply fragment elements by norm_factor * gamma
pub trait MetalTransformAtom {
    /// One-time setup before the K-loop.
    fn emit_prologue(&self, _msl: &mut MslBuilder, _config: &MetalGemmConfig) {}

    /// Per-K-iteration setup (e.g., load gamma for this K chunk).
    fn emit_k_setup(&self, _msl: &mut MslBuilder, _config: &MetalGemmConfig, _k_iter: u32) {}

    /// Transform A fragments in-place between load and MMA.
    fn emit_transform(&self, msl: &mut MslBuilder, config: &MetalGemmConfig);
}

/// How accumulators are post-processed before store.
///
/// This is WHERE FUSION HAPPENS for post-GEMM operations.
///
/// Identity: store raw accumulators
/// SiLu: x * sigmoid(x) on accumulator elements
/// ResidualAdd: load residual from device memory, add to accumulators
pub trait MetalEpilogueAtom {
    /// Transform accumulators in-place before store.
    fn emit_epilogue(&self, msl: &mut MslBuilder, config: &MetalGemmConfig);
}

// ═══════════════════════════════════════════════════════════════════
// Default implementations
// ═══════════════════════════════════════════════════════════════════

/// Identity transform — no modification to A fragments.
pub struct IdentityTransform;

impl MetalTransformAtom for IdentityTransform {
    fn emit_transform(&self, _msl: &mut MslBuilder, _config: &MetalGemmConfig) {
        // No-op: fragments pass through unchanged.
    }
}

/// Identity epilogue — store accumulators directly.
pub struct StoreEpilogue;

impl MetalEpilogueAtom for StoreEpilogue {
    fn emit_epilogue(&self, _msl: &mut MslBuilder, _config: &MetalGemmConfig) {
        // No-op: accumulators stored as-is by the pipeline's store phase.
    }
}

/// Async copy loader (apple9 / M3+).
/// Uses simdgroup_event for hardware-accelerated device→threadgroup copies.
pub struct AsyncCopyLoader;

impl MetalCopyAtom for AsyncCopyLoader {
    fn emit_tile_load(&self, msl: &mut MslBuilder, config: &MetalGemmConfig) {
        msl.comment("Async copy: device → threadgroup (simdgroup_event)");
        msl.set("LEADING_BLOCK_DIM_A", config.leading_block_dim('A').to_string());
        msl.set("LEADING_BLOCK_DIM_B", config.leading_block_dim('B').to_string());
        msl.set("BLOCK_BYTES_A", config.block_bytes('A').to_string());

        msl.block(r#"
if (sidx == 0) {
    ushort2 A_tile_src(K_tile_dimension, M_tile_dimension);
    ushort2 B_tile_src(N_tile_dimension, K_tile_dimension);

    simdgroup_event events[2];
    events[0].async_copy<{{LEADING_BLOCK_DIM_A}}, 32>(
        A_block, A_tile_dst, A_src, A_leading_dimension, A_tile_src, A_trans);
    events[1].async_copy<{{LEADING_BLOCK_DIM_B}}, 32>(
        B_block, B_tile_dst, B_src, B_leading_dimension, B_tile_src, B_trans);
    simdgroup_event::wait(2, events);
}
"#);
    }

    fn emit_sync(&self, msl: &mut MslBuilder, _config: &MetalGemmConfig) {
        msl.raw("threadgroup_barrier(mem_flags::mem_threadgroup);");
    }
}

/// Direct loader (apple8 / M1/M2).
/// Each thread loads elements directly from device to threadgroup memory.
pub struct DirectLoader;

impl MetalCopyAtom for DirectLoader {
    fn emit_tile_load(&self, msl: &mut MslBuilder, _config: &MetalGemmConfig) {
        msl.comment("Direct load: device → threadgroup (no async)");
        // MFA's non-async path: each simdgroup loads directly from device memory
        // using simdgroup_matrix_storage::load(), which goes device→register,
        // then the multiply_accumulate function reads from device memory directly.
        // No explicit threadgroup copy needed — the data stays in registers.
    }

    fn emit_sync(&self, _msl: &mut MslBuilder, _config: &MetalGemmConfig) {
        // No threadgroup barrier needed — data is in registers.
    }
}

/// Standard simdgroup MMA using simdgroup_multiply_accumulate.
pub struct SimdgroupMma;

impl MetalMmaAtom for SimdgroupMma {
    fn emit_multiply_accumulate(&self, msl: &mut MslBuilder, config: &MetalGemmConfig) {
        msl.set("REGISTER_M", config.register_m().to_string());
        msl.set("REGISTER_N", config.register_n().to_string());
        msl.set("REGISTER_NAME_A", config.register_precisions.a.msl_name());
        msl.set("REGISTER_NAME_B", config.register_precisions.b.msl_name());
        msl.set("REGISTER_NAME_C", config.register_precisions.c.msl_name());

        msl.block(r#"
#pragma clang loop unroll(full)
for (ushort m = 0; m < {{REGISTER_M}}; m += 8) {
    ushort2 origin(0, m);
    auto A = get_sram(A_sram, 8, origin);
    A->load(A_src, A_leading_dim, ushort2(k, m), A_trans);
}
#pragma clang loop unroll(full)
for (ushort n = 0; n < {{REGISTER_N}}; n += 8) {
    ushort2 origin(n, 0);
    auto B = get_sram(B_sram, {{REGISTER_N}}, origin);
    B->load(B_src, B_leading_dim, ushort2(n, k), B_trans);
}
#pragma clang loop unroll(full)
for (ushort m = 0; m < {{REGISTER_M}}; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < {{REGISTER_N}}; n += 8) {
        auto A = get_sram(A_sram, 8, ushort2(0, m));
        auto B = get_sram(B_sram, {{REGISTER_N}}, ushort2(n, 0));
        auto C = get_sram(C_sram, {{REGISTER_N}}, ushort2(n, m));
        C->multiply(*A, *B);
    }
}
"#);
    }
}
