/// GEMM kernel MSL emitter.
///
/// Port of MFA's GEMMKernel::createSource().
/// Called at COMPILE TIME by the proc macro to generate an MSL string.
///
/// The emitter composes atoms (CopyAtom, MmaAtom, TransformAtom, EpilogueAtom)
/// into a single MSL kernel function. The atoms determine what the kernel does;
/// the emitter determines the structure (K-loop, barriers, tile addressing).

use crate::config::MetalGemmConfig;
use crate::msl_builder::MslBuilder;
use crate::atoms::*;

/// Build a complete GEMM kernel MSL source string.
///
/// This is the main entry point. The proc macro calls this with the desired
/// atom configuration and gets back an MSL string to embed.
pub fn build_gemm_msl(
    config: &MetalGemmConfig,
    copy: &dyn MetalCopyAtom,
    transform: &dyn MetalTransformAtom,
    mma: &dyn MetalMmaAtom,
    epilogue: &dyn MetalEpilogueAtom,
) -> String {
    let mut msl = MslBuilder::new();

    emit_headers(&mut msl, config);
    emit_constants(&mut msl, config);
    emit_utilities(&mut msl, config);
    emit_kernel_signature(&mut msl, config);
    msl.open_brace();
    {
        emit_thread_setup(&mut msl, config);
        emit_accumulator_init(&mut msl, config);
        transform.emit_prologue(&mut msl, config);
        emit_k_loop(&mut msl, config, copy, transform, mma);
        epilogue.emit_epilogue(&mut msl, config);
        emit_store_c(&mut msl, config);
    }
    msl.close_brace();

    msl.finish()
}

/// Convenience: build a standalone GEMM (identity transform, direct store).
pub fn build_standalone_gemm(config: &MetalGemmConfig) -> String {
    let copy: Box<dyn MetalCopyAtom> = if config.prefer_async_load {
        Box::new(AsyncCopyLoader)
    } else {
        Box::new(DirectLoader)
    };

    build_gemm_msl(
        config,
        copy.as_ref(),
        &IdentityTransform,
        &SimdgroupMma,
        &StoreEpilogue,
    )
}

// ═══════════════════════════════════════════════════════════════════
// MSL emission helpers
// ═══════════════════════════════════════════════════════════════════

fn emit_headers(msl: &mut MslBuilder, config: &MetalGemmConfig) {
    // Inline the simdgroup event header (async copy support)
    let event_header = crate::headers::simdgroup_event_header(config.prefer_async_load);
    msl.append(event_header);
    msl.blank();

    // Inline the simdgroup_matrix_storage header (load/store/multiply)
    let needs_bf16 = config.memory_precisions.a == crate::config::Precision::BF16
        || config.memory_precisions.b == crate::config::Precision::BF16;
    let matrix_header = crate::headers::simdgroup_matrix_storage_header(needs_bf16);
    msl.append(matrix_header);
    msl.blank();

    msl.raw("#include <metal_stdlib>");
    msl.raw("using namespace metal;");
    msl.blank();
}

fn emit_constants(msl: &mut MslBuilder, config: &MetalGemmConfig) {
    msl.set("BLOCK_M", config.block_m.to_string());
    msl.set("BLOCK_N", config.block_n.to_string());
    msl.set("BLOCK_K", config.block_k.to_string());
    msl.set("REGISTER_M", config.register_m().to_string());
    msl.set("REGISTER_N", config.register_n().to_string());
    msl.set("SPLITS_M", config.splits[1].to_string());
    msl.set("SPLITS_N", config.splits[0].to_string());
    msl.set("THREADGROUP_SIZE", config.threadgroup_size().to_string());
    msl.set("MEMORY_NAME_A", config.memory_precisions.a.msl_name());
    msl.set("MEMORY_NAME_B", config.memory_precisions.b.msl_name());
    msl.set("MEMORY_NAME_C", config.memory_precisions.c.msl_name());
    msl.set("REGISTER_NAME_A", config.register_precisions.a.msl_name());
    msl.set("REGISTER_NAME_B", config.register_precisions.b.msl_name());
    msl.set("REGISTER_NAME_C", config.register_precisions.c.msl_name());

    msl.block(r#"
constant uint M_group = {{BLOCK_M}};
constant uint N_group = {{BLOCK_N}};
constant uint K_group = {{BLOCK_K}};
"#);
}

fn emit_utilities(msl: &mut MslBuilder, config: &MetalGemmConfig) {
    msl.set("REGISTER_M", config.register_m().to_string());
    msl.set("REGISTER_N", config.register_n().to_string());

    // Morton order helper (2D simdgroup thread layout)
    msl.block(r#"
METAL_FUNC ushort2 morton_order(ushort thread_index_in_simdgroup) {
    ushort lane_id = thread_index_in_simdgroup;
    ushort quad_id = lane_id / 4;
    ushort2 result;
    result.x = extract_bits(quad_id, 0, 1) | (extract_bits(lane_id, 0, 1) << 1);
    result.y = extract_bits(quad_id, 1, 2);
    return result * 8;
}
"#);

    // get_sram helper (index into simdgroup_matrix_storage array)
    msl.block(r#"
template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* get_sram(
    thread simdgroup_matrix_storage<T> *sram,
    ushort sram_leading_dim,
    ushort2 matrix_origin
) {
    return sram + (matrix_origin.y / 8) * (sram_leading_dim / 8) + (matrix_origin.x / 8);
}
"#);
}

fn emit_kernel_signature(msl: &mut MslBuilder, config: &MetalGemmConfig) {
    msl.set("MEMORY_NAME_A", config.memory_precisions.a.msl_name());
    msl.set("MEMORY_NAME_B", config.memory_precisions.b.msl_name());
    msl.set("MEMORY_NAME_C", config.memory_precisions.c.msl_name());
    msl.set("THREADGROUP_SIZE", config.threadgroup_size().to_string());
    msl.set("THREADGROUP_MEMORY", config.threadgroup_memory().to_string());

    msl.block(r#"
kernel void gemm(
    device {{MEMORY_NAME_A}} *A [[buffer(0)]],
    device {{MEMORY_NAME_B}} *B [[buffer(1)]],
    device {{MEMORY_NAME_C}} *C [[buffer(2)]],
    constant uint4 *matrix_offsets [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort sidx [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]]
)
"#);
}

fn emit_thread_setup(msl: &mut MslBuilder, config: &MetalGemmConfig) {
    msl.set("REGISTER_M", config.register_m().to_string());
    msl.set("REGISTER_N", config.register_n().to_string());
    msl.set("SPLITS_N", config.splits[0].to_string());
    msl.set("THREADGROUP_MEMORY", config.threadgroup_memory().to_string());

    msl.block(r#"
// Unpack matrix dimensions from constants buffer.
uint M = matrix_offsets[0][0];
uint N = matrix_offsets[0][1];
uint K = matrix_offsets[0][2];

// Threadgroup memory allocation.
threadgroup uchar threadgroup_block[{{THREADGROUP_MEMORY}}];

// Simdgroup position within the threadgroup.
ushort2 sid(sidx % {{SPLITS_N}}, sidx / {{SPLITS_N}});
ushort2 morton_offset = morton_order(lane_id);

// Global tile position.
uint M_offset = gid.y * M_group;
uint N_offset = gid.x * N_group;

// Early exit for out-of-bounds simdgroups.
if (M_offset + sid.y * {{REGISTER_M}} >= M ||
    N_offset + sid.x * {{REGISTER_N}} >= N) {
    return;
}

ushort2 offset_in_group(sid.x * {{REGISTER_N}} + morton_offset.x,
                        sid.y * {{REGISTER_M}} + morton_offset.y);

// Edge handling.
ushort N_remainder = (N % {{REGISTER_N}} == 0) ? {{REGISTER_N}} : N % {{REGISTER_N}};
ushort N_shift = {{REGISTER_N}} - N_remainder;
uint N_edge = N - (N % N_group);
if ((N_shift != 0) && (gid.x * N_group >= N_edge)) {
    N_offset -= N_shift;
}
"#);
}

fn emit_accumulator_init(msl: &mut MslBuilder, config: &MetalGemmConfig) {
    msl.set("REGISTER_M", config.register_m().to_string());
    msl.set("REGISTER_N", config.register_n().to_string());
    msl.set("REGISTER_NAME_C", config.register_precisions.c.msl_name());

    msl.block(r#"
// Initialize accumulators to zero.
simdgroup_matrix_storage<{{REGISTER_NAME_C}}> C_sram[
    ({{REGISTER_M}} / 8) * ({{REGISTER_N}} / 8)];
#pragma clang loop unroll(full)
for (ushort m = 0; m < {{REGISTER_M}}; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < {{REGISTER_N}}; n += 8) {
        auto C = get_sram(C_sram, {{REGISTER_N}}, ushort2(n, m));
        *C = simdgroup_matrix_storage<{{REGISTER_NAME_C}}>(0);
    }
}
"#);
}

fn emit_k_loop(
    msl: &mut MslBuilder,
    config: &MetalGemmConfig,
    copy: &dyn MetalCopyAtom,
    transform: &dyn MetalTransformAtom,
    mma: &dyn MetalMmaAtom,
) {
    msl.set("BLOCK_BYTES_A", config.block_bytes('A').to_string());
    msl.set("MEMORY_NAME_A", config.memory_precisions.a.msl_name());
    msl.set("MEMORY_NAME_B", config.memory_precisions.b.msl_name());

    msl.comment("K-loop: iterate over the K dimension in tiles of K_group");
    msl.raw("for (uint k = 0; k < K; k += K_group) {");
    msl.indent();
    {
        // Tile pointers
        msl.block(r#"
auto A_block = (threadgroup {{MEMORY_NAME_A}}*)(threadgroup_block);
auto B_block = (threadgroup {{MEMORY_NAME_B}}*)(threadgroup_block + {{BLOCK_BYTES_A}});
"#);

        // Load tiles (CopyAtom)
        copy.emit_tile_load(msl, config);

        // Barrier
        copy.emit_sync(msl, config);

        // Per-K-step transform setup
        transform.emit_k_setup(msl, config, 0);

        // Transform A fragments (TransformAtom slot — WHERE FUSION HAPPENS)
        transform.emit_transform(msl, config);

        // Multiply-accumulate (MmaAtom)
        mma.emit_multiply_accumulate(msl, config);

        // Final barrier before next iteration
        msl.raw("threadgroup_barrier(mem_flags::mem_threadgroup);");
    }
    msl.dedent();
    msl.raw("}");
}

fn emit_store_c(msl: &mut MslBuilder, config: &MetalGemmConfig) {
    msl.set("REGISTER_M", config.register_m().to_string());
    msl.set("REGISTER_N", config.register_n().to_string());
    msl.set("REGISTER_NAME_C", config.register_precisions.c.msl_name());

    msl.block(r#"
// Store accumulators to device memory.
#pragma clang loop unroll(full)
for (ushort m = 0; m < {{REGISTER_M}}; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < {{REGISTER_N}}; n += 8) {
        auto C = get_sram(C_sram, {{REGISTER_N}}, ushort2(n, m));
        uint2 C_offset(N_offset + offset_in_group.x + n,
                       M_offset + offset_in_group.y + m);
        C->store(C, N, C_offset, false);
    }
}
"#);
}
