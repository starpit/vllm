/// GEMM kernel MSL emitter.
///
/// Composes atoms into a complete Metal kernel using native simdgroup_matrix API.
/// The emitter controls the loop structure; atoms emit the per-phase MSL code.
///
/// Called at COMPILE TIME by the proc macro to generate an MSL string.
use crate::atoms::*;
use crate::config::MetalGemmConfig;
use crate::msl_builder::MslBuilder;

/// Build a complete GEMM kernel MSL source string from composable atoms.
///
/// This is the main entry point. The proc macro calls this with the desired
/// atom configuration and gets back an MSL string to embed.
pub fn build_gemm_msl(
    config: &MetalGemmConfig,
    tile_copy: &dyn TileCopyAtom,
    frag_load: &dyn FragmentLoadAtom,
    transform: &dyn TransformAtom,
    mma: &dyn MmaAtom,
    epilogue: &dyn EpilogueAtom,
) -> String {
    let mut msl = MslBuilder::new();

    // Set all config-derived template variables once.
    set_config_vars(&mut msl, config);

    // Headers — just the Metal standard library (native simdgroup_matrix API).
    msl.raw("#include <metal_stdlib>");
    msl.raw("using namespace metal;");
    msl.blank();

    // Constants
    msl.block(
        r#"
constant uint M_group = {{BLOCK_M}};
constant uint N_group = {{BLOCK_N}};
constant uint K_group = {{BLOCK_K}};
"#,
    );

    // Kernel signature
    msl.block(
        r#"
kernel void gemm(
    device {{MEMORY_NAME_A}} *A [[buffer(0)]],
    device {{MEMORY_NAME_B}} *B [[buffer(1)]],
    device {{MEMORY_NAME_C}} *C [[buffer(2)]],
    constant uint4 *matrix_offsets [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort sidx [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]]
)
"#,
    );
    msl.open_brace();

    // Thread setup
    msl.block(
        r#"
uint M = matrix_offsets[0][0];
uint N = matrix_offsets[0][1];
uint K = matrix_offsets[0][2];

threadgroup uchar threadgroup_block[{{THREADGROUP_MEMORY}}];

uint M_offset = gid.y * M_group;
uint N_offset = gid.x * N_group;
if (M_offset >= M || N_offset >= N) return;

// Multi-simdgroup: each simdgroup handles a sub-tile within the block.
// sid_m, sid_n = simdgroup position within the splits grid.
ushort sid_n = sidx % {{SPLITS_N}};
ushort sid_m = sidx / {{SPLITS_N}};
uint sid_M_offset = M_offset + sid_m * {{REGISTER_M}};
uint sid_N_offset = N_offset + sid_n * {{REGISTER_N}};

bool A_trans = {{A_TRANS}};
bool B_trans = {{B_TRANS}};
"#,
    );

    // Accumulators — per simdgroup sub-tile
    msl.block(
        r#"
simdgroup_matrix<{{REGISTER_NAME_C}}, 8> C_sram[{{TILES_M}}][{{TILES_N}}];
for (ushort tm = 0; tm < {{TILES_M}}; tm++) {
    for (ushort tn = 0; tn < {{TILES_N}}; tn++) {
        C_sram[tm][tn] = make_filled_simdgroup_matrix<{{REGISTER_NAME_C}}, 8>(0);
    }
}
"#,
    );

    // Transform prologue (e.g., load norm weights before K-loop)
    transform.emit_prologue(&mut msl, config);

    // K-loop
    msl.block(
        r#"
for (uint k = 0; k < K; k += K_group) {
    auto A_block = (threadgroup {{MEMORY_NAME_A}}*)(threadgroup_block);
    auto B_block = (threadgroup {{MEMORY_NAME_B}}*)(threadgroup_block + {{BLOCK_BYTES_A}});
"#,
    );
    msl.indent();

    // Phase 0: Tile copy (device → threadgroup)
    tile_copy.emit_tile_load(&mut msl, config);
    tile_copy.emit_tile_sync(&mut msl, config);

    // Phase 1-3: Fragment load + transform + multiply
    // A_mat loaded once per (kt, tm), reused across tn.
    msl.block(
        r#"
for (ushort kt = 0; kt < K_group / 8; kt++) {
    for (ushort tm = 0; tm < {{TILES_M}}; tm++) {
        simdgroup_matrix<{{MEMORY_NAME_A}}, 8> A_mat;
"#,
    );
    msl.indent();
    msl.indent();

    // Load A
    frag_load.emit_load_a(&mut msl, config);

    // Transform A (fusion slot — identity for standalone GEMM)
    if !transform.is_identity() {
        transform.emit_k_setup(&mut msl, config);
        transform.emit_transform(&mut msl, config);
    }

    // Inner N loop: load B and multiply
    msl.line("for (ushort tn = 0; tn < {{TILES_N}}; tn++) {");
    msl.indent();
    msl.line("simdgroup_matrix<{{MEMORY_NAME_B}}, 8> B_mat;");

    frag_load.emit_load_b(&mut msl, config);
    mma.emit_multiply(&mut msl, config);

    msl.dedent();
    msl.raw("}"); // tn loop

    msl.dedent();
    msl.dedent();
    msl.raw("    }"); // tm loop
    msl.raw("}"); // kt loop

    msl.raw("threadgroup_barrier(mem_flags::mem_threadgroup);");
    msl.dedent();
    msl.raw("}"); // K-loop

    // Epilogue (e.g., SiLU on accumulators)
    epilogue.emit_epilogue(&mut msl, config);

    // Store accumulators to device memory (only tiles within bounds)
    msl.block(
        r#"
for (ushort tm = 0; tm < {{TILES_M}}; tm++) {
    if (sid_M_offset + tm * 8 >= M) continue;
    for (ushort tn = 0; tn < {{TILES_N}}; tn++) {
        if (sid_N_offset + tn * 8 >= N) continue;
        simdgroup_store(C_sram[tm][tn], C + (sid_M_offset + tm * 8) * N,
            N, ulong2(sid_N_offset + tn * 8, 0));
    }
}
"#,
    );

    msl.close_brace(); // kernel
    msl.finish()
}

/// Convenience: build a standalone GEMM with default atoms.
pub fn build_standalone_gemm(config: &MetalGemmConfig) -> String {
    build_gemm_msl(
        config,
        &LoopTileCopy,
        &NativeFragmentLoad,
        &IdentityTransform,
        &NativeMma,
        &IdentityEpilogue,
    )
}

/// Set all config-derived template variables on the MslBuilder.
fn set_config_vars(msl: &mut MslBuilder, config: &MetalGemmConfig) {
    msl.set("BLOCK_M", config.block_m.to_string());
    msl.set("BLOCK_N", config.block_n.to_string());
    msl.set("BLOCK_K", config.block_k.to_string());
    msl.set("REGISTER_M", config.register_m().to_string());
    msl.set("REGISTER_N", config.register_n().to_string());
    msl.set("TILES_M", (config.register_m() / 8).to_string());
    msl.set("TILES_N", (config.register_n() / 8).to_string());
    msl.set("SPLITS_N", config.splits[0].to_string());
    msl.set("SPLITS_M", config.splits[1].to_string());
    msl.set("MEMORY_NAME_A", config.memory_precisions.a.msl_name());
    msl.set("MEMORY_NAME_B", config.memory_precisions.b.msl_name());
    msl.set("MEMORY_NAME_C", config.memory_precisions.c.msl_name());
    msl.set("REGISTER_NAME_C", config.register_precisions.c.msl_name());
    msl.set("THREADGROUP_SIZE", config.threadgroup_size().to_string());
    msl.set(
        "THREADGROUP_MEMORY",
        config.threadgroup_memory().to_string(),
    );
    msl.set("BLOCK_BYTES_A", config.block_bytes('A').to_string());
    msl.set(
        "A_TRANS",
        if config.transpose[0] { "true" } else { "false" },
    );
    msl.set(
        "B_TRANS",
        if config.transpose[1] { "true" } else { "false" },
    );
}
