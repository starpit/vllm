/// GEMM kernel MSL emitter.
///
/// Port of MFA's GEMMKernel::createSource().
/// Called at COMPILE TIME by the proc macro to generate an MSL string.
///
/// The emitter composes atoms (CopyAtom, MmaAtom, TransformAtom, EpilogueAtom)
/// into a single MSL kernel function. The atoms determine what the kernel does;
/// the emitter determines the structure (K-loop, barriers, tile addressing).
use crate::atoms::*;
use crate::config::MetalGemmConfig;
use crate::msl_builder::MslBuilder;

/// Build a complete GEMM kernel MSL source string.
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

    emit_headers(&mut msl, config);
    emit_constants(&mut msl, config);
    emit_utilities(&mut msl, config);
    emit_kernel_signature(&mut msl, config);
    msl.open_brace();
    {
        emit_thread_setup(&mut msl, config);
        emit_accumulator_init(&mut msl, config);
        transform.emit_prologue(&mut msl, config);
        emit_k_loop(&mut msl, config, tile_copy, frag_load, transform, mma);
        epilogue.emit_epilogue(&mut msl, config);
        emit_store_c(&mut msl, config);
    }
    msl.close_brace();

    msl.finish()
}

/// Convenience: build a standalone GEMM (identity transform, direct store).
///
/// Uses Metal 4 native simdgroup_matrix API (simdgroup_load/store/multiply_accumulate).
/// No custom headers needed — just #include <metal_stdlib>.
pub fn build_standalone_gemm(config: &MetalGemmConfig) -> String {
    let mut msl = MslBuilder::new();

    msl.raw("#include <metal_stdlib>");
    msl.raw("using namespace metal;");
    msl.blank();

    // Constants
    msl.set("BLOCK_M", config.block_m.to_string());
    msl.set("BLOCK_N", config.block_n.to_string());
    msl.set("BLOCK_K", config.block_k.to_string());
    msl.set("REGISTER_M", config.register_m().to_string());
    msl.set("REGISTER_N", config.register_n().to_string());
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

bool A_trans = {{A_TRANS}};
bool B_trans = {{B_TRANS}};
"#,
    );

    // Accumulators — native simdgroup_matrix types
    let tiles_m = config.register_m() / 8;
    let tiles_n = config.register_n() / 8;
    msl.set("TILES_M", tiles_m.to_string());
    msl.set("TILES_N", tiles_n.to_string());
    let k_tiles = config.block_k / 8;
    msl.set("K_TILES", k_tiles.to_string());

    msl.block(
        r#"
// Accumulators: TILES_M × TILES_N output tiles of 8×8 each.
simdgroup_matrix<{{REGISTER_NAME_C}}, 8> C_sram[{{TILES_M}}][{{TILES_N}}];
for (ushort tm = 0; tm < {{TILES_M}}; tm++) {
    for (ushort tn = 0; tn < {{TILES_N}}; tn++) {
        C_sram[tm][tn] = make_filled_simdgroup_matrix<{{REGISTER_NAME_C}}, 8>(0);
    }
}
"#,
    );

    // K-loop
    msl.block(
        r#"
for (uint k = 0; k < K; k += K_group) {
    // Copy tiles: device → threadgroup (all threads cooperate)
    auto A_block = (threadgroup {{MEMORY_NAME_A}}*)(threadgroup_block);
    auto B_block = (threadgroup {{MEMORY_NAME_B}}*)(threadgroup_block + {{BLOCK_BYTES_A}});
"#,
    );
    msl.indent();

    // Tile copy (loop-based, zero-fill padding)
    let a_lead = config.leading_block_dim('A');
    let b_lead = config.leading_block_dim('B');
    let a_rows = if config.transpose[0] {
        config.block_k
    } else {
        config.block_m
    };
    let b_rows = if config.transpose[1] {
        config.block_k
    } else {
        config.block_n
    };
    msl.set("A_LEAD", a_lead.to_string());
    msl.set("B_LEAD", b_lead.to_string());
    msl.set("A_TG_TOTAL", (a_rows as u32 * a_lead as u32).to_string());
    msl.set("A_TG_COLS", a_lead.to_string());
    msl.set("B_TG_TOTAL", (b_rows as u32 * b_lead as u32).to_string());
    msl.set("B_TG_COLS", b_lead.to_string());

    msl.block(r#"
{
    ushort tid = sidx * 32 + lane_id;
    uint A_lead_dim = A_trans ? M : K;
    uint B_lead_dim = B_trans ? K : N;
    ushort M_tile = min(uint(M_group), M - M_offset);
    ushort N_tile = min(uint(N_group), N - N_offset);
    ushort K_tile = min(uint(K_group), K - k);

    for (ushort i = tid; i < {{A_TG_TOTAL}}; i += {{THREADGROUP_SIZE}}) {
        ushort row = i / {{A_TG_COLS}};
        ushort col = i % {{A_TG_COLS}};
        bool valid = A_trans ? (row < K_tile && col < M_tile) : (row < M_tile && col < K_tile);
        if (valid) {
            // A_trans: tg[k_local, m_local], device A[m, k] → A[(M_offset+col)*A_lead_dim + (k+row)]
            // !A_trans: tg[m_local, k_local], device A[m, k] → A[(M_offset+row)*A_lead_dim + (k+col)]
            uint idx = A_trans
                ? (M_offset + col) * A_lead_dim + (k + row)
                : (M_offset + row) * A_lead_dim + (k + col);
            A_block[i] = A[idx];
        } else {
            A_block[i] = 0;
        }
    }

    for (ushort i = tid; i < {{B_TG_TOTAL}}; i += {{THREADGROUP_SIZE}}) {
        ushort row = i / {{B_TG_COLS}};
        ushort col = i % {{B_TG_COLS}};
        bool valid = B_trans ? (row < K_tile && col < N_tile) : (row < N_tile && col < K_tile);
        if (valid) {
            // B_trans: tg[k_local, n_local], device B[n, k] → B[(N_offset+col)*B_lead_dim + (k+row)]
            // !B_trans: tg[n_local, k_local], device B[n, k] → B[(N_offset+row)*B_lead_dim + (k+col)]
            uint idx = B_trans
                ? (N_offset + col) * B_lead_dim + (k + row)
                : (N_offset + row) * B_lead_dim + (k + col);
            B_block[i] = B[idx];
        } else {
            B_block[i] = 0;
        }
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
"#);

    // Fragment load + multiply (using native simdgroup_load)
    msl.block(
        r#"
// Inner K-step loop: load fragments and multiply-accumulate.
for (ushort kt = 0; kt < K_group / 8; kt++) {
    for (ushort tm = 0; tm < {{TILES_M}}; tm++) {
        for (ushort tn = 0; tn < {{TILES_N}}; tn++) {
            simdgroup_matrix<{{MEMORY_NAME_A}}, 8> A_mat;
            simdgroup_matrix<{{MEMORY_NAME_B}}, 8> B_mat;

            // Tile copy already arranged data in threadgroup memory
            // with the correct layout (transposed if needed).
            // Load without additional transposition.
            simdgroup_load(A_mat, A_block, {{A_LEAD}},
                ulong2(kt * 8, tm * 8));
            simdgroup_load(B_mat, B_block, {{B_LEAD}},
                ulong2(tn * 8, kt * 8));

            simdgroup_multiply_accumulate(C_sram[tm][tn], A_mat, B_mat, C_sram[tm][tn]);
        }
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
"#,
    );

    msl.dedent();
    msl.raw("}"); // end K-loop

    // Store accumulators to device memory
    msl.block(
        r#"
// Store accumulators to device memory.
for (ushort tm = 0; tm < {{TILES_M}}; tm++) {
    for (ushort tn = 0; tn < {{TILES_N}}; tn++) {
        simdgroup_store(C_sram[tm][tn], C + (M_offset + tm * 8) * N,
            N, ulong2(N_offset + tn * 8, 0));
    }
}
"#,
    );

    msl.close_brace(); // end kernel
    msl.finish()
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

    msl.block(
        r#"
constant uint M_group = {{BLOCK_M}};
constant uint N_group = {{BLOCK_N}};
constant uint K_group = {{BLOCK_K}};
"#,
    );
}

fn emit_utilities(msl: &mut MslBuilder, config: &MetalGemmConfig) {
    msl.set("REGISTER_M", config.register_m().to_string());
    msl.set("REGISTER_N", config.register_n().to_string());

    // Morton order helper (2D simdgroup thread layout)
    msl.block(
        r#"
METAL_FUNC ushort2 morton_order(ushort thread_index_in_simdgroup) {
    ushort lane_id = thread_index_in_simdgroup;
    ushort quad_id = lane_id / 4;
    ushort2 result;
    result.x = extract_bits(quad_id, 0, 1) | (extract_bits(lane_id, 0, 1) << 1);
    result.y = extract_bits(quad_id, 1, 2);
    return result * 8;
}
"#,
    );

    // get_sram helper (index into simdgroup_matrix_storage array)
    msl.block(
        r#"
template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* get_sram(
    thread simdgroup_matrix_storage<T> *sram,
    ushort sram_leading_dim,
    ushort2 matrix_origin
) {
    return sram + (matrix_origin.y / 8) * (sram_leading_dim / 8) + (matrix_origin.x / 8);
}
"#,
    );
}

fn emit_kernel_signature(msl: &mut MslBuilder, config: &MetalGemmConfig) {
    msl.set("MEMORY_NAME_A", config.memory_precisions.a.msl_name());
    msl.set("MEMORY_NAME_B", config.memory_precisions.b.msl_name());
    msl.set("MEMORY_NAME_C", config.memory_precisions.c.msl_name());
    msl.set("THREADGROUP_SIZE", config.threadgroup_size().to_string());
    msl.set(
        "THREADGROUP_MEMORY",
        config.threadgroup_memory().to_string(),
    );

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
}

fn emit_thread_setup(msl: &mut MslBuilder, config: &MetalGemmConfig) {
    msl.set("REGISTER_M", config.register_m().to_string());
    msl.set("REGISTER_N", config.register_n().to_string());
    msl.set("SPLITS_N", config.splits[0].to_string());
    msl.set(
        "THREADGROUP_MEMORY",
        config.threadgroup_memory().to_string(),
    );
    msl.set(
        "A_TRANS",
        if config.transpose[0] { "true" } else { "false" },
    );
    msl.set(
        "B_TRANS",
        if config.transpose[1] { "true" } else { "false" },
    );

    msl.block(
        r#"
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

// Transpose flags.
bool A_trans = {{A_TRANS}};
bool B_trans = {{B_TRANS}};
"#,
    );
}

fn emit_accumulator_init(msl: &mut MslBuilder, config: &MetalGemmConfig) {
    msl.set("REGISTER_M", config.register_m().to_string());
    msl.set("REGISTER_N", config.register_n().to_string());
    msl.set("REGISTER_NAME_C", config.register_precisions.c.msl_name());

    msl.block(
        r#"
// Initialize accumulators to zero.
simdgroup_matrix_storage<{{REGISTER_NAME_C}}> C_sram[
    ({{REGISTER_M}} / 8) * ({{REGISTER_N}} / 8)];
#pragma clang loop unroll(full)
for (ushort m = 0; m < {{REGISTER_M}}; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < {{REGISTER_N}}; n += 8) {
        auto C = get_sram(C_sram, {{REGISTER_N}}, ushort2(n, m));
        *C = simdgroup_matrix_storage<{{REGISTER_NAME_C}}>(vec<{{REGISTER_NAME_C}}, 2>(0));
    }
}
"#,
    );
}

fn emit_k_loop(
    msl: &mut MslBuilder,
    config: &MetalGemmConfig,
    tile_copy: &dyn TileCopyAtom,
    frag_load: &dyn FragmentLoadAtom,
    transform: &dyn TransformAtom,
    mma: &dyn MmaAtom,
) {
    let block_bytes_a = config.block_bytes('A');
    let leading_a = config.leading_block_dim('A').to_string();
    let leading_b = config.leading_block_dim('B').to_string();
    let a_trans = config.transpose[0];
    let b_trans = config.transpose[1];

    msl.set("BLOCK_BYTES_A", block_bytes_a.to_string());
    msl.set("MEMORY_NAME_A", config.memory_precisions.a.msl_name());
    msl.set("MEMORY_NAME_B", config.memory_precisions.b.msl_name());
    msl.set("REGISTER_NAME_A", config.register_precisions.a.msl_name());
    msl.set("REGISTER_NAME_B", config.register_precisions.b.msl_name());
    msl.set("REGISTER_NAME_C", config.register_precisions.c.msl_name());
    msl.set("REGISTER_M", config.register_m().to_string());
    msl.set("REGISTER_N", config.register_n().to_string());
    msl.set("LEADING_BLOCK_DIM_A", leading_a.clone());
    msl.set("LEADING_BLOCK_DIM_B", leading_b.clone());

    // Declare fragment arrays before the loop
    msl.block(
        r#"
simdgroup_matrix_storage<{{REGISTER_NAME_A}}> A_sram[
    ({{REGISTER_M}} / 8) * (K_group / 8)];
simdgroup_matrix_storage<{{REGISTER_NAME_B}}> B_sram[
    (K_group / 8) * ({{REGISTER_N}} / 8)];
"#,
    );

    msl.comment("K-loop: iterate over the K dimension in tiles of K_group");
    msl.raw("for (uint k = 0; k < K; k += K_group) {");
    msl.indent();
    {
        // Tile pointers for threadgroup memory
        msl.block(
            r#"
auto A_block = (threadgroup {{MEMORY_NAME_A}}*)(threadgroup_block);
auto B_block = (threadgroup {{MEMORY_NAME_B}}*)(threadgroup_block + {{BLOCK_BYTES_A}});
"#,
        );

        // Phase 0: Tile copy (device → threadgroup)
        tile_copy.emit_tile_load(msl, config);
        tile_copy.emit_tile_sync(msl, config);

        // Compute threadgroup source pointers for fragment loads
        msl.block(
            r#"
ushort2 A_block_offset(morton_offset.x, offset_in_group.y);
ushort2 B_block_offset(offset_in_group.x, morton_offset.y);
auto A_block_src = simdgroup_matrix_storage<{{MEMORY_NAME_A}}>::apply_offset(
    A_block, {{LEADING_BLOCK_DIM_A}}, A_block_offset, A_trans);
auto B_block_src = simdgroup_matrix_storage<{{MEMORY_NAME_B}}>::apply_offset(
    B_block, {{LEADING_BLOCK_DIM_B}}, B_block_offset, B_trans);
"#,
        );

        // Inner K-step loop (8 elements per step within K_group)
        msl.raw("#pragma clang loop unroll(full)");
        msl.raw("for (ushort k_inner = 0; k_inner < K_group; k_inner += 8) {");
        msl.indent();
        {
            // Phase 1: Load A fragments
            frag_load.emit_load_a(msl, config, "k_inner", "A_block_src", &leading_a, a_trans);

            // Phase 1.5: Transform A fragments (FUSION SLOT)
            if !transform.is_identity() {
                transform.emit_k_setup(msl, config, "k_inner");
                transform.emit_transform(msl, config);
            }

            // Phase 2: Load B fragments
            frag_load.emit_load_b(msl, config, "k_inner", "B_block_src", &leading_b, b_trans);

            // Phase 3: Multiply C += A × B
            mma.emit_multiply(msl, config);
        }
        msl.dedent();
        msl.raw("}");

        // Barrier before next K-tile load
        msl.raw("threadgroup_barrier(mem_flags::mem_threadgroup);");
    }
    msl.dedent();
    msl.raw("}");
}

fn emit_store_c(msl: &mut MslBuilder, config: &MetalGemmConfig) {
    msl.set("REGISTER_M", config.register_m().to_string());
    msl.set("REGISTER_N", config.register_n().to_string());
    msl.set("REGISTER_NAME_C", config.register_precisions.c.msl_name());
    msl.set("MEMORY_NAME_C", config.memory_precisions.c.msl_name());

    // Fast path: direct store from registers to device memory.
    // MFA's createStoreC uses this when the tile is fully within bounds.
    msl.block(
        r#"
// Store accumulators to device memory (fast path).
{
    uint2 C_offset(N_offset + offset_in_group.x,
                   M_offset + offset_in_group.y);
    auto C_dst = simdgroup_matrix_storage<{{MEMORY_NAME_C}}>::apply_offset(
        C, N, C_offset);

#pragma clang loop unroll(full)
    for (ushort m = 0; m < {{REGISTER_M}}; m += 8) {
#pragma clang loop unroll(full)
        for (ushort n = 0; n < {{REGISTER_N}}; n += 8) {
            ushort2 origin(n, m);
            auto C_acc = get_sram(C_sram, {{REGISTER_N}}, origin);
            C_acc->store(C_dst, N, origin);
        }
    }
}
"#,
    );
}
