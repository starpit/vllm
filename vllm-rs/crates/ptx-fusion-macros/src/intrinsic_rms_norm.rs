//! rms_norm intrinsic: generate rms_norm computation from scratch as PTX,
//! tailored to the GEMM's thread-to-row mapping.
//!
//! Instead of extracting from a .ptx file, we GENERATE the rms_norm PTX
//! because the formula is simple and the thread mapping must match the
//! consumer GEMM's tile layout.
//!
//! The generated code has two parts:
//! - **Prologue**: cooperative reduction (sum of squares -> inv_rms) for
//!   each tile row. Runs once before the first A-load in the GEMM's loop.
//! - **Per-element**: at each A-load site, multiply input by weight and
//!   inv_rms (via `{INPUT}` and `{ELEM_IDX}` placeholders).
//!
//! Thread-to-row mapping for CUTLASS bf16 GEMM (PitchLinearWarpRakedThreadMap):
//!   m_row = (tid.x % 32) / 4 + (tid.x / 32) * 16
//!   Each thread handles 2 rows: m_row and m_row + 8
//!   4 threads per row cooperate on the K-dimension reduction
//!
//! This is a first-class intrinsic: it returns a `PointwiseComputation`
//! that composes with `replace_a_loads_with_inline_fn`. No hardcoded
//! CUTLASS registers — thread-to-row is computed from `%tid.x` and
//! `%ctaid.x`, A_ptr/stride come from named params.

/// Build a `PointwiseComputation` for rms_norm prologue injection.
///
/// `tile_m`: the M dimension of the GEMM tile (e.g., 64 for 64x128x32).
/// `tile_n`: the N dimension of the GEMM tile (e.g., 128 for 64x128x32).
/// `entry_name`: the desired fused kernel name.
pub fn rms_norm_computation(
    tile_m: usize,
    tile_n: usize,
    entry_name: &str,
) -> crate::fuse_general::PointwiseComputation {
    let tile_m_log2 = match tile_m {
        64 => 6,
        128 => 7,
        _ => panic!("unsupported tile_m={tile_m}, expected 64 or 128"),
    };
    let tile_n_val = tile_n as u32;

    // ── Extra params (prepended to kernel entry) ──
    let extra_params = vec![
        ".param .u64 _ferrite_rms_weight,".into(),
        ".param .f32 _ferrite_rms_epsilon,".into(),
        ".param .u32 _ferrite_rms_hidden,".into(),
        ".param .u64 _ferrite_rms_a_ptr,".into(),
        ".param .u64 _ferrite_rms_a_stride,".into(),
        ".param .s32 _ferrite_rms_n,".into(), // N dimension for swizzle unswizzle
    ];

    // ── Extra register declarations (named, not numbered) ──
    let extra_reg_decls = vec![
        ".reg .f32 %f_rms_inv0, %f_rms_inv1;".into(),
        ".reg .f32 %f_rms_sq0, %f_rms_sq1;".into(),
        ".reg .f32 %f_rms_eps, %f_rms_hdnf;".into(),
        ".reg .f32 %f_rms_t0, %f_rms_t1;".into(),
        ".reg .f32 %f_rms_wt0, %f_rms_wt1, %f_rms_wt2, %f_rms_wt3;".into(),
        ".reg .f32 %f_rms_wt4, %f_rms_wt5, %f_rms_wt6, %f_rms_wt7;".into(),
        ".reg .f32 %f_rms_inv;".into(), // selected inv_rms for current site
        ".reg .b32 %r_rms_k, %r_rms_step, %r_rms_end, %r_rms_hdn;".into(),
        ".reg .b32 %r_rms_d0, %r_rms_d1, %r_rms_d2, %r_rms_d3;".into(),
        ".reg .b32 %r_rms_w0, %r_rms_w1, %r_rms_w2, %r_rms_w3;".into(),
        ".reg .b32 %r_rms_par;".into(),   // load parity
        ".reg .b32 %r_rms_swiz;".into(),  // swizzle_log for block ID unswizzle
        ".reg .b32 %r_rms_gridn;".into(), // grid_n = ceil(N / tile_n)
        ".reg .b64 %rd_rms_wt, %rd_rms_in, %rd_rms_str;".into(),
        ".reg .b64 %rd_rms_rb0, %rd_rms_rb1, %rd_rms_rb;".into(), // row bases + selected
        ".reg .b64 %rd_rms_cur, %rd_rms_wa;".into(),
        ".reg .b64 %rd_rms_rowbytes;".into(), // stride * 2 (bytes per row)
        ".reg .b64 %rd_rms_koff;".into(),     // K byte offset within row
        ".reg .b16 %h_rms_a, %h_rms_b;".into(),
        ".reg .pred %p_rms_lp, %p_rms_par;".into(),
    ];

    // ── Param loads (emitted once, after register declarations) ──
    let param_loads = vec![
        "ld.param.u64 \t%rd_rms_wt, [_ferrite_rms_weight];".into(),
        "ld.param.f32 \t%f_rms_eps, [_ferrite_rms_epsilon];".into(),
        "ld.param.u32 \t%r_rms_hdn, [_ferrite_rms_hidden];".into(),
        "cvt.rn.f32.u32 \t%f_rms_hdnf, %r_rms_hdn;".into(),
        "ld.param.u64 \t%rd_rms_in, [_ferrite_rms_a_ptr];".into(),
        "ld.param.u64 \t%rd_rms_str, [_ferrite_rms_a_stride];".into(),
    ];

    // ── Prologue: compute inv_rms for 2 tile rows ──
    let mut prologue = Vec::new();

    prologue.push("// -- FERRITE: rms_norm prologue (first-class intrinsic) --".into());

    // Unswizzle ctaid.x to get the M tile index.
    // GemmIdentityThreadblockSwizzle<4> maps block IDs as:
    //   grid_x = grid_m * tile, grid_y = ceil(grid_n / tile)
    //   m_tile = ctaid.x >> swizzle_log
    // where swizzle_log depends on grid_n = ceil(N / tile_n).
    // Read N from _ferrite_rms_n param to compute at runtime.
    prologue.push("ld.param.s32 \t%r_rms_gridn, [_ferrite_rms_n];".into());
    prologue.push(format!(
        "add.s32 \t%r_rms_gridn, %r_rms_gridn, {};",
        tile_n_val - 1
    ));
    prologue.push(format!(
        "shr.u32 \t%r_rms_gridn, %r_rms_gridn, {};",
        tile_n.trailing_zeros()
    )); // grid_n = ceil(N / tile_n)
    // swizzle_log: 0 if grid_n<2, 1 if grid_n<3, 2 if grid_n>=3 (SWIZZLE_N=4)
    prologue.push("mov.u32 \t%r_rms_swiz, 0;".into());
    prologue.push("setp.ge.u32 \t%p_rms_par, %r_rms_gridn, 2;".into());
    prologue.push("@%p_rms_par mov.u32 \t%r_rms_swiz, 1;".into());
    prologue.push("setp.ge.u32 \t%p_rms_par, %r_rms_gridn, 3;".into());
    prologue.push("@%p_rms_par mov.u32 \t%r_rms_swiz, 2;".into());

    // Thread-to-row mapping:
    //   m_rel = (tid.x % 32) / 4 + (tid.x / 32) * 16
    //   m_tile = ctaid.x >> swizzle_log
    //   m_abs = m_tile * tile_m + m_rel
    prologue.push("mov.u32 \t%r_rms_d0, %tid.x;".into());
    prologue.push("and.b32 \t%r_rms_d1, %r_rms_d0, 31;".into()); // lane = tid.x % 32
    prologue.push("shr.u32 \t%r_rms_d1, %r_rms_d1, 2;".into()); // lane / 4
    prologue.push("shr.u32 \t%r_rms_d2, %r_rms_d0, 5;".into()); // warp = tid.x / 32
    prologue.push("shl.b32 \t%r_rms_d2, %r_rms_d2, 4;".into()); // warp * 16
    prologue.push("add.u32 \t%r_rms_d1, %r_rms_d1, %r_rms_d2;".into()); // m_rel
    prologue.push("mov.u32 \t%r_rms_d2, %ctaid.x;".into());
    prologue.push("shr.u32 \t%r_rms_d2, %r_rms_d2, %r_rms_swiz;".into()); // m_tile = ctaid.x >> swizzle_log
    prologue.push(format!("shl.b32 \t%r_rms_d2, %r_rms_d2, {tile_m_log2};")); // m_tile * tile_m
    prologue.push("add.u32 \t%r_rms_d1, %r_rms_d1, %r_rms_d2;".into()); // m_abs (row 0)

    // row_base_0 = a_ptr + m_abs * stride * 2 (bf16 = 2 bytes per element)
    prologue.push("cvt.s64.s32 \t%rd_rms_rb0, %r_rms_d1;".into());
    prologue.push("mul.lo.s64 \t%rd_rms_rb0, %rd_rms_str, %rd_rms_rb0;".into());
    prologue.push("shl.b64 \t%rd_rms_rb0, %rd_rms_rb0, 1;".into());
    prologue.push("add.s64 \t%rd_rms_rb0, %rd_rms_in, %rd_rms_rb0;".into());

    // row_base_1 = row_base_0 + 8 * stride * 2 (the m_rel+8 row)
    prologue.push("shl.b64 \t%rd_rms_rb1, %rd_rms_str, 4;".into()); // stride * 16 = stride * 8 * 2
    prologue.push("add.s64 \t%rd_rms_rb1, %rd_rms_rb0, %rd_rms_rb1;".into());

    // ── Reduction: sum of squares for row 0 ──
    prologue.push("mov.f32 \t%f_rms_sq0, 0f00000000;".into());
    prologue.push("mov.f32 \t%f_rms_sq1, 0f00000000;".into());

    // k_start = (tid.x % 4) * 16 bytes (8 bf16 elements), k_step = 64 bytes
    prologue.push("and.b32 \t%r_rms_k, %r_rms_d0, 3;".into()); // tid.x % 4
    prologue.push("shl.b32 \t%r_rms_k, %r_rms_k, 4;".into()); // * 16 bytes
    prologue.push("mov.u32 \t%r_rms_step, 64;".into()); // 4 threads * 16 bytes
    prologue.push("shl.b32 \t%r_rms_end, %r_rms_hdn, 1;".into()); // hidden * 2 bytes

    // Row 0 loop
    prologue.push("cvt.u64.u32 \t%rd_rms_cur, %r_rms_k;".into());
    prologue.push("add.s64 \t%rd_rms_cur, %rd_rms_rb0, %rd_rms_cur;".into());

    prologue.push("$L_ferrite_rms_r0:".into());
    prologue.push("setp.lt.u32 \t%p_rms_lp, %r_rms_k, %r_rms_end;".into());
    prologue.push("@!%p_rms_lp bra $L_ferrite_rms_r0d;".into());

    prologue.push(
        "ld.global.v4.b32 \t{%r_rms_d0, %r_rms_d1, %r_rms_d2, %r_rms_d3}, [%rd_rms_cur];".into(),
    );

    // Unpack 4 b32 (8 bf16) and accumulate squares
    for j in 0..4 {
        let r = format!("%r_rms_d{j}");
        prologue.push(format!("mov.b32 \t{{%h_rms_a, %h_rms_b}}, {r};"));
        prologue.push("cvt.f32.bf16 \t%f_rms_t0, %h_rms_a;".into());
        prologue.push("cvt.f32.bf16 \t%f_rms_t1, %h_rms_b;".into());
        prologue.push("fma.rn.f32 \t%f_rms_sq0, %f_rms_t0, %f_rms_t0, %f_rms_sq0;".into());
        prologue.push("fma.rn.f32 \t%f_rms_sq0, %f_rms_t1, %f_rms_t1, %f_rms_sq0;".into());
    }

    // Advance cursor
    prologue.push("cvt.u64.u32 \t%rd_rms_wa, %r_rms_step;".into()); // temp
    prologue.push("add.s64 \t%rd_rms_cur, %rd_rms_cur, %rd_rms_wa;".into());
    prologue.push("add.u32 \t%r_rms_k, %r_rms_k, %r_rms_step;".into());
    prologue.push("bra $L_ferrite_rms_r0;".into());
    prologue.push("$L_ferrite_rms_r0d:".into());

    // Row 1 loop (same structure, accumulating to sq1)
    // Reset k_start from tid.x (need to reload since %r_rms_d0 was clobbered)
    prologue.push("mov.u32 \t%r_rms_d0, %tid.x;".into());
    prologue.push("and.b32 \t%r_rms_k, %r_rms_d0, 3;".into());
    prologue.push("shl.b32 \t%r_rms_k, %r_rms_k, 4;".into());
    prologue.push("cvt.u64.u32 \t%rd_rms_cur, %r_rms_k;".into());
    prologue.push("add.s64 \t%rd_rms_cur, %rd_rms_rb1, %rd_rms_cur;".into());

    prologue.push("$L_ferrite_rms_r1:".into());
    prologue.push("setp.lt.u32 \t%p_rms_lp, %r_rms_k, %r_rms_end;".into());
    prologue.push("@!%p_rms_lp bra $L_ferrite_rms_r1d;".into());

    prologue.push(
        "ld.global.v4.b32 \t{%r_rms_d0, %r_rms_d1, %r_rms_d2, %r_rms_d3}, [%rd_rms_cur];".into(),
    );

    for j in 0..4 {
        let r = format!("%r_rms_d{j}");
        prologue.push(format!("mov.b32 \t{{%h_rms_a, %h_rms_b}}, {r};"));
        prologue.push("cvt.f32.bf16 \t%f_rms_t0, %h_rms_a;".into());
        prologue.push("cvt.f32.bf16 \t%f_rms_t1, %h_rms_b;".into());
        prologue.push("fma.rn.f32 \t%f_rms_sq1, %f_rms_t0, %f_rms_t0, %f_rms_sq1;".into());
        prologue.push("fma.rn.f32 \t%f_rms_sq1, %f_rms_t1, %f_rms_t1, %f_rms_sq1;".into());
    }

    prologue.push("mov.u32 \t%r_rms_d0, %tid.x;".into()); // reload before next iter
    prologue.push("cvt.u64.u32 \t%rd_rms_wa, %r_rms_step;".into());
    prologue.push("add.s64 \t%rd_rms_cur, %rd_rms_cur, %rd_rms_wa;".into());
    prologue.push("add.u32 \t%r_rms_k, %r_rms_k, %r_rms_step;".into());
    prologue.push("bra $L_ferrite_rms_r1;".into());
    prologue.push("$L_ferrite_rms_r1d:".into());

    // Butterfly reduction across 4 threads (shfl.sync.bfly xor 1, then xor 2)
    for xor_mask in [1, 2] {
        prologue.push(format!(
            "shfl.sync.bfly.b32 \t%f_rms_t0, %f_rms_sq0, {xor_mask}, 31, -1;"
        ));
        prologue.push("add.f32 \t%f_rms_sq0, %f_rms_sq0, %f_rms_t0;".into());
        prologue.push(format!(
            "shfl.sync.bfly.b32 \t%f_rms_t0, %f_rms_sq1, {xor_mask}, 31, -1;"
        ));
        prologue.push("add.f32 \t%f_rms_sq1, %f_rms_sq1, %f_rms_t0;".into());
    }

    // inv_rms = rsqrt(sum_sq / hidden + epsilon)
    prologue.push("div.rn.f32 \t%f_rms_sq0, %f_rms_sq0, %f_rms_hdnf;".into());
    prologue.push("add.f32 \t%f_rms_sq0, %f_rms_sq0, %f_rms_eps;".into());
    prologue.push("rsqrt.approx.f32 \t%f_rms_inv0, %f_rms_sq0;".into());

    prologue.push("div.rn.f32 \t%f_rms_sq1, %f_rms_sq1, %f_rms_hdnf;".into());
    prologue.push("add.f32 \t%f_rms_sq1, %f_rms_sq1, %f_rms_eps;".into());
    prologue.push("rsqrt.approx.f32 \t%f_rms_inv1, %f_rms_sq1;".into());

    // Precompute row_bytes = stride * 2 (bytes per row, for per-site K-offset computation)
    prologue.push("shl.b64 \t%rd_rms_rowbytes, %rd_rms_str, 1;".into());

    prologue.push("bar.sync \t15;".into()); // avoid conflict with CUTLASS barrier 0
    prologue.push("// -- FERRITE: rms_norm prologue done --".into());

    // ── Per-site: select inv_rms/row_base by parity, load weight, unpack ──
    let mut per_site = Vec::new();

    // Compute K-byte-offset within the row using modular arithmetic.
    // k_byte_offset = (gmem_src - a_ptr) % row_bytes
    // This is correct regardless of which row the load accesses.
    per_site.push("// FERRITE: rms_norm per-site (load weight + select inv_rms)".into());
    per_site.push("sub.s64 \t%rd_rms_koff, {GMEM_SRC}, %rd_rms_in;".into()); // byte offset from A start
    per_site.push("rem.u64 \t%rd_rms_koff, %rd_rms_koff, %rd_rms_rowbytes;".into()); // mod row_bytes

    // Weight address: weight_ptr + k_byte_offset
    per_site.push("add.s64 \t%rd_rms_wa, %rd_rms_wt, %rd_rms_koff;".into());

    // Select inv_rms: compare gmem_src against row_base_1 to determine which row
    // If gmem_src < row_base_1, this is row 0 → inv_rms_0, else row 1 → inv_rms_1
    per_site.push("setp.lt.u64 \t%p_rms_par, {GMEM_SRC}, %rd_rms_rb1;".into());
    per_site.push("selp.f32 \t%f_rms_inv, %f_rms_inv0, %f_rms_inv1, %p_rms_par;".into());

    // Load weight[k:k+8] (4 b32 = 8 bf16)
    per_site.push(
        "@{MASK_PRED} ld.global.v4.b32 \t{%r_rms_w0, %r_rms_w1, %r_rms_w2, %r_rms_w3}, [%rd_rms_wa];"
            .into(),
    );

    // Unpack all 8 weight bf16s into named f32 registers
    for j in 0..4 {
        let r = format!("%r_rms_w{j}");
        let f_lo = format!("%f_rms_wt{}", j * 2);
        let f_hi = format!("%f_rms_wt{}", j * 2 + 1);
        per_site.push(format!("mov.b32 \t{{%h_rms_a, %h_rms_b}}, {r};"));
        per_site.push(format!("cvt.f32.bf16 \t{f_lo}, %h_rms_a;"));
        per_site.push(format!("cvt.f32.bf16 \t{f_hi}, %h_rms_b;"));
    }

    // ── Per-element instructions: multiply by weight and inv_rms ──
    let instructions = vec![
        "mul.f32 {INPUT}, {INPUT}, %f_rms_wt{ELEM_IDX};".into(),
        "mul.f32 {INPUT}, {INPUT}, %f_rms_inv;".into(),
    ];

    crate::fuse_general::PointwiseComputation {
        instructions,
        param_loads,
        prologue,
        extra_reg_decls,
        extra_params,
        per_site,
        entry_name: Some(entry_name.to_string()),
        scratch_f32_count: 0,
        scratch_b32_count: 0,
    }
}
