// Flash Attention Forward kernel — hand-written PTX — HEAD_DIM=128
//
// BLOCK_M=128, BLOCK_N=32, HEAD_DIM=128, 128 threads (4 warps)
// Matches FA2's exact SM89 configuration: 128x32 for non-causal, 48KB smem
// enables 2 CTAs per SM for better occupancy.
//
// Q,K,V,O: [batch*heads, seq, 128] contiguous f16 (stride_seq = 128)
//
// Grid: (ceil(seq_q / 128), batch * heads, 1)
//
// Smem layout (all with B128 swizzle, 128-byte row stride within each page):
//   Q page 0:       0..16383     (128 rows × 64 cols × 2B = 16384)
//   Q page 1:       16384..32767 (128 rows × 64 cols × 2B = 16384)
//   KV buf 0 pg 0:  32768..36863 (32 × 64 × 2B = 4096)
//   KV buf 0 pg 1:  36864..40959 (32 × 64 × 2B = 4096)
//   KV buf 1 pg 0:  40960..45055 (32 × 64 × 2B = 4096)
//   KV buf 1 pg 1:  45056..49151 (32 × 64 × 2B = 4096)
//   Total: 49152 bytes (48KB)
//
// V transpose: V[32×128] staged into Q smem, then transposed to
//   V_t[128×32] in current KV buffer (64-byte row stride, custom swizzle)
//
// MMA config: m16n8k16
//   Q@K^T: S[128×32] — 2 m-tiles × 8 k-iters × 4 n-tiles = 64 MMA
//   P@V:   O[128×128] — 2 m-tiles × 2 k-iters × 16 n-tiles = 64 MMA

pub fn emit_flash_attn_fwd_hdim128() -> String {
    let mut s = String::with_capacity(600 * 1024);
    emit_kernel(&mut s);
    s
}

pub const SMEM_BYTES: u32 = 49152;

fn emit_kernel(s: &mut String) {
    // ─── Header ───
    s.push_str(
        r#".version 8.7
.target sm_89
.address_size 64

.extern .shared .align 16 .b8 global_smem[];

.visible .entry flash_attn_fwd_hdim128(
	.param .u64 .ptr .global .align 16 param_Q,
	.param .u64 .ptr .global .align 16 param_K,
	.param .u64 .ptr .global .align 16 param_V,
	.param .u64 .ptr .global .align 16 param_O,
	.param .u32 param_seq_len,
	.param .f32 param_scale,
	.param .u32 param_stride_batch
)
.reqntid 128
{
	.reg .pred 	%p<30>;
	.reg .b32 	%r<860>;
	.reg .b64 	%rd<80>;
	.reg .b16 	%h<70>;

"#,
    );

    // ─── Parameters ───
    w(s, "ld.param.b64 \t%rd1, [param_Q];");
    w(s, "ld.param.b64 \t%rd2, [param_K];");
    w(s, "ld.param.b64 \t%rd3, [param_V];");
    w(s, "ld.param.b64 \t%rd4, [param_O];");
    w(s, "ld.param.b32 \t%r1, [param_seq_len];");
    w(s, "ld.param.b32 \t%r2, [param_scale];"); // qk_scale (f32 bits)
    w(s, "ld.param.b32 \t%r3, [param_stride_batch];");
    blank(s);

    // ─── Indexing ───
    w(s, "mov.u32 \t%r4, %ctaid.x;"); // block_m index
    w(s, "mov.u32 \t%r5, %ctaid.y;"); // batch_head index
    w(s, "mov.u32 \t%r6, %tid.x;"); // thread id (0..127)
    w(s, "shl.b32 \t%r7, %r4, 7;"); // block_m_start = block_m * 128
    blank(s);

    // Base pointers adjusted for batch/head
    w(s, "mul.lo.s32 \t%r8, %r5, %r3;");
    w(s, "mad.wide.s32 \t%rd10, %r8, 2, %rd1;"); // Q_base
    w(s, "mad.wide.s32 \t%rd11, %r8, 2, %rd2;"); // K_base
    w(s, "mad.wide.s32 \t%rd12, %r8, 2, %rd3;"); // V_base
    w(s, "mad.wide.s32 \t%rd13, %r8, 2, %rd4;"); // O_base
    blank(s);

    // ─── cp.async thread decomposition ───
    // 128 threads, each loads 16 bytes per round.
    // cp_row_in_chunk = tid/8 (0..15), cp_col_idx = tid%8, col_bytes = idx*16
    w(s, "shr.u32 \t%r9, %r6, 3;"); // cp_row_in_chunk = tid/8 (0..15)
    w(s, "and.b32 \t%r10, %r6, 7;"); // cp_col_idx = tid%8
    w(s, "shl.b32 \t%r11, %r10, 4;"); // col_bytes = cp_col_idx * 16
    w(s, "mov.b32 \t%r12, global_smem;");
    blank(s);

    // ─── Load Q → smem (two pages) ───
    // Q is [128 rows × 128 cols], row stride 256 bytes.
    // Page 0 (cols 0-63, smem offset 0): 8 rounds of 16 rows
    // Page 1 (cols 64-127, smem offset 16384): 8 rounds of 16 rows
    for page in 0..2u32 {
        let page_smem_offset = page * 16384;
        let global_col_offset = page * 128; // bytes: 64 cols * 2B
        for round in 0..8u32 {
            let row_base = round * 16;
            // Global address: Q_base + (block_m_start + row_base + cp_row) * 256 + col_bytes + global_col_offset
            w(s, "add.s32 \t%r60, %r7, %r9;"); // block_m_start + cp_row_in_chunk
            if row_base > 0 {
                w(s, &format!("add.s32 \t%r60, %r60, {};", row_base));
            }
            w(s, "setp.lt.s32 \t%p20, %r60, %r1;");
            w(s, "selp.b32 \t%r72, 16, 0, %p20;");
            // Global byte offset: row * 256 + col_bytes + page_col_offset
            w(s, "shl.b32 \t%r62, %r60, 8;"); // row * 256
            w(s, "add.s32 \t%r63, %r62, %r11;"); // + col_bytes
            if global_col_offset > 0 {
                w(s, &format!("add.s32 \t%r63, %r63, {};", global_col_offset));
            }
            w(s, "cvt.u64.u32 \t%rd20, %r63;");
            w(s, "add.s64 \t%rd21, %rd10, %rd20;"); // global addr

            // Smem address within page (128-byte row stride) with B128 swizzle
            let smem_row_base = row_base * 128;
            w(s, "shl.b32 \t%r64, %r9, 7;"); // cp_row_in_chunk * 128
            w(s, "add.s32 \t%r65, %r64, %r11;"); // + col_bytes
            if smem_row_base > 0 {
                w(s, &format!("add.s32 \t%r65, %r65, {};", smem_row_base));
            }
            // B128 swizzle: byte ^ ((byte & 0x380) >> 3)
            w(s, "and.b32 \t%r66, %r65, 896;"); // 0x380
            w(s, "shr.u32 \t%r67, %r66, 3;");
            w(s, "xor.b32 \t%r68, %r65, %r67;");
            w(
                s,
                &format!("add.s32 \t%r69, %r68, {};", page_smem_offset),
            );
            w(s, "add.s32 \t%r69, %r69, %r12;"); // + smem base
            s.push_str(
                "\tcp.async.cg.shared.global [ %r69 + 0 ], [ %rd21 + 0 ], 0x10, %r72;\n",
            );
        }
    }
    w(s, "cp.async.commit_group;");
    w(s, "cp.async.wait_group \t0;");
    w(s, "bar.sync \t0;");
    blank(s);

    // ─── Warp/lane decomposition ───
    w(s, "shr.u32 \t%r30, %r6, 5;"); // warp_id (0..3)
    w(s, "and.b32 \t%r31, %r6, 31;"); // lane_id (0..31)
    blank(s);

    // ─── Q ldmatrix addresses (with B128 swizzle) ───
    // Each warp owns 32 rows of Q (warp_id * 32), split into 2 m-tiles of 16 rows.
    w(s, "and.b32 \t%r32, %r31, 7;"); // row_in_frag = lane % 8
    w(s, "shr.u32 \t%r33, %r31, 3;"); // frag_id = lane / 8 (0..3)
    w(s, "shr.u32 \t%r34, %r33, 1;"); // row_half (0 or 1)
    w(s, "and.b32 \t%r35, %r33, 1;"); // col_half (0 or 1)

    w(s, "shl.b32 \t%r36, %r30, 5;"); // warp_id * 32
    w(s, "shl.b32 \t%r37, %r34, 3;"); // row_half * 8
    w(s, "shl.b32 \t%r40, %r35, 4;"); // col_half * 16 bytes
    blank(s);

    // Load Q fragments for both m-tiles:
    // m-tile 0: Q_frag %r100..%r131 (8 k-iters × 4 regs)
    // m-tile 1: Q_frag %r132..%r163 (8 k-iters × 4 regs)
    for mt in 0..2u32 {
        let mt_row_off = mt * 16;
        for ki in 0..8u32 {
            let page = ki / 4; // page 0 or 1
            let ki_in_page = ki % 4; // 0..3 within page
            let page_offset = page * 16384;
            let k_byte_off = ki_in_page * 32;
            let base = 100 + mt * 32 + ki * 4;

            // Q smem row: warp_id*32 + mt*16 + row_half*8 + row_in_frag
            w(s, "add.s32 \t%r38, %r36, %r37;"); // warp_id*32 + row_half*8
            w(s, "add.s32 \t%r39, %r38, %r32;"); // + row_in_frag
            if mt_row_off > 0 {
                w(s, &format!("add.s32 \t%r39, %r39, {};", mt_row_off));
            }
            // Byte offset within page: row * 128 + col_half * 16 + k_byte_off
            w(s, "shl.b32 \t%r41, %r39, 7;"); // row * 128 bytes
            w(s, "add.s32 \t%r42, %r41, %r40;"); // + col_half * 16
            if k_byte_off > 0 {
                w(s, &format!("add.s32 \t%r70, %r42, {};", k_byte_off));
            } else {
                w(s, "mov.b32 \t%r70, %r42;");
            }
            // B128 swizzle within page
            w(s, "and.b32 \t%r71, %r70, 896;");
            w(s, "shr.u32 \t%r72, %r71, 3;");
            w(s, "xor.b32 \t%r73, %r70, %r72;");
            w(s, &format!("add.s32 \t%r74, %r73, {};", page_offset));
            w(s, "add.s32 \t%r74, %r74, %r12;"); // + smem base
            s.push_str(&format!(
                "\tldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r74];\n",
                base,
                base + 1,
                base + 2,
                base + 3
            ));
        }
    }
    blank(s);

    // ─── K/V ldmatrix addressing (128-byte row stride within page) ───
    // For K (B operand, transposed): ldmatrix.trans.x4
    w(s, "shl.b32 \t%r44, %r33, 4;"); // frag_id * 16 bytes
    w(s, "shl.b32 \t%r45, %r32, 7;"); // (lane%8) * 128
    w(s, "add.s32 \t%r46, %r45, %r44;"); // base within KV page
    blank(s);

    // ─── V_t ldmatrix addressing (64-byte row stride) ───
    w(s, "shl.b32 \t%r47, %r32, 6;"); // (lane%8) * 64
    w(s, "add.s32 \t%r48, %r47, %r44;"); // + frag_id * 16 = base within V_t
    blank(s);

    // ─── V transpose addressing ───
    // Thread tid transposes V column tid to V_t row tid
    w(s, "shr.u32 \t%r49, %r6, 6;"); // v_staging_page = tid/64 (0 or 1)
    w(s, "shl.b32 \t%r50, %r49, 12;"); // page * 4096
    w(s, "and.b32 \t%r51, %r6, 63;"); // col_in_page = tid%64
    w(s, "shl.b32 \t%r52, %r51, 1;"); // col_byte_in_page = col_in_page * 2
    w(s, "shl.b32 \t%r53, %r6, 6;"); // V_t row byte offset = tid * 64
    blank(s);

    // ─── Initialize O accumulators ───
    // O accum: 2 m-tiles × 16 n-tiles × 4 regs = 128 regs
    // m-tile 0: %r200..%r263, m-tile 1: %r264..%r327
    w(s, "mov.b32 \t%r199, 0;");
    for i in 200..264u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r199;\n"));
    }
    for i in 264..328u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r199;\n"));
    }
    // m_i: m-tile 0: %r330, %r331; m-tile 1: %r334, %r335
    w(s, "mov.b32 \t%r330, 0xFF800000;"); // -inf
    w(s, "mov.b32 \t%r331, 0xFF800000;");
    w(s, "mov.b32 \t%r334, 0xFF800000;");
    w(s, "mov.b32 \t%r335, 0xFF800000;");
    // l_i: m-tile 0: %r332, %r333; m-tile 1: %r336, %r337
    w(s, "mov.b32 \t%r332, 0x00000000;");
    w(s, "mov.b32 \t%r333, 0x00000000;");
    w(s, "mov.b32 \t%r336, 0x00000000;");
    w(s, "mov.b32 \t%r337, 0x00000000;");
    blank(s);

    // ─── KV-loop with double-buffered K/V ───
    w(s, "mov.b32 \t%r340, 0;"); // kv_start
    w(s, "setp.lt.s32 \t%p1, %r340, %r1;");
    w(s, "@!%p1 bra \t$L_EPILOGUE;");
    blank(s);

    // Load K[0] into KV buf 0 (smem offset 32768, two pages)
    emit_cp_async_kv_two_page(s, 32768, "%rd11", false);
    w(s, "cp.async.commit_group;");
    w(s, "cp.async.wait_group \t0;");
    w(s, "bar.sync \t0;");
    blank(s);

    s.push_str("$L_KV_LOOP:\n");
    blank(s);

    // ─── Issue cp.async for NEXT K block into alternate buffer ───
    w(s, "add.s32 \t%r341, %r340, 32;"); // next_kv_start
    w(s, "setp.lt.s32 \t%p2, %r341, %r1;"); // has_next_k?

    // Compute current buffer offset: buf = (kv_start >> 5) & 1
    w(s, "shr.u32 \t%r342, %r340, 5;"); // kv_start / 32
    w(s, "and.b32 \t%r343, %r342, 1;"); // buf index (0 or 1)
    w(s, "shl.b32 \t%r344, %r343, 13;"); // buf * 8192
    w(s, "add.s32 \t%r345, %r344, 32768;"); // current KV smem offset

    // Next buffer
    w(s, "xor.b32 \t%r346, %r343, 1;"); // next buf index
    w(s, "shl.b32 \t%r347, %r346, 13;"); // next_buf * 8192
    w(s, "add.s32 \t%r348, %r347, 32768;"); // next KV smem offset

    // Pipeline: start loading next K while we compute with current K
    w(s, "@!%p2 bra \t$L_SKIP_NEXT_K;");
    w(s, "mov.b32 \t%r349, %r340;"); // save current kv_start
    w(s, "mov.b32 \t%r340, %r341;"); // temporarily set kv_start to next
    emit_cp_async_kv_two_page_dynamic(s, "%r348", "%rd11");
    w(s, "cp.async.commit_group;"); // commit group A (next K)
    w(s, "mov.b32 \t%r340, %r349;"); // restore kv_start
    s.push_str("$L_SKIP_NEXT_K:\n");
    blank(s);

    // ─── ldmatrix K from current buffer (B operand for Q@K^T) ───
    // K is [32 rows × 128 cols] in two pages at current buffer.
    // 4 n-tiles (BLOCK_N=32) × 4 k-pairs (HEAD_DIM=128/32=4) = 16 ldmatrix loads
    // K_frag at %r400..%r463
    for n in 0..4u32 {
        for kp in 0..4u32 {
            let page = kp / 2; // page 0 or 1
            let kp_in_page = kp % 2; // 0 or 1
            let page_offset = page * 4096; // within KV buffer
            let n_off = n * 8 * 128; // n * 8 rows * 128 bytes/row
            let kp_off = kp_in_page * 64; // kp * 32 cols * 2 bytes
            let total_off = n_off + kp_off;
            let base = 400 + (n * 4 + kp) * 4;

            w(s, &format!("add.s32 \t%r70, %r46, {};", total_off));
            w(s, "and.b32 \t%r71, %r70, 896;");
            w(s, "shr.u32 \t%r72, %r71, 3;");
            w(s, "xor.b32 \t%r73, %r70, %r72;");
            w(
                s,
                &format!("add.s32 \t%r74, %r73, {};", page_offset),
            );
            w(s, "add.s32 \t%r74, %r74, %r345;"); // + current buf offset
            w(s, "add.s32 \t%r74, %r74, %r12;"); // + smem base
            s.push_str(&format!(
                "\tldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r74];\n",
                base, base + 1, base + 2, base + 3
            ));
        }
    }
    blank(s);

    // ─── MMA: S = Q @ K^T ───
    // S accum for m-tile 0: %r470..%r485 (4 n-tiles × 4 regs = 16 regs)
    // S accum for m-tile 1: %r486..%r501 (4 n-tiles × 4 regs = 16 regs)
    w(s, "mov.b32 \t%r469, 0;");
    for i in 470..486u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r469;\n"));
    }
    for i in 486..502u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r469;\n"));
    }
    blank(s);

    // 2 m-tiles × 8 k-iters × 4 n-tiles = 64 MMA calls
    for mt in 0..2u32 {
        let q_base_start = 100 + mt * 32;
        let s_accum_start = 470 + mt * 16;
        for ki in 0..8u32 {
            let q_base = q_base_start + ki * 4;
            for n in 0..4u32 {
                let s_base = s_accum_start + n * 4;
                let k_pair = ki / 2;
                let sub_ki = ki % 2;
                let k_frag_base = 400 + (n * 4 + k_pair) * 4;
                let b0 = k_frag_base + sub_ki * 2;
                let b1 = b0 + 1;
                s.push_str(&format!(
                    "\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                     {{ %r{}, %r{}, %r{}, %r{} }}, \
                     {{ %r{}, %r{}, %r{}, %r{} }}, \
                     {{ %r{}, %r{} }}, \
                     {{ %r{}, %r{}, %r{}, %r{} }};\n",
                    s_base,
                    s_base + 1,
                    s_base + 2,
                    s_base + 3,
                    q_base,
                    q_base + 1,
                    q_base + 2,
                    q_base + 3,
                    b0,
                    b1,
                    s_base,
                    s_base + 1,
                    s_base + 2,
                    s_base + 3
                ));
            }
        }
    }
    blank(s);

    // ─── Issue cp.async for V staging into Q smem ───
    // V[32 × 128] loaded into Q smem as two pages (reusing Q space).
    // Page 0: V[*][0:63] at smem 0, Page 1: V[*][64:127] at smem 4096
    emit_cp_async_v_staging(s);
    w(s, "cp.async.commit_group;"); // commit group B (V staging)
    blank(s);

    // ─── Online softmax (for both m-tiles) ───
    // Process m-tile 0 (S in %r470..%r485) and m-tile 1 (S in %r486..%r501)

    // Step 1: Scale S
    for i in 470..486u32 {
        s.push_str(&format!("\tmul.f32 \t%r{i}, %r{i}, %r2;\n"));
    }
    for i in 486..502u32 {
        s.push_str(&format!("\tmul.f32 \t%r{i}, %r{i}, %r2;\n"));
    }
    blank(s);

    // Step 2: Row max — for each m-tile, 2 row groups (rows 0..7, 8..15)
    // m-tile 0: row_max in %r502, %r503
    w(s, "mov.b32 \t%r502, 0xFF800000;");
    w(s, "mov.b32 \t%r503, 0xFF800000;");
    for n in 0..4u32 {
        let base = 470 + n * 4;
        s.push_str(&format!("\tmax.f32 \t%r502, %r502, %r{};\n", base));
        s.push_str(&format!("\tmax.f32 \t%r502, %r502, %r{};\n", base + 1));
        s.push_str(&format!("\tmax.f32 \t%r503, %r503, %r{};\n", base + 2));
        s.push_str(&format!("\tmax.f32 \t%r503, %r503, %r{};\n", base + 3));
    }
    // Warp-level reduction via butterfly shuffle
    // With BLOCK_N=32 and 4 n-tiles, each n-tile covers 8 cols.
    // MMA output distribution: thread (lane%4) owns 2 columns within the n-tile.
    // Row max needs reduction across lane%4 (4 lanes that share a row).
    // Actually with n=4 n-tiles and lane%4 indexing, shuffle distance depends:
    // For n-tiles 0..3, lane/4 gives the n-tile-group.
    // Each MMA n-tile produces 2 values per thread. Across 4 n-tiles, we get 8 values
    // in regs s_base+0..+3 for 4 n-tiles. The row-max across all columns requires
    // shuffling to get values from threads that own other columns of the same row.
    // With BLOCK_N=32 and m16n8k16: each n-tile is 8 cols. Thread lane owns
    // 2 elements at col (lane/4)*2 within the 8-col tile. But n-tiles cover
    // cols 0-7, 8-15, 16-23, 24-31. So across all 4 n-tiles, one thread
    // covers 4×2=8 column positions, but these span all 32 cols with stride.
    // The 4 threads at lane%4 = 0,1,2,3 each own different column pairs within
    // each n-tile. So we DO need to reduce across lane%4.
    w(s, "shfl.sync.bfly.b32 \t%r504, %r502, 2, 31, -1;");
    w(s, "max.f32 \t%r502, %r502, %r504;");
    w(s, "shfl.sync.bfly.b32 \t%r505, %r502, 1, 31, -1;");
    w(s, "max.f32 \t%r502, %r502, %r505;");
    w(s, "shfl.sync.bfly.b32 \t%r506, %r503, 2, 31, -1;");
    w(s, "max.f32 \t%r503, %r503, %r506;");
    w(s, "shfl.sync.bfly.b32 \t%r507, %r503, 1, 31, -1;");
    w(s, "max.f32 \t%r503, %r503, %r507;");
    blank(s);

    // m-tile 1: row_max in %r508, %r509
    w(s, "mov.b32 \t%r508, 0xFF800000;");
    w(s, "mov.b32 \t%r509, 0xFF800000;");
    for n in 0..4u32 {
        let base = 486 + n * 4;
        s.push_str(&format!("\tmax.f32 \t%r508, %r508, %r{};\n", base));
        s.push_str(&format!("\tmax.f32 \t%r508, %r508, %r{};\n", base + 1));
        s.push_str(&format!("\tmax.f32 \t%r509, %r509, %r{};\n", base + 2));
        s.push_str(&format!("\tmax.f32 \t%r509, %r509, %r{};\n", base + 3));
    }
    w(s, "shfl.sync.bfly.b32 \t%r510, %r508, 2, 31, -1;");
    w(s, "max.f32 \t%r508, %r508, %r510;");
    w(s, "shfl.sync.bfly.b32 \t%r511, %r508, 1, 31, -1;");
    w(s, "max.f32 \t%r508, %r508, %r511;");
    w(s, "shfl.sync.bfly.b32 \t%r512, %r509, 2, 31, -1;");
    w(s, "max.f32 \t%r509, %r509, %r512;");
    w(s, "shfl.sync.bfly.b32 \t%r513, %r509, 1, 31, -1;");
    w(s, "max.f32 \t%r509, %r509, %r513;");
    blank(s);

    // Step 3: m_new = max(m_old, row_max)
    w(s, "max.f32 \t%r514, %r330, %r502;"); // m-tile 0
    w(s, "max.f32 \t%r515, %r331, %r503;");
    w(s, "max.f32 \t%r516, %r334, %r508;"); // m-tile 1
    w(s, "max.f32 \t%r517, %r335, %r509;");
    blank(s);

    // Step 4: alpha = exp2(m_old - m_new)
    w(s, "sub.f32 \t%r518, %r330, %r514;"); // m-tile 0
    w(s, "sub.f32 \t%r519, %r331, %r515;");
    w(s, "ex2.approx.ftz.f32 \t%r520, %r518;");
    w(s, "ex2.approx.ftz.f32 \t%r521, %r519;");
    w(s, "sub.f32 \t%r522, %r334, %r516;"); // m-tile 1
    w(s, "sub.f32 \t%r523, %r335, %r517;");
    w(s, "ex2.approx.ftz.f32 \t%r524, %r522;");
    w(s, "ex2.approx.ftz.f32 \t%r525, %r523;");
    blank(s);

    // Step 5: P = exp2(S_scaled - m_new)
    for n in 0..4u32 {
        let base = 470 + n * 4;
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r514;\n", b = base));
        s.push_str(&format!("\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n", b = base));
        s.push_str(&format!(
            "\tsub.f32 \t%r{b}, %r{b}, %r514;\n",
            b = base + 1
        ));
        s.push_str(&format!(
            "\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n",
            b = base + 1
        ));
        s.push_str(&format!(
            "\tsub.f32 \t%r{b}, %r{b}, %r515;\n",
            b = base + 2
        ));
        s.push_str(&format!(
            "\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n",
            b = base + 2
        ));
        s.push_str(&format!(
            "\tsub.f32 \t%r{b}, %r{b}, %r515;\n",
            b = base + 3
        ));
        s.push_str(&format!(
            "\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n",
            b = base + 3
        ));
    }
    for n in 0..4u32 {
        let base = 486 + n * 4;
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r516;\n", b = base));
        s.push_str(&format!("\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n", b = base));
        s.push_str(&format!(
            "\tsub.f32 \t%r{b}, %r{b}, %r516;\n",
            b = base + 1
        ));
        s.push_str(&format!(
            "\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n",
            b = base + 1
        ));
        s.push_str(&format!(
            "\tsub.f32 \t%r{b}, %r{b}, %r517;\n",
            b = base + 2
        ));
        s.push_str(&format!(
            "\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n",
            b = base + 2
        ));
        s.push_str(&format!(
            "\tsub.f32 \t%r{b}, %r{b}, %r517;\n",
            b = base + 3
        ));
        s.push_str(&format!(
            "\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n",
            b = base + 3
        ));
    }
    blank(s);

    // Step 6: Row sum of P
    w(s, "mov.b32 \t%r526, 0x00000000;"); // m-tile 0
    w(s, "mov.b32 \t%r527, 0x00000000;");
    for n in 0..4u32 {
        let base = 470 + n * 4;
        s.push_str(&format!("\tadd.f32 \t%r526, %r526, %r{};\n", base));
        s.push_str(&format!("\tadd.f32 \t%r526, %r526, %r{};\n", base + 1));
        s.push_str(&format!("\tadd.f32 \t%r527, %r527, %r{};\n", base + 2));
        s.push_str(&format!("\tadd.f32 \t%r527, %r527, %r{};\n", base + 3));
    }
    w(s, "shfl.sync.bfly.b32 \t%r528, %r526, 2, 31, -1;");
    w(s, "add.f32 \t%r526, %r526, %r528;");
    w(s, "shfl.sync.bfly.b32 \t%r529, %r526, 1, 31, -1;");
    w(s, "add.f32 \t%r526, %r526, %r529;");
    w(s, "shfl.sync.bfly.b32 \t%r530, %r527, 2, 31, -1;");
    w(s, "add.f32 \t%r527, %r527, %r530;");
    w(s, "shfl.sync.bfly.b32 \t%r531, %r527, 1, 31, -1;");
    w(s, "add.f32 \t%r527, %r527, %r531;");

    w(s, "mov.b32 \t%r532, 0x00000000;"); // m-tile 1
    w(s, "mov.b32 \t%r533, 0x00000000;");
    for n in 0..4u32 {
        let base = 486 + n * 4;
        s.push_str(&format!("\tadd.f32 \t%r532, %r532, %r{};\n", base));
        s.push_str(&format!("\tadd.f32 \t%r532, %r532, %r{};\n", base + 1));
        s.push_str(&format!("\tadd.f32 \t%r533, %r533, %r{};\n", base + 2));
        s.push_str(&format!("\tadd.f32 \t%r533, %r533, %r{};\n", base + 3));
    }
    w(s, "shfl.sync.bfly.b32 \t%r534, %r532, 2, 31, -1;");
    w(s, "add.f32 \t%r532, %r532, %r534;");
    w(s, "shfl.sync.bfly.b32 \t%r535, %r532, 1, 31, -1;");
    w(s, "add.f32 \t%r532, %r532, %r535;");
    w(s, "shfl.sync.bfly.b32 \t%r536, %r533, 2, 31, -1;");
    w(s, "add.f32 \t%r533, %r533, %r536;");
    w(s, "shfl.sync.bfly.b32 \t%r537, %r533, 1, 31, -1;");
    w(s, "add.f32 \t%r533, %r533, %r537;");
    blank(s);

    // Step 7: Rescale O accumulators: O *= alpha
    // m-tile 0: 16 n-tiles × 4 regs at %r200..%r263
    for n in 0..16u32 {
        let base = 200 + n * 4;
        s.push_str(&format!(
            "\tmul.f32 \t%r{b}, %r{b}, %r520;\n",
            b = base
        ));
        s.push_str(&format!(
            "\tmul.f32 \t%r{b}, %r{b}, %r520;\n",
            b = base + 1
        ));
        s.push_str(&format!(
            "\tmul.f32 \t%r{b}, %r{b}, %r521;\n",
            b = base + 2
        ));
        s.push_str(&format!(
            "\tmul.f32 \t%r{b}, %r{b}, %r521;\n",
            b = base + 3
        ));
    }
    // m-tile 1: at %r264..%r327
    for n in 0..16u32 {
        let base = 264 + n * 4;
        s.push_str(&format!(
            "\tmul.f32 \t%r{b}, %r{b}, %r524;\n",
            b = base
        ));
        s.push_str(&format!(
            "\tmul.f32 \t%r{b}, %r{b}, %r524;\n",
            b = base + 1
        ));
        s.push_str(&format!(
            "\tmul.f32 \t%r{b}, %r{b}, %r525;\n",
            b = base + 2
        ));
        s.push_str(&format!(
            "\tmul.f32 \t%r{b}, %r{b}, %r525;\n",
            b = base + 3
        ));
    }
    blank(s);

    // Step 8: Update l_i, m_i
    w(s, "fma.rn.f32 \t%r332, %r332, %r520, %r526;"); // m-tile 0
    w(s, "fma.rn.f32 \t%r333, %r333, %r521, %r527;");
    w(s, "mov.b32 \t%r330, %r514;");
    w(s, "mov.b32 \t%r331, %r515;");
    w(s, "fma.rn.f32 \t%r336, %r336, %r524, %r532;"); // m-tile 1
    w(s, "fma.rn.f32 \t%r337, %r337, %r525, %r533;");
    w(s, "mov.b32 \t%r334, %r516;");
    w(s, "mov.b32 \t%r335, %r517;");
    blank(s);

    // ─── Convert P to f16x2 in registers for P@V MMA ───
    // BLOCK_N=32 → 2 k-iters for P@V (32/16=2)
    // m-tile 0: P_frag %r540..%r547 (2 k-iters × 4 regs)
    for ki in 0..2u32 {
        let n_lo = ki * 2;
        let n_hi = n_lo + 1;
        let p_base = 540 + ki * 4;
        let s_lo = 470 + n_lo * 4;
        let s_hi = 470 + n_hi * 4;

        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base,
            s_lo + 1,
            s_lo
        ));
        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base + 1,
            s_hi + 1,
            s_hi
        ));
        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base + 2,
            s_lo + 3,
            s_lo + 2
        ));
        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base + 3,
            s_hi + 3,
            s_hi + 2
        ));
    }
    // m-tile 1: P_frag %r548..%r555
    for ki in 0..2u32 {
        let n_lo = ki * 2;
        let n_hi = n_lo + 1;
        let p_base = 548 + ki * 4;
        let s_lo = 486 + n_lo * 4;
        let s_hi = 486 + n_hi * 4;

        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base,
            s_lo + 1,
            s_lo
        ));
        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base + 1,
            s_hi + 1,
            s_hi
        ));
        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base + 2,
            s_lo + 3,
            s_lo + 2
        ));
        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base + 3,
            s_hi + 3,
            s_hi + 2
        ));
    }
    blank(s);

    // ─── Wait for V staging + next K to complete ───
    w(s, "cp.async.wait_group \t0;");
    w(s, "bar.sync \t0;");
    blank(s);

    // ─── Transpose V from staging (Q smem) to V_t in current KV buffer ───
    // V staging: two pages at smem offsets 0 and 4096 (each [32×64], 128B rows, B128 swizzle)
    // V_t: [128 rows × 32 cols] at current KV buffer, 64B rows, custom swizzle
    // Thread tid transposes V column tid to V_t row tid
    emit_v_transpose(s);
    w(s, "bar.sync \t0;");
    blank(s);

    // ─── ldmatrix V_t from current KV buffer ───
    // V_t is [128 rows × 32 cols], 64-byte row stride, custom swizzle (0x1C0, >>3)
    // 16 n-tiles × 1 k-pair × 4 regs = 64 regs at %r560..%r623
    for n in 0..16u32 {
        let n_off = n * 512; // n * 8 rows * 64 bytes/row
        let base = 560 + n * 4;

        w(s, &format!("add.s32 \t%r70, %r48, {};", n_off));
        // V_t swizzle: byte ^ ((byte & 0xC0) >> 2) — preserves 16B alignment
        w(s, "and.b32 \t%r71, %r70, 192;"); // 0xC0
        w(s, "shr.u32 \t%r72, %r71, 2;");
        w(s, "xor.b32 \t%r73, %r70, %r72;");
        w(s, "add.s32 \t%r74, %r73, %r345;"); // + current buf offset
        w(s, "add.s32 \t%r74, %r74, %r12;"); // + smem base
        s.push_str(&format!(
            "\tldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r74];\n",
            base, base + 1, base + 2, base + 3
        ));
    }
    blank(s);

    // ─── MMA: O += P @ V (both m-tiles) ───
    for mt in 0..2u32 {
        let p_base_start = if mt == 0 { 540 } else { 548 };
        let o_base_start = if mt == 0 { 200 } else { 264 };
        for ki in 0..2u32 {
            let p_base = p_base_start + ki * 4;
            for n in 0..16u32 {
                let o_base = o_base_start + n * 4;
                let k_pair = ki / 2; // always 0
                let sub_ki = ki % 2;
                let v_frag_base = 560 + n * 4;
                let b0 = v_frag_base + sub_ki * 2;
                let b1 = b0 + 1;
                let _ = k_pair; // suppress unused warning in logic
                s.push_str(&format!(
                    "\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                     {{ %r{}, %r{}, %r{}, %r{} }}, \
                     {{ %r{}, %r{}, %r{}, %r{} }}, \
                     {{ %r{}, %r{} }}, \
                     {{ %r{}, %r{}, %r{}, %r{} }};\n",
                    o_base,
                    o_base + 1,
                    o_base + 2,
                    o_base + 3,
                    p_base,
                    p_base + 1,
                    p_base + 2,
                    p_base + 3,
                    b0,
                    b1,
                    o_base,
                    o_base + 1,
                    o_base + 2,
                    o_base + 3
                ));
            }
        }
    }
    blank(s);

    // ─── Loop advance ───
    w(s, "add.s32 \t%r340, %r340, 32;");
    w(s, "setp.lt.s32 \t%p1, %r340, %r1;");
    w(s, "@%p1 bra \t$L_KV_LOOP;");
    blank(s);

    // ═══════════════════════════════════════════════════════════════
    // EPILOGUE: O = O / l_i, convert to f16, store
    // ═══════════════════════════════════════════════════════════════
    s.push_str("$L_EPILOGUE:\n");

    // Compute store addresses for O
    // MMA output layout: lane/4 gives the row within the 16-row m-tile
    w(s, "shr.u32 \t%r770, %r31, 2;"); // lane/4 = mma_row_in_16
    w(s, "and.b32 \t%r771, %r31, 3;"); // lane%4
    w(s, "shl.b32 \t%r772, %r771, 2;"); // (lane%4)*4 bytes
    blank(s);

    // Process both m-tiles
    for mt in 0..2u32 {
        let mt_row_off = mt * 16;
        let o_base_start = if mt == 0 { 200 } else { 264 };
        let l_i_0 = if mt == 0 { 332 } else { 336 };
        let l_i_1 = if mt == 0 { 333 } else { 337 };

        // row_0 = warp_id*32 + mt*16 + lane/4
        w(s, "add.s32 \t%r773, %r770, %r36;"); // lane/4 + warp_id*32
        if mt_row_off > 0 {
            w(s, &format!("add.s32 \t%r773, %r773, {};", mt_row_off));
        }
        w(s, "add.s32 \t%r774, %r773, 8;"); // row_1

        // O /= l_i
        for n in 0..16u32 {
            let base = o_base_start + n * 4;
            s.push_str(&format!(
                "\tdiv.full.f32 \t%r{b}, %r{b}, %r{l};\n",
                b = base,
                l = l_i_0
            ));
            s.push_str(&format!(
                "\tdiv.full.f32 \t%r{b}, %r{b}, %r{l};\n",
                b = base + 1,
                l = l_i_0
            ));
            s.push_str(&format!(
                "\tdiv.full.f32 \t%r{b}, %r{b}, %r{l};\n",
                b = base + 2,
                l = l_i_1
            ));
            s.push_str(&format!(
                "\tdiv.full.f32 \t%r{b}, %r{b}, %r{l};\n",
                b = base + 3,
                l = l_i_1
            ));
        }
        blank(s);

        // Store O to global memory
        w(s, "add.s32 \t%r775, %r7, %r773;"); // global_row_0
        w(s, "add.s32 \t%r776, %r7, %r774;"); // global_row_1

        // Bounds check
        w(s, "setp.lt.s32 \t%p10, %r775, %r1;");
        w(s, "setp.lt.s32 \t%p11, %r776, %r1;");

        // row * 256 (row stride for d=128)
        w(s, "shl.b32 \t%r777, %r775, 8;"); // row * 256
        w(s, "add.s32 \t%r778, %r777, %r772;"); // + (lane%4)*4
        w(s, "cvt.u64.u32 \t%rd50, %r778;");
        w(s, "add.s64 \t%rd51, %rd13, %rd50;"); // O global addr row 0

        w(s, "shl.b32 \t%r779, %r776, 8;");
        w(s, "add.s32 \t%r780, %r779, %r772;");
        w(s, "cvt.u64.u32 \t%rd52, %r780;");
        w(s, "add.s64 \t%rd53, %rd13, %rd52;"); // O global addr row 1
        blank(s);

        // Convert f32 to f16x2 and store (16 n-tiles)
        for n in 0..16u32 {
            let base = o_base_start + n * 4;
            let h0 = 780 + mt * 40 + n * 2;
            let h1 = h0 + 1;
            s.push_str(&format!(
                "\tcvt.rn.f16x2.f32 \t%r{h0}, %r{}, %r{};\n",
                base + 1,
                base
            ));
            s.push_str(&format!(
                "\tcvt.rn.f16x2.f32 \t%r{h1}, %r{}, %r{};\n",
                base + 3,
                base + 2
            ));
            let n_off = n * 16;
            s.push_str(&format!(
                "\t@%p10 st.global.b32 [ %rd51 + {n_off} ], %r{h0};\n"
            ));
            s.push_str(&format!(
                "\t@%p11 st.global.b32 [ %rd53 + {n_off} ], %r{h1};\n"
            ));
        }
        blank(s);
    }

    w(s, "ret;");
    s.push_str("}\n");
}

/// Load K or V tile (32 rows × 128 cols) into two-page KV buffer.
/// Uses kv_start from %r340. Writes to smem at base_offset (page 0)
/// and base_offset+4096 (page 1).
/// 128 threads × 16 bytes = 2048 per round. Need 4096/2048 = 2 rounds per page.
fn emit_cp_async_kv_two_page(s: &mut String, base_offset: u32, base_ptr: &str, _commit: bool) {
    for page in 0..2u32 {
        let page_smem_off = base_offset + page * 4096;
        let global_col_off = page * 128; // 64 cols * 2B
        for round in 0..2u32 {
            let row_off = round * 16;
            w(s, "add.s32 \t%r60, %r340, %r9;"); // kv_start + cp_row_in_chunk
            if row_off > 0 {
                w(s, &format!("add.s32 \t%r60, %r60, {};", row_off));
            }
            w(s, "setp.lt.s32 \t%p20, %r60, %r1;");
            w(s, "selp.b32 \t%r72, 16, 0, %p20;");
            // Global addr: base + row * 256 + col_bytes + global_col_off
            w(s, "shl.b32 \t%r62, %r60, 8;"); // row * 256
            w(s, "add.s32 \t%r63, %r62, %r11;"); // + col_bytes
            if global_col_off > 0 {
                w(s, &format!("add.s32 \t%r63, %r63, {};", global_col_off));
            }
            w(s, "cvt.u64.u32 \t%rd20, %r63;");
            w(s, &format!("add.s64 \t%rd21, {base_ptr}, %rd20;"));
            // Smem with B128 swizzle (128-byte row stride within page)
            let smem_row_off = row_off * 128;
            w(s, "shl.b32 \t%r64, %r9, 7;"); // cp_row * 128
            w(s, "add.s32 \t%r65, %r64, %r11;"); // + col_bytes
            if smem_row_off > 0 {
                w(s, &format!("add.s32 \t%r65, %r65, {};", smem_row_off));
            }
            w(s, "and.b32 \t%r66, %r65, 896;"); // 0x380
            w(s, "shr.u32 \t%r67, %r66, 3;");
            w(s, "xor.b32 \t%r68, %r65, %r67;");
            w(s, &format!("add.s32 \t%r69, %r68, {};", page_smem_off));
            w(s, "add.s32 \t%r69, %r69, %r12;");
            s.push_str(
                "\tcp.async.cg.shared.global [ %r69 + 0 ], [ %rd21 + 0 ], 0x10, %r72;\n",
            );
        }
    }
}

/// Same as above but with dynamic smem base offset in a register.
fn emit_cp_async_kv_two_page_dynamic(s: &mut String, smem_off_reg: &str, base_ptr: &str) {
    for page in 0..2u32 {
        let global_col_off = page * 128;
        let page_local_off = page * 4096;
        for round in 0..2u32 {
            let row_off = round * 16;
            w(s, "add.s32 \t%r60, %r340, %r9;");
            if row_off > 0 {
                w(s, &format!("add.s32 \t%r60, %r60, {};", row_off));
            }
            w(s, "setp.lt.s32 \t%p20, %r60, %r1;");
            w(s, "selp.b32 \t%r72, 16, 0, %p20;");
            w(s, "shl.b32 \t%r62, %r60, 8;");
            w(s, "add.s32 \t%r63, %r62, %r11;");
            if global_col_off > 0 {
                w(s, &format!("add.s32 \t%r63, %r63, {};", global_col_off));
            }
            w(s, "cvt.u64.u32 \t%rd20, %r63;");
            w(s, &format!("add.s64 \t%rd21, {base_ptr}, %rd20;"));
            let smem_row_off = row_off * 128;
            w(s, "shl.b32 \t%r64, %r9, 7;");
            w(s, "add.s32 \t%r65, %r64, %r11;");
            if smem_row_off > 0 {
                w(s, &format!("add.s32 \t%r65, %r65, {};", smem_row_off));
            }
            w(s, "and.b32 \t%r66, %r65, 896;");
            w(s, "shr.u32 \t%r67, %r66, 3;");
            w(s, "xor.b32 \t%r68, %r65, %r67;");
            if page_local_off > 0 {
                w(
                    s,
                    &format!("add.s32 \t%r68, %r68, {};", page_local_off),
                );
            }
            w(s, &format!("add.s32 \t%r69, %r68, {};", smem_off_reg));
            w(s, "add.s32 \t%r69, %r69, %r12;");
            s.push_str(
                "\tcp.async.cg.shared.global [ %r69 + 0 ], [ %rd21 + 0 ], 0x10, %r72;\n",
            );
        }
    }
}

/// Load V[32×128] into Q smem as staging for transpose.
/// Page 0: V[*][0:63] at smem 0, Page 1: V[*][64:127] at smem 4096
/// Uses kv_start from %r340.
fn emit_cp_async_v_staging(s: &mut String) {
    for page in 0..2u32 {
        let page_smem_off = page * 4096;
        let global_col_off = page * 128;
        for round in 0..2u32 {
            let row_off = round * 16;
            w(s, "add.s32 \t%r60, %r340, %r9;"); // kv_start + cp_row
            if row_off > 0 {
                w(s, &format!("add.s32 \t%r60, %r60, {};", row_off));
            }
            w(s, "setp.lt.s32 \t%p20, %r60, %r1;");
            w(s, "selp.b32 \t%r72, 16, 0, %p20;");
            // Global addr: V_base + row * 256 + col_bytes + global_col_off
            w(s, "shl.b32 \t%r62, %r60, 8;"); // row * 256
            w(s, "add.s32 \t%r63, %r62, %r11;"); // + col_bytes
            if global_col_off > 0 {
                w(s, &format!("add.s32 \t%r63, %r63, {};", global_col_off));
            }
            w(s, "cvt.u64.u32 \t%rd20, %r63;");
            w(s, "add.s64 \t%rd21, %rd12, %rd20;"); // V_base
            // Smem with B128 swizzle
            let smem_row_off = row_off * 128;
            w(s, "shl.b32 \t%r64, %r9, 7;");
            w(s, "add.s32 \t%r65, %r64, %r11;");
            if smem_row_off > 0 {
                w(s, &format!("add.s32 \t%r65, %r65, {};", smem_row_off));
            }
            w(s, "and.b32 \t%r66, %r65, 896;");
            w(s, "shr.u32 \t%r67, %r66, 3;");
            w(s, "xor.b32 \t%r68, %r65, %r67;");
            w(s, &format!("add.s32 \t%r69, %r68, {};", page_smem_off));
            w(s, "add.s32 \t%r69, %r69, %r12;");
            s.push_str(
                "\tcp.async.cg.shared.global [ %r69 + 0 ], [ %rd21 + 0 ], 0x10, %r72;\n",
            );
        }
    }
}

/// Transpose V from staging (Q smem) to V_t in current KV buffer.
/// V staging: page 0 at smem 0, page 1 at smem 4096 (each [32×64], 128B rows, B128 swizzle)
/// V_t output: [128×32] at current KV buffer (%r345), 64B rows, swizzle mask 0x1C0>>3
/// Thread tid transposes V column tid to V_t row tid.
fn emit_v_transpose(s: &mut String) {
    // %r49 = tid/64 (page), %r50 = page*4096, %r52 = col_byte_in_page, %r53 = tid*64

    // Read 32 f16 from V staging (one per V row), write 16 b32 to V_t
    for c_pair in 0..16u32 {
        let c0 = c_pair * 2;
        let c1 = c0 + 1;

        // Read V[c0][tid]: page_offset + c0*128 + col_byte_in_page, with B128 swizzle
        let c0_row_off = c0 * 128;
        if c0_row_off > 0 {
            w(s, &format!("add.s32 \t%r60, %r52, {};", c0_row_off));
        } else {
            w(s, "mov.b32 \t%r60, %r52;");
        }
        w(s, "and.b32 \t%r61, %r60, 896;"); // B128 swizzle
        w(s, "shr.u32 \t%r62, %r61, 3;");
        w(s, "xor.b32 \t%r63, %r60, %r62;");
        w(s, "add.s32 \t%r64, %r63, %r50;"); // + page offset
        w(s, "add.s32 \t%r64, %r64, %r12;"); // + smem base
        s.push_str(&format!("\tld.shared.u16 \t%h{}, [%r64];\n", c_pair * 2));

        // Read V[c1][tid]
        let c1_row_off = c1 * 128;
        w(s, &format!("add.s32 \t%r60, %r52, {};", c1_row_off));
        w(s, "and.b32 \t%r61, %r60, 896;");
        w(s, "shr.u32 \t%r62, %r61, 3;");
        w(s, "xor.b32 \t%r63, %r60, %r62;");
        w(s, "add.s32 \t%r64, %r63, %r50;");
        w(s, "add.s32 \t%r64, %r64, %r12;");
        s.push_str(&format!(
            "\tld.shared.u16 \t%h{}, [%r64];\n",
            c_pair * 2 + 1
        ));

        // Pack two f16 into b32: c0 in low half, c1 in high half
        s.push_str(&format!(
            "\tmov.b32 \t%r65, {{%h{}, %h{}}};\n",
            c_pair * 2,
            c_pair * 2 + 1
        ));

        // Write to V_t: row tid, col pair c_pair
        // V_t offset = tid*64 + c_pair*4
        let col_off = c_pair * 4;
        if col_off > 0 {
            w(s, &format!("add.s32 \t%r66, %r53, {};", col_off));
        } else {
            w(s, "mov.b32 \t%r66, %r53;");
        }
        // V_t swizzle: byte ^ ((byte & 0xC0) >> 2) — preserves 16B alignment
        w(s, "and.b32 \t%r67, %r66, 192;"); // 0xC0
        w(s, "shr.u32 \t%r68, %r67, 2;");
        w(s, "xor.b32 \t%r69, %r66, %r68;");
        w(s, "add.s32 \t%r69, %r69, %r345;"); // + current KV buf offset
        w(s, "add.s32 \t%r69, %r69, %r12;"); // + smem base
        s.push_str("\tst.shared.b32 [%r69], %r65;\n");
    }
}

fn w(s: &mut String, line: &str) {
    s.push('\t');
    s.push_str(line);
    s.push('\n');
}

fn blank(s: &mut String) {
    s.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_and_query_kernel() -> (String, u32) {
        let ptx = emit_flash_attn_fwd_hdim128();
        let mut regs = 0u32;
        for line in ptx.lines() {
            if line.contains(".reg .b32") && line.contains("%r<") {
                if let Some(n) = line.split("%r<").nth(1).and_then(|s| s.split('>').next()) {
                    regs = n.parse().unwrap_or(0);
                }
            }
        }
        (ptx, regs)
    }

    #[test]
    fn test_hdim128_ptx_is_valid_ascii() {
        let (ptx, _) = load_and_query_kernel();
        for (i, b) in ptx.bytes().enumerate() {
            assert!(b < 128, "Non-ASCII byte 0x{:02x} at position {}", b, i);
        }
    }

    #[test]
    fn test_hdim128_has_single_entry_point() {
        let (ptx, _) = load_and_query_kernel();
        let entries: Vec<_> = ptx
            .lines()
            .filter(|l| {
                l.contains(".visible .entry") || l.contains(".entry flash_attn_fwd_hdim128")
            })
            .collect();
        assert_eq!(
            entries.len(),
            1,
            "Must have exactly one entry point, found: {:?}",
            entries
        );
    }

    #[test]
    fn test_hdim128_has_correct_thread_count() {
        let (ptx, _) = load_and_query_kernel();
        assert!(
            ptx.contains(".reqntid 128"),
            "Must require 128 threads per block"
        );
    }

    #[test]
    fn test_hdim128_has_two_mma_phases() {
        let (ptx, _) = load_and_query_kernel();
        let mma_count = ptx.lines().filter(|l| l.contains("mma.sync")).count();
        // Q@K^T: 64 MMA + P@V: 64 MMA = 128 total (per loop body)
        assert!(
            mma_count >= 64,
            "Need at least 64 MMA for Q@K^T + P@V, got {}",
            mma_count
        );
    }

    #[test]
    fn test_hdim128_has_online_softmax() {
        let (ptx, _) = load_and_query_kernel();
        let ex2_count = ptx.lines().filter(|l| l.contains("ex2.approx")).count();
        assert!(ex2_count > 0, "Must have ex2.approx for online softmax");
        let shfl_count = ptx.lines().filter(|l| l.contains("shfl")).count();
        assert!(shfl_count > 0, "Must have shuffle for row-max reduction");
    }

    #[test]
    fn test_hdim128_has_kv_loop() {
        let (ptx, _) = load_and_query_kernel();
        assert!(
            ptx.contains("$L_KV_LOOP"),
            "Must have a KV-loop label"
        );
    }

    #[test]
    fn test_hdim128_uses_cp_async() {
        let (ptx, _) = load_and_query_kernel();
        let cp_count = ptx.lines().filter(|l| l.contains("cp.async.cg")).count();
        // Q: 16, K first: 4, next K: 4, V staging: 4 = at least 20
        assert!(
            cp_count >= 8,
            "Need at least 8 cp.async loads, got {}",
            cp_count
        );
    }

    #[test]
    fn test_hdim128_uses_ldmatrix() {
        let (ptx, _) = load_and_query_kernel();
        let ldm_count = ptx.lines().filter(|l| l.contains("ldmatrix")).count();
        // Q: 16, K: 16, V_t: 16 = 48
        assert!(
            ldm_count >= 16,
            "Need at least 16 ldmatrix, got {}",
            ldm_count
        );
    }

    #[test]
    fn test_hdim128_has_v_transpose() {
        let (ptx, _) = load_and_query_kernel();
        // V transpose uses ld.shared.u16 and st.shared.b32
        let ld_shared = ptx.lines().filter(|l| l.contains("ld.shared.u16")).count();
        let st_shared = ptx
            .lines()
            .filter(|l| l.contains("st.shared.b32"))
            .count();
        assert!(
            ld_shared >= 16,
            "Need ld.shared.u16 for V transpose, got {}",
            ld_shared
        );
        assert!(
            st_shared >= 8,
            "Need st.shared.b32 for V transpose, got {}",
            st_shared
        );
    }

    #[test]
    fn test_hdim128_register_budget() {
        let (_, regs) = load_and_query_kernel();
        assert!(
            regs <= 860,
            "Virtual b32 regs should be <= 860, got {}",
            regs
        );
        assert!(
            regs >= 200,
            "Need at least 200 b32 regs for flash attn d=128, got {}",
            regs
        );
    }

    #[test]
    fn test_hdim128_shared_memory_declaration() {
        let (ptx, _) = load_and_query_kernel();
        assert!(
            ptx.contains("global_smem") || ptx.contains(".shared"),
            "Must declare shared memory"
        );
    }

    #[test]
    fn test_hdim128_score_matrix_is_f32() {
        let (ptx, _) = load_and_query_kernel();
        assert!(
            ptx.contains("f32.f16.f16.f32"),
            "MMA must use f32 accumulation for scores"
        );
    }

    #[test]
    fn test_hdim128_has_output_normalization() {
        let (ptx, _) = load_and_query_kernel();
        let has_div = ptx.contains("rcp.approx")
            || ptx.contains("div.approx")
            || ptx.contains("div.full");
        assert!(has_div, "Must normalize output O by 1/l_i");
    }

    #[test]
    fn test_hdim128_stores_output() {
        let (ptx, _) = load_and_query_kernel();
        let st_count = ptx.lines().filter(|l| l.contains("st.global")).count();
        // 2 m-tiles × 16 n-tiles × 2 row groups = 64 stores
        assert!(st_count >= 32, "Must store output to global memory, got {}", st_count);
    }

    #[test]
    fn test_hdim128_no_spills() {
        let (ptx, _) = load_and_query_kernel();
        let local_count = ptx
            .lines()
            .filter(|l| l.contains("st.local") || l.contains("ld.local"))
            .count();
        assert_eq!(local_count, 0, "Must have no local memory spills");
    }

    #[test]
    fn test_hdim128_ptx_size_reasonable() {
        let (ptx, _) = load_and_query_kernel();
        let line_count = ptx.lines().count();
        assert!(
            line_count >= 400,
            "Flash attn d=128 should be at least 400 lines, got {}",
            line_count
        );
        assert!(
            line_count <= 10000,
            "Flash attn d=128 should be at most 10000 lines, got {}",
            line_count
        );
    }

    #[test]
    fn test_hdim128_instruction_counts() {
        let (ptx, _) = load_and_query_kernel();
        let mma = ptx.lines().filter(|l| l.contains("mma.sync")).count();
        let ldm = ptx.lines().filter(|l| l.contains("ldmatrix")).count();
        let cpa = ptx.lines().filter(|l| l.contains("cp.async.cg")).count();
        let ex2 = ptx.lines().filter(|l| l.contains("ex2.approx")).count();

        eprintln!(
            "Flash attn d=128 instruction counts: mma={}, ldmatrix={}, cp.async={}, ex2={}",
            mma, ldm, cpa, ex2
        );

        // Q@K^T: 64 + P@V: 64 = 128 MMA
        assert!(mma >= 64, "Need at least 64 MMA, got {}", mma);
        // Q: 16 + K: 16 + V_t: 16 = 48
        assert!(ldm >= 32, "Need at least 32 ldmatrix, got {}", ldm);
        assert!(cpa >= 8, "Need at least 8 cp.async, got {}", cpa);
        assert!(ex2 >= 4, "Need at least 4 ex2 for softmax, got {}", ex2);
    }

    #[test]
    fn test_hdim128_has_two_page_q_load() {
        let (ptx, _) = load_and_query_kernel();
        // Q load uses 16 rounds of cp.async (8 per page × 2 pages)
        let kv_loop_pos = ptx.find("KV_LOOP").unwrap_or(ptx.len());
        let prologue = &ptx[..kv_loop_pos];
        let cp_before = prologue
            .lines()
            .filter(|l| l.contains("cp.async.cg"))
            .count();
        // Q: 16 rounds + K first: 4 rounds = 20
        assert!(
            cp_before >= 16,
            "Need at least 16 cp.async in prologue (Q + first K), got {}",
            cp_before
        );
    }

    #[test]
    fn test_hdim128_block_n_32() {
        let (ptx, _) = load_and_query_kernel();
        // KV loop advances by 32 (BLOCK_N=32)
        assert!(
            ptx.contains("add.s32 \t%r340, %r340, 32"),
            "KV loop must advance by 32 (BLOCK_N=32)"
        );
    }

    #[test]
    fn test_hdim128_scale_applied() {
        let (ptx, _) = load_and_query_kernel();
        assert!(
            ptx.contains("param_scale"),
            "Must accept a scale parameter"
        );
    }

    #[test]
    fn test_hdim128_output_is_f16() {
        let (ptx, _) = load_and_query_kernel();
        let has_f16_store =
            ptx.contains("cvt.rn.f16x2.f32") && ptx.contains("st.global");
        assert!(has_f16_store, "Must store output as f16");
    }

    #[test]
    fn test_hdim128_v_transpose_uses_custom_swizzle() {
        let (ptx, _) = load_and_query_kernel();
        // V_t swizzle mask is 0xC0 = 192 (different from B128's 896)
        assert!(
            ptx.contains("and.b32") && ptx.contains("192"),
            "V_t must use custom swizzle mask 0xC0 (192)"
        );
    }

    #[test]
    fn test_hdim128_q_loads_8_k_iters() {
        let (ptx, _) = load_and_query_kernel();
        // Q ldmatrix: 2 m-tiles × 8 k-iters = 16 non-trans ldmatrix calls
        let non_trans_ldm = ptx
            .lines()
            .filter(|l| l.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"))
            .count();
        assert_eq!(
            non_trans_ldm, 16,
            "Q ldmatrix should be exactly 16 (2 m-tiles × 8 k-iters), got {}",
            non_trans_ldm
        );
    }
}
