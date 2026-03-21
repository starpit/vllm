// Flash Attention Forward kernel — hand-written PTX
//
// BLOCK_M=128, BLOCK_N=64, HEAD_DIM=64, 128 threads (4 warps)
// Matches FA2's exact configuration for maximum register availability
// and ptxas scheduling freedom.
//
// No causal mask, no paged KV cache, contiguous Q/K/V/O layout
//
// Q,K,V,O: [batch*heads, seq, 64] contiguous f16 (stride_seq = 64)
//
// Grid: (ceil(seq_q / 128), batch * heads, 1)
//
// Smem layout (with B128 swizzle):
//   Q region:  0..16383    (128 rows × 64 cols × 2B = 16384)
//   KV buf 0:  16384..24575 (64 × 64 × 2B = 8192)
//   KV buf 1:  24576..32767 (64 × 64 × 2B = 8192)
//   Total: 32768 bytes
//
// MMA config: m16n8k16
//   Each of 4 warps handles 32 rows (2 m-tiles of 16 rows each)
//   Q@K^T: S[128×64] — 4 warps × 2 m-tiles × 8 n-tiles × 4 k-iters = 256 MMA
//   P@V:   O[128×64] — 4 warps × 2 m-tiles × 8 n-tiles × 4 k-iters = 256 MMA
//
// Key optimizations (FA2-style):
//   1. 4 warps (128 threads) = more regs/thread = room for aggressive unrolling
//   2. B128 XOR swizzle on shared memory to eliminate bank conflicts
//   3. K/V double-buffering: cp.async for next K while computing current Q@K^T
//   4. P stays in registers (no smem round-trip)
//   5. V load overlapped with softmax computation

pub fn emit_flash_attn_fwd() -> String {
    let mut s = String::with_capacity(400 * 1024);
    emit_kernel(&mut s);
    s
}

pub fn emit_flash_attn_fwd_causal() -> String {
    let mut s = String::with_capacity(500 * 1024);
    emit_kernel_causal(&mut s);
    s
}

pub const SMEM_BYTES: u32 = 32768;

fn emit_kernel(s: &mut String) {
    // ─── Header ───
    s.push_str(
        r#".version 8.7
.target sm_89
.address_size 64

.extern .shared .align 16 .b8 global_smem[];

.visible .entry flash_attn_fwd(
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
	.reg .b32 	%r<800>;
	.reg .b64 	%rd<80>;

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
    // 128 threads, each loads 16 bytes.
    // cp_row_in_chunk = tid/8 (0..15), cp_col_idx = tid%8, col_bytes = idx*16
    // For Q (128 rows): 128*128 bytes / (128*16) = 8 rounds of 16 rows each
    // For KV (64 rows): 64*128 bytes / (128*16) = 4 rounds of 16 rows each
    w(s, "shr.u32 \t%r9, %r6, 3;"); // cp_row_in_chunk = tid/8 (0..15)
    w(s, "and.b32 \t%r10, %r6, 7;"); // cp_col_idx = tid%8
    w(s, "shl.b32 \t%r11, %r10, 4;"); // col_bytes = cp_col_idx * 16
    w(s, "mov.b32 \t%r12, global_smem;");
    blank(s);

    // ─── Load Q → smem (offset 0) with B128 swizzle ───
    // 8 rounds: each loads 16 rows (128 threads / 8 cols_per_row = 16 rows)
    for round in 0..8u32 {
        let row_base = round * 16;
        // Global address: Q_base + (block_m_start + row_base + cp_row) * 128 + col_bytes
        w(s, "add.s32 \t%r60, %r7, %r9;"); // block_m_start + cp_row_in_chunk
        if row_base > 0 {
            w(s, &format!("add.s32 \t%r60, %r60, {};", row_base));
        }
        // Bounds check
        w(s, "setp.lt.s32 \t%p20, %r60, %r1;");
        w(s, "selp.b32 \t%r72, 16, 0, %p20;");
        // Global byte offset
        w(s, "shl.b32 \t%r62, %r60, 7;"); // row * 128
        w(s, "add.s32 \t%r63, %r62, %r11;"); // + col_bytes
        w(s, "cvt.u64.u32 \t%rd20, %r63;");
        w(s, "add.s64 \t%rd21, %rd10, %rd20;"); // global addr

        // Smem address with swizzle
        let smem_row_base = row_base * 128; // byte offset for this round's rows
        w(s, "shl.b32 \t%r64, %r9, 7;"); // cp_row_in_chunk * 128
        w(s, "add.s32 \t%r65, %r64, %r11;"); // + col_bytes
        if smem_row_base > 0 {
            w(s, &format!("add.s32 \t%r65, %r65, {};", smem_row_base));
        }
        // Apply B128 swizzle: swizzled = byte ^ ((byte & 0x380) >> 3)
        w(s, "and.b32 \t%r66, %r65, 896;"); // 0x380
        w(s, "shr.u32 \t%r67, %r66, 3;");
        w(s, "xor.b32 \t%r68, %r65, %r67;");
        w(s, "add.s32 \t%r69, %r68, %r12;"); // + smem base
        s.push_str("\tcp.async.cg.shared.global [ %r69 + 0 ], [ %rd21 + 0 ], 0x10, %r72;\n");
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
    // MMA m16n8k16 A operand: ldmatrix.x4 loads 4 matrices of 8x8 elements.
    // Thread lane maps to row_in_frag = lane % 8, frag_id = lane / 8.
    // frag_id: bits [1]=row_half (0,1 → +0,+8 rows), [0]=col_half (0,1 → +0,+16 bytes)
    w(s, "and.b32 \t%r32, %r31, 7;"); // row_in_frag = lane % 8
    w(s, "shr.u32 \t%r33, %r31, 3;"); // frag_id = lane / 8 (0..3)
    w(s, "shr.u32 \t%r34, %r33, 1;"); // row_half (0 or 1)
    w(s, "and.b32 \t%r35, %r33, 1;"); // col_half (0 or 1)

    w(s, "shl.b32 \t%r36, %r30, 5;"); // warp_id * 32
    w(s, "shl.b32 \t%r37, %r34, 3;"); // row_half * 8
    w(s, "shl.b32 \t%r40, %r35, 4;"); // col_half * 16 bytes
    blank(s);

    // Load Q fragments for both m-tiles (stay live for entire KV-loop):
    // m-tile 0: Q_frag %r100..%r115 (4 k-iters × 4 regs)
    // m-tile 1: Q_frag %r116..%r131 (4 k-iters × 4 regs)
    for mt in 0..2u32 {
        let mt_row_off = mt * 16; // 0 or 16 rows within warp's 32-row block
        // Q smem row for this m-tile: warp_id*32 + mt*16 + row_half*8 + row_in_frag
        w(s, &format!("add.s32 \t%r38, %r36, %r37;")); // warp_id*32 + row_half*8
        w(s, &format!("add.s32 \t%r39, %r38, %r32;")); // + row_in_frag
        if mt_row_off > 0 {
            w(s, &format!("add.s32 \t%r39, %r39, {};", mt_row_off));
        }
        // Q linear byte offset: row * 128 + col_half * 16
        w(s, "shl.b32 \t%r41, %r39, 7;"); // row * 128 bytes
        w(s, "add.s32 \t%r42, %r41, %r40;"); // + col_half * 16

        for ki in 0..4u32 {
            let base = 100 + mt * 16 + ki * 4;
            let k_byte_off = ki * 32;
            w(s, &format!("add.s32 \t%r70, %r42, {};", k_byte_off));
            w(s, "and.b32 \t%r71, %r70, 896;");
            w(s, "shr.u32 \t%r72, %r71, 3;");
            w(s, "xor.b32 \t%r73, %r70, %r72;");
            w(s, "add.s32 \t%r74, %r73, %r12;"); // + smem base
            s.push_str(&format!(
                "\tldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r74];\n",
                base, base + 1, base + 2, base + 3
            ));
        }
    }
    blank(s);

    // ─── K/V ldmatrix addressing ───
    // For K (B operand, transposed): ldmatrix.trans.x4
    // B operand thread mapping for ldmatrix.trans:
    //   row_in_frag = lane % 8, frag_id = lane / 8
    //   base linear_byte = row_in_frag * 128 + frag_id * 16
    w(s, "shl.b32 \t%r44, %r33, 4;"); // frag_id * 16 bytes
    w(s, "shl.b32 \t%r45, %r32, 7;"); // (lane%8) * 128
    w(s, "add.s32 \t%r46, %r45, %r44;"); // base linear byte within KV tile
    blank(s);

    // ─── Initialize accumulators ───
    // O accum: 2 m-tiles × 8 n-tiles × 4 regs = 64 regs per warp
    // m-tile 0: %r200..%r231, m-tile 1: %r560..%r591
    w(s, "mov.b32 \t%r199, 0;");
    for i in 200..232u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r199;\n"));
    }
    for i in 560..592u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r199;\n"));
    }
    // m_i: 4 values (2 per m-tile, each m-tile has rows 0..7 and 8..15)
    // m-tile 0: %r232, %r233; m-tile 1: %r592, %r593
    w(s, "mov.b32 \t%r232, 0xFF800000;"); // -inf
    w(s, "mov.b32 \t%r233, 0xFF800000;");
    w(s, "mov.b32 \t%r592, 0xFF800000;");
    w(s, "mov.b32 \t%r593, 0xFF800000;");
    // l_i: m-tile 0: %r234, %r235; m-tile 1: %r594, %r595
    w(s, "mov.b32 \t%r234, 0x00000000;");
    w(s, "mov.b32 \t%r235, 0x00000000;");
    w(s, "mov.b32 \t%r594, 0x00000000;");
    w(s, "mov.b32 \t%r595, 0x00000000;");
    blank(s);

    // ─── KV-loop with double-buffered K/V ───
    w(s, "mov.b32 \t%r236, 0;"); // kv_start
    w(s, "setp.lt.s32 \t%p1, %r236, %r1;");
    w(s, "@!%p1 bra \t$L_EPILOGUE;");
    blank(s);

    // Load K[0] into KV buf 0 (smem offset 16384)
    emit_cp_async_kv_swizzled(s, 16384, "%rd11", false);
    w(s, "cp.async.commit_group;");
    w(s, "cp.async.wait_group \t0;");
    w(s, "bar.sync \t0;");
    blank(s);

    s.push_str("$L_KV_LOOP:\n");
    blank(s);

    // ─── Issue cp.async for NEXT K block into alternate buffer ───
    w(s, "add.s32 \t%r237, %r236, 64;"); // next_kv_start
    w(s, "setp.lt.s32 \t%p2, %r237, %r1;"); // has_next_k?

    // Compute current buffer offset: buf = (kv_start >> 6) & 1
    w(s, "shr.u32 \t%r238, %r236, 6;"); // kv_start / 64
    w(s, "and.b32 \t%r239, %r238, 1;"); // buf index (0 or 1)
    w(s, "shl.b32 \t%r240, %r239, 13;"); // buf * 8192
    w(s, "add.s32 \t%r241, %r240, 16384;"); // current KV smem offset

    // Next buffer
    w(s, "xor.b32 \t%r242, %r239, 1;"); // next buf index
    w(s, "shl.b32 \t%r243, %r242, 13;"); // next_buf * 8192
    w(s, "add.s32 \t%r244, %r243, 16384;"); // next KV smem offset

    // Pipeline: start loading next K while we compute with current K
    w(s, "@!%p2 bra \t$L_SKIP_NEXT_K;");
    w(s, "mov.b32 \t%r250, %r236;"); // save current kv_start
    w(s, "mov.b32 \t%r236, %r237;"); // temporarily set kv_start to next
    emit_cp_async_kv_swizzled_dynamic(s, "%r244", "%rd11");
    w(s, "cp.async.commit_group;");
    w(s, "mov.b32 \t%r236, %r250;"); // restore kv_start
    s.push_str("$L_SKIP_NEXT_K:\n");
    blank(s);

    // ─── ldmatrix K from current buffer (B operand for Q@K^T) ───
    // K_frag at %r300..%r363 (8 n-tiles × 2 k-pairs × 4 regs = 64 regs)
    for n in 0..8u32 {
        for kp in 0..2u32 {
            let base = 300 + (n * 2 + kp) * 4;
            let n_off = n * 1024;
            let kp_off = kp * 64;
            let total_off = n_off + kp_off;
            w(s, &format!("add.s32 \t%r70, %r46, {};", total_off));
            w(s, "and.b32 \t%r71, %r70, 896;");
            w(s, "shr.u32 \t%r72, %r71, 3;");
            w(s, "xor.b32 \t%r73, %r70, %r72;");
            w(s, "add.s32 \t%r74, %r73, %r241;"); // + current buf offset
            w(s, "add.s32 \t%r74, %r74, %r12;"); // + smem base
            s.push_str(&format!(
                "\tldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r74];\n",
                base, base + 1, base + 2, base + 3
            ));
        }
    }
    blank(s);

    // ─── MMA: S = Q @ K^T ───
    // S accum for m-tile 0: %r370..%r401 (8 n-tiles × 4 regs = 32 regs)
    // S accum for m-tile 1: %r600..%r631 (8 n-tiles × 4 regs = 32 regs)
    w(s, "mov.b32 \t%r369, 0;");
    for i in 370..402u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r369;\n"));
    }
    for i in 600..632u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r369;\n"));
    }
    blank(s);

    // 2 m-tiles × 4 k-iters × 8 n-tiles = 64 MMA calls
    for mt in 0..2u32 {
        let q_base_start = 100 + mt * 16; // Q frags for this m-tile
        let s_accum_start = if mt == 0 { 370 } else { 600 };
        for ki in 0..4u32 {
            let q_base = q_base_start + ki * 4;
            for n in 0..8u32 {
                let s_base = s_accum_start + n * 4;
                let k_pair = ki / 2;
                let sub_ki = ki % 2;
                let k_frag_base = 300 + (n * 2 + k_pair) * 4;
                let b0 = k_frag_base + sub_ki * 2;
                let b1 = b0 + 1;
                s.push_str(&format!(
                    "\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                     {{ %r{}, %r{}, %r{}, %r{} }}, \
                     {{ %r{}, %r{}, %r{}, %r{} }}, \
                     {{ %r{}, %r{} }}, \
                     {{ %r{}, %r{}, %r{}, %r{} }};\n",
                    s_base, s_base + 1, s_base + 2, s_base + 3,
                    q_base, q_base + 1, q_base + 2, q_base + 3,
                    b0, b1,
                    s_base, s_base + 1, s_base + 2, s_base + 3
                ));
            }
        }
    }
    blank(s);

    // Wait for next K load to complete (if it was issued)
    w(s, "cp.async.wait_group \t0;");
    w(s, "bar.sync \t0;");
    blank(s);

    // ─── Online softmax (for both m-tiles) ───
    // Process m-tile 0 (S in %r370..%r401) and m-tile 1 (S in %r600..%r631)

    // Step 1: Scale S
    for i in 370..402u32 {
        s.push_str(&format!("\tmul.f32 \t%r{i}, %r{i}, %r2;\n"));
    }
    for i in 600..632u32 {
        s.push_str(&format!("\tmul.f32 \t%r{i}, %r{i}, %r2;\n"));
    }
    blank(s);

    // Step 2: Row max — for each m-tile, 2 row groups (rows 0..7, 8..15)
    // m-tile 0: row_max in %r402, %r403
    w(s, "mov.b32 \t%r402, 0xFF800000;");
    w(s, "mov.b32 \t%r403, 0xFF800000;");
    for n in 0..8u32 {
        let base = 370 + n * 4;
        s.push_str(&format!("\tmax.f32 \t%r402, %r402, %r{};\n", base));
        s.push_str(&format!("\tmax.f32 \t%r402, %r402, %r{};\n", base + 1));
        s.push_str(&format!("\tmax.f32 \t%r403, %r403, %r{};\n", base + 2));
        s.push_str(&format!("\tmax.f32 \t%r403, %r403, %r{};\n", base + 3));
    }
    w(s, "shfl.sync.bfly.b32 \t%r404, %r402, 2, 31, -1;");
    w(s, "max.f32 \t%r402, %r402, %r404;");
    w(s, "shfl.sync.bfly.b32 \t%r405, %r402, 1, 31, -1;");
    w(s, "max.f32 \t%r402, %r402, %r405;");
    w(s, "shfl.sync.bfly.b32 \t%r406, %r403, 2, 31, -1;");
    w(s, "max.f32 \t%r403, %r403, %r406;");
    w(s, "shfl.sync.bfly.b32 \t%r407, %r403, 1, 31, -1;");
    w(s, "max.f32 \t%r403, %r403, %r407;");
    blank(s);

    // m-tile 1: row_max in %r632, %r633
    w(s, "mov.b32 \t%r632, 0xFF800000;");
    w(s, "mov.b32 \t%r633, 0xFF800000;");
    for n in 0..8u32 {
        let base = 600 + n * 4;
        s.push_str(&format!("\tmax.f32 \t%r632, %r632, %r{};\n", base));
        s.push_str(&format!("\tmax.f32 \t%r632, %r632, %r{};\n", base + 1));
        s.push_str(&format!("\tmax.f32 \t%r633, %r633, %r{};\n", base + 2));
        s.push_str(&format!("\tmax.f32 \t%r633, %r633, %r{};\n", base + 3));
    }
    w(s, "shfl.sync.bfly.b32 \t%r634, %r632, 2, 31, -1;");
    w(s, "max.f32 \t%r632, %r632, %r634;");
    w(s, "shfl.sync.bfly.b32 \t%r635, %r632, 1, 31, -1;");
    w(s, "max.f32 \t%r632, %r632, %r635;");
    w(s, "shfl.sync.bfly.b32 \t%r636, %r633, 2, 31, -1;");
    w(s, "max.f32 \t%r633, %r633, %r636;");
    w(s, "shfl.sync.bfly.b32 \t%r637, %r633, 1, 31, -1;");
    w(s, "max.f32 \t%r633, %r633, %r637;");
    blank(s);

    // Step 3: m_new = max(m_old, row_max)
    // m-tile 0
    w(s, "max.f32 \t%r408, %r232, %r402;");
    w(s, "max.f32 \t%r409, %r233, %r403;");
    // m-tile 1
    w(s, "max.f32 \t%r638, %r592, %r632;");
    w(s, "max.f32 \t%r639, %r593, %r633;");
    blank(s);

    // Step 4: alpha = exp2(m_old - m_new)
    // m-tile 0
    w(s, "sub.f32 \t%r410, %r232, %r408;");
    w(s, "sub.f32 \t%r411, %r233, %r409;");
    w(s, "ex2.approx.ftz.f32 \t%r412, %r410;");
    w(s, "ex2.approx.ftz.f32 \t%r413, %r411;");
    // m-tile 1
    w(s, "sub.f32 \t%r640, %r592, %r638;");
    w(s, "sub.f32 \t%r641, %r593, %r639;");
    w(s, "ex2.approx.ftz.f32 \t%r642, %r640;");
    w(s, "ex2.approx.ftz.f32 \t%r643, %r641;");
    blank(s);

    // Step 5: P = exp2(S_scaled - m_new)
    // m-tile 0
    for n in 0..8u32 {
        let base = 370 + n * 4;
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r408;\n", b = base));
        s.push_str(&format!("\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n", b = base));
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r408;\n", b = base + 1));
        s.push_str(&format!("\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n", b = base + 1));
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r409;\n", b = base + 2));
        s.push_str(&format!("\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n", b = base + 2));
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r409;\n", b = base + 3));
        s.push_str(&format!("\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n", b = base + 3));
    }
    // m-tile 1
    for n in 0..8u32 {
        let base = 600 + n * 4;
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r638;\n", b = base));
        s.push_str(&format!("\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n", b = base));
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r638;\n", b = base + 1));
        s.push_str(&format!("\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n", b = base + 1));
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r639;\n", b = base + 2));
        s.push_str(&format!("\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n", b = base + 2));
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r639;\n", b = base + 3));
        s.push_str(&format!("\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n", b = base + 3));
    }
    blank(s);

    // Step 6: Row sum of P
    // m-tile 0
    w(s, "mov.b32 \t%r414, 0x00000000;");
    w(s, "mov.b32 \t%r415, 0x00000000;");
    for n in 0..8u32 {
        let base = 370 + n * 4;
        s.push_str(&format!("\tadd.f32 \t%r414, %r414, %r{};\n", base));
        s.push_str(&format!("\tadd.f32 \t%r414, %r414, %r{};\n", base + 1));
        s.push_str(&format!("\tadd.f32 \t%r415, %r415, %r{};\n", base + 2));
        s.push_str(&format!("\tadd.f32 \t%r415, %r415, %r{};\n", base + 3));
    }
    w(s, "shfl.sync.bfly.b32 \t%r416, %r414, 2, 31, -1;");
    w(s, "add.f32 \t%r414, %r414, %r416;");
    w(s, "shfl.sync.bfly.b32 \t%r417, %r414, 1, 31, -1;");
    w(s, "add.f32 \t%r414, %r414, %r417;");
    w(s, "shfl.sync.bfly.b32 \t%r418, %r415, 2, 31, -1;");
    w(s, "add.f32 \t%r415, %r415, %r418;");
    w(s, "shfl.sync.bfly.b32 \t%r419, %r415, 1, 31, -1;");
    w(s, "add.f32 \t%r415, %r415, %r419;");
    // m-tile 1
    w(s, "mov.b32 \t%r644, 0x00000000;");
    w(s, "mov.b32 \t%r645, 0x00000000;");
    for n in 0..8u32 {
        let base = 600 + n * 4;
        s.push_str(&format!("\tadd.f32 \t%r644, %r644, %r{};\n", base));
        s.push_str(&format!("\tadd.f32 \t%r644, %r644, %r{};\n", base + 1));
        s.push_str(&format!("\tadd.f32 \t%r645, %r645, %r{};\n", base + 2));
        s.push_str(&format!("\tadd.f32 \t%r645, %r645, %r{};\n", base + 3));
    }
    w(s, "shfl.sync.bfly.b32 \t%r646, %r644, 2, 31, -1;");
    w(s, "add.f32 \t%r644, %r644, %r646;");
    w(s, "shfl.sync.bfly.b32 \t%r647, %r644, 1, 31, -1;");
    w(s, "add.f32 \t%r644, %r644, %r647;");
    w(s, "shfl.sync.bfly.b32 \t%r648, %r645, 2, 31, -1;");
    w(s, "add.f32 \t%r645, %r645, %r648;");
    w(s, "shfl.sync.bfly.b32 \t%r649, %r645, 1, 31, -1;");
    w(s, "add.f32 \t%r645, %r645, %r649;");
    blank(s);

    // Step 7: Rescale O accumulators: O *= alpha
    // m-tile 0
    for n in 0..8u32 {
        let base = 200 + n * 4;
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r412;\n", b = base));
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r412;\n", b = base + 1));
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r413;\n", b = base + 2));
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r413;\n", b = base + 3));
    }
    // m-tile 1
    for n in 0..8u32 {
        let base = 560 + n * 4;
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r642;\n", b = base));
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r642;\n", b = base + 1));
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r643;\n", b = base + 2));
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r643;\n", b = base + 3));
    }
    blank(s);

    // Step 8: Update l_i, m_i
    // m-tile 0
    w(s, "fma.rn.f32 \t%r234, %r234, %r412, %r414;");
    w(s, "fma.rn.f32 \t%r235, %r235, %r413, %r415;");
    w(s, "mov.b32 \t%r232, %r408;");
    w(s, "mov.b32 \t%r233, %r409;");
    // m-tile 1
    w(s, "fma.rn.f32 \t%r594, %r594, %r642, %r644;");
    w(s, "fma.rn.f32 \t%r595, %r595, %r643, %r645;");
    w(s, "mov.b32 \t%r592, %r638;");
    w(s, "mov.b32 \t%r593, %r639;");
    blank(s);

    // ─── Convert P to f16x2 in registers for P@V MMA ───
    // m-tile 0: P_frag %r460..%r475 (4 k-iters × 4 regs)
    for ki in 0..4u32 {
        let n_lo = ki * 2;
        let n_hi = n_lo + 1;
        let p_base = 460 + ki * 4;
        let s_lo = 370 + n_lo * 4;
        let s_hi = 370 + n_hi * 4;

        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base, s_lo + 1, s_lo
        ));
        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base + 1, s_hi + 1, s_hi
        ));
        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base + 2, s_lo + 3, s_lo + 2
        ));
        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base + 3, s_hi + 3, s_hi + 2
        ));
    }
    // m-tile 1: P_frag %r650..%r665 (4 k-iters × 4 regs)
    for ki in 0..4u32 {
        let n_lo = ki * 2;
        let n_hi = n_lo + 1;
        let p_base = 650 + ki * 4;
        let s_lo = 600 + n_lo * 4;
        let s_hi = 600 + n_hi * 4;

        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base, s_lo + 1, s_lo
        ));
        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base + 1, s_hi + 1, s_hi
        ));
        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base + 2, s_lo + 3, s_lo + 2
        ));
        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base + 3, s_hi + 3, s_hi + 2
        ));
    }
    blank(s);

    // ─── Load V block → current KV buffer (reusing K's space) ───
    emit_cp_async_kv_swizzled_dynamic(s, "%r241", "%rd12");
    w(s, "cp.async.commit_group;");
    w(s, "cp.async.wait_group \t0;");
    w(s, "bar.sync \t0;");
    blank(s);

    // ─── ldmatrix V (B operand for P@V) from current buffer ───
    // V_frag: %r480..%r543 (8 n-tiles × 2 k-pairs × 4 regs = 64 regs)
    for n in 0..8u32 {
        for kp in 0..2u32 {
            let base = 480 + (n * 2 + kp) * 4;
            let n_off = n * 1024;
            let kp_off = kp * 64;
            let total_off = n_off + kp_off;
            w(s, &format!("add.s32 \t%r70, %r46, {};", total_off));
            w(s, "and.b32 \t%r71, %r70, 896;");
            w(s, "shr.u32 \t%r72, %r71, 3;");
            w(s, "xor.b32 \t%r73, %r70, %r72;");
            w(s, "add.s32 \t%r74, %r73, %r241;"); // current buf offset
            w(s, "add.s32 \t%r74, %r74, %r12;"); // + smem base
            s.push_str(&format!(
                "\tldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r74];\n",
                base, base + 1, base + 2, base + 3
            ));
        }
    }
    blank(s);

    // ─── MMA: O += P @ V (both m-tiles) ───
    for mt in 0..2u32 {
        let p_base_start = if mt == 0 { 460 } else { 650 };
        let o_base_start = if mt == 0 { 200 } else { 560 };
        for ki in 0..4u32 {
            let p_base = p_base_start + ki * 4;
            for n in 0..8u32 {
                let o_base = o_base_start + n * 4;
                let k_pair = ki / 2;
                let sub_ki = ki % 2;
                let v_frag_base = 480 + (n * 2 + k_pair) * 4;
                let b0 = v_frag_base + sub_ki * 2;
                let b1 = b0 + 1;
                s.push_str(&format!(
                    "\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                     {{ %r{}, %r{}, %r{}, %r{} }}, \
                     {{ %r{}, %r{}, %r{}, %r{} }}, \
                     {{ %r{}, %r{} }}, \
                     {{ %r{}, %r{}, %r{}, %r{} }};\n",
                    o_base, o_base + 1, o_base + 2, o_base + 3,
                    p_base, p_base + 1, p_base + 2, p_base + 3,
                    b0, b1,
                    o_base, o_base + 1, o_base + 2, o_base + 3
                ));
            }
        }
    }
    blank(s);

    // ─── Loop advance ───
    w(s, "add.s32 \t%r236, %r236, 64;");
    w(s, "setp.lt.s32 \t%p1, %r236, %r1;");
    w(s, "@%p1 bra \t$L_KV_LOOP;");
    blank(s);

    // ═══════════════════════════════════════════════════════════════
    // EPILOGUE: O = O / l_i, convert to f16, store
    // ═══════════════════════════════════════════════════════════════
    s.push_str("$L_EPILOGUE:\n");

    // Compute store addresses for O
    // MMA output layout: lane/4 gives the row within the 16-row m-tile
    w(s, "shr.u32 \t%r420, %r31, 2;"); // lane/4 = mma_row_in_16
    w(s, "and.b32 \t%r423, %r31, 3;"); // lane%4
    w(s, "shl.b32 \t%r424, %r423, 2;"); // (lane%4)*4 bytes
    blank(s);

    // Process both m-tiles
    for mt in 0..2u32 {
        let mt_row_off = mt * 16;
        let o_base_start = if mt == 0 { 200 } else { 560 };
        let l_i_0 = if mt == 0 { 234 } else { 594 };
        let l_i_1 = if mt == 0 { 235 } else { 595 };

        // row_0 = warp_id*32 + mt*16 + lane/4
        // row_1 = row_0 + 8
        w(s, &format!("add.s32 \t%r421, %r420, %r36;")); // lane/4 + warp_id*32
        if mt_row_off > 0 {
            w(s, &format!("add.s32 \t%r421, %r421, {};", mt_row_off));
        }
        w(s, "add.s32 \t%r422, %r421, 8;"); // row_1

        // O /= l_i
        for n in 0..8u32 {
            let base = o_base_start + n * 4;
            s.push_str(&format!(
                "\tdiv.full.f32 \t%r{b}, %r{b}, %r{l};\n",
                b = base, l = l_i_0
            ));
            s.push_str(&format!(
                "\tdiv.full.f32 \t%r{b}, %r{b}, %r{l};\n",
                b = base + 1, l = l_i_0
            ));
            s.push_str(&format!(
                "\tdiv.full.f32 \t%r{b}, %r{b}, %r{l};\n",
                b = base + 2, l = l_i_1
            ));
            s.push_str(&format!(
                "\tdiv.full.f32 \t%r{b}, %r{b}, %r{l};\n",
                b = base + 3, l = l_i_1
            ));
        }
        blank(s);

        // Store O to global memory
        w(s, "add.s32 \t%r550, %r7, %r421;"); // global_row_0
        w(s, "add.s32 \t%r551, %r7, %r422;"); // global_row_1

        // Bounds check
        w(s, "setp.lt.s32 \t%p10, %r550, %r1;");
        w(s, "setp.lt.s32 \t%p11, %r551, %r1;");

        w(s, "shl.b32 \t%r552, %r550, 7;"); // row * 128
        w(s, "add.s32 \t%r553, %r552, %r424;"); // + (lane%4)*4
        w(s, "cvt.u64.u32 \t%rd50, %r553;");
        w(s, "add.s64 \t%rd51, %rd13, %rd50;"); // O global addr row 0

        w(s, "shl.b32 \t%r554, %r551, 7;");
        w(s, "add.s32 \t%r555, %r554, %r424;");
        w(s, "cvt.u64.u32 \t%rd52, %r555;");
        w(s, "add.s64 \t%rd53, %rd13, %rd52;"); // O global addr row 1
        blank(s);

        // Convert and store
        for n in 0..8u32 {
            let base = o_base_start + n * 4;
            let h0 = 670 + mt * 20 + n * 2;
            let h1 = h0 + 1;
            s.push_str(&format!(
                "\tcvt.rn.f16x2.f32 \t%r{h0}, %r{}, %r{};\n",
                base + 1, base
            ));
            s.push_str(&format!(
                "\tcvt.rn.f16x2.f32 \t%r{h1}, %r{}, %r{};\n",
                base + 3, base + 2
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

/// Emit cp.async for K or V tile with B128 swizzle.
/// Uses kv_start from %r236. Loads 64×64 tile into smem at given offset.
/// 128 threads × 16 bytes = 2048 per round. Need 8192/2048 = 4 rounds.
fn emit_cp_async_kv_swizzled(s: &mut String, smem_offset: u32, base_ptr: &str, _commit: bool) {
    // cp_row = tid/8 (0..15), cp_col_idx = tid%8, col_bytes = idx*16
    // Round 0: rows 0..15, Round 1: rows 16..31, Round 2: rows 32..47, Round 3: rows 48..63
    for round in 0..4u32 {
        let row_off = round * 16;
        w(s, "add.s32 \t%r60, %r236, %r9;"); // kv_start + cp_row_in_chunk
        if row_off > 0 {
            w(s, &format!("add.s32 \t%r60, %r60, {};", row_off));
        }
        w(s, "setp.lt.s32 \t%p20, %r60, %r1;");
        w(s, "selp.b32 \t%r72, 16, 0, %p20;");
        // Global addr
        w(s, "shl.b32 \t%r62, %r60, 7;"); // row * 128
        w(s, "add.s32 \t%r63, %r62, %r11;"); // + col_bytes
        w(s, "cvt.u64.u32 \t%rd20, %r63;");
        w(s, &format!("add.s64 \t%rd21, {base_ptr}, %rd20;"));
        // Smem with swizzle
        let smem_row_off = row_off * 128;
        w(s, "shl.b32 \t%r64, %r9, 7;"); // cp_row * 128
        w(s, "add.s32 \t%r65, %r64, %r11;"); // + col_bytes
        if smem_row_off > 0 {
            w(s, &format!("add.s32 \t%r65, %r65, {};", smem_row_off));
        }
        w(s, "and.b32 \t%r66, %r65, 896;"); // 0x380
        w(s, "shr.u32 \t%r67, %r66, 3;");
        w(s, "xor.b32 \t%r68, %r65, %r67;");
        w(s, &format!("add.s32 \t%r69, %r68, {};", smem_offset));
        w(s, "add.s32 \t%r69, %r69, %r12;");
        s.push_str("\tcp.async.cg.shared.global [ %r69 + 0 ], [ %rd21 + 0 ], 0x10, %r72;\n");
    }
}

/// Same as above but with dynamic smem offset in a register.
fn emit_cp_async_kv_swizzled_dynamic(s: &mut String, smem_off_reg: &str, base_ptr: &str) {
    for round in 0..4u32 {
        let row_off = round * 16;
        w(s, "add.s32 \t%r60, %r236, %r9;");
        if row_off > 0 {
            w(s, &format!("add.s32 \t%r60, %r60, {};", row_off));
        }
        w(s, "setp.lt.s32 \t%p20, %r60, %r1;");
        w(s, "selp.b32 \t%r72, 16, 0, %p20;");
        w(s, "shl.b32 \t%r62, %r60, 7;");
        w(s, "add.s32 \t%r63, %r62, %r11;");
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
        w(s, &format!("add.s32 \t%r69, %r68, {};", smem_off_reg));
        w(s, "add.s32 \t%r69, %r69, %r12;");
        s.push_str("\tcp.async.cg.shared.global [ %r69 + 0 ], [ %rd21 + 0 ], 0x10, %r72;\n");
    }
}

/// Emit causal flash attention kernel (d=64).
/// Same as non-causal but with:
/// 1. KV loop bounded by block_m_start + BLOCK_M
/// 2. Causal mask applied to S after scaling on diagonal blocks
fn emit_kernel_causal(s: &mut String) {
    // ─── Header ───
    s.push_str(
        r#".version 8.7
.target sm_89
.address_size 64

.extern .shared .align 16 .b8 global_smem[];

.visible .entry flash_attn_fwd_causal(
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
	.reg .pred 	%p<40>;
	.reg .b32 	%r<800>;
	.reg .b64 	%rd<80>;

"#,
    );

    // ─── Parameters ───
    w(s, "ld.param.b64 \t%rd1, [param_Q];");
    w(s, "ld.param.b64 \t%rd2, [param_K];");
    w(s, "ld.param.b64 \t%rd3, [param_V];");
    w(s, "ld.param.b64 \t%rd4, [param_O];");
    w(s, "ld.param.b32 \t%r1, [param_seq_len];");
    w(s, "ld.param.b32 \t%r2, [param_scale];");
    w(s, "ld.param.b32 \t%r3, [param_stride_batch];");
    blank(s);

    // ─── Indexing ───
    w(s, "mov.u32 \t%r4, %ctaid.x;");
    w(s, "mov.u32 \t%r5, %ctaid.y;");
    w(s, "mov.u32 \t%r6, %tid.x;");
    w(s, "shl.b32 \t%r7, %r4, 7;"); // block_m_start
    blank(s);

    // Compute KV loop upper bound for causal: min(seq_len, block_m_start + 128)
    // Round up to multiple of BLOCK_N=64: ((val + 63) / 64) * 64
    w(s, "add.s32 \t%r13, %r7, 128;"); // block_m_start + BLOCK_M
    w(s, "add.s32 \t%r14, %r13, 63;"); // + 63 for rounding
    w(s, "and.b32 \t%r15, %r14, -64;"); // round up to multiple of 64
    w(s, "min.s32 \t%r16, %r15, %r1;"); // min(rounded, seq_len)
    // %r16 = kv_end (causal upper bound)
    blank(s);

    // Base pointers adjusted for batch/head
    w(s, "mul.lo.s32 \t%r8, %r5, %r3;");
    w(s, "mad.wide.s32 \t%rd10, %r8, 2, %rd1;");
    w(s, "mad.wide.s32 \t%rd11, %r8, 2, %rd2;");
    w(s, "mad.wide.s32 \t%rd12, %r8, 2, %rd3;");
    w(s, "mad.wide.s32 \t%rd13, %r8, 2, %rd4;");
    blank(s);

    // cp.async thread decomposition
    w(s, "shr.u32 \t%r9, %r6, 3;");
    w(s, "and.b32 \t%r10, %r6, 7;");
    w(s, "shl.b32 \t%r11, %r10, 4;");
    w(s, "mov.b32 \t%r12, global_smem;");
    blank(s);

    // ─── Load Q → smem ───
    for round in 0..8u32 {
        let row_base = round * 16;
        w(s, "add.s32 \t%r60, %r7, %r9;");
        if row_base > 0 {
            w(s, &format!("add.s32 \t%r60, %r60, {};", row_base));
        }
        w(s, "setp.lt.s32 \t%p20, %r60, %r1;");
        w(s, "selp.b32 \t%r72, 16, 0, %p20;");
        w(s, "shl.b32 \t%r62, %r60, 7;");
        w(s, "add.s32 \t%r63, %r62, %r11;");
        w(s, "cvt.u64.u32 \t%rd20, %r63;");
        w(s, "add.s64 \t%rd21, %rd10, %rd20;");
        let smem_row_base = row_base * 128;
        w(s, "shl.b32 \t%r64, %r9, 7;");
        w(s, "add.s32 \t%r65, %r64, %r11;");
        if smem_row_base > 0 {
            w(s, &format!("add.s32 \t%r65, %r65, {};", smem_row_base));
        }
        w(s, "and.b32 \t%r66, %r65, 896;");
        w(s, "shr.u32 \t%r67, %r66, 3;");
        w(s, "xor.b32 \t%r68, %r65, %r67;");
        w(s, "add.s32 \t%r69, %r68, %r12;");
        s.push_str("\tcp.async.cg.shared.global [ %r69 + 0 ], [ %rd21 + 0 ], 0x10, %r72;\n");
    }
    w(s, "cp.async.commit_group;");
    w(s, "cp.async.wait_group \t0;");
    w(s, "bar.sync \t0;");
    blank(s);

    // ─── Warp/lane decomposition ───
    w(s, "shr.u32 \t%r30, %r6, 5;");
    w(s, "and.b32 \t%r31, %r6, 31;");
    w(s, "and.b32 \t%r32, %r31, 7;");
    w(s, "shr.u32 \t%r33, %r31, 3;");
    w(s, "shr.u32 \t%r34, %r33, 1;");
    w(s, "and.b32 \t%r35, %r33, 1;");
    w(s, "shl.b32 \t%r36, %r30, 5;");
    w(s, "shl.b32 \t%r37, %r34, 3;");
    w(s, "shl.b32 \t%r40, %r35, 4;");
    blank(s);

    // Precompute causal mask row positions for this thread
    // For MMA C fragment: row0 = lane/4 (within 16-row m-tile), row1 = row0 + 8
    // Global query row for m-tile mt:
    //   q_row_0 = block_m_start + warp_id*32 + mt*16 + lane/4
    //   q_row_1 = q_row_0 + 8
    w(s, "shr.u32 \t%r17, %r31, 2;"); // lane/4 = mma_row_0 within m-tile
    w(s, "add.s32 \t%r18, %r17, 8;"); // mma_row_1 = mma_row_0 + 8
    // For the causal mask column:
    //   k_col for n-tile n, regs d0/d2: n*8 + (lane%4)*2
    //   k_col for n-tile n, regs d1/d3: n*8 + (lane%4)*2 + 1
    w(s, "and.b32 \t%r19, %r31, 3;"); // lane%4
    w(s, "shl.b32 \t%r20, %r19, 1;"); // (lane%4)*2 = col offset within n-tile
    blank(s);

    // ─── Load Q fragments ───
    for mt in 0..2u32 {
        let mt_row_off = mt * 16;
        w(s, "add.s32 \t%r38, %r36, %r37;");
        w(s, "add.s32 \t%r39, %r38, %r32;");
        if mt_row_off > 0 {
            w(s, &format!("add.s32 \t%r39, %r39, {};", mt_row_off));
        }
        w(s, "shl.b32 \t%r41, %r39, 7;");
        w(s, "add.s32 \t%r42, %r41, %r40;");
        for ki in 0..4u32 {
            let base = 100 + mt * 16 + ki * 4;
            let k_byte_off = ki * 32;
            w(s, &format!("add.s32 \t%r70, %r42, {};", k_byte_off));
            w(s, "and.b32 \t%r71, %r70, 896;");
            w(s, "shr.u32 \t%r72, %r71, 3;");
            w(s, "xor.b32 \t%r73, %r70, %r72;");
            w(s, "add.s32 \t%r74, %r73, %r12;");
            s.push_str(&format!(
                "\tldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r74];\n",
                base, base + 1, base + 2, base + 3
            ));
        }
    }
    blank(s);

    // K/V ldmatrix addressing
    w(s, "shl.b32 \t%r44, %r33, 4;");
    w(s, "shl.b32 \t%r45, %r32, 7;");
    w(s, "add.s32 \t%r46, %r45, %r44;");
    blank(s);

    // ─── Initialize accumulators ───
    w(s, "mov.b32 \t%r199, 0;");
    for i in 200..232u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r199;\n"));
    }
    for i in 560..592u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r199;\n"));
    }
    w(s, "mov.b32 \t%r232, 0xFF800000;");
    w(s, "mov.b32 \t%r233, 0xFF800000;");
    w(s, "mov.b32 \t%r592, 0xFF800000;");
    w(s, "mov.b32 \t%r593, 0xFF800000;");
    w(s, "mov.b32 \t%r234, 0x00000000;");
    w(s, "mov.b32 \t%r235, 0x00000000;");
    w(s, "mov.b32 \t%r594, 0x00000000;");
    w(s, "mov.b32 \t%r595, 0x00000000;");
    blank(s);

    // ─── KV-loop (causal: use %r16 as upper bound instead of %r1) ───
    w(s, "mov.b32 \t%r236, 0;");
    w(s, "setp.lt.s32 \t%p1, %r236, %r16;"); // use kv_end
    w(s, "@!%p1 bra \t$L_EPILOGUE;");
    blank(s);

    // Load K[0] into KV buf 0
    emit_cp_async_kv_swizzled(s, 16384, "%rd11", false);
    w(s, "cp.async.commit_group;");
    w(s, "cp.async.wait_group \t0;");
    w(s, "bar.sync \t0;");
    blank(s);

    s.push_str("$L_KV_LOOP:\n");
    blank(s);

    // Issue cp.async for NEXT K block into alternate buffer
    w(s, "add.s32 \t%r237, %r236, 64;");
    w(s, "setp.lt.s32 \t%p2, %r237, %r16;"); // use kv_end
    w(s, "shr.u32 \t%r238, %r236, 6;");
    w(s, "and.b32 \t%r239, %r238, 1;");
    w(s, "shl.b32 \t%r240, %r239, 13;");
    w(s, "add.s32 \t%r241, %r240, 16384;");
    w(s, "xor.b32 \t%r242, %r239, 1;");
    w(s, "shl.b32 \t%r243, %r242, 13;");
    w(s, "add.s32 \t%r244, %r243, 16384;");
    w(s, "@!%p2 bra \t$L_SKIP_NEXT_K;");
    w(s, "mov.b32 \t%r250, %r236;");
    w(s, "mov.b32 \t%r236, %r237;");
    emit_cp_async_kv_swizzled_dynamic(s, "%r244", "%rd11");
    w(s, "cp.async.commit_group;");
    w(s, "mov.b32 \t%r236, %r250;");
    s.push_str("$L_SKIP_NEXT_K:\n");
    blank(s);

    // ─── ldmatrix K from current buffer ───
    for n in 0..8u32 {
        for kp in 0..2u32 {
            let base = 300 + (n * 2 + kp) * 4;
            let n_off = n * 1024;
            let kp_off = kp * 64;
            let total_off = n_off + kp_off;
            w(s, &format!("add.s32 \t%r70, %r46, {};", total_off));
            w(s, "and.b32 \t%r71, %r70, 896;");
            w(s, "shr.u32 \t%r72, %r71, 3;");
            w(s, "xor.b32 \t%r73, %r70, %r72;");
            w(s, "add.s32 \t%r74, %r73, %r241;");
            w(s, "add.s32 \t%r74, %r74, %r12;");
            s.push_str(&format!(
                "\tldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r74];\n",
                base, base + 1, base + 2, base + 3
            ));
        }
    }
    blank(s);

    // ─── MMA: S = Q @ K^T ───
    w(s, "mov.b32 \t%r369, 0;");
    for i in 370..402u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r369;\n"));
    }
    for i in 600..632u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r369;\n"));
    }
    blank(s);

    for mt in 0..2u32 {
        let q_base_start = 100 + mt * 16;
        let s_accum_start = if mt == 0 { 370 } else { 600 };
        for ki in 0..4u32 {
            let q_base = q_base_start + ki * 4;
            for n in 0..8u32 {
                let s_base = s_accum_start + n * 4;
                let k_pair = ki / 2;
                let sub_ki = ki % 2;
                let k_frag_base = 300 + (n * 2 + k_pair) * 4;
                let b0 = k_frag_base + sub_ki * 2;
                let b1 = b0 + 1;
                s.push_str(&format!(
                    "\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                     {{ %r{}, %r{}, %r{}, %r{} }}, \
                     {{ %r{}, %r{}, %r{}, %r{} }}, \
                     {{ %r{}, %r{} }}, \
                     {{ %r{}, %r{}, %r{}, %r{} }};\n",
                    s_base, s_base + 1, s_base + 2, s_base + 3,
                    q_base, q_base + 1, q_base + 2, q_base + 3,
                    b0, b1,
                    s_base, s_base + 1, s_base + 2, s_base + 3
                ));
            }
        }
    }
    blank(s);

    w(s, "cp.async.wait_group \t0;");
    w(s, "bar.sync \t0;");
    blank(s);

    // ─── Scale S ───
    for i in 370..402u32 {
        s.push_str(&format!("\tmul.f32 \t%r{i}, %r{i}, %r2;\n"));
    }
    for i in 600..632u32 {
        s.push_str(&format!("\tmul.f32 \t%r{i}, %r{i}, %r2;\n"));
    }
    blank(s);

    // ─── Causal mask ───
    // For each S element: if kv_start + k_col > block_m_start + q_row, set to -inf
    // Compute per-thread q_row values (absolute):
    //   For m-tile mt: q_row_0 = block_m_start + warp_id*32 + mt*16 + lane/4
    //                  q_row_1 = q_row_0 + 8
    // k_col (absolute) for n-tile n: kv_start + n*8 + (lane%4)*2 + {0 or 1}
    //
    // d[0], d[2]: col_offset = n*8 + (lane%4)*2
    // d[1], d[3]: col_offset = n*8 + (lane%4)*2 + 1
    for mt in 0..2u32 {
        let s_start = if mt == 0 { 370 } else { 600 };
        let mt_off = mt * 16;
        // Absolute query row for this m-tile: block_m_start + warp_id*32 + mt*16 + mma_row
        // %r17 = lane/4 (mma_row_0), %r18 = lane/4 + 8 (mma_row_1)
        // q_row_0 = %r7 + %r36 + mt_off + %r17
        w(s, "add.s32 \t%r750, %r7, %r36;"); // block_m_start + warp_id*32
        w(s, &format!("add.s32 \t%r751, %r750, {};", mt_off)); // + mt*16
        w(s, "add.s32 \t%r752, %r751, %r17;"); // q_row_0 (abs)
        w(s, "add.s32 \t%r753, %r751, %r18;"); // q_row_1 (abs)

        for n in 0..8u32 {
            let base = s_start + n * 4;
            let n_col_base = n * 8; // n-tile col offset

            // k_col_even = kv_start + n*8 + (lane%4)*2
            w(s, &format!("add.s32 \t%r754, %r236, {};", n_col_base));
            w(s, "add.s32 \t%r755, %r754, %r20;"); // + (lane%4)*2 = col for d[0]/d[2]
            w(s, "add.s32 \t%r756, %r755, 1;"); // col for d[1]/d[3]

            // d[0]: row=q_row_0, col=%r755. Mask if col > row.
            w(s, &format!("setp.gt.s32 \t%p30, %r755, %r752;"));
            w(
                s,
                &format!(
                    "@%p30 mov.b32 \t%r{}, 0xFF800000;",
                    base
                ),
            );
            // d[1]: row=q_row_0, col=%r756
            w(s, &format!("setp.gt.s32 \t%p31, %r756, %r752;"));
            w(
                s,
                &format!(
                    "@%p31 mov.b32 \t%r{}, 0xFF800000;",
                    base + 1
                ),
            );
            // d[2]: row=q_row_1, col=%r755
            w(s, &format!("setp.gt.s32 \t%p32, %r755, %r753;"));
            w(
                s,
                &format!(
                    "@%p32 mov.b32 \t%r{}, 0xFF800000;",
                    base + 2
                ),
            );
            // d[3]: row=q_row_1, col=%r756
            w(s, &format!("setp.gt.s32 \t%p33, %r756, %r753;"));
            w(
                s,
                &format!(
                    "@%p33 mov.b32 \t%r{}, 0xFF800000;",
                    base + 3
                ),
            );
        }
    }
    blank(s);

    // ─── Online softmax (identical to non-causal from here) ───
    // Step 2: Row max
    w(s, "mov.b32 \t%r402, 0xFF800000;");
    w(s, "mov.b32 \t%r403, 0xFF800000;");
    for n in 0..8u32 {
        let base = 370 + n * 4;
        s.push_str(&format!("\tmax.f32 \t%r402, %r402, %r{};\n", base));
        s.push_str(&format!("\tmax.f32 \t%r402, %r402, %r{};\n", base + 1));
        s.push_str(&format!("\tmax.f32 \t%r403, %r403, %r{};\n", base + 2));
        s.push_str(&format!("\tmax.f32 \t%r403, %r403, %r{};\n", base + 3));
    }
    w(s, "shfl.sync.bfly.b32 \t%r404, %r402, 2, 31, -1;");
    w(s, "max.f32 \t%r402, %r402, %r404;");
    w(s, "shfl.sync.bfly.b32 \t%r405, %r402, 1, 31, -1;");
    w(s, "max.f32 \t%r402, %r402, %r405;");
    w(s, "shfl.sync.bfly.b32 \t%r406, %r403, 2, 31, -1;");
    w(s, "max.f32 \t%r403, %r403, %r406;");
    w(s, "shfl.sync.bfly.b32 \t%r407, %r403, 1, 31, -1;");
    w(s, "max.f32 \t%r403, %r403, %r407;");
    blank(s);

    w(s, "mov.b32 \t%r632, 0xFF800000;");
    w(s, "mov.b32 \t%r633, 0xFF800000;");
    for n in 0..8u32 {
        let base = 600 + n * 4;
        s.push_str(&format!("\tmax.f32 \t%r632, %r632, %r{};\n", base));
        s.push_str(&format!("\tmax.f32 \t%r632, %r632, %r{};\n", base + 1));
        s.push_str(&format!("\tmax.f32 \t%r633, %r633, %r{};\n", base + 2));
        s.push_str(&format!("\tmax.f32 \t%r633, %r633, %r{};\n", base + 3));
    }
    w(s, "shfl.sync.bfly.b32 \t%r634, %r632, 2, 31, -1;");
    w(s, "max.f32 \t%r632, %r632, %r634;");
    w(s, "shfl.sync.bfly.b32 \t%r635, %r632, 1, 31, -1;");
    w(s, "max.f32 \t%r632, %r632, %r635;");
    w(s, "shfl.sync.bfly.b32 \t%r636, %r633, 2, 31, -1;");
    w(s, "max.f32 \t%r633, %r633, %r636;");
    w(s, "shfl.sync.bfly.b32 \t%r637, %r633, 1, 31, -1;");
    w(s, "max.f32 \t%r633, %r633, %r637;");
    blank(s);

    // m_new = max(m_old, row_max)
    w(s, "max.f32 \t%r408, %r232, %r402;");
    w(s, "max.f32 \t%r409, %r233, %r403;");
    w(s, "max.f32 \t%r638, %r592, %r632;");
    w(s, "max.f32 \t%r639, %r593, %r633;");
    blank(s);

    // alpha = exp2(m_old - m_new)
    w(s, "sub.f32 \t%r410, %r232, %r408;");
    w(s, "sub.f32 \t%r411, %r233, %r409;");
    w(s, "ex2.approx.ftz.f32 \t%r412, %r410;");
    w(s, "ex2.approx.ftz.f32 \t%r413, %r411;");
    w(s, "sub.f32 \t%r640, %r592, %r638;");
    w(s, "sub.f32 \t%r641, %r593, %r639;");
    w(s, "ex2.approx.ftz.f32 \t%r642, %r640;");
    w(s, "ex2.approx.ftz.f32 \t%r643, %r641;");
    blank(s);

    // P = exp2(S - m_new)
    for n in 0..8u32 {
        let base = 370 + n * 4;
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r408;\n", b = base));
        s.push_str(&format!("\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n", b = base));
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r408;\n", b = base + 1));
        s.push_str(&format!("\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n", b = base + 1));
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r409;\n", b = base + 2));
        s.push_str(&format!("\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n", b = base + 2));
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r409;\n", b = base + 3));
        s.push_str(&format!("\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n", b = base + 3));
    }
    for n in 0..8u32 {
        let base = 600 + n * 4;
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r638;\n", b = base));
        s.push_str(&format!("\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n", b = base));
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r638;\n", b = base + 1));
        s.push_str(&format!("\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n", b = base + 1));
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r639;\n", b = base + 2));
        s.push_str(&format!("\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n", b = base + 2));
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r639;\n", b = base + 3));
        s.push_str(&format!("\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n", b = base + 3));
    }
    blank(s);

    // Row sum
    w(s, "mov.b32 \t%r414, 0x00000000;");
    w(s, "mov.b32 \t%r415, 0x00000000;");
    for n in 0..8u32 {
        let base = 370 + n * 4;
        s.push_str(&format!("\tadd.f32 \t%r414, %r414, %r{};\n", base));
        s.push_str(&format!("\tadd.f32 \t%r414, %r414, %r{};\n", base + 1));
        s.push_str(&format!("\tadd.f32 \t%r415, %r415, %r{};\n", base + 2));
        s.push_str(&format!("\tadd.f32 \t%r415, %r415, %r{};\n", base + 3));
    }
    w(s, "shfl.sync.bfly.b32 \t%r416, %r414, 2, 31, -1;");
    w(s, "add.f32 \t%r414, %r414, %r416;");
    w(s, "shfl.sync.bfly.b32 \t%r417, %r414, 1, 31, -1;");
    w(s, "add.f32 \t%r414, %r414, %r417;");
    w(s, "shfl.sync.bfly.b32 \t%r418, %r415, 2, 31, -1;");
    w(s, "add.f32 \t%r415, %r415, %r418;");
    w(s, "shfl.sync.bfly.b32 \t%r419, %r415, 1, 31, -1;");
    w(s, "add.f32 \t%r415, %r415, %r419;");
    w(s, "mov.b32 \t%r644, 0x00000000;");
    w(s, "mov.b32 \t%r645, 0x00000000;");
    for n in 0..8u32 {
        let base = 600 + n * 4;
        s.push_str(&format!("\tadd.f32 \t%r644, %r644, %r{};\n", base));
        s.push_str(&format!("\tadd.f32 \t%r644, %r644, %r{};\n", base + 1));
        s.push_str(&format!("\tadd.f32 \t%r645, %r645, %r{};\n", base + 2));
        s.push_str(&format!("\tadd.f32 \t%r645, %r645, %r{};\n", base + 3));
    }
    w(s, "shfl.sync.bfly.b32 \t%r646, %r644, 2, 31, -1;");
    w(s, "add.f32 \t%r644, %r644, %r646;");
    w(s, "shfl.sync.bfly.b32 \t%r647, %r644, 1, 31, -1;");
    w(s, "add.f32 \t%r644, %r644, %r647;");
    w(s, "shfl.sync.bfly.b32 \t%r648, %r645, 2, 31, -1;");
    w(s, "add.f32 \t%r645, %r645, %r648;");
    w(s, "shfl.sync.bfly.b32 \t%r649, %r645, 1, 31, -1;");
    w(s, "add.f32 \t%r645, %r645, %r649;");
    blank(s);

    // O rescaling
    for n in 0..8u32 {
        let base = 200 + n * 4;
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r412;\n", b = base));
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r412;\n", b = base + 1));
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r413;\n", b = base + 2));
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r413;\n", b = base + 3));
    }
    for n in 0..8u32 {
        let base = 560 + n * 4;
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r642;\n", b = base));
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r642;\n", b = base + 1));
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r643;\n", b = base + 2));
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r643;\n", b = base + 3));
    }
    blank(s);

    // Update l_i, m_i
    w(s, "fma.rn.f32 \t%r234, %r234, %r412, %r414;");
    w(s, "fma.rn.f32 \t%r235, %r235, %r413, %r415;");
    w(s, "mov.b32 \t%r232, %r408;");
    w(s, "mov.b32 \t%r233, %r409;");
    w(s, "fma.rn.f32 \t%r594, %r594, %r642, %r644;");
    w(s, "fma.rn.f32 \t%r595, %r595, %r643, %r645;");
    w(s, "mov.b32 \t%r592, %r638;");
    w(s, "mov.b32 \t%r593, %r639;");
    blank(s);

    // P → f16x2 conversion
    for ki in 0..4u32 {
        let n_lo = ki * 2;
        let n_hi = n_lo + 1;
        let p_base = 460 + ki * 4;
        let s_lo = 370 + n_lo * 4;
        let s_hi = 370 + n_hi * 4;
        s.push_str(&format!("\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n", p_base, s_lo + 1, s_lo));
        s.push_str(&format!("\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n", p_base + 1, s_hi + 1, s_hi));
        s.push_str(&format!("\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n", p_base + 2, s_lo + 3, s_lo + 2));
        s.push_str(&format!("\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n", p_base + 3, s_hi + 3, s_hi + 2));
    }
    for ki in 0..4u32 {
        let n_lo = ki * 2;
        let n_hi = n_lo + 1;
        let p_base = 650 + ki * 4;
        let s_lo = 600 + n_lo * 4;
        let s_hi = 600 + n_hi * 4;
        s.push_str(&format!("\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n", p_base, s_lo + 1, s_lo));
        s.push_str(&format!("\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n", p_base + 1, s_hi + 1, s_hi));
        s.push_str(&format!("\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n", p_base + 2, s_lo + 3, s_lo + 2));
        s.push_str(&format!("\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n", p_base + 3, s_hi + 3, s_hi + 2));
    }
    blank(s);

    // V load
    emit_cp_async_kv_swizzled_dynamic(s, "%r241", "%rd12");
    w(s, "cp.async.commit_group;");
    w(s, "cp.async.wait_group \t0;");
    w(s, "bar.sync \t0;");
    blank(s);

    // V ldmatrix
    for n in 0..8u32 {
        for kp in 0..2u32 {
            let base = 480 + (n * 2 + kp) * 4;
            let n_off = n * 1024;
            let kp_off = kp * 64;
            let total_off = n_off + kp_off;
            w(s, &format!("add.s32 \t%r70, %r46, {};", total_off));
            w(s, "and.b32 \t%r71, %r70, 896;");
            w(s, "shr.u32 \t%r72, %r71, 3;");
            w(s, "xor.b32 \t%r73, %r70, %r72;");
            w(s, "add.s32 \t%r74, %r73, %r241;");
            w(s, "add.s32 \t%r74, %r74, %r12;");
            s.push_str(&format!(
                "\tldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r74];\n",
                base, base + 1, base + 2, base + 3
            ));
        }
    }
    blank(s);

    // MMA: O += P @ V
    for mt in 0..2u32 {
        let p_base_start = if mt == 0 { 460 } else { 650 };
        let o_base_start = if mt == 0 { 200 } else { 560 };
        for ki in 0..4u32 {
            let p_base = p_base_start + ki * 4;
            for n in 0..8u32 {
                let o_base = o_base_start + n * 4;
                let k_pair = ki / 2;
                let sub_ki = ki % 2;
                let v_frag_base = 480 + (n * 2 + k_pair) * 4;
                let b0 = v_frag_base + sub_ki * 2;
                let b1 = b0 + 1;
                s.push_str(&format!(
                    "\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                     {{ %r{}, %r{}, %r{}, %r{} }}, \
                     {{ %r{}, %r{}, %r{}, %r{} }}, \
                     {{ %r{}, %r{} }}, \
                     {{ %r{}, %r{}, %r{}, %r{} }};\n",
                    o_base, o_base + 1, o_base + 2, o_base + 3,
                    p_base, p_base + 1, p_base + 2, p_base + 3,
                    b0, b1,
                    o_base, o_base + 1, o_base + 2, o_base + 3
                ));
            }
        }
    }
    blank(s);

    // Loop advance (uses kv_end = %r16 for causal)
    w(s, "add.s32 \t%r236, %r236, 64;");
    w(s, "setp.lt.s32 \t%p1, %r236, %r16;");
    w(s, "@%p1 bra \t$L_KV_LOOP;");
    blank(s);

    // ═══ EPILOGUE ═══
    s.push_str("$L_EPILOGUE:\n");
    w(s, "shr.u32 \t%r420, %r31, 2;");
    w(s, "and.b32 \t%r423, %r31, 3;");
    w(s, "shl.b32 \t%r424, %r423, 2;");
    blank(s);

    for mt in 0..2u32 {
        let mt_row_off = mt * 16;
        let o_base_start = if mt == 0 { 200 } else { 560 };
        let l_i_0 = if mt == 0 { 234 } else { 594 };
        let l_i_1 = if mt == 0 { 235 } else { 595 };

        w(s, "add.s32 \t%r421, %r420, %r36;");
        if mt_row_off > 0 {
            w(s, &format!("add.s32 \t%r421, %r421, {};", mt_row_off));
        }
        w(s, "add.s32 \t%r422, %r421, 8;");

        for n in 0..8u32 {
            let base = o_base_start + n * 4;
            s.push_str(&format!("\tdiv.full.f32 \t%r{b}, %r{b}, %r{l};\n", b = base, l = l_i_0));
            s.push_str(&format!("\tdiv.full.f32 \t%r{b}, %r{b}, %r{l};\n", b = base + 1, l = l_i_0));
            s.push_str(&format!("\tdiv.full.f32 \t%r{b}, %r{b}, %r{l};\n", b = base + 2, l = l_i_1));
            s.push_str(&format!("\tdiv.full.f32 \t%r{b}, %r{b}, %r{l};\n", b = base + 3, l = l_i_1));
        }
        blank(s);

        w(s, "add.s32 \t%r550, %r7, %r421;");
        w(s, "add.s32 \t%r551, %r7, %r422;");
        w(s, "setp.lt.s32 \t%p10, %r550, %r1;");
        w(s, "setp.lt.s32 \t%p11, %r551, %r1;");
        w(s, "shl.b32 \t%r552, %r550, 7;");
        w(s, "add.s32 \t%r553, %r552, %r424;");
        w(s, "cvt.u64.u32 \t%rd50, %r553;");
        w(s, "add.s64 \t%rd51, %rd13, %rd50;");
        w(s, "shl.b32 \t%r554, %r551, 7;");
        w(s, "add.s32 \t%r555, %r554, %r424;");
        w(s, "cvt.u64.u32 \t%rd52, %r555;");
        w(s, "add.s64 \t%rd53, %rd13, %rd52;");
        blank(s);

        for n in 0..8u32 {
            let base = o_base_start + n * 4;
            let h0 = 670 + mt * 20 + n * 2;
            let h1 = h0 + 1;
            s.push_str(&format!("\tcvt.rn.f16x2.f32 \t%r{h0}, %r{}, %r{};\n", base + 1, base));
            s.push_str(&format!("\tcvt.rn.f16x2.f32 \t%r{h1}, %r{}, %r{};\n", base + 3, base + 2));
            let n_off = n * 16;
            s.push_str(&format!("\t@%p10 st.global.b32 [ %rd51 + {n_off} ], %r{h0};\n"));
            s.push_str(&format!("\t@%p11 st.global.b32 [ %rd53 + {n_off} ], %r{h1};\n"));
        }
        blank(s);
    }

    w(s, "ret;");
    s.push_str("}\n");
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
        let ptx = emit_flash_attn_fwd();
        // Check register count
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
    fn test_flash_attn_ptx_is_valid_ascii() {
        let (ptx, _) = load_and_query_kernel();
        for (i, b) in ptx.bytes().enumerate() {
            assert!(b < 128, "Non-ASCII byte 0x{:02x} at position {}", b, i);
        }
    }

    #[test]
    fn test_flash_attn_has_single_entry_point() {
        let (ptx, _) = load_and_query_kernel();
        let entries: Vec<_> = ptx.lines()
            .filter(|l| l.contains(".visible .entry") || l.contains(".entry flash_attn_fwd"))
            .collect();
        assert_eq!(entries.len(), 1, "Must have exactly one entry point, found: {:?}", entries);
    }

    #[test]
    fn test_flash_attn_has_two_mma_phases() {
        let (ptx, _) = load_and_query_kernel();
        let mma_count = ptx.lines().filter(|l| l.contains("mma.sync")).count();
        // Q@K^T and P@V: expect 64+64 = 128 MMA (2 m-tiles × 4 k × 8 n × 2 phases)
        assert!(mma_count >= 32, "Need at least 32 MMA for Q@K^T + P@V, got {}", mma_count);
    }

    #[test]
    fn test_flash_attn_has_online_softmax() {
        let (ptx, _) = load_and_query_kernel();
        let ex2_count = ptx.lines().filter(|l| l.contains("ex2.approx")).count();
        assert!(ex2_count > 0, "Must have ex2.approx for online softmax");
        let shfl_count = ptx.lines().filter(|l| l.contains("shfl")).count();
        assert!(shfl_count > 0, "Must have shuffle for row-max reduction");
    }

    #[test]
    fn test_flash_attn_has_kv_loop() {
        let (ptx, _) = load_and_query_kernel();
        assert!(ptx.contains("$L_KV_LOOP") || ptx.contains("$L_KVLOOP") || ptx.contains("KV_LOOP"),
            "Must have a KV-loop label");
    }

    #[test]
    fn test_flash_attn_uses_cp_async() {
        let (ptx, _) = load_and_query_kernel();
        let cp_count = ptx.lines().filter(|l| l.contains("cp.async.cg")).count();
        assert!(cp_count >= 4, "Need at least 4 cp.async (K + V loads), got {}", cp_count);
    }

    #[test]
    fn test_flash_attn_uses_ldmatrix() {
        let (ptx, _) = load_and_query_kernel();
        let ldm_count = ptx.lines().filter(|l| l.contains("ldmatrix")).count();
        assert!(ldm_count >= 8, "Need at least 8 ldmatrix (Q + K + V), got {}", ldm_count);
    }

    #[test]
    fn test_flash_attn_has_q_load_prologue() {
        let (ptx, _) = load_and_query_kernel();
        let q_load = ptx.find("cp.async").unwrap_or(usize::MAX);
        let kv_loop = ptx.find("KV_LOOP").or(ptx.find("KVLOOP")).unwrap_or(usize::MAX);
        assert!(q_load < kv_loop, "Q must be loaded before the KV loop");
    }

    #[test]
    fn test_flash_attn_has_output_normalization() {
        let (ptx, _) = load_and_query_kernel();
        let has_div = ptx.contains("rcp.approx") || ptx.contains("div.approx") || ptx.contains("div.full");
        assert!(has_div, "Must normalize output O by 1/l_i");
    }

    #[test]
    fn test_flash_attn_stores_output() {
        let (ptx, _) = load_and_query_kernel();
        let st_count = ptx.lines().filter(|l| l.contains("st.global")).count();
        assert!(st_count > 0, "Must store output to global memory");
    }

    #[test]
    fn test_flash_attn_no_spills() {
        let (ptx, _) = load_and_query_kernel();
        let local_count = ptx.lines().filter(|l| l.contains("st.local") || l.contains("ld.local")).count();
        assert_eq!(local_count, 0, "Must have no local memory spills");
    }

    #[test]
    fn test_flash_attn_register_budget() {
        let (_, regs) = load_and_query_kernel();
        assert!(regs <= 800, "Virtual b32 regs should be <= 800, got {}", regs);
        assert!(regs >= 100, "Need at least 100 b32 regs for flash attn, got {}", regs);
    }

    #[test]
    fn test_flash_attn_shared_memory_declaration() {
        let (ptx, _) = load_and_query_kernel();
        assert!(ptx.contains("global_smem") || ptx.contains(".shared"),
            "Must declare shared memory");
    }

    #[test]
    fn test_flash_attn_reqntid() {
        let (ptx, _) = load_and_query_kernel();
        assert!(ptx.contains(".reqntid 128"),
            "Must require 128 threads per block");
    }

    // ═══════════════════════════════════════════════════════════════
    // Algorithmic / structural invariant tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_flash_attn_score_matrix_is_f32() {
        let (ptx, _) = load_and_query_kernel();
        assert!(ptx.contains("f32.f16.f16.f32"),
            "MMA must use f32 accumulation for scores");
    }

    #[test]
    fn test_flash_attn_has_row_max_reduction() {
        let (ptx, _) = load_and_query_kernel();
        let max_count = ptx.lines().filter(|l| l.contains("max.f32")).count();
        assert!(max_count > 0, "Must have max.f32 for row-max computation");
    }

    #[test]
    fn test_flash_attn_has_accumulator_rescaling() {
        let (ptx, _) = load_and_query_kernel();
        let mul_count = ptx.lines().filter(|l| l.contains("mul.f32")).count();
        assert!(mul_count > 10, "Must have mul.f32 for O rescaling, got {}", mul_count);
    }

    #[test]
    fn test_flash_attn_has_p_to_f16_conversion() {
        let (ptx, _) = load_and_query_kernel();
        let cvt_count = ptx.lines()
            .filter(|l: &&str| l.contains("cvt.rn.f16.f32") || l.contains("cvt.rn.f16x2.f32"))
            .count();
        assert!(cvt_count > 0, "Must convert P from f32 to f16 for P@V MMA");
    }

    #[test]
    fn test_flash_attn_q_loaded_once() {
        let (ptx, _) = load_and_query_kernel();
        let kv_loop_pos = ptx.find("KV_LOOP").or(ptx.find("KVLOOP")).unwrap_or(ptx.len());
        let prologue = &ptx[..kv_loop_pos];
        let cp_before = prologue.lines().filter(|l| l.contains("cp.async.cg")).count();
        assert!(cp_before >= 2, "Must have cp.async for Q in prologue (got {} before KV loop)", cp_before);
    }

    #[test]
    fn test_flash_attn_kv_loop_has_two_gemm_phases() {
        let (ptx, _) = load_and_query_kernel();
        let kv_start = ptx.find("KV_LOOP").or(ptx.find("KVLOOP")).unwrap_or(0);
        let kv_end = ptx[kv_start..].find("bra").map(|p| kv_start + p + 500).unwrap_or(ptx.len());
        let kv_body = &ptx[kv_start..kv_end.min(ptx.len())];

        let first_mma = kv_body.find("mma.sync");
        let first_ex2 = kv_body.find("ex2.approx");

        if let (Some(mma_pos), Some(ex2_pos)) = (first_mma, first_ex2) {
            assert!(mma_pos < ex2_pos, "First MMA (Q@K^T) must come before ex2 (softmax)");
            let after_ex2 = &kv_body[ex2_pos..];
            assert!(after_ex2.contains("mma.sync"), "Must have MMA after softmax (P@V phase)");
        }
    }

    #[test]
    fn test_flash_attn_scale_applied() {
        let (ptx, _) = load_and_query_kernel();
        assert!(ptx.contains("param_scale") || ptx.contains("scale"),
            "Must accept a scale parameter");
    }

    #[test]
    fn test_flash_attn_output_is_f16() {
        let (ptx, _) = load_and_query_kernel();
        let has_f16_store = ptx.contains("st.global.b16") ||
                           ptx.contains("st.global.v2.b32") ||
                           ptx.contains("st.global.b32") ||
                           (ptx.contains("cvt.rn.f16x2.f32") && ptx.contains("st.global"));
        assert!(has_f16_store, "Must store output as f16");
    }

    #[test]
    fn test_flash_attn_barrier_between_phases() {
        let (ptx, _) = load_and_query_kernel();
        let barrier_count = ptx.lines().filter(|l| l.contains("bar.sync")).count();
        assert!(barrier_count >= 2, "Need at least 2 barriers (Q sync, KV sync), got {}", barrier_count);
    }

    #[test]
    fn test_flash_attn_ptx_size_reasonable() {
        let (ptx, _) = load_and_query_kernel();
        let line_count = ptx.lines().count();
        assert!(line_count >= 200, "Flash attn should be at least 200 lines, got {}", line_count);
        assert!(line_count <= 5000, "Flash attn should be at most 5000 lines, got {}", line_count);
    }

    #[test]
    fn test_flash_attn_instruction_counts() {
        let (ptx, _) = load_and_query_kernel();
        let mma = ptx.lines().filter(|l| l.contains("mma.sync")).count();
        let ldm = ptx.lines().filter(|l| l.contains("ldmatrix")).count();
        let cpa = ptx.lines().filter(|l| l.contains("cp.async.cg")).count();
        let ex2 = ptx.lines().filter(|l| l.contains("ex2.approx")).count();

        eprintln!("Flash attn instruction counts: mma={}, ldmatrix={}, cp.async={}, ex2={}",
                  mma, ldm, cpa, ex2);

        assert!(mma >= 16, "Need at least 16 MMA (Q@K^T + P@V), got {}", mma);
        assert!(ldm >= 8, "Need at least 8 ldmatrix, got {}", ldm);
        assert!(cpa >= 4, "Need at least 4 cp.async, got {}", cpa);
        assert!(ex2 >= 4, "Need at least 4 ex2 for softmax, got {}", ex2);
    }
}
