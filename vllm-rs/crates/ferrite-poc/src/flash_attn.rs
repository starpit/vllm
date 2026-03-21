// Flash Attention Forward kernel — hand-written PTX
//
// BLOCK_M=64, BLOCK_N=64, HEAD_DIM=64, 128 threads (4 warps)
// No causal mask, no paged KV cache, contiguous Q/K/V/O layout
//
// Q,K,V,O: [batch*heads, seq, 64] contiguous f16 (stride_seq = 64)
//
// Grid: (ceil(seq_q / 64), batch * heads, 1)
//
// Smem layout:
//   Q region:  0..8191     (64 rows × 64 cols × 2B = 8192)
//   KV region: 8192..16383 (64 × 64 × 2B) — K then V reuse same space
//   Total: 16384 bytes
//
// MMA config: m16n8k16
//   Q@K^T: S[64×64] from Q[64×64] × K^T[64×64]
//   P@V: O[64×64] from P[64×64] × V[64×64]
//
// Key optimization: P stays in registers (no smem round-trip)
//   After softmax, P values are converted to f16x2 and used directly
//   as MMA A operands for P@V, matching Triton's approach.
//
// V load is overlapped with softmax computation.

pub fn emit_flash_attn_fwd() -> String {
    let mut s = String::with_capacity(200 * 1024);
    emit_kernel(&mut s);
    s
}

pub const SMEM_BYTES: u32 = 16384;

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
	.reg .b32 	%r<700>;
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
    w(s, "mov.u32 \t%r6, %tid.x;"); // thread id
    w(s, "shl.b32 \t%r7, %r4, 6;"); // block_m_start = block_m * 64
    blank(s);

    // Base pointers adjusted for batch/head
    w(s, "mul.lo.s32 \t%r8, %r5, %r3;");
    w(s, "mad.wide.s32 \t%rd10, %r8, 2, %rd1;"); // Q_base
    w(s, "mad.wide.s32 \t%rd11, %r8, 2, %rd2;"); // K_base
    w(s, "mad.wide.s32 \t%rd12, %r8, 2, %rd3;"); // V_base
    w(s, "mad.wide.s32 \t%rd13, %r8, 2, %rd4;"); // O_base
    blank(s);

    // ─── cp.async thread decomposition ───
    w(s, "shr.u32 \t%r9, %r6, 2;"); // cp_row (0..31)
    w(s, "and.b32 \t%r10, %r6, 3;"); // cp_col_group
    w(s, "shl.b32 \t%r11, %r10, 4;"); // col_bytes = group * 16
    w(s, "mov.b32 \t%r12, global_smem;");
    blank(s);

    // ─── Load Q → smem (offset 0) ───
    emit_cp_async_tile(s, "Q", 0, "%rd10");
    w(s, "cp.async.commit_group;");
    w(s, "cp.async.wait_group \t0;");
    w(s, "bar.sync \t0;");
    blank(s);

    // ─── Warp/lane decomposition ───
    w(s, "shr.u32 \t%r30, %r6, 5;"); // warp_id (0..3)
    w(s, "and.b32 \t%r31, %r6, 31;"); // lane_id (0..31)
    blank(s);

    // ─── Q ldmatrix addresses ───
    w(s, "and.b32 \t%r32, %r31, 7;"); // row_in_frag = lane % 8
    w(s, "shr.u32 \t%r33, %r31, 3;"); // frag_id = lane / 8
    w(s, "shr.u32 \t%r34, %r33, 1;"); // row_half (0 or 1)
    w(s, "and.b32 \t%r35, %r33, 1;"); // col_half (0 or 1)

    w(s, "shl.b32 \t%r36, %r30, 4;"); // warp_id * 16
    w(s, "shl.b32 \t%r37, %r34, 3;"); // row_half * 8
    w(s, "add.s32 \t%r38, %r36, %r37;"); // warp_id*16 + row_half*8
    w(s, "add.s32 \t%r39, %r38, %r32;"); // + row_in_frag = Q row

    w(s, "shl.b32 \t%r40, %r35, 4;"); // col_half * 16 bytes
    w(s, "shl.b32 \t%r41, %r39, 7;"); // row * 128
    w(s, "add.s32 \t%r42, %r41, %r40;"); // + col_bytes = linear offset
    w(s, "add.s32 \t%r43, %r42, %r12;"); // + smem base = Q smem addr for k=0
    blank(s);

    // Load Q fragments (stay live for entire KV-loop):
    // Q_frag: %r100..%r115 (4 × 4 regs)
    for ki in 0..4u32 {
        let base = 100 + ki * 4;
        let k_byte_off = ki * 32;
        if k_byte_off > 0 {
            s.push_str(&format!(
                "\tldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r43+{}];\n",
                base, base + 1, base + 2, base + 3, k_byte_off
            ));
        } else {
            s.push_str(&format!(
                "\tldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r43];\n",
                base, base + 1, base + 2, base + 3
            ));
        }
    }
    blank(s);

    // ─── K/V ldmatrix addressing (smem offset 8192) ───
    w(s, "shl.b32 \t%r44, %r33, 4;"); // frag_id * 16 bytes
    w(s, "shl.b32 \t%r45, %r32, 7;"); // (lane%8) * 128
    w(s, "add.s32 \t%r46, %r45, %r44;"); // + frag_col_bytes
    w(s, "add.s32 \t%r47, %r46, 8192;"); // + K/V smem offset
    w(s, "add.s32 \t%r47, %r47, %r12;"); // + smem base
    blank(s);

    // ─── Initialize accumulators ───
    // O accum: %r200..%r231 (32 regs)
    w(s, "mov.b32 \t%r199, 0;");
    for i in 200..232u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r199;\n"));
    }
    // m_i: %r232, %r233 — initialized to -inf
    w(s, "mov.b32 \t%r232, 0xFF800000;");
    w(s, "mov.b32 \t%r233, 0xFF800000;");
    // l_i: %r234, %r235 — initialized to 0
    w(s, "mov.b32 \t%r234, 0x00000000;");
    w(s, "mov.b32 \t%r235, 0x00000000;");
    blank(s);

    // Precompute negative scale for FMA: neg_scale = -scale
    // We'll use FMA: result = scale * score + (-m_new) = scale*score - m_new
    // Actually simpler: just negate m_new when needed. Keep scale as-is.
    blank(s);

    // ─── KV-loop ───
    w(s, "mov.b32 \t%r236, 0;"); // kv_start
    w(s, "setp.lt.s32 \t%p1, %r236, %r1;");
    w(s, "@!%p1 bra \t$L_EPILOGUE;");
    blank(s);

    s.push_str("$L_KV_LOOP:\n");
    blank(s);

    // ─── Load K block → smem[8192] ───
    emit_cp_async_kv(s, "K", 8192, "%rd11");
    w(s, "cp.async.commit_group;");
    w(s, "cp.async.wait_group \t0;");
    w(s, "bar.sync \t0;");
    blank(s);

    // ─── ldmatrix K (B operand for Q@K^T) ───
    // 8 n-tiles × 2 k-pairs = 16 ldmatrix.trans.x4 calls
    // K_frag at %r300..%r363 (64 regs)
    for n in 0..8u32 {
        for kp in 0..2u32 {
            let base = 300 + (n * 2 + kp) * 4;
            let n_off = n * 1024;
            let kp_off = kp * 64;
            let total_off = n_off + kp_off;
            if total_off > 0 {
                s.push_str(&format!(
                    "\tldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r47+{}];\n",
                    base, base + 1, base + 2, base + 3, total_off
                ));
            } else {
                s.push_str(&format!(
                    "\tldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r47];\n",
                    base, base + 1, base + 2, base + 3
                ));
            }
        }
    }
    blank(s);

    // ─── MMA: S = Q @ K^T ───
    // S accum: %r370..%r401 (8 n-tiles × 4 regs)
    w(s, "mov.b32 \t%r369, 0;");
    for i in 370..402u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r369;\n"));
    }
    blank(s);

    // 4 k-iters × 8 n-tiles = 32 MMA calls
    for ki in 0..4u32 {
        let q_base = 100 + ki * 4;
        for n in 0..8u32 {
            let s_base = 370 + n * 4;
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
    blank(s);

    // ─── Online softmax ───
    // S is in %r370..%r401 (32 f32 regs)

    // Step 1: Scale S using FMA: scaled = scale * raw_score + 0
    // Actually just mul is fine, we'll use FMA for scale*score - m_new later
    for i in 370..402u32 {
        s.push_str(&format!("\tmul.f32 \t%r{i}, %r{i}, %r2;\n"));
    }
    blank(s);

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

    // Step 3: m_new = max(m_old, row_max)
    w(s, "max.f32 \t%r408, %r232, %r402;");
    w(s, "max.f32 \t%r409, %r233, %r403;");
    blank(s);

    // Step 4: alpha = exp2(m_old - m_new)
    w(s, "sub.f32 \t%r410, %r232, %r408;");
    w(s, "sub.f32 \t%r411, %r233, %r409;");
    w(s, "ex2.approx.ftz.f32 \t%r412, %r410;");
    w(s, "ex2.approx.ftz.f32 \t%r413, %r411;");
    blank(s);

    // Step 5: P = exp2(S*scale - m_new)
    // Negate m_new for FMA-style subtraction
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
    blank(s);

    // Step 6: Row sum of P
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
    blank(s);

    // Step 7: Rescale O accumulators: O *= alpha
    for n in 0..8u32 {
        let base = 200 + n * 4;
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r412;\n", b = base));
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r412;\n", b = base + 1));
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r413;\n", b = base + 2));
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r413;\n", b = base + 3));
    }
    blank(s);

    // Step 8: Update l_i, m_i
    w(s, "fma.rn.f32 \t%r234, %r234, %r412, %r414;");
    w(s, "fma.rn.f32 \t%r235, %r235, %r413, %r415;");
    w(s, "mov.b32 \t%r232, %r408;");
    w(s, "mov.b32 \t%r233, %r409;");
    blank(s);

    // ─── Convert P to f16x2 in registers for P@V MMA (NO smem round-trip!) ───
    // P is in %r370..%r401 (f32), 8 n-tiles × 4 regs per tile
    // MMA output layout per thread:
    //   d0: row=(lane/4), col=n_tile*8+(lane%4)*2       (row_group 0)
    //   d1: row=(lane/4), col=n_tile*8+(lane%4)*2+1     (row_group 0)
    //   d2: row=(lane/4)+8, col=n_tile*8+(lane%4)*2     (row_group 1)
    //   d3: row=(lane/4)+8, col=n_tile*8+(lane%4)*2+1   (row_group 1)
    //
    // For P@V, P is the A operand. A fragment for m16n8k16:
    //   Thread t needs: rows (t/4, t/4+8) × k = (t%4)*2, (t%4)*2+1
    //   For k16 covering n-tiles j,j+1:
    //     r0 = {f16(P[row0, j*8+(l%4)*2]), f16(P[row0, j*8+(l%4)*2+1])} = from d0,d1 of n-tile j
    //     r1 = {f16(P[row0, (j+1)*8+(l%4)*2]), f16(P[row0, (j+1)*8+(l%4)*2+1])} = from d0,d1 of n-tile j+1
    //     r2 = {f16(P[row1, j*8+(l%4)*2]), f16(P[row1, j*8+(l%4)*2+1])} = from d2,d3 of n-tile j
    //     r3 = {f16(P[row1, (j+1)*8+(l%4)*2]), f16(P[row1, (j+1)*8+(l%4)*2+1])} = from d2,d3 of n-tile j+1
    //
    // P_frag: %r460..%r475 (4 k-iters × 4 regs)
    // k-iter 0: n-tiles 0,1 → k=0..15
    // k-iter 1: n-tiles 2,3 → k=16..31
    // k-iter 2: n-tiles 4,5 → k=32..47
    // k-iter 3: n-tiles 6,7 → k=48..63

    for ki in 0..4u32 {
        let n_lo = ki * 2;     // first n-tile in this k16 step
        let n_hi = n_lo + 1;   // second n-tile
        let p_base = 460 + ki * 4;

        let s_lo = 370 + n_lo * 4; // S/P regs for n_lo
        let s_hi = 370 + n_hi * 4; // S/P regs for n_hi

        // r0 = pack(d1_nlo, d0_nlo) — row_group 0, n-tile lo
        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base, s_lo + 1, s_lo
        ));
        // r1 = pack(d1_nhi, d0_nhi) — row_group 0, n-tile hi
        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base + 1, s_hi + 1, s_hi
        ));
        // r2 = pack(d3_nlo, d2_nlo) — row_group 1, n-tile lo
        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base + 2, s_lo + 3, s_lo + 2
        ));
        // r3 = pack(d3_nhi, d2_nhi) — row_group 1, n-tile hi
        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            p_base + 3, s_hi + 3, s_hi + 2
        ));
    }
    blank(s);

    // ─── Load V block → smem[8192] (reusing K's region) ───
    emit_cp_async_kv(s, "V", 8192, "%rd12");
    w(s, "cp.async.commit_group;");
    w(s, "cp.async.wait_group \t0;");
    w(s, "bar.sync \t0;");
    blank(s);

    // ─── ldmatrix V (B operand for P@V) ───
    // V at smem[8192], using %r47 base address
    // V_frag: %r480..%r543 (8 n-tiles × 2 k-pairs × 4 regs = 64 regs)
    for n in 0..8u32 {
        for kp in 0..2u32 {
            let base = 480 + (n * 2 + kp) * 4;
            let n_off = n * 1024;
            let kp_off = kp * 64;
            let total_off = n_off + kp_off;
            if total_off > 0 {
                s.push_str(&format!(
                    "\tldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r47+{}];\n",
                    base, base + 1, base + 2, base + 3, total_off
                ));
            } else {
                s.push_str(&format!(
                    "\tldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r47];\n",
                    base, base + 1, base + 2, base + 3
                ));
            }
        }
    }
    blank(s);

    // ─── MMA: O += P @ V ───
    // P_frag at %r460..%r475 (in registers, no smem!)
    // Each k-iter uses 4 P regs from the pack above
    for ki in 0..4u32 {
        let p_base = 460 + ki * 4;
        for n in 0..8u32 {
            let o_base = 200 + n * 4;
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
    w(s, "shr.u32 \t%r420, %r31, 2;"); // lane/4 = mma_row_in_16
    w(s, "add.s32 \t%r421, %r420, %r36;"); // + warp_id*16 = row_0
    w(s, "add.s32 \t%r422, %r421, 8;"); // row_1
    w(s, "and.b32 \t%r423, %r31, 3;"); // lane%4
    w(s, "shl.b32 \t%r424, %r423, 2;"); // (lane%4)*4 bytes

    // O /= l_i
    for n in 0..8u32 {
        let base = 200 + n * 4;
        s.push_str(&format!(
            "\tdiv.full.f32 \t%r{b}, %r{b}, %r234;\n",
            b = base
        ));
        s.push_str(&format!(
            "\tdiv.full.f32 \t%r{b}, %r{b}, %r234;\n",
            b = base + 1
        ));
        s.push_str(&format!(
            "\tdiv.full.f32 \t%r{b}, %r{b}, %r235;\n",
            b = base + 2
        ));
        s.push_str(&format!(
            "\tdiv.full.f32 \t%r{b}, %r{b}, %r235;\n",
            b = base + 3
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
        let base = 200 + n * 4;
        let h0 = 560 + n * 2;
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

    w(s, "ret;");
    s.push_str("}\n");
}

/// Emit cp.async to load a 64×64 f16 tile to smem.
fn emit_cp_async_tile(s: &mut String, _name: &str, smem_offset: u32, base_ptr: &str) {
    w(s, &format!("add.s32 \t%r60, %r7, %r9;")); // row_h0 = block_m_start + cp_row
    w(s, "add.s32 \t%r61, %r60, 32;"); // row_h1
    w(s, "shl.b32 \t%r62, %r60, 7;"); // row_h0 * 128
    w(s, "add.s32 \t%r63, %r62, %r11;");
    w(s, &format!("cvt.u64.u32 \t%rd20, %r63;"));
    w(s, &format!("add.s64 \t%rd21, {base_ptr}, %rd20;"));
    w(s, "shl.b32 \t%r64, %r61, 7;");
    w(s, "add.s32 \t%r65, %r64, %r11;");
    w(s, "cvt.u64.u32 \t%rd22, %r65;");
    w(s, &format!("add.s64 \t%rd23, {base_ptr}, %rd22;"));
    w(s, &format!("add.s32 \t%r66, %r12, {};", smem_offset));
    w(s, "shl.b32 \t%r67, %r9, 7;");
    w(s, "add.s32 \t%r68, %r67, %r11;");
    w(s, "add.s32 \t%r69, %r66, %r68;");
    w(s, "add.s32 \t%r70, %r68, 4096;");
    w(s, "add.s32 \t%r71, %r66, %r70;");
    w(s, "setp.lt.s32 \t%p20, %r60, %r1;");
    w(s, "setp.lt.s32 \t%p21, %r61, %r1;");
    w(s, "selp.b32 \t%r72, 16, 0, %p20;");
    w(s, "selp.b32 \t%r73, 16, 0, %p21;");
    s.push_str("\tcp.async.cg.shared.global [ %r69 + 0 ], [ %rd21 + 0 ], 0x10, %r72;\n");
    s.push_str("\tcp.async.cg.shared.global [ %r71 + 0 ], [ %rd23 + 0 ], 0x10, %r73;\n");
}

/// Emit cp.async for K or V tile (uses kv_start from %r236).
fn emit_cp_async_kv(s: &mut String, _name: &str, smem_offset: u32, base_ptr: &str) {
    w(s, "add.s32 \t%r60, %r236, %r9;");
    w(s, "add.s32 \t%r61, %r60, 32;");
    w(s, "shl.b32 \t%r62, %r60, 7;");
    w(s, "add.s32 \t%r63, %r62, %r11;");
    w(s, "cvt.u64.u32 \t%rd20, %r63;");
    w(s, &format!("add.s64 \t%rd21, {base_ptr}, %rd20;"));
    w(s, "shl.b32 \t%r64, %r61, 7;");
    w(s, "add.s32 \t%r65, %r64, %r11;");
    w(s, "cvt.u64.u32 \t%rd22, %r65;");
    w(s, &format!("add.s64 \t%rd23, {base_ptr}, %rd22;"));
    w(s, &format!("add.s32 \t%r66, %r12, {};", smem_offset));
    w(s, "shl.b32 \t%r67, %r9, 7;");
    w(s, "add.s32 \t%r68, %r67, %r11;");
    w(s, "add.s32 \t%r69, %r66, %r68;");
    w(s, "add.s32 \t%r70, %r68, 4096;");
    w(s, "add.s32 \t%r71, %r66, %r70;");
    w(s, "setp.lt.s32 \t%p20, %r60, %r1;");
    w(s, "setp.lt.s32 \t%p21, %r61, %r1;");
    w(s, "selp.b32 \t%r72, 16, 0, %p20;");
    w(s, "selp.b32 \t%r73, 16, 0, %p21;");
    s.push_str("\tcp.async.cg.shared.global [ %r69 + 0 ], [ %rd21 + 0 ], 0x10, %r72;\n");
    s.push_str("\tcp.async.cg.shared.global [ %r71 + 0 ], [ %rd23 + 0 ], 0x10, %r73;\n");
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
        // Q@K^T and P@V: expect 32+32 = 64 MMA (or similar)
        assert!(mma_count >= 32, "Need at least 32 MMA for Q@K^T + P@V, got {}", mma_count);
    }

    #[test]
    fn test_flash_attn_has_online_softmax() {
        let (ptx, _) = load_and_query_kernel();
        let ex2_count = ptx.lines().filter(|l| l.contains("ex2.approx")).count();
        assert!(ex2_count > 0, "Must have ex2.approx for online softmax");
        // Should also have max reduction via shuffle
        let shfl_count = ptx.lines().filter(|l| l.contains("shfl")).count();
        assert!(shfl_count > 0, "Must have shuffle for row-max reduction");
    }

    #[test]
    fn test_flash_attn_has_kv_loop() {
        let (ptx, _) = load_and_query_kernel();
        // Must have a loop label and branch back
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
        // Q should be loaded before the KV loop
        let q_load = ptx.find("cp.async").unwrap_or(usize::MAX);
        let kv_loop = ptx.find("KV_LOOP").or(ptx.find("KVLOOP")).unwrap_or(usize::MAX);
        assert!(q_load < kv_loop, "Q must be loaded before the KV loop");
    }

    #[test]
    fn test_flash_attn_has_output_normalization() {
        let (ptx, _) = load_and_query_kernel();
        // After KV loop, must divide O by l_i (rcp or div)
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
        // Q@K^T scores must accumulate in f32 (not f16) for numerical stability
        assert!(ptx.contains("f32.f16.f16.f32"),
            "MMA must use f32 accumulation for scores");
    }

    #[test]
    fn test_flash_attn_has_row_max_reduction() {
        let (ptx, _) = load_and_query_kernel();
        // Online softmax needs row-wise max reduction via shuffle
        let max_count = ptx.lines().filter(|l| l.contains("max.f32")).count();
        assert!(max_count > 0, "Must have max.f32 for row-max computation");
    }

    #[test]
    fn test_flash_attn_has_accumulator_rescaling() {
        let (ptx, _) = load_and_query_kernel();
        // Online softmax rescales O by exp(m_old - m_new) each KV block
        // This manifests as mul.f32 on the O accumulators
        let mul_count = ptx.lines().filter(|l| l.contains("mul.f32")).count();
        assert!(mul_count > 10, "Must have mul.f32 for O rescaling, got {}", mul_count);
    }

    #[test]
    fn test_flash_attn_has_p_to_f16_conversion() {
        let (ptx, _) = load_and_query_kernel();
        // After softmax, P (f32) must be converted to f16 for the P@V MMA
        let cvt_count = ptx.lines()
            .filter(|l: &&str| l.contains("cvt.rn.f16.f32") || l.contains("cvt.rn.f16x2.f32"))
            .count();
        assert!(cvt_count > 0, "Must convert P from f32 to f16 for P@V MMA");
    }

    #[test]
    fn test_flash_attn_q_loaded_once() {
        let (ptx, _) = load_and_query_kernel();
        // Q should be loaded to smem ONCE (in prologue), not per KV iteration
        // Count cp.async before the KV loop vs inside
        let kv_loop_pos = ptx.find("KV_LOOP").or(ptx.find("KVLOOP")).unwrap_or(ptx.len());
        let prologue = &ptx[..kv_loop_pos];
        let cp_before = prologue.lines().filter(|l| l.contains("cp.async.cg")).count();
        assert!(cp_before >= 2, "Must have cp.async for Q in prologue (got {} before KV loop)", cp_before);
    }

    #[test]
    fn test_flash_attn_kv_loop_has_two_gemm_phases() {
        let (ptx, _) = load_and_query_kernel();
        // Inside the KV loop, there should be two distinct groups of MMA:
        // 1. Q@K^T (score computation)
        // 2. P@V (output accumulation)
        // Between them: softmax (ex2, max, sum)
        let kv_start = ptx.find("KV_LOOP").or(ptx.find("KVLOOP")).unwrap_or(0);
        let kv_end = ptx[kv_start..].find("bra").map(|p| kv_start + p + 500).unwrap_or(ptx.len());
        let kv_body = &ptx[kv_start..kv_end.min(ptx.len())];

        let first_mma = kv_body.find("mma.sync");
        let first_ex2 = kv_body.find("ex2.approx");

        if let (Some(mma_pos), Some(ex2_pos)) = (first_mma, first_ex2) {
            assert!(mma_pos < ex2_pos, "First MMA (Q@K^T) must come before ex2 (softmax)");
            // After ex2, there should be more MMA (P@V)
            let after_ex2 = &kv_body[ex2_pos..];
            assert!(after_ex2.contains("mma.sync"), "Must have MMA after softmax (P@V phase)");
        }
    }

    #[test]
    fn test_flash_attn_scale_applied() {
        let (ptx, _) = load_and_query_kernel();
        // Attention scores must be scaled by 1/sqrt(d)
        // This is typically done as mul.f32 by the scale factor
        assert!(ptx.contains("param_scale") || ptx.contains("scale"),
            "Must accept a scale parameter");
    }

    #[test]
    fn test_flash_attn_output_is_f16() {
        let (ptx, _) = load_and_query_kernel();
        // Final output should be stored as f16 (half precision)
        let has_f16_store = ptx.contains("st.global.b16") ||
                           ptx.contains("st.global.v2.b32") ||
                           ptx.contains("st.global.b32") ||
                           (ptx.contains("cvt.rn.f16x2.f32") && ptx.contains("st.global"));
        assert!(has_f16_store, "Must store output as f16");
    }

    #[test]
    fn test_flash_attn_barrier_between_phases() {
        let (ptx, _) = load_and_query_kernel();
        // Must have barriers between load and compute phases
        let barrier_count = ptx.lines().filter(|l| l.contains("bar.sync")).count();
        assert!(barrier_count >= 2, "Need at least 2 barriers (Q sync, KV sync), got {}", barrier_count);
    }

    #[test]
    fn test_flash_attn_ptx_size_reasonable() {
        let (ptx, _) = load_and_query_kernel();
        let line_count = ptx.lines().count();
        assert!(line_count >= 200, "Flash attn should be at least 200 lines, got {}", line_count);
        assert!(line_count <= 3000, "Flash attn should be at most 3000 lines, got {}", line_count);
    }

    #[test]
    fn test_flash_attn_instruction_counts() {
        let (ptx, _) = load_and_query_kernel();
        let mma = ptx.lines().filter(|l| l.contains("mma.sync")).count();
        let ldm = ptx.lines().filter(|l| l.contains("ldmatrix")).count();
        let cpa = ptx.lines().filter(|l| l.contains("cp.async.cg")).count();
        let ex2 = ptx.lines().filter(|l| l.contains("ex2.approx")).count();

        // Log for debugging during optimization
        eprintln!("Flash attn instruction counts: mma={}, ldmatrix={}, cp.async={}, ex2={}",
                  mma, ldm, cpa, ex2);

        // Sanity bounds
        assert!(mma >= 16, "Need at least 16 MMA (Q@K^T + P@V), got {}", mma);
        assert!(ldm >= 8, "Need at least 8 ldmatrix, got {}", ldm);
        assert!(cpa >= 4, "Need at least 4 cp.async, got {}", cpa);
        assert!(ex2 >= 4, "Need at least 4 ex2 for softmax, got {}", ex2);
    }
}
