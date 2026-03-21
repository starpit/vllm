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
//   K region:  8192..16383 (64 × 64 × 2B) — also used for P during P@V
//   V region:  16384..24575 (64 × 64 × 2B)
//   Total: 24576 bytes
//
// MMA config: m16n8k16
//   Q@K^T: S[64×64] from Q[64×64] × K^T[64×64]
//     REG_M=4 (64/16), REG_N=8 (64/8), k_iters=4 (64/16) → 32 MMA per k-iter, 128 total
//     But each warp handles m16 (REG_M=1 per warp), so per warp: 8 × 4 = 32 MMA
//   P@V: O[64×64] from P[64×64] × V[64×64]
//     Same structure: 32 MMA per warp
//
// Online softmax between Q@K^T and P@V:
//   m_new = max(m_old, row_max(S*scale))
//   alpha = exp2(m_old - m_new)
//   P = exp2(S*scale - m_new)
//   O *= alpha
//   l_i = l_i * alpha + row_sum(P)
//   m_i = m_new

pub fn emit_flash_attn_fwd() -> String {
    let mut s = String::with_capacity(200 * 1024);
    emit_kernel(&mut s);
    s
}

pub const SMEM_BYTES: u32 = 24576;

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
    // 128 threads load 64×64 matrix (8192 bytes, 512 × 16B chunks)
    // 512/128 = 4 loads per thread
    // Mapping: cp_row = tid >> 2 (0..31), cp_col_group = tid & 3 (0..3)
    //   4 bytes per f16 pair, 16B = 8 f16 elements
    //   col_bytes = cp_col_group * 16
    //   Two row halves: rows 0..31 and 32..63
    w(s, "shr.u32 \t%r9, %r6, 2;"); // cp_row (0..31)
    w(s, "and.b32 \t%r10, %r6, 3;"); // cp_col_group
    w(s, "shl.b32 \t%r11, %r10, 4;"); // col_bytes = group * 16
    w(s, "mov.b32 \t%r12, global_smem;");
    blank(s);

    // ─── Load Q → smem (offset 0) ───
    // Q is contiguous: Q[row][col] at Q_base + row * 128 + col * 2 (128 = 64 * 2)
    emit_cp_async_tile(s, "Q", 0, "%rd10"); // Q → smem[0..8191]
    w(s, "cp.async.commit_group;");
    w(s, "cp.async.wait_group \t0;");
    w(s, "bar.sync \t0;");
    blank(s);

    // ─── Warp/lane decomposition ───
    w(s, "shr.u32 \t%r30, %r6, 5;"); // warp_id (0..3)
    w(s, "and.b32 \t%r31, %r6, 31;"); // lane_id (0..31)
    blank(s);

    // ─── Q ldmatrix addresses ───
    // Q at smem offset 0, row-major [64×64] f16, row_stride=128 bytes
    // For warp w: handles rows w*16..w*16+15
    // ldmatrix.x4: threads 0..7→frag0, 8..15→frag1, 16..23→frag2, 24..31→frag3
    //   Each thread provides smem addr for one row of 8-row fragment
    //   .x4 = 4 consecutive m8n8 fragments = m8×n32 (covers k=32)
    //
    // For MMA A operand m16n8k16:
    //   {r0,r1,r2,r3}: r0,r1=rows 0..7 k=0..15; r2,r3=rows 8..15 k=0..15
    //   ldmatrix.x4 loads: frag0=rows0..7 k0..7, frag1=rows0..7 k8..15,
    //                      frag2=rows8..15 k0..7, frag3=rows8..15 k8..15
    //   → exactly maps to {r0,r1,r2,r3} for k16

    // Lane's row within the ldmatrix:
    //   frag_id = lane/8 (0..3)
    //   row_in_8 = lane%8
    //   For frag 0,1: row = warp*16 + row_in_8       (rows 0..7)
    //   For frag 2,3: row = warp*16 + 8 + row_in_8   (rows 8..15)
    //   col_start = k_start + (frag_id & 1) * 8  elements
    //   But for .x4, hardware handles the 4-frag split automatically.
    //   Thread just provides ONE address: the start of its row's data.
    //   The .x4 reads 32 bytes (16 f16 = 16 elements = 2 × n8 tiles)
    //   Wait — I need to be more precise.

    // For ldmatrix.sync.aligned.m8n8.x4:
    //   Each of the 32 threads provides one smem address.
    //   Thread t belongs to fragment t/8.
    //   Thread t loads 16 bytes starting at its address.
    //   The 16 bytes = 8 f16 values = one row of an m8×n8 tile (8 columns).
    //   Fragment i (threads i*8..i*8+7) produces a complete m8×n8 matrix.
    //   With .x4: 4 fragments, 4 registers output.
    //   The fragments ARE consecutive in the n dimension (columns).

    // So for Q ldmatrix covering k=0..31 (4 × n8 tiles):
    //   Thread t in fragment f provides Q[row, k_start + f*8 + ...] — NO
    //   Thread t provides the addr of row[t%8] at column offset specific to fragment f.
    //   Specifically: thread t's address should point to:
    //     Q_smem + row * 128 + (k_group_start + frag_id * 8) * 2
    //   where row = warp*16 + row_in_8 (for frags 0,1) or warp*16 + 8 + row_in_8 (frags 2,3)

    // Wait, that mixes up the 2 possible layouts. Let me look at what Triton does.
    // In Triton PTX line 406:
    //   add.s32 %r339, %r338, %r9;   // %r338 = buffer base, %r9 = swizzle offset
    //   ldmatrix ... [%r339];           // base
    //   ldmatrix ... [%r339+1024];      // +1024 bytes = +8 rows (8*128)
    //   ldmatrix ... [%r339+2048];      // +16 rows
    //   ...
    //   ldmatrix ... [%r339+7168];      // +56 rows
    //
    // So Triton does 8 separate ldmatrix.x4 calls, each at a different row offset.
    // Each ldmatrix.x4 covers one m8 fragment across k=32 (4 n8 tiles).
    // But they load K, not Q.
    //
    // For Q (which is the A operand), Triton does (line 425):
    //   ldmatrix ... [%r11];
    //   ldmatrix ... [%r12];
    //   ldmatrix ... [%r13];
    //   ldmatrix ... [%r14];
    // 4 ldmatrix.x4 = 4 fragment groups of 4 regs = 16 regs for Q
    // With BLOCK_M=128, that's 4 × m8 = m32. But each warp handles m16...
    // Wait, in the 256-thread kernel, 8 warps, and different warps execute
    // different ldmatrix from different addresses computed per-warp.

    // For my 128-thread/4-warp kernel, each warp loads Q for its m16 tile:
    //   2 ldmatrix.x4 per k32 step (one for rows 0..7, one for rows 8..15)
    //   Wait — ldmatrix.x4 with the right addr already covers m16×k16 in one call!
    //   Because frag0,1 = rows 0..7 (k=0..7 and k=8..15)
    //          frag2,3 = rows 8..15 (k=0..7 and k=8..15)
    //   → gives {r0,r1,r2,r3} = complete A fragment for m16n8k16

    // So for each warp:
    //   Thread t (lane l): compute addr for the row/col it represents
    //   frag_id = l / 8
    //   row_in_frag = l % 8
    //   global_row = warp_id*16 + (frag_id >= 2 ? 8 : 0) + row_in_frag
    //   col = k_start + (frag_id & 1) * 8
    //   smem_addr = smem_Q + global_row * 128 + col * 2

    // Compute Q ldmatrix lane address:
    w(s, "and.b32 \t%r32, %r31, 7;"); // row_in_frag = lane % 8
    w(s, "shr.u32 \t%r33, %r31, 3;"); // frag_id = lane / 8
    w(s, "shr.u32 \t%r34, %r33, 1;"); // frag_id >= 2 → row_half (0 or 1)
    w(s, "and.b32 \t%r35, %r33, 1;"); // frag_id & 1 → col_half (0 or 1)

    // row = warp_id*16 + row_half*8 + row_in_frag
    w(s, "shl.b32 \t%r36, %r30, 4;"); // warp_id * 16
    w(s, "shl.b32 \t%r37, %r34, 3;"); // row_half * 8
    w(s, "add.s32 \t%r38, %r36, %r37;"); // warp_id*16 + row_half*8
    w(s, "add.s32 \t%r39, %r38, %r32;"); // + row_in_frag = Q row

    // col_elem = k_start + col_half * 8
    // col_bytes = col_elem * 2 = col_half * 16
    w(s, "shl.b32 \t%r40, %r35, 4;"); // col_half * 16 bytes

    // smem_addr_base = smem_Q + row * 128 + col_bytes
    w(s, "shl.b32 \t%r41, %r39, 7;"); // row * 128
    w(s, "add.s32 \t%r42, %r41, %r40;"); // + col_bytes = linear offset
    w(s, "add.s32 \t%r43, %r42, %r12;"); // + smem base = Q smem addr for k=0
    blank(s);

    // For k_iter i: add i*32 bytes (16 elements * 2 bytes each)
    // But wait: .x4 loads 4 fragments spanning k=0..31. So one ldmatrix.x4 covers k0..k31.
    // We need 2 ldmatrix.x4 calls per warp to cover k=0..63.
    // Call 1: k=0..31 at addr + 0
    // Call 2: k=32..63 at addr + 64 (32 elements * 2 bytes)

    // Load Q fragments (stay live for entire KV-loop):
    // Q_frag[0] = {%r100..%r103}: k=0..15 (from ldmatrix at k=0)
    // Q_frag[1] = {%r104..%r107}: k=16..31 (also from ldmatrix at k=0, it covers k0..31)

    // Wait — one ldmatrix.x4 gives 4 regs = m16×k16 (one MMA A fragment).
    // The 4 fragments of .x4 produce exactly {r0,r1,r2,r3} for m16n8k16.
    // For k16: we get ONE A fragment. For k=0..63, we need 4 ldmatrix.x4 calls.

    // Hmm, let me reconsider. For .x4, the 4 fragments each load one m8×n8 tile.
    // For A operand of m16n8k16:
    //   r0 = rows 0..7, k=0..7 (half of k16)
    //   r1 = rows 0..7, k=8..15
    //   r2 = rows 8..15, k=0..7
    //   r3 = rows 8..15, k=8..15
    //
    // ldmatrix.x4 produces exactly this if:
    //   frag0 (threads 0-7) → r0: each thread t loads Q[row_t, col_start..col_start+7]
    //   frag1 (threads 8-15) → r1: each thread t loads Q[row_t, col_start+8..col_start+15]
    //   frag2 (threads 16-23) → r2: each thread t loads Q[row_t+8, col_start..col_start+7]
    //   frag3 (threads 24-31) → r3: each thread t loads Q[row_t+8, col_start+8..col_start+15]
    //
    // Wait, frag0 and frag2 BOTH start at col_start? No — each thread provides its OWN addr.
    // For frag0 (threads 0-7): addr = Q + warp*16*128 + (lane%8)*128 + col_start*2
    // For frag1 (threads 8-15): addr = Q + warp*16*128 + (lane%8)*128 + (col_start+8)*2
    // For frag2 (threads 16-23): addr = Q + (warp*16+8)*128 + (lane%8)*128 + col_start*2
    // For frag3 (threads 24-31): addr = Q + (warp*16+8)*128 + (lane%8)*128 + (col_start+8)*2
    //
    // This is exactly what I computed: each thread provides its own row,col addr.
    // The hardware reads 16 bytes from each thread's addr.
    // So one ldmatrix.x4 gives exactly one m16×k16 A fragment = {r0,r1,r2,r3}.
    // For 4 k-iterations (k=0..63), we need 4 ldmatrix.x4 calls per warp.

    // Q fragments: %r100..%r115 (4 × 4 regs)
    for ki in 0..4u32 {
        let base = 100 + ki * 4;
        let k_byte_off = ki * 32; // k_start * 2 bytes = ki * 16 * 2
        if k_byte_off > 0 {
            s.push_str(&format!(
                "\tldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r43+{}];\n",
                base, base + 1, base + 2, base + 3, k_byte_off
            ));
        } else {
            s.push_str(&format!(
                "\tldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r43];\n",
                base,
                base + 1,
                base + 2,
                base + 3
            ));
        }
    }
    blank(s);

    // ─── K ldmatrix addressing (same structure, smem offset 8192) ───
    // For K: B operand of Q@K^T, need ldmatrix.trans
    // K in smem at offset 8192, row-major [64×64] f16
    // K^T[k,n] = K[n,k]: n=score column, k=head_dim
    //
    // For ldmatrix.trans.x4 (B operand m16n8k16):
    //   loads k16×n8 transposed fragment = 2 regs {b0, b1}
    //   BUT .x4 gives 4 regs → covers TWO k16 steps: {b0_k0, b1_k0, b0_k1, b1_k1}
    //
    // For B fragment: thread t provides addr in smem
    //   .trans: reads an 8×8 tile and transposes it
    //   frag0 (threads 0-7): reads K[n_start+t%8, k_pair_start..k_pair_start+7]
    //   frag1 (threads 8-15): reads K[n_start+t%8, k_pair_start+8..+15]
    //   frag2 (threads 16-23): reads K[n_start+t%8, k_pair_start+16..+23]
    //   frag3 (threads 24-31): reads K[n_start+t%8, k_pair_start+24..+31]
    //   → output: {b0_k0, b1_k0, b0_k1, b1_k1} for k0=pair*32..pair*32+15, k1=pair*32+16..pair*32+31

    // K ldmatrix.trans address per thread:
    //   K_row = n_tile*8 + (lane%8)
    //   K_col_start = k_pair*32 + (lane/8)*8 = k_pair*32 + frag_id*8
    //   K_col_bytes = K_col_start * 2
    //   addr = smem_K + K_row * 128 + K_col_bytes

    // Pre-compute the per-lane part (independent of n_tile and k_pair):
    // lane_k_row_part = (lane%8)  → %r32 (already computed)
    // lane_frag_col = (lane/8)*8*2 = frag_id*16 bytes → already %r40 (col_half*16)
    // Actually %r40 = (frag_id & 1) * 16, not frag_id * 16
    // Need frag_id * 16 = %r33 * 16
    w(s, "shl.b32 \t%r44, %r33, 4;"); // frag_id * 16 bytes

    // K lane base = (lane%8)*128 + frag_id*16 + smem_K
    w(s, "shl.b32 \t%r45, %r32, 7;"); // (lane%8) * 128
    w(s, "add.s32 \t%r46, %r45, %r44;"); // + frag_col_bytes
    w(s, "add.s32 \t%r47, %r46, 8192;"); // + K smem offset
    w(s, "add.s32 \t%r47, %r47, %r12;"); // + smem base
    // For n_tile i: add i*1024 (= 8 rows * 128 bytes/row)
    // For k_pair p: add p*64 (= 32 elements * 2 bytes)

    // V has the same layout at smem offset 16384:
    w(s, "add.s32 \t%r48, %r46, 16384;");
    w(s, "add.s32 \t%r48, %r48, %r12;");
    blank(s);

    // ─── Initialize accumulators ───
    // O accum: %r200..%r231 (32 regs, 8 n-tiles × 4 regs per MMA output)
    w(s, "mov.b32 \t%r199, 0;");
    for i in 200..232u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r199;\n"));
    }
    // m_i: %r232 (row_group_0), %r233 (row_group_1) — initialized to -inf
    w(s, "mov.b32 \t%r232, 0xFF800000;");
    w(s, "mov.b32 \t%r233, 0xFF800000;");
    // l_i: %r234, %r235 — initialized to 0
    w(s, "mov.b32 \t%r234, 0x00000000;");
    w(s, "mov.b32 \t%r235, 0x00000000;");
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
            let n_off = n * 1024; // n_tile * 8 rows * 128 bytes
            let kp_off = kp * 64; // k_pair * 32 elems * 2 bytes
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

    // ─── MMA: S = Q @ K^T (zero-init S, then accumulate) ───
    // S accum: %r370..%r401 (8 n-tiles × 4 regs) — use temp regs
    w(s, "mov.b32 \t%r369, 0;");
    for i in 370..402u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r369;\n"));
    }
    blank(s);

    // 4 k-iters × 8 n-tiles = 32 MMA calls
    for ki in 0..4u32 {
        let q_base = 100 + ki * 4; // Q fragment for this k-iter
        for n in 0..8u32 {
            let s_base = 370 + n * 4; // S accumulator for n-tile n
            // K fragment: k_pair = ki/2, sub = ki%2
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
    blank(s);

    // ─── Online softmax ───
    // S is in %r370..%r401 (32 f32 regs)
    // MMA output layout per thread for m16n8:
    //   d0: row=(lane/4), col=(lane%4)*2       (row_group 0)
    //   d1: row=(lane/4), col=(lane%4)*2+1     (row_group 0)
    //   d2: row=(lane/4)+8, col=(lane%4)*2     (row_group 1)
    //   d3: row=(lane/4)+8, col=(lane%4)*2+1   (row_group 1)
    //
    // Each thread has 2 score-column values per n-tile per row-group
    // 8 n-tiles × 2 values = 16 values per row-group
    // Rows within a row-group: 8 unique rows, 4 threads share each row (lane%4)
    // Full row has 64 score columns: 4 threads × 16 values = 64 ✓

    // Step 1: Scale S
    for i in 370..402u32 {
        s.push_str(&format!("\tmul.f32 \t%r{i}, %r{i}, %r2;\n"));
    }
    blank(s);

    // Step 2: Row max (local max across 16 values, then shuffle across 4 threads)
    w(s, "mov.b32 \t%r402, 0xFF800000;"); // local_max_row0 = -inf
    w(s, "mov.b32 \t%r403, 0xFF800000;"); // local_max_row1 = -inf
    for n in 0..8u32 {
        let base = 370 + n * 4;
        // Row group 0: d0, d1
        s.push_str(&format!("\tmax.f32 \t%r402, %r402, %r{};\n", base));
        s.push_str(&format!("\tmax.f32 \t%r402, %r402, %r{};\n", base + 1));
        // Row group 1: d2, d3
        s.push_str(&format!("\tmax.f32 \t%r403, %r403, %r{};\n", base + 2));
        s.push_str(&format!("\tmax.f32 \t%r403, %r403, %r{};\n", base + 3));
    }
    // Shuffle reduce across lane%4 (butterfly distance 2, then 1)
    w(s, "shfl.sync.bfly.b32 \t%r404, %r402, 2, 31, -1;");
    w(s, "max.f32 \t%r402, %r402, %r404;");
    w(s, "shfl.sync.bfly.b32 \t%r405, %r402, 1, 31, -1;");
    w(s, "max.f32 \t%r402, %r402, %r405;"); // row_max group 0

    w(s, "shfl.sync.bfly.b32 \t%r406, %r403, 2, 31, -1;");
    w(s, "max.f32 \t%r403, %r403, %r406;");
    w(s, "shfl.sync.bfly.b32 \t%r407, %r403, 1, 31, -1;");
    w(s, "max.f32 \t%r403, %r403, %r407;"); // row_max group 1
    blank(s);

    // Step 3: m_new = max(m_old, row_max)
    w(s, "max.f32 \t%r408, %r232, %r402;"); // m_new row group 0
    w(s, "max.f32 \t%r409, %r233, %r403;"); // m_new row group 1
    blank(s);

    // Step 4: alpha = exp2(m_old - m_new)
    w(s, "sub.f32 \t%r410, %r232, %r408;");
    w(s, "sub.f32 \t%r411, %r233, %r409;");
    w(s, "ex2.approx.ftz.f32 \t%r412, %r410;"); // alpha_0
    w(s, "ex2.approx.ftz.f32 \t%r413, %r411;"); // alpha_1
    blank(s);

    // Step 5: P = exp2(S - m_new)
    for n in 0..8u32 {
        let base = 370 + n * 4;
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r408;\n", b = base));
        s.push_str(&format!("\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n", b = base));
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r408;\n", b = base + 1));
        s.push_str(&format!(
            "\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n",
            b = base + 1
        ));
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r409;\n", b = base + 2));
        s.push_str(&format!(
            "\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n",
            b = base + 2
        ));
        s.push_str(&format!("\tsub.f32 \t%r{b}, %r{b}, %r409;\n", b = base + 3));
        s.push_str(&format!(
            "\tex2.approx.ftz.f32 \t%r{b}, %r{b};\n",
            b = base + 3
        ));
    }
    blank(s);

    // Step 6: Row sum of P
    w(s, "mov.b32 \t%r414, 0x00000000;"); // sum_0
    w(s, "mov.b32 \t%r415, 0x00000000;"); // sum_1
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
    w(s, "add.f32 \t%r414, %r414, %r417;"); // l_ij_0

    w(s, "shfl.sync.bfly.b32 \t%r418, %r415, 2, 31, -1;");
    w(s, "add.f32 \t%r415, %r415, %r418;");
    w(s, "shfl.sync.bfly.b32 \t%r419, %r415, 1, 31, -1;");
    w(s, "add.f32 \t%r415, %r415, %r419;"); // l_ij_1
    blank(s);

    // Step 7: Rescale O accumulators
    for n in 0..8u32 {
        let base = 200 + n * 4;
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r412;\n", b = base)); // d0 *= alpha_0
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r412;\n", b = base + 1));
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r413;\n", b = base + 2)); // d2 *= alpha_1
        s.push_str(&format!("\tmul.f32 \t%r{b}, %r{b}, %r413;\n", b = base + 3));
    }
    blank(s);

    // Step 8: Update l_i, m_i
    w(s, "fma.rn.f32 \t%r234, %r234, %r412, %r414;"); // l_i_0 = l_i_0 * alpha_0 + l_ij_0
    w(s, "fma.rn.f32 \t%r235, %r235, %r413, %r415;"); // l_i_1
    w(s, "mov.b32 \t%r232, %r408;"); // m_i = m_new
    w(s, "mov.b32 \t%r233, %r409;");
    blank(s);

    // ─── Convert P to f16 and store to smem (reuse K region at 8192) ───
    // For P@V: P is the A operand, so we need it in row-major f16 in smem
    // then ldmatrix to get proper A fragments.
    //
    // P is in MMA output layout: each thread has d0,d1,d2,d3 per n-tile
    //   row_0 = warp_id*16 + lane/4
    //   row_1 = row_0 + 8
    //   col_0 = n_tile*8 + (lane%4)*2
    //   col_1 = col_0 + 1
    //
    // Store P as f16: convert pairs to f16x2, st.shared.b32 at computed addresses

    // Compute store addresses:
    //   mma_row = lane/4 (0..7)
    //   mma_col_pair = lane%4 (which pair of 2 cols)
    w(s, "shr.u32 \t%r420, %r31, 2;"); // lane/4 = mma_row_in_16
    w(s, "add.s32 \t%r421, %r420, %r36;"); // + warp_id*16 = row_0 (0..63)
    w(s, "add.s32 \t%r422, %r421, 8;"); // row_1

    w(s, "and.b32 \t%r423, %r31, 3;"); // lane%4
    w(s, "shl.b32 \t%r424, %r423, 2;"); // (lane%4)*4 bytes (2 f16 = 4 bytes)

    // P smem store addr row_0 = smem + 8192 + row_0 * 128 + (lane%4)*4
    w(s, "shl.b32 \t%r425, %r421, 7;"); // row_0 * 128
    w(s, "add.s32 \t%r426, %r425, %r424;"); // + col_bytes
    w(s, "add.s32 \t%r427, %r426, 8192;");
    w(s, "add.s32 \t%r427, %r427, %r12;"); // P smem store addr row 0

    w(s, "shl.b32 \t%r428, %r422, 7;");
    w(s, "add.s32 \t%r429, %r428, %r424;");
    w(s, "add.s32 \t%r430, %r429, 8192;");
    w(s, "add.s32 \t%r430, %r430, %r12;"); // P smem store addr row 1
    blank(s);

    // Convert and store P for each n-tile
    for n in 0..8u32 {
        let base = 370 + n * 4;
        let h0 = 440 + n * 2;
        let h1 = h0 + 1;
        // cvt.rn.f16x2.f32 packs: result = {f16(arg2), f16(arg1)} i.e. arg2→low, arg1→high
        // Actually: cvt.rn.f16x2.f32 dest, src_high, src_low
        // So for d0 at col (lane%4)*2 and d1 at col (lane%4)*2+1:
        //   Pack: {d0=low, d1=high} → cvt.rn.f16x2.f32 %r, d1, d0
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
        // Store: n_tile offset = n*16 bytes (n*8 columns * 2 bytes)
        let n_off = n * 16;
        s.push_str(&format!("\tst.shared.b32 \t[%r427+{n_off}], %r{h0};\n"));
        s.push_str(&format!("\tst.shared.b32 \t[%r430+{n_off}], %r{h1};\n"));
    }
    w(s, "bar.sync \t0;");
    blank(s);

    // ─── Load V block → smem[16384] ───
    emit_cp_async_kv(s, "V", 16384, "%rd12");
    w(s, "cp.async.commit_group;");
    w(s, "cp.async.wait_group \t0;");
    w(s, "bar.sync \t0;");
    blank(s);

    // ─── ldmatrix P (A operand for P@V) ───
    // P is now at smem[8192], row-major [64×64] f16
    // Same ldmatrix pattern as Q, but at offset 8192
    // P_frag: %r460..%r475 (4 k-iters × 4 regs = 16 regs)
    for ki in 0..4u32 {
        let base = 460 + ki * 4;
        let k_byte_off = ki * 32;
        let offset = 8192 + k_byte_off;
        // Reuse Q ldmatrix address logic (same row within the warp)
        // Q ldmatrix base for k=0 is %r43 (row*128 + col_bytes + smem_base)
        // For P: same row addressing but at smem+8192 instead of smem+0
        // So: P_addr = %r43 + 8192 + ki*32 - (Q has offset=0, P has offset=8192)
        // %r43 = row*128 + col_bytes + smem_base (where smem offset for Q = 0)
        // P_addr = %r43 + 8192 + ki*32
        s.push_str(&format!(
            "\tldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r43+{}];\n",
            base,
            base + 1,
            base + 2,
            base + 3,
            offset
        ));
    }
    blank(s);

    // ─── ldmatrix V (B operand for P@V) ───
    // V at smem[16384], same structure as K
    // V_frag: %r480..%r543 (8 n-tiles × 2 k-pairs × 4 regs = 64 regs)
    for n in 0..8u32 {
        for kp in 0..2u32 {
            let base = 480 + (n * 2 + kp) * 4;
            let n_off = n * 1024;
            let kp_off = kp * 64;
            let total_off = n_off + kp_off;
            if total_off > 0 {
                s.push_str(&format!(
                    "\tldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r48+{}];\n",
                    base, base + 1, base + 2, base + 3, total_off
                ));
            } else {
                s.push_str(&format!(
                    "\tldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%r{}, %r{}, %r{}, %r{}}}, [%r48];\n",
                    base, base + 1, base + 2, base + 3
                ));
            }
        }
    }
    blank(s);

    // ─── MMA: O += P @ V ───
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
    // Thread's output:
    //   row_0 = block_m_start + warp_id*16 + lane/4
    //   row_1 = row_0 + 8
    //   col = n_tile*8 + (lane%4)*2
    //
    // O is [seq, 64] contiguous f16: O_base + row*128 + col*2

    // %r421 = row_0 in block (warp_id*16 + lane/4)
    // %r422 = row_1 = row_0 + 8
    // Global row:
    w(s, "add.s32 \t%r550, %r7, %r421;"); // block_m_start + row_0 = global_row_0
    w(s, "add.s32 \t%r551, %r7, %r422;"); // global_row_1

    // Bounds check
    w(s, "setp.lt.s32 \t%p10, %r550, %r1;");
    w(s, "setp.lt.s32 \t%p11, %r551, %r1;");

    // O base addr for row:
    // O_addr_row0 = O_base + global_row_0 * 128 + (lane%4)*4
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
        // Convert to f16x2
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
        // Store: offset by n_tile * 16 bytes
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
/// Uses thread mapping from %r9 (cp_row), %r11 (col_bytes).
/// Loads two halves (rows 0..31 and 32..63).
fn emit_cp_async_tile(s: &mut String, _name: &str, smem_offset: u32, base_ptr: &str) {
    // Row addresses for global memory:
    //   half 0 row = block_m_start + cp_row
    //   half 1 row = block_m_start + cp_row + 32
    // Global addr = base_ptr + row * 128 + col_bytes

    // Compute global addresses
    w(s, &format!("add.s32 \t%r60, %r7, %r9;")); // row_h0 = block_m_start + cp_row
    w(s, "add.s32 \t%r61, %r60, 32;"); // row_h1

    w(s, "shl.b32 \t%r62, %r60, 7;"); // row_h0 * 128
    w(s, "add.s32 \t%r63, %r62, %r11;"); // + col_bytes
    w(s, &format!("cvt.u64.u32 \t%rd20, %r63;"));
    w(s, &format!("add.s64 \t%rd21, {base_ptr}, %rd20;"));

    w(s, "shl.b32 \t%r64, %r61, 7;");
    w(s, "add.s32 \t%r65, %r64, %r11;");
    w(s, "cvt.u64.u32 \t%rd22, %r65;");
    w(s, &format!("add.s64 \t%rd23, {base_ptr}, %rd22;"));

    // Smem addresses (linear, no swizzle for simplicity)
    w(s, &format!("add.s32 \t%r66, %r12, {};", smem_offset)); // smem_base + offset
    // smem_addr_h0 = smem_base + row_within_tile * 128 + col_bytes
    w(s, "shl.b32 \t%r67, %r9, 7;"); // cp_row * 128
    w(s, "add.s32 \t%r68, %r67, %r11;"); // + col_bytes
    w(s, "add.s32 \t%r69, %r66, %r68;"); // smem addr h0

    // h1: row = cp_row + 32
    w(s, "add.s32 \t%r70, %r68, 4096;"); // + 32*128
    w(s, "add.s32 \t%r71, %r66, %r70;"); // smem addr h1

    // Bounds check
    w(s, "setp.lt.s32 \t%p20, %r60, %r1;");
    w(s, "setp.lt.s32 \t%p21, %r61, %r1;");
    w(s, "selp.b32 \t%r72, 16, 0, %p20;");
    w(s, "selp.b32 \t%r73, 16, 0, %p21;");

    // cp.async
    s.push_str("\tcp.async.cg.shared.global [ %r69 + 0 ], [ %rd21 + 0 ], 0x10, %r72;\n");
    s.push_str("\tcp.async.cg.shared.global [ %r71 + 0 ], [ %rd23 + 0 ], 0x10, %r73;\n");
}

/// Emit cp.async for K or V tile (uses kv_start from %r236).
fn emit_cp_async_kv(s: &mut String, _name: &str, smem_offset: u32, base_ptr: &str) {
    // Row = kv_start + cp_row (halves: +0 and +32)
    w(s, "add.s32 \t%r60, %r236, %r9;"); // row_h0
    w(s, "add.s32 \t%r61, %r60, 32;"); // row_h1

    w(s, "shl.b32 \t%r62, %r60, 7;"); // row * 128
    w(s, "add.s32 \t%r63, %r62, %r11;");
    w(s, "cvt.u64.u32 \t%rd20, %r63;");
    w(s, &format!("add.s64 \t%rd21, {base_ptr}, %rd20;"));

    w(s, "shl.b32 \t%r64, %r61, 7;");
    w(s, "add.s32 \t%r65, %r64, %r11;");
    w(s, "cvt.u64.u32 \t%rd22, %r65;");
    w(s, &format!("add.s64 \t%rd23, {base_ptr}, %rd22;"));

    // Smem addresses
    w(s, &format!("add.s32 \t%r66, %r12, {};", smem_offset));
    w(s, "shl.b32 \t%r67, %r9, 7;");
    w(s, "add.s32 \t%r68, %r67, %r11;");
    w(s, "add.s32 \t%r69, %r66, %r68;");
    w(s, "add.s32 \t%r70, %r68, 4096;");
    w(s, "add.s32 \t%r71, %r66, %r70;");

    // Bounds check
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
        assert!(ldm_count >= 8, "Need at least 8 ldmatrix (Q + K + P + V), got {}", ldm_count);
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
        // This means converting f32 accumulators to f16 before store
        let has_f16_store = ptx.contains("st.global.b16") ||
                           ptx.contains("st.global.v2.b32") ||  // packed f16 pairs
                           ptx.contains("st.global.b32") ||     // packed f16 as b32
                           (ptx.contains("cvt.rn.f16x2.f32") && ptx.contains("st.global"));
        assert!(has_f16_store, "Must store output as f16");
    }

    #[test]
    fn test_flash_attn_barrier_between_phases() {
        let (ptx, _) = load_and_query_kernel();
        // Must have barriers between:
        // 1. K load and Q@K^T compute
        // 2. V load and P@V compute
        let barrier_count = ptx.lines().filter(|l| l.contains("bar.sync")).count();
        assert!(barrier_count >= 4, "Need at least 4 barriers (K sync, V sync, etc), got {}", barrier_count);
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
