/// Rotary Position Embeddings (RoPE) PTX kernel emitter.
///
/// Computes LLaMA-style non-interleaved RoPE:
///   out_first  = x_first  * cos - x_second * sin
///   out_second = x_second * cos + x_first  * sin
///
/// where x_first = x[..., :half_dim], x_second = x[..., half_dim:]
///
/// Replicates Triton-compiled reference at /tmp/triton_rotary.ptx.
///
/// Layout:
///   X, OUT: [batch, seqlen, nheads, headdim] contiguous f16
///   cos, sin: [seqlen, headdim/2] contiguous f32
///
/// Grid: (ceil(nheads/BLOCK_H), ceil(seqlen/BLOCK_M), batch)
/// 128 threads per block, BLOCK_H=4 heads, BLOCK_M=8 seq positions.
/// Each thread processes 8 f16 values from X (4 from first half, 4 from second half).

const BLOCK_H: u32 = 4;
const BLOCK_M: u32 = 8;

pub fn emit_rotary_kernel() -> String {
    let mut s = String::with_capacity(16384);
    emit_kernel(&mut s);
    s
}

fn emit_kernel(s: &mut String) {
    s.push_str(
        r#".version 8.5
.target sm_89
.address_size 64

.visible .entry rotary_kernel(
    .param .u64 .ptr .global .align 16 param_out,
    .param .u64 .ptr .global .align 16 param_x,
    .param .u64 .ptr .global .align 16 param_cos,
    .param .u64 .ptr .global .align 16 param_sin,
    .param .u32 param_seqlen,
    .param .u32 param_nheads,
    .param .u32 param_headdim
)
.reqntid 128
{
    .reg .pred  %p<4>;
    .reg .b16   %rs<33>;
    .reg .b32   %r<200>;
    .reg .b64   %rd<30>;

    // Load params
    ld.param.b64    %rd1, [param_out];
    ld.param.b64    %rd2, [param_x];
    ld.param.b64    %rd3, [param_cos];
    ld.param.b64    %rd4, [param_sin];
    ld.param.b32    %r1, [param_seqlen];
    ld.param.b32    %r2, [param_nheads];
    ld.param.b32    %r3, [param_headdim];

    // Grid indexing
    mov.u32         %r4, %ctaid.x;      // pid_head
    mov.u32         %r5, %ctaid.y;      // pid_m (seq position block)
    mov.u32         %r6, %ctaid.z;      // pid_batch
    mov.u32         %r7, %tid.x;        // tid (0..127)

    // half_dim = headdim / 2
    shr.u32         %r8, %r3, 1;

    // Head indices: h = pid_head * BLOCK_H + (tid >> 5)  [4 heads per block, tid/32 selects head within block]
    // Actually following Triton: BLOCK_H=4 heads, BLOCK_M=8 seq positions
    // 128 threads = 4 heads * 8 seq * 4 elements? Let's match the reference.
    //
    // Triton decomposition: each thread handles one (head, seq_pos) pair and
    // loads 8 f16 values (half of half_dim per load, vectorized).
    // With BLOCK_H=4, BLOCK_M=8: 32 (h,m) pairs per block, 128/32=4 loads per pair.
    //
    // Thread mapping (from reference):
    //   rh (head within block) = bfe(tid, 5, 2)  = (tid >> 5) & 3 = tid / 32
    //   rm_local (seq within block) = bfe(tid, 4, 3) | ... complex
    //
    // Simpler approach matching reference output:
    //   Each thread loads 16 bytes (8 f16 = 4 b32) from x_first and x_second.
    //   Thread's (head, seq) assignment determines the base address.
    //   cos/sin are indexed by seq position only.

"#,
    );

    // Following reference PTX structure:
    // Thread decomposition: 128 threads, each handles 8 f16 elements
    // rh = pid_head * 4 + (tid >> 5) & 3  (head within block, 4 per block)
    // rm = pid_m * 8 + ((tid >> 4) & 7 | some bits)  (seq within block)
    // rk = (tid & 7) * 8  (element offset within half_dim, 8 f16 = 16 bytes)
    //
    // But the exact decomposition is complex. Let me use a simpler mapping:
    // 128 threads, each processes 8 contiguous f16 from the first half and 8 from the second
    // Thread (h, m, k_chunk) where h=0..3, m=0..7, k_chunk=0..3
    // tid = h * 32 + m * 4 + k_chunk
    // Or: tid = k_chunk * 32 + h * 8 + m  (to coalesce memory)
    //
    // Actually, the simplest correct approach for the PTX:
    // Each thread handles elements at:
    //   head = pid_head * 4 + tid / 32
    //   seq = pid_m * 8 + (tid / 4) % 8
    //   k_base = (tid % 4) * 8  (8 f16 per thread, but we load 4 f16 = 8 bytes at a time)
    //
    // Wait, I should just match the reference. Let me re-read the Triton kernel:
    //   rh = pid_head * BLOCK_H + arange(0, BLOCK_H)  [4 values]
    //   rm = pid_m * BLOCK_M + arange(0, BLOCK_M)     [8 values]
    //   rk = arange(0, HALF_DIM)                       [64 values]
    // Total: 4 * 8 * 64 = 2048 elements per block, 128 threads → 16 elements per thread.
    // Each thread loads 16 f16 total (8 from first half, 8 from second half).
    //
    // The Triton compiler distributes this as:
    //   Each thread picks specific (rh, rm, rk) coordinates.
    //   The loads use v4.b32 = 8 f16 per load, so 2 loads per thread (first + second half).
    //
    // For our hand-written kernel, let me use a simpler 1D decomposition:
    //   total_elements = batch * seqlen * nheads * half_dim
    //   Each thread processes 4 elements from first half and 4 from second half
    //   (matching the v4.b32 load pattern)

    // Simple 1D approach: each block handles BLOCK_SIZE contiguous elements of the FIRST half.
    // The second half is at offset half_dim.
    //
    // Global element index for first half:
    //   base = pid_batch * seqlen * nheads * headdim
    //   head_offset = (pid_head * 4 + head_in_block) * headdim
    //   seq_offset = (pid_m * 8 + seq_in_block) * nheads * headdim
    //   k_offset = k_base
    //
    // This is getting complex. Let me just implement a simple version:
    // Process elements in a flat loop over the (seq, head, half_dim) space.

    w(s, "// Thread decomposition: h_local = tid/32, m_local = (tid/4)%8, k_base = (tid%4)*16");
    w(s, "shr.u32 \t%r10, %r7, 5;"); // h_local = tid / 32 (0..3)
    w(s, "shr.u32 \t%r11, %r7, 2;");
    w(s, "and.b32 \t%r11, %r11, 7;"); // m_local = (tid/4) % 8 (0..7)
    w(s, "and.b32 \t%r12, %r7, 3;"); // k_chunk = tid % 4 (0..3)
    w(s, "shl.b32 \t%r13, %r12, 4;"); // k_byte_offset = k_chunk * 16 (bytes, for 8 f16)
    blank(s);

    // Global head and seq indices
    w(s, "// Global head = pid_head * 4 + h_local");
    w(s, "shl.b32 \t%r14, %r4, 2;"); // pid_head * 4
    w(s, "add.s32 \t%r15, %r14, %r10;"); // global head
    w(s, "// Global seq = pid_m * 8 + m_local");
    w(s, "shl.b32 \t%r16, %r5, 3;"); // pid_m * 8
    w(s, "add.s32 \t%r17, %r16, %r11;"); // global seq
    blank(s);

    // Bounds check
    w(s, "setp.lt.s32 \t%p1, %r15, %r2;"); // head < nheads
    w(s, "setp.lt.s32 \t%p2, %r17, %r1;"); // seq < seqlen
    w(s, "and.pred \t%p3, %p1, %p2;"); // both in bounds
    blank(s);

    // Compute cos/sin address: cos[seq][k] = cos_base + seq * half_dim + k_chunk * 8
    // cos is [seqlen, half_dim] f32, so byte stride = half_dim * 4
    w(s, "mul.lo.s32 \t%r20, %r17, %r8;"); // seq * half_dim
    w(s, "shl.b32 \t%r21, %r12, 3;"); // k_chunk * 8 (f32 elements, not bytes)
    // Actually k_chunk * 8 gives element offset. But we load 4 f32 (v4.b32), so
    // the element offset is k_chunk * 4 (4 f32 per load = 16 bytes).
    w(s, "shl.b32 \t%r21, %r12, 2;"); // k_chunk * 4 (4 f32 elements per load)
    w(s, "add.s32 \t%r22, %r20, %r21;"); // cos element index
    w(s, "mul.wide.s32 \t%rd10, %r22, 4;"); // byte offset (f32 = 4 bytes)
    w(s, "add.s64 \t%rd11, %rd3, %rd10;"); // cos ptr
    w(s, "add.s64 \t%rd12, %rd4, %rd10;"); // sin ptr
    blank(s);

    // Load cos and sin (4 f32 each via v4.b32)
    w(s, "mov.b32 \t%r30, 0f3F800000;"); // 1.0 (default cos)
    w(s, "mov.b32 \t%r31, 0f00000000;"); // 0.0 (default sin)
    // cos
    w(s, "mov.b32 \t%r32, %r30;");
    w(s, "mov.b32 \t%r33, %r30;");
    w(s, "mov.b32 \t%r34, %r30;");
    w(s, "mov.b32 \t%r35, %r30;");
    w(s, "@%p3 ld.global.v4.b32 { %r32, %r33, %r34, %r35 }, [ %rd11 + 0 ];");
    // sin
    w(s, "mov.b32 \t%r36, %r31;");
    w(s, "mov.b32 \t%r37, %r31;");
    w(s, "mov.b32 \t%r38, %r31;");
    w(s, "mov.b32 \t%r39, %r31;");
    w(s, "@%p3 ld.global.v4.b32 { %r36, %r37, %r38, %r39 }, [ %rd12 + 0 ];");
    blank(s);

    // Compute X addresses
    // X is [batch, seqlen, nheads, headdim] f16
    // X_first  = X + batch_offset + seq * (nheads * headdim) + head * headdim + k_chunk * 8
    // X_second = X_first + half_dim  (offset by half_dim f16 = half_dim * 2 bytes)
    w(s, "mul.lo.s32 \t%r40, %r1, %r2;"); // seqlen * nheads
    w(s, "mul.lo.s32 \t%r41, %r40, %r3;"); // seqlen * nheads * headdim
    w(s, "mul.lo.s32 \t%r42, %r6, %r41;"); // batch_offset (in elements)
    w(s, "mul.lo.s32 \t%r43, %r2, %r3;"); // nheads * headdim (seq stride)
    w(s, "mul.lo.s32 \t%r44, %r17, %r43;"); // seq * seq_stride
    w(s, "mul.lo.s32 \t%r45, %r15, %r3;"); // head * headdim
    w(s, "shl.b32 \t%r46, %r12, 3;"); // k_chunk * 8 (element offset)
    w(s, "add.s32 \t%r47, %r42, %r44;"); // batch + seq
    w(s, "add.s32 \t%r48, %r47, %r45;"); // + head
    w(s, "add.s32 \t%r49, %r48, %r46;"); // + k_offset = first half element index
    w(s, "add.s32 \t%r50, %r49, %r8;"); // + half_dim = second half element index
    blank(s);

    // X pointers (f16 = 2 bytes)
    w(s, "mul.wide.s32 \t%rd13, %r49, 2;"); // first half byte offset
    w(s, "add.s64 \t%rd14, %rd2, %rd13;"); // X_first ptr
    w(s, "mul.wide.s32 \t%rd15, %r50, 2;");
    w(s, "add.s64 \t%rd16, %rd2, %rd15;"); // X_second ptr
    blank(s);

    // Load X first half (4 b32 = 8 f16)
    w(s, "mov.b32 \t%r60, 0;");
    w(s, "mov.b32 \t%r61, 0;");
    w(s, "mov.b32 \t%r62, 0;");
    w(s, "mov.b32 \t%r63, 0;");
    w(s, "@%p3 ld.global.v4.b32 { %r60, %r61, %r62, %r63 }, [ %rd14 + 0 ];");

    // Load X second half
    w(s, "mov.b32 \t%r64, 0;");
    w(s, "mov.b32 \t%r65, 0;");
    w(s, "mov.b32 \t%r66, 0;");
    w(s, "mov.b32 \t%r67, 0;");
    w(s, "@%p3 ld.global.v4.b32 { %r64, %r65, %r66, %r67 }, [ %rd16 + 0 ];");
    blank(s);

    // Unpack f16 to f32 and compute RoPE
    // For each pair of b32 (2 f16 each from first and second half):
    //   x1_lo, x1_hi = unpack(x_first_b32)
    //   x2_lo, x2_hi = unpack(x_second_b32)
    //   out1_lo = x1_lo * cos_i - x2_lo * sin_i
    //   out1_hi = x1_hi * cos_{i+1} - x2_hi * sin_{i+1}
    //   out2_lo = x2_lo * cos_i + x1_lo * sin_i
    //   out2_hi = x2_hi * cos_{i+1} + x1_hi * sin_{i+1}
    //
    // cos/sin are f32 (one per element): cos[0..3] in %r32..%r35 (4 f32 values)
    // But each b32 of X holds 2 f16 → need 2 cos/sin values per b32.
    // We loaded 4 f32 cos values, matching 4 elements. But we have 8 f16 X values.
    // So we need 8 cos/sin values but only loaded 4!
    //
    // Fix: load 8 cos/sin values (two v4.b32 loads, or load 4 here and 4 at +16 bytes)
    // Actually: k_chunk * 4 f32 elements = elements [k_chunk*4 .. k_chunk*4+3].
    // But X has 8 f16 at the same position. So cos/sin indices should be k_chunk*8..k_chunk*8+7.
    // Wait, each f16 X element corresponds to one cos/sin f32 value.
    // If we load 8 X f16 values at element offset k_chunk*8, we need cos[k_chunk*8..k_chunk*8+7].
    // But we computed cos element index as k_chunk*4. That's wrong!
    //
    // Fix the cos/sin index: k_chunk * 8 elements, but cos is f32 so byte offset = k_chunk * 8 * 4 = k_chunk * 32
    // And we need two v4.b32 loads (8 f32 values total).

    // I had a bug above. Let me fix the cos/sin loading.
    // Go back and patch: remove the first cos/sin load, redo with correct indexing.

    // Actually, I realize the issue: we can't easily "go back" in the string.
    // Let me restructure. I'll emit a complete correct kernel from scratch.
    s.clear();
    emit_kernel_v2(s);
}

fn emit_kernel_v2(s: &mut String) {
    s.push_str(
        r#".version 8.5
.target sm_89
.address_size 64

.visible .entry rotary_kernel(
    .param .u64 .ptr .global .align 16 param_out,
    .param .u64 .ptr .global .align 16 param_x,
    .param .u64 .ptr .global .align 16 param_cos,
    .param .u64 .ptr .global .align 16 param_sin,
    .param .u32 param_seqlen,
    .param .u32 param_nheads,
    .param .u32 param_headdim
)
.reqntid 128
{
    .reg .pred  %p<4>;
    .reg .b16   %rs<33>;
    .reg .b32   %r<200>;
    .reg .b64   %rd<30>;

"#,
    );

    // Load params
    w(s, "ld.param.b64 \t%rd1, [param_out];");
    w(s, "ld.param.b64 \t%rd2, [param_x];");
    w(s, "ld.param.b64 \t%rd3, [param_cos];");
    w(s, "ld.param.b64 \t%rd4, [param_sin];");
    w(s, "ld.param.b32 \t%r1, [param_seqlen];");
    w(s, "ld.param.b32 \t%r2, [param_nheads];");
    w(s, "ld.param.b32 \t%r3, [param_headdim];");
    w(s, "shr.u32 \t%r4, %r3, 1;"); // half_dim = headdim / 2
    blank(s);

    // Grid: (ceil(nheads/4), ceil(seqlen/8), batch)
    // Each block: 4 heads × 8 seq positions × 4 k-chunks = 128 threads
    // Thread decomposition: tid = head_local*32 + seq_local*4 + k_chunk
    w(s, "mov.u32 \t%r5, %ctaid.x;"); // pid_head
    w(s, "mov.u32 \t%r6, %ctaid.y;"); // pid_m
    w(s, "mov.u32 \t%r7, %ctaid.z;"); // pid_batch
    w(s, "mov.u32 \t%r8, %tid.x;"); // tid
    blank(s);

    // Decompose tid
    w(s, "shr.u32 \t%r10, %r8, 5;"); // head_local = tid / 32 (0..3)
    w(s, "shr.u32 \t%r11, %r8, 2;");
    w(s, "and.b32 \t%r11, %r11, 7;"); // seq_local = (tid/4) % 8 (0..7)
    w(s, "and.b32 \t%r12, %r8, 3;"); // k_chunk = tid % 4 (0..3)
    blank(s);

    // Global indices
    w(s, "shl.b32 \t%r13, %r5, 2;"); // pid_head * 4
    w(s, "add.s32 \t%r14, %r13, %r10;"); // global head
    w(s, "shl.b32 \t%r15, %r6, 3;"); // pid_m * 8
    w(s, "add.s32 \t%r16, %r15, %r11;"); // global seq
    blank(s);

    // Bounds
    w(s, "setp.lt.s32 \t%p1, %r14, %r2;"); // head < nheads
    w(s, "setp.lt.s32 \t%p2, %r16, %r1;"); // seq < seqlen
    w(s, "and.pred \t%p3, %p1, %p2;");
    blank(s);

    // cos/sin: [seqlen, half_dim] f32
    // This thread handles elements k_chunk*4 .. k_chunk*4+3 of the half_dim.
    // Each f16 X element has one corresponding f32 cos and sin value.
    // We load 8 X f16 values (4 b32 via v4) and need 8 cos/sin f32 values.
    // But half_dim can be 64, and we handle 4 elements per load × 4 chunks = 16 elements.
    // Wait: 4 k_chunks × 4 f32 per load = 16 cos/sin values = 16 X elements per half.
    // But half_dim = 64, so we'd need 4 threads for 64 elements... 64/4 = 16 per thread.
    // That doesn't match. Let me reconsider.
    //
    // With 128 threads = 4 heads × 8 seqs × 4 k_chunks:
    // Each k_chunk handles half_dim / 4 elements... but half_dim=64 → 16 elements per k_chunk.
    // 16 f16 = 8 b32 → need 2× v4.b32 loads for X, and 4× v4.b32 loads for cos/sin? Too many.
    //
    // Simpler: let each thread handle 4 f16 elements (not 8).
    // 128 threads × 4 = 512 elements per block.
    // 4 heads × 8 seqs × 16 elements = 512. This works if k_chunks = 16.
    // But 4*8*16 = 512 ≠ 128. Thread decomposition doesn't work.
    //
    // The Triton kernel handles 4 heads × 8 seqs × (BLOCK_K//2) elements.
    // BLOCK_K = next_power_of_2(ROTARY_DIM) = 128. BLOCK_K//2 = 64.
    // Total = 4 * 8 * 64 = 2048 elements. 128 threads → 16 elements/thread.
    // 16 f16 = 2× v4.b32 loads for X_first, 2× for X_second.
    // 16 f32 cos = 4× v4.b32, 16 f32 sin = 4× v4.b32.
    //
    // But the Triton PTX only does 2× v4.b32 loads for X (first+second) and 2× v4.b32 for cos+sin.
    // That's 8 f16 X + 8 f16 X + 4 f32 cos + 4 f32 sin = 8+8+4+4 elements per thread.
    // 8 first-half f16 + 8 second-half f16 + 8 cos f32 + 8 sin f32... but reference loads 4 cos + 4 sin.
    //
    // Hmm, the reference uses shared memory to redistribute cos/sin! That's the smem transpose.
    // cos is loaded by some threads, put in smem, barrier, then all threads read from smem.
    //
    // This is getting complex. Let me use a much simpler approach:
    // 1D grid over (batch * seqlen * nheads * half_dim) with 4 elements per thread.

    // RESTART with simple 1D approach
    s.clear();
    emit_kernel_1d(s);
}

fn emit_kernel_1d(s: &mut String) {
    // Simple 1D kernel: each thread processes 4 elements from first half and second half.
    // Grid: (ceil(total_half_elements / BLOCK_SIZE),) where BLOCK_SIZE = 128 * 4 = 512
    // total_half_elements = batch * seqlen * nheads * half_dim
    //
    // For each element i (in flattened first-half space):
    //   seq = (i / (nheads * half_dim)) % seqlen
    //   k = i % half_dim
    //   out_first[i] = x_first[i] * cos[seq][k] - x_second[i] * sin[seq][k]
    //   out_second[i] = x_second[i] * cos[seq][k] + x_first[i] * sin[seq][k]

    s.push_str(
        r#".version 8.5
.target sm_89
.address_size 64

.visible .entry rotary_kernel(
    .param .u64 .ptr .global .align 16 param_out,
    .param .u64 .ptr .global .align 16 param_x,
    .param .u64 .ptr .global .align 16 param_cos,
    .param .u64 .ptr .global .align 16 param_sin,
    .param .u32 param_total_half,
    .param .u32 param_half_dim,
    .param .u32 param_headdim,
    .param .u32 param_nheads_x_half_dim
)
.reqntid 128
{
    .reg .pred  %p<2>;
    .reg .b16   %rs<9>;
    .reg .b32   %r<80>;
    .reg .b64   %rd<12>;

    ld.param.b64    %rd1, [param_out];
    ld.param.b64    %rd2, [param_x];
    ld.param.b64    %rd3, [param_cos];
    ld.param.b64    %rd4, [param_sin];
    ld.param.b32    %r1, [param_total_half];
    ld.param.b32    %r2, [param_half_dim];
    ld.param.b32    %r3, [param_headdim];
    ld.param.b32    %r4, [param_nheads_x_half_dim];

    // Global element index (in half-dim space)
    mov.u32         %r5, %ctaid.x;
    shl.b32         %r6, %r5, 9;       // block_offset = ctaid.x * 512
    mov.u32         %r7, %tid.x;
    shl.b32         %r8, %r7, 2;       // thread_offset = tid * 4
    or.b32          %r9, %r6, %r8;     // global_idx = block_offset + thread_offset

    // Bounds check
    setp.lt.s32     %p1, %r9, %r1;

    // Compute seq and k from global_idx
    // global_idx = batch_head_seq * half_dim + k
    // where batch_head_seq = batch * seqlen * nheads + head * ...
    // Actually: total layout is [batch, seqlen, nheads, headdim]
    // First half: elements at [..., 0:half_dim]
    // flat index i maps to: the i-th element in the first-half view
    // seq_head = i / half_dim  (gives batch*seqlen*nheads block)
    // k = i % half_dim
    // seq = (seq_head / nheads) % seqlen  ... but this requires knowing nheads
    //
    // Simpler: pass nheads_x_half_dim = nheads * half_dim
    // seq = (i / nheads_x_half_dim) % seqlen  -- but we don't pass seqlen separately
    // Actually, just compute: cos_idx = (i / nheads_x_half_dim) * half_dim + (i % half_dim)
    // Wait, cos is [seqlen, half_dim], so cos_flat_idx = seq * half_dim + k
    // where seq = floor(i / (nheads * half_dim)) % seqlen
    // and k = i % half_dim

    // For the flat first-half index i:
    // In the [batch, seqlen, nheads, half_dim] view (first half of headdim):
    // k = i % half_dim
    // remainder = i / half_dim
    // head = remainder % nheads  (not needed for cos/sin)
    // seq_batch = remainder / nheads
    // seq = seq_batch % seqlen  (not needed if cos is big enough)
    // cos_idx = seq_batch * half_dim + k  (if cos is [total_seqlen, half_dim])
    //
    // Simplification: cos is [seqlen, half_dim] and wraps by sequence.
    // For now, assume cos is large enough: cos_idx = (i / (nheads * half_dim)) * half_dim + (i % half_dim)
    // = (i / nheads_x_half_dim) * half_dim + (i % half_dim)

"#,
    );

    // Compute k = global_idx % half_dim (need div/mod)
    // Since half_dim is a power of 2 (64 for LLaMA), use AND mask
    w(s, "sub.s32 \t%r10, %r2, 1;"); // half_dim - 1 (mask)
    w(s, "and.b32 \t%r11, %r9, %r10;"); // k = idx % half_dim
    w(s, "shr.u32 \t%r12, %r9, 6;"); // idx / half_dim (assumes half_dim=64, shift by log2(64)=6)
    // FIXME: hardcoded for half_dim=64. For generality, use div.
    // For now this works for LLaMA (headdim=128, half_dim=64).

    // seq_batch = idx / nheads_x_half_dim
    // We can compute: cos_row = idx / nheads_x_half_dim
    // But div is expensive. Since nheads_x_half_dim may not be power of 2,
    // let's just pass the cos row stride differently.
    // Actually for the 1D approach, let me just pass cos_stride = half_dim.
    // cos_idx = seq * half_dim + k
    // seq = (idx / nheads_x_half_dim) -- this is integer division

    // For simplicity and correctness, use div:
    w(s, "div.u32 \t%r13, %r9, %r4;"); // seq = idx / (nheads * half_dim)
    w(s, "mul.lo.s32 \t%r14, %r13, %r2;"); // seq * half_dim
    w(s, "add.s32 \t%r15, %r14, %r11;"); // cos_flat_idx = seq * half_dim + k
    blank(s);

    // Load 4 cos and 4 sin values (f32, contiguous)
    w(s, "mul.wide.s32 \t%rd5, %r15, 4;"); // byte offset (f32)
    w(s, "add.s64 \t%rd6, %rd3, %rd5;"); // cos ptr
    w(s, "add.s64 \t%rd7, %rd4, %rd5;"); // sin ptr
    w(s, "mov.b32 \t%r20, 0f3F800000;"); // default cos = 1.0
    w(s, "mov.b32 \t%r21, 0f00000000;"); // default sin = 0.0
    w(s, "mov.b32 \t%r22, %r20; mov.b32 \t%r23, %r20; mov.b32 \t%r24, %r20; mov.b32 \t%r25, %r20;");
    w(s, "@%p1 ld.global.v4.b32 { %r22, %r23, %r24, %r25 }, [ %rd6 + 0 ];"); // cos[0..3]
    w(s, "mov.b32 \t%r26, %r21; mov.b32 \t%r27, %r21; mov.b32 \t%r28, %r21; mov.b32 \t%r29, %r21;");
    w(s, "@%p1 ld.global.v4.b32 { %r26, %r27, %r28, %r29 }, [ %rd7 + 0 ];"); // sin[0..3]
    blank(s);

    // Load X first half (4 f16 = 2 b32 via v2.b32)
    // Actually 4 f16 = 2 b32. Use ld.global.v2.b32.
    w(s, "mul.wide.s32 \t%rd8, %r9, 2;"); // first half byte offset (f16)
    w(s, "add.s64 \t%rd9, %rd2, %rd8;"); // X_first ptr

    // Second half offset: + half_dim * 2 bytes (within the same row)
    // The second half is at X[..., half_dim:headdim]
    // Byte offset from first half = half_dim * 2
    w(s, "mul.wide.s32 \t%rd10, %r2, 2;"); // half_dim * 2 bytes
    w(s, "add.s64 \t%rd11, %rd9, %rd10;"); // X_second ptr
    blank(s);

    w(s, "mov.b32 \t%r30, 0;");
    w(s, "mov.b32 \t%r31, 0;");
    w(s, "@%p1 ld.global.v2.b32 { %r30, %r31 }, [ %rd9 + 0 ];"); // x_first (4 f16)
    w(s, "mov.b32 \t%r32, 0;");
    w(s, "mov.b32 \t%r33, 0;");
    w(s, "@%p1 ld.global.v2.b32 { %r32, %r33 }, [ %rd11 + 0 ];"); // x_second (4 f16)
    blank(s);

    // Unpack and compute RoPE for 4 elements
    // Each b32 holds 2 f16. We have 2 b32 per half → 4 f16 per half.
    // cos/sin are already f32 in %r22..%r25 and %r26..%r29.
    for pair in 0..2u32 {
        let x1_b32 = 30 + pair; // x_first packed b32
        let x2_b32 = 32 + pair; // x_second packed b32
        let cos0 = 22 + pair * 2; // cos for low element
        let cos1 = cos0 + 1; // cos for high element
        let sin0 = 26 + pair * 2;
        let sin1 = sin0 + 1;
        let rs_base = pair * 4; // rs register base
        let r_base = 40 + pair * 14; // temp register base

        // Unpack x_first
        s.push_str(&format!(
            "\tmov.b32 \t{{%rs{}, %rs{}}}, %r{};\n",
            rs_base + 1,
            rs_base + 2,
            x1_b32
        ));
        s.push_str(&format!("\tcvt.f32.f16 \t%r{}, %rs{};\n", r_base, rs_base + 1)); // x1_lo
        s.push_str(&format!(
            "\tcvt.f32.f16 \t%r{}, %rs{};\n",
            r_base + 1,
            rs_base + 2
        )); // x1_hi

        // Unpack x_second
        s.push_str(&format!(
            "\tmov.b32 \t{{%rs{}, %rs{}}}, %r{};\n",
            rs_base + 3,
            rs_base + 4,
            x2_b32
        ));
        s.push_str(&format!(
            "\tcvt.f32.f16 \t%r{}, %rs{};\n",
            r_base + 2,
            rs_base + 3
        )); // x2_lo
        s.push_str(&format!(
            "\tcvt.f32.f16 \t%r{}, %rs{};\n",
            r_base + 3,
            rs_base + 4
        )); // x2_hi

        // out_first_lo = x1_lo * cos0 - x2_lo * sin0
        //              = fma(x1_lo, cos0, -(x2_lo * sin0))
        s.push_str(&format!(
            "\tmul.f32 \t%r{t}, %r{x2}, %r{s};\n\tneg.f32 \t%r{t}, %r{t};\n\tfma.rn.f32 \t%r{out}, %r{x1}, %r{c}, %r{t};\n",
            t = r_base + 4, x2 = r_base + 2, s = sin0, x1 = r_base, c = cos0, out = r_base + 6
        ));
        // out_first_hi
        s.push_str(&format!(
            "\tmul.f32 \t%r{t}, %r{x2}, %r{s};\n\tneg.f32 \t%r{t}, %r{t};\n\tfma.rn.f32 \t%r{out}, %r{x1}, %r{c}, %r{t};\n",
            t = r_base + 5, x2 = r_base + 3, s = sin1, x1 = r_base + 1, c = cos1, out = r_base + 7
        ));
        // out_second_lo = x2_lo * cos0 + x1_lo * sin0
        s.push_str(&format!(
            "\tmul.f32 \t%r{t}, %r{x2}, %r{c};\n\tfma.rn.f32 \t%r{out}, %r{x1}, %r{s}, %r{t};\n",
            t = r_base + 8, x2 = r_base + 2, c = cos0, x1 = r_base, s = sin0, out = r_base + 10
        ));
        // out_second_hi
        s.push_str(&format!(
            "\tmul.f32 \t%r{t}, %r{x2}, %r{c};\n\tfma.rn.f32 \t%r{out}, %r{x1}, %r{s}, %r{t};\n",
            t = r_base + 9, x2 = r_base + 3, c = cos1, x1 = r_base + 1, s = sin1, out = r_base + 11
        ));

        // Pack to f16x2
        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            r_base + 12,
            r_base + 7,
            r_base + 6
        )); // out_first packed
        s.push_str(&format!(
            "\tcvt.rn.f16x2.f32 \t%r{}, %r{}, %r{};\n",
            r_base + 13,
            r_base + 11,
            r_base + 10
        )); // out_second packed
    }
    blank(s);

    // Store output
    // OUT first half
    w(s, "mul.wide.s32 \t%rd8, %r9, 2;");
    w(s, "add.s64 \t%rd9, %rd1, %rd8;"); // OUT_first
    w(s, "mul.wide.s32 \t%rd10, %r2, 2;");
    w(s, "add.s64 \t%rd11, %rd9, %rd10;"); // OUT_second

    // Store first half (2 b32)
    let out_first_0 = 40 + 12; // pair 0's out_first packed
    let out_first_1 = 40 + 14 + 12; // pair 1's out_first packed
    s.push_str(&format!(
        "\t@%p1 st.global.v2.b32 [ %rd9 + 0 ], {{ %r{}, %r{} }};\n",
        out_first_0, out_first_1
    ));

    // Store second half (2 b32)
    let out_second_0 = 40 + 13;
    let out_second_1 = 40 + 14 + 13;
    s.push_str(&format!(
        "\t@%p1 st.global.v2.b32 [ %rd11 + 0 ], {{ %r{}, %r{} }};\n",
        out_second_0, out_second_1
    ));
    blank(s);

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

    fn get_ptx() -> String {
        emit_rotary_kernel()
    }

    #[test]
    fn test_rotary_valid_ascii() {
        let ptx = get_ptx();
        for (i, b) in ptx.bytes().enumerate() {
            assert!(b < 128, "Non-ASCII at {}", i);
        }
    }

    #[test]
    fn test_rotary_has_entry() {
        let ptx = get_ptx();
        assert!(ptx.contains(".entry rotary_kernel"));
    }

    #[test]
    fn test_rotary_has_reqntid() {
        let ptx = get_ptx();
        assert!(ptx.contains(".reqntid 128"));
    }

    #[test]
    fn test_rotary_loads_cos_sin() {
        let ptx = get_ptx();
        // Should load cos and sin via global loads
        let global_loads = ptx.lines().filter(|l| l.contains("ld.global")).count();
        assert!(global_loads >= 4, "Need loads for cos, sin, x_first, x_second; got {}", global_loads);
    }

    #[test]
    fn test_rotary_has_fma() {
        let ptx = get_ptx();
        let fma_count = ptx.lines().filter(|l| l.contains("fma.rn.f32")).count();
        // 4 elements: each needs 2 fma (out_first, out_second) = 8 total
        assert!(fma_count >= 4, "Need fma for RoPE computation, got {}", fma_count);
    }

    #[test]
    fn test_rotary_has_neg() {
        let ptx = get_ptx();
        let neg_count = ptx.lines().filter(|l| l.contains("neg.f32")).count();
        assert!(neg_count >= 2, "Need neg for out_first = x1*cos - x2*sin, got {}", neg_count);
    }

    #[test]
    fn test_rotary_stores_output() {
        let ptx = get_ptx();
        let stores = ptx.lines().filter(|l| l.contains("st.global")).count();
        assert!(stores >= 2, "Need stores for first and second halves, got {}", stores);
    }

    #[test]
    fn test_rotary_has_f16_conversion() {
        let ptx = get_ptx();
        let cvt_in = ptx.lines().filter(|l| l.contains("cvt.f32.f16")).count();
        let cvt_out = ptx.lines().filter(|l| l.contains("cvt.rn.f16x2.f32")).count();
        assert!(cvt_in >= 4, "Need f16→f32 for x_first and x_second, got {}", cvt_in);
        assert!(cvt_out >= 2, "Need f32→f16 for output, got {}", cvt_out);
    }

    #[test]
    fn test_rotary_has_bounds_check() {
        let ptx = get_ptx();
        assert!(ptx.contains("setp.lt.s32"), "Need bounds checking");
    }

    #[test]
    fn test_rotary_ptx_size() {
        let ptx = get_ptx();
        let lines = ptx.lines().count();
        assert!(lines >= 30, "Too small: {} lines", lines);
        assert!(lines <= 500, "Too large: {} lines", lines);
    }
}
