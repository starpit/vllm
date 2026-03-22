/// Rotary Position Embeddings (RoPE) PTX kernel emitter.
///
/// Produces PTX that is instruction-by-instruction equivalent to the
/// Triton-compiled reference at `/tmp/triton_rotary.ptx`, with only
/// param names changed (and workspace pointers removed).
///
/// Layout:
///   X, OUT: [batch, seqlen, nheads, headdim] contiguous f16
///   cos, sin: [seqlen, headdim/2] contiguous f32
///
/// Grid: (ceil(nheads/4), ceil(seqlen/8), batch)
/// 128 threads per block, BLOCK_H=4 heads, BLOCK_M=8 seq positions.
///
/// Params (10 total, matching Triton minus 2 workspace ptrs):
///   param_out, param_x, param_cos, param_sin (u64 ptrs)
///   param_seqlen, param_nheads (u32)
///   param_stride_out_seqlen, param_stride_out_nheads, param_stride_out_headdim (u32)
///   param_stride_x_seqlen (u32)

pub fn emit_rotary_kernel() -> String {
    let mut s = String::with_capacity(32768);
    emit_kernel(&mut s);
    s
}

fn emit_kernel(s: &mut String) {
    // Header - matches reference exactly (version 8.7, sm_89)
    s.push_str(
        "\
.version 8.7
.target sm_89
.address_size 64

.extern .shared .align 16 .b8 global_smem[];

.visible .entry rotary_kernel(
\t.param .u64 .ptr .global .align 1 param_out,
\t.param .u64 .ptr .global .align 1 param_x,
\t.param .u64 .ptr .global .align 1 param_cos,
\t.param .u64 .ptr .global .align 1 param_sin,
\t.param .u32 param_seqlen,
\t.param .u32 param_nheads,
\t.param .u32 param_stride_out_seqlen,
\t.param .u32 param_stride_out_nheads,
\t.param .u32 param_stride_out_headdim,
\t.param .u32 param_stride_x_seqlen
)
.reqntid 128
{
\t.reg .pred \t%p<5>;
\t.reg .b16 \t%rs<33>;
\t.reg .b32 \t%r<236>;
\t.reg .b64 \t%rd<29>;

");

    // Line 37-38: Load OUT and X base pointers
    w(s, "ld.param.b64 \t%rd11, [param_out];");
    w(s, "ld.param.b64 \t%rd12, [param_x];");

    // Line 41: pid_head = ctaid.x
    w(s, "mov.u32 \t%r43, %ctaid.x;");
    // Line 42: Load COS ptr
    w(s, "ld.param.b64 \t%rd13, [param_cos];");
    // Line 44: pid_m = ctaid.y
    w(s, "mov.u32 \t%r44, %ctaid.y;");
    // Line 46: pid_batch = ctaid.z
    w(s, "mov.u32 \t%r45, %ctaid.z;");
    // Line 47: Load SIN ptr
    w(s, "ld.param.b64 \t%rd14, [param_sin];");

    // Line 49: pid_head * 4 (shl 2)
    w(s, "shl.b32 \t%r46, %r43, 2;");
    // Line 50: Load seqlen
    w(s, "ld.param.b32 \t%r47, [param_seqlen];");
    // Line 51: Load nheads
    w(s, "ld.param.b32 \t%r48, [param_nheads];");

    // Line 53: tid.x
    w(s, "mov.u32 \t%r49, %tid.x;");
    // Line 54: Load stride_out_seqlen
    w(s, "ld.param.b32 \t%r50, [param_stride_out_seqlen];");

    // Line 55: bfe.u32 %r51, %r49, 3, 2 -- extract bits [4:3] of tid (head within block)
    w(s, "bfe.u32 \t%r51, %r49, 3, 2;");
    // Line 56: Load stride_out_nheads
    w(s, "ld.param.b32 \t%r52, [param_stride_out_nheads];");

    // Line 58: or -- global head index = head_in_block | (pid_head * 4)
    w(s, "or.b32 \t%r53, %r51, %r46;");
    // Line 59: Load stride_out_headdim
    w(s, "ld.param.b32 \t%r54, [param_stride_out_headdim];");

    // Line 61: pid_m * 8
    w(s, "shl.b32 \t%r55, %r44, 3;");
    // Line 62: Load stride_x_seqlen
    w(s, "ld.param.b32 \t%r56, [param_stride_x_seqlen];");

    // Line 64-66: Thread decomposition for seq indices
    w(s, "bfe.u32 \t%r57, %r49, 4, 3;");  // bits [6:4] of tid
    w(s, "and.b32 \t%r58, %r49, 96;");     // tid & 96 = bits [6:5]
    w(s, "bfe.u32 \t%r59, %r49, 5, 2;");   // bits [6:5] of tid

    // Line 68-69: seq indices
    w(s, "or.b32 \t%r60, %r57, %r55;");    // seq index for cos/sin load
    w(s, "or.b32 \t%r61, %r59, %r55;");    // seq index for X load

    // Line 71: Bounds check for cos/sin: seq < seqlen
    w(s, "setp.lt.s32 \t%p1, %r60, %r47;");

    // Line 73-76: k-index computation
    w(s, "shl.b32 \t%r62, %r49, 2;");      // tid * 4
    w(s, "and.b32 \t%r63, %r62, 60;");     // (tid*4) & 60 = low 4 bits * 4
    w(s, "and.b32 \t%r64, %r49, 7;");      // tid & 7
    w(s, "shl.b32 \t%r65, %r64, 3;");      // (tid & 7) * 8

    // Line 78-84: cos address computation
    w(s, "shl.b32 \t%r66, %r60, 6;");      // seq * 64 (HALF_DIM)
    w(s, "mul.wide.u32 \t%rd15, %r66, 4;"); // byte offset (f32)
    w(s, "add.s64 \t%rd16, %rd13, %rd15;"); // COS + seq*64*4
    w(s, "mul.wide.u32 \t%rd17, %r63, 4;"); // k offset in bytes
    w(s, "add.s64 \t%rd1, %rd16, %rd17;");  // final cos ptr

    // Line 85: default cos value = 1.0f (1065353216 = 0x3F800000)
    w(s, "mov.b32 \t%r5, 1065353216;");

    // Line 87-93: Load cos (4 x f32 via v4.b32, predicated)
    w(s, "mov.u32 %r1, %r5;");
    w(s, "mov.u32 %r2, %r5;");
    w(s, "mov.u32 %r3, %r5;");
    w(s, "mov.u32 %r4, %r5;");
    w(s, "@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];");

    // Line 95-97: sin address
    w(s, "add.s64 \t%rd18, %rd14, %rd15;");
    w(s, "add.s64 \t%rd2, %rd18, %rd17;");

    // Line 98: default sin value = 0
    w(s, "mov.b32 \t%r10, 0;");

    // Line 100-106: Load sin (4 x f32 via v4.b32, predicated)
    w(s, "mov.u32 %r6, %r10;");
    w(s, "mov.u32 %r7, %r10;");
    w(s, "mov.u32 %r8, %r10;");
    w(s, "mov.u32 %r9, %r10;");
    w(s, "@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];");

    // Line 108-114: X base address computation
    w(s, "mul.lo.s32 \t%r67, %r45, %r47;");  // pid_batch * seqlen
    w(s, "mul.lo.s32 \t%r68, %r67, %r48;");  // * nheads
    w(s, "shl.b32 \t%r69, %r68, 7;");        // * 128 (headdim, ROTARY_DIM)
    w(s, "mul.wide.s32 \t%rd19, %r69, 2;");  // byte offset (f16 = 2 bytes)
    w(s, "add.s64 \t%rd20, %rd12, %rd19;");  // X + batch_offset
    w(s, "add.s64 \t%rd21, %rd11, %rd19;");  // OUT + batch_offset

    // X ptr computation: head * stride_x_nheads + seq * stride_x_seqlen
    // stride_x_nheads = headdim = 128 (hardcoded, shift by 7)
    // stride_x_seqlen = %r56 (runtime param)
    w(s, "shl.b32 \t%r70, %r53, 7;");         // head * 128 (stride_x_nheads hardcoded)
    w(s, "mad.wide.s32 \t%rd22, %r70, 2, %rd20;"); // X + batch_off + head*128*2

    // seq offset with stride_x_seqlen
    w(s, "mul.lo.s32 \t%r71, %r56, %r61;");   // stride_x_seqlen * seq_idx
    w(s, "shl.b32 \t%r72, %r56, 2;");         // stride_x_seqlen * 4
    w(s, "add.s32 \t%r73, %r71, %r72;");      // stride_x_seqlen * (seq_idx + 4)

    w(s, "mad.wide.s32 \t%rd23, %r71, 2, %rd22;");  // X ptr for first seq group
    w(s, "mad.wide.s32 \t%rd24, %r73, 2, %rd22;");  // X ptr for second seq group

    // k offset
    w(s, "mul.wide.u32 \t%rd25, %r65, 2;");  // (tid & 7) * 8 * 2 bytes
    w(s, "add.s64 \t%rd3, %rd23, %rd25;");   // first X ptr
    w(s, "add.s64 \t%rd4, %rd24, %rd25;");   // second X ptr

    // Line 133-137: Bounds check for X: head < nheads AND seq < seqlen
    w(s, "setp.lt.s32 \t%p3, %r53, %r48;");  // head < nheads
    w(s, "setp.lt.s32 \t%p4, %r61, %r47;");  // seq < seqlen
    w(s, "and.pred \t%p2, %p4, %p3;");

    // Line 139-149: Load first X block (8 f16 = 4 b32, predicated)
    w(s, "mov.u32 %r11, %r10;");
    w(s, "mov.u32 %r12, %r10;");
    w(s, "mov.u32 %r13, %r10;");
    w(s, "mov.u32 %r14, %r10;");
    w(s, "@%p2 ld.global.v4.b32 { %r11, %r12, %r13, %r14 }, [ %rd3 + 0 ];");

    // Unpack f16 pairs to individual f16 registers
    w(s, "mov.b32 \t{%rs1, %rs2}, %r11;");
    w(s, "mov.b32 \t{%rs3, %rs4}, %r12;");
    w(s, "mov.b32 \t{%rs5, %rs6}, %r13;");
    w(s, "mov.b32 \t{%rs7, %rs8}, %r14;");

    // Line 150-160: Load second X block (for second seq group offset)
    w(s, "mov.u32 %r15, %r10;");
    w(s, "mov.u32 %r16, %r10;");
    w(s, "mov.u32 %r17, %r10;");
    w(s, "mov.u32 %r18, %r10;");
    w(s, "@%p2 ld.global.v4.b32 { %r15, %r16, %r17, %r18 }, [ %rd4 + 0 ];");

    w(s, "mov.b32 \t{%rs9, %rs10}, %r15;");
    w(s, "mov.b32 \t{%rs11, %rs12}, %r16;");
    w(s, "mov.b32 \t{%rs13, %rs14}, %r17;");
    w(s, "mov.b32 \t{%rs15, %rs16}, %r18;");

    // Line 162-177: Convert first X block f16 -> f32
    for (i, rs) in (1..=8).enumerate() {
        let r = 74 + i as u32;
        w(s, &format!("cvt.f32.f16 \t%r{}, %rs{};", r, rs));
    }

    // Line 170-177: Convert second X block f16 -> f32
    for (i, rs) in (9..=16).enumerate() {
        let r = 82 + i as u32;
        w(s, &format!("cvt.f32.f16 \t%r{}, %rs{};", r, rs));
    }

    // Line 179-180: Load second chunk of X (at +128 bytes = +64 f16 = half_dim offset)
    w(s, "add.s64 \t%rd5, %rd3, 128;");
    w(s, "add.s64 \t%rd6, %rd4, 128;");

    // Line 182-192: Load third X block
    w(s, "mov.u32 %r19, %r10;");
    w(s, "mov.u32 %r20, %r10;");
    w(s, "mov.u32 %r21, %r10;");
    w(s, "mov.u32 %r22, %r10;");
    w(s, "@%p2 ld.global.v4.b32 { %r19, %r20, %r21, %r22 }, [ %rd5 + 0 ];");

    w(s, "mov.b32 \t{%rs17, %rs18}, %r19;");
    w(s, "mov.b32 \t{%rs19, %rs20}, %r20;");
    w(s, "mov.b32 \t{%rs21, %rs22}, %r21;");
    w(s, "mov.b32 \t{%rs23, %rs24}, %r22;");

    // Line 193-203: Load fourth X block
    w(s, "mov.u32 %r23, %r10;");
    w(s, "mov.u32 %r24, %r10;");
    w(s, "mov.u32 %r25, %r10;");
    w(s, "mov.u32 %r26, %r10;");
    w(s, "@%p2 ld.global.v4.b32 { %r23, %r24, %r25, %r26 }, [ %rd6 + 0 ];");

    w(s, "mov.b32 \t{%rs25, %rs26}, %r23;");
    w(s, "mov.b32 \t{%rs27, %rs28}, %r24;");
    w(s, "mov.b32 \t{%rs29, %rs30}, %r25;");
    w(s, "mov.b32 \t{%rs31, %rs32}, %r26;");

    // Line 205-220: Convert third and fourth X blocks f16 -> f32
    for (i, rs) in (17..=24).enumerate() {
        let r = 90 + i as u32;
        w(s, &format!("cvt.f32.f16 \t%r{}, %rs{};", r, rs));
    }
    for (i, rs) in (25..=32).enumerate() {
        let r = 98 + i as u32;
        w(s, &format!("cvt.f32.f16 \t%r{}, %rs{};", r, rs));
    }

    // ============================================================
    // Line 222-249: Shared memory transpose for cos/sin redistribution
    // ============================================================

    // Compute smem write address
    w(s, "shl.b32 \t%r106, %r49, 3;");       // tid * 8
    w(s, "and.b32 \t%r107, %r106, 1008;");    // (tid*8) & 1008
    w(s, "and.b32 \t%r108, %r49, 1;");        // tid & 1
    w(s, "neg.s32 \t%r109, %r108;");          // -(tid & 1) -> 0 or -1
    w(s, "and.b32 \t%r110, %r109, 1088;");    // mask & 1088
    w(s, "xor.b32 \t%r111, %r110, %r107;");   // swizzle
    w(s, "mov.b32 \t%r112, global_smem;");     // smem base
    w(s, "add.s32 \t%r113, %r112, %r111;");   // smem write ptr

    // Store cos to smem
    w(s, "st.shared.v4.b32 \t[%r113], {%r1, %r2, %r3, %r4};");
    w(s, "bar.sync \t0;");

    // Compute smem read address
    w(s, "shl.b32 \t%r114, %r64, 4;");        // (tid & 7) * 16
    w(s, "shl.b32 \t%r115, %r58, 2;");        // (tid & 96) * 4
    w(s, "or.b32 \t%r116, %r114, %r115;");    // combined read offset
    w(s, "add.s32 \t%r117, %r112, %r116;");   // smem read ptr

    // Read cos from smem (4 reads for the redistribution)
    w(s, "ld.shared.v4.b32 \t{%r118, %r119, %r120, %r121}, [%r117];");
    w(s, "ld.shared.v4.b32 \t{%r122, %r123, %r124, %r125}, [%r117+512];");

    w(s, "xor.b32 \t%r126, %r116, 64;");      // swizzled offset
    w(s, "add.s32 \t%r127, %r112, %r126;");
    w(s, "ld.shared.v4.b32 \t{%r128, %r129, %r130, %r131}, [%r127+1024];");
    w(s, "ld.shared.v4.b32 \t{%r132, %r133, %r134, %r135}, [%r127+1536];");

    // Now do the same transpose for sin
    w(s, "bar.sync \t0;");
    w(s, "st.shared.v4.b32 \t[%r113], {%r6, %r7, %r8, %r9};");
    w(s, "bar.sync \t0;");

    w(s, "ld.shared.v4.b32 \t{%r136, %r137, %r138, %r139}, [%r117];");
    w(s, "ld.shared.v4.b32 \t{%r140, %r141, %r142, %r143}, [%r117+512];");
    w(s, "ld.shared.v4.b32 \t{%r144, %r145, %r146, %r147}, [%r127+1024];");
    w(s, "ld.shared.v4.b32 \t{%r148, %r149, %r150, %r151}, [%r127+1536];");

    // ============================================================
    // Line 251-266: x_second * sin (16 multiplies for first-half computation)
    // ============================================================
    // These are: sin_redistributed * x_second_f32
    // sin regs: %r136-%r151, x_second regs: %r90-%r105
    w(s, "mul.f32 \t%r152, %r136, %r90;");
    w(s, "mul.f32 \t%r153, %r137, %r91;");
    w(s, "mul.f32 \t%r154, %r138, %r92;");
    w(s, "mul.f32 \t%r155, %r139, %r93;");
    w(s, "mul.f32 \t%r156, %r144, %r94;");
    w(s, "mul.f32 \t%r157, %r145, %r95;");
    w(s, "mul.f32 \t%r158, %r146, %r96;");
    w(s, "mul.f32 \t%r159, %r147, %r97;");
    w(s, "mul.f32 \t%r160, %r140, %r98;");
    w(s, "mul.f32 \t%r161, %r141, %r99;");
    w(s, "mul.f32 \t%r162, %r142, %r100;");
    w(s, "mul.f32 \t%r163, %r143, %r101;");
    w(s, "mul.f32 \t%r164, %r148, %r102;");
    w(s, "mul.f32 \t%r165, %r149, %r103;");
    w(s, "mul.f32 \t%r166, %r150, %r104;");
    w(s, "mul.f32 \t%r167, %r151, %r105;");

    // ============================================================
    // Line 268-299: out_first = x_first * cos - x_second * sin
    // Pattern: neg the mul result, then fma(cos_val, x_first, neg_result)
    // cos regs: %r118-%r135, x_first regs: %r74-%r89
    // ============================================================
    w(s, "neg.f32 \t%r168, %r152;");
    w(s, "fma.rn.f32 \t%r169, %r118, %r74, %r168;");
    w(s, "neg.f32 \t%r170, %r153;");
    w(s, "fma.rn.f32 \t%r171, %r119, %r75, %r170;");
    w(s, "neg.f32 \t%r172, %r154;");
    w(s, "fma.rn.f32 \t%r173, %r120, %r76, %r172;");
    w(s, "neg.f32 \t%r174, %r155;");
    w(s, "fma.rn.f32 \t%r175, %r121, %r77, %r174;");
    w(s, "neg.f32 \t%r176, %r156;");
    w(s, "fma.rn.f32 \t%r177, %r128, %r78, %r176;");
    w(s, "neg.f32 \t%r178, %r157;");
    w(s, "fma.rn.f32 \t%r179, %r129, %r79, %r178;");
    w(s, "neg.f32 \t%r180, %r158;");
    w(s, "fma.rn.f32 \t%r181, %r130, %r80, %r180;");
    w(s, "neg.f32 \t%r182, %r159;");
    w(s, "fma.rn.f32 \t%r183, %r131, %r81, %r182;");
    w(s, "neg.f32 \t%r184, %r160;");
    w(s, "fma.rn.f32 \t%r185, %r122, %r82, %r184;");
    w(s, "neg.f32 \t%r186, %r161;");
    w(s, "fma.rn.f32 \t%r187, %r123, %r83, %r186;");
    w(s, "neg.f32 \t%r188, %r162;");
    w(s, "fma.rn.f32 \t%r189, %r124, %r84, %r188;");
    w(s, "neg.f32 \t%r190, %r163;");
    w(s, "fma.rn.f32 \t%r191, %r125, %r85, %r190;");
    w(s, "neg.f32 \t%r192, %r164;");
    w(s, "fma.rn.f32 \t%r193, %r132, %r86, %r192;");
    w(s, "neg.f32 \t%r194, %r165;");
    w(s, "fma.rn.f32 \t%r195, %r133, %r87, %r194;");
    w(s, "neg.f32 \t%r196, %r166;");
    w(s, "fma.rn.f32 \t%r197, %r134, %r88, %r196;");
    w(s, "neg.f32 \t%r198, %r167;");
    w(s, "fma.rn.f32 \t%r199, %r135, %r89, %r198;");

    // ============================================================
    // Line 301-316: out_second = x_second * cos + x_first * sin
    // Pattern: mul(sin_val, x_first), then fma(cos_val, x_second, mul_result)
    // Wait - reference is: mul(sin, x_first) then fma(cos, x_second, that)
    // Actually reference line 301: mul.f32 %r200, %r136, %r74 -- sin * x_first
    // Then line 318: fma.rn.f32 %r216, %r118, %r90, %r200 -- cos * x_second + sin*x_first
    // ============================================================
    w(s, "mul.f32 \t%r200, %r136, %r74;");
    w(s, "mul.f32 \t%r201, %r137, %r75;");
    w(s, "mul.f32 \t%r202, %r138, %r76;");
    w(s, "mul.f32 \t%r203, %r139, %r77;");
    w(s, "mul.f32 \t%r204, %r144, %r78;");
    w(s, "mul.f32 \t%r205, %r145, %r79;");
    w(s, "mul.f32 \t%r206, %r146, %r80;");
    w(s, "mul.f32 \t%r207, %r147, %r81;");
    w(s, "mul.f32 \t%r208, %r140, %r82;");
    w(s, "mul.f32 \t%r209, %r141, %r83;");
    w(s, "mul.f32 \t%r210, %r142, %r84;");
    w(s, "mul.f32 \t%r211, %r143, %r85;");
    w(s, "mul.f32 \t%r212, %r148, %r86;");
    w(s, "mul.f32 \t%r213, %r149, %r87;");
    w(s, "mul.f32 \t%r214, %r150, %r88;");
    w(s, "mul.f32 \t%r215, %r151, %r89;");

    // Line 318-333: fma for out_second
    w(s, "fma.rn.f32 \t%r216, %r118, %r90, %r200;");
    w(s, "fma.rn.f32 \t%r217, %r119, %r91, %r201;");
    w(s, "fma.rn.f32 \t%r218, %r120, %r92, %r202;");
    w(s, "fma.rn.f32 \t%r219, %r121, %r93, %r203;");
    w(s, "fma.rn.f32 \t%r220, %r128, %r94, %r204;");
    w(s, "fma.rn.f32 \t%r221, %r129, %r95, %r205;");
    w(s, "fma.rn.f32 \t%r222, %r130, %r96, %r206;");
    w(s, "fma.rn.f32 \t%r223, %r131, %r97, %r207;");
    w(s, "fma.rn.f32 \t%r224, %r122, %r98, %r208;");
    w(s, "fma.rn.f32 \t%r225, %r123, %r99, %r209;");
    w(s, "fma.rn.f32 \t%r226, %r124, %r100, %r210;");
    w(s, "fma.rn.f32 \t%r227, %r125, %r101, %r211;");
    w(s, "fma.rn.f32 \t%r228, %r132, %r102, %r212;");
    w(s, "fma.rn.f32 \t%r229, %r133, %r103, %r213;");
    w(s, "fma.rn.f32 \t%r230, %r134, %r104, %r214;");
    w(s, "fma.rn.f32 \t%r231, %r135, %r105, %r215;");

    // ============================================================
    // Line 335-347: Output address computation
    // ============================================================
    w(s, "mul.lo.s32 \t%r232, %r52, %r53;");  // stride_out_nheads * head_idx
    w(s, "mad.wide.s32 \t%rd26, %r232, 2, %rd21;"); // OUT + batch_off + head*stride*2

    w(s, "mul.lo.s32 \t%r233, %r50, %r61;");  // stride_out_seqlen * seq_idx
    w(s, "shl.b32 \t%r234, %r50, 2;");        // stride_out_seqlen * 4
    w(s, "add.s32 \t%r235, %r233, %r234;");   // stride_out_seqlen * (seq_idx + 4)

    w(s, "mad.wide.s32 \t%rd27, %r233, 2, %rd26;");
    w(s, "mad.wide.s32 \t%rd28, %r235, 2, %rd26;");

    w(s, "add.s64 \t%rd7, %rd27, %rd25;");
    w(s, "add.s64 \t%rd8, %rd28, %rd25;");

    // ============================================================
    // Line 349-380: Convert f32 -> f16x2 and store output
    // ============================================================

    // Store out_first block 1 (8 f16 = 4 b32)
    w(s, "cvt.rn.f16x2.f32 \t%r27, %r171, %r169;");
    w(s, "cvt.rn.f16x2.f32 \t%r28, %r175, %r173;");
    w(s, "cvt.rn.f16x2.f32 \t%r29, %r179, %r177;");
    w(s, "cvt.rn.f16x2.f32 \t%r30, %r183, %r181;");
    w(s, "@%p2 st.global.v4.b32 [ %rd7 + 0 ], { %r27, %r28, %r29, %r30 };");

    // Store out_first block 2 (second seq group)
    w(s, "cvt.rn.f16x2.f32 \t%r31, %r187, %r185;");
    w(s, "cvt.rn.f16x2.f32 \t%r32, %r191, %r189;");
    w(s, "cvt.rn.f16x2.f32 \t%r33, %r195, %r193;");
    w(s, "cvt.rn.f16x2.f32 \t%r34, %r199, %r197;");
    w(s, "@%p2 st.global.v4.b32 [ %rd8 + 0 ], { %r31, %r32, %r33, %r34 };");

    // Store out_second block 1 (at +128 bytes = half_dim offset)
    w(s, "add.s64 \t%rd9, %rd7, 128;");
    w(s, "add.s64 \t%rd10, %rd8, 128;");

    w(s, "cvt.rn.f16x2.f32 \t%r35, %r217, %r216;");
    w(s, "cvt.rn.f16x2.f32 \t%r36, %r219, %r218;");
    w(s, "cvt.rn.f16x2.f32 \t%r37, %r221, %r220;");
    w(s, "cvt.rn.f16x2.f32 \t%r38, %r223, %r222;");
    w(s, "@%p2 st.global.v4.b32 [ %rd9 + 0 ], { %r35, %r36, %r37, %r38 };");

    // Store out_second block 2
    w(s, "cvt.rn.f16x2.f32 \t%r39, %r225, %r224;");
    w(s, "cvt.rn.f16x2.f32 \t%r40, %r227, %r226;");
    w(s, "cvt.rn.f16x2.f32 \t%r41, %r229, %r228;");
    w(s, "cvt.rn.f16x2.f32 \t%r42, %r231, %r230;");
    w(s, "@%p2 st.global.v4.b32 [ %rd10 + 0 ], { %r39, %r40, %r41, %r42 };");

    // Return
    w(s, "ret;");
    s.push_str("}\n");
}

fn w(s: &mut String, line: &str) {
    s.push('\t');
    s.push_str(line);
    s.push('\n');
}

#[allow(dead_code)]
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
        let global_loads = ptx.lines().filter(|l| l.contains("ld.global")).count();
        assert!(
            global_loads >= 6,
            "Need loads for cos, sin, x (4 blocks); got {}",
            global_loads
        );
    }

    #[test]
    fn test_rotary_has_fma() {
        let ptx = get_ptx();
        let fma_count = ptx.lines().filter(|l| l.contains("fma.rn.f32")).count();
        assert_eq!(fma_count, 32, "Need 32 fma (16 out_first + 16 out_second)");
    }

    #[test]
    fn test_rotary_has_neg() {
        let ptx = get_ptx();
        let neg_count = ptx.lines().filter(|l| l.contains("neg.f32")).count();
        assert_eq!(
            neg_count, 16,
            "Need 16 neg.f32 for out_first computation"
        );
    }

    #[test]
    fn test_rotary_stores_output() {
        let ptx = get_ptx();
        let stores = ptx.lines().filter(|l| l.contains("st.global")).count();
        assert_eq!(stores, 4, "Need 4 stores (2 first-half + 2 second-half)");
    }

    #[test]
    fn test_rotary_has_f16_conversion() {
        let ptx = get_ptx();
        let cvt_in = ptx.lines().filter(|l| l.contains("cvt.f32.f16")).count();
        let cvt_out = ptx
            .lines()
            .filter(|l| l.contains("cvt.rn.f16x2.f32"))
            .count();
        assert_eq!(cvt_in, 32, "Need 32 f16->f32 conversions");
        assert_eq!(cvt_out, 16, "Need 16 f32->f16x2 conversions");
    }

    #[test]
    fn test_rotary_has_bounds_check() {
        let ptx = get_ptx();
        assert!(ptx.contains("setp.lt.s32"), "Need bounds checking");
    }

    #[test]
    fn test_rotary_has_shared_memory() {
        let ptx = get_ptx();
        assert!(
            ptx.contains("global_smem"),
            "Need shared memory for cos/sin transpose"
        );
        let smem_stores = ptx.lines().filter(|l| l.contains("st.shared")).count();
        let smem_loads = ptx.lines().filter(|l| l.contains("ld.shared")).count();
        assert_eq!(smem_stores, 2, "Need 2 smem stores (cos + sin)");
        assert_eq!(smem_loads, 8, "Need 8 smem loads (4 cos + 4 sin)");
    }

    #[test]
    fn test_rotary_has_bfe() {
        let ptx = get_ptx();
        let bfe_count = ptx.lines().filter(|l| l.contains("bfe.u32")).count();
        assert_eq!(
            bfe_count, 3,
            "Need 3 bfe.u32 for thread decomposition"
        );
    }

    #[test]
    fn test_rotary_has_bar_sync() {
        let ptx = get_ptx();
        let bar_count = ptx.lines().filter(|l| l.contains("bar.sync")).count();
        assert_eq!(bar_count, 3, "Need 3 bar.sync for smem transpose");
    }

    #[test]
    fn test_rotary_register_allocation() {
        let ptx = get_ptx();
        assert!(ptx.contains("%p<5>"), "Need 5 predicate registers");
        assert!(ptx.contains("%rs<33>"), "Need 33 b16 registers");
        assert!(ptx.contains("%r<236>"), "Need 236 b32 registers");
        assert!(ptx.contains("%rd<29>"), "Need 29 b64 registers");
    }

    #[test]
    fn test_rotary_ptx_size() {
        let ptx = get_ptx();
        let lines = ptx.lines().count();
        assert!(lines >= 100, "Too small: {} lines", lines);
        assert!(lines <= 500, "Too large: {} lines", lines);
    }

    #[test]
    fn test_rotary_mul_count() {
        let ptx = get_ptx();
        let mul_f32 = ptx.lines().filter(|l| {
            let trimmed = l.trim();
            trimmed.starts_with("mul.f32")
        }).count();
        assert_eq!(mul_f32, 32, "Need 32 mul.f32 (16 for out_first prep + 16 for out_second prep)");
    }

    #[test]
    fn test_rotary_10_params() {
        let ptx = get_ptx();
        // Count param declarations (lines with .param inside the entry signature, before .reqntid)
        let param_lines = ptx.lines().filter(|l| {
            let t = l.trim();
            t.starts_with(".param")
        }).count();
        // 4 ptr params + 6 u32 params = 10 total
        assert_eq!(param_lines, 10, "Need exactly 10 params (no workspace ptrs)");
    }
}
