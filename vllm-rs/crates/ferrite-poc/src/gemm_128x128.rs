// Hand-crafted 128×128×32 GEMM PTX kernel matching Triton's 48.5 TFLOPS.
//
// Tile: 128×128, BK=32, 128 threads (4 warps), 2-stage pipeline
// 64 mma.sync per K-iteration, 16 ldmatrix, 8 cp.async
// Smem: 2 × (A:8192 + B:8192) = 32768 bytes
// Warp layout: 2×2 (2 warps in M, 2 in N)
//   Each warp: 64 rows × 64 cols, REG_M=4, REG_N=8
// Output: f32 (for verification), store directly to global

pub fn emit_ptx_128x128() -> String {
    let mut s = String::with_capacity(80 * 1024);
    emit_kernel(&mut s);
    s
}

fn emit_kernel(s: &mut String) {
    // ─── Header ───
    s.push_str(r#".version 8.7
.target sm_89
.address_size 64

.extern .shared .align 16 .b8 global_smem[];

.visible .entry gemm_128x128(
	.param .u64 .ptr .global .align 16 param_A,
	.param .u64 .ptr .global .align 16 param_B,
	.param .u64 .ptr .global .align 16 param_C,
	.param .u32 param_M,
	.param .u32 param_N,
	.param .u32 param_K
)
.reqntid 128
{
	.reg .pred 	%p<12>;
	.reg .b32 	%r<500>;
	.reg .b64 	%rd<120>;

"#);

    // ─── Parameter loads ───
    w(s, "ld.param.b64 \t%rd1, [param_A];");
    w(s, "ld.param.b64 \t%rd2, [param_B];");
    w(s, "ld.param.b64 \t%rd3, [param_C];");
    w(s, "ld.param.b32 \t%r1, [param_M];");
    w(s, "ld.param.b32 \t%r2, [param_N];");   // N = C stride, B col stride
    w(s, "ld.param.b32 \t%r3, [param_K];");   // K = A col stride
    blank(s);

    // ─── Thread/block indexing ───
    // %r4=block_n, %r5=block_m, %r6=tid
    // %r7=block_col*128, %r8=block_row*128
    w(s, "mov.u32 \t%r4, %ctaid.x;");
    w(s, "mov.u32 \t%r5, %ctaid.y;");
    w(s, "shl.b32 \t%r7, %r4, 7;");
    w(s, "shl.b32 \t%r8, %r5, 7;");
    w(s, "mov.u32 \t%r6, %tid.x;");
    blank(s);

    // ─── cp.async thread mapping ───
    // A: row=bfe(tid,2,5), col=(tid&3)*8
    w(s, "bfe.u32 \t%r9, %r6, 2, 5;");       // A_row within chunk
    w(s, "and.b32 \t%r18, %r6, 3;");
    w(s, "shl.b32 \t%r19, %r18, 3;");         // A_col = (tid&3)*8
    // B: row=bfe(tid,4,3), col=(tid&15)*8
    w(s, "bfe.u32 \t%r11, %r6, 4, 3;");       // B_row within chunk
    w(s, "or.b32 \t%r12, %r11, 8;");           // B_row+8
    w(s, "or.b32 \t%r13, %r11, 16;");          // B_row+16
    w(s, "or.b32 \t%r14, %r11, 24;");          // B_row+24
    w(s, "and.b32 \t%r15, %r6, 15;");
    w(s, "shl.b32 \t%r16, %r15, 3;");          // B_col = (tid&15)*8
    blank(s);

    // ─── A global pointers (4 chunks × 32 rows) ───
    // A[row][col] @ A + (row*K + col)*2
    w(s, "or.b32 \t%r17, %r9, %r8;");          // row0 = block_row | A_row
    w(s, "mul.lo.s32 \t%r20, %r3, %r17;");     // row0 * K
    w(s, "shl.b32 \t%r21, %r3, 5;");            // K * 32
    w(s, "add.s32 \t%r22, %r20, %r21;");
    w(s, "add.s32 \t%r23, %r22, %r21;");
    w(s, "add.s32 \t%r24, %r23, %r21;");
    w(s, "mad.wide.s32 \t%rd11, %r20, 2, %rd1;");
    w(s, "mad.wide.s32 \t%rd12, %r22, 2, %rd1;");
    w(s, "mad.wide.s32 \t%rd13, %r23, 2, %rd1;");
    w(s, "mad.wide.s32 \t%rd14, %r24, 2, %rd1;");
    w(s, "mul.wide.u32 \t%rd10, %r19, 2;");
    w(s, "add.s64 \t%rd15, %rd11, %rd10;");
    w(s, "add.s64 \t%rd16, %rd12, %rd10;");
    w(s, "add.s64 \t%rd17, %rd13, %rd10;");
    w(s, "add.s64 \t%rd18, %rd14, %rd10;");
    blank(s);

    // ─── B global pointers (4 chunks × 8 rows) ───
    // B[row][col] @ B + (row*N + col)*2
    w(s, "or.b32 \t%r25, %r16, %r7;");         // B_col = block_col | tid_col
    w(s, "mul.lo.s32 \t%r26, %r2, %r11;");     // brow0 * N
    w(s, "shl.b32 \t%r27, %r2, 3;");            // N * 8
    w(s, "add.s32 \t%r28, %r26, %r27;");
    w(s, "add.s32 \t%r29, %r28, %r27;");
    w(s, "add.s32 \t%r30, %r29, %r27;");
    w(s, "mad.wide.s32 \t%rd20, %r26, 2, %rd2;");
    w(s, "mad.wide.s32 \t%rd21, %r28, 2, %rd2;");
    w(s, "mad.wide.s32 \t%rd22, %r29, 2, %rd2;");
    w(s, "mad.wide.s32 \t%rd23, %r30, 2, %rd2;");
    w(s, "mul.wide.u32 \t%rd19, %r25, 2;");
    w(s, "add.s64 \t%rd24, %rd20, %rd19;");
    w(s, "add.s64 \t%rd25, %rd21, %rd19;");
    w(s, "add.s64 \t%rd26, %rd22, %rd19;");
    w(s, "add.s64 \t%rd27, %rd23, %rd19;");
    w(s, "shl.b32 \t%r31, %r2, 5;");            // N*32 (B K-stride)
    blank(s);

    // ─── Smem swizzle for cp.async (matching Triton) ───
    w(s, "shl.b32 \t%r32, %r6, 4;");            // tid*16
    w(s, "and.b32 \t%r33, %r32, 2032;");
    w(s, "and.b32 \t%r34, %r6, 24;");
    w(s, "shl.b32 \t%r35, %r34, 1;");
    w(s, "xor.b32 \t%r36, %r33, %r35;");        // A cp swizzle
    w(s, "mov.b32 \t%r37, global_smem;");
    w(s, "add.s32 \t%r38, %r37, %r36;");        // A smem base
    w(s, "and.b32 \t%r10, %r6, 112;");           // tid & 0x70
    w(s, "xor.b32 \t%r39, %r33, %r10;");        // B cp swizzle
    w(s, "add.s32 \t%r40, %r37, %r39;");         // B smem base
    blank(s);

    // ─── Prologue: load tile 0 ───
    w(s, "setp.gt.s32 \t%p1, %r3, 0;");
    w(s, "selp.b32 \t%r41, 16, 0, %p1;");
    // A tile 0 (4 cp.async into buffer 0)
    for (i, off) in [0i32, 2048, 4096, 6144].iter().enumerate() {
        if *off > 0 {
            s.push_str(&format!("\tadd.s32 \t%r{}, %r38, {off};\n", 42 + i - 1));
            s.push_str(&format!("\tcp.async.cg.shared.global [ %r{} + 0 ], [ %rd{} + 0 ], 0x10, %r41;\n", 42 + i - 1, 15 + i));
        } else {
            s.push_str(&format!("\tcp.async.cg.shared.global [ %r38 + 0 ], [ %rd15 + 0 ], 0x10, %r41;\n"));
        }
    }
    w(s, "cp.async.commit_group;");
    // B tile 0 (4 cp.async, B region starts at 16384)
    for (i, off) in [16384i32, 18432, 20480, 22528].iter().enumerate() {
        s.push_str(&format!("\tadd.s32 \t%r{}, %r40, {off};\n", 45 + i));
        s.push_str(&format!("\tcp.async.cg.shared.global [ %r{} + 0 ], [ %rd{} + 0 ], 0x10, %r41;\n", 45 + i, 24 + i));
    }
    w(s, "cp.async.commit_group;");
    blank(s);

    // ─── Advance for tile 1 ───
    w(s, "setp.gt.s32 \t%p2, %r3, 32;");
    w(s, "add.s64 \t%rd28, %rd15, 64;");        // A + BK*2
    w(s, "add.s64 \t%rd29, %rd16, 64;");
    w(s, "add.s64 \t%rd30, %rd17, 64;");
    w(s, "add.s64 \t%rd31, %rd18, 64;");
    w(s, "mul.wide.s32 \t%rd32, %r31, 2;");     // N*32*2 bytes
    w(s, "add.s64 \t%rd33, %rd24, %rd32;");
    w(s, "add.s64 \t%rd34, %rd25, %rd32;");
    w(s, "add.s64 \t%rd35, %rd26, %rd32;");
    w(s, "add.s64 \t%rd36, %rd27, %rd32;");
    blank(s);

    // Load tile 1 into buffer 1
    w(s, "bar.sync \t0;");
    w(s, "selp.b32 \t%r49, 16, 0, %p2;");
    // A tile 1 (buffer 1 = +8192)
    for (i, off) in [8192i32, 10240, 12288, 14336].iter().enumerate() {
        s.push_str(&format!("\tadd.s32 \t%r{}, %r38, {off};\n", 50 + i));
        s.push_str(&format!("\tcp.async.cg.shared.global [ %r{} + 0 ], [ %rd{} + 0 ], 0x10, %r49;\n", 50 + i, 28 + i));
    }
    w(s, "cp.async.commit_group;");
    // B tile 1 (buffer 1 = +8192 from B base)
    for (i, off) in [24576i32, 26624, 28672, 30720].iter().enumerate() {
        s.push_str(&format!("\tadd.s32 \t%r{}, %r40, {off};\n", 54 + i));
        s.push_str(&format!("\tcp.async.cg.shared.global [ %r{} + 0 ], [ %rd{} + 0 ], 0x10, %r49;\n", 54 + i, 33 + i));
    }
    w(s, "cp.async.commit_group;");
    blank(s);

    // ─── Loop setup ───
    w(s, "@%p1 bra \t$L_LOOP_SETUP;");
    w(s, "bra.uni \t$L_K0_FALLTHROUGH;");
    blank(s);

    s.push_str("$L_LOOP_SETUP:\n");
    w(s, "add.s32 \t%r60, %r3, -64;");          // K - 2*BK

    // A ldmatrix offsets (matching Triton lines 216-232)
    w(s, "shl.b32 \t%r58, %r6, 3;");            // tid*8
    w(s, "and.b32 \t%r59, %r6, 16;");           // tid & 16
    w(s, "shl.b32 \t%r62, %r15, 6;");           // (tid&15)<<6
    w(s, "and.b32 \t%r63, %r58, 48;");           // (tid*8)&48
    w(s, "and.b32 \t%r64, %r32, 1024;");         // (tid*16)&1024
    w(s, "or.b32 \t%r65, %r62, %r63;");
    w(s, "xor.b32 \t%r66, %r65, %r59;");
    w(s, "or.b32 \t%r67, %r66, %r64;");          // a_off_ki0
    w(s, "xor.b32 \t%r68, %r67, 32;");           // a_off_ki1
    blank(s);

    // B ldmatrix offsets (matching Triton lines 223-232)
    w(s, "shl.b32 \t%r69, %r6, 8;");             // tid<<8
    w(s, "and.b32 \t%r70, %r69, 7936;");
    w(s, "and.b32 \t%r71, %r32, 112;");          // (tid*16)&112
    w(s, "shr.u32 \t%r72, %r6, 1;");
    w(s, "and.b32 \t%r73, %r72, 16;");
    w(s, "xor.b32 \t%r74, %r71, %r73;");
    w(s, "or.b32 \t%r75, %r74, %r70;");          // b_off_rn0
    w(s, "xor.b32 \t%r76, %r75, 32;");
    w(s, "xor.b32 \t%r77, %r75, 64;");
    w(s, "xor.b32 \t%r78, %r75, 96;");
    blank(s);

    // Loop pointers: start at tile 2 (after 2 prologue loads)
    w(s, "add.s64 \t%rd50, %rd15, 128;");        // A tile2
    w(s, "add.s64 \t%rd51, %rd16, 128;");
    w(s, "add.s64 \t%rd52, %rd17, 128;");
    w(s, "add.s64 \t%rd53, %rd18, 128;");
    w(s, "shl.b64 \t%rd45, %rd32, 1;");          // 2 × N*32*2
    w(s, "add.s64 \t%rd54, %rd24, %rd45;");       // B tile2
    w(s, "add.s64 \t%rd55, %rd25, %rd45;");
    w(s, "add.s64 \t%rd56, %rd26, %rd45;");
    w(s, "add.s64 \t%rd57, %rd27, %rd45;");
    blank(s);

    // Initialize accumulators %r200..%r327 = 0
    w(s, "mov.b32 \t%r100, 0x00000000;");
    for i in 200..328u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r100;\n"));
    }
    blank(s);

    // Buffer toggle
    w(s, "mov.b32 \t%r101, 1;");                 // read_ctr
    w(s, "mov.b32 \t%r102, -1;");                // write_ctr
    w(s, "mov.b32 \t%r103, 0;");                 // k_counter
    blank(s);

    // ═══════════════════════════════════════════════════════════════
    // K-LOOP
    // ═══════════════════════════════════════════════════════════════
    s.push_str("$L_KLOOP:\n");
    w(s, "setp.lt.s32 \t%p3, %r103, %r60;");    // k_counter < K-64

    // Toggle read buffer
    w(s, "add.s32 \t%r104, %r101, 1;");
    w(s, "setp.gt.s32 \t%p4, %r104, 1;");
    w(s, "selp.b32 \t%r101, 0, %r104, %p4;");

    w(s, "cp.async.wait_group \t2;");
    w(s, "bar.sync \t0;");

    // Read buffer base = smem + read_ctr << 13
    w(s, "shl.b32 \t%r105, %r101, 13;");
    w(s, "add.s32 \t%r106, %r37, %r105;");
    blank(s);

    // ─── ldmatrix A (8 loads) ───
    w(s, "add.s32 \t%r107, %r106, %r67;");       // A ki0 addr
    w(s, "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r108, %r109, %r110, %r111}, [%r107];");
    w(s, "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r112, %r113, %r114, %r115}, [%r107+2048];");
    w(s, "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r116, %r117, %r118, %r119}, [%r107+4096];");
    w(s, "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r120, %r121, %r122, %r123}, [%r107+6144];");
    w(s, "add.s32 \t%r124, %r106, %r68;");       // A ki1 addr
    w(s, "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r125, %r126, %r127, %r128}, [%r124];");
    w(s, "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r129, %r130, %r131, %r132}, [%r124+2048];");
    w(s, "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r133, %r134, %r135, %r136}, [%r124+4096];");
    w(s, "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r137, %r138, %r139, %r140}, [%r124+6144];");
    blank(s);

    // ─── ldmatrix B transposed (8 loads) ───
    w(s, "add.s32 \t%r141, %r106, %r75;");       // B rn0
    w(s, "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r142, %r143, %r144, %r145}, [%r141+16384];");
    w(s, "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r146, %r147, %r148, %r149}, [%r141+16512];");
    w(s, "add.s32 \t%r150, %r106, %r76;");       // B rn1
    w(s, "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r151, %r152, %r153, %r154}, [%r150+16384];");
    w(s, "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r155, %r156, %r157, %r158}, [%r150+16512];");
    w(s, "add.s32 \t%r159, %r106, %r77;");       // B rn2
    w(s, "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r160, %r161, %r162, %r163}, [%r159+16384];");
    w(s, "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r164, %r165, %r166, %r167}, [%r159+16512];");
    w(s, "add.s32 \t%r168, %r106, %r78;");       // B rn3
    w(s, "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r169, %r170, %r171, %r172}, [%r168+16384];");
    w(s, "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r173, %r174, %r175, %r176}, [%r168+16512];");
    blank(s);

    // ─── MMA (64 total) ───
    // ki0: A{0..3} × B_ki0{0..7}
    let a_ki0 = [[108,109,110,111],[112,113,114,115],[116,117,118,119],[120,121,122,123]];
    // B ki0: from ldmatrix.trans first 2 regs of each
    let b_ki0 = [[142,143],[151,152],[160,161],[169,170],[146,147],[155,156],[164,165],[173,174]];
    let a_ki1 = [[125,126,127,128],[129,130,131,132],[133,134,135,136],[137,138,139,140]];
    let b_ki1 = [[144,145],[153,154],[162,163],[171,172],[148,149],[157,158],[166,167],[175,176]];

    let mut acc = 200u32;
    for am in 0..4 {
        for bn in 0..8 {
            let a = &a_ki0[am];
            let b = &b_ki0[bn];
            s.push_str(&format!(
                "\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{ %r{}, %r{}, %r{}, %r{} }}, {{ %r{}, %r{}, %r{}, %r{} }}, {{ %r{}, %r{} }}, {{ %r{}, %r{}, %r{}, %r{} }};\n",
                acc, acc+1, acc+2, acc+3, a[0], a[1], a[2], a[3], b[0], b[1], acc, acc+1, acc+2, acc+3
            ));
            acc += 4;
        }
    }
    acc = 200;
    for am in 0..4 {
        for bn in 0..8 {
            let a = &a_ki1[am];
            let b = &b_ki1[bn];
            s.push_str(&format!(
                "\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{ %r{}, %r{}, %r{}, %r{} }}, {{ %r{}, %r{}, %r{}, %r{} }}, {{ %r{}, %r{} }}, {{ %r{}, %r{}, %r{}, %r{} }};\n",
                acc, acc+1, acc+2, acc+3, a[0], a[1], a[2], a[3], b[0], b[1], acc, acc+1, acc+2, acc+3
            ));
            acc += 4;
        }
    }
    blank(s);

    // ─── Next-tile loads ───
    w(s, "add.s32 \t%r177, %r102, 1;");
    w(s, "setp.gt.s32 \t%p5, %r177, 1;");
    w(s, "selp.b32 \t%r102, 0, %r177, %p5;");
    w(s, "shl.b32 \t%r178, %r102, 13;");
    w(s, "add.s32 \t%r179, %r37, %r178;");
    w(s, "bar.sync \t0;");
    w(s, "add.s32 \t%r180, %r179, %r36;");       // write A base
    w(s, "selp.b32 \t%r181, 16, 0, %p3;");

    // A next-tile (4 cp.async)
    w(s, "cp.async.cg.shared.global [ %r180 + 0 ], [ %rd50 + 0 ], 0x10, %r181;");
    w(s, "add.s32 \t%r182, %r180, 2048;");
    w(s, "cp.async.cg.shared.global [ %r182 + 0 ], [ %rd51 + 0 ], 0x10, %r181;");
    w(s, "add.s32 \t%r183, %r180, 4096;");
    w(s, "cp.async.cg.shared.global [ %r183 + 0 ], [ %rd52 + 0 ], 0x10, %r181;");
    w(s, "add.s32 \t%r184, %r180, 6144;");
    w(s, "cp.async.cg.shared.global [ %r184 + 0 ], [ %rd53 + 0 ], 0x10, %r181;");
    w(s, "cp.async.commit_group;");

    // B next-tile (4 cp.async)
    w(s, "add.s32 \t%r185, %r179, %r39;");
    w(s, "add.s32 \t%r186, %r185, 16384;");
    w(s, "cp.async.cg.shared.global [ %r186 + 0 ], [ %rd54 + 0 ], 0x10, %r181;");
    w(s, "add.s32 \t%r187, %r185, 18432;");
    w(s, "cp.async.cg.shared.global [ %r187 + 0 ], [ %rd55 + 0 ], 0x10, %r181;");
    w(s, "add.s32 \t%r188, %r185, 20480;");
    w(s, "cp.async.cg.shared.global [ %r188 + 0 ], [ %rd56 + 0 ], 0x10, %r181;");
    w(s, "add.s32 \t%r189, %r185, 22528;");
    w(s, "cp.async.cg.shared.global [ %r189 + 0 ], [ %rd57 + 0 ], 0x10, %r181;");
    w(s, "cp.async.commit_group;");
    blank(s);

    // Advance loop pointers
    w(s, "add.s32 \t%r103, %r103, 32;");
    w(s, "add.s64 \t%rd50, %rd50, 64;");
    w(s, "add.s64 \t%rd51, %rd51, 64;");
    w(s, "add.s64 \t%rd52, %rd52, 64;");
    w(s, "add.s64 \t%rd53, %rd53, 64;");
    w(s, "add.s64 \t%rd54, %rd54, %rd32;");
    w(s, "add.s64 \t%rd55, %rd55, %rd32;");
    w(s, "add.s64 \t%rd56, %rd56, %rd32;");
    w(s, "add.s64 \t%rd57, %rd57, %rd32;");
    w(s, "setp.lt.s32 \t%p6, %r103, %r3;");
    w(s, "@%p6 bra \t$L_KLOOP;");
    w(s, "bra.uni \t$L_STORE;");
    blank(s);

    // ─── K=0 fallthrough ───
    s.push_str("$L_K0_FALLTHROUGH:\n");
    w(s, "mov.b32 \t%r100, 0x00000000;");
    for i in 200..328u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r100;\n"));
    }
    blank(s);

    // ═══════════════════════════════════════════════════════════════
    // Store C (f32)
    // ═══════════════════════════════════════════════════════════════
    s.push_str("$L_STORE:\n");
    w(s, "cp.async.wait_group \t0;");
    w(s, "bar.sync \t0;");
    blank(s);

    // Thread decomposition for store
    // warp_m = (tid >= 64) ? 1 : 0  (warps 2,3 handle rows 64-127)
    // warp_n = (warp_id & 1)        (warps 1,3 handle cols 64-127)
    // Actually: warp_id = tid >> 5
    //   warp 0 (tid 0-31):   warp_m=0, warp_n=0
    //   warp 1 (tid 32-63):  warp_m=0, warp_n=1
    //   warp 2 (tid 64-95):  warp_m=1, warp_n=0
    //   warp 3 (tid 96-127): warp_m=1, warp_n=1
    //
    // MMA accumulator (am, bn), d0..d3:
    //   lane = tid & 31
    //   mma_row = lane >> 2  (0..7 within m16 tile)
    //   mma_col = (lane & 3) * 2  (0,2,4,6 within n8 tile)
    //
    //   global_row_d0 = block_row + warp_m*16 + am*32 + mma_row
    //   global_row_d2 = global_row_d0 + 8
    //   global_col_d0 = block_col + warp_n*64 + bn*8 + mma_col
    //   global_col_d1 = global_col_d0 + 1

    w(s, "and.b32 \t%r328, %r6, 31;");           // lane
    w(s, "shr.u32 \t%r329, %r6, 5;");            // warp_id
    w(s, "shr.u32 \t%r330, %r329, 1;");           // warp_m (0 or 1)
    w(s, "and.b32 \t%r331, %r329, 1;");           // warp_n (0 or 1)
    w(s, "shr.u32 \t%r332, %r328, 2;");           // mma_row (0..7)
    w(s, "and.b32 \t%r333, %r328, 3;");           // lane & 3
    w(s, "shl.b32 \t%r334, %r333, 1;");           // mma_col (0,2,4,6)
    blank(s);

    // base_row = block_row + warp_m*16 + mma_row
    w(s, "shl.b32 \t%r335, %r330, 4;");           // warp_m * 16
    w(s, "add.s32 \t%r336, %r8, %r335;");         // block_row + warp_m*16
    w(s, "add.s32 \t%r337, %r336, %r332;");       // + mma_row = base_row
    blank(s);

    // base_col = block_col + warp_n*64 + mma_col
    w(s, "shl.b32 \t%r338, %r331, 6;");           // warp_n * 64
    w(s, "add.s32 \t%r339, %r7, %r338;");         // block_col + warp_n*64
    w(s, "add.s32 \t%r340, %r339, %r334;");       // + mma_col = base_col
    blank(s);

    // For each tile (am=0..3, bn=0..7):
    //   row0 = base_row + am*32, row1 = row0 + 8
    //   col = base_col + bn*8
    //   C[row0][col..col+1] = (d0, d1) as st.v2.b32
    //   C[row1][col..col+1] = (d2, d3) as st.v2.b32

    // We precompute row0*N and row1*N for each am value using temp regs %r350-365
    // Then for each bn, compute column offset and store

    acc = 200;
    for am in 0..4u32 {
        // row0 = base_row + am*32
        s.push_str(&format!("\tadd.s32 \t%r350, %r337, {};\n", am * 32));
        // row1 = row0 + 8
        s.push_str("\tadd.s32 \t%r351, %r350, 8;\n");
        // row0_off = row0 * N
        s.push_str("\tmul.lo.s32 \t%r352, %r2, %r350;\n");
        // row1_off = row1 * N
        s.push_str("\tmul.lo.s32 \t%r353, %r2, %r351;\n");
        // C_row0 = C + row0*N*4 (f32 bytes)
        s.push_str("\tmad.wide.s32 \t%rd70, %r352, 4, %rd3;\n");
        // C_row1 = C + row1*N*4
        s.push_str("\tmad.wide.s32 \t%rd71, %r353, 4, %rd3;\n");

        for bn in 0..8u32 {
            let d0 = acc;
            let d1 = acc + 1;
            let d2 = acc + 2;
            let d3 = acc + 3;

            // col = base_col + bn*8
            // col_byte_offset = col * 4
            s.push_str(&format!("\tadd.s32 \t%r354, %r340, {};\n", bn * 8));
            s.push_str("\tmul.wide.u32 \t%rd72, %r354, 4;\n");

            // addr_row0 = C_row0 + col*4
            s.push_str("\tadd.s64 \t%rd73, %rd70, %rd72;\n");
            // addr_row1 = C_row1 + col*4
            s.push_str("\tadd.s64 \t%rd74, %rd71, %rd72;\n");

            // st.global.v2.b32 for (d0, d1) and (d2, d3)
            s.push_str(&format!("\tst.global.v2.b32 [ %rd73 + 0 ], {{ %r{d0}, %r{d1} }};\n"));
            s.push_str(&format!("\tst.global.v2.b32 [ %rd74 + 0 ], {{ %r{d2}, %r{d3} }};\n"));

            acc += 4;
        }
    }

    w(s, "ret;");
    s.push_str("}\n");
}

fn w(s: &mut String, line: &str) {
    s.push_str("\t");
    s.push_str(line);
    s.push_str("\n");
}

fn blank(s: &mut String) {
    s.push_str("\n");
}
