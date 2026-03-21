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
    s.push_str(
        r#".version 8.7
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

"#,
    );

    // ─── Parameter loads ───
    w(s, "ld.param.b64 \t%rd1, [param_A];");
    w(s, "ld.param.b64 \t%rd2, [param_B];");
    w(s, "ld.param.b64 \t%rd3, [param_C];");
    w(s, "ld.param.b32 \t%r1, [param_M];");
    w(s, "ld.param.b32 \t%r2, [param_N];"); // N = C stride, B col stride
    w(s, "ld.param.b32 \t%r3, [param_K];"); // K = A col stride
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
    w(s, "bfe.u32 \t%r9, %r6, 2, 5;"); // A_row within chunk
    w(s, "and.b32 \t%r18, %r6, 3;");
    w(s, "shl.b32 \t%r19, %r18, 3;"); // A_col = (tid&3)*8
    // B: row=bfe(tid,4,3), col=(tid&15)*8
    w(s, "bfe.u32 \t%r11, %r6, 4, 3;"); // B_row within chunk
    w(s, "or.b32 \t%r12, %r11, 8;"); // B_row+8
    w(s, "or.b32 \t%r13, %r11, 16;"); // B_row+16
    w(s, "or.b32 \t%r14, %r11, 24;"); // B_row+24
    w(s, "and.b32 \t%r15, %r6, 15;");
    w(s, "shl.b32 \t%r16, %r15, 3;"); // B_col = (tid&15)*8
    blank(s);

    // ─── A global pointers (4 chunks × 32 rows) ───
    // A[row][col] @ A + (row*K + col)*2
    w(s, "or.b32 \t%r17, %r9, %r8;"); // row0 = block_row | A_row
    w(s, "mul.lo.s32 \t%r20, %r3, %r17;"); // row0 * K
    w(s, "shl.b32 \t%r21, %r3, 5;"); // K * 32
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
    w(s, "or.b32 \t%r25, %r16, %r7;"); // B_col = block_col | tid_col
    w(s, "mul.lo.s32 \t%r26, %r2, %r11;"); // brow0 * N
    w(s, "shl.b32 \t%r27, %r2, 3;"); // N * 8
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
    w(s, "shl.b32 \t%r31, %r2, 5;"); // N*32 (B K-stride)
    blank(s);

    // ─── Smem swizzle for cp.async (matching Triton) ───
    w(s, "shl.b32 \t%r32, %r6, 4;"); // tid*16
    w(s, "and.b32 \t%r33, %r32, 2032;");
    w(s, "and.b32 \t%r34, %r6, 24;");
    w(s, "shl.b32 \t%r35, %r34, 1;");
    w(s, "xor.b32 \t%r36, %r33, %r35;"); // A cp swizzle
    w(s, "mov.b32 \t%r37, global_smem;");
    w(s, "add.s32 \t%r38, %r37, %r36;"); // A smem base
    w(s, "and.b32 \t%r10, %r6, 112;"); // tid & 0x70
    w(s, "xor.b32 \t%r39, %r33, %r10;"); // B cp swizzle
    w(s, "add.s32 \t%r40, %r37, %r39;"); // B smem base
    blank(s);

    // ─── Prologue: load tile 0 ───
    w(s, "setp.gt.s32 \t%p1, %r3, 0;");
    w(s, "selp.b32 \t%r41, 16, 0, %p1;");
    // A tile 0 (4 cp.async into buffer 0)
    for (i, off) in [0i32, 2048, 4096, 6144].iter().enumerate() {
        if *off > 0 {
            s.push_str(&format!("\tadd.s32 \t%r{}, %r38, {off};\n", 42 + i - 1));
            s.push_str(&format!(
                "\tcp.async.cg.shared.global [ %r{} + 0 ], [ %rd{} + 0 ], 0x10, %r41;\n",
                42 + i - 1,
                15 + i
            ));
        } else {
            s.push_str(&format!(
                "\tcp.async.cg.shared.global [ %r38 + 0 ], [ %rd15 + 0 ], 0x10, %r41;\n"
            ));
        }
    }
    w(s, "cp.async.commit_group;");
    // B tile 0 (4 cp.async, B region starts at 16384)
    for (i, off) in [16384i32, 18432, 20480, 22528].iter().enumerate() {
        s.push_str(&format!("\tadd.s32 \t%r{}, %r40, {off};\n", 45 + i));
        s.push_str(&format!(
            "\tcp.async.cg.shared.global [ %r{} + 0 ], [ %rd{} + 0 ], 0x10, %r41;\n",
            45 + i,
            24 + i
        ));
    }
    w(s, "cp.async.commit_group;");
    blank(s);

    // ─── Advance for tile 1 ───
    w(s, "setp.gt.s32 \t%p2, %r3, 32;");
    w(s, "add.s64 \t%rd28, %rd15, 64;"); // A + BK*2
    w(s, "add.s64 \t%rd29, %rd16, 64;");
    w(s, "add.s64 \t%rd30, %rd17, 64;");
    w(s, "add.s64 \t%rd31, %rd18, 64;");
    w(s, "mul.wide.s32 \t%rd32, %r31, 2;"); // N*32*2 bytes
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
        s.push_str(&format!(
            "\tcp.async.cg.shared.global [ %r{} + 0 ], [ %rd{} + 0 ], 0x10, %r49;\n",
            50 + i,
            28 + i
        ));
    }
    w(s, "cp.async.commit_group;");
    // B tile 1 (buffer 1 = +8192 from B base)
    for (i, off) in [24576i32, 26624, 28672, 30720].iter().enumerate() {
        s.push_str(&format!("\tadd.s32 \t%r{}, %r40, {off};\n", 54 + i));
        s.push_str(&format!(
            "\tcp.async.cg.shared.global [ %r{} + 0 ], [ %rd{} + 0 ], 0x10, %r49;\n",
            54 + i,
            33 + i
        ));
    }
    w(s, "cp.async.commit_group;");
    blank(s);

    // ─── Loop setup ───
    w(s, "@%p1 bra \t$L_LOOP_SETUP;");
    w(s, "bra.uni \t$L_K0_FALLTHROUGH;");
    blank(s);

    s.push_str("$L_LOOP_SETUP:\n");
    w(s, "add.s32 \t%r60, %r3, -64;"); // K - 2*BK

    // A ldmatrix offsets (matching Triton lines 216-232)
    w(s, "shl.b32 \t%r58, %r6, 3;"); // tid*8
    w(s, "and.b32 \t%r59, %r6, 16;"); // tid & 16
    w(s, "shl.b32 \t%r62, %r15, 6;"); // (tid&15)<<6
    w(s, "and.b32 \t%r63, %r58, 48;"); // (tid*8)&48
    w(s, "and.b32 \t%r64, %r32, 1024;"); // (tid*16)&1024
    w(s, "or.b32 \t%r65, %r62, %r63;");
    w(s, "xor.b32 \t%r66, %r65, %r59;");
    w(s, "or.b32 \t%r67, %r66, %r64;"); // a_off_ki0
    w(s, "xor.b32 \t%r68, %r67, 32;"); // a_off_ki1
    blank(s);

    // B ldmatrix offsets (matching Triton lines 223-232)
    w(s, "shl.b32 \t%r69, %r6, 8;"); // tid<<8
    w(s, "and.b32 \t%r70, %r69, 7936;");
    w(s, "and.b32 \t%r71, %r32, 112;"); // (tid*16)&112
    w(s, "shr.u32 \t%r72, %r6, 1;");
    w(s, "and.b32 \t%r73, %r72, 16;");
    w(s, "xor.b32 \t%r74, %r71, %r73;");
    w(s, "or.b32 \t%r75, %r74, %r70;"); // b_off_rn0
    w(s, "xor.b32 \t%r76, %r75, 32;");
    w(s, "xor.b32 \t%r77, %r75, 64;");
    w(s, "xor.b32 \t%r78, %r75, 96;");
    blank(s);

    // Loop pointers: start at tile 2 (after 2 prologue loads)
    w(s, "add.s64 \t%rd50, %rd15, 128;"); // A tile2
    w(s, "add.s64 \t%rd51, %rd16, 128;");
    w(s, "add.s64 \t%rd52, %rd17, 128;");
    w(s, "add.s64 \t%rd53, %rd18, 128;");
    w(s, "shl.b64 \t%rd45, %rd32, 1;"); // 2 × N*32*2
    w(s, "add.s64 \t%rd54, %rd24, %rd45;"); // B tile2
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
    w(s, "mov.b32 \t%r101, 1;"); // read_ctr
    w(s, "mov.b32 \t%r102, -1;"); // write_ctr
    w(s, "mov.b32 \t%r103, 0;"); // k_counter
    blank(s);

    // ═══════════════════════════════════════════════════════════════
    // K-LOOP
    // ═══════════════════════════════════════════════════════════════
    s.push_str("$L_KLOOP:\n");
    w(s, "setp.lt.s32 \t%p3, %r103, %r60;"); // k_counter < K-64

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
    w(s, "add.s32 \t%r107, %r106, %r67;"); // A ki0 addr
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r108, %r109, %r110, %r111}, [%r107];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r112, %r113, %r114, %r115}, [%r107+2048];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r116, %r117, %r118, %r119}, [%r107+4096];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r120, %r121, %r122, %r123}, [%r107+6144];",
    );
    w(s, "add.s32 \t%r124, %r106, %r68;"); // A ki1 addr
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r125, %r126, %r127, %r128}, [%r124];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r129, %r130, %r131, %r132}, [%r124+2048];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r133, %r134, %r135, %r136}, [%r124+4096];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r137, %r138, %r139, %r140}, [%r124+6144];",
    );
    blank(s);

    // ─── ldmatrix B transposed (8 loads) ───
    w(s, "add.s32 \t%r141, %r106, %r75;"); // B rn0
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r142, %r143, %r144, %r145}, [%r141+16384];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r146, %r147, %r148, %r149}, [%r141+16512];",
    );
    w(s, "add.s32 \t%r150, %r106, %r76;"); // B rn1
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r151, %r152, %r153, %r154}, [%r150+16384];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r155, %r156, %r157, %r158}, [%r150+16512];",
    );
    w(s, "add.s32 \t%r159, %r106, %r77;"); // B rn2
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r160, %r161, %r162, %r163}, [%r159+16384];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r164, %r165, %r166, %r167}, [%r159+16512];",
    );
    w(s, "add.s32 \t%r168, %r106, %r78;"); // B rn3
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r169, %r170, %r171, %r172}, [%r168+16384];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r173, %r174, %r175, %r176}, [%r168+16512];",
    );
    blank(s);

    // ─── MMA (64 total) ───
    // ki0: A{0..3} × B_ki0{0..7}
    let a_ki0 = [
        [108, 109, 110, 111],
        [112, 113, 114, 115],
        [116, 117, 118, 119],
        [120, 121, 122, 123],
    ];
    // B ki0: from ldmatrix.trans first 2 regs of each
    let b_ki0 = [
        [142, 143],
        [151, 152],
        [160, 161],
        [169, 170],
        [146, 147],
        [155, 156],
        [164, 165],
        [173, 174],
    ];
    let a_ki1 = [
        [125, 126, 127, 128],
        [129, 130, 131, 132],
        [133, 134, 135, 136],
        [137, 138, 139, 140],
    ];
    let b_ki1 = [
        [144, 145],
        [153, 154],
        [162, 163],
        [171, 172],
        [148, 149],
        [157, 158],
        [166, 167],
        [175, 176],
    ];

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
    w(s, "add.s32 \t%r180, %r179, %r36;"); // write A base
    w(s, "selp.b32 \t%r181, 16, 0, %p3;");

    // A next-tile (4 cp.async)
    w(
        s,
        "cp.async.cg.shared.global [ %r180 + 0 ], [ %rd50 + 0 ], 0x10, %r181;",
    );
    w(s, "add.s32 \t%r182, %r180, 2048;");
    w(
        s,
        "cp.async.cg.shared.global [ %r182 + 0 ], [ %rd51 + 0 ], 0x10, %r181;",
    );
    w(s, "add.s32 \t%r183, %r180, 4096;");
    w(
        s,
        "cp.async.cg.shared.global [ %r183 + 0 ], [ %rd52 + 0 ], 0x10, %r181;",
    );
    w(s, "add.s32 \t%r184, %r180, 6144;");
    w(
        s,
        "cp.async.cg.shared.global [ %r184 + 0 ], [ %rd53 + 0 ], 0x10, %r181;",
    );
    w(s, "cp.async.commit_group;");

    // B next-tile (4 cp.async)
    w(s, "add.s32 \t%r185, %r179, %r39;");
    w(s, "add.s32 \t%r186, %r185, 16384;");
    w(
        s,
        "cp.async.cg.shared.global [ %r186 + 0 ], [ %rd54 + 0 ], 0x10, %r181;",
    );
    w(s, "add.s32 \t%r187, %r185, 18432;");
    w(
        s,
        "cp.async.cg.shared.global [ %r187 + 0 ], [ %rd55 + 0 ], 0x10, %r181;",
    );
    w(s, "add.s32 \t%r188, %r185, 20480;");
    w(
        s,
        "cp.async.cg.shared.global [ %r188 + 0 ], [ %rd56 + 0 ], 0x10, %r181;",
    );
    w(s, "add.s32 \t%r189, %r185, 22528;");
    w(
        s,
        "cp.async.cg.shared.global [ %r189 + 0 ], [ %rd57 + 0 ], 0x10, %r181;",
    );
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

    w(s, "and.b32 \t%r328, %r6, 31;"); // lane
    w(s, "shr.u32 \t%r329, %r6, 5;"); // warp_id
    w(s, "shr.u32 \t%r330, %r329, 1;"); // warp_m (0 or 1)
    w(s, "and.b32 \t%r331, %r329, 1;"); // warp_n (0 or 1)
    w(s, "shr.u32 \t%r332, %r328, 2;"); // mma_row (0..7)
    w(s, "and.b32 \t%r333, %r328, 3;"); // lane & 3
    w(s, "shl.b32 \t%r334, %r333, 1;"); // mma_col (0,2,4,6)
    blank(s);

    // base_row = block_row + warp_m*16 + mma_row
    w(s, "shl.b32 \t%r335, %r330, 4;"); // warp_m * 16
    w(s, "add.s32 \t%r336, %r8, %r335;"); // block_row + warp_m*16
    w(s, "add.s32 \t%r337, %r336, %r332;"); // + mma_row = base_row
    blank(s);

    // base_col = block_col + warp_n*64 + mma_col
    w(s, "shl.b32 \t%r338, %r331, 6;"); // warp_n * 64
    w(s, "add.s32 \t%r339, %r7, %r338;"); // block_col + warp_n*64
    w(s, "add.s32 \t%r340, %r339, %r334;"); // + mma_col = base_col
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
            s.push_str(&format!(
                "\tst.global.v2.b32 [ %rd73 + 0 ], {{ %r{d0}, %r{d1} }};\n"
            ));
            s.push_str(&format!(
                "\tst.global.v2.b32 [ %rd74 + 0 ], {{ %r{d2}, %r{d3} }};\n"
            ));

            acc += 4;
        }
    }

    w(s, "ret;");
    s.push_str("}\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// Fused RmsNorm → GEMM → SiLU  megakernel (128×128 tile)
//
// Same proven 48 TFLOPS GEMM core with:
//   Phase 0: RmsNorm — compute per-row rsqrt(mean_sq + eps), preload gamma
//   Phase 1: GEMM K-loop — after each ldmatrix A, in-place normalize fragments
//   Phase 2: SiLU epilogue — x * sigmoid(x) on each f32 accumulator
// ═══════════════════════════════════════════════════════════════════════════

pub fn emit_fused_128x128() -> String {
    let mut s = String::with_capacity(120 * 1024);
    emit_fused_kernel(&mut s);
    s
}

fn emit_fused_kernel(s: &mut String) {
    // ─── Header ───
    s.push_str(
        r#".version 8.7
.target sm_89
.address_size 64

.extern .shared .align 16 .b8 global_smem[];

.visible .entry fused_rmsnorm_gemm_silu(
	.param .u64 .ptr .global .align 16 param_input,
	.param .u64 .ptr .global .align 16 param_wnorm,
	.param .u64 .ptr .global .align 16 param_wgemm,
	.param .u64 .ptr .global .align 16 param_output,
	.param .u32 param_N,
	.param .u32 param_K
)
.reqntid 128
{
	.reg .pred 	%p<16>;
	.reg .b16 	%h<16>;
	.reg .f32 	%f<140>;
	.reg .b32 	%r<520>;
	.reg .b64 	%rd<130>;

"#,
    );

    // ─── Parameter loads ───
    // param_input  = A matrix [M, K] f16  (to be RmsNorm'd)
    // param_wnorm  = gamma [K] f16
    // param_wgemm  = B matrix [K, N] f16
    // param_output = C matrix [M, N] f32
    w(s, "ld.param.b64 \t%rd1, [param_input];"); // A (input)
    w(s, "ld.param.b64 \t%rd2, [param_wgemm];"); // B (gemm weight)
    w(s, "ld.param.b64 \t%rd3, [param_output];"); // C (output)
    w(s, "ld.param.b64 \t%rd4, [param_wnorm];"); // gamma (rmsnorm weight)
    w(s, "ld.param.b32 \t%r2, [param_N];"); // N
    w(s, "ld.param.b32 \t%r3, [param_K];"); // K
    blank(s);

    // ─── Thread/block indexing ───
    // For the fused kernel, block_y selects rows (batch dimension)
    // block_x selects output columns
    w(s, "mov.u32 \t%r4, %ctaid.x;"); // block_n
    w(s, "mov.u32 \t%r5, %ctaid.y;"); // block_m
    w(s, "shl.b32 \t%r7, %r4, 7;"); // block_col = block_n * 128
    w(s, "shl.b32 \t%r8, %r5, 7;"); // block_row = block_m * 128
    w(s, "mov.u32 \t%r6, %tid.x;"); // tid
    blank(s);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 0: RmsNorm — compute norm factors for 128 rows
    // ═══════════════════════════════════════════════════════════════
    //
    // Each of the 128 threads handles one row.
    // Compute sum_sq over K elements, then rsqrt(sum_sq/K + eps).
    // Store 128 f32 norm factors at smem[32768..33279].
    // Then preload gamma[0..K-1] f16 at smem[33280..33280+2*K-1].

    // Row index for this thread's norm computation
    w(s, "add.s32 \t%r400, %r8, %r6;"); // global_row = block_row + tid

    // Pointer to this row in input: input + global_row * K * 2
    w(s, "mul.lo.s32 \t%r401, %r400, %r3;"); // global_row * K
    w(s, "mad.wide.s32 \t%rd80, %r401, 2, %rd1;"); // &input[global_row][0]
    blank(s);

    // Sum of squares over K elements (loop with f32 accumulation)
    // Process 8 f16 elements per iteration (= 16 bytes = ld.global.v4.b32)
    w(s, "mov.f32 \t%f0, 0f00000000;"); // sum_sq = 0.0
    w(s, "mov.s32 \t%r402, 0;"); // k_idx = 0 (in elements)
    w(s, "setp.gt.s32 \t%p10, %r3, 0;");
    w(s, "@!%p10 bra \t$L_NORM_DONE;");
    blank(s);

    s.push_str("$L_NORM_LOOP:\n");
    // Load 8 f16 elements (4 × b32) per iteration via v4.b32
    w(s, "mul.wide.s32 \t%rd81, %r402, 2;");
    w(s, "add.s64 \t%rd82, %rd80, %rd81;");
    w(
        s,
        "ld.global.v4.b32 \t{%r500, %r501, %r502, %r503}, [%rd82];",
    );
    // Unpack all 8 f16 to f32 and accumulate
    w(s, "mov.b32 \t{%h0, %h1}, %r500;");
    w(s, "cvt.f32.f16 \t%f1, %h0;");
    w(s, "cvt.f32.f16 \t%f2, %h1;");
    w(s, "fma.rn.f32 \t%f0, %f1, %f1, %f0;");
    w(s, "fma.rn.f32 \t%f0, %f2, %f2, %f0;");
    w(s, "mov.b32 \t{%h0, %h1}, %r501;");
    w(s, "cvt.f32.f16 \t%f1, %h0;");
    w(s, "cvt.f32.f16 \t%f2, %h1;");
    w(s, "fma.rn.f32 \t%f0, %f1, %f1, %f0;");
    w(s, "fma.rn.f32 \t%f0, %f2, %f2, %f0;");
    w(s, "mov.b32 \t{%h0, %h1}, %r502;");
    w(s, "cvt.f32.f16 \t%f1, %h0;");
    w(s, "cvt.f32.f16 \t%f2, %h1;");
    w(s, "fma.rn.f32 \t%f0, %f1, %f1, %f0;");
    w(s, "fma.rn.f32 \t%f0, %f2, %f2, %f0;");
    w(s, "mov.b32 \t{%h0, %h1}, %r503;");
    w(s, "cvt.f32.f16 \t%f1, %h0;");
    w(s, "cvt.f32.f16 \t%f2, %h1;");
    w(s, "fma.rn.f32 \t%f0, %f1, %f1, %f0;");
    w(s, "fma.rn.f32 \t%f0, %f2, %f2, %f0;");
    w(s, "add.s32 \t%r402, %r402, 8;");
    w(s, "setp.lt.s32 \t%p11, %r402, %r3;");
    w(s, "@%p11 bra \t$L_NORM_LOOP;");
    blank(s);

    s.push_str("$L_NORM_DONE:\n");
    // norm_factor = rsqrt(sum_sq / K + eps)
    w(s, "cvt.rn.f32.s32 \t%f3, %r3;"); // K as float
    w(s, "div.rn.f32 \t%f4, %f0, %f3;"); // mean_sq = sum_sq / K
    w(s, "add.f32 \t%f4, %f4, 0f358637BD;"); // + eps (1e-6)
    w(s, "rsqrt.approx.f32 \t%f5, %f4;"); // rsqrt(mean_sq + eps)
    blank(s);

    // Store norm factor to smem[32768 + tid * 4]
    // %r404 = global_smem base (reused throughout)
    w(s, "mov.b32 \t%r404, global_smem;");
    w(s, "shl.b32 \t%r405, %r6, 2;"); // tid * 4
    w(s, "add.s32 \t%r419, %r404, 32768;"); // norm_smem_base = smem + 32768
    w(s, "add.s32 \t%r406, %r419, %r405;"); // norm_smem_base + tid*4
    w(s, "st.shared.f32 \t[%r406], %f5;");
    blank(s);

    // ─── Preload gamma weights into smem[33280..33280+2*K-1] ───
    // Each thread loads K/128 f16 elements. Use v4.b32 loads (8 f16 per iter).
    // For K=4096, each thread loads 32 elements = 4 iterations of 8.
    // gamma smem base = smem + 33280
    w(s, "bar.sync \t0;");
    w(s, "add.s32 \t%r420, %r404, 33280;"); // gamma_smem_base
    w(s, "shr.u32 \t%r407, %r3, 7;"); // K / 128 (elems per thread)
    w(s, "mul.lo.s32 \t%r408, %r6, %r407;"); // tid * (K/128) = start elem
    w(s, "add.s32 \t%r409, %r408, %r407;"); // end elem
    w(s, "mov.s32 \t%r410, %r408;"); // loop var (element index)
    blank(s);

    s.push_str("$L_GAMMA_LOAD:\n");
    w(s, "setp.lt.s32 \t%p12, %r410, %r409;");
    w(s, "@!%p12 bra \t$L_GAMMA_DONE;");
    // Load 8 f16 elements (v4.b32 = 16 bytes) from global
    w(s, "mul.wide.s32 \t%rd83, %r410, 2;"); // byte offset
    w(s, "add.s64 \t%rd84, %rd4, %rd83;");
    w(
        s,
        "ld.global.v4.b32 \t{%r500, %r501, %r502, %r503}, [%rd84];",
    );
    // Store 16 bytes to smem as v4.b32
    w(s, "shl.b32 \t%r412, %r410, 1;"); // element * 2 = byte offset
    w(s, "add.s32 \t%r413, %r420, %r412;");
    w(
        s,
        "st.shared.v4.b32 \t[%r413], {%r500, %r501, %r502, %r503};",
    );
    w(s, "add.s32 \t%r410, %r410, 8;");
    w(s, "bra.uni \t$L_GAMMA_LOAD;");
    blank(s);

    s.push_str("$L_GAMMA_DONE:\n");
    w(s, "bar.sync \t0;");
    blank(s);

    // ═══════════════════════════════════════════════════════════════
    // Load norm factors into registers for the K-loop
    // ═══════════════════════════════════════════════════════════════
    //
    // Each thread in a warp processes specific rows determined by:
    //   warp_m = warp_id / 2 (0 or 1) -> row offset 0 or 64
    //   lane = tid & 31
    //   group_id = lane / 4 (0..7) -> mma row within m16 tile
    //
    // For rm=0..3, the rows are:
    //   row_lo = warp_m*64 + rm*16 + group_id  (within block)
    //   row_hi = row_lo + 8
    //
    // We need norm factors for these 8 rows (4 rm × 2 halves).
    // Load them from smem and pack as f16x2 (same value in both halves).

    w(s, "shr.u32 \t%r329, %r6, 5;"); // warp_id
    w(s, "shr.u32 \t%r330, %r329, 1;"); // warp_m
    w(s, "and.b32 \t%r331, %r329, 1;"); // warp_n
    w(s, "and.b32 \t%r328, %r6, 31;"); // lane
    w(s, "shr.u32 \t%r415, %r328, 2;"); // group_id = lane / 4
    blank(s);

    // warp_m_offset = warp_m * 64
    w(s, "shl.b32 \t%r416, %r330, 6;");

    // %r419 = norm_smem_base = smem + 32768 (already computed above)
    // Load 8 norm factors: rm=0..3 x {lo, hi}
    for rm in 0..4u32 {
        let row_base_off = rm * 16;
        // lo row index (within block) = warp_m*64 + rm*16 + group_id
        s.push_str(&format!("\tadd.s32 \t%r417, %r416, {};\n", row_base_off));
        s.push_str("\tadd.s32 \t%r417, %r417, %r415;\n");
        // smem addr = norm_smem_base + row * 4
        s.push_str("\tshl.b32 \t%r418, %r417, 2;\n");
        s.push_str("\tadd.s32 \t%r418, %r419, %r418;\n");
        s.push_str(&format!("\tld.shared.f32 \t%f{}, [%r418];\n", 10 + rm * 2));
        // hi row = lo row + 8 -> addr + 32 bytes
        s.push_str("\tadd.s32 \t%r418, %r418, 32;\n");
        s.push_str(&format!("\tld.shared.f32 \t%f{}, [%r418];\n", 11 + rm * 2));
    }
    blank(s);

    // Convert norm factors to f16 and pack as f16x2 (same value both halves)
    // %f10..%f17 are the 8 norm factors (rm0_lo, rm0_hi, rm1_lo, rm1_hi, ...)
    // Pack into %r470..%r477 as f16x2
    for i in 0..8u32 {
        s.push_str(&format!("\tcvt.rn.f16.f32 \t%h2, %f{};\n", 10 + i));
        s.push_str(&format!("\tmov.b32 \t%r{}, {{%h2, %h2}};\n", 470 + i));
    }
    blank(s);

    // ═══════════════════════════════════════════════════════════════
    // STANDARD GEMM SETUP (identical to standalone 128×128)
    // ═══════════════════════════════════════════════════════════════

    // ─── cp.async thread mapping ───
    w(s, "bfe.u32 \t%r9, %r6, 2, 5;"); // A_row within chunk
    w(s, "and.b32 \t%r18, %r6, 3;");
    w(s, "shl.b32 \t%r19, %r18, 3;"); // A_col = (tid&3)*8
    w(s, "bfe.u32 \t%r11, %r6, 4, 3;"); // B_row within chunk
    w(s, "or.b32 \t%r12, %r11, 8;");
    w(s, "or.b32 \t%r13, %r11, 16;");
    w(s, "or.b32 \t%r14, %r11, 24;");
    w(s, "and.b32 \t%r15, %r6, 15;");
    w(s, "shl.b32 \t%r16, %r15, 3;"); // B_col = (tid&15)*8
    blank(s);

    // ─── A global pointers (4 chunks × 32 rows) ───
    w(s, "or.b32 \t%r17, %r9, %r8;");
    w(s, "mul.lo.s32 \t%r20, %r3, %r17;");
    w(s, "shl.b32 \t%r21, %r3, 5;");
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
    w(s, "or.b32 \t%r25, %r16, %r7;");
    w(s, "mul.lo.s32 \t%r26, %r2, %r11;");
    w(s, "shl.b32 \t%r27, %r2, 3;");
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
    w(s, "shl.b32 \t%r31, %r2, 5;");
    blank(s);

    // ─── Smem swizzle for cp.async ───
    w(s, "shl.b32 \t%r32, %r6, 4;");
    w(s, "and.b32 \t%r33, %r32, 2032;");
    w(s, "and.b32 \t%r34, %r6, 24;");
    w(s, "shl.b32 \t%r35, %r34, 1;");
    w(s, "xor.b32 \t%r36, %r33, %r35;");
    w(s, "mov.b32 \t%r37, global_smem;");
    w(s, "add.s32 \t%r38, %r37, %r36;");
    w(s, "and.b32 \t%r10, %r6, 112;");
    w(s, "xor.b32 \t%r39, %r33, %r10;");
    w(s, "add.s32 \t%r40, %r37, %r39;");
    blank(s);

    // ─── Prologue: load tile 0 ───
    w(s, "setp.gt.s32 \t%p1, %r3, 0;");
    w(s, "selp.b32 \t%r41, 16, 0, %p1;");
    for (i, off) in [0i32, 2048, 4096, 6144].iter().enumerate() {
        if *off > 0 {
            s.push_str(&format!("\tadd.s32 \t%r{}, %r38, {off};\n", 42 + i - 1));
            s.push_str(&format!(
                "\tcp.async.cg.shared.global [ %r{} + 0 ], [ %rd{} + 0 ], 0x10, %r41;\n",
                42 + i - 1,
                15 + i
            ));
        } else {
            s.push_str("\tcp.async.cg.shared.global [ %r38 + 0 ], [ %rd15 + 0 ], 0x10, %r41;\n");
        }
    }
    w(s, "cp.async.commit_group;");
    for (i, off) in [16384i32, 18432, 20480, 22528].iter().enumerate() {
        s.push_str(&format!("\tadd.s32 \t%r{}, %r40, {off};\n", 45 + i));
        s.push_str(&format!(
            "\tcp.async.cg.shared.global [ %r{} + 0 ], [ %rd{} + 0 ], 0x10, %r41;\n",
            45 + i,
            24 + i
        ));
    }
    w(s, "cp.async.commit_group;");
    blank(s);

    // ─── Advance for tile 1 ───
    w(s, "setp.gt.s32 \t%p2, %r3, 32;");
    w(s, "add.s64 \t%rd28, %rd15, 64;");
    w(s, "add.s64 \t%rd29, %rd16, 64;");
    w(s, "add.s64 \t%rd30, %rd17, 64;");
    w(s, "add.s64 \t%rd31, %rd18, 64;");
    w(s, "mul.wide.s32 \t%rd32, %r31, 2;");
    w(s, "add.s64 \t%rd33, %rd24, %rd32;");
    w(s, "add.s64 \t%rd34, %rd25, %rd32;");
    w(s, "add.s64 \t%rd35, %rd26, %rd32;");
    w(s, "add.s64 \t%rd36, %rd27, %rd32;");
    blank(s);

    // Load tile 1 into buffer 1
    w(s, "bar.sync \t0;");
    w(s, "selp.b32 \t%r49, 16, 0, %p2;");
    for (i, off) in [8192i32, 10240, 12288, 14336].iter().enumerate() {
        s.push_str(&format!("\tadd.s32 \t%r{}, %r38, {off};\n", 50 + i));
        s.push_str(&format!(
            "\tcp.async.cg.shared.global [ %r{} + 0 ], [ %rd{} + 0 ], 0x10, %r49;\n",
            50 + i,
            28 + i
        ));
    }
    w(s, "cp.async.commit_group;");
    for (i, off) in [24576i32, 26624, 28672, 30720].iter().enumerate() {
        s.push_str(&format!("\tadd.s32 \t%r{}, %r40, {off};\n", 54 + i));
        s.push_str(&format!(
            "\tcp.async.cg.shared.global [ %r{} + 0 ], [ %rd{} + 0 ], 0x10, %r49;\n",
            54 + i,
            33 + i
        ));
    }
    w(s, "cp.async.commit_group;");
    blank(s);

    // ─── Loop setup ───
    w(s, "@%p1 bra \t$L_LOOP_SETUP;");
    w(s, "bra.uni \t$L_K0_FALLTHROUGH;");
    blank(s);

    s.push_str("$L_LOOP_SETUP:\n");
    w(s, "add.s32 \t%r60, %r3, -64;");

    // A ldmatrix offsets
    w(s, "shl.b32 \t%r58, %r6, 3;");
    w(s, "and.b32 \t%r59, %r6, 16;");
    w(s, "shl.b32 \t%r62, %r15, 6;");
    w(s, "and.b32 \t%r63, %r58, 48;");
    w(s, "and.b32 \t%r64, %r32, 1024;");
    w(s, "or.b32 \t%r65, %r62, %r63;");
    w(s, "xor.b32 \t%r66, %r65, %r59;");
    w(s, "or.b32 \t%r67, %r66, %r64;");
    w(s, "xor.b32 \t%r68, %r67, 32;");
    blank(s);

    // B ldmatrix offsets
    w(s, "shl.b32 \t%r69, %r6, 8;");
    w(s, "and.b32 \t%r70, %r69, 7936;");
    w(s, "and.b32 \t%r71, %r32, 112;");
    w(s, "shr.u32 \t%r72, %r6, 1;");
    w(s, "and.b32 \t%r73, %r72, 16;");
    w(s, "xor.b32 \t%r74, %r71, %r73;");
    w(s, "or.b32 \t%r75, %r74, %r70;");
    w(s, "xor.b32 \t%r76, %r75, 32;");
    w(s, "xor.b32 \t%r77, %r75, 64;");
    w(s, "xor.b32 \t%r78, %r75, 96;");
    blank(s);

    // Loop pointers
    w(s, "add.s64 \t%rd50, %rd15, 128;");
    w(s, "add.s64 \t%rd51, %rd16, 128;");
    w(s, "add.s64 \t%rd52, %rd17, 128;");
    w(s, "add.s64 \t%rd53, %rd18, 128;");
    w(s, "shl.b64 \t%rd45, %rd32, 1;");
    w(s, "add.s64 \t%rd54, %rd24, %rd45;");
    w(s, "add.s64 \t%rd55, %rd25, %rd45;");
    w(s, "add.s64 \t%rd56, %rd26, %rd45;");
    w(s, "add.s64 \t%rd57, %rd27, %rd45;");
    blank(s);

    // Initialize accumulators
    w(s, "mov.b32 \t%r100, 0x00000000;");
    for i in 200..328u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r100;\n"));
    }
    blank(s);

    // Buffer toggle
    w(s, "mov.b32 \t%r101, 1;");
    w(s, "mov.b32 \t%r102, -1;");
    w(s, "mov.b32 \t%r103, 0;");
    blank(s);

    // ═══════════════════════════════════════════════════════════════
    // K-LOOP (with in-place RmsNorm transform on A fragments)
    // ═══════════════════════════════════════════════════════════════
    s.push_str("$L_KLOOP:\n");
    w(s, "setp.lt.s32 \t%p3, %r103, %r60;");

    // Toggle read buffer
    w(s, "add.s32 \t%r104, %r101, 1;");
    w(s, "setp.gt.s32 \t%p4, %r104, 1;");
    w(s, "selp.b32 \t%r101, 0, %r104, %p4;");

    w(s, "cp.async.wait_group \t2;");
    w(s, "bar.sync \t0;");

    w(s, "shl.b32 \t%r105, %r101, 13;");
    w(s, "add.s32 \t%r106, %r37, %r105;");
    blank(s);

    // ─── Load gamma for current K-tile from preloaded smem ───
    //
    // For m16n8k16 MMA, A-operand register mapping per thread:
    //   tg = lane % 4 (tid_in_group)
    //   reg0: k = tg*2, tg*2+1       (rows: group_id)
    //   reg1: k = tg*2, tg*2+1       (rows: group_id+8)
    //   reg2: k = 8+tg*2, 8+tg*2+1   (rows: group_id)
    //   reg3: k = 8+tg*2, 8+tg*2+1   (rows: group_id+8)
    //
    // So we need 4 gamma packed values (b32 = 2 consecutive f16):
    //   gamma_ki0_lo: gamma[k_counter + tg*2]       (for reg0, reg1)
    //   gamma_ki0_hi: gamma[k_counter + 8 + tg*2]   (for reg2, reg3)
    //   gamma_ki1_lo: gamma[k_counter + 16 + tg*2]  (for reg0, reg1 of ki1)
    //   gamma_ki1_hi: gamma[k_counter + 24 + tg*2]  (for reg2, reg3 of ki1)

    w(s, "and.b32 \t%r480, %r6, 3;"); // tg = tid & 3
    w(s, "shl.b32 \t%r481, %r480, 1;"); // tg * 2
    w(s, "add.s32 \t%r482, %r103, %r481;"); // k_counter + tg*2

    // gamma smem base in %r420 (already computed: smem + 33280)
    // gamma smem byte addr = gamma_smem_base + element_index * 2
    w(s, "shl.b32 \t%r484, %r482, 1;"); // (k_counter + tg*2) * 2
    w(s, "add.s32 \t%r486, %r420, %r484;"); // &gamma[k_counter + tg*2]
    w(s, "ld.shared.b32 \t%r487, [%r486];"); // gamma_ki0_lo
    w(s, "ld.shared.b32 \t%r491, [%r486 + 16];"); // gamma_ki0_hi (+8 elems * 2 bytes)
    w(s, "ld.shared.b32 \t%r490, [%r486 + 32];"); // gamma_ki1_lo (+16 elems * 2 bytes)
    w(s, "ld.shared.b32 \t%r492, [%r486 + 48];"); // gamma_ki1_hi (+24 elems * 2 bytes)
    blank(s);

    // ─── ldmatrix A (8 loads) + in-place RmsNorm transform ───
    w(s, "add.s32 \t%r107, %r106, %r67;");
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r108, %r109, %r110, %r111}, [%r107];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r112, %r113, %r114, %r115}, [%r107+2048];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r116, %r117, %r118, %r119}, [%r107+4096];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r120, %r121, %r122, %r123}, [%r107+6144];",
    );
    w(s, "add.s32 \t%r124, %r106, %r68;");
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r125, %r126, %r127, %r128}, [%r124];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r129, %r130, %r131, %r132}, [%r124+2048];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r133, %r134, %r135, %r136}, [%r124+4096];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r137, %r138, %r139, %r140}, [%r124+6144];",
    );
    blank(s);

    // ─── In-place RmsNorm transform on A fragments ───
    // For each ldmatrix A group (rm=0..3):
    //   Registers hold packed f16x2 with rows (group_id, group_id+8)
    //   within the rm*16 tile.
    //
    //   Norm factor for lo half = %r{470 + rm*2}  (packed f16x2: norm[row_lo])
    //   Norm factor for hi half = %r{471 + rm*2}  (packed f16x2: norm[row_hi])
    //
    //   But ldmatrix packs (row_lo, row_hi) into a single b32 register:
    //   low 16 bits = f16 from row_lo, high 16 bits = f16 from row_hi.
    //   So we need a MIXED norm factor: {norm_lo_f16, norm_hi_f16}.
    //
    //   Actually, for m16n8k16 MMA, each b32 A-fragment register contains
    //   two f16 values from the SAME row (consecutive k-columns).
    //   So norm factor is the same for both halves.
    //
    //   Wait -- let's be precise. For m16n8k16:
    //   A operand: 4 b32 registers. Each register holds 2 f16 at (row, k) and (row, k+1).
    //   The row mapping across the 4 registers for thread with group_id g:
    //     reg0: row = g      (k positions set by tg)
    //     reg1: row = g + 8
    //     reg2: row = g      (different k)
    //     reg3: row = g + 8  (different k)
    //
    //   So for each 4-register group from ldmatrix (rm=0..3):
    //     reg0, reg2 -> row = group_id + rm*16 -> norm = %r{470 + rm*2}
    //     reg1, reg3 -> row = group_id + 8 + rm*16 -> norm = %r{471 + rm*2}

    // ki0 fragments: regs 108-123 (4 groups of 4)
    // rm=0: 108,109,110,111  rm=1: 112,113,114,115  rm=2: 116,117,118,119  rm=3: 120,121,122,123
    let a_ki0_regs = [
        [108, 109, 110, 111],
        [112, 113, 114, 115],
        [116, 117, 118, 119],
        [120, 121, 122, 123],
    ];
    let a_ki1_regs = [
        [125, 126, 127, 128],
        [129, 130, 131, 132],
        [133, 134, 135, 136],
        [137, 138, 139, 140],
    ];

    // Apply norm + gamma to ki0 fragments
    // Gamma registers: %r487 = ki0_lo (regs 0,1), %r491 = ki0_hi (regs 2,3)
    //                  %r490 = ki1_lo (regs 0,1), %r492 = ki1_hi (regs 2,3)
    for rm in 0..4u32 {
        let norm_lo = 470 + rm * 2; // norm for row = group_id (regs 0,2)
        let norm_hi = 471 + rm * 2; // norm for row = group_id+8 (regs 1,3)
        let regs = &a_ki0_regs[rm as usize];
        // reg0 (row_lo, k_lo): norm_lo * gamma_ki0_lo
        s.push_str(&format!(
            "\tmul.rn.f16x2 \t%r{r}, %r{r}, %r{n};\n",
            r = regs[0],
            n = norm_lo
        ));
        s.push_str(&format!(
            "\tmul.rn.f16x2 \t%r{r}, %r{r}, %r487;\n",
            r = regs[0]
        ));
        // reg1 (row_hi, k_lo): norm_hi * gamma_ki0_lo
        s.push_str(&format!(
            "\tmul.rn.f16x2 \t%r{r}, %r{r}, %r{n};\n",
            r = regs[1],
            n = norm_hi
        ));
        s.push_str(&format!(
            "\tmul.rn.f16x2 \t%r{r}, %r{r}, %r487;\n",
            r = regs[1]
        ));
        // reg2 (row_lo, k_hi): norm_lo * gamma_ki0_hi
        s.push_str(&format!(
            "\tmul.rn.f16x2 \t%r{r}, %r{r}, %r{n};\n",
            r = regs[2],
            n = norm_lo
        ));
        s.push_str(&format!(
            "\tmul.rn.f16x2 \t%r{r}, %r{r}, %r491;\n",
            r = regs[2]
        ));
        // reg3 (row_hi, k_hi): norm_hi * gamma_ki0_hi
        s.push_str(&format!(
            "\tmul.rn.f16x2 \t%r{r}, %r{r}, %r{n};\n",
            r = regs[3],
            n = norm_hi
        ));
        s.push_str(&format!(
            "\tmul.rn.f16x2 \t%r{r}, %r{r}, %r491;\n",
            r = regs[3]
        ));
    }
    // Apply norm + gamma to ki1 fragments
    for rm in 0..4u32 {
        let norm_lo = 470 + rm * 2;
        let norm_hi = 471 + rm * 2;
        let regs = &a_ki1_regs[rm as usize];
        // reg0 (row_lo, k_lo): norm_lo * gamma_ki1_lo
        s.push_str(&format!(
            "\tmul.rn.f16x2 \t%r{r}, %r{r}, %r{n};\n",
            r = regs[0],
            n = norm_lo
        ));
        s.push_str(&format!(
            "\tmul.rn.f16x2 \t%r{r}, %r{r}, %r490;\n",
            r = regs[0]
        ));
        // reg1 (row_hi, k_lo): norm_hi * gamma_ki1_lo
        s.push_str(&format!(
            "\tmul.rn.f16x2 \t%r{r}, %r{r}, %r{n};\n",
            r = regs[1],
            n = norm_hi
        ));
        s.push_str(&format!(
            "\tmul.rn.f16x2 \t%r{r}, %r{r}, %r490;\n",
            r = regs[1]
        ));
        // reg2 (row_lo, k_hi): norm_lo * gamma_ki1_hi
        s.push_str(&format!(
            "\tmul.rn.f16x2 \t%r{r}, %r{r}, %r{n};\n",
            r = regs[2],
            n = norm_lo
        ));
        s.push_str(&format!(
            "\tmul.rn.f16x2 \t%r{r}, %r{r}, %r492;\n",
            r = regs[2]
        ));
        // reg3 (row_hi, k_hi): norm_hi * gamma_ki1_hi
        s.push_str(&format!(
            "\tmul.rn.f16x2 \t%r{r}, %r{r}, %r{n};\n",
            r = regs[3],
            n = norm_hi
        ));
        s.push_str(&format!(
            "\tmul.rn.f16x2 \t%r{r}, %r{r}, %r492;\n",
            r = regs[3]
        ));
    }
    blank(s);

    // ─── ldmatrix B transposed (8 loads) ───
    w(s, "add.s32 \t%r141, %r106, %r75;");
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r142, %r143, %r144, %r145}, [%r141+16384];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r146, %r147, %r148, %r149}, [%r141+16512];",
    );
    w(s, "add.s32 \t%r150, %r106, %r76;");
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r151, %r152, %r153, %r154}, [%r150+16384];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r155, %r156, %r157, %r158}, [%r150+16512];",
    );
    w(s, "add.s32 \t%r159, %r106, %r77;");
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r160, %r161, %r162, %r163}, [%r159+16384];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r164, %r165, %r166, %r167}, [%r159+16512];",
    );
    w(s, "add.s32 \t%r168, %r106, %r78;");
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r169, %r170, %r171, %r172}, [%r168+16384];",
    );
    w(
        s,
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r173, %r174, %r175, %r176}, [%r168+16512];",
    );
    blank(s);

    // ─── MMA (64 total) ───
    let a_ki0 = [
        [108, 109, 110, 111],
        [112, 113, 114, 115],
        [116, 117, 118, 119],
        [120, 121, 122, 123],
    ];
    let b_ki0 = [
        [142, 143],
        [151, 152],
        [160, 161],
        [169, 170],
        [146, 147],
        [155, 156],
        [164, 165],
        [173, 174],
    ];
    let a_ki1 = [
        [125, 126, 127, 128],
        [129, 130, 131, 132],
        [133, 134, 135, 136],
        [137, 138, 139, 140],
    ];
    let b_ki1 = [
        [144, 145],
        [153, 154],
        [162, 163],
        [171, 172],
        [148, 149],
        [157, 158],
        [166, 167],
        [175, 176],
    ];

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
    w(s, "add.s32 \t%r180, %r179, %r36;");
    w(s, "selp.b32 \t%r181, 16, 0, %p3;");

    w(
        s,
        "cp.async.cg.shared.global [ %r180 + 0 ], [ %rd50 + 0 ], 0x10, %r181;",
    );
    w(s, "add.s32 \t%r182, %r180, 2048;");
    w(
        s,
        "cp.async.cg.shared.global [ %r182 + 0 ], [ %rd51 + 0 ], 0x10, %r181;",
    );
    w(s, "add.s32 \t%r183, %r180, 4096;");
    w(
        s,
        "cp.async.cg.shared.global [ %r183 + 0 ], [ %rd52 + 0 ], 0x10, %r181;",
    );
    w(s, "add.s32 \t%r184, %r180, 6144;");
    w(
        s,
        "cp.async.cg.shared.global [ %r184 + 0 ], [ %rd53 + 0 ], 0x10, %r181;",
    );
    w(s, "cp.async.commit_group;");

    w(s, "add.s32 \t%r185, %r179, %r39;");
    w(s, "add.s32 \t%r186, %r185, 16384;");
    w(
        s,
        "cp.async.cg.shared.global [ %r186 + 0 ], [ %rd54 + 0 ], 0x10, %r181;",
    );
    w(s, "add.s32 \t%r187, %r185, 18432;");
    w(
        s,
        "cp.async.cg.shared.global [ %r187 + 0 ], [ %rd55 + 0 ], 0x10, %r181;",
    );
    w(s, "add.s32 \t%r188, %r185, 20480;");
    w(
        s,
        "cp.async.cg.shared.global [ %r188 + 0 ], [ %rd56 + 0 ], 0x10, %r181;",
    );
    w(s, "add.s32 \t%r189, %r185, 22528;");
    w(
        s,
        "cp.async.cg.shared.global [ %r189 + 0 ], [ %rd57 + 0 ], 0x10, %r181;",
    );
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
    w(s, "bra.uni \t$L_SILU;");
    blank(s);

    // ─── K=0 fallthrough ───
    s.push_str("$L_K0_FALLTHROUGH:\n");
    w(s, "mov.b32 \t%r100, 0x00000000;");
    for i in 200..328u32 {
        s.push_str(&format!("\tmov.b32 \t%r{i}, %r100;\n"));
    }
    blank(s);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 2: SiLU epilogue — x * sigmoid(x) on each accumulator
    // ═══════════════════════════════════════════════════════════════
    //
    // sigmoid(x) = 1 / (1 + exp(-x))
    //            = 1 / (1 + 2^(-x * log2(e)))
    //
    // For each f32 accumulator:
    //   neg.f32      %f, %acc;
    //   mul.f32      %f, %f, 0F3FB8AA3B;   // * log2(e)
    //   ex2.approx.f32 %f, %f;
    //   add.f32      %f, %f, 0F3F800000;   // + 1.0
    //   rcp.approx.f32 %f, %f;
    //   mul.f32      %acc, %acc, %f;        // x * sigmoid(x)

    s.push_str("$L_SILU:\n");
    w(s, "cp.async.wait_group \t0;");
    w(s, "bar.sync \t0;");
    blank(s);

    for i in 200..328u32 {
        // Use %f120 as scratch
        s.push_str(&format!("\tmov.b32 \t%f120, %r{i};\n"));
        s.push_str("\tneg.f32 \t%f121, %f120;\n");
        s.push_str("\tmul.f32 \t%f121, %f121, 0F3FB8AA3B;\n"); // log2(e)
        s.push_str("\tex2.approx.f32 \t%f121, %f121;\n");
        s.push_str("\tadd.f32 \t%f121, %f121, 0F3F800000;\n"); // 1.0
        s.push_str("\trcp.approx.f32 \t%f121, %f121;\n");
        s.push_str("\tmul.f32 \t%f120, %f120, %f121;\n"); // x * sigmoid(x)
        s.push_str(&format!("\tmov.b32 \t%r{i}, %f120;\n"));
    }
    blank(s);

    // ═══════════════════════════════════════════════════════════════
    // Store C (f32) — identical to standalone kernel
    // ═══════════════════════════════════════════════════════════════
    // warp_m, warp_n, lane already computed above (%r328..%r331)
    w(s, "shr.u32 \t%r332, %r328, 2;"); // mma_row
    w(s, "and.b32 \t%r333, %r328, 3;");
    w(s, "shl.b32 \t%r334, %r333, 1;"); // mma_col

    w(s, "shl.b32 \t%r335, %r330, 4;"); // warp_m * 16
    w(s, "add.s32 \t%r336, %r8, %r335;");
    w(s, "add.s32 \t%r337, %r336, %r332;"); // base_row

    w(s, "shl.b32 \t%r338, %r331, 6;"); // warp_n * 64
    w(s, "add.s32 \t%r339, %r7, %r338;");
    w(s, "add.s32 \t%r340, %r339, %r334;"); // base_col
    blank(s);

    acc = 200;
    for am in 0..4u32 {
        s.push_str(&format!("\tadd.s32 \t%r350, %r337, {};\n", am * 32));
        s.push_str("\tadd.s32 \t%r351, %r350, 8;\n");
        s.push_str("\tmul.lo.s32 \t%r352, %r2, %r350;\n");
        s.push_str("\tmul.lo.s32 \t%r353, %r2, %r351;\n");
        s.push_str("\tmad.wide.s32 \t%rd70, %r352, 4, %rd3;\n");
        s.push_str("\tmad.wide.s32 \t%rd71, %r353, 4, %rd3;\n");

        for bn in 0..8u32 {
            let d0 = acc;
            let d1 = acc + 1;
            let d2 = acc + 2;
            let d3 = acc + 3;

            s.push_str(&format!("\tadd.s32 \t%r354, %r340, {};\n", bn * 8));
            s.push_str("\tmul.wide.u32 \t%rd72, %r354, 4;\n");
            s.push_str("\tadd.s64 \t%rd73, %rd70, %rd72;\n");
            s.push_str("\tadd.s64 \t%rd74, %rd71, %rd72;\n");
            s.push_str(&format!(
                "\tst.global.v2.b32 [ %rd73 + 0 ], {{ %r{d0}, %r{d1} }};\n"
            ));
            s.push_str(&format!(
                "\tst.global.v2.b32 [ %rd74 + 0 ], {{ %r{d2}, %r{d3} }};\n"
            ));

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
