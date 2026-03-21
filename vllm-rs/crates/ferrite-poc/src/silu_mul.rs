/// SiLU×Up (silu_mul) PTX kernel — LITERAL COPY of Triton-compiled reference.
///
/// Computes: out[i] = silu(gate[i]) * up[i]  where silu(x) = x * sigmoid(x)
///
/// Reference: /tmp/triton_silu_mul.ptx (Triton 3.x, sm_89, BLOCK_SIZE=1024)
/// Verified correct: max err 0.000086 against f64 CPU reference.
///
/// This emitter produces the EXACT same PTX as the Triton compiler, with only
/// param names changed. Following the handoff rule: copy before innovating.

pub fn emit_silu_mul_kernel() -> String {
    // Literal copy of the Triton-compiled PTX, stripped of debug info and labels,
    // with param names changed to our convention.
    // Original: silu_mul_kernel_param_{0..5} (6 params, 2 are Triton workspace)
    // Ours: param_out, param_gate, param_up, param_n_elements (4 params)
    r#".version 8.7
.target sm_89
.address_size 64

.visible .entry silu_mul_kernel(
	.param .u64 .ptr .global .align 1 param_out,
	.param .u64 .ptr .global .align 1 param_gate,
	.param .u64 .ptr .global .align 1 param_up,
	.param .u32 param_n_elements
)
.reqntid 128
{
	.reg .pred 	%p<2>;
	.reg .b16 	%rs<17>;
	.reg .b32 	%r<94>;
	.reg .b64 	%rd<8>;

	ld.param.b64 	%rd4, [param_out];
	ld.param.b64 	%rd5, [param_gate];
	mov.u32 	%r13, %ctaid.x;
	shl.b32 	%r14, %r13, 10;
	ld.param.b64 	%rd6, [param_up];
	ld.param.b32 	%r15, [param_n_elements];
	mov.u32 	%r16, %tid.x;
	shl.b32 	%r17, %r16, 3;
	and.b32 	%r18, %r17, 1016;
	or.b32 	%r19, %r18, %r14;
	setp.lt.s32 	%p1, %r19, %r15;
	mul.wide.s32 	%rd7, %r19, 2;
	add.s64 	%rd1, %rd5, %rd7;
	mov.u32 %r1, 0x0;
	mov.u32 %r2, 0x0;
	mov.u32 %r3, 0x0;
	mov.u32 %r4, 0x0;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	add.s64 	%rd2, %rd6, %rd7;
	mov.u32 %r5, 0x0;
	mov.u32 %r6, 0x0;
	mov.u32 %r7, 0x0;
	mov.u32 %r8, 0x0;
	@%p1 ld.global.v4.b32 { %r5, %r6, %r7, %r8 }, [ %rd2 + 0 ];
	add.s64 	%rd3, %rd4, %rd7;
	mov.b32 	{%rs1, %rs2}, %r1;
	cvt.f32.f16 	%r20, %rs2;
	cvt.f32.f16 	%r21, %rs1;
	mov.b32 	{%rs3, %rs4}, %r5;
	cvt.f32.f16 	%r22, %rs4;
	cvt.f32.f16 	%r23, %rs3;
	mov.b32 	%r24, 0f00000000;
	sub.f32 	%r25, %r24, %r21;
	sub.f32 	%r26, %r24, %r20;
	mul.f32 	%r27, %r25, 0f3FB8AA3B;
	ex2.approx.f32 	%r28, %r27;
	mul.f32 	%r29, %r26, 0f3FB8AA3B;
	ex2.approx.f32 	%r30, %r29;
	add.f32 	%r31, %r28, 0f3F800000;
	add.f32 	%r32, %r30, 0f3F800000;
	mov.b32 	%r33, 0f3F800000;
	div.full.f32 	%r34, %r33, %r31;
	div.full.f32 	%r35, %r33, %r32;
	mul.f32 	%r36, %r35, %r20;
	mul.f32 	%r37, %r34, %r21;
	mul.f32 	%r38, %r37, %r23;
	mul.f32 	%r39, %r36, %r22;
	cvt.rn.f16x2.f32 	%r9, %r39, %r38;
	mov.b32 	{%rs5, %rs6}, %r2;
	cvt.f32.f16 	%r40, %rs6;
	cvt.f32.f16 	%r41, %rs5;
	mov.b32 	{%rs7, %rs8}, %r6;
	cvt.f32.f16 	%r42, %rs8;
	cvt.f32.f16 	%r43, %rs7;
	sub.f32 	%r44, %r24, %r41;
	sub.f32 	%r45, %r24, %r40;
	mul.f32 	%r46, %r44, 0f3FB8AA3B;
	ex2.approx.f32 	%r47, %r46;
	mul.f32 	%r48, %r45, 0f3FB8AA3B;
	ex2.approx.f32 	%r49, %r48;
	add.f32 	%r50, %r47, 0f3F800000;
	add.f32 	%r51, %r49, 0f3F800000;
	div.full.f32 	%r52, %r33, %r50;
	div.full.f32 	%r53, %r33, %r51;
	mul.f32 	%r54, %r53, %r40;
	mul.f32 	%r55, %r52, %r41;
	mul.f32 	%r56, %r55, %r43;
	mul.f32 	%r57, %r54, %r42;
	cvt.rn.f16x2.f32 	%r10, %r57, %r56;
	mov.b32 	{%rs9, %rs10}, %r3;
	cvt.f32.f16 	%r58, %rs10;
	cvt.f32.f16 	%r59, %rs9;
	mov.b32 	{%rs11, %rs12}, %r7;
	cvt.f32.f16 	%r60, %rs12;
	cvt.f32.f16 	%r61, %rs11;
	sub.f32 	%r62, %r24, %r59;
	sub.f32 	%r63, %r24, %r58;
	mul.f32 	%r64, %r62, 0f3FB8AA3B;
	ex2.approx.f32 	%r65, %r64;
	mul.f32 	%r66, %r63, 0f3FB8AA3B;
	ex2.approx.f32 	%r67, %r66;
	add.f32 	%r68, %r65, 0f3F800000;
	add.f32 	%r69, %r67, 0f3F800000;
	div.full.f32 	%r70, %r33, %r68;
	div.full.f32 	%r71, %r33, %r69;
	mul.f32 	%r72, %r71, %r58;
	mul.f32 	%r73, %r70, %r59;
	mul.f32 	%r74, %r73, %r61;
	mul.f32 	%r75, %r72, %r60;
	cvt.rn.f16x2.f32 	%r11, %r75, %r74;
	mov.b32 	{%rs13, %rs14}, %r4;
	cvt.f32.f16 	%r76, %rs14;
	cvt.f32.f16 	%r77, %rs13;
	mov.b32 	{%rs15, %rs16}, %r8;
	cvt.f32.f16 	%r78, %rs16;
	cvt.f32.f16 	%r79, %rs15;
	sub.f32 	%r80, %r24, %r77;
	sub.f32 	%r81, %r24, %r76;
	mul.f32 	%r82, %r80, 0f3FB8AA3B;
	ex2.approx.f32 	%r83, %r82;
	mul.f32 	%r84, %r81, 0f3FB8AA3B;
	ex2.approx.f32 	%r85, %r84;
	add.f32 	%r86, %r83, 0f3F800000;
	add.f32 	%r87, %r85, 0f3F800000;
	div.full.f32 	%r88, %r33, %r86;
	div.full.f32 	%r89, %r33, %r87;
	mul.f32 	%r90, %r89, %r76;
	mul.f32 	%r91, %r88, %r77;
	mul.f32 	%r92, %r91, %r79;
	mul.f32 	%r93, %r90, %r78;
	cvt.rn.f16x2.f32 	%r12, %r93, %r92;
	@%p1 st.global.v4.b32 [ %rd3 + 0 ], { %r9, %r10, %r11, %r12 };
	ret;
}
"#.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_ptx() -> String {
        emit_silu_mul_kernel()
    }

    #[test]
    fn test_silu_mul_valid_ascii() {
        let ptx = get_ptx();
        for (i, b) in ptx.bytes().enumerate() {
            assert!(b < 128, "Non-ASCII at {}", i);
        }
    }

    #[test]
    fn test_silu_mul_has_entry() {
        assert!(get_ptx().contains(".entry silu_mul_kernel"));
    }

    #[test]
    fn test_silu_mul_has_reqntid() {
        assert!(get_ptx().contains(".reqntid 128"));
    }

    #[test]
    fn test_silu_mul_vectorized_loads() {
        let ptx = get_ptx();
        let v4_loads = ptx.lines().filter(|l| l.contains("ld.global.v4.b32")).count();
        assert_eq!(v4_loads, 2, "Need 2 v4 loads (gate + up), got {}", v4_loads);
    }

    #[test]
    fn test_silu_mul_vectorized_store() {
        let ptx = get_ptx();
        let v4_stores = ptx.lines().filter(|l| l.contains("st.global.v4.b32")).count();
        assert_eq!(v4_stores, 1, "Need 1 v4 store (output), got {}", v4_stores);
    }

    #[test]
    fn test_silu_mul_has_sigmoid() {
        let ptx = get_ptx();
        let ex2 = ptx.lines().filter(|l| l.contains("ex2.approx.f32")).count();
        assert_eq!(ex2, 8, "Need 8 ex2 for sigmoid, got {}", ex2);
    }

    #[test]
    fn test_silu_mul_has_log2e() {
        assert!(get_ptx().contains("0f3FB8AA3B"), "Need log2(e) constant");
    }

    #[test]
    fn test_silu_mul_has_div() {
        let ptx = get_ptx();
        let div = ptx.lines().filter(|l| l.contains("div.full.f32")).count();
        assert_eq!(div, 8, "Need 8 divs for sigmoid, got {}", div);
    }

    #[test]
    fn test_silu_mul_has_f16_conversion() {
        let ptx = get_ptx();
        let cvt_in = ptx.lines().filter(|l| l.contains("cvt.f32.f16")).count();
        let cvt_out = ptx.lines().filter(|l| l.contains("cvt.rn.f16x2.f32")).count();
        assert_eq!(cvt_in, 16, "Need 16 f16→f32, got {}", cvt_in);
        assert_eq!(cvt_out, 4, "Need 4 f16x2 packs, got {}", cvt_out);
    }

    #[test]
    fn test_silu_mul_register_count() {
        let ptx = get_ptx();
        assert!(ptx.contains("%r<94>"), "Should use exactly 94 b32 registers");
        assert!(ptx.contains("%rs<17>"), "Should use exactly 17 b16 registers");
    }

    #[test]
    fn test_silu_mul_matches_triton_structure() {
        let ptx = get_ptx();
        // Verify exact instruction counts matching Triton reference
        let sub = ptx.lines().filter(|l| l.trim().starts_with("sub.f32")).count();
        let mul = ptx.lines().filter(|l| l.trim().starts_with("mul.f32")).count();
        let add = ptx.lines().filter(|l| l.trim().starts_with("add.f32")).count();
        assert_eq!(sub, 8, "Need 8 sub.f32 (negation), got {}", sub);
        assert_eq!(mul, 24, "Need 24 mul.f32 (8 scale + 8 silu + 8 result), got {}", mul);
        assert_eq!(add, 8, "Need 8 add.f32 (1+exp), got {}", add);
    }

    #[test]
    fn test_silu_mul_ptx_size() {
        let ptx = get_ptx();
        let lines = ptx.lines().count();
        // Literal copy should be ~130 lines (matching reference minus debug info)
        assert!(lines >= 100 && lines <= 200, "Expected ~130 lines, got {}", lines);
    }
}
