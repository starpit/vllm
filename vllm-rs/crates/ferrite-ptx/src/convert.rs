// ===========================================================================
// f32 -> f16 conversion kernel
//
// Vectorized element-wise conversion: loads 4×f32, converts to 4×f16,
// stores as 2×b32 (packed f16 pairs). Each thread handles 4 elements.
//
// Grid: 1D, ceil(N / (block_size * 4)) blocks
// Block: block_size threads (typically 256)
// ===========================================================================

/// Build a standalone f32 -> f16 conversion kernel.
///
/// Kernel signature: `cvt_f32_to_f16(f32* input, f16* output, u32 N)`
///
/// Each thread converts 4 elements using vectorized loads/stores.
/// `N` is the total number of elements.
pub fn build_cvt_f32_to_f16_kernel(sm_arch: &str) -> String {
    format!(
        r#".version 8.7
.target {sm_arch}
.address_size 64

.visible .entry cvt_f32_to_f16(
	.param .u64 .ptr .global .align 16 param_in,
	.param .u64 .ptr .global .align 16 param_out,
	.param .u32 param_N
)
.reqntid 256
{{
	.reg .pred %p0;
	.reg .b32 %r<8>;
	.reg .b16 %h<4>;
	.reg .f32 %f<4>;
	.reg .b64 %rd<8>;

	ld.param.b64 	%rd0, [param_in];
	ld.param.b64 	%rd1, [param_out];
	ld.param.b32 	%r0, [param_N];

	// global_idx = blockIdx.x * 256 + threadIdx.x
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	shl.b32 	%r3, %r1, 8;
	add.s32 	%r3, %r3, %r2;
	// element_idx = global_idx * 4
	shl.b32 	%r3, %r3, 2;

	// bounds check: if element_idx >= N, skip
	setp.ge.u32 	%p0, %r3, %r0;
	@%p0 bra 	$L_DONE;

	// Load 4 x f32
	mul.wide.u32 	%rd2, %r3, 4;
	add.s64 	%rd3, %rd0, %rd2;
	ld.global.v4.f32 	{{%f0, %f1, %f2, %f3}}, [%rd3];

	// Convert f32 -> f16
	cvt.rn.f16.f32 	%h0, %f0;
	cvt.rn.f16.f32 	%h1, %f1;
	cvt.rn.f16.f32 	%h2, %f2;
	cvt.rn.f16.f32 	%h3, %f3;

	// Pack into b32 pairs
	mov.b32 	%r4, {{%h0, %h1}};
	mov.b32 	%r5, {{%h2, %h3}};

	// Store 2 x b32 (= 4 x f16)
	mul.wide.u32 	%rd4, %r3, 2;
	add.s64 	%rd5, %rd1, %rd4;
	st.global.v2.b32 	[%rd5], {{%r4, %r5}};

$L_DONE:
	ret;
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cvt_kernel_generates_valid_ptx() {
        let ptx = build_cvt_f32_to_f16_kernel("sm_89");
        assert!(
            ptx.contains(".visible .entry cvt_f32_to_f16("),
            "Must have entry point"
        );
        assert!(ptx.contains("cvt.rn.f16.f32"), "Must convert f32 to f16");
        // Check for non-ASCII
        for (i, b) in ptx.bytes().enumerate() {
            assert!(b < 128, "Non-ASCII byte 0x{:02x} at position {}", b, i);
        }
        println!("CVT PTX: {} bytes", ptx.len());
    }
}
