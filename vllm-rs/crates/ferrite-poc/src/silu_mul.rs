/// SiLU×Up (silu_mul) PTX kernel emitter.
///
/// Computes: out[i] = silu(gate[i]) * up[i]  where silu(x) = x / (1 + exp(-x))
///
/// Uses f16 tensors with vectorized loads/stores (ld.global.v4.b32 = 8 f16 per thread).
/// 128 threads per block, BLOCK_SIZE=1024 elements per block (8 elements/thread).
///
/// Matches the Triton-compiled reference at /tmp/triton_silu_mul.ptx.

const BLOCK_SIZE: u32 = 1024;
const THREADS_PER_BLOCK: u32 = 128;
const ELEMS_PER_THREAD: u32 = BLOCK_SIZE / THREADS_PER_BLOCK; // 8

/// Emit the silu_mul PTX kernel as a string.
pub fn emit_silu_mul_kernel() -> String {
    assert_eq!(ELEMS_PER_THREAD, 8, "8 f16 elements per thread = 4 x b32");

    // We process 8 f16 values per thread, loaded as 4 x b32 via ld.global.v4.b32.
    // Each b32 holds 2 f16 values (f16x2 packed).
    // We need:
    //   - 4 b32 regs for gate (g0..g3)
    //   - 4 b32 regs for up (u0..u3)
    //   - 4 b32 regs for output (reuse gate regs after processing)
    //   - address regs, temporaries for silu computation
    //
    // Register budget per pair:
    //   2 x f32 for unpacked gate halves
    //   2 x f32 for unpacked up halves
    //   temps for sigmoid computation
    //   1 x b32 for packed f16x2 output

    let mut ptx = String::with_capacity(8192);

    // Header
    ptx.push_str("\
.version 8.5
.target sm_89
.address_size 64

.visible .entry silu_mul_kernel(
    .param .u64 .ptr .global .align 16 param_out,
    .param .u64 .ptr .global .align 16 param_gate,
    .param .u64 .ptr .global .align 16 param_up,
    .param .u32 param_n_elements
)
.reqntid 128
{
    // Register declarations
    .reg .pred  %p<2>;
    .reg .b16   %rs<17>;
    .reg .b32   %r<110>;
    .reg .b64   %rd<8>;

    // Load parameters
    ld.param.b64    %rd4, [param_out];
    ld.param.b64    %rd5, [param_gate];
    ld.param.b64    %rd6, [param_up];
    ld.param.b32    %r15, [param_n_elements];

    // Compute global element offset: block_offset + thread_offset
    // block_offset = ctaid.x * 1024 (= ctaid.x << 10)
    mov.u32         %r13, %ctaid.x;
    shl.b32         %r14, %r13, 10;

    // thread_offset = tid.x * 8 (= tid.x << 3), masked to 1016
    mov.u32         %r16, %tid.x;
    shl.b32         %r17, %r16, 3;
    and.b32         %r18, %r17, 1016;

    // global offset = block_offset | thread_offset
    or.b32          %r19, %r18, %r14;

    // Bounds check: offset < n_elements
    setp.lt.s32     %p1, %r19, %r15;

    // Byte offset = offset * 2 (f16 = 2 bytes)
    mul.wide.s32    %rd7, %r19, 2;

    // Gate pointer
    add.s64         %rd1, %rd5, %rd7;

    // Vectorized load gate: 4 x b32 = 8 x f16
    mov.u32 %r1, 0x0;
    mov.u32 %r2, 0x0;
    mov.u32 %r3, 0x0;
    mov.u32 %r4, 0x0;
    @%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];

    // Up pointer
    add.s64         %rd2, %rd6, %rd7;

    // Vectorized load up: 4 x b32 = 8 x f16
    mov.u32 %r5, 0x0;
    mov.u32 %r6, 0x0;
    mov.u32 %r7, 0x0;
    mov.u32 %r8, 0x0;
    @%p1 ld.global.v4.b32 { %r5, %r6, %r7, %r8 }, [ %rd2 + 0 ];

    // Output pointer
    add.s64         %rd3, %rd4, %rd7;

    // Constant: 0.0f for negation
    mov.b32         %r24, 0f00000000;
    // Constant: 1.0f
    mov.b32         %r33, 0f3F800000;

");

    // Process 4 pairs of f16 values (r1..r4 = gate, r5..r8 = up)
    // Each b32 register holds 2 packed f16 values.
    // We unpack, compute silu*up in f32, pack back to f16x2.
    //
    // Register allocation for silu computation:
    //   Pair i uses gate reg r{i+1}, up reg r{i+5}, output into r{i+9}
    //   Temporaries allocated from %r20 onwards (matching reference PTX)

    let pairs = [
        // (gate_b32, up_b32, out_b32, rs_lo, rs_hi, rs_up_lo, rs_up_hi, reg_base)
        // reg_base must not conflict with constants: %r24=0.0, %r33=1.0
        (1, 5, 9,   1, 2, 3, 4,     35),  // pair 0: regs %r35-%r52, output %r9
        (2, 6, 10,  5, 6, 7, 8,     53),  // pair 1: regs %r53-%r70, output %r10
        (3, 7, 11,  9, 10, 11, 12,  71),  // pair 2: regs %r71-%r88, output %r11
        (4, 8, 12,  13, 14, 15, 16, 89),  // pair 3: regs %r89-%r106, output %r12
    ];

    for &(gate, up, out, rs_lo, rs_hi, rs_up_lo, rs_up_hi, base) in &pairs {
        ptx.push_str(&format!("\
    // --- Process pair from %r{gate} (gate) and %r{up} (up) ---
    // Unpack gate f16x2 -> two f32
    mov.b32         {{%rs{rs_lo}, %rs{rs_hi}}}, %r{gate};
    cvt.f32.f16     %r{hi_f32}, %rs{rs_hi};
    cvt.f32.f16     %r{lo_f32}, %rs{rs_lo};

    // Unpack up f16x2 -> two f32
    mov.b32         {{%rs{rs_up_lo}, %rs{rs_up_hi}}}, %r{up};
    cvt.f32.f16     %r{up_hi_f32}, %rs{rs_up_hi};
    cvt.f32.f16     %r{up_lo_f32}, %rs{rs_up_lo};

    // SiLU for low element: sigmoid(gate_lo) * gate_lo
    sub.f32         %r{neg_lo}, %r24, %r{lo_f32};
    mul.f32         %r{scaled_lo}, %r{neg_lo}, 0f3FB8AA3B;
    ex2.approx.f32  %r{exp_lo}, %r{scaled_lo};
    add.f32         %r{denom_lo}, %r{exp_lo}, 0f3F800000;
    div.full.f32    %r{sig_lo}, %r33, %r{denom_lo};
    mul.f32         %r{silu_lo}, %r{sig_lo}, %r{lo_f32};
    mul.f32         %r{result_lo}, %r{silu_lo}, %r{up_lo_f32};

    // SiLU for high element: sigmoid(gate_hi) * gate_hi
    sub.f32         %r{neg_hi}, %r24, %r{hi_f32};
    mul.f32         %r{scaled_hi}, %r{neg_hi}, 0f3FB8AA3B;
    ex2.approx.f32  %r{exp_hi}, %r{scaled_hi};
    add.f32         %r{denom_hi}, %r{exp_hi}, 0f3F800000;
    div.full.f32    %r{sig_hi}, %r33, %r{denom_hi};
    mul.f32         %r{silu_hi}, %r{sig_hi}, %r{hi_f32};
    mul.f32         %r{result_hi}, %r{silu_hi}, %r{up_hi_f32};

    // Pack two f32 results back to f16x2
    cvt.rn.f16x2.f32 %r{out}, %r{result_hi}, %r{result_lo};

",
            gate = gate,
            up = up,
            out = out,
            rs_lo = rs_lo,
            rs_hi = rs_hi,
            rs_up_lo = rs_up_lo,
            rs_up_hi = rs_up_hi,
            // f32 unpacked values
            hi_f32 = base,
            lo_f32 = base + 1,
            up_hi_f32 = base + 2,
            up_lo_f32 = base + 3,
            // SiLU computation for low element
            neg_lo = base + 4,
            scaled_lo = base + 6,
            exp_lo = base + 7,
            denom_lo = base + 8,
            sig_lo = base + 9,
            silu_lo = base + 10,
            result_lo = base + 11,
            // SiLU computation for high element
            neg_hi = base + 5,
            scaled_hi = base + 12,
            exp_hi = base + 13,
            denom_hi = base + 14,
            sig_hi = base + 15,
            silu_hi = base + 16,
            result_hi = base + 17,
        ));
    }

    // Vectorized store and return
    ptx.push_str("\
    // Vectorized store: 4 x b32 = 8 x f16
    @%p1 st.global.v4.b32 [ %rd3 + 0 ], { %r9, %r10, %r11, %r12 };

    ret;
}
");

    ptx
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
        let ptx = get_ptx();
        assert!(ptx.contains(".entry silu_mul_kernel"));
    }

    #[test]
    fn test_silu_mul_has_reqntid() {
        let ptx = get_ptx();
        assert!(ptx.contains(".reqntid 128"));
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
        // sigmoid(x) = 1 / (1 + exp(-x))
        // exp(-x) = ex2.approx.f32(-x * log2(e))
        let ex2_count = ptx.lines().filter(|l| l.contains("ex2.approx.f32")).count();
        assert_eq!(ex2_count, 8, "Need 8 ex2 (one per f16 element), got {}", ex2_count);
    }

    #[test]
    fn test_silu_mul_has_log2e_constant() {
        let ptx = get_ptx();
        // log2(e) ≈ 1.4427 = 0x3FB8AA3B in f32
        assert!(ptx.contains("0f3FB8AA3B"), "Need log2(e) constant for sigmoid");
    }

    #[test]
    fn test_silu_mul_has_div() {
        let ptx = get_ptx();
        let div_count = ptx.lines().filter(|l| l.contains("div.full.f32")).count();
        assert_eq!(div_count, 8, "Need 8 divs (1/(1+exp(-x)) per element), got {}", div_count);
    }

    #[test]
    fn test_silu_mul_has_f16_conversion() {
        let ptx = get_ptx();
        let cvt_in = ptx.lines().filter(|l| l.contains("cvt.f32.f16")).count();
        let cvt_out = ptx.lines().filter(|l| l.contains("cvt.rn.f16x2.f32")).count();
        assert_eq!(cvt_in, 16, "Need 16 f16→f32 (8 gate + 8 up), got {}", cvt_in);
        assert_eq!(cvt_out, 4, "Need 4 f16x2 packs (8 f16 output), got {}", cvt_out);
    }

    #[test]
    fn test_silu_mul_has_bounds_check() {
        let ptx = get_ptx();
        assert!(ptx.contains("setp.lt.s32"), "Need bounds checking predicate");
    }

    #[test]
    fn test_silu_mul_no_shared_memory() {
        let ptx = get_ptx();
        let smem = ptx.lines().filter(|l| l.contains("shared")).count();
        assert_eq!(smem, 0, "SiLU×up should not use shared memory, found {}", smem);
    }

    #[test]
    fn test_silu_mul_no_constant_register_conflicts() {
        let ptx = get_ptx();
        // %r24 = 0.0 constant, %r33 = 1.0 constant
        // No instruction should WRITE to these registers after they're set
        // (except the initial mov.b32)
        let lines: Vec<&str> = ptx.lines().collect();
        let mut past_constants = false;
        for line in &lines {
            if line.contains("mov.b32") && line.contains("0f3F800000") {
                past_constants = true;
                continue;
            }
            if past_constants {
                // After constants are set, nothing should write to %r24 or %r33 as destination
                let trimmed = line.trim();
                if trimmed.starts_with("//") || trimmed.is_empty() { continue; }
                // Check if the instruction writes to %r24 or %r33
                // Pattern: "instruction %r24," at the start of operands
                if let Some(tab_pos) = trimmed.find('\t') {
                    let after_op = &trimmed[tab_pos+1..];
                    if after_op.starts_with("%r24,") || after_op.starts_with("%r33,") {
                        panic!("Register conflict: constant register overwritten: {}", trimmed);
                    }
                }
            }
        }
    }

    #[test]
    fn test_silu_mul_ptx_size() {
        let ptx = get_ptx();
        let lines = ptx.lines().count();
        assert!(lines >= 50, "Too small: {} lines", lines);
        assert!(lines <= 300, "Too large: {} lines", lines);
    }
}
