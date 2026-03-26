//! Fuse an elementwise operation into a CUTLASS GEMM epilogue.
//!
//! Strategy: find each `cvt.rn.bf16x2.f32 %rN, %fA, %fB` in the epilogue,
//! inject the elementwise op on %fA and %fB (in-place) before the conversion.
//! The rest of the kernel is unchanged — same grid, same params, same stores.
//!
//! This is register-level fusion: the op runs on f32 accumulator values
//! that are already in registers, before bf16 conversion, before GMEM write.

/// Inject SiLU activation into a CUTLASS GEMM epilogue.
///
/// For each `cvt.rn.bf16x2.f32 %rN, %fA, %fB`, inserts SiLU(%fA) and SiLU(%fB)
/// in-place before the conversion. Uses dedicated scratch registers.
pub fn inject_silu_into_epilogue(ptx: &str) -> Result<String, String> {
    let lines: Vec<&str> = ptx.lines().collect();
    let mut result = Vec::new();
    let mut scratch_needed = false;

    for line in lines.iter() {
        let trimmed = line.trim();

        // Match: cvt.rn.bf16x2.f32 %rN, %fA, %fB;
        if trimmed.starts_with("cvt.rn.bf16x2.f32")
            && let Some((_dest, src_a, src_b)) = parse_bf16_cvt(trimmed) {
                scratch_needed = true;
                // Inject SiLU on both source f32 registers
                result.push("\t// FERRITE: inject SiLU before bf16 conversion".to_string());
                emit_silu_inplace(&mut result, &src_a);
                emit_silu_inplace(&mut result, &src_b);
                // Then the original conversion
                result.push(format!("\t{trimmed}"));
                continue;
            }

        result.push(line.to_string());
    }

    if !scratch_needed {
        return Err("no cvt.rn.bf16x2.f32 sites found — is this a bf16 GEMM epilogue?".into());
    }

    // Insert scratch register declarations after the initial .reg block
    // (not after inline asm .reg lines deeper in the body)
    let mut output = Vec::new();
    let mut in_initial_regs = false;
    let mut inserted_scratch = false;

    for line in &result {
        let t = line.trim();

        // Detect the initial .reg block: starts with first .reg after '{'
        if t == "{" || t.ends_with('{') {
            in_initial_regs = true;
        }

        // When we're past the initial .reg block (first non-.reg, non-empty,
        // non-.shared, non-comment line after seeing .reg), insert scratch
        if in_initial_regs
            && !inserted_scratch
            && !t.starts_with(".reg")
            && !t.starts_with(".shared")
            && !t.starts_with(".local")
            && !t.starts_with("//")
            && !t.is_empty()
            && !t.starts_with("{")
        {
            output.push("\t// FERRITE: scratch registers for SiLU injection".to_string());
            output.push("\t.reg .f32 \t%f_silu<4>;".to_string());
            output.push("\t.reg .b32 \t%r_silu<2>;".to_string());
            inserted_scratch = true;
        }

        output.push(line.clone());
    }

    Ok(output.join("\n"))
}

/// Parse `cvt.rn.bf16x2.f32 %rN, %fA, %fB;`
/// Returns (dest_reg, src_a, src_b)
fn parse_bf16_cvt(instr: &str) -> Option<(String, String, String)> {
    // cvt.rn.bf16x2.f32 %r1518, %f1625, %f1626;
    let parts: Vec<&str> = instr
        .split([',', ' ', '\t'])
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() >= 4 {
        let dest = parts[1].trim_end_matches(',').to_string();
        let src_a = parts[2].trim_end_matches(',').to_string();
        let src_b = parts[3].trim_end_matches(';').to_string();
        Some((dest, src_a, src_b))
    } else {
        None
    }
}

/// Emit SiLU in-place on a single f32 register.
/// SiLU(x) = x / (1 + exp(-x))
///
/// Uses the same fast exp approximation nvcc generates:
///   sigmoid(x) = 1 / (1 + exp(-x))
///   exp(-x) = 2^(-x/ln2) via range reduction + polynomial
fn emit_silu_inplace(out: &mut Vec<String>, reg: &str) {
    // We use %f_silu0..3 and %r_silu0..1 as scratch.
    // The computation is:
    //   neg.f32       %f_silu0, %reg          // -x
    //   mul.f32       %f_silu1, %f_silu0, C1  // -x * (1/ln2) approx part
    //   add.f32       %f_silu1, %f_silu1, 0.5
    //   cvt.sat.f32   %f_silu1, %f_silu1      // clamp to [0,1]
    //   fma.rm.f32    %f_silu2, %f_silu1, C2, C3  // range reduction
    //   add.f32       %f_silu3, %f_silu2, C4
    //   neg.f32       %f_silu3, %f_silu3
    //   fma.rn.f32    %f_silu3, %f_silu0, C5, %f_silu3
    //   fma.rn.f32    %f_silu3, %f_silu0, C6, %f_silu3
    //   mov.b32       %r_silu0, %f_silu2
    //   shl.b32       %r_silu0, %r_silu0, 23
    //   mov.b32       %f_silu2, %r_silu0
    //   ex2.approx.ftz.f32 %f_silu3, %f_silu3
    //   fma.rn.f32    %f_silu3, %f_silu3, %f_silu2, 0f3F800000  // 1 + exp(-x)
    //   div.rn.f32    %reg, %reg, %f_silu3                       // x / (1+exp(-x))

    out.push(format!("\tneg.f32 \t%f_silu0, {reg};"));
    out.push("\tfma.rn.f32 \t%f_silu1, %f_silu0, 0f3BBB989D, 0f3F000000;".to_string());
    out.push("\tcvt.sat.f32.f32 \t%f_silu1, %f_silu1;".to_string());
    out.push("\tfma.rm.f32 \t%f_silu2, %f_silu1, 0f437C0000, 0f4B400001;".to_string());
    out.push("\tadd.f32 \t%f_silu3, %f_silu2, 0fCB40007F;".to_string());
    out.push("\tneg.f32 \t%f_silu3, %f_silu3;".to_string());
    out.push("\tfma.rn.f32 \t%f_silu3, %f_silu0, 0f3FB8AA3B, %f_silu3;".to_string());
    out.push("\tfma.rn.f32 \t%f_silu3, %f_silu0, 0f32A57060, %f_silu3;".to_string());
    out.push("\tmov.b32 \t%r_silu0, %f_silu2;".to_string());
    out.push("\tshl.b32 \t%r_silu0, %r_silu0, 23;".to_string());
    out.push("\tmov.b32 \t%f_silu2, %r_silu0;".to_string());
    out.push("\tex2.approx.ftz.f32 \t%f_silu3, %f_silu3;".to_string());
    out.push("\tfma.rn.f32 \t%f_silu3, %f_silu3, %f_silu2, 0f3F800000;".to_string());
    out.push(format!("\tdiv.rn.f32 \t{reg}, {reg}, %f_silu3;"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bf16_cvt() {
        let (d, a, b) = parse_bf16_cvt("cvt.rn.bf16x2.f32 %r1518, %f1625, %f1626;").unwrap();
        assert_eq!(d, "%r1518");
        assert_eq!(a, "%f1625");
        assert_eq!(b, "%f1626");
    }

    #[test]
    fn test_inject_silu_adds_scratch_regs() {
        let ptx = ".version 8.8\n.target sm_89\n.address_size 64\n\n\
                    .visible .entry test()\n{\n\
                    \t.reg .f32 %f<4>;\n\
                    \t.reg .b32 %r<4>;\n\n\
                    \tmul.f32 %f1, %f0, %f2;\n\
                    \tcvt.rn.bf16x2.f32 %r1, %f0, %f1;\n\
                    \tst.global.v2.u32 [%r3], {%r1, %r2};\n\
                    \tret;\n}\n";

        let result = inject_silu_into_epilogue(ptx).unwrap();
        assert!(result.contains("%f_silu"), "should have scratch f32 regs");
        assert!(result.contains("%r_silu"), "should have scratch b32 regs");
        assert!(
            result.contains("FERRITE: inject SiLU"),
            "should have marker"
        );
        assert!(result.contains("neg.f32"), "should have SiLU computation");
        assert!(
            result.contains("cvt.rn.bf16x2.f32"),
            "should keep original conversion"
        );
    }
}
