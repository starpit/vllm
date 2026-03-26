//! Fuse an elementwise operation into a CUTLASS GEMM epilogue.
//!
//! Strategy: find each `cvt.rn.bf16x2.f32 %rN, %fA, %fB` in the epilogue,
//! inject the elementwise op on %fA and %fB (in-place) before the conversion.
//! The rest of the kernel is unchanged — same grid, same params, same stores.
//!
//! This is register-level fusion: the op runs on f32 accumulator values
//! that are already in registers, before bf16 conversion, before GMEM write.

/// Activation functions that can be injected into GEMM epilogues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFn {
    Silu,
    Gelu,
    Relu,
}

impl ActivationFn {
    /// Name used in PTX comments.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Silu => "SiLU",
            Self::Gelu => "GELU",
            Self::Relu => "ReLU",
        }
    }

    /// Number of .f32 scratch registers needed.
    pub fn f32_scratch_count(&self) -> usize {
        match self {
            Self::Silu => 4,
            Self::Gelu => 8,
            Self::Relu => 0,
        }
    }

    /// Number of .b32 scratch registers needed.
    pub fn b32_scratch_count(&self) -> usize {
        match self {
            Self::Silu => 2,
            Self::Gelu => 2,
            Self::Relu => 0,
        }
    }

    /// Parse from a string token (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "Silu" | "SiLU" | "SILU" | "silu" => Some(Self::Silu),
            "Gelu" | "GeLU" | "GELU" | "gelu" => Some(Self::Gelu),
            "Relu" | "ReLU" | "RELU" | "relu" => Some(Self::Relu),
            _ => None,
        }
    }
}

/// Inject an activation into a CUTLASS GEMM epilogue (bf16 path).
///
/// For each `cvt.rn.bf16x2.f32 %rN, %fA, %fB`, inserts the activation
/// on %fA and %fB in-place before the conversion.
pub fn inject_activation_into_epilogue(ptx: &str, act: ActivationFn) -> Result<String, String> {
    let lines: Vec<&str> = ptx.lines().collect();
    let mut result = Vec::new();
    let mut sites_found = false;

    for line in lines.iter() {
        let trimmed = line.trim();

        if trimmed.starts_with("cvt.rn.bf16x2.f32")
            && let Some((_dest, src_a, src_b)) = parse_bf16_cvt(trimmed)
        {
            sites_found = true;
            result.push(format!(
                "\t// FERRITE: inject {} before bf16 conversion",
                act.name()
            ));
            emit_activation_inplace(&mut result, &src_a, act);
            emit_activation_inplace(&mut result, &src_b, act);
            result.push(format!("\t{trimmed}"));
            continue;
        }

        result.push(line.to_string());
    }

    if !sites_found {
        return Err("no cvt.rn.bf16x2.f32 sites found -- is this a bf16 GEMM epilogue?".into());
    }

    insert_scratch_registers(result, act)
}

/// Inject an activation before each `st.global.f32` store in an f32 GEMM.
pub fn inject_activation_into_f32_stores(ptx: &str, act: ActivationFn) -> Result<String, String> {
    let lines: Vec<&str> = ptx.lines().collect();
    let mut result = Vec::new();
    let mut sites_found = false;

    for line in lines.iter() {
        let trimmed = line.trim();

        if (trimmed.starts_with("st.global.f32")
            || (trimmed.starts_with("@") && trimmed.contains("st.global.f32")))
            && let Some(val_reg) = extract_store_value_f32(trimmed)
        {
            sites_found = true;
            result.push(format!(
                "\t// FERRITE: inject {} before f32 store",
                act.name()
            ));
            emit_activation_inplace(&mut result, &val_reg, act);
            result.push(format!("\t{trimmed}"));
            continue;
        }

        result.push(line.to_string());
    }

    if !sites_found {
        return Err("no st.global.f32 sites found".into());
    }

    insert_scratch_registers(result, act)
}

/// Backward-compatible wrapper: inject SiLU into bf16 epilogue.
#[allow(dead_code)]
pub fn inject_silu_into_epilogue(ptx: &str) -> Result<String, String> {
    inject_activation_into_epilogue(ptx, ActivationFn::Silu)
}

/// Backward-compatible wrapper: inject SiLU into f32 stores.
#[allow(dead_code)]
pub fn inject_silu_into_f32_stores(ptx: &str) -> Result<String, String> {
    inject_activation_into_f32_stores(ptx, ActivationFn::Silu)
}

// ── scratch register insertion ──────────────────────────────────────

/// Insert scratch register declarations after the initial .reg block.
fn insert_scratch_registers(lines: Vec<String>, act: ActivationFn) -> Result<String, String> {
    let f_count = act.f32_scratch_count();
    let r_count = act.b32_scratch_count();

    if f_count == 0 && r_count == 0 {
        return Ok(lines.join("\n"));
    }

    let mut output = Vec::new();
    let mut in_initial_regs = false;
    let mut inserted_scratch = false;

    for line in &lines {
        let t = line.trim();

        if t == "{" || t.ends_with('{') {
            in_initial_regs = true;
        }

        if in_initial_regs
            && !inserted_scratch
            && !t.starts_with(".reg")
            && !t.starts_with(".shared")
            && !t.starts_with(".local")
            && !t.starts_with("//")
            && !t.is_empty()
            && !t.starts_with("{")
        {
            output.push(format!(
                "\t// FERRITE: scratch registers for {} injection",
                act.name()
            ));
            if f_count > 0 {
                output.push(format!("\t.reg .f32 \t%f_act<{f_count}>;"));
            }
            if r_count > 0 {
                output.push(format!("\t.reg .b32 \t%r_act<{r_count}>;"));
            }
            inserted_scratch = true;
        }

        output.push(line.clone());
    }

    Ok(output.join("\n"))
}

// ── activation emitters ─────────────────────────────────────────────

/// Dispatch to the right activation emitter.
fn emit_activation_inplace(out: &mut Vec<String>, reg: &str, act: ActivationFn) {
    match act {
        ActivationFn::Silu => emit_silu_inplace(out, reg),
        ActivationFn::Gelu => emit_gelu_inplace(out, reg),
        ActivationFn::Relu => emit_relu_inplace(out, reg),
    }
}

/// ReLU(x) = max(x, 0)
fn emit_relu_inplace(out: &mut Vec<String>, reg: &str) {
    out.push(format!("\tmax.f32 \t{reg}, {reg}, 0f00000000;"));
}

/// SiLU(x) = x / (1 + exp(-x))
///
/// Uses fast exp approximation via range reduction + ex2.approx.
/// Scratch: %f_act0..3, %r_act0.
fn emit_silu_inplace(out: &mut Vec<String>, reg: &str) {
    out.push(format!("\tneg.f32 \t%f_act0, {reg};"));
    out.push("\tfma.rn.f32 \t%f_act1, %f_act0, 0f3BBB989D, 0f3F000000;".to_string());
    out.push("\tcvt.sat.f32.f32 \t%f_act1, %f_act1;".to_string());
    out.push("\tfma.rm.f32 \t%f_act2, %f_act1, 0f437C0000, 0f4B400001;".to_string());
    out.push("\tadd.f32 \t%f_act3, %f_act2, 0fCB40007F;".to_string());
    out.push("\tneg.f32 \t%f_act3, %f_act3;".to_string());
    out.push("\tfma.rn.f32 \t%f_act3, %f_act0, 0f3FB8AA3B, %f_act3;".to_string());
    out.push("\tfma.rn.f32 \t%f_act3, %f_act0, 0f32A57060, %f_act3;".to_string());
    out.push("\tmov.b32 \t%r_act0, %f_act2;".to_string());
    out.push("\tshl.b32 \t%r_act0, %r_act0, 23;".to_string());
    out.push("\tmov.b32 \t%f_act2, %r_act0;".to_string());
    out.push("\tex2.approx.ftz.f32 \t%f_act3, %f_act3;".to_string());
    out.push("\tfma.rn.f32 \t%f_act3, %f_act3, %f_act2, 0f3F800000;".to_string());
    out.push(format!("\tdiv.rn.f32 \t{reg}, {reg}, %f_act3;"));
}

/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// tanh(z) is computed as 2*sigmoid(2z) - 1 using the same fast exp path.
/// Scratch: %f_act0..7, %r_act0.
///
/// Constants (IEEE 754 hex):
///   sqrt(2/pi)         = 0.7978845608  => 0f3F4C422A
///   0.044715           = 0.044715      => 0f3D372713
///   sqrt(2/pi)*0.044715 = 0.035677..   => 0f3D122279
fn emit_gelu_inplace(out: &mut Vec<String>, reg: &str) {
    // Step 1: inner = sqrt(2/pi) * x * (1 + 0.044715 * x^2)
    //       = (BETA*KAPPA) * x^2 + BETA, all multiplied by x
    out.push(format!("\tmul.f32 \t%f_act0, {reg}, {reg};"));
    // fma: BETA_KAPPA * x^2 + BETA
    out.push("\tfma.rn.f32 \t%f_act1, %f_act0, 0f3D122279, 0f3F4C422A;".to_string());
    out.push(format!("\tmul.f32 \t%f_act1, %f_act1, {reg};"));

    // Step 2: z = -2 * inner (for sigmoid(-2*inner) = 1/(1+exp(2*inner)))
    out.push("\tmul.f32 \t%f_act2, %f_act1, 0fC0000000;".to_string());

    // Step 3: fast exp(z) where z = -2*inner
    out.push("\tfma.rn.f32 \t%f_act3, %f_act2, 0f3BBB989D, 0f3F000000;".to_string());
    out.push("\tcvt.sat.f32.f32 \t%f_act3, %f_act3;".to_string());
    out.push("\tfma.rm.f32 \t%f_act4, %f_act3, 0f437C0000, 0f4B400001;".to_string());
    out.push("\tadd.f32 \t%f_act5, %f_act4, 0fCB40007F;".to_string());
    out.push("\tneg.f32 \t%f_act5, %f_act5;".to_string());
    out.push("\tfma.rn.f32 \t%f_act5, %f_act2, 0f3FB8AA3B, %f_act5;".to_string());
    out.push("\tfma.rn.f32 \t%f_act5, %f_act2, 0f32A57060, %f_act5;".to_string());
    out.push("\tmov.b32 \t%r_act0, %f_act4;".to_string());
    out.push("\tshl.b32 \t%r_act0, %r_act0, 23;".to_string());
    out.push("\tmov.b32 \t%f_act6, %r_act0;".to_string());
    out.push("\tex2.approx.ftz.f32 \t%f_act5, %f_act5;".to_string());
    // 1 + exp(-2*inner)
    out.push("\tfma.rn.f32 \t%f_act7, %f_act5, %f_act6, 0f3F800000;".to_string());

    // Step 4: tanh = 2/(1+exp(-2z)) - 1, then 0.5*x*(1+tanh)
    out.push("\tdiv.rn.f32 \t%f_act7, 0f40000000, %f_act7;".to_string());
    // tanh = 2*sigmoid - 1
    out.push("\tadd.f32 \t%f_act7, %f_act7, 0fBF800000;".to_string());
    // 1 + tanh
    out.push("\tadd.f32 \t%f_act7, %f_act7, 0f3F800000;".to_string());
    // 0.5 * (1 + tanh)
    out.push("\tmul.f32 \t%f_act7, %f_act7, 0f3F000000;".to_string());
    // x * 0.5 * (1 + tanh)
    out.push(format!("\tmul.f32 \t{reg}, {reg}, %f_act7;"));
}

// ── PTX parsing helpers ─────────────────────────────────────────────

/// Extract the value register from `st.global.f32 [addr], %fN;`
fn extract_store_value_f32(instr: &str) -> Option<String> {
    let parts: Vec<&str> = instr
        .split([',', ' ', '\t'])
        .filter(|s| !s.is_empty())
        .collect();
    if let Some(last) = parts.last() {
        let val = last.trim_end_matches(';');
        if val.starts_with("%f") {
            return Some(val.to_string());
        }
    }
    None
}

/// Parse `cvt.rn.bf16x2.f32 %rN, %fA, %fB;`
fn parse_bf16_cvt(instr: &str) -> Option<(String, String, String)> {
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

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_BF16_PTX: &str = ".version 8.8\n.target sm_89\n.address_size 64\n\n\
                    .visible .entry test()\n{\n\
                    \t.reg .f32 %f<4>;\n\
                    \t.reg .b32 %r<4>;\n\n\
                    \tmul.f32 %f1, %f0, %f2;\n\
                    \tcvt.rn.bf16x2.f32 %r1, %f0, %f1;\n\
                    \tst.global.v2.u32 [%r3], {%r1, %r2};\n\
                    \tret;\n}\n";

    const TEST_F32_PTX: &str = ".version 8.8\n.target sm_89\n.address_size 64\n\n\
                    .visible .entry test()\n{\n\
                    \t.reg .f32 %f<4>;\n\
                    \t.reg .b32 %r<4>;\n\
                    \t.reg .b64 %rd<4>;\n\n\
                    \tmul.f32 %f1, %f0, %f2;\n\
                    \tst.global.f32 [%rd1], %f1;\n\
                    \tret;\n}\n";

    #[test]
    fn test_parse_bf16_cvt() {
        let (d, a, b) = parse_bf16_cvt("cvt.rn.bf16x2.f32 %r1518, %f1625, %f1626;").unwrap();
        assert_eq!(d, "%r1518");
        assert_eq!(a, "%f1625");
        assert_eq!(b, "%f1626");
    }

    #[test]
    fn test_inject_silu_bf16() {
        let result = inject_silu_into_epilogue(TEST_BF16_PTX).unwrap();
        assert!(result.contains("%f_act"), "should have scratch f32 regs");
        assert!(result.contains("%r_act"), "should have scratch b32 regs");
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

    #[test]
    fn test_inject_gelu_bf16() {
        let result = inject_activation_into_epilogue(TEST_BF16_PTX, ActivationFn::Gelu).unwrap();
        assert!(result.contains("%f_act<8>"), "GELU needs 8 f32 scratch");
        assert!(result.contains("FERRITE: inject GELU"));
        assert!(
            result.contains("0f3F4C422A"),
            "should have sqrt(2/pi) constant"
        );
        assert!(
            result.contains("cvt.rn.bf16x2.f32"),
            "should keep original conversion"
        );
    }

    #[test]
    fn test_inject_relu_bf16() {
        let result = inject_activation_into_epilogue(TEST_BF16_PTX, ActivationFn::Relu).unwrap();
        assert!(
            !result.contains(".reg .f32 \t%f_act"),
            "ReLU needs no f32 scratch"
        );
        assert!(result.contains("max.f32"), "should have ReLU max");
        assert!(result.contains("FERRITE: inject ReLU"));
    }

    #[test]
    fn test_inject_silu_f32_stores() {
        let result = inject_silu_into_f32_stores(TEST_F32_PTX).unwrap();
        assert!(result.contains("FERRITE: inject SiLU before f32 store"));
        assert!(result.contains("neg.f32"));
        assert!(result.contains("st.global.f32"));
    }

    #[test]
    fn test_inject_gelu_f32_stores() {
        let result = inject_activation_into_f32_stores(TEST_F32_PTX, ActivationFn::Gelu).unwrap();
        assert!(result.contains("FERRITE: inject GELU before f32 store"));
        assert!(result.contains("0f3F4C422A"));
    }

    #[test]
    fn test_activation_fn_from_str() {
        assert_eq!(ActivationFn::from_str("Silu"), Some(ActivationFn::Silu));
        assert_eq!(ActivationFn::from_str("GELU"), Some(ActivationFn::Gelu));
        assert_eq!(ActivationFn::from_str("relu"), Some(ActivationFn::Relu));
        assert_eq!(ActivationFn::from_str("unknown"), None);
    }
}
