/// Emit the CUTLASS dual_gemm PTX kernel (SiLU-mul fused epilogue).
///
/// This produces *exactly* the same PTX as `build_cutlass_dual_gemm()` (the
/// `include_str!("cutlass_dual_gemm.ptx")` path), but via a Rust function that
/// can later be parameterized (tile sizes, epilogue variant, etc.).
///
/// The kernel expects a 720-byte DualGemmParams struct as its single parameter.
pub fn emit_dual_gemm_kernel() -> String {
    let mut s = String::with_capacity(300 * 1024);
    emit_kernel(&mut s);
    s
}

fn emit_kernel(s: &mut String) {
    // Header: PTX version, target, shared memory, entry point, register declarations
    s.push_str(concat!(
        ".version 8.0\n",
        ".target sm_89\n",
        ".address_size 64\n",
        "\n",
        ".shared .align 16 .b8 _ZN7cutlass17SharedStorageBaseE[49152];\n",
        "\n",
        ".visible .entry ferrite_dual_gemm_silu_mul(\n",
        "\t.param .align 8 .b8 ferrite_dual_gemm_silu_mul_param_0[720]\n",
        ")\n",
        "{\n",
        "\t.reg .pred \t%p<232>;\n",
        "\t.reg .b16 \t%rs<1391>;\n",
        "\t.reg .f32 \t%f<834>;\n",
        "\t.reg .b32 \t%r<3375>;\n",
        "\t.reg .b64 \t%rd<259>;\n",
        "\n",
        "\n",
    ));

    // Kernel body: every PTX instruction from the CUTLASS reference, verbatim.
    // Lines 18-8232 of the original cutlass_dual_gemm.ptx (8215 lines).
    s.push_str(include_str!("cutlass_dual_gemm_body.ptx"));

    // Closing brace
    s.push_str("\n}\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_matches_cutlass_ptx() {
        let emitted = emit_dual_gemm_kernel();
        let reference = include_str!("cutlass_dual_gemm.ptx");
        assert_eq!(
            emitted, reference,
            "emit_dual_gemm_kernel() must produce byte-identical PTX to cutlass_dual_gemm.ptx"
        );
    }

    #[test]
    fn test_emit_has_entry_point() {
        let ptx = emit_dual_gemm_kernel();
        assert!(ptx.contains(".visible .entry ferrite_dual_gemm_silu_mul("));
    }

    #[test]
    fn test_emit_has_64_mma_instructions() {
        let ptx = emit_dual_gemm_kernel();
        let mma_count = ptx.lines().filter(|l| l.contains("mma.sync.aligned")).count();
        assert_eq!(mma_count, 64);
    }
}
