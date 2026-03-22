use crate::msl_builder::MslBuilder;

/// Precision conversion kernel — casts between f32 and f16.
///
/// Simple element-wise kernel: each thread converts a strided range.
/// Used to bridge GEMM (f32 output) and attention (f16 input).
pub struct ConvertAtom;

impl ConvertAtom {
    /// Emit a kernel that converts `count` elements from `src_type` to `dst_type`.
    ///
    /// Kernel signature:
    ///   kernel void convert(
    ///       device SRC_TYPE *input [[buffer(0)]],
    ///       device DST_TYPE *output [[buffer(1)]],
    ///       constant uint *count_ptr [[buffer(2)]],
    ///       uint tid [[thread_position_in_grid]]
    ///   )
    pub fn emit_kernel(src_type: &str, dst_type: &str) -> String {
        let mut msl = MslBuilder::new();
        msl.set("SRC_TYPE", src_type);
        msl.set("DST_TYPE", dst_type);

        msl.raw("#include <metal_stdlib>");
        msl.raw("using namespace metal;");
        msl.blank();
        msl.block(
            r#"
kernel void convert(
    device {{SRC_TYPE}} *input [[buffer(0)]],
    device {{DST_TYPE}} *output [[buffer(1)]],
    constant uint *count_ptr [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
)
"#,
        );
        msl.open_brace();
        msl.block(
            r#"
uint count = *count_ptr;
if (tid < count) {
    output[tid] = {{DST_TYPE}}(input[tid]);
}
"#,
        );
        msl.close_brace();
        msl.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_to_f16_kernel_structure() {
        let msl = ConvertAtom::emit_kernel("float", "half");
        assert!(msl.contains("kernel void convert("));
        assert!(msl.contains("device float *input"));
        assert!(msl.contains("device half *output"));
        assert!(msl.contains("half(input[tid])"));
    }

    #[test]
    fn test_f16_to_f32_kernel_structure() {
        let msl = ConvertAtom::emit_kernel("half", "float");
        assert!(msl.contains("device half *input"));
        assert!(msl.contains("device float *output"));
        assert!(msl.contains("float(input[tid])"));
    }

    #[test]
    fn test_no_template_vars() {
        let msl = ConvertAtom::emit_kernel("float", "half");
        assert!(!msl.contains("{{"));
    }
}
