use super::EpilogueAtom;
use crate::config::MetalGemmConfig;
use crate::msl_builder::MslBuilder;

/// RoPE epilogue — applies Rotary Positional Embedding to accumulator elements.
///
/// For each pair of elements (x_2i, x_{2i+1}):
///   out_2i   = x_2i * cos(theta) - x_{2i+1} * sin(theta)
///   out_{2i+1} = x_2i * sin(theta) + x_{2i+1} * cos(theta)
///
/// Where theta = position * base^(-2i / d_head).
///
/// The position index and head dimension are expected as kernel parameters
/// (`position` and `d_head`) in scope when the emitted MSL executes.
/// `rope_base` defaults to 10000.0.
pub struct RoPEAtom;

impl EpilogueAtom for RoPEAtom {
    fn emit_epilogue(&self, msl: &mut MslBuilder, config: &MetalGemmConfig) {
        msl.set("TILES_M", (config.register_m() / 8).to_string());
        msl.set("TILES_N", (config.register_n() / 8).to_string());
        msl.block(
            r#"
// RoPE: Rotary Positional Embedding
for (ushort tm = 0; tm < {{TILES_M}}; tm++) {
    for (ushort tn = 0; tn < {{TILES_N}}; tn++) {
        thread auto &elems = C_sram[tm][tn].thread_elements();
        for (int i = 0; i < 64; i += 2) {
            int dim_idx = (tn * 64 + i) / 2;
            float theta = float(position) * pow(rope_base, -2.0f * float(dim_idx) / float(d_head));
            float cos_t = cos(theta);
            float sin_t = sin(theta);
            auto x0 = elems[i];
            auto x1 = elems[i + 1];
            elems[i]     = x0 * cos_t - x1 * sin_t;
            elems[i + 1] = x0 * sin_t + x1 * cos_t;
        }
    }
}
"#,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emit_rope_msl() -> String {
        let mut msl = MslBuilder::new();
        let config = MetalGemmConfig::default_apple9_f16();
        RoPEAtom.emit_epilogue(&mut msl, &config);
        msl.finish()
    }

    #[test]
    fn test_emitted_msl_contains_cos_sin() {
        let msl = emit_rope_msl();
        assert!(msl.contains("cos("), "MSL should contain cos() call");
        assert!(msl.contains("sin("), "MSL should contain sin() call");
    }

    #[test]
    fn test_emitted_msl_contains_thread_elements() {
        let msl = emit_rope_msl();
        assert!(
            msl.contains("thread_elements()"),
            "MSL should access thread_elements()"
        );
    }

    #[test]
    fn test_emitted_msl_contains_rotation_formula() {
        let msl = emit_rope_msl();
        // The rotation formula: x0 * cos_t - x1 * sin_t (real part)
        assert!(
            msl.contains("x0 * cos_t - x1 * sin_t"),
            "MSL should contain the rotation subtract pattern"
        );
        // The rotation formula: x0 * sin_t + x1 * cos_t (imaginary part)
        assert!(
            msl.contains("x0 * sin_t + x1 * cos_t"),
            "MSL should contain the rotation add pattern"
        );
    }

    #[test]
    fn test_rope_is_not_identity() {
        assert!(!RoPEAtom.is_identity());
    }

    #[test]
    fn test_emitted_msl_contains_rope_base() {
        let msl = emit_rope_msl();
        assert!(
            msl.contains("rope_base"),
            "MSL should reference rope_base parameter"
        );
    }
}
