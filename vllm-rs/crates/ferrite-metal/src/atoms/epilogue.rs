use super::EpilogueAtom;
use crate::config::MetalGemmConfig;
use crate::msl_builder::MslBuilder;

/// Identity epilogue — store accumulators directly.
pub struct IdentityEpilogue;

impl EpilogueAtom for IdentityEpilogue {
    fn is_identity(&self) -> bool {
        true
    }
}

/// SiLU epilogue — applies x * sigmoid(x) to each accumulator element.
///
/// Operates on C_sram via thread_elements() after the K-loop completes.
pub struct SiLuEpilogue;

impl EpilogueAtom for SiLuEpilogue {
    fn emit_epilogue(&self, msl: &mut MslBuilder, config: &MetalGemmConfig) {
        msl.set("TILES_M", (config.register_m() / 8).to_string());
        msl.set("TILES_N", (config.register_n() / 8).to_string());
        msl.block(
            r#"
// SiLU: x * sigmoid(x) = x / (1 + exp(-x))
for (ushort tm = 0; tm < {{TILES_M}}; tm++) {
    for (ushort tn = 0; tn < {{TILES_N}}; tn++) {
        thread auto &elems = C_sram[tm][tn].thread_elements();
        for (int i = 0; i < 64; i++) {
            auto x = elems[i];
            elems[i] = x / (1 + exp(-x));
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

    #[test]
    fn test_identity_is_identity() {
        assert!(IdentityEpilogue.is_identity());
    }

    #[test]
    fn test_silu_is_not_identity() {
        assert!(!SiLuEpilogue.is_identity());
    }
}
