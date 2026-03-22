use super::MmaAtom;
use crate::config::MetalGemmConfig;
use crate::msl_builder::MslBuilder;

/// Native simdgroup multiply-accumulate.
pub struct NativeMma;

impl MmaAtom for NativeMma {
    fn emit_multiply(&self, msl: &mut MslBuilder, _config: &MetalGemmConfig) {
        msl.raw("simdgroup_multiply_accumulate(C_sram[tm][tn], A_mat, B_mat, C_sram[tm][tn]);");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emits_multiply_accumulate() {
        let config = MetalGemmConfig::default_apple9_f16();
        let mut msl = MslBuilder::new();
        NativeMma.emit_multiply(&mut msl, &config);
        let s = msl.finish();
        assert!(s.contains("simdgroup_multiply_accumulate"));
        assert!(s.contains("C_sram[tm][tn]"));
        assert!(s.contains("A_mat"));
        assert!(s.contains("B_mat"));
    }
}
