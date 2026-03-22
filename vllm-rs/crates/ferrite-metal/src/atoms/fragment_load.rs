use super::FragmentLoadAtom;
use crate::config::MetalGemmConfig;
use crate::msl_builder::MslBuilder;

/// Native simdgroup_load from threadgroup memory.
///
/// Uses Metal 4's hardware intrinsic — no custom headers needed.
pub struct NativeFragmentLoad;

impl FragmentLoadAtom for NativeFragmentLoad {
    fn emit_load_a(&self, msl: &mut MslBuilder, config: &MetalGemmConfig) {
        msl.set("A_LEAD", config.leading_block_dim('A').to_string());
        msl.block(
            r#"
simdgroup_load(A_mat, A_block, {{A_LEAD}}, ulong2(kt * 8, tm * 8));
"#,
        );
    }

    fn emit_load_b(&self, msl: &mut MslBuilder, config: &MetalGemmConfig) {
        msl.set("B_LEAD", config.leading_block_dim('B').to_string());
        msl.block(
            r#"
simdgroup_load(B_mat, B_block, {{B_LEAD}}, ulong2(tn * 8, kt * 8));
"#,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_a() {
        let config = MetalGemmConfig::default_apple9_f16();
        let mut msl = MslBuilder::new();
        NativeFragmentLoad.emit_load_a(&mut msl, &config);
        let s = msl.finish();
        assert!(s.contains("simdgroup_load(A_mat"));
        assert!(s.contains("A_block"));
        assert!(s.contains("tm * 8"));
        assert!(s.contains("kt * 8"));
    }

    #[test]
    fn test_load_b() {
        let config = MetalGemmConfig::default_apple9_f16();
        let mut msl = MslBuilder::new();
        NativeFragmentLoad.emit_load_b(&mut msl, &config);
        let s = msl.finish();
        assert!(s.contains("simdgroup_load(B_mat"));
        assert!(s.contains("B_block"));
        assert!(s.contains("tn * 8"));
    }
}
