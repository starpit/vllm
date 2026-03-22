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
        // A_block is [block_m, block_k]. This simdgroup's rows start at sid_m * REGISTER_M.
        msl.block(
            r#"
simdgroup_load(A_mat, A_block, {{A_LEAD}}, ulong2(kt * 8, sid_m * {{REGISTER_M}} + tm * 8));
"#,
        );
    }

    fn emit_load_b(&self, msl: &mut MslBuilder, config: &MetalGemmConfig) {
        msl.set("B_LEAD", config.leading_block_dim('B').to_string());
        // B_block is [block_k, block_n]. This simdgroup's cols start at sid_n * REGISTER_N.
        msl.block(
            r#"
simdgroup_load(B_mat, B_block, {{B_LEAD}}, ulong2(sid_n * {{REGISTER_N}} + tn * 8, kt * 8));
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
        msl.set("REGISTER_M", config.register_m().to_string());
        msl.set("A_LEAD", config.leading_block_dim('A').to_string());
        NativeFragmentLoad.emit_load_a(&mut msl, &config);
        let s = msl.finish();
        assert!(s.contains("simdgroup_load(A_mat"));
        assert!(s.contains("A_block"));
        assert!(s.contains("sid_m"), "Must offset by simdgroup M position");
        assert!(s.contains("kt * 8"));
    }

    #[test]
    fn test_load_b() {
        let config = MetalGemmConfig::default_apple9_f16();
        let mut msl = MslBuilder::new();
        msl.set("REGISTER_N", config.register_n().to_string());
        msl.set("B_LEAD", config.leading_block_dim('B').to_string());
        NativeFragmentLoad.emit_load_b(&mut msl, &config);
        let s = msl.finish();
        assert!(s.contains("simdgroup_load(B_mat"));
        assert!(s.contains("B_block"));
        assert!(s.contains("sid_n"), "Must offset by simdgroup N position");
    }
}
