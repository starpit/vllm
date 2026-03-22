use super::EpilogueAtom;
use crate::config::MetalGemmConfig;
use crate::msl_builder::MslBuilder;

/// Residual-add epilogue — loads a residual tensor from device memory and adds
/// it element-wise to the C_sram accumulators after the K-loop.
///
/// The residual buffer pointer is expected as an additional kernel parameter
/// (`device half *residual`). Each simdgroup tile loads its corresponding 8x8
/// block via simdgroup_load, then adds through thread_elements().
pub struct ResidualAddEpilogue;

impl EpilogueAtom for ResidualAddEpilogue {
    fn emit_epilogue(&self, msl: &mut MslBuilder, config: &MetalGemmConfig) {
        msl.set("TILES_M", (config.register_m() / 8).to_string());
        msl.set("TILES_N", (config.register_n() / 8).to_string());
        msl.set("C_TYPE", config.register_precisions.c.msl_name());
        msl.block(
            r#"
// Residual add: C_sram += residual
for (ushort tm = 0; tm < {{TILES_M}}; tm++) {
    for (ushort tn = 0; tn < {{TILES_N}}; tn++) {
        simdgroup_matrix<{{C_TYPE}}, 8, 8> R_tile;
        simdgroup_load(R_tile, residual + (sid_m + tm * 8) * N + (sid_n + tn * 8), N);
        thread auto &c_elems = C_sram[tm][tn].thread_elements();
        thread auto &r_elems = R_tile.thread_elements();
        for (int i = 0; i < 64; i++) {
            c_elems[i] = c_elems[i] + r_elems[i];
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
    fn test_residual_add_not_identity() {
        assert!(!ResidualAddEpilogue.is_identity());
    }

    #[test]
    fn test_residual_add_emits_simdgroup_load() {
        let mut msl = MslBuilder::new();
        let config = MetalGemmConfig::default_apple9_f16();
        ResidualAddEpilogue.emit_epilogue(&mut msl, &config);
        let s = msl.finish();
        assert!(
            s.contains("simdgroup_load"),
            "ResidualAdd must emit simdgroup_load for the residual tile"
        );
    }

    #[test]
    fn test_residual_add_emits_addition() {
        let mut msl = MslBuilder::new();
        let config = MetalGemmConfig::default_apple9_f16();
        ResidualAddEpilogue.emit_epilogue(&mut msl, &config);
        let s = msl.finish();
        assert!(
            s.contains("c_elems[i] + r_elems[i]"),
            "ResidualAdd must emit element-wise addition"
        );
    }

    #[test]
    fn test_residual_add_tile_loops() {
        let mut msl = MslBuilder::new();
        let config = MetalGemmConfig::default_apple9_f16();
        ResidualAddEpilogue.emit_epilogue(&mut msl, &config);
        let s = msl.finish();
        // register_m=32, register_n=32 => tiles_m=4, tiles_n=4
        assert!(s.contains("tm < 4"), "Expected TILES_M=4 for 32/8");
        assert!(s.contains("tn < 4"), "Expected TILES_N=4 for 32/8");
    }
}
