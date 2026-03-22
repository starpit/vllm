use super::EpilogueAtom;
use crate::config::MetalGemmConfig;
use crate::msl_builder::MslBuilder;

/// Element-wise multiply epilogue — loads another tensor from device memory and
/// multiplies it element-wise with the C_sram accumulators after the K-loop.
///
/// Used for gate * up fusion in FFN blocks. The gate buffer pointer is expected
/// as an additional kernel parameter (`device half *gate`). Each simdgroup tile
/// loads its corresponding 8x8 block via simdgroup_load, then multiplies
/// through thread_elements().
pub struct ElementMulEpilogue;

impl EpilogueAtom for ElementMulEpilogue {
    fn emit_epilogue(&self, msl: &mut MslBuilder, config: &MetalGemmConfig) {
        msl.set("TILES_M", (config.register_m() / 8).to_string());
        msl.set("TILES_N", (config.register_n() / 8).to_string());
        msl.set("C_TYPE", config.register_precisions.c.msl_name());
        msl.block(
            r#"
// Element-wise multiply: C_sram *= gate
for (ushort tm = 0; tm < {{TILES_M}}; tm++) {
    for (ushort tn = 0; tn < {{TILES_N}}; tn++) {
        simdgroup_matrix<{{C_TYPE}}, 8, 8> G_tile;
        simdgroup_load(G_tile, gate + (sid_M_offset + tm * 8) * N + (sid_N_offset + tn * 8), N);
        thread auto &c_elems = C_sram[tm][tn].thread_elements();
        thread auto &g_elems = G_tile.thread_elements();
        for (int i = 0; i < 64; i++) {
            c_elems[i] = c_elems[i] * g_elems[i];
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
    fn test_element_mul_not_identity() {
        assert!(!ElementMulEpilogue.is_identity());
    }

    #[test]
    fn test_element_mul_emits_simdgroup_load() {
        let mut msl = MslBuilder::new();
        let config = MetalGemmConfig::default_apple9_f16();
        ElementMulEpilogue.emit_epilogue(&mut msl, &config);
        let s = msl.finish();
        assert!(
            s.contains("simdgroup_load"),
            "ElementMul must emit simdgroup_load for the gate tile"
        );
    }

    #[test]
    fn test_element_mul_emits_multiplication() {
        let mut msl = MslBuilder::new();
        let config = MetalGemmConfig::default_apple9_f16();
        ElementMulEpilogue.emit_epilogue(&mut msl, &config);
        let s = msl.finish();
        assert!(
            s.contains("c_elems[i] * g_elems[i]"),
            "ElementMul must emit element-wise multiplication"
        );
    }

    #[test]
    fn test_element_mul_tile_loops() {
        let mut msl = MslBuilder::new();
        let config = MetalGemmConfig::default_apple9_f16();
        ElementMulEpilogue.emit_epilogue(&mut msl, &config);
        let s = msl.finish();
        assert!(s.contains("tm < 4"), "Expected TILES_M=4 for 32/8");
        assert!(s.contains("tn < 4"), "Expected TILES_N=4 for 32/8");
    }
}
