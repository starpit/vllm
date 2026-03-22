use super::TileCopyAtom;
use crate::config::MetalGemmConfig;
use crate::msl_builder::MslBuilder;

/// Loop-based tile copy (universal, works on all Metal versions).
///
/// All threads cooperate to copy tiles from device to threadgroup memory
/// with zero-fill padding. No simdgroup_event dependency.
pub struct LoopTileCopy;

impl TileCopyAtom for LoopTileCopy {
    fn emit_tile_load(&self, msl: &mut MslBuilder, config: &MetalGemmConfig) {
        let a_lead = config.leading_block_dim('A');
        let b_lead = config.leading_block_dim('B');
        let a_rows = if config.transpose[0] {
            config.block_k
        } else {
            config.block_m
        };
        let b_rows = if config.transpose[1] {
            config.block_k
        } else {
            config.block_n
        };

        msl.set("THREADGROUP_SIZE", config.threadgroup_size().to_string());
        msl.set("A_TG_TOTAL", (a_rows as u32 * a_lead as u32).to_string());
        msl.set("A_TG_COLS", a_lead.to_string());
        msl.set("B_TG_TOTAL", (b_rows as u32 * b_lead as u32).to_string());
        msl.set("B_TG_COLS", b_lead.to_string());

        msl.block(
            r#"
{
    ushort tid = sidx * 32 + lane_id;
    uint A_lead_dim = A_trans ? M : K;
    uint B_lead_dim = B_trans ? K : N;
    ushort M_tile = min(uint(M_group), M - M_offset);
    ushort N_tile = min(uint(N_group), N - N_offset);
    ushort K_tile = min(uint(K_group), K - k);

    for (ushort i = tid; i < {{A_TG_TOTAL}}; i += {{THREADGROUP_SIZE}}) {
        ushort row = i / {{A_TG_COLS}};
        ushort col = i % {{A_TG_COLS}};
        bool valid = A_trans ? (row < K_tile && col < M_tile) : (row < M_tile && col < K_tile);
        if (valid) {
            uint idx = A_trans
                ? (M_offset + col) * A_lead_dim + (k + row)
                : (M_offset + row) * A_lead_dim + (k + col);
            A_block[i] = A[idx];
        } else {
            A_block[i] = 0;
        }
    }

    for (ushort i = tid; i < {{B_TG_TOTAL}}; i += {{THREADGROUP_SIZE}}) {
        ushort row = i / {{B_TG_COLS}};
        ushort col = i % {{B_TG_COLS}};
        bool valid = B_trans ? (row < K_tile && col < N_tile) : (row < N_tile && col < K_tile);
        if (valid) {
            uint idx = B_trans
                ? (N_offset + col) * B_lead_dim + (k + row)
                : (N_offset + row) * B_lead_dim + (k + col);
            B_block[i] = B[idx];
        } else {
            B_block[i] = 0;
        }
    }
}
"#,
        );
    }

    fn emit_tile_sync(&self, msl: &mut MslBuilder, _config: &MetalGemmConfig) {
        msl.raw("threadgroup_barrier(mem_flags::mem_threadgroup);");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emits_cooperative_loop() {
        let config = MetalGemmConfig::default_apple8_f16();
        let mut msl = MslBuilder::new();
        LoopTileCopy.emit_tile_load(&mut msl, &config);
        let s = msl.finish();
        assert!(s.contains("for (ushort i = tid"));
        assert!(s.contains("A_block[i]"));
        assert!(s.contains("B_block[i]"));
        assert!(s.contains("= 0"), "Must zero-fill padding");
    }

    #[test]
    fn test_sync_emits_barrier() {
        let config = MetalGemmConfig::default_apple8_f16();
        let mut msl = MslBuilder::new();
        LoopTileCopy.emit_tile_sync(&mut msl, &config);
        assert!(msl.finish().contains("threadgroup_barrier"));
    }
}
