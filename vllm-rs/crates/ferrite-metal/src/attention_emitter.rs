/// FlashAttention kernel MSL emitter.
///
/// Online softmax algorithm — never materializes the full attention matrix.
/// Softmax goes through threadgroup memory for correct per-row reduction.
/// Uses native Metal 4 simdgroup_matrix<T, 8> API.
use crate::config::Precision;
use crate::msl_builder::MslBuilder;

#[derive(Clone, Debug)]
pub struct AttentionConfig {
    pub block_r: u16,
    pub block_c: u16,
    pub d_head: u16,
    pub num_heads: u16,
    pub causal: bool,
    pub memory_precision: Precision,
    pub accumulator_precision: Precision,
}

impl AttentionConfig {
    pub fn default_f16() -> Self {
        Self {
            block_r: 32,
            block_c: 32,
            d_head: 128,
            num_heads: 32,
            causal: false,
            memory_precision: Precision::FP16,
            accumulator_precision: Precision::FP32,
        }
    }
    pub fn default_f16_causal() -> Self {
        let mut c = Self::default_f16();
        c.causal = true;
        c
    }
    pub fn tiles_r(&self) -> u16 {
        self.block_r / 8
    }
    pub fn tiles_c(&self) -> u16 {
        self.block_c / 8
    }
    pub fn tiles_d(&self) -> u16 {
        self.d_head / 8
    }
}

pub fn build_attention_msl(config: &AttentionConfig) -> String {
    let mut msl = MslBuilder::new();
    set_vars(&mut msl, config);

    msl.raw("#include <metal_stdlib>");
    msl.raw("using namespace metal;");
    msl.blank();
    msl.block(
        r#"
constant uint BLOCK_R = {{BLOCK_R}};
constant uint BLOCK_C = {{BLOCK_C}};
constant uint D_HEAD = {{D_HEAD}};
"#,
    );
    msl.block(
        r#"
kernel void attention(
    device {{MEM_TYPE}} *Q [[buffer(0)]],
    device {{MEM_TYPE}} *K [[buffer(1)]],
    device {{MEM_TYPE}} *V [[buffer(2)]],
    device {{ACC_TYPE}} *O [[buffer(3)]],
    constant uint4 *params [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort sidx [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]]
)
"#,
    );
    msl.open_brace();

    // Setup
    msl.block(
        r#"
uint seq_len = params[0][0];
uint d_head = params[0][1];
uint num_heads = params[0][2];
uint head_idx = gid.z;
uint row_offset = gid.y * BLOCK_R;
if (row_offset >= seq_len) return;
uint head_stride = seq_len * d_head;
device {{MEM_TYPE}} *Q_head = Q + head_idx * head_stride;
device {{MEM_TYPE}} *K_head = K + head_idx * head_stride;
device {{MEM_TYPE}} *V_head = V + head_idx * head_stride;
device {{ACC_TYPE}} *O_head = O + head_idx * head_stride;
threadgroup {{MEM_TYPE}} Q_smem[BLOCK_R * D_HEAD];
threadgroup {{MEM_TYPE}} K_smem[BLOCK_C * D_HEAD];
threadgroup {{MEM_TYPE}} V_smem[BLOCK_C * D_HEAD];
threadgroup {{ACC_TYPE}} S_smem[BLOCK_R * BLOCK_C];
threadgroup {{ACC_TYPE}} O_smem[BLOCK_R * D_HEAD];
ushort tid = sidx * 32 + lane_id;
for (uint i = tid; i < BLOCK_R * D_HEAD; i += 32) { O_smem[i] = 0; }
{{ACC_TYPE}} m_row[BLOCK_R];
{{ACC_TYPE}} l_row[BLOCK_R];
for (uint i = 0; i < BLOCK_R; i++) { m_row[i] = -INFINITY; l_row[i] = 0.0; }
"#,
    );

    // Load Q
    emit_load(
        &mut msl,
        "Q_smem",
        "Q_head",
        "BLOCK_R",
        "D_HEAD",
        "row_offset",
    );
    msl.raw("threadgroup_barrier(mem_flags::mem_threadgroup);");

    // KV loop
    if config.causal {
        msl.line("uint kv_end = min(uint(row_offset + BLOCK_R), seq_len);");
        msl.line("for (uint c = 0; c < kv_end; c += BLOCK_C) {");
    } else {
        msl.line("for (uint c = 0; c < seq_len; c += BLOCK_C) {");
    }
    msl.indent();

    // Load K
    emit_load(&mut msl, "K_smem", "K_head", "BLOCK_C", "D_HEAD", "c");
    msl.raw("threadgroup_barrier(mem_flags::mem_threadgroup);");

    // GEMM 1: S = Q * K^T via simdgroup MMA, store to S_smem
    msl.comment("GEMM 1: S = Q * K^T");
    msl.block(
        r#"
{
    simdgroup_matrix<{{ACC_TYPE}}, 8> S_acc[{{TILES_R}}][{{TILES_C}}];
    for (ushort tr = 0; tr < {{TILES_R}}; tr++)
        for (ushort tc = 0; tc < {{TILES_C}}; tc++)
            S_acc[tr][tc] = make_filled_simdgroup_matrix<{{ACC_TYPE}}, 8>(0);
    for (ushort dk = 0; dk < {{TILES_D}}; dk++) {
        for (ushort tr = 0; tr < {{TILES_R}}; tr++) {
            simdgroup_matrix<{{MEM_TYPE}}, 8> Q_frag;
            simdgroup_load(Q_frag, Q_smem, D_HEAD, ulong2(dk * 8, tr * 8));
            for (ushort tc = 0; tc < {{TILES_C}}; tc++) {
                simdgroup_matrix<{{MEM_TYPE}}, 8> KT_frag;
                simdgroup_load(KT_frag, K_smem, D_HEAD, ulong2(dk * 8, tc * 8), true);
                simdgroup_multiply_accumulate(S_acc[tr][tc], Q_frag, KT_frag, S_acc[tr][tc]);
            }
        }
    }
    for (ushort tr = 0; tr < {{TILES_R}}; tr++)
        for (ushort tc = 0; tc < {{TILES_C}}; tc++)
            simdgroup_store(S_acc[tr][tc], S_smem + tr * 8 * BLOCK_C, BLOCK_C, ulong2(tc * 8, 0));
}
threadgroup_barrier(mem_flags::mem_threadgroup);
"#,
    );

    // Scale + causal mask + per-row online softmax (all threadgroup)
    msl.comment("Scale, mask, per-row online softmax");
    msl.block(
        r#"
{
    {{ACC_TYPE}} scale = rsqrt({{ACC_TYPE}}(D_HEAD));
    for (uint i = tid; i < BLOCK_R * BLOCK_C; i += 32) S_smem[i] *= scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
"#,
    );
    if config.causal {
        msl.block(
            r#"
    for (uint i = tid; i < BLOCK_R * BLOCK_C; i += 32) {
        uint ri = row_offset + (i / BLOCK_C);
        uint cj = c + (i % BLOCK_C);
        if (cj > ri || ri >= seq_len || cj >= seq_len) S_smem[i] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"#,
        );
    }
    msl.block(
        r#"
    for (uint row = tid; row < BLOCK_R; row += 32) {
        if (row_offset + row >= seq_len) continue;
        {{ACC_TYPE}} row_max = -INFINITY;
        for (uint j = 0; j < BLOCK_C && (c + j) < seq_len; j++)
            row_max = max(row_max, S_smem[row * BLOCK_C + j]);
        {{ACC_TYPE}} m_new = max(m_row[row], row_max);
        {{ACC_TYPE}} correction = exp(m_row[row] - m_new);
        for (uint d = 0; d < D_HEAD; d++)
            O_smem[row * D_HEAD + d] *= correction;
        {{ACC_TYPE}} row_sum = 0.0;
        for (uint j = 0; j < BLOCK_C && (c + j) < seq_len; j++) {
            uint idx = row * BLOCK_C + j;
            S_smem[idx] = exp(S_smem[idx] - m_new);
            row_sum += S_smem[idx];
        }
        l_row[row] = correction * l_row[row] + row_sum;
        m_row[row] = m_new;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
"#,
    );

    // Load V
    emit_load(&mut msl, "V_smem", "V_head", "BLOCK_C", "D_HEAD", "c");
    msl.raw("threadgroup_barrier(mem_flags::mem_threadgroup);");

    // GEMM 2: O += P * V
    msl.comment("GEMM 2: O += P * V");
    msl.block(
        r#"
{
    simdgroup_matrix<{{ACC_TYPE}}, 8> O_acc[{{TILES_R}}][{{TILES_D}}];
    for (ushort tr = 0; tr < {{TILES_R}}; tr++)
        for (ushort td = 0; td < {{TILES_D}}; td++)
            O_acc[tr][td] = make_filled_simdgroup_matrix<{{ACC_TYPE}}, 8>(0);
    for (ushort tc = 0; tc < {{TILES_C}}; tc++) {
        for (ushort tr = 0; tr < {{TILES_R}}; tr++) {
            simdgroup_matrix<{{ACC_TYPE}}, 8> P_frag;
            simdgroup_load(P_frag, S_smem + tr * 8 * BLOCK_C, BLOCK_C, ulong2(tc * 8, 0));
            for (ushort td = 0; td < {{TILES_D}}; td++) {
                simdgroup_matrix<{{MEM_TYPE}}, 8> V_frag;
                simdgroup_load(V_frag, V_smem, D_HEAD, ulong2(td * 8, tc * 8));
                simdgroup_multiply_accumulate(O_acc[tr][td], P_frag, V_frag, O_acc[tr][td]);
            }
        }
    }
    threadgroup {{ACC_TYPE}} O_scratch[8 * 8];
    for (ushort tr = 0; tr < {{TILES_R}}; tr++) {
        for (ushort td = 0; td < {{TILES_D}}; td++) {
            simdgroup_store(O_acc[tr][td], O_scratch, 8, ulong2(0, 0));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (ushort i = tid; i < 64; i += 32) {
                ushort r = i / 8;
                ushort d = i % 8;
                O_smem[(tr * 8 + r) * D_HEAD + (td * 8 + d)] += O_scratch[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
"#,
    );

    msl.dedent();
    msl.raw("}"); // KV loop
    msl.blank();

    // Final: O /= l
    msl.comment("Final normalization");
    msl.block(
        r#"
for (uint row = tid; row < BLOCK_R; row += 32) {
    if (row_offset + row >= seq_len) continue;
    {{ACC_TYPE}} inv_l = (l_row[row] > 0) ? (1.0 / l_row[row]) : 0.0;
    for (uint d = 0; d < D_HEAD; d++)
        O_smem[row * D_HEAD + d] *= inv_l;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
"#,
    );

    // Store
    msl.comment("Store output");
    msl.block(
        r#"
for (uint i = tid; i < BLOCK_R * D_HEAD; i += 32) {
    uint row = i / D_HEAD;
    uint col = i % D_HEAD;
    uint global_row = row_offset + row;
    if (global_row < seq_len)
        O_head[global_row * d_head + col] = O_smem[i];
}
"#,
    );

    msl.close_brace();
    msl.finish()
}

fn emit_load(msl: &mut MslBuilder, smem: &str, dev: &str, rows: &str, cols: &str, base: &str) {
    msl.raw("{");
    msl.indent();
    msl.raw(&format!(
        "for (uint i = tid; i < {} * {}; i += 32) {{",
        rows, cols
    ));
    msl.indent();
    msl.raw(&format!("uint row = i / {};", cols));
    msl.raw(&format!("uint col = i % {};", cols));
    msl.raw(&format!("uint gr = {} + row;", base));
    msl.raw(&format!(
        "{}[i] = (gr < seq_len) ? {}[gr * d_head + col] : 0;",
        smem, dev
    ));
    msl.dedent();
    msl.raw("}");
    msl.dedent();
    msl.raw("}");
}

fn set_vars(msl: &mut MslBuilder, c: &AttentionConfig) {
    msl.set("BLOCK_R", c.block_r.to_string());
    msl.set("BLOCK_C", c.block_c.to_string());
    msl.set("D_HEAD", c.d_head.to_string());
    msl.set("NUM_HEADS", c.num_heads.to_string());
    msl.set("TILES_R", c.tiles_r().to_string());
    msl.set("TILES_C", c.tiles_c().to_string());
    msl.set("TILES_D", c.tiles_d().to_string());
    msl.set("MEM_TYPE", c.memory_precision.msl_name());
    msl.set("ACC_TYPE", c.accumulator_precision.msl_name());
}

#[cfg(test)]
mod tests {
    use super::*;
    fn default_msl() -> String {
        build_attention_msl(&AttentionConfig::default_f16())
    }
    fn causal_msl() -> String {
        build_attention_msl(&AttentionConfig::default_f16_causal())
    }

    #[test]
    fn test_kernel_structure() {
        let msl = default_msl();
        assert!(msl.contains("kernel void attention("));
        assert!(msl.contains("simdgroup_multiply_accumulate"));
        assert!(msl.contains("exp("));
        assert!(msl.contains("rsqrt("));
    }

    #[test]
    fn test_both_gemms() {
        let msl = default_msl();
        assert!(msl.matches("simdgroup_multiply_accumulate").count() >= 2);
    }

    #[test]
    fn test_online_softmax() {
        let msl = default_msl();
        assert!(msl.contains("m_row"));
        assert!(msl.contains("l_row"));
        assert!(msl.contains("correction"));
        assert!(msl.contains("row * BLOCK_C"));
    }

    #[test]
    fn test_k_transpose() {
        let msl = default_msl();
        assert!(msl.contains("true)"));
    }

    #[test]
    fn test_causal() {
        let msl = causal_msl();
        assert!(msl.contains("-INFINITY"));
        assert!(msl.contains("cj > ri"));
    }

    #[test]
    fn test_balanced_braces() {
        for msl in [default_msl(), causal_msl()] {
            let o = msl.chars().filter(|c| *c == '{').count();
            let c = msl.chars().filter(|c| *c == '}').count();
            assert_eq!(o, c, "Unbalanced: {} vs {}", o, c);
        }
    }

    #[test]
    fn test_no_template_vars() {
        for msl in [default_msl(), causal_msl()] {
            assert!(!msl.contains("{{"), "Unreplaced template var");
        }
    }
}
