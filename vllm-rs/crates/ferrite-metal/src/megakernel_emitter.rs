/// Megakernel emitter — sequences multiple operations into one Metal kernel.
///
/// Uses a persistent threadgroup pattern with atomic phase counters
/// for cross-threadgroup synchronization. Each phase (RmsNorm, GEMM,
/// Attention, etc.) uses a different subset of threadgroups.
///
/// For the initial version, handles one transformer block:
///   RmsNorm → Convert → GEMM(Q,K,V) → Convert → Attention → Convert → GEMM(O) → ResidualAdd
use crate::config::Precision;
use crate::msl_builder::MslBuilder;

/// Configuration for the megakernel.
#[derive(Clone, Debug)]
pub struct MegakernelConfig {
    pub seq_len: u32,
    pub d_model: u32,
    pub d_head: u32,
    pub d_ffn: u32, // FFN intermediate dimension (typically 4 * d_model or 8/3 * d_model)
    pub num_heads: u32,
    pub num_threadgroups: u32,
    pub threads_per_tg: u32,
    pub eps: f32,
    pub causal: bool,
    pub precision: Precision,
}

impl MegakernelConfig {
    /// Small config for testing.
    pub fn test_config() -> Self {
        Self {
            seq_len: 4,
            d_model: 8,
            d_head: 8,
            d_ffn: 16, // 2x d_model for testing
            num_heads: 1,
            num_threadgroups: 1,
            threads_per_tg: 32,
            eps: 1e-5,
            causal: false,
            precision: Precision::FP16,
        }
    }
}

/// Build a megakernel that executes one transformer block.
///
/// All operations run within one kernel dispatch. Data flows through
/// device memory buffers between phases. Phases are synchronized
/// via atomic counters when multiple threadgroups are used.
///
/// For num_threadgroups=1 (small models), no atomic sync needed.
pub fn build_megakernel_msl(config: &MegakernelConfig) -> String {
    let mut msl = MslBuilder::new();

    msl.set("SEQ_LEN", config.seq_len.to_string());
    msl.set("D_MODEL", config.d_model.to_string());
    msl.set("D_HEAD", config.d_head.to_string());
    msl.set("NUM_HEADS", config.num_heads.to_string());
    msl.set("EPS", format!("{:.10}", config.eps));
    msl.set("MEM_TYPE", config.precision.msl_name());
    msl.set("D_FFN", config.d_ffn.to_string());
    msl.set("NUM_TG", config.num_threadgroups.to_string());
    msl.set("THREADS_PER_TG", config.threads_per_tg.to_string());

    msl.raw("#include <metal_stdlib>");
    msl.raw("using namespace metal;");
    msl.blank();

    // Kernel signature: all buffers for one transformer block
    msl.block(
        r#"
kernel void transformer_block(
    // Input / output
    device float *x [[buffer(0)]],           // [seq_len, d_model] f32 input
    device float *out [[buffer(1)]],         // [seq_len, d_model] f32 output

    // Weights
    device float *gamma [[buffer(2)]],       // [d_model] RmsNorm weights
    device {{MEM_TYPE}} *Wq [[buffer(3)]],   // [d_model, d_model] query
    device {{MEM_TYPE}} *Wk [[buffer(4)]],   // [d_model, d_model] key
    device {{MEM_TYPE}} *Wv [[buffer(5)]],   // [d_model, d_model] value
    device {{MEM_TYPE}} *Wo [[buffer(6)]],   // [d_model, d_model] output proj

    // Intermediates (pre-allocated by host)
    device float *h [[buffer(7)]],           // [seq_len, d_model] RmsNorm output
    device {{MEM_TYPE}} *h_f16 [[buffer(8)]],// [seq_len, d_model] converted
    device float *qkv [[buffer(9)]],         // [3, seq_len, d_model] Q,K,V f32
    device {{MEM_TYPE}} *qkv_f16 [[buffer(10)]], // [3, seq_len, d_model] Q,K,V f16
    device float *attn_out [[buffer(11)]],   // [seq_len, d_model] attention output
    device {{MEM_TYPE}} *attn_f16 [[buffer(12)]], // converted
    device float *o [[buffer(13)]],          // [seq_len, d_model] output projection

    // FFN weights
    device float *gamma_ffn [[buffer(14)]],  // [d_model] FFN RmsNorm weights
    device {{MEM_TYPE}} *W_gate [[buffer(15)]], // [d_ffn, d_model] gate projection
    device {{MEM_TYPE}} *W_up [[buffer(16)]],   // [d_ffn, d_model] up projection
    device {{MEM_TYPE}} *W_down [[buffer(17)]], // [d_model, d_ffn] down projection

    // FFN intermediates
    device float *h_ffn [[buffer(18)]],      // [seq_len, d_model] FFN RmsNorm output
    device {{MEM_TYPE}} *h_ffn_f16 [[buffer(19)]], // converted
    device float *gate_out [[buffer(20)]],   // [seq_len, d_ffn] gate projection
    device float *up_out [[buffer(21)]],     // [seq_len, d_ffn] up projection
    device {{MEM_TYPE}} *ffn_act [[buffer(22)]], // [seq_len, d_ffn] silu(gate)*up as f16
    device float *down_out [[buffer(23)]],   // [seq_len, d_model] down projection

    // Sync (for multi-threadgroup)
    device atomic_uint *phase_counter [[buffer(24)]],

    // Thread info
    uint gid [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    ushort sidx [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]]
)
"#,
    );
    msl.open_brace();

    // Constants
    msl.block(
        r#"
uint seq_len = {{SEQ_LEN}};
uint d_model = {{D_MODEL}};
uint d_head = {{D_HEAD}};
uint num_heads = {{NUM_HEADS}};
uint d_ffn = {{D_FFN}};
uint total_elems = seq_len * d_model;
uint ffn_elems = seq_len * d_ffn;
uint num_tg = {{NUM_TG}};
"#,
    );

    // Phase sync helper (no-op for single threadgroup)
    if config.num_threadgroups > 1 {
        msl.block(
            r#"
// Wait for all threadgroups to reach this phase.
auto sync_phase = [&](uint expected_count) {
    if (tid_in_tg == 0) {
        atomic_fetch_add_explicit(phase_counter, 1u, memory_order_release);
        while (atomic_load_explicit(phase_counter, memory_order_acquire) < expected_count) {}
    }
    threadgroup_barrier(mem_flags::mem_device);
};
"#,
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase 1: RmsNorm(x) → h
    // ═══════════════════════════════════════════════════════════════
    msl.comment("Phase 1: RmsNorm");
    msl.block(
        r#"
{
    // Each threadgroup handles rows in a strided pattern
    for (uint row = gid; row < seq_len; row += num_tg) {
        float sum_sq = 0.0;
        for (uint i = tid_in_tg; i < d_model; i += {{THREADS_PER_TG}}) {
            float val = x[row * d_model + i];
            sum_sq += val * val;
        }
        // Reduce within simdgroup
        sum_sq = simd_sum(sum_sq);
        // Cross-simdgroup reduce via shared memory
        threadgroup float shared_sum[1];
        if (sidx == 0 && lane_id == 0) shared_sum[0] = sum_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = shared_sum[0];
        float scale = rsqrt(total / float(d_model) + {{EPS}});
        for (uint i = tid_in_tg; i < d_model; i += {{THREADS_PER_TG}}) {
            h[row * d_model + i] = x[row * d_model + i] * scale * gamma[i];
        }
    }
}
"#,
    );

    emit_phase_sync(&mut msl, config, 1);

    // ═══════════════════════════════════════════════════════════════
    // Phase 2: Convert h f32 → f16
    // ═══════════════════════════════════════════════════════════════
    msl.comment("Phase 2: Convert f32 → f16");
    msl.block(
        r#"
{
    for (uint i = gid * {{THREADS_PER_TG}} + tid_in_tg; i < total_elems;
         i += num_tg * {{THREADS_PER_TG}}) {
        h_f16[i] = {{MEM_TYPE}}(h[i]);
    }
}
"#,
    );

    emit_phase_sync(&mut msl, config, 2);

    // ═══════════════════════════════════════════════════════════════
    // Phase 3: GEMM Q,K,V = h_f16 @ W{q,k,v}
    // ═══════════════════════════════════════════════════════════════
    // For simplicity with single threadgroup: naive GEMM per element
    msl.comment("Phase 3: GEMM Q, K, V (naive for single-TG)");
    msl.block(
        r#"
{
    device {{MEM_TYPE}} *weights[3] = {Wq, Wk, Wv};
    for (uint w = 0; w < 3; w++) {
        device float *dst = qkv + w * total_elems;
        device {{MEM_TYPE}} *W = weights[w];
        // C = A @ B^T where A=[seq_len, d_model], B=[d_model, d_model]
        for (uint idx = gid * {{THREADS_PER_TG}} + tid_in_tg; idx < total_elems;
             idx += num_tg * {{THREADS_PER_TG}}) {
            uint row = idx / d_model;
            uint col = idx % d_model;
            float sum = 0.0;
            for (uint k = 0; k < d_model; k++) {
                sum += float(h_f16[row * d_model + k]) * float(W[col * d_model + k]);
            }
            dst[idx] = sum;
        }
    }
}
"#,
    );

    emit_phase_sync(&mut msl, config, 3);

    // ═══════════════════════════════════════════════════════════════
    // Phase 4: Convert QKV f32 → f16
    // ═══════════════════════════════════════════════════════════════
    msl.comment("Phase 4: Convert QKV f32 → f16");
    msl.block(
        r#"
{
    for (uint i = gid * {{THREADS_PER_TG}} + tid_in_tg; i < 3 * total_elems;
         i += num_tg * {{THREADS_PER_TG}}) {
        qkv_f16[i] = {{MEM_TYPE}}(qkv[i]);
    }
}
"#,
    );

    emit_phase_sync(&mut msl, config, 4);

    // ═══════════════════════════════════════════════════════════════
    // Phase 5: Attention
    // ═══════════════════════════════════════════════════════════════
    msl.comment("Phase 5: Attention (naive for single-TG)");
    msl.block(
        r#"
{
    device {{MEM_TYPE}} *Q = qkv_f16;
    device {{MEM_TYPE}} *K = qkv_f16 + total_elems;
    device {{MEM_TYPE}} *V = qkv_f16 + 2 * total_elems;
    float attn_scale = rsqrt(float(d_head));

    for (uint row = gid; row < seq_len; row += num_tg) {
        // S[row, :] = Q[row] dot K[:] * scale
        // Only one thread per row for simplicity
        if (tid_in_tg == 0) {
            float s[{{SEQ_LEN}}];
            float max_s = -INFINITY;
            for (uint j = 0; j < seq_len; j++) {
                float dot = 0.0;
                for (uint d = 0; d < d_head; d++) {
                    dot += float(Q[row * d_model + d]) * float(K[j * d_model + d]);
                }
                s[j] = dot * attn_scale;
                max_s = max(max_s, s[j]);
            }
            // Softmax
            float sum_exp = 0.0;
            for (uint j = 0; j < seq_len; j++) {
                s[j] = exp(s[j] - max_s);
                sum_exp += s[j];
            }
            float inv_sum = 1.0 / sum_exp;
            // O = P @ V
            for (uint d = 0; d < d_head; d++) {
                float val = 0.0;
                for (uint j = 0; j < seq_len; j++) {
                    val += s[j] * inv_sum * float(V[j * d_model + d]);
                }
                attn_out[row * d_model + d] = val;
            }
        }
    }
}
"#,
    );

    emit_phase_sync(&mut msl, config, 5);

    // ═══════════════════════════════════════════════════════════════
    // Phase 6: Convert attention output f32 → f16
    // ═══════════════════════════════════════════════════════════════
    msl.comment("Phase 6: Convert attn f32 → f16");
    msl.block(
        r#"
{
    for (uint i = gid * {{THREADS_PER_TG}} + tid_in_tg; i < total_elems;
         i += num_tg * {{THREADS_PER_TG}}) {
        attn_f16[i] = {{MEM_TYPE}}(attn_out[i]);
    }
}
"#,
    );

    emit_phase_sync(&mut msl, config, 6);

    // ═══════════════════════════════════════════════════════════════
    // Phase 7: GEMM O = attn_f16 @ Wo
    // ═══════════════════════════════════════════════════════════════
    msl.comment("Phase 7: GEMM output projection (naive)");
    msl.block(
        r#"
{
    for (uint idx = gid * {{THREADS_PER_TG}} + tid_in_tg; idx < total_elems;
         idx += num_tg * {{THREADS_PER_TG}}) {
        uint row = idx / d_model;
        uint col = idx % d_model;
        float sum = 0.0;
        for (uint k = 0; k < d_model; k++) {
            sum += float(attn_f16[row * d_model + k]) * float(Wo[col * d_model + k]);
        }
        o[idx] = sum;
    }
}
"#,
    );

    emit_phase_sync(&mut msl, config, 7);

    // ═══════════════════════════════════════════════════════════════
    // Phase 8: Residual add: x = x + o (in-place, feeds into FFN)
    // ═══════════════════════════════════════════════════════════════
    msl.comment("Phase 8: Residual add (attention)");
    msl.block(
        r#"
{
    for (uint i = gid * {{THREADS_PER_TG}} + tid_in_tg; i < total_elems;
         i += num_tg * {{THREADS_PER_TG}}) {
        x[i] = x[i] + o[i];
    }
}
"#,
    );

    emit_phase_sync(&mut msl, config, 8);

    // ═══════════════════════════════════════════════════════════════
    // Phase 9: FFN RmsNorm(x) → h_ffn
    // ═══════════════════════════════════════════════════════════════
    msl.comment("Phase 9: FFN RmsNorm");
    msl.block(
        r#"
{
    for (uint row = gid; row < seq_len; row += num_tg) {
        float sum_sq = 0.0;
        for (uint i = tid_in_tg; i < d_model; i += {{THREADS_PER_TG}}) {
            float val = x[row * d_model + i];
            sum_sq += val * val;
        }
        sum_sq = simd_sum(sum_sq);
        threadgroup float shared_sum_ffn[1];
        if (sidx == 0 && lane_id == 0) shared_sum_ffn[0] = sum_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_sq = shared_sum_ffn[0];
        float scale = rsqrt(total_sq / float(d_model) + {{EPS}});
        for (uint i = tid_in_tg; i < d_model; i += {{THREADS_PER_TG}}) {
            h_ffn[row * d_model + i] = x[row * d_model + i] * scale * gamma_ffn[i];
        }
    }
}
"#,
    );

    emit_phase_sync(&mut msl, config, 9);

    // ═══════════════════════════════════════════════════════════════
    // Phase 10: Convert h_ffn f32 → f16, then GEMM gate and up
    // ═══════════════════════════════════════════════════════════════
    msl.comment("Phase 10: Convert + GEMM gate, up");
    msl.block(
        r#"
{
    // Convert h_ffn to f16
    for (uint i = gid * {{THREADS_PER_TG}} + tid_in_tg; i < total_elems;
         i += num_tg * {{THREADS_PER_TG}}) {
        h_ffn_f16[i] = {{MEM_TYPE}}(h_ffn[i]);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // GEMM gate = h_ffn_f16 @ W_gate^T  [seq_len, d_model] × [d_ffn, d_model]^T → [seq_len, d_ffn]
    for (uint idx = gid * {{THREADS_PER_TG}} + tid_in_tg; idx < ffn_elems;
         idx += num_tg * {{THREADS_PER_TG}}) {
        uint row = idx / d_ffn;
        uint col = idx % d_ffn;
        float sum = 0.0;
        for (uint k = 0; k < d_model; k++) {
            sum += float(h_ffn_f16[row * d_model + k]) * float(W_gate[col * d_model + k]);
        }
        gate_out[idx] = sum;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // GEMM up = h_ffn_f16 @ W_up^T  [seq_len, d_model] × [d_ffn, d_model]^T → [seq_len, d_ffn]
    for (uint idx = gid * {{THREADS_PER_TG}} + tid_in_tg; idx < ffn_elems;
         idx += num_tg * {{THREADS_PER_TG}}) {
        uint row = idx / d_ffn;
        uint col = idx % d_ffn;
        float sum = 0.0;
        for (uint k = 0; k < d_model; k++) {
            sum += float(h_ffn_f16[row * d_model + k]) * float(W_up[col * d_model + k]);
        }
        up_out[idx] = sum;
    }
}
"#,
    );

    emit_phase_sync(&mut msl, config, 10);

    // ═══════════════════════════════════════════════════════════════
    // Phase 11: SiLU(gate) * up → ffn_act (as f16)
    // ═══════════════════════════════════════════════════════════════
    msl.comment("Phase 11: SiLU(gate) * up → f16");
    msl.block(
        r#"
{
    for (uint i = gid * {{THREADS_PER_TG}} + tid_in_tg; i < ffn_elems;
         i += num_tg * {{THREADS_PER_TG}}) {
        float g = gate_out[i];
        float silu_g = g / (1.0 + exp(-g));
        ffn_act[i] = {{MEM_TYPE}}(silu_g * up_out[i]);
    }
}
"#,
    );

    emit_phase_sync(&mut msl, config, 11);

    // ═══════════════════════════════════════════════════════════════
    // Phase 12: GEMM down = ffn_act @ W_down^T  [seq_len, d_ffn] × [d_model, d_ffn]^T → [seq_len, d_model]
    // ═══════════════════════════════════════════════════════════════
    msl.comment("Phase 12: GEMM down projection");
    msl.block(
        r#"
{
    for (uint idx = gid * {{THREADS_PER_TG}} + tid_in_tg; idx < total_elems;
         idx += num_tg * {{THREADS_PER_TG}}) {
        uint row = idx / d_model;
        uint col = idx % d_model;
        float sum = 0.0;
        for (uint k = 0; k < d_ffn; k++) {
            sum += float(ffn_act[row * d_ffn + k]) * float(W_down[col * d_ffn + k]);
        }
        down_out[idx] = sum;
    }
}
"#,
    );

    emit_phase_sync(&mut msl, config, 12);

    // ═══════════════════════════════════════════════════════════════
    // Phase 13: Final residual add: out = x + down_out
    // ═══════════════════════════════════════════════════════════════
    msl.comment("Phase 13: Residual add (FFN)");
    msl.block(
        r#"
{
    for (uint i = gid * {{THREADS_PER_TG}} + tid_in_tg; i < total_elems;
         i += num_tg * {{THREADS_PER_TG}}) {
        out[i] = x[i] + down_out[i];
    }
}
"#,
    );

    msl.close_brace();
    msl.finish()
}

fn emit_phase_sync(msl: &mut MslBuilder, config: &MegakernelConfig, phase: u32) {
    if config.num_threadgroups > 1 {
        msl.raw(&format!(
            "sync_phase({} * num_tg); // wait for phase {}",
            phase, phase
        ));
    }
    // Device memory fence to ensure writes are visible
    msl.raw("threadgroup_barrier(mem_flags::mem_device);");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_megakernel_compiles() {
        let msl = build_megakernel_msl(&MegakernelConfig::test_config());
        assert!(msl.contains("kernel void transformer_block("));
        assert!(msl.contains("Phase 1: RmsNorm"));
        assert!(msl.contains("Phase 5: Attention"));
        assert!(msl.contains("Phase 11: SiLU"));
        assert!(msl.contains("Phase 13: Residual add"));
    }

    #[test]
    fn test_megakernel_has_all_phases() {
        let msl = build_megakernel_msl(&MegakernelConfig::test_config());
        for phase in 1..=13 {
            assert!(
                msl.contains(&format!("Phase {}", phase)),
                "Missing phase {}",
                phase
            );
        }
    }

    #[test]
    fn test_megakernel_no_template_vars() {
        let msl = build_megakernel_msl(&MegakernelConfig::test_config());
        assert!(
            !msl.contains("{{"),
            "Unreplaced: {}",
            msl.lines().find(|l| l.contains("{{")).unwrap_or("???")
        );
    }

    #[test]
    fn test_megakernel_balanced_braces() {
        let msl = build_megakernel_msl(&MegakernelConfig::test_config());
        let o = msl.chars().filter(|c| *c == '{').count();
        let c = msl.chars().filter(|c| *c == '}').count();
        assert_eq!(o, c, "Unbalanced: {} vs {}", o, c);
    }
}
