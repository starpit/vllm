use crate::msl_builder::MslBuilder;

/// RmsNorm atom — standalone reduction + scale operation.
///
/// Normalizes input vectors by their RMS value and scales by gamma weights.
/// Formula: output[i] = (x[i] / sqrt(mean(x^2) + eps)) * gamma[i]
///
/// This is NOT a GEMM sub-atom. It emits a complete kernel body that:
/// 1. Loads input x and gamma from device memory
/// 2. Computes sum(x^2) via simdgroup reduction (simd_sum) + cross-simdgroup
///    reduction through threadgroup memory
/// 3. Computes rsqrt(mean + eps)
/// 4. Writes output[i] = x[i] * scale * gamma[i]
pub struct RmsNormAtom {
    /// Epsilon for numerical stability (typically 1e-5 or 1e-6).
    pub eps: f32,
    /// MSL type name for the input/output data (e.g. "half", "float").
    pub dtype: String,
    /// Number of simdgroups per threadgroup.
    pub simdgroups_per_threadgroup: u32,
}

impl RmsNormAtom {
    pub fn new(eps: f32, dtype: &str, simdgroups_per_threadgroup: u32) -> Self {
        Self {
            eps,
            dtype: dtype.to_string(),
            simdgroups_per_threadgroup,
        }
    }

    /// Emit the complete RmsNorm kernel MSL.
    ///
    /// Assumes one threadgroup processes one row of length `hidden_size`.
    /// Each thread processes multiple elements in a strided loop.
    /// Reduction uses simd_sum within a simdgroup, then cross-simdgroup
    /// reduction via threadgroup shared memory.
    pub fn emit_kernel(&self, mut msl: MslBuilder) -> String {
        msl.set("DTYPE", &self.dtype);
        msl.set("EPS", format!("{:e}", self.eps));
        msl.set(
            "SIMDGROUPS_PER_TG",
            self.simdgroups_per_threadgroup.to_string(),
        );

        msl.raw("#include <metal_stdlib>");
        msl.raw("using namespace metal;");
        msl.blank();

        msl.block(
            r#"
kernel void rmsnorm(
    device {{DTYPE}} *input [[buffer(0)]],
    device {{DTYPE}} *gamma [[buffer(1)]],
    device {{DTYPE}} *output [[buffer(2)]],
    constant uint &hidden_size [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]
)
"#,
        );
        msl.open_brace();

        // Threadgroup memory for cross-simdgroup reduction
        msl.block(
            r#"
threadgroup float shared_sums[{{SIMDGROUPS_PER_TG}}];

// Each row starts at tgid * hidden_size
uint row_offset = tgid * hidden_size;
"#,
        );

        // Phase 1: Compute partial sum of squares (strided loop)
        msl.block(
            r#"
// Phase 1: Each thread accumulates sum(x^2) over its elements
float thread_sum = 0.0f;
for (uint i = tid; i < hidden_size; i += {{SIMDGROUPS_PER_TG}} * 32) {
    float val = float(input[row_offset + i]);
    thread_sum += val * val;
}
"#,
        );

        // Phase 2: simdgroup reduction via simd_sum
        msl.block(
            r#"
// Phase 2: Reduce within each simdgroup using simd_sum
float simd_total = simd_sum(thread_sum);
"#,
        );

        // Phase 3: Cross-simdgroup reduction via threadgroup memory
        msl.block(
            r#"
// Phase 3: Cross-simdgroup reduction via threadgroup memory
if (simd_lane == 0) {
    shared_sums[simd_gid] = simd_total;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// First simdgroup reduces across all simdgroup partial sums
float total_sum = 0.0f;
if (simd_gid == 0 && simd_lane < {{SIMDGROUPS_PER_TG}}) {
    total_sum = shared_sums[simd_lane];
}
total_sum = simd_sum(total_sum);

// Broadcast total_sum to all simdgroups via shared memory
if (simd_gid == 0 && simd_lane == 0) {
    shared_sums[0] = total_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
total_sum = shared_sums[0];
"#,
        );

        // Phase 4: Compute scale = rsqrt(mean + eps)
        msl.block(
            r#"
// Phase 4: Compute normalization scale
float scale = rsqrt(total_sum / float(hidden_size) + {{EPS}});
"#,
        );

        // Phase 5: Apply normalization and gamma scaling
        msl.block(
            r#"
// Phase 5: Normalize and scale by gamma
for (uint i = tid; i < hidden_size; i += {{SIMDGROUPS_PER_TG}} * 32) {
    float val = float(input[row_offset + i]);
    output[row_offset + i] = {{DTYPE}}(val * scale * float(gamma[i]));
}
"#,
        );

        msl.close_brace();
        msl.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emit_rmsnorm_msl() -> String {
        let atom = RmsNormAtom::new(1e-5, "half", 4);
        let msl = MslBuilder::new();
        atom.emit_kernel(msl)
    }

    #[test]
    fn test_emits_simd_sum_for_reduction() {
        let source = emit_rmsnorm_msl();
        assert!(
            source.contains("simd_sum"),
            "RmsNorm MSL must use simd_sum for simdgroup reduction"
        );
    }

    #[test]
    fn test_emits_gamma_scaling() {
        let source = emit_rmsnorm_msl();
        assert!(
            source.contains("gamma"),
            "RmsNorm MSL must scale by gamma weights"
        );
    }

    #[test]
    fn test_emits_rsqrt() {
        let source = emit_rmsnorm_msl();
        assert!(
            source.contains("rsqrt"),
            "RmsNorm MSL must use rsqrt for 1/sqrt(mean + eps)"
        );
    }

    #[test]
    fn test_emits_epsilon() {
        let source = emit_rmsnorm_msl();
        // eps = 1e-5 should appear in the source
        assert!(
            source.contains("1e-5"),
            "RmsNorm MSL must include epsilon constant"
        );
    }

    #[test]
    fn test_emits_threadgroup_barrier() {
        let source = emit_rmsnorm_msl();
        assert!(
            source.contains("threadgroup_barrier"),
            "RmsNorm MSL must have threadgroup_barrier for cross-simdgroup sync"
        );
    }

    #[test]
    fn test_emits_shared_memory_for_cross_simdgroup() {
        let source = emit_rmsnorm_msl();
        assert!(
            source.contains("threadgroup float shared_sums"),
            "RmsNorm MSL must declare threadgroup memory for cross-simdgroup reduction"
        );
    }

    #[test]
    fn test_dtype_substitution() {
        let atom = RmsNormAtom::new(1e-6, "float", 2);
        let msl = MslBuilder::new();
        let source = atom.emit_kernel(msl);
        assert!(
            source.contains("device float *input"),
            "RmsNorm MSL must substitute dtype into kernel signature"
        );
        assert!(
            source.contains("device float *gamma"),
            "RmsNorm MSL must substitute dtype for gamma buffer"
        );
    }

    #[test]
    fn test_kernel_signature() {
        let source = emit_rmsnorm_msl();
        assert!(
            source.contains("kernel void rmsnorm"),
            "RmsNorm MSL must emit a Metal kernel function"
        );
    }
}
