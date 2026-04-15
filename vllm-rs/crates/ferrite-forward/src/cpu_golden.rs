// SPDX-License-Identifier: Apache-2.0
//
// Golden reference code — straightforward per-index loops win on
// readability. Suppress the usual clippy "iterate by element"
// pedantry so the math stays legible.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
//! CPU golden model implementations for each ferrite op.
//!
//! Pure-Rust reference implementations of the same operations the
//! GPU kernels perform. Used for correctness testing and future
//! per-layer golden-diff harnesses: run an op on GPU, run the same
//! op here on CPU, compare outputs.
//!
//! All functions operate on flat `&[f32]` slices (convert bf16→f32
//! before calling). Shape conventions match the ferrite globals
//! layout.
//!
//! Originally `ferrite-solver/src/cpu_golden.rs`; moved here as
//! part of the Step G legacy-delete (the solver crate and its
//! `ferrite_macros::forward!{}` consumer are gone).

/// RMS normalization: output = (x / rms(x)) * weight
///
/// - `input`: [hidden_dim] — one row of activations
/// - `weight`: [hidden_dim] — per-element scale
/// - `output`: [hidden_dim] — written by this function
/// - `eps`: RMS norm epsilon (typically 1e-5)
pub fn rmsnorm(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    let n = input.len();
    assert_eq!(n, weight.len());
    assert_eq!(n, output.len());

    let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    for i in 0..n {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

/// Matrix multiply: output = input @ weight^T
///
/// - `input`: [m, k] row-major
/// - `weight`: [n, k] row-major (transposed in multiply)
/// - `output`: [m, n] row-major
///
/// Serial — this is reference code for correctness checks, not a
/// performance path. Parallelise only if a golden-harness regime
/// starts showing measurable wall-clock pain.
pub fn gemm(input: &[f32], weight: &[f32], output: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(input.len(), m * k);
    assert_eq!(weight.len(), n * k);
    assert_eq!(output.len(), m * n);

    for i in 0..m {
        let in_row = &input[i * k..(i + 1) * k];
        for j in 0..n {
            let w_row = &weight[j * k..(j + 1) * k];
            let mut sum = 0.0_f32;
            for l in 0..k {
                sum += in_row[l] * w_row[l];
            }
            output[i * n + j] = sum;
        }
    }
}

/// Matrix multiply with residual add: output = (input @ weight^T) + residual
///
/// - `input`: [m, k]
/// - `weight`: [n, k]
/// - `residual`: [m, n]
/// - `output`: [m, n] — written as matmul result + residual
pub fn gemm_add(
    input: &[f32],
    weight: &[f32],
    residual: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    gemm(input, weight, output, m, k, n);
    for i in 0..m * n {
        output[i] += residual[i];
    }
}

/// SiLU activation: output = x * sigmoid(x)
pub fn silu(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    for i in 0..input.len() {
        let x = input[i];
        output[i] = x / (1.0 + (-x).exp());
    }
}

/// Elementwise multiply: output = a * b
pub fn mul(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() {
        output[i] = a[i] * b[i];
    }
}

/// Rotary position embedding (RoPE).
///
/// Applies rotation to query/key vectors using cos/sin tables.
/// - `qk`: [seq_len, dim] — query or key vectors (modified in-place via output)
/// - `cos_table`: [max_pos, head_dim/2] or [max_pos, head_dim]
/// - `sin_table`: same shape as cos_table
/// - `positions`: [seq_len] — position indices
/// - `head_dim`: dimension of each attention head
/// - `output`: [seq_len, dim] — rotated vectors
pub fn rope(
    qk: &[f32],
    cos_table: &[f32],
    sin_table: &[f32],
    positions: &[i32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    output: &mut [f32],
) {
    let half_dim = head_dim / 2;
    let dim = num_heads * head_dim;
    assert_eq!(qk.len(), seq_len * dim);
    assert_eq!(output.len(), seq_len * dim);

    output.copy_from_slice(qk);

    for s in 0..seq_len {
        let pos = positions[s] as usize;
        for h in 0..num_heads {
            let base = s * dim + h * head_dim;
            for d in 0..half_dim {
                let cos_val = cos_table[pos * head_dim + d];
                let sin_val = sin_table[pos * head_dim + d];
                let x0 = qk[base + d];
                let x1 = qk[base + half_dim + d];
                output[base + d] = x0 * cos_val - x1 * sin_val;
                output[base + half_dim + d] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

/// Scaled dot-product attention (decode, single query token).
///
/// - `q`: [1, num_heads * head_dim]
/// - `k_cache`: [seq_len, num_kv_heads, head_dim]
/// - `v_cache`: [seq_len, num_kv_heads, head_dim]
/// - `output`: [1, num_heads * head_dim]
/// - GQA: num_heads / num_kv_heads queries share one KV head
pub fn attention_decode(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    output: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn_scale: f32,
) {
    let gqa_ratio = num_heads / num_kv_heads;

    for h in 0..num_heads {
        let kv_h = h / gqa_ratio;
        let q_offset = h * head_dim;

        // Compute attention scores
        let mut scores = vec![0.0_f32; seq_len];
        for s in 0..seq_len {
            let k_offset = s * num_kv_heads * head_dim + kv_h * head_dim;
            let mut dot = 0.0_f32;
            for d in 0..head_dim {
                dot += q[q_offset + d] * k_cache[k_offset + d];
            }
            scores[s] = dot * attn_scale;
        }

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0_f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        for s in &mut scores {
            *s /= sum_exp;
        }

        // Weighted sum of values
        for d in 0..head_dim {
            let mut val = 0.0_f32;
            for s in 0..seq_len {
                let v_offset = s * num_kv_heads * head_dim + kv_h * head_dim + d;
                val += scores[s] * v_cache[v_offset];
            }
            output[q_offset + d] = val;
        }
    }
}

/// Scaled dot-product attention (prefill, variable-length sequences).
///
/// - `q`: [total_tokens, num_heads * head_dim]
/// - `k`: [total_tokens, num_kv_heads * head_dim]
/// - `v`: [total_tokens, num_kv_heads * head_dim]
/// - `output`: [total_tokens, num_heads * head_dim]
/// - `seq_starts`: [num_seqs + 1] — cumulative token offsets
/// - Causal masking applied
pub fn attention_prefill(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_starts: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn_scale: f32,
) {
    let gqa_ratio = num_heads / num_kv_heads;
    let num_seqs = seq_starts.len() - 1;

    for seq in 0..num_seqs {
        let start = seq_starts[seq];
        let end = seq_starts[seq + 1];
        let seq_len = end - start;

        for h in 0..num_heads {
            let kv_h = h / gqa_ratio;

            for qi in 0..seq_len {
                let q_offset = (start + qi) * num_heads * head_dim + h * head_dim;

                // Compute scores (causal: only attend to positions <= qi)
                let attend_len = qi + 1;
                let mut scores = vec![0.0_f32; attend_len];
                for ki in 0..attend_len {
                    let k_offset = (start + ki) * num_kv_heads * head_dim + kv_h * head_dim;
                    let mut dot = 0.0_f32;
                    for d in 0..head_dim {
                        dot += q[q_offset + d] * k[k_offset + d];
                    }
                    scores[ki] = dot * attn_scale;
                }

                // Softmax
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0_f32;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    sum_exp += *s;
                }
                for s in &mut scores {
                    *s /= sum_exp;
                }

                // Weighted sum
                for d in 0..head_dim {
                    let mut val = 0.0_f32;
                    for ki in 0..attend_len {
                        let v_offset = (start + ki) * num_kv_heads * head_dim + kv_h * head_dim + d;
                        val += scores[ki] * v[v_offset];
                    }
                    output[q_offset + d] = val;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_identity() {
        // With weight=1 and uniform input, output = input / rms(input)
        let input = vec![1.0_f32; 128];
        let weight = vec![1.0_f32; 128];
        let mut output = vec![0.0_f32; 128];
        rmsnorm(&input, &weight, &mut output, 1e-5);
        // rms(1,1,...,1) = 1, so output ≈ 1.0
        for &v in &output {
            assert!((v - 1.0).abs() < 1e-3, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn test_rmsnorm_scaling() {
        let input = vec![2.0_f32; 64];
        let weight = vec![0.5_f32; 64];
        let mut output = vec![0.0_f32; 64];
        rmsnorm(&input, &weight, &mut output, 1e-5);
        // rms([2,...,2]) = 2, so normalized = 1.0, * 0.5 = 0.5
        for &v in &output {
            assert!((v - 0.5).abs() < 1e-3, "expected ~0.5, got {v}");
        }
    }

    #[test]
    fn test_gemm_simple() {
        // [1,2] x [2,2]^T = [1*1+2*0, 1*0+2*1] = [1, 2]
        let input = vec![1.0, 2.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let mut output = vec![0.0; 2];
        gemm(&input, &weight, &mut output, 1, 2, 2);
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_silu() {
        let input = vec![0.0, 1.0, -1.0];
        let mut output = vec![0.0; 3];
        silu(&input, &mut output);
        assert!((output[0] - 0.0).abs() < 1e-5); // silu(0) = 0
        assert!((output[1] - 0.7310586).abs() < 1e-4); // silu(1) ≈ 0.731
        assert!((output[2] - (-0.2689414)).abs() < 1e-4); // silu(-1) ≈ -0.269
    }

    #[test]
    fn test_attention_decode_single_head() {
        let head_dim = 4;
        let seq_len = 2;
        let num_heads = 1;
        let num_kv_heads = 1;

        let q = vec![1.0, 0.0, 0.0, 0.0]; // query
        let k_cache = vec![
            1.0, 0.0, 0.0, 0.0, // k[0]
            0.0, 1.0, 0.0, 0.0, // k[1]
        ];
        let v_cache = vec![
            1.0, 2.0, 3.0, 4.0, // v[0]
            5.0, 6.0, 7.0, 8.0, // v[1]
        ];
        let mut output = vec![0.0; 4];
        let scale = 1.0 / (head_dim as f32).sqrt();

        attention_decode(
            &q,
            &k_cache,
            &v_cache,
            &mut output,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        );

        // q·k[0] = 1*scale, q·k[1] = 0 → softmax ≈ [0.622, 0.378]
        // output ≈ 0.622 * v[0] + 0.378 * v[1]
        assert!(output[0] > 0.5 && output[0] < 4.0); // sanity check
    }
}
