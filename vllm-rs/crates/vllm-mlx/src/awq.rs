// SPDX-License-Identifier: Apache-2.0
//! AWQ dequantization for MLX.
//!
//! Converts AWQ-format weights (qweight/qzeros/scales) into standard
//! float weight tensors that can be used by the regular (non-quantized) MLX
//! model factories. Dequantization is done once at load time.
//!
//! AWQ packs along columns with an interleave order:
//! stored order in i32: `[col0, col2, col4, col6, col1, col3, col5, col7]`
//! To recover logical order, apply reverse-interleave `[0,4,1,5,2,6,3,7]`.

use std::collections::HashMap;

use mlx_rs::Array;
use tracing::info;

use vllm_model::layers::awq::AwqConfig;

/// AWQ reverse-interleave mapping for INT4 (pack_factor=8).
const AWQ_REVERSE_INTERLEAVE: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

/// Dequantize AWQ weights in-place, replacing `*.qweight`/`*.qzeros`/`*.scales`
/// with a single `*.weight` tensor per linear layer.
///
/// Returns the modified weight map with standard tensor names.
pub fn dequantize_awq_weights(
    mut weights: HashMap<String, Array>,
    config: &AwqConfig,
) -> Result<HashMap<String, Array>, Box<dyn std::error::Error + Send + Sync>> {
    let pack_factor = 32 / config.bits;
    let mask = (1u32 << config.bits) - 1;

    // Collect all AWQ layer prefixes (e.g. "model.layers.0.self_attn.q_proj").
    let prefixes: Vec<String> = weights
        .keys()
        .filter(|k| k.ends_with(".qweight"))
        .map(|k| k.strip_suffix(".qweight").unwrap().to_string())
        .collect();

    info!(
        "AWQ: dequantizing {} linear layers (bits={}, group_size={})",
        prefixes.len(),
        config.bits,
        config.group_size
    );

    for prefix in &prefixes {
        let qweight = weights
            .remove(&format!("{prefix}.qweight"))
            .ok_or_else(|| format!("missing {prefix}.qweight"))?;
        let qzeros = weights
            .remove(&format!("{prefix}.qzeros"))
            .ok_or_else(|| format!("missing {prefix}.qzeros"))?;
        let scales = weights
            .remove(&format!("{prefix}.scales"))
            .ok_or_else(|| format!("missing {prefix}.scales"))?;

        let dequantized = dequantize_layer(&qweight, &qzeros, &scales, config, pack_factor, mask)?;

        weights.insert(format!("{prefix}.weight"), dequantized);
    }

    // Evaluate all dequantized weights to materialize them.
    let deq_arrays: Vec<&Array> = weights.values().collect();
    mlx_rs::transforms::eval(deq_arrays.into_iter())?;

    info!("AWQ: dequantization complete");
    Ok(weights)
}

/// Dequantize a single AWQ linear layer.
///
/// Returns weight tensor of shape `[out_features, in_features]` (transposed
/// from AWQ's `[in_features, out_features]` to match nn::Linear convention).
fn dequantize_layer(
    qweight: &Array,
    qzeros: &Array,
    scales: &Array,
    config: &AwqConfig,
    pack_factor: usize,
    mask: u32,
) -> Result<Array, Box<dyn std::error::Error + Send + Sync>> {
    let scales_dtype = scales.dtype();

    // qweight: [in_features, out_features/pack_factor]
    let qw_shape = qweight.shape();
    let in_features = qw_shape[0] as usize;
    let packed_cols = qw_shape[1] as usize;
    let out_features = packed_cols * pack_factor;

    // Unpack qweight on CPU: [in, out/pack] i32 → [in, out] f32 with AWQ reverse-interleave
    let unpacked_weight = unpack_cols_awq_cpu(
        qweight,
        config.bits,
        pack_factor,
        mask,
        in_features,
        out_features,
    )?;
    let unpacked_weight = unpacked_weight.as_dtype(scales_dtype)?;

    // Unpack qzeros: [num_groups, out/pack] → [num_groups, out_features]
    let qz_shape = qzeros.shape();
    let num_groups = qz_shape[0] as usize;
    let zeros = unpack_cols_awq_cpu(
        qzeros,
        config.bits,
        pack_factor,
        mask,
        num_groups,
        out_features,
    )?;
    let zeros = zeros.as_dtype(scales_dtype)?;

    // Build group indices and gather scales/zeros per row.
    let group_size = if num_groups > 0 {
        in_features.div_ceil(num_groups)
    } else {
        in_features
    };

    // AWQ has no g_idx — always use sequential group mapping.
    let indices: Vec<u32> = (0..in_features).map(|i| (i / group_size) as u32).collect();
    let idx = Array::from_slice(&indices, &[in_features as i32]);
    let scales_per_row = scales.take_axis(&idx, 0)?;
    let zeros_per_row = zeros.take_axis(&idx, 0)?;

    // Dequantize: weight = scales * (unpacked - zeros)
    // Result: [in_features, out_features]
    let diff = unpacked_weight.subtract(&zeros_per_row)?;
    let dequantized = diff.multiply(&scales_per_row)?;

    // Transpose to [out_features, in_features] to match nn::Linear convention.
    let dequantized = dequantized.t();

    Ok(dequantized)
}

/// Unpack columns with AWQ interleave: [rows, packed_cols] i32 → [rows, out_cols] f32
#[allow(clippy::needless_range_loop)]
fn unpack_cols_awq_cpu(
    packed: &Array,
    bits: usize,
    pack_factor: usize,
    mask: u32,
    rows: usize,
    out_cols: usize,
) -> Result<Array, Box<dyn std::error::Error + Send + Sync>> {
    let packed_cols = packed.shape()[1] as usize;

    mlx_rs::transforms::eval(std::iter::once(packed))?;
    let packed_i32 = packed.as_dtype(mlx_rs::Dtype::Int32)?;
    mlx_rs::transforms::eval(std::iter::once(&packed_i32))?;
    let data: Vec<i32> = packed_i32.as_slice::<i32>().to_vec();

    let actual_cols = (packed_cols * pack_factor).min(out_cols);
    let mut unpacked = vec![0.0f32; rows * actual_cols];

    for row in 0..rows {
        for pc in 0..packed_cols {
            let pval = data[row * packed_cols + pc] as u32;
            for j in 0..pack_factor {
                // AWQ reverse-interleave: nibble j maps to logical offset.
                let logical_offset = if pack_factor == 8 {
                    AWQ_REVERSE_INTERLEAVE[j]
                } else {
                    j
                };
                let col = pc * pack_factor + logical_offset;
                if col >= actual_cols {
                    continue;
                }
                let val = (pval >> (j * bits)) & mask;
                unpacked[row * actual_cols + col] = val as f32;
            }
        }
    }

    Ok(Array::from_slice(
        &unpacked,
        &[rows as i32, actual_cols as i32],
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Dtype;

    /// Pack 8 INT4 values into one i32 using AWQ interleave order.
    ///
    /// AWQ stores nibble j from logical column ORDER[j],
    /// where ORDER = [0, 2, 4, 6, 1, 3, 5, 7].
    fn pack_awq_i32(vals: &[u32; 8]) -> i32 {
        let order_map: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
        let mut packed: u32 = 0;
        for (j, &src_col) in order_map.iter().enumerate() {
            packed |= (vals[src_col] & 0xF) << (j * 4);
        }
        packed as i32
    }

    #[test]
    fn test_unpack_cols_awq_cpu_basic() {
        // Pack values 0..7 in AWQ interleave order → unpack to [1, 8] with correct logical order
        let packed = pack_awq_i32(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let qweight = Array::from_slice(&[packed], &[1, 1]); // [1 row, 1 packed_col]

        let result = unpack_cols_awq_cpu(&qweight, 4, 8, 0xF, 1, 8).unwrap();
        mlx_rs::transforms::eval(std::iter::once(&result)).unwrap();

        assert_eq!(result.shape(), &[1, 8]);
        let vals: Vec<f32> = result
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        for i in 0..8 {
            assert!(
                (vals[i] - i as f32).abs() < 0.01,
                "expected {i}, got {} at position {i}",
                vals[i]
            );
        }
    }

    #[test]
    fn test_dequantize_layer_zeros_cancel() {
        // All values = 8, zeros = 8, scales = 2.0 → dequantized = 2*(8-8) = 0
        let packed = pack_awq_i32(&[8; 8]);
        let zero_packed = pack_awq_i32(&[8; 8]);

        let qweight = Array::from_slice(&[packed], &[1, 1]); // [1 in_feat, 1 packed_col]
        let qzeros = Array::from_slice(&[zero_packed], &[1, 1]); // [1 group, 1 packed_col]
        let scales = Array::from_slice(&[2.0f32; 8], &[1, 8]); // [1 group, 8 out_feat]

        let config = AwqConfig {
            bits: 4,
            group_size: 1,
            zero_point: true,
        };

        let result = dequantize_layer(&qweight, &qzeros, &scales, &config, 8, 0xF).unwrap();
        mlx_rs::transforms::eval(std::iter::once(&result)).unwrap();

        // Transposed: [8, 1]
        assert_eq!(result.shape(), &[8, 1]);
        let vals: Vec<f32> = result
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        for (i, v) in vals.iter().enumerate() {
            assert!(v.abs() < 0.01, "expected 0.0 at {i}, got {v}");
        }
    }

    #[test]
    fn test_dequantize_awq_weights_end_to_end() {
        // Single AWQ layer: 1 input feature, 8 output features
        // Pack 1..8 in AWQ order, zeros=0, scales=1.0
        let packed = pack_awq_i32(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let zero_packed = pack_awq_i32(&[0; 8]);

        let qweight = Array::from_slice(&[packed], &[1, 1]); // [1, 1]
        let qzeros = Array::from_slice(&[zero_packed], &[1, 1]); // [1, 1]
        let scales = Array::from_slice(&[1.0f32; 8], &[1, 8]); // [1, 8]

        let mut weights = HashMap::new();
        weights.insert("layer.qweight".to_string(), qweight);
        weights.insert("layer.qzeros".to_string(), qzeros);
        weights.insert("layer.scales".to_string(), scales);

        let config = AwqConfig {
            bits: 4,
            group_size: 1,
            zero_point: true,
        };

        let result = dequantize_awq_weights(weights, &config).unwrap();

        assert!(result.contains_key("layer.weight"));
        assert!(!result.contains_key("layer.qweight"));

        let w = result.get("layer.weight").unwrap();
        // Transposed: [8, 1]
        assert_eq!(w.shape(), &[8, 1]);

        let vals: Vec<f32> = w
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let sum: f32 = vals.iter().sum();
        let expected: f32 = (1..=8).map(|i| i as f32).sum();
        assert!(
            (sum - expected).abs() < 0.1,
            "expected sum {expected}, got {sum}"
        );
    }

    #[test]
    fn test_dequantize_awq_preserves_non_quantized() {
        let norm_weight = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let packed = pack_awq_i32(&[0; 8]);
        let zero_packed = pack_awq_i32(&[0; 8]);

        let qweight = Array::from_slice(&[packed], &[1, 1]);
        let qzeros = Array::from_slice(&[zero_packed], &[1, 1]);
        let scales = Array::from_slice(&[1.0f32; 8], &[1, 8]);

        let mut weights = HashMap::new();
        weights.insert("model.norm.weight".to_string(), norm_weight);
        weights.insert("layer.qweight".to_string(), qweight);
        weights.insert("layer.qzeros".to_string(), qzeros);
        weights.insert("layer.scales".to_string(), scales);

        let config = AwqConfig {
            bits: 4,
            group_size: 1,
            zero_point: true,
        };

        let result = dequantize_awq_weights(weights, &config).unwrap();
        assert!(result.contains_key("model.norm.weight"));
        assert!(result.contains_key("layer.weight"));
        assert_eq!(result.len(), 2);
    }
}
