// SPDX-License-Identifier: Apache-2.0
//! GPTQ dequantization for MLX.
//!
//! Converts GPTQ-format weights (qweight/qzeros/scales/g_idx) into standard
//! float weight tensors that can be used by the regular (non-quantized) MLX
//! model factories. Dequantization is done once at load time.

use std::collections::HashMap;

use mlx_rs::{Array, Dtype};
use tracing::info;

use vllm_model::layers::gptq::GptqConfig;

/// Dequantize GPTQ weights in-place, replacing `*.qweight`/`*.qzeros`/`*.scales`
/// with a single `*.weight` tensor per linear layer.
///
/// Returns the modified weight map with standard tensor names.
pub fn dequantize_gptq_weights(
    mut weights: HashMap<String, Array>,
    config: &GptqConfig,
) -> Result<HashMap<String, Array>, Box<dyn std::error::Error + Send + Sync>> {
    let pack_factor = 32 / config.bits;
    let mask = (1u32 << config.bits) - 1;

    // Collect all GPTQ layer prefixes (e.g. "model.layers.0.self_attn.q_proj").
    let prefixes: Vec<String> = weights
        .keys()
        .filter(|k| k.ends_with(".qweight"))
        .map(|k| k.strip_suffix(".qweight").unwrap().to_string())
        .collect();

    info!(
        "GPTQ: dequantizing {} linear layers (bits={}, group_size={})",
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
        let g_idx = weights.remove(&format!("{prefix}.g_idx"));

        let dequantized = dequantize_layer(
            &qweight,
            &qzeros,
            &scales,
            g_idx.as_ref(),
            config,
            pack_factor,
            mask,
        )?;

        weights.insert(format!("{prefix}.weight"), dequantized);
    }

    // Evaluate all dequantized weights to materialize them.
    let deq_arrays: Vec<&Array> = weights.values().collect();
    mlx_rs::transforms::eval(deq_arrays.into_iter())?;

    info!("GPTQ: dequantization complete");
    Ok(weights)
}

/// Dequantize a single GPTQ linear layer.
///
/// Returns weight tensor of shape `[out_features, in_features]` (transposed
/// from GPTQ's `[in_features, out_features]` to match nn::Linear convention).
fn dequantize_layer(
    qweight: &Array,
    qzeros: &Array,
    scales: &Array,
    g_idx: Option<&Array>,
    config: &GptqConfig,
    pack_factor: usize,
    mask: u32,
) -> Result<Array, Box<dyn std::error::Error + Send + Sync>> {
    let scales_dtype = scales.dtype();

    // qweight: [in_features/pack_factor, out_features]
    let qw_shape = qweight.shape();
    let packed_rows = qw_shape[0] as usize;
    let out_features = qw_shape[1] as usize;
    let in_features = packed_rows * pack_factor;

    // Unpack qweight on CPU: [in/pack, out] i32 → [in, out] f32
    let unpacked_weight = unpack_rows_cpu(
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
    let zeros = unpack_cols_cpu(
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

    let (scales_per_row, zeros_per_row) = if let Some(g_idx) = g_idx {
        let g_idx = g_idx.as_dtype(Dtype::Uint32)?;
        let sp = scales.take_axis(&g_idx, 0)?;
        let zp = zeros.take_axis(&g_idx, 0)?;
        (sp, zp)
    } else {
        let indices: Vec<u32> = (0..in_features).map(|i| (i / group_size) as u32).collect();
        let idx = Array::from_slice(&indices, &[in_features as i32]);
        let sp = scales.take_axis(&idx, 0)?;
        let zp = zeros.take_axis(&idx, 0)?;
        (sp, zp)
    };

    // Dequantize: weight = scales * (unpacked - zeros)
    // Result: [in_features, out_features]
    let diff = unpacked_weight.subtract(&zeros_per_row)?;
    let dequantized = diff.multiply(&scales_per_row)?;

    // Transpose to [out_features, in_features] to match nn::Linear convention.
    let dequantized = dequantized.t();

    Ok(dequantized)
}

/// Unpack rows: [packed_rows, cols] i32 → [out_rows, cols] f32
/// GPTQ qweight packs along dimension 0.
fn unpack_rows_cpu(
    packed: &Array,
    bits: usize,
    pack_factor: usize,
    mask: u32,
    out_rows: usize,
    cols: usize,
) -> Result<Array, Box<dyn std::error::Error + Send + Sync>> {
    let packed_rows = packed.shape()[0] as usize;

    // Read i32 data.
    mlx_rs::transforms::eval(std::iter::once(packed))?;
    let packed_i32 = packed.as_dtype(Dtype::Int32)?;
    mlx_rs::transforms::eval(std::iter::once(&packed_i32))?;
    let data: Vec<i32> = packed_i32.as_slice::<i32>().to_vec();

    let actual_rows = (packed_rows * pack_factor).min(out_rows);
    let mut unpacked = vec![0.0f32; actual_rows * cols];

    for pr in 0..packed_rows {
        for col in 0..cols {
            let pval = data[pr * cols + col] as u32;
            for j in 0..pack_factor {
                let row = pr * pack_factor + j;
                if row >= actual_rows {
                    break;
                }
                let val = (pval >> (j * bits)) & mask;
                unpacked[row * cols + col] = val as f32;
            }
        }
    }

    Ok(Array::from_slice(
        &unpacked,
        &[actual_rows as i32, cols as i32],
    ))
}

/// Unpack columns: [rows, packed_cols] i32 → [rows, out_cols] f32
/// GPTQ qzeros packs along dimension 1.
fn unpack_cols_cpu(
    packed: &Array,
    bits: usize,
    pack_factor: usize,
    mask: u32,
    rows: usize,
    out_cols: usize,
) -> Result<Array, Box<dyn std::error::Error + Send + Sync>> {
    let packed_cols = packed.shape()[1] as usize;

    mlx_rs::transforms::eval(std::iter::once(packed))?;
    let packed_i32 = packed.as_dtype(Dtype::Int32)?;
    mlx_rs::transforms::eval(std::iter::once(&packed_i32))?;
    let data: Vec<i32> = packed_i32.as_slice::<i32>().to_vec();

    let actual_cols = (packed_cols * pack_factor).min(out_cols);
    let mut unpacked = vec![0.0f32; rows * actual_cols];

    for row in 0..rows {
        for pc in 0..packed_cols {
            let pval = data[row * packed_cols + pc] as u32;
            for j in 0..pack_factor {
                let col = pc * pack_factor + j;
                if col >= actual_cols {
                    break;
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

    /// Pack 8 INT4 values into one i32 (GPTQ row packing order).
    fn pack_gptq_i32(vals: &[u32; 8]) -> i32 {
        let mut packed: u32 = 0;
        for (j, &v) in vals.iter().enumerate() {
            packed |= (v & 0xF) << (j * 4);
        }
        packed as i32
    }

    #[test]
    fn test_unpack_rows_cpu() {
        // Pack values 0..7 into one i32 → qweight [1, 1] → unpack to [8, 1]
        let packed_val = pack_gptq_i32(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let qweight = Array::from_slice(&[packed_val], &[1, 1]);

        let result = unpack_rows_cpu(&qweight, 4, 8, 0xF, 8, 1).unwrap();
        mlx_rs::transforms::eval(std::iter::once(&result)).unwrap();

        assert_eq!(result.shape(), &[8, 1]);
        let vals: Vec<f32> = result
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        for i in 0..8 {
            assert!(
                (vals[i] - i as f32).abs() < 0.01,
                "expected {i}, got {}",
                vals[i]
            );
        }
    }

    #[test]
    fn test_unpack_cols_cpu() {
        // Pack values 0..7 into one i32 → qzeros [1, 1] → unpack to [1, 8]
        let packed_val = pack_gptq_i32(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let qzeros = Array::from_slice(&[packed_val], &[1, 1]);

        let result = unpack_cols_cpu(&qzeros, 4, 8, 0xF, 1, 8).unwrap();
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
                "expected {i}, got {}",
                vals[i]
            );
        }
    }

    #[test]
    fn test_dequantize_layer_identity() {
        // qweight: all 8s (val=8), qzeros: all 8s → unpacked - zeros = 0
        // scales = 2.0 → dequantized = 0.0 for all elements
        let packed_val = pack_gptq_i32(&[8; 8]);
        let qweight = Array::from_slice(&[packed_val], &[1, 1]); // [1, 1] → 8 rows, 1 col
        let zero_packed = pack_gptq_i32(&[8; 8]);
        let qzeros = Array::from_slice(&[zero_packed], &[1, 1]); // [1, 1] → 1 group, 8 cols
        let scales = Array::from_slice(&[2.0f32], &[1, 1]); // [1, 1]

        let config = GptqConfig {
            bits: 4,
            group_size: 8,
            desc_act: false,
            sym: true,
        };

        let result = dequantize_layer(&qweight, &qzeros, &scales, None, &config, 8, 0xF).unwrap();
        mlx_rs::transforms::eval(std::iter::once(&result)).unwrap();

        // Output is transposed: [out_features=1, in_features=8]
        assert_eq!(result.shape(), &[1, 8]);
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
    fn test_dequantize_gptq_weights_end_to_end() {
        // Single GPTQ layer: 8 input features, 1 output feature
        // qweight [1, 1], qzeros [1, 1], scales [1, 1]
        // values = 1..8, zeros = 0, scale = 1.0 → sum = 1+2+...+8 = 36
        let packed_val = pack_gptq_i32(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let qweight = Array::from_slice(&[packed_val], &[1, 1]);
        let zero_packed = pack_gptq_i32(&[0; 8]);
        let qzeros = Array::from_slice(&[zero_packed], &[1, 1]);
        let scales = Array::from_slice(&[1.0f32], &[1, 1]);

        let mut weights = HashMap::new();
        weights.insert("layer.qweight".to_string(), qweight);
        weights.insert("layer.qzeros".to_string(), qzeros);
        weights.insert("layer.scales".to_string(), scales);

        let config = GptqConfig {
            bits: 4,
            group_size: 8,
            desc_act: false,
            sym: true,
        };

        let result = dequantize_gptq_weights(weights, &config).unwrap();

        assert!(result.contains_key("layer.weight"));
        assert!(!result.contains_key("layer.qweight"));

        let w = result.get("layer.weight").unwrap();
        // Transposed: [1, 8]
        assert_eq!(w.shape(), &[1, 8]);

        let vals: Vec<f32> = w
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let sum: f32 = vals.iter().sum();
        assert!((sum - 36.0).abs() < 0.1, "expected sum 36.0, got {sum}");
    }
}
