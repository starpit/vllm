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
