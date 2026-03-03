// SPDX-License-Identifier: Apache-2.0
//! BitsAndBytes dequantization for MLX (NF4/FP4 4-bit and INT8 8-bit).
//!
//! Converts BnB-format weights into standard float weight tensors that can be
//! used by the regular (non-quantized) MLX model factories. Dequantization is
//! done once at load time.
//!
//! NF4/FP4: packed uint8 + absmax → float via lookup table
//! INT8: int8 stored as uint8 + per-row SCB → float via `val * (SCB / 127)`
//!
//! Supports double quantization (absmax stored as uint8 with nested scales).

use std::collections::HashMap;

use mlx_rs::Array;
use serde::Deserialize;
use tracing::info;

use vllm_model::layers::bnb::{BnbNf4Config, BnbQuantType};

/// NF4 lookup table — 16 normal distribution quantiles.
#[allow(clippy::excessive_precision)]
const NF4_TABLE: [f32; 16] = [
    -1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.0910500, 0.0, 0.0795803,
    0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.5626170, 0.7229568, 1.0,
];

/// FP4 lookup table.
const FP4_TABLE: [f32; 16] = [
    0.0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0, -0.0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0,
];

/// Metadata parsed from the `bitsandbytes__nf4` JSON blob.
#[derive(Deserialize)]
struct BnbQuantState {
    shape: Vec<usize>,
    #[serde(default = "default_blocksize")]
    blocksize: usize,
    #[serde(default)]
    nested_blocksize: usize,
    #[serde(default)]
    nested_offset: f64,
}

fn default_blocksize() -> usize {
    64
}

/// Dequantize BnB NF4/FP4 weights in-place, replacing `*.weight` (packed uint8)
/// and `*.weight.absmax` with a single `*.weight` float tensor per linear layer.
///
/// Returns the modified weight map with standard tensor names.
pub fn dequantize_bnb_weights(
    mut weights: HashMap<String, Array>,
    config: &BnbNf4Config,
    _hf_config: &vllm_model::weight::HfModelConfig,
) -> Result<HashMap<String, Array>, Box<dyn std::error::Error + Send + Sync>> {
    let table = match config.quant_type {
        BnbQuantType::NF4 => &NF4_TABLE,
        BnbQuantType::FP4 => &FP4_TABLE,
    };

    // Collect NF4/FP4 layer prefixes (identified by "{prefix}.weight.absmax").
    let prefixes: Vec<String> = weights
        .keys()
        .filter(|k| k.ends_with(".weight.absmax"))
        .map(|k| k.strip_suffix(".weight.absmax").unwrap().to_string())
        .collect();

    // Collect INT8 layer prefixes (identified by "{prefix}.SCB").
    let int8_prefixes: Vec<String> = weights
        .keys()
        .filter(|k| k.ends_with(".SCB"))
        .map(|k| k.strip_suffix(".SCB").unwrap().to_string())
        .collect();

    info!(
        "BnB: dequantizing {} NF4/FP4 + {} INT8 linear layers",
        prefixes.len(),
        int8_prefixes.len(),
    );

    for prefix in &prefixes {
        let packed = weights
            .remove(&format!("{prefix}.weight"))
            .ok_or_else(|| format!("missing {prefix}.weight"))?;
        let absmax_raw = weights
            .remove(&format!("{prefix}.weight.absmax"))
            .ok_or_else(|| format!("missing {prefix}.weight.absmax"))?;

        // Parse shape from quant_state JSON blob.
        let quant_state_key_nf4 = format!("{prefix}.weight.quant_state.bitsandbytes__nf4");
        let quant_state_key_fp4 = format!("{prefix}.weight.quant_state.bitsandbytes__fp4");
        let quant_state = weights
            .remove(&quant_state_key_nf4)
            .or_else(|| weights.remove(&quant_state_key_fp4));

        let (out_features, in_features, blocksize, nested_blocksize, nested_offset) =
            if let Some(qs) = quant_state {
                mlx_rs::transforms::eval(std::iter::once(&qs))?;
                let qs_u8 = qs.as_dtype(mlx_rs::Dtype::Uint8)?;
                mlx_rs::transforms::eval(std::iter::once(&qs_u8))?;
                let bytes: Vec<u8> = qs_u8.as_slice::<u8>().to_vec();
                let meta: BnbQuantState = serde_json::from_slice(&bytes)
                    .map_err(|e| format!("failed to parse quant_state for {prefix}: {e}"))?;
                (
                    meta.shape[0],
                    meta.shape[1],
                    meta.blocksize,
                    meta.nested_blocksize,
                    meta.nested_offset,
                )
            } else {
                // Fallback: infer from packed size (less reliable).
                let total = packed
                    .shape()
                    .iter()
                    .map(|&d| d as usize)
                    .product::<usize>()
                    * 2;
                (total, 1, config.blocksize, 0, 0.0)
            };

        // Handle double quantization: absmax stored as uint8, needs dequant via
        // nested_absmax (f32 scales) and nested_quant_map (f32[256] codebook).
        let nested_absmax = weights.remove(&format!("{prefix}.weight.nested_absmax"));
        let nested_quant_map = weights.remove(&format!("{prefix}.weight.nested_quant_map"));

        let absmax_f32 = if let (Some(na), Some(nqm)) = (nested_absmax, nested_quant_map) {
            // Double quantization: absmax_raw is uint8 indices into nested_quant_map,
            // scaled by nested_absmax with nested_blocksize.
            dequantize_absmax(&absmax_raw, &na, &nqm, nested_blocksize, nested_offset)?
        } else {
            // Simple: absmax is already f32.
            let a = absmax_raw.as_dtype(mlx_rs::Dtype::Float32)?;
            mlx_rs::transforms::eval(std::iter::once(&a))?;
            a.as_slice::<f32>().to_vec()
        };

        // Remove remaining metadata tensors.
        weights.remove(&format!("{prefix}.weight.quant_map"));
        weights.remove(&format!("{prefix}.weight.blocksize"));
        weights.remove(&format!("{prefix}.weight.dtype"));
        weights.remove(&format!("{prefix}.weight.shape"));

        let dequantized = dequantize_layer(
            &packed,
            &absmax_f32,
            table,
            blocksize,
            out_features,
            in_features,
        )?;

        weights.insert(format!("{prefix}.weight"), dequantized);
    }

    // Dequantize INT8 layers.
    for prefix in &int8_prefixes {
        let weight_raw = weights
            .remove(&format!("{prefix}.weight"))
            .ok_or_else(|| format!("missing {prefix}.weight"))?;
        let scb = weights
            .remove(&format!("{prefix}.SCB"))
            .ok_or_else(|| format!("missing {prefix}.SCB"))?;

        // Parse shape from quant_state JSON if available.
        let quant_state_key = format!("{prefix}.weight.quant_state.bitsandbytes__nf4");
        let quant_state = weights.remove(&quant_state_key);

        let (out_features, in_features) = if let Some(qs) = quant_state {
            mlx_rs::transforms::eval(std::iter::once(&qs))?;
            let qs_u8 = qs.as_dtype(mlx_rs::Dtype::Uint8)?;
            mlx_rs::transforms::eval(std::iter::once(&qs_u8))?;
            let bytes: Vec<u8> = qs_u8.as_slice::<u8>().to_vec();
            if let Ok(meta) = serde_json::from_slice::<BnbQuantState>(&bytes) {
                (meta.shape[0], meta.shape[1])
            } else {
                // Infer from weight shape.
                let shape = weight_raw.shape();
                (shape[0] as usize, shape[1] as usize)
            }
        } else {
            // Infer from weight tensor shape directly.
            let shape = weight_raw.shape();
            (shape[0] as usize, shape[1] as usize)
        };

        // Remove any remaining metadata tensors.
        weights.remove(&format!("{prefix}.weight.quant_map"));
        weights.remove(&format!("{prefix}.weight.blocksize"));
        weights.remove(&format!("{prefix}.weight.dtype"));
        weights.remove(&format!("{prefix}.weight.shape"));

        let dequantized = dequantize_int8_layer(&weight_raw, &scb, out_features, in_features)?;
        weights.insert(format!("{prefix}.weight"), dequantized);
    }

    // Evaluate all dequantized weights to materialize them.
    let deq_arrays: Vec<&Array> = weights.values().collect();
    mlx_rs::transforms::eval(deq_arrays.into_iter())?;

    info!("BnB: dequantization complete");
    Ok(weights)
}

/// Dequantize double-quantized absmax values.
///
/// In double quantization, the per-block absmax is itself quantized:
/// - `absmax_raw`: uint8 indices into the nested codebook
/// - `nested_absmax`: f32 per-block scales for the nested quantization
/// - `nested_quant_map`: f32[256] codebook mapping uint8 → float
/// - `nested_blocksize`: block size for the nested quantization
/// - `nested_offset`: constant offset subtracted during nested quantization
fn dequantize_absmax(
    absmax_raw: &Array,
    nested_absmax: &Array,
    nested_quant_map: &Array,
    nested_blocksize: usize,
    nested_offset: f64,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    mlx_rs::transforms::eval([absmax_raw, nested_absmax, nested_quant_map].into_iter())?;

    let raw_u8 = absmax_raw.as_dtype(mlx_rs::Dtype::Uint8)?;
    mlx_rs::transforms::eval(std::iter::once(&raw_u8))?;
    let raw_data: Vec<u8> = raw_u8.as_slice::<u8>().to_vec();

    let na_f32 = nested_absmax.as_dtype(mlx_rs::Dtype::Float32)?;
    mlx_rs::transforms::eval(std::iter::once(&na_f32))?;
    let nested_scales: Vec<f32> = na_f32.as_slice::<f32>().to_vec();

    let nqm_f32 = nested_quant_map.as_dtype(mlx_rs::Dtype::Float32)?;
    mlx_rs::transforms::eval(std::iter::once(&nqm_f32))?;
    let codebook: Vec<f32> = nqm_f32.as_slice::<f32>().to_vec();

    let n = raw_data.len();
    let bs = if nested_blocksize > 0 {
        nested_blocksize
    } else {
        256
    };

    let mut result = vec![0.0f32; n];
    for i in 0..n {
        let code = raw_data[i] as usize;
        let block_idx = i / bs;
        let scale = if block_idx < nested_scales.len() {
            nested_scales[block_idx]
        } else {
            1.0
        };
        result[i] = scale * codebook[code] + nested_offset as f32;
    }

    Ok(result)
}

/// Dequantize a single BnB NF4/FP4 linear layer.
///
/// Returns weight tensor of shape `[out_features, in_features]` matching
/// nn::Linear convention.
fn dequantize_layer(
    packed: &Array,
    absmax_data: &[f32],
    table: &[f32; 16],
    blocksize: usize,
    out_features: usize,
    in_features: usize,
) -> Result<Array, Box<dyn std::error::Error + Send + Sync>> {
    // Read packed uint8 data.
    mlx_rs::transforms::eval(std::iter::once(packed))?;
    let packed_u8 = packed.as_dtype(mlx_rs::Dtype::Uint8)?;
    mlx_rs::transforms::eval(std::iter::once(&packed_u8))?;
    let packed_data: Vec<u8> = packed_u8.as_slice::<u8>().to_vec();

    let total = out_features * in_features;

    // Dequantize.
    let mut out = vec![0.0f32; total];
    for i in 0..total {
        let byte = packed_data[i / 2];
        let nibble = ((byte >> ((i % 2) * 4)) & 0xF) as usize;
        let block_idx = i / blocksize;
        out[i] = absmax_data[block_idx] * table[nibble];
    }

    Ok(Array::from_slice(
        &out,
        &[out_features as i32, in_features as i32],
    ))
}

/// Dequantize a single BnB INT8 linear layer.
///
/// INT8 weights are stored as uint8 (reinterpreted as signed int8), with
/// per-row absmax scales (SCB). Dequantization:
///   `weight[row][col] = int8_val * (SCB[row] / 127.0)`
///
/// Returns weight tensor of shape `[out_features, in_features]`.
fn dequantize_int8_layer(
    weight_raw: &Array,
    scb: &Array,
    out_features: usize,
    in_features: usize,
) -> Result<Array, Box<dyn std::error::Error + Send + Sync>> {
    mlx_rs::transforms::eval([weight_raw, scb].into_iter())?;

    let weight_u8 = weight_raw.as_dtype(mlx_rs::Dtype::Uint8)?;
    mlx_rs::transforms::eval(std::iter::once(&weight_u8))?;
    let weight_data: Vec<u8> = weight_u8.as_slice::<u8>().to_vec();

    let scb_f32 = scb.as_dtype(mlx_rs::Dtype::Float32)?;
    mlx_rs::transforms::eval(std::iter::once(&scb_f32))?;
    let scb_data: Vec<f32> = scb_f32.as_slice::<f32>().to_vec();

    let total = out_features * in_features;
    let mut out = vec![0.0f32; total];

    for (row, &scb_val) in scb_data.iter().enumerate().take(out_features) {
        let scale = scb_val / 127.0;
        for col in 0..in_features {
            let idx = row * in_features + col;
            let signed_val = weight_data[idx] as i8;
            out[idx] = signed_val as f32 * scale;
        }
    }

    Ok(Array::from_slice(
        &out,
        &[out_features as i32, in_features as i32],
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Dtype;

    /// Pack nibble values into uint8 bytes (2 nibbles per byte, low nibble first).
    fn pack_nibbles(nibbles: &[u8]) -> Vec<u8> {
        let num_bytes = (nibbles.len() + 1) / 2;
        let mut bytes = vec![0u8; num_bytes];
        for (i, &nib) in nibbles.iter().enumerate() {
            bytes[i / 2] |= (nib & 0xF) << ((i % 2) * 4);
        }
        bytes
    }

    /// Helper to build a minimal HfModelConfig with a given hidden_size.
    fn hf_config_with_hidden(hidden_size: usize) -> vllm_model::weight::HfModelConfig {
        serde_json::from_value(serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": hidden_size,
        }))
        .unwrap()
    }

    #[test]
    fn test_dequantize_layer_nf4_basic() {
        // 4 elements: nibble indices [0, 7, 8, 15]
        // NF4: [-1.0, 0.0, 0.0796, 1.0], absmax=2.0
        let nibbles = vec![0u8, 7, 8, 15];
        let packed_bytes = pack_nibbles(&nibbles);
        let packed = Array::from_slice(&packed_bytes, &[packed_bytes.len() as i32]);
        let absmax = vec![2.0f32];

        let result = dequantize_layer(&packed, &absmax, &NF4_TABLE, 64, 1, 4).unwrap();
        mlx_rs::transforms::eval(std::iter::once(&result)).unwrap();

        assert_eq!(result.shape(), &[1, 4]);
        let vals: Vec<f32> = result
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();

        assert!(
            (vals[0] - (-2.0)).abs() < 0.001,
            "expected -2.0, got {}",
            vals[0]
        );
        assert!(
            (vals[1] - 0.0).abs() < 0.001,
            "expected 0.0, got {}",
            vals[1]
        );
        assert!(
            (vals[2] - 0.15916).abs() < 0.001,
            "expected ~0.159, got {}",
            vals[2]
        );
        assert!(
            (vals[3] - 2.0).abs() < 0.001,
            "expected 2.0, got {}",
            vals[3]
        );
    }

    #[test]
    fn test_dequantize_layer_fp4() {
        let nibbles = vec![1u8, 1, 1, 1];
        let packed_bytes = pack_nibbles(&nibbles);
        let packed = Array::from_slice(&packed_bytes, &[packed_bytes.len() as i32]);
        let absmax = vec![4.0f32];

        let result = dequantize_layer(&packed, &absmax, &FP4_TABLE, 64, 1, 4).unwrap();
        mlx_rs::transforms::eval(std::iter::once(&result)).unwrap();

        let vals: Vec<f32> = result
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        for (i, v) in vals.iter().enumerate() {
            assert!((v - 0.25).abs() < 0.001, "expected 0.25 at {i}, got {v}");
        }
    }

    #[test]
    fn test_dequantize_layer_multiple_blocks() {
        let nibbles = vec![15u8; 4];
        let packed_bytes = pack_nibbles(&nibbles);
        let packed = Array::from_slice(&packed_bytes, &[packed_bytes.len() as i32]);
        let absmax = vec![2.0f32, 3.0];

        let result = dequantize_layer(&packed, &absmax, &NF4_TABLE, 2, 1, 4).unwrap();
        mlx_rs::transforms::eval(std::iter::once(&result)).unwrap();

        let vals: Vec<f32> = result
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert!((vals[0] - 2.0).abs() < 0.001);
        assert!((vals[1] - 2.0).abs() < 0.001);
        assert!((vals[2] - 3.0).abs() < 0.001);
        assert!((vals[3] - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_dequantize_layer_explicit_shape() {
        // 8 elements, explicit shape [2, 4]
        let nibbles = vec![15u8; 8];
        let packed_bytes = pack_nibbles(&nibbles);
        let packed = Array::from_slice(&packed_bytes, &[packed_bytes.len() as i32]);
        let absmax = vec![1.0f32];

        let result = dequantize_layer(&packed, &absmax, &NF4_TABLE, 64, 2, 4).unwrap();
        mlx_rs::transforms::eval(std::iter::once(&result)).unwrap();

        assert_eq!(result.shape(), &[2, 4]);
    }

    #[test]
    fn test_parse_quant_state_json() {
        let json = r#"{"quant_type": "nf4", "blocksize": 64, "dtype": "bfloat16", "shape": [2048, 8192], "nested_blocksize": 256, "nested_dtype": "float32", "nested_offset": 0.092}"#;
        let meta: BnbQuantState = serde_json::from_str(json).unwrap();
        assert_eq!(meta.shape, vec![2048, 8192]);
        assert_eq!(meta.blocksize, 64);
        assert_eq!(meta.nested_blocksize, 256);
        assert!((meta.nested_offset - 0.092).abs() < 0.001);
    }

    #[test]
    fn test_dequantize_absmax_double_quant() {
        // Simple double-quant test: 4 absmax values, codebook maps idx→float, scale=2.0
        let raw_data = vec![0u8, 1, 2, 3]; // indices into codebook
        let absmax_raw = Array::from_slice(&raw_data, &[4]);

        // Codebook: 256 entries, first 4 are [0.0, 0.5, 1.0, 1.5]
        let mut codebook_data = vec![0.0f32; 256];
        codebook_data[0] = 0.0;
        codebook_data[1] = 0.5;
        codebook_data[2] = 1.0;
        codebook_data[3] = 1.5;
        let nested_quant_map = Array::from_slice(&codebook_data, &[256]);

        // One block of 4, scale = 2.0
        let nested_absmax = Array::from_slice(&[2.0f32], &[1]);

        let result =
            dequantize_absmax(&absmax_raw, &nested_absmax, &nested_quant_map, 4, 0.1).unwrap();

        assert_eq!(result.len(), 4);
        // result[i] = scale * codebook[raw[i]] + offset
        assert!((result[0] - (2.0 * 0.0 + 0.1)).abs() < 0.001);
        assert!((result[1] - (2.0 * 0.5 + 0.1)).abs() < 0.001);
        assert!((result[2] - (2.0 * 1.0 + 0.1)).abs() < 0.001);
        assert!((result[3] - (2.0 * 1.5 + 0.1)).abs() < 0.001);
    }

    #[test]
    fn test_dequantize_bnb_weights_with_quant_state() {
        // End-to-end with quant_state JSON providing shape.
        let nibbles = vec![15u8; 8]; // 8 elements
        let packed_bytes = pack_nibbles(&nibbles);
        let packed = Array::from_slice(&packed_bytes, &[packed_bytes.len() as i32]);
        let absmax = Array::from_slice(&[5.0f32], &[1]);

        // quant_state JSON specifying [4, 2] shape
        let qs_json =
            br#"{"quant_type": "nf4", "blocksize": 64, "dtype": "bfloat16", "shape": [4, 2]}"#;
        let qs_array = Array::from_slice(qs_json.as_slice(), &[qs_json.len() as i32]);

        let mut weights = HashMap::new();
        weights.insert("layer.weight".to_string(), packed);
        weights.insert("layer.weight.absmax".to_string(), absmax);
        weights.insert(
            "layer.weight.quant_state.bitsandbytes__nf4".to_string(),
            qs_array,
        );

        let config = BnbNf4Config {
            quant_type: BnbQuantType::NF4,
            blocksize: 64,
            double_quant: false,
        };
        let hf_config = hf_config_with_hidden(2);

        let result = dequantize_bnb_weights(weights, &config, &hf_config).unwrap();
        let w = result.get("layer.weight").unwrap();

        // Shape from quant_state: [4, 2]
        assert_eq!(w.shape(), &[4, 2]);
    }

    #[test]
    fn test_dequantize_bnb_weights_preserves_non_quantized() {
        let norm_weight = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);

        let nibbles = vec![7u8; 4];
        let packed_bytes = pack_nibbles(&nibbles);
        let packed = Array::from_slice(&packed_bytes, &[packed_bytes.len() as i32]);
        let absmax = Array::from_slice(&[1.0f32], &[1]);

        let mut weights = HashMap::new();
        weights.insert("model.norm.weight".to_string(), norm_weight);
        weights.insert("layer.weight".to_string(), packed);
        weights.insert("layer.weight.absmax".to_string(), absmax);

        let config = BnbNf4Config::default();
        let hf_config = hf_config_with_hidden(4);

        let result = dequantize_bnb_weights(weights, &config, &hf_config).unwrap();

        assert!(result.contains_key("model.norm.weight"));
        assert!(result.contains_key("layer.weight"));
        assert_eq!(result.len(), 2);

        let norm = result.get("model.norm.weight").unwrap();
        let norm_vals: Vec<f32> = norm
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert_eq!(norm_vals, vec![1.0, 2.0, 3.0]);
    }

    // -----------------------------------------------------------------------
    // INT8 tests
    // -----------------------------------------------------------------------

    fn i8_to_u8_bytes(vals: &[i8]) -> Vec<u8> {
        vals.iter().map(|&v| v as u8).collect()
    }

    #[test]
    fn test_dequantize_int8_layer_basic() {
        // 1 row, 4 cols. INT8 values: [127, -127, 0, 64]
        // SCB = 2.54 → scale = 2.54/127 = 0.02
        let weight_bytes = i8_to_u8_bytes(&[127, -127, 0, 64]);
        let weight = Array::from_slice(&weight_bytes, &[1, 4]);
        let scb = Array::from_slice(&[2.54f32], &[1]);

        let result = dequantize_int8_layer(&weight, &scb, 1, 4).unwrap();
        mlx_rs::transforms::eval(std::iter::once(&result)).unwrap();

        assert_eq!(result.shape(), &[1, 4]);
        let vals: Vec<f32> = result
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();

        assert!(
            (vals[0] - 2.54).abs() < 0.01,
            "expected 2.54, got {}",
            vals[0]
        );
        assert!(
            (vals[1] - (-2.54)).abs() < 0.01,
            "expected -2.54, got {}",
            vals[1]
        );
        assert!(
            (vals[2] - 0.0).abs() < 0.01,
            "expected 0.0, got {}",
            vals[2]
        );
        assert!(
            (vals[3] - 1.28).abs() < 0.01,
            "expected 1.28, got {}",
            vals[3]
        );
    }

    #[test]
    fn test_dequantize_int8_layer_multiple_rows() {
        // 2 rows, 2 cols. Row 0: SCB=2.54, Row 1: SCB=1.27
        let weight_bytes = i8_to_u8_bytes(&[127, 127, 127, 127]);
        let weight = Array::from_slice(&weight_bytes, &[2, 2]);
        let scb = Array::from_slice(&[2.54f32, 1.27], &[2]);

        let result = dequantize_int8_layer(&weight, &scb, 2, 2).unwrap();
        mlx_rs::transforms::eval(std::iter::once(&result)).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        let vals: Vec<f32> = result
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();

        assert!((vals[0] - 2.54).abs() < 0.01);
        assert!((vals[1] - 2.54).abs() < 0.01);
        assert!((vals[2] - 1.27).abs() < 0.01);
        assert!((vals[3] - 1.27).abs() < 0.01);
    }

    #[test]
    fn test_dequantize_bnb_weights_int8() {
        // End-to-end: INT8 layer identified by .SCB
        let weight_bytes = i8_to_u8_bytes(&[127i8; 4]);
        let weight = Array::from_slice(&weight_bytes, &[2, 2]);
        let scb = Array::from_slice(&[2.54f32, 1.27], &[2]);

        let mut weights = HashMap::new();
        weights.insert("layer.weight".to_string(), weight);
        weights.insert("layer.SCB".to_string(), scb);

        let config = BnbNf4Config::default();
        let hf_config = hf_config_with_hidden(2);

        let result = dequantize_bnb_weights(weights, &config, &hf_config).unwrap();
        let w = result.get("layer.weight").unwrap();

        assert_eq!(w.shape(), &[2, 2]);
        let vals: Vec<f32> = w
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert!((vals[0] - 2.54).abs() < 0.01);
        assert!((vals[2] - 1.27).abs() < 0.01);
    }
}
