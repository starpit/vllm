// SPDX-License-Identifier: Apache-2.0
//! Minimal GGUF format parser from `&[u8]`.
//!
//! No-std friendly, WASM compatible — no filesystem access.
//! Parses the GGUF v3 binary format: header, metadata, tensor info, data section.

use std::collections::HashMap;

/// GGUF tensor element types (subset we care about).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufDType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    BF16 = 30,
}

impl GgufDType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Block size (number of elements per quantization block).
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
        }
    }

    /// Bytes per block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18, // 2B scale + 16B nibbles (32 elements)
            Self::Q4_1 => 20, // 2B scale + 2B min + 16B nibbles
            Self::Q5_0 => 22, // 2B scale + 4B high bits + 16B low nibbles
            Self::Q5_1 => 24,
            Self::Q8_0 => 34, // 2B scale + 32B int8s
            Self::Q8_1 => 40,
            Self::Q2K => 256,
            Self::Q3K => 256,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
        }
    }
}

/// A GGUF metadata value.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

/// Info about a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: GgufDType,
    /// Offset from the start of the data section.
    pub offset: usize,
}

impl TensorInfo {
    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }

    /// Total size in bytes.
    pub fn size_bytes(&self) -> usize {
        let numel = self.numel();
        let bs = self.dtype.block_size();
        let num_blocks = numel / bs;
        num_blocks * self.dtype.block_bytes()
    }
}

/// A cursor for reading little-endian binary data from a byte slice.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.data.len() {
            return Err(format!(
                "unexpected EOF at offset {} (need {n} bytes, have {})",
                self.pos,
                self.remaining()
            ));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, String> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, String> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i8(&mut self) -> Result<i8, String> {
        Ok(self.read_u8()? as i8)
    }

    fn read_i16(&mut self) -> Result<i16, String> {
        Ok(self.read_u16()? as i16)
    }

    fn read_i32(&mut self) -> Result<i32, String> {
        Ok(self.read_u32()? as i32)
    }

    fn read_i64(&mut self) -> Result<i64, String> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_bool(&mut self) -> Result<bool, String> {
        Ok(self.read_u8()? != 0)
    }

    fn read_string(&mut self) -> Result<String, String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| format!("invalid UTF-8 string: {e}"))
    }

    fn read_value(&mut self, vtype: u32) -> Result<GgufValue, String> {
        match vtype {
            0 => Ok(GgufValue::U8(self.read_u8()?)),
            1 => Ok(GgufValue::I8(self.read_i8()?)),
            2 => Ok(GgufValue::U16(self.read_u16()?)),
            3 => Ok(GgufValue::I16(self.read_i16()?)),
            4 => Ok(GgufValue::U32(self.read_u32()?)),
            5 => Ok(GgufValue::I32(self.read_i32()?)),
            6 => Ok(GgufValue::F32(self.read_f32()?)),
            7 => Ok(GgufValue::Bool(self.read_bool()?)),
            8 => Ok(GgufValue::String(self.read_string()?)),
            9 => {
                // Array
                let elem_type = self.read_u32()?;
                let len = self.read_u64()? as usize;
                let mut arr = Vec::with_capacity(len.min(1024));
                for _ in 0..len {
                    arr.push(self.read_value(elem_type)?);
                }
                Ok(GgufValue::Array(arr))
            }
            10 => Ok(GgufValue::U64(self.read_u64()?)),
            11 => Ok(GgufValue::I64(self.read_i64()?)),
            12 => Ok(GgufValue::F64(self.read_f64()?)),
            other => Err(format!("unknown GGUF value type: {other}")),
        }
    }
}

/// Parsed GGUF file.
pub struct GgufReader<'a> {
    data: &'a [u8],
    metadata: HashMap<String, GgufValue>,
    tensors: HashMap<String, TensorInfo>,
    data_offset: usize,
}

impl<'a> GgufReader<'a> {
    /// Parse a GGUF file from raw bytes.
    pub fn parse(data: &'a [u8]) -> Result<Self, String> {
        let mut cursor = Cursor::new(data);

        // Header
        let magic = cursor.read_u32()?;
        if magic != 0x46554747 {
            // "GGUF" in little-endian
            return Err(format!(
                "invalid GGUF magic: 0x{magic:08x} (expected 0x46554747)"
            ));
        }
        let version = cursor.read_u32()?;
        if !(2..=3).contains(&version) {
            return Err(format!(
                "unsupported GGUF version: {version} (expected 2 or 3)"
            ));
        }
        let tensor_count = cursor.read_u64()? as usize;
        let metadata_count = cursor.read_u64()? as usize;

        // Metadata
        let mut metadata = HashMap::with_capacity(metadata_count);
        for _ in 0..metadata_count {
            let key = cursor.read_string()?;
            let vtype = cursor.read_u32()?;
            let value = cursor.read_value(vtype)?;
            metadata.insert(key, value);
        }

        // Tensor info
        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = cursor.read_string()?;
            let ndims = cursor.read_u32()? as usize;
            let mut shape = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                shape.push(cursor.read_u64()? as usize);
            }
            // GGUF stores shape in reverse order (innermost first)
            // We want [rows, cols] like the rest of our code
            shape.reverse();
            let dtype_raw = cursor.read_u32()?;
            let dtype = GgufDType::from_u32(dtype_raw)
                .ok_or_else(|| format!("unknown tensor dtype {dtype_raw} for {name}"))?;
            let offset = cursor.read_u64()? as usize;
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    shape,
                    dtype,
                    offset,
                },
            );
        }

        // Data section starts at the next alignment boundary (typically 32 bytes)
        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::U32(a)) => *a as usize,
            Some(GgufValue::U64(a)) => *a as usize,
            _ => 32,
        };
        let data_offset = cursor.pos.div_ceil(alignment) * alignment;

        Ok(Self {
            data,
            metadata,
            tensors,
            data_offset,
        })
    }

    /// Get the raw tensor bytes for a named tensor.
    pub fn tensor_data(&self, name: &str) -> Result<&'a [u8], String> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor not found: {name}"))?;
        let start = self.data_offset + info.offset;
        let size = info.size_bytes();
        if start + size > self.data.len() {
            return Err(format!(
                "tensor {name} data out of bounds: {start}+{size} > {}",
                self.data.len()
            ));
        }
        Ok(&self.data[start..start + size])
    }

    /// Get tensor info by name.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    // -- Metadata accessors --

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key)? {
            GgufValue::U32(v) => Some(*v),
            GgufValue::I32(v) => Some(*v as u32),
            GgufValue::U64(v) => Some(*v as u32),
            GgufValue::I64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key)? {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key)? {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Extract model config from GGUF metadata.
    /// Supports both `llama.*` and `qwen2.*` key prefixes.
    pub fn model_config(&self) -> Result<crate::model::ModelConfig, String> {
        // Try llama.* keys first, then qwen2.* as fallback
        let get = |key: &str| -> Option<u32> {
            self.get_u32(key)
                .or_else(|| self.get_u32(&key.replace("llama.", "qwen2.")))
        };
        let get_f = |key: &str| -> Option<f32> {
            self.get_f32(key)
                .or_else(|| self.get_f32(&key.replace("llama.", "qwen2.")))
        };

        let hidden_size = get("llama.embedding_length")
            .ok_or("missing embedding_length (tried llama.* and qwen2.*)")?
            as usize;
        let num_hidden_layers = get("llama.block_count").ok_or("missing block_count")? as usize;
        let num_attention_heads =
            get("llama.attention.head_count").ok_or("missing attention.head_count")? as usize;
        let num_key_value_heads =
            get("llama.attention.head_count_kv").unwrap_or(num_attention_heads as u32) as usize;
        let intermediate_size =
            get("llama.feed_forward_length").ok_or("missing feed_forward_length")? as usize;
        let vocab_size = get("llama.vocab_size")
            .or_else(|| {
                self.tensor_info("token_embd.weight")
                    .map(|t| t.shape[0] as u32)
            })
            .ok_or("missing vocab_size (no vocab_size metadata or token_embd.weight)")?
            as usize;
        let max_position_embeddings = get("llama.context_length").unwrap_or(2048) as usize;
        let rms_norm_eps = get_f("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f64;
        let rope_theta = get_f("llama.rope.freq_base").unwrap_or(10000.0) as f64;

        Ok(crate::model::ModelConfig {
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            num_hidden_layers,
            intermediate_size,
            vocab_size,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
        })
    }
}

/// Dequantize a Q4_0 tensor to f32.
/// Q4_0 block = 18 bytes: 2B f16 scale + 16B nibbles for 32 elements.
/// Nibble layout: byte j contains elem j (low nibble) and elem j+16 (high nibble).
/// Dequant: (nibble - 8) * scale.
pub fn dequantize_q4_0_to_f32(data: &[u8], numel: usize) -> Vec<f32> {
    let num_blocks = numel / 32;
    let mut out = vec![0.0f32; numel];
    for b in 0..num_blocks {
        let block = &data[b * 18..];
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        for j in 0..16 {
            let byte = block[2 + j];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            out[b * 32 + j] = lo as f32 * scale;
            out[b * 32 + j + 16] = hi as f32 * scale;
        }
    }
    out
}

/// Dequantize a Q4_1 tensor to f32.
/// Q4_1 block = 20 bytes: 2B f16 scale + 2B f16 min + 16B nibbles for 32 elements.
/// Dequant: nibble * scale + min.
pub fn dequantize_q4_1_to_f32(data: &[u8], numel: usize) -> Vec<f32> {
    let num_blocks = numel / 32;
    let mut out = vec![0.0f32; numel];
    for b in 0..num_blocks {
        let block = &data[b * 20..];
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let min = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        for j in 0..16 {
            let byte = block[4 + j];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            out[b * 32 + j] = lo * scale + min;
            out[b * 32 + j + 16] = hi * scale + min;
        }
    }
    out
}

/// Dequantize a Q6_K tensor to f32.
/// Q6_K super-block = 210 bytes, covers 256 elements.
/// Layout: ql[128] + qh[64] + scales[16] + d(f16)
/// Follows llama.cpp `dequantize_row_q6_K` exactly.
pub fn dequantize_q6k_to_f32(data: &[u8], numel: usize) -> Vec<f32> {
    let num_blocks = numel / 256;
    let mut out = vec![0.0f32; numel];
    for b in 0..num_blocks {
        let block = &data[b * 210..];
        let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();

        let dst = &mut out[b * 256..b * 256 + 256];

        // Process two halves of 128 elements (ql += 64, qh += 32, sc += 8 per half)
        for h in 0..2usize {
            let ql = &block[h * 64..];
            let qh = &block[128 + h * 32..];
            let sc = &block[192 + h * 8..];
            let y_off = h * 128;

            for l in 0..32usize {
                let is = l / 16;

                let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;

                dst[y_off + l] = d * sc[is] as i8 as f32 * q1 as f32;
                dst[y_off + l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                dst[y_off + l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                dst[y_off + l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
            }
        }
    }
    out
}

/// Dequantize a Q8_0 tensor to f32.
/// Q8_0 block = 34 bytes: 2B f16 scale + 32B int8 values for 32 elements.
pub fn dequantize_q8_0_to_f32(data: &[u8], numel: usize) -> Vec<f32> {
    let num_blocks = numel / 32;
    let mut out = vec![0.0f32; numel];
    for b in 0..num_blocks {
        let block = &data[b * 34..];
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        for j in 0..32 {
            out[b * 32 + j] = block[2 + j] as i8 as f32 * scale;
        }
    }
    out
}

/// Dequantize f16 tensor bytes to f32.
pub fn dequantize_f16_to_f32(data: &[u8], numel: usize) -> Vec<f32> {
    data.chunks_exact(2)
        .take(numel)
        .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

/// Dequantize bf16 tensor bytes to f32.
pub fn dequantize_bf16_to_f32(data: &[u8], numel: usize) -> Vec<f32> {
    data.chunks_exact(2)
        .take(numel)
        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

/// Dequantize f32 tensor bytes to f32 (just a memcpy-reinterpret).
pub fn dequantize_f32_to_f32(data: &[u8], numel: usize) -> Vec<f32> {
    bytemuck::cast_slice::<u8, f32>(&data[..numel * 4]).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_q4_0() {
        // One block: scale = 1.0 (f16), nibbles all = 8 (which means value 0)
        let mut block = vec![0u8; 18];
        // f16 for 1.0 = 0x3C00
        block[0] = 0x00;
        block[1] = 0x3C;
        // All nibbles = 0x88 → lo=8, hi=8 → both dequant to (8-8)*1.0 = 0.0
        for i in 0..16 {
            block[2 + i] = 0x88;
        }
        let result = dequantize_q4_0_to_f32(&block, 32);
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert!((v - 0.0).abs() < 1e-6, "expected 0.0, got {v}");
        }

        // Test with non-trivial values: scale=2.0, first byte = 0x9A → lo=10-8=2, hi=9-8=1
        let mut block2 = vec![0u8; 18];
        // f16 for 2.0 = 0x4000
        block2[0] = 0x00;
        block2[1] = 0x40;
        block2[2] = 0x9A; // lo nibble = A=10, hi nibble = 9
        for i in 1..16 {
            block2[2 + i] = 0x88; // zeros
        }
        let result2 = dequantize_q4_0_to_f32(&block2, 32);
        assert!(
            (result2[0] - 4.0).abs() < 1e-3,
            "elem 0: got {}",
            result2[0]
        ); // (10-8)*2 = 4
        assert!(
            (result2[16] - 2.0).abs() < 1e-3,
            "elem 16: got {}",
            result2[16]
        ); // (9-8)*2 = 2
    }

    #[test]
    fn test_gguf_dtype_sizes() {
        assert_eq!(GgufDType::Q4_0.block_size(), 32);
        assert_eq!(GgufDType::Q4_0.block_bytes(), 18);
        assert_eq!(GgufDType::F16.block_size(), 1);
        assert_eq!(GgufDType::F16.block_bytes(), 2);
    }
}
