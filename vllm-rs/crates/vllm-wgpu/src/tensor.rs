// SPDX-License-Identifier: Apache-2.0
//! `WgpuTensor` — a GPU-resident tensor backed by a `wgpu::Buffer`.

use std::sync::Arc;

use crate::WgpuError;
use crate::device::WgpuDevice;

/// Data type for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgpuDType {
    F32,
    U32,
    /// Packed f16×2 — each u32 holds two f16 values via `unpack2x16float`.
    /// Numel counts the *logical* f16 element count (must be even).
    F16Packed,
    /// Q4_0 quantized blocks padded to 20 bytes (5 × u32) for GPU alignment.
    /// Layout per block: u32[0] = f16 scale in low 16 bits (high 16 = 0),
    /// u32[1..5] = 32 nibble values reordered for sequential access.
    /// Buffer is indexed as `blocks[(kg * N + col) * 5 + word]` where
    /// kg = K-group index (0..K/32), col = output column (0..N).
    /// `shape` stores `[K, N]` (the logical transposed weight dimensions).
    Q4_0Packed,
}

impl WgpuDType {
    /// Size in bytes of one logical element.
    /// For Q4_0Packed this returns 0 — use `WgpuTensor::size_bytes()` instead,
    /// which computes from the block structure.
    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 | Self::U32 => 4,
            Self::F16Packed => 2,
            Self::Q4_0Packed => 0, // not meaningful per-element; use tensor-level size
        }
    }
}

/// A tensor stored as a `wgpu::Buffer` on the GPU.
#[derive(Clone)]
pub struct WgpuTensor {
    pub(crate) buffer: Arc<wgpu::Buffer>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: WgpuDType,
    pub(crate) device: WgpuDevice,
}

impl WgpuTensor {
    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        if self.dtype == WgpuDType::Q4_0Packed {
            // shape = [K, N], blocks = (K/32) * N * 20 bytes
            let k = self.shape[0];
            let n = self.shape[1];
            (k / 32) * n * 20
        } else {
            self.numel() * self.dtype.size_bytes()
        }
    }

    /// Shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Data type.
    pub fn dtype(&self) -> WgpuDType {
        self.dtype
    }

    /// Create a tensor from f32 data on the host.
    pub fn from_f32(device: &WgpuDevice, shape: &[usize], data: &[f32]) -> Result<Self, WgpuError> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(WgpuError::ShapeMismatch {
                expected: numel,
                got: data.len(),
            });
        }

        let buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tensor_f32"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        Ok(Self {
            buffer: Arc::new(buffer),
            shape: shape.to_vec(),
            dtype: WgpuDType::F32,
            device: device.clone(),
        })
    }

    /// Create a tensor from u32 data on the host.
    pub fn from_u32(device: &WgpuDevice, shape: &[usize], data: &[u32]) -> Result<Self, WgpuError> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(WgpuError::ShapeMismatch {
                expected: numel,
                got: data.len(),
            });
        }

        let buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tensor_u32"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        Ok(Self {
            buffer: Arc::new(buffer),
            shape: shape.to_vec(),
            dtype: WgpuDType::U32,
            device: device.clone(),
        })
    }

    /// Create an uninitialized (zero) tensor, reusing a pooled buffer if available.
    /// The underlying buffer may be larger than the logical tensor size due to
    /// bucket-aligned allocation, but the tensor metadata tracks the logical shape.
    pub fn zeros(device: &WgpuDevice, shape: &[usize], dtype: WgpuDType) -> Self {
        let numel: usize = shape.iter().product();
        let size = (numel * dtype.size_bytes()) as u64;

        let buffer = {
            let mut pool = device.buffer_pool.lock().unwrap();
            if let Some(buf) = pool.get(size) {
                buf
            } else {
                drop(pool);
                // Allocate at bucket-aligned size so the buffer can be reused
                // by future requests in the same bucket.
                let alloc_size = crate::device::BufferPool::alloc_size(size);
                Arc::new(device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("tensor_pooled"),
                    size: alloc_size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }))
            }
        };

        Self {
            buffer,
            shape: shape.to_vec(),
            dtype,
            device: device.clone(),
        }
    }

    /// Read tensor data back to host as f32.
    pub async fn to_f32(&self) -> Result<Vec<f32>, WgpuError> {
        // Flush any batched commands before reading back.
        self.device.flush();

        let size = self.size_bytes() as u64;
        let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .device
            .create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, size);
        self.device.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.device.poll(wgpu::Maintain::Wait);
        rx.await
            .map_err(|_| WgpuError::BufferMap)?
            .map_err(|_| WgpuError::BufferMap)?;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(result)
    }

    /// Reshape the tensor (no data copy, just changes metadata).
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, WgpuError> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(WgpuError::ShapeMismatch {
                expected: self.numel(),
                got: new_numel,
            });
        }
        Ok(Self {
            buffer: self.buffer.clone(),
            shape: new_shape.to_vec(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
}

impl Drop for WgpuTensor {
    fn drop(&mut self) {
        // Return buffer to pool if we hold the only reference.
        // Weight tensors and KV caches have multiple references and won't be recycled.
        if Arc::strong_count(&self.buffer) == 1 {
            let buf = self.buffer.clone();
            self.device.buffer_pool.lock().unwrap().put(buf);
        }
    }
}

impl WgpuTensor {
    /// Create a transposed copy of a 2D tensor. [N, K] → [K, N].
    /// Done on CPU at load time (not performance-critical).
    pub fn transpose_2d_cpu(&self, data: &[f32]) -> Result<Self, WgpuError> {
        if self.shape.len() != 2 {
            return Err(WgpuError::InvalidShape("transpose_2d requires 2D".into()));
        }
        let n = self.shape[0];
        let k = self.shape[1];
        let mut transposed = vec![0.0f32; n * k];
        for i in 0..n {
            for j in 0..k {
                transposed[j * n + i] = data[i * k + j];
            }
        }
        Self::from_f32(&self.device, &[k, n], &transposed)
    }
}

impl WgpuTensor {
    /// Create an f16-packed tensor from f32 data.
    /// Converts each pair of f32 values to f16 and packs them into one u32
    /// via the same encoding as WGSL `unpack2x16float`.
    /// `data.len()` must be even.
    pub fn from_f32_as_f16_packed(
        device: &WgpuDevice,
        shape: &[usize],
        data: &[f32],
    ) -> Result<Self, WgpuError> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(WgpuError::ShapeMismatch {
                expected: numel,
                got: data.len(),
            });
        }
        if !numel.is_multiple_of(2) {
            return Err(WgpuError::InvalidShape(
                "f16 packed requires even element count".into(),
            ));
        }

        // Pack pairs of f16 into u32
        let packed: Vec<u32> = data
            .chunks_exact(2)
            .map(|pair| {
                let a = half::f16::from_f32(pair[0]).to_bits() as u32;
                let b = half::f16::from_f32(pair[1]).to_bits() as u32;
                a | (b << 16)
            })
            .collect();

        let buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tensor_f16"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        Ok(Self {
            buffer: Arc::new(buffer),
            shape: shape.to_vec(),
            dtype: WgpuDType::F16Packed,
            device: device.clone(),
        })
    }

    /// Transpose a 2D tensor and store as f16-packed along the first (K) dimension.
    /// Input data: [N, K] row-major → Output: [K, N] with f16 pairs packed along K.
    /// The buffer layout is `array<u32>` with `(K/2) * N` elements where:
    ///   `packed[k_pair * N + col] = pack_f16(transposed[2*k_pair, col], transposed[2*k_pair+1, col])`
    /// This gives the matvec_t_f16 shader coalesced reads (adjacent threads read adjacent u32s).
    /// K must be even. Done on CPU at load time.
    pub fn transpose_2d_cpu_f16(&self, data: &[f32]) -> Result<Self, WgpuError> {
        if self.shape.len() != 2 {
            return Err(WgpuError::InvalidShape("transpose_2d requires 2D".into()));
        }
        let n = self.shape[0]; // out_features
        let k = self.shape[1]; // in_features
        if !k.is_multiple_of(2) {
            return Err(WgpuError::InvalidShape(
                "f16 packed transpose requires even K".into(),
            ));
        }

        // First transpose [N, K] → [K, N]
        let mut transposed = vec![0.0f32; n * k];
        for i in 0..n {
            for j in 0..k {
                transposed[j * n + i] = data[i * k + j];
            }
        }

        // Pack along K dimension: pairs of consecutive K rows at each col
        let k_half = k / 2;
        let mut packed = Vec::with_capacity(k_half * n);
        for kp in 0..k_half {
            for col in 0..n {
                let a = half::f16::from_f32(transposed[2 * kp * n + col]).to_bits() as u32;
                let b = half::f16::from_f32(transposed[(2 * kp + 1) * n + col]).to_bits() as u32;
                packed.push(a | (b << 16));
            }
        }

        let buffer = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tensor_f16_t"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        Ok(Self {
            buffer: Arc::new(buffer),
            shape: vec![k, n],
            dtype: WgpuDType::F16Packed,
            device: self.device.clone(),
        })
    }
}

impl WgpuTensor {
    /// Create a Q4_0 packed, transposed tensor from raw GGUF Q4_0 bytes.
    ///
    /// Input: raw Q4_0 blocks in row-major `[N, K]` order (N rows of K elements).
    /// Each row has K/32 blocks of 18 bytes each.
    ///
    /// Output: GPU buffer in transposed `[K/32, N]` block order, each block padded
    /// from 18 to 20 bytes (5 u32s), with nibbles reordered for sequential access.
    ///
    /// GPU layout: `blocks[(kg * N + col) * 5 + word]`
    ///   - word 0: f16 scale in low 16 bits
    ///   - words 1-4: 32 nibble values, reordered so element `e` is at
    ///     nibble position `e` (low nibble of byte e/2 for even e, high for odd).
    pub fn from_q4_0_transposed(
        device: &WgpuDevice,
        n: usize,
        k: usize,
        raw_bytes: &[u8],
    ) -> Result<Self, WgpuError> {
        if !k.is_multiple_of(32) {
            return Err(WgpuError::InvalidShape(
                "Q4_0 requires K divisible by 32".into(),
            ));
        }
        let blocks_per_row = k / 32;
        let expected_bytes = n * blocks_per_row * 18;
        if raw_bytes.len() != expected_bytes {
            return Err(WgpuError::ShapeMismatch {
                expected: expected_bytes,
                got: raw_bytes.len(),
            });
        }

        let total_blocks = (k / 32) * n;
        let mut packed = vec![0u32; total_blocks * 5];

        for kg in 0..blocks_per_row {
            for col in 0..n {
                // Source block: row=col, block=kg (row-major)
                let src_off = (col * blocks_per_row + kg) * 18;
                let src = &raw_bytes[src_off..src_off + 18];

                // Destination: transposed order
                let dst_base = (kg * n + col) * 5;

                // Word 0: scale (f16 in low 16 bits)
                let scale_bits = u16::from_le_bytes([src[0], src[1]]) as u32;
                packed[dst_base] = scale_bits;

                // Reorder nibbles: GGUF Q4_0 byte j has elem j (low) and elem j+16 (high).
                // We want sequential: elem 0..31 in sequential nibble positions.
                // Output: byte e/2 contains elem e in its low nibble (even e) or high nibble (odd e).
                // So output u32 w contains elements 4w..4w+3 as nibbles [lo0|hi0|lo1|hi1] per byte.
                let nibbles = &src[2..18]; // 16 bytes of nibble data

                // Unpack all 32 elements
                let mut elems = [0u8; 32];
                for j in 0..16 {
                    elems[j] = nibbles[j] & 0x0F;
                    elems[j + 16] = (nibbles[j] >> 4) & 0x0F;
                }

                // Repack into 4 u32s with sequential nibble order
                for w in 0..4u32 {
                    let base = (w * 8) as usize;
                    let mut word = 0u32;
                    for b in 0..4 {
                        let lo = elems[base + b * 2] as u32;
                        let hi = elems[base + b * 2 + 1] as u32;
                        word |= (lo | (hi << 4)) << (b * 8);
                    }
                    packed[dst_base + 1 + w as usize] = word;
                }
            }
        }

        let buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tensor_q4_0"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        Ok(Self {
            buffer: Arc::new(buffer),
            shape: vec![k, n],
            dtype: WgpuDType::Q4_0Packed,
            device: device.clone(),
        })
    }
}

/// Extension trait for `wgpu::Device` to create buffers with data.
use wgpu::util::DeviceExt;
