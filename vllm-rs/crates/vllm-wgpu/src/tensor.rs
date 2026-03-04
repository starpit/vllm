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
}

impl WgpuDType {
    /// Size in bytes of one element.
    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 | Self::U32 => 4,
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
        self.numel() * self.dtype.size_bytes()
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

    /// Create an uninitialized (zero) tensor.
    pub fn zeros(device: &WgpuDevice, shape: &[usize], dtype: WgpuDType) -> Self {
        let numel: usize = shape.iter().product();
        let size = (numel * dtype.size_bytes()) as u64;

        let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tensor_zeros"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer: Arc::new(buffer),
            shape: shape.to_vec(),
            dtype,
            device: device.clone(),
        }
    }

    /// Read tensor data back to host as f32.
    pub async fn to_f32(&self) -> Result<Vec<f32>, WgpuError> {
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

/// Extension trait for `wgpu::Device` to create buffers with data.
use wgpu::util::DeviceExt;
