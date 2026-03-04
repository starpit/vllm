// SPDX-License-Identifier: Apache-2.0
//! GPU compute operations dispatched via WGSL shaders.

use crate::WgpuError;
use crate::device::WgpuDevice;
use crate::tensor::{WgpuDType, WgpuTensor};

use wgpu::util::DeviceExt;

fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

/// Create a uniform buffer from a `[u32; 4]` params block.
fn params_buffer(device: &WgpuDevice, data: [u32; 4]) -> wgpu::Buffer {
    device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::UNIFORM,
        })
}

/// Run a compute shader with the given bind group entries.
fn dispatch(
    device: &WgpuDevice,
    shader_src: &str,
    entries: &[wgpu::BindGroupEntry],
    workgroups: [u32; 3],
) {
    let module = device
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

    // Use layout: None so wgpu auto-derives bind group layout from the shader.
    let pipeline = device
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries,
    });

    let mut encoder = device.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
    }
    device.queue.submit(Some(encoder.finish()));
}

// ---------------------------------------------------------------------------
// Public operations
// ---------------------------------------------------------------------------

/// Matrix multiplication: C = A * B.
/// A: [M, K], B: [K, N] → C: [M, N]
pub fn matmul(a: &WgpuTensor, b: &WgpuTensor) -> Result<WgpuTensor, WgpuError> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(WgpuError::InvalidShape("matmul requires 2D tensors".into()));
    }
    let m = a.shape[0] as u32;
    let k = a.shape[1] as u32;
    let n = b.shape[1] as u32;
    if a.shape[1] != b.shape[0] {
        return Err(WgpuError::InvalidShape(
            "matmul inner dimensions mismatch".into(),
        ));
    }

    let output = WgpuTensor::zeros(&a.device, &[m as usize, n as usize], WgpuDType::F32);
    let params = params_buffer(&a.device, [m, k, n, 0]);

    dispatch(
        &a.device,
        include_str!("shaders/matmul.wgsl"),
        &[
            bge(0, &a.buffer),
            bge(1, &b.buffer),
            bge(2, &output.buffer),
            bge(3, &params),
        ],
        [div_ceil(m, 16), div_ceil(n, 16), 1],
    );

    Ok(output)
}

/// Matrix multiplication with B transposed: C = A * B^T.
/// A: [M, K], B: [N, K] → C: [M, N]
/// This matches how Linear layers store weights: W is [out_features, in_features].
pub fn matmul_t(a: &WgpuTensor, b: &WgpuTensor) -> Result<WgpuTensor, WgpuError> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(WgpuError::InvalidShape(
            "matmul_t requires 2D tensors".into(),
        ));
    }
    let m = a.shape[0] as u32;
    let k = a.shape[1] as u32;
    let n = b.shape[0] as u32;
    if a.shape[1] != b.shape[1] {
        return Err(WgpuError::InvalidShape(format!(
            "matmul_t: A is [{m}, {k}], B is [{n}, {}] — inner dims must match",
            b.shape[1]
        )));
    }

    let output = WgpuTensor::zeros(&a.device, &[m as usize, n as usize], WgpuDType::F32);
    let params = params_buffer(&a.device, [m, k, n, 0]);

    dispatch(
        &a.device,
        include_str!("shaders/matmul_t.wgsl"),
        &[
            bge(0, &a.buffer),
            bge(1, &b.buffer),
            bge(2, &output.buffer),
            bge(3, &params),
        ],
        [div_ceil(m, 16), div_ceil(n, 16), 1],
    );

    Ok(output)
}

/// Linear layer: y = x @ W^T + bias.
/// x: [M, K], weight: [N, K], bias: Option<[N]> → y: [M, N]
pub fn linear(
    x: &WgpuTensor,
    weight: &WgpuTensor,
    bias: Option<&WgpuTensor>,
) -> Result<WgpuTensor, WgpuError> {
    let mut out = matmul_t(x, weight)?;
    if let Some(b) = bias {
        out = add(&out, &b.reshape(&out.shape)?)?;
    }
    Ok(out)
}

/// Element-wise addition: c = a + b.
pub fn add(a: &WgpuTensor, b: &WgpuTensor) -> Result<WgpuTensor, WgpuError> {
    if a.numel() != b.numel() {
        return Err(WgpuError::ShapeMismatch {
            expected: a.numel(),
            got: b.numel(),
        });
    }
    let total = a.numel() as u32;
    let output = WgpuTensor::zeros(&a.device, &a.shape, WgpuDType::F32);
    let params = params_buffer(&a.device, [total, 0, 0, 0]);

    dispatch(
        &a.device,
        include_str!("shaders/add.wgsl"),
        &[
            bge(0, &a.buffer),
            bge(1, &b.buffer),
            bge(2, &output.buffer),
            bge(3, &params),
        ],
        [div_ceil(total, 256), 1, 1],
    );

    Ok(output)
}

/// RMS normalization.
/// input: [N, D], weight: [D] → output: [N, D]
pub fn rms_norm(
    input: &WgpuTensor,
    weight: &WgpuTensor,
    eps: f32,
) -> Result<WgpuTensor, WgpuError> {
    if input.shape.len() != 2 {
        return Err(WgpuError::InvalidShape("rms_norm requires 2D input".into()));
    }
    let n = input.shape[0] as u32;
    let d = input.shape[1] as u32;

    let output = WgpuTensor::zeros(&input.device, &input.shape, WgpuDType::F32);
    let params = params_buffer(&input.device, [n, d, eps.to_bits(), 0]);

    dispatch(
        &input.device,
        include_str!("shaders/rms_norm.wgsl"),
        &[
            bge(0, &input.buffer),
            bge(1, &weight.buffer),
            bge(2, &output.buffer),
            bge(3, &params),
        ],
        [div_ceil(n, 256), 1, 1],
    );

    Ok(output)
}

/// Fused SiLU(gate) * up.
pub fn silu_mul(gate: &WgpuTensor, up: &WgpuTensor) -> Result<WgpuTensor, WgpuError> {
    if gate.numel() != up.numel() {
        return Err(WgpuError::ShapeMismatch {
            expected: gate.numel(),
            got: up.numel(),
        });
    }
    let total = gate.numel() as u32;
    let output = WgpuTensor::zeros(&gate.device, &gate.shape, WgpuDType::F32);
    let params = params_buffer(&gate.device, [total, 0, 0, 0]);

    dispatch(
        &gate.device,
        include_str!("shaders/silu_mul.wgsl"),
        &[
            bge(0, &gate.buffer),
            bge(1, &up.buffer),
            bge(2, &output.buffer),
            bge(3, &params),
        ],
        [div_ceil(total, 256), 1, 1],
    );

    Ok(output)
}

/// Row-wise softmax.
/// input: [N, D] → output: [N, D]
pub fn softmax(input: &WgpuTensor) -> Result<WgpuTensor, WgpuError> {
    if input.shape.len() != 2 {
        return Err(WgpuError::InvalidShape("softmax requires 2D input".into()));
    }
    let n = input.shape[0] as u32;
    let d = input.shape[1] as u32;

    let output = WgpuTensor::zeros(&input.device, &input.shape, WgpuDType::F32);
    let params = params_buffer(&input.device, [n, d, 0, 0]);

    dispatch(
        &input.device,
        include_str!("shaders/softmax.wgsl"),
        &[
            bge(0, &input.buffer),
            bge(1, &output.buffer),
            bge(2, &params),
        ],
        [div_ceil(n, 256), 1, 1],
    );

    Ok(output)
}

/// Embedding lookup.
/// table: [vocab_size, dim], indices: [N] → output: [N, dim]
pub fn embedding(table: &WgpuTensor, indices: &WgpuTensor) -> Result<WgpuTensor, WgpuError> {
    if table.shape.len() != 2 || indices.shape.len() != 1 {
        return Err(WgpuError::InvalidShape(
            "embedding: table must be 2D, indices 1D".into(),
        ));
    }
    let n = indices.shape[0] as u32;
    let dim = table.shape[1] as u32;

    let output = WgpuTensor::zeros(&table.device, &[n as usize, dim as usize], WgpuDType::F32);
    let params = params_buffer(&table.device, [n, dim, 0, 0]);

    dispatch(
        &table.device,
        include_str!("shaders/embedding.wgsl"),
        &[
            bge(0, &table.buffer),
            bge(1, &indices.buffer),
            bge(2, &output.buffer),
            bge(3, &params),
        ],
        [div_ceil(n * dim, 256), 1, 1],
    );

    Ok(output)
}

/// RoPE (Rotary Position Embedding) applied in-place to qk tensor.
/// qk: [N, num_heads, head_dim], positions: [N]
/// cos_cache/sin_cache: [max_seq_len, head_dim/2]
pub fn rope(
    qk: &WgpuTensor,
    cos_cache: &WgpuTensor,
    sin_cache: &WgpuTensor,
    positions: &WgpuTensor,
    num_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
) -> Result<WgpuTensor, WgpuError> {
    let n = qk.shape[0] as u32;
    let half_dim = head_dim / 2;
    let total_pairs = n * num_heads * half_dim;

    // Clone qk into output (RoPE modifies in-place)
    let output = WgpuTensor::zeros(&qk.device, &qk.shape, WgpuDType::F32);

    // Copy qk → output first
    let mut encoder = qk.device.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&qk.buffer, 0, &output.buffer, 0, qk.size_bytes() as u64);
    qk.device.queue.submit(Some(encoder.finish()));

    let params = params_buffer(&qk.device, [n, num_heads, head_dim, max_seq_len]);

    dispatch(
        &qk.device,
        include_str!("shaders/rope.wgsl"),
        &[
            bge(0, &output.buffer),
            bge(1, &cos_cache.buffer),
            bge(2, &sin_cache.buffer),
            bge(3, &positions.buffer),
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params.as_entire_binding(),
            },
        ],
        [div_ceil(total_pairs, 256), 1, 1],
    );

    Ok(output)
}

/// Helper to create a `BindGroupEntry` for a storage/uniform buffer.
fn bge(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
