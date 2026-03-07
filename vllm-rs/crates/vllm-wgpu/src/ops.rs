// SPDX-License-Identifier: Apache-2.0
//! GPU compute operations dispatched via WGSL shaders.

use std::sync::Arc;

use crate::WgpuError;
use crate::device::WgpuDevice;
use crate::tensor::{WgpuDType, WgpuTensor};

use wgpu::util::DeviceExt;

fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

/// Get or create a cached uniform buffer from a `[u32; 4]` params block.
fn params_buffer(device: &WgpuDevice, data: [u32; 4]) -> Arc<wgpu::Buffer> {
    device
        .params_cache
        .lock()
        .unwrap()
        .cache4
        .entry(data)
        .or_insert_with(|| {
            Arc::new(
                device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("params"),
                        contents: bytemuck::cast_slice(&data),
                        usage: wgpu::BufferUsages::UNIFORM,
                    }),
            )
        })
        .clone()
}

/// Get or create a cached uniform buffer from a `[u32; 8]` params block.
fn params_buffer_8(device: &WgpuDevice, data: [u32; 8]) -> Arc<wgpu::Buffer> {
    device
        .params_cache
        .lock()
        .unwrap()
        .cache8
        .entry(data)
        .or_insert_with(|| {
            Arc::new(
                device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("params8"),
                        contents: bytemuck::cast_slice(&data),
                        usage: wgpu::BufferUsages::UNIFORM,
                    }),
            )
        })
        .clone()
}

/// Run a compute shader with the given bind group entries.
/// Uses cached pipelines and batched command encoding.
/// Consecutive dispatches are grouped into a single compute pass on flush.
fn dispatch(
    device: &WgpuDevice,
    shader_src: &'static str,
    entries: &[wgpu::BindGroupEntry],
    workgroups: [u32; 3],
) {
    let (pipeline, layout) = device.get_pipeline(shader_src);

    let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries,
    });

    device
        .batcher
        .lock()
        .unwrap()
        .push_dispatch(pipeline, bind_group, workgroups);
}

/// Dispatch variant for dynamically-generated shader strings.
fn dispatch_dynamic(
    device: &WgpuDevice,
    shader_src: &str,
    entries: &[wgpu::BindGroupEntry],
    workgroups: [u32; 3],
) {
    let (pipeline, layout) = device.get_pipeline_owned(shader_src);

    let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries,
    });

    device
        .batcher
        .lock()
        .unwrap()
        .push_dispatch(pipeline, bind_group, workgroups);
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
/// For M=1 (decode), uses a specialized matvec kernel for much better performance.
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

    if m == 1 {
        // Specialized matvec with row-major W (non-coalesced but still faster than tiled for M=1)
        let params = params_buffer(&a.device, [k, n, 0, 0]);
        dispatch(
            &a.device,
            include_str!("shaders/matvec_t_rowmajor.wgsl"),
            &[
                bge(0, &a.buffer),
                bge(1, &b.buffer),
                bge(2, &output.buffer),
                bge(3, &params),
            ],
            [div_ceil(n, 256), 1, 1],
        );
    } else {
        let params = params_buffer(&a.device, [m, k, n, 0]);
        // Use register-tiled kernel for M>1: 4×4 tile per thread, 64×64 per workgroup
        dispatch(
            &a.device,
            include_str!("shaders/matmul_t_tiled.wgsl"),
            &[
                bge(0, &a.buffer),
                bge(1, &b.buffer),
                bge(2, &output.buffer),
                bge(3, &params),
            ],
            [div_ceil(m, 64), div_ceil(n, 64), 1],
        );
    }

    Ok(output)
}

/// Matrix multiplication with B transposed, using pre-transposed B for M=1 matvec.
/// w: [N, K], w_t: [K, N] (pre-transposed copy of w).
/// For M=1, uses w_t for coalesced memory access. For M>1, uses w with tiled kernel.
pub fn matmul_t_with_transposed(
    a: &WgpuTensor,
    w: &WgpuTensor,
    w_t: &WgpuTensor,
) -> Result<WgpuTensor, WgpuError> {
    if a.shape.len() != 2 || w.shape.len() != 2 {
        return Err(WgpuError::InvalidShape(
            "matmul_t requires 2D tensors".into(),
        ));
    }
    let m = a.shape[0] as u32;
    let k = a.shape[1] as u32;
    let n = w.shape[0] as u32;
    if a.shape[1] != w.shape[1] {
        return Err(WgpuError::InvalidShape(format!(
            "matmul_t: A is [{m}, {k}], W is [{n}, {}] — inner dims must match",
            w.shape[1]
        )));
    }

    let output = WgpuTensor::zeros(&a.device, &[m as usize, n as usize], WgpuDType::F32);

    if m == 1 {
        // Use pre-transposed weights for coalesced reads
        let params = params_buffer(&a.device, [k, n, 0, 0]);
        let shader = match w_t.dtype {
            WgpuDType::Q4_0Packed => include_str!("shaders/matvec_t_q4_0.wgsl"),
            WgpuDType::F16Packed => include_str!("shaders/matvec_t_f16.wgsl"),
            _ => include_str!("shaders/matvec_t.wgsl"),
        };
        let wg_count = div_ceil(n, 256);
        dispatch(
            &a.device,
            shader,
            &[
                bge(0, &a.buffer),
                bge(1, &w_t.buffer),
                bge(2, &output.buffer),
                bge(3, &params),
            ],
            [wg_count, 1, 1],
        );
    } else {
        let params = params_buffer(&a.device, [m, k, n, 0]);
        // Use register-tiled kernel for M>1: 4×4 tile per thread, 64×64 per workgroup
        let shader = if w.dtype == WgpuDType::F16Packed {
            include_str!("shaders/matmul_t_tiled_f16.wgsl")
        } else {
            include_str!("shaders/matmul_t_tiled.wgsl")
        };
        dispatch(
            &a.device,
            shader,
            &[
                bge(0, &a.buffer),
                bge(1, &w.buffer),
                bge(2, &output.buffer),
                bge(3, &params),
            ],
            [div_ceil(m, 64), div_ceil(n, 64), 1],
        );
    }

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

/// Broadcast add: c = a + broadcast(b) where a is [M, D] and b is [D].
/// Adds b to every row of a.
pub fn add_row_broadcast(a: &WgpuTensor, b: &WgpuTensor) -> Result<WgpuTensor, WgpuError> {
    let total = a.numel() as u32;
    let d = b.numel() as u32;
    let output = WgpuTensor::zeros(&a.device, &a.shape, WgpuDType::F32);
    let params = params_buffer(&a.device, [total, d, 0, 0]);

    dispatch(
        &a.device,
        include_str!("shaders/add_broadcast.wgsl"),
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

    let output = WgpuTensor::zeros(&qk.device, &qk.shape, WgpuDType::F32);

    // Copy qk → output first (batched)
    qk.device.batcher.lock().unwrap().push_copy(
        qk.buffer.clone(),
        0,
        output.buffer.clone(),
        0,
        qk.size_bytes() as u64,
    );

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

/// Slice a tensor along the last dimension (GPU-side buffer copy).
/// Returns a new tensor with shape `[..leading_dims, length]`.
pub fn slice_last_dim(
    input: &WgpuTensor,
    offset: usize,
    length: usize,
) -> Result<WgpuTensor, WgpuError> {
    if input.shape.is_empty() {
        return Err(WgpuError::InvalidShape("cannot slice empty shape".into()));
    }
    let last_dim = *input.shape.last().unwrap();
    if offset + length > last_dim {
        return Err(WgpuError::InvalidShape(format!(
            "slice out of bounds: offset {offset} + length {length} > {last_dim}"
        )));
    }

    let leading: usize = input.shape[..input.shape.len() - 1].iter().product();
    let rows = if leading == 0 { 1 } else { leading };

    let mut out_shape = input.shape.clone();
    *out_shape.last_mut().unwrap() = length;
    let output = WgpuTensor::zeros(&input.device, &out_shape, input.dtype);

    if rows == 1 {
        // Single contiguous region — use buffer copy
        let byte_offset = (offset * input.dtype.size_bytes()) as u64;
        let byte_length = (length * input.dtype.size_bytes()) as u64;
        input.device.batcher.lock().unwrap().push_copy(
            input.buffer.clone(),
            byte_offset,
            output.buffer.clone(),
            0,
            byte_length,
        );
    } else {
        // Multiple rows — use a shader to gather strided slices
        let params = params_buffer_8(
            &input.device,
            [
                rows as u32,
                last_dim as u32,
                offset as u32,
                length as u32,
                0,
                0,
                0,
                0,
            ],
        );

        dispatch(
            &input.device,
            include_str!("shaders/slice.wgsl"),
            &[
                bge(0, &input.buffer),
                bge(1, &output.buffer),
                bge(2, &params),
            ],
            [div_ceil((rows * length) as u32, 256), 1, 1],
        );
    }

    Ok(output)
}

/// GPU-side single-query decode attention with KV cache.
/// q: [1, num_q_heads * head_dim]
/// k_cache, v_cache: [max_seq, num_kv_heads * head_dim] (GPU buffers)
/// seq_len: number of valid tokens in cache (including current)
/// Returns: [1, num_q_heads * head_dim]
pub fn attention(
    q: &WgpuTensor,
    k_cache: &WgpuTensor,
    v_cache: &WgpuTensor,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
) -> Result<WgpuTensor, WgpuError> {
    let q_size = (num_q_heads * head_dim) as usize;
    let output = WgpuTensor::zeros(&q.device, &[1, q_size], WgpuDType::F32);

    let scale_bits = (1.0f32 / (head_dim as f32).sqrt()).to_bits();
    let params = params_buffer_8(
        &q.device,
        [
            num_q_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            scale_bits,
            0,
            0,
            0,
        ],
    );

    dispatch(
        &q.device,
        include_str!("shaders/attention.wgsl"),
        &[
            bge(0, &q.buffer),
            bge(1, &k_cache.buffer),
            bge(2, &v_cache.buffer),
            bge(3, &output.buffer),
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params.as_entire_binding(),
            },
        ],
        [num_q_heads, 1, 1],
    );

    Ok(output)
}

/// Write a row into a cache buffer at a given position.
/// cache: [max_seq, dim], data: [1, dim], position: row index.
pub fn cache_write(
    cache: &WgpuTensor,
    data: &WgpuTensor,
    position: usize,
) -> Result<(), WgpuError> {
    let dim = cache.shape[1];
    let byte_offset = (position * dim * cache.dtype.size_bytes()) as u64;
    let byte_length = (dim * cache.dtype.size_bytes()) as u64;

    cache.device.batcher.lock().unwrap().push_copy(
        data.buffer.clone(),
        0,
        cache.buffer.clone(),
        byte_offset,
        byte_length,
    );
    Ok(())
}

/// Fused SiLU(gate) * up from a concatenated [gate | up] tensor.
/// input: [N, 2*half_size] → output: [N, half_size]
/// Replaces: slice_last_dim(gate) + slice_last_dim(up) + silu_mul
pub fn silu_mul_split(input: &WgpuTensor, half_size: usize) -> Result<WgpuTensor, WgpuError> {
    let rows = if input.shape.len() <= 1 {
        1
    } else {
        input.shape[..input.shape.len() - 1].iter().product()
    };
    let total = (rows * half_size) as u32;
    let stride = (2 * half_size) as u32;
    let output = WgpuTensor::zeros(&input.device, &[rows, half_size], WgpuDType::F32);
    let params = params_buffer(&input.device, [total, half_size as u32, stride, 0]);

    dispatch(
        &input.device,
        include_str!("shaders/silu_mul_split.wgsl"),
        &[
            bge(0, &input.buffer),
            bge(1, &output.buffer),
            bge(2, &params),
        ],
        [div_ceil(total, 256), 1, 1],
    );

    Ok(output)
}

/// Fused residual add + RMS normalization.
/// residual: [N, D], input: [N, D], weight: [D]
/// Returns (normed, residual_out) where residual_out = residual + input.
pub fn fused_add_rms_norm(
    residual: &WgpuTensor,
    input: &WgpuTensor,
    weight: &WgpuTensor,
    eps: f32,
) -> Result<(WgpuTensor, WgpuTensor), WgpuError> {
    let n = residual.shape[0] as u32;
    let d = residual.shape[1] as u32;
    let output = WgpuTensor::zeros(&residual.device, &residual.shape, WgpuDType::F32);
    let residual_out = WgpuTensor::zeros(&residual.device, &residual.shape, WgpuDType::F32);
    let params = params_buffer(&residual.device, [n, d, eps.to_bits(), 0]);

    dispatch(
        &residual.device,
        include_str!("shaders/fused_add_rms_norm.wgsl"),
        &[
            bge(0, &residual.buffer),
            bge(1, &input.buffer),
            bge(2, &weight.buffer),
            bge(3, &output.buffer),
            wgpu::BindGroupEntry {
                binding: 4,
                resource: residual_out.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: params.as_entire_binding(),
            },
        ],
        [div_ceil(n, 256), 1, 1],
    );

    Ok((output, residual_out))
}

/// Fused add + RMS norm + matvec with transposed f16 weights.
/// Computes: hidden_new = residual + addition; normed = rms_norm(hidden_new); out = normed @ w_t
/// Replaces: fused_add_rms_norm + matvec = 2 dispatches → 1.
/// K must be ≤ 1024.
#[allow(clippy::too_many_arguments)]
pub fn fused_add_rms_norm_matvec(
    residual: &WgpuTensor,
    addition: &WgpuTensor,
    norm_weight: &WgpuTensor,
    w_t: &WgpuTensor,
    n_out: u32,
    eps: f32,
) -> Result<(WgpuTensor, WgpuTensor), WgpuError> {
    let k = residual.shape[1] as u32;
    let matvec_out = WgpuTensor::zeros(&residual.device, &[1, n_out as usize], WgpuDType::F32);
    let hidden_out = WgpuTensor::zeros(&residual.device, &residual.shape, WgpuDType::F32);
    let params = params_buffer(&residual.device, [k, n_out, eps.to_bits(), 0]);

    let base_shader = if w_t.dtype == WgpuDType::Q4_0Packed {
        include_str!("shaders/fused_add_rms_norm_matvec_t_q4_0.wgsl")
    } else {
        include_str!("shaders/fused_add_rms_norm_matvec_t_f16.wgsl")
    };
    let shader_src = base_shader.replace("__MAX_K__", &k.to_string());
    dispatch_dynamic(
        &residual.device,
        &shader_src,
        &[
            bge(0, &residual.buffer),
            bge(1, &addition.buffer),
            bge(2, &norm_weight.buffer),
            bge(3, &w_t.buffer),
            wgpu::BindGroupEntry {
                binding: 4,
                resource: matvec_out.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: hidden_out.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: params.as_entire_binding(),
            },
        ],
        [div_ceil(n_out, 256), 1, 1],
    );

    Ok((matvec_out, hidden_out))
}

/// Fused QKV slice + RoPE + KV cache write.
/// Takes QKV output, applies RoPE to Q and K, writes K/V to cache, returns Q_rope.
/// Replaces: 3× slice + 2× rope + 2× cache_write = 7 ops → 1 dispatch.
#[allow(clippy::too_many_arguments)]
pub fn rope_slice_cache(
    qkv: &WgpuTensor,
    cos_cache: &WgpuTensor,
    sin_cache: &WgpuTensor,
    k_cache: &WgpuTensor,
    v_cache: &WgpuTensor,
    q_size: usize,
    kv_size: usize,
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    position: usize,
    max_seq_len: usize,
) -> Result<WgpuTensor, WgpuError> {
    let q_out = WgpuTensor::zeros(&qkv.device, &[1, q_size], WgpuDType::F32);
    let total = (q_size + kv_size) as u32;

    let params = params_buffer_8(
        &qkv.device,
        [
            q_size as u32,
            kv_size as u32,
            head_dim as u32,
            position as u32,
            num_q_heads as u32,
            num_kv_heads as u32,
            max_seq_len as u32,
            kv_size as u32, // cache_stride
        ],
    );

    dispatch(
        &qkv.device,
        include_str!("shaders/rope_slice_cache.wgsl"),
        &[
            bge(0, &qkv.buffer),
            bge(1, &cos_cache.buffer),
            bge(2, &sin_cache.buffer),
            bge(3, &q_out.buffer),
            bge(4, &k_cache.buffer),
            bge(5, &v_cache.buffer),
            wgpu::BindGroupEntry {
                binding: 6,
                resource: params.as_entire_binding(),
            },
        ],
        [div_ceil(total, 256), 1, 1],
    );

    Ok(q_out)
}

/// Fused matvec + argmax: computes y = argmax(x * W_t) without materializing all N logits.
/// Uses 2 dispatches: partial argmax per workgroup, then final reduction.
pub async fn matvec_argmax(
    x: &WgpuTensor,
    w_t: &WgpuTensor,
    k: u32,
    n: u32,
) -> Result<u32, WgpuError> {
    let num_wg = div_ceil(n, 256);
    let partial_vals = WgpuTensor::zeros(&x.device, &[num_wg as usize], WgpuDType::F32);
    let partial_idxs = WgpuTensor::zeros(&x.device, &[num_wg as usize], WgpuDType::U32);
    let params = params_buffer(&x.device, [k, n, 0, 0]);

    dispatch(
        &x.device,
        include_str!("shaders/matvec_argmax.wgsl"),
        &[
            bge(0, &x.buffer),
            bge(1, &w_t.buffer),
            bge(2, &partial_vals.buffer),
            bge(3, &partial_idxs.buffer),
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params.as_entire_binding(),
            },
        ],
        [num_wg, 1, 1],
    );

    // Final reduction: argmax over partial results
    let final_out = WgpuTensor::zeros(&x.device, &[1], WgpuDType::U32);
    let params2 = params_buffer(&x.device, [num_wg, 0, 0, 0]);

    dispatch(
        &x.device,
        include_str!("shaders/argmax_partial.wgsl"),
        &[
            bge(0, &partial_vals.buffer),
            bge(1, &partial_idxs.buffer),
            bge(2, &final_out.buffer),
            bge(3, &params2),
        ],
        [1, 1, 1],
    );

    readback_u32(&x.device, &final_out).await
}

/// Fused matvec + argmax with row-major weights W: [N, K].
pub async fn matvec_argmax_rowmajor(
    x: &WgpuTensor,
    w: &WgpuTensor,
    k: u32,
    n: u32,
) -> Result<u32, WgpuError> {
    let num_wg = div_ceil(n, 256);
    let partial_vals = WgpuTensor::zeros(&x.device, &[num_wg as usize], WgpuDType::F32);
    let partial_idxs = WgpuTensor::zeros(&x.device, &[num_wg as usize], WgpuDType::U32);
    let params = params_buffer(&x.device, [k, n, 0, 0]);

    let shader = if w.dtype == WgpuDType::F16Packed {
        include_str!("shaders/matvec_argmax_rowmajor_f16.wgsl")
    } else {
        include_str!("shaders/matvec_argmax_rowmajor.wgsl")
    };
    dispatch(
        &x.device,
        shader,
        &[
            bge(0, &x.buffer),
            bge(1, &w.buffer),
            bge(2, &partial_vals.buffer),
            bge(3, &partial_idxs.buffer),
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params.as_entire_binding(),
            },
        ],
        [num_wg, 1, 1],
    );

    let final_out = WgpuTensor::zeros(&x.device, &[1], WgpuDType::U32);
    let params2 = params_buffer(&x.device, [num_wg, 0, 0, 0]);

    dispatch(
        &x.device,
        include_str!("shaders/argmax_partial.wgsl"),
        &[
            bge(0, &partial_vals.buffer),
            bge(1, &partial_idxs.buffer),
            bge(2, &final_out.buffer),
            bge(3, &params2),
        ],
        [1, 1, 1],
    );

    readback_u32(&x.device, &final_out).await
}

/// Fused matvec + argmax with transposed weights W_t: [K, N].
/// Supports both f16-packed and Q4_0-packed weights.
/// Uses coalesced memory access (adjacent threads read adjacent cols).
pub async fn matvec_argmax_transposed(
    x: &WgpuTensor,
    w_t: &WgpuTensor,
    k: u32,
    n: u32,
) -> Result<u32, WgpuError> {
    let num_wg = div_ceil(n, 256);
    let partial_vals = WgpuTensor::zeros(&x.device, &[num_wg as usize], WgpuDType::F32);
    let partial_idxs = WgpuTensor::zeros(&x.device, &[num_wg as usize], WgpuDType::U32);
    let params = params_buffer(&x.device, [k, n, 0, 0]);

    let shader = if w_t.dtype == WgpuDType::Q4_0Packed {
        include_str!("shaders/matvec_argmax_t_q4_0.wgsl")
    } else {
        include_str!("shaders/matvec_argmax_t_f16.wgsl")
    };
    dispatch(
        &x.device,
        shader,
        &[
            bge(0, &x.buffer),
            bge(1, &w_t.buffer),
            bge(2, &partial_vals.buffer),
            bge(3, &partial_idxs.buffer),
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params.as_entire_binding(),
            },
        ],
        [num_wg, 1, 1],
    );

    let final_out = WgpuTensor::zeros(&x.device, &[1], WgpuDType::U32);
    let params2 = params_buffer(&x.device, [num_wg, 0, 0, 0]);

    dispatch(
        &x.device,
        include_str!("shaders/argmax_partial.wgsl"),
        &[
            bge(0, &partial_vals.buffer),
            bge(1, &partial_idxs.buffer),
            bge(2, &final_out.buffer),
            bge(3, &params2),
        ],
        [1, 1, 1],
    );

    readback_u32(&x.device, &final_out).await
}

/// Read back a single u32 from a GPU buffer. Batches the copy into the
/// command batcher so dispatches + copy go in one submit.
async fn readback_u32(device: &WgpuDevice, src: &WgpuTensor) -> Result<u32, WgpuError> {
    let staging = Arc::new(device.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback_staging"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    // Batch the copy with pending dispatches — one submit for everything
    device
        .batcher
        .lock()
        .unwrap()
        .push_copy(src.buffer.clone(), 0, staging.clone(), 0, 4);
    device.flush();

    let slice = staging.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.device.poll(wgpu::Maintain::Wait);
    rx.await
        .map_err(|_| WgpuError::BufferMap)?
        .map_err(|_| WgpuError::BufferMap)?;
    let data = slice.get_mapped_range();
    let result = bytemuck::cast_slice::<u8, u32>(&data)[0];
    drop(data);
    staging.unmap();
    Ok(result)
}

/// GPU-side argmax: returns the index of the maximum value.
/// input: [N] → u32 index. Avoids reading all N floats back to CPU.
pub async fn argmax(input: &WgpuTensor) -> Result<u32, WgpuError> {
    let n = input.numel() as u32;
    let output = WgpuTensor::zeros(&input.device, &[1], WgpuDType::U32);
    let params = params_buffer(&input.device, [n, 0, 0, 0]);

    dispatch(
        &input.device,
        include_str!("shaders/argmax.wgsl"),
        &[
            bge(0, &input.buffer),
            bge(1, &output.buffer),
            bge(2, &params),
        ],
        [1, 1, 1],
    );

    readback_u32(&input.device, &output).await
}

/// Write M contiguous rows into a cache buffer starting at `start_position`.
/// cache: [max_seq, dim], data: [M, dim], writes rows start..start+M.
pub fn cache_write_batch(
    cache: &WgpuTensor,
    data: &WgpuTensor,
    start_position: usize,
    count: usize,
) -> Result<(), WgpuError> {
    let dim = cache.shape[1];
    let byte_offset = (start_position * dim * cache.dtype.size_bytes()) as u64;
    let byte_length = (count * dim * cache.dtype.size_bytes()) as u64;

    cache.device.batcher.lock().unwrap().push_copy(
        data.buffer.clone(),
        0,
        cache.buffer.clone(),
        byte_offset,
        byte_length,
    );
    Ok(())
}

/// Causal self-attention for prefill (M>1).
/// q: [M, num_q_heads * head_dim]
/// k_new, v_new: [M, num_kv_heads * head_dim] (new KV to write to cache)
/// k_cache, v_cache: [max_seq, num_kv_heads * head_dim]
/// cache_len: tokens already in cache before this prefill
/// Returns: [M, num_q_heads * head_dim]
#[allow(clippy::too_many_arguments)]
pub fn attention_prefill(
    q: &WgpuTensor,
    k_new: &WgpuTensor,
    v_new: &WgpuTensor,
    k_cache: &WgpuTensor,
    v_cache: &WgpuTensor,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_m: u32,
    cache_len: u32,
) -> Result<WgpuTensor, WgpuError> {
    let q_size = (num_q_heads * head_dim) as usize;
    let output = WgpuTensor::zeros(&q.device, &[seq_m as usize, q_size], WgpuDType::F32);

    let scale_bits = (1.0f32 / (head_dim as f32).sqrt()).to_bits();
    let params = params_buffer_8(
        &q.device,
        [
            num_q_heads,
            num_kv_heads,
            head_dim,
            seq_m,
            scale_bits,
            cache_len,
            0,
            0,
        ],
    );

    dispatch(
        &q.device,
        include_str!("shaders/attention_prefill.wgsl"),
        &[
            bge(0, &q.buffer),
            bge(1, &k_cache.buffer),
            bge(2, &v_cache.buffer),
            bge(3, &k_new.buffer),
            bge(4, &v_new.buffer),
            bge(5, &output.buffer),
            wgpu::BindGroupEntry {
                binding: 6,
                resource: params.as_entire_binding(),
            },
        ],
        [num_q_heads, 1, 1],
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
