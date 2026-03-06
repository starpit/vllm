// SPDX-License-Identifier: Apache-2.0
//! Rotary embedding kernels.
//!
//! Trait abstraction for rotary position embedding (RoPE) kernels.
//! Port of: `csrc/pos_encoding_kernels.cu`
//!
//! ## Simplifications vs Python vLLM
//!
//! The CUDA kernels here are simplified compared to Python vLLM's:
//! - Scalar loads instead of vectorized loads
//! - NeoX-style only (no GPT-J interleaved mode)
//! - No packed half2 arithmetic
//!
//! These will be upgraded to match Python vLLM's performance in a follow-up.

use candle_core::Tensor;

use crate::error::KernelResult;

/// Rotary embedding kernel interface.
pub trait RotaryKernels: Send + Sync {
    /// Apply rotary embedding to query and key tensors.
    ///
    /// * `positions` — position indices [batch] or [seq_len]
    /// * `query` — query tensor [num_tokens, num_heads * head_dim]
    /// * `key` — key tensor [num_tokens, num_kv_heads * head_dim]
    /// * `cos_sin_cache` — precomputed [max_pos, rotary_dim]
    /// * `is_neox` — whether to use NeoX-style rotation (split in half)
    ///
    /// Returns (rotated_query, rotated_key).
    ///
    /// Port of: `void rotary_embedding(positions, query, key, head_size,
    ///           cos_sin_cache, is_neox)`
    fn rotary_embedding(
        &self,
        positions: &Tensor,
        query: &Tensor,
        key: &Tensor,
        cos_sin_cache: &Tensor,
        is_neox: bool,
    ) -> KernelResult<(Tensor, Tensor)>;
}

/// CPU implementation of rotary kernels (for testing).
pub struct CpuRotaryKernels;

impl RotaryKernels for CpuRotaryKernels {
    fn rotary_embedding(
        &self,
        positions: &Tensor,
        query: &Tensor,
        key: &Tensor,
        cos_sin_cache: &Tensor,
        _is_neox: bool,
    ) -> KernelResult<(Tensor, Tensor)> {
        // Gather cos/sin for the given positions.
        // cos_sin_cache shape: [max_pos, rotary_dim] where first half is cos, second half is sin.
        let rotary_dim = cos_sin_cache.dim(1)?;
        let half = rotary_dim / 2;

        let gathered = cos_sin_cache.index_select(positions, 0)?; // [num_tokens, rotary_dim]
        let cos = gathered.narrow(1, 0, half)?; // [num_tokens, half]
        let sin = gathered.narrow(1, half, half)?; // [num_tokens, half]

        let q_rot = apply_rotary_1d(query, &cos, &sin, half)?;
        let k_rot = apply_rotary_1d(key, &cos, &sin, half)?;

        Ok((q_rot, k_rot))
    }
}

/// Apply rotary to a flat [num_tokens, dim] tensor.
/// Only rotates the first `2 * half` dimensions, leaving the rest unchanged.
fn apply_rotary_1d(x: &Tensor, cos: &Tensor, sin: &Tensor, half: usize) -> KernelResult<Tensor> {
    let dim = x.dim(1)?;
    let rot_dim = 2 * half;

    if rot_dim > dim {
        return Err(crate::error::KernelError::Shape(format!(
            "rotary dim {} > tensor dim {}",
            rot_dim, dim
        )));
    }

    let x_rot = x.narrow(1, 0, rot_dim)?;
    let x1 = x_rot.narrow(1, 0, half)?;
    let x2 = x_rot.narrow(1, half, half)?;

    // Rotate: [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
    let r1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
    let r2 = (x1.broadcast_mul(sin)? + x2.broadcast_mul(cos)?)?;
    let rotated = Tensor::cat(&[&r1, &r2], 1)?;

    if rot_dim < dim {
        let pass_through = x.narrow(1, rot_dim, dim - rot_dim)?;
        let result = Tensor::cat(&[&rotated, &pass_through], 1)?;
        Ok(result)
    } else {
        Ok(rotated)
    }
}

// ---------------------------------------------------------------------------
// CUDA implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod cuda_ffi {
    unsafe extern "C" {
        pub fn rotary_embedding_f32(
            positions: *const u32,
            query: *mut f32,
            key: *mut f32,
            cos_sin_cache: *const f32,
            rotary_dim: i32,
            total_q_dim: i32,
            total_k_dim: i32,
            head_size: i32,
            num_tokens: i32,
        );
        pub fn rotary_embedding_f16(
            positions: *const u32,
            query: *mut u16,
            key: *mut u16,
            cos_sin_cache: *const u16,
            rotary_dim: i32,
            total_q_dim: i32,
            total_k_dim: i32,
            head_size: i32,
            num_tokens: i32,
        );
        pub fn rotary_embedding_bf16(
            positions: *const u32,
            query: *mut u16,
            key: *mut u16,
            cos_sin_cache: *const u16,
            rotary_dim: i32,
            total_q_dim: i32,
            total_k_dim: i32,
            head_size: i32,
            num_tokens: i32,
        );
    }
}

/// CUDA implementation of rotary kernels.
#[cfg(feature = "cuda")]
pub struct CudaRotaryKernels;

#[cfg(feature = "cuda")]
impl CudaRotaryKernels {
    /// Extract a raw device pointer (as usize) from a contiguous CUDA tensor.
    fn device_ptr_of<T: cudarc::driver::DeviceRepr + candle_core::cuda_backend::CudaDType>(
        tensor: &Tensor,
    ) -> KernelResult<usize> {
        use cudarc::driver::DevicePtr;
        let cuda_dev = tensor
            .device()
            .as_cuda_device()
            .map_err(|e| crate::error::KernelError::Other(format!("{e}")))?;
        let stream = cuda_dev.cuda_stream();
        let (storage, layout) = tensor.storage_and_layout();
        match &*storage {
            candle_core::Storage::Cuda(cuda_storage) => {
                let slice = cuda_storage.as_cuda_slice::<T>()?;
                let view = slice.slice(layout.start_offset()..);
                let (ptr, _sync_guard) = view.device_ptr(&stream);
                Ok(ptr as usize)
            }
            _ => Err(crate::error::KernelError::Other(
                "expected CUDA tensor".to_string(),
            )),
        }
    }
}

#[cfg(feature = "cuda")]
impl RotaryKernels for CudaRotaryKernels {
    fn rotary_embedding(
        &self,
        positions: &Tensor,
        query: &Tensor,
        key: &Tensor,
        cos_sin_cache: &Tensor,
        _is_neox: bool,
    ) -> KernelResult<(Tensor, Tensor)> {
        use candle_core::DType;

        let rotary_dim = cos_sin_cache.dim(1)?;
        let q_dims = query.shape().dims();
        let total_q_dim = *q_dims
            .last()
            .ok_or_else(|| crate::error::KernelError::Other("empty query tensor".to_string()))?;
        let num_tokens: usize = q_dims[..q_dims.len() - 1].iter().product();
        let k_dims = key.shape().dims();
        let total_k_dim = *k_dims
            .last()
            .ok_or_else(|| crate::error::KernelError::Other("empty key tensor".to_string()))?;

        // For the flat trait API, treat the entire dim as one "head".
        // Phase 3.8 will wire per-head rotation via RotaryEmbedding.
        let head_size = total_q_dim;

        // Clone query/key for out-of-place semantics, make contiguous.
        let q_out = query.contiguous()?.clone();
        let k_out = key.contiguous()?.clone();
        let positions = positions.contiguous()?;
        let cos_sin_cache = cos_sin_cache.contiguous()?;

        match query.dtype() {
            DType::F32 => {
                let p = Self::device_ptr_of::<u32>(&positions)?;
                let q = Self::device_ptr_of::<f32>(&q_out)?;
                let k = Self::device_ptr_of::<f32>(&k_out)?;
                let c = Self::device_ptr_of::<f32>(&cos_sin_cache)?;
                unsafe {
                    cuda_ffi::rotary_embedding_f32(
                        p as *const u32,
                        q as *mut f32,
                        k as *mut f32,
                        c as *const f32,
                        rotary_dim as i32,
                        total_q_dim as i32,
                        total_k_dim as i32,
                        head_size as i32,
                        num_tokens as i32,
                    );
                }
            }
            DType::F16 => {
                let p = Self::device_ptr_of::<u32>(&positions)?;
                let q = Self::device_ptr_of::<half::f16>(&q_out)?;
                let k = Self::device_ptr_of::<half::f16>(&k_out)?;
                let c = Self::device_ptr_of::<half::f16>(&cos_sin_cache)?;
                unsafe {
                    cuda_ffi::rotary_embedding_f16(
                        p as *const u32,
                        q as *mut u16,
                        k as *mut u16,
                        c as *const u16,
                        rotary_dim as i32,
                        total_q_dim as i32,
                        total_k_dim as i32,
                        head_size as i32,
                        num_tokens as i32,
                    );
                }
            }
            DType::BF16 => {
                let p = Self::device_ptr_of::<u32>(&positions)?;
                let q = Self::device_ptr_of::<half::bf16>(&q_out)?;
                let k = Self::device_ptr_of::<half::bf16>(&k_out)?;
                let c = Self::device_ptr_of::<half::bf16>(&cos_sin_cache)?;
                unsafe {
                    cuda_ffi::rotary_embedding_bf16(
                        p as *const u32,
                        q as *mut u16,
                        k as *mut u16,
                        c as *const u16,
                        rotary_dim as i32,
                        total_q_dim as i32,
                        total_k_dim as i32,
                        head_size as i32,
                        num_tokens as i32,
                    );
                }
            }
            _ => {
                return CpuRotaryKernels.rotary_embedding(
                    &positions,
                    query,
                    key,
                    &cos_sin_cache,
                    _is_neox,
                );
            }
        }
        Ok((q_out, k_out))
    }
}

// ---------------------------------------------------------------------------
// Fused RoPE via candle CustomOp2 (bypasses cudarc event overhead)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod fused_rope {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::CudaDType;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{CpuStorage, CudaStorage, CustomOp2, DType, Layout, Result, Shape, Tensor};

    /// Fused RoPE CustomOp — rotates a single tensor (Q or K) in-place on the
    /// GPU using the fused CUDA kernel, bypassing the decomposed candle ops that
    /// would otherwise launch 5-7 separate kernels per tensor.
    ///
    /// Inputs to `apply_op2`: `(x, positions)`
    /// Stored: `cos_sin_cache` tensor (persistent, precomputed).
    struct FusedRotaryOp {
        /// Combined cos|sin cache: `[max_pos, rotary_dim]`.
        /// Layout per row: `[cos_0..cos_{half-1}, sin_0..sin_{half-1}]`.
        cos_sin_cache: Tensor,
        /// Dimension per head.
        head_size: usize,
    }

    impl FusedRotaryOp {
        fn cuda_fwd_t<T: CudaDType + cudarc::driver::DeviceRepr>(
            &self,
            x: &CudaStorage,
            x_l: &Layout,
            pos: &CudaStorage,
            pos_l: &Layout,
            dtype: DType,
        ) -> Result<(CudaStorage, Shape)> {
            let dev = x.device();
            let stream = dev.cuda_stream();
            let out_shape = x_l.shape().clone();
            let elem_count = out_shape.elem_count();

            // Validate inputs.
            let num_tokens = pos_l.shape().dims1()?;
            let total_dim: usize = x_l.shape().elem_count() / num_tokens;

            let rotary_dim = self
                .cos_sin_cache
                .dim(1)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

            // Get all raw pointers in a single block to minimize event overhead.
            // Each device_ptr() call records events in cudarc; fewer calls = less overhead.
            let x_slice = x.as_cuda_slice::<T>()?;
            let x_view = x_slice.slice(x_l.start_offset()..);
            let pos_slice = pos.as_cuda_slice::<u32>()?;
            let pos_view = pos_slice.slice(pos_l.start_offset()..);
            let (cache_storage, cache_layout) = self.cos_sin_cache.storage_and_layout();
            let cache_cuda = match &*cache_storage {
                candle_core::Storage::Cuda(c) => c,
                _ => candle_core::bail!("cos_sin_cache must be a CUDA tensor"),
            };
            let cache_slice = cache_cuda.as_cuda_slice::<T>()?;
            let cache_view = cache_slice.slice(cache_layout.start_offset()..);

            // Allocate output, copy input, and launch kernel — 3 device_ptr calls total.
            let dst = unsafe { dev.alloc::<T>(elem_count)? };
            unsafe {
                let (src_ptr, _g1) = x_view.device_ptr(&stream);
                let (dst_ptr, _g2) = dst.device_ptr(&stream);
                let (pos_ptr, _g3) = pos_view.device_ptr(&stream);
                let (cache_ptr, _g4) = cache_view.device_ptr(&stream);

                // Device-to-device copy: input → output buffer.
                cudarc::driver::result::memcpy_dtod_async(
                    dst_ptr,
                    src_ptr,
                    elem_count * std::mem::size_of::<T>(),
                    stream.cu_stream(),
                )
                .map_err(|e| candle_core::Error::Msg(format!("dtod copy: {e}")))?;

                // Launch fused RoPE kernel (modifies dst in-place).
                // Pass as "query" with total_k_dim=0 to process only this tensor.
                match dtype {
                    DType::F32 => {
                        super::cuda_ffi::rotary_embedding_f32(
                            pos_ptr as *const u32,
                            dst_ptr as *mut f32,
                            std::ptr::null_mut(),
                            cache_ptr as *const f32,
                            rotary_dim as i32,
                            total_dim as i32,
                            0,
                            self.head_size as i32,
                            num_tokens as i32,
                        );
                    }
                    DType::F16 => {
                        super::cuda_ffi::rotary_embedding_f16(
                            pos_ptr as *const u32,
                            dst_ptr as *mut u16,
                            std::ptr::null_mut(),
                            cache_ptr as *const u16,
                            rotary_dim as i32,
                            total_dim as i32,
                            0,
                            self.head_size as i32,
                            num_tokens as i32,
                        );
                    }
                    DType::BF16 => {
                        super::cuda_ffi::rotary_embedding_bf16(
                            pos_ptr as *const u32,
                            dst_ptr as *mut u16,
                            std::ptr::null_mut(),
                            cache_ptr as *const u16,
                            rotary_dim as i32,
                            total_dim as i32,
                            0,
                            self.head_size as i32,
                            num_tokens as i32,
                        );
                    }
                    dt => candle_core::bail!("fused RoPE unsupported dtype {dt:?}"),
                }
            }

            let dst_storage = CudaStorage::wrap_cuda_slice(dst, dev.clone());
            Ok((dst_storage, out_shape))
        }
    }

    impl CustomOp2 for FusedRotaryOp {
        fn name(&self) -> &'static str {
            "fused-rotary-embedding"
        }

        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("fused RoPE is CUDA-only; use candle ops on CPU")
        }

        fn cuda_fwd(
            &self,
            x: &CudaStorage,
            x_l: &Layout,
            pos: &CudaStorage,
            pos_l: &Layout,
        ) -> Result<(CudaStorage, Shape)> {
            let dt = x.dtype();
            match dt {
                DType::F32 => self.cuda_fwd_t::<f32>(x, x_l, pos, pos_l, dt),
                DType::F16 => self.cuda_fwd_t::<half::f16>(x, x_l, pos, pos_l, dt),
                DType::BF16 => self.cuda_fwd_t::<half::bf16>(x, x_l, pos, pos_l, dt),
                dt => candle_core::bail!("fused RoPE unsupported dtype {dt:?}"),
            }
        }
    }

    /// Apply fused rotary embedding on CUDA via CustomOp2.
    ///
    /// * `x` — tensor to rotate: `[num_tokens, num_heads, head_dim]` or `[num_tokens, total_dim]`
    /// * `positions` — `[num_tokens]` u32
    /// * `cos_sin_cache` — `[max_pos, rotary_dim]` combined cache
    /// * `head_size` — dimension per head
    ///
    /// Returns rotated tensor (same shape as input).
    pub fn fused_rotary_apply(
        x: &Tensor,
        positions: &Tensor,
        cos_sin_cache: &Tensor,
        head_size: usize,
    ) -> candle_core::Result<Tensor> {
        // The kernel expects contiguous [num_tokens, total_dim] layout.
        let x = x.contiguous()?;
        let positions = positions.contiguous()?;
        let op = FusedRotaryOp {
            cos_sin_cache: cos_sin_cache.clone(),
            head_size,
        };
        x.apply_op2(&positions, op)
    }

    /// Apply fused rotary embedding to both Q and K in a single kernel launch.
    ///
    /// Allocates separate Q and K output buffers, copies inputs, then launches
    /// one kernel that processes both. Saves one kernel launch per layer vs
    /// calling `fused_rotary_apply` twice.
    pub fn fused_rotary_apply_qk(
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
        cos_sin_cache: &Tensor,
        head_size: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaDType;
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        fn fwd_t<T: CudaDType + cudarc::driver::DeviceRepr>(
            q: &Tensor,
            k: &Tensor,
            positions: &Tensor,
            cos_sin_cache: &Tensor,
            head_size: usize,
        ) -> candle_core::Result<(Tensor, Tensor)> {
            let (q_storage, q_layout) = q.storage_and_layout();
            let q_cuda = match &*q_storage {
                candle_core::Storage::Cuda(c) => c,
                _ => candle_core::bail!("Q must be CUDA"),
            };
            let (k_storage, k_layout) = k.storage_and_layout();
            let k_cuda = match &*k_storage {
                candle_core::Storage::Cuda(c) => c,
                _ => candle_core::bail!("K must be CUDA"),
            };
            let (pos_storage, pos_layout) = positions.storage_and_layout();
            let pos_cuda = match &*pos_storage {
                candle_core::Storage::Cuda(c) => c,
                _ => candle_core::bail!("positions must be CUDA"),
            };
            let (cache_storage, cache_layout) = cos_sin_cache.storage_and_layout();
            let cache_cuda = match &*cache_storage {
                candle_core::Storage::Cuda(c) => c,
                _ => candle_core::bail!("cos_sin_cache must be CUDA"),
            };

            let dev = q_cuda.device();
            let stream = dev.cuda_stream();
            let num_tokens = pos_layout.shape().dims1()?;
            let q_elem = q_layout.shape().elem_count();
            let k_elem = k_layout.shape().elem_count();
            let total_q_dim = q_elem / num_tokens;
            let total_k_dim = k_elem / num_tokens;
            let rotary_dim = cache_layout.shape().dims()[1];

            // Get raw pointers.
            let q_slice = q_cuda
                .as_cuda_slice::<T>()?
                .slice(q_layout.start_offset()..);
            let k_slice = k_cuda
                .as_cuda_slice::<T>()?
                .slice(k_layout.start_offset()..);
            let pos_slice = pos_cuda
                .as_cuda_slice::<u32>()?
                .slice(pos_layout.start_offset()..);
            let cache_slice = cache_cuda
                .as_cuda_slice::<T>()?
                .slice(cache_layout.start_offset()..);

            // Allocate separate output buffers.
            let q_dst = unsafe { dev.alloc::<T>(q_elem)? };
            let k_dst = unsafe { dev.alloc::<T>(k_elem)? };

            unsafe {
                let (q_src_ptr, _g1) = q_slice.device_ptr(&stream);
                let (k_src_ptr, _g2) = k_slice.device_ptr(&stream);
                let (q_dst_ptr, _g3) = q_dst.device_ptr(&stream);
                let (k_dst_ptr, _g4) = k_dst.device_ptr(&stream);
                let (pos_ptr, _g5) = pos_slice.device_ptr(&stream);
                let (cache_ptr, _g6) = cache_slice.device_ptr(&stream);

                // Copy Q and K to output buffers.
                let q_bytes = q_elem * std::mem::size_of::<T>();
                let k_bytes = k_elem * std::mem::size_of::<T>();
                cudarc::driver::result::memcpy_dtod_async(
                    q_dst_ptr,
                    q_src_ptr,
                    q_bytes,
                    stream.cu_stream(),
                )
                .map_err(|e| candle_core::Error::Msg(format!("dtod Q: {e}")))?;
                cudarc::driver::result::memcpy_dtod_async(
                    k_dst_ptr,
                    k_src_ptr,
                    k_bytes,
                    stream.cu_stream(),
                )
                .map_err(|e| candle_core::Error::Msg(format!("dtod K: {e}")))?;

                // Single kernel launch for both Q and K.
                match q.dtype() {
                    candle_core::DType::F32 => {
                        super::cuda_ffi::rotary_embedding_f32(
                            pos_ptr as *const u32,
                            q_dst_ptr as *mut f32,
                            k_dst_ptr as *mut f32,
                            cache_ptr as *const f32,
                            rotary_dim as i32,
                            total_q_dim as i32,
                            total_k_dim as i32,
                            head_size as i32,
                            num_tokens as i32,
                        );
                    }
                    candle_core::DType::F16 => {
                        super::cuda_ffi::rotary_embedding_f16(
                            pos_ptr as *const u32,
                            q_dst_ptr as *mut u16,
                            k_dst_ptr as *mut u16,
                            cache_ptr as *const u16,
                            rotary_dim as i32,
                            total_q_dim as i32,
                            total_k_dim as i32,
                            head_size as i32,
                            num_tokens as i32,
                        );
                    }
                    candle_core::DType::BF16 => {
                        super::cuda_ffi::rotary_embedding_bf16(
                            pos_ptr as *const u32,
                            q_dst_ptr as *mut u16,
                            k_dst_ptr as *mut u16,
                            cache_ptr as *const u16,
                            rotary_dim as i32,
                            total_q_dim as i32,
                            total_k_dim as i32,
                            head_size as i32,
                            num_tokens as i32,
                        );
                    }
                    dt => candle_core::bail!("fused RoPE QK unsupported dtype {dt:?}"),
                }
            }

            let q_out_storage = candle_core::CudaStorage::wrap_cuda_slice(q_dst, dev.clone());
            let k_out_storage = candle_core::CudaStorage::wrap_cuda_slice(k_dst, dev.clone());

            let q_out = candle_core::Tensor::from_storage(
                candle_core::Storage::Cuda(q_out_storage),
                q.shape().clone(),
                candle_core::op::BackpropOp::none(),
                false,
            );
            let k_out = candle_core::Tensor::from_storage(
                candle_core::Storage::Cuda(k_out_storage),
                k.shape().clone(),
                candle_core::op::BackpropOp::none(),
                false,
            );

            Ok((q_out, k_out))
        }

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let positions = positions.contiguous()?;

        match q.dtype() {
            candle_core::DType::F32 => fwd_t::<f32>(&q, &k, &positions, cos_sin_cache, head_size),
            candle_core::DType::F16 => {
                fwd_t::<half::f16>(&q, &k, &positions, cos_sin_cache, head_size)
            }
            candle_core::DType::BF16 => {
                fwd_t::<half::bf16>(&q, &k, &positions, cos_sin_cache, head_size)
            }
            dt => candle_core::bail!("fused RoPE QK unsupported dtype {dt:?}"),
        }
    }
}

#[cfg(feature = "cuda")]
pub use fused_rope::fused_rotary_apply;

#[cfg(feature = "cuda")]
pub use fused_rope::fused_rotary_apply_qk;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn make_cos_sin_cache(max_pos: usize, half_dim: usize) -> Tensor {
        // Simple cache: cos and sin for frequencies
        let rotary_dim = 2 * half_dim;
        let mut data = vec![0.0f32; max_pos * rotary_dim];
        for pos in 0..max_pos {
            for i in 0..half_dim {
                let freq = 1.0 / 10000f64.powf(2.0 * i as f64 / (2 * half_dim) as f64);
                let angle = pos as f64 * freq;
                data[pos * rotary_dim + i] = angle.cos() as f32;
                data[pos * rotary_dim + half_dim + i] = angle.sin() as f32;
            }
        }
        Tensor::from_slice(&data, (max_pos, rotary_dim), &Device::Cpu).unwrap()
    }

    #[test]
    fn test_cpu_rotary_position_zero() {
        let kernels = CpuRotaryKernels;
        let cache = make_cos_sin_cache(10, 4); // rotary_dim = 8
        let positions = Tensor::new(&[0u32], &Device::Cpu).unwrap();
        let q = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], &Device::Cpu).unwrap();
        let k = q.clone();

        let (q_rot, _k_rot) = kernels
            .rotary_embedding(&positions, &q, &k, &cache, true)
            .unwrap();
        assert_eq!(q_rot.dims(), &[1, 8]);

        // At position 0, cos=1, sin=0 -> output = input
        let vals = q_rot.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, v) in vals.iter().enumerate() {
            assert!(
                (v - (i as f32 + 1.0)).abs() < 1e-4,
                "pos 0 should be identity, got {} at idx {}",
                v,
                i
            );
        }
    }

    #[test]
    fn test_cpu_rotary_shape_preservation() {
        let kernels = CpuRotaryKernels;
        let cache = make_cos_sin_cache(100, 4);
        let positions = Tensor::new(&[0u32, 1, 2, 3], &Device::Cpu).unwrap();
        let q = Tensor::ones(&[4, 8], DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones(&[4, 8], DType::F32, &Device::Cpu).unwrap();

        let (q_rot, k_rot) = kernels
            .rotary_embedding(&positions, &q, &k, &cache, true)
            .unwrap();
        assert_eq!(q_rot.dims(), &[4, 8]);
        assert_eq!(k_rot.dims(), &[4, 8]);
    }

    #[test]
    fn test_cpu_rotary_partial_dim() {
        // Test when rotary_dim < total dim (pass-through for remaining dims)
        let kernels = CpuRotaryKernels;
        let cache = make_cos_sin_cache(10, 2); // rotary_dim = 4, but tensor dim = 8
        let positions = Tensor::new(&[0u32], &Device::Cpu).unwrap();
        let q = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], &Device::Cpu).unwrap();
        let k = q.clone();

        let (q_rot, _) = kernels
            .rotary_embedding(&positions, &q, &k, &cache, true)
            .unwrap();
        assert_eq!(q_rot.dims(), &[1, 8]);

        // Last 4 dims should be unchanged (pass-through).
        let vals = q_rot.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((vals[4] - 5.0).abs() < 1e-4);
        assert!((vals[5] - 6.0).abs() < 1e-4);
        assert!((vals[6] - 7.0).abs() < 1e-4);
        assert!((vals[7] - 8.0).abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // CUDA tests — compare CUDA kernel output against CPU reference
    // -----------------------------------------------------------------------

    #[cfg(feature = "cuda")]
    fn cuda_device() -> Device {
        Device::new_cuda(0).expect("CUDA device required for this test")
    }

    #[cfg(feature = "cuda")]
    fn assert_close(label: &str, cpu: &[f32], cuda: &[f32], tol: f64) {
        assert_eq!(cpu.len(), cuda.len(), "{label}: length mismatch");
        for (i, (c, g)) in cpu.iter().zip(cuda.iter()).enumerate() {
            assert!(
                (c - g).abs() as f64 <= tol,
                "{label} mismatch at {i}: cpu={c} cuda={g}"
            );
        }
    }

    /// Helper: run rotary_embedding on CPU and CUDA, compare results.
    #[cfg(feature = "cuda")]
    fn assert_rotary_cuda_matches_cpu(
        num_tokens: usize,
        dim: usize,
        half_dim: usize,
        dtype: DType,
        tol: f64,
    ) {
        use super::CudaRotaryKernels;

        let max_pos = 128;
        let cache_cpu = make_cos_sin_cache(max_pos, half_dim)
            .to_dtype(dtype)
            .unwrap();
        let positions_cpu = Tensor::new(
            (0..num_tokens as u32).collect::<Vec<_>>().as_slice(),
            &Device::Cpu,
        )
        .unwrap();
        let q_cpu = Tensor::randn(0f32, 1.0, &[num_tokens, dim], &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let k_cpu = Tensor::randn(0f32, 1.0, &[num_tokens, dim], &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        // CPU reference.
        let (q_ref, k_ref) = CpuRotaryKernels
            .rotary_embedding(&positions_cpu, &q_cpu, &k_cpu, &cache_cpu, true)
            .unwrap();

        // CUDA kernel.
        let dev = cuda_device();
        let cache_gpu = cache_cpu.to_device(&dev).unwrap();
        let positions_gpu = positions_cpu.to_device(&dev).unwrap();
        let q_gpu = q_cpu.to_device(&dev).unwrap();
        let k_gpu = k_cpu.to_device(&dev).unwrap();
        let (q_cuda, k_cuda) = CudaRotaryKernels
            .rotary_embedding(&positions_gpu, &q_gpu, &k_gpu, &cache_gpu, true)
            .unwrap();
        let q_cuda = q_cuda.to_device(&Device::Cpu).unwrap();
        let k_cuda = k_cuda.to_device(&Device::Cpu).unwrap();

        // Compare queries.
        let q_ref_vals = q_ref
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let q_cuda_vals = q_cuda
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_close(
            &format!("rotary_q_{dtype:?}"),
            &q_ref_vals,
            &q_cuda_vals,
            tol,
        );

        // Compare keys.
        let k_ref_vals = k_ref
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let k_cuda_vals = k_cuda
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_close(
            &format!("rotary_k_{dtype:?}"),
            &k_ref_vals,
            &k_cuda_vals,
            tol,
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_rotary_f32() {
        assert_rotary_cuda_matches_cpu(4, 64, 32, DType::F32, 1e-4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_rotary_f16() {
        assert_rotary_cuda_matches_cpu(4, 64, 32, DType::F16, 5e-2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_rotary_bf16() {
        assert_rotary_cuda_matches_cpu(4, 64, 32, DType::BF16, 5e-2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_rotary_partial_dim() {
        // rotary_dim (2*16=32) < total dim (64) → pass-through for remaining dims.
        assert_rotary_cuda_matches_cpu(4, 64, 16, DType::F32, 1e-4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_rotary_many_tokens() {
        assert_rotary_cuda_matches_cpu(32, 128, 64, DType::F32, 1e-4);
    }

    // -----------------------------------------------------------------------
    // Fused RoPE CustomOp tests — compare against CPU reference
    // -----------------------------------------------------------------------

    /// Helper: run fused_rotary_apply on CUDA and compare against CPU per-head rotation.
    ///
    /// The CUDA kernel rotates each head independently, so the CPU reference must
    /// also do per-head rotation (not flat rotation like CpuRotaryKernels).
    #[cfg(feature = "cuda")]
    fn assert_fused_rotary_matches_cpu(
        num_tokens: usize,
        num_heads: usize,
        head_dim: usize,
        dtype: DType,
        tol: f64,
    ) {
        let half_dim = head_dim / 2;
        let max_pos = 128;

        // Build cos_sin_cache in the combined layout [max_pos, head_dim]
        // with [cos_half | sin_half].
        let cache = make_cos_sin_cache(max_pos, half_dim)
            .to_dtype(dtype)
            .unwrap();

        let positions_cpu = Tensor::new(
            (0..num_tokens as u32).collect::<Vec<_>>().as_slice(),
            &Device::Cpu,
        )
        .unwrap();

        // Input tensors: [num_tokens, num_heads, head_dim]
        let x_cpu = Tensor::randn(0f32, 1.0, &[num_tokens, num_heads, head_dim], &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        // CPU per-head reference: rotate each head independently.
        let mut ref_heads = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let head_slice = x_cpu.narrow(1, h, 1).unwrap().squeeze(1).unwrap(); // [num_tokens, head_dim]
            let (rotated, _) = CpuRotaryKernels
                .rotary_embedding(&positions_cpu, &head_slice, &head_slice, &cache, true)
                .unwrap();
            ref_heads.push(rotated.unsqueeze(1).unwrap());
        }
        let ref_slices: Vec<&Tensor> = ref_heads.iter().collect();
        let ref_result = Tensor::cat(&ref_slices, 1).unwrap();

        // Fused CUDA path.
        let dev = cuda_device();
        let x_gpu = x_cpu.to_device(&dev).unwrap();
        let positions_gpu = positions_cpu.to_device(&dev).unwrap();
        let cache_gpu = cache.to_device(&dev).unwrap();

        let fused_result = super::fused_rotary_apply(&x_gpu, &positions_gpu, &cache_gpu, head_dim)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();

        // Compare.
        let ref_vals = ref_result
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let fused_vals = fused_result
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_close("fused_rotary", &ref_vals, &fused_vals, tol);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_fused_rotary_f32() {
        assert_fused_rotary_matches_cpu(4, 8, 64, DType::F32, 1e-4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_fused_rotary_bf16() {
        assert_fused_rotary_matches_cpu(4, 8, 64, DType::BF16, 5e-2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_fused_rotary_f16() {
        assert_fused_rotary_matches_cpu(4, 8, 64, DType::F16, 5e-2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_fused_rotary_many_tokens() {
        // Simulates prefill with larger batch.
        assert_fused_rotary_matches_cpu(32, 24, 128, DType::BF16, 5e-2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_fused_rotary_gqa() {
        // GQA: fewer KV heads (2) vs Q heads (8).
        assert_fused_rotary_matches_cpu(4, 2, 64, DType::BF16, 5e-2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_fused_rotary_single_token() {
        // Decode step: single token.
        assert_fused_rotary_matches_cpu(1, 24, 128, DType::BF16, 5e-2);
    }

    /// Build cos_sin_cache the same way `RotaryEmbedding::new()` does:
    /// compute cos/sin with duplicated freqs, then narrow + cat.
    /// This produces a non-trivially-strided tensor (unlike `make_cos_sin_cache`
    /// which uses `from_slice` and is always contiguous).
    #[cfg(feature = "cuda")]
    fn make_cos_sin_cache_via_narrow(
        max_pos: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Tensor {
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| (1.0 / 10000f64.powf(2.0 * i as f64 / head_dim as f64)) as f32)
            .collect();
        let inv_freq_tensor = Tensor::from_slice(&inv_freq, half_dim, device).unwrap();
        let positions: Vec<f32> = (0..max_pos).map(|p| p as f32).collect();
        let pos_tensor = Tensor::from_slice(&positions, max_pos, device).unwrap();
        let pos_2d = pos_tensor.reshape((max_pos, 1)).unwrap();
        let inv_freq_2d = inv_freq_tensor.reshape((1, half_dim)).unwrap();
        let freqs = pos_2d.matmul(&inv_freq_2d).unwrap();
        let freqs_full = Tensor::cat(&[&freqs, &freqs], 1).unwrap();
        let cos_cache = freqs_full.cos().unwrap().to_dtype(dtype).unwrap();
        let sin_cache = freqs_full.sin().unwrap().to_dtype(dtype).unwrap();
        // This is the exact pattern from RotaryEmbedding::new() that was
        // producing a non-contiguous tensor and causing garbled output.
        Tensor::cat(
            &[
                &cos_cache.narrow(1, 0, half_dim).unwrap(),
                &sin_cache.narrow(1, 0, half_dim).unwrap(),
            ],
            1,
        )
        .unwrap()
        .contiguous()
        .unwrap()
    }

    /// Regression test: fused RoPE with cache built via narrow+cat (model path).
    ///
    /// This catches the bug where `Tensor::cat` of column-narrowed views
    /// produces a tensor with unexpected layout that the CUDA kernel
    /// misinterprets via raw pointer arithmetic.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_fused_rotary_narrow_cat_cache() {
        let dev = cuda_device();
        let num_tokens = 4;
        let num_heads = 8;
        let head_dim = 128;
        let half_dim = head_dim / 2;
        let max_pos = 128;

        // Build cache via narrow+cat on GPU (model construction path).
        let cache_gpu = make_cos_sin_cache_via_narrow(max_pos, head_dim, DType::BF16, &dev);

        // Build equivalent cache via from_slice on CPU (test construction path).
        let cache_ref = make_cos_sin_cache(max_pos, half_dim)
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&dev)
            .unwrap();

        let positions =
            Tensor::new((0..num_tokens as u32).collect::<Vec<_>>().as_slice(), &dev).unwrap();
        let x = Tensor::randn(0f32, 1.0, &[num_tokens, num_heads, head_dim], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&dev)
            .unwrap();

        // Both caches should produce identical results.
        let result_narrow =
            super::fused_rotary_apply(&x, &positions, &cache_gpu, head_dim).unwrap();
        let result_ref = super::fused_rotary_apply(&x, &positions, &cache_ref, head_dim).unwrap();

        let vals_narrow = result_narrow
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let vals_ref = result_ref
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_close("narrow_cat_cache", &vals_ref, &vals_narrow, 5e-2);
    }
}
