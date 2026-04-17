// SPDX-License-Identifier: Apache-2.0
//! Rotary positional embedding cache and rope scaling config.

use anyhow::Result;

use ferrite_cuda_core::device::GpuDevice;
use ferrite_cuda_core::dtype::DType;
use ferrite_cuda_core::tensor::GpuTensor;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Parsed LLaMA config (mirrors the HF config).
/// Llama 3.x rope_scaling parameters.
#[derive(Debug, Clone)]
pub struct Llama3RopeScaling {
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_max_position_embeddings: usize,
}

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub head_dim: usize,
    pub tie_word_embeddings: bool,
    pub llama3_rope_scaling: Option<Llama3RopeScaling>,
}

// ---------------------------------------------------------------------------
// RoPE cache
// ---------------------------------------------------------------------------

/// Pre-computed rotary embedding cos/sin cache on GPU.
pub struct RotaryCache {
    /// `[max_pos, rotary_dim]` combined cos|sin cache (used by RoPE kernels).
    pub cos_sin_cache: GpuTensor,
    /// `[max_pos, rotary_dim/2]` separate cos cache (used by FA2 fused RoPE).
    pub cos_cache: GpuTensor,
    /// `[max_pos, rotary_dim/2]` separate sin cache (used by FA2 fused RoPE).
    pub sin_cache: GpuTensor,
    pub head_dim: usize,
}

impl RotaryCache {
    /// Build the cos/sin cache on GPU.
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    pub unsafe fn new(
        head_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        Self::new_from_stream(
            head_dim,
            max_pos,
            rope_theta,
            llama3_scaling,
            dtype,
            device.compute_stream,
        )
    }

    /// Build the cos/sin cache on GPU using an explicit CUDA stream.
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    pub unsafe fn new_from_stream(
        head_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let rotary_dim = head_dim; // full rotary for LLaMA
        let half = rotary_dim / 2;

        // Compute inverse frequencies, optionally with llama3 scaling.
        let inv_freqs: Vec<f64> = (0..half)
            .map(|i| {
                let freq = 1.0 / rope_theta.powf(2.0 * i as f64 / rotary_dim as f64);
                if let Some(scaling) = llama3_scaling {
                    let old_context_len = scaling.original_max_position_embeddings as f64;
                    let low_freq_wavelen = old_context_len / scaling.low_freq_factor;
                    let high_freq_wavelen = old_context_len / scaling.high_freq_factor;
                    let wavelen = 2.0 * std::f64::consts::PI / freq;
                    if wavelen < high_freq_wavelen {
                        freq // high frequency: keep as-is
                    } else if wavelen > low_freq_wavelen {
                        freq / scaling.factor // low frequency: scale down
                    } else {
                        // smooth interpolation
                        let smooth = (old_context_len / wavelen - scaling.low_freq_factor)
                            / (scaling.high_freq_factor - scaling.low_freq_factor);
                        (1.0 - smooth) * freq / scaling.factor + smooth * freq
                    }
                } else {
                    freq
                }
            })
            .collect();

        // Build on CPU, then copy to GPU.
        let mut cache = vec![0f32; max_pos * rotary_dim];
        for pos in 0..max_pos {
            for i in 0..half {
                let angle = pos as f64 * inv_freqs[i];
                cache[pos * rotary_dim + i] = angle.cos() as f32;
                cache[pos * rotary_dim + half + i] = angle.sin() as f32;
            }
        }

        let nbytes = max_pos * rotary_dim * dtype.size_bytes();
        let gpu_ptr = ferrite_cuda_core::driver::mem_alloc(nbytes)?;

        // Convert to target dtype and upload.
        match dtype {
            DType::F32 => {
                let host = ferrite_cuda_core::driver::mem_alloc_host(nbytes)?;
                std::ptr::copy_nonoverlapping(cache.as_ptr() as *const u8, host, nbytes);
                ferrite_cuda_core::driver::memcpy_htod_async(gpu_ptr, host, nbytes, stream)?;
                ferrite_cuda_core::driver::stream_synchronize(stream)?;
                ferrite_cuda_core::driver::mem_free_host(host)?;
            }
            DType::F16 => {
                let f16_data: Vec<half::f16> =
                    cache.iter().map(|&v| half::f16::from_f32(v)).collect();
                let host = ferrite_cuda_core::driver::mem_alloc_host(nbytes)?;
                std::ptr::copy_nonoverlapping(f16_data.as_ptr() as *const u8, host, nbytes);
                ferrite_cuda_core::driver::memcpy_htod_async(gpu_ptr, host, nbytes, stream)?;
                ferrite_cuda_core::driver::stream_synchronize(stream)?;
                ferrite_cuda_core::driver::mem_free_host(host)?;
            }
            DType::BF16 => {
                let bf16_data: Vec<half::bf16> =
                    cache.iter().map(|&v| half::bf16::from_f32(v)).collect();
                let host = ferrite_cuda_core::driver::mem_alloc_host(nbytes)?;
                std::ptr::copy_nonoverlapping(bf16_data.as_ptr() as *const u8, host, nbytes);
                ferrite_cuda_core::driver::memcpy_htod_async(gpu_ptr, host, nbytes, stream)?;
                ferrite_cuda_core::driver::stream_synchronize(stream)?;
                ferrite_cuda_core::driver::mem_free_host(host)?;
            }
            _ => anyhow::bail!("unsupported dtype for RoPE cache: {:?}", dtype),
        }

        let cos_sin_cache = GpuTensor::new(gpu_ptr, &[max_pos, rotary_dim], dtype);

        // Build separate cos/sin caches for FA2 fused RoPE (needs contiguous buffers).
        let (cos_cache, sin_cache) =
            Self::build_separate_cos_sin(&cache, max_pos, rotary_dim, dtype, stream)?;

        Ok(Self {
            cos_sin_cache,
            cos_cache,
            sin_cache,
            head_dim,
        })
    }

    /// Build cos/sin cache with partial rotary dimension (rotary_dim < head_dim).
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    pub unsafe fn new_partial(
        head_dim: usize,
        rotary_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        // The pos_encoding_kernels.cu already handles head_size > rotary_dim
        // by copying non-rotary elements through. The cos_sin_cache just needs
        // to have shape [max_pos, rotary_dim] where rotary_dim <= head_dim.
        let half = rotary_dim / 2;

        let inv_freqs: Vec<f64> = (0..half)
            .map(|i| {
                let freq = 1.0 / rope_theta.powf(2.0 * i as f64 / rotary_dim as f64);
                if let Some(scaling) = llama3_scaling {
                    let old_context_len = scaling.original_max_position_embeddings as f64;
                    let low_freq_wavelen = old_context_len / scaling.low_freq_factor;
                    let high_freq_wavelen = old_context_len / scaling.high_freq_factor;
                    let wavelen = 2.0 * std::f64::consts::PI / freq;
                    if wavelen < high_freq_wavelen {
                        freq
                    } else if wavelen > low_freq_wavelen {
                        freq / scaling.factor
                    } else {
                        let smooth = (old_context_len / wavelen - scaling.low_freq_factor)
                            / (scaling.high_freq_factor - scaling.low_freq_factor);
                        (1.0 - smooth) * freq / scaling.factor + smooth * freq
                    }
                } else {
                    freq
                }
            })
            .collect();

        let mut cache = vec![0f32; max_pos * rotary_dim];
        for pos in 0..max_pos {
            for i in 0..half {
                let angle = pos as f64 * inv_freqs[i];
                cache[pos * rotary_dim + i] = angle.cos() as f32;
                cache[pos * rotary_dim + half + i] = angle.sin() as f32;
            }
        }

        let nbytes = max_pos * rotary_dim * dtype.size_bytes();
        let gpu_ptr = ferrite_cuda_core::driver::mem_alloc(nbytes)?;

        match dtype {
            DType::F32 => {
                let host = ferrite_cuda_core::driver::mem_alloc_host(nbytes)?;
                std::ptr::copy_nonoverlapping(cache.as_ptr() as *const u8, host, nbytes);
                ferrite_cuda_core::driver::memcpy_htod_async(
                    gpu_ptr,
                    host,
                    nbytes,
                    device.compute_stream,
                )?;
                ferrite_cuda_core::driver::stream_synchronize(device.compute_stream)?;
                ferrite_cuda_core::driver::mem_free_host(host)?;
            }
            DType::F16 => {
                let f16_data: Vec<half::f16> =
                    cache.iter().map(|&v| half::f16::from_f32(v)).collect();
                let host = ferrite_cuda_core::driver::mem_alloc_host(nbytes)?;
                std::ptr::copy_nonoverlapping(f16_data.as_ptr() as *const u8, host, nbytes);
                ferrite_cuda_core::driver::memcpy_htod_async(
                    gpu_ptr,
                    host,
                    nbytes,
                    device.compute_stream,
                )?;
                ferrite_cuda_core::driver::stream_synchronize(device.compute_stream)?;
                ferrite_cuda_core::driver::mem_free_host(host)?;
            }
            DType::BF16 => {
                let bf16_data: Vec<half::bf16> =
                    cache.iter().map(|&v| half::bf16::from_f32(v)).collect();
                let host = ferrite_cuda_core::driver::mem_alloc_host(nbytes)?;
                std::ptr::copy_nonoverlapping(bf16_data.as_ptr() as *const u8, host, nbytes);
                ferrite_cuda_core::driver::memcpy_htod_async(
                    gpu_ptr,
                    host,
                    nbytes,
                    device.compute_stream,
                )?;
                ferrite_cuda_core::driver::stream_synchronize(device.compute_stream)?;
                ferrite_cuda_core::driver::mem_free_host(host)?;
            }
            _ => anyhow::bail!("unsupported dtype for RoPE cache: {:?}", dtype),
        }

        let cos_sin_cache = GpuTensor::new(gpu_ptr, &[max_pos, rotary_dim], dtype);

        // Build separate cos/sin caches for FA2 fused RoPE (needs contiguous buffers).
        let (cos_cache, sin_cache) = Self::build_separate_cos_sin(
            &cache,
            max_pos,
            rotary_dim,
            dtype,
            device.compute_stream,
        )?;

        Ok(Self {
            cos_sin_cache,
            cos_cache,
            sin_cache,
            head_dim,
        })
    }

    unsafe fn build_separate_cos_sin(
        cache: &[f32],
        max_pos: usize,
        rotary_dim: usize,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<(GpuTensor, GpuTensor)> {
        let half = rotary_dim / 2;
        let half_nbytes = max_pos * half * dtype.size_bytes();
        let cos_data: Vec<f32> = (0..max_pos)
            .flat_map(|p| (0..half).map(move |i| cache[p * rotary_dim + i]))
            .collect();
        let sin_data: Vec<f32> = (0..max_pos)
            .flat_map(|p| (0..half).map(move |i| cache[p * rotary_dim + half + i]))
            .collect();

        let cos_gpu = ferrite_cuda_core::driver::mem_alloc(half_nbytes)?;
        let sin_gpu = ferrite_cuda_core::driver::mem_alloc(half_nbytes)?;
        let host = ferrite_cuda_core::driver::mem_alloc_host(half_nbytes)?;

        macro_rules! upload {
            ($data:expr, $gpu:expr, $T:ty) => {{
                let converted: Vec<$T> = $data.iter().map(|&v| <$T>::from_f32(v)).collect();
                std::ptr::copy_nonoverlapping(converted.as_ptr() as *const u8, host, half_nbytes);
                ferrite_cuda_core::driver::memcpy_htod_async($gpu, host, half_nbytes, stream)?;
            }};
        }

        match dtype {
            DType::F16 => {
                upload!(cos_data, cos_gpu, half::f16);
                upload!(sin_data, sin_gpu, half::f16);
            }
            DType::BF16 => {
                upload!(cos_data, cos_gpu, half::bf16);
                upload!(sin_data, sin_gpu, half::bf16);
            }
            DType::F32 => {
                std::ptr::copy_nonoverlapping(cos_data.as_ptr() as *const u8, host, half_nbytes);
                ferrite_cuda_core::driver::memcpy_htod_async(cos_gpu, host, half_nbytes, stream)?;
                std::ptr::copy_nonoverlapping(sin_data.as_ptr() as *const u8, host, half_nbytes);
                ferrite_cuda_core::driver::memcpy_htod_async(sin_gpu, host, half_nbytes, stream)?;
            }
            _ => anyhow::bail!("unsupported dtype for RoPE cache: {:?}", dtype),
        }
        ferrite_cuda_core::driver::stream_synchronize(stream)?;
        ferrite_cuda_core::driver::mem_free_host(host)?;

        Ok((
            GpuTensor::new(cos_gpu, &[max_pos, half], dtype),
            GpuTensor::new(sin_gpu, &[max_pos, half], dtype),
        ))
    }
}
