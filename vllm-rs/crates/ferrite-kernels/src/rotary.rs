// SPDX-License-Identifier: Apache-2.0
//! Rotary positional embedding cache and rope scaling config.

use anyhow::Result;

#[cfg(feature = "cuda")]
use ferrite_cuda_core::device::GpuDevice;
use ferrite_cuda_core::dtype::DType;
use ferrite_cuda_core::tensor::GpuTensor;
use ferrite_cuda_core::weights::GpuWeights;
use rayon::prelude::*;

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

/// DeepSeek-V2 YaRN NTK-by-parts scaling parameters.
/// Mirrors the `rope_scaling` subobject in the HF config JSON.
#[derive(Debug, Clone)]
pub struct YarnRopeScaling {
    pub factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub mscale: f64,
    pub mscale_all_dim: f64,
    pub original_max_position_embeddings: usize,
}

#[cfg(feature = "cuda")]
fn yarn_get_mscale(scale: f64, mscale: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

#[cfg(feature = "cuda")]
fn yarn_find_correction_range(
    beta_fast: f64,
    beta_slow: f64,
    dim: usize,
    base: f64,
    original_max_pos: usize,
) -> (f64, f64) {
    let n_orig = original_max_pos as f64;
    let low =
        (n_orig / (beta_fast * 2.0 * std::f64::consts::PI)).ln() / (2.0 / dim as f64 * base.ln());
    let high =
        (n_orig / (beta_slow * 2.0 * std::f64::consts::PI)).ln() / (2.0 / dim as f64 * base.ln());
    (
        low.floor().max(0.0),
        high.ceil().min(dim as f64 / 2.0 - 1.0),
    )
}

#[cfg(feature = "cuda")]
fn yarn_linear_ramp_mask(low: f64, high: f64, dim: usize) -> Vec<f64> {
    let len = dim / 2;
    (0..len)
        .map(|i| {
            let t = i as f64;
            if low >= high {
                if t < low { 0.0 } else { 1.0 }
            } else if t < low {
                0.0
            } else if t > high {
                1.0
            } else {
                (t - low) / (high - low)
            }
        })
        .collect()
}

/// Phi-3 / Phi-3.5 LongRoPE (su-scaling) parameters. Mirrors the
/// fields Python vLLM's `Phi3LongRoPEScaledRotaryEmbedding`
/// consumes 1:1 (see
/// `vllm/model_executor/layers/rotary_embedding/phi3_long_rope_scaled_rope.py`).
///
/// Selection between `short_factor` and `long_factor` is a GLOBAL
/// flag keyed on `max_model_len > original_max_position_embeddings`
/// (not a per-position switchover). The caller supplies
/// `max_model_len` — typically the HF config's
/// `max_position_embeddings` unless the user passes a smaller
/// `--max-model-len` — and the constructor bakes exactly one of
/// (short_factor, short_mscale) or (long_factor, long_mscale) into
/// the cache for every position.
///
/// `short_mscale` / `long_mscale` scale the cos/sin values
/// (equivalent to scaling attention logits). When the HF config
/// omits them, Python falls back to the Phi-3 paper formula
/// `sqrt(1 + ln(max_pos/original_max_pos) / ln(original_max_pos))`
/// for both; the caller must materialize that default here.
#[derive(Debug, Clone)]
pub struct LongRopeScaling {
    pub short_factor: Vec<f64>,
    pub long_factor: Vec<f64>,
    pub original_max_position_embeddings: usize,
    pub short_mscale: f64,
    pub long_mscale: f64,
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
    /// MRoPE section lengths (T/H/W) for arches with multimodal RoPE
    /// (Qwen2-VL, Qwen2.5-VL). `None` for every text-only arch — the
    /// kernel takes the existing 1D-positions fast path. `Some([a,b,c])`
    /// with `a+b+c == rotary_dim/2` selects the MRoPE path: kernel reads
    /// three position values per token and dispatches each rotary pair
    /// through the section that owns it. Plumbed through Phase A1; the
    /// MRoPE math path itself lands in Phase A2. See
    /// `~/.claude/plans/distributed-mapping-map.md`.
    pub mrope_section: Option<[u32; 3]>,
}

/// Cast an f32 cos/sin buffer to `dtype` (F32 / F16 / BF16) and
/// upload via [`GpuWeights::alloc_packed_from_host`]. Backend-neutral
/// — used by [`RotaryCache::new_from_gpuweights`].
fn upload_via_gpuweights(
    weights: &mut GpuWeights,
    cache: &[f32],
    shape: &[usize],
    dtype: DType,
) -> Result<GpuTensor> {
    let bytes: Vec<u8> = match dtype {
        DType::F32 => {
            let mut v = Vec::with_capacity(cache.len() * 4);
            for &f in cache {
                v.extend_from_slice(&f.to_ne_bytes());
            }
            v
        }
        DType::F16 => {
            let mut v = Vec::with_capacity(cache.len() * 2);
            for &f in cache {
                v.extend_from_slice(&half::f16::from_f32(f).to_bits().to_ne_bytes());
            }
            v
        }
        DType::BF16 => {
            let mut v = Vec::with_capacity(cache.len() * 2);
            for &f in cache {
                v.extend_from_slice(&half::bf16::from_f32(f).to_bits().to_ne_bytes());
            }
            v
        }
        _ => anyhow::bail!("unsupported dtype for RoPE cache: {:?}", dtype),
    };
    weights.alloc_packed_from_host(&bytes, shape, dtype)
}

/// Allocate + upload a `[max_pos, rotary_dim]` f32 cos|sin cache to
/// GPU memory in the target dtype. Extracted so variants (standard,
/// LongRoPE, partial) share one upload path.
///
/// # Safety
/// Requires valid CUDA context and stream.
#[cfg(feature = "cuda")]
unsafe fn upload_combined_cos_sin(
    cache: &[f32],
    max_pos: usize,
    rotary_dim: usize,
    dtype: DType,
    stream: cudarc::driver::sys::CUstream,
) -> Result<GpuTensor> {
    let nbytes = max_pos * rotary_dim * dtype.size_bytes();
    let gpu_ptr = ferrite_cuda_core::driver::mem_alloc(nbytes)?;
    let host = ferrite_cuda_core::driver::mem_alloc_host(nbytes)?;
    match dtype {
        DType::F32 => {
            std::ptr::copy_nonoverlapping(cache.as_ptr() as *const u8, host, nbytes);
        }
        DType::F16 => {
            let f16_data: Vec<half::f16> = cache.iter().map(|&v| half::f16::from_f32(v)).collect();
            std::ptr::copy_nonoverlapping(f16_data.as_ptr() as *const u8, host, nbytes);
        }
        DType::BF16 => {
            let bf16_data: Vec<half::bf16> =
                cache.iter().map(|&v| half::bf16::from_f32(v)).collect();
            std::ptr::copy_nonoverlapping(bf16_data.as_ptr() as *const u8, host, nbytes);
        }
        _ => anyhow::bail!("unsupported dtype for RoPE cache: {:?}", dtype),
    }
    ferrite_cuda_core::driver::memcpy_htod_async(gpu_ptr, host, nbytes, stream)?;
    ferrite_cuda_core::driver::stream_synchronize(stream)?;
    ferrite_cuda_core::driver::mem_free_host(host)?;
    Ok(GpuTensor::new(gpu_ptr, &[max_pos, rotary_dim], dtype))
}

impl RotaryCache {
    /// Build the cos/sin cache on GPU.
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    #[cfg(feature = "cuda")]
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
    #[cfg(feature = "cuda")]
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
            mrope_section: None,
        })
    }

    /// Backend-neutral, stream-free counterpart to [`new_from_stream`].
    /// Computes the cos/sin tables on CPU (identical math) and uploads
    /// them via the [`GpuWeights`] active [`DeviceAllocator`]. Single
    /// [`alloc_packed_from_host`] call per cache (combined / cos / sin).
    ///
    /// Currently covers the basic case: full rotary
    /// (`rotary_dim == head_dim`), no scaling or Llama3 scaling. Other
    /// variants (LongRoPE, partial-rotary, YaRN) are still cuda-only —
    /// porting them follows the same pattern but lifts more host-math
    /// helpers; deferred until a metal model needs them.
    ///
    /// [`new_from_stream`]: Self::new_from_stream
    /// [`alloc_packed_from_host`]: GpuWeights::alloc_packed_from_host
    /// [`DeviceAllocator`]: ferrite_cuda_core::DeviceAllocator
    pub fn new_from_gpuweights(
        weights: &mut GpuWeights,
        head_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
    ) -> Result<Self> {
        let rotary_dim = head_dim;
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

        // Parallel cos/sin fill — `max_pos × rotary_dim` is millions of
        // trig calls for long-context models (Llama-3.2 ships
        // max_position_embeddings = 131072). Rayon par_chunks_mut over
        // rows is safe (each pos writes its own row) and turns ~80ms of
        // single-threaded math into ~10ms on a typical M-series core
        // count.
        let mut cache = vec![0f32; max_pos * rotary_dim];
        cache
            .par_chunks_mut(rotary_dim)
            .enumerate()
            .for_each(|(pos, row)| {
                for i in 0..half {
                    let angle = pos as f64 * inv_freqs[i];
                    row[i] = angle.cos() as f32;
                    row[half + i] = angle.sin() as f32;
                }
            });

        let cos_sin_cache = upload_via_gpuweights(weights, &cache, &[max_pos, rotary_dim], dtype)?;

        // Split `cos_cache` / `sin_cache` are wgpu-only — every cuda
        // and metal kernel reads the unified `cos_sin_cache` above.
        // Building + uploading two ~32 MB caches that nothing reads
        // costs ~30-50ms on Llama-3.2-3B at max_pos = 131072. Skip
        // them under the cuda + metal feature combos.
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let (cos_cache, sin_cache) = (
            unsafe { ferrite_cuda_core::GpuTensor::new(std::ptr::null_mut(), &[0usize], dtype) },
            unsafe { ferrite_cuda_core::GpuTensor::new(std::ptr::null_mut(), &[0usize], dtype) },
        );
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let (cos_cache, sin_cache) = {
            let mut cos_data: Vec<f32> = Vec::with_capacity(max_pos * half);
            let mut sin_data: Vec<f32> = Vec::with_capacity(max_pos * half);
            for p in 0..max_pos {
                for i in 0..half {
                    cos_data.push(cache[p * rotary_dim + i]);
                    sin_data.push(cache[p * rotary_dim + half + i]);
                }
            }
            let cos_cache = upload_via_gpuweights(weights, &cos_data, &[max_pos, half], dtype)?;
            let sin_cache = upload_via_gpuweights(weights, &sin_data, &[max_pos, half], dtype)?;
            (cos_cache, sin_cache)
        };

        Ok(Self {
            cos_sin_cache,
            cos_cache,
            sin_cache,
            head_dim,
            mrope_section: None,
        })
    }

    /// Phi-3 / Phi-3.5 LongRoPE (su-scaling) variant of
    /// `new_from_stream`. Builds a unified `[max_pos, rotary_dim]` cache
    /// that selects `short_factor` below
    /// `original_max_position_embeddings` and `long_factor` at or
    /// above it, with `attention_factor` baked into the cos/sin
    /// values. Kernel dispatch is unchanged — the per-position branch
    /// is entirely a property of the cache values.
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    #[cfg(feature = "cuda")]
    pub unsafe fn new_longrope_from_stream(
        head_dim: usize,
        max_pos: usize,
        max_model_len: usize,
        rope_theta: f64,
        longrope: &LongRopeScaling,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        Self::build_longrope(
            head_dim,
            head_dim,
            max_pos,
            max_model_len,
            rope_theta,
            longrope,
            dtype,
            stream,
        )
    }

    /// Shared backing for `new_longrope_from_stream` (full rotary,
    /// `rotary_dim == head_dim`) and `new_partial_longrope_from_stream`
    /// (Phi-4-mini, `rotary_dim < head_dim`). Values are computed in
    /// `f32` end-to-end to mirror Python vLLM's
    /// `Phi3LongRoPEScaledRotaryEmbedding` (`torch.float` = float32):
    /// `inv_freq`, `t`, `freqs = t ⊗ inv_freq`, `cos/sin`, and the
    /// `* mscale` multiply all happen at f32 precision before the
    /// final cast to the target dtype.
    ///
    /// The caller supplies `max_model_len`. The Python side sets
    /// `use_long_rope = max_model_len > original_max_position_embeddings`
    /// as a global flag at init time and threads it into `forward` by
    /// offsetting positions into a concatenated `[short ; long]`
    /// cache. We collapse that to one cache of shape
    /// `[max_pos, rotary_dim]` whose values at each position match
    /// what Python's `index_select(long_short_cache, positions + off)`
    /// returns:
    /// - `use_long_rope = true`  ⇒ cache[pos] uses `long_factor` + `long_mscale` for pos in 0..max_pos
    /// - `use_long_rope = false` ⇒ cache[pos] uses `short_factor` + `short_mscale` for pos in 0..orig_max
    ///   (positions ≥ orig_max are unreachable when max_model_len ≤ orig_max; filled with the
    ///   short-factor extrapolation for safety.)
    #[cfg(feature = "cuda")]
    unsafe fn build_longrope(
        head_dim: usize,
        rotary_dim: usize,
        max_pos: usize,
        max_model_len: usize,
        rope_theta: f64,
        longrope: &LongRopeScaling,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let half = rotary_dim / 2;
        if longrope.short_factor.len() != half || longrope.long_factor.len() != half {
            anyhow::bail!(
                "LongRoPE factor length mismatch: short={}, long={}, expected {} (= rotary_dim/2)",
                longrope.short_factor.len(),
                longrope.long_factor.len(),
                half,
            );
        }
        // Python's `use_long_rope = max_model_len > original_max_position_embeddings`.
        let use_long_rope = max_model_len > longrope.original_max_position_embeddings;
        // Factor + mscale are a PAIR chosen by the flag — Python
        // builds two caches and indexes whichever; we bake the
        // selected one into a single cache.
        let (factor, mscale): (&[f64], f64) = if use_long_rope {
            (&longrope.long_factor, longrope.long_mscale)
        } else {
            (&longrope.short_factor, longrope.short_mscale)
        };

        // Python: `torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim` (float32).
        // Then `base ** exp_arr` (float32). Then `rescale * (...)` (float32).
        // Then `inv_freq = 1.0 / (...)` (float32).
        let base_f32 = rope_theta as f32;
        let rotary_dim_f32 = rotary_dim as f32;
        let inv_freq: Vec<f32> = (0..half)
            .map(|k| {
                let exp = (2 * k) as f32 / rotary_dim_f32;
                let scaled = (factor[k] as f32) * base_f32.powf(exp);
                1.0f32 / scaled
            })
            .collect();

        let mscale_f32 = mscale as f32;

        // Python: `t = torch.arange(max_pos, dtype=torch.float)` (float32).
        // `freqs = t outer inv_freq` (float32).
        // `cos/sin = freqs.cos()/sin() * mscale` (float32).
        // `cache = cat([cos, sin], dim=-1)` — layout matches my
        // `cache[pos, 0..half] = cos`, `cache[pos, half..] = sin`.
        let mut cache = vec![0f32; max_pos * rotary_dim];
        for pos in 0..max_pos {
            let pos_f32 = pos as f32;
            for i in 0..half {
                let angle = pos_f32 * inv_freq[i];
                cache[pos * rotary_dim + i] = angle.cos() * mscale_f32;
                cache[pos * rotary_dim + half + i] = angle.sin() * mscale_f32;
            }
        }

        let cos_sin_cache = upload_combined_cos_sin(&cache, max_pos, rotary_dim, dtype, stream)?;
        let (cos_cache, sin_cache) =
            Self::build_separate_cos_sin(&cache, max_pos, rotary_dim, dtype, stream)?;

        Ok(Self {
            cos_sin_cache,
            cos_cache,
            sin_cache,
            head_dim,
            mrope_section: None,
        })
    }

    /// Phi-4-mini variant: partial rotary (`rotary_dim < head_dim`) +
    /// LongRoPE (su-scaling). Factor vectors are length `rotary_dim/2`;
    /// non-rotary tail dims are passed through by the kernel as in
    /// `new_partial`.
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    #[cfg(feature = "cuda")]
    pub unsafe fn new_partial_longrope_from_stream(
        head_dim: usize,
        rotary_dim: usize,
        max_pos: usize,
        max_model_len: usize,
        rope_theta: f64,
        longrope: &LongRopeScaling,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        Self::build_longrope(
            head_dim,
            rotary_dim,
            max_pos,
            max_model_len,
            rope_theta,
            longrope,
            dtype,
            stream,
        )
    }

    /// Build a YaRN NTK-by-parts RoPE cos/sin cache for DeepSeek-V2 MLA.
    ///
    /// `rope_head_dim` is the head dimension of the rope portion only
    /// (= `qk_rope_head_dim` = 64 for V2-Lite). The cache shape is
    /// `[max_pos, rope_head_dim]` with cos in the first half and sin
    /// in the second half.
    ///
    /// Bakes `yarn_get_mscale(factor, mscale)` into the cos/sin values so
    /// the rotary kernel output is already mscale-corrected, matching
    /// Python's `DeepseekScalingRotaryEmbedding` which multiplies the
    /// rotary output by `self.mscale`.
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    #[cfg(feature = "cuda")]
    pub unsafe fn new_yarn_from_stream(
        rope_head_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        yarn: &YarnRopeScaling,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let half = rope_head_dim / 2;
        let factor = yarn.factor;
        let (low, high) = yarn_find_correction_range(
            yarn.beta_fast,
            yarn.beta_slow,
            rope_head_dim,
            rope_theta,
            yarn.original_max_position_embeddings,
        );
        let ramp = yarn_linear_ramp_mask(low, high, rope_head_dim);
        // Python's DeepseekScalingRotaryEmbedding bakes
        //   mscale = yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)
        // into the cos/sin cache (mscale_all_dim^2 is handled separately in the attention
        // softmax scale). For DeepSeek V2-Lite mscale==mscale_all_dim so this is 1.0.
        let mscale_num = yarn_get_mscale(factor, yarn.mscale);
        let mscale_den = if yarn.mscale_all_dim != 0.0 {
            yarn_get_mscale(factor, yarn.mscale_all_dim)
        } else {
            1.0
        };
        let mscale = (mscale_num / mscale_den) as f32;

        let inv_freqs: Vec<f64> = (0..half)
            .map(|i| {
                let freq = 1.0 / rope_theta.powf(2.0 * i as f64 / rope_head_dim as f64);
                let freq_inter = freq / factor;
                // ramp[i]=0 → original (high-freq, small i), ramp[i]=1 → interpolated (low-freq, large i).
                // Matches Python: inv_freq = interp*ramp + extrap*(1-ramp)
                freq * (1.0 - ramp[i]) + freq_inter * ramp[i]
            })
            .collect();

        let mut cache = vec![0f32; max_pos * rope_head_dim];
        for pos in 0..max_pos {
            for i in 0..half {
                let angle = pos as f64 * inv_freqs[i];
                cache[pos * rope_head_dim + i] = angle.cos() as f32 * mscale;
                cache[pos * rope_head_dim + half + i] = angle.sin() as f32 * mscale;
            }
        }

        let cos_sin_cache = upload_combined_cos_sin(&cache, max_pos, rope_head_dim, dtype, stream)?;
        let (cos_cache, sin_cache) =
            Self::build_separate_cos_sin(&cache, max_pos, rope_head_dim, dtype, stream)?;

        Ok(Self {
            cos_sin_cache,
            cos_cache,
            sin_cache,
            head_dim: rope_head_dim,
            mrope_section: None,
        })
    }

    /// Build cos/sin cache with partial rotary dimension (rotary_dim < head_dim).
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    #[cfg(feature = "cuda")]
    pub unsafe fn new_partial(
        head_dim: usize,
        rotary_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        Self::new_partial_from_stream(
            head_dim,
            rotary_dim,
            max_pos,
            rope_theta,
            llama3_scaling,
            dtype,
            device.compute_stream,
        )
    }

    /// Partial-rotary variant of `new_from_stream` (takes an explicit
    /// CUDA stream). The pos_encoding kernel passes non-rotary tail
    /// elements through unchanged, so the cache only covers the first
    /// `rotary_dim` of each head.
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    #[cfg(feature = "cuda")]
    pub unsafe fn new_partial_from_stream(
        head_dim: usize,
        rotary_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
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

        let cos_sin_cache = upload_combined_cos_sin(&cache, max_pos, rotary_dim, dtype, stream)?;
        let (cos_cache, sin_cache) =
            Self::build_separate_cos_sin(&cache, max_pos, rotary_dim, dtype, stream)?;

        Ok(Self {
            cos_sin_cache,
            cos_cache,
            sin_cache,
            head_dim,
            mrope_section: None,
        })
    }

    #[cfg(feature = "cuda")]
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
