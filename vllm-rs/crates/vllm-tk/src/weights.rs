// SPDX-License-Identifier: Apache-2.0
//! Weight loading for the TK KVM megakernel.
//!
//! Loads HuggingFace LLaMA safetensors into the stacked layout expected by TK:
//! weights are `[1, num_layers, output_dim, input_dim]` (the GL `d` dim = num_layers).

#[cfg(feature = "cuda")]
mod inner {
    use anyhow::Result;
    use cudarc::driver::sys::CUstream;
    use vllm_cuda::model::llama::LlamaConfig;
    use vllm_cuda::{DType, GpuDevice, GpuTensor, GpuWeights, driver};

    /// All GPU tensors needed to launch the TK megakernel.
    pub struct TkWeights {
        // Stacked projection weights [1, L, tiles, input_dim] bf16
        pub qkv_proj: GpuTensor,
        pub o_proj: GpuTensor,
        pub gate_proj: GpuTensor,
        pub up_proj: GpuTensor,
        pub down_proj: GpuTensor,

        // Stacked norm weights [1, 1, L, hidden_dim] bf16
        pub attn_norm: GpuTensor,
        pub mlp_norm: GpuTensor,

        // LM head (single layer)
        pub lm_head_norm: GpuTensor, // [1, 1, 1, hidden_dim]
        pub lm_head: GpuTensor,      // [1, 1, vocab_size, hidden_dim]

        // Embedding (for token gather before megakernel)
        pub embed_tokens: GpuTensor, // [vocab_size, hidden_dim]

        // RoPE tables — separate cos and sin, each [1, 1, max_pos, head_dim] f32
        // TK layout: cos[pos, i] = cos[pos, i+half] = cos(pos * freq_i)
        pub rope_cos: GpuTensor,
        pub rope_sin: GpuTensor,

        // Ownership of GPU memory backing the tensors above.
        // Dropped last → frees GPU memory when TkWeights is dropped.
        _gpu_allocs: Vec<vllm_cuda::alloc::RawGpuMem>,
    }

    impl TkWeights {
        /// Load LLaMA weights from safetensors into TK's stacked layout.
        ///
        /// # Safety
        /// Requires valid CUDA context and stream.
        pub unsafe fn load(
            weights: &mut GpuWeights,
            config: &LlamaConfig,
            device: &GpuDevice,
        ) -> Result<Self> {
            let stream = device.compute_stream;
            let nl = config.num_hidden_layers;
            let hd = config.hidden_size;
            let id = config.intermediate_size;
            let hdm = config.head_dim;
            let nah = config.num_attention_heads;
            let nkh = config.num_kv_heads;
            let vs = config.vocab_size;

            let dtype = DType::BF16;
            let elem = dtype.size_bytes();

            // nbh = total QKV heads (Q + K + V)
            let nbh = nah + 2 * nkh;
            let qkv_out = nbh * hdm; // total QKV output dim
            let q_out = nah * hdm;
            let kv_out = nkh * hdm;

            // ── QKV projection: fuse Q, K, V per layer, stack across layers ──
            // Layout: [1, L, qkv_out, hd] where within each layer:
            //   rows 0..q_out = Q, q_out..q_out+kv_out = K, q_out+kv_out.. = V
            let qkv_layer_bytes = qkv_out * hd * elem;
            let qkv_total = nl * qkv_layer_bytes;
            let qkv_ptr = unsafe { driver::mem_alloc(qkv_total)? };
            weights.record_alloc(qkv_ptr, qkv_total);
            let q_bytes = q_out * hd * elem;
            let k_bytes = kv_out * hd * elem;
            for i in 0..nl {
                let base = unsafe { qkv_ptr.add(i * qkv_layer_bytes) };
                unsafe {
                    weights.take_into(
                        &format!("model.layers.{i}.self_attn.q_proj.weight"),
                        base,
                        stream,
                    )?;
                    weights.take_into(
                        &format!("model.layers.{i}.self_attn.k_proj.weight"),
                        base.add(q_bytes),
                        stream,
                    )?;
                    weights.take_into(
                        &format!("model.layers.{i}.self_attn.v_proj.weight"),
                        base.add(q_bytes + k_bytes),
                        stream,
                    )?;
                }
            }
            let qkv_proj = unsafe { GpuTensor::new(qkv_ptr, &[1, nl, qkv_out, hd], dtype) };

            // ── O projection: [1, L, hd, hd] ──
            let o_proj = unsafe {
                Self::stack_layers(
                    weights,
                    "self_attn.o_proj.weight",
                    nl,
                    hd,
                    hd,
                    dtype,
                    stream,
                )?
            };

            // ── Gate projection: [1, L, id, hd] ──
            let gate_proj = unsafe {
                Self::stack_layers(weights, "mlp.gate_proj.weight", nl, id, hd, dtype, stream)?
            };

            // ── Up projection: [1, L, id, hd] ──
            let up_proj = unsafe {
                Self::stack_layers(weights, "mlp.up_proj.weight", nl, id, hd, dtype, stream)?
            };

            // ── Down projection: [1, L, hd, id] ──
            let down_proj = unsafe {
                Self::stack_layers(weights, "mlp.down_proj.weight", nl, hd, id, dtype, stream)?
            };

            // ── Attention norm weights: [1, 1, L, hd] ──
            let attn_norm = unsafe {
                Self::stack_1d(weights, "input_layernorm.weight", nl, hd, dtype, stream)?
            };

            // ── MLP norm weights: [1, 1, L, hd] ──
            let mlp_norm = unsafe {
                Self::stack_1d(
                    weights,
                    "post_attention_layernorm.weight",
                    nl,
                    hd,
                    dtype,
                    stream,
                )?
            };

            // ── LM head norm: single [1, 1, 1, hd] ──
            let lm_head_norm_raw = weights.take("model.norm.weight")?;
            let lm_head_norm = unsafe {
                GpuTensor::new(lm_head_norm_raw.as_mut_ptr::<u8>(), &[1, 1, 1, hd], dtype)
            };

            // ── Embed tokens + LM head ──
            // For tied embeddings, both share the same GPU allocation.
            let embed_raw = weights.take("model.embed_tokens.weight")?;
            let embed_ptr = embed_raw.as_mut_ptr::<u8>();
            let embed_tokens = unsafe { GpuTensor::new(embed_ptr, &[vs, hd], dtype) };

            let lm_head = if config.tie_word_embeddings {
                unsafe { GpuTensor::new(embed_ptr, &[1, 1, vs, hd], dtype) }
            } else {
                let t = weights.take("lm_head.weight")?;
                unsafe { GpuTensor::new(t.as_mut_ptr::<u8>(), &[1, 1, vs, hd], dtype) }
            };

            // ── RoPE tables ──
            let (rope_cos, rope_sin) = unsafe { Self::build_rope_tables(config, device)? };

            // Take ownership of all GPU allocations so they outlive the GpuWeights.
            let gpu_allocs = weights.take_gpu_allocs();

            Ok(Self {
                qkv_proj,
                o_proj,
                gate_proj,
                up_proj,
                down_proj,
                attn_norm,
                mlp_norm,
                lm_head_norm,
                lm_head,
                embed_tokens,
                rope_cos,
                rope_sin,
                _gpu_allocs: gpu_allocs,
            })
        }

        /// Stack a 2D weight across all layers: `model.layers.{i}.{suffix}` → `[1, L, rows, cols]`.
        unsafe fn stack_layers(
            weights: &mut GpuWeights,
            suffix: &str,
            nl: usize,
            rows: usize,
            cols: usize,
            dtype: DType,
            stream: CUstream,
        ) -> Result<GpuTensor> {
            let elem = dtype.size_bytes();
            let layer_bytes = rows * cols * elem;
            let total = nl * layer_bytes;
            let ptr = unsafe { driver::mem_alloc(total)? };
            weights.record_alloc(ptr, total);
            for i in 0..nl {
                unsafe {
                    weights.take_into(
                        &format!("model.layers.{i}.{suffix}"),
                        ptr.add(i * layer_bytes),
                        stream,
                    )?;
                }
            }
            Ok(unsafe { GpuTensor::new(ptr, &[1, nl, rows, cols], dtype) })
        }

        /// Stack a 1D weight (norms) across all layers: → `[1, 1, L, dim]`.
        unsafe fn stack_1d(
            weights: &mut GpuWeights,
            suffix: &str,
            nl: usize,
            dim: usize,
            dtype: DType,
            stream: CUstream,
        ) -> Result<GpuTensor> {
            let elem = dtype.size_bytes();
            let layer_bytes = dim * elem;
            let total = nl * layer_bytes;
            let ptr = unsafe { driver::mem_alloc(total)? };
            weights.record_alloc(ptr, total);
            for i in 0..nl {
                unsafe {
                    weights.take_into(
                        &format!("model.layers.{i}.{suffix}"),
                        ptr.add(i * layer_bytes),
                        stream,
                    )?;
                }
            }
            Ok(unsafe { GpuTensor::new(ptr, &[1, 1, nl, dim], dtype) })
        }

        /// Build TK-format RoPE tables: separate cos and sin, each `[1, 1, max_pos, head_dim]` f32.
        ///
        /// TK layout: `cos[pos, i] = cos[pos, i + half] = cos(pos * inv_freq[i])` —
        /// each half-head-dim value is duplicated in first and second half.
        pub(crate) unsafe fn build_rope_tables(
            config: &LlamaConfig,
            device: &GpuDevice,
        ) -> Result<(GpuTensor, GpuTensor)> {
            let hdm = config.head_dim;
            let half = hdm / 2;
            let max_pos = config.max_position_embeddings;
            let rope_theta = config.rope_theta;

            // Compute inverse frequencies with optional Llama3 scaling.
            let inv_freqs: Vec<f64> = (0..half)
                .map(|i| {
                    let freq = 1.0 / rope_theta.powf(2.0 * i as f64 / hdm as f64);
                    if let Some(ref scaling) = config.llama3_rope_scaling {
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

            // Build CPU tables: cos[pos, i] = cos[pos, i+half] = cos(pos * inv_freq[i])
            let n = max_pos * hdm;
            let mut cos_buf = vec![0f32; n];
            let mut sin_buf = vec![0f32; n];
            for pos in 0..max_pos {
                for i in 0..half {
                    let angle = pos as f64 * inv_freqs[i];
                    let c = angle.cos() as f32;
                    let s = angle.sin() as f32;
                    cos_buf[pos * hdm + i] = c;
                    cos_buf[pos * hdm + half + i] = c;
                    sin_buf[pos * hdm + i] = s;
                    sin_buf[pos * hdm + half + i] = s;
                }
            }

            // Upload to GPU as f32.
            let nbytes = n * 4;
            let stream = device.compute_stream;

            let cos_ptr = unsafe { driver::mem_alloc(nbytes)? };
            let cos_host = unsafe { driver::mem_alloc_host(nbytes)? };
            unsafe {
                std::ptr::copy_nonoverlapping(cos_buf.as_ptr() as *const u8, cos_host, nbytes);
                driver::memcpy_htod_async(cos_ptr, cos_host, nbytes, stream)?;
            }

            let sin_ptr = unsafe { driver::mem_alloc(nbytes)? };
            let sin_host = unsafe { driver::mem_alloc_host(nbytes)? };
            unsafe {
                std::ptr::copy_nonoverlapping(sin_buf.as_ptr() as *const u8, sin_host, nbytes);
                driver::memcpy_htod_async(sin_ptr, sin_host, nbytes, stream)?;
                driver::stream_synchronize(stream)?;
                driver::mem_free_host(cos_host)?;
                driver::mem_free_host(sin_host)?;
            }

            let cos = unsafe { GpuTensor::new(cos_ptr, &[1, 1, max_pos, hdm], DType::F32) };
            let sin = unsafe { GpuTensor::new(sin_ptr, &[1, 1, max_pos, hdm], DType::F32) };

            Ok((cos, sin))
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::TkWeights;

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::TkWeights;
    use vllm_cuda::model::llama::LlamaConfig;
    use vllm_cuda::{DType, GpuDevice, GpuWeights};

    const LLAMA_1B_DIR: &str = "/root/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c";

    fn llama_1b_config() -> LlamaConfig {
        LlamaConfig {
            hidden_size: 2048,
            num_attention_heads: 32,
            num_kv_heads: 8,
            num_hidden_layers: 16,
            intermediate_size: 8192,
            vocab_size: 128256,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            head_dim: 64,
            tie_word_embeddings: true,
            llama3_rope_scaling: None,
        }
    }

    #[test]
    #[ignore] // requires CUDA GPU + model weights on disk
    fn test_tk_weight_loading_shapes() {
        let device = GpuDevice::new(0).expect("GpuDevice::new");
        let mut weights =
            GpuWeights::from_dir(LLAMA_1B_DIR, device.compute_stream).expect("load safetensors");
        weights.set_target_dtype(DType::BF16);
        weights.start_precast();

        let config = llama_1b_config();
        let tk =
            unsafe { TkWeights::load(&mut weights, &config, &device).expect("TkWeights::load") };

        let nl = 16usize;
        let hd = 2048usize;
        let id = 8192usize;
        let hdm = 64usize;
        let nah = 32usize;
        let nkh = 8usize;
        let vs = 128256usize;
        let nbh = nah + 2 * nkh; // 48
        let max_pos = 131072usize;

        // QKV: [1, L, nbh*hdm, hd] = [1, 16, 3072, 2048]
        assert_eq!(
            tk.qkv_proj.shape(),
            &[1, nl as u32, (nbh * hdm) as u32, hd as u32]
        );

        // O: [1, L, hd, hd] = [1, 16, 2048, 2048]
        assert_eq!(tk.o_proj.shape(), &[1, nl as u32, hd as u32, hd as u32]);

        // Gate: [1, L, id, hd] = [1, 16, 8192, 2048]
        assert_eq!(tk.gate_proj.shape(), &[1, nl as u32, id as u32, hd as u32]);

        // Up: [1, L, id, hd] = [1, 16, 8192, 2048]
        assert_eq!(tk.up_proj.shape(), &[1, nl as u32, id as u32, hd as u32]);

        // Down: [1, L, hd, id] = [1, 16, 2048, 8192]
        assert_eq!(tk.down_proj.shape(), &[1, nl as u32, hd as u32, id as u32]);

        // Norms: [1, 1, L, hd] = [1, 1, 16, 2048]
        assert_eq!(tk.attn_norm.shape(), &[1, 1, nl as u32, hd as u32]);
        assert_eq!(tk.mlp_norm.shape(), &[1, 1, nl as u32, hd as u32]);

        // LM head norm: [1, 1, 1, hd]
        assert_eq!(tk.lm_head_norm.shape(), &[1, 1, 1, hd as u32]);

        // LM head: [1, 1, vs, hd] (tied with embed_tokens)
        assert_eq!(tk.lm_head.shape(), &[1, 1, vs as u32, hd as u32]);

        // Embed tokens: [vs, hd]
        assert_eq!(tk.embed_tokens.shape(), &[vs as u32, hd as u32]);

        // Tied: lm_head and embed_tokens share the same pointer
        assert_eq!(
            tk.lm_head.as_ptr::<u8>() as usize,
            tk.embed_tokens.as_ptr::<u8>() as usize,
            "tied embeddings should share the same GPU pointer"
        );

        // RoPE: [1, 1, max_pos, hdm] f32
        assert_eq!(tk.rope_cos.shape(), &[1, 1, max_pos as u32, hdm as u32]);
        assert_eq!(tk.rope_sin.shape(), &[1, 1, max_pos as u32, hdm as u32]);
    }

    #[test]
    #[ignore] // requires CUDA GPU + model weights on disk
    fn test_tk_weight_loading_nonzero() {
        // Verify that loaded weights are not all zeros (data actually transferred).
        let device = GpuDevice::new(0).expect("GpuDevice::new");
        let mut weights =
            GpuWeights::from_dir(LLAMA_1B_DIR, device.compute_stream).expect("load safetensors");
        weights.set_target_dtype(DType::BF16);
        weights.start_precast();

        let config = llama_1b_config();
        let tk =
            unsafe { TkWeights::load(&mut weights, &config, &device).expect("TkWeights::load") };

        // Read first 64 bytes of qkv_proj back to host — should not be all zeros.
        let check_bytes = 64usize;
        let mut host_buf = vec![0u8; check_bytes];
        unsafe {
            vllm_cuda::driver::memcpy_dtoh_async(
                host_buf.as_mut_ptr(),
                tk.qkv_proj.as_ptr::<u8>() as *const u8,
                check_bytes,
                device.compute_stream,
            )
            .expect("memcpy_dtoh");
            vllm_cuda::driver::stream_synchronize(device.compute_stream).unwrap();
        }
        assert!(
            host_buf.iter().any(|&b| b != 0),
            "qkv_proj should contain non-zero data after loading real weights"
        );

        // Same check for attn_norm (1D stacked weights).
        let mut norm_buf = vec![0u8; check_bytes];
        unsafe {
            vllm_cuda::driver::memcpy_dtoh_async(
                norm_buf.as_mut_ptr(),
                tk.attn_norm.as_ptr::<u8>() as *const u8,
                check_bytes,
                device.compute_stream,
            )
            .expect("memcpy_dtoh");
            vllm_cuda::driver::stream_synchronize(device.compute_stream).unwrap();
        }
        assert!(
            norm_buf.iter().any(|&b| b != 0),
            "attn_norm should contain non-zero data after loading real weights"
        );
    }

    #[test]
    #[ignore] // requires CUDA GPU
    fn test_tk_rope_tables() {
        // Verify rope_cos and rope_sin have correct structure:
        // cos[pos, i] == cos[pos, i + half] for all i < half (duplicated layout).
        let device = GpuDevice::new(0).expect("GpuDevice::new");
        let config = LlamaConfig {
            hidden_size: 2048,
            num_attention_heads: 32,
            num_kv_heads: 8,
            num_hidden_layers: 1, // doesn't matter for rope
            intermediate_size: 8192,
            vocab_size: 128256,
            max_position_embeddings: 256, // small for test speed
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            head_dim: 64,
            tie_word_embeddings: true,
            llama3_rope_scaling: None,
        };

        let (cos, sin) =
            unsafe { TkWeights::build_rope_tables(&config, &device).expect("build_rope_tables") };

        assert_eq!(cos.shape(), &[1, 1, 256, 64]);
        assert_eq!(sin.shape(), &[1, 1, 256, 64]);

        // Read a few positions back and verify duplication.
        let hdm = 64usize;
        let half = hdm / 2;
        let read_positions = 4usize;
        let read_elems = read_positions * hdm;
        let mut cos_host = vec![0f32; read_elems];
        unsafe {
            vllm_cuda::driver::memcpy_dtoh_async(
                cos_host.as_mut_ptr() as *mut u8,
                cos.as_ptr::<u8>() as *const u8,
                read_elems * 4,
                device.compute_stream,
            )
            .expect("memcpy_dtoh");
            vllm_cuda::driver::stream_synchronize(device.compute_stream).unwrap();
        }

        for pos in 0..read_positions {
            for i in 0..half {
                let first = cos_host[pos * hdm + i];
                let second = cos_host[pos * hdm + half + i];
                assert!(
                    (first - second).abs() < 1e-6,
                    "cos[{pos},{i}]={first} != cos[{pos},{}]={second}",
                    half + i
                );
            }
        }

        // Verify pos=0 has cos=1.0 for all dims (cos(0*freq) = 1).
        for i in 0..hdm {
            assert!(
                (cos_host[i] - 1.0).abs() < 1e-6,
                "cos[0,{i}]={} expected 1.0",
                cos_host[i]
            );
        }
    }
}
