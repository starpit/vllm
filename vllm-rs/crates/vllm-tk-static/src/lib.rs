// SPDX-License-Identifier: Apache-2.0
//! Static megakernel for LLaMA decode/prefill (sm89), with multi-variant support.
//!
//! Multiple model dimension variants are compiled at build time.
//! At runtime, `KernelVariant::from_dims()` selects the correct compiled kernel
//! based on the model's (HD, ID, HDM, NAH, NKH) dimensions.
//!
//! NL (num_layers) is a runtime parameter — models with different layer counts
//! but the same dimension tuple share a kernel variant.

#[cfg(feature = "cuda")]
mod ffi;

// The megakernel! proc-macro generates typed handles, TkTensorArg, newtypes,
// and LaunchArgs (baked to 1B dims for compile-time verification in tests).
vllm_tk_macros::megakernel! {
    kernel llama_sm89<NL=16, HD=2048, ID=8192, HDM=64, NAH=32, NKH=8, VS=128256> {
        for layer in 0..NL {
            let normed = rmsnorm(hidden_states, attn_norm[layer]);
            let qkv = gemm(normed, qkv_weights[layer]);
            let (q, k, v) = rope_append(qkv, positions, kv_cache[layer]);
            let attn = attention_decode(q, k, v, kv_cache[layer], block_table);
            hidden_states = gemm_add(attn, o_proj[layer], hidden_states);

            let normed2 = rmsnorm(hidden_states, mlp_norm[layer]);
            let gate = silu(gemm(normed2, gate_weights[layer]));
            let up = gemm(normed2, up_weights[layer]);
            hidden_states = gemm_add(gate * up, down_proj[layer], hidden_states);
        }
        let normed = rmsnorm(hidden_states, lm_head_norm);
        logits = gemm(normed, lm_head);
    }
}

/// A compiled kernel variant selected at runtime based on model dimensions.
///
/// Each variant is specialized on (HD, ID, HDM) — dimensions that appear in
/// TK's GL type parameters (shared memory tile shapes). NL (num_layers) is
/// runtime. Other dims (NAH, NKH, VS) are baked per variant since they feed
/// into constexpr tile count calculations.
///
/// The specialization key is (HD, HDM) since intermediate_dim, NAH, NKH are
/// determined by the architecture for a given (HD, HDM) pair.
#[derive(Debug, Clone, Copy)]
pub enum KernelVariant {
    /// Llama 1B: HD=2048, HDM=64
    Hd2048Hdm64,
    /// Llama 8B: HD=4096, HDM=128
    Hd4096Hdm128,
    // NOTE: Llama 3B (GQA_RATIO=3) not supported by attention kernel (needs 4 or 8).
    // NOTE: 70B/405B exceed sm89 shared memory budget.
}

/// Supported variant configs: (HD, ID, HDM, NAH, NKH, description).
const SUPPORTED_VARIANTS: &[(usize, usize, usize, usize, usize, &str)] = &[
    (2048, 8192, 64, 32, 8, "Llama 1B"),
    (4096, 14336, 128, 32, 8, "Llama 8B"),
];

impl KernelVariant {
    /// Select the compiled kernel variant matching the given model dimensions.
    ///
    /// Matches on the full (HD, ID, HDM, NAH, NKH) tuple to ensure correctness.
    /// Returns an error listing supported configurations if no match exists.
    pub fn from_dims(
        hidden_dim: usize,
        intermediate_dim: usize,
        head_dim: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
    ) -> Result<Self, String> {
        match (
            hidden_dim,
            intermediate_dim,
            head_dim,
            num_attention_heads,
            num_kv_heads,
        ) {
            (2048, 8192, 64, 32, 8) => Ok(Self::Hd2048Hdm64),
            (4096, 14336, 128, 32, 8) => Ok(Self::Hd4096Hdm128),
            _ => {
                let mut msg = format!(
                    "no compiled TK kernel variant for dims (HD={hidden_dim}, ID={intermediate_dim}, \
                     HDM={head_dim}, NAH={num_attention_heads}, NKH={num_kv_heads}). \
                     Supported variants:\n"
                );
                for &(hd, id, hdm, nah, nkh, desc) in SUPPORTED_VARIANTS {
                    msg.push_str(&format!(
                        "  - HD={hd}, ID={id}, HDM={hdm}, NAH={nah}, NKH={nkh} ({desc})\n"
                    ));
                }
                msg.push_str("To add support, add a new entry to the variant table in vllm-tk-static/build.rs");
                Err(msg)
            }
        }
    }

    /// Launch the decode kernel for this variant.
    ///
    /// # Safety
    /// All TkTensorArg pointers must point to valid GPU memory with correct shapes.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_decode(
        &self,
        ffi_args: &[TkTensorArg; 33],
        attn_scale: f32,
        rms_norm_eps: f32,
        num_pages: i32,
        batch_size: i32,
        num_prefill_tokens: i32,
        num_layers: i32,
        stream: u64,
    ) -> i32 {
        let f = self.decode_fn();
        unsafe {
            f(
                ffi_args[0],
                ffi_args[1],
                ffi_args[2],
                ffi_args[3],
                ffi_args[4],
                ffi_args[5],
                ffi_args[6],
                ffi_args[7],
                ffi_args[8],
                ffi_args[9],
                ffi_args[10],
                ffi_args[11],
                ffi_args[12],
                ffi_args[13],
                ffi_args[14],
                ffi_args[15],
                ffi_args[16],
                ffi_args[17],
                ffi_args[18],
                ffi_args[19],
                ffi_args[20],
                ffi_args[21],
                ffi_args[22],
                ffi_args[23],
                ffi_args[24],
                ffi_args[25],
                ffi_args[26],
                ffi_args[27],
                ffi_args[28],
                ffi_args[29],
                ffi_args[30],
                ffi_args[31],
                ffi_args[32],
                attn_scale,
                rms_norm_eps,
                num_pages,
                batch_size,
                num_prefill_tokens,
                num_layers,
                stream,
            )
        }
    }

    /// Launch the prefill kernel for this variant.
    ///
    /// # Safety
    /// All TkTensorArg pointers must point to valid GPU memory with correct shapes.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_prefill(
        &self,
        ffi_args: &[TkTensorArg; 33],
        attn_scale: f32,
        rms_norm_eps: f32,
        num_pages: i32,
        batch_size: i32,
        num_prefill_tokens: i32,
        num_layers: i32,
        stream: u64,
        seq_chunk_lens: *const i32,
        seq_extend_offsets: *const i32,
    ) -> i32 {
        let f = self.prefill_fn();
        unsafe {
            f(
                ffi_args[0],
                ffi_args[1],
                ffi_args[2],
                ffi_args[3],
                ffi_args[4],
                ffi_args[5],
                ffi_args[6],
                ffi_args[7],
                ffi_args[8],
                ffi_args[9],
                ffi_args[10],
                ffi_args[11],
                ffi_args[12],
                ffi_args[13],
                ffi_args[14],
                ffi_args[15],
                ffi_args[16],
                ffi_args[17],
                ffi_args[18],
                ffi_args[19],
                ffi_args[20],
                ffi_args[21],
                ffi_args[22],
                ffi_args[23],
                ffi_args[24],
                ffi_args[25],
                ffi_args[26],
                ffi_args[27],
                ffi_args[28],
                ffi_args[29],
                ffi_args[30],
                ffi_args[31],
                ffi_args[32],
                attn_scale,
                rms_norm_eps,
                num_pages,
                batch_size,
                num_prefill_tokens,
                num_layers,
                stream,
                seq_chunk_lens,
                seq_extend_offsets,
            )
        }
    }

    #[cfg(feature = "cuda")]
    fn decode_fn(&self) -> ffi::DecodeLaunchFn {
        match self {
            Self::Hd2048Hdm64 => ffi::llama_sm89_hd2048_hdm64_decode_static_launch,
            Self::Hd4096Hdm128 => ffi::llama_sm89_hd4096_hdm128_decode_static_launch,
        }
    }

    #[cfg(feature = "cuda")]
    fn prefill_fn(&self) -> ffi::PrefillLaunchFn {
        match self {
            Self::Hd2048Hdm64 => ffi::llama_sm89_hd2048_hdm64_prefill_static_launch,
            Self::Hd4096Hdm128 => ffi::llama_sm89_hd4096_hdm128_prefill_static_launch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_exists_with_correct_dims() {
        assert_eq!(MegakernelLlamaSm89::NL, 16);
        assert_eq!(MegakernelLlamaSm89::HD, 2048);
        assert_eq!(MegakernelLlamaSm89::ID, 8192);
        assert_eq!(MegakernelLlamaSm89::QKV_DIM, 3072);
        assert_eq!(MegakernelLlamaSm89::NUM_OPS, 13);
    }

    #[test]
    fn pipeline_diagram_not_empty() {
        assert!(!MegakernelLlamaSm89::PIPELINE_DIAGRAM.is_empty());
    }

    #[test]
    fn cuda_source_has_both_kernels() {
        let src = MegakernelLlamaSm89::CUDA_SOURCE;
        assert!(src.contains("llama_sm89_decode_static"));
        assert!(src.contains("llama_sm89_prefill_static"));
        assert!(src.contains("OPCODE_GQA_AttentionDecode"));
        assert!(src.contains("OPCODE_GQA_AttentionPrefill"));
    }

    #[test]
    fn kernel_variant_from_dims() {
        // 1B
        assert!(matches!(
            KernelVariant::from_dims(2048, 8192, 64, 32, 8),
            Ok(KernelVariant::Hd2048Hdm64)
        ));
        // 3B — GQA_RATIO=3 not supported by attention kernel
        assert!(KernelVariant::from_dims(3072, 8192, 128, 24, 8).is_err());
        // 8B
        assert!(matches!(
            KernelVariant::from_dims(4096, 14336, 128, 32, 8),
            Ok(KernelVariant::Hd4096Hdm128)
        ));
        // 70B — exceeds sm89 shared memory, should fail
        assert!(KernelVariant::from_dims(8192, 28672, 128, 64, 8).is_err());
        // Unsupported
        let err = KernelVariant::from_dims(1024, 4096, 64, 16, 4);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("no compiled TK kernel variant"));
    }

    #[test]
    fn typed_handles_exist() {
        let _: fn(&GpuActivation<2048>) = |_| {};
        let _: fn(&GpuActivationBig<8192>) = |_| {};
        let _: fn(&GpuWeight<3072, 2048>) = |_| {};
        let _: fn(&GpuWeightBig<2048, 8192>) = |_| {};
        let _: fn(&GpuNormWeight<2048>) = |_| {};
        let _: fn(&GpuLogits<128256>) = |_| {};
        let _: fn(&GpuKvCache) = |_| {};
        let _: fn(&GpuRopeTable<64>) = |_| {};
        let _: fn(&GpuMetaVec) = |_| {};
        let _: fn(&GpuBarrier) = |_| {};
        let _: fn(&GpuVmLayout) = |_| {};
    }

    #[test]
    fn tk_tensor_arg_new() {
        let arg1d = TkTensorArg::new(0xDEAD, &[128]);
        assert_eq!(arg1d.ptr, 0xDEAD);
        assert_eq!((arg1d.b, arg1d.d, arg1d.r, arg1d.c), (1, 1, 1, 128));

        let arg2d = TkTensorArg::new(0xBEEF, &[32, 2048]);
        assert_eq!((arg2d.b, arg2d.d, arg2d.r, arg2d.c), (1, 1, 32, 2048));
    }

    #[test]
    fn cuda_source_has_runtime_num_layers() {
        let src = MegakernelLlamaSm89::CUDA_SOURCE;
        // NL should NOT be a constexpr
        assert!(!src.contains("static constexpr int NL"));
        // Loop should use runtime num_layers
        assert!(src.contains("for (int layer = 0; layer < num_layers; layer++)"));
        // Launch wrapper should accept num_layers
        assert!(src.contains("int num_layers"));
    }
}
