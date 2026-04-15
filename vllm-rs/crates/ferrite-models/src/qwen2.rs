// SPDX-License-Identifier: Apache-2.0
//! Qwen2 model architecture via the legacy `forward!{}` macro.
//!
//! Same as Llama except QKV projections include bias terms.
//!
//! Migration to `#[forward]` is blocked on HANDOFF.md Step C
//! (`CublasFusedQkvGemmWithBiasImpl` port — gap #5).
#![allow(clippy::possible_missing_comma)]

#[allow(unused_imports)]
use ferrite_cuda_core::alloc::OwnedTensor;
#[allow(unused_imports)]
use ferrite_cuda_core::device::GpuDevice;
#[allow(unused_imports)]
use ferrite_cuda_core::driver;
#[allow(unused_imports)]
use ferrite_cuda_core::dtype::DType;
#[allow(unused_imports)]
use ferrite_cuda_core::tensor::{GpuTensor, TensorView};
#[allow(unused_imports)]
use ferrite_cuda_core::weights::GpuWeights;

#[allow(unused_imports)]
use ferrite_kernels::attention_helpers;
#[allow(unused_imports)]
use ferrite_kernels::kernels;
#[allow(unused_imports)]
use ferrite_kernels::kv_cache::KvCachePool;
#[allow(unused_imports)]
use ferrite_kernels::layers::{Embedding, Linear, LinearLayer, RmsNorm};
#[allow(unused_imports)]
use ferrite_kernels::rotary::{LlamaConfig, RotaryCache};

// CUTLASS GEMM FFI — extern "C" declarations for solver-selected tile configs.
macro_rules! cutlass_gemm_ffi {
    ($($name:ident),* $(,)?) => {
        #[cfg(feature = "cuda")]
        unsafe extern "C" {
            $(
                pub fn $name(
                    c: *mut u16, a: *const u16, b: *const u16,
                    m: i32, n: i32, k: i32,
                    alpha: f32, beta: f32, stream: u64,
                ) -> i32;
            )*
        }
    };
}

cutlass_gemm_ffi!(
    cutlass_gemm_32x64_s4_launch,
    cutlass_gemm_32x64_s3_launch,
    cutlass_gemm_32x128_s4_launch,
    cutlass_gemm_32x128_s3_launch,
    cutlass_gemm_32x256_s3_launch,
    cutlass_gemm_64x64_s4_launch,
    cutlass_gemm_64x64_s3_launch,
    cutlass_gemm_64x128_s4_launch,
    cutlass_gemm_64x128_s3_launch,
    cutlass_gemm_128x64_s4_launch,
    cutlass_gemm_128x64_s3_launch,
    cutlass_gemm_128x128_s4_launch,
    cutlass_gemm_128x128_s3_launch,
    cutlass_gemm_128x256_s3_launch,
    cutlass_gemm_256x64_s4_launch,
    cutlass_gemm_256x64_s3_launch,
    cutlass_gemv_launch,
    cutlass_gemm_128x128_launch,
    cutlass_gemm_64x64_launch,
);

ferrite_macros::forward! {
    hidden_states = embed(input_ids, embed_tokens);
    for layer in 0..NL {
        let normed = rmsnorm(hidden_states, input_layernorm[layer]);
        let q = gemm(normed, self_attn.q_proj[layer]);
        let q = bias_add(q, self_attn.q_proj.bias[layer]);
        let k = gemm(normed, self_attn.k_proj[layer]);
        let k = bias_add(k, self_attn.k_proj.bias[layer]);
        let v = gemm(normed, self_attn.v_proj[layer]);
        let v = bias_add(v, self_attn.v_proj.bias[layer]);
        let (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
        let attn = attention(q, k, v, kv_cache[layer], block_table);
        let oproj = gemm(attn, self_attn.o_proj[layer]);
        hidden_states = add(oproj, hidden_states);

        let normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
        let gate = silu(gemm(normed2, mlp.gate_proj[layer]));
        let up = gemm(normed2, mlp.up_proj[layer]);
        let down = gemm(gate * up, mlp.down_proj[layer]);
        hidden_states = add(down, hidden_states);
    }
    hidden_states = rmsnorm(hidden_states, norm);
    logits = gemm(hidden_states, lm_head);

    models: [
        // Qwen2.5 0.5B
        { layers: 24, hidden: 896, intermediate: 4864, heads: 14, kv_heads: 2, head_dim: 64, vocab: 151936 },
        // Qwen2.5 1.5B
        { layers: 28, hidden: 1536, intermediate: 8960, heads: 12, kv_heads: 2, head_dim: 128, vocab: 151936 },
        // Qwen2.5 3B
        { layers: 36, hidden: 2048, intermediate: 11008, heads: 16, kv_heads: 2, head_dim: 128, vocab: 151936 },
        // Qwen2.5 7B
        { layers: 28, hidden: 3584, intermediate: 18944, heads: 28, kv_heads: 4, head_dim: 128, vocab: 151936 },
        // Qwen2.5 14B
        { layers: 48, hidden: 5120, intermediate: 13824, heads: 40, kv_heads: 8, head_dim: 128, vocab: 152064 },
        // Qwen2.5 32B
        { layers: 64, hidden: 5120, intermediate: 27648, heads: 40, kv_heads: 8, head_dim: 128, vocab: 152064 },
        // Qwen2.5 72B
        { layers: 80, hidden: 8192, intermediate: 29568, heads: 64, kv_heads: 8, head_dim: 128, vocab: 152064 },
    ],
    target: l4_sm89,
    workloads: [1..4096],
}
