// SPDX-License-Identifier: Apache-2.0
//! LLaMA model architecture via forward!() macro.

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
// Linking happens in vllm-cuda (the final binary crate).
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
        let k = gemm(normed, self_attn.k_proj[layer]);
        let v = gemm(normed, self_attn.v_proj[layer]);
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
        // Llama 3.2 3B (first — used by the solver until `models: runtime` lands)
        { layers: 28, hidden: 3072, intermediate: 8192, heads: 24, kv_heads: 8, head_dim: 128, vocab: 128256 },
        // Llama 3.2 1B
        { layers: 16, hidden: 2048, intermediate: 8192, heads: 32, kv_heads: 8, head_dim: 64, vocab: 128256 },
        // Llama 3.1 8B
        { layers: 32, hidden: 4096, intermediate: 14336, heads: 32, kv_heads: 8, head_dim: 128, vocab: 128256 },
        // Llama 3.1 70B
        { layers: 80, hidden: 8192, intermediate: 28672, heads: 64, kv_heads: 8, head_dim: 128, vocab: 128256 },
    ],
    target: l4_sm89,
    workloads: [1..4096],
}
