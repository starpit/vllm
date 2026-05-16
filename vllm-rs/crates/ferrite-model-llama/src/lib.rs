// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! LLaMA — the math. The `#[forward]` attribute macro reads the
//! body below, finds `crates/ferrite-model-llama/configs/` by walking up
//! from this crate, and for every config JSON in it emits
//! specialized `Weights` + `forward` under `ferrite_models::llama`.

use ferrite_forward_macro::forward;

#[forward(
    workloads = [1, 8, 64, 512, 4096],
    sk_buckets = [128, 512, 2048, 8192],
)]
fn llama() {
    hidden_states = embed(input_ids, embed_tokens);
    for layer in 0..num_hidden_layers {
        normed = rmsnorm(hidden_states, input_layernorm[layer]);
        q = gemm(normed, self_attn.q_proj[layer]);
        k = gemm(normed, self_attn.k_proj[layer]);
        v = gemm(normed, self_attn.v_proj[layer]);
        (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
        attn = attention(q, k, v, kv_cache[layer], block_table);
        oproj = gemm(attn, self_attn.o_proj[layer]);
        hidden_states = add(oproj, hidden_states);

        normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
        gate = silu(gemm(normed2, mlp.gate_proj[layer]));
        up = gemm(normed2, mlp.up_proj[layer]);
        down = gemm(gate * up, mlp.down_proj[layer]);
        hidden_states = add(down, hidden_states);
    }
    normed = rmsnorm(hidden_states, norm);
    logits = gemm(normed, lm_head);
}

#[cfg(all(test, feature = "metal"))]
mod metal_emission_tests {
    /// Sanity-check that the macro emits the per-canonical metal
    /// surface (real Weights struct + accessor methods + load fn +
    /// METAL_BUCKETS static + metal_pool fn) for at least one model
    /// in this arch. The check is structural — it doesn't run the
    /// loader, just asserts the symbols exist and resolve to the
    /// expected types.
    #[test]
    fn tinyllama_metal_symbols_resolve() {
        // METAL_BUCKETS is a non-empty `&[MetalBucketSpec<Weights>]`.
        let buckets: &[::ferrite_forward::interpreter::metal::MetalBucketSpec<
            crate::tinyllama_1_1b::Weights,
        >] = crate::tinyllama_1_1b::METAL_BUCKETS;
        assert!(
            !buckets.is_empty(),
            "TinyLlama-1.1B emits at least one bucket"
        );
        // metal_pool resolves as an `fn(...) -> Result<MetalWorkerPool, PoolBuildError>`.
        // We don't call it (no Device available in unit-test ctx); just
        // taking the fn-pointer proves the symbol + signature compiled.
        let _ctor: fn(
            ::std::sync::Arc<::ferrite_forward::interpreter::metal::__re::Device>,
            &crate::tinyllama_1_1b::Weights,
            ::std::sync::Arc<::ferrite_cuda_core::MetalAllocator>,
            ::ferrite_forward::interpreter::metal::RuntimeFactory,
            usize,
        ) -> ::core::result::Result<
            ::ferrite_forward::interpreter::metal::MetalWorkerPool<crate::tinyllama_1_1b::Weights>,
            ::ferrite_forward::interpreter::metal::PoolBuildError,
        > = crate::tinyllama_1_1b::metal_pool;
        // load() resolves as a stream-free fn returning `Result<Weights>`.
        // Same fn-pointer-only check; no GpuWeights instance available in
        // unit-test ctx.
        let _loader: fn(
            &mut ::ferrite_cuda_core::weights::GpuWeights,
            ::ferrite_cuda_core::CUstream,
            usize,
            u8,
        ) -> ::anyhow::Result<crate::tinyllama_1_1b::Weights> = crate::tinyllama_1_1b::load;
    }
}
