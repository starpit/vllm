// SPDX-License-Identifier: Apache-2.0
//! Static megakernel for LLaMA decode (sm89).
//!
//! This crate uses `megakernel!` to generate a compile-time verified,
//! statically dispatched CUDA megakernel. The kernel calls the same TK ops
//! as the VM-based KVM kernel, but without the instruction fetch, opcode
//! dispatch, or page allocator overhead.
//!
//! With `--features cuda`, build.rs compiles the generated CUDA via cudaforge.
//! Without `cuda`, this crate only provides the Rust types and diagram.

#[cfg(feature = "cuda")]
mod ffi;

// Invoke the megakernel! proc-macro. This generates:
// - Const-generic tensor types: Activation<D>, Weight<R,C>, Weight1D<D>, KvCache, Metadata
// - MegakernelLlamaSm89 struct with typed launch() and CUDA_SOURCE const
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
        // Decode kernel
        assert!(src.contains("llama_sm89_decode_static"));
        assert!(src.contains("OPCODE_GQA_AttentionDecode"));
        // Prefill kernel
        assert!(src.contains("llama_sm89_prefill_static"));
        assert!(src.contains("OPCODE_GQA_AttentionPrefill"));
        assert!(src.contains("total_tokens"));
        assert!(src.contains("run_op_ext<"));
        // Shared
        assert!(src.contains("run_op<"));
    }

    #[test]
    fn typed_wrappers_exist() {
        // Verify the const-generic types compile with correct dimensions
        let _: fn(&Activation<2048>) = |_| {};
        let _: fn(&Weight<3072, 2048>) = |_| {};
        let _: fn(&Weight1D<2048>) = |_| {};
        let _: fn(&KvCache) = |_| {};
        let _: fn(&Metadata) = |_| {};
    }

    #[test]
    fn launch_args_struct_exists() {
        // Verify LaunchArgs has expected fields with correct types.
        let args = LaunchArgs {
            barrier: TkTensorArg::null(),
            instructions: TkTensorArg::null(),
            timings: TkTensorArg::null(),
            qkv_weights: TkTensorArg::null(),
            attn_norm: TkTensorArg::null(),
            o_proj: TkTensorArg::null(),
            mlp_norm: TkTensorArg::null(),
            up_weights: TkTensorArg::null(),
            gate_weights: TkTensorArg::null(),
            down_proj: TkTensorArg::null(),
            lm_head_norm: TkTensorArg::null(),
            lm_head: TkTensorArg::null(),
            k_cache: TkTensorArg::null(),
            v_cache: TkTensorArg::null(),
            rope_cos: TkTensorArg::null(),
            rope_sin: TkTensorArg::null(),
            hidden_states: TkTensorArg::null(),
            rms_rope: TkTensorArg::null(),
            rms_gate: TkTensorArg::null(),
            q_post_rope: TkTensorArg::null(),
            attn_out: TkTensorArg::null(),
            silu_out: TkTensorArg::null(),
            rms_lm: TkTensorArg::null(),
            logits: TkTensorArg::null(),
            position_ids: TkTensorArg::null(),
            kv_indptr: TkTensorArg::null(),
            kv_indices: TkTensorArg::null(),
            kv_last_page: TkTensorArg::null(),
            kv_append: TkTensorArg::null(),
            prefill_qo_indptr: TkTensorArg::null(),
            prefill_kv_indptr: TkTensorArg::null(),
            prefill_kv_indices: TkTensorArg::null(),
            prefill_kv_last_page_len: TkTensorArg::null(),
            attn_scale: 0.125,
            rms_norm_eps: 1e-5,
            num_pages: 0,
        };
        // Verify the struct is constructible and fields have expected values
        assert_eq!(args.barrier.ptr, 0);
        assert_eq!(args.attn_scale, 0.125);
        assert_eq!(args.rms_norm_eps, 1e-5);
        assert_eq!(args.num_pages, 0);
    }

    #[test]
    fn tk_tensor_arg_new() {
        let arg1d = TkTensorArg::new(0xDEAD, &[128]);
        assert_eq!(arg1d.ptr, 0xDEAD);
        assert_eq!((arg1d.b, arg1d.d, arg1d.r, arg1d.c), (1, 1, 1, 128));

        let arg2d = TkTensorArg::new(0xBEEF, &[32, 2048]);
        assert_eq!((arg2d.b, arg2d.d, arg2d.r, arg2d.c), (1, 1, 32, 2048));

        let arg3d = TkTensorArg::new(0xCAFE, &[4, 32, 64]);
        assert_eq!((arg3d.b, arg3d.d, arg3d.r, arg3d.c), (1, 4, 32, 64));

        let arg4d = TkTensorArg::new(0xF00D, &[2, 4, 32, 64]);
        assert_eq!((arg4d.b, arg4d.d, arg4d.r, arg4d.c), (2, 4, 32, 64));
    }

    #[test]
    #[should_panic(expected = "expected 1-4D shape")]
    fn tk_tensor_arg_5d_panics() {
        TkTensorArg::new(0, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn const_generic_type_safety() {
        // These compile: shapes match model dimensions
        let _hd: Activation<2048> = unsafe { Activation::from_ptr(std::ptr::null_mut()) };
        let _id: Activation<8192> = unsafe { Activation::from_ptr(std::ptr::null_mut()) };
        let _w: Weight<8192, 2048> = unsafe { Weight::from_ptr(std::ptr::null()) };
        let _n: Weight1D<2048> = unsafe { Weight1D::from_ptr(std::ptr::null()) };

        // Verify round-trip through pointer extraction
        assert!(_hd.as_ptr().is_null());
        assert!(_w.as_ptr().is_null());
        assert!(_n.as_ptr().is_null());
    }

    #[test]
    fn model_dimension_constants() {
        // QKV_DIM = (NAH + 2*NKH) * HDM = (32 + 2*8) * 64 = 3072
        assert_eq!(MegakernelLlamaSm89::QKV_DIM, 3072);
        assert_eq!(
            MegakernelLlamaSm89::NAH * MegakernelLlamaSm89::HDM,
            MegakernelLlamaSm89::HD
        );
        assert_eq!(MegakernelLlamaSm89::NAH % MegakernelLlamaSm89::NKH, 0);
        assert_eq!(MegakernelLlamaSm89::HD % MegakernelLlamaSm89::HDM, 0);
    }

    #[test]
    fn cuda_source_contains_launch_wrappers() {
        let src = MegakernelLlamaSm89::CUDA_SOURCE;
        // Flat-arg launch wrappers for FFI
        assert!(src.contains("llama_sm89_decode_static_launch"));
        assert!(src.contains("llama_sm89_prefill_static_launch"));
        // TkTensorArg type and globals construction
        assert!(src.contains("TkTensorArg"));
        assert!(src.contains("make_arg<G::"));
        // Both kernels reference all TK ops
        assert!(src.contains("attn_norm<config, globals>"));
        assert!(src.contains("o_proj<config, globals>"));
        assert!(src.contains("lm_head<config, globals>"));
    }

    #[test]
    fn cuda_source_has_semaphore_init() {
        let src = MegakernelLlamaSm89::CUDA_SOURCE;
        // sm89 single-arg arrive() loop
        assert!(src.contains("arrive("));
        // Parallel semaphore init
        assert!(src.contains("INSTRUCTION_PIPELINE_STAGES"));
        assert!(src.contains("NUM_PAGES"));
    }
}
