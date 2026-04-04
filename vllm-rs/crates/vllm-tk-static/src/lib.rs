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
    fn typed_handles_exist() {
        // Verify typed GPU handle types compile with correct dimensions.
        // These are function type checks — if the type doesn't exist, it won't compile.
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
    fn launch_args_typed_fields() {
        // Verify LaunchArgs has typed fields that enforce model dimensions.
        // Use a dummy non-null pointer (0x1000) — we never dereference it.
        let dummy = 0x1000usize as *mut u8;
        let args = unsafe {
            LaunchArgs {
                barrier: GpuBarrier::from_raw(dummy),
                instructions: GpuVmLayout::from_raw(dummy),
                timings: GpuVmLayout::from_raw(dummy),
                qkv_weights: GpuWeight::from_raw(dummy),
                attn_norm: GpuNormWeight::from_raw(dummy),
                o_proj: GpuWeight::from_raw(dummy),
                mlp_norm: GpuNormWeight::from_raw(dummy),
                up_weights: GpuWeight::from_raw(dummy),
                gate_weights: GpuWeight::from_raw(dummy),
                down_proj: GpuWeightBig::from_raw(dummy),
                lm_head_norm: GpuNormWeight::from_raw(dummy),
                lm_head: GpuWeight::from_raw(dummy),
                k_cache: GpuKvCache::from_raw(dummy),
                v_cache: GpuKvCache::from_raw(dummy),
                rope_cos: GpuRopeTable::from_raw(dummy),
                rope_sin: GpuRopeTable::from_raw(dummy),
                hidden_states: GpuActivation::from_raw(dummy),
                rms_rope: GpuActivation::from_raw(dummy),
                rms_gate: GpuActivation::from_raw(dummy),
                q_post_rope: GpuActivation::from_raw(dummy),
                attn_out: GpuActivation::from_raw(dummy),
                silu_out: GpuActivationBig::from_raw(dummy),
                rms_lm: GpuActivation::from_raw(dummy),
                logits: GpuLogits::from_raw(dummy),
                position_ids: GpuMetaVec::from_raw(dummy),
                kv_indptr: GpuMetaVec::from_raw(dummy),
                kv_indices: GpuMetaVec::from_raw(dummy),
                kv_last_page: GpuMetaVec::from_raw(dummy),
                kv_append: GpuMetaVec::from_raw(dummy),
                prefill_qo_indptr: GpuMetaVec::from_raw(dummy),
                prefill_kv_indptr: GpuMetaVec::from_raw(dummy),
                prefill_kv_indices: GpuMetaVec::from_raw(dummy),
                prefill_kv_last_page_len: GpuMetaVec::from_raw(dummy),
                attn_scale: 0.125,
                rms_norm_eps: 1e-5,
                num_pages: 0,
                prefill_num_seqs: 0,
                prefill_num_kv_pages: 0,
            }
        };
        assert_eq!(args.attn_scale, 0.125);
        assert_eq!(args.rms_norm_eps, 1e-5);
        assert_eq!(args.num_pages, 0);
        // Typed handles preserve pointer
        assert_eq!(args.qkv_weights.ptr_u64(), 0x1000);
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
        let dummy = 0x1000usize as *mut u8;
        let _hd: GpuActivation<2048> = unsafe { GpuActivation::from_raw(dummy) };
        let _id: GpuActivationBig<8192> = unsafe { GpuActivationBig::from_raw(dummy) };
        let _w: GpuWeight<8192, 2048> = unsafe { GpuWeight::from_raw(dummy) };
        let _n: GpuNormWeight<2048> = unsafe { GpuNormWeight::from_raw(dummy) };

        // Verify round-trip through pointer extraction
        assert_eq!(_hd.as_ptr(), dummy);
        assert_eq!(_w.as_ptr(), dummy);
        assert_eq!(_n.as_ptr(), dummy);
    }

    #[test]
    #[should_panic(expected = "null GPU pointer")]
    fn null_handle_panics() {
        // NonNull enforces non-null at construction time
        unsafe { GpuActivation::<2048>::from_raw(std::ptr::null_mut()) };
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
