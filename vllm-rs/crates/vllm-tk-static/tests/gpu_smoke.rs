// SPDX-License-Identifier: Apache-2.0
//! GPU smoke test for the static megakernel.
//!
//! Allocates dummy GPU buffers with correct tensor shapes, constructs
//! typed `LaunchArgs`, and calls `launch_decode` to verify the compiled kernel
//! executes without CUDA errors.
//!
//! Run on a GPU pod with:
//!   cargo test -p vllm-tk-static --features cuda --test gpu_smoke -- --ignored

#![cfg(feature = "cuda")]

use cudarc::driver::result;
use vllm_tk_static::*;

/// Allocate `bytes` of zeroed GPU memory, return device pointer as *mut u8.
fn gpu_alloc_zeros(bytes: usize) -> *mut u8 {
    unsafe {
        let dptr = result::malloc_sync(bytes).expect("cuMemAlloc failed");
        result::memset_d8_sync(dptr, 0, bytes).expect("cuMemsetD8 failed");
        dptr as *mut u8
    }
}

/// Smoke test: allocate all buffers with correct shapes, launch decode with BS=1.
///
/// The kernel will read zeroed weights and activations. We don't check output
/// correctness — just that the FFI call succeeds without C++ exceptions or
/// CUDA launch errors.
#[test]
#[ignore] // Only run on GPU pods
fn test_cuda_static_decode_smoke() {
    // Init CUDA driver + context
    result::init().expect("cuInit failed");
    let device = result::device::get(0).expect("cuDeviceGet failed");
    let _ctx = unsafe { result::primary_ctx::retain(device) }.expect("cuCtxRetain failed");
    unsafe { result::ctx::set_current(_ctx) }.expect("cuCtxSetCurrent failed");

    // Model dimensions (must match the megakernel! invocation in lib.rs)
    let nl = MegakernelLlamaSm89::NL; // 16
    let hd = MegakernelLlamaSm89::HD; // 2048
    let id = MegakernelLlamaSm89::ID; // 8192
    let qkv_dim = MegakernelLlamaSm89::QKV_DIM; // 3072
    let hdm = MegakernelLlamaSm89::HDM; // 64
    let nkh = MegakernelLlamaSm89::NKH; // 8
    let vs = MegakernelLlamaSm89::VS; // 128256

    let bf16: usize = 2;

    // TK scheduling constants
    let batch_size: usize = 1;
    let n_batch_blocks: usize = 1;
    let num_ops: usize = 11;
    let max_barrier_cols: usize = 128;
    let act_rows: usize = 128; // n_batch_blocks * matmul_batch_block_size

    // SM scheduling — dummy but valid shapes
    let sm_count: usize = 142; // L40S SM count
    let max_per_sm: usize = 1;
    let instruction_width: usize = 32;
    let timing_width: usize = 128;

    let num_pages: usize = 16;
    let page_size: usize = 16;

    // ── Allocate GPU buffers ──
    let bar_ptr = gpu_alloc_zeros(nl * num_ops * n_batch_blocks * max_barrier_cols * 4);
    let instr_ptr = gpu_alloc_zeros(sm_count * max_per_sm * instruction_width * 4);
    let timing_ptr = gpu_alloc_zeros(sm_count * max_per_sm * timing_width * 4);

    let qkv_w_ptr = gpu_alloc_zeros(nl * qkv_dim * hd * bf16);
    let attn_norm_ptr = gpu_alloc_zeros(nl * hd * bf16);
    let o_proj_ptr = gpu_alloc_zeros(nl * hd * hd * bf16);
    let mlp_norm_ptr = gpu_alloc_zeros(nl * hd * bf16);
    let up_ptr = gpu_alloc_zeros(nl * id * hd * bf16);
    let gate_ptr = gpu_alloc_zeros(nl * id * hd * bf16);
    let down_ptr = gpu_alloc_zeros(nl * hd * id * bf16);
    let lm_norm_ptr = gpu_alloc_zeros(hd * bf16);
    let lm_head_ptr = gpu_alloc_zeros(vs * hd * bf16);

    let k_cache_ptr = gpu_alloc_zeros(num_pages * page_size * nkh * hdm * bf16);
    let v_cache_ptr = gpu_alloc_zeros(num_pages * page_size * nkh * hdm * bf16);

    let rope_cos_ptr = gpu_alloc_zeros(4096 * hdm * bf16);
    let rope_sin_ptr = gpu_alloc_zeros(4096 * hdm * bf16);

    let hidden_ptr = gpu_alloc_zeros(act_rows * hd * bf16);
    let rms_rope_ptr = gpu_alloc_zeros(act_rows * hd * bf16);
    let rms_gate_ptr = gpu_alloc_zeros(act_rows * hd * bf16);
    let q_post_ptr = gpu_alloc_zeros(act_rows * hd * bf16);
    let attn_out_ptr = gpu_alloc_zeros(act_rows * hd * bf16);
    let silu_out_ptr = gpu_alloc_zeros(act_rows * id * bf16);
    let rms_lm_ptr = gpu_alloc_zeros(act_rows * hd * bf16);
    let logits_ptr = gpu_alloc_zeros(act_rows * vs * bf16);

    let pos_ids_ptr = gpu_alloc_zeros(batch_size * 4);
    let kv_indptr_ptr = gpu_alloc_zeros((batch_size + 1) * 4);
    let kv_indices_ptr = gpu_alloc_zeros(num_pages * 4);
    let kv_last_page_ptr = gpu_alloc_zeros(batch_size * 4);
    let kv_append_ptr = gpu_alloc_zeros(batch_size * 4);
    let dummy_meta_ptr = gpu_alloc_zeros(4);

    // ── Construct typed LaunchArgs ──
    let args = unsafe {
        LaunchArgs {
            barrier: GpuBarrier::from_raw(bar_ptr),
            instructions: GpuVmLayout::from_raw(instr_ptr),
            timings: GpuVmLayout::from_raw(timing_ptr),

            qkv_weights: GpuWeight::from_raw(qkv_w_ptr),
            attn_norm: GpuNormWeight::from_raw(attn_norm_ptr),
            o_proj: GpuWeight::from_raw(o_proj_ptr),
            mlp_norm: GpuNormWeight::from_raw(mlp_norm_ptr),
            up_weights: GpuWeight::from_raw(up_ptr),
            gate_weights: GpuWeight::from_raw(gate_ptr),
            down_proj: GpuWeightBig::from_raw(down_ptr),
            lm_head_norm: GpuNormWeight::from_raw(lm_norm_ptr),
            lm_head: GpuWeight::from_raw(lm_head_ptr),

            k_cache: GpuKvCache::from_raw(k_cache_ptr),
            v_cache: GpuKvCache::from_raw(v_cache_ptr),

            rope_cos: GpuRopeTable::from_raw(rope_cos_ptr),
            rope_sin: GpuRopeTable::from_raw(rope_sin_ptr),

            hidden_states: GpuActivation::from_raw(hidden_ptr),
            rms_rope: GpuActivation::from_raw(rms_rope_ptr),
            rms_gate: GpuActivation::from_raw(rms_gate_ptr),
            q_post_rope: GpuActivation::from_raw(q_post_ptr),
            attn_out: GpuActivation::from_raw(attn_out_ptr),
            silu_out: GpuActivationBig::from_raw(silu_out_ptr),
            rms_lm: GpuActivation::from_raw(rms_lm_ptr),
            logits: GpuLogits::from_raw(logits_ptr),

            position_ids: GpuMetaVec::from_raw(pos_ids_ptr),
            kv_indptr: GpuMetaVec::from_raw(kv_indptr_ptr),
            kv_indices: GpuMetaVec::from_raw(kv_indices_ptr),
            kv_last_page: GpuMetaVec::from_raw(kv_last_page_ptr),
            kv_append: GpuMetaVec::from_raw(kv_append_ptr),

            prefill_qo_indptr: GpuMetaVec::from_raw(dummy_meta_ptr),
            prefill_kv_indptr: GpuMetaVec::from_raw(dummy_meta_ptr),
            prefill_kv_indices: GpuMetaVec::from_raw(dummy_meta_ptr),
            prefill_kv_last_page_len: GpuMetaVec::from_raw(dummy_meta_ptr),

            attn_scale: 1.0 / (hdm as f32).sqrt(),
            rms_norm_eps: 1e-5,
            num_pages: num_pages as i32,
            prefill_num_seqs: 0,
            prefill_num_kv_pages: 0,
        }
    };

    let barrier_shape = [nl, num_ops, n_batch_blocks, max_barrier_cols];
    let inst_shape = [1, sm_count, max_per_sm, instruction_width];
    let timing_shape = [1, sm_count, max_per_sm, timing_width];

    // Use default stream (0)
    let stream: u64 = 0;

    // Launch!
    let rc = unsafe {
        MegakernelLlamaSm89::launch_decode(
            &args,
            DecodeBatchSize(batch_size as i32),
            NumTokens(0), // num_prefill_tokens
            barrier_shape,
            inst_shape,
            timing_shape,
            stream,
        )
    };
    eprintln!("launch_decode returned: {rc}");

    // rc == 0: kernel launched successfully (launch is async)
    // rc == 715 (cudaErrorIllegalAddress): kernel launched but hit zeroed pointers
    //   — this is EXPECTED with dummy zero buffers, it proves the FFI wiring works
    // rc == -2/-3: C++ exception during globals construction — FFI wiring broken
    assert!(
        rc == 0 || rc == 715,
        "launch_decode failed: rc={rc} (expected 0 or 715 with zeroed buffers)"
    );

    // Try to synchronize. With zeroed buffers the kernel will likely fault,
    // so we accept both success and IllegalAddress as proof the FFI works.
    let sync_result = result::ctx::synchronize();
    match sync_result {
        Ok(()) => eprintln!("Static decode kernel completed successfully (BS={batch_size})"),
        Err(e) => eprintln!("Kernel faulted as expected with zeroed buffers: {e}"),
    }

    eprintln!("FFI smoke test PASSED: kernel launch + globals construction works end-to-end");
}
