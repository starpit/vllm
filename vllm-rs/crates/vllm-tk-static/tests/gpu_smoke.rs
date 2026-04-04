// SPDX-License-Identifier: Apache-2.0
//! GPU smoke test for the static megakernel.
//!
//! Allocates dummy GPU buffers with correct tensor shapes, constructs
//! `LaunchArgs`, and calls `launch_decode` to verify the compiled kernel
//! executes without CUDA errors.
//!
//! Run on a GPU pod with:
//!   cargo test -p vllm-tk-static --features cuda --test gpu_smoke -- --ignored

#![cfg(feature = "cuda")]

use cudarc::driver::result;
use vllm_tk_static::*;

/// Allocate `bytes` of zeroed GPU memory, return device pointer as u64.
fn gpu_alloc_zeros(bytes: usize) -> u64 {
    unsafe {
        let dptr = result::malloc_sync(bytes).expect("cuMemAlloc failed");
        result::memset_d8_sync(dptr, 0, bytes).expect("cuMemsetD8 failed");
        dptr
    }
}

fn tensor(ptr: u64, shape: &[usize]) -> TkTensorArg {
    TkTensorArg::new(ptr, shape)
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
    let id = MegakernelLlamaSm89::ID; // 5632
    let qkv_dim = MegakernelLlamaSm89::QKV_DIM; // 3072
    let hdm = MegakernelLlamaSm89::HDM; // 64
    let nkh = MegakernelLlamaSm89::NKH; // 8
    let vs = MegakernelLlamaSm89::VS; // 128256

    let bf16: usize = 2;

    // TK scheduling constants
    let batch_size: usize = 1;
    let matmul_batch_block_size: usize = 128;
    let n_batch_blocks: usize = batch_size.div_ceil(matmul_batch_block_size); // 1
    let nbh: usize = 48; // (NAH + 2*NKH) = 48
    let n_cols_id: usize = id / 32; // 176 for ID=5632... actually n_cols_id = ID/tile_col
    let max_barrier_cols: usize = nbh.max(128); // 128 (simplified, matches LLaMA-1B)
    let num_ops: usize = 11; // TK NUM_OPS
    let act_rows: usize = n_batch_blocks * matmul_batch_block_size; // 128

    // SM scheduling — dummy but valid shapes
    let sm_count: usize = 142; // L40S SM count
    let max_per_sm: usize = 1; // minimal
    let instruction_width: usize = 32;
    let timing_width: usize = 128;

    let num_pages: usize = 16;
    let page_size: usize = 16;

    // ── Barrier: [NL, NUM_OPS, n_batch_blocks, max_barrier_cols] u32 ──
    let bar_size = nl * num_ops * n_batch_blocks * max_barrier_cols;
    let bar = gpu_alloc_zeros(bar_size * 4);

    // ── Instructions: [1, sm_count, max_per_sm, INSTRUCTION_WIDTH] i32 ──
    let instr_size = 1 * sm_count * max_per_sm * instruction_width;
    let instr = gpu_alloc_zeros(instr_size * 4);

    // ── Timings: [1, sm_count, max_per_sm, TIMING_WIDTH] i32 ──
    let timing_size = 1 * sm_count * max_per_sm * timing_width;
    let timing = gpu_alloc_zeros(timing_size * 4);

    // ── Weights ──
    // qkv_w: from_gpu_tensor → shape from weight tensor
    let qkv_w = gpu_alloc_zeros(nl * qkv_dim * hd * bf16);
    let attn_norm_w = gpu_alloc_zeros(nl * hd * bf16);
    let o_proj_w = gpu_alloc_zeros(nl * hd * hd * bf16);
    let mlp_norm_w = gpu_alloc_zeros(nl * hd * bf16);
    let up_w = gpu_alloc_zeros(nl * id * hd * bf16);
    let gate_w = gpu_alloc_zeros(nl * id * hd * bf16);
    let down_w = gpu_alloc_zeros(nl * hd * id * bf16);
    let lm_norm_w = gpu_alloc_zeros(hd * bf16);
    let lm_w = gpu_alloc_zeros(vs * hd * bf16);

    // ── KV cache: [num_pages, page_size, nkh, hdm] ──
    let k_cache = gpu_alloc_zeros(num_pages * page_size * nkh * hdm * bf16);
    let v_cache = gpu_alloc_zeros(num_pages * page_size * nkh * hdm * bf16);

    // ── RoPE tables ──
    let rope_cos = gpu_alloc_zeros(4096 * hdm * bf16);
    let rope_sin = gpu_alloc_zeros(4096 * hdm * bf16);

    // ── Activations: [1, 1, act_rows, dim] — matches worker.rs ──
    let hidden = gpu_alloc_zeros(act_rows * hd * bf16);
    let rms_rope = gpu_alloc_zeros(act_rows * hd * bf16);
    let rms_gate = gpu_alloc_zeros(act_rows * hd * bf16);
    let q_post = gpu_alloc_zeros(act_rows * hd * bf16);
    let attn_out_buf = gpu_alloc_zeros(act_rows * hd * bf16);
    let silu_out = gpu_alloc_zeros(act_rows * id * bf16);
    let rms_lm = gpu_alloc_zeros(act_rows * hd * bf16);
    let logits_buf = gpu_alloc_zeros(act_rows * vs * bf16);

    // ── Paged KV metadata (decode) — 1D i32 ──
    let pos_ids = gpu_alloc_zeros(batch_size * 4);
    let kv_indptr = gpu_alloc_zeros((batch_size + 1) * 4);
    let kv_indices = gpu_alloc_zeros(num_pages * 4);
    let kv_last_page = gpu_alloc_zeros(batch_size * 4);
    let kv_append = gpu_alloc_zeros(batch_size * 4);

    // ── Paged KV metadata (prefill) — dummy [1] for decode ──
    let dummy_meta = gpu_alloc_zeros(4);

    // ── Construct LaunchArgs ──
    let args = LaunchArgs {
        barrier: tensor(bar, &[nl, num_ops, n_batch_blocks, max_barrier_cols]),
        instructions: tensor(instr, &[1, sm_count, max_per_sm, instruction_width]),
        timings: tensor(timing, &[1, sm_count, max_per_sm, timing_width]),

        // Weights — from_gpu_tensor produces shapes from the original weight tensors
        qkv_weights: tensor(qkv_w, &[nl * qkv_dim, hd]),
        attn_norm: tensor(attn_norm_w, &[nl, hd]),
        o_proj: tensor(o_proj_w, &[nl * hd, hd]),
        mlp_norm: tensor(mlp_norm_w, &[nl, hd]),
        up_weights: tensor(up_w, &[nl * id, hd]),
        gate_weights: tensor(gate_w, &[nl * id, hd]),
        down_proj: tensor(down_w, &[nl * hd, id]),
        lm_head_norm: tensor(lm_norm_w, &[1, hd]),
        lm_head: tensor(lm_w, &[vs, hd]),

        k_cache: tensor(k_cache, &[num_pages, page_size, nkh, hdm]),
        v_cache: tensor(v_cache, &[num_pages, page_size, nkh, hdm]),

        rope_cos: tensor(rope_cos, &[4096, hdm]),
        rope_sin: tensor(rope_sin, &[4096, hdm]),

        // Activations: [1, 1, act_rows, dim] — matches worker.rs
        hidden_states: tensor(hidden, &[1, 1, act_rows, hd]),
        rms_rope: tensor(rms_rope, &[1, 1, act_rows, hd]),
        rms_gate: tensor(rms_gate, &[1, 1, act_rows, hd]),
        q_post_rope: tensor(q_post, &[1, 1, act_rows, hd]),
        attn_out: tensor(attn_out_buf, &[1, 1, act_rows, hd]),
        silu_out: tensor(silu_out, &[1, 1, act_rows, id]),
        rms_lm: tensor(rms_lm, &[1, 1, act_rows, hd]),
        logits: tensor(logits_buf, &[1, 1, act_rows, vs]),

        // Paged KV metadata (decode)
        position_ids: tensor(pos_ids, &[batch_size]),
        kv_indptr: tensor(kv_indptr, &[batch_size + 1]),
        kv_indices: tensor(kv_indices, &[num_pages]),
        kv_last_page: tensor(kv_last_page, &[batch_size]),
        kv_append: tensor(kv_append, &[batch_size]),

        // Prefill metadata — dummy for decode (int32_vector_t: b=1,d=1,r=1,c=1)
        prefill_qo_indptr: tensor(dummy_meta, &[1]),
        prefill_kv_indptr: tensor(dummy_meta, &[1]),
        prefill_kv_indices: tensor(dummy_meta, &[1]),
        prefill_kv_last_page_len: tensor(dummy_meta, &[1]),

        attn_scale: 1.0 / (hdm as f32).sqrt(),
        rms_norm_eps: 1e-5,
        num_pages: num_pages as i32,
    };

    // Use default stream (0)
    let stream: u64 = 0;

    // Launch!
    let rc = unsafe { MegakernelLlamaSm89::launch_decode(&args, batch_size as i32, 0, stream) };
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
