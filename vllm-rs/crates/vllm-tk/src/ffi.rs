// SPDX-License-Identifier: Apache-2.0
//! Raw FFI declarations for the TK KVM LLaMA sm89 megakernel.

/// Flat tensor descriptor matching the C-side `TkTensorArg` struct.
/// Each field of the TK globals struct is passed as one of these.
///
/// TK's `gl<T, B, D, R, C>` type stores a raw device pointer plus 4 dimensions.
/// Positive template params are compile-time (static) dims; `-1` means runtime (dynamic).
/// `make_gl<GL>(ptr, b, d, r, c)` validates static dims and stores dynamic ones.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkTensorArg {
    pub ptr: u64,
    pub b: i32,
    pub d: i32,
    pub r: i32,
    pub c: i32,
}

impl TkTensorArg {
    /// Create from a raw device pointer and shape.
    /// Shape is padded to 4D with leading 1s:
    /// - 1D [c]         → (1, 1, 1, c)
    /// - 2D [r, c]      → (1, 1, r, c)
    /// - 3D [d, r, c]   → (1, d, r, c)
    /// - 4D [b, d, r, c]
    pub fn new(ptr: u64, shape: &[usize]) -> Self {
        let (b, d, r, c) = match shape.len() {
            1 => (1, 1, 1, shape[0] as i32),
            2 => (1, 1, shape[0] as i32, shape[1] as i32),
            3 => (1, shape[0] as i32, shape[1] as i32, shape[2] as i32),
            4 => (
                shape[0] as i32,
                shape[1] as i32,
                shape[2] as i32,
                shape[3] as i32,
            ),
            _ => panic!("TkTensorArg: expected 1-4D shape, got {}D", shape.len()),
        };
        Self { ptr, b, d, r, c }
    }

    /// Convert from a `GpuTensor` (pads shape to 4D with leading 1s).
    #[cfg(feature = "cuda")]
    pub fn from_gpu_tensor(t: vllm_cuda::GpuTensor) -> Self {
        let s = t.shape(); // &[u32]
        let shape: Vec<usize> = s.iter().map(|&d| d as usize).collect();
        Self::new(t.as_ptr::<u8>() as u64, &shape)
    }

    /// A null/zero tensor arg (for unused fields like timings).
    pub fn null() -> Self {
        Self {
            ptr: 0,
            b: 0,
            d: 0,
            r: 0,
            c: 0,
        }
    }
}

unsafe extern "C" {
    pub fn tk_llama_1b_launch(
        // VM state
        bar: TkTensorArg,
        instructions: TkTensorArg,
        timings: TkTensorArg,
        // Weights
        qkv_w: TkTensorArg,
        attn_norm_w: TkTensorArg,
        o_w: TkTensorArg,
        mlp_norm_w: TkTensorArg,
        up_w: TkTensorArg,
        gate_w: TkTensorArg,
        down_w: TkTensorArg,
        lm_norm_w: TkTensorArg,
        lm_w: TkTensorArg,
        // KV cache
        k_cache: TkTensorArg,
        v_cache: TkTensorArg,
        // RoPE
        rope_cos: TkTensorArg,
        rope_sin: TkTensorArg,
        // Activations
        hidden: TkTensorArg,
        rms_rope: TkTensorArg,
        rms_gate: TkTensorArg,
        q_post: TkTensorArg,
        attn_out: TkTensorArg,
        silu: TkTensorArg,
        rms_lm: TkTensorArg,
        logits: TkTensorArg,
        // Paged KV metadata — decode
        pos_ids: TkTensorArg,
        kv_indptr: TkTensorArg,
        kv_indices: TkTensorArg,
        kv_last_page: TkTensorArg,
        kv_append: TkTensorArg,
        // Paged KV metadata — prefill
        prefill_qo_indptr: TkTensorArg,
        prefill_kv_indptr: TkTensorArg,
        prefill_kv_indices: TkTensorArg,
        prefill_kv_last_page_len: TkTensorArg,
        // Scalars
        attn_scale: f32,
        rms_norm_eps: f32,
        num_pages: i32,
        batch_size: i32,
        num_prefill_tokens: i32,
        // Stream
        stream: u64,
    );
}
