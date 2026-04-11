// SPDX-License-Identifier: Apache-2.0
//! TK op-level test harness.
//!
//! Each TK op (rms_norm, gemm, attention, etc.) gets a standalone CUDA kernel
//! that can be launched independently for testing. The build.rs generates per-op
//! `.cu` files and compiles them into a static library.
//!
//! With `--features cuda`:
//! - FFI declarations for `test_{op_name}_launch(...)` are available
//! - GPU tests in `tests/op_tests.rs` exercise each op in isolation

/// The flat tensor descriptor passed to CUDA launch wrappers via FFI.
/// Must match the `TkTensorArg` struct in the generated CUDA code exactly.
///
/// **Do not construct directly** — use the typed wrappers below which enforce
/// the correct `(b, d, r, c)` mapping for each GL type at compile time.
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
    /// Escape hatch for VM state tensors (barriers, instructions, timings)
    /// that don't map to a standard GL type.
    pub fn raw(ptr: u64, shape: &[usize]) -> Self {
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
            _ => panic!("TkTensorArg: unsupported shape rank {}", shape.len()),
        };
        Self { ptr, b, d, r, c }
    }

    /// Create a zero/null tensor arg (for unused slots).
    pub fn null() -> Self {
        Self {
            ptr: 0,
            b: 1,
            d: 1,
            r: 1,
            c: 1,
        }
    }
}

// ── Typed tensor descriptors ──
//
// Each type mirrors one GL type from llama_sm89.cuh. They are
// #[repr(transparent)] over TkTensorArg so the ABI is identical,
// but the Rust type system prevents passing a WeightArg where a
// NormWeightArg is expected.

macro_rules! typed_tensor_arg {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug)]
        pub struct $name(TkTensorArg);

        // Allow conversion to raw TkTensorArg when needed
        impl From<$name> for TkTensorArg {
            fn from(t: $name) -> Self { t.0 }
        }
    };
}

typed_tensor_arg!(
    /// `weights_t = gl<bf16, 1, -1, -1, hidden_dim>`
    /// Also used for `weights_big_t` (same shape, different C dim).
    WeightArg
);
impl WeightArg {
    /// Create from `[1, num_layers, output_dim, input_dim]`.
    pub fn new(ptr: u64, num_layers: usize, output_dim: usize, input_dim: usize) -> Self {
        Self(TkTensorArg {
            ptr,
            b: 1,
            d: num_layers as i32,
            r: output_dim as i32,
            c: input_dim as i32,
        })
    }
}

typed_tensor_arg!(
    /// `norm_weights_t = gl<bf16, 1, 1, -1, hidden_dim>`
    NormWeightArg
);
impl NormWeightArg {
    /// Create from `[1, 1, num_layers, hidden_dim]`.
    pub fn new(ptr: u64, num_layers: usize, hidden_dim: usize) -> Self {
        Self(TkTensorArg {
            ptr,
            b: 1,
            d: 1,
            r: num_layers as i32,
            c: hidden_dim as i32,
        })
    }
}

typed_tensor_arg!(
    /// `activations_t = gl<bf16, 1, 1, -1, dim>`
    /// Also used for `activations_big_t`.
    ActivationArg
);
impl ActivationArg {
    /// Create from `[1, 1, batch, dim]`.
    pub fn new(ptr: u64, batch: usize, dim: usize) -> Self {
        Self(TkTensorArg {
            ptr,
            b: 1,
            d: 1,
            r: batch as i32,
            c: dim as i32,
        })
    }
}

typed_tensor_arg!(
    /// `logits_t = gl<bf16, 1, 1, -1, -1>`
    LogitsArg
);
impl LogitsArg {
    /// Create from `[1, 1, batch, vocab_size]`.
    pub fn new(ptr: u64, batch: usize, vocab_size: usize) -> Self {
        Self(TkTensorArg {
            ptr,
            b: 1,
            d: 1,
            r: batch as i32,
            c: vocab_size as i32,
        })
    }
}

typed_tensor_arg!(
    /// `kv_cache_t = gl<bf16, -1, -1, num_kv_heads, head_dim>`
    KvCacheArg
);
impl KvCacheArg {
    /// Create from `[total_pages, page_size, num_kv_heads, head_dim]`.
    pub fn new(
        ptr: u64,
        total_pages: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self(TkTensorArg {
            ptr,
            b: total_pages as i32,
            d: page_size as i32,
            r: num_kv_heads as i32,
            c: head_dim as i32,
        })
    }
}

typed_tensor_arg!(
    /// `rope_table_t = gl<float, 1, 1, -1, head_dim>`
    RopeArg
);
impl RopeArg {
    /// Create from `[1, 1, max_positions, head_dim]`.
    pub fn new(ptr: u64, max_positions: usize, head_dim: usize) -> Self {
        Self(TkTensorArg {
            ptr,
            b: 1,
            d: 1,
            r: max_positions as i32,
            c: head_dim as i32,
        })
    }
}

typed_tensor_arg!(
    /// `int32_vector_t = gl<int, 1, 1, 1, -1>`
    IntVecArg
);
impl IntVecArg {
    /// Create from `[1, 1, 1, len]`.
    pub fn new(ptr: u64, len: usize) -> Self {
        Self(TkTensorArg {
            ptr,
            b: 1,
            d: 1,
            r: 1,
            c: len as i32,
        })
    }
}

typed_tensor_arg!(
    /// `barriers = gl<uint, -1, -1, -1, -1>`
    BarrierArg
);
impl BarrierArg {
    /// Create from `[num_layers, num_ops, batch_blocks, cols]`.
    pub fn new(
        ptr: u64,
        num_layers: usize,
        num_ops: usize,
        batch_blocks: usize,
        cols: usize,
    ) -> Self {
        Self(TkTensorArg {
            ptr,
            b: num_layers as i32,
            d: num_ops as i32,
            r: batch_blocks as i32,
            c: cols as i32,
        })
    }
}

/// Op names that have compiled test kernels.
pub const OP_NAMES: &[&str] = &[
    "attn_norm",
    "qkv_rope_append",
    "attention_decode",
    "o_proj_residual",
    "mlp_norm",
    "gate_silu",
    "up_matmul",
    "down_proj_residual",
    "lm_head_norm",
    "lm_head",
    "attention_prefill",
];

#[cfg(feature = "cuda")]
pub mod ffi {
    use super::*;

    // FFI launch functions for each op's test kernel.
    // Parameters use typed wrappers — the compiler rejects mismatched types.
    macro_rules! declare_test_launch {
        ($name:ident) => {
            unsafe extern "C" {
                pub fn $name(
                    // VM state (3 tensors)
                    bar: BarrierArg,
                    instructions: TkTensorArg, // no standard GL type
                    timings: TkTensorArg,      // no standard GL type
                    // Weights (9 tensors)
                    qkv_w: WeightArg,
                    attn_norm_w: NormWeightArg,
                    o_w: WeightArg,
                    mlp_norm_w: NormWeightArg,
                    up_w: WeightArg,
                    gate_w: WeightArg,
                    down_w: WeightArg, // weights_big_t, same repr
                    lm_norm_w: NormWeightArg,
                    lm_w: WeightArg,
                    // KV cache (2 tensors)
                    k_cache: KvCacheArg,
                    v_cache: KvCacheArg,
                    // RoPE (2 tensors)
                    rope_cos: RopeArg,
                    rope_sin: RopeArg,
                    // Activations (8 tensors)
                    hidden: ActivationArg,
                    rms_rope: ActivationArg,
                    rms_gate: ActivationArg,
                    q_post: ActivationArg,
                    attn_out: ActivationArg,
                    silu: ActivationArg, // activations_big_t, same repr
                    rms_lm: ActivationArg,
                    logits_arg: LogitsArg,
                    // Paged KV metadata — decode (5 tensors)
                    pos_ids: IntVecArg,
                    kv_indptr: IntVecArg,
                    kv_indices: IntVecArg,
                    kv_last_page: IntVecArg,
                    kv_append: IntVecArg,
                    // Paged KV metadata — prefill (4 tensors)
                    prefill_qo_indptr: IntVecArg,
                    prefill_kv_indptr: IntVecArg,
                    prefill_kv_indices: IntVecArg,
                    prefill_kv_last_page_len: IntVecArg,
                    // Scalars
                    attn_scale: f32,
                    rms_norm_eps: f32,
                    num_pages: i32,
                    batch_size: i32,
                    num_prefill_tokens: i32,
                    num_layers: i32,
                    // CUDA stream
                    stream: u64,
                ) -> i32;
            }
        };
    }

    declare_test_launch!(test_attn_norm_launch);
    declare_test_launch!(test_mlp_norm_launch);
    declare_test_launch!(test_lm_head_norm_launch);
    declare_test_launch!(test_qkv_rope_append_launch);
    declare_test_launch!(test_attention_decode_launch);
    declare_test_launch!(test_attention_prefill_launch);
    declare_test_launch!(test_o_proj_residual_launch);
    declare_test_launch!(test_gate_silu_launch);
    declare_test_launch!(test_up_matmul_launch);
    declare_test_launch!(test_down_proj_residual_launch);
    declare_test_launch!(test_lm_head_launch);

    // Inline kernels (no KVM protocol — static tile pipeline)
    declare_test_launch!(inline_rmsnorm_launch);
    declare_test_launch!(inline_gemm_launch);
    declare_test_launch!(fused_rmsnorm_gemm_launch);
    declare_test_launch!(fused_mlp_launch);
    declare_test_launch!(cp5_fused_mlp_launch);
    declare_test_launch!(fused_layer_launch);
    declare_test_launch!(fused_full_layer_launch);
    declare_test_launch!(fused_multi_layer_launch);
    declare_test_launch!(inline_attention_decode_launch);
    declare_test_launch!(fused_multi_sm_launch);
    declare_test_launch!(fused_multi_sm_profile_launch);
    declare_test_launch!(fused_prefill_attn_launch);
    declare_test_launch!(fused_prefill_layer_launch);

    // Phase 3b — scheduled megakernel placeholder. Different signature
    // (just three pointers / counters), declared by hand instead of via
    // the macro.
    // The scheduled megakernel signature is generated per variant — same
    // shape, different symbol suffix. We declare it as a macro to keep the
    // declarations in sync.
    macro_rules! decl_scheduled_megakernel {
        ($launch:ident, $launch_per_wave:ident, $launch_per_kind:ident, $num_nodes:ident, $num_ctas:ident, $num_waves:ident) => {
            unsafe extern "C" {
                pub fn $launch(
                    hidden_states: *mut std::ffi::c_void,
                    rms_rope: *mut std::ffi::c_void,
                    qkv: *mut std::ffi::c_void,
                    q_post_rope: *mut std::ffi::c_void,
                    attn_out: *mut std::ffi::c_void,
                    rms_gate: *mut std::ffi::c_void,
                    silu_out: *mut std::ffi::c_void,
                    k_cache: *mut std::ffi::c_void,
                    v_cache: *mut std::ffi::c_void,
                    prefill_kv_indices: *const i32,
                    prefill_kv_indptr: *const i32,
                    prefill_qo_indptr: *const i32,
                    attn_norm_w: *mut std::ffi::c_void,
                    mlp_norm_w: *mut std::ffi::c_void,
                    qkv_w: *mut std::ffi::c_void,
                    o_w: *mut std::ffi::c_void,
                    gate_w: *mut std::ffi::c_void,
                    up_w: *mut std::ffi::c_void,
                    down_w: *mut std::ffi::c_void,
                    eps: f32,
                    attn_scale: f32,
                    flags: *mut u32,
                    tick_counter: *mut u32,
                    barrier_arrived: *mut u32,
                    phase_clocks: *mut u64,
                    // Device pointer to a `PersistentParams[NUM_LAYERS]`
                    // array, populated host-side via the FlashInfer
                    // shim helper. May be null in test paths that
                    // don't exercise the FlashInferAttentionLayer
                    // dispatch arm.
                    flashinfer_params: *mut std::ffi::c_void,
                    stream: *mut std::ffi::c_void,
                );
                // CP3: per-kind launcher. Same signature as the
                // legacy launcher above; loops cudaLaunchCooperativeKernel
                // over each wave in the schedule, dispatching to a
                // per-kind `__global__` template instantiation based on
                // the wave's `BoundKernel::kernel_tag()`. Each per-kind
                // instantiation has its own NVCC register / shmem
                // budget, eliminating the spills observed in the
                // legacy single-`__global__` kernel. Picked at
                // runtime via the FERRITE_PER_KIND_LOWERING env var.
                pub fn $launch_per_kind(
                    hidden_states: *mut std::ffi::c_void,
                    rms_rope: *mut std::ffi::c_void,
                    qkv: *mut std::ffi::c_void,
                    q_post_rope: *mut std::ffi::c_void,
                    attn_out: *mut std::ffi::c_void,
                    rms_gate: *mut std::ffi::c_void,
                    silu_out: *mut std::ffi::c_void,
                    k_cache: *mut std::ffi::c_void,
                    v_cache: *mut std::ffi::c_void,
                    prefill_kv_indices: *const i32,
                    prefill_kv_indptr: *const i32,
                    prefill_qo_indptr: *const i32,
                    attn_norm_w: *mut std::ffi::c_void,
                    mlp_norm_w: *mut std::ffi::c_void,
                    qkv_w: *mut std::ffi::c_void,
                    o_w: *mut std::ffi::c_void,
                    gate_w: *mut std::ffi::c_void,
                    up_w: *mut std::ffi::c_void,
                    down_w: *mut std::ffi::c_void,
                    eps: f32,
                    attn_scale: f32,
                    flags: *mut u32,
                    tick_counter: *mut u32,
                    barrier_arrived: *mut u32,
                    phase_clocks: *mut u64,
                    flashinfer_params: *mut std::ffi::c_void,
                    stream: *mut std::ffi::c_void,
                );
                // CP2: per-wave launcher. Same signature as the
                // legacy launcher above; loops cudaLaunchCooperativeKernel
                // over each wave in the schedule. Picked at runtime
                // via the FERRITE_PER_WAVE_LOWERING env var. See
                // `templates/scheduled/megakernel.cu` for the body.
                pub fn $launch_per_wave(
                    hidden_states: *mut std::ffi::c_void,
                    rms_rope: *mut std::ffi::c_void,
                    qkv: *mut std::ffi::c_void,
                    q_post_rope: *mut std::ffi::c_void,
                    attn_out: *mut std::ffi::c_void,
                    rms_gate: *mut std::ffi::c_void,
                    silu_out: *mut std::ffi::c_void,
                    k_cache: *mut std::ffi::c_void,
                    v_cache: *mut std::ffi::c_void,
                    prefill_kv_indices: *const i32,
                    prefill_kv_indptr: *const i32,
                    prefill_qo_indptr: *const i32,
                    attn_norm_w: *mut std::ffi::c_void,
                    mlp_norm_w: *mut std::ffi::c_void,
                    qkv_w: *mut std::ffi::c_void,
                    o_w: *mut std::ffi::c_void,
                    gate_w: *mut std::ffi::c_void,
                    up_w: *mut std::ffi::c_void,
                    down_w: *mut std::ffi::c_void,
                    eps: f32,
                    attn_scale: f32,
                    flags: *mut u32,
                    tick_counter: *mut u32,
                    barrier_arrived: *mut u32,
                    phase_clocks: *mut u64,
                    flashinfer_params: *mut std::ffi::c_void,
                    stream: *mut std::ffi::c_void,
                );
                pub fn $num_nodes() -> u32;
                pub fn $num_ctas() -> u32;
                pub fn $num_waves() -> u32;
            }
        };
    }
    // CP4: cuBLAS FFI for the GEMM-only microbench. cuBLAS exposes
    // a stable C API; we declare just what the bench needs and link
    // dynamically against libcublas.so.12 (build.rs adds -lcublas).
    pub type CublasHandle = *mut std::ffi::c_void;
    pub type CublasStatus = i32;
    pub const CUBLAS_OP_N: i32 = 0;
    pub const CUBLAS_OP_T: i32 = 1;
    pub const CUDA_R_16BF: i32 = 14;
    pub const CUBLAS_GEMM_DEFAULT: i32 = -1;
    pub const CUBLAS_COMPUTE_32F: i32 = 68;
    unsafe extern "C" {
        pub fn cublasCreate_v2(handle: *mut CublasHandle) -> CublasStatus;
        pub fn cublasDestroy_v2(handle: CublasHandle) -> CublasStatus;
        pub fn cublasSetStream_v2(
            handle: CublasHandle,
            stream: *mut std::ffi::c_void,
        ) -> CublasStatus;
        #[allow(clippy::too_many_arguments)]
        pub fn cublasGemmEx(
            handle: CublasHandle,
            transa: i32,
            transb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: *const f32,
            a: *const std::ffi::c_void,
            atype: i32,
            lda: i32,
            b: *const std::ffi::c_void,
            btype: i32,
            ldb: i32,
            beta: *const f32,
            c: *mut std::ffi::c_void,
            ctype: i32,
            ldc: i32,
            compute_type: i32,
            algo: i32,
        ) -> CublasStatus;
    }

    // CP4: vllm-rs fused kernels (linked from libvllm_kernels.a). The
    // signatures here mirror crates/vllm-cuda/src/kernels.rs for the
    // bf16 entry points the natural sm_89 lowering would call.
    unsafe extern "C" {
        pub fn rms_norm_bf16(
            out: *mut u16,
            input: *const u16,
            weight: *const u16,
            epsilon: f32,
            num_tokens: i32,
            hidden_size: i32,
            stream: *mut std::ffi::c_void,
        );
        pub fn silu_and_mul_fused_bf16(
            out: *mut u16,
            gate_up: *const u16,
            num_tokens: i32,
            d: i32,
            stream: *mut std::ffi::c_void,
        );
        pub fn rotary_embedding_bf16(
            positions: *const u32,
            query: *mut u16,
            key: *mut u16,
            cos_sin_cache: *const u16,
            rotary_dim: i32,
            total_q_dim: i32,
            total_k_dim: i32,
            head_size: i32,
            num_tokens: i32,
            stream: *mut std::ffi::c_void,
        );
        pub fn fused_qkv_rope_cache_bf16(
            q_out: *mut u16,
            key_cache: *mut u16,
            value_cache: *mut u16,
            qkv: *const u16,
            positions: *const u32,
            cos_sin_cache: *const u16,
            slot_mapping: *const i64,
            q_size: i32,
            kv_size: i32,
            total_dim: i32,
            rotary_dim: i32,
            head_size: i32,
            num_tokens: i32,
            stream: *mut std::ffi::c_void,
        );
    }

    // CP5: standalone CUTLASS GEMM launchers — all tile configs.
    // C[M,N] = alpha * A[M,K] @ B[K,N]^T + beta * C[M,N]
    macro_rules! cutlass_gemm_ffi {
        ($($name:ident),* $(,)?) => {
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
        // CUTLASS GEMV (M=1 specialization)
        cutlass_gemv_launch,
        // Legacy aliases
        cutlass_gemm_128x128_launch,
        cutlass_gemm_64x64_launch,
    );

    decl_scheduled_megakernel!(
        launch_scheduled_megakernel_tiny,
        launch_scheduled_megakernel_tiny_per_wave,
        launch_scheduled_megakernel_tiny_per_kind,
        scheduled_megakernel_tiny_num_nodes,
        scheduled_megakernel_tiny_num_ctas,
        scheduled_megakernel_tiny_num_waves
    );
    decl_scheduled_megakernel!(
        launch_scheduled_megakernel_medium,
        launch_scheduled_megakernel_medium_per_wave,
        launch_scheduled_megakernel_medium_per_kind,
        scheduled_megakernel_medium_num_nodes,
        scheduled_megakernel_medium_num_ctas,
        scheduled_megakernel_medium_num_waves
    );
    decl_scheduled_megakernel!(
        launch_scheduled_megakernel_llama_3_2_1b_seq64,
        launch_scheduled_megakernel_llama_3_2_1b_seq64_per_wave,
        launch_scheduled_megakernel_llama_3_2_1b_seq64_per_kind,
        scheduled_megakernel_llama_3_2_1b_seq64_num_nodes,
        scheduled_megakernel_llama_3_2_1b_seq64_num_ctas,
        scheduled_megakernel_llama_3_2_1b_seq64_num_waves
    );
    decl_scheduled_megakernel!(
        launch_scheduled_megakernel_llama_3_2_1b_seq1024,
        launch_scheduled_megakernel_llama_3_2_1b_seq1024_per_wave,
        launch_scheduled_megakernel_llama_3_2_1b_seq1024_per_kind,
        scheduled_megakernel_llama_3_2_1b_seq1024_num_nodes,
        scheduled_megakernel_llama_3_2_1b_seq1024_num_ctas,
        scheduled_megakernel_llama_3_2_1b_seq1024_num_waves
    );

    // Phase A.3 — FlashInfer attention runner shim. Standalone path that
    // calls flashinfer::BatchPagedAttentionPersistent end-to-end via
    // csrc/flashinfer_attention_shim.cu. Used by the smoke test to validate
    // the runner against a CPU reference BEFORE we wire it into
    // tile_attention. Pointer types are u16 because cudarc has no native
    // bf16 type — bf16 buffers are uploaded/downloaded via reinterpret.
    unsafe extern "C" {
        /// Returns 0 on success. Non-zero error codes match
        /// `FlashInferShimStatus` in the C++ shim.
        pub fn run_flashinfer_attention_smoke(
            q: *mut u16,          // device  [seq_len, num_qo_heads, head_dim]
            k: *mut u16,          // device  [num_pages, page_size, num_kv_heads, head_dim]
            v: *mut u16,          // device  same layout as k
            kv_indices: *mut i32, // device  [num_pages]
            o: *mut u16,          // device  [seq_len, num_qo_heads, head_dim]
            seq_len: i32,
            num_qo_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            page_size: i32,
            num_pages: i32,
            // Workspaces sized by the caller (typically derived from
            // `TargetProfile::flashinfer_*_workspace_bytes`). No
            // magic numbers in the C++ shim.
            float_ws_bytes: usize,
            int_ws_bytes: usize,
            sm_scale: f32,
            stream: u64,
        ) -> i32;
    }

    /// Opaque handle returned by `setup_flashinfer_params_for_megakernel`
    /// — owns the float / int workspace buffers and the device-resident
    /// `PersistentParams[NUM_LAYERS]` array. Mirrors the C struct
    /// `FlashInferAttentionPlan` in
    /// `csrc/flashinfer_attention_shim.cu`. Free with
    /// `teardown_flashinfer_attention_plan`.
    #[repr(C)]
    pub struct FlashInferAttentionPlan {
        pub float_ws_d: *mut std::ffi::c_void,
        pub int_ws_d: *mut std::ffi::c_void,
        pub int_ws_h: *mut std::ffi::c_void,
        pub params_d: *mut std::ffi::c_void, // PersistentParams[num_layers]
        // CP5-D-1: cooperative grid dims for per-layer FlashInfer
        // launches via `cp5_run_flashinfer_attention_for_layer`.
        pub num_blks_x: i32,
        pub num_blks_y: i32,
    }

    impl Default for FlashInferAttentionPlan {
        fn default() -> Self {
            Self {
                float_ws_d: std::ptr::null_mut(),
                int_ws_d: std::ptr::null_mut(),
                int_ws_h: std::ptr::null_mut(),
                params_d: std::ptr::null_mut(),
                num_blks_x: 0,
                num_blks_y: 0,
            }
        }
    }

    unsafe extern "C" {
        /// Build the per-launch FlashInfer plan and per-layer
        /// `PersistentParams` array for the scheduled megakernel.
        /// Call once per launch; pass `out_plan->params_d` as the
        /// `flashinfer_params` arg to the megakernel launcher; free
        /// with `teardown_flashinfer_attention_plan` after the launch
        /// has synchronized.
        pub fn setup_flashinfer_params_for_megakernel(
            q_post_rope: *mut u16,
            k_cache_layer0: *mut u16,
            v_cache_layer0: *mut u16,
            kv_indices: *mut i32,
            attn_out: *mut u16,
            seq_len: i32,
            num_qo_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            page_size: i32,
            pages_per_layer: i32,
            num_layers: i32,
            // = TargetProfile::cooperative_grid_size(); must equal
            // the megakernel's NUM_CTAS so the planner's
            // work_indptr[blockIdx.y] indexing covers all the
            // planned work.
            target_num_clusters: i32,
            // Workspaces sized by the caller (typically derived from
            // `TargetProfile::flashinfer_*_workspace_bytes`). No
            // magic numbers in the C++ shim.
            float_ws_bytes: usize,
            int_ws_bytes: usize,
            sm_scale: f32,
            stream: u64,
            out_plan: *mut FlashInferAttentionPlan,
        ) -> i32;

        pub fn teardown_flashinfer_attention_plan(plan: *mut FlashInferAttentionPlan);

        /// CP5-D-1: per-layer FlashInfer launcher. Reuses the per-launch
        /// plan built by `setup_flashinfer_params_for_megakernel` (single
        /// PersistentParams per layer in `plan->params_d`) and dispatches
        /// the persistent runner via a tiny `__global__` wrapper that
        /// calls `BlockBatchPagedAttentionPersistent::Run` for one layer.
        /// Same code path as the megakernel's inline FlashInfer call,
        /// just dispatched from the host one layer at a time. Picked
        /// from the CP5 interpreter's flashinfer_standalone_fa2 dispatch.
        pub fn cp5_run_flashinfer_attention_for_layer(
            plan: *mut FlashInferAttentionPlan,
            layer_idx: i32,
            stream: *mut std::ffi::c_void,
        ) -> i32;
    }
}
