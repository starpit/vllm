// SPDX-License-Identifier: Apache-2.0
//! FFI declarations for solver-adjacent GPU test kernels.
//!
//! Two consumers:
//!   - `tests/gpu_cost_sweep.rs` — benchmarks cuBLAS + the full CUTLASS grid
//!     + GEMV, producing the CSV the solver's cost model reads.
//!   - `tests/flashinfer_attention_test.rs` — exercises the FlashInfer
//!     attention shim end-to-end against a CPU reference.
//!
//! The .cu files behind these FFIs are compiled by `build.rs`:
//!   - `crates/vllm-cuda/csrc/cutlass_standalone_gemm.cu` (canonical)
//!   - `crates/vllm-tk-test-harness/csrc/flashinfer_attention_shim.cu`

#[cfg(feature = "cuda")]
pub mod ffi {
    // ── cuBLAS FFI — the GEMM baseline in gpu_cost_sweep. ──
    //
    // cuBLAS exposes a stable C API; we declare just what the bench needs
    // and link dynamically against libcublas.so.12 (build.rs adds -lcublas).
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

    // ── CUTLASS standalone GEMM launchers — all tile configs + GEMV. ──
    //
    // C[M,N] = alpha * A[M,K] @ B[K,N]^T + beta * C[M,N]
    //
    // Each symbol is stamped out by the `CUTLASS_GEMM(...)` macro in
    // `vllm-cuda/csrc/cutlass_standalone_gemm.cu`. Adding a new tile
    // config is a one-line entry there plus a line here.
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
    );

    // ── FlashInfer attention shim — standalone smoke path. ──
    //
    // Exercises `flashinfer::BatchPagedAttentionPersistent` via
    // `csrc/flashinfer_attention_shim.cu` with synthetic inputs at
    // LLaMA-1B head dims. Used by `flashinfer_attention_test.rs` to
    // validate the runner against a CPU reference.
    unsafe extern "C" {
        /// Returns 0 on success.
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
            // Must match the `num_sm` value the caller used to size the
            // workspaces above. The shim passes this through to its
            // forked `TwoStageHolisticPlanWithNumSm` planner so the
            // planner's internal cluster count is guaranteed to agree
            // with the allocated workspace — otherwise the planner's
            // bump-allocator overruns the buffer and the persistent
            // runner aborts. Typically
            // `device_num_sms * cooperative_blocks_per_sm`, i.e. what
            // `TargetProfile::cooperative_grid_size()` returns.
            target_num_clusters: i32,
            sm_scale: f32,
            stream: u64,
        ) -> i32;
    }
}
