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
//!   - `crates/ferrite-test-harness/csrc/flashinfer_attention_shim.cu`

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
        // CUTLASS SplitK variants — K-reduction across CTAs
        cutlass_gemm_64x64_s4_sk2_launch,
        cutlass_gemm_64x64_s4_sk4_launch,
        cutlass_gemm_64x64_s4_sk8_launch,
        cutlass_gemm_128x128_s3_sk2_launch,
        cutlass_gemm_128x128_s3_sk4_launch,
        cutlass_gemm_128x128_s3_sk8_launch,
        cutlass_gemm_64x128_s4_sk2_launch,
        cutlass_gemm_64x128_s4_sk4_launch,
        cutlass_gemm_64x128_s4_sk8_launch,
        cutlass_gemm_32x64_s4_sk4_launch,
        cutlass_gemm_32x64_s4_sk8_launch,
        cutlass_gemm_32x64_s4_sk16_launch,
        // Round 2: closing cuBLAS gaps at M=128/1024
        cutlass_gemm_128x128_s4_sk2_launch,
        cutlass_gemm_128x128_s4_sk4_launch,
        cutlass_gemm_128x128_s4_sk8_launch,
        cutlass_gemm_128x64_s4_sk2_launch,
        cutlass_gemm_128x64_s4_sk4_launch,
        cutlass_gemm_128x64_s4_sk8_launch,
        cutlass_gemm_256x64_s4_sk2_launch,
        cutlass_gemm_256x64_s4_sk4_launch,
        cutlass_gemm_64x64_s4_sk16_launch,
        cutlass_gemm_64x128_s4_sk16_launch,
        // TB_K=64 variants
        cutlass_gemm_64x64_k64_s4_launch,
        cutlass_gemm_64x64_k64_s3_launch,
        cutlass_gemm_64x128_k64_s4_launch,
        cutlass_gemm_64x128_k64_s3_launch,
        cutlass_gemm_128x64_k64_s4_launch,
        cutlass_gemm_128x64_k64_s3_launch,
        cutlass_gemm_128x128_k64_s4_launch,
        cutlass_gemm_128x128_k64_s3_launch,
        cutlass_gemm_256x64_k64_s4_launch,
        cutlass_gemm_256x64_k64_s3_launch,
        cutlass_gemm_128x256_k64_s3_launch,
        cutlass_gemm_32x64_k64_s4_launch,
        cutlass_gemm_32x128_k64_s4_launch,
        // TB_K=64 splitK variants
        cutlass_gemm_64x64_k64_s4_sk2_launch,
        cutlass_gemm_64x64_k64_s4_sk4_launch,
        cutlass_gemm_64x64_k64_s4_sk8_launch,
        cutlass_gemm_128x128_k64_s3_sk2_launch,
        cutlass_gemm_128x128_k64_s3_sk4_launch,
        // k64 stages=2 variants
        cutlass_gemm_128x128_k64_s2_launch,
        cutlass_gemm_64x64_k64_s2_launch,
        cutlass_gemm_64x128_k64_s2_launch,
        cutlass_gemm_128x64_k64_s2_launch,
        cutlass_gemm_256x64_k64_s2_launch,
        // 256×128 s2
        cutlass_gemm_256x128_s2_launch,
        // stages=2 variants
        cutlass_gemm_64x64_s2_launch,
        cutlass_gemm_64x128_s2_launch,
        cutlass_gemm_128x64_s2_launch,
        cutlass_gemm_128x128_s2_launch,
        cutlass_gemm_128x256_s2_launch,
        cutlass_gemm_256x64_s2_launch,
        // Deep pipeline variants (stages=5-10)
        cutlass_gemm_64x128_s5_launch,
        cutlass_gemm_64x128_s6_launch,
        cutlass_gemm_64x128_s7_launch,
        cutlass_gemm_64x128_s8_launch,
        cutlass_gemm_128x128_s5_launch,
        cutlass_gemm_128x128_s6_launch,
        cutlass_gemm_64x64_s5_launch,
        cutlass_gemm_64x64_s6_launch,
        cutlass_gemm_64x64_s8_launch,
        cutlass_gemm_64x64_s10_launch,
        cutlass_gemm_128x64_s5_launch,
        cutlass_gemm_128x64_s6_launch,
        cutlass_gemm_128x64_s7_launch,
        cutlass_gemm_128x64_s8_launch,
        cutlass_gemm_64x256_s4_launch,
        cutlass_gemm_64x256_s5_launch,
        cutlass_gemm_128x256_s4_launch,
        cutlass_gemm_256x64_s5_launch,
        cutlass_gemm_256x64_s6_launch,
        // CTA-swizzled variants (L2 locality optimization)
        cutlass_gemm_128x128_sw_s4_launch,
        cutlass_gemm_128x128_sw_s3_launch,
        cutlass_gemm_128x128_sw_s2_launch,
        cutlass_gemm_128x256_sw_s3_launch,
        cutlass_gemm_128x256_sw_s2_launch,
        cutlass_gemm_256x64_sw_s4_launch,
        cutlass_gemm_256x64_sw_s3_launch,
        cutlass_gemm_64x128_sw_s4_launch,
        cutlass_gemm_64x128_sw_s3_launch,
        // 64×256 tile (cuBLAS heuristic pick for M=128 K=8192)
        cutlass_gemm_64x256_s3_launch,
        cutlass_gemm_64x256_s2_launch,
        cutlass_gemm_64x256_sw_s3_launch,
        cutlass_gemm_64x256_sw_s2_launch,
        // 64×256 splitK
        cutlass_gemm_64x256_s3_sk2_launch,
        cutlass_gemm_64x256_s3_sk4_launch,
        cutlass_gemm_64x256_s3_sk8_launch,
        // 8-warp (256 thread) variants — matching cuBLAS thread count
        cutlass_gemm_64x128_k64w8_s3_launch,
        cutlass_gemm_64x128_k64w8_s4_launch,
        cutlass_gemm_64x128_k64w8_s2_launch,
        cutlass_gemm_64x128_w8_s3_launch,
        cutlass_gemm_64x128_w8_s4_launch,
        cutlass_gemm_64x128_w8_s5_launch,
        cutlass_gemm_64x128_w8_s6_launch,
        cutlass_gemm_128x128_w8_s3_launch,
        cutlass_gemm_128x128_w8_s4_launch,
        cutlass_gemm_128x128_w8_s5_launch,
        // Deep pipeline splitK
        cutlass_gemm_64x128_s5_sk4_launch,
        cutlass_gemm_64x128_s6_sk4_launch,
        cutlass_gemm_64x128_s6_sk2_launch,
        cutlass_gemm_64x128_s6_sk8_launch,
        cutlass_gemm_64x128_s7_sk4_launch,
        cutlass_gemm_64x128_s8_sk4_launch,
        cutlass_gemm_128x128_s5_sk4_launch,
        cutlass_gemm_128x128_s5_sk8_launch,
        cutlass_gemm_128x64_s5_sk4_launch,
        cutlass_gemm_128x64_s6_sk4_launch,
    );

    // ── CUTLASS 3.x sm90 (Hopper) launchers — wgmma + TMA. ──
    //
    // Only available when compiled on sm90+ (build.rs sets `cuda_arch_sm90`).
    // ── ThunderKittens GEMM launchers — persistent-grid wgmma+TMA. ──
    #[cfg(cuda_arch_sm90)]
    cutlass_gemm_ffi!(
        tk_gemm_128x256_launch,
        tk_gemm_128x128_launch,
        tk_gemm_64x256_launch,
        tk_gemm_64x128_launch,
        tk_gemm_64x64_launch,
    );

    #[cfg(cuda_arch_sm90)]
    cutlass_gemm_ffi!(
        cutlass_sm90_gemm_128x128_coop_launch,
        cutlass_sm90_gemm_128x256_coop_launch,
        cutlass_sm90_gemm_256x128_coop_launch,
        cutlass_sm90_gemm_64x128_ws_launch,
        cutlass_sm90_gemm_128x64_ws_launch,
        cutlass_sm90_gemm_64x64_ws_launch,
        cutlass_sm90_gemm_128x128_pp_launch,
        cutlass_sm90_gemm_64x128_pp_launch,
        cutlass_sm90_gemm_128x64_pp_launch,
        cutlass_sm90_gemm_128x128_c2x1_launch,
        cutlass_sm90_gemm_128x256_c2x1_launch,
    );

    // ── CUTLASS GEMM + SiLU + Mul (EVT epilogue fusion) ──
    unsafe extern "C" {
        pub fn cutlass_gemm_silu_mul_launch(
            d: *mut u16,        // [M, N] output: silu(gate) * up
            a: *const u16,      // [M, K] normed hidden states
            b_gate: *const u16, // [N, K] gate weight
            c_up: *mut u16,     // [M, N] up-projection output (aux)
            m: i32,
            n: i32,
            k: i32,
            stream: u64,
        ) -> i32;
    }

    // ── Null kernel for measuring launch overhead ──
    unsafe extern "C" {
        pub fn null_kernel_launch(stream: u64);
    }

    // ── Barrier / handoff microbenchmarks (barrier_sweep.cu) ──
    unsafe extern "C" {
        /// Cooperative kernel: N grid-wide barriers.
        /// Returns cudaError_t (0 = success).
        pub fn grid_sync_bench_launch(
            n_iters: u32,
            grid_size: u32,
            block_size: u32,
            stream: u64,
        ) -> i32;

        /// mbarrier handoff: producer→consumer N times (sm90+ only).
        /// Returns -1 on unsupported arch.
        pub fn mbarrier_handoff_bench_launch(n_iters: u32, stream: u64) -> i32;

        /// gmem flag spin: producer atomicExch, consumer polls, N times.
        pub fn gmem_flag_bench_launch(n_iters: u32, flag_ptr: u64, stream: u64) -> i32;

        /// __syncthreads() baseline: N block barriers.
        pub fn syncthreads_bench_launch(n_iters: u32, block_size: u32, stream: u64);
    }

    // ── Elementwise kernels (from vllm-cuda/csrc/) ──
    //
    // Used by `gpu_cost_sweep.rs` to measure RMSNorm, SiLU+Mul,
    // and RoPE costs on each GPU target. These symbols come from
    // the `libvllm_kernels.a` static lib linked by build.rs.
    unsafe extern "C" {
        /// RMS norm: out[num_tokens, hidden_size] = rms_norm(input, weight, eps).
        pub fn rms_norm_bf16(
            out: *mut u16,
            input: *const u16,
            weight: *const u16,
            epsilon: f32,
            num_tokens: i32,
            hidden_size: i32,
            stream: u64,
        );

        /// Fused SiLU + element-wise mul: out[num_tokens, d] = silu(gate_up[..,:d]) * gate_up[..,d:].
        pub fn silu_and_mul_fused_bf16(
            out: *mut u16,
            gate_up: *const u16,
            num_tokens: i32,
            d: i32,
            stream: u64,
        );

        /// Fused QKV split + RoPE: splits QKV and applies rotary embeddings.
        #[allow(clippy::too_many_arguments)]
        pub fn fused_qkv_rope_bf16(
            q_out: *mut u16,
            k_out: *mut u16,
            v_out: *mut u16,
            qkv: *const u16,
            positions: *const u16, // actually i64 positions
            cos_sin_cache: *const u16,
            q_size: i32,
            kv_size: i32,
            total_dim: i32,
            rotary_dim: i32,
            head_size: i32,
            num_tokens: i32,
            stream: u64,
        );
    }

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
