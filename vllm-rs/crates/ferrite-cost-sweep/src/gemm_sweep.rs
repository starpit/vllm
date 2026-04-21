// SPDX-License-Identifier: Apache-2.0
//! GEMM sweep: cuBLAS baseline + every CUTLASS tile variant in
//! `ferrite_kernels::cutlass::CUTLASS_TILE_ZOO` + GEMV @ M=1 +
//! fused CUTLASS SiLU×Mul epilogue, across a dense `(M, N, K)` grid
//! covering LLaMA 1B–70B model shapes.
//!
//! Ports the GEMM portion of the pre-refactor sweep in
//! `ferrite-test-harness/tests/gpu_cost_sweep.rs` (deleted Step G),
//! trimmed to the 17 variants the current solver registers in its
//! `CutlassGemmImpl` library entries.
//!
//! Elementwise ops that the solver picks as singletons
//! (rms_norm_bf16, silu_and_mul_fused_bf16) and the fused QKV RoPE
//! kernel are swept too — their cost rows flow into `cost_gemm` /
//! `elementwise_cost` via the same `CostTable`.

#![cfg(feature = "cuda")]

use cudarc::cublas::sys as cublas;
use cudarc::driver::sys;
use ferrite_kernels::cutlass::{
    cutlass_gemm_32x64_s3_launch, cutlass_gemm_32x64_s4_launch, cutlass_gemm_32x128_s3_launch,
    cutlass_gemm_32x128_s4_launch, cutlass_gemm_32x256_s3_launch, cutlass_gemm_64x64_s3_launch,
    cutlass_gemm_64x64_s4_launch, cutlass_gemm_64x64_s4_sk2_launch,
    cutlass_gemm_64x64_s4_sk4_launch, cutlass_gemm_64x64_s4_sk8_launch,
    cutlass_gemm_64x128_s3_launch, cutlass_gemm_64x128_s4_launch,
    cutlass_gemm_64x128_s4_sk2_launch, cutlass_gemm_64x128_s4_sk4_launch,
    cutlass_gemm_64x128_s4_sk8_launch, cutlass_gemm_128x64_s3_launch,
    cutlass_gemm_128x64_s4_launch, cutlass_gemm_128x64_s4_sk2_launch,
    cutlass_gemm_128x64_s4_sk4_launch, cutlass_gemm_128x64_s4_sk8_launch,
    cutlass_gemm_128x128_s3_launch, cutlass_gemm_128x128_s4_launch,
    cutlass_gemm_128x128_s4_sk2_launch, cutlass_gemm_128x128_s4_sk4_launch,
    cutlass_gemm_128x128_s4_sk8_launch, cutlass_gemm_128x256_s3_launch,
    cutlass_gemm_256x64_s3_launch, cutlass_gemm_256x64_s4_launch, cutlass_gemm_bias_launch,
    cutlass_gemm_silu_mul_launch, cutlass_gemv_launch,
};

use crate::util::{bench_kernel, gpu_alloc_zeros};

// Elementwise kernels live in vllm-cuda's csrc and are declared
// extern-private inside ferrite-kernels::kernels. The cost-sweep
// needs its own decls since it times the raw kernels at the extern
// level (no OwnedTensor wrapping).
#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn rms_norm_bf16(
        out: *mut u16,
        input: *const u16,
        weight: *const u16,
        epsilon: f32,
        weight_offset: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: sys::CUstream,
    );

    fn silu_and_mul_fused_bf16(
        out: *mut u16,
        input: *const u16,
        num_tokens: i32,
        d: i32,
        stream: sys::CUstream,
    );

    fn fused_qkv_rope_bf16(
        q_out: *mut u16,
        k_out: *mut u16,
        v_out: *mut u16,
        qkv: *const u16,
        positions: *const u16,
        cos_sin: *const u16,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        rotary_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: sys::CUstream,
    );
}

const WARMUP: u32 = 5;
const ITERS: u32 = 5;

/// Shapes covering the four per-layer GEMMs (QKV, O, gate|up, down)
/// across every arch in `ferrite-models`. Columns are `(N, K)`.
///
/// Without the small-hidden block, CutlassGemmImpl gets
/// `UNCALIBRATED_COST_US` on SmolLM-135M / Qwen2-0.5B / Qwen3-0.6B
/// shapes while GemmRefImpl falls back to an analytical cuBLAS cost
/// — guaranteeing cuBLAS wins on small arches regardless of kernel
/// reality. Every Gemm tile the test goldens exercise needs a row
/// here.
const NK_SHAPES: &[(u32, u32)] = &[
    // ── Tiny models (hidden < 1k) ──
    // SmolLM-135M: hidden=576, intermediate=1536, qkv_size=(576+2*192)=960.
    (576, 576),
    (960, 576),
    (1536, 576),
    (576, 1536),
    (3072, 576),
    // SmolLM-360M: hidden=960, intermediate=2560, qkv_size=(960+2*320)=1600.
    (960, 960),
    (1600, 960),
    (2560, 960),
    (960, 2560),
    (5120, 960),
    // ── Small models (1–2B) ──
    // Qwen2-0.5B: hidden=896, intermediate=4864, qkv=(896+2*128)=1152.
    (896, 896),
    (1152, 896),
    (4864, 896),
    (896, 4864),
    (9728, 896),
    // Qwen3-0.6B: hidden=1024, intermediate=3072, qkv=(2048+2*1024)=4096.
    (1024, 1024),
    (4096, 1024),
    (3072, 1024),
    (1024, 3072),
    (6144, 1024),
    // Granite 3.1-2B / gemma2-2b-ish: hidden=2048 already below.
    // ── Medium-small (1B–3B, hidden=2048–3072) ──
    (2048, 2048),
    (3072, 2048),
    (3072, 3072),
    (4608, 3072),
    (5632, 2048),
    (2048, 5632),
    (8192, 2048),
    (8192, 3072),
    (2048, 8192),
    (3072, 8192),
    // ── 7B–13B (hidden=4096) ──
    (4096, 4096),
    (6144, 4096),
    (11008, 4096),
    (14336, 4096),
    (4096, 11008),
    (4096, 14336),
    // ── 30B–70B (hidden=8192) ──
    (8192, 8192),
    (10240, 8192),
    (28672, 8192),
    (8192, 28672),
];

/// `num_tokens` grid — matches the solver's default workload sweep.
/// GEMV (M=1) + every standard cutlass tile (M∈{2..4096}) get
/// calibrated at each point.
const M_VALUES: &[u32] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

/// Hidden sizes used for the elementwise sweeps (RMSNorm, SiLU×Mul).
const HIDDEN_SIZES: &[u32] = &[2048, 3072, 4096, 8192];

/// Named RoPE configs — `(name, q_size, kv_size, head_size, rotary_dim)`.
/// Keep `name` aligned with the CSV rows the solver reads in its
/// `elementwise_cost` helper.
const ROPE_CONFIGS: &[(&str, u32, u32, u32, u32)] = &[
    ("rope_1b", 2048, 512, 64, 64),
    ("rope_7b", 4096, 1024, 128, 128),
    ("rope_70b", 8192, 1024, 128, 128),
];

/// Run the sweep. Writes CSV rows to stdout. `launch_overhead_us` is
/// subtracted from every measurement so reported timings are
/// compute-only — matches the convention `CostTable` expects.
pub fn run(launch_overhead_us: f64) {
    let stream: sys::CUstream = std::ptr::null_mut();

    // cuBLAS handle (dropped at end of fn). Default stream is fine —
    // we only enqueue one launch at a time.
    let mut handle: cublas::cublasHandle_t = std::ptr::null_mut();
    unsafe {
        let s = cublas::cublasCreate_v2(&mut handle);
        assert_eq!(s, cublas::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        // `cublas::sys::cudaStream_t` and `driver::sys::CUstream` are
        // the same underlying pointer type with different wrapper
        // structs — cast through raw pointer.
        let s = cublas::cublasSetStream_v2(handle, stream as *mut _);
        assert_eq!(s, cublas::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }

    for &(n, k) in NK_SHAPES {
        for &m in M_VALUES {
            bench_one_shape(handle, stream, m, n, k, launch_overhead_us);
        }
    }

    unsafe { cublas::cublasDestroy_v2(handle) };

    sweep_elementwise(stream, launch_overhead_us);
    sweep_rope(stream, launch_overhead_us);
}

fn bench_one_shape(
    handle: cublas::cublasHandle_t,
    stream: sys::CUstream,
    m: u32,
    n: u32,
    k: u32,
    launch_overhead_us: f64,
) {
    // Input/output buffers for the whole shape. All kernels share
    // the same A/B/C layout: A is `[M, K]` row-major, B is `[N, K]`
    // row-major (cuBLAS-style weight — B^T is the math operand),
    // C is `[M, N]`.
    let a = gpu_alloc_zeros((m * k) as usize * 2);
    let b = gpu_alloc_zeros((n * k) as usize * 2);
    let c = gpu_alloc_zeros((m * n) as usize * 2);

    let m_i = m as i32;
    let n_i = n as i32;
    let k_i = k as i32;
    let one: f32 = 1.0;
    let zero: f32 = 0.0;

    // ── cuBLAS baseline ──
    let cublas_gemm = || unsafe {
        cublas::cublasGemmEx(
            handle,
            cublas::cublasOperation_t::CUBLAS_OP_T,
            cublas::cublasOperation_t::CUBLAS_OP_N,
            n_i,
            m_i,
            k_i,
            &one as *const f32 as *const _,
            b as *const _,
            cublas::cudaDataType_t::CUDA_R_16BF,
            k_i,
            a as *const _,
            cublas::cudaDataType_t::CUDA_R_16BF,
            k_i,
            &zero as *const f32 as *const _,
            c as *mut _,
            cublas::cudaDataType_t::CUDA_R_16BF,
            n_i,
            cublas::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        );
    };
    let us = (bench_kernel(stream, WARMUP, ITERS, cublas_gemm) - launch_overhead_us).max(0.0);
    println!("cublas,{m},{n},{k},{us:.1}");

    // ── CUTLASS tile zoo — the 16 variants the current solver registers ──
    macro_rules! bench_cutlass {
        ($($name:literal => $fn:ident),* $(,)?) => {
            $(
                let us = (bench_kernel(stream, WARMUP, ITERS, || unsafe {
                    $fn(
                        c as *mut u16, a as *const u16, b as *const u16,
                        m_i, n_i, k_i, 1.0, 0.0, stream as u64,
                    );
                }) - launch_overhead_us).max(0.0);
                println!(concat!($name, ",{},{},{},{:.1}"), m, n, k, us);
            )*
        };
    }
    bench_cutlass!(
        "cutlass_32x64_s3"   => cutlass_gemm_32x64_s3_launch,
        "cutlass_32x64_s4"   => cutlass_gemm_32x64_s4_launch,
        "cutlass_32x128_s3"  => cutlass_gemm_32x128_s3_launch,
        "cutlass_32x128_s4"  => cutlass_gemm_32x128_s4_launch,
        "cutlass_32x256_s3"  => cutlass_gemm_32x256_s3_launch,
        "cutlass_64x64_s3"   => cutlass_gemm_64x64_s3_launch,
        "cutlass_64x64_s4"   => cutlass_gemm_64x64_s4_launch,
        "cutlass_64x128_s3"  => cutlass_gemm_64x128_s3_launch,
        "cutlass_64x128_s4"  => cutlass_gemm_64x128_s4_launch,
        "cutlass_128x64_s3"  => cutlass_gemm_128x64_s3_launch,
        "cutlass_128x64_s4"  => cutlass_gemm_128x64_s4_launch,
        "cutlass_128x128_s3" => cutlass_gemm_128x128_s3_launch,
        "cutlass_128x128_s4" => cutlass_gemm_128x128_s4_launch,
        "cutlass_128x256_s3" => cutlass_gemm_128x256_s3_launch,
        "cutlass_256x64_s3"  => cutlass_gemm_256x64_s3_launch,
        "cutlass_256x64_s4"  => cutlass_gemm_256x64_s4_launch,
    );

    // ── CUTLASS tile zoo, beta=1.0 residual-add variant ──
    //
    // Same kernel family, but `beta=1.0` — epilogue aux-reads `C` then
    // writes `alpha*A*B + beta*C` back. Measured separately because
    // the extra `[M, N]` read adds a real BW cost the DP needs to
    // account for when comparing `CutlassGemmAddImpl` against the
    // alternative `(singleton gemm + fused_add_rms_norm)` path on
    // residual-stream chains.
    //
    // `c` is pre-populated (zeros here; the measurement is stable
    // regardless of residual contents since the epilogue's work is
    // bounded by shape, not value).
    macro_rules! bench_cutlass_add {
        ($($name:literal => $fn:ident),* $(,)?) => {
            $(
                let us = (bench_kernel(stream, WARMUP, ITERS, || unsafe {
                    $fn(
                        c as *mut u16, a as *const u16, b as *const u16,
                        m_i, n_i, k_i, 1.0, 1.0, stream as u64,
                    );
                }) - launch_overhead_us).max(0.0);
                println!(concat!($name, ",{},{},{},{:.1}"), m, n, k, us);
            )*
        };
    }
    bench_cutlass_add!(
        "cutlass_32x64_s3_add"   => cutlass_gemm_32x64_s3_launch,
        "cutlass_32x64_s4_add"   => cutlass_gemm_32x64_s4_launch,
        "cutlass_32x128_s3_add"  => cutlass_gemm_32x128_s3_launch,
        "cutlass_32x128_s4_add"  => cutlass_gemm_32x128_s4_launch,
        "cutlass_32x256_s3_add"  => cutlass_gemm_32x256_s3_launch,
        "cutlass_64x64_s3_add"   => cutlass_gemm_64x64_s3_launch,
        "cutlass_64x64_s4_add"   => cutlass_gemm_64x64_s4_launch,
        "cutlass_64x128_s3_add"  => cutlass_gemm_64x128_s3_launch,
        "cutlass_64x128_s4_add"  => cutlass_gemm_64x128_s4_launch,
        "cutlass_128x64_s3_add"  => cutlass_gemm_128x64_s3_launch,
        "cutlass_128x64_s4_add"  => cutlass_gemm_128x64_s4_launch,
        "cutlass_128x128_s3_add" => cutlass_gemm_128x128_s3_launch,
        "cutlass_128x128_s4_add" => cutlass_gemm_128x128_s4_launch,
        "cutlass_128x256_s3_add" => cutlass_gemm_128x256_s3_launch,
        "cutlass_256x64_s3_add"  => cutlass_gemm_256x64_s3_launch,
        "cutlass_256x64_s4_add"  => cutlass_gemm_256x64_s4_launch,
    );

    // ── CUTLASS SplitK parallel variants ──
    //
    // Narrow 4-tile × 3-split grid picked to cover tall-skinny
    // small-N/large-K shapes where the standard tile zoo leaves
    // cuBLAS winning (e.g. Qwen2-0.5B down_proj @ prefill).
    //
    // Each kernel needs an f32 scratch of `split_k × M × N × 4` bytes
    // (GemmSplitKParallel contract). We allocate the largest at sk=8
    // once and reuse it across splits — sizing matches the safe
    // wrapper's `alloc.alloc_tensor(&[sk*M*N], F32)` at runtime.
    let ws_bytes = 8usize * (m as usize) * (n as usize) * 4;
    let splitk_ws = gpu_alloc_zeros(ws_bytes);
    macro_rules! bench_cutlass_splitk {
        ($($name:literal => $fn:ident),* $(,)?) => {
            $(
                let us = (bench_kernel(stream, WARMUP, ITERS, || unsafe {
                    $fn(
                        c as *mut u16, a as *const u16, b as *const u16,
                        m_i, n_i, k_i, 1.0, 0.0,
                        splitk_ws as *mut u8,
                        stream as u64,
                    );
                }) - launch_overhead_us).max(0.0);
                println!(concat!($name, ",{},{},{},{:.1}"), m, n, k, us);
            )*
        };
    }
    bench_cutlass_splitk!(
        "cutlass_64x64_s4_split2"   => cutlass_gemm_64x64_s4_sk2_launch,
        "cutlass_64x64_s4_split4"   => cutlass_gemm_64x64_s4_sk4_launch,
        "cutlass_64x64_s4_split8"   => cutlass_gemm_64x64_s4_sk8_launch,
        "cutlass_64x128_s4_split2"  => cutlass_gemm_64x128_s4_sk2_launch,
        "cutlass_64x128_s4_split4"  => cutlass_gemm_64x128_s4_sk4_launch,
        "cutlass_64x128_s4_split8"  => cutlass_gemm_64x128_s4_sk8_launch,
        "cutlass_128x64_s4_split2"  => cutlass_gemm_128x64_s4_sk2_launch,
        "cutlass_128x64_s4_split4"  => cutlass_gemm_128x64_s4_sk4_launch,
        "cutlass_128x64_s4_split8"  => cutlass_gemm_128x64_s4_sk8_launch,
        "cutlass_128x128_s4_split2" => cutlass_gemm_128x128_s4_sk2_launch,
        "cutlass_128x128_s4_split4" => cutlass_gemm_128x128_s4_sk4_launch,
        "cutlass_128x128_s4_split8" => cutlass_gemm_128x128_s4_sk8_launch,
    );

    // ── GEMV (M=1 only) ──
    // SIMT kernel specialised for batch-1 decode.
    if m == 1 {
        let us = (bench_kernel(stream, WARMUP, ITERS, || unsafe {
            cutlass_gemv_launch(
                c as *mut u16,
                a as *const u16,
                b as *const u16,
                m_i,
                n_i,
                k_i,
                1.0,
                0.0,
                stream as u64,
            );
        }) - launch_overhead_us)
            .max(0.0);
        println!("cutlass_gemv,{m},{n},{k},{us:.1}");
    }

    // ── CUTLASS fused Gate+Up+Silu+Mul (EVT epilogue) ──
    //
    // Picked by `CutlassFusedGateUpSiluMulImpl` at emit time. The
    // measurement here covers ONLY the gate GEMM + silu+mul epilogue;
    // the up-projection GEMM is a separate standalone cutlass call
    // whose cost is captured by the tile zoo rows above. `cost_us`
    // sums both halves from the CSV at solver time.
    //
    // Kernel signature: d[M,N] = silu(a @ b_gate^T) * c_up, where
    // N = intermediate_size (the gate width), K = hidden_size.
    // For sweep purposes we reuse the outer loop's (N, K) — MLP
    // shapes populate correct rows, non-MLP shapes are dead but
    // harmless (matcher rejects them).
    let c_up = gpu_alloc_zeros((m * n) as usize * 2);
    let us = (bench_kernel(stream, WARMUP, ITERS, || unsafe {
        cutlass_gemm_silu_mul_launch(
            c as *mut u16,
            a as *const u16,
            b as *const u16,
            c_up as *mut u16,
            m_i,
            n_i,
            k_i,
            stream as u64,
        );
    }) - launch_overhead_us)
        .max(0.0);
    println!("cutlass_fused_gate_up_silu_mul,{m},{n},{k},{us:.1}");

    // ── CUTLASS fused GEMM + bias (EVT row-broadcast) ──
    //
    // Picked by `CutlassFusedGemmBiasImpl`. Kernel writes
    // D[M,N] = A @ B^T + bias[N] in one launch; bias is [N] bf16.
    let bias_n = gpu_alloc_zeros(n as usize * 2);
    let us = (bench_kernel(stream, WARMUP, ITERS, || unsafe {
        cutlass_gemm_bias_launch(
            c as *mut u16,
            a as *const u16,
            b as *const u16,
            bias_n as *const u16,
            m_i,
            n_i,
            k_i,
            stream as u64,
        );
    }) - launch_overhead_us)
        .max(0.0);
    println!("cutlass_fused_gemm_bias,{m},{n},{k},{us:.1}");

    unsafe {
        sys::cuMemFree_v2(a);
        sys::cuMemFree_v2(b);
        sys::cuMemFree_v2(c);
        sys::cuMemFree_v2(c_up);
        sys::cuMemFree_v2(bias_n);
        sys::cuMemFree_v2(splitk_ws);
    }
}

fn sweep_elementwise(stream: sys::CUstream, launch_overhead_us: f64) {
    for &hidden in HIDDEN_SIZES {
        for &m in M_VALUES {
            let input = gpu_alloc_zeros((m * hidden) as usize * 2);
            let output = gpu_alloc_zeros((m * hidden) as usize * 2);
            let weight = gpu_alloc_zeros(hidden as usize * 2);

            // RMSNorm. N column = hidden, K column = 0 (convention:
            // elementwise ops that fold K into their cost use K=0).
            let rms_us = (bench_kernel(stream, WARMUP, ITERS, || unsafe {
                rms_norm_bf16(
                    output as *mut u16,
                    input as *const u16,
                    weight as *const u16,
                    1e-5,
                    0.0,
                    m as i32,
                    hidden as i32,
                    stream,
                );
            }) - launch_overhead_us)
                .max(0.0);
            println!("rms_norm,{m},{hidden},0,{rms_us:.2}");

            // SiLU×Mul. Input is [M, 2*hidden] gate-up packed buffer.
            let gate_up = gpu_alloc_zeros((m * hidden * 2) as usize * 2);
            let silu_us = (bench_kernel(stream, WARMUP, ITERS, || unsafe {
                silu_and_mul_fused_bf16(
                    output as *mut u16,
                    gate_up as *const u16,
                    m as i32,
                    hidden as i32,
                    stream,
                );
            }) - launch_overhead_us)
                .max(0.0);
            println!("silu_mul,{m},{hidden},0,{silu_us:.2}");

            unsafe {
                sys::cuMemFree_v2(input);
                sys::cuMemFree_v2(output);
                sys::cuMemFree_v2(weight);
                sys::cuMemFree_v2(gate_up);
            }
        }
    }
}

fn sweep_rope(stream: sys::CUstream, launch_overhead_us: f64) {
    // `cos_sin_cache` is `[max_pos, rotary_dim]`. We just need it
    // allocated — the kernel only reads `rotary_dim` entries per
    // token so the position values don't matter for timing.
    const MAX_POS: u32 = 4096;

    for &(name, q_size, kv_size, head_size, rotary_dim) in ROPE_CONFIGS {
        let total_dim = q_size + 2 * kv_size;
        let cos_sin = gpu_alloc_zeros((MAX_POS * rotary_dim) as usize * 2);
        for &m in M_VALUES {
            let qkv = gpu_alloc_zeros((m * total_dim) as usize * 2);
            let q_out = gpu_alloc_zeros((m * q_size) as usize * 2);
            let k_out = gpu_alloc_zeros((m * kv_size) as usize * 2);
            let v_out = gpu_alloc_zeros((m * kv_size) as usize * 2);
            // positions: i64 per token.
            let positions = gpu_alloc_zeros(m as usize * 8);

            let us = (bench_kernel(stream, WARMUP, ITERS, || unsafe {
                fused_qkv_rope_bf16(
                    q_out as *mut u16,
                    k_out as *mut u16,
                    v_out as *mut u16,
                    qkv as *const u16,
                    positions as *const u16,
                    cos_sin as *const u16,
                    q_size as i32,
                    kv_size as i32,
                    total_dim as i32,
                    rotary_dim as i32,
                    head_size as i32,
                    m as i32,
                    stream,
                );
            }) - launch_overhead_us)
                .max(0.0);
            // N = total_dim (fused QKV width); K = rotary_dim.
            println!("{name},{m},{total_dim},{rotary_dim},{us:.2}");

            unsafe {
                sys::cuMemFree_v2(qkv);
                sys::cuMemFree_v2(q_out);
                sys::cuMemFree_v2(k_out);
                sys::cuMemFree_v2(v_out);
                sys::cuMemFree_v2(positions);
            }
        }
        unsafe { sys::cuMemFree_v2(cos_sin) };
    }
}
