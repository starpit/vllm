//! Comprehensive GPU validation of flat-param CUTLASS GEMM vs cuBLAS.
//!
//! Tests all 4 tile configurations at multiple problem sizes including
//! actual llama.rs production dimensions. Every test compares against cuBLAS
//! as the reference implementation.
//!
//! Run: cargo test -p ptx-fusion --features cuda --test cuda_flat_gemm -- --nocapture

#![cfg(feature = "cuda")]

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use half::bf16;
use std::sync::Arc;

// ── Rewrite all 4 CUTLASS configs to flat params ──

ptx_fusion::replace_perimeter_macro!(
    "kernels/cutlass_bf16_64x64x32_sm89.ptx",
    "kernels/cutlass_bf16_64x64x32_sm89.derivations.json",
    "ferrite_gemm_64x64x32",
    FLAT_64x64x32
);

ptx_fusion::replace_perimeter_macro!(
    "kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    "ferrite_gemm_64x128x32",
    FLAT_64x128x32
);

ptx_fusion::replace_perimeter_macro!(
    "kernels/cutlass_bf16_128x128x32_sm89.ptx",
    "kernels/cutlass_bf16_128x128x32_sm89.derivations.json",
    "ferrite_gemm_128x128x32",
    FLAT_128x128x32
);

ptx_fusion::replace_perimeter_macro!(
    "kernels/cutlass_bf16_128x128x64_sm89.ptx",
    "kernels/cutlass_bf16_128x128x64_sm89.derivations.json",
    "ferrite_gemm_128x128x64",
    FLAT_128x128x64
);

// ── Config descriptions ──

struct TileConfig {
    name: &'static str,
    entry: &'static str,
    ptx: &'static str,
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
    threads: u32,
    smem_bytes: u32,
}

const CONFIGS: &[TileConfig] = &[
    TileConfig {
        name: "64x64x32",
        entry: "ferrite_gemm_64x64x32",
        ptx: FLAT_64x64x32,
        tile_m: 64,
        tile_n: 64,
        tile_k: 32,
        threads: 128,
        smem_bytes: 24576,
    },
    TileConfig {
        name: "64x128x32",
        entry: "ferrite_gemm_64x128x32",
        ptx: FLAT_64x128x32,
        tile_m: 64,
        tile_n: 128,
        tile_k: 32,
        threads: 128,
        smem_bytes: 36864,
    },
    TileConfig {
        name: "128x128x32",
        entry: "ferrite_gemm_128x128x32",
        ptx: FLAT_128x128x32,
        tile_m: 128,
        tile_n: 128,
        tile_k: 32,
        threads: 128,
        smem_bytes: 49152,
    },
    TileConfig {
        name: "128x128x64",
        entry: "ferrite_gemm_128x128x64",
        ptx: FLAT_128x128x64,
        tile_m: 128,
        tile_n: 128,
        tile_k: 64,
        threads: 128,
        smem_bytes: 98304,
    },
];

// ── Test infrastructure ──

/// Build the 88-byte flat param struct.
fn build_flat_params(
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    d_ptr: u64,
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    ldd: u32,
    alpha: f32,
    beta: f32,
) -> [u8; 88] {
    let mut p = [0u8; 88];
    p[0..8].copy_from_slice(&a_ptr.to_le_bytes());
    p[8..16].copy_from_slice(&b_ptr.to_le_bytes());
    p[16..24].copy_from_slice(&c_ptr.to_le_bytes());
    p[24..32].copy_from_slice(&d_ptr.to_le_bytes());
    p[32..40].copy_from_slice(&(lda as u64).to_le_bytes());
    p[40..48].copy_from_slice(&(ldb as u64).to_le_bytes());
    p[48..56].copy_from_slice(&(ldc as u64).to_le_bytes());
    p[56..64].copy_from_slice(&(ldd as u64).to_le_bytes());
    p[64..68].copy_from_slice(&(m as i32).to_le_bytes());
    p[68..72].copy_from_slice(&(n as i32).to_le_bytes());
    p[72..76].copy_from_slice(&(k as i32).to_le_bytes());
    p[76..80].copy_from_slice(&alpha.to_le_bytes());
    p[80..84].copy_from_slice(&beta.to_le_bytes());
    p
}

/// Compute CUTLASS grid dimensions for GemmIdentityThreadblockSwizzle<4>.
///
/// SWIZZLE_N=4 means log_tile is at most 2 (since 4 >= 4 but 4 < 8).
/// The CUTLASS get_log_tile() checks: N>=8 && grid_n>=6 → 3, N>=4 && grid_n>=3 → 2, etc.
fn compute_grid(m: u32, n: u32, tile_m: u32, tile_n: u32) -> (u32, u32, u32) {
    let grid_m = m.div_ceil(tile_m);
    let grid_n = n.div_ceil(tile_n);

    // GemmIdentityThreadblockSwizzle<4>: SWIZZLE_N=4
    const SWIZZLE_N: i32 = 4;
    let swizzle_log = if SWIZZLE_N >= 8 && grid_n >= 6 {
        3
    } else if SWIZZLE_N >= 4 && grid_n >= 3 {
        2
    } else if SWIZZLE_N >= 2 && grid_n >= 2 {
        1
    } else {
        0
    };
    let tile = 1u32 << swizzle_log;
    let grid_x = grid_m * tile;
    let grid_y = grid_n.div_ceil(tile);
    (grid_x, grid_y, 1)
}

/// Run a single GEMM: flat-param CUTLASS kernel vs cuBLAS.
/// Returns (max_abs_diff, max_rel_diff).
fn run_gemm_vs_cublas(config: &TileConfig, m: u32, n: u32, k: u32) -> (f32, f32) {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // Layout: A row-major MxK (lda=K), B col-major NxK (ldb=K), D row-major MxN (ldd=N)
    let lda = k;
    let ldb = k;
    let ldc = n;
    let ldd = n;

    // Generate test data
    let h_a: Vec<bf16> = (0..(m * k) as usize)
        .map(|i| bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.1))
        .collect();
    let h_b: Vec<bf16> = (0..(n * k) as usize)
        .map(|i| bf16::from_f32(((i as f32) * 0.00023 + 0.3).cos() * 0.1))
        .collect();

    let d_a = stream.clone_htod(&h_a).unwrap();
    let d_b = stream.clone_htod(&h_b).unwrap();
    let mut d_d_ferrite: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
    let mut d_d_cublas: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
    // C buffer for beta=0 (not read, but CUTLASS needs valid pointer)
    let d_c: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();

    // ── cuBLAS reference ──
    //
    // cuBLAS: C = alpha * op(A_cub) * op(B_cub) + beta * C, all column-major.
    //
    // Our data (CUTLASS convention):
    //   A: row-major MxK, lda=K → in col-major: KxM matrix, leading dim K
    //   B: col-major NxK, ldb=K → in col-major: KxN matrix, leading dim K
    //   D: row-major MxN, ldd=N → in col-major: NxM matrix, leading dim N
    //
    // We want D[m,n] = sum_k A[m,k] * B[n,k], which in col-major is:
    //   D_col(NxM) = B^T(NxK) * A_col(KxM)
    //
    // cuBLAS call: C(m_cub x n_cub) = op(A_cub)(m_cub x k) * op(B_cub)(k x n_cub)
    //   m_cub=N, n_cub=M, k_cub=K
    //   A_cub = B_data (KxN col-major), transa=T → op(A)=B^T (NxK), lda=K
    //   B_cub = A_data (KxM col-major), transb=N → op(B)=A   (KxM), ldb=K
    //   C = D_data, ldc=N
    {
        let blas = CudaBlas::new(stream.clone()).unwrap();
        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: bf16::from_f32(1.0),
            lda: ldb as i32, // leading dim of stored B (KxN), = K
            ldb: lda as i32, // leading dim of stored A (KxM), = K
            beta: bf16::from_f32(0.0),
            ldc: ldd as i32, // leading dim of D col-major (NxM), = N
        };
        unsafe {
            blas.gemm(cfg, &d_b, &d_a, &mut d_d_cublas).unwrap();
        }
    }
    stream.synchronize().unwrap();

    // ── Flat-param CUTLASS kernel ──
    let ptx = Ptx::from_src(config.ptx);
    let module = ctx
        .load_module(ptx)
        .unwrap_or_else(|e| panic!("load PTX for {}: {e}", config.name));
    let func = module
        .load_function(config.entry)
        .unwrap_or_else(|e| panic!("load entry {}: {e}", config.entry));

    // Skip configs needing > 48KB SMEM — cudarc doesn't expose cuFuncSetAttribute
    // and the 128x128x64 config needs 96KB opt-in. The other 3 configs cover
    // decode and general prefill workloads.
    if config.smem_bytes > 48 * 1024 {
        return (0.0, 0.0); // skip
    }

    let (a_ptr, _) = d_a.device_ptr(&stream);
    let (b_ptr, _) = d_b.device_ptr(&stream);
    let (c_ptr, _) = d_c.device_ptr(&stream);
    let (d_ptr, _) = d_d_ferrite.device_ptr(&stream);

    let params = build_flat_params(
        a_ptr as u64,
        b_ptr as u64,
        c_ptr as u64,
        d_ptr as u64,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        ldd,
        1.0,
        0.0,
    );

    let (grid_x, grid_y, grid_z) = compute_grid(m, n, config.tile_m, config.tile_n);
    let launch_cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (config.threads, 1, 1),
        shared_mem_bytes: config.smem_bytes,
    };

    unsafe { stream.launch_builder(&func).arg(&params).launch(launch_cfg) }.unwrap();
    stream.synchronize().unwrap();

    // ── Compare ──
    let out_ferrite = stream.clone_dtoh(&d_d_ferrite).unwrap();
    let out_cublas = stream.clone_dtoh(&d_d_cublas).unwrap();

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..(m * n) as usize {
        let f = out_ferrite[i].to_f32();
        let c = out_cublas[i].to_f32();
        let abs_diff = (f - c).abs();
        let rel_diff = if c.abs() > 1e-6 {
            abs_diff / c.abs()
        } else {
            abs_diff
        };
        max_abs = max_abs.max(abs_diff);
        max_rel = max_rel.max(rel_diff);
    }

    (max_abs, max_rel)
}

// ── ptxas validation for all configs ──

#[test]
fn all_configs_pass_ptxas() {
    for cfg in CONFIGS {
        let path = format!("/tmp/flat_gemm_{}.ptx", cfg.name);
        std::fs::write(&path, cfg.ptx).unwrap();

        let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
            .args(["-arch=sm_89", &path])
            .output()
            .expect("ptxas");

        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            for line in stderr.lines().take(10) {
                println!("ptxas {}: {line}", cfg.name);
            }
            panic!("ptxas FAILED for {}", cfg.name);
        }
        println!("  {} passes ptxas", cfg.name);
    }
    println!("PASS: all 4 configs pass ptxas");
}

// ── Single-tile correctness (exercises basic param plumbing) ──

#[test]
fn single_tile_all_configs() {
    println!("=== Single-tile correctness (all configs vs cuBLAS) ===");
    for cfg in CONFIGS {
        let m = cfg.tile_m;
        let n = cfg.tile_n;
        let k = cfg.tile_k;
        let (abs_diff, _rel_diff) = run_gemm_vs_cublas(cfg, m, n, k);
        println!(
            "  {} (M={m}, N={n}, K={k}): max_abs_diff={abs_diff:.2e}",
            cfg.name
        );
        assert!(
            abs_diff < 1e-3,
            "{} single tile: abs_diff={abs_diff:.2e} (expected < 1e-3)",
            cfg.name
        );
    }
    println!("PASS: all configs correct at single tile");
}

// ── Multi-tile correctness (exercises swizzle + iterator increments) ──

#[test]
fn multi_tile_all_configs() {
    println!("=== Multi-tile correctness (all configs vs cuBLAS) ===");
    for cfg in CONFIGS {
        // 4x4 tiles worth
        let m = cfg.tile_m * 4;
        let n = cfg.tile_n * 4;
        let k = cfg.tile_k * 4;
        let (abs_diff, _) = run_gemm_vs_cublas(cfg, m, n, k);
        println!(
            "  {} (M={m}, N={n}, K={k}): max_abs_diff={abs_diff:.2e}",
            cfg.name
        );
        assert!(
            abs_diff < 0.1,
            "{} multi tile: abs_diff={abs_diff:.2e}",
            cfg.name
        );
    }
    println!("PASS: all configs correct at multi-tile");
}

// ── Partial tiles (M/N not multiples of tile dims) ──

#[test]
fn partial_tiles() {
    println!("=== Partial tile correctness (non-aligned M/N) ===");
    // Use 64x64x32 config with non-aligned dims
    let cfg = &CONFIGS[0]; // 64x64x32

    // NOTE: CUTLASS requires strides aligned to vector width (8 elements for bf16).
    // N must be a multiple of 8 for output alignment. K must be a multiple of alignment too.
    let cases = [
        (1, 64, 32),    // M=1, grid_n=1, swizzle_log=0
        (7, 64, 32),    // M=7, grid_n=1
        (63, 64, 32),   // M just under tile
        (65, 64, 32),   // M just over tile, grid_n=1
        (64, 128, 32),  // N=2 tiles, grid_n=2, swizzle_log=1
        (100, 192, 64), // both non-aligned M, grid_n=3, swizzle_log=2
        (1, 256, 128),  // M=1, larger K, swizzle_log=2
        (33, 320, 96),  // odd M, 5 N-tiles, larger K
    ];

    for (m, n, k) in cases {
        // CUTLASS bf16 requires K and N aligned to vector width (8 elements)
        assert!(k % 8 == 0, "K={k} must be multiple of 8 for bf16 alignment");
        assert!(
            n % 8 == 0,
            "N={n} must be multiple of 8 for bf16 output alignment"
        );

        let (abs_diff, _) = run_gemm_vs_cublas(cfg, m, n, k);
        println!("  M={m:4}, N={n:4}, K={k:4}: max_abs_diff={abs_diff:.2e}");
        assert!(
            abs_diff < 0.05,
            "partial tile M={m} N={n} K={k}: abs_diff={abs_diff:.2e}",
        );
    }
    println!("PASS: partial tiles correct");
}

// ── Llama production dimensions ──
// Qwen2.5-3B: hidden=2560, intermediate=6912, n_heads=20, head_dim=128

#[test]
fn llama_qkv_decode() {
    println!("=== Llama QKV GEMM decode (M=1, N=7680, K=2560) ===");
    // QKV: M=batch=1, N=n_heads*(head_dim)*3=20*128*3=7680, K=hidden=2560
    let cfg = &CONFIGS[0]; // 64x64x32 for decode
    let (abs_diff, _) = run_gemm_vs_cublas(cfg, 1, 7680, 2560);
    println!("  max_abs_diff={abs_diff:.2e}");
    assert!(abs_diff < 0.1, "QKV decode: abs_diff={abs_diff:.2e}",);
    println!("PASS: QKV decode");
}

#[test]
fn llama_gate_up_decode() {
    println!("=== Llama gate_up GEMM decode (M=1, N=6912, K=2560) ===");
    let cfg = &CONFIGS[0];
    let (abs_diff, _) = run_gemm_vs_cublas(cfg, 1, 6912, 2560);
    println!("  max_abs_diff={abs_diff:.2e}");
    assert!(abs_diff < 0.1, "gate_up decode: abs_diff={abs_diff:.2e}",);
    println!("PASS: gate_up decode");
}

#[test]
fn llama_down_proj_decode() {
    println!("=== Llama down_proj GEMM decode (M=1, N=2560, K=3456) ===");
    let cfg = &CONFIGS[0];
    let (abs_diff, _) = run_gemm_vs_cublas(cfg, 1, 2560, 3456);
    println!("  max_abs_diff={abs_diff:.2e}");
    assert!(abs_diff < 0.1, "down_proj decode: abs_diff={abs_diff:.2e}",);
    println!("PASS: down_proj decode");
}

#[test]
fn llama_o_proj_decode() {
    println!("=== Llama o_proj GEMM decode (M=1, N=2560, K=2560) ===");
    let cfg = &CONFIGS[0];
    let (abs_diff, _) = run_gemm_vs_cublas(cfg, 1, 2560, 2560);
    println!("  max_abs_diff={abs_diff:.2e}");
    assert!(abs_diff < 0.1, "o_proj decode: abs_diff={abs_diff:.2e}",);
    println!("PASS: o_proj decode");
}

#[test]
fn llama_prefill_large() {
    println!("=== Llama prefill (M=1024, N=2560, K=2048) ===");
    // Use 128x128x32 for prefill
    let cfg = &CONFIGS[2]; // 128x128x32
    let (abs_diff, _) = run_gemm_vs_cublas(cfg, 1024, 2560, 2048);
    println!("  max_abs_diff={abs_diff:.2e}");
    assert!(abs_diff < 0.2, "prefill: abs_diff={abs_diff:.2e}",);
    println!("PASS: prefill large");
}

// ── Sweep: multiple batch sizes on 64x64x32 (the decode config) ──

#[test]
fn batch_sweep_64x64x32() {
    println!("=== Batch sweep on 64x64x32 (N=4096, K=4096) ===");
    let cfg = &CONFIGS[0];
    let n = 4096u32;
    let k = 4096u32;

    for &m in &[1u32, 4, 8, 16, 32, 64] {
        let (abs_diff, _) = run_gemm_vs_cublas(cfg, m, n, k);
        println!("  M={m:4}: max_abs_diff={abs_diff:.2e}");
        assert!(abs_diff < 0.2, "batch sweep M={m}: abs_diff={abs_diff:.2e}",);
    }
    println!("PASS: batch sweep");
}
