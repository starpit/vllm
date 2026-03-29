//! Test ferrite GEMM against cuBLAS using real model weights.
//!
//! Loads actual Qwen2.5-0.5B-Instruct weights from disk and compares
//! the ferrite flat-param CUTLASS GEMM output against cuBLAS at the
//! exact production dimensions with the exact production launch path.
//!
//! Run: cargo test -p vllm-cuda --features ferrite --test ferrite_gemm_real_weights -- --nocapture

#![cfg(feature = "ferrite")]

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::sys as cusys;
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use half::bf16;
use safetensors::SafeTensors;
use std::sync::Arc;

const MODEL_PATH: &str = concat!(
    env!("HOME"),
    "/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/",
    "snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors"
);

// The same flat-param CUTLASS kernel used in production
ptx_fusion::replace_perimeter_macro!(
    "../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    "ferrite_gemm_64x128x32",
    FLAT_GEMM_PTX
);

fn load_bf16_tensor(st: &SafeTensors, name: &str) -> Vec<bf16> {
    let tensor = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("missing tensor {name}: {e}"));
    assert_eq!(
        tensor.dtype(),
        safetensors::Dtype::BF16,
        "{name} is not bf16"
    );
    tensor
        .data()
        .chunks_exact(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]))
        .collect()
}

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

fn compute_grid(m: u32, n: u32, tile_m: u32, tile_n: u32) -> (u32, u32, u32) {
    let grid_m = m.div_ceil(tile_m);
    let grid_n = n.div_ceil(tile_n);
    let swizzle_log = if grid_n >= 3 {
        2
    } else if grid_n >= 2 {
        1
    } else {
        0
    };
    let tile = 1u32 << swizzle_log;
    (grid_m * tile, grid_n.div_ceil(tile), 1)
}

/// Launch ferrite GEMM using the EXACT same mechanism as production:
/// CU_LAUNCH_PARAM_BUFFER_POINTER with 88-byte aligned buffer.
unsafe fn launch_ferrite_gemm_production(
    func: cusys::CUfunction,
    stream: cusys::CUstream,
    params: &[u8; 88],
    m: u32,
    n: u32,
    tile_m: u32,
    tile_n: u32,
    threads: u32,
    smem: u32,
) {
    let (gx, gy, _) = compute_grid(m, n, tile_m, tile_n);

    #[repr(align(8))]
    struct Aligned([u8; 88]);
    let aligned = Aligned(*params);

    let mut param_size = 88usize;
    let extra: [*mut std::ffi::c_void; 5] = [
        cusys::CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT as *mut _,
        aligned.0.as_ptr() as *mut _,
        cusys::CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT as *mut _,
        &mut param_size as *mut usize as *mut _,
        cusys::CU_LAUNCH_PARAM_END_AS_INT as *mut _,
    ];

    let result = cusys::cuLaunchKernel(
        func,
        gx,
        gy,
        1,
        threads,
        1,
        1,
        smem,
        stream,
        std::ptr::null_mut(),
        extra.as_ptr() as *mut *mut _,
    );
    assert_eq!(
        result,
        cusys::cudaError_enum::CUDA_SUCCESS,
        "ferrite cuLaunchKernel failed: {result:?}"
    );
}

/// Launch ferrite GEMM using cudarc's launch_builder (per-arg kernelParams).
/// This is how the GPU tests launch — known to produce correct results.
unsafe fn launch_ferrite_gemm_cudarc(
    func: &cudarc::driver::CudaFunction,
    stream: &cudarc::driver::CudaStream,
    params: &[u8; 88],
    m: u32,
    n: u32,
    tile_m: u32,
    tile_n: u32,
    threads: u32,
    smem: u32,
) {
    let (gx, gy, gz) = compute_grid(m, n, tile_m, tile_n);
    let cfg = LaunchConfig {
        grid_dim: (gx, gy, gz),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem,
    };
    stream.launch_builder(func).arg(params).launch(cfg).unwrap();
}

#[test]
fn test1_ferrite_gemm_vs_cublas_real_weights() {
    println!("=== Test 1: ferrite GEMM vs cuBLAS with real Qwen2.5-0.5B weights ===");

    // Load model weights
    let data = std::fs::read(MODEL_PATH).expect("failed to read model safetensors");
    let st = SafeTensors::deserialize(&data).expect("failed to parse safetensors");

    // Load layer 0 QKV weight: [1152, 896] bf16
    let qkv_weight = load_bf16_tensor(&st, "model.layers.0.self_attn.q_proj.weight");
    let k_weight = load_bf16_tensor(&st, "model.layers.0.self_attn.k_proj.weight");
    let v_weight = load_bf16_tensor(&st, "model.layers.0.self_attn.v_proj.weight");

    // Concat QKV weights: q [896, 896] + k [128, 896] + v [128, 896] = [1152, 896]
    let hidden = 896u32;
    let q_size = 896u32; // 14 heads * 64 head_dim
    let kv_size = 128u32; // 2 heads * 64 head_dim
    let qkv_n = q_size + 2 * kv_size; // 1152

    let mut h_weight = Vec::with_capacity((qkv_n * hidden) as usize);
    h_weight.extend_from_slice(&qkv_weight);
    h_weight.extend_from_slice(&k_weight);
    h_weight.extend_from_slice(&v_weight);
    assert_eq!(h_weight.len(), (qkv_n * hidden) as usize);

    // Create realistic input: small random values like embeddings
    let m = 128u32; // batch size
    let k = hidden;
    let n = qkv_n;

    let h_input: Vec<bf16> = (0..(m * k) as usize)
        .map(|i| bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.1))
        .collect();

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let d_input = stream.clone_htod(&h_input).unwrap();
    let d_weight = stream.clone_htod(&h_weight).unwrap();
    let mut d_out_cublas: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
    let d_out_ferrite_cudarc: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
    let d_out_ferrite_prod: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();

    let (a_ptr, _) = d_input.device_ptr(&stream);
    let (b_ptr, _) = d_weight.device_ptr(&stream);
    let (out_cublas_ptr, _) = d_out_cublas.device_ptr(&stream);
    let (out_cudarc_ptr, _) = d_out_ferrite_cudarc.device_ptr(&stream);
    let (out_prod_ptr, _) = d_out_ferrite_prod.device_ptr(&stream);

    // ── cuBLAS reference ──
    {
        let blas = CudaBlas::new(stream.clone()).unwrap();
        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: bf16::from_f32(1.0),
            lda: k as i32,
            ldb: k as i32,
            beta: bf16::from_f32(0.0),
            ldc: n as i32,
        };
        let mut d_out = d_out_cublas;
        unsafe { blas.gemm(cfg, &d_weight, &d_input, &mut d_out).unwrap() };
        d_out_cublas = d_out;
    }
    stream.synchronize().unwrap();

    // ── Ferrite via cudarc launch_builder (known working in tests) ──
    let module = ctx.load_module(Ptx::from_src(FLAT_GEMM_PTX)).unwrap();
    let func = module.load_function("ferrite_gemm_64x128x32").unwrap();

    let params_cudarc = build_flat_params(
        a_ptr as u64,
        b_ptr as u64,
        out_cudarc_ptr as u64,
        out_cudarc_ptr as u64,
        m,
        n,
        k,
        k,
        k,
        n,
        n,
        1.0,
        0.0,
    );
    unsafe {
        launch_ferrite_gemm_cudarc(&func, &stream, &params_cudarc, m, n, 64, 128, 128, 36864);
    }
    stream.synchronize().unwrap();

    // ── Ferrite via production CU_LAUNCH_PARAM_BUFFER_POINTER ──
    // Load module via raw driver API (same as ferrite.rs load_flat_module)
    let ptx_cstr = std::ffi::CString::new(FLAT_GEMM_PTX).unwrap();
    let mut raw_module: cusys::CUmodule = std::ptr::null_mut();
    let mut raw_func: cusys::CUfunction = std::ptr::null_mut();
    unsafe {
        let r = cusys::cuModuleLoadData(&mut raw_module, ptx_cstr.as_ptr() as *const _);
        assert_eq!(
            r,
            cusys::cudaError_enum::CUDA_SUCCESS,
            "cuModuleLoadData: {r:?}"
        );
        let entry = std::ffi::CString::new("ferrite_gemm_64x128x32").unwrap();
        let r = cusys::cuModuleGetFunction(&mut raw_func, raw_module, entry.as_ptr());
        assert_eq!(
            r,
            cusys::cudaError_enum::CUDA_SUCCESS,
            "cuModuleGetFunction: {r:?}"
        );
    }

    let params_prod = build_flat_params(
        a_ptr as u64,
        b_ptr as u64,
        out_prod_ptr as u64,
        out_prod_ptr as u64,
        m,
        n,
        k,
        k,
        k,
        n,
        n,
        1.0,
        0.0,
    );
    // Use a raw stream for the production launch path
    let mut raw_stream: cusys::CUstream = std::ptr::null_mut();
    unsafe {
        cusys::cuStreamCreate(&mut raw_stream, 0);
        launch_ferrite_gemm_production(
            raw_func,
            raw_stream,
            &params_prod,
            m,
            n,
            64,
            128,
            128,
            36864,
        );
        cusys::cuStreamSynchronize(raw_stream);
    }
    stream.synchronize().unwrap();

    // ── Compare all three ──
    let out_cublas = stream.clone_dtoh(&d_out_cublas).unwrap();
    let out_cudarc = stream.clone_dtoh(&d_out_ferrite_cudarc).unwrap();
    let out_prod = stream.clone_dtoh(&d_out_ferrite_prod).unwrap();

    let mut max_cudarc_vs_cublas = 0.0f32;
    let mut max_prod_vs_cublas = 0.0f32;
    let mut max_prod_vs_cudarc = 0.0f32;

    for i in 0..(m * n) as usize {
        let c = out_cublas[i].to_f32();
        let d = out_cudarc[i].to_f32();
        let p = out_prod[i].to_f32();
        max_cudarc_vs_cublas = max_cudarc_vs_cublas.max((d - c).abs());
        max_prod_vs_cublas = max_prod_vs_cublas.max((p - c).abs());
        max_prod_vs_cudarc = max_prod_vs_cudarc.max((p - d).abs());
    }

    println!("  M={m}, N={n}, K={k} (real QKV weights)");
    println!("  cudarc launch vs cuBLAS:     {max_cudarc_vs_cublas:.2e}");
    println!("  production launch vs cuBLAS: {max_prod_vs_cublas:.2e}");
    println!("  production vs cudarc:        {max_prod_vs_cudarc:.2e}");

    // Print first 8 values from each
    print!("  cuBLAS  [0..8]: ");
    for i in 0..8 {
        print!("{:.4} ", out_cublas[i].to_f32());
    }
    println!();
    print!("  cudarc  [0..8]: ");
    for i in 0..8 {
        print!("{:.4} ", out_cudarc[i].to_f32());
    }
    println!();
    print!("  prod    [0..8]: ");
    for i in 0..8 {
        print!("{:.4} ", out_prod[i].to_f32());
    }
    println!();

    assert!(
        max_cudarc_vs_cublas < 1.0,
        "cudarc launch disagrees with cuBLAS: {max_cudarc_vs_cublas:.2e}"
    );
    assert!(
        max_prod_vs_cublas < 1.0,
        "production launch disagrees with cuBLAS: {max_prod_vs_cublas:.2e}"
    );
    assert!(
        max_prod_vs_cudarc < 0.01,
        "production launch disagrees with cudarc: {max_prod_vs_cudarc:.2e}"
    );

    println!("PASS: ferrite GEMM matches cuBLAS with real weights");
}
