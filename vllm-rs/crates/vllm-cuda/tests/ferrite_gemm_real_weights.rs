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

// bf16 rms_norm kernel for reference path
ptx_fusion::extract_entry!(
    "../ptx-fusion/kernels/vllm_rms_norm.ptx",
    "_Z15rms_norm_kernelI13__nv_bfloat16EvPT_PKS1_S4_fi",
    RMS_NORM_BF16_PTX
);

// The fused norm+GEMM kernel from compile!
const FUSED_NORM_GEMM: ptx_fusion::FeriteKernel = ptx_fusion::compile!(
    a = rms_norm,
    b = gemm_64x128x32,
    bind = { a.output => b.param_0 },
    name = "test_fused_norm_gemm",
);

/// Test 2: fused norm+GEMM vs separate rms_norm + cuBLAS GEMM.
/// Uses real model weights and the EXACT production launch mechanism.
#[test]
fn test2_fused_norm_gemm_vs_separate_real_weights() {
    println!("=== Test 2: fused norm+GEMM vs separate, real weights ===");

    let data = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&data).expect("parse safetensors");

    let hidden = 896u32;
    let q_size = 896u32;
    let kv_size = 128u32;
    let n = q_size + 2 * kv_size; // 1152
    let k = hidden;
    let m = 128u32;
    let eps = 1e-6f32;

    // Load weights
    let qkv_weight = {
        let mut w = load_bf16_tensor(&st, "model.layers.0.self_attn.q_proj.weight");
        w.extend_from_slice(&load_bf16_tensor(
            &st,
            "model.layers.0.self_attn.k_proj.weight",
        ));
        w.extend_from_slice(&load_bf16_tensor(
            &st,
            "model.layers.0.self_attn.v_proj.weight",
        ));
        w
    };
    let norm_weight = load_bf16_tensor(&st, "model.layers.0.input_layernorm.weight");

    // Create input
    let h_input: Vec<bf16> = (0..(m * k) as usize)
        .map(|i| bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.1))
        .collect();

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let d_input = stream.clone_htod(&h_input).unwrap();
    let d_qkv_weight = stream.clone_htod(&qkv_weight).unwrap();
    let d_norm_weight = stream.clone_htod(&norm_weight).unwrap();

    // ── Reference: GPU rms_norm then cuBLAS GEMM ──
    let rms_module = ctx.load_module(Ptx::from_src(RMS_NORM_BF16_PTX)).unwrap();
    let rms_func = rms_module
        .load_function("_Z15rms_norm_kernelI13__nv_bfloat16EvPT_PKS1_S4_fi")
        .unwrap();

    let d_normed: CudaSlice<bf16> = stream.alloc_zeros((m * k) as usize).unwrap();
    let (inp_p, _) = d_input.device_ptr(&stream);
    let (nw_p, _) = d_norm_weight.device_ptr(&stream);
    let (normed_p, _) = d_normed.device_ptr(&stream);

    unsafe {
        stream
            .launch_builder(&rms_func)
            .arg(&normed_p) // output
            .arg(&inp_p) // input
            .arg(&nw_p) // weight
            .arg(&eps)
            .arg(&(k as i32))
            .launch(LaunchConfig {
                grid_dim: (m, 1, 1),
                block_dim: (256.min(k), 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .unwrap();

    // cuBLAS GEMM on normed data
    let mut d_ref_out: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
    {
        let blas = CudaBlas::new(stream.clone()).unwrap();
        let (qw_p, _) = d_qkv_weight.device_ptr(&stream);
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
        unsafe {
            blas.gemm(cfg, &d_qkv_weight, &d_normed, &mut d_ref_out)
                .unwrap()
        };
    }
    stream.synchronize().unwrap();

    // ── Fused: norm+GEMM via compile! kernel, production launch ──
    let d_fused_out: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
    let (qw_p, _) = d_qkv_weight.device_ptr(&stream);
    let (fused_out_p, _) = d_fused_out.device_ptr(&stream);

    // Build flat GEMM params (same as launch_fused_norm_gemm does)
    let flat = build_flat_params(
        inp_p as u64,       // A_ptr = input (norm reads from it)
        qw_p as u64,        // B_ptr = QKV weight
        fused_out_p as u64, // C_ptr
        fused_out_p as u64, // D_ptr
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

    // Load fused kernel via raw driver API (same as get_or_load_kernel)
    let ptx_cstr = std::ffi::CString::new(FUSED_NORM_GEMM.ptx).unwrap();
    let entry_cstr = std::ffi::CString::new(FUSED_NORM_GEMM.entry).unwrap();
    let mut fmod: cusys::CUmodule = std::ptr::null_mut();
    let mut ffunc: cusys::CUfunction = std::ptr::null_mut();
    unsafe {
        let r = cusys::cuModuleLoadData(&mut fmod, ptx_cstr.as_ptr() as *const _);
        assert_eq!(
            r,
            cusys::cudaError_enum::CUDA_SUCCESS,
            "load fused module: {r:?}"
        );
        let r = cusys::cuModuleGetFunction(&mut ffunc, fmod, entry_cstr.as_ptr());
        assert_eq!(
            r,
            cusys::cudaError_enum::CUDA_SUCCESS,
            "get fused func: {r:?}"
        );
    }

    // Pack params EXACTLY like launch_fused_norm_gemm in ferrite.rs
    let prefix = FUSED_NORM_GEMM.extra_param_bytes as usize;
    let total = prefix + 88;
    let mut params = vec![0u8; total];
    params[0..8].copy_from_slice(&(inp_p as u64).to_le_bytes());
    params[8..16].copy_from_slice(&(nw_p as u64).to_le_bytes());
    params[16..20].copy_from_slice(&eps.to_le_bytes());
    params[20..24].copy_from_slice(&k.to_le_bytes()); // hidden
    params[24..32].copy_from_slice(&(k as u64).to_le_bytes()); // a_stride
    params[prefix..prefix + 88].copy_from_slice(&flat);

    println!("  prefix={prefix}, total={total}");

    let (gx, gy, _) = compute_grid(m, n, 64, 128);

    // Launch via CU_LAUNCH_PARAM_BUFFER_POINTER (production path)
    unsafe {
        #[repr(align(8))]
        struct Aligned([u8; 256]);
        let mut aligned = Aligned([0u8; 256]);
        aligned.0[..total].copy_from_slice(&params);

        let mut param_size = total;
        let extra: [*mut std::ffi::c_void; 5] = [
            cusys::CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT as *mut _,
            aligned.0.as_ptr() as *mut _,
            cusys::CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT as *mut _,
            &mut param_size as *mut usize as *mut _,
            cusys::CU_LAUNCH_PARAM_END_AS_INT as *mut _,
        ];

        let mut raw_stream: cusys::CUstream = std::ptr::null_mut();
        cusys::cuStreamCreate(&mut raw_stream, 0);

        let r = cusys::cuLaunchKernel(
            ffunc,
            gx,
            gy,
            1,
            128,
            1,
            1,
            FUSED_NORM_GEMM.smem_bytes,
            raw_stream,
            std::ptr::null_mut(),
            extra.as_ptr() as *mut *mut _,
        );
        assert_eq!(
            r,
            cusys::cudaError_enum::CUDA_SUCCESS,
            "fused launch: {r:?}"
        );
        cusys::cuStreamSynchronize(raw_stream);
    }

    // Also launch via cudarc per-arg for comparison
    let d_fused_out2: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
    let (fused_out2_p, _) = d_fused_out2.device_ptr(&stream);
    let flat2 = build_flat_params(
        inp_p as u64,
        qw_p as u64,
        fused_out2_p as u64,
        fused_out2_p as u64,
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
    let fused_module = ctx.load_module(Ptx::from_src(FUSED_NORM_GEMM.ptx)).unwrap();
    let fused_func = fused_module.load_function(FUSED_NORM_GEMM.entry).unwrap();
    unsafe {
        stream
            .launch_builder(&fused_func)
            .arg(&(inp_p as u64))
            .arg(&(nw_p as u64))
            .arg(&eps)
            .arg(&k)
            .arg(&(k as u64))
            .arg(&flat2)
            .launch(LaunchConfig {
                grid_dim: (gx, gy, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: FUSED_NORM_GEMM.smem_bytes,
            })
    }
    .unwrap();
    stream.synchronize().unwrap();

    // Compare
    let ref_out = stream.clone_dtoh(&d_ref_out).unwrap();
    let fused_prod = stream.clone_dtoh(&d_fused_out).unwrap();
    let fused_cudarc = stream.clone_dtoh(&d_fused_out2).unwrap();

    let mut max_prod_vs_ref = 0.0f32;
    let mut max_cudarc_vs_ref = 0.0f32;
    let mut max_prod_vs_cudarc = 0.0f32;
    for i in 0..(m * n) as usize {
        let r = ref_out[i].to_f32();
        let p = fused_prod[i].to_f32();
        let c = fused_cudarc[i].to_f32();
        max_prod_vs_ref = max_prod_vs_ref.max((p - r).abs());
        max_cudarc_vs_ref = max_cudarc_vs_ref.max((c - r).abs());
        max_prod_vs_cudarc = max_prod_vs_cudarc.max((p - c).abs());
    }

    println!("  M={m}, N={n}, K={k} (real weights, eps={eps})");
    println!("  fused(cudarc) vs separate:   {max_cudarc_vs_ref:.2e}");
    println!("  fused(prod)   vs separate:   {max_prod_vs_ref:.2e}");
    println!("  fused(prod)   vs fused(cudarc): {max_prod_vs_cudarc:.2e}");

    print!("  separate[0..8]: ");
    for i in 0..8 {
        print!("{:.4} ", ref_out[i].to_f32());
    }
    println!();
    print!("  fused_cd[0..8]: ");
    for i in 0..8 {
        print!("{:.4} ", fused_cudarc[i].to_f32());
    }
    println!();
    print!("  fused_pr[0..8]: ");
    for i in 0..8 {
        print!("{:.4} ", fused_prod[i].to_f32());
    }
    println!();

    // Allow some tolerance for bf16 rms_norm differences
    let tol = 1.0 + (k as f32 / 512.0).ceil();
    assert!(
        max_cudarc_vs_ref < tol,
        "fused(cudarc) vs separate too large: {max_cudarc_vs_ref:.2e}"
    );
    assert!(
        max_prod_vs_ref < tol,
        "fused(prod) vs separate too large: {max_prod_vs_ref:.2e}"
    );
    assert!(
        max_prod_vs_cudarc < 0.01,
        "prod vs cudarc disagree: {max_prod_vs_cudarc:.2e}"
    );

    println!("PASS: fused norm+GEMM matches separate with real weights");
}

/// Test 3: gemm_accumulate (beta=1.0) — the o_proj path.
/// D = alpha * A @ B^T + beta * D (in-place accumulate).
#[test]
fn test3_gemm_accumulate_beta1_real_weights() {
    println!("=== Test 3: GEMM accumulate (beta=1.0) with real weights ===");

    let data = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&data).expect("parse safetensors");

    let hidden = 896u32;
    let q_size = 896u32;
    let m = 128u32;

    // o_proj weight: [hidden, q_size] = [896, 896]
    let o_weight = load_bf16_tensor(&st, "model.layers.0.self_attn.o_proj.weight");

    // Simulate: attn_output [M, q_size] and residual [M, hidden]
    let h_attn: Vec<bf16> = (0..(m * q_size) as usize)
        .map(|i| bf16::from_f32(((i as f32) * 0.00023 + 0.1).sin() * 0.5))
        .collect();
    let h_residual: Vec<bf16> = (0..(m * hidden) as usize)
        .map(|i| bf16::from_f32(((i as f32) * 0.00037 - 0.3).cos() * 2.0))
        .collect();

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let d_attn = stream.clone_htod(&h_attn).unwrap();
    let d_o_weight = stream.clone_htod(&o_weight).unwrap();

    // ── cuBLAS reference: D = A @ B^T + D ──
    let mut d_ref = stream.clone_htod(&h_residual).unwrap();
    {
        let blas = CudaBlas::new(stream.clone()).unwrap();
        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: hidden as i32,
            n: m as i32,
            k: q_size as i32,
            alpha: bf16::from_f32(1.0),
            lda: q_size as i32, // o_weight is [hidden, q_size], stored row-major
            ldb: q_size as i32,
            beta: bf16::from_f32(1.0), // accumulate!
            ldc: hidden as i32,
        };
        unsafe { blas.gemm(cfg, &d_o_weight, &d_attn, &mut d_ref).unwrap() };
    }
    stream.synchronize().unwrap();

    // ── Ferrite: production CU_LAUNCH_PARAM_BUFFER_POINTER with beta=1.0 ──
    let mut d_ferrite = stream.clone_htod(&h_residual).unwrap();
    let (a_ptr, _) = d_attn.device_ptr(&stream);
    let (b_ptr, _) = d_o_weight.device_ptr(&stream);
    let (d_ptr, _) = d_ferrite.device_ptr(&stream);

    let params = build_flat_params(
        a_ptr as u64,
        b_ptr as u64,
        d_ptr as u64,
        d_ptr as u64, // C = D (accumulate in-place)
        m,
        hidden,
        q_size,
        q_size,
        q_size,
        hidden,
        hidden,
        1.0,
        1.0, // alpha=1, beta=1
    );

    let ptx_cstr = std::ffi::CString::new(FLAT_GEMM_PTX).unwrap();
    let entry_cstr = std::ffi::CString::new("ferrite_gemm_64x128x32").unwrap();
    let mut raw_module: cusys::CUmodule = std::ptr::null_mut();
    let mut raw_func: cusys::CUfunction = std::ptr::null_mut();
    unsafe {
        cusys::cuModuleLoadData(&mut raw_module, ptx_cstr.as_ptr() as *const _);
        cusys::cuModuleGetFunction(&mut raw_func, raw_module, entry_cstr.as_ptr());

        let mut raw_stream: cusys::CUstream = std::ptr::null_mut();
        cusys::cuStreamCreate(&mut raw_stream, 0);
        launch_ferrite_gemm_production(
            raw_func, raw_stream, &params, m, hidden, 64, 128, 128, 36864,
        );
        cusys::cuStreamSynchronize(raw_stream);
    }

    let ref_out = stream.clone_dtoh(&d_ref).unwrap();
    let fer_out = stream.clone_dtoh(&d_ferrite).unwrap();

    let mut max_diff = 0.0f32;
    for i in 0..(m * hidden) as usize {
        let d = (ref_out[i].to_f32() - fer_out[i].to_f32()).abs();
        if d > max_diff {
            max_diff = d;
        }
    }

    println!("  M={m}, N={hidden}, K={q_size}, alpha=1, beta=1 (real o_proj weight)");
    println!("  ferrite vs cuBLAS: {max_diff:.2e}");
    print!("  cuBLAS [0..8]: ");
    for i in 0..8 {
        print!("{:.4} ", ref_out[i].to_f32());
    }
    println!();
    print!("  ferrite[0..8]: ");
    for i in 0..8 {
        print!("{:.4} ", fer_out[i].to_f32());
    }
    println!();

    assert!(
        max_diff < 1.0,
        "gemm_accumulate beta=1 broken: {max_diff:.2e}"
    );
    println!("PASS: gemm_accumulate beta=1.0 matches cuBLAS");
}

/// Test 5: the exact broken sequence. add residual+hs on CPU, then compare:
///   Path A: fused_add_rms_norm_inplace(hs, res) → CUTLASS GEMM on normed hs
///   Path B: fused_norm_gemm(res_after_add) — fused norm+GEMM on accumulated residual
///   Path C: rms_norm(res_after_add) → CUTLASS GEMM — separate norm+GEMM on same data
/// If B≠C, the fused_norm_gemm kernel itself is wrong at this scale.
/// If B=C but A≠B, the rms_norm implementations differ.
#[test]
fn test5_add_then_fused_norm_gemm() {
    println!("=== Test 5: add + fused_norm_gemm vs fused_add_rms_norm + GEMM ===");

    let data = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&data).expect("parse safetensors");

    let hidden = 896u32;
    let n = 1152u32;
    let k = hidden;
    let m = 1024u32; // production size
    let eps = 1e-6f32;

    let qkv_weight = {
        let mut w = load_bf16_tensor(&st, "model.layers.0.self_attn.q_proj.weight");
        w.extend_from_slice(&load_bf16_tensor(
            &st,
            "model.layers.0.self_attn.k_proj.weight",
        ));
        w.extend_from_slice(&load_bf16_tensor(
            &st,
            "model.layers.0.self_attn.v_proj.weight",
        ));
        w
    };
    let norm_weight = load_bf16_tensor(&st, "model.layers.0.input_layernorm.weight");

    // Simulate residual (large) and hidden_states (small) — like production
    let h_residual: Vec<bf16> = (0..(m * k) as usize)
        .map(|i| bf16::from_f32(((i as f32) * 0.00013 - 0.3).cos() * 5.0))
        .collect();
    let h_hs: Vec<bf16> = (0..(m * k) as usize)
        .map(|i| bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.1))
        .collect();

    // CPU add: res_added = residual + hidden_states
    let h_res_added: Vec<bf16> = h_residual
        .iter()
        .zip(h_hs.iter())
        .map(|(r, h)| bf16::from_f32(r.to_f32() + h.to_f32()))
        .collect();

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let d_qkv_weight = stream.clone_htod(&qkv_weight).unwrap();
    let d_norm_weight = stream.clone_htod(&norm_weight).unwrap();
    let (qw_p, _) = d_qkv_weight.device_ptr(&stream);
    let (nw_p, _) = d_norm_weight.device_ptr(&stream);

    // ── Path B: fused_norm_gemm on res_added ──
    let d_res_b = stream.clone_htod(&h_res_added).unwrap();
    let d_out_b: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
    let (res_b_p, _) = d_res_b.device_ptr(&stream);
    let (out_b_p, _) = d_out_b.device_ptr(&stream);

    let flat_b = build_flat_params(
        res_b_p as u64,
        qw_p as u64,
        out_b_p as u64,
        out_b_p as u64,
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

    let fused_module = ctx.load_module(Ptx::from_src(FUSED_NORM_GEMM.ptx)).unwrap();
    let fused_func = fused_module.load_function(FUSED_NORM_GEMM.entry).unwrap();
    unsafe {
        let (gx, gy, gz) = compute_grid(m, n, 64, 128);
        stream
            .launch_builder(&fused_func)
            .arg(&(res_b_p as u64))
            .arg(&(nw_p as u64))
            .arg(&eps)
            .arg(&k)
            .arg(&(k as u64))
            .arg(&flat_b)
            .launch(LaunchConfig {
                grid_dim: (gx, gy, gz),
                block_dim: (128, 1, 1),
                shared_mem_bytes: FUSED_NORM_GEMM.smem_bytes,
            })
    }
    .unwrap();

    // ── Path C: separate rms_norm + CUTLASS GEMM on same res_added ──
    let d_res_c = stream.clone_htod(&h_res_added).unwrap();
    let d_normed_c: CudaSlice<bf16> = stream.alloc_zeros((m * k) as usize).unwrap();
    let (res_c_p, _) = d_res_c.device_ptr(&stream);
    let (normed_c_p, _) = d_normed_c.device_ptr(&stream);

    let rms_module = ctx.load_module(Ptx::from_src(RMS_NORM_BF16_PTX)).unwrap();
    let rms_func = rms_module
        .load_function("_Z15rms_norm_kernelI13__nv_bfloat16EvPT_PKS1_S4_fi")
        .unwrap();
    unsafe {
        stream
            .launch_builder(&rms_func)
            .arg(&normed_c_p)
            .arg(&res_c_p)
            .arg(&nw_p)
            .arg(&eps)
            .arg(&(k as i32))
            .launch(LaunchConfig {
                grid_dim: (m, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .unwrap();

    let d_out_c: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
    let (out_c_p, _) = d_out_c.device_ptr(&stream);
    let flat_c = build_flat_params(
        normed_c_p as u64,
        qw_p as u64,
        out_c_p as u64,
        out_c_p as u64,
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
    let gemm_module = ctx.load_module(Ptx::from_src(FLAT_GEMM_PTX)).unwrap();
    let gemm_func = gemm_module.load_function("ferrite_gemm_64x128x32").unwrap();
    unsafe {
        let (gx, gy, gz) = compute_grid(m, n, 64, 128);
        stream
            .launch_builder(&gemm_func)
            .arg(&flat_c)
            .launch(LaunchConfig {
                grid_dim: (gx, gy, gz),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 36864,
            })
    }
    .unwrap();
    stream.synchronize().unwrap();

    let out_b = stream.clone_dtoh(&d_out_b).unwrap();
    let out_c = stream.clone_dtoh(&d_out_c).unwrap();

    let mut max_bc = 0.0f32;
    let mut worst = 0;
    for i in 0..(m * n) as usize {
        let b = out_b[i].to_f32();
        let c = out_c[i].to_f32();
        let d = (b - c).abs();
        if d > max_bc {
            max_bc = d;
            worst = i;
        }
    }

    println!("  M={m}, N={n}, K={k}");
    println!("  fused_norm_gemm vs separate norm+GEMM: {max_bc:.2e} at [{worst}]");
    print!("  fused [0..8]: ");
    for i in 0..8 {
        print!("{:.4} ", out_b[i].to_f32());
    }
    println!();
    print!("  separ [0..8]: ");
    for i in 0..8 {
        print!("{:.4} ", out_c[i].to_f32());
    }
    println!();

    let tol = 1.0 + (k as f32 / 512.0).ceil();
    assert!(max_bc < tol, "fused vs separate: {max_bc:.2e}");
    println!("PASS");
}

/// Test 4: simulate 2 layers of the ferrite control flow vs standard.
/// Skip attention — just test the residual/norm/GEMM threading.
/// Layer pattern (simplified, no attention):
///   Standard: fused_add_rms_norm(hs, res) → GEMM → fused_add_rms_norm(gemm_out, res) → GEMM
///   Ferrite:  add(res, hs) → fused_norm_gemm(res) → add(res, gemm_out) → fused_norm_gemm(res)
#[test]
fn test4_two_layer_control_flow() {
    println!("=== Test 4: 2-layer control flow ferrite vs standard ===");

    let data = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&data).expect("parse safetensors");

    let hidden = 896u32;
    let m = 64u32;
    let k = hidden;
    // Use a square GEMM for simplicity (hidden → hidden)
    let n = hidden;
    let eps = 1e-6f32;

    let h_weight_l0 = load_bf16_tensor(&st, "model.layers.0.self_attn.q_proj.weight");
    let h_norm_l0 = load_bf16_tensor(&st, "model.layers.0.input_layernorm.weight");
    let h_weight_l1 = load_bf16_tensor(&st, "model.layers.1.self_attn.q_proj.weight");
    let h_norm_l1 = load_bf16_tensor(&st, "model.layers.1.input_layernorm.weight");

    // Initial tensors: hidden_states (from embedding) and residual=None initially
    let h_hs0: Vec<bf16> = (0..(m * k) as usize)
        .map(|i| bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.1))
        .collect();

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let d_weight_l0 = stream.clone_htod(&h_weight_l0).unwrap();
    let d_norm_l0 = stream.clone_htod(&h_norm_l0).unwrap();
    let d_weight_l1 = stream.clone_htod(&h_weight_l1).unwrap();
    let d_norm_l1 = stream.clone_htod(&h_norm_l1).unwrap();

    let (wl0, _) = d_weight_l0.device_ptr(&stream);
    let (nwl0, _) = d_norm_l0.device_ptr(&stream);
    let (wl1, _) = d_weight_l1.device_ptr(&stream);
    let (nwl1, _) = d_norm_l1.device_ptr(&stream);

    // Load rms_norm kernel
    let rms_module = ctx.load_module(Ptx::from_src(RMS_NORM_BF16_PTX)).unwrap();
    let rms_func = rms_module
        .load_function("_Z15rms_norm_kernelI13__nv_bfloat16EvPT_PKS1_S4_fi")
        .unwrap();

    let blas = CudaBlas::new(stream.clone()).unwrap();

    // ── Standard path: 2 layers ──
    // Layer 0 (no residual): normed = rms_norm(hs), out = GEMM(normed, W), residual = hs
    let d_std_hs = stream.clone_htod(&h_hs0).unwrap();
    let d_std_normed: CudaSlice<bf16> = stream.alloc_zeros((m * k) as usize).unwrap();
    let (std_hs_p, _) = d_std_hs.device_ptr(&stream);
    let (std_normed_p, _) = d_std_normed.device_ptr(&stream);

    // rms_norm(hs) → normed
    unsafe {
        stream
            .launch_builder(&rms_func)
            .arg(&std_normed_p)
            .arg(&std_hs_p)
            .arg(&nwl0)
            .arg(&eps)
            .arg(&(k as i32))
            .launch(LaunchConfig {
                grid_dim: (m, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .unwrap();

    // GEMM: out0 = normed @ W_l0^T
    let mut d_std_out0: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
    {
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
        unsafe {
            blas.gemm(cfg, &d_weight_l0, &d_std_normed, &mut d_std_out0)
                .unwrap()
        };
    }

    // Standard layer 0 returns: (out0, residual=hs)
    // residual = d_std_hs (unmodified)

    // Layer 1 (has residual): fused_add_rms_norm(out0, residual)
    // → residual += out0, out0 = norm(residual)
    let (std_out0_p, _) = d_std_out0.device_ptr(&stream);
    unsafe {
        // fused_add_rms_norm_inplace(hidden_states=out0, residual=hs)
        let rms_norm_bf16 = |input: u64,
                             residual: u64,
                             weight: u64,
                             eps: f32,
                             hidden: i32,
                             stream: &Arc<cudarc::driver::CudaStream>| {
            // We need the fused_add_rms_norm kernel. Let's use separate ops instead.
            // add_inplace(residual, input): can't easily call vllm kernel from test.
            // Instead: manually compute on CPU or use a workaround.
        };
    }

    // Actually, I can't easily call fused_add_rms_norm_inplace from the test.
    // Let me simulate it with: residual += out0, normed = rms_norm(residual)
    // This is the ferrite path's approach. If both paths do the same thing, comparison is valid.

    // Layer 1: residual += out0
    // Need add_inplace kernel... or just do it on CPU
    stream.synchronize().unwrap();
    let std_residual = stream.clone_dtoh(&d_std_hs).unwrap(); // residual = original hs
    let std_out0 = stream.clone_dtoh(&d_std_out0).unwrap();

    // CPU: residual += out0
    let std_residual_l1: Vec<bf16> = std_residual
        .iter()
        .zip(std_out0.iter())
        .map(|(r, o)| bf16::from_f32(r.to_f32() + o.to_f32()))
        .collect();

    // Upload and norm
    let d_std_res_l1 = stream.clone_htod(&std_residual_l1).unwrap();
    let d_std_normed_l1: CudaSlice<bf16> = stream.alloc_zeros((m * k) as usize).unwrap();
    let (std_res_l1_p, _) = d_std_res_l1.device_ptr(&stream);
    let (std_normed_l1_p, _) = d_std_normed_l1.device_ptr(&stream);

    unsafe {
        stream
            .launch_builder(&rms_func)
            .arg(&std_normed_l1_p)
            .arg(&std_res_l1_p)
            .arg(&nwl1)
            .arg(&eps)
            .arg(&(k as i32))
            .launch(LaunchConfig {
                grid_dim: (m, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .unwrap();

    // GEMM: out1 = normed_l1 @ W_l1^T
    let mut d_std_out1: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
    {
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
        unsafe {
            blas.gemm(cfg, &d_weight_l1, &d_std_normed_l1, &mut d_std_out1)
                .unwrap()
        };
    }
    stream.synchronize().unwrap();

    // ── Ferrite path: 2 layers ──
    // Layer 0: same as standard (no residual)
    let d_fer_hs = stream.clone_htod(&h_hs0).unwrap();
    let (fer_hs_p, _) = d_fer_hs.device_ptr(&stream);

    // fused_norm_gemm on hs (same as norm + GEMM since no residual add)
    let d_fer_out0: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
    let (fer_out0_p, _) = d_fer_out0.device_ptr(&stream);

    let fused_module = ctx.load_module(Ptx::from_src(FUSED_NORM_GEMM.ptx)).unwrap();
    let fused_func = fused_module.load_function(FUSED_NORM_GEMM.entry).unwrap();

    let flat0 = build_flat_params(
        fer_hs_p as u64,
        wl0 as u64,
        fer_out0_p as u64,
        fer_out0_p as u64,
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
        let (gx, gy, gz) = compute_grid(m, n, 64, 128);
        stream
            .launch_builder(&fused_func)
            .arg(&(fer_hs_p as u64))
            .arg(&(nwl0 as u64))
            .arg(&eps)
            .arg(&k)
            .arg(&(k as u64))
            .arg(&flat0)
            .launch(LaunchConfig {
                grid_dim: (gx, gy, gz),
                block_dim: (128, 1, 1),
                shared_mem_bytes: FUSED_NORM_GEMM.smem_bytes,
            })
    }
    .unwrap();

    // Ferrite layer 0 returns: (out0, residual=hs)

    // Layer 1: add_inplace(residual=hs, hidden_states=out0)
    // Then fused_norm_gemm(residual)
    // Simulate add on CPU (same as standard path above)
    stream.synchronize().unwrap();
    let fer_out0_h = stream.clone_dtoh(&d_fer_out0).unwrap();
    let fer_residual_l1: Vec<bf16> = h_hs0
        .iter()
        .zip(fer_out0_h.iter())
        .map(|(r, o)| bf16::from_f32(r.to_f32() + o.to_f32()))
        .collect();

    let d_fer_res_l1 = stream.clone_htod(&fer_residual_l1).unwrap();
    let (fer_res_l1_p, _) = d_fer_res_l1.device_ptr(&stream);

    let d_fer_out1: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
    let (fer_out1_p, _) = d_fer_out1.device_ptr(&stream);

    let flat1 = build_flat_params(
        fer_res_l1_p as u64,
        wl1 as u64,
        fer_out1_p as u64,
        fer_out1_p as u64,
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
        let (gx, gy, gz) = compute_grid(m, n, 64, 128);
        stream
            .launch_builder(&fused_func)
            .arg(&(fer_res_l1_p as u64))
            .arg(&(nwl1 as u64))
            .arg(&eps)
            .arg(&k)
            .arg(&(k as u64))
            .arg(&flat1)
            .launch(LaunchConfig {
                grid_dim: (gx, gy, gz),
                block_dim: (128, 1, 1),
                shared_mem_bytes: FUSED_NORM_GEMM.smem_bytes,
            })
    }
    .unwrap();
    stream.synchronize().unwrap();

    // Compare layer 0 outputs
    let std_out0_h = stream.clone_dtoh(&d_std_out0).unwrap();
    let fer_out0_h = stream.clone_dtoh(&d_fer_out0).unwrap();
    let mut max_l0 = 0.0f32;
    for (s, f) in std_out0_h.iter().zip(fer_out0_h.iter()) {
        max_l0 = max_l0.max((s.to_f32() - f.to_f32()).abs());
    }

    // Compare layer 1 outputs
    let std_out1_h = stream.clone_dtoh(&d_std_out1).unwrap();
    let fer_out1_h = stream.clone_dtoh(&d_fer_out1).unwrap();
    let mut max_l1 = 0.0f32;
    for (s, f) in std_out1_h.iter().zip(fer_out1_h.iter()) {
        max_l1 = max_l1.max((s.to_f32() - f.to_f32()).abs());
    }

    println!("  Layer 0 diff: {max_l0:.2e}");
    println!("  Layer 1 diff: {max_l1:.2e}");
    print!("  std L1[0..8]: ");
    for i in 0..8 {
        print!("{:.4} ", std_out1_h[i].to_f32());
    }
    println!();
    print!("  fer L1[0..8]: ");
    for i in 0..8 {
        print!("{:.4} ", fer_out1_h[i].to_f32());
    }
    println!();

    let tol = 1.0 + (k as f32 / 512.0).ceil();
    assert!(max_l0 < tol, "layer 0 diff too large: {max_l0:.2e}");
    assert!(max_l1 < tol, "layer 1 diff too large: {max_l1:.2e}");
    println!("PASS: 2-layer control flow matches");
}

/// Test 6: chain 24 layers, each doing add+fused_norm_gemm.
/// Compare fused vs separate at each layer. Use real weights per layer.
/// The residual accumulates across layers (like production).
#[test]
fn test6_24_layer_chain() {
    println!("=== Test 6: 24-layer chain, fused vs separate ===");

    let data = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&data).expect("parse safetensors");

    let hidden = 896u32;
    let m = 64u32; // decode batch
    let k = hidden;
    let n = hidden; // use q_proj [896,896] for simplicity (square GEMM)
    let eps = 1e-6f32;
    let num_layers = 24;

    // Load all layer weights
    let mut weights = Vec::new();
    let mut norm_weights = Vec::new();
    for layer in 0..num_layers {
        weights.push(load_bf16_tensor(
            &st,
            &format!("model.layers.{layer}.self_attn.q_proj.weight"),
        ));
        norm_weights.push(load_bf16_tensor(
            &st,
            &format!("model.layers.{layer}.input_layernorm.weight"),
        ));
    }

    // Initial hidden_states (like embedding output)
    let h_hs_init: Vec<bf16> = (0..(m * k) as usize)
        .map(|i| bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.1))
        .collect();

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // Upload all weights
    let d_weights: Vec<CudaSlice<bf16>> = weights
        .iter()
        .map(|w| stream.clone_htod(w).unwrap())
        .collect();
    let d_norms: Vec<CudaSlice<bf16>> = norm_weights
        .iter()
        .map(|w| stream.clone_htod(w).unwrap())
        .collect();

    let rms_module = ctx.load_module(Ptx::from_src(RMS_NORM_BF16_PTX)).unwrap();
    let rms_func = rms_module
        .load_function("_Z15rms_norm_kernelI13__nv_bfloat16EvPT_PKS1_S4_fi")
        .unwrap();
    let gemm_module = ctx.load_module(Ptx::from_src(FLAT_GEMM_PTX)).unwrap();
    let gemm_func = gemm_module.load_function("ferrite_gemm_64x128x32").unwrap();
    let fused_module = ctx.load_module(Ptx::from_src(FUSED_NORM_GEMM.ptx)).unwrap();
    let fused_func = fused_module.load_function(FUSED_NORM_GEMM.entry).unwrap();

    // State for both paths — start identical
    let mut fused_residual: Vec<bf16> = vec![bf16::ZERO; (m * k) as usize]; // no residual initially
    let mut fused_hs: Vec<bf16> = h_hs_init.clone();
    let mut sep_residual: Vec<bf16> = vec![bf16::ZERO; (m * k) as usize];
    let mut sep_hs: Vec<bf16> = h_hs_init.clone();

    let (gx, gy, gz) = compute_grid(m, n, 64, 128);

    for layer in 0..num_layers {
        let (w_p, _) = d_weights[layer].device_ptr(&stream);
        let (nw_p, _) = d_norms[layer].device_ptr(&stream);

        // CPU: residual += hidden_states (both paths)
        for i in 0..(m * k) as usize {
            fused_residual[i] = bf16::from_f32(fused_residual[i].to_f32() + fused_hs[i].to_f32());
            sep_residual[i] = bf16::from_f32(sep_residual[i].to_f32() + sep_hs[i].to_f32());
        }

        // ── Fused path: fused_norm_gemm(residual) ──
        let d_fused_res = stream.clone_htod(&fused_residual).unwrap();
        let d_fused_out: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
        let (fr_p, _) = d_fused_res.device_ptr(&stream);
        let (fo_p, _) = d_fused_out.device_ptr(&stream);

        let flat_f = build_flat_params(
            fr_p as u64,
            w_p as u64,
            fo_p as u64,
            fo_p as u64,
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
            stream
                .launch_builder(&fused_func)
                .arg(&(fr_p as u64))
                .arg(&(nw_p as u64))
                .arg(&eps)
                .arg(&k)
                .arg(&(k as u64))
                .arg(&flat_f)
                .launch(LaunchConfig {
                    grid_dim: (gx, gy, gz),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: FUSED_NORM_GEMM.smem_bytes,
                })
        }
        .unwrap();

        // ── Separate path: rms_norm(residual) then GEMM ──
        let d_sep_res = stream.clone_htod(&sep_residual).unwrap();
        let d_sep_normed: CudaSlice<bf16> = stream.alloc_zeros((m * k) as usize).unwrap();
        let d_sep_out: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
        let (sr_p, _) = d_sep_res.device_ptr(&stream);
        let (sn_p, _) = d_sep_normed.device_ptr(&stream);
        let (so_p, _) = d_sep_out.device_ptr(&stream);

        unsafe {
            stream
                .launch_builder(&rms_func)
                .arg(&sn_p)
                .arg(&sr_p)
                .arg(&nw_p)
                .arg(&eps)
                .arg(&(k as i32))
                .launch(LaunchConfig {
                    grid_dim: (m, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .unwrap();

        let flat_s = build_flat_params(
            sn_p as u64,
            w_p as u64,
            so_p as u64,
            so_p as u64,
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
            stream
                .launch_builder(&gemm_func)
                .arg(&flat_s)
                .launch(LaunchConfig {
                    grid_dim: (gx, gy, gz),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 36864,
                })
        }
        .unwrap();

        stream.synchronize().unwrap();

        // Read back outputs — these become next layer's hidden_states
        fused_hs = stream.clone_dtoh(&d_fused_out).unwrap();
        sep_hs = stream.clone_dtoh(&d_sep_out).unwrap();

        // Compare
        let mut max_diff = 0.0f32;
        for i in 0..(m * n) as usize {
            let d = (fused_hs[i].to_f32() - sep_hs[i].to_f32()).abs();
            if d > max_diff {
                max_diff = d;
            }
        }

        // Check residual divergence too
        let mut max_res_diff = 0.0f32;
        for i in 0..(m * k) as usize {
            let d = (fused_residual[i].to_f32() - sep_residual[i].to_f32()).abs();
            if d > max_res_diff {
                max_res_diff = d;
            }
        }

        println!(
            "  layer {layer:2}: output_diff={max_diff:.2e}  residual_diff={max_res_diff:.2e}  hs[0]={:.4}",
            fused_hs[0].to_f32()
        );

        let tol = 2.0 + (k as f32 / 256.0).ceil();
        if max_diff > tol {
            println!("  FAIL at layer {layer}!");
            print!("  fused [0..8]: ");
            for i in 0..8 {
                print!("{:.4} ", fused_hs[i].to_f32());
            }
            println!();
            print!("  separ [0..8]: ");
            for i in 0..8 {
                print!("{:.4} ", sep_hs[i].to_f32());
            }
            println!();
            panic!("layer {layer} diverged: {max_diff:.2e}");
        }
    }
    println!("PASS: 24-layer chain matches");
}

/// Test 7: same as test 6 but do the add on GPU and REUSE buffers
/// across layers (simulating caching allocator). The key difference:
/// instead of clone_htod each layer, keep persistent GPU buffers.
#[test]
fn test7_24_layer_gpu_reuse() {
    println!("=== Test 7: 24-layer chain, GPU add, buffer reuse ===");

    let data = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&data).expect("parse safetensors");

    let hidden = 896u32;
    let m = 64u32;
    let k = hidden;
    let n = hidden;
    let eps = 1e-6f32;
    let num_layers = 24;

    let mut weights = Vec::new();
    let mut norm_weights = Vec::new();
    for layer in 0..num_layers {
        weights.push(load_bf16_tensor(
            &st,
            &format!("model.layers.{layer}.self_attn.q_proj.weight"),
        ));
        norm_weights.push(load_bf16_tensor(
            &st,
            &format!("model.layers.{layer}.input_layernorm.weight"),
        ));
    }

    let h_hs_init: Vec<bf16> = (0..(m * k) as usize)
        .map(|i| bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.1))
        .collect();

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let d_weights: Vec<_> = weights
        .iter()
        .map(|w| stream.clone_htod(w).unwrap())
        .collect();
    let d_norms: Vec<_> = norm_weights
        .iter()
        .map(|w| stream.clone_htod(w).unwrap())
        .collect();

    let rms_module = ctx.load_module(Ptx::from_src(RMS_NORM_BF16_PTX)).unwrap();
    let rms_func = rms_module
        .load_function("_Z15rms_norm_kernelI13__nv_bfloat16EvPT_PKS1_S4_fi")
        .unwrap();
    let gemm_module = ctx.load_module(Ptx::from_src(FLAT_GEMM_PTX)).unwrap();
    let gemm_func = gemm_module.load_function("ferrite_gemm_64x128x32").unwrap();
    let fused_module = ctx.load_module(Ptx::from_src(FUSED_NORM_GEMM.ptx)).unwrap();
    let fused_func = fused_module.load_function(FUSED_NORM_GEMM.entry).unwrap();

    let (gx, gy, gz) = compute_grid(m, n, 64, 128);

    // Persistent GPU buffers — reused each layer (like caching allocator)
    // Fused path
    let mut d_fused_res = stream
        .clone_htod(&vec![bf16::ZERO; (m * k) as usize])
        .unwrap();
    let mut d_fused_hs = stream.clone_htod(&h_hs_init).unwrap();
    let d_fused_out: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
    // Separate path
    let mut d_sep_res = stream
        .clone_htod(&vec![bf16::ZERO; (m * k) as usize])
        .unwrap();
    let mut d_sep_hs = stream.clone_htod(&h_hs_init).unwrap();
    let d_sep_normed: CudaSlice<bf16> = stream.alloc_zeros((m * k) as usize).unwrap();
    let d_sep_out: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();

    for layer in 0..num_layers {
        let (w_p, _) = d_weights[layer].device_ptr(&stream);
        let (nw_p, _) = d_norms[layer].device_ptr(&stream);

        // GPU add via round-trip (correct, just slow)
        {
            stream.synchronize().unwrap();
            let mut res_h = stream.clone_dtoh(&d_fused_res).unwrap();
            let hs_h = stream.clone_dtoh(&d_fused_hs).unwrap();
            for i in 0..res_h.len() {
                res_h[i] = bf16::from_f32(res_h[i].to_f32() + hs_h[i].to_f32());
            }
            stream.memcpy_htod(&res_h, &mut d_fused_res).unwrap();

            let mut res_h = stream.clone_dtoh(&d_sep_res).unwrap();
            let hs_h = stream.clone_dtoh(&d_sep_hs).unwrap();
            for i in 0..res_h.len() {
                res_h[i] = bf16::from_f32(res_h[i].to_f32() + hs_h[i].to_f32());
            }
            stream.memcpy_htod(&res_h, &mut d_sep_res).unwrap();
        }

        let (fr_p, _) = d_fused_res.device_ptr(&stream);
        let (fo_p, _) = d_fused_out.device_ptr(&stream);
        let (sr_p, _) = d_sep_res.device_ptr(&stream);
        let (sn_p, _) = d_sep_normed.device_ptr(&stream);
        let (so_p, _) = d_sep_out.device_ptr(&stream);

        // ── Fused path ──
        let flat_f = build_flat_params(
            fr_p as u64,
            w_p as u64,
            fo_p as u64,
            fo_p as u64,
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
            stream
                .launch_builder(&fused_func)
                .arg(&(fr_p as u64))
                .arg(&(nw_p as u64))
                .arg(&eps)
                .arg(&k)
                .arg(&(k as u64))
                .arg(&flat_f)
                .launch(LaunchConfig {
                    grid_dim: (gx, gy, gz),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: FUSED_NORM_GEMM.smem_bytes,
                })
        }
        .unwrap();

        // ── Separate path ──
        unsafe {
            stream
                .launch_builder(&rms_func)
                .arg(&sn_p)
                .arg(&sr_p)
                .arg(&nw_p)
                .arg(&eps)
                .arg(&(k as i32))
                .launch(LaunchConfig {
                    grid_dim: (m, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .unwrap();

        let flat_s = build_flat_params(
            sn_p as u64,
            w_p as u64,
            so_p as u64,
            so_p as u64,
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
            stream
                .launch_builder(&gemm_func)
                .arg(&flat_s)
                .launch(LaunchConfig {
                    grid_dim: (gx, gy, gz),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 36864,
                })
        }
        .unwrap();

        stream.synchronize().unwrap();

        // Output becomes next layer's hidden_states
        // Copy output → hs buffer (reuse same buffers)
        let fused_out_h = stream.clone_dtoh(&d_fused_out).unwrap();
        let sep_out_h = stream.clone_dtoh(&d_sep_out).unwrap();

        // Compare
        let mut max_diff = 0.0f32;
        for i in 0..(m * n) as usize {
            let d = (fused_out_h[i].to_f32() - sep_out_h[i].to_f32()).abs();
            if d > max_diff {
                max_diff = d;
            }
        }

        println!(
            "  layer {layer:2}: diff={max_diff:.2e}  hs[0]={:.4}",
            fused_out_h[0].to_f32()
        );

        let tol = 2.0 + (k as f32 / 256.0).ceil();
        if max_diff > tol {
            print!("  fused [0..8]: ");
            for i in 0..8 {
                print!("{:.4} ", fused_out_h[i].to_f32());
            }
            println!();
            print!("  separ [0..8]: ");
            for i in 0..8 {
                print!("{:.4} ", sep_out_h[i].to_f32());
            }
            println!();
            panic!("layer {layer} diverged: {max_diff:.2e}");
        }

        // Upload output as next layer's hidden_states
        stream.memcpy_htod(&fused_out_h, &mut d_fused_hs).unwrap();
        stream.memcpy_htod(&sep_out_h, &mut d_sep_hs).unwrap();
    }
    println!("PASS: 24-layer chain (GPU add, buffer reuse) matches");
}

/// Test 8: full layer simulation — two fused_norm_gemm per layer with
/// a GEMM+accumulate between them (simulating QKV→attn→o_proj→MLP).
/// Fused path: add+fused_norm_gemm(QKV) → GEMM_accum(o_proj) → fused_norm_gemm(MLP)
/// Separate path: fused_add_rms_norm+GEMM → GEMM_accum → fused_add_rms_norm+GEMM
#[test]
fn test8_full_layer_sim() {
    println!("=== Test 8: full layer simulation (2 fused per layer + o_proj accum) ===");

    let data = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&data).expect("parse safetensors");

    let hidden = 896u32;
    let m = 64u32;
    let k = hidden;
    let eps = 1e-6f32;
    let num_layers = 24;

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // Load kernels
    let rms_module = ctx.load_module(Ptx::from_src(RMS_NORM_BF16_PTX)).unwrap();
    let rms_func = rms_module
        .load_function("_Z15rms_norm_kernelI13__nv_bfloat16EvPT_PKS1_S4_fi")
        .unwrap();
    let gemm_module = ctx.load_module(Ptx::from_src(FLAT_GEMM_PTX)).unwrap();
    let gemm_func = gemm_module.load_function("ferrite_gemm_64x128x32").unwrap();
    let fused_module = ctx.load_module(Ptx::from_src(FUSED_NORM_GEMM.ptx)).unwrap();
    let fused_func = fused_module.load_function(FUSED_NORM_GEMM.entry).unwrap();

    // Use q_proj as QKV weight, o_proj for accumulate, gate_up for MLP
    // (all [896,896] for 0.5B q_proj and o_proj)
    let mut qkv_weights = Vec::new();
    let mut o_weights = Vec::new();
    let mut gate_weights = Vec::new();
    let mut norm1_weights = Vec::new();
    let mut norm2_weights = Vec::new();
    for layer in 0..num_layers {
        qkv_weights.push(
            stream
                .clone_htod(&load_bf16_tensor(
                    &st,
                    &format!("model.layers.{layer}.self_attn.q_proj.weight"),
                ))
                .unwrap(),
        );
        o_weights.push(
            stream
                .clone_htod(&load_bf16_tensor(
                    &st,
                    &format!("model.layers.{layer}.self_attn.o_proj.weight"),
                ))
                .unwrap(),
        );
        gate_weights.push(
            stream
                .clone_htod(&load_bf16_tensor(
                    &st,
                    &format!("model.layers.{layer}.self_attn.k_proj.weight"),
                ))
                .unwrap(),
        ); // just need [hidden,hidden]-ish
        norm1_weights.push(
            stream
                .clone_htod(&load_bf16_tensor(
                    &st,
                    &format!("model.layers.{layer}.input_layernorm.weight"),
                ))
                .unwrap(),
        );
        norm2_weights.push(
            stream
                .clone_htod(&load_bf16_tensor(
                    &st,
                    &format!("model.layers.{layer}.post_attention_layernorm.weight"),
                ))
                .unwrap(),
        );
    }

    let h_init: Vec<bf16> = (0..(m * k) as usize)
        .map(|i| bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.1))
        .collect();

    let n = hidden; // square GEMMs
    let (gx, gy, gz) = compute_grid(m, n, 64, 128);

    // Persistent buffers
    let mut d_fused_res = stream
        .clone_htod(&vec![bf16::ZERO; (m * k) as usize])
        .unwrap();
    let mut d_fused_hs = stream.clone_htod(&h_init).unwrap();
    let mut d_sep_res = stream
        .clone_htod(&vec![bf16::ZERO; (m * k) as usize])
        .unwrap();
    let mut d_sep_hs = stream.clone_htod(&h_init).unwrap();

    for layer in 0..num_layers {
        let (qw, _) = qkv_weights[layer].device_ptr(&stream);
        let (ow, _) = o_weights[layer].device_ptr(&stream);
        let (gw, _) = gate_weights[layer].device_ptr(&stream);
        let (nw1, _) = norm1_weights[layer].device_ptr(&stream);
        let (nw2, _) = norm2_weights[layer].device_ptr(&stream);

        // === Step 1: residual += hidden_states (CPU round-trip) ===
        stream.synchronize().unwrap();
        let mut fr = stream.clone_dtoh(&d_fused_res).unwrap();
        let fh = stream.clone_dtoh(&d_fused_hs).unwrap();
        for i in 0..fr.len() {
            fr[i] = bf16::from_f32(fr[i].to_f32() + fh[i].to_f32());
        }
        stream.memcpy_htod(&fr, &mut d_fused_res).unwrap();

        let mut sr = stream.clone_dtoh(&d_sep_res).unwrap();
        let sh = stream.clone_dtoh(&d_sep_hs).unwrap();
        for i in 0..sr.len() {
            sr[i] = bf16::from_f32(sr[i].to_f32() + sh[i].to_f32());
        }
        stream.memcpy_htod(&sr, &mut d_sep_res).unwrap();

        let (fr_p, _) = d_fused_res.device_ptr(&stream);
        let (sr_p, _) = d_sep_res.device_ptr(&stream);

        // === Step 2: QKV norm+GEMM ===
        // Fused
        let d_fqkv: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
        let (fqkv_p, _) = d_fqkv.device_ptr(&stream);
        let flat = build_flat_params(
            fr_p as u64,
            qw as u64,
            fqkv_p as u64,
            fqkv_p as u64,
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
            stream
                .launch_builder(&fused_func)
                .arg(&(fr_p as u64))
                .arg(&(nw1 as u64))
                .arg(&eps)
                .arg(&k)
                .arg(&(k as u64))
                .arg(&flat)
                .launch(LaunchConfig {
                    grid_dim: (gx, gy, gz),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: FUSED_NORM_GEMM.smem_bytes,
                })
        }
        .unwrap();

        // Separate: norm then GEMM
        let d_snormed: CudaSlice<bf16> = stream.alloc_zeros((m * k) as usize).unwrap();
        let d_sqkv: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
        let (sn_p, _) = d_snormed.device_ptr(&stream);
        let (sqkv_p, _) = d_sqkv.device_ptr(&stream);
        unsafe {
            stream
                .launch_builder(&rms_func)
                .arg(&sn_p)
                .arg(&sr_p)
                .arg(&(nw1 as u64))
                .arg(&eps)
                .arg(&(k as i32))
                .launch(LaunchConfig {
                    grid_dim: (m, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .unwrap();
        let flat = build_flat_params(
            sn_p as u64,
            qw as u64,
            sqkv_p as u64,
            sqkv_p as u64,
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
            stream
                .launch_builder(&gemm_func)
                .arg(&flat)
                .launch(LaunchConfig {
                    grid_dim: (gx, gy, gz),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 36864,
                })
        }
        .unwrap();

        // === Step 3: o_proj accumulate — residual += qkv @ o_weight (simulate attn=identity) ===
        // Fused path: residual += fqkv @ o_weight
        let flat = build_flat_params(
            fqkv_p as u64,
            ow as u64,
            fr_p as u64,
            fr_p as u64,
            m,
            n,
            k,
            k,
            k,
            n,
            n,
            1.0,
            1.0,
        );
        unsafe {
            stream
                .launch_builder(&gemm_func)
                .arg(&flat)
                .launch(LaunchConfig {
                    grid_dim: (gx, gy, gz),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 36864,
                })
        }
        .unwrap();

        // Separate path
        let flat = build_flat_params(
            sqkv_p as u64,
            ow as u64,
            sr_p as u64,
            sr_p as u64,
            m,
            n,
            k,
            k,
            k,
            n,
            n,
            1.0,
            1.0,
        );
        unsafe {
            stream
                .launch_builder(&gemm_func)
                .arg(&flat)
                .launch(LaunchConfig {
                    grid_dim: (gx, gy, gz),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 36864,
                })
        }
        .unwrap();

        // === Step 4: MLP norm+GEMM (second fused call, same residual) ===
        let (fr_p, _) = d_fused_res.device_ptr(&stream);
        let (sr_p, _) = d_sep_res.device_ptr(&stream);

        // Fused
        let d_fmlp: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
        let (fmlp_p, _) = d_fmlp.device_ptr(&stream);
        let flat = build_flat_params(
            fr_p as u64,
            gw as u64,
            fmlp_p as u64,
            fmlp_p as u64,
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
            stream
                .launch_builder(&fused_func)
                .arg(&(fr_p as u64))
                .arg(&(nw2 as u64))
                .arg(&eps)
                .arg(&k)
                .arg(&(k as u64))
                .arg(&flat)
                .launch(LaunchConfig {
                    grid_dim: (gx, gy, gz),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: FUSED_NORM_GEMM.smem_bytes,
                })
        }
        .unwrap();

        // Separate
        let d_snormed2: CudaSlice<bf16> = stream.alloc_zeros((m * k) as usize).unwrap();
        let d_smlp: CudaSlice<bf16> = stream.alloc_zeros((m * n) as usize).unwrap();
        let (sn2_p, _) = d_snormed2.device_ptr(&stream);
        let (smlp_p, _) = d_smlp.device_ptr(&stream);
        unsafe {
            stream
                .launch_builder(&rms_func)
                .arg(&sn2_p)
                .arg(&sr_p)
                .arg(&(nw2 as u64))
                .arg(&eps)
                .arg(&(k as i32))
                .launch(LaunchConfig {
                    grid_dim: (m, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .unwrap();
        let flat = build_flat_params(
            sn2_p as u64,
            gw as u64,
            smlp_p as u64,
            smlp_p as u64,
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
            stream
                .launch_builder(&gemm_func)
                .arg(&flat)
                .launch(LaunchConfig {
                    grid_dim: (gx, gy, gz),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 36864,
                })
        }
        .unwrap();

        stream.synchronize().unwrap();

        // MLP output becomes next layer's hidden_states
        let fmlp_h = stream.clone_dtoh(&d_fmlp).unwrap();
        let smlp_h = stream.clone_dtoh(&d_smlp).unwrap();

        let mut max_diff = 0.0f32;
        let mut worst_idx = 0;
        for i in 0..(m * n) as usize {
            let d = (fmlp_h[i].to_f32() - smlp_h[i].to_f32()).abs();
            if d > max_diff {
                max_diff = d;
                worst_idx = i;
            }
        }

        let row = worst_idx / n as usize;
        let col = worst_idx % n as usize;
        // Check value ranges
        let fmax = fmlp_h
            .iter()
            .map(|v| v.to_f32().abs())
            .fold(0.0f32, f32::max);
        let smax = smlp_h
            .iter()
            .map(|v| v.to_f32().abs())
            .fold(0.0f32, f32::max);
        println!(
            "  layer {layer:2}: diff={max_diff:.2e} at [{worst_idx}] (row={row},col={col})  fmax={fmax:.1} smax={smax:.1}"
        );

        if max_diff > 5.0 {
            print!("  fused [0..8]: ");
            for i in 0..8 {
                print!("{:.4} ", fmlp_h[i].to_f32());
            }
            println!();
            print!("  separ [0..8]: ");
            for i in 0..8 {
                print!("{:.4} ", smlp_h[i].to_f32());
            }
            println!();
            println!(
                "  at worst: fused={:.4} separ={:.4}",
                fmlp_h[worst_idx].to_f32(),
                smlp_h[worst_idx].to_f32()
            );
            // Check how many elements have large diffs
            let big = (0..(m * n) as usize)
                .filter(|&i| (fmlp_h[i].to_f32() - smlp_h[i].to_f32()).abs() > 1.0)
                .count();
            println!("  elements with diff > 1.0: {big} / {}", m * n);
            panic!("layer {layer} diverged: {max_diff:.2e}");
        }

        stream.memcpy_htod(&fmlp_h, &mut d_fused_hs).unwrap();
        stream.memcpy_htod(&smlp_h, &mut d_sep_hs).unwrap();
    }
    println!("PASS: full layer simulation matches");
}
