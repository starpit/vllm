// SPDX-License-Identifier: Apache-2.0
//! Ferrite CUTLASS GEMM integration using flat-param perimeter-replaced kernels.
//!
//! Each CUTLASS tile configuration is rewritten at compile time by
//! `replace_perimeter_macro!` to accept a flat 88-byte param struct
//! (pointers, strides, M/N/K, alpha, beta) instead of the opaque
//! 368-byte CUTLASS Params struct. No CUTLASS host code needed.
//!
//! Enable with: `--features ferrite`

use anyhow::{Result, bail};
use cudarc::driver::sys::{self, CUfunction, CUmodule, CUstream};
use std::collections::HashMap;
use std::sync::Mutex;

use crate::alloc::{CachingAllocator, OwnedTensor};
use crate::tensor::GpuTensor;
use ptx_fusion::FeriteKernel;

// ── Compile-time: rewrite CUTLASS PTX to flat-param layout ──

ptx_fusion::replace_perimeter_macro!(
    "../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    "ferrite_gemm_64x128x32",
    FLAT_64X128X32_PTX
);

ptx_fusion::replace_perimeter_macro!(
    "../ptx-fusion/kernels/cutlass_bf16_128x128x32_sm89.ptx",
    "../ptx-fusion/kernels/cutlass_bf16_128x128x32_sm89.derivations.json",
    "ferrite_gemm_128x128x32",
    FLAT_128X128X32_PTX
);

ptx_fusion::replace_perimeter_macro!(
    "../ptx-fusion/kernels/cutlass_bf16_128x128x64_sm89.ptx",
    "../ptx-fusion/kernels/cutlass_bf16_128x128x64_sm89.derivations.json",
    "ferrite_gemm_128x128x64",
    FLAT_128X128X64_PTX
);

// ── Tile configuration ──

struct TileConfig {
    name: &'static str,
    entry: &'static str,
    tile_m: u32,
    tile_n: u32,
    threads: u32,
    smem_bytes: u32,
    func: CUfunction,
    _module: CUmodule,
}

/// Ferrite CUTLASS GEMM dispatcher.
pub struct FerriteCutlass {
    configs: Vec<TileConfig>,
}

impl FerriteCutlass {
    /// Initialize: load flat-param PTX modules and resolve entry points.
    pub unsafe fn new() -> Result<Self> {
        // Only 64x128x32 for now — larger configs need investigation
        let specs: &[(&str, &str, &str, u32, u32, u32, u32)] = &[(
            "64x128x32",
            "ferrite_gemm_64x128x32",
            FLAT_64X128X32_PTX,
            64,
            128,
            128,
            36864,
        )];

        let mut configs = Vec::new();
        for &(name, entry, ptx, tm, tn, threads, smem) in specs {
            assert!(
                ptx.contains("ferrite_params[88]"),
                "ferrite: PTX for {name} is not flat-param!"
            );
            let (module, func) = load_flat_module(ptx, entry, smem)?;
            configs.push(TileConfig {
                name,
                entry,
                tile_m: tm,
                tile_n: tn,
                threads,
                smem_bytes: smem,
                func,
                _module: module,
            });
        }

        configs.sort_by_key(|c| c.tile_m);
        Ok(FerriteCutlass { configs })
    }

    /// Select the best tile config for the given M dimension.
    fn select(&self, m: u32) -> &TileConfig {
        if m <= self.configs[0].tile_m {
            return &self.configs[0];
        }
        self.configs
            .iter()
            .rev()
            .find(|c| c.tile_m <= m.max(64))
            .unwrap_or(&self.configs[0])
    }

    /// D = alpha * A @ B^T + beta * C
    pub unsafe fn gemm(
        &self,
        a: GpuTensor,
        b: GpuTensor,
        c: Option<GpuTensor>,
        alpha: f32,
        beta: f32,
        alloc: &mut CachingAllocator,
        stream: CUstream,
    ) -> OwnedTensor {
        let m = a.dim(0) as u32;
        let k = a.dim(1) as u32;
        let n = b.dim(0) as u32;
        debug_assert_eq!(b.dim(1) as u32, k, "K dimension mismatch");

        let out = alloc.alloc_tensor(&[m as usize, n as usize], a.dtype());

        let config = self.select(m);

        let grid_m = m.div_ceil(config.tile_m);
        let grid_n = n.div_ceil(config.tile_n);
        let swizzle_log = compute_swizzle_log(grid_n);
        let tile = 1u32 << swizzle_log;
        let grid_x = grid_m * tile;
        let grid_y = grid_n.div_ceil(tile);

        let c_ptr = match c {
            Some(ct) => ct.raw_ptr() as u64,
            None => out.as_gpu_tensor().raw_ptr() as u64,
        };
        let actual_beta = if c.is_some() { beta } else { 0.0 };

        let params = build_flat_params(
            a.raw_ptr() as u64,
            b.raw_ptr() as u64,
            c_ptr,
            out.as_gpu_tensor().raw_ptr() as u64,
            m,
            n,
            k,
            k,
            k,
            n,
            n,
            alpha,
            actual_beta,
        );

        launch_kernel(
            config.func,
            stream,
            grid_x,
            grid_y,
            config.threads,
            config.smem_bytes,
            &params,
        );

        out
    }

    /// D = alpha * A @ B^T + beta * D (accumulate in-place, no allocation).
    pub unsafe fn gemm_accumulate(
        &self,
        a: GpuTensor,
        b: GpuTensor,
        d: GpuTensor,
        alpha: f32,
        beta: f32,
        stream: CUstream,
    ) {
        let m = a.dim(0) as u32;
        let k = a.dim(1) as u32;
        let n = b.dim(0) as u32;
        debug_assert_eq!(b.dim(1) as u32, k, "K dimension mismatch");

        let config = self.select(m);

        let grid_m = m.div_ceil(config.tile_m);
        let grid_n = n.div_ceil(config.tile_n);
        let swizzle_log = compute_swizzle_log(grid_n);
        let tile = 1u32 << swizzle_log;
        let grid_x = grid_m * tile;
        let grid_y = grid_n.div_ceil(tile);

        let params = build_flat_params(
            a.raw_ptr() as u64,
            b.raw_ptr() as u64,
            d.raw_ptr() as u64,
            d.raw_ptr() as u64,
            m,
            n,
            k,
            k,
            k,
            n,
            n,
            alpha,
            beta,
        );

        launch_kernel(
            config.func,
            stream,
            grid_x,
            grid_y,
            config.threads,
            config.smem_bytes,
            &params,
        );
    }
}

// ── Flat param builder (88 bytes) ──

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

// ── Internal helpers ──

unsafe fn load_flat_module(
    ptx: &str,
    entry_name: &str,
    smem_bytes: u32,
) -> Result<(CUmodule, CUfunction)> {
    let mut module: CUmodule = std::ptr::null_mut();
    let ptx_cstr =
        std::ffi::CString::new(ptx).map_err(|e| anyhow::anyhow!("PTX null byte: {e}"))?;

    let result = sys::cuModuleLoadData(&mut module, ptx_cstr.as_ptr() as *const _);
    if result != sys::cudaError_enum::CUDA_SUCCESS {
        bail!("cuModuleLoadData failed: {result:?}");
    }

    let entry_cstr = std::ffi::CString::new(entry_name).unwrap();
    let mut func: CUfunction = std::ptr::null_mut();
    let result = sys::cuModuleGetFunction(&mut func, module, entry_cstr.as_ptr());
    if result != sys::cudaError_enum::CUDA_SUCCESS {
        bail!("cuModuleGetFunction({entry_name}): {result:?}");
    }

    if smem_bytes > 48 * 1024 {
        let result = sys::cuFuncSetAttribute(
            func,
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem_bytes as i32,
        );
        if result != sys::cudaError_enum::CUDA_SUCCESS {
            bail!("cuFuncSetAttribute(SMEM={smem_bytes}): {result:?}");
        }
    }

    Ok((module, func))
}

fn compute_swizzle_log(grid_n: u32) -> u32 {
    const SWIZZLE_N: u32 = 4;
    if SWIZZLE_N >= 8 && grid_n >= 6 {
        3
    } else if SWIZZLE_N >= 4 && grid_n >= 3 {
        2
    } else if SWIZZLE_N >= 2 && grid_n >= 2 {
        1
    } else {
        0
    }
}

unsafe fn launch_kernel(
    func: CUfunction,
    stream: CUstream,
    grid_x: u32,
    grid_y: u32,
    block_x: u32,
    smem_bytes: u32,
    params: &[u8; 88],
) {
    #[repr(align(8))]
    struct AlignedParams([u8; 88]);
    let aligned = AlignedParams(*params);

    let mut param_size = 88usize;
    let extra: [*mut std::ffi::c_void; 5] = [
        sys::CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT as *mut _,
        aligned.0.as_ptr() as *mut _,
        sys::CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT as *mut _,
        &mut param_size as *mut usize as *mut _,
        sys::CU_LAUNCH_PARAM_END_AS_INT as *mut _,
    ];

    let result = sys::cuLaunchKernel(
        func,
        grid_x,
        grid_y,
        1,
        block_x,
        1,
        1,
        smem_bytes,
        stream,
        std::ptr::null_mut(),
        extra.as_ptr() as *mut *mut _,
    );
    assert_eq!(
        result,
        sys::cudaError_enum::CUDA_SUCCESS,
        "ferrite cuLaunchKernel failed: {result:?}"
    );
}

// ── FeriteKernel launch support ──

struct CudaFunc(CUmodule, CUfunction);
unsafe impl Send for CudaFunc {}
unsafe impl Sync for CudaFunc {}

static KERNEL_CACHE: Mutex<Option<HashMap<&'static str, CudaFunc>>> = Mutex::new(None);

fn get_or_load_kernel(kernel: &FeriteKernel) -> CUfunction {
    let mut cache = KERNEL_CACHE.lock().unwrap();
    let cache = cache.get_or_insert_with(HashMap::new);

    if let Some(cf) = cache.get(kernel.entry) {
        return cf.1;
    }

    let (module, func) = unsafe {
        load_flat_module(kernel.ptx, kernel.entry, kernel.smem_bytes)
            .unwrap_or_else(|e| panic!("ferrite: failed to load kernel '{}': {e}", kernel.entry))
    };
    cache.insert(kernel.entry, CudaFunc(module, func));
    func
}

/// Launch a compile!'d fused norm+GEMM kernel.
pub unsafe fn launch_fused_norm_gemm(
    kernel: &FeriteKernel,
    input: GpuTensor,
    weight: GpuTensor,
    norm_weight: GpuTensor,
    epsilon: f32,
    hidden: u32,
    c: Option<GpuTensor>,
    alpha: f32,
    beta: f32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let func = get_or_load_kernel(kernel);

    let m = input.dim(0) as u32;
    let k = input.dim(1) as u32;
    let n = weight.dim(0) as u32;

    let out = alloc.alloc_tensor(&[m as usize, n as usize], input.dtype());

    let grid_m = m.div_ceil(kernel.tile_m);
    let grid_n = n.div_ceil(kernel.tile_n);
    let swizzle_log = compute_swizzle_log(grid_n);
    let tile = 1u32 << swizzle_log;
    let grid_x = grid_m * tile;
    let grid_y = grid_n.div_ceil(tile);

    let c_ptr = match c {
        Some(ct) => ct.raw_ptr() as u64,
        None => out.as_gpu_tensor().raw_ptr() as u64,
    };
    let actual_beta = if c.is_some() { beta } else { 0.0 };

    let flat = build_flat_params(
        input.raw_ptr() as u64,
        weight.raw_ptr() as u64,
        c_ptr,
        out.as_gpu_tensor().raw_ptr() as u64,
        m,
        n,
        k,
        k,
        k,
        n,
        n,
        alpha,
        actual_beta,
    );

    let prefix = kernel.extra_param_bytes as usize;
    let total = prefix + 88;
    let mut params = vec![0u8; total];
    params[0..8].copy_from_slice(&(input.raw_ptr() as u64).to_le_bytes());
    params[8..16].copy_from_slice(&(norm_weight.raw_ptr() as u64).to_le_bytes());
    params[16..20].copy_from_slice(&epsilon.to_le_bytes());
    params[20..24].copy_from_slice(&hidden.to_le_bytes());
    params[24..32].copy_from_slice(&(k as u64).to_le_bytes());
    params[prefix..prefix + 88].copy_from_slice(&flat);

    launch_kernel_raw(
        func,
        stream,
        grid_x,
        grid_y,
        kernel.threads,
        kernel.smem_bytes,
        &params,
    );

    out
}

/// Launch a plain GEMM with compile!'d kernel, accumulating into an existing buffer.
pub unsafe fn launch_gemm_accumulate(
    kernel: &FeriteKernel,
    a: GpuTensor,
    b: GpuTensor,
    d: GpuTensor,
    alpha: f32,
    beta: f32,
    stream: CUstream,
) {
    let func = get_or_load_kernel(kernel);
    let m = a.dim(0) as u32;
    let k = a.dim(1) as u32;
    let n = b.dim(0) as u32;

    let grid_m = m.div_ceil(kernel.tile_m);
    let grid_n = n.div_ceil(kernel.tile_n);
    let swizzle_log = compute_swizzle_log(grid_n);
    let tile = 1u32 << swizzle_log;
    let grid_x = grid_m * tile;
    let grid_y = grid_n.div_ceil(tile);

    let params = build_flat_params(
        a.raw_ptr() as u64,
        b.raw_ptr() as u64,
        d.raw_ptr() as u64,
        d.raw_ptr() as u64,
        m,
        n,
        k,
        k,
        k,
        n,
        n,
        alpha,
        beta,
    );

    launch_kernel_raw(
        func,
        stream,
        grid_x,
        grid_y,
        kernel.threads,
        kernel.smem_bytes,
        &params,
    );
}

/// Generic kernel launch with variable-size param buffer.
unsafe fn launch_kernel_raw(
    func: CUfunction,
    stream: CUstream,
    grid_x: u32,
    grid_y: u32,
    block_x: u32,
    smem_bytes: u32,
    params: &[u8],
) {
    #[repr(align(8))]
    struct Aligned([u8; 512]);
    let mut aligned = Aligned([0u8; 512]);
    let len = params.len().min(512);
    aligned.0[..len].copy_from_slice(&params[..len]);

    let mut param_size = len;
    let extra: [*mut std::ffi::c_void; 5] = [
        sys::CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT as *mut _,
        aligned.0.as_ptr() as *mut _,
        sys::CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT as *mut _,
        &mut param_size as *mut usize as *mut _,
        sys::CU_LAUNCH_PARAM_END_AS_INT as *mut _,
    ];

    let result = sys::cuLaunchKernel(
        func,
        grid_x,
        grid_y,
        1,
        block_x,
        1,
        1,
        smem_bytes,
        stream,
        std::ptr::null_mut(),
        extra.as_ptr() as *mut *mut _,
    );
    assert_eq!(
        result,
        sys::cudaError_enum::CUDA_SUCCESS,
        "ferrite cuLaunchKernel failed: {result:?}"
    );
}

// ── MLP pipeline kernel ──

static MLP_PIPELINE_FUNC: Mutex<Option<CudaFunc>> = Mutex::new(None);

fn get_or_load_mlp_pipeline(ptx: &str) -> CUfunction {
    let mut cache = MLP_PIPELINE_FUNC.lock().unwrap();
    if let Some(ref cf) = *cache {
        return cf.1;
    }
    let (module, func) = unsafe {
        load_flat_module(ptx, "ferrite_mlp_pipeline", 36864)
            .unwrap_or_else(|e| panic!("ferrite: failed to load MLP pipeline: {e}"))
    };
    *cache = Some(CudaFunc(module, func));
    func
}

/// Launch the fused MLP pipeline (single kernel for the entire MLP block).
///
/// Phases: rms_norm -> GEMM_gate_up -> barrier -> SiLU+mul -> GEMM_down
pub unsafe fn launch_mlp_pipeline(
    ptx: &str,
    input: GpuTensor,
    norm_weight: GpuTensor,
    epsilon: f32,
    gate_up_weight: GpuTensor,
    down_weight: GpuTensor,
    num_sm: u32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let func = get_or_load_mlp_pipeline(ptx);

    let m = input.dim(0) as u32;
    let hidden = input.dim(1) as u32;
    let gate_up_n = gate_up_weight.dim(0) as u32;
    let intermediate = gate_up_n / 2;
    let down_n = down_weight.dim(0) as u32;
    let tile_m = 64u32;
    let tile_n = 128u32;

    let gate_up_buf = alloc.alloc_tensor(&[m as usize, gate_up_n as usize], input.dtype());
    let output = alloc.alloc_tensor(&[m as usize, down_n as usize], input.dtype());

    // Zero-init persistent counters + barrier (3 x u32 packed into a u32 tensor)
    let counters = alloc.alloc_tensor(&[4], crate::dtype::DType::I32);
    let counters_ptr = counters.as_gpu_tensor().raw_ptr() as u64;
    let r = sys::cuMemsetD32Async(counters_ptr, 0, 4, stream);
    assert_eq!(r, sys::cudaError_enum::CUDA_SUCCESS);

    let phase1_gemm = build_flat_params(
        input.raw_ptr() as u64,
        gate_up_weight.raw_ptr() as u64,
        gate_up_buf.as_gpu_tensor().raw_ptr() as u64,
        gate_up_buf.as_gpu_tensor().raw_ptr() as u64,
        m,
        gate_up_n,
        hidden,
        hidden,
        hidden,
        gate_up_n,
        gate_up_n,
        1.0,
        0.0,
    );
    let phase2_gemm = build_flat_params(
        gate_up_buf.as_gpu_tensor().raw_ptr() as u64,
        down_weight.raw_ptr() as u64,
        output.as_gpu_tensor().raw_ptr() as u64,
        output.as_gpu_tensor().raw_ptr() as u64,
        m,
        down_n,
        intermediate,
        gate_up_n,
        intermediate,
        down_n,
        down_n,
        1.0,
        0.0,
    );

    let gn1 = gate_up_n.div_ceil(tile_n);
    let s1 = 1u32 << compute_swizzle_log(gn1);
    let total1 = m.div_ceil(tile_m) * s1 * gn1.div_ceil(s1);
    let gn2 = down_n.div_ceil(tile_n);
    let s2 = 1u32 << compute_swizzle_log(gn2);
    let total2 = m.div_ceil(tile_m) * s2 * gn2.div_ceil(s2);

    let intermediate_bytes = (intermediate as u64) * 2;

    let mut params = vec![0u8; 300];
    let mut o = 0usize;
    macro_rules! p64 {
        ($v:expr) => {
            params[o..o + 8].copy_from_slice(&($v as u64).to_le_bytes());
            o += 8;
        };
    }
    macro_rules! p32 {
        ($v:expr) => {
            params[o..o + 4].copy_from_slice(&($v as u32).to_le_bytes());
            o += 4;
        };
    }
    macro_rules! pf32 {
        ($v:expr) => {
            params[o..o + 4].copy_from_slice(&($v as f32).to_le_bytes());
            o += 4;
        };
    }

    p64!(input.raw_ptr());
    p64!(norm_weight.raw_ptr());
    pf32!(epsilon);
    p32!(hidden);
    p64!(hidden as u64);
    params[o..o + 88].copy_from_slice(&phase1_gemm);
    o += 88;
    p64!(intermediate_bytes);
    params[o..o + 88].copy_from_slice(&phase2_gemm);
    o += 88;
    p64!(counters_ptr); // counter1
    p64!(counters_ptr + 4); // counter2
    p64!(counters_ptr + 8); // barrier
    p32!(num_sm);
    p32!(total1);
    p32!(total2);

    launch_kernel_raw(func, stream, num_sm, 1, 128, 36864, &params[..o]);
    drop(gate_up_buf);
    drop(counters);
    output
}
