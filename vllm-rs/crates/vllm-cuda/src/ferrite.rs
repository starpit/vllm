// SPDX-License-Identifier: Apache-2.0
//! Ferrite CUTLASS GEMM integration.
//!
//! Provides a feature-gated CUTLASS bf16 GEMM path as an alternative to cuBLAS.
//! Multiple tile configurations are compiled at build time (via ptx-fusion proc
//! macros) and the best is selected at runtime based on problem size.
//!
//! Enable with: `--features ferrite`

use anyhow::{Result, bail};
use cudarc::driver::sys::{self, CUfunction, CUmodule, CUstream};

use crate::DType;
use crate::alloc::{CachingAllocator, OwnedTensor};
use crate::tensor::GpuTensor;

// ── Compile-time: extract each CUTLASS config from the multi-entry PTX ──

// PTX files live in ptx-fusion's kernels/ directory.
// The extract_entry! macro resolves paths relative to CARGO_MANIFEST_DIR,
// so we use a relative path from vllm-cuda to ptx-fusion.
const PTX_PATH: &str = "../ptx-fusion/kernels/cutlass_bf16_configs_sm89.ptx";

ptx_fusion::extract_entry!(
    "../ptx-fusion/kernels/cutlass_bf16_configs_sm89.ptx",
    "GemmShapeILi64ELi128ELi32E",
    CONFIG_64X128X32_PTX
);

ptx_fusion::extract_entry!(
    "../ptx-fusion/kernels/cutlass_bf16_configs_sm89.ptx",
    "GemmShapeILi128ELi128ELi32E",
    CONFIG_128X128X32_PTX
);

ptx_fusion::extract_entry!(
    "../ptx-fusion/kernels/cutlass_bf16_configs_sm89.ptx",
    "GemmShapeILi128ELi128ELi64E",
    CONFIG_128X128X64_PTX
);

// ── Runtime tile configuration ──

struct TileConfig {
    name: &'static str,
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
    threads: u32,
    smem_bytes: u32,
    func: CUfunction,
    _module: CUmodule,
    /// Iterator params: value = stride * slope + intercept
    a_slope: [i64; 4],
    a_intercept: [i64; 4],
    b_slope: [i64; 4],
    b_intercept: [i64; 4],
    cd_slope: [i64; 8],
}

/// Ferrite CUTLASS GEMM dispatcher.
///
/// Loads CUTLASS PTX modules once at device init and provides a `gemm()`
/// method that selects the best tile config and launches the kernel.
pub struct FerriteCutlass {
    configs: Vec<TileConfig>,
}

impl FerriteCutlass {
    /// Initialize: load all CUTLASS PTX modules and resolve entry points.
    pub unsafe fn new() -> Result<Self> {
        let specs: Vec<(
            &str,
            &str,
            u32,
            u32,
            u32,
            u32,
            u32,
            [i64; 4],
            [i64; 4],
            [i64; 4],
            [i64; 4],
            [i64; 8],
        )> = vec![
            (
                "64x128x32",
                CONFIG_64X128X32_PTX,
                64,
                128,
                32,
                128,
                36864,
                [1, 16, -16, 0],
                [0, 0, 64, 64],
                [1, 16, -48, 0],
                [0, 0, 64, 64],
                [2, 4, -2, -2, 16, 64, 128, 32],
            ),
            // NOTE: 128x128x32 and 128x128x64 removed — produce garbage.
            // Only 64x128x32 is verified correct in model inference. — produces garbage in model inference.
            // The iterator constants are verified correct, but something in the
            // PTX or launch path is broken. Needs investigation.
            // Keep only 64x128x32 and 128x128x32 for now.
        ];

        let mut configs = Vec::new();
        for (name, ptx, tm, tn, tk, threads, smem, a_s, a_i, b_s, b_i, cd_s) in specs {
            let (module, func) = load_ptx_module(ptx)?;
            configs.push(TileConfig {
                name,
                tile_m: tm,
                tile_n: tn,
                tile_k: tk,
                threads,
                smem_bytes: smem,
                func,
                _module: module,
                a_slope: a_s,
                a_intercept: a_i,
                b_slope: b_s,
                b_intercept: b_i,
                cd_slope: cd_s,
            });
        }

        // Sort by tile_m ascending for selection
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
    ///
    /// A: [M, K] bf16 row-major (input activation)
    /// B: [N, K] bf16 row-major (weight, NOT transposed — equivalent to col-major [K, N])
    /// C: [M, N] bf16 row-major (residual, or same as D for in-place)
    /// D: [M, N] bf16 row-major (output)
    ///
    /// If `c` is None, beta is forced to 0.0 and a zero buffer is used for C.
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

        // Build CUTLASS params (368 bytes)
        let c_ptr = match c {
            Some(ct) => ct.raw_ptr() as u64,
            None => out.as_gpu_tensor().raw_ptr() as u64, // D == C when beta=0
        };
        let actual_beta = if c.is_some() { beta } else { 0.0 };

        let params = build_params(
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
            config,
            alpha,
            actual_beta,
        );

        // Compute grid — must match CUTLASS get_grid_shape exactly:
        //   tile = 1 << get_log_tile(grid_tiled_shape)
        //   grid = (grid_m * tile, ceil(grid_n / tile), 1)
        let grid_m = m.div_ceil(config.tile_m);
        let grid_n = n.div_ceil(config.tile_n);
        let swizzle_log = compute_swizzle_log(grid_m as i32, grid_n as i32);
        let tile = 1u32 << swizzle_log;
        let grid_x = grid_m * tile;
        let grid_y = grid_n.div_ceil(tile);

        // Launch
        launch_cutlass(
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
}

// ── Internal helpers ──

/// Load a PTX string as a module and resolve its single entry function.
unsafe fn load_ptx_module(ptx: &str) -> Result<(CUmodule, CUfunction)> {
    let mut module: CUmodule = std::ptr::null_mut();
    let ptx_cstr =
        std::ffi::CString::new(ptx).map_err(|e| anyhow::anyhow!("PTX contains null byte: {e}"))?;

    let result = sys::cuModuleLoadData(&mut module, ptx_cstr.as_ptr() as *const _);
    if result != sys::cudaError_enum::CUDA_SUCCESS {
        bail!("cuModuleLoadData failed: {result:?}");
    }

    // Find the entry point (first function containing "_ZN7cutlass")
    let entry_name = find_entry_name(ptx)?;
    let entry_cstr = std::ffi::CString::new(entry_name.as_str()).unwrap();

    let mut func: CUfunction = std::ptr::null_mut();
    let result = sys::cuModuleGetFunction(&mut func, module, entry_cstr.as_ptr());
    if result != sys::cudaError_enum::CUDA_SUCCESS {
        bail!("cuModuleGetFunction failed for {entry_name}: {result:?}");
    }

    // Set max dynamic SMEM if needed
    // (L4 supports up to 99KB optin)
    let result = sys::cuFuncSetAttribute(
        func,
        sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        99 * 1024, // 99KB — covers all our configs
    );
    if result != sys::cudaError_enum::CUDA_SUCCESS {
        bail!("cuFuncSetAttribute (SMEM) failed: {result:?}");
    }

    Ok((module, func))
}

fn find_entry_name(ptx: &str) -> Result<String> {
    for line in ptx.lines() {
        let t = line.trim();
        if t.contains(".entry")
            && t.contains('(')
            && let Some(start) = t.find("_ZN7cutlass")
        {
            let end = t.find('(').unwrap_or(t.len());
            return Ok(t[start..end].trim().to_string());
        }
    }
    bail!("no CUTLASS entry found in PTX");
}

/// Build the 368-byte CUTLASS Params struct.
fn build_params(
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
    config: &TileConfig,
    alpha: f32,
    beta: f32,
) -> [u8; 368] {
    let mut p = [0u8; 368];

    // problem_size
    w32(&mut p, 0, m as i32);
    w32(&mut p, 4, n as i32);
    w32(&mut p, 8, k as i32);

    // grid_tiled_shape
    let grid_m = m.div_ceil(config.tile_m) as i32;
    let grid_n = n.div_ceil(config.tile_n) as i32;
    w32(&mut p, 12, grid_m);
    w32(&mut p, 16, grid_n);
    w32(&mut p, 20, 1);

    // swizzle_log_tile
    w32(&mut p, 24, compute_swizzle_log(grid_m, grid_n) as i32);

    // params_A (4x i64)
    let lda64 = lda as i64;
    for i in 0..4 {
        w64(
            &mut p,
            32 + i * 8,
            lda64 * config.a_slope[i] + config.a_intercept[i],
        );
    }
    wu64(&mut p, 64, a_ptr);
    w64(&mut p, 72, lda64);

    // params_B (4x i64)
    let ldb64 = ldb as i64;
    for i in 0..4 {
        w64(
            &mut p,
            80 + i * 8,
            ldb64 * config.b_slope[i] + config.b_intercept[i],
        );
    }
    wu64(&mut p, 112, b_ptr);
    w64(&mut p, 120, ldb64);

    // params_C (8x i64)
    let ldc64 = ldc as i64;
    for i in 0..8 {
        w64(&mut p, 128 + i * 8, ldc64 * config.cd_slope[i]);
    }
    wu64(&mut p, 192, c_ptr);
    w64(&mut p, 200, ldc64);

    // params_D (8x i64)
    let ldd64 = ldd as i64;
    for i in 0..8 {
        w64(&mut p, 208 + i * 8, ldd64 * config.cd_slope[i]);
    }
    wu64(&mut p, 272, d_ptr);
    w64(&mut p, 280, ldd64);

    // output_op: alpha, beta
    wf32(&mut p, 288, alpha);
    wf32(&mut p, 292, beta);

    // gemm_k_size
    w32(&mut p, 336, k as i32);

    p
}

/// Launch a CUTLASS kernel via raw CUDA driver API.
unsafe fn launch_cutlass(
    func: CUfunction,
    stream: CUstream,
    grid_x: u32,
    grid_y: u32,
    block_x: u32,
    smem_bytes: u32,
    params: &[u8; 368],
) {
    let mut param_size = 368usize;
    let extra: [*mut std::ffi::c_void; 5] = [
        sys::CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT as *mut _,
        params.as_ptr() as *mut _,
        sys::CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT as *mut _,
        &mut param_size as *mut usize as *mut _,
        sys::CU_LAUNCH_PARAM_END_AS_INT as *mut _,
    ];

    let result = sys::cuLaunchKernel(
        func,
        grid_x,
        grid_y,
        1, // grid
        block_x,
        1,
        1,                    // block
        smem_bytes,           // shared mem
        stream,               // stream
        std::ptr::null_mut(), // kernelParams (unused with extra)
        extra.as_ptr() as *mut *mut _,
    );
    debug_assert_eq!(
        result,
        sys::cudaError_enum::CUDA_SUCCESS,
        "cuLaunchKernel failed: {result:?}"
    );
}

/// Compute swizzle log for GemmIdentityThreadblockSwizzle<4>.
///
/// Compute swizzle log for GemmIdentityThreadblockSwizzle<N>.
///
/// From CUTLASS threadblock_swizzle.h get_log_tile():
///   if N >= 8 && n >= 6 → 3
///   if N >= 4 && n >= 3 → 2
///   if N >= 2 && n >= 2 → 1
///   else → 0
fn compute_swizzle_log(_grid_m: i32, grid_n: i32) -> u32 {
    const SWIZZLE_N: i32 = 4; // GemmIdentityThreadblockSwizzle<4>
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

fn w32(buf: &mut [u8], off: usize, v: i32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}
fn w64(buf: &mut [u8], off: usize, v: i64) {
    buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
}
fn wu64(buf: &mut [u8], off: usize, v: u64) {
    buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
}
fn wf32(buf: &mut [u8], off: usize, v: f32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}
