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
    "../ptx-fusion/kernels/cutlass_bf16_16x128x32_sm89.ptx",
    "../ptx-fusion/kernels/cutlass_bf16_16x128x32_sm89.derivations.json",
    "ferrite_gemm_16x128x32",
    FLAT_16X128X32_PTX
);

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

// ── Gemv kernel for M=1 decode ──

const GEMV_PTX: &str = include_str!("../../ptx-fusion/kernels/ferrite_gemv_bf16.ptx");

// ── MLP block: gate_up GEMM → SiLU+mul → down GEMM in one kernel ──

const MLP_BLOCK_PTX: &str = ptx_fusion::persistent_mlp_block!(
    "../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    "../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.derivations.json",
    "../ptx-fusion/kernels/vllm_silu_mul.ptx",
    "ferrite_mlp_block"
);

const MLP_REGTRANSFER_PTX: &str = ptx_fusion::register_transfer_mlp!(
    "../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.ptx",
    "../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.derivations.json",
    "../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.ptx",
    "../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.derivations.json",
    "../ptx-fusion/kernels/vllm_silu_mul.ptx",
    "ferrite_mlp_regtransfer"
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
    /// bf16 gemv kernel for M=1 decode (hand-written, coalesced).
    gemv: Option<CudaFunc>,
    /// Fused MLP block kernel (gate_up → SiLU → down), persistent version.
    mlp_block: Option<CudaFunc>,
    /// Register transfer MLP kernel (gate_up → SiLU → down), non-persistent.
    mlp_regtransfer: Option<CudaFunc>,
    /// Device memory for global barrier counters (reused across launches).
    barrier_counters: *mut std::ffi::c_void,
}

impl FerriteCutlass {
    /// Initialize: load flat-param PTX modules and resolve entry points.
    pub unsafe fn new() -> Result<Self> {
        let specs: &[(&str, &str, &str, u32, u32, u32, u32)] = &[
            (
                "16x128x32",
                "ferrite_gemm_16x128x32",
                FLAT_16X128X32_PTX,
                16,
                128,
                64,
                27648,
            ),
            (
                "64x128x32",
                "ferrite_gemm_64x128x32",
                FLAT_64X128X32_PTX,
                64,
                128,
                128,
                36864,
            ),
        ];

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

        // Load gemv kernel for M=1 decode (1024 bytes SMEM for reduction)
        let gemv = match load_flat_module(GEMV_PTX, "ferrite_gemv_bf16", 1024) {
            Ok((module, func)) => Some(CudaFunc(module, func)),
            Err(e) => {
                eprintln!("ferrite: gemv kernel load failed (non-fatal): {e}");
                None
            }
        };

        // Load the fused MLP block kernel
        let mlp_block = match load_flat_module(MLP_BLOCK_PTX, "ferrite_mlp_block", 36864 + 512) {
            Ok((module, func)) => Some(CudaFunc(module, func)),
            Err(e) => {
                eprintln!("ferrite: MLP block kernel load failed (non-fatal): {e}");
                None
            }
        };

        // Load register transfer MLP kernel (40KB SMEM: 24KB CUTLASS + 8KB gate + 8KB up)
        let mlp_regtransfer = match load_flat_module(MLP_REGTRANSFER_PTX, "ferrite_mlp_regtransfer", 40960) {
            Ok((module, func)) => Some(CudaFunc(module, func)),
            Err(e) => {
                eprintln!("ferrite: register transfer MLP kernel load failed (non-fatal): {e}");
                None
            }
        };

        // Allocate barrier counters: [tile_counter(u32)] + [mtile_done array(u32 × max_mtiles)]
        // Max M in practice: 2048 tokens → ceil(2048/64) = 32 M-tiles. Allocate for 64.
        let barrier_alloc_bytes = 4 + 64 * 4; // 260 bytes
        let mut barrier_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let result = sys::cuMemAlloc_v2(
            &mut barrier_ptr as *mut *mut _ as *mut u64,
            barrier_alloc_bytes,
        );
        if result != sys::cudaError_enum::CUDA_SUCCESS {
            bail!("cuMemAlloc for barrier counters: {result:?}");
        }

        Ok(FerriteCutlass {
            configs,
            gemv,
            mlp_block,
            mlp_regtransfer,
            barrier_counters: barrier_ptr,
        })
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

        // Use gemv for M=1 (decode): 6x faster than tiled GEMM
        if m == 1 && c.is_none() && alpha == 1.0 && beta == 0.0 {
            if let Some(ref gemv) = self.gemv {
                return self.launch_gemv(gemv.1, a, b, n, k, alloc, stream);
            }
        }

        let config = self.select(m);

        // CUTLASS reads full [tile_m × K] A-tiles via cp.async even past M.
        // Pad output to tile_m boundary so downstream consumers (which may be
        // inputs to the next GEMM) have enough backing memory.
        let padded_m = (m.div_ceil(config.tile_m) * config.tile_m) as usize;
        let mut out = alloc.alloc_tensor(&[padded_m, n as usize], a.dtype());
        unsafe { out.reshape(&[m as usize, n as usize], a.dtype()) };

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

    /// Launch persistent fused MLP block: gate_up GEMM → SiLU+mul → down GEMM.
    ///
    /// Uses a persistent work-queue loop with per-M-tile barriers.
    /// Replaces 3 separate launches with a single kernel launch.
    pub unsafe fn launch_mlp_block(
        &self,
        normed_input: GpuTensor,
        gate_up_weight: GpuTensor,
        down_weight: GpuTensor,
        intermediate_size: u32,
        alloc: &mut CachingAllocator,
        stream: CUstream,
    ) -> (OwnedTensor, OwnedTensor) {
        let func = match &self.mlp_block {
            Some(cf) => cf.1,
            None => panic!("ferrite: MLP block kernel not loaded"),
        };

        let m = normed_input.dim(0) as u32;
        let hidden = normed_input.dim(1) as u32;
        let gate_up_cols = 2 * intermediate_size;

        // Compute grid dims for each phase
        let tile_m = 64u32;
        let tile_n = 128u32;

        // CUTLASS accesses full [tile_m × tile_n] tiles via cp.async even past
        // the logical M boundary.  Pad intermediate + output buffers to the tile
        // boundary.  The input (normed_input) is already padded because it came
        // from a ferrite GEMM output which is also padded.
        let padded_m = m.div_ceil(tile_m) * tile_m;

        let mut gate_up_buf = alloc.alloc_tensor(
            &[padded_m as usize, gate_up_cols as usize],
            normed_input.dtype(),
        );
        unsafe { gate_up_buf.reshape(&[m as usize, gate_up_cols as usize], normed_input.dtype()) };
        let mut output =
            alloc.alloc_tensor(&[padded_m as usize, hidden as usize], normed_input.dtype());
        unsafe { output.reshape(&[m as usize, hidden as usize], normed_input.dtype()) };

        let grid_m = m.div_ceil(tile_m);

        let grid_n_0 = gate_up_cols.div_ceil(tile_n);
        let sw0 = compute_swizzle_log(grid_n_0);
        let tile_0 = 1u32 << sw0;
        let gx0 = grid_m * tile_0;
        let gy0 = grid_n_0.div_ceil(tile_0);
        let phase0_tiles = gx0 * gy0;

        let grid_n_1 = hidden.div_ceil(tile_n);
        let sw1 = compute_swizzle_log(grid_n_1);
        let tile_1 = 1u32 << sw1;
        let gx1 = grid_m * tile_1;
        let gy1 = grid_n_1.div_ceil(tile_1);
        let phase1_tiles = gx1 * gy1;

        let total_tiles = phase0_tiles + phase1_tiles;
        let num_sms = 58u32; // L4
        let num_blocks = total_tiles.min(num_sms);

        // N-tiles per M-tile for phase 0 barrier
        let ntiles_per_m_0 = tile_0 * gy0;

        // Zero persistent counter + M-tile done counters
        // barrier_counters layout: [0] = tile counter, [1..] = mtile_done array
        let max_mtiles = grid_m;
        let counter_bytes = 4 + max_mtiles * 4; // u32 counter + u32 per m-tile
        // Use stream-ordered memset so zeroing completes before the kernel
        sys::cuMemsetD32Async(
            self.barrier_counters as u64,
            0,
            (counter_bytes / 4) as usize,
            stream,
        );

        let counter_ptr = self.barrier_counters as u64;
        let mtile_done_ptr = counter_ptr + 4; // right after the tile counter

        // Phase 0 params
        let params_gate_up = build_flat_params(
            normed_input.raw_ptr() as u64,
            gate_up_weight.raw_ptr() as u64,
            gate_up_buf.as_gpu_tensor().raw_ptr() as u64,
            gate_up_buf.as_gpu_tensor().raw_ptr() as u64,
            m,
            gate_up_cols,
            hidden,
            hidden,
            hidden,
            gate_up_cols,
            gate_up_cols,
            1.0,
            0.0,
        );

        // Phase 1 params
        let intermediate_bytes = (intermediate_size as u64) * 2;
        let params_down = build_flat_params(
            gate_up_buf.as_gpu_tensor().raw_ptr() as u64,
            down_weight.raw_ptr() as u64,
            output.as_gpu_tensor().raw_ptr() as u64,
            output.as_gpu_tensor().raw_ptr() as u64,
            m,
            hidden,
            intermediate_size,
            gate_up_cols,
            intermediate_size,
            hidden,
            hidden,
            1.0,
            0.0,
        );

        // Use kernelParams API (same as cudarc::launch_builder).
        // Each entry is a pointer to the parameter value.
        let mut kp: [*mut std::ffi::c_void; 10] = [
            &counter_ptr as *const u64 as *mut _,
            &total_tiles as *const u32 as *mut _,
            &gx0 as *const u32 as *mut _,
            &phase0_tiles as *const u32 as *mut _,
            &gx1 as *const u32 as *mut _,
            &ntiles_per_m_0 as *const u32 as *mut _,
            &mtile_done_ptr as *const u64 as *mut _,
            &params_gate_up as *const [u8; 88] as *mut _,
            &intermediate_bytes as *const u64 as *mut _,
            &params_down as *const [u8; 88] as *mut _,
        ];

        let result = sys::cuLaunchKernel(
            func,
            num_blocks,
            1,
            1,
            128,
            1,
            1,
            36864 + 512,
            stream,
            kp.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        assert_eq!(
            result,
            sys::cudaError_enum::CUDA_SUCCESS,
            "ferrite launch_mlp_block failed: {result:?}"
        );

        (output, gate_up_buf)
    }

    /// Launch the gemv kernel: y[1,N] = x[1,K] @ W[N,K]^T
    unsafe fn launch_gemv(
        &self,
        func: CUfunction,
        a: GpuTensor, // x [1, K]
        b: GpuTensor, // W [N, K]
        n: u32,
        k: u32,
        alloc: &mut CachingAllocator,
        stream: CUstream,
    ) -> OwnedTensor {
        let out = alloc.alloc_tensor(&[1, n as usize], a.dtype());

        let w_ptr = b.raw_ptr() as u64;
        let x_ptr = a.raw_ptr() as u64;
        let y_ptr = out.as_gpu_tensor().raw_ptr() as u64;
        let stride_w = k;

        let grid_x = n.div_ceil(32); // 32 output columns per block

        let mut kp: [*mut std::ffi::c_void; 6] = [
            &w_ptr as *const u64 as *mut _,
            &x_ptr as *const u64 as *mut _,
            &y_ptr as *const u64 as *mut _,
            &n as *const u32 as *mut _,
            &k as *const u32 as *mut _,
            &stride_w as *const u32 as *mut _,
        ];

        let result = sys::cuLaunchKernel(
            func,
            grid_x,
            1,
            1,
            256, // 256 threads per block
            1,
            1,
            1024, // shared memory for reduction (256 * 4 bytes)
            stream,
            kp.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        assert_eq!(
            result,
            sys::cudaError_enum::CUDA_SUCCESS,
            "ferrite launch_gemv failed: {result:?}"
        );

        out
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

    // JIT-compile with FTZ=true to match the source kernels' --use_fast_math
    // compilation. Without this, scalar FP operations (add.f32, fma.rn.f32)
    // in the fused prologue handle subnormals differently than the original
    // C kernels, causing ~1 ULP divergence that compounds over layers.
    let mut options = [sys::CUjit_option::CU_JIT_FTZ];
    let mut values: [*mut std::ffi::c_void; 1] = [1usize as *mut _]; // 1 = enable FTZ
    let result = sys::cuModuleLoadDataEx(
        &mut module,
        ptx_cstr.as_ptr() as *const _,
        1,
        options.as_mut_ptr(),
        values.as_mut_ptr(),
    );
    if result != sys::cudaError_enum::CUDA_SUCCESS {
        bail!("cuModuleLoadDataEx failed: {result:?}");
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

    // Per-arg launch via kernelParams (not CU_LAUNCH_PARAM_BUFFER).
    // The kernelParams API lets CUDA handle alignment automatically.
    let mut rms_input = input.raw_ptr() as u64;
    let mut rms_weight = norm_weight.raw_ptr() as u64;
    let mut rms_epsilon = epsilon;
    let mut rms_hidden = hidden;
    let mut rms_stride = k as u64;

    let mut kernel_params: [*mut std::ffi::c_void; 6] = [
        &mut rms_input as *mut u64 as *mut _,
        &mut rms_weight as *mut u64 as *mut _,
        &mut rms_epsilon as *mut f32 as *mut _,
        &mut rms_hidden as *mut u32 as *mut _,
        &mut rms_stride as *mut u64 as *mut _,
        flat.as_ptr() as *mut _,
    ];

    let result = sys::cuLaunchKernel(
        func,
        grid_x,
        grid_y,
        1,
        kernel.threads,
        1,
        1,
        kernel.smem_bytes,
        stream,
        kernel_params.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    assert_eq!(
        result,
        sys::cudaError_enum::CUDA_SUCCESS,
        "ferrite launch_fused_norm_gemm failed: {result:?}"
    );

    out
}

/// Launch a fused add+norm+GEMM kernel (two-input prologue).
///
/// The prologue reads from `residual` and `hs_input`, computes f32(res + hs),
/// writes bf16(sum) back to `residual`, and computes inv_rms from the f32 sums.
/// The GEMM body reads from `residual` (now holding bf16 sums), normalizes each
/// element (mul inv_rms * weight), and feeds into MMA.
///
/// This matches `fused_add_rms_norm_inplace` precision: inv_rms is computed from
/// f32 sums (not bf16-truncated values).
pub unsafe fn launch_fused_add_norm_gemm(
    kernel: &FeriteKernel,
    residual: GpuTensor,
    hs_input: GpuTensor,
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

    let m = residual.dim(0) as u32;
    let k = residual.dim(1) as u32;
    let n = weight.dim(0) as u32;

    let out = alloc.alloc_tensor(&[m as usize, n as usize], residual.dtype());

    // The prologue is pure-read: it reads from both residual and hs_input,
    // adds in f32, computes inv_rms. No GMEM writes — all blocks see identical
    // original data. The per-site code loads from both buffers and adds at each
    // A-load site.
    //
    // IMPORTANT: This function does NOT update the residual buffer. The caller
    // must run add_inplace(residual, hs_input) AFTER this call to update the
    // residual for downstream layers. We don't do it here because the caller
    // controls tensor lifetimes — if hs_input is dropped before the stream
    // executes add_inplace, the GPU reads freed memory.

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
        residual.raw_ptr() as u64,
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

    let mut rms_input = residual.raw_ptr() as u64;
    let mut rms_hs_input = hs_input.raw_ptr() as u64;
    let mut rms_weight = norm_weight.raw_ptr() as u64;
    let mut rms_epsilon = epsilon;
    let mut rms_hidden = hidden;
    let mut rms_stride = k as u64;

    let mut kernel_params: [*mut std::ffi::c_void; 7] = [
        &mut rms_input as *mut u64 as *mut _,
        &mut rms_hs_input as *mut u64 as *mut _,
        &mut rms_weight as *mut u64 as *mut _,
        &mut rms_epsilon as *mut f32 as *mut _,
        &mut rms_hidden as *mut u32 as *mut _,
        &mut rms_stride as *mut u64 as *mut _,
        flat.as_ptr() as *mut _,
    ];

    let result = sys::cuLaunchKernel(
        func,
        grid_x,
        grid_y,
        1,
        kernel.threads,
        1,
        1,
        kernel.smem_bytes,
        stream,
        kernel_params.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    assert_eq!(
        result,
        sys::cudaError_enum::CUDA_SUCCESS,
        "ferrite launch_fused_add_norm_gemm failed: {result:?}"
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
