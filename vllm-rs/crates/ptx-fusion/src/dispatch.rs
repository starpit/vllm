//! Runtime CUTLASS GEMM dispatcher with compile-time tile selection.
//!
//! Compiles multiple CUTLASS bf16 GEMM tile configurations at build time
//! via proc macros, then selects the best at runtime based on problem size.
//!
//! ```rust,ignore
//! use ptx_fusion::dispatch::{CutlassDispatch, CutlassConfigSpec};
//!
//! // At compile time: extract configs from multi-entry PTX
//! ptx_fusion::extract_entry!("kernels/cutlass_bf16_configs_sm89.ptx",
//!     "GemmShapeILi64ELi64ELi32E", CONFIG_64x64);
//! ptx_fusion::extract_entry!("kernels/cutlass_bf16_configs_sm89.ptx",
//!     "GemmShapeILi128ELi128ELi32E", CONFIG_128x128);
//!
//! // At runtime: load and dispatch
//! let dispatch = CutlassDispatch::new(&ctx, vec![
//!     CutlassConfigSpec::new("64x64x32", 64, 64, 32, 128, 24576, CONFIG_64x64),
//!     CutlassConfigSpec::new("128x128x32", 128, 128, 32, 128, 49152, CONFIG_128x128),
//! ])?;
//! let config = dispatch.select(batch_size);
//! ```

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// A single compiled CUTLASS GEMM configuration.
pub struct CutlassConfig {
    pub name: &'static str,
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub threads: u32,
    pub smem_bytes: u32,
    /// The loaded CUfunction, ready to launch.
    pub func: CudaFunction,
    _module: Arc<CudaModule>,
}

/// Specification for a config (before loading).
pub struct CutlassConfigSpec {
    pub name: &'static str,
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub threads: u32,
    pub smem_bytes: u32,
    pub ptx: &'static str,
}

impl CutlassConfigSpec {
    pub fn new(
        name: &'static str,
        tile_m: u32,
        tile_n: u32,
        tile_k: u32,
        threads: u32,
        smem_bytes: u32,
        ptx: &'static str,
    ) -> Self {
        Self {
            name,
            tile_m,
            tile_n,
            tile_k,
            threads,
            smem_bytes,
            ptx,
        }
    }
}

/// Dispatcher that selects the best tile config per problem size.
///
/// Selection heuristic (from L4 benchmarks):
/// - M ≤ tile_m of smallest config → use smallest tile (low launch overhead)
/// - M > 256 → use largest tile that fits in SMEM
/// - Otherwise → use the medium tile
pub struct CutlassDispatch {
    configs: Vec<CutlassConfig>,
}

impl CutlassDispatch {
    /// Load all compiled CUTLASS configs from PTX.
    pub fn new(ctx: &Arc<CudaContext>, specs: Vec<CutlassConfigSpec>) -> Result<Self, String> {
        let mut configs = Vec::new();
        for spec in specs {
            let entry = find_entry_name(spec.ptx)?;
            let module = ctx
                .load_module(Ptx::from_src(spec.ptx))
                .map_err(|e| format!("load module for {}: {e}", spec.name))?;
            let func = module
                .load_function(&entry)
                .map_err(|e| format!("load function {entry}: {e}"))?;

            configs.push(CutlassConfig {
                name: spec.name,
                tile_m: spec.tile_m,
                tile_n: spec.tile_n,
                tile_k: spec.tile_k,
                threads: spec.threads,
                smem_bytes: spec.smem_bytes,
                func,
                _module: module,
            });
        }

        // Sort by tile_m ascending for selection
        configs.sort_by_key(|c| c.tile_m);

        Ok(CutlassDispatch { configs })
    }

    /// Select the best config for the given M dimension.
    ///
    /// Heuristic from L4 bf16 benchmarks:
    /// - Small M (decode): smallest tile wins (less wasted work, lower overhead)
    /// - Large M (prefill): largest tile wins (better compute utilization)
    pub fn select(&self, m: u32) -> &CutlassConfig {
        if self.configs.len() == 1 {
            return &self.configs[0];
        }

        // For very small M (≤ smallest tile), always use smallest
        if m <= self.configs[0].tile_m {
            return &self.configs[0];
        }

        // Find the largest tile whose tile_m ≤ max(m, 64)
        let effective_m = m.max(64);
        self.configs
            .iter()
            .rev()
            .find(|c| c.tile_m <= effective_m)
            .unwrap_or(&self.configs[0])
    }

    /// Get all loaded configs (for custom selection or profiling).
    pub fn configs(&self) -> &[CutlassConfig] {
        &self.configs
    }

    /// Compute grid dimensions for a CUTLASS launch.
    pub fn grid_dim(config: &CutlassConfig, m: u32, n: u32) -> (u32, u32, u32) {
        let grid_m = m.div_ceil(config.tile_m);
        let grid_n = n.div_ceil(config.tile_n);

        // GemmIdentityThreadblockSwizzle<4> reorders tiles for L2 locality
        let swizzle_log = compute_swizzle_log(grid_m as i32, grid_n as i32);
        let swizzle = 1u32 << swizzle_log;
        let grid_x = grid_m * grid_n.div_ceil(swizzle);
        let grid_y = swizzle;
        (grid_x, grid_y, 1)
    }
}

// ── GemmParams builder ──

/// Iterator parameter constants for a CUTLASS tile configuration.
///
/// Each field is `stride * slope + intercept`, derived empirically from
/// the CUTLASS C++ API. These constants are tile-config-specific but
/// stride-independent — compute once per config, apply for any stride.
#[derive(Clone, Copy)]
pub struct IteratorConstants {
    /// params_A: [stride, inc_strided, inc_next, inc_advance]
    /// Each: value = lda * slope + intercept
    pub a_slope: [i64; 4],
    pub a_intercept: [i64; 4],
    /// params_B: same structure, using ldb
    pub b_slope: [i64; 4],
    pub b_intercept: [i64; 4],
    /// params_C/D: [8 values], each = ldc * slope (no intercept)
    pub cd_slope: [i64; 8],
}

/// Pre-computed constants for the three production configs.
impl IteratorConstants {
    /// 64x128x32, 3 stages
    pub const CONFIG_64X128X32: Self = Self {
        a_slope: [1, 16, -16, 0],
        a_intercept: [0, 0, 64, 64],
        b_slope: [1, 16, -48, 0],
        b_intercept: [0, 0, 64, 64],
        cd_slope: [2, 4, -2, -2, 16, 64, 128, 32],
    };

    /// 128x128x32, 3 stages
    pub const CONFIG_128X128X32: Self = Self {
        a_slope: [1, 16, -48, 0],
        a_intercept: [0, 0, 64, 64],
        b_slope: [1, 16, -48, 0],
        b_intercept: [0, 0, 64, 64],
        cd_slope: [2, 4, -2, -2, 16, 128, 256, 32],
    };

    /// 128x128x64, 3 stages
    pub const CONFIG_128X128X64: Self = Self {
        a_slope: [1, 8, -56, 0],
        a_intercept: [0, 0, 128, 128],
        b_slope: [1, 8, -56, 0],
        b_intercept: [0, 0, 128, 128],
        cd_slope: [2, 4, -2, -2, 16, 128, 256, 32],
    };
}

/// CUTLASS Params struct (368 bytes).
///
/// Layout (validated via offsetof against CUTLASS C++ API):
/// ```text
/// [  0- 11] problem_size: {m, n, k} as 3x i32
/// [ 12- 23] grid_tiled_shape: {grid_m, grid_n, 1} as 3x i32
/// [ 24- 27] swizzle_log_tile: i32
/// [ 28- 31] padding
/// [ 32- 63] params_A: 4x i64 (stride-linear iterator params)
/// [ 64- 79] ref_A: {ptr: u64, stride: i64}
/// [ 80-111] params_B: 4x i64
/// [112-127] ref_B: {ptr: u64, stride: i64}
/// [128-191] params_C: 8x i64 (epilogue iterator params)
/// [192-207] ref_C: {ptr: u64, stride: i64}
/// [208-271] params_D: 8x i64
/// [272-287] ref_D: {ptr: u64, stride: i64}
/// [288-327] output_op: {alpha: f32, beta: f32, ...padding}
/// [328-335] semaphore: ptr (null)
/// [336-339] gemm_k_size: i32
/// [340-343] padding
/// [344-367] gather/scatter indices: 3x ptr (null)
/// ```
#[repr(C, align(8))]
pub struct GemmParams {
    pub bytes: [u8; 368],
}

impl GemmParams {
    /// Build params from pointers, dimensions, and per-config constants.
    pub fn new(
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
        tile_m: u32,
        tile_n: u32,
        consts: &IteratorConstants,
    ) -> Self {
        let mut p = GemmParams { bytes: [0u8; 368] };

        // problem_size
        w32(&mut p.bytes, 0, m as i32);
        w32(&mut p.bytes, 4, n as i32);
        w32(&mut p.bytes, 8, k as i32);

        // grid_tiled_shape
        let grid_m = m.div_ceil(tile_m) as i32;
        let grid_n = n.div_ceil(tile_n) as i32;
        w32(&mut p.bytes, 12, grid_m);
        w32(&mut p.bytes, 16, grid_n);
        w32(&mut p.bytes, 20, 1); // batch

        // swizzle_log_tile
        w32(&mut p.bytes, 24, compute_swizzle_log(grid_m, grid_n) as i32);

        // params_A (4x i64)
        let lda64 = lda as i64;
        for i in 0..4 {
            w64(
                &mut p.bytes,
                32 + i * 8,
                lda64 * consts.a_slope[i] + consts.a_intercept[i],
            );
        }
        // ref_A
        wu64(&mut p.bytes, 64, a_ptr);
        w64(&mut p.bytes, 72, lda64);

        // params_B (4x i64)
        let ldb64 = ldb as i64;
        for i in 0..4 {
            w64(
                &mut p.bytes,
                80 + i * 8,
                ldb64 * consts.b_slope[i] + consts.b_intercept[i],
            );
        }
        // ref_B
        wu64(&mut p.bytes, 112, b_ptr);
        w64(&mut p.bytes, 120, ldb64);

        // params_C (8x i64)
        let ldc64 = ldc as i64;
        for i in 0..8 {
            w64(&mut p.bytes, 128 + i * 8, ldc64 * consts.cd_slope[i]);
        }
        // ref_C
        wu64(&mut p.bytes, 192, c_ptr);
        w64(&mut p.bytes, 200, ldc64);

        // params_D (8x i64) — same constants as C
        let ldd64 = ldd as i64;
        for i in 0..8 {
            w64(&mut p.bytes, 208 + i * 8, ldd64 * consts.cd_slope[i]);
        }
        // ref_D
        wu64(&mut p.bytes, 272, d_ptr);
        w64(&mut p.bytes, 280, ldd64);

        // output_op: alpha=1.0, beta=0.0
        wf32(&mut p.bytes, 288, 1.0);
        wf32(&mut p.bytes, 292, 0.0);

        // gemm_k_size
        w32(&mut p.bytes, 336, k as i32);

        p
    }

    /// Set alpha and beta for the epilogue (default: alpha=1.0, beta=0.0).
    pub fn set_epilogue(&mut self, alpha: f32, beta: f32) {
        wf32(&mut self.bytes, 288, alpha);
        wf32(&mut self.bytes, 292, beta);
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

/// Find the CUTLASS entry name in PTX.
fn find_entry_name(ptx: &str) -> Result<String, String> {
    for line in ptx.lines() {
        let t = line.trim();
        if t.contains(".entry")
            && t.contains('(')
            && let Some(start) = t.find("_ZN7cutlass").or_else(|| t.find("fused_"))
        {
            let end = t.find('(').unwrap_or(t.len());
            return Ok(t[start..end].trim().to_string());
        }
    }
    Err("no entry found in PTX".into())
}

/// Compute swizzle log for GemmIdentityThreadblockSwizzle<4>.
fn compute_swizzle_log(grid_m: i32, grid_n: i32) -> u32 {
    let max_dim = grid_m.max(grid_n);
    let min_dim = grid_m.min(grid_n).max(1);
    let ratio = (max_dim / min_dim).min(4);
    if ratio >= 4 {
        2
    } else if ratio >= 2 {
        1
    } else {
        0
    }
}
