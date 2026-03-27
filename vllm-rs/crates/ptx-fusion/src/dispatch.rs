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
