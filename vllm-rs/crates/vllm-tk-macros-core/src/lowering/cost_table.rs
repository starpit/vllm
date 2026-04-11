// SPDX-License-Identifier: Apache-2.0
//! GPU cost table: measured kernel latencies across a (M, N, K) grid.
//!
//! ## Adding a new GPU
//!
//! 1. Run the microbench sweep on the target GPU:
//!    ```bash
//!    CUDA_PATH=/usr/local/cuda-12.9 cargo test -p vllm-tk-test-harness \
//!      --features cuda --test scheduled_megakernel_test \
//!      gpu_cost_sweep -- --ignored --nocapture
//!    ```
//!    This prints CSV to stdout. Redirect to a file:
//!    ```bash
//!    ... 2>/dev/null > crates/vllm-tk-macros-core/data/cost_l4_sm89.csv
//!    ```
//!
//! 2. The CSV has columns: `kernel,M,N,K,cost_us`
//!    - kernel: `cublas`, `cutlass_64x64`, `cutlass_128x128`
//!    - M: num_tokens (1, 2, 4, 8, ..., 4096)
//!    - N: output dim
//!    - K: input dim
//!    - cost_us: median wall-clock microseconds
//!
//! 3. The proc macro reads this CSV at compile time via `include_str!`.
//!    The solver interpolates from the grid for any (M, N, K) the model needs.

use std::collections::HashMap;

/// One measured data point.
#[derive(Clone, Debug)]
struct CostPoint3D {
    m: u32,
    n: u32,
    k: u32,
    cost_us: f64,
}

/// Kernel family identifier — matches the CSV kernel column exactly.
/// Examples: `"cublas"`, `"cutlass_64x64_s4"`, `"cutlass_128x128_s3"`.
pub type KernelFamily = String;

/// Cost table loaded from CSV. Provides `lookup(kernel, m, n, k) -> f64`.
#[derive(Clone, Debug)]
pub struct GpuCostGrid {
    pub gpu_name: String,
    /// Per-kernel measured data, keyed by kernel name.
    tables: HashMap<KernelFamily, Vec<CostPoint3D>>,
    /// Sorted unique M values for interpolation.
    m_grid: Vec<u32>,
}

impl GpuCostGrid {
    /// Parse a CSV string (as produced by the sweep test).
    pub fn from_csv(gpu_name: &str, csv: &str) -> Self {
        let mut tables: HashMap<KernelFamily, Vec<CostPoint3D>> = HashMap::new();
        let mut m_set = std::collections::BTreeSet::new();

        for line in csv.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("kernel") {
                continue; // skip header/comments
            }
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() < 5 {
                continue;
            }
            let kernel = cols[0].trim().to_string();
            let m: u32 = cols[1].trim().parse().unwrap_or(0);
            let n: u32 = cols[2].trim().parse().unwrap_or(0);
            let k: u32 = cols[3].trim().parse().unwrap_or(0);
            let cost_us: f64 = cols[4].trim().parse().unwrap_or(0.0);
            if m == 0 || n == 0 || k == 0 || cost_us <= 0.0 {
                continue;
            }
            m_set.insert(m);
            tables
                .entry(kernel)
                .or_default()
                .push(CostPoint3D { m, n, k, cost_us });
        }

        GpuCostGrid {
            gpu_name: gpu_name.to_string(),
            tables,
            m_grid: m_set.into_iter().collect(),
        }
    }

    /// Look up cost for a (kernel, M, N, K) query.
    /// Finds the two nearest (N, K) pairs in the data and interpolates
    /// between their M-curves.
    pub fn lookup(&self, kernel: &str, m: u32, n: u32, k: u32) -> f64 {
        let points = match self.tables.get(kernel) {
            Some(p) => p,
            None => return self.roofline_fallback(m, n, k),
        };

        // Find exact (N, K) match first.
        let exact: Vec<_> = points.iter().filter(|p| p.n == n && p.k == k).collect();
        if !exact.is_empty() {
            return Self::interpolate_m(&exact, m);
        }

        // No exact match — find nearest (N, K) by Euclidean distance
        // and scale by the ratio of compute (M*N*K).
        let (nearest_n, nearest_k, nearest_cost) = self.nearest_nk(points, m, n, k);
        if nearest_cost > 0.0 {
            // Scale by compute ratio.
            let query_flops = m as f64 * n as f64 * k as f64;
            let nearest_flops = m as f64 * nearest_n as f64 * nearest_k as f64;
            return nearest_cost * (query_flops / nearest_flops.max(1.0));
        }

        self.roofline_fallback(m, n, k)
    }

    /// Interpolate along the M dimension from exact (N, K) matches.
    fn interpolate_m(points: &[&CostPoint3D], m: u32) -> f64 {
        if points.len() == 1 {
            return points[0].cost_us;
        }
        // Sort by M.
        let mut sorted: Vec<_> = points.iter().map(|p| (p.m, p.cost_us)).collect();
        sorted.sort_by_key(|(pm, _)| *pm);

        if m <= sorted[0].0 {
            return sorted[0].1;
        }
        let last = sorted.len() - 1;
        if m >= sorted[last].0 {
            // Linear extrapolation from last two.
            if last == 0 {
                return sorted[0].1;
            }
            let (m1, c1) = sorted[last - 1];
            let (m2, c2) = sorted[last];
            let slope = (c2 - c1) / (m2 - m1) as f64;
            return c2 + slope * (m - m2) as f64;
        }
        let idx = sorted.partition_point(|(pm, _)| *pm <= m);
        let (m0, c0) = sorted[idx - 1];
        let (m1, c1) = sorted[idx];
        let t = (m - m0) as f64 / (m1 - m0) as f64;
        c0 + t * (c1 - c0)
    }

    /// Find nearest (N, K) match and return its interpolated M cost.
    fn nearest_nk(&self, points: &[CostPoint3D], m: u32, n: u32, k: u32) -> (u32, u32, f64) {
        // Collect unique (N, K) pairs.
        let mut nk_set: HashMap<(u32, u32), Vec<&CostPoint3D>> = HashMap::new();
        for p in points {
            nk_set.entry((p.n, p.k)).or_default().push(p);
        }

        let mut best_dist = f64::MAX;
        let mut best = (0u32, 0u32, 0.0f64);
        for ((pn, pk), pts) in &nk_set {
            let dist = ((n as f64 - *pn as f64).powi(2) + (k as f64 - *pk as f64).powi(2)).sqrt();
            if dist < best_dist {
                best_dist = dist;
                let pts_refs: Vec<_> = pts.to_vec();
                best = (*pn, *pk, Self::interpolate_m(&pts_refs, m));
            }
        }
        best
    }

    fn roofline_fallback(&self, m: u32, n: u32, k: u32) -> f64 {
        // Simple roofline: max(BW-bound, compute-bound) + launch overhead.
        let bw_us = (n as f64 * k as f64 * 2.0) / 300_000.0; // ~300 GB/s
        let compute_us = (2.0 * m as f64 * n as f64 * k as f64) / 85_000_000.0; // ~85 TFLOPS bf16
        5.0 + bw_us.max(compute_us)
    }

    /// Whether we have any measured data for this kernel.
    pub fn has_data(&self, kernel: &str) -> bool {
        self.tables.contains_key(kernel)
    }
}

// ── Built-in CSV data ───────────────────────────────────────────

/// Load the L4 sm_89 cost grid from the built-in CSV.
/// Returns None if the CSV doesn't exist yet (sweep not run).
pub fn load_l4_sm89() -> Option<GpuCostGrid> {
    // The CSV is included at compile time from the data/ directory.
    // If the file doesn't exist, the build fails — which is intentional:
    // you must run the sweep before building with the solver.
    let csv = include_str!("../../data/cost_l4_sm89.csv");
    Some(GpuCostGrid::from_csv("L4 sm_89", csv))
}

/// Load the L40S sm_89 cost grid from the built-in CSV.
pub fn load_l40s_sm89() -> Option<GpuCostGrid> {
    let csv = include_str!("../../data/cost_l40s_sm89.csv");
    Some(GpuCostGrid::from_csv("L40S sm_89", csv))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CSV: &str = "\
kernel,M,N,K,cost_us
cublas,1,3072,2048,13.5
cublas,32,3072,2048,17.6
cublas,1024,3072,2048,156.5
cublas,1,8192,2048,27.2
cublas,32,8192,2048,39.0
cublas,1024,8192,2048,392.5
cutlass_64x64,1,8192,2048,29.5
cutlass_64x64,32,8192,2048,30.5
cutlass_64x64,1024,8192,2048,621.2
";

    #[test]
    fn parse_csv() {
        let grid = GpuCostGrid::from_csv("test", SAMPLE_CSV);
        assert!(grid.has_data("cublas"));
        assert!(grid.has_data("cutlass_64x64"));
        assert!(!grid.has_data("cutlass_128x128"));
    }

    #[test]
    fn exact_lookup() {
        let grid = GpuCostGrid::from_csv("test", SAMPLE_CSV);
        let cost = grid.lookup("cublas", 32, 3072, 2048);
        assert!((cost - 17.6).abs() < 0.1);
    }

    #[test]
    fn interpolated_lookup() {
        let grid = GpuCostGrid::from_csv("test", SAMPLE_CSV);
        // M=16 between M=1 (13.5) and M=32 (17.6) for (N=3072, K=2048).
        let cost = grid.lookup("cublas", 16, 3072, 2048);
        assert!(cost > 13.5 && cost < 17.6, "interpolated cost {cost}");
    }

    #[test]
    fn nearest_nk_fallback() {
        let grid = GpuCostGrid::from_csv("test", SAMPLE_CSV);
        // Query (N=4096, K=2048) — not in the data. Should find nearest
        // and scale by compute ratio.
        let cost = grid.lookup("cublas", 32, 4096, 2048);
        assert!(cost > 0.0, "should return positive cost");
    }
}
