// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal target profiles for Apple Silicon devices.
//!
//! Provides device-specific parameters and cost models for M1, M2, M3, and M4
//! chips to enable optimal kernel selection in ferrite's solver.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Apple Silicon architecture generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AppleSiliconGen {
    M1,
    M2,
    M3,
    M4,
}

/// Metal device profile containing hardware specs and cost models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalTargetProfile {
    /// Architecture generation (M1/M2/M3/M4)
    pub generation: AppleSiliconGen,

    /// Number of GPU cores
    pub gpu_cores: u32,

    /// Peak TFLOPS for FP16 operations
    pub peak_tflops_fp16: f64,

    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f64,

    /// Unified memory size in GB
    pub unified_memory_gb: u32,

    /// Maximum threadgroup memory in bytes (32KB for all Apple Silicon)
    pub threadgroup_memory_bytes: u32,

    /// Maximum threads per threadgroup
    pub max_threads_per_threadgroup: u32,

    /// Cost table: kernel_name -> (M, N, K) -> microseconds
    /// Empty initially, populated by microbenchmarks
    pub cost_table: BTreeMap<String, Vec<CostEntry>>,
}

/// Single cost measurement for a specific (M, N, K) shape
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEntry {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub cost_us: f64,
}

impl MetalTargetProfile {
    /// Look up cost for a specific kernel and shape
    pub fn cost_us_for(&self, kernel: &str, m: u32, n: u32, k: u32) -> Option<f64> {
        let entries = self.cost_table.get(kernel)?;

        // Exact match
        for entry in entries {
            if entry.m == m && entry.n == n && entry.k == k {
                return Some(entry.cost_us);
            }
        }

        // TODO: Linear interpolation for unmeasured shapes
        None
    }

    /// Load cost table from CSV file
    ///
    /// CSV format:
    /// ```
    /// kernel,M,N,K,cost_us
    /// metal_rmsnorm_f16,1,2048,0,170.78
    /// ```
    pub fn load_costs_from_csv(&mut self, csv_content: &str) -> Result<(), String> {
        for line in csv_content.lines() {
            // Skip comments and header
            if line.starts_with('#') || line.starts_with("kernel,") {
                continue;
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() != 5 {
                continue; // Skip malformed lines
            }

            let kernel = parts[0].to_string();
            let m: u32 = parts[1].parse().map_err(|e| format!("Invalid M: {}", e))?;
            let n: u32 = parts[2].parse().map_err(|e| format!("Invalid N: {}", e))?;
            let k: u32 = parts[3].parse().map_err(|e| format!("Invalid K: {}", e))?;
            let cost_us: f64 = parts[4]
                .parse()
                .map_err(|e| format!("Invalid cost: {}", e))?;

            let entry = CostEntry { m, n, k, cost_us };
            self.cost_table.entry(kernel).or_default().push(entry);
        }

        Ok(())
    }
}

/// M1 Max device profile (32 GPU cores) with measured costs
pub fn m1_max_with_costs() -> MetalTargetProfile {
    let mut profile = MetalTargetProfile {
        generation: AppleSiliconGen::M1,
        gpu_cores: 32,
        peak_tflops_fp16: 10.4,       // M1 Max has 4x the GPU cores of base M1
        memory_bandwidth_gbps: 400.0, // M1 Max has much higher bandwidth
        unified_memory_gb: 64,
        threadgroup_memory_bytes: 32768,
        max_threads_per_threadgroup: 1024,
        cost_table: BTreeMap::new(),
    };

    // Load measured costs from embedded CSV
    let csv_data = include_str!("../profiles/cost_m1_max.csv");
    profile
        .load_costs_from_csv(csv_data)
        .expect("Failed to load M1 Max cost data");

    profile
}

/// M1 device profile (base model, 8 GPU cores)
pub const M1_8CORE: MetalTargetProfile = MetalTargetProfile {
    generation: AppleSiliconGen::M1,
    gpu_cores: 8,
    peak_tflops_fp16: 2.6,
    memory_bandwidth_gbps: 68.25,
    unified_memory_gb: 16,
    threadgroup_memory_bytes: 32768,
    max_threads_per_threadgroup: 1024,
    cost_table: BTreeMap::new(),
};

/// M2 device profile (10 GPU cores)
pub const M2_10CORE: MetalTargetProfile = MetalTargetProfile {
    generation: AppleSiliconGen::M2,
    gpu_cores: 10,
    peak_tflops_fp16: 3.6,
    memory_bandwidth_gbps: 100.0,
    unified_memory_gb: 24,
    threadgroup_memory_bytes: 32768,
    max_threads_per_threadgroup: 1024,
    cost_table: BTreeMap::new(),
};

/// M3 device profile (base model, 10 GPU cores)
pub const M3_10CORE: MetalTargetProfile = MetalTargetProfile {
    generation: AppleSiliconGen::M3,
    gpu_cores: 10,
    peak_tflops_fp16: 4.0,
    memory_bandwidth_gbps: 100.0,
    unified_memory_gb: 24,
    threadgroup_memory_bytes: 32768,
    max_threads_per_threadgroup: 1024,
    cost_table: BTreeMap::new(),
};

/// M4 device profile (10 GPU cores)
pub const M4_10CORE: MetalTargetProfile = MetalTargetProfile {
    generation: AppleSiliconGen::M4,
    gpu_cores: 10,
    peak_tflops_fp16: 4.5,
    memory_bandwidth_gbps: 120.0,
    unified_memory_gb: 24,
    threadgroup_memory_bytes: 32768,
    max_threads_per_threadgroup: 1024,
    cost_table: BTreeMap::new(),
};

/// M4 device profile with measured costs loaded from
/// `profiles/cost_m4.csv` (regenerate via
/// `cargo run -p ferrite-metal-cost-sweep --release > profiles/cost_m4.csv`).
/// Used by `detect_device()` when the runtime chip identifies as M4 so
/// the solver's per-impl `cost_us` can consult empirical rows instead
/// of falling back to the analytical roofline.
pub fn m4_with_costs() -> MetalTargetProfile {
    let mut profile = M4_10CORE.clone();
    let csv = include_str!("../profiles/cost_m4.csv");
    profile
        .load_costs_from_csv(csv)
        .expect("Failed to load M4 cost data");
    profile
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_constants() {
        assert_eq!(M1_8CORE.generation, AppleSiliconGen::M1);
        assert_eq!(M1_8CORE.gpu_cores, 8);
        assert_eq!(M2_10CORE.gpu_cores, 10);
        assert!(M3_10CORE.peak_tflops_fp16 > M2_10CORE.peak_tflops_fp16);
    }

    #[test]
    fn test_m1_max_loads_costs() {
        let profile = m1_max_with_costs();

        // Verify costs were loaded
        assert!(
            !profile.cost_table.is_empty(),
            "Cost table should not be empty"
        );

        // Check for specific kernels
        assert!(profile.cost_table.contains_key("metal_rmsnorm_f16"));
        assert!(profile.cost_table.contains_key("metal_rmsnorm_bf16"));

        // Verify we can look up a cost
        let cost = profile.cost_us_for("metal_rmsnorm_f16", 1, 2048, 0);
        assert!(cost.is_some(), "Should find cost for (1, 2048, 0)");
        assert!(cost.unwrap() > 0.0, "Cost should be positive");
    }

    #[test]
    fn test_cost_lookup() {
        let mut profile = M1_8CORE.clone();

        // Add a test entry
        profile.cost_table.insert(
            "test_kernel".to_string(),
            vec![CostEntry {
                m: 128,
                n: 4096,
                k: 0,
                cost_us: 42.5,
            }],
        );

        // Exact match should work
        assert_eq!(profile.cost_us_for("test_kernel", 128, 4096, 0), Some(42.5));

        // Non-existent shape should return None
        assert_eq!(profile.cost_us_for("test_kernel", 256, 4096, 0), None);

        // Non-existent kernel should return None
        assert_eq!(profile.cost_us_for("nonexistent", 128, 4096, 0), None);
    }
}
