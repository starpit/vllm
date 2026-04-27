// SPDX-License-Identifier: Apache-2.0
//! Target profiles — hardware metadata the cost model consumes.
//!
//! Profile data lives in the `ferrite-cuda-targets` crate as `pub
//! const ProfileDef` values; this module bridges from those to the
//! `TargetProfile` shape `impl_lib.rs` consumes (`CostTable` parsed
//! from the embedded CSV, plus the analytic spec fields).

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use ferrite_cuda_targets::ProfileDef;

/// Empirical GPU cost table: `(kernel_name, M, N, K) -> cost_us`.
///
/// Populated from `target_profiles/cost_<profile>.csv` when present.
/// Keyed by `kernel` column value (`cublas`, `cutlass_128x128_s4`,
/// `cutlass_gemv`, …) to support both the cuBLAS reference line and
/// each cutlass tile variant.
///
/// `kernel_set` is a denormalized cache of the distinct kernel names
/// seen in `entries` — populated lazily on first access. Callers like
/// `Implementation::target_compatible` hit this in their inner loop
/// (per tile × per impl × per workload), so a hot-path `contains()`
/// must be O(1), not O(entries × clone_each_String).
#[derive(Clone, Debug, Default)]
pub struct CostTable {
    entries: HashMap<(String, u32, u32, u32), f64>,
    kernel_set: HashSet<String>,
}

impl CostTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn insert(&mut self, kernel: impl Into<String>, m: u32, n: u32, k: u32, cost_us: f64) {
        let kernel = kernel.into();
        self.kernel_set.insert(kernel.clone());
        self.entries.insert((kernel, m, n, k), cost_us);
    }

    pub fn get(&self, kernel: &str, m: u32, n: u32, k: u32) -> Option<f64> {
        // Use a borrowed-key tuple to skip the per-call String allocation.
        // HashMap's Borrow impl on tuples doesn't quite let us borrow the
        // String directly, so we still allocate here — but this is the
        // cold path (called once the candidate is being evaluated, not
        // per-impl-per-tile-per-workload).
        self.entries.get(&(kernel.to_string(), m, n, k)).copied()
    }

    /// O(1) membership check on kernel names. Used by hot-path
    /// `target_compatible` to gate impls without scanning the cost
    /// table per call.
    pub fn has_kernel(&self, kernel: &str) -> bool {
        self.kernel_set.contains(kernel)
    }

    /// Every distinct kernel name observed in the CSV. Useful for
    /// debugging and tests; NOT for hot paths — use [`has_kernel`] for
    /// existence checks.
    pub fn kernel_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.kernel_set.iter().cloned().collect();
        names.sort();
        names
    }
}

/// Per-kernel roofline coefficients fit from the CSV rows.
///
/// The CSV is a sparse grid; without extrapolation, every (kernel, M,
/// N, K) shape outside the grid falls back to `UNCALIBRATED_COST_US`,
/// which auto-loses the solver's cost duel. That makes coverage gaps
/// (e.g. lm_head at vocab-N) silent correctness bugs in impl
/// selection.
///
/// Instead, fit a 2-coefficient roofline per kernel:
///   `t_us = max(flops / (peak_tflops · eff_tflops · 1e6),
///              bytes / (peak_bw_gbps · eff_bw  · 1e3))`
/// where `eff_tflops` and `eff_bw` are the median achieved fraction
/// of the GPU's peaks across rows that fall in the matching regime
/// (compute-bound vs memory-bound, decided by per-row peak ratio).
///
/// Defaults to 1.0 (kernel hits peak — optimistic) when no rows in a
/// regime exist. The fallback only fires on grid-miss; exact lookups
/// always win and bypass it.
#[derive(Clone, Copy, Debug)]
pub struct KernelFit {
    pub eff_tflops: f64,
    pub eff_bw: f64,
}

impl Default for KernelFit {
    fn default() -> Self {
        Self {
            eff_tflops: 1.0,
            eff_bw: 1.0,
        }
    }
}

/// Bytes touched by a (M, N, K) GEMM in bf16: A (M×K) + B (K×N) + C
/// (M×N), 2 bytes per element. Elementwise rows in the CSV use K=0;
/// the formula still gives sensible memory traffic
/// (`2·(M·N + 0 + 0) = 2·M·N`) so the same predictor covers them.
fn gemm_bytes(m: u32, n: u32, k: u32) -> f64 {
    let m = m as f64;
    let n = n as f64;
    let k = k as f64;
    2.0 * (m * k + k * n + m * n)
}

fn gemm_flops(m: u32, n: u32, k: u32) -> f64 {
    2.0 * (m as f64) * (n as f64) * (k as f64)
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(v[v.len() / 2])
}

/// Fit one `KernelFit` per kernel name from the CSV rows. Each row
/// classifies as compute-bound or memory-bound by which side of the
/// roofline (`flops / peak_tflops` vs `bytes / peak_bw_gbps`) is
/// larger; the row's measured `eff_tflops` (resp. `eff_bw`) is the
/// achieved fraction of that peak.
fn fit_kernels(
    table: &CostTable,
    peak_tflops: f64,
    peak_bw_gbps: f64,
) -> HashMap<String, KernelFit> {
    let mut compute_samples: HashMap<String, Vec<f64>> = HashMap::new();
    let mut bw_samples: HashMap<String, Vec<f64>> = HashMap::new();
    for ((kernel, m, n, k), &t_us) in &table.entries {
        if t_us <= 0.0 || !t_us.is_finite() {
            continue;
        }
        let flops = gemm_flops(*m, *n, *k);
        let bytes = gemm_bytes(*m, *n, *k);
        let t_compute_peak_us = flops / (peak_tflops * 1.0e6);
        let t_mem_peak_us = bytes / (peak_bw_gbps * 1.0e3);
        if t_compute_peak_us >= t_mem_peak_us {
            // Compute-bound row: eff_tflops = ideal / measured.
            compute_samples
                .entry(kernel.clone())
                .or_default()
                .push(t_compute_peak_us / t_us);
        } else {
            bw_samples
                .entry(kernel.clone())
                .or_default()
                .push(t_mem_peak_us / t_us);
        }
    }
    let mut out: HashMap<String, KernelFit> = HashMap::new();
    for kernel in &table.kernel_set {
        let eff_tflops = compute_samples
            .remove(kernel)
            .and_then(median)
            .unwrap_or(1.0);
        let eff_bw = bw_samples.remove(kernel).and_then(median).unwrap_or(1.0);
        out.insert(kernel.clone(), KernelFit { eff_tflops, eff_bw });
    }
    out
}

/// Hardware characteristics a cost model uses to estimate kernel
/// timing. All units are explicit. Add fields here as the cost
/// model learns to use more.
#[derive(Clone, Debug)]
pub struct TargetProfile {
    pub name: String,
    pub source_path: PathBuf,
    /// sm_XX — 89 = Ada, 90 = Hopper.
    pub compute_capability: u32,
    /// Number of streaming multiprocessors.
    pub num_sms: u32,
    /// Peak FP16 tensor-core throughput, teraflops.
    pub peak_tflops_fp16: f64,
    /// Global memory bandwidth, gigabytes per second.
    pub memory_bandwidth_gbps: f64,
    /// Shared memory per SM, kilobytes.
    pub shared_memory_per_sm_kb: u32,
    /// Empirical cost table loaded from `cost_<name>.csv` alongside
    /// the JSON, when present. Populated with the GPU-swept
    /// measurements from prior ferrite (cublas + every cutlass tile
    /// variant across a grid of `(M, N, K)`). Empty when no CSV is
    /// present — cost impls fall back to their analytic formula.
    pub cost_table: CostTable,
    /// Per-kernel roofline coefficients fit from `cost_table` rows
    /// against this profile's peaks. Used by [`cost_us_for`] to
    /// extrapolate on grid-miss instead of falling through to
    /// `UNCALIBRATED_COST_US`.
    pub kernel_fits: HashMap<String, KernelFit>,
}

impl TargetProfile {
    /// Look up an empirical cost. Tries exact CSV lookup first; on
    /// miss, returns a roofline-extrapolated cost derived from this
    /// kernel's `KernelFit`. Returns `None` only when the kernel has
    /// no rows in the CSV at all (i.e. `kernel_fits` doesn't know it).
    pub fn cost_us_for(&self, kernel: &str, m: u32, n: u32, k: u32) -> Option<f64> {
        if let Some(t) = self.cost_table.get(kernel, m, n, k) {
            return Some(t);
        }
        self.predict(kernel, m, n, k)
    }

    /// Roofline prediction for `kernel` at `(m, n, k)`. `None` when
    /// the kernel name has no fit (i.e. zero CSV rows). Always returns
    /// `Some` for known kernels — defaults to peak (eff=1.0) when a
    /// regime has no calibration samples.
    ///
    /// Each side is clamped at the hardware roofline (peak compute /
    /// peak bandwidth). This matters because the calibration sweep
    /// runs warmup+iters back-to-back and small shapes serve from L2
    /// — the fit then sees `eff_bw > 1.0` (faster than DRAM). That's
    /// fine for cache-resident shapes (exact lookup wins anyway) but
    /// would extrapolate to physically impossible speeds on
    /// cache-overflowing shapes like lm_head. Clamp at peak fixes
    /// that asymmetrically: the predictor can be slower than peak
    /// (kernel inefficiency) but never faster.
    pub fn predict(&self, kernel: &str, m: u32, n: u32, k: u32) -> Option<f64> {
        let fit = self.kernel_fits.get(kernel)?;
        let flops = gemm_flops(m, n, k);
        let bytes = gemm_bytes(m, n, k);
        let t_compute_peak_us = flops / (self.peak_tflops_fp16 * 1.0e6);
        let t_mem_peak_us = bytes / (self.memory_bandwidth_gbps * 1.0e3);
        let t_compute_us =
            (flops / (self.peak_tflops_fp16 * fit.eff_tflops * 1.0e6)).max(t_compute_peak_us);
        let t_mem_us =
            (bytes / (self.memory_bandwidth_gbps * fit.eff_bw * 1.0e3)).max(t_mem_peak_us);
        Some(t_compute_us.max(t_mem_us))
    }
}

/// Build a `TargetProfile` from a `ferrite-cuda-targets` profile
/// const, parsing the embedded CSV bytes into a `CostTable`. The
/// proc-macro calls this once per `#[forward]` invocation after
/// resolving the active GPU (`ferrite_cuda_targets::detect()`).
pub fn from_profile_def(def: &ProfileDef) -> TargetProfile {
    let cost_table = parse_cost_csv(def.cost_csv);
    let kernel_fits = fit_kernels(&cost_table, def.peak_tflops_fp16, def.memory_bandwidth_gbps);
    TargetProfile {
        name: def.name.to_string(),
        source_path: PathBuf::new(),
        compute_capability: def.compute_capability,
        num_sms: def.num_sms,
        peak_tflops_fp16: def.peak_tflops_fp16,
        memory_bandwidth_gbps: def.memory_bandwidth_gbps,
        shared_memory_per_sm_kb: def.shared_memory_per_sm_kb,
        cost_table,
        kernel_fits,
    }
}

/// Parse the embedded `cost_<gpu>.csv` bytes into a `CostTable`.
/// Format: `kernel,M,N,K,cost_us` rows. Comment lines (`#`-prefixed)
/// and the header are skipped; malformed rows are silently dropped
/// rather than aborting the whole parse.
pub fn parse_cost_csv(csv: &str) -> CostTable {
    let mut table = CostTable::new();
    for raw in csv.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with("kernel,") {
            continue;
        }
        let mut parts = line.splitn(5, ',');
        let (Some(kernel), Some(m), Some(n), Some(k), Some(cost)) = (
            parts.next(),
            parts.next(),
            parts.next(),
            parts.next(),
            parts.next(),
        ) else {
            continue;
        };
        let (Ok(m), Ok(n), Ok(k), Ok(cost_us)) = (
            m.trim().parse::<u32>(),
            n.trim().parse::<u32>(),
            k.trim().parse::<u32>(),
            cost.trim().parse::<f64>(),
        ) else {
            continue;
        };
        table.insert(kernel.trim(), m, n, k, cost_us);
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrite_cuda_targets::{H100_SM90, L4_SM89};

    #[test]
    fn from_profile_def_carries_spec_fields() {
        let l4 = from_profile_def(&L4_SM89);
        assert_eq!(l4.compute_capability, 89);
        assert_eq!(l4.num_sms, 58);
        assert!(l4.peak_tflops_fp16 > 100.0);

        let h100 = from_profile_def(&H100_SM90);
        assert_eq!(h100.compute_capability, 90);
        assert!(h100.peak_tflops_fp16 > l4.peak_tflops_fp16);
    }

    #[test]
    fn cost_csv_parses_into_table() {
        let l4 = from_profile_def(&L4_SM89);
        // gpu_cost_sweep produces thousands of rows across kernel ×
        // (M, N, K). The embedded CSV survives the parse.
        assert!(!l4.cost_table.is_empty(), "expected cost_l4_sm89 rows");
        assert!(l4.cost_table.len() > 1000);
        let kinds = l4.cost_table.kernel_names();
        assert!(kinds.iter().any(|k| k == "cublas"), "cublas baseline");
        assert!(
            kinds.iter().any(|k| k.starts_with("cutlass_")),
            "cutlass tile variants"
        );
        assert!(
            kinds.iter().any(|k| k == "cutlass_gemv"),
            "cutlass_gemv for M=1"
        );
    }

    #[test]
    fn cost_lookup_round_trips() {
        let l4 = from_profile_def(&L4_SM89);
        // The first data row in cost_l4_sm89.csv is
        // `cublas,1,2048,2048,9.7`. Use it as a canary.
        let c = l4.cost_us_for("cublas", 1, 2048, 2048);
        assert!(c.is_some(), "cublas 1x2048x2048 missing from table");
        let us = c.unwrap();
        assert!(us > 0.0 && us.is_finite());
    }

    #[test]
    fn predict_extrapolates_to_unswept_lm_head_shapes() {
        // Llama-3 8B lm_head at decode: M=1, N=128256, K=4096. Far
        // outside the L4 sweep grid (max gemv N is ~28k). Before the
        // roofline fallback this returned None and the impl auto-lost
        // to cuBLAS via `UNCALIBRATED_COST_US`. Now it must return a
        // finite, plausible cost.
        let l4 = from_profile_def(&L4_SM89);
        let t = l4.cost_us_for("cutlass_gemv", 1, 128256, 4096);
        let t = t.expect("cutlass_gemv should have a fit on L4");
        assert!(
            t.is_finite() && t > 0.0,
            "predicted cost must be finite-positive: {t}"
        );
        // Sanity bound: at L4's 300 GB/s with full bf16 weights
        // (128256·4096·2 = ~1.05 GB), pure-bandwidth lower bound is
        // ~3500 us. A reasonable prediction sits within 1× to 50× of
        // that — anything wildly outside means the fit broke.
        assert!(
            (1_000.0..=200_000.0).contains(&t),
            "lm_head decode cost wildly off: {t} us"
        );
    }

    #[test]
    fn predict_holds_out_a_dram_bound_row_within_factor() {
        // Hold out a DRAM-bound row whose bytes overflow L4's 48 MB L2:
        // (M=4096, N=8192, K=8192) ≈ 320 MB total, definitely streaming
        // from DRAM. Predictor + peak-clamp must reconstruct within 3×.
        //
        // Cache-resident shapes are intentionally NOT testable this way
        // — the fit reads them as super-peak (data served from L2) and
        // the clamp pessimistically returns peak-DRAM time, which is
        // the right answer for the lm_head extrapolation regime.
        let l4 = from_profile_def(&L4_SM89);
        let mut found = None;
        for k in ["cutlass_128x128_s4", "cutlass_128x128_s3", "cublas"] {
            let key = (k.to_string(), 4096u32, 8192u32, 8192u32);
            if l4.cost_table.entries.contains_key(&key) {
                found = Some((k, key));
                break;
            }
        }
        let (kname, target_key) = found.expect("no DRAM-bound 4096x8192x8192 row in L4 sweep");
        let truth = l4.cost_table.entries[&target_key];
        let mut held_out = l4.cost_table.clone();
        held_out.entries.remove(&target_key);
        let fits = fit_kernels(&held_out, l4.peak_tflops_fp16, l4.memory_bandwidth_gbps);
        let synth = TargetProfile {
            cost_table: held_out,
            kernel_fits: fits,
            ..l4.clone()
        };
        let pred = synth.predict(kname, 4096, 8192, 8192).expect("predict");
        let ratio = pred / truth;
        assert!(
            (0.33..=3.0).contains(&ratio),
            "{kname} held-out prediction off: pred={pred} truth={truth} ratio={ratio}"
        );
    }

    #[test]
    fn parse_cost_csv_skips_garbage() {
        let csv = "\
            # comment line\n\
            kernel,M,N,K,cost_us\n\
            \n\
            cublas,1,2,3,4.5\n\
            malformed_row\n\
            cutlass_128x128_s3,8,16,32,7.0\n\
            cublas,not_a_number,2,3,4.5\n";
        let table = parse_cost_csv(csv);
        assert_eq!(table.len(), 2);
        assert_eq!(table.get("cublas", 1, 2, 3), Some(4.5));
        assert_eq!(table.get("cutlass_128x128_s3", 8, 16, 32), Some(7.0));
    }
}
