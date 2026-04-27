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

/// Per-kernel cost-vs-shape coefficients fit from the CSV rows.
///
/// Models kernel time as a two-line roofline of two independent
/// linear regressions — one over flops (compute-bound regime),
/// one over bytes (memory-bound regime):
///
/// ```text
///   t_us = max(γ · flops + δ,   α · bytes + β)
/// ```
///
/// Why a 2-parameter fit and not the previous single-`eff_bw` scale
/// of peak: kernel cost has TWO physical components — a fixed
/// overhead (kernel launch + on-chip cache-served traffic, captured
/// in `β`/`δ`) and a per-byte/per-flop saturation rate (`α`/`γ`).
/// The earlier `eff_bw = peak_us / measured_us` median conflated
/// cache-resident rows (super-peak) with DRAM-bound rows (real
/// efficiency), and a hardware-peak clamp at predict time hid the
/// bias in the cache regime but ALSO collapsed every kernel to the
/// same prediction at memory-bound vocab-N shapes — leaving lm_head
/// prefill picks dominated by calibration-bias noise rather than
/// real DRAM behavior.
///
/// The linear fit captures both regimes naturally: at small shapes
/// `β`/`δ` dominate (cache+overhead); at large shapes `α·bytes` /
/// `γ·flops` dominate (real saturation rate). No magic L2-size
/// threshold required.
///
/// Constraints applied at fit time, not predict time:
///   - slopes (`α`, `γ`) clamped to ≥ peak (kernel can't beat
///     hardware ceiling at saturation; values below peak indicate
///     noise / cache-resident rows polluting the slope)
///   - intercepts (`β`, `δ`) clamped to ≥ 0 (negative overhead is
///     unphysical)
///   - degenerate samples (0–1 rows in a regime, all-equal x) fall
///     back to peak slope + zero intercept
#[derive(Clone, Copy, Debug)]
pub struct KernelFit {
    /// Per-byte saturation time at kernel-achieved DRAM efficiency,
    /// in microseconds. `1/α` is effective DRAM bandwidth in
    /// bytes/μs. Clamped ≥ peak DRAM time per byte.
    pub alpha_us_per_byte: f64,
    /// Memory-side fixed cost (kernel launch overhead + on-chip /
    /// cache-served traffic), microseconds. Independent of bytes.
    pub beta_us: f64,
    /// Per-flop saturation time at kernel-achieved compute
    /// efficiency, microseconds. Clamped ≥ peak compute time per
    /// flop.
    pub gamma_us_per_flop: f64,
    /// Compute-side fixed cost, microseconds. Independent of flops.
    pub delta_us: f64,
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

/// Constrained least-squares fit `t = slope·x + intercept` over the
/// sample set, with `slope ≥ min_slope` and `intercept ≥ 0`. Used to
/// fit per-byte (memory regime) and per-flop (compute regime) cost
/// curves with the physical constraints baked in:
///   - slope can't be smaller than the hardware ceiling (`1/peak_bw`
///     per byte, `1/peak_tflops` per flop) — a kernel beating peak
///     would be unphysical and is noise from cache-resident rows
///   - intercept can't be negative — overhead is non-negative
///
/// Strategy: unconstrained OLS first; if either coefficient lies
/// outside the feasible region, project onto the boundary by fixing
/// the violating term and refitting the other. Two single-parameter
/// fits in the worst case — exact for the constrained QP at the
/// face we'd land on.
fn fit_constrained(samples: &[(f64, f64)], min_slope: f64) -> (f64, f64) {
    if samples.is_empty() {
        return (min_slope, 0.0);
    }
    let n = samples.len() as f64;
    let sum_x: f64 = samples.iter().map(|s| s.0).sum();
    let sum_t: f64 = samples.iter().map(|s| s.1).sum();
    if samples.len() == 1 {
        let intercept = (sum_t - min_slope * sum_x).max(0.0);
        return (min_slope, intercept);
    }
    let sum_xt: f64 = samples.iter().map(|s| s.0 * s.1).sum();
    let sum_xx: f64 = samples.iter().map(|s| s.0 * s.0).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    let mut slope;
    let mut intercept;
    if denom < 1e-30 {
        // All x identical — slope is unidentifiable. Fall back to
        // peak slope + best-fit intercept.
        slope = min_slope;
        intercept = ((sum_t - min_slope * sum_x) / n).max(0.0);
        return (slope, intercept);
    }
    slope = (n * sum_xt - sum_x * sum_t) / denom;
    intercept = (sum_t - slope * sum_x) / n;
    if slope < min_slope {
        // Fix slope at peak, fit intercept alone.
        slope = min_slope;
        intercept = (sum_t - min_slope * sum_x) / n;
    }
    if intercept < 0.0 {
        // Fix intercept at zero, fit slope alone (through origin).
        intercept = 0.0;
        slope = if sum_xx > 1e-30 {
            (sum_xt / sum_xx).max(min_slope)
        } else {
            min_slope
        };
    }
    (slope, intercept)
}

/// Fit one `KernelFit` per kernel name from the CSV rows. Each row
/// classifies as compute-bound or memory-bound by which side of the
/// roofline (`flops / peak_tflops` vs `bytes / peak_bw_gbps`) is
/// larger; rows in each regime feed a constrained least-squares
/// regression of `t_us` against `flops` or `bytes` respectively.
fn fit_kernels(
    table: &CostTable,
    peak_tflops: f64,
    peak_bw_gbps: f64,
) -> HashMap<String, KernelFit> {
    // Hardware-ceiling slopes — kernel can't beat these at saturation.
    let min_gamma = 1.0 / (peak_tflops * 1.0e6); // μs per flop
    let min_alpha = 1.0 / (peak_bw_gbps * 1.0e3); // μs per byte

    let mut compute_rows: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
    let mut memory_rows: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
    for ((kernel, m, n, k), &t_us) in &table.entries {
        if t_us <= 0.0 || !t_us.is_finite() {
            continue;
        }
        let flops = gemm_flops(*m, *n, *k);
        let bytes = gemm_bytes(*m, *n, *k);
        let t_compute_peak_us = flops * min_gamma;
        let t_mem_peak_us = bytes * min_alpha;
        if t_compute_peak_us >= t_mem_peak_us {
            compute_rows
                .entry(kernel.clone())
                .or_default()
                .push((flops, t_us));
        } else {
            memory_rows
                .entry(kernel.clone())
                .or_default()
                .push((bytes, t_us));
        }
    }
    let mut out: HashMap<String, KernelFit> = HashMap::new();
    for kernel in &table.kernel_set {
        let (gamma_us_per_flop, delta_us) = compute_rows
            .remove(kernel)
            .map(|s| fit_constrained(&s, min_gamma))
            .unwrap_or((min_gamma, 0.0));
        let (alpha_us_per_byte, beta_us) = memory_rows
            .remove(kernel)
            .map(|s| fit_constrained(&s, min_alpha))
            .unwrap_or((min_alpha, 0.0));
        out.insert(
            kernel.clone(),
            KernelFit {
                alpha_us_per_byte,
                beta_us,
                gamma_us_per_flop,
                delta_us,
            },
        );
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

    /// Roofline prediction for `kernel` at `(m, n, k)`, using the
    /// kernel's `KernelFit`. `None` when the kernel name has no
    /// rows in the CSV at all.
    ///
    /// `t_us = max(γ·flops + δ,  α·bytes + β)` — pure
    /// linear-of-shape, no clamp needed because the fit was
    /// constrained at fit time (slopes ≥ peak hardware ceiling,
    /// intercepts ≥ 0).
    pub fn predict(&self, kernel: &str, m: u32, n: u32, k: u32) -> Option<f64> {
        let fit = self.kernel_fits.get(kernel)?;
        let flops = gemm_flops(m, n, k);
        let bytes = gemm_bytes(m, n, k);
        let t_compute_us = fit.gamma_us_per_flop * flops + fit.delta_us;
        let t_mem_us = fit.alpha_us_per_byte * bytes + fit.beta_us;
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
        // from DRAM. Predictor must reconstruct within 3× from the
        // remaining (α·bytes + β) fit.
        //
        // 3× is loose enough to absorb regression noise (small samples
        // per kernel can yield noisy slopes) but tight enough that a
        // unit-conversion bug or constraint-projection error would blow
        // through it.
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
    fn lm_head_prefill_predictions_are_physical() {
        // Llama-3 8B lm_head at small prefill: M=8, N=128256, K=4096.
        // Bytes ≈ 1.05 GB — outside the calibration grid, both cuBLAS
        // and CUTLASS variants predict via the linear fit. The
        // invariants this test pins (which the eff-scale predictor
        // could violate when its peak-clamp triggered):
        //
        //   1. Predictions are finite-positive and within 0.5×–10× of
        //      the pure peak-DRAM bound — physics, not collapsed to
        //      identical clamp values.
        //   2. CUTLASS variants spread out (different α, β per
        //      kernel) — the eff-scale predictor would force them all
        //      to the same peak-DRAM time at memory-bound regime.
        //
        // Notably this does NOT assert "CUTLASS beats cuBLAS." The
        // linear fit honestly says cuBLAS has a slightly lower
        // saturation slope at this regime — closing that gap is a
        // kernel-engineering job, not a cost-model job.
        let l4 = from_profile_def(&L4_SM89);
        let cublas_t = l4
            .cost_us_for("cublas", 8, 128256, 4096)
            .expect("cuBLAS fit");
        // Peak DRAM lower bound: bytes / peak_bw.
        let bytes = gemm_bytes(8, 128256, 4096);
        let t_peak_us = bytes / (l4.memory_bandwidth_gbps * 1.0e3);
        assert!(
            (0.5 * t_peak_us..=10.0 * t_peak_us).contains(&cublas_t),
            "cuBLAS prediction should be within 0.5×–10× of peak DRAM time \
             ({t_peak_us} us); got {cublas_t} us"
        );
        // Spread check: gather predictions across the M≥2-eligible
        // CUTLASS Gemm tile zoo. They should NOT all coincide (which
        // would indicate the predictor had collapsed to peak).
        let m2_eligible: Vec<&str> = l4
            .cost_table
            .kernel_set
            .iter()
            .map(|s| s.as_str())
            .filter(|k| {
                k.starts_with("cutlass_")
                    && !k.starts_with("cutlass_gemv")
                    && !k.contains("fused")
                    && !k.contains("bias")
                    && !k.ends_with("_add")
                    && !k.contains("_split")
                    && !k.contains("_sk")
            })
            .collect();
        let mut seen: Vec<f64> = m2_eligible
            .iter()
            .filter_map(|k| l4.cost_us_for(k, 8, 128256, 4096))
            .collect();
        seen.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert!(seen.len() >= 4, "expected several M≥2 CUTLASS predictions");
        let spread = seen.last().unwrap() - seen.first().unwrap();
        let pivot = seen.first().unwrap();
        assert!(
            spread / pivot > 0.001,
            "CUTLASS predictions at lm_head shape should differ between \
             tile variants (got spread={spread} on pivot={pivot}); if all \
             collapse to one value the predictor has lost per-kernel \
             distinguishability"
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
