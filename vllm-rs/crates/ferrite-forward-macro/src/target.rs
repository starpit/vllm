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
/// Three-parameter cost model fitted per-kernel from the measured CSV
/// rows. Form:
///
/// ```text
/// cost_us(M, N, K) ≈ mnk_coeff * (M*N*K)
///                   + surface_coeff * (M*N + N*K + M*K)
///                   + launch_us
/// ```
///
/// - `mnk_coeff` (units: µs/flop) captures compute throughput in the
///   compute-bound regime — proportional to 1 / TFLOPS.
/// - `surface_coeff` (units: µs/elem) captures memory bandwidth for
///   the A + B + C tensor traffic — proportional to 1 / (BW/elem_size).
/// - `launch_us` is the fixed per-kernel-launch overhead, plus any
///   constant-time setup (allocator hits, kernel-arg marshalling).
///
/// Used as the prediction fallback when an exact `(kernel, M, N, K)`
/// row isn't in the CSV. Without this, the cost-driven DP fell back
/// to roofline (structurally optimistic) for unseen shapes — every
/// new model architecture not in the sweep grid got bad tile picks.
///
/// Fit method: ordinary least squares on the linear model above,
/// closed-form via normal equations. Robust to missing regimes
/// (OLS handles overdetermined as long as the design matrix has full
/// rank; we degrade gracefully when it doesn't, see `fit`).
#[derive(Clone, Debug)]
pub struct KernelCostModel {
    pub mnk_coeff: f64,
    pub surface_coeff: f64,
    pub launch_us: f64,
    /// Median measured cost across all fit points — used as a fallback
    /// when the linear model would predict a non-physical value (e.g.
    /// negative cost from extrapolation outside the fit's support).
    pub median_us: f64,
    /// Bounding box of the fit data on each axis. Predictions outside
    /// this box (with a 2× safety margin) are unreliable — a global
    /// linear fit can't extrapolate the BW/compute-bound regime change
    /// that happens when one axis grows by an order of magnitude past
    /// the grid. At qwen2-0.5b's lm_head shape (M=1, N=151936, K=896),
    /// `cutlass_16x64_s3`'s fit predicted 74 µs while nsys measured
    /// 1160 µs — a 15× under-count that made the DP pick the tile over
    /// `cutlass_gemv` (correctly predicted at 802 µs because the gemv
    /// fit grid covers similar tall-skinny shapes). With out-of-box
    /// rejection the predictor returns `None` and the caller falls
    /// back to roofline (~850 µs at this shape, much closer).
    pub min_m: u32,
    pub max_m: u32,
    pub min_n: u32,
    pub max_n: u32,
    pub min_k: u32,
    pub max_k: u32,
}

impl KernelCostModel {
    /// Fit a 3-parameter linear model from `(M, N, K, cost_us)`
    /// points. Returns `None` if there are fewer than 3 distinct
    /// points (underdetermined) — caller should fall back to the
    /// kernel's median cost.
    fn fit(points: &[(u32, u32, u32, f64)]) -> Option<Self> {
        if points.len() < 3 {
            return None;
        }
        // Build design matrix X (Nx3) and target vector y (N).
        // X[i] = [M*N*K, M*N + N*K + M*K, 1.0]  (third col = launch term)
        // We solve (X^T X) β = X^T y for β = [mnk_coeff, surface_coeff, launch_us]
        // via the closed-form normal equations, computing the 3x3
        // inverse explicitly.
        let mut xtx = [[0.0f64; 3]; 3];
        let mut xty = [0.0f64; 3];
        let mut all_costs: Vec<f64> = Vec::with_capacity(points.len());
        let mut min_m = u32::MAX;
        let mut max_m = 0u32;
        let mut min_n = u32::MAX;
        let mut max_n = 0u32;
        let mut min_k = u32::MAX;
        let mut max_k = 0u32;

        for &(m, n, k, cost) in points {
            min_m = min_m.min(m);
            max_m = max_m.max(m);
            min_n = min_n.min(n);
            max_n = max_n.max(n);
            min_k = min_k.min(k);
            max_k = max_k.max(k);
            let m = m as f64;
            let n = n as f64;
            let k = k as f64;
            let f = [m * n * k, m * n + n * k + m * k, 1.0];
            for i in 0..3 {
                xty[i] += f[i] * cost;
                for j in 0..3 {
                    xtx[i][j] += f[i] * f[j];
                }
            }
            all_costs.push(cost);
        }

        // Solve 3x3 system via Cramer's rule.
        let det = det3(xtx);
        if det.abs() < 1e-30 {
            // Singular — measurements lie on a degenerate manifold
            // (e.g. all rows at the same M, or proportional shapes).
            return None;
        }

        let mut beta = [0.0f64; 3];
        for i in 0..3 {
            let mut col = xtx;
            for r in 0..3 {
                col[r][i] = xty[r];
            }
            beta[i] = det3(col) / det;
        }

        // Median for non-physical-prediction fallback.
        all_costs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_us = all_costs[all_costs.len() / 2];

        Some(KernelCostModel {
            mnk_coeff: beta[0],
            surface_coeff: beta[1],
            launch_us: beta[2],
            median_us,
            min_m,
            max_m,
            min_n,
            max_n,
            min_k,
            max_k,
        })
    }

    /// Predict cost at an unseen `(M, N, K)`. Returns `None` when the
    /// query is outside the fit's bounding box (with 2× safety margin
    /// per axis) — extrapolation past the sweep grid can mispredict by
    /// 15× (see field docs). The caller then falls back to roofline,
    /// which is structurally pessimistic but stays in the right
    /// order-of-magnitude.
    ///
    /// In-box queries clamp non-physical predictions (negative or
    /// absurd) to `median_us` so the DP doesn't see garbage values
    /// from interpolation between sparse points.
    fn predict(&self, m: u32, n: u32, k: u32) -> Option<f64> {
        // Out-of-support rejection: extrapolating past the fit grid is
        // unreliable for kernels whose cost regime changes (BW-bound
        // vs compute-bound) across the (M, N, K) space. 2× margin past
        // the observed bounds covers shape-rounding (sweep step sizes)
        // without admitting wild extrapolation.
        let m_lo = (self.min_m / 2).max(1);
        let m_hi = self.max_m.saturating_mul(2);
        let n_lo = (self.min_n / 2).max(1);
        let n_hi = self.max_n.saturating_mul(2);
        let k_lo = (self.min_k / 2).max(1);
        let k_hi = self.max_k.saturating_mul(2);
        if m < m_lo || m > m_hi || n < n_lo || n > n_hi || k < k_lo || k > k_hi {
            return None;
        }
        let m = m as f64;
        let n = n as f64;
        let k = k as f64;
        let pred = self.mnk_coeff * (m * n * k)
            + self.surface_coeff * (m * n + n * k + m * k)
            + self.launch_us;
        // Reject non-physical predictions — model can interpolate
        // negative cost or absurd values between sparse points. 100x
        // median_us is the absurdity ceiling; a real GEMM never
        // exceeds it on the sweep's shape grid.
        if pred.is_finite() && pred >= 0.0 && pred <= 100.0 * self.median_us {
            Some(pred)
        } else {
            Some(self.median_us)
        }
    }
}

fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

#[derive(Clone, Debug, Default)]
pub struct CostTable {
    entries: HashMap<(String, u32, u32, u32), f64>,
    kernel_set: HashSet<String>,
    /// Per-kernel fitted predictors, populated via `build_predictors`
    /// after all CSV rows are inserted. Lookup miss in `entries`
    /// falls through to the predictor for that kernel; if both are
    /// absent the kernel has no cost data at all and `get` returns
    /// `None` (callers fall back to roofline).
    predictors: HashMap<String, KernelCostModel>,
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

    /// Build per-kernel cost predictors from the inserted rows.
    /// Call once after all CSV rows have been inserted (the CSV
    /// parser does this via [`finalize`]). Subsequent `get` calls
    /// will fall through to the predictor on exact-row miss.
    pub fn build_predictors(&mut self) {
        // Group rows by kernel name.
        let mut per_kernel: HashMap<String, Vec<(u32, u32, u32, f64)>> = HashMap::new();
        for ((kernel, m, n, k), cost) in &self.entries {
            per_kernel
                .entry(kernel.clone())
                .or_default()
                .push((*m, *n, *k, *cost));
        }
        for (kernel, points) in per_kernel {
            if let Some(model) = KernelCostModel::fit(&points) {
                self.predictors.insert(kernel, model);
            }
        }
    }

    /// Finalize the table after all CSV rows are loaded. Currently
    /// just builds predictors; future invariant checks hang here too.
    pub fn finalize(&mut self) {
        self.build_predictors();
    }

    /// Exact-row lookup. Returns `Some(measured)` if the precise
    /// `(kernel, M, N, K)` was swept; `None` otherwise. Use
    /// [`get_or_predict`] for the fallback-aware flavor.
    pub fn get(&self, kernel: &str, m: u32, n: u32, k: u32) -> Option<f64> {
        // Use a borrowed-key tuple to skip the per-call String allocation.
        // HashMap's Borrow impl on tuples doesn't quite let us borrow the
        // String directly, so we still allocate here — but this is the
        // cold path (called once the candidate is being evaluated, not
        // per-impl-per-tile-per-workload).
        self.entries.get(&(kernel.to_string(), m, n, k)).copied()
    }

    /// Exact-row lookup with predictor fallback. Returns `Some` if
    /// either the row is measured or the kernel has a fitted
    /// predictor (i.e. ≥ 3 measured points exist for this kernel).
    /// Returns `None` only when the kernel name is completely absent
    /// from the CSV — the caller then falls back to roofline or
    /// `UNCALIBRATED_COST_US`.
    ///
    /// This is the `Impl::cost_us` hot path's preferred entry point.
    pub fn get_or_predict(&self, kernel: &str, m: u32, n: u32, k: u32) -> Option<f64> {
        if let Some(measured) = self.get(kernel, m, n, k) {
            return Some(measured);
        }
        self.predictors
            .get(&kernel.to_string())
            .and_then(|model| model.predict(m, n, k))
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
}

impl TargetProfile {
    /// Look up an empirical cost. Returns `None` only when the kernel
    /// name is absent from the CSV entirely; missing exact rows are
    /// filled by the per-kernel fitted predictor (see
    /// [`CostTable::get_or_predict`]).
    ///
    /// Pre-predictor behavior: this returned `None` whenever the
    /// exact `(kernel, M, N, K)` row was missing, sending the cost
    /// path to roofline. Roofline is structurally optimistic —
    /// every new arch with an unswept `intermediate_size` got bad
    /// tile picks. The fitted predictor closes that gap.
    pub fn cost_us_for(&self, kernel: &str, m: u32, n: u32, k: u32) -> Option<f64> {
        self.cost_table.get_or_predict(kernel, m, n, k)
    }
}

/// Build a `TargetProfile` from a `ferrite-cuda-targets` profile
/// const, parsing the embedded CSV bytes into a `CostTable`. The
/// proc-macro calls this once per `#[forward]` invocation after
/// resolving the active GPU (`ferrite_cuda_targets::detect()`).
pub fn from_profile_def(def: &ProfileDef) -> TargetProfile {
    TargetProfile {
        name: def.name.to_string(),
        source_path: PathBuf::new(),
        compute_capability: def.compute_capability,
        num_sms: def.num_sms,
        peak_tflops_fp16: def.peak_tflops_fp16,
        memory_bandwidth_gbps: def.memory_bandwidth_gbps,
        shared_memory_per_sm_kb: def.shared_memory_per_sm_kb,
        cost_table: parse_cost_csv(def.cost_csv),
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
    table.finalize();
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

    #[test]
    fn predictor_recovers_synthetic_linear_costs() {
        // Generate a synthetic ground-truth: cost = a*MNK + b*surface + c
        // and verify the OLS fitter recovers (a, b, c) and predicts
        // unseen shapes within rounding error.
        let a_true = 1.0e-6; // 1 µs per million flop
        let b_true = 1.0e-3; // 1 µs per kelem of memory traffic
        let c_true = 2.5; // 2.5 µs launch overhead
        let synth = |m: u32, n: u32, k: u32| {
            let mf = m as f64;
            let nf = n as f64;
            let kf = k as f64;
            a_true * (mf * nf * kf) + b_true * (mf * nf + nf * kf + mf * kf) + c_true
        };

        let mut table = CostTable::new();
        for &(m, n, k) in &[
            (1u32, 64, 64),
            (4, 128, 128),
            (16, 256, 256),
            (64, 512, 512),
            (1, 4096, 4096),
            (8, 8192, 1024),
        ] {
            table.insert("synth", m, n, k, synth(m, n, k));
        }
        table.finalize();

        // Predictor should now exist for `synth`.
        let exact = table.get_or_predict("synth", 4, 128, 128).unwrap();
        assert!((exact - synth(4, 128, 128)).abs() < 1e-9);

        // Unseen shape — predictor fills in.
        let pred = table.get_or_predict("synth", 32, 1024, 1024).unwrap();
        let truth = synth(32, 1024, 1024);
        assert!(
            (pred - truth).abs() / truth < 0.01,
            "predictor diverges: pred={pred}, truth={truth}"
        );

        // Absent kernel still returns None.
        assert!(table.get_or_predict("not_in_csv", 1, 2, 3).is_none());
    }

    #[test]
    fn predictor_handles_too_few_points() {
        // < 3 points → no predictor, exact-only fallback.
        let mut table = CostTable::new();
        table.insert("scarce", 1, 64, 64, 5.0);
        table.insert("scarce", 2, 64, 64, 7.0);
        table.finalize();
        // Exact still works.
        assert_eq!(table.get_or_predict("scarce", 1, 64, 64), Some(5.0));
        // Off-grid returns None (predictor wasn't built).
        assert!(table.get_or_predict("scarce", 4, 64, 64).is_none());
    }
}
