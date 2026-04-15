// SPDX-License-Identifier: Apache-2.0
//! Target profiles — hardware metadata the cost model consumes.
//!
//! Mirrors [`config`](crate::config): JSON files in a `target_profiles/`
//! directory, one per hardware target. Separate from model configs
//! because targets describe hardware (what the code runs on) while
//! configs describe models (what the model's dimensions are).

#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// Empirical GPU cost table: `(kernel_name, M, N, K) -> cost_us`.
///
/// Populated from `target_profiles/cost_<profile>.csv` when present.
/// Keyed by `kernel` column value (`cublas`, `cutlass_128x128_s4`,
/// `cutlass_gemv`, …) to support both the cuBLAS reference line and
/// each cutlass tile variant.
#[derive(Clone, Debug, Default)]
pub struct CostTable {
    entries: HashMap<(String, u32, u32, u32), f64>,
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
        self.entries.insert((kernel.into(), m, n, k), cost_us);
    }

    pub fn get(&self, kernel: &str, m: u32, n: u32, k: u32) -> Option<f64> {
        self.entries.get(&(kernel.to_string(), m, n, k)).copied()
    }

    /// Every distinct kernel name observed in the CSV. Useful for
    /// the solver library to enumerate `cutlass_*` variants without
    /// hardcoding them.
    pub fn kernel_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .entries
            .keys()
            .map(|(k, _, _, _)| k.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
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
    /// Look up an empirical cost. Returns `None` when either the
    /// profile has no CSV, or the (kernel, M, N, K) shape isn't in
    /// the swept grid.
    pub fn cost_us_for(&self, kernel: &str, m: u32, n: u32, k: u32) -> Option<f64> {
        self.cost_table.get(kernel, m, n, k)
    }
}

#[derive(Debug)]
pub enum TargetError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    NotADirectory(PathBuf),
    MissingField {
        path: PathBuf,
        field: &'static str,
    },
    BadField {
        path: PathBuf,
        field: &'static str,
        reason: &'static str,
    },
}

impl std::fmt::Display for TargetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "reading {}: {source}", path.display()),
            Self::Json { path, source } => write!(f, "parsing {}: {source}", path.display()),
            Self::NotADirectory(p) => write!(f, "not a directory: {}", p.display()),
            Self::MissingField { path, field } => {
                write!(f, "{}: missing field `{field}`", path.display())
            }
            Self::BadField {
                path,
                field,
                reason,
            } => write!(f, "{}: bad field `{field}`: {reason}", path.display()),
        }
    }
}

impl std::error::Error for TargetError {}

pub fn load_dir(dir: &Path) -> Result<Vec<TargetProfile>, TargetError> {
    if !dir.is_dir() {
        return Err(TargetError::NotADirectory(dir.to_path_buf()));
    }
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|source| TargetError::Io {
            path: dir.to_path_buf(),
            source,
        })?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    paths.sort();
    paths.iter().map(|p| load_file(p)).collect()
}

pub fn load_file(path: &Path) -> Result<TargetProfile, TargetError> {
    let contents = fs::read_to_string(path).map_err(|source| TargetError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let json: Value = serde_json::from_str(&contents).map_err(|source| TargetError::Json {
        path: path.to_path_buf(),
        source,
    })?;

    let name = get_str(&json, "name", path)?.to_string();

    // Look for `cost_<name>.csv` alongside the JSON. Missing CSV is
    // not an error — the solver falls back to analytic formulas.
    let cost_table = if let Some(parent) = path.parent() {
        let csv_path = parent.join(format!("cost_{name}.csv"));
        if csv_path.is_file() {
            load_cost_csv(&csv_path)?
        } else {
            CostTable::new()
        }
    } else {
        CostTable::new()
    };

    Ok(TargetProfile {
        name,
        source_path: path.to_path_buf(),
        compute_capability: get_u64(&json, "compute_capability", path)? as u32,
        num_sms: get_u64(&json, "num_sms", path)? as u32,
        peak_tflops_fp16: get_f64(&json, "peak_tflops_fp16", path)?,
        memory_bandwidth_gbps: get_f64(&json, "memory_bandwidth_gbps", path)?,
        shared_memory_per_sm_kb: get_u64(&json, "shared_memory_per_sm_kb", path)? as u32,
        cost_table,
    })
}

/// Parse a GPU cost CSV produced by the prior ferrite's
/// `gpu_cost_sweep`. Skips comment lines starting with `#` and a
/// `kernel,M,N,K,cost_us` header line. Silently skips malformed
/// rows rather than aborting the whole load.
fn load_cost_csv(path: &Path) -> Result<CostTable, TargetError> {
    let contents = fs::read_to_string(path).map_err(|source| TargetError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut table = CostTable::new();
    for raw in contents.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Header row starts with "kernel,".
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
    Ok(table)
}

fn get_str<'a>(json: &'a Value, field: &'static str, path: &Path) -> Result<&'a str, TargetError> {
    json.get(field)
        .and_then(Value::as_str)
        .ok_or(TargetError::MissingField {
            path: path.to_path_buf(),
            field,
        })
}

fn get_u64(json: &Value, field: &'static str, path: &Path) -> Result<u64, TargetError> {
    json.get(field)
        .and_then(Value::as_u64)
        .ok_or(TargetError::MissingField {
            path: path.to_path_buf(),
            field,
        })
}

fn get_f64(json: &Value, field: &'static str, path: &Path) -> Result<f64, TargetError> {
    json.get(field)
        .and_then(Value::as_f64)
        .ok_or(TargetError::MissingField {
            path: path.to_path_buf(),
            field,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo_target_profiles() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("target_profiles")
    }

    #[test]
    fn load_real_target_profiles() {
        let dir = repo_target_profiles();
        let profiles = load_dir(&dir).expect("load target profiles");
        assert!(profiles.len() >= 2, "at least l4_sm89 + h100_sm90");

        let l4 = profiles.iter().find(|p| p.name == "l4_sm89").unwrap();
        assert_eq!(l4.compute_capability, 89);
        assert_eq!(l4.num_sms, 58);
        assert!(l4.peak_tflops_fp16 > 100.0);

        let h100 = profiles.iter().find(|p| p.name == "h100_sm90").unwrap();
        assert_eq!(h100.compute_capability, 90);
        assert!(h100.peak_tflops_fp16 > l4.peak_tflops_fp16);
    }

    #[test]
    fn cost_csv_loaded_alongside_target_profile() {
        let dir = repo_target_profiles();
        let profiles = load_dir(&dir).expect("load target profiles");
        let l4 = profiles.iter().find(|p| p.name == "l4_sm89").unwrap();
        // CSV must have loaded — prior ferrite's GPU cost sweep
        // produced thousands of rows across kernel × (M, N, K).
        assert!(!l4.cost_table.is_empty(), "expected cost_l4_sm89.csv");
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
        let dir = repo_target_profiles();
        let profiles = load_dir(&dir).expect("load target profiles");
        let l4 = profiles.iter().find(|p| p.name == "l4_sm89").unwrap();
        // The first data row in cost_l4_sm89.csv is
        // `cublas,1,2048,2048,9.7`. Use it as a canary.
        let c = l4.cost_us_for("cublas", 1, 2048, 2048);
        assert!(c.is_some(), "cublas 1x2048x2048 missing from table");
        let us = c.unwrap();
        assert!(us > 0.0 && us.is_finite());
    }

    #[test]
    fn missing_dir_errors() {
        assert!(matches!(
            load_dir(Path::new("/nonexistent")),
            Err(TargetError::NotADirectory(_))
        ));
    }

    #[test]
    fn missing_field_errors() {
        let tmp = std::env::temp_dir().join("ferrite_forward_target_test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let bad = tmp.join("bad.json");
        fs::write(&bad, r#"{"name": "bad"}"#).unwrap();
        let err = load_file(&bad).unwrap_err();
        assert!(matches!(err, TargetError::MissingField { .. }));
        fs::remove_dir_all(&tmp).ok();
    }
}
