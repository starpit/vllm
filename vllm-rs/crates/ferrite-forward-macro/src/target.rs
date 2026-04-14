// SPDX-License-Identifier: Apache-2.0
//! Target profiles — hardware metadata the cost model consumes.
//!
//! Mirrors [`config`](crate::config): JSON files in a `target_profiles/`
//! directory, one per hardware target. Separate from model configs
//! because targets describe hardware (what the code runs on) while
//! configs describe models (what the model's dimensions are).

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

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

    Ok(TargetProfile {
        name: get_str(&json, "name", path)?.to_string(),
        source_path: path.to_path_buf(),
        compute_capability: get_u64(&json, "compute_capability", path)? as u32,
        num_sms: get_u64(&json, "num_sms", path)? as u32,
        peak_tflops_fp16: get_f64(&json, "peak_tflops_fp16", path)?,
        memory_bandwidth_gbps: get_f64(&json, "memory_bandwidth_gbps", path)?,
        shared_memory_per_sm_kb: get_u64(&json, "shared_memory_per_sm_kb", path)? as u32,
    })
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
