//! compile! proc macro: reads ferrite.toml manifest, runs fusion transforms,
//! emits a typed launch function.
//!
//! Usage:
//! ```rust,ignore
//! const NORM_QKV: FerriteFn = compile!(
//!     a = intrinsic(rms_norm),
//!     b = gemm_64x128x32,
//!     bind = { a.output => b.param_0 },
//! );
//!
//! // In forward():
//! let qkv = NORM_QKV.launch(stream, alloc, rms_weight, epsilon, hidden, input, b_weight);
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

/// Parsed kernel entry from ferrite.toml.
#[derive(Debug, Clone)]
pub(crate) struct ManifestKernel {
    pub name: String,
    pub ptx_path: String,
    pub derivations_path: String,
    pub tile: (u32, u32, u32),
    pub threads: u32,
    pub smem: u32,
}

/// Parsed ferrite.toml manifest.
#[derive(Debug)]
pub(crate) struct Manifest {
    pub kernels: BTreeMap<String, ManifestKernel>,
    pub intrinsics: Vec<String>,
}

/// Parse ferrite.toml (minimal TOML parser — no serde dependency in proc macros).
pub(crate) fn parse_manifest(toml: &str) -> Result<Manifest, String> {
    let mut kernels = BTreeMap::new();
    let mut intrinsics = Vec::new();
    let mut current_section = String::new();
    let mut current_kernel: Option<(String, ManifestKernel)> = None;

    for line in toml.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Section header: [kernels.gemm_64x128x32] or [intrinsics]
        if line.starts_with('[') && line.ends_with(']') {
            // Flush previous kernel
            if let Some((name, kernel)) = current_kernel.take() {
                kernels.insert(name, kernel);
            }

            let section = &line[1..line.len() - 1];
            current_section = section.to_string();

            if let Some(kernel_name) = section.strip_prefix("kernels.") {
                current_kernel = Some((
                    kernel_name.to_string(),
                    ManifestKernel {
                        name: kernel_name.to_string(),
                        ptx_path: String::new(),
                        derivations_path: String::new(),
                        tile: (0, 0, 0),
                        threads: 0,
                        smem: 0,
                    },
                ));
            }
            continue;
        }

        // Key = value
        if let Some((key, val)) = line.split_once('=') {
            let key = key.trim();
            let val = val.trim().trim_matches('"');

            if current_section == "intrinsics" {
                if val == "true" {
                    intrinsics.push(key.to_string());
                }
            } else if let Some((_, ref mut kernel)) = current_kernel {
                match key {
                    "ptx" => kernel.ptx_path = val.to_string(),
                    "derivations" => kernel.derivations_path = val.to_string(),
                    "threads" => {
                        kernel.threads = val.parse().map_err(|e| format!("bad threads: {e}"))?
                    }
                    "smem" => kernel.smem = val.parse().map_err(|e| format!("bad smem: {e}"))?,
                    "tile" => {
                        // Parse [64, 128, 32]
                        let inner = val.trim_start_matches('[').trim_end_matches(']');
                        let parts: Vec<u32> = inner
                            .split(',')
                            .map(|s| s.trim().parse().map_err(|e| format!("bad tile value: {e}")))
                            .collect::<Result<_, _>>()?;
                        if parts.len() != 3 {
                            return Err(format!("tile must have 3 values, got {}", parts.len()));
                        }
                        kernel.tile = (parts[0], parts[1], parts[2]);
                    }
                    _ => {} // ignore unknown keys
                }
            }
        }
    }

    // Flush last kernel
    if let Some((name, kernel)) = current_kernel {
        kernels.insert(name, kernel);
    }

    Ok(Manifest {
        kernels,
        intrinsics,
    })
}

/// Find the ferrite.toml manifest relative to the proc macro's invocation site.
pub(crate) fn find_manifest() -> Result<(String, PathBuf), String> {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").map_err(|_| "CARGO_MANIFEST_DIR not set")?;
    // The manifest is in ptx-fusion/kernels/ferrite.toml
    // From vllm-cuda (or any crate), we need to find ptx-fusion relative to the workspace
    let workspace = PathBuf::from(&manifest_dir);

    // Try relative paths from common locations
    let candidates = [
        workspace.join("kernels/ferrite.toml"), // from ptx-fusion itself
        workspace.join("../ptx-fusion/kernels/ferrite.toml"), // from ptx-fusion-macros
        workspace.join("../../ptx-fusion/kernels/ferrite.toml"), // from vllm-cuda
    ];

    for path in &candidates {
        if path.exists() {
            let content = std::fs::read_to_string(path)
                .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
            let kernel_dir = path.parent().unwrap().to_path_buf();
            return Ok((content, kernel_dir));
        }
    }

    Err(format!(
        "ferrite.toml not found (searched from {manifest_dir})"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_manifest_basic() {
        let toml = r#"
[kernels.gemm_64x128x32]
ptx = "cutlass_bf16_64x128x32_sm89.ptx"
derivations = "cutlass_bf16_64x128x32_sm89.derivations.json"
tile = [64, 128, 32]
threads = 128
smem = 36864

[intrinsics]
rms_norm = true
silu = true
"#;
        let m = parse_manifest(toml).unwrap();
        assert_eq!(m.kernels.len(), 1);
        let k = &m.kernels["gemm_64x128x32"];
        assert_eq!(k.tile, (64, 128, 32));
        assert_eq!(k.threads, 128);
        assert_eq!(k.smem, 36864);
        assert!(m.intrinsics.contains(&"rms_norm".to_string()));
        assert!(m.intrinsics.contains(&"silu".to_string()));
    }
}
