// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Maps GPU class + count to GCE machine type and accelerator type.

use anyhow::{Result, bail};

/// Resolved GCE machine and accelerator types.
pub struct GpuConfig {
    pub machine_type: String,
    pub accelerator_type: String,
    pub accelerator_count: u32,
}

/// Resolve a human-friendly GPU class name and count into GCE API values.
pub fn resolve(gpu_class: &str, gpu_count: u32) -> Result<GpuConfig> {
    match gpu_class {
        "l40s" => resolve_l40s(gpu_count),
        "a100-40" => resolve_a100_40(gpu_count),
        "a100-80" => resolve_a100_80(gpu_count),
        "h100" => resolve_h100(gpu_count),
        _ => bail!("unknown GPU class {gpu_class:?}; supported: l40s, a100-40, a100-80, h100"),
    }
}

fn resolve_l40s(count: u32) -> Result<GpuConfig> {
    let machine_type = match count {
        1 => "g2-standard-4",
        2 => "g2-standard-8",
        4 => "g2-standard-16",
        8 => "g2-standard-48",
        _ => bail!("l40s supports gpu_count 1, 2, 4, or 8 (got {count})"),
    };
    Ok(GpuConfig {
        machine_type: machine_type.to_string(),
        accelerator_type: "nvidia-l4".to_string(),
        accelerator_count: count,
    })
}

fn resolve_a100_40(count: u32) -> Result<GpuConfig> {
    let machine_type = match count {
        1 => "a2-highgpu-1g",
        2 => "a2-highgpu-2g",
        4 => "a2-highgpu-4g",
        8 => "a2-highgpu-8g",
        _ => bail!("a100-40 supports gpu_count 1, 2, 4, or 8 (got {count})"),
    };
    Ok(GpuConfig {
        machine_type: machine_type.to_string(),
        accelerator_type: "nvidia-tesla-a100".to_string(),
        accelerator_count: count,
    })
}

fn resolve_a100_80(count: u32) -> Result<GpuConfig> {
    let machine_type = match count {
        1 => "a2-ultragpu-1g",
        2 => "a2-ultragpu-2g",
        4 => "a2-ultragpu-4g",
        8 => "a2-ultragpu-8g",
        _ => bail!("a100-80 supports gpu_count 1, 2, 4, or 8 (got {count})"),
    };
    Ok(GpuConfig {
        machine_type: machine_type.to_string(),
        accelerator_type: "nvidia-a100-80gb".to_string(),
        accelerator_count: count,
    })
}

fn resolve_h100(count: u32) -> Result<GpuConfig> {
    let machine_type = match count {
        1 => "a3-highgpu-1g",
        2 => "a3-highgpu-2g",
        4 => "a3-highgpu-4g",
        8 => "a3-highgpu-8g",
        _ => bail!("h100 supports gpu_count 1, 2, 4, or 8 (got {count})"),
    };
    Ok(GpuConfig {
        machine_type: machine_type.to_string(),
        accelerator_type: "nvidia-h100-80gb".to_string(),
        accelerator_count: count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l40s_valid() {
        let cfg = resolve("l40s", 1).unwrap();
        assert_eq!(cfg.machine_type, "g2-standard-4");
        assert_eq!(cfg.accelerator_type, "nvidia-l4");

        let cfg = resolve("l40s", 8).unwrap();
        assert_eq!(cfg.machine_type, "g2-standard-48");
    }

    #[test]
    fn test_h100_valid() {
        let cfg = resolve("h100", 4).unwrap();
        assert_eq!(cfg.machine_type, "a3-highgpu-4g");
        assert_eq!(cfg.accelerator_type, "nvidia-h100-80gb");
    }

    #[test]
    fn test_unknown_class() {
        assert!(resolve("v100", 1).is_err());
    }

    #[test]
    fn test_invalid_count() {
        assert!(resolve("l40s", 3).is_err());
    }
}
