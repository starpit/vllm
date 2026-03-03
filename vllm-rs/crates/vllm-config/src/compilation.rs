// SPDX-License-Identifier: Apache-2.0
//! CUDA graph configuration for decode-step acceleration.

/// CUDA graph configuration.
///
/// When enabled, the engine captures CUDA graphs for decode steps at
/// power-of-2 batch sizes and replays them instead of launching individual
/// kernels. This eliminates per-kernel launch overhead (~1-3ms per step).
#[derive(Debug, Clone)]
pub struct CudaGraphConfig {
    /// Whether CUDA graphs are enabled. Default: true on CUDA.
    pub enabled: bool,
    /// Batch sizes to capture graphs for.
    /// Default: `[1, 2, 4, 8, 16, 32, 64, 128, 256]`.
    pub capture_sizes: Vec<usize>,
    /// Number of warmup runs before capture (ensures stable kernel selection).
    pub num_warmups: usize,
}

impl Default for CudaGraphConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            capture_sizes: vec![1, 2, 4, 8, 16, 32, 64, 128, 256],
            num_warmups: 2,
        }
    }
}

impl CudaGraphConfig {
    /// Parse a comma-separated list of batch sizes (e.g. "1,2,4,8").
    pub fn parse_sizes(s: &str) -> Vec<usize> {
        let mut sizes: Vec<usize> = s
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        sizes.sort();
        sizes.dedup();
        sizes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = CudaGraphConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.capture_sizes, vec![1, 2, 4, 8, 16, 32, 64, 128, 256]);
        assert_eq!(cfg.num_warmups, 2);
    }

    #[test]
    fn test_parse_sizes() {
        assert_eq!(CudaGraphConfig::parse_sizes("1,2,4,8"), vec![1, 2, 4, 8]);
        assert_eq!(CudaGraphConfig::parse_sizes("8,4,2,1"), vec![1, 2, 4, 8]);
        assert_eq!(CudaGraphConfig::parse_sizes("4,4,8"), vec![4, 8]);
        assert_eq!(CudaGraphConfig::parse_sizes(""), Vec::<usize>::new());
        assert_eq!(CudaGraphConfig::parse_sizes("abc,4"), vec![4]);
    }
}
