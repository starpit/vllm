// SPDX-License-Identifier: Apache-2.0
//! CUDA graph configuration for decode-step acceleration.

/// CUDA graph mode - controls how graphs are captured and replayed.
///
/// Matches Python vLLM's `CudaGraphMode` enum for piecewise graph support.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum CudaGraphMode {
    /// No CUDA graphs - eager execution only.
    None,
    /// Piecewise graphs only - attention excluded from graphs.
    Piecewise,
    /// Full monolithic graphs only - entire forward pass captured.
    Full,
    /// Full for uniform decode, piecewise for mixed batches (DEFAULT).
    /// Matches Python vLLM's default behavior.
    #[default]
    FullAndPiecewise,
    /// Full for uniform decode, eager for mixed batches.
    FullDecodeOnly,
}

impl std::str::FromStr for CudaGraphMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s).ok_or_else(|| format!("unknown CudaGraphMode: {s}"))
    }
}

impl CudaGraphMode {
    /// Get the mode to use for uniform decode batches.
    pub fn decode_mode(&self) -> CudaGraphMode {
        match self {
            CudaGraphMode::FullAndPiecewise => CudaGraphMode::Full,
            CudaGraphMode::FullDecodeOnly => CudaGraphMode::Full,
            _ => *self,
        }
    }

    /// Get the mode to use for mixed/prefill batches.
    pub fn mixed_mode(&self) -> CudaGraphMode {
        match self {
            CudaGraphMode::FullAndPiecewise => CudaGraphMode::Piecewise,
            CudaGraphMode::FullDecodeOnly => CudaGraphMode::None,
            _ => *self,
        }
    }

    /// Parse from string (e.g., "full_and_piecewise", "full", "piecewise", "none").
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" => Some(Self::None),
            "piecewise" => Some(Self::Piecewise),
            "full" => Some(Self::Full),
            "full_and_piecewise" | "full-and-piecewise" => Some(Self::FullAndPiecewise),
            "full_decode_only" | "full-decode-only" => Some(Self::FullDecodeOnly),
            _ => None,
        }
    }
}

/// CUDA graph configuration.
///
/// When enabled, the engine captures CUDA graphs for decode steps at
/// power-of-2 batch sizes and replays them instead of launching individual
/// kernels. This eliminates per-kernel launch overhead (~1-3ms per step).
#[derive(Debug, Clone, serde::Serialize)]
pub struct CudaGraphConfig {
    /// Whether CUDA graphs are enabled. Default: true on CUDA.
    pub enabled: bool,
    /// Graph capture mode. Default: FullAndPiecewise.
    pub mode: CudaGraphMode,
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
            mode: CudaGraphMode::default(),
            capture_sizes: vec![1, 2, 4, 8, 16, 32, 64, 128, 256],
            num_warmups: 2,
        }
    }
}

impl CudaGraphConfig {
    /// Compute Python-matching auto capture sizes based on max_num_seqs.
    ///
    /// Matches Python vLLM's `get_capture_sizes()`:
    /// - 1, 2, 4
    /// - 8, 16, 24, ..., min(256, max_num_seqs+1) in steps of 8
    /// - Then steps of 16 up to max_num_seqs
    pub fn auto_capture_sizes(max_num_seqs: usize) -> Vec<usize> {
        let mut sizes = vec![1, 2, 4];
        let mut bs = 8;
        while bs < 256.min(max_num_seqs + 1) {
            sizes.push(bs);
            bs += 8;
        }
        while bs <= max_num_seqs {
            sizes.push(bs);
            bs += 16;
        }
        sizes
    }

    /// Parse a comma-separated list of batch sizes (e.g. "1,2,4,8").
    pub fn parse_sizes(s: &str) -> Vec<usize> {
        if s.trim().eq_ignore_ascii_case("auto") {
            return Vec::new(); // empty → cuda_worker computes Python-matching sizes
        }
        let mut sizes: Vec<usize> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
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
        assert_eq!(cfg.mode, CudaGraphMode::FullAndPiecewise);
        assert_eq!(cfg.capture_sizes, vec![1, 2, 4, 8, 16, 32, 64, 128, 256]);
        assert_eq!(cfg.num_warmups, 2);
    }

    #[test]
    fn test_cuda_graph_mode() {
        assert_eq!(
            CudaGraphMode::FullAndPiecewise.decode_mode(),
            CudaGraphMode::Full
        );
        assert_eq!(
            CudaGraphMode::FullAndPiecewise.mixed_mode(),
            CudaGraphMode::Piecewise
        );
        assert_eq!(
            CudaGraphMode::FullDecodeOnly.decode_mode(),
            CudaGraphMode::Full
        );
        assert_eq!(
            CudaGraphMode::FullDecodeOnly.mixed_mode(),
            CudaGraphMode::None
        );
        assert_eq!(CudaGraphMode::Full.decode_mode(), CudaGraphMode::Full);
        assert_eq!(CudaGraphMode::Full.mixed_mode(), CudaGraphMode::Full);
    }

    #[test]
    fn test_parse_mode() {
        assert_eq!(
            CudaGraphMode::parse("full_and_piecewise"),
            Some(CudaGraphMode::FullAndPiecewise)
        );
        assert_eq!(
            CudaGraphMode::parse("full-and-piecewise"),
            Some(CudaGraphMode::FullAndPiecewise)
        );
        assert_eq!(CudaGraphMode::parse("full"), Some(CudaGraphMode::Full));
        assert_eq!(
            CudaGraphMode::parse("piecewise"),
            Some(CudaGraphMode::Piecewise)
        );
        assert_eq!(CudaGraphMode::parse("none"), Some(CudaGraphMode::None));
        assert_eq!(CudaGraphMode::parse("invalid"), None);
    }

    #[test]
    fn test_auto_capture_sizes() {
        let sizes = CudaGraphConfig::auto_capture_sizes(256);
        assert_eq!(sizes[0], 1);
        assert_eq!(sizes[1], 2);
        assert_eq!(sizes[2], 4);
        assert_eq!(sizes[3], 8);
        // Should include 248 (last step-8 before 256)
        assert!(sizes.contains(&248));
        // 256 is >= 256.min(257) so it's in the step-16 loop
        assert!(sizes.contains(&256));
        // Small max_num_seqs
        let small = CudaGraphConfig::auto_capture_sizes(4);
        assert_eq!(small, vec![1, 2, 4]);
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
