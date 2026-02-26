// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Parallel / distributed execution configuration, ported from
//! `vllm/config/parallel.py`.

use serde::{Deserialize, Serialize};

/// Configuration for the distributed execution.
///
/// Ported from `vllm.config.parallel.ParallelConfig`.  Only the fields needed
/// by the scheduler and KV cache manager are included here; the full Python
/// class carries many more runtime/networking fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Number of pipeline parallel groups.
    #[serde(default = "one")]
    pub pipeline_parallel_size: usize,

    /// Number of tensor parallel groups.
    #[serde(default = "one")]
    pub tensor_parallel_size: usize,

    /// Number of prefill context parallel groups.
    #[serde(default = "one")]
    pub prefill_context_parallel_size: usize,

    /// Number of data parallel groups.
    #[serde(default = "one")]
    pub data_parallel_size: usize,

    /// Number of local data parallel groups.
    #[serde(default = "one")]
    pub data_parallel_size_local: usize,

    /// Rank of this process within the data parallel group.
    #[serde(default)]
    pub data_parallel_rank: usize,

    /// Equal to `data_parallel_rank` but not used for torch process groups
    /// and not overridden for dense models.
    #[serde(default)]
    pub data_parallel_index: usize,

    /// Number of decode context parallel groups.  TP size must be divisible
    /// by this value since DCP reuses the TP GPUs.
    #[serde(default = "one")]
    pub decode_context_parallel_size: usize,

    /// Interleave size of KV cache storage while using DCP or PCP.
    #[serde(default = "one")]
    pub cp_kv_cache_interleave_size: usize,

    /// Whether expert parallelism is enabled for MoE layers.
    #[serde(default)]
    pub enable_expert_parallel: bool,

    /// Whether expert-parallel load balancing is enabled.
    #[serde(default)]
    pub enable_eplb: bool,

    /// Whether the model is known to be an MoE model.
    #[serde(default)]
    pub is_moe_model: Option<bool>,

    /// Disable the custom all-reduce kernel and fall back to NCCL.
    #[serde(default)]
    pub disable_custom_all_reduce: bool,

    /// Enable dual batch overlap for the model executor.
    #[serde(default)]
    pub enable_dbo: bool,

    /// Number of micro-batches (ubatch_size).
    #[serde(default)]
    pub ubatch_size: usize,

    /// Global rank in distributed setup.
    #[serde(default)]
    pub rank: usize,

    /// Number of nodes.
    #[serde(default = "one")]
    pub nnodes: usize,

    /// Node rank within the distributed setup.
    #[serde(default)]
    pub node_rank: usize,
}

fn one() -> usize {
    1
}

impl ParallelConfig {
    /// World size is `pipeline_parallel_size * tensor_parallel_size *
    /// prefill_context_parallel_size`.
    pub fn world_size(&self) -> usize {
        self.pipeline_parallel_size * self.tensor_parallel_size * self.prefill_context_parallel_size
    }

    /// World size including data parallelism: `world_size * data_parallel_size`.
    pub fn world_size_across_dp(&self) -> usize {
        self.world_size() * self.data_parallel_size
    }

    /// Whether micro-batching is in use (DBO or explicit ubatch_size > 1).
    pub fn use_ubatching(&self) -> bool {
        self.enable_dbo || self.ubatch_size > 1
    }

    /// Number of micro-batches when ubatching is enabled.
    pub fn num_ubatches(&self) -> usize {
        if self.enable_dbo { 2 } else { self.ubatch_size }
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            pipeline_parallel_size: 1,
            tensor_parallel_size: 1,
            prefill_context_parallel_size: 1,
            data_parallel_size: 1,
            data_parallel_size_local: 1,
            data_parallel_rank: 0,
            data_parallel_index: 0,
            decode_context_parallel_size: 1,
            cp_kv_cache_interleave_size: 1,
            enable_expert_parallel: false,
            enable_eplb: false,
            is_moe_model: None,
            disable_custom_all_reduce: false,
            enable_dbo: false,
            ubatch_size: 0,
            rank: 0,
            nnodes: 1,
            node_rank: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_parallel_config() {
        let cfg = ParallelConfig::default();
        assert_eq!(cfg.world_size(), 1);
        assert_eq!(cfg.world_size_across_dp(), 1);
        assert!(!cfg.use_ubatching());
    }

    #[test]
    fn test_world_size_computation() {
        let cfg = ParallelConfig {
            pipeline_parallel_size: 2,
            tensor_parallel_size: 4,
            data_parallel_size: 2,
            ..Default::default()
        };
        assert_eq!(cfg.world_size(), 8);
        assert_eq!(cfg.world_size_across_dp(), 16);
    }

    #[test]
    fn test_parallel_config_roundtrip() {
        let cfg = ParallelConfig {
            tensor_parallel_size: 4,
            data_parallel_size: 2,
            decode_context_parallel_size: 2,
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: ParallelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.tensor_parallel_size, cfg2.tensor_parallel_size);
        assert_eq!(cfg.data_parallel_size, cfg2.data_parallel_size);
        assert_eq!(
            cfg.decode_context_parallel_size,
            cfg2.decode_context_parallel_size
        );
    }
}
