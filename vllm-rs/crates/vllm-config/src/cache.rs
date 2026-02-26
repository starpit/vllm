// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! KV cache configuration types, ported from `vllm/config/cache.py` and
//! `vllm/v1/kv_cache_interface.py`.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CacheConfig  (from vllm/config/cache.py)
// ---------------------------------------------------------------------------

/// Data type used for KV cache storage.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheDType {
    #[default]
    Auto,
    #[serde(rename = "bfloat16")]
    BFloat16,
    Fp8,
    #[serde(rename = "fp8_e4m3")]
    Fp8E4m3,
    #[serde(rename = "fp8_e5m2")]
    Fp8E5m2,
    #[serde(rename = "fp8_inc")]
    Fp8Inc,
    #[serde(rename = "fp8_ds_mla")]
    Fp8DsMla,
}

/// Mamba cache mode.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MambaCacheMode {
    /// No Mamba state caching (prefix caching disabled).
    #[default]
    None,
    /// Cache the Mamba state at every `i * block_size` position.
    All,
    /// Only cache when the token is at position `i * block_size` AND it is the
    /// last token of a scheduler step.
    Align,
}

/// Hash algorithm used for prefix caching.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PrefixCachingHashAlgo {
    #[default]
    Sha256,
    #[serde(rename = "sha256_cbor")]
    Sha256Cbor,
    Xxhash,
    #[serde(rename = "xxhash_cbor")]
    XxhashCbor,
}

/// Configuration for the KV cache.
///
/// Ported from `vllm.config.cache.CacheConfig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Size of a contiguous cache block in number of tokens.
    /// On CUDA devices, only block sizes up to 32 are supported.
    /// If `None`, will be set by the platform at runtime.
    #[serde(default)]
    pub block_size: Option<usize>,

    /// Fraction of GPU memory to use for the model executor (0, 1].
    #[serde(default = "default_gpu_memory_utilization")]
    pub gpu_memory_utilization: f64,

    /// CPU swap space per GPU in GiB.
    #[serde(default = "default_swap_space")]
    pub swap_space: f64,

    /// Data type for KV cache storage.
    #[serde(default)]
    pub cache_dtype: CacheDType,

    /// Whether the model is attention-free (no KV cache needed).
    #[serde(default)]
    pub is_attention_free: bool,

    /// Override for the number of GPU blocks (for testing).
    #[serde(default)]
    pub num_gpu_blocks_override: Option<usize>,

    /// Sliding window size (set from ModelConfig).
    #[serde(default)]
    pub sliding_window: Option<usize>,

    /// Whether prefix caching is enabled.
    #[serde(default = "bool_true")]
    pub enable_prefix_caching: bool,

    /// Hash algorithm for prefix caching.
    #[serde(default)]
    pub prefix_caching_hash_algo: PrefixCachingHashAlgo,

    /// CPU offload space per GPU in GiB. Deprecated -- use OffloadConfig.
    #[serde(default)]
    pub cpu_offload_gb: f64,

    /// Whether to dynamically calculate k/v scales for fp8 KV cache.
    #[serde(default)]
    pub calculate_kv_scales: bool,

    /// CPU KV cache space in bytes (CPU backend only).
    #[serde(default)]
    pub cpu_kvcache_space_bytes: Option<usize>,

    /// Optional override for Mamba page size.
    #[serde(default)]
    pub mamba_page_size_padded: Option<usize>,

    /// Size of a contiguous Mamba cache block (must be multiple of 8).
    #[serde(default)]
    pub mamba_block_size: Option<usize>,

    /// Data type for the Mamba cache (conv + ssm state).
    #[serde(default = "default_mamba_dtype")]
    pub mamba_cache_dtype: String,

    /// Data type for the Mamba SSM state only.
    #[serde(default = "default_mamba_dtype")]
    pub mamba_ssm_cache_dtype: String,

    /// Cache strategy for Mamba layers.
    #[serde(default)]
    pub mamba_cache_mode: MambaCacheMode,

    // -- Post-profiling fields (set at runtime) --
    /// Number of GPU blocks allocated after profiling.
    #[serde(default)]
    pub num_gpu_blocks: Option<usize>,

    /// Number of CPU blocks allocated after profiling.
    #[serde(default)]
    pub num_cpu_blocks: Option<usize>,

    /// Enable fast prefill optimization for KV sharing setups.
    #[serde(default)]
    pub kv_sharing_fast_prefill: bool,

    /// Size of KV cache per GPU in bytes. `None` means auto-detect from
    /// `gpu_memory_utilization`.
    #[serde(default)]
    pub kv_cache_memory_bytes: Option<usize>,

    /// KV offloading buffer size in GiB. `None` means no offloading.
    #[serde(default)]
    pub kv_offloading_size: Option<f64>,

    /// Backend for KV cache offloading.
    #[serde(default = "default_kv_offloading_backend")]
    pub kv_offloading_backend: String,
}

fn default_gpu_memory_utilization() -> f64 {
    0.9
}

fn default_swap_space() -> f64 {
    4.0
}

fn bool_true() -> bool {
    true
}

fn default_mamba_dtype() -> String {
    "auto".to_string()
}

fn default_kv_offloading_backend() -> String {
    "native".to_string()
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            block_size: None,
            gpu_memory_utilization: 0.9,
            swap_space: 4.0,
            cache_dtype: CacheDType::default(),
            is_attention_free: false,
            num_gpu_blocks_override: None,
            sliding_window: None,
            enable_prefix_caching: true,
            prefix_caching_hash_algo: PrefixCachingHashAlgo::default(),
            cpu_offload_gb: 0.0,
            calculate_kv_scales: false,
            cpu_kvcache_space_bytes: None,
            mamba_page_size_padded: None,
            mamba_block_size: None,
            mamba_cache_dtype: "auto".to_string(),
            mamba_ssm_cache_dtype: "auto".to_string(),
            mamba_cache_mode: MambaCacheMode::default(),
            num_gpu_blocks: None,
            num_cpu_blocks: None,
            kv_sharing_fast_prefill: false,
            kv_cache_memory_bytes: None,
            kv_offloading_size: None,
            kv_offloading_backend: "native".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// KV cache interface types  (from vllm/v1/kv_cache_interface.py)
// ---------------------------------------------------------------------------

/// The type of a KV cache spec, describing the attention pattern for a group
/// of layers.
///
/// This is a simplified Rust enum that captures the variants from the Python
/// class hierarchy rooted at `KVCacheSpec`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum KVCacheSpecType {
    /// Standard full (causal) attention.
    FullAttention {
        num_kv_heads: usize,
        head_size: usize,
        /// Head size for values (may differ from key head size in GQA-V).
        #[serde(default)]
        head_size_v: Option<usize>,
        /// Optional sliding window applied on top of full attention.
        #[serde(default)]
        sliding_window: Option<usize>,
        /// Optional attention chunk size.
        #[serde(default)]
        attention_chunk_size: Option<usize>,
    },
    /// Multi-Latent Attention (DeepSeek-style).
    MLA {
        num_kv_heads: usize,
        head_size: usize,
        #[serde(default)]
        cache_dtype_str: Option<String>,
    },
    /// Sliding window attention with a fixed window.
    SlidingWindow {
        num_kv_heads: usize,
        head_size: usize,
        sliding_window: usize,
    },
    /// Chunked local attention with a fixed chunk size.
    ChunkedLocalAttention {
        num_kv_heads: usize,
        head_size: usize,
        attention_chunk_size: usize,
    },
    /// Cross-attention for encoder-decoder models.
    CrossAttention {
        num_kv_heads: usize,
        head_size: usize,
    },
    /// Encoder-only attention (no KV cache needed at inference).
    EncoderOnlyAttention {
        num_kv_heads: usize,
        head_size: usize,
    },
    /// Mamba (state-space model) cache.
    Mamba {
        /// Shapes of the Mamba state tensors.
        shapes: Vec<Vec<usize>>,
        #[serde(default = "default_mamba_type")]
        mamba_type: String,
        #[serde(default)]
        mamba_cache_mode: MambaCacheMode,
    },
}

fn default_mamba_type() -> String {
    "mamba2".to_string()
}

/// A single KV cache spec for a group of layers.
///
/// Ported from `vllm.v1.kv_cache_interface.KVCacheSpec` and its subclasses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheSpec {
    /// Number of tokens in a single cache block.
    pub block_size: usize,

    /// The specific attention type and associated parameters.
    #[serde(flatten)]
    pub spec_type: KVCacheSpecType,
}

/// A group of model layers that share the same KV cache block table.
///
/// Ported from `vllm.v1.kv_cache_interface.KVCacheGroupSpec`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheGroupSpec {
    /// The names of model layers in this group.
    pub layer_names: Vec<String>,

    /// The KV cache spec shared by all layers in this group.
    pub kv_cache_spec: KVCacheSpec,
}

/// Describes how a single KV cache tensor should be initialised by workers.
///
/// Ported from `vllm.v1.kv_cache_interface.KVCacheTensor`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheTensor {
    /// Size of the KV cache tensor in bytes.
    pub size: usize,

    /// Layer names that share this tensor.
    pub shared_by: Vec<String>,
}

/// The complete KV cache configuration of a model.
///
/// Ported from `vllm.v1.kv_cache_interface.KVCacheConfig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheConfig {
    /// The number of KV cache blocks.
    pub num_blocks: usize,

    /// How the model runner should initialise the KV cache tensors.
    pub kv_cache_tensors: Vec<KVCacheTensor>,

    /// The KV cache groups of the model.  For models with only one type of
    /// attention there is a single group containing all layers.  Hybrid models
    /// will have multiple groups.
    pub kv_cache_groups: Vec<KVCacheGroupSpec>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_cache_config() {
        let cfg = CacheConfig::default();
        assert!(cfg.enable_prefix_caching);
        assert_eq!(cfg.gpu_memory_utilization, 0.9);
        assert_eq!(cfg.mamba_cache_mode, MambaCacheMode::None);
        assert!(cfg.num_gpu_blocks.is_none());
    }

    #[test]
    fn test_cache_config_roundtrip() {
        let cfg = CacheConfig {
            block_size: Some(16),
            num_gpu_blocks: Some(1024),
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: CacheConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.block_size, cfg2.block_size);
        assert_eq!(cfg.num_gpu_blocks, cfg2.num_gpu_blocks);
    }

    #[test]
    fn test_kv_cache_spec_full_attention_roundtrip() {
        let spec = KVCacheSpec {
            block_size: 16,
            spec_type: KVCacheSpecType::FullAttention {
                num_kv_heads: 8,
                head_size: 128,
                head_size_v: None,
                sliding_window: None,
                attention_chunk_size: None,
            },
        };
        let json = serde_json::to_string(&spec).unwrap();
        let spec2: KVCacheSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec.block_size, spec2.block_size);
    }

    #[test]
    fn test_kv_cache_config_roundtrip() {
        let kv_cfg = KVCacheConfig {
            num_blocks: 512,
            kv_cache_tensors: vec![KVCacheTensor {
                size: 1024 * 1024,
                shared_by: vec!["layer0".to_string(), "layer1".to_string()],
            }],
            kv_cache_groups: vec![KVCacheGroupSpec {
                layer_names: vec!["layer0".to_string(), "layer1".to_string()],
                kv_cache_spec: KVCacheSpec {
                    block_size: 16,
                    spec_type: KVCacheSpecType::FullAttention {
                        num_kv_heads: 8,
                        head_size: 128,
                        head_size_v: Some(128),
                        sliding_window: None,
                        attention_chunk_size: None,
                    },
                },
            }],
        };
        let json = serde_json::to_string(&kv_cfg).unwrap();
        let kv_cfg2: KVCacheConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(kv_cfg.num_blocks, kv_cfg2.num_blocks);
        assert_eq!(kv_cfg.kv_cache_groups.len(), kv_cfg2.kv_cache_groups.len());
    }
}
