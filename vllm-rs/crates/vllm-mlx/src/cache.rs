// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! MLX KV cache — simple per-layer contiguous cache using `mlx_rs::Array`.
//!
//! MLX's lazy evaluation means KV cache concatenation is fused into the
//! forward graph, so there's no separate dispatch overhead. We start with
//! a simple contiguous per-request cache (matching CandleWorker's legacy
//! path). Paged KV cache is deferred to Phase 10c.

use mlx_rs::Array;

/// Per-layer KV cache: `(key, value)` arrays.
///
/// Keys and values have shape `[1, num_kv_heads, cached_seq_len, head_dim]`.
pub type LayerKvCache = (Array, Array);

/// KV cache for a model: one optional `(key, value)` per layer.
///
/// `None` entries indicate a layer with no cached K/V yet (first call).
pub type MlxKvCache = Vec<Option<LayerKvCache>>;

/// Create an empty KV cache for a model with `num_layers` layers.
pub fn empty_kv_cache(num_layers: usize) -> MlxKvCache {
    vec![None; num_layers]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_kv_cache() {
        let cache = empty_kv_cache(4);
        assert_eq!(cache.len(), 4);
        for entry in &cache {
            assert!(entry.is_none());
        }
    }
}
