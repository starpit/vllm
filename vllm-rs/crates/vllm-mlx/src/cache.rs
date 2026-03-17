// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! MLX KV cache — simple concatenation cache matching mlx-lm's pattern.
//!
//! Each decode step concatenates the new K/V token onto the existing cache.
//! This is the same approach as mlx-lm's `ConcatenateKVCache` / `KVCache`.
//! MLX can fuse the concatenation into the forward pass graph so it executes
//! as part of the single eval() — no separate kernel dispatches for cache
//! management.

use mlx_rs::Array;
use mlx_rs::error::Exception;

/// Per-layer KV cache using concatenation.
///
/// Keys and values have shape `[1, num_kv_heads, seq_len, head_dim]`.
/// On each update, new tokens are concatenated along dim 2.
#[derive(Clone)]
pub struct MlxLayerKvCache {
    k: Array,
    v: Array,
    seq_len: usize,
}

impl MlxLayerKvCache {
    /// Create from initial K/V arrays (typically from prefill).
    fn new(k: Array, v: Array) -> Self {
        let seq_len = k.dim(2) as usize;
        Self { k, v, seq_len }
    }

    /// Concatenate new K/V tokens and return the full cache.
    ///
    /// `new_k` and `new_v` have shape `[1, heads, new_len, dim]`.
    /// Returns `(full_k, full_v)` covering all cached positions.
    pub fn update_and_view(
        &mut self,
        new_k: &Array,
        new_v: &Array,
    ) -> Result<(Array, Array), Exception> {
        self.k = mlx_rs::ops::concatenate_axis(&[&self.k, new_k], 2)?;
        self.v = mlx_rs::ops::concatenate_axis(&[&self.v, new_v], 2)?;
        self.seq_len = self.k.dim(2) as usize;
        Ok((self.k.clone(), self.v.clone()))
    }

    /// Number of cached sequence positions.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Truncate to a shorter sequence length.
    ///
    /// Used when cloning a cached KV for a new request that only matches
    /// a prefix of the original sequence.
    pub fn truncate(&mut self, new_seq_len: usize) {
        use mlx_rs::ops::indexing::TryIndexOp;
        assert!(new_seq_len <= self.seq_len);
        if new_seq_len < self.seq_len {
            self.k = self
                .k
                .try_index((.., .., ..new_seq_len as i32, ..))
                .unwrap();
            self.v = self
                .v
                .try_index((.., .., ..new_seq_len as i32, ..))
                .unwrap();
            self.seq_len = new_seq_len;
        }
    }
}

/// Batch information for batched forward passes across multiple requests.
///
/// Enables running a single `forward_batch` call over concatenated tokens from
/// multiple requests, splitting only where per-request state is needed (RoPE, KV cache, SDPA).
pub struct MlxBatchInfo {
    /// Number of requests in this batch.
    pub num_reqs: usize,
    /// Per-request query lengths (number of tokens being forwarded).
    pub q_lens: Vec<usize>,
    /// Prefix-sum start positions into the flat `[total_tokens, ...]` tensor.
    pub offsets: Vec<usize>,
    /// Per-request RoPE offsets (position of the first token in each request).
    pub rope_offsets: Vec<i32>,
    /// Total number of tokens across all requests.
    pub total_tokens: usize,
}

impl MlxBatchInfo {
    /// Build batch info from per-request token counts and rope offsets.
    pub fn new(q_lens: Vec<usize>, rope_offsets: Vec<i32>) -> Self {
        let num_reqs = q_lens.len();
        let mut offsets = Vec::with_capacity(num_reqs);
        let mut running = 0usize;
        for &ql in &q_lens {
            offsets.push(running);
            running += ql;
        }
        Self {
            num_reqs,
            q_lens,
            offsets,
            rope_offsets,
            total_tokens: running,
        }
    }
}

/// Per-layer KV cache entry: `(key, value)` arrays (legacy type alias).
pub type LayerKvCache = (Array, Array);

/// KV cache for a model: one optional pre-allocated entry per layer.
///
/// `None` entries indicate a layer with no cached K/V yet (first call).
pub type MlxKvCache = Vec<Option<MlxLayerKvCache>>;

/// Create an empty KV cache for a model with `num_layers` layers.
pub fn empty_kv_cache(num_layers: usize) -> MlxKvCache {
    (0..num_layers).map(|_| None).collect()
}

/// Update a per-layer KV cache entry and return the full cache for attention.
///
/// On first call (cache=None): stores K/V directly.
/// On subsequent calls: concatenates new tokens onto existing cache.
///
/// Returns `(full_k, full_v)` covering all cached positions.
pub fn kv_cache_update(
    cache: &mut Option<MlxLayerKvCache>,
    new_k: &Array,
    new_v: &Array,
) -> Result<(Array, Array), Exception> {
    match cache {
        Some(entry) => entry.update_and_view(new_k, new_v),
        None => {
            let entry = MlxLayerKvCache::new(new_k.clone(), new_v.clone());
            let k = entry.k.clone();
            let v = entry.v.clone();
            *cache = Some(entry);
            Ok((k, v))
        }
    }
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

    #[test]
    fn test_kv_cache_prefill() {
        let k = Array::zeros::<f32>(&[1, 4, 8, 16]).unwrap();
        let v = Array::zeros::<f32>(&[1, 4, 8, 16]).unwrap();

        let mut cache: Option<MlxLayerKvCache> = None;
        let (kv, vv) = kv_cache_update(&mut cache, &k, &v).unwrap();
        mlx_rs::transforms::eval([&kv, &vv]).unwrap();

        assert!(cache.is_some());
        assert_eq!(kv.shape(), &[1, 4, 8, 16]);
        assert_eq!(vv.shape(), &[1, 4, 8, 16]);
        assert_eq!(cache.as_ref().unwrap().seq_len(), 8);
    }

    #[test]
    fn test_kv_cache_decode_steps() {
        // Prefill: 4 tokens
        let k = Array::ones::<f32>(&[1, 2, 4, 8]).unwrap();
        let v = Array::ones::<f32>(&[1, 2, 4, 8]).unwrap();
        let mut cache: Option<MlxLayerKvCache> = None;
        kv_cache_update(&mut cache, &k, &v).unwrap();

        // Decode: 3 steps of 1 token each
        for step in 0..3 {
            let dk = Array::ones::<f32>(&[1, 2, 1, 8])
                .unwrap()
                .multiply(Array::from_f32(2.0 + step as f32))
                .unwrap();
            let dv = dk.clone();
            let (kv, vv) = kv_cache_update(&mut cache, &dk, &dv).unwrap();
            mlx_rs::transforms::eval([&kv, &vv]).unwrap();

            let expected_seq = 4 + step + 1;
            assert_eq!(kv.dim(2) as usize, expected_seq);
            assert_eq!(vv.dim(2) as usize, expected_seq);
        }

        assert_eq!(cache.as_ref().unwrap().seq_len(), 7);
    }

    #[test]
    fn test_truncate() {
        let k = Array::zeros::<f32>(&[1, 4, 16, 8]).unwrap();
        let v = Array::zeros::<f32>(&[1, 4, 16, 8]).unwrap();

        let mut entry = MlxLayerKvCache::new(k, v);
        assert_eq!(entry.seq_len(), 16);

        entry.truncate(10);
        assert_eq!(entry.seq_len(), 10);

        // Can still write into the truncated cache.
        let dk = Array::zeros::<f32>(&[1, 4, 1, 8]).unwrap();
        let dv = Array::zeros::<f32>(&[1, 4, 1, 8]).unwrap();
        let (kv, vv) = entry.update_and_view(&dk, &dv).unwrap();
        mlx_rs::transforms::eval([&kv, &vv]).unwrap();
        assert_eq!(kv.dim(2) as usize, 11);
        assert_eq!(entry.seq_len(), 11);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_truncate_panics_on_larger() {
        let k = Array::zeros::<f32>(&[1, 4, 8, 8]).unwrap();
        let v = Array::zeros::<f32>(&[1, 4, 8, 8]).unwrap();
        let mut entry = MlxLayerKvCache::new(k, v);
        entry.truncate(16); // larger than seq_len=8 → panic
    }
}
