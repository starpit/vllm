// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! MLX KV cache — pre-allocated per-layer cache using `mlx_rs::Array`.
//!
//! Pre-allocates KV buffers with extra capacity so that decode steps can
//! use `slice_update` (O(1) write) instead of `concatenate` (O(seq) copy).
//! MLX can perform slice_update in-place when the buffer has a single owner
//! (refcount=1), avoiding allocation and data movement entirely.

use mlx_rs::Array;
use mlx_rs::error::Exception;
use mlx_rs::ops::indexing::TryIndexMutOp;
use mlx_rs::ops::indexing::TryIndexOp;

/// Extra capacity added beyond the initial sequence length when pre-allocating.
/// 256 covers typical decode lengths (64–128) with room to spare.
const PREALLOC_HEADROOM: usize = 256;

/// Per-layer KV cache with pre-allocated buffers.
///
/// Keys and values have shape `[1, num_kv_heads, capacity, head_dim]`.
/// Only `seq_len` positions are filled; the rest is padding (zeros).
#[derive(Clone)]
pub struct MlxLayerKvCache {
    k: Array,
    v: Array,
    seq_len: usize,
    capacity: usize,
}

impl MlxLayerKvCache {
    /// Create from pre-existing K/V arrays (e.g., assembled from per-block cache).
    ///
    /// Pre-allocates extra capacity beyond the initial sequence length.
    pub fn from_kv(k: &Array, v: &Array) -> Result<Self, Exception> {
        Self::new(k, v)
    }

    /// Create from initial K/V arrays (typically from prefill).
    ///
    /// Pre-allocates extra capacity beyond the initial sequence length.
    fn new(k: &Array, v: &Array) -> Result<Self, Exception> {
        let seq_len = k.dim(2) as usize;
        let capacity = seq_len + PREALLOC_HEADROOM;

        let padded_k = pad_seq_dim(k, capacity)?;
        let padded_v = pad_seq_dim(v, capacity)?;

        Ok(Self {
            k: padded_k,
            v: padded_v,
            seq_len,
            capacity,
        })
    }

    /// Grow the buffer to at least `min_capacity`.
    fn grow(&mut self, min_capacity: usize) -> Result<(), Exception> {
        let new_capacity = min_capacity + PREALLOC_HEADROOM;
        let new_k = pad_seq_dim(&self.k_view()?, new_capacity)?;
        let new_v = pad_seq_dim(&self.v_view()?, new_capacity)?;
        self.k = new_k;
        self.v = new_v;
        self.capacity = new_capacity;
        Ok(())
    }

    /// Write new K/V tokens into the buffer and return views of the full cache.
    ///
    /// `new_k` and `new_v` have shape `[1, heads, new_len, dim]`.
    /// Returns `(k_view, v_view)` covering `[0..seq_len+new_len]`.
    pub fn update_and_view(
        &mut self,
        new_k: &Array,
        new_v: &Array,
    ) -> Result<(Array, Array), Exception> {
        let new_len = new_k.dim(2) as usize;

        if self.seq_len + new_len > self.capacity {
            self.grow(self.seq_len + new_len)?;
        }

        let start = self.seq_len as i32;
        let end = (self.seq_len + new_len) as i32;

        // Slice update — MLX can do this in-place when refcount=1.
        self.k.try_index_mut((.., .., start..end, ..), new_k)?;
        self.v.try_index_mut((.., .., start..end, ..), new_v)?;

        self.seq_len += new_len;

        Ok((self.k_view()?, self.v_view()?))
    }

    /// Number of cached sequence positions (logical length, not buffer capacity).
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Reset the logical length without touching the buffer.
    ///
    /// Used when cloning a cached KV for a new request that only matches
    /// a prefix of the original sequence.
    pub fn truncate(&mut self, new_seq_len: usize) {
        assert!(new_seq_len <= self.seq_len);
        self.seq_len = new_seq_len;
    }

    /// View of the filled K portion: `[1, heads, seq_len, dim]`.
    fn k_view(&self) -> Result<Array, Exception> {
        self.k.try_index((.., .., ..self.seq_len as i32, ..))
    }

    /// View of the filled V portion: `[1, heads, seq_len, dim]`.
    fn v_view(&self) -> Result<Array, Exception> {
        self.v.try_index((.., .., ..self.seq_len as i32, ..))
    }

    /// Slice K for a range of positions: `[1, heads, end-start, dim]`.
    pub fn k_slice(&self, start: i32, end: i32) -> Result<Array, Exception> {
        self.k.try_index((.., .., start..end, ..))
    }

    /// Slice V for a range of positions: `[1, heads, end-start, dim]`.
    pub fn v_slice(&self, start: i32, end: i32) -> Result<Array, Exception> {
        self.v.try_index((.., .., start..end, ..))
    }
}

/// Pad (or create) an array to have `capacity` along dim 2 (sequence dim).
///
/// Input shape: `[1, heads, current_len, dim]`
/// Output shape: `[1, heads, capacity, dim]`
fn pad_seq_dim(arr: &Array, capacity: usize) -> Result<Array, Exception> {
    let current_len = arr.dim(2) as usize;
    if current_len >= capacity {
        return Ok(arr.clone());
    }
    let pad_len = capacity - current_len;
    let batch = arr.dim(0);
    let heads = arr.dim(1);
    let dim = arr.dim(3);
    let padding =
        mlx_rs::Array::zeros::<f32>(&[batch, heads, pad_len as i32, dim])?.as_dtype(arr.dtype())?;
    mlx_rs::ops::concatenate_axis(&[arr.clone(), padding], 2)
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

/// Update a per-layer KV cache entry and return views for attention.
///
/// On first call (cache=None): pre-allocates a buffer and stores K/V.
/// On subsequent calls: uses slice_update for O(1) writes.
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
            let entry = MlxLayerKvCache::new(new_k, new_v)?;
            let k_view = entry.k_view()?;
            let v_view = entry.v_view()?;
            *cache = Some(entry);
            Ok((k_view, v_view))
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
    fn test_prealloc_kv_cache_prefill() {
        let k = Array::zeros::<f32>(&[1, 4, 8, 16]).unwrap();
        let v = Array::zeros::<f32>(&[1, 4, 8, 16]).unwrap();

        let mut cache: Option<MlxLayerKvCache> = None;
        let (kv, vv) = kv_cache_update(&mut cache, &k, &v).unwrap();
        mlx_rs::transforms::eval([&kv, &vv]).unwrap();

        assert!(cache.is_some());
        assert_eq!(kv.shape(), &[1, 4, 8, 16]);
        assert_eq!(vv.shape(), &[1, 4, 8, 16]);
        assert_eq!(cache.as_ref().unwrap().seq_len, 8);
        assert_eq!(cache.as_ref().unwrap().capacity, 8 + PREALLOC_HEADROOM);
    }

    #[test]
    fn test_prealloc_kv_cache_decode_steps() {
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

        let mut entry = MlxLayerKvCache::new(&k, &v).unwrap();
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
        let mut entry = MlxLayerKvCache::new(&k, &v).unwrap();
        entry.truncate(16); // larger than seq_len=8 → panic
    }
}
