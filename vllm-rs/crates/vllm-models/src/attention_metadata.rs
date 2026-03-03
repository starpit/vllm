// SPDX-License-Identifier: Apache-2.0
//! Attention metadata for cross-request batched forward passes.
//!
//! `AttentionMetadata` describes how tokens from multiple requests are
//! packed into a single flat `[total_tokens, ...]` tensor. The per-request
//! slicing information lets attention layers split Q/K/V back to per-request
//! tensors while all other ops (embedding, projections, norms, MLP, lm_head)
//! run on the full batched tensor.

use std::cell::OnceCell;

use candle_core::{Device, Tensor};

/// Metadata describing a batch of requests packed into flat tensors.
///
/// All token IDs and positions are concatenated in request order:
/// `[req_0_tokens..., req_1_tokens..., ...]`. The `query_start_loc` offsets
/// let attention layers narrow to per-request slices.
pub struct AttentionMetadata {
    /// Number of requests in this batch.
    pub num_reqs: usize,
    /// Total number of tokens across all requests.
    pub total_tokens: usize,
    /// Cumulative token offsets, length `num_reqs + 1`.
    /// `query_start_loc[i]` is the start of request `i`'s tokens in the flat tensor.
    /// `query_start_loc[num_reqs]` == `total_tokens`.
    pub query_start_loc: Vec<usize>,
    /// Tokens scheduled for each request (prefill: prompt len, decode: 1).
    pub q_lens: Vec<usize>,
    /// Total context length per request (cached + new tokens).
    pub seq_lens: Vec<usize>,
    /// Per-request paged block IDs.
    pub block_ids: Vec<Vec<usize>>,
    /// Tokens already in block pool per request.
    pub tokens_before: Vec<usize>,
    /// Whether each request is prefill vs decode.
    pub is_prefill: Vec<bool>,
    /// Request IDs in batch order.
    pub req_ids: Vec<String>,

    // -- Lazily-computed GPU tensor caches (avoid per-layer H2D copies) --
    /// `cu_seqlens_q` as a GPU tensor — same across all layers in a step.
    cu_seqlens_q_cache: OnceCell<Tensor>,
    /// `cu_seqlens_k` as a GPU tensor — valid only when there is no sliding window
    /// (i.e. `kv_len == seq_lens[i]` for every request).
    cu_seqlens_k_cache: OnceCell<Tensor>,
}

impl std::fmt::Debug for AttentionMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttentionMetadata")
            .field("num_reqs", &self.num_reqs)
            .field("total_tokens", &self.total_tokens)
            .field("query_start_loc", &self.query_start_loc)
            .field("q_lens", &self.q_lens)
            .field("seq_lens", &self.seq_lens)
            .field("block_ids", &self.block_ids)
            .field("tokens_before", &self.tokens_before)
            .field("is_prefill", &self.is_prefill)
            .field("req_ids", &self.req_ids)
            .finish()
    }
}

impl Clone for AttentionMetadata {
    fn clone(&self) -> Self {
        Self {
            num_reqs: self.num_reqs,
            total_tokens: self.total_tokens,
            query_start_loc: self.query_start_loc.clone(),
            q_lens: self.q_lens.clone(),
            seq_lens: self.seq_lens.clone(),
            block_ids: self.block_ids.clone(),
            tokens_before: self.tokens_before.clone(),
            is_prefill: self.is_prefill.clone(),
            req_ids: self.req_ids.clone(),
            // Don't clone cached GPU tensors — they'll be recomputed if needed.
            cu_seqlens_q_cache: OnceCell::new(),
            cu_seqlens_k_cache: OnceCell::new(),
        }
    }
}

impl AttentionMetadata {
    /// Create a new `AttentionMetadata` with empty GPU caches.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_reqs: usize,
        total_tokens: usize,
        query_start_loc: Vec<usize>,
        q_lens: Vec<usize>,
        seq_lens: Vec<usize>,
        block_ids: Vec<Vec<usize>>,
        tokens_before: Vec<usize>,
        is_prefill: Vec<bool>,
        req_ids: Vec<String>,
    ) -> Self {
        Self {
            num_reqs,
            total_tokens,
            query_start_loc,
            q_lens,
            seq_lens,
            block_ids,
            tokens_before,
            is_prefill,
            req_ids,
            cu_seqlens_q_cache: OnceCell::new(),
            cu_seqlens_k_cache: OnceCell::new(),
        }
    }

    /// Returns `(start_offset, q_len)` for request `req_idx` in the flat tensor.
    pub fn request_slice(&self, req_idx: usize) -> (usize, usize) {
        let start = self.query_start_loc[req_idx];
        let q_len = self.q_lens[req_idx];
        (start, q_len)
    }

    /// True if every request in the batch is a decode (q_len == 1).
    pub fn is_all_decode(&self) -> bool {
        self.is_prefill.iter().all(|&p| !p)
    }

    /// Create metadata for a padded all-decode batch (CUDA graph capture/replay).
    ///
    /// All requests are decode (q_len=1). Padded slots (`actual_bs..padded_bs`)
    /// use dummy `seq_lens=1`, a single dummy block ID (block 0), and
    /// `tokens_before=0`.
    pub fn padded_decode(
        padded_bs: usize,
        actual_bs: usize,
        seq_lens: &[usize],
        block_ids: &[Vec<usize>],
        tokens_before: &[usize],
    ) -> Self {
        debug_assert!(actual_bs <= padded_bs);
        debug_assert_eq!(seq_lens.len(), actual_bs);
        debug_assert_eq!(block_ids.len(), actual_bs);
        debug_assert_eq!(tokens_before.len(), actual_bs);

        let mut full_seq_lens = seq_lens.to_vec();
        let mut full_block_ids = block_ids.to_vec();
        let mut full_tokens_before = tokens_before.to_vec();
        let mut full_is_prefill = vec![false; actual_bs];
        let mut full_req_ids: Vec<String> =
            (0..actual_bs).map(|i| format!("__graph_{i}")).collect();

        // Pad dummy slots.
        for _ in actual_bs..padded_bs {
            full_seq_lens.push(1);
            full_block_ids.push(vec![0]); // dummy block 0
            full_tokens_before.push(0);
            full_is_prefill.push(false);
            full_req_ids.push("__pad__".to_string());
        }

        let query_start_loc: Vec<usize> = (0..=padded_bs).collect();
        let q_lens = vec![1; padded_bs];

        Self::new(
            padded_bs,
            padded_bs, // total_tokens = padded_bs (all decode, 1 token each)
            query_start_loc,
            q_lens,
            full_seq_lens,
            full_block_ids,
            full_tokens_before,
            full_is_prefill,
            full_req_ids,
        )
    }

    /// Lazily compute and cache `cu_seqlens_q` as a GPU u32 tensor.
    ///
    /// This is derived from `query_start_loc` and is the same for every
    /// attention layer in a step.  Caching it avoids re-creating the tensor
    /// (CPU→GPU copy) on every layer call.
    pub fn cu_seqlens_q_gpu(&self, device: &Device) -> candle_core::Result<&Tensor> {
        if let Some(t) = self.cu_seqlens_q_cache.get() {
            return Ok(t);
        }
        let vals: Vec<u32> = self.query_start_loc.iter().map(|&x| x as u32).collect();
        let t = Tensor::from_slice(&vals, vals.len(), device)?;
        let _ = self.cu_seqlens_q_cache.set(t);
        Ok(self.cu_seqlens_q_cache.get().unwrap())
    }

    /// Lazily compute and cache `cu_seqlens_k` as a GPU u32 tensor.
    ///
    /// Without sliding window, the KV sequence length per request equals
    /// `seq_lens[i]`, so `cu_seqlens_k = cumsum(seq_lens)`.  This is
    /// layer-independent and can be cached.
    ///
    /// With sliding window, the actual KV length may differ per layer,
    /// so callers should compute it from the gathered KV lengths instead.
    pub fn cu_seqlens_k_gpu(&self, device: &Device) -> candle_core::Result<&Tensor> {
        if let Some(t) = self.cu_seqlens_k_cache.get() {
            return Ok(t);
        }
        let mut cumlen: Vec<u32> = Vec::with_capacity(self.num_reqs + 1);
        cumlen.push(0);
        for &sl in &self.seq_lens {
            cumlen.push(cumlen.last().unwrap() + sl as u32);
        }
        let t = Tensor::from_slice(&cumlen, cumlen.len(), device)?;
        let _ = self.cu_seqlens_k_cache.set(t);
        Ok(self.cu_seqlens_k_cache.get().unwrap())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> AttentionMetadata {
        // Batch: req0 is prefill with 5 tokens, req1 is decode with 1 token.
        AttentionMetadata::new(
            2,
            6,
            vec![0, 5, 6],
            vec![5, 1],
            vec![5, 10],
            vec![vec![0, 1], vec![2, 3]],
            vec![0, 9],
            vec![true, false],
            vec!["req-0".into(), "req-1".into()],
        )
    }

    #[test]
    fn test_request_slice() {
        let meta = sample_metadata();
        assert_eq!(meta.request_slice(0), (0, 5));
        assert_eq!(meta.request_slice(1), (5, 1));
    }

    #[test]
    fn test_is_all_decode() {
        let meta = sample_metadata();
        assert!(!meta.is_all_decode());

        let all_decode = AttentionMetadata::new(
            2,
            2,
            vec![0, 1, 2],
            vec![1, 1],
            vec![10, 20],
            vec![vec![0], vec![1]],
            vec![9, 19],
            vec![false, false],
            vec!["a".into(), "b".into()],
        );
        assert!(all_decode.is_all_decode());
    }

    #[test]
    fn test_padded_decode() {
        let meta = AttentionMetadata::padded_decode(
            8,
            3,
            &[10, 20, 5],
            &[vec![0, 1], vec![2, 3, 4], vec![5]],
            &[9, 19, 4],
        );
        assert_eq!(meta.num_reqs, 8);
        assert_eq!(meta.total_tokens, 8);
        assert!(meta.is_all_decode());
        assert_eq!(meta.q_lens, vec![1; 8]);
        assert_eq!(meta.query_start_loc, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        // Real requests.
        assert_eq!(meta.seq_lens[0], 10);
        assert_eq!(meta.seq_lens[1], 20);
        assert_eq!(meta.seq_lens[2], 5);
        // Padded slots.
        for i in 3..8 {
            assert_eq!(meta.seq_lens[i], 1);
            assert_eq!(meta.block_ids[i], vec![0]);
            assert_eq!(meta.tokens_before[i], 0);
            assert_eq!(meta.req_ids[i], "__pad__");
        }
    }

    #[test]
    fn test_padded_decode_exact_size() {
        // actual_bs == padded_bs: no padding needed.
        let meta = AttentionMetadata::padded_decode(
            2,
            2,
            &[10, 20],
            &[vec![0], vec![1]],
            &[9, 19],
        );
        assert_eq!(meta.num_reqs, 2);
        assert!(meta.is_all_decode());
    }

    #[test]
    fn test_query_start_loc_consistency() {
        let meta = sample_metadata();
        assert_eq!(meta.query_start_loc.len(), meta.num_reqs + 1);
        assert_eq!(*meta.query_start_loc.last().unwrap(), meta.total_tokens);
        for i in 0..meta.num_reqs {
            assert_eq!(
                meta.query_start_loc[i + 1] - meta.query_start_loc[i],
                meta.q_lens[i]
            );
        }
    }
}
