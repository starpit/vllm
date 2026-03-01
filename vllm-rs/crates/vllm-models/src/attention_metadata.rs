// SPDX-License-Identifier: Apache-2.0
//! Attention metadata for cross-request batched forward passes.
//!
//! `AttentionMetadata` describes how tokens from multiple requests are
//! packed into a single flat `[total_tokens, ...]` tensor. The per-request
//! slicing information lets attention layers split Q/K/V back to per-request
//! tensors while all other ops (embedding, projections, norms, MLP, lm_head)
//! run on the full batched tensor.

/// Metadata describing a batch of requests packed into flat tensors.
///
/// All token IDs and positions are concatenated in request order:
/// `[req_0_tokens..., req_1_tokens..., ...]`. The `query_start_loc` offsets
/// let attention layers narrow to per-request slices.
#[derive(Debug, Clone)]
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
}

impl AttentionMetadata {
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
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> AttentionMetadata {
        // Batch: req0 is prefill with 5 tokens, req1 is decode with 1 token.
        AttentionMetadata {
            num_reqs: 2,
            total_tokens: 6,
            query_start_loc: vec![0, 5, 6],
            q_lens: vec![5, 1],
            seq_lens: vec![5, 10],
            block_ids: vec![vec![0, 1], vec![2, 3]],
            tokens_before: vec![0, 9],
            is_prefill: vec![true, false],
            req_ids: vec!["req-0".into(), "req-1".into()],
        }
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

        let all_decode = AttentionMetadata {
            num_reqs: 2,
            total_tokens: 2,
            query_start_loc: vec![0, 1, 2],
            q_lens: vec![1, 1],
            seq_lens: vec![10, 20],
            block_ids: vec![vec![0], vec![1]],
            tokens_before: vec![9, 19],
            is_prefill: vec![false, false],
            req_ids: vec!["a".into(), "b".into()],
        };
        assert!(all_decode.is_all_decode());
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
