// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Persistent `InputBatch` that maintains pre-allocated buffers across engine
//! steps, eliminating redundant allocations on the decode hot path.
//!
//! Port of the Python V1 `InputBatch` concept: delta-update a dense array of
//! per-request slots instead of rebuilding all model inputs from scratch every
//! step.

use std::collections::HashMap;

use vllm_model::AttentionMetadata;

// ---------------------------------------------------------------------------
// InputBatch
// ---------------------------------------------------------------------------

/// Persistent batch state that survives across engine steps.
///
/// Requests occupy dense slots (0..num_active). Finished requests are
/// swap-removed with the last slot to keep the array compact without gaps.
pub struct InputBatch {
    // --- Slot management ---
    /// req_id → slot index.
    req_id_to_slot: HashMap<String, usize>,
    /// Ordered req_ids, length == num_active.
    req_ids: Vec<String>,

    // --- Per-slot persistent state (indexed by slot) ---
    /// Most recently appended token ID (decode input).
    last_token_ids: Vec<u32>,
    /// Current sequence position (prompt_len + num_generated - 1).
    positions: Vec<u32>,
    /// Number of tokens already written to KV block pool.
    tokens_in_pool: Vec<usize>,
    /// Block table per request (paged KV cache block IDs).
    block_tables: Vec<Vec<usize>>,
    /// Whether this slot is in its first step (prefill).
    is_prefill: Vec<bool>,
    /// Full prompt tokens for prefill slots (consumed after first prepare_inputs).
    prefill_tokens: Vec<Option<Vec<u32>>>,
    /// Position offset for prefill (num_computed_tokens from scheduler).
    prefill_pos_offset: Vec<u32>,

    // --- Reusable per-step buffers (cleared + refilled each step) ---
    flat_token_ids: Vec<u32>,
    flat_positions: Vec<u32>,

    // --- Reusable output buffers (swapped out in prepare_inputs, swapped back via reclaim) ---
    req_inputs_buf: Vec<ReqSlice>,
    query_start_loc_buf: Vec<usize>,
    q_lens_buf: Vec<usize>,
    seq_lens_buf: Vec<usize>,
    block_ids_buf: Vec<Vec<usize>>,
    tokens_before_buf: Vec<usize>,
    is_prefill_buf: Vec<bool>,
    req_ids_buf: Vec<String>,
}

impl Default for InputBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl InputBatch {
    /// Create an empty InputBatch.
    pub fn new() -> Self {
        Self {
            req_id_to_slot: HashMap::new(),
            req_ids: Vec::new(),
            last_token_ids: Vec::new(),
            positions: Vec::new(),
            tokens_in_pool: Vec::new(),
            block_tables: Vec::new(),
            is_prefill: Vec::new(),
            prefill_tokens: Vec::new(),
            prefill_pos_offset: Vec::new(),
            flat_token_ids: Vec::new(),
            flat_positions: Vec::new(),
            req_inputs_buf: Vec::new(),
            query_start_loc_buf: Vec::new(),
            q_lens_buf: Vec::new(),
            seq_lens_buf: Vec::new(),
            block_ids_buf: Vec::new(),
            tokens_before_buf: Vec::new(),
            is_prefill_buf: Vec::new(),
            req_ids_buf: Vec::new(),
        }
    }

    /// Number of active requests.
    pub fn num_active(&self) -> usize {
        self.req_ids.len()
    }

    /// Add a new request (called for scheduled_new_reqs).
    pub fn add_request(
        &mut self,
        req_id: String,
        prompt_token_ids: &[u32],
        block_ids: Vec<usize>,
        num_computed_tokens: u32,
    ) {
        let slot = self.req_ids.len();
        self.req_id_to_slot.insert(req_id.clone(), slot);
        self.req_ids.push(req_id);

        // The last token of the prompt (used if this transitions to decode
        // without an explicit set_last_token call).
        let last_tok = prompt_token_ids.last().copied().unwrap_or(0);
        self.last_token_ids.push(last_tok);

        // Position will be set properly during prepare_inputs for prefill.
        self.positions.push(0);
        // When prefix caching provides computed tokens, those KV entries are
        // already in the block pool — initialize tokens_in_pool accordingly.
        self.tokens_in_pool.push(num_computed_tokens as usize);
        self.block_tables.push(block_ids);
        self.is_prefill.push(true);
        self.prefill_tokens.push(Some(prompt_token_ids.to_vec()));
        self.prefill_pos_offset.push(num_computed_tokens);
    }

    /// Remove a finished request (swap-remove to keep dense packing).
    pub fn remove_request(&mut self, req_id: &str) {
        let Some(slot) = self.req_id_to_slot.remove(req_id) else {
            return;
        };
        let last = self.req_ids.len() - 1;
        if slot != last {
            // Swap the last slot into the removed slot's position.
            self.req_ids.swap(slot, last);
            self.last_token_ids.swap(slot, last);
            self.positions.swap(slot, last);
            self.tokens_in_pool.swap(slot, last);
            self.block_tables.swap(slot, last);
            self.is_prefill.swap(slot, last);
            self.prefill_tokens.swap(slot, last);
            self.prefill_pos_offset.swap(slot, last);

            // Update the swapped request's slot mapping.
            let moved_req_id = self.req_ids[slot].clone();
            self.req_id_to_slot.insert(moved_req_id, slot);
        }
        // Pop the last element (now the removed request).
        self.req_ids.pop();
        self.last_token_ids.pop();
        self.positions.pop();
        self.tokens_in_pool.pop();
        self.block_tables.pop();
        self.is_prefill.pop();
        self.prefill_tokens.pop();
        self.prefill_pos_offset.pop();
    }

    /// Remove all finished requests.
    pub fn remove_finished(&mut self, finished_req_ids: &std::collections::HashSet<String>) {
        // Collect slots to remove (iterate in reverse so swap-remove is safe).
        let mut to_remove: Vec<String> = Vec::new();
        for req_id in finished_req_ids {
            if self.req_id_to_slot.contains_key(req_id) {
                to_remove.push(req_id.clone());
            }
        }
        for req_id in &to_remove {
            self.remove_request(req_id);
        }
    }

    /// Update block table for a cached request that got new blocks.
    pub fn update_blocks(&mut self, req_id: &str, new_block_ids: Vec<usize>) {
        if let Some(&slot) = self.req_id_to_slot.get(req_id) {
            self.block_tables[slot] = new_block_ids;
        }
    }

    /// Get the block table for a request.
    pub fn block_table(&self, req_id: &str) -> Option<&[usize]> {
        self.req_id_to_slot
            .get(req_id)
            .map(|&slot| self.block_tables[slot].as_slice())
    }

    /// Get tokens-in-pool count for a request.
    pub fn tokens_in_pool_for(&self, req_id: &str) -> usize {
        self.req_id_to_slot
            .get(req_id)
            .map(|&slot| self.tokens_in_pool[slot])
            .unwrap_or(0)
    }

    /// Check if a request is known to this batch.
    pub fn contains(&self, req_id: &str) -> bool {
        self.req_id_to_slot.contains_key(req_id)
    }

    /// Lightweight query for the greedy graph fast path.
    ///
    /// Returns `(req_ids, block_tables, tokens_in_pool)` without building
    /// full `PreparedInputs`. This avoids the cost of `prepare_inputs` when
    /// the GPU self-updates all metadata.
    pub fn fast_path_info(&self) -> (&[String], &[Vec<usize>], &[usize]) {
        (&self.req_ids, &self.block_tables, &self.tokens_in_pool)
    }

    /// Token counts for the super-fast graph path's `PendingCommit`.
    ///
    /// The super-fast path is always a pure decode batch (all q_len=1), so each
    /// request contributes exactly 1 input token. This MUST return `vec![1; n]`,
    /// NOT `tokens_in_pool` — using cumulative `tokens_in_pool` would cause
    /// exponential growth when later passed to `commit_step` as `input_token_count`.
    pub fn fast_path_token_counts(&self) -> Vec<usize> {
        vec![1; self.req_ids.len()]
    }

    /// Prepare model inputs for the current step.
    ///
    /// Returns `(req_inputs, attn_meta, batch_block_ids, batch_tokens_before)`
    /// where `req_inputs` contains per-request token_ids, positions, and spec
    /// decode info.
    ///
    /// `spec_decode_tokens` maps req_id → draft tokens for speculative decode.
    pub fn prepare_inputs(
        &mut self,
        spec_decode_tokens: &HashMap<String, Vec<u32>>,
    ) -> PreparedInputs {
        self.flat_token_ids.clear();
        self.flat_positions.clear();

        let num_reqs = self.req_ids.len();

        // Reuse pre-allocated buffers (retain capacity across steps).
        let mut req_inputs = std::mem::take(&mut self.req_inputs_buf);
        req_inputs.clear();
        let mut query_start_loc = std::mem::take(&mut self.query_start_loc_buf);
        query_start_loc.clear();
        let mut q_lens = std::mem::take(&mut self.q_lens_buf);
        q_lens.clear();
        let mut seq_lens = std::mem::take(&mut self.seq_lens_buf);
        seq_lens.clear();
        let mut batch_block_ids = std::mem::take(&mut self.block_ids_buf);
        batch_block_ids.clear();
        let mut batch_tokens_before = std::mem::take(&mut self.tokens_before_buf);
        batch_tokens_before.clear();
        let mut is_prefill_vec = std::mem::take(&mut self.is_prefill_buf);
        is_prefill_vec.clear();
        let mut batch_req_ids = std::mem::take(&mut self.req_ids_buf);
        batch_req_ids.clear();

        let mut offset = 0usize;

        for slot in 0..num_reqs {
            let req_id = &self.req_ids[slot];
            let tb = self.tokens_in_pool[slot];

            if self.is_prefill[slot] {
                // Prefill: emit the full prompt tokens.
                let prompt_tokens = self.prefill_tokens[slot]
                    .as_ref()
                    .expect("prefill slot must have prompt tokens");
                let pos_offset = self.prefill_pos_offset[slot];
                let num_tokens = prompt_tokens.len();

                let token_start = self.flat_token_ids.len();
                self.flat_token_ids.extend_from_slice(prompt_tokens);
                for i in 0..num_tokens as u32 {
                    self.flat_positions.push(pos_offset + i);
                }

                query_start_loc.push(offset);
                q_lens.push(num_tokens);
                seq_lens.push(tb + num_tokens);
                batch_block_ids.push(self.block_tables[slot].clone());
                batch_tokens_before.push(tb);
                is_prefill_vec.push(true);
                batch_req_ids.push(req_id.clone());

                req_inputs.push(ReqSlice {
                    req_id: req_id.clone(),
                    token_start,
                    token_count: num_tokens,
                    spec_token_ids: Vec::new(),
                });

                offset += num_tokens;
            } else {
                // Decode: emit last_token_id + optional spec decode tokens.
                let spec_tokens = spec_decode_tokens.get(req_id).cloned().unwrap_or_default();
                let position = self.positions[slot];

                let token_start = self.flat_token_ids.len();

                if spec_tokens.is_empty() {
                    // Normal single-token decode.
                    self.flat_token_ids.push(self.last_token_ids[slot]);
                    self.flat_positions.push(position);

                    query_start_loc.push(offset);
                    q_lens.push(1);
                    seq_lens.push(tb + 1);
                    batch_block_ids.push(self.block_tables[slot].clone());
                    batch_tokens_before.push(tb);
                    is_prefill_vec.push(false);
                    batch_req_ids.push(req_id.clone());

                    req_inputs.push(ReqSlice {
                        req_id: req_id.clone(),
                        token_start,
                        token_count: 1,
                        spec_token_ids: Vec::new(),
                    });

                    offset += 1;
                } else {
                    // Speculative decode: [last_token, draft_0, ..., draft_K-1].
                    let total = 1 + spec_tokens.len();
                    self.flat_token_ids.push(self.last_token_ids[slot]);
                    self.flat_positions.push(position);
                    for (j, &draft_tok) in spec_tokens.iter().enumerate() {
                        self.flat_token_ids.push(draft_tok);
                        self.flat_positions.push(position + 1 + j as u32);
                    }

                    query_start_loc.push(offset);
                    q_lens.push(total);
                    seq_lens.push(tb + total);
                    batch_block_ids.push(self.block_tables[slot].clone());
                    batch_tokens_before.push(tb);
                    is_prefill_vec.push(false);
                    batch_req_ids.push(req_id.clone());

                    req_inputs.push(ReqSlice {
                        req_id: req_id.clone(),
                        token_start,
                        token_count: total,
                        spec_token_ids: spec_tokens,
                    });

                    offset += total;
                }
            }
        }
        query_start_loc.push(offset);

        let attn_meta = AttentionMetadata::new(
            num_reqs,
            offset,
            query_start_loc,
            q_lens,
            seq_lens,
            batch_block_ids,
            batch_tokens_before,
            is_prefill_vec,
            batch_req_ids,
        );

        PreparedInputs {
            req_inputs,
            flat_token_ids: std::mem::take(&mut self.flat_token_ids),
            flat_positions: std::mem::take(&mut self.flat_positions),
            attn_meta,
        }
    }

    /// Commit step results after sampling.
    ///
    /// For each request: updates positions, tokens_in_pool, clears prefill,
    /// and stores the last sampled token.
    pub fn commit_step(
        &mut self,
        req_id: &str,
        sampled_tokens: &[u32],
        input_token_count: usize,
        was_spec_decode: bool,
    ) {
        let Some(&slot) = self.req_id_to_slot.get(req_id) else {
            return;
        };

        let tb = self.tokens_in_pool[slot];

        // Update tokens_in_pool.
        let new_tokens_in_cache = if was_spec_decode {
            // Spec decode: only accepted tokens get cached (input tokens up to
            // the first rejection). `sampled_tokens.len()` == num_accepted + 1
            // which also equals the number of input tokens correctly in cache
            // (last_token + accepted drafts).
            sampled_tokens.len()
        } else {
            input_token_count
        };
        self.tokens_in_pool[slot] = tb + new_tokens_in_cache;

        // Update position for the next decode step.
        // The next input token (last sampled) sits at position = tokens_in_pool.
        // For normal decode: tokens_in_pool + 1 - 1 = tokens_in_pool. ✓
        // For spec decode: tokens_in_pool (accepted tokens are already counted). ✓
        self.positions[slot] = self.tokens_in_pool[slot] as u32;

        // Store the last sampled token for the next decode step.
        if let Some(&last) = sampled_tokens.last() {
            self.last_token_ids[slot] = last;
        }

        // Clear prefill state.
        if self.is_prefill[slot] {
            self.is_prefill[slot] = false;
            self.prefill_tokens[slot] = None;
        }
    }

    /// Set the last token for a request (after sampling).
    pub fn set_last_token(&mut self, req_id: &str, token_id: u32) {
        if let Some(&slot) = self.req_id_to_slot.get(req_id) {
            self.last_token_ids[slot] = token_id;
        }
    }

    /// Re-enter prefill mode for a request that has more prompt/sequence
    /// tokens to process (chunked prefill continuation).
    ///
    /// Called when a running request still has tokens in `token_buffers` beyond
    /// `tokens_in_pool` — i.e., it is mid-prefill and the scheduler has
    /// allocated new blocks for the next chunk.  `tokens` is the slice to
    /// process this step; `pos_offset` is `num_computed_tokens` (= tokens
    /// already in the KV cache = current `tokens_in_pool`).
    ///
    /// Matches Python InputBatch behaviour: prompt token IDs are stored
    /// persistently and sliced each step via `num_computed_tokens`.
    pub fn set_prefill_continuation(&mut self, req_id: &str, tokens: Vec<u32>, pos_offset: u32) {
        if let Some(&slot) = self.req_id_to_slot.get(req_id) {
            self.is_prefill[slot] = true;
            self.prefill_tokens[slot] = Some(tokens);
            self.prefill_pos_offset[slot] = pos_offset;
        }
    }

    /// Reclaim reusable buffers from a consumed `PreparedInputs`.
    ///
    /// Call this at the end of `execute_model` to return Vec capacity back to
    /// `InputBatch`, avoiding heap allocations on the next `prepare_inputs`.
    pub fn reclaim_buffers(&mut self, prepared: PreparedInputs) {
        self.flat_token_ids = prepared.flat_token_ids;
        self.flat_positions = prepared.flat_positions;
        self.req_inputs_buf = prepared.req_inputs;
        // Reclaim AttentionMetadata buffers.
        let meta = prepared.attn_meta;
        self.query_start_loc_buf = meta.query_start_loc;
        self.q_lens_buf = meta.q_lens;
        self.seq_lens_buf = meta.seq_lens;
        self.block_ids_buf = meta.block_ids;
        self.tokens_before_buf = meta.tokens_before;
        self.is_prefill_buf = meta.is_prefill;
        self.req_ids_buf = meta.req_ids;
    }
}

// ---------------------------------------------------------------------------
// PreparedInputs — output of prepare_inputs()
// ---------------------------------------------------------------------------

/// Output of `InputBatch::prepare_inputs()`.
///
/// All data is owned so the caller can freely borrow other fields of the
/// parent struct while using these results.
pub struct PreparedInputs {
    /// Per-request slicing info.
    pub req_inputs: Vec<ReqSlice>,
    /// Flat token IDs for all requests (moved from InputBatch).
    pub flat_token_ids: Vec<u32>,
    /// Flat positions for all requests (moved from InputBatch).
    pub flat_positions: Vec<u32>,
    /// Attention metadata (also owns block_ids and tokens_before).
    pub attn_meta: AttentionMetadata,
}

/// Per-request slice info within the flat tensors.
pub struct ReqSlice {
    /// Request ID.
    pub req_id: String,
    /// Start index in flat_token_ids / flat_positions.
    pub token_start: usize,
    /// Number of tokens for this request.
    pub token_count: usize,
    /// Speculative decode draft tokens (empty for normal decode/prefill).
    pub spec_token_ids: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Greedy rejection sampling (pure, no GPU dependency)
// ---------------------------------------------------------------------------

/// Result of greedy rejection sampling for a single request.
#[derive(Debug, Clone, PartialEq)]
pub struct RejectionResult {
    /// Accepted token IDs (target argmax values). Length is 1..=num_drafts+1.
    /// On full acceptance: K target-verified tokens + 1 bonus token.
    /// On partial acceptance: M target-verified tokens + 1 recovered token.
    /// On first rejection: 1 recovered token.
    pub accepted_tokens: Vec<u32>,
    /// Number of draft tokens that were accepted (0..=num_drafts).
    pub num_accepted_drafts: usize,
}

/// Perform greedy rejection sampling for one request.
///
/// Given `target_ids` (argmax of the model's logits at each position) and
/// `draft_token_ids` (proposed draft tokens), accept the longest prefix of
/// matching drafts and return the accepted tokens.
///
/// Layout:
/// - `target_ids[0]` = argmax at the real token position (verifies draft[0])
/// - `target_ids[i]` = argmax at draft[i-1] position (verifies draft[i])
/// - `target_ids[K]` = bonus token (only used if all K drafts accepted)
///
/// This matches Python vLLM's `_rejection_sample_kernel` for greedy decoding.
pub fn greedy_rejection_sample(target_ids: &[u32], draft_token_ids: &[u32]) -> RejectionResult {
    if draft_token_ids.is_empty() {
        // Normal request: no drafts, just 1 token.
        return RejectionResult {
            accepted_tokens: vec![target_ids[0]],
            num_accepted_drafts: 0,
        };
    }

    let mut accepted = Vec::with_capacity(draft_token_ids.len() + 1);

    for i in 0..draft_token_ids.len() {
        let target = target_ids[i];
        accepted.push(target);
        if target != draft_token_ids[i] {
            // Mismatch: target is the recovered token. Stop.
            return RejectionResult {
                num_accepted_drafts: i,
                accepted_tokens: accepted,
            };
        }
    }

    // All drafts accepted — append bonus token from the last logit position.
    let bonus = target_ids[draft_token_ids.len()];
    accepted.push(bonus);
    RejectionResult {
        num_accepted_drafts: draft_token_ids.len(),
        accepted_tokens: accepted,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_remove_request() {
        let mut batch = InputBatch::new();
        assert_eq!(batch.num_active(), 0);

        batch.add_request("r1".into(), &[10, 20, 30], vec![0, 1], 0);
        assert_eq!(batch.num_active(), 1);
        assert!(batch.contains("r1"));
        assert_eq!(batch.block_table("r1"), Some(&[0, 1][..]));

        batch.add_request("r2".into(), &[40, 50], vec![2], 0);
        assert_eq!(batch.num_active(), 2);

        batch.remove_request("r1");
        assert_eq!(batch.num_active(), 1);
        assert!(!batch.contains("r1"));
        assert!(batch.contains("r2"));
        // r2 should have been swapped into slot 0.
        assert_eq!(batch.block_table("r2"), Some(&[2][..]));

        batch.remove_request("r2");
        assert_eq!(batch.num_active(), 0);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10], vec![0], 0);
        batch.remove_request("nonexistent"); // should not panic
        assert_eq!(batch.num_active(), 1);
    }

    #[test]
    fn test_swap_remove_preserves_mapping() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10], vec![0], 0);
        batch.add_request("r2".into(), &[20], vec![1], 0);
        batch.add_request("r3".into(), &[30], vec![2], 0);

        // Remove r1 (slot 0) — r3 should move to slot 0.
        batch.remove_request("r1");
        assert_eq!(batch.num_active(), 2);
        assert!(batch.contains("r2"));
        assert!(batch.contains("r3"));
        assert_eq!(batch.block_table("r3"), Some(&[2][..]));
        assert_eq!(batch.block_table("r2"), Some(&[1][..]));
    }

    #[test]
    fn test_prepare_inputs_prefill() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30], vec![0, 1], 0);

        let spec = HashMap::new();
        let prepared = batch.prepare_inputs(&spec);

        assert_eq!(prepared.flat_token_ids, &[10, 20, 30]);
        assert_eq!(prepared.flat_positions, &[0, 1, 2]);
        assert_eq!(prepared.req_inputs.len(), 1);
        assert_eq!(prepared.req_inputs[0].token_count, 3);
        assert!(prepared.attn_meta.is_prefill[0]);
        assert_eq!(prepared.attn_meta.num_reqs, 1);
        assert_eq!(prepared.attn_meta.total_tokens, 3);
    }

    #[test]
    fn test_prepare_inputs_prefill_with_offset() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30], vec![0], 5);

        let spec = HashMap::new();
        let prepared = batch.prepare_inputs(&spec);

        assert_eq!(prepared.flat_token_ids, &[10, 20, 30]);
        assert_eq!(prepared.flat_positions, &[5, 6, 7]);
    }

    #[test]
    fn test_prefill_to_decode_transition() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30], vec![0, 1], 0);

        // First step: prefill.
        let spec = HashMap::new();
        let prepared = batch.prepare_inputs(&spec);
        assert_eq!(prepared.req_inputs[0].token_count, 3);
        assert!(prepared.attn_meta.is_prefill[0]);

        // Commit: 3 prompt tokens cached, sampled token 99.
        batch.commit_step("r1", &[99], 3, false);

        // Second step: should be decode.
        let prepared = batch.prepare_inputs(&spec);
        assert_eq!(prepared.flat_token_ids, &[99]);
        assert_eq!(prepared.req_inputs[0].token_count, 1);
        assert!(!prepared.attn_meta.is_prefill[0]);
        assert_eq!(prepared.attn_meta.tokens_before[0], 3);
    }

    #[test]
    fn test_decode_multiple_steps() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20], vec![0], 0);

        let spec = HashMap::new();

        // Prefill.
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[30], 2, false);

        // Decode step 1.
        let prepared = batch.prepare_inputs(&spec);
        assert_eq!(prepared.flat_token_ids, &[30]);
        assert_eq!(prepared.flat_positions, &[2]); // position 2 = after tokens 0,1 in pool + sampled
        batch.commit_step("r1", &[40], 1, false);

        // Decode step 2.
        let prepared = batch.prepare_inputs(&spec);
        assert_eq!(prepared.flat_token_ids, &[40]);
        assert_eq!(prepared.flat_positions, &[3]);
    }

    #[test]
    fn test_mixed_prefill_and_decode() {
        let mut batch = InputBatch::new();
        // r1 already past prefill.
        batch.add_request("r1".into(), &[10, 20], vec![0], 0);
        let spec = HashMap::new();
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[30], 2, false);

        // Add r2 as new prefill while r1 is decoding.
        batch.add_request("r2".into(), &[50, 60, 70], vec![1, 2], 0);

        let prepared = batch.prepare_inputs(&spec);
        assert_eq!(prepared.attn_meta.num_reqs, 2);
        // r1 is decode (1 token), r2 is prefill (3 tokens).
        assert_eq!(prepared.attn_meta.total_tokens, 4);

        // Find r1 and r2 in the results.
        let r1_idx = prepared
            .req_inputs
            .iter()
            .position(|r| r.req_id == "r1")
            .unwrap();
        let r2_idx = prepared
            .req_inputs
            .iter()
            .position(|r| r.req_id == "r2")
            .unwrap();

        assert_eq!(prepared.req_inputs[r1_idx].token_count, 1);
        assert_eq!(prepared.req_inputs[r2_idx].token_count, 3);
        assert!(!prepared.attn_meta.is_prefill[r1_idx]);
        assert!(prepared.attn_meta.is_prefill[r2_idx]);
    }

    #[test]
    fn test_spec_decode_tokens() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10], vec![0], 0);
        let spec = HashMap::new();
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[20], 1, false);

        // Decode with spec tokens.
        let mut spec = HashMap::new();
        spec.insert("r1".to_string(), vec![30, 40]);

        let prepared = batch.prepare_inputs(&spec);
        // Should have 3 tokens: [last_token=20, draft_30, draft_40].
        assert_eq!(prepared.req_inputs[0].token_count, 3);
        assert_eq!(prepared.req_inputs[0].spec_token_ids, vec![30, 40]);
        assert_eq!(prepared.flat_token_ids, &[20, 30, 40]);
    }

    #[test]
    fn test_update_blocks() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10], vec![0], 0);
        assert_eq!(batch.block_table("r1"), Some(&[0][..]));

        batch.update_blocks("r1", vec![0, 1, 2]);
        assert_eq!(batch.block_table("r1"), Some(&[0, 1, 2][..]));
    }

    #[test]
    fn test_remove_finished() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10], vec![0], 0);
        batch.add_request("r2".into(), &[20], vec![1], 0);
        batch.add_request("r3".into(), &[30], vec![2], 0);

        let mut finished = std::collections::HashSet::new();
        finished.insert("r1".to_string());
        finished.insert("r3".to_string());

        batch.remove_finished(&finished);
        assert_eq!(batch.num_active(), 1);
        assert!(batch.contains("r2"));
    }

    #[test]
    fn test_tokens_in_pool_tracking() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30], vec![0], 0);

        assert_eq!(batch.tokens_in_pool_for("r1"), 0);

        let spec = HashMap::new();
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[40], 3, false);
        assert_eq!(batch.tokens_in_pool_for("r1"), 3);

        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[50], 1, false);
        assert_eq!(batch.tokens_in_pool_for("r1"), 4);
    }

    #[test]
    fn test_spec_decode_commit() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10], vec![0], 0);
        let spec = HashMap::new();
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[20], 1, false);
        assert_eq!(batch.tokens_in_pool_for("r1"), 1);

        // Spec decode: 3 tokens input, 2 accepted.
        batch.commit_step("r1", &[30, 40], 3, true);
        // tokens_in_pool should increase by sampled.len() (2), not input count (3).
        assert_eq!(batch.tokens_in_pool_for("r1"), 3);
    }

    #[test]
    fn test_spec_decode_multi_token_commit() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30], vec![0], 0);
        let spec = HashMap::new();

        // Prefill.
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[40], 3, false);
        assert_eq!(batch.tokens_in_pool_for("r1"), 3);

        // Decode step: normal.
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[50], 1, false);
        assert_eq!(batch.tokens_in_pool_for("r1"), 4);

        // Spec decode: 4 input tokens [50, d0, d1, d2], all 3 accepted + bonus.
        // sampled = [d0, d1, d2, bonus] = 4 tokens.
        batch.commit_step("r1", &[60, 70, 80, 90], 4, true);
        // tokens_in_pool += 4 (the 4 input tokens are all correctly cached).
        assert_eq!(batch.tokens_in_pool_for("r1"), 8);

        // Next decode should use the last sampled token (90) at position 8.
        let prepared = batch.prepare_inputs(&spec);
        assert_eq!(prepared.flat_token_ids, &[90]);
        assert_eq!(prepared.flat_positions, &[8]);
    }

    #[test]
    fn test_spec_decode_partial_accept_commit() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20], vec![0], 0);
        let spec = HashMap::new();

        // Prefill.
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[30], 2, false);
        assert_eq!(batch.tokens_in_pool_for("r1"), 2);

        // Spec decode: 4 input tokens [30, d0, d1, d2], only 1 accepted.
        // sampled = [d0, recovered] = 2 tokens.
        batch.commit_step("r1", &[40, 50], 4, true);
        // tokens_in_pool += 2 (30 and d0 are correctly cached).
        assert_eq!(batch.tokens_in_pool_for("r1"), 4);

        // Next decode should use the last sampled token (50) at position 4.
        let prepared = batch.prepare_inputs(&spec);
        assert_eq!(prepared.flat_token_ids, &[50]);
        assert_eq!(prepared.flat_positions, &[4]);
    }

    #[test]
    fn test_query_start_loc_consistency() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20], vec![0], 0);
        batch.add_request("r2".into(), &[30, 40, 50], vec![1], 0);

        let spec = HashMap::new();
        let prepared = batch.prepare_inputs(&spec);
        let meta = &prepared.attn_meta;

        assert_eq!(meta.query_start_loc.len(), meta.num_reqs + 1);
        assert_eq!(*meta.query_start_loc.last().unwrap(), meta.total_tokens);
        for i in 0..meta.num_reqs {
            assert_eq!(
                meta.query_start_loc[i + 1] - meta.query_start_loc[i],
                meta.q_lens[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Greedy rejection sampling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rejection_no_drafts() {
        // Normal request: no draft tokens, single target token.
        let result = greedy_rejection_sample(&[42], &[]);
        assert_eq!(result.accepted_tokens, vec![42]);
        assert_eq!(result.num_accepted_drafts, 0);
    }

    #[test]
    fn test_rejection_all_accept() {
        // All 3 draft tokens match target argmax → 3 accepted + 1 bonus = 4 tokens.
        // Drafts:     [10, 20, 30]
        // Target IDs: [10, 20, 30, 99]  (99 = bonus)
        let result = greedy_rejection_sample(&[10, 20, 30, 99], &[10, 20, 30]);
        assert_eq!(result.accepted_tokens, vec![10, 20, 30, 99]);
        assert_eq!(result.num_accepted_drafts, 3);
    }

    #[test]
    fn test_rejection_first_reject() {
        // First draft doesn't match → output 1 recovered token.
        // Drafts:     [10, 20, 30]
        // Target IDs: [77, ...]  (77 != 10)
        let result = greedy_rejection_sample(&[77, 20, 30, 99], &[10, 20, 30]);
        assert_eq!(result.accepted_tokens, vec![77]);
        assert_eq!(result.num_accepted_drafts, 0);
    }

    #[test]
    fn test_rejection_partial_accept_middle() {
        // First 2 drafts match, 3rd doesn't → 2 accepted + 1 recovered = 3 tokens.
        // Drafts:     [10, 20, 30]
        // Target IDs: [10, 20, 55, 99]  (55 != 30)
        let result = greedy_rejection_sample(&[10, 20, 55, 99], &[10, 20, 30]);
        assert_eq!(result.accepted_tokens, vec![10, 20, 55]);
        assert_eq!(result.num_accepted_drafts, 2);
    }

    #[test]
    fn test_rejection_partial_accept_second() {
        // First draft matches, second doesn't → 1 accepted + 1 recovered = 2 tokens.
        // Drafts:     [10, 20, 30]
        // Target IDs: [10, 88, 30, 99]  (88 != 20)
        let result = greedy_rejection_sample(&[10, 88, 30, 99], &[10, 20, 30]);
        assert_eq!(result.accepted_tokens, vec![10, 88]);
        assert_eq!(result.num_accepted_drafts, 1);
    }

    #[test]
    fn test_rejection_single_draft_accept() {
        // Single draft token, matches.
        let result = greedy_rejection_sample(&[10, 99], &[10]);
        assert_eq!(result.accepted_tokens, vec![10, 99]);
        assert_eq!(result.num_accepted_drafts, 1);
    }

    #[test]
    fn test_rejection_single_draft_reject() {
        // Single draft token, doesn't match.
        let result = greedy_rejection_sample(&[77, 99], &[10]);
        assert_eq!(result.accepted_tokens, vec![77]);
        assert_eq!(result.num_accepted_drafts, 0);
    }

    #[test]
    fn test_rejection_five_drafts_all_accept() {
        // 5 drafts, all match → 5 accepted + 1 bonus = 6 tokens.
        let drafts = vec![1, 2, 3, 4, 5];
        let targets = vec![1, 2, 3, 4, 5, 99];
        let result = greedy_rejection_sample(&targets, &drafts);
        assert_eq!(result.accepted_tokens, vec![1, 2, 3, 4, 5, 99]);
        assert_eq!(result.num_accepted_drafts, 5);
    }

    #[test]
    fn test_rejection_five_drafts_last_reject() {
        // 5 drafts, last one doesn't match → 4 accepted + 1 recovered = 5 tokens.
        let drafts = vec![1, 2, 3, 4, 5];
        let targets = vec![1, 2, 3, 4, 77, 99];
        let result = greedy_rejection_sample(&targets, &drafts);
        assert_eq!(result.accepted_tokens, vec![1, 2, 3, 4, 77]);
        assert_eq!(result.num_accepted_drafts, 4);
    }

    /// Regression test for the super-fast graph path deferred commit bug.
    ///
    /// Simulates the deferred commit pattern from FerriteWorker's super-fast
    /// graph path. Each iteration:
    ///   1. Capture `token_counts` via `fast_path_token_counts()` BEFORE
    ///      resolving the previous step's pending commit.
    ///   2. Resolve the previous step with its captured `token_counts`.
    ///   3. Store the new `token_counts` for the next iteration.
    ///
    /// The old code used `tokens_in_pool` instead of `fast_path_token_counts()`,
    /// causing Fibonacci-like exponential growth of `tokens_in_pool`.
    /// With the fix, `fast_path_token_counts()` returns `[1; n]` and growth
    /// is linear (exactly +1 per step).
    #[test]
    fn test_fast_path_token_counts_linear_growth() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30], vec![0, 1], 0);
        let spec = HashMap::new();

        // Step 1: prefill.
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[40], 3, false);
        assert_eq!(batch.tokens_in_pool_for("r1"), 3);

        // Step 2: first graph decode (normal path). Deferred, token_count=1.
        let mut pending_tc = vec![1usize];

        // Steps 3..20: super-fast path loop.
        for step in 3..=20 {
            // Capture token_counts BEFORE resolving the previous pending commit.
            // This is the line that was buggy: old code did tokens_in_pool.to_vec().
            let new_pending_tc = batch.fast_path_token_counts();
            assert_eq!(
                new_pending_tc,
                vec![1],
                "fast_path_token_counts must always be [1]"
            );

            // Resolve previous step's deferred commit.
            batch.commit_step("r1", &[40 + step], pending_tc[0], false);

            // tokens_in_pool should be exactly: prompt_len + (step - 2)
            // because we've resolved (step - 2) decode commits so far.
            let expected = 3 + (step - 2) as usize;
            assert_eq!(
                batch.tokens_in_pool_for("r1"),
                expected,
                "step {step}: tokens_in_pool should be {expected} (linear), got {}",
                batch.tokens_in_pool_for("r1")
            );

            pending_tc = new_pending_tc;
        }

        // After 18 super-fast steps, tokens_in_pool should be 3 + 18 = 21.
        assert_eq!(batch.tokens_in_pool_for("r1"), 21);
    }

    #[test]
    fn test_rejection_output_length_invariants() {
        // For K drafts:
        //   - All accept: output length = K + 1
        //   - First reject: output length = 1
        //   - M accepted (0 < M < K): output length = M + 1
        for k in 1..=8 {
            let drafts: Vec<u32> = (1..=k).collect();

            // All accept
            let mut targets: Vec<u32> = (1..=k).collect();
            targets.push(99); // bonus
            let result = greedy_rejection_sample(&targets, &drafts);
            assert_eq!(
                result.accepted_tokens.len(),
                k as usize + 1,
                "all-accept with {k} drafts should produce {0} tokens",
                k + 1
            );
            assert_eq!(result.num_accepted_drafts, k as usize);

            // First reject
            let mut targets: Vec<u32> = vec![0]; // mismatch (draft[0] = 1)
            targets.extend(2..=k);
            targets.push(99);
            let result = greedy_rejection_sample(&targets, &drafts);
            assert_eq!(
                result.accepted_tokens.len(),
                1,
                "first-reject with {k} drafts should produce 1 token"
            );
            assert_eq!(result.num_accepted_drafts, 0);
        }
    }

    #[test]
    fn test_rejection_mixed_batch_simulation() {
        // Simulate a mixed batch: some requests have drafts, some don't.
        struct ReqInput {
            target_ids: Vec<u32>,
            draft_ids: Vec<u32>,
        }

        let batch = [
            // Normal request (no drafts)
            ReqInput {
                target_ids: vec![42],
                draft_ids: vec![],
            },
            // Spec decode, all accept (3 drafts)
            ReqInput {
                target_ids: vec![10, 20, 30, 99],
                draft_ids: vec![10, 20, 30],
            },
            // Spec decode, partial accept (5 drafts, first 2 match)
            ReqInput {
                target_ids: vec![1, 2, 77, 4, 5, 99],
                draft_ids: vec![1, 2, 3, 4, 5],
            },
            // Normal request (no drafts)
            ReqInput {
                target_ids: vec![55],
                draft_ids: vec![],
            },
            // Spec decode, first reject (2 drafts)
            ReqInput {
                target_ids: vec![88, 20, 99],
                draft_ids: vec![10, 20],
            },
        ];

        let results: Vec<_> = batch
            .iter()
            .map(|r| greedy_rejection_sample(&r.target_ids, &r.draft_ids))
            .collect();

        // Normal: 1 token
        assert_eq!(results[0].accepted_tokens, vec![42]);
        assert_eq!(results[0].num_accepted_drafts, 0);

        // All accept: 4 tokens (3 + bonus)
        assert_eq!(results[1].accepted_tokens, vec![10, 20, 30, 99]);
        assert_eq!(results[1].num_accepted_drafts, 3);

        // Partial: 3 tokens (2 accepted + recovered)
        assert_eq!(results[2].accepted_tokens, vec![1, 2, 77]);
        assert_eq!(results[2].num_accepted_drafts, 2);

        // Normal: 1 token
        assert_eq!(results[3].accepted_tokens, vec![55]);
        assert_eq!(results[3].num_accepted_drafts, 0);

        // First reject: 1 token
        assert_eq!(results[4].accepted_tokens, vec![88]);
        assert_eq!(results[4].num_accepted_drafts, 0);
    }

    #[test]
    fn test_rejection_bonus_token_differs_from_drafts() {
        // Verify the bonus token comes from target_ids[K], not from drafts.
        let drafts = vec![10, 20];
        let targets = vec![10, 20, 777]; // bonus = 777
        let result = greedy_rejection_sample(&targets, &drafts);
        assert_eq!(result.accepted_tokens, vec![10, 20, 777]);
        assert_eq!(*result.accepted_tokens.last().unwrap(), 777);
    }

    #[test]
    fn test_rejection_recovered_token_is_target_not_draft() {
        // On rejection at position i, output target_ids[i] (not draft_ids[i]).
        let drafts = vec![10, 20, 30];
        let targets = vec![10, 55, 30, 99]; // reject at position 1: target=55, draft=20
        let result = greedy_rejection_sample(&targets, &drafts);
        assert_eq!(result.accepted_tokens, vec![10, 55]);
        assert_eq!(result.accepted_tokens[1], 55); // recovered token is target, not draft (20)
    }

    #[test]
    fn test_rejection_num_accepted_plus_output_len() {
        // Invariant: accepted_tokens.len() = num_accepted_drafts + 1
        for num_drafts in 1u32..=6 {
            let drafts: Vec<u32> = (1..=num_drafts).collect();

            // Test every possible rejection point
            for reject_at in 0..=num_drafts {
                let mut targets: Vec<u32> = (1..=num_drafts).collect();
                targets.push(99); // bonus
                if reject_at < num_drafts {
                    targets[reject_at as usize] = 0; // force mismatch
                }

                let result = greedy_rejection_sample(&targets, &drafts);
                assert_eq!(
                    result.accepted_tokens.len(),
                    result.num_accepted_drafts + 1,
                    "invariant: len = num_accepted + 1 (drafts={num_drafts}, reject_at={reject_at})"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Pipeline parallelism: set_last_token overrides commit_step's dummy token
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_last_token_overrides_commit_step() {
        // Simulates PP non-last stage: commit_step with dummy 0, then
        // set_last_token with the real sampled token from the scheduler.
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30], vec![0, 1], 0);

        // Simulate prefill → decode transition.
        let no_spec = std::collections::HashMap::new();
        let prepared = batch.prepare_inputs(&no_spec);
        assert_eq!(prepared.flat_token_ids, vec![10, 20, 30]);
        batch.commit_step("r1", &[99], 3, false); // normal commit with real token

        // Verify decode uses token 99.
        let prepared2 = batch.prepare_inputs(&no_spec);
        assert_eq!(prepared2.flat_token_ids, vec![99]);
        batch.reclaim_buffers(prepared2);

        // Now simulate PP non-last stage: commit with dummy 0.
        batch.commit_step("r1", &[0], 1, false);

        // Without fix: next decode would use token 0.
        // With fix: set_last_token overrides it.
        batch.set_last_token("r1", 42);

        let prepared3 = batch.prepare_inputs(&no_spec);
        assert_eq!(
            prepared3.flat_token_ids,
            vec![42],
            "set_last_token must override the dummy 0 from commit_step"
        );
        batch.reclaim_buffers(prepared3);
    }

    #[test]
    fn test_set_last_token_multiple_requests() {
        // Multiple requests in PP mode, each getting different real tokens.
        let mut batch = InputBatch::new();
        let no_spec = std::collections::HashMap::new();
        batch.add_request("r1".into(), &[10, 20], vec![0, 1], 0);
        batch.add_request("r2".into(), &[30, 40], vec![2, 3], 0);

        let _ = batch.prepare_inputs(&no_spec);
        batch.commit_step("r1", &[100], 2, false);
        batch.commit_step("r2", &[200], 2, false);

        // Decode step: commit with dummy 0 (PP non-last stage).
        let p2 = batch.prepare_inputs(&no_spec);
        batch.reclaim_buffers(p2);
        batch.commit_step("r1", &[0], 1, false);
        batch.commit_step("r2", &[0], 1, false);

        // Override with real tokens.
        batch.set_last_token("r1", 101);
        batch.set_last_token("r2", 201);

        let prepared = batch.prepare_inputs(&no_spec);
        // Both requests should use their real tokens.
        assert!(
            prepared.flat_token_ids.contains(&101),
            "r1 should embed token 101, got {:?}",
            prepared.flat_token_ids
        );
        assert!(
            prepared.flat_token_ids.contains(&201),
            "r2 should embed token 201, got {:?}",
            prepared.flat_token_ids
        );
        batch.reclaim_buffers(prepared);
    }

    #[test]
    fn test_set_last_token_nonexistent_request_is_noop() {
        let mut batch = InputBatch::new();
        // Should not panic.
        batch.set_last_token("nonexistent", 42);
    }

    // -----------------------------------------------------------------------
    // Preemption / resumption tests
    //
    // These tests directly model the invariants that FerriteWorker must maintain
    // during KV cache preemption and resumption.  The root bug was:
    //   add_request(req_id, all_tokens, blocks_for_scheduled_chunk)
    // where len(all_tokens) > blocks_for_scheduled_chunk * block_size, causing
    // seq_lens > available_blocks → slot_mapping = -1 → CUDA fault.
    // -----------------------------------------------------------------------

    /// Helper: simulate the invariant FerriteWorker must uphold when re-adding a
    /// resumed request.  Returns true if block coverage is sufficient.
    fn blocks_cover_tokens(tokens: &[u32], block_ids: &[usize], block_size: usize) -> bool {
        if tokens.is_empty() {
            return true;
        }
        let needed = tokens.len().div_ceil(block_size);
        block_ids.len() >= needed
    }

    /// After preemption, remove_request leaves the batch without the request.
    #[test]
    fn test_preemption_removes_from_batch() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30], vec![0, 1], 0);
        batch.add_request("r2".into(), &[40, 50], vec![2], 0);
        assert_eq!(batch.num_active(), 2);

        // Preempt r1.
        batch.remove_request("r1");
        assert_eq!(batch.num_active(), 1);
        assert!(!batch.contains("r1"));
        assert!(batch.contains("r2"));
    }

    /// A preempted request that is re-added as prefill must have is_prefill=true
    /// and tokens_in_pool initialized from num_computed.
    #[test]
    fn test_resumed_request_is_prefill() {
        let mut batch = InputBatch::new();
        // r1 runs a full prefill+decode cycle.
        batch.add_request("r1".into(), &[10, 20, 30], vec![0, 1], 0);
        let spec = HashMap::new();
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[99], 3, false);
        // r1 is now in decode mode, tokens_in_pool=3.
        assert_eq!(batch.tokens_in_pool_for("r1"), 3);

        // Simulate preemption: remove.
        batch.remove_request("r1");
        assert!(!batch.contains("r1"));

        // Simulate resumption: re-add as fresh prefill.
        // Block coverage: 3 tokens need 2 blocks at block_size=2.
        batch.add_request("r1".into(), &[10, 20, 30], vec![5, 6], 0);
        assert!(batch.contains("r1"));
        assert_eq!(batch.tokens_in_pool_for("r1"), 0); // reset to 0

        // prepare_inputs should treat r1 as prefill.
        let prepared = batch.prepare_inputs(&spec);
        let r1_idx = prepared
            .req_inputs
            .iter()
            .position(|r| r.req_id == "r1")
            .unwrap();
        assert!(
            prepared.attn_meta.is_prefill[r1_idx],
            "resumed request must be scheduled as prefill"
        );
        assert_eq!(prepared.req_inputs[r1_idx].token_count, 3);
    }

    /// Resumption must not call commit_step on behalf of the preempted/resumed
    /// slot — the slot should be clean (no phantom tokens_in_pool growth).
    #[test]
    fn test_resumed_tokens_in_pool_is_zero_from_scratch() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30, 40, 50], vec![0, 1, 2], 0);
        let spec = HashMap::new();
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[99], 5, false);
        // Decode step 1.
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[100], 1, false);
        // Decode step 2.
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[101], 1, false);
        assert_eq!(batch.tokens_in_pool_for("r1"), 7);

        // Preempt.
        batch.remove_request("r1");

        // Resume from scratch (all KV evicted, num_computed=0).
        batch.add_request("r1".into(), &[10, 20, 30], vec![0, 1], 0);
        assert_eq!(
            batch.tokens_in_pool_for("r1"),
            0,
            "resumed-from-scratch must have tokens_in_pool=0"
        );
    }

    /// Resume with partial prefix cache: num_computed > 0 means KV for the
    /// first num_computed tokens is still valid.
    #[test]
    fn test_resumed_with_partial_prefix_cache() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30, 40], vec![0, 1], 2);
        // tokens_in_pool = 2 (from prefix cache)
        assert_eq!(batch.tokens_in_pool_for("r1"), 2);

        let spec = HashMap::new();
        let prepared = batch.prepare_inputs(&spec);
        // Positions start at 2 (prefix cache provides tokens 0,1).
        assert_eq!(prepared.flat_positions, &[2, 3, 4, 5]);
    }

    /// Key invariant: blocks must cover ALL tokens passed to add_request.
    /// block_size=16, 48 tokens need 3 blocks; providing 2 blocks is unsafe.
    #[test]
    fn test_block_coverage_invariant() {
        let block_size = 16;
        // 32 tokens, 2 blocks → exactly covered.
        let tokens_32: Vec<u32> = (0u32..32).collect();
        let blocks_2 = vec![0, 1];
        assert!(
            blocks_cover_tokens(&tokens_32, &blocks_2, block_size),
            "32 tokens with 2 blocks of size 16 should be covered"
        );

        // 33 tokens, 2 blocks → NOT covered (needs 3).
        let tokens_33: Vec<u32> = (0u32..33).collect();
        assert!(
            !blocks_cover_tokens(&tokens_33, &blocks_2, block_size),
            "33 tokens with 2 blocks of size 16 must NOT be covered"
        );

        // 33 tokens, 3 blocks → covered.
        let blocks_3 = vec![0, 1, 2];
        assert!(
            blocks_cover_tokens(&tokens_33, &blocks_3, block_size),
            "33 tokens with 3 blocks of size 16 should be covered"
        );
    }

    /// Simulate the exact FerriteWorker bug: full token_buffers passed to
    /// add_request but blocks only cover a scheduled chunk.
    ///
    /// This tests the invariant that was VIOLATED before the fix:
    ///   token_buffers = prompt(10) + output(20) = 30 tokens
    ///   num_scheduled = 16 tokens (chunked prefill)
    ///   new_block_ids covers 16 tokens = 1 block of size 16
    ///
    /// With the bug: add_request(30 tokens, 1 block) → seq_lens=30, blocks=1
    ///   → slot_mapping=-1 for tokens 17..30 → CUDA fault
    ///
    /// With the fix: add_request(16 tokens, 1 block) → seq_lens=16, blocks=1 ✓
    #[test]
    fn test_resumed_token_truncation_to_scheduled_chunk() {
        let block_size = 16;
        // Simulate a request with 30 tokens total (10 prompt + 20 outputs generated
        // before preemption).
        let full_token_buffer: Vec<u32> = (0u32..30).collect();
        let num_computed: u32 = 0; // all KV evicted
        let num_scheduled: usize = 16; // only first 16 tokens scheduled this step
        let new_block_ids = vec![0usize]; // 1 block covers 16 tokens

        // THE FIX: truncate to the scheduled chunk.
        let start = num_computed as usize;
        let end = (start + num_scheduled).min(full_token_buffer.len());
        let tokens_to_add = &full_token_buffer[start..end];

        assert_eq!(tokens_to_add.len(), 16);
        assert!(
            blocks_cover_tokens(tokens_to_add, &new_block_ids, block_size),
            "truncated tokens must be covered by allocated blocks"
        );

        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), tokens_to_add, new_block_ids, num_computed);
        assert_eq!(batch.tokens_in_pool_for("r1"), 0);
        assert_eq!(batch.block_table("r1"), Some(&[0usize][..]));

        let spec = HashMap::new();
        let prepared = batch.prepare_inputs(&spec);
        assert_eq!(prepared.req_inputs[0].token_count, 16);
        assert_eq!(prepared.attn_meta.seq_lens[0], 16); // tb(0) + tokens(16)
    }

    /// Without truncation (BUG scenario): add_request with more tokens than
    /// blocks can hold — the seq_lens would exceed block capacity.
    #[test]
    fn test_full_token_buffer_exceeds_block_coverage() {
        let block_size = 16;
        let full_tokens: Vec<u32> = (0u32..30).collect();
        let one_block = vec![0usize]; // covers only 16 tokens

        // This is what the BUGGY code would do:
        assert!(
            !blocks_cover_tokens(&full_tokens, &one_block, block_size),
            "BUG scenario: 30 tokens exceeds 1 block of 16 — would cause slot_mapping=-1"
        );
    }

    /// Chunked prefill resumption: first chunk is scheduled, second comes later.
    /// Each chunk must be independently covered by its block allocation.
    #[test]
    fn test_chunked_resumption_two_steps() {
        let block_size = 16;
        // Full sequence: 40 tokens.
        let full_buf: Vec<u32> = (0u32..40).collect();
        let num_computed = 0u32;

        // Step 1: schedule first 16 tokens, allocate 1 block.
        let num_sched_1 = 16usize;
        let blocks_1 = vec![0usize];
        let chunk1 = &full_buf[num_computed as usize..num_computed as usize + num_sched_1];
        assert!(blocks_cover_tokens(chunk1, &blocks_1, block_size));

        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), chunk1, blocks_1, num_computed);
        let spec = HashMap::new();
        let prepared = batch.prepare_inputs(&spec);
        assert_eq!(prepared.req_inputs[0].token_count, 16);
        batch.commit_step("r1", &[999], 16, false);
        // tokens_in_pool = 16 after first chunk.
        assert_eq!(batch.tokens_in_pool_for("r1"), 16);

        // Step 2: resume / continue with next 24 tokens (positions 16..40).
        // Scheduler re-adds with num_computed=16, new_block_ids for 24 more tokens (2 blocks).
        let num_computed_2 = 16u32;
        let num_sched_2 = 24usize;
        let blocks_2 = vec![0usize, 1, 2]; // full 3-block set covering 48 positions (16..48)

        // Truncate to the scheduled chunk.
        let start = num_computed_2 as usize;
        let end = (start + num_sched_2).min(full_buf.len());
        let chunk2 = &full_buf[start..end];
        assert_eq!(chunk2.len(), 24);
        assert!(blocks_cover_tokens(chunk2, &blocks_2, block_size));

        // Re-add as prefill for the second chunk.
        batch.remove_request("r1");
        batch.add_request("r1".into(), chunk2, blocks_2, num_computed_2);
        let prepared2 = batch.prepare_inputs(&spec);
        assert_eq!(prepared2.req_inputs[0].token_count, 24);
        assert_eq!(prepared2.attn_meta.seq_lens[0], 16 + 24); // tb + tokens
    }

    /// Preempted and resumed in the same batch step: the preemption removes
    /// the old slot and the resumption re-adds a fresh one; no phantom state.
    #[test]
    fn test_same_step_preemption_and_resumption() {
        let mut batch = InputBatch::new();
        // r1 has been running for a while.
        batch.add_request("r1".into(), &[10, 20, 30], vec![0, 1], 0);
        let spec = HashMap::new();
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[99], 3, false);
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[100], 1, false);
        assert_eq!(batch.tokens_in_pool_for("r1"), 4);

        // Same step: preempt r1 (remove stale slot), then immediately resume.
        batch.remove_request("r1"); // preempt
        assert!(!batch.contains("r1"));

        // Resume as prefill from scratch with fresh blocks.
        batch.add_request("r1".into(), &[10, 20], vec![5, 6], 0); // only 2-token chunk scheduled
        assert!(batch.contains("r1"));
        assert_eq!(
            batch.tokens_in_pool_for("r1"),
            0,
            "same-step resume must start with tokens_in_pool=0"
        );

        // prepare_inputs must treat r1 as prefill.
        let prepared = batch.prepare_inputs(&spec);
        let r1_idx = prepared
            .req_inputs
            .iter()
            .position(|r| r.req_id == "r1")
            .unwrap();
        assert!(prepared.attn_meta.is_prefill[r1_idx]);
        assert_eq!(prepared.req_inputs[r1_idx].token_count, 2);
    }

    /// Multiple preemptions: request preempted, resumed, preempted again,
    /// resumed again. Invariants must hold throughout.
    #[test]
    fn test_multiple_preemption_cycles() {
        let mut batch = InputBatch::new();
        let spec = HashMap::new();

        // First run.
        batch.add_request("r1".into(), &[1, 2, 3, 4], vec![0, 1], 0);
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[10], 4, false);
        assert_eq!(batch.tokens_in_pool_for("r1"), 4);

        // First preemption.
        batch.remove_request("r1");

        // First resumption: full re-prefill (only first 4 tokens scheduled).
        batch.add_request("r1".into(), &[1, 2, 3, 4], vec![2, 3], 0);
        assert_eq!(batch.tokens_in_pool_for("r1"), 0);
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[11], 4, false);
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[12], 1, false);
        assert_eq!(batch.tokens_in_pool_for("r1"), 5);

        // Second preemption.
        batch.remove_request("r1");
        assert!(!batch.contains("r1"));

        // Second resumption: partial prefix cache, num_computed=4.
        batch.add_request("r1".into(), &[1], vec![4, 5], 4); // 1 new token at pos 4
        assert_eq!(
            batch.tokens_in_pool_for("r1"),
            4,
            "partial cache: tokens_in_pool initialized to num_computed"
        );

        let prepared = batch.prepare_inputs(&spec);
        let r1_idx = prepared
            .req_inputs
            .iter()
            .position(|r| r.req_id == "r1")
            .unwrap();
        assert!(prepared.attn_meta.is_prefill[r1_idx]);
        // seq_lens = tb(4) + tokens(1) = 5
        assert_eq!(prepared.attn_meta.seq_lens[r1_idx], 5);
        // Position of the single token = 4 (prefix cache offset).
        assert_eq!(prepared.flat_positions, &[4]);
    }

    /// commit_step must be a no-op for a removed (preempted) request.
    /// This guards against stale slot reuse when another request gets the
    /// same slot index via swap-remove.
    #[test]
    fn test_commit_step_noop_for_removed_request() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30], vec![0], 0);
        batch.add_request("r2".into(), &[40, 50], vec![1], 0);
        let spec = HashMap::new();
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[99], 3, false);
        batch.commit_step("r2", &[100], 2, false);

        // Preempt r1.
        batch.remove_request("r1");

        // A stale commit_step for r1 should be a no-op (not corrupt r2).
        let tip_before = batch.tokens_in_pool_for("r2");
        batch.commit_step("r1", &[0], 99, false); // r1 no longer in batch
        let tip_after = batch.tokens_in_pool_for("r2");
        assert_eq!(
            tip_before, tip_after,
            "stale commit_step must not corrupt surviving requests"
        );
    }

    /// Verify that block_tables are correctly initialized for a resumed request,
    /// i.e., update_blocks followed by remove+add_request leaves the right table.
    #[test]
    fn test_resumed_block_table_overwrite() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20], vec![0, 1], 0);
        let spec = HashMap::new();
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[99], 2, false);

        // Simulate a block extension (normal decode operation).
        batch.update_blocks("r1", vec![0, 1, 2, 3]);
        assert_eq!(batch.block_table("r1"), Some(&[0, 1, 2, 3][..]));

        // Preempt.
        batch.remove_request("r1");

        // Resume with entirely new blocks.
        batch.add_request("r1".into(), &[10, 20], vec![7, 8], 0);
        assert_eq!(
            batch.block_table("r1"),
            Some(&[7, 8][..]),
            "resumed request must use the new block allocation, not the old one"
        );
    }

    /// Seq_lens computed during prepare_inputs equals tokens_in_pool + num_tokens.
    /// For prefill: seq_lens = num_computed + prompt_len.
    /// This must hold for both fresh prefill and resumed prefill.
    #[test]
    fn test_seq_lens_formula_for_prefill() {
        let mut batch = InputBatch::new();
        let spec = HashMap::new();

        // Fresh prefill: no prefix cache.
        batch.add_request("r1".into(), &[1, 2, 3, 4, 5], vec![0, 1], 0);
        let p = batch.prepare_inputs(&spec);
        assert_eq!(p.attn_meta.seq_lens[0], 5, "seq_lens = 0 + 5 = 5");

        // Resumed prefill: 2 tokens already cached.
        batch.add_request("r2".into(), &[10, 20, 30], vec![2, 3], 2);
        let p2 = batch.prepare_inputs(&spec);
        let r2_idx = p2.req_inputs.iter().position(|r| r.req_id == "r2").unwrap();
        assert_eq!(
            p2.attn_meta.seq_lens[r2_idx], 5,
            "seq_lens = 2 (cached) + 3 (tokens) = 5"
        );
    }

    /// After resumption with num_computed=0, commit_step correctly advances
    /// tokens_in_pool for subsequent decode steps.
    #[test]
    fn test_resumed_then_normal_decode_progression() {
        let mut batch = InputBatch::new();
        let spec = HashMap::new();

        // Resumed as full prefill (no prefix cache, 4-token chunk, 1 block of size 4).
        batch.add_request("r1".into(), &[10, 20, 30, 40], vec![0], 0);
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[99], 4, false);
        assert_eq!(batch.tokens_in_pool_for("r1"), 4);

        // Decode step 1.
        let p1 = batch.prepare_inputs(&spec);
        assert!(!p1.attn_meta.is_prefill[0]);
        assert_eq!(p1.flat_token_ids, &[99]);
        assert_eq!(p1.flat_positions, &[4]);
        batch.commit_step("r1", &[100], 1, false);
        assert_eq!(batch.tokens_in_pool_for("r1"), 5);

        // Decode step 2.
        let p2 = batch.prepare_inputs(&spec);
        assert_eq!(p2.flat_token_ids, &[100]);
        assert_eq!(p2.flat_positions, &[5]);
    }

    // -----------------------------------------------------------------------
    // Extended preemption/resumption coverage
    // Every nuance of the FerriteWorker preemption fix is covered below.
    // -----------------------------------------------------------------------

    /// Removing the LAST active request leaves an empty batch.
    #[test]
    fn test_preempt_last_request_leaves_empty_batch() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[1, 2], vec![0], 0);
        batch.remove_request("r1");
        assert_eq!(batch.num_active(), 0);
        assert!(!batch.contains("r1"));
    }

    /// Removing the FIRST of N requests triggers swap-remove; all surviving
    /// requests must remain slot-consistent and individually removable.
    #[test]
    fn test_preempt_first_of_many_slot_consistency() {
        let mut batch = InputBatch::new();
        for i in 0u32..5 {
            batch.add_request(format!("r{i}"), &[i], vec![i as usize], 0);
        }
        // Preempt r0 (slot 0); r4 should swap into slot 0.
        batch.remove_request("r0");
        assert_eq!(batch.num_active(), 4);
        // Every surviving request must still be findable and removable.
        for i in 1u32..5 {
            let id = format!("r{i}");
            assert!(batch.contains(&id));
            assert_eq!(batch.block_table(&id), Some(&[i as usize][..]));
        }
        // Remove them all one by one to verify no corruption.
        for i in 1u32..5 {
            batch.remove_request(&format!("r{i}"));
        }
        assert_eq!(batch.num_active(), 0);
    }

    /// Preempting a middle request: slot indices of all other requests must
    /// remain valid (no off-by-one errors from the swap-remove).
    #[test]
    fn test_preempt_middle_request_slot_consistency() {
        let mut batch = InputBatch::new();
        batch.add_request("a".into(), &[1], vec![10], 0);
        batch.add_request("b".into(), &[2], vec![20], 0);
        batch.add_request("c".into(), &[3], vec![30], 0);
        batch.add_request("d".into(), &[4], vec![40], 0);

        // Preempt "b" (slot 1). "d" (last slot = 3) should fill slot 1.
        batch.remove_request("b");
        assert_eq!(batch.num_active(), 3);
        assert!(!batch.contains("b"));
        assert_eq!(batch.block_table("a"), Some(&[10][..]));
        assert_eq!(batch.block_table("c"), Some(&[30][..]));
        assert_eq!(batch.block_table("d"), Some(&[40][..]));
    }

    /// Re-using the same req_id after preemption must not inherit any stale
    /// state from the previous slot (e.g., old block tables or tokens_in_pool).
    #[test]
    fn test_reused_req_id_no_stale_state() {
        let mut batch = InputBatch::new();
        let spec = HashMap::new();

        // First run: r1 runs for a while.
        batch.add_request("r1".into(), &[1, 2, 3], vec![0, 1], 0);
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[99], 3, false);
        for _ in 0..5 {
            let _ = batch.prepare_inputs(&spec);
            batch.commit_step("r1", &[0], 1, false);
        }
        assert_eq!(batch.tokens_in_pool_for("r1"), 8);
        assert_eq!(batch.block_table("r1").unwrap().len(), 2); // blocks 0,1

        // Preempt.
        batch.remove_request("r1");

        // Resume with completely different blocks and fewer tokens.
        batch.add_request("r1".into(), &[10, 20], vec![5], 0);
        // Must have fresh state, not the old 8 tokens in pool.
        assert_eq!(
            batch.tokens_in_pool_for("r1"),
            0,
            "reused req_id must have tokens_in_pool=0"
        );
        assert_eq!(
            batch.block_table("r1"),
            Some(&[5][..]),
            "reused req_id must use the new block table"
        );
    }

    /// Preemption of a request that was NEVER prefilled (just added then
    /// immediately preempted before the first prepare_inputs) must be safe.
    #[test]
    fn test_preempt_before_first_prefill() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[1, 2, 3], vec![0], 0);
        batch.add_request("r2".into(), &[4, 5], vec![1], 0);

        // Immediately preempt r1 without calling prepare_inputs.
        batch.remove_request("r1");
        assert_eq!(batch.num_active(), 1);
        assert!(batch.contains("r2"));
    }

    /// Block coverage: boundary at exactly one block full.
    #[test]
    fn test_block_coverage_exact_boundary() {
        let block_size = 16;
        let tokens: Vec<u32> = (0..16).collect();
        let blocks = vec![0usize];
        assert!(
            blocks_cover_tokens(&tokens, &blocks, block_size),
            "16 tokens, 1 block of 16: exactly covered"
        );
    }

    /// Block coverage: one token over the boundary requires an extra block.
    #[test]
    fn test_block_coverage_one_over_boundary() {
        let block_size = 16;
        let tokens: Vec<u32> = (0..17).collect();
        let one_block = vec![0usize];
        let two_blocks = vec![0usize, 1];

        assert!(
            !blocks_cover_tokens(&tokens, &one_block, block_size),
            "17 tokens with 1 block should NOT be covered"
        );
        assert!(
            blocks_cover_tokens(&tokens, &two_blocks, block_size),
            "17 tokens with 2 blocks should be covered"
        );
    }

    /// block_size=1 edge case: each token needs its own block.
    #[test]
    fn test_block_coverage_block_size_one() {
        let block_size = 1;
        let tokens: Vec<u32> = (0..5).collect();
        let five_blocks: Vec<usize> = (0..5).collect();
        let four_blocks: Vec<usize> = (0..4).collect();

        assert!(blocks_cover_tokens(&tokens, &five_blocks, block_size));
        assert!(!blocks_cover_tokens(&tokens, &four_blocks, block_size));
    }

    /// Empty token slice: always covered regardless of block count.
    #[test]
    fn test_block_coverage_empty_tokens() {
        let block_size = 16;
        let empty: Vec<u32> = vec![];
        let no_blocks: Vec<usize> = vec![];
        assert!(
            blocks_cover_tokens(&empty, &no_blocks, block_size),
            "empty token slice with no blocks should be covered"
        );
    }

    /// The resumed chunk's slice must correctly start at num_computed.
    /// If prefix cache provides 8 tokens, the chunk starts at index 8.
    #[test]
    fn test_chunk_slice_starts_at_num_computed() {
        let full_buf: Vec<u32> = (100..120).collect(); // 20 tokens
        let num_computed = 8u32;
        let num_scheduled = 6usize;

        let start = num_computed as usize;
        let end = (start + num_scheduled).min(full_buf.len());
        let chunk = &full_buf[start..end];

        assert_eq!(chunk.len(), 6);
        // Tokens 108..114 (0-indexed into 100..120 → values 108..114).
        assert_eq!(chunk[0], 108);
        assert_eq!(chunk[5], 113);
    }

    /// When num_computed + num_scheduled > len(full_buf), the chunk must be
    /// clamped to the end of the buffer (no out-of-bounds panic).
    #[test]
    fn test_chunk_clamp_at_buffer_end() {
        let full_buf: Vec<u32> = (0..10).collect();
        let num_computed = 7u32;
        let num_scheduled = 10usize; // would go to index 17 without clamping

        let start = num_computed as usize;
        let end = (start + num_scheduled).min(full_buf.len());
        let chunk = &full_buf[start..end];

        assert_eq!(chunk.len(), 3); // only 3 tokens remain (indices 7, 8, 9)
        assert_eq!(chunk, &[7u32, 8, 9]);
    }

    /// Resumption chunk size must exactly match block capacity.
    /// This is the key invariant that prevents slot_mapping=-1.
    #[test]
    fn test_resumed_chunk_matches_block_capacity() {
        // Simulate a long request preempted after 1024 tokens.
        // Scheduler re-schedules with 128-token chunk, allocates 8 blocks of 16.
        let block_size = 16usize;
        let full_buf: Vec<u32> = (0..1024).collect();
        let num_computed = 0u32;
        let num_scheduled = 128usize;
        let new_block_ids: Vec<usize> = (0..8).collect(); // 8 blocks × 16 = 128 capacity

        let start = num_computed as usize;
        let end = (start + num_scheduled).min(full_buf.len());
        let chunk = &full_buf[start..end];

        assert_eq!(chunk.len(), 128);
        assert_eq!(new_block_ids.len() * block_size, 128);
        assert!(blocks_cover_tokens(chunk, &new_block_ids, block_size));
    }

    /// Simulate a realistic 3-request mixed batch:
    ///   r1 = normal decode (never preempted)
    ///   r2 = preempted last step, resumed this step as prefill
    ///   r3 = brand new request
    ///
    /// All three must coexist correctly in prepare_inputs.
    #[test]
    fn test_mixed_batch_with_preempted_and_new() {
        let mut batch = InputBatch::new();
        let spec = HashMap::new();

        // r1: running decode.
        batch.add_request("r1".into(), &[10, 20], vec![0], 0);
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[99], 2, false);
        // r1 is now decode.

        // r2: was running, preempted, re-admitted this step.
        batch.add_request("r2".into(), &[30, 40, 50], vec![1, 2], 0);
        let _ = batch.prepare_inputs(&spec); // r2 is prefill
        batch.commit_step("r2", &[100], 3, false);
        // r2 is now decode.
        batch.remove_request("r2"); // preempted

        // r3: brand new.
        batch.add_request("r3".into(), &[60, 70], vec![3], 0);

        // r2 resumed as prefill (only 2 tokens scheduled, 1 block).
        batch.add_request("r2".into(), &[30, 40], vec![4], 0);

        assert_eq!(batch.num_active(), 3); // r1 (decode), r2 (prefill), r3 (prefill)

        let prepared = batch.prepare_inputs(&spec);
        assert_eq!(prepared.attn_meta.num_reqs, 3);

        let r1_idx = prepared
            .req_inputs
            .iter()
            .position(|r| r.req_id == "r1")
            .unwrap();
        let r2_idx = prepared
            .req_inputs
            .iter()
            .position(|r| r.req_id == "r2")
            .unwrap();
        let r3_idx = prepared
            .req_inputs
            .iter()
            .position(|r| r.req_id == "r3")
            .unwrap();

        assert!(!prepared.attn_meta.is_prefill[r1_idx], "r1 must be decode");
        assert!(prepared.attn_meta.is_prefill[r2_idx], "r2 must be prefill");
        assert!(prepared.attn_meta.is_prefill[r3_idx], "r3 must be prefill");

        assert_eq!(prepared.req_inputs[r1_idx].token_count, 1);
        assert_eq!(prepared.req_inputs[r2_idx].token_count, 2);
        assert_eq!(prepared.req_inputs[r3_idx].token_count, 2);

        // r1 decode: seq_lens = tokens_in_pool(2) + 1 = 3.
        assert_eq!(prepared.attn_meta.seq_lens[r1_idx], 3);
        // r2 prefill (fresh): seq_lens = 0 + 2 = 2.
        assert_eq!(prepared.attn_meta.seq_lens[r2_idx], 2);
        // r3 prefill (fresh): seq_lens = 0 + 2 = 2.
        assert_eq!(prepared.attn_meta.seq_lens[r3_idx], 2);
    }

    /// update_blocks for a resumed request must happen BEFORE remove+add_request,
    /// and the final block table must come from add_request, not update_blocks.
    ///
    /// This tests that the FerriteWorker pattern:
    ///   1. update_blocks(req_id, new_block_ids)  ← from cached-reqs loop
    ///   2. remove_request(req_id)               ← preemption fixup
    ///   3. add_request(req_id, tokens, new_block_ids, num_computed) ← resumption
    ///
    /// leaves the correct final state.
    #[test]
    fn test_update_blocks_then_readd_final_state() {
        let mut batch = InputBatch::new();
        let spec = HashMap::new();

        batch.add_request("r1".into(), &[1, 2, 3], vec![0, 1], 0);
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[99], 3, false);
        // r1 is decode, block_table = [0, 1].

        // Step 1: update_blocks with new allocation (as if scheduler gave new blocks).
        batch.update_blocks("r1", vec![10, 11, 12]);
        assert_eq!(batch.block_table("r1"), Some(&[10, 11, 12][..]));

        // Step 2: preempt.
        batch.remove_request("r1");

        // Step 3: re-add with final resumption blocks.
        batch.add_request("r1".into(), &[1, 2], vec![20, 21], 0);

        // Final state: add_request blocks win.
        assert_eq!(
            batch.block_table("r1"),
            Some(&[20, 21][..]),
            "final block table must be from add_request, not the intermediate update_blocks"
        );
        assert_eq!(
            batch.tokens_in_pool_for("r1"),
            0,
            "re-added request must have tokens_in_pool=0"
        );
    }

    /// commit_step called for a resumed (prefill) slot must update tokens_in_pool
    /// to exactly num_computed + input_token_count (no residual from previous decode).
    #[test]
    fn test_commit_after_resumption_correct_tokens_in_pool() {
        let mut batch = InputBatch::new();
        let spec = HashMap::new();

        // Long initial run.
        batch.add_request("r1".into(), &[1, 2, 3, 4, 5, 6, 7, 8], vec![0, 1, 2, 3], 0);
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[99], 8, false);
        for _ in 0..10 {
            let _ = batch.prepare_inputs(&spec);
            batch.commit_step("r1", &[0], 1, false);
        }
        assert_eq!(batch.tokens_in_pool_for("r1"), 18); // 8 + 10

        // Preempt, then resume with 4-token chunk.
        batch.remove_request("r1");
        batch.add_request("r1".into(), &[1, 2, 3, 4], vec![0, 1], 0);
        assert_eq!(batch.tokens_in_pool_for("r1"), 0);

        // Commit the prefill step.
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[88], 4, false);
        assert_eq!(
            batch.tokens_in_pool_for("r1"),
            4,
            "after prefill commit: tokens_in_pool = 0 + 4 = 4 (not 18+4)"
        );
    }

    /// Verify positions emitted during resumed prefill.
    /// With num_computed=0: positions are 0,1,2,...,N-1.
    /// With num_computed=K: positions are K, K+1, ..., K+N-1.
    #[test]
    fn test_resumed_prefill_positions() {
        let mut batch = InputBatch::new();
        let spec = HashMap::new();

        // Fresh resumption (no prefix cache).
        batch.add_request("r1".into(), &[10, 20, 30], vec![0, 1], 0);
        let p = batch.prepare_inputs(&spec);
        assert_eq!(p.flat_positions, &[0, 1, 2]);
        batch.reclaim_buffers(p);

        // Resumption with prefix cache (4 tokens pre-computed).
        batch.remove_request("r1");
        batch.add_request("r1".into(), &[40, 50, 60], vec![0, 1], 4);
        let p2 = batch.prepare_inputs(&spec);
        assert_eq!(
            p2.flat_positions,
            &[4, 5, 6],
            "resumed with num_computed=4: positions must be 4,5,6"
        );
    }

    /// query_start_loc invariants must hold in a batch containing a resumed
    /// (multi-token prefill) request alongside a normal decode request.
    #[test]
    fn test_query_start_loc_with_resumed_prefill() {
        let mut batch = InputBatch::new();
        let spec = HashMap::new();

        // r1 in decode.
        batch.add_request("r1".into(), &[1, 2], vec![0], 0);
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[99], 2, false);

        // r2 resumed as 5-token prefill.
        batch.add_request("r2".into(), &[10, 20, 30, 40, 50], vec![1, 2, 3], 0);

        let prepared = batch.prepare_inputs(&spec);
        let meta = &prepared.attn_meta;

        // query_start_loc must have num_reqs + 1 entries.
        assert_eq!(meta.query_start_loc.len(), meta.num_reqs + 1);
        // Last entry = total_tokens.
        assert_eq!(*meta.query_start_loc.last().unwrap(), meta.total_tokens);
        // Each span matches q_lens.
        for i in 0..meta.num_reqs {
            assert_eq!(
                meta.query_start_loc[i + 1] - meta.query_start_loc[i],
                meta.q_lens[i]
            );
        }
        // Total = 1 (decode r1) + 5 (prefill r2) = 6.
        assert_eq!(meta.total_tokens, 6);
    }

    /// Preempting ALL requests simultaneously leaves an empty, consistent batch.
    #[test]
    fn test_preempt_all_requests_simultaneously() {
        let mut batch = InputBatch::new();
        let spec = HashMap::new();

        for i in 0u32..6 {
            batch.add_request(format!("r{i}"), &[i, i + 1], vec![i as usize], 0);
        }
        let _ = batch.prepare_inputs(&spec);
        for i in 0u32..6 {
            batch.commit_step(&format!("r{i}"), &[100 + i], 2, false);
        }

        // Preempt all.
        for i in 0u32..6 {
            batch.remove_request(&format!("r{i}"));
        }
        assert_eq!(batch.num_active(), 0);

        // Re-admit all as fresh prefills.
        for i in 0u32..6 {
            batch.add_request(
                format!("r{i}"),
                &[i, i + 1],
                vec![10 + i as usize, 20 + i as usize],
                0,
            );
        }
        assert_eq!(batch.num_active(), 6);

        let prepared = batch.prepare_inputs(&spec);
        // All 6 requests must be prefill.
        assert!(
            prepared.attn_meta.is_prefill.iter().all(|&p| p),
            "all re-added requests must be prefill"
        );
    }

    /// Stress: 10 preemption+resumption cycles on the same request. After each
    /// cycle, tokens_in_pool and block_table must reflect the fresh state only.
    #[test]
    fn test_many_preemption_cycles_stress() {
        let mut batch = InputBatch::new();
        let spec = HashMap::new();
        let prompt: Vec<u32> = (1..=8).collect();

        for cycle in 0u32..10 {
            let blocks: Vec<usize> = vec![cycle as usize * 2, cycle as usize * 2 + 1];
            batch.add_request("r1".into(), &prompt, blocks.clone(), 0);
            assert_eq!(
                batch.tokens_in_pool_for("r1"),
                0,
                "cycle {cycle}: tokens_in_pool must be 0 on fresh add"
            );
            assert_eq!(
                batch.block_table("r1").unwrap(),
                blocks.as_slice(),
                "cycle {cycle}: block table must match fresh allocation"
            );

            // Prefill + decode.
            let _ = batch.prepare_inputs(&spec);
            batch.commit_step("r1", &[99 + cycle], 8, false);
            let _ = batch.prepare_inputs(&spec);
            batch.commit_step("r1", &[100 + cycle], 1, false);
            assert_eq!(batch.tokens_in_pool_for("r1"), 9);

            // Preempt.
            batch.remove_request("r1");
        }
        assert_eq!(batch.num_active(), 0);
    }

    /// The resumption chunk for a request that was mid-second-prefill chunk
    /// (num_computed > 0) must use the right slice: buf[num_computed..num_computed+num_scheduled].
    #[test]
    fn test_chunked_prefill_second_chunk_slice() {
        // Full buffer: prompt tokens [0..64].
        let full_buf: Vec<u32> = (0..64).collect();

        // First chunk: tokens 0..32, num_computed=0, num_scheduled=32.
        let chunk1 = &full_buf[0..32];
        assert_eq!(chunk1.len(), 32);

        // Second chunk: tokens 32..64, num_computed=32, num_scheduled=32.
        let num_computed_2 = 32u32;
        let num_scheduled_2 = 32usize;
        let start = num_computed_2 as usize;
        let end = (start + num_scheduled_2).min(full_buf.len());
        let chunk2 = &full_buf[start..end];

        assert_eq!(chunk2.len(), 32);
        assert_eq!(chunk2[0], 32, "second chunk must start at token index 32");
        assert_eq!(chunk2[31], 63, "second chunk must end at token index 63");

        // Verify block coverage for the second chunk (2 blocks of size 16 = 32).
        let block_size = 16;
        let blocks: Vec<usize> = (2..4).collect(); // blocks 2 and 3 for second chunk
        assert!(blocks_cover_tokens(chunk2, &blocks, block_size));
    }

    /// `tokens_before` in attn_meta must equal `tokens_in_pool` for all slots.
    /// For a resumed prefill with num_computed=0, tokens_before must be 0.
    /// For a resumed prefill with num_computed=K, tokens_before must be K.
    #[test]
    fn test_tokens_before_for_resumed_prefill() {
        let mut batch = InputBatch::new();
        let spec = HashMap::new();

        // r1: resumed from scratch.
        batch.add_request("r1".into(), &[1, 2, 3], vec![0, 1], 0);
        // r2: resumed with partial prefix cache (6 tokens precomputed).
        batch.add_request("r2".into(), &[4, 5, 6, 7], vec![2, 3, 4], 6);

        let prepared = batch.prepare_inputs(&spec);
        let r1_idx = prepared
            .req_inputs
            .iter()
            .position(|r| r.req_id == "r1")
            .unwrap();
        let r2_idx = prepared
            .req_inputs
            .iter()
            .position(|r| r.req_id == "r2")
            .unwrap();

        assert_eq!(
            prepared.attn_meta.tokens_before[r1_idx], 0,
            "r1 resumed from scratch: tokens_before must be 0"
        );
        assert_eq!(
            prepared.attn_meta.tokens_before[r2_idx], 6,
            "r2 with prefix cache: tokens_before must be 6"
        );
    }

    /// `seq_lens[i]` = `tokens_before[i]` + `q_lens[i]` for all requests.
    /// This must hold for mixed batches with preempted/resumed/new/decode requests.
    #[test]
    fn test_seq_lens_equals_tokens_before_plus_q_lens() {
        let mut batch = InputBatch::new();
        let spec = HashMap::new();

        // r1: decode (2 tokens in pool).
        batch.add_request("r1".into(), &[1, 2], vec![0], 0);
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[99], 2, false);

        // r2: resumed prefill, no prefix cache.
        batch.add_request("r2".into(), &[10, 20, 30], vec![1, 2], 0);

        // r3: resumed prefill, 4 tokens cached.
        batch.add_request("r3".into(), &[40, 50], vec![3, 4], 4);

        let prepared = batch.prepare_inputs(&spec);
        let meta = &prepared.attn_meta;
        for i in 0..meta.num_reqs {
            assert_eq!(
                meta.seq_lens[i],
                meta.tokens_before[i] + meta.q_lens[i],
                "req {i}: seq_lens must equal tokens_before + q_lens"
            );
        }
    }

    /// After preempting a request that had a PENDING spec-decode commit
    /// (was_spec_decode=true), removing it must not affect other requests.
    #[test]
    fn test_preempt_request_with_spec_decode_history() {
        let mut batch = InputBatch::new();
        let spec_map = HashMap::new();

        // r1 and r2 both run.
        batch.add_request("r1".into(), &[1, 2], vec![0], 0);
        batch.add_request("r2".into(), &[3, 4], vec![1], 0);
        let _ = batch.prepare_inputs(&spec_map);
        batch.commit_step("r1", &[10], 2, false);
        batch.commit_step("r2", &[20], 2, false);

        // r1 runs speculative decode: 2 drafts accepted + 1 bonus = 3 tokens.
        let _ = batch.prepare_inputs(&spec_map);
        batch.commit_step("r1", &[11, 12, 13], 3, true);
        assert_eq!(batch.tokens_in_pool_for("r1"), 5); // 2 + 3

        // r2 normal decode.
        let _ = batch.prepare_inputs(&spec_map);
        batch.commit_step("r2", &[21], 1, false);
        assert_eq!(batch.tokens_in_pool_for("r2"), 3);

        // Preempt r1 (the one with spec-decode history).
        batch.remove_request("r1");
        assert!(!batch.contains("r1"));
        assert_eq!(batch.num_active(), 1);

        // r2 must be unaffected.
        assert_eq!(batch.tokens_in_pool_for("r2"), 3);
        let prepared = batch.prepare_inputs(&spec_map);
        assert_eq!(prepared.req_inputs[0].req_id, "r2");
        assert!(!prepared.attn_meta.is_prefill[0]);
    }

    /// Verify that after N decode steps, preemption + resumption with a
    /// single-token chunk correctly computes seq_lens as 0 + 1 = 1.
    #[test]
    fn test_resumed_single_token_chunk() {
        let mut batch = InputBatch::new();
        let spec = HashMap::new();

        // r1: long sequence, 100 decode steps.
        batch.add_request("r1".into(), &[1, 2, 3, 4, 5], vec![0, 1, 2], 0);
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[99], 5, false);
        for i in 0..100 {
            let _ = batch.prepare_inputs(&spec);
            batch.commit_step("r1", &[100 + i], 1, false);
        }
        assert_eq!(batch.tokens_in_pool_for("r1"), 105);

        // Preempt, resume with single-token chunk (extreme chunked prefill).
        batch.remove_request("r1");
        batch.add_request("r1".into(), &[1], vec![0], 0); // 1 token, 1 block
        assert_eq!(batch.tokens_in_pool_for("r1"), 0);

        let prepared = batch.prepare_inputs(&spec);
        assert_eq!(
            prepared.attn_meta.seq_lens[0], 1,
            "single-token resumed prefill: seq_lens must be 1"
        );
        assert!(prepared.attn_meta.is_prefill[0]);
    }

    // -----------------------------------------------------------------------
    // set_prefill_continuation — running multi-chunk prefill
    // -----------------------------------------------------------------------

    /// Verify that set_prefill_continuation re-arms a decode slot for prefill.
    /// Simulates a running request whose prompt spans two scheduler chunks.
    #[test]
    fn test_set_prefill_continuation_basic() {
        let mut batch = InputBatch::new();
        let spec = HashMap::new();

        // Chunk 1: tokens 0..5 of a 10-token prompt.
        batch.add_request("r1".into(), &[0, 1, 2, 3, 4], vec![0], 0);
        let p1 = batch.prepare_inputs(&spec);
        assert_eq!(p1.req_inputs[0].token_count, 5);
        assert!(p1.attn_meta.is_prefill[0]);
        batch.commit_step("r1", &[99], 5, false);
        assert_eq!(batch.tokens_in_pool_for("r1"), 5);

        // After commit, slot is in decode mode.
        let p_dec = batch.prepare_inputs(&spec);
        assert!(!p_dec.attn_meta.is_prefill[0]);
        assert_eq!(p_dec.attn_meta.q_lens[0], 1); // decode: 1 token

        // Scheduler sends chunk 2: tokens 5..10.
        batch.set_prefill_continuation("r1", vec![5, 6, 7, 8, 9], 5);

        // Now prepare_inputs must emit the full 5-token chunk as prefill.
        let p2 = batch.prepare_inputs(&spec);
        assert!(
            p2.attn_meta.is_prefill[0],
            "must be prefill after continuation"
        );
        assert_eq!(p2.req_inputs[0].token_count, 5);
        assert_eq!(p2.attn_meta.seq_lens[0], 5 + 5); // tokens_in_pool + chunk
        // Positions: 5, 6, 7, 8, 9.
        assert_eq!(p2.attn_meta.query_start_loc[0], 0);
    }

    /// Verify tokens_in_pool, positions, and seq_lens after two-chunk prefill
    /// followed by a decode step.
    #[test]
    fn test_set_prefill_continuation_then_decode() {
        let mut batch = InputBatch::new();
        let spec = HashMap::new();

        // Chunk 1: tokens 0..4.
        batch.add_request("r1".into(), &[10, 20, 30, 40], vec![0], 0);
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[50], 4, false); // tokens_in_pool = 4

        // Chunk 2: tokens 4..8.
        batch.set_prefill_continuation("r1", vec![50, 60, 70, 80], 4);
        let p2 = batch.prepare_inputs(&spec);
        assert_eq!(p2.req_inputs[0].token_count, 4);
        assert_eq!(p2.attn_meta.seq_lens[0], 8);
        batch.commit_step("r1", &[90], 4, false); // tokens_in_pool = 8

        // Now in decode mode — emit one token at position 8.
        let pd = batch.prepare_inputs(&spec);
        assert!(!pd.attn_meta.is_prefill[0]);
        assert_eq!(pd.attn_meta.q_lens[0], 1);
        assert_eq!(pd.attn_meta.seq_lens[0], 9); // tokens_in_pool + 1
    }

    /// Verify slot_mapping validity invariant: seq_lens ≤ blocks * block_size.
    #[test]
    fn test_set_prefill_continuation_slot_mapping_invariant() {
        let block_size = 4usize;
        let mut batch = InputBatch::new();
        let spec = HashMap::new();

        // 2 blocks, 8 token capacity.  Chunk 1: tokens 0..4 (1 block).
        batch.add_request("r1".into(), &[0, 1, 2, 3], vec![0], 0);
        let _ = batch.prepare_inputs(&spec);
        batch.commit_step("r1", &[99], 4, false);

        // Chunk 2: tokens 4..8 (block 1 appended).
        batch.update_blocks("r1", vec![0, 1]);
        batch.set_prefill_continuation("r1", vec![4, 5, 6, 7], 4);
        let p2 = batch.prepare_inputs(&spec);
        // seq_lens = 4 + 4 = 8.  Blocks cover 8 tokens.  No -1 slot.
        assert_eq!(p2.attn_meta.seq_lens[0], 8);
        assert!(
            p2.attn_meta.seq_lens[0] <= p2.attn_meta.block_ids[0].len() * block_size,
            "seq_lens must not exceed block capacity"
        );
    }
}
