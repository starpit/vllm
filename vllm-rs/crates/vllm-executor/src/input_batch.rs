// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Persistent `InputBatch` that maintains pre-allocated buffers across engine
//! steps, eliminating redundant allocations on the decode hot path.
//!
//! Port of the Python V1 `InputBatch` concept: delta-update a dense array of
//! per-request slots instead of rebuilding all model inputs from scratch every
//! step.

use std::collections::HashMap;

use vllm_models::AttentionMetadata;

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
    /// Simulates the deferred commit pattern from CudaWorker's super-fast
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

        let batch = vec![
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
}
