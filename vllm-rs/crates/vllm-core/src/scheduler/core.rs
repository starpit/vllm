// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Main scheduler implementation, ported from `vllm/v1/core/sched/scheduler.py`.
//!
//! The scheduler is responsible for deciding which requests to process at each
//! scheduling step and how many tokens to allocate to each request. It manages
//! the request lifecycle through waiting, running, and finished states.
//!
//! This initial Rust port simplifies the KV cache interaction by tracking a
//! simple block counter rather than full block allocation. A trait
//! [`KVCacheManagerOps`] is defined so that the real KV cache manager (built
//! by another agent) can be plugged in later.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

use tracing::warn;
use vllm_common::{Request, RequestStatus};
use vllm_config::{SchedulerConfig, SchedulerPolicy};

use super::interface::{PauseState, SchedulerInterface};
use super::output::{CachedRequestData, NewRequestData, SchedulerOutput};
use super::request_queue::{RequestQueue, SchedulingPolicy, create_request_queue};

// ---------------------------------------------------------------------------
// KVCacheManagerOps -- trait for KV cache interaction
// ---------------------------------------------------------------------------

/// Trait that abstracts the KV cache manager operations needed by the
/// scheduler.
///
/// The real `KVCacheManager` (in the `kv_cache_manager` module) will
/// implement this trait. For testing and initial bring-up, a simple
/// block-counting implementation is provided.
pub trait KVCacheManagerOps: Send {
    /// Try to allocate blocks for a request.
    ///
    /// Returns the block IDs (one `Vec<usize>` per KV cache group) if
    /// allocation succeeded, or `None` if there are insufficient free
    /// blocks.
    fn allocate_slots(
        &mut self,
        request: &Request,
        num_new_tokens: usize,
        num_lookahead_tokens: usize,
    ) -> Option<Vec<Vec<usize>>>;

    /// Free all blocks held by a request.
    fn free(&mut self, request_id: &str);

    /// Get the block IDs currently assigned to a request.
    fn get_blocks(&self, request_id: &str) -> Vec<Vec<usize>>;

    /// Get the number of computed tokens from prefix cache for a new
    /// request.
    ///
    /// Returns `(num_computed_tokens, block_ids)` where `block_ids` are
    /// the cached blocks.
    fn get_computed_blocks(&self, request: &Request) -> (u32, Vec<Vec<usize>>);

    /// Notify the KV cache manager that a new scheduling step is starting.
    fn new_step_starts(&mut self);

    /// Reset the prefix cache. Returns `true` if successful.
    fn reset_prefix_cache(&mut self) -> bool;

    /// Number of free blocks available.
    fn num_free_blocks(&self) -> usize;

    /// Total number of blocks.
    fn num_total_blocks(&self) -> usize;

    /// Block size (tokens per block).
    fn block_size(&self) -> usize;

    /// KV cache usage as a fraction in `[0.0, 1.0]`.
    fn usage(&self) -> f64;

    /// Number of blocks retained in the prefix cache.
    fn num_cached_blocks(&self) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// SimpleBlockTracker -- a minimal KV cache manager for initial bring-up
// ---------------------------------------------------------------------------

/// A block tracker that satisfies [`KVCacheManagerOps`].
///
/// Full Python parity with `vllm/v1/core/block_pool.py`:
///
/// - **Reference counting**: Each block has a `ref_cnt`. Multiple requests
///   sharing a cached prefix share the same block (ref_cnt > 1).
/// - **Free queue**: Doubly-linked list emulated via `VecDeque`. Blocks enter
///   the free queue only when ref_cnt drops to 0. Stale entries (blocks
///   reclaimed via `touch()`) are skipped lazily on pop.
/// - **Lazy eviction**: Hash→block mappings persist after `free()` and are
///   only removed when the block is popped from the free queue for a new
///   allocation (`_maybe_evict_cached_block`).
/// - **`touch()`**: Increments ref_cnt. If ref_cnt was 0 (block in free queue),
///   the block is logically removed from the free queue (stale entry skipped
///   lazily).
/// - **`free_blocks()`**: Decrements ref_cnt. Blocks with ref_cnt == 0 are
///   appended to the free queue.
/// - Blocks are freed in reverse order so tail (decode) blocks are evicted
///   first and prefix blocks survive longest (matching Python's
///   `reversed(req_blocks)` in `SingleTypeKVCacheManager.free()`).
pub struct SimpleBlockTracker {
    total_blocks: usize,
    block_size: usize,

    /// Per-block reference count. ref_cnt > 0 means allocated (possibly shared).
    /// ref_cnt == 0 means in free queue (eviction candidate).
    /// Matches Python's `KVCacheBlock.ref_cnt`.
    ref_cnt: Vec<usize>,

    /// Number of blocks with ref_cnt == 0. Maintained incrementally to avoid
    /// O(n) scans.
    num_free_blocks: usize,

    /// Eviction queue: blocks with ref_cnt == 0. Pop from front (LRU).
    /// May contain stale entries (blocks whose ref_cnt > 0 due to `touch()`);
    /// these are skipped during `allocate_fresh_blocks`.
    free_queue: VecDeque<usize>,

    /// request_id -> (block_ids, num_blocks_held)
    allocations: HashMap<String, (Vec<Vec<usize>>, usize)>,

    // --- Prefix caching fields ---
    /// Whether prefix caching is enabled.
    enable_caching: bool,

    /// Hash of a full block's token content → block ID.
    /// Persists after free(). Only removed when the block is popped
    /// from the free queue and given to a new allocation
    /// (lazy eviction, matching Python's `_maybe_evict_cached_block`).
    block_hash_to_id: HashMap<u64, usize>,

    /// Reverse mapping: block ID → hash. For O(1) hash cleanup when
    /// a block is evicted during allocation.
    block_id_to_hash: HashMap<usize, u64>,

    /// Request ID → ordered list of block hashes.
    req_to_hashes: HashMap<String, Vec<u64>>,
}

impl SimpleBlockTracker {
    /// Create a new block tracker with the given number of GPU blocks and
    /// block size.
    pub fn new(num_gpu_blocks: usize, block_size: usize) -> Self {
        Self {
            total_blocks: num_gpu_blocks,
            block_size,
            ref_cnt: vec![0; num_gpu_blocks],
            num_free_blocks: num_gpu_blocks,
            free_queue: (0..num_gpu_blocks).collect(),
            allocations: HashMap::new(),
            enable_caching: false,
            block_hash_to_id: HashMap::new(),
            block_id_to_hash: HashMap::new(),
            req_to_hashes: HashMap::new(),
        }
    }

    /// Create a new block tracker with prefix caching enabled.
    pub fn with_caching(num_gpu_blocks: usize, block_size: usize) -> Self {
        let mut tracker = Self::new(num_gpu_blocks, block_size);
        tracker.enable_caching = true;
        tracker
    }

    /// Compute how many blocks a request needs for the given number of tokens.
    fn blocks_needed(num_tokens: usize, block_size: usize) -> usize {
        if num_tokens == 0 {
            return 0;
        }
        num_tokens.div_ceil(block_size)
    }

    /// Hash a block-sized chunk of token IDs.
    fn hash_block(tokens: &[u32]) -> u64 {
        let mut hasher = std::hash::DefaultHasher::new();
        tokens.hash(&mut hasher);
        hasher.finish()
    }

    /// Try to allocate `count` fresh block IDs from the free queue.
    ///
    /// Pops blocks from the front of `free_queue`, skipping stale entries
    /// (blocks with ref_cnt > 0, already reclaimed via `touch()`). When a
    /// popped block has a hash mapping, the mapping is removed (lazy eviction
    /// matching Python's `_maybe_evict_cached_block`).
    fn allocate_fresh_blocks(&mut self, count: usize) -> Option<Vec<usize>> {
        if self.num_free_blocks < count {
            return None;
        }

        let mut ids = Vec::with_capacity(count);
        while ids.len() < count {
            let block_id = self.free_queue.pop_front()?;

            // Skip stale entries: blocks reclaimed via touch() have ref_cnt > 0.
            if self.ref_cnt[block_id] > 0 {
                continue;
            }

            // Set ref_cnt to 1 (allocated).
            self.ref_cnt[block_id] = 1;
            self.num_free_blocks -= 1;

            // Lazy eviction: remove hash mapping if this block was cached.
            // Matches Python's `_maybe_evict_cached_block`.
            if let Some(hash) = self.block_id_to_hash.remove(&block_id) {
                self.block_hash_to_id.remove(&hash);
            }

            ids.push(block_id);
        }
        Some(ids)
    }

    /// Touch a block: increment ref_cnt. If ref_cnt was 0 (block in free
    /// queue), the block is logically removed — the stale entry in
    /// `free_queue` is skipped lazily on the next `allocate_fresh_blocks`.
    ///
    /// Matches Python's `BlockPool.touch()`.
    fn touch_block(&mut self, block_id: usize) {
        if self.ref_cnt[block_id] == 0 {
            // Block was free, now allocated — decrement free count.
            // The stale free_queue entry will be skipped lazily.
            self.num_free_blocks -= 1;
        }
        self.ref_cnt[block_id] += 1;
    }

    /// Free blocks: decrement ref_cnt for each block. Blocks whose ref_cnt
    /// drops to 0 are appended to the free queue.
    ///
    /// Matches Python's `BlockPool.free_blocks()`.
    fn free_blocks(&mut self, block_ids: &[usize]) {
        for &bid in block_ids {
            debug_assert!(self.ref_cnt[bid] > 0, "double-free of block {bid}");
            self.ref_cnt[bid] -= 1;
            if self.ref_cnt[bid] == 0 {
                self.num_free_blocks += 1;
                self.free_queue.push_back(bid);
            }
        }
    }
}

impl KVCacheManagerOps for SimpleBlockTracker {
    fn allocate_slots(
        &mut self,
        request: &Request,
        num_new_tokens: usize,
        num_lookahead_tokens: usize,
    ) -> Option<Vec<Vec<usize>>> {
        let total_tokens =
            request.num_computed_tokens as usize + num_new_tokens + num_lookahead_tokens;
        let needed = Self::blocks_needed(total_tokens, self.block_size);

        // How many blocks does this request already hold?
        let currently_held = self
            .allocations
            .get(&request.request_id)
            .map(|(_, n)| *n)
            .unwrap_or(0);

        // If this is a new/re-admitted request with cached prefix, seed the
        // allocation with the cached block IDs (they still hold KV data).
        // Touch each cached block to increment ref_cnt (block sharing).
        if currently_held == 0 && self.enable_caching && request.num_computed_tokens > 0 {
            let num_cached_blocks = request.num_computed_tokens as usize / self.block_size;
            if num_cached_blocks > 0 {
                let all_tokens = &request.all_token_ids;
                let mut cached_ids = Vec::new();
                let mut hashes = Vec::new();
                for i in 0..num_cached_blocks {
                    let start = i * self.block_size;
                    let end = start + self.block_size;
                    if end <= all_tokens.len() {
                        let hash = Self::hash_block(&all_tokens[start..end]);
                        if let Some(&bid) = self.block_hash_to_id.get(&hash) {
                            cached_ids.push(bid);
                            hashes.push(hash);
                        } else {
                            break;
                        }
                    }
                }

                if !cached_ids.is_empty() {
                    let cached_count = cached_ids.len();
                    let additional = needed.saturating_sub(cached_count);

                    // Python checks free capacity BEFORE touching cached
                    // blocks. Touching moves ref_cnt 0→1 which removes
                    // blocks from the free pool. Count how many cached
                    // blocks are evictable (ref_cnt==0) — those will be
                    // "consumed" from the free pool when touched.
                    // Matches Python's `_get_num_evictable_blocks` +
                    // `get_num_blocks_to_allocate`.
                    let num_evictable: usize = cached_ids
                        .iter()
                        .filter(|&&bid| self.ref_cnt[bid] == 0)
                        .count();
                    if additional + num_evictable > self.num_free_blocks {
                        return None;
                    }

                    // Now safe to touch — we verified capacity above.
                    for &bid in &cached_ids {
                        self.touch_block(bid);
                    }

                    let fresh_ids = if additional > 0 {
                        self.allocate_fresh_blocks(additional)?
                    } else {
                        Vec::new()
                    };

                    let mut all_ids = cached_ids;
                    all_ids.extend(fresh_ids);

                    self.req_to_hashes
                        .insert(request.request_id.clone(), hashes);

                    self.allocations
                        .insert(request.request_id.clone(), (vec![all_ids.clone()], needed));
                    return Some(vec![all_ids]);
                }
            }
        }

        let additional = needed.saturating_sub(currently_held);
        if additional == 0 {
            // No new blocks needed — return existing allocation.
            return Some(
                self.allocations
                    .get(&request.request_id)
                    .map(|(blocks, _)| blocks.clone())
                    .unwrap_or_else(|| vec![Vec::new()]),
            );
        }

        let new_block_ids = self.allocate_fresh_blocks(additional)?;

        // Merge with existing allocation.
        let entry = self
            .allocations
            .entry(request.request_id.clone())
            .or_insert_with(|| (vec![Vec::new()], 0));
        entry.0[0].extend(new_block_ids.iter().copied());
        entry.1 = needed;

        // When caching is enabled, record block hashes for full blocks.
        if self.enable_caching {
            let all_tokens = &request.all_token_ids;
            let all_block_ids = &entry.0[0];
            let mut hashes = self
                .req_to_hashes
                .remove(&request.request_id)
                .unwrap_or_default();

            // Hash all full blocks (prompt + decode) that we haven't hashed yet.
            let num_full_blocks = total_tokens / self.block_size;
            for i in hashes.len()..num_full_blocks {
                let start = i * self.block_size;
                let end = start + self.block_size;
                if end <= all_tokens.len() && i < all_block_ids.len() {
                    let hash = Self::hash_block(&all_tokens[start..end]);
                    let bid = all_block_ids[i];
                    self.block_hash_to_id.insert(hash, bid);
                    self.block_id_to_hash.insert(bid, hash);
                    hashes.push(hash);
                }
            }

            if !hashes.is_empty() {
                self.req_to_hashes
                    .insert(request.request_id.clone(), hashes);
            }
        }

        // Return all block IDs for this request (single KV cache group).
        Some(entry.0.clone())
    }

    fn free(&mut self, request_id: &str) {
        if let Some((block_ids_groups, _num_blocks)) = self.allocations.remove(request_id) {
            // Remove per-request hash tracking (hashes stay in block_hash_to_id).
            self.req_to_hashes.remove(request_id);

            // Free blocks in REVERSE order so tail (decode) blocks enter the
            // free queue first and are evicted first, while prefix blocks
            // survive longest. Matches Python's `reversed(req_blocks)`.
            //
            // Uses free_blocks() which decrements ref_cnt and only adds to
            // free queue when ref_cnt drops to 0 (shared blocks stay allocated).
            for group in &block_ids_groups {
                let reversed: Vec<usize> = group.iter().rev().copied().collect();
                self.free_blocks(&reversed);
            }
        }
    }

    fn get_blocks(&self, request_id: &str) -> Vec<Vec<usize>> {
        self.allocations
            .get(request_id)
            .map(|(blocks, _)| blocks.clone())
            .unwrap_or_else(|| vec![Vec::new()])
    }

    fn get_computed_blocks(&self, request: &Request) -> (u32, Vec<Vec<usize>>) {
        if !self.enable_caching {
            return (0, vec![Vec::new()]);
        }

        // Hash full-block-sized chunks of all tokens (prompt + decode) and
        // look for cached matches. A block is a hit if the hash exists in
        // the cache — it may be in the free queue (ref_cnt==0) or shared
        // (ref_cnt>0), both are valid cache hits.
        // Matches Python's `find_longest_cache_hit` which calls
        // `block_pool.get_cached_block()` — returns block regardless of
        // whether it's in the free queue.
        let all_tokens = &request.all_token_ids;
        let mut matched_block_ids = Vec::new();
        let mut num_matched_tokens = 0u32;

        let num_full_blocks = all_tokens.len() / self.block_size;
        for i in 0..num_full_blocks {
            let start = i * self.block_size;
            let end = start + self.block_size;
            let chunk = &all_tokens[start..end];
            let hash = Self::hash_block(chunk);

            if let Some(&block_id) = self.block_hash_to_id.get(&hash) {
                matched_block_ids.push(block_id);
                num_matched_tokens += self.block_size as u32;
            } else {
                break;
            }
        }

        (num_matched_tokens, vec![matched_block_ids])
    }

    fn new_step_starts(&mut self) {
        // Nothing to do for the simple tracker.
    }

    fn reset_prefix_cache(&mut self) -> bool {
        if !self.enable_caching {
            return true;
        }
        self.block_hash_to_id.clear();
        self.block_id_to_hash.clear();
        self.req_to_hashes.clear();
        true
    }

    fn num_free_blocks(&self) -> usize {
        self.num_free_blocks
    }

    fn num_total_blocks(&self) -> usize {
        self.total_blocks
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn usage(&self) -> f64 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        1.0 - self.num_free_blocks as f64 / self.total_blocks as f64
    }

    fn num_cached_blocks(&self) -> usize {
        // Cached blocks = free blocks that have a hash mapping.
        self.block_id_to_hash
            .keys()
            .filter(|&&bid| self.ref_cnt[bid] == 0)
            .count()
    }
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/// The main scheduler implementation.
///
/// Ported from `vllm.v1.core.sched.scheduler.Scheduler`.
///
/// The scheduling algorithm works in two phases:
/// 1. Schedule RUNNING requests -- assign tokens and handle preemption when
///    blocks are exhausted.
/// 2. Schedule WAITING requests -- compute cached blocks, assign tokens, and
///    allocate new blocks.
///
/// The output is a [`SchedulerOutput`] that tells the model runner exactly
/// which requests to process and how many tokens each should get.
pub struct Scheduler {
    // -- Configuration --
    /// Maximum number of requests that can be in the running state.
    max_num_running_reqs: usize,
    /// Maximum number of tokens to schedule in a single step.
    max_num_scheduled_tokens: usize,
    /// Maximum context length supported by the model.
    max_model_len: usize,
    /// Whether chunked prefill is enabled.
    enable_chunked_prefill: bool,
    /// Prefills longer than this threshold are split across steps.
    long_prefill_token_threshold: usize,
    /// Number of speculative lookahead tokens (0 = no speculation).
    num_lookahead_tokens: usize,
    /// Whether async scheduling is enabled (pre-scheduling overlap).
    /// When true, `update_after_schedule` increments `num_output_placeholders`
    /// for non-prefill requests so the next `schedule()` call accounts for
    /// tokens that are in-flight on the GPU but not yet finalized.
    async_scheduling: bool,
    /// Whether pipeline parallelism is active. When true and
    /// `async_scheduling` is false, the scheduler populates `new_token_ids`
    /// in `CachedRequestData`.
    use_pp: bool,

    // -- Request state --
    /// All tracked requests: `req_id -> Request`.
    requests: HashMap<String, Request>,
    /// Queue of requests waiting to be scheduled.
    waiting: Box<dyn RequestQueue>,
    /// Requests currently in the running state.
    running: Vec<Request>,
    /// Index: `req_id -> index in running`. Kept in sync for O(1) lookup.
    running_req_idx: HashMap<String, usize>,
    /// Request IDs finished between the previous and current steps.
    finished_req_ids: HashSet<String>,
    /// Scheduling pause state.
    pause_state: PauseState,

    // -- KV cache --
    /// The KV cache manager (abstracted via trait).
    kv_cache: Box<dyn KVCacheManagerOps>,
}

impl Scheduler {
    /// Create a new scheduler with a `SchedulerConfig` and a KV cache
    /// manager.
    pub fn new(
        scheduler_config: &SchedulerConfig,
        max_model_len: usize,
        kv_cache: Box<dyn KVCacheManagerOps>,
    ) -> Self {
        let policy = match &scheduler_config.policy {
            SchedulerPolicy::Fcfs => SchedulingPolicy::Fcfs,
            SchedulerPolicy::Priority => SchedulingPolicy::Priority,
        };

        let max_num_scheduled_tokens = scheduler_config
            .max_num_scheduled_tokens
            .unwrap_or(scheduler_config.max_num_batched_tokens);

        let async_scheduling = scheduler_config.async_scheduling.unwrap_or(false);
        let use_pp = scheduler_config.use_pp;

        Self {
            max_num_running_reqs: scheduler_config.max_num_seqs,
            max_num_scheduled_tokens,
            max_model_len,
            enable_chunked_prefill: scheduler_config.enable_chunked_prefill,
            long_prefill_token_threshold: scheduler_config.long_prefill_token_threshold,
            num_lookahead_tokens: scheduler_config.num_lookahead_tokens,
            async_scheduling,
            use_pp,

            requests: HashMap::new(),
            waiting: create_request_queue(policy),
            running: Vec::new(),
            running_req_idx: HashMap::new(),
            finished_req_ids: HashSet::new(),
            pause_state: PauseState::Unpaused,

            kv_cache,
        }
    }

    /// Convenience constructor using a [`SimpleBlockTracker`].
    ///
    /// Useful for testing.
    pub fn with_simple_blocks(
        scheduler_config: &SchedulerConfig,
        max_model_len: usize,
        num_gpu_blocks: usize,
        block_size: usize,
    ) -> Self {
        let kv_cache = Box::new(SimpleBlockTracker::new(num_gpu_blocks, block_size));
        Self::new(scheduler_config, max_model_len, kv_cache)
    }

    /// KV cache usage as a fraction in `[0.0, 1.0]`.
    pub fn kv_cache_usage(&self) -> f64 {
        self.kv_cache.usage()
    }

    /// Total number of GPU KV cache blocks.
    pub fn num_total_blocks(&self) -> usize {
        self.kv_cache.num_total_blocks()
    }

    /// Number of GPU KV cache blocks currently in use.
    pub fn num_used_blocks(&self) -> usize {
        self.kv_cache.num_total_blocks() - self.kv_cache.num_free_blocks()
    }

    /// Number of blocks retained in the prefix cache.
    pub fn num_cached_blocks(&self) -> usize {
        self.kv_cache.num_cached_blocks()
    }

    // -- Internal helpers --

    /// Push a request onto the running queue and update the O(1) index.
    fn running_push(&mut self, request: Request) {
        let idx = self.running.len();
        self.running_req_idx.insert(request.request_id.clone(), idx);
        self.running.push(request);
    }

    /// Remove the request at `pos` from the running queue and repair the index.
    ///
    /// After `Vec::remove(pos)`, every element at index > pos shifts down by
    /// one.  We fix those up in O(n) — acceptable because removals are rare
    /// (only on finish / preemption).
    fn running_remove(&mut self, pos: usize) -> Request {
        let req = self.running.remove(pos);
        self.running_req_idx.remove(&req.request_id);
        // Decrement indices for all elements that shifted.
        for idx in self.running_req_idx.values_mut() {
            if *idx > pos {
                *idx -= 1;
            }
        }
        req
    }

    /// Preempt a request: free its KV cache blocks, mark it as preempted,
    /// and move it back to the waiting queue.
    fn preempt_request(&mut self, request: &mut Request) {
        assert_eq!(
            request.status,
            RequestStatus::Running,
            "Only running requests can be preempted"
        );
        self.kv_cache.free(&request.request_id);
        request.status = RequestStatus::Preempted;
        request.num_computed_tokens = 0;
        request.spec_token_ids.clear();
        request.num_preemptions += 1;
        // Async scheduling: reset placeholder count so the request is
        // re-scheduled from scratch after resuming.
        request.num_output_placeholders = 0;

        // Sync to canonical requests map.
        if let Some(canonical) = self.requests.get_mut(&request.request_id) {
            canonical.status = RequestStatus::Preempted;
            canonical.num_computed_tokens = 0;
            canonical.spec_token_ids.clear();
            canonical.num_preemptions = request.num_preemptions;
            canonical.num_output_placeholders = 0;
        }
    }

    /// Build `CachedRequestData` for running + resumed requests.
    fn make_cached_request_data(
        &self,
        running_reqs: &[Request],
        resumed_reqs: &[Request],
        num_scheduled_tokens: &HashMap<String, usize>,
        req_to_new_blocks: &HashMap<String, Vec<Vec<usize>>>,
    ) -> CachedRequestData {
        let mut req_ids = Vec::new();
        let mut resumed_req_ids = HashSet::new();
        let mut new_token_ids = Vec::new();
        let mut new_block_ids = Vec::new();
        let mut num_computed_tokens_vec = Vec::new();
        let mut num_output_tokens_vec = Vec::new();

        // When PP is active and async scheduling is off, the scheduler sends
        // sampled tokens back because there's no direct communication between
        // the first-stage worker and the last-stage worker.
        // Matches Python: vllm/v1/core/sched/scheduler.py _make_cached_request_data
        let send_pp_tokens = self.use_pp && !self.async_scheduling;

        for (idx, req) in running_reqs.iter().chain(resumed_reqs.iter()).enumerate() {
            let is_resumed = idx >= running_reqs.len();
            let req_id = req.request_id.clone();

            if is_resumed {
                resumed_req_ids.insert(req_id.clone());
            }

            if send_pp_tokens {
                let num_tokens = num_scheduled_tokens.get(&req_id).copied().unwrap_or(0);
                let start = req.num_computed_tokens as usize;
                let end = (start + num_tokens).min(req.all_token_ids.len());
                let token_ids = req.all_token_ids[start..end].to_vec();
                new_token_ids.push(token_ids);
            }

            // Block IDs: convert to the output format. Look up before moving req_id.
            let blocks = req_to_new_blocks.get(&req_id).cloned();
            new_block_ids.push(blocks);

            req_ids.push(req_id);

            num_computed_tokens_vec.push(req.num_computed_tokens);
            num_output_tokens_vec
                .push(req.num_output_tokens() as u32 + req.num_output_placeholders);
        }

        CachedRequestData {
            req_ids,
            resumed_req_ids,
            new_token_ids,
            new_block_ids,
            num_computed_tokens: num_computed_tokens_vec,
            num_output_tokens: num_output_tokens_vec,
        }
    }

    /// Update computed token counts after scheduling and clear finished IDs.
    fn update_after_schedule(&mut self, output: &SchedulerOutput) {
        for (req_id, &num_tokens) in &output.num_scheduled_tokens {
            if let Some(request) = self.requests.get_mut(req_id) {
                request.num_computed_tokens += num_tokens as u32;
                request.is_prefill_chunk = (request.num_computed_tokens as usize)
                    < request.num_tokens() + request.num_output_placeholders as usize;

                // Async scheduling: increment placeholders for non-prefill
                // decode requests. Each scheduled decode step will produce
                // 1 token (+ num_spec_tokens draft tokens) whose values are
                // unknown until `update_from_output` runs.
                if self.async_scheduling && !request.is_prefill_chunk {
                    let cur_num_spec_tokens = output
                        .scheduled_spec_decode_tokens
                        .get(req_id)
                        .map(|v| v.len() as u32)
                        .unwrap_or(0);
                    request.num_output_placeholders += 1 + cur_num_spec_tokens;
                }
            }
        }

        // Sync the running list with the authoritative requests map.
        for running_req in &mut self.running {
            if let Some(canonical) = self.requests.get(&running_req.request_id) {
                running_req.num_computed_tokens = canonical.num_computed_tokens;
                running_req.is_prefill_chunk = canonical.is_prefill_chunk;
                running_req.num_output_placeholders = canonical.num_output_placeholders;
            }
        }

        // finished_req_ids already moved into the output via mem::take.
    }

    /// Try to finish a single request. Returns the `(req_id, client_index)`
    /// if the request was found and not already finished.
    fn finish_single_request(
        &mut self,
        request_id: &str,
        status: RequestStatus,
    ) -> Option<(String, u32)> {
        // Check if the request exists.
        let request = self.requests.get(request_id)?;
        if request.status.is_finished() {
            return None;
        }
        let client_index = request.client_index;
        let req_id = request.request_id.clone();

        // Remove from running queue.
        if let Some(&pos) = self.running_req_idx.get(request_id) {
            let mut request = self.running_remove(pos);
            self.kv_cache.free(&request.request_id);
            request.status = status;
            self.requests.insert(request.request_id.clone(), request);
        } else {
            // Remove from waiting queue.
            self.waiting.remove_request(request_id);
            if let Some(r) = self.requests.get_mut(request_id) {
                r.status = status;
            }
        }

        self.finished_req_ids.insert(req_id.clone());
        Some((req_id, client_index))
    }

    // -- Request accessors (used by EngineCore for stop criteria) --

    /// Get a reference to a request by ID.
    pub fn get_request(&self, request_id: &str) -> Option<&Request> {
        self.requests.get(request_id)
    }

    /// Get a mutable reference to a request by ID.
    pub fn get_request_mut(&mut self, request_id: &str) -> Option<&mut Request> {
        self.requests.get_mut(request_id)
    }

    /// Set speculative draft token IDs on a request, updating both the
    /// canonical `requests` map and the `running` list.
    pub fn set_spec_token_ids(&mut self, request_id: &str, spec_token_ids: Vec<u32>) {
        if let Some(request) = self.requests.get_mut(request_id) {
            request.spec_token_ids = spec_token_ids.clone();
        }
        if let Some(&idx) = self.running_req_idx.get(request_id) {
            self.running[idx].spec_token_ids = spec_token_ids;
        }
    }

    /// Rewind `num_computed_tokens` for rejected speculative decode drafts.
    ///
    /// When spec decode drafts are rejected, the scheduler already advanced
    /// `num_computed_tokens` by the full scheduled count (including all drafts).
    /// This method decrements it by `num_rejected` so the KV cache position is
    /// correct for the next step.
    ///
    /// Matches Python's `scheduler.update_from_output()`:
    ///   `request.num_computed_tokens -= num_rejected`
    ///   `request.num_output_placeholders -= num_rejected`
    pub fn rewind_num_computed_tokens(&mut self, request_id: &str, num_rejected: usize) {
        if let Some(request) = self.requests.get_mut(request_id) {
            request.num_computed_tokens = request
                .num_computed_tokens
                .saturating_sub(num_rejected as u32);
            request.num_output_placeholders = request
                .num_output_placeholders
                .saturating_sub(num_rejected as u32);
        }
        if let Some(&idx) = self.running_req_idx.get(request_id) {
            let running_req = &mut self.running[idx];
            running_req.num_computed_tokens = running_req
                .num_computed_tokens
                .saturating_sub(num_rejected as u32);
            running_req.num_output_placeholders = running_req
                .num_output_placeholders
                .saturating_sub(num_rejected as u32);
        }
    }

    /// Append output token IDs to a request, updating both the canonical
    /// `requests` map and the `running` list.
    pub fn append_output_tokens(&mut self, request_id: &str, token_ids: &[u32]) {
        if let Some(request) = self.requests.get_mut(request_id) {
            request.append_output_token_ids(token_ids);
            // Async scheduling: decrement placeholders now that actual
            // tokens have arrived from the GPU.
            request.num_output_placeholders = request
                .num_output_placeholders
                .saturating_sub(token_ids.len() as u32);
        }
        // Also update the running list copy via O(1) index lookup.
        if let Some(&idx) = self.running_req_idx.get(request_id) {
            let running_req = &mut self.running[idx];
            running_req.append_output_token_ids(token_ids);
            running_req.num_output_placeholders = running_req
                .num_output_placeholders
                .saturating_sub(token_ids.len() as u32);
        }
    }
}

impl SchedulerInterface for Scheduler {
    fn schedule(&mut self) -> SchedulerOutput {
        // The scheduling algorithm, ported from Python's Scheduler.schedule():
        //
        // 1. Schedule RUNNING requests first: assign tokens, handle preemption
        //    when blocks run out.
        // 2. Schedule WAITING requests next: compute prefix cache hits,
        //    assign tokens, allocate blocks.
        // 3. Build and return SchedulerOutput.

        let mut scheduled_new_reqs: Vec<Request> = Vec::new();
        let mut scheduled_resumed_reqs: Vec<Request> = Vec::new();
        let mut scheduled_running_reqs: Vec<Request> = Vec::new();
        let mut preempted_reqs: Vec<Request> = Vec::new();

        let mut req_to_new_blocks: HashMap<String, Vec<Vec<usize>>> = HashMap::new();
        let mut num_scheduled_tokens: HashMap<String, usize> = HashMap::new();
        let mut token_budget = self.max_num_scheduled_tokens;
        let mut scheduled_spec_decode_tokens: HashMap<String, Vec<u32>> = HashMap::new();

        if self.pause_state == PauseState::PausedAll {
            token_budget = 0;
        }

        self.kv_cache.new_step_starts();

        // ---------------------------------------------------------------
        // Phase 1: Schedule RUNNING requests
        // ---------------------------------------------------------------
        let mut req_index = 0;
        while req_index < self.running.len() && token_budget > 0 {
            let request = &self.running[req_index];

            // Async scheduling: skip requests that will certainly hit
            // max_tokens once the in-flight step completes. Without this
            // guard the scheduler would schedule one extra decode step
            // because `update_from_output` (which checks the finish
            // condition) hasn't run yet for the previous step.
            //
            // The formula is: (num_computed_tokens + 1) - (num_output_placeholders - 1)
            //   = num_computed_tokens + 2 - num_output_placeholders
            // Since placeholders are included in num_computed_tokens, we
            // subtract (placeholders - 1) to count only the guaranteed
            // minimum tokens (all drafts rejected).
            if request.num_output_placeholders > 0 {
                let effective_computed = request
                    .num_computed_tokens
                    .saturating_add(2)
                    .saturating_sub(request.num_output_placeholders);
                let max_total = request.num_prompt_tokens + request.max_tokens;
                if effective_computed >= max_total {
                    req_index += 1;
                    continue;
                }
            }

            // How many tokens does this request need computed?
            let num_new_tokens_raw = request
                .num_tokens_with_spec()
                .saturating_add(request.num_output_placeholders as usize)
                .saturating_sub(request.num_computed_tokens as usize);

            let mut num_new_tokens = num_new_tokens_raw;

            // Apply long-prefill threshold.
            if self.long_prefill_token_threshold > 0
                && num_new_tokens > self.long_prefill_token_threshold
            {
                num_new_tokens = self.long_prefill_token_threshold;
            }

            // Respect token budget.
            num_new_tokens = num_new_tokens.min(token_budget);

            // Ensure we don't exceed max model length.
            let max_remaining = self
                .max_model_len
                .saturating_sub(1 + request.num_computed_tokens as usize);
            num_new_tokens = num_new_tokens.min(max_remaining);

            if num_new_tokens == 0 {
                req_index += 1;
                continue;
            }

            // Try to allocate blocks. We need to clone the request because
            // allocate_slots takes &Request but we also need &mut self.kv_cache.
            // TODO: Refactor allocate_slots to take minimal fields to avoid this clone.
            let request_clone = self.running[req_index].clone();
            let new_blocks = self.kv_cache.allocate_slots(
                &request_clone,
                num_new_tokens,
                self.num_lookahead_tokens,
            );

            if let Some(blocks) = new_blocks {
                // Successfully allocated.
                scheduled_running_reqs.push(self.running[req_index].clone());
                let request_id = self.running[req_index].request_id.clone();

                // Handle speculative decode tokens.
                if !self.running[req_index].spec_token_ids.is_empty() {
                    let num_scheduled_spec = num_new_tokens
                        .saturating_add(self.running[req_index].num_computed_tokens as usize)
                        .saturating_sub(self.running[req_index].num_tokens());
                    if num_scheduled_spec > 0 {
                        let spec_ids = &self.running[req_index].spec_token_ids;
                        let truncated: Vec<u32> =
                            spec_ids.iter().take(num_scheduled_spec).copied().collect();
                        scheduled_spec_decode_tokens.insert(request_id.clone(), truncated);
                    }
                    // Clear spec tokens for next step (both running list and canonical map).
                    self.running[req_index].spec_token_ids.clear();
                    if let Some(canonical) = self.requests.get_mut(&request_id) {
                        canonical.spec_token_ids.clear();
                    }
                }

                req_to_new_blocks.insert(request_id.clone(), blocks);
                num_scheduled_tokens.insert(request_id, num_new_tokens);
                token_budget -= num_new_tokens;
                req_index += 1;
            } else {
                // Allocation failed -- preempt the last running request.
                // (FCFS: preempt from the back; Priority would pick the
                // lowest-priority request, but for simplicity we always
                // preempt from the back.)
                if self.running.len() <= 1 {
                    // Can't preempt -- the only running request is this one.
                    break;
                }

                // Preempt the last request (lowest priority in FCFS order).
                let preempt_idx = self.running.len() - 1;
                if preempt_idx == req_index {
                    // The request we're trying to schedule is the last one;
                    // preempt it.
                    let mut preempted = self.running_remove(preempt_idx);
                    self.preempt_request(&mut preempted);

                    // Remove from scheduled lists if it was already scheduled.
                    let pid = preempted.request_id.clone();
                    if let Some(tokens) = num_scheduled_tokens.remove(&pid) {
                        token_budget += tokens;
                    }
                    req_to_new_blocks.remove(&pid);
                    scheduled_spec_decode_tokens.remove(&pid);
                    scheduled_running_reqs.retain(|r| r.request_id != pid);

                    // Clone the request for the waiting queue; move into preempted list.
                    self.waiting.prepend_request(preempted.clone());
                    preempted_reqs.push(preempted);
                    break;
                } else {
                    let mut preempted = self.running_remove(preempt_idx);
                    self.preempt_request(&mut preempted);

                    // Restore budget if this request was scheduled.
                    let pid = preempted.request_id.clone();
                    if let Some(tokens) = num_scheduled_tokens.remove(&pid) {
                        token_budget += tokens;
                    }
                    req_to_new_blocks.remove(&pid);
                    scheduled_spec_decode_tokens.remove(&pid);
                    scheduled_running_reqs.retain(|r| r.request_id != pid);

                    self.waiting.prepend_request(preempted.clone());
                    preempted_reqs.push(preempted);
                    // Retry the current request.
                    continue;
                }
            }
        }

        // ---------------------------------------------------------------
        // Phase 2: Schedule WAITING requests
        // ---------------------------------------------------------------
        if preempted_reqs.is_empty() && self.pause_state == PauseState::Unpaused {
            while !self.waiting.is_empty() && token_budget > 0 {
                if self.running.len() >= self.max_num_running_reqs {
                    break;
                }

                // Pop from the waiting queue directly to avoid peek+clone.
                let mut request = match self.waiting.pop_request() {
                    Some(r) => r,
                    None => break,
                };
                let request_id = request.request_id.clone();

                // Get computed blocks from prefix cache.
                let (num_cached_tokens, _cached_blocks) =
                    self.kv_cache.get_computed_blocks(&request);

                // How many tokens need to be scheduled.
                let total_tokens = request.num_tokens();

                // When the prefix cache covers the entire prompt, backing up
                // by one block ensures the model has real input tokens to
                // process. Without this, num_new_tokens would be 0 and the
                // request would be stuck in WAITING forever. Reprocessing the
                // last block is cheap and establishes the decode invariant
                // (num_tokens > num_computed after the first output token is
                // generated).
                let num_computed_tokens =
                    if num_cached_tokens as usize >= total_tokens && num_cached_tokens > 0 {
                        let bs = self.kv_cache.block_size();
                        (((num_cached_tokens as usize) / bs).saturating_sub(1) * bs) as u32
                    } else {
                        num_cached_tokens
                    };

                let num_new_tokens_raw = total_tokens.saturating_sub(num_computed_tokens as usize);

                let mut num_new_tokens = num_new_tokens_raw;

                // Apply long-prefill threshold.
                if self.long_prefill_token_threshold > 0
                    && num_new_tokens > self.long_prefill_token_threshold
                {
                    num_new_tokens = self.long_prefill_token_threshold;
                }

                // If chunked prefill is disabled, skip if the request
                // doesn't fit in the remaining budget.
                if !self.enable_chunked_prefill && num_new_tokens > token_budget {
                    // Put the request back.
                    self.waiting.prepend_request(request);
                    break;
                }

                num_new_tokens = num_new_tokens.min(token_budget);
                if num_new_tokens == 0 {
                    // Put the request back.
                    self.waiting.prepend_request(request);
                    break;
                }

                // Allocate slots for the effective lookahead. For new
                // requests that haven't been computed yet, we use 0
                // lookahead tokens (matches Python: only running requests
                // get lookahead).
                let effective_lookahead = if request.num_computed_tokens == 0 {
                    0
                } else {
                    self.num_lookahead_tokens
                };

                // Set num_computed_tokens for proper block allocation.
                let orig_computed = request.num_computed_tokens;
                request.num_computed_tokens = num_computed_tokens;

                let new_blocks =
                    self.kv_cache
                        .allocate_slots(&request, num_new_tokens, effective_lookahead);

                match new_blocks {
                    Some(blocks) => {
                        // Mutate request to running state.
                        let was_waiting = request.status == RequestStatus::Waiting;
                        let was_preempted = request.status == RequestStatus::Preempted;

                        request.status = RequestStatus::Running;
                        request.num_computed_tokens = num_computed_tokens;
                        if request.num_cached_tokens < 0 {
                            request.num_cached_tokens = num_computed_tokens as i32;
                        }

                        // One clone goes to scheduled list, original goes to running.
                        if was_waiting {
                            scheduled_new_reqs.push(request.clone());
                        } else if was_preempted {
                            scheduled_resumed_reqs.push(request.clone());
                        } else {
                            warn!("Unexpected request status for {}", request.request_id);
                        }

                        // Move to running list (no extra clone).
                        self.running_push(request);

                        req_to_new_blocks.insert(request_id.clone(), blocks);
                        num_scheduled_tokens.insert(request_id.clone(), num_new_tokens);
                        token_budget -= num_new_tokens;

                        // Update the requests map.
                        if let Some(r) = self.requests.get_mut(&request_id) {
                            r.status = RequestStatus::Running;
                            r.num_computed_tokens = num_computed_tokens;
                            if r.num_cached_tokens < 0 {
                                r.num_cached_tokens = num_computed_tokens as i32;
                            }
                        }
                    }
                    None => {
                        // Cannot allocate -- put request back and stop.
                        request.num_computed_tokens = orig_computed;
                        self.waiting.prepend_request(request);
                        break;
                    }
                }
            }
        }

        // ---------------------------------------------------------------
        // Phase 3: Build SchedulerOutput
        // ---------------------------------------------------------------
        let total_num_scheduled_tokens: usize = num_scheduled_tokens.values().sum();

        // Build NewRequestData for newly scheduled requests.
        // Use into_iter() to move fields out instead of cloning.
        let new_reqs_data: Vec<NewRequestData> = scheduled_new_reqs
            .into_iter()
            .map(|req| {
                let blocks = req_to_new_blocks
                    .get(&req.request_id)
                    .cloned()
                    .unwrap_or_else(|| vec![Vec::new()]);
                NewRequestData::new(
                    req.request_id,
                    Some(req.prompt_token_ids),
                    blocks,
                    req.num_computed_tokens,
                    Some(req.sampling_params),
                    req.mm_data,
                )
            })
            .collect();

        // Build CachedRequestData.
        let cached_reqs_data = self.make_cached_request_data(
            &scheduled_running_reqs,
            &scheduled_resumed_reqs,
            &num_scheduled_tokens,
            &req_to_new_blocks,
        );

        let preempted_req_ids: HashSet<String> =
            preempted_reqs.into_iter().map(|r| r.request_id).collect();

        let output = SchedulerOutput {
            scheduled_new_reqs: new_reqs_data,
            scheduled_cached_reqs: cached_reqs_data,
            num_scheduled_tokens,
            total_num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            scheduled_encoder_inputs: HashMap::new(),
            num_common_prefix_blocks: Vec::new(),
            finished_req_ids: std::mem::take(&mut self.finished_req_ids),
            free_encoder_mm_hashes: Vec::new(),
            preempted_req_ids: if preempted_req_ids.is_empty() {
                None
            } else {
                Some(preempted_req_ids)
            },
        };

        // Post-schedule updates.
        self.update_after_schedule(&output);

        output
    }

    fn add_request(&mut self, request: Request) {
        self.requests
            .insert(request.request_id.clone(), request.clone());
        self.waiting.add_request(request);
    }

    fn finish_requests(
        &mut self,
        request_ids: &[&str],
        finished_status: RequestStatus,
    ) -> Vec<(String, u32)> {
        let mut result = Vec::new();
        for &req_id in request_ids {
            if let Some(pair) = self.finish_single_request(req_id, finished_status) {
                result.push(pair);
            }
        }
        result
    }

    fn get_num_unfinished_requests(&self) -> usize {
        self.running.len() + self.waiting.len()
    }

    fn get_unfinished_request_ids(&self) -> Vec<String> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .map(|r| r.request_id.clone())
            .collect()
    }

    fn has_finished_requests(&self) -> bool {
        !self.finished_req_ids.is_empty()
    }

    fn pause_state(&self) -> PauseState {
        self.pause_state
    }

    fn set_pause_state(&mut self, state: PauseState) {
        self.pause_state = state;
    }

    fn reset_prefix_cache(&mut self) -> bool {
        if !self.running.is_empty() {
            return false;
        }
        self.kv_cache.reset_prefix_cache()
    }

    fn get_request_counts(&self) -> (usize, usize) {
        (self.running.len(), self.waiting.len())
    }

    fn kv_cache_usage(&self) -> f64 {
        self.kv_cache.usage()
    }

    fn num_total_blocks(&self) -> usize {
        self.kv_cache.num_total_blocks()
    }

    fn num_used_blocks(&self) -> usize {
        self.kv_cache.num_total_blocks() - self.kv_cache.num_free_blocks()
    }

    fn shutdown(&mut self) {
        // Free all running requests via drain to avoid intermediate Vec<String>.
        self.running_req_idx.clear();
        for req in self.running.drain(..) {
            self.kv_cache.free(&req.request_id);
        }

        // Drain waiting queue.
        self.waiting.drain_all();

        self.requests.clear();
        self.finished_req_ids.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use vllm_common::SamplingParams;
    use vllm_config::SchedulerConfig;

    // Helper: create a default scheduler config for testing.
    fn test_scheduler_config() -> SchedulerConfig {
        SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 4,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            ..Default::default()
        }
    }

    // Helper: create a test request.
    fn make_request(id: &str, num_prompt_tokens: usize) -> Request {
        let prompt: Vec<u32> = (0..num_prompt_tokens as u32).collect();
        Request::new(
            id.into(),
            prompt,
            SamplingParams {
                max_tokens: Some(100),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        )
    }

    fn make_priority_request(
        id: &str,
        num_prompt_tokens: usize,
        priority: i32,
        arrival: f64,
    ) -> Request {
        let prompt: Vec<u32> = (0..num_prompt_tokens as u32).collect();
        Request::new(
            id.into(),
            prompt,
            SamplingParams {
                max_tokens: Some(100),
                ..Default::default()
            },
            arrival,
            0,
            priority,
            None,
        )
    }

    // ----- Basic scheduling tests -----

    #[test]
    fn test_empty_schedule() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        let output = sched.schedule();
        assert_eq!(output.total_num_scheduled_tokens, 0);
        assert!(output.scheduled_new_reqs.is_empty());
        assert_eq!(output.scheduled_cached_reqs.num_reqs(), 0);
    }

    #[test]
    fn test_add_and_schedule_single_request() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        let req = make_request("r1", 10);
        sched.add_request(req);

        assert_eq!(sched.get_num_unfinished_requests(), 1);
        assert!(sched.has_unfinished_requests());
        assert_eq!(sched.get_request_counts(), (0, 1));

        let output = sched.schedule();
        assert_eq!(output.scheduled_new_reqs.len(), 1);
        assert_eq!(output.scheduled_new_reqs[0].req_id, "r1");
        assert_eq!(*output.num_scheduled_tokens.get("r1").unwrap(), 10);
        assert_eq!(output.total_num_scheduled_tokens, 10);
        assert_eq!(sched.get_request_counts(), (1, 0));
    }

    #[test]
    fn test_schedule_multiple_requests() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.add_request(make_request("r2", 20));
        sched.add_request(make_request("r3", 30));

        let output = sched.schedule();
        assert_eq!(output.scheduled_new_reqs.len(), 3);
        assert_eq!(output.total_num_scheduled_tokens, 60);
    }

    #[test]
    fn test_max_num_seqs_limit() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 1024,
            max_num_seqs: 2,
            enable_chunked_prefill: true,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.add_request(make_request("r2", 10));
        sched.add_request(make_request("r3", 10));

        let output = sched.schedule();
        // Only 2 requests should be scheduled due to max_num_seqs=2.
        assert_eq!(output.scheduled_new_reqs.len(), 2);
        assert_eq!(sched.get_request_counts(), (2, 1));
    }

    #[test]
    fn test_token_budget_limit() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 25,
            max_num_seqs: 10,
            enable_chunked_prefill: true,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.add_request(make_request("r2", 10));
        sched.add_request(make_request("r3", 10));

        let output = sched.schedule();
        // Total budget is 25. r1 (10) + r2 (10) = 20. r3 needs 10 but only
        // 5 remain. With chunked prefill, r3 gets 5.
        assert_eq!(output.total_num_scheduled_tokens, 25);
        assert_eq!(output.scheduled_new_reqs.len(), 3);
    }

    #[test]
    fn test_chunked_prefill_disabled() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 25,
            max_num_seqs: 10,
            enable_chunked_prefill: false,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.add_request(make_request("r2", 10));
        sched.add_request(make_request("r3", 10));

        let output = sched.schedule();
        // With chunked prefill disabled, r3 (10 tokens) doesn't fit in
        // the remaining budget (5), so only r1 and r2 are scheduled.
        assert_eq!(output.total_num_scheduled_tokens, 20);
        assert_eq!(output.scheduled_new_reqs.len(), 2);
    }

    // ----- Request lifecycle tests -----

    #[test]
    fn test_request_lifecycle() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Add request.
        sched.add_request(make_request("r1", 10));
        assert_eq!(sched.get_num_unfinished_requests(), 1);

        // Schedule it.
        let _output = sched.schedule();
        assert_eq!(sched.get_request_counts(), (1, 0));

        // Finish it.
        let finished = sched.finish_requests(&["r1"], RequestStatus::FinishedStopped);
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].0, "r1");
        assert_eq!(sched.get_num_unfinished_requests(), 0);
        assert!(sched.has_finished_requests());

        // The next schedule call should report the finished request.
        let output = sched.schedule();
        assert!(output.finished_req_ids.contains("r1"));
    }

    #[test]
    fn test_finish_nonexistent_request() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        let finished = sched.finish_requests(&["nonexistent"], RequestStatus::FinishedAborted);
        assert!(finished.is_empty());
    }

    #[test]
    fn test_finish_waiting_request() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));

        // Finish before scheduling.
        let finished = sched.finish_requests(&["r1"], RequestStatus::FinishedAborted);
        assert_eq!(finished.len(), 1);
        assert_eq!(sched.get_num_unfinished_requests(), 0);
    }

    // ----- Preemption tests -----

    #[test]
    fn test_preemption_when_blocks_exhausted() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 10,
            enable_chunked_prefill: true,
            ..Default::default()
        };
        // Only 4 blocks of size 16 = 64 tokens total.
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 4, 16);

        // Add requests that together need more than 4 blocks.
        sched.add_request(make_request("r1", 16)); // 1 block
        sched.add_request(make_request("r2", 16)); // 1 block
        sched.add_request(make_request("r3", 16)); // 1 block
        sched.add_request(make_request("r4", 16)); // 1 block
        sched.add_request(make_request("r5", 16)); // would need 5th block

        let output = sched.schedule();

        // Not all 5 can be scheduled. At most 4 blocks available.
        assert!(output.total_num_scheduled_tokens <= 64);
        let total_scheduled =
            output.scheduled_new_reqs.len() + output.scheduled_cached_reqs.num_reqs();
        // At most 4 requests can be scheduled.
        assert!(total_scheduled <= 4);
    }

    #[test]
    fn test_running_request_preemption() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 10,
            enable_chunked_prefill: true,
            ..Default::default()
        };
        // Very limited blocks: only 2 blocks of size 16 = 32 tokens.
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 2, 16);

        // First step: schedule r1 (16 tokens, 1 block).
        sched.add_request(make_request("r1", 16));
        let output1 = sched.schedule();
        assert_eq!(output1.scheduled_new_reqs.len(), 1);
        assert_eq!(output1.scheduled_new_reqs[0].req_id, "r1");

        // Simulate r1 generating an output token.
        sched.append_output_tokens("r1", &[99]);

        // Second step: r1 is running (needs 1 token), schedule it.
        let output2 = sched.schedule();
        // r1 should be in cached_reqs (already scheduled before).
        assert!(output2.num_scheduled_tokens.contains_key("r1"));
    }

    // ----- Pause state tests -----

    #[test]
    fn test_pause_all() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.set_pause_state(PauseState::PausedAll);

        let output = sched.schedule();
        assert_eq!(output.total_num_scheduled_tokens, 0);
        assert!(output.scheduled_new_reqs.is_empty());
    }

    #[test]
    fn test_pause_new() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Schedule r1 normally first.
        sched.add_request(make_request("r1", 10));
        let _output1 = sched.schedule();
        assert_eq!(sched.get_request_counts(), (1, 0));

        // Simulate r1 producing a token.
        sched.append_output_tokens("r1", &[99]);

        // Now pause new requests and add r2.
        sched.add_request(make_request("r2", 10));
        sched.set_pause_state(PauseState::PausedNew);

        let output2 = sched.schedule();
        // r1 (running) should still be scheduled, but r2 (waiting) should not.
        assert!(output2.num_scheduled_tokens.contains_key("r1"));
        assert!(!output2.num_scheduled_tokens.contains_key("r2"));
    }

    // ----- Shutdown test -----

    #[test]
    fn test_shutdown() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.add_request(make_request("r2", 20));
        let _output = sched.schedule();

        sched.shutdown();
        assert_eq!(sched.get_num_unfinished_requests(), 0);
        assert!(!sched.has_unfinished_requests());
    }

    // ----- Reset prefix cache test -----

    #[test]
    fn test_reset_prefix_cache_with_running_requests() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        let _output = sched.schedule();

        // Should fail because there are running requests.
        assert!(!sched.reset_prefix_cache());
    }

    #[test]
    fn test_reset_prefix_cache_without_running_requests() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // No running requests -- should succeed.
        assert!(sched.reset_prefix_cache());
    }

    // ----- Long prefill threshold test -----

    #[test]
    fn test_long_prefill_threshold() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 1024,
            max_num_seqs: 10,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 50,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Request with 200 prompt tokens.
        sched.add_request(make_request("r1", 200));

        let output = sched.schedule();
        // Should schedule at most 50 tokens due to the threshold.
        assert_eq!(*output.num_scheduled_tokens.get("r1").unwrap(), 50);
    }

    // ----- Multiple scheduling steps test -----

    #[test]
    fn test_multi_step_scheduling() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 20,
            max_num_seqs: 10,
            enable_chunked_prefill: true,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Request with 50 prompt tokens -- needs multiple steps.
        sched.add_request(make_request("r1", 50));

        // Step 1: schedule up to 20 tokens.
        let output1 = sched.schedule();
        assert_eq!(*output1.num_scheduled_tokens.get("r1").unwrap(), 20);

        // Step 2: schedule next chunk. The request is now running.
        let output2 = sched.schedule();
        assert_eq!(*output2.num_scheduled_tokens.get("r1").unwrap(), 20);

        // Step 3: schedule remaining 10 tokens.
        let output3 = sched.schedule();
        assert_eq!(*output3.num_scheduled_tokens.get("r1").unwrap(), 10);
    }

    // ----- has_requests / has_finished tests -----

    #[test]
    fn test_has_requests_with_finished() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        let _output = sched.schedule();

        // Finish r1.
        sched.finish_requests(&["r1"], RequestStatus::FinishedStopped);

        // No unfinished, but has finished.
        assert!(!sched.has_unfinished_requests());
        assert!(sched.has_finished_requests());
        assert!(sched.has_requests());

        // After a schedule step, finished IDs are flushed.
        let _output = sched.schedule();
        assert!(!sched.has_requests());
    }

    // ----- Priority scheduling tests -----

    #[test]
    fn test_priority_scheduling_order() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 2,
            enable_chunked_prefill: true,
            policy: SchedulerPolicy::Priority,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Add a low-priority request first, then a high-priority one.
        sched.add_request(make_priority_request("r_low", 10, 10, 1.0));
        sched.add_request(make_priority_request("r_high", 10, -1, 2.0));

        let output = sched.schedule();
        assert_eq!(output.scheduled_new_reqs.len(), 2);

        // Both should be scheduled since max_num_seqs=2,
        // but r_high should come first in the new_reqs list because the
        // priority queue pops it first.
        let scheduled_ids: Vec<&str> = output
            .scheduled_new_reqs
            .iter()
            .map(|r| r.req_id.as_str())
            .collect();
        assert_eq!(scheduled_ids[0], "r_high");
        assert_eq!(scheduled_ids[1], "r_low");
    }

    #[test]
    fn test_priority_scheduling_with_limited_seqs() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 1, // Only 1 seq at a time.
            enable_chunked_prefill: true,
            policy: SchedulerPolicy::Priority,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // The high-priority request should be scheduled first.
        sched.add_request(make_priority_request("r_low", 10, 10, 1.0));
        sched.add_request(make_priority_request("r_high", 10, -1, 2.0));

        let output = sched.schedule();
        assert_eq!(output.scheduled_new_reqs.len(), 1);
        assert_eq!(output.scheduled_new_reqs[0].req_id, "r_high");
    }

    // -- Request accessor tests --

    #[test]
    fn test_get_request() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 5));
        assert!(sched.get_request("r1").is_some());
        assert_eq!(sched.get_request("r1").unwrap().request_id, "r1");
        assert!(sched.get_request("nonexistent").is_none());
    }

    #[test]
    fn test_get_request_mut() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 5));
        if let Some(req) = sched.get_request_mut("r1") {
            req.max_tokens = 42;
        }
        assert_eq!(sched.get_request("r1").unwrap().max_tokens, 42);
    }

    #[test]
    fn test_append_output_tokens() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 5));

        // Schedule to move from waiting to running.
        let _ = sched.schedule();

        // Append output tokens.
        sched.append_output_tokens("r1", &[100, 101]);

        let req = sched.get_request("r1").unwrap();
        assert_eq!(req.output_token_ids, vec![100, 101]);
        assert_eq!(req.num_output_tokens(), 2);
        // all_token_ids should be prompt + output.
        assert_eq!(req.all_token_ids.len(), 7); // 5 prompt + 2 output
    }

    #[test]
    fn test_append_output_tokens_nonexistent() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Appending to a nonexistent request should not panic.
        sched.append_output_tokens("nonexistent", &[1, 2, 3]);
    }

    // ----- KV cache usage tests -----

    #[test]
    fn test_simple_block_tracker_usage_empty() {
        let tracker = SimpleBlockTracker::new(100, 16);
        assert!((tracker.usage() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_simple_block_tracker_usage_zero_blocks() {
        let tracker = SimpleBlockTracker::new(0, 16);
        assert!((tracker.usage() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_simple_block_tracker_usage_after_alloc() {
        let mut tracker = SimpleBlockTracker::new(100, 16);
        // Allocate a request needing 2 blocks (32 tokens / 16 block_size).
        let req = make_request("r1", 32);
        tracker.allocate_slots(&req, 32, 0);
        // 2 out of 100 blocks used = 0.02
        assert!((tracker.usage() - 0.02).abs() < 1e-9);
    }

    #[test]
    fn test_simple_block_tracker_usage_after_free() {
        let mut tracker = SimpleBlockTracker::new(100, 16);
        let req = make_request("r1", 32);
        tracker.allocate_slots(&req, 32, 0);
        tracker.free("r1");
        assert!((tracker.usage() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scheduler_kv_cache_usage() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Initially empty.
        assert!((sched.kv_cache_usage() - 0.0).abs() < f64::EPSILON);

        // Add and schedule a request to trigger block allocation.
        let req = make_request("r1", 32);
        sched.add_request(req);
        sched.schedule();

        // After scheduling, some blocks should be allocated.
        assert!(sched.kv_cache_usage() > 0.0);
    }

    // ----- Async scheduling tests -----

    fn async_scheduler_config() -> SchedulerConfig {
        SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 4,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            async_scheduling: Some(true),
            ..Default::default()
        }
    }

    /// Helper: create a request with a specific max_tokens.
    fn make_request_with_max(id: &str, num_prompt_tokens: usize, max_tokens: u32) -> Request {
        let prompt: Vec<u32> = (0..num_prompt_tokens as u32).collect();
        Request::new(
            id.into(),
            prompt,
            SamplingParams {
                max_tokens: Some(max_tokens),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        )
    }

    #[test]
    fn test_async_placeholder_increment_on_decode() {
        let cfg = async_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Add request, schedule prefill (non-chunked).
        sched.add_request(make_request("r1", 10));
        let output1 = sched.schedule();

        // After a completed (non-chunked) prefill, is_prefill_chunk=false
        // so placeholders are incremented to 1 (the prefill step will
        // produce 1 output token whose value is unknown).
        assert_eq!(output1.scheduled_new_reqs.len(), 1);
        let req = sched.get_request("r1").unwrap();
        assert_eq!(req.num_output_placeholders, 1);

        // Simulate prefill producing a first output token.
        sched.append_output_tokens("r1", &[99]);
        assert_eq!(sched.get_request("r1").unwrap().num_output_placeholders, 0);

        // Now schedule decode step 2.
        let output2 = sched.schedule();
        assert!(output2.num_scheduled_tokens.contains_key("r1"));

        // After schedule, placeholder incremented again for the decode.
        let req = sched.get_request("r1").unwrap();
        assert_eq!(
            req.num_output_placeholders, 1,
            "Decode request should have 1 placeholder after async schedule"
        );
    }

    #[test]
    fn test_async_placeholder_decrement_on_output() {
        let cfg = async_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.schedule(); // prefill

        // First decode token from prefill.
        sched.append_output_tokens("r1", &[99]);

        // Schedule decode (sets placeholder = 1).
        sched.schedule();
        assert_eq!(sched.get_request("r1").unwrap().num_output_placeholders, 1);

        // Simulate GPU returning 1 token → decrement placeholder.
        sched.append_output_tokens("r1", &[100]);
        assert_eq!(sched.get_request("r1").unwrap().num_output_placeholders, 0);
    }

    #[test]
    fn test_async_back_to_back_schedule_no_double_schedule() {
        // Core test: two consecutive schedule() calls without
        // update_from_output in between must not double-schedule the
        // same position.
        let cfg = async_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));

        // Prefill.
        let output1 = sched.schedule();
        assert_eq!(*output1.num_scheduled_tokens.get("r1").unwrap(), 10);

        // Simulate prefill producing first token.
        sched.append_output_tokens("r1", &[99]);

        // First decode schedule → r1 needs 1 token.
        let output2 = sched.schedule();
        assert_eq!(
            *output2.num_scheduled_tokens.get("r1").unwrap(),
            1,
            "First decode step should schedule 1 token"
        );
        // After schedule, placeholder = 1.

        // Second schedule() WITHOUT update_from_output — simulates
        // pre-scheduling while GPU is still running.
        let output3 = sched.schedule();
        assert_eq!(
            *output3.num_scheduled_tokens.get("r1").unwrap(),
            1,
            "Second decode should schedule 1 token (placeholder prevents overlap)"
        );

        // Placeholder should now be 2 (one for each in-flight step).
        let req = sched.get_request("r1").unwrap();
        assert_eq!(req.num_output_placeholders, 2);
    }

    #[test]
    fn test_async_max_tokens_guard_prevents_over_scheduling() {
        let cfg = async_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Request with only 2 max_tokens.
        sched.add_request(make_request_with_max("r1", 10, 2));
        sched.schedule(); // prefill (10 tokens)

        // Simulate generating 1st output token.
        sched.append_output_tokens("r1", &[99]);

        // Decode step 1 → schedules 1 token, placeholder = 1.
        let output2 = sched.schedule();
        assert_eq!(*output2.num_scheduled_tokens.get("r1").unwrap(), 1);
        assert_eq!(sched.get_request("r1").unwrap().num_output_placeholders, 1);

        // Now pre-schedule step 2 (without finalization).
        // num_computed = 12 (10 prompt + 1 output + 1 placeholder),
        // max_total = 10 + 2 = 12.
        // Guard: 12 + 2 - 1 = 13 >= 12 → skip!
        let output3 = sched.schedule();
        assert!(
            !output3.num_scheduled_tokens.contains_key("r1"),
            "Max-tokens guard should prevent scheduling request about to finish"
        );
    }

    #[test]
    fn test_async_preemption_resets_placeholders() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 10,
            enable_chunked_prefill: true,
            async_scheduling: Some(true),
            ..Default::default()
        };
        // 3 blocks of size 8 = 24 tokens of KV capacity.
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 3, 8);

        // Schedule r1 (8 tokens = 1 block) and r2 (8 tokens = 1 block).
        sched.add_request(make_request("r1", 8));
        sched.add_request(make_request("r2", 8));
        sched.schedule(); // prefill both

        // Both should have placeholder = 1 after completed prefill.
        assert_eq!(sched.get_request("r1").unwrap().num_output_placeholders, 1);
        assert_eq!(sched.get_request("r2").unwrap().num_output_placeholders, 1);

        // Simulate output tokens for both.
        sched.append_output_tokens("r1", &[99]);
        sched.append_output_tokens("r2", &[99]);
        assert_eq!(sched.get_request("r1").unwrap().num_output_placeholders, 0);
        assert_eq!(sched.get_request("r2").unwrap().num_output_placeholders, 0);

        // Generate several more tokens for r1 to consume more blocks.
        // r1 now has 8 prompt + 1 output = 9 tokens → 2 blocks.
        // r2 has 8 prompt + 1 output = 9 tokens → 2 blocks.
        // Total: 4 blocks needed but only 3 available.

        // Schedule decode step: this should trigger preemption because
        // r1 (2 blocks) + r2 (2 blocks) > 3 blocks.
        let output = sched.schedule();

        // One of the two should be preempted. Check which.
        // (FCFS preempts from the back, so r2 should be preempted.)
        if sched.get_request("r2").unwrap().status == RequestStatus::Preempted {
            assert_eq!(
                sched.get_request("r2").unwrap().num_output_placeholders,
                0,
                "Preemption must reset placeholders"
            );
        } else if sched.get_request("r1").unwrap().status == RequestStatus::Preempted {
            assert_eq!(
                sched.get_request("r1").unwrap().num_output_placeholders,
                0,
                "Preemption must reset placeholders"
            );
        }

        // Only one should be scheduled.
        assert!(
            output.num_scheduled_tokens.len() <= 2,
            "At most 2 requests should be scheduled with limited blocks"
        );
    }

    #[test]
    fn test_async_placeholder_saturating_decrement() {
        // Ensure decrement never underflows.
        let cfg = async_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.schedule(); // prefill

        // Placeholders = 0. Appending tokens should not underflow.
        sched.append_output_tokens("r1", &[99, 100, 101]);
        assert_eq!(sched.get_request("r1").unwrap().num_output_placeholders, 0);
    }

    #[test]
    fn test_sync_mode_no_placeholders() {
        // Verify that without async_scheduling, placeholders are never set.
        let cfg = test_scheduler_config(); // async_scheduling = None (false)
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.schedule(); // prefill

        sched.append_output_tokens("r1", &[99]);

        sched.schedule(); // decode
        assert_eq!(
            sched.get_request("r1").unwrap().num_output_placeholders,
            0,
            "Sync mode should never set placeholders"
        );
    }

    // -----------------------------------------------------------------------
    // Prefix caching tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefix_cache_basic() {
        // Two requests with the same 32-token prompt should share cached blocks.
        let block_size = 16;
        let mut tracker = SimpleBlockTracker::with_caching(100, block_size);

        // Make a 32-token prompt (2 full blocks).
        let prompt: Vec<u32> = (0..32).collect();
        let mut r1 = Request::new(
            "r1".into(),
            prompt.clone(),
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );

        // First request: no cache hits.
        let (cached, _) = tracker.get_computed_blocks(&r1);
        assert_eq!(cached, 0);

        // Allocate for r1.
        let blocks = tracker.allocate_slots(&r1, 32, 0).unwrap();
        assert_eq!(blocks[0].len(), 2);
        let block0 = blocks[0][0];
        let block1 = blocks[0][1];

        // Simulate completion: r1 fully computed and freed.
        r1.num_computed_tokens = 32;
        tracker.free("r1");

        // Now both blocks should be in the cache.
        assert_eq!(tracker.num_cached_blocks(), 2);

        // Second request with the same prompt.
        let r2 = Request::new(
            "r2".into(),
            prompt.clone(),
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            2.0,
            0,
            0,
            None,
        );

        // Should get 32 cached tokens.
        let (cached2, cached_blocks2) = tracker.get_computed_blocks(&r2);
        assert_eq!(cached2, 32);
        assert_eq!(cached_blocks2[0].len(), 2);
        assert_eq!(cached_blocks2[0][0], block0);
        assert_eq!(cached_blocks2[0][1], block1);
    }

    #[test]
    fn test_prefix_cache_partial_match() {
        // Request with longer prompt should match the shared prefix.
        let block_size = 16;
        let mut tracker = SimpleBlockTracker::with_caching(100, block_size);

        // First request: 32 tokens.
        let prompt32: Vec<u32> = (0..32).collect();
        let mut r1 = Request::new(
            "r1".into(),
            prompt32.clone(),
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        tracker.allocate_slots(&r1, 32, 0).unwrap();
        r1.num_computed_tokens = 32;
        tracker.free("r1");

        // Second request: 48 tokens, first 32 overlap.
        let prompt48: Vec<u32> = (0..48).collect();
        let r2 = Request::new(
            "r2".into(),
            prompt48,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            2.0,
            0,
            0,
            None,
        );
        let (cached, _) = tracker.get_computed_blocks(&r2);
        assert_eq!(cached, 32, "first 2 blocks should match");
    }

    #[test]
    fn test_prefix_cache_no_match_different_prompt() {
        let block_size = 16;
        let mut tracker = SimpleBlockTracker::with_caching(100, block_size);

        let prompt_a: Vec<u32> = (0..32).collect();
        let mut r1 = Request::new(
            "r1".into(),
            prompt_a,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        tracker.allocate_slots(&r1, 32, 0).unwrap();
        r1.num_computed_tokens = 32;
        tracker.free("r1");

        // Different prompt — no match.
        let prompt_b: Vec<u32> = (100..132).collect();
        let r2 = Request::new(
            "r2".into(),
            prompt_b,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            2.0,
            0,
            0,
            None,
        );
        let (cached, _) = tracker.get_computed_blocks(&r2);
        assert_eq!(cached, 0);
    }

    #[test]
    fn test_prefix_cache_eviction_under_pressure() {
        // With limited blocks, cached blocks get evicted FIFO.
        let block_size = 16;
        let mut tracker = SimpleBlockTracker::with_caching(4, block_size);

        // Fill all 4 blocks with r1 (64 tokens).
        let prompt64: Vec<u32> = (0..64).collect();
        let mut r1 = Request::new(
            "r1".into(),
            prompt64,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        tracker.allocate_slots(&r1, 64, 0).unwrap();
        r1.num_computed_tokens = 64;
        tracker.free("r1");
        assert_eq!(tracker.num_cached_blocks(), 4);

        // New request needs 2 blocks — must evict 2 cached blocks.
        let prompt_new: Vec<u32> = (200..232).collect();
        let r2 = Request::new(
            "r2".into(),
            prompt_new,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            2.0,
            0,
            0,
            None,
        );
        let blocks = tracker.allocate_slots(&r2, 32, 0);
        assert!(blocks.is_some(), "should evict cached blocks to make room");
        assert_eq!(
            tracker.num_cached_blocks(),
            2,
            "2 of 4 cached blocks should remain"
        );
    }

    #[test]
    fn test_prefix_cache_reset() {
        let block_size = 16;
        let mut tracker = SimpleBlockTracker::with_caching(100, block_size);

        let prompt: Vec<u32> = (0..32).collect();
        let mut r1 = Request::new(
            "r1".into(),
            prompt.clone(),
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        tracker.allocate_slots(&r1, 32, 0).unwrap();
        r1.num_computed_tokens = 32;
        tracker.free("r1");
        assert_eq!(tracker.num_cached_blocks(), 2);

        tracker.reset_prefix_cache();
        assert_eq!(tracker.num_cached_blocks(), 0);

        // After reset, no cache hits.
        let r2 = Request::new(
            "r2".into(),
            prompt,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            2.0,
            0,
            0,
            None,
        );
        let (cached, _) = tracker.get_computed_blocks(&r2);
        assert_eq!(cached, 0);
    }

    #[test]
    fn test_prefix_cache_allocate_reuses_cached_blocks() {
        // When allocating for a request with cached prefix, the cached block IDs
        // should appear in the allocation.
        let block_size = 16;
        let mut tracker = SimpleBlockTracker::with_caching(100, block_size);

        let prompt: Vec<u32> = (0..32).collect();
        let mut r1 = Request::new(
            "r1".into(),
            prompt.clone(),
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        let r1_blocks = tracker.allocate_slots(&r1, 32, 0).unwrap();
        let cached_block0 = r1_blocks[0][0];
        let cached_block1 = r1_blocks[0][1];
        r1.num_computed_tokens = 32;
        tracker.free("r1");

        // r2 has same prompt + 8 extra tokens = 40 tokens total.
        let prompt40: Vec<u32> = (0..40).collect();
        let mut r2 = Request::new(
            "r2".into(),
            prompt40,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            2.0,
            0,
            0,
            None,
        );
        // Simulate what the scheduler does: get_computed_blocks, then set
        // num_computed_tokens, then allocate.
        let (cached, _) = tracker.get_computed_blocks(&r2);
        assert_eq!(cached, 32);
        r2.num_computed_tokens = cached;

        let r2_blocks = tracker.allocate_slots(&r2, 8, 0).unwrap();
        // Should reuse the 2 cached blocks + 1 new block.
        assert_eq!(r2_blocks[0].len(), 3);
        assert_eq!(r2_blocks[0][0], cached_block0);
        assert_eq!(r2_blocks[0][1], cached_block1);
    }

    #[test]
    fn test_block_id_recycling() {
        // Verify that block IDs are recycled and stay within bounds.
        // With 4 blocks, serve many sequential requests — IDs must stay in 0..4.
        let block_size = 16;
        let mut tracker = SimpleBlockTracker::new(4, block_size);

        for round in 0..20 {
            let prompt: Vec<u32> = (0..16).collect();
            let id = format!("r{round}");
            let r = Request::new(
                id.clone(),
                prompt,
                SamplingParams {
                    max_tokens: Some(1),
                    ..Default::default()
                },
                round as f64,
                0,
                0,
                None,
            );
            let blocks = tracker.allocate_slots(&r, 16, 0).expect("should allocate");
            // Block IDs must be within pool bounds.
            for &bid in &blocks[0] {
                assert!(bid < 4, "block ID {bid} out of bounds (round {round})");
            }
            tracker.free(&id);
        }
    }

    #[test]
    fn test_prefix_cache_full_prompt_not_stuck() {
        // When the prefix cache covers the entire prompt, the scheduler
        // must still schedule at least one block of tokens so the request
        // can start decoding. Regression test for the bench-latency hang.
        let block_size = 16;
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 4,
            enable_chunked_prefill: true,
            ..Default::default()
        };
        let kv: Box<dyn KVCacheManagerOps> =
            Box::new(SimpleBlockTracker::with_caching(100, block_size));
        let mut sched = Scheduler::new(&cfg, 8192, kv);

        // 32-token prompt = 2 full blocks with block_size=16.
        let prompt: Vec<u32> = (0..32).collect();

        // --- Iteration 1: no cache, should schedule 32 tokens ---
        let r1 = Request::new(
            "r1".into(),
            prompt.clone(),
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        sched.add_request(r1);
        let output1 = sched.schedule();
        let sched_tokens_1 = *output1.num_scheduled_tokens.get("r1").unwrap();
        assert_eq!(
            sched_tokens_1, 32,
            "first request should schedule all 32 prompt tokens"
        );

        // Simulate completion: update computed tokens, finish, free.
        sched.finish_requests(&["r1"], RequestStatus::FinishedStopped);
        // Consume the finished ID from the scheduler.
        let _ = sched.schedule();

        // --- Iteration 2: same prompt, fully cached ---
        let r2 = Request::new(
            "r2".into(),
            prompt.clone(),
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            2.0,
            0,
            0,
            None,
        );
        sched.add_request(r2);
        let output2 = sched.schedule();

        // Must schedule >0 tokens (the last block gets reprocessed).
        let sched_tokens_2 = *output2
            .num_scheduled_tokens
            .get("r2")
            .expect("r2 should be scheduled");
        assert!(
            sched_tokens_2 > 0,
            "fully-cached request must still schedule tokens, got 0"
        );
        // Specifically: we back up by one block, so 16 tokens get reprocessed.
        assert_eq!(
            sched_tokens_2, 16,
            "should reprocess the last block ({block_size} tokens)"
        );
    }

    #[test]
    fn test_block_id_recycling_with_caching() {
        // Same as above but with caching — evicted cached blocks must also
        // produce IDs within bounds.
        let block_size = 16;
        let mut tracker = SimpleBlockTracker::with_caching(4, block_size);

        for round in 0..20 {
            // Each request uses a different prompt so no cache hits.
            let base = (round * 16) as u32 + 1000;
            let prompt: Vec<u32> = (base..base + 16).collect();
            let id = format!("r{round}");
            let mut r = Request::new(
                id.clone(),
                prompt,
                SamplingParams {
                    max_tokens: Some(1),
                    ..Default::default()
                },
                round as f64,
                0,
                0,
                None,
            );
            let blocks = tracker.allocate_slots(&r, 16, 0).expect("should allocate");
            for &bid in &blocks[0] {
                assert!(bid < 4, "block ID {bid} out of bounds (round {round})");
            }
            r.num_computed_tokens = 16;
            tracker.free(&id);
        }
    }

    // -----------------------------------------------------------------------
    // Speculative decoding tests
    // -----------------------------------------------------------------------

    fn spec_decode_scheduler_config() -> SchedulerConfig {
        SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 4,
            enable_chunked_prefill: true,
            num_lookahead_tokens: 5,
            ..Default::default()
        }
    }

    #[test]
    fn test_spec_decode_tokens_scheduled() {
        // Verify that spec_token_ids are included in the scheduling output.
        let cfg = spec_decode_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.schedule(); // prefill
        sched.append_output_tokens("r1", &[99]);

        // Set spec tokens on the request.
        sched.set_spec_token_ids("r1", vec![100, 101, 102, 103, 104]);

        let output = sched.schedule(); // decode with spec tokens

        // Should schedule 1 (real token) + 5 (spec tokens) = 6 new tokens.
        let num_scheduled = output.num_scheduled_tokens.get("r1").copied().unwrap_or(0);
        assert_eq!(num_scheduled, 6, "Should schedule 1 real + 5 spec tokens");

        // Spec tokens should appear in scheduled_spec_decode_tokens.
        let spec_tokens = output.scheduled_spec_decode_tokens.get("r1").unwrap();
        assert_eq!(spec_tokens, &vec![100, 101, 102, 103, 104]);
    }

    #[test]
    fn test_spec_decode_tokens_cleared_after_schedule() {
        // After scheduling, spec_token_ids should be cleared.
        let cfg = spec_decode_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.schedule(); // prefill
        sched.append_output_tokens("r1", &[99]);

        sched.set_spec_token_ids("r1", vec![100, 101, 102]);

        sched.schedule(); // consumes spec tokens

        // Spec tokens should be cleared.
        let req = sched.get_request("r1").unwrap();
        assert!(
            req.spec_token_ids.is_empty(),
            "spec_token_ids should be cleared after schedule"
        );
    }

    #[test]
    fn test_spec_decode_tokens_truncated_by_budget() {
        // When token budget is limited, spec tokens should be truncated.
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 4, // Very tight budget
            max_num_seqs: 4,
            enable_chunked_prefill: true,
            num_lookahead_tokens: 5,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 2));
        sched.schedule(); // prefill (2 tokens)
        sched.append_output_tokens("r1", &[99]);

        sched.set_spec_token_ids("r1", vec![100, 101, 102, 103, 104]); // 5 draft tokens

        let output = sched.schedule(); // budget = 4, need 1+5=6, should truncate

        let num_scheduled = output.num_scheduled_tokens.get("r1").copied().unwrap_or(0);
        assert!(
            num_scheduled <= 4,
            "num_scheduled_tokens ({num_scheduled}) should not exceed budget (4)"
        );

        // Spec tokens in output should be truncated accordingly.
        if let Some(spec_tokens) = output.scheduled_spec_decode_tokens.get("r1") {
            assert!(
                spec_tokens.len() < 5,
                "Spec tokens should be truncated when budget is tight"
            );
        }
    }

    #[test]
    fn test_spec_decode_lookahead_allocates_blocks() {
        // Verify that num_lookahead_tokens causes extra block allocation.
        // With lookahead=5, a decode step needs more blocks than without.
        let cfg_no_spec = SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 4,
            num_lookahead_tokens: 0,
            ..Default::default()
        };
        let cfg_spec = spec_decode_scheduler_config(); // lookahead=5

        // Use very limited blocks to see the difference.
        let mut sched_no = Scheduler::with_simple_blocks(&cfg_no_spec, 8192, 2, 16);
        let mut sched_sp = Scheduler::with_simple_blocks(&cfg_spec, 8192, 2, 16);

        // Add a request that fills nearly all blocks during prefill.
        sched_no.add_request(make_request("r1", 15));
        sched_sp.add_request(make_request("r1", 15));

        let out_no = sched_no.schedule(); // prefill
        let out_sp = sched_sp.schedule(); // prefill

        assert!(out_no.num_scheduled_tokens.contains_key("r1"));
        assert!(out_sp.num_scheduled_tokens.contains_key("r1"));

        // Append one token to move to decode.
        sched_no.append_output_tokens("r1", &[99]);
        sched_sp.append_output_tokens("r1", &[99]);

        // Decode step: without lookahead, 1 token is easy.
        // With lookahead=5, we need 1+5=6 slots which might be tight.
        let out_no = sched_no.schedule();
        let out_sp = sched_sp.schedule();

        // Both should schedule (we have enough blocks), but this confirms
        // lookahead is wired into the allocation path.
        assert!(
            out_no.num_scheduled_tokens.contains_key("r1"),
            "No-spec should schedule"
        );
        assert!(
            out_sp.num_scheduled_tokens.contains_key("r1"),
            "Spec should schedule"
        );
    }

    #[test]
    fn test_spec_decode_mixed_batch() {
        // Some requests have spec tokens, some don't.
        let cfg = spec_decode_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.add_request(make_request("r2", 10));

        sched.schedule(); // prefill both
        sched.append_output_tokens("r1", &[99]);
        sched.append_output_tokens("r2", &[99]);

        // Only r1 has spec tokens.
        sched.set_spec_token_ids("r1", vec![100, 101, 102]);

        let output = sched.schedule();

        // r1 should have 4 tokens (1 real + 3 spec).
        let n1 = output.num_scheduled_tokens.get("r1").copied().unwrap_or(0);
        assert_eq!(n1, 4, "r1 should schedule 1+3 tokens");

        // r2 should have 1 token (normal decode).
        let n2 = output.num_scheduled_tokens.get("r2").copied().unwrap_or(0);
        assert_eq!(n2, 1, "r2 should schedule 1 token");

        // Only r1 should have spec decode tokens in output.
        assert!(output.scheduled_spec_decode_tokens.contains_key("r1"));
        assert!(!output.scheduled_spec_decode_tokens.contains_key("r2"));
    }

    #[test]
    fn test_spec_decode_num_tokens_with_spec() {
        // Verify num_tokens_with_spec() matches scheduler's expectation.
        let mut req = make_request("r1", 10);
        assert_eq!(req.num_tokens_with_spec(), 10);

        req.spec_token_ids = vec![100, 101, 102];
        assert_eq!(req.num_tokens_with_spec(), 13);

        req.spec_token_ids.clear();
        assert_eq!(req.num_tokens_with_spec(), 10);
    }

    #[test]
    fn test_spec_decode_rewind_on_rejection() {
        // Verify that rewind_num_computed_tokens correctly decrements
        // num_computed_tokens when spec decode drafts are rejected.
        // This matches Python's scheduler.update_from_output().
        let cfg = spec_decode_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.schedule(); // prefill: num_computed_tokens = 10
        sched.append_output_tokens("r1", &[99]);

        // Set 5 spec tokens and schedule them.
        sched.set_spec_token_ids("r1", vec![100, 101, 102, 103, 104]);
        let output = sched.schedule(); // num_computed_tokens += 6 (1 real + 5 spec)

        let num_scheduled = output.num_scheduled_tokens.get("r1").copied().unwrap_or(0);
        assert_eq!(num_scheduled, 6);

        // After schedule: num_computed_tokens = 10 + 6 = 16.
        let req = sched.get_request("r1").unwrap();
        assert_eq!(req.num_computed_tokens, 16);

        // Simulate partial rejection: only 2 of 5 drafts accepted (3 rejected).
        // The model would return 3 tokens (2 accepted + 1 recovered/bonus).
        sched.rewind_num_computed_tokens("r1", 3);

        let req = sched.get_request("r1").unwrap();
        assert_eq!(
            req.num_computed_tokens, 13,
            "num_computed_tokens should be 16 - 3 = 13"
        );
    }

    #[test]
    fn test_spec_decode_rewind_all_rejected() {
        // All drafts rejected: rewind by full draft count.
        let cfg = spec_decode_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.schedule(); // prefill: num_computed = 10
        sched.append_output_tokens("r1", &[99]);

        sched.set_spec_token_ids("r1", vec![100, 101, 102]);
        sched.schedule(); // num_computed = 10 + 4 = 14

        // All 3 drafts rejected: model returns 1 token (recovered).
        // num_accepted = 0, num_rejected = 3.
        sched.rewind_num_computed_tokens("r1", 3);

        let req = sched.get_request("r1").unwrap();
        assert_eq!(
            req.num_computed_tokens, 11,
            "num_computed_tokens should be 14 - 3 = 11"
        );
    }

    #[test]
    fn test_spec_decode_rewind_none_rejected() {
        // All drafts accepted: no rewind needed.
        let cfg = spec_decode_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.schedule();
        sched.append_output_tokens("r1", &[99]);

        sched.set_spec_token_ids("r1", vec![100, 101]);
        sched.schedule(); // num_computed = 10 + 3 = 13

        // All 2 drafts accepted: model returns 3 tokens (2 accepted + bonus).
        // num_accepted = 2, num_rejected = 0. No rewind.
        sched.rewind_num_computed_tokens("r1", 0);

        let req = sched.get_request("r1").unwrap();
        assert_eq!(req.num_computed_tokens, 13, "no rewind when all accepted");
    }

    // -----------------------------------------------------------------------
    // Pipeline parallelism tests
    // -----------------------------------------------------------------------

    fn pp_scheduler_config() -> SchedulerConfig {
        SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 4,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            async_scheduling: Some(false),
            use_pp: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_pp_new_token_ids_populated_on_decode() {
        // When use_pp=true and async_scheduling=false, the scheduler must
        // populate new_token_ids in CachedRequestData so non-last PP stages
        // can embed the correct token.
        let cfg = pp_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Add request with 10 prompt tokens.
        sched.add_request(make_request("r1", 10));

        // Step 1: Prefill. No cached reqs yet.
        let output1 = sched.schedule();
        assert_eq!(output1.scheduled_new_reqs.len(), 1);
        assert!(output1.scheduled_cached_reqs.req_ids.is_empty());

        // Simulate prefill producing token 42.
        sched.append_output_tokens("r1", &[42]);

        // Step 2: Decode. Request is now cached.
        let output2 = sched.schedule();
        assert_eq!(output2.scheduled_cached_reqs.req_ids.len(), 1);
        assert_eq!(output2.scheduled_cached_reqs.req_ids[0], "r1");

        // new_token_ids should contain the token the worker needs to embed.
        assert_eq!(output2.scheduled_cached_reqs.new_token_ids.len(), 1);
        assert!(
            !output2.scheduled_cached_reqs.new_token_ids[0].is_empty(),
            "PP sync scheduling must populate new_token_ids for cached requests"
        );
        // The token should be 42 (the one we appended).
        assert!(
            output2.scheduled_cached_reqs.new_token_ids[0].contains(&42),
            "new_token_ids should contain the sampled token"
        );
    }

    #[test]
    fn test_pp_new_token_ids_multiple_decode_steps() {
        // Verify new_token_ids is correct across multiple decode steps.
        let cfg = pp_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 5));

        // Prefill.
        sched.schedule();
        sched.append_output_tokens("r1", &[100]);

        // Decode step 1: should see token 100.
        let out1 = sched.schedule();
        assert_eq!(out1.scheduled_cached_reqs.new_token_ids.len(), 1);
        assert!(out1.scheduled_cached_reqs.new_token_ids[0].contains(&100));

        // Simulate decode producing token 200.
        sched.append_output_tokens("r1", &[200]);

        // Decode step 2: should see token 200.
        let out2 = sched.schedule();
        assert_eq!(out2.scheduled_cached_reqs.new_token_ids.len(), 1);
        assert!(out2.scheduled_cached_reqs.new_token_ids[0].contains(&200));

        // Simulate decode producing token 300.
        sched.append_output_tokens("r1", &[300]);

        // Decode step 3: should see token 300.
        let out3 = sched.schedule();
        assert_eq!(out3.scheduled_cached_reqs.new_token_ids.len(), 1);
        assert!(out3.scheduled_cached_reqs.new_token_ids[0].contains(&300));
    }

    #[test]
    fn test_pp_new_token_ids_multiple_requests() {
        // Verify new_token_ids works with multiple concurrent requests.
        let cfg = pp_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 5));
        sched.add_request(make_request("r2", 5));

        // Prefill both.
        sched.schedule();
        sched.append_output_tokens("r1", &[10]);
        sched.append_output_tokens("r2", &[20]);

        // Decode: both should have their correct new_token_ids.
        let out = sched.schedule();
        assert_eq!(out.scheduled_cached_reqs.req_ids.len(), 2);
        assert_eq!(out.scheduled_cached_reqs.new_token_ids.len(), 2);

        // Find which index is which request.
        for (i, req_id) in out.scheduled_cached_reqs.req_ids.iter().enumerate() {
            let tokens = &out.scheduled_cached_reqs.new_token_ids[i];
            match req_id.as_str() {
                "r1" => assert!(tokens.contains(&10), "r1 should have token 10"),
                "r2" => assert!(tokens.contains(&20), "r2 should have token 20"),
                _ => panic!("unexpected req_id"),
            }
        }
    }

    #[test]
    fn test_pp_no_new_token_ids_without_pp() {
        // Without use_pp, new_token_ids should be empty.
        let cfg = test_scheduler_config(); // use_pp = false
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 5));
        sched.schedule();
        sched.append_output_tokens("r1", &[42]);

        let out = sched.schedule();
        assert!(
            out.scheduled_cached_reqs.new_token_ids.is_empty(),
            "Without PP, new_token_ids should be empty"
        );
    }

    #[test]
    fn test_pp_no_new_token_ids_with_async_scheduling() {
        // With use_pp + async_scheduling, new_token_ids should be empty
        // (tokens go via GPU broadcast instead).
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 4,
            enable_chunked_prefill: true,
            async_scheduling: Some(true),
            use_pp: true,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 5));
        sched.schedule();
        sched.append_output_tokens("r1", &[42]);

        let out = sched.schedule();
        assert!(
            out.scheduled_cached_reqs.new_token_ids.is_empty(),
            "PP + async scheduling should NOT populate new_token_ids (uses GPU broadcast)"
        );
    }

    #[test]
    fn test_decode_blocks_cached_after_preemption() {
        // After preemption, decode blocks should be recoverable from the cache.
        let block_size = 16;
        let mut tracker = SimpleBlockTracker::with_caching(20, block_size);

        // 32-token prompt + 32 decode tokens = 64 tokens = 4 full blocks.
        let prompt: Vec<u32> = (0..32).collect();
        let mut r1 = Request::new(
            "r1".into(),
            prompt,
            SamplingParams {
                max_tokens: Some(100),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );

        // Allocate prompt blocks (2 blocks for 32 tokens).
        let blocks = tracker.allocate_slots(&r1, 32, 0).unwrap();
        assert_eq!(blocks[0].len(), 2);
        r1.num_computed_tokens = 32;

        // Simulate 32 decode tokens, one at a time, allocating as needed.
        for i in 0..32u32 {
            r1.append_output_token_ids(&[100 + i]);
            r1.num_computed_tokens += 1;
            let _ = tracker.allocate_slots(&r1, 1, 0).unwrap();
        }

        // We have 4 full blocks + 1 partial (the last allocate_slots added a 5th
        // block for the partial tail). Only the 4 full blocks get hashed.
        let alloc = tracker.get_blocks("r1");
        assert_eq!(alloc[0].len(), 5);
        let original_block_ids: Vec<usize> = alloc[0][..4].to_vec();

        // Preempt: free blocks, reset computed tokens (but all_token_ids survives).
        tracker.free("r1");
        r1.num_computed_tokens = 0;
        r1.status = RequestStatus::Preempted;

        // 4 full blocks should be in the cache; the partial block goes to free list.
        assert_eq!(tracker.num_cached_blocks(), 4);

        // Re-admission: get_computed_blocks should find all 4 cached blocks.
        let (cached_tokens, cached_blocks) = tracker.get_computed_blocks(&r1);
        assert_eq!(cached_tokens, 64);
        assert_eq!(cached_blocks[0].len(), 4);
        assert_eq!(cached_blocks[0], original_block_ids);
    }
}
