// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Request and request-status types, ported from `vllm/v1/request.py`.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::engine_io::FinishReason;
use crate::multimodal::MultimodalData;
use crate::sampling::SamplingParams;

// ---------------------------------------------------------------------------
// BlockKind — span annotation for block hashing
// ---------------------------------------------------------------------------

/// Annotation for a block's hashing behavior in the span-aware cache.
///
/// The `/v1/query/execute` endpoint produces a sparse map of block indices to
/// `BlockKind` values. The block hasher reads these annotations to decide
/// parent-hash chaining.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockKind {
    /// Position-independent block: parent hash is reset to `NONE_HASH`,
    /// making this block cacheable regardless of where it appears in the
    /// sequence.
    Relocatable,
    /// Prefix-dependent block: all preceding tokens are folded into the hash,
    /// forcing recomputation when any prior context differs.
    Prefixed,
}

/// Sparse map of block index to [`BlockKind`] for span-aware block hashing.
pub type BlockAnnotations = BTreeMap<usize, BlockKind>;

/// Compute per-block RoPE rotation flags for a single block.
///
/// Returns `(is_relocatable, is_unrotated)`:
/// - `is_relocatable`: true if the block is annotated as [`BlockKind::Relocatable`].
///   Post-attention un-rotation will remove RoPE from this block.
/// - `is_unrotated`: true if the block was written in a prior step (its K is
///   currently stored without RoPE). Pre-attention rotation will apply RoPE.
///
/// `block_idx`: logical block index in the sequence.
/// `block_size`: tokens per block.
/// `seq_len`: total sequence length.
/// `tokens_before`: number of tokens computed before this step.
pub fn compute_block_flags(
    annotations: &BlockAnnotations,
    block_idx: usize,
    block_size: usize,
    seq_len: usize,
    tokens_before: usize,
) -> (bool, bool) {
    let is_relocatable = annotations.get(&block_idx) == Some(&BlockKind::Relocatable);
    let block_end_pos = ((block_idx + 1) * block_size).min(seq_len);
    let was_previously_written = block_end_pos <= tokens_before;
    let is_unrotated = is_relocatable && was_previously_written;
    (is_relocatable, is_unrotated)
}

// ---------------------------------------------------------------------------
// RequestStatus
// ---------------------------------------------------------------------------

/// Status of a request as it moves through the engine pipeline.
///
/// Values after `Preempted` are considered "finished".
/// This mirrors the Python `RequestStatus(IntEnum)` from `vllm/v1/request.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RequestStatus {
    Waiting = 1,
    WaitingForFsm = 2,
    WaitingForRemoteKvs = 3,
    WaitingForStreamingReq = 4,
    Running = 5,
    Preempted = 6,
    // --- everything below is "finished" ---
    FinishedStopped = 7,
    FinishedLengthCapped = 8,
    FinishedAborted = 9,
    FinishedIgnored = 10,
    FinishedError = 11,
}

impl RequestStatus {
    /// Returns `true` if the status represents a terminal (finished) state.
    ///
    /// A request is finished if its discriminant is greater than `Preempted`.
    pub fn is_finished(self) -> bool {
        (self as u8) > (Self::Preempted as u8)
    }

    /// Map a finished status to its corresponding [`FinishReason`].
    ///
    /// Returns `None` for non-finished statuses (with the exception of
    /// `WaitingForStreamingReq`, which maps to `Stop` -- matching the
    /// Python `_FINISHED_REASON_MAP`).
    pub fn get_finished_reason(self) -> Option<FinishReason> {
        match self {
            Self::FinishedStopped => Some(FinishReason::Stop),
            Self::FinishedLengthCapped => Some(FinishReason::Length),
            Self::FinishedAborted => Some(FinishReason::Abort),
            // Ignored requests hit the model length cap, so the reason is Length.
            Self::FinishedIgnored => Some(FinishReason::Length),
            Self::FinishedError => Some(FinishReason::Error),
            Self::WaitingForStreamingReq => Some(FinishReason::Stop),
            _ => None,
        }
    }
}

impl std::fmt::Display for RequestStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Match Python's `__str__` which returns the variant name.
        // Use write_str with a match to avoid Debug formatter overhead.
        f.write_str(match self {
            Self::Waiting => "Waiting",
            Self::WaitingForFsm => "WaitingForFsm",
            Self::WaitingForRemoteKvs => "WaitingForRemoteKvs",
            Self::WaitingForStreamingReq => "WaitingForStreamingReq",
            Self::Running => "Running",
            Self::Preempted => "Preempted",
            Self::FinishedStopped => "FinishedStopped",
            Self::FinishedLengthCapped => "FinishedLengthCapped",
            Self::FinishedAborted => "FinishedAborted",
            Self::FinishedIgnored => "FinishedIgnored",
            Self::FinishedError => "FinishedError",
        })
    }
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// Core scheduler-level representation of an in-flight request.
///
/// Ported from the Python `Request` class in `vllm/v1/request.py`.
/// Heavy Python-only fields (tensors, block-hashers, structured-output FSM
/// state, LoRA requests, multimodal features) are omitted here; they will be
/// added in later phases or handled via trait objects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    /// Unique identifier for this request.
    pub request_id: String,

    /// Index of the front-end client that owns this request.
    pub client_index: u32,

    /// Scheduling priority (lower = higher priority).
    pub priority: i32,

    /// Sampling parameters for generation.
    pub sampling_params: SamplingParams,

    /// Wall-clock arrival time (seconds since epoch).
    pub arrival_time: f64,

    /// Current lifecycle status.
    pub status: RequestStatus,

    /// Maximum number of tokens to generate for this request.
    pub max_tokens: u32,

    /// The tokenized prompt.
    pub prompt_token_ids: Vec<u32>,

    /// Tokens generated so far (output only, excludes prompt).
    pub output_token_ids: Vec<u32>,

    /// Concatenation of `prompt_token_ids` and `output_token_ids`.
    pub all_token_ids: Vec<u32>,

    /// Speculative-decoding draft token IDs (if any).
    pub spec_token_ids: Vec<u32>,

    /// Number of tokens that have been computed (prompt + output KV cached).
    pub num_computed_tokens: u32,

    /// Length of the prompt in tokens.
    pub num_prompt_tokens: u32,

    /// Optional per-request cache salt for prefix-cache isolation.
    pub cache_salt: Option<String>,

    /// Number of prompt tokens served from cache (local + external).
    /// -1 means "not yet known".
    pub num_cached_tokens: i32,

    /// `true` while the request is being prefilled in chunks (not yet
    /// finished prefill).
    pub is_prefill_chunk: bool,

    /// How many times the scheduler has preempted this request.
    pub num_preemptions: u32,

    /// Number of tokens computed remotely (P/D disaggregated serving).
    pub num_external_computed_tokens: u32,

    /// Number of placeholder output tokens reserved for async scheduling.
    pub num_output_placeholders: u32,

    /// Whether this is a pooling (embedding) request rather than generation.
    /// Pooling requests are finished after one forward pass (no decode loop).
    pub is_pooling: bool,

    /// Multimodal data (images) for vision-language models.
    /// Set once at request creation, consumed during the first prefill step.
    #[serde(skip)]
    pub mm_data: Option<MultimodalData>,

    /// Sparse map of block index → [`BlockKind`] for span-aware block hashing.
    /// Only blocks with non-default hashing behavior are present.
    /// `None` means all blocks use normal parent-chained hashing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_annotations: Option<BlockAnnotations>,

    /// 🦭 When true, pad and hash the final partial block on completion so
    /// future requests can get a cache hit on this request's full output.
    #[serde(default)]
    pub seal: bool,

    /// When true, deprioritize this request's cached blocks for eviction
    /// after generation completes. Used for one-shot consumers like inner
    /// generates in a nested generation pattern.
    #[serde(default)]
    pub volatile: bool,
}

impl Request {
    /// Create a new `Request` with the given core fields.
    ///
    /// Initializes all mutable counters to their default state.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_id: String,
        prompt_token_ids: Vec<u32>,
        sampling_params: SamplingParams,
        arrival_time: f64,
        client_index: u32,
        priority: i32,
        cache_salt: Option<String>,
    ) -> Self {
        let num_prompt_tokens = prompt_token_ids.len() as u32;
        let max_tokens = sampling_params.max_tokens.unwrap_or(u32::MAX);

        Self {
            request_id,
            client_index,
            priority,
            sampling_params,
            arrival_time,
            status: RequestStatus::Waiting,
            max_tokens,
            all_token_ids: prompt_token_ids.clone(),
            prompt_token_ids,
            output_token_ids: Vec::new(),
            spec_token_ids: Vec::new(),
            num_computed_tokens: 0,
            num_prompt_tokens,
            cache_salt,
            num_cached_tokens: -1,
            is_prefill_chunk: false,
            num_preemptions: 0,
            num_external_computed_tokens: 0,
            num_output_placeholders: 0,
            is_pooling: false,
            mm_data: None,
            block_annotations: None,
            seal: false,
            volatile: false,
        }
    }

    /// Append one or more output token IDs, updating both `output_token_ids`
    /// and `all_token_ids`.
    pub fn append_output_token_ids(&mut self, token_ids: &[u32]) {
        self.output_token_ids.extend_from_slice(token_ids);
        self.all_token_ids.extend_from_slice(token_ids);
    }

    /// The total number of tokens (prompt + output) currently tracked.
    pub fn num_tokens(&self) -> usize {
        self.all_token_ids.len()
    }

    /// Total tokens including speculative draft tokens.
    pub fn num_tokens_with_spec(&self) -> usize {
        self.all_token_ids.len() + self.spec_token_ids.len()
    }

    /// The number of output tokens generated so far.
    pub fn num_output_tokens(&self) -> usize {
        self.output_token_ids.len()
    }

    /// Whether this request has reached a terminal state.
    pub fn is_finished(&self) -> bool {
        self.status.is_finished()
    }

    /// The finish reason, if the request is in a terminal state.
    pub fn get_finished_reason(&self) -> Option<FinishReason> {
        self.status.get_finished_reason()
    }
}

// ---------------------------------------------------------------------------
// Ordering (for priority scheduling)
// ---------------------------------------------------------------------------

impl PartialEq for Request {
    fn eq(&self, other: &Self) -> bool {
        self.request_id == other.request_id
    }
}

impl Eq for Request {}

impl PartialOrd for Request {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Ordering used by the scheduler's priority queue.
///
/// Lower priority value => higher scheduling priority.
/// Ties are broken by arrival time (earlier first), then by request ID
/// (lexicographic).
impl Ord for Request {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| {
                self.arrival_time
                    .partial_cmp(&other.arrival_time)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| self.request_id.cmp(&other.request_id))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(id: &str, priority: i32, arrival: f64) -> Request {
        Request::new(
            id.into(),
            vec![1, 2, 3],
            SamplingParams::default(),
            arrival,
            0,
            priority,
            None,
        )
    }

    // -- RequestStatus tests --

    #[test]
    fn test_status_is_finished() {
        assert!(!RequestStatus::Waiting.is_finished());
        assert!(!RequestStatus::WaitingForFsm.is_finished());
        assert!(!RequestStatus::WaitingForRemoteKvs.is_finished());
        assert!(!RequestStatus::WaitingForStreamingReq.is_finished());
        assert!(!RequestStatus::Running.is_finished());
        assert!(!RequestStatus::Preempted.is_finished());

        assert!(RequestStatus::FinishedStopped.is_finished());
        assert!(RequestStatus::FinishedLengthCapped.is_finished());
        assert!(RequestStatus::FinishedAborted.is_finished());
        assert!(RequestStatus::FinishedIgnored.is_finished());
        assert!(RequestStatus::FinishedError.is_finished());
    }

    #[test]
    fn test_status_finish_reasons() {
        assert_eq!(
            RequestStatus::FinishedStopped.get_finished_reason(),
            Some(FinishReason::Stop)
        );
        assert_eq!(
            RequestStatus::FinishedLengthCapped.get_finished_reason(),
            Some(FinishReason::Length)
        );
        assert_eq!(
            RequestStatus::FinishedAborted.get_finished_reason(),
            Some(FinishReason::Abort)
        );
        assert_eq!(
            RequestStatus::FinishedIgnored.get_finished_reason(),
            Some(FinishReason::Length)
        );
        assert_eq!(
            RequestStatus::FinishedError.get_finished_reason(),
            Some(FinishReason::Error)
        );
        assert_eq!(
            RequestStatus::WaitingForStreamingReq.get_finished_reason(),
            Some(FinishReason::Stop)
        );
    }

    #[test]
    fn test_status_non_finished_reason_is_none() {
        assert_eq!(RequestStatus::Waiting.get_finished_reason(), None);
        assert_eq!(RequestStatus::Running.get_finished_reason(), None);
        assert_eq!(RequestStatus::Preempted.get_finished_reason(), None);
    }

    #[test]
    fn test_status_display() {
        assert_eq!(format!("{}", RequestStatus::Waiting), "Waiting");
        assert_eq!(format!("{}", RequestStatus::Running), "Running");
        assert_eq!(
            format!("{}", RequestStatus::FinishedStopped),
            "FinishedStopped"
        );
    }

    #[test]
    fn test_status_repr_values() {
        assert_eq!(RequestStatus::Waiting as u8, 1);
        assert_eq!(RequestStatus::WaitingForFsm as u8, 2);
        assert_eq!(RequestStatus::WaitingForRemoteKvs as u8, 3);
        assert_eq!(RequestStatus::WaitingForStreamingReq as u8, 4);
        assert_eq!(RequestStatus::Running as u8, 5);
        assert_eq!(RequestStatus::Preempted as u8, 6);
        assert_eq!(RequestStatus::FinishedStopped as u8, 7);
        assert_eq!(RequestStatus::FinishedLengthCapped as u8, 8);
        assert_eq!(RequestStatus::FinishedAborted as u8, 9);
        assert_eq!(RequestStatus::FinishedIgnored as u8, 10);
        assert_eq!(RequestStatus::FinishedError as u8, 11);
    }

    // -- Request construction tests --

    #[test]
    fn test_new_request_defaults() {
        let req = make_request("r1", 0, 1.0);
        assert_eq!(req.request_id, "r1");
        assert_eq!(req.status, RequestStatus::Waiting);
        assert_eq!(req.num_prompt_tokens, 3);
        assert_eq!(req.num_computed_tokens, 0);
        assert_eq!(req.num_cached_tokens, -1);
        assert!(!req.is_prefill_chunk);
        assert_eq!(req.num_preemptions, 0);
        assert!(req.output_token_ids.is_empty());
        assert_eq!(req.all_token_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_append_output_tokens() {
        let mut req = make_request("r1", 0, 1.0);
        req.append_output_token_ids(&[10, 11]);
        assert_eq!(req.output_token_ids, vec![10, 11]);
        assert_eq!(req.all_token_ids, vec![1, 2, 3, 10, 11]);
        assert_eq!(req.num_tokens(), 5);
        assert_eq!(req.num_output_tokens(), 2);
    }

    #[test]
    fn test_num_tokens_with_spec() {
        let mut req = make_request("r1", 0, 1.0);
        req.spec_token_ids = vec![99, 100, 101];
        assert_eq!(req.num_tokens_with_spec(), 6); // 3 prompt + 3 spec
    }

    #[test]
    fn test_is_finished() {
        let mut req = make_request("r1", 0, 1.0);
        assert!(!req.is_finished());

        req.status = RequestStatus::FinishedStopped;
        assert!(req.is_finished());
        assert_eq!(req.get_finished_reason(), Some(FinishReason::Stop));
    }

    // -- Ordering tests --

    #[test]
    fn test_ordering_by_priority() {
        let r_high = make_request("a", 0, 1.0); // higher priority (lower value)
        let r_low = make_request("b", 10, 1.0);
        assert!(r_high < r_low);
    }

    #[test]
    fn test_ordering_by_arrival_time() {
        let r_early = make_request("a", 0, 1.0);
        let r_late = make_request("b", 0, 2.0);
        assert!(r_early < r_late);
    }

    #[test]
    fn test_ordering_by_request_id() {
        let r_a = make_request("aaa", 0, 1.0);
        let r_b = make_request("bbb", 0, 1.0);
        assert!(r_a < r_b);
    }

    #[test]
    fn test_equality_by_request_id() {
        let r1 = make_request("r1", 0, 1.0);
        let r2 = make_request("r1", 5, 99.0);
        assert_eq!(r1, r2); // equality is by request_id only
    }

    #[test]
    fn test_sort_requests() {
        let r1 = make_request("c", 0, 3.0);
        let r2 = make_request("a", 0, 1.0);
        let r3 = make_request("b", 0, 2.0);
        let r4 = make_request("d", -1, 10.0); // highest priority

        let mut reqs = [r1, r2, r3, r4];
        reqs.sort();

        let ids: Vec<&str> = reqs.iter().map(|r| r.request_id.as_str()).collect();
        // d has priority -1 (first), then a,b,c ordered by arrival time
        assert_eq!(ids, vec!["d", "a", "b", "c"]);
    }

    // -- Serde tests --

    #[test]
    fn test_request_serde_roundtrip() {
        let req = make_request("r1", 0, 1.0);
        let json = serde_json::to_string(&req).unwrap();
        let req2: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(req2.request_id, "r1");
        assert_eq!(req2.num_prompt_tokens, 3);
        assert_eq!(req2.status, RequestStatus::Waiting);
    }

    #[test]
    fn test_request_status_serde_roundtrip() {
        let status = RequestStatus::FinishedAborted;
        let json = serde_json::to_string(&status).unwrap();
        let status2: RequestStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status, status2);
    }

    // -----------------------------------------------------------------------
    // compute_block_flags tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_block_flags_unannotated_block() {
        // No annotations → never flagged, regardless of position or timing.
        let ann = BTreeMap::new();
        let (is_reloc, is_unrot) = compute_block_flags(&ann, 0, 16, 64, 32);
        assert!(!is_reloc);
        assert!(!is_unrot);
    }

    #[test]
    fn test_block_flags_relocatable_freshly_written() {
        // Relocatable block written THIS step: is_relocatable=true,
        // is_unrotated=false (K has RoPE from QKV projection).
        let mut ann = BTreeMap::new();
        ann.insert(0, BlockKind::Relocatable);
        // block 0, block_size=4, seq_len=8, tokens_before=0 (all new)
        let (is_reloc, is_unrot) = compute_block_flags(&ann, 0, 4, 8, 0);
        assert!(is_reloc);
        assert!(!is_unrot); // freshly written → still has RoPE
    }

    #[test]
    fn test_block_flags_relocatable_previously_cached() {
        // Relocatable block from a prior step: is_relocatable=true,
        // is_unrotated=true (post-attention un-rotated it last step).
        let mut ann = BTreeMap::new();
        ann.insert(0, BlockKind::Relocatable);
        // block 0, block_size=4, seq_len=8, tokens_before=4 (block 0 fully cached)
        let (is_reloc, is_unrot) = compute_block_flags(&ann, 0, 4, 8, 4);
        assert!(is_reloc);
        assert!(is_unrot); // prior step → K is unrotated
    }

    #[test]
    fn test_block_flags_prefixed_never_flagged() {
        // Prefixed blocks are NOT Relocatable — they should never be
        // flagged for rotation, regardless of cache state.
        let mut ann = BTreeMap::new();
        ann.insert(2, BlockKind::Prefixed);
        let (is_reloc, is_unrot) = compute_block_flags(&ann, 2, 4, 16, 12);
        assert!(!is_reloc);
        assert!(!is_unrot);
    }

    #[test]
    fn test_block_flags_mixed_annotations() {
        // Sequence: [Relocatable, Relocatable, Prefixed]
        // All previously cached (tokens_before covers all).
        let mut ann = BTreeMap::new();
        ann.insert(0, BlockKind::Relocatable);
        ann.insert(1, BlockKind::Relocatable);
        ann.insert(2, BlockKind::Prefixed);

        let block_size = 4;
        let seq_len = 12;
        let tokens_before = 12; // all cached

        // Block 0: Relocatable, cached → unrotated
        let (r, u) = compute_block_flags(&ann, 0, block_size, seq_len, tokens_before);
        assert!(r);
        assert!(u);

        // Block 1: Relocatable, cached → unrotated
        let (r, u) = compute_block_flags(&ann, 1, block_size, seq_len, tokens_before);
        assert!(r);
        assert!(u);

        // Block 2: Prefixed → not flagged
        let (r, u) = compute_block_flags(&ann, 2, block_size, seq_len, tokens_before);
        assert!(!r);
        assert!(!u);
    }

    #[test]
    fn test_block_flags_partially_written_block() {
        // Block is being written this step (block_end > tokens_before).
        let mut ann = BTreeMap::new();
        ann.insert(1, BlockKind::Relocatable);
        // block 1 spans positions 4..8, tokens_before=6 → partially written
        let (is_reloc, is_unrot) = compute_block_flags(&ann, 1, 4, 8, 6);
        assert!(is_reloc);
        assert!(!is_unrot); // not fully cached → freshly written
    }
}
