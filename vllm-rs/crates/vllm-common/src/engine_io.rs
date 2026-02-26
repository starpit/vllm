// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Engine-core I/O types, ported from `vllm/v1/engine/__init__.py`.
//!
//! These types form the wire protocol between the engine front-end
//! (API server) and the engine core (scheduler + workers).

use serde::{Deserialize, Serialize};

use crate::sampling::SamplingParams;

// ---------------------------------------------------------------------------
// FinishReason
// ---------------------------------------------------------------------------

/// Reason a request finished.
///
/// Int-valued for compact serialization over IPC / ZMQ.
///
/// * `Stop`   -- a stop string or stop token was emitted.
/// * `Length` -- `max_tokens` was consumed or `max_model_len` was reached.
/// * `Abort`  -- aborted by the client.
/// * `Error`  -- a retryable request-level internal error (e.g. KV load
///   failure). Always surfaced as HTTP 500.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum FinishReason {
    Stop = 0,
    Length = 1,
    Abort = 2,
    Error = 3,
}

/// The canonical string representations, matching the OpenAI API.
const FINISH_REASON_STRINGS: [&str; 4] = ["stop", "length", "abort", "error"];

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(FINISH_REASON_STRINGS[*self as usize])
    }
}

// ---------------------------------------------------------------------------
// EngineCoreEventType / EngineCoreEvent
// ---------------------------------------------------------------------------

/// The type of a timestamped engine-core event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum EngineCoreEventType {
    Queued = 1,
    Scheduled = 2,
    Preempted = 3,
}

/// A timestamped engine-core event associated with a request.
///
/// The timestamp is monotonic (within the engine-core process) and is used
/// by the engine front-end to compute inter-event intervals. It should *not*
/// be compared with timestamps from other processes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EngineCoreEvent {
    /// What happened.
    pub event_type: EngineCoreEventType,
    /// Monotonic timestamp (seconds).
    pub timestamp: f64,
}

impl EngineCoreEvent {
    /// Convenience constructor.
    pub fn new(event_type: EngineCoreEventType, timestamp: f64) -> Self {
        Self {
            event_type,
            timestamp,
        }
    }
}

// ---------------------------------------------------------------------------
// StopReason
// ---------------------------------------------------------------------------

/// The reason generation stopped, which may be either a token ID or a string.
///
/// Mirrors the Python `stop_reason: int | str | None` field.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StopReason {
    /// Stopped because a specific token ID was emitted.
    Token(u32),
    /// Stopped because a specific string was emitted.
    String(String),
}

// ---------------------------------------------------------------------------
// EngineCoreRequest
// ---------------------------------------------------------------------------

/// A request submitted to the engine core by the front-end.
///
/// Mirrors the Python `EngineCoreRequest` msgspec struct.
/// Fields that depend on Python-only types (tensors, LoRA, multimodal
/// features, pooling params) are omitted in this initial Rust port and
/// will be added as those subsystems are brought up.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreRequest {
    /// Unique request identifier.
    pub request_id: String,

    /// The tokenized prompt. `None` when prompt embeddings are provided
    /// instead (multimodal / embedding models).
    pub prompt_token_ids: Option<Vec<u32>>,

    /// Sampling parameters (for generative models).
    pub sampling_params: Option<SamplingParams>,

    /// Wall-clock arrival time (seconds since epoch).
    pub arrival_time: f64,

    /// Index of the front-end client, used to route outputs back to the
    /// correct client when the front-end is scaled out.
    pub client_index: u32,

    /// Scheduling priority (lower = higher priority).
    pub priority: i32,

    /// Optional cache salt for prefix-cache isolation.
    pub cache_salt: Option<String>,

    /// In data-parallel mode, the rank this request should be sent to.
    pub data_parallel_rank: Option<u32>,
}

// ---------------------------------------------------------------------------
// EngineCoreOutput
// ---------------------------------------------------------------------------

/// Per-request output emitted by the engine core after each scheduler step.
///
/// Mirrors the Python `EngineCoreOutput` msgspec struct. Fields that depend
/// on tensor types (logprobs, pooling output, routed experts) are omitted
/// in this initial port.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreOutput {
    /// The request this output belongs to.
    pub request_id: String,

    /// Newly generated token IDs since the last output.
    pub new_token_ids: Vec<u32>,

    /// Set when the request has finished; `None` while still running.
    pub finish_reason: Option<FinishReason>,

    /// Why generation stopped (token ID or string), if applicable.
    pub stop_reason: Option<StopReason>,

    /// Number of prompt tokens served from prefix cache (local + external).
    pub num_cached_tokens: u32,

    /// Events (queued, scheduled, preempted) recorded during this step.
    pub events: Option<Vec<EngineCoreEvent>>,
}

impl EngineCoreOutput {
    /// Whether the request has finished.
    pub fn finished(&self) -> bool {
        self.finish_reason.is_some()
    }
}

// ---------------------------------------------------------------------------
// EngineCoreOutputs
// ---------------------------------------------------------------------------

/// A batch of outputs from one engine-core step, possibly spanning multiple
/// requests.
///
/// Mirrors the Python `EngineCoreOutputs` msgspec struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreOutputs {
    /// Index of the engine that produced these outputs (for multi-engine /
    /// data-parallel setups).
    pub engine_index: u32,

    /// Per-request outputs.
    pub outputs: Vec<EngineCoreOutput>,

    /// Monotonic timestamp of when the step completed.
    pub timestamp: f64,
}

impl Default for EngineCoreOutputs {
    fn default() -> Self {
        Self {
            engine_index: 0,
            outputs: Vec::new(),
            timestamp: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- FinishReason tests --

    #[test]
    fn test_finish_reason_display() {
        assert_eq!(format!("{}", FinishReason::Stop), "stop");
        assert_eq!(format!("{}", FinishReason::Length), "length");
        assert_eq!(format!("{}", FinishReason::Abort), "abort");
        assert_eq!(format!("{}", FinishReason::Error), "error");
    }

    #[test]
    fn test_finish_reason_repr() {
        assert_eq!(FinishReason::Stop as u8, 0);
        assert_eq!(FinishReason::Length as u8, 1);
        assert_eq!(FinishReason::Abort as u8, 2);
        assert_eq!(FinishReason::Error as u8, 3);
    }

    #[test]
    fn test_finish_reason_serde_roundtrip() {
        for reason in [
            FinishReason::Stop,
            FinishReason::Length,
            FinishReason::Abort,
            FinishReason::Error,
        ] {
            let json = serde_json::to_string(&reason).unwrap();
            let back: FinishReason = serde_json::from_str(&json).unwrap();
            assert_eq!(reason, back);
        }
    }

    // -- EngineCoreEventType / EngineCoreEvent tests --

    #[test]
    fn test_event_type_repr() {
        assert_eq!(EngineCoreEventType::Queued as u8, 1);
        assert_eq!(EngineCoreEventType::Scheduled as u8, 2);
        assert_eq!(EngineCoreEventType::Preempted as u8, 3);
    }

    #[test]
    fn test_event_new() {
        let evt = EngineCoreEvent::new(EngineCoreEventType::Scheduled, 42.5);
        assert_eq!(evt.event_type, EngineCoreEventType::Scheduled);
        assert_eq!(evt.timestamp, 42.5);
    }

    #[test]
    fn test_event_serde_roundtrip() {
        let evt = EngineCoreEvent::new(EngineCoreEventType::Preempted, 100.0);
        let json = serde_json::to_string(&evt).unwrap();
        let evt2: EngineCoreEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(evt, evt2);
    }

    // -- StopReason tests --

    #[test]
    fn test_stop_reason_token() {
        let sr = StopReason::Token(50256);
        assert_eq!(sr, StopReason::Token(50256));
    }

    #[test]
    fn test_stop_reason_string() {
        let sr = StopReason::String("</s>".into());
        assert_eq!(sr, StopReason::String("</s>".into()));
    }

    #[test]
    fn test_stop_reason_serde_roundtrip() {
        let cases = vec![
            StopReason::Token(123),
            StopReason::String("END".into()),
        ];
        for sr in cases {
            let json = serde_json::to_string(&sr).unwrap();
            let sr2: StopReason = serde_json::from_str(&json).unwrap();
            assert_eq!(sr, sr2);
        }
    }

    // -- EngineCoreRequest tests --

    #[test]
    fn test_engine_core_request_minimal() {
        let req = EngineCoreRequest {
            request_id: "req-1".into(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            sampling_params: Some(SamplingParams::default()),
            arrival_time: 1000.0,
            client_index: 0,
            priority: 0,
            cache_salt: None,
            data_parallel_rank: None,
        };
        assert_eq!(req.request_id, "req-1");
        assert_eq!(req.prompt_token_ids.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_engine_core_request_serde_roundtrip() {
        let req = EngineCoreRequest {
            request_id: "req-2".into(),
            prompt_token_ids: Some(vec![10, 20]),
            sampling_params: Some(SamplingParams {
                temperature: 0.5,
                max_tokens: Some(128),
                ..Default::default()
            }),
            arrival_time: 999.0,
            client_index: 2,
            priority: 5,
            cache_salt: Some("salt-abc".into()),
            data_parallel_rank: Some(1),
        };
        let json = serde_json::to_string(&req).unwrap();
        let req2: EngineCoreRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req2.request_id, "req-2");
        assert_eq!(req2.client_index, 2);
        assert_eq!(req2.priority, 5);
        assert_eq!(req2.cache_salt.as_deref(), Some("salt-abc"));
        assert_eq!(req2.data_parallel_rank, Some(1));
    }

    // -- EngineCoreOutput tests --

    #[test]
    fn test_engine_core_output_not_finished() {
        let out = EngineCoreOutput {
            request_id: "r1".into(),
            new_token_ids: vec![42, 43],
            finish_reason: None,
            stop_reason: None,
            num_cached_tokens: 10,
            events: None,
        };
        assert!(!out.finished());
    }

    #[test]
    fn test_engine_core_output_finished() {
        let out = EngineCoreOutput {
            request_id: "r1".into(),
            new_token_ids: vec![42],
            finish_reason: Some(FinishReason::Stop),
            stop_reason: Some(StopReason::Token(50256)),
            num_cached_tokens: 0,
            events: Some(vec![
                EngineCoreEvent::new(EngineCoreEventType::Queued, 1.0),
                EngineCoreEvent::new(EngineCoreEventType::Scheduled, 2.0),
            ]),
        };
        assert!(out.finished());
        assert_eq!(out.finish_reason, Some(FinishReason::Stop));
        assert_eq!(out.events.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_engine_core_output_serde_roundtrip() {
        let out = EngineCoreOutput {
            request_id: "r1".into(),
            new_token_ids: vec![1, 2, 3],
            finish_reason: Some(FinishReason::Length),
            stop_reason: None,
            num_cached_tokens: 5,
            events: None,
        };
        let json = serde_json::to_string(&out).unwrap();
        let out2: EngineCoreOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(out2.request_id, "r1");
        assert_eq!(out2.new_token_ids, vec![1, 2, 3]);
        assert_eq!(out2.finish_reason, Some(FinishReason::Length));
        assert_eq!(out2.num_cached_tokens, 5);
    }

    // -- EngineCoreOutputs tests --

    #[test]
    fn test_engine_core_outputs_default() {
        let outs = EngineCoreOutputs::default();
        assert_eq!(outs.engine_index, 0);
        assert!(outs.outputs.is_empty());
        assert_eq!(outs.timestamp, 0.0);
    }

    #[test]
    fn test_engine_core_outputs_with_data() {
        let outs = EngineCoreOutputs {
            engine_index: 2,
            outputs: vec![
                EngineCoreOutput {
                    request_id: "a".into(),
                    new_token_ids: vec![10],
                    finish_reason: None,
                    stop_reason: None,
                    num_cached_tokens: 0,
                    events: None,
                },
                EngineCoreOutput {
                    request_id: "b".into(),
                    new_token_ids: vec![20],
                    finish_reason: Some(FinishReason::Stop),
                    stop_reason: Some(StopReason::String("END".into())),
                    num_cached_tokens: 3,
                    events: None,
                },
            ],
            timestamp: 1234.5,
        };
        assert_eq!(outs.outputs.len(), 2);
        assert_eq!(outs.engine_index, 2);
        assert_eq!(outs.timestamp, 1234.5);
    }

    #[test]
    fn test_engine_core_outputs_serde_roundtrip() {
        let outs = EngineCoreOutputs {
            engine_index: 1,
            outputs: vec![EngineCoreOutput {
                request_id: "r1".into(),
                new_token_ids: vec![7, 8, 9],
                finish_reason: None,
                stop_reason: None,
                num_cached_tokens: 0,
                events: Some(vec![EngineCoreEvent::new(
                    EngineCoreEventType::Queued,
                    0.5,
                )]),
            }],
            timestamp: 42.0,
        };
        let json = serde_json::to_string(&outs).unwrap();
        let outs2: EngineCoreOutputs = serde_json::from_str(&json).unwrap();
        assert_eq!(outs2.engine_index, 1);
        assert_eq!(outs2.outputs.len(), 1);
        assert_eq!(outs2.outputs[0].new_token_ids, vec![7, 8, 9]);
        assert_eq!(outs2.timestamp, 42.0);
    }
}
