// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Async engine interface for the serving layer.
//!
//! Provides the bridge between the HTTP server and the engine core client.
//! Handles request lifecycle: validate → tokenize → submit → poll → detokenize → respond.
//!
//! Port of: `vllm/v1/engine/async_llm.py` (subset)

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rayon::prelude::*;
use tokio::sync::{Mutex, Notify, mpsc, oneshot};
use tracing::{debug, error, info};
use uuid::Uuid;
#[cfg(feature = "multimodal")]
use vllm_common::multimodal::{ImageData, MultimodalData};
use vllm_common::sampling::GuidedGrammar;
use vllm_common::{EngineCoreOutput, EngineCoreRequest, FinishReason, SamplingParams, StopReason};
use vllm_engine::core_client::EngineCoreClient;
use vllm_engine::executor::{Executor, ModelRunnerOutput};

#[cfg(feature = "chat-template")]
use crate::chat_template::ChatTemplate;
use crate::detokenizer::IncrementalDetokenizer;
use crate::error::{ServeError, ServeResult};
use crate::protocol;
use crate::tokenizer::Tokenizer;
use crate::tool_parser::{
    DeltaToolCall, StreamingToolParserState, ToolCallParser, ToolParserDelta,
};

// ---------------------------------------------------------------------------
// Parallel detokenization type aliases
// ---------------------------------------------------------------------------

/// Phase 1 work item: (new_token_ids, stop_terminated, finish_reason, detokenizer).
type DetokWork = (Vec<u32>, bool, Option<FinishReason>, IncrementalDetokenizer);
/// Phase 2 result: (detokenizer, detokenizer_stop_string, delta_text).
type DetokResult = (IncrementalDetokenizer, Option<String>, Option<String>);

// ---------------------------------------------------------------------------
// ControlMessage — commands sent to the step loop from HTTP handlers.
// ---------------------------------------------------------------------------

enum ControlMessage {
    /// Reset the prefix cache. Returns `true` if successful.
    ResetPrefixCache(oneshot::Sender<bool>),
}

/// Bundled arguments for `spawn_step_loop_inner` / `spawn_step_loop_async`.
struct StepLoopArgs {
    client: Box<dyn EngineCoreClient + Send>,
    request_rx: mpsc::UnboundedReceiver<EngineCoreRequest>,
    embed_rx: mpsc::UnboundedReceiver<EmbedRequest>,
    control_rx: mpsc::UnboundedReceiver<ControlMessage>,
    requests: Arc<Mutex<HashMap<String, RequestState>>>,
    notify: Arc<Notify>,
    alive: Arc<AtomicBool>,
    no_progress_timeout: Duration,
}

// ---------------------------------------------------------------------------
// RequestState
// ---------------------------------------------------------------------------

/// Tracks the state of an in-flight request.
struct RequestState {
    /// Accumulated generated token IDs.
    generated_token_ids: Vec<u32>,

    /// Number of prompt tokens.
    num_prompt_tokens: u32,

    /// Number of cached prompt tokens.
    num_cached_tokens: u32,

    /// Finish reason, if completed.
    finish_reason: Option<FinishReason>,

    /// Error message if the request was aborted due to an engine error.
    error: Option<String>,

    /// Stop reason, if applicable.
    stop_reason: Option<StopReason>,

    /// Channel to send streaming output deltas.
    stream_tx: Option<mpsc::UnboundedSender<StreamDelta>>,

    /// Incremental detokenizer for this request (None if no tokenizer).
    detokenizer: Option<IncrementalDetokenizer>,

    /// Position in response choices (for n>1 / multi-prompt support).
    choice_index: u32,

    /// When this request was submitted (for TTFT calculation).
    submit_time: Instant,
    /// When the first output token was received (None until first token).
    first_token_time: Option<Instant>,
    /// When the last output token was received (for ITL calculation).
    last_token_time: Option<Instant>,
    /// Number of inter-token intervals observed (for computing avg ITL).
    itl_count: u32,
    /// Sum of inter-token latencies in seconds (for computing avg ITL).
    itl_sum: f64,

    /// Accumulated per-token log-probabilities (if requested).
    logprobs: Vec<vllm_common::LogprobsOutput>,

    /// Per-prompt-token log-probabilities (if requested). Set once during prefill.
    /// Position 0 is None (no prior context); positions 1..n are Some.
    prompt_logprobs: Option<Vec<Option<vllm_common::LogprobsOutput>>>,

    /// Streaming tool parser state (if tool parsing is active for this request).
    tool_parser_state: Option<Box<dyn StreamingToolParserState + Send>>,

    /// Accumulated generated text so far (for streaming tool parsing).
    accumulated_text: String,

    /// Whether any tool call deltas have been emitted (for setting finish_reason).
    tool_calls_emitted: bool,

    /// Forced function name from `tool_choice: {function: {name}}` (for filtering).
    forced_function_name: Option<String>,

    /// Pooling output (embedding vector), set when the engine is in pooling mode.
    pooler_output: Option<Vec<f32>>,
}

/// A delta sent to a streaming response.
#[derive(Debug, Clone)]
pub struct StreamDelta {
    /// Choice index in the response (for n>1 support).
    pub index: u32,
    /// New token IDs generated in this step.
    pub new_token_ids: Vec<u32>,
    /// Decoded text for this step (None if no tokenizer).
    pub text: Option<String>,
    /// Finish reason, if the request completed in this step.
    pub finish_reason: Option<FinishReason>,
    /// Stop reason, if applicable.
    pub stop_reason: Option<StopReason>,
    /// Per-token log-probabilities for this step (if requested).
    pub logprobs: Option<Vec<vllm_common::LogprobsOutput>>,
    /// Tool call deltas for streaming tool parsing.
    pub tool_call_deltas: Option<Vec<DeltaToolCall>>,
}

// ---------------------------------------------------------------------------
// AsyncEngine
// ---------------------------------------------------------------------------

/// An embedding request sent to the step loop.
struct EmbedRequest {
    token_id_seqs: Vec<Vec<u32>>,
    reply: tokio::sync::oneshot::Sender<ServeResult<Vec<Vec<f32>>>>,
}

/// Maximum time the step loop can run with pending requests but no output
/// tokens before aborting all requests. Acts as a safety net for scheduler
/// bugs where requests get stuck without making forward progress.
const DEFAULT_NO_PROGRESS_TIMEOUT: Duration = Duration::from_secs(60);

/// An async wrapper around the engine core client, providing request lifecycle
/// management for the HTTP serving layer.
///
/// Architecture: the engine client lives exclusively in the background step
/// loop (spawned by `spawn_step_loop`). HTTP handlers submit requests via an
/// `mpsc` channel and wait on a `Notify` for results. This eliminates mutex
/// contention between the slow model forward pass and fast request submission.
///
/// This is a simplified version of Python's `AsyncLLM`. It manages:
/// - Request submission and tracking
/// - Tokenization and detokenization (when a tokenizer is provided)
/// - Polling the engine for outputs
/// - Routing outputs to the correct response (streaming or non-streaming)
pub struct AsyncEngine {
    /// In-flight request states. Only held briefly for HashMap reads/writes,
    /// never during model execution.
    requests: Arc<Mutex<HashMap<String, RequestState>>>,
    /// Channel to send new `EngineCoreRequest`s to the step loop.
    request_tx: mpsc::UnboundedSender<EngineCoreRequest>,
    /// Channel to send embedding requests to the step loop.
    embed_tx: mpsc::UnboundedSender<EmbedRequest>,
    /// Channel to send control messages (e.g. reset prefix cache) to the step loop.
    control_tx: mpsc::UnboundedSender<ControlMessage>,
    /// The engine client + channel receivers, held until `spawn_step_loop`
    /// moves them into the background task. `None` after the loop starts.
    #[allow(clippy::type_complexity)]
    pending_loop: std::sync::Mutex<
        Option<(
            Box<dyn EngineCoreClient + Send>,
            mpsc::UnboundedReceiver<EngineCoreRequest>,
            mpsc::UnboundedReceiver<EmbedRequest>,
            mpsc::UnboundedReceiver<ControlMessage>,
        )>,
    >,
    model_name: String,
    max_model_len: usize,
    /// Notified after every engine step so request handlers can check results.
    notify: Arc<Notify>,
    /// Optional tokenizer for encoding prompts and decoding outputs.
    tokenizer: Option<Arc<Tokenizer>>,
    /// Optional chat template for formatting chat messages.
    #[cfg(feature = "chat-template")]
    chat_template: Option<Arc<ChatTemplate>>,
    /// Optional tool call parser for extracting structured tool calls from output.
    tool_parser: Option<Arc<dyn ToolCallParser>>,
    /// Whether async scheduling is enabled (overlap GPU execution with CPU scheduling).
    async_scheduling: bool,
    /// Set to `true` when the step loop is running; cleared on exit.
    /// Checked by `poll_until_done` to detect a dead step loop.
    step_loop_alive: Arc<AtomicBool>,
    /// Multimodal config: image token ID for placeholder expansion.
    /// `None` for text-only models.
    image_token_id: Option<u32>,
    /// Number of image tokens per image (vision encoder output patches).
    mm_tokens_per_image: usize,
    /// Image preprocessing size (pixels). 0 if not a VLM.
    mm_image_size: usize,
    /// Multimodal model type for preprocessing dispatch.
    /// "siglip" (Gemma3), "qwen2_vl" (Qwen2-VL/Qwen2.5-VL), or empty.
    mm_model_type: String,
    /// Whether the engine is in pooling mode (embedding requests go through scheduler).
    is_pooling: bool,
    /// Maximum time the step loop can have pending requests with no output
    /// before aborting them. Configurable for tests (default: 60s).
    no_progress_timeout: Duration,
}

impl AsyncEngine {
    /// Create a new `AsyncEngine` wrapping the given client.
    ///
    /// **Important**: call `spawn_step_loop` to start the background engine
    /// loop. Without it, submitted requests will never be processed.
    pub fn new(
        client: Box<dyn EngineCoreClient + Send>,
        model_name: String,
        max_model_len: usize,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let (embed_tx, embed_rx) = mpsc::unbounded_channel();
        let (control_tx, control_rx) = mpsc::unbounded_channel();

        Self {
            requests: Arc::new(Mutex::new(HashMap::new())),
            request_tx: tx,
            embed_tx,
            control_tx,
            pending_loop: std::sync::Mutex::new(Some((client, rx, embed_rx, control_rx))),
            model_name,
            max_model_len,
            notify: Arc::new(Notify::new()),
            tokenizer: None,
            #[cfg(feature = "chat-template")]
            chat_template: None,
            tool_parser: None,
            async_scheduling: false,
            step_loop_alive: Arc::new(AtomicBool::new(false)),
            image_token_id: None,
            mm_tokens_per_image: 0,
            mm_image_size: 0,
            mm_model_type: String::new(),
            is_pooling: false,
            no_progress_timeout: DEFAULT_NO_PROGRESS_TIMEOUT,
        }
    }

    /// Enable or disable pooling mode.
    pub fn set_is_pooling(&mut self, enabled: bool) {
        self.is_pooling = enabled;
    }

    /// Whether the engine is in pooling mode.
    pub fn is_pooling(&self) -> bool {
        self.is_pooling
    }

    /// Reset the prefix cache. Returns `true` if successful, `false` if there
    /// are running requests blocking the reset.
    ///
    /// This sends a control message to the step loop which calls through to the
    /// engine core client's `reset_prefix_cache`.
    pub async fn reset_prefix_cache(&self) -> ServeResult<bool> {
        let (tx, rx) = oneshot::channel();
        self.control_tx
            .send(ControlMessage::ResetPrefixCache(tx))
            .map_err(|_| ServeError::Internal("step loop not running".into()))?;
        rx.await
            .map_err(|_| ServeError::Internal("step loop dropped reply".into()))
    }

    /// Override the no-progress watchdog timeout (for testing).
    #[cfg(test)]
    fn set_no_progress_timeout(&mut self, timeout: Duration) {
        self.no_progress_timeout = timeout;
    }

    /// Create a new `AsyncEngine` with a tokenizer for real text processing.
    pub fn with_tokenizer(
        client: Box<dyn EngineCoreClient + Send>,
        model_name: String,
        max_model_len: usize,
        tokenizer: Arc<Tokenizer>,
    ) -> Self {
        let mut engine = Self::new(client, model_name, max_model_len);
        engine.tokenizer = Some(tokenizer);
        engine
    }

    /// Create a new `AsyncEngine` with a tokenizer and chat template.
    #[cfg(feature = "chat-template")]
    pub fn with_tokenizer_and_template(
        client: Box<dyn EngineCoreClient + Send>,
        model_name: String,
        max_model_len: usize,
        tokenizer: Arc<Tokenizer>,
        chat_template: Arc<ChatTemplate>,
    ) -> Self {
        let mut engine = Self::new(client, model_name, max_model_len);
        engine.tokenizer = Some(tokenizer);
        engine.chat_template = Some(chat_template);
        engine
    }

    /// Set the tool call parser on this engine.
    pub fn set_tool_parser(&mut self, parser: Arc<dyn ToolCallParser>) {
        self.tool_parser = Some(parser);
    }

    /// Enable or disable async scheduling.
    pub fn set_async_scheduling(&mut self, enabled: bool) {
        self.async_scheduling = enabled;
    }

    /// Configure multimodal (VLM) support.
    ///
    /// * `image_token_id` — the token ID used as image placeholder (e.g. 255999 for Gemma 3)
    /// * `mm_tokens_per_image` — number of tokens per image (vision encoder patches)
    /// * `mm_image_size` — pixel size for image preprocessing (e.g. 224 for SigLIP)
    pub fn set_multimodal_config(
        &mut self,
        image_token_id: u32,
        mm_tokens_per_image: usize,
        mm_image_size: usize,
    ) {
        self.image_token_id = Some(image_token_id);
        self.mm_tokens_per_image = mm_tokens_per_image;
        self.mm_image_size = mm_image_size;
    }

    /// Set the multimodal model type for preprocessing dispatch.
    /// e.g., "qwen2_vl" for Qwen2-VL/Qwen2.5-VL.
    pub fn set_mm_model_type(&mut self, model_type: &str) {
        self.mm_model_type = model_type.to_string();
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Get the maximum model length.
    pub fn max_model_len(&self) -> usize {
        self.max_model_len
    }

    /// Whether this engine has a tokenizer configured.
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    /// Get a reference to the tokenizer, if available.
    pub fn tokenizer(&self) -> Option<&Arc<Tokenizer>> {
        self.tokenizer.as_ref()
    }

    /// Get a reference to the chat template, if available.
    #[cfg(feature = "chat-template")]
    pub fn chat_template(&self) -> Option<&Arc<ChatTemplate>> {
        self.chat_template.as_ref()
    }

    // -----------------------------------------------------------------------
    // Request handling
    // -----------------------------------------------------------------------

    /// Add a non-streaming chat completion request.
    ///
    /// Returns the completed response when generation finishes.
    /// Supports `n > 1`: generates `n` independent completions.
    pub async fn chat_completion(
        &self,
        request: protocol::ChatCompletionRequest,
    ) -> ServeResult<protocol::ChatCompletionResponse> {
        if self.is_pooling {
            return Err(ServeError::Validation(
                "server is in pooling mode — chat completions are not supported".to_string(),
            ));
        }
        let base_id = request
            .request_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.model_name.clone());
        let n = request.n.max(1) as usize;

        let mut sampling_params = self.build_sampling_params_from_chat(&request)?;

        // Tokenize prompt once.
        let mut ec_request = self.chat_to_engine_request(&base_id, &request, &sampling_params)?;
        let mut prompt_token_ids = ec_request.prompt_token_ids.clone().unwrap_or_default();

        // Truncate prompt tokens from the left (keep the last N).
        truncate_prompt(&mut prompt_token_ids, request.truncate_prompt_tokens)?;
        ec_request.prompt_token_ids = Some(prompt_token_ids.clone());

        let num_prompt_tokens = prompt_token_ids.len() as u32;

        // Resolve max_tokens: None → remaining capacity, Some(v) → min(v, remaining).
        self.resolve_max_tokens(&mut sampling_params, prompt_token_ids.len());

        debug!(
            request_id = %base_id,
            prompt_tokens = num_prompt_tokens,
            max_tokens = ?sampling_params.max_tokens,
            max_model_len = self.max_model_len,
            temperature = sampling_params.temperature,
            "chat completion request"
        );

        // Per-HTTP-request metrics (once, not per child).
        #[cfg(feature = "metrics")]
        {
            let metrics = crate::metrics::VllmMetrics::global();
            metrics.requests_total.inc();
            metrics.prompt_tokens_total.inc_by(num_prompt_tokens as u64);
        }

        // Submit n child requests.
        let mut child_ids = Vec::with_capacity(n);
        for i in 0..n {
            let child_id = if n == 1 {
                base_id.clone()
            } else {
                format!("{base_id}-{i}")
            };

            let mut sp = sampling_params.clone();
            sp.seed = sp.seed.map(|s| s.wrapping_add(i as u64));

            let detokenizer = if sp.detokenize {
                self.tokenizer.as_ref().map(|tok| {
                    IncrementalDetokenizer::new(
                        Arc::clone(tok),
                        &prompt_token_ids,
                        sp.stop.clone(),
                        sp.min_tokens,
                        sp.include_stop_str_in_output,
                        sp.skip_special_tokens,
                    )
                })
            } else {
                None
            };

            let mut ec_req = ec_request.clone();
            ec_req.request_id = child_id.clone();
            ec_req.sampling_params = Some(sp);

            self.submit_request(
                child_id.clone(),
                ec_req,
                num_prompt_tokens,
                None,
                detokenizer,
                i as u32,
                None, // no streaming tool parser for non-streaming requests
                None,
            )
            .await?;

            child_ids.push(child_id);
        }

        // Poll all children and build choices.
        let mut choices = Vec::with_capacity(n);
        let mut total_completion_tokens = 0u32;
        let mut total_cached_tokens = 0u32;

        for (i, child_id) in child_ids.iter().enumerate() {
            let state = self.poll_until_done(child_id).await?;
            let completion_tokens = state.generated_token_ids.len() as u32;
            total_completion_tokens += completion_tokens;
            total_cached_tokens += state.num_cached_tokens;

            let finish_reason_str = state
                .finish_reason
                .map(|r| r.to_string())
                .unwrap_or_else(|| "stop".to_string());

            let text = if let Some(mut detok) = state.detokenizer {
                detok.get_next_output_text(true, false)
            } else {
                placeholder_text(&state.generated_token_ids)
            };

            let chat_logprobs = if state.logprobs.is_empty() {
                None
            } else {
                Some(build_chat_logprobs(
                    &state.logprobs,
                    self.tokenizer.as_deref(),
                ))
            };

            // Try tool call extraction if parser is configured and request has tools.
            let (final_content, final_tool_calls, final_finish_reason) =
                if let Some(ref parser) = self.tool_parser {
                    if request.tools.is_some() && !is_tool_choice_none(&request.tool_choice) {
                        let extracted = parser.extract_tool_calls(&text);
                        if extracted.tools_called {
                            // Filter by forced function name if tool_choice specifies one.
                            let tool_calls = if let Some(forced) =
                                get_tool_choice_function_name(&request.tool_choice)
                            {
                                extracted
                                    .tool_calls
                                    .into_iter()
                                    .filter(|tc| tc.function.name == forced)
                                    .collect::<Vec<_>>()
                            } else {
                                extracted.tool_calls
                            };
                            if tool_calls.is_empty() {
                                (Some(text), None, finish_reason_str)
                            } else {
                                (
                                    extracted.content,
                                    Some(tool_calls),
                                    "tool_calls".to_string(),
                                )
                            }
                        } else {
                            (Some(text), None, finish_reason_str)
                        }
                    } else {
                        (Some(text), None, finish_reason_str)
                    }
                } else {
                    (Some(text), None, finish_reason_str)
                };

            // Build prompt logprobs if available.
            let prompt_logprobs_content = state.prompt_logprobs.as_ref().map(|plps| {
                plps.iter()
                    .map(|opt_lp| {
                        opt_lp.as_ref().map(|lp| {
                            let token_str = self
                                .tokenizer
                                .as_ref()
                                .and_then(|tok| tok.decode(&[lp.sampled.token_id], false).ok())
                                .unwrap_or_else(|| format!("<token_{}>", lp.sampled.token_id));
                            let top_logprobs: Vec<protocol::ChatCompletionLogProb> = lp
                                .top_logprobs
                                .iter()
                                .map(|tlp| {
                                    let t = self
                                        .tokenizer
                                        .as_ref()
                                        .and_then(|tok| tok.decode(&[tlp.token_id], false).ok())
                                        .unwrap_or_else(|| format!("<token_{}>", tlp.token_id));
                                    protocol::ChatCompletionLogProb {
                                        token: t,
                                        logprob: tlp.logprob as f64,
                                        bytes: None,
                                    }
                                })
                                .collect();
                            protocol::ChatCompletionLogProbsContent {
                                token: token_str,
                                logprob: lp.sampled.logprob as f64,
                                bytes: None,
                                top_logprobs,
                            }
                        })
                    })
                    .collect()
            });

            choices.push(protocol::ChatCompletionResponseChoice {
                index: i as u32,
                message: protocol::ChatMessage {
                    role: "assistant".to_string(),
                    content: final_content,
                    refusal: None,
                    tool_calls: final_tool_calls,
                    reasoning: None,
                },
                logprobs: chat_logprobs,
                finish_reason: Some(final_finish_reason),
                stop_reason: state.stop_reason.map(|sr| match sr {
                    StopReason::Token(id) => serde_json::Value::Number(id.into()),
                    StopReason::String(s) => serde_json::Value::String(s),
                }),
                prompt_logprobs: prompt_logprobs_content,
            });
        }

        let usage = protocol::UsageInfo {
            prompt_tokens: num_prompt_tokens,
            completion_tokens: Some(total_completion_tokens),
            total_tokens: num_prompt_tokens + total_completion_tokens,
            prompt_tokens_details: if total_cached_tokens > 0 {
                Some(protocol::PromptTokenUsageInfo {
                    cached_tokens: Some(total_cached_tokens),
                })
            } else {
                None
            },
        };

        Ok(protocol::ChatCompletionResponse::new(model, choices, usage))
    }

    /// Render a chat completion request: apply the chat template and tokenize,
    /// but do not generate any tokens.
    ///
    /// Returns `[conversation, engine_prompts]` matching Python vLLM's format.
    ///
    /// Port of: `POST /v1/chat/completions/render`
    pub fn render_chat_completion(
        &self,
        request: protocol::ChatCompletionRequest,
    ) -> ServeResult<protocol::ChatCompletionRenderResponse> {
        if self.is_pooling {
            return Err(ServeError::Validation(
                "server is in pooling mode — chat completions are not supported".to_string(),
            ));
        }

        // Build conversation: the original messages as JSON values.
        let conversation: Vec<serde_json::Value> = request
            .messages
            .iter()
            .map(|msg| serde_json::to_value(msg).unwrap_or_default())
            .collect();

        // Apply chat template + tokenize to get the rendered prompt.
        let base_id = request
            .request_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let sampling_params = self.build_sampling_params_from_chat(&request)?;
        let ec_request = self.chat_to_engine_request(&base_id, &request, &sampling_params)?;
        let token_ids = ec_request.prompt_token_ids.unwrap_or_default();

        // Decode token IDs back to text for the prompt field.
        let prompt_text = if let Some(tok) = &self.tokenizer {
            tok.decode(&token_ids, true)
                .unwrap_or_else(|_| String::new())
        } else {
            String::new()
        };

        let engine_prompts = vec![protocol::RenderEnginePrompt {
            prompt: serde_json::Value::String(prompt_text),
        }];

        Ok((conversation, engine_prompts))
    }

    /// Add a streaming chat completion request.
    ///
    /// Returns a receiver that yields streaming deltas.
    /// Supports `n > 1`: all child requests share one channel; each delta
    /// carries its `choice_index` so the SSE layer can route correctly.
    pub async fn chat_completion_stream(
        &self,
        request: protocol::ChatCompletionRequest,
    ) -> ServeResult<(String, String, mpsc::UnboundedReceiver<StreamDelta>)> {
        let base_id = request
            .request_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.model_name.clone());
        let n = request.n.max(1) as usize;

        let mut sampling_params = self.build_sampling_params_from_chat(&request)?;
        let mut ec_request = self.chat_to_engine_request(&base_id, &request, &sampling_params)?;
        let mut prompt_token_ids = ec_request.prompt_token_ids.clone().unwrap_or_default();

        // Truncate prompt tokens from the left (keep the last N).
        truncate_prompt(&mut prompt_token_ids, request.truncate_prompt_tokens)?;
        ec_request.prompt_token_ids = Some(prompt_token_ids.clone());

        let num_prompt_tokens = prompt_token_ids.len() as u32;

        // Resolve max_tokens: None → remaining capacity, Some(v) → min(v, remaining).
        self.resolve_max_tokens(&mut sampling_params, prompt_token_ids.len());

        // Per-HTTP-request metrics.
        #[cfg(feature = "metrics")]
        {
            let metrics = crate::metrics::VllmMetrics::global();
            metrics.requests_total.inc();
            metrics.prompt_tokens_total.inc_by(num_prompt_tokens as u64);
        }

        // Determine if tool parsing is active for this request.
        let use_tool_parser = self.tool_parser.is_some()
            && request.tools.is_some()
            && !is_tool_choice_none(&request.tool_choice);

        let forced_fn = get_tool_choice_function_name(&request.tool_choice);

        // All n children share one channel.
        let (tx, rx) = mpsc::unbounded_channel();

        for i in 0..n {
            let child_id = if n == 1 {
                base_id.clone()
            } else {
                format!("{base_id}-{i}")
            };

            let mut sp = sampling_params.clone();
            sp.seed = sp.seed.map(|s| s.wrapping_add(i as u64));

            let detokenizer = if sp.detokenize {
                self.tokenizer.as_ref().map(|tok| {
                    IncrementalDetokenizer::new(
                        Arc::clone(tok),
                        &prompt_token_ids,
                        sp.stop.clone(),
                        sp.min_tokens,
                        sp.include_stop_str_in_output,
                        sp.skip_special_tokens,
                    )
                })
            } else {
                None
            };

            let mut ec_req = ec_request.clone();
            ec_req.request_id = child_id.clone();
            ec_req.sampling_params = Some(sp);

            // Create streaming tool parser state if applicable.
            let tool_state = if use_tool_parser {
                self.tool_parser
                    .as_ref()
                    .map(|p| p.create_streaming_state())
            } else {
                None
            };

            self.submit_request(
                child_id,
                ec_req,
                num_prompt_tokens,
                Some(tx.clone()),
                detokenizer,
                i as u32,
                tool_state,
                forced_fn.clone(),
            )
            .await?;
        }

        // Drop the original sender so rx closes when all children finish.
        drop(tx);

        Ok((base_id, model, rx))
    }

    /// Add a non-streaming completion request.
    ///
    /// Supports `n > 1` and multi-prompt: generates `num_prompts * n` choices.
    pub async fn completion(
        &self,
        request: protocol::CompletionRequest,
    ) -> ServeResult<protocol::CompletionResponse> {
        if self.is_pooling {
            return Err(ServeError::Validation(
                "server is in pooling mode — text completions are not supported".to_string(),
            ));
        }
        let base_id = request
            .request_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.model_name.clone());
        let n = request.n.max(1) as usize;

        let sampling_params = self.build_sampling_params_from_completion(&request)?;

        // Normalize prompt to Vec<Vec<u32>>.
        let mut prompts = self.tokenize_completion_prompts(&request)?;

        // Truncate each prompt from the left (keep the last N tokens).
        for p in &mut prompts {
            truncate_prompt(p, request.truncate_prompt_tokens)?;
        }
        let total = prompts.len() * n;

        // Per-HTTP-request metrics (once).
        let total_prompt_tokens: u32 = prompts.iter().map(|p| p.len() as u32).sum();
        #[cfg(feature = "metrics")]
        {
            let metrics = crate::metrics::VllmMetrics::global();
            metrics.requests_total.inc();
            metrics
                .prompt_tokens_total
                .inc_by(total_prompt_tokens as u64);
        }

        // Submit all child requests.
        let mut child_ids: Vec<(String, u32)> = Vec::with_capacity(total);

        for (p_idx, prompt_ids) in prompts.iter().enumerate() {
            let num_prompt_tokens = prompt_ids.len() as u32;

            for n_idx in 0..n {
                let choice_index = (p_idx * n + n_idx) as u32;
                let child_id = if total == 1 {
                    base_id.clone()
                } else {
                    format!("{base_id}-{choice_index}")
                };

                let mut sp = sampling_params.clone();
                sp.seed = sp.seed.map(|s| s.wrapping_add(n_idx as u64));
                self.resolve_max_tokens(&mut sp, prompt_ids.len());

                let ec_req = EngineCoreRequest {
                    request_id: child_id.clone(),
                    prompt_token_ids: Some(prompt_ids.clone()),
                    sampling_params: Some(sp.clone()),
                    arrival_time: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64(),
                    client_index: 0,
                    priority: request.priority,
                    cache_salt: request.cache_salt.clone(),
                    data_parallel_rank: None,
                    is_pooling: false,
                    mm_data: None,
                };

                let detokenizer = self.tokenizer.as_ref().map(|tok| {
                    IncrementalDetokenizer::new(
                        Arc::clone(tok),
                        prompt_ids,
                        sp.stop.clone(),
                        sp.min_tokens,
                        sp.include_stop_str_in_output,
                        sp.skip_special_tokens,
                    )
                });

                self.submit_request(
                    child_id.clone(),
                    ec_req,
                    num_prompt_tokens,
                    None,
                    detokenizer,
                    choice_index,
                    None, // no tool parsing for completions
                    None,
                )
                .await?;

                child_ids.push((child_id, choice_index));
            }
        }

        // Poll all children and build choices.
        let mut choices = Vec::with_capacity(total);
        let mut total_completion_tokens = 0u32;

        for (child_id, choice_index) in &child_ids {
            let state = self.poll_until_done(child_id).await?;
            let completion_tokens = state.generated_token_ids.len() as u32;
            total_completion_tokens += completion_tokens;

            let finish_reason_str = state
                .finish_reason
                .map(|r| r.to_string())
                .unwrap_or_else(|| "stop".to_string());

            let text = if let Some(mut detok) = state.detokenizer {
                detok.get_next_output_text(true, false)
            } else {
                placeholder_text(&state.generated_token_ids)
            };

            let completion_logprobs = if state.logprobs.is_empty() {
                None
            } else {
                Some(build_completion_logprobs(
                    &state.logprobs,
                    self.tokenizer.as_deref(),
                ))
            };

            // Build prompt logprobs for completion response if available.
            let prompt_lps = state.prompt_logprobs.as_ref().and_then(|plps| {
                // Convert Option<LogprobsOutput> entries (skipping the None for pos 0)
                // into a flat Vec for build_completion_logprobs.
                let flat: Vec<vllm_common::LogprobsOutput> =
                    plps.iter().filter_map(|opt| opt.clone()).collect();
                if flat.is_empty() {
                    None
                } else {
                    Some(build_completion_logprobs(&flat, self.tokenizer.as_deref()))
                }
            });

            choices.push(protocol::CompletionResponseChoice {
                index: *choice_index,
                text,
                logprobs: completion_logprobs,
                finish_reason: Some(finish_reason_str),
                stop_reason: state.stop_reason.map(|sr| match sr {
                    StopReason::Token(id) => serde_json::Value::Number(id.into()),
                    StopReason::String(s) => serde_json::Value::String(s),
                }),
                prompt_logprobs: prompt_lps,
            });
        }

        let usage = protocol::UsageInfo {
            prompt_tokens: total_prompt_tokens,
            completion_tokens: Some(total_completion_tokens),
            total_tokens: total_prompt_tokens + total_completion_tokens,
            prompt_tokens_details: None,
        };

        Ok(protocol::CompletionResponse::new(model, choices, usage))
    }

    // -----------------------------------------------------------------------
    // Abort
    // -----------------------------------------------------------------------

    /// Abort a request by ID.
    ///
    /// Removes the request from tracking. The engine core will eventually
    /// notice the request has no consumer and clean it up.
    pub async fn abort_request(&self, request_id: &str) {
        let mut reqs = self.requests.lock().await;
        reqs.remove(request_id);
    }

    // -----------------------------------------------------------------------
    // Embedding
    // -----------------------------------------------------------------------

    /// Process an embedding request.
    ///
    /// In pooling mode, routes through the scheduler (batched, lifecycle-managed).
    /// In default mode, uses the side-channel embed path (sequential).
    pub async fn embeddings(
        &self,
        request: protocol::EmbeddingRequest,
    ) -> ServeResult<protocol::EmbeddingResponse> {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.model_name.clone());

        // Tokenize inputs into token ID sequences.
        let token_id_seqs = self.tokenize_embedding_inputs(&request)?;
        let total_prompt_tokens: u32 = token_id_seqs.iter().map(|s| s.len() as u32).sum();

        let embeddings: Vec<Vec<f32>> = if self.is_pooling {
            // Pooling mode: route each input through the scheduler as a pooling request.
            let mut results = Vec::with_capacity(token_id_seqs.len());
            for token_ids in &token_id_seqs {
                let request_id = Uuid::new_v4().to_string();
                let num_prompt_tokens = token_ids.len() as u32;

                let ec_request = EngineCoreRequest {
                    request_id: request_id.clone(),
                    prompt_token_ids: Some(token_ids.clone()),
                    sampling_params: Some(SamplingParams {
                        max_tokens: Some(1), // Will never be used — pooling finishes in one pass.
                        ..Default::default()
                    }),
                    arrival_time: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64(),
                    client_index: 0,
                    priority: 0,
                    cache_salt: None,
                    data_parallel_rank: None,
                    is_pooling: true,
                    mm_data: None,
                };

                self.submit_request(
                    request_id.clone(),
                    ec_request,
                    num_prompt_tokens,
                    None,
                    None,
                    0,
                    None,
                    None,
                )
                .await?;

                // Wait for the request to complete.
                let state = self.poll_until_done(&request_id).await?;

                // Extract the embedding vector from the pooler output.
                let emb = state.pooler_output.ok_or_else(|| {
                    ServeError::Internal("pooling request completed without embedding".into())
                })?;
                results.push(emb);
            }
            results
        } else {
            // Default mode: send to step loop via embed channel (bypasses scheduler).
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            self.embed_tx
                .send(EmbedRequest {
                    token_id_seqs,
                    reply: reply_tx,
                })
                .map_err(|_| ServeError::Internal("embed channel closed".into()))?;

            reply_rx
                .await
                .map_err(|_| ServeError::Internal("embed reply channel closed".into()))??
        };

        // Apply optional dimension truncation and re-normalize.
        let data: Vec<protocol::EmbeddingObject> = embeddings
            .into_iter()
            .enumerate()
            .map(|(i, mut emb)| {
                if let Some(dims) = request.dimensions
                    && dims < emb.len()
                {
                    emb.truncate(dims);
                    // Re-normalize after truncation.
                    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for x in &mut emb {
                            *x /= norm;
                        }
                    }
                }
                protocol::EmbeddingObject {
                    index: i,
                    object: "embedding".to_string(),
                    embedding: emb,
                }
            })
            .collect();

        Ok(protocol::EmbeddingResponse::new(
            model,
            data,
            protocol::EmbeddingUsage {
                prompt_tokens: total_prompt_tokens,
                total_tokens: total_prompt_tokens,
            },
        ))
    }

    /// Tokenize embedding request inputs into token ID sequences.
    fn tokenize_embedding_inputs(
        &self,
        request: &protocol::EmbeddingRequest,
    ) -> ServeResult<Vec<Vec<u32>>> {
        match &request.input {
            protocol::EmbeddingInput::Single(text) => {
                let ids = self.tokenize_embed_text(text)?;
                Ok(vec![ids])
            }
            protocol::EmbeddingInput::Multiple(items) => {
                let mut seqs = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        protocol::EmbeddingInputItem::Text(text) => {
                            seqs.push(self.tokenize_embed_text(text)?);
                        }
                        protocol::EmbeddingInputItem::TokenIds(ids) => {
                            seqs.push(ids.clone());
                        }
                    }
                }
                Ok(seqs)
            }
        }
    }

    /// Tokenize a single text string for embedding.
    fn tokenize_embed_text(&self, text: &str) -> ServeResult<Vec<u32>> {
        if let Some(ref tokenizer) = self.tokenizer {
            tokenizer.encode(text, false)
        } else {
            // Fallback: byte-level tokenization (for testing without a tokenizer).
            Ok(text.bytes().map(|b| b as u32).collect())
        }
    }

    // -----------------------------------------------------------------------
    // Engine step loop
    // -----------------------------------------------------------------------

    /// Start the background step loop.
    ///
    /// Takes ownership of the engine client and moves it into a background
    /// task. After this call, the engine client is exclusively owned by the
    /// step loop — all interaction goes through channels and the shared
    /// request map.
    ///
    /// Must be called exactly once. Panics if called twice.
    pub fn spawn_step_loop(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let (client, rx, embed_rx, control_rx) = self
            .pending_loop
            .lock()
            .expect("spawn_step_loop lock poisoned")
            .take()
            .expect("spawn_step_loop called twice");

        let alive = Arc::clone(&self.step_loop_alive);
        // Set alive BEFORE spawning to avoid a race where poll_until_done
        // sees alive=false before the spawned task starts.
        alive.store(true, Ordering::Release);

        let args = StepLoopArgs {
            client,
            request_rx: rx,
            embed_rx,
            control_rx,
            requests: Arc::clone(&self.requests),
            notify: Arc::clone(&self.notify),
            alive,
            no_progress_timeout: self.no_progress_timeout,
        };
        if self.async_scheduling {
            info!("Async scheduling enabled — overlapping GPU execution with CPU scheduling");
            Self::spawn_step_loop_async(args)
        } else {
            Self::spawn_step_loop_inner(args)
        }
    }

    /// Handle a control message from the HTTP layer.
    fn handle_control_message(client: &mut Box<dyn EngineCoreClient + Send>, msg: ControlMessage) {
        match msg {
            ControlMessage::ResetPrefixCache(reply) => {
                let result = client.reset_prefix_cache().unwrap_or(false);
                let _ = reply.send(result);
            }
        }
    }

    /// Internal: actually spawn the step loop with the client.
    fn spawn_step_loop_inner(args: StepLoopArgs) -> tokio::task::JoinHandle<()> {
        let StepLoopArgs {
            mut client,
            mut request_rx,
            mut embed_rx,
            mut control_rx,
            requests,
            notify,
            alive,
            no_progress_timeout,
        } = args;
        tokio::spawn(async move {
            let mut last_progress = Instant::now();
            loop {
                // 0. Drain pending embedding requests (synchronous, bypasses scheduler).
                while let Ok(embed_req) = embed_rx.try_recv() {
                    let result =
                        tokio::task::block_in_place(|| client.embed(embed_req.token_id_seqs));
                    let _ = embed_req.reply.send(result.map_err(ServeError::from));
                }

                // 0b. Drain control messages.
                while let Ok(msg) = control_rx.try_recv() {
                    Self::handle_control_message(&mut client, msg);
                }

                // 1. Drain pending request submissions (non-blocking).
                //    This never blocks on a mutex — the channel is lock-free.
                let mut added = false;
                while let Ok(ec_request) = request_rx.try_recv() {
                    if let Err(e) = client.add_request(ec_request) {
                        error!("Failed to add request: {}", e);
                    }
                    added = true;
                }

                // 2. Check if there's work. If not, wait for a request.
                let has_requests = {
                    let reqs = requests.lock().await;
                    !reqs.is_empty()
                };

                if !has_requests && !added {
                    // Reset watchdog timer when idle (no pending requests).
                    last_progress = Instant::now();
                    // Block until a new request arrives on either channel.
                    tokio::select! {
                        Some(ec_request) = request_rx.recv() => {
                            if let Err(e) = client.add_request(ec_request) {
                                error!("Failed to add request: {}", e);
                            }
                        }
                        Some(embed_req) = embed_rx.recv() => {
                            let result = tokio::task::block_in_place(|| {
                                client.embed(embed_req.token_id_seqs)
                            });
                            let _ = embed_req.reply.send(result.map_err(ServeError::from));
                            // After handling embed, continue loop to check for more work.
                            notify.notify_waiters();
                            continue;
                        }
                        Some(msg) = control_rx.recv() => {
                            Self::handle_control_message(&mut client, msg);
                            continue;
                        }
                        else => break, // All channels closed, engine dropped.
                    }
                }

                // 3. Run one engine step. This is synchronous (model forward)
                //    so we use block_in_place to let tokio schedule other work.
                let step_result = tokio::task::block_in_place(|| client.get_output());

                match step_result {
                    Ok((outputs, model_executed)) => {
                        // 4. Update scheduler gauges from stats.
                        #[cfg(feature = "metrics")]
                        if let Some(stats) = &outputs.scheduler_stats {
                            let m = crate::metrics::VllmMetrics::global();
                            m.num_requests_running.set(stats.num_running_reqs as f64);
                            m.num_requests_waiting.set(stats.num_waiting_reqs as f64);
                            m.kv_cache_usage_perc.set(stats.kv_cache_usage);
                            m.gpu_cache_blocks_used
                                .set(stats.gpu_cache_blocks_used as i64);
                            m.gpu_cache_blocks_total
                                .set(stats.gpu_cache_blocks_total as i64);
                            m.prefix_cache_blocks.set(stats.num_cached_blocks as i64);
                        }

                        // 5. Route outputs to requests (brief lock).
                        if !outputs.outputs.is_empty() {
                            last_progress = Instant::now();

                            // Phase 1: accumulate tokens, take detokenizers (brief lock).
                            let detok_work = {
                                let mut reqs = requests.lock().await;
                                Self::process_outputs_phase1(&mut reqs, &outputs.outputs)
                            };

                            // Phase 2: parallel detokenize (no lock held).
                            let detok_results = Self::parallel_detokenize(detok_work);

                            // Phase 3: apply detok results, send streams (brief lock).
                            {
                                let mut reqs = requests.lock().await;
                                Self::process_outputs_phase3(
                                    &mut reqs,
                                    outputs.outputs,
                                    detok_results,
                                );
                            }

                            // Wake waiters only when there are actual outputs.
                            notify.notify_waiters();
                        } else if has_requests && last_progress.elapsed() > no_progress_timeout {
                            // Watchdog: no output tokens for too long with pending
                            // requests — likely a scheduler bug. Abort everything.
                            let reqs_count = {
                                let r = requests.lock().await;
                                r.len()
                            };
                            error!(
                                "No output tokens for {:?} with {} pending requests \
                                 — aborting all (no-progress watchdog)",
                                last_progress.elapsed(),
                                reqs_count,
                            );
                            let err_msg = format!(
                                "No-progress watchdog: no output tokens for {:?}",
                                last_progress.elapsed(),
                            );
                            client.abort_running_requests();
                            let mut reqs = requests.lock().await;
                            for req_state in reqs.values_mut() {
                                if req_state.finish_reason.is_none() {
                                    req_state.finish_reason = Some(FinishReason::Abort);
                                    req_state.error = Some(err_msg.clone());
                                }
                            }
                            drop(reqs);
                            notify.notify_waiters();
                            last_progress = Instant::now();
                        } else if !model_executed {
                            // No work done — yield to avoid busy-spinning.
                            tokio::task::yield_now().await;
                        }
                    }
                    Err(e) => {
                        error!("Engine step error: {}", e);
                        // Abort all running requests to prevent infinite retry loop.
                        // get_output() internally does schedule + execute + finalize,
                        // and we don't know which requests were in the failed batch,
                        // so abort everything the scheduler currently has.
                        let err_msg = format!("Engine step error: {e}");
                        client.abort_running_requests();
                        let mut reqs = requests.lock().await;
                        for req_state in reqs.values_mut() {
                            if req_state.finish_reason.is_none() {
                                req_state.finish_reason = Some(FinishReason::Abort);
                                req_state.error = Some(err_msg.clone());
                            }
                        }
                        drop(reqs);
                        notify.notify_waiters();
                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    }
                }
            }
            alive.store(false, Ordering::Release);
            notify.notify_waiters(); // Wake any poll_until_done waiters so they see the dead loop.
        })
    }

    /// Async scheduling step loop: overlaps GPU execution with CPU work.
    ///
    /// The executor is moved to a dedicated OS thread. The tokio task handles
    /// scheduling, output processing, and request drain concurrently with
    /// GPU execution.
    fn spawn_step_loop_async(args: StepLoopArgs) -> tokio::task::JoinHandle<()> {
        let StepLoopArgs {
            mut client,
            mut request_rx,
            mut embed_rx,
            mut control_rx,
            requests,
            notify,
            alive,
            no_progress_timeout,
        } = args;
        // Take the executor out of the client for the dedicated thread.
        let executor = client
            .take_executor()
            .expect("async scheduling requires an executor");

        // Bounded channel with capacity 2: allows up to 2 batches in flight
        // (one executing, one queued) without blocking the tokio thread.
        let (sched_tx, sched_rx) = tokio::sync::mpsc::channel::<ExecutorWork>(2);
        let (model_tx, mut model_rx) = tokio::sync::mpsc::channel::<ExecutorResult>(2);

        // Spawn the executor on a dedicated OS thread.
        std::thread::Builder::new()
            .name("vllm-executor".into())
            .spawn(move || executor_thread_loop(executor, sched_rx, model_tx))
            .expect("failed to spawn executor thread");

        tokio::spawn(async move {
            // Pre-scheduling with deferred finalization:
            //
            // GPU: ─execute(N)───────────────────────┬─execute(N+1)──
            //                                        │ (pre-queued)
            // CPU: finalize(N-1)→sched(N+1)→wait(N)→send│→finalize(N)→...
            //
            // The scheduler uses `num_output_placeholders` to account for
            // tokens that are in-flight on the GPU but not yet finalized.
            // This allows schedule(N+1) to run before finalize(N) without
            // double-scheduling the same position.
            let mut deferred: Option<(
                Box<vllm_core::scheduler::output::SchedulerOutput>,
                ModelRunnerOutput,
            )> = None;
            let mut gpu_in_flight: u32 = 0;
            let mut last_progress = Instant::now();

            loop {
                // 0. Drain pending embedding requests.
                while let Ok(embed_req) = embed_rx.try_recv() {
                    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
                    if sched_tx
                        .send(ExecutorWork::Embed(embed_req.token_id_seqs, reply_tx))
                        .await
                        .is_err()
                    {
                        let _ = embed_req.reply.send(Err(ServeError::Internal(
                            "executor thread shut down".into(),
                        )));
                        continue;
                    }
                    let result = reply_rx
                        .await
                        .unwrap_or(Err(ServeError::Internal("embed reply lost".into())));
                    let _ = embed_req.reply.send(result);
                }

                // 0b. Drain control messages.
                while let Ok(msg) = control_rx.try_recv() {
                    Self::handle_control_message(&mut client, msg);
                }

                // 1. Drain pending request submissions (non-blocking).
                let mut added = false;
                while let Ok(ec_request) = request_rx.try_recv() {
                    if let Err(e) = client.add_request(ec_request) {
                        error!("Failed to add request: {}", e);
                    }
                    added = true;
                }

                // 2. Finalize the previously completed step. The GPU is
                //    already running (or about to run) the next batch, so
                //    this CPU work overlaps with GPU execution.
                if let Some((prev_sched, mut prev_output)) = deferred.take() {
                    prev_output.resolve();
                    match client.finalize_step(&prev_sched, &prev_output) {
                        Ok(outputs) => {
                            if route_step_outputs(&requests, outputs).await {
                                last_progress = Instant::now();
                            }
                        }
                        Err(e) => {
                            error!("finalize_step error: {}", e);
                        }
                    }
                    notify.notify_waiters();
                }

                // 3. Pre-schedule: fill the pipeline up to 2 in-flight
                //    batches. With placeholder tracking, the scheduler
                //    correctly accounts for tokens still on the GPU.
                while gpu_in_flight < 2 {
                    match client.schedule_next() {
                        Ok(Some(sched)) => {
                            if sched_tx
                                .send(ExecutorWork::Execute(Box::new(sched)))
                                .await
                                .is_err()
                            {
                                break; // Executor thread exited.
                            }
                            gpu_in_flight += 1;
                        }
                        Ok(None) => break, // Nothing to schedule.
                        Err(e) => {
                            error!("schedule_next error: {}", e);
                            break;
                        }
                    }
                }

                // 4. Wait for the oldest GPU result.
                if gpu_in_flight > 0 {
                    match model_rx.recv().await {
                        Some(ExecutorResult::Model(Ok(model_output), sched)) => {
                            deferred = Some((sched, model_output));
                            gpu_in_flight -= 1;
                        }
                        Some(ExecutorResult::Model(Err(e), sched)) => {
                            error!("Executor error: {}", e);
                            let err_msg = format!("Executor error: {e}");
                            let req_ids: Vec<String> =
                                sched.num_scheduled_tokens.keys().cloned().collect();
                            client
                                .abort_requests(&req_ids)
                                .unwrap_or_else(|e| error!("abort_requests failed: {e}"));
                            let mut reqs = requests.lock().await;
                            for req_id in &req_ids {
                                if let Some(req_state) = reqs.get_mut(req_id) {
                                    req_state.finish_reason = Some(FinishReason::Abort);
                                    req_state.error = Some(err_msg.clone());
                                }
                            }
                            drop(reqs);
                            notify.notify_waiters();
                            gpu_in_flight -= 1;
                        }
                        None => {
                            // Executor thread exited.
                            break;
                        }
                    }

                    // Drain requests that arrived while waiting for GPU.
                    while let Ok(ec_request) = request_rx.try_recv() {
                        if let Err(e) = client.add_request(ec_request) {
                            error!("Failed to add request: {}", e);
                        }
                    }
                } else {
                    // 5. No GPU work — idle-wait for a new request.
                    let has_requests = {
                        let reqs = requests.lock().await;
                        !reqs.is_empty()
                    };

                    if !has_requests && !added {
                        last_progress = Instant::now();
                        tokio::select! {
                            Some(ec_request) = request_rx.recv() => {
                                if let Err(e) = client.add_request(ec_request) {
                                    error!("Failed to add request: {}", e);
                                }
                            }
                            Some(embed_req) = embed_rx.recv() => {
                                let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
                                if sched_tx
                                    .send(ExecutorWork::Embed(
                                        embed_req.token_id_seqs,
                                        reply_tx,
                                    ))
                                    .await
                                    .is_err()
                                {
                                    let _ = embed_req.reply.send(Err(ServeError::Internal(
                                        "executor thread shut down".into(),
                                    )));
                                } else {
                                    let result = reply_rx.await.unwrap_or(Err(
                                        ServeError::Internal("embed reply lost".into()),
                                    ));
                                    let _ = embed_req.reply.send(result);
                                }
                                notify.notify_waiters();
                                continue;
                            }
                            Some(msg) = control_rx.recv() => {
                                Self::handle_control_message(&mut client, msg);
                                continue;
                            }
                            else => break, // All channels closed.
                        }
                    } else if has_requests && last_progress.elapsed() > no_progress_timeout {
                        // Watchdog: no output tokens for too long with pending
                        // requests — likely a scheduler bug. Abort everything.
                        let reqs_count = {
                            let r = requests.lock().await;
                            r.len()
                        };
                        error!(
                            "No output tokens for {:?} with {} pending requests \
                             — aborting all (no-progress watchdog, async)",
                            last_progress.elapsed(),
                            reqs_count,
                        );
                        let err_msg = format!(
                            "No-progress watchdog: no output tokens for {:?}",
                            last_progress.elapsed(),
                        );
                        client.abort_running_requests();
                        let mut reqs = requests.lock().await;
                        for req_state in reqs.values_mut() {
                            if req_state.finish_reason.is_none() {
                                req_state.finish_reason = Some(FinishReason::Abort);
                                req_state.error = Some(err_msg.clone());
                            }
                        }
                        drop(reqs);
                        notify.notify_waiters();
                        last_progress = Instant::now();
                    } else {
                        tokio::task::yield_now().await;
                    }
                }
            }

            // Finalize any remaining deferred output before shutdown.
            if let Some((prev_sched, mut prev_output)) = deferred.take() {
                prev_output.resolve();
                if let Ok(outputs) = client.finalize_step(&prev_sched, &prev_output) {
                    route_step_outputs(&requests, outputs).await;
                }
                notify.notify_waiters();
            }

            alive.store(false, Ordering::Release);
            notify.notify_waiters(); // Wake any poll_until_done waiters so they see the dead loop.

            // Shutdown: drop the sender so the executor thread exits.
            let _ = sched_tx.send(ExecutorWork::Shutdown).await;
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Submit a request to the engine.
    ///
    /// Inserts the request state into the map (brief lock), then sends the
    /// engine request to the step loop via channel (non-blocking).
    ///
    /// Note: `requests_total` and `prompt_tokens_total` metrics are the
    /// caller's responsibility (once per HTTP request). This method only
    /// increments `requests_active` (once per engine-core request).
    #[allow(clippy::too_many_arguments)]
    async fn submit_request(
        &self,
        request_id: String,
        ec_request: EngineCoreRequest,
        num_prompt_tokens: u32,
        stream_tx: Option<mpsc::UnboundedSender<StreamDelta>>,
        detokenizer: Option<IncrementalDetokenizer>,
        choice_index: u32,
        tool_parser_state: Option<Box<dyn StreamingToolParserState + Send>>,
        forced_function_name: Option<String>,
    ) -> ServeResult<()> {
        #[cfg(feature = "metrics")]
        {
            let metrics = crate::metrics::VllmMetrics::global();
            metrics.requests_active.inc();
        }

        // Insert request state (brief lock, no engine interaction).
        {
            let mut reqs = self.requests.lock().await;
            reqs.insert(
                request_id,
                RequestState {
                    generated_token_ids: Vec::new(),
                    num_prompt_tokens,
                    num_cached_tokens: 0,
                    finish_reason: None,
                    error: None,
                    stop_reason: None,
                    stream_tx,
                    detokenizer,
                    choice_index,
                    submit_time: Instant::now(),
                    first_token_time: None,
                    last_token_time: None,
                    itl_count: 0,
                    itl_sum: 0.0,
                    logprobs: Vec::new(),
                    prompt_logprobs: None,
                    tool_parser_state,
                    accumulated_text: String::new(),
                    tool_calls_emitted: false,
                    forced_function_name,
                    pooler_output: None,
                },
            );
        }

        // Send to the step loop via channel (non-blocking, no mutex).
        self.request_tx
            .send(ec_request)
            .map_err(|_| ServeError::Engine("engine step loop shut down".to_string()))?;

        Ok(())
    }

    /// Poll until a request completes, returning its final state.
    ///
    /// This does NOT call `step()` itself — the background step loop
    /// (spawned by `spawn_step_loop`) drives the engine. This method
    /// just waits for the `Notify` signal after each step and checks
    /// whether the request is done. This allows many concurrent requests
    /// to wait efficiently without fighting over the engine mutex.
    async fn poll_until_done(&self, request_id: &str) -> ServeResult<RequestState> {
        loop {
            // Register interest in notifications BEFORE checking state.
            // This prevents the classic Notify race: if notify_waiters()
            // fires between our check and our await, we still catch it.
            let notified = self.notify.notified();

            // Check if the step loop is still alive.
            if !self.step_loop_alive.load(Ordering::Acquire) {
                // Step loop has exited — check one more time for results,
                // then error out if the request isn't done.
                let mut reqs = self.requests.lock().await;
                if let Some(req_state) = reqs.get(request_id)
                    && req_state.finish_reason.is_some()
                {
                    let state = reqs.remove(request_id).unwrap();
                    if let Some(ref err) = state.error {
                        return Err(ServeError::Internal(err.clone()));
                    }
                    return Ok(state);
                }
                return Err(ServeError::Internal(
                    "step loop exited before request completed".to_string(),
                ));
            }

            // Check if the request is done (brief lock).
            {
                let mut reqs = self.requests.lock().await;
                if let Some(req_state) = reqs.get(request_id)
                    && req_state.finish_reason.is_some()
                {
                    let state = reqs.remove(request_id).unwrap();
                    if let Some(ref err) = state.error {
                        return Err(ServeError::Internal(err.clone()));
                    }
                    return Ok(state);
                } else if !reqs.contains_key(request_id) {
                    return Err(ServeError::RequestNotFound(request_id.to_string()));
                }
            }

            // Wait for the background step loop to produce new outputs.
            notified.await;
        }
    }

    // ------------------------------------------------------------------
    // Three-phase parallel detokenization
    // ------------------------------------------------------------------

    /// Phase 1: Under lock — accumulate tokens, timing, logprobs on each
    /// request and take out the detokenizer for parallel processing.
    ///
    /// Returns one entry per output. `Some(...)` contains the detokenizer
    /// and the data it needs; `None` means the output had no detokenizer
    /// (or the request was not found).
    fn process_outputs_phase1(
        requests: &mut HashMap<String, RequestState>,
        outputs: &[EngineCoreOutput],
    ) -> Vec<Option<DetokWork>> {
        outputs
            .iter()
            .map(|output| {
                let Some(req_state) = requests.get_mut(&output.request_id) else {
                    debug!("Output for unknown request {}, ignoring", output.request_id);
                    return None;
                };

                // Track output tokens.
                #[cfg(feature = "metrics")]
                {
                    let metrics = crate::metrics::VllmMetrics::global();
                    metrics
                        .output_tokens_total
                        .inc_by(output.new_token_ids.len() as u64);
                }

                // --- TTFT / ITL timing ---
                let now = Instant::now();
                if !output.new_token_ids.is_empty() {
                    if req_state.first_token_time.is_none() {
                        req_state.first_token_time = Some(now);
                        #[cfg(feature = "metrics")]
                        {
                            let ttft = now.duration_since(req_state.submit_time).as_secs_f64();
                            crate::metrics::VllmMetrics::global()
                                .time_to_first_token_seconds
                                .observe(ttft);
                        }
                    } else if let Some(last) = req_state.last_token_time {
                        let itl = now.duration_since(last).as_secs_f64();
                        #[cfg(feature = "metrics")]
                        crate::metrics::VllmMetrics::global()
                            .inter_token_latency_seconds
                            .observe(itl);
                        req_state.itl_count += 1;
                        req_state.itl_sum += itl;
                    }
                    req_state.last_token_time = Some(now);
                }

                // Accumulate tokens and logprobs.
                req_state.generated_token_ids.extend(&output.new_token_ids);
                if let Some(lps) = &output.new_logprobs {
                    req_state.logprobs.extend(lps.iter().cloned());
                }
                if let Some(plps) = &output.new_prompt_logprobs {
                    req_state.prompt_logprobs = Some(plps.clone());
                }

                // Update cached tokens.
                if output.num_cached_tokens > 0 {
                    req_state.num_cached_tokens = output.num_cached_tokens;
                }

                // Store pooler output if present.
                if output.pooler_output.is_some() {
                    req_state.pooler_output = output.pooler_output.clone();
                }

                // Take the detokenizer out for parallel processing.
                let stop_terminated = output.finish_reason == Some(FinishReason::Stop);
                req_state.detokenizer.take().map(|detok| {
                    (
                        output.new_token_ids.clone(),
                        stop_terminated,
                        output.finish_reason,
                        detok,
                    )
                })
            })
            .collect()
    }

    /// Phase 2: No lock — run detokenization in parallel using Rayon.
    ///
    /// Each work item runs `detok.update()` + `detok.get_next_output_text()`
    /// concurrently across a thread pool, removing tokenizer decode calls
    /// from the critical path.
    fn parallel_detokenize(work: Vec<Option<DetokWork>>) -> Vec<Option<DetokResult>> {
        work.into_par_iter()
            .map(|item| {
                item.map(
                    |(new_token_ids, stop_terminated, finish_reason, mut detok)| {
                        let detok_stop = detok.update(&new_token_ids, stop_terminated);
                        let is_finished = finish_reason.is_some() || detok_stop.is_some();
                        let delta_text = detok.get_next_output_text(is_finished, true);
                        (detok, detok_stop, Some(delta_text))
                    },
                )
            })
            .collect()
    }

    /// Phase 3: Under lock — put detokenizers back, apply detok results,
    /// send streaming deltas, handle finish/cleanup.
    fn process_outputs_phase3(
        requests: &mut HashMap<String, RequestState>,
        outputs: Vec<EngineCoreOutput>,
        detok_results: Vec<Option<DetokResult>>,
    ) {
        for (output, detok_result) in outputs.into_iter().zip(detok_results) {
            let Some(req_state) = requests.get_mut(&output.request_id) else {
                // Already logged in Phase 1.
                continue;
            };

            // Put the detokenizer back (if we took it) and extract results.
            let (detokenizer_stop, delta_text) = match detok_result {
                Some((detok, stop, text)) => {
                    req_state.detokenizer = Some(detok);
                    (stop, text)
                }
                None => (None, None),
            };

            let step_logprobs = output.new_logprobs.clone();

            // Determine finish/stop reason. Detokenizer stop takes priority.
            let is_finished = output.finish_reason.is_some() || detokenizer_stop.is_some();
            let (delta_finish_reason, delta_stop_reason) = if let Some(stop_str) = detokenizer_stop
            {
                (Some(FinishReason::Stop), Some(StopReason::String(stop_str)))
            } else {
                (output.finish_reason, output.stop_reason.clone())
            };

            // Send streaming delta if applicable.
            if let Some(tx) = &req_state.stream_tx {
                // If tool parser state is active, route through it.
                if let Some(ref mut parser_state) = req_state.tool_parser_state {
                    if let Some(ref text) = delta_text {
                        let previous_text = req_state.accumulated_text.clone();
                        req_state.accumulated_text.push_str(text);
                        let current_text = req_state.accumulated_text.clone();

                        let parser_result =
                            parser_state.process_delta(&previous_text, &current_text, text);

                        match parser_result {
                            ToolParserDelta::Content(content) => {
                                let delta = StreamDelta {
                                    index: req_state.choice_index,
                                    new_token_ids: output.new_token_ids.clone(),
                                    text: Some(content),
                                    finish_reason: if is_finished && !req_state.tool_calls_emitted {
                                        delta_finish_reason
                                    } else {
                                        None
                                    },
                                    stop_reason: if is_finished && !req_state.tool_calls_emitted {
                                        delta_stop_reason.clone()
                                    } else {
                                        None
                                    },
                                    logprobs: step_logprobs.clone(),
                                    tool_call_deltas: None,
                                };
                                let _ = tx.send(delta);
                            }
                            ToolParserDelta::ToolCalls(tool_deltas) => {
                                // Filter by forced function name if specified.
                                let tool_deltas =
                                    if let Some(ref forced) = req_state.forced_function_name {
                                        tool_deltas
                                            .into_iter()
                                            .filter(|d| {
                                                d.function_name.as_ref().is_none_or(|n| n == forced)
                                            })
                                            .collect::<Vec<_>>()
                                    } else {
                                        tool_deltas
                                    };
                                if tool_deltas.is_empty() {
                                    // Filtered out — don't emit.
                                } else {
                                    req_state.tool_calls_emitted = true;
                                    let delta = StreamDelta {
                                        index: req_state.choice_index,
                                        new_token_ids: output.new_token_ids.clone(),
                                        text: None,
                                        finish_reason: None,
                                        stop_reason: None,
                                        logprobs: step_logprobs.clone(),
                                        tool_call_deltas: Some(tool_deltas),
                                    };
                                    let _ = tx.send(delta);
                                }
                            }
                            ToolParserDelta::None => {
                                // Buffering, don't send anything yet.
                            }
                        }

                        // Send finish delta separately if finished and tool calls were emitted.
                        if is_finished && req_state.tool_calls_emitted {
                            let finish_delta = StreamDelta {
                                index: req_state.choice_index,
                                new_token_ids: vec![],
                                text: None,
                                finish_reason: Some(FinishReason::Stop),
                                stop_reason: delta_stop_reason.clone(),
                                logprobs: None,
                                tool_call_deltas: None,
                            };
                            let _ = tx.send(finish_delta);
                        }
                    } else if is_finished {
                        // No text but finished — send finish delta.
                        let fr = if req_state.tool_calls_emitted {
                            Some(FinishReason::Stop)
                        } else {
                            delta_finish_reason
                        };
                        let delta = StreamDelta {
                            index: req_state.choice_index,
                            new_token_ids: output.new_token_ids.clone(),
                            text: None,
                            finish_reason: fr,
                            stop_reason: delta_stop_reason.clone(),
                            logprobs: step_logprobs.clone(),
                            tool_call_deltas: None,
                        };
                        let _ = tx.send(delta);
                    }
                } else {
                    // No tool parsing — normal streaming path.
                    let delta = StreamDelta {
                        index: req_state.choice_index,
                        new_token_ids: output.new_token_ids,
                        text: delta_text,
                        finish_reason: delta_finish_reason,
                        stop_reason: delta_stop_reason.clone(),
                        logprobs: step_logprobs.clone(),
                        tool_call_deltas: None,
                    };
                    let _ = tx.send(delta);
                }
            }

            // Update request state finish/stop reason.
            if is_finished && req_state.finish_reason.is_none() {
                req_state.finish_reason = delta_finish_reason;
                req_state.stop_reason = delta_stop_reason.or(output.stop_reason);
            }

            // Log and clean up when done.
            if is_finished {
                let now = Instant::now();
                let total_latency = now.duration_since(req_state.submit_time).as_secs_f64();
                let ttft_ms = req_state
                    .first_token_time
                    .map(|t| t.duration_since(req_state.submit_time).as_secs_f64() * 1000.0);
                let avg_itl_ms = if req_state.itl_count > 0 {
                    Some(req_state.itl_sum / req_state.itl_count as f64 * 1000.0)
                } else {
                    None
                };
                let prompt_tokens = req_state.num_prompt_tokens;
                let completion_tokens = req_state.generated_token_ids.len() as u32;

                let finish_str = req_state
                    .finish_reason
                    .map(|r| r.to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                let stop_str = req_state
                    .stop_reason
                    .as_ref()
                    .map(|sr| match sr {
                        StopReason::Token(id) => format!("token:{id}"),
                        StopReason::String(s) => format!("string:{s}"),
                    })
                    .unwrap_or_default();

                tracing::info!(
                    request_id = %output.request_id,
                    prompt_tokens = prompt_tokens,
                    completion_tokens = completion_tokens,
                    finish_reason = finish_str,
                    stop_reason = stop_str,
                    latency_ms = format!("{:.1}", total_latency * 1000.0),
                    ttft_ms = ttft_ms.map(|v| format!("{v:.1}")).unwrap_or_else(|| "-".into()),
                    avg_itl_ms = avg_itl_ms.map(|v| format!("{v:.1}")).unwrap_or_else(|| "-".into()),
                    "request finished"
                );

                #[cfg(feature = "metrics")]
                {
                    let metrics = crate::metrics::VllmMetrics::global();
                    metrics.request_latency_seconds.observe(total_latency);
                    metrics.requests_active.dec();
                    metrics.requests_success_total.inc();
                }
                // Drop stream sender; capture whether this was a streaming request.
                let was_streaming = req_state.stream_tx.take().is_some();

                // Streaming requests are cleaned up here — there's no poll_until_done
                // consumer. Non-streaming requests stay for poll_until_done to remove.
                if was_streaming {
                    requests.remove(&output.request_id);
                }
            }
        }
    }

    /// Process an engine output for a single request (single-threaded path,
    /// used by tests).
    #[cfg(test)]
    fn process_output(requests: &mut HashMap<String, RequestState>, output: EngineCoreOutput) {
        let Some(req_state) = requests.get_mut(&output.request_id) else {
            debug!("Output for unknown request {}, ignoring", output.request_id);
            return;
        };

        // Track output tokens.
        #[cfg(feature = "metrics")]
        {
            let metrics = crate::metrics::VllmMetrics::global();
            metrics
                .output_tokens_total
                .inc_by(output.new_token_ids.len() as u64);
        }

        // --- TTFT / ITL timing ---
        let now = Instant::now();
        if !output.new_token_ids.is_empty() {
            if req_state.first_token_time.is_none() {
                // First token: record TTFT.
                req_state.first_token_time = Some(now);
                #[cfg(feature = "metrics")]
                {
                    let ttft = now.duration_since(req_state.submit_time).as_secs_f64();
                    crate::metrics::VllmMetrics::global()
                        .time_to_first_token_seconds
                        .observe(ttft);
                }
            } else if let Some(last) = req_state.last_token_time {
                // Subsequent token: record inter-token latency.
                let itl = now.duration_since(last).as_secs_f64();
                #[cfg(feature = "metrics")]
                crate::metrics::VllmMetrics::global()
                    .inter_token_latency_seconds
                    .observe(itl);
                req_state.itl_count += 1;
                req_state.itl_sum += itl;
            }
            req_state.last_token_time = Some(now);
        }

        // Accumulate tokens and logprobs.
        req_state.generated_token_ids.extend(&output.new_token_ids);
        if let Some(lps) = &output.new_logprobs {
            req_state.logprobs.extend(lps.iter().cloned());
        }
        if let Some(plps) = output.new_prompt_logprobs {
            req_state.prompt_logprobs = Some(plps);
        }
        let step_logprobs = output.new_logprobs.clone();

        // Update cached tokens.
        if output.num_cached_tokens > 0 {
            req_state.num_cached_tokens = output.num_cached_tokens;
        }

        // Store pooler output (embedding vector) if present.
        if output.pooler_output.is_some() {
            req_state.pooler_output = output.pooler_output;
        }

        // Determine if the engine already terminated (stop token).
        let stop_terminated = output.finish_reason == Some(FinishReason::Stop);

        // Update the detokenizer with new tokens.
        let mut detokenizer_stop = None;
        if let Some(detok) = &mut req_state.detokenizer {
            detokenizer_stop = detok.update(&output.new_token_ids, stop_terminated);
        }

        // Determine finish/stop reason. Detokenizer stop takes priority.
        let is_finished = output.finish_reason.is_some() || detokenizer_stop.is_some();
        let (delta_finish_reason, delta_stop_reason) = if let Some(stop_str) = detokenizer_stop {
            (Some(FinishReason::Stop), Some(StopReason::String(stop_str)))
        } else {
            (output.finish_reason, output.stop_reason.clone())
        };

        // Get delta text for streaming.
        let delta_text = req_state
            .detokenizer
            .as_mut()
            .map(|detok| detok.get_next_output_text(is_finished, true));

        // Send streaming delta if applicable.
        if let Some(tx) = &req_state.stream_tx {
            // If tool parser state is active, route through it.
            if let Some(ref mut parser_state) = req_state.tool_parser_state {
                if let Some(ref text) = delta_text {
                    let previous_text = req_state.accumulated_text.clone();
                    req_state.accumulated_text.push_str(text);
                    let current_text = req_state.accumulated_text.clone();

                    let parser_result =
                        parser_state.process_delta(&previous_text, &current_text, text);

                    match parser_result {
                        ToolParserDelta::Content(content) => {
                            let delta = StreamDelta {
                                index: req_state.choice_index,
                                new_token_ids: output.new_token_ids.clone(),
                                text: Some(content),
                                finish_reason: if is_finished && !req_state.tool_calls_emitted {
                                    delta_finish_reason
                                } else {
                                    None
                                },
                                stop_reason: if is_finished && !req_state.tool_calls_emitted {
                                    delta_stop_reason.clone()
                                } else {
                                    None
                                },
                                logprobs: step_logprobs.clone(),
                                tool_call_deltas: None,
                            };
                            let _ = tx.send(delta);
                        }
                        ToolParserDelta::ToolCalls(tool_deltas) => {
                            // Filter by forced function name if specified.
                            let tool_deltas =
                                if let Some(ref forced) = req_state.forced_function_name {
                                    tool_deltas
                                        .into_iter()
                                        .filter(|d| {
                                            d.function_name.as_ref().is_none_or(|n| n == forced)
                                        })
                                        .collect::<Vec<_>>()
                                } else {
                                    tool_deltas
                                };
                            if tool_deltas.is_empty() {
                                // Filtered out — don't emit.
                            } else {
                                req_state.tool_calls_emitted = true;
                                let delta = StreamDelta {
                                    index: req_state.choice_index,
                                    new_token_ids: output.new_token_ids.clone(),
                                    text: None,
                                    finish_reason: None,
                                    stop_reason: None,
                                    logprobs: step_logprobs.clone(),
                                    tool_call_deltas: Some(tool_deltas),
                                };
                                let _ = tx.send(delta);
                            }
                        }
                        ToolParserDelta::None => {
                            // Buffering, don't send anything yet.
                        }
                    }

                    // Send finish delta separately if finished and tool calls were emitted.
                    if is_finished && req_state.tool_calls_emitted {
                        let finish_delta = StreamDelta {
                            index: req_state.choice_index,
                            new_token_ids: vec![],
                            text: None,
                            finish_reason: Some(FinishReason::Stop), // overridden to tool_calls in server.rs
                            stop_reason: delta_stop_reason.clone(),
                            logprobs: None,
                            tool_call_deltas: None,
                        };
                        let _ = tx.send(finish_delta);
                    }
                } else if is_finished {
                    // No text but finished — send finish delta.
                    let fr = if req_state.tool_calls_emitted {
                        Some(FinishReason::Stop)
                    } else {
                        delta_finish_reason
                    };
                    let delta = StreamDelta {
                        index: req_state.choice_index,
                        new_token_ids: output.new_token_ids.clone(),
                        text: None,
                        finish_reason: fr,
                        stop_reason: delta_stop_reason.clone(),
                        logprobs: step_logprobs.clone(),
                        tool_call_deltas: None,
                    };
                    let _ = tx.send(delta);
                }
            } else {
                // No tool parsing — normal streaming path.
                let delta = StreamDelta {
                    index: req_state.choice_index,
                    new_token_ids: output.new_token_ids,
                    text: delta_text,
                    finish_reason: delta_finish_reason,
                    stop_reason: delta_stop_reason.clone(),
                    logprobs: step_logprobs.clone(),
                    tool_call_deltas: None,
                };
                let _ = tx.send(delta);
            }
        }

        // Update request state finish/stop reason.
        if is_finished && req_state.finish_reason.is_none() {
            req_state.finish_reason = delta_finish_reason;
            req_state.stop_reason = delta_stop_reason.or(output.stop_reason);
        }

        // Log and clean up when done.
        if is_finished {
            let total_latency = now.duration_since(req_state.submit_time).as_secs_f64();
            let ttft_ms = req_state
                .first_token_time
                .map(|t| t.duration_since(req_state.submit_time).as_secs_f64() * 1000.0);
            let avg_itl_ms = if req_state.itl_count > 0 {
                Some(req_state.itl_sum / req_state.itl_count as f64 * 1000.0)
            } else {
                None
            };
            let prompt_tokens = req_state.num_prompt_tokens;
            let completion_tokens = req_state.generated_token_ids.len() as u32;

            let finish_str = req_state
                .finish_reason
                .map(|r| r.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let stop_str = req_state
                .stop_reason
                .as_ref()
                .map(|sr| match sr {
                    StopReason::Token(id) => format!("token:{id}"),
                    StopReason::String(s) => format!("string:{s}"),
                })
                .unwrap_or_default();

            tracing::info!(
                request_id = %output.request_id,
                prompt_tokens = prompt_tokens,
                completion_tokens = completion_tokens,
                finish_reason = finish_str,
                stop_reason = stop_str,
                latency_ms = format!("{:.1}", total_latency * 1000.0),
                ttft_ms = ttft_ms.map(|v| format!("{v:.1}")).unwrap_or_else(|| "-".into()),
                avg_itl_ms = avg_itl_ms.map(|v| format!("{v:.1}")).unwrap_or_else(|| "-".into()),
                "request finished"
            );

            #[cfg(feature = "metrics")]
            {
                let metrics = crate::metrics::VllmMetrics::global();
                metrics.request_latency_seconds.observe(total_latency);
                metrics.requests_active.dec();
                metrics.requests_success_total.inc();
            }
            // Drop stream sender; capture whether this was a streaming request.
            let was_streaming = req_state.stream_tx.take().is_some();

            // Streaming requests are cleaned up here — there's no poll_until_done
            // consumer. Non-streaming requests stay for poll_until_done to remove.
            if was_streaming {
                requests.remove(&output.request_id);
            }
        }
    }

    /// Convert a chat completion request to an engine core request.
    fn chat_to_engine_request(
        &self,
        request_id: &str,
        request: &protocol::ChatCompletionRequest,
        sampling_params: &SamplingParams,
    ) -> ServeResult<EngineCoreRequest> {
        // Build text from chat messages — using chat template if available.
        #[cfg(feature = "chat-template")]
        let text = if let Some(template) = &self.chat_template {
            // Convert messages to JSON values so templates can access all fields
            // (tool_calls, tool_call_id, name, etc.).
            let message_values: Vec<serde_json::Value> = request
                .messages
                .iter()
                .map(|msg| {
                    let mut val = serde_json::to_value(msg).unwrap_or_default();
                    // For tool_calls where function.arguments is a JSON string,
                    // parse it into a JSON object so templates using `| items` work.
                    if let Some(tool_calls) = val.get_mut("tool_calls")
                        && let Some(arr) = tool_calls.as_array_mut()
                    {
                        for tc in arr.iter_mut() {
                            if let Some(func) = tc.get_mut("function")
                                && let Some(args) = func.get("arguments")
                                && let Some(args_str) = args.as_str()
                                && let Ok(parsed) =
                                    serde_json::from_str::<serde_json::Value>(args_str)
                            {
                                func.as_object_mut()
                                    .unwrap()
                                    .insert("arguments".to_string(), parsed);
                            }
                        }
                    }
                    val
                })
                .collect();

            // Convert tools to JSON, respecting tool_choice.
            let tools_value = match &request.tool_choice {
                Some(tc) if tc.as_str() == Some("none") => None,
                _ => request
                    .tools
                    .as_ref()
                    .and_then(|t| serde_json::to_value(t).ok()),
            };

            template.apply(&message_values, true, tools_value.as_ref())?
        } else {
            // Fallback: concatenate messages with newlines.
            let mut text = String::new();
            for msg in &request.messages {
                if let Some(content) = &msg.content
                    && let Some(s) = content.as_str()
                {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(s);
                }
            }
            text
        };
        #[cfg(not(feature = "chat-template"))]
        let text = {
            // Fallback: concatenate messages with newlines.
            let mut text = String::new();
            for msg in &request.messages {
                if let Some(content) = &msg.content
                    && let Some(s) = content.as_str()
                {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(s);
                }
            }
            text
        };

        // Tokenize the text, or fall back to byte-value IDs.
        #[allow(unused_mut)]
        let mut token_ids = if let Some(tok) = &self.tokenizer {
            if text.is_empty() {
                vec![]
            } else {
                tok.encode(&text, true)?
            }
        } else if text.is_empty() {
            vec![0] // BOS placeholder
        } else {
            text.as_bytes().iter().map(|&b| b as u32).collect()
        };

        // Extract images from message content arrays and build multimodal data.
        #[cfg(feature = "multimodal")]
        let mm_data = if self.image_token_id.is_some() {
            self.extract_images_from_messages(&request.messages, &mut token_ids)?
        } else {
            None
        };
        #[cfg(not(feature = "multimodal"))]
        let mm_data = None;

        Ok(EngineCoreRequest {
            request_id: request_id.to_string(),
            prompt_token_ids: Some(token_ids),
            sampling_params: Some(sampling_params.clone()),
            arrival_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
            client_index: 0,
            priority: request.priority,
            cache_salt: request.cache_salt.clone(),
            data_parallel_rank: None,
            is_pooling: false,
            mm_data,
        })
    }

    /// Normalize a completion prompt to a list of token ID sequences.
    fn tokenize_completion_prompts(
        &self,
        request: &protocol::CompletionRequest,
    ) -> ServeResult<Vec<Vec<u32>>> {
        match &request.prompt {
            Some(protocol::CompletionPrompt::Single(text)) => {
                Ok(vec![self.tokenize_text(text, false)?])
            }
            Some(protocol::CompletionPrompt::Multiple(texts)) => {
                texts.iter().map(|t| self.tokenize_text(t, false)).collect()
            }
            Some(protocol::CompletionPrompt::TokenIds(ids)) => Ok(vec![ids.clone()]),
            Some(protocol::CompletionPrompt::MultipleTokenIds(vv)) => Ok(vv.clone()),
            None => Ok(vec![vec![0]]),
        }
    }

    /// Tokenize a text string, using the tokenizer if available.
    fn tokenize_text(&self, text: &str, add_special: bool) -> ServeResult<Vec<u32>> {
        if let Some(tok) = &self.tokenizer {
            if text.is_empty() {
                Ok(vec![])
            } else {
                Ok(tok.encode(text, add_special)?)
            }
        } else if text.is_empty() {
            Ok(vec![0])
        } else {
            Ok(text.as_bytes().iter().map(|&b| b as u32).collect())
        }
    }

    /// Convert a completion request to an engine core request.
    #[cfg(test)]
    fn completion_to_engine_request(
        &self,
        request_id: &str,
        request: &protocol::CompletionRequest,
        sampling_params: &SamplingParams,
    ) -> ServeResult<EngineCoreRequest> {
        let token_ids = match &request.prompt {
            Some(protocol::CompletionPrompt::Single(text)) => {
                if let Some(tok) = &self.tokenizer {
                    tok.encode(text, false)?
                } else {
                    text.as_bytes().iter().map(|&b| b as u32).collect()
                }
            }
            Some(protocol::CompletionPrompt::TokenIds(ids)) => ids.clone(),
            Some(protocol::CompletionPrompt::Multiple(texts)) => {
                let text = texts.first().map(|t| t.as_str()).unwrap_or("");
                if let Some(tok) = &self.tokenizer {
                    tok.encode(text, false)?
                } else {
                    text.as_bytes().iter().map(|&b| b as u32).collect()
                }
            }
            Some(protocol::CompletionPrompt::MultipleTokenIds(id_lists)) => {
                id_lists.first().cloned().unwrap_or_else(|| vec![0])
            }
            None => vec![0],
        };

        Ok(EngineCoreRequest {
            request_id: request_id.to_string(),
            prompt_token_ids: Some(token_ids),
            sampling_params: Some(sampling_params.clone()),
            arrival_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
            client_index: 0,
            priority: request.priority,
            cache_salt: request.cache_salt.clone(),
            data_parallel_rank: None,
            is_pooling: false,
            mm_data: None,
        })
    }

    /// Extract image URLs from chat message content arrays, decode/preprocess
    /// images, expand image placeholder tokens, and build `MultimodalData`.
    ///
    /// Returns `None` if no images are present.
    #[cfg(feature = "multimodal")]
    fn extract_images_from_messages(
        &self,
        messages: &[protocol::ChatCompletionMessageParam],
        token_ids: &mut Vec<u32>,
    ) -> ServeResult<Option<MultimodalData>> {
        let image_token_id = match self.image_token_id {
            Some(id) => id,
            None => return Ok(None),
        };
        let image_size = self.mm_image_size;
        if image_size == 0 {
            return Ok(None);
        }

        let mut images: Vec<ImageData> = Vec::new();

        // Walk messages looking for content arrays with image_url parts.
        for msg in messages {
            if let Some(content) = &msg.content
                && let Some(parts) = content.as_array()
            {
                for part in parts {
                    if part.get("type").and_then(|t| t.as_str()) == Some("image_url")
                        && let Some(image_url_obj) = part.get("image_url")
                    {
                        let url = image_url_obj
                            .get("url")
                            .and_then(|u| u.as_str())
                            .unwrap_or("");
                        if url.is_empty() {
                            continue;
                        }

                        // Decode the image.
                        let img_bytes = if url.starts_with("data:") {
                            vllm_model::image::decode_data_uri(url).map_err(|e| {
                                ServeError::Validation(format!("failed to decode data URI: {e}"))
                            })?
                        } else {
                            // For now, only data URIs are supported.
                            // HTTP URL download would require async reqwest.
                            return Err(ServeError::Validation(
                                "HTTP image URLs not yet supported; use base64 data URIs".into(),
                            ));
                        };

                        let dyn_image =
                            vllm_model::image::decode_image(&img_bytes).map_err(|e| {
                                ServeError::Validation(format!("failed to decode image: {e}"))
                            })?;

                        let image_data = if self.mm_model_type == "qwen2_vl" {
                            // Qwen2-VL: smart_resize to target dimensions, CLIP normalization.
                            // Factor = patch_size(14) * spatial_merge_size(2) = 28.
                            let factor = 28;
                            let (target_h, target_w) = vllm_model::image::smart_resize(
                                dyn_image.height() as usize,
                                dyn_image.width() as usize,
                                factor,
                                256 * 28 * 28,  // min_pixels
                                1280 * 28 * 28, // max_pixels
                            );
                            vllm_model::image::preprocess_qwen2_vl(&dyn_image, target_h, target_w)
                        } else {
                            vllm_model::image::preprocess_siglip(&dyn_image, image_size)
                        };
                        images.push(image_data);
                    }
                }
            }
        }

        if images.is_empty() {
            return Ok(None);
        }

        // Expand image placeholders in token IDs.
        let placeholders = vllm_model::image::expand_image_placeholders(
            token_ids,
            image_token_id,
            self.mm_tokens_per_image,
        );

        if placeholders.len() != images.len() {
            debug!(
                "Image count mismatch: {} images but {} placeholders in token sequence",
                images.len(),
                placeholders.len()
            );
        }

        Ok(Some(MultimodalData {
            images,
            image_placeholders: placeholders,
        }))
    }

    /// Resolve `max_tokens` to fit within `max_model_len - prompt_len`.
    ///
    /// - If `None`, sets it to the remaining capacity.
    /// - If `Some(val)`, caps it at the remaining capacity.
    fn resolve_max_tokens(&self, sampling_params: &mut SamplingParams, num_prompt_tokens: usize) {
        let remaining = self.max_model_len.saturating_sub(num_prompt_tokens);
        match sampling_params.max_tokens {
            None => sampling_params.max_tokens = Some(remaining as u32),
            Some(val) => sampling_params.max_tokens = Some(val.min(remaining as u32)),
        }
    }

    /// Build SamplingParams from a chat completion request.
    fn build_sampling_params_from_chat(
        &self,
        request: &protocol::ChatCompletionRequest,
    ) -> ServeResult<SamplingParams> {
        let stop = match &request.stop {
            Some(protocol::StopCondition::Single(s)) => vec![s.clone()],
            Some(protocol::StopCondition::Multiple(v)) => v.clone(),
            None => vec![],
        };
        // Resolve max_tokens: max_completion_tokens takes priority over max_tokens.
        let max_tokens = request.max_completion_tokens.or(request.max_tokens);

        // Chat API: logprobs is bool, top_logprobs is count.
        let logprobs = if request.logprobs == Some(true) {
            Some(request.top_logprobs.unwrap_or(0) as i32)
        } else {
            None
        };

        // Parse response_format → guided_grammar, or guided_regex (mutually exclusive).
        let guided_grammar =
            resolve_guided_grammar(&request.response_format, &request.guided_regex)?;

        // Tokenize bad_words strings into token sequences.
        let bad_words_token_ids = self.tokenize_bad_words(&request.bad_words)?;

        Ok(SamplingParams {
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(0),
            min_p: request.min_p.unwrap_or(0.0),
            max_tokens,
            repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            seed: request.seed.map(|s| s as u64),
            ignore_eos: request.ignore_eos,
            min_tokens: request.min_tokens,
            stop,
            stop_token_ids: request.stop_token_ids.clone(),
            include_stop_str_in_output: request.include_stop_str_in_output,
            skip_special_tokens: request.skip_special_tokens,
            logprobs,
            prompt_logprobs: request.prompt_logprobs.map(|n| n as i32),
            logit_bias: parse_logit_bias(&request.logit_bias),
            guided_grammar,
            allowed_token_ids: request.allowed_token_ids.clone(),
            bad_words_token_ids,
            ..Default::default()
        })
    }

    /// Build SamplingParams from a completion request.
    fn build_sampling_params_from_completion(
        &self,
        request: &protocol::CompletionRequest,
    ) -> ServeResult<SamplingParams> {
        let stop = match &request.stop {
            Some(protocol::StopCondition::Single(s)) => vec![s.clone()],
            Some(protocol::StopCondition::Multiple(v)) => v.clone(),
            None => vec![],
        };
        // Completion API: logprobs is directly the count.
        let logprobs = request.logprobs.map(|n| n as i32);

        // guided_regex → guided_grammar (completions have no response_format).
        let guided_grammar = request
            .guided_regex
            .as_ref()
            .map(|pattern| GuidedGrammar::Regex {
                pattern: pattern.clone(),
            });

        // Tokenize bad_words strings into token sequences.
        let bad_words_token_ids = self.tokenize_bad_words(&request.bad_words)?;

        Ok(SamplingParams {
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(0),
            min_p: request.min_p.unwrap_or(0.0),
            max_tokens: request.max_tokens,
            repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            seed: request.seed.map(|s| s as u64),
            ignore_eos: request.ignore_eos,
            min_tokens: request.min_tokens,
            stop,
            stop_token_ids: request.stop_token_ids.clone(),
            include_stop_str_in_output: request.include_stop_str_in_output,
            skip_special_tokens: request.skip_special_tokens,
            logprobs,
            prompt_logprobs: request.prompt_logprobs.map(|n| n as i32),
            logit_bias: parse_logit_bias(&request.logit_bias),
            guided_grammar,
            allowed_token_ids: request.allowed_token_ids.clone(),
            bad_words_token_ids,
            ..Default::default()
        })
    }

    /// Tokenize bad_words strings into token ID sequences.
    fn tokenize_bad_words(
        &self,
        bad_words: &Option<Vec<String>>,
    ) -> ServeResult<Option<Vec<Vec<u32>>>> {
        let Some(words) = bad_words else {
            return Ok(None);
        };
        if words.is_empty() {
            return Ok(None);
        }
        let Some(ref tokenizer) = self.tokenizer else {
            return Ok(None);
        };
        let mut result = Vec::with_capacity(words.len());
        for word in words {
            let ids = tokenizer.encode(word, false)?;
            if !ids.is_empty() {
                result.push(ids);
            }
        }
        if result.is_empty() {
            Ok(None)
        } else {
            Ok(Some(result))
        }
    }
}

// ---------------------------------------------------------------------------
// Async scheduling types and executor thread
// ---------------------------------------------------------------------------

use vllm_engine::error::EngineResult;

/// Work item sent from the step loop to the executor thread.
enum ExecutorWork {
    /// Execute a model forward pass.
    Execute(Box<vllm_core::scheduler::output::SchedulerOutput>),
    /// Compute embeddings (reply goes back via the oneshot).
    Embed(
        Vec<Vec<u32>>,
        tokio::sync::oneshot::Sender<ServeResult<Vec<Vec<f32>>>>,
    ),
    /// Shut down the executor.
    Shutdown,
}

/// Result sent back from the executor thread to the step loop.
enum ExecutorResult {
    /// Model forward pass result, together with the scheduler output that
    /// produced it (returned so the step loop can finalize without cloning).
    Model(
        EngineResult<ModelRunnerOutput>,
        Box<vllm_core::scheduler::output::SchedulerOutput>,
    ),
}

/// The executor thread's main loop.
///
/// Receives work items from the step loop, executes them, and sends results
/// back. Runs on a dedicated OS thread so GPU work doesn't block tokio.
fn executor_thread_loop(
    mut executor: Box<dyn Executor>,
    mut rx: tokio::sync::mpsc::Receiver<ExecutorWork>,
    tx: tokio::sync::mpsc::Sender<ExecutorResult>,
) {
    while let Some(work) = rx.blocking_recv() {
        match work {
            ExecutorWork::Execute(sched) => {
                let result = executor.execute_model(&sched);
                // Send the sched back so the step loop can finalize without cloning.
                if tx
                    .blocking_send(ExecutorResult::Model(result, sched))
                    .is_err()
                {
                    break; // Step loop dropped its receiver.
                }
            }
            ExecutorWork::Embed(seqs, reply) => {
                let result = executor
                    .embed(seqs)
                    .map_err(|e| ServeError::Engine(e.to_string()));
                let _ = reply.send(result);
            }
            ExecutorWork::Shutdown => {
                executor.shutdown();
                break;
            }
        }
    }
}

/// Route step outputs to request states and update metrics.
/// Returns `true` if any output tokens were produced.
async fn route_step_outputs(
    requests: &Mutex<HashMap<String, RequestState>>,
    outputs: vllm_engine::engine_core::StepOutputs,
) -> bool {
    let mut had_outputs = false;
    for (_, engine_outputs) in outputs {
        // Update scheduler gauges from stats.
        #[cfg(feature = "metrics")]
        if let Some(stats) = &engine_outputs.scheduler_stats {
            let m = crate::metrics::VllmMetrics::global();
            m.num_requests_running.set(stats.num_running_reqs as f64);
            m.num_requests_waiting.set(stats.num_waiting_reqs as f64);
            m.kv_cache_usage_perc.set(stats.kv_cache_usage);
            m.gpu_cache_blocks_used
                .set(stats.gpu_cache_blocks_used as i64);
            m.gpu_cache_blocks_total
                .set(stats.gpu_cache_blocks_total as i64);
            m.prefix_cache_blocks.set(stats.num_cached_blocks as i64);
        }

        if !engine_outputs.outputs.is_empty() {
            had_outputs = true;

            // Phase 1: accumulate tokens, take detokenizers (brief lock).
            let detok_work = {
                let mut reqs = requests.lock().await;
                AsyncEngine::process_outputs_phase1(&mut reqs, &engine_outputs.outputs)
            };

            // Phase 2: parallel detokenize (no lock held).
            let detok_results = AsyncEngine::parallel_detokenize(detok_work);

            // Phase 3: apply detok results, send streams (brief lock).
            {
                let mut reqs = requests.lock().await;
                AsyncEngine::process_outputs_phase3(
                    &mut reqs,
                    engine_outputs.outputs,
                    detok_results,
                );
            }
        }
    }
    had_outputs
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert API-style logit_bias (string keys, f64 values) to sampler format.
///
/// OpenAI API uses string token IDs as keys; we parse them to u32.
/// Invalid keys are silently ignored.
/// Truncate prompt token IDs from the left, keeping the last N tokens.
///
/// `truncate` values: `None` → no-op, `Some(-1)` → no-op (model max),
/// `Some(n)` where n >= 1 → keep last n tokens.
fn truncate_prompt(tokens: &mut Vec<u32>, truncate: Option<i64>) -> ServeResult<()> {
    let Some(n) = truncate else { return Ok(()) };
    if n == -1 {
        // -1 means "use model's max input length", which is already enforced
        // by resolve_max_tokens — nothing to do here.
        return Ok(());
    }
    if n < 1 {
        return Err(ServeError::Validation(format!(
            "truncate_prompt_tokens must be >= 1 or -1, got {n}"
        )));
    }
    let n = n as usize;
    if n < tokens.len() {
        let start = tokens.len() - n;
        tokens.drain(..start);
    }
    Ok(())
}

fn parse_logit_bias(
    api_bias: &Option<std::collections::HashMap<String, f64>>,
) -> Option<std::collections::HashMap<u32, f32>> {
    let bias = api_bias.as_ref()?;
    if bias.is_empty() {
        return None;
    }
    let parsed: std::collections::HashMap<u32, f32> = bias
        .iter()
        .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|tid| (tid, v as f32)))
        .collect();
    if parsed.is_empty() {
        None
    } else {
        Some(parsed)
    }
}

/// Parse `response_format` from an API request into a `GuidedGrammar`.
///
/// - `type: "text"` → `None` (no constraint)
/// - `type: "json_object"` → `Some(GuidedGrammar::Json)`
/// - `type: "json_schema"` → `Some(GuidedGrammar::JsonSchema { schema })`
fn parse_response_format(
    rf: &Option<protocol::ResponseFormat>,
) -> ServeResult<Option<GuidedGrammar>> {
    let Some(rf) = rf else {
        return Ok(None);
    };
    match rf.format_type.as_str() {
        "text" => Ok(None),
        "json_object" => Ok(Some(GuidedGrammar::Json)),
        "json_schema" => {
            let schema = rf
                .json_schema
                .as_ref()
                .and_then(|js| js.json_schema.clone())
                .ok_or_else(|| {
                    ServeError::Validation(
                        "response_format type 'json_schema' requires json_schema.schema".into(),
                    )
                })?;
            Ok(Some(GuidedGrammar::JsonSchema { schema }))
        }
        other => Err(ServeError::Validation(format!(
            "unsupported response_format type: {other:?}. Must be 'text', 'json_object', or 'json_schema'",
        ))),
    }
}

/// Resolve `response_format` and `guided_regex` into a single `GuidedGrammar`.
///
/// These are mutually exclusive — returns an error if both are set.
fn resolve_guided_grammar(
    response_format: &Option<protocol::ResponseFormat>,
    guided_regex: &Option<String>,
) -> ServeResult<Option<GuidedGrammar>> {
    let from_rf = parse_response_format(response_format)?;
    let from_regex = guided_regex.as_ref().map(|pattern| GuidedGrammar::Regex {
        pattern: pattern.clone(),
    });
    match (from_rf, from_regex) {
        (Some(_), Some(_)) => Err(ServeError::Validation(
            "response_format and guided_regex are mutually exclusive".into(),
        )),
        (Some(g), None) => Ok(Some(g)),
        (None, Some(g)) => Ok(Some(g)),
        (None, None) => Ok(None),
    }
}

/// Convert engine logprobs to chat completion logprobs format.
fn build_chat_logprobs(
    logprobs: &[vllm_common::LogprobsOutput],
    tokenizer: Option<&Tokenizer>,
) -> protocol::ChatCompletionLogProbs {
    let content: Vec<protocol::ChatCompletionLogProbsContent> = logprobs
        .iter()
        .map(|lp| {
            let token_str = tokenizer
                .and_then(|tok| tok.decode(&[lp.sampled.token_id], false).ok())
                .unwrap_or_else(|| format!("<token_{}>", lp.sampled.token_id));

            let top_logprobs: Vec<protocol::ChatCompletionLogProb> = lp
                .top_logprobs
                .iter()
                .map(|tlp| {
                    let t = tokenizer
                        .and_then(|tok| tok.decode(&[tlp.token_id], false).ok())
                        .unwrap_or_else(|| format!("<token_{}>", tlp.token_id));
                    protocol::ChatCompletionLogProb {
                        token: t,
                        logprob: tlp.logprob as f64,
                        bytes: None,
                    }
                })
                .collect();

            protocol::ChatCompletionLogProbsContent {
                token: token_str,
                logprob: lp.sampled.logprob as f64,
                bytes: None,
                top_logprobs,
            }
        })
        .collect();

    protocol::ChatCompletionLogProbs {
        content: Some(content),
    }
}

/// Convert engine logprobs to completion logprobs format.
fn build_completion_logprobs(
    logprobs: &[vllm_common::LogprobsOutput],
    tokenizer: Option<&Tokenizer>,
) -> protocol::CompletionLogProbs {
    let mut text_offset = Vec::with_capacity(logprobs.len());
    let mut token_logprobs = Vec::with_capacity(logprobs.len());
    let mut tokens = Vec::with_capacity(logprobs.len());
    let mut top_logprobs = Vec::with_capacity(logprobs.len());

    let mut offset = 0u32;
    for lp in logprobs {
        let token_str = tokenizer
            .and_then(|tok| tok.decode(&[lp.sampled.token_id], false).ok())
            .unwrap_or_else(|| format!("<token_{}>", lp.sampled.token_id));

        text_offset.push(offset);
        offset += token_str.len() as u32;

        token_logprobs.push(Some(lp.sampled.logprob as f64));
        tokens.push(token_str);

        if lp.top_logprobs.is_empty() {
            top_logprobs.push(None);
        } else {
            let top: std::collections::HashMap<String, f64> = lp
                .top_logprobs
                .iter()
                .map(|tlp| {
                    let t = tokenizer
                        .and_then(|tok| tok.decode(&[tlp.token_id], false).ok())
                        .unwrap_or_else(|| format!("<token_{}>", tlp.token_id));
                    (t, tlp.logprob as f64)
                })
                .collect();
            top_logprobs.push(Some(top));
        }
    }

    protocol::CompletionLogProbs {
        text_offset,
        token_logprobs,
        tokens,
        top_logprobs,
    }
}

/// Check if tool_choice is set to "none" (disabling tool use).
fn is_tool_choice_none(tool_choice: &Option<serde_json::Value>) -> bool {
    tool_choice.as_ref().and_then(|v| v.as_str()) == Some("none")
}

/// Extract the forced function name from `tool_choice`, if it specifies one.
///
/// OpenAI format: `{"type": "function", "function": {"name": "get_weather"}}`
fn get_tool_choice_function_name(tool_choice: &Option<serde_json::Value>) -> Option<String> {
    let val = tool_choice.as_ref()?;
    let obj = val.as_object()?;
    let func = obj.get("function")?.as_object()?;
    func.get("name")?.as_str().map(|s| s.to_string())
}

/// Generate placeholder text from token IDs (used when no tokenizer is available).
fn placeholder_text(token_ids: &[u32]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for id in token_ids {
        let _ = write!(s, "<token_{id}>");
    }
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::make_test_tokenizer;
    use vllm_config::{SchedulerConfig, SchedulerPolicy};
    use vllm_engine::core_client::InprocClient;
    use vllm_engine::engine_core::EngineCoreConfig;
    use vllm_engine::executor::NoopExecutor;
    use vllm_protocol::messages::PauseMode;

    fn make_test_request_state(
        stream_tx: Option<mpsc::UnboundedSender<StreamDelta>>,
    ) -> RequestState {
        RequestState {
            generated_token_ids: Vec::new(),
            num_prompt_tokens: 5,
            num_cached_tokens: 0,
            finish_reason: None,
            error: None,
            stop_reason: None,
            stream_tx,
            detokenizer: None,
            choice_index: 0,
            submit_time: Instant::now(),
            first_token_time: None,
            last_token_time: None,
            itl_count: 0,
            itl_sum: 0.0,
            logprobs: Vec::new(),
            prompt_logprobs: None,
            tool_parser_state: None,
            accumulated_text: String::new(),
            tool_calls_emitted: false,
            forced_function_name: None,
            pooler_output: None,
        }
    }

    fn make_engine_config() -> EngineCoreConfig {
        EngineCoreConfig {
            scheduler_config: SchedulerConfig {
                max_num_batched_tokens: 8192,
                max_num_seqs: 256,
                max_num_scheduled_tokens: None,
                policy: SchedulerPolicy::Fcfs,
                enable_chunked_prefill: true,
                long_prefill_token_threshold: 0,
                ..Default::default()
            },
            max_model_len: 4096,
            num_gpu_blocks: 1024,
            block_size: 16,
            engine_index: 0,
            async_scheduling: false,
            use_spec_decode: false,
            ngram_proposer_config: None,
            eos_token_ids: vec![],
            is_pooling: false,
            enable_prefix_caching: false,
        }
    }

    fn make_test_engine() -> AsyncEngine {
        let executor = Box::new(NoopExecutor::new(1024));
        let client = Box::new(InprocClient::new(make_engine_config(), executor));
        AsyncEngine::new(client, "test-model".to_string(), 4096)
    }

    fn make_test_engine_with_tokenizer() -> AsyncEngine {
        let executor = Box::new(NoopExecutor::new(1024));
        let client = Box::new(InprocClient::new(make_engine_config(), executor));
        let tok = Arc::new(make_test_tokenizer());
        AsyncEngine::with_tokenizer(client, "test-model".to_string(), 4096, tok)
    }

    fn make_chat_request() -> protocol::ChatCompletionRequest {
        protocol::ChatCompletionRequest {
            model: Some("test".to_string()),
            messages: vec![protocol::ChatCompletionMessageParam {
                role: "user".to_string(),
                content: Some(serde_json::Value::String("Hello".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: Some(0.7),
            top_p: Some(0.9),
            n: 1,
            max_tokens: Some(100),
            max_completion_tokens: None,
            stream: false,
            stream_options: None,
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            prompt_logprobs: None,
            seed: Some(42),
            response_format: None,
            tools: None,
            tool_choice: None,
            user: None,
            top_k: None,
            min_p: None,
            repetition_penalty: None,
            min_tokens: 0,
            stop_token_ids: vec![],
            include_stop_str_in_output: false,
            ignore_eos: false,
            skip_special_tokens: true,
            priority: 0,
            cache_salt: None,
            request_id: None,
            guided_regex: None,
            allowed_token_ids: None,
            bad_words: None,
            truncate_prompt_tokens: None,
        }
    }

    #[test]
    fn test_engine_new() {
        let engine = make_test_engine();
        assert_eq!(engine.model_name(), "test-model");
        assert_eq!(engine.max_model_len(), 4096);
        assert!(!engine.has_tokenizer());
    }

    #[test]
    fn test_engine_with_tokenizer() {
        let engine = make_test_engine_with_tokenizer();
        assert_eq!(engine.model_name(), "test-model");
        assert!(engine.has_tokenizer());
    }

    #[test]
    fn test_chat_to_engine_request_without_tokenizer() {
        let engine = make_test_engine();
        let request = make_chat_request();
        let params = engine.build_sampling_params_from_chat(&request).unwrap();
        let ec_req = engine
            .chat_to_engine_request("test-1", &request, &params)
            .unwrap();
        assert_eq!(ec_req.request_id, "test-1");
        assert!(ec_req.prompt_token_ids.is_some());
        assert!(!ec_req.prompt_token_ids.as_ref().unwrap().is_empty());
        // Without tokenizer, prompt is byte values of "Hello".
        let ids = ec_req.prompt_token_ids.unwrap();
        assert_eq!(ids.len(), 5); // b"Hello" = 5 bytes
    }

    #[test]
    fn test_chat_to_engine_request_with_tokenizer() {
        let engine = make_test_engine_with_tokenizer();
        let request = make_chat_request();
        let params = engine.build_sampling_params_from_chat(&request).unwrap();
        let ec_req = engine
            .chat_to_engine_request("test-1", &request, &params)
            .unwrap();
        assert_eq!(ec_req.request_id, "test-1");
        let ids = ec_req.prompt_token_ids.unwrap();
        assert!(!ids.is_empty());
        // With tokenizer, "Hello" is properly tokenized.

        let sp = ec_req.sampling_params.unwrap();
        assert!((sp.temperature - 0.7).abs() < 0.01);
        assert!((sp.top_p - 0.9).abs() < 0.01);
        assert_eq!(sp.max_tokens, Some(100));
        assert_eq!(sp.seed, Some(42));
    }

    #[test]
    fn test_completion_to_engine_request_token_ids() {
        let engine = make_test_engine();
        let request = protocol::CompletionRequest {
            model: None,
            prompt: Some(protocol::CompletionPrompt::TokenIds(vec![1, 2, 3, 4])),
            echo: false,
            temperature: None,
            top_p: None,
            n: 1,
            max_tokens: Some(50),
            stream: false,
            stream_options: None,
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            logit_bias: None,
            logprobs: None,
            prompt_logprobs: None,
            suffix: None,
            seed: None,
            user: None,
            top_k: None,
            min_p: None,
            repetition_penalty: None,
            min_tokens: 0,
            stop_token_ids: vec![],
            include_stop_str_in_output: false,
            ignore_eos: false,
            skip_special_tokens: true,
            priority: 0,
            cache_salt: None,
            request_id: None,
            guided_regex: None,
            allowed_token_ids: None,
            bad_words: None,
            truncate_prompt_tokens: None,
        };

        let params = engine
            .build_sampling_params_from_completion(&request)
            .unwrap();
        let ec_req = engine
            .completion_to_engine_request("comp-1", &request, &params)
            .unwrap();
        assert_eq!(ec_req.request_id, "comp-1");
        assert_eq!(ec_req.prompt_token_ids, Some(vec![1, 2, 3, 4]));
        assert_eq!(ec_req.sampling_params.unwrap().max_tokens, Some(50));
    }

    #[test]
    fn test_completion_to_engine_request_with_tokenizer() {
        let engine = make_test_engine_with_tokenizer();
        let request = protocol::CompletionRequest {
            model: None,
            prompt: Some(protocol::CompletionPrompt::Single(
                "Hello world".to_string(),
            )),
            echo: false,
            temperature: None,
            top_p: None,
            n: 1,
            max_tokens: Some(50),
            stream: false,
            stream_options: None,
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            logit_bias: None,
            logprobs: None,
            prompt_logprobs: None,
            suffix: None,
            seed: None,
            user: None,
            top_k: None,
            min_p: None,
            repetition_penalty: None,
            min_tokens: 0,
            stop_token_ids: vec![],
            include_stop_str_in_output: false,
            ignore_eos: false,
            skip_special_tokens: true,
            priority: 0,
            cache_salt: None,
            request_id: None,
            guided_regex: None,
            allowed_token_ids: None,
            bad_words: None,
            truncate_prompt_tokens: None,
        };

        let params = engine
            .build_sampling_params_from_completion(&request)
            .unwrap();
        let ec_req = engine
            .completion_to_engine_request("comp-1", &request, &params)
            .unwrap();
        let ids = ec_req.prompt_token_ids.unwrap();
        assert!(!ids.is_empty());
        // "Hello world" properly tokenized by byte-level BPE.
    }

    #[test]
    fn test_process_output_unknown_request() {
        let mut requests = HashMap::new();
        let output = EngineCoreOutput {
            request_id: "unknown".to_string(),
            new_token_ids: vec![1, 2, 3],
            finish_reason: None,
            stop_reason: None,
            num_cached_tokens: 0,
            events: None,
            new_logprobs: None,
            new_prompt_logprobs: None,
            pooler_output: None,
        };
        // Should not panic.
        AsyncEngine::process_output(&mut requests, output);
    }

    #[test]
    fn test_process_output_accumulates_tokens() {
        let mut requests = HashMap::new();
        requests.insert("req-1".to_string(), make_test_request_state(None));

        // First output.
        AsyncEngine::process_output(
            &mut requests,
            EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![10, 11],
                finish_reason: None,
                stop_reason: None,
                num_cached_tokens: 0,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: None,
            },
        );

        // Second output.
        AsyncEngine::process_output(
            &mut requests,
            EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![12],
                finish_reason: Some(FinishReason::Stop),
                stop_reason: Some(StopReason::Token(50256)),
                num_cached_tokens: 3,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: None,
            },
        );

        let state = requests.get("req-1").unwrap();
        assert_eq!(state.generated_token_ids, vec![10, 11, 12]);
        assert_eq!(state.finish_reason, Some(FinishReason::Stop));
        assert_eq!(state.stop_reason, Some(StopReason::Token(50256)));
        assert_eq!(state.num_cached_tokens, 3);
    }

    #[test]
    fn test_process_output_streaming() {
        let (tx, mut rx) = mpsc::unbounded_channel();

        let mut requests = HashMap::new();
        requests.insert("req-1".to_string(), make_test_request_state(Some(tx)));

        AsyncEngine::process_output(
            &mut requests,
            EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![10, 11],
                finish_reason: None,
                stop_reason: None,
                num_cached_tokens: 0,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: None,
            },
        );

        let delta = rx.try_recv().unwrap();
        assert_eq!(delta.new_token_ids, vec![10, 11]);
        assert!(delta.finish_reason.is_none());
        assert!(delta.text.is_none()); // No tokenizer = no text.

        // Finish.
        AsyncEngine::process_output(
            &mut requests,
            EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![12],
                finish_reason: Some(FinishReason::Length),
                stop_reason: None,
                num_cached_tokens: 0,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: None,
            },
        );

        let delta = rx.try_recv().unwrap();
        assert_eq!(delta.new_token_ids, vec![12]);
        assert_eq!(delta.finish_reason, Some(FinishReason::Length));

        // Finished streaming request should be removed from the map.
        assert!(requests.get("req-1").is_none());
    }

    #[test]
    fn test_process_output_with_detokenizer() {
        let tok = Arc::new(make_test_tokenizer());
        // Encode a prompt and some continuation.
        let prompt = "Hi";
        let prompt_ids = tok.encode(prompt, false).unwrap();
        let full_text = format!("{prompt} there");
        let full_ids = tok.encode(&full_text, false).unwrap();
        let output_ids = full_ids[prompt_ids.len()..].to_vec();

        let detok =
            IncrementalDetokenizer::new(Arc::clone(&tok), &prompt_ids, vec![], 0, false, false);

        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut requests = HashMap::new();
        {
            let mut state = make_test_request_state(Some(tx));
            state.num_prompt_tokens = prompt_ids.len() as u32;
            state.detokenizer = Some(detok);
            requests.insert("req-1".to_string(), state);
        }

        AsyncEngine::process_output(
            &mut requests,
            EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: output_ids,
                finish_reason: Some(FinishReason::Length),
                stop_reason: None,
                num_cached_tokens: 0,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: None,
            },
        );

        let delta = rx.try_recv().unwrap();
        // With detokenizer, we should get real text.
        assert!(delta.text.is_some());
        let text = delta.text.unwrap();
        assert!(
            text.contains("there"),
            "Expected 'there' in delta text, got: {text:?}"
        );
    }

    #[test]
    fn test_placeholder_text() {
        assert_eq!(placeholder_text(&[]), "");
        assert_eq!(placeholder_text(&[1, 2]), "<token_1><token_2>");
    }

    #[test]
    fn test_build_sampling_params_with_stop() {
        let engine = make_test_engine();
        let mut request = make_chat_request();
        request.stop = Some(protocol::StopCondition::Multiple(vec![
            "END".to_string(),
            "STOP".to_string(),
        ]));
        request.include_stop_str_in_output = true;

        let params = engine.build_sampling_params_from_chat(&request).unwrap();
        assert_eq!(params.stop, vec!["END", "STOP"]);
        assert!(params.include_stop_str_in_output);
    }

    fn make_completion_request(
        prompt: Option<protocol::CompletionPrompt>,
    ) -> protocol::CompletionRequest {
        protocol::CompletionRequest {
            model: None,
            prompt,
            echo: false,
            temperature: None,
            top_p: None,
            n: 1,
            max_tokens: Some(50),
            stream: false,
            stream_options: None,
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            logit_bias: None,
            logprobs: None,
            prompt_logprobs: None,
            suffix: None,
            seed: None,
            user: None,
            top_k: None,
            min_p: None,
            repetition_penalty: None,
            min_tokens: 0,
            stop_token_ids: vec![],
            include_stop_str_in_output: false,
            ignore_eos: false,
            skip_special_tokens: true,
            priority: 0,
            cache_salt: None,
            request_id: None,
            guided_regex: None,
            allowed_token_ids: None,
            bad_words: None,
            truncate_prompt_tokens: None,
        }
    }

    // -- n>1 and multi-prompt tests --

    #[test]
    fn test_truncate_prompt_none() {
        let mut tokens = vec![1, 2, 3, 4, 5];
        truncate_prompt(&mut tokens, None).unwrap();
        assert_eq!(tokens, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_truncate_prompt_minus_one() {
        let mut tokens = vec![1, 2, 3, 4, 5];
        truncate_prompt(&mut tokens, Some(-1)).unwrap();
        assert_eq!(tokens, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_truncate_prompt_keeps_last_n() {
        let mut tokens = vec![1, 2, 3, 4, 5];
        truncate_prompt(&mut tokens, Some(3)).unwrap();
        assert_eq!(tokens, vec![3, 4, 5]);
    }

    #[test]
    fn test_truncate_prompt_larger_than_len() {
        let mut tokens = vec![1, 2, 3];
        truncate_prompt(&mut tokens, Some(10)).unwrap();
        assert_eq!(tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_truncate_prompt_zero_invalid() {
        let mut tokens = vec![1, 2, 3];
        assert!(truncate_prompt(&mut tokens, Some(0)).is_err());
    }

    #[test]
    fn test_truncate_prompt_negative_invalid() {
        let mut tokens = vec![1, 2, 3];
        assert!(truncate_prompt(&mut tokens, Some(-2)).is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_chat_completion_n1_regression() {
        let engine = Arc::new(make_test_engine());
        engine.spawn_step_loop();

        let request = make_chat_request();
        let response = engine.chat_completion(request).await.unwrap();

        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].index, 0);
        assert!(response.usage.prompt_tokens > 0);
        assert!(response.usage.completion_tokens.unwrap_or(0) > 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_chat_completion_n2() {
        let engine = Arc::new(make_test_engine());
        engine.spawn_step_loop();

        let mut request = make_chat_request();
        request.n = 2;
        let response = engine.chat_completion(request).await.unwrap();

        assert_eq!(response.choices.len(), 2);
        assert_eq!(response.choices[0].index, 0);
        assert_eq!(response.choices[1].index, 1);
        // prompt_tokens counted once (not * n).
        assert!(response.usage.prompt_tokens > 0);
        // completion_tokens is sum of both choices.
        assert!(response.usage.completion_tokens.unwrap_or(0) > 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_completion_n1_regression() {
        let engine = Arc::new(make_test_engine());
        engine.spawn_step_loop();

        let request =
            make_completion_request(Some(protocol::CompletionPrompt::TokenIds(vec![1, 2, 3])));
        let response = engine.completion(request).await.unwrap();

        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].index, 0);
        assert_eq!(response.usage.prompt_tokens, 3);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_completion_n3() {
        let engine = Arc::new(make_test_engine());
        engine.spawn_step_loop();

        let mut request =
            make_completion_request(Some(protocol::CompletionPrompt::TokenIds(vec![1, 2, 3])));
        request.n = 3;
        let response = engine.completion(request).await.unwrap();

        assert_eq!(response.choices.len(), 3);
        assert_eq!(response.choices[0].index, 0);
        assert_eq!(response.choices[1].index, 1);
        assert_eq!(response.choices[2].index, 2);
        // prompt_tokens counted once.
        assert_eq!(response.usage.prompt_tokens, 3);
        // completion_tokens summed across all 3.
        assert!(response.usage.completion_tokens.unwrap_or(0) > 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_completion_multi_prompt_n1() {
        let engine = Arc::new(make_test_engine());
        engine.spawn_step_loop();

        let request =
            make_completion_request(Some(protocol::CompletionPrompt::MultipleTokenIds(vec![
                vec![1, 2],
                vec![3, 4, 5],
            ])));
        let response = engine.completion(request).await.unwrap();

        assert_eq!(response.choices.len(), 2);
        assert_eq!(response.choices[0].index, 0);
        assert_eq!(response.choices[1].index, 1);
        // prompt_tokens = sum of prompt lengths: 2 + 3 = 5.
        assert_eq!(response.usage.prompt_tokens, 5);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_completion_multi_prompt_n2() {
        let engine = Arc::new(make_test_engine());
        engine.spawn_step_loop();

        let mut request =
            make_completion_request(Some(protocol::CompletionPrompt::MultipleTokenIds(vec![
                vec![1, 2],
                vec![3, 4, 5],
            ])));
        request.n = 2;
        let response = engine.completion(request).await.unwrap();

        // 2 prompts * 2 = 4 choices.
        assert_eq!(response.choices.len(), 4);
        assert_eq!(response.choices[0].index, 0);
        assert_eq!(response.choices[1].index, 1);
        assert_eq!(response.choices[2].index, 2);
        assert_eq!(response.choices[3].index, 3);
        // prompt_tokens = sum of unique prompt lengths: 2 + 3 = 5.
        assert_eq!(response.usage.prompt_tokens, 5);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_chat_completion_stream_n2() {
        let engine = Arc::new(make_test_engine());
        engine.spawn_step_loop();

        let mut request = make_chat_request();
        request.n = 2;
        request.stream = true;

        let (_request_id, _model, mut rx) = engine.chat_completion_stream(request).await.unwrap();

        // Collect all deltas.
        let mut seen_indices = std::collections::HashSet::new();
        while let Some(delta) = rx.recv().await {
            seen_indices.insert(delta.index);
        }

        // Both choice indices should appear.
        assert!(
            seen_indices.contains(&0),
            "Missing index 0 in stream deltas"
        );
        assert!(
            seen_indices.contains(&1),
            "Missing index 1 in stream deltas"
        );
    }

    #[test]
    fn test_tokenize_completion_prompts_single() {
        let engine = make_test_engine();
        let request = make_completion_request(Some(protocol::CompletionPrompt::Single(
            "hello".to_string(),
        )));
        let prompts = engine.tokenize_completion_prompts(&request).unwrap();
        assert_eq!(prompts.len(), 1);
        // Without tokenizer, byte values of "hello".
        assert_eq!(prompts[0].len(), 5);
    }

    #[test]
    fn test_tokenize_completion_prompts_multiple() {
        let engine = make_test_engine();
        let request = make_completion_request(Some(protocol::CompletionPrompt::Multiple(vec![
            "foo".to_string(),
            "bar".to_string(),
        ])));
        let prompts = engine.tokenize_completion_prompts(&request).unwrap();
        assert_eq!(prompts.len(), 2);
    }

    #[test]
    fn test_tokenize_completion_prompts_token_ids() {
        let engine = make_test_engine();
        let request =
            make_completion_request(Some(protocol::CompletionPrompt::TokenIds(vec![10, 20, 30])));
        let prompts = engine.tokenize_completion_prompts(&request).unwrap();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0], vec![10, 20, 30]);
    }

    #[test]
    fn test_tokenize_completion_prompts_none() {
        let engine = make_test_engine();
        let request = make_completion_request(None);
        let prompts = engine.tokenize_completion_prompts(&request).unwrap();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0], vec![0]);
    }

    #[test]
    fn test_stream_delta_has_choice_index() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut requests = HashMap::new();
        {
            let mut state = make_test_request_state(Some(tx));
            state.num_prompt_tokens = 3;
            state.choice_index = 7;
            requests.insert("req-idx".to_string(), state);
        }

        AsyncEngine::process_output(
            &mut requests,
            EngineCoreOutput {
                request_id: "req-idx".to_string(),
                new_token_ids: vec![42],
                finish_reason: Some(FinishReason::Stop),
                stop_reason: None,
                num_cached_tokens: 0,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: None,
            },
        );

        let delta = rx.try_recv().unwrap();
        assert_eq!(delta.index, 7);
    }

    // ---------------------------------------------------------------
    // parse_response_format tests
    // ---------------------------------------------------------------

    #[test]
    fn test_parse_response_format_none() {
        let result = parse_response_format(&None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_response_format_text() {
        let rf = protocol::ResponseFormat {
            format_type: "text".to_string(),
            json_schema: None,
        };
        let result = parse_response_format(&Some(rf)).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_response_format_json_object() {
        let rf = protocol::ResponseFormat {
            format_type: "json_object".to_string(),
            json_schema: None,
        };
        let result = parse_response_format(&Some(rf)).unwrap();
        assert!(matches!(result, Some(GuidedGrammar::Json)));
    }

    #[test]
    fn test_parse_response_format_json_schema() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"]
        });
        let rf = protocol::ResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: Some(protocol::JsonSchemaResponseFormat {
                name: "test".to_string(),
                description: None,
                json_schema: Some(schema.clone()),
                strict: None,
            }),
        };
        let result = parse_response_format(&Some(rf)).unwrap();
        match result {
            Some(GuidedGrammar::JsonSchema { schema: s }) => {
                assert_eq!(s, schema);
            }
            other => panic!("expected JsonSchema, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_response_format_json_schema_missing_schema() {
        let rf = protocol::ResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: None,
        };
        let result = parse_response_format(&Some(rf));
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_response_format_invalid_type() {
        let rf = protocol::ResponseFormat {
            format_type: "xml".to_string(),
            json_schema: None,
        };
        let result = parse_response_format(&Some(rf));
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // resolve_guided_grammar tests
    // ---------------------------------------------------------------

    #[test]
    fn test_resolve_guided_grammar_none() {
        let result = resolve_guided_grammar(&None, &None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_guided_grammar_regex_only() {
        let result = resolve_guided_grammar(&None, &Some("[0-9]+".to_string())).unwrap();
        match result {
            Some(GuidedGrammar::Regex { pattern }) => assert_eq!(pattern, "[0-9]+"),
            other => panic!("expected Regex, got {other:?}"),
        }
    }

    #[test]
    fn test_resolve_guided_grammar_response_format_only() {
        let rf = protocol::ResponseFormat {
            format_type: "json_object".to_string(),
            json_schema: None,
        };
        let result = resolve_guided_grammar(&Some(rf), &None).unwrap();
        assert!(matches!(result, Some(GuidedGrammar::Json)));
    }

    #[test]
    fn test_resolve_guided_grammar_conflict() {
        let rf = protocol::ResponseFormat {
            format_type: "json_object".to_string(),
            json_schema: None,
        };
        let result = resolve_guided_grammar(&Some(rf), &Some("[0-9]+".to_string()));
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // tool_choice helper tests
    // ---------------------------------------------------------------

    #[test]
    fn test_is_tool_choice_none_with_none_value() {
        assert!(!is_tool_choice_none(&None));
    }

    #[test]
    fn test_is_tool_choice_none_with_none_string() {
        assert!(is_tool_choice_none(&Some(serde_json::json!("none"))));
    }

    #[test]
    fn test_is_tool_choice_none_with_auto() {
        assert!(!is_tool_choice_none(&Some(serde_json::json!("auto"))));
    }

    #[test]
    fn test_get_tool_choice_function_name_none() {
        assert_eq!(get_tool_choice_function_name(&None), None);
    }

    #[test]
    fn test_get_tool_choice_function_name_auto() {
        assert_eq!(
            get_tool_choice_function_name(&Some(serde_json::json!("auto"))),
            None
        );
    }

    #[test]
    fn test_get_tool_choice_function_name_object() {
        let tc = serde_json::json!({
            "type": "function",
            "function": {"name": "get_weather"}
        });
        assert_eq!(
            get_tool_choice_function_name(&Some(tc)),
            Some("get_weather".to_string())
        );
    }

    #[test]
    fn test_get_tool_choice_function_name_missing_name() {
        let tc = serde_json::json!({"type": "function", "function": {}});
        assert_eq!(get_tool_choice_function_name(&Some(tc)), None);
    }

    // -------------------------------------------------------------------
    // Pooling mode tests
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_pooling_mode_rejects_chat_completion() {
        let mut engine = make_test_engine();
        engine.set_is_pooling(true);
        let engine = Arc::new(engine);
        engine.spawn_step_loop();

        let request: protocol::ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "Hello"}]
        }))
        .unwrap();
        let result = engine.chat_completion(request).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("pooling mode"),
            "Error should mention pooling mode, got: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_pooling_mode_rejects_completion() {
        let mut engine = make_test_engine();
        engine.set_is_pooling(true);
        let engine = Arc::new(engine);
        engine.spawn_step_loop();

        let request: protocol::CompletionRequest =
            serde_json::from_value(serde_json::json!({"prompt": "Hello"})).unwrap();
        let result = engine.completion(request).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("pooling mode"),
            "Error should mention pooling mode, got: {err_msg}"
        );
    }

    #[test]
    fn test_process_output_stores_pooler_output() {
        let mut requests = HashMap::new();
        requests.insert("pool-1".to_string(), make_test_request_state(None));

        let embedding = vec![0.1, 0.2, 0.3];
        AsyncEngine::process_output(
            &mut requests,
            EngineCoreOutput {
                request_id: "pool-1".to_string(),
                new_token_ids: vec![],
                finish_reason: Some(FinishReason::Stop),
                stop_reason: None,
                num_cached_tokens: 0,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: Some(embedding.clone()),
            },
        );

        // Request should be removed (finished + no stream).
        // Check that pooler_output was set before removal by using streaming.
        let mut requests2 = HashMap::new();
        let (tx, _rx) = mpsc::unbounded_channel();
        requests2.insert("pool-2".to_string(), make_test_request_state(Some(tx)));

        AsyncEngine::process_output(
            &mut requests2,
            EngineCoreOutput {
                request_id: "pool-2".to_string(),
                new_token_ids: vec![],
                finish_reason: None, // Not finished yet, so state stays.
                stop_reason: None,
                num_cached_tokens: 0,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: Some(embedding.clone()),
            },
        );

        let state = requests2.get("pool-2").unwrap();
        assert_eq!(state.pooler_output, Some(embedding));
    }

    #[test]
    fn test_pooling_mode_flag() {
        let mut engine = make_test_engine();
        assert!(!engine.is_pooling());
        engine.set_is_pooling(true);
        assert!(engine.is_pooling());
    }

    // -----------------------------------------------------------------------
    // FailingExecutor — always returns Err from execute_model
    // -----------------------------------------------------------------------

    /// An executor that always fails on `execute_model`.
    /// Used to test the error-abort path in the step loops.
    struct FailingExecutor;

    impl vllm_engine::executor::Executor for FailingExecutor {
        fn execute_model(
            &mut self,
            _scheduler_output: &vllm_core::scheduler::output::SchedulerOutput,
        ) -> vllm_engine::error::EngineResult<vllm_engine::executor::ModelRunnerOutput> {
            Err(vllm_engine::error::EngineError::Executor(
                "simulated forward pass failure".into(),
            ))
        }

        fn initialize_cache(
            &mut self,
            _num_gpu_blocks: usize,
            _num_cpu_blocks: usize,
        ) -> vllm_engine::error::EngineResult<()> {
            Ok(())
        }

        fn determine_available_memory(&mut self) -> vllm_engine::error::EngineResult<Vec<usize>> {
            Ok(vec![1024 * 16 * 1024])
        }

        fn shutdown(&mut self) {}

        fn is_sleeping(&self) -> bool {
            false
        }
    }

    fn make_failing_engine(async_scheduling: bool) -> AsyncEngine {
        let executor = Box::new(FailingExecutor);
        let mut config = make_engine_config();
        config.async_scheduling = async_scheduling;
        let client = Box::new(InprocClient::new(config, executor));
        let mut engine = AsyncEngine::new(client, "test-model".to_string(), 4096);
        engine.set_async_scheduling(async_scheduling);
        engine
    }

    /// A mock client that always returns empty outputs (simulates the
    /// scheduler never scheduling any requests — the no-progress bug).
    struct NoProgressClient;

    impl EngineCoreClient for NoProgressClient {
        fn get_output(
            &mut self,
        ) -> vllm_engine::error::EngineResult<(vllm_common::EngineCoreOutputs, bool)> {
            // Simulate scheduler returning nothing to schedule.
            std::thread::sleep(std::time::Duration::from_millis(10));
            Ok((
                vllm_common::EngineCoreOutputs {
                    engine_index: 0,
                    outputs: vec![],
                    timestamp: 0.0,
                    scheduler_stats: None,
                },
                false,
            ))
        }

        fn add_request(
            &mut self,
            _request: EngineCoreRequest,
        ) -> vllm_engine::error::EngineResult<()> {
            Ok(())
        }

        fn abort_requests(
            &mut self,
            _request_ids: &[String],
        ) -> vllm_engine::error::EngineResult<()> {
            Ok(())
        }

        fn abort_running_requests(&mut self) {}

        fn shutdown(&mut self) -> vllm_engine::error::EngineResult<()> {
            Ok(())
        }

        fn reset_prefix_cache(&mut self) -> vllm_engine::error::EngineResult<bool> {
            Ok(true)
        }

        fn pause_scheduler(&mut self, _mode: PauseMode) -> vllm_engine::error::EngineResult<()> {
            Ok(())
        }

        fn resume_scheduler(&mut self) -> vllm_engine::error::EngineResult<()> {
            Ok(())
        }

        fn is_scheduler_paused(&self) -> bool {
            false
        }
    }

    fn make_no_progress_engine() -> AsyncEngine {
        let client: Box<dyn EngineCoreClient + Send> = Box::new(NoProgressClient);
        let mut engine = AsyncEngine::new(client, "test-model".to_string(), 4096);
        engine.set_no_progress_timeout(Duration::from_secs(2));
        engine
    }

    /// Test that the no-progress watchdog aborts requests stuck with no output.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_no_progress_watchdog() {
        let engine = Arc::new(make_no_progress_engine());
        engine.spawn_step_loop();

        let request = make_chat_request();

        // Submit a request. The mock client never produces output tokens,
        // so the watchdog should abort it after ~2 seconds.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            engine.chat_completion(request),
        )
        .await;

        // Must not time out (watchdog should fire within ~2s).
        let inner = result.expect("request timed out — no-progress watchdog did not fire");

        // Must be an error (aborted by watchdog).
        let err = inner.expect_err("expected error from no-progress watchdog");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("No-progress watchdog") || err_msg.contains("step loop exited"),
            "unexpected error: {err_msg}"
        );
    }

    /// Test that a forward pass failure in the **sync** step loop aborts the
    /// request and returns an error instead of hanging forever.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_sync_loop_aborts_on_executor_error() {
        let engine = Arc::new(make_failing_engine(false));
        engine.spawn_step_loop();

        let request = make_chat_request();

        // This should NOT hang — the executor error should abort the request
        // and poll_until_done should return an error.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            engine.chat_completion(request),
        )
        .await;

        // Must not time out.
        let inner = result.expect("request timed out — step loop hang not fixed");

        // Must be an error (not a successful completion).
        let err = inner.expect_err("expected error from failing executor");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("simulated forward pass failure")
                || err_msg.contains("Engine step error")
                || err_msg.contains("step loop exited"),
            "unexpected error: {err_msg}"
        );
    }

    /// Test that a forward pass failure in the **async** step loop aborts the
    /// request and returns an error instead of hanging forever.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_async_loop_aborts_on_executor_error() {
        let engine = Arc::new(make_failing_engine(true));
        engine.spawn_step_loop();

        let request = make_chat_request();

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            engine.chat_completion(request),
        )
        .await;

        let inner = result.expect("request timed out — async step loop hang not fixed");

        let err = inner.expect_err("expected error from failing executor");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("simulated forward pass failure")
                || err_msg.contains("Executor error")
                || err_msg.contains("step loop exited"),
            "unexpected error: {err_msg}"
        );
    }

    #[test]
    fn test_render_chat_completion() {
        let engine = make_test_engine_with_tokenizer();
        let request = make_chat_request();
        let (conversation, engine_prompts) = engine.render_chat_completion(request).unwrap();

        // Conversation should contain the original message.
        assert_eq!(conversation.len(), 1);
        assert_eq!(conversation[0]["role"], "user");
        assert_eq!(conversation[0]["content"], "Hello");

        // Engine prompts should have exactly one entry with a non-empty prompt.
        assert_eq!(engine_prompts.len(), 1);
        let prompt = engine_prompts[0].prompt.as_str().unwrap();
        assert!(!prompt.is_empty(), "rendered prompt should not be empty");
    }

    #[test]
    fn test_render_chat_completion_pooling_mode_rejected() {
        let mut engine = make_test_engine();
        engine.set_is_pooling(true);
        let request = make_chat_request();
        let result = engine.render_chat_completion(request);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("pooling mode"));
    }

    #[tokio::test]
    async fn test_reset_prefix_cache_via_control_channel() {
        let engine = Arc::new(make_test_engine());
        engine.spawn_step_loop();

        // No running requests — should succeed.
        let result = engine.reset_prefix_cache().await.unwrap();
        assert!(
            result,
            "reset_prefix_cache should return true with no running requests"
        );
    }
}
