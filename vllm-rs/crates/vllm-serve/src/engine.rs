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
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::{Mutex, Notify, mpsc};
use tracing::{debug, error};
use uuid::Uuid;
use vllm_common::sampling::GuidedGrammar;
use vllm_common::{EngineCoreOutput, EngineCoreRequest, FinishReason, SamplingParams, StopReason};
use vllm_engine::core_client::EngineCoreClient;

use crate::chat_template::ChatTemplate;
use crate::detokenizer::IncrementalDetokenizer;
use crate::error::{ServeError, ServeResult};
use crate::protocol;
use crate::tokenizer::Tokenizer;
use crate::tool_parser::{
    DeltaToolCall, StreamingToolParserState, ToolCallParser, ToolParserDelta,
};

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

    /// Streaming tool parser state (if tool parsing is active for this request).
    tool_parser_state: Option<Box<dyn StreamingToolParserState + Send>>,

    /// Accumulated generated text so far (for streaming tool parsing).
    accumulated_text: String,

    /// Whether any tool call deltas have been emitted (for setting finish_reason).
    tool_calls_emitted: bool,
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
    /// The engine client + channel receiver, held until `spawn_step_loop`
    /// moves them into the background task. `None` after the loop starts.
    #[allow(clippy::type_complexity)]
    pending_loop: std::sync::Mutex<
        Option<(
            Box<dyn EngineCoreClient + Send>,
            mpsc::UnboundedReceiver<EngineCoreRequest>,
        )>,
    >,
    model_name: String,
    max_model_len: usize,
    /// Notified after every engine step so request handlers can check results.
    notify: Arc<Notify>,
    /// Optional tokenizer for encoding prompts and decoding outputs.
    tokenizer: Option<Arc<Tokenizer>>,
    /// Optional chat template for formatting chat messages.
    chat_template: Option<Arc<ChatTemplate>>,
    /// Optional tool call parser for extracting structured tool calls from output.
    tool_parser: Option<Arc<dyn ToolCallParser>>,
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

        Self {
            requests: Arc::new(Mutex::new(HashMap::new())),
            request_tx: tx,
            pending_loop: std::sync::Mutex::new(Some((client, rx))),
            model_name,
            max_model_len,
            notify: Arc::new(Notify::new()),
            tokenizer: None,
            chat_template: None,
            tool_parser: None,
        }
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
        let ec_request = self.chat_to_engine_request(&base_id, &request, &sampling_params)?;
        let prompt_token_ids = ec_request.prompt_token_ids.clone().unwrap_or_default();
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
        let metrics = crate::metrics::VllmMetrics::global();
        metrics.requests_total.inc();
        metrics.prompt_tokens_total.inc_by(num_prompt_tokens as u64);

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

            let detokenizer = self.tokenizer.as_ref().map(|tok| {
                IncrementalDetokenizer::new(
                    Arc::clone(tok),
                    &prompt_token_ids,
                    sp.stop.clone(),
                    sp.min_tokens,
                    sp.include_stop_str_in_output,
                    sp.skip_special_tokens,
                )
            });

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
                            (
                                extracted.content,
                                Some(extracted.tool_calls),
                                "tool_calls".to_string(),
                            )
                        } else {
                            (Some(text), None, finish_reason_str)
                        }
                    } else {
                        (Some(text), None, finish_reason_str)
                    }
                } else {
                    (Some(text), None, finish_reason_str)
                };

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
        let ec_request = self.chat_to_engine_request(&base_id, &request, &sampling_params)?;
        let prompt_token_ids = ec_request.prompt_token_ids.clone().unwrap_or_default();
        let num_prompt_tokens = prompt_token_ids.len() as u32;

        // Resolve max_tokens: None → remaining capacity, Some(v) → min(v, remaining).
        self.resolve_max_tokens(&mut sampling_params, prompt_token_ids.len());

        // Per-HTTP-request metrics.
        let metrics = crate::metrics::VllmMetrics::global();
        metrics.requests_total.inc();
        metrics.prompt_tokens_total.inc_by(num_prompt_tokens as u64);

        // Determine if tool parsing is active for this request.
        let use_tool_parser = self.tool_parser.is_some()
            && request.tools.is_some()
            && !is_tool_choice_none(&request.tool_choice);

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

            let detokenizer = self.tokenizer.as_ref().map(|tok| {
                IncrementalDetokenizer::new(
                    Arc::clone(tok),
                    &prompt_token_ids,
                    sp.stop.clone(),
                    sp.min_tokens,
                    sp.include_stop_str_in_output,
                    sp.skip_special_tokens,
                )
            });

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
        let base_id = request
            .request_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.model_name.clone());
        let n = request.n.max(1) as usize;

        let sampling_params = self.build_sampling_params_from_completion(&request);

        // Normalize prompt to Vec<Vec<u32>>.
        let prompts = self.tokenize_completion_prompts(&request)?;
        let total = prompts.len() * n;

        // Per-HTTP-request metrics (once).
        let total_prompt_tokens: u32 = prompts.iter().map(|p| p.len() as u32).sum();
        let metrics = crate::metrics::VllmMetrics::global();
        metrics.requests_total.inc();
        metrics
            .prompt_tokens_total
            .inc_by(total_prompt_tokens as u64);

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

            choices.push(protocol::CompletionResponseChoice {
                index: *choice_index,
                text,
                logprobs: completion_logprobs,
                finish_reason: Some(finish_reason_str),
                stop_reason: state.stop_reason.map(|sr| match sr {
                    StopReason::Token(id) => serde_json::Value::Number(id.into()),
                    StopReason::String(s) => serde_json::Value::String(s),
                }),
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
        let (client, rx) = self
            .pending_loop
            .lock()
            .expect("spawn_step_loop lock poisoned")
            .take()
            .expect("spawn_step_loop called twice");

        Self::spawn_step_loop_inner(
            client,
            rx,
            Arc::clone(&self.requests),
            Arc::clone(&self.notify),
        )
    }

    /// Internal: actually spawn the step loop with the client.
    fn spawn_step_loop_inner(
        mut client: Box<dyn EngineCoreClient + Send>,
        mut request_rx: mpsc::UnboundedReceiver<EngineCoreRequest>,
        requests: Arc<Mutex<HashMap<String, RequestState>>>,
        notify: Arc<Notify>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            loop {
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
                    // Block until a new request arrives on the channel.
                    match request_rx.recv().await {
                        Some(ec_request) => {
                            if let Err(e) = client.add_request(ec_request) {
                                error!("Failed to add request: {}", e);
                            }
                        }
                        None => break, // Channel closed, engine dropped.
                    }
                }

                // 3. Run one engine step. This is synchronous (model forward)
                //    so we use block_in_place to let tokio schedule other work.
                let step_result = tokio::task::block_in_place(|| client.get_output());

                match step_result {
                    Ok(outputs) => {
                        // 4. Route outputs to requests (brief lock).
                        if !outputs.outputs.is_empty() {
                            let mut reqs = requests.lock().await;
                            for output in outputs.outputs {
                                Self::process_output(&mut reqs, output);
                            }
                        }
                    }
                    Err(e) => {
                        error!("Engine step error: {}", e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    }
                }

                // 5. Wake all waiting handlers so they can check their results.
                notify.notify_waiters();
            }
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
    ) -> ServeResult<()> {
        let metrics = crate::metrics::VllmMetrics::global();
        metrics.requests_active.inc();

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
                    tool_parser_state,
                    accumulated_text: String::new(),
                    tool_calls_emitted: false,
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

            // Check if the request is done (brief lock).
            {
                let mut reqs = self.requests.lock().await;
                if let Some(req_state) = reqs.get(request_id) {
                    if req_state.finish_reason.is_some() {
                        return Ok(reqs.remove(request_id).unwrap());
                    }
                } else {
                    return Err(ServeError::RequestNotFound(request_id.to_string()));
                }
            }

            // Wait for the background step loop to produce new outputs.
            notified.await;
        }
    }

    /// Process an engine output for a single request.
    fn process_output(requests: &mut HashMap<String, RequestState>, output: EngineCoreOutput) {
        let Some(req_state) = requests.get_mut(&output.request_id) else {
            debug!("Output for unknown request {}, ignoring", output.request_id);
            return;
        };

        // Track output tokens.
        let metrics = crate::metrics::VllmMetrics::global();
        metrics
            .output_tokens_total
            .inc_by(output.new_token_ids.len() as u64);

        // --- TTFT / ITL timing ---
        let now = Instant::now();
        if !output.new_token_ids.is_empty() {
            if req_state.first_token_time.is_none() {
                // First token: record TTFT.
                req_state.first_token_time = Some(now);
                let ttft = now.duration_since(req_state.submit_time).as_secs_f64();
                metrics.time_to_first_token_seconds.observe(ttft);
            } else if let Some(last) = req_state.last_token_time {
                // Subsequent token: record inter-token latency.
                let itl = now.duration_since(last).as_secs_f64();
                metrics.inter_token_latency_seconds.observe(itl);
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
        let step_logprobs = output.new_logprobs.clone();

        // Update cached tokens.
        if output.num_cached_tokens > 0 {
            req_state.num_cached_tokens = output.num_cached_tokens;
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

            metrics.request_latency_seconds.observe(total_latency);
            // Drop stream sender; capture whether this was a streaming request.
            let was_streaming = req_state.stream_tx.take().is_some();
            metrics.requests_active.dec();
            metrics.requests_success_total.inc();

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

        // Tokenize the text, or fall back to byte-value IDs.
        let token_ids = if let Some(tok) = &self.tokenizer {
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
        })
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

        // Parse response_format → guided_grammar.
        let guided_grammar = parse_response_format(&request.response_format)?;

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
            logit_bias: parse_logit_bias(&request.logit_bias),
            guided_grammar,
            ..Default::default()
        })
    }

    /// Build SamplingParams from a completion request.
    fn build_sampling_params_from_completion(
        &self,
        request: &protocol::CompletionRequest,
    ) -> SamplingParams {
        let stop = match &request.stop {
            Some(protocol::StopCondition::Single(s)) => vec![s.clone()],
            Some(protocol::StopCondition::Multiple(v)) => v.clone(),
            None => vec![],
        };
        // Completion API: logprobs is directly the count.
        let logprobs = request.logprobs.map(|n| n as i32);

        SamplingParams {
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
            logit_bias: parse_logit_bias(&request.logit_bias),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert API-style logit_bias (string keys, f64 values) to sampler format.
///
/// OpenAI API uses string token IDs as keys; we parse them to u32.
/// Invalid keys are silently ignored.
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

    fn make_test_request_state(
        stream_tx: Option<mpsc::UnboundedSender<StreamDelta>>,
    ) -> RequestState {
        RequestState {
            generated_token_ids: Vec::new(),
            num_prompt_tokens: 5,
            num_cached_tokens: 0,
            finish_reason: None,
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
            tool_parser_state: None,
            accumulated_text: String::new(),
            tool_calls_emitted: false,
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
            eos_token_ids: vec![],
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
        };

        let params = engine.build_sampling_params_from_completion(&request);
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
        };

        let params = engine.build_sampling_params_from_completion(&request);
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
        }
    }

    // -- n>1 and multi-prompt tests --

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
}
