// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Offline batch inference API.
//!
//! Provides `LLM` — a synchronous, Python-like programmatic interface for
//! running inference without an HTTP server. Mirrors the Python
//! `vllm.LLM(model=...).generate()` / `.chat()` pattern.
//!
//! ```rust,no_run
//! use vllm_serve::llm::LLM;
//!
//! let llm = LLM::new("HuggingFaceTB/SmolLM2-135M")?;
//! let outputs = llm.generate(&["Hello, world!"], None)?;
//! for output in &outputs {
//!     println!("{}", output.outputs[0].text);
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```

use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::task::JoinHandle;

pub use vllm_common::SamplingParams;

use crate::engine::AsyncEngine;
use crate::init::{InitializedStack, VllmConfig};
use crate::protocol;

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// A single completion output (one of possibly `n` per prompt).
#[derive(Debug, Clone)]
pub struct CompletionOutput {
    /// Index within the request's `n` completions.
    pub index: u32,
    /// Generated text.
    pub text: String,
    /// Generated token IDs.
    pub token_ids: Vec<u32>,
    /// Why generation stopped (e.g. "stop", "length").
    pub finish_reason: Option<String>,
}

/// Output for a single prompt / request.
#[derive(Debug, Clone)]
pub struct RequestOutput {
    /// Unique request identifier.
    pub request_id: String,
    /// The original prompt text (if available).
    pub prompt: Option<String>,
    /// Prompt token IDs.
    pub prompt_token_ids: Vec<u32>,
    /// One or more completion outputs.
    pub outputs: Vec<CompletionOutput>,
    /// Whether generation is complete.
    pub finished: bool,
}

// ---------------------------------------------------------------------------
// ChatMessage — ergonomic wrapper
// ---------------------------------------------------------------------------

/// A chat message for `LLM::chat()`.
///
/// Thin convenience type so callers don't need to construct
/// `protocol::ChatCompletionMessageParam` directly.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    /// Create a new chat message.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    /// Shorthand for a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    /// Shorthand for a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    /// Shorthand for an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

// ---------------------------------------------------------------------------
// LLMBuilder
// ---------------------------------------------------------------------------

/// Builder for configuring and constructing an [`LLM`] instance.
pub struct LLMBuilder {
    config: VllmConfig,
}

impl LLMBuilder {
    /// Create a builder for the given model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            config: VllmConfig {
                model: model.into(),
                ..VllmConfig::default()
            },
        }
    }

    /// Set the device ("cpu", "cuda:0", "metal", "auto").
    pub fn device(mut self, device: impl Into<String>) -> Self {
        self.config.device = device.into();
        self
    }

    /// Set the weight dtype ("auto", "float16", "bfloat16", "float32").
    pub fn dtype(mut self, dtype: impl Into<String>) -> Self {
        self.config.dtype = dtype.into();
        self
    }

    /// Set the maximum model context length.
    pub fn max_model_len(mut self, len: usize) -> Self {
        self.config.max_model_len = Some(len);
        self
    }

    /// Set the maximum number of concurrent sequences.
    pub fn max_num_seqs(mut self, n: usize) -> Self {
        self.config.max_num_seqs = n;
        self
    }

    /// Set the KV cache block size in tokens.
    pub fn block_size(mut self, size: usize) -> Self {
        self.config.block_size = size;
        self
    }

    /// Set the fraction of GPU memory for KV cache (0.0–1.0).
    pub fn gpu_memory_utilization(mut self, frac: f64) -> Self {
        self.config.gpu_memory_utilization = frac;
        self
    }

    /// Set the HuggingFace token for gated models.
    pub fn hf_token(mut self, token: impl Into<String>) -> Self {
        self.config.hf_token = Some(token.into());
        self
    }

    /// Set a specific GGUF filename to download.
    pub fn gguf_file(mut self, filename: impl Into<String>) -> Self {
        self.config.gguf_file = Some(filename.into());
        self
    }

    /// Build the [`LLM`] instance, loading the model.
    pub fn build(self) -> Result<LLM> {
        LLM::from_config(self.config)
    }
}

// ---------------------------------------------------------------------------
// LLM
// ---------------------------------------------------------------------------

/// Offline batch inference engine.
///
/// Owns a tokio runtime and an [`AsyncEngine`]. Provides synchronous
/// `generate()` and `chat()` methods that block until all outputs are ready.
pub struct LLM {
    engine: Arc<AsyncEngine>,
    runtime: tokio::runtime::Runtime,
    _step_handle: JoinHandle<()>,
    model_name: String,
    max_model_len: usize,
}

impl LLM {
    /// Create an LLM with default settings for the given model.
    ///
    /// Equivalent to `LLM::builder(model).build()`.
    pub fn new(model: impl Into<String>) -> Result<Self> {
        Self::builder(model).build()
    }

    /// Return a builder for fine-grained configuration.
    pub fn builder(model: impl Into<String>) -> LLMBuilder {
        LLMBuilder::new(model)
    }

    /// Internal constructor from a fully-specified config.
    fn from_config(config: VllmConfig) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .context("failed to create tokio runtime")?;

        let InitializedStack {
            engine,
            model_name,
            max_model_len,
        } = runtime.block_on(async {
            // initialize_stack is sync but we run inside the runtime so
            // spawn_step_loop (which needs a tokio context) works later.
            tokio::task::block_in_place(|| crate::init::initialize_stack(&config))
        })?;

        // Enter the runtime context so tokio::spawn works inside
        // spawn_step_loop (needed for async scheduling path).
        let _guard = runtime.enter();
        let step_handle = engine.spawn_step_loop();

        Ok(Self {
            engine,
            runtime,
            _step_handle: step_handle,
            model_name,
            max_model_len,
        })
    }

    /// The model name / HuggingFace ID.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// The maximum context length.
    pub fn max_model_len(&self) -> usize {
        self.max_model_len
    }

    // -----------------------------------------------------------------------
    // generate()
    // -----------------------------------------------------------------------

    /// Generate text completions for one or more prompts.
    ///
    /// Each prompt produces a [`RequestOutput`] with one or more
    /// [`CompletionOutput`]s (controlled by `SamplingParams::n`).
    pub fn generate(
        &self,
        prompts: &[&str],
        params: Option<SamplingParams>,
    ) -> Result<Vec<RequestOutput>> {
        let params = params.unwrap_or_default();
        params
            .validate()
            .map_err(|e| anyhow::anyhow!("invalid sampling params: {e}"))?;

        let n = params.n.max(1) as usize;

        let mut outputs = Vec::with_capacity(prompts.len());

        for (i, &prompt_text) in prompts.iter().enumerate() {
            let request = build_completion_request(prompt_text, &params, &self.model_name);

            let response = self
                .runtime
                .block_on(self.engine.completion(request))
                .map_err(|e| anyhow::anyhow!("completion failed for prompt {i}: {e}"))?;

            outputs.push(completion_response_to_output(
                &response,
                Some(prompt_text),
                n,
            ));
        }

        Ok(outputs)
    }

    // -----------------------------------------------------------------------
    // chat()
    // -----------------------------------------------------------------------

    /// Generate a chat completion from a list of messages.
    ///
    /// Returns a single [`RequestOutput`] (since chat is typically one
    /// conversation at a time). Use `SamplingParams::n` for multiple
    /// completions of the same conversation.
    pub fn chat(
        &self,
        messages: &[ChatMessage],
        params: Option<SamplingParams>,
    ) -> Result<RequestOutput> {
        let params = params.unwrap_or_default();
        params
            .validate()
            .map_err(|e| anyhow::anyhow!("invalid sampling params: {e}"))?;

        let n = params.n.max(1) as usize;
        let request = build_chat_request(messages, &params, &self.model_name);

        let response = self
            .runtime
            .block_on(self.engine.chat_completion(request))
            .map_err(|e| anyhow::anyhow!("chat completion failed: {e}"))?;

        Ok(chat_response_to_output(&response, n))
    }
}

// ---------------------------------------------------------------------------
// Request builders
// ---------------------------------------------------------------------------

fn build_completion_request(
    prompt: &str,
    params: &SamplingParams,
    model: &str,
) -> protocol::CompletionRequest {
    let stop = if params.stop.is_empty() {
        None
    } else {
        Some(protocol::StopCondition::Multiple(params.stop.clone()))
    };

    protocol::CompletionRequest {
        model: Some(model.to_string()),
        prompt: Some(protocol::CompletionPrompt::Single(prompt.to_string())),
        echo: false,
        temperature: Some(params.temperature),
        top_p: Some(params.top_p),
        n: params.n,
        max_tokens: params.max_tokens,
        stream: false,
        stream_options: None,
        stop,
        frequency_penalty: Some(params.frequency_penalty),
        presence_penalty: Some(params.presence_penalty),
        logit_bias: None,
        logprobs: params.logprobs.map(|v| v.max(0) as u32),
        prompt_logprobs: params.prompt_logprobs.map(|v| v.max(0) as u32),
        suffix: None,
        seed: params.seed.map(|s| s as i64),
        user: None,
        top_k: Some(params.top_k),
        min_p: Some(params.min_p),
        repetition_penalty: Some(params.repetition_penalty),
        min_tokens: params.min_tokens,
        stop_token_ids: params.stop_token_ids.clone(),
        include_stop_str_in_output: params.include_stop_str_in_output,
        ignore_eos: params.ignore_eos,
        skip_special_tokens: params.skip_special_tokens,
        priority: 0,
        cache_salt: None,
        request_id: None,
        guided_regex: None,
    }
}

fn build_chat_request(
    messages: &[ChatMessage],
    params: &SamplingParams,
    model: &str,
) -> protocol::ChatCompletionRequest {
    let msgs: Vec<protocol::ChatCompletionMessageParam> = messages
        .iter()
        .map(|m| protocol::ChatCompletionMessageParam {
            role: m.role.clone(),
            content: Some(serde_json::Value::String(m.content.clone())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        })
        .collect();

    let stop = if params.stop.is_empty() {
        None
    } else {
        Some(protocol::StopCondition::Multiple(params.stop.clone()))
    };

    protocol::ChatCompletionRequest {
        model: Some(model.to_string()),
        messages: msgs,
        temperature: Some(params.temperature),
        top_p: Some(params.top_p),
        n: params.n,
        max_tokens: params.max_tokens,
        max_completion_tokens: None,
        stream: false,
        stream_options: None,
        stop,
        frequency_penalty: Some(params.frequency_penalty),
        presence_penalty: Some(params.presence_penalty),
        logit_bias: None,
        logprobs: None,
        top_logprobs: params.logprobs.map(|v| v.max(0) as u32),
        prompt_logprobs: params.prompt_logprobs.map(|v| v.max(0) as u32),
        seed: params.seed.map(|s| s as i64),
        response_format: None,
        tools: None,
        tool_choice: None,
        user: None,
        top_k: Some(params.top_k),
        min_p: Some(params.min_p),
        repetition_penalty: Some(params.repetition_penalty),
        min_tokens: params.min_tokens,
        stop_token_ids: params.stop_token_ids.clone(),
        include_stop_str_in_output: params.include_stop_str_in_output,
        ignore_eos: params.ignore_eos,
        skip_special_tokens: params.skip_special_tokens,
        priority: 0,
        cache_salt: None,
        request_id: None,
        guided_regex: None,
    }
}

// ---------------------------------------------------------------------------
// Response → RequestOutput mapping
// ---------------------------------------------------------------------------

fn completion_response_to_output(
    resp: &protocol::CompletionResponse,
    prompt: Option<&str>,
    _n: usize,
) -> RequestOutput {
    let outputs: Vec<CompletionOutput> = resp
        .choices
        .iter()
        .map(|c| CompletionOutput {
            index: c.index,
            text: c.text.clone(),
            token_ids: Vec::new(), // Token IDs not exposed in CompletionResponse
            finish_reason: c.finish_reason.clone(),
        })
        .collect();

    RequestOutput {
        request_id: resp.id.clone(),
        prompt: prompt.map(|s| s.to_string()),
        prompt_token_ids: Vec::new(), // Not available from CompletionResponse
        outputs,
        finished: true,
    }
}

fn chat_response_to_output(resp: &protocol::ChatCompletionResponse, _n: usize) -> RequestOutput {
    let outputs: Vec<CompletionOutput> = resp
        .choices
        .iter()
        .map(|c| CompletionOutput {
            index: c.index,
            text: c.message.content.clone().unwrap_or_default(),
            token_ids: Vec::new(),
            finish_reason: c.finish_reason.clone(),
        })
        .collect();

    RequestOutput {
        request_id: resp.id.clone(),
        prompt: None,
        prompt_token_ids: Vec::new(),
        outputs,
        finished: true,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let builder = LLMBuilder::new("test-model");
        assert_eq!(builder.config.model, "test-model");
        assert_eq!(builder.config.device, "auto");
        assert_eq!(builder.config.dtype, "auto");
        assert_eq!(builder.config.max_num_seqs, 256);
        assert_eq!(builder.config.block_size, 16);
        assert!(builder.config.max_model_len.is_none());
    }

    #[test]
    fn test_builder_chaining() {
        let builder = LLMBuilder::new("test-model")
            .device("cpu")
            .dtype("float16")
            .max_model_len(2048)
            .max_num_seqs(32)
            .block_size(8)
            .gpu_memory_utilization(0.5);

        assert_eq!(builder.config.device, "cpu");
        assert_eq!(builder.config.dtype, "float16");
        assert_eq!(builder.config.max_model_len, Some(2048));
        assert_eq!(builder.config.max_num_seqs, 32);
        assert_eq!(builder.config.block_size, 8);
        assert!((builder.config.gpu_memory_utilization - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_chat_message_constructors() {
        let sys = ChatMessage::system("You are helpful.");
        assert_eq!(sys.role, "system");
        assert_eq!(sys.content, "You are helpful.");

        let user = ChatMessage::user("Hello");
        assert_eq!(user.role, "user");

        let asst = ChatMessage::assistant("Hi there!");
        assert_eq!(asst.role, "assistant");
    }

    #[test]
    fn test_completion_output_debug() {
        let output = CompletionOutput {
            index: 0,
            text: "hello".to_string(),
            token_ids: vec![1, 2, 3],
            finish_reason: Some("stop".to_string()),
        };
        // Ensure Debug is derived and doesn't panic.
        let _ = format!("{output:?}");
    }

    #[test]
    fn test_request_output_debug() {
        let output = RequestOutput {
            request_id: "test-id".to_string(),
            prompt: Some("Hello".to_string()),
            prompt_token_ids: vec![1, 2],
            outputs: vec![CompletionOutput {
                index: 0,
                text: "world".to_string(),
                token_ids: vec![3],
                finish_reason: None,
            }],
            finished: true,
        };
        let _ = format!("{output:?}");
    }

    #[test]
    fn test_build_completion_request() {
        let params = SamplingParams {
            temperature: 0.7,
            max_tokens: Some(100),
            stop: vec!["END".to_string()],
            ..SamplingParams::default()
        };
        let req = build_completion_request("Hello", &params, "test-model");
        assert_eq!(req.model, Some("test-model".to_string()));
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.max_tokens, Some(100));
        assert!(!req.stream);
        assert!(req.stop.is_some());
    }

    #[test]
    fn test_build_chat_request() {
        let messages = vec![ChatMessage::system("Be helpful."), ChatMessage::user("Hi")];
        let params = SamplingParams::default();
        let req = build_chat_request(&messages, &params, "test-model");
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, "system");
        assert_eq!(req.messages[1].role, "user");
        assert!(!req.stream);
    }

    #[test]
    fn test_completion_response_to_output() {
        let resp = protocol::CompletionResponse {
            id: "cmpl-123".to_string(),
            object: "text_completion".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![protocol::CompletionResponseChoice {
                index: 0,
                text: "world".to_string(),
                logprobs: None,
                finish_reason: Some("stop".to_string()),
                stop_reason: None,
                prompt_logprobs: None,
            }],
            usage: protocol::UsageInfo {
                prompt_tokens: 1,
                total_tokens: 2,
                completion_tokens: Some(1),
                prompt_tokens_details: None,
            },
        };
        let output = completion_response_to_output(&resp, Some("hello"), 1);
        assert_eq!(output.request_id, "cmpl-123");
        assert_eq!(output.prompt, Some("hello".to_string()));
        assert_eq!(output.outputs.len(), 1);
        assert_eq!(output.outputs[0].text, "world");
        assert!(output.finished);
    }

    #[test]
    fn test_chat_response_to_output() {
        let resp = protocol::ChatCompletionResponse {
            id: "chat-456".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![protocol::ChatCompletionResponseChoice {
                index: 0,
                message: protocol::ChatMessage {
                    role: "assistant".to_string(),
                    content: Some("Hi there!".to_string()),
                    refusal: None,
                    tool_calls: None,
                    reasoning: None,
                },
                logprobs: None,
                finish_reason: Some("stop".to_string()),
                stop_reason: None,
                prompt_logprobs: None,
            }],
            system_fingerprint: None,
            usage: protocol::UsageInfo {
                prompt_tokens: 2,
                total_tokens: 5,
                completion_tokens: Some(3),
                prompt_tokens_details: None,
            },
        };
        let output = chat_response_to_output(&resp, 1);
        assert_eq!(output.request_id, "chat-456");
        assert_eq!(output.outputs[0].text, "Hi there!");
        assert!(output.finished);
    }
}
