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

use vllm_config::CudaGraphConfig;

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
// Prompt — text or pre-tokenized input (mirrors Python's PromptType)
// ---------------------------------------------------------------------------

/// A prompt for [`LLM::generate()`].
///
/// Mirrors Python vLLM's `PromptType`: either a text string or pre-tokenized
/// token IDs. Use the `From` impls for ergonomic construction:
///
/// ```rust
/// use vllm_serve::llm::Prompt;
///
/// let text: Prompt = "Hello, world!".into();
/// let token_ids: Prompt = vec![1u32, 2, 3].into();
/// ```
#[derive(Debug, Clone)]
pub enum Prompt {
    /// A text prompt (will be tokenized by the engine).
    Text(String),
    /// Pre-tokenized prompt token IDs (skips tokenization).
    TokenIds(Vec<u32>),
}

impl From<&str> for Prompt {
    fn from(s: &str) -> Self {
        Prompt::Text(s.to_string())
    }
}

impl From<String> for Prompt {
    fn from(s: String) -> Self {
        Prompt::Text(s)
    }
}

impl From<Vec<u32>> for Prompt {
    fn from(ids: Vec<u32>) -> Self {
        Prompt::TokenIds(ids)
    }
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

    /// Set the number of GPUs for tensor parallelism.
    pub fn tensor_parallel_size(mut self, n: usize) -> Self {
        self.config.tensor_parallel_size = n;
        self
    }

    /// Enable or disable prefix caching (KV cache reuse for shared prefixes).
    pub fn enable_prefix_caching(mut self, enabled: bool) -> Self {
        self.config.enable_prefix_caching = enabled;
        self
    }

    /// Set the CUDA graph configuration for decode acceleration.
    pub fn cuda_graph_config(mut self, config: CudaGraphConfig) -> Self {
        self.config.cuda_graph_config = Some(config);
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

    /// Generate completions for one or more prompts.
    ///
    /// Accepts both text and pre-tokenized prompts via [`Prompt`], mirroring
    /// Python vLLM's `PromptType`. Set `SamplingParams::detokenize` to
    /// `false` to skip detokenization (e.g. for benchmarking).
    ///
    /// Each prompt produces a [`RequestOutput`] with one or more
    /// [`CompletionOutput`]s (controlled by `SamplingParams::n`).
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use vllm_serve::llm::LLM;
    /// # let llm = LLM::new("model")?;
    /// // Text prompts:
    /// llm.generate(&["Hello", "World"], None)?;
    ///
    /// // Token ID prompts:
    /// llm.generate(&[vec![1u32, 2, 3]], None)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn generate<P: Into<Prompt> + Clone>(
        &self,
        prompts: &[P],
        params: Option<SamplingParams>,
    ) -> Result<Vec<RequestOutput>> {
        let params = params.unwrap_or_default();
        params
            .validate()
            .map_err(|e| anyhow::anyhow!("invalid sampling params: {e}"))?;

        let n = params.n.max(1) as usize;

        // Convert all prompts to Prompt enum.
        let prompts: Vec<Prompt> = prompts.iter().map(|p| p.clone().into()).collect();

        // Batch all prompts into a single CompletionRequest so the scheduler
        // sees them together and can batch prefill + decode efficiently.
        let request = build_batched_completion_request(&prompts, &params, &self.model_name);

        let response = self
            .runtime
            .block_on(self.engine.completion(request))
            .map_err(|e| anyhow::anyhow!("completion failed: {e}"))?;

        // The engine returns one CompletionResponseChoice per (prompt, n) pair.
        // Group them back into per-prompt RequestOutputs.
        let mut outputs = Vec::with_capacity(prompts.len());
        let choices_per_prompt = n;
        for (p_idx, chunk) in response.choices.chunks(choices_per_prompt).enumerate() {
            let completion_outputs: Vec<CompletionOutput> = chunk
                .iter()
                .map(|c| CompletionOutput {
                    index: c.index,
                    text: c.text.clone(),
                    token_ids: Vec::new(),
                    finish_reason: c.finish_reason.clone(),
                })
                .collect();

            let prompt_text = match &prompts[p_idx] {
                Prompt::Text(text) => Some(text.clone()),
                Prompt::TokenIds(_) => None,
            };

            outputs.push(RequestOutput {
                request_id: format!("{}-{p_idx}", response.id),
                prompt: prompt_text,
                prompt_token_ids: match &prompts[p_idx] {
                    Prompt::TokenIds(ids) => ids.clone(),
                    Prompt::Text(_) => Vec::new(),
                },
                outputs: completion_outputs,
                finished: true,
            });
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

/// Build a single batched CompletionRequest from multiple prompts.
///
/// Groups text prompts as `Multiple` and token-ID prompts as `MultipleTokenIds`.
/// Mixed prompt types are not supported (token IDs take precedence if all are
/// token IDs, otherwise all are text).
fn build_batched_completion_request(
    prompts: &[Prompt],
    params: &SamplingParams,
    model: &str,
) -> protocol::CompletionRequest {
    let prompt = if prompts.len() == 1 {
        match &prompts[0] {
            Prompt::Text(text) => protocol::CompletionPrompt::Single(text.clone()),
            Prompt::TokenIds(ids) => protocol::CompletionPrompt::TokenIds(ids.clone()),
        }
    } else {
        // Check if all prompts are the same variant.
        let all_token_ids = prompts.iter().all(|p| matches!(p, Prompt::TokenIds(_)));
        if all_token_ids {
            let seqs: Vec<Vec<u32>> = prompts
                .iter()
                .map(|p| match p {
                    Prompt::TokenIds(ids) => ids.clone(),
                    Prompt::Text(_) => unreachable!(),
                })
                .collect();
            protocol::CompletionPrompt::MultipleTokenIds(seqs)
        } else {
            let texts: Vec<String> = prompts
                .iter()
                .map(|p| match p {
                    Prompt::Text(text) => text.clone(),
                    Prompt::TokenIds(ids) => format!("<token_ids:{}>", ids.len()),
                })
                .collect();
            protocol::CompletionPrompt::Multiple(texts)
        }
    };

    let stop = if params.stop.is_empty() {
        None
    } else {
        Some(protocol::StopCondition::Multiple(params.stop.clone()))
    };

    protocol::CompletionRequest {
        model: Some(model.to_string()),
        prompt: Some(prompt),
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
    fn test_build_batched_completion_request_single_text() {
        let params = SamplingParams {
            temperature: 0.7,
            max_tokens: Some(100),
            stop: vec!["END".to_string()],
            ..SamplingParams::default()
        };
        let prompts = vec![Prompt::Text("Hello".to_string())];
        let req = build_batched_completion_request(&prompts, &params, "test-model");
        assert_eq!(req.model, Some("test-model".to_string()));
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.max_tokens, Some(100));
        assert!(!req.stream);
        assert!(req.stop.is_some());
        assert!(matches!(
            req.prompt,
            Some(protocol::CompletionPrompt::Single(ref s)) if s == "Hello"
        ));
    }

    #[test]
    fn test_build_batched_completion_request_multiple_token_ids() {
        let params = SamplingParams {
            temperature: 0.7,
            max_tokens: Some(100),
            ignore_eos: true,
            ..SamplingParams::default()
        };
        let prompts = vec![
            Prompt::TokenIds(vec![10, 20, 30]),
            Prompt::TokenIds(vec![40, 50]),
        ];
        let req = build_batched_completion_request(&prompts, &params, "test-model");
        assert_eq!(req.model, Some("test-model".to_string()));
        assert!(matches!(
            req.prompt,
            Some(protocol::CompletionPrompt::MultipleTokenIds(ref seqs))
                if seqs.len() == 2 && seqs[0] == [10, 20, 30] && seqs[1] == [40, 50]
        ));
    }

    #[test]
    fn test_build_batched_completion_request_single_token_ids() {
        let params = SamplingParams::default();
        let prompts = vec![Prompt::TokenIds(vec![1, 2, 3])];
        let req = build_batched_completion_request(&prompts, &params, "m");
        assert!(matches!(
            req.prompt,
            Some(protocol::CompletionPrompt::TokenIds(ref ids)) if ids == &[1, 2, 3]
        ));
    }

    #[test]
    fn test_build_batched_completion_request_multiple_text() {
        let params = SamplingParams::default();
        let prompts = vec![Prompt::Text("Hello".into()), Prompt::Text("World".into())];
        let req = build_batched_completion_request(&prompts, &params, "m");
        assert!(matches!(
            req.prompt,
            Some(protocol::CompletionPrompt::Multiple(ref texts))
                if texts.len() == 2 && texts[0] == "Hello" && texts[1] == "World"
        ));
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

    #[test]
    fn test_prompt_from_str() {
        let p: Prompt = "hello".into();
        assert!(matches!(p, Prompt::Text(s) if s == "hello"));
    }

    #[test]
    fn test_prompt_from_string() {
        let p: Prompt = String::from("world").into();
        assert!(matches!(p, Prompt::Text(s) if s == "world"));
    }

    #[test]
    fn test_prompt_from_token_ids() {
        let p: Prompt = vec![1u32, 2, 3].into();
        assert!(matches!(p, Prompt::TokenIds(ids) if ids == [1, 2, 3]));
    }
}
