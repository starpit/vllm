// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `/v1/query/execute` — execute a SPNL span query.
//!
//! Accepts a JSON span query, tokenizes it using the server's model tokenizer,
//! produces block annotations for relocatable caching, and submits the
//! resulting token sequence to the engine for generation.
//!
//! This is the Rust equivalent of the Python vLLM `/v1/query/execute` endpoint.
//! The query format matches the SPNL crate's `SingleGenerateQuery` schema.

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Query, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{error, info};

use spnl_core::ir::Message;
use spnl_core::optimizer::llo::llir::{
    Bulk, NonGenerateInput, Repeat, SingleGenerate, SingleGenerateQuery,
};

use crate::chat_template::TemplateMessage;
use crate::engine::StreamDelta;
use crate::error::{ServeError, ServeResult};
use crate::protocol;
use crate::server::AppState;
use crate::tokenizer::Tokenizer;

// ---------------------------------------------------------------------------
// Query parameters
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Deserialize)]
pub(crate) struct ExecuteQueryParams {
    #[serde(default)]
    stream: Option<bool>,
}

// ---------------------------------------------------------------------------
// Span configuration
// ---------------------------------------------------------------------------

struct SpanConfig {
    pad_token: u32,
    block_size: usize,
}

impl SpanConfig {
    #[allow(dead_code)]
    fn with_pad_token(block_size: usize, pad_token: u32) -> Self {
        Self {
            pad_token,
            block_size,
        }
    }

    /// Resolve pad token from `VLLM_V1_SPANS_PAD_TOKEN` env var, falling back
    /// to the tokenizer's encoding of `" "` (whitespace), then 0.
    fn from_tokenizer(block_size: usize, tokenizer: &crate::tokenizer::Tokenizer) -> Self {
        let pad_token = std::env::var("VLLM_V1_SPANS_PAD_TOKEN")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .or_else(|| tokenizer.token_to_id(" "))
            .unwrap_or(0);

        Self {
            pad_token,
            block_size,
        }
    }
}

// ---------------------------------------------------------------------------
// Tokenization state and helpers
// ---------------------------------------------------------------------------

use std::collections::BTreeMap;
use vllm_common::BlockKind;

/// Accumulated state during recursive tokenization.
struct TokenizeState {
    tokens: Vec<u32>,
    annotations: BTreeMap<usize, BlockKind>,
    /// Whether we're currently inside a relocatable context.
    in_relocatable: bool,
}

impl TokenizeState {
    fn new() -> Self {
        Self {
            tokens: Vec::new(),
            annotations: BTreeMap::new(),
            in_relocatable: false,
        }
    }

    /// Pad `tokens` to the next block boundary.
    fn pad_to_block(&mut self, cfg: &SpanConfig) {
        let remainder = self.tokens.len() % cfg.block_size;
        if remainder > 0 && remainder < cfg.block_size {
            let pad_count = cfg.block_size - remainder;
            self.tokens
                .extend(std::iter::repeat_n(cfg.pad_token, pad_count));
        }
    }

    /// Pad to block boundary and record an annotation for the next block.
    fn annotate_next_block(&mut self, kind: BlockKind, cfg: &SpanConfig) {
        self.pad_to_block(cfg);
        let block_index = self.tokens.len() / cfg.block_size;
        self.annotations.insert(block_index, kind);
    }
}

/// Tokenize a single message using the chat template and tokenizer.
fn tokenize_message(
    msg: &Message,
    tokenizer: &Tokenizer,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    state: &mut TokenizeState,
) -> ServeResult<()> {
    let tmpl_msg = TemplateMessage {
        role: msg.role().to_string(),
        content: msg.content().to_string(),
    };
    let rendered = template.apply_simple(&[tmpl_msg], false)?;
    let ids = tokenizer.encode(&rendered, false)?;

    match msg {
        Message::Assistant(_) => {
            // For assistant messages, crop to block boundary (drop suffix tokens).
            // This matches spnl's extend_crop behavior.
            let end = ids.len() + state.tokens.len();
            let nearest_block_boundary = end / cfg.block_size * cfg.block_size;
            let amount_to_crop =
                std::cmp::min(ids.len(), end.saturating_sub(nearest_block_boundary));
            let extra_end = ids.len() - amount_to_crop;
            state.tokens.extend_from_slice(&ids[..extra_end]);
        }
        _ => {
            state.tokens.extend_from_slice(&ids);
        }
    }
    Ok(())
}

/// Recursively tokenize a NonGenerateInput tree.
fn tokenize_input(
    input: &NonGenerateInput,
    tokenizer: &Tokenizer,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    state: &mut TokenizeState,
) -> ServeResult<()> {
    match input {
        NonGenerateInput::Seq(v) | NonGenerateInput::Par(v) => {
            for child in v {
                tokenize_input(child, tokenizer, template, cfg, state)?;
            }
        }

        NonGenerateInput::Cross(v) => {
            let (left, right) = v.split_at(v.len().saturating_sub(1));
            for child in left {
                tokenize_input(child, tokenizer, template, cfg, state)?;
            }
            if !right.is_empty() {
                state.in_relocatable = false;
                state.annotate_next_block(BlockKind::Prefixed, cfg);
                for child in right {
                    tokenize_input(child, tokenizer, template, cfg, state)?;
                }
            }
        }

        NonGenerateInput::Plus(v) => {
            let prev_in_relocatable = state.in_relocatable;
            for child in v {
                state.annotate_next_block(BlockKind::Relocatable, cfg);
                state.in_relocatable = true;
                tokenize_input(child, tokenizer, template, cfg, state)?;
            }
            state.in_relocatable = prev_in_relocatable;
        }

        NonGenerateInput::Message(msg) => {
            tokenize_message(msg, tokenizer, template, cfg, state)?;
        }
    }
    Ok(())
}

/// Add the final assistant generation prompt.
///
/// Many HuggingFace chat templates crash on an empty messages list, so we
/// render with a dummy user message both with and without
/// `add_generation_prompt`, then take the suffix that only appears in the
/// "with" variant — that's the assistant prompt.
fn add_generation_prompt(
    tokenizer: &Tokenizer,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    state: &mut TokenizeState,
) -> ServeResult<()> {
    // If we're inside a relocatable context, add another relocatable boundary
    // for the generation prompt so it gets its own block.
    if state.in_relocatable {
        state.annotate_next_block(BlockKind::Relocatable, cfg);
    }

    let dummy = TemplateMessage {
        role: "user".to_string(),
        content: "x".to_string(),
    };
    let without = template.apply_simple(std::slice::from_ref(&dummy), false)?;
    let with = template.apply_simple(&[dummy], true)?;

    // The generation prompt is the suffix that `with` has beyond `without`.
    let suffix = with.strip_prefix(&without).unwrap_or(&with);
    if !suffix.is_empty() {
        let ids = tokenizer.encode(suffix, false)?;
        state.tokens.extend_from_slice(&ids);
    }
    Ok(())
}

/// Result of tokenizing a span query: tokens + block annotations.
struct SpanTokenized {
    tokens: Vec<u32>,
    annotations: Option<BTreeMap<usize, BlockKind>>,
}

/// Tokenize a full SingleGenerate into a token sequence with annotations.
fn tokenize_span_query(
    spec: &SingleGenerate,
    tokenizer: &Tokenizer,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
) -> ServeResult<SpanTokenized> {
    let mut state = TokenizeState::new();
    tokenize_input(&spec.input, tokenizer, template, cfg, &mut state)?;
    add_generation_prompt(tokenizer, template, cfg, &mut state)?;
    let annotations = if state.annotations.is_empty() {
        None
    } else {
        Some(state.annotations)
    };
    Ok(SpanTokenized {
        tokens: state.tokens,
        annotations,
    })
}

/// Tokenize a map input (user message with relocatable prefix, padded).
fn tokenize_map_input(
    text: &str,
    tokenizer: &Tokenizer,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
) -> ServeResult<SpanTokenized> {
    let mut state = TokenizeState::new();
    state.annotate_next_block(BlockKind::Relocatable, cfg);
    tokenize_message(
        &Message::User(text.to_string()),
        tokenizer,
        template,
        cfg,
        &mut state,
    )?;
    state.pad_to_block(cfg);
    let annotations = if state.annotations.is_empty() {
        None
    } else {
        Some(state.annotations)
    };
    Ok(SpanTokenized {
        tokens: state.tokens,
        annotations,
    })
}

/// Build a CompletionRequest from tokenized prompt IDs and generation metadata.
fn build_completion_request(
    model: &str,
    prompt: protocol::CompletionPrompt,
    annotations: Option<BTreeMap<usize, BlockKind>>,
    n: u32,
    max_tokens: u32,
    temperature: f32,
    stream: bool,
) -> protocol::CompletionRequest {
    protocol::CompletionRequest {
        model: Some(model.to_string()),
        prompt: Some(prompt),
        echo: false,
        temperature: Some(temperature as f64),
        top_p: None,
        n,
        max_tokens: Some(max_tokens),
        stream,
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
        guided_grammar: None,
        allowed_token_ids: None,
        bad_words: None,
        truncate_prompt_tokens: None,
        block_annotations: annotations,
        seal: false,
        volatile: false,
    }
}

// ---------------------------------------------------------------------------
// SSE streaming (mirrors server.rs stream_completion_response)
// ---------------------------------------------------------------------------

fn stream_response(
    request_id: String,
    model: String,
    rx: tokio::sync::mpsc::UnboundedReceiver<StreamDelta>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let stream = UnboundedReceiverStream::new(rx).map(move |delta| {
        let finish_reason_str = delta.finish_reason.map(|r| r.to_string());

        let text = delta.text.unwrap_or_else(|| {
            use std::fmt::Write;
            let mut s = String::new();
            for id in &delta.new_token_ids {
                let _ = write!(s, "<token_{id}>");
            }
            s
        });

        let chunk = protocol::CompletionStreamResponse::new(
            format!("cmpl-{}", request_id),
            model.clone(),
            vec![protocol::CompletionResponseStreamChoice {
                index: delta.index,
                text,
                logprobs: None,
                finish_reason: finish_reason_str,
                stop_reason: delta.stop_reason.map(|sr| match sr {
                    vllm_common::StopReason::Token(id) => serde_json::Value::Number(id.into()),
                    vllm_common::StopReason::String(s) => serde_json::Value::String(s),
                }),
            }],
        );

        let data = serde_json::to_string(&chunk).unwrap_or_default();
        Ok(Event::default().data(data))
    });

    let done_stream = tokio_stream::once(Ok(Event::default().data("[DONE]")));
    let full_stream = stream.chain(done_stream);
    Sse::new(full_stream).keep_alive(KeepAlive::default())
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// POST /v1/query/execute
///
/// Accepts a JSON span query (SPNL format), tokenizes it with the model's
/// tokenizer, and executes it as a completion request.
///
/// Query params: `?stream=true` for SSE streaming.
pub(crate) async fn execute_query(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ExecuteQueryParams>,
    body: String,
) -> Response {
    let stream = params.stream.unwrap_or(false);
    info!("POST /v1/query/execute (stream={})", stream);

    match execute_query_inner(&state, &body, stream).await {
        Ok(response) => response,
        Err(e) => {
            error!("/v1/query/execute failed: {e}");
            e.into_response()
        }
    }
}

async fn execute_query_inner(state: &AppState, body: &str, stream: bool) -> ServeResult<Response> {
    let tokenizer = state
        .engine
        .tokenizer()
        .ok_or_else(|| ServeError::Validation("Tokenizer not available".to_string()))?;

    let template = state
        .engine
        .chat_template()
        .ok_or_else(|| ServeError::Validation("Chat template not available".to_string()))?;

    let block_size = state
        .vllm_config
        .as_ref()
        .map(|c| c.block_size)
        .unwrap_or(16);

    let cfg = SpanConfig::from_tokenizer(block_size, tokenizer);

    let query: SingleGenerateQuery = serde_json::from_str(body)
        .map_err(|e| ServeError::Validation(format!("Invalid span query: {e}")))?;

    match query {
        SingleGenerateQuery::SingleGenerate(spec) => {
            execute_single(state, &spec, 1, stream, tokenizer, template, &cfg).await
        }
        SingleGenerateQuery::Bulk(Bulk::Repeat(Repeat { n, generate: spec })) => {
            execute_single(state, &spec, n, stream, tokenizer, template, &cfg).await
        }
        SingleGenerateQuery::Bulk(Bulk::Map(map)) => {
            execute_map(state, &map, stream, tokenizer, template, &cfg).await
        }
    }
}

/// Execute a single (possibly n>1) generation from a tokenized span query.
async fn execute_single(
    state: &AppState,
    spec: &SingleGenerate,
    n: u8,
    stream: bool,
    tokenizer: &Arc<Tokenizer>,
    template: &Arc<crate::chat_template::ChatTemplate>,
    cfg: &SpanConfig,
) -> ServeResult<Response> {
    let span_tok = tokenize_span_query(spec, tokenizer, template, cfg)?;

    let max_tokens = spec
        .metadata
        .max_tokens
        .filter(|&t| t > 0)
        .map(|t| t as u32)
        .unwrap_or(2048);
    let temperature = spec.metadata.temperature.unwrap_or(0.0);

    let request = build_completion_request(
        &spec.metadata.model,
        protocol::CompletionPrompt::TokenIds(span_tok.tokens),
        span_tok.annotations,
        n.max(1) as u32,
        max_tokens,
        temperature,
        stream,
    );

    if stream {
        let (request_id, model, rx) = state.engine.completion_stream(request).await?;
        Ok(stream_response(request_id, model, rx).into_response())
    } else {
        let response = state.engine.completion(request).await?;
        Ok(Json(response).into_response())
    }
}

/// Execute a map (bulk) completion: one output per input string.
async fn execute_map(
    state: &AppState,
    map: &spnl_core::ir::Map,
    stream: bool,
    tokenizer: &Arc<Tokenizer>,
    template: &Arc<crate::chat_template::ChatTemplate>,
    cfg: &SpanConfig,
) -> ServeResult<Response> {
    let mut all_ids: Vec<Vec<u32>> = Vec::with_capacity(map.inputs.len());
    // All map inputs get the same annotation structure (single relocatable block 0).
    let mut combined_annotations: Option<BTreeMap<usize, BlockKind>> = None;
    for input_text in &map.inputs {
        let span_tok = tokenize_map_input(input_text, tokenizer, template, cfg)?;
        all_ids.push(span_tok.tokens);
        if combined_annotations.is_none() {
            combined_annotations = span_tok.annotations;
        }
    }

    let max_tokens = map
        .metadata
        .max_tokens
        .filter(|&t| t > 0)
        .map(|t| t as u32)
        .unwrap_or(2048);
    let temperature = map.metadata.temperature.unwrap_or(0.0);

    let request = build_completion_request(
        &map.metadata.model,
        protocol::CompletionPrompt::MultipleTokenIds(all_ids),
        combined_annotations,
        1,
        max_tokens,
        temperature,
        stream,
    );

    if stream {
        let (request_id, model, rx) = state.engine.completion_stream(request).await?;
        Ok(stream_response(request_id, model, rx).into_response())
    } else {
        let response = state.engine.completion(request).await?;
        Ok(Json(response).into_response())
    }
}
