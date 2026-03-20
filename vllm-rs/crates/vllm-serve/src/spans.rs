// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `/v1/query/execute` — execute a SPNL span query.
//!
//! Accepts a JSON span query, tokenizes it using the server's model tokenizer,
//! inserts span control tokens (plus/cross) at block boundaries, and submits
//! the resulting token sequence to the engine for generation.
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
// Span configuration (from environment variables)
// ---------------------------------------------------------------------------

struct SpanConfig {
    pad_token: u32,
    plus_token: Option<u32>,
    cross_token: Option<u32>,
    block_size: usize,
}

impl SpanConfig {
    fn from_env(block_size: usize) -> Self {
        let plus_token = std::env::var("VLLM_V1_SPANS_TOKEN_PLUS")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|&v| v >= 0)
            .map(|v| v as u32);

        let cross_token = std::env::var("VLLM_V1_SPANS_TOKEN_CROSS")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|&v| v >= 0)
            .map(|v| v as u32);

        let pad_token = std::env::var("VLLM_V1_SPANS_PAD_TOKEN")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(27); // default matches Python patch

        Self {
            pad_token,
            plus_token,
            cross_token,
            block_size,
        }
    }
}

// ---------------------------------------------------------------------------
// Tokenization helpers
// ---------------------------------------------------------------------------

/// Pad `tokens` to the next block boundary.
fn pad_to_block(tokens: &mut Vec<u32>, block_size: usize, pad_token: u32) {
    let remainder = tokens.len() % block_size;
    if remainder > 0 && remainder < block_size {
        let pad_count = block_size - remainder;
        tokens.extend(std::iter::repeat_n(pad_token, pad_count));
    }
}

/// Pad to block boundary, then push token.
fn pad_push(tokens: &mut Vec<u32>, token: u32, block_size: usize, pad_token: u32) {
    pad_to_block(tokens, block_size, pad_token);
    tokens.push(token);
}

/// Tokenize a single message using the chat template and tokenizer.
fn tokenize_message(
    msg: &Message,
    tokenizer: &Tokenizer,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    tokens: &mut Vec<u32>,
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
            let end = ids.len() + tokens.len();
            let nearest_block_boundary = end / cfg.block_size * cfg.block_size;
            let amount_to_crop =
                std::cmp::min(ids.len(), end.saturating_sub(nearest_block_boundary));
            let extra_end = ids.len() - amount_to_crop;
            tokens.extend_from_slice(&ids[..extra_end]);
        }
        _ => {
            tokens.extend_from_slice(&ids);
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
    tokens: &mut Vec<u32>,
) -> ServeResult<()> {
    match input {
        NonGenerateInput::Seq(v) | NonGenerateInput::Par(v) => {
            for child in v {
                tokenize_input(child, tokenizer, template, cfg, tokens)?;
            }
        }

        NonGenerateInput::Cross(v) => {
            // Add cross token prior to last entry (separates context from query).
            let (left, right) = v.split_at(v.len().saturating_sub(1));
            for child in left {
                tokenize_input(child, tokenizer, template, cfg, tokens)?;
            }
            if !right.is_empty() {
                if let Some(cross_token) = cfg.cross_token {
                    pad_push(tokens, cross_token, cfg.block_size, cfg.pad_token);
                }
                for child in right {
                    tokenize_input(child, tokenizer, template, cfg, tokens)?;
                }
            }
        }

        NonGenerateInput::Plus(v) => {
            for child in v {
                if let Some(plus_token) = cfg.plus_token {
                    pad_push(tokens, plus_token, cfg.block_size, cfg.pad_token);
                }
                tokenize_input(child, tokenizer, template, cfg, tokens)?;
            }
        }

        NonGenerateInput::Message(msg) => {
            tokenize_message(msg, tokenizer, template, cfg, tokens)?;
        }
    }
    Ok(())
}

/// Check if we are "in a plus" — there is a plus token with no following cross token.
fn in_plus(tokens: &[u32], cfg: &SpanConfig) -> bool {
    if let (Some(plus_token), Some(cross_token)) = (cfg.plus_token, cfg.cross_token) {
        for &token in tokens.iter().rev() {
            if token == cross_token {
                return false;
            } else if token == plus_token {
                return true;
            }
        }
    }
    false
}

/// Add the final assistant generation prompt token.
///
/// Many HuggingFace chat templates crash on an empty messages list, so we
/// render with a dummy user message both with and without
/// `add_generation_prompt`, then take the suffix that only appears in the
/// "with" variant — that's the assistant prompt.
fn add_generation_prompt(
    tokenizer: &Tokenizer,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    tokens: &mut Vec<u32>,
) -> ServeResult<()> {
    if in_plus(tokens, cfg)
        && let Some(plus_token) = cfg.plus_token
    {
        pad_push(tokens, plus_token, cfg.block_size, cfg.pad_token);
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
        tokens.extend_from_slice(&ids);
    }
    Ok(())
}

/// Tokenize a full SingleGenerate into a token sequence.
fn tokenize_span_query(
    spec: &SingleGenerate,
    tokenizer: &Tokenizer,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
) -> ServeResult<Vec<u32>> {
    let mut tokens = Vec::new();
    tokenize_input(&spec.input, tokenizer, template, cfg, &mut tokens)?;
    add_generation_prompt(tokenizer, template, cfg, &mut tokens)?;
    Ok(tokens)
}

/// Tokenize a map input (user message with plus token prefix, padded).
fn tokenize_map_input(
    text: &str,
    tokenizer: &Tokenizer,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
) -> ServeResult<Vec<u32>> {
    let mut tokens = Vec::new();
    if let Some(plus_token) = cfg.plus_token {
        pad_push(&mut tokens, plus_token, cfg.block_size, cfg.pad_token);
    }
    tokenize_message(
        &Message::User(text.to_string()),
        tokenizer,
        template,
        cfg,
        &mut tokens,
    )?;
    pad_to_block(&mut tokens, cfg.block_size, cfg.pad_token);
    Ok(tokens)
}

/// Build a CompletionRequest from tokenized prompt IDs and generation metadata.
fn build_completion_request(
    model: &str,
    prompt: protocol::CompletionPrompt,
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

    let cfg = SpanConfig::from_env(block_size);

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
    let prompt_ids = tokenize_span_query(spec, tokenizer, template, cfg)?;

    let max_tokens = spec
        .metadata
        .max_tokens
        .filter(|&t| t > 0)
        .map(|t| t as u32)
        .unwrap_or(2048);
    let temperature = spec.metadata.temperature.unwrap_or(0.0);

    let request = build_completion_request(
        &spec.metadata.model,
        protocol::CompletionPrompt::TokenIds(prompt_ids),
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
    for input_text in &map.inputs {
        all_ids.push(tokenize_map_input(input_text, tokenizer, template, cfg)?);
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
