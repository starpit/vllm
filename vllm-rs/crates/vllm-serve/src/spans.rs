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

pub(crate) struct SpanConfig {
    pub(crate) pad_token: Option<u32>,
    pub(crate) block_size: usize,
}

impl SpanConfig {
    #[allow(dead_code)]
    pub(crate) fn with_pad_token(block_size: usize, pad_token: u32) -> Self {
        Self {
            pad_token: Some(pad_token),
            block_size,
        }
    }

    /// Resolve pad token from `VLLM_V1_SPANS_PAD_TOKEN` env var, falling back
    /// to the tokenizer's encoding of `" "` (whitespace), then 0.
    ///
    /// Set `VLLM_V1_SPANS_PAD_TOKEN=-1` to disable padding entirely.
    pub(crate) fn from_tokenizer(
        block_size: usize,
        _tokenizer: &crate::tokenizer::Tokenizer,
    ) -> Self {
        // Default: no padding. Padding between Plus children injects tokens
        // that corrupt text content and destroy model accuracy (verified by
        // NIAH bench). Set VLLM_V1_SPANS_PAD_TOKEN to a token ID to enable
        // padding (e.g. for synthetic benchmarks like `bench spans` that use
        // pre-aligned token sequences).
        let pad_token =
            std::env::var("VLLM_V1_SPANS_PAD_TOKEN")
                .ok()
                .and_then(|v| match v.parse::<i64>() {
                    Ok(-1) => None,
                    Ok(id) if id >= 0 => Some(id as u32),
                    _ => None,
                });

        tracing::info!("[SPANS] pad_token={pad_token:?}, block_size={block_size}");
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
    /// When true, the next message must NOT be consolidated with the previous
    /// one, even if they share the same role. Set at structural boundaries
    /// (e.g. Cross left→right) to preserve message turn structure.
    break_consolidation: bool,
    /// Messages rendered so far for incremental tokenization.
    /// Each `tokenize_message` call appends to this and re-renders the full
    /// conversation, taking only the delta tokens. This avoids spurious BOS
    /// tokens between messages in a Seq.
    rendered_messages: Vec<TemplateMessage>,
    /// Number of token IDs produced by the last full render of
    /// `rendered_messages` — the delta starts after this offset.
    rendered_token_count: usize,
}

impl TokenizeState {
    fn new() -> Self {
        Self {
            tokens: Vec::new(),
            annotations: BTreeMap::new(),
            in_relocatable: false,
            break_consolidation: false,
            rendered_messages: Vec::new(),
            rendered_token_count: 0,
        }
    }

    /// Pad `tokens` to the next block boundary (no-op if pad_token is None).
    fn pad_to_block(&mut self, cfg: &SpanConfig) {
        if let Some(pad_token) = cfg.pad_token {
            let remainder = self.tokens.len() % cfg.block_size;
            if remainder > 0 && remainder < cfg.block_size {
                let pad_count = cfg.block_size - remainder;
                self.tokens
                    .extend(std::iter::repeat_n(pad_token, pad_count));
            }
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
///
/// Uses incremental rendering: the message is appended to the conversation
/// accumulated in `state.rendered_messages`, the full conversation is
/// re-rendered, and only the delta tokens (new tokens beyond the previous
/// render) are added to `state.tokens`. This avoids spurious BOS tokens
/// when multiple messages appear in a Seq.
fn tokenize_message(
    msg: &Message,
    tokenizer: &Tokenizer,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    state: &mut TokenizeState,
) -> ServeResult<()> {
    let role = msg.role().to_string();
    let content = msg.content().to_string();

    // Consolidate consecutive messages with the same role — avoids inserting
    // redundant role header tokens between chunks of the same type (e.g.
    // multiple user messages from Plus children). Consolidation is suppressed
    // at structural boundaries (break_consolidation flag).
    let should_consolidate = !state.break_consolidation
        && state
            .rendered_messages
            .last()
            .is_some_and(|last| last.role == role);
    state.break_consolidation = false;

    if should_consolidate {
        let last = state.rendered_messages.last_mut().unwrap();
        last.content.push('\n');
        last.content.push_str(&content);
    } else {
        state
            .rendered_messages
            .push(TemplateMessage { role, content });
    }
    let rendered = template.apply_simple(&state.rendered_messages, false)?;
    let all_ids = tokenizer.encode(&rendered, false)?;

    // When consolidating, the previous rendering's template suffix (e.g.,
    // <|eot_id|> in Llama 3, <|im_end|>\n in ChatML) is still in
    // state.tokens but has shifted position in the new rendering. Fix by
    // detecting and removing the stale suffix before computing the delta.
    if should_consolidate && state.rendered_token_count > 0 {
        let old_count = state.rendered_token_count;
        let state_len = state.tokens.len();
        // Check up to 32 trailing tokens for stale template suffix.
        // Real tokenizers: 1-3 tokens (e.g., <|eot_id|> = 1, <|im_end|>\n = 2).
        // Byte-level test tokenizer: up to ~12 tokens.
        let max_check = old_count.min(state_len).min(32);
        let mut stale_count = 0;
        for k in 1..=max_check {
            let si = state_len - k;
            let ai = old_count - k;
            if ai < all_ids.len() && state.tokens[si] != all_ids[ai] {
                stale_count = k;
            } else {
                break;
            }
        }
        if stale_count > 0 {
            state.tokens.truncate(state_len - stale_count);
            state.rendered_token_count -= stale_count;
        }
    }

    // Take only the delta — tokens added by this message.
    let new_ids = &all_ids[state.rendered_token_count..];
    state.rendered_token_count = all_ids.len();

    match msg {
        Message::Assistant(_) => {
            // For assistant messages, crop to block boundary (drop suffix tokens).
            // This matches spnl's extend_crop behavior.
            let end = new_ids.len() + state.tokens.len();
            let nearest_block_boundary = end / cfg.block_size * cfg.block_size;
            let amount_to_crop =
                std::cmp::min(new_ids.len(), end.saturating_sub(nearest_block_boundary));
            let extra_end = new_ids.len() - amount_to_crop;
            state.tokens.extend_from_slice(&new_ids[..extra_end]);
        }
        _ => {
            state.tokens.extend_from_slice(new_ids);
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
                state.break_consolidation = true;
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
/// Uses the accumulated conversation in `state.rendered_messages` to extract
/// the generation prompt in context. If no messages have been accumulated
/// (e.g. after an annotation boundary reset), falls back to a dummy user
/// message to avoid template crashes on empty message lists.
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

    // Use the actual accumulated messages for correct gen prompt extraction.
    // If empty (e.g. after annotation boundary reset), use a dummy.
    let messages = if state.rendered_messages.is_empty() {
        vec![TemplateMessage {
            role: "user".to_string(),
            content: "x".to_string(),
        }]
    } else {
        state.rendered_messages.clone()
    };

    // Render the conversation with add_generation_prompt=true and take the
    // token delta from the current rendered_token_count. This ensures the
    // gen prompt tokens are computed identically to how tokenize_message
    // computes message deltas (full-string encoding, not isolated substring),
    // avoiding BPE boundary mismatches.
    let with = template.apply_simple(&messages, true)?;
    let all_ids = tokenizer.encode(&with, false)?;
    let new_ids = &all_ids[state.rendered_token_count..];
    if !new_ids.is_empty() {
        state.tokens.extend_from_slice(new_ids);
        state.rendered_token_count = all_ids.len();
    }
    Ok(())
}

/// Result of tokenizing a span query: tokens + block annotations.
pub(crate) struct SpanTokenized {
    pub(crate) tokens: Vec<u32>,
    pub(crate) annotations: Option<BTreeMap<usize, BlockKind>>,
}

/// Tokenize a full SingleGenerate into a token sequence with annotations.
pub(crate) fn tokenize_span_query(
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
pub(crate) fn tokenize_map_input(
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_template::ChatTemplate;

    /// Regression test for the stale template-suffix bug.
    ///
    /// When messages are consolidated (same role, appended with \n), the
    /// template's end-of-turn token (e.g., Llama 3's `<|eot_id|>` = 128009)
    /// from the PREVIOUS rendering remains in `state.tokens` even though it
    /// has shifted in the new rendering. Without the fix, these stale tokens
    /// corrupt the token sequence and destroy model accuracy.
    ///
    /// This test uses synthetic token arrays to exercise the fix logic
    /// directly, simulating what happens with a real tokenizer.
    #[test]
    fn test_stale_suffix_removal_on_consolidation() {
        // Simulate Llama 3 tokenization:
        // Step 1: [{system: "hi"}, {user: "A"}]
        //   Tokens: [BOS, SYS_HEAD, .., EOT, USR_HEAD, .., A, DOT, EOT]
        //   Simplified: [10, 20, 30, 999, 40, 50, 60, 70, 999]
        //   where 999 = <|eot_id|>
        let ids_step1: Vec<u32> = vec![10, 20, 30, 999, 40, 50, 60, 70, 999];

        // Step 2: [{system: "hi"}, {user: "A\nB"}] (consolidated)
        //   Tokens: [BOS, SYS_HEAD, .., EOT, USR_HEAD, .., A, DOT, NL, B, EOT]
        //   Simplified: [10, 20, 30, 999, 40, 50, 60, 70, 80, 90, 999]
        //   The content "A" is extended to "A\nB", so after "A" and "DOT" we
        //   get NL=80, B=90, then EOT=999 at the end.
        let ids_step2: Vec<u32> = vec![10, 20, 30, 999, 40, 50, 60, 70, 80, 90, 999];

        // Key observation: ids_step1 and ids_step2 share prefix [10..70].
        // ids_step1[8] = 999 (EOT), ids_step2[8] = 80 (NL) — MISMATCH.

        // Without fix: state.tokens = ids_step1, delta = ids_step2[9..] = [90, 999]
        let old_count = ids_step1.len(); // 9
        let delta_buggy = &ids_step2[old_count..]; // [90, 999]
        let mut buggy = ids_step1.clone();
        buggy.extend_from_slice(delta_buggy);
        // buggy = [10, 20, 30, 999, 40, 50, 60, 70, 999, 90, 999]
        // The stale 999 (EOT) at position 8 corrupts the sequence!

        assert_ne!(
            buggy, ids_step2,
            "Without fix, stale EOT (999) must cause a mismatch",
        );
        assert!(
            buggy.windows(3).any(|w| w == [999, 90, 999]),
            "Buggy sequence should have stale EOT between content: {:?}",
            buggy,
        );

        // With fix: detect stale suffix by scanning backwards.
        let mut fixed = ids_step1.clone();
        let state_len = fixed.len();
        let max_check = old_count.min(state_len).min(32);
        let mut stale_count = 0;
        for k in 1..=max_check {
            let si = state_len - k;
            let ai = old_count - k;
            if ai < ids_step2.len() && fixed[si] != ids_step2[ai] {
                stale_count = k;
            } else {
                break;
            }
        }
        assert_eq!(stale_count, 1, "Should detect 1 stale suffix token (EOT)");
        fixed.truncate(state_len - stale_count);
        let adjusted_count = old_count - stale_count;
        let delta_fixed = &ids_step2[adjusted_count..];
        fixed.extend_from_slice(delta_fixed);

        assert_eq!(
            fixed, ids_step2,
            "After fix, tokens must match one-shot rendering.\n\
             Fixed:    {:?}\n\
             Expected: {:?}",
            fixed, ids_step2,
        );
    }

    /// Verify the fix handles multi-token template suffixes (e.g., ChatML's
    /// `<|im_end|>\n` which is 2 tokens with a proper tokenizer).
    #[test]
    fn test_stale_suffix_removal_multi_token() {
        // Simulate ChatML: suffix = [IM_END=500, NL=10]
        let ids_step1: Vec<u32> = vec![1, 2, 3, 100, 200, 300, 500, 10];
        let ids_step2: Vec<u32> = vec![1, 2, 3, 100, 200, 300, 50, 400, 500, 10];
        // ids_step1[6] = 500 (IM_END), ids_step2[6] = 50 (content continuation)
        // ids_step1[7] = 10 (NL),      ids_step2[7] = 400 (more content)
        // Stale suffix = 2 tokens [500, 10]

        let old_count = ids_step1.len();
        let mut fixed = ids_step1.clone();
        let state_len = fixed.len();
        let max_check = old_count.min(state_len).min(32);
        let mut stale_count = 0;
        for k in 1..=max_check {
            let si = state_len - k;
            let ai = old_count - k;
            if ai < ids_step2.len() && fixed[si] != ids_step2[ai] {
                stale_count = k;
            } else {
                break;
            }
        }
        assert_eq!(stale_count, 2, "Should detect 2 stale suffix tokens");
        fixed.truncate(state_len - stale_count);
        let adjusted_count = old_count - stale_count;
        fixed.extend_from_slice(&ids_step2[adjusted_count..]);

        assert_eq!(fixed, ids_step2);
    }

    /// Verify the fix is a no-op when no consolidation occurs (different roles).
    #[test]
    fn test_no_stale_suffix_without_consolidation() {
        // When consecutive messages have different roles, there's no
        // consolidation and no stale suffix. The fix should be a no-op.
        let ids_step1: Vec<u32> = vec![10, 20, 30, 999]; // system msg
        let ids_step2: Vec<u32> = vec![10, 20, 30, 999, 40, 50, 60, 999]; // + user msg

        let old_count = ids_step1.len();
        // No consolidation → no suffix removal needed.
        // Delta = ids_step2[old_count..] = [40, 50, 60, 999]
        let mut result = ids_step1.clone();
        result.extend_from_slice(&ids_step2[old_count..]);
        assert_eq!(
            result, ids_step2,
            "No consolidation should produce correct tokens"
        );
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
