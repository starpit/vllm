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

use spnl_core::ir::{Generate, GenerateMetadata, Message, Query as SpnlQuery};
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

    /// Align `tokens` to the next block boundary.
    ///
    /// If a pad token is configured, pads forward. Otherwise, truncates
    /// backward to the current block boundary (dropping partial-block tail
    /// tokens). Truncation is safe for Relocatable fragments because the
    /// dropped tokens belong to the *previous* fragment's tail — losing a
    /// few tokens at the seam is far better than destroying all cache reuse.
    fn align_to_block(&mut self, cfg: &SpanConfig) {
        let remainder = self.tokens.len() % cfg.block_size;
        if remainder == 0 {
            return;
        }
        if let Some(pad_token) = cfg.pad_token {
            let pad_count = cfg.block_size - remainder;
            self.tokens
                .extend(std::iter::repeat_n(pad_token, pad_count));
        } else {
            // Truncate to current block boundary.
            self.tokens.truncate(self.tokens.len() - remainder);
        }
    }

    /// Align to block boundary and record an annotation for the next block.
    fn annotate_next_block(&mut self, kind: BlockKind, cfg: &SpanConfig) {
        self.align_to_block(cfg);
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
                state.break_consolidation = true;
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
    state.align_to_block(cfg);
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

async fn execute_query_inner(
    state: &Arc<AppState>,
    body: &str,
    stream: bool,
) -> ServeResult<Response> {
    let state_ref: &AppState = state.as_ref();
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

    // Try to parse as a full SpnlQuery first (supports nested generates).
    // Fall back to SingleGenerateQuery for backward compatibility.
    if let Ok(query) = serde_json::from_str::<SpnlQuery>(body) {
        // RAG: index any Augment nodes, then rewrite them to retrieved fragments.
        #[cfg(feature = "rag")]
        let query = {
            let mut aug_options = crate::augment::AugmentOptions {
                current_model: Some(state.engine.model_name().to_string()),
                embedder: Some(std::sync::Arc::new(
                    crate::augment::embed::AsyncEngineEmbedder::new(state.engine.clone()),
                )),
                tokenizer: state.engine.tokenizer().cloned(),
                sidecar_manager: Some(std::sync::Arc::new(crate::augment::SidecarManager::new())),
                ..Default::default()
            };
            aug_options.summarizer = Some(std::sync::Arc::new(
                crate::augment::summarize::AppStateSummarizer::new(std::sync::Arc::clone(state)),
            ));
            aug_options.apply_env_overrides();
            crate::augment::index(&query, &aug_options)
                .await
                .map_err(|e| ServeError::Validation(format!("RAG indexing failed: {e}")))?;
            optimize_augments(&query, &aug_options)
                .await
                .map_err(|e| ServeError::Validation(format!("RAG retrieval failed: {e}")))?
        };
        // HLO: insert prepare completions for Plus children.
        let query = hlo_insert_prepares(&query);
        return dispatch_spnl_query(
            state_ref, &query, stream, tokenizer, template, &cfg, block_size,
        )
        .await;
    }

    let query: SingleGenerateQuery = serde_json::from_str(body)
        .map_err(|e| ServeError::Validation(format!("Invalid span query: {e}")))?;

    match query {
        SingleGenerateQuery::SingleGenerate(spec) => {
            execute_single(state_ref, &spec, 1, stream, tokenizer, template, &cfg).await
        }
        SingleGenerateQuery::Bulk(Bulk::Repeat(Repeat { n, generate: spec })) => {
            execute_single(state_ref, &spec, n, stream, tokenizer, template, &cfg).await
        }
        SingleGenerateQuery::Bulk(Bulk::Map(map)) => {
            execute_map(state_ref, &map, stream, tokenizer, template, &cfg).await
        }
    }
}

// ---------------------------------------------------------------------------
// HLO — copied from spnl-run/src/optimizer/hlo.rs
// ---------------------------------------------------------------------------

/// Wrap a 1-token inner generate around a fragment (hlo.rs:48-70)
fn prepare_fragment(m: &SpnlQuery, parent_generate: &Generate) -> SpnlQuery {
    SpnlQuery::Generate(Generate {
        metadata: GenerateMetadata {
            model: parent_generate.metadata.model.clone(),
            max_tokens: Some(1),
            temperature: Some(0.0),
        },
        input: Box::new(m.clone()),
    })
}

/// Wrap a list of queries into a monad (hlo.rs:72-80)
fn prepare_monad(prepares: Vec<SpnlQuery>) -> Option<SpnlQuery> {
    if !prepares.is_empty() {
        Some(SpnlQuery::Monad(SpnlQuery::Plus(prepares).into()))
    } else {
        None
    }
}

/// Rewrite a Generate's input: find Plus nodes and insert prepares before them.
/// Non-recursive on the result — each Plus is transformed exactly once.
fn hlo_rewrite_generate_input(input: &SpnlQuery, g: &Generate) -> SpnlQuery {
    match input {
        SpnlQuery::Seq(v) => {
            let mut out = Vec::new();
            for child in v {
                if let SpnlQuery::Plus(fragments) = child {
                    // Insert Monad(Plus([prepare(f) for f])) before the Plus.
                    let prepares: Vec<_> =
                        fragments.iter().map(|m| prepare_fragment(m, g)).collect();
                    if let Some(monad) = prepare_monad(prepares) {
                        out.push(monad);
                    }
                    out.push(child.clone());
                } else {
                    out.push(child.clone());
                }
            }
            SpnlQuery::Seq(out)
        }
        SpnlQuery::Plus(fragments) => {
            // Plus directly as input (not in a Seq): wrap in Seq with prepares.
            let prepares: Vec<_> = fragments.iter().map(|m| prepare_fragment(m, g)).collect();
            SpnlQuery::Seq(
                [prepare_monad(prepares), Some(input.clone())]
                    .into_iter()
                    .flatten()
                    .collect(),
            )
        }
        _ => input.clone(),
    }
}

/// Top-level HLO: for a Generate with Plus in its input, insert prepares.
pub(crate) fn hlo_insert_prepares(query: &SpnlQuery) -> SpnlQuery {
    match query {
        SpnlQuery::Generate(g) => SpnlQuery::Generate(Generate {
            metadata: g.metadata.clone(),
            input: Box::new(hlo_rewrite_generate_input(&g.input, g)),
        }),
        _ => query.clone(),
    }
}

/// Dispatch a full `SpnlQuery` to the appropriate execution path.
async fn dispatch_spnl_query(
    state: &AppState,
    query: &SpnlQuery,
    stream: bool,
    tokenizer: &Arc<Tokenizer>,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    block_size: usize,
) -> ServeResult<Response> {
    match query {
        // Nested generate: outer Generate whose input contains inner Generate nodes.
        SpnlQuery::Generate(g) if has_nested_generates(&g.input) => {
            execute_nested_generate(state, g, stream, tokenizer, template, cfg, block_size).await
        }

        // Flat generate: no nested generates — convert to SingleGenerate and use existing path.
        SpnlQuery::Generate(g) => {
            let spec = outer_generate_to_single(g);
            execute_single(state, &spec, 1, stream, tokenizer, template, cfg).await
        }

        // Seq: execute each generate in sequence, collect all results.
        SpnlQuery::Seq(children) => {
            execute_seq_query(
                state, children, stream, tokenizer, template, cfg, block_size,
            )
            .await
        }

        // Bulk variants at the top level.
        SpnlQuery::Bulk(spnl_core::ir::Bulk::Repeat(r)) => {
            let spec = outer_generate_to_single(&r.generate);
            execute_single(state, &spec, r.n, stream, tokenizer, template, cfg).await
        }
        SpnlQuery::Bulk(spnl_core::ir::Bulk::Map(map)) => {
            execute_map(state, map, stream, tokenizer, template, cfg).await
        }

        // Monad: execute for side-effect (cache warming), discard output.
        SpnlQuery::Monad(inner) => {
            Box::pin(dispatch_spnl_query(
                state, inner, false, tokenizer, template, cfg, block_size,
            ))
            .await?;
            Ok(Json(serde_json::json!({"monad": true})).into_response())
        }

        // Plus at top level: execute each child (used by Monad for prepare batches).
        SpnlQuery::Plus(children) => {
            for child in children {
                Box::pin(dispatch_spnl_query(
                    state, child, false, tokenizer, template, cfg, block_size,
                ))
                .await?;
            }
            Ok(Json(serde_json::json!({"plus": true})).into_response())
        }

        other => Err(ServeError::Validation(format!(
            "Unsupported top-level query variant: {}",
            query_variant_name(other)
        ))),
    }
}

/// Execute a `Seq` of queries, collecting all generate results.
async fn execute_seq_query(
    state: &AppState,
    children: &[SpnlQuery],
    stream: bool,
    tokenizer: &Arc<Tokenizer>,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    block_size: usize,
) -> ServeResult<Response> {
    let mut steps: Vec<QueryStep> = Vec::new();
    for (i, child) in children.iter().enumerate() {
        match child {
            SpnlQuery::Generate(g) if has_nested_generates(&g.input) => {
                // Nested generate inside a Seq: execute it fully.
                // We can't stream intermediate results, so always use non-streaming here.
                let resp =
                    execute_nested_generate(state, g, false, tokenizer, template, cfg, block_size)
                        .await?;
                // Extract nested steps from the response body.
                let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
                    .await
                    .map_err(|e| ServeError::Internal(format!("body read: {e}")))?;
                let nested: NestedQueryResponse = serde_json::from_slice(&body)
                    .map_err(|e| ServeError::Internal(format!("nested parse: {e}")))?;
                steps.extend(nested.steps);
            }
            SpnlQuery::Generate(g) => {
                let spec = outer_generate_to_single(g);
                let resp = execute_single(state, &spec, 1, false, tokenizer, template, cfg).await?;
                let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
                    .await
                    .map_err(|e| ServeError::Internal(format!("body read: {e}")))?;
                let completion: protocol::CompletionResponse = serde_json::from_slice(&body)
                    .map_err(|e| ServeError::Internal(format!("completion parse: {e}")))?;
                steps.push(QueryStep {
                    label: format!("step[{i}]"),
                    response: completion,
                });
            }
            _ => {
                return Err(ServeError::Validation(format!(
                    "Seq child at index {i} is not a Generate — only Generate nodes are \
                     supported inside a top-level Seq query"
                )));
            }
        }
    }

    if stream {
        // Return steps as a single SSE event (streaming not meaningful for Seq).
        let json = serde_json::to_string(&NestedQueryResponse { steps })
            .map_err(|e| ServeError::Internal(format!("serialize: {e}")))?;
        let event_stream = tokio_stream::once(Ok::<Event, Infallible>(Event::default().data(json)));
        let done_stream = tokio_stream::once(Ok(Event::default().data("[DONE]")));
        Ok(Sse::new(event_stream.chain(done_stream))
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        Ok(Json(NestedQueryResponse { steps }).into_response())
    }
}

/// Return a short name for a query variant (for error messages).
fn query_variant_name(query: &SpnlQuery) -> &'static str {
    match query {
        SpnlQuery::Generate(_) => "Generate",
        SpnlQuery::Seq(_) => "Seq",
        SpnlQuery::Par(_) => "Par",
        SpnlQuery::Cross(_) => "Cross",
        SpnlQuery::Plus(_) => "Plus",
        SpnlQuery::Monad(_) => "Monad",
        SpnlQuery::Bulk(_) => "Bulk",
        SpnlQuery::Message(_) => "Message",
        SpnlQuery::Zip(_) => "Zip",
        #[cfg(feature = "rag")]
        SpnlQuery::Augment(_) => "Augment",
    }
}

/// Recursively rewrite `Augment` nodes into `Plus(Message(...))` fragments
/// by retrieving from the pre-built LEANN index.
#[cfg(feature = "rag")]
pub(crate) fn optimize_augments<'a>(
    query: &'a SpnlQuery,
    options: &'a crate::augment::AugmentOptions,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<SpnlQuery>> + Send + 'a>> {
    Box::pin(async move {
        match query {
            SpnlQuery::Augment(a) => {
                let fragments =
                    crate::augment::retrieve(&a.embedding_model, &a.body, &a.doc, options).await?;
                let fragment_nodes: Vec<SpnlQuery> = fragments
                    .into_iter()
                    .map(|s| SpnlQuery::Message(Message::User(s)))
                    .collect();
                Ok(SpnlQuery::Plus(fragment_nodes))
            }
            SpnlQuery::Generate(g) => {
                let optimized_input = Box::new(optimize_augments(&g.input, options).await?);
                Ok(SpnlQuery::Generate(Generate {
                    metadata: g.metadata.clone(),
                    input: optimized_input,
                }))
            }
            SpnlQuery::Seq(v) => {
                let mut out = Vec::with_capacity(v.len());
                for child in v {
                    out.push(optimize_augments(child, options).await?);
                }
                Ok(SpnlQuery::Seq(out))
            }
            SpnlQuery::Plus(v) => {
                let mut out = Vec::with_capacity(v.len());
                for child in v {
                    out.push(optimize_augments(child, options).await?);
                }
                Ok(SpnlQuery::Plus(out))
            }
            SpnlQuery::Cross(v) => {
                let mut out = Vec::with_capacity(v.len());
                for child in v {
                    out.push(optimize_augments(child, options).await?);
                }
                Ok(SpnlQuery::Cross(out))
            }
            other => Ok(other.clone()),
        }
    })
}

/// Execute a single (possibly n>1) generation from a tokenized span query.
async fn execute_single(
    state: &AppState,
    spec: &SingleGenerate,
    n: u8,
    stream: bool,
    tokenizer: &Arc<Tokenizer>,
    template: &crate::chat_template::ChatTemplate,
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

/// Like `execute_single` but returns the assistant's plain-text response.
///
/// Used by RAG indexing (e.g. RAPTOR phase 2 summarization) to drive a
/// single non-streaming generation through the in-process engine while
/// reusing the spans tokenization + scheduling pipeline.
#[cfg(feature = "rag")]
pub(crate) async fn execute_single_text(
    state: &AppState,
    spec: &SingleGenerate,
) -> ServeResult<String> {
    let tokenizer = state
        .engine
        .tokenizer()
        .ok_or_else(|| ServeError::Internal("execute_single_text requires a tokenizer".into()))?;
    let template = state.engine.chat_template().ok_or_else(|| {
        ServeError::Internal("execute_single_text requires a chat template".into())
    })?;
    let block_size = state
        .vllm_config
        .as_ref()
        .map(|c| c.block_size)
        .unwrap_or(16);
    let cfg = SpanConfig::from_tokenizer(block_size, tokenizer);

    let span_tok = tokenize_span_query(spec, tokenizer, template.as_ref(), &cfg)?;
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
        1,
        max_tokens,
        temperature,
        false,
    );

    let response = state.engine.completion(request).await?;
    Ok(response
        .choices
        .into_iter()
        .next()
        .map(|c| c.text)
        .unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
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

// ---------------------------------------------------------------------------
// Nested query execution
// ---------------------------------------------------------------------------

/// One generate step in a nested query result.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct QueryStep {
    /// Human-readable label, e.g. "inner[0]", "outer".
    pub label: String,
    /// The completion response for this step.
    #[serde(flatten)]
    pub response: protocol::CompletionResponse,
}

/// Response from `/v1/query/execute` for a nested (multi-generate) query.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct NestedQueryResponse {
    pub steps: Vec<QueryStep>,
}

/// Returns true if the query contains any nested `Generate` nodes.
fn has_nested_generates(query: &SpnlQuery) -> bool {
    match query {
        SpnlQuery::Generate(_) => true,
        SpnlQuery::Seq(v) | SpnlQuery::Par(v) | SpnlQuery::Cross(v) | SpnlQuery::Plus(v) => {
            v.iter().any(has_nested_generates)
        }
        SpnlQuery::Monad(inner) => has_nested_generates(inner),
        _ => false,
    }
}

/// Collect all `Generate` nodes from a query tree in DFS order.
fn collect_generates<'a>(query: &'a SpnlQuery, out: &mut Vec<&'a Generate>) {
    match query {
        SpnlQuery::Generate(g) => out.push(g),
        SpnlQuery::Seq(v) | SpnlQuery::Par(v) | SpnlQuery::Cross(v) | SpnlQuery::Plus(v) => {
            for child in v {
                collect_generates(child, out);
            }
        }
        SpnlQuery::Monad(inner) => collect_generates(inner, out),
        _ => {}
    }
}

/// Strip `Generate` nodes from a `SpnlQuery` tree, replacing each with an
/// empty `Seq`. Used to extract the non-generate message content of an outer
/// generate's input for tokenization.
fn strip_generates(query: &SpnlQuery) -> NonGenerateInput {
    match query {
        SpnlQuery::Generate(_) => NonGenerateInput::Seq(vec![]),
        SpnlQuery::Message(msg) => NonGenerateInput::Message(msg.clone()),
        SpnlQuery::Seq(v) => NonGenerateInput::Seq(v.iter().map(strip_generates).collect()),
        SpnlQuery::Par(v) => NonGenerateInput::Par(v.iter().map(strip_generates).collect()),
        SpnlQuery::Plus(v) => NonGenerateInput::Plus(v.iter().map(strip_generates).collect()),
        SpnlQuery::Cross(v) => NonGenerateInput::Cross(v.iter().map(strip_generates).collect()),
        SpnlQuery::Monad(inner) => strip_generates(inner),
        _ => NonGenerateInput::Seq(vec![]),
    }
}

/// Returns true if a `NonGenerateInput` tree has any leaf `Message` nodes.
fn non_generate_input_has_messages(input: &NonGenerateInput) -> bool {
    match input {
        NonGenerateInput::Message(_) => true,
        NonGenerateInput::Seq(v)
        | NonGenerateInput::Par(v)
        | NonGenerateInput::Plus(v)
        | NonGenerateInput::Cross(v) => v.iter().any(non_generate_input_has_messages),
    }
}

/// Convert a `SpnlQuery::Generate` (whose `input: Box<SpnlQuery>` may contain
/// nested `Generate` nodes) into a `SingleGenerate` by stripping inner
/// generates from the input tree.  The resulting `SingleGenerate` covers only
/// the non-generate message content of the outer input.
pub(crate) fn outer_generate_to_single(g: &Generate) -> SingleGenerate {
    SingleGenerate {
        metadata: g.metadata.clone(),
        input: strip_generates(&g.input),
    }
}

/// Synchronous nested query execution — used by `LLM::execute_query`.
///
/// Parses `spnl_json` as a full `SpnlQuery` and executes it, supporting nested
/// generates.  The `generate` closure wraps the caller's synchronous generate
/// path (e.g. `LLM::generate_impl`); it receives `(prompts, sampling_params,
/// seal, volatile)` and returns `Vec<RequestOutput>`.
///
/// Returns the outputs of the final (outermost) generate step.
/// Wrap a timed `generate` call into a single-step `QueryOutput`.
fn timed_generate(
    label: &str,
    prompts: &[crate::llm::Prompt],
    sp: Option<vllm_common::SamplingParams>,
    seal: bool,
    volatile: bool,
    generate: &mut impl FnMut(
        &[crate::llm::Prompt],
        Option<vllm_common::SamplingParams>,
        bool,
        bool,
    ) -> anyhow::Result<Vec<crate::llm::RequestOutput>>,
) -> anyhow::Result<crate::llm::QueryOutput> {
    let t0 = std::time::Instant::now();
    let outputs = generate(prompts, sp, seal, volatile)?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let steps = outputs
        .into_iter()
        .enumerate()
        .map(|(i, output)| {
            let lbl = if i == 0 {
                label.to_string()
            } else {
                format!("{label}[{i}]")
            };
            crate::llm::GenerateStep {
                label: lbl,
                output,
                elapsed_ms,
            }
        })
        .collect();
    Ok(crate::llm::QueryOutput { steps })
}

/// Execute a pre-parsed `SpnlQuery` synchronously (no JSON round-trip).
///
/// This is the primary entry point for callers that already have a typed query.
/// Handles RAG augmentation (when enabled) and dispatches to the appropriate
/// generate path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_spnl_struct_sync(
    query: SpnlQuery,
    params: Option<vllm_common::SamplingParams>,
    seal: bool,
    volatile: bool,
    #[cfg(feature = "rag")] aug_options: &crate::augment::AugmentOptions,
    tokenizer: &Arc<Tokenizer>,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    block_size: usize,
    generate: &mut impl FnMut(
        &[crate::llm::Prompt],
        Option<vllm_common::SamplingParams>,
        bool,
        bool,
    ) -> anyhow::Result<Vec<crate::llm::RequestOutput>>,
) -> anyhow::Result<crate::llm::QueryOutput> {
    #[cfg(feature = "rag")]
    let query = {
        // We use a `current_thread` runtime so that any await inside
        // `augment::index` resumes on the calling OS thread — required by
        // the `SyncClosureSummarizer` thread-local trick below.
        let rt = tokio::runtime::Handle::try_current().unwrap_or_else(|_| {
            let rt = Box::leak(Box::new(
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("failed to create current_thread runtime"),
            ));
            rt.handle().clone()
        });

        // Build a synchronous summarizer closure over the offline `generate`
        // FnMut. RAPTOR's per-cluster summary calls land here via
        // `SyncClosureSummarizer`, which reads the installed slot.
        let mut summarize_fn = |spec: &SingleGenerate| -> anyhow::Result<String> {
            let span_tok = tokenize_span_query(spec, tokenizer, template, cfg)
                .map_err(|e| anyhow::anyhow!("span tokenization failed: {e}"))?;
            let prompt = crate::llm::Prompt::TokenIds(span_tok.tokens);
            let sp = vllm_common::SamplingParams {
                max_tokens: spec
                    .metadata
                    .max_tokens
                    .filter(|&t| t > 0)
                    .map(|t| t as u32),
                temperature: spec.metadata.temperature.unwrap_or(0.0) as f64,
                ..Default::default()
            };
            let outputs = generate(&[prompt], Some(sp), false, false)?;
            Ok(outputs
                .first()
                .and_then(|r| r.outputs.first())
                .map(|o| o.text.clone())
                .unwrap_or_default())
        };

        let mut aug_options_local = aug_options.clone();
        aug_options_local.summarizer = Some(std::sync::Arc::new(
            crate::augment::summarize::SyncClosureSummarizer,
        ));

        crate::augment::summarize::with_sync_summarizer(
            &mut summarize_fn,
            || -> anyhow::Result<SpnlQuery> {
                rt.block_on(crate::augment::index(&query, &aug_options_local))
                    .map_err(|e| anyhow::anyhow!("RAG indexing failed: {e}"))?;
                rt.block_on(optimize_augments(&query, &aug_options_local))
            },
        )?
    };
    // HLO: insert prepare completions for Plus children.
    let query = hlo_insert_prepares(&query);
    dispatch_spnl_query_sync(
        &query, params, seal, volatile, tokenizer, template, cfg, block_size, generate,
    )
}

/// Execute a SPNL query from a JSON string. Parses as `SpnlQuery` first,
/// falls back to `SingleGenerateQuery` for backward compatibility.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_spnl_query_sync(
    spnl_json: &str,
    params: Option<vllm_common::SamplingParams>,
    seal: bool,
    volatile: bool,
    #[cfg(feature = "rag")] aug_options: &crate::augment::AugmentOptions,
    tokenizer: &Arc<Tokenizer>,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    block_size: usize,
    mut generate: impl FnMut(
        &[crate::llm::Prompt],
        Option<vllm_common::SamplingParams>,
        bool,
        bool,
    ) -> anyhow::Result<Vec<crate::llm::RequestOutput>>,
) -> anyhow::Result<crate::llm::QueryOutput> {
    use spnl_core::optimizer::llo::llir::{Bulk, Repeat, SingleGenerateQuery};

    // Try full SpnlQuery first (supports nested generates).
    if let Ok(query) = serde_json::from_str::<SpnlQuery>(spnl_json) {
        return execute_spnl_struct_sync(
            query,
            params,
            seal,
            volatile,
            #[cfg(feature = "rag")]
            aug_options,
            tokenizer,
            template,
            cfg,
            block_size,
            &mut generate,
        );
    }

    // Fallback: parse as SingleGenerateQuery for backward compatibility.
    let query: SingleGenerateQuery =
        serde_json::from_str(spnl_json).map_err(|e| anyhow::anyhow!("invalid SPNL query: {e}"))?;

    match query {
        SingleGenerateQuery::SingleGenerate(spec) => {
            let span_tok = tokenize_span_query(&spec, tokenizer, template, cfg)
                .map_err(|e| anyhow::anyhow!("span tokenization failed: {e}"))?;
            let mut sp = merge_spnl_params_sync(&spec.metadata, params);
            resolve_max_tokens_sync(&mut sp, span_tok.tokens.len(), block_size);
            let prompt = span_tok_to_prompt(span_tok);
            timed_generate("outer", &[prompt], Some(sp), seal, volatile, &mut generate)
        }
        SingleGenerateQuery::Bulk(Bulk::Repeat(Repeat { n, generate: spec })) => {
            let span_tok = tokenize_span_query(&spec, tokenizer, template, cfg)
                .map_err(|e| anyhow::anyhow!("span tokenization failed: {e}"))?;
            let mut sp = merge_spnl_params_sync(&spec.metadata, params);
            sp.n = n as u32;
            resolve_max_tokens_sync(&mut sp, span_tok.tokens.len(), block_size);
            let prompt = span_tok_to_prompt(span_tok);
            timed_generate("outer", &[prompt], Some(sp), seal, volatile, &mut generate)
        }
        SingleGenerateQuery::Bulk(Bulk::Map(map)) => {
            let mut prompts = Vec::with_capacity(map.inputs.len());
            for input_text in &map.inputs {
                let span_tok = tokenize_map_input(input_text, tokenizer, template, cfg)
                    .map_err(|e| anyhow::anyhow!("span tokenization failed: {e}"))?;
                prompts.push(span_tok_to_prompt(span_tok));
            }
            let mut sp = merge_spnl_params_sync(&map.metadata, params);
            if let Some(first) = prompts.first() {
                let len = match first {
                    crate::llm::Prompt::TokenIds(ids) => ids.len(),
                    crate::llm::Prompt::TokenIdsWithAnnotations(ids, _) => ids.len(),
                    crate::llm::Prompt::Text(_) => 0,
                };
                resolve_max_tokens_sync(&mut sp, len, block_size);
            }
            timed_generate("outer", &prompts, Some(sp), seal, volatile, &mut generate)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_spnl_query_sync(
    query: &SpnlQuery,
    params: Option<vllm_common::SamplingParams>,
    seal: bool,
    volatile: bool,
    tokenizer: &Arc<Tokenizer>,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    block_size: usize,
    generate: &mut impl FnMut(
        &[crate::llm::Prompt],
        Option<vllm_common::SamplingParams>,
        bool,
        bool,
    ) -> anyhow::Result<Vec<crate::llm::RequestOutput>>,
) -> anyhow::Result<crate::llm::QueryOutput> {
    match query {
        SpnlQuery::Generate(g) if has_nested_generates(&g.input) => execute_nested_generate_sync(
            g, params, seal, volatile, tokenizer, template, cfg, block_size, generate,
        ),
        SpnlQuery::Generate(g) => {
            let spec = outer_generate_to_single(g);
            let span_tok = tokenize_span_query(&spec, tokenizer, template, cfg)
                .map_err(|e| anyhow::anyhow!("span tokenization failed: {e}"))?;
            let mut sp = merge_spnl_params_sync(&spec.metadata, params);
            resolve_max_tokens_sync(&mut sp, span_tok.tokens.len(), block_size);
            let prompt = span_tok_to_prompt(span_tok);
            timed_generate("outer", &[prompt], Some(sp), seal, volatile, generate)
        }
        SpnlQuery::Seq(children) => {
            let mut last = crate::llm::QueryOutput { steps: Vec::new() };
            for child in children {
                last = dispatch_spnl_query_sync(
                    child, None, seal, volatile, tokenizer, template, cfg, block_size, generate,
                )?;
            }
            Ok(last)
        }
        SpnlQuery::Bulk(spnl_core::ir::Bulk::Repeat(r)) => {
            let spec = outer_generate_to_single(&r.generate);
            let span_tok = tokenize_span_query(&spec, tokenizer, template, cfg)
                .map_err(|e| anyhow::anyhow!("span tokenization failed: {e}"))?;
            let mut sp = merge_spnl_params_sync(&spec.metadata, params);
            sp.n = r.n as u32;
            resolve_max_tokens_sync(&mut sp, span_tok.tokens.len(), block_size);
            let prompt = span_tok_to_prompt(span_tok);
            timed_generate("outer", &[prompt], Some(sp), seal, volatile, generate)
        }
        SpnlQuery::Bulk(spnl_core::ir::Bulk::Map(map)) => {
            let mut prompts = Vec::with_capacity(map.inputs.len());
            for input_text in &map.inputs {
                let span_tok = tokenize_map_input(input_text, tokenizer, template, cfg)
                    .map_err(|e| anyhow::anyhow!("span tokenization failed: {e}"))?;
                prompts.push(span_tok_to_prompt(span_tok));
            }
            let mut sp = merge_spnl_params_sync(&map.metadata, params);
            if let Some(first) = prompts.first() {
                let len = match first {
                    crate::llm::Prompt::TokenIds(ids) => ids.len(),
                    crate::llm::Prompt::TokenIdsWithAnnotations(ids, _) => ids.len(),
                    crate::llm::Prompt::Text(_) => 0,
                };
                resolve_max_tokens_sync(&mut sp, len, block_size);
            }
            timed_generate("outer", &prompts, Some(sp), seal, volatile, generate)
        }
        // Monad: execute for side-effect (cache warming), discard output.
        SpnlQuery::Monad(inner) => {
            dispatch_spnl_query_sync(
                inner, None, seal, volatile, tokenizer, template, cfg, block_size, generate,
            )?;
            Ok(crate::llm::QueryOutput { steps: Vec::new() })
        }

        // Plus at top level: batch all Generate children as a single call.
        SpnlQuery::Plus(children) => {
            let mut prompts = Vec::new();
            let sp = vllm_common::SamplingParams {
                max_tokens: Some(1),
                temperature: 0.0,
                ..Default::default()
            };
            for child in children {
                if let SpnlQuery::Generate(g) = child {
                    let spec = outer_generate_to_single(g);
                    let span_tok = tokenize_span_query(&spec, tokenizer, template, cfg)
                        .map_err(|e| anyhow::anyhow!("prepare tokenization failed: {e}"))?;
                    prompts.push(span_tok_to_prompt(span_tok));
                }
            }
            if !prompts.is_empty() {
                let _ = generate(&prompts, Some(sp), false, false)?;
            }
            Ok(crate::llm::QueryOutput { steps: Vec::new() })
        }

        other => Err(anyhow::anyhow!(
            "unsupported top-level query variant: {}",
            query_variant_name(other)
        )),
    }
}

/// Sync nested generate: execute inner generates, build outer prompt, execute outer.
/// Uses raw token IDs from `RequestOutput` directly — no text round-tripping needed.
#[allow(clippy::too_many_arguments)]
fn execute_nested_generate_sync(
    outer_g: &Generate,
    params: Option<vllm_common::SamplingParams>,
    seal: bool,
    volatile: bool,
    tokenizer: &Arc<Tokenizer>,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    block_size: usize,
    generate: &mut impl FnMut(
        &[crate::llm::Prompt],
        Option<vllm_common::SamplingParams>,
        bool,
        bool,
    ) -> anyhow::Result<Vec<crate::llm::RequestOutput>>,
) -> anyhow::Result<crate::llm::QueryOutput> {
    let mut inner_gens: Vec<&Generate> = Vec::new();
    collect_generates(&outer_g.input, &mut inner_gens);

    let mut outer_tokens: Vec<u32> = Vec::new();
    let mut outer_annotations: BTreeMap<usize, BlockKind> = BTreeMap::new();
    let mut steps: Vec<crate::llm::GenerateStep> = Vec::new();

    // Execute each inner generate with seal=true, volatile=true.
    for (i, inner_g) in inner_gens.iter().enumerate() {
        let inner_input = strip_generates(&inner_g.input);
        let inner_spec = SingleGenerate {
            metadata: inner_g.metadata.clone(),
            input: inner_input,
        };
        let inner_tok = tokenize_span_query(&inner_spec, tokenizer, template, cfg)
            .map_err(|e| anyhow::anyhow!("inner span tokenization failed: {e}"))?;
        let inner_prompt_tokens = inner_tok.tokens.clone();

        let mut sp = merge_spnl_params_sync(&inner_g.metadata, None);
        resolve_max_tokens_sync(&mut sp, inner_tok.tokens.len(), block_size);
        let prompt = span_tok_to_prompt(inner_tok);

        let t0 = std::time::Instant::now();
        let results = generate(&[prompt], Some(sp), true, true)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Build inner block: prompt_token_ids + output_token_ids (includes EOS + pads).
        let block_idx = outer_tokens.len() / block_size;
        outer_annotations.insert(block_idx, BlockKind::Relocatable);
        outer_tokens.extend_from_slice(&inner_prompt_tokens);
        if let Some(result) = results.first() {
            if let Some(output) = result.outputs.first() {
                outer_tokens.extend_from_slice(&output.token_ids);
            }
            steps.push(crate::llm::GenerateStep {
                label: format!("inner[{i}]"),
                output: result.clone(),
                elapsed_ms,
            });
        }
    }

    // Tokenize non-generate messages from outer input as Prefixed.
    let outer_spec = outer_generate_to_single(outer_g);
    if non_generate_input_has_messages(&outer_spec.input) {
        let msg_start_block = outer_tokens.len() / block_size;
        outer_annotations.insert(msg_start_block, BlockKind::Prefixed);
        let msg_tok = tokenize_span_query(&outer_spec, tokenizer, template, cfg)
            .map_err(|e| anyhow::anyhow!("outer span tokenization failed: {e}"))?;
        if let Some(ann) = msg_tok.annotations {
            for (k, v) in ann {
                outer_annotations.insert(msg_start_block + k, v);
            }
        }
        outer_tokens.extend_from_slice(&msg_tok.tokens);
    }

    let mut sp = merge_spnl_params_sync(&outer_g.metadata, params);
    resolve_max_tokens_sync(&mut sp, outer_tokens.len(), block_size);
    let outer_prompt = crate::llm::Prompt::TokenIdsWithAnnotations(outer_tokens, outer_annotations);

    let t0 = std::time::Instant::now();
    let results = generate(&[outer_prompt], Some(sp), seal, volatile)?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if let Some(result) = results.into_iter().next() {
        steps.push(crate::llm::GenerateStep {
            label: "outer".to_string(),
            output: result,
            elapsed_ms,
        });
    }
    Ok(crate::llm::QueryOutput { steps })
}

/// Merge SPNL metadata with optional caller params (caller takes precedence).
///
/// When `caller` is `None` the metadata values are used directly, bypassing
/// `SamplingParams::default()` so that defaults like `max_tokens: Some(16)`
/// don't silently override metadata-specified values.
fn merge_spnl_params_sync(
    metadata: &spnl_core::ir::GenerateMetadata,
    caller: Option<vllm_common::SamplingParams>,
) -> vllm_common::SamplingParams {
    match caller {
        Some(mut sp) => {
            // Caller params take precedence; fill in only what caller left unset.
            if sp.max_tokens.is_none() {
                sp.max_tokens = metadata.max_tokens.filter(|&t| t > 0).map(|t| t as u32);
            }
            if sp.temperature == 0.0
                && let Some(t) = metadata.temperature
            {
                sp.temperature = t as f64;
            }
            sp
        }
        None => {
            // No caller — build from metadata, then apply defaults for the rest.
            vllm_common::SamplingParams {
                max_tokens: metadata.max_tokens.filter(|&t| t > 0).map(|t| t as u32),
                temperature: metadata.temperature.map(|t| t as f64).unwrap_or(0.0),
                ..vllm_common::SamplingParams::default()
            }
        }
    }
}

/// Resolve max_tokens against model context length (no-op placeholder; mirrors LLM::resolve_max_tokens).
fn resolve_max_tokens_sync(
    sp: &mut vllm_common::SamplingParams,
    _prompt_len: usize,
    _block_size: usize,
) {
    if sp.max_tokens.is_none() {
        sp.max_tokens = Some(2048);
    }
}

/// Convert `SpanTokenized` to a `Prompt`.
fn span_tok_to_prompt(span_tok: SpanTokenized) -> crate::llm::Prompt {
    match span_tok.annotations {
        Some(ann) if !ann.is_empty() => {
            crate::llm::Prompt::TokenIdsWithAnnotations(span_tok.tokens, ann)
        }
        _ => crate::llm::Prompt::TokenIds(span_tok.tokens),
    }
}

/// Execute a `SpnlQuery::Generate` that contains nested inner generates.
///
/// Algorithm (mirrors `bench spans --nested`):
/// 1. Collect all inner `Generate` nodes from `outer_g.input` (DFS order).
/// 2. Execute each inner generate with seal=true, volatile=true.
/// 3. Build the outer prompt:
///    - For each inner: [inner_prompt_tokens] + [re-tokenized output] + [EOS if stop]
///      as a Relocatable block.
///    - If the outer input has non-generate messages: tokenize them as Prefixed.
/// 4. Execute the outer generate.
/// 5. Return a `NestedQueryResponse` with all steps.
async fn execute_nested_generate(
    state: &AppState,
    outer_g: &Generate,
    stream: bool,
    tokenizer: &Arc<Tokenizer>,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    block_size: usize,
) -> ServeResult<Response> {
    // 1. Collect inner generates.
    let mut inner_gens: Vec<&Generate> = Vec::new();
    collect_generates(&outer_g.input, &mut inner_gens);

    let mut steps: Vec<QueryStep> = Vec::new();
    let mut outer_tokens: Vec<u32> = Vec::new();
    let mut outer_annotations: BTreeMap<usize, BlockKind> = BTreeMap::new();

    let eos_token_id = tokenizer.eos_token_id();

    // 2. Execute each inner generate.
    for (i, inner_g) in inner_gens.iter().enumerate() {
        // Convert inner Generate's input (Box<SpnlQuery>) to NonGenerateInput.
        // Inner generates must not themselves contain nested generates.
        let inner_input = strip_generates(&inner_g.input);
        let inner_spec = SingleGenerate {
            metadata: inner_g.metadata.clone(),
            input: inner_input,
        };

        // Tokenize the inner generate's prompt (includes generation prompt prefix).
        let inner_tok = tokenize_span_query(&inner_spec, tokenizer, template, cfg)?;
        let inner_prompt_tokens = inner_tok.tokens.clone();

        let max_tokens = inner_g
            .metadata
            .max_tokens
            .filter(|&t| t > 0)
            .map(|t| t as u32)
            .unwrap_or(2048);
        let temperature = inner_g.metadata.temperature.unwrap_or(0.0);

        // Execute inner generate: seal=true so output is block-aligned in KV cache.
        let mut request = build_completion_request(
            &inner_g.metadata.model,
            protocol::CompletionPrompt::TokenIds(inner_tok.tokens),
            inner_tok.annotations,
            1,
            max_tokens,
            temperature,
            false, // never stream inner generates
        );
        request.seal = true;
        request.volatile = true;
        // skip_special_tokens=true (default) so choices[0].text has no EOS/pads.

        let inner_response = state.engine.completion(request).await?;

        // 3a. Reconstruct inner token sequence for the outer prompt.
        //     inner_prompt_tokens + re_tokenize(output_text) + [EOS if stop]
        //     Seal guarantees this is block-aligned.
        let block_idx = outer_tokens.len() / block_size;
        outer_annotations.insert(block_idx, BlockKind::Relocatable);
        outer_tokens.extend_from_slice(&inner_prompt_tokens);

        if let Some(choice) = inner_response.choices.first() {
            // Re-tokenize the decoded output (round-trips correctly for BPE).
            let content_ids = tokenizer.encode(&choice.text, false)?;
            outer_tokens.extend_from_slice(&content_ids);

            // Append EOS if generation stopped naturally (not max_tokens).
            if choice.finish_reason.as_deref() == Some("stop")
                && let Some(eos) = eos_token_id
            {
                outer_tokens.push(eos);
            }
        }

        steps.push(QueryStep {
            label: format!("inner[{i}]"),
            response: inner_response,
        });
    }

    // 3b. Tokenize non-generate messages from the outer input (Prefixed).
    let outer_input_stripped = outer_generate_to_single(outer_g);
    if non_generate_input_has_messages(&outer_input_stripped.input) {
        let msg_start_block = outer_tokens.len() / block_size;
        outer_annotations.insert(msg_start_block, BlockKind::Prefixed);

        let msg_tok = tokenize_span_query(&outer_input_stripped, tokenizer, template, cfg)?;
        outer_tokens.extend_from_slice(&msg_tok.tokens);
        // Merge any annotations from msg_tok (offset by msg_start_block).
        if let Some(ann) = msg_tok.annotations {
            for (k, v) in ann {
                outer_annotations.insert(msg_start_block + k, v);
            }
        }
    }

    // 4. Execute the outer generate.
    let outer_max_tokens = outer_g
        .metadata
        .max_tokens
        .filter(|&t| t > 0)
        .map(|t| t as u32)
        .unwrap_or(2048);
    let outer_temperature = outer_g.metadata.temperature.unwrap_or(0.0);

    let outer_request = build_completion_request(
        &outer_g.metadata.model,
        protocol::CompletionPrompt::TokenIds(outer_tokens),
        Some(outer_annotations),
        1,
        outer_max_tokens,
        outer_temperature,
        stream,
    );

    // Nested queries always return a NestedQueryResponse (no streaming support for inner steps).
    let outer_response = state.engine.completion(outer_request).await?;
    steps.push(QueryStep {
        label: "outer".to_string(),
        response: outer_response,
    });
    Ok(Json(NestedQueryResponse { steps }).into_response())
}

/// Execute a map (bulk) completion: one output per input string.
async fn execute_map(
    state: &AppState,
    map: &spnl_core::ir::Map,
    stream: bool,
    tokenizer: &Arc<Tokenizer>,
    template: &crate::chat_template::ChatTemplate,
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
