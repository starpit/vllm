// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Axum HTTP server with OpenAI-compatible API routes.
//!
//! Provides endpoints:
//! - `POST /v1/chat/completions` — chat completion (streaming + non-streaming)
//! - `POST /v1/completions` — text completion (streaming + non-streaming)
//! - `GET  /v1/models` — list available models
//! - `GET  /health` — health check
//! - `GET  /version` — version info
//!
//! Port of: `vllm/entrypoints/openai/api_server.py` (subset)

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::classify::ServerErrorsFailureClass;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{Span, info};

use crate::engine::{AsyncEngine, StreamDelta};
use crate::protocol;

// ---------------------------------------------------------------------------
// Server configuration
// ---------------------------------------------------------------------------

/// Configuration for the HTTP server.
pub struct ServerConfig {
    /// Address to bind to (e.g., "0.0.0.0:8000").
    pub bind_address: String,

    /// vLLM version string.
    pub version: String,

    /// Whether CORS is enabled.
    pub cors_enabled: bool,

    /// Whether the `/metrics` endpoint is enabled.
    pub metrics_enabled: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0:8000".to_string(),
            version: "0.1.0-rust".to_string(),
            cors_enabled: true,
            metrics_enabled: false,
        }
    }
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

/// Shared application state.
pub struct AppState {
    pub engine: Arc<AsyncEngine>,
    pub config: ServerConfig,
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Build the axum router with all API routes.
pub fn build_router(state: Arc<AppState>) -> Router {
    let mut router = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .route("/version", get(version));

    if state.config.metrics_enabled {
        router = router.route("/metrics", get(metrics));
    }

    let mut router = router.with_state(state.clone());

    // Log requests at INFO level: method, URI, status, latency.
    router = router.layer(
        TraceLayer::new_for_http()
            .make_span_with(|request: &http::Request<_>| {
                tracing::info_span!(
                    "request",
                    method = %request.method(),
                    uri = %request.uri(),
                )
            })
            .on_response(
                |response: &http::Response<_>, latency: std::time::Duration, _span: &Span| {
                    info!(status = %response.status(), latency = ?latency, "response");
                },
            )
            .on_failure(
                |error: ServerErrorsFailureClass, latency: std::time::Duration, _span: &Span| {
                    tracing::error!(%error, latency = ?latency, "request failed");
                },
            ),
    );

    if state.config.cors_enabled {
        router = router.layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );
    }

    router
}

/// Start the HTTP server.
pub async fn serve(state: Arc<AppState>) -> Result<(), Box<dyn std::error::Error>> {
    let router = build_router(state.clone());
    let listener = tokio::net::TcpListener::bind(&state.config.bind_address).await?;
    info!(
        "vLLM Rust server listening on {}",
        state.config.bind_address
    );
    axum::serve(listener, router).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

/// POST /v1/chat/completions
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<protocol::ChatCompletionRequest>,
) -> Response {
    info!(
        "POST /v1/chat/completions: model={:?}, max_tokens={:?}, stream={}",
        request.model, request.max_tokens, request.stream
    );
    if request.stream {
        match state.engine.chat_completion_stream(request).await {
            Ok((request_id, model, rx)) => {
                stream_chat_response(request_id, model, rx).into_response()
            }
            Err(e) => e.into_response(),
        }
    } else {
        match state.engine.chat_completion(request).await {
            Ok(response) => Json(response).into_response(),
            Err(e) => e.into_response(),
        }
    }
}

/// POST /v1/completions
async fn completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<protocol::CompletionRequest>,
) -> Response {
    if request.stream {
        // Streaming completions: similar pattern to chat, simplified.
        // For now, return non-streaming even when stream=true.
        // TODO: Implement streaming completions.
        match state.engine.completion(request).await {
            Ok(response) => Json(response).into_response(),
            Err(e) => e.into_response(),
        }
    } else {
        match state.engine.completion(request).await {
            Ok(response) => Json(response).into_response(),
            Err(e) => e.into_response(),
        }
    }
}

/// GET /v1/models
async fn list_models(State(state): State<Arc<AppState>>) -> Json<protocol::ModelList> {
    let mut card = protocol::ModelCard::new(state.engine.model_name().to_string());
    card.max_model_len = Some(state.engine.max_model_len());
    Json(protocol::ModelList::new(vec![card]))
}

/// GET /health
async fn health() -> Json<protocol::HealthResponse> {
    Json(protocol::HealthResponse { status: "ok" })
}

/// GET /version
async fn version(State(state): State<Arc<AppState>>) -> Json<protocol::VersionResponse> {
    Json(protocol::VersionResponse {
        version: state.config.version.clone(),
    })
}

/// GET /metrics — Prometheus metrics endpoint.
async fn metrics() -> String {
    crate::metrics::VllmMetrics::global().encode()
}

// ---------------------------------------------------------------------------
// SSE streaming
// ---------------------------------------------------------------------------

/// Build an SSE stream for chat completions.
fn stream_chat_response(
    request_id: String,
    model: String,
    rx: tokio::sync::mpsc::UnboundedReceiver<StreamDelta>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let stream = UnboundedReceiverStream::new(rx).map(move |delta| {
        let finish_reason_str = delta.finish_reason.map(|r| r.to_string());

        // Use detokenized text if available, otherwise fall back to placeholders.
        let text = delta.text.unwrap_or_else(|| {
            use std::fmt::Write;
            let mut s = String::new();
            for id in &delta.new_token_ids {
                let _ = write!(s, "<token_{id}>");
            }
            s
        });

        let chunk = protocol::ChatCompletionStreamResponse::new(
            format!("chatcmpl-{}", request_id),
            model.clone(),
            vec![protocol::ChatCompletionResponseStreamChoice {
                index: delta.index,
                delta: protocol::DeltaMessage {
                    role: None,
                    content: if text.is_empty() { None } else { Some(text) },
                    reasoning: None,
                    tool_calls: None,
                },
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

    // Append [DONE] sentinel after the stream ends.
    let done_stream = tokio_stream::once(Ok(Event::default().data("[DONE]")));
    let full_stream = stream.chain(done_stream);

    Sse::new(full_stream).keep_alive(KeepAlive::default())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::StatusCode;
    use http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    use vllm_config::{SchedulerConfig, SchedulerPolicy};
    use vllm_engine::core_client::InprocClient;
    use vllm_engine::engine_core::EngineCoreConfig;
    use vllm_engine::executor::NoopExecutor;

    fn make_test_state() -> Arc<AppState> {
        let engine_config = EngineCoreConfig {
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
            eos_token_id: None,
        };
        let executor = Box::new(NoopExecutor::new(1024));
        let client = Box::new(InprocClient::new(engine_config, executor));
        let engine = Arc::new(AsyncEngine::new(client, "test-model".to_string(), 4096));

        Arc::new(AppState {
            engine,
            config: ServerConfig::default(),
        })
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let state = make_test_state();
        let app = build_router(state);

        let request = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["status"], "ok");
    }

    #[tokio::test]
    async fn test_version_endpoint() {
        let state = make_test_state();
        let app = build_router(state);

        let request = Request::builder()
            .uri("/version")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let parsed: protocol::VersionResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed.version, "0.1.0-rust");
    }

    #[tokio::test]
    async fn test_models_endpoint() {
        let state = make_test_state();
        let app = build_router(state);

        let request = Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let parsed: protocol::ModelList = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed.object, "list");
        assert_eq!(parsed.data.len(), 1);
        assert_eq!(parsed.data[0].id, "test-model");
        assert_eq!(parsed.data[0].max_model_len, Some(4096));
    }

    #[tokio::test]
    async fn test_chat_completions_bad_json() {
        let state = make_test_state();
        let app = build_router(state);

        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from("not json"))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        // Axum returns 400 for malformed JSON.
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_chat_completions_missing_messages() {
        let state = make_test_state();
        let app = build_router(state);

        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model": "test"}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        // Missing required field "messages" → 422
        assert_eq!(
            response.status(),
            StatusCode::UNPROCESSABLE_ENTITY,
            "actual status: {}",
            response.status()
        );
    }

    #[tokio::test]
    async fn test_completions_bad_json() {
        let state = make_test_state();
        let app = build_router(state);

        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from("{invalid"))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        // Axum returns 400 for malformed JSON.
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    fn make_test_state_with_metrics() -> Arc<AppState> {
        let engine_config = EngineCoreConfig {
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
            eos_token_id: None,
        };
        let executor = Box::new(NoopExecutor::new(1024));
        let client = Box::new(InprocClient::new(engine_config, executor));
        let engine = Arc::new(AsyncEngine::new(client, "test-model".to_string(), 4096));

        Arc::new(AppState {
            engine,
            config: ServerConfig {
                metrics_enabled: true,
                ..ServerConfig::default()
            },
        })
    }

    #[tokio::test]
    async fn test_metrics_endpoint_enabled() {
        let state = make_test_state_with_metrics();
        let app = build_router(state);

        let request = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("vllm_requests_total"));
    }

    #[tokio::test]
    async fn test_metrics_endpoint_disabled() {
        let state = make_test_state(); // default has metrics_enabled: false
        let app = build_router(state);

        let request = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_not_found() {
        let state = make_test_state();
        let app = build_router(state);

        let request = Request::builder()
            .uri("/nonexistent")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
