// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! HTTP client wrapper for E2E testing.

use anyhow::{Context, Result, bail};
use vllm_serve::protocol::{
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamResponse, CompletionRequest,
    CompletionResponse, ModelList, VersionResponse,
};

/// A thin HTTP client for talking to a running vLLM server.
pub struct Client {
    inner: reqwest::Client,
    base_url: String,
}

impl Client {
    /// Create a new client pointing at the given base URL.
    pub fn new(base_url: &str) -> Self {
        Self {
            inner: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// GET /health — returns true if the server is healthy.
    pub async fn health(&self) -> Result<bool> {
        let resp = self
            .inner
            .get(format!("{}/health", self.base_url))
            .send()
            .await?;
        Ok(resp.status().is_success())
    }

    /// GET /version — returns the version string.
    pub async fn version(&self) -> Result<VersionResponse> {
        let resp = self
            .inner
            .get(format!("{}/version", self.base_url))
            .send()
            .await?;
        resp.json()
            .await
            .context("failed to parse version response")
    }

    /// GET /v1/models — list available models.
    pub async fn list_models(&self) -> Result<ModelList> {
        let resp = self
            .inner
            .get(format!("{}/v1/models", self.base_url))
            .send()
            .await?;
        resp.json()
            .await
            .context("failed to parse model list response")
    }

    /// POST /v1/chat/completions — non-streaming chat completion.
    pub async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        assert!(!request.stream, "use chat_completion_stream for streaming");

        let resp = self
            .inner
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(request)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("chat completion failed with status {status}: {body}");
        }

        resp.json()
            .await
            .context("failed to parse chat completion response")
    }

    /// POST /v1/chat/completions with stream=true — returns collected stream chunks.
    pub async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<Vec<ChatCompletionStreamResponse>> {
        assert!(request.stream, "use chat_completion for non-streaming");

        let resp = self
            .inner
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(request)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("streaming chat completion failed with status {status}: {body}");
        }

        let body = resp.text().await?;
        parse_sse_chunks(&body)
    }

    /// POST /v1/completions — non-streaming text completion.
    pub async fn completion(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        let resp = self
            .inner
            .post(format!("{}/v1/completions", self.base_url))
            .json(request)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("completion failed with status {status}: {body}");
        }

        resp.json()
            .await
            .context("failed to parse completion response")
    }

    /// POST /v1/chat/completions — returns raw response for status code checking.
    pub async fn chat_completion_raw(&self, body: &serde_json::Value) -> Result<reqwest::Response> {
        let resp = self
            .inner
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(body)
            .send()
            .await?;
        Ok(resp)
    }

    /// GET /metrics — returns raw Prometheus text.
    pub async fn metrics(&self) -> Result<String> {
        let resp = self
            .inner
            .get(format!("{}/metrics", self.base_url))
            .send()
            .await?;
        resp.text().await.context("failed to read metrics")
    }
}

/// Parse SSE response body into individual JSON chunks.
fn parse_sse_chunks(body: &str) -> Result<Vec<ChatCompletionStreamResponse>> {
    let mut chunks = Vec::new();

    for line in body.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                break;
            }
            let chunk: ChatCompletionStreamResponse =
                serde_json::from_str(data).context(format!("failed to parse SSE chunk: {data}"))?;
            chunks.push(chunk);
        }
    }

    Ok(chunks)
}
