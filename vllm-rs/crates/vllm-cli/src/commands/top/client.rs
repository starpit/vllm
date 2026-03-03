// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! HTTP client for connecting to a running vLLM server.

use anyhow::{Context, Result};
use serde::Deserialize;
use tokio::sync::mpsc;

/// Mirrors [`vllm_serve::protocol::StatsResponse`] so vllm-cli doesn't
/// depend on vllm-serve at the type level.
#[derive(Debug, Clone, Deserialize)]
pub struct StatsResponse {
    pub model_name: String,
    pub version: String,
    pub requests_total: u64,
    #[allow(dead_code)]
    pub requests_success: u64,
    #[allow(dead_code)]
    pub requests_failed: u64,
    pub prompt_tokens_total: u64,
    pub output_tokens_total: u64,
    pub requests_active: i64,
    pub num_requests_running: f64,
    pub num_requests_waiting: f64,
    pub kv_cache_usage: f64,
    pub gpu_cache_blocks_used: i64,
    pub gpu_cache_blocks_total: i64,
    pub ttft_sum: f64,
    pub ttft_count: u64,
    pub itl_sum: f64,
    pub itl_count: u64,
    pub latency_sum: f64,
    pub latency_count: u64,
}

pub struct StatsClient {
    client: reqwest::Client,
    base_url: String,
}

impl StatsClient {
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .expect("failed to build reqwest client"),
            base_url: format!("http://{}:{}", host, port),
        }
    }

    pub async fn check_health(&self) -> Result<()> {
        let url = format!("{}/health", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .context("failed to connect to server")?;
        if !resp.status().is_success() {
            anyhow::bail!("health check returned status {}", resp.status());
        }
        Ok(())
    }

    pub async fn fetch_stats(&self) -> Result<StatsResponse> {
        let url = format!("{}/stats", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .context("failed to fetch /stats")?;
        if !resp.status().is_success() {
            anyhow::bail!(
                "/stats returned status {} — is --enable-metrics on?",
                resp.status()
            );
        }
        resp.json::<StatsResponse>()
            .await
            .context("failed to parse /stats JSON")
    }

    /// Connect to the `/stats/live` SSE endpoint and return a receiver
    /// that yields `StatsResponse` for each server-sent event.
    ///
    /// The SSE connection runs in a spawned task; dropping the receiver
    /// causes the task to exit on the next send attempt.
    pub fn subscribe_stats_live(&self, interval_ms: u64) -> mpsc::UnboundedReceiver<StatsResponse> {
        let (tx, rx) = mpsc::unbounded_channel();
        let url = format!("{}/stats/live?interval={}", self.base_url, interval_ms);
        // Build a new client without the 5s timeout (SSE is long-lived).
        let client = reqwest::Client::new();

        tokio::spawn(async move {
            loop {
                match Self::sse_loop(&client, &url, &tx).await {
                    Ok(()) => break, // receiver dropped
                    Err(_) => {
                        // Connection lost — retry after a short delay.
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    }
                }
            }
        });

        rx
    }

    /// Read SSE events from a long-lived connection, parse and send.
    /// Returns Ok(()) when the receiver is dropped, Err on connection failure.
    async fn sse_loop(
        client: &reqwest::Client,
        url: &str,
        tx: &mpsc::UnboundedSender<StatsResponse>,
    ) -> Result<()> {
        use reqwest::header;

        let resp = client
            .get(url)
            .header(header::ACCEPT, "text/event-stream")
            .send()
            .await
            .context("SSE connect failed")?;

        if !resp.status().is_success() {
            anyhow::bail!("/stats/live returned status {}", resp.status());
        }

        // Read the response body as a byte stream and parse SSE frames.
        let mut bytes_stream = resp.bytes_stream();
        let mut buf = String::new();

        use tokio_stream::StreamExt;
        while let Some(chunk) = bytes_stream.next().await {
            let chunk = chunk.context("SSE read error")?;
            buf.push_str(&String::from_utf8_lossy(&chunk));

            // SSE frames end with \n\n. Process all complete frames.
            while let Some(end) = buf.find("\n\n") {
                let frame = buf[..end].to_string();
                buf.drain(..end + 2);

                // Extract the "data:" line(s).
                let data: String = frame
                    .lines()
                    .filter_map(|line| line.strip_prefix("data:"))
                    .collect();

                if data.is_empty() {
                    continue; // keepalive or comment
                }

                if let Ok(stats) = serde_json::from_str::<StatsResponse>(&data)
                    && tx.send(stats).is_err()
                {
                    return Ok(()); // receiver dropped
                }
            }
        }

        anyhow::bail!("SSE stream ended")
    }
}
