// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Interactive `chat` and `complete` CLI subcommands.
//!
//! `vllm chat` supports two modes:
//! - **In-process** (with `--model`): loads the model locally using the `LLM`
//!   API and runs inference directly — no server needed.
//! - **Remote** (without `--model`): connects to a running vLLM server's
//!   OpenAI-compatible API for streaming chat completions.
//!
//! `vllm complete` always uses remote mode (connects to a running server).

use std::io::{self, BufRead, Write};

use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::Value;

use crate::args::{ChatArgs, CompleteArgs};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn read_line(prompt: &str) -> Option<String> {
    print!("{prompt}");
    io::stdout().flush().ok();
    let stdin = io::stdin();
    let mut line = String::new();
    match stdin.lock().read_line(&mut line) {
        Ok(0) => None,
        Ok(_) => Some(line.trim_end().to_string()),
        Err(_) => None,
    }
}

// ---------------------------------------------------------------------------
// In-process chat (LLM API)
// ---------------------------------------------------------------------------

/// Collect per-token timestamps for benchmarking.
struct BenchStats {
    startup_ms: f64,
    first_token: Option<std::time::Instant>,
    token_times: Vec<std::time::Instant>,
    gen_start: std::time::Instant,
}

impl BenchStats {
    fn new(startup_ms: f64) -> Self {
        Self {
            startup_ms,
            first_token: None,
            token_times: Vec::new(),
            gen_start: std::time::Instant::now(),
        }
    }

    fn record_token(&mut self) {
        let now = std::time::Instant::now();
        if self.first_token.is_none() {
            self.first_token = Some(now);
        }
        self.token_times.push(now);
    }

    fn print(&self, num_output_tokens: usize) {
        eprintln!();
        eprintln!("--- bench ---");
        eprintln!("startup     : {:.1} ms", self.startup_ms);

        if let Some(first) = self.first_token {
            let ttft = first.duration_since(self.gen_start).as_secs_f64() * 1000.0;
            eprintln!("TTFT        : {ttft:.1} ms");
        }

        if self.token_times.len() >= 2 {
            let itls: Vec<f64> = self
                .token_times
                .windows(2)
                .map(|w| w[1].duration_since(w[0]).as_secs_f64() * 1000.0)
                .collect();
            let mean_itl = itls.iter().sum::<f64>() / itls.len() as f64;
            eprintln!("mean ITL    : {mean_itl:.1} ms");
        }

        // Decode throughput: exclude TTFT, measure from first to last token.
        if let (Some(first), Some(last)) = (self.first_token, self.token_times.last()) {
            let decode_secs = last.duration_since(first).as_secs_f64();
            // num_output_tokens includes the first token, but decode_secs
            // starts after the first token, so we measure (N-1) intervals.
            if decode_secs > 0.0 && num_output_tokens > 1 {
                let tps = (num_output_tokens - 1) as f64 / decode_secs;
                eprintln!("tok/sec     : {tps:.1}");
            }
        }

        eprintln!("output toks : {num_output_tokens}");
        eprintln!("-------------");
    }
}

fn run_chat_inproc(args: &ChatArgs, model: &str) -> Result<()> {
    use vllm_serve::llm::{ChatMessage, LLM};

    if let Ok(level) = std::env::var("RUST_LOG") {
        vllm_common::telemetry::init_tracing(&level);
    }

    let t0 = std::time::Instant::now();

    let mut builder = LLM::builder(model).device(&args.device).dtype(&args.dtype);
    // Chat is strictly sequential — one blocking chat_stream call at a
    // time (REPL, -q, multi-prompt, and --bench alike), so exactly one
    // sequence is ever running. Declare that instead of inheriting the
    // server default (256): hybrid GDN arches reserve a recurrent-state
    // slot per max_num_seqs up-front (~61 MiB/slot on Qwen3.5-35B —
    // 15.7 GiB at the default, which is the difference between the 35B
    // fitting on a 32 GiB box or failing the budget guard).
    builder = builder.max_num_seqs(1);
    if let Some(ref token) = args.hf_token {
        builder = builder.hf_token(token);
    }
    if let Some(ref gguf) = args.gguf_file {
        builder = builder.gguf_file(gguf);
    }
    if let Some(len) = args.max_model_len {
        builder = builder.max_model_len(len);
    }
    // VLLM_GPU_MEMORY_UTILIZATION env override: lets perf-debugging
    // workflows shrink the KV cache without touching the API. Defaults
    // to 0.9 (LLM builder default) when unset.
    if let Ok(s) = std::env::var("VLLM_GPU_MEMORY_UTILIZATION")
        && let Ok(f) = s.parse::<f64>()
    {
        builder = builder.gpu_memory_utilization(f);
    }
    builder = builder.tensor_parallel_size(args.tensor_parallel_size);
    builder = builder.enforce_eager(args.enforce_eager);
    if let Some(ref tpl) = args.chat_template {
        builder = builder.chat_template(tpl.clone());
    }
    let mut llm = builder.build()?;
    let startup_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!("Using model: {}", llm.model_name());

    let mut conversation: Vec<ChatMessage> = Vec::new();
    if let Some(ref system_prompt) = args.system_prompt {
        conversation.push(ChatMessage::system(system_prompt));
    }

    // Match Python's `vllm chat`: omit max_tokens so the server resolves it
    // to the full remaining context window, letting the model emit EOS
    // naturally. Base = the model's generation_config.json defaults
    // (temperature/top_p/top_k/min_p/repetition_penalty — e.g. Qwen3.5
    // thinkers ship 1.0/0.95/20 and loop endlessly without them);
    // explicit CLI flags override.
    let params = {
        let mut p = llm.default_sampling_params();
        // None => resolve to the full remaining window. Deliberately NOT
        // `.or(p.max_tokens)`: the struct default is the OpenAI 16,
        // which would silently cap chat at 16 tokens. Only an explicit
        // model recommendation (generation_config max_new_tokens) wins.
        p.max_tokens = args.max_tokens.or(llm.generation_max_new_tokens());
        if let Some(t) = args.temperature {
            p.temperature = t;
        }
        Some(p)
    };

    // Non-interactive mode: --prompt (multi-turn) or --quick (single turn).
    let prompts: Vec<String> = if !args.prompt.is_empty() {
        args.prompt.clone()
    } else if let Some(ref q) = args.quick {
        vec![q.clone()]
    } else {
        vec![]
    };
    if !prompts.is_empty() {
        // In --bench mode, do an untimed warmup first.
        if args.bench {
            conversation.push(ChatMessage::user(&prompts[0]));
            eprint!("(warmup) ");
            let warmup_params = vllm_serve::llm::SamplingParams {
                max_tokens: Some(1),
                ..Default::default()
            };
            let _ = llm.chat_stream(&conversation, Some(warmup_params), |_| {});
            eprintln!("done");
            conversation.pop();
        }

        for (i, message) in prompts.iter().enumerate() {
            conversation.push(ChatMessage::user(message));

            let mut stats = args.bench.then(|| BenchStats::new(startup_ms));

            if prompts.len() > 1 {
                eprintln!("[turn {}] {}", i + 1, message);
            }
            let output = llm.chat_stream(&conversation, params.clone(), |token| {
                print!("{token}");
                io::stdout().flush().ok();
                if let Some(ref mut s) = stats {
                    s.record_token();
                }
            })?;
            println!();

            // DIAGNOSTIC: print token IDs + finish reason so we can
            // see what the model actually produced (vs garbage tokens
            // or EOS).
            if std::env::var_os("VLLM_PRINT_TOKEN_IDS").is_some() {
                eprintln!(
                    "[diag] token_ids={:?} text={:?} finish_reason={:?}",
                    output.outputs[0].token_ids,
                    output.outputs[0].text,
                    output.outputs[0].finish_reason,
                );
            }

            if let Some(s) = stats {
                s.print(output.outputs[0].token_ids.len());
            }

            conversation.push(ChatMessage::assistant(&output.outputs[0].text));
        }
        return Ok(());
    }

    println!("Please enter a message for the chat model:");
    while let Some(input) = read_line("> ") {
        if input.is_empty() {
            continue;
        }
        conversation.push(ChatMessage::user(&input));

        let mut stats = args.bench.then(|| BenchStats::new(startup_ms));

        let output = llm.chat_stream(&conversation, params.clone(), |token| {
            print!("{token}");
            io::stdout().flush().ok();
            if let Some(ref mut s) = stats {
                s.record_token();
            }
        })?;
        println!();

        if let Some(s) = stats {
            s.print(output.outputs[0].token_ids.len());
        }

        conversation.push(ChatMessage::assistant(&output.outputs[0].text));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Remote chat (OpenAI-compatible API)
// ---------------------------------------------------------------------------

fn build_client(api_key: &str) -> Client {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::AUTHORIZATION,
        format!("Bearer {api_key}")
            .parse()
            .expect("valid header value"),
    );
    Client::builder()
        .default_headers(headers)
        .build()
        .expect("failed to build HTTP client")
}

async fn resolve_model_remote(
    client: &Client,
    base_url: &str,
    explicit: Option<&str>,
) -> Result<String> {
    if let Some(name) = explicit {
        return Ok(name.to_string());
    }
    let resp = client
        .get(format!("{base_url}/models"))
        .send()
        .await
        .context("failed to list models from server")?;
    let body: Value = resp.json().await.context("invalid JSON from /models")?;
    body["data"][0]["id"]
        .as_str()
        .map(|s| s.to_string())
        .context("no models available on the server")
}

/// Stream SSE using chunked transfer — reads line-by-line from the response
/// body for true streaming output.
async fn stream_sse_chunked(
    resp: reqwest::Response,
    extract_content: fn(&Value) -> Option<&str>,
) -> Result<String> {
    use futures_util::StreamExt;

    let mut output = String::new();
    let mut buf = String::new();
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("error reading SSE stream")?;
        buf.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(newline_pos) = buf.find('\n') {
            let line = buf[..newline_pos].trim().to_string();
            buf = buf[newline_pos + 1..].to_string();

            if !line.starts_with("data: ") {
                continue;
            }
            let data = &line[6..];
            if data == "[DONE]" {
                println!();
                return Ok(output);
            }
            if let Ok(chunk) = serde_json::from_str::<Value>(data)
                && let Some(content) = extract_content(&chunk)
            {
                output.push_str(content);
                print!("{content}");
                io::stdout().flush().ok();
            }
        }
    }
    println!();
    Ok(output)
}

fn extract_chat_content(chunk: &Value) -> Option<&str> {
    chunk["choices"][0]["delta"]["content"].as_str()
}

fn extract_completion_text(chunk: &Value) -> Option<&str> {
    chunk["choices"][0]["text"].as_str()
}

async fn run_chat_remote(args: &ChatArgs) -> Result<()> {
    let api_key = args
        .api_key
        .as_deref()
        .or(std::env::var("OPENAI_API_KEY").ok().as_deref())
        .unwrap_or("EMPTY")
        .to_string();

    let client = build_client(&api_key);
    let model = resolve_model_remote(&client, &args.url, args.model_name.as_deref()).await?;
    println!("Using model: {model}");

    let mut conversation: Vec<Value> = Vec::new();
    if let Some(ref system_prompt) = args.system_prompt {
        conversation.push(serde_json::json!({
            "role": "system",
            "content": system_prompt,
        }));
    }

    let mut chat_body = serde_json::json!({
        "model": model,
        "stream": true,
    });
    if let Some(mt) = args.max_tokens {
        chat_body["max_tokens"] = serde_json::json!(mt);
    }

    if let Some(ref message) = args.quick {
        conversation.push(serde_json::json!({
            "role": "user",
            "content": message,
        }));
        chat_body["messages"] = serde_json::json!(conversation);
        let resp = client
            .post(format!("{}/chat/completions", args.url))
            .json(&chat_body)
            .send()
            .await
            .context("failed to send chat completion request")?;
        stream_sse_chunked(resp, extract_chat_content).await?;
        return Ok(());
    }

    println!("Please enter a message for the chat model:");
    while let Some(input) = read_line("> ") {
        if input.is_empty() {
            continue;
        }
        conversation.push(serde_json::json!({
            "role": "user",
            "content": input,
        }));
        chat_body["messages"] = serde_json::json!(conversation);
        let resp = client
            .post(format!("{}/chat/completions", args.url))
            .json(&chat_body)
            .send()
            .await
            .context("failed to send chat completion request")?;
        let output = stream_sse_chunked(resp, extract_chat_content).await?;
        conversation.push(serde_json::json!({
            "role": "assistant",
            "content": output,
        }));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

pub async fn run_chat(args: ChatArgs) -> Result<()> {
    if let Some(ref model) = args.resolved_model() {
        // In-process mode — block on sync LLM API.
        let model = model.clone();
        tokio::task::spawn_blocking(move || run_chat_inproc(&args, &model))
            .await
            .context("chat task panicked")?
    } else {
        // Remote mode — connect to running server.
        run_chat_remote(&args).await
    }
}

pub async fn run_complete(args: CompleteArgs) -> Result<()> {
    let api_key = args
        .api_key
        .as_deref()
        .or(std::env::var("OPENAI_API_KEY").ok().as_deref())
        .unwrap_or("EMPTY")
        .to_string();

    let client = build_client(&api_key);
    let model = resolve_model_remote(&client, &args.url, args.model_name.as_deref()).await?;
    println!("Using model: {model}");

    let mut body = serde_json::json!({
        "model": model,
        "stream": true,
    });
    if let Some(max_tokens) = args.max_tokens {
        body["max_tokens"] = serde_json::json!(max_tokens);
    }

    if let Some(ref prompt) = args.quick {
        body["prompt"] = serde_json::json!(prompt);
        let resp = client
            .post(format!("{}/completions", args.url))
            .json(&body)
            .send()
            .await
            .context("failed to send completion request")?;
        stream_sse_chunked(resp, extract_completion_text).await?;
        return Ok(());
    }

    println!("Please enter prompt to complete:");
    while let Some(input) = read_line("> ") {
        if input.is_empty() {
            continue;
        }
        body["prompt"] = serde_json::json!(input);
        let resp = client
            .post(format!("{}/completions", args.url))
            .json(&body)
            .send()
            .await
            .context("failed to send completion request")?;
        stream_sse_chunked(resp, extract_completion_text).await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_chat_content() {
        let chunk = serde_json::json!({
            "choices": [{"delta": {"content": "Hello"}}]
        });
        assert_eq!(extract_chat_content(&chunk), Some("Hello"));

        let empty = serde_json::json!({"choices": [{"delta": {}}]});
        assert_eq!(extract_chat_content(&empty), None);
    }

    #[test]
    fn test_extract_completion_text() {
        let chunk = serde_json::json!({
            "choices": [{"text": "world"}]
        });
        assert_eq!(extract_completion_text(&chunk), Some("world"));

        let null_text = serde_json::json!({"choices": [{"text": null}]});
        assert_eq!(extract_completion_text(&null_text), None);
    }

    #[test]
    fn test_bench_stats_no_tokens() {
        let stats = BenchStats::new(100.0);
        assert!(stats.first_token.is_none());
        assert!(stats.token_times.is_empty());
        assert!((stats.startup_ms - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bench_stats_records_tokens() {
        let mut stats = BenchStats::new(50.0);
        stats.record_token();
        assert!(stats.first_token.is_some());
        assert_eq!(stats.token_times.len(), 1);

        stats.record_token();
        stats.record_token();
        assert_eq!(stats.token_times.len(), 3);
        // First token should not change after initial set.
        let first = stats.first_token.unwrap();
        assert!(stats.token_times[0] == first);
    }

    #[test]
    fn test_bench_stats_print_does_not_panic() {
        // With 0 tokens.
        let stats = BenchStats::new(10.0);
        stats.print(0);

        // With some tokens.
        let mut stats = BenchStats::new(10.0);
        stats.record_token();
        std::thread::sleep(std::time::Duration::from_millis(1));
        stats.record_token();
        stats.print(2);
    }
}
