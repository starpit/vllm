// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm bench serve` — online serving benchmark.
//!
//! Sends concurrent HTTP requests to a running vLLM server (OpenAI-compatible
//! API) and measures per-request latency metrics: TTFT, TPOT, ITL, and
//! end-to-end latency (E2EL).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use tokio::sync::Semaphore;

use crate::args::BenchServeArgs;

/// Per-request result.
struct RequestResult {
    /// Time to first token (seconds).
    ttft: f64,
    /// Inter-token latencies (seconds) — one per token after the first.
    itl: Vec<f64>,
    /// Total end-to-end latency (seconds).
    e2el: f64,
    /// Number of output tokens generated.
    output_tokens: usize,
    success: bool,
}

/// Fetch the first model name from the server's /v1/models endpoint.
async fn get_model_from_server(client: &Client, base_url: &str) -> Result<String> {
    let url = format!("{base_url}/v1/models");
    let resp: serde_json::Value = client.get(&url).send().await?.json().await?;
    let model = resp["data"][0]["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("No models found on server at {base_url}"))?;
    Ok(model.to_string())
}

/// Send a single streaming completions request and measure timing.
async fn send_request(
    client: &Client,
    api_url: &str,
    model: &str,
    prompt: &str,
    output_len: usize,
    args: &BenchServeArgs,
    request_id: &str,
) -> RequestResult {
    let mut body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "logprobs": null,
        "stream": true,
        "stream_options": {"include_usage": true},
        "repetition_penalty": 1.0,
    });

    if args.ignore_eos {
        body["ignore_eos"] = serde_json::json!(true);
    }
    if let Some(t) = args.temperature {
        body["temperature"] = serde_json::json!(t);
    }
    if let Some(p) = args.top_p {
        body["top_p"] = serde_json::json!(p);
    }
    if let Some(k) = args.top_k {
        body["top_k"] = serde_json::json!(k);
    }

    let request_start = Instant::now();
    let mut first_token_time: Option<Instant> = None;
    let mut last_token_time = request_start;
    let mut itl = Vec::new();
    let mut output_tokens = 0usize;

    let mut req = client
        .post(api_url)
        .header("x-request-id", request_id)
        .json(&body);
    if let Some(ref key) = args.api_key {
        req = req.bearer_auth(key);
    }

    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Request failed: {e}");
            return RequestResult {
                ttft: 0.0,
                itl: vec![],
                e2el: request_start.elapsed().as_secs_f64(),
                output_tokens: 0,
                success: false,
            };
        }
    };

    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        eprintln!("Request failed with status {status}: {body_text}");
        return RequestResult {
            ttft: 0.0,
            itl: vec![],
            e2el: request_start.elapsed().as_secs_f64(),
            output_tokens: 0,
            success: false,
        };
    }

    // Parse SSE stream.
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut usage_completion_tokens: Option<usize> = None;

    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(_) => break,
        };
        buf.push_str(&String::from_utf8_lossy(&chunk));

        // Process complete SSE lines.
        while let Some(pos) = buf.find("\n\n") {
            let event = buf[..pos].to_string();
            buf = buf[pos + 2..].to_string();

            for line in event.lines() {
                let line = line.trim();
                if line == "data: [DONE]" {
                    continue;
                }
                if let Some(data) = line.strip_prefix("data: ")
                    && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data)
                {
                    // Match Python's TTFT/ITL logic exactly:
                    // - First chunk with `choices` → TTFT
                    // - Every subsequent chunk with `choices` → ITL
                    // - `most_recent_timestamp` updated on every choices chunk
                    if parsed
                        .get("choices")
                        .is_some_and(|c| c.as_array().is_some_and(|a| !a.is_empty()))
                    {
                        let now = Instant::now();
                        if first_token_time.is_none() {
                            first_token_time = Some(now);
                        } else {
                            itl.push(now.duration_since(last_token_time).as_secs_f64());
                        }
                        last_token_time = now;
                    } else if let Some(ct) = parsed["usage"]["completion_tokens"].as_u64() {
                        usage_completion_tokens = Some(ct as usize);
                    }
                }
            }
        }
    }

    // Prefer server-reported token count.
    if let Some(ct) = usage_completion_tokens {
        output_tokens = ct;
    }

    // Python: output.latency = most_recent_timestamp - st (last choices chunk time).
    let e2el = last_token_time.duration_since(request_start).as_secs_f64();
    let ttft = first_token_time
        .map(|t| t.duration_since(request_start).as_secs_f64())
        .unwrap_or(e2el);

    RequestResult {
        ttft,
        itl,
        e2el,
        output_tokens,
        success: first_token_time.is_some(),
    }
}

/// Compute percentile of a sorted slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Compute standard deviation.
fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

/// Sample from exponential distribution using inverse CDF.
/// Returns -ln(U) / lambda where U is uniform in (0,1).
fn exponential_sample(rng_state: &mut u64, rate: f64) -> f64 {
    // xorshift64 → uniform (0,1)
    *rng_state ^= *rng_state << 13;
    *rng_state ^= *rng_state >> 7;
    *rng_state ^= *rng_state << 17;
    let u = (*rng_state as f64) / (u64::MAX as f64);
    // Clamp away from 0 to avoid -ln(0) = inf.
    let u = u.max(1e-15);
    -u.ln() / rate
}

pub(crate) async fn run_bench_serve(args: BenchServeArgs) -> Result<()> {
    let client_builder = Client::builder().timeout(Duration::from_secs(3600));
    let client = if args.insecure {
        client_builder.danger_accept_invalid_certs(true).build()?
    } else {
        client_builder.build()?
    };

    // Resolve model name.
    let model = match &args.model {
        Some(m) => m.clone(),
        None => {
            eprintln!("No --model specified, fetching from server...");
            get_model_from_server(&client, &args.base_url).await?
        }
    };

    let api_url = format!("{}{}", args.base_url, args.endpoint);
    eprintln!("vLLM Rust — serving benchmark");
    eprintln!("Model: {model}");
    eprintln!("API URL: {api_url}");
    eprintln!(
        "num_prompts: {}, input_len: {}, output_len: {}, request_rate: {}",
        args.num_prompts,
        args.input_len,
        args.output_len,
        if args.request_rate.is_infinite() {
            "inf".to_string()
        } else {
            format!("{:.1}", args.request_rate)
        }
    );

    // Load tokenizer from HuggingFace hub (matches Python's get_tokenizer).
    eprintln!("Loading tokenizer for {model}...");
    let tokenizer = {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(model.clone());
        let tokenizer_path = repo.get("tokenizer.json")?;
        tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?
    };

    // Build allowed tokens (exclude special tokens), matching Python's RandomDataset.
    let vocab_size = tokenizer.get_vocab_size(false) as u32;
    let special_ids: std::collections::HashSet<u32> = tokenizer
        .get_added_vocabulary()
        .get_added_tokens_decoder()
        .iter()
        .filter(|(_, t)| t.special)
        .map(|(id, _)| *id)
        .collect();
    let allowed_tokens: Vec<u32> = (0..vocab_size)
        .filter(|id| !special_ids.contains(id))
        .collect();
    let num_allowed = allowed_tokens.len();
    eprintln!("Tokenizer loaded: vocab_size={vocab_size}, allowed_tokens={num_allowed}");

    // Generate random text prompts matching Python's RandomDataset.generate_token_sequence:
    // token_ids = allowed_tokens[(offset + index + arange(input_len)) % num_allowed]
    // then decode → re-encode → truncate to target length → decode again.
    eprintln!("Generating {} random prompts...", args.num_prompts);
    let prompts: Vec<String> = (0..args.num_prompts)
        .map(|i| {
            let offset = i * 7 + args.seed as usize; // deterministic offset
            let token_ids: Vec<u32> = (0..args.input_len)
                .map(|j| allowed_tokens[(offset + i + j) % num_allowed])
                .collect();
            // Decode to text.
            let text = tokenizer.decode(&token_ids, true).unwrap_or_default();
            // Re-encode to verify length and truncate if needed.
            let encoding = tokenizer.encode(text.as_str(), false).unwrap();
            let re_encoded = encoding.get_ids();
            if re_encoded.len() > args.input_len {
                // Truncate and decode again.
                let truncated = &re_encoded[..args.input_len];
                tokenizer.decode(truncated, true).unwrap_or(text)
            } else {
                text
            }
        })
        .collect();

    // Pre-flight check: send one request to verify connectivity.
    eprintln!("Sending pre-flight request to verify connectivity...");
    {
        let preflight_args = BenchServeArgs {
            model: Some(model.clone()),
            base_url: args.base_url.clone(),
            endpoint: args.endpoint.clone(),
            num_prompts: 1,
            input_len: args.input_len,
            output_len: args.output_len,
            request_rate: f64::INFINITY,
            max_concurrency: None,
            seed: 0,
            disable_tqdm: true,
            output_json: None,
            percentile_metrics: String::new(),
            metric_percentiles: String::new(),
            ignore_eos: args.ignore_eos,
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            api_key: args.api_key.clone(),
            insecure: args.insecure,
        };
        let result = send_request(
            &client,
            &api_url,
            &model,
            &prompts[0],
            args.output_len,
            &preflight_args,
            "preflight",
        )
        .await;
        if !result.success {
            anyhow::bail!("Pre-flight request failed. Check server connectivity and model name.");
        }
        eprintln!("Pre-flight request succeeded.");
    }

    // Progress bar.
    let pb = if !args.disable_tqdm {
        let pb = ProgressBar::new(args.num_prompts as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "Benchmarking {wide_bar:.cyan/blue} {pos}/{len} [{elapsed}<{eta}, {per_sec}]",
            )
            .unwrap()
            .with_key("per_sec", crate::fmt_tqdm_rate),
        );
        Some(pb)
    } else {
        None
    };

    let completed = Arc::new(AtomicUsize::new(0));
    let semaphore = args.max_concurrency.map(|n| Arc::new(Semaphore::new(n)));

    // Separate RNG state for delay sampling.
    let mut delay_rng: u64 = args.seed.wrapping_add(42).max(1);

    let benchmark_start = Instant::now();
    let mut handles = Vec::with_capacity(args.num_prompts);

    for (i, prompt_text) in prompts.into_iter().enumerate() {
        // Inter-request delay: exponential distribution (Poisson process).
        if !args.request_rate.is_infinite() && i > 0 {
            let delay = exponential_sample(&mut delay_rng, args.request_rate);
            tokio::time::sleep(Duration::from_secs_f64(delay)).await;
        }

        let client = client.clone();
        let api_url = api_url.clone();
        let model = model.clone();
        let completed = completed.clone();
        let pb = pb.clone();
        let sem = semaphore.clone();
        let request_id = format!("bench-{i}");

        // Copy args we need into the task. BenchServeArgs isn't Clone,
        // so extract the fields we need.
        let ignore_eos = args.ignore_eos;
        let temperature = args.temperature;
        let top_p = args.top_p;
        let top_k = args.top_k;
        let api_key = args.api_key.clone();
        let insecure = args.insecure;
        let output_len = args.output_len;

        let task_args = BenchServeArgs {
            model: Some(model.clone()),
            base_url: args.base_url.clone(),
            endpoint: args.endpoint.clone(),
            num_prompts: args.num_prompts,
            input_len: args.input_len,
            output_len,
            request_rate: args.request_rate,
            max_concurrency: args.max_concurrency,
            seed: args.seed,
            disable_tqdm: args.disable_tqdm,
            output_json: None,
            percentile_metrics: args.percentile_metrics.clone(),
            metric_percentiles: args.metric_percentiles.clone(),
            ignore_eos,
            temperature,
            top_p,
            top_k,
            api_key,
            insecure,
        };

        handles.push(tokio::spawn(async move {
            let _permit = match &sem {
                Some(s) => Some(s.acquire().await.unwrap()),
                None => None,
            };
            let result = send_request(
                &client,
                &api_url,
                &model,
                &prompt_text,
                output_len,
                &task_args,
                &request_id,
            )
            .await;
            completed.fetch_add(1, Ordering::Relaxed);
            if let Some(ref pb) = pb {
                pb.inc(1);
            }
            result
        }));
    }

    // Collect results.
    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        results.push(handle.await?);
    }

    let total_time = benchmark_start.elapsed().as_secs_f64();
    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    // Compute metrics.
    let successful: Vec<&RequestResult> = results.iter().filter(|r| r.success).collect();
    let num_success = successful.len();
    let num_fail = results.len() - num_success;

    if num_success == 0 {
        anyhow::bail!("All {num_fail} requests failed. Check server connectivity and model name.");
    }

    let total_output_tokens: usize = successful.iter().map(|r| r.output_tokens).sum();
    let total_input_tokens: usize = num_success * args.input_len;

    let mut ttfts: Vec<f64> = successful.iter().map(|r| r.ttft).collect();
    ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // TPOT = (e2el - ttft) / (output_tokens - 1) for requests with >1 token.
    let mut tpots: Vec<f64> = successful
        .iter()
        .filter(|r| r.output_tokens > 1)
        .map(|r| (r.e2el - r.ttft) / (r.output_tokens - 1) as f64)
        .collect();
    tpots.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut itls: Vec<f64> = successful
        .iter()
        .flat_map(|r| r.itl.iter().copied())
        .collect();
    itls.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut e2els: Vec<f64> = successful.iter().map(|r| r.e2el).collect();
    e2els.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let selected_metrics: Vec<&str> = args.percentile_metrics.split(',').collect();
    let selected_pcts: Vec<f64> = args
        .metric_percentiles
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    // Print summary (matches Python's benchmark_serving.py format).
    println!();
    println!("{:=^50}", " Serving Benchmark Result ");
    println!("{:<40} {:<10}", "Successful requests:", num_success);
    println!("{:<40} {:<10}", "Failed requests:", num_fail);
    if let Some(mc) = args.max_concurrency {
        println!("{:<40} {:<10}", "Maximum request concurrency:", mc);
    }
    if !args.request_rate.is_infinite() {
        println!(
            "{:<40} {:<10.2}",
            "Request rate configured (RPS):", args.request_rate
        );
    }
    println!("{:<40} {:<10.2}", "Benchmark duration (s):", total_time);
    println!("{:<40} {:<10}", "Total input tokens:", total_input_tokens);
    println!(
        "{:<40} {:<10}",
        "Total generated tokens:", total_output_tokens
    );
    println!(
        "{:<40} {:<10.2}",
        "Request throughput (req/s):",
        num_success as f64 / total_time
    );
    println!(
        "{:<40} {:<10.2}",
        "Output token throughput (tok/s):",
        total_output_tokens as f64 / total_time
    );
    println!(
        "{:<40} {:<10.2}",
        "Total token throughput (tok/s):",
        (total_input_tokens + total_output_tokens) as f64 / total_time
    );

    // Print per-metric stats (matches Python's process_one_metric format).
    let print_metric = |name: &str, header: &str, data: &[f64]| {
        if data.is_empty() {
            return;
        }
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let median = percentile(data, 50.0);
        println!("{:-^50}", header);
        println!(
            "{:<40} {:<10.2}",
            format!("Mean {name} (ms):"),
            mean * 1000.0
        );
        println!(
            "{:<40} {:<10.2}",
            format!("Median {name} (ms):"),
            median * 1000.0
        );
        for &p in &selected_pcts {
            let p_word = if p == p.floor() {
                format!("{}", p as i64)
            } else {
                format!("{p}")
            };
            println!(
                "{:<40} {:<10.2}",
                format!("P{p_word} {name} (ms):"),
                percentile(data, p) * 1000.0
            );
        }
    };

    for metric in &selected_metrics {
        match *metric {
            "ttft" => print_metric("TTFT", "Time to First Token", &ttfts),
            "tpot" => print_metric("TPOT", "Time per Output Token (excl. 1st token)", &tpots),
            "itl" => print_metric("ITL", "Inter-token Latency", &itls),
            "e2el" => print_metric("E2EL", "End-to-end Latency", &e2els),
            _ => eprintln!("Unknown metric: {metric}"),
        }
    }
    println!("{:=^50}", "");

    // JSON output (matches Python's result dict).
    if let Some(ref path) = args.output_json {
        let mean = |d: &[f64]| {
            if d.is_empty() {
                0.0
            } else {
                d.iter().sum::<f64>() / d.len() as f64
            }
        };

        let mut json = serde_json::json!({
            "duration": total_time,
            "completed": num_success,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "request_throughput": num_success as f64 / total_time,
            "output_throughput": total_output_tokens as f64 / total_time,
            "total_token_throughput": (total_input_tokens + total_output_tokens) as f64 / total_time,
        });
        let obj = json.as_object_mut().unwrap();

        let add_metric_json =
            |obj: &mut serde_json::Map<String, serde_json::Value>, attr: &str, data: &[f64]| {
                obj.insert(
                    format!("mean_{attr}_ms"),
                    serde_json::json!(mean(data) * 1000.0),
                );
                obj.insert(
                    format!("median_{attr}_ms"),
                    serde_json::json!(percentile(data, 50.0) * 1000.0),
                );
                obj.insert(
                    format!("std_{attr}_ms"),
                    serde_json::json!(std_dev(data) * 1000.0),
                );
                for &p in &selected_pcts {
                    let p_word = if p == p.floor() {
                        format!("{}", p as i64)
                    } else {
                        format!("{p}")
                    };
                    obj.insert(
                        format!("p{p_word}_{attr}_ms"),
                        serde_json::json!(percentile(data, p) * 1000.0),
                    );
                }
            };

        for metric in &selected_metrics {
            match *metric {
                "ttft" => add_metric_json(obj, "ttft", &ttfts),
                "tpot" => add_metric_json(obj, "tpot", &tpots),
                "itl" => add_metric_json(obj, "itl", &itls),
                "e2el" => add_metric_json(obj, "e2el", &e2els),
                _ => {}
            }
        }
        std::fs::write(path, serde_json::to_string_pretty(&json)?)?;
        eprintln!("Results written to {path}");
    }

    Ok(())
}
