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
    prompt_tokens: &[u32],
    output_len: usize,
    args: &BenchServeArgs,
) -> RequestResult {
    let mut body = serde_json::json!({
        "model": model,
        "prompt": prompt_tokens,
        "max_tokens": output_len,
        "stream": true,
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

    let mut req = client.post(api_url).json(&body);
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
                    // Check if this chunk has content (a token).
                    let has_content = parsed["choices"][0]["text"]
                        .as_str()
                        .is_some_and(|t| !t.is_empty());
                    if has_content {
                        let now = Instant::now();
                        if first_token_time.is_none() {
                            first_token_time = Some(now);
                        } else {
                            itl.push(now.duration_since(last_token_time).as_secs_f64());
                        }
                        last_token_time = now;
                        output_tokens += 1;
                    }
                }
            }
        }
    }

    let e2el = request_start.elapsed().as_secs_f64();
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

    // Generate random prompts.
    let mut rng_state: u64 = args.seed;
    let mut next_rng = || -> u64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    let prompts: Vec<Vec<u32>> = (0..args.num_prompts)
        .map(|_| {
            (0..args.input_len)
                .map(|_| (next_rng() % 10000) as u32)
                .collect()
        })
        .collect();

    // Progress bar.
    let pb = if !args.disable_tqdm {
        let pb = ProgressBar::new(args.num_prompts as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "Benchmarking {wide_bar:.cyan/blue} {pos}/{len} [{elapsed}<{eta}]",
            )
            .unwrap(),
        );
        Some(pb)
    } else {
        None
    };

    let completed = Arc::new(AtomicUsize::new(0));
    let semaphore = args.max_concurrency.map(|n| Arc::new(Semaphore::new(n)));

    let benchmark_start = Instant::now();
    let mut handles = Vec::with_capacity(args.num_prompts);

    for (i, prompt_tokens) in prompts.into_iter().enumerate() {
        // Inter-request delay (Poisson process).
        if !args.request_rate.is_infinite() && i > 0 {
            // Exponential inter-arrival time.
            let delay = 1.0 / args.request_rate;
            tokio::time::sleep(Duration::from_secs_f64(delay)).await;
        }

        let client = client.clone();
        let api_url = api_url.clone();
        let model = model.clone();
        let completed = completed.clone();
        let pb = pb.clone();
        let sem = semaphore.clone();

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
                &prompt_tokens,
                output_len,
                &task_args,
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

    // Print summary.
    println!();
    println!("============ Serving Benchmark Result ============");
    println!("Successful requests:     {num_success}");
    println!("Failed requests:         {num_fail}");
    println!("Benchmark duration (s):  {total_time:.2}");
    println!("Total input tokens:      {total_input_tokens}");
    println!("Total generated tokens:  {total_output_tokens}");
    println!(
        "Request throughput:      {:.2} requests/s",
        num_success as f64 / total_time
    );
    println!(
        "Output token throughput: {:.2} tokens/s",
        total_output_tokens as f64 / total_time
    );
    println!(
        "Total token throughput:  {:.2} tokens/s",
        (total_input_tokens + total_output_tokens) as f64 / total_time
    );

    // Print per-metric stats.
    let print_metric = |name: &str, data: &[f64], unit: &str| {
        if data.is_empty() {
            return;
        }
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let median = percentile(data, 50.0);
        println!("---------------{name}---------------");
        println!("Mean {name} ({unit}):    {:.2}", mean * 1000.0);
        println!("Median {name} ({unit}):  {:.2}", median * 1000.0);
        for &p in &selected_pcts {
            println!(
                "P{p:.0} {name} ({unit}):     {:.2}",
                percentile(data, p) * 1000.0
            );
        }
    };

    for metric in &selected_metrics {
        match *metric {
            "ttft" => print_metric("TTFT", &ttfts, "ms"),
            "tpot" => print_metric("TPOT", &tpots, "ms"),
            "itl" => print_metric("ITL", &itls, "ms"),
            "e2el" => print_metric("E2EL", &e2els, "ms"),
            _ => eprintln!("Unknown metric: {metric}"),
        }
    }
    println!("==================================================");

    // JSON output.
    if let Some(ref path) = args.output_json {
        let mut json = serde_json::json!({
            "duration": total_time,
            "completed": num_success,
            "failed": num_fail,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "request_throughput": num_success as f64 / total_time,
            "output_throughput": total_output_tokens as f64 / total_time,
        });
        let obj = json.as_object_mut().unwrap();
        for &p in &selected_pcts {
            if selected_metrics.contains(&"ttft") {
                obj.insert(
                    format!("ttft_p{p:.0}_ms"),
                    serde_json::json!(percentile(&ttfts, p) * 1000.0),
                );
            }
            if selected_metrics.contains(&"tpot") {
                obj.insert(
                    format!("tpot_p{p:.0}_ms"),
                    serde_json::json!(percentile(&tpots, p) * 1000.0),
                );
            }
            if selected_metrics.contains(&"itl") {
                obj.insert(
                    format!("itl_p{p:.0}_ms"),
                    serde_json::json!(percentile(&itls, p) * 1000.0),
                );
            }
            if selected_metrics.contains(&"e2el") {
                obj.insert(
                    format!("e2el_p{p:.0}_ms"),
                    serde_json::json!(percentile(&e2els, p) * 1000.0),
                );
            }
        }
        std::fs::write(path, serde_json::to_string_pretty(&json)?)?;
        eprintln!("Results written to {path}");
    }

    Ok(())
}
