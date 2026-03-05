// SPDX-License-Identifier: Apache-2.0
//! Native CLI chat demo using the WebGPU backend (Metal/Vulkan/DX12).
//!
//! Usage:
//!   cargo run -p vllm-examples --example native_chat --release
//!   cargo run -p vllm-examples --example native_chat --release -- --prompt "hello" --bench
//!   cargo run -p vllm-examples --example native_chat --release -- --model Qwen/Qwen2-0.5B-Instruct

use std::io::{self, Write};

use vllm_examples::engine::BrowserEngine;
use vllm_examples::worker::WgpuWorker;

struct Args {
    model_id: String,
    /// Optional separate tokenizer path (for GGUF models without adjacent tokenizer.json)
    tokenizer: Option<String>,
    prompt: Option<String>,
    bench: bool,
    profile: bool,
    max_tokens: usize,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model_id = "Qwen/Qwen2-0.5B-Instruct".to_string();
    let mut tokenizer = None;
    let mut prompt = None;
    let mut bench = false;
    let mut profile = false;
    let mut max_tokens = 200;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_id = args.get(i).cloned().unwrap_or(model_id);
            }
            "--tokenizer" => {
                i += 1;
                tokenizer = args.get(i).cloned();
            }
            "--prompt" => {
                i += 1;
                prompt = args.get(i).cloned();
            }
            "--bench" => bench = true,
            "--profile" => profile = true,
            "--max-tokens" => {
                i += 1;
                max_tokens = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(max_tokens);
            }
            other if !other.starts_with('-') && prompt.is_none() && i == 0 => {
                // Legacy: first positional arg is model_id
                model_id = other.to_string();
            }
            _ => {}
        }
        i += 1;
    }
    Args {
        model_id,
        tokenizer,
        prompt,
        bench,
        profile,
        max_tokens,
    }
}

/// Format a user message using the appropriate chat template.
fn apply_chat_template(model_id: &str, user_message: &str) -> Option<String> {
    let id = model_id.to_lowercase();
    if id.contains("llama") && (id.contains("instruct") || id.contains("chat")) {
        // Llama 3 chat format
        Some(format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
You are a helpful assistant.<|eot_id|>\
<|start_header_id|>user<|end_header_id|>\n\n\
{user_message}<|eot_id|>\
<|start_header_id|>assistant<|end_header_id|>\n\n"
        ))
    } else if id.contains("instruct") || id.contains("chat") {
        // ChatML format (Qwen2, SmolLM-Instruct, etc.)
        Some(format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
<|im_start|>user\n{user_message}<|im_end|>\n\
<|im_start|>assistant\n"
        ))
    } else {
        None
    }
}

/// Generation stats returned by `run_generation`.
struct GenStats {
    prompt_tokens: usize,
    generated: usize,
    ttft_ms: f64,
    decode_tps: f64,
}

fn run_generation(
    engine: &mut BrowserEngine,
    tokenizer: &tokenizers::Tokenizer,
    model_id: &str,
    input: &str,
    max_new_tokens: usize,
    silent: bool,
) -> GenStats {
    let prompt = apply_chat_template(model_id, input).unwrap_or_else(|| input.to_string());
    let encoding = tokenizer
        .encode(prompt.as_str(), false)
        .expect("tokenization failed");
    let ids = encoding.get_ids().to_vec();
    let prompt_tokens = ids.len();
    engine.token_ids = ids;
    engine.prefill_pos = 0;
    engine.worker.reset_kv();

    let prefill_start = std::time::Instant::now();
    pollster::block_on(engine.prefill()).expect("prefill failed");
    let ttft_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let eos_id = tokenizer
        .token_to_id("<|eot_id|>")
        .or_else(|| tokenizer.token_to_id("<|im_end|>"))
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .or_else(|| tokenizer.token_to_id("</s>"))
        .unwrap_or(2);

    let decode_start = std::time::Instant::now();
    let mut generated = 0;
    let mut stdout = io::stdout();

    for _ in 0..max_new_tokens {
        match pollster::block_on(engine.step()) {
            Ok(token_id) => {
                generated += 1;
                if token_id == eos_id || token_id == 0 {
                    break;
                }
                if !silent {
                    let text = tokenizer.decode(&[token_id], false).unwrap_or_default();
                    print!("{text}");
                    stdout.flush().unwrap();
                }
            }
            Err(e) => {
                eprintln!("\n[error: {e}]");
                break;
            }
        }
    }

    let decode_elapsed = decode_start.elapsed().as_secs_f64();
    let decode_tps = if decode_elapsed > 0.0 {
        generated as f64 / decode_elapsed
    } else {
        0.0
    };
    GenStats {
        prompt_tokens,
        generated,
        ttft_ms,
        decode_tps,
    }
}

fn main() {
    let args = parse_args();

    eprintln!("Initializing WebGPU device...");
    let device = pollster::block_on(vllm_wgpu::WgpuDevice::new())
        .expect("Failed to initialize WebGPU device");

    eprintln!("Loading {}...", args.model_id);
    let (worker, config, tokenizer) = if args.model_id.ends_with(".gguf")
        && std::path::Path::new(&args.model_id).exists()
        && args.tokenizer.is_some()
    {
        // GGUF file with separate tokenizer
        let (worker, config) =
            WgpuWorker::from_gguf_path(device, std::path::Path::new(&args.model_id))
                .expect("Failed to load GGUF model");
        let tok_path = args.tokenizer.as_ref().unwrap();
        let tokenizer = tokenizers::Tokenizer::from_file(tok_path)
            .unwrap_or_else(|e| panic!("Failed to load tokenizer {tok_path}: {e}"));
        (worker, config, tokenizer)
    } else {
        WgpuWorker::from_pretrained(device, &args.model_id).expect("Failed to load model")
    };
    eprintln!(
        "  {} layers, hidden={}, vocab={}\n",
        config.num_hidden_layers, config.hidden_size, config.vocab_size,
    );

    let mut engine = BrowserEngine::new(worker, config);

    if args.profile {
        let prompt = args.prompt.as_deref().unwrap_or("why is the sky blue?");
        eprintln!("Warmup...");
        let _warmup = run_generation(&mut engine, &tokenizer, &args.model_id, prompt, 3, true);

        eprintln!("Profiling (sync barriers after every op — slow)...\n");
        // Run a single profiled forward step at a meaningful position
        engine.worker.reset_kv();
        let chat =
            apply_chat_template(&args.model_id, prompt).unwrap_or_else(|| prompt.to_string());
        let encoding = tokenizer
            .encode(chat.as_str(), false)
            .expect("tokenization failed");
        let ids = encoding.get_ids();
        // Prefill prompt tokens normally
        for (i, &id) in ids.iter().enumerate() {
            pollster::block_on(engine.worker.forward_one(id, i)).unwrap();
        }
        // Profile a single decode step
        let pos = ids.len();
        let last_token = *ids.last().unwrap_or(&1);
        let (_tok, report) =
            pollster::block_on(engine.worker.forward_one_profiled(last_token, pos)).unwrap();
        println!("{report}");

        // Also time the real forward pass (no extra syncs) for comparison
        let mut real_times = Vec::new();
        for i in 0..20 {
            let p = pos + 1 + i;
            let start = std::time::Instant::now();
            let _ = pollster::block_on(engine.worker.forward_one(last_token, p)).unwrap();
            real_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        real_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = real_times[real_times.len() / 2];
        println!(
            "\nReal forward (no extra sync): {:.2} ms median → {:.1} tok/s",
            median,
            1000.0 / median
        );
        return;
    }

    if let Some(prompt) = &args.prompt {
        if args.bench {
            // Warmup run (first token compiles shaders)
            eprintln!("Warmup...");
            let _warmup = run_generation(&mut engine, &tokenizer, &args.model_id, prompt, 5, true);

            // Bench run
            eprintln!("Benchmarking...");
            let stats = run_generation(
                &mut engine,
                &tokenizer,
                &args.model_id,
                prompt,
                args.max_tokens,
                true,
            );
            println!(
                "TTFT: {:.1} ms ({} prompt tokens, {:.1} ms/tok)",
                stats.ttft_ms,
                stats.prompt_tokens,
                stats.ttft_ms / stats.prompt_tokens as f64,
            );
            println!(
                "Decode: {:.1} tok/s ({} tokens)",
                stats.decode_tps, stats.generated,
            );
        } else {
            let stats = run_generation(
                &mut engine,
                &tokenizer,
                &args.model_id,
                prompt,
                args.max_tokens,
                false,
            );
            println!(
                "\n[{} tokens, {:.1} tok/s, TTFT {:.1} ms]",
                stats.generated, stats.decode_tps, stats.ttft_ms,
            );
        }
        return;
    }

    // Interactive mode
    println!("Type a prompt and press Enter. Ctrl-D to quit.\n");
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush().unwrap();

        let mut input = String::new();
        if stdin.read_line(&mut input).unwrap() == 0 {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        let stats = run_generation(
            &mut engine,
            &tokenizer,
            &args.model_id,
            input,
            args.max_tokens,
            false,
        );
        println!(
            "\n[{} tokens, {:.1} tok/s, TTFT {:.1} ms]\n",
            stats.generated, stats.decode_tps, stats.ttft_ms,
        );
    }
}
