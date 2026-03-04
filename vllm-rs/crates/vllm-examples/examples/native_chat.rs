// SPDX-License-Identifier: Apache-2.0
//! Native CLI chat demo using the WebGPU backend (Metal/Vulkan/DX12).
//!
//! Usage:
//!   cargo run -p vllm-examples --example native_chat --release
//!   cargo run -p vllm-examples --example native_chat --release -- Qwen/Qwen2-0.5B-Instruct
//!   cargo run -p vllm-examples --example native_chat --release -- HuggingFaceTB/SmolLM-135M
//!
//! Downloads the model from HuggingFace, loads weights onto the GPU via wgpu,
//! and runs interactive chat with greedy autoregressive generation.

use std::io::{self, Write};

use vllm_examples::engine::BrowserEngine;
use vllm_examples::worker::WgpuWorker;

/// Format a user message using ChatML template (used by Qwen2, SmolLM-Instruct, etc.)
/// Returns None for base models that don't use chat templates.
fn apply_chat_template(model_id: &str, user_message: &str) -> Option<String> {
    let id = model_id.to_lowercase();
    if id.contains("instruct") || id.contains("chat") {
        // ChatML format: works for Qwen2-Instruct, SmolLM-Instruct, many others
        Some(format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\n{user_message}<|im_end|>\n\
             <|im_start|>assistant\n"
        ))
    } else {
        None
    }
}

fn main() {
    let model_id = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Qwen/Qwen2-0.5B-Instruct".to_string());

    println!("Initializing WebGPU device...");
    let device = pollster::block_on(vllm_wgpu::WgpuDevice::new())
        .expect("Failed to initialize WebGPU device");

    println!("Loading {model_id}...");
    let (worker, config, tokenizer) =
        WgpuWorker::from_pretrained(device, &model_id).expect("Failed to load model");
    println!(
        "  {} layers, hidden={}, vocab={}\n",
        config.num_hidden_layers, config.hidden_size, config.vocab_size,
    );

    let is_chat =
        model_id.to_lowercase().contains("instruct") || model_id.to_lowercase().contains("chat");
    if is_chat {
        println!("Chat mode (instruction-tuned model detected)");
    } else {
        println!("Completion mode (base model — will continue text, not answer questions)");
    }

    let mut engine = BrowserEngine::new(worker, config);

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

        // Apply chat template if applicable, otherwise use raw input
        let prompt = apply_chat_template(&model_id, input).unwrap_or_else(|| input.to_string());

        // Tokenize
        let encoding = tokenizer
            .encode(prompt.as_str(), false)
            .expect("tokenization failed");
        let ids = encoding.get_ids().to_vec();
        engine.token_ids = ids;
        engine.prefill_pos = 0;
        engine.worker.reset_kv();

        // Prefill: process all prompt tokens to build KV cache
        pollster::block_on(engine.prefill()).expect("prefill failed");

        // Detect EOS token ID from tokenizer
        let eos_id = tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| tokenizer.token_to_id("</s>"))
            .unwrap_or(2);

        // Generate
        let max_new_tokens = 200;
        let start = std::time::Instant::now();
        let mut generated = 0;

        for _ in 0..max_new_tokens {
            match pollster::block_on(engine.step()) {
                Ok(token_id) => {
                    generated += 1;
                    if token_id == eos_id || token_id == 0 {
                        break;
                    }
                    let text = tokenizer.decode(&[token_id], false).unwrap_or_default();
                    print!("{text}");
                    stdout.flush().unwrap();
                }
                Err(e) => {
                    println!("\n[error: {e}]");
                    break;
                }
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        let tps = if elapsed > 0.0 {
            generated as f64 / elapsed
        } else {
            0.0
        };
        println!("\n[{generated} tokens, {tps:.1} tok/s]\n");
    }
}
