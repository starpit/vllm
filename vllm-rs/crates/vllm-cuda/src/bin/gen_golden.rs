// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
//! Golden file generator: runs CudaWorker prefill and saves reference logits.
//!
//! Usage (on GPU pod):
//!   cargo run -p vllm-cuda --features cuda --bin gen_golden --release
//!
//! Generates crates/vllm-tk-static/golden/*.json files with top-k logits.

use std::path::{Path, PathBuf};

use half::f16;
use serde_json::json;
use vllm_cuda::device::GpuDevice;
use vllm_cuda::dtype::DType;
use vllm_cuda::kv_cache::KvCachePool;
use vllm_cuda::model::llama::{Llama3RopeScaling, LlamaConfig, LlamaForCausalLM};
use vllm_cuda::weights::GpuWeights;
use vllm_cuda::{OwnedTensor, driver};

/// Resolve model directory from HF cache.
fn resolve_model_dir(model_id: &str) -> PathBuf {
    let api = hf_hub::api::sync::ApiBuilder::from_env()
        .build()
        .expect("hf_hub API");
    let repo = api.model(model_id.to_string());
    let config_path = repo.get("config.json").expect("download config.json");
    config_path.parent().unwrap().to_path_buf()
}

/// Parse LlamaConfig from config.json.
fn parse_llama_config(model_dir: &Path) -> LlamaConfig {
    let config_path = model_dir.join("config.json");
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_path).expect("read config.json"))
            .expect("parse config.json");

    let hidden_size = data["hidden_size"].as_u64().unwrap() as usize;
    let num_attention_heads = data["num_attention_heads"].as_u64().unwrap() as usize;
    let num_kv_heads = data
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(num_attention_heads as u64) as usize;
    let head_dim = data
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(hidden_size / num_attention_heads);

    let llama3_rope_scaling = data.get("rope_scaling").and_then(|rs| {
        let rope_type = rs
            .get("rope_type")
            .or_else(|| rs.get("type"))
            .and_then(|v| v.as_str())?;
        if rope_type != "llama3" {
            return None;
        }
        Some(Llama3RopeScaling {
            factor: rs.get("factor")?.as_f64()?,
            low_freq_factor: rs
                .get("low_freq_factor")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0),
            high_freq_factor: rs
                .get("high_freq_factor")
                .and_then(|v| v.as_f64())
                .unwrap_or(4.0),
            original_max_position_embeddings: rs
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(8192) as usize,
        })
    });

    LlamaConfig {
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        num_hidden_layers: data["num_hidden_layers"].as_u64().unwrap() as usize,
        intermediate_size: data["intermediate_size"].as_u64().unwrap() as usize,
        vocab_size: data["vocab_size"].as_u64().unwrap() as usize,
        max_position_embeddings: data
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize,
        rms_norm_eps: data
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5) as f32,
        rope_theta: data
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0),
        head_dim,
        tie_word_embeddings: data
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        llama3_rope_scaling,
    }
}

unsafe fn h2d_u32(data: &[u32], device: &mut GpuDevice) -> OwnedTensor {
    let t = device.caching.alloc_tensor(&[data.len()], DType::U32);
    driver::memcpy_htod_async(
        t.as_gpu_tensor().as_mut_ptr::<u8>(),
        data.as_ptr() as *const u8,
        data.len() * 4,
        device.compute_stream,
    )
    .expect("h2d u32");
    t
}

unsafe fn h2d_i32(data: &[i32], device: &mut GpuDevice) -> OwnedTensor {
    let t = device.caching.alloc_tensor(&[data.len()], DType::I32);
    driver::memcpy_htod_async(
        t.as_gpu_tensor().as_mut_ptr::<u8>(),
        data.as_ptr() as *const u8,
        data.len() * 4,
        device.compute_stream,
    )
    .expect("h2d i32");
    t
}

unsafe fn h2d_i64(data: &[i64], device: &mut GpuDevice) -> OwnedTensor {
    let t = device.caching.alloc_tensor(&[data.len()], DType::I64);
    driver::memcpy_htod_async(
        t.as_gpu_tensor().as_mut_ptr::<u8>(),
        data.as_ptr() as *const u8,
        data.len() * 8,
        device.compute_stream,
    )
    .expect("h2d i64");
    t
}

unsafe fn gpu_bf16_to_f32(
    ptr: *const u8,
    numel: usize,
    stream: cudarc::driver::sys::CUstream,
) -> Vec<f32> {
    let mut host_bf16 = vec![0u16; numel];
    driver::memcpy_dtoh_async(host_bf16.as_mut_ptr() as *mut u8, ptr, numel * 2, stream)
        .expect("dtoh logits");
    driver::stream_synchronize(stream).expect("sync");
    host_bf16
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect()
}

/// Run CudaWorker prefill and return last-token logits as f32.
fn cuda_worker_prefill(model_dir: &Path, config: &LlamaConfig, input_ids: &[u32]) -> Vec<f32> {
    let num_tokens = input_ids.len();
    let block_size = 16;
    let num_blocks = 64;

    let mut device = GpuDevice::new(0).expect("GpuDevice");
    let mut weights = GpuWeights::from_dir(model_dir, device.compute_stream).expect("GpuWeights");
    let model = LlamaForCausalLM::load(&mut weights, config, DType::BF16, &device).expect("load");

    let kv_cache = unsafe {
        KvCachePool::new(
            config.num_hidden_layers,
            num_blocks,
            block_size,
            config.num_kv_heads,
            config.head_dim,
            DType::BF16,
        )
        .expect("KvCachePool")
    };

    let positions: Vec<u32> = (0..num_tokens as u32).collect();
    let slot_mapping: Vec<i64> = (0..num_tokens as i64).collect();
    let cu_seqlens_q = vec![0i32, num_tokens as i32];
    let seqused_k = vec![num_tokens as i32];
    let blocks_needed = num_tokens.div_ceil(block_size);
    let block_table: Vec<i32> = (0..blocks_needed as i32).collect();

    unsafe {
        let gpu_input_ids = h2d_u32(input_ids, &mut device);
        let gpu_positions = h2d_u32(&positions, &mut device);
        let gpu_slot_mapping = h2d_i64(&slot_mapping, &mut device);
        let gpu_cu_seqlens_q = h2d_i32(&cu_seqlens_q, &mut device);
        let gpu_seqused_k = h2d_i32(&seqused_k, &mut device);
        let mut gpu_block_table = h2d_i32(&block_table, &mut device);
        gpu_block_table.reshape(&[1, blocks_needed], DType::I32);

        let logits_owned = model.forward(
            gpu_input_ids.view(),
            gpu_positions.view(),
            gpu_slot_mapping.view(),
            gpu_cu_seqlens_q.view(),
            gpu_seqused_k.view(),
            gpu_block_table.view(),
            num_tokens,
            num_tokens,
            &kv_cache,
            &mut device,
            None,
        );

        let vocab_size = config.vocab_size;
        let all_logits = gpu_bf16_to_f32(
            logits_owned.as_gpu_tensor().as_ptr::<u8>(),
            num_tokens * vocab_size,
            device.compute_stream,
        );

        all_logits[(num_tokens - 1) * vocab_size..].to_vec()
    }
}

/// Generate golden file for a single model.
fn generate_golden(model_id: &str, output_dir: &Path, input_ids: &[u32]) {
    eprintln!("\n{}", "=".repeat(60));
    eprintln!("Generating golden for: {model_id}");
    eprintln!("{}", "=".repeat(60));

    let model_dir = resolve_model_dir(model_id);
    let config = parse_llama_config(&model_dir);

    eprintln!(
        "  hidden_size={}, layers={}, vocab={}, heads={}, kv_heads={}, head_dim={}",
        config.hidden_size,
        config.num_hidden_layers,
        config.vocab_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
    );

    let logits = cuda_worker_prefill(&model_dir, &config, input_ids);
    eprintln!("  Got {} logits for last token", logits.len());

    // Extract top-20 tokens
    let mut sorted: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_k = 20;
    let top_tokens: Vec<serde_json::Value> = sorted[..top_k]
        .iter()
        .map(|&(id, val)| json!({"token_id": id, "logit": val}))
        .collect();

    eprintln!("  Top-5:");
    for (id, val) in &sorted[..5] {
        eprintln!("    token {id}: {val:.4}");
    }

    // Save golden file
    let safe_name = model_id.replace('/', "__");
    let golden = json!({
        "model_id": model_id,
        "input_ids": input_ids,
        "num_tokens": input_ids.len(),
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "top_k_logits": top_tokens,
        "top1_token_id": sorted[0].0,
        "top1_logit": sorted[0].1,
    });

    let out_path = output_dir.join(format!("{safe_name}.json"));
    std::fs::write(&out_path, serde_json::to_string_pretty(&golden).unwrap())
        .expect("write golden file");
    eprintln!("  Saved: {}", out_path.display());
}

fn main() {
    // Fixed input: 5 real tokens + padding to 128 (TK requires multiple of 128)
    let mut input_ids = vec![0u32; 128];
    input_ids[0] = 128000; // <|begin_of_text|>
    input_ids[1] = 9906; // Hello
    input_ids[2] = 1917; // world
    input_ids[3] = 11; // ,
    input_ids[4] = 1268; // how

    // Output to vllm-tk-static/golden/ (sibling crate)
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("vllm-tk-static")
        .join("golden");
    std::fs::create_dir_all(&golden_dir).expect("create golden dir");

    let models = [
        "unsloth/Llama-3.2-1B-Instruct",
        "unsloth/Llama-3.2-3B-Instruct",
        "unsloth/Llama-3.1-8B-Instruct",
    ];

    for model_id in &models {
        generate_golden(model_id, &golden_dir, &input_ids);
    }

    eprintln!("\nAll golden files generated successfully!");
}
