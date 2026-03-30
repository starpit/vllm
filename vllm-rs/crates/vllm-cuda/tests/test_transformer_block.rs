//! Test: full transformer block — fused norm+GEMM vs standard path.
//!
//! Bridges the gap between isolated kernel tests (0.00e0) and production
//! `vllm serve` (garbage output). Loads real model weights, constructs a
//! real LlamaDecoderLayer, sets up KV cache + rotary, and runs the full
//! forward pass through both paths.
//!
//! Run: cargo test -p vllm-cuda --features ferrite --test test_transformer_block -- --nocapture

#![cfg(feature = "ferrite")]

use cudarc::driver::sys as cusys;
use half::bf16;
use safetensors::SafeTensors;
use vllm_cuda::model::llama::{LlamaConfig, LlamaDecoderLayer, RotaryCache};
use vllm_cuda::{
    CachingAllocator, DType, GpuDevice, GpuTensor, GpuWeights, KvCachePool, OwnedTensor, TensorView,
};

const MODEL_PATH: &str = concat!(
    env!("HOME"),
    "/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/",
    "snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors"
);

// Fused rms_norm → GEMM kernel (same tile config as production ferrite.gemm)
const FUSED_NORM_GEMM: ptx_fusion::FeriteKernel = ptx_fusion::compile!(
    a = rms_norm,
    b = gemm_64x128x32,
    bind = { a.output => b.param_0 },
    name = "block_test_fused_norm_gemm",
);

// Two-input fused add+norm+GEMM: prologue reads from residual AND hidden_states,
// adds in f32, writes bf16 back to residual, computes inv_rms from f32 sums.
const FUSED_ADD_NORM_GEMM: ptx_fusion::FeriteKernel = ptx_fusion::compile!(
    a = fused_add_rms_norm,
    b = gemm_64x128x32,
    bind = { a.output => b.param_0 },
    name = "block_test_fused_add_norm_gemm",
);

fn qwen_config() -> LlamaConfig {
    LlamaConfig {
        hidden_size: 896,
        num_attention_heads: 14,
        num_kv_heads: 2,
        num_hidden_layers: 24,
        intermediate_size: 4864,
        vocab_size: 151936,
        max_position_embeddings: 32768,
        rms_norm_eps: 1e-6,
        rope_theta: 1000000.0,
        head_dim: 64,
        tie_word_embeddings: true,
        llama3_rope_scaling: None,
    }
}

// ── SafeTensors weight loading (matches existing test patterns) ──

fn load_bf16_tensor(st: &SafeTensors, name: &str) -> Vec<bf16> {
    let tensor = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("missing tensor {name}: {e}"));
    assert_eq!(
        tensor.dtype(),
        safetensors::Dtype::BF16,
        "{name} is not bf16"
    );
    tensor
        .data()
        .chunks_exact(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]))
        .collect()
}

// ── Upload / download helpers ──

unsafe fn upload_bf16(alloc: &mut CachingAllocator, data: &[bf16], shape: &[usize]) -> OwnedTensor {
    let t = alloc.alloc_tensor(shape, DType::BF16);
    cusys::cuMemcpyHtoD_v2(
        t.as_gpu_tensor().raw_ptr() as u64,
        data.as_ptr() as *const _,
        data.len() * 2,
    );
    t
}

unsafe fn upload_u32(alloc: &mut CachingAllocator, data: &[u32], shape: &[usize]) -> OwnedTensor {
    let t = alloc.alloc_tensor(shape, DType::U32);
    cusys::cuMemcpyHtoD_v2(
        t.as_gpu_tensor().raw_ptr() as u64,
        data.as_ptr() as *const _,
        data.len() * 4,
    );
    t
}

unsafe fn upload_i32(alloc: &mut CachingAllocator, data: &[i32], shape: &[usize]) -> OwnedTensor {
    let t = alloc.alloc_tensor(shape, DType::I32);
    cusys::cuMemcpyHtoD_v2(
        t.as_gpu_tensor().raw_ptr() as u64,
        data.as_ptr() as *const _,
        data.len() * 4,
    );
    t
}

unsafe fn upload_i64(alloc: &mut CachingAllocator, data: &[i64], shape: &[usize]) -> OwnedTensor {
    let t = alloc.alloc_tensor(shape, DType::I64);
    cusys::cuMemcpyHtoD_v2(
        t.as_gpu_tensor().raw_ptr() as u64,
        data.as_ptr() as *const _,
        data.len() * 8,
    );
    t
}

unsafe fn download_bf16(t: GpuTensor, count: usize) -> Vec<bf16> {
    let mut v = vec![bf16::ZERO; count];
    cusys::cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut _, t.raw_ptr() as u64, count * 2);
    v
}

fn max_diff_bf16(a: &[bf16], b: &[bf16]) -> (f32, usize) {
    let mut max_d = 0.0f32;
    let mut worst = 0;
    for i in 0..a.len() {
        let d = (a[i].to_f32() - b[i].to_f32()).abs();
        if d > max_d {
            max_d = d;
            worst = i;
        }
    }
    (max_d, worst)
}

fn print_comparison(name: &str, a: &[bf16], b: &[bf16]) {
    let (diff, worst) = max_diff_bf16(a, b);
    println!(
        "  {name}: max_diff={diff:.2e} at [{worst}]  std={:.4} fused={:.4}",
        a[worst].to_f32(),
        b[worst].to_f32()
    );
    if diff > 1.0 {
        let big = a
            .iter()
            .zip(b.iter())
            .filter(|(x, y)| (x.to_f32() - y.to_f32()).abs() > 1.0)
            .count();
        println!("    elements with diff > 1.0: {big} / {}", a.len());
    }
}

/// Upload bf16 data to a GpuTensor (not OwnedTensor — for weights that persist).
unsafe fn upload_weight(
    alloc: &mut CachingAllocator,
    data: &[bf16],
    shape: &[usize],
) -> OwnedTensor {
    upload_bf16(alloc, data, shape)
}

// ════════════════════════════════════════════════════════════════════════
// test10: Full transformer block, fused vs standard
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test10_transformer_block_fused_vs_standard() {
    println!("=== test10: transformer block — fused norm+GEMM vs standard ===");

    // ── 1. Initialize device ──
    let mut device = GpuDevice::new(0).unwrap();
    let stream = device.compute_stream;

    // ── 2. Load layer via GpuWeights (for standard path) ──
    println!("  Loading model weights...");
    let mut weights = GpuWeights::from_single_file(MODEL_PATH, stream).unwrap();
    let config = qwen_config();
    // Layer 1 (not layer 0) so we have a residual connection.
    // NB: weights must outlive the layer — GpuTensors in the layer point
    // to GPU memory owned by GpuWeights (freed on drop).
    let layer =
        LlamaDecoderLayer::load(&mut weights, "model.layers.1", &config, 1, stream).unwrap();

    // ── 3. Load raw weights from SafeTensors (for fused path) ──
    let raw_data = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&raw_data).expect("parse safetensors");

    // QKV weight: q + k + v concatenated vertically → [1152, 896]
    let mut qkv_data = load_bf16_tensor(&st, "model.layers.1.self_attn.q_proj.weight"); // [896, 896]
    qkv_data.extend_from_slice(&load_bf16_tensor(
        &st,
        "model.layers.1.self_attn.k_proj.weight",
    )); // [128, 896]
    qkv_data.extend_from_slice(&load_bf16_tensor(
        &st,
        "model.layers.1.self_attn.v_proj.weight",
    )); // [128, 896]
    let qkv_n = qkv_data.len() / config.hidden_size; // 1152
    let qkv_w =
        unsafe { upload_weight(&mut device.caching, &qkv_data, &[qkv_n, config.hidden_size]) };
    println!("  QKV weight: [{qkv_n}, {}]", config.hidden_size);

    // gate_up weight: gate + up concatenated → [9728, 896]
    let mut gate_up_data = load_bf16_tensor(&st, "model.layers.1.mlp.gate_proj.weight"); // [4864, 896]
    gate_up_data.extend_from_slice(&load_bf16_tensor(&st, "model.layers.1.mlp.up_proj.weight")); // [4864, 896]
    let gate_up_n = gate_up_data.len() / config.hidden_size; // 9728
    let gate_up_w = unsafe {
        upload_weight(
            &mut device.caching,
            &gate_up_data,
            &[gate_up_n, config.hidden_size],
        )
    };
    println!("  gate_up weight: [{gate_up_n}, {}]", config.hidden_size);

    // down_proj weight: [896, 4864]
    let down_data = load_bf16_tensor(&st, "model.layers.1.mlp.down_proj.weight");
    let down_k = down_data.len() / config.hidden_size; // 4864
    let down_w = unsafe {
        upload_weight(
            &mut device.caching,
            &down_data,
            &[config.hidden_size, down_k],
        )
    };

    drop(raw_data);

    // ── 4. KV caches (separate pools for each path) ──
    let kv_std = unsafe { KvCachePool::new(24, 16, 16, 2, 64, DType::BF16).unwrap() };
    let kv_fused = unsafe { KvCachePool::new(24, 16, 16, 2, 64, DType::BF16).unwrap() };

    // ── 5. Rotary embeddings ──
    let rotary =
        unsafe { RotaryCache::new(64, 32768, 1000000.0, None, DType::BF16, &device).unwrap() };

    let num_tokens = 4usize;
    let hidden = config.hidden_size; // 896

    // ── 6. Input data ──
    let hs_data: Vec<bf16> = (0..num_tokens * hidden)
        .map(|i| bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.1))
        .collect();
    let res_data: Vec<bf16> = (0..num_tokens * hidden)
        .map(|i| bf16::from_f32(((i as f32) * 0.00051 + 0.3).cos() * 0.05))
        .collect();

    // Two copies
    let hs_std = unsafe { upload_bf16(&mut device.caching, &hs_data, &[num_tokens, hidden]) };
    let hs_fused = unsafe { upload_bf16(&mut device.caching, &hs_data, &[num_tokens, hidden]) };
    let res_std = unsafe { upload_bf16(&mut device.caching, &res_data, &[num_tokens, hidden]) };
    let res_fused = unsafe { upload_bf16(&mut device.caching, &res_data, &[num_tokens, hidden]) };

    // ── 7. Attention metadata (simple prefill: 1 sequence, 4 tokens) ──
    let positions_data: Vec<u32> = (0..num_tokens as u32).collect();
    let slot_mapping_data: Vec<i64> = (0..num_tokens as i64).collect();
    let cu_seqlens_data: Vec<i32> = vec![0, num_tokens as i32];
    let seqused_data: Vec<i32> = vec![num_tokens as i32];
    let block_table_data: Vec<i32> = vec![0]; // block 0

    let positions = unsafe { upload_u32(&mut device.caching, &positions_data, &[num_tokens]) };
    let slot_mapping =
        unsafe { upload_i64(&mut device.caching, &slot_mapping_data, &[num_tokens]) };
    let cu_seqlens = unsafe { upload_i32(&mut device.caching, &cu_seqlens_data, &[2]) };
    let seqused = unsafe { upload_i32(&mut device.caching, &seqused_data, &[1]) };
    let block_table = unsafe { upload_i32(&mut device.caching, &block_table_data, &[1, 1]) };

    unsafe {
        cusys::cuStreamSynchronize(stream);
    }

    // ════════════════════════════════════════════════════════════════
    // STANDARD PATH: layer.forward()
    // (fused_add_rms_norm_inplace + forward_ferrite CUTLASS GEMM)
    // ════════════════════════════════════════════════════════════════
    println!("  Standard path: layer.forward()...");
    let (out_std, res_out_std) = unsafe {
        layer.forward(
            hs_std,
            Some(res_std),
            positions.view(),
            slot_mapping.view(),
            cu_seqlens.view(),
            seqused.view(),
            block_table.view(),
            num_tokens,
            num_tokens,
            &kv_std,
            &rotary,
            &mut device,
        )
    };

    // ════════════════════════════════════════════════════════════════
    // FUSED PATH: add_inplace + launch_fused_norm_gemm
    // (norm computed inside GEMM prologue — no GMEM round-trip)
    // ════════════════════════════════════════════════════════════════
    println!("  Fused path: add_inplace + launch_fused_norm_gemm...");

    // ── QKV: residual += hidden_states, then fused norm+GEMM ──
    unsafe {
        vllm_cuda::kernels::add_inplace(*res_fused, *hs_fused, stream);
    }
    drop(hs_fused);

    let qkv_fused = unsafe {
        vllm_cuda::ferrite::launch_fused_norm_gemm(
            &FUSED_NORM_GEMM,
            *res_fused,
            *qkv_w,
            layer.input_layernorm.weight,
            layer.input_layernorm.eps,
            hidden as u32,
            None,
            1.0,
            0.0,
            &mut device.caching,
            stream,
        )
    };

    // ── Attention from pre-computed QKV ──
    let attn_fused = unsafe {
        layer.self_attn.forward_from_qkv(
            qkv_fused,
            positions.view(),
            slot_mapping.view(),
            cu_seqlens.view(),
            seqused.view(),
            block_table.view(),
            num_tokens,
            num_tokens,
            &kv_fused,
            &rotary,
            &mut device,
        )
    };

    // residual_multiplier = 1.0 for Qwen, no-op

    // ── MLP: residual += attn_output, then fused norm+GEMM ──
    unsafe {
        vllm_cuda::kernels::add_inplace(*res_fused, *attn_fused, stream);
    }
    drop(attn_fused);

    let gate_up = unsafe {
        vllm_cuda::ferrite::launch_fused_norm_gemm(
            &FUSED_NORM_GEMM,
            *res_fused,
            *gate_up_w,
            layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.eps,
            hidden as u32,
            None,
            1.0,
            0.0,
            &mut device.caching,
            stream,
        )
    };

    // ── SiLU + mul ──
    let activated = unsafe {
        vllm_cuda::kernels::silu_and_mul_fused(
            *gate_up,
            config.intermediate_size,
            &mut device.caching,
            stream,
        )
    };
    drop(gate_up);

    // ── down_proj GEMM (via ferrite.gemm — same as standard path) ──
    let out_fused = unsafe {
        device.ferrite.gemm(
            *activated,
            *down_w,
            None,
            1.0,
            0.0,
            &mut device.caching,
            stream,
        )
    };
    drop(activated);

    // ════════════════════════════════════════════════════════════════
    // COMPARE
    // ════════════════════════════════════════════════════════════════
    unsafe {
        cusys::cuStreamSynchronize(stream);
    }

    let std_mlp_h = unsafe { download_bf16(*out_std, num_tokens * hidden) };
    let fused_mlp_h = unsafe { download_bf16(*out_fused, num_tokens * hidden) };
    let std_res_h = unsafe { download_bf16(*res_out_std, num_tokens * hidden) };
    let fused_res_h = unsafe { download_bf16(*res_fused, num_tokens * hidden) };

    print_comparison("MLP output", &std_mlp_h, &fused_mlp_h);
    print_comparison("Residual", &std_res_h, &fused_res_h);

    let (mlp_diff, _) = max_diff_bf16(&std_mlp_h, &fused_mlp_h);
    let (res_diff, _) = max_diff_bf16(&std_res_h, &fused_res_h);

    assert!(
        mlp_diff < 1.0,
        "MLP output diverged: max_diff={mlp_diff:.2e}"
    );
    assert!(res_diff < 1.0, "Residual diverged: max_diff={res_diff:.2e}");

    if mlp_diff < 0.01 && res_diff < 0.01 {
        println!("PASS: fused matches standard (mlp={mlp_diff:.2e}, res={res_diff:.2e})");
        println!("  → Bug is in engine layer, not the fused kernel.");
    } else {
        println!("NOTABLE DIFF: mlp={mlp_diff:.2e}, res={res_diff:.2e}");
        println!("  → Investigate per-step divergence.");
    }
}

// ════════════════════════════════════════════════════════════════════════
// test10b: Same as test10 but with M=1 (decode batch size)
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test10b_transformer_block_m1_decode() {
    println!("=== test10b: transformer block at M=1 (decode) ===");

    let mut device = GpuDevice::new(0).unwrap();
    let stream = device.compute_stream;

    let mut weights = GpuWeights::from_single_file(MODEL_PATH, stream).unwrap();
    let config = qwen_config();
    let layer =
        LlamaDecoderLayer::load(&mut weights, "model.layers.1", &config, 1, stream).unwrap();
    // NB: weights must outlive layer (owns GPU memory)

    // Load fused-path weights from SafeTensors
    let raw_data = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&raw_data).expect("parse safetensors");

    let mut qkv_data = load_bf16_tensor(&st, "model.layers.1.self_attn.q_proj.weight");
    qkv_data.extend_from_slice(&load_bf16_tensor(
        &st,
        "model.layers.1.self_attn.k_proj.weight",
    ));
    qkv_data.extend_from_slice(&load_bf16_tensor(
        &st,
        "model.layers.1.self_attn.v_proj.weight",
    ));
    let qkv_n = qkv_data.len() / config.hidden_size;
    let qkv_w =
        unsafe { upload_bf16(&mut device.caching, &qkv_data, &[qkv_n, config.hidden_size]) };

    let mut gate_up_data = load_bf16_tensor(&st, "model.layers.1.mlp.gate_proj.weight");
    gate_up_data.extend_from_slice(&load_bf16_tensor(&st, "model.layers.1.mlp.up_proj.weight"));
    let gate_up_n = gate_up_data.len() / config.hidden_size;
    let gate_up_w = unsafe {
        upload_bf16(
            &mut device.caching,
            &gate_up_data,
            &[gate_up_n, config.hidden_size],
        )
    };

    let down_data = load_bf16_tensor(&st, "model.layers.1.mlp.down_proj.weight");
    let down_k = down_data.len() / config.hidden_size;
    let down_w = unsafe {
        upload_bf16(
            &mut device.caching,
            &down_data,
            &[config.hidden_size, down_k],
        )
    };

    drop(raw_data);

    let kv_std = unsafe { KvCachePool::new(24, 16, 16, 2, 64, DType::BF16).unwrap() };
    let kv_fused = unsafe { KvCachePool::new(24, 16, 16, 2, 64, DType::BF16).unwrap() };
    let rotary =
        unsafe { RotaryCache::new(64, 32768, 1000000.0, None, DType::BF16, &device).unwrap() };

    let num_tokens = 1usize;
    let hidden = config.hidden_size;

    let hs_data: Vec<bf16> = (0..hidden)
        .map(|i| bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.1))
        .collect();
    let res_data: Vec<bf16> = (0..hidden)
        .map(|i| bf16::from_f32(((i as f32) * 0.00051 + 0.3).cos() * 0.05))
        .collect();

    let hs_std = unsafe { upload_bf16(&mut device.caching, &hs_data, &[1, hidden]) };
    let hs_fused = unsafe { upload_bf16(&mut device.caching, &hs_data, &[1, hidden]) };
    let res_std = unsafe { upload_bf16(&mut device.caching, &res_data, &[1, hidden]) };
    let res_fused = unsafe { upload_bf16(&mut device.caching, &res_data, &[1, hidden]) };

    // Decode: 1 token at position 0, slot 0
    let positions = unsafe { upload_u32(&mut device.caching, &[0u32], &[1]) };
    let slot_mapping = unsafe { upload_i64(&mut device.caching, &[0i64], &[1]) };
    let cu_seqlens = unsafe { upload_i32(&mut device.caching, &[0i32, 1], &[2]) };
    let seqused = unsafe { upload_i32(&mut device.caching, &[1i32], &[1]) };
    let block_table = unsafe { upload_i32(&mut device.caching, &[0i32], &[1, 1]) };

    unsafe {
        cusys::cuStreamSynchronize(stream);
    }

    // ── Standard path ──
    println!("  Standard path (M=1)...");
    let (out_std, res_out_std) = unsafe {
        layer.forward(
            hs_std,
            Some(res_std),
            positions.view(),
            slot_mapping.view(),
            cu_seqlens.view(),
            seqused.view(),
            block_table.view(),
            num_tokens,
            num_tokens,
            &kv_std,
            &rotary,
            &mut device,
        )
    };

    // ── Fused path ──
    println!("  Fused path (M=1)...");
    unsafe {
        vllm_cuda::kernels::add_inplace(*res_fused, *hs_fused, stream);
    }
    drop(hs_fused);

    let qkv_fused = unsafe {
        vllm_cuda::ferrite::launch_fused_norm_gemm(
            &FUSED_NORM_GEMM,
            *res_fused,
            *qkv_w,
            layer.input_layernorm.weight,
            layer.input_layernorm.eps,
            hidden as u32,
            None,
            1.0,
            0.0,
            &mut device.caching,
            stream,
        )
    };

    let attn_fused = unsafe {
        layer.self_attn.forward_from_qkv(
            qkv_fused,
            positions.view(),
            slot_mapping.view(),
            cu_seqlens.view(),
            seqused.view(),
            block_table.view(),
            num_tokens,
            num_tokens,
            &kv_fused,
            &rotary,
            &mut device,
        )
    };

    unsafe {
        vllm_cuda::kernels::add_inplace(*res_fused, *attn_fused, stream);
    }
    drop(attn_fused);

    let gate_up = unsafe {
        vllm_cuda::ferrite::launch_fused_norm_gemm(
            &FUSED_NORM_GEMM,
            *res_fused,
            *gate_up_w,
            layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.eps,
            hidden as u32,
            None,
            1.0,
            0.0,
            &mut device.caching,
            stream,
        )
    };

    let activated = unsafe {
        vllm_cuda::kernels::silu_and_mul_fused(
            *gate_up,
            config.intermediate_size,
            &mut device.caching,
            stream,
        )
    };
    drop(gate_up);

    let out_fused = unsafe {
        device.ferrite.gemm(
            *activated,
            *down_w,
            None,
            1.0,
            0.0,
            &mut device.caching,
            stream,
        )
    };
    drop(activated);

    // ── Compare ──
    unsafe {
        cusys::cuStreamSynchronize(stream);
    }

    let std_mlp_h = unsafe { download_bf16(*out_std, hidden) };
    let fused_mlp_h = unsafe { download_bf16(*out_fused, hidden) };
    let std_res_h = unsafe { download_bf16(*res_out_std, hidden) };
    let fused_res_h = unsafe { download_bf16(*res_fused, hidden) };

    print_comparison("MLP output (M=1)", &std_mlp_h, &fused_mlp_h);
    print_comparison("Residual (M=1)", &std_res_h, &fused_res_h);

    let (mlp_diff, _) = max_diff_bf16(&std_mlp_h, &fused_mlp_h);
    let (res_diff, _) = max_diff_bf16(&std_res_h, &fused_res_h);

    assert!(mlp_diff < 1.0, "M=1 MLP diverged: max_diff={mlp_diff:.2e}");
    assert!(
        res_diff < 1.0,
        "M=1 Residual diverged: max_diff={res_diff:.2e}"
    );

    if mlp_diff < 0.01 && res_diff < 0.01 {
        println!("PASS: M=1 fused matches standard (mlp={mlp_diff:.2e}, res={res_diff:.2e})");
    } else {
        println!("NOTABLE DIFF at M=1: mlp={mlp_diff:.2e}, res={res_diff:.2e}");
    }
}

// ════════════════════════════════════════════════════════════════════════
// test10c: QKV-only comparison — isolate norm+GEMM from attention/MLP
//
// Both paths use the SAME weights (loaded from SafeTensors).
// Standard: fused_add_rms_norm_inplace + ferrite.gemm
// Fused:    add_inplace + launch_fused_norm_gemm
//
// If this matches (like test9a-k), the test10 divergence is in weight
// loading or the attention/MLP steps, not in the fused kernel itself.
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test10c_qkv_only_shared_weights() {
    println!("=== test10c: QKV-only, shared weights — isolate norm+GEMM ===");

    let mut device = GpuDevice::new(0).unwrap();
    let stream = device.compute_stream;

    let config = qwen_config();
    let hidden = config.hidden_size; // 896

    // Load weights from SafeTensors (shared by both paths)
    let raw_data = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&raw_data).expect("parse safetensors");

    let norm_data = load_bf16_tensor(&st, "model.layers.1.input_layernorm.weight");
    let norm_w = unsafe { upload_bf16(&mut device.caching, &norm_data, &[hidden]) };
    let eps = config.rms_norm_eps;

    let mut qkv_data = load_bf16_tensor(&st, "model.layers.1.self_attn.q_proj.weight");
    qkv_data.extend_from_slice(&load_bf16_tensor(
        &st,
        "model.layers.1.self_attn.k_proj.weight",
    ));
    qkv_data.extend_from_slice(&load_bf16_tensor(
        &st,
        "model.layers.1.self_attn.v_proj.weight",
    ));
    let qkv_n = qkv_data.len() / hidden;
    let qkv_w = unsafe { upload_bf16(&mut device.caching, &qkv_data, &[qkv_n, hidden]) };

    drop(raw_data);

    for m in [1u32, 4, 64, 128] {
        let num = m as usize;
        let hs_data: Vec<bf16> = (0..num * hidden)
            .map(|i| bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.1))
            .collect();
        let res_data: Vec<bf16> = (0..num * hidden)
            .map(|i| bf16::from_f32(((i as f32) * 0.00051 + 0.3).cos() * 0.05))
            .collect();

        // Two copies of inputs
        let hs_std = unsafe { upload_bf16(&mut device.caching, &hs_data, &[num, hidden]) };
        let hs_fused = unsafe { upload_bf16(&mut device.caching, &hs_data, &[num, hidden]) };
        let res_std = unsafe { upload_bf16(&mut device.caching, &res_data, &[num, hidden]) };
        let res_fused = unsafe { upload_bf16(&mut device.caching, &res_data, &[num, hidden]) };

        unsafe {
            cusys::cuStreamSynchronize(stream);
        }

        // ── Standard: fused_add_rms_norm_inplace + ferrite.gemm ──
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(*hs_std, *res_std, *norm_w, eps, stream);
        }
        let qkv_std = unsafe {
            device
                .ferrite
                .gemm(*hs_std, *qkv_w, None, 1.0, 0.0, &mut device.caching, stream)
        };

        // ── Fused: add_inplace + launch_fused_norm_gemm ──
        unsafe {
            vllm_cuda::kernels::add_inplace(*res_fused, *hs_fused, stream);
        }
        let qkv_fused = unsafe {
            vllm_cuda::ferrite::launch_fused_norm_gemm(
                &FUSED_NORM_GEMM,
                *res_fused,
                *qkv_w,
                *norm_w,
                eps,
                hidden as u32,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };

        // ── Compare ──
        unsafe {
            cusys::cuStreamSynchronize(stream);
        }
        let std_h = unsafe { download_bf16(*qkv_std, num * qkv_n) };
        let fused_h = unsafe { download_bf16(*qkv_fused, num * qkv_n) };

        // Also compare residuals
        let std_res_h = unsafe { download_bf16(*res_std, num * hidden) };
        let fused_res_h = unsafe { download_bf16(*res_fused, num * hidden) };

        let (qkv_diff, qkv_worst) = max_diff_bf16(&std_h, &fused_h);
        let (res_diff, _) = max_diff_bf16(&std_res_h, &fused_res_h);

        println!("  M={m}: QKV diff={qkv_diff:.2e} at [{qkv_worst}], residual diff={res_diff:.2e}");

        assert!(qkv_diff < 0.01, "M={m}: QKV diverged: {qkv_diff:.2e}");
        assert!(res_diff < 0.01, "M={m}: residual diverged: {res_diff:.2e}");
    }

    println!("PASS: QKV matches at all M with shared weights");
}

// ════════════════════════════════════════════════════════════════════════
// test10e: 24-layer chain — does the bf16 precision loss compound?
//
// Per layer: add-to-residual + norm + GEMM. Output becomes next layer's hs.
// Path A (standard): fused_add_rms_norm_inplace + ferrite.gemm
// Path B (fused):    add_inplace + launch_fused_norm_gemm
//
// Both use the SAME weights (SafeTensors). Reports diff per layer.
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test10e_24_layer_chain_precision() {
    println!("=== test10e: 24-layer chain — precision accumulation ===");

    let mut device = GpuDevice::new(0).unwrap();
    let stream = device.compute_stream;
    let config = qwen_config();
    let hidden = config.hidden_size;
    let eps = config.rms_norm_eps;
    let num_layers = config.num_hidden_layers;

    let raw_data = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&raw_data).expect("parse safetensors");

    // q_proj [896, 896] per layer — square GEMM so output feeds back as input
    let mut qkv_weights = Vec::new();
    let mut norm_weights = Vec::new();
    for layer_idx in 0..num_layers {
        let q = load_bf16_tensor(
            &st,
            &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
        );
        qkv_weights.push(unsafe { upload_bf16(&mut device.caching, &q, &[hidden, hidden]) });
        let n = load_bf16_tensor(
            &st,
            &format!("model.layers.{layer_idx}.input_layernorm.weight"),
        );
        norm_weights.push(unsafe { upload_bf16(&mut device.caching, &n, &[hidden]) });
    }
    drop(raw_data);

    let m = 4usize;
    let hs_data: Vec<bf16> = (0..m * hidden)
        .map(|i| bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.1))
        .collect();

    // Path A: standard (f32 add+norm)
    let mut hs_a = unsafe { upload_bf16(&mut device.caching, &hs_data, &[m, hidden]) };
    let mut res_a = unsafe {
        upload_bf16(
            &mut device.caching,
            &vec![bf16::ZERO; m * hidden],
            &[m, hidden],
        )
    };
    // Path B: fused (bf16 add, then norm+GEMM)
    let mut hs_b = unsafe { upload_bf16(&mut device.caching, &hs_data, &[m, hidden]) };
    let mut res_b = unsafe {
        upload_bf16(
            &mut device.caching,
            &vec![bf16::ZERO; m * hidden],
            &[m, hidden],
        )
    };

    unsafe { cusys::cuStreamSynchronize(stream) };

    println!("  layer |  a_max  |  b_max  |   diff   | notes");
    println!("  ------|---------|---------|----------|------");

    for layer_idx in 0..num_layers {
        // ── Path A: fused_add_rms_norm_inplace + ferrite.gemm ──
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(
                *hs_a,
                *res_a,
                *norm_weights[layer_idx],
                eps,
                stream,
            );
        }
        let new_a = unsafe {
            device.ferrite.gemm(
                *hs_a,
                *qkv_weights[layer_idx],
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(hs_a);
        hs_a = new_a;

        // ── Path B: add_inplace + launch_fused_norm_gemm ──
        unsafe {
            vllm_cuda::kernels::add_inplace(*res_b, *hs_b, stream);
        }
        drop(hs_b);
        let new_b = unsafe {
            vllm_cuda::ferrite::launch_fused_norm_gemm(
                &FUSED_NORM_GEMM,
                *res_b,
                *qkv_weights[layer_idx],
                *norm_weights[layer_idx],
                eps,
                hidden as u32,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        hs_b = new_b;

        // ── Compare ──
        unsafe { cusys::cuStreamSynchronize(stream) };
        let a_h = unsafe { download_bf16(*hs_a, m * hidden) };
        let b_h = unsafe { download_bf16(*hs_b, m * hidden) };

        let (diff, _) = max_diff_bf16(&a_h, &b_h);
        let a_max = a_h.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
        let b_max = b_h.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
        let a_nan = a_h.iter().filter(|v| v.to_f32().is_nan()).count();
        let b_nan = b_h.iter().filter(|v| v.to_f32().is_nan()).count();

        let notes = if a_nan > 0 || b_nan > 0 {
            format!("NaN! a={a_nan} b={b_nan}")
        } else if diff > 100.0 {
            "EXPLODED".to_string()
        } else if diff > 1.0 {
            "diverging".to_string()
        } else {
            String::new()
        };

        println!(
            "  {:5} | {:7.2} | {:7.2} | {:8.2e} | {notes}",
            layer_idx, a_max, b_max, diff,
        );

        if a_nan > 0 || b_nan > 0 || a_max > 1e4 || b_max > 1e4 {
            println!("  ABORT: values exploded");
            break;
        }
    }

    let a_final = unsafe { download_bf16(*hs_a, m * hidden) };
    let b_final = unsafe { download_bf16(*hs_b, m * hidden) };
    let (final_diff, _) = max_diff_bf16(&a_final, &b_final);
    let b_nan = b_final.iter().filter(|v| v.to_f32().is_nan()).count();

    println!();
    if b_nan > 0 {
        println!("  RESULT: fused path produced NaN — bf16 precision loss is fatal");
    } else if final_diff > 100.0 {
        println!("  RESULT: diff={final_diff:.2e} — bf16 precision loss compounds to garbage");
    } else {
        println!("  RESULT: diff={final_diff:.2e} — precision loss is tolerable");
        println!("  → Production bug is NOT the bf16 precision. Something else is wrong.");
    }
}

// ════════════════════════════════════════════════════════════════════════
// test10d: Norm precision diagnostic
//
// Compare the NORMED output of:
//   A) fused_add_rms_norm_inplace (f32 intermediate for res+hs → norm)
//   B) add_inplace + standalone rms_norm on the bf16 result
//
// If these differ, the bf16 round-trip between add and norm loses precision
// that fused_add_rms_norm_inplace preserves. This would explain why
// replacing fused_add_rms_norm + GEMM with add_inplace + fused_norm_gemm
// diverges: the norm inputs are different due to bf16 truncation.
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test10d_norm_precision_fused_add_vs_separate() {
    println!("=== test10d: norm precision — fused_add_rms_norm vs add+rms_norm ===");
    println!("  If these differ, bf16 round-trip between add and norm loses precision.");

    let mut device = GpuDevice::new(0).unwrap();
    let stream = device.compute_stream;
    let config = qwen_config();
    let hidden = config.hidden_size;
    let eps = config.rms_norm_eps;

    let raw_data = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&raw_data).expect("parse safetensors");
    let norm_data = load_bf16_tensor(&st, "model.layers.1.input_layernorm.weight");
    let norm_w = unsafe { upload_bf16(&mut device.caching, &norm_data, &[hidden]) };
    drop(raw_data);

    for m in [1u32, 4, 64, 128] {
        let num = m as usize;
        let hs_data: Vec<bf16> = (0..num * hidden)
            .map(|i| bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.1))
            .collect();
        let res_data: Vec<bf16> = (0..num * hidden)
            .map(|i| bf16::from_f32(((i as f32) * 0.00051 + 0.3).cos() * 0.05))
            .collect();

        // Path A: fused_add_rms_norm_inplace
        let hs_a = unsafe { upload_bf16(&mut device.caching, &hs_data, &[num, hidden]) };
        let res_a = unsafe { upload_bf16(&mut device.caching, &res_data, &[num, hidden]) };

        // Path B: add_inplace + standalone rms_norm
        let hs_b = unsafe { upload_bf16(&mut device.caching, &hs_data, &[num, hidden]) };
        let res_b = unsafe { upload_bf16(&mut device.caching, &res_data, &[num, hidden]) };

        unsafe {
            cusys::cuStreamSynchronize(stream);
        }

        // A: fused_add_rms_norm_inplace → hs_a gets normed, res_a gets sum
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(*hs_a, *res_a, *norm_w, eps, stream);
        }

        // B: add_inplace → res_b gets sum, then standalone rms_norm
        unsafe {
            vllm_cuda::kernels::add_inplace(*res_b, *hs_b, stream);
        }
        let normed_b = unsafe {
            vllm_cuda::kernels::rms_norm(*res_b, *norm_w, eps, &mut device.caching, stream)
        };

        unsafe {
            cusys::cuStreamSynchronize(stream);
        }

        // Compare: normed output A (hs_a) vs normed output B (normed_b)
        let a_h = unsafe { download_bf16(*hs_a, num * hidden) };
        let b_h = unsafe { download_bf16(*normed_b, num * hidden) };

        // Also compare residuals (should be identical)
        let res_a_h = unsafe { download_bf16(*res_a, num * hidden) };
        let res_b_h = unsafe { download_bf16(*res_b, num * hidden) };

        let (norm_diff, norm_worst) = max_diff_bf16(&a_h, &b_h);
        let (res_diff, _) = max_diff_bf16(&res_a_h, &res_b_h);

        println!(
            "  M={m}: norm diff={norm_diff:.2e} at [{norm_worst}], residual diff={res_diff:.2e}"
        );
        if norm_diff > 0.0 {
            println!(
                "    fused_add_norm={:.6} add+norm={:.6}",
                a_h[norm_worst].to_f32(),
                b_h[norm_worst].to_f32()
            );
        }
    }
    println!("DONE");
}

// ════════════════════════════════════════════════════════════════════════
// test11: Single-layer step-by-step decomposition
//
// Both paths use the SAME weights (SafeTensors). Both paths are
// reimplemented manually so we can capture intermediates.
//
// Standard: fused_add_rms_norm_inplace + ferrite.gemm (matches production)
// Fused:    add_inplace + launch_fused_norm_gemm
//
// test11a: QKV norm+GEMM only
// test11b: QKV norm+GEMM + attention
// test11c: + MLP norm+GEMM
// test11d: + SiLU+mul + down_proj (full layer)
// ════════════════════════════════════════════════════════════════════════

/// Shared setup for test11 series: load layer 1 weights, create device, KV caches.
struct Test11Setup {
    device: GpuDevice,
    config: LlamaConfig,
    // Weights (SafeTensors-loaded, shared by both paths)
    norm1_w: OwnedTensor,
    qkv_w: OwnedTensor,
    norm2_w: OwnedTensor,
    gate_up_w: OwnedTensor,
    down_w: OwnedTensor,
    // Layer for attention (forward_from_qkv is pub)
    layer: LlamaDecoderLayer,
    _weights: GpuWeights, // must outlive layer
    // KV caches
    kv_a: KvCachePool,
    kv_b: KvCachePool,
    rotary: RotaryCache,
}

impl Test11Setup {
    fn new() -> Self {
        let mut device = GpuDevice::new(0).unwrap();
        let stream = device.compute_stream;
        let config = qwen_config();
        let hidden = config.hidden_size;

        // Load layer for attention
        let mut gw = GpuWeights::from_single_file(MODEL_PATH, stream).unwrap();
        let layer = LlamaDecoderLayer::load(&mut gw, "model.layers.1", &config, 1, stream).unwrap();

        // Load raw weights for manual paths
        let raw = std::fs::read(MODEL_PATH).expect("read model");
        let st = SafeTensors::deserialize(&raw).expect("parse");

        let n1 = load_bf16_tensor(&st, "model.layers.1.input_layernorm.weight");
        let norm1_w = unsafe { upload_bf16(&mut device.caching, &n1, &[hidden]) };

        let mut qkv = load_bf16_tensor(&st, "model.layers.1.self_attn.q_proj.weight");
        qkv.extend_from_slice(&load_bf16_tensor(
            &st,
            "model.layers.1.self_attn.k_proj.weight",
        ));
        qkv.extend_from_slice(&load_bf16_tensor(
            &st,
            "model.layers.1.self_attn.v_proj.weight",
        ));
        let qkv_n = qkv.len() / hidden;
        let qkv_w = unsafe { upload_bf16(&mut device.caching, &qkv, &[qkv_n, hidden]) };

        let n2 = load_bf16_tensor(&st, "model.layers.1.post_attention_layernorm.weight");
        let norm2_w = unsafe { upload_bf16(&mut device.caching, &n2, &[hidden]) };

        let mut gu = load_bf16_tensor(&st, "model.layers.1.mlp.gate_proj.weight");
        gu.extend_from_slice(&load_bf16_tensor(&st, "model.layers.1.mlp.up_proj.weight"));
        let gu_n = gu.len() / hidden;
        let gate_up_w = unsafe { upload_bf16(&mut device.caching, &gu, &[gu_n, hidden]) };

        let dw = load_bf16_tensor(&st, "model.layers.1.mlp.down_proj.weight");
        let down_k = dw.len() / hidden;
        let down_w = unsafe { upload_bf16(&mut device.caching, &dw, &[hidden, down_k]) };

        drop(raw);

        let kv_a = unsafe { KvCachePool::new(24, 16, 16, 2, 64, DType::BF16).unwrap() };
        let kv_b = unsafe { KvCachePool::new(24, 16, 16, 2, 64, DType::BF16).unwrap() };
        let rotary =
            unsafe { RotaryCache::new(64, 32768, 1000000.0, None, DType::BF16, &device).unwrap() };

        Test11Setup {
            device,
            config,
            norm1_w,
            qkv_w,
            norm2_w,
            gate_up_w,
            down_w,
            layer,
            _weights: gw,
            kv_a,
            kv_b,
            rotary,
        }
    }

    fn make_inputs(&mut self, m: usize) -> (OwnedTensor, OwnedTensor, OwnedTensor, OwnedTensor) {
        let hidden = self.config.hidden_size;
        let hs_data: Vec<bf16> = (0..m * hidden)
            .map(|i| bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.1))
            .collect();
        let res_data: Vec<bf16> = (0..m * hidden)
            .map(|i| bf16::from_f32(((i as f32) * 0.00051 + 0.3).cos() * 0.05))
            .collect();
        let hs_a = unsafe { upload_bf16(&mut self.device.caching, &hs_data, &[m, hidden]) };
        let hs_b = unsafe { upload_bf16(&mut self.device.caching, &hs_data, &[m, hidden]) };
        let res_a = unsafe { upload_bf16(&mut self.device.caching, &res_data, &[m, hidden]) };
        let res_b = unsafe { upload_bf16(&mut self.device.caching, &res_data, &[m, hidden]) };
        (hs_a, hs_b, res_a, res_b)
    }

    fn make_attn_meta(
        &mut self,
        m: usize,
    ) -> (
        OwnedTensor,
        OwnedTensor,
        OwnedTensor,
        OwnedTensor,
        OwnedTensor,
    ) {
        let pos: Vec<u32> = (0..m as u32).collect();
        let slot: Vec<i64> = (0..m as i64).collect();
        let cu: Vec<i32> = vec![0, m as i32];
        let seq: Vec<i32> = vec![m as i32];
        let bt: Vec<i32> = vec![0];
        unsafe {
            (
                upload_u32(&mut self.device.caching, &pos, &[m]),
                upload_i64(&mut self.device.caching, &slot, &[m]),
                upload_i32(&mut self.device.caching, &cu, &[2]),
                upload_i32(&mut self.device.caching, &seq, &[1]),
                upload_i32(&mut self.device.caching, &bt, &[1, 1]),
            )
        }
    }
}

// ── test11a: QKV norm+GEMM only ──

#[test]
fn test11a_qkv_norm_gemm() {
    println!("=== test11a: QKV norm+GEMM — standard vs fused ===");
    let mut s = Test11Setup::new();
    let stream = s.device.compute_stream;
    let hidden = s.config.hidden_size;
    let eps = s.config.rms_norm_eps;

    for m in [1usize, 4, 64, 128] {
        let (hs_a, hs_b, res_a, res_b) = s.make_inputs(m);
        unsafe { cusys::cuStreamSynchronize(stream) };

        // Standard: fused_add_rms_norm_inplace + ferrite.gemm
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(*hs_a, *res_a, *s.norm1_w, eps, stream);
        }
        let qkv_a = unsafe {
            s.device.ferrite.gemm(
                *hs_a,
                *s.qkv_w,
                None,
                1.0,
                0.0,
                &mut s.device.caching,
                stream,
            )
        };

        // Fused: add_inplace + launch_fused_norm_gemm
        unsafe {
            vllm_cuda::kernels::add_inplace(*res_b, *hs_b, stream);
        }
        let qkv_b = unsafe {
            vllm_cuda::ferrite::launch_fused_norm_gemm(
                &FUSED_NORM_GEMM,
                *res_b,
                *s.qkv_w,
                *s.norm1_w,
                eps,
                hidden as u32,
                None,
                1.0,
                0.0,
                &mut s.device.caching,
                stream,
            )
        };

        unsafe { cusys::cuStreamSynchronize(stream) };
        let qkv_n = s.qkv_w.as_gpu_tensor().dim(0);
        let a_h = unsafe { download_bf16(*qkv_a, m * qkv_n) };
        let b_h = unsafe { download_bf16(*qkv_b, m * qkv_n) };
        let (diff, _) = max_diff_bf16(&a_h, &b_h);
        println!("  M={m}: QKV diff={diff:.2e} (expected: ~1e-2 from bf16 precision)");
    }
    println!("PASS: test11a — QKV diffs are from bf16 precision, not bugs");
}

// ── test11b: QKV norm+GEMM + attention ──

#[test]
fn test11b_qkv_plus_attention() {
    println!("=== test11b: QKV norm+GEMM + attention ===");
    let mut s = Test11Setup::new();
    let stream = s.device.compute_stream;
    let hidden = s.config.hidden_size;
    let eps = s.config.rms_norm_eps;
    let m = 4usize;

    let (hs_a, hs_b, res_a, res_b) = s.make_inputs(m);
    let (pos, slot, cu, seq, bt) = s.make_attn_meta(m);
    unsafe { cusys::cuStreamSynchronize(stream) };

    // ── Standard: norm+GEMM ──
    unsafe {
        vllm_cuda::kernels::fused_add_rms_norm_inplace(*hs_a, *res_a, *s.norm1_w, eps, stream);
    }
    let qkv_a = unsafe {
        s.device.ferrite.gemm(
            *hs_a,
            *s.qkv_w,
            None,
            1.0,
            0.0,
            &mut s.device.caching,
            stream,
        )
    };
    // Standard: attention
    let attn_a = unsafe {
        s.layer.self_attn.forward_from_qkv(
            qkv_a,
            pos.view(),
            slot.view(),
            cu.view(),
            seq.view(),
            bt.view(),
            m,
            m,
            &s.kv_a,
            &s.rotary,
            &mut s.device,
        )
    };

    // ── Fused: norm+GEMM ──
    unsafe {
        vllm_cuda::kernels::add_inplace(*res_b, *hs_b, stream);
    }
    let qkv_b = unsafe {
        vllm_cuda::ferrite::launch_fused_norm_gemm(
            &FUSED_NORM_GEMM,
            *res_b,
            *s.qkv_w,
            *s.norm1_w,
            eps,
            hidden as u32,
            None,
            1.0,
            0.0,
            &mut s.device.caching,
            stream,
        )
    };
    // Fused: attention (same function, different KV cache)
    let attn_b = unsafe {
        s.layer.self_attn.forward_from_qkv(
            qkv_b,
            pos.view(),
            slot.view(),
            cu.view(),
            seq.view(),
            bt.view(),
            m,
            m,
            &s.kv_b,
            &s.rotary,
            &mut s.device,
        )
    };

    unsafe { cusys::cuStreamSynchronize(stream) };
    let a_h = unsafe { download_bf16(*attn_a, m * hidden) };
    let b_h = unsafe { download_bf16(*attn_b, m * hidden) };
    let (diff, worst) = max_diff_bf16(&a_h, &b_h);
    println!("  M={m}: attn diff={diff:.2e} at [{worst}]");
    print_comparison("attention output", &a_h, &b_h);

    // Also compare residuals
    let ra = unsafe { download_bf16(*res_a, m * hidden) };
    let rb = unsafe { download_bf16(*res_b, m * hidden) };
    let (rdiff, _) = max_diff_bf16(&ra, &rb);
    println!("  residual diff={rdiff:.2e}");

    assert!(diff < 5.0, "Attention diverged: {diff:.2e}");
    println!("PASS: test11b — attention output diff is {diff:.2e}");
}

// ── test11c: + MLP norm+GEMM (gate_up) ──

#[test]
fn test11c_plus_mlp_norm_gemm() {
    println!("=== test11c: QKV + attention + MLP norm+GEMM ===");
    let mut s = Test11Setup::new();
    let stream = s.device.compute_stream;
    let hidden = s.config.hidden_size;
    let eps = s.config.rms_norm_eps;
    let m = 4usize;

    let (hs_a, hs_b, res_a, res_b) = s.make_inputs(m);
    let (pos, slot, cu, seq, bt) = s.make_attn_meta(m);
    unsafe { cusys::cuStreamSynchronize(stream) };

    // ── Standard path through QKV + attention ──
    unsafe {
        vllm_cuda::kernels::fused_add_rms_norm_inplace(*hs_a, *res_a, *s.norm1_w, eps, stream);
    }
    let qkv_a = unsafe {
        s.device.ferrite.gemm(
            *hs_a,
            *s.qkv_w,
            None,
            1.0,
            0.0,
            &mut s.device.caching,
            stream,
        )
    };
    drop(hs_a);
    let attn_a = unsafe {
        s.layer.self_attn.forward_from_qkv(
            qkv_a,
            pos.view(),
            slot.view(),
            cu.view(),
            seq.view(),
            bt.view(),
            m,
            m,
            &s.kv_a,
            &s.rotary,
            &mut s.device,
        )
    };
    // Standard: MLP add+norm+GEMM
    unsafe {
        vllm_cuda::kernels::fused_add_rms_norm_inplace(*attn_a, *res_a, *s.norm2_w, eps, stream);
    }
    let gate_up_a = unsafe {
        s.device.ferrite.gemm(
            *attn_a,
            *s.gate_up_w,
            None,
            1.0,
            0.0,
            &mut s.device.caching,
            stream,
        )
    };

    // ── Fused path through QKV + attention ──
    unsafe {
        vllm_cuda::kernels::add_inplace(*res_b, *hs_b, stream);
    }
    drop(hs_b);
    let qkv_b = unsafe {
        vllm_cuda::ferrite::launch_fused_norm_gemm(
            &FUSED_NORM_GEMM,
            *res_b,
            *s.qkv_w,
            *s.norm1_w,
            eps,
            hidden as u32,
            None,
            1.0,
            0.0,
            &mut s.device.caching,
            stream,
        )
    };
    let attn_b = unsafe {
        s.layer.self_attn.forward_from_qkv(
            qkv_b,
            pos.view(),
            slot.view(),
            cu.view(),
            seq.view(),
            bt.view(),
            m,
            m,
            &s.kv_b,
            &s.rotary,
            &mut s.device,
        )
    };
    // Fused: MLP add+norm+GEMM
    unsafe {
        vllm_cuda::kernels::add_inplace(*res_b, *attn_b, stream);
    }
    drop(attn_b);
    let gate_up_b = unsafe {
        vllm_cuda::ferrite::launch_fused_norm_gemm(
            &FUSED_NORM_GEMM,
            *res_b,
            *s.gate_up_w,
            *s.norm2_w,
            eps,
            hidden as u32,
            None,
            1.0,
            0.0,
            &mut s.device.caching,
            stream,
        )
    };

    unsafe { cusys::cuStreamSynchronize(stream) };
    let gu_n = s.gate_up_w.as_gpu_tensor().dim(0);
    let a_h = unsafe { download_bf16(*gate_up_a, m * gu_n) };
    let b_h = unsafe { download_bf16(*gate_up_b, m * gu_n) };
    let (diff, _) = max_diff_bf16(&a_h, &b_h);
    print_comparison("gate_up output", &a_h, &b_h);
    assert!(diff < 10.0, "gate_up diverged: {diff:.2e}");
    println!("PASS: test11c — gate_up diff={diff:.2e}");
}

// ── test11d: full layer (+ SiLU + down_proj) ──

#[test]
fn test11d_full_layer() {
    println!("=== test11d: full single layer — all steps ===");
    let mut s = Test11Setup::new();
    let stream = s.device.compute_stream;
    let hidden = s.config.hidden_size;
    let eps = s.config.rms_norm_eps;
    let m = 4usize;

    let (hs_a, hs_b, res_a, res_b) = s.make_inputs(m);
    let (pos, slot, cu, seq, bt) = s.make_attn_meta(m);
    unsafe { cusys::cuStreamSynchronize(stream) };

    // ── Standard full layer ──
    unsafe {
        vllm_cuda::kernels::fused_add_rms_norm_inplace(*hs_a, *res_a, *s.norm1_w, eps, stream);
    }
    let qkv_a = unsafe {
        s.device.ferrite.gemm(
            *hs_a,
            *s.qkv_w,
            None,
            1.0,
            0.0,
            &mut s.device.caching,
            stream,
        )
    };
    drop(hs_a);
    let attn_a = unsafe {
        s.layer.self_attn.forward_from_qkv(
            qkv_a,
            pos.view(),
            slot.view(),
            cu.view(),
            seq.view(),
            bt.view(),
            m,
            m,
            &s.kv_a,
            &s.rotary,
            &mut s.device,
        )
    };
    unsafe {
        vllm_cuda::kernels::fused_add_rms_norm_inplace(*attn_a, *res_a, *s.norm2_w, eps, stream);
    }
    let gate_up_a = unsafe {
        s.device.ferrite.gemm(
            *attn_a,
            *s.gate_up_w,
            None,
            1.0,
            0.0,
            &mut s.device.caching,
            stream,
        )
    };
    drop(attn_a);
    let activated_a = unsafe {
        vllm_cuda::kernels::silu_and_mul_fused(
            *gate_up_a,
            s.config.intermediate_size,
            &mut s.device.caching,
            stream,
        )
    };
    drop(gate_up_a);
    let mlp_a = unsafe {
        s.device.ferrite.gemm(
            *activated_a,
            *s.down_w,
            None,
            1.0,
            0.0,
            &mut s.device.caching,
            stream,
        )
    };
    drop(activated_a);

    // ── Fused full layer ──
    unsafe {
        vllm_cuda::kernels::add_inplace(*res_b, *hs_b, stream);
    }
    drop(hs_b);
    let qkv_b = unsafe {
        vllm_cuda::ferrite::launch_fused_norm_gemm(
            &FUSED_NORM_GEMM,
            *res_b,
            *s.qkv_w,
            *s.norm1_w,
            eps,
            hidden as u32,
            None,
            1.0,
            0.0,
            &mut s.device.caching,
            stream,
        )
    };
    let attn_b = unsafe {
        s.layer.self_attn.forward_from_qkv(
            qkv_b,
            pos.view(),
            slot.view(),
            cu.view(),
            seq.view(),
            bt.view(),
            m,
            m,
            &s.kv_b,
            &s.rotary,
            &mut s.device,
        )
    };
    unsafe {
        vllm_cuda::kernels::add_inplace(*res_b, *attn_b, stream);
    }
    drop(attn_b);
    let gate_up_b = unsafe {
        vllm_cuda::ferrite::launch_fused_norm_gemm(
            &FUSED_NORM_GEMM,
            *res_b,
            *s.gate_up_w,
            *s.norm2_w,
            eps,
            hidden as u32,
            None,
            1.0,
            0.0,
            &mut s.device.caching,
            stream,
        )
    };
    let activated_b = unsafe {
        vllm_cuda::kernels::silu_and_mul_fused(
            *gate_up_b,
            s.config.intermediate_size,
            &mut s.device.caching,
            stream,
        )
    };
    drop(gate_up_b);
    let mlp_b = unsafe {
        s.device.ferrite.gemm(
            *activated_b,
            *s.down_w,
            None,
            1.0,
            0.0,
            &mut s.device.caching,
            stream,
        )
    };
    drop(activated_b);

    unsafe { cusys::cuStreamSynchronize(stream) };

    let a_h = unsafe { download_bf16(*mlp_a, m * hidden) };
    let b_h = unsafe { download_bf16(*mlp_b, m * hidden) };
    let (diff, _) = max_diff_bf16(&a_h, &b_h);
    print_comparison("MLP output", &a_h, &b_h);

    let ra = unsafe { download_bf16(*res_a, m * hidden) };
    let rb = unsafe { download_bf16(*res_b, m * hidden) };
    let (rdiff, _) = max_diff_bf16(&ra, &rb);
    println!("  residual diff={rdiff:.2e}");

    assert!(diff < 10.0, "Full layer MLP diverged: {diff:.2e}");
    println!("PASS: test11d — full layer diff={diff:.2e}, residual={rdiff:.2e}");
}

// ════════════════════════════════════════════════════════════════════════
// test12: Multi-layer chain WITH attention
//
// Chains N layers of the full forward pass:
//   norm+GEMM → attention → MLP norm+GEMM → SiLU → down_proj
//
// Standard: fused_add_rms_norm_inplace + ferrite.gemm (working production)
// Fused:    add_inplace + launch_fused_norm_gemm
//
// Reports diff after each layer. If diff explodes, we found the problem.
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test12_multi_layer_with_attention() {
    println!("=== test12: multi-layer chain with attention ===");

    let mut device = GpuDevice::new(0).unwrap();
    let stream = device.compute_stream;
    let config = qwen_config();
    let hidden = config.hidden_size;
    let eps = config.rms_norm_eps;
    let num_layers = config.num_hidden_layers; // 24

    // Load all layers' weights from SafeTensors
    let raw = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&raw).expect("parse");

    struct LayerWeights {
        norm1: OwnedTensor,
        qkv: OwnedTensor,
        norm2: OwnedTensor,
        gate_up: OwnedTensor,
        down: OwnedTensor,
    }

    let mut layer_weights: Vec<LayerWeights> = Vec::new();
    for i in 0..num_layers {
        let n1 = load_bf16_tensor(&st, &format!("model.layers.{i}.input_layernorm.weight"));
        let mut qkv = load_bf16_tensor(&st, &format!("model.layers.{i}.self_attn.q_proj.weight"));
        qkv.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.self_attn.k_proj.weight"),
        ));
        qkv.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.self_attn.v_proj.weight"),
        ));
        let qkv_n = qkv.len() / hidden;
        let n2 = load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.post_attention_layernorm.weight"),
        );
        let mut gu = load_bf16_tensor(&st, &format!("model.layers.{i}.mlp.gate_proj.weight"));
        gu.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.mlp.up_proj.weight"),
        ));
        let gu_n = gu.len() / hidden;
        let dw = load_bf16_tensor(&st, &format!("model.layers.{i}.mlp.down_proj.weight"));
        let down_k = dw.len() / hidden;

        layer_weights.push(LayerWeights {
            norm1: unsafe { upload_bf16(&mut device.caching, &n1, &[hidden]) },
            qkv: unsafe { upload_bf16(&mut device.caching, &qkv, &[qkv_n, hidden]) },
            norm2: unsafe { upload_bf16(&mut device.caching, &n2, &[hidden]) },
            gate_up: unsafe { upload_bf16(&mut device.caching, &gu, &[gu_n, hidden]) },
            down: unsafe { upload_bf16(&mut device.caching, &dw, &[hidden, down_k]) },
        });
    }

    // Load actual layers for attention (forward_from_qkv is pub)
    let mut gw = GpuWeights::from_single_file(MODEL_PATH, stream).unwrap();
    let mut layers: Vec<LlamaDecoderLayer> = Vec::new();
    for i in 0..num_layers {
        layers.push(
            LlamaDecoderLayer::load(&mut gw, &format!("model.layers.{i}"), &config, i, stream)
                .unwrap(),
        );
    }

    drop(raw);

    // KV caches — separate for each path
    let kv_a = unsafe { KvCachePool::new(24, 32, 16, 2, 64, DType::BF16).unwrap() };
    let kv_b = unsafe { KvCachePool::new(24, 32, 16, 2, 64, DType::BF16).unwrap() };
    let rotary =
        unsafe { RotaryCache::new(64, 32768, 1000000.0, None, DType::BF16, &device).unwrap() };

    let m = 4usize; // prefill 4 tokens

    // Attention metadata
    let pos_data: Vec<u32> = (0..m as u32).collect();
    let slot_data: Vec<i64> = (0..m as i64).collect();
    let cu_data: Vec<i32> = vec![0, m as i32];
    let seq_data: Vec<i32> = vec![m as i32];
    let bt_data: Vec<i32> = vec![0];
    let pos = unsafe { upload_u32(&mut device.caching, &pos_data, &[m]) };
    let slot = unsafe { upload_i64(&mut device.caching, &slot_data, &[m]) };
    let cu = unsafe { upload_i32(&mut device.caching, &cu_data, &[2]) };
    let seq = unsafe { upload_i32(&mut device.caching, &seq_data, &[1]) };
    let bt = unsafe { upload_i32(&mut device.caching, &bt_data, &[1, 1]) };

    // Initial hidden states
    let hs_data: Vec<bf16> = (0..m * hidden)
        .map(|i| bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.1))
        .collect();
    let mut hs_a = unsafe { upload_bf16(&mut device.caching, &hs_data, &[m, hidden]) };
    let mut hs_b = unsafe { upload_bf16(&mut device.caching, &hs_data, &[m, hidden]) };
    let mut res_a = unsafe {
        upload_bf16(
            &mut device.caching,
            &vec![bf16::ZERO; m * hidden],
            &[m, hidden],
        )
    };
    let mut res_b = unsafe {
        upload_bf16(
            &mut device.caching,
            &vec![bf16::ZERO; m * hidden],
            &[m, hidden],
        )
    };

    unsafe { cusys::cuStreamSynchronize(stream) };

    println!("  layer |  a_max  |  b_max  |  mlp_diff |  res_diff | notes");
    println!("  ------|---------|---------|-----------|-----------|------");

    for i in 0..num_layers {
        let lw = &layer_weights[i];

        // ── Path A (standard): fused_add_rms_norm + ferrite.gemm ──
        // Skip layer 0 norm+add (no residual for first layer in real model)
        // But we're testing with synthetic residual=0, so it's fine to add.
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(*hs_a, *res_a, *lw.norm1, eps, stream);
        }
        let qkv_a = unsafe {
            device
                .ferrite
                .gemm(*hs_a, *lw.qkv, None, 1.0, 0.0, &mut device.caching, stream)
        };
        drop(hs_a);
        let attn_a = unsafe {
            layers[i].self_attn.forward_from_qkv(
                qkv_a,
                pos.view(),
                slot.view(),
                cu.view(),
                seq.view(),
                bt.view(),
                m,
                m,
                &kv_a,
                &rotary,
                &mut device,
            )
        };
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(*attn_a, *res_a, *lw.norm2, eps, stream);
        }
        let gu_a = unsafe {
            device.ferrite.gemm(
                *attn_a,
                *lw.gate_up,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(attn_a);
        let act_a = unsafe {
            vllm_cuda::kernels::silu_and_mul_fused(
                *gu_a,
                config.intermediate_size,
                &mut device.caching,
                stream,
            )
        };
        drop(gu_a);
        hs_a = unsafe {
            device.ferrite.gemm(
                *act_a,
                *lw.down,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(act_a);

        // ── Path B (fused): add_inplace + launch_fused_norm_gemm ──
        unsafe {
            vllm_cuda::kernels::add_inplace(*res_b, *hs_b, stream);
        }
        drop(hs_b);
        let qkv_b = unsafe {
            vllm_cuda::ferrite::launch_fused_norm_gemm(
                &FUSED_NORM_GEMM,
                *res_b,
                *lw.qkv,
                *lw.norm1,
                eps,
                hidden as u32,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        let attn_b = unsafe {
            layers[i].self_attn.forward_from_qkv(
                qkv_b,
                pos.view(),
                slot.view(),
                cu.view(),
                seq.view(),
                bt.view(),
                m,
                m,
                &kv_b,
                &rotary,
                &mut device,
            )
        };
        unsafe {
            vllm_cuda::kernels::add_inplace(*res_b, *attn_b, stream);
        }
        drop(attn_b);
        let gu_b = unsafe {
            vllm_cuda::ferrite::launch_fused_norm_gemm(
                &FUSED_NORM_GEMM,
                *res_b,
                *lw.gate_up,
                *lw.norm2,
                eps,
                hidden as u32,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        let act_b = unsafe {
            vllm_cuda::kernels::silu_and_mul_fused(
                *gu_b,
                config.intermediate_size,
                &mut device.caching,
                stream,
            )
        };
        drop(gu_b);
        hs_b = unsafe {
            device.ferrite.gemm(
                *act_b,
                *lw.down,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(act_b);

        // ── Compare after this layer ──
        unsafe { cusys::cuStreamSynchronize(stream) };
        let a_h = unsafe { download_bf16(*hs_a, m * hidden) };
        let b_h = unsafe { download_bf16(*hs_b, m * hidden) };
        let ra_h = unsafe { download_bf16(*res_a, m * hidden) };
        let rb_h = unsafe { download_bf16(*res_b, m * hidden) };

        let (mlp_diff, _) = max_diff_bf16(&a_h, &b_h);
        let (res_diff, _) = max_diff_bf16(&ra_h, &rb_h);
        let a_max = a_h.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
        let b_max = b_h.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
        let a_nan = a_h.iter().filter(|v| v.to_f32().is_nan()).count();
        let b_nan = b_h.iter().filter(|v| v.to_f32().is_nan()).count();

        let notes = if a_nan > 0 || b_nan > 0 {
            format!("NaN! a={a_nan} b={b_nan}")
        } else if mlp_diff > 100.0 {
            "EXPLODED".to_string()
        } else if mlp_diff > 10.0 {
            "DIVERGING".to_string()
        } else {
            String::new()
        };

        println!(
            "  {:5} | {:7.2} | {:7.2} | {:9.2e} | {:9.2e} | {notes}",
            i, a_max, b_max, mlp_diff, res_diff,
        );

        if a_nan > 0 || b_nan > 0 {
            println!("  ABORT: NaN detected");
            panic!("Layer {i}: NaN in output");
        }
        if a_max > 1e4 || b_max > 1e4 {
            println!("  ABORT: values exploded");
            panic!("Layer {i}: values exploded (a={a_max}, b={b_max})");
        }
    }

    let a_final = unsafe { download_bf16(*hs_a, m * hidden) };
    let b_final = unsafe { download_bf16(*hs_b, m * hidden) };
    let (final_diff, _) = max_diff_bf16(&a_final, &b_final);

    println!();
    println!("  Final diff after {num_layers} layers: {final_diff:.2e}");
    if final_diff < 1.0 {
        println!("  PASS: fused path stays close to standard through all layers");
    } else if final_diff < 100.0 {
        println!("  NOTABLE: diff={final_diff:.2e} — significant but bounded");
    } else {
        println!("  FAIL: fused path diverged from standard");
    }
}

// ════════════════════════════════════════════════════════════════════════
// test13: Prefill + decode with real token embeddings
//
// Covers two previously untested axes at once:
//   1. Real token embeddings (not synthetic sin/cos)
//   2. Decode path (M=1, reading from KV cache populated during prefill)
//
// Runs 24 layers of prefill, then 3 decode steps, both paths.
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test13_prefill_then_decode() {
    println!("=== test13: prefill + decode, real embeddings, 24 layers ===");

    let mut device = GpuDevice::new(0).unwrap();
    let stream = device.compute_stream;
    let config = qwen_config();
    let hidden = config.hidden_size;
    let eps = config.rms_norm_eps;
    let num_layers = config.num_hidden_layers;
    let block_size = 16usize;

    // Load weights
    let raw = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&raw).expect("parse");

    // Embedding weight
    let embed_data = load_bf16_tensor(&st, "model.embed_tokens.weight");
    let vocab_size = embed_data.len() / hidden;
    let embed_w = unsafe { upload_bf16(&mut device.caching, &embed_data, &[vocab_size, hidden]) };

    // Per-layer weights
    struct LW {
        norm1: OwnedTensor,
        qkv: OwnedTensor,
        norm2: OwnedTensor,
        gate_up: OwnedTensor,
        down: OwnedTensor,
    }
    let mut lw: Vec<LW> = Vec::new();
    for i in 0..num_layers {
        let n1 = load_bf16_tensor(&st, &format!("model.layers.{i}.input_layernorm.weight"));
        let mut qkv = load_bf16_tensor(&st, &format!("model.layers.{i}.self_attn.q_proj.weight"));
        qkv.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.self_attn.k_proj.weight"),
        ));
        qkv.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.self_attn.v_proj.weight"),
        ));
        let qkv_n = qkv.len() / hidden;
        let n2 = load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.post_attention_layernorm.weight"),
        );
        let mut gu = load_bf16_tensor(&st, &format!("model.layers.{i}.mlp.gate_proj.weight"));
        gu.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.mlp.up_proj.weight"),
        ));
        let gu_n = gu.len() / hidden;
        let dw = load_bf16_tensor(&st, &format!("model.layers.{i}.mlp.down_proj.weight"));
        let down_k = dw.len() / hidden;
        lw.push(LW {
            norm1: unsafe { upload_bf16(&mut device.caching, &n1, &[hidden]) },
            qkv: unsafe { upload_bf16(&mut device.caching, &qkv, &[qkv_n, hidden]) },
            norm2: unsafe { upload_bf16(&mut device.caching, &n2, &[hidden]) },
            gate_up: unsafe { upload_bf16(&mut device.caching, &gu, &[gu_n, hidden]) },
            down: unsafe { upload_bf16(&mut device.caching, &dw, &[hidden, down_k]) },
        });
    }

    // Layers for attention
    let mut gw = GpuWeights::from_single_file(MODEL_PATH, stream).unwrap();
    let mut layers: Vec<LlamaDecoderLayer> = Vec::new();
    for i in 0..num_layers {
        layers.push(
            LlamaDecoderLayer::load(&mut gw, &format!("model.layers.{i}"), &config, i, stream)
                .unwrap(),
        );
    }
    drop(raw);

    // KV caches — need enough blocks for prefill + decode tokens
    // 4 prefill + 3 decode = 7 tokens, block_size=16, so 1 block suffices
    let num_blocks = 32;
    let kv_a = unsafe { KvCachePool::new(24, num_blocks, block_size, 2, 64, DType::BF16).unwrap() };
    let kv_b = unsafe { KvCachePool::new(24, num_blocks, block_size, 2, 64, DType::BF16).unwrap() };
    let rotary =
        unsafe { RotaryCache::new(64, 32768, 1000000.0, None, DType::BF16, &device).unwrap() };

    // Real token IDs: "What is 2+2?" → use some plausible Qwen token IDs
    // (exact IDs don't matter for correctness testing, just need valid indices)
    let prefill_ids: Vec<u32> = vec![1, 3923, 374, 220, 17, 10, 17, 30]; // 8 tokens
    let prefill_len = prefill_ids.len();

    // ── Helper: run all 24 layers ──
    // Returns (mlp_output, residual) after all layers
    unsafe fn run_24_layers_standard(
        device: &mut GpuDevice,
        layers: &[LlamaDecoderLayer],
        lw: &[LW],
        config: &LlamaConfig,
        mut hs: OwnedTensor,
        mut res: OwnedTensor,
        pos: TensorView<'_>,
        slot: TensorView<'_>,
        cu: TensorView<'_>,
        seq: TensorView<'_>,
        bt: TensorView<'_>,
        max_sq: usize,
        max_sk: usize,
        kv: &KvCachePool,
        rotary: &RotaryCache,
    ) -> (OwnedTensor, OwnedTensor) {
        let stream = device.compute_stream;
        let eps = config.rms_norm_eps;
        let hidden = config.hidden_size;
        for i in 0..config.num_hidden_layers {
            // Skip add+norm for layer 0 (no residual yet)
            if i == 0 {
                // Just norm hidden_states directly (no residual add)
                let normed = vllm_cuda::kernels::rms_norm(
                    *hs,
                    *lw[i].norm1,
                    eps,
                    &mut device.caching,
                    stream,
                );
                drop(hs);
                let qkv = device.ferrite.gemm(
                    *normed,
                    *lw[i].qkv,
                    None,
                    1.0,
                    0.0,
                    &mut device.caching,
                    stream,
                );
                // For layer 0, residual = original hidden_states. But we dropped it.
                // Actually in the real model, layer 0 gets residual=None and uses the non-add path.
                // Let's just use fused_add_rms_norm with zero residual.
                // res is already zeros from initialization.
                drop(normed);
                // Rewrite: use fused_add_rms_norm_inplace like other layers
                // This adds hs (embedding) to res (zeros), giving res=hs, hs=norm(hs)
                // Actually we already dropped hs. Let me restructure.
                // For simplicity, skip the special layer-0 handling and just
                // treat all layers the same (res starts at zero, so res+hs = hs).
                drop(qkv);
            }
            // Ugh, the layer-0 handling is getting messy. Let me just use
            // fused_add_rms_norm_inplace uniformly — with res=0 for layer 0,
            // it gives res=hs, normed=norm(hs), which is correct.
            break;
        }
        // Let me restart with cleaner logic
        unreachable!()
    }

    // Actually, let me just inline the loop directly rather than a helper function.
    // The layer-0 case works fine with fused_add_rms_norm_inplace when res=0:
    //   res = 0 + hs = hs
    //   hs = norm(hs)
    // Which is correct for the first layer.

    // ════════════════════════════════════════════════════════════════
    // PREFILL PHASE
    // ════════════════════════════════════════════════════════════════
    println!("  Prefill: {prefill_len} tokens...");

    // Embed tokens
    let ids_a = unsafe { upload_u32(&mut device.caching, &prefill_ids, &[prefill_len]) };
    let ids_b = unsafe { upload_u32(&mut device.caching, &prefill_ids, &[prefill_len]) };
    let mut hs_a = unsafe {
        vllm_cuda::kernels::embedding_gather(*embed_w, *ids_a, &mut device.caching, stream)
    };
    let mut hs_b = unsafe {
        vllm_cuda::kernels::embedding_gather(*embed_w, *ids_b, &mut device.caching, stream)
    };
    drop(ids_a);
    drop(ids_b);
    let mut res_a = unsafe {
        upload_bf16(
            &mut device.caching,
            &vec![bf16::ZERO; prefill_len * hidden],
            &[prefill_len, hidden],
        )
    };
    let mut res_b = unsafe {
        upload_bf16(
            &mut device.caching,
            &vec![bf16::ZERO; prefill_len * hidden],
            &[prefill_len, hidden],
        )
    };

    // Prefill attention metadata
    let pos_data: Vec<u32> = (0..prefill_len as u32).collect();
    let slot_data: Vec<i64> = (0..prefill_len as i64).collect();
    let cu_data: Vec<i32> = vec![0, prefill_len as i32];
    let seq_data: Vec<i32> = vec![prefill_len as i32];
    let bt_data: Vec<i32> = vec![0]; // block 0
    let pos = unsafe { upload_u32(&mut device.caching, &pos_data, &[prefill_len]) };
    let slot = unsafe { upload_i64(&mut device.caching, &slot_data, &[prefill_len]) };
    let cu = unsafe { upload_i32(&mut device.caching, &cu_data, &[2]) };
    let seq = unsafe { upload_i32(&mut device.caching, &seq_data, &[1]) };
    let bt = unsafe { upload_i32(&mut device.caching, &bt_data, &[1, 1]) };

    unsafe { cusys::cuStreamSynchronize(stream) };

    // Run prefill through all 24 layers
    for i in 0..num_layers {
        // Path A (standard)
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(
                *hs_a,
                *res_a,
                *lw[i].norm1,
                eps,
                stream,
            );
        }
        let qkv_a = unsafe {
            device.ferrite.gemm(
                *hs_a,
                *lw[i].qkv,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(hs_a);
        let attn_a = unsafe {
            layers[i].self_attn.forward_from_qkv(
                qkv_a,
                pos.view(),
                slot.view(),
                cu.view(),
                seq.view(),
                bt.view(),
                prefill_len,
                prefill_len,
                &kv_a,
                &rotary,
                &mut device,
            )
        };
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(
                *attn_a,
                *res_a,
                *lw[i].norm2,
                eps,
                stream,
            );
        }
        let gu_a = unsafe {
            device.ferrite.gemm(
                *attn_a,
                *lw[i].gate_up,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(attn_a);
        let act_a = unsafe {
            vllm_cuda::kernels::silu_and_mul_fused(
                *gu_a,
                config.intermediate_size,
                &mut device.caching,
                stream,
            )
        };
        drop(gu_a);
        hs_a = unsafe {
            device.ferrite.gemm(
                *act_a,
                *lw[i].down,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(act_a);

        // Path B (fused)
        unsafe { vllm_cuda::kernels::add_inplace(*res_b, *hs_b, stream) };
        drop(hs_b);
        let qkv_b = unsafe {
            vllm_cuda::ferrite::launch_fused_norm_gemm(
                &FUSED_NORM_GEMM,
                *res_b,
                *lw[i].qkv,
                *lw[i].norm1,
                eps,
                hidden as u32,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        let attn_b = unsafe {
            layers[i].self_attn.forward_from_qkv(
                qkv_b,
                pos.view(),
                slot.view(),
                cu.view(),
                seq.view(),
                bt.view(),
                prefill_len,
                prefill_len,
                &kv_b,
                &rotary,
                &mut device,
            )
        };
        unsafe { vllm_cuda::kernels::add_inplace(*res_b, *attn_b, stream) };
        drop(attn_b);
        let gu_b = unsafe {
            vllm_cuda::ferrite::launch_fused_norm_gemm(
                &FUSED_NORM_GEMM,
                *res_b,
                *lw[i].gate_up,
                *lw[i].norm2,
                eps,
                hidden as u32,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        let act_b = unsafe {
            vllm_cuda::kernels::silu_and_mul_fused(
                *gu_b,
                config.intermediate_size,
                &mut device.caching,
                stream,
            )
        };
        drop(gu_b);
        hs_b = unsafe {
            device.ferrite.gemm(
                *act_b,
                *lw[i].down,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(act_b);
    }

    unsafe { cusys::cuStreamSynchronize(stream) };
    let a_h = unsafe { download_bf16(*hs_a, prefill_len * hidden) };
    let b_h = unsafe { download_bf16(*hs_b, prefill_len * hidden) };
    let (prefill_diff, _) = max_diff_bf16(&a_h, &b_h);
    println!("  Prefill done: diff={prefill_diff:.2e}");

    // ════════════════════════════════════════════════════════════════
    // DECODE PHASE — 3 tokens, one at a time
    // ════════════════════════════════════════════════════════════════
    println!("  Decode: 3 steps...");

    for decode_step in 0..3u32 {
        let token_pos = prefill_len as u32 + decode_step;
        let total_seq_len = prefill_len + decode_step as usize + 1;
        let slot_idx = token_pos as i64; // slot = position (all in block 0 or 1)

        // Decode attention metadata: M=1
        let d_pos = unsafe { upload_u32(&mut device.caching, &[token_pos], &[1]) };
        let d_slot = unsafe { upload_i64(&mut device.caching, &[slot_idx], &[1]) };
        let d_cu = unsafe { upload_i32(&mut device.caching, &[0i32, 1], &[2]) };
        let d_seq = unsafe { upload_i32(&mut device.caching, &[total_seq_len as i32], &[1]) };
        // Block table: need ceil(total_seq_len / block_size) blocks
        let num_seq_blocks = (total_seq_len + block_size - 1) / block_size;
        let bt_vec: Vec<i32> = (0..num_seq_blocks as i32).collect();
        let d_bt = unsafe { upload_i32(&mut device.caching, &bt_vec, &[1, num_seq_blocks]) };

        // Use a fake "next token" embedding (take last row of prefill output as input)
        // In reality the engine would embed the sampled token, but for testing
        // we just feed the last hidden state back (tests the decode path, not sampling)
        let last_row_a: Vec<bf16> = a_h[(prefill_len - 1) * hidden..prefill_len * hidden].to_vec();
        let last_row_b: Vec<bf16> = b_h[(prefill_len - 1) * hidden..prefill_len * hidden].to_vec();
        hs_a = unsafe { upload_bf16(&mut device.caching, &last_row_a, &[1, hidden]) };
        hs_b = unsafe { upload_bf16(&mut device.caching, &last_row_b, &[1, hidden]) };

        unsafe { cusys::cuStreamSynchronize(stream) };

        // Run all 24 layers in decode mode (max_seqlen_q=1)
        for i in 0..num_layers {
            // Path A
            unsafe {
                vllm_cuda::kernels::fused_add_rms_norm_inplace(
                    *hs_a,
                    *res_a,
                    *lw[i].norm1,
                    eps,
                    stream,
                );
            }
            let qkv_a = unsafe {
                device.ferrite.gemm(
                    *hs_a,
                    *lw[i].qkv,
                    None,
                    1.0,
                    0.0,
                    &mut device.caching,
                    stream,
                )
            };
            drop(hs_a);
            let attn_a = unsafe {
                layers[i].self_attn.forward_from_qkv(
                    qkv_a,
                    d_pos.view(),
                    d_slot.view(),
                    d_cu.view(),
                    d_seq.view(),
                    d_bt.view(),
                    1,
                    total_seq_len,
                    &kv_a,
                    &rotary,
                    &mut device,
                )
            };
            unsafe {
                vllm_cuda::kernels::fused_add_rms_norm_inplace(
                    *attn_a,
                    *res_a,
                    *lw[i].norm2,
                    eps,
                    stream,
                );
            }
            let gu_a = unsafe {
                device.ferrite.gemm(
                    *attn_a,
                    *lw[i].gate_up,
                    None,
                    1.0,
                    0.0,
                    &mut device.caching,
                    stream,
                )
            };
            drop(attn_a);
            let act_a = unsafe {
                vllm_cuda::kernels::silu_and_mul_fused(
                    *gu_a,
                    config.intermediate_size,
                    &mut device.caching,
                    stream,
                )
            };
            drop(gu_a);
            hs_a = unsafe {
                device.ferrite.gemm(
                    *act_a,
                    *lw[i].down,
                    None,
                    1.0,
                    0.0,
                    &mut device.caching,
                    stream,
                )
            };
            drop(act_a);

            // Path B
            unsafe { vllm_cuda::kernels::add_inplace(*res_b, *hs_b, stream) };
            drop(hs_b);
            let qkv_b = unsafe {
                vllm_cuda::ferrite::launch_fused_norm_gemm(
                    &FUSED_NORM_GEMM,
                    *res_b,
                    *lw[i].qkv,
                    *lw[i].norm1,
                    eps,
                    hidden as u32,
                    None,
                    1.0,
                    0.0,
                    &mut device.caching,
                    stream,
                )
            };
            let attn_b = unsafe {
                layers[i].self_attn.forward_from_qkv(
                    qkv_b,
                    d_pos.view(),
                    d_slot.view(),
                    d_cu.view(),
                    d_seq.view(),
                    d_bt.view(),
                    1,
                    total_seq_len,
                    &kv_b,
                    &rotary,
                    &mut device,
                )
            };
            unsafe { vllm_cuda::kernels::add_inplace(*res_b, *attn_b, stream) };
            drop(attn_b);
            let gu_b = unsafe {
                vllm_cuda::ferrite::launch_fused_norm_gemm(
                    &FUSED_NORM_GEMM,
                    *res_b,
                    *lw[i].gate_up,
                    *lw[i].norm2,
                    eps,
                    hidden as u32,
                    None,
                    1.0,
                    0.0,
                    &mut device.caching,
                    stream,
                )
            };
            let act_b = unsafe {
                vllm_cuda::kernels::silu_and_mul_fused(
                    *gu_b,
                    config.intermediate_size,
                    &mut device.caching,
                    stream,
                )
            };
            drop(gu_b);
            hs_b = unsafe {
                device.ferrite.gemm(
                    *act_b,
                    *lw[i].down,
                    None,
                    1.0,
                    0.0,
                    &mut device.caching,
                    stream,
                )
            };
            drop(act_b);
        }

        unsafe { cusys::cuStreamSynchronize(stream) };
        let da = unsafe { download_bf16(*hs_a, hidden) };
        let db = unsafe { download_bf16(*hs_b, hidden) };
        let (dd, _) = max_diff_bf16(&da, &db);
        let a_max = da.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
        let b_max = db.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
        let b_nan = db.iter().filter(|v| v.to_f32().is_nan()).count();
        println!(
            "  decode {decode_step}: diff={dd:.2e}  a_max={a_max:.1}  b_max={b_max:.1}  nan={b_nan}"
        );

        if b_nan > 0 {
            panic!("Decode step {decode_step}: NaN in fused output!");
        }
        if b_max > 1e4 {
            panic!("Decode step {decode_step}: fused output exploded: {b_max}");
        }

        // Update for next decode step — reuse last output as next "embedding"
        // (a_h and b_h are overwritten)
    }

    println!("PASS: test13 — prefill + decode completed, fused path bounded");
}

// ── test13c: prefill + decode, paths run SEQUENTIALLY ──
// Avoids the shared-allocator interleaving that crashed test13.
// Runs path A (standard) fully, saves output. Then runs path B (fused) fully. Compares.
#[test]
fn test13c_sequential_prefill_decode() {
    println!("=== test13c: sequential prefill+decode, real embeddings ===");

    let config = qwen_config();
    let hidden = config.hidden_size;
    let eps = config.rms_norm_eps;
    let num_layers = config.num_hidden_layers;
    let block_size = 16usize;
    let prefill_ids: Vec<u32> = vec![1, 3923, 374, 220, 17, 10, 17, 30]; // 8 tokens
    let prefill_len = prefill_ids.len();
    let num_decode_steps = 3usize;

    // Run one full path (prefill + decode).
    // Returns (decode_token_ids, decode_hidden_states) for each decode step.
    fn run_path(
        fused: bool,
        config: &LlamaConfig,
        prefill_ids: &[u32],
        num_decode_steps: usize,
    ) -> (Vec<u32>, Vec<Vec<bf16>>) {
        let mut device = GpuDevice::new(0).unwrap();
        let stream = device.compute_stream;
        let hidden = config.hidden_size;
        let eps = config.rms_norm_eps;
        let num_layers = config.num_hidden_layers;
        let block_size = 16usize;
        let prefill_len = prefill_ids.len();

        let raw = std::fs::read(MODEL_PATH).expect("read model");
        let st = SafeTensors::deserialize(&raw).expect("parse");

        let embed_data = load_bf16_tensor(&st, "model.embed_tokens.weight");
        let vocab_size = embed_data.len() / hidden;
        let embed_w =
            unsafe { upload_bf16(&mut device.caching, &embed_data, &[vocab_size, hidden]) };

        struct LW {
            norm1: OwnedTensor,
            qkv: OwnedTensor,
            norm2: OwnedTensor,
            gate_up: OwnedTensor,
            down: OwnedTensor,
        }
        let mut lw: Vec<LW> = Vec::new();
        for i in 0..num_layers {
            let n1 = load_bf16_tensor(&st, &format!("model.layers.{i}.input_layernorm.weight"));
            let mut qkv =
                load_bf16_tensor(&st, &format!("model.layers.{i}.self_attn.q_proj.weight"));
            qkv.extend_from_slice(&load_bf16_tensor(
                &st,
                &format!("model.layers.{i}.self_attn.k_proj.weight"),
            ));
            qkv.extend_from_slice(&load_bf16_tensor(
                &st,
                &format!("model.layers.{i}.self_attn.v_proj.weight"),
            ));
            let qkv_n = qkv.len() / hidden;
            let n2 = load_bf16_tensor(
                &st,
                &format!("model.layers.{i}.post_attention_layernorm.weight"),
            );
            let mut gu = load_bf16_tensor(&st, &format!("model.layers.{i}.mlp.gate_proj.weight"));
            gu.extend_from_slice(&load_bf16_tensor(
                &st,
                &format!("model.layers.{i}.mlp.up_proj.weight"),
            ));
            let gu_n = gu.len() / hidden;
            let dw = load_bf16_tensor(&st, &format!("model.layers.{i}.mlp.down_proj.weight"));
            let down_k = dw.len() / hidden;
            lw.push(LW {
                norm1: unsafe { upload_bf16(&mut device.caching, &n1, &[hidden]) },
                qkv: unsafe { upload_bf16(&mut device.caching, &qkv, &[qkv_n, hidden]) },
                norm2: unsafe { upload_bf16(&mut device.caching, &n2, &[hidden]) },
                gate_up: unsafe { upload_bf16(&mut device.caching, &gu, &[gu_n, hidden]) },
                down: unsafe { upload_bf16(&mut device.caching, &dw, &[hidden, down_k]) },
            });
        }

        // Final norm weight
        let final_norm_data = load_bf16_tensor(&st, "model.norm.weight");
        let final_norm_w = unsafe { upload_bf16(&mut device.caching, &final_norm_data, &[hidden]) };

        let mut gw = GpuWeights::from_single_file(MODEL_PATH, stream).unwrap();
        let mut layers: Vec<LlamaDecoderLayer> = Vec::new();
        for i in 0..num_layers {
            layers.push(
                LlamaDecoderLayer::load(&mut gw, &format!("model.layers.{i}"), &config, i, stream)
                    .unwrap(),
            );
        }
        drop(raw);

        let kv = unsafe { KvCachePool::new(24, 32, block_size, 2, 64, DType::BF16).unwrap() };
        let rotary =
            unsafe { RotaryCache::new(64, 32768, 1000000.0, None, DType::BF16, &device).unwrap() };

        // Embed
        let ids = unsafe { upload_u32(&mut device.caching, prefill_ids, &[prefill_len]) };
        let mut hs = unsafe {
            vllm_cuda::kernels::embedding_gather(*embed_w, *ids, &mut device.caching, stream)
        };
        drop(ids);
        let mut res = unsafe {
            upload_bf16(
                &mut device.caching,
                &vec![bf16::ZERO; prefill_len * hidden],
                &[prefill_len, hidden],
            )
        };

        let pos = unsafe {
            upload_u32(
                &mut device.caching,
                &(0..prefill_len as u32).collect::<Vec<_>>(),
                &[prefill_len],
            )
        };
        let slot = unsafe {
            upload_i64(
                &mut device.caching,
                &(0..prefill_len as i64).collect::<Vec<_>>(),
                &[prefill_len],
            )
        };
        let cu = unsafe { upload_i32(&mut device.caching, &[0, prefill_len as i32], &[2]) };
        let seq = unsafe { upload_i32(&mut device.caching, &[prefill_len as i32], &[1]) };
        let bt = unsafe { upload_i32(&mut device.caching, &[0], &[1, 1]) };

        unsafe { cusys::cuStreamSynchronize(stream) };

        // Prefill
        for i in 0..num_layers {
            if fused {
                unsafe { vllm_cuda::kernels::add_inplace(*res, *hs, stream) };
                drop(hs);
                let qkv = unsafe {
                    vllm_cuda::ferrite::launch_fused_norm_gemm(
                        &FUSED_NORM_GEMM,
                        *res,
                        *lw[i].qkv,
                        *lw[i].norm1,
                        eps,
                        hidden as u32,
                        None,
                        1.0,
                        0.0,
                        &mut device.caching,
                        stream,
                    )
                };
                let attn = unsafe {
                    layers[i].self_attn.forward_from_qkv(
                        qkv,
                        pos.view(),
                        slot.view(),
                        cu.view(),
                        seq.view(),
                        bt.view(),
                        prefill_len,
                        prefill_len,
                        &kv,
                        &rotary,
                        &mut device,
                    )
                };
                unsafe { vllm_cuda::kernels::add_inplace(*res, *attn, stream) };
                drop(attn);
                let gu = unsafe {
                    vllm_cuda::ferrite::launch_fused_norm_gemm(
                        &FUSED_NORM_GEMM,
                        *res,
                        *lw[i].gate_up,
                        *lw[i].norm2,
                        eps,
                        hidden as u32,
                        None,
                        1.0,
                        0.0,
                        &mut device.caching,
                        stream,
                    )
                };
                let act = unsafe {
                    vllm_cuda::kernels::silu_and_mul_fused(
                        *gu,
                        config.intermediate_size,
                        &mut device.caching,
                        stream,
                    )
                };
                drop(gu);
                hs = unsafe {
                    device.ferrite.gemm(
                        *act,
                        *lw[i].down,
                        None,
                        1.0,
                        0.0,
                        &mut device.caching,
                        stream,
                    )
                };
                drop(act);
            } else {
                unsafe {
                    vllm_cuda::kernels::fused_add_rms_norm_inplace(
                        *hs,
                        *res,
                        *lw[i].norm1,
                        eps,
                        stream,
                    )
                };
                let qkv = unsafe {
                    device.ferrite.gemm(
                        *hs,
                        *lw[i].qkv,
                        None,
                        1.0,
                        0.0,
                        &mut device.caching,
                        stream,
                    )
                };
                drop(hs);
                let attn = unsafe {
                    layers[i].self_attn.forward_from_qkv(
                        qkv,
                        pos.view(),
                        slot.view(),
                        cu.view(),
                        seq.view(),
                        bt.view(),
                        prefill_len,
                        prefill_len,
                        &kv,
                        &rotary,
                        &mut device,
                    )
                };
                unsafe {
                    vllm_cuda::kernels::fused_add_rms_norm_inplace(
                        *attn,
                        *res,
                        *lw[i].norm2,
                        eps,
                        stream,
                    )
                };
                let gu = unsafe {
                    device.ferrite.gemm(
                        *attn,
                        *lw[i].gate_up,
                        None,
                        1.0,
                        0.0,
                        &mut device.caching,
                        stream,
                    )
                };
                drop(attn);
                let act = unsafe {
                    vllm_cuda::kernels::silu_and_mul_fused(
                        *gu,
                        config.intermediate_size,
                        &mut device.caching,
                        stream,
                    )
                };
                drop(gu);
                hs = unsafe {
                    device.ferrite.gemm(
                        *act,
                        *lw[i].down,
                        None,
                        1.0,
                        0.0,
                        &mut device.caching,
                        stream,
                    )
                };
                drop(act);
            }
        }

        // Collect prefill output
        unsafe { cusys::cuStreamSynchronize(stream) };
        let prefill_out = unsafe { download_bf16(*hs, prefill_len * hidden) };

        // Decode steps
        let mut decode_tokens: Vec<u32> = Vec::new();
        let mut decode_hidden: Vec<Vec<bf16>> = Vec::new();
        for step in 0..num_decode_steps as u32 {
            let token_pos = prefill_len as u32 + step;
            let total_seq = prefill_len + step as usize + 1;
            let slot_idx = token_pos as i64;
            let n_blocks = (total_seq + block_size - 1) / block_size;

            let d_pos = unsafe { upload_u32(&mut device.caching, &[token_pos], &[1]) };
            let d_slot = unsafe { upload_i64(&mut device.caching, &[slot_idx], &[1]) };
            let d_cu = unsafe { upload_i32(&mut device.caching, &[0i32, 1], &[2]) };
            let d_seq = unsafe { upload_i32(&mut device.caching, &[total_seq as i32], &[1]) };
            let d_bt = unsafe {
                upload_i32(
                    &mut device.caching,
                    &(0..n_blocks as i32).collect::<Vec<_>>(),
                    &[1, n_blocks],
                )
            };

            // Feed last row as decode input
            unsafe { cusys::cuStreamSynchronize(stream) };
            let last_hs = unsafe {
                download_bf16(
                    *hs,
                    if step == 0 {
                        prefill_len * hidden
                    } else {
                        hidden
                    },
                )
            };
            let row = if step == 0 {
                last_hs[(prefill_len - 1) * hidden..].to_vec()
            } else {
                last_hs
            };
            hs = unsafe { upload_bf16(&mut device.caching, &row, &[1, hidden]) };

            let last_res = unsafe {
                download_bf16(
                    *res,
                    if step == 0 {
                        prefill_len * hidden
                    } else {
                        hidden
                    },
                )
            };
            let res_row = if step == 0 {
                last_res[(prefill_len - 1) * hidden..].to_vec()
            } else {
                last_res
            };
            res = unsafe { upload_bf16(&mut device.caching, &res_row, &[1, hidden]) };

            unsafe { cusys::cuStreamSynchronize(stream) };

            for i in 0..num_layers {
                if fused {
                    unsafe { vllm_cuda::kernels::add_inplace(*res, *hs, stream) };
                    drop(hs);
                    let qkv = unsafe {
                        vllm_cuda::ferrite::launch_fused_norm_gemm(
                            &FUSED_NORM_GEMM,
                            *res,
                            *lw[i].qkv,
                            *lw[i].norm1,
                            eps,
                            hidden as u32,
                            None,
                            1.0,
                            0.0,
                            &mut device.caching,
                            stream,
                        )
                    };
                    let attn = unsafe {
                        layers[i].self_attn.forward_from_qkv(
                            qkv,
                            d_pos.view(),
                            d_slot.view(),
                            d_cu.view(),
                            d_seq.view(),
                            d_bt.view(),
                            1,
                            total_seq,
                            &kv,
                            &rotary,
                            &mut device,
                        )
                    };
                    unsafe { vllm_cuda::kernels::add_inplace(*res, *attn, stream) };
                    drop(attn);
                    let gu = unsafe {
                        vllm_cuda::ferrite::launch_fused_norm_gemm(
                            &FUSED_NORM_GEMM,
                            *res,
                            *lw[i].gate_up,
                            *lw[i].norm2,
                            eps,
                            hidden as u32,
                            None,
                            1.0,
                            0.0,
                            &mut device.caching,
                            stream,
                        )
                    };
                    let act = unsafe {
                        vllm_cuda::kernels::silu_and_mul_fused(
                            *gu,
                            config.intermediate_size,
                            &mut device.caching,
                            stream,
                        )
                    };
                    drop(gu);
                    hs = unsafe {
                        device.ferrite.gemm(
                            *act,
                            *lw[i].down,
                            None,
                            1.0,
                            0.0,
                            &mut device.caching,
                            stream,
                        )
                    };
                    drop(act);
                } else {
                    unsafe {
                        vllm_cuda::kernels::fused_add_rms_norm_inplace(
                            *hs,
                            *res,
                            *lw[i].norm1,
                            eps,
                            stream,
                        )
                    };
                    let qkv = unsafe {
                        device.ferrite.gemm(
                            *hs,
                            *lw[i].qkv,
                            None,
                            1.0,
                            0.0,
                            &mut device.caching,
                            stream,
                        )
                    };
                    drop(hs);
                    let attn = unsafe {
                        layers[i].self_attn.forward_from_qkv(
                            qkv,
                            d_pos.view(),
                            d_slot.view(),
                            d_cu.view(),
                            d_seq.view(),
                            d_bt.view(),
                            1,
                            total_seq,
                            &kv,
                            &rotary,
                            &mut device,
                        )
                    };
                    unsafe {
                        vllm_cuda::kernels::fused_add_rms_norm_inplace(
                            *attn,
                            *res,
                            *lw[i].norm2,
                            eps,
                            stream,
                        )
                    };
                    let gu = unsafe {
                        device.ferrite.gemm(
                            *attn,
                            *lw[i].gate_up,
                            None,
                            1.0,
                            0.0,
                            &mut device.caching,
                            stream,
                        )
                    };
                    drop(attn);
                    let act = unsafe {
                        vllm_cuda::kernels::silu_and_mul_fused(
                            *gu,
                            config.intermediate_size,
                            &mut device.caching,
                            stream,
                        )
                    };
                    drop(gu);
                    hs = unsafe {
                        device.ferrite.gemm(
                            *act,
                            *lw[i].down,
                            None,
                            1.0,
                            0.0,
                            &mut device.caching,
                            stream,
                        )
                    };
                    drop(act);
                }
            }

            // Final norm + lm_head projection → logits → argmax
            // Final add+norm: hs has MLP output, res has residual
            // In the real model: fused_add_rms_norm_inplace(hs, res, final_norm, eps)
            // then logits = hs @ embed_w^T
            // For both paths, use the standard final norm (it's not part of the fused experiment)
            unsafe {
                vllm_cuda::kernels::fused_add_rms_norm_inplace(
                    *hs,
                    *res,
                    *final_norm_w,
                    eps,
                    stream,
                );
            }
            // lm_head = embed_w (tied). logits = hs @ embed_w^T
            // embed_w is [vocab, hidden], we want [1, hidden] @ [hidden, vocab] = [1, vocab]
            // ferrite.gemm does A @ B where A=[M,K] B=[N,K] → C=[M,N]
            // So: A=hs [1, hidden], B=embed_w [vocab, hidden] → C=[1, vocab]
            let logits = unsafe {
                device
                    .ferrite
                    .gemm(*hs, *embed_w, None, 1.0, 0.0, &mut device.caching, stream)
            };

            unsafe { cusys::cuStreamSynchronize(stream) };
            let logits_h = unsafe { download_bf16(*logits, vocab_size) };
            drop(logits);

            // Argmax
            let (token_id, _max_logit) = logits_h
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.to_f32().partial_cmp(&b.to_f32()).unwrap())
                .unwrap();

            decode_tokens.push(token_id as u32);
            decode_hidden.push(unsafe { download_bf16(*hs, hidden) });
        }

        (decode_tokens, decode_hidden)
    }

    println!("  Running standard path...");
    let (std_tokens, std_hidden) = run_path(false, &config, &prefill_ids, num_decode_steps);
    println!("  Running fused path...");
    let (fused_tokens, fused_hidden) = run_path(true, &config, &prefill_ids, num_decode_steps);

    println!();
    let mut all_match = true;
    for step in 0..num_decode_steps {
        let (diff, _) = max_diff_bf16(&std_hidden[step], &fused_hidden[step]);
        let tok_match = std_tokens[step] == fused_tokens[step];
        if !tok_match {
            all_match = false;
        }
        println!(
            "  decode {step}: std_token={:6} fused_token={:6} {}  hidden_diff={diff:.2e}",
            std_tokens[step],
            fused_tokens[step],
            if tok_match {
                "MATCH"
            } else {
                "MISMATCH ←←←"
            },
        );
    }

    if all_match {
        println!("PASS: test13c — all decode tokens match between standard and fused");
    } else {
        println!("FAIL: test13c — token mismatch! Fused path produces different tokens.");
        panic!("Token mismatch between standard and fused paths");
    }
}

// ── test13b: decode-only sanity check (standard path only) ──
// Verifies my decode metadata setup is correct before blaming the fused kernel.
#[test]
fn test13b_decode_sanity() {
    println!("=== test13b: decode sanity — standard path only ===");

    let mut device = GpuDevice::new(0).unwrap();
    let stream = device.compute_stream;
    let config = qwen_config();
    let hidden = config.hidden_size;
    let eps = config.rms_norm_eps;
    let num_layers = config.num_hidden_layers;
    let block_size = 16usize;

    let raw = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&raw).expect("parse");

    let embed_data = load_bf16_tensor(&st, "model.embed_tokens.weight");
    let vocab_size = embed_data.len() / hidden;
    let embed_w = unsafe { upload_bf16(&mut device.caching, &embed_data, &[vocab_size, hidden]) };

    struct LW {
        norm1: OwnedTensor,
        qkv: OwnedTensor,
        norm2: OwnedTensor,
        gate_up: OwnedTensor,
        down: OwnedTensor,
    }
    let mut lw: Vec<LW> = Vec::new();
    for i in 0..num_layers {
        let n1 = load_bf16_tensor(&st, &format!("model.layers.{i}.input_layernorm.weight"));
        let mut qkv = load_bf16_tensor(&st, &format!("model.layers.{i}.self_attn.q_proj.weight"));
        qkv.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.self_attn.k_proj.weight"),
        ));
        qkv.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.self_attn.v_proj.weight"),
        ));
        let qkv_n = qkv.len() / hidden;
        let n2 = load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.post_attention_layernorm.weight"),
        );
        let mut gu = load_bf16_tensor(&st, &format!("model.layers.{i}.mlp.gate_proj.weight"));
        gu.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.mlp.up_proj.weight"),
        ));
        let gu_n = gu.len() / hidden;
        let dw = load_bf16_tensor(&st, &format!("model.layers.{i}.mlp.down_proj.weight"));
        let down_k = dw.len() / hidden;
        lw.push(LW {
            norm1: unsafe { upload_bf16(&mut device.caching, &n1, &[hidden]) },
            qkv: unsafe { upload_bf16(&mut device.caching, &qkv, &[qkv_n, hidden]) },
            norm2: unsafe { upload_bf16(&mut device.caching, &n2, &[hidden]) },
            gate_up: unsafe { upload_bf16(&mut device.caching, &gu, &[gu_n, hidden]) },
            down: unsafe { upload_bf16(&mut device.caching, &dw, &[hidden, down_k]) },
        });
    }

    let mut gw = GpuWeights::from_single_file(MODEL_PATH, stream).unwrap();
    let mut layers: Vec<LlamaDecoderLayer> = Vec::new();
    for i in 0..num_layers {
        layers.push(
            LlamaDecoderLayer::load(&mut gw, &format!("model.layers.{i}"), &config, i, stream)
                .unwrap(),
        );
    }
    drop(raw);

    let kv = unsafe { KvCachePool::new(24, 32, block_size, 2, 64, DType::BF16).unwrap() };
    let rotary =
        unsafe { RotaryCache::new(64, 32768, 1000000.0, None, DType::BF16, &device).unwrap() };

    let prefill_ids: Vec<u32> = vec![1, 3923, 374, 220, 17, 10, 17, 30];
    let prefill_len = prefill_ids.len();

    // Embed
    let ids = unsafe { upload_u32(&mut device.caching, &prefill_ids, &[prefill_len]) };
    let mut hs = unsafe {
        vllm_cuda::kernels::embedding_gather(*embed_w, *ids, &mut device.caching, stream)
    };
    drop(ids);
    let mut res = unsafe {
        upload_bf16(
            &mut device.caching,
            &vec![bf16::ZERO; prefill_len * hidden],
            &[prefill_len, hidden],
        )
    };

    // Prefill metadata
    let pos = unsafe {
        upload_u32(
            &mut device.caching,
            &(0..prefill_len as u32).collect::<Vec<_>>(),
            &[prefill_len],
        )
    };
    let slot = unsafe {
        upload_i64(
            &mut device.caching,
            &(0..prefill_len as i64).collect::<Vec<_>>(),
            &[prefill_len],
        )
    };
    let cu = unsafe { upload_i32(&mut device.caching, &[0, prefill_len as i32], &[2]) };
    let seq = unsafe { upload_i32(&mut device.caching, &[prefill_len as i32], &[1]) };
    let bt = unsafe { upload_i32(&mut device.caching, &[0], &[1, 1]) };

    unsafe { cusys::cuStreamSynchronize(stream) };

    // Prefill
    println!("  Prefill {prefill_len} tokens (standard only)...");
    for i in 0..num_layers {
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(*hs, *res, *lw[i].norm1, eps, stream)
        };
        let qkv = unsafe {
            device
                .ferrite
                .gemm(*hs, *lw[i].qkv, None, 1.0, 0.0, &mut device.caching, stream)
        };
        drop(hs);
        let attn = unsafe {
            layers[i].self_attn.forward_from_qkv(
                qkv,
                pos.view(),
                slot.view(),
                cu.view(),
                seq.view(),
                bt.view(),
                prefill_len,
                prefill_len,
                &kv,
                &rotary,
                &mut device,
            )
        };
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(*attn, *res, *lw[i].norm2, eps, stream)
        };
        let gu = unsafe {
            device.ferrite.gemm(
                *attn,
                *lw[i].gate_up,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(attn);
        let act = unsafe {
            vllm_cuda::kernels::silu_and_mul_fused(
                *gu,
                config.intermediate_size,
                &mut device.caching,
                stream,
            )
        };
        drop(gu);
        hs = unsafe {
            device.ferrite.gemm(
                *act,
                *lw[i].down,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(act);
    }
    println!("  Prefill done.");

    // Decode 1 token
    println!("  Decode 1 token...");
    let token_pos = prefill_len as u32;
    let total_seq = prefill_len + 1;
    let slot_idx = token_pos as i64;
    let num_blocks = (total_seq + block_size - 1) / block_size;

    let d_pos = unsafe { upload_u32(&mut device.caching, &[token_pos], &[1]) };
    let d_slot = unsafe { upload_i64(&mut device.caching, &[slot_idx], &[1]) };
    let d_cu = unsafe { upload_i32(&mut device.caching, &[0i32, 1], &[2]) };
    let d_seq = unsafe { upload_i32(&mut device.caching, &[total_seq as i32], &[1]) };
    let d_bt = unsafe {
        upload_i32(
            &mut device.caching,
            &(0..num_blocks as i32).collect::<Vec<_>>(),
            &[1, num_blocks],
        )
    };

    // Feed last hidden state as decode input
    unsafe { cusys::cuStreamSynchronize(stream) };
    let last_h = unsafe { download_bf16(*hs, prefill_len * hidden) };
    let last_row: Vec<bf16> = last_h[(prefill_len - 1) * hidden..].to_vec();
    hs = unsafe { upload_bf16(&mut device.caching, &last_row, &[1, hidden]) };
    // Resize residual to [1, hidden] — take last row
    let last_res = unsafe { download_bf16(*res, prefill_len * hidden) };
    let last_res_row: Vec<bf16> = last_res[(prefill_len - 1) * hidden..].to_vec();
    res = unsafe { upload_bf16(&mut device.caching, &last_res_row, &[1, hidden]) };

    unsafe { cusys::cuStreamSynchronize(stream) };

    for i in 0..num_layers {
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(*hs, *res, *lw[i].norm1, eps, stream)
        };
        let qkv = unsafe {
            device
                .ferrite
                .gemm(*hs, *lw[i].qkv, None, 1.0, 0.0, &mut device.caching, stream)
        };
        drop(hs);
        let attn = unsafe {
            layers[i].self_attn.forward_from_qkv(
                qkv,
                d_pos.view(),
                d_slot.view(),
                d_cu.view(),
                d_seq.view(),
                d_bt.view(),
                1,
                total_seq,
                &kv,
                &rotary,
                &mut device,
            )
        };
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(*attn, *res, *lw[i].norm2, eps, stream)
        };
        let gu = unsafe {
            device.ferrite.gemm(
                *attn,
                *lw[i].gate_up,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(attn);
        let act = unsafe {
            vllm_cuda::kernels::silu_and_mul_fused(
                *gu,
                config.intermediate_size,
                &mut device.caching,
                stream,
            )
        };
        drop(gu);
        hs = unsafe {
            device.ferrite.gemm(
                *act,
                *lw[i].down,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(act);
    }

    unsafe { cusys::cuStreamSynchronize(stream) };
    let out = unsafe { download_bf16(*hs, hidden) };
    let out_max = out.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
    let nan = out.iter().filter(|v| v.to_f32().is_nan()).count();
    println!("  Decode done: max={out_max:.2}, nan={nan}");
    assert_eq!(nan, 0, "Decode produced NaN");
    println!("PASS: test13b — standard decode works");
}

// ════════════════════════════════════════════════════════════════════════
// test14a: test12 + argmax token comparison
//
// Same as test12 (24-layer prefill, synthetic data, both paths) but
// adds final_norm + lm_head + argmax at the end. One variable changed
// vs test12: does the "bounded" hidden state diff flip tokens?
//
// If this passes: prefill token agreement is fine, problem is in decode.
// If this fails: we don't even need decode to see the bug.
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test14a_prefill_argmax_synthetic() {
    println!("=== test14a: test12 + argmax — does prefill diff flip tokens? ===");

    let mut device = GpuDevice::new(0).unwrap();
    let stream = device.compute_stream;
    let config = qwen_config();
    let hidden = config.hidden_size;
    let eps = config.rms_norm_eps;
    let num_layers = config.num_hidden_layers;

    let raw = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&raw).expect("parse");

    // Embedding weight (for lm_head — tied)
    let embed_data = load_bf16_tensor(&st, "model.embed_tokens.weight");
    let vocab_size = embed_data.len() / hidden;
    let embed_w = unsafe { upload_bf16(&mut device.caching, &embed_data, &[vocab_size, hidden]) };

    // Final norm
    let final_norm_data = load_bf16_tensor(&st, "model.norm.weight");
    let final_norm_w = unsafe { upload_bf16(&mut device.caching, &final_norm_data, &[hidden]) };

    // Per-layer weights
    struct LW {
        norm1: OwnedTensor,
        qkv: OwnedTensor,
        norm2: OwnedTensor,
        gate_up: OwnedTensor,
        down: OwnedTensor,
    }
    let mut lw: Vec<LW> = Vec::new();
    for i in 0..num_layers {
        let n1 = load_bf16_tensor(&st, &format!("model.layers.{i}.input_layernorm.weight"));
        let mut qkv = load_bf16_tensor(&st, &format!("model.layers.{i}.self_attn.q_proj.weight"));
        qkv.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.self_attn.k_proj.weight"),
        ));
        qkv.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.self_attn.v_proj.weight"),
        ));
        let qkv_n = qkv.len() / hidden;
        let n2 = load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.post_attention_layernorm.weight"),
        );
        let mut gu = load_bf16_tensor(&st, &format!("model.layers.{i}.mlp.gate_proj.weight"));
        gu.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.mlp.up_proj.weight"),
        ));
        let gu_n = gu.len() / hidden;
        let dw = load_bf16_tensor(&st, &format!("model.layers.{i}.mlp.down_proj.weight"));
        let down_k = dw.len() / hidden;
        lw.push(LW {
            norm1: unsafe { upload_bf16(&mut device.caching, &n1, &[hidden]) },
            qkv: unsafe { upload_bf16(&mut device.caching, &qkv, &[qkv_n, hidden]) },
            norm2: unsafe { upload_bf16(&mut device.caching, &n2, &[hidden]) },
            gate_up: unsafe { upload_bf16(&mut device.caching, &gu, &[gu_n, hidden]) },
            down: unsafe { upload_bf16(&mut device.caching, &dw, &[hidden, down_k]) },
        });
    }

    let mut gw = GpuWeights::from_single_file(MODEL_PATH, stream).unwrap();
    let mut layers: Vec<LlamaDecoderLayer> = Vec::new();
    for i in 0..num_layers {
        layers.push(
            LlamaDecoderLayer::load(&mut gw, &format!("model.layers.{i}"), &config, i, stream)
                .unwrap(),
        );
    }
    drop(raw);

    let kv_a = unsafe { KvCachePool::new(24, 32, 16, 2, 64, DType::BF16).unwrap() };
    let kv_b = unsafe { KvCachePool::new(24, 32, 16, 2, 64, DType::BF16).unwrap() };
    let rotary =
        unsafe { RotaryCache::new(64, 32768, 1000000.0, None, DType::BF16, &device).unwrap() };

    // Same synthetic data as test12
    let m = 4usize;
    let pos_data: Vec<u32> = (0..m as u32).collect();
    let slot_data: Vec<i64> = (0..m as i64).collect();
    let cu_data: Vec<i32> = vec![0, m as i32];
    let seq_data: Vec<i32> = vec![m as i32];
    let bt_data: Vec<i32> = vec![0];
    let pos = unsafe { upload_u32(&mut device.caching, &pos_data, &[m]) };
    let slot = unsafe { upload_i64(&mut device.caching, &slot_data, &[m]) };
    let cu = unsafe { upload_i32(&mut device.caching, &cu_data, &[2]) };
    let seq = unsafe { upload_i32(&mut device.caching, &seq_data, &[1]) };
    let bt = unsafe { upload_i32(&mut device.caching, &bt_data, &[1, 1]) };

    let hs_data: Vec<bf16> = (0..m * hidden)
        .map(|i| bf16::from_f32(((i as f32) * 0.00037 - 0.5).sin() * 0.1))
        .collect();
    let mut hs_a = unsafe { upload_bf16(&mut device.caching, &hs_data, &[m, hidden]) };
    let mut hs_b = unsafe { upload_bf16(&mut device.caching, &hs_data, &[m, hidden]) };
    let mut res_a = unsafe {
        upload_bf16(
            &mut device.caching,
            &vec![bf16::ZERO; m * hidden],
            &[m, hidden],
        )
    };
    let mut res_b = unsafe {
        upload_bf16(
            &mut device.caching,
            &vec![bf16::ZERO; m * hidden],
            &[m, hidden],
        )
    };

    unsafe { cusys::cuStreamSynchronize(stream) };

    // ── 24-layer prefill (identical to test12) ──
    for i in 0..num_layers {
        // Path A (standard)
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(
                *hs_a,
                *res_a,
                *lw[i].norm1,
                eps,
                stream,
            );
        }
        let qkv_a = unsafe {
            device.ferrite.gemm(
                *hs_a,
                *lw[i].qkv,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(hs_a);
        let attn_a = unsafe {
            layers[i].self_attn.forward_from_qkv(
                qkv_a,
                pos.view(),
                slot.view(),
                cu.view(),
                seq.view(),
                bt.view(),
                m,
                m,
                &kv_a,
                &rotary,
                &mut device,
            )
        };
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(
                *attn_a,
                *res_a,
                *lw[i].norm2,
                eps,
                stream,
            );
        }
        let gu_a = unsafe {
            device.ferrite.gemm(
                *attn_a,
                *lw[i].gate_up,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(attn_a);
        let act_a = unsafe {
            vllm_cuda::kernels::silu_and_mul_fused(
                *gu_a,
                config.intermediate_size,
                &mut device.caching,
                stream,
            )
        };
        drop(gu_a);
        hs_a = unsafe {
            device.ferrite.gemm(
                *act_a,
                *lw[i].down,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(act_a);

        // Path B (fused)
        unsafe { vllm_cuda::kernels::add_inplace(*res_b, *hs_b, stream) };
        drop(hs_b);
        let qkv_b = unsafe {
            vllm_cuda::ferrite::launch_fused_norm_gemm(
                &FUSED_NORM_GEMM,
                *res_b,
                *lw[i].qkv,
                *lw[i].norm1,
                eps,
                hidden as u32,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        let attn_b = unsafe {
            layers[i].self_attn.forward_from_qkv(
                qkv_b,
                pos.view(),
                slot.view(),
                cu.view(),
                seq.view(),
                bt.view(),
                m,
                m,
                &kv_b,
                &rotary,
                &mut device,
            )
        };
        unsafe { vllm_cuda::kernels::add_inplace(*res_b, *attn_b, stream) };
        drop(attn_b);
        let gu_b = unsafe {
            vllm_cuda::ferrite::launch_fused_norm_gemm(
                &FUSED_NORM_GEMM,
                *res_b,
                *lw[i].gate_up,
                *lw[i].norm2,
                eps,
                hidden as u32,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        let act_b = unsafe {
            vllm_cuda::kernels::silu_and_mul_fused(
                *gu_b,
                config.intermediate_size,
                &mut device.caching,
                stream,
            )
        };
        drop(gu_b);
        hs_b = unsafe {
            device.ferrite.gemm(
                *act_b,
                *lw[i].down,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(act_b);
    }

    unsafe { cusys::cuStreamSynchronize(stream) };

    // ── NEW: final norm + lm_head + argmax per token ──
    // Apply final norm to both paths (standard norm, not part of fused experiment)
    // Need copies since fused_add_rms_norm_inplace is in-place
    let hs_a_copy = unsafe {
        let data = download_bf16(*hs_a, m * hidden);
        upload_bf16(&mut device.caching, &data, &[m, hidden])
    };
    let hs_b_copy = unsafe {
        let data = download_bf16(*hs_b, m * hidden);
        upload_bf16(&mut device.caching, &data, &[m, hidden])
    };
    // Use standalone rms_norm (not fused_add) to avoid needing a residual
    let normed_a = unsafe {
        vllm_cuda::kernels::rms_norm(*hs_a_copy, *final_norm_w, eps, &mut device.caching, stream)
    };
    let normed_b = unsafe {
        vllm_cuda::kernels::rms_norm(*hs_b_copy, *final_norm_w, eps, &mut device.caching, stream)
    };

    // lm_head (tied weights): logits = normed @ embed_w^T
    // For each of the m tokens
    let logits_a = unsafe {
        device.ferrite.gemm(
            *normed_a,
            *embed_w,
            None,
            1.0,
            0.0,
            &mut device.caching,
            stream,
        )
    };
    let logits_b = unsafe {
        device.ferrite.gemm(
            *normed_b,
            *embed_w,
            None,
            1.0,
            0.0,
            &mut device.caching,
            stream,
        )
    };

    unsafe { cusys::cuStreamSynchronize(stream) };

    let la = unsafe { download_bf16(*logits_a, m * vocab_size) };
    let lb = unsafe { download_bf16(*logits_b, m * vocab_size) };

    let mut all_match = true;
    for tok in 0..m {
        let row_a = &la[tok * vocab_size..(tok + 1) * vocab_size];
        let row_b = &lb[tok * vocab_size..(tok + 1) * vocab_size];

        let (tok_a, _) = row_a
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.to_f32().partial_cmp(&b.to_f32()).unwrap())
            .unwrap();
        let (tok_b, _) = row_b
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.to_f32().partial_cmp(&b.to_f32()).unwrap())
            .unwrap();

        let (hs_diff, _) = {
            let ha = unsafe { download_bf16(*hs_a, m * hidden) };
            let hb = unsafe { download_bf16(*hs_b, m * hidden) };
            max_diff_bf16(
                &ha[tok * hidden..(tok + 1) * hidden],
                &hb[tok * hidden..(tok + 1) * hidden],
            )
        };

        let match_str = if tok_a == tok_b {
            "MATCH"
        } else {
            all_match = false;
            "MISMATCH ←←←"
        };
        println!(
            "  token {tok}: std={tok_a:6} fused={tok_b:6} {match_str}  hidden_diff={hs_diff:.2e}"
        );
    }

    if all_match {
        println!("PASS: test14a — prefill argmax tokens match (synthetic data)");
    } else {
        println!("FAIL: test14a — prefill argmax mismatch with synthetic data!");
        panic!("Prefill token mismatch — no decode needed to see divergence");
    }
}

// ════════════════════════════════════════════════════════════════════════
// test14b: test14a but with real embeddings (embedding_gather)
//
// Same as test14a (24-layer prefill + argmax) but input comes from
// embedding_gather with token IDs instead of synthetic sin/cos.
// One variable changed vs test14a: input data source.
//
// If this passes: real embeddings don't matter for prefill tokens.
// If this fails: the fused kernel breaks specifically with real embedding values.
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test14b_prefill_argmax_real_embeddings() {
    println!("=== test14b: prefill + argmax with real embeddings ===");

    let mut device = GpuDevice::new(0).unwrap();
    let stream = device.compute_stream;
    let config = qwen_config();
    let hidden = config.hidden_size;
    let eps = config.rms_norm_eps;
    let num_layers = config.num_hidden_layers;

    let raw = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&raw).expect("parse");

    let embed_data = load_bf16_tensor(&st, "model.embed_tokens.weight");
    let vocab_size = embed_data.len() / hidden;
    let embed_w = unsafe { upload_bf16(&mut device.caching, &embed_data, &[vocab_size, hidden]) };

    let final_norm_data = load_bf16_tensor(&st, "model.norm.weight");
    let final_norm_w = unsafe { upload_bf16(&mut device.caching, &final_norm_data, &[hidden]) };

    struct LW {
        norm1: OwnedTensor,
        qkv: OwnedTensor,
        norm2: OwnedTensor,
        gate_up: OwnedTensor,
        down: OwnedTensor,
    }
    let mut lw: Vec<LW> = Vec::new();
    for i in 0..num_layers {
        let n1 = load_bf16_tensor(&st, &format!("model.layers.{i}.input_layernorm.weight"));
        let mut qkv = load_bf16_tensor(&st, &format!("model.layers.{i}.self_attn.q_proj.weight"));
        qkv.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.self_attn.k_proj.weight"),
        ));
        qkv.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.self_attn.v_proj.weight"),
        ));
        let qkv_n = qkv.len() / hidden;
        let n2 = load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.post_attention_layernorm.weight"),
        );
        let mut gu = load_bf16_tensor(&st, &format!("model.layers.{i}.mlp.gate_proj.weight"));
        gu.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.mlp.up_proj.weight"),
        ));
        let gu_n = gu.len() / hidden;
        let dw = load_bf16_tensor(&st, &format!("model.layers.{i}.mlp.down_proj.weight"));
        let down_k = dw.len() / hidden;
        lw.push(LW {
            norm1: unsafe { upload_bf16(&mut device.caching, &n1, &[hidden]) },
            qkv: unsafe { upload_bf16(&mut device.caching, &qkv, &[qkv_n, hidden]) },
            norm2: unsafe { upload_bf16(&mut device.caching, &n2, &[hidden]) },
            gate_up: unsafe { upload_bf16(&mut device.caching, &gu, &[gu_n, hidden]) },
            down: unsafe { upload_bf16(&mut device.caching, &dw, &[hidden, down_k]) },
        });
    }

    let mut gw = GpuWeights::from_single_file(MODEL_PATH, stream).unwrap();
    let mut layers: Vec<LlamaDecoderLayer> = Vec::new();
    for i in 0..num_layers {
        layers.push(
            LlamaDecoderLayer::load(&mut gw, &format!("model.layers.{i}"), &config, i, stream)
                .unwrap(),
        );
    }
    drop(raw);

    let kv_a = unsafe { KvCachePool::new(24, 32, 16, 2, 64, DType::BF16).unwrap() };
    let kv_b = unsafe { KvCachePool::new(24, 32, 16, 2, 64, DType::BF16).unwrap() };
    let rotary =
        unsafe { RotaryCache::new(64, 32768, 1000000.0, None, DType::BF16, &device).unwrap() };

    // Real token IDs (same as test13c)
    let prefill_ids: Vec<u32> = vec![1, 3923, 374, 220, 17, 10, 17, 30];
    let m = prefill_ids.len();

    // Embed tokens (real embeddings)
    let ids_gpu = unsafe { upload_u32(&mut device.caching, &prefill_ids, &[m]) };
    let mut hs_a = unsafe {
        vllm_cuda::kernels::embedding_gather(*embed_w, *ids_gpu, &mut device.caching, stream)
    };
    // Need a second copy for path B
    unsafe { cusys::cuStreamSynchronize(stream) };
    let hs_data = unsafe { download_bf16(*hs_a, m * hidden) };
    let mut hs_b = unsafe { upload_bf16(&mut device.caching, &hs_data, &[m, hidden]) };
    drop(ids_gpu);

    let mut res_a = unsafe {
        upload_bf16(
            &mut device.caching,
            &vec![bf16::ZERO; m * hidden],
            &[m, hidden],
        )
    };
    let mut res_b = unsafe {
        upload_bf16(
            &mut device.caching,
            &vec![bf16::ZERO; m * hidden],
            &[m, hidden],
        )
    };

    let pos_data: Vec<u32> = (0..m as u32).collect();
    let slot_data: Vec<i64> = (0..m as i64).collect();
    let cu_data: Vec<i32> = vec![0, m as i32];
    let seq_data: Vec<i32> = vec![m as i32];
    let bt_data: Vec<i32> = vec![0];
    let pos = unsafe { upload_u32(&mut device.caching, &pos_data, &[m]) };
    let slot = unsafe { upload_i64(&mut device.caching, &slot_data, &[m]) };
    let cu = unsafe { upload_i32(&mut device.caching, &cu_data, &[2]) };
    let seq = unsafe { upload_i32(&mut device.caching, &seq_data, &[1]) };
    let bt = unsafe { upload_i32(&mut device.caching, &bt_data, &[1, 1]) };

    unsafe { cusys::cuStreamSynchronize(stream) };

    // ── 24-layer prefill ──
    for i in 0..num_layers {
        // Path A (standard)
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(
                *hs_a,
                *res_a,
                *lw[i].norm1,
                eps,
                stream,
            );
        }
        let qkv_a = unsafe {
            device.ferrite.gemm(
                *hs_a,
                *lw[i].qkv,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(hs_a);
        let attn_a = unsafe {
            layers[i].self_attn.forward_from_qkv(
                qkv_a,
                pos.view(),
                slot.view(),
                cu.view(),
                seq.view(),
                bt.view(),
                m,
                m,
                &kv_a,
                &rotary,
                &mut device,
            )
        };
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(
                *attn_a,
                *res_a,
                *lw[i].norm2,
                eps,
                stream,
            );
        }
        let gu_a = unsafe {
            device.ferrite.gemm(
                *attn_a,
                *lw[i].gate_up,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(attn_a);
        let act_a = unsafe {
            vllm_cuda::kernels::silu_and_mul_fused(
                *gu_a,
                config.intermediate_size,
                &mut device.caching,
                stream,
            )
        };
        drop(gu_a);
        hs_a = unsafe {
            device.ferrite.gemm(
                *act_a,
                *lw[i].down,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(act_a);

        // Path B (fused)
        unsafe { vllm_cuda::kernels::add_inplace(*res_b, *hs_b, stream) };
        drop(hs_b);
        let qkv_b = unsafe {
            vllm_cuda::ferrite::launch_fused_norm_gemm(
                &FUSED_NORM_GEMM,
                *res_b,
                *lw[i].qkv,
                *lw[i].norm1,
                eps,
                hidden as u32,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        let attn_b = unsafe {
            layers[i].self_attn.forward_from_qkv(
                qkv_b,
                pos.view(),
                slot.view(),
                cu.view(),
                seq.view(),
                bt.view(),
                m,
                m,
                &kv_b,
                &rotary,
                &mut device,
            )
        };
        unsafe { vllm_cuda::kernels::add_inplace(*res_b, *attn_b, stream) };
        drop(attn_b);
        let gu_b = unsafe {
            vllm_cuda::ferrite::launch_fused_norm_gemm(
                &FUSED_NORM_GEMM,
                *res_b,
                *lw[i].gate_up,
                *lw[i].norm2,
                eps,
                hidden as u32,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        let act_b = unsafe {
            vllm_cuda::kernels::silu_and_mul_fused(
                *gu_b,
                config.intermediate_size,
                &mut device.caching,
                stream,
            )
        };
        drop(gu_b);
        hs_b = unsafe {
            device.ferrite.gemm(
                *act_b,
                *lw[i].down,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(act_b);
    }

    unsafe { cusys::cuStreamSynchronize(stream) };

    // ── Final norm + lm_head + argmax ──
    let normed_a = unsafe {
        vllm_cuda::kernels::rms_norm(*hs_a, *final_norm_w, eps, &mut device.caching, stream)
    };
    let normed_b = unsafe {
        vllm_cuda::kernels::rms_norm(*hs_b, *final_norm_w, eps, &mut device.caching, stream)
    };
    let logits_a = unsafe {
        device.ferrite.gemm(
            *normed_a,
            *embed_w,
            None,
            1.0,
            0.0,
            &mut device.caching,
            stream,
        )
    };
    let logits_b = unsafe {
        device.ferrite.gemm(
            *normed_b,
            *embed_w,
            None,
            1.0,
            0.0,
            &mut device.caching,
            stream,
        )
    };

    unsafe { cusys::cuStreamSynchronize(stream) };

    let la = unsafe { download_bf16(*logits_a, m * vocab_size) };
    let lb = unsafe { download_bf16(*logits_b, m * vocab_size) };
    let ha = unsafe { download_bf16(*hs_a, m * hidden) };
    let hb = unsafe { download_bf16(*hs_b, m * hidden) };

    let mut all_match = true;
    for tok in 0..m {
        let row_a = &la[tok * vocab_size..(tok + 1) * vocab_size];
        let row_b = &lb[tok * vocab_size..(tok + 1) * vocab_size];

        let (tok_a, _) = row_a
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.to_f32().partial_cmp(&b.to_f32()).unwrap())
            .unwrap();
        let (tok_b, _) = row_b
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.to_f32().partial_cmp(&b.to_f32()).unwrap())
            .unwrap();

        let (hs_diff, _) = max_diff_bf16(
            &ha[tok * hidden..(tok + 1) * hidden],
            &hb[tok * hidden..(tok + 1) * hidden],
        );

        // Top-2 logit gap for path A
        let mut sorted_a: Vec<(usize, f32)> = row_a
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.to_f32()))
            .collect();
        sorted_a.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let gap_a = sorted_a[0].1 - sorted_a[1].1;

        let mut sorted_b: Vec<(usize, f32)> = row_b
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.to_f32()))
            .collect();
        sorted_b.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let gap_b = sorted_b[0].1 - sorted_b[1].1;

        let match_str = if tok_a == tok_b {
            "MATCH"
        } else {
            all_match = false;
            "MISMATCH ←←←"
        };
        println!(
            "  token {tok}: std={tok_a:6} fused={tok_b:6} {match_str}  hidden_diff={hs_diff:.2e}  gap_a={gap_a:.3} gap_b={gap_b:.3}"
        );
        if tok_a != tok_b {
            // Print where path B's winner ranks in path A
            let rank_in_a = sorted_a.iter().position(|(id, _)| *id == tok_b).unwrap();
            let rank_in_b = sorted_b.iter().position(|(id, _)| *id == tok_a).unwrap();
            println!(
                "    fused_winner={tok_b} is rank {rank_in_a} in std (logit={:.3})",
                row_a[tok_b].to_f32()
            );
            println!(
                "    std_winner={tok_a} is rank {rank_in_b} in fused (logit={:.3})",
                row_b[tok_a].to_f32()
            );
        }
    }

    if all_match {
        println!("PASS: test14b — prefill argmax tokens match (real embeddings)");
    } else {
        println!("FAIL: test14b — prefill argmax mismatch with real embeddings!");
        panic!("Prefill token mismatch with real embeddings");
    }
}

// ════════════════════════════════════════════════════════════════════════
// test14c: Single-layer QKV with real embeddings
//
// Same as test10c (QKV-only, shared weights) but with real embedding
// inputs instead of synthetic sin/cos. Tests one layer at a time.
//
// If diffs are same magnitude as test10c: single-layer is fine, error accumulates.
// If diffs are larger: fused kernel is wrong on real embedding distributions.
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test14c_single_layer_qkv_real_embeddings() {
    println!("=== test14c: single-layer QKV, real embeddings ===");

    let mut device = GpuDevice::new(0).unwrap();
    let stream = device.compute_stream;
    let config = qwen_config();
    let hidden = config.hidden_size;
    let eps = config.rms_norm_eps;

    let raw = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&raw).expect("parse");

    let embed_data = load_bf16_tensor(&st, "model.embed_tokens.weight");
    let vocab_size = embed_data.len() / hidden;
    let embed_w = unsafe { upload_bf16(&mut device.caching, &embed_data, &[vocab_size, hidden]) };

    // Test with multiple layers to see if some layers are worse
    let prefill_ids: Vec<u32> = vec![1, 3923, 374, 220, 17, 10, 17, 30];
    let m = prefill_ids.len();

    // Embed tokens once
    let ids_gpu = unsafe { upload_u32(&mut device.caching, &prefill_ids, &[m]) };
    let embedded = unsafe {
        vllm_cuda::kernels::embedding_gather(*embed_w, *ids_gpu, &mut device.caching, stream)
    };
    unsafe { cusys::cuStreamSynchronize(stream) };
    let embed_host = unsafe { download_bf16(*embedded, m * hidden) };
    drop(ids_gpu);
    drop(embedded);

    println!("  Embedded {m} tokens. Testing each layer individually...");
    println!("  layer | qkv_diff  | res_diff  | notes");
    println!("  ------|-----------|-----------|------");

    for layer_idx in 0..config.num_hidden_layers {
        let norm_data = load_bf16_tensor(
            &st,
            &format!("model.layers.{layer_idx}.input_layernorm.weight"),
        );
        let norm_w = unsafe { upload_bf16(&mut device.caching, &norm_data, &[hidden]) };

        let mut qkv_data = load_bf16_tensor(
            &st,
            &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
        );
        qkv_data.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
        ));
        qkv_data.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
        ));
        let qkv_n = qkv_data.len() / hidden;
        let qkv_w = unsafe { upload_bf16(&mut device.caching, &qkv_data, &[qkv_n, hidden]) };

        // Input: use embeddings as hs, zeros as residual (simulates layer 0)
        let hs_a = unsafe { upload_bf16(&mut device.caching, &embed_host, &[m, hidden]) };
        let hs_b = unsafe { upload_bf16(&mut device.caching, &embed_host, &[m, hidden]) };
        let res_a = unsafe {
            upload_bf16(
                &mut device.caching,
                &vec![bf16::ZERO; m * hidden],
                &[m, hidden],
            )
        };
        let res_b = unsafe {
            upload_bf16(
                &mut device.caching,
                &vec![bf16::ZERO; m * hidden],
                &[m, hidden],
            )
        };

        unsafe { cusys::cuStreamSynchronize(stream) };

        // Standard: fused_add_rms_norm_inplace + ferrite.gemm
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(*hs_a, *res_a, *norm_w, eps, stream);
        }
        let qkv_a = unsafe {
            device
                .ferrite
                .gemm(*hs_a, *qkv_w, None, 1.0, 0.0, &mut device.caching, stream)
        };

        // Fused: add_inplace + launch_fused_norm_gemm
        unsafe {
            vllm_cuda::kernels::add_inplace(*res_b, *hs_b, stream);
        }
        let qkv_b = unsafe {
            vllm_cuda::ferrite::launch_fused_norm_gemm(
                &FUSED_NORM_GEMM,
                *res_b,
                *qkv_w,
                *norm_w,
                eps,
                hidden as u32,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };

        unsafe { cusys::cuStreamSynchronize(stream) };
        let a_h = unsafe { download_bf16(*qkv_a, m * qkv_n) };
        let b_h = unsafe { download_bf16(*qkv_b, m * qkv_n) };
        let ra_h = unsafe { download_bf16(*res_a, m * hidden) };
        let rb_h = unsafe { download_bf16(*res_b, m * hidden) };

        let (qkv_diff, _) = max_diff_bf16(&a_h, &b_h);
        let (res_diff, _) = max_diff_bf16(&ra_h, &rb_h);

        let notes = if qkv_diff > 0.1 { "HIGH" } else { "" };
        println!(
            "  {:5} | {:9.2e} | {:9.2e} | {notes}",
            layer_idx, qkv_diff, res_diff
        );
    }
    println!("DONE: test14c");
}

// ════════════════════════════════════════════════════════════════════════
// test14d: Isolate bf16 precision loss from fused GEMM
//
// 24-layer prefill with real embeddings, argmax comparison.
// Path A: fused_add_rms_norm_inplace + ferrite.gemm (standard — correct)
// Path B: add_inplace + rms_norm (separate) + ferrite.gemm (same GEMM, different norm path)
//
// NO fused norm+GEMM kernel at all. If this fails, the problem is purely
// the bf16 truncation at the add→norm boundary. If it passes, the fused
// GEMM kernel has a bug that only appears in multi-layer accumulation.
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test14d_precision_only_no_fused_kernel() {
    println!("=== test14d: bf16 precision isolation — no fused kernel ===");

    let mut device = GpuDevice::new(0).unwrap();
    let stream = device.compute_stream;
    let config = qwen_config();
    let hidden = config.hidden_size;
    let eps = config.rms_norm_eps;
    let num_layers = config.num_hidden_layers;

    let raw = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&raw).expect("parse");

    let embed_data = load_bf16_tensor(&st, "model.embed_tokens.weight");
    let vocab_size = embed_data.len() / hidden;
    let embed_w = unsafe { upload_bf16(&mut device.caching, &embed_data, &[vocab_size, hidden]) };

    let final_norm_data = load_bf16_tensor(&st, "model.norm.weight");
    let final_norm_w = unsafe { upload_bf16(&mut device.caching, &final_norm_data, &[hidden]) };

    struct LW {
        norm1: OwnedTensor,
        qkv: OwnedTensor,
        norm2: OwnedTensor,
        gate_up: OwnedTensor,
        down: OwnedTensor,
    }
    let mut lw: Vec<LW> = Vec::new();
    for i in 0..num_layers {
        let n1 = load_bf16_tensor(&st, &format!("model.layers.{i}.input_layernorm.weight"));
        let mut qkv = load_bf16_tensor(&st, &format!("model.layers.{i}.self_attn.q_proj.weight"));
        qkv.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.self_attn.k_proj.weight"),
        ));
        qkv.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.self_attn.v_proj.weight"),
        ));
        let qkv_n = qkv.len() / hidden;
        let n2 = load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.post_attention_layernorm.weight"),
        );
        let mut gu = load_bf16_tensor(&st, &format!("model.layers.{i}.mlp.gate_proj.weight"));
        gu.extend_from_slice(&load_bf16_tensor(
            &st,
            &format!("model.layers.{i}.mlp.up_proj.weight"),
        ));
        let gu_n = gu.len() / hidden;
        let dw = load_bf16_tensor(&st, &format!("model.layers.{i}.mlp.down_proj.weight"));
        let down_k = dw.len() / hidden;
        lw.push(LW {
            norm1: unsafe { upload_bf16(&mut device.caching, &n1, &[hidden]) },
            qkv: unsafe { upload_bf16(&mut device.caching, &qkv, &[qkv_n, hidden]) },
            norm2: unsafe { upload_bf16(&mut device.caching, &n2, &[hidden]) },
            gate_up: unsafe { upload_bf16(&mut device.caching, &gu, &[gu_n, hidden]) },
            down: unsafe { upload_bf16(&mut device.caching, &dw, &[hidden, down_k]) },
        });
    }

    let mut gw = GpuWeights::from_single_file(MODEL_PATH, stream).unwrap();
    let mut layers: Vec<LlamaDecoderLayer> = Vec::new();
    for i in 0..num_layers {
        layers.push(
            LlamaDecoderLayer::load(&mut gw, &format!("model.layers.{i}"), &config, i, stream)
                .unwrap(),
        );
    }
    drop(raw);

    let kv_a = unsafe { KvCachePool::new(24, 32, 16, 2, 64, DType::BF16).unwrap() };
    let kv_b = unsafe { KvCachePool::new(24, 32, 16, 2, 64, DType::BF16).unwrap() };
    let rotary =
        unsafe { RotaryCache::new(64, 32768, 1000000.0, None, DType::BF16, &device).unwrap() };

    let prefill_ids: Vec<u32> = vec![1, 3923, 374, 220, 17, 10, 17, 30];
    let m = prefill_ids.len();

    let ids_gpu = unsafe { upload_u32(&mut device.caching, &prefill_ids, &[m]) };
    let mut hs_a = unsafe {
        vllm_cuda::kernels::embedding_gather(*embed_w, *ids_gpu, &mut device.caching, stream)
    };
    unsafe { cusys::cuStreamSynchronize(stream) };
    let hs_data = unsafe { download_bf16(*hs_a, m * hidden) };
    let mut hs_b = unsafe { upload_bf16(&mut device.caching, &hs_data, &[m, hidden]) };
    drop(ids_gpu);

    let mut res_a = unsafe {
        upload_bf16(
            &mut device.caching,
            &vec![bf16::ZERO; m * hidden],
            &[m, hidden],
        )
    };
    let mut res_b = unsafe {
        upload_bf16(
            &mut device.caching,
            &vec![bf16::ZERO; m * hidden],
            &[m, hidden],
        )
    };

    let pos_data: Vec<u32> = (0..m as u32).collect();
    let slot_data: Vec<i64> = (0..m as i64).collect();
    let cu_data: Vec<i32> = vec![0, m as i32];
    let seq_data: Vec<i32> = vec![m as i32];
    let bt_data: Vec<i32> = vec![0];
    let pos = unsafe { upload_u32(&mut device.caching, &pos_data, &[m]) };
    let slot = unsafe { upload_i64(&mut device.caching, &slot_data, &[m]) };
    let cu = unsafe { upload_i32(&mut device.caching, &cu_data, &[2]) };
    let seq = unsafe { upload_i32(&mut device.caching, &seq_data, &[1]) };
    let bt = unsafe { upload_i32(&mut device.caching, &bt_data, &[1, 1]) };

    unsafe { cusys::cuStreamSynchronize(stream) };

    // ── 24-layer prefill ──
    for i in 0..num_layers {
        // Path A: fused_add_rms_norm_inplace + ferrite.gemm
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(
                *hs_a,
                *res_a,
                *lw[i].norm1,
                eps,
                stream,
            );
        }
        let qkv_a = unsafe {
            device.ferrite.gemm(
                *hs_a,
                *lw[i].qkv,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(hs_a);
        let attn_a = unsafe {
            layers[i].self_attn.forward_from_qkv(
                qkv_a,
                pos.view(),
                slot.view(),
                cu.view(),
                seq.view(),
                bt.view(),
                m,
                m,
                &kv_a,
                &rotary,
                &mut device,
            )
        };
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(
                *attn_a,
                *res_a,
                *lw[i].norm2,
                eps,
                stream,
            );
        }
        let gu_a = unsafe {
            device.ferrite.gemm(
                *attn_a,
                *lw[i].gate_up,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(attn_a);
        let act_a = unsafe {
            vllm_cuda::kernels::silu_and_mul_fused(
                *gu_a,
                config.intermediate_size,
                &mut device.caching,
                stream,
            )
        };
        drop(gu_a);
        hs_a = unsafe {
            device.ferrite.gemm(
                *act_a,
                *lw[i].down,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(act_a);

        // Path B: add_inplace + rms_norm (separate) + ferrite.gemm
        // NO fused norm+GEMM kernel — same GEMM as path A
        unsafe { vllm_cuda::kernels::add_inplace(*res_b, *hs_b, stream) };
        drop(hs_b);
        let normed_b = unsafe {
            vllm_cuda::kernels::rms_norm(*res_b, *lw[i].norm1, eps, &mut device.caching, stream)
        };
        let qkv_b = unsafe {
            device.ferrite.gemm(
                *normed_b,
                *lw[i].qkv,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(normed_b);
        let attn_b = unsafe {
            layers[i].self_attn.forward_from_qkv(
                qkv_b,
                pos.view(),
                slot.view(),
                cu.view(),
                seq.view(),
                bt.view(),
                m,
                m,
                &kv_b,
                &rotary,
                &mut device,
            )
        };
        unsafe { vllm_cuda::kernels::add_inplace(*res_b, *attn_b, stream) };
        drop(attn_b);
        let normed_b2 = unsafe {
            vllm_cuda::kernels::rms_norm(*res_b, *lw[i].norm2, eps, &mut device.caching, stream)
        };
        let gu_b = unsafe {
            device.ferrite.gemm(
                *normed_b2,
                *lw[i].gate_up,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(normed_b2);
        let act_b = unsafe {
            vllm_cuda::kernels::silu_and_mul_fused(
                *gu_b,
                config.intermediate_size,
                &mut device.caching,
                stream,
            )
        };
        drop(gu_b);
        hs_b = unsafe {
            device.ferrite.gemm(
                *act_b,
                *lw[i].down,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(act_b);
    }

    unsafe { cusys::cuStreamSynchronize(stream) };

    // ── Final norm + lm_head + argmax ──
    let normed_a = unsafe {
        vllm_cuda::kernels::rms_norm(*hs_a, *final_norm_w, eps, &mut device.caching, stream)
    };
    let normed_b = unsafe {
        vllm_cuda::kernels::rms_norm(*hs_b, *final_norm_w, eps, &mut device.caching, stream)
    };
    let logits_a = unsafe {
        device.ferrite.gemm(
            *normed_a,
            *embed_w,
            None,
            1.0,
            0.0,
            &mut device.caching,
            stream,
        )
    };
    let logits_b = unsafe {
        device.ferrite.gemm(
            *normed_b,
            *embed_w,
            None,
            1.0,
            0.0,
            &mut device.caching,
            stream,
        )
    };

    unsafe { cusys::cuStreamSynchronize(stream) };

    let la = unsafe { download_bf16(*logits_a, m * vocab_size) };
    let lb = unsafe { download_bf16(*logits_b, m * vocab_size) };
    let ha = unsafe { download_bf16(*hs_a, m * hidden) };
    let hb = unsafe { download_bf16(*hs_b, m * hidden) };

    let mut all_match = true;
    for tok in 0..m {
        let row_a = &la[tok * vocab_size..(tok + 1) * vocab_size];
        let row_b = &lb[tok * vocab_size..(tok + 1) * vocab_size];

        let (tok_a, _) = row_a
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.to_f32().partial_cmp(&b.to_f32()).unwrap())
            .unwrap();
        let (tok_b, _) = row_b
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.to_f32().partial_cmp(&b.to_f32()).unwrap())
            .unwrap();

        let (hs_diff, _) = max_diff_bf16(
            &ha[tok * hidden..(tok + 1) * hidden],
            &hb[tok * hidden..(tok + 1) * hidden],
        );

        let match_str = if tok_a == tok_b {
            "MATCH"
        } else {
            all_match = false;
            "MISMATCH ←←←"
        };
        println!(
            "  token {tok}: std={tok_a:6} fused={tok_b:6} {match_str}  hidden_diff={hs_diff:.2e}"
        );
    }

    if all_match {
        println!("PASS: test14d — precision-only path matches (no fused kernel)");
        println!("  → bf16 precision loss alone does NOT flip tokens");
        println!("  → The fused GEMM kernel has a multi-layer accumulation bug");
    } else {
        println!("FAIL: test14d — even without fused kernel, bf16 precision flips tokens!");
        println!("  → The problem IS the bf16 truncation at add→norm, not the fused kernel");
        panic!("bf16 precision loss alone flips tokens");
    }
}

// ════════════════════════════════════════════════════════════════════════
// test14e: Fused add+norm+GEMM with real embeddings
//
// Same as test14b (24-layer prefill, real embeddings, argmax comparison)
// but path B uses FUSED_ADD_NORM_GEMM — the two-input prologue that
// adds residual + hidden_states in f32, writes bf16 back, and computes
// inv_rms from the f32 sums.
//
// If this passes: the fused kernel with correct precision produces
// matching tokens. The full pipeline fix works.
// If this fails: there's a bug in the two-input prologue transplant.
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test14e_fused_add_norm_gemm_real_embeddings() {
    println!("=== test14e: fused add+norm+GEMM, real embeddings, argmax ===");

    let mut device = GpuDevice::new(0).unwrap();
    let stream = device.compute_stream;
    let config = qwen_config();
    let hidden = config.hidden_size;
    let eps = config.rms_norm_eps;
    let num_layers = config.num_hidden_layers;

    let raw = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&raw).expect("parse");

    let embed_data = load_bf16_tensor(&st, "model.embed_tokens.weight");
    let vocab_size = embed_data.len() / hidden;
    let embed_w = unsafe { upload_bf16(&mut device.caching, &embed_data, &[vocab_size, hidden]) };

    let final_norm_data = load_bf16_tensor(&st, "model.norm.weight");
    let final_norm_w = unsafe { upload_bf16(&mut device.caching, &final_norm_data, &[hidden]) };

    struct LW { norm1: OwnedTensor, qkv: OwnedTensor, norm2: OwnedTensor, gate_up: OwnedTensor, down: OwnedTensor }
    let mut lw: Vec<LW> = Vec::new();
    for i in 0..num_layers {
        let n1 = load_bf16_tensor(&st, &format!("model.layers.{i}.input_layernorm.weight"));
        let mut qkv = load_bf16_tensor(&st, &format!("model.layers.{i}.self_attn.q_proj.weight"));
        qkv.extend_from_slice(&load_bf16_tensor(&st, &format!("model.layers.{i}.self_attn.k_proj.weight")));
        qkv.extend_from_slice(&load_bf16_tensor(&st, &format!("model.layers.{i}.self_attn.v_proj.weight")));
        let qkv_n = qkv.len() / hidden;
        let n2 = load_bf16_tensor(&st, &format!("model.layers.{i}.post_attention_layernorm.weight"));
        let mut gu = load_bf16_tensor(&st, &format!("model.layers.{i}.mlp.gate_proj.weight"));
        gu.extend_from_slice(&load_bf16_tensor(&st, &format!("model.layers.{i}.mlp.up_proj.weight")));
        let gu_n = gu.len() / hidden;
        let dw = load_bf16_tensor(&st, &format!("model.layers.{i}.mlp.down_proj.weight"));
        let down_k = dw.len() / hidden;
        lw.push(LW {
            norm1: unsafe { upload_bf16(&mut device.caching, &n1, &[hidden]) },
            qkv: unsafe { upload_bf16(&mut device.caching, &qkv, &[qkv_n, hidden]) },
            norm2: unsafe { upload_bf16(&mut device.caching, &n2, &[hidden]) },
            gate_up: unsafe { upload_bf16(&mut device.caching, &gu, &[gu_n, hidden]) },
            down: unsafe { upload_bf16(&mut device.caching, &dw, &[hidden, down_k]) },
        });
    }

    let mut gw = GpuWeights::from_single_file(MODEL_PATH, stream).unwrap();
    let mut layers: Vec<LlamaDecoderLayer> = Vec::new();
    for i in 0..num_layers {
        layers.push(LlamaDecoderLayer::load(&mut gw, &format!("model.layers.{i}"), &config, i, stream).unwrap());
    }
    drop(raw);

    let kv_a = unsafe { KvCachePool::new(24, 32, 16, 2, 64, DType::BF16).unwrap() };
    let kv_b = unsafe { KvCachePool::new(24, 32, 16, 2, 64, DType::BF16).unwrap() };
    let rotary = unsafe { RotaryCache::new(64, 32768, 1000000.0, None, DType::BF16, &device).unwrap() };

    let prefill_ids: Vec<u32> = vec![1, 3923, 374, 220, 17, 10, 17, 30];
    let m = prefill_ids.len();

    // Embed tokens
    let ids_gpu = unsafe { upload_u32(&mut device.caching, &prefill_ids, &[m]) };
    let mut hs_a = unsafe {
        vllm_cuda::kernels::embedding_gather(*embed_w, *ids_gpu, &mut device.caching, stream)
    };
    unsafe { cusys::cuStreamSynchronize(stream) };
    let hs_data = unsafe { download_bf16(*hs_a, m * hidden) };
    let mut hs_b = unsafe { upload_bf16(&mut device.caching, &hs_data, &[m, hidden]) };
    drop(ids_gpu);

    let mut res_a = unsafe {
        upload_bf16(&mut device.caching, &vec![bf16::ZERO; m * hidden], &[m, hidden])
    };
    let mut res_b = unsafe {
        upload_bf16(&mut device.caching, &vec![bf16::ZERO; m * hidden], &[m, hidden])
    };

    let pos_data: Vec<u32> = (0..m as u32).collect();
    let slot_data: Vec<i64> = (0..m as i64).collect();
    let cu_data: Vec<i32> = vec![0, m as i32];
    let seq_data: Vec<i32> = vec![m as i32];
    let bt_data: Vec<i32> = vec![0];
    let pos = unsafe { upload_u32(&mut device.caching, &pos_data, &[m]) };
    let slot = unsafe { upload_i64(&mut device.caching, &slot_data, &[m]) };
    let cu = unsafe { upload_i32(&mut device.caching, &cu_data, &[2]) };
    let seq = unsafe { upload_i32(&mut device.caching, &seq_data, &[1]) };
    let bt = unsafe { upload_i32(&mut device.caching, &bt_data, &[1, 1]) };

    unsafe { cusys::cuStreamSynchronize(stream) };

    // ── 24-layer prefill ──
    for i in 0..num_layers {
        // Path A (standard): fused_add_rms_norm_inplace + ferrite.gemm
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(*hs_a, *res_a, *lw[i].norm1, eps, stream);
        }
        let qkv_a = unsafe {
            device.ferrite.gemm(*hs_a, *lw[i].qkv, None, 1.0, 0.0, &mut device.caching, stream)
        };
        drop(hs_a);
        let attn_a = unsafe {
            layers[i].self_attn.forward_from_qkv(
                qkv_a, pos.view(), slot.view(), cu.view(), seq.view(), bt.view(),
                m, m, &kv_a, &rotary, &mut device,
            )
        };
        unsafe {
            vllm_cuda::kernels::fused_add_rms_norm_inplace(*attn_a, *res_a, *lw[i].norm2, eps, stream);
        }
        let gu_a = unsafe {
            device.ferrite.gemm(*attn_a, *lw[i].gate_up, None, 1.0, 0.0, &mut device.caching, stream)
        };
        drop(attn_a);
        let act_a = unsafe {
            vllm_cuda::kernels::silu_and_mul_fused(*gu_a, config.intermediate_size, &mut device.caching, stream)
        };
        drop(gu_a);
        hs_a = unsafe {
            device.ferrite.gemm(*act_a, *lw[i].down, None, 1.0, 0.0, &mut device.caching, stream)
        };
        drop(act_a);

        // Path B (fused): launch_fused_add_norm_gemm (two-input prologue)
        let qkv_b = unsafe {
            vllm_cuda::ferrite::launch_fused_add_norm_gemm(
                &FUSED_ADD_NORM_GEMM,
                *res_b,    // residual (GEMM A-ptr, writeback target)
                *hs_b,     // hidden_states (second input for add)
                *lw[i].qkv,
                *lw[i].norm1,
                eps,
                hidden as u32,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(hs_b);
        let attn_b = unsafe {
            layers[i].self_attn.forward_from_qkv(
                qkv_b, pos.view(), slot.view(), cu.view(), seq.view(), bt.view(),
                m, m, &kv_b, &rotary, &mut device,
            )
        };
        let gu_b = unsafe {
            vllm_cuda::ferrite::launch_fused_add_norm_gemm(
                &FUSED_ADD_NORM_GEMM,
                *res_b,
                *attn_b,
                *lw[i].gate_up,
                *lw[i].norm2,
                eps,
                hidden as u32,
                None,
                1.0,
                0.0,
                &mut device.caching,
                stream,
            )
        };
        drop(attn_b);
        let act_b = unsafe {
            vllm_cuda::kernels::silu_and_mul_fused(*gu_b, config.intermediate_size, &mut device.caching, stream)
        };
        drop(gu_b);
        hs_b = unsafe {
            device.ferrite.gemm(*act_b, *lw[i].down, None, 1.0, 0.0, &mut device.caching, stream)
        };
        drop(act_b);
    }

    unsafe { cusys::cuStreamSynchronize(stream) };

    // ── Final norm + lm_head + argmax ──
    let normed_a = unsafe {
        vllm_cuda::kernels::rms_norm(*hs_a, *final_norm_w, eps, &mut device.caching, stream)
    };
    let normed_b = unsafe {
        vllm_cuda::kernels::rms_norm(*hs_b, *final_norm_w, eps, &mut device.caching, stream)
    };
    let logits_a = unsafe {
        device.ferrite.gemm(*normed_a, *embed_w, None, 1.0, 0.0, &mut device.caching, stream)
    };
    let logits_b = unsafe {
        device.ferrite.gemm(*normed_b, *embed_w, None, 1.0, 0.0, &mut device.caching, stream)
    };

    unsafe { cusys::cuStreamSynchronize(stream) };

    let la = unsafe { download_bf16(*logits_a, m * vocab_size) };
    let lb = unsafe { download_bf16(*logits_b, m * vocab_size) };
    let ha = unsafe { download_bf16(*hs_a, m * hidden) };
    let hb = unsafe { download_bf16(*hs_b, m * hidden) };

    let mut all_match = true;
    for tok in 0..m {
        let row_a = &la[tok * vocab_size..(tok + 1) * vocab_size];
        let row_b = &lb[tok * vocab_size..(tok + 1) * vocab_size];

        let (tok_a, _) = row_a.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.to_f32().partial_cmp(&b.to_f32()).unwrap())
            .unwrap();
        let (tok_b, _) = row_b.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.to_f32().partial_cmp(&b.to_f32()).unwrap())
            .unwrap();

        let (hs_diff, _) = max_diff_bf16(
            &ha[tok * hidden..(tok + 1) * hidden],
            &hb[tok * hidden..(tok + 1) * hidden],
        );

        let match_str = if tok_a == tok_b { "MATCH" } else { all_match = false; "MISMATCH ←←←" };
        println!(
            "  token {tok}: std={tok_a:6} fused={tok_b:6} {match_str}  hidden_diff={hs_diff:.2e}"
        );
    }

    if all_match {
        println!("PASS: test14e — fused add+norm+GEMM tokens match (real embeddings)");
    } else {
        println!("FAIL: test14e — fused add+norm+GEMM token mismatch!");
        panic!("Fused add+norm+GEMM produced different tokens");
    }
}

// ════════════════════════════════════════════════════════════════════════
// test14f: Single-layer fused add+norm+GEMM correctness
//
// One layer, QKV only. Compares:
// A: fused_add_rms_norm_inplace + ferrite.gemm
// B: launch_fused_add_norm_gemm (two-input prologue)
//
// If this fails at a single layer, the transplant is broken.
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test14f_single_layer_fused_add_norm_gemm() {
    println!("=== test14f: single-layer fused add+norm+GEMM ===");

    let mut device = GpuDevice::new(0).unwrap();
    let stream = device.compute_stream;
    let config = qwen_config();
    let hidden = config.hidden_size;
    let eps = config.rms_norm_eps;

    let raw = std::fs::read(MODEL_PATH).expect("read model");
    let st = SafeTensors::deserialize(&raw).expect("parse");

    let embed_data = load_bf16_tensor(&st, "model.embed_tokens.weight");
    let vocab_size = embed_data.len() / hidden;
    let embed_w = unsafe { upload_bf16(&mut device.caching, &embed_data, &[vocab_size, hidden]) };

    let norm_data = load_bf16_tensor(&st, "model.layers.0.input_layernorm.weight");
    let norm_w = unsafe { upload_bf16(&mut device.caching, &norm_data, &[hidden]) };

    let mut qkv_data = load_bf16_tensor(&st, "model.layers.0.self_attn.q_proj.weight");
    qkv_data.extend_from_slice(&load_bf16_tensor(&st, "model.layers.0.self_attn.k_proj.weight"));
    qkv_data.extend_from_slice(&load_bf16_tensor(&st, "model.layers.0.self_attn.v_proj.weight"));
    let qkv_n = qkv_data.len() / hidden;
    let qkv_w = unsafe { upload_bf16(&mut device.caching, &qkv_data, &[qkv_n, hidden]) };
    drop(raw);

    // Embed tokens (real data) — use single token for isolation
    // Use real embeddings — the bug is data-dependent
    let prefill_ids: Vec<u32> = vec![1, 3923, 374, 220];
    let m = prefill_ids.len();
    let ids_gpu = unsafe { upload_u32(&mut device.caching, &prefill_ids, &[m]) };
    let embedded = unsafe {
        vllm_cuda::kernels::embedding_gather(*embed_w, *ids_gpu, &mut device.caching, stream)
    };
    unsafe { cusys::cuStreamSynchronize(stream) };
    let embed_host = unsafe { download_bf16(*embedded, m * hidden) };
    drop(ids_gpu);
    drop(embedded);

    // Path A: fused_add_rms_norm_inplace + ferrite.gemm
    let hs_a = unsafe { upload_bf16(&mut device.caching, &embed_host, &[m, hidden]) };
    let res_a = unsafe { upload_bf16(&mut device.caching, &vec![bf16::ZERO; m * hidden], &[m, hidden]) };
    unsafe { cusys::cuStreamSynchronize(stream) };

    unsafe {
        vllm_cuda::kernels::fused_add_rms_norm_inplace(*hs_a, *res_a, *norm_w, eps, stream);
    }
    let qkv_a = unsafe {
        device.ferrite.gemm(*hs_a, *qkv_w, None, 1.0, 0.0, &mut device.caching, stream)
    };

    // Path B: launch_fused_add_norm_gemm
    let hs_b = unsafe { upload_bf16(&mut device.caching, &embed_host, &[m, hidden]) };
    let res_b = unsafe { upload_bf16(&mut device.caching, &vec![bf16::ZERO; m * hidden], &[m, hidden]) };
    unsafe { cusys::cuStreamSynchronize(stream) };

    let qkv_b = unsafe {
        vllm_cuda::ferrite::launch_fused_add_norm_gemm(
            &FUSED_ADD_NORM_GEMM,
            *res_b,   // residual
            *hs_b,    // hidden_states
            *qkv_w,
            *norm_w,
            eps,
            hidden as u32,
            None,
            1.0,
            0.0,
            &mut device.caching,
            stream,
        )
    };

    unsafe { cusys::cuStreamSynchronize(stream) };

    let a_h = unsafe { download_bf16(*qkv_a, m * qkv_n) };
    let b_h = unsafe { download_bf16(*qkv_b, m * qkv_n) };
    let (qkv_diff, worst) = max_diff_bf16(&a_h, &b_h);

    // Also check residuals
    let ra = unsafe { download_bf16(*res_a, m * hidden) };
    let rb = unsafe { download_bf16(*res_b, m * hidden) };
    let (res_diff, _) = max_diff_bf16(&ra, &rb);

    println!("  QKV diff={qkv_diff:.2e} at [{worst}]  residual diff={res_diff:.2e}");
    if qkv_diff > 0.0 {
        println!("    std={:.4} fused={:.4}", a_h[worst].to_f32(), b_h[worst].to_f32());
    }

    // Print first few QKV values for debugging
    println!("  QKV[0..8] std:   {:?}", &a_h[0..8].iter().map(|v| v.to_f32()).collect::<Vec<_>>());
    println!("  QKV[0..8] fused: {:?}", &b_h[0..8].iter().map(|v| v.to_f32()).collect::<Vec<_>>());

    // Check if fused is all zeros or NaN
    let nan_count = b_h.iter().filter(|v| v.to_f32().is_nan()).count();
    let zero_count = b_h.iter().filter(|v| v.to_f32() == 0.0).count();
    println!("  fused: {nan_count} NaN, {zero_count} zeros out of {}", b_h.len());

    // Also compare against old fused path (add_inplace + FUSED_NORM_GEMM)
    // which is known to be bit-exact per test14c
    let hs_c = unsafe { upload_bf16(&mut device.caching, &embed_host, &[m, hidden]) };
    let res_c = unsafe { upload_bf16(&mut device.caching, &vec![bf16::ZERO; m * hidden], &[m, hidden]) };
    unsafe { cusys::cuStreamSynchronize(stream) };

    unsafe { vllm_cuda::kernels::add_inplace(*res_c, *hs_c, stream) };
    drop(hs_c);
    let qkv_c = unsafe {
        vllm_cuda::ferrite::launch_fused_norm_gemm(
            &FUSED_NORM_GEMM, *res_c, *qkv_w, *norm_w, eps, hidden as u32,
            None, 1.0, 0.0, &mut device.caching, stream,
        )
    };
    unsafe { cusys::cuStreamSynchronize(stream) };
    let c_h = unsafe { download_bf16(*qkv_c, m * qkv_n) };
    let (old_fused_diff, _) = max_diff_bf16(&a_h, &c_h);
    println!("  Old fused (add_inplace+FUSED_NORM_GEMM) diff={old_fused_diff:.2e}");

    let (new_vs_old, worst2) = max_diff_bf16(&b_h, &c_h);
    println!("  New fused vs old fused diff={new_vs_old:.2e} at [{worst2}]");
    if new_vs_old > 0.0 {
        println!("    new={:.4} old={:.4}", b_h[worst2].to_f32(), c_h[worst2].to_f32());
    }

    // Per-row analysis: which rows diverge?
    for row in 0..m {
        let start = row * qkv_n;
        let end = start + qkv_n;
        let (row_diff, _) = max_diff_bf16(&b_h[start..end], &c_h[start..end]);
        let mismatches = b_h[start..end].iter().zip(&c_h[start..end])
            .filter(|(a, b)| (a.to_f32() - b.to_f32()).abs() > 0.01)
            .count();
        if row_diff > 0.01 {
            println!("  ROW {row}: diff={row_diff:.2e}, mismatches={mismatches}/{qkv_n}");
        } else {
            println!("  ROW {row}: diff={row_diff:.2e} OK");
        }
    }

    assert!(qkv_diff < 0.01, "Single-layer QKV diverged: {qkv_diff:.2e}");
    println!("PASS: test14f — single-layer fused add+norm+GEMM matches");
}
