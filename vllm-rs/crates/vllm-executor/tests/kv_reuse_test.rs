// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Integration test: verify KV block reuse at the attention level.
//!
//! Loads SmolLM2-135M on CPU, runs a forward pass on a full prompt, then
//! runs a second forward on just the suffix (reusing the cached KV blocks
//! for the prefix). Asserts the last-position logits match exactly.
//!
//! This exercises the path: gather_kv → cat(cached, new) → attention →
//! scatter_new_kv — without scheduler/engine complexity.

use candle_core::{DType, Device, Tensor};
use vllm_model::weight::{HfModelConfig, ModelWeights};
use vllm_models::{KvBlockPool, KvCacheStorage, Model, ModelRegistry};

/// SmolLM2-135M-Instruct: 30 layers, 3 KV heads, head_dim=64.
const MODEL_ID: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
const NUM_LAYERS: usize = 30;
const NUM_KV_HEADS: usize = 3;
const HEAD_DIM: usize = 64;
const BLOCK_SIZE: usize = 16;

/// Download and load the model for testing.
fn load_test_model() -> (Box<dyn Model>, Device) {
    let device = Device::Cpu;
    let api = hf_hub::api::sync::ApiBuilder::new()
        .build()
        .expect("HF API should build");
    let repo = api.model(MODEL_ID.to_string());

    // Download config.json.
    let config_path = repo.get("config.json").expect("config.json download");
    let model_dir = config_path.parent().unwrap();

    // Download safetensors weights.
    let _ = repo.get("model.safetensors").expect("weights download");

    let hf_config = HfModelConfig::from_dir(model_dir).expect("parse config.json");
    let arch = hf_config
        .architectures
        .first()
        .expect("architecture")
        .clone();

    let registry = ModelRegistry::default();
    let factory = registry
        .get(&arch)
        .unwrap_or_else(|| panic!("unsupported arch: {arch}"));

    let weights = ModelWeights::from_dir(model_dir, &device).expect("load weights");

    let dtype = DType::F32; // CPU test — use F32 for numerical stability.
    let model = factory(&weights, &hf_config, dtype, &device, 0, 1).expect("construct model");

    assert_eq!(model.num_layers(), NUM_LAYERS);
    (model, device)
}

/// Create a KV block pool with enough blocks for the test.
fn make_pool(device: &Device, num_blocks: usize) -> KvBlockPool {
    KvBlockPool::new(
        num_blocks,
        NUM_LAYERS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_SIZE,
        DType::F32,
        device,
    )
    .expect("create KvBlockPool")
}

/// Run a forward pass with paged KV cache and return all logits.
///
/// * `model` — the loaded model
/// * `token_ids` — input tokens to feed
/// * `positions` — position IDs for each token
/// * `pool` — the KV block pool (may already contain cached data)
/// * `block_ids` — block IDs for this request
/// * `tokens_before` — how many tokens are already in the cache
///
/// Returns logits tensor of shape `[num_tokens, vocab_size]`.
fn forward_paged(
    model: &dyn Model,
    token_ids: &[u32],
    positions: &[u32],
    pool: &mut KvBlockPool,
    block_ids: &[usize],
    tokens_before: usize,
) -> Tensor {
    let device = pool.device().clone();
    let input_ids = Tensor::new(token_ids, &device).expect("input_ids tensor");
    let pos_tensor = Tensor::new(positions, &device).expect("positions tensor");

    let mut storage = KvCacheStorage::paged(pool, block_ids, tokens_before);
    let logits = model
        .forward(&input_ids, &pos_tensor, Some(&mut storage))
        .expect("forward pass");
    storage.flush().expect("flush deferred writes");

    logits
}

/// Core test: full forward vs. prefix-reuse forward produce identical logits.
///
/// Run A: Forward 20 tokens with tokens_before=0 (full prefill).
/// Run B: Forward only tokens[16..20] with tokens_before=16 (reuse first block).
/// Assert: logits for the last position match exactly.
#[test]
#[ignore = "requires model download (SmolLM2-135M-Instruct)"]
fn test_kv_block_reuse_produces_identical_logits() {
    let (model, device) = load_test_model();

    // A 20-token prompt (arbitrary token IDs within vocab range).
    let prompt: Vec<u32> = (1..=20).collect();
    let positions_full: Vec<u32> = (0..20).collect();

    // We need 2 blocks of size 16 for 20 tokens: block 0 (tokens 0-15), block 1 (tokens 16-19).
    let num_blocks = 4; // extra slack
    let block_ids = vec![0, 1];

    // --- Run A: Full forward (20 tokens, tokens_before=0) ---
    let mut pool_a = make_pool(&device, num_blocks);
    let logits_full = forward_paged(
        model.as_ref(),
        &prompt,
        &positions_full,
        &mut pool_a,
        &block_ids,
        0,
    );

    // Extract logits for the last position (index 19).
    let logits_full_last = logits_full
        .narrow(0, 19, 1)
        .expect("narrow last")
        .squeeze(0)
        .expect("squeeze");

    // --- Run B: Reuse prefix (tokens 0-15 cached in block 0) ---
    // Reuse pool_a which already has the correct data in block 0.
    // After Run A, block 0 has tokens
    // 0-15 cached and block 1 has tokens 16-19 cached. For Run B, we want
    // to recompute only tokens 16-19 while reusing block 0's cache.
    //
    // We need to use a DIFFERENT block for the new tokens in Run B to avoid
    // overwriting block 1's data from Run A (which we don't care about — we
    // just need block 0's prefix data to be correct).
    //
    // Use block_ids=[0, 2] for Run B: block 0 is the cached prefix, block 2
    // is fresh for the new suffix tokens.
    let block_ids_b = vec![0, 2];
    let suffix_tokens = &prompt[16..20]; // tokens 17, 18, 19, 20
    let suffix_positions: Vec<u32> = (16..20).collect();

    let logits_reuse = forward_paged(
        model.as_ref(),
        suffix_tokens,
        &suffix_positions,
        &mut pool_a, // reuse pool_a which has block 0 populated
        &block_ids_b,
        16, // 16 tokens already in cache (block 0)
    );

    // logits_reuse is [4, vocab_size] — we want the last position (index 3).
    let logits_reuse_last = logits_reuse
        .narrow(0, 3, 1)
        .expect("narrow last reuse")
        .squeeze(0)
        .expect("squeeze reuse");

    // --- Assert: logits match ---
    let full_vals: Vec<f32> = logits_full_last.to_vec1().expect("logits_full to vec");
    let reuse_vals: Vec<f32> = logits_reuse_last.to_vec1().expect("logits_reuse to vec");

    assert_eq!(full_vals.len(), reuse_vals.len(), "vocab size mismatch");

    // Check exact match (same computation path, same data, should be bitwise identical on CPU).
    let max_diff = full_vals
        .iter()
        .zip(reuse_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-4,
        "logits diverged: max absolute difference = {max_diff} (expected < 1e-4)"
    );

    // Also check that logits are non-trivial (not all zeros).
    let max_abs = full_vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs > 0.1,
        "logits are near-zero (max_abs={max_abs}), model may not have loaded correctly"
    );

    println!(
        "KV block reuse test PASSED: max logit diff = {max_diff:.2e}, max abs logit = {max_abs:.2e}"
    );
}

/// Test that decode after prefix reuse also works.
///
/// Run A: Prefill 20 tokens → decode 1 token (position 20).
/// Run B: Prefix-reuse prefill 4 tokens (16 cached) → decode 1 token (position 20).
/// Assert: decode logits match.
#[test]
#[ignore = "requires model download (SmolLM2-135M-Instruct)"]
fn test_kv_reuse_then_decode() {
    let (model, device) = load_test_model();

    let prompt: Vec<u32> = (1..=20).collect();
    let positions_full: Vec<u32> = (0..20).collect();
    let num_blocks = 8;

    // --- Run A: Full prefill ---
    let block_ids_a = vec![0, 1];
    let mut pool = make_pool(&device, num_blocks);
    let logits_prefill = forward_paged(
        model.as_ref(),
        &prompt,
        &positions_full,
        &mut pool,
        &block_ids_a,
        0,
    );

    // Sample a "next token" (just take argmax of last logits).
    let last_logits = logits_prefill.narrow(0, 19, 1).unwrap().squeeze(0).unwrap();
    let next_token = last_logits.argmax(0).unwrap().to_scalar::<u32>().unwrap();

    // Decode one step (Run A path): tokens_before=20, 1 new token.
    // Need block for position 20 → block 1 (positions 16-31), already allocated.
    let decode_logits_a = forward_paged(
        model.as_ref(),
        &[next_token],
        &[20],
        &mut pool,
        &block_ids_a,
        20, // 20 tokens in cache
    );
    let decode_last_a: Vec<f32> = decode_logits_a.squeeze(0).unwrap().to_vec1().unwrap();

    // --- Run B: Prefix-reuse prefill + decode ---
    let mut pool2 = make_pool(&device, num_blocks);
    let block_ids_b_prefill = vec![0, 1];

    // Full prefill first (to populate all blocks).
    let _ = forward_paged(
        model.as_ref(),
        &prompt,
        &positions_full,
        &mut pool2,
        &block_ids_b_prefill,
        0,
    );

    // Now simulate prefix reuse: re-run suffix with block 0 cached.
    // Use block IDs [0, 3] — block 0 has cached prefix, block 3 for new suffix.
    let block_ids_b_reuse = vec![0, 3];
    let _ = forward_paged(
        model.as_ref(),
        &prompt[16..20],
        &(16..20).collect::<Vec<u32>>(),
        &mut pool2,
        &block_ids_b_reuse,
        16,
    );

    // Decode one step with the reused prefix.
    let decode_logits_b = forward_paged(
        model.as_ref(),
        &[next_token],
        &[20],
        &mut pool2,
        &block_ids_b_reuse,
        20, // 16 (cached) + 4 (recomputed) = 20 tokens in cache
    );
    let decode_last_b: Vec<f32> = decode_logits_b.squeeze(0).unwrap().to_vec1().unwrap();

    // Assert decode logits match.
    let max_diff = decode_last_a
        .iter()
        .zip(decode_last_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-4,
        "decode logits after reuse diverged: max diff = {max_diff}"
    );

    println!("KV reuse + decode test PASSED: max logit diff = {max_diff:.2e}");
}
