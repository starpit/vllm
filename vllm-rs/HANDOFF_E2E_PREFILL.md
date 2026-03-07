# Handoff: E2E Test Gaps + Prefill Graph Investigation

Worktree: `.claude/worktrees/cuda-models/` on branch `feat/cuda-qwen2-gemma2`

## Bug found and fixed

**Root cause**: Paged FlashAttention-2 produces incorrect results for prefill (q_len > 1).
Python vLLM uses contiguous (non-paged) FA2 for prefill and paged FA2 only for decode.
Our CudaWorker was using paged FA2 for both.

**Two bugs, same root cause**:
1. **CudaWorker always garbage**: Prefill graphs (`PrefillGraphRunner`) captured paged FA2 for
   q_len > 1. Since prefill graphs fire for most chat prompts, every first token was wrong,
   cascading into full garbage output.
2. **Candle worker multi-turn garbage**: Paged FA2 used during prefill on CUDA. First turn
   happened to work (non-batched path), second turn corrupted.

**Fix applied** (3 files):
- `crates/vllm-cuda/src/model/llama.rs`: Prefill uses `flash_attn_contiguous()`, decode uses `flash_attn_paged()`. Gate: `max_seqlen_q == 1`.
- `crates/vllm-cuda/src/model/gemma2.rs`: Same prefill/decode split.
- `crates/vllm-cuda/src/kernels.rs`: Added FFI + wrapper for `run_mha` (non-paged FA2). Added `flash_attn_contiguous()` public API.
- `third_party/candle-flash-attn/kernels/flash_api.cu`: Added `cudaStream_t` param to `run_mha` (was hardcoded to stream 0).
- `crates/vllm-executor/src/cuda_worker.rs`: Prefill graphs disabled (`use_prefill_graph = false`).

**Verified**: `vllm chat --model unsloth/Llama-3.2-3B-Instruct` on L40S produces coherent Rayleigh scattering explanation.

## Task 1: E2E test gaps to close

### 1a. Semantic completion correctness (CUDA)
Current CUDA E2E tests only check `!resp.choices[0].text.is_empty()`. They should validate content:
```rust
// In test_cuda_qwen2_completion and similar:
let text = &resp.choices[0].text.to_lowercase();
assert!(text.contains("paris") || text.contains("france"),
    "expected 'paris' in completion of 'The capital of France is', got: {}", text);
```

### 1b. Multi-turn chat E2E (NEW — catches KV cache continuity bugs)
Add a 2-turn chat test that verifies the model remembers context from turn 1:
```rust
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_multi_turn_chat() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start().await.unwrap();
    let client = Client::new(server.base_url());

    // Turn 1: establish a fact
    let req1 = ChatCompletionRequest {
        model: None,
        messages: vec![
            user_msg("My name is Claude. Please remember that. Reply with just 'OK'."),
        ],
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..Default::default()
    };
    let resp1 = client.chat_completion(&req1).await.unwrap();
    assert_valid_chat_response(&resp1);

    // Turn 2: query the fact — requires KV cache from turn 1
    let req2 = ChatCompletionRequest {
        model: None,
        messages: vec![
            user_msg("My name is Claude. Please remember that. Reply with just 'OK'."),
            assistant_msg(&resp1.choices[0].message.content.clone().unwrap_or_default()),
            user_msg("What is my name?"),
        ],
        max_tokens: Some(20),
        temperature: Some(0.0),
        ..Default::default()
    };
    let resp2 = client.chat_completion(&req2).await.unwrap();
    assert_valid_chat_response(&resp2);
    let text = resp2.choices[0].message.content.as_deref().unwrap_or("").to_lowercase();
    assert!(text.contains("claude"),
        "turn 2 should remember 'Claude', got: {}", text);
}
```

### 1c. Non-greedy sampling E2E (NEW — catches GPU sampling bugs)
Add a test with temperature > 0 that validates output coherence:
```rust
async fn test_cuda_nongreedy_chat() {
    // ... setup ...
    let request = ChatCompletionRequest {
        messages: vec![user_msg("Say hello in one sentence.")],
        max_tokens: Some(50),
        temperature: Some(0.7),  // non-greedy
        ..Default::default()
    };
    let resp = client.chat_completion(&request).await.unwrap();
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 5);
    // Check it contains actual words, not garbage
    let word_count = text.split_whitespace().count();
    assert!(word_count >= 3, "expected at least 3 words, got: {}", text);
}
```

### Where to add tests
- File: `crates/vllm-e2e/tests/e1_basic_serving.rs` (after existing `test_cuda_*` tests)
- Need `assistant_msg()` helper if not already in `e2e/src/assertions.rs`
- All CUDA tests are `#[cfg(feature = "cuda")]` + `#[ignore]`
- Also add equivalent tests for `cuda-backend` feature (CudaWorker path) if there's a feature gate

## Task 2: Investigate prefill graph fix

Prefill graphs are currently disabled (`use_prefill_graph = false`). They provided 9-10% prefill latency improvement. Options to re-enable:

### Option A: Capture prefill graphs with contiguous FA2
- During `PrefillGraphRunner::capture()`, the model forward uses paged FA2
- Fix: make the model aware of capture mode, or pass a flag to use contiguous FA2 during capture
- Problem: contiguous FA2 needs fresh K/V tensors, but graph capture allocates them in the arena. After capture, the arena is reset. During replay, the arena is reset to the same state — K/V would land at the same pointers. This SHOULD work since `reshape_and_cache` writes to paged cache AND the contiguous K/V are used for attention in the same forward pass.
- The prefill graph captures: embedding → layers (with attention) → lm_head → argmax. The contiguous FA2 would be captured inside each layer's attention. K/V are ephemeral (arena), Q/K/V come from the same QKV GEMM.

### Option B: Use paged FA2 but fix the kernel
- Investigate WHY paged FA2 fails for q_len > 1
- The `run_mha_paged` kernel passes `page_block_size` and `block_table` to the FA2 kernel
- Possible issues: stride calculations, block table indexing for multi-token queries, num_splits auto-tuning
- Compare parameter setup between `run_mha` and `run_mha_paged` in `flash_api.cu`

### Recommended approach
Option A is safer and matches Python vLLM's architecture. The model's `forward()` already has the prefill/decode split — just need to ensure graph capture uses the contiguous path.

## Pod testing methodology

```bash
# Kill stale GPU processes
oc rsh nick bash -c 'kill $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) 2>/dev/null'

# Sync files (from worktree dir: .claude/worktrees/cuda-models/vllm-rs/)
oc rsync crates/ nick:/root/vllm/vllm-rs/crates/ --exclude=target --delete
# For single files:
cat <file> | oc rsh nick bash -c 'cat > /root/vllm/vllm-rs/<file>'
# For third_party changes:
oc rsync third_party/ nick:/root/vllm/vllm-rs/third_party/ --exclude=target

# Build
oc rsh nick bash -c 'cd /root/vllm/vllm-rs && export RUSTC_WRAPPER=/usr/bin/sccache && cargo build -p vllm-cli --features cuda-backend --release 2>&1'

# Quick chat test (pipe input, discard logs)
oc rsh nick bash -c 'cd /root/vllm/vllm-rs && echo "why is the sky blue?" | ./target/release/vllm chat --model unsloth/Llama-3.2-3B-Instruct 2>/dev/null'

# Chat test with logs (to verify code paths)
oc rsh nick bash -c 'cd /root/vllm/vllm-rs && echo "why is the sky blue?" | RUST_LOG=info ./target/release/vllm chat --model unsloth/Llama-3.2-3B-Instruct 2>&1'

# Unit tests
oc rsh nick bash -c 'cd /root/vllm/vllm-rs && export RUSTC_WRAPPER=/usr/bin/sccache && cargo test -p vllm-cuda --features cuda -- --include-ignored 2>&1'

# E2E tests
oc rsh nick bash -c 'cd /root/vllm/vllm-rs && export RUSTC_WRAPPER=/usr/bin/sccache && cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda -- --ignored --test-threads=1 2>&1'

# Bench latency (verify no regression from disabling prefill graphs)
oc rsh nick bash -c 'cd /root/vllm/vllm-rs && ./target/release/vllm bench latency --model Qwen/Qwen2.5-0.5B --temperature 0 2>&1'

# IMPORTANT: Llama-3.2-3B is gated — 401 on pod. Use Qwen2.5-0.5B or Qwen2.5-3B for benchmarking.
# IMPORTANT: Never run two bench commands concurrently (GPU contention = garbage numbers).
# IMPORTANT: Always kill stale processes before testing (GPU memory leak from crashed runs).
```

## Key files
- LLaMA attention (prefill/decode split): `crates/vllm-cuda/src/model/llama.rs:340-390`
- Gemma2 attention (same split): `crates/vllm-cuda/src/model/gemma2.rs:275-310`
- Contiguous FA2 wrapper: `crates/vllm-cuda/src/kernels.rs` (`flash_attn_contiguous`)
- FA2 C entry points: `third_party/candle-flash-attn/kernels/flash_api.cu` (`run_mha`, `run_mha_paged`)
- Prefill graph disable: `crates/vllm-executor/src/cuda_worker.rs:1358` (`use_prefill_graph = false`)
- Prefill graph runner: `crates/vllm-cuda/src/graph.rs` (`PrefillGraphRunner`)
- E2E tests: `crates/vllm-e2e/tests/e1_basic_serving.rs`
- E2E assertions: `crates/vllm-e2e/src/assertions.rs`
