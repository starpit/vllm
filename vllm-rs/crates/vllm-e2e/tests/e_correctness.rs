// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Correctness tests: compare our engine's greedy logprobs against HF Transformers golden refs.
//!
//! Golden references are pre-generated JSON files in `testdata/golden/`.
//! See `scripts/generate_golden_refs.py` to regenerate them.
//!
//! Run with:
//!   cargo test -p vllm-e2e --features e2e,cuda --release --test e_correctness -- --ignored --test-threads=1

#![cfg(feature = "e2e")]

use vllm_e2e::assertions::{check_logprobs_close, extract_engine_output, load_golden_refs};
use vllm_e2e::{Client, TestModels, TestServer};
use vllm_serve::protocol::{CompletionPrompt, CompletionRequest};

fn completion_request(prompt: &str, max_tokens: u32, logprobs: u32) -> CompletionRequest {
    CompletionRequest {
        prompt: Some(CompletionPrompt::Single(prompt.to_string())),
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        logprobs: Some(logprobs),
        ..serde_json::from_str("{}").unwrap()
    }
}

async fn run_correctness_test(model: &str, golden_key: &str) {
    let golden = load_golden_refs(golden_key);

    let server = TestServer::builder(model)
        .start()
        .await
        .expect("server should start");
    let client = Client::new(server.base_url());

    for (i, golden_result) in golden.results.iter().enumerate() {
        let req = completion_request(
            &golden_result.prompt,
            golden.max_tokens,
            golden.num_logprobs,
        );
        let resp = client
            .completion(&req)
            .await
            .expect("completion should succeed");
        assert!(!resp.choices.is_empty(), "prompt {i}: no choices returned");

        let engine_output = extract_engine_output(&resp.choices[0]);
        check_logprobs_close(golden_result, &engine_output, i);
    }
}

// ---------------------------------------------------------------------------
// CUDA correctness tests
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_correctness_qwen2_0_5b() {
    run_correctness_test(TestModels::QWEN2_0_5B_CUDA, "qwen2_0_5b").await;
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_correctness_smollm_135m() {
    run_correctness_test(TestModels::SMOLLM_135M_CUDA, "smollm_135m").await;
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_correctness_gemma2_2b() {
    run_correctness_test(TestModels::GEMMA2, "gemma2_2b").await;
}
