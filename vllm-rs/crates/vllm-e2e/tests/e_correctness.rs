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

use vllm_e2e::assertions::{
    check_logprobs_close, check_logprobs_close_with_threshold, extract_engine_output,
    load_golden_refs,
};
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
    run_correctness_test_with_threshold(model, golden_key, 10).await
}

/// BNB4 variant — the fused-dequant-GEMM in
/// `bitsandbytes.matmul_4bit` accumulates in a different order than
/// ferrite's dequant-to-scratch → cuBLAS-GEMM pipeline. Same
/// `code[nibble] * absmax[block]` math per-weight, but the dot-
/// product accumulation order along K differs between the two
/// kernels. Over 28 layers × per-token the noise is enough to push
/// token-candidates outside a top-20 window within a few decode
/// steps. Loosen the threshold so top-N exits at position >= 3 are
/// warnings instead of hard fails — the full golden still catches
/// gross regressions (e.g. wrong tensor shape, wrong absmax
/// ordering) because the first 3 tokens and the top-N sets
/// themselves still line up.
async fn run_correctness_test_with_threshold(
    model: &str,
    golden_key: &str,
    late_divergence_threshold: usize,
) {
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
        check_logprobs_close_with_threshold(
            golden_result,
            &engine_output,
            i,
            late_divergence_threshold,
        );
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

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_correctness_granite_3_3_2b() {
    run_correctness_test(TestModels::GRANITE, "granite_3_3_2b").await;
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_correctness_llama_3_2_1b_awq() {
    // AWQ Llama-3.2-1B via the ferrite-forward Marlin* impl family +
    // the `ferrite_kernels::layers_quant` AWQ→Marlin loader. Golden
    // generated from Python vLLM on `AMead10/Llama-3.2-1B-Instruct-AWQ`.
    run_correctness_test(TestModels::LLAMA_3_2_1B_AWQ, "llama_3_2_1b_awq").await;
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_correctness_gemma2_2b_gptq() {
    // GPTQ Gemma2-2B via ferrite — exercises MarlinFusedGateUpGeluMulImpl
    // (new, landed alongside this test) on top of the alternating
    // sliding/full attention + softcap stack. Golden generated from
    // Python vLLM on `qilowoq/gemma-2-2B-it-4Bit-GPTQ`.
    run_correctness_test(TestModels::GEMMA2_2B_GPTQ_INT4, "gemma2_2b_gptq").await;
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_correctness_tinyllama_1b_gptq_desc_act() {
    // GPTQ TinyLlama-1.1B-Chat-v0.3 — exercises the desc_act=true
    // code path (g_idx argsort + sort_indices → gptq_repack_into perm)
    // that Qwen2.5-0.5B-GPTQ and Gemma2-2B-GPTQ skip (both desc_act=false).
    run_correctness_test(
        TestModels::TINYLLAMA_1B_GPTQ_DESC_ACT,
        "tinyllama_1b_gptq_desc_act",
    )
    .await;
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_correctness_tinyllama_1b_w4a16_ct() {
    // Compressed-tensors INT4 (Neural Magic pack-quantized) —
    // exercises `GptqLayout::WeightPacked` in
    // `MarlinLinear::load_gptq[_concat]`: `.weight_packed [N, K/8]`
    // + `.weight_scale [N, num_groups]` sniffed off disk, CPU-
    // transposed to AutoGPTQ-native `[K/8, N]` / `[num_groups, N]`,
    // then the same uint4b8 `gptq_repack_into` + Marlin kernel
    // everything else uses. Golden generated from Python vLLM on
    // `nm-testing/TinyLlama-1.1B-Chat-v1.0-W4A16-e2e`.
    run_correctness_test(TestModels::TINYLLAMA_1B_W4A16_CT, "tinyllama_1b_w4a16_ct").await;
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_correctness_qwen2_0_5b_gptq() {
    // GPTQ Qwen2.5-0.5B via the ferrite-forward Marlin* impl family
    // + `ferrite_kernels::layers_quant::MarlinLinear::load_gptq` /
    // `load_gptq_concat`. Golden generated from Python vLLM on
    // `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4` (symmetric, desc_act=false,
    // group_size=128). Both FERRITE_ENABLED and FERRITE_DISABLE=1
    // runs should match within the same top-N tolerance the other
    // goldens use.
    run_correctness_test(TestModels::QWEN2_0_5B_GPTQ_INT4, "qwen2_0_5b_gptq").await;
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_correctness_qwen3_0_6b() {
    run_correctness_test(TestModels::QWEN3_0_6B_CUDA, "qwen3_0_6b").await;
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_correctness_qwen3_0_6b_bnb_4bit() {
    // BNB4 NF4 Qwen3-0.6B — exercises ferrite-forward's
    // `Bnb4bitLinear::load[_concat]` + `Bnb4GemmImpl` /
    // `Bnb4FusedGateUpSiluMulImpl` / `Bnb4FusedQkvRope*Impl` path.
    // Qwen3's per-head QK-norm prevents the fused QKV matcher from
    // claiming q/k/v gemms adjacent to RopeAppend, so this also
    // covers the Bnb4GemmImpl singleton path for leftover gemms
    // (`gemm_is_fusion_partner` deference is intentionally off for
    // BNB4). Golden generated from Python vLLM on
    // `unsloth/Qwen3-0.6B-bnb-4bit`.
    //
    // Threshold=3: `bitsandbytes.matmul_4bit` accumulates dequant-
    // GEMM fused per-block; our path dequants to scratch then
    // cuBLAS-GEMMs. Same math per-weight, different K-axis
    // accumulation order — top-N-window drift exceeds bf16 noise
    // and crosses the window within 3-5 decode steps. The
    // threshold=3 comparison still catches tensor-shape / absmax-
    // ordering / wrong-kernel regressions (prefill must produce
    // matching first-3 tokens) while tolerating the known
    // accumulation-order divergence that's endemic to BNB4 cross-
    // implementation comparison.
    run_correctness_test_with_threshold(TestModels::QWEN3_0_6B_BNB_4BIT, "qwen3_0_6b_bnb_4bit", 3)
        .await;
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_correctness_gemma3_1b() {
    run_correctness_test(TestModels::GEMMA3_1B_IT_CUDA, "gemma3_1b").await;
}
