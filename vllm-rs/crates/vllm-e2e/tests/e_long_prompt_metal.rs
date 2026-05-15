// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E coverage for long-prompt / chunked-prefill on Metal.
//!
//! The metal Llama bucket spec is `workloads = [1, 8, 64, 512, 4096]`
//! (`crates/ferrite-model-llama/src/lib.rs`). A prompt that fits in
//! the largest bucket runs as ONE contiguous prefill call; a prompt
//! larger than 4096 tokens forces the engine to chunk, and chunks
//! 2+ must read the prior chunk's K from the paged KV cache.
//!
//! The contiguous prefill kernel cannot read prior cached K — its
//! K-axis is `cu_seqlens_q` (new tokens only). Phase A landed the
//! `attention_prefill_sdpa_v2_paged_*` kernel + plumbing
//! (`60ef837f4`); Phase B (worker bucket plumbing + lowering arm +
//! engine `seqused_k` for prefill) is required to actually route
//! chunked prefill through the paged kernel.
//!
//! Phase B landed: the metal macro adapter now emits
//! `Instruction::AttentionPrefillPaged` for prefill on every Llama-arch
//! model (`crates/ferrite-forward-macro/src/metal/attention.rs`), and
//! the lowering arm routes that to `KernelId::AttentionPrefillSdpaPaged`.
//! Engine-side `seqused_k` / `block_table` were already populated by
//! `FerriteWorker::execute_model` (`vllm-executor/src/ferrite_worker.rs`).
//! Both chunked-prefill and multi-turn tests below are now gating
//! regressions — drop the `#[ignore]` and the suite must stay green.
//!
//! Run with:
//!   `cargo test -p vllm-e2e --features e2e --test e_long_prompt_metal`

#![cfg(feature = "e2e")]
#![cfg(target_os = "macos")]

use vllm_e2e::assertions::assert_valid_completion_response;
use vllm_e2e::{Client, TestServer};
use vllm_serve::protocol::{CompletionPrompt, CompletionRequest};

/// Greedy completion request with a fixed `max_tokens` so the test
/// terminates predictably even if the model fixates on a pattern.
fn greedy_completion(prompt: &str, max_tokens: u32) -> CompletionRequest {
    CompletionRequest {
        prompt: Some(CompletionPrompt::Single(prompt.to_string())),
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        ..serde_json::from_str(r#"{}"#).unwrap()
    }
}

/// Repeat a sentence until the rough token estimate hits the target.
/// SmolLM-135M's tokenizer averages ~4 chars/token on this kind of
/// boilerplate prose, so we count chars and divide.
fn build_prompt(target_tokens: usize, suffix: &str) -> String {
    let sentence = "The grass is green and the sky is blue. ";
    let chars_per_token = 4;
    let target_chars = target_tokens * chars_per_token;
    let reps = (target_chars / sentence.len()).max(1);
    let mut s = String::with_capacity(target_chars + suffix.len());
    for _ in 0..reps {
        s.push_str(sentence);
    }
    s.push_str(suffix);
    s
}

async fn start_metal_server() -> (TestServer, Client) {
    // SmolLM-135M-Instruct uses the same llama-arch bucket spec
    // (`workloads = [1, 8, 64, 512, 4096]`) as Llama-3.2 — the
    // 4096-token bucket cap is what triggers chunking.
    //
    // SmolLM-135M's default `max_model_len` is 2048; bump it to 8192
    // so a >4096-token prompt fits in the model context (otherwise
    // the engine bails before chunking matters — finish_reason="length"
    // with completion_tokens=0).
    let server = TestServer::builder(vllm_e2e::TestModels::SMOLLM)
        .with_device("metal")
        .with_args(&["--max-model-len", "8192"])
        .start()
        .await
        .expect("server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

/// Sanity baseline: ~1k-token prompt fits in the M=4096 bucket as a
/// single (paged) prefill call. Catches regressions in the prefill
/// path independent of the chunked / multi-turn cases below.
#[tokio::test(flavor = "multi_thread")]
async fn long_prompt_metal_within_bucket_completes() {
    let (_server, client) = start_metal_server().await;

    let prompt = build_prompt(1000, "Q: What color is the grass? A:");
    let request = greedy_completion(&prompt, 8);

    let resp = client
        .completion(&request)
        .await
        .expect("completion request should succeed (single-bucket prefill)");
    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "single-bucket prefill must produce a non-empty completion (~1k-token prompt)"
    );
}

/// Chunked-prefill regression: ~5000-token prompt forces the engine
/// to split the prefill across two steps (chunk 1 = 4096 tokens,
/// chunk 2 ≈ 900 tokens). Chunk 2 must attend over chunk 1's prior
/// cached K — only the paged prefill kernel can do this. Phase B
/// (commits landing on top of `4224bc2e4`) wired this through the
/// macro adapter + lowering arm; before that, this case panicked or
/// produced repetitive garbage. Validated end-to-end on
/// Llama-3.2-3B at 5000 tokens (`"The grass is green."`).
#[tokio::test(flavor = "multi_thread")]
async fn long_prompt_metal_chunked_prefill_completes() {
    let (_server, client) = start_metal_server().await;

    // ~5000 tokens — needs chunked prefill (one bucket=4096 chunk +
    // one smaller chunk). Suffix is a short instruction so the model
    // has something concrete to answer (post-fix coherence check).
    let prompt = build_prompt(5000, "Q: What color is the grass? A:");
    let request = greedy_completion(&prompt, 16);

    let resp = client
        .completion(&request)
        .await
        .expect("chunked-prefill completion must not panic");
    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "chunked-prefill must produce a non-empty completion"
    );

    // Coherence smoke: the model should mention "green" given the
    // prompt's overwhelming prior. A successful paged-prefill leaves
    // the question salient; a broken chunked-prefill produces
    // garbage / repeats the input pattern.
    assert!(
        text.to_lowercase().contains("green"),
        "post-fix coherence: model should answer 'green' to the explicit question — got {text:?}"
    );
}

/// Multi-turn regression: build a conversation whose accumulated
/// context exceeds the 4096-token bucket cap on turn 2+. Each turn
/// after the first must attend over the prior turns' cached K — the
/// same primitive as chunked prefill. Gated by the same Phase B
/// landing as `long_prompt_metal_chunked_prefill_completes`.
#[tokio::test(flavor = "multi_thread")]
async fn long_prompt_metal_multi_turn_continuation_completes() {
    let (_server, client) = start_metal_server().await;

    // Turn 1: prompt ≈ 3000 tokens (single bucket, fine).
    let turn1 = build_prompt(3000, "Q: What is your favorite color? A:");
    let resp1 = client
        .completion(&greedy_completion(&turn1, 32))
        .await
        .expect("turn 1 must succeed (single-bucket prefill)");
    assert_valid_completion_response(&resp1);

    // Turn 2: append the model's reply + another long block, totaling
    // ~5500 tokens of context. Without paged prefill the worker has
    // to re-prefill from scratch (wasteful) AND chunk past the bucket
    // cap (broken).
    let turn2 = format!(
        "{}{}\n{}",
        turn1,
        resp1.choices[0].text,
        build_prompt(
            2500,
            "Q: Reflecting on the above, what one word answers the question? A:"
        ),
    );
    let resp2 = client
        .completion(&greedy_completion(&turn2, 16))
        .await
        .expect("turn 2 (chunked context) must not panic");
    assert_valid_completion_response(&resp2);
    assert!(
        !resp2.choices[0].text.is_empty(),
        "multi-turn continuation must produce a non-empty completion"
    );
}
