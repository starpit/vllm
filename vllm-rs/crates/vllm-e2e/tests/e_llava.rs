// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! LLaVA-1.5 (LlavaForConditionalGeneration) image-inference E2E.
//!
//! Validates the OpenAI CLIP-ViT-L/14 + Vicuna-7B stack end-to-end
//! through the multimodal request pipeline:
//!
//! - The chat template renders the `image` content-part marker (after
//!   the universal `image_url` → `image` part-type normalization in
//!   `vllm-serve`); HF LlavaProcessor expands it to 576 copies of
//!   `image_token_index = 32000`.
//! - Engine dispatches generically through
//!   `ferrite_vision::MmMetadata` declared by `ferrite-model-llava`
//!   (placeholder = `image_token_index` 32000, fixed 336² resize,
//!   CLIP mean/std normalization, 576 fixed tokens per image).
//! - Vision encoder runs (DSL body) and 576 projected vectors are
//!   spliced into the placeholder positions. Phase-H load-time
//!   surface — `vision_class_embedding_fold` collapses CLIP's CLS
//!   contribution into `position_embedding[0]`, and the LLaVA-side
//!   `pack_pixels_with_cls` prepends a zero row to pixels — together
//!   reproducing HF's `concat(cls, patches) + position_embedding`
//!   without a dedicated `cls_prepend` op.
//! - `Instruction::StripCls` drops the CLS row pre-projector, so the
//!   spliced rows count matches `_get_image_seq_length = 576`.
//!
//! Mirrors `e_gemma3_mm.rs` (4 cases) — one per arch's smoke /
//! text-only / image-coherence / two-image-distinguishability.
//!
//! Model: llava-hf/llava-1.5-7b-hf (~14 GB FP16)
//!
//! Run with:
//! ```bash
//! cargo test --release -p vllm-e2e --features e2e,cuda \
//!     --test e_llava -- --ignored --test-threads=1
//! ```

#![cfg(feature = "e2e")]

use std::path::{Path, PathBuf};

use vllm_e2e::assertions::{assert_coherent_text, assert_valid_chat_response};
use vllm_e2e::{Client, TestModels, TestServer};
use vllm_serve::protocol::{ChatCompletionMessageParam, ChatCompletionRequest};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load_fixture_b64(name: &str) -> String {
    use base64::Engine;
    let path = fixture_dir().join(name);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("fixture {} must exist: {e}", path.display()));
    base64::engine::general_purpose::STANDARD.encode(&bytes)
}

fn default_chat_request() -> ChatCompletionRequest {
    serde_json::from_str(r#"{"messages": []}"#).unwrap()
}

fn user_msg(content: &str) -> ChatCompletionMessageParam {
    ChatCompletionMessageParam {
        role: "user".to_string(),
        content: Some(serde_json::Value::String(content.to_string())),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

fn user_msg_with_image(text: &str, image_base64: &str) -> ChatCompletionMessageParam {
    let content = serde_json::json!([
        {
            "type": "image_url",
            "image_url": { "url": format!("data:image/png;base64,{}", image_base64) }
        },
        { "type": "text", "text": text }
    ]);
    ChatCompletionMessageParam {
        role: "user".to_string(),
        content: Some(content),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

fn assert_mentions_any(text: &str, keywords: &[&str], ctx: &str) {
    let lower = text.to_lowercase();
    let hit = keywords.iter().any(|k| lower.contains(&k.to_lowercase()));
    assert!(
        hit,
        "{ctx}: response does not mention any of {keywords:?} — \
         vision encoder likely did not engage. text: {text:?}",
    );
}

async fn start_llava() -> (TestServer, Client) {
    let server = TestServer::builder(TestModels::LLAVA_1_5_7B_HF)
        .start()
        .await
        .expect("LLaVA-1.5-7B server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_llava_server_starts() {
    let (server, client) = start_llava().await;
    assert!(client.health().await.unwrap(), "server should be healthy");
    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
    drop(server);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_llava_text_only_chat() {
    let (_server, client) = start_llava().await;
    let request = ChatCompletionRequest {
        messages: vec![user_msg("What is 2 + 2?")],
        max_tokens: Some(32),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 1);
}

/// Image-coherence smoke. The whole CLIP + 2-layer-MLP-projector chain
/// has to land 576 spliced patches at the placeholder positions for
/// the model to mention "red" or "circle". Three Phase-H pieces all
/// have to be live at once: load-time `vision_class_embedding_fold`
/// folding `class_embedding` into `position_embedding[0]`,
/// `pack_pixels_with_cls` prepending the zero row to pixels in the
/// wrapper, and `Instruction::StripCls` dropping that row before the
/// projector runs. If any one is broken, the projector either runs on
/// the wrong row count or against drifted features and the response
/// describes "an unrelated scene".
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_llava_image_red_circle() {
    let (_server, client) = start_llava().await;
    let red = load_fixture_b64("red_circle_224.png");
    let request = ChatCompletionRequest {
        messages: vec![user_msg_with_image(
            "Describe this image. What color and shape do you see?",
            &red,
        )],
        max_tokens: Some(64),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 4);
    assert_mentions_any(text, &["red", "circle", "round"], "LLaVA image-coherence");
}

/// Two distinguishable images — guards against false KV reuse across
/// requests with different image bytes that hash-collide on token IDs.
/// Same arch-agnostic spirit as the qwen2-vl / gemma3-mm equivalents.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_llava_two_distinguishable_images() {
    let (_server, client) = start_llava().await;

    let red = load_fixture_b64("red_circle_224.png");
    let blue = load_fixture_b64("blue_square_224.png");

    let mk = |b64: &str| ChatCompletionRequest {
        messages: vec![user_msg_with_image(
            "Describe this image. What color and shape do you see?",
            b64,
        )],
        max_tokens: Some(48),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp_a = client.chat_completion(&mk(&red)).await.unwrap();
    let text_a = resp_a.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text_a, 4);
    assert_mentions_any(text_a, &["red", "circle", "round"], "image A (red circle)");

    let resp_b = client.chat_completion(&mk(&blue)).await.unwrap();
    let text_b = resp_b.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text_b, 4);
    assert_mentions_any(
        text_b,
        &["blue", "square", "rectangle"],
        "image B (blue square)",
    );

    let lower_b = text_b.to_lowercase();
    assert!(
        !(lower_b.contains("red") && lower_b.contains("circle")),
        "image B response describes the red circle ({text_b:?}) — \
         prefix cache likely reused image A's vision KV.",
    );
}
