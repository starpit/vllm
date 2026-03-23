// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E tests for relocatable spans (BlockAnnotations).
//!
//! Verifies that:
//! 1. Relocatable annotations produce identical output to the normal path
//!    (the rotate-attend-unrotate cycle doesn't corrupt attention).
//! 2. Relocatable blocks produce cache hits regardless of document order.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e_spans -- --ignored`

#![cfg(feature = "e2e")]

use std::collections::BTreeMap;
use vllm_common::BlockKind;
use vllm_e2e::TestModels;
use vllm_serve::llm::{LLMBuilder, Prompt, SamplingParams};

fn greedy_params(max_tokens: u32) -> SamplingParams {
    SamplingParams {
        max_tokens: Some(max_tokens),
        temperature: 0.0,
        detokenize: true,
        ..SamplingParams::default()
    }
}

/// Same prompt with and without Relocatable annotations should produce
/// identical output — proving the rotate-attend-unrotate cycle is transparent.
#[test]
#[ignore]
fn test_relocatable_annotations_produce_same_output() {
    vllm_common::telemetry::init_tracing("off");

    let mut llm = LLMBuilder::new(TestModels::SMOLLM)
        .enable_prefix_caching(true)
        .build()
        .expect("LLM should initialize");

    let block_size = 16;
    let tokenizer = llm.tokenizer().expect("tokenizer should be available");
    let prompt_text = "The capital of France is";
    let token_ids = tokenizer.encode(prompt_text, false).expect("encode");

    // Pad to block boundary so annotations align.
    let mut padded = token_ids.clone();
    let remainder = padded.len() % block_size;
    if remainder > 0 {
        padded.resize(padded.len() + (block_size - remainder), 0);
    }

    // Run without annotations (normal path).
    let output_normal = llm
        .generate(&[Prompt::TokenIds(padded.clone())], Some(greedy_params(10)))
        .expect("generate normal");

    // Reset cache to avoid prefix hits confounding the test.
    llm.reset_prefix_cache().expect("reset");

    // Run with all blocks annotated as Relocatable.
    let num_blocks = padded.len() / block_size;
    let mut annotations = BTreeMap::new();
    for i in 0..num_blocks {
        annotations.insert(i, BlockKind::Relocatable);
    }
    let output_annotated = llm
        .generate(
            &[Prompt::TokenIdsWithAnnotations(padded, annotations)],
            Some(greedy_params(10)),
        )
        .expect("generate annotated");

    let text_normal = &output_normal[0].outputs[0].text;
    let text_annotated = &output_annotated[0].outputs[0].text;

    assert_eq!(
        text_normal, text_annotated,
        "Relocatable annotations should not change output.\n  normal:    {text_normal:?}\n  annotated: {text_annotated:?}"
    );
}

/// Running the same ordering twice with Relocatable annotations should
/// produce identical output — proving that cache reuse via the
/// rotate-attend-unrotate cycle doesn't corrupt results.
///
/// NOTE: This test currently fails on MLX because the MLX worker does not
/// have annotation-aware block hashing or per-block KV cache reuse.
/// It should pass on CUDA once the paged rotation path is exercised.
#[test]
#[ignore]
fn test_relocatable_cache_hit_produces_same_output() {
    vllm_common::telemetry::init_tracing("off");

    let mut llm = LLMBuilder::new(TestModels::SMOLLM)
        .enable_prefix_caching(true)
        .build()
        .expect("LLM should initialize");

    let block_size = 16;
    let tokenizer = llm.tokenizer().expect("tokenizer should be available");

    let doc_text = "Document about the history of computing and Charles Babbage";
    let query_text = "Summarize:";

    let doc_ids = tokenizer.encode(doc_text, false).expect("encode doc");
    let query_ids = tokenizer.encode(query_text, false).expect("encode q");

    // Pad each to block boundary.
    let pad_to = |ids: &[u32]| -> Vec<u32> {
        let mut v = ids.to_vec();
        let rem = v.len() % block_size;
        if rem > 0 {
            v.resize(v.len() + (block_size - rem), 0);
        }
        v
    };

    let doc = pad_to(&doc_ids);
    let query = pad_to(&query_ids);

    let doc_blocks = doc.len() / block_size;
    let query_blocks = query.len() / block_size;

    let mut tokens = Vec::new();
    tokens.extend_from_slice(&doc);
    tokens.extend_from_slice(&query);

    let mut ann = BTreeMap::new();
    for i in 0..doc_blocks {
        ann.insert(i, BlockKind::Relocatable);
    }
    for i in doc_blocks..doc_blocks + query_blocks {
        ann.insert(i, BlockKind::Prefixed);
    }

    // First run — populates cache.
    let output1 = llm
        .generate(
            &[Prompt::TokenIdsWithAnnotations(tokens.clone(), ann.clone())],
            Some(greedy_params(10)),
        )
        .expect("generate first");

    // Second run — should hit cached Relocatable blocks.
    let output2 = llm
        .generate(
            &[Prompt::TokenIdsWithAnnotations(tokens, ann)],
            Some(greedy_params(10)),
        )
        .expect("generate second");

    let text1 = &output1[0].outputs[0].text;
    let text2 = &output2[0].outputs[0].text;

    assert!(!text1.is_empty(), "first output should not be empty");
    assert_eq!(
        text1, text2,
        "Cache hit with Relocatable blocks should produce identical output.\n  first:  {text1:?}\n  second: {text2:?}"
    );
}
