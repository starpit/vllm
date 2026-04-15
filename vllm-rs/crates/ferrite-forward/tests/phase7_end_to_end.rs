// SPDX-License-Identifier: Apache-2.0
//! End-to-end integration test — the ONE test that proves the
//! macro actually drives the full pipeline.
//!
//! The whole pipeline (parse → classify → shape-infer → CFG → unroll
//! → solve × workloads → schedule × workloads) runs at compile time.
//! If any pass fails for any (model × workload) combo, this file
//! fails to compile.
//!
//! The emitted `llama::*` modules carry real values from the
//! solver + scheduler. We assert on them to prove the pipeline's
//! output isn't dummy.

use ferrite_forward::forward;

// The real Llama body — SwiGLU MLP (`silu(gate) * up`), exercising
// the Expr::Mul path that used to emit OpKind::Add by mistake.
#[forward(
    models_dir = "../../../model_architectures/llama",
    target = "../../../target_profiles/l4_sm89.json",
    workloads = [1, 8, 64, 512, 4096],
)]
fn llama() {
    hidden_states = embed(input_ids, embed_tokens);
    for layer in 0..num_hidden_layers {
        normed = rmsnorm(hidden_states, input_layernorm[layer]);
        q = gemm(normed, self_attn.q_proj[layer]);
        k = gemm(normed, self_attn.k_proj[layer]);
        v = gemm(normed, self_attn.v_proj[layer]);
        (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
        attn = attention(q, k, v, kv_cache[layer], block_table);
        oproj = gemm(attn, self_attn.o_proj[layer]);
        hidden_states = add(oproj, hidden_states);

        normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
        gate = silu(gemm(normed2, mlp.gate_proj[layer]));
        up = gemm(normed2, mlp.up_proj[layer]);
        down = gemm(gate * up, mlp.down_proj[layer]);
        hidden_states = add(down, hidden_states);
    }
    normed = rmsnorm(hidden_states, norm);
    logits = gemm(normed, lm_head);
}

#[test]
fn llama_3_2_1b_tile_count_matches_expected() {
    // Llama-3.2-1B has 16 layers. Per-layer SwiGLU body is 15
    // tiles: rmsnorm, qgemm, kgemm, vgemm, rope, attn, ogemm, add,
    // rmsnorm2, gate_gemm, silu, up_gemm, mul, down_gemm, add2.
    // Pre: embed (1). Post: rmsnorm + lm_head (2).
    // Total: 1 + 16 × 15 + 2 = 243 tiles.
    assert_eq!(llama::llama_3_2_1b::NUM_TILES, 243);
}

#[test]
fn llama_3_1_8b_tile_count_matches_expected() {
    // 32 layers × 15 + 3 = 483 tiles.
    assert_eq!(llama::llama_3_1_8b::NUM_TILES, 483);
}

#[test]
fn every_llama_config_was_compiled() {
    // The nine Llama configs. If any didn't go through the
    // pipeline this file wouldn't compile.
    let _: usize = llama::llama_2_7b::NUM_TILES;
    let _: usize = llama::llama_2_13b::NUM_TILES;
    let _: usize = llama::llama_2_70b::NUM_TILES;
    let _: usize = llama::llama_3_8b::NUM_TILES;
    let _: usize = llama::llama_3_70b::NUM_TILES;
    let _: usize = llama::llama_3_1_8b::NUM_TILES;
    let _: usize = llama::llama_3_1_70b::NUM_TILES;
    let _: usize = llama::llama_3_2_1b::NUM_TILES;
    let _: usize = llama::llama_3_2_3b::NUM_TILES;
}

#[test]
fn every_workload_point_was_solved_for_llama_3_2_1b() {
    let _: f64 = llama::llama_3_2_1b::m_1::PREDICTED_US;
    let _: f64 = llama::llama_3_2_1b::m_8::PREDICTED_US;
    let _: f64 = llama::llama_3_2_1b::m_64::PREDICTED_US;
    let _: f64 = llama::llama_3_2_1b::m_512::PREDICTED_US;
    let _: f64 = llama::llama_3_2_1b::m_4096::PREDICTED_US;
}

#[test]
fn solver_produced_finite_positive_cost_across_workloads() {
    // For the starter library (single-tile claims), subgraph count
    // equals tile count. And predicted cost must be finite > 0.
    let points: [f64; 5] = [
        llama::llama_3_2_1b::m_1::PREDICTED_US,
        llama::llama_3_2_1b::m_8::PREDICTED_US,
        llama::llama_3_2_1b::m_64::PREDICTED_US,
        llama::llama_3_2_1b::m_512::PREDICTED_US,
        llama::llama_3_2_1b::m_4096::PREDICTED_US,
    ];
    for us in points {
        assert!(us > 0.0 && us.is_finite(), "bogus predicted_us: {us}");
    }
    // Prefill dwarfs decode (regression check for the silent-
    // gemm-cost-drop that used to bite). Bind to locals so
    // clippy::assertions_on_constants doesn't flag the compile-
    // time comparison.
    let decode: f64 = llama::llama_3_2_1b::m_1::PREDICTED_US;
    let prefill: f64 = llama::llama_3_2_1b::m_4096::PREDICTED_US;
    assert!(prefill > decode * 10.0);
}

#[test]
fn arch_level_weights_enum_and_dispatch_exist() {
    // Observe the compiler emitted the arch-level `Weights` enum
    // with one variant per compiled model, and the dispatching
    // `forward` fn. Under #[cfg(feature = "cuda")] only — these
    // reference cuda-only types (GpuWeights, CUstream, GpuDevice).
    #[cfg(feature = "cuda")]
    {
        // Type-level observation: an enum variant for the 1B model
        // exists and wraps that model's `Weights`. If the dispatcher
        // wasn't emitted this line wouldn't type-check.
        fn _probe(w: llama::llama_3_2_1b::Weights) -> llama::Weights {
            llama::Weights::Llama_3_2_1b(w)
        }
        let _: fn(llama::llama_3_2_1b::Weights) -> llama::Weights = _probe;
    }
}

#[test]
fn scheduler_produces_linear_chain_on_fused_body() {
    // The real Llama body post-fusion is a serial chain: every
    // subgraph depends on the previous one. Q/K/V gemms that used to
    // be parallel are now claimed by `FusedQkvRopeCacheImpl` as a
    // single subgraph (their parallelism is consumed internally).
    // Gate/up gemms that used to be parallel are likewise inside the
    // `FusedGateUpSiluMulImpl` claim. Result: waves == subgraphs.
    //
    // If a future fusion leaves genuinely independent subgraphs, this
    // test loosens — but for the current impl library the serial
    // chain is the correct expectation.
    let waves: usize = llama::llama_3_2_1b::m_1::NUM_WAVES;
    let subgraphs: usize = llama::llama_3_2_1b::m_1::NUM_SUBGRAPHS;
    assert_eq!(
        waves, subgraphs,
        "post-fusion Llama body is a serial chain; waves must match subgraphs \
         (waves={waves}, subgraphs={subgraphs})",
    );
}
