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
fn scheduler_merged_waves_below_subgraph_count() {
    // Wavefront scheduling collapses mutually-independent subgraphs
    // into shared waves — qkv gemms all read `normed` and are
    // concurrent, so a wavefront schedule puts them in one wave.
    // Wave count must therefore be strictly below subgraph count.
    // (Bind to locals so clippy doesn't flag the const comparison.)
    let waves: usize = llama::llama_3_2_1b::m_1::NUM_WAVES;
    let subgraphs: usize = llama::llama_3_2_1b::m_1::NUM_SUBGRAPHS;
    assert!(
        waves < subgraphs,
        "waves ({waves}) should be fewer than subgraphs ({subgraphs}) — merging broken",
    );
}

// ── Model struct emission (task #6) ─────────────────────────────────
//
// The emitted `Model` / `Layer` / `Model::load` types reference
// ferrite-kernels runtime types that require cuda. Guard these
// tests on the `cuda` feature so a cudaless build still builds the
// crate (just without the Model type). We can't construct a Model
// without real GPU weights, but we can observe its static shape:
// field presence, types, NUM_LAYERS constant, load signature.

#[cfg(feature = "cuda")]
#[test]
fn model_struct_has_per_layer_weights() {
    // Layer struct should expose every per-layer weight used in the
    // DSL body with the right runtime type. Checking the presence
    // (via field access in a function that's never called) forces
    // the compiler to resolve the types.
    #[allow(dead_code)]
    fn _type_check(layer: &llama::llama_3_2_1b::Layer) {
        let _: &::ferrite_kernels::layers::RmsNorm = &layer.input_layernorm;
        let _: &::ferrite_kernels::layers::Linear = &layer.self_attn_q_proj;
        let _: &::ferrite_kernels::layers::Linear = &layer.self_attn_k_proj;
        let _: &::ferrite_kernels::layers::Linear = &layer.self_attn_v_proj;
        let _: &::ferrite_kernels::layers::Linear = &layer.self_attn_o_proj;
        let _: &::ferrite_kernels::layers::RmsNorm = &layer.post_attention_layernorm;
        let _: &::ferrite_kernels::layers::Linear = &layer.mlp_gate_proj;
        let _: &::ferrite_kernels::layers::Linear = &layer.mlp_up_proj;
        let _: &::ferrite_kernels::layers::Linear = &layer.mlp_down_proj;
    }
}

#[cfg(feature = "cuda")]
#[test]
fn model_struct_has_global_weights() {
    #[allow(dead_code)]
    fn _type_check(model: &llama::llama_3_2_1b::Model) {
        let _: &Vec<llama::llama_3_2_1b::Layer> = &model.layers;
        let _: &::ferrite_kernels::layers::Embedding = &model.embed_tokens;
        let _: &::ferrite_kernels::layers::RmsNorm = &model.norm;
        let _: &::ferrite_kernels::layers::Linear = &model.lm_head;
    }
}

#[cfg(feature = "cuda")]
#[test]
fn model_num_layers_baked_in_from_config() {
    // llama-3.2-1b has 16 layers; llama-3.1-8b has 32.
    assert_eq!(llama::llama_3_2_1b::Model::NUM_LAYERS, 16);
    assert_eq!(llama::llama_3_1_8b::Model::NUM_LAYERS, 32);
    assert_eq!(llama::llama_3_1_70b::Model::NUM_LAYERS, 80);
}

#[cfg(feature = "cuda")]
#[test]
fn model_load_has_expected_signature() {
    // Confirm the load fn compiles with the documented signature.
    // Never called — we just need the type check.
    #[allow(dead_code)]
    fn _sig_check() -> for<'a> unsafe fn(
        &'a mut ::ferrite_cuda_core::weights::GpuWeights,
        f32,
    ) -> ::anyhow::Result<llama::llama_3_2_1b::Model> {
        llama::llama_3_2_1b::Model::load
    }
}
