// SPDX-License-Identifier: Apache-2.0
//! CUDA source generation for the throughput TK megakernel.
//!
//! Renders a per-arch `.cu` source that:
//! - overrides `LLAMA_*` macros from `cross-gpu-llama/llama.cuh` for the
//!   target arch's dims (num_layers, hidden_dim, head_dim, GQA factor, …);
//! - includes the 13 throughput op .cu files (attention_decode,
//!   attention_prefill, qkv_rope_append, gate_silu, up_matmul,
//!   matmul_adds, lm_head, batched_rms_norm, inc_barriers,
//!   all_device_barrier);
//! - instantiates `mk<llama_config, llama_70b_globals, op1, …, op13>`
//!   (the KvmMega framework's persistent template entrypoint);
//! - emits an `extern "C" int tk_megakernel_<model>_launch(...)`
//!   wrapper taking flat C-friendly args (raw pointers + scalars +
//!   shape ints) and constructing the `globals_t` struct.
//!
//! Ported from `worktree-ferrite-mega@417e16bda`, with the older
//! branch's `generate_megakernel` / `DevicePhase`-based DAG codegen
//! stripped — that depended on data structures (`DevicePhase`,
//! `GeneratedMegakernel`) that don't exist on this branch's
//! Instruction-tape IR. Only the TK-specific generator survives;
//! the encoder side that drives this lives elsewhere
//! (`interpreters/`) and lands in a follow-up commit.

#![allow(dead_code)]

use std::fmt::Write;

/// Model-specific dimensions for the TK throughput megakernel. These
/// are the inputs to `generate_tk_megakernel`; values must agree with
/// the loaded weights' shapes and the cross-gpu-llama op tile
/// constraints (e.g. `num_attention_heads / num_kv_heads <= 16` for
/// the GQA tile, `intermediate_dim % matmul_out_block_size == 0` for
/// the gate/up/down matmul fan-out).
pub struct TkModelDims {
    pub num_layers: u32,
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub head_dim: u32,
    pub num_attention_heads: u32,
    pub num_kv_heads: u32,
    pub kv_page_size: u32,
    pub prefill_kv_block_size: u32,
    pub decode_kv_block_size: u32,
    pub matmul_out_block_size: u32,
    pub matmul_batch_block_size: u32,
    pub vocab_size: u32,
    pub sm_count: u32,
    pub num_devices: u32,
}

/// Generated TK megakernel CUDA source + metadata.
#[derive(Clone, Debug)]
pub struct GeneratedTkMegakernel {
    pub cuda_source: String,
    pub launch_fn_name: String,
    /// `(c_type, param_name)` for the extern "C" launch wrapper.
    pub flat_params: Vec<(String, String)>,
}

pub fn generate_tk_megakernel(model_name: &str, dims: &TkModelDims) -> GeneratedTkMegakernel {
    let launch_fn_name = format!("tk_megakernel_{model_name}_launch");
    let d = dims;

    let mut src = String::new();

    // ── Header ──
    writeln!(
        src,
        "// Auto-generated TK megakernel (throughput) for {model_name}"
    )
    .unwrap();
    writeln!(
        src,
        "// DO NOT EDIT — regenerate via the forward! proc macro."
    )
    .unwrap();
    writeln!(src).unwrap();

    // ── Override model dimension #defines BEFORE including llama.cuh ──
    // llama.cuh defines defaults (70B); we override for our model.
    writeln!(src, "#define LLAMA_NUM_LAYERS {}", d.num_layers).unwrap();
    writeln!(src, "#define LLAMA_HIDDEN_DIM {}", d.hidden_dim).unwrap();
    writeln!(src, "#define LLAMA_INTERMEDIATE_DIM {}", d.intermediate_dim).unwrap();
    writeln!(src, "#define LLAMA_HEAD_DIM {}", d.head_dim).unwrap();
    writeln!(
        src,
        "#define LLAMA_NUM_ATTENTION_HEADS {}",
        d.num_attention_heads
    )
    .unwrap();
    writeln!(src, "#define LLAMA_NUM_KV_HEADS {}", d.num_kv_heads).unwrap();
    writeln!(src, "#define LLAMA_KV_PAGE_SIZE {}", d.kv_page_size).unwrap();
    writeln!(
        src,
        "#define LLAMA_PREFILL_KV_BLOCK_SIZE {}",
        d.prefill_kv_block_size
    )
    .unwrap();
    writeln!(
        src,
        "#define LLAMA_DECODE_KV_BLOCK_SIZE {}",
        d.decode_kv_block_size
    )
    .unwrap();
    writeln!(
        src,
        "#define LLAMA_MATMUL_OUT_BLOCK_SIZE {}",
        d.matmul_out_block_size
    )
    .unwrap();
    writeln!(
        src,
        "#define LLAMA_MATMUL_BATCH_BLOCK_SIZE {}",
        d.matmul_batch_block_size
    )
    .unwrap();
    writeln!(src, "#define SM_COUNT {}", d.sm_count).unwrap();
    writeln!(src, "#define LLAMA_NUM_DEVICES {}", d.num_devices).unwrap();
    writeln!(src).unwrap();

    // ── Precompiled header deps (kittens + megakernel must precede all ops) ──
    writeln!(src, "#include \"kittens.cuh\"").unwrap();
    writeln!(src, "#include \"megakernel.cuh\"").unwrap();
    writeln!(src).unwrap();

    // ── Op includes (order mirrors llama.cu from the vendored cross-gpu-llama) ──
    writeln!(src, "#include \"attention_decode.cu\"").unwrap();
    writeln!(src, "#include \"attention_prefill.cu\"").unwrap();
    writeln!(src, "#include \"batched_rms_norm.cu\"").unwrap();
    writeln!(src, "#include \"gate_silu.cu\"").unwrap();
    writeln!(src, "#include \"inc_barriers.cu\"").unwrap();
    writeln!(src, "#include \"llama.cuh\"").unwrap();
    writeln!(src, "#include \"lm_head.cu\"").unwrap();
    writeln!(src, "#include \"matmul_adds.cu\"").unwrap();
    writeln!(src, "#include \"qkv_rope_append.cu\"").unwrap();
    writeln!(src, "#include \"up_matmul.cu\"").unwrap();
    writeln!(src, "#include \"all_device_barrier.cu\"").unwrap();
    writeln!(src).unwrap();

    // ── Namespaces and op aliases ──
    writeln!(src, "using namespace kittens;").unwrap();
    writeln!(src, "using namespace megakernel;").unwrap();
    writeln!(src).unwrap();

    writeln!(src, "struct ops {{").unwrap();
    writeln!(
        src,
        "    using attn_norm_op = attn_norm<llama_config, llama_70b_globals>;"
    )
    .unwrap();
    writeln!(
        src,
        "    using qkv_rope_append_op = qkv_rope_append<llama_config, llama_70b_globals>;"
    )
    .unwrap();
    writeln!(
        src,
        "    using attention_prefill_op = attention_prefill<llama_config, llama_70b_globals>;"
    )
    .unwrap();
    writeln!(
        src,
        "    using attention_decode_op = attention_decode<llama_config, llama_70b_globals>;"
    )
    .unwrap();
    writeln!(
        src,
        "    using o_proj_op = o_proj<llama_config, llama_70b_globals>;"
    )
    .unwrap();
    writeln!(
        src,
        "    using mlp_norm_op = mlp_norm<llama_config, llama_70b_globals>;"
    )
    .unwrap();
    writeln!(
        src,
        "    using gate_silu_op = gate_silu<llama_config, llama_70b_globals>;"
    )
    .unwrap();
    writeln!(
        src,
        "    using up_matmul_op = up_matmul<llama_config, llama_70b_globals>;"
    )
    .unwrap();
    writeln!(
        src,
        "    using downproj_op = downproj<llama_config, llama_70b_globals>;"
    )
    .unwrap();
    writeln!(
        src,
        "    using lm_head_norm_op = lm_head_norm<llama_config, llama_70b_globals>;"
    )
    .unwrap();
    writeln!(
        src,
        "    using lm_head_op = lm_head<llama_config, llama_70b_globals>;"
    )
    .unwrap();
    writeln!(
        src,
        "    using barrier_inc_op = barrier_inc<llama_config, llama_70b_globals>;"
    )
    .unwrap();
    writeln!(
        src,
        "    using all_device_barrier_op = all_device_barrier<llama_config, llama_70b_globals>;"
    )
    .unwrap();
    writeln!(src, "}};").unwrap();
    writeln!(src).unwrap();

    // ── extern "C" launch wrapper ──
    let flat_params = build_tk_flat_params();

    writeln!(src, "extern \"C\" int {launch_fn_name}(").unwrap();
    for (c_type, name) in &flat_params {
        writeln!(src, "    {c_type} {name},").unwrap();
    }
    writeln!(src, "    uint64_t __stream)").unwrap();
    writeln!(src, "{{").unwrap();
    writeln!(src, "    using G = llama_70b_globals;").unwrap();
    writeln!(src).unwrap();

    if dims.num_devices > 1 {
        // Multi-GPU: construct pgl types with device arrays
        writeln!(src, "    int dev_ids[{}] = {{}};", dims.num_devices).unwrap();
        writeln!(src, "    // TODO: fill dev_ids for multi-GPU").unwrap();
        writeln!(src).unwrap();

        // Barriers
        writeln!(
            src,
            "    uint* bar_ptrs[{0}]; bar_ptrs[0] = (uint*)bar_ptr;",
            dims.num_devices
        )
        .unwrap();
        writeln!(src, "    typename G::barriers Bar_(dev_ids, bar_ptrs, (size_t)bar_b, (size_t)bar_d, (size_t)bar_r, (size_t)bar_c);").unwrap();
    } else {
        // Single-GPU: gl_as_pgl inherits gl constructor — direct ptr + args
        writeln!(src, "    typename G::barriers Bar_(").unwrap();
        writeln!(
            src,
            "        (uint*)bar_ptr, (size_t)bar_b, (size_t)bar_d, (size_t)bar_r, (size_t)bar_c);"
        )
        .unwrap();
    }
    writeln!(src).unwrap();

    // Instructions/timings: plain gl (not pgl)
    // instruction_layout = gl<int, 1, 1, -1, 32> (ENABLE_GLOBAL_WORK_QUEUE=true)
    // b=1(nullptr), d=1(nullptr), r=-1(size_t), c=32(nullptr)
    writeln!(src, "    typename G::instruction_layout instructions_(").unwrap();
    writeln!(
        src,
        "        (int*)instructions_ptr, nullptr, nullptr, (size_t)total_instructions, nullptr);"
    )
    .unwrap();
    writeln!(src, "    typename G::timing_layout timings_(").unwrap();
    writeln!(
        src,
        "        (int*)timings_ptr, nullptr, nullptr, (size_t)total_instructions, nullptr);"
    )
    .unwrap();
    writeln!(src).unwrap();

    // global_instruction_index: gl<int, 1,1,1,1> — all compile-time
    writeln!(src, "    gl<int, 1, 1, 1, 1> global_instruction_index_((int*)global_inst_idx_ptr, nullptr, nullptr, nullptr, nullptr);").unwrap();
    writeln!(src).unwrap();

    // ── Weights: gl<bf16, 1, -1, -1, hidden_dim, st_bf<256,64>> ──
    writeln!(src, "    typename G::weights_t qkv_w_(").unwrap();
    writeln!(src, "        (__nv_bfloat16*)qkv_weights_ptr, nullptr, (size_t)qkv_weights_depth, (size_t)qkv_weights_rows, nullptr);").unwrap();
    writeln!(src, "    typename G::norm_weights_t attn_norm_w_(").unwrap();
    writeln!(src, "        (__nv_bfloat16*)attn_norm_weights_ptr, nullptr, nullptr, (size_t)attn_norm_weights_rows, nullptr);").unwrap();
    writeln!(src, "    typename G::weights_t o_w_(").unwrap();
    writeln!(src, "        (__nv_bfloat16*)o_weights_ptr, nullptr, (size_t)o_weights_depth, (size_t)o_weights_rows, nullptr);").unwrap();
    writeln!(src, "    typename G::norm_weights_t mlp_norm_w_(").unwrap();
    writeln!(src, "        (__nv_bfloat16*)mlp_norm_weights_ptr, nullptr, nullptr, (size_t)mlp_norm_weights_rows, nullptr);").unwrap();
    writeln!(src, "    typename G::weights_t up_w_(").unwrap();
    writeln!(src, "        (__nv_bfloat16*)up_weights_ptr, nullptr, (size_t)up_weights_depth, (size_t)up_weights_rows, nullptr);").unwrap();
    writeln!(src, "    typename G::weights_t gate_w_(").unwrap();
    writeln!(src, "        (__nv_bfloat16*)gate_weights_ptr, nullptr, (size_t)gate_weights_depth, (size_t)gate_weights_rows, nullptr);").unwrap();
    writeln!(src, "    typename G::weights_big_indim_t down_w_(").unwrap();
    writeln!(src, "        (__nv_bfloat16*)down_weights_ptr, nullptr, (size_t)down_weights_depth, (size_t)down_weights_rows, nullptr);").unwrap();
    writeln!(src, "    typename G::norm_weights_t lm_head_norm_w_(").unwrap();
    writeln!(src, "        (__nv_bfloat16*)lm_head_norm_weights_ptr, nullptr, nullptr, (size_t)lm_head_norm_weights_rows, nullptr);").unwrap();
    writeln!(src, "    typename G::weights_t lm_head_w_(").unwrap();
    writeln!(src, "        (__nv_bfloat16*)lm_head_weights_ptr, nullptr, (size_t)lm_head_weights_depth, (size_t)lm_head_weights_rows, nullptr);").unwrap();
    writeln!(src).unwrap();

    // ── KV cache: gl with dual TMA descriptors ──
    writeln!(src, "    typename G::kv_cache_t k_cache_(").unwrap();
    writeln!(src, "        (__nv_bfloat16*)k_cache_ptr, (size_t)k_cache_batch, (size_t)k_cache_depth, nullptr, nullptr);").unwrap();
    writeln!(src, "    typename G::kv_cache_t v_cache_(").unwrap();
    writeln!(src, "        (__nv_bfloat16*)v_cache_ptr, (size_t)v_cache_batch, (size_t)v_cache_depth, nullptr, nullptr);").unwrap();
    writeln!(src).unwrap();

    // ── RoPE: gl<float, 1, 1, -1, head_dim> ──
    writeln!(src, "    typename G::rope_table_t rope_cos_(").unwrap();
    writeln!(
        src,
        "        (float*)rope_cos_ptr, nullptr, nullptr, (size_t)rope_rows, nullptr);"
    )
    .unwrap();
    writeln!(src, "    typename G::rope_table_t rope_sin_(").unwrap();
    writeln!(
        src,
        "        (float*)rope_sin_ptr, nullptr, nullptr, (size_t)rope_rows, nullptr);"
    )
    .unwrap();
    writeln!(src).unwrap();

    // ── Activations ──
    if dims.num_devices > 1 {
        // Multi-GPU: pgl constructor (dev_ids, ptrs, args...)
        writeln!(
            src,
            "    __nv_bfloat16* hs_ptrs[{0}]; hs_ptrs[0] = (__nv_bfloat16*)hidden_states_ptr;",
            dims.num_devices
        )
        .unwrap();
        writeln!(src, "    typename G::activations_parallel_t hidden_states_(dev_ids, hs_ptrs, nullptr, nullptr, (size_t)batch_size, nullptr);").unwrap();
        writeln!(src).unwrap();
        writeln!(src, "    __nv_bfloat16* rms_rope_ptrs[{0}]; rms_rope_ptrs[0] = (__nv_bfloat16*)rms_rope_intermediates_ptr;", dims.num_devices).unwrap();
        writeln!(src, "    typename G::activations_parallel_mc_t rms_rope_intermediates_(dev_ids, rms_rope_ptrs, nullptr, nullptr, (size_t)batch_size, nullptr);").unwrap();
        writeln!(src).unwrap();
        writeln!(src, "    __nv_bfloat16* rms_gate_ptrs[{0}]; rms_gate_ptrs[0] = (__nv_bfloat16*)rms_gate_intermediates_ptr;", dims.num_devices).unwrap();
        writeln!(src, "    typename G::activations_parallel_mc_t rms_gate_intermediates_(dev_ids, rms_gate_ptrs, nullptr, nullptr, (size_t)batch_size, nullptr);").unwrap();
    } else {
        // Single-GPU: gl_as_pgl inherits gl constructor (ptr, args...)
        writeln!(
            src,
            "    typename G::activations_parallel_t hidden_states_("
        )
        .unwrap();
        writeln!(src, "        (__nv_bfloat16*)hidden_states_ptr, nullptr, nullptr, (size_t)batch_size, nullptr);").unwrap();
        writeln!(src).unwrap();
        writeln!(
            src,
            "    typename G::activations_parallel_mc_t rms_rope_intermediates_("
        )
        .unwrap();
        writeln!(src, "        (__nv_bfloat16*)rms_rope_intermediates_ptr, nullptr, nullptr, (size_t)batch_size, nullptr);").unwrap();
        writeln!(src).unwrap();
        writeln!(
            src,
            "    typename G::activations_parallel_mc_t rms_gate_intermediates_("
        )
        .unwrap();
        writeln!(src, "        (__nv_bfloat16*)rms_gate_intermediates_ptr, nullptr, nullptr, (size_t)batch_size, nullptr);").unwrap();
    }
    writeln!(src).unwrap();

    // q_post_rope: gl<bf16, 1,1,-1,-1>
    writeln!(src, "    typename G::activations_t q_post_rope_(").unwrap();
    writeln!(src, "        (__nv_bfloat16*)q_post_rope_ptr, nullptr, nullptr, (size_t)batch_size, (size_t)q_post_rope_cols);").unwrap();
    writeln!(src).unwrap();

    // attn_out: parallel type
    if dims.num_devices > 1 {
        writeln!(src, "    __nv_bfloat16* attn_out_ptrs[{0}]; attn_out_ptrs[0] = (__nv_bfloat16*)attn_out_ptr;", dims.num_devices).unwrap();
        writeln!(src, "    typename G::activations_parallel_t attn_out_(dev_ids, attn_out_ptrs, nullptr, nullptr, (size_t)batch_size, nullptr);").unwrap();
    } else {
        writeln!(src, "    typename G::activations_parallel_t attn_out_(").unwrap();
        writeln!(
            src,
            "        (__nv_bfloat16*)attn_out_ptr, nullptr, nullptr, (size_t)batch_size, nullptr);"
        )
        .unwrap();
    }
    writeln!(src).unwrap();

    // silu_out: gl<bf16, 1,1,-1, intermediate_dim/num_devices>
    writeln!(src, "    typename G::activations_big_indim_t silu_out_(").unwrap();
    writeln!(
        src,
        "        (__nv_bfloat16*)silu_out_ptr, nullptr, nullptr, (size_t)batch_size, nullptr);"
    )
    .unwrap();
    writeln!(src).unwrap();

    // rms_lm_head_intermediates
    writeln!(src, "#ifdef LLAMA_BROADCAST_LM_HEAD_NORM").unwrap();
    if dims.num_devices > 1 {
        writeln!(src, "    __nv_bfloat16* rms_lm_ptrs[{0}]; rms_lm_ptrs[0] = (__nv_bfloat16*)rms_lm_head_intermediates_ptr;", dims.num_devices).unwrap();
        writeln!(src, "    typename G::activations_parallel_mc_t rms_lm_head_intermediates_(dev_ids, rms_lm_ptrs, nullptr, nullptr, (size_t)batch_size, nullptr);").unwrap();
    } else {
        writeln!(
            src,
            "    typename G::activations_parallel_mc_t rms_lm_head_intermediates_("
        )
        .unwrap();
        writeln!(src, "        (__nv_bfloat16*)rms_lm_head_intermediates_ptr, nullptr, nullptr, (size_t)batch_size, nullptr);").unwrap();
    }
    writeln!(src, "#else").unwrap();
    writeln!(
        src,
        "    typename G::activations_t rms_lm_head_intermediates_("
    )
    .unwrap();
    writeln!(src, "        (__nv_bfloat16*)rms_lm_head_intermediates_ptr, nullptr, nullptr, (size_t)batch_size, nullptr);").unwrap();
    writeln!(src, "#endif").unwrap();
    writeln!(src).unwrap();

    // logits: gl<bf16, 1,1,-1,-1>
    writeln!(src, "    typename G::logits_t logits_(").unwrap();
    writeln!(src, "        (__nv_bfloat16*)logits_ptr, nullptr, nullptr, (size_t)batch_size, (size_t)logits_cols);").unwrap();
    writeln!(src).unwrap();

    // ── int32 vectors ──
    writeln!(src, "    typename G::int32_vector_t position_ids_((int*)position_ids_ptr, nullptr, nullptr, nullptr, (size_t)batch_size);").unwrap();
    writeln!(src, "    typename G::int32_vector_t kv_append_indices_((int*)kv_append_indices_ptr, nullptr, nullptr, nullptr, (size_t)batch_size);").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    typename G::int32_vector_t prefill_qo_indptr_((int*)prefill_qo_indptr_ptr, nullptr, nullptr, nullptr, (size_t)prefill_qo_indptr_len);").unwrap();
    writeln!(src, "    typename G::int32_vector_t prefill_kv_indptr_((int*)prefill_kv_indptr_ptr, nullptr, nullptr, nullptr, (size_t)prefill_kv_indptr_len);").unwrap();
    writeln!(src, "    typename G::int32_vector_t prefill_kv_indices_((int*)prefill_kv_indices_ptr, nullptr, nullptr, nullptr, (size_t)prefill_kv_indices_len);").unwrap();
    writeln!(src, "    typename G::int32_vector_t prefill_kv_last_page_len_((int*)prefill_kv_last_page_len_ptr, nullptr, nullptr, nullptr, (size_t)prefill_kv_last_page_len_len);").unwrap();
    writeln!(src, "    typename G::int32_vector_t decode_kv_indptr_((int*)decode_kv_indptr_ptr, nullptr, nullptr, nullptr, (size_t)decode_kv_indptr_len);").unwrap();
    writeln!(src, "    typename G::int32_vector_t decode_kv_indices_((int*)decode_kv_indices_ptr, nullptr, nullptr, nullptr, (size_t)decode_kv_indices_len);").unwrap();
    writeln!(src, "    typename G::int32_vector_t decode_kv_last_page_len_((int*)decode_kv_last_page_len_ptr, nullptr, nullptr, nullptr, (size_t)decode_kv_last_page_len_len);").unwrap();
    writeln!(src).unwrap();

    // ── Aggregate globals_t (field order must match llama.cuh) ──
    writeln!(src, "    G g {{").unwrap();
    writeln!(
        src,
        "        Bar_, instructions_, timings_, global_instruction_index_,"
    )
    .unwrap();
    writeln!(src, "        qkv_w_, attn_norm_w_, o_w_, mlp_norm_w_,").unwrap();
    writeln!(src, "        up_w_, gate_w_, down_w_,").unwrap();
    writeln!(src, "        lm_head_norm_w_, lm_head_w_,").unwrap();
    writeln!(src, "        k_cache_, v_cache_,").unwrap();
    writeln!(src, "        rope_cos_, rope_sin_,").unwrap();
    writeln!(
        src,
        "        hidden_states_, rms_rope_intermediates_, rms_gate_intermediates_,"
    )
    .unwrap();
    writeln!(src, "        q_post_rope_, attn_out_, silu_out_,").unwrap();
    writeln!(src, "        rms_lm_head_intermediates_, logits_,").unwrap();
    writeln!(src, "        position_ids_, kv_append_indices_,").unwrap();
    writeln!(src, "        prefill_qo_indptr_, prefill_kv_indptr_, prefill_kv_indices_, prefill_kv_last_page_len_,").unwrap();
    writeln!(
        src,
        "        decode_kv_indptr_, decode_kv_indices_, decode_kv_last_page_len_,"
    )
    .unwrap();
    writeln!(
        src,
        "        attn_scale, rms_norm_eps, num_pages, batch_size, num_prefill_tokens,"
    )
    .unwrap();
    writeln!(src, "        0  // dev_idx = 0 for single-GPU").unwrap();
    writeln!(src, "    }};").unwrap();
    writeln!(src).unwrap();

    // ── Launch ──
    writeln!(src, "    dim3 grid = g.grid();").unwrap();
    writeln!(src, "    dim3 block = g.block();").unwrap();
    writeln!(src, "    int smem = g.dynamic_shared_memory();").unwrap();
    writeln!(src, "    cudaStream_t stream = (cudaStream_t)__stream;").unwrap();
    writeln!(src).unwrap();
    // Diagnostic: print key values to verify struct construction (stderr for immediate flush)
    writeln!(src, "    fprintf(stderr, \"TK launch: sizeof(G)=%zu grid=%d block=%d smem=%d\\n\", sizeof(G), grid.x, block.x, smem);").unwrap();
    writeln!(
        src,
        "    fprintf(stderr, \"  instructions: ptr=%p rows=%d batch_size=%d num_prefill=%d\\n\","
    )
    .unwrap();
    writeln!(src, "           (void*)g.instructions.raw_ptr, (int)g.instructions.rows(), g.batch_size, g.num_prefill_tokens);").unwrap();
    writeln!(src, "    fprintf(stderr, \"  global_inst_idx: ptr=%p\\n\", (void*)g.global_instruction_index.raw_ptr);").unwrap();
    writeln!(
        src,
        "    fprintf(stderr, \"  hidden_states: ptr=%p\\n\", (void*)g.hidden_states.raw_ptr);"
    )
    .unwrap();
    writeln!(src, "    fprintf(stderr, \"  Bar: ptr=%p batch=%d depth=%d rows=%d cols=%d\\n\", (void*)g.Bar.raw_ptr, (int)g.Bar.batch(), (int)g.Bar.depth(), (int)g.Bar.rows(), (int)g.Bar.cols());").unwrap();
    writeln!(src, "    fprintf(stderr, \"  k_cache: ptr=%p batch=%d depth=%d\\n\", (void*)g.k_cache.raw_ptr, (int)g.k_cache.batch(), (int)g.k_cache.depth());").unwrap();
    writeln!(src, "    fprintf(stderr, \"  attn_scale=%f rms_eps=%f num_pages=%d dev_idx=%d\\n\", g.attn_scale, g.rms_norm_eps, g.num_pages, g.dev_idx);").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    auto kernel = mk<llama_config, llama_70b_globals,").unwrap();
    writeln!(
        src,
        "        ops::attn_norm_op, ops::qkv_rope_append_op, ops::attention_decode_op,"
    )
    .unwrap();
    writeln!(
        src,
        "        ops::attention_prefill_op, ops::o_proj_op, ops::mlp_norm_op,"
    )
    .unwrap();
    writeln!(
        src,
        "        ops::gate_silu_op, ops::up_matmul_op, ops::downproj_op,"
    )
    .unwrap();
    writeln!(
        src,
        "        ops::lm_head_norm_op, ops::lm_head_op, ops::barrier_inc_op,"
    )
    .unwrap();
    writeln!(src, "        ops::all_device_barrier_op>;").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    cudaFuncSetAttribute((void*)kernel,").unwrap();
    writeln!(
        src,
        "        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);"
    )
    .unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    kernel<<<grid, block, smem, stream>>>(g);").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    return (int)cudaGetLastError();").unwrap();
    writeln!(src, "}}").unwrap();

    GeneratedTkMegakernel {
        cuda_source: src,
        launch_fn_name,
        flat_params,
    }
}

/// Build the flat parameter list for the throughput TK launch wrapper.
/// Each entry is `(c_type, param_name)`.
fn build_tk_flat_params() -> Vec<(String, String)> {
    let mut p = Vec::new();
    let ptr = |name: &str| ("uint64_t".to_string(), name.to_string());
    let dim = |name: &str| ("int".to_string(), name.to_string());

    // VM state: barriers pgl
    p.push(ptr("bar_ptr"));
    p.push(dim("bar_b"));
    p.push(dim("bar_d"));
    p.push(dim("bar_r"));
    p.push(dim("bar_c"));

    // Instructions (work-stealing: flat [1, total_instructions, 32])
    p.push(ptr("instructions_ptr"));
    p.push(dim("total_instructions"));
    p.push(ptr("timings_ptr"));
    p.push(ptr("global_inst_idx_ptr"));

    // Weight tensors: gl<bf16, 1, -1, -1, hidden_dim, st_bf<256,64>>
    for name in &[
        "qkv_weights",
        "o_weights",
        "up_weights",
        "gate_weights",
        "lm_head_weights",
    ] {
        p.push(ptr(&format!("{name}_ptr")));
        p.push(dim(&format!("{name}_depth")));
        p.push(dim(&format!("{name}_rows")));
    }
    // down: gl<bf16, 1, -1, -1, intermediate_dim/num_devices, st_bf<256,64>>
    p.push(ptr("down_weights_ptr"));
    p.push(dim("down_weights_depth"));
    p.push(dim("down_weights_rows"));

    // Norm weights: gl<bf16, 1, 1, -1, hidden_dim>
    for name in &[
        "attn_norm_weights",
        "mlp_norm_weights",
        "lm_head_norm_weights",
    ] {
        p.push(ptr(&format!("{name}_ptr")));
        p.push(dim(&format!("{name}_rows")));
    }

    // KV cache: gl<bf16, -1, -1, num_kv_heads, head_dim>
    for name in &["k_cache", "v_cache"] {
        p.push(ptr(&format!("{name}_ptr")));
        p.push(dim(&format!("{name}_batch"))); // num_layers * num_pages
        p.push(dim(&format!("{name}_depth"))); // page_size
    }

    // RoPE
    p.push(ptr("rope_cos_ptr"));
    p.push(ptr("rope_sin_ptr"));
    p.push(dim("rope_rows"));

    // Activations (pgl wrappers — single ptr for num_devices=1)
    p.push(ptr("hidden_states_ptr"));
    p.push(ptr("rms_rope_intermediates_ptr"));
    p.push(ptr("rms_gate_intermediates_ptr"));
    p.push(ptr("q_post_rope_ptr"));
    p.push(dim("q_post_rope_cols"));
    p.push(ptr("attn_out_ptr"));
    p.push(ptr("silu_out_ptr"));
    p.push(ptr("rms_lm_head_intermediates_ptr"));
    p.push(ptr("logits_ptr"));
    p.push(dim("logits_cols"));

    // Paged KV metadata (int32 vectors)
    p.push(ptr("position_ids_ptr"));
    p.push(ptr("kv_append_indices_ptr"));

    p.push(ptr("prefill_qo_indptr_ptr"));
    p.push(dim("prefill_qo_indptr_len"));
    p.push(ptr("prefill_kv_indptr_ptr"));
    p.push(dim("prefill_kv_indptr_len"));
    p.push(ptr("prefill_kv_indices_ptr"));
    p.push(dim("prefill_kv_indices_len"));
    p.push(ptr("prefill_kv_last_page_len_ptr"));
    p.push(dim("prefill_kv_last_page_len_len"));

    p.push(ptr("decode_kv_indptr_ptr"));
    p.push(dim("decode_kv_indptr_len"));
    p.push(ptr("decode_kv_indices_ptr"));
    p.push(dim("decode_kv_indices_len"));
    p.push(ptr("decode_kv_last_page_len_ptr"));
    p.push(dim("decode_kv_last_page_len_len"));

    // Scalars
    p.push(("float".to_string(), "attn_scale".to_string()));
    p.push(("float".to_string(), "rms_norm_eps".to_string()));
    p.push(dim("num_pages"));
    p.push(dim("batch_size"));
    p.push(dim("num_prefill_tokens"));

    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_tk_megakernel_for_llama_1b() {
        let dims = TkModelDims {
            num_layers: 16,
            hidden_dim: 2048,
            intermediate_dim: 8192,
            head_dim: 64,
            num_attention_heads: 32,
            num_kv_heads: 8,
            kv_page_size: 128,
            prefill_kv_block_size: 128,
            decode_kv_block_size: 16,
            matmul_out_block_size: 256,
            matmul_batch_block_size: 128,
            vocab_size: 128256,
            sm_count: 132,
            num_devices: 1,
        };
        let result = generate_tk_megakernel("llama_1b", &dims);
        assert_eq!(result.launch_fn_name, "tk_megakernel_llama_1b_launch");
        assert!(result.cuda_source.contains("llama.cuh"));
        assert!(result.cuda_source.contains("LLAMA_NUM_DEVICES 1"));
        assert!(result.cuda_source.contains("LLAMA_NUM_LAYERS 16"));
        assert!(result.cuda_source.contains("LLAMA_HIDDEN_DIM 2048"));
        assert!(result.cuda_source.contains("qkv_rope_append_op"));
        assert!(result.cuda_source.contains("attention_decode_op"));
        assert!(result.cuda_source.contains("attention_prefill_op"));
        assert!(
            result
                .cuda_source
                .contains("mk<llama_config, llama_70b_globals")
        );
        assert!(
            result
                .cuda_source
                .contains("extern \"C\" int tk_megakernel_llama_1b_launch")
        );
        assert!(!result.flat_params.is_empty());
    }
}
