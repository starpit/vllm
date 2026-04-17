// SPDX-License-Identifier: Apache-2.0
//! CUDA source generation for megakernel compilation units.
//!
//! A megakernel groups multiple DeviceCallable subgraphs from a single
//! wave into one `__global__` cooperative kernel. This module generates
//! the `.cu` source from a list of [`DevicePhase`]s:
//!
//! 1. An internal params struct aggregating per-phase parameters.
//! 2. A `__global__` kernel that executes phases with grid sync.
//! 3. An `extern "C"` launch wrapper taking flat C-friendly args.
//!
//! The Rust codegen calls only the `extern "C"` wrapper with raw
//! pointers and scalars. CUDA compilation is handled by
//! `ferrite-cuda-builder/build.rs`.

#![allow(dead_code)]

use std::fmt::Write;

use crate::impl_lib::DevicePhase;

/// Generated CUDA source + metadata for one megakernel.
#[derive(Clone, Debug)]
pub struct GeneratedMegakernel {
    /// The complete `.cu` source code.
    pub cuda_source: String,
    /// The `extern "C"` launch function name.
    pub launch_fn_name: String,
    /// Per-phase flat parameter descriptors for the Rust FFI caller.
    /// Each entry is `(c_type, param_name)`.
    pub flat_params: Vec<(String, String)>,
    /// Number of inter-phase barrier counters needed (phases - 1).
    pub num_barriers: usize,
}

/// Generate a `.cu` source for one megakernel wave.
///
/// `wave_idx` disambiguates multiple megakernels within a model
/// (e.g. `megakernel_w0`, `megakernel_w1`).
pub fn generate_megakernel(wave_idx: usize, phases: &[DevicePhase]) -> GeneratedMegakernel {
    assert!(
        !phases.is_empty(),
        "megakernel requires at least 1 phase (got {})",
        phases.len()
    );

    let launch_fn_name = format!("megakernel_w{wave_idx}_launch");
    let params_struct_name = format!("MegakernelW{wave_idx}Params");
    let kernel_name = format!("megakernel_w{wave_idx}");

    let mut src = String::new();

    // ── Header ──
    writeln!(src, "// Auto-generated megakernel for wave {wave_idx}").unwrap();
    writeln!(
        src,
        "// DO NOT EDIT — regenerate via the forward! proc macro."
    )
    .unwrap();
    writeln!(src).unwrap();
    writeln!(src, "#include <cuda_bf16.h>").unwrap();
    writeln!(src, "#include <cooperative_groups.h>").unwrap();
    writeln!(src, "#include \"megakernel_ops.cuh\"").unwrap();
    writeln!(src).unwrap();

    // ── Collect all flat params across phases ──
    let mut all_flat: Vec<(String, String)> = Vec::new();
    for phase in phases {
        all_flat.extend(phase.flat_params.iter().cloned());
    }

    // ── Internal params struct ──
    writeln!(src, "struct {params_struct_name} {{").unwrap();
    for (i, phase) in phases.iter().enumerate() {
        writeln!(src, "    // Phase {i}").unwrap();
        for field in &phase.internal_fields {
            writeln!(src, "    {field};").unwrap();
        }
    }
    writeln!(src, "}};").unwrap();
    writeln!(src).unwrap();

    // ── __global__ kernel ──
    writeln!(
        src,
        "extern \"C\" __global__ void {kernel_name}({params_struct_name} p) {{"
    )
    .unwrap();
    writeln!(src, "    namespace cg = cooperative_groups;").unwrap();
    writeln!(src, "    extern __shared__ char smem[];").unwrap();
    writeln!(src).unwrap();

    // Destructure params struct into local variables so kernel_body
    // lines can reference bare names (p0_out, p1_input, etc.).
    for phase in phases {
        for field in &phase.internal_fields {
            // field is e.g. "__nv_bfloat16* p0_out" — extract the name (last token)
            let name = field.split_whitespace().last().unwrap_or("");
            // Strip leading * for pointer fields
            let name = name.trim_start_matches('*');
            writeln!(src, "    auto {name} = p.{name};").unwrap();
        }
    }
    writeln!(src).unwrap();

    for (i, phase) in phases.iter().enumerate() {
        if i > 0 {
            writeln!(src, "    cg::this_grid().sync();").unwrap();
            writeln!(src).unwrap();
        }
        writeln!(src, "    // Phase {i}").unwrap();
        for line in &phase.kernel_body {
            writeln!(src, "    {line}").unwrap();
        }
        writeln!(src).unwrap();
    }

    writeln!(src, "}}").unwrap();
    writeln!(src).unwrap();

    // ── extern "C" launch wrapper ──
    writeln!(src, "extern \"C\" int {launch_fn_name}(").unwrap();
    for (c_type, name) in &all_flat {
        writeln!(src, "    {c_type} {name},").unwrap();
    }
    writeln!(src, "    int __grid_x, int __block_x,").unwrap();
    writeln!(src, "    size_t __smem_bytes,").unwrap();
    writeln!(src, "    uint64_t __stream)").unwrap();
    writeln!(src, "{{").unwrap();
    writeln!(src, "    {params_struct_name} params;").unwrap();

    for phase in phases {
        for line in &phase.params_build {
            writeln!(src, "    {line}").unwrap();
        }
    }

    writeln!(src).unwrap();
    writeln!(src, "    dim3 grid(__grid_x);").unwrap();
    writeln!(src, "    dim3 block(__block_x);").unwrap();
    writeln!(src, "    void* args[] = {{ &params }};").unwrap();
    writeln!(src, "    return cudaLaunchCooperativeKernel(").unwrap();
    writeln!(src, "        (void*){kernel_name},").unwrap();
    writeln!(
        src,
        "        grid, block, args, __smem_bytes, (cudaStream_t)__stream);"
    )
    .unwrap();
    writeln!(src, "}}").unwrap();

    let num_barriers = if phases.len() > 1 {
        phases.len() - 1
    } else {
        0
    };
    GeneratedMegakernel {
        cuda_source: src,
        launch_fn_name,
        flat_params: all_flat,
        num_barriers,
    }
}

// ── TK megakernel codegen ────────────────────────────────────────
//
// Emits a `.cu` following the exact KVM pattern (llama.cu + llama.cuh):
//   1. kittens.cuh + megakernel.cuh includes
//   2. Model-specific `config` struct + `globals_t` template
//   3. Op includes + aliases
//   4. extern "C" launch wrapper
//
// The ops, config struct, and globals_t are taken directly from the
// vendored Megakernels repo (`third_party/Megakernels/`).

/// Model dimensions needed to instantiate the TK megakernel.
#[derive(Clone, Debug)]
pub struct TkModelDims {
    pub num_layers: u32,
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub head_dim: u32,
    pub num_attention_heads: u32,
    pub num_kv_heads: u32,
    pub kv_block_size: u32,
    pub matvec_block_size: u32,
    pub vocab_size: u32,
    pub sm_count: u32,
}

/// Generated TK megakernel CUDA source + metadata.
#[derive(Clone, Debug)]
pub struct GeneratedTkMegakernel {
    pub cuda_source: String,
    pub launch_fn_name: String,
    /// `(c_type, param_name)` for the extern "C" launch wrapper.
    pub flat_params: Vec<(String, String)>,
}

/// Generate a `.cu` source for a TK megakernel following the KVM pattern.
///
/// This emits:
/// - The fixed `config` struct (16 consumer warps, static scheduling)
/// - A `globals_t` template instantiated with model dimensions
/// - Op includes and aliases matching the vendored Llama ops
/// - An `extern "C"` launch wrapper that fills `globals_t` from flat
///   C params and calls `mk<config, globals, ops...>`
pub fn generate_tk_megakernel(
    model_name: &str,
    dims: &TkModelDims,
) -> GeneratedTkMegakernel {
    let launch_fn_name = format!("tk_megakernel_{model_name}_launch");
    let globals_typedef = format!("{model_name}_globals");

    let mut src = String::new();

    // ── Header ──
    writeln!(src, "// Auto-generated TK megakernel for {model_name}").unwrap();
    writeln!(src, "// DO NOT EDIT — regenerate via the forward! proc macro.").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "#include <cstring>").unwrap();
    writeln!(src, "#include \"kittens.cuh\"").unwrap();
    writeln!(src, "#include \"megakernel.cuh\"").unwrap();
    writeln!(src).unwrap();

    // ── Opcodes (matching llama.cuh) ──
    writeln!(src, "#define OPCODE_RMS_QKV_MatVecRopeAppend 1").unwrap();
    writeln!(src, "#define OPCODE_PartialAttention 2").unwrap();
    writeln!(src, "#define OPCODE_AttentionReduction 3").unwrap();
    writeln!(src, "#define OPCODE_O_ProjResidual 4").unwrap();
    writeln!(src, "#define OPCODE_RMS_DoubleMatVecSiLU 5").unwrap();
    writeln!(src, "#define OPCODE_DownProjResidual 6").unwrap();
    writeln!(src, "#define OPCODE_RMS_LM_Head 7").unwrap();
    writeln!(src).unwrap();

    // ── Config struct (matching KVM) ──
    writeln!(src, "struct config {{").unwrap();
    writeln!(src, "    static constexpr int INSTRUCTION_PIPELINE_STAGES = 2;").unwrap();
    writeln!(src, "    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS = 1;").unwrap();
    writeln!(src, "    static constexpr int INSTRUCTION_WIDTH = 32;").unwrap();
    writeln!(src, "    using instruction_t = int[INSTRUCTION_WIDTH];").unwrap();
    writeln!(src, "    static constexpr int TIMING_WIDTH = 128;").unwrap();
    writeln!(src, "    using timing_t = int[TIMING_WIDTH];").unwrap();
    writeln!(src, "    static constexpr int DYNAMIC_SEMAPHORES = 32;").unwrap();
    writeln!(src, "    static constexpr bool ENABLE_GLOBAL_WORK_QUEUE = false;").unwrap();
    writeln!(src, "    static constexpr int GLOBAL_WORK_QUEUE_PARTITIONS = 1;").unwrap();
    writeln!(src, "    static constexpr int NUM_CONSUMER_WARPS = 16;").unwrap();
    writeln!(src, "    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;").unwrap();
    writeln!(src, "    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;").unwrap();
    writeln!(src, "    static constexpr int NUM_BLOCKS = 1;").unwrap();
    writeln!(src, "    static constexpr int CLUSTER_BLOCKS = 1;").unwrap();
    writeln!(src, "    static constexpr int MAX_SHARED_MEMORY = ::kittens::MAX_SHARED_MEMORY;").unwrap();
    writeln!(src, "    static constexpr int SCRATCH_BYTES = 4096;").unwrap();
    writeln!(src, "    static constexpr int STATIC_SHARED_MEMORY =").unwrap();
    writeln!(src, "        512 + INSTRUCTION_PIPELINE_STAGES *").unwrap();
    writeln!(src, "                  (SCRATCH_BYTES + (INSTRUCTION_WIDTH + TIMING_WIDTH) * 4 +").unwrap();
    writeln!(src, "                   DYNAMIC_SEMAPHORES * 8);").unwrap();
    writeln!(src, "    static constexpr int DYNAMIC_SHARED_MEMORY =").unwrap();
    writeln!(src, "        MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;").unwrap();
    writeln!(src, "    static constexpr int PAGE_SIZE = 16384;").unwrap();
    writeln!(src, "    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;").unwrap();
    writeln!(src, "    static constexpr bool TIMING_RECORD_ENABLED = false;").unwrap();
    writeln!(src, "    static constexpr bool GMEM_SPIN_LOOP_SLEEP_NANOS = 20;").unwrap();
    writeln!(src, "    static constexpr int CONSUMER_REGISTERS = 104;").unwrap();
    writeln!(src, "    static constexpr int NON_CONSUMER_REGISTERS = 64;").unwrap();
    writeln!(src, "}};").unwrap();
    writeln!(src).unwrap();

    // ── globals_t ──
    let d = dims;
    writeln!(src, "template <int _num_layers, int _hidden_dim, int _intermediate_dim,").unwrap();
    writeln!(src, "          int _head_dim, int _num_attention_heads, int _num_kv_heads,").unwrap();
    writeln!(src, "          int _kv_block_size, int _matvec_block_size, int _sm_count>").unwrap();
    writeln!(src, "struct globals_t {{").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    constexpr static int num_layers = _num_layers;").unwrap();
    writeln!(src, "    constexpr static int matvec_block_size = _matvec_block_size;").unwrap();
    writeln!(src, "    constexpr static int kv_block_size = _kv_block_size;").unwrap();
    writeln!(src, "    constexpr static int head_dim = _head_dim;").unwrap();
    writeln!(src, "    constexpr static int hidden_dim = _hidden_dim;").unwrap();
    writeln!(src, "    constexpr static int intermediate_dim = _intermediate_dim;").unwrap();
    writeln!(src, "    constexpr static int num_attention_heads = _num_attention_heads;").unwrap();
    writeln!(src, "    constexpr static int num_kv_heads = _num_kv_heads;").unwrap();
    writeln!(src, "    constexpr static int sm_count = _sm_count;").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    using instruction_layout = megakernel::instruction_layout<config>;").unwrap();
    writeln!(src, "    using timing_layout = megakernel::timing_layout<config>;").unwrap();
    writeln!(src).unwrap();
    // Type aliases for kittens global layouts
    writeln!(src, "    using weights_t =").unwrap();
    writeln!(src, "        kittens::gl<kittens::bf16, 1, -1, -1, hidden_dim,").unwrap();
    writeln!(src, "           kittens::st_bf<matvec_block_size, 512>>;").unwrap();
    writeln!(src, "    using weights_big_indim_t =").unwrap();
    writeln!(src, "        kittens::gl<kittens::bf16, 1, -1, -1, intermediate_dim,").unwrap();
    writeln!(src, "           kittens::st_bf<matvec_block_size, 512>>;").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    using activations_t = kittens::gl<kittens::bf16, 1, 1, 1, hidden_dim,").unwrap();
    writeln!(src, "        kittens::sv_bf<hidden_dim>, kittens::sv_bf<head_dim>, kittens::sv_bf<matvec_block_size>>;").unwrap();
    writeln!(src, "    using activations_big_indim_t =").unwrap();
    writeln!(src, "        kittens::gl<kittens::bf16, 1, 1, 1, intermediate_dim, kittens::sv_bf<intermediate_dim>,").unwrap();
    writeln!(src, "           kittens::sv_bf<hidden_dim>, kittens::sv_bf<matvec_block_size>>;").unwrap();
    writeln!(src, "    using logits_t = kittens::gl<kittens::bf16, 1, 1, 1, -1, kittens::sv_bf<matvec_block_size>>;").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    using norm_weights_t = kittens::gl<kittens::bf16, 1, 1, -1, hidden_dim,").unwrap();
    writeln!(src, "        kittens::sv_bf<hidden_dim>, kittens::sv_bf<matvec_block_size>>;").unwrap();
    writeln!(src, "    using rope_table_t = kittens::gl<float, 1, 1, -1, head_dim, kittens::sv_fl<head_dim>>;").unwrap();
    writeln!(src, "    using kv_cache_t = kittens::gl<kittens::bf16, -1, -1, -1, head_dim,").unwrap();
    writeln!(src, "        kittens::sv_bf<matvec_block_size>,").unwrap();
    writeln!(src, "        kittens::tma::descriptor<kittens::st_bf<kv_block_size, head_dim>, 1>>;").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    using attn_out_intermediates_t =").unwrap();
    writeln!(src, "        kittens::gl<float, 1, num_attention_heads, -1, head_dim, kittens::sv_fl<head_dim>>;").unwrap();
    writeln!(src, "    using attn_lse_intermediates_t = kittens::gl<float, 1, 1, num_attention_heads, -1,").unwrap();
    writeln!(src, "        kittens::sv_fl<((sm_count + 15) / 16) * 16>>;").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    using barriers =").unwrap();
    writeln!(src, "        kittens::gl<uint, 1, -1, -1, num_attention_heads + 2 * num_kv_heads>;").unwrap();
    writeln!(src).unwrap();
    // Fields
    writeln!(src, "    barriers Bar;").unwrap();
    writeln!(src, "    instruction_layout instructions;").unwrap();
    writeln!(src, "    timing_layout timings;").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    weights_t qkv_weights;").unwrap();
    writeln!(src, "    norm_weights_t attn_norm_weights;").unwrap();
    writeln!(src, "    weights_t o_weights;").unwrap();
    writeln!(src, "    norm_weights_t mlp_norm_weights;").unwrap();
    writeln!(src, "    weights_t up_weights;").unwrap();
    writeln!(src, "    weights_t gate_weights;").unwrap();
    writeln!(src, "    weights_big_indim_t down_weights;").unwrap();
    writeln!(src, "    norm_weights_t lm_head_norm_weights;").unwrap();
    writeln!(src, "    weights_t lm_head_weights;").unwrap();
    writeln!(src, "    kv_cache_t k_cache;").unwrap();
    writeln!(src, "    kv_cache_t v_cache;").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    rope_table_t rope_cos;").unwrap();
    writeln!(src, "    rope_table_t rope_sin;").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    activations_t hidden_states;").unwrap();
    writeln!(src, "    activations_t q_post_rope;").unwrap();
    writeln!(src, "    activations_t attn_out;").unwrap();
    writeln!(src, "    attn_lse_intermediates_t attn_lse_intermediates;").unwrap();
    writeln!(src, "    attn_out_intermediates_t attn_out_intermediates;").unwrap();
    writeln!(src, "    activations_big_indim_t silu_out;").unwrap();
    writeln!(src, "    logits_t logits;").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    unsigned int pos_id;").unwrap();
    writeln!(src, "    float attn_scale;").unwrap();
    writeln!(src, "    float rms_norm_eps;").unwrap();
    writeln!(src, "    bool skip_attn_reduction;").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    dim3 grid() {{ return dim3(sm_count); }}").unwrap();
    writeln!(src, "    dim3 block() {{ return dim3(config::NUM_THREADS); }}").unwrap();
    writeln!(src, "    int dynamic_shared_memory() {{ return config::DYNAMIC_SHARED_MEMORY; }}").unwrap();
    writeln!(src, "}};").unwrap();
    writeln!(src).unwrap();

    // ── Typedef instantiation ──
    writeln!(
        src,
        "typedef globals_t<{}, {}, {}, {}, {}, {}, {}, {}, {}> {globals_typedef};",
        d.num_layers, d.hidden_dim, d.intermediate_dim, d.head_dim,
        d.num_attention_heads, d.num_kv_heads, d.kv_block_size,
        d.matvec_block_size, d.sm_count,
    ).unwrap();
    writeln!(src).unwrap();

    // ── Forward declarations for ops ──
    writeln!(src, "template <typename config = config, typename globals = {globals_typedef}>").unwrap();
    writeln!(src, "struct attention_partial;").unwrap();
    writeln!(src, "template <typename config = config, typename globals = {globals_typedef}>").unwrap();
    writeln!(src, "struct attention_reduction;").unwrap();
    writeln!(src, "template <typename config = config, typename globals = {globals_typedef}>").unwrap();
    writeln!(src, "struct rms_qkv_rope_append;").unwrap();
    writeln!(src, "template <typename config = config, typename globals = {globals_typedef}>").unwrap();
    writeln!(src, "struct downproj;").unwrap();
    writeln!(src, "template <typename config = config, typename globals = {globals_typedef}>").unwrap();
    writeln!(src, "struct o_proj;").unwrap();
    writeln!(src, "template <typename config = config, typename globals = {globals_typedef}>").unwrap();
    writeln!(src, "struct rms_upgate_silu;").unwrap();
    writeln!(src, "template <typename config = config, typename globals = {globals_typedef}>").unwrap();
    writeln!(src, "struct rms_lm_head;").unwrap();
    writeln!(src).unwrap();

    // ── Op includes ──
    // Prevent llama.cuh from being included by the op .cu files — our generated
    // code already defines config, globals_t, and the forward declarations.
    writeln!(src, "#define LLAMA_CUH_INCLUDED").unwrap();
    // The ops use LLAMA_1B_* macros for compile-time template args (tile sizes).
    writeln!(src, "#define LLAMA_1B_NUM_LAYERS {}", d.num_layers).unwrap();
    writeln!(src, "#define LLAMA_1B_HIDDEN_DIM {}", d.hidden_dim).unwrap();
    writeln!(src, "#define LLAMA_1B_INTERMEDIATE_DIM {}", d.intermediate_dim).unwrap();
    writeln!(src, "#define LLAMA_1B_HEAD_DIM {}", d.head_dim).unwrap();
    writeln!(src, "#define LLAMA_1B_NUM_ATTENTION_HEADS {}", d.num_attention_heads).unwrap();
    writeln!(src, "#define LLAMA_1B_NUM_KV_HEADS {}", d.num_kv_heads).unwrap();
    writeln!(src, "#define LLAMA_1B_KV_BLOCK_SIZE {}", d.kv_block_size).unwrap();
    writeln!(src, "#define LLAMA_1B_MATVEC_BLOCK_SIZE {}", d.matvec_block_size).unwrap();
    writeln!(src, "#define LLAMA_1B_LM_HEAD_BLOCK_SIZE 32").unwrap();
    writeln!(src, "#define LLAMA_1B_VOCAB_SIZE 128256").unwrap(); // TODO: make configurable
    // The ops hardcode `llama_1b_globals` — alias it to our model-specific globals.
    writeln!(src, "using llama_1b_globals = {globals_typedef};").unwrap();
    writeln!(src, "// Op implementations from vendored Megakernels").unwrap();
    writeln!(src, "#include \"rms_matvec_rope_append.cu\"").unwrap();
    writeln!(src, "#include \"attention_partial.cu\"").unwrap();
    writeln!(src, "#include \"attention_reduction.cu\"").unwrap();
    writeln!(src, "#include \"matvec_adds.cu\"").unwrap();
    writeln!(src, "#include \"upgate.cu\"").unwrap();
    writeln!(src, "#include \"rms_lm_head.cu\"").unwrap();
    writeln!(src).unwrap();

    // ── Op aliases ──
    writeln!(src, "using namespace kittens;").unwrap();
    writeln!(src, "using namespace megakernel;").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "using rms_qkv_rope_append_op = rms_qkv_rope_append<config, {globals_typedef}>;").unwrap();
    writeln!(src, "using attention_partial_op = attention_partial<config, {globals_typedef}>;").unwrap();
    writeln!(src, "using attention_reduction_op = attention_reduction<config, {globals_typedef}>;").unwrap();
    writeln!(src, "using o_proj_op = o_proj<config, {globals_typedef}>;").unwrap();
    writeln!(src, "using rms_upgate_silu_op = rms_upgate_silu<config, {globals_typedef}>;").unwrap();
    writeln!(src, "using downproj_op = downproj<config, {globals_typedef}>;").unwrap();
    writeln!(src, "using rms_lm_head_op = rms_lm_head<config, {globals_typedef}>;").unwrap();
    writeln!(src).unwrap();

    // ── Kernel type alias ──
    writeln!(src, "using kernel_t = decltype(&mk<config, {globals_typedef},").unwrap();
    writeln!(src, "    attention_partial_op, attention_reduction_op,").unwrap();
    writeln!(src, "    rms_qkv_rope_append_op, downproj_op,").unwrap();
    writeln!(src, "    o_proj_op, rms_upgate_silu_op, rms_lm_head_op>);").unwrap();
    writeln!(src).unwrap();

    // ── extern "C" launch wrapper ──
    //
    // Takes flat C params (raw pointers + scalars), fills a globals_t
    // struct, and launches the kernel. This is what the Rust FFI calls.
    let flat_params = build_tk_flat_params();

    writeln!(src, "extern \"C\" int {launch_fn_name}(").unwrap();
    for (c_type, name) in &flat_params {
        writeln!(src, "    {c_type} {name},").unwrap();
    }
    writeln!(src, "    uint64_t __stream)").unwrap();
    writeln!(src, "{{").unwrap();
    // gl<> deletes its default constructor, so we can't write `globals_t g;`.
    // Use aligned storage + reinterpret_cast instead — globals_t is a POD-like
    // bag of pointers and scalars that gets passed by value to the kernel.
    writeln!(src, "    alignas({globals_typedef}) char __g_buf[sizeof({globals_typedef})];").unwrap();
    writeln!(src, "    memset(__g_buf, 0, sizeof(__g_buf));").unwrap();
    writeln!(src, "    auto& g = *reinterpret_cast<{globals_typedef}*>(__g_buf);").unwrap();
    writeln!(src).unwrap();
    // Fill globals from flat params — each kittens::gl field needs
    // raw_ptr + dynamic dimensions set.
    writeln!(src, "    // VM state").unwrap();
    writeln!(src, "    g.Bar.raw_ptr = (uint*)bar_ptr;").unwrap();
    writeln!(src, "    g.Bar.depth_internal = bar_depth;").unwrap();
    writeln!(src, "    g.Bar.rows_internal = bar_rows;").unwrap();
    writeln!(src, "    g.instructions.raw_ptr = (int*)instructions_ptr;").unwrap();
    writeln!(src, "    g.timings.raw_ptr = (int*)timings_ptr;").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    // Weights").unwrap();
    writeln!(src, "    g.qkv_weights.raw_ptr = (__nv_bfloat16*)qkv_weights_ptr;").unwrap();
    writeln!(src, "    g.qkv_weights.depth_internal = qkv_weights_depth;").unwrap();
    writeln!(src, "    g.qkv_weights.rows_internal = qkv_weights_rows;").unwrap();
    writeln!(src, "    g.attn_norm_weights.raw_ptr = (__nv_bfloat16*)attn_norm_weights_ptr;").unwrap();
    writeln!(src, "    g.attn_norm_weights.rows_internal = attn_norm_weights_rows;").unwrap();
    writeln!(src, "    g.o_weights.raw_ptr = (__nv_bfloat16*)o_weights_ptr;").unwrap();
    writeln!(src, "    g.o_weights.depth_internal = o_weights_depth;").unwrap();
    writeln!(src, "    g.o_weights.rows_internal = o_weights_rows;").unwrap();
    writeln!(src, "    g.mlp_norm_weights.raw_ptr = (__nv_bfloat16*)mlp_norm_weights_ptr;").unwrap();
    writeln!(src, "    g.mlp_norm_weights.rows_internal = mlp_norm_weights_rows;").unwrap();
    writeln!(src, "    g.up_weights.raw_ptr = (__nv_bfloat16*)up_weights_ptr;").unwrap();
    writeln!(src, "    g.up_weights.depth_internal = up_weights_depth;").unwrap();
    writeln!(src, "    g.up_weights.rows_internal = up_weights_rows;").unwrap();
    writeln!(src, "    g.gate_weights.raw_ptr = (__nv_bfloat16*)gate_weights_ptr;").unwrap();
    writeln!(src, "    g.gate_weights.depth_internal = gate_weights_depth;").unwrap();
    writeln!(src, "    g.gate_weights.rows_internal = gate_weights_rows;").unwrap();
    writeln!(src, "    g.down_weights.raw_ptr = (__nv_bfloat16*)down_weights_ptr;").unwrap();
    writeln!(src, "    g.down_weights.depth_internal = down_weights_depth;").unwrap();
    writeln!(src, "    g.down_weights.rows_internal = down_weights_rows;").unwrap();
    writeln!(src, "    g.lm_head_norm_weights.raw_ptr = (__nv_bfloat16*)lm_head_norm_weights_ptr;").unwrap();
    writeln!(src, "    g.lm_head_norm_weights.rows_internal = lm_head_norm_weights_rows;").unwrap();
    writeln!(src, "    g.lm_head_weights.raw_ptr = (__nv_bfloat16*)lm_head_weights_ptr;").unwrap();
    writeln!(src, "    g.lm_head_weights.depth_internal = lm_head_weights_depth;").unwrap();
    writeln!(src, "    g.lm_head_weights.rows_internal = lm_head_weights_rows;").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    // KV cache").unwrap();
    writeln!(src, "    g.k_cache.raw_ptr = (__nv_bfloat16*)k_cache_ptr;").unwrap();
    writeln!(src, "    g.k_cache.batch_internal = k_cache_batch;").unwrap();
    writeln!(src, "    g.k_cache.depth_internal = k_cache_depth;").unwrap();
    writeln!(src, "    g.k_cache.rows_internal = k_cache_rows;").unwrap();
    writeln!(src, "    g.v_cache.raw_ptr = (__nv_bfloat16*)v_cache_ptr;").unwrap();
    writeln!(src, "    g.v_cache.batch_internal = v_cache_batch;").unwrap();
    writeln!(src, "    g.v_cache.depth_internal = v_cache_depth;").unwrap();
    writeln!(src, "    g.v_cache.rows_internal = v_cache_rows;").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    // RoPE tables").unwrap();
    writeln!(src, "    g.rope_cos.raw_ptr = (float*)rope_cos_ptr;").unwrap();
    writeln!(src, "    g.rope_cos.rows_internal = rope_rows;").unwrap();
    writeln!(src, "    g.rope_sin.raw_ptr = (float*)rope_sin_ptr;").unwrap();
    writeln!(src, "    g.rope_sin.rows_internal = rope_rows;").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    // Activation buffers").unwrap();
    writeln!(src, "    g.hidden_states.raw_ptr = (__nv_bfloat16*)hidden_states_ptr;").unwrap();
    writeln!(src, "    g.q_post_rope.raw_ptr = (__nv_bfloat16*)q_post_rope_ptr;").unwrap();
    writeln!(src, "    g.attn_out.raw_ptr = (__nv_bfloat16*)attn_out_ptr;").unwrap();
    writeln!(src, "    g.attn_lse_intermediates.raw_ptr = (float*)attn_lse_ptr;").unwrap();
    writeln!(src, "    g.attn_lse_intermediates.cols_internal = attn_lse_rows;").unwrap();
    writeln!(src, "    g.attn_out_intermediates.raw_ptr = (float*)attn_out_intermediates_ptr;").unwrap();
    writeln!(src, "    g.attn_out_intermediates.rows_internal = attn_out_intermediates_rows;").unwrap();
    writeln!(src, "    g.silu_out.raw_ptr = (__nv_bfloat16*)silu_out_ptr;").unwrap();
    writeln!(src, "    g.logits.raw_ptr = (__nv_bfloat16*)logits_ptr;").unwrap();
    writeln!(src, "    g.logits.cols_internal = logits_cols;").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    // Scalars").unwrap();
    writeln!(src, "    g.pos_id = pos_id;").unwrap();
    writeln!(src, "    g.attn_scale = attn_scale;").unwrap();
    writeln!(src, "    g.rms_norm_eps = rms_norm_eps;").unwrap();
    writeln!(src, "    g.skip_attn_reduction = skip_attn_reduction;").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    // Launch").unwrap();
    writeln!(src, "    dim3 grid = g.grid();").unwrap();
    writeln!(src, "    dim3 block = g.block();").unwrap();
    writeln!(src, "    int smem = g.dynamic_shared_memory();").unwrap();
    writeln!(src, "    cudaStream_t stream = (cudaStream_t)__stream;").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    cudaFuncSetAttribute(").unwrap();
    writeln!(src, "        (void*)mk<config, {globals_typedef},").unwrap();
    writeln!(src, "            attention_partial_op, attention_reduction_op,").unwrap();
    writeln!(src, "            rms_qkv_rope_append_op, downproj_op,").unwrap();
    writeln!(src, "            o_proj_op, rms_upgate_silu_op, rms_lm_head_op>,").unwrap();
    writeln!(src, "        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    mk<config, {globals_typedef},").unwrap();
    writeln!(src, "        attention_partial_op, attention_reduction_op,").unwrap();
    writeln!(src, "        rms_qkv_rope_append_op, downproj_op,").unwrap();
    writeln!(src, "        o_proj_op, rms_upgate_silu_op, rms_lm_head_op>").unwrap();
    writeln!(src, "        <<<grid, block, smem, stream>>>(g);").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "    return (int)cudaGetLastError();").unwrap();
    writeln!(src, "}}").unwrap();

    GeneratedTkMegakernel {
        cuda_source: src,
        launch_fn_name,
        flat_params,
    }
}

/// Build the flat parameter list for the TK launch wrapper.
/// Each entry is `(c_type, param_name)`.
fn build_tk_flat_params() -> Vec<(String, String)> {
    let mut p = Vec::new();
    let ptr = |name: &str| ("uint64_t".to_string(), name.to_string());
    let dim = |name: &str| ("int".to_string(), name.to_string());

    // VM state
    p.push(ptr("bar_ptr"));
    p.push(dim("bar_depth"));
    p.push(dim("bar_rows"));
    p.push(ptr("instructions_ptr"));
    p.push(ptr("timings_ptr"));

    // Weight tensors (ptr + dynamic dims for each -1 template arg)
    for name in &[
        "qkv_weights", "o_weights", "up_weights", "gate_weights", "lm_head_weights",
    ] {
        p.push(ptr(&format!("{name}_ptr")));
        p.push(dim(&format!("{name}_depth")));
        p.push(dim(&format!("{name}_rows")));
    }
    p.push(ptr("down_weights_ptr"));
    p.push(dim("down_weights_depth"));
    p.push(dim("down_weights_rows"));

    for name in &["attn_norm_weights", "mlp_norm_weights", "lm_head_norm_weights"] {
        p.push(ptr(&format!("{name}_ptr")));
        p.push(dim(&format!("{name}_rows")));
    }

    // KV cache
    for name in &["k_cache", "v_cache"] {
        p.push(ptr(&format!("{name}_ptr")));
        p.push(dim(&format!("{name}_batch")));
        p.push(dim(&format!("{name}_depth")));
        p.push(dim(&format!("{name}_rows")));
    }

    // RoPE
    p.push(ptr("rope_cos_ptr"));
    p.push(ptr("rope_sin_ptr"));
    p.push(dim("rope_rows"));

    // Activation buffers
    p.push(ptr("hidden_states_ptr"));
    p.push(ptr("q_post_rope_ptr"));
    p.push(ptr("attn_out_ptr"));
    p.push(ptr("attn_lse_ptr"));
    p.push(dim("attn_lse_rows"));
    p.push(ptr("attn_out_intermediates_ptr"));
    p.push(dim("attn_out_intermediates_rows"));
    p.push(ptr("silu_out_ptr"));
    p.push(ptr("logits_ptr"));
    p.push(dim("logits_cols"));

    // Scalars
    p.push(("unsigned int".to_string(), "pos_id".to_string()));
    p.push(("float".to_string(), "attn_scale".to_string()));
    p.push(("float".to_string(), "rms_norm_eps".to_string()));
    p.push(("int".to_string(), "skip_attn_reduction".to_string()));

    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_two_phase_megakernel() {
        let p0 = DevicePhase {
            flat_params: vec![
                ("void*".into(), "p0_out".into()),
                ("const void*".into(), "p0_input".into()),
                ("int".into(), "p0_n".into()),
            ],
            kernel_body: vec!["dc_rms_norm<__nv_bfloat16>(p0_out, p0_input, p0_n);".into()],
            params_build: vec![
                "params.p0_out = (__nv_bfloat16*)p0_out;".into(),
                "params.p0_input = (const __nv_bfloat16*)p0_input;".into(),
                "params.p0_n = p0_n;".into(),
            ],
            internal_fields: vec![
                "__nv_bfloat16* p0_out".into(),
                "const __nv_bfloat16* p0_input".into(),
                "int p0_n".into(),
            ],
            preamble: vec![],
        };
        let p1 = DevicePhase {
            flat_params: vec![
                ("void*".into(), "p1_x".into()),
                ("float".into(), "p1_scalar".into()),
                ("int".into(), "p1_n".into()),
            ],
            kernel_body: vec![
                "dc_scalar_mul_inplace<__nv_bfloat16>(p1_x, p1_scalar, p1_n, 1);".into(),
            ],
            params_build: vec![
                "params.p1_x = (__nv_bfloat16*)p1_x;".into(),
                "params.p1_scalar = p1_scalar;".into(),
                "params.p1_n = p1_n;".into(),
            ],
            internal_fields: vec![
                "__nv_bfloat16* p1_x".into(),
                "float p1_scalar".into(),
                "int p1_n".into(),
            ],
            preamble: vec![],
        };

        let result = generate_megakernel(0, &[p0, p1]);
        assert_eq!(result.launch_fn_name, "megakernel_w0_launch");
        assert!(result.cuda_source.contains("megakernel_w0_launch"));
        assert!(result.cuda_source.contains("megakernel_ops.cuh"));
        assert!(result.cuda_source.contains("cg::this_grid().sync()"));
        assert!(result.cuda_source.contains("cudaLaunchCooperativeKernel"));
        // Params struct is destructured into locals for kernel body
        assert!(result.cuda_source.contains("auto p0_out = p.p0_out;"));
        assert!(result.cuda_source.contains("auto p1_scalar = p.p1_scalar;"));
        assert_eq!(result.flat_params.len(), 6); // 3 + 3
    }

    #[test]
    #[should_panic(expected = "at least 1 phase")]
    fn panics_on_zero_phases() {
        generate_megakernel(0, &[]);
    }

    #[test]
    fn generates_tk_megakernel_for_llama_1b() {
        let dims = TkModelDims {
            num_layers: 16,
            hidden_dim: 2048,
            intermediate_dim: 8192,
            head_dim: 64,
            num_attention_heads: 32,
            num_kv_heads: 8,
            kv_block_size: 16,
            matvec_block_size: 16,
            vocab_size: 128256,
            sm_count: 132,
        };
        let result = generate_tk_megakernel("llama_1b", &dims);
        assert_eq!(result.launch_fn_name, "tk_megakernel_llama_1b_launch");
        assert!(result.cuda_source.contains("kittens.cuh"));
        assert!(result.cuda_source.contains("megakernel.cuh"));
        assert!(result.cuda_source.contains("ENABLE_GLOBAL_WORK_QUEUE = false"));
        assert!(result.cuda_source.contains("NUM_CONSUMER_WARPS = 16"));
        assert!(result.cuda_source.contains("globals_t<16, 2048, 8192, 64, 32, 8, 16, 16, 132>"));
        assert!(result.cuda_source.contains("rms_qkv_rope_append_op"));
        assert!(result.cuda_source.contains("mk<config, llama_1b_globals"));
        assert!(result.cuda_source.contains("extern \"C\" int tk_megakernel_llama_1b_launch"));
        assert!(!result.flat_params.is_empty());
    }
}
