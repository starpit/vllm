// SPDX-License-Identifier: Apache-2.0
//! CUDA codegen for the static megakernel.
//!
//! Generates a kernel that calls existing TK ops directly — same warp roles
//! (loader, consumer, storer, launcher), same `state<Config>`, same semaphores,
//! same tile types. The only difference: no controller warp, no instruction
//! fetch from global memory, no opcode dispatch switch.
//!
//! The generated kernel preserves the TK execution model exactly:
//! - Warps 0..NUM_CONSUMER_WARPS: consumer (call Op::consumer::run)
//! - Warp NUM_CONSUMER_WARPS: loader (call Op::loader::run)
//! - Warp NUM_CONSUMER_WARPS+1: storer (call Op::storer::run)
//! - Warp NUM_CONSUMER_WARPS+2: launcher (call Op::launcher::run)
//! - Warp NUM_CONSUMER_WARPS+3: FREED (was controller in the VM)

use crate::dag::*;
use std::collections::HashMap;
use std::fmt::Write;

/// Map from DSL op names to TK C++ op type names.
#[allow(dead_code)]
struct OpMapping {
    /// The C++ type name of the TK op (e.g. "attn_norm", "qkv_rope_append").
    cpp_type: &'static str,
    /// The opcode constant (e.g. "OPCODE_AttnNorm").
    opcode: &'static str,
    /// Whether this op has a non-trivial storer (some ops do storer work in consumer).
    has_storer: bool,
    /// Whether this op has a non-trivial launcher.
    has_launcher: bool,
}

fn op_mappings() -> HashMap<&'static str, OpMapping> {
    let mut m = HashMap::new();
    m.insert(
        "attn_norm",
        OpMapping {
            cpp_type: "attn_norm<config, globals>",
            opcode: "OPCODE_AttnNorm",
            has_storer: true,
            has_launcher: false,
        },
    );
    m.insert(
        "qkv_rope_append",
        OpMapping {
            cpp_type: "qkv_rope_append<config, globals>",
            opcode: "OPCODE_QKV_RopeAppend",
            has_storer: false, // consumer does the store
            has_launcher: false,
        },
    );
    m.insert(
        "attention_decode",
        OpMapping {
            cpp_type: "attention_decode<config, globals>",
            opcode: "OPCODE_GQA_AttentionDecode",
            has_storer: true,
            has_launcher: false,
        },
    );
    m.insert(
        "o_proj_residual",
        OpMapping {
            cpp_type: "o_proj<config, globals>",
            opcode: "OPCODE_O_ProjResidual",
            has_storer: false,
            has_launcher: false,
        },
    );
    m.insert(
        "mlp_norm",
        OpMapping {
            cpp_type: "mlp_norm<config, globals>",
            opcode: "OPCODE_MlpNorm",
            has_storer: true,
            has_launcher: false,
        },
    );
    m.insert(
        "gate_silu",
        OpMapping {
            cpp_type: "gate_silu<config, globals>",
            opcode: "OPCODE_GateSiLU",
            has_storer: false,
            has_launcher: false,
        },
    );
    m.insert(
        "up_matmul",
        OpMapping {
            cpp_type: "up_matmul<config, globals>",
            opcode: "OPCODE_UpMatmul",
            has_storer: false,
            has_launcher: false,
        },
    );
    m.insert(
        "down_proj_residual",
        OpMapping {
            cpp_type: "downproj<config, globals>",
            opcode: "OPCODE_DownProjResidual",
            has_storer: false,
            has_launcher: false,
        },
    );
    m.insert(
        "lm_head_norm",
        OpMapping {
            cpp_type: "lm_head_norm<config, globals>",
            opcode: "OPCODE_LM_HeadNorm",
            has_storer: true,
            has_launcher: false,
        },
    );
    m.insert(
        "lm_head",
        OpMapping {
            cpp_type: "lm_head<config, globals>",
            opcode: "OPCODE_LM_Head",
            has_storer: false,
            has_launcher: false,
        },
    );
    m.insert(
        "attention_prefill",
        OpMapping {
            cpp_type: "attention_prefill<config, globals>",
            opcode: "OPCODE_GQA_AttentionPrefill",
            has_storer: true,
            has_launcher: false,
        },
    );
    m
}

/// The sequence of TK ops for one LLaMA layer (decode path).
fn llama_layer_ops() -> Vec<&'static str> {
    vec![
        "attn_norm",
        "qkv_rope_append",
        "attention_decode",
        "o_proj_residual",
        "mlp_norm",
        "gate_silu",
        "up_matmul",
        "down_proj_residual",
    ]
}

/// Generate the static megakernel CUDA source (both decode and prefill kernels).
///
/// This generates two kernels from a single DSL definition:
/// 1. **Decode kernel**: 1 token per sequence, `attention_decode` op
/// 2. **Prefill kernel**: variable-length sequences, `attention_prefill` op,
///    multi-batch-block matmuls, per-sequence extend metadata
///
/// Both share the same preamble, helpers, and model constants. All other ops
/// (norms, matmuls, gate/up/down) are identical — only tile counts and
/// attention dispatch differ.
pub fn generate_static_kernel(dag: &ModelDag) -> String {
    let mut out = String::new();

    // ── Shared preamble ──
    emit_preamble(&mut out, dag);
    emit_run_op_template(&mut out);
    let (nl, hd, id, hdm, nah, nkh, vs) = emit_model_constants(&mut out, dag);
    let nbh = nah + 2 * nkh;
    let gqa_ratio = nah / nkh;
    emit_optimal_out_block(&mut out);

    // ── Decode kernel ──
    emit_decode_kernel(&mut out, dag, nl, nbh, gqa_ratio);

    // ── Prefill kernel ──
    emit_prefill_kernel(&mut out, dag, nl, nbh, nkh);

    // ── TkTensorArg + flat-arg launch wrappers ──
    emit_tensor_arg_and_globals_helper(&mut out);
    emit_decode_launch_wrapper(&mut out, dag);
    emit_prefill_launch_wrapper(&mut out, dag);

    // Suppress unused variable warnings for codegen params
    let _ = (hd, id, hdm, vs);

    out
}

fn emit_preamble(out: &mut String, dag: &ModelDag) {
    writeln!(out, "// GENERATED by megakernel! proc-macro — do not edit").unwrap();
    writeln!(out, "// Static megakernel: same TK ops, no VM dispatch").unwrap();
    writeln!(
        out,
        "// Emits both decode and prefill kernels from one DSL definition."
    )
    .unwrap();
    writeln!(out, "//").unwrap();
    writeln!(out, "// Pipeline: {}", dag.name).unwrap();
    writeln!(out, "// Ops: {}", dag.ops.len()).unwrap();
    writeln!(out).unwrap();

    // Override model dimension macros before the header defines its defaults.
    // This parameterizes llama_sm89_globals (and all op files that use it)
    // with the variant's specific dimensions.
    let nl = dag.params.get("NL").copied().unwrap_or(16);
    let hd = dag.params.get("HD").copied().unwrap_or(2048);
    let id = dag.params.get("ID").copied().unwrap_or(8192);
    let hdm = dag.params.get("HDM").copied().unwrap_or(64);
    let nah = dag.params.get("NAH").copied().unwrap_or(32);
    let nkh = dag.params.get("NKH").copied().unwrap_or(8);
    writeln!(out, "#define SM89_NUM_LAYERS             {nl}").unwrap();
    writeln!(out, "#define SM89_HIDDEN_DIM             {hd}").unwrap();
    writeln!(out, "#define SM89_INTERMEDIATE_DIM       {id}").unwrap();
    writeln!(out, "#define SM89_HEAD_DIM               {hdm}").unwrap();
    writeln!(out, "#define SM89_NUM_ATTENTION_HEADS    {nah}").unwrap();
    writeln!(out, "#define SM89_NUM_KV_HEADS           {nkh}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#include \"llama_sm89.cuh\"").unwrap();
    writeln!(out, "#include \"rms_norm_sm89.cu\"").unwrap();
    writeln!(out, "#include \"qkv_rope_append_sm89.cu\"").unwrap();
    writeln!(out, "#include \"attention_decode_sm89.cu\"").unwrap();
    writeln!(out, "#include \"attention_prefill_sm89.cu\"").unwrap();
    writeln!(out, "#include \"matmul_adds_sm89.cu\"").unwrap();
    writeln!(out, "#include \"gate_silu_sm89.cu\"").unwrap();
    writeln!(out, "#include \"up_matmul_sm89.cu\"").unwrap();
    writeln!(out, "#include \"lm_head_sm89.cu\"").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "using namespace kittens;").unwrap();
    writeln!(out, "using namespace kittens::prototype;").unwrap();
    writeln!(out, "using namespace kittens::prototype::vm;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "using config = llama_sm89_config;").unwrap();
    writeln!(out, "using globals = llama_sm89_globals;").unwrap();
    writeln!(out).unwrap();
}

fn emit_run_op_template(out: &mut String) {
    writeln!(
        out,
        "// Run a single TK op with warp role dispatch (no VM overhead)."
    )
    .unwrap();
    writeln!(
        out,
        "// This replaces the controller's instruction fetch + opcode switch."
    )
    .unwrap();
    writeln!(out, "template<typename Op>").unwrap();
    writeln!(out, "__device__ void run_op(").unwrap();
    writeln!(out, "    const globals &g, state<config> &kvms,").unwrap();
    writeln!(out, "    int opcode, int layer, int row, int col)").unwrap();
    writeln!(out, "{{").unwrap();
    writeln!(out, "    if (threadIdx.x == 0) {{").unwrap();
    writeln!(out, "        kvms.instruction()[0] = opcode;").unwrap();
    writeln!(out, "        kvms.instruction()[1] = layer;").unwrap();
    writeln!(out, "        kvms.instruction()[2] = row;").unwrap();
    writeln!(out, "        kvms.instruction()[3] = col;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        // Identity page mapping").unwrap();
    writeln!(out, "        for (int i = 0; i < config::NUM_PAGES; i++)").unwrap();
    writeln!(out, "            kvms.pid_order()[i] = i;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        Op::controller::init_semaphores(g, kvms);").unwrap();
    writeln!(out, "    }}").unwrap();
    // Memory fence ensures semaphore init values are visible to all threads.
    // Use barrier 15 with explicit thread count to avoid collision with
    // group<NUM_CONSUMER_WARPS>::sync(0) inside ops (barrier 0, different thread count).
    writeln!(out, "    asm volatile(\"membar.cta;\\n\" ::: \"memory\");").unwrap();
    writeln!(out, "    group<config::NUM_WARPS>::sync(15);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    kvms.pid_order_shared_addr = static_cast<uint32_t>("
    )
    .unwrap();
    writeln!(
        out,
        "        __cvta_generic_to_shared(&kvms.pid_order()[0]));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int wid = warpid();").unwrap();
    writeln!(out, "    if (wid < config::NUM_CONSUMER_WARPS) {{").unwrap();
    writeln!(out, "        Op::consumer::run(g, kvms);").unwrap();
    writeln!(out, "    }} else {{").unwrap();
    writeln!(out, "        switch (wid - config::NUM_CONSUMER_WARPS) {{").unwrap();
    writeln!(out, "        case 0: Op::loader::run(g, kvms); break;").unwrap();
    writeln!(out, "        case 1: Op::storer::run(g, kvms); break;").unwrap();
    writeln!(out, "        case 2: Op::launcher::run(g, kvms); break;").unwrap();
    writeln!(out, "        default: break; // freed controller warp").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    group<config::NUM_WARPS>::sync(15);  // post-op: all warps done"
    )
    .unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    // Extended 6-field variant for attention_prefill (adds extend_offset, chunk_len)
    writeln!(
        out,
        "// Extended run_op for attention_prefill: 6-field instruction."
    )
    .unwrap();
    writeln!(out, "template<typename Op>").unwrap();
    writeln!(out, "__device__ void run_op_ext(").unwrap();
    writeln!(out, "    const globals &g, state<config> &kvms,").unwrap();
    writeln!(
        out,
        "    int opcode, int layer, int seq_idx, int prefill_block_idx,"
    )
    .unwrap();
    writeln!(out, "    int kv_head_idx, int extend_offset)").unwrap();
    writeln!(out, "{{").unwrap();
    // Validate: run_op_ext has 7 fields (opcode + 6 data fields matching prefill_instruction).
    // prefill_instruction reads: [1]=layer, [2]=seq_idx, [3]=prefill_block_idx,
    //   [4]=prefill_token_offset, [5]=kv_head_idx
    writeln!(out, "    if (threadIdx.x == 0) {{").unwrap();
    writeln!(out, "        kvms.instruction()[0] = opcode;").unwrap();
    writeln!(out, "        kvms.instruction()[1] = layer;").unwrap();
    writeln!(out, "        kvms.instruction()[2] = seq_idx;").unwrap();
    writeln!(out, "        kvms.instruction()[3] = prefill_block_idx;").unwrap();
    writeln!(out, "        kvms.instruction()[4] = extend_offset;").unwrap();
    writeln!(out, "        kvms.instruction()[5] = kv_head_idx;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (int i = 0; i < config::NUM_PAGES; i++)").unwrap();
    writeln!(out, "            kvms.pid_order()[i] = i;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        Op::controller::init_semaphores(g, kvms);").unwrap();
    writeln!(out, "    }}").unwrap();
    // Memory fence + barrier 15 with explicit thread count (avoids barrier 0 collision).
    writeln!(out, "    asm volatile(\"membar.cta;\\n\" ::: \"memory\");").unwrap();
    writeln!(out, "    group<config::NUM_WARPS>::sync(15);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    kvms.pid_order_shared_addr = static_cast<uint32_t>("
    )
    .unwrap();
    writeln!(
        out,
        "        __cvta_generic_to_shared(&kvms.pid_order()[0]));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int wid = warpid();").unwrap();
    writeln!(out, "    if (wid < config::NUM_CONSUMER_WARPS) {{").unwrap();
    writeln!(out, "        Op::consumer::run(g, kvms);").unwrap();
    writeln!(out, "    }} else {{").unwrap();
    writeln!(out, "        switch (wid - config::NUM_CONSUMER_WARPS) {{").unwrap();
    writeln!(out, "        case 0: Op::loader::run(g, kvms); break;").unwrap();
    writeln!(out, "        case 1: Op::storer::run(g, kvms); break;").unwrap();
    writeln!(out, "        case 2: Op::launcher::run(g, kvms); break;").unwrap();
    writeln!(out, "        default: break;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    group<config::NUM_WARPS>::sync(15);  // post-op: all warps done"
    )
    .unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

/// Returns (nl, hd, id, hdm, nah, nkh, vs)
fn emit_model_constants(
    out: &mut String,
    dag: &ModelDag,
) -> (usize, usize, usize, usize, usize, usize, usize) {
    let nl = dag.params.get("NL").copied().unwrap_or(32);
    let hd = dag.params.get("HD").copied().unwrap_or(2048);
    let id = dag.params.get("ID").copied().unwrap_or(5632);
    let hdm = dag.params.get("HDM").copied().unwrap_or(64);
    let nah = dag.params.get("NAH").copied().unwrap_or(32);
    let nkh = dag.params.get("NKH").copied().unwrap_or(8);
    let vs = dag.params.get("VS").copied().unwrap_or(128256);
    let nbh = nah + 2 * nkh;
    let gqa_ratio = nah / nkh;

    writeln!(out, "// ── Model constants ──").unwrap();
    // All model dimensions except NL are compile-time constants, used in GL type
    // parameters (tile shapes, shared memory vector sizes). These define the kernel
    // specialization variant. NL (num_layers) is runtime — passed via launch wrapper.
    writeln!(out, "static constexpr int HD = {};", hd).unwrap();
    writeln!(out, "static constexpr int ID = {};", id).unwrap();
    writeln!(out, "static constexpr int HDM = {};", hdm).unwrap();
    writeln!(out, "static constexpr int NAH = {};", nah).unwrap();
    writeln!(out, "static constexpr int NKH = {};", nkh).unwrap();
    writeln!(out, "static constexpr int VS = {};", vs).unwrap();
    writeln!(out, "static constexpr int NBH = {};  // NAH + 2*NKH", nbh).unwrap();
    writeln!(
        out,
        "static constexpr int GQA_RATIO = {};  // NAH / NKH (attn batch block)",
        gqa_ratio
    )
    .unwrap();
    writeln!(out).unwrap();

    (nl, hd, id, hdm, nah, nkh, vs)
}

fn emit_optimal_out_block(out: &mut String) {
    // Don't emit our own — use the one already defined in llama_sm89.cuh
    // (kittens::prototype::vm::optimal_out_block)
    writeln!(
        out,
        "// optimal_out_block is defined in llama_sm89.cuh — using that directly."
    )
    .unwrap();
    writeln!(out).unwrap();
}

/// Emit shared memory + state<config> setup code (identical for both kernels).
fn emit_state_setup(out: &mut String) {
    writeln!(out, "    // ── state<config> setup (identical to VM) ──").unwrap();
    writeln!(out, "    __shared__ instruction_state_t<config>").unwrap();
    writeln!(
        out,
        "        instruction_state[config::INSTRUCTION_PIPELINE_STAGES];"
    )
    .unwrap();
    writeln!(out, "    __shared__ semaphore").unwrap();
    writeln!(
        out,
        "        page_finished[config::NUM_PAGES][config::INSTRUCTION_PIPELINE_STAGES_BITS],"
    )
    .unwrap();
    writeln!(
        out,
        "        instruction_arrived[config::INSTRUCTION_PIPELINE_STAGES],"
    )
    .unwrap();
    writeln!(
        out,
        "        instruction_finished[config::INSTRUCTION_PIPELINE_STAGES],"
    )
    .unwrap();
    writeln!(out, "        semaphores_ready;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    extern __shared__ int __shm[];").unwrap();
    writeln!(
        out,
        "    void *aligned_shm_addr = (void *)((1023 + (uint64_t)&__shm[0]) & ~(uint64_t)1023);"
    )
    .unwrap();
    writeln!(out, "    state<config>::page_array_t &pages =").unwrap();
    writeln!(
        out,
        "        *reinterpret_cast<state<config>::page_array_t *>(aligned_shm_addr);"
    )
    .unwrap();
    writeln!(out).unwrap();

    // ── Initialize semaphores (same as VM's kvm_internal) ──
    // Uses threadIdx parallelism, not single-thread init.
    writeln!(out, "    // Zero initial timings memory.").unwrap();
    writeln!(out, "    if (threadIdx.x < config::TIMING_WIDTH) {{").unwrap();
    writeln!(
        out,
        "        for (int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES; i++)"
    )
    .unwrap();
    writeln!(
        out,
        "            instruction_state[i].timings[threadIdx.x] = 0;"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    if (threadIdx.x < config::INSTRUCTION_PIPELINE_STAGES) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        init_semaphore(instruction_arrived[threadIdx.x], 1);"
    )
    .unwrap();
    writeln!(
        out,
        "        init_semaphore(instruction_finished[threadIdx.x], config::NUM_WARPS - 1);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    if (threadIdx.x < config::NUM_PAGES) {{").unwrap();
    writeln!(
        out,
        "        for (int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES_BITS; i++) {{"
    )
    .unwrap();
    writeln!(
        out,
        "            auto count = config::NUM_CONSUMER_WARPS * (1 << i);"
    )
    .unwrap();
    writeln!(
        out,
        "            init_semaphore(page_finished[threadIdx.x][i], count);"
    )
    .unwrap();
    writeln!(
        out,
        "            // sm89: no multi-count arrive; call arrive() count times"
    )
    .unwrap();
    writeln!(
        out,
        "            for (int _a = 0; _a < count; _a++) arrive(page_finished[threadIdx.x][i]);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    if (threadIdx.x == 0) {{").unwrap();
    writeln!(out, "        init_semaphore(semaphores_ready, 1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    // Memory fence + barrier 15 — matches kvm_internal in vm.cuh.
    // membar.cta ensures semaphore init values are visible before any thread proceeds.
    // Uses group<NUM_WARPS>::sync(15) (barrier 15 with explicit thread count) to avoid
    // collision with group<NUM_CONSUMER_WARPS>::sync(0) inside ops.
    writeln!(out, "    asm volatile(\"membar.cta;\\n\" ::: \"memory\");").unwrap();
    writeln!(out, "    group<config::NUM_WARPS>::sync(15);").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "    uint64_t start_time = 0;").unwrap();
    writeln!(out, "    state<config> kvms{{").unwrap();
    writeln!(out, "        instruction_state,").unwrap();
    writeln!(out, "        instruction_arrived,").unwrap();
    writeln!(out, "        instruction_finished,").unwrap();
    writeln!(out, "        0, 0,  // instruction_index, instruction_ring").unwrap();
    writeln!(out, "        {{}},    // reg_pid_order (zero-init)").unwrap();
    writeln!(out, "        pages,").unwrap();
    writeln!(out, "        page_finished,").unwrap();
    writeln!(out, "        semaphores_ready,").unwrap();
    writeln!(out, "        start_time").unwrap();
    writeln!(out, "    }};").unwrap();
    writeln!(out).unwrap();
}

/// Emit the per-op tile-count variables (shared by both kernels for matmul ops).
fn emit_tile_count_vars(out: &mut String, batch_expr: &str) {
    writeln!(out, "    // ── Per-op tile counts ──").unwrap();
    writeln!(
        out,
        "    const int qkv_out_block = optimal_out_block(NBH * HDM, sm_count, HDM);"
    )
    .unwrap();
    writeln!(
        out,
        "    const int o_proj_out_block = optimal_out_block(HD, sm_count, 32);"
    )
    .unwrap();
    writeln!(
        out,
        "    const int gate_out_block = optimal_out_block(ID, sm_count, 32);"
    )
    .unwrap();
    writeln!(
        out,
        "    const int lm_head_out_block = optimal_out_block(VS, sm_count, 128);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int n_cols_hd = HD / o_proj_out_block;").unwrap();
    writeln!(out, "    const int n_cols_id = ID / gate_out_block;").unwrap();
    writeln!(out, "    const int n_cols_vs = VS / lm_head_out_block;").unwrap();
    writeln!(
        out,
        "    const int n_attn_bb = ({batch_expr} + GQA_RATIO - 1) / GQA_RATIO;"
    )
    .unwrap();
    writeln!(out).unwrap();
}

/// Emit SM tile distribution loop for a single op.
fn emit_op_tile_loop(
    out: &mut String,
    indent: &str,
    mapping: &OpMapping,
    tile_count: &str,
    row_expr: &str,
    col_expr: &str,
) {
    writeln!(
        out,
        "{indent}// ── {} ── ({tile_count} tiles)",
        mapping.cpp_type
    )
    .unwrap();
    writeln!(out, "{indent}{{").unwrap();
    writeln!(out, "{indent}    const int total = {tile_count};").unwrap();
    writeln!(
        out,
        "{indent}    const int my_start = sm * total / sm_count;"
    )
    .unwrap();
    writeln!(
        out,
        "{indent}    const int my_end = (sm + 1) * total / sm_count;"
    )
    .unwrap();
    writeln!(
        out,
        "{indent}    for (int t = my_start; t < my_end; t++) {{"
    )
    .unwrap();
    writeln!(
        out,
        "{indent}        run_op<{}>(g, kvms, {}, layer, {row_expr}, {col_expr});",
        mapping.cpp_type, mapping.opcode
    )
    .unwrap();
    writeln!(out, "{indent}    }}").unwrap();
    writeln!(out, "{indent}}}").unwrap();
}

// ════════════════════════════════════════════════════════════════════
// Decode kernel
// ════════════════════════════════════════════════════════════════════

fn emit_decode_kernel(out: &mut String, dag: &ModelDag, nl: usize, nbh: usize, gqa_ratio: usize) {
    let _ = (nl, nbh, gqa_ratio); // used in emitted constants

    writeln!(
        out,
        "// ════════════════════════════════════════════════════════════"
    )
    .unwrap();
    writeln!(
        out,
        "// DECODE kernel: 1 token per sequence, batch_size sequences"
    )
    .unwrap();
    writeln!(
        out,
        "// ════════════════════════════════════════════════════════════"
    )
    .unwrap();
    writeln!(out).unwrap();

    writeln!(out, "__global__ __launch_bounds__(config::NUM_THREADS, 1)").unwrap();
    writeln!(
        out,
        "void {}_decode_static(const globals g, int batch_size, int num_layers) {{",
        dag.name
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (batch_size <= 0 || batch_size > 128) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int sm = blockIdx.x;").unwrap();
    writeln!(out, "    const int sm_count = gridDim.x;").unwrap();
    writeln!(out).unwrap();

    emit_tile_count_vars(out, "batch_size");
    emit_state_setup(out);

    let mappings = op_mappings();

    // Decode layer ops: attention_decode tiles are (n_attn_bb * NKH)
    let layer_op_tiles: Vec<(&str, &str, &str, &str)> = vec![
        ("attn_norm", "batch_size", "t", "0"),
        ("qkv_rope_append", "NBH", "0", "t"),
        (
            "attention_decode",
            "(n_attn_bb * NKH)",
            "t / NKH",
            "t % NKH",
        ),
        ("o_proj_residual", "n_cols_hd", "0", "t"),
        ("mlp_norm", "batch_size", "t", "0"),
        ("gate_silu", "n_cols_id", "0", "t"),
        ("up_matmul", "n_cols_id", "0", "t"),
        ("down_proj_residual", "n_cols_hd", "0", "t"),
    ];

    writeln!(out, "    // ── Layer loop ──").unwrap();
    writeln!(
        out,
        "    for (int layer = 0; layer < num_layers; layer++) {{"
    )
    .unwrap();

    for (op_name, tile_count, row_expr, col_expr) in &layer_op_tiles {
        if let Some(mapping) = mappings.get(op_name) {
            writeln!(out).unwrap();
            emit_op_tile_loop(out, "        ", mapping, tile_count, row_expr, col_expr);
        }
    }

    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();

    // Post-loop: lm_head_norm + lm_head
    emit_post_loop(out, &mappings, "batch_size");

    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

// ════════════════════════════════════════════════════════════════════
// Prefill kernel
// ════════════════════════════════════════════════════════════════════

fn emit_prefill_kernel(out: &mut String, dag: &ModelDag, nl: usize, nbh: usize, nkh: usize) {
    let _ = (nl, nbh, nkh); // used in emitted constants

    writeln!(
        out,
        "// ════════════════════════════════════════════════════════════"
    )
    .unwrap();
    writeln!(out, "// PREFILL kernel: variable-length sequences").unwrap();
    writeln!(out, "// total_tokens = sum of all chunk_lens").unwrap();
    writeln!(
        out,
        "// Per-sequence metadata: chunk_lens[], extend_offsets[]"
    )
    .unwrap();
    writeln!(
        out,
        "// ════════════════════════════════════════════════════════════"
    )
    .unwrap();
    writeln!(out).unwrap();

    writeln!(out, "__global__ __launch_bounds__(config::NUM_THREADS, 1)").unwrap();
    writeln!(out, "void {}_prefill_static(", dag.name).unwrap();
    writeln!(out, "    const globals g,").unwrap();
    writeln!(out, "    int total_tokens,").unwrap();
    writeln!(out, "    int num_seqs,").unwrap();
    writeln!(out, "    int num_layers,").unwrap();
    writeln!(out, "    const int *__restrict__ seq_chunk_lens,").unwrap();
    writeln!(out, "    const int *__restrict__ seq_extend_offsets)").unwrap();
    writeln!(out, "{{").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (total_tokens <= 0) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int sm = blockIdx.x;").unwrap();
    writeln!(out, "    const int sm_count = gridDim.x;").unwrap();
    writeln!(out).unwrap();

    // Prefill uses total_tokens for batch dimension (matmul tiles over all tokens)
    // n_matmul_blocks = ceil(total_tokens / 128) for matmul batch blocks
    writeln!(
        out,
        "    // Prefill: matmul batch blocks tile over total_tokens"
    )
    .unwrap();
    writeln!(
        out,
        "    const int n_matmul_blocks = (total_tokens + 127) / 128;"
    )
    .unwrap();
    writeln!(out).unwrap();

    emit_tile_count_vars(out, "total_tokens");
    emit_state_setup(out);

    let mappings = op_mappings();

    // Prefill layer ops — same as decode except:
    // 1. Norm ops tile over total_tokens (not batch_size)
    // 2. Matmul ops tile as (n_matmul_blocks * n_cols_X) with row=t/n_cols, col=t%n_cols
    // 3. Attention uses attention_prefill with per-sequence iteration
    writeln!(out, "    // ── Layer loop ──").unwrap();
    writeln!(
        out,
        "    for (int layer = 0; layer < num_layers; layer++) {{"
    )
    .unwrap();
    writeln!(out).unwrap();

    // attn_norm: 1 tile per token
    if let Some(m) = mappings.get("attn_norm") {
        emit_op_tile_loop(out, "        ", m, "total_tokens", "t", "0");
    }

    // qkv_rope_append: tiles over (n_matmul_blocks * NBH) — each matmul block × head-block
    if let Some(m) = mappings.get("qkv_rope_append") {
        writeln!(out).unwrap();
        writeln!(
            out,
            "        // qkv_rope_append: (n_matmul_blocks * NBH) tiles"
        )
        .unwrap();
        writeln!(out, "        {{").unwrap();
        writeln!(out, "            const int total = n_matmul_blocks * NBH;").unwrap();
        writeln!(
            out,
            "            const int my_start = sm * total / sm_count;"
        )
        .unwrap();
        writeln!(
            out,
            "            const int my_end = (sm + 1) * total / sm_count;"
        )
        .unwrap();
        writeln!(
            out,
            "            for (int t = my_start; t < my_end; t++) {{"
        )
        .unwrap();
        writeln!(
            out,
            "                run_op<{}>(g, kvms, {}, layer, t / NBH, t % NBH);",
            m.cpp_type, m.opcode
        )
        .unwrap();
        writeln!(out, "            }}").unwrap();
        writeln!(out, "        }}").unwrap();
    }

    // attention_prefill: iterate over sequences, each produces
    // (ceil(chunk_len/16) * NKH) tiles with 6-field instruction
    if let Some(m) = mappings.get("attention_prefill") {
        writeln!(out).unwrap();
        writeln!(
            out,
            "        // attention_prefill: per-sequence variable-length tiles"
        )
        .unwrap();
        writeln!(out, "        for (int seq = 0; seq < num_seqs; seq++) {{").unwrap();
        writeln!(
            out,
            "            const int chunk_len = seq_chunk_lens[seq];"
        )
        .unwrap();
        writeln!(
            out,
            "            const int extend_offset = seq_extend_offsets[seq];"
        )
        .unwrap();
        writeln!(
            out,
            "            const int n_q_blocks = (chunk_len + 15) / 16;"
        )
        .unwrap();
        writeln!(out, "            const int total = n_q_blocks * NKH;").unwrap();
        writeln!(
            out,
            "            const int my_start = sm * total / sm_count;"
        )
        .unwrap();
        writeln!(
            out,
            "            const int my_end = (sm + 1) * total / sm_count;"
        )
        .unwrap();
        writeln!(
            out,
            "            for (int t = my_start; t < my_end; t++) {{"
        )
        .unwrap();
        writeln!(
            out,
            "                run_op_ext<{}>(g, kvms, {}, layer, seq,",
            m.cpp_type, m.opcode
        )
        .unwrap();
        writeln!(out, "                    t / NKH, t % NKH, extend_offset);").unwrap();
        writeln!(out, "            }}").unwrap();
        writeln!(out, "        }}").unwrap();
    }

    // o_proj_residual: (n_matmul_blocks * n_cols_hd) tiles
    if let Some(m) = mappings.get("o_proj_residual") {
        writeln!(out).unwrap();
        writeln!(
            out,
            "        // o_proj_residual: (n_matmul_blocks * n_cols_hd) tiles"
        )
        .unwrap();
        writeln!(out, "        {{").unwrap();
        writeln!(
            out,
            "            const int total = n_matmul_blocks * n_cols_hd;"
        )
        .unwrap();
        writeln!(
            out,
            "            const int my_start = sm * total / sm_count;"
        )
        .unwrap();
        writeln!(
            out,
            "            const int my_end = (sm + 1) * total / sm_count;"
        )
        .unwrap();
        writeln!(
            out,
            "            for (int t = my_start; t < my_end; t++) {{"
        )
        .unwrap();
        writeln!(
            out,
            "                run_op<{}>(g, kvms, {}, layer, t / n_cols_hd, t % n_cols_hd);",
            m.cpp_type, m.opcode
        )
        .unwrap();
        writeln!(out, "            }}").unwrap();
        writeln!(out, "        }}").unwrap();
    }

    // mlp_norm: 1 tile per token
    if let Some(m) = mappings.get("mlp_norm") {
        writeln!(out).unwrap();
        emit_op_tile_loop(out, "        ", m, "total_tokens", "t", "0");
    }

    // gate_silu: (n_matmul_blocks * n_cols_id) tiles
    if let Some(m) = mappings.get("gate_silu") {
        writeln!(out).unwrap();
        writeln!(
            out,
            "        // gate_silu: (n_matmul_blocks * n_cols_id) tiles"
        )
        .unwrap();
        writeln!(out, "        {{").unwrap();
        writeln!(
            out,
            "            const int total = n_matmul_blocks * n_cols_id;"
        )
        .unwrap();
        writeln!(
            out,
            "            const int my_start = sm * total / sm_count;"
        )
        .unwrap();
        writeln!(
            out,
            "            const int my_end = (sm + 1) * total / sm_count;"
        )
        .unwrap();
        writeln!(
            out,
            "            for (int t = my_start; t < my_end; t++) {{"
        )
        .unwrap();
        writeln!(
            out,
            "                run_op<{}>(g, kvms, {}, layer, t / n_cols_id, t % n_cols_id);",
            m.cpp_type, m.opcode
        )
        .unwrap();
        writeln!(out, "            }}").unwrap();
        writeln!(out, "        }}").unwrap();
    }

    // up_matmul: same tiling as gate_silu
    if let Some(m) = mappings.get("up_matmul") {
        writeln!(out).unwrap();
        writeln!(
            out,
            "        // up_matmul: (n_matmul_blocks * n_cols_id) tiles"
        )
        .unwrap();
        writeln!(out, "        {{").unwrap();
        writeln!(
            out,
            "            const int total = n_matmul_blocks * n_cols_id;"
        )
        .unwrap();
        writeln!(
            out,
            "            const int my_start = sm * total / sm_count;"
        )
        .unwrap();
        writeln!(
            out,
            "            const int my_end = (sm + 1) * total / sm_count;"
        )
        .unwrap();
        writeln!(
            out,
            "            for (int t = my_start; t < my_end; t++) {{"
        )
        .unwrap();
        writeln!(
            out,
            "                run_op<{}>(g, kvms, {}, layer, t / n_cols_id, t % n_cols_id);",
            m.cpp_type, m.opcode
        )
        .unwrap();
        writeln!(out, "            }}").unwrap();
        writeln!(out, "        }}").unwrap();
    }

    // down_proj_residual: (n_matmul_blocks * n_cols_hd) tiles
    if let Some(m) = mappings.get("down_proj_residual") {
        writeln!(out).unwrap();
        writeln!(
            out,
            "        // down_proj_residual: (n_matmul_blocks * n_cols_hd) tiles"
        )
        .unwrap();
        writeln!(out, "        {{").unwrap();
        writeln!(
            out,
            "            const int total = n_matmul_blocks * n_cols_hd;"
        )
        .unwrap();
        writeln!(
            out,
            "            const int my_start = sm * total / sm_count;"
        )
        .unwrap();
        writeln!(
            out,
            "            const int my_end = (sm + 1) * total / sm_count;"
        )
        .unwrap();
        writeln!(
            out,
            "            for (int t = my_start; t < my_end; t++) {{"
        )
        .unwrap();
        writeln!(
            out,
            "                run_op<{}>(g, kvms, {}, layer, t / n_cols_hd, t % n_cols_hd);",
            m.cpp_type, m.opcode
        )
        .unwrap();
        writeln!(out, "            }}").unwrap();
        writeln!(out, "        }}").unwrap();
    }

    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();

    // Post-loop: lm_head_norm + lm_head (tile over total_tokens)
    emit_post_loop(out, &mappings, "total_tokens");

    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

/// Emit the post-loop lm_head_norm + lm_head ops.
fn emit_post_loop(out: &mut String, mappings: &HashMap<&str, OpMapping>, batch_expr: &str) {
    writeln!(out, "    // ── Post-loop: lm_head_norm + lm_head ──").unwrap();
    if let Some(m) = mappings.get("lm_head_norm") {
        writeln!(out, "    {{").unwrap();
        writeln!(out, "        const int total = {batch_expr};").unwrap();
        writeln!(out, "        const int my_start = sm * total / sm_count;").unwrap();
        writeln!(
            out,
            "        const int my_end = (sm + 1) * total / sm_count;"
        )
        .unwrap();
        writeln!(out, "        for (int t = my_start; t < my_end; t++) {{").unwrap();
        writeln!(
            out,
            "            run_op<{}>(g, kvms, {}, 0, t, 0);",
            m.cpp_type, m.opcode
        )
        .unwrap();
        writeln!(out, "        }}").unwrap();
        writeln!(out, "    }}").unwrap();
    }
    if let Some(m) = mappings.get("lm_head") {
        writeln!(out, "    {{").unwrap();
        writeln!(out, "        const int total = n_cols_vs;").unwrap();
        writeln!(out, "        const int my_start = sm * total / sm_count;").unwrap();
        writeln!(
            out,
            "        const int my_end = (sm + 1) * total / sm_count;"
        )
        .unwrap();
        writeln!(out, "        for (int t = my_start; t < my_end; t++) {{").unwrap();
        writeln!(
            out,
            "            run_op<{}>(g, kvms, {}, 0, 0, t);",
            m.cpp_type, m.opcode
        )
        .unwrap();
        writeln!(out, "        }}").unwrap();
        writeln!(out, "    }}").unwrap();
    }
}

/// Emit the TkTensorArg struct + make_arg helper (same as tk_launch.cu).
fn emit_tensor_arg_and_globals_helper(out: &mut String) {
    writeln!(out, "// Flat tensor descriptor passed from Rust via FFI.").unwrap();
    writeln!(out, "struct TkTensorArg {{").unwrap();
    writeln!(out, "    uint64_t ptr;").unwrap();
    writeln!(out, "    int b, d, r, c;").unwrap();
    writeln!(out, "}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "// Helper: construct a gl<> from a TkTensorArg.").unwrap();
    writeln!(out, "template<typename GL>").unwrap();
    writeln!(out, "static inline GL make_arg(const TkTensorArg &a) {{").unwrap();
    writeln!(out, "    return make_gl<GL>(a.ptr, a.b, a.d, a.r, a.c);").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

/// Emit the globals construction code (shared between decode and prefill launch wrappers).
/// Identical to tk_launch.cu's globals construction.
fn emit_globals_construction(out: &mut String, indent: &str) {
    // Use `globals` type alias (= llama_sm89_globals from the header).
    // NL in the template is the header's macro value — only used for weight shape
    // metadata, not the runtime loop bound (which uses num_layers param).
    writeln!(out, "{indent}using G = globals;").unwrap();
    writeln!(out, "{indent}G g {{").unwrap();
    writeln!(out, "{indent}    // VM state").unwrap();
    writeln!(out, "{indent}    make_arg<G::barriers>(bar),").unwrap();
    writeln!(
        out,
        "{indent}    make_arg<G::instruction_layout>(instructions),"
    )
    .unwrap();
    writeln!(out, "{indent}    make_arg<G::timing_layout>(timings),").unwrap();
    writeln!(out, "{indent}    // Weights").unwrap();
    writeln!(out, "{indent}    make_arg<G::weights_t>(qkv_w),").unwrap();
    writeln!(out, "{indent}    make_arg<G::norm_weights_t>(attn_norm_w),").unwrap();
    writeln!(out, "{indent}    make_arg<G::weights_t>(o_w),").unwrap();
    writeln!(out, "{indent}    make_arg<G::norm_weights_t>(mlp_norm_w),").unwrap();
    writeln!(out, "{indent}    make_arg<G::weights_t>(up_w),").unwrap();
    writeln!(out, "{indent}    make_arg<G::weights_t>(gate_w),").unwrap();
    writeln!(out, "{indent}    make_arg<G::weights_big_t>(down_w),").unwrap();
    writeln!(out, "{indent}    make_arg<G::norm_weights_t>(lm_norm_w),").unwrap();
    writeln!(out, "{indent}    make_arg<G::weights_t>(lm_w),").unwrap();
    writeln!(out, "{indent}    // KV cache").unwrap();
    writeln!(out, "{indent}    make_arg<G::kv_cache_t>(k_cache),").unwrap();
    writeln!(out, "{indent}    make_arg<G::kv_cache_t>(v_cache),").unwrap();
    writeln!(out, "{indent}    // RoPE").unwrap();
    writeln!(out, "{indent}    make_arg<G::rope_table_t>(rope_cos),").unwrap();
    writeln!(out, "{indent}    make_arg<G::rope_table_t>(rope_sin),").unwrap();
    writeln!(out, "{indent}    // Activations").unwrap();
    writeln!(out, "{indent}    make_arg<G::activations_t>(hidden),").unwrap();
    writeln!(out, "{indent}    make_arg<G::activations_t>(rms_rope),").unwrap();
    writeln!(out, "{indent}    make_arg<G::activations_t>(rms_gate),").unwrap();
    writeln!(out, "{indent}    make_arg<G::activations_t>(q_post),").unwrap();
    writeln!(out, "{indent}    make_arg<G::activations_t>(attn_out),").unwrap();
    writeln!(out, "{indent}    make_arg<G::activations_big_t>(silu),").unwrap();
    writeln!(out, "{indent}    make_arg<G::activations_t>(rms_lm),").unwrap();
    writeln!(out, "{indent}    make_arg<G::logits_t>(logits_arg),").unwrap();
    writeln!(out, "{indent}    // Paged KV metadata — decode").unwrap();
    writeln!(out, "{indent}    make_arg<G::int32_vector_t>(pos_ids),").unwrap();
    writeln!(out, "{indent}    make_arg<G::int32_vector_t>(kv_indptr),").unwrap();
    writeln!(out, "{indent}    make_arg<G::int32_vector_t>(kv_indices),").unwrap();
    writeln!(
        out,
        "{indent}    make_arg<G::int32_vector_t>(kv_last_page),"
    )
    .unwrap();
    writeln!(out, "{indent}    make_arg<G::int32_vector_t>(kv_append),").unwrap();
    writeln!(out, "{indent}    // Prefill KV metadata").unwrap();
    writeln!(
        out,
        "{indent}    make_arg<G::int32_vector_t>(prefill_qo_indptr),"
    )
    .unwrap();
    writeln!(
        out,
        "{indent}    make_arg<G::int32_vector_t>(prefill_kv_indptr),"
    )
    .unwrap();
    writeln!(
        out,
        "{indent}    make_arg<G::int32_vector_t>(prefill_kv_indices),"
    )
    .unwrap();
    writeln!(
        out,
        "{indent}    make_arg<G::int32_vector_t>(prefill_kv_last_page_len),"
    )
    .unwrap();
    writeln!(out, "{indent}    num_prefill_tokens,").unwrap();
    writeln!(out, "{indent}    // Scalars").unwrap();
    writeln!(out, "{indent}    attn_scale,").unwrap();
    writeln!(out, "{indent}    rms_norm_eps,").unwrap();
    writeln!(out, "{indent}    num_pages,").unwrap();
    writeln!(out, "{indent}    batch_size,").unwrap();
    writeln!(out, "{indent}}};").unwrap();
}

/// The flat-arg parameter list shared by both decode and prefill launch wrappers.
/// Matches tk_llama_1b_launch in tk_launch.cu exactly.
const LAUNCH_PARAMS: &str = "\
    // VM state (3 tensors)
    TkTensorArg bar, TkTensorArg instructions, TkTensorArg timings,
    // Weights (9 tensors)
    TkTensorArg qkv_w, TkTensorArg attn_norm_w, TkTensorArg o_w,
    TkTensorArg mlp_norm_w, TkTensorArg up_w, TkTensorArg gate_w,
    TkTensorArg down_w, TkTensorArg lm_norm_w, TkTensorArg lm_w,
    // KV cache (2 tensors)
    TkTensorArg k_cache, TkTensorArg v_cache,
    // RoPE (2 tensors)
    TkTensorArg rope_cos, TkTensorArg rope_sin,
    // Activations (8 tensors)
    TkTensorArg hidden, TkTensorArg rms_rope, TkTensorArg rms_gate,
    TkTensorArg q_post, TkTensorArg attn_out, TkTensorArg silu,
    TkTensorArg rms_lm, TkTensorArg logits_arg,
    // Paged KV metadata — decode (5 tensors)
    TkTensorArg pos_ids, TkTensorArg kv_indptr, TkTensorArg kv_indices,
    TkTensorArg kv_last_page, TkTensorArg kv_append,
    // Paged KV metadata — prefill (4 tensors)
    TkTensorArg prefill_qo_indptr, TkTensorArg prefill_kv_indptr,
    TkTensorArg prefill_kv_indices, TkTensorArg prefill_kv_last_page_len,
    // Scalars
    float attn_scale, float rms_norm_eps,
    int num_pages, int batch_size, int num_prefill_tokens, int num_layers,
    // CUDA stream
    uint64_t stream";

/// Emit host-side assertions for dynamic dims that `make_gl` doesn't check (the -1 dims).
/// These fire before cudaLaunchKernel, catching mismatched runtime shapes cheaply.
fn emit_dynamic_dim_assertions(out: &mut String, _dag: &ModelDag, indent: &str, is_prefill: bool) {
    // Barrier must cover all pipeline stages (layers)
    writeln!(
        out,
        "{indent}// ── Dynamic dim assertions (catch -1 dims that make_gl skips) ──"
    )
    .unwrap();
    writeln!(out, "{indent}if (num_layers <= 0) {{ fprintf(stderr, \"ASSERT: num_layers %d <= 0\\n\", num_layers); fflush(stderr); return -100; }}").unwrap();
    writeln!(out, "{indent}if (bar.b < num_layers) {{ fprintf(stderr, \"ASSERT: barrier batch %d < num_layers %d\\n\", bar.b, num_layers); fflush(stderr); return -100; }}").unwrap();

    // Activations: rows must cover batch
    writeln!(out, "{indent}if (hidden.r < batch_size) {{ fprintf(stderr, \"ASSERT: hidden rows %d < batch_size %d\\n\", hidden.r, batch_size); fflush(stderr); return -101; }}").unwrap();

    // Scalars must be sane
    writeln!(out, "{indent}if (batch_size <= 0) {{ fprintf(stderr, \"ASSERT: batch_size %d <= 0\\n\", batch_size); fflush(stderr); return -106; }}").unwrap();
    writeln!(out, "{indent}if (num_pages <= 0) {{ fprintf(stderr, \"ASSERT: num_pages %d <= 0\\n\", num_pages); fflush(stderr); return -107; }}").unwrap();

    // KV metadata must be non-empty
    writeln!(out, "{indent}if (kv_indices.c < 1) {{ fprintf(stderr, \"ASSERT: kv_indices empty\\n\"); fflush(stderr); return -108; }}").unwrap();

    if is_prefill {
        writeln!(out, "{indent}if (num_prefill_tokens <= 0) {{ fprintf(stderr, \"ASSERT: num_prefill_tokens %d <= 0\\n\", num_prefill_tokens); fflush(stderr); return -109; }}").unwrap();
        writeln!(out, "{indent}if (seq_chunk_lens == nullptr) {{ fprintf(stderr, \"ASSERT: seq_chunk_lens is null\\n\"); fflush(stderr); return -110; }}").unwrap();
        writeln!(out, "{indent}if (seq_extend_offsets == nullptr) {{ fprintf(stderr, \"ASSERT: seq_extend_offsets is null\\n\"); fflush(stderr); return -111; }}").unwrap();
        // batch_size is num_seqs for prefill — must be <= num_prefill_tokens
        writeln!(out, "{indent}if (batch_size > num_prefill_tokens) {{ fprintf(stderr, \"ASSERT: batch_size(num_seqs) %d > num_prefill_tokens %d — did you pass total_tokens for both?\\n\", batch_size, num_prefill_tokens); fflush(stderr); return -112; }}").unwrap();
        // Validate sum(seq_chunk_lens) == num_prefill_tokens (host-side, seq_chunk_lens is device ptr so we read via cudaMemcpy)
        writeln!(
            out,
            "{indent}{{ int *h_chunks = (int*)alloca(batch_size * sizeof(int));"
        )
        .unwrap();
        writeln!(out, "{indent}  cudaMemcpyAsync(h_chunks, seq_chunk_lens, batch_size * sizeof(int), cudaMemcpyDeviceToHost, (cudaStream_t)stream);").unwrap();
        writeln!(
            out,
            "{indent}  cudaStreamSynchronize((cudaStream_t)stream);"
        )
        .unwrap();
        writeln!(
            out,
            "{indent}  int sum = 0; for (int i = 0; i < batch_size; i++) sum += h_chunks[i];"
        )
        .unwrap();
        writeln!(out, "{indent}  if (sum != num_prefill_tokens) {{ fprintf(stderr, \"ASSERT: sum(seq_chunk_lens)=%d != num_prefill_tokens=%d\\n\", sum, num_prefill_tokens); fflush(stderr); return -113; }} }}").unwrap();
    }

    writeln!(out).unwrap();
}

fn emit_decode_launch_wrapper(out: &mut String, dag: &ModelDag) {
    writeln!(out, "// ── C launch wrapper (decode) ──").unwrap();
    writeln!(
        out,
        "// Same flat-arg calling convention as tk_llama_1b_launch."
    )
    .unwrap();
    writeln!(
        out,
        "// Wrapped in try/catch to prevent C++ exceptions from crossing FFI boundary."
    )
    .unwrap();
    writeln!(out, "extern \"C\" int {}_decode_static_launch(", dag.name).unwrap();
    writeln!(out, "{}", LAUNCH_PARAMS).unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "  try {{").unwrap();
    emit_globals_construction(out, "    ");
    writeln!(out).unwrap();
    emit_dynamic_dim_assertions(out, dag, "    ", false);
    writeln!(out, "    int shmem = g.dynamic_shared_memory();").unwrap();
    writeln!(
        out,
        "    auto err = cudaFuncSetAttribute({}_decode_static,",
        dag.name
    )
    .unwrap();
    writeln!(
        out,
        "        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);"
    )
    .unwrap();
    writeln!(out, "    if (err != cudaSuccess) return (int)err;").unwrap();
    writeln!(
        out,
        "    {}_decode_static<<<g.grid(), g.block(), shmem, (cudaStream_t)stream>>>(",
        dag.name
    )
    .unwrap();
    writeln!(out, "        g, batch_size, num_layers);").unwrap();
    writeln!(out, "    err = cudaGetLastError();").unwrap();
    writeln!(out, "    return (int)err;").unwrap();
    writeln!(out, "  }} catch (const std::exception &e) {{").unwrap();
    writeln!(
        out,
        "    fprintf(stderr, \"decode launch C++ exception: %s\\n\", e.what()); fflush(stderr);"
    )
    .unwrap();
    writeln!(out, "    return -2;").unwrap();
    writeln!(out, "  }} catch (...) {{").unwrap();
    writeln!(
        out,
        "    fprintf(stderr, \"decode launch unknown C++ exception\\n\"); fflush(stderr);"
    )
    .unwrap();
    writeln!(out, "    return -3;").unwrap();
    writeln!(out, "  }}").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

fn emit_prefill_launch_wrapper(out: &mut String, dag: &ModelDag) {
    writeln!(out, "// ── C launch wrapper (prefill) ──").unwrap();
    writeln!(out, "extern \"C\" int {}_prefill_static_launch(", dag.name).unwrap();
    writeln!(out, "{},", LAUNCH_PARAMS).unwrap();
    writeln!(out, "    // Prefill per-sequence metadata").unwrap();
    writeln!(
        out,
        "    const int *seq_chunk_lens, const int *seq_extend_offsets"
    )
    .unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "  try {{").unwrap();
    writeln!(out, "    fprintf(stderr, \"prefill launch: batch_size=%d num_layers=%d num_prefill_tokens=%d\\n\", batch_size, num_layers, num_prefill_tokens); fflush(stderr);").unwrap();
    emit_globals_construction(out, "    ");
    writeln!(out).unwrap();
    writeln!(out, "    fprintf(stderr, \"prefill: globals constructed, shmem=%d\\n\", g.dynamic_shared_memory()); fflush(stderr);").unwrap();
    emit_dynamic_dim_assertions(out, dag, "    ", true);
    writeln!(out, "    int shmem = g.dynamic_shared_memory();").unwrap();
    writeln!(
        out,
        "    auto err = cudaFuncSetAttribute({}_prefill_static,",
        dag.name
    )
    .unwrap();
    writeln!(
        out,
        "        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);"
    )
    .unwrap();
    writeln!(out, "    if (err != cudaSuccess) return (int)err;").unwrap();
    writeln!(
        out,
        "    {}_prefill_static<<<g.grid(), g.block(), shmem, (cudaStream_t)stream>>>(",
        dag.name
    )
    .unwrap();
    writeln!(
        out,
        "        g, num_prefill_tokens, batch_size, num_layers, seq_chunk_lens, seq_extend_offsets);"
    )
    .unwrap();
    writeln!(out, "    err = cudaGetLastError();").unwrap();
    writeln!(out, "    return (int)err;").unwrap();
    writeln!(out, "  }} catch (const std::exception &e) {{").unwrap();
    writeln!(
        out,
        "    fprintf(stderr, \"prefill launch C++ exception: %s\\n\", e.what()); fflush(stderr);"
    )
    .unwrap();
    writeln!(out, "    return -2;").unwrap();
    writeln!(out, "  }} catch (...) {{").unwrap();
    writeln!(
        out,
        "    fprintf(stderr, \"prefill launch unknown C++ exception\\n\"); fflush(stderr);"
    )
    .unwrap();
    writeln!(out, "    return -1;").unwrap();
    writeln!(out, "  }}").unwrap();
    writeln!(out, "}}").unwrap();
}

/// Generate a debug variant of the static megakernel that syncs and writes
/// to a debug buffer after each op. This allows identifying which op crashes.
///
/// The debug buffer is an int array of size NUM_OPS_PER_LAYER * num_layers + 2.
/// Each entry is set to 1 after the corresponding op completes successfully.
/// If the kernel crashes, the last 0 entry indicates the crashing op.
pub fn generate_debug_kernel(dag: &ModelDag) -> String {
    let mut out = String::new();

    emit_preamble(&mut out, dag);
    emit_run_op_template(&mut out);
    let (nl, _hd, _id, _hdm, _nah, nkh, _vs) = emit_model_constants(&mut out, dag);
    let nbh = _nah + 2 * nkh;
    let gqa_ratio = _nah / nkh;
    emit_optimal_out_block(&mut out);

    // Debug decode kernel
    writeln!(out, "__global__ __launch_bounds__(config::NUM_THREADS, 1)").unwrap();
    writeln!(
        out,
        "void {}_decode_debug(const globals g, int batch_size, int num_layers, int *debug_buf) {{",
        dag.name
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (batch_size <= 0 || batch_size > 128) return;").unwrap();
    writeln!(out, "    const int sm = blockIdx.x;").unwrap();
    writeln!(out, "    const int sm_count = gridDim.x;").unwrap();
    writeln!(out).unwrap();
    emit_tile_count_vars(&mut out, "batch_size");
    emit_state_setup(&mut out);

    let mappings = op_mappings();
    let layer_ops: Vec<(&str, &str, &str, &str)> = vec![
        ("attn_norm", "batch_size", "t", "0"),
        ("qkv_rope_append", "NBH", "0", "t"),
        (
            "attention_decode",
            "(n_attn_bb * NKH)",
            "t / NKH",
            "t % NKH",
        ),
        ("o_proj_residual", "n_cols_hd", "0", "t"),
        ("mlp_norm", "batch_size", "t", "0"),
        ("gate_silu", "n_cols_id", "0", "t"),
        ("up_matmul", "n_cols_id", "0", "t"),
        ("down_proj_residual", "n_cols_hd", "0", "t"),
    ];

    writeln!(
        out,
        "    for (int layer = 0; layer < num_layers; layer++) {{"
    )
    .unwrap();

    for (op_idx, (op_name, tile_count, row_expr, col_expr)) in layer_ops.iter().enumerate() {
        if let Some(mapping) = mappings.get(op_name) {
            writeln!(out).unwrap();
            emit_op_tile_loop(
                &mut out, "        ", mapping, tile_count, row_expr, col_expr,
            );
            // Debug sync + marker
            writeln!(out, "        __syncthreads();").unwrap();
            writeln!(
                out,
                "        if (threadIdx.x == 0 && blockIdx.x == 0) debug_buf[layer * 8 + {op_idx}] = 1;"
            )
            .unwrap();
            writeln!(out, "        __syncthreads();").unwrap();
        }
    }

    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();

    // Post-loop: lm_head_norm + lm_head with debug markers
    let post_ops = [
        ("lm_head_norm", "batch_size"),
        ("lm_head", "batch_size * n_cols_vs"),
    ];
    for (post_idx, (op_name, tile_count)) in post_ops.iter().enumerate() {
        if let Some(mapping) = mappings.get(op_name) {
            let (row_expr, col_expr) = if *op_name == "lm_head_norm" {
                ("t", "0")
            } else {
                ("t / n_cols_vs", "t % n_cols_vs")
            };
            emit_op_tile_loop(&mut out, "    ", mapping, tile_count, row_expr, col_expr);
            writeln!(out, "    __syncthreads();").unwrap();
            writeln!(
                out,
                "    if (threadIdx.x == 0 && blockIdx.x == 0) debug_buf[num_layers * 8 + {post_idx}] = 1;"
            )
            .unwrap();
            writeln!(out, "    __syncthreads();").unwrap();
        }
    }

    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    // Launch wrapper
    emit_tensor_arg_and_globals_helper(&mut out);
    writeln!(out, "extern \"C\" int {}_decode_debug_launch(", dag.name).unwrap();
    writeln!(out, "{},", LAUNCH_PARAMS).unwrap();
    writeln!(out, "    int *debug_buf").unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "  try {{").unwrap();
    emit_globals_construction(&mut out, "    ");
    writeln!(out).unwrap();
    writeln!(out, "    int shmem = g.dynamic_shared_memory();").unwrap();
    writeln!(
        out,
        "    auto err = cudaFuncSetAttribute({}_decode_debug,",
        dag.name
    )
    .unwrap();
    writeln!(
        out,
        "        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);"
    )
    .unwrap();
    writeln!(out, "    if (err != cudaSuccess) return (int)err;").unwrap();
    writeln!(
        out,
        "    {}_decode_debug<<<g.grid(), g.block(), shmem, (cudaStream_t)stream>>>(",
        dag.name
    )
    .unwrap();
    writeln!(out, "        g, batch_size, num_layers, debug_buf);").unwrap();
    writeln!(out, "    err = cudaGetLastError();").unwrap();
    writeln!(out, "    return (int)err;").unwrap();
    writeln!(out, "  }} catch (const std::exception &e) {{").unwrap();
    writeln!(
        out,
        "    fprintf(stderr, \"debug launch exception: %s\\n\", e.what()); fflush(stderr);"
    )
    .unwrap();
    writeln!(out, "    return -2;").unwrap();
    writeln!(out, "  }} catch (...) {{ return -3; }}").unwrap();
    writeln!(out, "}}").unwrap();

    let _ = (nl, nbh, gqa_ratio);
    out
}

/// Generate a standalone CUDA test kernel for a single TK op.
///
/// The generated file contains:
/// 1. Full preamble + includes (same as static megakernel)
/// 2. `run_op` template
/// 3. Model constants
/// 4. A `__global__ void test_{op_name}(globals g, int batch_size, int num_layers)`
///    kernel that sets up state and runs ONE op for all tiles in layer 0
/// 5. A C launch wrapper `extern "C" int test_{op_name}_launch(...)`
///
/// This enables testing each op in isolation on the GPU.
pub fn generate_single_op_kernel(dag: &ModelDag, op_name: &str) -> Result<String, String> {
    let mappings = op_mappings();
    let mapping = mappings
        .get(op_name)
        .ok_or_else(|| format!("unknown op: {op_name}"))?;

    let mut out = String::new();

    // ── Preamble (same includes as full megakernel) ──
    emit_preamble(&mut out, dag);
    emit_run_op_template(&mut out);
    let (_nl, _hd, _id, _hdm, _nah, _nkh, _vs) = emit_model_constants(&mut out, dag);
    emit_optimal_out_block(&mut out);

    // ── Test kernel: runs ONE op across all tiles for layer 0 ──
    writeln!(
        out,
        "__global__ void __launch_bounds__(config::NUM_THREADS, 1)"
    )
    .unwrap();
    writeln!(
        out,
        "test_{op_name}(const globals g, int batch_size, int num_layers) {{"
    )
    .unwrap();
    writeln!(out, "    const int sm = blockIdx.x;").unwrap();
    writeln!(out, "    const int sm_count = gridDim.x;").unwrap();
    writeln!(out).unwrap();

    emit_tile_count_vars(&mut out, "batch_size");
    emit_state_setup(&mut out);

    // Determine tile count and row/col expressions based on op type
    let (tile_count, row_expr, col_expr) = match op_name {
        "attn_norm" | "mlp_norm" | "lm_head_norm" => ("batch_size".to_string(), "t", "0"),
        "qkv_rope_append" => ("batch_size * NBH".to_string(), "t / NBH", "t % NBH"),
        "attention_decode" => ("n_attn_bb".to_string(), "t", "0"),
        "o_proj_residual" => (
            "batch_size * n_cols_hd".to_string(),
            "t / n_cols_hd",
            "t % n_cols_hd",
        ),
        "gate_silu" | "up_matmul" | "down_proj_residual" => (
            "batch_size * n_cols_id".to_string(),
            "t / n_cols_id",
            "t % n_cols_id",
        ),
        "lm_head" => (
            "batch_size * n_cols_vs".to_string(),
            "t / n_cols_vs",
            "t % n_cols_vs",
        ),
        "attention_prefill" => {
            // Prefill uses run_op_ext, not run_op — handled separately below
            ("0".to_string(), "0", "0")
        }
        _ => return Err(format!("unsupported op for single-op test: {op_name}")),
    };

    if op_name == "attention_prefill" {
        // Prefill attention uses run_op_ext with different instruction layout.
        // For single-op testing, we run it with seq_idx=0, prefill_block_idx=t, etc.
        writeln!(
            out,
            "    // attention_prefill: simplified single-sequence test"
        )
        .unwrap();
        writeln!(out, "    const int total = NKH;  // one tile per KV head").unwrap();
        writeln!(out, "    const int my_start = sm * total / sm_count;").unwrap();
        writeln!(out, "    const int my_end = (sm + 1) * total / sm_count;").unwrap();
        writeln!(out, "    for (int t = my_start; t < my_end; t++) {{").unwrap();
        writeln!(
            out,
            "        run_op_ext<{}>(\n            g, kvms, {}, 0, 0, 0, t, 0);",
            mapping.cpp_type, mapping.opcode
        )
        .unwrap();
        writeln!(out, "    }}").unwrap();
    } else {
        writeln!(out, "    const int layer = 0;").unwrap();
        emit_op_tile_loop(&mut out, "    ", mapping, &tile_count, row_expr, col_expr);
    }

    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    // ── C launch wrapper ──
    emit_tensor_arg_and_globals_helper(&mut out);
    writeln!(out, "extern \"C\" int test_{op_name}_launch(").unwrap();
    writeln!(out, "{}", LAUNCH_PARAMS).unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out, "  try {{").unwrap();
    emit_globals_construction(&mut out, "    ");
    writeln!(out).unwrap();
    writeln!(out, "    int shmem = g.dynamic_shared_memory();").unwrap();
    writeln!(out, "    auto err = cudaFuncSetAttribute(test_{op_name},").unwrap();
    writeln!(
        out,
        "        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);"
    )
    .unwrap();
    writeln!(out, "    if (err != cudaSuccess) return (int)err;").unwrap();
    writeln!(
        out,
        "    test_{op_name}<<<g.grid(), g.block(), shmem, (cudaStream_t)stream>>>("
    )
    .unwrap();
    writeln!(out, "        g, batch_size, num_layers);").unwrap();
    writeln!(out, "    err = cudaGetLastError();").unwrap();
    writeln!(out, "    return (int)err;").unwrap();
    writeln!(out, "  }} catch (const std::exception &e) {{").unwrap();
    writeln!(
        out,
        "    fprintf(stderr, \"test_{op_name} C++ exception: %s\\n\", e.what()); fflush(stderr);"
    )
    .unwrap();
    writeln!(out, "    return -2;").unwrap();
    writeln!(out, "  }} catch (...) {{").unwrap();
    writeln!(
        out,
        "    fprintf(stderr, \"test_{op_name} unknown C++ exception\\n\"); fflush(stderr);"
    )
    .unwrap();
    writeln!(out, "    return -3;").unwrap();
    writeln!(out, "  }}").unwrap();
    writeln!(out, "}}").unwrap();

    Ok(out)
}

/// List all op names that can be used with `generate_single_op_kernel`.
pub fn available_op_names() -> Vec<&'static str> {
    vec![
        "attn_norm",
        "qkv_rope_append",
        "attention_decode",
        "o_proj_residual",
        "mlp_norm",
        "gate_silu",
        "up_matmul",
        "down_proj_residual",
        "lm_head_norm",
        "lm_head",
        "attention_prefill",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_valid_structure() {
        let input: proc_macro2::TokenStream = quote::quote! {
            kernel llama_sm89<NL=16, HD=2048, ID=5632, HDM=64, NAH=32, NKH=8, VS=128256> {
                for layer in 0..NL {
                    let normed = rmsnorm(hidden_states, attn_norm[layer]);
                    let qkv = gemm(normed, qkv_weights[layer]);
                    let (q, k, v) = rope_append(qkv, positions, kv_cache[layer]);
                    let attn = attention_decode(q, k, v, kv_cache[layer], block_table);
                    hidden_states = gemm_add(attn, o_proj[layer], hidden_states);

                    let normed2 = rmsnorm(hidden_states, mlp_norm[layer]);
                    let gate = silu(gemm(normed2, gate_weights[layer]));
                    let up = gemm(normed2, up_weights[layer]);
                    hidden_states = gemm_add(gate * up, down_proj[layer], hidden_states);
                }
                let normed = rmsnorm(hidden_states, lm_head_norm);
                logits = gemm(normed, lm_head);
            }
        };
        let def: crate::parse::MegakernelDef = syn::parse2(input).unwrap();
        let dag = crate::parse::build_dag(&def).unwrap();

        let cuda = generate_static_kernel(&dag);

        // Shared preamble
        assert!(cuda.contains("GENERATED by megakernel!"));
        assert!(cuda.contains("run_op<"));
        assert!(cuda.contains("run_op_ext<"));
        assert!(cuda.contains("Op::consumer::run(g, kvms)"));
        assert!(cuda.contains("Op::loader::run(g, kvms)"));
        assert!(cuda.contains("Identity page mapping"));

        // Model constants (NL is runtime, not emitted as constexpr)
        assert!(!cuda.contains("static constexpr int NL"));
        assert!(cuda.contains("static constexpr int HD = 2048;"));
        assert!(cuda.contains("static constexpr int NBH = 48;"));

        // ── Decode kernel ──
        assert!(cuda.contains(
            "void llama_sm89_decode_static(const globals g, int batch_size, int num_layers)"
        ));
        assert!(cuda.contains("OPCODE_GQA_AttentionDecode"));
        assert!(cuda.contains("t / NKH, t % NKH"));

        // ── Prefill kernel ──
        assert!(cuda.contains("void llama_sm89_prefill_static("));
        assert!(cuda.contains("int total_tokens,"));
        assert!(cuda.contains("int num_seqs,"));
        assert!(cuda.contains("OPCODE_GQA_AttentionPrefill"));
        assert!(cuda.contains("n_matmul_blocks"));
        assert!(cuda.contains("seq_chunk_lens[seq]"));
        assert!(cuda.contains("seq_extend_offsets[seq]"));
        assert!(cuda.contains("run_op_ext<"));

        // Both kernels have layer loop
        // (appears twice — once in decode, once in prefill)
        assert_eq!(
            cuda.matches("for (int layer = 0; layer < num_layers; layer++)")
                .count(),
            2
        );

        // SM tile distribution
        assert!(cuda.contains("const int sm = blockIdx.x;"));
        assert!(cuda.contains("const int my_start = sm * total / sm_count;"));

        // Both launch wrappers with flat TkTensorArg args
        assert!(cuda.contains("_decode_static_launch("));
        assert!(cuda.contains("_prefill_static_launch("));
        assert!(cuda.contains("struct TkTensorArg"));
        assert!(cuda.contains("make_arg<G::weights_t>"));
        assert!(cuda.contains("make_arg<G::kv_cache_t>"));
        assert!(cuda.contains("g.dynamic_shared_memory()"));

        // No VM dispatch
        assert!(!cuda.contains("dispatch_op"));
        assert!(!cuda.contains("load_instructions"));
        assert!(!cuda.contains("MAKE_WORKER"));

        // Print for inspection
        eprintln!("{cuda}");
    }

    #[test]
    fn generates_single_op_kernels() {
        let input: proc_macro2::TokenStream = quote::quote! {
            kernel llama_sm89<NL=16, HD=2048, ID=5632, HDM=64, NAH=32, NKH=8, VS=128256> {
                for layer in 0..NL {
                    let normed = rmsnorm(hidden_states, attn_norm[layer]);
                    let qkv = gemm(normed, qkv_weights[layer]);
                    let (q, k, v) = rope_append(qkv, positions, kv_cache[layer]);
                    let attn = attention_decode(q, k, v, kv_cache[layer], block_table);
                    hidden_states = gemm_add(attn, o_proj[layer], hidden_states);

                    let normed2 = rmsnorm(hidden_states, mlp_norm[layer]);
                    let gate = silu(gemm(normed2, gate_weights[layer]));
                    let up = gemm(normed2, up_weights[layer]);
                    hidden_states = gemm_add(gate * up, down_proj[layer], hidden_states);
                }
                let normed = rmsnorm(hidden_states, lm_head_norm);
                logits = gemm(normed, lm_head);
            }
        };
        let def: crate::parse::MegakernelDef = syn::parse2(input).unwrap();
        let dag = crate::parse::build_dag(&def).unwrap();

        // Test each op generates valid CUDA
        for op_name in available_op_names() {
            let cuda = generate_single_op_kernel(&dag, op_name)
                .unwrap_or_else(|e| panic!("failed for {op_name}: {e}"));

            // Must have preamble
            assert!(
                cuda.contains("GENERATED by megakernel!"),
                "preamble missing for {op_name}"
            );
            // Must have test kernel
            assert!(
                cuda.contains(&format!("test_{op_name}(")),
                "test kernel missing for {op_name}"
            );
            // Must have launch wrapper
            assert!(
                cuda.contains(&format!("test_{op_name}_launch(")),
                "launch wrapper missing for {op_name}"
            );
            // Must have run_op or run_op_ext
            assert!(
                cuda.contains("run_op<") || cuda.contains("run_op_ext<"),
                "run_op missing for {op_name}"
            );
        }

        // Verify attn_norm specifically
        let norm_cuda = generate_single_op_kernel(&dag, "attn_norm").unwrap();
        assert!(norm_cuda.contains("OPCODE_AttnNorm"));
        assert!(norm_cuda.contains("attn_norm<config, globals>"));

        // Verify attention_prefill uses run_op_ext
        let prefill_cuda = generate_single_op_kernel(&dag, "attention_prefill").unwrap();
        assert!(prefill_cuda.contains("run_op_ext<"));
        assert!(prefill_cuda.contains("OPCODE_GQA_AttentionPrefill"));
    }
}
