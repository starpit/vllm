// C-linkage wrapper for the TK KVM LLaMA sm89 megakernel.
// Allows Rust (or any C caller) to launch the kernel without matching
// the C++ template types. Each gl<> field is passed as a TkTensorArg
// struct (ptr + 4 dims), and we reconstruct the globals struct here.

#include "llama_sm89.cuh"

#include "rms_norm_sm89.cu"
#include "qkv_rope_append_sm89.cu"
#include "attention_prefill_sm89.cu"
#include "attention_decode_sm89.cu"
#include "matmul_adds_sm89.cu"
#include "gate_silu_sm89.cu"
#include "up_matmul_sm89.cu"
#include "lm_head_sm89.cu"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

// Concrete op types for the sm89 KVM.
using attn_norm_op     = attn_norm    <llama_sm89_config, llama_sm89_globals>;
using qkv_op           = qkv_rope_append<llama_sm89_config, llama_sm89_globals>;
using attn_prefill_op  = attention_prefill<llama_sm89_config, llama_sm89_globals>;
using attn_op          = attention_decode<llama_sm89_config, llama_sm89_globals>;
using o_proj_op        = o_proj       <llama_sm89_config, llama_sm89_globals>;
using mlp_norm_op      = mlp_norm     <llama_sm89_config, llama_sm89_globals>;
using gate_silu_op     = gate_silu    <llama_sm89_config, llama_sm89_globals>;
using up_matmul_op     = up_matmul    <llama_sm89_config, llama_sm89_globals>;
using downproj_op      = downproj     <llama_sm89_config, llama_sm89_globals>;
using lm_head_norm_op  = lm_head_norm <llama_sm89_config, llama_sm89_globals>;
using lm_head_op       = lm_head      <llama_sm89_config, llama_sm89_globals>;

// The kernel function pointer (needed for cudaFuncSetAttribute).
static auto kvm_kernel = kvm<llama_sm89_config,
    llama_sm89_globals,
    attn_norm_op, qkv_op, attn_prefill_op, attn_op, o_proj_op, mlp_norm_op,
    gate_silu_op, up_matmul_op, downproj_op, lm_head_norm_op, lm_head_op>;

// Flat tensor descriptor passed from Rust via FFI.
struct TkTensorArg {
    uint64_t ptr;
    int b, d, r, c;
};

// Helper: construct a gl<> from a TkTensorArg.
template<typename GL>
static inline GL make_arg(const TkTensorArg &a) {
    return make_gl<GL>(a.ptr, a.b, a.d, a.r, a.c);
}

extern "C" void tk_llama_1b_launch(
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
    int num_pages, int batch_size, int num_prefill_tokens,
    // CUDA stream
    uint64_t stream
) {
    using G = llama_sm89_globals;

    G g {
        // VM state
        make_arg<G::barriers>(bar),
        make_arg<G::instruction_layout>(instructions),
        make_arg<G::timing_layout>(timings),

        // Weights
        make_arg<G::weights_t>(qkv_w),
        make_arg<G::norm_weights_t>(attn_norm_w),
        make_arg<G::weights_t>(o_w),
        make_arg<G::norm_weights_t>(mlp_norm_w),
        make_arg<G::weights_t>(up_w),
        make_arg<G::weights_t>(gate_w),
        make_arg<G::weights_big_t>(down_w),
        make_arg<G::norm_weights_t>(lm_norm_w),
        make_arg<G::weights_t>(lm_w),

        // KV cache
        make_arg<G::kv_cache_t>(k_cache),
        make_arg<G::kv_cache_t>(v_cache),

        // RoPE
        make_arg<G::rope_table_t>(rope_cos),
        make_arg<G::rope_table_t>(rope_sin),

        // Activations
        make_arg<G::activations_t>(hidden),
        make_arg<G::activations_t>(rms_rope),
        make_arg<G::activations_t>(rms_gate),
        make_arg<G::activations_t>(q_post),
        make_arg<G::activations_t>(attn_out),
        make_arg<G::activations_big_t>(silu),
        make_arg<G::activations_t>(rms_lm),
        make_arg<G::logits_t>(logits_arg),

        // Paged KV metadata
        make_arg<G::int32_vector_t>(pos_ids),
        make_arg<G::int32_vector_t>(kv_indptr),
        make_arg<G::int32_vector_t>(kv_indices),
        make_arg<G::int32_vector_t>(kv_last_page),
        make_arg<G::int32_vector_t>(kv_append),

        // Prefill KV metadata
        make_arg<G::int32_vector_t>(prefill_qo_indptr),
        make_arg<G::int32_vector_t>(prefill_kv_indptr),
        make_arg<G::int32_vector_t>(prefill_kv_indices),
        make_arg<G::int32_vector_t>(prefill_kv_last_page_len),
        num_prefill_tokens,

        // Scalars
        attn_scale,
        rms_norm_eps,
        num_pages,
        batch_size,
    };

    int dynamic_shmem = g.dynamic_shared_memory();
    cudaError_t attr_err = cudaFuncSetAttribute(kvm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shmem);
    if (attr_err != cudaSuccess) {
        printf("TK LAUNCH ERROR: cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(attr_err));
        return;
    }
    kvm_kernel<<<g.grid(), g.block(), dynamic_shmem, (cudaStream_t)stream>>>(g);
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        printf("TK LAUNCH ERROR: kernel launch failed: %s\n", cudaGetErrorString(launch_err));
    }
}
