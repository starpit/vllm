// SPDX-License-Identifier: Apache-2.0
//
// Hand-rendered output of FlashInfer's
//   csrc/batch_attention_customize_config.jinja
// at commit 08ab45d67705b301ee66e63c6999c934c72dd41c, for the LLaMA-3.2-1B
// prefill config:
//
//   DTypeQ = DTypeKV = DTypeO = __nv_bfloat16
//   IdType                    = int32_t
//   HEAD_DIM_QK = HEAD_DIM_VO = 64
//   POS_ENCODING_MODE         = PosEncodingMode::kNone
//   AttentionVariant          = StandardAttention<false>  (no logits soft cap)
//   MASK_MODE                 = MaskMode::kCausal
//
// When you bump the FLASHINFER_COMMIT pin in vllm-tk-test-harness/build.rs
// you must re-render this file from the upstream jinja template (or check
// that nothing relevant changed). The PersistentParams struct definition
// is COUPLED to the field set that BlockBatchPagedAttentionPersistent::Run
// reads — additions / renames upstream will silently break this shim.
//
// The longer-term plan is to render this file at build time via minijinja
// (already a workspace dep), once we extend the cudaforge sparse checkout
// to include FlashInfer's csrc/ directory. For now: hand-rendered, vendored,
// pinned by SHA.
#pragma once

// Match the include order from FlashInfer's
// csrc/batch_attention_customize_config.jinja exactly. Reordering trips
// inter-header dependencies (variant_helper.cuh references math::, etc).
#include <flashinfer/page.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/utils.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/fastdiv.cuh>
#include <flashinfer/attention/variant_helper.cuh>
#include <flashinfer/profiler.cuh>

using namespace flashinfer;

// StandardAttention is built into flashinfer (variant_helper.cuh) — no
// per-config decl needed. Reproduce just enough of it here so this .inc
// is self-sufficient as the param-set definition for our shim TU.
template <bool UseLogitsSoftCap>
struct StandardAttention : AttentionVariantBase {
  float sm_scale_log2;
  float soft_cap_pre_tanh_scale;
  static constexpr bool use_logits_soft_cap = UseLogitsSoftCap;

  template <typename Params>
  __device__ __host__ StandardAttention(const Params& params, uint32_t /*batch_idx*/,
                                        uint8_t* /*smem_ptr*/) {
    if constexpr (UseLogitsSoftCap) {
      soft_cap_pre_tanh_scale = params.sm_scale * math::ptx_rcp(params.logits_soft_cap);
      sm_scale_log2 = math::log2e * params.logits_soft_cap;
    } else {
      sm_scale_log2 = params.sm_scale * math::log2e;
    }
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    if constexpr (UseLogitsSoftCap) {
      logits = float(math::tanh(logits * soft_cap_pre_tanh_scale));
    }
    return logits;
  })
};

using DTypeQ = __nv_bfloat16;
using DTypeKV = __nv_bfloat16;
using DTypeO = __nv_bfloat16;
using IdType = int32_t;

constexpr int HEAD_DIM_QK = 64;
constexpr int HEAD_DIM_VO = 64;
constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;

struct PersistentParams {
  using DTypeQ = ::DTypeQ;
  using DTypeKV = ::DTypeKV;
  using DTypeO = ::DTypeO;
  using IdType = ::IdType;

  DTypeQ* q;
  DTypeKV* k;
  DTypeKV* v;
  DTypeO* o;
  DTypeO* partial_o;
  float* partial_lse;
  DTypeO* final_o;
  float* final_lse;

  IdType* q_indptr;
  IdType* kv_indptr;
  IdType* partial_indptr;
  IdType* kv_indices;
  IdType* q_len;
  IdType* kv_len;
  IdType* q_start;
  IdType* kv_start;
  IdType* kv_end;
  IdType* kv_head_idx_arr;
  IdType* work_indptr;
  IdType* len_kv_chunk;

  // for state reduction
  IdType* merge_indptr;
  IdType* merge_o_indices;
  IdType* num_packed_qo_len;

  uint32_t num_kv_heads;
  uint_fastdiv gqa_group_size;
  uint_fastdiv page_size;

  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t k_stride_page;
  uint32_t k_stride_h;
  uint32_t k_stride_n;
  uint32_t v_stride_page;
  uint32_t v_stride_h;
  uint32_t v_stride_n;

  float sm_scale;
  float logits_soft_cap;
  float v_scale;

  PROFILER_CLOSURE_PARAMS_DECL
};
