// SPDX-License-Identifier: Apache-2.0
//
// Phase A.2 standalone shim around FlashInfer's
//   BlockBatchPagedAttentionPersistent::Run
// runner. Single extern "C" entry point that:
//   1. Builds host-side qo_indptr / kv_indptr / kv_len arrays for one
//      sequence (batch_size = 1).
//   2. Allocates float + int workspace buffers (page-locked staging mirror
//      for the int workspace, per FlashInfer's planner contract).
//   3. Calls TwoStageHolisticPlan to fill the workspace with all the
//      indirection arrays the persistent runner needs.
//   4. Builds two PersistentParams (the long-Q / short-Q dual-config dispatch).
//   5. Launches BatchPagedAttentionPersistent.
//
// This is NOT integrated into the scheduled megakernel. It's a standalone
// proof-of-correctness path: smoke-test the FlashInfer runner end to end
// against our existing dims / layouts before we touch tile_attention.
//
// Inputs/outputs use the layout the scheduled megakernel will eventually
// hand us:
//   q : [seq_len, num_qo_heads, head_dim]                          NHD
//   k : [num_pages, page_size, num_kv_heads, head_dim]             NHD
//   v : [num_pages, page_size, num_kv_heads, head_dim]             NHD
//   o : [seq_len, num_qo_heads, head_dim]                          NHD
//   kv_indices : [num_pages]                                       int32
//
// Single sequence (batch_size = 1), causal mask, no logits soft cap.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "flashinfer_inst/batch_attention_config.inc"

#include <flashinfer/attention/persistent.cuh>
#include <flashinfer/attention/scheduler.cuh>

using namespace flashinfer;

namespace {

// Workspaces are sized generously — we only run this on small (≤1024 token,
// 1B-class) prefill in the smoke test, so 16 MiB each is plenty and the
// alloc/free cost is hidden behind the CPU reference run.
constexpr size_t kFloatWorkspaceBytes = 16ull * 1024 * 1024;
constexpr size_t kIntWorkspaceBytes = 16ull * 1024 * 1024;

// Cast helper that adds a byte offset to a void* and reinterprets.
template <typename T>
__host__ __device__ inline T* offset_ptr(void* base, size_t byte_offset) {
  return reinterpret_cast<T*>(static_cast<uint8_t*>(base) + byte_offset);
}

}  // namespace

// Result codes for the Rust caller — keep these in sync with the FFI
// binding in the test file.
enum class FlashInferShimStatus : int32_t {
  Ok = 0,
  CudaMallocFailed = 1,
  CudaMallocHostFailed = 2,
  PlanFailed = 3,
  RunFailed = 4,
  CudaSyncFailed = 5,
};

extern "C" int32_t run_flashinfer_attention_smoke(
    DTypeQ* q,                  // device  [seq_len, num_qo_heads, head_dim]
    DTypeKV* k,                 // device  [num_pages, page_size, num_kv_heads, head_dim]
    DTypeKV* v,                 // device  same layout as k
    int32_t* kv_indices,        // device  [num_pages]
    DTypeO* o,                  // device  [seq_len, num_qo_heads, head_dim]
    int32_t seq_len, int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t num_pages, float sm_scale, cudaStream_t stream) {
  if (head_dim != HEAD_DIM_QK) {
    std::fprintf(stderr,
                 "flashinfer_attention_shim: head_dim=%d does not match "
                 "compiled HEAD_DIM_QK=%d\n",
                 head_dim, HEAD_DIM_QK);
    return static_cast<int32_t>(FlashInferShimStatus::RunFailed);
  }

  // ── Workspace allocation ──
  void* float_ws_d = nullptr;
  void* int_ws_d = nullptr;
  void* int_ws_h = nullptr;  // page-locked mirror — required by TwoStageHolisticPlan

  if (cudaMalloc(&float_ws_d, kFloatWorkspaceBytes) != cudaSuccess) {
    return static_cast<int32_t>(FlashInferShimStatus::CudaMallocFailed);
  }
  if (cudaMalloc(&int_ws_d, kIntWorkspaceBytes) != cudaSuccess) {
    cudaFree(float_ws_d);
    return static_cast<int32_t>(FlashInferShimStatus::CudaMallocFailed);
  }
  if (cudaMallocHost(&int_ws_h, kIntWorkspaceBytes) != cudaSuccess) {
    cudaFree(float_ws_d);
    cudaFree(int_ws_d);
    return static_cast<int32_t>(FlashInferShimStatus::CudaMallocHostFailed);
  }

  // ── Plan inputs (host arrays for a single-sequence batch) ──
  IdType qo_indptr_h[2] = {0, seq_len};
  IdType kv_indptr_h[2] = {0, num_pages};
  IdType kv_len_h[1] = {seq_len};

  HolisticPlanInfo<2> plan_info;
  cudaError_t status = TwoStageHolisticPlan<IdType>(
      float_ws_d, kFloatWorkspaceBytes, int_ws_d, int_ws_h, kIntWorkspaceBytes, plan_info,
      qo_indptr_h, kv_indptr_h, kv_len_h, /*batch_size=*/1, num_qo_heads, num_kv_heads, head_dim,
      /*causal=*/true, stream);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "TwoStageHolisticPlan failed: %s\n", cudaGetErrorString(status));
    cudaFree(float_ws_d);
    cudaFree(int_ws_d);
    cudaFreeHost(int_ws_h);
    return static_cast<int32_t>(FlashInferShimStatus::PlanFailed);
  }

  // ── Build PersistentParams for both dispatch tasks (long-Q + short-Q). ──
  PersistentParams params[2] = {};
  IdType* len_kv_chunk_d = offset_ptr<IdType>(int_ws_d, plan_info.len_kv_chunk_offset);

  for (int i = 0; i < 2; ++i) {
    params[i].q = q;
    params[i].k = k;
    params[i].v = v;
    params[i].kv_indices = kv_indices;
    params[i].o = o;
    params[i].final_o = o;
    params[i].final_lse = nullptr;
    params[i].partial_o = offset_ptr<DTypeO>(float_ws_d, plan_info.partial_o_offset);
    params[i].partial_lse = offset_ptr<float>(float_ws_d, plan_info.partial_lse_offset);

    params[i].q_indptr = offset_ptr<IdType>(int_ws_d, plan_info.tasks[i].q_indptr_offset);
    params[i].kv_indptr = offset_ptr<IdType>(int_ws_d, plan_info.tasks[i].kv_indptr_offset);
    params[i].partial_indptr = offset_ptr<IdType>(int_ws_d, plan_info.tasks[i].partial_indptr_offset);
    params[i].q_len = offset_ptr<IdType>(int_ws_d, plan_info.tasks[i].q_len_offset);
    params[i].kv_len = offset_ptr<IdType>(int_ws_d, plan_info.tasks[i].kv_len_offset);
    params[i].q_start = offset_ptr<IdType>(int_ws_d, plan_info.tasks[i].q_start_offset);
    params[i].kv_start = offset_ptr<IdType>(int_ws_d, plan_info.tasks[i].kv_start_offset);
    params[i].kv_end = offset_ptr<IdType>(int_ws_d, plan_info.tasks[i].kv_end_offset);
    params[i].kv_head_idx_arr = offset_ptr<IdType>(int_ws_d, plan_info.tasks[i].kv_head_idx_offset);
    params[i].work_indptr = offset_ptr<IdType>(int_ws_d, plan_info.tasks[i].work_indptr_offset);
    params[i].len_kv_chunk = len_kv_chunk_d + i;

    params[i].merge_indptr = offset_ptr<IdType>(int_ws_d, plan_info.merge_indptr_offset);
    params[i].merge_o_indices = offset_ptr<IdType>(int_ws_d, plan_info.merge_o_indices_offset);
    params[i].num_packed_qo_len = offset_ptr<IdType>(int_ws_d, plan_info.num_qo_len_offset);

    params[i].num_kv_heads = static_cast<uint32_t>(num_kv_heads);
    params[i].gqa_group_size = uint_fastdiv(static_cast<uint32_t>(num_qo_heads / num_kv_heads));
    params[i].page_size = uint_fastdiv(static_cast<uint32_t>(page_size));

    // NHD layout strides.
    params[i].q_stride_n = static_cast<uint32_t>(num_qo_heads * head_dim);
    params[i].q_stride_h = static_cast<uint32_t>(head_dim);
    params[i].k_stride_page = static_cast<uint32_t>(page_size * num_kv_heads * head_dim);
    params[i].k_stride_n = static_cast<uint32_t>(num_kv_heads * head_dim);
    params[i].k_stride_h = static_cast<uint32_t>(head_dim);
    params[i].v_stride_page = static_cast<uint32_t>(page_size * num_kv_heads * head_dim);
    params[i].v_stride_n = static_cast<uint32_t>(num_kv_heads * head_dim);
    params[i].v_stride_h = static_cast<uint32_t>(head_dim);

    params[i].sm_scale = sm_scale;
    params[i].v_scale = 1.0f;
    params[i].logits_soft_cap = 0.0f;
  }

  // ── Launch the persistent runner. ──
  status = BatchPagedAttentionPersistent<
      /*CTA_TILE_Q_1=*/128, /*CTA_TILE_Q_2=*/16, HEAD_DIM_QK, HEAD_DIM_VO, MaskMode::kCausal,
      StandardAttention</*UseLogitsSoftCap=*/false>, PersistentParams>(
      params[0], params[1], plan_info.num_blks_x, plan_info.num_blks_y, stream);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "BatchPagedAttentionPersistent failed: %s\n", cudaGetErrorString(status));
    cudaFree(float_ws_d);
    cudaFree(int_ws_d);
    cudaFreeHost(int_ws_h);
    return static_cast<int32_t>(FlashInferShimStatus::RunFailed);
  }

  status = cudaStreamSynchronize(stream);
  cudaFree(float_ws_d);
  cudaFree(int_ws_d);
  cudaFreeHost(int_ws_h);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaStreamSynchronize failed: %s\n", cudaGetErrorString(status));
    return static_cast<int32_t>(FlashInferShimStatus::CudaSyncFailed);
  }

  return static_cast<int32_t>(FlashInferShimStatus::Ok);
}
