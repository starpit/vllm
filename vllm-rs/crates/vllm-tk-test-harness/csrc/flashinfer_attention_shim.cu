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
#include <vector>

#include "flashinfer_inst/batch_attention_config.inc"

#include <flashinfer/attention/persistent.cuh>
#include <flashinfer/attention/scheduler.cuh>

using namespace flashinfer;

namespace {

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
    int32_t page_size, int32_t num_pages, size_t float_ws_bytes, size_t int_ws_bytes,
    float sm_scale, cudaStream_t stream) {
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

  if (cudaMalloc(&float_ws_d, float_ws_bytes) != cudaSuccess) {
    return static_cast<int32_t>(FlashInferShimStatus::CudaMallocFailed);
  }
  if (cudaMalloc(&int_ws_d, int_ws_bytes) != cudaSuccess) {
    cudaFree(float_ws_d);
    return static_cast<int32_t>(FlashInferShimStatus::CudaMallocFailed);
  }
  if (cudaMallocHost(&int_ws_h, int_ws_bytes) != cudaSuccess) {
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
      float_ws_d, float_ws_bytes, int_ws_d, int_ws_h, int_ws_bytes, plan_info,
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

// ─────────────────────────────────────────────────────────────────────
// Forked planner (Phase C2b-step-3).
//
// This is a vendored copy of `flashinfer::TwoStageHolisticPlan` from
// `flashinfer/attention/scheduler.cuh`, with one change: `num_sm`
// is an explicit caller-provided parameter instead of being computed
// as `cudaDeviceGetAttribute(MultiProcessorCount) * 2`.
//
// Why fork: the upstream planner hardcodes the assumption that 2
// CTAs fit per SM under cooperative-launch residency. That's true
// for FlashInfer's standalone attention kernel (small per-CTA
// resource footprint), but FALSE for our megakernel — the
// megakernel's CTA holds enough dynamic shmem (FlashInfer's
// KTraits1::SharedStorage plus the existing tile body arena) that
// only **one** CTA fits per SM on L4. The schedule reflects this
// via `TargetProfile::cooperative_blocks_per_sm`, and the planner
// must agree or its `work_indptr` slots beyond `num_sm * 1` go
// unconsumed (producing wrong attention output).
//
// Vendored from FlashInfer SHA pinned in vllm-tk-test-harness/build.rs.
// Re-vendor when the SHA bumps. The only diff from upstream is:
//   - new `int num_sm_in` parameter at the end
//   - the cudaDeviceGetAttribute + doubling block replaced with
//     `int num_sm = num_sm_in;`
template <typename IdType>
inline cudaError_t TwoStageHolisticPlanWithNumSm(
    void* float_buffer, size_t float_workspace_size_in_bytes, void* int_buffer,
    void* page_locked_int_buffer, size_t int_workspace_size_in_bytes,
    HolisticPlanInfo<2>& plan_info, IdType* qo_indptr_h, IdType* kv_indptr_h,
    IdType* kv_len_arr_h, uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t head_dim, bool causal, cudaStream_t stream, int num_sm_in) {
  constexpr uint32_t NUM_TASKS = 2;
  const uint32_t CTA_TILE_Q_SIZES[NUM_TASKS] = {128, 16};

  uint32_t gqa_group_size = num_qo_heads / num_kv_heads;
  // ── DIFF FROM UPSTREAM: num_sm is caller-supplied. ──
  // Upstream queries cudaDevAttrMultiProcessorCount and then doubles
  // (assuming 2 CTAs/SM cooperative residency). We take the value
  // from the Rust-side TargetProfile, which already factors in our
  // megakernel's actual residency.
  int num_sm = num_sm_in;

  // step 0. determine the number of blocks in x and y dimensions
  std::vector<std::tuple<int, int, int>> idx_qo_kv_len_vec[NUM_TASKS];
  for (uint32_t i = 0; i < batch_size; ++i) {
    if (qo_indptr_h[i + 1] - qo_indptr_h[i] < 0) {
      std::ostringstream err_msg;
      err_msg << "qo_indptr[" << i + 1 << "]" << qo_indptr_h[i + 1] << " - qo_indptr[" << i
              << "]" << qo_indptr_h[i] << " should be non-negative";
      FLASHINFER_ERROR(err_msg.str());
    }

    int qo_len = qo_indptr_h[i + 1] - qo_indptr_h[i];
    int packed_qo_len = qo_len * gqa_group_size;
    int kv_len = kv_len_arr_h[i];

    if (packed_qo_len > CTA_TILE_Q_SIZES[1]) {
      idx_qo_kv_len_vec[0].push_back({i, qo_len, kv_len});
    } else {
      idx_qo_kv_len_vec[1].push_back({i, qo_len, kv_len});
    }
  }

  int cluster_size = 1;
  int num_clusters = num_sm / cluster_size;
  plan_info.num_blks_x = cluster_size;
  plan_info.num_blks_y = num_clusters;

  auto f = [](int x) {
    if (x <= 128) {
      return 128;
    }
    return ceil_div(x, 256) * 256;
  };

  MinHeap cluster_cost_heap(num_clusters);
  AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);

  const int max_total_num_works = 65536;
  const int max_num_kv_splits =
      4 * num_clusters * cluster_size * (CTA_TILE_Q_SIZES[0] + CTA_TILE_Q_SIZES[1]);

  int64_t total_kv_lens = 0;
  for (uint32_t task = 0; task < NUM_TASKS; ++task) {
    int cluster_tile_q = CTA_TILE_Q_SIZES[task] * cluster_size;
    for (auto& [_, qo_len, kv_len] : idx_qo_kv_len_vec[task]) {
      int packed_qo_len = qo_len * gqa_group_size;
      int num_qo_tiles = ceil_div(packed_qo_len, cluster_tile_q);
      for (int qo_tile_idx = num_qo_tiles - 1; qo_tile_idx >= 0; --qo_tile_idx) {
        int effective_kv_len =
            causal ? packed_causal_kv_end(qo_len, kv_len, qo_tile_idx, cluster_tile_q,
                                          num_qo_tiles, gqa_group_size)
                   : kv_len;
        total_kv_lens += effective_kv_len;
      }
    }
  }

  int partial_o_nnz = 0;
  std::vector<IdType> merge_indptr, merge_o_indices, num_expand_qo_len_vec;
  std::vector<IdType> cluster_len_kv_chunk(NUM_TASKS, 0);
  merge_indptr.push_back(partial_o_nnz);
  for (uint32_t task = 0; task < NUM_TASKS; ++task) {
    int cluster_tile_q = CTA_TILE_Q_SIZES[task] * cluster_size;
    int kv_len_limit = f(std::max(ceil_div(total_kv_lens * num_kv_heads, num_clusters), 1L));
    if (cluster_tile_q >= 64) {
      kv_len_limit /= std::min(num_kv_heads, 2U);
    }
    cluster_len_kv_chunk[task] = kv_len_limit;
    std::vector<std::vector<IdType>> cluster_q_indptr(num_clusters, std::vector<IdType>()),
        cluster_kv_indptr(num_clusters, std::vector<IdType>()),
        cluster_q_len(num_clusters, std::vector<IdType>()),
        cluster_kv_len(num_clusters, std::vector<IdType>()),
        cluster_q_start(num_clusters, std::vector<IdType>()),
        cluster_kv_start(num_clusters, std::vector<IdType>()),
        cluster_kv_end(num_clusters, std::vector<IdType>()),
        cluster_kv_head_idx(num_clusters, std::vector<IdType>()),
        cluster_partial_indptr(num_clusters, std::vector<IdType>());

    for (auto& [i, qo_len, kv_len] : idx_qo_kv_len_vec[task]) {
      int packed_qo_len = qo_len * gqa_group_size;
      int num_qo_tiles = ceil_div(packed_qo_len, cluster_tile_q);
      for (int qo_tile_idx = 0; qo_tile_idx < num_qo_tiles; ++qo_tile_idx) {
        int remaining_len = causal
                                ? packed_causal_kv_end(qo_len, kv_len, qo_tile_idx,
                                                       cluster_tile_q, num_qo_tiles, gqa_group_size)
                                : kv_len;
        int kv_start = 0;
        bool split_kv = remaining_len > kv_len_limit;
        int num_kv_tiles = split_kv ? ceil_div(remaining_len, kv_len_limit) : 1;
        int row_tile_size =
            std::min(cluster_tile_q, packed_qo_len - qo_tile_idx * cluster_tile_q);
        bool zero_kv_len = (remaining_len == 0);
        while (remaining_len > 0 || zero_kv_len) {
          int actual_len = std::min(remaining_len, kv_len_limit);
          for (uint32_t kv_head_idx = 0; kv_head_idx < num_kv_heads; ++kv_head_idx) {
            auto [cluster_idx, accum_cost] = cluster_cost_heap.pop();
            cluster_cost_heap.insert(
                {cluster_idx, accum_cost + cost_function(cluster_tile_q, actual_len)});
            cluster_q_len[cluster_idx].push_back(qo_len);
            cluster_kv_len[cluster_idx].push_back(kv_len);
            cluster_q_indptr[cluster_idx].push_back(qo_indptr_h[i]);
            cluster_kv_indptr[cluster_idx].push_back(kv_indptr_h[i]);
            cluster_partial_indptr[cluster_idx].push_back(partial_o_nnz);
            cluster_q_start[cluster_idx].push_back(qo_tile_idx * cluster_tile_q);
            cluster_kv_start[cluster_idx].push_back(kv_start);
            cluster_kv_end[cluster_idx].push_back(kv_start + actual_len);
            cluster_kv_head_idx[cluster_idx].push_back(kv_head_idx);
          }
          remaining_len -= actual_len;
          zero_kv_len = (remaining_len == 0);
          kv_start += actual_len;
          if (zero_kv_len) {
            break;
          }
        }
        if (split_kv) {
          for (int row = 0; row < row_tile_size; ++row) {
            merge_indptr.push_back(merge_indptr.back() + num_kv_tiles);
            auto q = (qo_tile_idx * cluster_tile_q + row) / gqa_group_size,
                 r = (qo_tile_idx * cluster_tile_q + row) % gqa_group_size;
            merge_o_indices.push_back((qo_indptr_h[i] + q) * num_kv_heads * gqa_group_size + r);
          }
          partial_o_nnz += row_tile_size * num_kv_tiles;
        }
      }
    }

    std::vector<IdType> work_indptr_vec(num_clusters + 1, 0);
    for (int i = 0; i < num_clusters; ++i) {
      work_indptr_vec[i + 1] = work_indptr_vec[i] + cluster_q_indptr[i].size();
    }
    int total_num_works = work_indptr_vec.back();
    if (total_num_works > max_total_num_works) {
      std::ostringstream err_msg;
      err_msg << "total_num_works (#q tiles * #kv tiles) " << total_num_works
              << " exceeds max_total_num_works " << max_total_num_works;
      FLASHINFER_ERROR(err_msg.str());
    }
    auto q_indptr_vec = flatten(cluster_q_indptr, total_num_works);
    auto kv_indptr_vec = flatten(cluster_kv_indptr, total_num_works);
    auto partial_indptr_vec = flatten(cluster_partial_indptr, total_num_works);
    auto q_len_vec = flatten(cluster_q_len, total_num_works);
    auto kv_len_vec = flatten(cluster_kv_len, total_num_works);
    auto q_start_vec = flatten(cluster_q_start, total_num_works);
    auto kv_start_vec = flatten(cluster_kv_start, total_num_works);
    auto kv_end_vec = flatten(cluster_kv_end, total_num_works);
    auto kv_head_idx_vec = flatten(cluster_kv_head_idx, total_num_works);

    plan_info.tasks[task].q_indptr_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "q_indptr");
    plan_info.tasks[task].kv_indptr_offset = int_allocator.aligned_alloc_offset(
        sizeof(IdType) * max_total_num_works, 16, "kv_indptr");
    plan_info.tasks[task].partial_indptr_offset = int_allocator.aligned_alloc_offset(
        sizeof(IdType) * max_total_num_works, 16, "partial_indptr");
    plan_info.tasks[task].q_len_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "q_len");
    plan_info.tasks[task].kv_len_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "kv_len");
    plan_info.tasks[task].q_start_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "q_start");
    plan_info.tasks[task].kv_start_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "kv_start");
    plan_info.tasks[task].kv_end_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "kv_end");
    plan_info.tasks[task].kv_head_idx_offset = int_allocator.aligned_alloc_offset(
        sizeof(IdType) * max_total_num_works, 16, "kv_head_idx");
    plan_info.tasks[task].work_indptr_offset = int_allocator.aligned_alloc_offset(
        sizeof(IdType) * max_total_num_works, 16, "work_indptr");

    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].q_indptr_offset,
                           q_indptr_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].kv_indptr_offset,
                           kv_indptr_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].partial_indptr_offset,
                           partial_indptr_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].q_len_offset, q_len_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].kv_len_offset,
                           kv_len_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].q_start_offset,
                           q_start_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].kv_start_offset,
                           kv_start_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].kv_end_offset,
                           kv_end_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].kv_head_idx_offset,
                           kv_head_idx_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].work_indptr_offset,
                           work_indptr_vec);
  }
  plan_info.len_kv_chunk_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * NUM_TASKS, 16, "len_kv_chunk");
  CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.len_kv_chunk_offset,
                         cluster_len_kv_chunk);

  if ((int)merge_indptr.size() > max_num_kv_splits) {
    std::ostringstream err_msg;
    err_msg << "Number of kv splits " << merge_indptr.size() << " exceeds max buffer size "
            << max_num_kv_splits << ". Please increase the threshold.";
    FLASHINFER_ERROR(err_msg.str());
  }

  num_expand_qo_len_vec.push_back(merge_indptr.size() - 1);
  plan_info.merge_indptr_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_num_kv_splits, 16, "merge_indptr");
  plan_info.merge_o_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_num_kv_splits, 16, "merge_o_indices");
  plan_info.num_qo_len_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType), 16, "num_qo_len_offset");
  CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.merge_indptr_offset, merge_indptr);
  CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.merge_o_indices_offset,
                         merge_o_indices);
  CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.num_qo_len_offset,
                         num_expand_qo_len_vec);

  size_t num_bytes_to_copy = int_allocator.num_allocated_bytes();
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, num_bytes_to_copy,
                                       cudaMemcpyHostToDevice, stream));
  constexpr size_t sizeof_dtype_o = 2;

  AlignedAllocator float_allocator(float_buffer, float_workspace_size_in_bytes);
  plan_info.partial_o_offset = float_allocator.aligned_alloc_offset(
      max_num_kv_splits * sizeof_dtype_o * head_dim * num_kv_heads, 16, "holistic_partial_o");
  plan_info.partial_lse_offset = float_allocator.aligned_alloc_offset(
      max_num_kv_splits * sizeof(float) * num_kv_heads, 16, "holistic_partial_lse");

  return cudaSuccess;
}

// ─────────────────────────────────────────────────────────────────────
// Phase C2b-step-3 — megakernel-side helper.
//
// `setup_flashinfer_params_for_megakernel` builds the per-launch
// FlashInfer plan via TwoStageHolisticPlanWithNumSm ONCE, then
// constructs a device-resident PersistentParams[NUM_LAYERS] array.
// Each layer's params share most fields and differ only in the
// per-layer k/v base pointer (the megakernel stores layer L's pages
// at offset `layer * pages_per_layer * page_stride` in g.k_cache /
// g.v_cache).
//
// `target_num_clusters` is the FlashInfer planner's `num_blks_y` and
// must equal the megakernel's `NUM_CTAS` — both come from
// `TargetProfile::cooperative_grid_size()` on the Rust side, so they
// stay in sync. The megakernel template emits a static_assert that
// would fire at codegen time if they ever drifted.
//
// The Rust harness calls this once per launch, gets back a
// FlashInferAttentionPlan handle (workspace ptrs + params array
// device ptr), passes the params ptr to the megakernel launcher,
// and frees the workspaces after the kernel synchronizes via
// `teardown_flashinfer_attention_plan`.
//
// Layout / FlashInfer config: NHD, single sequence (batch_size=1),
// causal mask. Same constraints as `run_flashinfer_attention_smoke`.

struct FlashInferAttentionPlan {
  void* float_ws_d;
  void* int_ws_d;
  void* int_ws_h;
  PersistentParams* params_d;  // device, length = num_layers
  // Cooperative grid dims for the per-layer FlashInfer launcher
  // (`cp5_run_flashinfer_attention_for_layer`). Stored on the plan
  // because the planner picked them — the host would otherwise
  // have to reach into the planner output to get them.
  int32_t num_blks_x;
  int32_t num_blks_y;
};

extern "C" int32_t setup_flashinfer_params_for_megakernel(
    DTypeQ* q_post_rope, DTypeKV* k_cache_layer0, DTypeKV* v_cache_layer0, int32_t* kv_indices,
    DTypeO* attn_out, int32_t seq_len, int32_t num_qo_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t page_size, int32_t pages_per_layer, int32_t num_layers,
    int32_t target_num_clusters, size_t float_ws_bytes, size_t int_ws_bytes, float sm_scale,
    cudaStream_t stream, FlashInferAttentionPlan* out_plan) {
  if (head_dim != HEAD_DIM_QK) {
    std::fprintf(stderr,
                 "setup_flashinfer_params_for_megakernel: head_dim=%d does not match "
                 "compiled HEAD_DIM_QK=%d\n",
                 head_dim, HEAD_DIM_QK);
    return static_cast<int32_t>(FlashInferShimStatus::RunFailed);
  }

  out_plan->float_ws_d = nullptr;
  out_plan->int_ws_d = nullptr;
  out_plan->int_ws_h = nullptr;
  out_plan->params_d = nullptr;

  if (cudaMalloc(&out_plan->float_ws_d, float_ws_bytes) != cudaSuccess) {
    return static_cast<int32_t>(FlashInferShimStatus::CudaMallocFailed);
  }
  if (cudaMalloc(&out_plan->int_ws_d, int_ws_bytes) != cudaSuccess) {
    cudaFree(out_plan->float_ws_d);
    return static_cast<int32_t>(FlashInferShimStatus::CudaMallocFailed);
  }
  if (cudaMallocHost(&out_plan->int_ws_h, int_ws_bytes) != cudaSuccess) {
    cudaFree(out_plan->float_ws_d);
    cudaFree(out_plan->int_ws_d);
    return static_cast<int32_t>(FlashInferShimStatus::CudaMallocHostFailed);
  }

  IdType qo_indptr_h[2] = {0, seq_len};
  IdType kv_indptr_h[2] = {0, pages_per_layer};
  IdType kv_len_h[1] = {seq_len};

  HolisticPlanInfo<2> plan_info;
  // Forked planner that takes `num_sm` as a parameter rather than
  // querying the device. We pass `target_num_clusters` (=
  // TargetProfile::cooperative_grid_size()) so the plan's
  // num_blks_y == NUM_CTAS in the megakernel template — see the
  // function-level comment on TwoStageHolisticPlanWithNumSm.
  cudaError_t status = TwoStageHolisticPlanWithNumSm<IdType>(
      out_plan->float_ws_d, float_ws_bytes, out_plan->int_ws_d, out_plan->int_ws_h,
      int_ws_bytes, plan_info, qo_indptr_h, kv_indptr_h, kv_len_h, /*batch_size=*/1,
      num_qo_heads, num_kv_heads, head_dim, /*causal=*/true, stream,
      /*num_sm_in=*/target_num_clusters);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "TwoStageHolisticPlanWithNumSm failed: %s\n",
                 cudaGetErrorString(status));
    cudaFree(out_plan->float_ws_d);
    cudaFree(out_plan->int_ws_d);
    cudaFreeHost(out_plan->int_ws_h);
    return static_cast<int32_t>(FlashInferShimStatus::PlanFailed);
  }

  std::vector<PersistentParams> params_h(num_layers);
  IdType* len_kv_chunk_d = offset_ptr<IdType>(out_plan->int_ws_d, plan_info.len_kv_chunk_offset);

  // Both task slots (CTA_TILE_Q=128 and =16) share the same plan
  // workspace; for our prefill workload every work item lands in
  // task 0 (long-Q), so we always populate from task 0's offsets.
  constexpr int kTaskIdx = 0;

  const size_t k_layer_stride = (size_t)pages_per_layer * (size_t)page_size *
                                (size_t)num_kv_heads * (size_t)head_dim;

  for (int layer = 0; layer < num_layers; ++layer) {
    PersistentParams& p = params_h[layer];
    p.q = q_post_rope;
    p.k = k_cache_layer0 + (size_t)layer * k_layer_stride;
    p.v = v_cache_layer0 + (size_t)layer * k_layer_stride;
    p.kv_indices = kv_indices;
    // FlashInfer writes the attention output into attn_out, which is
    // the same buffer the megakernel's downstream tile_o_proj reads
    // from (g.attn_out) — i.e. we plug into the same input/output
    // stream as the original hand-written tile_attention.
    p.o = attn_out;
    p.final_o = attn_out;
    p.final_lse = nullptr;
    p.partial_o = offset_ptr<DTypeO>(out_plan->float_ws_d, plan_info.partial_o_offset);
    p.partial_lse = offset_ptr<float>(out_plan->float_ws_d, plan_info.partial_lse_offset);

    p.q_indptr =
        offset_ptr<IdType>(out_plan->int_ws_d, plan_info.tasks[kTaskIdx].q_indptr_offset);
    p.kv_indptr =
        offset_ptr<IdType>(out_plan->int_ws_d, plan_info.tasks[kTaskIdx].kv_indptr_offset);
    p.partial_indptr =
        offset_ptr<IdType>(out_plan->int_ws_d, plan_info.tasks[kTaskIdx].partial_indptr_offset);
    p.q_len = offset_ptr<IdType>(out_plan->int_ws_d, plan_info.tasks[kTaskIdx].q_len_offset);
    p.kv_len = offset_ptr<IdType>(out_plan->int_ws_d, plan_info.tasks[kTaskIdx].kv_len_offset);
    p.q_start = offset_ptr<IdType>(out_plan->int_ws_d, plan_info.tasks[kTaskIdx].q_start_offset);
    p.kv_start =
        offset_ptr<IdType>(out_plan->int_ws_d, plan_info.tasks[kTaskIdx].kv_start_offset);
    p.kv_end = offset_ptr<IdType>(out_plan->int_ws_d, plan_info.tasks[kTaskIdx].kv_end_offset);
    p.kv_head_idx_arr =
        offset_ptr<IdType>(out_plan->int_ws_d, plan_info.tasks[kTaskIdx].kv_head_idx_offset);
    p.work_indptr =
        offset_ptr<IdType>(out_plan->int_ws_d, plan_info.tasks[kTaskIdx].work_indptr_offset);
    p.len_kv_chunk = len_kv_chunk_d + kTaskIdx;

    p.merge_indptr = offset_ptr<IdType>(out_plan->int_ws_d, plan_info.merge_indptr_offset);
    p.merge_o_indices = offset_ptr<IdType>(out_plan->int_ws_d, plan_info.merge_o_indices_offset);
    p.num_packed_qo_len = offset_ptr<IdType>(out_plan->int_ws_d, plan_info.num_qo_len_offset);

    p.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
    p.gqa_group_size = uint_fastdiv(static_cast<uint32_t>(num_qo_heads / num_kv_heads));
    p.page_size = uint_fastdiv(static_cast<uint32_t>(page_size));

    p.q_stride_n = static_cast<uint32_t>(num_qo_heads * head_dim);
    p.q_stride_h = static_cast<uint32_t>(head_dim);
    p.k_stride_page = static_cast<uint32_t>(page_size * num_kv_heads * head_dim);
    p.k_stride_n = static_cast<uint32_t>(num_kv_heads * head_dim);
    p.k_stride_h = static_cast<uint32_t>(head_dim);
    p.v_stride_page = static_cast<uint32_t>(page_size * num_kv_heads * head_dim);
    p.v_stride_n = static_cast<uint32_t>(num_kv_heads * head_dim);
    p.v_stride_h = static_cast<uint32_t>(head_dim);

    p.sm_scale = sm_scale;
    p.v_scale = 1.0f;
    p.logits_soft_cap = 0.0f;
  }

  if (cudaMalloc(&out_plan->params_d, num_layers * sizeof(PersistentParams)) != cudaSuccess) {
    cudaFree(out_plan->float_ws_d);
    cudaFree(out_plan->int_ws_d);
    cudaFreeHost(out_plan->int_ws_h);
    return static_cast<int32_t>(FlashInferShimStatus::CudaMallocFailed);
  }
  cudaMemcpyAsync(out_plan->params_d, params_h.data(),
                  num_layers * sizeof(PersistentParams), cudaMemcpyHostToDevice, stream);

  out_plan->num_blks_x = static_cast<int32_t>(plan_info.num_blks_x);
  out_plan->num_blks_y = static_cast<int32_t>(plan_info.num_blks_y);

  return static_cast<int32_t>(FlashInferShimStatus::Ok);
}

extern "C" void teardown_flashinfer_attention_plan(FlashInferAttentionPlan* plan) {
  if (plan->params_d) cudaFree(plan->params_d);
  if (plan->float_ws_d) cudaFree(plan->float_ws_d);
  if (plan->int_ws_d) cudaFree(plan->int_ws_d);
  if (plan->int_ws_h) cudaFreeHost(plan->int_ws_h);
  plan->params_d = nullptr;
  plan->float_ws_d = nullptr;
  plan->int_ws_d = nullptr;
  plan->int_ws_h = nullptr;
}

// ─────────────────────────────────────────────────────────────────────
// CP5-D-1: per-layer FlashInfer launcher.
//
// The CP5 lowering interpreter calls this once per Attention tile in
// the solved Assignment. It reuses the per-launch plan built by
// `setup_flashinfer_params_for_megakernel` (single-task params per
// layer in `plan->params_d`) and dispatches the persistent runner via
// a tiny `__global__` wrapper that calls
// `BlockBatchPagedAttentionPersistent::Run` for one layer.
//
// **Why a wrapper kernel and not `BatchPagedAttentionPersistent`**:
// the existing per-launch plan stores ONE PersistentParams per layer
// (the long-Q task-0 entry; for our prefill workload all work lives
// in task 0). The host launcher
// `flashinfer::BatchPagedAttentionPersistent` expects TWO params (one
// per dispatch task). Calling it would either need a second per-layer
// task or would double-process task 0. The wrapper kernel below uses
// the same `BlockBatchPagedAttentionPersistent::Run` device function
// the megakernel calls inline — same exact code path, just dispatched
// from the host one layer at a time.

namespace cp5_per_layer_attn {

constexpr uint32_t FI_CTA_TILE_Q = 128;
constexpr uint32_t FI_NUM_WARPS_Q = flashinfer::get_num_warps_q(FI_CTA_TILE_Q);
constexpr uint32_t FI_NUM_WARPS_KV = flashinfer::get_num_warps_kv(FI_CTA_TILE_Q);
constexpr uint32_t FI_NUM_MMA_Q = flashinfer::get_num_mma_q(FI_CTA_TILE_Q);
constexpr uint32_t FI_NUM_MMA_KV = 4;
constexpr uint32_t FI_NUM_MMA_D_QK = HEAD_DIM_QK / 16;
constexpr uint32_t FI_NUM_MMA_D_VO = HEAD_DIM_VO / 16;

using KTraits = flashinfer::KernelTraits<
    flashinfer::MaskMode::kCausal,
    /*CTA_TILE_Q=*/FI_CTA_TILE_Q,
    /*NUM_MMA_Q=*/FI_NUM_MMA_Q,
    /*NUM_MMA_KV=*/FI_NUM_MMA_KV,
    /*NUM_MMA_D_QK=*/FI_NUM_MMA_D_QK,
    /*NUM_MMA_D_VO=*/FI_NUM_MMA_D_VO,
    /*NUM_WARPS_Q=*/FI_NUM_WARPS_Q,
    /*NUM_WARPS_KV=*/FI_NUM_WARPS_KV,
    flashinfer::PosEncodingMode::kNone,
    DTypeQ, DTypeKV, DTypeO, float, IdType,
    StandardAttention</*UseLogitsSoftCap=*/false>>;

using Runner = flashinfer::BlockBatchPagedAttentionPersistent<KTraits, PersistentParams>;

constexpr uint32_t kNumThreads = KTraits::NUM_THREADS;
constexpr size_t kSharedStorageBytes = sizeof(KTraits::SharedStorage);

}  // namespace cp5_per_layer_attn

// One-layer wrapper kernel: reads `params_array[layer_idx]` and runs
// the persistent FlashInfer dispatch on this CTA's slice of work.
__global__ void cp5_one_layer_attention_kernel(PersistentParams* params_array,
                                                int32_t layer_idx) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  auto& smem_storage =
      *reinterpret_cast<cp5_per_layer_attn::KTraits::SharedStorage*>(smem_raw);
  cp5_per_layer_attn::Runner::Run(params_array[layer_idx], &smem_storage);
}

extern "C" int32_t cp5_run_flashinfer_attention_for_layer(
    FlashInferAttentionPlan* plan,
    int32_t layer_idx,
    cudaStream_t stream)
{
  size_t smem_bytes = cp5_per_layer_attn::kSharedStorageBytes;

  // Opt the wrapper into the per-CTA dynamic shmem carveout. This
  // call is idempotent — safe to call before every launch.
  auto attr_err = cudaFuncSetAttribute(
      (const void*)cp5_one_layer_attention_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes));
  if (attr_err != cudaSuccess) {
    std::fprintf(stderr,
                 "cp5_run_flashinfer_attention_for_layer: cudaFuncSetAttribute failed: %s\n",
                 cudaGetErrorString(attr_err));
    return -1;
  }

  PersistentParams* params_d = plan->params_d;
  void* args[] = { &params_d, &layer_idx };

  // Match the megakernel's launch shape: blockIdx.y carries the
  // persistent CTA id (FlashInfer reads `work_indptr[blockIdx.y]`),
  // blockIdx.x is the cluster Q-split (always 1 for our prefill
  // config). num_blks_y is the cooperative grid size from the
  // planner — same value the megakernel uses.
  //
  // **cudaLaunchKernel, not cudaLaunchCooperativeKernel**:
  // FlashInfer's `BlockBatchPagedAttentionPersistent::Run` uses
  // `__syncthreads()` (block-local) and gmem-flag work distribution
  // via `work_indptr[blockIdx.y]`. It does NOT call
  // `cooperative_groups::this_grid().sync()`, so it does not need
  // cooperative-launch semantics. Using the regular launcher
  // (a) keeps this kernel capturable in CUDA graphs and
  // (b) avoids the per-launch cooperative-grid residency check
  //     overhead the megakernel pays.
  dim3 grid(plan->num_blks_x, plan->num_blks_y);
  dim3 block(cp5_per_layer_attn::kNumThreads);

  cp5_one_layer_attention_kernel<<<grid, block, smem_bytes, stream>>>(params_d, layer_idx);
  auto launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    std::fprintf(stderr,
                 "cp5_run_flashinfer_attention_for_layer[layer=%d]: launch failed: %s\n",
                 layer_idx, cudaGetErrorString(launch_err));
    return -2;
  }
  return 0;
}
