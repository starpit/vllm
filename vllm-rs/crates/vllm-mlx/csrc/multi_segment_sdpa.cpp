// Multi-segment SDPA — Metal kernel for attention across multiple contiguous K/V segments.
// Each segment is a span (cached or active). The kernel iterates segments with online softmax
// and applies RoPE to K on-the-fly using per-segment position offsets + cos/sin cache.
//
// Based on MLX's sdpa_vector.h — same SIMD structure, same online softmax.
//
// K/V layout: [kv_heads, total_tokens, D] (heads-first, tokens contiguous within head).
// For single segment, this is a zero-cost reshape of [1, kv_heads, seg_len, D].
// For multiple segments, concatenation along the seq dim produces this layout.

#include "multi_segment_sdpa.h"

#include "mlx/mlx.h"
#include "mlx/fast.h"
#include "mlx/primitives.h"
#include "mlx/fast_primitives.h"
#include "mlx/backend/metal/device.h"

#include "mlx/c/array.h"
#include "mlx/c/stream.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"

#include <sstream>

namespace mlx::core::fast {

static const char* multi_segment_sdpa_source = R"METAL(
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_BFLOAT__)
typedef bfloat bfloat16_t;
#endif

struct SegmentDesc {
  int start;            // token offset within the per-head token sequence
  int length;           // number of tokens
  int position_offset;  // RoPE position of first token in this segment
  int needs_rope;       // 1 = apply RoPE on-the-fly (span), 0 = K already has RoPE (active cache)
};

// Multi-segment SDPA with fused RoPE-on-read.
//
// K/V layout: [kv_heads, total_tokens, D] (heads-first).
// Within each head, tokens are at stride D (contiguous).
// head_stride separates heads (may differ from total_tokens * D for cache views).
//
// RoPE approach (matching Python vLLM Triton patch):
// Q and K are split into first-half and second-half of head_dim.
// Each thread loads both halves. RoPE rotation is local (no cross-thread shuffle).
// QK dot product = dot(Q_a, K_rot_a) + dot(Q_b, K_rot_b).
//
// Per-segment needs_rope flag: active cache has pre-rotated K (skip RoPE),
// span segments have unrotated K (apply RoPE on-the-fly).
template <typename T, int D, int V = D>
[[kernel]] void multi_segment_sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],           // [kv_heads, total_tokens, D]
    const device T* values [[buffer(2)]],          // [kv_heads, total_tokens, V]
    device T* out [[buffer(3)]],
    const device SegmentDesc* segments [[buffer(4)]],
    const device float* cos_sin_cache [[buffer(5)]],
    const constant int& num_segments [[buffer(6)]],
    const constant int& gqa_factor [[buffer(7)]],
    const constant float& scale [[buffer(8)]],
    const constant int& rotary_dim [[buffer(9)]],
    const constant int& head_stride [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int half_D = D / 2;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  // Split-half when D/2 divides evenly by BD; shuffle otherwise (e.g., D=96).
  constexpr bool USE_SPLIT_HALF = (half_D % BD == 0);
  constexpr int half_per_thread = USE_SPLIT_HALF ? (half_D / BD) : 0;

  typedef float U;

  thread U q_a[USE_SPLIT_HALF ? half_per_thread : 1];
  thread U q_b[USE_SPLIT_HALF ? half_per_thread : 1];
  thread U k_a[USE_SPLIT_HALF ? half_per_thread : 1];
  thread U k_b[USE_SPLIT_HALF ? half_per_thread : 1];
  thread U q_full[USE_SPLIT_HALF ? 1 : qk_per_thread];
  thread U k_full[USE_SPLIT_HALF ? 1 : qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const int head_idx = tid.x;
  const int kv_head_idx = head_idx / gqa_factor;

  const device T* q_base = queries + head_idx * D;
  out += head_idx * V + simd_gid * v_per_thread;

  const device T* k_head = keys + kv_head_idx * head_stride;
  const device T* v_head = values + kv_head_idx * head_stride;

  // Load Q.
  if constexpr (USE_SPLIT_HALF) {
    const int half_offset = simd_lid * half_per_thread;
    for (int i = 0; i < half_per_thread; i++) {
      q_a[i] = static_cast<U>(scale) * q_base[half_offset + i];
      q_b[i] = static_cast<U>(scale) * q_base[half_D + half_offset + i];
    }
  } else {
    for (int i = 0; i < qk_per_thread; i++) {
      q_full[i] = static_cast<U>(scale) * q_base[simd_lid * qk_per_thread + i];
    }
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  for (int seg = 0; seg < num_segments; seg++) {
    const int seg_start = segments[seg].start;
    const int seg_len = segments[seg].length;
    const int seg_pos_offset = segments[seg].position_offset;
    const bool seg_needs_rope = segments[seg].needs_rope;

    const device T* seg_k = k_head + seg_start * D;
    const device T* seg_v = v_head + seg_start * D;

    for (int i = simd_gid; i < seg_len; i += BN) {
      const device T* k_ptr = seg_k + i * D;
      U score = 0;

      if constexpr (USE_SPLIT_HALF) {
        // Split-half path: load both halves, local RoPE, two half-dots.
        const int half_offset = simd_lid * half_per_thread;
        for (int j = 0; j < half_per_thread; j++) {
          k_a[j] = k_ptr[half_offset + j];
          k_b[j] = k_ptr[half_D + half_offset + j];
        }
        if (seg_needs_rope && rotary_dim > 0) {
          const int pos = seg_pos_offset + i;
          const device float* cos_ptr = cos_sin_cache + pos * rotary_dim;
          const device float* sin_ptr = cos_ptr + half_D;
          for (int j = 0; j < half_per_thread; j++) {
            const int d = half_offset + j;
            U c = cos_ptr[d];
            U s = sin_ptr[d];
            U ka = k_a[j], kb = k_b[j];
            k_a[j] = ka * c - kb * s;
            k_b[j] = kb * c + ka * s;
          }
        }
        for (int j = 0; j < half_per_thread; j++) {
          score += q_a[j] * k_a[j] + q_b[j] * k_b[j];
        }
      } else {
        // Shuffle path: consecutive elements, simd_shuffle for RoPE pairs.
        const int elem_offset = simd_lid * qk_per_thread;
        for (int j = 0; j < qk_per_thread; j++) {
          k_full[j] = k_ptr[elem_offset + j];
        }
        if (seg_needs_rope && rotary_dim > 0) {
          const int pos = seg_pos_offset + i;
          const device float* cos_ptr = cos_sin_cache + pos * rotary_dim;
          const device float* sin_ptr = cos_ptr + half_D;
          const int pair_lane_offset = half_D / qk_per_thread;
          const bool is_first_half = (elem_offset < half_D);
          const int pair_lane = is_first_half
              ? (simd_lid + pair_lane_offset)
              : (simd_lid - pair_lane_offset);
          for (int j = 0; j < qk_per_thread; j++) {
            const int d = elem_offset + j;
            if (d >= rotary_dim) continue;
            U k_val = k_full[j];
            U k_pair = simd_shuffle(k_val, pair_lane);
            if (is_first_half) {
              k_full[j] = k_val * cos_ptr[d] - k_pair * sin_ptr[d];
            } else {
              const int pair_d = d - half_D;
              k_full[j] = k_val * cos_ptr[pair_d] + k_pair * sin_ptr[pair_d];
            }
          }
        }
        for (int j = 0; j < qk_per_thread; j++) {
          score += q_full[j] * k_full[j];
        }
      }
      score = simd_sum(score);

      // Online softmax.
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // V accumulation (stride D between tokens, full dimension).
      const device T* v_ptr = seg_v + i * D + simd_lid * v_per_thread;
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * v_ptr[j];
      }
    }
  }

  // Cross-simdgroup reduction (identical to sdpa_vector).
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

#define instantiate_ms_sdpa(type, dim) \
  template [[host_name("multi_segment_sdpa_vector_" #type "_" #dim)]] \
  [[kernel]] void multi_segment_sdpa_vector<type, dim>( \
      const device type*, const device type*, const device type*, \
      device type*, const device SegmentDesc*, const device float*, \
      const constant int&, const constant int&, const constant float&, \
      const constant int&, const constant int&, \
      uint3, uint3, uint, uint);

instantiate_ms_sdpa(float, 64)
instantiate_ms_sdpa(float, 96)
instantiate_ms_sdpa(float, 128)
instantiate_ms_sdpa(float, 256)
instantiate_ms_sdpa(bfloat16_t, 64)
instantiate_ms_sdpa(bfloat16_t, 96)
instantiate_ms_sdpa(bfloat16_t, 128)
instantiate_ms_sdpa(bfloat16_t, 256)
instantiate_ms_sdpa(half, 64)
instantiate_ms_sdpa(half, 96)
instantiate_ms_sdpa(half, 128)
instantiate_ms_sdpa(half, 256)
)METAL";

// ---------------------------------------------------------------------------
// MultiSegmentSDPA primitive
// ---------------------------------------------------------------------------

class MultiSegmentSDPA : public Primitive {
 public:
  MultiSegmentSDPA(Stream stream, float scale, int head_dim, int rotary_dim,
                   int num_kv_heads, int num_segments,
                   std::vector<int> seg_starts,
                   std::vector<int> seg_lengths,
                   std::vector<int> seg_pos_offsets,
                   std::vector<int> seg_needs_rope)
      : Primitive(stream),
        scale_(scale),
        head_dim_(head_dim),
        rotary_dim_(rotary_dim),
        num_kv_heads_(num_kv_heads),
        num_segments_(num_segments),
        seg_starts_(std::move(seg_starts)),
        seg_lengths_(std::move(seg_lengths)),
        seg_pos_offsets_(std::move(seg_pos_offsets)),
        seg_needs_rope_(std::move(seg_needs_rope)) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("MultiSegmentSDPA only supports GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override;

  DEFINE_PRINT(MultiSegmentSDPA);
  bool is_equivalent(const Primitive&) const override { return false; }

 private:
  float scale_;
  int head_dim_;
  int rotary_dim_;
  int num_kv_heads_;
  int num_segments_;
  std::vector<int> seg_starts_;
  std::vector<int> seg_lengths_;
  std::vector<int> seg_pos_offsets_;
  std::vector<int> seg_needs_rope_;
};

void MultiSegmentSDPA::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // inputs[0] = query        [num_heads, head_dim]
  // inputs[1] = k_cat        [kv_heads, total_tokens, head_dim]
  // inputs[2] = v_cat        [kv_heads, total_tokens, head_dim]
  // inputs[3] = cos_sin_cache [max_pos, rotary_dim]
  auto& s = stream();
  auto& q = inputs[0];
  auto& k_cat = inputs[1];
  auto& v_cat = inputs[2];
  auto& cos_sin = inputs[3];

  int num_heads = q.shape(0);
  int gqa_factor = num_heads / num_kv_heads_;
  // Use actual stride from the array — handles non-contiguous views
  // (e.g., cache slices where capacity > seq_len).
  int head_stride = k_cat.strides()[0];

  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  auto& d = metal::device(s.device);

  // Build segment descriptor GPU buffer.
  struct GpuSegDesc { int start; int length; int position_offset; int needs_rope; };
  std::vector<GpuSegDesc> descs(num_segments_);
  for (int i = 0; i < num_segments_; i++) {
    descs[i] = {seg_starts_[i], seg_lengths_[i], seg_pos_offsets_[i], seg_needs_rope_[i]};
  }
  size_t desc_bytes = num_segments_ * sizeof(GpuSegDesc);

  // Get kernel.
  std::string lib_name = "vllm_multi_segment_sdpa";
  auto lib = d.get_library(lib_name, [] {
    return std::string(multi_segment_sdpa_source);
  });

  std::ostringstream kname;
  kname << "multi_segment_sdpa_vector_";
  if (q.dtype() == float32) kname << "float";
  else if (q.dtype() == bfloat16) kname << "bfloat16_t";
  else if (q.dtype() == float16) kname << "half";
  kname << "_" << head_dim_;

  auto kernel = d.get_kernel(kname.str(), lib);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k_cat, 1);
  compute_encoder.set_input_array(v_cat, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(descs.data(), desc_bytes, 4);
  compute_encoder.set_input_array(cos_sin, 5);
  compute_encoder.set_bytes(num_segments_, 6);
  compute_encoder.set_bytes(gqa_factor, 7);
  compute_encoder.set_bytes(scale_, 8);
  compute_encoder.set_bytes(rotary_dim_, 9);
  compute_encoder.set_bytes(head_stride, 10);

  MTL::Size grid = MTL::Size(num_heads, 1, 1);
  MTL::Size group = MTL::Size(1024, 1, 1);
  compute_encoder.dispatch_threadgroups(grid, group);
}

// ---------------------------------------------------------------------------
// C++ API
// ---------------------------------------------------------------------------

array multi_segment_sdpa_op(
    const array& query,
    const std::vector<array>& k_segments,
    const std::vector<array>& v_segments,
    const std::vector<int>& position_offsets,
    const std::vector<int>& needs_rope,
    const array& cos_sin_cache,
    float scale,
    int rotary_dim,
    StreamOrDevice s) {

  int num_heads = query.shape(1);
  int head_dim = query.shape(3);
  int num_kv_heads = k_segments[0].shape(1);
  int num_segments = k_segments.size();

  // Compute segment starts and lengths.
  std::vector<int> seg_starts(num_segments);
  std::vector<int> seg_lengths(num_segments);
  int total_tokens = 0;
  for (int i = 0; i < num_segments; i++) {
    seg_starts[i] = total_tokens;
    seg_lengths[i] = k_segments[i].shape(2); // [1, kv_heads, seg_len, head_dim]
    total_tokens += seg_lengths[i];
  }

  // Reshape Q: [1, heads, 1, D] → [heads, D]
  auto q_flat = mlx::core::reshape(query, {num_heads, head_dim}, s);

  // K/V: keep heads-first layout [kv_heads, total_tokens, D].
  // For single segment: reshape [1, kv_heads, seg_len, D] → [kv_heads, seg_len, D] (zero-copy).
  // For multiple segments: concatenate along seq dim (dim 2) then squeeze batch.
  auto make_cat = [&](const std::vector<array>& segs) -> array {
    if (num_segments == 1) {
      return mlx::core::reshape(segs[0], {num_kv_heads, seg_lengths[0], head_dim}, s);
    }
    auto cat = mlx::core::concatenate(segs, 2, s);
    return mlx::core::reshape(cat, {num_kv_heads, total_tokens, head_dim}, s);
  };
  auto k_cat = make_cat(k_segments);
  auto v_cat = make_cat(v_segments);

  auto out_shape = std::vector<int>{1, num_heads, 1, head_dim};

  return array(
      std::move(out_shape),
      query.dtype(),
      std::make_shared<MultiSegmentSDPA>(
          to_stream(s), scale, head_dim, rotary_dim, num_kv_heads,
          num_segments, std::move(seg_starts), std::move(seg_lengths),
          std::vector<int>(position_offsets.begin(), position_offsets.end()),
          std::vector<int>(needs_rope.begin(), needs_rope.end())),
      {q_flat, k_cat, v_cat, cos_sin_cache});
}

} // namespace mlx::core::fast

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------

extern "C" int vllm_multi_segment_sdpa(
    vllm_mlx_array* result,
    vllm_mlx_array query,
    const vllm_mlx_array* k_segments,
    const vllm_mlx_array* v_segments,
    int num_segments,
    vllm_mlx_array seg_position_offsets,
    const int* seg_needs_rope,
    vllm_mlx_array cos_sin_cache,
    float scale,
    int rotary_dim,
    vllm_mlx_stream stream) {
  try {
    mlx_stream s_h = {stream.ctx};
    mlx_array q_h = {query.ctx};
    mlx_array cos_sin_h = {cos_sin_cache.ctx};
    mlx_array offsets_h = {seg_position_offsets.ctx};

    std::vector<mlx::core::array> k_segs, v_segs;
    for (int i = 0; i < num_segments; i++) {
      mlx_array kh = {k_segments[i].ctx};
      mlx_array vh = {v_segments[i].ctx};
      k_segs.push_back(mlx_array_get_(kh));
      v_segs.push_back(mlx_array_get_(vh));
    }

    // Extract position offsets from MLX array.
    auto offsets_arr = mlx_array_get_(offsets_h);
    mlx::core::eval({offsets_arr});
    auto* offsets_data = offsets_arr.data<int32_t>();
    std::vector<int> pos_offsets(offsets_data, offsets_data + num_segments);
    std::vector<int> needs_rope(seg_needs_rope, seg_needs_rope + num_segments);

    auto out = mlx::core::fast::multi_segment_sdpa_op(
        mlx_array_get_(q_h),
        k_segs,
        v_segs,
        pos_offsets,
        needs_rope,
        mlx_array_get_(cos_sin_h),
        scale,
        rotary_dim,
        mlx_stream_get_(s_h));

    // Allocate a new mlx::core::array on the heap for the result.
    // The Rust Array wrapper will manage lifetime via mlx_array_free.
    result->ctx = new mlx::core::array(std::move(out));
    return 0;
  } catch (const std::exception& e) {
    return -1;
  }
}
