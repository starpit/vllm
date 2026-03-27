# Span Support: Relocatable KV Cache Blocks

Spans enable position-independent KV cache block reuse. A "span" is a
contiguous subsequence of tokens (e.g., a RAG document) whose cached K/V
can be reused at any position in any request without recomputation.

The key insight: store K **without** RoPE. Apply RoPE at attention time
using the token's actual position, not the position it was originally
computed at. This makes cached blocks relocatable.

Both the CUDA (Python/Triton) and MLX (Rust/Metal) implementations use
the same approach. There is no "spans enabled" switch — K is always
stored without RoPE.

## CUDA Implementation (Python vLLM Triton Patch)

### RoPE Removal at Insertion

`vllm/model_executor/layers/rotary_embedding/base.py`: K rotation is
skipped entirely. Q still gets RoPE as usual.

```python
# key rotation disabled — K stored without RoPE
ops.rotary_embedding(positions, query,
    None,  # was: key
    self.head_size, self.cos_sin_cache, ...)
```

### Fused RoPE in Triton Attention Kernel

`vllm/v1/attention/ops/triton_unified_attention.py`: The attention
kernel loads Q and K split into first-half and second-half of head_dim.
It loads cos/sin from `cos_sin_cache` using the token's absolute
position (`seq_offset`), applies the rotation to K in registers, then
computes the dot product as two half-dimension matrix multiplies:

```python
# Load K in two halves
K_a = tl.load(key_cache_ptr + k_offset_a, ...)  # first half
K_b = tl.load(key_cache_ptr + k_offset_b, ...)  # second half

# Load cos/sin for this token's position
cos = tl.load(cos_sin_cache_ptr + cos_cache_offset, ...)
sin = tl.load(cos_sin_cache_ptr + sin_cache_offset, ...)

# Rotate K in registers
K_rot_a = K_a * cos - K_b * sin
K_rot_b = K_b * cos + K_a * sin

# Two half-dimension dot products
S += scale * tl.dot(Q_a, K_rot_a)
S += scale * tl.dot(Q_b, K_rot_b)
```

This achieves ~0% overhead because the rotation hides behind K load
latency, and two half-dimension `tl.dot` calls have the same throughput
as one full-dimension dot.

### cos_sin_cache Plumbing

The model's `rotary_emb.cos_sin_cache` tensor (already computed at init)
is threaded through:

- `gpu_model_runner.py` reads it from the first attention layer's
  `rotary_emb` and passes it into `CommonAttentionMetadata`
- `TritonAttentionMetadataBuilder` copies it into `TritonAttentionMetadata`
- The Triton kernel receives it as `cos_sin_cache_ptr` buffer parameter

### Block Hashing

`vllm/v1/core/kv_cache_utils.py`: When a block starts with a span
separator token (`VLLM_V1_SPANS_TOKEN_PLUS`), its hash parent is reset
to `NONE_HASH`, making it cache-matchable regardless of preceding
context. This is how the same document gets a cache hit when it appears
at different positions in different requests.

## MLX Implementation (Rust/Metal)

### Architecture

Same principle as CUDA: K stored without RoPE, fused RoPE at attention
time. The implementation is a custom Metal kernel dispatched via an MLX
C++ Primitive.

**Files:**
- `csrc/multi_segment_sdpa.cpp` — Metal kernel source + C++ Primitive + C API
- `csrc/multi_segment_sdpa.h` — C API header
- `src/multi_segment_sdpa.rs` — Rust FFI wrapper
- `src/models/llama.rs` — Integration into attention/decoder/model layers
- `build.rs` — cc::Build against MLX headers

### Kernel Design

The Metal kernel (`multi_segment_sdpa_vector`) uses the same split-half
approach as the Triton kernel:

- 32 SIMD lanes, 32 simdgroups per threadgroup (1024 threads)
- Each thread loads `half_per_thread = (D/2) / 32` elements from each
  half of K
- RoPE rotation is local per-thread — no cross-thread communication
- QK dot = sum of two half-dimension partial dots, reduced via
  `simd_sum`
- Online softmax carries state across segments (max, sum_exp, output
  rescaling)
- Cross-simdgroup reduction via threadgroup memory transpose + simd_sum

For D=96 (Gemma), `D/2 = 48` doesn't divide evenly by 32. The kernel
uses `if constexpr` to select a shuffle-based path: consecutive elements
per thread, `simd_shuffle` to read the RoPE-paired element from another
thread. This works because `half_rot % qk_per_thread == 0` (48 % 3 == 0).

### K/V Layout

K/V are passed in heads-first layout `[kv_heads, total_tokens, D]`. For
a single segment, this is a zero-cost reshape of the cache's native
`[1, kv_heads, seq_len, D]` layout (just drops the leading dim).

The kernel reads `head_stride` from the actual array strides, not from
`total_tokens * D`. This handles non-contiguous cache views correctly
(e.g., a slice of a pre-allocated buffer where stride between heads is
`capacity * D`, not `seq_len * D`).

### Integration into LlamaAttention

`forward()` unconditionally stores K without RoPE. RoPE is applied to Q
only at projection time. At attention time:

- **Decode (q_len=1):** Multi-segment SDPA kernel with fused RoPE.
  Single segment = the active cache. `needs_rope=true`, `position_offset=0`.
- **Prefill (q_len>1):** `apply_rope_to_cached_k` (per-position MLX
  `fast::rope` calls) + native SDPA with causal mask. The kernel doesn't
  support prefill (no causal masking).
- **Small head_dim (<64):** Falls back to `apply_rope_to_cached_k` +
  native SDPA (kernel requires D >= 64).

`forward_with_segments()` on the model accepts external span segments
(per-layer K/V arrays without RoPE + position offsets) for future
multi-segment decode when the scheduler provides cached span blocks.

### Performance

Benchmarked at 32 heads, 8 KV heads, D=128 (Llama-class):

| kv_len | native SDPA | fused RoPE kernel | ratio |
|--------|-------------|-------------------|-------|
| 128    | 313 us      | 324 us            | 1.03x |
| 512    | 392 us      | 362 us            | 0.92x |
| 2048   | 436 us      | 495 us            | 1.13x |
| 4096   | 1138 us     | 869 us            | 0.76x |

The no-RoPE kernel is 0.70-0.85x native (faster). Fused RoPE adds
minimal overhead. At large context lengths the kernel is faster than
native SDPA because the heads-first layout has better cache locality.

## What We Tried and Failed at for MLX

### 1. Paged KV Cache (Dead End)

MLX's lazy evaluation model creates new arrays for every operation.
There is no way to do in-place scatter writes to a persistent buffer.
Every approach to paged KV cache on MLX hit the same wall:

- `try_index_mut` with reshape: copies the entire buffer
- Custom Metal kernel with `copy_shared_buffer`: corrupts shared state
- Reshape + write + reshape: copies the entire pool

This is fundamentally incompatible with paged KV cache, which requires
in-place scatter writes to a shared block pool. See
`feedback_mlx_no_inplace_scatter.md`.

### 2. Concatenation-Based Kernel (1.5-2.8x Overhead)

The first working kernel concatenated all K/V segments into flat
`[total_tokens, kv_heads, D]` buffers before dispatching the Metal
kernel. This used MLX lazy ops (transpose + concatenate) which added
substantial overhead even for single-segment decode:

- kv_len=128: 1.89x vs native SDPA
- kv_len=4096: 2.15x vs native SDPA

The transpose was needed because the kernel expected tokens-first layout
but the cache stores heads-first. Eliminating the transpose (switching
the kernel to heads-first layout) and using the array's actual stride
instead of computing it from shape fixed most of the overhead.

### 3. simd_shuffle RoPE (Divergence Bug + Overhead)

The first RoPE implementation used `simd_shuffle` for cross-thread
element pairing. Each thread held consecutive elements, and the paired
element for RoPE rotation was in a different thread:

```metal
// Thread 0 holds dims 0-3, needs dim 64 from thread 16
U k_pair = simd_shuffle(k[pair_j], pair_lane);
```

Two bugs:
- **Divergence:** First-half and second-half threads took different
  if/else branches. When second-half threads executed their branch,
  first-half threads' `k[]` values were already modified. The shuffle
  read stale data.
- **Wrong shuffle semantics:** `simd_shuffle(k[pair_j], pair_lane)`
  reads the value of `k[pair_j]` from thread `pair_lane` — but
  `pair_j` is evaluated per-thread, so each thread shuffles a different
  local index.

The divergence fix (uniform shuffle with `pair_lane = simd_lid +/-
pair_lane_offset`) made it correct, but the shuffle path was still
measurably slower than the split-half approach for D=64/128/256
because `simd_shuffle` has higher latency than local register access.

### 4. Per-Segment needs_rope Optimization (Unnecessary Complexity)

We built a per-segment `needs_rope` flag so that active cache segments
(K stored WITH RoPE) could skip the fused rotation while span segments
(K WITHOUT RoPE) would get it. This required:

- Separate span KV cache (`SpanKvEntry`, content-hash indexed)
- Per-request span state tracking (`RequestSpanState`)
- Worker-level span registration and lookup APIs
- Special "span prefill" step to extract K without RoPE

All of this was unnecessary. The Python CUDA patch stores ALL K without
RoPE and fuses RoPE for ALL K — no per-segment distinction. Our
benchmarks showed the fused RoPE kernel is at parity with native SDPA,
so the optimization was solving a non-problem. We removed all of it.

### 5. C API Null Pointer Crash

The initial C API wrapper created the result array handle incorrectly:

```cpp
mlx_array res_h = {result->ctx};  // ctx is NULL
mlx_array_set_(res_h, out);       // dereferences NULL → crash
```

Fixed by allocating directly: `result->ctx = new mlx::core::array(...)`.

### 6. Non-Contiguous Cache View Stride Mismatch

The KV cache pre-allocates `[1, kv_heads, capacity, D]` and returns
views `[1, kv_heads, seq_len, D]` where `seq_len < capacity`. The
stride between heads in the view is `capacity * D`, not `seq_len * D`.
The kernel initially computed `head_stride = total_tokens * D` which
was wrong for cache views. Fixed by reading the actual stride from the
array: `head_stride = k_cat.strides()[0]`.
