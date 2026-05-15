# Phase 4.1: Attention Kernel Survey

## CUDA Implementation Analysis

### File: `csrc/attention/attention_kernels.cuh`

**Core Algorithm**: FlashAttention-style paged attention with numerical stability

### Key Components

#### 1. Paged Attention V1 (Single-pass)
```cuda
paged_attention_v1_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>
```
- Single kernel launch for full attention computation
- Grid: `(num_heads, num_seqs, 1)`
- Suitable for short sequences (< 2048 tokens typically)

#### 2. Paged Attention V2 (Partitioned)
```cuda
paged_attention_v2_kernel<..., PARTITION_SIZE>
paged_attention_v2_reduce_kernel<...>
```
- Two-stage approach for long sequences
- Stage 1: Compute attention per partition (Grid: `(num_heads, num_seqs, max_num_partitions)`)
- Stage 2: Reduce across partitions (Grid: `(num_heads, num_seqs)`)
- Enables processing sequences > GPU memory limits

### Algorithm Steps

#### Stage 1: Q-K Attention Scores
1. **Load Query**: Each thread group loads part of query vector (vectorized)
2. **Iterate KV Blocks**: 
   - Load key from paged cache (non-contiguous blocks)
   - Compute Q·K dot product using thread group reduction
   - Apply scaling factor and ALiBi bias
   - Store logits to shared memory
3. **Softmax**:
   - Find max logit (warp reduction → block reduction)
   - Compute exp(logit - max) and sum
   - Normalize: softmax = exp / sum

#### Stage 2: Attention-Value Multiplication
1. **Iterate KV Blocks**:
   - Load value from paged cache
   - Multiply by softmax weights
   - Accumulate in registers
2. **Warp Reduction**: Reduce partial sums within warp
3. **Block Reduction**: Reduce across warps via shared memory
4. **Write Output**: Final attention output

### Performance Optimizations

#### Memory Access Patterns
- **Vectorized Loads**: 16-byte aligned loads (VEC_SIZE = 16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)))
- **Coalesced Access**: Thread groups organized for coalesced memory reads
- **Shared Memory**: 
  - Logits buffer: `float logits[num_tokens]`
  - Reduction workspace: `float red_smem[2 * NUM_WARPS]`
  - Output accumulation: `float out_smem[NUM_WARPS * HEAD_SIZE]`

#### Compute Optimizations
- **Warp Shuffles**: Fast intra-warp reductions (`VLLM_SHFL_XOR_SYNC`)
- **Thread Group Size**: `THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1)`
  - Balances parallelism vs memory bandwidth
- **Register Blocking**: Accumulate in registers before shared memory

#### Numerical Stability
- **Max Subtraction**: `exp(logit - max_logit)` prevents overflow
- **Epsilon in Denominator**: `1 / (exp_sum + 1e-6)` prevents division by zero
- **Masked Logits**: Set to 0.0 (not -inf) for padding tokens

### Special Features

#### 1. Paged KV Cache
```cuda
const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
```
- Non-contiguous storage: blocks can be anywhere in memory
- Enables efficient memory management for variable-length sequences

#### 2. Grouped Query Attention (GQA)
```cuda
const int num_queries_per_kv = num_heads / num_kv_heads;
const int kv_head_idx = head_idx / num_queries_per_kv;
```
- Multiple query heads share same KV heads
- Reduces KV cache memory by factor of `num_queries_per_kv`

#### 3. ALiBi Positional Bias
```cuda
qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;
```
- Linear bias based on relative position
- Per-head slopes for different attention patterns

#### 4. Block-Sparse Attention
```cuda
if constexpr (IS_BLOCK_SPARSE) {
  const bool is_remote = ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
  const bool is_local = (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
  if (!is_remote && !is_local) {
    logits[token_idx - start_token_idx] = -FLT_MAX;
    continue;
  }
}
```
- Skip computation for non-attended blocks
- Combines local + strided attention patterns

#### 5. FP8 Quantization
```cuda
if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
  k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset);
} else {
  Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(k_ptr + offset);
  k_vecs[j] = fp8::scaled_convert<K_vec, Quant_vec, KV_DTYPE>(k_vec_quant, *k_scale);
}
```
- On-the-fly dequantization during load
- Reduces KV cache memory by 2x (FP8 vs FP16)

### Thread Organization

#### Thread Hierarchy
```
Block (NUM_THREADS = 128 typical)
├── Warps (NUM_WARPS = NUM_THREADS / 32)
│   └── Thread Groups (THREAD_GROUP_SIZE = MAX(32 / BLOCK_SIZE, 1))
│       └── Threads (process VEC_SIZE elements each)
```

#### Example: HEAD_SIZE=128, BLOCK_SIZE=16, NUM_THREADS=128
- `THREAD_GROUP_SIZE = 32 / 16 = 2`
- `NUM_THREAD_GROUPS = 128 / 2 = 64`
- `VEC_SIZE = 16 / (2 * 2) = 4` (for FP16)
- Each thread group processes 1 token's Q·K dot product

### Metal Port Strategy

#### Phase 4.1.1: Basic Single-Head Attention (No Paging)
**Goal**: Prove Metal can do attention with correct numerics
- Simplified: contiguous KV cache, single head, no quantization
- Focus on: Q·K matmul, softmax, attention·V matmul
- Verify numerical accuracy vs CUDA reference

#### Phase 4.1.2: Add Paged KV Cache
**Goal**: Handle non-contiguous memory access
- Implement block table lookup in Metal
- Verify performance with scattered reads

#### Phase 4.1.3: Multi-Head and GQA
**Goal**: Scale to production attention patterns
- Add head dimension to kernel
- Implement GQA (multiple Q heads per KV head)

#### Phase 4.1.4: Optimizations
**Goal**: Match CUDA performance
- Tune threadgroup sizes (Metal: 32-1024 threads)
- Optimize threadgroup memory usage (32KB limit)
- Use simdgroup operations for reductions
- Profile and iterate

#### Phase 4.1.5: Advanced Features
**Goal**: Feature parity with CUDA
- ALiBi positional bias
- Block-sparse attention
- FP8 quantization support
- Partitioned attention (v2) for long sequences

### Metal-Specific Considerations

#### 1. Threadgroup Memory (vs CUDA Shared Memory)
- **Limit**: 32KB per threadgroup (same as CUDA)
- **Usage**: 
  - Logits: `float[num_tokens]` (e.g., 2048 tokens = 8KB)
  - Reduction workspace: `float[2 * NUM_WARPS]` (e.g., 8 warps = 64 bytes)
  - Output buffer: `float[NUM_WARPS * HEAD_SIZE]` (e.g., 8 * 128 = 4KB)
  - **Total**: ~12KB for typical config (plenty of headroom)

#### 2. Simdgroup Operations (vs CUDA Warp Shuffles)
```metal
// CUDA: sum = VLLM_SHFL_XOR_SYNC(sum, mask);
// Metal: sum = simd_shuffle_xor(sum, mask);

// CUDA: sum = VLLM_SHFL_SYNC(sum, 0);
// Metal: sum = simd_broadcast(sum, 0);
```
- Metal simdgroups are 32-wide (same as CUDA warps)
- Similar shuffle/broadcast primitives available

#### 3. Memory Model
- **CUDA**: Explicit `__shared__` keyword
- **Metal**: `threadgroup` address space
- **CUDA**: `__syncthreads()` for barriers
- **Metal**: `threadgroup_barrier(mem_flags::mem_threadgroup)`

#### 4. Vectorization
- **CUDA**: `float4`, `half2`, etc.
- **Metal**: `float4`, `half4`, etc. (similar)
- Both support aligned vector loads

#### 5. Atomic Operations
- **CUDA**: `atomicAdd`, `atomicMax`
- **Metal**: `atomic_fetch_add_explicit`, `atomic_fetch_max_explicit`
- Metal requires explicit memory order (e.g., `memory_order_relaxed`)

### Performance Targets

#### M1 Max (32 GPU cores, 400 GB/s memory bandwidth)
- **Theoretical Peak**: 
  - FP16 compute: ~10 TFLOPS
  - Memory bandwidth: 400 GB/s
- **Attention Characteristics**:
  - Memory-bound for small head sizes (64-128)
  - Compute-bound for large head sizes (256+) or long sequences
- **Target Performance**:
  - Match or exceed MLX attention performance
  - Within 80% of CUDA performance on equivalent hardware

### Testing Strategy

#### Unit Tests
1. **Numerical Accuracy**: Compare Metal vs CUDA outputs (element-wise error < 1e-5)
2. **Edge Cases**: 
   - Single token (seq_len=1)
   - Power-of-2 sequence lengths (128, 256, 512, 1024, 2048)
   - Non-power-of-2 lengths (100, 333, 777)
   - Padding tokens (seq_len < block_size)
3. **Data Types**: FP16, BF16, FP32

#### Integration Tests
1. **Full Forward Pass**: Attention in context of transformer layer
2. **Batched Inference**: Multiple sequences in parallel
3. **Long Context**: Sequences > 4096 tokens (test partitioning)

#### Performance Tests
1. **Microbenchmarks**: Isolated attention kernel timing
2. **Roofline Analysis**: Measure achieved vs theoretical performance
3. **Profiling**: Metal GPU timeline, memory bandwidth utilization

### Next Steps

1. **Create Metal Shader**: `vllm-rs/crates/ferrite-metal-kernels/shaders/attention.metal`
2. **Implement Basic Attention**: Single-head, contiguous KV cache
3. **Write Unit Tests**: Verify numerical correctness
4. **Benchmark**: Compare vs CUDA reference
5. **Iterate**: Add features incrementally (paging, GQA, etc.)

### References
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [vLLM Paged Attention Blog](https://blog.vllm.ai/2023/06/20/vllm.html)
- [Metal Shading Language Spec](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
