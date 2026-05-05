# Phase 4.4: AWQ Quantization Survey

## Overview
AWQ (Activation-aware Weight Quantization) is a 4-bit weight quantization method that reduces model size by 4x while maintaining accuracy. This document surveys the CUDA implementation to guide the Metal port.

## AWQ Format

### Weight Storage
- **Quantization:** INT4 (4-bit integers, range 0-15)
- **Packing:** 8 weights packed into each `uint32` (32 bits / 4 bits = 8 weights)
- **Layout:** Weights stored in row-major order, packed sequentially

### Quantization Parameters
- **Scales:** FP16 values, one per group of weights
- **Zeros:** INT4 values (packed like weights), one per group
- **Group Size (G):** Typically 128, controls quantization granularity

### Dequantization Formula
```
dequantized_weight = (int4_weight - zero) * scale
```

Where:
- `int4_weight`: 4-bit integer (0-15)
- `zero`: 4-bit zero-point (0-15)
- `scale`: FP16 scaling factor

## CUDA Implementation Analysis

### 1. INT4 Unpacking (`dequantize_s4_to_fp16x2`)

**Input:** `uint32` containing 8 packed INT4 values
**Output:** `uint4` (4x `uint32`) containing 8 FP16 values

**Algorithm:**
1. Extract INT4 values using bit masks:
   - `BOTTOM_MASK = 0x000f000f` (extracts bits 0-3, 16-19)
   - `TOP_MASK = 0x00f000f0` (extracts bits 4-7, 20-23)
2. Convert to FP16 using magic number trick:
   - Add `0x64006400` to create FP16 representation
   - Subtract `0x64006400` in FP16 to get final value
3. Uses inline PTX assembly (`lop3.b32`, `sub.f16x2`, `fma.rn.f16x2`)

**Key CUDA features:**
- `lop3.b32`: 3-input LUT-based logic operation (no Metal equivalent)
- `sub.f16x2`: Packed FP16 subtraction (2 values at once)
- `fma.rn.f16x2`: Packed FP16 fused multiply-add

### 2. Fused GEMM Kernel (`gemm_forward_4bit_cuda_m16nXk32`)

**Purpose:** Compute `C = A @ dequantize(B)` where B is INT4-quantized

**Algorithm:**
1. Load activations (A) into shared memory
2. Load quantized weights (B) from global memory
3. Dequantize weights on-the-fly: `(B - zero) * scale`
4. Store dequantized weights in shared memory
5. Perform matrix multiplication using tensor cores
6. Write results to global memory

**Key optimizations:**
- Shared memory for data reuse
- Warp-level primitives (`ldmatrix.sync`)
- Tensor core operations (`mma.sync.aligned`)
- Split-K parallelization for large matrices

**CUDA-specific features:**
- Tensor cores (no Metal equivalent, use MPS instead)
- `ldmatrix`: Load matrix fragments for tensor cores
- `mma.sync`: Matrix multiply-accumulate on tensor cores

### 3. Standalone Dequantization (`dequantize_weights`)

**Purpose:** Dequantize entire weight matrix for debugging/testing

**Algorithm:**
1. Each thread loads one `uint32` (8 packed INT4 weights)
2. Load corresponding scales and zeros
3. Dequantize using `dequantize_s4_to_fp16x2`
4. Apply scales and zeros: `(weight - zero) * scale`
5. Write 8 FP16 values to output

## Metal Port Strategy

### Approach 1: Separate Dequantize + MPS GEMM (Recommended for Phase 4.4)

**Pros:**
- Simpler implementation
- Leverages optimized MPS GEMM
- Easier to debug and validate

**Cons:**
- Extra memory bandwidth (write dequantized weights, read for GEMM)
- May be slower than fused approach

**Implementation:**
1. Metal compute shader for dequantization
2. Use MPS `MPSMatrixMultiplication` for GEMM
3. Similar to existing `MetalGemm` implementation

### Approach 2: Fused Dequantize + GEMM (Future optimization)

**Pros:**
- Better memory bandwidth utilization
- Potentially faster (no intermediate buffer)

**Cons:**
- More complex implementation
- Need to implement GEMM from scratch (MPS doesn't support custom input transforms)
- Harder to optimize without tensor cores

**Defer to Phase 5 (Optimization)**

## Metal Implementation Plan

### Phase 4.4.1: INT4 Unpacking Shader

**File:** `vllm-rs/crates/ferrite-metal-kernels/shaders/awq_dequantize.metal`

**Kernel:** `awq_unpack_int4_to_fp16`

**Algorithm:**
```metal
// Input: uint32 with 8 packed INT4 values
// Output: 8 half values

kernel void awq_unpack_int4_to_fp16(
    device const uint* packed_weights [[buffer(0)]],
    device half* unpacked_weights [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    uint packed = packed_weights[gid];
    
    // Extract 8 INT4 values using bit shifts and masks
    for (int i = 0; i < 8; i++) {
        uint shift = i * 4;
        uint mask = 0xF;
        uint int4_val = (packed >> shift) & mask;
        
        // Convert to half (0-15 range)
        unpacked_weights[gid * 8 + i] = half(int4_val);
    }
}
```

**Metal bit operations:**
- `>>` (right shift)
- `&` (bitwise AND)
- No need for complex PTX tricks, simple shift-and-mask works

### Phase 4.4.2: Dequantization Shader

**Kernel:** `awq_dequantize_weights`

**Algorithm:**
```metal
kernel void awq_dequantize_weights(
    device const uint* packed_weights [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const uint* packed_zeros [[buffer(2)]],
    device half* dequantized_weights [[buffer(3)]],
    constant uint& group_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    // 1. Unpack 8 INT4 weights
    uint packed = packed_weights[gid];
    half weights[8];
    for (int i = 0; i < 8; i++) {
        weights[i] = half((packed >> (i * 4)) & 0xF);
    }
    
    // 2. Unpack 8 INT4 zeros
    uint group_idx = (gid * 8) / group_size;
    uint packed_zero = packed_zeros[group_idx];
    half zeros[8];
    for (int i = 0; i < 8; i++) {
        zeros[i] = half((packed_zero >> (i * 4)) & 0xF);
    }
    
    // 3. Load scales
    half scale_vals[8];
    for (int i = 0; i < 8; i++) {
        scale_vals[i] = scales[group_idx * 8 + i];
    }
    
    // 4. Dequantize: (weight - zero) * scale
    for (int i = 0; i < 8; i++) {
        dequantized_weights[gid * 8 + i] = 
            (weights[i] - zeros[i]) * scale_vals[i];
    }
}
```

### Phase 4.4.3: Rust Wrapper

**File:** `vllm-rs/crates/ferrite-metal-kernels/src/awq.rs`

**Struct:** `MetalAwqDequantize`

**Methods:**
- `new(device: &MetalDevice) -> Result<Self>`
- `dequantize(packed_weights, scales, zeros, group_size) -> Result<Buffer>`
- `gemm(activations, dequantized_weights) -> Result<Buffer>` (uses MPS)

### Phase 4.4.4: Unit Tests

**File:** `vllm-rs/crates/ferrite-metal-kernels/tests/awq_test.rs`

**Test cases:**
1. `test_awq_unpack_int4`: Verify INT4 unpacking correctness
2. `test_awq_dequantize`: Verify dequantization with known scales/zeros
3. `test_awq_gemm`: Verify GEMM output vs reference (CPU)
4. `test_awq_numerical_stability`: Test with edge cases (zeros, max values)

### Phase 4.4.5: Benchmarking

**Metrics:**
- Dequantization throughput (GB/s)
- GEMM throughput (TFLOPS)
- End-to-end latency (dequantize + GEMM)
- Memory bandwidth utilization

**Comparison:**
- Metal vs CUDA (if available)
- Separate vs fused (future)

## Key Differences: CUDA vs Metal

| Feature | CUDA | Metal |
|---------|------|-------|
| INT4 unpacking | Inline PTX (`lop3.b32`) | Shift-and-mask |
| Packed FP16 ops | `sub.f16x2`, `fma.rn.f16x2` | Scalar ops (compiler may vectorize) |
| Tensor cores | `mma.sync` | MPS (opaque) |
| Shared memory | Explicit | Threadgroup memory |
| Warp primitives | `ldmatrix`, shuffle | Simdgroup ops |

## Performance Expectations

### Memory Bandwidth
- INT4 weights: 4x smaller than FP16
- Dequantization: Read INT4, write FP16 (net 1.25x bandwidth vs FP16)
- GEMM: Standard FP16 bandwidth

### Compute
- Dequantization: Memory-bound (simple arithmetic)
- GEMM: Compute-bound (MPS should be near-optimal)

### Estimated Speedup
- vs FP16 GEMM: ~2-3x faster (less memory bandwidth, same compute)
- vs CUDA AWQ: 80-90% performance (MPS vs tensor cores)

## References

1. AWQ Paper: https://arxiv.org/abs/2306.00978
2. CUDA Implementation: `csrc/quantization/awq/`
3. Metal Compute Best Practices: https://developer.apple.com/metal/
4. MPS Documentation: https://developer.apple.com/documentation/metalperformanceshaders

## Next Steps

1. Implement `awq_dequantize.metal` shader
2. Create `awq.rs` Rust wrapper
3. Add unit tests
4. Benchmark and validate
5. (Phase 5) Explore fused dequantize + GEMM optimization
