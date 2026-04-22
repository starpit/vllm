# Building vLLM-RS Without cuBLAS

This document explains how to build and use vLLM-RS without cuBLAS dependencies.

## Overview

As of the ferrite-no-cublas work, vLLM-RS can be built and run without cuBLAS/cuBLASLt libraries. All core inference paths (dense, FP8, BNB4, block-FP8) now use CUTLASS kernels instead of cuBLAS.

## Quick Start

### Build without cuBLAS

```bash
# Build vllm CLI with CUDA but without cuBLAS
cargo build -p vllm-cli --features cuda --no-default-features --release

# Or use the Makefile
make -f Makefile.cublas-free build-cublas-free
```

### Build with cuBLAS (traditional)

```bash
# Build with both CUDA and cuBLAS
cargo build -p vllm-cli --features cuda,cublas --no-default-features --release

# Or use the Makefile
make -f Makefile.cublas-free build-with-cublas
```

## Feature Flags

The workspace now has separate `cuda` and `cublas` features:

- **`cuda`**: Enables CUDA runtime and CUTLASS kernels (required for GPU inference)
- **`cublas`**: Enables cuBLAS/cuBLASLt support (optional, for legacy paths)

### Crate-level features

- `ferrite-cuda-core`: `cuda`, `cublas`, `nccl`
- `ferrite-kernels`: `cuda`, `cublas`, `nccl`
- `ferrite-forward`: `cuda`, `cublas`
- `vllm-cuda`: `cuda`, `cublas`, `nccl`
- `vllm-executor`: `cuda`, `cublas`, `nccl`
- `vllm-cli`: `cuda`, `cublas`, `nccl`, ...

## What Works Without cuBLAS

All major inference paths work without cuBLAS:

### ✅ Fully Supported (No cuBLAS Required)

- **Dense BF16/F16**: Uses CUTLASS GEMM/GEMV/split-K
- **FP8 (tensor-scaled)**: Uses CUTLASS scaled_mm kernels
- **BNB4 (4-bit quantization)**: Dequantizes to scratch, then CUTLASS GEMM
- **Block-FP8**: Dequantizes to scratch, then CUTLASS GEMM
- **Marlin**: Already cuBLAS-free
- **GGML**: Already cuBLAS-free

### ⚠️ Limited Support

Some operations still use cuBLAS when the `cublas` feature is enabled:

- **`scale_inplace`**: Uses `cublasScalEx` for vectorized scaling
  - Fallback: Could use custom CUDA kernel (deferred to Phase 8)
- **Legacy dense paths**: `GemmRefImpl`, `FusedGemmBiasImpl`
  - Only registered when `allow_cublas_fallbacks = true`

## Validation

### Check that crates compile without cuBLAS

```bash
make -f Makefile.cublas-free check-cublas-free
```

This checks:
- `ferrite-cuda-core`
- `ferrite-kernels`
- `ferrite-forward`
- `vllm-cuda`
- `vllm-executor`

### Run tests

```bash
make -f Makefile.cublas-free test-cublas-free
```

## Runtime Behavior

### With `cublas` feature disabled

- `GpuDevice` is created without a `CublasHandle`
- Dense GEMM operations use CUTLASS implementations
- BNB4 and block-FP8 use CUTLASS after dequantization
- Solver will fail explicitly if a tile has no CUTLASS coverage

### With `cublas` feature enabled (default for compatibility)

- `GpuDevice` includes an `Option<CublasHandle>`
- Legacy cuBLAS paths remain available as fallbacks
- Most paths still prefer CUTLASS when available

## Solver Configuration

The `#[forward]` macro supports a `cublas_free` flag:

```rust
#[forward(
    cublas_free = true,  // Disable cuBLAS fallback registration
    // ... other args
)]
```

When `cublas_free = true`:
- `GemmRefImpl` (cuBLAS GEMM) is not registered
- `FusedGemmBiasImpl` (cuBLAS GEMM+bias) is not registered
- Solve failures provide actionable error messages about missing CUTLASS coverage

## Error Messages

When a tile has no implementation in cublas-free mode, you'll see:

```
No implementation matched tile <TileName> in cublas-free mode.

For dense GEMM tiles, check target profile CSV for:
  - cutlass_gemm (prefill)
  - cutlass_gemv (decode, M=1)
  - cutlass_gemm_splitk (large K)

For BiasAdd tiles, check:
  - cutlass_fused_gemm_bias coverage

To use cuBLAS fallback, disable cublas_free mode or file an issue.
```

## Known Limitations

1. **NVCC required**: Building still requires CUDA toolkit with `nvcc` for CUTLASS compilation
2. **Target coverage**: Some exotic tile sizes may not have CUTLASS implementations
3. **Performance**: Some paths may be slower than cuBLAS (benchmarking ongoing)

## Migration Guide

### For existing deployments

1. **No changes required**: The `cublas` feature is still enabled by default
2. **To opt into cublas-free**: Build with `--features cuda --no-default-features`
3. **To verify**: Use `make -f Makefile.cublas-free check-cublas-free`

### For new deployments

1. **Recommended**: Start with `cuda` only (no `cublas`)
2. **If issues arise**: Enable `cublas` as fallback
3. **Report**: File issues for missing CUTLASS coverage

## Development

### Adding new CUTLASS kernels

1. Add kernel to `ferrite-kernels/src/cutlass.rs`
2. Register implementation in `ferrite-forward-macro/src/impl_lib.rs`
3. Update cost model in target profile CSV
4. Test with `make -f Makefile.cublas-free test-cublas-free`

### Preventing cuBLAS reintroduction

- Use `#[cfg(feature = "cublas")]` for cuBLAS-only code
- Gate `GpuDevice.cublas` field access with `.as_ref()` or `.as_mut()`
- Add tests that verify cublas-free builds succeed

## References

- [ferrite-no-cublas.md](../tmp/ferrite-no-cublas.md) - Full implementation plan
- [Makefile.cublas-free](Makefile.cublas-free) - Build commands
- Phase 1-7 commits in `ferrite-no-cublas` worktree

## Status

- ✅ Phase 1-7: Complete (API cleanup, BNB4, block-FP8, Cargo features)
- ⏳ Phase 8: Cleanup and performance parity (ongoing)

Last updated: 2026-04-22