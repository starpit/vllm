// Benchmark: CUTLASS bf16 GEMM vs cuBLAS on L4 (sm_89).
//
// Tests multiple CUTLASS tile configurations at production sizes
// (M=batch*seq, K=4096, N=4096) and compares against cuBLAS.
//
// Compile:
//   nvcc -arch=sm_89 -O2 -std=c++17 \
//     -I$HOME/.cache/cutlass/include \
//     -o /tmp/cutlass_bench cutlass_bench.cu -lcublas -lcuda
//
// Run:
//   /tmp/cutlass_bench

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>

#define CHECK_CUDA(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %d: %s\n", __LINE__, cudaGetErrorString(e)); exit(1); } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t e = call; \
    if (e != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS %d: %d\n", __LINE__, e); exit(1); } \
} while(0)

using bf16 = cutlass::bfloat16_t;

// CUTLASS GEMM configurations to test
// A: bf16 RowMajor, B: bf16 ColumnMajor, C: bf16 RowMajor, Accum: f32

// Config 1: 128x128x32, 3 stages (medium tile)
using Gemm_128x128x32 = cutlass::gemm::device::Gemm<
    bf16, cutlass::layout::RowMajor,
    bf16, cutlass::layout::ColumnMajor,
    bf16, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<bf16, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
    3, 8, 8
>;

// Config 2: 128x256x64, 3 stages (large tile)
using Gemm_128x256x64 = cutlass::gemm::device::Gemm<
    bf16, cutlass::layout::RowMajor,
    bf16, cutlass::layout::ColumnMajor,
    bf16, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<bf16, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
    3, 8, 8
>;

// Config 3: 256x128x64, 3 stages
using Gemm_256x128x64 = cutlass::gemm::device::Gemm<
    bf16, cutlass::layout::RowMajor,
    bf16, cutlass::layout::ColumnMajor,
    bf16, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<bf16, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
    3, 8, 8
>;

// Config 4: 64x64x32, 3 stages (our test tile)
using Gemm_64x64x32 = cutlass::gemm::device::Gemm<
    bf16, cutlass::layout::RowMajor,
    bf16, cutlass::layout::ColumnMajor,
    bf16, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<bf16, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
    3, 8, 8
>;

template<typename GemmOp>
float bench_cutlass(int M, int N, int K, bf16* d_A, bf16* d_B, bf16* d_C, bf16* d_D,
                    int warmup = 10, int iters = 100) {
    GemmOp gemm_op;
    typename GemmOp::Arguments args({M,N,K},
        {d_A, K}, {d_B, K}, {d_C, N}, {d_D, N}, {1.0f, 0.0f});

    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        return -1.0f; // can't run this config
    }

    // Warmup
    for (int i = 0; i < warmup; i++) {
        gemm_op(args);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        gemm_op(args);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double us = std::chrono::duration<double, std::micro>(end - start).count() / iters;
    return (float)us;
}

float bench_cublas(int M, int N, int K, __nv_bfloat16* d_A, __nv_bfloat16* d_B,
                   __nv_bfloat16* d_D, cublasHandle_t handle,
                   int warmup = 10, int iters = 100) {
    // cuBLAS: C = alpha * A * B^T + beta * C
    // A is MxK row-major = KxM col-major
    // B is NxK col-major (our layout) = KxN col-major transposed
    // We want: D = A * B where A is MxK (row), B is NxK (col) → D is MxN
    // cuBLAS column-major: D_col = B^T * A^T, but we have row-major A...
    //
    // Simplest: use cublasGemmEx with CUBLAS_OP_N/CUBLAS_OP_T
    // D (MxN row) = A (MxK row) * B^T (KxN from NxK col)
    // In cublas col-major terms: D^T (NxM) = B (NxK) * A^T (KxM)
    // So: m=N, n=M, k=K, A_cublas=B, B_cublas=A, lda=K, ldb=K, ldc=N

    float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < warmup; i++) {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, CUDA_R_16BF, K,
            d_A, CUDA_R_16BF, K,
            &beta,
            d_D, CUDA_R_16BF, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, CUDA_R_16BF, K,
            d_A, CUDA_R_16BF, K,
            &beta,
            d_D, CUDA_R_16BF, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double us = std::chrono::duration<double, std::micro>(end - start).count() / iters;
    return (float)us;
}

int main() {
    CHECK_CUDA(cudaSetDevice(0));

    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));
    CHECK_CUBLAS(cublasSetMathMode(cublas, CUBLAS_TF32_TENSOR_OP_MATH));

    // Test sizes: (M, N, K) representing typical LLaMA operations
    struct TestCase { int M, N, K; const char* desc; };
    TestCase cases[] = {
        {1,    4096, 4096, "decode bs=1"},
        {8,    4096, 4096, "decode bs=8"},
        {32,   4096, 4096, "decode bs=32"},
        {128,  4096, 4096, "prefill 128"},
        {512,  4096, 4096, "prefill 512"},
        {2048, 4096, 4096, "prefill 2048"},
        // Gate+up proj (N = 2*intermediate = 2*11008 ≈ 22016 for LLaMA-7B)
        {128, 11008, 4096, "gate_up bs=128"},
        // Down proj (K = intermediate)
        {128, 4096, 11008, "down bs=128"},
    };

    int max_elems = 2048 * 22016; // largest allocation needed
    bf16 *d_A, *d_B, *d_C, *d_D_cutlass, *d_D_cublas;
    CHECK_CUDA(cudaMalloc(&d_A, max_elems * 2));
    CHECK_CUDA(cudaMalloc(&d_B, max_elems * 2));
    CHECK_CUDA(cudaMalloc(&d_C, max_elems * 2));
    CHECK_CUDA(cudaMalloc(&d_D_cutlass, max_elems * 2));
    CHECK_CUDA(cudaMalloc(&d_D_cublas, max_elems * 2));

    // Fill with random-ish data
    {
        std::vector<bf16> h(max_elems);
        for (int i = 0; i < max_elems; i++) h[i] = bf16(sinf(i * 0.001f) * 0.1f);
        CHECK_CUDA(cudaMemcpy(d_A, h.data(), max_elems*2, cudaMemcpyHostToDevice));
        for (int i = 0; i < max_elems; i++) h[i] = bf16(cosf(i * 0.001f) * 0.1f);
        CHECK_CUDA(cudaMemcpy(d_B, h.data(), max_elems*2, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_C, 0, max_elems * 2));
    }

    printf("%-20s %10s %10s %10s %10s %10s\n",
           "Config", "cuBLAS", "64x64x32", "128x128x32", "128x256x64", "256x128x64");
    printf("%-20s %10s %10s %10s %10s %10s\n",
           "", "(us)", "(us)", "(us)", "(us)", "(us)");
    printf("────────────────────────────────────────────────────────────────────────────────────\n");

    for (auto& tc : cases) {
        float cublas_us = bench_cublas(tc.M, tc.N, tc.K,
            reinterpret_cast<__nv_bfloat16*>(d_A),
            reinterpret_cast<__nv_bfloat16*>(d_B),
            reinterpret_cast<__nv_bfloat16*>(d_D_cublas),
            cublas);

        float c64   = bench_cutlass<Gemm_64x64x32>(tc.M, tc.N, tc.K, d_A, d_B, d_C, d_D_cutlass);
        float c128  = bench_cutlass<Gemm_128x128x32>(tc.M, tc.N, tc.K, d_A, d_B, d_C, d_D_cutlass);
        float c128x = bench_cutlass<Gemm_128x256x64>(tc.M, tc.N, tc.K, d_A, d_B, d_C, d_D_cutlass);
        float c256  = bench_cutlass<Gemm_256x128x64>(tc.M, tc.N, tc.K, d_A, d_B, d_C, d_D_cutlass);

        auto fmt = [](float us) -> const char* {
            static char bufs[5][16];
            static int idx = 0;
            char* buf = bufs[idx++ % 5];
            if (us < 0) snprintf(buf, 16, "n/a");
            else snprintf(buf, 16, "%.1f", us);
            return buf;
        };

        printf("%-20s %10s %10s %10s %10s %10s\n",
               tc.desc, fmt(cublas_us), fmt(c64), fmt(c128), fmt(c128x), fmt(c256));
    }

    printf("\nTFLOPS = 2*M*N*K / time_us / 1e6\n");
    printf("L4 peak bf16 tensor: ~120 TFLOPS\n");

    cublasDestroy(cublas);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_D_cutlass); cudaFree(d_D_cublas);
    return 0;
}
