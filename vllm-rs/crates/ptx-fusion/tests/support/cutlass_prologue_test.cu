// GPU correctness tests for CUTLASS bf16 GEMM prologue fusion.
//
// Mode 1: explicit A-loads
//   ./test <original.ptx> <modified.ptx>
//   Compares original cp.async vs explicit ld+st replacement.
//
// Mode 2: fused rms_norm
//   ./test --fused <fused.ptx>
//   Compares rms_norm+GEMM (separate) vs fused kernel (single launch).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/gemm/device/gemm.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } \
} while(0)

#define CHECK_CU(call) do { \
    CUresult e = call; \
    if (e != CUDA_SUCCESS) { const char* m; cuGetErrorString(e, &m); fprintf(stderr, "CU %s:%d: %s\n", __FILE__, __LINE__, m); exit(1); } \
} while(0)

using CutlassGemm = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        cutlass::bfloat16_t, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
    3, 8, 8
>;

using GemmKernel = CutlassGemm::GemmKernel;
using bf16 = cutlass::bfloat16_t;

std::string read_file(const char* path) {
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    return std::string((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
}

std::string find_entry(const std::string& ptx, const char* hint = "_ZN7cutlass") {
    size_t pos = 0;
    while ((pos = ptx.find(".entry", pos)) != std::string::npos) {
        // Try the hint first, then any name
        size_t ns = ptx.find(hint, pos);
        if (ns == std::string::npos || ns > pos + 500) {
            // Try fused_ prefix
            ns = ptx.find("fused_", pos);
        }
        if (ns != std::string::npos && ns < pos + 500) {
            size_t paren = ptx.find('(', ns);
            if (paren != std::string::npos) {
                std::string name = ptx.substr(ns, paren - ns);
                while (!name.empty() && (name.back() <= ' ')) name.pop_back();
                return name;
            }
        }
        pos += 6;
    }
    fprintf(stderr, "no entry found in PTX\n"); exit(1);
}

void launch_ptx(const std::string& ptx, const std::string& entry,
                const void* params, size_t params_sz,
                dim3 grid, dim3 block, int smem) {
    CUmodule mod; CUfunction func;
    CHECK_CU(cuModuleLoadData(&mod, ptx.c_str()));
    CHECK_CU(cuModuleGetFunction(&func, mod, entry.c_str()));
    if (smem > 48*1024)
        CHECK_CU(cuFuncSetAttribute(func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
    size_t sz = params_sz;
    void* extra[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER, const_cast<void*>(params),
        CU_LAUNCH_PARAM_BUFFER_SIZE, &sz,
        CU_LAUNCH_PARAM_END
    };
    CHECK_CU(cuLaunchKernel(func, grid.x,grid.y,grid.z,
                            block.x,block.y,block.z, smem, 0, nullptr, extra));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CU(cuModuleUnload(mod));
}

GemmKernel::Params make_kernel_params(
    bf16* d_A, bf16* d_B, bf16* d_C, bf16* d_D,
    int M, int N, int K, int lda, int ldb, int ldc, int ldd)
{
    using namespace cutlass;
    using namespace cutlass::gemm;
    GemmCoord problem(M, N, K);
    CutlassGemm::ThreadblockSwizzle swizzle;
    GemmCoord grid_tiled = swizzle.get_tiled_shape(problem, {64, 64, 32}, 1);
    TensorRef<bfloat16_t, layout::RowMajor> ref_A(d_A, layout::RowMajor(lda));
    TensorRef<bfloat16_t, layout::ColumnMajor> ref_B(d_B, layout::ColumnMajor(ldb));
    TensorRef<bfloat16_t, layout::RowMajor> ref_C(d_C, layout::RowMajor(ldc));
    TensorRef<bfloat16_t, layout::RowMajor> ref_D(d_D, layout::RowMajor(ldd));
    GemmKernel::Epilogue::OutputOp::Params epilogue_op(1.0f, 0.0f);
    return GemmKernel::Params{
        problem, grid_tiled, ref_A, ref_B, ref_C, ref_D,
        epilogue_op, nullptr, nullptr, nullptr, nullptr
    };
}

// ── Mode 1: explicit A-loads test ──

int test_explicit(const char* orig_path, const char* mod_path) {
    const int M = 64, N = 64, K = 32;
    const int lda = K, ldb = K, ldc = N, ldd = N;

    std::vector<bf16> h_A(M*K), h_B(N*K), h_C(M*N);
    for (int i = 0; i < M*K; i++) h_A[i] = bf16(sinf(i*0.037f-0.5f)*0.5f);
    for (int i = 0; i < N*K; i++) h_B[i] = bf16(cosf(i*0.023f+0.3f)*0.5f);
    for (int i = 0; i < M*N; i++) h_C[i] = bf16(0.0f);

    bf16 *d_A, *d_B, *d_C, *d_ref, *d_out_orig, *d_out_mod;
    CHECK_CUDA(cudaMalloc(&d_A, M*K*2));
    CHECK_CUDA(cudaMalloc(&d_B, N*K*2));
    CHECK_CUDA(cudaMalloc(&d_C, M*N*2));
    CHECK_CUDA(cudaMalloc(&d_ref, M*N*2));
    CHECK_CUDA(cudaMalloc(&d_out_orig, M*N*2));
    CHECK_CUDA(cudaMalloc(&d_out_mod, M*N*2));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M*K*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), N*K*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), M*N*2, cudaMemcpyHostToDevice));

    // Reference via CUTLASS API
    {
        CHECK_CUDA(cudaMemset(d_ref, 0, M*N*2));
        CutlassGemm gemm_op;
        CutlassGemm::Arguments args({M,N,K},
            {d_A, lda}, {d_B, ldb}, {d_C, ldc}, {d_ref, ldd}, {1.0f, 0.0f});
        gemm_op(args); CHECK_CUDA(cudaDeviceSynchronize());
    }

    auto params_orig = make_kernel_params(d_A, d_B, d_C, d_out_orig, M,N,K, lda,ldb,ldc,ldd);
    auto params_mod  = make_kernel_params(d_A, d_B, d_C, d_out_mod,  M,N,K, lda,ldb,ldc,ldd);

    printf("Params: %zu bytes, SharedStorage: %zu bytes\n",
           sizeof(GemmKernel::Params), sizeof(GemmKernel::SharedStorage));

    std::string orig_ptx = read_file(orig_path);
    std::string mod_ptx = read_file(mod_path);
    std::string entry_orig = find_entry(orig_ptx);
    std::string entry_mod = find_entry(mod_ptx);

    printf("Entry: %.80s...\n", entry_orig.c_str());

    CutlassGemm::ThreadblockSwizzle swizzle;
    dim3 grid = swizzle.get_grid_shape(params_orig.grid_tiled_shape);
    dim3 block(GemmKernel::kThreadCount, 1, 1);
    int smem = int(sizeof(GemmKernel::SharedStorage));
    printf("Grid: (%d,%d,%d)  Block: %d  SMEM: %d\n", grid.x,grid.y,grid.z, block.x, smem);

    CHECK_CUDA(cudaMemset(d_out_orig, 0, M*N*2));
    launch_ptx(orig_ptx, entry_orig, &params_orig, sizeof(params_orig), grid, block, smem);

    CHECK_CUDA(cudaMemset(d_out_mod, 0, M*N*2));
    launch_ptx(mod_ptx, entry_mod, &params_mod, sizeof(params_mod), grid, block, smem);

    std::vector<bf16> out_ref(M*N), out_o(M*N), out_m(M*N);
    CHECK_CUDA(cudaMemcpy(out_ref.data(), d_ref, M*N*2, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(out_o.data(), d_out_orig, M*N*2, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(out_m.data(), d_out_mod, M*N*2, cudaMemcpyDeviceToHost));

    float sum = 0;
    for (int i = 0; i < M*N; i++) sum += fabsf(float(out_ref[i]));
    if (sum < 0.01f) { fprintf(stderr, "FAIL: reference is zeros\n"); return 1; }

    float max_orig = 0, max_mod = 0, max_direct = 0;
    for (int i = 0; i < M*N; i++) {
        max_orig = fmaxf(max_orig, fabsf(float(out_o[i]) - float(out_ref[i])));
        max_mod = fmaxf(max_mod, fabsf(float(out_m[i]) - float(out_ref[i])));
        max_direct = fmaxf(max_direct, fabsf(float(out_m[i]) - float(out_o[i])));
    }

    printf("Reference: [%.4f, %.4f, %.4f, %.4f]\n", float(out_ref[0]),float(out_ref[1]),float(out_ref[2]),float(out_ref[3]));
    printf("Original:  [%.4f, %.4f, %.4f, %.4f]\n", float(out_o[0]),float(out_o[1]),float(out_o[2]),float(out_o[3]));
    printf("Modified:  [%.4f, %.4f, %.4f, %.4f]\n", float(out_m[0]),float(out_m[1]),float(out_m[2]),float(out_m[3]));
    printf("Max diff (orig vs ref): %.2e\n", max_orig);
    printf("Max diff (mod vs ref):  %.2e\n", max_mod);
    printf("Max diff (mod vs orig): %.2e\n", max_direct);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_ref); cudaFree(d_out_orig); cudaFree(d_out_mod);

    if (max_orig > 1e-3f || max_direct > 1e-3f) {
        fprintf(stderr, "FAIL\n"); return 1;
    }
    printf("PASS: explicit A-loads produce identical output (orig_vs_ref=%.2e, mod_vs_orig=%.2e)\n",
           max_orig, max_direct);
    return 0;
}

// ── Mode 2: fused rms_norm test ──

// CPU rms_norm reference: normalize input, produce bf16 output
void cpu_rms_norm_bf16(
    const bf16* input, const bf16* weight, bf16* output,
    int rows, int hidden, float epsilon)
{
    for (int m = 0; m < rows; m++) {
        float sum_sq = 0;
        for (int k = 0; k < hidden; k++) {
            float v = float(input[m * hidden + k]);
            sum_sq += v * v;
        }
        float inv_rms = 1.0f / sqrtf(sum_sq / hidden + epsilon);
        for (int k = 0; k < hidden; k++) {
            float v = float(input[m * hidden + k]);
            float w = float(weight[k]);
            output[m * hidden + k] = bf16(v * w * inv_rms);
        }
    }
}

int test_fused(const char* fused_path) {
    // For the fused test: hidden = K = 32 (one CUTLASS K-tile)
    // input is M x hidden, weight is hidden, B is N x K (col-major)
    const int M = 64, N = 64, K = 32;
    const int hidden = K;  // rms_norm hidden = GEMM K
    const float epsilon = 1e-5f;
    const int lda = K, ldb = K, ldc = N, ldd = N;

    // Host data
    std::vector<bf16> h_input(M * hidden), h_weight(hidden), h_B(N * K), h_C(M * N);
    for (int i = 0; i < M * hidden; i++)
        h_input[i] = bf16(sinf(i * 0.037f - 0.5f) * 0.5f);
    for (int i = 0; i < hidden; i++)
        h_weight[i] = bf16(1.0f + i * 0.01f);
    for (int i = 0; i < N * K; i++)
        h_B[i] = bf16(cosf(i * 0.023f + 0.3f) * 0.5f);
    for (int i = 0; i < M * N; i++) h_C[i] = bf16(0.0f);

    // CPU reference: rms_norm then GEMM
    std::vector<bf16> h_A_normalized(M * K);
    cpu_rms_norm_bf16(h_input.data(), h_weight.data(), h_A_normalized.data(),
                      M, hidden, epsilon);

    // Device memory
    bf16 *d_input, *d_weight, *d_A_norm, *d_B, *d_C, *d_ref, *d_fused;
    CHECK_CUDA(cudaMalloc(&d_input, M * hidden * 2));
    CHECK_CUDA(cudaMalloc(&d_weight, hidden * 2));
    CHECK_CUDA(cudaMalloc(&d_A_norm, M * K * 2));
    CHECK_CUDA(cudaMalloc(&d_B, N * K * 2));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * 2));
    CHECK_CUDA(cudaMalloc(&d_ref, M * N * 2));
    CHECK_CUDA(cudaMalloc(&d_fused, M * N * 2));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), M*hidden*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, h_weight.data(), hidden*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A_norm, h_A_normalized.data(), M*K*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), N*K*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), M*N*2, cudaMemcpyHostToDevice));

    // 1. Reference: CUTLASS GEMM with pre-normalized A
    {
        CHECK_CUDA(cudaMemset(d_ref, 0, M*N*2));
        CutlassGemm gemm_op;
        CutlassGemm::Arguments args({M,N,K},
            {d_A_norm, lda}, {d_B, ldb}, {d_C, ldc}, {d_ref, ldd}, {1.0f, 0.0f});
        auto status = gemm_op(args);
        CHECK_CUDA(cudaDeviceSynchronize());
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS reference GEMM failed\n"); return 1;
        }
    }

    // 2. Fused kernel: rms_norm + GEMM in one launch
    // The fused kernel params layout:
    //   [0:8]   weight_ptr (u64)
    //   [8:12]  epsilon (f32)
    //   [12:16] hidden (u32)
    //   [16:384] gemm_params (368 bytes, with input_ptr in A_ptr slot)
    {
        // Build GEMM params with input_ptr in the A_ptr slot
        auto gemm_params = make_kernel_params(
            reinterpret_cast<bf16*>(d_input),  // A_ptr = input_ptr!
            reinterpret_cast<bf16*>(d_B),
            reinterpret_cast<bf16*>(d_C),
            reinterpret_cast<bf16*>(d_fused),
            M, N, K, lda, ldb, ldc, ldd);

        // Construct fused params buffer
        struct alignas(8) {
            uint64_t weight_ptr;
            float epsilon;
            uint32_t hidden;
            // gemm_params follows at offset 16, but alignment might add padding
        } fused_prefix;

        fused_prefix.weight_ptr = reinterpret_cast<uint64_t>(d_weight);
        fused_prefix.epsilon = epsilon;
        fused_prefix.hidden = hidden;

        // Flat buffer: prefix + gemm_params
        // Alignment: prefix is 16 bytes, gemm_params needs align 8 → offset 16 is fine
        std::vector<uint8_t> fused_params(16 + sizeof(gemm_params));
        memcpy(fused_params.data(), &fused_prefix, 16);
        memcpy(fused_params.data() + 16, &gemm_params, sizeof(gemm_params));

        std::string fused_ptx = read_file(fused_path);
        std::string fused_entry = find_entry(fused_ptx, "fused_");

        printf("Fused entry: %s\n", fused_entry.c_str());
        printf("Fused params: %zu bytes (prefix=%zu + gemm=%zu)\n",
               fused_params.size(), sizeof(fused_prefix), sizeof(gemm_params));

        CutlassGemm::ThreadblockSwizzle swizzle;
        dim3 grid = swizzle.get_grid_shape(gemm_params.grid_tiled_shape);
        dim3 block(GemmKernel::kThreadCount, 1, 1);
        int smem = int(sizeof(GemmKernel::SharedStorage));

        printf("Grid: (%d,%d,%d)  Block: %d  SMEM: %d\n",
               grid.x, grid.y, grid.z, block.x, smem);

        CHECK_CUDA(cudaMemset(d_fused, 0, M*N*2));
        launch_ptx(fused_ptx, fused_entry, fused_params.data(), fused_params.size(),
                   grid, block, smem);
    }

    // Compare
    std::vector<bf16> out_ref(M*N), out_fused(M*N);
    CHECK_CUDA(cudaMemcpy(out_ref.data(), d_ref, M*N*2, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(out_fused.data(), d_fused, M*N*2, cudaMemcpyDeviceToHost));

    float sum = 0;
    for (int i = 0; i < M*N; i++) sum += fabsf(float(out_ref[i]));
    if (sum < 0.01f) {
        fprintf(stderr, "FAIL: reference output is all zeros\n"); return 1;
    }

    float max_diff = 0;
    int first_bad = -1;
    for (int i = 0; i < M*N; i++) {
        float diff = fabsf(float(out_fused[i]) - float(out_ref[i]));
        if (diff > max_diff) {
            max_diff = diff;
            if (first_bad < 0 && diff > 0.01f) first_bad = i;
        }
    }

    printf("Reference: [%.4f, %.4f, %.4f, %.4f]\n",
           float(out_ref[0]), float(out_ref[1]), float(out_ref[2]), float(out_ref[3]));
    printf("Fused:     [%.4f, %.4f, %.4f, %.4f]\n",
           float(out_fused[0]), float(out_fused[1]), float(out_fused[2]), float(out_fused[3]));
    printf("Max diff:  %.2e\n", max_diff);

    // Check for all-zeros in fused output (common failure mode)
    float fused_sum = 0;
    for (int i = 0; i < M*N; i++) fused_sum += fabsf(float(out_fused[i]));
    if (fused_sum < 0.01f) {
        fprintf(stderr, "FAIL: fused output is all zeros (kernel likely crashed)\n");
        return 1;
    }

    cudaFree(d_input); cudaFree(d_weight); cudaFree(d_A_norm);
    cudaFree(d_B); cudaFree(d_C); cudaFree(d_ref); cudaFree(d_fused);

    // Allow some tolerance for bf16 rounding + rsqrt.approx
    if (max_diff > 0.05f) {
        fprintf(stderr, "FAIL: fused kernel diverges from reference (%.2e)\n", max_diff);
        if (first_bad >= 0) {
            fprintf(stderr, "  first bad at [%d]: ref=%.6f fused=%.6f\n",
                    first_bad, float(out_ref[first_bad]), float(out_fused[first_bad]));
        }
        return 1;
    }

    printf("PASS: fused rms_norm+CUTLASS GEMM matches reference (max_diff=%.2e)\n", max_diff);
    return 0;
}

int main(int argc, char** argv) {
    CHECK_CU(cuInit(0));

    if (argc == 3 && std::string(argv[1]) == "--fused") {
        return test_fused(argv[2]);
    } else if (argc == 3) {
        return test_explicit(argv[1], argv[2]);
    } else {
        fprintf(stderr, "usage:\n");
        fprintf(stderr, "  %s <original.ptx> <modified.ptx>  (explicit A-load test)\n", argv[0]);
        fprintf(stderr, "  %s --fused <fused.ptx>            (fused rms_norm test)\n", argv[0]);
        return 1;
    }
}
