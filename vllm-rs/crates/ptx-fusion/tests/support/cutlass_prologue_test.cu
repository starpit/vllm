// GPU correctness test: CUTLASS bf16 GEMM with explicit A-loads vs original.
//
// Strategy: use device::Gemm::operator() for the reference output, then
// launch original and modified PTX via driver API with manually constructed
// kernel-level params. Compare all three.

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

std::string read_file(const char* path) {
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    return std::string((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
}

std::string find_entry(const std::string& ptx) {
    size_t pos = 0;
    while ((pos = ptx.find(".entry", pos)) != std::string::npos) {
        size_t ns = ptx.find("_ZN7cutlass", pos);
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

// Construct kernel-level params by replicating what device::Gemm::initialize() does.
// This avoids needing access to the private params_ member.
GemmKernel::Params make_kernel_params(
    cutlass::bfloat16_t* d_A, cutlass::bfloat16_t* d_B,
    cutlass::bfloat16_t* d_C, cutlass::bfloat16_t* d_D,
    int M, int N, int K, int lda, int ldb, int ldc, int ldd)
{
    using namespace cutlass;
    using namespace cutlass::gemm;

    GemmCoord problem(M, N, K);

    CutlassGemm::ThreadblockSwizzle swizzle;
    GemmCoord grid_tiled = swizzle.get_tiled_shape(
        problem, {64, 64, 32}, 1);

    // Construct TensorRefs with non-const pointers (as kernel expects)
    TensorRef<bfloat16_t, layout::RowMajor> ref_A(d_A, layout::RowMajor(lda));
    TensorRef<bfloat16_t, layout::ColumnMajor> ref_B(d_B, layout::ColumnMajor(ldb));
    TensorRef<bfloat16_t, layout::RowMajor> ref_C(d_C, layout::RowMajor(ldc));
    TensorRef<bfloat16_t, layout::RowMajor> ref_D(d_D, layout::RowMajor(ldd));

    typename GemmKernel::Epilogue::OutputOp::Params epilogue_op(1.0f, 0.0f);

    return GemmKernel::Params{
        problem, grid_tiled,
        ref_A, ref_B, ref_C, ref_D,
        epilogue_op,
        nullptr,  // workspace/semaphore
        nullptr, nullptr, nullptr  // gather/scatter indices
    };
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <original.ptx> <modified.ptx>\n", argv[0]);
        return 1;
    }
    CHECK_CU(cuInit(0));

    using bf16 = cutlass::bfloat16_t;
    const int M = 64, N = 64, K = 32;
    const int lda = K, ldb = K, ldc = N, ldd = N;

    // Host data
    std::vector<bf16> h_A(M*K), h_B(N*K), h_C(M*N);
    for (int i = 0; i < M*K; i++)
        h_A[i] = bf16(sinf(i*0.037f - 0.5f) * 0.5f);
    for (int i = 0; i < N*K; i++)
        h_B[i] = bf16(cosf(i*0.023f + 0.3f) * 0.5f);
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

    // 1. CUTLASS API reference
    {
        CHECK_CUDA(cudaMemset(d_ref, 0, M*N*2));
        CutlassGemm gemm_op;
        CutlassGemm::Arguments args({M,N,K},
            {d_A, lda}, {d_B, ldb}, {d_C, ldc}, {d_ref, ldd}, {1.0f, 0.0f});
        auto status = gemm_op(args);
        CHECK_CUDA(cudaDeviceSynchronize());
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS reference failed\n"); return 1;
        }
    }

    // Build kernel params for PTX launches
    auto params_orig = make_kernel_params(d_A, d_B, d_C, d_out_orig, M,N,K, lda,ldb,ldc,ldd);
    auto params_mod  = make_kernel_params(d_A, d_B, d_C, d_out_mod,  M,N,K, lda,ldb,ldc,ldd);

    printf("Params: %zu bytes, SharedStorage: %zu bytes\n",
           sizeof(GemmKernel::Params), sizeof(GemmKernel::SharedStorage));

    std::string orig_ptx = read_file(argv[1]);
    std::string mod_ptx = read_file(argv[2]);
    std::string entry_orig = find_entry(orig_ptx);
    std::string entry_mod = find_entry(mod_ptx);

    printf("Entry: %.80s...\n", entry_orig.c_str());

    CutlassGemm::ThreadblockSwizzle swizzle;
    dim3 grid = swizzle.get_grid_shape(params_orig.grid_tiled_shape);
    dim3 block(GemmKernel::kThreadCount, 1, 1);
    int smem = int(sizeof(GemmKernel::SharedStorage));

    printf("Grid: (%d,%d,%d)  Block: %d  SMEM: %d\n",
           grid.x, grid.y, grid.z, block.x, smem);

    // 2. Original PTX via driver API
    CHECK_CUDA(cudaMemset(d_out_orig, 0, M*N*2));
    launch_ptx(orig_ptx, entry_orig, &params_orig, sizeof(params_orig),
               grid, block, smem);

    // 3. Modified PTX via driver API
    CHECK_CUDA(cudaMemset(d_out_mod, 0, M*N*2));
    launch_ptx(mod_ptx, entry_mod, &params_mod, sizeof(params_mod),
               grid, block, smem);

    // Read back all three
    std::vector<bf16> out_ref(M*N), out_o(M*N), out_m(M*N);
    CHECK_CUDA(cudaMemcpy(out_ref.data(), d_ref, M*N*2, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(out_o.data(), d_out_orig, M*N*2, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(out_m.data(), d_out_mod, M*N*2, cudaMemcpyDeviceToHost));

    // Sanity
    float sum = 0;
    for (int i = 0; i < M*N; i++) sum += fabsf(float(out_ref[i]));
    if (sum < 0.01f) {
        fprintf(stderr, "FAIL: CUTLASS reference output is all zeros\n"); return 1;
    }

    // Compare: original PTX vs reference
    float max_diff_orig = 0;
    for (int i = 0; i < M*N; i++) {
        float diff = fabsf(float(out_o[i]) - float(out_ref[i]));
        max_diff_orig = fmaxf(max_diff_orig, diff);
    }

    // Compare: modified PTX vs reference
    float max_diff_mod = 0;
    for (int i = 0; i < M*N; i++) {
        float diff = fabsf(float(out_m[i]) - float(out_ref[i]));
        max_diff_mod = fmaxf(max_diff_mod, diff);
    }

    // Compare: modified vs original directly
    float max_diff_direct = 0;
    for (int i = 0; i < M*N; i++) {
        float diff = fabsf(float(out_m[i]) - float(out_o[i]));
        max_diff_direct = fmaxf(max_diff_direct, diff);
    }

    printf("Reference: [%.4f, %.4f, %.4f, %.4f]\n",
           float(out_ref[0]), float(out_ref[1]), float(out_ref[2]), float(out_ref[3]));
    printf("Original:  [%.4f, %.4f, %.4f, %.4f]\n",
           float(out_o[0]), float(out_o[1]), float(out_o[2]), float(out_o[3]));
    printf("Modified:  [%.4f, %.4f, %.4f, %.4f]\n",
           float(out_m[0]), float(out_m[1]), float(out_m[2]), float(out_m[3]));
    printf("Max diff (orig vs ref):     %.2e\n", max_diff_orig);
    printf("Max diff (modified vs ref): %.2e\n", max_diff_mod);
    printf("Max diff (modified vs orig): %.2e\n", max_diff_direct);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_ref); cudaFree(d_out_orig); cudaFree(d_out_mod);

    int pass = 1;
    if (max_diff_orig > 1e-3f) {
        fprintf(stderr, "FAIL: original PTX diverges from reference (%.2e)\n"
                "  This means the params struct layout doesn't match the PTX kernel.\n",
                max_diff_orig);
        pass = 0;
    }
    if (max_diff_direct > 1e-3f) {
        fprintf(stderr, "FAIL: modified PTX diverges from original (%.2e)\n", max_diff_direct);
        pass = 0;
    }

    if (pass) {
        printf("PASS: explicit A-loads produce identical output "
               "(orig_vs_ref=%.2e, mod_vs_orig=%.2e)\n",
               max_diff_orig, max_diff_direct);
    }
    return pass ? 0 : 1;
}
