// Probe CUTLASS Params struct to extract derived field formulas.
//
// For each raw param (lda, ldb, ldc, ldd, M, N, K, alpha, beta):
//   1. Construct Params with that raw param = 0, all else fixed
//   2. Construct Params with that raw param = 1 (or +1 from base)
//   3. Diff the two byte dumps → slope per derived field
//
// Output: JSON array of {offset, size, depends_on, slope, intercept}
//
// Compile:
//   nvcc -O2 -std=c++17 -arch=sm_89 \
//     -I$HOME/.cache/cutlass/include \
//     -o probe_cutlass_params probe_cutlass_params.cu
//
// Run:
//   ./probe_cutlass_params > cutlass_gemm_bf16_64x64x32.derivations.json

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <map>

using bf16 = cutlass::bfloat16_t;

// Must match the config that produced cutlass_gemm_bf16_sm89.ptx
using CutlassGemm = cutlass::gemm::device::Gemm<
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
    3, 8, 8>;

using GemmKernel = CutlassGemm::GemmKernel;

struct RawParams {
    int M, N, K;
    int lda, ldb, ldc, ldd;
    float alpha, beta;
};

// Construct the CUTLASS Params struct from raw params.
// Uses dummy device pointers (0x1000, 0x2000, etc.) since we only care about
// the derived fields, not the pointer values.
GemmKernel::Params make_params(const RawParams& r) {
    using namespace cutlass;
    using namespace cutlass::gemm;

    bf16* d_A = reinterpret_cast<bf16*>(0x1000);
    bf16* d_B = reinterpret_cast<bf16*>(0x2000);
    bf16* d_C = reinterpret_cast<bf16*>(0x3000);
    bf16* d_D = reinterpret_cast<bf16*>(0x4000);

    GemmCoord problem(r.M, r.N, r.K);
    CutlassGemm::ThreadblockSwizzle swizzle;
    GemmCoord grid_tiled = swizzle.get_tiled_shape(
        problem, {64, 64, 32}, 1);

    TensorRef<bfloat16_t, layout::RowMajor>    ref_A(d_A, layout::RowMajor(r.lda));
    TensorRef<bfloat16_t, layout::ColumnMajor> ref_B(d_B, layout::ColumnMajor(r.ldb));
    TensorRef<bfloat16_t, layout::RowMajor>    ref_C(d_C, layout::RowMajor(r.ldc));
    TensorRef<bfloat16_t, layout::RowMajor>    ref_D(d_D, layout::RowMajor(r.ldd));

    GemmKernel::Epilogue::OutputOp::Params epilogue_op(r.alpha, r.beta);

    return GemmKernel::Params{
        problem, grid_tiled, ref_A, ref_B, ref_C, ref_D,
        epilogue_op, nullptr, nullptr, nullptr, nullptr
    };
}

// Dump the Params struct as a byte array
void dump_bytes(const GemmKernel::Params& p, uint8_t* out) {
    memcpy(out, &p, sizeof(p));
}

// Read a field from the byte dump at a given offset and size
int64_t read_field(const uint8_t* bytes, int offset, int size) {
    int64_t val = 0;
    memcpy(&val, bytes + offset, size);
    return val;
}

float read_f32(const uint8_t* bytes, int offset) {
    float val;
    memcpy(&val, bytes + offset, 4);
    return val;
}

int main() {
    const int PARAM_SIZE = sizeof(GemmKernel::Params);
    fprintf(stderr, "Params struct size: %d bytes\n", PARAM_SIZE);

    // Base raw params — chosen to be non-zero so we can detect constant fields
    RawParams base = {
        .M = 256, .N = 512, .K = 128,
        .lda = 128, .ldb = 128, .ldc = 512, .ldd = 512,
        .alpha = 1.0f, .beta = 0.0f
    };

    // Get baseline byte dump
    auto base_params = make_params(base);
    std::vector<uint8_t> base_bytes(PARAM_SIZE);
    dump_bytes(base_params, base_bytes.data());

    // For each raw param, perturb it and observe which bytes change
    struct Perturbation {
        const char* name;
        RawParams perturbed;
    };

    // We perturb each param by a known delta
    std::vector<Perturbation> perturbations;

    // Integer params: perturb by +1
    {
        RawParams p = base; p.lda = base.lda + 1;
        perturbations.push_back({"lda", p});
    }
    {
        RawParams p = base; p.ldb = base.ldb + 1;
        perturbations.push_back({"ldb", p});
    }
    {
        RawParams p = base; p.ldc = base.ldc + 1;
        perturbations.push_back({"ldc", p});
    }
    {
        RawParams p = base; p.ldd = base.ldd + 1;
        perturbations.push_back({"ldd", p});
    }
    {
        // M perturbation: need to go from 256 to 320 (+64 = one tile) to avoid
        // fractional tile effects. But for slope detection, +1 is fine for fields
        // that depend linearly.
        RawParams p = base; p.M = base.M + 64;
        perturbations.push_back({"M", p});
    }
    {
        RawParams p = base; p.N = base.N + 64;
        perturbations.push_back({"N", p});
    }
    {
        RawParams p = base; p.K = base.K + 32;
        perturbations.push_back({"K", p});
    }

    // Also test with completely different raw params to verify linearity
    RawParams check = {
        .M = 1024, .N = 2560, .K = 2048,
        .lda = 2048, .ldb = 2048, .ldc = 2560, .ldd = 2560,
        .alpha = 1.0f, .beta = 0.0f
    };

    printf("{\n");
    printf("  \"param_size\": %d,\n", PARAM_SIZE);
    printf("  \"base\": {\"M\": %d, \"N\": %d, \"K\": %d, \"lda\": %d, \"ldb\": %d, \"ldc\": %d, \"ldd\": %d},\n",
           base.M, base.N, base.K, base.lda, base.ldb, base.ldc, base.ldd);

    // Dump field map: for each 4-byte or 8-byte aligned offset, dump the value
    printf("  \"fields\": [\n");
    bool first_field = true;

    // Pre-compute all perturbation and check byte dumps
    std::vector<std::vector<uint8_t>> pert_bytes_vec;
    for (auto& pert : perturbations) {
        auto p = make_params(pert.perturbed);
        std::vector<uint8_t> bytes(PARAM_SIZE);
        dump_bytes(p, bytes.data());
        pert_bytes_vec.push_back(bytes);
    }

    auto check_params = make_params(check);
    std::vector<uint8_t> check_bytes(PARAM_SIZE);
    dump_bytes(check_params, check_bytes.data());

    // Alpha/beta perturbation byte dumps
    RawParams pa = base; pa.alpha = 2.0f;
    auto pa_params = make_params(pa);
    std::vector<uint8_t> alpha_bytes(PARAM_SIZE);
    dump_bytes(pa_params, alpha_bytes.data());

    RawParams pb = base; pb.beta = 1.0f;
    auto pb_params = make_params(pb);
    std::vector<uint8_t> beta_bytes(PARAM_SIZE);
    dump_bytes(pb_params, beta_bytes.data());

    // Scan at 4-byte granularity. For u64 fields (detected by Phase 1),
    // the consumer will combine adjacent 4-byte entries.
    // We also emit 8-byte (u64) views at 8-byte-aligned offsets.
    for (int offset = 0; offset < PARAM_SIZE; offset += 4) {
        int32_t base_i32;
        memcpy(&base_i32, base_bytes.data() + offset, 4);
        int32_t check_i32;
        memcpy(&check_i32, check_bytes.data() + offset, 4);

        if (!first_field) printf(",\n");
        first_field = false;

        printf("    {\"offset\": %d, \"size\": 4, \"base_value\": %d, \"check_value\": %d",
               offset, base_i32, check_i32);

        // Check for float fields (alpha/beta)
        float base_f32;
        memcpy(&base_f32, base_bytes.data() + offset, 4);
        float alpha_f32, beta_f32;
        memcpy(&alpha_f32, alpha_bytes.data() + offset, 4);
        memcpy(&beta_f32, beta_bytes.data() + offset, 4);

        // Only compare floats if neither is NaN
        bool is_alpha = !isnan(alpha_f32) && !isnan(base_f32) && (alpha_f32 != base_f32);
        bool is_beta = !isnan(beta_f32) && !isnan(base_f32) && (beta_f32 != base_f32);
        if (is_alpha || is_beta) {
            printf(", \"base_f32\": %.6f", base_f32);
            if (is_alpha) printf(", \"alpha_f32\": %.6f", alpha_f32);
            if (is_beta) printf(", \"beta_f32\": %.6f", beta_f32);
        }

        // Check integer dependencies
        std::map<std::string, int32_t> deps;
        for (size_t i = 0; i < perturbations.size(); i++) {
            int32_t pert_i32;
            memcpy(&pert_i32, pert_bytes_vec[i].data() + offset, 4);
            int32_t diff = pert_i32 - base_i32;
            if (diff != 0) {
                deps[perturbations[i].name] = diff;
            }
        }

        if (!deps.empty()) {
            printf(", \"depends_on\": {");
            bool first_dep = true;
            for (auto& [name, slope] : deps) {
                if (!first_dep) printf(", ");
                first_dep = false;
                printf("\"%s\": %d", name.c_str(), slope);
            }
            printf("}");
        }

        printf("}");
    }

    printf("\n  ]\n");
    printf("}\n");

    return 0;
}
