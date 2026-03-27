// Build script: compile vllm-cuda kernels to PTX and CUTLASS configs to PTX + derivations.
// Only runs when the "cuda" feature is enabled.
//
// Artifacts land in kernels/ (checked into git). On a normal build where kernels/
// already contains up-to-date files, this is a no-op. The CUDA toolchain is only
// needed when kernel sources change.

use std::path::PathBuf;
use std::process::Command;

/// A CUTLASS bf16 GEMM configuration to compile and probe.
struct CutlassConfig {
    name: &'static str,
    tb_shape: (u32, u32, u32),
    warp_shape: (u32, u32, u32),
    epilogue_vec: u32,
    stages: u32,
    align_a: u32,
    align_b: u32,
}

const CUTLASS_CONFIGS: &[CutlassConfig] = &[
    CutlassConfig {
        name: "cutlass_bf16_64x64x32_sm89",
        tb_shape: (64, 64, 32),
        warp_shape: (32, 32, 32),
        epilogue_vec: 4,
        stages: 3,
        align_a: 8,
        align_b: 8,
    },
    CutlassConfig {
        name: "cutlass_bf16_64x128x32_sm89",
        tb_shape: (64, 128, 32),
        warp_shape: (32, 64, 32),
        epilogue_vec: 8,
        stages: 3,
        align_a: 8,
        align_b: 8,
    },
    CutlassConfig {
        name: "cutlass_bf16_128x128x32_sm89",
        tb_shape: (128, 128, 32),
        warp_shape: (64, 64, 32),
        epilogue_vec: 8,
        stages: 3,
        align_a: 8,
        align_b: 8,
    },
    CutlassConfig {
        name: "cutlass_bf16_128x128x64_sm89",
        tb_shape: (128, 128, 64),
        warp_shape: (64, 64, 64),
        epilogue_vec: 8,
        stages: 3,
        align_a: 8,
        align_b: 8,
    },
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let vllm_cuda_csrc = manifest_dir
        .parent()
        .unwrap()
        .join("vllm-cuda")
        .join("csrc");
    let kernel_dir = manifest_dir.join("kernels");
    let support_dir = manifest_dir.join("tests").join("support");
    let nvcc = find_nvcc();

    // ── vllm-cuda kernels ──

    compile_ptx(
        &nvcc,
        &vllm_cuda_csrc,
        &kernel_dir,
        "vllm_rms_norm",
        r#"
#include "layernorm_kernels.cu"
template __global__ void rms_norm_kernel<float>(
    float* __restrict__, const float* __restrict__,
    const float* __restrict__, float, int);
"#,
    );

    compile_ptx(
        &nvcc,
        &vllm_cuda_csrc,
        &kernel_dir,
        "vllm_silu_mul",
        r#"
#include "activation_kernels.cu"
template __global__ void act_and_mul_kernel<silu, float>(
    float* __restrict__, const float* __restrict__,
    const float* __restrict__, int);
"#,
    );

    // ── CUTLASS configs: compile PTX + probe derivations ──
    // Only triggers when build.rs changes (configs are defined here).
    // The compile_cutlass_configs.cu in tests/support/ is the reference
    // for manual compilation; build.rs generates equivalent source inline.

    println!(
        "cargo:rerun-if-changed={}",
        support_dir.join("compile_cutlass_configs.cu").display()
    );

    if let Some(cutlass_inc) = find_cutlass_include() {
        for cfg in CUTLASS_CONFIGS {
            compile_cutlass_ptx(&nvcc, &cutlass_inc, &kernel_dir, cfg);
            probe_cutlass_derivations(&nvcc, &cutlass_inc, &kernel_dir, cfg);
        }
    } else {
        // No CUTLASS headers — check if pre-compiled artifacts exist
        let mut missing = false;
        for cfg in CUTLASS_CONFIGS {
            let ptx = kernel_dir.join(format!("{}.ptx", cfg.name));
            let json = kernel_dir.join(format!("{}.derivations.json", cfg.name));
            if !ptx.exists() || !json.exists() {
                missing = true;
                println!(
                    "cargo:warning=missing CUTLASS artifacts for {} (no CUTLASS include found)",
                    cfg.name
                );
            }
        }
        if !missing {
            println!("cargo:warning=using pre-compiled CUTLASS artifacts from kernels/");
        }
    }
}

/// Check if an output file exists and is newer than build.rs (our source of truth
/// for CUTLASS configs). If so, skip recompilation.
fn is_up_to_date(output: &std::path::Path) -> bool {
    if !output.exists() {
        return false;
    }
    let build_rs = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("build.rs");
    match (
        std::fs::metadata(output).and_then(|m| m.modified()),
        std::fs::metadata(&build_rs).and_then(|m| m.modified()),
    ) {
        (Ok(out_time), Ok(build_time)) => out_time >= build_time,
        _ => false,
    }
}

fn find_nvcc() -> String {
    if let Ok(home) = std::env::var("CUDA_HOME") {
        let p = format!("{home}/bin/nvcc");
        if std::path::Path::new(&p).exists() {
            return p;
        }
    }
    for ver in ["12.9", "12.8", "12.6", "12.4", "12.2", "12.0"] {
        let p = format!("/usr/local/cuda-{ver}/bin/nvcc");
        if std::path::Path::new(&p).exists() {
            return p;
        }
    }
    "nvcc".to_string()
}

fn find_cutlass_include() -> Option<String> {
    if let Ok(v) = std::env::var("CUTLASS_INCLUDE") {
        if std::path::Path::new(&v).join("cutlass/cutlass.h").exists() {
            return Some(v);
        }
    }
    let home = std::env::var("HOME").unwrap_or_default();
    for c in [
        format!("{home}/.cache/cutlass/include"),
        "/usr/local/cutlass/include".to_string(),
    ] {
        if std::path::Path::new(&c).join("cutlass/cutlass.h").exists() {
            return Some(c);
        }
    }
    None
}

fn compile_ptx(
    nvcc: &str,
    include_dir: &std::path::Path,
    out_dir: &std::path::Path,
    name: &str,
    source: &str,
) {
    let tmp_cu = std::env::temp_dir().join(format!("{name}_ferrite.cu"));
    let out_ptx = out_dir.join(format!("{name}.ptx"));

    std::fs::write(&tmp_cu, source).expect("write temp .cu");
    println!("cargo:rerun-if-changed={}", include_dir.display());

    let output = Command::new(nvcc)
        .args([
            "-ptx",
            "-arch=sm_89",
            &format!("-I{}", include_dir.display()),
            tmp_cu.to_str().unwrap(),
            "-o",
            out_ptx.to_str().unwrap(),
        ])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            println!(
                "cargo:warning=compiled {name}.ptx ({} bytes)",
                std::fs::metadata(&out_ptx).map(|m| m.len()).unwrap_or(0)
            );
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            println!("cargo:warning=nvcc failed for {name}: {stderr}");
        }
        Err(e) => {
            println!("cargo:warning=nvcc not found for {name}: {e}");
        }
    }

    let _ = std::fs::remove_file(&tmp_cu);
}

fn cutlass_gemm_typedef(cfg: &CutlassConfig, alias: &str) -> String {
    let (tm, tn, tk) = cfg.tb_shape;
    let (wm, wn, wk) = cfg.warp_shape;
    let ev = cfg.epilogue_vec;
    let stages = cfg.stages;
    let (aa, ab) = (cfg.align_a, cfg.align_b);

    format!(
        r#"using {alias} = cutlass::gemm::device::Gemm<
    bf16, cutlass::layout::RowMajor,
    bf16, cutlass::layout::ColumnMajor,
    bf16, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<{tm}, {tn}, {tk}>,
    cutlass::gemm::GemmShape<{wm}, {wn}, {wk}>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<bf16, {ev}, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
    {stages}, {aa}, {ab}>;
"#
    )
}

fn compile_cutlass_ptx(
    nvcc: &str,
    cutlass_include: &str,
    out_dir: &std::path::Path,
    cfg: &CutlassConfig,
) {
    let out_ptx = out_dir.join(format!("{}.ptx", cfg.name));

    // Skip if PTX already exists and is newer than build.rs
    if is_up_to_date(&out_ptx) {
        return;
    }

    let source = format!(
        r#"#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
using bf16 = cutlass::bfloat16_t;
{typedef}
template __global__ void cutlass::Kernel<TheGemm::GemmKernel>(TheGemm::GemmKernel::Params);
"#,
        typedef = cutlass_gemm_typedef(cfg, "TheGemm")
    );

    let tmp_cu = std::env::temp_dir().join(format!("{}_ferrite.cu", cfg.name));
    std::fs::write(&tmp_cu, &source).expect("write temp .cu");

    let output = Command::new(nvcc)
        .args([
            "-ptx",
            "-arch=sm_89",
            "-O2",
            "-std=c++17",
            &format!("-I{cutlass_include}"),
            tmp_cu.to_str().unwrap(),
            "-o",
            out_ptx.to_str().unwrap(),
        ])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            println!(
                "cargo:warning=compiled {}.ptx ({} bytes)",
                cfg.name,
                std::fs::metadata(&out_ptx).map(|m| m.len()).unwrap_or(0)
            );
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            let short: String = stderr.lines().take(5).collect::<Vec<_>>().join("\n");
            println!("cargo:warning=nvcc failed for {}: {short}", cfg.name);
        }
        Err(e) => {
            println!("cargo:warning=nvcc not found for {}: {e}", cfg.name);
        }
    }

    let _ = std::fs::remove_file(&tmp_cu);
}

fn probe_cutlass_derivations(
    nvcc: &str,
    cutlass_include: &str,
    out_dir: &std::path::Path,
    cfg: &CutlassConfig,
) {
    let (tm, tn, tk) = cfg.tb_shape;
    let out_json = out_dir.join(format!("{}.derivations.json", cfg.name));

    // Skip if derivations already exist and are newer than build.rs
    if is_up_to_date(&out_json) {
        return;
    }

    let source = format!(
        r#"#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <map>
#include <string>

using bf16 = cutlass::bfloat16_t;
{typedef}
using GemmKernel = TheGemm::GemmKernel;

struct RawParams {{
    int M, N, K;
    int lda, ldb, ldc, ldd;
    float alpha, beta;
}};

GemmKernel::Params make_params(const RawParams& r) {{
    using namespace cutlass;
    using namespace cutlass::gemm;
    GemmCoord problem(r.M, r.N, r.K);
    TheGemm::ThreadblockSwizzle swizzle;
    GemmCoord grid_tiled = swizzle.get_tiled_shape(problem, {{{tm}, {tn}, {tk}}}, 1);
    TensorRef<bfloat16_t, layout::RowMajor>    ref_A(reinterpret_cast<bf16*>(0x1000), layout::RowMajor(r.lda));
    TensorRef<bfloat16_t, layout::ColumnMajor> ref_B(reinterpret_cast<bf16*>(0x2000), layout::ColumnMajor(r.ldb));
    TensorRef<bfloat16_t, layout::RowMajor>    ref_C(reinterpret_cast<bf16*>(0x3000), layout::RowMajor(r.ldc));
    TensorRef<bfloat16_t, layout::RowMajor>    ref_D(reinterpret_cast<bf16*>(0x4000), layout::RowMajor(r.ldd));
    GemmKernel::Epilogue::OutputOp::Params epilogue_op(r.alpha, r.beta);
    return GemmKernel::Params{{
        problem, grid_tiled, ref_A, ref_B, ref_C, ref_D,
        epilogue_op, nullptr, nullptr, nullptr, nullptr
    }};
}}

int main() {{
    const int SZ = sizeof(GemmKernel::Params);
    RawParams base = {{256, 512, 128, 128, 128, 512, 512, 1.0f, 0.0f}};
    std::vector<uint8_t> base_bytes(SZ);
    auto bp = make_params(base);
    memcpy(base_bytes.data(), &bp, SZ);

    struct Pert {{ const char* name; RawParams p; }};
    std::vector<Pert> perts;
    {{ RawParams p=base; p.lda+=1;  perts.push_back({{"lda", p}}); }}
    {{ RawParams p=base; p.ldb+=1;  perts.push_back({{"ldb", p}}); }}
    {{ RawParams p=base; p.ldc+=1;  perts.push_back({{"ldc", p}}); }}
    {{ RawParams p=base; p.ldd+=1;  perts.push_back({{"ldd", p}}); }}
    {{ RawParams p=base; p.M+={tm}; perts.push_back({{"M", p}}); }}
    {{ RawParams p=base; p.N+={tn}; perts.push_back({{"N", p}}); }}
    {{ RawParams p=base; p.K+={tk}; perts.push_back({{"K", p}}); }}

    std::vector<std::vector<uint8_t>> pert_bytes;
    for (auto& pt : perts) {{
        auto pp = make_params(pt.p);
        std::vector<uint8_t> b(SZ);
        memcpy(b.data(), &pp, SZ);
        pert_bytes.push_back(b);
    }}

    RawParams chk = {{1024, 2560, 2048, 2048, 2048, 2560, 2560, 1.0f, 0.0f}};
    auto cp = make_params(chk);
    std::vector<uint8_t> chk_bytes(SZ);
    memcpy(chk_bytes.data(), &cp, SZ);

    RawParams pa=base; pa.alpha=2.0f;
    auto pap = make_params(pa);
    std::vector<uint8_t> alpha_bytes(SZ);
    memcpy(alpha_bytes.data(), &pap, SZ);

    RawParams pb=base; pb.beta=1.0f;
    auto pbp = make_params(pb);
    std::vector<uint8_t> beta_bytes(SZ);
    memcpy(beta_bytes.data(), &pbp, SZ);

    printf("{{\n");
    printf("  \"param_size\": %d,\n", SZ);
    printf("  \"tile\": [{tm}, {tn}, {tk}],\n");
    printf("  \"fields\": [\n");
    bool first = true;
    for (int off = 0; off < SZ; off += 4) {{
        int32_t bv, cv;
        memcpy(&bv, base_bytes.data()+off, 4);
        memcpy(&cv, chk_bytes.data()+off, 4);

        if (!first) printf(",\n");
        first = false;
        printf("    {{\"offset\": %d, \"base_value\": %d, \"check_value\": %d", off, bv, cv);

        float bf, af, btf;
        memcpy(&bf, base_bytes.data()+off, 4);
        memcpy(&af, alpha_bytes.data()+off, 4);
        memcpy(&btf, beta_bytes.data()+off, 4);
        bool is_alpha = !isnan(af) && !isnan(bf) && af != bf;
        bool is_beta = !isnan(btf) && !isnan(bf) && btf != bf;
        if (is_alpha || is_beta) {{
            printf(", \"base_f32\": %.6f", bf);
            if (is_alpha) printf(", \"alpha_f32\": %.6f", af);
            if (is_beta) printf(", \"beta_f32\": %.6f", btf);
        }}

        std::map<std::string,int32_t> deps;
        for (size_t i=0; i<perts.size(); i++) {{
            int32_t pv;
            memcpy(&pv, pert_bytes[i].data()+off, 4);
            if (pv != bv) deps[perts[i].name] = pv - bv;
        }}
        if (!deps.empty()) {{
            printf(", \"depends_on\": {{");
            bool fd = true;
            for (auto& [n,s] : deps) {{
                if (!fd) printf(", ");
                fd = false;
                printf("\"%s\": %d", n.c_str(), s);
            }}
            printf("}}");
        }}
        printf("}}");
    }}
    printf("\n  ]\n}}\n");
    return 0;
}}
"#,
        typedef = cutlass_gemm_typedef(cfg, "TheGemm"),
        tm = tm,
        tn = tn,
        tk = tk,
    );

    let tmp_cu = std::env::temp_dir().join(format!("{}_probe.cu", cfg.name));
    let tmp_bin = std::env::temp_dir().join(format!("{}_probe", cfg.name));

    std::fs::write(&tmp_cu, &source).expect("write probe .cu");

    let compile = Command::new(nvcc)
        .args([
            "-O2",
            "-std=c++17",
            "-arch=sm_89",
            &format!("-I{cutlass_include}"),
            tmp_cu.to_str().unwrap(),
            "-o",
            tmp_bin.to_str().unwrap(),
        ])
        .output();

    match compile {
        Ok(o) if o.status.success() => {}
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            let short: String = stderr.lines().take(5).collect::<Vec<_>>().join("\n");
            println!(
                "cargo:warning=probe compile failed for {}: {short}",
                cfg.name
            );
            let _ = std::fs::remove_file(&tmp_cu);
            return;
        }
        Err(e) => {
            println!("cargo:warning=nvcc not found for probe {}: {e}", cfg.name);
            let _ = std::fs::remove_file(&tmp_cu);
            return;
        }
    }

    let run = Command::new(tmp_bin.to_str().unwrap()).output();

    match run {
        Ok(o) if o.status.success() => {
            std::fs::write(&out_json, &o.stdout).expect("write derivations json");
            println!(
                "cargo:warning=probed {} derivations ({} bytes)",
                cfg.name,
                o.stdout.len()
            );
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            println!("cargo:warning=probe run failed for {}: {stderr}", cfg.name);
        }
        Err(e) => {
            println!("cargo:warning=probe execution failed for {}: {e}", cfg.name);
        }
    }

    let _ = std::fs::remove_file(&tmp_cu);
    let _ = std::fs::remove_file(&tmp_bin);
}
