// SPDX-License-Identifier: Apache-2.0
//! CUDA source generation for megakernel compilation units.
//!
//! A `DeviceCallable` compilation unit groups multiple ops into a
//! single `__global__` kernel. This module generates the `.cu`
//! source code for each such unit:
//!
//! 1. A params struct aggregating per-phase parameters.
//! 2. A `__global__` kernel that executes phases sequentially with
//!    grid sync between them.
//! 3. An `extern "C"` launch wrapper that takes flat C-friendly
//!    args, builds the internal params struct, and calls
//!    `cudaLaunchCooperativeKernel`.
//!
//! For CUTLASS GEMM/GEMV phases, the launch wrapper constructs
//! CUTLASS kernel params from raw pointers + dims using the
//! `device::Gemm::initialize()` path, then passes the kernel-level
//! params to the megakernel.
//!
//! Generated `.cu` files are written to a deterministic cache
//! location. A companion crate's `build.rs` compiles them.

use std::fmt::Write;

use crate::lowering::assignment::CompilationUnitId;
use crate::lowering::implementation::LaunchKind;

use super::dispatch::{DispatchEntry, DispatchSequence, GemmPhase, ImplDispatchKind};

/// One megakernel compilation unit ready for `.cu` emission.
#[derive(Clone, Debug)]
pub struct MegakernelUnit {
    /// Compilation unit id (used in naming).
    pub unit_id: CompilationUnitId,
    /// Phases in execution order (each phase = one dispatch entry).
    pub phases: Vec<MegakernelPhase>,
}

/// One phase within a megakernel — corresponds to one op.
#[derive(Clone, Debug)]
pub struct MegakernelPhase {
    pub kind: ImplDispatchKind,
    pub gemm_phase: Option<GemmPhase>,
    pub is_attn_norm: Option<bool>,
    pub impl_name: &'static str,
}

/// Generated CUDA source + metadata for one megakernel.
#[derive(Clone, Debug)]
pub struct GeneratedMegakernel {
    /// The compilation unit this kernel implements.
    pub unit_id: CompilationUnitId,
    /// The complete `.cu` source code.
    pub cuda_source: String,
    /// The `extern "C"` launch function name.
    pub launch_fn_name: String,
    /// Per-phase flat parameter descriptors for the Rust FFI caller.
    /// Each entry is `(c_type, param_name)` matching the launch wrapper
    /// signature. The Rust codegen uses this to emit the FFI declaration
    /// and populate the call args.
    pub flat_params: Vec<(String, String)>,
}

/// Extract `DeviceCallable` compilation units from a dispatch
/// sequence and generate CUDA source for each.
pub fn extract_megakernel_units(ds: &DispatchSequence) -> Vec<MegakernelUnit> {
    let mut units = Vec::new();
    for (unit_id, entries) in ds.units() {
        // Only generate megakernels for units with DeviceCallable entries.
        let dc_entries: Vec<&DispatchEntry> = entries
            .into_iter()
            .filter(|e| e.launch_kind == LaunchKind::DeviceCallable)
            .filter(|e| e.kind != ImplDispatchKind::Noop)
            .collect();
        if dc_entries.len() < 2 {
            // Single-op units don't need megakernel treatment.
            continue;
        }
        let phases = dc_entries
            .iter()
            .map(|e| MegakernelPhase {
                kind: e.kind,
                gemm_phase: e.gemm_phase,
                is_attn_norm: e.is_attn_norm,
                impl_name: e.impl_name,
            })
            .collect();
        units.push(MegakernelUnit {
            unit_id,
            phases,
        });
    }
    units
}

/// A flat parameter in the `extern "C"` launch function signature.
struct FlatParam {
    c_type: String,
    name: String,
}

/// Generate CUDA source for a single megakernel unit.
///
/// The generated `.cu` has:
/// 1. An internal params struct (may contain C++ types like CUTLASS
///    kernel params that are NOT exposed across the C ABI).
/// 2. A `__global__` cooperative kernel that reads the internal
///    params and dispatches each phase.
/// 3. An `extern "C"` launch wrapper that takes flat C-compatible
///    args (raw pointers, ints, floats), builds the internal params,
///    and calls `cudaLaunchCooperativeKernel`.
///
/// The Rust side only calls the `extern "C"` wrapper with flat args.
pub fn generate_cuda_source(unit: &MegakernelUnit) -> GeneratedMegakernel {
    let unit_idx = unit.unit_id.0;
    let launch_fn_name = format!("megakernel_unit{unit_idx}_launch");
    let params_struct_name = format!("MegakernelUnit{unit_idx}Params");
    let kernel_name = format!("megakernel_unit{unit_idx}");

    let mut src = String::new();

    // ── Header ──
    writeln!(src, "// Auto-generated megakernel for compilation unit {unit_idx}").unwrap();
    writeln!(src, "// DO NOT EDIT — regenerate via the forward! proc macro.").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "#include <cuda_bf16.h>").unwrap();
    writeln!(src, "#include <cooperative_groups.h>").unwrap();

    // Include CUTLASS if any GEMM phase.
    let has_cutlass_gemm = unit.phases.iter().any(|p| {
        matches!(p.kind, ImplDispatchKind::CutlassGemm { .. })
    });
    if has_cutlass_gemm {
        writeln!(src, "#include <cutlass/cutlass.h>").unwrap();
        writeln!(src, "#include <cutlass/gemm/device/gemm.h>").unwrap();
        writeln!(src, "#include <cutlass/epilogue/thread/linear_combination.h>").unwrap();
    }

    writeln!(src, "#include \"megakernel_ops.cuh\"").unwrap();
    writeln!(src).unwrap();

    // ── CUTLASS type aliases (one per distinct GEMM config) ──
    let mut gemm_aliases: Vec<(u32, u32, u32, String)> = Vec::new(); // (tile_m, tile_n, stages, alias)
    for (i, phase) in unit.phases.iter().enumerate() {
        if let ImplDispatchKind::CutlassGemm { tile_m, tile_n, stages } = phase.kind {
            let alias = format!("GemmKernel_p{i}");
            // Use the same template params as cutlass_standalone_gemm.cu.
            // TB_K is always 32 for sm89 (the solver encodes tile_m x tile_n only).
            let tb_k = 32;
            let (warp_m, warp_n) = warp_shape_for_tb(tile_m, tile_n);
            writeln!(src, "// Phase {i}: CUTLASS GEMM {tile_m}x{tile_n} s{stages}").unwrap();
            writeln!(src, "using DeviceGemm_p{i} = cutlass::gemm::device::Gemm<").unwrap();
            writeln!(src, "    cutlass::bfloat16_t, cutlass::layout::RowMajor,").unwrap();
            writeln!(src, "    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,").unwrap();
            writeln!(src, "    cutlass::bfloat16_t, cutlass::layout::RowMajor,").unwrap();
            writeln!(src, "    float,").unwrap();
            writeln!(src, "    cutlass::arch::OpClassTensorOp,").unwrap();
            writeln!(src, "    cutlass::arch::Sm80,").unwrap();
            writeln!(src, "    cutlass::gemm::GemmShape<{tile_m}, {tile_n}, {tb_k}>,").unwrap();
            writeln!(src, "    cutlass::gemm::GemmShape<{warp_m}, {warp_n}, {tb_k}>,").unwrap();
            writeln!(src, "    cutlass::gemm::GemmShape<16, 8, 16>,").unwrap();
            writeln!(src, "    cutlass::epilogue::thread::LinearCombination<").unwrap();
            writeln!(src, "        cutlass::bfloat16_t, 8, float, float>,").unwrap();
            writeln!(src, "    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,").unwrap();
            writeln!(src, "    {stages}").unwrap();
            writeln!(src, ">;").unwrap();
            writeln!(src, "using {alias} = typename DeviceGemm_p{i}::GemmKernel;").unwrap();
            writeln!(src).unwrap();
            gemm_aliases.push((tile_m, tile_n, stages, alias));
        }
    }

    // ── Collect flat params for the extern "C" launch wrapper ──
    let mut flat_params: Vec<FlatParam> = Vec::new();
    for (i, phase) in unit.phases.iter().enumerate() {
        let mut fp = phase_flat_params(phase, i);
        flat_params.append(&mut fp);
    }

    // ── Internal params struct ──
    // May contain C++ types (CUTLASS kernel Params) that are NOT
    // exposed across the C ABI boundary.
    writeln!(src, "struct {params_struct_name} {{").unwrap();
    for (i, phase) in unit.phases.iter().enumerate() {
        let comment = phase_comment(phase);
        writeln!(src, "    // Phase {i}: {comment}").unwrap();
        for field in phase_internal_fields(phase, i) {
            writeln!(src, "    {field};").unwrap();
        }
    }
    writeln!(src, "}};").unwrap();
    writeln!(src).unwrap();

    // ── __global__ kernel ──
    writeln!(
        src,
        "extern \"C\" __global__ void {kernel_name}({params_struct_name} p) {{"
    )
    .unwrap();
    writeln!(src, "    namespace cg = cooperative_groups;").unwrap();
    writeln!(src, "    extern __shared__ char smem[];").unwrap();
    writeln!(src).unwrap();

    for (i, phase) in unit.phases.iter().enumerate() {
        if i > 0 {
            writeln!(src, "    cg::this_grid().sync();").unwrap();
            writeln!(src).unwrap();
        }
        writeln!(src, "    // Phase {i}: {}", phase_comment(phase)).unwrap();
        for line in phase_kernel_body(phase, i) {
            writeln!(src, "    {line}").unwrap();
        }
        writeln!(src).unwrap();
    }

    writeln!(src, "}}").unwrap();
    writeln!(src).unwrap();

    // ── extern "C" launch wrapper ──
    // Takes flat C args, builds internal params, launches cooperative kernel.
    writeln!(src, "extern \"C\" int {launch_fn_name}(").unwrap();
    for fp in flat_params.iter() {
        let comma = ",";
        writeln!(src, "    {} {}{}", fp.c_type, fp.name, comma).unwrap();
    }
    writeln!(src, "    int __grid_x, int __block_x,").unwrap();
    writeln!(src, "    size_t __smem_bytes,").unwrap();
    writeln!(src, "    uint64_t __stream)").unwrap();
    writeln!(src, "{{").unwrap();
    writeln!(src, "    {params_struct_name} params;").unwrap();

    // Populate internal params from flat args.
    for (i, phase) in unit.phases.iter().enumerate() {
        for line in phase_params_build(phase, i) {
            writeln!(src, "    {line}").unwrap();
        }
    }

    writeln!(src).unwrap();
    writeln!(src, "    dim3 grid(__grid_x);").unwrap();
    writeln!(src, "    dim3 block(__block_x);").unwrap();
    writeln!(src, "    void* args[] = {{ &params }};").unwrap();
    writeln!(src, "    return cudaLaunchCooperativeKernel(").unwrap();
    writeln!(src, "        (void*){kernel_name},").unwrap();
    writeln!(src, "        grid, block, args, __smem_bytes, (cudaStream_t)__stream);").unwrap();
    writeln!(src, "}}").unwrap();

    let all_flat: Vec<(String, String)> = flat_params
        .iter()
        .map(|fp| (fp.c_type.clone(), fp.name.clone()))
        .collect();

    GeneratedMegakernel {
        unit_id: unit.unit_id,
        cuda_source: src,
        launch_fn_name,
        flat_params: all_flat,
    }
}

/// Write generated megakernel `.cu` files to the cache directory.
///
/// The cache is at `~/.cache/cudaforge/megakernels/`. Files are
/// content-addressed by hash of the source, so unchanged sources
/// don't trigger recompilation.
///
/// Returns the paths of all written `.cu` files.
pub fn write_megakernels_to_cache(
    megakernels: &[GeneratedMegakernel],
) -> Vec<std::path::PathBuf> {
    use std::io::Write as IoWrite;

    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let cache_dir = std::path::PathBuf::from(home)
        .join(".cache/cudaforge/megakernels");
    std::fs::create_dir_all(&cache_dir).ok();

    let mut paths = Vec::new();
    for mk in megakernels {
        // Content-addressed filename: megakernel_unit{N}_{hash8}.cu
        let hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            mk.cuda_source.hash(&mut hasher);
            format!("{:016x}", hasher.finish())
        };
        let filename = format!("megakernel_unit{}_{}.cu", mk.unit_id.0, &hash[..8]);
        let path = cache_dir.join(&filename);

        // Only write if content changed (avoid build system churn).
        let needs_write = match std::fs::read_to_string(&path) {
            Ok(existing) => existing != mk.cuda_source,
            Err(_) => true,
        };
        if needs_write {
            let mut f = std::fs::File::create(&path).expect("failed to write megakernel .cu");
            f.write_all(mk.cuda_source.as_bytes())
                .expect("failed to write megakernel .cu");
        }
        paths.push(path);
    }
    paths
}

// ── Per-phase helpers ─────────────────────────────────────────────

/// Human-readable comment for a phase.
fn phase_comment(phase: &MegakernelPhase) -> String {
    match phase.kind {
        ImplDispatchKind::RmsNorm => {
            let pos = if phase.is_attn_norm == Some(true) {
                "attn"
            } else {
                "mlp"
            };
            format!("RmsNorm ({pos})")
        }
        ImplDispatchKind::SiluAndMul => "SiLU+Mul".to_string(),
        ImplDispatchKind::FusedQkvRopeCache => "FusedQkvRopeCache".to_string(),
        ImplDispatchKind::PrefillRopeCache => "PrefillRopeCache".to_string(),
        ImplDispatchKind::CutlassGemm { tile_m, tile_n, stages } => {
            let label = phase
                .gemm_phase
                .map(|p| format!("{p:?}"))
                .unwrap_or_default();
            format!("CUTLASS GEMM {label} {tile_m}x{tile_n} s{stages}")
        }
        ImplDispatchKind::CutlassGemv => {
            let label = phase
                .gemm_phase
                .map(|p| format!("{p:?}"))
                .unwrap_or_default();
            format!("CUTLASS GEMV {label}")
        }
        _ => format!("{:?}", phase.kind),
    }
}

/// Flat C-compatible parameters for one phase in the launch wrapper
/// signature. These are the args the Rust side passes.
fn phase_flat_params(phase: &MegakernelPhase, idx: usize) -> Vec<FlatParam> {
    let p = format!("p{idx}");
    match phase.kind {
        ImplDispatchKind::RmsNorm => vec![
            FlatParam { c_type: "void*".into(), name: format!("{p}_out") },
            FlatParam { c_type: "const void*".into(), name: format!("{p}_input") },
            FlatParam { c_type: "const void*".into(), name: format!("{p}_weight") },
            FlatParam { c_type: "float".into(), name: format!("{p}_eps") },
            FlatParam { c_type: "int".into(), name: format!("{p}_hidden_size") },
            FlatParam { c_type: "int".into(), name: format!("{p}_num_tokens") },
        ],
        ImplDispatchKind::SiluAndMul => vec![
            FlatParam { c_type: "void*".into(), name: format!("{p}_out") },
            FlatParam { c_type: "const void*".into(), name: format!("{p}_input") },
            FlatParam { c_type: "int".into(), name: format!("{p}_d") },
            FlatParam { c_type: "int".into(), name: format!("{p}_num_tokens") },
        ],
        ImplDispatchKind::CutlassGemm { .. } => vec![
            FlatParam { c_type: "void*".into(), name: format!("{p}_C") },
            FlatParam { c_type: "const void*".into(), name: format!("{p}_A") },
            FlatParam { c_type: "const void*".into(), name: format!("{p}_B") },
            FlatParam { c_type: "int".into(), name: format!("{p}_M") },
            FlatParam { c_type: "int".into(), name: format!("{p}_N") },
            FlatParam { c_type: "int".into(), name: format!("{p}_K") },
            FlatParam { c_type: "float".into(), name: format!("{p}_alpha") },
            FlatParam { c_type: "float".into(), name: format!("{p}_beta") },
        ],
        ImplDispatchKind::CutlassGemv => vec![
            FlatParam { c_type: "void*".into(), name: format!("{p}_out") },
            FlatParam { c_type: "const void*".into(), name: format!("{p}_x") },
            FlatParam { c_type: "const void*".into(), name: format!("{p}_W") },
            FlatParam { c_type: "int".into(), name: format!("{p}_N") },
            FlatParam { c_type: "int".into(), name: format!("{p}_K") },
            FlatParam { c_type: "float".into(), name: format!("{p}_alpha") },
            FlatParam { c_type: "float".into(), name: format!("{p}_beta") },
        ],
        ImplDispatchKind::FusedQkvRopeCache => vec![
            FlatParam { c_type: "void*".into(), name: format!("{p}_q_out") },
            FlatParam { c_type: "void*".into(), name: format!("{p}_key_cache") },
            FlatParam { c_type: "void*".into(), name: format!("{p}_value_cache") },
            FlatParam { c_type: "const void*".into(), name: format!("{p}_qkv") },
            FlatParam { c_type: "const void*".into(), name: format!("{p}_positions") },
            FlatParam { c_type: "const void*".into(), name: format!("{p}_cos_sin_cache") },
            FlatParam { c_type: "const void*".into(), name: format!("{p}_slot_mapping") },
            FlatParam { c_type: "int".into(), name: format!("{p}_q_size") },
            FlatParam { c_type: "int".into(), name: format!("{p}_kv_size") },
            FlatParam { c_type: "int".into(), name: format!("{p}_head_dim") },
            FlatParam { c_type: "int".into(), name: format!("{p}_num_tokens") },
        ],
        ImplDispatchKind::PrefillRopeCache => vec![
            FlatParam { c_type: "void*".into(), name: format!("{p}_q_out") },
            FlatParam { c_type: "void*".into(), name: format!("{p}_k_out") },
            FlatParam { c_type: "void*".into(), name: format!("{p}_v_out") },
            FlatParam { c_type: "const void*".into(), name: format!("{p}_qkv") },
            FlatParam { c_type: "const void*".into(), name: format!("{p}_positions") },
            FlatParam { c_type: "const void*".into(), name: format!("{p}_cos_sin_cache") },
            FlatParam { c_type: "int".into(), name: format!("{p}_q_size") },
            FlatParam { c_type: "int".into(), name: format!("{p}_kv_size") },
            FlatParam { c_type: "int".into(), name: format!("{p}_head_dim") },
            FlatParam { c_type: "int".into(), name: format!("{p}_num_tokens") },
        ],
        ImplDispatchKind::TkAttentionDecode | ImplDispatchKind::TkAttentionPrefill => {
            // TK attention is embedded in the megakernel via the
            // FlashInfer persistent runner. Its params are complex
            // (BlockPersistentRunner state). For now, emit a placeholder
            // that will be filled when TK integration lands.
            vec![
                FlatParam { c_type: "void*".into(), name: format!("{p}_runner_state") },
            ]
        }
        _ => vec![],
    }
}

/// Internal params struct fields (may contain C++ types).
fn phase_internal_fields(phase: &MegakernelPhase, idx: usize) -> Vec<String> {
    let p = format!("p{idx}");
    match phase.kind {
        ImplDispatchKind::RmsNorm => vec![
            format!("__nv_bfloat16* {p}_out"),
            format!("const __nv_bfloat16* {p}_input"),
            format!("const __nv_bfloat16* {p}_weight"),
            format!("float {p}_eps"),
            format!("int {p}_hidden_size"),
            format!("int {p}_num_tokens"),
        ],
        ImplDispatchKind::SiluAndMul => vec![
            format!("__nv_bfloat16* {p}_out"),
            format!("const __nv_bfloat16* {p}_input"),
            format!("int {p}_d"),
            format!("int {p}_num_tokens"),
        ],
        ImplDispatchKind::CutlassGemm { .. } => vec![
            // CUTLASS kernel params — built by the launch wrapper
            // from flat C args via device::Gemm::initialize().
            format!("typename GemmKernel_p{idx}::Params {p}_gemm_params"),
        ],
        ImplDispatchKind::CutlassGemv => vec![
            format!("__nv_bfloat16* {p}_out"),
            format!("const __nv_bfloat16* {p}_x"),
            format!("const __nv_bfloat16* {p}_W"),
            format!("int {p}_N"),
            format!("int {p}_K"),
            format!("float {p}_alpha"),
            format!("float {p}_beta"),
        ],
        ImplDispatchKind::FusedQkvRopeCache => vec![
            format!("__nv_bfloat16* {p}_q_out"),
            format!("__nv_bfloat16* {p}_key_cache"),
            format!("__nv_bfloat16* {p}_value_cache"),
            format!("const __nv_bfloat16* {p}_qkv"),
            format!("const uint32_t* {p}_positions"),
            format!("const __nv_bfloat16* {p}_cos_sin_cache"),
            format!("const int64_t* {p}_slot_mapping"),
            format!("int {p}_q_size"),
            format!("int {p}_kv_size"),
            format!("int {p}_head_dim"),
            format!("int {p}_num_tokens"),
        ],
        ImplDispatchKind::PrefillRopeCache => vec![
            format!("__nv_bfloat16* {p}_q_out"),
            format!("__nv_bfloat16* {p}_k_out"),
            format!("__nv_bfloat16* {p}_v_out"),
            format!("const __nv_bfloat16* {p}_qkv"),
            format!("const uint32_t* {p}_positions"),
            format!("const __nv_bfloat16* {p}_cos_sin_cache"),
            format!("int {p}_q_size"),
            format!("int {p}_kv_size"),
            format!("int {p}_head_dim"),
            format!("int {p}_num_tokens"),
        ],
        ImplDispatchKind::TkAttentionDecode | ImplDispatchKind::TkAttentionPrefill => vec![
            format!("void* {p}_runner_state"),
        ],
        _ => vec![],
    }
}

/// Kernel body lines for one phase (device code).
fn phase_kernel_body(phase: &MegakernelPhase, idx: usize) -> Vec<String> {
    let p = format!("p.p{idx}");
    match phase.kind {
        ImplDispatchKind::RmsNorm => vec![
            format!("dc_rms_norm({p}_out, {p}_input, {p}_weight, {p}_eps, {p}_hidden_size, {p}_num_tokens, smem);"),
        ],
        ImplDispatchKind::SiluAndMul => vec![
            format!("dc_silu_and_mul({p}_out, {p}_input, {p}_d, {p}_num_tokens);"),
        ],
        ImplDispatchKind::CutlassGemm { .. } => vec![
            format!("// CUTLASS GEMM kernel-level invocation"),
            format!("GemmKernel_p{idx}()({p}_gemm_params,"),
            format!("    *reinterpret_cast<typename GemmKernel_p{idx}::SharedStorage*>(smem));"),
        ],
        ImplDispatchKind::CutlassGemv => vec![
            format!("dc_gemv({p}_out, {p}_x, {p}_W, {p}_N, {p}_K, {p}_alpha, {p}_beta, {p}_N);"),
        ],
        ImplDispatchKind::FusedQkvRopeCache => vec![
            format!("dc_fused_qkv_rope_cache({p}_q_out, {p}_key_cache, {p}_value_cache,"),
            format!("    {p}_qkv, {p}_positions, {p}_cos_sin_cache, {p}_slot_mapping,"),
            format!("    {p}_q_size, {p}_kv_size, {p}_head_dim, {p}_num_tokens, smem);"),
        ],
        ImplDispatchKind::PrefillRopeCache => vec![
            // PrefillRopeCache is split_qkv + rotary + kv_cache_write.
            // For now, emit a TODO — the device-callable prefill rope
            // has different semantics (explicit K/V output, no paged cache).
            // H100 only; L40S decode plans don't hit this path.
            format!("// PrefillRopeCache: not yet device-callable (H100 sm90 only)"),
            format!("// The prefill path uses split_qkv + rotary_inplace + write_kv_cache"),
            format!("// as separate host launches. Megakernel fusion requires a combined"),
            format!("// dc_prefill_rope_cache() op in megakernel_ops.cuh."),
        ],
        ImplDispatchKind::TkAttentionDecode | ImplDispatchKind::TkAttentionPrefill => vec![
            format!("// TK attention: FlashInfer BlockPersistentRunner"),
            format!("// Integration pending — requires embedding the persistent"),
            format!("// runner's device-side dispatch in the megakernel."),
        ],
        _ => vec![],
    }
}

/// Launch wrapper lines to build internal params from flat args.
fn phase_params_build(phase: &MegakernelPhase, idx: usize) -> Vec<String> {
    let p = format!("p{idx}");
    match phase.kind {
        ImplDispatchKind::RmsNorm => vec![
            format!("params.{p}_out = (__nv_bfloat16*){p}_out;"),
            format!("params.{p}_input = (const __nv_bfloat16*){p}_input;"),
            format!("params.{p}_weight = (const __nv_bfloat16*){p}_weight;"),
            format!("params.{p}_eps = {p}_eps;"),
            format!("params.{p}_hidden_size = {p}_hidden_size;"),
            format!("params.{p}_num_tokens = {p}_num_tokens;"),
        ],
        ImplDispatchKind::SiluAndMul => vec![
            format!("params.{p}_out = (__nv_bfloat16*){p}_out;"),
            format!("params.{p}_input = (const __nv_bfloat16*){p}_input;"),
            format!("params.{p}_d = {p}_d;"),
            format!("params.{p}_num_tokens = {p}_num_tokens;"),
        ],
        ImplDispatchKind::CutlassGemm { .. } => vec![
            format!("// Build CUTLASS kernel params from flat args via device::Gemm"),
            format!("{{"),
            format!("    typename DeviceGemm_p{idx}::Arguments args("),
            format!("        {{{p}_M, {p}_N, {p}_K}},"),
            format!("        {{(cutlass::bfloat16_t const*){p}_A, {p}_K}},"),
            format!("        {{(cutlass::bfloat16_t const*){p}_B, {p}_K}},"),
            format!("        {{(cutlass::bfloat16_t*){p}_C, {p}_N}},"),
            format!("        {{(cutlass::bfloat16_t*){p}_C, {p}_N}},"),
            format!("        {{{p}_alpha, {p}_beta}}"),
            format!("    );"),
            format!("    DeviceGemm_p{idx} gemm_op;"),
            format!("    gemm_op.initialize(args, nullptr);"),
            // Extract kernel params. GemmUniversalBase stores params_
            // as the sole data member, so we can safely reinterpret.
            format!("    static_assert("),
            format!("        sizeof(DeviceGemm_p{idx}) >= sizeof(typename GemmKernel_p{idx}::Params),"),
            format!("        \"DeviceGemm layout assumption violated\");"),
            format!("    params.{p}_gemm_params = *reinterpret_cast<"),
            format!("        typename GemmKernel_p{idx}::Params const*>(&gemm_op);"),
            format!("}}"),
        ],
        ImplDispatchKind::CutlassGemv => vec![
            format!("params.{p}_out = (__nv_bfloat16*){p}_out;"),
            format!("params.{p}_x = (const __nv_bfloat16*){p}_x;"),
            format!("params.{p}_W = (const __nv_bfloat16*){p}_W;"),
            format!("params.{p}_N = {p}_N;"),
            format!("params.{p}_K = {p}_K;"),
            format!("params.{p}_alpha = {p}_alpha;"),
            format!("params.{p}_beta = {p}_beta;"),
        ],
        ImplDispatchKind::FusedQkvRopeCache => vec![
            format!("params.{p}_q_out = (__nv_bfloat16*){p}_q_out;"),
            format!("params.{p}_key_cache = (__nv_bfloat16*){p}_key_cache;"),
            format!("params.{p}_value_cache = (__nv_bfloat16*){p}_value_cache;"),
            format!("params.{p}_qkv = (const __nv_bfloat16*){p}_qkv;"),
            format!("params.{p}_positions = (const uint32_t*){p}_positions;"),
            format!("params.{p}_cos_sin_cache = (const __nv_bfloat16*){p}_cos_sin_cache;"),
            format!("params.{p}_slot_mapping = (const int64_t*){p}_slot_mapping;"),
            format!("params.{p}_q_size = {p}_q_size;"),
            format!("params.{p}_kv_size = {p}_kv_size;"),
            format!("params.{p}_head_dim = {p}_head_dim;"),
            format!("params.{p}_num_tokens = {p}_num_tokens;"),
        ],
        ImplDispatchKind::PrefillRopeCache => vec![
            format!("params.{p}_q_out = (__nv_bfloat16*){p}_q_out;"),
            format!("params.{p}_k_out = (__nv_bfloat16*){p}_k_out;"),
            format!("params.{p}_v_out = (__nv_bfloat16*){p}_v_out;"),
            format!("params.{p}_qkv = (const __nv_bfloat16*){p}_qkv;"),
            format!("params.{p}_positions = (const uint32_t*){p}_positions;"),
            format!("params.{p}_cos_sin_cache = (const __nv_bfloat16*){p}_cos_sin_cache;"),
            format!("params.{p}_q_size = {p}_q_size;"),
            format!("params.{p}_kv_size = {p}_kv_size;"),
            format!("params.{p}_head_dim = {p}_head_dim;"),
            format!("params.{p}_num_tokens = {p}_num_tokens;"),
        ],
        ImplDispatchKind::TkAttentionDecode | ImplDispatchKind::TkAttentionPrefill => vec![
            format!("params.{p}_runner_state = {p}_runner_state;"),
        ],
        _ => vec![],
    }
}

/// Map threadblock shape to warp shape (matching cutlass_standalone_gemm.cu).
fn warp_shape_for_tb(tile_m: u32, tile_n: u32) -> (u32, u32) {
    match (tile_m, tile_n) {
        (64, 64) => (32, 32),
        (64, 128) => (32, 64),
        (128, 64) => (64, 32),
        (128, 128) => (64, 32),
        (128, 256) => (64, 64),
        (256, 64) => (64, 32),
        (256, 128) => (64, 32),
        // Default: split evenly.
        _ => (tile_m / 2, tile_n / 2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_and_generate_smoke() {
        let unit = MegakernelUnit {
            unit_id: CompilationUnitId(0),
            phases: vec![
                MegakernelPhase {
                    kind: ImplDispatchKind::RmsNorm,
                    gemm_phase: None,
                    is_attn_norm: Some(true),
                    impl_name: "dc_vllm_rs_rms_norm",
                },
                MegakernelPhase {
                    kind: ImplDispatchKind::CutlassGemv,
                    gemm_phase: Some(GemmPhase::Q),
                    is_attn_norm: None,
                    impl_name: "dc_cutlass_gemv_q",
                },
                MegakernelPhase {
                    kind: ImplDispatchKind::SiluAndMul,
                    gemm_phase: None,
                    is_attn_norm: None,
                    impl_name: "dc_vllm_rs_silu_and_mul_fused",
                },
            ],
        };

        let result = generate_cuda_source(&unit);
        assert!(result.cuda_source.contains("megakernel_unit0"));
        assert!(result.cuda_source.contains("MegakernelUnit0Params"));
        assert!(result.cuda_source.contains("cg::this_grid().sync()"));
        assert!(result.cuda_source.contains("dc_rms_norm"));
        assert!(result.cuda_source.contains("dc_gemv"));
        assert!(result.cuda_source.contains("dc_silu_and_mul"));
        assert!(result.cuda_source.contains("cudaLaunchCooperativeKernel"));
        assert_eq!(result.launch_fn_name, "megakernel_unit0_launch");

        // Flat params should include all phases' args.
        assert!(!result.flat_params.is_empty());
        let param_names: Vec<&str> = result.flat_params.iter().map(|p| p.1.as_str()).collect();
        assert!(param_names.contains(&"p0_out"));     // RmsNorm
        assert!(param_names.contains(&"p1_out"));     // GEMV
        assert!(param_names.contains(&"p2_out"));     // SiLU
    }

    #[test]
    fn test_cutlass_gemm_phase() {
        let unit = MegakernelUnit {
            unit_id: CompilationUnitId(1),
            phases: vec![
                MegakernelPhase {
                    kind: ImplDispatchKind::RmsNorm,
                    gemm_phase: None,
                    is_attn_norm: Some(true),
                    impl_name: "dc_vllm_rs_rms_norm",
                },
                MegakernelPhase {
                    kind: ImplDispatchKind::CutlassGemm {
                        tile_m: 128,
                        tile_n: 128,
                        stages: 4,
                    },
                    gemm_phase: Some(GemmPhase::Q),
                    is_attn_norm: None,
                    impl_name: "dc_cutlass_q_128x128_s4",
                },
            ],
        };

        let result = generate_cuda_source(&unit);
        // Should have CUTLASS type alias.
        assert!(result.cuda_source.contains("DeviceGemm_p1"));
        assert!(result.cuda_source.contains("GemmKernel_p1"));
        // Should have CUTLASS kernel invocation in kernel body.
        assert!(result.cuda_source.contains("GemmKernel_p1()(p.p1_gemm_params"));
        // Should have params build from flat args.
        assert!(result.cuda_source.contains("gemm_op.initialize(args"));
        // Flat params should have GEMM args.
        let param_names: Vec<&str> = result.flat_params.iter().map(|p| p.1.as_str()).collect();
        assert!(param_names.contains(&"p1_C"));
        assert!(param_names.contains(&"p1_A"));
        assert!(param_names.contains(&"p1_M"));
    }

    #[test]
    fn test_fused_qkv_rope_cache_phase() {
        let unit = MegakernelUnit {
            unit_id: CompilationUnitId(2),
            phases: vec![
                MegakernelPhase {
                    kind: ImplDispatchKind::CutlassGemv,
                    gemm_phase: Some(GemmPhase::Q),
                    is_attn_norm: None,
                    impl_name: "dc_cutlass_gemv_q",
                },
                MegakernelPhase {
                    kind: ImplDispatchKind::FusedQkvRopeCache,
                    gemm_phase: None,
                    is_attn_norm: None,
                    impl_name: "dc_vllm_rs_fused_qkv_rope_cache",
                },
            ],
        };

        let result = generate_cuda_source(&unit);
        assert!(result.cuda_source.contains("dc_fused_qkv_rope_cache"));
        assert!(result.cuda_source.contains("p1_q_out"));
        assert!(result.cuda_source.contains("p1_key_cache"));
        assert!(result.cuda_source.contains("p1_slot_mapping"));
        // Verify the launch wrapper builds params correctly.
        assert!(result.cuda_source.contains("params.p1_q_out = (__nv_bfloat16*)p1_q_out"));
    }
}
