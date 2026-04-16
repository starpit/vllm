// SPDX-License-Identifier: Apache-2.0
//! CUDA source generation for megakernel compilation units.
//!
//! A megakernel groups multiple DeviceCallable subgraphs from a single
//! wave into one `__global__` cooperative kernel. This module generates
//! the `.cu` source from a list of [`DevicePhase`]s:
//!
//! 1. An internal params struct aggregating per-phase parameters.
//! 2. A `__global__` kernel that executes phases with grid sync.
//! 3. An `extern "C"` launch wrapper taking flat C-friendly args.
//!
//! The Rust codegen calls only the `extern "C"` wrapper with raw
//! pointers and scalars. CUDA compilation is handled by
//! `ferrite-cuda-builder/build.rs`.

#![allow(dead_code)]

use std::fmt::Write;

use crate::impl_lib::DevicePhase;

/// Generated CUDA source + metadata for one megakernel.
#[derive(Clone, Debug)]
pub struct GeneratedMegakernel {
    /// The complete `.cu` source code.
    pub cuda_source: String,
    /// The `extern "C"` launch function name.
    pub launch_fn_name: String,
    /// Per-phase flat parameter descriptors for the Rust FFI caller.
    /// Each entry is `(c_type, param_name)`.
    pub flat_params: Vec<(String, String)>,
    /// Number of inter-phase barrier counters needed (phases - 1).
    pub num_barriers: usize,
}

/// Generate a `.cu` source for one megakernel wave.
///
/// `wave_idx` disambiguates multiple megakernels within a model
/// (e.g. `megakernel_w0`, `megakernel_w1`).
pub fn generate_megakernel(wave_idx: usize, phases: &[DevicePhase]) -> GeneratedMegakernel {
    assert!(
        !phases.is_empty(),
        "megakernel requires at least 1 phase (got {})",
        phases.len()
    );

    let launch_fn_name = format!("megakernel_w{wave_idx}_launch");
    let params_struct_name = format!("MegakernelW{wave_idx}Params");
    let kernel_name = format!("megakernel_w{wave_idx}");

    let mut src = String::new();

    // ── Header ──
    writeln!(src, "// Auto-generated megakernel for wave {wave_idx}").unwrap();
    writeln!(
        src,
        "// DO NOT EDIT — regenerate via the forward! proc macro."
    )
    .unwrap();
    writeln!(src).unwrap();
    writeln!(src, "#include <cuda_bf16.h>").unwrap();
    writeln!(src, "#include <cooperative_groups.h>").unwrap();
    writeln!(src, "#include \"megakernel_ops.cuh\"").unwrap();
    writeln!(src).unwrap();

    // ── Collect all flat params across phases ──
    let mut all_flat: Vec<(String, String)> = Vec::new();
    for phase in phases {
        all_flat.extend(phase.flat_params.iter().cloned());
    }

    // ── Internal params struct ──
    writeln!(src, "struct {params_struct_name} {{").unwrap();
    for (i, phase) in phases.iter().enumerate() {
        writeln!(src, "    // Phase {i}").unwrap();
        for field in &phase.internal_fields {
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

    // Destructure params struct into local variables so kernel_body
    // lines can reference bare names (p0_out, p1_input, etc.).
    for phase in phases {
        for field in &phase.internal_fields {
            // field is e.g. "__nv_bfloat16* p0_out" — extract the name (last token)
            let name = field.split_whitespace().last().unwrap_or("");
            // Strip leading * for pointer fields
            let name = name.trim_start_matches('*');
            writeln!(src, "    auto {name} = p.{name};").unwrap();
        }
    }
    writeln!(src).unwrap();

    for (i, phase) in phases.iter().enumerate() {
        if i > 0 {
            writeln!(src, "    cg::this_grid().sync();").unwrap();
            writeln!(src).unwrap();
        }
        writeln!(src, "    // Phase {i}").unwrap();
        for line in &phase.kernel_body {
            writeln!(src, "    {line}").unwrap();
        }
        writeln!(src).unwrap();
    }

    writeln!(src, "}}").unwrap();
    writeln!(src).unwrap();

    // ── extern "C" launch wrapper ──
    writeln!(src, "extern \"C\" int {launch_fn_name}(").unwrap();
    for (c_type, name) in &all_flat {
        writeln!(src, "    {c_type} {name},").unwrap();
    }
    writeln!(src, "    int __grid_x, int __block_x,").unwrap();
    writeln!(src, "    size_t __smem_bytes,").unwrap();
    writeln!(src, "    uint64_t __stream)").unwrap();
    writeln!(src, "{{").unwrap();
    writeln!(src, "    {params_struct_name} params;").unwrap();

    for phase in phases {
        for line in &phase.params_build {
            writeln!(src, "    {line}").unwrap();
        }
    }

    writeln!(src).unwrap();
    writeln!(src, "    dim3 grid(__grid_x);").unwrap();
    writeln!(src, "    dim3 block(__block_x);").unwrap();
    writeln!(src, "    void* args[] = {{ &params }};").unwrap();
    writeln!(src, "    return cudaLaunchCooperativeKernel(").unwrap();
    writeln!(src, "        (void*){kernel_name},").unwrap();
    writeln!(
        src,
        "        grid, block, args, __smem_bytes, (cudaStream_t)__stream);"
    )
    .unwrap();
    writeln!(src, "}}").unwrap();

    let num_barriers = if phases.len() > 1 {
        phases.len() - 1
    } else {
        0
    };
    GeneratedMegakernel {
        cuda_source: src,
        launch_fn_name,
        flat_params: all_flat,
        num_barriers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_two_phase_megakernel() {
        let p0 = DevicePhase {
            flat_params: vec![
                ("void*".into(), "p0_out".into()),
                ("const void*".into(), "p0_input".into()),
                ("int".into(), "p0_n".into()),
            ],
            kernel_body: vec!["dc_rms_norm<__nv_bfloat16>(p0_out, p0_input, p0_n);".into()],
            params_build: vec![
                "params.p0_out = (__nv_bfloat16*)p0_out;".into(),
                "params.p0_input = (const __nv_bfloat16*)p0_input;".into(),
                "params.p0_n = p0_n;".into(),
            ],
            internal_fields: vec![
                "__nv_bfloat16* p0_out".into(),
                "const __nv_bfloat16* p0_input".into(),
                "int p0_n".into(),
            ],
            preamble: vec![],
        };
        let p1 = DevicePhase {
            flat_params: vec![
                ("void*".into(), "p1_x".into()),
                ("float".into(), "p1_scalar".into()),
                ("int".into(), "p1_n".into()),
            ],
            kernel_body: vec![
                "dc_scalar_mul_inplace<__nv_bfloat16>(p1_x, p1_scalar, p1_n, 1);".into(),
            ],
            params_build: vec![
                "params.p1_x = (__nv_bfloat16*)p1_x;".into(),
                "params.p1_scalar = p1_scalar;".into(),
                "params.p1_n = p1_n;".into(),
            ],
            internal_fields: vec![
                "__nv_bfloat16* p1_x".into(),
                "float p1_scalar".into(),
                "int p1_n".into(),
            ],
            preamble: vec![],
        };

        let result = generate_megakernel(0, &[p0, p1]);
        assert_eq!(result.launch_fn_name, "megakernel_w0_launch");
        assert!(result.cuda_source.contains("megakernel_w0_launch"));
        assert!(result.cuda_source.contains("megakernel_ops.cuh"));
        assert!(result.cuda_source.contains("cg::this_grid().sync()"));
        assert!(result.cuda_source.contains("cudaLaunchCooperativeKernel"));
        // Params struct is destructured into locals for kernel body
        assert!(result.cuda_source.contains("auto p0_out = p.p0_out;"));
        assert!(result.cuda_source.contains("auto p1_scalar = p.p1_scalar;"));
        assert_eq!(result.flat_params.len(), 6); // 3 + 3
    }

    #[test]
    #[should_panic(expected = "at least 1 phase")]
    fn panics_on_zero_phases() {
        generate_megakernel(0, &[]);
    }
}
