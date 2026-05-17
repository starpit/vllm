// SPDX-License-Identifier: Apache-2.0
//! End-to-end test of `synthesize_persistent_chunk`: synthesize the
//! kernel, compile it via `newLibraryWithSource`, dispatch with real
//! buffers, verify output is correct.
//!
//! Three trivial phases:
//!   phase 0: each TG writes its tg_id to `out_a[tg_id]`
//!   phase 1: each TG reads `out_a[(tg_id + 1) % num_tgs]` and
//!            writes to `out_b[tg_id]` — exercises cross-TG read
//!            from phase 0's writes, which only works if the
//!            cross-TG barrier propagates phase 0's writes visibly.
//!   phase 2: each TG sums out_a[tg_id] + out_b[tg_id] into out_c[tg_id]
//!
//! Expected output:
//!   out_a[i] = i
//!   out_b[i] = (i + 1) % num_tgs
//!   out_c[i] = out_a[i] + out_b[i]
//!
//! If the cross-TG barrier between phase 0 and phase 1 doesn't
//! actually wait for all TGs, phase 1's read sees stale `out_a`
//! values and `out_b` is wrong. So this test is a correctness check
//! on the barrier as well as a compile check.

use crate::util::{self, Buffer, CommandQueue};
use ferrite_fusion_synth::atom::{AtomConstantValue, AtomCtx};
use ferrite_fusion_synth::atom_lib::AddRmsNormAtom;
use ferrite_fusion_synth::fuse_pass::{
    pre_attn_kernel_scope_prologue, synthesize_persistent_chunk,
    synthesize_persistent_chunk_with_preamble, PersistentPhase, SynthesisBackend,
};
use objc2::AnyThread;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};
use std::ptr::NonNull;

pub fn run(_launch_overhead_us: f64) {
    eprintln!("\n=== synthesize_persistent_chunk end-to-end test ===");

    let num_tgs: u32 = 16;
    let threads_per_tg: u32 = 64;
    // ── Subtest 1: 3-phase POC (cross-TG read across atomic barrier) ──
    eprintln!("  --- Subtest 1: 3-phase cross-TG read ---");

    let phases = vec![
        PersistentPhase {
            name: "write_tg_id".to_string(),
            body: r#"
    if (tid == 0u) {
        // Cross-TG visibility on Apple Metal requires atomic memory
        // ops, not regular stores + threadgroup_barrier(mem_device).
        // Bitcast float→uint and use atomic_store; the next phase
        // does the inverse on read.
        uint v = as_type<uint>(float(tg_id));
        atomic_store_explicit(
            (device atomic_uint*)&out_a[tg_id], v, memory_order_relaxed);
    }
"#
            .to_string(),
        },
        PersistentPhase {
            name: "cross_tg_read".to_string(),
            body: r#"
    if (tid == 0u) {
        uint nbr = (tg_id + 1u) % num_tgs;
        uint v = atomic_load_explicit(
            (device atomic_uint*)&out_a[nbr], memory_order_relaxed);
        out_b[tg_id] = as_type<float>(v);
    }
"#
            .to_string(),
        },
        PersistentPhase {
            name: "sum".to_string(),
            body: r#"
    if (tid == 0u) {
        out_c[tg_id] = out_a[tg_id] + out_b[tg_id];
    }
"#
            .to_string(),
        },
    ];

    let extra_sig = r#",
    device float* out_a [[buffer(2)]],
    device float* out_b [[buffer(3)]],
    device float* out_c [[buffer(4)]]"#;

    let kernel = synthesize_persistent_chunk(
        SynthesisBackend::Metal,
        "synth_persistent_e2e_test",
        threads_per_tg,
        extra_sig,
        "",
        &phases,
    );
    eprintln!(
        "  synthesized {} bytes of MSL with {} phases",
        kernel.source.len(),
        phases.len()
    );
    if std::env::var_os("FERRITE_DEBUG_PRINT_MSL").is_some() {
        eprintln!("  --- emitted MSL ---");
        for (i, line) in kernel.source.lines().enumerate() {
            eprintln!("  {:>3}: {}", i + 1, line);
        }
        eprintln!("  --- end MSL ---");
    }

    // Compile
    let device = util::device();
    let source = NSString::from_str(&kernel.source);
    let options = objc2_metal::MTLCompileOptions::new();
    let library = match device.newLibraryWithSource_options_error(&source, Some(&options)) {
        Ok(lib) => lib,
        Err(e) => {
            eprintln!("  ✗ library compilation FAILED");
            eprintln!("  {:?}", e);
            eprintln!("  --- synthesized source: ---");
            for (i, line) in kernel.source.lines().enumerate() {
                eprintln!("  {:>3}: {}", i + 1, line);
            }
            panic!("synthesize_persistent_chunk emitted invalid MSL");
        }
    };
    eprintln!("  ✓ MSL compiled");

    let function = library
        .newFunctionWithName(&NSString::from_str(&kernel.symbol))
        .expect("symbol lookup");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("pipeline creation");
    eprintln!("  ✓ pipeline created (maxThreads/TG = {})", pipeline.maxTotalThreadsPerThreadgroup());

    // Buffers
    let queue = util::new_command_queue();
    let counter = util::create_buffer(4);
    let out_a = util::create_buffer((num_tgs as usize) * 4);
    let out_b = util::create_buffer((num_tgs as usize) * 4);
    let out_c = util::create_buffer((num_tgs as usize) * 4);

    // Zero everything
    for b in [&counter, &out_a, &out_b, &out_c] {
        let ptr = b.contents().as_ptr() as *mut u32;
        let n = b.length() / 4;
        for i in 0..n {
            unsafe { ptr.add(i).write(0); }
        }
    }

    // Dispatch
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&counter), 0, 0);
        enc.setBytes_length_atIndex(
            NonNull::new_unchecked(&num_tgs as *const u32 as *mut std::ffi::c_void),
            4, 1,
        );
        enc.setBuffer_offset_atIndex(Some(&out_a), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&out_b), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&out_c), 0, 4);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: num_tgs as usize, height: 1, depth: 1 },
        MTLSize { width: threads_per_tg as usize, height: 1, depth: 1 },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    eprintln!("  ✓ dispatched and completed");

    // Verify
    let read = |buf: &Buffer| -> Vec<f32> {
        let ptr = buf.contents().as_ptr() as *const f32;
        (0..num_tgs as usize).map(|i| unsafe { *ptr.add(i) }).collect()
    };
    let a = read(&out_a);
    let b = read(&out_b);
    let c = read(&out_c);

    let mut all_ok = true;
    for i in 0..num_tgs as usize {
        let exp_a = i as f32;
        let exp_b = ((i + 1) % num_tgs as usize) as f32;
        let exp_c = exp_a + exp_b;
        if a[i] != exp_a || b[i] != exp_b || c[i] != exp_c {
            eprintln!(
                "  ✗ tg_id={} a={} (exp {}) b={} (exp {}) c={} (exp {})",
                i, a[i], exp_a, b[i], exp_b, c[i], exp_c
            );
            all_ok = false;
        }
    }
    if all_ok {
        eprintln!("  ✓ all {} TGs produced correct output across 3 phases", num_tgs);
        eprintln!("  ✓ cross-TG barrier propagated phase-0 writes visibly to phase 1");
    } else {
        panic!("synth_persistent_test: cross-TG visibility FAILED");
    }

    // ── Subtest 2: realistic HIDDEN-shaped multi-phase persistent kernel ──
    eprintln!("  --- Subtest 2: 8-phase HIDDEN-shaped chain ---");
    run_realistic_subtest(device, &queue, num_tgs, threads_per_tg);

    // ── Subtest 3: real atom (AddRmsNormAtom) composes + compiles ──
    eprintln!("  --- Subtest 3: AddRmsNormAtom in persistent kernel ---");
    run_atom_compose_subtest(device);

    // ── Subtest 4: AddRmsNormAtom dispatched + verified vs CPU ref ──
    eprintln!("  --- Subtest 4: AddRmsNormAtom dispatch + correctness ---");
    run_atom_dispatch_subtest(device, &queue);

    // ── Subtest 5: production-shape persistent pre-attn compiles ──
    eprintln!("  --- Subtest 5: synthesize_pre_attn_chunk_persistent compiles ---");
    run_persistent_pre_attn_compile_subtest(device);
}

/// Validates that `synthesize_pre_attn_chunk_persistent` produces MSL
/// that compiles on the runtime path (`newLibraryWithSource`) for the
/// real production model shapes (Llama-3.2-1B and -3B 4bit). Catches
/// signature/index/scoping bugs that the host-side `cargo test`
/// structural assertions can't see.
fn run_persistent_pre_attn_compile_subtest(device: &crate::util::Device) {
    use ferrite_fusion_synth::fuse_pass::{
        synthesize_pre_attn_chunk_persistent, synthesize_pre_attn_init_chunk_persistent,
        ChunkConstants,
    };

    let shapes = [
        ("Llama-3.2-1B-Instruct-4bit", ChunkConstants {
            hidden:        2048,
            num_q_heads:   32,
            num_kv_heads:  8,
            head_dim:      64,
            rot_dim:       64,
            block_size:    16,
            intermediate:  8192,
            m:             1,
            group_size:    64,
            rms_norm_eps:  1e-5,
            has_linear_bias: false,
        }),
        ("Llama-3.2-3B-Instruct-4bit", ChunkConstants {
            hidden:        3072,
            num_q_heads:   24,
            num_kv_heads:  8,
            head_dim:      128,
            rot_dim:       128,
            block_size:    16,
            intermediate:  8192,
            m:             1,
            group_size:    64,
            rms_norm_eps:  1e-5,
            has_linear_bias: false,
        }),
    ];

    for (label, consts) in &shapes {
        for (variant_name, build) in [
            ("non-init", synthesize_pre_attn_chunk_persistent as fn(_, _, _, _) -> _),
            ("init",     synthesize_pre_attn_init_chunk_persistent),
        ] {
            let kernel = build(
                ferrite_fusion_synth::fuse_pass::SynthesisBackend::Metal,
                "bfloat",
                "half",
                consts,
            );
            let source = NSString::from_str(&kernel.source);
            let options = objc2_metal::MTLCompileOptions::new();
            match device.newLibraryWithSource_options_error(&source, Some(&options)) {
                Ok(library) => {
                    let function = library.newFunctionWithName(
                        &NSString::from_str(&kernel.symbol)
                    );
                    if function.is_none() {
                        panic!("{label}/{variant_name}: function `{}` not found in library",
                            kernel.symbol);
                    }
                    eprintln!(
                        "  ✓ {label}/{variant_name}: {} bytes, symbol={}",
                        kernel.source.len(),
                        kernel.symbol,
                    );
                }
                Err(e) => {
                    eprintln!("  ✗ {label}/{variant_name}: MSL compilation FAILED");
                    eprintln!("  {:?}", e);
                    if std::env::var_os("FERRITE_DEBUG_PRINT_MSL").is_some() {
                        for (i, line) in kernel.source.lines().enumerate() {
                            eprintln!("  {:>4}: {}", i + 1, line);
                        }
                    } else {
                        eprintln!("  (re-run with FERRITE_DEBUG_PRINT_MSL=1 to dump source)");
                    }
                    panic!("persistent pre-attn compile subtest: {label}/{variant_name} failed");
                }
            }
        }
    }
}

/// Compile-only test: AddRmsNormAtom emits its body via Atom trait,
/// PersistentPhase::from_atom_metal wraps it, persistent-chunk
/// synthesis bakes the model-shape constants as constexpr, and the
/// resulting MSL must compile cleanly under newLibraryWithSource.
/// Dispatch + correctness verification deferred to a follow-up that
/// stages real model weights.
fn run_atom_compose_subtest(device: &crate::util::Device) {
    let atom = AddRmsNormAtom::default();
    let in_names = vec![
        "__residual_io".to_string(),
        "__delta".to_string(),
        "__rms_weight".to_string(),
    ];
    let out_names = vec!["__x_norm".to_string()];
    let consts: Vec<(&'static str, AtomConstantValue)> = vec![
        ("HIDDEN", AtomConstantValue::Uint(2048)),
        ("NUM_Q", AtomConstantValue::Uint(32)),
        ("NUM_KV", AtomConstantValue::Uint(8)),
        ("HEAD_DIM", AtomConstantValue::Uint(64)),
        ("EPS", AtomConstantValue::Float(1e-5)),
    ];
    let ctx = AtomCtx {
        bound_inputs: &in_names,
        bound_outputs: &out_names,
        constants: &consts,
        t_act: "bfloat",
        t_scale: "half",
    };
    let phase = PersistentPhase::from_atom_metal("add_rms_norm", &atom, &ctx)
        .expect("AddRmsNormAtom should emit Metal body");

    // Inline metal_kittens.h verbatim — newLibraryWithSource doesn't
    // resolve relative includes. Strip `#pragma once` to suppress the
    // "in main file" warning (mirrors `inline_header` in fuse_pass).
    let mk_header_raw =
        include_str!("../../ferrite-metal-kernels/shaders/metal_kittens.h");
    let mk_header: String = mk_header_raw
        .lines()
        .filter(|l| l.trim() != "#pragma once")
        .collect::<Vec<_>>()
        .join("\n");

    let preamble = format!(
        r#"
// metal_kittens.h provides MK_SIMD_SIZE, MK_ROWS_PER_SIMDGROUP,
// mk_tg_rmsnorm_scale, etc. that AddRmsNormAtom's body references.
// === inlined metal_kittens.h ===
{mk}
// === end inlined metal_kittens.h ===

// Model-invariant shape constants baked at synth time, matching what
// the existing synthesize_pre_attn_chunk does at file scope.
constant constexpr uint  HIDDEN     = 2048u;
constant constexpr uint  NUM_Q      = 32u;
constant constexpr uint  NUM_KV     = 8u;
constant constexpr uint  HEAD_DIM   = 64u;
constant constexpr uint  ROT_DIM    = 64u;
constant constexpr uint  BLOCK_SIZE = 16u;
constant constexpr uint  M          = 1u;
constant constexpr float EPS        = 1e-5f;
"#,
        mk = mk_header
    );

    let prologue = pre_attn_kernel_scope_prologue(
        "bfloat",
        "HIDDEN", "HEAD_DIM", "NUM_Q", "NUM_KV", "ROT_DIM",
        "BLOCK_SIZE", "EPS", "M",
        "__x_norm", "__qmv_smem",
        2048, 256,
    );

    let extra_sig = ",\n    device bfloat* __residual_io [[buffer(2)]],\n    device const bfloat* __delta [[buffer(3)]],\n    device const half* __rms_weight [[buffer(4)]]";

    let kernel = synthesize_persistent_chunk_with_preamble(
        SynthesisBackend::Metal,
        "synth_persistent_atom_compose",
        128,
        extra_sig,
        &preamble,
        &prologue,
        std::slice::from_ref(&phase),
    );

    eprintln!(
        "  synthesized {} bytes of MSL (1 atom phase + scaffolding)",
        kernel.source.len()
    );
    if std::env::var_os("FERRITE_DEBUG_PRINT_MSL").is_some() {
        for (i, line) in kernel.source.lines().enumerate() {
            eprintln!("  {:>3}: {}", i + 1, line);
        }
    }

    let source = NSString::from_str(&kernel.source);
    let options = objc2_metal::MTLCompileOptions::new();
    match device.newLibraryWithSource_options_error(&source, Some(&options)) {
        Ok(_lib) => {
            eprintln!("  ✓ AddRmsNormAtom body compiled inside persistent-chunk envelope");
            eprintln!("  ✓ atom composition path is wired end-to-end at the MSL level");
            eprintln!("    (subtest 4 below dispatches + verifies vs CPU reference)");
        }
        Err(e) => {
            eprintln!("  ✗ MSL compilation FAILED — atom composition needs adjustment");
            eprintln!("  {:?}", e);
            if std::env::var_os("FERRITE_DEBUG_PRINT_MSL").is_none() {
                eprintln!("  (re-run with FERRITE_DEBUG_PRINT_MSL=1 to see the source)");
            }
            panic!("atom-compose subtest: synthesized MSL failed to compile");
        }
    }
}

/// End-to-end dispatch of the atom-composed persistent kernel.
/// AddRmsNormAtom (one phase) + copy-TG-mem-to-device (second phase,
/// debug-only). Verifies:
///   - residual_io[i] unchanged after kernel (delta is all zero, so
///     `residual + delta == residual` holds bit-exact in bf16 and
///     dodges the Q-writeback / K-V-read race that exists in the
///     production fused kernel)
///   - out_xnorm[i] matches a CPU RMSNorm reference within bf16-band
///     tolerance
///
/// Grid: (M=1, NUM_HEADS_TOTAL=NUM_Q+2*NUM_KV, 1) × threads/TG
/// = MK_SIMD_SIZE * HEAD_DIM / MK_ROWS_PER_SIMDGROUP = 32*64/4 = 512.
fn run_atom_dispatch_subtest(device: &crate::util::Device, queue: &CommandQueue) {
    // Llama-3.2-1B-shape constants (HIDDEN = NUM_Q * HEAD_DIM so the
    // Q-head residual writeback covers all of HIDDEN; that's required
    // for the post-condition `residual_io == residual + delta`).
    const HIDDEN: usize    = 2048;
    const NUM_Q: usize     = 32;
    const NUM_KV: usize    = 8;
    const HEAD_DIM: usize  = 64;
    const NUM_HEADS_TOTAL: usize = NUM_Q + 2 * NUM_KV;
    const M: usize         = 1;
    const EPS: f32         = 1e-5;
    const MK_SIMD_SIZE: usize = 32;
    const MK_ROWS_PER_SIMDGROUP: usize = 4;
    const THREADS_PER_TG: u32 = (MK_SIMD_SIZE * HEAD_DIM / MK_ROWS_PER_SIMDGROUP) as u32; // 512
    assert_eq!(NUM_Q * HEAD_DIM, HIDDEN, "Q-head writeback must cover all of HIDDEN");

    // ── Synthesize persistent kernel: phase 0 = AddRmsNormAtom,
    //                                  phase 1 = copy __x_norm → device.
    let atom = AddRmsNormAtom::default();
    let in_names = vec![
        "__residual_io".to_string(),
        "__delta".to_string(),
        "__rms_weight".to_string(),
    ];
    let out_names = vec!["__x_norm".to_string()];
    let consts: Vec<(&'static str, AtomConstantValue)> = vec![
        ("HIDDEN", AtomConstantValue::Uint(HIDDEN as u32)),
        ("NUM_Q", AtomConstantValue::Uint(NUM_Q as u32)),
        ("NUM_KV", AtomConstantValue::Uint(NUM_KV as u32)),
        ("HEAD_DIM", AtomConstantValue::Uint(HEAD_DIM as u32)),
        ("EPS", AtomConstantValue::Float(EPS)),
    ];
    let ctx = AtomCtx {
        bound_inputs: &in_names,
        bound_outputs: &out_names,
        constants: &consts,
        t_act: "bfloat",
        t_scale: "half",
    };
    let atom_phase = PersistentPhase::from_atom_metal("add_rms_norm", &atom, &ctx)
        .expect("AddRmsNormAtom should emit Metal body");

    // Debug-copy phase: TG (__t=0, __head=0) copies its own __x_norm
    // [0..HIDDEN] to the device-side output buffer. All TGs compute
    // the same __x_norm (it's a per-token activation, not per-head),
    // so picking any one TG is fine. Filtering by both __t and __head
    // singles out exactly one TG out of M * NUM_HEADS_TOTAL.
    let copy_phase = PersistentPhase {
        name: "copy_xnorm_to_device".to_string(),
        body: r#"
    if (__t == 0u && __head == 0u) {
        for (uint __i = __tid; __i < __hidden; __i += __threads_per_tg) {
            __out_xnorm[__i] = __x_norm[__i];
        }
    }
"#
        .to_string(),
    };

    // Inline metal_kittens.h (same trick as compose subtest above).
    let mk_header_raw =
        include_str!("../../ferrite-metal-kernels/shaders/metal_kittens.h");
    let mk_header: String = mk_header_raw
        .lines()
        .filter(|l| l.trim() != "#pragma once")
        .collect::<Vec<_>>()
        .join("\n");

    let preamble = format!(
        r#"
// === inlined metal_kittens.h ===
{mk}
// === end inlined metal_kittens.h ===

constant constexpr uint  HIDDEN     = {hidden}u;
constant constexpr uint  NUM_Q      = {num_q}u;
constant constexpr uint  NUM_KV     = {num_kv}u;
constant constexpr uint  HEAD_DIM   = {head_dim}u;
constant constexpr uint  ROT_DIM    = {head_dim}u;
constant constexpr uint  BLOCK_SIZE = 16u;
constant constexpr uint  M          = {m}u;
constant constexpr float EPS        = {eps:e}f;
"#,
        mk = mk_header,
        hidden = HIDDEN, num_q = NUM_Q, num_kv = NUM_KV,
        head_dim = HEAD_DIM, m = M, eps = EPS,
    );

    let prologue = pre_attn_kernel_scope_prologue(
        "bfloat",
        "HIDDEN", "HEAD_DIM", "NUM_Q", "NUM_KV", "ROT_DIM",
        "BLOCK_SIZE", "EPS", "M",
        "__x_norm", "__qmv_smem",
        HIDDEN as u32, HEAD_DIM as u32,
    );

    let extra_sig = ",\n    device bfloat* __residual_io [[buffer(2)]],\n    device const bfloat* __delta [[buffer(3)]],\n    device const half* __rms_weight [[buffer(4)]],\n    device bfloat* __out_xnorm [[buffer(5)]]";

    let phases = vec![atom_phase, copy_phase];
    let kernel = synthesize_persistent_chunk_with_preamble(
        SynthesisBackend::Metal,
        "synth_persistent_atom_dispatch",
        THREADS_PER_TG,
        extra_sig,
        &preamble,
        &prologue,
        &phases,
    );

    eprintln!(
        "  synthesized {} bytes of MSL ({} phases: atom + copy)",
        kernel.source.len(),
        phases.len(),
    );
    if std::env::var_os("FERRITE_DEBUG_PRINT_MSL").is_some() {
        for (i, line) in kernel.source.lines().enumerate() {
            eprintln!("  {:>3}: {}", i + 1, line);
        }
    }

    // ── Compile + PSO ──
    let source = NSString::from_str(&kernel.source);
    let options = objc2_metal::MTLCompileOptions::new();
    let library = match device.newLibraryWithSource_options_error(&source, Some(&options)) {
        Ok(lib) => lib,
        Err(e) => {
            eprintln!("  ✗ library compilation FAILED");
            eprintln!("  {:?}", e);
            if std::env::var_os("FERRITE_DEBUG_PRINT_MSL").is_none() {
                eprintln!("  (re-run with FERRITE_DEBUG_PRINT_MSL=1 to inspect the source)");
            }
            panic!("dispatch subtest: MSL compilation FAILED");
        }
    };
    let function = library
        .newFunctionWithName(&NSString::from_str(&kernel.symbol))
        .expect("symbol lookup");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("pipeline creation");
    let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
    eprintln!("  ✓ pipeline created (maxThreads/TG = {})", max_tg);
    assert!(max_tg >= THREADS_PER_TG as usize,
        "PSO doesn't fit our requested threads/TG ({} > {})", THREADS_PER_TG, max_tg);

    // ── Buffers + initialization ──
    use half::{bf16, f16};

    let counter   = util::create_buffer(4);
    let residual  = util::create_buffer(M * HIDDEN * 2);     // bf16
    let delta     = util::create_buffer(M * HIDDEN * 2);     // bf16 (all zeros)
    let rms_w     = util::create_buffer(HIDDEN * 2);          // half
    let out_xnorm = util::create_buffer(HIDDEN * 2);          // bf16

    // residual: deterministic, non-trivial pattern; small magnitudes
    //   so rmsnorm scale is well-conditioned (~1).
    // delta:    all zeros (sidesteps Q-writeback vs K/V-read race).
    // rms_weight: deterministic ramp.
    let mut cpu_residual = vec![bf16::ZERO; HIDDEN];
    let mut cpu_rms_w    = vec![f16::ZERO;  HIDDEN];
    for i in 0..HIDDEN {
        // Pattern: scaled sine — bf16 representable, sum-of-squares ~ HIDDEN/2.
        let x = ((i as f32) * 0.013).sin() * 0.5;
        cpu_residual[i] = bf16::from_f32(x);
        let w = 0.9 + (i as f32 / HIDDEN as f32) * 0.2; // ramp 0.9 → 1.1
        cpu_rms_w[i] = f16::from_f32(w);
    }

    unsafe {
        let p = residual.contents().as_ptr() as *mut bf16;
        for i in 0..HIDDEN { p.add(i).write(cpu_residual[i]); }
        let p = delta.contents().as_ptr() as *mut bf16;
        for i in 0..HIDDEN { p.add(i).write(bf16::ZERO); }
        let p = rms_w.contents().as_ptr() as *mut f16;
        for i in 0..HIDDEN { p.add(i).write(cpu_rms_w[i]); }
        let p = out_xnorm.contents().as_ptr() as *mut bf16;
        for i in 0..HIDDEN { p.add(i).write(bf16::from_f32(f32::NAN)); }
        let p = counter.contents().as_ptr() as *mut u32;
        *p = 0;
    }

    // ── CPU reference (mirrors atom body for delta=0) ──
    // r_new = residual + 0  ⇒  sumsq accumulated in f32 over bf16 inputs
    let mut sumsq = 0.0f64;
    for i in 0..HIDDEN {
        let v = cpu_residual[i].to_f32() as f64;
        sumsq += v * v;
    }
    let mean_sq = sumsq / HIDDEN as f64;
    let scale = 1.0 / (mean_sq + EPS as f64).sqrt();
    let mut cpu_xnorm = vec![0.0f32; HIDDEN];
    for i in 0..HIDDEN {
        let v_pre = cpu_residual[i].to_f32() as f64;
        let w     = cpu_rms_w[i].to_f32() as f64;
        cpu_xnorm[i] = (v_pre * scale * w) as f32;
    }

    // ── Dispatch ──
    let num_tgs: u32 = (M * NUM_HEADS_TOTAL) as u32;
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&counter), 0, 0);
        enc.setBytes_length_atIndex(
            NonNull::new_unchecked(&num_tgs as *const u32 as *mut std::ffi::c_void),
            4, 1,
        );
        enc.setBuffer_offset_atIndex(Some(&residual),  0, 2);
        enc.setBuffer_offset_atIndex(Some(&delta),     0, 3);
        enc.setBuffer_offset_atIndex(Some(&rms_w),     0, 4);
        enc.setBuffer_offset_atIndex(Some(&out_xnorm), 0, 5);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: M, height: NUM_HEADS_TOTAL, depth: 1 },
        MTLSize { width: THREADS_PER_TG as usize, height: 1, depth: 1 },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    eprintln!("  ✓ dispatched grid=({},{},1) × threads=({},1,1), {} TGs total",
        M, NUM_HEADS_TOTAL, THREADS_PER_TG, num_tgs);

    // ── Verify residual_io unchanged (delta=0) ──
    let mut residual_max_diff = 0.0f32;
    let mut residual_diffs = 0;
    unsafe {
        let p = residual.contents().as_ptr() as *const bf16;
        for i in 0..HIDDEN {
            let got = (*p.add(i)).to_f32();
            let exp = cpu_residual[i].to_f32();
            let d = (got - exp).abs();
            if d > residual_max_diff { residual_max_diff = d; }
            if d > 0.0 { residual_diffs += 1; }
        }
    }
    if residual_diffs == 0 {
        eprintln!("  ✓ residual_io bit-identical after kernel (delta=0 round-trip)");
    } else {
        eprintln!("  ✗ residual_io changed in {} slots, max |diff|={:.6}",
            residual_diffs, residual_max_diff);
        panic!("dispatch subtest: residual_io diverged from input");
    }

    // ── Verify out_xnorm matches CPU reference (bf16 tolerance) ──
    // bf16 has ~3 decimal digits of mantissa precision; tolerate ~1e-2
    // absolute / 1% relative.
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut nan_count = 0;
    unsafe {
        let p = out_xnorm.contents().as_ptr() as *const bf16;
        for i in 0..HIDDEN {
            let got = (*p.add(i)).to_f32();
            let exp = cpu_xnorm[i];
            if got.is_nan() { nan_count += 1; continue; }
            let abs_err = (got - exp).abs();
            let rel_err = abs_err / exp.abs().max(1e-6);
            if abs_err > max_abs_err { max_abs_err = abs_err; }
            if rel_err > max_rel_err { max_rel_err = rel_err; }
        }
    }
    eprintln!(
        "  out_xnorm vs CPU ref: max |abs_err|={:.5}, max rel_err={:.5}, nan={}/{}",
        max_abs_err, max_rel_err, nan_count, HIDDEN
    );
    let tol_abs = 1.5e-2;
    let tol_rel = 2.0e-2;
    if nan_count == 0 && max_abs_err < tol_abs && max_rel_err < tol_rel {
        eprintln!("  ✓ AddRmsNormAtom in persistent kernel matches CPU ref within bf16 tolerance");
    } else {
        panic!("dispatch subtest: x_norm diverged from CPU reference \
                (max_abs={:.5} > {}, max_rel={:.5} > {}, nan={})",
            max_abs_err, tol_abs, max_rel_err, tol_rel, nan_count);
    }
}

/// 8-phase persistent kernel where each phase does HIDDEN-shaped
/// work — reduces and scales the prior phase's output cross-TG. End
/// state should be deterministic given the initial pattern.
fn run_realistic_subtest(
    device: &crate::util::Device,
    queue: &CommandQueue,
    num_tgs: u32,
    threads_per_tg: u32,
) {
    const HIDDEN: usize = 2048;
    // Each TG handles `HIDDEN / num_tgs` elements per phase. 16 TGs ×
    // 128 elements/TG = 2048 = HIDDEN. ✓
    let per_tg = HIDDEN / num_tgs as usize;
    assert!(per_tg > 0 && per_tg * num_tgs as usize == HIDDEN);

    // Build 8 phases, each doing the same kind of work:
    //   in[i]  = atomic_load
    //   out[i] = in[i] * 1.001 + 0.0001 * tg_id
    // After 8 phases, out[i] = pow(1.001, 8) * in_initial[i] + small constant accumulation.
    let mut phases = Vec::new();
    for p in 0..8 {
        // Phases alternate ring0 → ring1 → ring0 → ...
        let (in_buf, out_buf) = if p % 2 == 0 { ("ring0", "ring1") } else { ("ring1", "ring0") };
        let body = format!(
            r#"
    {{
        uint per_tg = {hidden}u / num_tgs;
        for (uint i = 0; i < per_tg; ++i) {{
            uint idx = tg_id * per_tg + i;
            if (idx < {hidden}u && tid == i % {tpt}u) {{
                uint v = atomic_load_explicit(
                    (device atomic_uint*)&{in_buf}[idx], memory_order_relaxed);
                float x = as_type<float>(v);
                float y = x * 1.001f + 0.0001f * float(tg_id);
                atomic_store_explicit(
                    (device atomic_uint*)&{out_buf}[idx],
                    as_type<uint>(y), memory_order_relaxed);
            }}
        }}
    }}
"#,
            hidden = HIDDEN,
            tpt = threads_per_tg,
            in_buf = in_buf,
            out_buf = out_buf,
        );
        phases.push(PersistentPhase {
            name: format!("phase_{}_in_{}_out_{}", p, in_buf, out_buf),
            body,
        });
    }

    let extra_sig = r#",
    device uint* ring0 [[buffer(2)]],
    device uint* ring1 [[buffer(3)]]"#;
    let kernel = synthesize_persistent_chunk(
        SynthesisBackend::Metal,
        "synth_persistent_realistic_test",
        threads_per_tg,
        extra_sig,
        "",
        &phases,
    );
    eprintln!(
        "  synthesized {} bytes of MSL with {} phases (HIDDEN={})",
        kernel.source.len(),
        phases.len(),
        HIDDEN
    );

    let source = NSString::from_str(&kernel.source);
    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .expect("realistic library compilation");
    let function = library
        .newFunctionWithName(&NSString::from_str(&kernel.symbol))
        .expect("symbol lookup");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("realistic pipeline creation");
    eprintln!("  ✓ realistic MSL compiled + PSO created");

    let counter = util::create_buffer(4);
    let ring0 = util::create_buffer(HIDDEN * 4);
    let ring1 = util::create_buffer(HIDDEN * 4);

    // Init ring0 with f32(1.0) bit pattern, ring1 with 0
    {
        let p0 = ring0.contents().as_ptr() as *mut u32;
        for i in 0..HIDDEN {
            unsafe { *p0.add(i) = 0x3f800000; }
        }
        let p1 = ring1.contents().as_ptr() as *mut u32;
        for i in 0..HIDDEN {
            unsafe { *p1.add(i) = 0; }
        }
        let pc = counter.contents().as_ptr() as *mut u32;
        unsafe { *pc = 0; }
    }

    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&counter), 0, 0);
        enc.setBytes_length_atIndex(
            NonNull::new_unchecked(&num_tgs as *const u32 as *mut std::ffi::c_void),
            4, 1,
        );
        enc.setBuffer_offset_atIndex(Some(&ring0), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&ring1), 0, 3);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: num_tgs as usize, height: 1, depth: 1 },
        MTLSize { width: threads_per_tg as usize, height: 1, depth: 1 },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();

    // After 8 phases (even count), final output is in ring0.
    let p0 = ring0.contents().as_ptr() as *const f32;
    let mut max_err = 0.0f32;
    let mut nan_count = 0;
    for i in 0..HIDDEN {
        let got = unsafe { *p0.add(i) };
        // tg_id for slot i = i / per_tg
        let tg_id = (i / per_tg) as f32;
        // Compute expected: starting at 1.0, apply 8 iterations of
        // y = x * 1.001 + 0.0001 * tg_id
        let mut x = 1.0f32;
        for _ in 0..8 {
            x = x * 1.001 + 0.0001 * tg_id;
        }
        let err = (got - x).abs() / x.abs().max(1e-6);
        if got.is_nan() { nan_count += 1; }
        if err > max_err { max_err = err; }
    }
    eprintln!(
        "  realistic test: max rel_err = {:.6}, nan_count = {}/{}",
        max_err, nan_count, HIDDEN
    );
    if max_err < 0.001 && nan_count == 0 {
        eprintln!("  ✓ 8-phase HIDDEN-shaped persistent kernel produced correct output");
    } else {
        panic!("synth_persistent_test: realistic subtest FAILED (max_err {:.4})", max_err);
    }
}
