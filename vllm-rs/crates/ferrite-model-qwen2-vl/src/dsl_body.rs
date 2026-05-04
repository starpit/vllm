// SPDX-License-Identifier: Apache-2.0
//! Qwen2-VL vision encoder DSL body — stub closing G.5.e.
//!
//! Currently a one-op smoke through `quick_gelu(pixels)`: it
//! exercises the full pipeline (parse → classify → CFG → unroll →
//! `vision_lowering::materialize_pixels` → solve → schedule →
//! codegen) and proves the vision-prelude classification, the
//! pixels-extern materialization pass (G.5.e.1), and the
//! `LoadPixelsImpl` / `Instruction::LoadPixels` runtime variant
//! all hand off cleanly.
//!
//! Replaced wholesale by G.5.f's real encoder body — patch_embed
//! gemm, 32-block transformer loop, ln_q + merger MLP — once the
//! reshape/bound-arithmetic and weight-anchoring bits land.

use ferrite_forward::vision_forward;

#[vision_forward(workloads = [256, 1024, 4096, 16384])]
fn qwen2_vl() {
    out = quick_gelu(pixels);
}
