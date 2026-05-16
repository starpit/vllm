// SPDX-License-Identifier: Apache-2.0
// Ferrite-TK megakernel substrate — device pointer-type aliases.
//
// See FERRITE_TK_PLAN.md at the repo root.
//
// Each codegen'd `.cu` passes its pointer args positionally
// through the kernel signature — no `Globals` wrapper struct.
// Every scalar (model dims, workload shape, eps) is baked in as
// `static constexpr`, since ferrite generates one kernel per
// variant.
//
// This header only exposes the device-pointer aliases op headers
// lean on. Nothing more.

#pragma once

// Our kernels are device-only. KITTENS_NO_HOST suppresses the host-side
// standard library includes (<cuda_runtime.h>, <iostream>, <vector>, etc.)
// that kittens.cuh otherwise pulls in, keeping the preprocessed TU small.
#ifndef KITTENS_NO_HOST
#define KITTENS_NO_HOST
#endif

#include "kittens.cuh"

namespace ferrite {

// Raw device-pointer aliases. bf16 is the only activation dtype
// for Phase 2-3 baseline; fp32 shows up as accumulator-tensor
// pointers, u32 for KV block tables, i32 for cross-SM barrier
// counters. Add quant-weight ptr aliases in the follow-up
// quantized-variant phase, not here.
using bf16_ptr = __nv_bfloat16*;
using bf16_cptr = const __nv_bfloat16*;
using f32_ptr = float*;
using f32_cptr = const float*;
using u32_ptr = uint32_t*;
using u32_cptr = const uint32_t*;
using i32_ptr = int32_t*;

} // namespace ferrite
