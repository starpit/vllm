// SPDX-License-Identifier: Apache-2.0
//
// 256-byte typed instruction wire format. Mirrored byte-for-byte by
// `ferrite_wavefront::mk::instruction::Instruction` on the Rust side;
// `static_assert(sizeof(instruction_t) == 256)` here MUST match the
// equivalent Rust-side `assert_eq!(size_of::<Instruction>(), 256)`.
//
// The kernel's `instructions_t` global is `kittens::gl<instruction_t,
// ...>` (NOT `gl<int, ...>`) so the C++ side reads typed fields by
// name, never by raw `int[N]` index decode.

#pragma once

#include <cstdint>

namespace ferrite_mk {

// ── Tensor reference — small handle into MKGlobals ───────────────────
// `tensor_kind` selects which `gl<>` in MKGlobals; `byte_offset` is the
// outer-axis offset for layered weights (per-layer pointer offset).
struct tensor_ref_t {
    uint32_t tensor_kind;     // enum value (per-canonical)
    uint32_t layer_or_index;  // layer index for stacked weights, or row index
    uint64_t byte_offset;     // base-pointer offset in bytes
};

// ── Barrier reference — points into the in-kernel barrier table ──────
// Each (layer, opcode, block_idx/4) gets a unique barrier slot; src/dst
// barriers are referenced by index here.
struct barrier_ref_t {
    int32_t  index;            // -1 = none
    uint32_t expected_arrives; // arrive-count for this barrier
};

// ── 256-byte typed instruction. ───────────────────────────────────────
// Field order is part of the ABI — Rust-side mirror MUST match exactly.
// Fixed `_pad` brings the total to exactly 256 bytes regardless of
// indices[16] alignment.
struct alignas(16) instruction_t {
    uint16_t       opcode;             //  2  IType discriminant
    uint16_t       layer_idx;          //  2  decoder layer
    uint32_t       _pad0;              //  4
    tensor_ref_t   src[4];             // 64  up to 4 input refs
    tensor_ref_t   dst[2];             // 32  up to 2 output refs
    int32_t        indices[16];        // 64  per-IType numeric params
    barrier_ref_t  src_barriers[2];    // 16
    barrier_ref_t  dst_barriers[2];    // 16
    uint8_t        _pad1[56];          // 56  to bring total to 256
};
static_assert(sizeof(instruction_t) == 256,
    "instruction_t MUST be exactly 256 bytes — see Rust-side mirror.");

// NoOp opcode — every per-canonical schedule pads its per-SM queue
// with these so idle SMs in persistent multi-CTA dispatch don't hang.
constexpr uint16_t OPCODE_NOOP = 0;

}  // namespace ferrite_mk
