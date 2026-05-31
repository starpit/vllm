// SPDX-License-Identifier: Apache-2.0
//
// Hopper-native persistent-CTA megakernel scaffold for ferrite-wavefront.
//
// Hand-written ONCE; reused across every per-canonical .cu the macro
// emits. The persistent-CTA layout, mbarrier discipline, page pool,
// instruction ring, and role dispatch live here. Per-IType .cuh files
// fill in compute bodies only — they MUST NOT redefine any constant
// or barrier in this file.
//
// Layout cribbed from ~/git/Megakernels main include/{config,megakernel}.cuh
// (Hopper-tested, TK 1.0). We translate to TK 2.0 primitives at the
// per-IType .cuh tier.
//
// Substrate constants are compile-time. Every IType template is
// parameterized on shape const generics (HIDDEN_SIZE, INTERMEDIATE_SIZE,
// NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, VOCAB_SIZE, NUM_LAYERS,
// KV_BLOCK_SIZE) supplied by the macro per canonical — no Llama-1B
// literals in this file.

#pragma once

#include "kittens.cuh"

namespace ferrite_mk {

// ── CTA layout ────────────────────────────────────────────────────────
// 20 warps total = 4 service warps (loader/storer/launcher/controller)
// + 16 consumer warps. Warpgroup-aligned for Hopper wgmma.

constexpr int NUM_PAGES                    = 13;
constexpr int PAGE_SIZE                    = 16384;       // 16 KB per page
constexpr int INSTRUCTION_PIPELINE_STAGES  = 2;
constexpr int INPUT_PIPELINE_STAGES        = 3;
constexpr int OUTPUT_PIPELINE_STAGES       = 3;
constexpr int NUM_CONSUMER_WARPS           = 16;
constexpr int NUM_NON_CONSUMER_WARPS       = 4;
constexpr int NUM_WARPS                    = NUM_NON_CONSUMER_WARPS + NUM_CONSUMER_WARPS;
constexpr int NUM_THREADS                  = NUM_WARPS * 32;
constexpr int DYNAMIC_SHARED_MEMORY        = NUM_PAGES * PAGE_SIZE;  // 208 KB
constexpr int CONSUMER_REGISTERS           = 224;
constexpr int NON_CONSUMER_REGISTERS       = 56;
constexpr int DYNAMIC_SEMAPHORES           = 32;

// Service-warp role IDs. WG0 lane = warpgroup::warpid() switch index in
// the kernel entry. Order matches main-branch's mks dispatch.
enum ServiceRole : int {
    ROLE_LOADER     = 0,
    ROLE_STORER     = 1,
    ROLE_LAUNCHER   = 2,
    ROLE_CONTROLLER = 3,
};

// ── Phase 0 stub ─────────────────────────────────────────────────────
//
// The full scaffold (init handshake, fence.proxy.async.shared::cta,
// role-dispatch switch, controller's instruction-fetch + pid_order +
// init_semaphores, loader/storer/launcher worker loops, instruction
// ring, NoOp opcode) lands across phases 0-1. Phase 0 only requires
// this header to be #include-able; the per-canonical .cu builds
// against an empty IType pack as a compile gate.

}  // namespace ferrite_mk
