// SPDX-License-Identifier: Apache-2.0
// Ferrite-TK megakernel substrate — 4-way warp role dispatch.
//
// See FERRITE_TK_PLAN.md, "Warp layout — four roles".
//
// Each warp in the kernel takes exactly one of four roles based
// on its warpid(). Consumer warps get high register allocation
// (MMA accumulation, softmax, reductions); loader / launcher /
// storer warps get the low allocation (TMA issue, wgmma issue,
// TMA store issue). On Hopper first-cut the launcher role is
// folded into consumer — this header still exposes the
// launcher slot so codegen can emit an explicit launcher_body
// when we split it out.
//
// The dispatch itself is a single `if`/`else` that each warp
// evaluates on kernel entry. Codegen emits calls to the four
// body functions (`consumer_body`, `loader_body`,
// `launcher_body`, `storer_body`) directly — this header only
// exposes the role-id helpers and register-budget wrappers.

#pragma once

#include "kittens.cuh"

namespace ferrite {

// Non-consumer slot indices, stable across every variant the
// codegen emits. warpid() values in
// [0, NUM_CONSUMER_WARPS) are consumers; the three slots above
// that are the non-consumer roles, ordered as below.
enum NonConsumerSlot : int {
    kLoaderSlot   = 0,
    kLauncherSlot = 1,
    kStorerSlot   = 2,
    kNumNonConsumerSlots = 3,
};

// Register-allocation wrappers. Used by codegen immediately
// after role dispatch, before the role body runs:
//
//   if (warpid() < Config::NUM_CONSUMER_WARPS) {
//       ferrite::set_consumer_registers<Config>();
//       consumer_body(g, ss, warpid());
//   } else {
//       ferrite::set_non_consumer_registers<Config>();
//       switch (warpid() - Config::NUM_CONSUMER_WARPS) {
//           case ferrite::kLoaderSlot:   loader_body  (g, ss); break;
//           case ferrite::kLauncherSlot: launcher_body(g, ss); break;
//           case ferrite::kStorerSlot:   storer_body  (g, ss); break;
//       }
//   }
template <typename Config>
__device__ __forceinline__ void set_consumer_registers() {
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(Config::CONSUMER_REGISTERS));
#endif
}

template <typename Config>
__device__ __forceinline__ void set_non_consumer_registers() {
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(Config::NON_CONSUMER_REGISTERS));
#endif
}

} // namespace ferrite
