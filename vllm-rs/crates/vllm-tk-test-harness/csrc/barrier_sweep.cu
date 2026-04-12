// SPDX-License-Identifier: Apache-2.0
// Barrier / handoff microbenchmarks for the solver cost model.
//
// Each benchmark measures the per-invocation cost of a synchronization
// mechanism by running it N times inside a kernel and timing the total.
// The caller divides by N to get per-invocation cost.
//
// All kernels write their device-side elapsed clocks to a host-visible
// output so the Rust side can convert to microseconds via the GPU clock.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>

namespace cg = cooperative_groups;

// ── 1. Grid sync benchmark ──
// Cooperative kernel: all threads do N grid-wide barriers.
// Measures cooperative_groups::this_grid().sync() cost.
__global__ void grid_sync_bench_kernel(uint32_t n_iters) {
    auto grid = cg::this_grid();
    // Warmup
    grid.sync();

    // Timed region — device-side timing via globaltimer.
    // We only read the timer on thread 0; all threads participate
    // in the sync.
    uint64_t t0 = 0;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    }

    for (uint32_t i = 0; i < n_iters; i++) {
        grid.sync();
    }

    // Not used — host times with CUDA events instead.
    (void)t0;
}

extern "C" int grid_sync_bench_launch(uint32_t n_iters, uint32_t grid_size,
                                       uint32_t block_size, uint64_t stream) {
    void* args[] = { &n_iters };
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)grid_sync_bench_kernel,
        dim3(grid_size), dim3(block_size),
        args, 0, (cudaStream_t)stream);
    return (int)err;
}

// ── 2. mbarrier handoff benchmark (sm90+) ──
// Single thread does N arrive+wait round-trips on an mbarrier.
// Measures the pure hardware barrier latency without inter-thread
// coordination complexity.
//
// The phase parity flips after each arrive completes, so we track
// it and use try_wait.parity to wait for each phase.
#if __CUDA_ARCH__ >= 900 || !defined(__CUDA_ARCH__)

__global__ void mbarrier_handoff_bench_kernel(uint32_t n_iters) {
    // Only thread 0 participates.
    if (threadIdx.x != 0) return;

    __shared__ alignas(8) uint64_t mbar;
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(&mbar);

    // Initialize with arrival count = 1.
    asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(smem_addr));

    for (uint32_t i = 0; i < n_iters; i++) {
        // Arrive (decrement expected count → barrier completes).
        asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(smem_addr));

        // Wait for phase to flip from 0 to 1 (i.e., parity != 0).
        // After init, internal phase is 0; after arrive completes,
        // it flips to 1. try_wait.parity(0) succeeds when internal
        // phase != 0.
        uint32_t done = 0;
        while (!done) {
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n\t"
                "selp.u32 %0, 1, 0, p;\n\t"
                "}"
                : "=r"(done)
                : "r"(smem_addr), "r"(0)  // always wait for phase != 0
            );
        }

        // Re-init resets internal phase back to 0 and count to 1.
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(smem_addr));
    }
}

extern "C" int mbarrier_handoff_bench_launch(uint32_t n_iters, uint64_t stream) {
    mbarrier_handoff_bench_kernel<<<1, 32, 0, (cudaStream_t)stream>>>(n_iters);
    return (int)cudaGetLastError();
}

#else
extern "C" int mbarrier_handoff_bench_launch(uint32_t n_iters, uint64_t stream) {
    (void)n_iters; (void)stream;
    return -1; // not supported on this arch
}
#endif

// ── 3. Global memory flag spin benchmark ──
// Single thread does N atomicExch + load-back round-trips on a gmem flag.
// Measures the cost of a gmem-flag-based handoff (the sm89 fallback).
__global__ void gmem_flag_bench_kernel(uint32_t n_iters, volatile uint32_t* flag) {
    if (threadIdx.x != 0) return;

    for (uint32_t i = 0; i < n_iters; i++) {
        // Producer side: write flag.
        atomicExch((uint32_t*)flag, i + 1);
        // Consumer side: read flag back (simulates polling).
        // Since it's the same thread, this is immediate — but still
        // forces a gmem round-trip through L2.
        while (atomicAdd((uint32_t*)flag, 0) != i + 1) {}
    }
}

extern "C" int gmem_flag_bench_launch(uint32_t n_iters, uint64_t flag_ptr,
                                       uint64_t stream) {
    volatile uint32_t* flag = (volatile uint32_t*)flag_ptr;
    gmem_flag_bench_kernel<<<1, 32, 0, (cudaStream_t)stream>>>(n_iters, flag);
    return (int)cudaGetLastError();
}

// ── 4. __syncthreads() benchmark ──
// Baseline: measures the cost of a block-level barrier.
__global__ void syncthreads_bench_kernel(uint32_t n_iters) {
    for (uint32_t i = 0; i < n_iters; i++) {
        __syncthreads();
    }
}

extern "C" void syncthreads_bench_launch(uint32_t n_iters, uint32_t block_size,
                                          uint64_t stream) {
    syncthreads_bench_kernel<<<1, block_size, 0, (cudaStream_t)stream>>>(n_iters);
}
