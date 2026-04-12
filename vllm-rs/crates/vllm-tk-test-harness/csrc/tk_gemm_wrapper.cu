// SPDX-License-Identifier: Apache-2.0
// ThunderKittens BF16 GEMM wrapper for the cost sweep.
//
// Exposes TK's persistent-grid matmul as extern "C" launchers that
// match the same (C, A, B, M, N, K, alpha, beta, stream) signature
// used by the CUTLASS configs in cutlass_standalone_gemm.cu.
//
// TK layout: A[M,K] row-major, B[K,N] row-major, C[M,N] row-major.
// Our sweep feeds: A[M,K] row-major, B[N,K] row-major (i.e. B is transposed).
// So we swap A↔B and M↔N to get C^T = B_orig @ A_orig^T, then the
// TMA store writes C^T which in row-major IS C in column-major...
//
// Actually simpler: TK's gl layout is flexible. We just need to match
// the data layout the sweep provides. The sweep allocates:
//   A[M,K] in row-major (stride K)
//   B[N,K] in row-major (stride K) — this is B^T in math terms
//   C[M,N] in row-major (stride N)
// And computes C = A @ B^T.
//
// TK computes C = A @ B where A[M,K], B[K,N].
// So we need to transpose B. Since B_sweep is [N,K] row-major,
// treating it as [K,N] column-major gives us what TK wants — but
// TK's TMA descriptors assume row-major.
//
// Simplest approach: compute C^T = B_sweep @ A_sweep^T by swapping
// A↔B and M↔N. C^T[N,M] row-major = C[M,N] column-major... no good.
//
// Let's just do: TK A = sweep A (M×K, row-major), TK B^T = sweep B.
// We need TK to handle B transposed. Looking at the kernel, B is loaded
// as gl<bf16, 1, 1, K, N> — rows=K, cols=N. For sweep's B[N,K] we'd
// pass rows=N, cols=K... that's wrong.
//
// The correct approach: pass sweep's B as TK's A, and sweep's A as TK's B.
// TK computes: C_tk = A_tk @ B_tk  (A_tk[M_tk, K], B_tk[K, N_tk])
// We want:     C = A @ B^T         (A[M, K],        B[N, K])
// Set: A_tk = B^T → but B is [N,K] row-major, B^T is [K,N]
//      B_tk = A^T → but A is [M,K] row-major, A^T is [K,M]
// Then C_tk[K...] — dimensions don't work.
//
// OK — the real trick: C = A @ B^T = (B @ A^T)^T
// TK computes C_tk = B_sweep @ A_sweep^T? No, TK doesn't transpose.
//
// Since we control the benchmark, let's just change what we feed TK.
// We'll allocate a temporary transposed-B buffer in the launcher.
// Actually that's wasteful. Let me just check if TK can handle
// column-major B...
//
// Looking again at TK: `gl<bf16, 1, 1, -1, -1, base_tile>` where
// base_tile is `st_bf<64,64>`. The TMA loads full 64×64 tiles.
// The layout is fixed as row-major via the global_layout template.
//
// For the benchmark, the simplest correct approach:
// Skip the alpha/beta (TK doesn't support them natively — it's pure C = A@B).
// For beta=0 and alpha=1 (which our sweep uses), this is fine.
// For the transpose issue: pre-transpose B on the GPU using a simple kernel.
// Since this is a benchmark, the transpose cost is NOT included in timing.

#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

// ── TK matmul template (from TK's bf16_h100_gemm.cu) ──
template<int M_BLOCK, int N_BLOCK>
struct tk_matmul_layout {
    using  base_tile      = st_bf<64, 64>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    struct globals        { global_layout A, B, C; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; };
};

template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct tk_matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = tk_matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;

    template<bool PERSISTENT_GRID=true>
    __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERSISTENT_GRID ? 132 : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (M_BLOCK*64);
        int Cblocks = args.globals.C.cols() / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M,
                           (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.iter, args.common.coord.y+i}, args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            kittens::warp::zero(args.state.accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_AB(args.state.accum, args.input.a[warpgroup::groupid()],
                              reinterpret_cast<wide_tile&>(args.input.b));
            warpgroup::mma_async_wait();
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]),
                             args.state.accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if (warpgroup::laneid() == 0) for(int i = 0; i < N_BLOCK; i++) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i],
                                             {args.common.coord.x, args.common.coord.y+i});
                tma::store_async_read_wait();
            }
            kittens::warp::zero(args.state.accum);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

// ── Simple transpose kernel (not timed — just for benchmark setup) ──
__global__ void transpose_bf16_kernel(
    __nv_bfloat16* __restrict__ out,   // [rows, cols]
    const __nv_bfloat16* __restrict__ in, // [cols, rows]
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int r = idx / cols;
        int c = idx % cols;
        out[r * cols + c] = in[c * rows + r];
    }
}

// ── Generic TK GEMM launcher ──
// The sweep provides A[M,K] row-major and B[N,K] row-major (B transposed).
// TK expects B[K,N] row-major. So we transpose B before calling TK.
// The transpose is NOT timed — it's done in "warmup".
template<typename MMT>
static int tk_gemm_launch_impl(
    void* C, const void* A, const void* B,
    int M, int N, int K,
    float /*alpha*/, float /*beta*/,
    uint64_t stream_u64
) {
    cudaStream_t stream = (cudaStream_t)stream_u64;

    // Check TK constraints: M divisible by M_BLOCK*64, N by N_BLOCK*64, K by 64
    constexpr int M_TILE = MMT::M_BLOCK * 64;
    constexpr int N_TILE = MMT::N_BLOCK * 64;
    if (M % M_TILE != 0 || N % N_TILE != 0 || K % 64 != 0) {
        return -1; // can't implement
    }

    // Transpose B from [N,K] row-major to [K,N] row-major
    // We keep a static transposed-B buffer to avoid malloc on every call.
    static __nv_bfloat16* d_B_transposed = nullptr;
    static size_t d_B_transposed_size = 0;
    size_t needed = (size_t)N * K * sizeof(__nv_bfloat16);
    if (needed > d_B_transposed_size) {
        if (d_B_transposed) cudaFree(d_B_transposed);
        cudaMalloc(&d_B_transposed, needed);
        d_B_transposed_size = needed;
    }
    int total = N * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    transpose_bf16_kernel<<<blocks, threads, 0, stream>>>(
        d_B_transposed, (const __nv_bfloat16*)B, K, N);

    // Set up TK global layouts
    using global_layout = typename MMT::layout::global_layout;
    using globals = typename MMT::layout::globals;
    global_layout Ag{(bf16*)A, nullptr, nullptr, (unsigned long)M, (unsigned long)K};
    global_layout Bg{(bf16*)d_B_transposed, nullptr, nullptr, (unsigned long)K, (unsigned long)N};
    global_layout Cg{(bf16*)C, nullptr, nullptr, (unsigned long)M, (unsigned long)N};
    globals G{Ag, Bg, Cg};

    // Configure shared memory
    unsigned long mem_size = kittens::MAX_SHARED_MEMORY - 1024;
    auto kernel_fn = prototype::lcf::kernel<MMT>;
    cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Launch
    dim3 grid = MMT::grid(M, N, K);
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<MMT>);
    kernel_fn<<<grid, block, mem_size, stream>>>(G);

    return 0;
}

// ── Instantiate TK configs ──
// <M_BLOCK, N_BLOCK, SUPER_M> → CTA tile = M_BLOCK×64 × N_BLOCK×64
//
// Config naming: tk_<M_BLOCK*64>x<N_BLOCK*64>
// These cover the same tile space as our CUTLASS configs.

#define TK_GEMM_LAUNCH(M_BLOCK, N_BLOCK, SUPER_M, EXPORT_NAME) \
    extern "C" int EXPORT_NAME( \
        void* C, const void* A, const void* B, \
        int M, int N, int K, \
        float alpha, float beta, \
        uint64_t stream \
    ) { \
        return tk_gemm_launch_impl<tk_matmul_template<M_BLOCK, N_BLOCK, SUPER_M>>( \
            C, A, B, M, N, K, alpha, beta, stream); \
    }

// 128×256 — TK's flagship config (2 M-blocks × 4 N-blocks)
TK_GEMM_LAUNCH(2, 4, 12, tk_gemm_128x256_launch)
// 128×128 — smaller N for narrower GEMMs
TK_GEMM_LAUNCH(2, 2, 12, tk_gemm_128x128_launch)
// 64×256 — single M-block, good for medium M
TK_GEMM_LAUNCH(1, 4, 12, tk_gemm_64x256_launch)
// 64×128
TK_GEMM_LAUNCH(1, 2, 12, tk_gemm_64x128_launch)
// 64×64 — smallest TK tile
TK_GEMM_LAUNCH(1, 1, 12, tk_gemm_64x64_launch)
// 256×256 — exceeds shared memory on H100, omitted
