// SPDX-License-Identifier: Apache-2.0
// FA2 attention body ported from ThunderKittens'
// `kernels/attention/mha_h100/mha_h100.cu` fwd_attend_ker, stripped
// of the `__global__` decorator so it's callable from our
// megakernel's top-level `__global__`.
//
// All the real FA2 logic (online softmax rescale, causal masking,
// pipelined K/V loads via TMA, wgmma QK^T + PV, epilogue log-sum-exp
// store) comes verbatim from TK. Only the outer function signature
// changed: `__global__ __launch_bounds__(...) fwd_attend_ker(const
// __grid_constant__ fwd_globals<D> g)` →
// `static __device__ void attn_prefill_body(const fwd_globals<D>&
// g)`.
//
// Context that needs to be in scope at include site:
//   - `kittens.cuh` (bf16, gl, sv_bf, st_bf, tma, warp, warpgroup, …)
//   - `cooperative_groups.h` for cg::this_grid() — not actually used
//     by fwd_attend_ker but pulled in by TK's other kernels; we skip.

#pragma once

namespace kittens_ferrite_attn {

// Warpgroup layout — matches TK mha_h100.cu lines 7-11.
constexpr int CONSUMER_WARPGROUPS = 3;
constexpr int PRODUCER_WARPGROUPS = 1;
constexpr int NUM_WARPGROUPS      = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
constexpr int NUM_WORKERS         = NUM_WARPGROUPS * kittens::WARPGROUP_WARPS;

template<int D> struct fwd_attend_ker_tile_dims {};
template<> struct fwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = 64;
    constexpr static int qo_height  = 4 * 16;
    constexpr static int kv_height  = 8 * 16;
    constexpr static int stages     = 4;
};
template<> struct fwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = 128;
    constexpr static int qo_height  = 4 * 16;
    constexpr static int kv_height  = 8 * 16;
    constexpr static int stages     = 2;
};

template<int D> struct fwd_globals {
    using q_tile    =                 kittens::st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using k_tile    =                 kittens::st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using v_tile    =                 kittens::st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using l_col_vec = kittens::col_vec<kittens::st_fl<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>>;
    using o_tile    =                 kittens::st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;

    using q_gl = kittens::gl<kittens::bf16, -1, -1, -1, -1, q_tile>;
    using k_gl = kittens::gl<kittens::bf16, -1, -1, -1, -1, k_tile>;
    using v_gl = kittens::gl<kittens::bf16, -1, -1, -1, -1, v_tile>;
    using l_gl = kittens::gl<float,         -1, -1, -1, -1, l_col_vec>;
    using o_gl = kittens::gl<kittens::bf16, -1, -1, -1, -1, o_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    l_gl l;
    o_gl o;
    const int N;
    const int hr;  // GQA ratio: num_q_heads / num_kv_heads.
};

// Verbatim port of fwd_attend_ker from mha_h100.cu, stripped of its
// `__global__ __launch_bounds__` decorator so it runs as a __device__
// function inside our megakernel. Must be called with a grid/block
// shape that matches what TK's launcher uses:
//   grid  = dim3(seq_len / (CONSUMER_WARPGROUPS * TILE_ROW_DIM<bf16> * 4), qo_heads, batch)
//   block = dim3(32 * NUM_WORKERS)  // 16 warps
//   dynamic shared = MAX_SHARED_MEMORY - 1024
template<int D, bool is_causal>
static __device__ void attn_prefill_body(const fwd_globals<D>& g) {
    using namespace kittens;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpid / kittens::WARPGROUP_WARPS;

    using K = fwd_attend_ker_tile_dims<D>;

    using q_tile    =         st_bf<K::qo_height, K::tile_width>;
    using k_tile    =         st_bf<K::kv_height, K::tile_width>;
    using v_tile    =         st_bf<K::kv_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;
    using o_tile    =         st_bf<K::qo_height, K::tile_width>;

    q_tile    (&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
    k_tile    (&k_smem)[K::stages]           = al.allocate<k_tile, K::stages          >();
    v_tile    (&v_smem)[K::stages]           = al.allocate<v_tile, K::stages          >();
    l_col_vec (&l_smem)[CONSUMER_WARPGROUPS] = al.allocate<l_col_vec, CONSUMER_WARPGROUPS>();
    auto      (*o_smem)                      = reinterpret_cast<o_tile(*)>(q_smem);

    int kv_blocks   = g.N / (K::kv_height);
    int kv_head_idx = blockIdx.y / g.hr;
    int seq_idx     = blockIdx.x * CONSUMER_WARPGROUPS;

    __shared__ kittens::semaphore qsmem_semaphore, k_smem_arrived[K::stages], v_smem_arrived[K::stages], compute_done[K::stages];
    if (threadIdx.x == 0) {
        init_semaphore(qsmem_semaphore, 0, 1);
        for(int j = 0; j < K::stages; j++) {
            init_semaphore(k_smem_arrived[j], 0, 1);
            init_semaphore(v_smem_arrived[j], 0, 1);
            init_semaphore(compute_done[j], CONSUMER_WARPGROUPS, 0);
        }

        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));

        for (int wg = 0; wg < CONSUMER_WARPGROUPS; wg++) {
            coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + wg, 0};
            tma::load_async(q_smem[wg], g.q, q_tile_idx, qsmem_semaphore);
        }

        for (int j = 0; j < K::stages - 1; j++) {
            coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, j, 0};
            tma::expect_bytes(k_smem_arrived[j], sizeof(k_tile));
            tma::load_async(k_smem[j], g.k, kv_tile_idx, k_smem_arrived[j]);
            tma::expect_bytes(v_smem_arrived[j], sizeof(v_tile));
            tma::load_async(v_smem[j], g.v, kv_tile_idx, v_smem_arrived[j]);
        }
    }
    __syncthreads();

    int pipe_idx = K::stages - 1;

    if(warpgroupid == NUM_WARPGROUPS-1) {
        warpgroup::decrease_registers<32>();

        int kv_iters;
        if constexpr (is_causal) {
            kv_iters = (seq_idx * (K::qo_height/kittens::TILE_ROW_DIM<bf16>)) - 1 + (CONSUMER_WARPGROUPS * (K::qo_height/kittens::TILE_ROW_DIM<bf16>));
            kv_iters = ((kv_iters / (K::kv_height/kittens::TILE_ROW_DIM<bf16>)) == 0) ? (0) : ((kv_iters / (K::kv_height/kittens::TILE_ROW_DIM<bf16>)) - 1);
        }
        else { kv_iters = kv_blocks-2; }

        if(warpid == NUM_WORKERS-4) {
            for (auto kv_idx = pipe_idx - 1; kv_idx <= kv_iters; kv_idx++) {
                coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, kv_idx + 1, 0};
                warp::tma::expect_bytes(k_smem_arrived[(kv_idx+1)%K::stages], sizeof(k_tile));
                warp::tma::load_async(k_smem[(kv_idx+1)%K::stages], g.k, kv_tile_idx, k_smem_arrived[(kv_idx+1)%K::stages]);
                warp::tma::expect_bytes(v_smem_arrived[(kv_idx+1)%K::stages], sizeof(v_tile));
                warp::tma::load_async(v_smem[(kv_idx+1)%K::stages], g.v, kv_tile_idx, v_smem_arrived[(kv_idx+1)%K::stages]);

                wait(compute_done[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            }
        }
    }
    else {
        warpgroup::increase_registers<160>();

        rt_fl<16, K::kv_height>  att_block;
        rt_bf<16, K::kv_height>  att_block_mma;
        rt_fl<16, K::tile_width> o_reg;

        col_vec<rt_fl<16, K::kv_height>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;

        warp::neg_infty(max_vec);
        warp::zero(norm_vec);
        warp::zero(o_reg);

        int kv_iters;
        if constexpr (is_causal) {
            kv_iters = (seq_idx * 4) - 1 + (CONSUMER_WARPGROUPS * 4);
            kv_iters = (kv_iters/8);
        }
        else { kv_iters = kv_blocks - 1; }

        wait(qsmem_semaphore, 0);

        for (auto kv_idx = 0; kv_idx <= kv_iters; kv_idx++) {
            wait(k_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[(kv_idx)%K::stages]);

            warp::copy(max_vec_last_scaled, max_vec);
            if constexpr (D == 64) { warp::mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.125f); }
            else                   { warp::mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.08838834764f); }

            warpgroup::mma_async_wait();

            if constexpr (is_causal) {
                const int q_blk = (seq_idx * (K::qo_height/kittens::TILE_ROW_DIM<bf16>)) + warpid;
                      int k_blk = (kv_idx * (K::kv_height/kittens::TILE_ROW_DIM<bf16>));

                #pragma unroll
                for(; k_blk == (kv_iters-1)*(K::kv_height/kittens::TILE_ROW_DIM<bf16>) || k_blk == (kv_iters)*(K::kv_height/kittens::TILE_ROW_DIM<bf16>); k_blk+=10000) {
                    #pragma unroll
                    for (auto j = 0; j < (K::kv_height/kittens::TILE_ROW_DIM<bf16>); j++) {
                        auto k_idx = k_blk + j;
                        auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(att_block.tiles[0][j]);

                        if      (k_idx >  q_blk) { warp::neg_infty  (attn_subtile); }
                        else if (k_idx == q_blk) { warp::make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty()); }
                        __syncwarp();
                    }
                }
            }

            warp::row_max(max_vec, att_block, max_vec);

            if constexpr (D == 64) {
                warp::mul(att_block, att_block,    1.44269504089f*0.125f);
                warp::mul(max_vec_scaled, max_vec, 1.44269504089f*0.125f);
            }
            else                   {
                warp::mul(att_block, att_block,    1.44269504089f*0.08838834764f);
                warp::mul(max_vec_scaled, max_vec, 1.44269504089f*0.08838834764f);
            }

            warp::sub_row(att_block, att_block, max_vec_scaled);
            warp::exp2(att_block, att_block);
            warp::sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            warp::exp2(max_vec_last_scaled,       max_vec_last_scaled);
            warp::mul(norm_vec,            norm_vec,     max_vec_last_scaled);
            warp::row_sum(norm_vec,  att_block, norm_vec);
            warp::add(att_block, att_block, 0.f);
            warp::copy(att_block_mma, att_block);
            warp::mul_row(o_reg, o_reg, max_vec_last_scaled);

            wait(v_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2);

            warpgroup::mma_AB(o_reg, att_block_mma, v_smem[(kv_idx)%K::stages]);
            warpgroup::mma_async_wait();

            if(warpgroup::laneid() == 0) arrive(compute_done[(kv_idx)%K::stages], 1);
        }

        warp::div_row(o_reg, o_reg, norm_vec);
        warpgroup::store(o_smem[warpgroupid], o_reg);
        warpgroup::sync(warpgroupid+4);

        if (warpid % 4 == 0) {
            coord<o_tile> o_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + warpgroupid, 0};
            warp::tma::store_async(g.o, o_smem[warpgroupid], o_tile_idx);
        }

        warp::mul(max_vec_scaled,   max_vec_scaled, 0.69314718056f);
        warp::log(norm_vec, norm_vec);
        warp::add(norm_vec, norm_vec, max_vec_scaled);

        if constexpr (D == 64) { warp::mul(norm_vec, norm_vec, -8.0f); }
        else                   { warp::mul(norm_vec, norm_vec, -11.313708499f); }

        warpgroup::store(l_smem[warpgroupid], norm_vec);
        warpgroup::sync(warpgroupid+4);

        if (warpid % 4 == 0) {
            coord<l_col_vec> tile_idx = {blockIdx.z, blockIdx.y, 0, (seq_idx) + warpgroupid};
            warp::tma::store_async(g.l, l_smem[warpgroupid], tile_idx);
        }
        warp::tma::store_async_wait();
    }
}

}  // namespace kittens_ferrite_attn
