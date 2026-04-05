// sm89 port of attention_decode.cu
// The consumer (Q@K^T, softmax, @V) already uses warp-scope mma.sync — no WGMMA!
// Changes vs Hopper:
//   - Loader: tma::expect/load_async → cp.async (group/warp::load_async) + arrive
//   - Launcher: s.wait_tensor_ready() → no-op (sm89 has no tensor allocator)
//   - Storer: tma::store_async → warp::store
//   - NUM_STAGES computed from shmem budget (sm89 has fewer pages)
//   - kv_cache_t has no tma::descriptor on sm89

#include "llama_sm89.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm {

using globals = llama_sm89_globals;
using config  = llama_sm89_config;

template <typename Config = config, typename Globals = globals>
struct attention_decode {
    static constexpr int opcode             = OPCODE_GQA_AttentionDecode;
    static constexpr int GQA_RATIO          = Globals::num_attention_heads / Globals::num_kv_heads;
    static constexpr int ATTN_BATCH_BLOCK_SIZE = GQA_RATIO;

    static_assert(GQA_RATIO >= 1 && GQA_RATIO <= 8, "GQA_RATIO must be 1..8");
    static_assert(GQA_RATIO * 2 <= 16, "GQA_RATIO * 2 must fit in q_st rows (16)");
    static_assert(ATTN_BATCH_BLOCK_SIZE <= Config::NUM_CONSUMER_WARPS,
                  "Need at least GQA_RATIO consumer warps for attention");

    static constexpr int head_dim       = Globals::head_dim;
    static constexpr int kv_block_size  = Globals::kv_block_size;
    static constexpr int iters_per_page = Globals::iters_per_page;
    static constexpr int kv_page_size   = Globals::kv_page_size;

    // KV tile: st_bf<kv_block_size, head_dim> per batch item
    using kv_st    = st_bf<kv_block_size, head_dim>;
    // Pages needed per stage: ceil(ATTN_BATCH_BLOCK_SIZE * kv_st / PAGE_SIZE) for K + same for V
    static constexpr int KV_TILE_BYTES  = sizeof(kv_st) * ATTN_BATCH_BLOCK_SIZE;
    static constexpr int K_PAGES        = (KV_TILE_BYTES + Config::PAGE_SIZE - 1) / Config::PAGE_SIZE;
    static constexpr int V_PAGES        = K_PAGES;
    static constexpr int PAGES_PER_STAGE = K_PAGES + V_PAGES;
    // Use as many stages as we can fit in the page budget.
    static constexpr int NUM_STAGES     = Config::NUM_PAGES / PAGES_PER_STAGE;
    static_assert(NUM_STAGES >= 1, "Not enough shmem pages for even 1 attention stage.");

    using q_rt       = rt_bf<16, head_dim>;
    using q_st       = st_bf<16, head_dim>;
    using k_rt       = rt_bf<kv_block_size, head_dim>;
    using v_rt       = rt_bf<kv_block_size, head_dim, col_l>;
    using attn_fl_rt = rt_fl<16, kv_block_size>;
    using attn_bf_rt = rt_bf<16, kv_block_size>;
    using max_vec_rv = col_vec<rt_fl<16, head_dim>>;
    using norm_vec_rv= col_vec<rt_fl<16, head_dim>>;
    using o_rt       = rt_fl<16, head_dim>;
    using o_rt_bf    = rt_bf<16, head_dim>;
    using o_sv       = sv_bf<head_dim>;

    struct parsed_instruction {
        int layer_idx;
        int batch_block_idx;
        int kv_head_idx;
        __device__ inline parsed_instruction(typename Config::instruction_t &i) {
            layer_idx       = i[1];
            batch_block_idx = i[2];
            kv_head_idx     = i[3];
        }
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    // ── semaphore accessors ──────────────────────────────────────────────────
    __device__ static inline semaphore &O_arrived(state<Config> &s)             { return s.semaphores()[0]; }
    __device__ static inline semaphore &K_arrived(state<Config> &s, int stage)  { return s.semaphores()[1 + stage * 2]; }
    __device__ static inline semaphore &V_arrived(state<Config> &s, int stage)  { return s.semaphores()[1 + stage * 2 + 1]; }
    __device__ static inline semaphore &K_finished(state<Config> &s, int stage) { return s.semaphores()[1 + NUM_STAGES * 2 + stage * 2]; }
    __device__ static inline semaphore &V_finished(state<Config> &s, int stage) { return s.semaphores()[1 + NUM_STAGES * 2 + stage * 2 + 1]; }

    // ── shmem layout helpers ─────────────────────────────────────────────────
    __device__ static inline void wait_KV_page(state<Config> &s, int stage) {
        // Stage uses pages [stage*PAGES_PER_STAGE .. stage*PAGES_PER_STAGE + K_PAGES + V_PAGES).
        for (int p = 0; p < PAGES_PER_STAGE; p++)
            s.wait_page_ready(s.pid(stage * PAGES_PER_STAGE + p));
    }
    __device__ static inline void finish_KV_page(state<Config> &s, int stage) {
        int count = Config::NUM_CONSUMER_WARPS / ATTN_BATCH_BLOCK_SIZE;
        for (int p = 0; p < PAGES_PER_STAGE; p++)
            s.finish_page(s.pid(stage * PAGES_PER_STAGE + p), count);
    }

    __device__ static inline kv_st &get_K_smem(state<Config> &s, int stage, int batch_idx) {
        // K tiles are in the first K_PAGES pages of the stage.
        char *base = reinterpret_cast<char *>(s.pages[s.pid(stage * PAGES_PER_STAGE)].data);
        return *reinterpret_cast<kv_st *>(base + sizeof(kv_st) * batch_idx);
    }
    __device__ static inline kv_st &get_V_smem(state<Config> &s, int stage, int batch_idx) {
        // V tiles follow K tiles (in their own pages).
        char *base = reinterpret_cast<char *>(s.pages[s.pid(stage * PAGES_PER_STAGE + K_PAGES)].data);
        return *reinterpret_cast<kv_st *>(base + sizeof(kv_st) * batch_idx);
    }

    // Q and O share the same scratch region (same as Hopper).
    // Scratch layout: o_sv[4] per batch item, ATTN_BATCH_BLOCK_SIZE items.
    __device__ static inline q_st &get_Q_smem(state<Config> &s, int batch_idx) {
        return *reinterpret_cast<q_st *>(reinterpret_cast<char *>(s.scratch()) + sizeof(o_sv) * batch_idx * 4);
    }
    __device__ static inline o_sv (&get_O_smem(state<Config> &s, int batch_idx))[4] {
        return *reinterpret_cast<o_sv(*)[4]>(reinterpret_cast<char *>(s.scratch()) + sizeof(o_sv) * batch_idx * 4);
    }

    // Load Q using cp.async (same approach as the original load_Q_async which already used cp.async).
    __device__ static inline void load_Q_async(q_st &dst,
                                               const typename Globals::activations_t &src,
                                               int batch_idx, int q_head_start_idx) {
        using T = typename q_st::dtype;
        constexpr int elem_per_memcpy = sizeof(float4) / sizeof(T);
        constexpr int memcpy_per_row  = head_dim / elem_per_memcpy;

        typename Globals::activations_t::dtype *src_ptr =
            (typename Globals::activations_t::dtype *)&src[coord<>{batch_idx, q_head_start_idx * head_dim}];
        uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
        int lid = warp::laneid();
        int col = (lid % memcpy_per_row) * elem_per_memcpy;
        int base_row = (lid < memcpy_per_row) ? 0 : 1;

        for (int i = 0; i < (GQA_RATIO / 2); i++) {
            int row = base_row + i * 2;
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::
                "r"(dst.idx(dst_ptr, {row, col})),
                "l"(&src_ptr[row * head_dim + col]) : "memory");
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    // store_4_rows: unchanged from the original (pure register→shmem, no TMA).
    template <ducks::sv::all SV, ducks::rt::all RT>
    __device__ static inline void store_4_rows(SV (&dst)[4], const RT &src) {
        static_assert(RT::rows == 16);
        static_assert(SV::length == src.cols);
        using T2 = typename RT::dtype;
        using U  = typename SV::dtype;
        using U2 = typename base_types::packing<U>::packed_type;
        uint32_t dst_ptr[4];
        for (int i = 0; i < 4; ++i)
            dst_ptr[i] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[i].data[0]));
        int lid = kittens::laneid();
        if (lid < 16) {
            int lr = lid / 4, lc = lid % 4;
            for (int j = 0; j < src.width; j++) {
                U2 tmp[2];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[2]);
                int ci = lc * 2 + j * 16;
                move<U2>::sts(dst_ptr[lr] + sizeof(U) * ci,     tmp[0]);
                move<U2>::sts(dst_ptr[lr] + sizeof(U) * (ci+8), tmp[1]);
            }
        }
    }

    template <ducks::rt::row_layout RT>
    __device__ static inline void right_fill(RT &dst, const RT &src, int col_idx,
                                             typename base_types::packing<typename RT::dtype>::unpacked_type val = 0) {
        if (col_idx >= dst.cols) return;
        for (int i = 0; i < dst.height; i++)
            for (int j = 0; j < dst.width; j++)
                for (int k = 0; k < dst.packed_per_tile; k++) {
                    auto &d = dst.tiles[i][j].data[k];
                    auto &sv = src.tiles[i][j].data[k];
                    int cx = (j * dst.tile_size_col) + ((k / 2) * 8) + ((warp::laneid() % 4) * 2);
                    int cy = cx + 1;
                    d.x = (cx >= col_idx) ? val : sv.x;
                    d.y = (cy >= col_idx) ? val : sv.y;
                }
    }

    // ── controller ───────────────────────────────────────────────────────────
    struct controller {
        static __device__ int release_lid(const Globals &g,
                                          typename Config::instruction_t &ins, int &query) {
            // Identity mapping: keep page order stable across instructions.
            // The original cycling formula (query % NUM_STAGES) * PAGES_PER_STAGE
            // creates aliased page slots when NUM_PAGES > NUM_STAGES * PAGES_PER_STAGE,
            // causing the loader's unused-page-release branch to free actively-used pages.
            return query;
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            init_semaphore(O_arrived(s), 0, ATTN_BATCH_BLOCK_SIZE);
            for (int i = 0; i < NUM_STAGES; i++) {
                // Loader is cooperative (all lanes load together), so K/V arrived count = 1.
                // Consumers are per-warp (4 warps), so K/V finished count = ATTN_BATCH_BLOCK_SIZE.
                init_semaphore(K_arrived(s, i),  0, 1);
                init_semaphore(V_arrived(s, i),  0, 1);
                init_semaphore(K_finished(s, i), 0, ATTN_BATCH_BLOCK_SIZE);
                init_semaphore(V_finished(s, i), 0, ATTN_BATCH_BLOCK_SIZE);
            }
            return 1 + 4 * NUM_STAGES;
        }
    };

    // ── loader ───────────────────────────────────────────────────────────────
    // On Hopper, each lane independently issues a TMA load (per-thread op).
    // On sm89, warp::load_async is cooperative — all 32 lanes must participate.
    // So we loop over batch items, with all lanes cooperating on each load.
    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            int lid = warp::laneid();

            // Each batch item in the attention block has its own paged KV sequence.
            // For simplicity in the loader, we process batch items sequentially,
            // loading K and V tiles for each item's page table.
            // The batch_block contains ATTN_BATCH_BLOCK_SIZE items.
            // For decode, each item contributes one new token, so we iterate
            // over the paged KV history for each item.

            // We use a simplified approach: process one batch item at a time
            // (matching the consumer's per-warp model).  The loader loads
            // KV for batch item 0 first, then item 1, etc.
            // TODO: For better pipelining, interleave items across stages.

            // For now, handle the common single-item case (ATTN_BATCH_BLOCK_SIZE items
            // processed by ATTN_BATCH_BLOCK_SIZE consumer warps, one warp per item).

            // Compute total_attn_blocks for the FIRST batch item (all items in a
            // decode batch have similar lengths; the consumer handles masking).
            int first_batch_idx = inst.batch_block_idx * ATTN_BATCH_BLOCK_SIZE;

            // We need the maximum total_attn_blocks across all items in this block.
            int max_total_blocks = 0;
            for (int b = 0; b < ATTN_BATCH_BLOCK_SIZE; b++) {
                int seq_idx = first_batch_idx + b;
                int indptr_start = g.decode_kv_indptr[{seq_idx}];
                int indptr_end   = g.decode_kv_indptr[{seq_idx + 1}];
                int np = indptr_end - indptr_start;
                int last_page_len = g.decode_kv_last_page_len[{seq_idx}];
                int total = ((np - 1) * iters_per_page) +
                            (last_page_len + kv_block_size - 1) / kv_block_size;
                max_total_blocks = max(max_total_blocks, total);
            }

            int used_stages = min(max_total_blocks, NUM_STAGES);
            int first_unused_page = used_stages * PAGES_PER_STAGE;

            // Release unused pages first (lane 0 only).
            if (lid == 0) {
                for (int p = first_unused_page; p < (int)Config::NUM_PAGES; p++) {
                    int unused = s.pid(p);
                    s.wait_page_ready(unused);
                    s.finish_page(unused, Config::NUM_CONSUMER_WARPS);
                }
            }
            __syncwarp();

            int batch_block_idx = (inst.batch_block_idx * ATTN_BATCH_BLOCK_SIZE)
                                / Globals::matmul_batch_block_size;

            for (int i = 0; i < max_total_blocks; ++i) {
                int stage = i % NUM_STAGES;

                // Wait for stage to be available (lane 0 handles semaphore waits).
                if (lid == 0) {
                    if (i >= NUM_STAGES) {
                        wait(K_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                        wait(V_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                    } else {
                        wait_KV_page(s, stage);
                    }
                }
                __syncwarp();

                // Wait for QKV write to KV cache (lane 0 spins).
                if (i == 0 && lid == 0) {
                    while (*(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_QKV_RopeAppend - 1,
                           batch_block_idx, Globals::num_attention_heads + inst.kv_head_idx}] < 1)
                        __nanosleep(20);
                }
                __syncwarp();

                // Load K from paged KV cache for each batch item.
                // Pre-computed per-item indptr_start values would be better,
                // but for ATTN_BATCH_BLOCK_SIZE <= 8 the overhead is minimal.
                for (int b = 0; b < ATTN_BATCH_BLOCK_SIZE; b++) {
                    int seq_idx = first_batch_idx + b;
                    int indptr_start_b = g.decode_kv_indptr[{seq_idx}];
                    int indptr_end_b   = g.decode_kv_indptr[{seq_idx + 1}];
                    int np_b = indptr_end_b - indptr_start_b;
                    int last_page_len_b = g.decode_kv_last_page_len[{seq_idx}];
                    int total_b = ((np_b - 1) * iters_per_page) +
                                  (last_page_len_b + kv_block_size - 1) / kv_block_size;
                    if (i < total_b) {
                        int kv_page_index = g.decode_kv_indices[{indptr_start_b + (i / iters_per_page)}];
                        int iter_in_page = i % iters_per_page;
                        warp::load_async<1, false>(get_K_smem(s, stage, b), g.k_cache,
                                         {(int)g.num_pages * inst.layer_idx + kv_page_index,
                                          iter_in_page, inst.kv_head_idx, 0});
                    }
                    // If i >= total_b, this item has no more KV blocks — smem stays zero/stale
                    // but the consumer will mask it out via seq_len.
                }

                // Wait for V barrier (lane 0 spins).
                if (i == 0 && lid == 0) {
                    while (*(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_QKV_RopeAppend - 1,
                           batch_block_idx, (int)Globals::num_attention_heads
                           + (int)Globals::num_kv_heads + inst.kv_head_idx}] < 1)
                        __nanosleep(20);
                }
                __syncwarp();

                // Load V from paged KV cache for each batch item.
                for (int b = 0; b < ATTN_BATCH_BLOCK_SIZE; b++) {
                    int seq_idx = first_batch_idx + b;
                    int indptr_start_b = g.decode_kv_indptr[{seq_idx}];
                    int indptr_end_b   = g.decode_kv_indptr[{seq_idx + 1}];
                    int np_b = indptr_end_b - indptr_start_b;
                    int last_page_len_b = g.decode_kv_last_page_len[{seq_idx}];
                    int total_b = ((np_b - 1) * iters_per_page) +
                                  (last_page_len_b + kv_block_size - 1) / kv_block_size;
                    if (i < total_b) {
                        int kv_page_index = g.decode_kv_indices[{indptr_start_b + (i / iters_per_page)}];
                        int iter_in_page = i % iters_per_page;
                        warp::load_async<1, false>(get_V_smem(s, stage, b), g.v_cache,
                                         {(int)g.num_pages * inst.layer_idx + kv_page_index,
                                          iter_in_page, inst.kv_head_idx, 0});
                    }
                }

                // warp::load_async already commits; wait for all to complete.
                asm volatile("cp.async.wait_all;\n" ::: "memory");
                __syncwarp();

                // Signal consumers (lane 0).
                if (lid == 0) {
                    arrive(K_arrived(s, stage));
                    arrive(V_arrived(s, stage));
                }
            }
        }
    };

    // ── launcher: no-op on sm89 ──────────────────────────────────────────────
    struct launcher {
        static __device__ void run(const Globals &g, state<Config> &s) {
#ifdef KITTENS_BLACKWELL
            if (warp::laneid() == 0) {
                s.wait_tensor_ready();
                arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
            }
#endif
        }
    };

    // ── consumer: same flash-attention math as Hopper (already warp-scope mma) ──
    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            int wid = group<Config::NUM_CONSUMER_WARPS>::warpid();

            if (wid < ATTN_BATCH_BLOCK_SIZE) {
                int batch_idx      = inst.batch_block_idx * ATTN_BATCH_BLOCK_SIZE + wid;
                int batch_block_idx = batch_idx / Globals::matmul_batch_block_size;
                int q_head_start   = inst.kv_head_idx * GQA_RATIO;

                // Wait for all GQA heads to be written.
                for (int i = 0; i < GQA_RATIO; i++) {
                    while (*(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_QKV_RopeAppend - 1,
                           batch_block_idx, inst.kv_head_idx * GQA_RATIO + i}] < 1)
                        __nanosleep(20);
                }

                q_st &Q_smem = get_Q_smem(s, wid);
                load_Q_async(Q_smem, g.q_post_rope, batch_idx, q_head_start);

                q_rt  Q_reg;  o_rt O_reg;
                k_rt  K_reg;  v_rt V_reg;
                attn_fl_rt attn_fl;  attn_bf_rt attn_bf;
                max_vec_rv max_vec, scaled_max, last_scaled_max, diff_scaled_max;
                norm_vec_rv norm_vec;

                warp::neg_infty(max_vec);
                warp::zero(last_scaled_max);
                warp::zero(norm_vec);
                warp::zero(O_reg);

                float softmax_temp = g.attn_scale * 1.44269504089f;

                warp::load_async_wait();
                warp::load(Q_reg, Q_smem);

                // Compute sequence length from paged KV metadata.
                int seq_idx = inst.batch_block_idx * ATTN_BATCH_BLOCK_SIZE + wid;
                int indptr_start = g.decode_kv_indptr[{seq_idx}];
                int indptr_end   = g.decode_kv_indptr[{seq_idx + 1}];
                int num_kv_pages = indptr_end - indptr_start;
                int last_page_len = g.decode_kv_last_page_len[{seq_idx}];
                int seq_len    = (num_kv_pages - 1) * Globals::kv_page_size + last_page_len;
                int total_blks = ((num_kv_pages - 1) * iters_per_page) +
                                 (last_page_len + kv_block_size - 1) / kv_block_size;

                for (int i = 0; i < total_blks; i++) {
                    int stage = i % NUM_STAGES;
                    kv_st &K_smem = get_K_smem(s, stage, wid);
                    kv_st &V_smem = get_V_smem(s, stage, wid);

                    warp::zero(attn_fl);
                    warp::wait(K_arrived(s, stage), (i / NUM_STAGES) % 2);
                    warp::load(K_reg, K_smem);
                    warp::mma_ABt(attn_fl, Q_reg, K_reg, attn_fl);
                    warp::sync();
                    warp::arrive(K_finished(s, stage));

                    if ((i + 1) * kv_block_size > seq_len)
                        right_fill(attn_fl, attn_fl, seq_len % kv_block_size, -999999999999.f);

                    warp::row_max(max_vec, attn_fl, max_vec);
                    warp::mul(attn_fl, attn_fl, softmax_temp);
                    warp::mul(scaled_max, max_vec, softmax_temp);
                    warp::sub_row(attn_fl, attn_fl, scaled_max);
                    warp::exp2(attn_fl, attn_fl);
                    warp::sub(diff_scaled_max, last_scaled_max, scaled_max);
                    warp::exp2(diff_scaled_max, diff_scaled_max);
                    warp::mul_row(O_reg, O_reg, diff_scaled_max);

                    warp::wait(V_arrived(s, stage), (i / NUM_STAGES) % 2);
                    warp::load(V_reg, V_smem);
                    warp::copy(attn_bf, attn_fl);
                    warp::mma_AB(O_reg, attn_bf, V_reg, O_reg);
                    warp::sync();
                    warp::arrive(V_finished(s, stage));

                    warp::mul(norm_vec, norm_vec, diff_scaled_max);
                    warp::row_sum(norm_vec, attn_fl, norm_vec);
                    warp::copy(last_scaled_max, scaled_max);

                    if (total_blks - i <= NUM_STAGES && warp::laneid() == 0)
                        finish_KV_page(s, stage);
                }

                warp::div_row(O_reg, O_reg, norm_vec);
                o_rt_bf O_bf;
                warp::copy(O_bf, O_reg);
                o_sv (&O_smem)[4] = get_O_smem(s, wid);
                store_4_rows(O_smem, O_bf);

                warp::sync();
                warp::arrive(O_arrived(s));
            }
        }
    };

    // ── storer ───────────────────────────────────────────────────────────────
    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            int lid           = warp::laneid();
            int q_head_start  = inst.kv_head_idx * GQA_RATIO;

            wait(O_arrived(s), 0);

            // Store O_smem to global memory using raw stores.
            // Each lane handles a portion of the data across all (batch, head) pairs.
            for (int batch_in_block = 0; batch_in_block < ATTN_BATCH_BLOCK_SIZE; batch_in_block++) {
                o_sv (&O_smem)[4] = get_O_smem(s, batch_in_block);
                int out_batch = inst.batch_block_idx * ATTN_BATCH_BLOCK_SIZE + batch_in_block;
                for (int head_in_group = 0; head_in_group < GQA_RATIO; head_in_group++) {
                    int out_head = q_head_start + head_in_group;
                    // Raw store: each lane writes 2 bf16 values in a loop
                    auto *dst = (bf16*)&g.attn_out[coord<>{out_batch, out_head * head_dim}];
                    auto *src = (bf16*)&O_smem[head_in_group].data[0];
                    for (int i = lid; i < head_dim; i += 32) {
                        dst[i] = src[i];
                    }
                }
            }

            __syncwarp();
            asm volatile("{fence.acq_rel.gpu;}");

            if (lid == 0) {
                int bb = (inst.batch_block_idx * ATTN_BATCH_BLOCK_SIZE) / Globals::matmul_batch_block_size;
                atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, bb, 0}], ATTN_BATCH_BLOCK_SIZE);
            }
        }
    };
};

} // namespace kittens::prototype::vm
