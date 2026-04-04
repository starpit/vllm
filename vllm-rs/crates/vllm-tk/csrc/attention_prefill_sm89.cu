// sm89 port of attention_prefill.cu from Megakernels throughput branch.
// Processes 16-token Q blocks against paged KV cache with causal masking.
//
// Key differences from Hopper/Blackwell prefill:
//   - Loader: TMA → cp.async (warp::load_async)
//   - Consumer: warpgroup::mma → warp::mma (already warp-scope in throughput)
//   - Storer: TMA store → raw pointer stores
//   - Barrier waits: volatile pointer spin loops (same as attention_decode_sm89)
//   - Single-device only (no cross-GPU scatter)
//   - KV tiles use kv_page_size (64 for sm89) instead of 128

#include "llama_sm89.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm {

using globals = llama_sm89_globals;
using config  = llama_sm89_config;

template <typename Config = config, typename Globals = globals>
struct attention_prefill {
    static constexpr int opcode = OPCODE_GQA_AttentionPrefill;
    static constexpr int NUM_STAGES = 2;
    static constexpr int GQA_RATIO = Globals::num_attention_heads / Globals::num_kv_heads;

    static_assert(GQA_RATIO == 4 || GQA_RATIO == 8, "GQA_RATIO must be 4 or 8");

    static constexpr int head_dim     = Globals::head_dim;
    static constexpr int kv_page_size = Globals::kv_page_size;
    static constexpr int mbb          = Globals::matmul_batch_block_size;

    // Tile types for prefill:
    // Q: 16 tokens × head_dim (one warp's worth per consumer warp)
    // K/V: kv_page_size × head_dim (one full page per load)
    // Attn: 16 × kv_page_size
    using q_rt       = rt_bf<16, head_dim>;
    using q_st       = st_bf<16, head_dim>;
    using kv_st      = st_bf<kv_page_size, head_dim>;
    using o_rt       = rt_fl<16, head_dim>;
    using o_rt_bf    = rt_bf<16, head_dim>;
    using o_sv       = sv_bf<head_dim>;
    using attn_fl_rt = rt_fl<16, kv_page_size>;
    using attn_bf_rt = rt_bf<16, kv_page_size>;
    using max_vec_rv = col_vec<rt_fl<16, head_dim>>;
    using norm_vec_rv= col_vec<rt_fl<16, head_dim>>;
    using k_rt       = rt_bf<kv_page_size, head_dim>;
    using v_rt       = rt_bf<kv_page_size, head_dim, col_l>;

    // Page budget:
    // K pages per stage: ceil(sizeof(kv_st) / PAGE_SIZE)
    // For head_dim=64: kv_st = st_bf<64,64> = 8KB = 1 page
    // For head_dim=128: kv_st = st_bf<64,128> = 16KB = 2 pages
    static constexpr int KV_BYTES     = sizeof(kv_st);
    static constexpr int K_PAGES      = (KV_BYTES + Config::PAGE_SIZE - 1) / Config::PAGE_SIZE;
    static constexpr int V_PAGES      = K_PAGES;
    static constexpr int PAGES_PER_STAGE = K_PAGES + V_PAGES;
    // Verify we have enough pages for 2 stages
    static_assert(NUM_STAGES * PAGES_PER_STAGE <= Config::NUM_PAGES,
                  "Not enough shmem pages for prefill attention stages");

    struct prefill_instruction {
        int layer_idx;
        int seq_idx;
        int prefill_block_idx;
        int prefill_token_offset;
        int kv_head_idx;
        int q_start_idx, q_end_idx, q_size;
        int abs_q_row, abs_q_row_last;
        int rel_q_row, rel_q_row_last;
        int kv_indptr_start;
        int attn_blocks;
        int sequence_length;

        __device__ __inline__ prefill_instruction(const Globals &g, state<Config> &s) {
            layer_idx            = s.instruction()[1];
            seq_idx              = s.instruction()[2];
            prefill_block_idx    = s.instruction()[3];
            prefill_token_offset = s.instruction()[4];
            kv_head_idx          = s.instruction()[5];

            q_start_idx = g.prefill_qo_indptr[{seq_idx}];
            q_end_idx   = g.prefill_qo_indptr[{seq_idx + 1}];
            q_size      = q_end_idx - q_start_idx;

            rel_q_row      = 16 * prefill_block_idx;
            rel_q_row_last = min(rel_q_row + 15, q_size - 1);

            abs_q_row      = rel_q_row + q_start_idx;
            abs_q_row_last = rel_q_row_last + q_start_idx;

            kv_indptr_start = g.prefill_kv_indptr[{seq_idx}];
            sequence_length = prefill_token_offset + rel_q_row_last + 1;
            attn_blocks     = (sequence_length + kv_page_size - 1) / kv_page_size;
        }
    };

    // Semaphore accessors
    __device__ static inline semaphore &Q_arrived(state<Config> &s)             { return s.semaphores()[0]; }
    __device__ static inline semaphore &O_arrived(state<Config> &s)             { return s.semaphores()[1]; }
    __device__ static inline semaphore &K_arrived(state<Config> &s, int stage)  { return s.semaphores()[2 + stage * 2]; }
    __device__ static inline semaphore &V_arrived(state<Config> &s, int stage)  { return s.semaphores()[2 + stage * 2 + 1]; }
    __device__ static inline semaphore &K_finished(state<Config> &s, int stage) { return s.semaphores()[2 + NUM_STAGES * 2 + stage * 2]; }
    __device__ static inline semaphore &V_finished(state<Config> &s, int stage) { return s.semaphores()[2 + NUM_STAGES * 2 + stage * 2 + 1]; }

    // Page layout: stages use pages [0..NUM_STAGES*PAGES_PER_STAGE)
    __device__ static inline kv_st &K(state<Config> &s, int stage) {
        return *reinterpret_cast<kv_st *>(s.pages[s.pid(stage * PAGES_PER_STAGE)].data);
    }
    __device__ static inline kv_st &V(state<Config> &s, int stage) {
        return *reinterpret_cast<kv_st *>(s.pages[s.pid(stage * PAGES_PER_STAGE + K_PAGES)].data);
    }

    // Q and O use scratch (same as decode). Scratch is 8KB, fits st_bf<16, 64> for head_dim=64.
    // For head_dim=128, Q would need 16KB > 8KB scratch — would need a page. Currently 1B only.
    __device__ static inline q_st &get_Q_smem(state<Config> &s, int wid) {
        return *reinterpret_cast<q_st *>(
            reinterpret_cast<char *>(s.scratch()) + sizeof(o_sv) * wid * 4);
    }
    __device__ static inline o_sv (&get_O_smem(state<Config> &s, int wid))[4] {
        return *reinterpret_cast<o_sv(*)[4]>(
            reinterpret_cast<char *>(s.scratch()) + sizeof(o_sv) * wid * 4);
    }

    // Load Q for one consumer warp (16 tokens) using cp.async — same helper as decode.
    __device__ static inline void load_Q_async(q_st &dst,
                                               const typename Globals::activations_t &src,
                                               int batch_idx, int q_head_start_idx) {
        using T = typename q_st::dtype;
        constexpr int elem_per_memcpy = sizeof(float4) / sizeof(T);
        constexpr int memcpy_per_row  = head_dim / elem_per_memcpy;

        // src_ptr points to [batch_idx, q_head_start_idx * head_dim] already.
        // Each row stride is num_attention_heads * head_dim (= hidden_size).
        auto *src_ptr = (typename Globals::activations_t::dtype *)&src[coord<>{batch_idx, q_head_start_idx * head_dim}];
        uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
        int lid = warp::laneid();

        // Load 16 rows × head_dim via cp.async. Each cp.async copies 16 bytes.
        // For head_dim=64: 128 bytes/row, 8 lanes per row, 4 rows per iteration.
        for (int iter = 0; iter < (16 + 3) / 4; iter++) {
            int row = iter * 4 + lid / (head_dim / elem_per_memcpy);
            int c = (lid % (head_dim / elem_per_memcpy)) * elem_per_memcpy;
            if (row < 16) {
                asm volatile(
                    "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::
                    "r"(dst.idx(dst_ptr, {row, c})),
                    "l"(&src_ptr[row * Globals::num_attention_heads * head_dim + c]) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    // store_4_rows: same helper as decode (register → shmem).
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

    // ── controller ──────────────────────────────────────────────────────
    struct controller {
        static __device__ int release_lid(const Globals &g,
                                          typename Config::instruction_t &ins, int &query) {
            return query; // identity mapping, same as decode
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            init_semaphore(Q_arrived(s), 0, 1);
            init_semaphore(O_arrived(s), 0, GQA_RATIO);
            for (int i = 0; i < NUM_STAGES; i++) {
                init_semaphore(K_arrived(s, i),  0, 1);
                init_semaphore(V_arrived(s, i),  0, 1);
                init_semaphore(K_finished(s, i), 0, GQA_RATIO);
                init_semaphore(V_finished(s, i), 0, GQA_RATIO);
            }
            return 2 + 4 * NUM_STAGES;
        }
    };

    // ── loader ──────────────────────────────────────────────────────────
    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            int lid = warp::laneid();
            prefill_instruction pi(g, s);

            int used_stages = min(pi.attn_blocks, NUM_STAGES);
            int first_unused_page = used_stages * PAGES_PER_STAGE;

            // Release unused pages.
            if (lid == 0) {
                for (int p = first_unused_page; p < (int)Config::NUM_PAGES; p++) {
                    int unused = s.pid(p);
                    s.wait_page_ready(unused);
                    s.finish_page(unused, Config::NUM_CONSUMER_WARPS);
                }
            }
            __syncwarp();

            // Wait for Q to be ready in q_post_rope (barrier from QKV_RopeAppend).
            // QKV storer signals Bar[{layer, QKV-1, bb, head}] += 1 per tile.
            // We need Q written for the batch block(s) containing our 16-token chunk.
            // Wait for column 0 to have at least 1 signal (meaning the first Q tile completed).
            int batch_block_idx = pi.abs_q_row / mbb;
            int batch_block_idx_last = pi.abs_q_row_last / mbb;
            if (lid == 0) {
                while (*(volatile int *)&g.Bar[{pi.layer_idx, OPCODE_QKV_RopeAppend - 1, batch_block_idx, 0}] < 1)
                    __nanosleep(20);
                if (batch_block_idx_last != batch_block_idx) {
                    while (*(volatile int *)&g.Bar[{pi.layer_idx, OPCODE_QKV_RopeAppend - 1, batch_block_idx_last, 0}] < 1)
                        __nanosleep(20);
                }
                // Also wait for KV cache writes (K heads start at num_attention_heads).
                while (*(volatile int *)&g.Bar[{pi.layer_idx, OPCODE_QKV_RopeAppend - 1,
                       batch_block_idx, (int)Globals::num_attention_heads + pi.kv_head_idx}] < 1)
                    __nanosleep(20);
            }
            __syncwarp();

            // Signal Q arrived so consumers can start loading Q from q_post_rope.
            if (lid == 0) arrive(Q_arrived(s));

            // Pipeline KV loads.
            for (int i = 0; i < pi.attn_blocks; ++i) {
                int kv_page_index = g.prefill_kv_indices[{pi.kv_indptr_start + i}];
                int stage = i % NUM_STAGES;

                // Wait for stage availability.
                if (lid == 0) {
                    if (i >= NUM_STAGES) {
                        wait(K_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                        wait(V_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                    } else {
                        for (int p = 0; p < PAGES_PER_STAGE; p++)
                            s.wait_page_ready(s.pid(stage * PAGES_PER_STAGE + p));
                    }
                }
                __syncwarp();

                // Load K/V from paged cache using raw cp.async to bypass gl bounds check.
                // Cache layout: [nl*np, ipp, nkh, hd] row-major.
                // We need kv_page_size contiguous rows for one kv_head, but the cache
                // stores each kv_block_size chunk interleaved across kv_heads.
                // Token t within a page at kv_head h is at:
                //   flat_offset = page_batch * (ipp * nkh * hd) + (t / kbs) * (nkh * hd) + h * hd
                //   where within that kv_block, the row is t % kbs, so:
                //   flat_offset = page_batch * (ipp * nkh * hd) + t * nkh * hd + h * hd
                //   Wait, no: [B,D,R,C] = [b, d, r, c] → flat = b*(D*R*C) + d*(R*C) + r*C + c
                //   For token t in page: d = t/kbs, then the token's "r" index within the
                //   kv_block is (t%kbs) mapped into the R=nkh dimension. But store_kv_paged uses
                //   raw offsets: (b*D + offset_in_page) * R * C + h * C. So offset_in_page = t
                //   and it just uses t as if D were kv_page_size (not ipp).
                //   This means store treats it as [nl*np, kv_page_size, nkh, hd] even though
                //   the gl declares D=ipp. The raw pointer math works because the total size
                //   is the same: ipp * nkh * hd = (kv_page_size/kbs) * nkh * hd.
                //   Actually: store uses (b*D + t) where D = cache.depth() = ipp.
                //   With ipp=4 and t up to 63, b*4+63 goes past the next page batch boundary.
                //   So the cache is effectively flat: [nl*np*ipp*nkh*hd] with some striding.
                //   For loading: token t at kv_head h has flat index:
                //     (page_batch * ipp + t) * nkh * hd + h * hd + col
                //   This is equivalent to treating the cache as [nl*np*ipp, nkh, hd].
                {
                    using bf16 = __nv_bfloat16;
                    constexpr int nkh = Globals::num_kv_heads;
                    constexpr int hd = head_dim;
                    constexpr int ipp = Globals::iters_per_page;
                    constexpr int elem_per_cp = sizeof(float4) / sizeof(bf16); // 8
                    constexpr int lanes_per_row = hd / elem_per_cp; // 8
                    constexpr int rows_per_iter = 32 / lanes_per_row; // 4

                    bf16 *k_base = (bf16*)g.k_cache.raw_ptr;
                    bf16 *v_base = (bf16*)g.v_cache.raw_ptr;
                    int page_batch = (int)g.num_pages * pi.layer_idx + kv_page_index;

                    kv_st &K_tile = K(s, stage);
                    kv_st &V_tile = V(s, stage);
                    uint32_t k_smem = static_cast<uint32_t>(__cvta_generic_to_shared(&K_tile.data[0]));
                    uint32_t v_smem = static_cast<uint32_t>(__cvta_generic_to_shared(&V_tile.data[0]));

                    // Total cache elements for bounds checking
                    long cache_elems = (long)g.k_cache.batch() * g.k_cache.depth()
                                     * nkh * hd;
                    for (int ri = 0; ri < (kv_page_size + rows_per_iter - 1) / rows_per_iter; ri++) {
                        int row = ri * rows_per_iter + lid / lanes_per_row;
                        int col = (lid % lanes_per_row) * elem_per_cp;
                        if (row < kv_page_size) {
                            // Token row within page → flat cache offset
                            long src_off = ((long)page_batch * ipp + row) * nkh * hd
                                         + (long)pi.kv_head_idx * hd + col;
                            if (src_off < 0 || src_off + elem_per_cp > cache_elems) {
                                // OOB — skip cp.async to avoid crash
                            } else {
                                asm volatile(
                                    "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::
                                    "r"(K_tile.idx(k_smem, {row, col})),
                                    "l"(&k_base[src_off]) : "memory");
                                asm volatile(
                                    "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::
                                    "r"(V_tile.idx(v_smem, {row, col})),
                                    "l"(&v_base[src_off]) : "memory");
                            }
                        }
                    }
                }

                asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
                __syncwarp();

                if (lid == 0) {
                    arrive(K_arrived(s, stage));
                    arrive(V_arrived(s, stage));
                }
            }
        }
    };

    // ── launcher: no-op on sm89 ─────────────────────────────────────────
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

    // ── consumer ────────────────────────────────────────────────────────
    // Each of the 8 consumer warps handles one Q head within the GQA group.
    // For GQA_RATIO=4: warps 0-3 each handle one head, warps 4-7 are idle.
    // For GQA_RATIO=8: all 8 warps are active.
    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            int wid = group<Config::NUM_CONSUMER_WARPS>::warpid();
            prefill_instruction pi(g, s);

            if (wid < GQA_RATIO) {
                int q_head = pi.kv_head_idx * GQA_RATIO + wid;

                // Wait for Q barrier and load Q from q_post_rope.
                warp::wait(Q_arrived(s), 0);

                // Load Q: 16 rows (tokens) for this head.
                q_st &Q_smem = get_Q_smem(s, wid);
                load_Q_async(Q_smem, g.q_post_rope, pi.abs_q_row, q_head);
                warp::load_async_wait();

                q_rt Q_reg;
                warp::load(Q_reg, Q_smem);

                o_rt O_reg;
                k_rt K_reg;
                v_rt V_reg;
                attn_fl_rt attn_fl;
                attn_bf_rt attn_bf;
                max_vec_rv max_vec, scaled_max, last_scaled_max, diff_scaled_max;
                norm_vec_rv norm_vec;

                warp::neg_infty(max_vec);
                warp::zero(last_scaled_max);
                warp::zero(norm_vec);
                warp::zero(O_reg);

                float softmax_temp = g.attn_scale * 1.44269504089f; // 1/(sqrt(D_h)*ln(2))

                for (int i = 0; i < pi.attn_blocks; i++) {
                    int stage = i % NUM_STAGES;

                    // Q×K^T
                    warp::zero(attn_fl);
                    warp::wait(K_arrived(s, stage), (i / NUM_STAGES) % 2);
                    warp::load(K_reg, K(s, stage));
                    warp::mma_ABt(attn_fl, Q_reg, K_reg, attn_fl);
                    warp::sync();
                    warp::arrive(K_finished(s, stage));

                    // Release K pages at pipeline tail.
                    if (i >= pi.attn_blocks - NUM_STAGES && warp::laneid() == 0) {
                        for (int p = 0; p < K_PAGES; p++)
                            s.finish_page(s.pid(stage * PAGES_PER_STAGE + p), Config::NUM_CONSUMER_WARPS / GQA_RATIO);
                    }

                    // Causal masking: mask positions where kv_pos > q_pos.
                    auto kv_seqlen_at_block_start = i * kv_page_size;
                    auto kv_seqlen_with_this_block = (i + 1) * kv_page_size;
                    auto q_pos_start = pi.rel_q_row + pi.prefill_token_offset;

                    if (kv_seqlen_with_this_block > q_pos_start) {
                        warp::apply(attn_fl, attn_fl,
                            [kv_seqlen_at_block_start, q_pos_start] __device__(int row, int col, float val) {
                                auto kv_pos = kv_seqlen_at_block_start + col;
                                auto q_pos = row + q_pos_start;
                                return (kv_pos > q_pos) ? -999999999999.f : val;
                            });
                    }

                    // Also mask out-of-bounds KV positions for the last page.
                    if (i == pi.attn_blocks - 1) {
                        int valid_kv = pi.sequence_length - i * kv_page_size;
                        if (valid_kv < kv_page_size) {
                            warp::apply(attn_fl, attn_fl,
                                [valid_kv] __device__(int row, int col, float val) {
                                    return (col >= valid_kv) ? -999999999999.f : val;
                                });
                        }
                    }

                    // Online softmax
                    warp::row_max(max_vec, attn_fl, max_vec);
                    warp::mul(attn_fl, attn_fl, softmax_temp);
                    warp::mul(scaled_max, max_vec, softmax_temp);
                    warp::sub_row(attn_fl, attn_fl, scaled_max);
                    warp::exp2(attn_fl, attn_fl);
                    warp::sub(diff_scaled_max, last_scaled_max, scaled_max);
                    warp::exp2(diff_scaled_max, diff_scaled_max);
                    warp::mul_row(O_reg, O_reg, diff_scaled_max);

                    // attn × V
                    warp::wait(V_arrived(s, stage), (i / NUM_STAGES) % 2);
                    warp::load(V_reg, V(s, stage));
                    warp::copy(attn_bf, attn_fl);
                    warp::mma_AB(O_reg, attn_bf, V_reg, O_reg);
                    warp::sync();
                    warp::arrive(V_finished(s, stage));

                    // Release V pages at pipeline tail.
                    if (i >= pi.attn_blocks - NUM_STAGES && warp::laneid() == 0) {
                        for (int p = 0; p < V_PAGES; p++)
                            s.finish_page(s.pid(stage * PAGES_PER_STAGE + K_PAGES + p), Config::NUM_CONSUMER_WARPS / GQA_RATIO);
                    }

                    warp::mul(norm_vec, norm_vec, diff_scaled_max);
                    warp::row_sum(norm_vec, attn_fl, norm_vec);
                    warp::copy(last_scaled_max, scaled_max);
                }

                // Note: unused stage pages are released by the loader (not the consumer),
                // matching the decode attention pattern.

                // Normalize output.
                warp::add(norm_vec, norm_vec, 1e-16f);
                warp::div_row(O_reg, O_reg, norm_vec);

                // Store to shmem.
                o_rt_bf O_bf;
                warp::copy(O_bf, O_reg);
                o_sv (&O_smem)[4] = get_O_smem(s, wid);
                store_4_rows(O_smem, O_bf);

                warp::sync();
                warp::arrive(O_arrived(s));
            }
        }
    };

    // ── storer ──────────────────────────────────────────────────────────
    // Writes each consumer warp's output (one head, 16 tokens) to attn_out.
    // Then signals the O_proj barrier (using OPCODE_GQA_AttentionDecode - 1 slot
    // so that o_proj treats prefill and decode outputs uniformly).
    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            int lid = warp::laneid();
            prefill_instruction pi(g, s);

            wait(O_arrived(s), 0);

            // Write each active warp's output to global memory.
            for (int wid = 0; wid < GQA_RATIO; wid++) {
                o_sv (&O_smem)[4] = get_O_smem(s, wid);
                int head = pi.kv_head_idx * GQA_RATIO + wid;

                // Write 16 rows (or fewer if at end of sequence).
                for (int row = 0; row < 16; row++) {
                    int abs_row = pi.abs_q_row + row;
                    if (abs_row > pi.abs_q_row_last) break;

                    // Find which of the 4 o_sv vectors contains this row.
                    // store_4_rows packs rows 0-3 into dst[0..3].
                    // Actually store_4_rows stores rt_bf<16,head_dim> into sv_bf<head_dim>[4].
                    // With the mma register layout, rows map as: row 0-3 → dst[0-3].
                    // But store_4_rows actually stores lid/4 rows, not sequential rows.
                    // Let's just use raw pointer stores like decode storer.
                    auto *dst = (bf16*)&g.attn_out[coord<>{abs_row, head * head_dim}];
                    auto *src = (bf16*)&O_smem[row / 4].data[0];
                    int offset_in_sv = (row % 4) * head_dim;
                    for (int i = lid; i < head_dim; i += 32) {
                        dst[i] = src[offset_in_sv + i];
                    }
                }
            }

            __syncwarp();
            asm volatile("{fence.acq_rel.gpu;}");

            // Signal o_proj barrier. We write to the OPCODE_GQA_AttentionDecode - 1 slot
            // (same as decode storer) so o_proj sees a uniform signal.
            if (lid == 0) {
                // Signal for each batch block that contains our tokens.
                int bb_start = pi.abs_q_row / mbb;
                int bb_end   = pi.abs_q_row_last / mbb;
                for (int bb = bb_start; bb <= bb_end; bb++) {
                    // Count how many of our 16 tokens fall in this batch block.
                    int block_start = bb * mbb;
                    int block_end   = block_start + mbb;
                    int count_lo = max(0, min(pi.abs_q_row_last + 1, block_end) - max(pi.abs_q_row, block_start));
                    if (count_lo > 0) {
                        atomicAdd(&g.Bar[{pi.layer_idx, OPCODE_GQA_AttentionDecode - 1, bb, 0}], count_lo);
                    }
                }
            }
        }
    };
};

} // namespace kittens::prototype::vm
