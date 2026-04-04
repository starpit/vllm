// sm89 port of qkv_rope_append.cu
// QKV projection + RoPE rotation + KV cache write.
// Changes vs Hopper:
//   - matmul_pipeline → matmul_pipeline_sm89
//   - rope_cos/sin loaded with cp.async into a page (not scratch+TMA)
//   - consumer uses rt_fl register acc (no tensor_alloc)
//   - storer uses warp::store (no warp::tma::store_async)

#include "llama_sm89.cuh"
#include "matmul_pipeline_sm89.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm {

using globals = llama_sm89_globals;
using config  = llama_sm89_config;

struct qkv_gmem_waiter_sm89;

// Store a head_dim-wide tile into the paged KV cache.
// For each of the 16 rows in the tile, look up the token's append_index
// from g.kv_append_indices to find the page and offset within that page.
// Cache layout: [num_layers * num_pages, page_size, num_kv_heads, head_dim]
template<typename Globals>
__device__ static inline void store_kv_paged(
        const Globals &g,
        const rt_bf<16, Globals::head_dim> &tile,
        int global_row,   // first batch row for this warp
        int layer, int kv_head_idx,
        bool is_k)  // true for K cache, false for V cache
{
    int lane = laneid();
    int r_lo = lane / 4;
    int r_hi = lane / 4 + 8;

    int token_lo = global_row + r_lo;
    int token_hi = global_row + r_hi;

    // Look up paged cache coordinates for each token.
    int append_idx_lo = g.kv_append_indices[{token_lo}];
    int append_idx_hi = g.kv_append_indices[{token_hi}];

    int page_idx_lo = append_idx_lo / Globals::kv_page_size;
    int offset_lo   = append_idx_lo % Globals::kv_page_size;
    int page_idx_hi = append_idx_hi / Globals::kv_page_size;
    int offset_hi   = append_idx_hi % Globals::kv_page_size;

    // Cache address: cache[num_pages * layer + page_idx, offset, kv_head, col]
    auto &cache = is_k ? g.k_cache : g.v_cache;
    __nv_bfloat16 *cache_ptr = (__nv_bfloat16 *)cache.raw_ptr;

    // Compute strides: [D0, D1, num_kv_heads, head_dim]
    // D1 = depth dim = page_size / kv_block_size (but for raw-ptr store we use page_size directly)
    // Actually, cache gl is [num_layers*num_pages, kv_page_size, num_kv_heads, head_dim] (depth=-1, rows=nkh, cols=hdm)
    // With gl<bf16, -1, -1, num_kv_heads, head_dim>, stride per batch = depth * rows * cols
    long stride_d0 = (long)cache.depth() * Globals::num_kv_heads * Globals::head_dim;
    long stride_d1 = (long)Globals::num_kv_heads * Globals::head_dim;
    long stride_kv_head = (long)Globals::head_dim;

    long base_lo_addr = (long)((int)g.num_pages * layer + page_idx_lo) * stride_d0
                      + (long)offset_lo * stride_kv_head  // Wait — offset is within the page
                      + (long)kv_head_idx * stride_kv_head;
    // Actually let me reconsider the cache layout.
    // gl<bf16, -1, -1, num_kv_heads, head_dim> means dims are [B, D, R, C]
    // where B = num_layers * num_pages, D = page_size / kv_block_size (iters_per_page),
    //       R = num_kv_heads, C = head_dim
    // But for paged append, offset_in_page is in units of TOKENS, not kv_blocks.
    // The throughput branch stores with: {num_pages * layer + page_idx, offset_in_page, kv_head, 0}
    // where offset_in_page = append_idx % kv_page_size
    // So each "depth" slot in the gl is one token position within the page.
    // That means D = kv_page_size (not kv_page_size / kv_block_size).
    // But for DECODE reads, D = kv_page_size / kv_block_size (iters_per_page) and each
    // "depth" slot is a kv_block_size chunk...
    // The throughput branch uses kv_page_size for stores but iters_per_page for loads.
    // The kv_st for loads is st_bf<kv_block_size, head_dim>, so each load gets kv_block_size rows.
    // For stores, it's one token at a time (sv_bf<head_dim>).
    //
    // Let me just use the gl indexing directly via store to be safe.
    // Unfortunately we have rt_bf<16, head_dim> which is 16 rows, but each row is a different token.
    // We need per-token stores. Convert to sv_bf<head_dim> and use warp::store per row.
    // That's expensive but correct. For sm89 without TMA this is the only option.

    // Actually — let's just do raw pointer math. The cache is row-major:
    // element at [b, d, r, c] is at offset b*(D*R*C) + d*(R*C) + r*C + c
    // where D = cache.depth(), R = num_kv_heads, C = head_dim
    long D = cache.depth();
    long R = Globals::num_kv_heads;
    long C = Globals::head_dim;

    long addr_lo = ((long)((int)g.num_pages * layer + page_idx_lo) * D + offset_lo) * R * C
                 + (long)kv_head_idx * C;
    long addr_hi = ((long)((int)g.num_pages * layer + page_idx_hi) * D + offset_hi) * R * C
                 + (long)kv_head_idx * C;

    using T2 = __nv_bfloat162;
    #pragma unroll
    for (int j = 0; j < tile.width; j++) {
        int c0 = j * tile.tile_size_col + 2 * (lane % 4);
        int c8 = c0 + 8;
        *(T2*)(cache_ptr + addr_lo + c0) = tile.tiles[0][j].data[0];
        *(T2*)(cache_ptr + addr_lo + c8) = tile.tiles[0][j].data[2];
        *(T2*)(cache_ptr + addr_hi + c0) = tile.tiles[0][j].data[1];
        *(T2*)(cache_ptr + addr_hi + c8) = tile.tiles[0][j].data[3];
    }
}

template <typename Config = config, typename Globals = globals>
struct qkv_rope_append {
    static constexpr int opcode        = OPCODE_QKV_RopeAppend;
    // Head-block boundaries in output-column-block units.
    // Head boundaries in output-column-block units.
    // Each col block is OUT_BLOCK cols.  heads_per_block = OUT_BLOCK / head_dim.
    // For LLaMA 1B (head_dim=64, OUT_BLOCK=128): 2 heads per block.
    //   Q: 32 heads / 2 = 16 blocks (cols 0..15), K: 8/2 = 4 (cols 16..19), V: 4 (cols 20..23)
    static constexpr int OUT_BLOCK     = Globals::qkv_out_block;
    static constexpr int HEADS_PER_BLOCK = OUT_BLOCK / Globals::head_dim;
    static_assert(OUT_BLOCK >= Globals::head_dim && OUT_BLOCK % Globals::head_dim == 0,
                  "qkv_out_block must be a multiple of head_dim");
    static constexpr int K_BLOCK_START = Globals::num_attention_heads / HEADS_PER_BLOCK;
    static constexpr int V_BLOCK_START = (Globals::num_attention_heads + Globals::num_kv_heads) / HEADS_PER_BLOCK;
    static constexpr int NUM_ITERS     = Globals::hidden_dim / SM89_PIPELINE_K_DIM;
    static constexpr int BATCH_BLOCK   = Globals::matmul_batch_block_size;

    // Rope vectors: sv_fl<head_dim> each.  Store cos+sin in one extra page (page 6 when GEMM uses 0-5).
    static constexpr int ROPE_PAGE = config::NUM_PAGES - 1;  // use last page for rope vectors

    struct parsed_instruction {
        int layer, row, col;
        __device__ inline parsed_instruction(typename Config::instruction_t &i) {
            layer = i[1]; row = i[2]; col = i[3];
        }
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    using pipeline = matmul_pipeline_sm89<Config, Globals, parsed_instruction,
                                          qkv_gmem_waiter_sm89,
                                          &Globals::rms_rope_intermediates,
                                          &Globals::qkv_weights,
                                          NUM_ITERS, OUT_BLOCK>;
    using acc_rt = typename pipeline::acc_rt;

    __device__ static inline semaphore &rope_arrived(state<Config> &s) {
        return s.semaphores()[pipeline::SEM_COUNT];
    }

    struct controller {
        static __device__ int release_lid(const Globals &g, typename Config::instruction_t &ins, int &q) {
            return pipeline::release_lid(g, ins, q);
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            init_semaphore(rope_arrived(s), 1);
            return pipeline::init_semaphores(s) + 1;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            pipeline::loader_loop(s, g, inst.layer);

            // Release matmul-unused pages (between pipeline pages and ROPE_PAGE)
            // so a following NOP instruction won't deadlock on wait_page_ready.
            pipeline::release_unused_pages(s, ROPE_PAGE);
            warp::sync();

            // Load rope cos/sin only for Q and K cols (not V).
            if (inst.col < V_BLOCK_START) {
                using rope_vec = sv_fl<Globals::head_dim>;
                int pid = s.pid(ROPE_PAGE);
                // Lane 0 waits for page to be ready; then all lanes load.
                if (laneid() == 0) s.wait_page_ready(pid);
                warp::sync();

                auto *rope_base = reinterpret_cast<char *>(s.pages[pid].data);
                rope_vec &rope_cos = *reinterpret_cast<rope_vec *>(rope_base);
                rope_vec &rope_sin = *reinterpret_cast<rope_vec *>(rope_base + sizeof(rope_vec));

                // Load rope for the first token in this batch block.
                // For decode, all tokens in the same batch block share instruction
                // context — but each has its own position. We load per-token
                // position_ids in the consumer via scratch (see consumer::run).
                // For the loader, load rope for position_ids[first_token_in_block]
                // as a representative (the consumer will re-load per-token).
                // Actually — on sm89 the loader puts ONE cos/sin pair into the rope
                // page and the consumer uses it for ALL 16 rows in the warp's tile.
                // For decode with batch_block_size tokens, each token has its own
                // position. We need to change the consumer to load per-token rope.
                // For now, load the first token's rope (the consumer overrides per-token).
                int first_token = inst.row * BATCH_BLOCK;
                int first_pos = g.position_ids[{first_token}];
                warp::load_async(rope_cos, g.rope_cos, {first_pos, 0});
                warp::load_async(rope_sin, g.rope_sin, {first_pos, 0});
                asm volatile("cp.async.commit_group;\n" ::: "memory");
                asm volatile("cp.async.wait_all;\n"     ::: "memory");
                if (laneid() == 0) arrive(rope_arrived(s));
            } else {
                // V columns don't use rope — release ROPE_PAGE so NOP won't deadlock.
                if (laneid() == 0) {
                    int pid = s.pid(ROPE_PAGE);
                    s.wait_page_ready(pid);
                    s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
                }
            }
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, state<Config> &s) {
            pipeline::launcher_loop(s, g);
        }
    };

    struct storer { static __device__ void run(const Globals &g, state<Config> &s) {} };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            // On sm89 consumers do warp::mma directly inside consumer_loop;
            // do NOT wait for outputs_arrived before entering — that would deadlock
            // because the loader needs consumers to arrive on inputs_finished first.
            acc_rt acc;
            pipeline::consumer_loop(s, g, acc, inst.layer);

            // Each consumer warp holds OUT_BLOCK output columns for
            // rows [warpid()*16 .. (warpid()+1)*16) of the QKV result.
            int global_row = inst.row * BATCH_BLOCK + warpid() * 16;
            bool is_q_block = (inst.col < K_BLOCK_START);
            bool is_k_block = (inst.col >= K_BLOCK_START) && (inst.col < V_BLOCK_START);
            bool is_v_block = (inst.col >= V_BLOCK_START);
            bool needs_rope = is_q_block || is_k_block;

            // Apply RoPE if needed.
            // For paged KV, each row in the warp's 16-row tile is a different token
            // with its own position_id. We apply rope per-row.
            // However, the mma register layout packs rows in pairs (r_lo = lane/4,
            // r_hi = lane/4 + 8), and data[k] holds float2 for (col, col+1).
            // The RoPE cos/sin values depend only on COLUMN position (not row),
            // and are the SAME for all rows with the SAME position_id.
            //
            // For decode: all tokens in a batch block typically have different positions.
            // But within a single warp tile (16 rows), the mma layout means
            // data[0] = (r_lo, cols), data[1] = (r_hi, cols).
            // r_lo and r_hi may have different position_ids!
            //
            // The simple correct approach: since we're already in f32 registers,
            // just load rope cos/sin per-position from global memory (it's only head_dim
            // floats per position, and the rope table is likely in L2).
            if (needs_rope) {
                // We don't actually need the rope page anymore — load directly from global.
                // But we still need to wait for it to maintain the semaphore protocol.
                wait(rope_arrived(s), 0);

                // For each of the 16 rows in the tile, look up its position_id.
                // Rows 0..7 are r_lo = lane/4 (for lanes 0..31), rows 8..15 are r_hi.
                // But in the mma layout, "row" is determined by which data[] slot.
                // data[0]/data[2] → r_lo = laneid/4
                // data[1]/data[3] → r_hi = laneid/4 + 8
                int lane = laneid();
                int r_lo = lane / 4;
                int r_hi = r_lo + 8;
                int token_lo = global_row + r_lo;
                int token_hi = global_row + r_hi;
                int pos_lo = g.position_ids[{token_lo}];
                int pos_hi = g.position_ids[{token_hi}];

                #pragma unroll
                for (int c = 0; c < acc_rt::width; c++) {
                    int col_in_head = (c * 16) % Globals::head_dim;
                    int col_base = col_in_head + 2 * (lane % 4);
                    int col_base8 = col_base + 8;

                    // Load cos/sin for both rows at the relevant columns.
                    float *cos_ptr = (float *)g.rope_cos.raw_ptr;
                    float *sin_ptr = (float *)g.rope_sin.raw_ptr;
                    int hdm = Globals::head_dim;

                    // data[0] = (r_lo, col_base:col_base+1)
                    // data[2] = (r_lo, col_base8:col_base8+1)
                    // data[1] = (r_hi, col_base:col_base+1)
                    // data[3] = (r_hi, col_base8:col_base8+1)

                    float cos_lo_0 = cos_ptr[pos_lo * hdm + col_base];
                    float cos_lo_1 = cos_ptr[pos_lo * hdm + col_base + 1];
                    float sin_lo_0 = sin_ptr[pos_lo * hdm + col_base];
                    float sin_lo_1 = sin_ptr[pos_lo * hdm + col_base + 1];

                    float cos_lo_8 = cos_ptr[pos_lo * hdm + col_base8];
                    float cos_lo_9 = cos_ptr[pos_lo * hdm + col_base8 + 1];
                    float sin_lo_8 = sin_ptr[pos_lo * hdm + col_base8];
                    float sin_lo_9 = sin_ptr[pos_lo * hdm + col_base8 + 1];

                    float cos_hi_0 = cos_ptr[pos_hi * hdm + col_base];
                    float cos_hi_1 = cos_ptr[pos_hi * hdm + col_base + 1];
                    float sin_hi_0 = sin_ptr[pos_hi * hdm + col_base];
                    float sin_hi_1 = sin_ptr[pos_hi * hdm + col_base + 1];

                    float cos_hi_8 = cos_ptr[pos_hi * hdm + col_base8];
                    float cos_hi_9 = cos_ptr[pos_hi * hdm + col_base8 + 1];
                    float sin_hi_8 = sin_ptr[pos_hi * hdm + col_base8];
                    float sin_hi_9 = sin_ptr[pos_hi * hdm + col_base8 + 1];

                    // Apply interleaved RoPE: x' = x*cos - y*sin, y' = x*sin + y*cos
                    auto &d0 = acc.tiles[0][c].data[0]; // (r_lo, col_base:col_base+1)
                    auto &d2 = acc.tiles[0][c].data[2]; // (r_lo, col_base8:col_base8+1)
                    auto &d1 = acc.tiles[0][c].data[1]; // (r_hi, col_base:col_base+1)
                    auto &d3 = acc.tiles[0][c].data[3]; // (r_hi, col_base8:col_base8+1)

                    float x0, y0, x1, y1;

                    // d0: r_lo, (col_base, col_base+1)
                    x0 = d0.x; y0 = d0.y;
                    d0.x = x0 * cos_lo_0 - y0 * sin_lo_0;
                    d0.y = x0 * sin_lo_1 + y0 * cos_lo_1;

                    // d2: r_lo, (col_base8, col_base8+1)
                    x0 = d2.x; y0 = d2.y;
                    d2.x = x0 * cos_lo_8 - y0 * sin_lo_8;
                    d2.y = x0 * sin_lo_9 + y0 * cos_lo_9;

                    // d1: r_hi, (col_base, col_base+1)
                    x1 = d1.x; y1 = d1.y;
                    d1.x = x1 * cos_hi_0 - y1 * sin_hi_0;
                    d1.y = x1 * sin_hi_1 + y1 * cos_hi_1;

                    // d3: r_hi, (col_base8, col_base8+1)
                    x1 = d3.x; y1 = d3.y;
                    d3.x = x1 * cos_hi_8 - y1 * sin_hi_8;
                    d3.y = x1 * sin_hi_9 + y1 * cos_hi_9;
                }
            }

            // Convert to bf16 and store.
            rt_bf<16, OUT_BLOCK> out_bf;
            warp::copy(out_bf, acc);

            if (is_q_block) {
                warp::store(g.q_post_rope, out_bf, {global_row / 16, inst.col});
            } else if (is_k_block) {
                // OUT_BLOCK may span multiple heads. Store each head separately.
                int kv_head_base = (inst.col - K_BLOCK_START) * HEADS_PER_BLOCK;
                #pragma unroll
                for (int h = 0; h < HEADS_PER_BLOCK; h++) {
                    constexpr int TILES_PER_HEAD = Globals::head_dim / 16;
                    rt_bf<16, Globals::head_dim> head_tile;
                    #pragma unroll
                    for (int t = 0; t < TILES_PER_HEAD; t++)
                        head_tile.tiles[0][t] = out_bf.tiles[0][h * TILES_PER_HEAD + t];
                    store_kv_paged(g, head_tile, global_row, inst.layer, kv_head_base + h, true);
                }
            } else {
                int kv_head_base = (inst.col - V_BLOCK_START) * HEADS_PER_BLOCK;
                #pragma unroll
                for (int h = 0; h < HEADS_PER_BLOCK; h++) {
                    constexpr int TILES_PER_HEAD = Globals::head_dim / 16;
                    rt_bf<16, Globals::head_dim> head_tile;
                    #pragma unroll
                    for (int t = 0; t < TILES_PER_HEAD; t++)
                        head_tile.tiles[0][t] = out_bf.tiles[0][h * TILES_PER_HEAD + t];
                    store_kv_paged(g, head_tile, global_row, inst.layer, kv_head_base + h, false);
                }
            }

            __threadfence();

            // Release the rope page — only one consumer warp should release,
            // otherwise 16 warps × 16 arrivals = 256 arrivals vs expected 16.
            if (needs_rope && laneid() == 0 && group<Config::NUM_CONSUMER_WARPS>::warpid() == 0)
                s.finish_page(s.pid(ROPE_PAGE), Config::NUM_CONSUMER_WARPS);

            group<Config::NUM_CONSUMER_WARPS>::sync(0);
            if (laneid() == 0 && ::kittens::warpid() == 0) {
                int start_bar = inst.col * HEADS_PER_BLOCK;
                for (int i = 0; i < HEADS_PER_BLOCK; i++)
                    atomicAdd(&g.Bar[{inst.layer, opcode - 1, inst.row, start_bar + i}], 1);
            }
        }
    };
};

struct qkv_gmem_waiter_sm89 {
    template <typename Cfg, typename G, typename Inst>
    static __device__ inline void gmem_wait(const G &g, state<Cfg> &s, Inst &inst) {
        int _w = 0;
        while (*(volatile int *)&g.Bar[{inst.layer, OPCODE_AttnNorm - 1, inst.row, 0}]
               < (int)G::matmul_batch_block_size) {
            __nanosleep(20);
            if (++_w > 50000000 && inst.col == 0 && warp::laneid() == 0) {
                printf("QKV HANG: waiting AttnNorm barrier, layer=%d row=%d val=%d need=%d\n",
                    inst.layer, inst.row,
                    *(volatile int *)&g.Bar[{inst.layer, OPCODE_AttnNorm - 1, inst.row, 0}],
                    (int)G::matmul_batch_block_size);
                break;
            }
        }
    }
};

} // namespace kittens::prototype::vm
