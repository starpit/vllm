// SPDX-License-Identifier: Apache-2.0
//
// Faithful port of mlx `block_sort` (mlx/backend/metal/kernels/sort.h)
// in ARG_SORT mode, contiguous variant only. mlx implements
// `argpartition` by full argsort then a host-side slice
// (`gpu_merge_sort` is dispatched with `argsort=true`; see
// `mlx/backend/metal/sort.cpp:342 ArgPartition::eval_gpu`). The MoE
// router likewise does `inds = mx.argpartition(gates, kth=-k,
// axis=-1)[..., -k:]` (qwen3_next.py:338, qwen3_moe.py:131): a full
// argsort followed by slicing the trailing `k` indices.
//
// Use is limited to the MoE router (`num_experts` ≤ 512), so the
// `single_block_sort` path (one threadgroup per row, all elements in
// threadgroup memory) is sufficient. The multi-block merge-sort path
// in mlx (`mb_block_sort` / `mb_block_partition` / `mb_block_merge`)
// is OMITTED.
//
// The `block_sort_nc` non-contiguous variant is also OMITTED: router
// logits are produced contiguously by the upstream gemm + softmax.
//
// Padding sentinel: `+INFINITY`. Differs from mlx (`quiet_NaN()`) —
// mlx uses NaN so its `LessThan` operator sorts NaNs last. Router
// inputs are post-softmax probabilities in `[0, 1]`, never NaN, so
// `+INFINITY` is equivalent for our purposes and avoids needing
// `metal::isnan` handling in the comparator. Real values fill the
// leading `axis_size` slots of the sorted buffer in ascending order;
// the trailing `k` indices (i.e. `[axis_size - k .. axis_size]`) are
// the argmax-top-k.

#include <metal_stdlib>
using namespace metal;

#define MLX_MTL_LOOP_UNROLL _Pragma("clang loop unroll(full)")

// ─── sentinel: largest representable value of T ─────────────────────
//
// mlx uses `Limits<T>::max` (`mlx/backend/metal/kernels/utils.h`).
// We inline the cases ferrite-metal instantiates here.

template <typename T>
inline T sort_pad_value();

template <>
inline float sort_pad_value<float>() { return float(INFINITY); }

template <>
inline half sort_pad_value<half>() { return half(INFINITY); }

template <>
inline bfloat sort_pad_value<bfloat>() { return bfloat(INFINITY); }

template <>
inline uint sort_pad_value<uint>() { return UINT_MAX; }

// ─── thread-level sort ──────────────────────────────────────────────

template <typename T>
static inline void thread_swap(thread T& a, thread T& b) {
    T w = a;
    a = b;
    b = w;
}

template <typename ValT, typename IdxT, short N_PER_THREAD>
struct ThreadSort {
    static inline void sort(
        thread ValT (&vals)[N_PER_THREAD],
        thread IdxT (&idxs)[N_PER_THREAD]) {
        MLX_MTL_LOOP_UNROLL
        for (short i = 0; i < N_PER_THREAD; ++i) {
            MLX_MTL_LOOP_UNROLL
            for (short j = i & 1; j < N_PER_THREAD - 1; j += 2) {
                if (vals[j + 1] < vals[j]) {
                    thread_swap(vals[j + 1], vals[j]);
                    thread_swap(idxs[j + 1], idxs[j]);
                }
            }
        }
    }
};

// ─── threadgroup-level merge sort ───────────────────────────────────

template <typename ValT, typename IdxT, short BLOCK_THREADS, short N_PER_THREAD>
struct BlockMergeSort {
    using thread_sort_t = ThreadSort<ValT, IdxT, N_PER_THREAD>;

    static inline int merge_partition(
        const threadgroup ValT* As,
        const threadgroup ValT* Bs,
        short A_sz,
        short B_sz,
        short sort_md) {
        short A_st = max(0, sort_md - B_sz);
        short A_ed = min(sort_md, A_sz);

        while (A_st < A_ed) {
            short md = A_st + (A_ed - A_st) / 2;
            auto a = As[md];
            auto b = Bs[sort_md - 1 - md];

            if (b < a) {
                A_ed = md;
            } else {
                A_st = md + 1;
            }
        }
        return A_ed;
    }

    static inline void merge_step(
        const threadgroup ValT* As,
        const threadgroup ValT* Bs,
        const threadgroup IdxT* As_idx,
        const threadgroup IdxT* Bs_idx,
        short A_sz,
        short B_sz,
        thread ValT (&vals)[N_PER_THREAD],
        thread IdxT (&idxs)[N_PER_THREAD],
        ValT init) {
        short a_idx = 0;
        short b_idx = 0;

        for (int i = 0; i < N_PER_THREAD; ++i) {
            auto a = (a_idx < A_sz) ? As[a_idx] : init;
            auto b = (b_idx < B_sz) ? Bs[b_idx] : init;
            bool pred = (b_idx < B_sz) && (a_idx >= A_sz || b < a);

            vals[i] = pred ? b : a;
            if (pred) {
                idxs[i] = Bs_idx[b_idx];
            } else {
                idxs[i] = (a_idx < A_sz) ? As_idx[a_idx] : IdxT(0);
            }

            b_idx += short(pred);
            a_idx += short(!pred);
        }
    }

    static inline void sort(
        threadgroup ValT* tgp_vals,
        threadgroup IdxT* tgp_idxs,
        int size_sorted_axis,
        ValT init,
        uint3 lid) {
        int idx = lid.x * N_PER_THREAD;

        thread ValT thread_vals[N_PER_THREAD];
        thread IdxT thread_idxs[N_PER_THREAD];
        for (int i = 0; i < N_PER_THREAD; ++i) {
            thread_vals[i] = tgp_vals[idx + i];
            thread_idxs[i] = tgp_idxs[idx + i];
        }

        if (idx < size_sorted_axis) {
            thread_sort_t::sort(thread_vals, thread_idxs);
        }

        for (int merge_threads = 2; merge_threads <= BLOCK_THREADS;
             merge_threads *= 2) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = 0; i < N_PER_THREAD; ++i) {
                tgp_vals[idx + i] = thread_vals[i];
                tgp_idxs[idx + i] = thread_idxs[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            int merge_group = int(lid.x) / merge_threads;
            int merge_lane = int(lid.x) % merge_threads;

            int sort_sz = N_PER_THREAD * merge_threads;
            int sort_st = N_PER_THREAD * merge_threads * merge_group;

            int A_st = sort_st;
            int A_ed = sort_st + sort_sz / 2;
            int B_st = sort_st + sort_sz / 2;
            int B_ed = sort_st + sort_sz;

            const threadgroup ValT* As = tgp_vals + A_st;
            const threadgroup ValT* Bs = tgp_vals + B_st;
            int A_sz = A_ed - A_st;
            int B_sz = B_ed - B_st;

            int sort_md = N_PER_THREAD * merge_lane;
            int partition = merge_partition(As, Bs, A_sz, B_sz, sort_md);

            As += partition;
            Bs += sort_md - partition;

            A_sz -= partition;
            B_sz -= sort_md - partition;

            const threadgroup IdxT* As_idx = tgp_idxs + A_st + partition;
            const threadgroup IdxT* Bs_idx = tgp_idxs + B_st + sort_md - partition;

            merge_step(
                As, Bs, As_idx, Bs_idx,
                A_sz, B_sz, thread_vals, thread_idxs, init);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < N_PER_THREAD; ++i) {
            tgp_vals[idx + i] = thread_vals[i];
            tgp_idxs[idx + i] = thread_idxs[i];
        }
    }
};

// ─── kernel ─────────────────────────────────────────────────────────
//
// Bindings:
//   buffer(0) = input   [batch, axis_size]   T
//   buffer(1) = output  [batch, axis_size]   uint32 (full argsort,
//                                            ascending; top-k lives at
//                                            [..., axis_size-k:])
//   buffer(2) = axis_size                    int
//
// Grid: (1, batch, 1) threadgroups, BLOCK_THREADS threads each.

template <typename T, short BLOCK_THREADS, short N_PER_THREAD>
inline void arg_block_sort_impl(
    const device T* inp,
    device uint* out,
    int axis_size,
    threadgroup T* tgp_vals,
    threadgroup uint* tgp_idxs,
    uint3 tid,
    uint3 lid) {
    constexpr int N_PER_BLOCK = BLOCK_THREADS * N_PER_THREAD;

    inp += size_t(tid.y) * size_t(axis_size);
    out += size_t(tid.y) * size_t(axis_size);

    // bfloat lacks an implicit converting ctor from `float`, so route
    // through an explicit cast — equivalent to mlx's
    // `Limits<bfloat>::max` (which is `+numeric_limits<bfloat>::infinity()`).
    // Integer T (uint) uses the type's max instead of INFINITY.
    const T POS_INF = sort_pad_value<T>();

    for (short i = lid.x; i < N_PER_BLOCK; i += BLOCK_THREADS) {
        tgp_vals[i] = i < axis_size ? inp[i] : POS_INF;
        tgp_idxs[i] = uint(i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    BlockMergeSort<T, uint, BLOCK_THREADS, N_PER_THREAD>::sort(
        tgp_vals, tgp_idxs, axis_size, POS_INF, lid);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = lid.x; i < axis_size; i += BLOCK_THREADS) {
        out[i] = tgp_idxs[i];
    }
}

#define ARG_SORT_INSTANTIATE(name, T, BN)                                   \
    [[host_name("c_arg_block_sort_" #name "_uint32_bn" #BN "_tn4")]]        \
    [[kernel, max_total_threads_per_threadgroup(BN)]]                       \
    void c_arg_block_sort_##name##_uint32_bn##BN##_tn4(                     \
        const device T* inp [[buffer(0)]],                                  \
        device uint* out [[buffer(1)]],                                     \
        constant int& axis_size [[buffer(2)]],                              \
        uint3 tid [[threadgroup_position_in_grid]],                         \
        uint3 lid [[thread_position_in_threadgroup]]) {                     \
        threadgroup T tgp_vals[BN * 4];                                     \
        threadgroup uint tgp_idxs[BN * 4];                                  \
        arg_block_sort_impl<T, BN, 4>(                                      \
            inp, out, axis_size, tgp_vals, tgp_idxs, tid, lid);             \
    }

// bn∈{32, 64, 128}, tn=4 — covers axis_size up to 32*4=128, 64*4=256,
// 128*4=512 respectively. MoE router rows are num_experts ∈
// {8 (Mixtral), 60-128 (Qwen3-MoE), 512 (Qwen3-Next)}.

ARG_SORT_INSTANTIATE(float32, float, 32)
ARG_SORT_INSTANTIATE(float32, float, 64)
ARG_SORT_INSTANTIATE(float32, float, 128)
ARG_SORT_INSTANTIATE(float16, half, 32)
ARG_SORT_INSTANTIATE(float16, half, 64)
ARG_SORT_INSTANTIATE(float16, half, 128)
ARG_SORT_INSTANTIATE(bfloat16, bfloat, 32)
ARG_SORT_INSTANTIATE(bfloat16, bfloat, 64)
ARG_SORT_INSTANTIATE(bfloat16, bfloat, 128)

// uint32 argsort is used by the MoE `_gather_sort` helper
// (mlx-lm/mlx_lm/models/switch_layers.py:12): expert indices flattened
// to [N*K] are sorted to compute `order` and again to compute
// `inv_order`. Padding sentinel is `UINT_MAX`.

ARG_SORT_INSTANTIATE(uint32, uint, 32)
ARG_SORT_INSTANTIATE(uint32, uint, 64)
ARG_SORT_INSTANTIATE(uint32, uint, 128)
