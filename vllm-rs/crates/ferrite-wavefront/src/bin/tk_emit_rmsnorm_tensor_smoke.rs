// SPDX-License-Identifier: Apache-2.0
//! Step C.2 minimum — descriptor-TMA over a real op (RmsNorm).
//!
//! Hand-emits a `.cu` that uses TK 2.0's typed `kittens::gl<>` +
//! `tma::load_async<cache_policy::NORMAL>` (vector form, on
//! `sv_bf<K>` operands) for the activation + weight loads and the
//! output store. m=1 RmsNorm has a 1×hidden activation row, which
//! doesn't fit the `st_bf<R, C>` 16-row minimum used in C.1's
//! `tk_tma_tensor_smoke`; the vector form (`sv_bf<K>`) handles 1D
//! shapes without padding.
//!
//! Compute body is the SAME `tk20::rmsnorm_consumer_body` register-
//! vector reduce that production uses (Phase 10) — only the gmem ↔
//! smem path differs.
//!
//! cuTensorMapEncodeTiled stays inside TK 2.0's `gl<>` host ctor;
//! Rust never touches the driver TMA API.

use std::path::PathBuf;

const KERNEL_NAME: &str = "tk_rmsnorm_tensor_smoke";

const HIDDEN: u32 = 2048;
const EPS: f32 = 1e-5;

fn cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("TK_EMIT_DIR") {
        return PathBuf::from(p);
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("cudaforge/megakernels")
}

fn cu_source() -> String {
    let hidden = HIDDEN;
    let eps = EPS;
    let k_tile = 128_u32;
    let k_blocks = hidden / k_tile;
    format!(
        r#"// Auto-emitted by tk_emit_rmsnorm_tensor_smoke (Step C.2 PoC).
// TK 2.0 typed gl<> + sv_bf vector descriptor TMA.
// `cp.async.bulk.tensor` PTX → SASS UTMALDG/UTMASTG.
#include "kittens.cuh"

using namespace kittens;

namespace {{

constexpr int HIDDEN = {hidden};

// 1D shared vector — `sv_bf<K>` requires K % 16 == 0.
using VEC_T = sv_bf<HIDDEN>;

// gl<>'s last template arg `TMA_Types... = VEC_T` causes the host
// ctor to build one CUtensorMap descriptor for vector TMA on this
// shape. `gl<bf16, 1, 1, 1, HIDDEN, ...>` — b=d=r=1, c=HIDDEN —
// effectively 1D (c-axis tiling).
using ACT_LAYOUT = gl<bf16, 1, 1, 1, HIDDEN, VEC_T>;

struct globals {{
    ACT_LAYOUT in_buf;
    ACT_LAYOUT w_buf;
    ACT_LAYOUT out_buf;
}};

__global__ __launch_bounds__(32) void {KERNEL_NAME}_kernel(
    const __grid_constant__ globals g
) {{
    extern __shared__ __align__(128) uint8_t __dyn_smem[];
    // Two sv_bf<HIDDEN> regions: x and weight.
    auto& x_vec = *reinterpret_cast<VEC_T*>(__dyn_smem);
    auto& w_vec = *reinterpret_cast<VEC_T*>(__dyn_smem + sizeof(VEC_T) + 128);

    __shared__ semaphore bar_x, bar_w;

    if (kittens::laneid() == 0) {{
        init_semaphore(bar_x, 0, 1);
        init_semaphore(bar_w, 0, 1);
    }}
    __syncthreads();

    if (kittens::laneid() == 0) {{
        kittens::group<1>::tma::expect_bytes(bar_x, sizeof(bf16) * HIDDEN);
        kittens::group<1>::tma::expect_bytes(bar_w, sizeof(bf16) * HIDDEN);
    }}
    __syncthreads();

    // Vector-form descriptor TMA loads.
    kittens::group<1>::tma::load_async<cache_policy::NORMAL>(
        x_vec, g.in_buf, {{0, 0, 0, 0}}, bar_x);
    kittens::group<1>::tma::load_async<cache_policy::NORMAL>(
        w_vec, g.w_buf,  {{0, 0, 0, 0}}, bar_w);

    kittens::group<1>::wait(bar_x, 0);
    kittens::group<1>::wait(bar_w, 0);
    __syncthreads();

    // RmsNorm compute body — same K_TILE register-vector reduce as
    // tk20::rmsnorm_consumer_body uses in production (Phase 10).
    using T_act = __nv_bfloat16;
    auto* __x_smem = reinterpret_cast<T_act*>(&x_vec);
    auto* __w_smem = reinterpret_cast<T_act*>(&w_vec);
    constexpr int K_TILE = {k_tile};
    constexpr int K_BLOCKS = {k_blocks};
    constexpr float __eps = {eps:?}f;
    const int __lane = static_cast<int>(threadIdx.x & 31);
    float __sumsq = 0.0f;
    for (int __k_i = 0; __k_i < K_BLOCKS; ++__k_i) {{
        kittens::rv_bf<K_TILE> __x_rv_bf;
        kittens::rv_fl<K_TILE> __x_rv_fl;
        kittens::warp::load(__x_rv_bf, x_vec.template subvec<K_TILE>(__k_i));
        kittens::warp::copy(__x_rv_fl, __x_rv_bf);
        kittens::warp::mul(__x_rv_fl, __x_rv_fl, __x_rv_fl);
        __sumsq += kittens::warp::sum(__x_rv_fl);
    }}
    const float __scale = rsqrtf(__sumsq / static_cast<float>(HIDDEN) + __eps);
    for (unsigned int __i = static_cast<unsigned int>(__lane);
         __i < static_cast<unsigned int>(HIDDEN); __i += 32u) {{
        const float __v = __bfloat162float(__x_smem[__i]);
        const float __g = __bfloat162float(__w_smem[__i]);
        __x_smem[__i] = __float2bfloat16(__v * __scale * __g);
    }}
    __syncthreads();

    // Vector-form descriptor TMA store. Result is in x_vec (in-place).
    kittens::group<1>::tma::store_async<cache_policy::NORMAL>(
        g.out_buf, x_vec, {{0, 0, 0, 0}});
    kittens::group<1>::tma::store_commit_group();
    kittens::group<1>::tma::store_async_wait();
}}

}}  // anonymous namespace

extern "C" cudaError_t launch_{KERNEL_NAME}(
    void* const* bufs,
    const uint32_t* /*u32_args*/,
    cudaStream_t stream
) {{
    constexpr size_t VEC_BYTES = sizeof(bf16) * HIDDEN;
    constexpr size_t DYN_SMEM = VEC_BYTES * 2 + 256;
    cudaError_t __err = cudaFuncSetAttribute(
        (const void*)&{KERNEL_NAME}_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)DYN_SMEM);
    if (__err != cudaSuccess) return __err;

    bf16* in_ptr  = reinterpret_cast<bf16*>(bufs[0]);
    bf16* w_ptr   = reinterpret_cast<bf16*>(bufs[1]);
    bf16* out_ptr = reinterpret_cast<bf16*>(bufs[2]);

    // gl<> ctor with all-nullptr dim args — every dim is compile-
    // time-fixed in the template. TK 2.0's `make_arg_t<d>` SFINAE
    // (util.cuh:21): non-runtime dims accept `nullptr`.
    globals g{{
        ACT_LAYOUT(in_ptr,  nullptr, nullptr, nullptr, nullptr),
        ACT_LAYOUT(w_ptr,   nullptr, nullptr, nullptr, nullptr),
        ACT_LAYOUT(out_ptr, nullptr, nullptr, nullptr, nullptr),
    }};

    {KERNEL_NAME}_kernel<<<1, 32, DYN_SMEM, stream>>>(g);
    return cudaGetLastError();
}}
"#
    )
}

fn main() {
    let out_dir = cache_dir();
    std::fs::create_dir_all(&out_dir).expect("mkdir cudaforge/megakernels");

    let src = cu_source();
    let path = out_dir.join(format!("{KERNEL_NAME}.cu"));
    std::fs::write(&path, &src).expect("write .cu");
    eprintln!(
        "wrote {} ({} bytes; hidden={})",
        path.display(),
        src.len(),
        HIDDEN,
    );
}
