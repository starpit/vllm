// SPDX-License-Identifier: Apache-2.0
//! Step C.1 smoke kernel: descriptor-TMA proof-of-concept.
//!
//! Hand-emits a `.cu` that uses TK 2.0's typed `gl<>` + `tma::load_async<>`
//! / `store_async<>` (descriptor form, lowers to `cp.async.bulk.tensor`
//! PTX) to roundtrip a single bf16 tile through smem. Validates the
//! end-to-end build + launch path before generalizing to the orchestrator
//! emit (Step C.2).
//!
//! The driver-API call (`cuTensorMapEncodeTiled`) is INSIDE TK 2.0's
//! `gl<>` host constructor, NOT in Rust. Per `feedback_dogfood_tk20_rust`.
//!
//! Hand-crafted (not orchestrator-generated): the orchestrator's emit
//! pipeline doesn't yet support typed `gl<>` kernel signatures + the
//! C++ host wrapper that constructs them — that's C.2 work. C.1 just
//! proves the .cu can be built + launched + correctness-verified
//! through ferrite-cuda-builder + the existing Rust launcher FFI.

use std::path::PathBuf;

const KERNEL_NAME: &str = "tk_tma_tensor_smoke";

/// Tile shape: 64 rows × 128 cols of bf16 = 16 KB, fits in one
/// dynamic-smem block. `st_bf<height, width>` is in TK 2.0 16-element
/// tiles (height=4 → 64 rows, width=8 → 128 cols).
const ROWS: u32 = 64;
const COLS: u32 = 128;

fn cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("TK_EMIT_DIR") {
        return PathBuf::from(p);
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("cudaforge/megakernels")
}

fn cu_source() -> String {
    let rows = ROWS;
    let cols = COLS;
    format!(
        r#"// Auto-emitted by tk_emit_tma_tensor_smoke (Step C.1 PoC).
// TK 2.0 typed gl<> + descriptor TMA. cp.async.bulk.tensor PTX.
#include "kittens.cuh"

using namespace kittens;

namespace {{

constexpr int ROWS = {rows};
constexpr int COLS = {cols};

// Tile shape: ROWS×COLS bf16. The `st_bf<H, W>` alias takes
// ROW/COL counts in ELEMENTS (not 16-element tiles, despite the
// `_height`/`_width` template arg names) — verified at
// `include/types/shared/st.cuh:312` and the `st<>` static_assert
// at line 79-80 (rows % 16 == 0, cols % 16 == 0 with swizzle).
using TILE_T = st_bf<ROWS, COLS>;

// Typed global layout. b=d=1 (effectively 2D); r/c baked as literals.
// `TILE_T` in TMA_Types... causes the gl<> ctor to build one
// CUtensorMap descriptor for this tile shape via TK 2.0's
// detail::tma::create_tensor_map → cuTensorMapEncodeTiled.
using BUF_LAYOUT = gl<bf16, 1, 1, ROWS, COLS, TILE_T>;

struct globals {{
    BUF_LAYOUT in_buf;
    BUF_LAYOUT out_buf;
}};

__global__ __launch_bounds__(32) void {KERNEL_NAME}_kernel(
    const __grid_constant__ globals g
) {{
    extern __shared__ __align__(128) uint8_t __dyn_smem[];
    auto& tile = *reinterpret_cast<TILE_T*>(__dyn_smem);

    __shared__ semaphore bar;

    if (kittens::laneid() == 0) {{
        init_semaphore(bar, 0, 1);
    }}
    __syncthreads();

    // Single warp issues the descriptor TMA load. Lane 0 of the
    // group<1> wrapper does the actual `cp.async.bulk.tensor` issue.
    if (kittens::laneid() == 0) {{
        kittens::group<1>::tma::expect_bytes(bar, sizeof(bf16) * ROWS * COLS);
    }}
    __syncthreads();
    kittens::group<1>::tma::load_async<dim::ROW, cache_policy::NORMAL>(
        tile, g.in_buf, {{0, 0, 0, 0}}, bar);

    // Wait for load complete (round 0 → phase 0).
    kittens::group<1>::wait(bar, 0);
    __syncthreads();

    // Descriptor TMA store back to gmem.
    kittens::group<1>::tma::store_async<dim::ROW, cache_policy::NORMAL>(
        g.out_buf, tile, {{0, 0, 0, 0}});
    kittens::group<1>::tma::store_commit_group();
    kittens::group<1>::tma::store_async_wait();
}}

}}  // anonymous namespace

// Host wrapper. Same FFI shape as the orchestrator-emitted launchers
// (void* const* bufs + uint32_t* u32_args + stream). Constructs
// typed gl<> instances via TK 2.0's host ctor — the
// cuTensorMapEncodeTiled call happens inside.
extern "C" cudaError_t launch_{KERNEL_NAME}(
    void* const* bufs,
    const uint32_t* /*u32_args*/,
    cudaStream_t stream
) {{
    constexpr size_t TILE_BYTES = sizeof(bf16) * ROWS * COLS;
    constexpr size_t DYN_SMEM = TILE_BYTES + 256;  // tile + alignment slack.
    cudaError_t __err = cudaFuncSetAttribute(
        (const void*)&{KERNEL_NAME}_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)DYN_SMEM
    );
    if (__err != cudaSuccess) return __err;

    bf16* in_ptr  = reinterpret_cast<bf16*>(bufs[0]);
    bf16* out_ptr = reinterpret_cast<bf16*>(bufs[1]);

    // Compile-time-fixed dims (b=d=1, r=ROWS, c=COLS in BUF_LAYOUT
    // template args) expect `nullptr` per TK 2.0's
    // `ducks::gl::make_arg_t<d>` SFINAE (util.cuh:21):
    // `std::conditional_t<rdim<d>, size_t, std::nullptr_t>`. The
    // dims are baked into the type, not passed at runtime.
    globals g{{
        BUF_LAYOUT(in_ptr,  nullptr, nullptr, nullptr, nullptr),
        BUF_LAYOUT(out_ptr, nullptr, nullptr, nullptr, nullptr),
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
        "wrote {} ({} bytes; tile={}×{} bf16)",
        path.display(),
        src.len(),
        ROWS,
        COLS,
    );
}
