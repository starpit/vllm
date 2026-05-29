// SPDX-License-Identifier: Apache-2.0
//! `TkProgram` → CUDA-source emit. The walker is a literal `match` over
//! [`TkInstr`]; it makes NO decisions. Every numeric the kernel needs
//! (page id, phase parity, warp role, tile shape, region offset) is in
//! the IR.
//!
//! This is the "trivial player" rule from `subtile_ir`'s design,
//! lifted into the warp tier: any cleverness in `emit_*` is a bug —
//! it belongs in the per-op lowering that built the [`TkProgram`].
//!
//! # tk20 dogfood
//!
//! Per `feedback_dogfood_tk20_rust` every TK 2.0 call must go through
//! a typed Rust API, never an inline `format!("kittens::warp::*")`.
//! The thin [`tk20`] module here is that API. It is intentionally a
//! stub at this stage: each function emits one TK 2.0 call as a
//! `String`. When the full `tk20` crate from ff-mega-codegen is
//! cherry-picked, the body of each function moves there 1:1; call
//! sites stay unchanged.

#![allow(dead_code)]

use crate::tk_warp_ir::{
    LoopBound, PageBarrier, TileShape, TkInstr, TkProgram, WarpRole, NUM_CONSUMER_WARPS,
};

// ── tk20 — typed CUDA-source emitters ───────────────────────────────

/// Stub for the `tk20::*` Rust API. Each function returns the textual
/// CUDA fragment for one TK 2.0 primitive call. Lane-0-gated TMA per
/// `feedback_tk20_tma_lane_gate`: the loader/storer paths use
/// `kittens::group<1>::tma::*` (not the thread-level `kittens::tma::*`).
pub mod tk20 {
    use super::PageBarrier;

    fn barrier_field(kind: PageBarrier) -> &'static str {
        match kind {
            PageBarrier::Ready => "page_ready",
            PageBarrier::Done => "page_done",
            PageBarrier::Consumed => "page_consumed",
        }
    }

    /// `kittens::group<N>::wait(barrier, phase)`. `n_warps` is the
    /// thread group the call is gated on. `phase_expr` is the literal
    /// CUDA expression — `"0"` / `"1"` for static phases, `"(__i & 1)"`
    /// or similar for runtime parities.
    pub fn wait(n_warps: u32, kind: PageBarrier, page_id: u8, phase_expr: &str) -> String {
        let bar = barrier_field(kind);
        format!("kittens::group<{n_warps}>::wait({bar}[{page_id}], {phase_expr});")
    }

    /// `kittens::group<N>::arrive(barrier)`.
    pub fn arrive(n_warps: u32, kind: PageBarrier, page_id: u8) -> String {
        let bar = barrier_field(kind);
        format!("kittens::group<{n_warps}>::arrive({bar}[{page_id}]);")
    }

    /// `kittens::group<1>::sync()` / `<N>::sync()`.
    pub fn sync(n_warps: u32) -> String {
        format!("kittens::group<{n_warps}>::sync();")
    }

    /// Total bytes a tile of `(rows, cols)` occupies given `elem_bytes`.
    fn tile_bytes(rows: u32, cols: u32, elem_bytes: u32) -> u32 {
        rows * cols * elem_bytes
    }

    /// Lane-0-gated TMA load. Emits the production "non-tensor TMA"
    /// pair: `expect_bytes` arms the page-ready barrier for the byte
    /// count, then `load_async(dst, src, bytes, ready)` fires the load.
    /// The barrier completes when the load delivers all expected bytes —
    /// no separate `arrive(Ready)` call is needed (and emitting one
    /// would over-count the arrivals and break round parity).
    ///
    /// Source: `include/ops/group/util/tma.cuh:18` (`expect_bytes`) and
    /// `:72` (`load_async(void*, void*, uint32_t, semaphore&)`).
    pub fn tma_load_async(
        page_id: u8,
        src_buf: u32,
        src_byte_off: u64,
        rows: u32,
        cols: u32,
        elem_bytes: u32,
        dyn_byte_off: Option<&str>,
    ) -> String {
        let bytes = tile_bytes(rows, cols, elem_bytes);
        let off_expr = match dyn_byte_off {
            Some(e) => format!("({src_byte_off}u + ({e}))"),
            None => format!("{src_byte_off}"),
        };
        // Cast through `uintptr_t` to drop `const` from the buffer arg
        // (the kernel signature uses `const __nv_bfloat16* __restrict__`
        // for inputs, but TK 2.0 `tma::load_async(void*, void*, ...)`
        // wants non-const). `reinterpret_cast<void*>(const char*)` is
        // rejected — `cannot cast away const`.
        format!(
            "kittens::group<1>::tma::expect_bytes(page_ready[{page_id}], {bytes}); \
             kittens::group<1>::tma::load_async(\
             reinterpret_cast<void*>(page_buf[{page_id}]), \
             reinterpret_cast<void*>(\
             reinterpret_cast<uintptr_t>(buf{src_buf}) + {off_expr}), \
             {bytes}, \
             page_ready[{page_id}]);"
        )
    }

    /// Lane-0-gated TMA store. Emits the production pair:
    /// `store_async(dst, src, bytes)` queues the store and
    /// `store_async_wait()` blocks the storer warp until the store
    /// completes (so the subsequent `arrive(Consumed)` is safe to use
    /// the page slot for the next round).
    ///
    /// Source: `include/ops/group/util/tma.cuh:82` (`store_async`) and
    /// `:46` (`store_async_wait<N=0>`).
    pub fn tma_store_async(
        page_id: u8,
        dst_buf: u32,
        dst_byte_off: u64,
        rows: u32,
        cols: u32,
        elem_bytes: u32,
        dyn_byte_off: Option<&str>,
    ) -> String {
        let bytes = tile_bytes(rows, cols, elem_bytes);
        let off_expr = match dyn_byte_off {
            Some(e) => format!("({dst_byte_off}u + ({e}))"),
            None => format!("{dst_byte_off}"),
        };
        format!(
            "kittens::group<1>::tma::store_async(\
             reinterpret_cast<void*>(reinterpret_cast<char*>(buf{dst_buf}) + {off_expr}), \
             reinterpret_cast<void*>(page_buf[{page_id}]), \
             {bytes}); \
             kittens::group<1>::tma::store_async_wait();"
        )
    }
}

// ── Role routing ───────────────────────────────────────────────────

fn role_guard(role: WarpRole) -> Option<String> {
    match role {
        WarpRole::All => None,
        WarpRole::Loader => Some("if (__role == ROLE_LOADER)".to_string()),
        WarpRole::Storer => Some("if (__role == ROLE_STORER)".to_string()),
        WarpRole::AllConsumers => Some("if (__role == ROLE_CONSUMER)".to_string()),
        WarpRole::Consumer(i) => {
            Some(format!("if (__role == ROLE_CONSUMER && __consumer_idx == {i})"))
        }
    }
}

/// `kittens::group<N>` width for a role: TMA/sync gates are sized to
/// the role's warp count.
fn role_group_width(role: WarpRole) -> u32 {
    match role {
        WarpRole::All => NUM_CONSUMER_WARPS as u32 + 2, // 8 consumers + loader + storer (illustrative)
        WarpRole::Loader | WarpRole::Storer | WarpRole::Consumer(_) => 1,
        WarpRole::AllConsumers => NUM_CONSUMER_WARPS as u32,
    }
}

// ── Walk ───────────────────────────────────────────────────────────

fn emit_one(instr: &TkInstr, out: &mut String) {
    if let TkInstr::ForLoop { var, count, body } = instr {
        // The loop hosts every role together; per-instr role guards
        // inside the body still route work to the right warp.
        out.push_str("    for (uint ");
        out.push_str(var);
        out.push_str(" = 0; ");
        out.push_str(var);
        out.push_str(" < ");
        out.push_str(&LoopBound::cuda_expr(count));
        out.push_str("; ++");
        out.push_str(var);
        out.push_str(") {\n");
        for inner in body {
            emit_one(inner, out);
        }
        out.push_str("    }\n");
        return;
    }

    let (role, body) = match instr {
        TkInstr::Wait {
            role,
            page_id,
            kind,
            phase,
        } => {
            let n = role_group_width(*role);
            (*role, tk20::wait(n, *kind, *page_id, &phase.cuda_expr()))
        }
        TkInstr::Arrive {
            role,
            page_id,
            kind,
        } => {
            let n = role_group_width(*role);
            (*role, tk20::arrive(n, *kind, *page_id))
        }
        TkInstr::LoadAsync {
            page_id,
            src,
            src_region,
            tile,
            dyn_byte_off,
        } => {
            // src_region.region carries (rows, cols) — use the IR tile
            // for the TMA descriptor and the region's column-offset
            // for the byte offset.
            let TileShape {
                rows,
                cols,
                elem_bytes,
            } = *tile;
            let byte_off = (src_region.region.cols.start as u64) * (elem_bytes as u64);
            (
                WarpRole::Loader,
                tk20::tma_load_async(
                    *page_id,
                    src.0,
                    byte_off,
                    rows,
                    cols,
                    elem_bytes,
                    dyn_byte_off.as_deref(),
                ),
            )
        }
        TkInstr::StoreAsync {
            page_id,
            dst,
            dst_region,
            tile,
            dyn_byte_off,
        } => {
            let TileShape {
                rows,
                cols,
                elem_bytes,
            } = *tile;
            let byte_off = (dst_region.region.cols.start as u64) * (elem_bytes as u64);
            (
                WarpRole::Storer,
                tk20::tma_store_async(
                    *page_id,
                    dst.0,
                    byte_off,
                    rows,
                    cols,
                    elem_bytes,
                    dyn_byte_off.as_deref(),
                ),
            )
        }
        TkInstr::Compute { role, body } => (*role, body.clone()),
        TkInstr::Sync { role } => {
            let n = role_group_width(*role);
            (*role, tk20::sync(n))
        }
        // Handled by the early return above. Reachable only if a
        // future refactor breaks that contract; an `unreachable!` is
        // the right tripwire.
        TkInstr::ForLoop { .. } => unreachable!("ForLoop handled by early return"),
    };

    match role_guard(role) {
        None => {
            out.push_str("    ");
            out.push_str(&body);
            out.push('\n');
        }
        Some(g) => {
            out.push_str("    ");
            out.push_str(&g);
            out.push_str(" {\n        ");
            out.push_str(&body);
            out.push_str("\n    }\n");
        }
    }
}

/// Emit the persistent CTA body for a [`TkProgram`]. Caller wraps it
/// in the kernel signature + page/scratch declarations + the role
/// dispatch (`__role`, `__consumer_idx`); this fn is *only* the body
/// the role-routed match arms produce.
pub fn emit_body(prog: &TkProgram) -> String {
    let mut out = String::new();
    for instr in &prog.instrs {
        emit_one(instr, &mut out);
    }
    out
}

// ── Kernel scaffold ────────────────────────────────────────────────

/// One typed buffer argument to the persistent-CTA kernel. The emit
/// puts these in the kernel signature in declaration order; the
/// in-IR `BufId` indexes into this list (`buf{id}` in the body matches
/// `args[id].name`).
#[derive(Clone, Debug)]
pub struct KernelArg {
    /// CUDA type, e.g. `"const __nv_bfloat16* __restrict__"`.
    pub ty: String,
    /// Identifier name in the kernel signature; the body references
    /// `buf{i}` where `i` is the position in [`KernelArgs::bufs`].
    /// The codegen synthesises a `#define buf{i} <name>` so the body's
    /// `buf3 /* +256 */` substitution lands on the right argument.
    pub name: String,
}

/// The full kernel arg pack. Runtime u32 args (e.g. the
/// [`crate::tk_warp_ir::LoopBound::RuntimeU32`] names referenced by
/// the body's ForLoops) must be present here too; the codegen
/// otherwise has nowhere to declare them.
#[derive(Clone, Debug, Default)]
pub struct KernelArgs {
    pub bufs: Vec<KernelArg>,
    /// Names of runtime u32 args (no type — always `uint32_t`).
    pub u32_args: Vec<String>,
}

/// Emit a complete TK 2.0 persistent-CTA kernel. The output is a
/// `.cu` snippet with `#include "kittens.cuh"`, the kernel signature
/// (`__global__ __launch_bounds__(...) void <name>(...)`), the page
/// pool / mbarrier declarations, the init handshake (matches TK 2.0:
/// `page_ready[i].init(0)`, `page_done[i].init(0)`,
/// `page_consumed[i].init(0)` then `arrive_pre`), the role dispatch
/// (warpid 0 = loader, 1 = storer, 2..9 = consumer), the body the
/// caller built via [`emit_body`], and the final group sync.
///
/// The total warp count is `1 (loader) + 1 (storer) +
/// NUM_CONSUMER_WARPS (consumers) = 10`, so threadIdx.x ranges over
/// `[0, 320)` and `__launch_bounds__(320)` is emitted.
///
/// This is a *placeholder* in the sense that the body's compute
/// fragments still reference symbols (`__page_smem`, `__weight_smem`,
/// `__q_smem`, etc.) the per-op atom is responsible for binding. The
/// scaffold does not synthesise those bindings — they're the per-op
/// pre-resolved fragments that today's atom_lib produces. What this
/// scaffold *does* guarantee is that the role dispatch, mbarrier
/// init, and final sync are byte-identical to TK 2.0.
pub fn emit_kernel(name: &str, args: &KernelArgs, prog: &TkProgram) -> String {
    use crate::tk_warp_ir::{NUM_CONSUMER_WARPS, NUM_PAGES};

    let total_warps = NUM_CONSUMER_WARPS as u32 + 2; // 1 loader + 1 storer + N consumers
    let total_threads = total_warps * 32;

    let mut out = String::new();
    out.push_str("#include \"kittens.cuh\"\n");
    out.push_str("\n");

    // Role constants. Matches the WarpRole emit in role_guard().
    out.push_str("#define ROLE_LOADER   0\n");
    out.push_str("#define ROLE_STORER   1\n");
    out.push_str("#define ROLE_CONSUMER 2\n");
    out.push_str("\n");

    // Kernel signature.
    out.push_str(&format!(
        "__global__ __launch_bounds__({total_threads}) void {name}(\n"
    ));
    let mut first = true;
    for arg in &args.bufs {
        if !first {
            out.push_str(",\n");
        }
        first = false;
        out.push_str(&format!("    {} {}", arg.ty, arg.name));
    }
    for u32_name in &args.u32_args {
        if !first {
            out.push_str(",\n");
        }
        first = false;
        out.push_str(&format!("    const uint32_t {u32_name}"));
    }
    out.push_str("\n) {\n");
    out.push_str("    using namespace kittens;\n");
    out.push_str("\n");

    // `buf{i}` aliases so the body's `buf3 /* +256 */` substitutions
    // resolve to the right named arg.
    for (i, arg) in args.bufs.iter().enumerate() {
        out.push_str(&format!("    auto& buf{i} = {};\n", arg.name));
    }
    out.push_str("\n");

    // Role dispatch.
    out.push_str("    const int __warpid = threadIdx.x / 32;\n");
    out.push_str("    int __role;\n");
    out.push_str("    if      (__warpid == 0) __role = ROLE_LOADER;\n");
    out.push_str("    else if (__warpid == 1) __role = ROLE_STORER;\n");
    out.push_str("    else                    __role = ROLE_CONSUMER;\n");
    out.push_str("    const int __consumer_idx = __warpid - 2;\n");
    out.push_str("    (void)__consumer_idx;\n");
    out.push_str("\n");

    // Page pool + mbarriers. The page byte-buffer is allocated as
    // dynamic shared memory; per-page typed views are the per-op
    // atom's responsibility.
    out.push_str(&format!(
        "    __shared__ kittens::semaphore page_ready[{}];\n",
        NUM_PAGES
    ));
    out.push_str(&format!(
        "    __shared__ kittens::semaphore page_done[{}];\n",
        NUM_PAGES
    ));
    out.push_str(&format!(
        "    __shared__ kittens::semaphore page_consumed[{}];\n",
        NUM_PAGES
    ));
    // Dynamic shared memory: TK 2.0 production pattern. The launcher must
    // set `cudaFuncAttributeMaxDynamicSharedMemorySize` to at least
    // `NUM_PAGES * PAGE_SIZE` so the pool fits. nvcc itself only requires
    // the `extern __shared__` declaration; ptxas will not count it against
    // the static smem cap (default 0xc000 on H100). Per-page typed views
    // are the per-op atom's responsibility; we expose `page_buf[i]` as
    // an array-of-PAGE_SIZE-byte rows so the rest of the emit code can
    // keep using `page_buf[id]` as before.
    // `__align__(128)` is the CUDA-canonical alignment attribute for an
    // `extern __shared__` array. `alignas(...)` collides with the
    // `__attribute__((shared))` that the `__shared__` macro expands to.
    out.push_str("    extern __shared__ __align__(128) uint8_t __dynamic_smem[];\n");
    out.push_str(&format!(
        "    auto (&page_buf)[{}][{}] = *reinterpret_cast<uint8_t(*)[{}][{}]>(__dynamic_smem);\n",
        NUM_PAGES,
        crate::tk_warp_ir::PAGE_SIZE,
        NUM_PAGES,
        crate::tk_warp_ir::PAGE_SIZE
    ));
    out.push_str("\n");

    // Init: lane-0 of warp-0 sets up all mbarriers; consumed is
    // pre-arrived so its first wait reads the post-arrive parity.
    // (TK 2.0 default; matches kittens header `page_consumed[i].arrive_pre()`.)
    out.push_str("    if (__warpid == 0 && (threadIdx.x & 31) == 0) {\n");
    out.push_str(&format!(
        "        for (int __i = 0; __i < {}; ++__i) {{\n",
        NUM_PAGES
    ));
    out.push_str("            kittens::init_semaphore(page_ready[__i], 0, 1);\n");
    out.push_str(&format!(
        "            kittens::init_semaphore(page_done[__i], 0, {});\n",
        NUM_CONSUMER_WARPS
    ));
    out.push_str("            kittens::init_semaphore(page_consumed[__i], 0, 1);\n");
    out.push_str("            kittens::arrive(page_consumed[__i]);\n");
    out.push_str("        }\n");
    out.push_str("    }\n");
    // CTA-wide sync. `kittens::group<N>::sync()` (barrier-less) is
    // only legal for single-warp groups (asserts `GROUP_WARPS==1`); the
    // multi-warp form takes a `bar.sync` barrier id. Easiest portable
    // choice for "all 10 warps converge" is just `__syncthreads()`.
    out.push_str("    __syncthreads();\n");
    out.push_str("\n");

    // Function-scope prelude: typed page views, persistent compute
    // accumulators that span multiple consumer compute steps. See
    // `TkProgram::prelude` for the contract.
    if !prog.prelude.is_empty() {
        out.push_str("    // ── tk_warp_ir prelude ──\n");
        out.push_str(&prog.prelude);
        if !prog.prelude.ends_with('\n') {
            out.push('\n');
        }
        out.push('\n');
    }

    // The role-routed body.
    out.push_str("    // ── tk_warp_ir body ──\n");
    out.push_str(&emit_body(prog));
    out.push_str("\n");

    // Final sync: every warp waits for the others before retiring.
    // CTA-wide sync. `kittens::group<N>::sync()` (barrier-less) is
    // only legal for single-warp groups (asserts `GROUP_WARPS==1`); the
    // multi-warp form takes a `bar.sync` barrier id. Easiest portable
    // choice for "all 10 warps converge" is just `__syncthreads()`.
    out.push_str("    __syncthreads();\n");
    out.push_str("}\n");
    out
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subtile::{Range, Region};
    use crate::subtile_ir::{BufId, RegionRef};
    use crate::tk_warp_ir::{Phase0, PageHandle};

    fn rr(buf: u32, c0: u32, w: u32) -> RegionRef {
        RegionRef {
            buffer: BufId(buf),
            region: Region {
                rows: Range::new(0, 1),
                cols: Range::new(c0, w),
            },
        }
    }

    /// A tiny tape: loader fills page 0, all consumers compute, storer
    /// drains. The phase parity in the emitted source matches the IR.
    #[test]
    fn emits_loader_consumer_storer_handshake() {
        let mut p = TkProgram::new();
        let page: PageHandle<Phase0> = PageHandle::fresh(0);

        // Round 0: every wait reads phase 0 (TK 2.0 round parity =
        // R & 1; per-barrier flips happen but the wait sees the
        // start-of-round parity).
        let page = p.wait(WarpRole::Loader, PageBarrier::Consumed, page);
        p.load_async(
            0,
            BufId(7),
            rr(7, 0, 64),
            TileShape {
                rows: 1,
                cols: 64,
                elem_bytes: 2,
            },
        );
        let page = p.arrive(WarpRole::Loader, PageBarrier::Ready, page);

        let page = p.wait(WarpRole::AllConsumers, PageBarrier::Ready, page);
        p.compute(WarpRole::AllConsumers, "/* rms reduce + scale */");
        let page = p.arrive(WarpRole::AllConsumers, PageBarrier::Done, page);

        let page = p.wait(WarpRole::Storer, PageBarrier::Done, page);
        p.store_async(
            0,
            BufId(8),
            rr(8, 0, 64),
            TileShape {
                rows: 1,
                cols: 64,
                elem_bytes: 2,
            },
        );
        let _page = p.arrive(WarpRole::Storer, PageBarrier::Consumed, page);

        let src = emit_body(&p);

        // The phase parities the codegen emits are a literal copy of
        // what the type system computed: round 0 → all waits read 0.
        assert!(src.contains("page_consumed[0], 0"), "loader wait phase=0\n{src}");
        assert!(src.contains("page_ready[0], 0"), "consumer wait phase=0\n{src}");
        assert!(src.contains("page_done[0], 0"), "storer wait phase=0\n{src}");

        // Roles route correctly: loader/storer are gated on __role,
        // the consumer body is in the consumer arm.
        assert!(src.contains("if (__role == ROLE_LOADER)"), "loader gate\n{src}");
        assert!(src.contains("if (__role == ROLE_STORER)"), "storer gate\n{src}");
        assert!(src.contains("if (__role == ROLE_CONSUMER)"), "consumer gate\n{src}");
        assert!(src.contains("rms reduce + scale"), "compute body pasted\n{src}");
    }

    #[test]
    fn tma_load_carries_src_byte_offset_and_arms_ready() {
        let mut p = TkProgram::new();
        // src region starts at column 128, elem_bytes=2 → byte_off=256.
        // tile = 1×64×2 bytes → 128 bytes total.
        p.load_async(
            5,
            BufId(3),
            rr(3, 128, 64),
            TileShape {
                rows: 1,
                cols: 64,
                elem_bytes: 2,
            },
        );
        let src = emit_body(&p);
        // expect_bytes arms page_ready[5] for 128 bytes.
        assert!(
            src.contains("kittens::group<1>::tma::expect_bytes(page_ready[5], 128);"),
            "expect_bytes arms ready barrier\n{src}"
        );
        // load_async references the same page_buf[5] dst, the byte-
        // offset src, and uses page_ready[5] as the semaphore.
        assert!(
            src.contains("reinterpret_cast<void*>(page_buf[5])"),
            "page dst\n{src}"
        );
        assert!(
            src.contains("reinterpret_cast<uintptr_t>(buf3) + 256"),
            "src byte offset (uintptr_t form so const is dropped)\n{src}"
        );
        assert!(
            src.contains(", 128, page_ready[5]);"),
            "load takes bytes + page_ready barrier\n{src}"
        );
    }

    #[test]
    fn kernel_scaffold_has_role_dispatch_and_mbarrier_init() {
        let mut p = TkProgram::new();
        let page: PageHandle<Phase0> = PageHandle::fresh(0);
        let _page = p.wait(WarpRole::Loader, PageBarrier::Consumed, page);

        let args = KernelArgs {
            bufs: vec![
                KernelArg {
                    ty: "const __nv_bfloat16* __restrict__".into(),
                    name: "x".into(),
                },
                KernelArg {
                    ty: "__nv_bfloat16* __restrict__".into(),
                    name: "out".into(),
                },
            ],
            u32_args: vec!["__num_kv_pages".into()],
        };
        let src = emit_kernel("tk_kernel_smoke", &args, &p);

        // Sanity: kernel signature references all args.
        assert!(src.contains("__global__ __launch_bounds__(320) void tk_kernel_smoke("), "{src}");
        assert!(src.contains("const __nv_bfloat16* __restrict__ x"), "{src}");
        assert!(src.contains("__nv_bfloat16* __restrict__ out"), "{src}");
        assert!(src.contains("const uint32_t __num_kv_pages"), "{src}");

        // Aliases: `buf0`, `buf1` map to named args.
        assert!(src.contains("auto& buf0 = x;"), "{src}");
        assert!(src.contains("auto& buf1 = out;"), "{src}");

        // Role dispatch.
        assert!(src.contains("__role = ROLE_LOADER"), "{src}");
        assert!(src.contains("__role = ROLE_STORER"), "{src}");
        assert!(src.contains("__role = ROLE_CONSUMER"), "{src}");
        assert!(src.contains("const int __consumer_idx"), "{src}");

        // Mbarrier init.
        assert!(src.contains("__shared__ kittens::semaphore page_ready[13]"), "{src}");
        assert!(src.contains("__shared__ kittens::semaphore page_done[13]"), "{src}");
        assert!(src.contains("__shared__ kittens::semaphore page_consumed[13]"), "{src}");
        assert!(src.contains("kittens::init_semaphore(page_ready"), "{src}");
        assert!(src.contains("kittens::init_semaphore(page_done"), "{src}");
        assert!(src.contains("kittens::init_semaphore(page_consumed"), "{src}");
        assert!(src.contains("kittens::arrive(page_consumed[__i])"), "{src}");

        // CTA-wide sync after init (10 warps = 320 threads). TK 2.0
        // `group<N>::sync()` is barrier-less and asserts N==1, so for
        // multi-warp the scaffold uses `__syncthreads()`.
        assert!(src.contains("__syncthreads();"), "init+final sync\n{src}");

        // Body lands inside the kernel.
        assert!(src.contains("if (__role == ROLE_LOADER)"), "body merged\n{src}");
    }

    /// End-to-end smoke: lower one RmsNorm and emit a complete kernel.
    /// Snapshot test — write the .cu next to the test so we can `oc
    /// rsync` it to the pod and feed nvcc.
    #[test]
    fn end_to_end_rmsnorm_kernel_snapshot() {
        use crate::tk_lower::{lower_rmsnorm, PageAllocator, RmsNormOp};
        use crate::tk_warp_ir::Phase0;

        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_rmsnorm::<Phase0>(
            RmsNormOp {
                x: BufId(0),
                weight: BufId(1),
                out: BufId(2),
                hidden: 2048,
                m: 1,
                act_elem: 2,
                eps: 1e-5,
                init: true,
            },
            &mut pages,
            &mut prog,
        );

        let args = KernelArgs {
            bufs: vec![
                KernelArg {
                    ty: "const __nv_bfloat16* __restrict__".into(),
                    name: "x".into(),
                },
                KernelArg {
                    ty: "const __nv_bfloat16* __restrict__".into(),
                    name: "weight".into(),
                },
                KernelArg {
                    ty: "__nv_bfloat16* __restrict__".into(),
                    name: "out".into(),
                },
            ],
            u32_args: vec![],
        };
        let src = emit_kernel("tk_rmsnorm_decode_h2048", &args, &prog);

        // Sanity: the body's six-step handshake is inside the kernel.
        assert!(src.contains("kittens::group<1>::wait(page_consumed[0], 0)"), "{src}");
        assert!(src.contains("kittens::group<8>::wait(page_ready[0], 0)"), "{src}");
        assert!(src.contains("kittens::group<1>::wait(page_done[0], 0)"), "{src}");
        // Mbarrier init.
        assert!(src.contains("kittens::init_semaphore(page_ready[__i], 0, 1);"));
        assert!(src.contains("kittens::init_semaphore(page_done[__i], 0, 8);"));
        // Compute body.
        assert!(src.contains("rsqrtf"));
    }

    #[test]
    fn sync_group_width_matches_role() {
        let mut p = TkProgram::new();
        p.sync(WarpRole::AllConsumers);
        p.sync(WarpRole::Loader);
        let src = emit_body(&p);
        // 8 consumers → group<8>; loader is one warp → group<1>.
        assert!(src.contains("kittens::group<8>::sync();"), "{src}");
        assert!(src.contains("kittens::group<1>::sync();"), "{src}");
    }
}
