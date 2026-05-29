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

use crate::tk_warp_ir::{PageBarrier, TkInstr, TkProgram, TileShape, WarpRole, NUM_CONSUMER_WARPS};

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
    /// thread group the call is gated on.
    pub fn wait(n_warps: u32, kind: PageBarrier, page_id: u8, phase: u32) -> String {
        let bar = barrier_field(kind);
        format!("kittens::group<{n_warps}>::wait({bar}[{page_id}], {phase});")
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

    /// Lane-0-gated TMA load (`kittens::group<1>::tma::load_async`).
    /// Caller has already resolved `(rows, cols, elem_bytes)` from the
    /// IR's [`super::TileShape`]; this is just text.
    pub fn tma_load_async(
        page_id: u8,
        src_buf: u32,
        src_byte_off: u64,
        rows: u32,
        cols: u32,
        elem_bytes: u32,
    ) -> String {
        // The real TK 2.0 call binds a typed sv/st descriptor; the
        // shape numerics flow as named template arguments. We carry
        // them as comments here so an audit can compare against the
        // ff-mega-codegen rendering unambiguously.
        format!(
            "kittens::group<1>::tma::load_async(\
             page_buf[{page_id}], \
             buf{src_buf} /* +{src_byte_off} */, \
             {{ {rows}, {cols} }} /* elem_bytes={elem_bytes} */);"
        )
    }

    pub fn tma_store_async(
        page_id: u8,
        dst_buf: u32,
        dst_byte_off: u64,
        rows: u32,
        cols: u32,
        elem_bytes: u32,
    ) -> String {
        format!(
            "kittens::group<1>::tma::store_async(\
             buf{dst_buf} /* +{dst_byte_off} */, \
             page_buf[{page_id}], \
             {{ {rows}, {cols} }} /* elem_bytes={elem_bytes} */);"
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
    let (role, body) = match instr {
        TkInstr::Wait {
            role,
            page_id,
            kind,
            phase,
        } => {
            let n = role_group_width(*role);
            (*role, tk20::wait(n, *kind, *page_id, *phase))
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
                tk20::tma_load_async(*page_id, src.0, byte_off, rows, cols, elem_bytes),
            )
        }
        TkInstr::StoreAsync {
            page_id,
            dst,
            dst_region,
            tile,
        } => {
            let TileShape {
                rows,
                cols,
                elem_bytes,
            } = *tile;
            let byte_off = (dst_region.region.cols.start as u64) * (elem_bytes as u64);
            (
                WarpRole::Storer,
                tk20::tma_store_async(*page_id, dst.0, byte_off, rows, cols, elem_bytes),
            )
        }
        TkInstr::Compute { role, body } => (*role, body.clone()),
        TkInstr::Sync { role } => {
            let n = role_group_width(*role);
            (*role, tk20::sync(n))
        }
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

        // Loader: wait page consumed (phase 0), TMA load, arrive page ready.
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

        // Consumer: wait page ready (phase 1 — flipped), Compute, arrive done.
        let page = p.wait(WarpRole::AllConsumers, PageBarrier::Ready, page);
        p.compute(WarpRole::AllConsumers, "/* rms reduce + scale */");
        let page = p.arrive(WarpRole::AllConsumers, PageBarrier::Done, page);

        // Storer: wait done (phase 0), TMA store, arrive consumed.
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
        // what the type system computed.
        assert!(src.contains("page_consumed[0], 0"), "loader wait phase=0\n{src}");
        assert!(src.contains("page_ready[0], 1"), "consumer wait phase=1\n{src}");
        assert!(src.contains("page_done[0], 0"), "storer wait phase=0\n{src}");

        // Roles route correctly: loader/storer are gated on __role,
        // the consumer body is in the consumer arm.
        assert!(src.contains("if (__role == ROLE_LOADER)"), "loader gate\n{src}");
        assert!(src.contains("if (__role == ROLE_STORER)"), "storer gate\n{src}");
        assert!(src.contains("if (__role == ROLE_CONSUMER)"), "consumer gate\n{src}");
        assert!(src.contains("rms reduce + scale"), "compute body pasted\n{src}");
    }

    #[test]
    fn tma_load_carries_src_byte_offset() {
        let mut p = TkProgram::new();
        // src region starts at column 128, elem_bytes=2 → byte_off=256.
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
        assert!(src.contains("buf3 /* +256 */"), "byte off baked in\n{src}");
        assert!(src.contains("page_buf[5]"), "page id baked in\n{src}");
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
