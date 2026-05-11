// SPDX-License-Identifier: Apache-2.0
//! Host-interpreter codegen.
//!
//! Replaces the per-tile inlined `let X = kernel_call(...);` body
//! the old `emit_subgraph` produced. The new shape per arch:
//!
//! - One Rust enum codegened from the FUF the solver actually
//!   solved for that arch. Variants are exactly the kernel calls
//!   the picked Impls produce, plus `Free` (drop-pass).
//! - One `static FORWARD_M_<N>: &[<Arch>Op] = &[…];` per (variant
//!   × workload-point). Each element is a per-arch enum value.
//! - One per-arch interpreter — `for op in slice { match op { … }
//!   }` — closed and exhaustive over the per-arch enum.
//!
//! No universal opcode enum. No central registry. No string-keyed
//! opcode lookup. No `_` catch-all arm. No `unsafe { transmute }`
//! or `from_wire_unchecked`. The compiler enforces exhaustiveness
//! over the enum the macro just minted from this arch's solved FUF.
//!
//! See `HANDOFF_INTERPRETER.md` for the full design.
//!
//! # Megakernel concerns are out of scope
//!
//! Megakernel codegen is a separate code generator. When it lands,
//! it gets its own translator from per-arch enums to its own wire
//! format. This module does not produce `[i32; 32]` packed rows,
//! does not match KVM tp_throughput opcode numbering, does not
//! emit `Noop` padding. Those are megakernel concerns.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, HashSet};

use proc_macro2::TokenStream;
use quote::quote;

use crate::classified::Program;
use crate::config::ModelParams;
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::{ImplementationLibrary, MatchInfo, OpInstance, OpcodeShape, SlotMap};
use crate::schedule::Loop;
use crate::shape::Shape;
use crate::solver::{Assignment, SubgraphId};

// ── Slot allocation ──────────────────────────────────────────────

/// Build a dense slot allocation for every `(tile, output_slot)`
/// in the FUF, in topological tile-id order. Codegen passes
/// `&SlotMap` to every `Implementation::fan_out` so emitted
/// `OpInstance` field-value tokens carry resolved slot indices.
///
/// One slot per (tile, output_slot). Used when no liveness
/// information is available; the colored variant
/// [`colored_slot_map`] is what `lower_bucket` actually picks for
/// emission.
pub fn build_slot_map(fuf: &Fuf) -> SlotMap {
    let mut sm = SlotMap::new();
    for node in &fuf.nodes {
        let n_outputs = node.outputs.len().max(1) as u8;
        for slot in 0..n_outputs {
            sm.insert(node.id, slot);
        }
    }
    sm
}

/// Build a colored slot allocation via linear-scan register
/// allocation on the solved FUF. Two non-overlapping live ranges
/// share the same slot index — so layer L's `q` register and layer
/// L+1's `q` register collapse to one slot, and the per-layer body
/// emitted by [`lower_bucket`] becomes byte-identical across every
/// layer iteration. That's the precondition the layer-loop
/// detection relies on.
///
/// Constraints honored:
/// 1. **Shape partitioning.** Each color carries the shape of the
///    tiles it holds. A freed color goes back to *its shape's* free
///    pool; a tile of a different shape can never reuse it. This is
///    load-bearing for in-place mutation: `cutlass_gemm_add` writes
///    `[M, N]` into the residual buffer, so the buffer must have
///    been allocated for `[M, N]`. If a `[M, num_kv_heads, head_dim]`
///    K-tile and a `[M, hidden]` residual tile share a slot because
///    their lifetimes don't overlap, the K-tile's smaller buffer
///    survives into the residual op and the in-place write goes OOB.
/// 2. **Live-range overlap (within a shape).** Two same-shape slots
///    co-live iff one's def position ≤ the other's last-use position
///    and vice versa. Co-live slots within a shape get distinct
///    colors.
/// 3. **Same-shape aliasing collapses.** When `output_alias`
///    declares dst aliases owner AND `dst.shape == owner.shape`,
///    the dst is the same physical buffer (in-place mutation
///    semantics: the kernel mutates owner's buffer and downstream
///    consumers read it). The allocator pins dst to owner's color
///    — same `__tiles` slot, no `View` entry, no `Op::Alias` row.
/// 4. **Different-shape aliasing keeps a View.** `Reshape`-style
///    aliases (dst shape ≠ owner shape) point at the owner's
///    storage with new metadata; they need their own slot to hold
///    a `View`/`Reshaped` entry. Shape partitioning already places
///    them in a different free pool from the owner, so the runtime
///    `View(ref_slot=owner_slot)` indirection is non-trivial.
/// 5. **Consume.** `consumes_input_tiles` declares a slot whose
///    `OwnedTensor` migrates into the consumer's output. The
///    consumed slot's last-use is the consume site — past that it's
///    dead and its color is freed.
/// 6. **Protected slots** (the per-bucket fn's return tile, and the
///    backbone-output slot for `forward_backbone`) never have their
///    color reused — they must stay alive past the slice's end so
///    the per-bucket fn can `take_owned` them.
#[allow(clippy::too_many_arguments)]
pub fn colored_slot_map(
    fuf: &Fuf,
    sfuf: &Assignment,
    loop_ir: &Loop,
    lib: &ImplementationLibrary,
    skip_subgraph: Option<SubgraphId>,
    protected: &HashSet<(TileId, u8)>,
) -> SlotMap {
    use std::collections::BTreeSet;

    // Walk subgraphs in execution order; assign each one a position.
    let mut order: HashMap<SubgraphId, usize> = HashMap::new();
    let mut order_arr: Vec<SubgraphId> = Vec::new();
    for wave in &loop_ir.waves {
        for (sg, _) in &wave.subgraphs {
            if Some(*sg) == skip_subgraph {
                continue;
            }
            order.insert(*sg, order_arr.len());
            order_arr.push(*sg);
        }
    }

    // Alias: dst -> src (immediate). The slice's runtime View::ref_slot
    // points at the *flattened* owner, so we resolve chains for the
    // last-use extension below; for the color-distinction constraint
    // we use the resolved owner since that's what the View holds.
    let mut alias_to_owner: HashMap<(TileId, u8), (TileId, u8)> = HashMap::new();
    for &sg in &order_arr {
        let imp_id = sfuf
            .impl_of(sg)
            .expect("every scheduled subgraph has an Impl");
        let imp = lib.get(imp_id);
        let claimed = sfuf.tiles_in_subgraph(sg);
        for (dst, src_opt) in imp.output_alias(&claimed, fuf) {
            if let Some(src) = src_opt {
                alias_to_owner.insert(dst, src);
            }
        }
    }
    let resolve = |start: (TileId, u8)| -> (TileId, u8) {
        let mut cur = start;
        let mut seen = HashSet::new();
        while seen.insert(cur) {
            match alias_to_owner.get(&cur) {
                Some(up) => cur = *up,
                None => break,
            }
        }
        cur
    };

    // Consume: a slot whose `OwnedTensor` migrates into the consumer.
    // After the consume site the source slot is empty — its color is
    // freeable past consumer's position, regardless of any cross-
    // subgraph reads (the drop pass already excludes consumed slots).
    let mut consumed: HashSet<(TileId, u8)> = HashSet::new();
    for &sg in &order_arr {
        let imp_id = sfuf.impl_of(sg).expect("every subgraph has an Impl");
        let imp = lib.get(imp_id);
        let claimed = sfuf.tiles_in_subgraph(sg);
        for upstream in imp.consumes_input_tiles(&claimed, fuf) {
            consumed.insert(upstream);
        }
    }

    // Per-tile sub-positions. Each tile gets a unique flat index by
    // walking subgraphs in execution order, then tiles within each
    // subgraph in claim order. Coarser per-subgraph positions would
    // collapse every tile in a subgraph to the same `dp`, so a
    // producer tile whose only reader sits in the SAME subgraph's
    // claim ends up freed at its own def — and the next tile in the
    // subgraph reuses its color. Fine for single-kernel claims (the
    // producer's output is never materialized in the arena), but
    // wrong for storage-polymorphic impls whose `fan_out` emits
    // multiple kernels per subgraph (MetalFusedGateUpSiluMulImpl's
    // affine path: AffineQmm gate, AffineQmm up, SiluMul). Per-tile
    // sub-positions let the within-subgraph read walk below record
    // the consumer's sub-position as the producer's `last_use`, so
    // gate's slot stays distinct from up's slot across the SiluMul
    // read.
    //
    // Layer-template byte-equivalence: per-tile positions increment
    // monotonically. Each layer body's tiles occupy the same
    // relative offset range, so per-layer color assignments stay
    // byte-equivalent (the linear-scan reg allocation runs against
    // the same free-pool state at the same relative positions).
    let mut tile_position: HashMap<TileId, usize> = HashMap::new();
    let mut next_pos: usize = 0;
    for &sg in &order_arr {
        for tile in sfuf.tiles_in_subgraph(sg) {
            tile_position.insert(tile, next_pos);
            next_pos += 1;
        }
    }

    // Last use per OWNER (resolving alias chains): the latest tile-
    // position that reads this owner's storage, directly or via a
    // View. Within-subgraph reads count too (the impl's fan_out may
    // emit a kernel chain whose intermediate outputs hit the arena).
    // Last use of a non-owner (a View slot itself) is computed
    // separately below.
    let mut owner_last_use: HashMap<(TileId, u8), usize> = HashMap::new();
    for &sg in &order_arr {
        for tile in sfuf.tiles_in_subgraph(sg) {
            let consumer_pos = tile_position[&tile];
            for input in &fuf.get(tile).inputs {
                if let FufInput::Tile { id, slot } = input {
                    let owner = resolve((*id, *slot));
                    owner_last_use
                        .entry(owner)
                        .and_modify(|p| *p = (*p).max(consumer_pos))
                        .or_insert(consumer_pos);
                }
            }
        }
    }
    // View slots' last_use: when is the View itself read? A View is
    // read whenever its dst slot appears as a Tile input to some
    // downstream tile. Same walk but without alias resolution.
    let mut view_last_use: HashMap<(TileId, u8), usize> = HashMap::new();
    for &sg in &order_arr {
        for tile in sfuf.tiles_in_subgraph(sg) {
            let consumer_pos = tile_position[&tile];
            for input in &fuf.get(tile).inputs {
                if let FufInput::Tile { id, slot } = input {
                    if alias_to_owner.contains_key(&(*id, *slot)) {
                        view_last_use
                            .entry((*id, *slot))
                            .and_modify(|p| *p = (*p).max(consumer_pos))
                            .or_insert(consumer_pos);
                    }
                }
            }
        }
    }

    // Collect every (tile, output_slot) pair, sorted by per-tile
    // sub-position.
    let mut def_pos: HashMap<TileId, usize> = HashMap::new();
    for (&tile, &pos) in &tile_position {
        def_pos.insert(tile, pos);
    }
    let mut pairs: Vec<(usize, TileId, u8)> = Vec::new();
    for &sg in &order_arr {
        for tile in sfuf.tiles_in_subgraph(sg) {
            let n_out = fuf.get(tile).outputs.len().max(1) as u8;
            for slot in 0..n_out {
                pairs.push((tile_position[&tile], tile, slot));
            }
        }
    }
    pairs.sort_by_key(|&(p, t, s)| (p, t, s));

    // Linear-scan, partitioned by shape. Each color is born tagged
    // with the shape of the tile that minted it; a freed color
    // returns to *that* shape's pool. A tile of a different shape
    // never reuses it.
    //
    // Aliases: an `output_alias` declaration whose dst and resolved
    // owner share a shape pins the dst to the owner's color. These
    // are in-place mutation aliases (CutlassGemmAdd's add output is
    // the residual buffer; FusedAddRmsNorm's outputs are the
    // mutated upstream buffers). They literally share storage, so
    // they're the same `__tiles` slot — no `View` entry, no
    // `Op::Alias` row, no separate active entry.
    //
    // Aliases whose dst and owner have *different* shapes are
    // metadata-only (Reshape). They get their own slot in their
    // own shape pool; a `View { ref_slot: owner_slot }` entry
    // populated by `Op::Alias(dst_slot, owner_slot)` indirects to
    // the owner's storage.
    let mut active: Vec<(usize, u32, (TileId, u8))> = Vec::new();
    let mut free_colors_by_shape: HashMap<Shape, BTreeSet<u32>> = HashMap::new();
    let mut color_shape: HashMap<u32, Shape> = HashMap::new();
    let mut next_color: u32 = 0;
    let mut sm = SlotMap::new();

    for (dp, tile, slot) in pairs {
        // Free expired colors before allocating. `lu <= dp` is the
        // standard "use kills before def" semantics: a slot whose
        // last reader is the subgraph at `dp` dies after that
        // read — so when we're allocating outputs at `dp`, its
        // color is reusable. Without `<=`, layer 0's body would
        // differ from layer 1's because the embed slot wouldn't
        // be freed in time for layer 0's add output to take its
        // color, breaking byte-equivalence across layers.
        active.retain(|&(lu, color, _ts)| {
            if lu <= dp {
                let s = color_shape[&color].clone();
                free_colors_by_shape.entry(s).or_default().insert(color);
                false
            } else {
                true
            }
        });

        let tile_shape = fuf.get(tile).outputs[slot as usize].clone();
        let is_alias = alias_to_owner.contains_key(&(tile, slot));

        // Same-shape alias collapse: dst pins to owner's color.
        // No active entry (the owner's already covers the combined
        // lifetime via `owner_last_use` resolution).
        if is_alias {
            let owner = resolve((tile, slot));
            let owner_shape = fuf.get(owner.0).outputs[owner.1 as usize].clone();
            if owner_shape == tile_shape {
                let owner_color = sm.of(owner.0, owner.1);
                sm.insert_at(tile, slot, owner_color);
                continue;
            }
        }

        // Compute lu_self for the active entry.
        let lu_self = if is_alias {
            // Different-shape alias (Reshape view): dies after its
            // last reader.
            *view_last_use.get(&(tile, slot)).unwrap_or(&dp)
        } else {
            // Owner: dies at its own last_use, OR at consume site
            // (whichever is later — consume IS a use).
            let from_reads = *owner_last_use.get(&(tile, slot)).unwrap_or(&dp);
            // Protected slots stay alive forever. Consumed slots
            // (in-place consume pattern) end at the consume site,
            // which `owner_last_use` already records as a read —
            // so `from_reads` is correct for both consumed and
            // non-consumed cases. Only protection is special.
            if protected.contains(&(tile, slot)) {
                usize::MAX
            } else {
                let _ = &consumed; // documenting reliance on the walk above
                from_reads
            }
        };

        // Pick from this shape's free pool; mint a new color if empty.
        let pool = free_colors_by_shape.entry(tile_shape.clone()).or_default();
        let color = if let Some(&c) = pool.iter().next() {
            pool.remove(&c);
            c
        } else {
            let c = next_color;
            next_color += 1;
            color_shape.insert(c, tile_shape.clone());
            c
        };

        sm.insert_at(tile, slot, color);
        active.push((lu_self, color, (tile, slot)));
    }

    sm
}

// ── Per-bucket lowering ──────────────────────────────────────────

/// Output of lowering one (variant × workload-point). The codegen
/// stitches these into the per-bucket forward fn body.
pub struct LoweredBucket {
    /// Op instances in execution order, including `Free` rows
    /// emitted by the drop pass at the same scheduling points the
    /// old codegen would have emitted `drop()` statements.
    pub instances: Vec<OpInstance>,
    /// Total size of the runtime tile table for this bucket.
    pub num_slots: u32,
    /// Slot index whose `Owned` entry is the bucket fn's return
    /// value.
    pub final_slot: u32,
}

/// Variant shapes the macro accumulates across every bucket of one
/// arch. Used to drive shape-agreement checking + per-variant
/// iter-index field discovery for `apply_loop_compression`.
///
/// Bodies for each variant live in `ferrite_forward::Instruction::eval`;
/// nothing per-arch needs the body here.
#[derive(Default)]
pub struct ArchOpcodes {
    /// Variant ident → shape. First insert wins; later inserts of
    /// the same variant ident must agree on shape (codegen panics
    /// on mismatch).
    by_name: BTreeMap<String, OpcodeShape>,
}

impl ArchOpcodes {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a variant shape. Panics if the same variant ident is
    /// registered with a structurally different shape.
    pub fn register(&mut self, shape: OpcodeShape) {
        let key = shape.name.to_string();
        if let Some(existing_shape) = self.by_name.get(&key) {
            assert_shapes_agree(existing_shape, &shape);
            return;
        }
        self.by_name.insert(key, shape);
    }

    /// Snapshot of registered variant shapes keyed by variant ident
    /// string. Includes the universal `Alias` / `Free` / `Loop`
    /// variants. Consumed by [`emit_bucket_static_slice`] to
    /// type-check positional field values against the registered
    /// shape per bucket.
    pub fn shapes_by_name(&self) -> BTreeMap<String, OpcodeShape> {
        let mut out: BTreeMap<String, OpcodeShape> = self.by_name.clone();
        out.insert("Alias".to_string(), alias_variant_shape());
        out.insert("Free".to_string(), free_variant_shape());
        out.insert("Loop".to_string(), loop_variant_shape());
        out
    }

    /// Iterate (variant_name, shape). Used by
    /// `apply_loop_compression` to build the per-variant layer-field
    /// position map.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &OpcodeShape)> {
        self.by_name.iter()
    }
}

/// Find the largest contiguous run of instances that can be
/// described as N copies of a P-instruction body, optionally
/// allowing a per-variant ITERATION-INDEX field to step linearly
/// (by exactly 1) between copies. Returns `Some((start, period,
/// num_iters))` on success, `None` when no such run exists.
///
/// This is generic loop detection — there's no semantic notion of
/// "layer" in here. The caller passes
/// `iter_index_field_per_variant`: for any variant where one of
/// its fields' value forms `c, c+1, c+2, …` across consecutive
/// candidate iterations, that field's index is in the map and
/// gets compared modulo iter offset; every other field is
/// compared byte-exactly. Variants not in the map have all fields
/// compared byte-exactly.
///
/// Algorithm: O((n × max_period) × (avg_iters × period_check_cost)).
/// `period_check_cost` is `O(P)` of pre-hashed fingerprint
/// equality + integer-add equality for iter-index fields.
fn detect_repeating_run(
    instances: &[OpInstance],
    iter_index_field_per_variant: &std::collections::HashMap<String, usize>,
) -> Option<(usize, usize, u32)> {
    let n = instances.len();
    if n < 2 {
        return None;
    }

    // Precompute per-instance: a fingerprint string covering only
    // the byte-exact-compared parts (variant ident + every field
    // that ISN'T the iter-index). Pre-hashed once so the inner
    // pattern-match loop is integer compare instead of repeated
    // TokenStream-to-String formatting (the n³ blow-up).
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let fp_and_iter: Vec<(u64, Option<u32>)> = instances
        .iter()
        .map(|inst| {
            let var_name = inst.name.to_string();
            let iter_field = iter_index_field_per_variant.get(&var_name).copied();
            let mut hasher = DefaultHasher::new();
            var_name.hash(&mut hasher);
            for (i, fv) in inst.field_values.iter().enumerate() {
                if Some(i) == iter_field {
                    continue;
                }
                fv.to_string().hash(&mut hasher);
            }
            let iter_val = iter_field.and_then(|i| parse_u32_literal(&inst.field_values[i]));
            (hasher.finish(), iter_val)
        })
        .collect();

    let mut best: Option<(usize, usize, u32, usize)> = None; // (start, period, iters, span)

    for start in 0..n {
        let max_period = (n - start) / 2;
        for period in 1..=max_period {
            // Quick reject: fingerprint of [start..start+P] must
            // equal fingerprint of [start+P..start+2P].
            let block_a = &fp_and_iter[start..start + period];
            let block_b = &fp_and_iter[start + period..start + 2 * period];
            if !blocks_match(block_a, block_b, 1) {
                continue;
            }
            // Confirmed at least 2 iterations. Try extending.
            let mut iters = 2u32;
            loop {
                let next_start = start + iters as usize * period;
                if next_start + period > n {
                    break;
                }
                let block_n = &fp_and_iter[next_start..next_start + period];
                if !blocks_match(block_a, block_n, iters) {
                    break;
                }
                iters += 1;
            }
            let span = period * iters as usize;
            let cand = (start, period, iters, span);
            if best.is_none_or(|b| cand.3 > b.3) {
                best = Some(cand);
            }
        }
    }

    best.map(|(s, p, n, _)| (s, p, n))
}

/// Two blocks of pre-hashed (fingerprint, iter_index_value) pairs
/// match iff the fingerprints are equal pairwise AND the
/// iter-index values, when present, satisfy `cand = base +
/// iter_offset`.
fn blocks_match(
    base: &[(u64, Option<u32>)],
    cand: &[(u64, Option<u32>)],
    iter_offset: u32,
) -> bool {
    if base.len() != cand.len() {
        return false;
    }
    for (b, c) in base.iter().zip(cand.iter()) {
        if b.0 != c.0 {
            return false;
        }
        match (b.1, c.1) {
            (Some(bv), Some(cv)) if cv == bv + iter_offset => {}
            (None, None) => {}
            _ => return false,
        }
    }
    true
}

/// Parse a TokenStream that looks like `<n>u32` or `<n>` (un-
/// suffixed) into a `u32`. Returns None for anything else
/// (function-item paths, expressions, etc.).
fn parse_u32_literal(ts: &TokenStream) -> Option<u32> {
    let s = ts.to_string();
    let s = s.trim();
    let s = s.strip_suffix("u32").unwrap_or(s);
    s.parse::<u32>().ok()
}

/// Apply loop compression to `lowered.instances` in place. When
/// [`detect_repeating_run`] finds a contiguous run, replace it
/// with one `Op::Loop` row plus a single iteration's body. The
/// iteration-index field on each body row is set to that row's
/// **per-row baseline** — the value the field had in iter 0 at
/// that row's position — instead of zeroed. The interpreter passes
/// the runtime iteration counter as `__layer`, and arm bodies
/// compute `let layer: u32 = __layer + layer;` so each row's
/// effective index is `__l + baseline`.
///
/// Per-row baselines are load-bearing because the body period can
/// span a layer boundary: in Llama, the body's trailing
/// `FusedAddRmsNorm(input_layernorm)` IS the *next* layer's
/// input_ln (fused with the residual add); its iter-0 layer is 1,
/// not 0, while the body's other rows have iter-0 layer 0.
/// Zeroing all rows to 0 collapses every iteration's input_ln to
/// `__l` — i.e., uses layer N's weights when computing layer N+1's
/// input ln. Numerical drift accumulates and decode turns to
/// garbage after a few tokens.
///
/// `iter_index_field_name` is the field name a variant uses to
/// carry its iteration-index value. Today's only such name is
/// `"layer"` (transformer-layer index); the function takes the
/// name as a parameter so the same code works for any future
/// loop construct (e.g., per-head, per-block) that adds a
/// different convention.
pub fn apply_loop_compression(
    arch_opcodes: &ArchOpcodes,
    lowered: &mut LoweredBucket,
    iter_index_field_name: &str,
) {
    use std::collections::HashMap;

    let mut iter_idx: HashMap<String, usize> = HashMap::new();
    for (name, shape) in arch_opcodes.iter() {
        for (i, (fname, _ty)) in shape.fields.iter().enumerate() {
            if fname == iter_index_field_name {
                iter_idx.insert(name.clone(), i);
                break;
            }
        }
    }

    let Some((start, period, iters)) = detect_repeating_run(&lowered.instances, &iter_idx) else {
        return;
    };

    let span_end = start + period * iters as usize;
    let mut new_instances: Vec<OpInstance> = Vec::new();
    new_instances.extend_from_slice(&lowered.instances[..start]);
    new_instances.push(loop_instance(iters, period as u32));
    for inst in &lowered.instances[start..start + period] {
        let mut copy = inst.clone();
        if let Some(&fi) = iter_idx.get(&inst.name.to_string()) {
            // Preserve the iter-0 baseline per row — the arm
            // computes `layer = __layer + baseline` at dispatch.
            let baseline = parse_u32_literal(&inst.field_values[fi]).unwrap_or(0);
            let lit = proc_macro2::Literal::u32_suffixed(baseline);
            copy.field_values[fi] = quote! { #lit };
        }
        new_instances.push(copy);
    }
    new_instances.extend_from_slice(&lowered.instances[span_end..]);
    lowered.instances = new_instances;
}

/// Emit one per-bucket
/// `static <ident>: &[__I] = &[…];` where `__I` is the per-canonical
/// alias for `::ferrite_forward::Instruction<Weights>`. Each row is
/// `__I::<Variant>(v0, v1, …)` — tuple-style construction matching
/// the variant declaration order in `Instruction<W>`. The match
/// between `OpInstance::field_values` order and `Instruction`
/// variant tuple order is enforced by the per-Impl
/// `OpcodeShape::fields` declaration (the codegen contract).
pub fn emit_bucket_static_slice(
    static_ident: &syn::Ident,
    shapes_by_name: &BTreeMap<String, OpcodeShape>,
    instances: &[OpInstance],
) -> TokenStream {
    let elements = instances.iter().map(|inst| {
        let var = &inst.name;
        let shape: &OpcodeShape = shapes_by_name
            .get(&inst.name.to_string())
            .unwrap_or_else(|| {
                panic!(
                    "OpInstance variant `{}` has no registered OpcodeShape — \
                     codegen invariant violated",
                    inst.name
                )
            });
        assert_eq!(
            shape.fields.len(),
            inst.field_values.len(),
            "OpInstance `{}`: field_values.len()={} but shape.fields.len()={}",
            inst.name,
            inst.field_values.len(),
            shape.fields.len()
        );
        let exprs = inst
            .field_values
            .iter()
            .map(|e| strip_int_suffixes(e.clone()));
        quote! {
            #var ( #(#exprs),* )
        }
    });
    quote! {
        static #static_ident: &[__I] = &[ #(#elements),* ];
    }
}

/// Strip integer-type suffixes (`u32`, `usize`, `u8`, `i32`, …)
/// from numeric literals in `ts`. The variant declaration in
/// `Instruction<W>` already pins the type; the suffix is redundant
/// and costs ~3-5 chars per slot/layer/flag field × thousands of
/// rows in cargo expand.
fn strip_int_suffixes(ts: TokenStream) -> TokenStream {
    use proc_macro2::{Group, Literal, TokenTree};
    let mut out = TokenStream::new();
    for tt in ts {
        match tt {
            TokenTree::Group(g) => {
                let inner = strip_int_suffixes(g.stream());
                let mut new_group = Group::new(g.delimiter(), inner);
                new_group.set_span(g.span());
                out.extend(std::iter::once(TokenTree::Group(new_group)));
            }
            TokenTree::Literal(lit) => {
                let s = lit.to_string();
                if let Some(stripped) = strip_int_suffix_str(&s)
                    && let Ok(u) = stripped.parse::<u64>()
                {
                    let mut new_lit = Literal::u64_unsuffixed(u);
                    new_lit.set_span(lit.span());
                    out.extend(std::iter::once(TokenTree::Literal(new_lit)));
                    continue;
                }
                out.extend(std::iter::once(TokenTree::Literal(lit)));
            }
            other => out.extend(std::iter::once(other)),
        }
    }
    out
}

fn strip_int_suffix_str(s: &str) -> Option<&str> {
    for suf in &[
        "usize", "isize", "u128", "i128", "u64", "i64", "u32", "i32", "u16", "i16", "u8", "i8",
    ] {
        if let Some(rest) = s.strip_suffix(suf) {
            return Some(rest);
        }
    }
    None
}

// ── Helpers ──────────────────────────────────────────────────────

fn assert_shapes_agree(a: &OpcodeShape, b: &OpcodeShape) {
    assert_eq!(
        a.name, b.name,
        "OpcodeShape registration: variant ident mismatch ({} vs {})",
        a.name, b.name
    );
    assert_eq!(
        a.fields.len(),
        b.fields.len(),
        "OpcodeShape `{}`: field count mismatch ({} vs {})",
        a.name,
        a.fields.len(),
        b.fields.len()
    );
    for ((an, at), (bn, bt)) in a.fields.iter().zip(b.fields.iter()) {
        assert_eq!(
            an, bn,
            "OpcodeShape `{}`: field name mismatch ({} vs {})",
            a.name, an, bn
        );
        let a_ty = quote! { #at }.to_string();
        let b_ty = quote! { #bt }.to_string();
        assert_eq!(
            a_ty, b_ty,
            "OpcodeShape `{}` field `{}`: type mismatch ({} vs {})",
            a.name, an, a_ty, b_ty
        );
    }
}

/// The universal `Free` variant codegen always emits. Not Impl-
/// driven; emitted at the drop-pass-determined scheduling points
/// to clear a tile slot.
pub fn free_variant_shape() -> OpcodeShape {
    OpcodeShape::new("Free", vec![("slot", syn::parse_quote!(u32))])
}

/// Construct a `Free(slot)` instance the drop pass can emit.
pub fn free_instance(slot: u32) -> OpInstance {
    OpInstance::new(
        syn::Ident::new("Free", proc_macro2::Span::call_site()),
        vec![{
            let lit = proc_macro2::Literal::u32_unsuffixed(slot);
            quote! { #lit }
        }],
    )
}

/// The universal `Alias` variant codegen emits at the start of
/// every per-bucket slice — one row per zero-copy `View` aliasing
/// pair the lowering surfaced via `output_alias`. The interpreter's
/// arm sets `__tiles[dst] = Some(view(src))`, the same setup the
/// per-bucket fn used to do as a separate prelude. Folding aliases
/// into the slice means there's no per-fn prelude duplication
/// between forward and forward_backbone.
pub fn alias_variant_shape() -> OpcodeShape {
    OpcodeShape::new(
        "Alias",
        vec![
            ("dst", syn::parse_quote!(u32)),
            ("src", syn::parse_quote!(u32)),
        ],
    )
}

/// Construct an `Alias(dst, src)` instance for the alias prelude.
pub fn alias_instance(dst: u32, src: u32) -> OpInstance {
    OpInstance::new(
        syn::Ident::new("Alias", proc_macro2::Span::call_site()),
        vec![
            {
                let lit = proc_macro2::Literal::u32_unsuffixed(dst);
                quote! { #lit }
            },
            {
                let lit = proc_macro2::Literal::u32_unsuffixed(src);
                quote! { #lit }
            },
        ],
    )
}

/// The universal `Loop` variant the layer-template detection
/// emits when a subsequence of the slice repeats N times. The
/// interpreter sees `Op::Loop(count, body_len)` and runs the
/// next `body_len` ops `count` times, threading the iteration
/// index through as `__layer`. Compresses a 40-layer transformer
/// body from 40·body rows to body+1 rows.
pub fn loop_variant_shape() -> OpcodeShape {
    OpcodeShape::new(
        "Loop",
        vec![
            ("count", syn::parse_quote!(u32)),
            ("body_len", syn::parse_quote!(u32)),
        ],
    )
}

/// Construct a `Loop(count, body_len)` instance the layer-template
/// detection prepends in front of a repeating sub-sequence of the
/// slice.
pub fn loop_instance(count: u32, body_len: u32) -> OpInstance {
    OpInstance::new(
        syn::Ident::new("Loop", proc_macro2::Span::call_site()),
        vec![
            {
                let lit = proc_macro2::Literal::u32_unsuffixed(count);
                quote! { #lit }
            },
            {
                let lit = proc_macro2::Literal::u32_unsuffixed(body_len);
                quote! { #lit }
            },
        ],
    )
}

// ── Bucket lowering driver ───────────────────────────────────────

/// Lower one (variant × workload-point) into [`LoweredBucket`].
/// Walks the same wave/loop the old codegen did, calls each picked
/// Impl's `fan_out`, interleaves `Free` instances at drop-pass
/// scheduling points, and registers each Impl's `OpcodeShape` into
/// `arch_opcodes` for shape-checking + iter-index discovery.
///
/// `final_tile` is the `(TileId, output_slot)` whose slot index will
/// be exposed as `LoweredBucket.final_slot`. The full forward passes
/// `(fuf.last(), 0)`; the backbone-only forward passes the input of
/// the skipped terminal subgraph. The drop pass is told to protect
/// this slot via the caller's `protected` set, since the per-bucket
/// fn `take_owned`s it as the return value.
#[allow(clippy::too_many_arguments)]
pub fn lower_bucket(
    fuf: &Fuf,
    sfuf: &Assignment,
    loop_ir: &Loop,
    program: &Program,
    _model: &ModelParams,
    lib: &ImplementationLibrary,
    bounds: &BTreeMap<String, u64>,
    skip_subgraph: Option<SubgraphId>,
    // `protected` is consumed by the *caller's* `colored_slot_map`
    // call (which also produces `slots`), not by the lowering walk
    // itself — without an explicit Free pass, `lower_bucket` only
    // emits Alias rows + per-Impl fan_out output, neither of which
    // needs to know which slots are pinned beyond slice end. Kept
    // in the signature so calls stay symmetric with `colored_slot_map`.
    _protected: &HashSet<(TileId, u8)>,
    arch_opcodes: &mut ArchOpcodes,
    final_tile: (TileId, u8),
    slots: &SlotMap,
) -> LoweredBucket {
    let num_slots = slots.total();

    // Aliases: every Impl's output_alias declares which of its
    // outputs borrow from an upstream owner.
    let mut alias_to_owner: HashMap<(TileId, u8), (TileId, u8)> = HashMap::new();
    for wave in &loop_ir.waves {
        for (sg, imp_id) in &wave.subgraphs {
            if Some(*sg) == skip_subgraph {
                continue;
            }
            let claimed = sfuf.tiles_in_subgraph(*sg);
            let imp = lib.get(*imp_id);
            for (dst, src_opt) in imp.output_alias(&claimed, fuf) {
                if let Some(src) = src_opt {
                    alias_to_owner.insert(dst, src);
                }
            }
        }
    }
    let resolve_owner = |start: (TileId, u8)| -> (TileId, u8) {
        let mut cur = start;
        let mut seen = HashSet::new();
        while seen.insert(cur) {
            match alias_to_owner.get(&cur) {
                Some(up) => cur = *up,
                None => break,
            }
        }
        cur
    };
    let mut aliases: Vec<(u32, u32)> = alias_to_owner
        .iter()
        .map(|(&dst, _)| {
            let owner = resolve_owner(dst);
            (slots.of(dst.0, dst.1), slots.of(owner.0, owner.1))
        })
        .filter(|(d, s)| d != s)
        .collect();
    aliases.sort();
    aliases.dedup();

    // Prepend Alias rows so the slice is self-contained: running it
    // sets up the zero-copy views, runs the kernels, and frees on its
    // own — no per-bucket fn alias prelude.
    //
    // Note: with `colored_slot_map`, the slot map already collapses
    // dead slots' colors into the free pool the moment they expire.
    // The next writer to that color overwrites the slot's
    // `Some(OwnedTensor)` — Rust drops the old tensor at the
    // overwrite, returning its GPU memory to the caching allocator
    // automatically. Explicit `Op::Free` rows would be redundant
    // (the next write does the same drop) and actively harmful for
    // the layer-template invariant (a Free emitted in layer L but
    // not layer L+1 — because in layer L the color isn't reused
    // before exit, but in layer L+1 it is — would break body byte-
    // equivalence). So we don't emit Free here at all.
    let mut instances: Vec<OpInstance> =
        aliases.iter().map(|&(d, s)| alias_instance(d, s)).collect();

    for wave in &loop_ir.waves {
        for (sg, imp_id) in &wave.subgraphs {
            if Some(*sg) == skip_subgraph {
                continue;
            }
            let claimed = sfuf.tiles_in_subgraph(*sg);
            let imp = lib.get(*imp_id);
            let m = MatchInfo {
                claimed_tiles: claimed.clone(),
                boundary_inputs: collect_boundary_inputs(fuf, &claimed),
                boundary_outputs: claimed.clone(),
            };
            let emits = imp
                .fan_out(&m, fuf, program, bounds, slots)
                .unwrap_or_else(|| {
                    panic!(
                        "Impl `{name}` (id {id}) has no fan_out — unmigrated to host \
                         interpreter IR. Override `opcode_shape` + `fan_out` on \
                         `{name}`, and ensure the matching `Instruction<W>` variant \
                         exists in `ferrite_forward::instr`.",
                        name = imp.name(),
                        id = imp_id.0,
                    )
                });
            // Eval bodies live in `ferrite_forward::Instruction::eval`
            // — register only the shape, used for static-slice
            // emission and `apply_loop_compression`'s per-variant
            // iter-index field discovery. `extra_opcode_shapes`
            // covers storage-polymorphic impls that fan out a
            // multi-variant mix (e.g. metal int4's decomposed q-MLP
            // emits `AffineQmm`/`SiluMul` from the same Impl whose
            // primary `opcode_shape` is `FusedGateUpSiluMul`).
            arch_opcodes.register(imp.opcode_shape());
            for extra in imp.extra_opcode_shapes() {
                arch_opcodes.register(extra);
            }
            instances.extend(emits);
        }
    }

    let final_slot = slots.of(final_tile.0, final_tile.1);

    LoweredBucket {
        instances,
        num_slots,
        final_slot,
    }
}

pub fn collect_boundary_inputs(fuf: &Fuf, claimed: &[TileId]) -> Vec<TileId> {
    let claimed_set: HashSet<TileId> = claimed.iter().copied().collect();
    let mut seen: HashSet<TileId> = HashSet::new();
    let mut out = Vec::new();
    for &t in claimed {
        for input in &fuf.get(t).inputs {
            if let FufInput::Tile { id, .. } = input
                && !claimed_set.contains(id)
                && seen.insert(*id)
            {
                out.push(*id);
            }
        }
    }
    out
}

type FreePlan = HashMap<SubgraphId, Vec<(TileId, u8)>>;

#[allow(clippy::too_many_arguments)]
fn compute_free_points(
    fuf: &Fuf,
    sfuf: &Assignment,
    loop_ir: &Loop,
    lib: &ImplementationLibrary,
    skip_subgraph: Option<SubgraphId>,
    protected: &HashSet<(TileId, u8)>,
    alias_to_owner: &HashMap<(TileId, u8), (TileId, u8)>,
) -> FreePlan {
    let mut order: HashMap<SubgraphId, usize> = HashMap::new();
    let mut next = 0;
    for wave in &loop_ir.waves {
        for (sg, _) in &wave.subgraphs {
            order.insert(*sg, next);
            next += 1;
        }
    }

    let mut consumed: HashSet<(TileId, u8)> = HashSet::new();
    for wave in &loop_ir.waves {
        for (sg, imp_id) in &wave.subgraphs {
            if Some(*sg) == skip_subgraph {
                continue;
            }
            let claimed = sfuf.tiles_in_subgraph(*sg);
            let imp = lib.get(*imp_id);
            for upstream in imp.consumes_input_tiles(&claimed, fuf) {
                consumed.insert(upstream);
            }
        }
    }

    let resolve = |start: (TileId, u8)| -> (TileId, u8) {
        let mut cur = start;
        let mut seen = HashSet::new();
        while seen.insert(cur) {
            match alias_to_owner.get(&cur) {
                Some(up) => cur = *up,
                None => break,
            }
        }
        cur
    };

    let mut last_use: HashMap<(TileId, u8), SubgraphId> = HashMap::new();
    for wave in &loop_ir.waves {
        for (sg, _) in &wave.subgraphs {
            if Some(*sg) == skip_subgraph {
                continue;
            }
            let claimed: HashSet<TileId> = sfuf.tiles_in_subgraph(*sg).into_iter().collect();
            for tile in &claimed {
                for input in &fuf.get(*tile).inputs {
                    if let FufInput::Tile { id, slot } = input {
                        if claimed.contains(id) {
                            continue;
                        }
                        let owner = resolve((*id, *slot));
                        let new_pos = order[sg];
                        let keep = match last_use.get(&owner) {
                            Some(prev) => order[prev] < new_pos,
                            None => true,
                        };
                        if keep {
                            last_use.insert(owner, *sg);
                        }
                    }
                }
            }
        }
    }

    let mut plan: FreePlan = HashMap::new();
    for (owner, sg) in last_use {
        if protected.contains(&owner) {
            continue;
        }
        if consumed.contains(&owner) {
            continue;
        }
        plan.entry(sg).or_default().push(owner);
    }
    for v in plan.values_mut() {
        v.sort();
    }
    plan
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuf::FufNode;
    use crate::shape::Dim;
    use quote::format_ident;

    /// Slot allocation packs (tile, output_slot) pairs in
    /// topological order. Test pins the order so emitted
    /// constructors and runtime indexing line up.
    #[test]
    fn slot_map_packs_fuf_outputs_densely() {
        let f = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: crate::classified::OpKind::Add,
                    inputs: vec![],
                    outputs: vec![vec![Dim::Lit(1)]],
                },
                FufNode {
                    id: TileId(1),
                    op: crate::classified::OpKind::Reshape,
                    inputs: vec![],
                    outputs: vec![vec![Dim::Lit(1)], vec![Dim::Lit(1)], vec![Dim::Lit(1)]],
                },
                FufNode {
                    id: TileId(2),
                    op: crate::classified::OpKind::Add,
                    inputs: vec![],
                    outputs: vec![vec![Dim::Lit(1)]],
                },
            ],
        };
        let sm = build_slot_map(&f);
        assert_eq!(sm.total(), 5);
        assert_eq!(sm.of(TileId(0), 0), 0);
        assert_eq!(sm.of(TileId(1), 0), 1);
        assert_eq!(sm.of(TileId(1), 1), 2);
        assert_eq!(sm.of(TileId(1), 2), 3);
        assert_eq!(sm.of(TileId(2), 0), 4);
    }

    // ── Coloring invariants ─────────────────────────────────────
    //
    // These tests construct synthetic FUFs + Assignments + Loops +
    // a tiny test-only ImplementationLibrary, run `colored_slot_map`,
    // and assert structural properties of the resulting SlotMap.
    // They're independent of any real arch's DSL: the point is to
    // pin invariants the colorer must maintain regardless of what
    // gets lowered through it.

    use crate::classified::OpKind;
    use crate::fuf::FufInput;
    use crate::impl_lib::{
        CostCtx, Handoff, ImplId, ImplementationLibrary, LaunchKind, Layout, MatchInfo, Resources,
        WeightAccessor,
    };
    use crate::schedule::{Loop, Wave};
    use crate::solver::{Assignment, SubgraphId};
    use crate::target::TargetProfile;
    use std::collections::HashSet;

    /// A test-only Impl that claims exactly one tile per subgraph and
    /// reports configurable `output_alias` / `consumes_input_tiles`.
    /// Lets the synthetic-FUF tests below exercise alias and consume
    /// constraints on the colorer without dragging in any real
    /// kernel-emitting Impl.
    #[derive(Debug)]
    struct StubImpl {
        name: &'static str,
        alias_to: Option<(TileId, u8)>,
        consumes: Vec<(TileId, u8)>,
    }

    impl crate::impl_lib::Implementation for StubImpl {
        fn name(&self) -> &'static str {
            self.name
        }
        fn target_compatible(&self, _profile: &TargetProfile) -> bool {
            true
        }
        fn matches(
            &self,
            _fuf: &Fuf,
            _seed: TileId,
            _profile: &TargetProfile,
        ) -> Option<MatchInfo> {
            None
        }
        fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
            0.0
        }
        fn resources(&self, _m: &MatchInfo) -> Resources {
            Resources::ZERO
        }
        fn launch_kind(&self) -> LaunchKind {
            LaunchKind::HostCallback
        }
        fn supported_input_handoffs(&self) -> &[Handoff] {
            &[]
        }
        fn supported_output_handoffs(&self) -> &[Handoff] {
            &[]
        }
        fn input_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
            vec![]
        }
        fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
            vec![]
        }
        fn required_weights(
            &self,
            _claimed: &[TileId],
            _fuf: &Fuf,
            _program: &crate::classified::Program,
        ) -> Vec<WeightAccessor> {
            vec![]
        }
        fn output_alias(
            &self,
            claimed_tiles: &[TileId],
            _fuf: &Fuf,
        ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
            // Single-tile claim — declare the alias-source for slot 0
            // if configured, else default (None = own OwnedTensor).
            let t = claimed_tiles[0];
            vec![((t, 0), self.alias_to)]
        }
        fn consumes_input_tiles(&self, _claimed: &[TileId], _fuf: &Fuf) -> Vec<(TileId, u8)> {
            self.consumes.clone()
        }
    }

    /// Build a single-output Add tile with the given upstream Tile
    /// inputs (each as `(producer_tile, output_slot)`).
    fn add_tile(id: u32, inputs: &[(TileId, u8)]) -> FufNode {
        FufNode {
            id: TileId(id),
            op: OpKind::Add,
            inputs: inputs
                .iter()
                .map(|&(id, slot)| FufInput::Tile { id, slot })
                .collect(),
            outputs: vec![vec![Dim::Lit(1)]],
        }
    }

    /// Build a single-tile per-subgraph Assignment with one
    /// Impl per subgraph, in tile-id order.
    fn linear_assignment(tiles: &[TileId], impls: &[ImplId]) -> Assignment {
        assert_eq!(tiles.len(), impls.len());
        let mut a = Assignment::default();
        for (i, &t) in tiles.iter().enumerate() {
            let sg = SubgraphId(i as u32);
            a.cover.insert(t, sg);
            a.impls.insert(sg, impls[i]);
        }
        a
    }

    /// Linear schedule: one wave per subgraph, in id order.
    fn linear_loop(n_subgraphs: u32) -> Loop {
        let waves = (0..n_subgraphs)
            .map(|i| Wave {
                subgraphs: vec![(SubgraphId(i), ImplId(0))], // ImplId here is unused by colorer
            })
            .collect();
        Loop { waves }
    }

    /// 3-node chain `t0 → t1 → t2`, no aliases. With `lu <= dp`
    /// kill-before-def semantics, every tile lands at the same slot:
    /// t0's color is freed at pos-1 (t1's def), t1 reuses it; same
    /// at pos-2. The kernel call's read of slot 0 happens before
    /// the overwrite (`__tiles[0] = Some(new)` drops the old
    /// `OwnedTensor` only after the kernel was already queued
    /// against its view), so 1 color is correct here.
    #[test]
    fn coloring_chain_collapses_to_one_color() {
        let f = Fuf {
            nodes: vec![
                add_tile(0, &[]),
                add_tile(1, &[(TileId(0), 0)]),
                add_tile(2, &[(TileId(1), 0)]),
            ],
        };
        let mut lib = ImplementationLibrary::new();
        let id_plain = lib.push(Box::new(StubImpl {
            name: "stub",
            alias_to: None,
            consumes: vec![],
        }));
        let sfuf = linear_assignment(&[TileId(0), TileId(1), TileId(2)], &[id_plain; 3]);
        let lp = linear_loop(3);
        let protected: HashSet<(TileId, u8)> = HashSet::new();
        let sm = colored_slot_map(&f, &sfuf, &lp, &lib, None, &protected);

        assert_eq!(sm.total(), 1, "serial chain coalesces to one slot");
        assert_eq!(sm.of(TileId(0), 0), sm.of(TileId(1), 0));
        assert_eq!(sm.of(TileId(1), 0), sm.of(TileId(2), 0));
    }

    /// A diamond `t0 → t1, t0 → t2, then t3 = f(t1, t2)` — t1 and
    /// t2 must coexist (both read t0; both written before t3 reads
    /// them) so they get distinct colors. Pins the must-not-collapse
    /// half of the colorer's job.
    #[test]
    fn coloring_diamond_keeps_concurrent_outputs_distinct() {
        let f = Fuf {
            nodes: vec![
                add_tile(0, &[]),
                add_tile(1, &[(TileId(0), 0)]),
                add_tile(2, &[(TileId(0), 0)]),
                add_tile(3, &[(TileId(1), 0), (TileId(2), 0)]),
            ],
        };
        let mut lib = ImplementationLibrary::new();
        let id_plain = lib.push(Box::new(StubImpl {
            name: "stub",
            alias_to: None,
            consumes: vec![],
        }));
        let sfuf = linear_assignment(
            &[TileId(0), TileId(1), TileId(2), TileId(3)],
            &[id_plain; 4],
        );
        let lp = linear_loop(4);
        let protected: HashSet<(TileId, u8)> = HashSet::new();
        let sm = colored_slot_map(&f, &sfuf, &lp, &lib, None, &protected);

        // t1 and t2 are co-live at the moment t2 is being written
        // (t1 was just written, t2 is being written, both must
        // remain in __tiles for t3 to read). Distinct colors.
        assert_ne!(sm.of(TileId(1), 0), sm.of(TileId(2), 0));
    }

    /// A `protected` slot's color must NEVER appear on any other
    /// tile — the per-bucket fn's `take_owned(final_slot)` runs at
    /// the very end and would corrupt other tiles if their colors
    /// collided with `final_slot`. The colorer keeps protected
    /// colors out of the free pool forever; here that means t0's
    /// color is unique even though t0 has no readers past pos-1.
    /// (t1 and t2 may still share a color with each other — that's
    /// fine.)
    #[test]
    fn coloring_protected_slot_color_is_unique() {
        let f = Fuf {
            nodes: vec![
                add_tile(0, &[]),
                add_tile(1, &[(TileId(0), 0)]),
                add_tile(2, &[(TileId(1), 0)]),
            ],
        };
        let mut lib = ImplementationLibrary::new();
        let id_plain = lib.push(Box::new(StubImpl {
            name: "stub",
            alias_to: None,
            consumes: vec![],
        }));
        let sfuf = linear_assignment(&[TileId(0), TileId(1), TileId(2)], &[id_plain; 3]);
        let lp = linear_loop(3);
        let mut protected: HashSet<(TileId, u8)> = HashSet::new();
        protected.insert((TileId(0), 0));
        let sm = colored_slot_map(&f, &sfuf, &lp, &lib, None, &protected);

        let c_protected = sm.of(TileId(0), 0);
        let c_t1 = sm.of(TileId(1), 0);
        let c_t2 = sm.of(TileId(2), 0);
        assert_ne!(
            c_protected, c_t1,
            "protected color must not be reused by t1"
        );
        assert_ne!(
            c_protected, c_t2,
            "protected color must not be reused by t2"
        );
    }

    /// Same-shape alias collapses: dst and source share the slot.
    /// `output_alias` says t1 aliases t0's storage; t0 and t1 have
    /// the same shape (both `[1]` here, modeling an in-place mutator
    /// like `cutlass_gemm_add` whose output IS the residual buffer
    /// post-mutation). The runtime never has two `__tiles` entries
    /// for one buffer — so the colorer must place them at the same
    /// color, no `View` indirection, no `Op::Alias` row.
    #[test]
    fn coloring_same_shape_alias_collapses_to_source() {
        let f = Fuf {
            nodes: vec![
                add_tile(0, &[]),               // source, shape [1]
                add_tile(1, &[(TileId(0), 0)]), // alias-dst, shape [1]
                add_tile(2, &[(TileId(1), 0)]), // reads via the alias
            ],
        };
        let mut lib = ImplementationLibrary::new();
        let id_plain = lib.push(Box::new(StubImpl {
            name: "stub",
            alias_to: None,
            consumes: vec![],
        }));
        let id_alias = lib.push(Box::new(StubImpl {
            name: "stub_alias",
            alias_to: Some((TileId(0), 0)),
            consumes: vec![],
        }));
        let sfuf = linear_assignment(
            &[TileId(0), TileId(1), TileId(2)],
            &[id_plain, id_alias, id_plain],
        );
        let lp = linear_loop(3);
        let protected: HashSet<(TileId, u8)> = HashSet::new();
        let sm = colored_slot_map(&f, &sfuf, &lp, &lib, None, &protected);

        assert_eq!(
            sm.of(TileId(0), 0),
            sm.of(TileId(1), 0),
            "same-shape alias dst must share color with source (in-place mutation)",
        );
    }

    /// `Instruction::AllReduce` (the row-parallel TP communicator)
    /// is one-tile in-place same-shape — its output IS the input
    /// buffer post-mutation, just like `cutlass_gemm_add` /
    /// `fused_add_rms_norm`. Coloring must place the AllReduce
    /// output at the same slot as its input, no `View` indirection.
    /// At runtime this lets `NcclGroup::all_reduce_inplace` operate
    /// directly on the tile slot the gemm output landed in, with no
    /// alias-row preamble in the bucket slice.
    ///
    /// Shape is the realistic gemm-output shape `[N, H]` with H>1
    /// rather than `[1]` — `coloring_disjoint_lifetimes_dont_share_
    /// across_shapes` is the load-bearing bug-driven test for shape
    /// partitioning, and pinning AllReduce on a non-trivial shape
    /// keeps both invariants in scope when something here regresses.
    #[test]
    fn coloring_allreduce_collapses_to_input_slot() {
        let f = Fuf {
            nodes: vec![
                // gemm output (row-parallel weight, e.g. o_proj or
                // down_proj). Shape [N=4, H=16] stands in for the
                // post-attention or post-MLP residual stream.
                FufNode {
                    id: TileId(0),
                    op: OpKind::Gemm,
                    inputs: vec![],
                    outputs: vec![vec![Dim::Lit(4), Dim::Lit(16)]],
                },
                // AllReduce in-place: output aliases gemm output,
                // same shape. This is the shape the lowering pass
                // (task #5) emits at every ShardDim1 weight at tp>1.
                FufNode {
                    id: TileId(1),
                    op: OpKind::Add, // OpKind::AllReduce will land with task #5
                    inputs: vec![FufInput::Tile {
                        id: TileId(0),
                        slot: 0,
                    }],
                    outputs: vec![vec![Dim::Lit(4), Dim::Lit(16)]],
                },
                // Downstream consumer (e.g. residual-add) reads via
                // the alias.
                FufNode {
                    id: TileId(2),
                    op: OpKind::Add,
                    inputs: vec![FufInput::Tile {
                        id: TileId(1),
                        slot: 0,
                    }],
                    outputs: vec![vec![Dim::Lit(4), Dim::Lit(16)]],
                },
            ],
        };
        let mut lib = ImplementationLibrary::new();
        let id_plain = lib.push(Box::new(StubImpl {
            name: "stub",
            alias_to: None,
            consumes: vec![],
        }));
        let id_all_reduce = lib.push(Box::new(StubImpl {
            name: "all_reduce",
            // Same-shape in-place: dst slot 0 aliases gemm output.
            alias_to: Some((TileId(0), 0)),
            consumes: vec![],
        }));
        let sfuf = linear_assignment(
            &[TileId(0), TileId(1), TileId(2)],
            &[id_plain, id_all_reduce, id_plain],
        );
        let lp = linear_loop(3);
        let protected: HashSet<(TileId, u8)> = HashSet::new();
        let sm = colored_slot_map(&f, &sfuf, &lp, &lib, None, &protected);

        assert_eq!(
            sm.of(TileId(0), 0),
            sm.of(TileId(1), 0),
            "AllReduce dst must share slot with input — in-place \
             same-shape, no View row, NCCL all_reduce_inplace \
             operates on the gemm output tile directly",
        );
    }

    /// Different-shape alias keeps its own slot: dst and source have
    /// distinct colors so the runtime can hold a `View { ref_slot:
    /// source_color }` entry at the dst slot. Models `Reshape` —
    /// dst metadata differs (rank or dims) but storage is shared via
    /// indirection. Source's color stays in its own shape pool; dst
    /// gets a fresh color in *its* shape pool. They can never collide.
    #[test]
    fn coloring_different_shape_alias_keeps_own_slot() {
        let f = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Add,
                    inputs: vec![],
                    outputs: vec![vec![Dim::Lit(6)]], // [6]
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Reshape,
                    inputs: vec![FufInput::Tile {
                        id: TileId(0),
                        slot: 0,
                    }],
                    outputs: vec![vec![Dim::Lit(2), Dim::Lit(3)]], // [2,3]
                },
                add_tile(2, &[(TileId(1), 0)]),
            ],
        };
        let mut lib = ImplementationLibrary::new();
        let id_plain = lib.push(Box::new(StubImpl {
            name: "stub",
            alias_to: None,
            consumes: vec![],
        }));
        let id_alias = lib.push(Box::new(StubImpl {
            name: "stub_reshape",
            alias_to: Some((TileId(0), 0)),
            consumes: vec![],
        }));
        let sfuf = linear_assignment(
            &[TileId(0), TileId(1), TileId(2)],
            &[id_plain, id_alias, id_plain],
        );
        let lp = linear_loop(3);
        let protected: HashSet<(TileId, u8)> = HashSet::new();
        let sm = colored_slot_map(&f, &sfuf, &lp, &lib, None, &protected);

        assert_ne!(
            sm.of(TileId(0), 0),
            sm.of(TileId(1), 0),
            "different-shape alias dst needs its own slot for the View entry",
        );
    }

    /// Shape-partitioned reuse: a `[1]` tile and an `[N]` tile cannot
    /// share a slot even when their lifetimes are disjoint. This is
    /// the load-bearing invariant for in-place mutations whose
    /// downstream op writes more bytes than the prior occupant's
    /// allocation. Without shape partitioning, a `[41, 8, 128]` K
    /// tile (84 KB) would hand its slot to a `[41, 3072]` residual
    /// tile (252 KB), and the next `cutlass_gemm_add` would write
    /// past the buffer's end — observed as the K=8192 N=5120 cublas
    /// panic on Llama 3.2 3B before this fix.
    #[test]
    fn coloring_disjoint_lifetimes_dont_share_across_shapes() {
        let f = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Add,
                    inputs: vec![],
                    outputs: vec![vec![Dim::Lit(8)]],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Add,
                    inputs: vec![FufInput::Tile {
                        id: TileId(0),
                        slot: 0,
                    }],
                    outputs: vec![vec![Dim::Lit(3072)]],
                },
            ],
        };
        let mut lib = ImplementationLibrary::new();
        let id_plain = lib.push(Box::new(StubImpl {
            name: "stub",
            alias_to: None,
            consumes: vec![],
        }));
        let sfuf = linear_assignment(&[TileId(0), TileId(1)], &[id_plain; 2]);
        let lp = linear_loop(2);
        let protected: HashSet<(TileId, u8)> = HashSet::new();
        let sm = colored_slot_map(&f, &sfuf, &lp, &lib, None, &protected);

        assert_ne!(
            sm.of(TileId(0), 0),
            sm.of(TileId(1), 0),
            "tiles of different shape never share a color, even with disjoint lifetimes",
        );
    }

    // ── Loop-detection invariants ───────────────────────────────

    /// Build a tiny OpInstance with one variant ident + a list of
    /// pre-stringified field values. Lets the loop-detection tests
    /// construct synthetic IR without going through fan_out.
    fn op(name: &str, fields: &[&str]) -> OpInstance {
        let parsed: Vec<TokenStream> = fields
            .iter()
            .map(|s| s.parse::<TokenStream>().unwrap())
            .collect();
        OpInstance::new(format_ident!("{}", name), parsed)
    }

    /// Three byte-identical `Foo()` rows → period 1, count 3.
    #[test]
    fn loop_detection_finds_simplest_run() {
        let v = vec![op("Foo", &[]), op("Foo", &[]), op("Foo", &[])];
        let map = std::collections::HashMap::new();
        let r = detect_repeating_run(&v, &map);
        assert_eq!(r, Some((0, 1, 3)));
    }

    /// A prefix + a 3×2 repeating block + a suffix: the detector
    /// returns the largest span `(start=2, period=2, iters=3)`.
    /// Pins the "find the largest contiguous repeating run" bit;
    /// the prefix/suffix residues stay where they are.
    #[test]
    fn loop_detection_picks_largest_span_amid_residue() {
        let v = vec![
            op("Pre", &[]),
            op("Pre", &[]),
            op("A", &[]),
            op("B", &[]),
            op("A", &[]),
            op("B", &[]),
            op("A", &[]),
            op("B", &[]),
            op("Post", &[]),
        ];
        let map = std::collections::HashMap::new();
        let r = detect_repeating_run(&v, &map);
        assert_eq!(r, Some((2, 2, 3)));
    }

    /// A run where one variant carries an iter-index field is
    /// detected only when the iter-index field steps by exactly 1
    /// between iterations. The other field values must be byte-
    /// equal across iterations. This pins the `iter_offset` check
    /// in `blocks_match`.
    #[test]
    fn loop_detection_handles_iter_index_field() {
        let mut map = std::collections::HashMap::new();
        map.insert("Norm".to_string(), 1usize); // field index 1 = layer
        let v = vec![
            op("Norm", &["wm.norm", "0u32"]),
            op("Norm", &["wm.norm", "1u32"]),
            op("Norm", &["wm.norm", "2u32"]),
        ];
        let r = detect_repeating_run(&v, &map);
        assert_eq!(r, Some((0, 1, 3)));
    }

    /// The iter-index check is strict: stepping by 2 (or any
    /// non-1 offset) does NOT count as a loop. The detector
    /// returns None.
    #[test]
    fn loop_detection_rejects_non_unit_iter_step() {
        let mut map = std::collections::HashMap::new();
        map.insert("Norm".to_string(), 1usize);
        let v = vec![
            op("Norm", &["wm.norm", "0u32"]),
            op("Norm", &["wm.norm", "2u32"]),
            op("Norm", &["wm.norm", "4u32"]),
        ];
        let r = detect_repeating_run(&v, &map);
        assert_eq!(r, None);
    }

    /// A non-iter field varies between rows → not a loop. Locks
    /// the "every non-iter field must be byte-equal" rule.
    #[test]
    fn loop_detection_rejects_non_iter_field_drift() {
        let mut map = std::collections::HashMap::new();
        map.insert("Norm".to_string(), 1usize);
        let v = vec![
            op("Norm", &["wm.input_layernorm", "0u32"]),
            op("Norm", &["wm.post_attention_layernorm", "1u32"]),
        ];
        let r = detect_repeating_run(&v, &map);
        assert_eq!(r, None);
    }

    /// `apply_loop_compression`: when a run is detected,
    /// `instances` becomes prefix + Op::Loop + one-iteration body
    /// (with each iter-index field set to that row's iter-0
    /// baseline) + suffix. Locks the rewrite shape end-to-end.
    #[test]
    fn loop_compression_emits_loop_and_keeps_baseline() {
        let mut arch_opcodes = ArchOpcodes::new();
        arch_opcodes.register(OpcodeShape::new(
            "Norm",
            vec![
                ("w", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        ));
        let mut lb = LoweredBucket {
            instances: vec![
                op("Norm", &["7u32", "0u32"]),
                op("Norm", &["7u32", "1u32"]),
                op("Norm", &["7u32", "2u32"]),
            ],
            num_slots: 1,
            final_slot: 0,
        };
        apply_loop_compression(&arch_opcodes, &mut lb, "layer");
        assert_eq!(lb.instances.len(), 2, "Loop + 1 body row");
        assert_eq!(lb.instances[0].name.to_string(), "Loop");
        // Loop fields: count, body_len. body_len = 1.
        assert_eq!(lb.instances[0].field_values[0].to_string(), "3");
        assert_eq!(lb.instances[0].field_values[1].to_string(), "1");
        assert_eq!(lb.instances[1].name.to_string(), "Norm");
        // Body's `layer` field carries the iter-0 baseline (0).
        assert_eq!(lb.instances[1].field_values[1].to_string(), "0u32");
    }

    /// Per-row baseline preservation: when the body period has rows
    /// whose iter-0 layer values differ (e.g., row A starts at 0,
    /// row B starts at 1 — Llama's body has `input_layernorm` at
    /// position 6 with baseline = 1 because it logically belongs to
    /// the *next* layer), `apply_loop_compression` must keep each
    /// row's baseline. Zeroing all rows to 0 silently uses layer N's
    /// weights for layer N+1's input ln — accumulating drift that
    /// turns decode into garbage after a few tokens.
    #[test]
    fn loop_compression_preserves_per_row_baseline() {
        let mut arch_opcodes = ArchOpcodes::new();
        arch_opcodes.register(OpcodeShape::new(
            "A",
            vec![("layer", syn::parse_quote!(u32))],
        ));
        arch_opcodes.register(OpcodeShape::new(
            "B",
            vec![("layer", syn::parse_quote!(u32))],
        ));
        let mut lb = LoweredBucket {
            instances: vec![
                // iter 0: A@0, B@1
                op("A", &["0u32"]),
                op("B", &["1u32"]),
                // iter 1: A@1, B@2
                op("A", &["1u32"]),
                op("B", &["2u32"]),
                // iter 2: A@2, B@3
                op("A", &["2u32"]),
                op("B", &["3u32"]),
            ],
            num_slots: 1,
            final_slot: 0,
        };
        apply_loop_compression(&arch_opcodes, &mut lb, "layer");
        assert_eq!(lb.instances.len(), 3, "Loop + 2 body rows");
        assert_eq!(lb.instances[0].name.to_string(), "Loop");
        assert_eq!(lb.instances[1].name.to_string(), "A");
        assert_eq!(
            lb.instances[1].field_values[0].to_string(),
            "0u32",
            "row A's baseline is 0",
        );
        assert_eq!(lb.instances[2].name.to_string(), "B");
        assert_eq!(
            lb.instances[2].field_values[0].to_string(),
            "1u32",
            "row B's baseline is 1 — must be preserved, not zeroed, so the arm \
             can compute `__layer + 1` for the iter-N execution",
        );
    }

    /// No loop in the IR → `apply_loop_compression` is a no-op.
    /// Important property: the pass must not corrupt instances
    /// when no run is present.
    #[test]
    fn loop_compression_is_noop_without_runs() {
        let mut arch_opcodes = ArchOpcodes::new();
        arch_opcodes.register(OpcodeShape::new("A", vec![("x", syn::parse_quote!(u32))]));
        let original = vec![op("A", &["0u32"])];
        let mut lb = LoweredBucket {
            instances: original.clone(),
            num_slots: 1,
            final_slot: 0,
        };
        apply_loop_compression(&arch_opcodes, &mut lb, "layer");
        assert_eq!(lb.instances.len(), 1);
        assert_eq!(lb.instances[0].name.to_string(), "A");
    }
}
