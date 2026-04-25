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
use quote::{ToTokens, quote};

use crate::classified::Program;
use crate::config::ModelParams;
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::{ImplementationLibrary, MatchInfo, OpInstance, OpcodeShape, SlotMap};
use crate::schedule::Loop;
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
/// 1. **Live-range overlap.** Two slots co-live iff one's def
///    position ≤ the other's last-use position and vice versa.
///    Co-live slots get distinct colors.
/// 2. **Aliasing.** A `View` slot's `ref_slot` points at its source
///    slot; runtime navigation requires the two be distinct entries
///    in `__tiles`. So `dst.color != src.color` for every alias
///    pair surfaced via `Implementation::output_alias`.
/// 3. **Consume.** `consumes_input_tiles` declares a slot whose
///    `OwnedTensor` migrates into the consumer's output. The
///    consumed slot's last-use is the consume site — past that it's
///    dead and its color is freed.
/// 4. **Protected slots** (the per-bucket fn's return tile, and the
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

    // Last use per OWNER (resolving alias chains): the latest
    // subgraph that reads this owner's storage, directly or via a
    // View. Last use of a non-owner (a View slot itself) is computed
    // separately below.
    let mut owner_last_use: HashMap<(TileId, u8), usize> = HashMap::new();
    for &sg in &order_arr {
        let pos = order[&sg];
        let claimed: HashSet<TileId> = sfuf.tiles_in_subgraph(sg).into_iter().collect();
        for tile in &claimed {
            for input in &fuf.get(*tile).inputs {
                if let FufInput::Tile { id, slot } = input {
                    if claimed.contains(id) {
                        continue;
                    }
                    let owner = resolve((*id, *slot));
                    owner_last_use
                        .entry(owner)
                        .and_modify(|p| *p = (*p).max(pos))
                        .or_insert(pos);
                }
            }
        }
    }
    // View slots' last_use: when is the View itself read? A View is
    // read whenever its dst slot appears as a Tile input to some
    // downstream subgraph. Same walk but without alias resolution.
    let mut view_last_use: HashMap<(TileId, u8), usize> = HashMap::new();
    for &sg in &order_arr {
        let pos = order[&sg];
        let claimed: HashSet<TileId> = sfuf.tiles_in_subgraph(sg).into_iter().collect();
        for tile in &claimed {
            for input in &fuf.get(*tile).inputs {
                if let FufInput::Tile { id, slot } = input {
                    if claimed.contains(id) {
                        continue;
                    }
                    if alias_to_owner.contains_key(&(*id, *slot)) {
                        view_last_use
                            .entry((*id, *slot))
                            .and_modify(|p| *p = (*p).max(pos))
                            .or_insert(pos);
                    }
                }
            }
        }
    }

    // Collect every (tile, output_slot) pair, sorted by def position.
    let mut def_pos: HashMap<TileId, usize> = HashMap::new();
    for &sg in &order_arr {
        for tile in sfuf.tiles_in_subgraph(sg) {
            def_pos.insert(tile, order[&sg]);
        }
    }
    let mut pairs: Vec<(usize, TileId, u8)> = Vec::new();
    for &sg in &order_arr {
        for tile in sfuf.tiles_in_subgraph(sg) {
            let n_out = fuf.get(tile).outputs.len().max(1) as u8;
            for slot in 0..n_out {
                pairs.push((order[&sg], tile, slot));
            }
        }
    }
    pairs.sort_by_key(|&(p, t, s)| (p, t, s));

    // Linear-scan: walk pairs in def order. Maintain `active` =
    // currently-live (color, last_use, owner_key). Free a color when
    // its owner's last_use < current def_pos. For aliases the
    // constraint is dst.color != src.color while both alive — encoded
    // by excluding src's color from the candidate pool when picking
    // dst's color.
    let mut active: Vec<(usize, u32, (TileId, u8))> = Vec::new();
    let mut free_colors: BTreeSet<u32> = BTreeSet::new();
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
                free_colors.insert(color);
                false
            } else {
                true
            }
        });

        // Compute this pair's own last_use.
        let is_alias = alias_to_owner.contains_key(&(tile, slot));
        let lu_self = if is_alias {
            // View dst: dies after its last reader.
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

        // Constraint: an alias dst's color must differ from its
        // resolved owner's color (the View's `ref_slot` points at
        // owner's slot index; they coexist in `__tiles`).
        let exclude: Option<u32> = if is_alias {
            let owner = resolve((tile, slot));
            // Owner's color was assigned earlier (smaller def_pos).
            Some(sm.of(owner.0, owner.1))
        } else {
            None
        };

        // Pick the smallest free color, skipping `exclude`.
        let color = {
            let mut chosen: Option<u32> = None;
            // BTreeSet iter ascending — picks lowest first.
            for &c in free_colors.iter() {
                if Some(c) == exclude {
                    continue;
                }
                chosen = Some(c);
                break;
            }
            match chosen {
                Some(c) => {
                    free_colors.remove(&c);
                    c
                }
                None => {
                    // Out of free colors — mint a new one. (If the
                    // new color happens to collide with `exclude`,
                    // that's impossible because `exclude` was
                    // already minted as a smaller color.)
                    let c = next_color;
                    next_color += 1;
                    c
                }
            }
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

/// Variant declarations + arm bodies the macro accumulates across
/// every bucket of one arch. The per-arch enum + per-arch
/// interpreter are emitted from these.
#[derive(Default)]
pub struct ArchOpcodes {
    /// Variant ident → (shape, interpreter-arm body). First insert
    /// wins; later inserts of the same variant ident must agree on
    /// shape (codegen panics on mismatch). Arm body comes from the
    /// first migrated Impl that owns this variant — same Impl is
    /// expected to declare the variant identically across arches,
    /// so duplicates with the same `interpreter_arm` are accepted.
    by_name: BTreeMap<String, (OpcodeShape, TokenStream)>,
    /// Variant ident → prelude `let <fname>: <fty> = <value>;` rows
    /// extracted from the per-row instruction stream by
    /// [`extract_arch_wide_constants`] (every instance of this
    /// variant carried the same byte-identical value, so we drop the
    /// field from the row and rebind it in the arm body).
    /// [`emit_interpreter`] partitions the entries: tuples that
    /// appear in 2+ variants lift to ONE fn-scope let in
    /// `__dispatch_one`; per-variant residuals stay at the top of
    /// their arm. Avoids ~`N variants × shared lines` of duplicate
    /// `let cos_sin_fn = Weights::rotary_cos_sin;` boilerplate in
    /// the expanded source.
    extracted_prelude: BTreeMap<String, Vec<(syn::Ident, syn::Type, TokenStream)>>,
}

impl ArchOpcodes {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a variant declaration + arm body. Panics if the
    /// same variant ident is registered with a structurally
    /// different shape.
    pub fn register(&mut self, shape: OpcodeShape, arm_body: TokenStream) {
        let key = shape.name.to_string();
        if let Some((existing_shape, _)) = self.by_name.get(&key) {
            assert_shapes_agree(existing_shape, &shape);
            return;
        }
        self.by_name.insert(key, (shape, arm_body));
    }

    /// Snapshot of registered variant shapes keyed by variant ident
    /// string. Includes the universal `Free` variant. Consumed by
    /// [`emit_bucket_static_slice`] to type-check positional field
    /// values against the registered shape per bucket.
    pub fn shapes_by_name(&self) -> BTreeMap<String, OpcodeShape> {
        let mut out: BTreeMap<String, OpcodeShape> = self
            .by_name
            .iter()
            .map(|(name, (shape, _))| (name.clone(), shape.clone()))
            .collect();
        out.insert("Alias".to_string(), alias_variant_shape());
        out.insert("Free".to_string(), free_variant_shape());
        out.insert("Loop".to_string(), loop_variant_shape());
        out
    }

    /// In-place mutation of a registered variant's stored shape +
    /// arm body. Used by [`extract_arch_wide_constants`] to drop
    /// arch-wide fields from a variant after it's been registered
    /// across multiple lower_bucket calls.
    pub fn replace(&mut self, variant_name: &str, shape: OpcodeShape, arm_body: TokenStream) {
        if let Some(slot) = self.by_name.get_mut(variant_name) {
            *slot = (shape, arm_body);
        }
    }

    /// Read access to (shape, arm_body) for a variant. Used by
    /// [`extract_arch_wide_constants`].
    pub fn get(&self, variant_name: &str) -> Option<&(OpcodeShape, TokenStream)> {
        self.by_name.get(variant_name)
    }

    /// Iterate (variant_name, (shape, arm_body)). Used by
    /// [`apply_layer_loop`] to build the per-variant layer-field
    /// position map.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &(OpcodeShape, TokenStream))> {
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
/// iteration-index field on the body's rows is zeroed — the
/// interpreter passes the runtime iteration counter as `__layer`,
/// which arm bodies use directly. No-op when no run is detected.
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
    for (name, (shape, _)) in arch_opcodes.iter() {
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
            copy.field_values[fi] = quote! { 0u32 };
        }
        new_instances.push(copy);
    }
    new_instances.extend_from_slice(&lowered.instances[span_end..]);
    lowered.instances = new_instances;
}

/// Drop fields from variants whose value is byte-identical across
/// every instance in every lowered bucket. The dropped fields are
/// rebound at the top of the variant's match-arm body via
/// `let <fname>: <fty> = <constant_value>;` so the unchanged kernel-
/// call body inside continues to compile against the same names.
///
/// The point: arch-wide knobs like `interleaved: true` /
/// `cos_sin_fn: Weights::rotary_cos_sin` / `weight_fn:
/// Weights::self_attn_qkv` ride on a row only because the Impl
/// emitted them as fields, not because they actually vary. After
/// this pass they're constants in the arm and the static slice's
/// rows shrink to just the per-claim parts (slot indices, layer
/// index, and per-claim accessors that genuinely differ between
/// claims).
///
/// `Free` is excluded — its `slot` field is by construction per-
/// instance, and `Free` is emitted directly by the codegen rather
/// than via an Impl, so its arm body isn't an arch_opcodes entry.
pub fn extract_arch_wide_constants(
    arch_opcodes: &mut ArchOpcodes,
    lowereds: &mut [&mut LoweredBucket],
) {
    use std::collections::HashMap;
    // Group every instance index across every lowered bucket by
    // variant ident. Indices are `(bucket_idx, instance_idx)`.
    let mut by_variant: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    for (b_idx, lb) in lowereds.iter().enumerate() {
        for (i_idx, inst) in lb.instances.iter().enumerate() {
            by_variant
                .entry(inst.name.to_string())
                .or_default()
                .push((b_idx, i_idx));
        }
    }

    for (variant_name, locs) in by_variant.iter() {
        if variant_name == "Free" {
            continue;
        }
        // Snapshot the registered shape + arm body up-front so we
        // can mutate `extracted_prelude` (a sibling field on
        // `arch_opcodes`) without holding an immutable borrow.
        let (shape, arm_body) = match arch_opcodes.get(variant_name) {
            Some((s, b)) => (s.clone(), b.clone()),
            None => continue,
        };
        let n_fields = shape.fields.len();
        if n_fields == 0 || locs.is_empty() {
            continue;
        }

        // Identify field indices whose value is identical across all
        // instances of this variant. Compare TokenStream-as-string —
        // OpInstance values are simple literal-or-path tokens and the
        // string form is stable.
        let mut keep: Vec<usize> = (0..n_fields).collect();
        let mut drop: Vec<(usize, syn::Ident, syn::Type, TokenStream)> = Vec::new();
        for fi in 0..n_fields {
            let first = locs[0];
            let first_val = lowereds[first.0].instances[first.1].field_values[fi].to_string();
            let all_same = locs
                .iter()
                .all(|&(b, i)| lowereds[b].instances[i].field_values[fi].to_string() == first_val);
            if all_same {
                let (fname, fty) = &shape.fields[fi];
                let val = lowereds[first.0].instances[first.1].field_values[fi].clone();
                drop.push((fi, fname.clone(), fty.clone(), val));
            }
        }
        if drop.is_empty() {
            continue;
        }
        let drop_idxs: std::collections::HashSet<usize> =
            drop.iter().map(|(i, _, _, _)| *i).collect();
        keep.retain(|i| !drop_idxs.contains(i));

        // Record the dropped (fname, fty, value) tuples in the
        // arch-level side map. `emit_interpreter` reads this map to
        // partition: tuples that appear (with byte-identical type +
        // value) in 2+ variants lift to ONE fn-scope let in
        // `__dispatch_one`; per-variant residuals stay at the top
        // of the arm. The arm body in `arch_opcodes.by_name` is
        // intentionally the ORIGINAL kernel call here — no prelude
        // folding — so the partition step has clean inputs.
        let extracted: Vec<(syn::Ident, syn::Type, TokenStream)> = drop
            .iter()
            .map(|(_, fname, fty, val)| (fname.clone(), fty.clone(), val.clone()))
            .collect();
        // Append in case multiple extraction passes contribute to
        // the same variant (today it's called once per arch, but
        // the API doesn't forbid repeats).
        arch_opcodes
            .extracted_prelude
            .entry(variant_name.clone())
            .or_default()
            .extend(extracted);

        // Strip dropped fields from the shape.
        let new_fields: Vec<(syn::Ident, syn::Type)> = shape
            .fields
            .iter()
            .enumerate()
            .filter(|(i, _)| !drop_idxs.contains(i))
            .map(|(_, f)| f.clone())
            .collect();
        let new_shape = OpcodeShape {
            name: shape.name.clone(),
            fields: new_fields,
        };

        // Body unchanged — emit_interpreter folds the prelude back
        // (either fn-scope for shared, arm-scope for residual).
        arch_opcodes.replace(variant_name, new_shape, arm_body);

        // Strip dropped fields from every instance of this variant.
        for &(b, i) in locs.iter() {
            lowereds[b].instances[i].field_values = lowereds[b].instances[i]
                .field_values
                .iter()
                .enumerate()
                .filter(|(i, _)| !drop_idxs.contains(i))
                .map(|(_, v)| v.clone())
                .collect();
        }
    }
}

impl ArchOpcodes {
    /// Emit the per-arch opcode enum. Variant order is sorted by
    /// name for deterministic output. The codegen always appends
    /// the universal `Alias` and `Free` variants — they are
    /// codegen-issued (alias prelude + drop pass), not Impl-issued.
    pub fn emit_enum(&self, enum_ident: &syn::Ident) -> TokenStream {
        let variants = self.by_name.values().map(|(shape, _)| variant_decl(shape));
        let alias_variant = variant_decl(&alias_variant_shape());
        let free_variant = variant_decl(&free_variant_shape());
        let loop_variant = variant_decl(&loop_variant_shape());
        quote! {
            /// Macro-codegened opcode enum. Variants come from the
            /// Impls the solver picked for this arch. `Free` is
            /// always present — emitted by the drop pass.
            ///
            /// The match in the per-arch interpreter is closed and
            /// exhaustive over this enum. Only `Copy` is derived —
            /// the interpreter matches by value (`match *__op`); we
            /// don't print or clone Op values, and dropping Debug
            /// avoids per-arch `impl ::core::fmt::Debug for Op {…}`
            /// boilerplate from cargo-expand.
            ///
            /// `Clone` is hand-impl'd as `*self` (Copy types have a
            /// trivial Clone). The `#[derive(Clone)]` expansion
            /// would otherwise be ~60 lines of `let _:
            /// AssertParamIsClone<…>;` per variant — pure
            /// cargo-expand bloat with no runtime effect.
            #[derive(Copy)]
            #[allow(non_camel_case_types, dead_code)]
            pub enum #enum_ident {
                #(#variants,)*
                #alias_variant,
                #free_variant,
                #loop_variant,
            }

            impl ::core::clone::Clone for #enum_ident {
                #[inline]
                fn clone(&self) -> Self { *self }
            }
        }
    }

    /// Emit the per-arch interpreter — a `__dispatch_one` helper
    /// that matches on a single op (closed over the per-arch enum,
    /// no `_` arm) and a `__interpret` driver that walks the slice
    /// honoring `Op::Loop` repeats.
    ///
    /// Splitting dispatch from driving lets `Op::Loop` re-use the
    /// SAME match arms when running its body N times — without
    /// duplicating the (large) per-Impl arm bodies. The driver loop
    /// is the only piece that needs to know about `Loop`, `Alias`,
    /// `Free` control-flow / setup semantics.
    pub fn emit_interpreter(
        &self,
        helper_ident: &syn::Ident,
        enum_ident: &syn::Ident,
    ) -> TokenStream {
        let dispatch_ident = quote::format_ident!("__dispatch_one");
        // Partition `extracted_prelude` into (shared, per-variant
        // residual). A tuple (fname, fty, value) is "shared" when
        // it appears in 2+ variants with byte-identical fty AND
        // value — those lift to ONE fn-scope let, dedup'd by
        // (fname, fty.to_string(), value.to_string()). The rest
        // stay at the top of their owning arm.
        //
        // Why dedup by string: `syn::Type` and `TokenStream` don't
        // implement Eq; their `to_string()` form is stable for the
        // simple literal-or-path tokens these prelude lets carry.
        let mut occurrences: BTreeMap<(String, String, String), usize> = BTreeMap::new();
        for entries in self.extracted_prelude.values() {
            // Within one variant, count each (key) at most once
            // (multiple instances of the variant get extracted into
            // a single shared row; we don't want the dedup to read
            // the same row twice here either).
            let mut seen_in_variant: std::collections::HashSet<(String, String, String)> =
                std::collections::HashSet::new();
            for (fname, fty, val) in entries {
                let key = (
                    fname.to_string(),
                    fty.to_token_stream().to_string(),
                    val.to_string(),
                );
                if seen_in_variant.insert(key.clone()) {
                    *occurrences.entry(key).or_insert(0) += 1;
                }
            }
        }
        // Walk one canonical (fname, fty, val) representative per
        // shared key — a key shared across N variants will be hit
        // N times, so we dedup the actual emit by inserting into
        // `shared_lets` at the FIRST occurrence only.
        //
        // Drop the type annotation on the emitted let — Rust infers
        // the type from `val` (fn-item path → fn item, bool / u32
        // literal → respective primitive, etc.) and arm-body call
        // sites only need the binding's NAME. Keeping the annotation
        // would expand multi-line `for<'a> fn(&'a Weights, u32) ->
        // &'a LinearLayer` types verbatim per binding (~4 lines per
        // weight_fn / cos_sin_fn declaration).
        let mut shared_lets: Vec<TokenStream> = Vec::new();
        let mut emitted_shared: std::collections::HashSet<(String, String, String)> =
            std::collections::HashSet::new();
        for entries in self.extracted_prelude.values() {
            for (fname, _fty, val) in entries {
                let key = (
                    fname.to_string(),
                    _fty.to_token_stream().to_string(),
                    val.to_string(),
                );
                if occurrences.get(&key).copied().unwrap_or(0) >= 2 && emitted_shared.insert(key) {
                    shared_lets.push(quote! { let #fname = #val; });
                }
            }
        }
        let arms = self.by_name.iter().map(|(name, (shape, body))| {
            let var = &shape.name;
            let pat = variant_pattern(shape);
            // Per-variant residual prelude: dropped lets that aren't
            // shared with another variant. These stay at the top of
            // the arm because they only matter to this body. Same
            // annotation-drop trick — body call sites use the
            // binding's name, not its annotated type.
            let residual: Vec<TokenStream> = self
                .extracted_prelude
                .get(name)
                .map(|entries| {
                    entries
                        .iter()
                        .filter(|(fname, fty, val)| {
                            let key = (
                                fname.to_string(),
                                fty.to_token_stream().to_string(),
                                val.to_string(),
                            );
                            occurrences.get(&key).copied().unwrap_or(0) < 2
                        })
                        .map(|(fname, _fty, val)| quote! { let #fname = #val; })
                        .collect()
                })
                .unwrap_or_default();
            // Per-arm `let layer: u32 = __layer;` is the bridge
            // from `Op::Loop`'s iteration counter to arm bodies.
            // For variants whose `OpcodeShape` has a `layer:`
            // field, the destructure pattern would otherwise bind
            // `layer` to the (zeroed-by-apply_loop_compression)
            // row literal; this shadow overrides with the runtime
            // iteration index. Variants without a `layer:` field
            // are unaffected — the binding is just unused
            // (suppressed by the fn's `unused_variables` allow).
            quote! {
                #enum_ident::#var #pat => {
                    let layer: u32 = __layer;
                    #(#residual)*
                    #body
                }
            }
        });
        quote! {
            /// Dispatch a single op. Reads the per-arm body the
            /// Impl declared via `interpreter_arm`. `__layer` is
            /// the loop iteration index when called from inside an
            /// `Op::Loop`, else 0.
            #[cfg(feature = "cuda")]
            #[allow(clippy::too_many_arguments, unused_unsafe, unused_variables)]
            unsafe fn #dispatch_ident(
                __op: #enum_ident,
                __layer: u32,
                __tiles: &mut ::std::vec::Vec<Option<::ferrite_forward::TileEntry>>,
                wm: &Weights,
                ctx: &::ferrite_forward::ForwardCtx,
                device: &mut ::ferrite_cuda_core::device::GpuDevice,
            ) {
                // Fn-scope shared prelude: extracted constants that
                // appear (with identical type + value) in 2+ arms.
                // Lifted ONCE here so each arm just references the
                // binding by name; without this lift, every arm
                // re-emits the same `let cos_sin_fn = …;`-style
                // line and cargo expand picks up the duplicates.
                #(#shared_lets)*
                match __op {
                    #(#arms,)*
                    #enum_ident::Alias(dst, src) => {
                        __tiles[dst as usize] = Some(::ferrite_forward::view(src));
                    }
                    #enum_ident::Free(slot) => {
                        __tiles[slot as usize] = None;
                    }
                    #enum_ident::Loop(_, _) => {
                        // Loop is driver-handled — never reaches dispatch.
                        unsafe { ::core::hint::unreachable_unchecked() }
                    }
                }
            }

            /// Driver loop. Walks `__ops` linearly except for
            /// `Op::Loop(count, body_len)` which re-runs the next
            /// `body_len` ops `count` times with `__layer` set to
            /// the iteration index.
            #[cfg(feature = "cuda")]
            #[allow(clippy::too_many_arguments)]
            unsafe fn #helper_ident(
                __ops: &[#enum_ident],
                __tiles: &mut ::std::vec::Vec<Option<::ferrite_forward::TileEntry>>,
                wm: &Weights,
                ctx: &::ferrite_forward::ForwardCtx,
                device: &mut ::ferrite_cuda_core::device::GpuDevice,
            ) {
                let mut __i: usize = 0;
                while __i < __ops.len() {
                    match __ops[__i] {
                        #enum_ident::Loop(count, body_len) => {
                            let body_start = __i + 1;
                            let body_end = body_start + body_len as usize;
                            for __l in 0..count {
                                for __j in body_start..body_end {
                                    unsafe {
                                        #dispatch_ident(
                                            __ops[__j], __l, __tiles, wm, ctx, device,
                                        );
                                    }
                                }
                            }
                            __i = body_end;
                        }
                        other => {
                            unsafe {
                                #dispatch_ident(other, 0, __tiles, wm, ctx, device);
                            }
                            __i += 1;
                        }
                    }
                }
            }
        }
    }
}

/// Emit one per-bucket `static FORWARD_<TAG>: &[<EnumName>] = &[…];`
/// using `OpInstance::field_values` (positional, in
/// `OpcodeShape::fields` order) wrapped in
/// `<EnumName>::<Variant> { … }` constructors. Free instances are
/// recognized by their variant ident `"Free"` and wrapped as
/// `<EnumName>::Free { slot: <expr> }`.
pub fn emit_bucket_static_slice(
    static_ident: &syn::Ident,
    enum_ident: &syn::Ident,
    shapes_by_name: &BTreeMap<String, OpcodeShape>,
    instances: &[OpInstance],
) -> TokenStream {
    let elements = instances.iter().map(|inst| {
        let var = &inst.name;
        // Free is always present; look up its shape from the standard helper.
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
        // Tuple-style row: `<Enum>::<Variant>(v0, v1, …)`. Field
        // declaration order from `shape.fields` matches the order of
        // `inst.field_values` (the asserts above pin this), so we
        // can drop the field-name binding here — the order alone
        // re-positions each value into the right tuple slot. The
        // single-line emission keeps a 1000-row slice at 1000 lines
        // instead of N × (fields + 2).
        let exprs = inst.field_values.iter();
        quote! {
            #enum_ident::#var ( #(#exprs),* )
        }
    });
    quote! {
        static #static_ident: &[#enum_ident] = &[ #(#elements),* ];
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// Render an OpcodeShape as a Rust tuple-style enum-variant
/// declaration — `Variant(T1, T2, …)`. Tuple-style keeps each
/// emitted row of a static slice on a single line, so a static
/// slice with N rows is N lines instead of N × (fields + 2). Field
/// names are still carried by [`OpcodeShape::fields`] so the match
/// arm in [`emit_interpreter`] can destructure positionally with
/// the names — same effect as named-field destructure for arm
/// bodies, but without the row-level verbosity.
fn variant_decl(shape: &OpcodeShape) -> TokenStream {
    let name = &shape.name;
    let tys = shape.fields.iter().map(|(_fname, fty)| fty);
    quote! { #name ( #(#tys),* ) }
}

/// Tuple-style destructure pattern — `(f1, f2, …)`. Bindings are
/// the field idents from [`OpcodeShape::fields`], so arm bodies
/// reference each by name.
fn variant_pattern(shape: &OpcodeShape) -> TokenStream {
    let names = shape.fields.iter().map(|(fname, _ty)| fname);
    quote! { ( #(#names),* ) }
}

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
/// scheduling points, and accumulates `(OpcodeShape,
/// interpreter_arm)` registrations into `arch_opcodes`.
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
    model: &ModelParams,
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
                         interpreter IR. Override `opcode_shape`, `fan_out`, and \
                         `interpreter_arm` on `{name}`.",
                        name = imp.name(),
                        id = imp_id.0,
                    )
                });
            arch_opcodes.register(imp.opcode_shape(), imp.interpreter_arm(model));
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

    /// A View dst has a separate slot in `__tiles` from its source;
    /// at runtime the View carries `ref_slot = src.color` and the
    /// interpreter dereferences. So `dst.color != src.color` is a
    /// hard constraint — the test checks the colorer never violates
    /// it even when the source is otherwise "free" by lifetime.
    #[test]
    fn coloring_alias_dst_distinct_from_source() {
        let f = Fuf {
            nodes: vec![
                add_tile(0, &[]),               // source
                add_tile(1, &[(TileId(0), 0)]), // alias-dst (Views t0)
                add_tile(2, &[(TileId(1), 0)]), // reads via the View
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
            alias_to: Some((TileId(0), 0)), // t1 aliases t0's storage
            consumes: vec![],
        }));
        let sfuf = linear_assignment(
            &[TileId(0), TileId(1), TileId(2)],
            &[id_plain, id_alias, id_plain],
        );
        let lp = linear_loop(3);
        let protected: HashSet<(TileId, u8)> = HashSet::new();
        let sm = colored_slot_map(&f, &sfuf, &lp, &lib, None, &protected);

        // t0 is the source; t1 is an alias dst. Both must have
        // distinct colors so the runtime View can navigate.
        assert_ne!(
            sm.of(TileId(0), 0),
            sm.of(TileId(1), 0),
            "alias dst color must differ from source color"
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
    /// (with iter-index field zeroed) + suffix. Locks the rewrite
    /// shape end-to-end.
    #[test]
    fn loop_compression_emits_loop_and_zeroes_iter_field() {
        let mut arch_opcodes = ArchOpcodes::new();
        arch_opcodes.register(
            OpcodeShape::new(
                "Norm",
                vec![
                    ("w", syn::parse_quote!(u32)),
                    ("layer", syn::parse_quote!(u32)),
                ],
            ),
            quote! {},
        );
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
        // Body's `layer` field zeroed.
        assert_eq!(lb.instances[1].field_values[1].to_string(), "0u32");
    }

    /// No loop in the IR → `apply_loop_compression` is a no-op.
    /// Important property: the pass must not corrupt instances
    /// when no run is present.
    #[test]
    fn loop_compression_is_noop_without_runs() {
        let mut arch_opcodes = ArchOpcodes::new();
        arch_opcodes.register(
            OpcodeShape::new("A", vec![("x", syn::parse_quote!(u32))]),
            quote! {},
        );
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

    /// `ArchOpcodes::emit_enum` produces a Rust enum with one
    /// variant per registered shape plus the universal `Free`.
    /// Locks the no-universal-registry contract: the variant set
    /// is exactly what got registered.
    #[test]
    fn arch_opcodes_emit_enum_includes_registered_plus_free() {
        let mut ops = ArchOpcodes::new();
        ops.register(
            OpcodeShape::new(
                "AttnNorm",
                vec![
                    ("layer", syn::parse_quote!(u32)),
                    ("in_slot", syn::parse_quote!(u32)),
                    ("out_slot", syn::parse_quote!(u32)),
                ],
            ),
            quote! { /* body */ },
        );
        let enum_ident = format_ident!("LlamaOp");
        let ts = ops.emit_enum(&enum_ident).to_string();
        assert!(ts.contains("enum LlamaOp"));
        assert!(ts.contains("AttnNorm"));
        assert!(ts.contains("Free"));
        assert!(!ts.contains("__Unmigrated"));
    }

    /// The codegen-emitted interpreter is a closed match — no `_`
    /// arm. Locks the design rule that exhaustiveness comes from
    /// the per-arch enum, not from a runtime catch-all.
    #[test]
    fn arch_interpreter_match_has_no_catchall() {
        let mut ops = ArchOpcodes::new();
        ops.register(
            OpcodeShape::new("AttnNorm", vec![("layer", syn::parse_quote!(u32))]),
            quote! {},
        );
        let enum_ident = format_ident!("LlamaOp");
        let helper_ident = format_ident!("__llama_interpret");
        let ts = ops.emit_interpreter(&helper_ident, &enum_ident).to_string();
        // No `_ =>` arm.
        assert!(
            !ts.contains("_ =>"),
            "interpreter must be exhaustive over the enum, no catch-all"
        );
        // Free arm always present.
        assert!(ts.contains("LlamaOp :: Free"));
        // Registered variant present.
        assert!(ts.contains("LlamaOp :: AttnNorm"));
        // No unsafe transmute / from_wire.
        assert!(!ts.contains("transmute"));
        assert!(!ts.contains("from_wire"));
    }

    /// `emit_enum` derives `Copy` automatically and impls `Clone`
    /// manually as `*self`. The default `#[derive(Clone)]` macro
    /// expands to a verbose AssertParamIsClone bound check per
    /// variant — ~60 lines for an arch with ~25 variants — that
    /// shows up in cargo expand and per-monomorphization compile
    /// work without any runtime benefit (Op is a plain Copy enum;
    /// Clone IS *self for any Copy type).
    #[test]
    fn arch_enum_uses_manual_clone_impl_to_skip_assertparamisclone() {
        let mut ops = ArchOpcodes::new();
        ops.register(
            OpcodeShape::new("AttnNorm", vec![("layer", syn::parse_quote!(u32))]),
            quote! {},
        );
        let enum_ident = format_ident!("LlamaOp");
        let ts = ops.emit_enum(&enum_ident).to_string();
        // Hand-rolled Clone present with `*self` body.
        assert!(
            ts.contains("impl :: core :: clone :: Clone for LlamaOp"),
            "expected manual Clone impl, got: {ts}"
        );
        assert!(ts.contains("fn clone (& self) -> Self { * self }"));
        // Derive list is just Copy, not Copy+Clone — that's how we
        // dodge the AssertParamIsClone expansion.
        assert!(ts.contains("# [derive (Copy)]"));
        assert!(!ts.contains("# [derive (Copy , Clone)]"));
    }

    /// `__dispatch_one` emits exactly ONE `let layer: u32 = __layer;`
    /// per arm and ZERO `let _ = layer;` lines. The unused-warning
    /// suppression is handled by the fn-level
    /// `#[allow(unused_variables)]`, so the `let _ = layer;` line
    /// the prior emit added per-arm + once at the fn top is pure
    /// expansion bloat (~3 lines × N arms × N model variants on
    /// llama).
    #[test]
    fn arch_interpreter_drops_layer_drop_lines() {
        let mut ops = ArchOpcodes::new();
        ops.register(
            OpcodeShape::new("AttnNorm", vec![("layer", syn::parse_quote!(u32))]),
            quote! {},
        );
        ops.register(
            OpcodeShape::new("Add", vec![("a_slot", syn::parse_quote!(u32))]),
            quote! {},
        );
        let enum_ident = format_ident!("LlamaOp");
        let helper_ident = format_ident!("__llama_interpret");
        let ts = ops.emit_interpreter(&helper_ident, &enum_ident).to_string();
        // Per-arm `let layer: u32 = __layer;` shadow MUST stay —
        // it's the bridge from Op::Loop's iteration counter to arm
        // bodies. 2 registered variants → at least 2 occurrences.
        let shadow_count = ts.matches("let layer : u32 = __layer ;").count();
        assert!(
            shadow_count >= 2,
            "expected one `let layer = __layer;` shadow per arm, got {shadow_count}"
        );
        // No `let _ = layer;` lines anywhere — the fn's
        // `#[allow(unused_variables)]` already suppresses warnings.
        assert!(
            !ts.contains("let _ = layer ;"),
            "expansion contains redundant `let _ = layer;` lines"
        );
        // The fn-top redundant `let layer = __layer; let _ = layer;`
        // is gone — match opens directly. Approximate by checking the
        // fn body opens `match __op`.
        assert!(ts.contains("__layer : u32 , __tiles"));
        // Find what comes immediately after the fn signature's
        // closing brace + opening body brace `{`. Should be
        // `match __op {` not `let layer ...`.
        let after_open = ts
            .split_once(", ) { ")
            .or_else(|| ts.split_once(") { "))
            .map(|(_, rhs)| rhs)
            .unwrap_or(&ts);
        assert!(
            after_open.starts_with("match __op {"),
            "fn body must open with `match __op`, got: {}",
            &after_open[..after_open.len().min(80)],
        );
    }

    /// Registering the same variant ident with structurally
    /// different fields is a codegen invariant violation.
    #[test]
    #[should_panic(expected = "field count mismatch")]
    fn arch_opcodes_register_panics_on_shape_disagreement() {
        let mut ops = ArchOpcodes::new();
        ops.register(
            OpcodeShape::new("AttnNorm", vec![("layer", syn::parse_quote!(u32))]),
            quote! {},
        );
        ops.register(
            OpcodeShape::new(
                "AttnNorm",
                vec![
                    ("layer", syn::parse_quote!(u32)),
                    ("extra", syn::parse_quote!(u32)),
                ],
            ),
            quote! {},
        );
    }

    /// `emit_bucket_static_slice` lowers each `OpInstance` to a
    /// `<EnumName>::<Variant> { f1: <expr>, … }` row. Field-init
    /// idents come from the shape; field-init exprs come from the
    /// instance, in declaration order.
    #[test]
    fn bucket_static_slice_renders_struct_constructors() {
        let mut shapes_by_name = BTreeMap::new();
        shapes_by_name.insert(
            "AttnNorm".to_string(),
            OpcodeShape::new(
                "AttnNorm",
                vec![
                    ("layer", syn::parse_quote!(u32)),
                    ("in_slot", syn::parse_quote!(u32)),
                    ("out_slot", syn::parse_quote!(u32)),
                ],
            ),
        );
        shapes_by_name.insert("Free".to_string(), free_variant_shape());

        let instances = vec![
            OpInstance::new(
                format_ident!("AttnNorm"),
                vec![quote! { 0u32 }, quote! { 1u32 }, quote! { 2u32 }],
            ),
            free_instance(1),
        ];
        let static_ident = format_ident!("FORWARD_M_1");
        let enum_ident = format_ident!("LlamaOp");
        let ts = emit_bucket_static_slice(&static_ident, &enum_ident, &shapes_by_name, &instances)
            .to_string();
        // Rows are tuple-style: `LlamaOp :: AttnNorm (0u32, 1u32, 2u32)`.
        assert!(ts.contains("FORWARD_M_1"));
        assert!(ts.contains("LlamaOp :: AttnNorm"));
        assert!(ts.contains("0u32 , 1u32 , 2u32"));
        assert!(ts.contains("LlamaOp :: Free"));
        assert!(ts.contains("(1)"));
        // No struct-style field labels in the row text.
        assert!(!ts.contains("layer :"));
    }

    /// `extract_arch_wide_constants` records dropped (fname, fty,
    /// value) tuples in `arch_opcodes.extracted_prelude`, and
    /// `emit_interpreter` partitions them: tuples present in 2+
    /// variants lift to ONE fn-scope `let` at the top of
    /// `__dispatch_one`; tuples in only one variant stay at the
    /// top of that variant's arm body. Without partitioning, every
    /// arm re-emits identical `let cos_sin_fn = …;` lines and the
    /// expand picks up N copies of the same Rust source.
    #[test]
    fn shared_extracted_prelude_lifts_to_fn_scope_in_dispatch_one() {
        let mut ops = ArchOpcodes::new();
        ops.register(
            OpcodeShape::new(
                "AttnA",
                vec![
                    ("layer", syn::parse_quote!(u32)),
                    ("interleaved", syn::parse_quote!(bool)),
                ],
            ),
            quote! {},
        );
        ops.register(
            OpcodeShape::new(
                "AttnB",
                vec![
                    ("layer", syn::parse_quote!(u32)),
                    ("interleaved", syn::parse_quote!(bool)),
                ],
            ),
            quote! {},
        );
        ops.register(
            OpcodeShape::new(
                "Solo",
                vec![
                    ("layer", syn::parse_quote!(u32)),
                    ("private_flag", syn::parse_quote!(bool)),
                ],
            ),
            quote! {},
        );

        // `interleaved = true` is shared across AttnA + AttnB → must
        // lift to fn scope. `private_flag = false` only appears on
        // Solo → must stay as a per-arm residual.
        ops.extracted_prelude.insert(
            "AttnA".to_string(),
            vec![(
                format_ident!("interleaved"),
                syn::parse_quote!(bool),
                quote! { true },
            )],
        );
        ops.extracted_prelude.insert(
            "AttnB".to_string(),
            vec![(
                format_ident!("interleaved"),
                syn::parse_quote!(bool),
                quote! { true },
            )],
        );
        ops.extracted_prelude.insert(
            "Solo".to_string(),
            vec![(
                format_ident!("private_flag"),
                syn::parse_quote!(bool),
                quote! { false },
            )],
        );

        let enum_ident = format_ident!("LlamaOp");
        let helper_ident = format_ident!("__llama_interpret");
        let ts = ops.emit_interpreter(&helper_ident, &enum_ident).to_string();

        // Shared `interleaved = true` lifts to ONE fn-scope let.
        // The fn-scope let lives between the open-brace of the fn
        // body and the `match __op {` line. Type annotation is
        // dropped — Rust infers `bool` from the literal `true`.
        let dispatch_open = ts
            .split_once("__layer : u32 ,")
            .and_then(|(_, rhs)| rhs.split_once("match __op {"))
            .map(|(prelude, _)| prelude)
            .expect("__dispatch_one signature must be present");
        assert!(
            dispatch_open.contains("let interleaved = true ;"),
            "shared `interleaved = true` should lift to fn-scope, \
             got prelude: {dispatch_open}"
        );
        // Solo's `private_flag` is NOT shared → stays in its arm.
        assert!(
            !dispatch_open.contains("let private_flag"),
            "private_flag should NOT lift to fn-scope, got: {dispatch_open}"
        );
        let solo_arm = ts
            .split_once("LlamaOp :: Solo")
            .map(|(_, rhs)| rhs)
            .expect("Solo arm must be present");
        // Slice the Solo arm body up to the next variant arm.
        let solo_body_end = solo_arm.find("LlamaOp ::").unwrap_or(solo_arm.len());
        let solo_body = &solo_arm[..solo_body_end];
        assert!(
            solo_body.contains("let private_flag = false ;"),
            "Solo's private_flag should stay in its arm, got: {solo_body}"
        );

        // The shared `interleaved` should appear EXACTLY once at
        // fn-scope (and zero times in each arm's body — the lifted
        // binding is in lexical scope already).
        let interleaved_count = ts.matches("let interleaved = true ;").count();
        assert_eq!(
            interleaved_count, 1,
            "shared `interleaved` should appear once (fn-scope), got {interleaved_count}"
        );
    }

    /// `OpInstance.field_values.len()` mismatching `shape.fields.len()`
    /// fails at lower-time, not silently emitting a wrong row.
    #[test]
    #[should_panic(expected = "field_values.len()")]
    fn bucket_static_slice_panics_on_field_count_mismatch() {
        let mut shapes_by_name = BTreeMap::new();
        shapes_by_name.insert(
            "Bad".to_string(),
            OpcodeShape::new(
                "Bad",
                vec![("a", syn::parse_quote!(u32)), ("b", syn::parse_quote!(u32))],
            ),
        );
        let instances = vec![OpInstance::new(format_ident!("Bad"), vec![quote! { 1u32 }])];
        let _ = emit_bucket_static_slice(
            &format_ident!("X"),
            &format_ident!("Y"),
            &shapes_by_name,
            &instances,
        );
    }
}
