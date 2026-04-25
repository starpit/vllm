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
use crate::solver::{Assignment, SubgraphId};

// ── Slot allocation ──────────────────────────────────────────────

/// Build a dense slot allocation for every `(tile, output_slot)`
/// in the FUF, in topological tile-id order. Codegen passes
/// `&SlotMap` to every `Implementation::fan_out` so emitted
/// `OpInstance` field-value tokens carry resolved slot indices.
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

// ── Per-bucket lowering ──────────────────────────────────────────

/// Output of lowering one (variant × workload-point). The codegen
/// stitches these into the per-bucket forward fn body.
pub struct LoweredBucket {
    /// Op instances in execution order, including `Free` rows
    /// emitted by the drop pass at the same scheduling points the
    /// old codegen would have emitted `drop()` statements.
    pub instances: Vec<OpInstance>,
    /// `(dst_slot, src_slot)` pairs for the View-aliasing prelude
    /// the per-bucket fn runs at entry, before the interpreter
    /// loop.
    pub aliases: Vec<(u32, u32)>,
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

    /// Emit the per-arch opcode enum. Variant order is sorted by
    /// name for deterministic output. The codegen always appends
    /// the universal `Free { slot: u32 }` variant — it is not
    /// driven by any Impl.
    pub fn emit_enum(&self, enum_ident: &syn::Ident) -> TokenStream {
        let variants = self.by_name.values().map(|(shape, _)| variant_decl(shape));
        let free_variant = variant_decl(&free_variant_shape());
        quote! {
            /// Macro-codegened opcode enum. Variants come from the
            /// Impls the solver picked for this arch. `Free` is
            /// always present — emitted by the drop pass.
            ///
            /// The match in the per-arch interpreter is closed and
            /// exhaustive over this enum.
            #[derive(Clone, Copy, Debug)]
            #[allow(non_camel_case_types, dead_code)]
            pub enum #enum_ident {
                #(#variants,)*
                #free_variant,
            }
        }
    }

    /// Emit the per-arch interpreter helper. Match is closed over
    /// the per-arch enum (no `_` arm). Each Impl-driven variant
    /// dispatches into the interpreter_arm body the Impl declared;
    /// `Free` clears the slot.
    pub fn emit_interpreter(
        &self,
        helper_ident: &syn::Ident,
        enum_ident: &syn::Ident,
    ) -> TokenStream {
        let arms = self.by_name.values().map(|(shape, body)| {
            let var = &shape.name;
            let pat = variant_pattern(shape);
            quote! {
                #enum_ident::#var #pat => { #body }
            }
        });
        quote! {
            #[cfg(feature = "cuda")]
            #[allow(clippy::too_many_arguments, unused_unsafe, unused_variables)]
            unsafe fn #helper_ident(
                __ops: &[#enum_ident],
                __tiles: &mut ::std::vec::Vec<Option<::ferrite_forward::TileEntry>>,
                wm: &Weights,
                ctx: &::ferrite_forward::ForwardCtx,
                device: &mut ::ferrite_cuda_core::device::GpuDevice,
            ) {
                for __op in __ops.iter() {
                    match *__op {
                        #(#arms,)*
                        #enum_ident::Free { slot } => {
                            __tiles[slot as usize] = None;
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
        let field_inits =
            shape
                .fields
                .iter()
                .zip(inst.field_values.iter())
                .map(|((fname, _ty), expr)| {
                    quote! { #fname: #expr }
                });
        quote! {
            #enum_ident::#var { #(#field_inits),* }
        }
    });
    quote! {
        static #static_ident: &[#enum_ident] = &[
            #(#elements),*
        ];
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// Render an OpcodeShape as a Rust enum-variant declaration —
/// `Variant { f1: T1, f2: T2 }`.
fn variant_decl(shape: &OpcodeShape) -> TokenStream {
    let name = &shape.name;
    let fields = shape
        .fields
        .iter()
        .map(|(fname, fty)| quote! { #fname: #fty });
    quote! { #name { #(#fields),* } }
}

/// Render a struct-style destructure pattern matching `variant_decl`
/// — `{ f1, f2 }`. Used in interpreter match arms so the body's
/// free identifiers (named to match the shape's fields) bind.
fn variant_pattern(shape: &OpcodeShape) -> TokenStream {
    let names = shape.fields.iter().map(|(fname, _ty)| fname);
    quote! { { #(#names),* } }
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

/// Construct a `Free { slot }` instance the drop pass can emit.
pub fn free_instance(slot: u32) -> OpInstance {
    OpInstance::new(
        syn::Ident::new("Free", proc_macro2::Span::call_site()),
        vec![{
            let lit = proc_macro2::Literal::u32_unsuffixed(slot);
            quote! { #lit }
        }],
    )
}

// ── Bucket lowering driver ───────────────────────────────────────

/// Lower one (variant × workload-point) into [`LoweredBucket`].
/// Walks the same wave/loop the old codegen did, calls each picked
/// Impl's `fan_out`, interleaves `Free` instances at drop-pass
/// scheduling points, and accumulates `(OpcodeShape,
/// interpreter_arm)` registrations into `arch_opcodes`.
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
    protected: &HashSet<(TileId, u8)>,
    arch_opcodes: &mut ArchOpcodes,
) -> LoweredBucket {
    let slots = build_slot_map(fuf);
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

    let drops = compute_free_points(
        fuf,
        sfuf,
        loop_ir,
        lib,
        skip_subgraph,
        protected,
        &alias_to_owner,
    );

    let mut instances: Vec<OpInstance> = Vec::new();
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
                .fan_out(&m, fuf, program, bounds, &slots)
                .unwrap_or_else(|| {
                    panic!(
                        "Impl `{name}` (id {id}) has no fan_out — unmigrated to host \
                         interpreter IR. Override `opcode_shape`, `fan_out`, and \
                         `interpreter_arm` on `{name}`.",
                        name = imp.name(),
                        id = imp_id.0,
                    )
                });
            arch_opcodes.register(imp.opcode_shape(), imp.interpreter_arm());
            instances.extend(emits);

            if let Some(slots_to_free) = drops.get(sg) {
                for &(t, s) in slots_to_free {
                    let slot_idx = slots.of(t, s);
                    instances.push(free_instance(slot_idx));
                }
            }
        }
    }

    let final_tile = fuf
        .nodes
        .last()
        .map(|n| n.id)
        .expect("non-empty FUF expected at lower_bucket entry");
    let final_slot = slots.of(final_tile, 0);

    LoweredBucket {
        instances,
        aliases,
        num_slots,
        final_slot,
    }
}

fn collect_boundary_inputs(fuf: &Fuf, claimed: &[TileId]) -> Vec<TileId> {
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
            quote! { let _ = layer; },
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
        assert!(ts.contains("FORWARD_M_1"));
        assert!(ts.contains("LlamaOp :: AttnNorm"));
        assert!(ts.contains("layer : 0u32"));
        assert!(ts.contains("in_slot : 1u32"));
        assert!(ts.contains("out_slot : 2u32"));
        assert!(ts.contains("LlamaOp :: Free"));
        assert!(ts.contains("slot : 1"));
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
