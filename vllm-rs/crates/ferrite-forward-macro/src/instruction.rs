// SPDX-License-Identifier: Apache-2.0
//! Dataflow megakernel instruction IR.
//!
//! Instead of BSP phases (all SMs execute phase 0, barrier, phase 1, …),
//! the dataflow megakernel uses **tile-level instructions** dispatched via
//! work-stealing. Each SM grabs the next instruction from a global atomic
//! counter, waits on its specific input barriers, executes, and signals
//! output barriers. SMs never idle — small GEMVs from different layers
//! run concurrently.
//!
//! This module defines the compile-time instruction representation.
//! The runtime scaffold (`megakernel_dataflow.cuh`) interprets these
//! instructions with warp-specialized execution (controller, loader,
//! consumer, storer warps).

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::classified::OpKind;
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::{ImplId, ImplementationLibrary};
use crate::solver::{Assignment, SubgraphId};

// ── Opcodes ──────────────────────────────────────────────────────

/// Opcode identifying which kernel function to dispatch for a tile
/// instruction. Each opcode has corresponding loader, consumer, and
/// storer code in the CUDA runtime scaffold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum Opcode {
    /// GEMV / GEMM: weight × activation. Loader fetches weight tile
    /// from HBM via TMA; consumers run warpgroup MMA.
    Gemv = 0,
    /// RMS normalization. Memory-bound; consumers read activation,
    /// compute norm, write normalized output.
    RmsNorm = 1,
    /// Rotary position embedding. Elementwise; consumers apply
    /// rotation in-place.
    Rope = 2,
    /// Fused SiLU+Mul (SwiGLU gate activation). Two inputs → one
    /// output, elementwise.
    SiluMul = 3,
    /// Fused GELU+Mul (GeGLU gate activation).
    GeluMul = 4,
    /// Elementwise addition (residual connection).
    Add = 5,
    /// Paged FlashAttention-2 decode.
    Attention = 6,
    /// Embedding table gather.
    Embed = 7,
    /// Tanh soft-cap (logit capping).
    TanhSoftCap = 8,
    /// Scalar multiply (e.g. 1/sqrt(d) scaling).
    ScalarMul = 9,
}

impl Opcode {
    pub fn as_u16(self) -> u16 {
        self as u16
    }
}

// ── Barrier descriptors ──────────────────────────────────────────

/// A barrier this instruction must wait on before reading its input.
/// The loader warp spins on `barriers[barrier_idx]` until it reaches
/// `expected_val` (monotonic counter — producer increments after
/// writing the tile this instruction reads).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BarrierWait {
    /// Index into the flat barrier array on device.
    pub barrier_idx: u32,
    /// Spin until `barriers[barrier_idx] >= expected_val`.
    pub expected_val: u32,
}

/// A barrier this instruction signals after writing its output.
/// The storer warp does `atomicAdd(&barriers[barrier_idx], 1)` with
/// release semantics after the output tile is visible in gmem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BarrierSignal {
    /// Index into the flat barrier array on device.
    pub barrier_idx: u32,
}

// ── Per-instruction descriptor ───────────────────────────────────

/// One tile-level instruction for the dataflow megakernel.
///
/// At runtime, the instruction tensor is `[N_instructions, 32]` u32
/// on device. The controller warp loads 32 words per instruction,
/// decodes the opcode, and dispatches to the appropriate warp roles.
///
/// At compile time, we build `InstructionDesc` with typed fields;
/// the codegen serializes them into the flat `[32]` u32 layout.
#[derive(Clone, Debug)]
pub struct InstructionDesc {
    /// Which kernel function to execute.
    pub opcode: Opcode,
    /// Layer index (for cross-layer interleaving visibility).
    pub layer: u16,
    /// Output tile row within this op's output tensor.
    pub tile_row: u16,
    /// Output tile column (for 2D tiling of large GEMMs).
    pub tile_col: u16,
    /// Which subgraph (from the solver assignment) this instruction
    /// belongs to. Used to look up weight pointers and dimensions
    /// in the per-op params.
    pub subgraph_id: SubgraphId,
    /// Which implementation realizes this subgraph.
    pub impl_id: ImplId,
    /// FUF tile IDs covered by this instruction (typically one tile,
    /// but fused ops like gate_up cover multiple).
    pub tiles: Vec<TileId>,

    /// Barriers to wait on before this instruction can read its inputs.
    pub wait_on: Vec<BarrierWait>,
    /// Barriers to signal after this instruction writes its outputs.
    pub signal: Vec<BarrierSignal>,

    /// Per-op payload words (up to 24 u32s). Encodes weight offsets,
    /// dimensions, strides — everything the loader/consumer/storer
    /// need beyond the opcode. Layout is opcode-specific.
    pub payload: Vec<u32>,
}

// ── Opcode dispatch table ────────────────────────────────────────

/// Maps an opcode to its CUDA function names for each warp role.
/// The codegen emits a `switch(opcode)` dispatch table using these.
#[derive(Clone, Debug)]
pub struct OpcodeEntry {
    pub opcode: Opcode,
    /// C function name for the loader warp (TMA weight fetch, etc.).
    pub loader_fn: String,
    /// C function name for consumer warps (MMA, elementwise, etc.).
    pub consumer_fn: String,
    /// C function name for the storer warp (TMA output store, etc.).
    pub storer_fn: String,
    /// Shared memory bytes this opcode needs per instruction.
    pub smem_bytes: u32,
}

// ── Full schedule ────────────────────────────────────────────────

/// The complete instruction schedule for one dataflow megakernel.
///
/// Built at compile time from the FUF + solver assignment + tile
/// decomposition. The codegen serializes this into:
/// 1. A host-side instruction tensor (uploaded to device before launch)
/// 2. A barrier array (zeroed before each launch)
/// 3. Opcode dispatch `switch` arms in the CUDA scaffold
#[derive(Clone, Debug)]
pub struct InstructionSchedule {
    /// Instructions in dispatch order. The work-stealing loop
    /// processes them in this order via `atomicAdd` on a global
    /// counter, so ordering affects which instructions run
    /// concurrently on different SMs.
    pub instructions: Vec<InstructionDesc>,
    /// Total number of barrier counters needed.
    pub num_barriers: usize,
    /// Opcode → dispatch function mapping.
    pub opcodes: Vec<OpcodeEntry>,
    /// Per-subgraph metadata needed by the launch wrapper to pass
    /// weight pointers, dimensions, etc. as kernel params.
    pub subgraph_params: HashMap<SubgraphId, SubgraphParams>,
}

/// Per-subgraph parameters that the Rust launch code must pass
/// to the dataflow kernel. These are the same values that the BSP
/// megakernel passes as flat params, but grouped by subgraph so
/// the instruction payload can index into them.
#[derive(Clone, Debug)]
pub struct SubgraphParams {
    pub subgraph_id: SubgraphId,
    pub impl_id: ImplId,
    /// `(c_type, param_name, rust_expr)` triples. The codegen
    /// emits these as kernel params and builds a device-side
    /// lookup table indexed by subgraph ordinal.
    pub params: Vec<(String, String, String)>,
}

// ── Instruction tensor serialization ─────────────────────────────

/// Number of u32 words per instruction in the device tensor.
pub const INSTRUCTION_WORDS: usize = 32;

impl InstructionDesc {
    /// Serialize this instruction into a fixed-size `[u32; 32]` for
    /// the device instruction tensor.
    ///
    /// Layout (word indices):
    /// ```text
    ///  [0]  opcode (u16) | layer (u16)
    ///  [1]  tile_row (u16) | tile_col (u16)
    ///  [2]  num_wait_on (u8) | num_signal (u8) | subgraph_ordinal (u16)
    ///  [3..3+num_wait]  barrier_idx for each wait
    ///  [3+num_wait..3+num_wait+num_wait]  expected_val for each wait
    ///  [next..next+num_signal]  barrier_idx for each signal
    ///  [remaining..]  payload words
    /// ```
    pub fn serialize(&self, subgraph_ordinal: u16) -> [u32; INSTRUCTION_WORDS] {
        let mut words = [0u32; INSTRUCTION_WORDS];

        // Word 0: opcode | layer
        words[0] = (self.opcode.as_u16() as u32) | ((self.layer as u32) << 16);
        // Word 1: tile_row | tile_col
        words[1] = (self.tile_row as u32) | ((self.tile_col as u32) << 16);
        // Word 2: num_wait | num_signal | subgraph_ordinal
        let nw = self.wait_on.len().min(255) as u32;
        let ns = self.signal.len().min(255) as u32;
        words[2] = nw | (ns << 8) | ((subgraph_ordinal as u32) << 16);

        let mut idx = 3;

        // Wait barrier indices
        for w in &self.wait_on {
            if idx < INSTRUCTION_WORDS {
                words[idx] = w.barrier_idx;
                idx += 1;
            }
        }
        // Wait expected values
        for w in &self.wait_on {
            if idx < INSTRUCTION_WORDS {
                words[idx] = w.expected_val;
                idx += 1;
            }
        }
        // Signal barrier indices
        for s in &self.signal {
            if idx < INSTRUCTION_WORDS {
                words[idx] = s.barrier_idx;
                idx += 1;
            }
        }
        // Payload
        for &p in &self.payload {
            if idx < INSTRUCTION_WORDS {
                words[idx] = p;
                idx += 1;
            }
        }

        words
    }
}

// ── OpKind → Opcode mapping ──────────────────────────────────────

/// Map a FUF `OpKind` to a dataflow `Opcode`.
///
/// Fused ops (SiluMul, GeluMul) map 1:1. Multi-tile fused subgraphs
/// (e.g. gate_up) are handled at the subgraph level by the scheduler
/// — this function handles individual tile ops.
fn opcode_for_op(op: OpKind) -> Option<Opcode> {
    match op {
        OpKind::Gemm => Some(Opcode::Gemv),
        OpKind::RmsNorm => Some(Opcode::RmsNorm),
        OpKind::RopeAppend => Some(Opcode::Rope),
        OpKind::Silu => Some(Opcode::SiluMul),
        OpKind::Gelu => Some(Opcode::GeluMul),
        OpKind::Add => Some(Opcode::Add),
        OpKind::Attention | OpKind::SlidingAttention => Some(Opcode::Attention),
        OpKind::Embed => Some(Opcode::Embed),
        OpKind::TanhSoftCap => Some(Opcode::TanhSoftCap),
        OpKind::Mul => Some(Opcode::ScalarMul),
    }
}

// ── Dataflow instruction scheduler ───────────────────────────────

/// Build a dataflow instruction schedule from a megakernel wave.
///
/// Each subgraph in the wave becomes one or more instructions (one
/// per FUF tile in the subgraph, for v0). Barriers are allocated
/// for every cross-subgraph data dependency edge.
///
/// The `subgraph_order` must be topologically sorted (the BSP
/// scheduler already provides this).
pub fn build_instruction_schedule(
    fuf: &Fuf,
    sfuf: &Assignment,
    lib: &ImplementationLibrary,
    subgraph_order: &[(SubgraphId, ImplId)],
) -> InstructionSchedule {
    // Step 1: Build cross-subgraph dependency edges.
    // An edge (producer_tile, consumer_tile) exists when consumer_tile
    // has a FufInput::Tile pointing at producer_tile, and they belong
    // to different subgraphs.
    //
    // Each unique (producer_tile, output_slot) gets one barrier counter.
    // The producer instruction signals it; each consumer instruction
    // waits on it.
    let mut barrier_map: HashMap<(TileId, u8), u32> = HashMap::new();
    let mut next_barrier: u32 = 0;

    // Collect all tiles in this wave.
    let wave_sgs: HashSet<SubgraphId> = subgraph_order.iter().map(|(sg, _)| *sg).collect();

    // For each tile in the wave, check its inputs for cross-subgraph deps.
    // Allocate a barrier for each unique (producer_tile, slot) pair.
    for &(sg, _) in subgraph_order {
        for tile in sfuf.tiles_in_subgraph(sg) {
            let node = fuf.get(tile);
            for input in &node.inputs {
                if let FufInput::Tile {
                    id: producer_tile,
                    slot,
                } = input
                    && let Some(producer_sg) = sfuf.subgraph_of(*producer_tile)
                    && producer_sg != sg
                    && wave_sgs.contains(&producer_sg)
                {
                    barrier_map
                        .entry((*producer_tile, *slot))
                        .or_insert_with(|| {
                            let idx = next_barrier;
                            next_barrier += 1;
                            idx
                        });
                }
            }
        }
    }

    // Step 2: Infer layer indices.
    // Heuristic: tiles from the same loop iteration get the same layer.
    // After unrolling, the FUF is a flat list where loop body tiles
    // repeat in groups. We estimate layer by counting how many tiles
    // of the same OpKind precede this one.
    let mut op_counters: HashMap<OpKind, u16> = HashMap::new();
    let mut tile_layer: HashMap<TileId, u16> = HashMap::new();
    for node in &fuf.nodes {
        let count = op_counters.entry(node.op).or_insert(0);
        tile_layer.insert(node.id, *count);
        *count += 1;
    }

    // Step 3: Build instructions from subgraphs.
    let mut instructions: Vec<InstructionDesc> = Vec::new();
    let mut used_opcodes: HashSet<Opcode> = HashSet::new();

    for &(sg, impl_id) in subgraph_order {
        let tiles = sfuf.tiles_in_subgraph(sg);

        // For fused subgraphs (multiple tiles), emit one instruction
        // for the "primary" tile (the output tile). For singletons,
        // one instruction per tile.
        if tiles.len() == 1 {
            let tile = tiles[0];
            let node = fuf.get(tile);
            let Some(opcode) = opcode_for_op(node.op) else {
                continue;
            };
            used_opcodes.insert(opcode);

            let layer = tile_layer.get(&tile).copied().unwrap_or(0);

            // Build wait_on: for each input that crosses subgraphs
            let mut wait_on = Vec::new();
            for input in &node.inputs {
                if let FufInput::Tile {
                    id: producer_tile,
                    slot,
                } = input
                    && let Some(&barrier_idx) = barrier_map.get(&(*producer_tile, *slot))
                {
                    wait_on.push(BarrierWait {
                        barrier_idx,
                        expected_val: 1,
                    });
                }
            }

            // Build signal: for each output that has a cross-subgraph consumer
            let mut signal = Vec::new();
            for slot in 0..node.outputs.len() as u8 {
                if let Some(&barrier_idx) = barrier_map.get(&(tile, slot)) {
                    signal.push(BarrierSignal { barrier_idx });
                }
            }

            instructions.push(InstructionDesc {
                opcode,
                layer,
                tile_row: 0,
                tile_col: 0,
                subgraph_id: sg,
                impl_id,
                tiles: vec![tile],
                wait_on,
                signal,
                payload: Vec::new(),
            });
        } else {
            // Fused multi-tile subgraph (e.g. gate_up_silu_mul).
            // Emit one instruction for the entire fused op.
            // Use the last tile's OpKind as the opcode hint, but
            // for gate_up it's Mul → maps to SiluMul or GeluMul.
            let primary_op = tiles
                .iter()
                .map(|t| fuf.get(*t).op)
                .find(|op| matches!(op, OpKind::Silu | OpKind::Gelu | OpKind::Mul))
                .or_else(|| tiles.first().map(|t| fuf.get(*t).op));

            let opcode = primary_op
                .and_then(opcode_for_op)
                .unwrap_or(Opcode::Gemv);
            used_opcodes.insert(opcode);

            let layer = tiles
                .first()
                .and_then(|t| tile_layer.get(t).copied())
                .unwrap_or(0);

            // Collect all cross-subgraph input barriers
            let mut wait_on = Vec::new();
            let mut seen_barriers: HashSet<u32> = HashSet::new();
            for &tile in &tiles {
                let node = fuf.get(tile);
                for input in &node.inputs {
                    if let FufInput::Tile {
                        id: producer_tile,
                        slot,
                    } = input
                        && let Some(&barrier_idx) = barrier_map.get(&(*producer_tile, *slot))
                        && seen_barriers.insert(barrier_idx)
                    {
                        wait_on.push(BarrierWait {
                            barrier_idx,
                            expected_val: 1,
                        });
                    }
                }
            }

            // Collect all output barriers
            let mut signal = Vec::new();
            for &tile in &tiles {
                let node = fuf.get(tile);
                for slot in 0..node.outputs.len() as u8 {
                    if let Some(&barrier_idx) = barrier_map.get(&(tile, slot)) {
                        signal.push(BarrierSignal { barrier_idx });
                    }
                }
            }

            instructions.push(InstructionDesc {
                opcode,
                layer,
                tile_row: 0,
                tile_col: 0,
                subgraph_id: sg,
                impl_id,
                tiles: tiles.clone(),
                wait_on,
                signal,
                payload: Vec::new(),
            });
        }
    }

    // Step 4: Build opcode dispatch table.
    let opcodes: Vec<OpcodeEntry> = used_opcodes
        .into_iter()
        .map(|op| OpcodeEntry {
            opcode: op,
            loader_fn: format!("load_{}", opcode_name(op)),
            consumer_fn: format!("compute_{}", opcode_name(op)),
            storer_fn: format!("store_{}", opcode_name(op)),
            smem_bytes: default_smem_for_opcode(op),
        })
        .collect();

    let _ = lib; // will be used in later phases for per-op params

    InstructionSchedule {
        instructions,
        num_barriers: next_barrier as usize,
        opcodes,
        subgraph_params: HashMap::new(), // filled by codegen (Phase 4)
    }
}

fn opcode_name(op: Opcode) -> &'static str {
    match op {
        Opcode::Gemv => "gemv",
        Opcode::RmsNorm => "rmsnorm",
        Opcode::Rope => "rope",
        Opcode::SiluMul => "silu_mul",
        Opcode::GeluMul => "gelu_mul",
        Opcode::Add => "add",
        Opcode::Attention => "attention",
        Opcode::Embed => "embed",
        Opcode::TanhSoftCap => "tanh_softcap",
        Opcode::ScalarMul => "scalar_mul",
    }
}

/// Assign instructions to SMs via round-robin, matching KVM's
/// `round_robin_assign_to_sms`. Returns one instruction list per SM.
///
/// When `ENABLE_GLOBAL_WORK_QUEUE = false`, each SM walks its own
/// static list — no work-stealing, no atomic counter, no controller
/// overhead.
pub fn assign_to_sms(
    instructions: &[InstructionDesc],
    num_sms: usize,
) -> Vec<Vec<InstructionDesc>> {
    let mut per_sm: Vec<Vec<InstructionDesc>> = (0..num_sms).map(|_| Vec::new()).collect();
    for (i, inst) in instructions.iter().enumerate() {
        per_sm[i % num_sms].push(inst.clone());
    }
    per_sm
}

/// Default shared memory estimate per opcode.
/// Refined when per-op kernel code is written (Phase 3).
fn default_smem_for_opcode(op: Opcode) -> u32 {
    match op {
        // GEMV: weight tile staging (128KB typical for 64×2048 bf16 tile)
        Opcode::Gemv => 131_072,
        // Attention: Q/K/V staging
        Opcode::Attention => 65_536,
        // Elementwise ops: minimal smem
        _ => 8_192,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::build_cfg;
    use crate::classify::classify;
    use crate::config::{self, ModelParams};
    use crate::fuf::unroll;
    use crate::impl_lib::starter_library;
    use crate::parse::parse_block;
    use crate::schedule::schedule;
    use crate::shape::infer;
    use crate::solver::solve;
    use crate::target::{TargetProfile, load_file as load_target};
    use std::path::PathBuf;

    fn llama_params() -> ModelParams {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join("model_architectures/llama/llama-3.2-1b.json");
        config::load_file(&path).unwrap()
    }

    fn h100_target() -> TargetProfile {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join("target_profiles/h100_sm90.json");
        load_target(&path).unwrap()
    }

    fn build_schedule_for(
        src: &str,
    ) -> (Fuf, Assignment, ImplementationLibrary, InstructionSchedule) {
        let params = llama_params();
        let target = h100_target();
        let file: syn::File =
            syn::parse_str(&format!("fn _c() {{ {src} }}")).expect("parse");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).unwrap();
        let program = classify(&ast).unwrap();
        let inferred = infer(&program).unwrap();
        let cfg = build_cfg(&program, &params).unwrap();
        let fuf = unroll(&cfg, &inferred).unwrap();
        let lib = starter_library();
        let workloads =
            solve(&fuf, &lib, &target, &inferred, &params.bounds, &[1]).unwrap();
        let sfuf = workloads.per_num_tokens[&1].clone();
        let loop_ir = schedule(&fuf, &sfuf, &lib);

        // Find the first megakernel wave.
        let mega_wave = loop_ir
            .waves
            .iter()
            .find(|w| w.is_megakernel)
            .expect("expected at least one megakernel wave on H100");

        let sched =
            build_instruction_schedule(&fuf, &sfuf, &lib, &mega_wave.subgraphs);
        (fuf, sfuf, lib, sched)
    }

    #[test]
    fn dataflow_schedule_has_instructions() {
        let (_, _, _, sched) = build_schedule_for(
            r#"
            hidden_states = embed(input_ids, embed_tokens);
            for layer in 0..num_hidden_layers {
                normed = rmsnorm(hidden_states, input_layernorm[layer]);
                q = gemm(normed, self_attn.q_proj[layer]);
                k = gemm(normed, self_attn.k_proj[layer]);
                v = gemm(normed, self_attn.v_proj[layer]);
                (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
                attn = attention(q, k, v, kv_cache[layer], block_table);
                oproj = gemm(attn, self_attn.o_proj[layer]);
                hidden_states = add(oproj, hidden_states);

                normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
                gate = silu(gemm(normed2, mlp.gate_proj[layer]));
                up = gemm(normed2, mlp.up_proj[layer]);
                down = gemm(gate * up, mlp.down_proj[layer]);
                hidden_states = add(down, hidden_states);
            }
            normed = rmsnorm(hidden_states, norm);
            logits = gemm(normed, lm_head);
            "#,
        );

        assert!(
            !sched.instructions.is_empty(),
            "dataflow schedule should have instructions"
        );
        assert!(
            sched.num_barriers > 0,
            "should have barriers between dependent ops"
        );
        assert!(
            !sched.opcodes.is_empty(),
            "should have opcode dispatch entries"
        );

        eprintln!(
            "Dataflow schedule: {} instructions, {} barriers, {} opcodes",
            sched.instructions.len(),
            sched.num_barriers,
            sched.opcodes.len(),
        );

        // Every instruction should have a valid opcode.
        let valid_opcodes: HashSet<Opcode> =
            sched.opcodes.iter().map(|e| e.opcode).collect();
        for inst in &sched.instructions {
            assert!(
                valid_opcodes.contains(&inst.opcode),
                "instruction opcode {:?} not in dispatch table",
                inst.opcode,
            );
        }
    }

    #[test]
    fn barriers_are_balanced() {
        let (_, _, _, sched) = build_schedule_for(
            r#"
            hidden_states = embed(input_ids, embed_tokens);
            for layer in 0..num_hidden_layers {
                normed = rmsnorm(hidden_states, input_layernorm[layer]);
                q = gemm(normed, self_attn.q_proj[layer]);
                oproj = gemm(q, self_attn.o_proj[layer]);
                hidden_states = add(oproj, hidden_states);
            }
            normed = rmsnorm(hidden_states, norm);
            logits = gemm(normed, lm_head);
            "#,
        );

        // Every barrier that is waited on must also be signalled
        // (and vice versa for correctness).
        let mut signalled: HashSet<u32> = HashSet::new();
        let mut waited: HashSet<u32> = HashSet::new();
        for inst in &sched.instructions {
            for s in &inst.signal {
                signalled.insert(s.barrier_idx);
            }
            for w in &inst.wait_on {
                waited.insert(w.barrier_idx);
            }
        }

        let wait_not_signal: Vec<_> =
            waited.difference(&signalled).collect();
        let signal_not_wait: Vec<_> =
            signalled.difference(&waited).collect();

        assert!(
            wait_not_signal.is_empty(),
            "barriers waited but never signalled: {:?}",
            wait_not_signal,
        );
        assert!(
            signal_not_wait.is_empty(),
            "barriers signalled but never waited: {:?}",
            signal_not_wait,
        );
    }

    #[test]
    fn serialize_roundtrip_basic() {
        let inst = InstructionDesc {
            opcode: Opcode::Gemv,
            layer: 3,
            tile_row: 7,
            tile_col: 0,
            subgraph_id: SubgraphId(10),
            impl_id: ImplId(2),
            tiles: vec![TileId(42)],
            wait_on: vec![BarrierWait {
                barrier_idx: 5,
                expected_val: 1,
            }],
            signal: vec![BarrierSignal { barrier_idx: 12 }],
            payload: vec![0xDEAD, 0xBEEF],
        };

        let words = inst.serialize(4); // subgraph ordinal 4

        // Word 0: opcode=0 | layer=3
        assert_eq!(words[0] & 0xFFFF, Opcode::Gemv as u32);
        assert_eq!(words[0] >> 16, 3);
        // Word 1: tile_row=7 | tile_col=0
        assert_eq!(words[1] & 0xFFFF, 7);
        assert_eq!(words[1] >> 16, 0);
        // Word 2: nw=1 | ns=1 | ordinal=4
        assert_eq!(words[2] & 0xFF, 1);       // num_wait
        assert_eq!((words[2] >> 8) & 0xFF, 1); // num_signal
        assert_eq!(words[2] >> 16, 4);         // ordinal
        // Word 3: wait barrier_idx
        assert_eq!(words[3], 5);
        // Word 4: wait expected_val
        assert_eq!(words[4], 1);
        // Word 5: signal barrier_idx
        assert_eq!(words[5], 12);
        // Words 6-7: payload
        assert_eq!(words[6], 0xDEAD);
        assert_eq!(words[7], 0xBEEF);
        // Rest zero
        assert_eq!(words[8], 0);
    }

    #[test]
    fn serialize_no_barriers() {
        let inst = InstructionDesc {
            opcode: Opcode::Add,
            layer: 0,
            tile_row: 0,
            tile_col: 0,
            subgraph_id: SubgraphId(0),
            impl_id: ImplId(0),
            tiles: vec![TileId(1)],
            wait_on: vec![],
            signal: vec![],
            payload: vec![42],
        };

        let words = inst.serialize(0);
        assert_eq!(words[0] & 0xFFFF, Opcode::Add as u32);
        assert_eq!(words[2] & 0xFF, 0);       // no waits
        assert_eq!((words[2] >> 8) & 0xFF, 0); // no signals
        assert_eq!(words[3], 42);              // payload starts at word 3
    }

    #[test]
    fn opcode_values_are_stable() {
        // These values are baked into the CUDA dispatch table.
        // Changing them breaks the runtime. Guard with a test.
        assert_eq!(Opcode::Gemv as u16, 0);
        assert_eq!(Opcode::RmsNorm as u16, 1);
        assert_eq!(Opcode::Rope as u16, 2);
        assert_eq!(Opcode::SiluMul as u16, 3);
        assert_eq!(Opcode::GeluMul as u16, 4);
        assert_eq!(Opcode::Add as u16, 5);
        assert_eq!(Opcode::Attention as u16, 6);
        assert_eq!(Opcode::Embed as u16, 7);
        assert_eq!(Opcode::TanhSoftCap as u16, 8);
        assert_eq!(Opcode::ScalarMul as u16, 9);
    }
}
