// SPDX-License-Identifier: Apache-2.0
//! GPU **megakernel** tape-player glue — the runtime half of the PD-wavefront
//! decode path.
//!
//! The [`MegaProgram`] (tape / shape-class table / operand slots / buffer
//! table) is produced AT COMPILE TIME by the macro: region lowering →
//! wavefront schedule → `ferrite_wavefront::mega::serialize`. All intelligence
//! — which atom, every operand, every byte offset, the `Wait`/`Signal` edges —
//! is already baked into that neutral data. This module does ONLY the two jobs
//! the compiler cannot (they need live GPU handles), per the trivial-player law
//! (`feedback_subtile_ir_trivial_player`):
//!
//!   1. [`resolve_mega_buffers`] — turn each neutral
//!      [`ferrite_wavefront::subtile_ir::BufferRef`] into a concrete
//!      `(MTLBuffer, base)`: weights via [`resolve_weight`], runtime inputs +
//!      the paged KV cache via [`RuntimeBindings`], and arena slots freshly
//!      allocated and sized by [`MegaProgram::arena_bytes`]. This is exactly
//!      `subtile_player::resolve_buffers` (and reuses its enum mappers); it
//!      differs only in the arena, which the megakernel sizes itself rather
//!      than borrowing the worker's per-op arena.
//!   2. [`build_operand_table`] — the bindless `gpuAddress` table the trivial
//!      `wavefront_player` reads: one `gpuAddress(buf) + base + byte_offset`
//!      u64 per operand slot (the addressing mechanism proven in
//!      `tests/wavefront_addr_probe.rs`).
//!
//! then [`dispatch_mega`] binds the five player buffers and launches `P`
//! co-resident threadgroups × 1024 threads. Zero scheduling, zero kernel
//! selection, zero offset math — all of that is in the [`MegaProgram`].
//!
//! The `wavefront_serialized_*` GPU integration tests in
//! `tests/wavefront_layer_gpu.rs` ARE this exact path with a synthetic
//! `BufId → Buffer` resolver; this module is the production resolver
//! (`resolve_weight` / `RuntimeBindings` / the sized arena) it slots into.
#![cfg(feature = "metal")]

use std::ffi::c_void;
use std::ptr::NonNull;

use ::objc2::msg_send;
use ::objc2::runtime::ProtocolObject;
use ::objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLResourceOptions, MTLResourceUsage, MTLSize,
};

use ferrite_cuda_core::MetalAllocator;
use ferrite_metal_kernels::quantized::{DequantDtype, ScaleDtype};
use ferrite_metal_kernels::residency::MetalResidencySet;
use ferrite_metal_kernels::specialized_pipeline_cache::PipelineKey;
use ferrite_wavefront::mega::MegaProgram;
use ferrite_wavefront::subtile_ir::{BufferRef, InputKind, WeightBundle, WeightRole};

use super::__re::{Buffer, CommandQueue, ComputePipelineState, Device};
use super::ids::LayerId;
use super::lowered::{RuntimeBindingKind, WeightBundleKind, WeightLocator, WeightTensor};
use super::lowering::{dequant_dtype_for, scale_dtype_for};
use super::runtime::RuntimeBindings;
use super::worker::{WorkerError, resolve_weight};
use crate::{CanonicalParams, WeightAccessors};

/// One resolved logical buffer: the concrete buffer plus the base byte-offset
/// of the whole tensor. A [`ferrite_wavefront::mega::OperandSlot`]'s carried
/// `byte_offset` (an N-block row offset, a column slice) is added on top.
pub type ResolvedBuffer = (Buffer, u64);

// ── pipeline selection ──────────────────────────────────────────────

/// The trivial player's symbol for a model's dtypes — the kernel is
/// instantiated per `(act, scale, group_size, bits)`. Only the shipped
/// instantiations resolve (today: bf16/f16 act × f16 scale × gs 64 × 4 bits;
/// see `INST_WL_PLAYER` in `shaders/wavefront_layer.metal`).
pub fn player_symbol(act: DequantDtype, scale: ScaleDtype, group_size: u32, bits: u32) -> String {
    format!(
        "wavefront_player_{}_s_{}_gs_{}_b_{}",
        act.symbol_infix(),
        scale.symbol_infix(),
        group_size,
        bits,
    )
}

/// The [`PipelineKey`] for the trivial player (library `wavefront_layer`, no
/// function constants — every shape rides in the tape, not in specialization).
/// The symbol is leaked to `'static` for the cache key; one decode forward has
/// exactly one player instantiation, process-lifetime.
pub fn player_pipeline_key<W: CanonicalParams>(group_size: u32, bits: u32) -> PipelineKey {
    let symbol: &'static str = Box::leak(
        player_symbol(
            dequant_dtype_for::<W>(),
            scale_dtype_for::<W>(),
            group_size,
            bits,
        )
        .into_boxed_str(),
    );
    PipelineKey::new("wavefront_layer", symbol, vec![])
}

// ── buffer-table resolution (BufferRef → (Buffer, base)) ─────────────

/// Translate the neutral wavefront `WeightBundle` to the metal weight-bundle
/// kind `resolve_weight` keys on.
fn bundle_to_metal(b: WeightBundle) -> WeightBundleKind {
    match b {
        WeightBundle::RmsNorm => WeightBundleKind::RmsNorm,
        WeightBundle::Embedding => WeightBundleKind::Embedding,
        WeightBundle::LinearLayer => WeightBundleKind::LinearLayer,
        WeightBundle::CosSin => WeightBundleKind::CosSin,
        WeightBundle::AffineQuantEmbedding => WeightBundleKind::AffineQuantEmbedding,
    }
}

/// Translate the neutral wavefront `WeightRole` to the metal weight tensor.
fn role_to_metal(r: WeightRole) -> WeightTensor {
    match r {
        WeightRole::Weight => WeightTensor::Weight,
        WeightRole::Bias => WeightTensor::Bias,
        WeightRole::AffineScales => WeightTensor::AffineScales,
        WeightRole::AffineBiases => WeightTensor::AffineBiases,
        WeightRole::AffineLinearBias => WeightTensor::AffineLinearBias,
    }
}

/// Translate the neutral wavefront `InputKind` to the runtime binding kind
/// `RuntimeBindings::buffer_for` resolves (it owns KV-cache-half selection).
fn input_to_metal(k: InputKind) -> RuntimeBindingKind {
    match k {
        InputKind::InputIds => RuntimeBindingKind::InputIds,
        InputKind::Positions => RuntimeBindingKind::Positions,
        InputKind::SlotMapping => RuntimeBindingKind::SlotMapping,
        InputKind::CuSeqlensQ => RuntimeBindingKind::CuSeqlensQ,
        InputKind::SeqUsedK => RuntimeBindingKind::SeqUsedK,
        InputKind::BlockTable => RuntimeBindingKind::BlockTable,
        InputKind::KvCacheK { layer } => RuntimeBindingKind::KvCacheK {
            layer: LayerId(layer),
        },
        InputKind::KvCacheV { layer } => RuntimeBindingKind::KvCacheV {
            layer: LayerId(layer),
        },
        InputKind::NumTokens => RuntimeBindingKind::NumTokensU32,
    }
}

/// Resolve the megakernel's neutral buffer table to concrete GPU buffers.
///
/// Arena slots are allocated fresh, sized by `prog.arena_bytes` (the megakernel
/// owns its intermediates — it does NOT borrow the worker's per-op arena);
/// weights resolve via [`resolve_weight`]; runtime inputs and the paged KV
/// cache via [`RuntimeBindings::buffer_for`]. Returns the per-`BufId`
/// `(Buffer, base)` table the operand-table builder indexes, plus the owned
/// arena buffers (the caller keeps them alive across the dispatch).
///
/// When a `residency` set is given, every freshly allocated arena buffer is
/// inserted: the bindless operands are reached only by `gpuAddress`, so they
/// MUST be resident (weights / runtime inputs were inserted at worker init).
pub fn resolve_mega_buffers<W: CanonicalParams + WeightAccessors>(
    prog: &MegaProgram,
    device: &Device,
    weights: &W,
    allocator: &MetalAllocator,
    runtime: &RuntimeBindings,
    // The per-op forward's embed-output buffer, bound to any
    // [`BufferRef::EmbeddedHidden`] source (embed-as-source — see the variant
    // doc). `None` ⇒ a program that references it fails to resolve.
    embedded_hidden: Option<&Buffer>,
    residency: Option<&MetalResidencySet>,
) -> Result<(Vec<ResolvedBuffer>, Vec<Buffer>), WorkerError> {
    // The megakernel's own arena: one fresh zeroed buffer per slot. Shared
    // storage matches the worker arena (and lets Tier-A read the result back).
    let arena: Vec<Buffer> = prog
        .arena_bytes
        .iter()
        .map(|&bytes| {
            let buf = device
                .newBufferWithLength_options(
                    (bytes as usize).max(1),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("mega arena newBufferWithLength_options returned nil");
            if let Some(r) = residency {
                r.insert(&buf);
            }
            buf
        })
        .collect();
    if let Some(r) = residency {
        r.commit();
    }

    let resolved = prog
        .buffers
        .iter()
        .map(|b| match b {
            BufferRef::ArenaSlot(slot) => arena
                .get(*slot as usize)
                .cloned()
                .map(|buf| (buf, 0u64))
                .ok_or(WorkerError::WeightLookupFailed {
                    reason: "wavefront-mega: arena slot out of range",
                }),
            BufferRef::Scratch(_) => Err(WorkerError::WeightLookupFailed {
                reason: "wavefront-mega: serializer emits no Scratch (k_chunks=1 only)",
            }),
            BufferRef::Input(kind) => Ok((runtime.buffer_for(input_to_metal(*kind)).clone(), 0u64)),
            BufferRef::EmbeddedHidden => embedded_hidden.map(|b| (b.clone(), 0u64)).ok_or(
                WorkerError::WeightLookupFailed {
                    reason: "wavefront-mega: EmbeddedHidden source but no embed-output buffer supplied",
                },
            ),
            BufferRef::Weight { bundle, role, loc } => resolve_weight(
                weights,
                allocator,
                &bundle_to_metal(*bundle),
                loc.layer,
                role_to_metal(*role),
                WeightLocator {
                    bucket: loc.bucket,
                    op_idx: loc.op_idx,
                    slot: loc.slot,
                },
            ),
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok((resolved, arena))
}

/// Build the bindless operand table: one `gpuAddress(buf) + base + byte_offset`
/// u64 per [`ferrite_wavefront::mega::OperandSlot`], in operand order. The
/// player casts each to a `device` pointer.
pub fn build_operand_table(prog: &MegaProgram, resolved: &[ResolvedBuffer]) -> Vec<u64> {
    prog.operands
        .iter()
        .map(|sl| {
            let (buf, base) = &resolved[sl.buffer.0 as usize];
            buf.gpuAddress() + *base + sl.byte_offset
        })
        .collect()
}

// ── dispatch ─────────────────────────────────────────────────────────

/// Dispatch the serialized megakernel: bind the five player buffers
/// (`tape / shapes / operands / tape_offsets / flags` at indices 0..5), make
/// every resolved operand buffer resident for the encoder (they are reached
/// only by `gpuAddress`, never `setBuffer`-bound), and launch `P` co-resident
/// threadgroups × 1024 threads, where `P = tape_offsets.len() - 1`.
///
/// Self-contained: it commits its own command buffer and blocks until done —
/// the shape the Tier-A / Tier-B A/B check needs. (A future perf path inlines
/// the dispatch into the forward's command buffer instead.)
pub fn dispatch_mega(
    prog: &MegaProgram,
    resolved: &[ResolvedBuffer],
    operand_addrs: &[u64],
    pipeline: &ComputePipelineState,
    device: &Device,
    queue: &CommandQueue,
) {
    let tape = buffer_from_bytes(device, &prog.tape_bytes());
    let shapes = buffer_from_bytes(device, &prog.shapes_bytes());
    let operands = buffer_from_bytes(device, &u64_le_bytes(operand_addrs));
    let tape_offsets = buffer_from_bytes(device, &prog.tape_offsets_bytes());
    let flags = zeroed_buffer(device, (prog.num_flags.max(1) * 4) as usize);

    let p = (prog.tape_offsets.len() as u32).saturating_sub(1).max(1);

    let cb = queue.commandBuffer().expect("mega cb");
    let enc = cb.computeCommandEncoder().expect("mega enc");
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&tape), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&shapes), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&operands), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&tape_offsets), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&flags), 0, 4);
    }
    // The operands are bindless (gpuAddress) — they must be resident. (The
    // five bound buffers above are made resident by `setBuffer` itself.)
    for (buf, _) in resolved {
        use_resource(&enc, buf, MTLResourceUsage::Read | MTLResourceUsage::Write);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: p as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
}

// ── small buffer helpers (mirror the integration-test harness) ───────

fn buffer_from_bytes(device: &Device, bytes: &[u8]) -> Buffer {
    if bytes.is_empty() {
        return zeroed_buffer(device, 1);
    }
    unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("newBufferWithBytes_length_options returned nil")
    }
}

fn zeroed_buffer(device: &Device, n_bytes: usize) -> Buffer {
    device
        .newBufferWithLength_options(n_bytes.max(1), MTLResourceOptions::StorageModeShared)
        .expect("newBufferWithLength_options returned nil")
}

fn use_resource(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buf: &Buffer,
    usage: MTLResourceUsage,
) {
    unsafe {
        let _: () = msg_send![enc, useResource: &**buf, usage: usage];
    }
}

fn u64_le_bytes(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
