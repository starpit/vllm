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
//!      [`ferrite_wavefront::metal_tape::BufferRef`] into a concrete
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
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLDevice, MTLResourceOptions, MTLResourceUsage, MTLSize,
};

use ferrite_cuda_core::MetalAllocator;
use ferrite_metal_kernels::quantized::{DequantDtype, ScaleDtype};
use ferrite_metal_kernels::residency::MetalResidencySet;
use ferrite_metal_kernels::specialized_pipeline_cache::PipelineKey;
use ferrite_wavefront::mega::MegaProgram;
use ferrite_wavefront::metal_tape::{BufferRef, InputKind, WeightBundle, WeightRole};

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
    // CRITICAL: do NOT insert arena buffers into the persistent residency set
    // — `dispatch_mega` already calls `use_resource(...)` for every arena
    // buffer on the encoder, which is the correct per-dispatch declaration.
    // The persistent residency set is for buffers that live ACROSS dispatches
    // (weights). Inserting fresh-per-step arena buffers there leaks them as
    // Metal "wired" memory — N decode steps × ~3000 slots accumulates GiBs
    // and pushes the OS into swap (observed 30 GiB wired at gpu-util=0.4).
    let _ = residency; // mark intentionally unused for arena allocation
    let arena: Vec<Buffer> = prog
        .arena_bytes
        .iter()
        .map(|&bytes| {
            device
                .newBufferWithLength_options(
                    (bytes as usize).max(1),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("mega arena newBufferWithLength_options returned nil")
        })
        .collect();

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

// ── cached dispatch (production path) ────────────────────────────────

/// All the per-program state `dispatch_mega` needs but which is STABLE across
/// decode steps (the arena is a fresh per-program allocation but doesn't have
/// to be re-allocated every step; the tape/shapes/tape-offsets are immutable;
/// the operands gpuAddress table is stable as long as the resolved buffers'
/// gpuAddresses are stable, which they are once the arena is cached).
///
/// Building this once and re-using it removed ~5 ms/decode-step of buffer
/// thrash (measured 12.7 → ~7.5 ms/tok TPOT on Llama-3.2-1B-4bit, where the
/// kernel-only `mega_exec` already sat at ~7.5 ms — i.e. the per-step setup
/// was the same order of magnitude as the kernel itself before this).
pub struct MegaDispatchCache {
    pub arena: Vec<Buffer>,
    pub resolved: Vec<ResolvedBuffer>,
    pub tape_buf: Buffer,
    pub shapes_buf: Buffer,
    pub operands_buf: Buffer,
    pub tape_offsets_buf: Buffer,
    pub flags_buf: Buffer,
    pub num_flags: u32,
    pub p: u32,
}

impl MegaDispatchCache {
    /// Build the cache once. After this, `dispatch_mega_cached` does no per-
    /// step allocation — it just zeroes flags (blit) and dispatches.
    pub fn build<W: CanonicalParams + WeightAccessors>(
        prog: &MegaProgram,
        device: &Device,
        weights: &W,
        allocator: &MetalAllocator,
        runtime: &RuntimeBindings,
        embedded_hidden: Option<&Buffer>,
    ) -> Result<Self, WorkerError> {
        let (resolved, arena) = resolve_mega_buffers(
            prog,
            device,
            weights,
            allocator,
            runtime,
            embedded_hidden,
            None,
        )?;
        let operand_addrs = build_operand_table(prog, &resolved);
        let tape_buf = buffer_from_bytes(device, &prog.tape_bytes());
        let shapes_buf = buffer_from_bytes(device, &prog.shapes_bytes());
        let operands_buf = buffer_from_bytes(device, &u64_le_bytes(&operand_addrs));
        let tape_offsets_buf = buffer_from_bytes(device, &prog.tape_offsets_bytes());
        let flags_buf = zeroed_buffer(device, (prog.num_flags.max(1) * 4) as usize);
        let p = (prog.tape_offsets.len() as u32).saturating_sub(1).max(1);

        // PERF DIAG (one-shot, on first dispatch): per-opcode count + BW
        // estimate. mega_exec is dominated by qmv weight reads; this tells us
        // total bytes-read per arm so we know which arm sets the BW floor.
        // FERRITE_WAVEFRONT_PROFILE=1 to enable.
        if std::env::var_os("FERRITE_WAVEFRONT_PROFILE").is_some() {
            print_tape_profile(prog);
        }

        Ok(Self {
            arena,
            resolved,
            tape_buf,
            shapes_buf,
            operands_buf,
            tape_offsets_buf,
            flags_buf,
            num_flags: prog.num_flags,
            p,
        })
    }
}

/// One-shot per-opcode count + BW estimate. The qmv arms dominate weight
/// read — every other arm is activation-band or scratch. The estimate is
/// `bytes_read / measured_BW`, which gives a kernel-time floor per arm
/// against which we can target optimisation.
fn print_tape_profile(prog: &MegaProgram) {
    use ferrite_wavefront::mega::op_kind;
    // Per-arm counters + bytes accumulator.
    let mut count = [0u64; 16];
    let mut bytes = [0u64; 16]; // weight read bytes (qmv) or activation bytes (rest)
    for ins in &prog.tape {
        // opcode::COMPUTE == 0
        if ins[0] != 0 {
            // BARRIER / PUBLISH / ACQUIRE / WAIT / SIGNAL — count via opcode index
            // (these are NOT op_kind values; bucketize as "other").
            count[15] += 1;
            continue;
        }
        let sb = ins[1] as usize;
        let shape = &prog.shapes[sb];
        let op = shape[0];
        count[op.min(15) as usize] += 1;
        match op {
            op_kind::QMV | op_kind::QMV_QUAD | op_kind::QMV_COH => {
                // qmv: weight read = K_window × N bytes, 4-bit packed → bytes ≈ K*N/2
                // shape (op, k_window, n, row_vec, …); row_vec=0 ⇒ row_vec=k.
                let k = shape[1] as u64;
                let n = shape[2] as u64;
                // 4-bit packed weight; group-aligned scales/biases are smaller-cost,
                // dominated by the weight read.
                bytes[op.min(15) as usize] += k * n / 2;
            }
            op_kind::RMSNORM | op_kind::SILU_MUL | op_kind::ADD | op_kind::ROPE => {
                // activation-only ops; reads ≈ n bf16 elements per op.
                bytes[op.min(15) as usize] += (shape[1] as u64) * 2;
            }
            op_kind::ATTN => {
                // attn reads Q (head_dim*qh_count) + K/V cache row blocks; rough estimate.
                let head_dim = shape[1] as u64;
                let n_q = shape[2] as u64;
                bytes[op.min(15) as usize] += head_dim * n_q * 4;
            }
            op_kind::SUM_REDUCE => {
                // SumReduce reads num_partials × n bf16.
                let n = shape[1] as u64;
                let np = shape[2] as u64;
                bytes[op.min(15) as usize] += np * n * 2;
            }
            op_kind::ROPE_APPEND => {
                let head_dim = shape[1] as u64;
                let n_kv = shape[2] as u64;
                bytes[op.min(15) as usize] += head_dim * n_kv * 4;
            }
            _ => {}
        }
    }
    let name = |op: u32| -> &'static str {
        match op {
            op_kind::QMV => "qmv",
            op_kind::PUBLISH => "publish",
            op_kind::ACQUIRE => "acquire",
            op_kind::RMSNORM => "rmsnorm",
            op_kind::SILU_MUL => "silu_mul",
            op_kind::ROPE => "rope",
            op_kind::ATTN => "attn",
            op_kind::ADD => "add",
            op_kind::ROPE_APPEND => "rope_append",
            op_kind::QMV_COH => "qmv_coh",
            op_kind::SUM_REDUCE => "sum_reduce",
            op_kind::QMV_QUAD => "qmv_quad",
            15 => "other (barrier/publish/acquire tape opcodes)",
            _ => "?",
        }
    };
    // Apple M5 measured ~87 GB/s under the mega's persistent-kernel pattern.
    const MEGA_BW_GBPS: f64 = 87.0;
    eprintln!("[wf-profile] per-arm tape breakdown (estimated kernel-time floor at {:.0} GB/s):", MEGA_BW_GBPS);
    let mut total_bytes: u64 = 0;
    for op in [
        op_kind::QMV, op_kind::QMV_QUAD, op_kind::QMV_COH, op_kind::ATTN,
        op_kind::RMSNORM, op_kind::SILU_MUL, op_kind::ADD, op_kind::ROPE,
        op_kind::ROPE_APPEND, op_kind::SUM_REDUCE,
    ] {
        let idx = op.min(15) as usize;
        if count[idx] == 0 {
            continue;
        }
        total_bytes += bytes[idx];
        let est_ms = (bytes[idx] as f64) / (MEGA_BW_GBPS * 1.0e9) * 1.0e3;
        eprintln!(
            "[wf-profile]   {:12}: count={:5}  bytes={:8.2} MB  est={:5.2} ms",
            name(op), count[idx], (bytes[idx] as f64) / 1.0e6, est_ms,
        );
    }
    let total_est = (total_bytes as f64) / (MEGA_BW_GBPS * 1.0e9) * 1.0e3;
    eprintln!(
        "[wf-profile]   {:12}: count={:5}",
        name(15), count[15],
    );
    eprintln!(
        "[wf-profile] sum: bytes={:.2} MB est={:.2} ms (vs measured mega_exec ~7.5 ms)",
        (total_bytes as f64) / 1.0e6, total_est,
    );
}

/// Dispatch using a pre-built [`MegaDispatchCache`]. Per-step cost is just
/// the flags zero + compute encoder + dispatch + commit/wait — no Metal
/// buffer creation.
pub fn dispatch_mega_cached(cache: &MegaDispatchCache, pipeline: &ComputePipelineState, queue: &CommandQueue) {
    // PERF DIAG (one-shot per process): Metal reports the max threads/TG the
    // KERNEL CAN ACTUALLY FIT given its register footprint. If this drops
    // below 1024, we're register-pressure-limited (kernel uses too many
    // registers per thread, Apple GPU can't fit a full 1024-thread TG) and
    // the per-core simdgroup occupancy is capped below 32, which throttles
    // BW directly. Use the OnceLock at static scope to print this exactly once.
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        use ::objc2_metal::MTLComputePipelineState;
        let max_tpg = pipeline.maxTotalThreadsPerThreadgroup();
        eprintln!(
            "[wf-perf] player pipeline max_threads_per_threadgroup = {} \
             ({:.0}% of 1024 — anything below 1024 means the kernel can't \
             fit a full TG due to register pressure, capping per-core \
             simdgroup occupancy)",
            max_tpg,
            100.0 * (max_tpg as f64) / 1024.0,
        );
    });
    let cb = queue.commandBuffer().expect("mega cb");
    // Zero the atomic flags from the prior step. fillBuffer on a Blit encoder
    // is the fastest way to clear u32 words on the GPU without a host copy.
    let flags_len = (cache.num_flags.max(1) * 4) as usize;
    let blit = cb.blitCommandEncoder().expect("mega blit enc");
    let range = objc2_foundation::NSRange::new(0, flags_len);
    unsafe {
        blit.fillBuffer_range_value(&cache.flags_buf, range, 0u8);
    }
    blit.endEncoding();

    let enc = cb.computeCommandEncoder().expect("mega enc");
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&cache.tape_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&cache.shapes_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&cache.operands_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&cache.tape_offsets_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&cache.flags_buf), 0, 4);
    }
    for (buf, _) in &cache.resolved {
        use_resource(&enc, buf, MTLResourceUsage::Read | MTLResourceUsage::Write);
    }
    let k_replicas = std::env::var("FERRITE_WAVEFRONT_K")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(1)
        .max(1);
    let tg_threads = std::env::var("FERRITE_WAVEFRONT_TG_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1024);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: cache.p as usize, height: k_replicas as usize, depth: 1 },
        MTLSize { width: tg_threads, height: 1, depth: 1 },
    );
    enc.endEncoding();
    let t_exec = std::time::Instant::now();
    cb.commit();
    cb.waitUntilCompleted();
    eprintln!(
        "[wf-perf] mega_exec={:.3}ms K={} (cached)",
        t_exec.elapsed().as_secs_f64() * 1e3,
        k_replicas
    );
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
    // PERF DIAG (droppable): FERRITE_WAVEFRONT_K replicates each worker into K
    // co-resident TGs (grid = P×K). K=1 is the production shape. K>1 is for
    // measuring BW headroom: if mega_exec roughly halves at K=2, the GPU has
    // outstanding-loads slack we're not exploiting; if it stays flat, the
    // 87 GB/s ceiling is real and we need different attack. (At K>1 with the
    // current kernel each replica runs the SAME tape — writes race; output is
    // garbage — so it's a diagnostic dispatch shape, not a correctness one.)
    let k_replicas = std::env::var("FERRITE_WAVEFRONT_K")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(1)
        .max(1);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: p as usize,
            height: k_replicas as usize,
            depth: 1,
        },
        MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    // PERF DIAG (droppable): time JUST the kernel (commit→wait), separate from
    // the per-step buffer rebuilds above — a production path caches those.
    let t_exec = std::time::Instant::now();
    cb.commit();
    cb.waitUntilCompleted();
    eprintln!(
        "[wf-perf] mega_exec={:.3}ms K={}",
        t_exec.elapsed().as_secs_f64() * 1e3,
        k_replicas
    );
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
