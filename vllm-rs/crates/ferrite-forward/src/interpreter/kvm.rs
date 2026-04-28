// SPDX-License-Identifier: Apache-2.0
//! KVM interpreter runtime helpers — sibling of the host
//! interpreter helpers in [`crate::instr`]. Consumed by the
//! per-canonical wrapper fns the macro emits via
//! `ferrite_forward_macro::interpreter::kvm::emit_wrapper_fn`.
//!
//! ## Surface
//!
//! * [`WeightPtrs`] — every weight tensor the megakernel reads,
//!   as raw GPU pointers + R values (out-feature dim per the per-
//!   layer slab the vendor's `mk` template indexes).
//! * [`RopePtrs`] — cos/sin GPU pointers + max position.
//! * [`ShapeConfig`] — the runtime constants from the canonical's
//!   bounds map. Compile-time ones (num_layers, hidden_dim, …)
//!   are baked into the per-canonical `.cu` source as `#define`s.
//! * [`ExternLaunch`] — function-pointer alias matching the C
//!   signature the per-canonical `.cu` source defines.
//! * [`launch`] — single entry point. Allocates scratch from
//!   `device.caching`, H2Ds the tape into a fresh device buffer,
//!   builds the C arg pack, calls the extern launcher, returns
//!   `(hidden, logits)` as `OwnedTensor`s.
//!
//! ## What's load-bearing here
//!
//! The vendor megakernel expects a specific KV-cache layout
//! (single contiguous `(layer × num_blocks × block_size ×
//! num_kv_heads × head_dim)` slab) that **does not match**
//! ferrite's per-layer `KvCachePool` allocations. Until a
//! parallel kvm-only KV pool is wired up, [`launch`] uses the
//! existing layer-0 view as the K/V pointer; for a single-layer
//! smoke test that's correct, for multi-layer it isn't. This is
//! flagged at the call site rather than silently producing wrong
//! output. See `MEGAKERNEL_HANDOFF.md §kv-layout` for the wiring
//! plan.

#![cfg(feature = "cuda")]

use ferrite_cuda_core::alloc::OwnedTensor;
use ferrite_cuda_core::device::GpuDevice;
use ferrite_cuda_core::driver;
use ferrite_cuda_core::dtype::DType;

use crate::ForwardCtx;
use crate::tile_table::TileEntry;

/// Per-canonical wrapper-fn pointer, generic over the canonical's
/// `Weights` type. Each macro invocation emits one of these per
/// kvm-eligible canonical and a parallel `KVM_WRAPPERS` table that
/// dispatch consults before falling through to the host
/// interpreter's `run`.
pub type KvmWrapperFn<W> = unsafe fn(
    &W,
    &ForwardCtx,
    &mut ferrite_cuda_core::device::GpuDevice,
    &mut Vec<Option<TileEntry>>,
);

/// Raw GPU pointers to every weight tensor the megakernel reads,
/// plus the per-projection out-feature dim (`R`) the vendor's
/// `mk` template needs as a runtime arg. Pointers are valid for
/// the lifetime of the calling forward — borrowed from `Weights`
/// by the wrapper fn before the `launch` call.
pub struct WeightPtrs {
    pub qkv: *mut u8,
    pub qkv_r: i32,
    pub attn_norm: *mut u8,
    pub o: *mut u8,
    pub o_r: i32,
    pub mlp_norm: *mut u8,
    pub up: *mut u8,
    pub up_r: i32,
    pub gate: *mut u8,
    pub gate_r: i32,
    pub down: *mut u8,
    pub down_r: i32,
    pub lm_head_norm: *mut u8,
    pub lm_head: *mut u8,
    pub lm_head_r: i32,
}

/// RoPE cache pointers + max position. cos/sin point at the same
/// allocation when the rope cache packs them as `[max_pos,
/// 2*head_dim/2]` (the host interpreter's representation). The
/// vendor megakernel reads them as separate `[max_pos, head_dim/2]`
/// slabs; the wrapper fn passes the same pointer twice and lets
/// the vendor kernel stride into it.
pub struct RopePtrs {
    pub cos: *mut u8,
    pub sin: *mut u8,
    pub max_pos: i32,
}

/// Runtime-resolved shape constants. Compile-time dims
/// (num_layers, hidden_dim, intermediate_dim, head_dim,
/// num_attention_heads, num_kv_heads, kv_page_size,
/// matmul_*_block_size, num_devices) are baked into the per-
/// canonical `.cu` source as preprocessor `#define`s, so they're
/// `const` from the megakernel's POV. The fields here are the
/// few values that legitimately vary per call (rms_norm_eps from
/// the loaded weights, attn_scale derived from head_dim) or that
/// the runtime helper needs to allocate scratch (vocab_size,
/// batch_size).
pub struct ShapeConfig {
    pub num_layers: i32,
    pub hidden_dim: i32,
    pub head_dim: i32,
    pub num_attention_heads: i32,
    pub num_kv_heads: i32,
    pub intermediate_dim: i32,
    pub num_devices: i32,
    pub vocab_size: i32,
    pub matmul_out_block_size: i32,
    pub matmul_batch_block_size: i32,
    pub kv_page_size: i32,
    pub attn_scale: f32,
    pub rms_norm_eps: f32,
}

/// Function-pointer alias for the per-canonical extern launcher.
/// Mirrors the signature emitted by
/// `ferrite_forward_macro::interpreter::kvm::emit_extern_decl`
/// (and matched by the per-canonical `.cu` source's `extern "C"`
/// definition).
#[allow(clippy::type_complexity, non_snake_case)]
pub type ExternLaunch = unsafe extern "C" fn(
    d_bar: *mut core::ffi::c_void,
    bar_d0: i32,
    bar_d1: i32,
    bar_d2: i32,
    bar_d3: i32,
    d_instructions: *mut core::ffi::c_void,
    num_instructions: i32,
    d_timings: *mut core::ffi::c_void,
    num_timing_rows: i32,
    d_global_instruction_index: *mut core::ffi::c_void,
    d_qkv_weights: *mut core::ffi::c_void,
    qkv_R: i32,
    d_attn_norm_weights: *mut core::ffi::c_void,
    d_o_weights: *mut core::ffi::c_void,
    o_R: i32,
    d_mlp_norm_weights: *mut core::ffi::c_void,
    d_up_weights: *mut core::ffi::c_void,
    up_R: i32,
    d_gate_weights: *mut core::ffi::c_void,
    gate_R: i32,
    d_down_weights: *mut core::ffi::c_void,
    down_R: i32,
    d_lm_head_norm_weights: *mut core::ffi::c_void,
    d_lm_head_weights: *mut core::ffi::c_void,
    lm_head_R: i32,
    d_k_cache: *mut core::ffi::c_void,
    kv_total_pages: i32,
    kv_page_size_runtime: i32,
    d_v_cache: *mut core::ffi::c_void,
    d_rope_cos: *mut core::ffi::c_void,
    max_pos: i32,
    d_rope_sin: *mut core::ffi::c_void,
    d_hidden_states: *mut core::ffi::c_void,
    batch_size_arg: i32,
    d_rms_rope_intermediates: *mut core::ffi::c_void,
    d_rms_gate_intermediates: *mut core::ffi::c_void,
    d_q_post_rope: *mut core::ffi::c_void,
    d_attn_out: *mut core::ffi::c_void,
    d_silu_out: *mut core::ffi::c_void,
    d_rms_lm_head_intermediates: *mut core::ffi::c_void,
    d_logits: *mut core::ffi::c_void,
    vocab_size_arg: i32,
    d_position_ids: *mut core::ffi::c_void,
    num_position_ids: i32,
    d_kv_append_indices: *mut core::ffi::c_void,
    d_prefill_qo_indptr: *mut core::ffi::c_void,
    num_prefill_qo: i32,
    d_prefill_kv_indptr: *mut core::ffi::c_void,
    d_prefill_kv_indices: *mut core::ffi::c_void,
    num_prefill_kv_indices: i32,
    d_prefill_kv_last_page_len: *mut core::ffi::c_void,
    d_decode_kv_indptr: *mut core::ffi::c_void,
    num_decode_seqs: i32,
    d_decode_kv_indices: *mut core::ffi::c_void,
    num_decode_kv_indices: i32,
    d_decode_kv_last_page_len: *mut core::ffi::c_void,
    attn_scale: f32,
    rms_norm_eps: f32,
    num_pages: i32,
    num_prefill_tokens: i32,
    dev_idx: i32,
    raw_stream: *mut core::ffi::c_void,
) -> i32;

/// Encode the static tape into a freshly allocated device buffer,
/// allocate every scratch tensor `globals_t` needs, then call the
/// per-canonical extern launcher. Returns `(hidden, logits)` as
/// owned tensors that outlive the call.
///
/// # Safety
///
/// * Every pointer in `weights` and `rope` must reference live
///   GPU memory at least until the returned `OwnedTensor`s are
///   read on `device.compute_stream` (the megakernel runs
///   asynchronously on that stream).
/// * `tape` must be the static tape generated by
///   `ferrite_forward_macro::interpreter::kvm::emit_tape_static`
///   for the same canonical whose extern launcher is `extern_fn`.
/// * `ctx.kv_cache` must have a layout compatible with the
///   vendor megakernel's `g.k_cache` / `g.v_cache` indexing.
///   This is currently only true for single-layer models — see
///   the module docstring.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch(
    extern_fn: ExternLaunch,
    ctx: &ForwardCtx,
    device: &mut GpuDevice,
    tape: &[[i32; 32]],
    weights: WeightPtrs,
    rope: RopePtrs,
    shape: ShapeConfig,
) -> Result<(OwnedTensor, OwnedTensor), KvmLaunchError> {
    let stream = device.compute_stream;
    let batch_size = ctx.input_ids.shape()[0] as i32;
    let num_pages = ctx.kv_cache.num_blocks as i32;
    let kv_page_size_runtime = ctx.kv_cache.block_size as i32;

    if shape.num_layers != ctx.kv_cache.num_layers as i32 {
        return Err(KvmLaunchError::LayerMismatch {
            shape_layers: shape.num_layers,
            kv_layers: ctx.kv_cache.num_layers as i32,
        });
    }

    // ── Tape — H2D the static into a fresh device buffer ──
    let tape_bytes = std::mem::size_of_val(tape);
    let tape_dev = device
        .caching
        .alloc_tensor(&[tape.len(), 32], DType::I32);
    unsafe {
        driver::memcpy_htod_async(
            tape_dev.raw_ptr(),
            tape.as_ptr() as *const u8,
            tape_bytes,
            stream,
        )
        .map_err(KvmLaunchError::Driver)?;
    }

    // ── Scratch — every per-call buffer in vendor's globals_t ──
    let h = shape.hidden_dim as usize;
    let bs = batch_size as usize;
    let qh = shape.num_attention_heads as usize;
    let hd = shape.head_dim as usize;
    let inter = shape.intermediate_dim as usize / shape.num_devices.max(1) as usize;

    let bar = device
        .caching
        .alloc_tensor(&[shape.num_layers as usize, 16, 16, 16], DType::U32);
    let timings = device.caching.alloc_tensor(&[1, 4], DType::I32);
    let global_instr_idx = device.caching.alloc_tensor(&[1], DType::I32);
    let hidden = device.caching.alloc_tensor(&[bs, h], DType::BF16);
    let rms_rope_inter = device.caching.alloc_tensor(&[bs, h], DType::BF16);
    let rms_gate_inter = device.caching.alloc_tensor(&[bs, h], DType::BF16);
    let q_post_rope = device.caching.alloc_tensor(&[bs, qh * hd], DType::BF16);
    let attn_out = device.caching.alloc_tensor(&[bs, qh * hd], DType::BF16);
    let silu_out = device.caching.alloc_tensor(&[bs, inter], DType::BF16);
    let rms_lm_head_inter = device.caching.alloc_tensor(&[bs, h], DType::BF16);
    let logits = device
        .caching
        .alloc_tensor(&[bs, shape.vocab_size as usize], DType::F32);

    // ── Sequence metadata — pass ctx tensors through directly ──
    let position_ids_ptr = ctx.positions.raw_ptr() as *mut u8;
    let num_position_ids = ctx.positions.shape()[0] as i32;
    let kv_append_indices_ptr = ctx.slot_mapping.raw_ptr() as *mut u8;

    // ── Initialize barrier + global instruction index to 0 ──
    let zero_bytes_bar = bar.size_bytes();
    let zero_bytes_idx = global_instr_idx.size_bytes();
    unsafe {
        driver::memset_d8(bar.raw_ptr(), 0, zero_bytes_bar, stream)
            .map_err(KvmLaunchError::Driver)?;
        driver::memset_d8(global_instr_idx.raw_ptr(), 0, zero_bytes_idx, stream)
            .map_err(KvmLaunchError::Driver)?;
    }

    // KV cache pointers: layer-0 view. See module docstring.
    let k_cache_view = ctx.kv_cache.k_cache(0);
    let v_cache_view = ctx.kv_cache.v_cache(0);

    let raw_stream = stream as *mut core::ffi::c_void;

    let rc = unsafe {
        extern_fn(
            bar.raw_ptr() as *mut _,
            shape.num_layers,
            16,
            16,
            16,
            tape_dev.raw_ptr() as *mut _,
            tape.len() as i32,
            timings.raw_ptr() as *mut _,
            1,
            global_instr_idx.raw_ptr() as *mut _,
            weights.qkv as *mut _,
            weights.qkv_r,
            weights.attn_norm as *mut _,
            weights.o as *mut _,
            weights.o_r,
            weights.mlp_norm as *mut _,
            weights.up as *mut _,
            weights.up_r,
            weights.gate as *mut _,
            weights.gate_r,
            weights.down as *mut _,
            weights.down_r,
            weights.lm_head_norm as *mut _,
            weights.lm_head as *mut _,
            weights.lm_head_r,
            k_cache_view.raw_ptr() as *mut _,
            num_pages,
            kv_page_size_runtime,
            v_cache_view.raw_ptr() as *mut _,
            rope.cos as *mut _,
            rope.max_pos,
            rope.sin as *mut _,
            hidden.raw_ptr() as *mut _,
            batch_size,
            rms_rope_inter.raw_ptr() as *mut _,
            rms_gate_inter.raw_ptr() as *mut _,
            q_post_rope.raw_ptr() as *mut _,
            attn_out.raw_ptr() as *mut _,
            silu_out.raw_ptr() as *mut _,
            rms_lm_head_inter.raw_ptr() as *mut _,
            logits.raw_ptr() as *mut _,
            shape.vocab_size,
            position_ids_ptr as *mut _,
            num_position_ids,
            kv_append_indices_ptr as *mut _,
            // Prefill / decode CSR splits — pass ctx slices
            // through; the megakernel inspects num_prefill_tokens
            // to choose decode vs prefill paths per row.
            ctx.cu_seqlens_q.raw_ptr() as *mut _,
            ctx.cu_seqlens_q.shape()[0].saturating_sub(1) as i32,
            ctx.cu_seqlens_q.raw_ptr() as *mut _,
            ctx.block_table.raw_ptr() as *mut _,
            (ctx.block_table.shape()[0] * ctx.block_table.shape()[1]) as i32,
            ctx.seqused_k.raw_ptr() as *mut _,
            ctx.cu_seqlens_q.raw_ptr() as *mut _,
            batch_size,
            ctx.block_table.raw_ptr() as *mut _,
            (ctx.block_table.shape()[0] * ctx.block_table.shape()[1]) as i32,
            ctx.seqused_k.raw_ptr() as *mut _,
            shape.attn_scale,
            shape.rms_norm_eps,
            num_pages,
            ctx.max_seqlen_q as i32,
            0,
            raw_stream,
        )
    };

    if rc != 0 {
        return Err(KvmLaunchError::LaunchReturned(rc));
    }
    Ok((hidden, logits))
}

#[derive(Debug)]
pub enum KvmLaunchError {
    Driver(anyhow::Error),
    LaunchReturned(i32),
    LayerMismatch { shape_layers: i32, kv_layers: i32 },
}

impl std::fmt::Display for KvmLaunchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Driver(e) => write!(f, "kvm launch driver error: {e}"),
            Self::LaunchReturned(rc) => {
                write!(f, "kvm extern launcher returned cudaError={rc}")
            }
            Self::LayerMismatch {
                shape_layers,
                kv_layers,
            } => write!(
                f,
                "kvm launch: canonical num_layers={shape_layers} but KvCachePool has \
                 num_layers={kv_layers}; vendor megakernel requires them equal"
            ),
        }
    }
}

impl std::error::Error for KvmLaunchError {}
