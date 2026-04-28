// SPDX-License-Identifier: Apache-2.0
//! Runtime support for the per-canonical KvmMega marshaling
//! wrapper.
//!
//! The proc-macro emits one `kvm_mega_<canonical>_m_<wp>` Rust
//! wrapper per (canonical, workload) bucket of every kvm-eligible
//! canonical. Each wrapper:
//!
//!   1. Extracts ~9 layered-weight base pointers from
//!      `Weights<W>` accessors (each is the `[L, R, H]` block
//!      that `crates/ferrite-forward/src/loaders.rs` allocates
//!      contiguously);
//!   2. Pulls the RoPE cos/sin tables off the same `Weights<W>`;
//!   3. Looks up the bucket's `KVM_FULL_M_<wp>` tape static (one
//!      contiguous instruction list, backbone rows then lm_head
//!      rows — see `interpreters/kvm_mega::emit_kvm_full_program`);
//!   4. Calls [`launch_kvm_mega`] with the assembled arg bundle.
//!
//! [`launch_kvm_mega`] does the per-call work: allocates the Bar
//! semaphore buffer, timings, global instruction index, device
//! tape, and ~7 activation scratch tensors; runs the paged-KV
//! CSR build via `paged_kv::device::build_paged_kv_metadata_on_device`;
//! and finally invokes the canonical's
//! `tk_megakernel_<canonical>_launch` extern fn pointer with the
//! fully marshaled args.
//!
//! The helper itself is canonical-agnostic — it takes the
//! `extern "C"` launcher fn pointer (whose name is the only
//! per-canonical thing) and the canonical's compile-time shape
//! constants as `KvmShapeConfig`. This keeps the per-bucket
//! emitted Rust trivially small (one fn ptr + one struct
//! literal + a delegating call).

use crate::ForwardCtx;
use crate::paged_kv;
use anyhow::Result;
use ferrite_cuda_core::alloc::OwnedTensor;
use ferrite_cuda_core::device::GpuDevice;
use ferrite_cuda_core::driver;
use ferrite_cuda_core::dtype::DType;

/// Layered-weight base pointers + reduction-dim sizes, mirroring
/// the C launcher's weight argument block. Each `*mut u8` is the
/// base of an `[L, R, H]` (or `[L, R]` for norms) contiguous
/// block — see `loaders::load_layered_*`. The `*_r` ints are the
/// runtime reduction-dim per layer (e.g. `qkv_r = q_size +
/// 2*kv_size`); vendor's `weights_t` template carries them as
/// runtime dims.
#[derive(Debug, Clone, Copy)]
pub struct KvmWeightPtrs {
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

/// RoPE cos/sin tables (both `[max_pos, head_dim]` f32) plus
/// the sequence-length compile-time max.
#[derive(Debug, Clone, Copy)]
pub struct KvmRopePtrs {
    pub cos: *mut u8,
    pub sin: *mut u8,
    pub max_pos: i32,
}

/// Shape constants the wrapper bakes from canonical bounds at
/// codegen time. Vendor's `globals_t` template parameters
/// (`LLAMA_NUM_LAYERS` etc.) are compiled into the
/// `tk_megakernel_<canonical>_launch` symbol; these copies are
/// what the helper needs at run time to size scratch and CSR
/// buffers correctly.
#[derive(Debug, Clone, Copy)]
pub struct KvmShapeConfig {
    pub num_layers: i32,
    pub hidden_dim: i32,
    pub head_dim: i32,
    pub num_attention_heads: i32,
    pub intermediate_dim: i32,
    pub num_devices: i32,
    pub batch_size: i32,
    pub vocab_size: i32,
    pub matmul_out_block_size: i32,
    pub matmul_batch_block_size: i32,
    pub kv_page_size: i32,
    pub attn_scale: f32,
    pub rms_norm_eps: f32,
}

/// 1:1 with the C signature emitted by
/// `interpreters/kvm_mega::emit_kvm_extern_decl`. Wrappers cast
/// the per-canonical extern fn item to this type and pass it in.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub type KvmExternLaunch = unsafe extern "C" fn(
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
    qkv_r: i32,
    d_attn_norm_weights: *mut core::ffi::c_void,
    d_o_weights: *mut core::ffi::c_void,
    o_r: i32,
    d_mlp_norm_weights: *mut core::ffi::c_void,
    d_up_weights: *mut core::ffi::c_void,
    up_r: i32,
    d_gate_weights: *mut core::ffi::c_void,
    gate_r: i32,
    d_down_weights: *mut core::ffi::c_void,
    down_r: i32,
    d_lm_head_norm_weights: *mut core::ffi::c_void,
    d_lm_head_weights: *mut core::ffi::c_void,
    lm_head_r: i32,
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

/// Bar buffer dimensions, per vendor's
/// `barriers = pgl<gl<uint, -1, -1, -1, -1>, num_devices>`.
/// The 4 dims are `[layer, opcode-1, batch_block, out_block]`
/// in vendor's indexing convention (see python_vm.py); we size
/// each upper bound conservatively from the canonical's shape
/// so the kernel never OOBs.
fn bar_dims(shape: &KvmShapeConfig) -> (i32, i32, i32, i32) {
    let layer_dim = shape.num_layers;
    // Vendor has 13 opcodes; round up to 16 to leave headroom
    // for new ops without bumping the wrapper.
    let opcode_dim = 16;
    let batch_block = shape.matmul_batch_block_size.max(1);
    let batch_block_dim = ((shape.batch_size + batch_block - 1) / batch_block).max(1);
    // Output-block dim covers per-row matmul opcodes; vendor's
    // `num_output_blocks` is hidden_dim / matmul_out_block_size.
    let out_block_dim = (shape.hidden_dim / shape.matmul_out_block_size.max(1)).max(1);
    (layer_dim, opcode_dim, batch_block_dim, out_block_dim)
}

/// `num_timing_rows` for the timings buffer. Vendor's runtime
/// uses one row per opcode invocation; sized to a generous
/// upper bound so we never write past the buffer in debug
/// builds. (Production turns timings off via the
/// `TIMING_RECORD_ENABLED` template parameter, but the
/// `globals_t` field is unconditional, so we still need
/// non-null storage.)
const NUM_TIMING_ROWS: i32 = 16384;

/// The runtime-side workhorse for one KvmMega forward call. The
/// per-canonical `kvm_mega_<canonical>_m_<wp>` wrapper
/// (codegen-emitted) hands us the extern launcher fn pointer,
/// the bucket's `KVM_FULL_M_<wp>` tape, weight/RoPE pointers
/// and shape constants. We allocate per-call scratch, build the
/// paged-KV CSR, and invoke the launcher.
///
/// On success we return the two `OwnedTensor`s the dispatch
/// code (`forward` / `forward_backbone`) consumes via
/// `tiles[backbone_slot]` and `tiles[terminal_slot]`:
///
/// * `hidden_states` — the post-final-norm activation buffer
///   the megakernel writes (filled before the lm_head matmul).
///   `forward_backbone` returns this directly.
/// * `logits` — the lm_head output. `forward` returns this.
///
/// All other scratch (Bar, timings, q_post_rope, etc.) is
/// caching-allocator-owned and freed on return — we
/// `stream_synchronize` at the end so the kernel has finished
/// reading from each block before its `Drop` runs.
///
/// # Safety
/// - Every pointer in `weights` / `rope` must be a valid
///   contiguous device allocation matching vendor's expected
///   shape (see [`crate::loaders`]).
/// - `extern_launch` must be the per-canonical
///   `tk_megakernel_<canonical>_launch` whose `globals_t`
///   template parameters match `shape`.
/// - `ctx` must outlive the launch; the kernel reads from KV
///   cache, RoPE tables, and the activation scratch
///   synchronously on the compute stream.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_kvm_mega(
    extern_launch: KvmExternLaunch,
    ctx: &ForwardCtx,
    device: &mut GpuDevice,
    full_tape: &[[i32; 32]],
    weights: KvmWeightPtrs,
    rope: KvmRopePtrs,
    shape: KvmShapeConfig,
) -> Result<(OwnedTensor, OwnedTensor)> {
    let pool = ctx.kv_cache;
    let kv_total_pages = (pool.num_layers * pool.num_blocks) as i32;
    let kv_page_size_runtime = pool.block_size as i32;
    let num_pages_arg = pool.num_blocks as i32;
    let dev_idx = 0;
    let stream = device.compute_stream;

    // Paged-KV CSR metadata — D2H, host build, H2D back. Tiny
    // (kilobytes); see `paged_kv::device` for cost analysis.
    let device_paged = paged_kv::device::build_paged_kv_metadata_on_device(
        device,
        ctx.block_table,
        ctx.seqused_k,
        ctx.cu_seqlens_q,
        ctx.slot_mapping,
        ctx.positions,
        shape.kv_page_size,
    )?;

    // Bar / timings / global_instruction_index — small device
    // scratch that vendor's `globals_t` references unconditionally.
    let (bar_d0, bar_d1, bar_d2, bar_d3) = bar_dims(&shape);
    let bar_elems = (bar_d0 as usize) * (bar_d1 as usize) * (bar_d2 as usize) * (bar_d3 as usize);
    let bar = device.alloc_gpu_tensor_zeroed(&[bar_elems], DType::U32);
    let timings_elems = (NUM_TIMING_ROWS as usize) * 128; // TIMING_WIDTH=128 in vendor
    let timings = device.caching.alloc_tensor(&[timings_elems], DType::I32);
    let global_instr_idx = device.alloc_gpu_tensor_zeroed(&[1], DType::I32);

    // Device tape. `full_tape` is `&'static [[i32; 32]]`; flatten
    // to bytes and H2D into a fresh OwnedTensor.
    let num_instructions = full_tape.len() as i32;
    let tape_owned: OwnedTensor = {
        let owned = device
            .caching
            .alloc_tensor(&[full_tape.len() * 32], DType::I32);
        let bytes = full_tape.len() * 32 * 4;
        let src_ptr = full_tape.as_ptr() as *const u8;
        unsafe {
            driver::memcpy_htod_async(owned.as_gpu_tensor().raw_ptr(), src_ptr, bytes, stream)?;
        }
        owned
    };

    // Activation scratch — sized per vendor's
    // `globals_t::activations_*` shapes. All bf16.
    let hidden_per_dev = (shape.hidden_dim / shape.num_devices.max(1)) as usize;
    let inter_per_dev = (shape.intermediate_dim / shape.num_devices.max(1)) as usize;
    let q_per_dev =
        (shape.num_attention_heads * shape.head_dim / shape.num_devices.max(1)) as usize;
    let batch_size = shape.batch_size as usize;
    let vocab_size = shape.vocab_size as usize;

    let alloc_act = |dev: &mut GpuDevice, dim1: usize| -> OwnedTensor {
        dev.caching.alloc_tensor(&[batch_size, dim1], DType::BF16)
    };

    let hidden_states = alloc_act(device, shape.hidden_dim as usize);
    let rms_rope_intermediates = alloc_act(device, hidden_per_dev);
    let rms_gate_intermediates = alloc_act(device, hidden_per_dev);
    let q_post_rope = alloc_act(device, q_per_dev);
    let attn_out = alloc_act(device, hidden_per_dev);
    let silu_out = alloc_act(device, inter_per_dev);
    let rms_lm_head_intermediates = alloc_act(device, hidden_per_dev);
    let logits = alloc_act(device, vocab_size);

    let cv = |t: &OwnedTensor| -> *mut core::ffi::c_void {
        t.as_gpu_tensor().raw_ptr() as *mut core::ffi::c_void
    };

    // Fire the launcher. Field order MUST match
    // `emit_kvm_extern_decl` in
    // `crates/ferrite-forward-macro/src/interpreters/kvm_mega.rs`.
    let rc = unsafe {
        extern_launch(
            bar.raw_ptr() as *mut core::ffi::c_void,
            bar_d0,
            bar_d1,
            bar_d2,
            bar_d3,
            tape_owned.as_gpu_tensor().raw_ptr() as *mut core::ffi::c_void,
            num_instructions,
            timings.as_gpu_tensor().raw_ptr() as *mut core::ffi::c_void,
            NUM_TIMING_ROWS,
            global_instr_idx.raw_ptr() as *mut core::ffi::c_void,
            weights.qkv as *mut core::ffi::c_void,
            weights.qkv_r,
            weights.attn_norm as *mut core::ffi::c_void,
            weights.o as *mut core::ffi::c_void,
            weights.o_r,
            weights.mlp_norm as *mut core::ffi::c_void,
            weights.up as *mut core::ffi::c_void,
            weights.up_r,
            weights.gate as *mut core::ffi::c_void,
            weights.gate_r,
            weights.down as *mut core::ffi::c_void,
            weights.down_r,
            weights.lm_head_norm as *mut core::ffi::c_void,
            weights.lm_head as *mut core::ffi::c_void,
            weights.lm_head_r,
            pool.k_cache_base_ptr() as *mut core::ffi::c_void,
            kv_total_pages,
            kv_page_size_runtime,
            pool.v_cache_base_ptr() as *mut core::ffi::c_void,
            rope.cos as *mut core::ffi::c_void,
            rope.max_pos,
            rope.sin as *mut core::ffi::c_void,
            cv(&hidden_states),
            shape.batch_size,
            cv(&rms_rope_intermediates),
            cv(&rms_gate_intermediates),
            cv(&q_post_rope),
            cv(&attn_out),
            cv(&silu_out),
            cv(&rms_lm_head_intermediates),
            cv(&logits),
            shape.vocab_size,
            device_paged.position_ids.as_gpu_tensor().raw_ptr() as *mut core::ffi::c_void,
            device_paged.num_position_ids,
            device_paged.kv_append_indices.as_gpu_tensor().raw_ptr() as *mut core::ffi::c_void,
            device_paged.prefill_qo_indptr.as_gpu_tensor().raw_ptr() as *mut core::ffi::c_void,
            device_paged.num_prefill_seqs,
            device_paged.prefill_kv_indptr.as_gpu_tensor().raw_ptr() as *mut core::ffi::c_void,
            device_paged.prefill_kv_indices.as_gpu_tensor().raw_ptr() as *mut core::ffi::c_void,
            device_paged.num_prefill_kv_indices,
            device_paged
                .prefill_kv_last_page_len
                .as_gpu_tensor()
                .raw_ptr() as *mut core::ffi::c_void,
            device_paged.decode_kv_indptr.as_gpu_tensor().raw_ptr() as *mut core::ffi::c_void,
            device_paged.num_decode_seqs,
            device_paged.decode_kv_indices.as_gpu_tensor().raw_ptr() as *mut core::ffi::c_void,
            device_paged.num_decode_kv_indices,
            device_paged
                .decode_kv_last_page_len
                .as_gpu_tensor()
                .raw_ptr() as *mut core::ffi::c_void,
            shape.attn_scale,
            shape.rms_norm_eps,
            num_pages_arg,
            device_paged.num_prefill_tokens,
            dev_idx,
            stream as *mut core::ffi::c_void,
        )
    };

    if rc != 0 {
        anyhow::bail!("tk_megakernel launch returned cudaError {rc}");
    }

    // The launcher is async on `compute_stream`. The caching
    // allocator's `free` is NOT stream-aware: dropping an
    // `OwnedTensor` immediately returns its block to the free
    // pool, and the next allocation can reuse the same memory
    // — even before the kernel finishes reading from it. Sync
    // the stream so every scratch tensor is genuinely dead
    // before its `Drop` runs at end of scope. The sync also
    // matches host-side interpreter semantics (forward returns
    // a usable logits tile).
    unsafe { driver::stream_synchronize(stream)? };

    // Explicit drops on the throwaway scratch — semantic
    // no-ops but make the lifetime intent obvious. Every
    // scratch tensor outlives the kernel because we synced
    // above. `hidden_states` and `logits` are returned to the
    // caller (placed into `tiles` by the wrapper).
    drop(tape_owned);
    drop(timings);
    drop(rms_rope_intermediates);
    drop(rms_gate_intermediates);
    drop(q_post_rope);
    drop(attn_out);
    drop(silu_out);
    drop(rms_lm_head_intermediates);
    drop(device_paged);
    // `bar` and `global_instr_idx` are plain `GpuTensor`
    // (Copy / leaked from `alloc_gpu_tensor_zeroed`); no Drop
    // hazard.
    Ok((hidden_states, logits))
}
