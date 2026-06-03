// SPDX-License-Identifier: Apache-2.0
//! CUDA dispatch glue for orchestrator-emitted PD-wavefront megakernels.
//!
//! `tk_codegen::emit_kernel` produces, per resolved decode canonical, a
//! `tk_decode_full_<stem>` kernel + `extern "C"` host wrapper compiled
//! into `libmegakernels.a`. The per-arch macro emits a small
//! `wavefront_megakernel_dispatch_cuda` fn that delegates here, passing
//! the per-canonical static recipe + FFI launcher fn pointer.
//!
//! Responsibilities at dispatch time:
//!   1. Run the embed step (host gather → `[num_tokens, hidden]` bf16).
//!   2. Allocate the op-output staging arena (one `OwnedTensor` per
//!      orchestrator op output).
//!   3. Resolve every source pointer the kernel reads — embed result,
//!      cos/sin, prefix KV cache rows, weight tensors via
//!      `WeightAccessors`, with byte offsets for fused-base sources.
//!   4. Call the FFI launcher with `bufs[]` (sources first, then op
//!      outputs) + `u32_args[]` + the compute stream.
//!   5. Return the result-op output as an `OwnedTensor` reshaped to
//!      `[num_tokens, vocab_size]` bf16 — the per-op `forward` path's
//!      logits shape.
//!
//! No unsafe outside what the FFI ABI requires; the static recipe is
//! the macro's compile-time witness that every source has a real
//! resolution path. A `None` return signals a hard launch failure
//! (cudaError != 0); the worker hook then falls back to the per-op
//! `forward` path.

#![cfg(feature = "cuda")]

use std::ffi::c_void;

use ferrite_cuda_core::alloc::OwnedTensor;
use ferrite_cuda_core::device::GpuDevice;
use ferrite_cuda_core::dtype::DType;

use crate::ForwardCtx;
use crate::instr::{CanonicalParams, WeightAccessors};

/// FFI signature emitted by `tk_codegen::emit_kernel`'s host wrapper.
pub type LaunchFn = unsafe extern "C" fn(
    bufs: *const *mut c_void,
    u32_args: *const u32,
    stream: *mut c_void,
) -> i32;

// ── Typed kernel u32-arg witnesses (Gap 18) ────────────────────────
//
// The kernel's `__num_kv_pages` runtime arg is the count of KV cache
// blocks for THIS sequence — not the cache pool size. Using the pool
// size (138868 for Llama-1B) caused the post-Gap-17 crash (Step E.9):
// the kernel iterated 138868 times reading 142 MB of mostly-
// uninitialized cache, eventually triggering CUDA_ERROR_LAUNCH_FAILED.
//
// `SeqBlockCount` newtype-wraps the u32 with a sealed constructor
// `from_max_seqlen_k(seqlen, block_size)`. The pool size has no path
// to construct one. `KernelU32ArgsBuilder::push_num_kv_pages`
// requires a `SeqBlockCount`, so passing
// `ctx.kv_cache.num_blocks as u32` directly is a Rust compile error.
//
// Verification: revert dispatch_cuda to `u32_builder.push_num_kv_pages(
// ctx.kv_cache.num_blocks as u32)` — the call fails to typecheck.

/// Per-sequence count of KV cache blocks. Constructed only via
/// `from_max_seqlen_k`, which derives the value from
/// `(max_seqlen_k, block_size)` — never from the cache pool size.
///
/// # Compile-fail proof (Gap 18)
///
/// Passing a raw `u32` (e.g., `kv_cache.num_blocks`) to
/// `push_num_kv_pages` is a Rust compile error.
/// ```compile_fail
/// use ferrite_forward::wavefront_cuda::*;
/// let mut b = KernelU32ArgsBuilder::new();
/// // The pool size is just a u32 — but push_num_kv_pages requires
/// // SeqBlockCount. Compile error.
/// let pool_size: u32 = 138868;
/// b.push_num_kv_pages(pool_size);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SeqBlockCount(u32);

impl SeqBlockCount {
    /// Construct from a sequence's `max_seqlen_k` (current K_LEN =
    /// prompt + decode position) and the cache `block_size`.
    pub fn from_max_seqlen_k(max_seqlen_k: u32, block_size: u32) -> Self {
        Self(max_seqlen_k.div_ceil(block_size))
    }

    /// Raw u32 for FFI. Crate-private — only the typed
    /// `KernelU32ArgsBuilder` should access this.
    pub(crate) fn raw(self) -> u32 {
        self.0
    }
}

/// Decode position for the current decode token. D2H-copied from
/// `ctx.positions[0]` once per dispatch. Newtype to keep it from
/// being conflated with `SeqBlockCount` or any other u32 arg.
#[derive(Clone, Copy, Debug)]
pub struct DecodePosition(u32);

impl DecodePosition {
    pub fn from_raw(pos: u32) -> Self {
        Self(pos)
    }

    pub(crate) fn raw(self) -> u32 {
        self.0
    }
}

/// Typed builder for kernel u32 args. Args are pushed in a fixed
/// order matching `KernelArgs::u32_args` (set by
/// `fixtures::orchestrator_kernel_args`): num_kv_pages first
/// (gated by has_attn_decode), then decode_position (gated by
/// has_rope). Wrong-typed pushes are Rust compile errors.
pub struct KernelU32ArgsBuilder {
    args: Vec<u32>,
}

impl KernelU32ArgsBuilder {
    pub fn new() -> Self {
        Self { args: Vec::new() }
    }

    /// Push the per-sequence KV block count. **Compile-time
    /// guarantee**: caller must produce a [`SeqBlockCount`]; the
    /// pool size (raw `u32` from `kv_cache.num_blocks`) cannot be
    /// passed without going through `SeqBlockCount::from_max_seqlen_k`,
    /// which only derives the value from
    /// `(max_seqlen_k, block_size)`.
    pub fn push_num_kv_pages(&mut self, count: SeqBlockCount) {
        self.args.push(count.raw());
    }

    /// Push the decode position.
    pub fn push_decode_position(&mut self, pos: DecodePosition) {
        self.args.push(pos.raw());
    }

    pub fn finalize(self) -> Vec<u32> {
        self.args
    }
}

impl Default for KernelU32ArgsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// One source's resolution recipe — typed at macro time, walked at
/// dispatch time. Each entry encodes the (bucket, op_idx, slot, layer)
/// tuple `WeightAccessors` consumes plus, where applicable, a
/// fused-base byte offset into the underlying tensor.
#[derive(Debug, Clone, Copy)]
pub enum SourceRecipeEntry {
    /// The post-embed hidden state — filled at dispatch from the
    /// embed step's `OwnedTensor`.
    EmbeddedHidden,
    /// Rotary cos slice — `wm.cos_sin_at(bucket, op_idx, slot, layer)`
    /// + 0-byte offset (cos lives at the front of the rotary tensor).
    Cos {
        bucket: u32,
        op_idx: u32,
        slot: u32,
        layer: u32,
    },
    /// Rotary sin slice — same accessor as `Cos` plus a per-canonical
    /// byte offset baked from `head_dim * elem_bytes`.
    Sin {
        bucket: u32,
        op_idx: u32,
        slot: u32,
        layer: u32,
        byte_offset: u64,
    },
    /// Prefix K cache for `layer` (paged kv-cache backing tensor for
    /// the K side). Resolved via `ctx.kv_cache.k_cache(layer)`.
    PrefixK { layer: u32 },
    /// Prefix V cache for `layer`. Resolved via
    /// `ctx.kv_cache.v_cache(layer)`.
    PrefixV { layer: u32 },
    /// RmsNorm gain weight — `wm.rms_norm_at(bucket, op_idx, slot,
    /// layer).weight`.
    RmsNormWeight {
        bucket: u32,
        op_idx: u32,
        slot: u32,
        layer: u32,
    },
    /// Dense linear weight — `wm.linear_at(...).dense_weight()` plus
    /// `byte_offset` (non-zero for fused-base sources where the
    /// orchestrator-side accessor merges multiple constituents into a
    /// single `__fused__`-keyed buffer).
    LinearDenseWeight {
        bucket: u32,
        op_idx: u32,
        slot: u32,
        layer: u32,
        byte_offset: u64,
    },
    /// Embedding weight — `wm.embedding_at(...).weight`. Used by
    /// arches that read `embed_tokens` as a forward source (e.g. tied
    /// lm_head paths on some canonicals).
    EmbeddingWeight {
        bucket: u32,
        op_idx: u32,
        slot: u32,
        layer: u32,
    },
}

/// Per-canonical baked dispatch artifacts. The macro emits this once
/// per resolved decode canonical and hands it to [`dispatch_cuda`].
pub struct DispatchSpec {
    /// `wm.embedding_at(embed_bucket, embed_op_idx, 0, 0)` is the
    /// `embed_tokens` weight at run time. Always bucket 0 / op 0 for
    /// the existing decode lowering, but kept explicit so future
    /// canonicals with a different first-op layout don't silently
    /// regress.
    pub embed_bucket: u32,
    pub embed_op_idx: u32,
    /// Parallel to the orchestrator's source list (the kernel's first
    /// `n_sources` `bufs[]` slots).
    pub source_recipe: &'static [SourceRecipeEntry],
    /// Per-op-output byte size, in BufId order — the orchestrator's
    /// `bufs[n_sources..]` set. Indices into this slice match the
    /// orchestrator's op indices.
    pub op_output_bytes: &'static [u64],
    /// True if any op in the lowered decode is `AttnDecode`; gates
    /// the `__num_kv_pages` u32 arg the kernel signature expects.
    pub has_attn_decode: bool,
    /// True if any op rotates (`RopeRotate` / `RopeAppend`); gates the
    /// `__decode_position` u32 arg. The dispatcher D2H copies
    /// `ctx.positions[0]` and pushes it after `__num_kv_pages` so the
    /// rope TMA loads can index `cos_sin_cache + pos * row_bytes`.
    pub has_rope: bool,
    /// Op index whose output buffer is the final logits row. Within
    /// `op_output_bytes`. Reshaped on return to
    /// `[num_tokens, vocab_size]` bf16.
    pub result_op_idx: u32,
    /// Logits column count.
    pub vocab_size: u64,
    /// Host-side wrapper for `tk_decode_full_<stem>`.
    pub launch_fn: LaunchFn,
}

/// Run the orchestrator-emitted megakernel for one decode step.
///
/// `Some(logits)` on success, where `logits` is an `OwnedTensor` of
/// shape `[num_tokens, vocab_size]` bf16 — same shape as the per-op
/// `forward` path's return. `None` if the launcher returned a
/// non-zero `cudaError` (the worker hook then falls back to per-op
/// forward).
///
/// # Safety
/// All tensors in `ctx` must point at valid GPU memory; `device`'s
/// caching allocator and compute stream must be live; `wm` must have
/// loaded the weights the recipe references.
#[allow(clippy::too_many_lines)]
pub unsafe fn dispatch_cuda<W: CanonicalParams + WeightAccessors>(
    wm: &W,
    ctx: &ForwardCtx<'_>,
    device: &mut GpuDevice,
    num_tokens: u64,
    spec: &DispatchSpec,
) -> Option<OwnedTensor> {
    use ferrite_kernels::kernels;

    // ── 1. Run embed: host gather → [num_tokens, hidden] bf16.
    let embed_layer = wm.embedding_at(spec.embed_bucket, spec.embed_op_idx, 0, 0u32);
    let embed_weight = embed_layer.weight;
    let vocab_per_rank = embed_weight.dim(0) as u32;
    let post_embed: OwnedTensor = unsafe {
        kernels::embedding_gather_masked(
            embed_weight,
            *ctx.input_ids,
            0,
            vocab_per_rank,
            &mut device.caching,
            device.compute_stream,
        )
    };

    // ── 2. Allocate op-output staging buffers as flat U8 tensors at
    //     128-byte alignment (TMA load minimum). The kernel writes one
    //     tile per op, sized by the lowered op's output shape; the
    //     declared bytes are the lower bound, the alloc is the
    //     padded actual.
    let n_ops = spec.op_output_bytes.len();
    let mut op_outputs: Vec<OwnedTensor> = Vec::with_capacity(n_ops);
    for &bytes in spec.op_output_bytes {
        let aligned = (bytes as usize).next_multiple_of(128).max(128);
        op_outputs.push(device.caching.alloc_tensor(&[aligned], DType::U8));
    }

    // ── 3. Resolve source ptrs walking the recipe.
    let n_sources = spec.source_recipe.len();
    let mut bufs: Vec<*mut c_void> = Vec::with_capacity(n_sources + n_ops);

    let post_embed_gpu = post_embed.as_gpu_tensor();
    let post_embed_ptr = post_embed_gpu.as_mut_ptr::<u8>() as *mut c_void;

    for entry in spec.source_recipe {
        let p: *mut c_void = match *entry {
            SourceRecipeEntry::EmbeddedHidden => post_embed_ptr,
            SourceRecipeEntry::Cos {
                bucket,
                op_idx,
                slot,
                layer,
            } => {
                let cos_sin = wm.cos_sin_at(bucket, op_idx, slot, layer);
                cos_sin.as_mut_ptr::<u8>() as *mut c_void
            }
            SourceRecipeEntry::Sin {
                bucket,
                op_idx,
                slot,
                layer,
                byte_offset,
            } => {
                let cos_sin = wm.cos_sin_at(bucket, op_idx, slot, layer);
                let base = cos_sin.as_mut_ptr::<u8>();
                unsafe { base.add(byte_offset as usize) as *mut c_void }
            }
            SourceRecipeEntry::PrefixK { layer } => {
                let kv = ctx.kv_cache.k_cache(layer as usize);
                kv.as_raw().as_mut_ptr::<u8>() as *mut c_void
            }
            SourceRecipeEntry::PrefixV { layer } => {
                let kv = ctx.kv_cache.v_cache(layer as usize);
                kv.as_raw().as_mut_ptr::<u8>() as *mut c_void
            }
            SourceRecipeEntry::RmsNormWeight {
                bucket,
                op_idx,
                slot,
                layer,
            } => {
                let rms = wm.rms_norm_at(bucket, op_idx, slot, layer);
                rms.weight.as_mut_ptr::<u8>() as *mut c_void
            }
            SourceRecipeEntry::LinearDenseWeight {
                bucket,
                op_idx,
                slot,
                layer,
                byte_offset,
            } => {
                let lin = wm.linear_at(bucket, op_idx, slot, layer);
                if !matches!(lin, ferrite_kernels::layers::LinearLayer::Dense(_)) {
                    // Quantized variants aren't covered by the
                    // first-cut emit; the macro must emit a matching
                    // recipe variant before that arch can dispatch.
                    eprintln!(
                        "[wavefront-cuda] LinearDenseWeight at \
                         ({bucket},{op_idx},{slot},{layer}): non-Dense \
                         variant — falling back",
                    );
                    return None;
                }
                let dense_w = lin.dense_weight();
                let base = dense_w.as_mut_ptr::<u8>();
                unsafe { base.add(byte_offset as usize) as *mut c_void }
            }
            SourceRecipeEntry::EmbeddingWeight {
                bucket,
                op_idx,
                slot,
                layer,
            } => {
                let e = wm.embedding_at(bucket, op_idx, slot, layer);
                e.weight.as_mut_ptr::<u8>() as *mut c_void
            }
        };
        bufs.push(p);
    }
    for buf in &op_outputs {
        bufs.push(buf.as_gpu_tensor().as_mut_ptr::<u8>() as *mut c_void);
    }

    // ── 4. u32 args — order matches the kernel signature:
    //   `__num_kv_pages` (gated by `has_attn_decode`)
    //   `__decode_position` (gated by `has_rope`)
    //
    // `__num_kv_pages` is the cache pool's static `num_blocks` (the
    // paged-attention block count). `__decode_position` is the
    // current decode token's position in its sequence — D2H copied
    // from `ctx.positions[0]` (`[1]` u32 for decode `num_tokens=1`).
    // The kernel's rope arms add `pos * row_bytes` to the cos/sin
    // TMA source pointer so each token reads its own row of
    // `cos_sin_cache`. `event_synchronize` on the dedicated d2h
    // event blocks the host ~5 us while the value lands; the
    // compute stream is gated downstream.
    // Typed u32-args builder. `KernelU32Args::push_num_kv_pages` takes
    // a [`SeqBlockCount`] — passing a pool-size count (Gap 18, the
    // crash bug E.9 fixed) is a Rust compile error. The newtype's
    // private constructor only accepts (max_seqlen_k, block_size).
    let mut u32_builder = KernelU32ArgsBuilder::new();
    if spec.has_attn_decode {
        let count = SeqBlockCount::from_max_seqlen_k(
            ctx.max_seqlen_k as u32,
            ctx.kv_cache.block_size as u32,
        );
        u32_builder.push_num_kv_pages(count);
    }
    if spec.has_rope {
        let mut pos_host: u32 = 0;
        unsafe {
            device
                .async_d2h(
                    (&raw mut pos_host).cast::<u8>(),
                    ctx.positions.raw_ptr().cast::<u8>(),
                    std::mem::size_of::<u32>(),
                )
                .expect("d2h decode position");
        }
        device.sync_d2h().expect("sync d2h decode position");
        u32_builder.push_decode_position(DecodePosition::from_raw(pos_host));
    }

    // Finalize u32 args from the typed builder. Order is fixed:
    // num_kv_pages (if present), then decode_position (if present).
    let u32_args: Vec<u32> = u32_builder.finalize();

    // ── 5. Call the FFI wrapper. Pointer table + u32 table are kept
    //     alive across the call by virtue of the `Vec`s outliving
    //     `launch_fn`; the launch wrapper sets
    //     `cudaFuncAttributeMaxDynamicSharedMemorySize` and launches
    //     `<<<1, total_threads, dyn_smem, stream>>>` synchronously
    //     w.r.t. the host (the kernel runs on the stream
    //     asynchronously).
    let stream_raw = device.compute_stream as *mut c_void;
    let err = unsafe { (spec.launch_fn)(bufs.as_ptr(), u32_args.as_ptr(), stream_raw) };
    if err != 0 {
        eprintln!("[wavefront-cuda] launch_tk_decode_full returned cudaError {err}");
        return None;
    }

    // ── 6. Pluck the result op output and reshape it in place from
    //     the U8 byte layout the kernel writes (the alloc is sized to
    //     128-byte alignment of the declared op-output bytes; the
    //     bf16 logits occupy the prefix `num_tokens * vocab_size * 2`
    //     bytes). `OwnedTensor::reshape` rewrites only metadata — the
    //     underlying alloc + caching-allocator bookkeeping is
    //     untouched.
    let mut result = op_outputs.swap_remove(spec.result_op_idx as usize);
    unsafe {
        result.reshape(
            &[num_tokens as usize, spec.vocab_size as usize],
            DType::BF16,
        );
    }
    drop(op_outputs);
    drop(post_embed);
    Some(result)
}
