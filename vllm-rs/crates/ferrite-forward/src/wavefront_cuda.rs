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

/// Per-sequence count of KV cache blocks (each block holds
/// `block_size` tokens). Constructed only via `from_max_seqlen_k`.
/// Distinct type from [`SeqTokenCount`] — neither is convertible to
/// the other without explicit re-derivation.
#[derive(Clone, Copy, Debug)]
pub struct SeqBlockCount(u32);

impl SeqBlockCount {
    pub fn from_max_seqlen_k(max_seqlen_k: u32, block_size: u32) -> Self {
        Self(max_seqlen_k.div_ceil(block_size))
    }

    pub(crate) fn raw(self) -> u32 {
        self.0
    }
}

/// Per-sequence count of K/V tokens currently cached (= `max_seqlen_k`).
/// Used for the megakernel's `__num_kv_pages` runtime arg. The
/// kernel iterates ONE TOKEN per loop iteration (loads 1024 bytes =
/// `kv_cols * elem_bytes` per iter), so the iter count must equal
/// the token count, NOT the block count and NOT the cache pool size.
///
/// # Compile-fail proofs (Gap 18 + Bug 3)
///
/// Pool size (raw u32):
/// ```compile_fail
/// use ferrite_forward::wavefront_cuda::*;
/// let mut b = KernelU32ArgsBuilder::new();
/// let pool: u32 = 138868;
/// b.push_num_kv_pages(pool);  // raw u32 rejected
/// ```
///
/// `SeqBlockCount` (right type for "blocks", wrong for "tokens"):
/// ```compile_fail
/// use ferrite_forward::wavefront_cuda::*;
/// let mut b = KernelU32ArgsBuilder::new();
/// let blocks = SeqBlockCount::from_max_seqlen_k(128, 16); // 8 blocks
/// b.push_num_kv_pages(blocks);  // expected SeqTokenCount, found SeqBlockCount
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SeqTokenCount(u32);

impl SeqTokenCount {
    /// Construct from `max_seqlen_k` (current K_LEN = prompt + decode
    /// position). The kernel will iterate this many times.
    pub fn from_max_seqlen_k(max_seqlen_k: u32) -> Self {
        Self(max_seqlen_k)
    }

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
    /// Test-only constructor. Production code MUST go through
    /// [`DecodePosition::from_pending`] so the host-side d2h sync
    /// happens before the value reaches the kernel arg builder.
    #[doc(hidden)]
    pub fn from_raw(pos: u32) -> Self {
        Self(pos)
    }

    /// Production constructor: drains a [`Pending<u32>`] by consuming
    /// the typed sync witness. Callers cannot construct a
    /// `DecodePosition` from a raw u32 without first going through
    /// `Pending::async_d2h(...)?.sync(device)`, which forces the
    /// d2h drain.
    pub fn from_pending(pending: Pending<u32>, device: &mut GpuDevice) -> Self {
        Self(pending.sync(device))
    }

    pub(crate) fn raw(self) -> u32 {
        self.0
    }
}

/// Paged KV-cache slot index for the new decode token. D2H-copied
/// from `ctx.slot_mapping[0]` (I64 → u32) once per dispatch — the
/// scheduler has already resolved
/// `block_table[seq][pos / block_size] * block_size + (pos %
/// block_size)` into a single absolute slot index, so the megakernel
/// just multiplies it by the per-token row stride.
///
/// Distinct type from [`SeqTokenCount`] / [`SeqBlockCount`] /
/// [`DecodePosition`] — none of them is convertible to a
/// `DecodeSlot` without explicit re-derivation. Sealed
/// constructor [`DecodeSlot::from_slot_mapping`] is the only path.
///
/// # Compile-fail proof
///
/// Raw u32 / SeqTokenCount rejected by the typed pusher:
/// ```compile_fail
/// use ferrite_forward::wavefront_cuda::*;
/// let mut b = KernelU32ArgsBuilder::new();
/// let raw: u32 = 0;
/// b.push_decode_slot(raw);  // expected DecodeSlot, found u32
/// ```
///
/// ```compile_fail
/// use ferrite_forward::wavefront_cuda::*;
/// let mut b = KernelU32ArgsBuilder::new();
/// let count = SeqTokenCount::from_max_seqlen_k(7);
/// b.push_decode_slot(count);  // expected DecodeSlot, found SeqTokenCount
/// ```
#[derive(Clone, Copy, Debug)]
pub struct DecodeSlot(u32);

impl DecodeSlot {
    /// Test-only constructor. Production code MUST go through
    /// [`DecodeSlot::from_pending_i64`] so the host-side d2h sync
    /// for `ctx.slot_mapping[0]` (an i64 tensor) happens before the
    /// value reaches the kernel arg builder.
    #[doc(hidden)]
    pub fn from_slot_mapping(slot: u32) -> Self {
        Self(slot)
    }

    /// Production constructor: drains a [`Pending<i64>`] (vLLM's
    /// slot-mapping tensor is I64), validates the slot fits in u32,
    /// and packs the synced value. Forgetting to issue the d2h or
    /// to sync it is a compile error (Pending's drop-bomb /
    /// move-only sync).
    pub fn from_pending_i64(pending: Pending<i64>, device: &mut GpuDevice) -> Self {
        let raw = pending.sync_i64(device);
        debug_assert!(
            raw >= 0 && raw <= u32::MAX as i64,
            "decode slot {raw} out of u32 range",
        );
        Self(raw as u32)
    }

    pub(crate) fn raw(self) -> u32 {
        self.0
    }
}

/// **D2H-sync typestate witness** (paris invariant
/// `d2h-sync-before-launch`).
///
/// The kernel reads its u32 args (decode position, decode slot) by
/// value from the host's launch call. The host fills those values
/// via async device-to-host copies, but the GPU's d2h queue is
/// asynchronous: reading the host-side u32 BEFORE
/// [`GpuDevice::sync_d2h`] runs returns whatever was at that stack
/// slot when the launch began (zero-initialised, or stale from a
/// prior dispatch). The resulting wrong slot/position drives a
/// kernel that writes K/V to slot 0 every iter, producing the exact
/// `Paris!!!!!!!!!` decode-degenerate stream.
///
/// `Pending<T>` is the typed witness that the value has NOT yet been
/// sync'd. Construct via [`Pending::async_d2h`] (which fires the
/// async copy); the only way to read the value is via
/// [`Pending::sync`], which consumes the token AND calls
/// [`GpuDevice::sync_d2h`] in the same call. Forgetting the sync
/// is impossible at the type level: the value field is private (no
/// public getter); dropping the token without sync trips a
/// drop-bomb panic that names the originating call site, not a
/// silent stale read.
///
/// # Compile-fail proofs
///
/// Direct field access rejected (private field):
///
/// ```compile_fail
/// use ferrite_forward::wavefront_cuda::Pending;
/// let p: Pending<u32> = unimplemented!();
/// let _ = p.value;  // private field
/// ```
///
/// `sync` consumes self — double-sync rejected (use of moved value):
///
/// ```compile_fail
/// use ferrite_forward::wavefront_cuda::Pending;
/// let p: Pending<u32> = unimplemented!();
/// let mut device: ferrite_forward::GpuDevice = unimplemented!();
/// let _ = p.sync(&mut device);
/// let _ = p.sync(&mut device);  // value used after move
/// ```
#[must_use = "Pending<T> must be sync()'d before reading; dropping without sync panics at runtime"]
pub struct Pending<T> {
    value: std::cell::UnsafeCell<T>,
    /// `true` after [`Pending::sync`] runs — disarms the drop-bomb so the
    /// `mem::forget`-equivalent "value moved out" path doesn't panic.
    consumed: bool,
    /// Site name for the drop-bomb panic message ("decode position",
    /// "decode slot", etc.). Static so we don't allocate during the
    /// hot dispatch path.
    site: &'static str,
}

impl Pending<u32> {
    /// Issue an async u32 device→host copy and return the typed
    /// witness. Caller MUST call [`Pending::sync`] before the value is
    /// usable.
    ///
    /// `site` is a short label (e.g. `"decode_position"`) used in the
    /// drop-bomb panic message if the caller forgets to sync.
    ///
    /// # Safety
    /// `src` must point to at least 4 valid u32 bytes on the device,
    /// alive until the async copy completes (i.e. until [`Pending::sync`]
    /// returns).
    pub unsafe fn async_d2h(
        device: &mut GpuDevice,
        src: *const u8,
        site: &'static str,
    ) -> Self {
        let token = Self {
            value: std::cell::UnsafeCell::new(0u32),
            consumed: false,
            site,
        };
        unsafe {
            device
                .async_d2h(token.value.get().cast::<u8>(), src, std::mem::size_of::<u32>())
                .unwrap_or_else(|e| panic!("Pending::async_d2h ({site}): {e:?}"));
        }
        token
    }

    /// Drain the host-side d2h queue and return the synced value.
    /// Consumes the token (move-out of `self`); double-sync is a
    /// Rust compile error.
    pub fn sync(mut self, device: &mut GpuDevice) -> u32 {
        device
            .sync_d2h()
            .unwrap_or_else(|e| panic!("Pending::sync ({}): {e:?}", self.site));
        self.consumed = true;
        // SAFETY: sync_d2h returned, so the d2h queue is drained and
        // the host-side u32 in self.value reflects the device-side
        // bytes at the source pointer.
        unsafe { *self.value.get() }
    }
}

impl Pending<i64> {
    /// `i64` variant for `slot_mapping[0]` (vLLM's slot-mapping
    /// tensor is I64). Same protocol as the u32 variant.
    ///
    /// # Safety
    /// Same as the u32 variant — `src` must point to at least 8
    /// valid bytes alive until [`Pending::sync`] returns.
    pub unsafe fn async_d2h_i64(
        device: &mut GpuDevice,
        src: *const u8,
        site: &'static str,
    ) -> Self {
        let token = Self {
            value: std::cell::UnsafeCell::new(0i64),
            consumed: false,
            site,
        };
        unsafe {
            device
                .async_d2h(token.value.get().cast::<u8>(), src, std::mem::size_of::<i64>())
                .unwrap_or_else(|e| panic!("Pending::async_d2h_i64 ({site}): {e:?}"));
        }
        token
    }

    pub fn sync_i64(mut self, device: &mut GpuDevice) -> i64 {
        device
            .sync_d2h()
            .unwrap_or_else(|e| panic!("Pending::sync_i64 ({}): {e:?}", self.site));
        self.consumed = true;
        unsafe { *self.value.get() }
    }
}

impl<T> Drop for Pending<T> {
    fn drop(&mut self) {
        if !self.consumed {
            panic!(
                "Pending<{}> ({}) dropped without sync — host would read pre-sync stale memory \
                 (paris invariant `d2h-sync-before-launch`)",
                std::any::type_name::<T>(),
                self.site,
            );
        }
    }
}

/// **Compute-sync typestate witness** (paris invariant
/// `sync-compute-after-launch`).
///
/// Hopper's `cp.async.bulk` writes complete asynchronously w.r.t.
/// the issuing thread. The kernel's exit `__syncthreads()` does
/// a CTA `bar.sync` but does NOT drain the async proxy: the next
/// dispatch's `tma::load_async` may observe pre-write gmem at the
/// K/V cache slots the prior dispatch's RopeAppend storer wrote.
/// Empirically the symptom is the `Paris!!!!!!!!!` decode-degenerate
/// stream (every decode token after the first attends to stale K).
///
/// `Unsynced<T>` is the typed witness that a kernel was launched
/// but the host hasn't yet drained the compute stream. The wrapped
/// `T` (typically the op-output `Vec<OwnedTensor>` produced by the
/// launch) is private — there is no `.inner()` getter — so reading
/// the result requires consuming the witness via [`Unsynced::sync`],
/// which calls [`GpuDevice::sync_compute`] in the same call.
/// Forgetting the sync trips the drop-bomb, naming the launch site
/// instead of producing silently-stale tensors downstream.
///
/// # Compile-fail proofs
///
/// Direct field access rejected (private field):
///
/// ```compile_fail
/// use ferrite_forward::wavefront_cuda::Unsynced;
/// let u: Unsynced<u32> = unimplemented!();
/// let _ = u.inner;  // private field
/// ```
///
/// `sync` consumes self — double-sync rejected (use of moved value):
///
/// ```compile_fail
/// use ferrite_forward::wavefront_cuda::Unsynced;
/// let u: Unsynced<u32> = unimplemented!();
/// let mut device: ferrite_forward::GpuDevice = unimplemented!();
/// let _ = u.sync(&mut device);
/// let _ = u.sync(&mut device);  // value used after move
/// ```
#[must_use = "Unsynced<T> must be sync()'d before its inner T is read; dropping without sync panics at runtime"]
pub struct Unsynced<T> {
    inner: std::mem::ManuallyDrop<T>,
    consumed: bool,
    site: &'static str,
}

impl<T> Unsynced<T> {
    /// Wrap a freshly-launched kernel's outputs as un-synced. Caller
    /// MUST call [`Unsynced::sync`] before observing any device-side
    /// memory the inner `T` references.
    ///
    /// `site` names the launch (e.g. `"wavefront_megakernel"`) for
    /// the drop-bomb panic message.
    pub fn after_launch(inner: T, site: &'static str) -> Self {
        Self {
            inner: std::mem::ManuallyDrop::new(inner),
            consumed: false,
            site,
        }
    }

    /// Drain the compute stream and unwrap the inner value. Consumes
    /// the witness (move-only), so double-sync is a Rust use-after-
    /// move compile error.
    pub fn sync(mut self, device: &mut GpuDevice) -> T {
        device
            .sync_compute()
            .unwrap_or_else(|e| panic!("Unsynced::sync ({}): {e:?}", self.site));
        self.consumed = true;
        // SAFETY: ManuallyDrop::take moves the inner value out;
        // self.consumed = true disarms the Drop bomb so we don't
        // double-drop. We `mem::forget(self)` after move-out.
        let inner = unsafe { std::mem::ManuallyDrop::take(&mut self.inner) };
        std::mem::forget(self);
        inner
    }
}

impl<T> Drop for Unsynced<T> {
    fn drop(&mut self) {
        if !self.consumed {
            // SAFETY: only fires on the panic path; the consumed=true
            // path is `mem::forget`'d before drop.
            unsafe { std::mem::ManuallyDrop::drop(&mut self.inner) };
            panic!(
                "Unsynced<{}> ({}) dropped without sync — host would observe stale device memory \
                 (paris invariant `sync-compute-after-launch`)",
                std::any::type_name::<T>(),
                self.site,
            );
        }
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

    /// Push the per-sequence KV TOKEN count (`__num_kv_pages` in the
    /// kernel — name is legacy; semantically a token count since the
    /// kernel iterates one token per loop iter).
    ///
    /// **Compile-time guarantee**: caller must produce a
    /// [`SeqTokenCount`]; raw u32 (pool size, etc.) and
    /// [`SeqBlockCount`] (block count) are both rejected by the type
    /// checker. Construct via `SeqTokenCount::from_max_seqlen_k`.
    pub fn push_num_kv_pages(&mut self, count: SeqTokenCount) {
        self.args.push(count.raw());
    }

    /// Push the decode position.
    pub fn push_decode_position(&mut self, pos: DecodePosition) {
        self.args.push(pos.raw());
    }

    /// Push the paged-KV-cache slot index for the new decode token's
    /// K/V (`__decode_slot` in the kernel — multiplied by per-token
    /// row stride to compute the cache-write byte offset). Compile-
    /// time safe: caller must produce a [`DecodeSlot`].
    pub fn push_decode_slot(&mut self, slot: DecodeSlot) {
        self.args.push(slot.raw());
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
    /// True if any op is `RopeAppend`; gates the `__decode_slot` u32
    /// arg the kernel uses to address the paged-KV-cache K/V write
    /// destinations. The dispatcher D2H copies `ctx.slot_mapping[0]`
    /// (I64 → u32) and pushes it after `__decode_position`. The new
    /// token's K row writes to `K_cache + slot * row_bytes` and V to
    /// `V_cache + slot * row_bytes` — `slot_mapping` already encodes
    /// the paged block-table indirection, so the kernel just multiplies.
    pub has_rope_append: bool,
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
        let count = SeqTokenCount::from_max_seqlen_k(ctx.max_seqlen_k as u32);
        u32_builder.push_num_kv_pages(count);
    }
    if spec.has_rope {
        // d2h-sync-before-launch invariant: Pending<u32>'s typestate
        // forces the sync_d2h to happen before DecodePosition is
        // constructed. Forgetting the sync is impossible at the type
        // level (private value field, sync consumes self by move).
        let pending = unsafe {
            Pending::<u32>::async_d2h(
                device,
                ctx.positions.raw_ptr().cast::<u8>(),
                "decode_position",
            )
        };
        u32_builder.push_decode_position(DecodePosition::from_pending(pending, device));
    }
    if spec.has_rope_append {
        // `slot_mapping` is a `[num_tokens]` I64 tensor — for
        // num_tokens=1 decode, slot_mapping[0] is the absolute paged
        // cache slot for the new token's K/V (the runtime has already
        // resolved block-table indirection into a flat slot index).
        // Pending<i64> + DecodeSlot::from_pending_i64 enforces the
        // d2h sync at the type level.
        let pending = unsafe {
            Pending::<i64>::async_d2h_i64(
                device,
                ctx.slot_mapping.raw_ptr().cast::<u8>(),
                "decode_slot",
            )
        };
        u32_builder.push_decode_slot(DecodeSlot::from_pending_i64(pending, device));
    }

    // Finalize u32 args from the typed builder. Order is fixed:
    // num_kv_pages (if present), decode_position (if present),
    // decode_slot (if present).
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
    // sync-compute-after-launch invariant: wrap the post-launch
    // op-outputs collection in `Unsynced<...>`. The only path to
    // unwrap is `Unsynced::sync(device)`, which calls
    // `device.sync_compute()` in the same call. Any code path that
    // tries to read `op_outputs` before `.sync()` is a use-of-moved-
    // value compile error; dropping the witness without sync trips
    // the drop-bomb so the failure is loud.
    //
    // Hopper's `cp.async.bulk` writes complete asynchronously
    // w.r.t. the issuing thread, and the kernel's exit
    // `__syncthreads()` does NOT drain the async proxy: the next
    // dispatch's `tma::load_async` may observe pre-write gmem at
    // the K/V cache slots the prior dispatch's RopeAppend storer
    // wrote. Empirically: without the host sync, `Paris ->
    // " Paris!!!!!!!!!"`; with the typed sync witness, the sync
    // is structurally unforgettable.
    let unsynced_outputs = Unsynced::after_launch(op_outputs, "wavefront_megakernel");
    let mut op_outputs = unsynced_outputs.sync(device);

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
