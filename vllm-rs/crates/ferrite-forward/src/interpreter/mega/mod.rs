// SPDX-License-Identifier: Apache-2.0
//! Rust ABI for the device-interpreter megakernel variants emitted
//! by `ferrite-forward-macro::interpreter::mega::emit_cu_variant`.
//!
//! See `FERRITE_TK_PLAN.md` at the repo root.
//!
//! # Phase 3d pool ABI
//!
//! Each emitted `.cu` exposes:
//!
//! ```cpp
//! extern "C" cudaError_t ferrite_<variant>_launch(
//!     __nv_bfloat16* const*        act_ptrs,     // [NUM_ACT_SLOTS]
//!     const __nv_bfloat16* const*  weight_ptrs,  // [NUM_WEIGHT_ACCESSORS * NUM_LAYERS]
//!     cudaStream_t                 stream);
//! ```
//!
//! Every variant-fixed scalar (model dims, workload shape, eps) is
//! a `static constexpr` on the C++ side — not carried through this
//! ABI. Activation slots and weight accessors come in as two
//! pointer-of-pointers; the host stages one device pointer per
//! slot / per (accessor × layer) before launch.
//!
//! [`LaunchArgs`] mirrors the positional ABI as a `#[repr(C)]`
//! struct for callers that want to stage args; the [`launch`]
//! helper takes individual pointers so callers can skip the struct
//! if they prefer.
//!
//! # Phase 3f pool ABI extensions
//!
//! Two additive extensions, each matching a codegen flag on the
//! C++ side:
//!
//! - [`LaunchArgsQkv`] / [`launch_qkv`] — Phase 3f-2b-iii
//!   (`needs_qkv_pools=true`). Appends four pool args
//!   (`positions`, `slot_mapping`, `key_cache_ptrs`,
//!   `value_cache_ptrs`) when the variant schedules any op that
//!   reads/writes the paged KV cache + rotary metadata (today:
//!   `FusedQkvRopeCache`, `AttentionViaCache`).
//! - [`LaunchArgsAttn`] / [`launch_attn`] — Phase 3f-2d-iv-b
//!   (`needs_attention_pools=true`). Appends two attention-
//!   specific args (`seq_lens`, `block_table`) on top of the QKV
//!   prefix when the variant schedules at least one
//!   `AttentionViaCache` op. Attention implies the QKV pool; the
//!   field order mirrors the emitted kernel signature (QKV prefix
//!   then attention tail).
//!
//! Deliberately **not** here:
//! - instruction-tape fields (`instructions`, `timings`,
//!   `global_instruction_index`) — no tape.
//! - per-device pgl pointer arrays / `Bar` barrier struct —
//!   Phase 1-5 is TP=1; cross-SM sync is plain gmem atomics.
//! - shape scalars — every one is a codegen-time constexpr.
//! - per-slot / per-accessor pointer arity in the ABI itself —
//!   the kernel already knows `NUM_ACT_SLOTS` and
//!   `NUM_WEIGHT_ACCESSORS` from its constexprs. The caller must
//!   size its pointer arrays to match; mismatched lengths are UB.

#![cfg(feature = "cuda")]

// Sprint 1b iter 2: the typed mega IR (`lowered.rs` + `lowering.rs`)
// moved out into the standalone `ferrite-mega-ir` crate so the
// proc-macro can depend on it. This module is now ABI-only — Rust
// types matching the emitted megakernel's positional `extern "C"`
// signature (LaunchArgs, LaunchTier, PersistentDecodeResources,
// etc.). See `ferrite-mega-ir` for `MegaTape`, `MegaTapeBuilder`,
// `Substrate`, and the typed newtypes.

use std::ffi::c_void;

use ferrite_cuda_core::tensor::TensorView;

/// Opaque `__nv_bfloat16` — represented as `u16` on the Rust side.
/// Callers stage bf16 tensors as `*const u16` / `*mut u16`, same
/// bit width as `__nv_bfloat16`.
pub type Bf16Ptr = *mut u16;
pub type Bf16CPtr = *const u16;

/// Device-resident array of activation-slot pointers. One entry
/// per dense slot in the variant's SlotMap. The kernel indexes
/// into this array by the compile-time slot index the codegen
/// picked for each op instance.
pub type ActPtrs = *const Bf16Ptr;

/// Device-resident array of weight pointers, laid out as
/// `[num_accessors][num_layers]` row-major. `weight_ptrs[w *
/// NUM_LAYERS + l]` is layer `l` of accessor `w`. Un-layered
/// accessors occupy `l == 0` only; the remaining `NUM_LAYERS - 1`
/// entries are unused padding (any value is fine; the kernel
/// never reads them).
pub type WeightPtrs = *const Bf16CPtr;

/// Positional mirror of the emitted `extern "C"
/// ferrite_<variant>_launch` signature. Field order is ABI
/// (positional args to the C fn); Rust callers can either stage a
/// `LaunchArgs` and pass its fields, or call [`launch`] directly
/// with the same pointers in the same order.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LaunchArgs {
    pub act_ptrs: ActPtrs,
    pub weight_ptrs: WeightPtrs,
    /// `barriers[NUM_EDGES]` — dense i32 cross-CTA atomic counter
    /// array for mega-kernel barrier pairs. Null when the variant's
    /// schedule has no Barrier ops (e.g. `NUM_EDGES == 0`); otherwise
    /// the host allocates + zero-inits a `NUM_EDGES * 4` byte device
    /// buffer and passes its base pointer here. See [`I32MutPtr`].
    pub barriers: I32MutPtr,
    /// Runtime mega-kernel trace verbosity (Phase 3 step 8). Populated
    /// from `FERRITE_MEGA_TRACE` at launch stage; `0` disables all
    /// trace stanzas in the emitted kernel (cost is one compare+branch
    /// per stanza, still cheap on the hot path). See
    /// `ferrite-kernels/csrc/tk/ferrite_trace.cuh` for the level
    /// ladder. Always passed regardless of the variant's tier.
    pub trace_level: i32,
}

/// Device-resident int64 array. Used for `slot_mapping` in the
/// extended (QKV) pool ABI — sized `[NUM_TOKENS]` on the C++ side;
/// the kernel reads `slot_mapping[tok]` for paged-KV writes.
///
/// `positions` is a `[NUM_TOKENS]` array too, but its element type
/// is uint32 (see [`U32Ptr`]) — matching the host-side staging in
/// `vllm-executor::cuda_worker` (which always `h2d_u32`s positions)
/// and the non-mega ferrite kernel bindings
/// (`fused_qkv_rope_cache_{f16,bf16}` all take `positions: *const
/// u32`).
pub type I64Ptr = *const i64;

/// Device-resident int32 array. Used for `seq_lens` in the
/// attention pool ABI extension. Sized `[NUM_TOKENS]` on the C++
/// side; the kernel reads `seq_lens[tok]` as the per-sequence KV
/// length used by `attention_partial` to bound its page walk.
pub type I32Ptr = *const i32;

/// Device-resident uint32 array. Used for two fields in the extended
/// pool ABI:
///
/// - `positions` (QKV tier) — sized `[NUM_TOKENS]` on the C++ side;
///   the kernel reads `positions[tok]` to index the rope cos/sin
///   tables. Host-side staging is always `h2d_u32` (see
///   `vllm-executor::cuda_worker`), and the non-mega ferrite
///   rope/cache kernels (`fused_qkv_rope_cache_{f16,bf16}`) already
///   take `positions: *const u32`, so this matches the host/kernel
///   consensus.
/// - `block_table` (Attn tier) — sized `[NUM_TOKENS,
///   MAX_PAGES_PER_SEQ]` row-major on the C++ side; the kernel reads
///   `block_table[tok * MAX_PAGES_PER_SEQ + p]` as the paged-cache
///   block index for page `p` of token `tok`'s sequence. Host-side
///   storage is `DType::I32`; the signed→unsigned reinterpret is
///   safe because physical block IDs live in `0..num_blocks` (see
///   [`block_table_ptr`] for the full rationale).
pub type U32Ptr = *const u32;

/// Device-resident mutable int32 array. Used for `barriers` in the
/// mega pool ABI (every tier): a dense `[NUM_EDGES]` array of i32
/// cross-CTA atomic counters, one slot per producer→consumer edge
/// inserted by `mega_lowering::insert_mega_barriers`. The emitted
/// kernel calls `ferrite::barrier_signal(&barriers[edge], 1)` from
/// producer CTAs and `ferrite::barrier_wait(&barriers[edge], count)`
/// from consumer CTAs, both ultimately reaching `atomicAdd` on this
/// pointer's storage. Host allocates + zero-inits the buffer on
/// every mega-kernel launch (see `emit_mega_forward_fn`); variants
/// with `NUM_EDGES == 0` pass null here, and the kernel never
/// dereferences the pointer.
pub type I32MutPtr = *mut i32;

/// Project a host-interpreter `seqused_k` view into the mega ABI
/// [`I32Ptr`] expected by [`LaunchArgsAttn::seq_lens`].
///
/// The host side stores `seqused_k` as a contiguous `DType::I32`
/// device tensor (see `vllm-cuda/src/graph.rs` and the existing
/// `attention_decode_from_cache` call path). The mega kernel reads
/// it as `const int32_t*` — same dtype, same bit width — so this
/// is a zero-copy pointer reinterpret with no staging.
///
/// The returned pointer's validity is bounded by the lifetime of
/// the view's backing tensor; callers must keep the source tensor
/// alive for the duration of the launch.
pub fn seq_lens_ptr(view: TensorView<'_>) -> I32Ptr {
    // `as_ptr::<i32>` is defined on `GpuTensor` and takes `self`
    // by value; `TensorView` derefs to `GpuTensor` but Rust's
    // method resolution won't auto-consume through `Deref` for
    // an owned-self method, so the call sites elsewhere in this
    // crate (see `instr.rs`'s `(*ctx.fwd.input_ids).dim(0)`)
    // deref explicitly. Same pattern here.
    (*view).as_ptr::<i32>()
}

/// Project a host-interpreter `block_table` view into the mega ABI
/// [`U32Ptr`] expected by [`LaunchArgsAttn::block_table`].
///
/// The host side stores `block_table` as `DType::I32` (signed) —
/// see e.g. `vllm-cuda/src/graph.rs`, which matches the vendored
/// FlashAttention + cudagraph paths. The mega kernel, however,
/// reads it as `const uint32_t*` (see `emit_cu_variant`'s ABI
/// emission in `ferrite-forward-macro/src/interpreter/mega.rs`).
/// Page indices are non-negative (physical block IDs in
/// `0..num_blocks`), so the signed / unsigned bit patterns agree
/// for every value the kernel will encounter and this is a
/// zero-copy pointer reinterpret.
///
/// The returned pointer's validity is bounded by the lifetime of
/// the view's backing tensor; callers must keep the source tensor
/// alive for the duration of the launch.
pub fn block_table_ptr(view: TensorView<'_>) -> U32Ptr {
    // See `seq_lens_ptr` for the explicit-deref rationale.
    (*view).as_ptr::<u32>()
}

/// Project a host-interpreter `positions` view into the mega ABI
/// [`U32Ptr`] expected by [`LaunchArgsQkv::positions`] /
/// [`LaunchArgsAttn::positions`].
///
/// The host side stages `positions` as a `DType::U32` device tensor
/// (every `ForwardCtx::positions` construction path in
/// `vllm-executor::cuda_worker` uses `h2d_u32`). The mega kernel
/// reads it as `const uint32_t*` — same dtype, same bit width — so
/// this is a zero-copy pointer reinterpret.
///
/// The returned pointer's validity is bounded by the lifetime of
/// the view's backing tensor; callers must keep the source tensor
/// alive for the duration of the launch.
pub fn positions_ptr(view: TensorView<'_>) -> U32Ptr {
    // See `seq_lens_ptr` for the explicit-deref rationale.
    (*view).as_ptr::<u32>()
}

/// Project a host-interpreter `input_ids` view into the mega ABI
/// [`U32Ptr`] expected by [`LaunchArgsQkv::input_ids`] /
/// [`LaunchArgsAttn::input_ids`].
///
/// The host side stages `input_ids` as a `DType::U32` device tensor
/// (every `input_ids` construction path in `vllm-cuda/src/graph.rs`
/// uses `GpuTensor::new(..., DType::U32)`). The mega kernel reads it
/// as `const uint32_t*` — same dtype, same bit width — zero-copy
/// pointer reinterpret.
///
/// The returned pointer's validity is bounded by the lifetime of
/// the view's backing tensor; callers must keep the source tensor
/// alive for the duration of the launch.
pub fn input_ids_ptr(view: TensorView<'_>) -> U32Ptr {
    // See `seq_lens_ptr` for the explicit-deref rationale.
    (*view).as_ptr::<u32>()
}

/// Project a host-interpreter `slot_mapping` view into the mega ABI
/// [`I64Ptr`] expected by [`LaunchArgsQkv::slot_mapping`] /
/// [`LaunchArgsAttn::slot_mapping`].
///
/// The host side stages `slot_mapping` as a `DType::I64` device
/// tensor (see `vllm-executor::cuda_worker::build_attention_tensors`
/// which goes through `h2d_i64`). The mega kernel reads it as
/// `const int64_t*` — same dtype, same bit width — zero-copy
/// pointer reinterpret.
///
/// The returned pointer's validity is bounded by the lifetime of
/// the view's backing tensor; callers must keep the source tensor
/// alive for the duration of the launch.
pub fn slot_mapping_ptr(view: TensorView<'_>) -> I64Ptr {
    // See `seq_lens_ptr` for the explicit-deref rationale.
    (*view).as_ptr::<i64>()
}

/// Per-layer paged K/V base pointers. Indexed by layer id at the
/// call site: `key_cache_ptrs[l]` / `value_cache_ptrs[l]` is the
/// bf16 base of layer `l`'s paged KV block pool. Layout matches
/// `[num_blocks, block_size, num_kv_heads, head_dim]` (vLLM-NHD) —
/// same convention the vendored `flash_api` path uses.
pub type KvPtrs = *const Bf16Ptr;

/// Positional mirror of the extended (QKV) pool ABI emitted when
/// the variant's schedule includes any op that reads/writes the
/// paged KV cache + rotary metadata. Today only
/// [`FusedQkvRopeCache`] triggers it; ops added later that touch
/// the same pools slot in without another ABI rev.
///
/// The extra four fields follow `weight_ptrs` in declaration order
/// — same order as the emitted kernel signature. Callers pair this
/// with [`LaunchFnQkv`] and [`launch_qkv`].
///
/// [`FusedQkvRopeCache`]: https://github.com/ferrite/impl-lib
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LaunchArgsQkv {
    pub act_ptrs: ActPtrs,
    pub weight_ptrs: WeightPtrs,
    /// `input_ids[NUM_TOKENS]` — per-token uint32 vocab index read by
    /// [`Embed`] ops. Added at position 2 (after `weight_ptrs`, before
    /// `positions`) in Phase 3f-2e-iii; every schedule that contains
    /// `Embed` triggers the QKV pool ABI so this field is in scope.
    /// Variants that don't use `Embed` still carry the field (callers
    /// can pass null) so the tier enum stays linear.
    pub input_ids: U32Ptr,
    pub positions: U32Ptr,
    pub slot_mapping: I64Ptr,
    pub key_cache_ptrs: KvPtrs,
    pub value_cache_ptrs: KvPtrs,
    /// `barriers[NUM_EDGES]` — same semantics as
    /// [`LaunchArgs::barriers`]. Present on every tier so the mega
    /// kernel ABI is uniform across Base/Qkv/Attn; callers that
    /// don't need barriers (variant has no Barrier ops / FERRITE_MEGA=0)
    /// pass null.
    pub barriers: I32MutPtr,
    /// Runtime mega-kernel trace verbosity. Same semantics as
    /// [`LaunchArgs::trace_level`].
    pub trace_level: i32,
}

/// Positional mirror of the attention-extended pool ABI emitted
/// when the variant's schedule contains at least one
/// `AttentionViaCache` op (Phase 3f-2d-iv-b). Layers the two
/// attention-specific args on top of [`LaunchArgsQkv`]:
///
/// - `seq_lens` — per-token int32 KV length, sized `[NUM_TOKENS]`.
/// - `block_table` — per-token paged-cache page indirection, sized
///   `[NUM_TOKENS, MAX_PAGES_PER_SEQ]` row-major as uint32.
///
/// Attention implies the QKV pool (keys/values come in via the
/// per-layer `key_cache_ptrs` / `value_cache_ptrs`), so the field
/// order is the QKV prefix followed by `seq_lens` then
/// `block_table` — same order as the emitted kernel signature.
/// Callers pair this with [`LaunchFnAttn`] and [`launch_attn`].
///
/// [`FusedQkvRopeCache`]: https://github.com/ferrite/impl-lib
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LaunchArgsAttn {
    pub act_ptrs: ActPtrs,
    pub weight_ptrs: WeightPtrs,
    /// `input_ids[NUM_TOKENS]` — same semantics as
    /// [`LaunchArgsQkv::input_ids`]. Attn tier inherits the QKV prefix,
    /// so the field lives at offset 16 here too.
    pub input_ids: U32Ptr,
    pub positions: U32Ptr,
    pub slot_mapping: I64Ptr,
    pub key_cache_ptrs: KvPtrs,
    pub value_cache_ptrs: KvPtrs,
    pub seq_lens: I32Ptr,
    pub block_table: U32Ptr,
    /// Row stride of `block_table` (= number of uint32 entries per
    /// token's row in the `[NUM_TOKENS, stride]` layout). Wave F adds
    /// this so the emitted `attention_partial::loader` can index
    /// `block_table[token * stride + p]` for batched decode. The
    /// host allocates `block_table` with
    /// `stride = max(block_ids[i].len())` across the in-flight
    /// sequences (see `cuda_worker::build_attention_tensors`) —
    /// hence a runtime arg rather than a codegen constexpr.
    /// NUM_TOKENS==1 variants may pass any value (the loader only
    /// ever dereferences `block_table[0 * stride + p] == block_table[p]`
    /// for a single-token tile); `0` is conventional but not required.
    pub block_table_stride: u32,
    /// `barriers[NUM_EDGES]` — same semantics as
    /// [`LaunchArgs::barriers`]. Trails the attention tail on the Attn
    /// tier; `dispatch_launch` threads it down to each tier's launcher.
    pub barriers: I32MutPtr,
    /// Runtime mega-kernel trace verbosity. Same semantics as
    /// [`LaunchArgs::trace_level`].
    pub trace_level: i32,
}

/// Function-pointer type matching the emitted `extern "C"
/// ferrite_<variant>_launch` signature. Caller supplies the fn
/// pointer; macro-emitted sites populate it from the variant's
/// extern decl.
///
/// Returns the CUDA runtime error code as `i32` (the raw
/// `cudaError_t` enum underlying type). Callers interpret
/// `0 == cudaSuccess`.
pub type LaunchFn = unsafe extern "C" fn(
    act_ptrs: ActPtrs,
    weight_ptrs: WeightPtrs,
    barriers: I32MutPtr,
    trace_level: i32,
    stream: *mut c_void,
) -> i32;

/// Function-pointer type matching the emitted extended (QKV)
/// `extern "C" ferrite_<variant>_launch` signature. Same shape as
/// [`LaunchFn`] with the four extra pool args spliced in between
/// `weight_ptrs` and `stream`, positionally matching
/// [`LaunchArgsQkv`].
///
/// Returns the CUDA runtime error code as `i32` (the raw
/// `cudaError_t` enum underlying type). Callers interpret
/// `0 == cudaSuccess`.
pub type LaunchFnQkv = unsafe extern "C" fn(
    act_ptrs: ActPtrs,
    weight_ptrs: WeightPtrs,
    input_ids: U32Ptr,
    positions: U32Ptr,
    slot_mapping: I64Ptr,
    key_cache_ptrs: KvPtrs,
    value_cache_ptrs: KvPtrs,
    barriers: I32MutPtr,
    trace_level: i32,
    stream: *mut c_void,
) -> i32;

/// Function-pointer type matching the emitted attention-extended
/// (`AttentionViaCache`) `extern "C" ferrite_<variant>_launch`
/// signature. Same shape as [`LaunchFnQkv`] with the two
/// attention-specific args (`seq_lens`, `block_table`) spliced in
/// between the KV pool pair and `stream`, positionally matching
/// [`LaunchArgsAttn`].
///
/// Returns the CUDA runtime error code as `i32` (the raw
/// `cudaError_t` enum underlying type). Callers interpret
/// `0 == cudaSuccess`.
pub type LaunchFnAttn = unsafe extern "C" fn(
    act_ptrs: ActPtrs,
    weight_ptrs: WeightPtrs,
    input_ids: U32Ptr,
    positions: U32Ptr,
    slot_mapping: I64Ptr,
    key_cache_ptrs: KvPtrs,
    value_cache_ptrs: KvPtrs,
    seq_lens: I32Ptr,
    block_table: U32Ptr,
    block_table_stride: u32,
    barriers: I32MutPtr,
    trace_level: i32,
    stream: *mut c_void,
) -> i32;

/// Thin wrapper around a per-variant launch fn pointer.
///
/// The stream is passed as `*mut c_void` (opaque `cudaStream_t`);
/// pass `std::ptr::null_mut()` for the default stream.
///
/// # Safety
///
/// `launch_fn` must be the extern-C symbol
/// `ferrite_<variant>_launch` paired with the variant that sourced
/// `args` — the ABI is positional and calling with the wrong
/// variant's pointers (say, a tiny-variant's act/weight ptrs
/// against a full-variant's symbol) is UB. Callers get the fn
/// pointer from a macro-emitted `extern "C" { fn
/// ferrite_<variant>_launch(...); }` block, so the pairing is
/// statically guaranteed at every call site the proc-macro writes.
///
/// `act_ptrs` must point to an array of at least `NUM_ACT_SLOTS`
/// device pointers in device-accessible memory (gmem on a single
/// GPU is sufficient). Likewise `weight_ptrs` must hold at least
/// `NUM_WEIGHT_ACCESSORS * NUM_LAYERS` entries. Both counts are
/// codegen-time constants recorded in the banner at the top of
/// the emitted `.cu`.
pub unsafe fn launch(
    launch_fn: LaunchFn,
    args: LaunchArgs,
    stream: *mut c_void,
) -> Result<(), i32> {
    let rc = unsafe {
        launch_fn(
            args.act_ptrs,
            args.weight_ptrs,
            args.barriers,
            args.trace_level,
            stream,
        )
    };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

/// Thin wrapper around a per-variant extended (QKV) launch fn
/// pointer. Mirrors [`launch`] for the extended pool ABI:
/// callers supply the extern-C symbol and the staged
/// [`LaunchArgsQkv`]; this helper unpacks fields in the positional
/// order the emitted kernel expects.
///
/// # Safety
///
/// `launch_fn` must be the extern-C symbol
/// `ferrite_<variant>_launch` for a variant whose codegen emitted
/// the extended pool ABI (i.e. `needs_qkv_pools=true` — the
/// variant's schedule contains at least one op that reads/writes
/// paged KV + rotary metadata). Calling with the wrong variant's
/// pointers is UB.
///
/// `positions` must point to at least `NUM_TOKENS` uint32 entries
/// in device-accessible memory; `slot_mapping` must point to at
/// least `NUM_TOKENS` int64 entries. `key_cache_ptrs` /
/// `value_cache_ptrs` must each hold `NUM_LAYERS` bf16 base pointers
/// (one per layer's paged KV pool).
/// Both counts are codegen-time constants recorded in the banner
/// at the top of the emitted `.cu`.
///
/// `act_ptrs` / `weight_ptrs` follow the same sizing rules as
/// [`launch`].
pub unsafe fn launch_qkv(
    launch_fn: LaunchFnQkv,
    args: LaunchArgsQkv,
    stream: *mut c_void,
) -> Result<(), i32> {
    let rc = unsafe {
        launch_fn(
            args.act_ptrs,
            args.weight_ptrs,
            args.input_ids,
            args.positions,
            args.slot_mapping,
            args.key_cache_ptrs,
            args.value_cache_ptrs,
            args.barriers,
            args.trace_level,
            stream,
        )
    };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

/// Thin wrapper around a per-variant attention-extended launch fn
/// pointer. Mirrors [`launch_qkv`] for the attention pool ABI
/// (Phase 3f-2d-iv-b): callers supply the extern-C symbol and the
/// staged [`LaunchArgsAttn`]; this helper unpacks fields in the
/// positional order the emitted kernel expects.
///
/// # Safety
///
/// `launch_fn` must be the extern-C symbol
/// `ferrite_<variant>_launch` for a variant whose codegen emitted
/// the attention pool ABI (i.e. the variant's schedule contains at
/// least one `AttentionViaCache` op). Calling with the wrong
/// variant's pointers is UB.
///
/// `seq_lens` must point to at least `NUM_TOKENS` int32 entries in
/// device-accessible memory. `block_table` must point to at least
/// `NUM_TOKENS * args.block_table_stride` uint32 entries row-major;
/// `block_table_stride` is the runtime row stride (Wave F).
///
/// `act_ptrs` / `weight_ptrs` / `positions` / `slot_mapping` /
/// `key_cache_ptrs` / `value_cache_ptrs` follow the same sizing
/// rules as [`launch_qkv`].
pub unsafe fn launch_attn(
    launch_fn: LaunchFnAttn,
    args: LaunchArgsAttn,
    stream: *mut c_void,
) -> Result<(), i32> {
    let rc = unsafe {
        launch_fn(
            args.act_ptrs,
            args.weight_ptrs,
            args.input_ids,
            args.positions,
            args.slot_mapping,
            args.key_cache_ptrs,
            args.value_cache_ptrs,
            args.seq_lens,
            args.block_table,
            args.block_table_stride,
            args.barriers,
            args.trace_level,
            stream,
        )
    };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

/// Which pool-ABI tier a megakernel variant was codegen'd for.
///
/// The proc-macro computes this at codegen time from the variant's
/// lowered schedule and pairs each variant with a fn-pointer of the
/// matching ABI:
///
/// - [`LaunchTier::Base`] — schedule touches neither the paged KV
///   cache nor rotary metadata. Variant exports a [`LaunchFn`] and
///   callers stage a [`LaunchArgs`].
/// - [`LaunchTier::Qkv`] — schedule includes at least one op with
///   `needs_qkv_pools=true` (today: [`FusedQkvRopeCache`]). Variant
///   exports a [`LaunchFnQkv`] and callers stage a [`LaunchArgsQkv`].
/// - [`LaunchTier::Attn`] — schedule additionally includes at least
///   one op with `needs_attention_pools=true` (today:
///   `AttentionViaCache`). Variant exports a [`LaunchFnAttn`] and
///   callers stage a [`LaunchArgsAttn`].
///
/// The tiers are nested: `Attn` ⊃ `Qkv` ⊃ `Base`. This is enforced
/// on the C++ side by the macro's `needs_qkv_pools =
/// needs_attention_pools || …` derivation, and on the Rust side by
/// the `#[repr(C)]` prefix-compat guarantee already asserted by the
/// `launch_args_attn_prefix_matches_qkv` test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchTier {
    Base,
    Qkv,
    Attn,
    /// Attn-superset: persistent multi-step cooperative kernel. M=1 only.
    /// Adds per-step arrays + output_token_ids. Launched via
    /// `cudaLaunchCooperativeKernel`; the Rust caller stages
    /// [`LaunchArgsMultiStep`] and calls [`launch_multi_step`].
    MultiStep,
    /// Persistent decode kernel. Launched once via
    /// `cudaLaunchCooperativeKernel`; communicates per-step input/output
    /// through a pinned-memory `FerritePinnedProtocol` buffer via
    /// cpu_step/gpu_step counters. No per-step kernel launches.
    /// Callers use [`PersistentDecodeSession`] which owns the protocol buffer.
    PersistentDecode,
}

/// Multi-step args: Attn-tier superset with per-step arrays.
/// Stages per-step pointer arrays into the multi-step cooperative kernel.
/// `input_ids_multi` is mutable — the kernel writes the next token
/// ID to `input_ids_multi[step+1]` via in-kernel argmax.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LaunchArgsMultiStep {
    pub act_ptrs: ActPtrs,
    pub weight_ptrs: WeightPtrs,
    /// Mutable per-step input token IDs. `[0]` is the initial token
    /// (host-provided); `[1..num_steps]` are filled by in-kernel argmax.
    pub input_ids_multi: *mut u32,
    /// Per-step rotary positions. `[num_steps]` device array.
    pub positions_multi: U32Ptr,
    /// Per-step paged-KV slot mappings. `[num_steps]` device array.
    pub slot_mapping_multi: I64Ptr,
    pub key_cache_ptrs: KvPtrs,
    pub value_cache_ptrs: KvPtrs,
    /// Per-step sequence lengths. `[num_steps]` device array.
    pub seq_lens_multi: I32Ptr,
    /// Paged-cache block table (shared across all steps).
    pub block_table: U32Ptr,
    pub block_table_stride: u32,
    pub barriers: I32MutPtr,
    pub trace_level: i32,
    pub num_steps: i32,
    /// Output token IDs written by in-kernel argmax. `[num_steps]` device array.
    pub output_token_ids: *mut u32,
}

/// Multi-step cooperative kernel launch fn pointer type. Matches the
/// `ferrite_{variant}_ms_launch` extern-C signature emitted for
/// `LaunchTier::MultiStep` variants.
pub type LaunchFnMultiStep = unsafe extern "C" fn(
    act_ptrs: ActPtrs,
    weight_ptrs: WeightPtrs,
    input_ids_multi: *mut u32,
    positions_multi: U32Ptr,
    slot_mapping_multi: I64Ptr,
    key_cache_ptrs: KvPtrs,
    value_cache_ptrs: KvPtrs,
    seq_lens_multi: I32Ptr,
    block_table: U32Ptr,
    block_table_stride: u32,
    barriers: I32MutPtr,
    trace_level: i32,
    num_steps: i32,
    output_token_ids: *mut u32,
    stream: *mut c_void,
) -> i32;

/// Call a `LaunchFnMultiStep` with staged args.
///
/// # Safety
/// The caller must ensure all device pointers in `args` are valid
/// and that the function pointer is the correct `_ms_launch` symbol.
pub unsafe fn launch_multi_step(
    f: LaunchFnMultiStep,
    args: LaunchArgsMultiStep,
    stream: *mut c_void,
) -> Result<(), i32> {
    let rc = unsafe {
        f(
            args.act_ptrs,
            args.weight_ptrs,
            args.input_ids_multi,
            args.positions_multi,
            args.slot_mapping_multi,
            args.key_cache_ptrs,
            args.value_cache_ptrs,
            args.seq_lens_multi,
            args.block_table,
            args.block_table_stride,
            args.barriers,
            args.trace_level,
            args.num_steps,
            args.output_token_ids,
            stream,
        )
    };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

/// Persistent decode launch args. The `protocol` pointer is the
/// type-erased `FerritePinnedProtocol<NUM_TOKENS, MAX_BLOCKS_PER_SEQ>*`
/// from the kernel's pinned memory buffer. All per-step coordination
/// goes through that buffer; there are no pre-staged arrays.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PersistentDecodeLaunchArgs {
    pub act_ptrs: ActPtrs,
    pub weight_ptrs: WeightPtrs,
    /// Type-erased pointer to the `FerritePinnedProtocol` pinned buffer.
    /// Cast by the kernel to `FerritePinnedProtocol<NUM_TOKENS, MAX_BLOCKS_PER_SEQ>*`.
    pub protocol: *mut std::ffi::c_void,
    pub key_cache_ptrs: KvPtrs,
    pub value_cache_ptrs: KvPtrs,
    pub barriers: I32MutPtr,
    pub trace_level: i32,
}

/// Persistent decode kernel launch fn pointer. Matches the
/// `ferrite_{variant}_launch` extern-C signature for persistent decode
/// variants. Performs a `cudaLaunchCooperativeKernel`; the kernel then
/// runs until `protocol->stop_flag` is set.
pub type LaunchFnPersistentDecode = unsafe extern "C" fn(
    act_ptrs: ActPtrs,
    weight_ptrs: WeightPtrs,
    protocol: *mut std::ffi::c_void,
    key_cache_ptrs: KvPtrs,
    value_cache_ptrs: KvPtrs,
    barriers: I32MutPtr,
    trace_level: i32,
    stream: *mut std::ffi::c_void,
) -> i32;

/// Launch a persistent decode kernel. Returns immediately after
/// `cudaLaunchCooperativeKernel` — the kernel continues running in
/// the background. Communicate with it via the protocol buffer's
/// cpu_step/gpu_step counters. To stop it: set `protocol.stop_flag = 1`
/// then increment `cpu_step`.
///
/// # Safety
/// The caller must ensure all device/pinned pointers in `args` are valid
/// and that the function pointer is the correct persistent decode launch symbol.
pub unsafe fn launch_persistent_decode(
    f: LaunchFnPersistentDecode,
    args: PersistentDecodeLaunchArgs,
    stream: *mut std::ffi::c_void,
) -> Result<(), i32> {
    let rc = unsafe {
        f(
            args.act_ptrs,
            args.weight_ptrs,
            args.protocol,
            args.key_cache_ptrs,
            args.value_cache_ptrs,
            args.barriers,
            args.trace_level,
            stream,
        )
    };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

/// Tier-tagged wrapper around a variant's launch fn pointer. The
/// macro emits one of these per canonical variant, picking the
/// constructor that matches the variant's ABI tier. Call sites that
/// see a generic variant (e.g. a per-bucket dispatcher keyed on
/// `(num_tokens, sk_bucket)`) can carry this as a single type and
/// defer the ABI match to [`dispatch_launch`].
///
/// The variant's tier is discoverable via [`LaunchFnAny::tier`].
#[derive(Clone, Copy)]
pub enum LaunchFnAny {
    Base(LaunchFn),
    Qkv(LaunchFnQkv),
    Attn(LaunchFnAttn),
    MultiStep(LaunchFnMultiStep),
    PersistentDecode(LaunchFnPersistentDecode),
}

impl LaunchFnAny {
    /// The ABI tier this fn pointer was emitted for.
    pub fn tier(&self) -> LaunchTier {
        match self {
            LaunchFnAny::Base(_) => LaunchTier::Base,
            LaunchFnAny::Qkv(_) => LaunchTier::Qkv,
            LaunchFnAny::Attn(_) => LaunchTier::Attn,
            LaunchFnAny::MultiStep(_) => LaunchTier::MultiStep,
            LaunchFnAny::PersistentDecode(_) => LaunchTier::PersistentDecode,
        }
    }
}

impl std::fmt::Debug for LaunchFnAny {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Avoid printing the raw fn-pointer address (varies per run,
        // pollutes snapshot diffs); the tier is the only stable,
        // useful discriminator for debug.
        write!(f, "LaunchFnAny::{:?}", self.tier())
    }
}

/// Tier-matched dispatch to the variant's launcher. The caller
/// stages the fat-superset [`LaunchArgsAttn`] once; this helper
/// matches on the fn pointer's tier and invokes the matching
/// [`launch`] / [`launch_qkv`] / [`launch_attn`] helper, projecting
/// the args struct down to the tier's required prefix via the
/// `#[repr(C)]` prefix-compat guarantee (see
/// `launch_args_attn_prefix_matches_qkv`).
///
/// This keeps the interpreter call site single-shape: stage
/// `LaunchArgsAttn`, carry `LaunchFnAny`, call `dispatch_launch`.
/// No per-tier branching above this layer.
///
/// # Safety
///
/// Same ABI-pairing rules as the three lower-level helpers:
///
/// - The fn pointer inside `fn_any` must be the extern-C symbol
///   `ferrite_<variant>_launch` for a variant that was codegen'd at
///   exactly the matching tier. The macro pairs these at every
///   emission site, so call sites that take `LaunchFnAny` from the
///   macro-emitted variant table satisfy this statically.
/// - The fields of `args` required by the dispatched tier must be
///   valid device pointers sized per the variant's constexprs
///   (`NUM_ACT_SLOTS`, `NUM_WEIGHT_ACCESSORS * NUM_LAYERS`,
///   `NUM_TOKENS`, etc.). Unused tail fields are ignored by the
///   lower tiers and may hold any value (null is fine).
pub unsafe fn dispatch_launch(
    fn_any: LaunchFnAny,
    args: LaunchArgsAttn,
    stream: *mut c_void,
) -> Result<(), i32> {
    match fn_any {
        LaunchFnAny::Base(f) => unsafe {
            launch(
                f,
                LaunchArgs {
                    act_ptrs: args.act_ptrs,
                    weight_ptrs: args.weight_ptrs,
                    barriers: args.barriers,
                    trace_level: args.trace_level,
                },
                stream,
            )
        },
        LaunchFnAny::Qkv(f) => unsafe {
            launch_qkv(
                f,
                LaunchArgsQkv {
                    act_ptrs: args.act_ptrs,
                    weight_ptrs: args.weight_ptrs,
                    input_ids: args.input_ids,
                    positions: args.positions,
                    slot_mapping: args.slot_mapping,
                    key_cache_ptrs: args.key_cache_ptrs,
                    value_cache_ptrs: args.value_cache_ptrs,
                    barriers: args.barriers,
                    trace_level: args.trace_level,
                },
                stream,
            )
        },
        LaunchFnAny::Attn(f) => unsafe { launch_attn(f, args, stream) },
        LaunchFnAny::MultiStep(_) => {
            // MultiStep kernels use launch_multi_step(), not dispatch_launch().
            // Callers that hold a MultiStep fn should call launch_multi_step directly.
            panic!("dispatch_launch: MultiStep variant requires launch_multi_step(), not dispatch_launch()");
        }
        LaunchFnAny::PersistentDecode(_) => {
            // PersistentDecode kernels are managed by PersistentDecodeSession, not dispatch_launch().
            panic!("dispatch_launch: PersistentDecode variant is managed by PersistentDecodeSession");
        }
    }
}

// ============================================================
// PersistentDecodeResources — GPU-side tensors kept alive for a
// persistent decode session.
// ============================================================

/// Device-side allocations that must outlive the persistent decode
/// kernel. Created by the generated `start_persistent_decode_<canonical>`
/// fn at session start; dropped (returning memory to the caching
/// pool) when the session ends.
#[cfg(feature = "cuda")]
pub struct PersistentDecodeResources {
    /// Activation scratch tensors — one per schedule slot. Their raw
    /// device pointers are stored in `act_ptrs_dev` and read by the
    /// persistent kernel on every decode step.
    pub slot_tensors: Vec<ferrite_cuda_core::alloc::OwnedTensor>,
    /// Device array of `*mut u16` pointers (one per slot).
    pub act_ptrs_dev: ferrite_cuda_core::alloc::OwnedTensor,
    /// Device array of `*const u16` weight pointers (NUM_ACCESSORS × NUM_LAYERS).
    pub weight_ptrs_dev: ferrite_cuda_core::alloc::OwnedTensor,
    /// Device `i32[NUM_EDGES]` barrier counters. The kernel resets these
    /// at the start of each step, so the CPU never needs to touch them.
    pub barriers_dev: Option<ferrite_cuda_core::alloc::OwnedTensor>,
}

// ============================================================
// PersistentDecodeSession — CPU-side manager for a persistent decode kernel.
// ============================================================

/// Protocol buffer layout constants for NUM_TOKENS=1, MAX_BLOCKS_PER_SEQ=512.
/// These must stay in sync with FerritePinnedProtocol<1, 512> in
/// ferrite_persistent_decode_protocol.cuh. Layout (packed, with explicit _align_pad):
///
///   [0]   cpu_step          u32
///   [4]   gpu_step          u32
///   [8]   stop_flag         u32
///   [12]  _sync_pad         u8×52  → total sync header = 64 bytes
///   [64]  input_ids[1]      u32
///   [68]  positions[1]      u32
///   [72]  seq_lens[1]       i32
///   [76]  _align_pad        u8×4   (N=1 is odd; 64+12=76, need 4 to reach 80)
///   [80]  slot_mapping[1]   i64
///   [88]  block_table_stride u32
///   [92]  _bt_stride_pad    u32
///   [96]  block_table[512]  u32×512 = 2048 bytes
///   [2144] output_tokens[1] u32
///   total: 2148 bytes; allocate 4096 (one page) for alignment
pub mod protocol_layout {
    pub const N: usize = 1;
    pub const MAX_BLOCKS: usize = 512;

    pub const OFFSET_CPU_STEP: usize = 0;
    pub const OFFSET_GPU_STEP: usize = 4;
    pub const OFFSET_STOP_FLAG: usize = 8;
    pub const OFFSET_INPUT_IDS: usize = 64;
    pub const OFFSET_POSITIONS: usize = 68;
    pub const OFFSET_SEQ_LENS: usize = 72;
    // 4 bytes _align_pad at [76]
    pub const OFFSET_SLOT_MAPPING: usize = 80;
    pub const OFFSET_BLOCK_TABLE_STRIDE: usize = 88;
    // 4 bytes _bt_stride_pad at [92]
    pub const OFFSET_BLOCK_TABLE: usize = 96;
    pub const OFFSET_OUTPUT_TOKENS: usize = 96 + MAX_BLOCKS * 4; // 2144
    pub const PROTOCOL_BYTES: usize = 4096; // page-aligned, actual use is 2148
}

/// Unsafe helper: volatile write u32 at byte offset into pinned buffer.
#[inline(always)]
unsafe fn proto_write_u32(buf: *mut u8, offset: usize, val: u32) {
    let p = buf.add(offset) as *mut u32;
    p.write_volatile(val);
}

/// Unsafe helper: volatile write i32.
#[inline(always)]
unsafe fn proto_write_i32(buf: *mut u8, offset: usize, val: i32) {
    let p = buf.add(offset) as *mut i32;
    p.write_volatile(val);
}

/// Unsafe helper: volatile write i64.
#[inline(always)]
unsafe fn proto_write_i64(buf: *mut u8, offset: usize, val: i64) {
    let p = buf.add(offset) as *mut i64;
    p.write_volatile(val);
}

/// Unsafe helper: volatile read u32.
#[inline(always)]
unsafe fn proto_read_u32(buf: *const u8, offset: usize) -> u32 {
    let p = buf.add(offset) as *const u32;
    p.read_volatile()
}

/// CPU-side manager for a persistent decode kernel session.
///
/// Owns the pinned protocol buffer, the GPU-side activation/weight
/// tensors (`PersistentDecodeResources`), and the last greedy output
/// token. Created once when the first M=1 decode step arrives;
/// kept alive until the batch changes or the server shuts down.
///
/// `resources` starts as `None`; the generated `forward()` dispatch
/// populates it on the first call via `start_persistent_decode_*`.
///
/// # Safety
/// The CUDA kernel runs concurrently on the GPU. All protocol-buffer
/// accesses must go through the provided methods, which insert the
/// appropriate memory fences.
pub struct PersistentDecodeSession {
    /// Pinned (page-locked) protocol buffer.
    /// Layout: `protocol_layout::PROTOCOL_BYTES` bytes.
    protocol: *mut u8,
    protocol_capacity: usize,
    /// Monotonically increasing step counter (CPU side).
    step: u32,
    /// GPU-side tensors held alive for the persistent kernel.
    /// `None` until the first `forward()` call populates it via
    /// `start_persistent_decode_*`.
    #[cfg(feature = "cuda")]
    pub resources: Option<PersistentDecodeResources>,
    /// Output token written by the most recent `poll_output_token()`.
    /// Read by the executor after `forward()` returns.
    pub last_output_token: u32,
}

// SAFETY: PersistentDecodeSession is used from a single executor thread. The protocol
// pointer is to pinned memory which is always valid while the session lives.
unsafe impl Send for PersistentDecodeSession {}

impl PersistentDecodeSession {
    /// Allocate the protocol buffer via `cudaMallocHost`.
    ///
    /// # Safety
    /// Must be called with an active CUDA context.
    pub unsafe fn alloc() -> Result<Self, i32> {
        use protocol_layout::PROTOCOL_BYTES;
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = cuda_malloc_host(&mut ptr, PROTOCOL_BYTES);
        if rc != 0 {
            return Err(rc);
        }
        let buf = ptr as *mut u8;
        // Zero-initialize the protocol buffer. cpu_step=0, gpu_step=0, stop_flag=0.
        std::ptr::write_bytes(buf, 0, PROTOCOL_BYTES);
        Ok(Self {
            protocol: buf,
            protocol_capacity: PROTOCOL_BYTES,
            step: 0,
            #[cfg(feature = "cuda")]
            resources: None,
            last_output_token: 0,
        })
    }

    /// Raw pointer to the protocol buffer. Passed to the persistent decode kernel as `protocol`.
    pub fn protocol_ptr(&self) -> *mut std::ffi::c_void {
        self.protocol as *mut std::ffi::c_void
    }

    /// Current step index (mirrors kernel's expected `__step`).
    pub fn current_step(&self) -> u32 {
        self.step
    }

    /// Write all per-step input fields for decode step `self.step`.
    /// Must be called BEFORE `signal_cpu_step()`.
    ///
    /// # Safety
    /// Writes to pinned memory; caller must not call concurrently with
    /// the kernel reading these fields (guaranteed by the cpu_step
    /// protocol: kernel only reads after cpu_step advances).
    pub unsafe fn write_step_input(
        &self,
        input_id: u32,
        position: u32,
        seq_len: i32,
        slot_mapping: i64,
        block_table_stride: u32,
        block_ids: &[u32],
    ) {
        use protocol_layout::*;
        let p = self.protocol;
        proto_write_u32(p, OFFSET_INPUT_IDS, input_id);
        proto_write_u32(p, OFFSET_POSITIONS, position);
        proto_write_i32(p, OFFSET_SEQ_LENS, seq_len);
        proto_write_i64(p, OFFSET_SLOT_MAPPING, slot_mapping);
        proto_write_u32(p, OFFSET_BLOCK_TABLE_STRIDE, block_table_stride);
        let n = block_ids.len().min(MAX_BLOCKS);
        for (i, &bid) in block_ids[..n].iter().enumerate() {
            proto_write_u32(p, OFFSET_BLOCK_TABLE + i * 4, bid);
        }
        // Zero-pad entries beyond actual page count.
        for i in n..block_table_stride.min(MAX_BLOCKS as u32) as usize {
            proto_write_u32(p, OFFSET_BLOCK_TABLE + i * 4, 0);
        }
    }

    /// Increment cpu_step to signal the kernel that step input is ready.
    /// A release fence ensures all input writes above are visible before
    /// the increment.
    ///
    /// # Safety
    /// Must be called after `write_step_input` completes.
    pub unsafe fn signal_cpu_step(&mut self) {
        use protocol_layout::OFFSET_CPU_STEP;
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        proto_write_u32(self.protocol, OFFSET_CPU_STEP, self.step + 1);
    }

    /// Spin-poll until gpu_step > current step. Returns the greedy
    /// argmax token written by the kernel to output_tokens[0].
    ///
    /// # Safety
    /// Must be called after `signal_cpu_step` for this step. Does not
    /// block CPU threads from doing other work — pure spin. For
    /// production use, replace with a condition variable or timed wait.
    pub unsafe fn poll_output_token(&mut self) -> u32 {
        use protocol_layout::{OFFSET_GPU_STEP, OFFSET_OUTPUT_TOKENS};
        loop {
            let gpu_step = proto_read_u32(self.protocol, OFFSET_GPU_STEP);
            if gpu_step > self.step {
                // Acquire fence: see kernel's writes to output_tokens.
                std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
                let token = proto_read_u32(self.protocol, OFFSET_OUTPUT_TOKENS);
                self.step += 1;
                self.last_output_token = token;
                return token;
            }
            std::hint::spin_loop();
        }
    }

    /// Signal the kernel to stop and wake it so it observes the flag.
    /// Writes stop_flag=1 then bumps cpu_step, so the kernel exits its
    /// spin-wait, crosses the grid.sync(), sees stop_flag, and breaks.
    /// Caller must synchronize the CUDA stream before dropping the session
    /// to ensure the kernel has actually exited before pinned memory is freed.
    ///
    /// # Safety
    /// After calling this, no further `write_step_input` / `signal_cpu_step` / `poll_output_token`
    /// calls are safe.
    pub unsafe fn request_stop(&self) {
        use protocol_layout::{OFFSET_CPU_STEP, OFFSET_STOP_FLAG};
        proto_write_u32(self.protocol, OFFSET_STOP_FLAG, 1);
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        // Bump cpu_step to wake the kernel from wait_for_cpu_step.
        // Without this the kernel spins forever and never sees stop_flag.
        proto_write_u32(self.protocol, OFFSET_CPU_STEP, self.step + 1);
    }
}

impl Drop for PersistentDecodeSession {
    fn drop(&mut self) {
        // Free the pinned host buffer. Caller must have called request_stop()
        // AND synchronized the CUDA stream before dropping, so the kernel has
        // already exited. Freeing pinned memory while the kernel is still live
        // is UB.
        unsafe {
            if !self.protocol.is_null() {
                let _ = cuda_free_host(self.protocol as *mut std::ffi::c_void);
            }
        }
    }
}

// Thin wrappers around the CUDA driver API for pinned memory alloc/free.
// These are only called on code paths that already have an active CUDA context.
#[cfg(feature = "cuda")]
unsafe fn cuda_malloc_host(ptr: &mut *mut std::ffi::c_void, size: usize) -> i32 {
    unsafe extern "C" {
        fn cudaMallocHost(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    }
    unsafe { cudaMallocHost(ptr, size) }
}
#[cfg(not(feature = "cuda"))]
unsafe fn cuda_malloc_host(_ptr: &mut *mut std::ffi::c_void, _size: usize) -> i32 {
    // Non-CUDA builds: pinned alloc not available; use regular alloc for tests.
    -1
}

#[cfg(feature = "cuda")]
unsafe fn cuda_free_host(ptr: *mut std::ffi::c_void) -> i32 {
    unsafe extern "C" {
        fn cudaFreeHost(ptr: *mut std::ffi::c_void) -> i32;
    }
    unsafe { cudaFreeHost(ptr) }
}
#[cfg(not(feature = "cuda"))]
unsafe fn cuda_free_host(_ptr: *mut std::ffi::c_void) -> i32 { 0 }

// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Pool ABI: three pointers (act, weight, barriers), each 8 bytes
    // on 64-bit targets, aligned to 8. No padding between fields.
    // Phase 3 step 6: barriers trails `weight_ptrs` on every tier so
    // the Base/Qkv/Attn projection in `dispatch_launch` can lift the
    // fat `LaunchArgsAttn.barriers` into the tier-specific arg.
    #[test]
    fn launch_args_abi_size() {
        // 24 bytes of pointers (3×8) + 4-byte `trace_level` + 4 pad
        // (Phase 3 step 8).
        assert_eq!(std::mem::size_of::<LaunchArgs>(), 32);
        assert_eq!(std::mem::align_of::<LaunchArgs>(), 8);
    }

    // Extended (QKV) pool ABI: eight pointers, each 8 bytes on
    // 64-bit targets, aligned to 8. Field order matches the emitted
    // kernel signature: input_ids (Phase 3f-2e-iii) follows
    // weight_ptrs, then positions, slot_mapping, key_cache_ptrs,
    // value_cache_ptrs, then barriers (Phase 3 step 6).
    #[test]
    fn launch_args_qkv_abi_size() {
        // 8 pointers × 8 bytes = 64, + 4-byte `trace_level` + 4 pad
        // (Phase 3 step 8) = 72.
        assert_eq!(std::mem::size_of::<LaunchArgsQkv>(), 72);
        assert_eq!(std::mem::align_of::<LaunchArgsQkv>(), 8);
    }

    // Field offsets match the positional ABI the emitted
    // extern-C launcher expects. If any offset drifts, calls
    // through `launch_qkv` will pass garbage — keep this tight.
    #[test]
    fn launch_args_qkv_field_offsets() {
        use std::mem::offset_of;
        assert_eq!(offset_of!(LaunchArgsQkv, act_ptrs), 0);
        assert_eq!(offset_of!(LaunchArgsQkv, weight_ptrs), 8);
        assert_eq!(offset_of!(LaunchArgsQkv, input_ids), 16);
        assert_eq!(offset_of!(LaunchArgsQkv, positions), 24);
        assert_eq!(offset_of!(LaunchArgsQkv, slot_mapping), 32);
        assert_eq!(offset_of!(LaunchArgsQkv, key_cache_ptrs), 40);
        assert_eq!(offset_of!(LaunchArgsQkv, value_cache_ptrs), 48);
        assert_eq!(offset_of!(LaunchArgsQkv, barriers), 56);
        assert_eq!(offset_of!(LaunchArgsQkv, trace_level), 64);
    }

    // Attention pool ABI: ten pointers, each 8 bytes on 64-bit
    // targets, aligned to 8. The QKV prefix (act_ptrs,
    // weight_ptrs, input_ids, positions, slot_mapping,
    // key_cache_ptrs, value_cache_ptrs) is followed by the two
    // attention-specific tail args (seq_lens, block_table), then
    // `barriers` (Phase 3 step 6), matching the emitted kernel
    // signature.
    #[test]
    fn launch_args_attn_abi_size() {
        // 10 pointers × 8 bytes = 80, + 4-byte `trace_level` + 4 pad
        // (Phase 3 step 8) = 88.
        assert_eq!(std::mem::size_of::<LaunchArgsAttn>(), 88);
        assert_eq!(std::mem::align_of::<LaunchArgsAttn>(), 8);
    }

    // Field offsets: QKV prefix occupies the same bytes as
    // `LaunchArgsQkv`, with `seq_lens` and `block_table` appended.
    // If any offset drifts, calls through `launch_attn` will pass
    // garbage — keep this tight.
    #[test]
    fn launch_args_attn_field_offsets() {
        use std::mem::offset_of;
        assert_eq!(offset_of!(LaunchArgsAttn, act_ptrs), 0);
        assert_eq!(offset_of!(LaunchArgsAttn, weight_ptrs), 8);
        assert_eq!(offset_of!(LaunchArgsAttn, input_ids), 16);
        assert_eq!(offset_of!(LaunchArgsAttn, positions), 24);
        assert_eq!(offset_of!(LaunchArgsAttn, slot_mapping), 32);
        assert_eq!(offset_of!(LaunchArgsAttn, key_cache_ptrs), 40);
        assert_eq!(offset_of!(LaunchArgsAttn, value_cache_ptrs), 48);
        assert_eq!(offset_of!(LaunchArgsAttn, seq_lens), 56);
        assert_eq!(offset_of!(LaunchArgsAttn, block_table), 64);
        // Wave F: `block_table_stride: u32` inserted between
        // `block_table` and `barriers`. 4-byte field at 72 with a
        // 4-byte tail pad before the 8-byte-aligned `barriers`.
        assert_eq!(offset_of!(LaunchArgsAttn, block_table_stride), 72);
        assert_eq!(offset_of!(LaunchArgsAttn, barriers), 80);
        assert_eq!(offset_of!(LaunchArgsAttn, trace_level), 88);
    }

    // The QKV prefix must be layout-compatible with `LaunchArgsQkv`
    // — same fields in the same positions, so an `&LaunchArgsAttn`
    // can reinterpret its prefix as `&LaunchArgsQkv` without
    // shifting. This guards against a future field reorder
    // sneaking in and silently diverging the two shapes.
    #[test]
    fn launch_args_attn_prefix_matches_qkv() {
        use std::mem::offset_of;
        assert_eq!(
            offset_of!(LaunchArgsAttn, act_ptrs),
            offset_of!(LaunchArgsQkv, act_ptrs),
        );
        assert_eq!(
            offset_of!(LaunchArgsAttn, weight_ptrs),
            offset_of!(LaunchArgsQkv, weight_ptrs),
        );
        assert_eq!(
            offset_of!(LaunchArgsAttn, input_ids),
            offset_of!(LaunchArgsQkv, input_ids),
        );
        assert_eq!(
            offset_of!(LaunchArgsAttn, positions),
            offset_of!(LaunchArgsQkv, positions),
        );
        assert_eq!(
            offset_of!(LaunchArgsAttn, slot_mapping),
            offset_of!(LaunchArgsQkv, slot_mapping),
        );
        assert_eq!(
            offset_of!(LaunchArgsAttn, key_cache_ptrs),
            offset_of!(LaunchArgsQkv, key_cache_ptrs),
        );
        assert_eq!(
            offset_of!(LaunchArgsAttn, value_cache_ptrs),
            offset_of!(LaunchArgsQkv, value_cache_ptrs),
        );
    }

    // `LaunchFnAny::tier` reports the variant the wrapper was
    // constructed with. Guards against a future reorder of the enum
    // arms silently shifting the discriminant→tier mapping.
    #[test]
    fn launch_fn_any_tier() {
        unsafe extern "C" fn stub_base(
            _a: ActPtrs,
            _w: WeightPtrs,
            _b: I32MutPtr,
            _tl: i32,
            _s: *mut c_void,
        ) -> i32 {
            0
        }
        unsafe extern "C" fn stub_qkv(
            _a: ActPtrs,
            _w: WeightPtrs,
            _iid: U32Ptr,
            _p: U32Ptr,
            _sm: I64Ptr,
            _kc: KvPtrs,
            _vc: KvPtrs,
            _b: I32MutPtr,
            _tl: i32,
            _s: *mut c_void,
        ) -> i32 {
            0
        }
        unsafe extern "C" fn stub_attn(
            _a: ActPtrs,
            _w: WeightPtrs,
            _iid: U32Ptr,
            _p: U32Ptr,
            _sm: I64Ptr,
            _kc: KvPtrs,
            _vc: KvPtrs,
            _sl: I32Ptr,
            _bt: U32Ptr,
            _b: I32MutPtr,
            _tl: i32,
            _s: *mut c_void,
        ) -> i32 {
            0
        }
        assert_eq!(LaunchFnAny::Base(stub_base).tier(), LaunchTier::Base);
        assert_eq!(LaunchFnAny::Qkv(stub_qkv).tier(), LaunchTier::Qkv);
        assert_eq!(LaunchFnAny::Attn(stub_attn).tier(), LaunchTier::Attn);
        // MultiStep tier
        unsafe extern "C" fn stub_ms(
            _a: ActPtrs, _b: WeightPtrs, _c: *mut u32, _d: U32Ptr, _e: I64Ptr,
            _f: KvPtrs, _g: KvPtrs, _h: I32Ptr, _i: U32Ptr, _j: u32,
            _k: I32MutPtr, _l: i32, _m: i32, _n: *mut u32, _s: *mut std::ffi::c_void,
        ) -> i32 { 0 }
        assert_eq!(LaunchFnAny::MultiStep(stub_ms).tier(), LaunchTier::MultiStep);
    }

    // `dispatch_launch` calls the fn pointer matching the tier of
    // the wrapper it was given — not any other. Uses an atomic tag
    // each stub writes on entry so we can read back which one ran.
    #[test]
    fn dispatch_launch_picks_tier() {
        use std::sync::atomic::{AtomicU8, Ordering};
        static TAG: AtomicU8 = AtomicU8::new(0);

        unsafe extern "C" fn stub_base(
            _a: ActPtrs,
            _w: WeightPtrs,
            _b: I32MutPtr,
            _tl: i32,
            _s: *mut c_void,
        ) -> i32 {
            TAG.store(1, Ordering::SeqCst);
            0
        }
        unsafe extern "C" fn stub_qkv(
            _a: ActPtrs,
            _w: WeightPtrs,
            _iid: U32Ptr,
            _p: U32Ptr,
            _sm: I64Ptr,
            _kc: KvPtrs,
            _vc: KvPtrs,
            _b: I32MutPtr,
            _tl: i32,
            _s: *mut c_void,
        ) -> i32 {
            TAG.store(2, Ordering::SeqCst);
            0
        }
        unsafe extern "C" fn stub_attn(
            _a: ActPtrs,
            _w: WeightPtrs,
            _iid: U32Ptr,
            _p: U32Ptr,
            _sm: I64Ptr,
            _kc: KvPtrs,
            _vc: KvPtrs,
            _sl: I32Ptr,
            _bt: U32Ptr,
            _bts: u32,
            _b: I32MutPtr,
            _tl: i32,
            _s: *mut c_void,
        ) -> i32 {
            TAG.store(3, Ordering::SeqCst);
            0
        }

        // Fat args with null pointers — the stubs never dereference,
        // and lower tiers drop the tail fields before invoking their
        // fn pointer, so nulls here are safe for this dispatch test.
        let args = LaunchArgsAttn {
            act_ptrs: std::ptr::null(),
            weight_ptrs: std::ptr::null(),
            input_ids: std::ptr::null(),
            positions: std::ptr::null(),
            slot_mapping: std::ptr::null(),
            key_cache_ptrs: std::ptr::null(),
            value_cache_ptrs: std::ptr::null(),
            seq_lens: std::ptr::null(),
            block_table: std::ptr::null(),
            block_table_stride: 0,
            barriers: std::ptr::null_mut(),
            trace_level: 0,
        };

        TAG.store(0, Ordering::SeqCst);
        unsafe {
            dispatch_launch(LaunchFnAny::Base(stub_base), args, std::ptr::null_mut()).unwrap();
        }
        assert_eq!(TAG.load(Ordering::SeqCst), 1);

        TAG.store(0, Ordering::SeqCst);
        unsafe {
            dispatch_launch(LaunchFnAny::Qkv(stub_qkv), args, std::ptr::null_mut()).unwrap();
        }
        assert_eq!(TAG.load(Ordering::SeqCst), 2);

        TAG.store(0, Ordering::SeqCst);
        unsafe {
            dispatch_launch(LaunchFnAny::Attn(stub_attn), args, std::ptr::null_mut()).unwrap();
        }
        assert_eq!(TAG.load(Ordering::SeqCst), 3);
    }

    // `dispatch_launch` propagates non-zero return codes from the
    // dispatched fn as `Err(rc)`. Catches a future refactor that
    // drops the rc check in the `Attn` arm (which is the only one
    // not going through a standalone `launch_*` wrapper today).
    #[test]
    fn dispatch_launch_propagates_error() {
        unsafe extern "C" fn stub_attn_err(
            _a: ActPtrs,
            _w: WeightPtrs,
            _iid: U32Ptr,
            _p: U32Ptr,
            _sm: I64Ptr,
            _kc: KvPtrs,
            _vc: KvPtrs,
            _sl: I32Ptr,
            _bt: U32Ptr,
            _bts: u32,
            _b: I32MutPtr,
            _tl: i32,
            _s: *mut c_void,
        ) -> i32 {
            42
        }
        let args = LaunchArgsAttn {
            act_ptrs: std::ptr::null(),
            weight_ptrs: std::ptr::null(),
            input_ids: std::ptr::null(),
            positions: std::ptr::null(),
            slot_mapping: std::ptr::null(),
            key_cache_ptrs: std::ptr::null(),
            value_cache_ptrs: std::ptr::null(),
            seq_lens: std::ptr::null(),
            block_table: std::ptr::null(),
            block_table_stride: 0,
            barriers: std::ptr::null_mut(),
            trace_level: 0,
        };
        let rc = unsafe {
            dispatch_launch(LaunchFnAny::Attn(stub_attn_err), args, std::ptr::null_mut())
        };
        assert_eq!(rc, Err(42));
    }

    // `seq_lens_ptr` / `block_table_ptr` share a test scaffold: the
    // host side stages a `DType::I32` device tensor; the projection
    // is a zero-copy pointer reinterpret. We don't need a real CUDA
    // allocation — the pointer is never dereferenced on the host
    // side — so the tests build a `GpuTensor` around a fake "device"
    // pointer and assert the raw bytes round-trip unchanged through
    // `as_ptr::<i32>()` / `as_ptr::<u32>()`.

    // Build a `TensorView<'static>` around a fake device pointer.
    // The resulting view must never be handed to a kernel; it exists
    // only to exercise the host-side pointer projections. The
    // lifetime is faked via `TensorView::from_raw`, which takes an
    // owned `GpuTensor` and attaches whatever lifetime the caller
    // binds — `'static` here because we never feed it to a real
    // kernel.
    unsafe fn fake_view_i32(ptr: *mut u8, shape: &[usize]) -> TensorView<'static> {
        use ferrite_cuda_core::dtype::DType;
        use ferrite_cuda_core::tensor::GpuTensor;
        let t = unsafe { GpuTensor::new(ptr, shape, DType::I32) };
        unsafe { TensorView::from_raw(t) }
    }

    // `seq_lens_ptr` surfaces the view's underlying device pointer
    // as an `I32Ptr` with no adjustment. A drift in signedness or an
    // accidental offset would change the bit pattern — this test
    // catches that with a known-nonzero pointer value.
    #[test]
    fn seq_lens_ptr_zero_copy() {
        let ptr = 0xDEAD_BEEF_1000_usize as *mut u8;
        let view = unsafe { fake_view_i32(ptr, &[16]) };
        let i32_ptr: I32Ptr = seq_lens_ptr(view);
        assert_eq!(i32_ptr as usize, ptr as usize);
    }

    // `block_table_ptr` is the signed→unsigned reinterpret arm. The
    // host-side tensor carries `DType::I32` but the mega kernel reads
    // it as `const uint32_t*` — same bit width, and page indices are
    // non-negative so the bit patterns agree. This test proves the
    // Rust side doesn't insert any conversion beyond the pointer
    // reinterpret.
    #[test]
    fn block_table_ptr_zero_copy() {
        let ptr = 0xFEED_FACE_2000_usize as *mut u8;
        let view = unsafe { fake_view_i32(ptr, &[16, 8]) };
        let u32_ptr: U32Ptr = block_table_ptr(view);
        assert_eq!(u32_ptr as usize, ptr as usize);
    }

    // A single fake tensor projected both ways yields pointers that
    // differ only in their typed view of the same device bytes.
    // Guards against a future refactor that routes one projection
    // through a stage-and-copy path while the other stays
    // zero-copy — the two tail fields of `LaunchArgsAttn` must share
    // the single-tensor-per-field assumption the macro relies on.
    #[test]
    fn seq_lens_and_block_table_share_underlying_ptr() {
        let ptr = 0x0123_4567_89AB_usize as *mut u8;
        let view = unsafe { fake_view_i32(ptr, &[16]) };
        let i32_ptr = seq_lens_ptr(view);
        let u32_ptr = block_table_ptr(view);
        assert_eq!(i32_ptr as usize, u32_ptr as usize);
        assert_eq!(i32_ptr as usize, ptr as usize);
    }

    // Build a `TensorView<'static>` around a fake device pointer
    // with `DType::U32` — the host staging dtype for `positions`.
    // Same contract as `fake_view_i32`: unsafe, never hand to a
    // kernel, exists only for the host-side pointer projection
    // tests.
    unsafe fn fake_view_u32(ptr: *mut u8, shape: &[usize]) -> TensorView<'static> {
        use ferrite_cuda_core::dtype::DType;
        use ferrite_cuda_core::tensor::GpuTensor;
        let t = unsafe { GpuTensor::new(ptr, shape, DType::U32) };
        unsafe { TensorView::from_raw(t) }
    }

    // Build a `TensorView<'static>` around a fake device pointer
    // with `DType::I64` — the host staging dtype for `slot_mapping`
    // (see `vllm-executor::cuda_worker::build_attention_tensors`).
    unsafe fn fake_view_i64(ptr: *mut u8, shape: &[usize]) -> TensorView<'static> {
        use ferrite_cuda_core::dtype::DType;
        use ferrite_cuda_core::tensor::GpuTensor;
        let t = unsafe { GpuTensor::new(ptr, shape, DType::I64) };
        unsafe { TensorView::from_raw(t) }
    }

    // `positions_ptr` surfaces the view's underlying device pointer
    // as a `U32Ptr` with no adjustment. Host dtype is `DType::U32`,
    // mega kernel reads `const uint32_t*` — same bit width, zero-
    // copy. A drift into a signed reinterpret or offset would
    // change the bit pattern and produce garbage rope indices.
    #[test]
    fn positions_ptr_zero_copy() {
        let ptr = 0xCAFE_BABE_3000_usize as *mut u8;
        let view = unsafe { fake_view_u32(ptr, &[16]) };
        let u32_ptr: U32Ptr = positions_ptr(view);
        assert_eq!(u32_ptr as usize, ptr as usize);
    }

    // `input_ids_ptr` is the analogue of `positions_ptr` — host dtype
    // `DType::U32`, mega kernel reads `const uint32_t*`, zero-copy
    // reinterpret. Guards against a future drift into a staging path.
    #[test]
    fn input_ids_ptr_zero_copy() {
        let ptr = 0xABCD_1234_5000_usize as *mut u8;
        let view = unsafe { fake_view_u32(ptr, &[16]) };
        let u32_ptr: U32Ptr = input_ids_ptr(view);
        assert_eq!(u32_ptr as usize, ptr as usize);
    }

    // `slot_mapping_ptr` is the dtype-match arm: host stages
    // `DType::I64`, mega reads `const int64_t*`. Pointer
    // reinterpret must preserve bits exactly — a drift to `i32`
    // or `u32` would truncate upper bits of the 64-bit block
    // slot IDs on a pool with >2³¹ slots.
    #[test]
    fn slot_mapping_ptr_zero_copy() {
        let ptr = 0xBADD_F00D_4000_usize as *mut u8;
        let view = unsafe { fake_view_i64(ptr, &[16]) };
        let i64_ptr: I64Ptr = slot_mapping_ptr(view);
        assert_eq!(i64_ptr as usize, ptr as usize);
    }

    // `positions_ptr` and `slot_mapping_ptr` are the two fields a
    // caller sources from separate backing tensors (unlike seq_lens
    // / block_table, which *can* share a view in a degenerate test
    // but never in practice). Sanity-check that they're independent
    // — projecting one tensor into one accessor must not carry over
    // to the other.
    #[test]
    fn positions_and_slot_mapping_are_independent() {
        let pos_ptr = 0x1111_1111_1000_usize as *mut u8;
        let sm_ptr = 0x2222_2222_2000_usize as *mut u8;
        let pos_view = unsafe { fake_view_u32(pos_ptr, &[16]) };
        let sm_view = unsafe { fake_view_i64(sm_ptr, &[16]) };
        let got_pos = positions_ptr(pos_view);
        let got_sm = slot_mapping_ptr(sm_view);
        assert_eq!(got_pos as usize, pos_ptr as usize);
        assert_eq!(got_sm as usize, sm_ptr as usize);
        assert_ne!(got_pos as usize, got_sm as usize);
    }
}
