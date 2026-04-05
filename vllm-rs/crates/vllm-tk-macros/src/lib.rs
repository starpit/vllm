// SPDX-License-Identifier: Apache-2.0
//! `megakernel!` proc-macro for compile-time verified GPU megakernels.
//!
//! This proc-macro parses a high-level model DAG description, verifies
//! all memory safety properties at compile time (buffer dimensions, shared
//! memory budget, barrier counts, aliasing), and generates:
//!
//! 1. A static CUDA megakernel (no runtime dispatch, no VM)
//! 2. Rust FFI bindings and a type-safe launch wrapper
//! 3. An ASCII pipeline diagram (accessible as a const)
//!
//! If it compiles, the kernel is memory-safe by construction.
//!
//! Core logic lives in `vllm-tk-macros-core` so build.rs can reuse it.

extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use vllm_tk_macros_core::{cuda_codegen, diagram, parse, verify};

/// The `megakernel!` macro: compile-time verified GPU megakernel generation.
///
/// # Example
/// ```ignore
/// megakernel! {
///     kernel llama_sm89<NL=32, HD=2048, ID=5632, HDM=64, NAH=32, NKH=8, VS=128256> {
///         for layer in 0..NL {
///             let normed = rmsnorm(hidden_states, attn_norm[layer]);
///             let qkv = gemm(normed, qkv_weights[layer]);
///             let (q, k, v) = rope_append(qkv, positions, kv_cache[layer]);
///             let attn = attention_decode(q, k, v, kv_cache[layer], block_table);
///             hidden_states = gemm_add(attn, o_proj[layer], hidden_states);
///
///             let normed2 = rmsnorm(hidden_states, mlp_norm[layer]);
///             let gate = silu(gemm(normed2, gate_weights[layer]));
///             let up = gemm(normed2, up_weights[layer]);
///             hidden_states = gemm_add(gate * up, down_proj[layer], hidden_states);
///         }
///         let normed = rmsnorm(hidden_states, lm_head_norm);
///         logits = gemm(normed, lm_head);
///     }
/// }
/// ```
///
/// # Compile-Time Checks
/// - Buffer dimension matching across all producer/consumer edges
/// - GEMM K-dimension agreement (A cols == B cols for A @ B^T)
/// - RopeAppend QKV dimension = (NAH + 2*NKH) * HDM
/// - AttentionDecode q/output shape agreement
/// - Barrier chain completeness (no miswired ops)
/// - Tile divisibility (dimensions compatible with TK tile sizes)
/// - Shared memory fits within sm89 hardware budget
/// - No uninitialized buffer reads
/// - No buffer aliasing (concurrent writes)
/// - No dead (unused) buffers
#[proc_macro]
pub fn megakernel(input: TokenStream) -> TokenStream {
    let input2: proc_macro2::TokenStream = input.into();

    // Phase 1: Parse the DSL
    let def: parse::MegakernelDef = match syn::parse2(input2) {
        Ok(d) => d,
        Err(e) => return e.to_compile_error().into(),
    };

    // Phase 2: Build the typed DAG
    let dag = match parse::build_dag(&def) {
        Ok(d) => d,
        Err(e) => {
            return syn::Error::new(proc_macro2::Span::call_site(), e)
                .to_compile_error()
                .into();
        }
    };

    // Phase 3: Verify all safety properties
    let errors = verify::verify(&dag);
    if !errors.is_empty() {
        let mut msg = String::from("megakernel! verification failed:\n");
        for e in &errors {
            msg.push_str(&format!("  {e}\n"));
        }
        return syn::Error::new(proc_macro2::Span::call_site(), msg)
            .to_compile_error()
            .into();
    }

    // Phase 4: Generate the pipeline diagram
    let diagram_str = diagram::render_diagram(&dag);

    // Phase 5: Generate CUDA kernel source
    let cuda_source = cuda_codegen::generate_static_kernel(&dag);

    // Phase 6: Generate Rust code with const-generic typed launch API
    let kernel_name = &def.name;
    let struct_name = syn::Ident::new(
        &format!("Megakernel{}", to_pascal_case(&kernel_name.to_string())),
        kernel_name.span(),
    );
    let num_ops = dag.ops.len();
    let num_buffers = dag.buffers.len();

    // Extract model dimensions as concrete values for const generics
    let hd = dag.params.get("HD").copied().unwrap_or(2048);
    let id = dag.params.get("ID").copied().unwrap_or(5632);
    let hdm = dag.params.get("HDM").copied().unwrap_or(64);
    let nah = dag.params.get("NAH").copied().unwrap_or(32);
    let nkh = dag.params.get("NKH").copied().unwrap_or(8);
    let vs = dag.params.get("VS").copied().unwrap_or(128256);
    let nl = dag.params.get("NL").copied().unwrap_or(32);
    let qkv_dim = (nah + 2 * nkh) * hdm;

    let expanded = quote! {
        // ── Newtype wrappers for launch parameters ──
        // Prevent mixing up num_seqs / num_tokens / batch_size (all i32).

        /// Number of sequences (prefill). NOT total tokens.
        #[derive(Debug, Clone, Copy)]
        pub struct NumSeqs(pub i32);

        /// Total number of tokens across all sequences.
        #[derive(Debug, Clone, Copy)]
        pub struct NumTokens(pub i32);

        /// Decode batch size (number of single-token sequences).
        #[derive(Debug, Clone, Copy)]
        pub struct DecodeBatchSize(pub i32);

        // ── Typed GPU buffer handles ──
        // Each category is a distinct type with const-generic dims.
        // NonNull enforces non-null at construction. Const generics enforce
        // dimension matching at compile time.

        /// GPU activation buffer: [batch, D]. Batch is runtime, D is compile-time.
        #[repr(transparent)]
        pub struct GpuActivation<const D: usize>(core::ptr::NonNull<u8>);

        /// GPU activation buffer with intermediate_dim: [batch, D].
        #[repr(transparent)]
        pub struct GpuActivationBig<const D: usize>(core::ptr::NonNull<u8>);

        /// GPU logits buffer: [batch, VS]. VS is compile-time.
        #[repr(transparent)]
        pub struct GpuLogits<const VS: usize>(core::ptr::NonNull<u8>);

        /// GPU weight matrix: [ROWS, COLS] (both compile-time).
        /// Used for qkv, o_proj, up, gate, lm_head weights.
        #[repr(transparent)]
        pub struct GpuWeight<const ROWS: usize, const COLS: usize>(core::ptr::NonNull<u8>);

        /// GPU weight matrix with swapped convention (hidden→intermediate):
        /// [ROWS, COLS] where COLS is intermediate_dim.
        #[repr(transparent)]
        pub struct GpuWeightBig<const ROWS: usize, const COLS: usize>(core::ptr::NonNull<u8>);

        /// GPU 1D norm weight: [NL, D] or [1, D].
        #[repr(transparent)]
        pub struct GpuNormWeight<const D: usize>(core::ptr::NonNull<u8>);

        /// GPU paged KV cache (opaque — dims are runtime).
        #[repr(transparent)]
        pub struct GpuKvCache(core::ptr::NonNull<u8>);

        /// GPU RoPE table: [max_pos, HDM].
        #[repr(transparent)]
        pub struct GpuRopeTable<const HDM: usize>(core::ptr::NonNull<u8>);

        /// GPU i32 metadata vector (runtime-length CSR arrays, position IDs, etc).
        #[repr(transparent)]
        pub struct GpuMetaVec(core::ptr::NonNull<u8>);

        /// GPU barrier tensor (runtime dims, opaque).
        #[repr(transparent)]
        pub struct GpuBarrier(core::ptr::NonNull<u8>);

        /// GPU instruction/timing layout (runtime dims, opaque).
        #[repr(transparent)]
        pub struct GpuVmLayout(core::ptr::NonNull<u8>);

        // Implement construction + pointer extraction for each handle type.
        macro_rules! impl_gpu_handle {
            ($ty:ident $(< $(const $g:ident : usize),+ >)?) => {
                impl$(<$(const $g: usize),+>)? $ty$(<$($g),+>)? {
                    /// Create from a raw GPU pointer.
                    ///
                    /// # Safety
                    /// `ptr` must point to valid, correctly-sized GPU memory.
                    pub unsafe fn from_raw(ptr: *mut u8) -> Self {
                        Self(core::ptr::NonNull::new(ptr)
                            .expect(concat!(stringify!($ty), ": null GPU pointer")))
                    }
                    /// Raw pointer (for FFI).
                    pub fn as_ptr(&self) -> *mut u8 { self.0.as_ptr() }
                    /// Pointer as u64 (for TkTensorArg).
                    pub fn ptr_u64(&self) -> u64 { self.0.as_ptr() as u64 }
                }
            }
        }

        impl_gpu_handle!(GpuActivation<const D: usize>);
        impl_gpu_handle!(GpuActivationBig<const D: usize>);
        impl_gpu_handle!(GpuLogits<const VS: usize>);
        impl_gpu_handle!(GpuWeight<const ROWS: usize, const COLS: usize>);
        impl_gpu_handle!(GpuWeightBig<const ROWS: usize, const COLS: usize>);
        impl_gpu_handle!(GpuNormWeight<const D: usize>);
        impl_gpu_handle!(GpuKvCache);
        impl_gpu_handle!(GpuRopeTable<const HDM: usize>);
        impl_gpu_handle!(GpuMetaVec);
        impl_gpu_handle!(GpuBarrier);
        impl_gpu_handle!(GpuVmLayout);

        /// Auto-generated megakernel. All safety properties verified at compile time.
        /// Buffer shapes are enforced via const generics in the launch signature.
        pub struct #struct_name;

        /// Flat tensor descriptor for FFI. Matches the C-side TkTensorArg.
        /// This is an internal type — users construct typed handles, and
        /// launch methods convert to TkTensorArg automatically.
        #[repr(C)]
        #[derive(Clone, Copy, Debug)]
        pub struct TkTensorArg {
            pub ptr: u64,
            pub b: i32,
            pub d: i32,
            pub r: i32,
            pub c: i32,
        }

        impl TkTensorArg {
            pub fn new(ptr: u64, shape: &[usize]) -> Self {
                let (b, d, r, c) = match shape.len() {
                    1 => (1, 1, 1, shape[0] as i32),
                    2 => (1, 1, shape[0] as i32, shape[1] as i32),
                    3 => (1, shape[0] as i32, shape[1] as i32, shape[2] as i32),
                    4 => (shape[0] as i32, shape[1] as i32, shape[2] as i32, shape[3] as i32),
                    _ => panic!("TkTensorArg: expected 1-4D shape, got {}D", shape.len()),
                };
                Self { ptr, b, d, r, c }
            }
            pub fn null() -> Self { Self { ptr: 0, b: 0, d: 0, r: 0, c: 0 } }
        }

        /// All buffers needed to launch the static megakernel.
        ///
        /// Every field uses a typed handle with compile-time dimension checks.
        /// Constructing this struct requires correctly-typed GPU buffers —
        /// shape mismatches and null pointers are compile/runtime errors.
        ///
        /// The launch methods convert these typed handles to flat FFI args
        /// with shapes baked in by the proc-macro.
        pub struct LaunchArgs {
            // ── VM state ──
            pub barrier: GpuBarrier,
            pub instructions: GpuVmLayout,
            pub timings: GpuVmLayout,

            // ── Weights (const-generic dims enforce model architecture match) ──
            pub qkv_weights: GpuWeight<{#qkv_dim}, {#hd}>,
            pub attn_norm: GpuNormWeight<{#hd}>,
            pub o_proj: GpuWeight<{#hd}, {#hd}>,
            pub mlp_norm: GpuNormWeight<{#hd}>,
            pub up_weights: GpuWeight<{#id}, {#hd}>,
            pub gate_weights: GpuWeight<{#id}, {#hd}>,
            pub down_proj: GpuWeightBig<{#hd}, {#id}>,
            pub lm_head_norm: GpuNormWeight<{#hd}>,
            pub lm_head: GpuWeight<{#vs}, {#hd}>,

            // ── KV cache ──
            pub k_cache: GpuKvCache,
            pub v_cache: GpuKvCache,

            // ── RoPE tables ──
            pub rope_cos: GpuRopeTable<{#hdm}>,
            pub rope_sin: GpuRopeTable<{#hdm}>,

            // ── Activation buffers (feature dim is compile-time) ──
            pub hidden_states: GpuActivation<{#hd}>,
            pub rms_rope: GpuActivation<{#hd}>,
            pub rms_gate: GpuActivation<{#hd}>,
            pub q_post_rope: GpuActivation<{#hd}>,
            pub attn_out: GpuActivation<{#hd}>,
            pub silu_out: GpuActivationBig<{#id}>,
            pub rms_lm: GpuActivation<{#hd}>,
            pub logits: GpuLogits<{#vs}>,

            // ── Paged KV metadata (decode) ──
            pub position_ids: GpuMetaVec,
            pub kv_indptr: GpuMetaVec,
            pub kv_indices: GpuMetaVec,
            pub kv_last_page: GpuMetaVec,
            pub kv_append: GpuMetaVec,

            // ── Paged KV metadata (prefill) ──
            pub prefill_qo_indptr: GpuMetaVec,
            pub prefill_kv_indptr: GpuMetaVec,
            pub prefill_kv_indices: GpuMetaVec,
            pub prefill_kv_last_page_len: GpuMetaVec,

            // ── Scalars ──
            pub attn_scale: f32,
            pub rms_norm_eps: f32,
            pub num_pages: i32,

            // ── Runtime model dimension ──
            pub num_layers: usize,

            // ── Prefill metadata sizes (set to 0 for decode-only) ──
            pub prefill_num_seqs: usize,
            pub prefill_num_kv_pages: usize,
        }

        impl #struct_name {
            // ── Model dimension constants ──
            pub const NL: usize = #nl;
            pub const HD: usize = #hd;
            pub const ID: usize = #id;
            pub const HDM: usize = #hdm;
            pub const NAH: usize = #nah;
            pub const NKH: usize = #nkh;
            pub const VS: usize = #vs;
            pub const QKV_DIM: usize = #qkv_dim;

            pub const NUM_OPS: usize = #num_ops;
            pub const NUM_BUFFERS: usize = #num_buffers;
            pub const PIPELINE_DIAGRAM: &str = #diagram_str;
            pub const CUDA_SOURCE: &str = #cuda_source;

            /// Convert typed LaunchArgs to flat TkTensorArg FFI args.
            ///
            /// Shapes are baked in from the proc-macro — no hand-written shape arrays.
            /// The `batch_rows` param sets the runtime row dim for activations/metadata.
            /// The `barrier_shape`, `inst_shape`, and `timing_shape` params set VM state tensor shapes.
            #[allow(clippy::possible_missing_comma)]
            fn args_to_ffi(
                args: &LaunchArgs,
                batch_rows: usize,
                num_layers: usize,
                barrier_shape: [usize; 4],
                inst_shape: [usize; 4],
                timing_shape: [usize; 4],
                dims: crate::KernelDims,
            ) -> [TkTensorArg; 33] {
                let hd = dims.hd;
                let id = dims.id;
                let hdm = dims.hdm;
                let nkh = dims.nkh;
                let vs = dims.vs;
                let qkv_dim = dims.qkv_dim();
                [
                    // VM state
                    TkTensorArg::new(args.barrier.ptr_u64(), &barrier_shape),
                    TkTensorArg::new(args.instructions.ptr_u64(), &inst_shape),
                    TkTensorArg::new(args.timings.ptr_u64(), &timing_shape),
                    // Weights — shapes use runtime dims from variant
                    TkTensorArg::new(args.qkv_weights.ptr_u64(), &[num_layers * qkv_dim, hd]),
                    TkTensorArg::new(args.attn_norm.ptr_u64(), &[num_layers, hd]),
                    TkTensorArg::new(args.o_proj.ptr_u64(), &[num_layers * hd, hd]),
                    TkTensorArg::new(args.mlp_norm.ptr_u64(), &[num_layers, hd]),
                    TkTensorArg::new(args.up_weights.ptr_u64(), &[num_layers * id, hd]),
                    TkTensorArg::new(args.gate_weights.ptr_u64(), &[num_layers * id, hd]),
                    TkTensorArg::new(args.down_proj.ptr_u64(), &[num_layers * hd, id]),
                    TkTensorArg::new(args.lm_head_norm.ptr_u64(), &[1, hd]),
                    TkTensorArg::new(args.lm_head.ptr_u64(), &[vs, hd]),
                    // KV cache
                    TkTensorArg::new(args.k_cache.ptr_u64(), &[args.num_pages as usize, 1, nkh, hdm]),
                    TkTensorArg::new(args.v_cache.ptr_u64(), &[args.num_pages as usize, 1, nkh, hdm]),
                    // RoPE
                    TkTensorArg::new(args.rope_cos.ptr_u64(), &[4096, hdm]),
                    TkTensorArg::new(args.rope_sin.ptr_u64(), &[4096, hdm]),
                    // Activations — batch dim is runtime
                    TkTensorArg::new(args.hidden_states.ptr_u64(), &[1, 1, batch_rows, hd]),
                    TkTensorArg::new(args.rms_rope.ptr_u64(), &[1, 1, batch_rows, hd]),
                    TkTensorArg::new(args.rms_gate.ptr_u64(), &[1, 1, batch_rows, hd]),
                    TkTensorArg::new(args.q_post_rope.ptr_u64(), &[1, 1, batch_rows, hd]),
                    TkTensorArg::new(args.attn_out.ptr_u64(), &[1, 1, batch_rows, hd]),
                    TkTensorArg::new(args.silu_out.ptr_u64(), &[1, 1, batch_rows, id]),
                    TkTensorArg::new(args.rms_lm.ptr_u64(), &[1, 1, batch_rows, hd]),
                    TkTensorArg::new(args.logits.ptr_u64(), &[1, 1, batch_rows, vs]),
                    // Decode KV metadata
                    TkTensorArg::new(args.position_ids.ptr_u64(), &[batch_rows]),
                    TkTensorArg::new(args.kv_indptr.ptr_u64(), &[batch_rows + 1]),
                    TkTensorArg::new(args.kv_indices.ptr_u64(), &[args.num_pages as usize]),
                    TkTensorArg::new(args.kv_last_page.ptr_u64(), &[batch_rows]),
                    TkTensorArg::new(args.kv_append.ptr_u64(), &[batch_rows]),
                    // Prefill KV metadata — sizes from LaunchArgs (1 for decode dummy)
                    TkTensorArg::new(args.prefill_qo_indptr.ptr_u64(), &[if args.prefill_num_seqs > 0 { args.prefill_num_seqs + 1 } else { 1 }]),
                    TkTensorArg::new(args.prefill_kv_indptr.ptr_u64(), &[if args.prefill_num_seqs > 0 { args.prefill_num_seqs + 1 } else { 1 }]),
                    TkTensorArg::new(args.prefill_kv_indices.ptr_u64(), &[if args.prefill_num_kv_pages > 0 { args.prefill_num_kv_pages } else { 1 }]),
                    TkTensorArg::new(args.prefill_kv_last_page_len.ptr_u64(), &[if args.prefill_num_seqs > 0 { args.prefill_num_seqs } else { 1 }]),
                ]
            }

            /// Launch the decode megakernel (1 token per sequence).
            ///
            /// All weight shapes are verified at compile time through the
            /// typed handles in `LaunchArgs`. Only `batch_size` and VM state
            /// shapes are runtime.
            ///
            /// # Safety
            /// All pointers in `args` must point to valid GPU memory.
            /// `batch_size` must be in [1, 128].
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn launch_decode(
                args: &LaunchArgs,
                variant: &crate::KernelVariant,
                batch_size: DecodeBatchSize,
                num_prefill_tokens: NumTokens,
                barrier_shape: [usize; 4],
                inst_shape: [usize; 4],
                timing_shape: [usize; 4],
                stream: u64,
            ) -> i32 {
                let bs = batch_size.0;
                let npt = num_prefill_tokens.0;
                debug_assert!(bs > 0, "batch_size must be > 0");
                debug_assert!(bs <= 128, "batch_size must be <= 128 (decode)");

                let batch_rows = (bs as usize).next_multiple_of(128);
                let nl = args.num_layers;
                let dims = variant.dims();
                let ffi = Self::args_to_ffi(args, batch_rows, nl, barrier_shape, inst_shape, timing_shape, dims);

                #[cfg(feature = "cuda")]
                {
                    variant.launch_decode(
                        &ffi,
                        args.attn_scale, args.rms_norm_eps,
                        args.num_pages, bs, npt, nl as i32,
                        stream,
                    )
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = (args, variant, bs, npt, barrier_shape, inst_shape, timing_shape, stream, ffi, nl, dims);
                    panic!("launch_decode requires --features cuda");
                }
            }

            /// Launch the prefill static megakernel.
            ///
            /// `num_seqs`: number of prefill sequences (NOT total tokens).
            /// `num_prefill_tokens`: total number of tokens across all sequences.
            /// These are distinct: num_seqs <= num_prefill_tokens.
            ///
            /// # Safety
            /// All pointers in `args` must point to valid GPU memory.
            /// `seq_chunk_lens` and `seq_extend_offsets` must have at least
            /// `num_seqs` elements.
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn launch_prefill(
                args: &LaunchArgs,
                variant: &crate::KernelVariant,
                num_seqs: NumSeqs,
                num_prefill_tokens: NumTokens,
                seq_chunk_lens: &GpuMetaVec,
                seq_extend_offsets: &GpuMetaVec,
                barrier_shape: [usize; 4],
                inst_shape: [usize; 4],
                timing_shape: [usize; 4],
                stream: u64,
            ) -> i32 {
                let ns = num_seqs.0;
                let npt = num_prefill_tokens.0;
                assert!(npt > 0, "num_prefill_tokens must be > 0");
                assert!(ns > 0, "num_seqs must be > 0");
                assert!(
                    ns <= npt,
                    "num_seqs ({ns}) > num_prefill_tokens ({npt}) — \
                     did you pass total_tokens for both?"
                );

                let batch_rows = (npt as usize).next_multiple_of(128);
                let nl = args.num_layers;
                let dims = variant.dims();
                let ffi = Self::args_to_ffi(args, batch_rows, nl, barrier_shape, inst_shape, timing_shape, dims);

                #[cfg(feature = "cuda")]
                {
                    variant.launch_prefill(
                        &ffi,
                        args.attn_scale, args.rms_norm_eps,
                        args.num_pages, ns, npt, nl as i32,
                        stream,
                        seq_chunk_lens.as_ptr() as *const i32,
                        seq_extend_offsets.as_ptr() as *const i32,
                    )
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = (args, variant, ns, npt, seq_chunk_lens, seq_extend_offsets, barrier_shape, inst_shape, timing_shape, stream, ffi, nl, dims);
                    panic!("launch_prefill requires --features cuda");
                }
            }
        }
    };

    // Phase 7: Generate variant dispatch if variants are defined
    let variant_code = if !def.variants.is_empty() {
        generate_variant_code(&def, &kernel_name.to_string())
    } else {
        quote! {}
    };

    let combined = quote! {
        #expanded
        #variant_code
    };

    combined.into()
}

/// Generate KernelVariant enum, FFI declarations, from_dims(), and dispatch methods
/// from the `variants { ... }` block in the megakernel! macro.
fn generate_variant_code(
    def: &parse::MegakernelDef,
    kernel_name: &str,
) -> proc_macro2::TokenStream {
    // Default VS from the kernel header params (shared across all LLaMA variants)
    let default_vs = def
        .params
        .iter()
        .find(|(k, _)| *k == "VS")
        .map(|(_, v)| *v)
        .unwrap_or(128256);

    let mut enum_variants = Vec::new();
    let mut from_dims_arms = Vec::new();
    let mut supported_entries = Vec::new();
    let mut decode_fn_arms = Vec::new();
    let mut prefill_fn_arms = Vec::new();
    let mut dims_arms = Vec::new();
    let mut ffi_externs = Vec::new();

    for v in &def.variants {
        let variant_ident = &v.name;
        let params: std::collections::HashMap<&str, usize> = v
            .params
            .iter()
            .map(|(k, v)| (k.to_string().leak() as &str, *v))
            .collect();

        let hd = params.get("HD").copied().unwrap_or(2048);
        let id = params.get("ID").copied().unwrap_or(8192);
        let hdm = params.get("HDM").copied().unwrap_or(64);
        let nah = params.get("NAH").copied().unwrap_or(32);
        let nkh = params.get("NKH").copied().unwrap_or(8);

        // Build FFI symbol names: {kernel_name}_{variant_suffix}_decode_static_launch
        // Convert CamelCase (Hd4096Hdm128) to snake_case (hd4096_hdm128)
        let variant_suffix = to_snake_case(&variant_ident.to_string());
        let decode_sym = syn::Ident::new(
            &format!("{kernel_name}_{variant_suffix}_decode_static_launch"),
            variant_ident.span(),
        );
        let prefill_sym = syn::Ident::new(
            &format!("{kernel_name}_{variant_suffix}_prefill_static_launch"),
            variant_ident.span(),
        );

        // Enum variant
        enum_variants.push(quote! { #variant_ident });

        // from_dims match arm
        let hd_lit = proc_macro2::Literal::usize_unsuffixed(hd);
        let id_lit = proc_macro2::Literal::usize_unsuffixed(id);
        let hdm_lit = proc_macro2::Literal::usize_unsuffixed(hdm);
        let nah_lit = proc_macro2::Literal::usize_unsuffixed(nah);
        let nkh_lit = proc_macro2::Literal::usize_unsuffixed(nkh);
        from_dims_arms.push(quote! {
            (#hd_lit, #id_lit, #hdm_lit, #nah_lit, #nkh_lit) => Ok(Self::#variant_ident)
        });

        // Supported variants entry for error message
        let desc = variant_ident.to_string();
        supported_entries.push(quote! {
            (#hd, #id, #hdm, #nah, #nkh, #desc)
        });

        // decode_fn / prefill_fn match arms
        decode_fn_arms.push(quote! {
            Self::#variant_ident => #decode_sym
        });
        prefill_fn_arms.push(quote! {
            Self::#variant_ident => #prefill_sym
        });

        let vs_val = default_vs;
        dims_arms.push(quote! {
            Self::#variant_ident => KernelDims {
                hd: #hd, id: #id, hdm: #hdm, nah: #nah, nkh: #nkh, vs: #vs_val,
            }
        });

        // FFI extern declarations
        ffi_externs.push(quote! {
            fn #decode_sym(
                bar: TkTensorArg, instructions: TkTensorArg, timings: TkTensorArg,
                qkv_w: TkTensorArg, attn_norm_w: TkTensorArg, o_w: TkTensorArg,
                mlp_norm_w: TkTensorArg, up_w: TkTensorArg, gate_w: TkTensorArg,
                down_w: TkTensorArg, lm_norm_w: TkTensorArg, lm_w: TkTensorArg,
                k_cache: TkTensorArg, v_cache: TkTensorArg,
                rope_cos: TkTensorArg, rope_sin: TkTensorArg,
                hidden: TkTensorArg, rms_rope: TkTensorArg, rms_gate: TkTensorArg,
                q_post: TkTensorArg, attn_out: TkTensorArg, silu: TkTensorArg,
                rms_lm: TkTensorArg, logits: TkTensorArg,
                pos_ids: TkTensorArg, kv_indptr: TkTensorArg, kv_indices: TkTensorArg,
                kv_last_page: TkTensorArg, kv_append: TkTensorArg,
                pfx_qo_indptr: TkTensorArg, pfx_kv_indptr: TkTensorArg,
                pfx_kv_indices: TkTensorArg, pfx_kv_last_page_len: TkTensorArg,
                attn_scale: f32, rms_norm_eps: f32,
                num_pages: i32, batch_size: i32, num_prefill_tokens: i32, num_layers: i32,
                stream: u64,
            ) -> i32;

            fn #prefill_sym(
                bar: TkTensorArg, instructions: TkTensorArg, timings: TkTensorArg,
                qkv_w: TkTensorArg, attn_norm_w: TkTensorArg, o_w: TkTensorArg,
                mlp_norm_w: TkTensorArg, up_w: TkTensorArg, gate_w: TkTensorArg,
                down_w: TkTensorArg, lm_norm_w: TkTensorArg, lm_w: TkTensorArg,
                k_cache: TkTensorArg, v_cache: TkTensorArg,
                rope_cos: TkTensorArg, rope_sin: TkTensorArg,
                hidden: TkTensorArg, rms_rope: TkTensorArg, rms_gate: TkTensorArg,
                q_post: TkTensorArg, attn_out: TkTensorArg, silu: TkTensorArg,
                rms_lm: TkTensorArg, logits: TkTensorArg,
                pos_ids: TkTensorArg, kv_indptr: TkTensorArg, kv_indices: TkTensorArg,
                kv_last_page: TkTensorArg, kv_append: TkTensorArg,
                pfx_qo_indptr: TkTensorArg, pfx_kv_indptr: TkTensorArg,
                pfx_kv_indices: TkTensorArg, pfx_kv_last_page_len: TkTensorArg,
                attn_scale: f32, rms_norm_eps: f32,
                num_pages: i32, batch_size: i32, num_prefill_tokens: i32, num_layers: i32,
                stream: u64,
                seq_chunk_lens: *const i32, seq_extend_offsets: *const i32,
            ) -> i32;
        });
    }

    quote! {
        /// Model dimensions for a kernel variant.
        #[derive(Debug, Clone, Copy)]
        pub struct KernelDims {
            pub hd: usize,
            pub id: usize,
            pub hdm: usize,
            pub nah: usize,
            pub nkh: usize,
            pub vs: usize,
        }

        impl KernelDims {
            /// QKV fused projection dimension: (NAH + 2*NKH) * HDM
            pub fn qkv_dim(&self) -> usize {
                (self.nah + 2 * self.nkh) * self.hdm
            }
        }

        /// Common launch signature type (decode). All variants share this signature.
        #[cfg(feature = "cuda")]
        type DecodeLaunchFn = unsafe extern "C" fn(
            TkTensorArg, TkTensorArg, TkTensorArg,
            TkTensorArg, TkTensorArg, TkTensorArg, TkTensorArg,
            TkTensorArg, TkTensorArg, TkTensorArg,
            TkTensorArg, TkTensorArg,
            TkTensorArg, TkTensorArg,
            TkTensorArg, TkTensorArg,
            TkTensorArg, TkTensorArg, TkTensorArg, TkTensorArg,
            TkTensorArg, TkTensorArg, TkTensorArg, TkTensorArg,
            TkTensorArg, TkTensorArg, TkTensorArg, TkTensorArg, TkTensorArg,
            TkTensorArg, TkTensorArg, TkTensorArg, TkTensorArg,
            f32, f32,
            i32, i32, i32, i32,
            u64,
        ) -> i32;

        /// Common launch signature type (prefill). Same as decode + seq metadata.
        #[cfg(feature = "cuda")]
        type PrefillLaunchFn = unsafe extern "C" fn(
            TkTensorArg, TkTensorArg, TkTensorArg,
            TkTensorArg, TkTensorArg, TkTensorArg, TkTensorArg,
            TkTensorArg, TkTensorArg, TkTensorArg,
            TkTensorArg, TkTensorArg,
            TkTensorArg, TkTensorArg,
            TkTensorArg, TkTensorArg,
            TkTensorArg, TkTensorArg, TkTensorArg, TkTensorArg,
            TkTensorArg, TkTensorArg, TkTensorArg, TkTensorArg,
            TkTensorArg, TkTensorArg, TkTensorArg, TkTensorArg, TkTensorArg,
            TkTensorArg, TkTensorArg, TkTensorArg, TkTensorArg,
            f32, f32,
            i32, i32, i32, i32,
            u64,
            *const i32, *const i32,
        ) -> i32;

        #[cfg(feature = "cuda")]
        unsafe extern "C" {
            #(#ffi_externs)*
        }

        /// A compiled kernel variant selected at runtime based on model dimensions.
        #[derive(Debug, Clone, Copy)]
        pub enum KernelVariant {
            #(#enum_variants,)*
        }

        impl KernelVariant {
            /// Return the model dimensions for this compiled variant.
            pub fn dims(&self) -> KernelDims {
                match self {
                    #(#dims_arms,)*
                }
            }

            /// Select the compiled kernel variant matching the given model dimensions.
            pub fn from_dims(
                hidden_dim: usize,
                intermediate_dim: usize,
                head_dim: usize,
                num_attention_heads: usize,
                num_kv_heads: usize,
            ) -> Result<Self, String> {
                match (hidden_dim, intermediate_dim, head_dim, num_attention_heads, num_kv_heads) {
                    #(#from_dims_arms,)*
                    _ => {
                        let supported: &[(usize, usize, usize, usize, usize, &str)] = &[
                            #(#supported_entries,)*
                        ];
                        let mut msg = format!(
                            "no compiled TK kernel variant for dims (HD={hidden_dim}, ID={intermediate_dim}, \
                             HDM={head_dim}, NAH={num_attention_heads}, NKH={num_kv_heads}). \
                             Supported variants:\n"
                        );
                        for &(hd, id, hdm, nah, nkh, desc) in supported {
                            msg.push_str(&format!(
                                "  - HD={hd}, ID={id}, HDM={hdm}, NAH={nah}, NKH={nkh} ({desc})\n"
                            ));
                        }
                        Err(msg)
                    }
                }
            }

            /// Launch the decode kernel for this variant.
            ///
            /// # Safety
            /// All TkTensorArg pointers must point to valid GPU memory with correct shapes.
            #[cfg(feature = "cuda")]
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn launch_decode(
                &self,
                ffi_args: &[TkTensorArg; 33],
                attn_scale: f32,
                rms_norm_eps: f32,
                num_pages: i32,
                batch_size: i32,
                num_prefill_tokens: i32,
                num_layers: i32,
                stream: u64,
            ) -> i32 {
                let f: DecodeLaunchFn = match self {
                    #(#decode_fn_arms,)*
                };
                unsafe {
                    f(
                        ffi_args[0], ffi_args[1], ffi_args[2],
                        ffi_args[3], ffi_args[4], ffi_args[5], ffi_args[6],
                        ffi_args[7], ffi_args[8], ffi_args[9],
                        ffi_args[10], ffi_args[11],
                        ffi_args[12], ffi_args[13],
                        ffi_args[14], ffi_args[15],
                        ffi_args[16], ffi_args[17], ffi_args[18], ffi_args[19],
                        ffi_args[20], ffi_args[21], ffi_args[22], ffi_args[23],
                        ffi_args[24], ffi_args[25], ffi_args[26], ffi_args[27], ffi_args[28],
                        ffi_args[29], ffi_args[30], ffi_args[31], ffi_args[32],
                        attn_scale, rms_norm_eps,
                        num_pages, batch_size, num_prefill_tokens, num_layers,
                        stream,
                    )
                }
            }

            /// Launch the prefill kernel for this variant.
            ///
            /// # Safety
            /// All TkTensorArg pointers must point to valid GPU memory with correct shapes.
            #[cfg(feature = "cuda")]
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn launch_prefill(
                &self,
                ffi_args: &[TkTensorArg; 33],
                attn_scale: f32,
                rms_norm_eps: f32,
                num_pages: i32,
                batch_size: i32,
                num_prefill_tokens: i32,
                num_layers: i32,
                stream: u64,
                seq_chunk_lens: *const i32,
                seq_extend_offsets: *const i32,
            ) -> i32 {
                let f: PrefillLaunchFn = match self {
                    #(#prefill_fn_arms,)*
                };
                unsafe {
                    f(
                        ffi_args[0], ffi_args[1], ffi_args[2],
                        ffi_args[3], ffi_args[4], ffi_args[5], ffi_args[6],
                        ffi_args[7], ffi_args[8], ffi_args[9],
                        ffi_args[10], ffi_args[11],
                        ffi_args[12], ffi_args[13],
                        ffi_args[14], ffi_args[15],
                        ffi_args[16], ffi_args[17], ffi_args[18], ffi_args[19],
                        ffi_args[20], ffi_args[21], ffi_args[22], ffi_args[23],
                        ffi_args[24], ffi_args[25], ffi_args[26], ffi_args[27], ffi_args[28],
                        ffi_args[29], ffi_args[30], ffi_args[31], ffi_args[32],
                        attn_scale, rms_norm_eps,
                        num_pages, batch_size, num_prefill_tokens, num_layers,
                        stream,
                        seq_chunk_lens, seq_extend_offsets,
                    )
                }
            }
        }
    }
}

fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|part| {
            let mut c = part.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            }
        })
        .collect()
}

/// Convert CamelCase to snake_case: "Hd4096Hdm128" → "hd4096_hdm128"
fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            // Insert underscore before uppercase letter, but not between consecutive
            // uppercase letters that start a word (e.g., "HDM" → "hdm" not "h_d_m")
            let prev = s.chars().nth(i - 1).unwrap();
            if prev.is_lowercase() || prev.is_ascii_digit() {
                result.push('_');
            }
        }
        result.push(ch.to_lowercase().next().unwrap());
    }
    result
}
