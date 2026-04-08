{# Multi-CTA GEMM phase implemented via CUTLASS device-side ThreadblockMma.
   Replaces our hand-rolled cooperative GEMM with CUTLASS's pipelined cp.async
   multistage mainloop. Currently used for down_proj as a proof-of-concept.

   Variables:
     phase_comment: header comment string
     a_ptr_expr:    C++ expression for the A operand base pointer (bf16*)
     b_ptr_expr:    C++ expression for the B operand base pointer (bf16*)
     out_ptr_expr:  C++ expression for the output base pointer (bf16*)
     m_dim:         M extent (e.g. q_size for our prefill kernel)
     k_dim_value:   K extent (kernel-wide dim, e.g. ID for down_proj)
     n_dim_value:   N extent (e.g. HD for down_proj)
#}
    // ════ {{ phase_comment }} ════
    {
    // Type aliases pulled from the file-scope `pfl_cutlass` namespace
    // (defined in preamble_header.cu). Fully-qualified names everywhere
    // because TK / kittens have their own bf16 type names that would shadow.
    using PflCutlassMma            = pfl_cutlass::ThreadblockMma;
    using PflCutlassIteratorA      = pfl_cutlass::IteratorA;
    using PflCutlassIteratorB      = pfl_cutlass::IteratorB;
    using PflCutlassSharedStorageT = pfl_cutlass::SharedStorage;
    using PflCutlassFragmentC      = typename PflCutlassMma::FragmentC;
    constexpr int kThreadblockM = pfl_cutlass::ThreadblockShape::kM;  // 256
    constexpr int kThreadblockN = pfl_cutlass::ThreadblockShape::kN;  // 128
    constexpr int kThreadblockK = pfl_cutlass::ThreadblockShape::kK;  // 32

    const int M = ({{ m_dim }});
    constexpr int K = ({{ k_dim_value }});
    constexpr int N = ({{ n_dim_value }});

    // Threadblock tile counts.
    const int row_tiles_cta = (M + kThreadblockM - 1) / kThreadblockM;
    constexpr int col_tiles_cta = (N + kThreadblockN - 1) / kThreadblockN;
    const int total_work_cta = row_tiles_cta * col_tiles_cta;

    // CUTLASS Mma owns the entire kernel-wide shmem region as its working set.
    // Re-cast __shm. The static_assert in preamble_header.cu ensures the size
    // fits the dynamic shmem opt-in budget.
    PflCutlassSharedStorageT &shared_storage =
        *reinterpret_cast<PflCutlassSharedStorageT*>(__shm);

    const int thread_idx = threadIdx.x;
    const int warp_idx   = thread_idx / 32;
    const int lane_idx   = thread_idx % 32;

    // Layout params (precomputed strides). Constructed on device because we
    // can't smuggle host-precomputed Params through the megakernel boundary.
    // NOTE: brace init to dodge the most-vexing-parse — `Params x(Layout(K))`
    // is parsed as a function declaration. `Params x{Layout{K}}` is not.
    pfl_cutlass::LayoutA layout_a{K};
    pfl_cutlass::LayoutB layout_b{K};
    typename PflCutlassIteratorA::Params params_A{layout_a};
    typename PflCutlassIteratorB::Params params_B{layout_b};

    cutlass::bfloat16_t *ptr_A = reinterpret_cast<cutlass::bfloat16_t*>({{ a_ptr_expr }});
    cutlass::bfloat16_t *ptr_B = reinterpret_cast<cutlass::bfloat16_t*>({{ b_ptr_expr }});
    __nv_bfloat16 *ptr_OUT = reinterpret_cast<__nv_bfloat16*>({{ out_ptr_expr }});

    // Each CTA processes (row_tile_cta, col_tile_cta) pairs in a strided loop.
    for (int wu = bid; wu < total_work_cta; wu += num_ctas) {
        const int rt = wu / col_tiles_cta;  // M tile index
        const int ct = wu % col_tiles_cta;  // N tile index

        const int tb_m = rt * kThreadblockM;
        const int tb_n = ct * kThreadblockN;

        // Construct iterators (per-work-unit, cheap — just precomputed
        // strides + a base pointer offset).
        PflCutlassIteratorA iter_A(
            params_A,
            ptr_A,
            /*extent*/ {M, K},
            thread_idx,
            /*tb_offset*/ {tb_m, 0},
            /*gather_indices*/ nullptr);
        PflCutlassIteratorB iter_B(
            params_B,
            ptr_B,
            /*extent*/ {K, N},
            thread_idx,
            /*tb_offset*/ {0, tb_n},
            /*gather_indices*/ nullptr);

        // Construct the threadblock-scoped Mma. Mma takes the inner
        // MmaSharedStorage member of the union, not the union itself.
        PflCutlassMma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

        PflCutlassFragmentC accum;
        accum.clear();

        constexpr int gemm_k_iterations = (K + kThreadblockK - 1) / kThreadblockK;

        // ── The main loop. CUTLASS multistage cp.async pipelined mainloop. ──
        mma(gemm_k_iterations, accum, iter_A, iter_B, accum);

        // ── Epilogue: variant selected at codegen time ──
        // LinearCombination     (default): D = α*acc + β*source  (β literal)
        // LinearCombinationSiluMul        : D = silu(α*acc) * source
{%- if silu_mul %}
        using Epilogue = pfl_cutlass::EpilogueSiluMul;
        using OutputTileIterator = pfl_cutlass::OutputTileIterator;
        using OutputOp = pfl_cutlass::OutputOpSiluMul;
{%- else %}
        using Epilogue = pfl_cutlass::Epilogue;
        using OutputTileIterator = pfl_cutlass::OutputTileIterator;
        using OutputOp = pfl_cutlass::OutputOpT;
{%- endif %}

        // Need a CUTLASS-side __syncthreads before the epilogue starts
        // touching shmem (the main loop and epilogue share __shm via the
        // union).
        __syncthreads();

        // Output element type as CUTLASS expects (bf16). Reinterpret the
        // raw __nv_bfloat16* (TK type) — same 16-bit layout, different name.
        cutlass::bfloat16_t *ptr_OUT_cl = reinterpret_cast<cutlass::bfloat16_t*>(ptr_OUT);

        pfl_cutlass::LayoutC layout_c{N};
        typename OutputTileIterator::Params params_C{layout_c};
        typename OutputTileIterator::Params params_D{layout_c};

        OutputTileIterator iter_C(
            params_C,
            ptr_OUT_cl,
            /*extent*/ {M, N},
            thread_idx,
            /*tb_offset*/ {tb_m, tb_n},
            /*scatter_indices*/ nullptr);
        OutputTileIterator iter_D(
            params_D,
            ptr_OUT_cl,
            /*extent*/ {M, N},
            thread_idx,
            /*tb_offset*/ {tb_m, tb_n},
            /*scatter_indices*/ nullptr);

        typename OutputOp::Params output_params{
            /*alpha=*/ 1.0f,
            /*beta=*/  {{ beta_literal }}};
        OutputOp output_op(output_params);

        // The epilogue reuses the SAME shared_storage region via the union.
        Epilogue epilogue(
            shared_storage.epilogue,
            thread_idx,
            warp_idx,
            lane_idx);
        epilogue(output_op, iter_D, accum, iter_C);
    }
    }
    __syncthreads();
