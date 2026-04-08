// LLaMA architecture + supported model variants. Single source of truth for
// the scheduled megakernel codegen registry. Adding a new model size means
// adding one entry to the `variants { ... }` block; build.rs walks the
// variants and emits one CUDA kernel per (variant_name, dims) tuple.
//
// The kernel header `<NL=2, HD=256, ...>` declares the *default* parameters
// (used as fallbacks if a variant omits them). The body is a parametric
// description of the LLaMA forward pass on those symbols.
//
// SEQ_LEN is parameterized like a model dim because the scheduled megakernel
// bakes the wave schedule against a specific seq length; different prefill
// lengths therefore need different variants. Future work: bucket SEQ_LEN
// onto a small fixed list per model size.
//
// The body below currently exists for documentation and future use — the
// scheduled megakernel pipeline reads only the variants block today and
// reifies the LLaMA forward pass via reify_llama (which has the 8-phase
// structure hardcoded). Plumbing the body through end-to-end is a follow-up.

kernel llama<NL=2, HD=256, ID=512, HDM=64, NAH=4, NKH=2, VS=128256, SEQ_LEN=32> {
    for layer in 0..NL {
        let normed = rmsnorm(hidden_states, attn_norm[layer]);
        let qkv = gemm(normed, qkv_weights[layer]);
        let (q, k, v) = rope_append(qkv, positions, kv_cache[layer]);
        let attn = attention_decode(q, k, v, kv_cache[layer], block_table);
        hidden_states = gemm_add(attn, o_proj[layer], hidden_states);

        let normed2 = rmsnorm(hidden_states, mlp_norm[layer]);
        let gate = silu(gemm(normed2, gate_weights[layer]));
        let up = gemm(normed2, up_weights[layer]);
        hidden_states = gemm_add(gate * up, down_proj[layer], hidden_states);
    }
}

variants {
    // ── Unit-test fixtures (small, fast, exhaustive validation) ──
    //
    // `tiny`: smallest viable correctness check — 2 layers, 32 tokens.
    tiny: {
        NL=2, HD=256, ID=512, HDM=64, NAH=4, NKH=2, VS=128256, SEQ_LEN=32
    },
    // `medium`: scaling sanity check — multi-page KV cache, 4 layers,
    // larger HD/ID, GQA ratio = 2.
    medium: {
        NL=4, HD=512, ID=1024, HDM=64, NAH=8, NKH=4, VS=128256, SEQ_LEN=64
    },

    // ── Real model variants ──
    //
    // LLaMA 3.2 1B: 16 layers, HD=2048, ID=8192, NAH=32, NKH=8, HDM=64.
    //
    // seq64: validation bucket. Real model dims, short prefill. Used for
    // golden-file end-to-end correctness — CPU forward at seq=64 runs in
    // ~5 minutes (one-time, slow, committed). Smallest "real" variant.
    llama_3_2_1b_seq64: {
        NL=16, HD=2048, ID=8192, HDM=64, NAH=32, NKH=8, VS=128256, SEQ_LEN=64
    },
    // seq1024: production prefill bucket. Used for benchmarking. Compiled
    // and linked but not validated against a CPU golden (cpu_forward
    // runtime would be ~80 minutes single-threaded). Cross-validation
    // against the existing fused prefill kernel is a separate task.
    llama_3_2_1b_seq1024: {
        NL=16, HD=2048, ID=8192, HDM=64, NAH=32, NKH=8, VS=128256, SEQ_LEN=1024
    },
}
