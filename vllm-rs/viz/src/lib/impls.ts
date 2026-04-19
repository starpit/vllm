/** Human-readable descriptor for each implementation name the solver
 * can pick. Enriches the raw `Implementation::name()` string from
 * ferrite-forward-macro's impl_lib with "what does this actually
 * dispatch to?" context.
 *
 * Fields:
 *  - `kernel`  — short label, shown ON the chip. Vendor/library name
 *                when applicable ("Marlin", "FlashInfer", "cuBLASLt",
 *                "CUTLASS") or "custom" for our own kernels.
 *  - `note`    — detail for the tooltip / expanded view: what the
 *                kernel actually does, with special emphasis on
 *                fusions (since those are the interesting bit that's
 *                invisible from the DSL).
 *
 * Keep this in sync with `crates/ferrite-forward-macro/src/impl_lib.rs`
 * — each `fn name() -> &'static str` there should have an entry here. */
export interface ImplDescriptor {
  kernel: string;
  note: string;
}

const IMPLS: Record<string, ImplDescriptor> = {
  // ── Embedding / reshape / elementwise ──
  embed_ref: {
    kernel: "custom",
    note: "gather: output[i] = embed_tokens[input_ids[i]]",
  },
  reshape_ref: {
    kernel: "custom",
    note: "no-op view rewrite (stride/shape only, no copy)",
  },
  add_ref: {
    kernel: "custom",
    note: "elementwise add — unfused path",
  },
  scalar_mul_inplace: {
    kernel: "custom",
    note: "in-place scale — rewrites the input tensor",
  },
  tanh_softcap_inplace: {
    kernel: "custom",
    note: "in-place soft-cap: x ← cap · tanh(x / cap)",
  },

  // ── Norms ──
  fused_add_rms_norm: {
    kernel: "custom (fused)",
    note: "residual add + rmsnorm in one kernel — skips a GMEM round-trip",
  },
  fused_add_rms_norm_with_offset: {
    kernel: "custom (fused)",
    note: "residual add + rmsnorm with a scalar weight offset (Gemma)",
  },
  scalar_offset_rms_norm: {
    kernel: "custom",
    note: "rmsnorm with a scalar offset on the weights (Gemma)",
  },

  // ── GEMM family ──
  fused_gemm_bias: {
    kernel: "cuBLASLt",
    note: "bf16 GEMM + bias add, via cuBLASLt epilogue",
  },
  gemm_ref: {
    kernel: "cuBLASLt",
    note: "bf16 GEMM (no bias); vendor-library fallback",
  },
  fused_gate_up_silu_mul: {
    kernel: "custom (fused)",
    note: "gate_proj + up_proj + silu(gate) * up — one kernel instead of three",
  },
  fused_gate_up_gelu_mul: {
    kernel: "custom (fused)",
    note: "gate_proj + up_proj + gelu(gate) * up",
  },

  // ── RoPE / QKV fusions ──
  fused_qkv_rope_cache: {
    kernel: "custom (fused)",
    note: "q/k/v projections + RoPE + KV-cache write in one kernel — the big decode-path fusion",
  },
  fused_qkv_qk_norm_rope_cache: {
    kernel: "custom (fused)",
    note: "qkv + QK-norm (Qwen3-style) + RoPE + KV-cache write",
  },
  fused_qkv_rope_prefill: {
    kernel: "custom (fused)",
    note: "prefill-path qkv + RoPE (no KV-cache write path)",
  },
  rope_append_ref: {
    kernel: "custom",
    note: "separate RoPE + KV-cache write — unfused fallback",
  },

  // ── Attention ──
  attention_via_cache: {
    kernel: "FlashAttention-2",
    note: "FA2 reading K/V from the paged KV-cache (decode path)",
  },
  attention_prefill_contiguous: {
    kernel: "FlashAttention-2",
    note: "FA2 prefill over contiguous K/V (no cache reads)",
  },
  sliding_attention_via_cache: {
    kernel: "FlashAttention-2",
    note: "FA2 + sliding-window mask, paged KV-cache (Gemma2/Mistral-sliding)",
  },
  sliding_attention_prefill_contiguous: {
    kernel: "FlashAttention-2",
    note: "FA2 prefill with sliding-window mask",
  },
  flashinfer_attention_decode: {
    kernel: "FlashInfer",
    note: "FlashInfer decode kernel — wins on long sk (KV-cache span)",
  },
  flashinfer_attention_prefill: {
    kernel: "FlashInfer",
    note: "FlashInfer prefill kernel",
  },

  // ── Quantized gemms (Marlin family — AWQ / GPTQ int4) ──
  marlin_awq_gemm: {
    kernel: "Marlin",
    note: "Marlin int4 GEMM, AWQ-packed weights (zero + scale per group)",
  },
  marlin_gptq_gemm: {
    kernel: "Marlin",
    note: "Marlin int4 GEMM, GPTQ-packed weights",
  },
  marlin_gptq_desc_act_gemm: {
    kernel: "Marlin",
    note: "Marlin int4 GEMM, GPTQ desc_act (activation reordering)",
  },
  marlin_ct_int4_sym_gemm: {
    kernel: "Marlin",
    note: "Marlin int4 GEMM, compressed-tensors symmetric int4",
  },
  marlin_bnb4_gemm: {
    kernel: "Marlin",
    note: "Marlin int4 GEMM, bitsandbytes NF4 / FP4",
  },

  // ── FP8 gemms (CUTLASS family) ──
  cutlass_fp8_gemm: {
    kernel: "CUTLASS",
    note: "CUTLASS FP8 GEMM, per-tensor dynamic quant",
  },
  cutlass_fp8_static_gemm: {
    kernel: "CUTLASS",
    note: "CUTLASS FP8 GEMM, static per-tensor scales (compressed-tensors FP8)",
  },
  cutlass_fp8_blockwise_gemm: {
    kernel: "CUTLASS",
    note: "CUTLASS FP8 GEMM, 128×128 blockwise scales",
  },
};

export function describeImpl(name: string): ImplDescriptor {
  return (
    IMPLS[name] ?? {
      kernel: "?",
      note: `unknown impl: ${name}`,
    }
  );
}
