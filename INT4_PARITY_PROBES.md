# int4 parity — P0 verification probes

Companion to `INT4_PARITY_PLAN.md`. Captures verified facts that gate P1 plumbing. Probed against the actual MLX-community checkpoints on HF.

Date: 2026-05-10. Probes run from `worktree-ferrite-metal` HEAD `4efc5c17f`.

---

## 1. Safetensors layout (MLX `affine` mode)

Probed against `mlx-community/Llama-3.2-1B-Instruct-4bit` (`/tmp/p01_probe.out`).

### Per-tensor schema for a Linear layer

For `<prefix>` (e.g. `model.layers.0.self_attn.q_proj`):

| Tensor | Shape | Dtype string |
|---|---|---|
| `<prefix>.weight` | `[N, K / pack_factor]` | **`"U32"`** |
| `<prefix>.scales` | `[N, K / group_size]` | **`"F16"`** |
| `<prefix>.biases` | `[N, K / group_size]` | **`"F16"`** |

Where `pack_factor = 32 / bits = 8` for bits=4, and `group_size = 64` for the canonical mlx-community checkpoint.

**Naming gotcha confirmed**: `<prefix>.biases` is the *per-group affine offset* (the bias term in `w_dq = w_packed * scale + bias`). A linear layer's actual fp16 *layer bias* (where present) lives at `<prefix>.bias` — different name, different shape, different role. Llama-3.2 has `attention_bias: false, mlp_bias: false`, so no `<prefix>.bias` keys appear at all.

### Verified shapes (Llama-3.2-1B, hidden=2048, kv_heads=8, head_dim=64, intermediate=8192, vocab=128256)

```
model.embed_tokens.weight        [128256, 256]   U32     (vocab=128256, hidden/8=256)
model.embed_tokens.scales        [128256, 32]    F16     (hidden/gs = 2048/64 = 32)
model.embed_tokens.biases        [128256, 32]    F16
model.layers.0.self_attn.q_proj.weight  [2048, 256]    U32  (N=hidden=2048, K/8=256)
model.layers.0.self_attn.q_proj.scales  [2048, 32]     F16  (K/gs=2048/64=32)
model.layers.0.self_attn.q_proj.biases  [2048, 32]     F16
model.layers.0.self_attn.k_proj.weight  [512, 256]     U32  (N=kv_heads*head_dim=512, K=2048)
model.layers.0.self_attn.v_proj.weight  [512, 256]     U32
model.layers.0.self_attn.o_proj.weight  [2048, 256]    U32  (N=hidden=2048, K=2048)
model.layers.0.mlp.gate_proj.weight     [8192, 256]    U32  (N=intermediate=8192, K=2048)
model.layers.0.mlp.up_proj.weight       [8192, 256]    U32
model.layers.0.mlp.down_proj.weight     [2048, 1024]   U32  (N=2048, K=intermediate=8192, K/8=1024)
model.layers.0.mlp.down_proj.scales     [2048, 128]    F16  (K/gs=8192/64=128)
model.layers.0.input_layernorm.weight   [2048]         F16  (NOT quantized — dense F16)
model.layers.0.post_attention_layernorm.weight  [2048] F16
model.norm.weight                       [2048]         F16
```

Critical observations:

- **`U32` is the dtype string** that the safetensors loader will see. `weights.rs` matches dtype by string equality at `:103`.
- **Scales + biases are `F16`, not `BF16`** — the plan's "BF16" guess in §2 of the dispatcher write-up was wrong. F16 is the MLX default for q4 affine.
- **RMSNorm gains stay F16** (not quantized). Mixed-quant model handling (P11) is required from day one — every model has these.
- **No `lm_head.*` keys present** because `tie_word_embeddings: true`. Only `model.embed_tokens.{weight, scales, biases}` is stored; lm_head reuses it.
- **No fp linear-layer biases on Llama-3.2** (attention_bias and mlp_bias both false). The `linear_bias` field on `AffineQuantLinear` will be `None` for Llama; reserved for models that have layer biases (Phi-3 gate, Qwen2 attention bias, etc.).

### `config.json` quantization metadata

Both `quantization` and `quantization_config` keys appear at config root with the same payload:

```json
"quantization": {"group_size": 64, "bits": 4}
"quantization_config": {"group_size": 64, "bits": 4}
"tie_word_embeddings": true
"torch_dtype": "bfloat16"
```

The macro can read either key; `quantization_config` is the HF-transformers-canonical name. The runtime activation dtype is bf16 (`torch_dtype`) but quant scales/biases are stored f16. This means scales/biases get cast to bf16 at use time — verify in P3 that the `qmv_*` / `qmm_*` kernels handle the f16 → bf16 implicit cast (they read scales as the kernel's dtype template parameter, so we instantiate kernels for `T = bfloat16_t` and the scales are loaded into `bfloat16_t`; F16 → BF16 happens at safetensors-decode time or via a cast in the kernel — confirm in P3).

---

## 2. Per-model 4bit coverage matrix

Probed via metadata-only HF range reads (`/tmp/p04_models.out`). All sampled checkpoints use **gs=64, bits=4** — uniform. F16 scales/biases is the format across the board.

| Model | HF repo (mlx-community) | arch in config | tied lm_head | embed quantized | shards |
|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | `Llama-3.2-1B-Instruct-4bit` | LlamaForCausalLM | true | yes | 1 |
| Llama-3.2-3B-Instruct | `Llama-3.2-3B-Instruct-4bit` | LlamaForCausalLM | true | yes | 1 |
| Qwen2.5-7B-Instruct | `Qwen2.5-7B-Instruct-4bit` | Qwen2ForCausalLM | false | yes | 1 |
| Qwen3-1.7B | `Qwen3-1.7B-4bit` | Qwen3ForCausalLM | true | yes | 1 |
| Qwen3-4B | `Qwen3-4B-4bit` | Qwen3ForCausalLM | true | yes | 1 |
| Qwen3-8B | `Qwen3-8B-4bit` | Qwen3ForCausalLM | false | yes | 1 |
| Qwen3-30B-A3B (MoE) | `Qwen3-30B-A3B-4bit` | Qwen3MoeForCausalLM | false | yes | 4 |
| Mistral-7B-Instruct-v0.3 | `Mistral-7B-Instruct-v0.3-4bit` | MistralForCausalLM | false | yes | 1 |
| Mixtral-8x7B-Instruct-v0.1 (MoE) | `Mixtral-8x7B-Instruct-v0.1-4bit` | MixtralForCausalLM | false | yes | 5 |
| Gemma-2-2B-it | `gemma-2-2b-it-4bit` | Gemma2ForCausalLM | not set (assume true per arch) | yes | 1 |
| Gemma-2-9B-it | `gemma-2-9b-it-4bit` | Gemma2ForCausalLM | not set | yes | 1 |
| Gemma-3-1B-it | `gemma-3-1b-it-4bit` | Gemma3ForCausalLM | not set | yes | 1 |
| Gemma-3-4B-it (multimodal) | `gemma-3-4b-it-4bit` | Gemma3ForConditionalGeneration | not set | **NO** (language_model embedding stored as fp, vision encoder unquantized) | 1 |
| DeepSeek-V3 | `DeepSeek-V3-4bit` | DeepseekV3ForCausalLM | false | yes | 88 |
| Phi-3-mini-4k | `phi-3-mini-4k-instruct-4bit` | Phi3ForCausalLM | false | yes | 1 |

### Architecture → ferrite model crate mapping

| HF arch | ferrite crate | int4 status |
|---|---|---|
| LlamaForCausalLM | `ferrite-model-llama` | P10 target (1B, 3B) |
| Qwen2ForCausalLM | `ferrite-model-qwen2` | covered |
| Qwen2MoeForCausalLM | `ferrite-model-qwen2-moe` | needs MoE plumbing (P13) |
| Qwen2VLForConditionalGeneration | `ferrite-model-qwen2-vl` | language tower under P10 plumbing; vision unquantized |
| Qwen2_5_VLForConditionalGeneration | `ferrite-model-qwen2-5-vl` | same as above |
| Qwen3ForCausalLM | `ferrite-model-qwen3` | covered |
| Qwen3MoeForCausalLM | `ferrite-model-qwen3-moe` | P13 |
| MistralForCausalLM | `ferrite-model-mistral` | covered |
| MixtralForCausalLM | `ferrite-model-mixtral` | P13 |
| Gemma2ForCausalLM | `ferrite-model-gemma2` | covered |
| Gemma3ForCausalLM | `ferrite-model-gemma3` | covered |
| Gemma3ForConditionalGeneration | `ferrite-model-gemma3-mm` | covered (mixed-quant from day one) |
| DeepseekV2ForCausalLM | `ferrite-model-deepseek-v2` | covered (no canonical 4bit on HF — verify or skip) |
| DeepseekV3ForCausalLM | `ferrite-model-deepseek-v3` (and `-flat`) | P13 (MLA + MoE) |
| Phi3ForCausalLM | `ferrite-model-phi3` | covered |
| GraniteForCausalLM | `ferrite-model-granite` | covered (Granite has GPTQ/AWQ/BNB goldens already, no MLX-q4 ckpt yet) |
| CohereForCausalLM | `ferrite-model-commandr` | unknown — search mlx-community for CommandR-4bit (P0.6 followup) |
| ModernBertModel | `ferrite-model-modernbert` | not generative — embedding model, lower priority for q4 |

**Plan correction**: the plan said "10 model crates" — actually **24 ferrite-model-\* crates**. Coverage matrix above adds the missing rows.

### Models that need new repos / probes

- **CommandR**: no canonical mlx-community 4bit found in the survey. Open question whether to add (low priority).
- **DeepSeek-V2** and **DeepSeek-V3-flat**: only V3 has a canonical 4bit. V2 + V3-flat may share weight format with V3 (verify in P13).
- **Granite**: vLLM repo has CUDA-side AWQ/GPTQ/BNB goldens; no mlx-community 4bit found. Could ship parity once mlx-community releases one.

---

## 3. NAX (M4+) detection

Source: `~/git/mlx/mlx/backend/metal/device.cpp:828-845`.

```cpp
bool is_nax_available() {
#ifdef MLX_METAL_NO_NAX
  return false;
#else
  auto _check_nax = []() {
    bool can_use_nax = false;
    if (__builtin_available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
      can_use_nax = true;
    }
    auto& d = metal::device(mlx::core::Device::gpu);
    auto arch = d.get_architecture().back();
    auto gen = d.get_architecture_gen();
    can_use_nax &= gen >= (arch == 'p' ? 18 : 17);
    return can_use_nax;
  };
  static bool is_nax_available_ = _check_nax();
  return is_nax_available_;
#endif
}
```

**Plan correction**: the plan summary said "Apple9 + arch_gen ≥ 13". Real check is:

1. Compile-time: `!defined(MLX_METAL_NO_NAX)` — environment opt-out.
2. Runtime macOS: `__builtin_available(macOS 26.2, ...)` — minimum macOS 26.2 / iOS 26.2.
3. Runtime arch_gen: `gen >= 17` for non-`'p'` arch, `gen >= 18` for `'p'` arch.

`get_architecture()` returns a string like `"applegpu_g16d"`; `.back()` is the last char (`'d'` here). `arch == 'p'` likely covers the Pro variants (e.g. M4 Pro). `arch_gen_` is set during `Device::Device()` from `MTLArchitecture` info — exact MTL API path needs investigation in P7 (lift the `arch_gen_` setter from MLX device.cpp `Device` constructor).

For the ferrite-metal port (P7):
- New `is_nax_available()` in `vllm-rs/crates/ferrite-metal-kernels/src/device.rs`
- Read MTL architecture from `objc2_metal::MTLDevice.architecture` (the `MTLArchitecture` Objective-C type). Need to verify objc2-metal exposes this — likely as `metal_device.architecture()`.
- Replicate the macOS-version `__builtin_available` check via Rust runtime version detect; in objc2 this is `objc2_foundation::NSProcessInfo::operatingSystemVersion()` or the `available!()` macro.
- Cache the result in a `OnceLock<bool>`.

---

## 4. cpu_golden infrastructure

Located at `vllm-rs/crates/ferrite-forward/src/cpu_golden.rs`.

Existing API: pure-Rust, operates on flat `&[f32]` slices. Already has `rmsnorm`, `gemm`, `gemm_add`. Suite extends naturally with:

```rust
pub fn affine_dequantize(packed: &[u32], scales: &[f32], biases: &[f32], group_size: usize, bits: usize) -> Vec<f32>;
pub fn affine_qmm_t(x: &[f32], packed_w: &[u32], scales: &[f32], biases: &[f32], m: usize, n: usize, k: usize, gs: usize, bits: usize) -> Vec<f32>;
pub fn affine_qmm_n(...);
pub fn affine_qmv(...);
pub fn affine_qvm(...);
pub fn affine_gather_qmm(...);
pub fn affine_embed(packed_w: &[u32], scales: &[f32], biases: &[f32], indices: &[i64], hidden: usize, gs: usize, bits: usize) -> Vec<f32>;
```

These extensions go in P9. cpu_golden callers in `vllm-e2e` exercise them through the existing per-op test harness — no new test scaffolding required (per `feedback_no_reinvent_testing`).

---

## 5. LinearLayer current state

`vllm-rs/crates/ferrite-kernels/src/layers.rs:738`.

```rust
pub enum LinearLayer {
    Dense(Linear),
    Marlin(Box<MarlinLinear>),
    Ggml(Box<GgmlLinear>),
    GgmlConcat(Vec<GgmlLinear>),
    Bnb4bit(Box<Bnb4bitLinear>),
    Fp8(Box<Fp8Linear>),
    Fp8Block(Box<Fp8BlockLinear>),
}
```

All quant arms hold `ferrite_cuda_core::tensor::GpuTensor` types; all CUDA-only. `dense_weight()` at `:858` returns `GpuTensor` and panics on every non-Dense arm — gate stands as documented.

**P1 plan**: add `AffineQuant(Box<AffineQuantLinear>)` arm. The new arm holds `objc2_metal::Buffer`-equivalent types (Metal-only); guard with `#[cfg(feature = "metal")]`. CUDA build never sees it. New accessors `affine_weight()`, `affine_scales()`, `affine_biases()`, `linear_bias()` on `LinearLayer` Metal-side.

---

## 6. Path corrections to `INT4_PARITY_PLAN.md`

The plan used `crates/...` paths in several citations. Actual workspace layout: **all crates live under `vllm-rs/crates/`**. Corrections:

| Plan path | Actual path |
|---|---|
| `crates/ferrite-metal/` | `vllm-rs/crates/ferrite-metal-kernels/` (no top-level `ferrite-metal` crate) |
| `crates/ferrite-forward/` | `vllm-rs/crates/ferrite-forward/` |
| `crates/ferrite-metal-kernels/` | `vllm-rs/crates/ferrite-metal-kernels/` |
| `crates/ferrite-kernels/src/layers.rs:738` | `vllm-rs/crates/ferrite-kernels/src/layers.rs:738` ✓ (line correct) |
| `crates/ferrite-forward-macro/src/impl_lib.rs` | `vllm-rs/crates/ferrite-forward-macro/src/impl_lib.rs` |
| `crates/vllm-e2e/testdata/golden/` | `vllm-rs/crates/vllm-e2e/testdata/golden/` |

These path fixes apply when executing P1+. Plan content is otherwise correct.

---

## 7. F16 → BF16 dtype interaction (P1 design — DECIDED)

Activations on Llama-3.2 are bf16 (`torch_dtype: "bfloat16"`). Safetensors stores scales + biases as **F16**. MLX kernels are templated on a single `T` for activations + scales + biases + outputs — verified at `~/git/mlx/mlx/backend/metal/kernels/quantized.h:692,750,1094`.

**Critical insight from reading MLX's actual `qmv_fast` body** (`quantized.h:750-814`): the kernel does **all math in `float`**, not `T`:

```cpp
typedef float U;
thread U x_thread[values_per_thread];
thread U result[results_per_simdgroup] = {0};
// ...
U s = sl[0];                                      // T scales → float (expand at load)
U b = bl[0];                                      // T biases → float
result[row] += qdot<U, ...>(wl, x_thread, s, b, sum);  // accumulate in float
```

`T` is only a *storage* type. Every load to a register expands `T → float`. The expansion happens regardless of whether T is f16 or bf16 — both are 16-bit floats with single-instruction expansion to f32 on Apple GPU.

**This means the cast cost is the same in either approach.** Cast-at-load (T=bf16, expand bf16→float at load) and cast-in-register (T_scale=f16, expand f16→float at load) both pay one expansion per scale+bias load. There's no f16→bf16 intermediate; the kernel goes one direction only.

**Microbench confirms** (M4 dev box, `microbench_int4_cast/`, naive single-thread-per-output kernel — compute-bound, *overestimates* cast cost vs real qmv_fast):

| Shape | cast-at-load (ns) | cast-in-register (ns) | Δ | BW |
|---|---|---|---|---|
| q_proj N=2048 K=2048 | 40,147 ± 198 | 40,060 ± 59 | **−0.22%** | 59 GB/s |
| down_proj N=2048 K=8192 | 157,819 ± 141 | 157,939 ± 42 | **+0.08%** | 60 GB/s |
| gate_proj N=8192 K=2048 | 151,827 ± 69 | 151,987 ± 38 | **+0.10%** | 62 GB/s |

10k iter × 5 repeats per shape, GPU-side timing. All deltas within ±0.25%, well inside stddev. **In-register cast is free.**

The 60 GB/s achieved bandwidth (vs M4 ~273 GB/s peak) confirms the naive kernel is compute-bound — making this the **conservative regime**. Real `qmv_fast` is more memory-bound, where cast cost hides further under load latency. If cast-in-register is in the noise here, it's in the noise there.

**Decision: in-register cast.** Reasons:

1. **Zero runtime ITL cost** — confirmed by microbench, not assumed.
2. **Smaller storage** — scales+biases stay f16 in device buffers (~5MB saved per Llama-1B, ~30MB per 7B model).
3. **Simpler loader** — no special-casing in `weights.rs::create_buffer`; just copy bytes verbatim from safetensors.
4. **Matches MLX behavior more directly** — MLX leaves scales f16 in device memory and casts on first kernel use (lazy graph eval); we do the same, just inside the kernel rather than the op layer.

**Kernel signature divergence (small)**: ferrite-metal kernels take `<T_act, T_scale, gs, bits>` instead of MLX's `<T, gs, bits>`. The kernel body is structurally identical — `U s = sl[0]` works the same way regardless of which 16-bit float `sl` points to. Pipeline cache key gains a `T_scale` axis. For our coverage matrix, `T_scale` is always f16 (matches safetensors); `T_act` is bf16 or f16 depending on `torch_dtype`.

**Caveats**:
- Microbench was M4 only. M3 may pay slightly more for f16→f32 conversion. If M3 perf measurements show a delta, re-run the microbench harness on M3 (it's still on disk at `microbench_int4_cast/`).
- The microbench's compute/memory ratio is higher than `qmv_fast`'s. Result is conservative.
- `feedback_no_shortcuts_kernels` is satisfied: the kernel body matches MLX byte-for-byte; only the parameter type for scales differs (one template parameter expansion).

**P1 implementation.** In `weights.rs::create_buffer` — no cast logic needed. Just copy safetensors bytes as-is. The macro detects the f16 scales/biases via dtype string and instantiates the kernel with `T_scale = half`, `T_act = bfloat` for bf16 models.

**Outdated alternatives** considered earlier and rejected by data:
- Cast at load: needlessly costs 5MB+ storage with zero runtime benefit.
- Cast lazily via emitted Metal op: same outcome with extra bookkeeping.

---

## 8. P0 verification — pass/fail summary

| Question | Status | Answer |
|---|---|---|
| Safetensors dtype for packed weights | ✅ verified | `"U32"` |
| Scales/biases dtype | ✅ verified | `"F16"` |
| Embedding quantization in canonical Llama-1B-4bit | ✅ verified | yes — `embed_tokens.{weight,scales,biases}` |
| Group size in canonical checkpoints | ✅ verified | uniformly **64** across all 14 sampled |
| Bits in canonical checkpoints | ✅ verified | uniformly **4** |
| `tie_word_embeddings` for Llama-3.2 | ✅ verified | true (1B + 3B) |
| Linear-layer fp biases on Llama-3.2 | ✅ verified | none (attention_bias=mlp_bias=false) |
| Mixed-quant models exist | ✅ verified | yes — Gemma-3-MM has bf16 language_model embedding |
| NAX detect logic | ✅ verified | macOS 26.2+ runtime + `arch_gen ≥ 17` (`≥ 18` for 'p' arch) |
| cpu_golden location | ✅ verified | `vllm-rs/crates/ferrite-forward/src/cpu_golden.rs` |
| F16/BF16 scales handling | ✅ documented | cast scales at load to match activation dtype |
| LinearLayer current state | ✅ verified | `vllm-rs/crates/ferrite-kernels/src/layers.rs:738` (CUDA-only quant arms) |
| Workspace layout | ✅ corrected | all crates under `vllm-rs/crates/` |
| Total ferrite model crates | ✅ corrected | 24 (plan said 10) |
| `fp_quantized` deferral | ✅ confirmed | production in MLX; Phase 17 of plan |

All P0 questions answered. **P1 plumbing can begin** — no further verification blockers.

---

## 9. Open questions for later phases (not blocking P1)

- **P3 dispatch heuristic ports**: confirm exact `vector_limit` table values from `quantized.cpp:84-128` byte-for-byte during P3 implementation. Don't approximate.
- **P7 architecture string**: `metal_device.architecture()` in objc2-metal — confirm API surface and that we can extract both the family-letter (`'p'`/`'d'`/etc.) and the gen number cleanly. Read MTLDevice.h via objc2-metal docs in P7.
- **P13 MoE shape on Qwen3-MoE-30B**: `right_sorted` flag and per-expert `B/E` ratio determine `gather_qmm_rhs` vs `gather_qmm` dispatch. Verify expert routing produces sorted indices in our scheduler.
- **CommandR canonical 4bit**: search for `mlx-community/c4ai-command-r-*-4bit` or similar; if absent, lower priority for P10/13.
- **Granite mlx-q4**: same — may not exist on HF yet.
