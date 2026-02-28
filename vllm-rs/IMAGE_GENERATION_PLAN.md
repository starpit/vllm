# Image & Video Generation: Feasibility & Plan

> **Created**: 2026-02-28 | **Updated**: 2026-02-28
> Research context: vllm-omni, Cosmos-Predict1, Python vLLM multimodal

## Background

vLLM (Python and Rust) is designed around **autoregressive text generation**. Image/video generation is a fundamentally different problem with two distinct approaches:

### Approach A: Diffusion-based (vllm-omni)

Models like Wan2.2, Flux, SD3 use iterative denoising (DiT / UNet). vllm-omni wraps Python vLLM as a dependency and bolts on a parallel diffusion engine with its own scheduling, parallelism, and output pipeline. The AR text engine and DiT engine are essentially two separate systems behind one API server.

- **Pros**: Production-proven models (Wan2.2 T2V, Flux/SD3 T2I), mature tooling (diffusers), active community
- **Cons**: Entirely different compute pattern (iterative denoising vs. autoregressive), no shared infrastructure with LLM serving beyond GPU parallelism primitives, requires a full diffusion scheduler + VAE decoder

### Approach B: Autoregressive visual tokens (Cosmos-style)

Models like NVIDIA Cosmos-Predict1 tokenize images/video into discrete visual tokens (via VQ-VAE) and predict them autoregressively with a standard transformer. The AR backbone looks like a text LLM (self-attention + FFN), but the pipeline requires:

1. **Visual tokenizer** (VQ-VAE encoder): image/video frames -> discrete token IDs
2. **AR transformer**: predicts next visual token (this part reuses LLM infrastructure)
3. **Visual detokenizer** (VQ-VAE decoder or diffusion decoder): token IDs -> image/video frames

- **Pros**: The AR transformer reuses existing vllm-rs infrastructure (KV cache, paging, sampling, batching). Cosmos-Predict1-4B config is a standard transformer (4096 dim, 32 heads, 16 layers, GQA with 8 KV heads, RMSNorm)
- **Cons**: Requires visual tokenizer/detokenizer (separate models), 3D RoPE for video, `model.pt` weight format (not safetensors), visual vocab (64K tokens) is same size as text but semantically different, end-to-end pipeline needs ~27-31 GB VRAM minimum

### Python vLLM's position

The vLLM project maintainers have explicitly stated ([#17106](https://github.com/vllm-project/vllm/issues/17106)):
- **No plans to support diffusion models** — "there is little performance gain in doing so, you might as well just use transformers library"
- **AR visual generation is welcome** — but no one has landed it; a Cosmos PR (#11968) went stale
- **vllm-omni** (`vllm-project/vllm-omni`) is the official spin-off for diffusion/omni generation

### Recommendation for vllm-rs

Approach B (AR visual tokens) is the natural fit. The AR transformer is already something we can serve; the new work is the tokenizer/detokenizer pipeline and 3D positional encoding. Diffusion (Approach A) would be a separate engine with no shared compute path — equivalent to building a new project.

---

## Implementation Progress

| Phase | Status | Details |
|-------|--------|---------|
| **V0. Landscape research** | **DONE** | Surveyed vllm-omni, Cosmos-Predict1, Python vLLM #17106. This document. |
| V1. Weight conversion | Not started | `model.pt` → safetensors converter + config.json |
| V2. 3D RoPE | Not started | Spatial + temporal positional encoding (candle + MLX) |
| V3. Cosmos AR backbone | Not started | `CosmosForCausalLM` model (candle + MLX) |
| V4. Visual tokenizer (Rust) | Not started | Pure Rust port of the Cosmos discrete tokenizer |
| V5. Image generation API | Not started | `/v1/images/generations` endpoint |
| V6. Video generation API | Not started | `/v1/videos` endpoint, MP4 encoding |
| V7. End-to-end pipeline | Not started | Tokenizer → AR → Detokenizer orchestration |

---

## Phase V0: Landscape Research (DONE)

Completed as part of this document. Key findings:

- vllm-omni is a **wrapper** around Python vLLM that monkey-patches internals and adds a parallel diffusion engine. It is not a fork.
- vllm-omni's video models (Wan2.2) are diffusion-based (3D DiT), not autoregressive. Entirely different compute pattern.
- vllm-omni adds: `POST /v1/videos` (base64 MP4 response, no streaming), `POST /v1/images/generations`, diffusion scheduling, DiT cache acceleration (CacheDiT/TeaCache), distributed connectors (SharedMemory/Mooncake/YuanRong), and parallelism for DiT (TP/PP/DP/SP/CFG).
- Cosmos-Predict1-4B is an AR transformer (standard self-attn + FFN + GQA) that predicts discrete visual tokens. The transformer itself is servable by existing vllm-rs infrastructure.
- Cosmos requires a separate visual tokenizer (VQ-VAE) for encode/decode and uses 3D RoPE.
- Cosmos weights ship as `model.pt` (PyTorch pickle), not safetensors. Would need conversion or a `.pt` loader.
- Neither Python vLLM nor vllm-omni supports Cosmos.
- Python vLLM has mature **video understanding** (video-in, text-out) for 20+ models but no video generation.

---

## Cosmos-Predict1 Detailed Architecture

> Source: `~/git/cosmos/cosmos-predict1/cosmos_predict1/autoregressive/`

### Model Variants

| Model | Layers | Dim | Heads | KV Heads | FFN | Params | HF Downloads |
|-------|--------|-----|-------|----------|-----|--------|-------------|
| Cosmos-Predict1-4B | 16 | 4096 | 32 | 8 | 14336 | ~4B | ~8 |
| Cosmos-Predict1-12B | 40 | 5120 | 40 | 8 | ? | ~12B | gated |
| Cosmos-Predict1-5B-Video2World | 16 | 4096 | 32 | 8 | 14336 | ~5B | ~46 |
| Cosmos-Predict1-13B-Video2World | 40 | 5120 | 40 | 8 | ? | ~13B | gated |
| Cosmos-Predict1-7B/14B-Text2World | — | — | — | — | — | Mistral-Nemo backbone | gated |

The 4B/5B variants are the primary targets — they fit in 16 GB unified memory (bf16) for Apple Silicon / MLX.

### AR Transformer Architecture

> Source: `autoregressive/networks/transformer.py`, `autoregressive/modules/attention.py`, `autoregressive/modules/mlp.py`

The transformer is structurally identical to LLaMA with two additions (3D RoPE, QK norm):

```
tok_embeddings: Embedding(vocab_size, dim)

for each layer [0..n_layers]:
    attention_norm: RMSNorm(dim, eps=1e-5)
    attention:
        wq: Linear(dim, n_heads * head_dim, bias=false)
        wk: Linear(dim, n_kv_heads * head_dim, bias=false)
        wv: Linear(dim, n_kv_heads * head_dim, bias=false)
        wo: Linear(n_heads * head_dim, dim, bias=false)
        q_norm: RMSNorm(head_dim)  # always enabled for Cosmos
        k_norm: RMSNorm(head_dim)  # always enabled for Cosmos
    ffn_norm: RMSNorm(dim, eps=1e-5)
    feed_forward:
        w1: Linear(dim, ffn_hidden_size, bias=false)  # gate
        w3: Linear(dim, ffn_hidden_size, bias=false)  # up
        w2: Linear(ffn_hidden_size, dim, bias=false)   # down
    # forward: x + attn(norm(x)); h + silu(w1(norm(h))) * w3(norm(h)) via w2

norm: RMSNorm(dim, eps=1e-5)  # final
output: Linear(dim, vocab_size, bias=false)  # lm_head, NOT tied to embedding
```

**MLP**: SwiGLU — `w2(silu(w1(x)) * w3(x))` — identical to LLaMA.

**Attention**: Standard GQA with `repeat_interleave` for KV head expansion. QK normalization is applied per-head *before* RoPE. KV cache is pre-allocated `[max_batch, n_kv_heads, max_seq_len, head_dim]` and updated by position index.

**Weight naming** (maps to our loading):
| Cosmos name | Equivalent |
|-------------|-----------|
| `tok_embeddings.weight` | embedding |
| `layers.{i}.attention.wq.weight` | q_proj |
| `layers.{i}.attention.wk.weight` | k_proj |
| `layers.{i}.attention.wv.weight` | v_proj |
| `layers.{i}.attention.wo.weight` | o_proj |
| `layers.{i}.attention.q_norm.weight` | q_norm |
| `layers.{i}.attention.k_norm.weight` | k_norm |
| `layers.{i}.attention_norm.weight` | input_layernorm |
| `layers.{i}.feed_forward.w1.weight` | gate_proj |
| `layers.{i}.feed_forward.w2.weight` | down_proj |
| `layers.{i}.feed_forward.w3.weight` | up_proj |
| `layers.{i}.ffn_norm.weight` | post_attention_layernorm |
| `norm.weight` | model final norm |
| `output.weight` | lm_head |

### Config Fields

> Source: `autoregressive/configs/base/model.py` — `ModelConfig` dataclass

```python
dim: int = 4096
n_layers: int = 32
n_heads: int = 32
n_kv_heads: int = 8
head_dim: Optional[int] = None        # defaults to dim // n_heads = 128
vocab_size: int = 128256               # base vocab; expanded to 64000+ for video
ffn_hidden_size: int = 14336
norm_eps: float = 1e-5
norm_type: str = "rmsnorm"
rope_theta: float = 500000
rope_dim: str = "1D"                   # "1D" or "3D"
apply_yarn: bool = False
yarn_scale: float = None
yarn_beta_fast: int = None             # high_freq_factor
yarn_beta_slow: int = None             # low_freq_factor
original_seq_len: int = None
use_qk_normalization: bool = False     # True for Cosmos video models
apply_abs_pos_emb: bool = False
video_latent_shape: list = None        # [T, H, W] e.g. [8, 24, 40]
original_latent_shape: list = None     # for YaRN scaling e.g. [3, 40, 64]
pad_to_multiple_of: int = None         # pad position embeddings
precision: str = "bfloat16"
pytorch_rope_version: str = "v2"       # TransformerEngine-style rotation
```

The JSON `config.json` we generate at conversion should map these fields to HF-style names.

### Inference Flow

> Source: `autoregressive/model.py` — `AutoRegressiveModel.generate()`, `autoregressive/utils/sampling.py`

1. **Tokenize**: Conditioning video frames → discrete VQ encoder → flat token sequence `[T*H*W]`
2. **Prefill**: Feed all conditioning tokens at once (standard prefill), get first predicted token
3. **Decode loop**: For `max_gen_len - 1` steps:
   - Feed last predicted token + position → get logits → sample next token
   - Default sampling: `temperature=0.6`, `top_p=0.9` (nucleus sampling)
   - Stop if all sequences hit stop tokens (but for video, typically generate fixed length)
4. **Detokenize**: Predicted token IDs → VQ decoder → video frames

Position tracking: `input_pos` is a 1D tensor of flat indices into the sequence. For 3D RoPE, the cos/sin cache is pre-built for the full `[T*H*W]` grid and indexed by these flat positions. The model doesn't explicitly track (t,h,w) at runtime — it's baked into the RoPE cache.

---

## Phase V1: Weight Conversion

**Goal**: Convert `model.pt` to safetensors + generate `config.json`.

**File**: `scripts/convert_cosmos_predict1.py`

HF repos ship `model.pt` (PyTorch pickle) only. No community safetensors or GGUF conversions exist.

```python
# Load checkpoint
checkpoint = torch.load("model.pt", map_location="cpu", mmap=True, weights_only=True)
weights = checkpoint["model"] if "model" in checkpoint else checkpoint

# Strip "model." prefix (Cosmos wraps weights under model.*)
weights = {k.removeprefix("model."): v for k, v in weights.items()}

# Save as safetensors
save_file(weights, "model.safetensors")
```

Also emit a `config.json` with `"architectures": ["CosmosPredict1ForCausalLM"]` and all the model hyperparameters mapped from the Cosmos `ModelConfig`.

---

## Phase V2: 3D Rotary Position Embeddings

**Goal**: Implement 3D RoPE for both candle and MLX backends.

> Source: `autoregressive/modules/embedding.py` — `RotaryPositionEmbedding`, `RotaryPositionEmbeddingPytorchV2`

### V2a. Dimension Split

For `head_dim=128` with `rope_dim="3D"`:
```
dim_h = head_dim // 6 * 2 = 42   (spatial height, also used for width)
dim_t = head_dim - 2 * dim_h = 44  (temporal)
# Total: 42 (H) + 42 (W) + 44 (T) = 128
```

### V2b. Inverse Frequency Computation

```python
# Spatial frequencies (shared for H and W dimensions)
spatial_inv_freq = 1.0 / (theta ** (arange(0, dim_h, 2)[:dim_h//2] / dim_h))
# → shape [21] for dim_h=42

# Temporal frequencies
temporal_inv_freq = 1.0 / (theta ** (arange(0, dim_t, 2)[:dim_t//2] / dim_t))
# → shape [22] for dim_t=44
```

### V2c. YaRN Scaling (Optional)

When `apply_yarn=true`, frequency-dependent scaling is applied independently to spatial and temporal frequencies:

```python
# Spatial: use original_latent_shape[1] (original H) as reference length
scale_factors_spatial = get_scale_factors(spatial_inv_freq, original_latent_shape[1])
spatial_inv_freq *= scale_factors_spatial

# Temporal: use original_latent_shape[0] (original T) as reference length
scale_factors_temporal = get_scale_factors(temporal_inv_freq, original_latent_shape[0])
temporal_inv_freq *= scale_factors_temporal

# Magnitude scaling
mscale = (0.1 * log(scale) + 1.0) * attn_factor
```

The `get_scale_factors` function is the standard YaRN smooth mask:
```python
high_freq_cutoff = 2π * beta_fast / original_seq_len
low_freq_cutoff = 2π * beta_slow / original_seq_len
smooth_mask = clamp((freq - low_freq_cutoff) / (high_freq_cutoff - low_freq_cutoff), 0, 1)
scale_factors = (1 - smooth_mask) / scale + smooth_mask
```

### V2d. 3D Embedding Construction

For latent shape `[T, H, W]` (e.g., `[8, 24, 40]`):

```python
half_emb_t = outer(seq[:T], temporal_inv_freq)   # [T, dim_t/2] = [8, 22]
half_emb_h = outer(seq[:H], spatial_inv_freq)    # [H, dim_h/2] = [24, 21]
half_emb_w = outer(seq[:W], spatial_inv_freq)    # [W, dim_h/2] = [40, 21]

# Broadcast each across all other dimensions, then duplicate for sin/cos halves
emb = cat([
    repeat(half_emb_t, "t d -> t h w d", h=H, w=W),   # [T,H,W, 22]
    repeat(half_emb_h, "h d -> t h w d", t=T, w=W),   # [T,H,W, 21]
    repeat(half_emb_w, "w d -> t h w d", t=T, h=H),   # [T,H,W, 21]
] * 2, dim=-1)  # duplicate → [T,H,W, 128]
# The *2 produces [half_t, half_h, half_w, half_t, half_h, half_w]

emb = reshape(emb, "(t h w) 1 1 d")  # flatten to [T*H*W, 1, 1, head_dim]
```

For text-to-video mode: prepend a zero embedding for the `<bov>` token.

### V2e. RoPE Application — TransformerEngine Convention

Cosmos uses the TE-style rotation (pairs adjacent elements), NOT the half-split convention used by LLaMA:

```python
def _rotate_half_te(x):
    # Reshape [..., d] to [..., 2, d//2], unbind to (x1, x2)
    x = x.view(x.shape[:-1] + (2, x.shape[-1] // 2))
    x1, x2 = x.unbind(dim=-2)
    return cat((-x2, x1), dim=-1)

def apply_rope(t, cos, sin):
    rot_dim = cos.shape[-1]
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t_rot = t_rot * cos + _rotate_half_te(t_rot) * sin
    return cat((t_rot, t_pass), dim=-1)
```

Pre-compute cos/sin from the embedding:
```python
cos_cached = cos(emb) * mscale   # [1, T*H*W, 1, head_dim]
sin_cached = sin(emb) * mscale   # [1, T*H*W, 1, head_dim]
```

At runtime, index by `input_pos`: `cos_cached[:, input_pos, :, :]`.

### V2f. Candle Implementation

New file: `crates/vllm-models/src/rope3d.rs`

```rust
pub struct RotaryEmbedding3D {
    cos_cached: Tensor,  // [1, total_positions, 1, head_dim]
    sin_cached: Tensor,  // [1, total_positions, 1, head_dim]
    mscale: f64,
}

impl RotaryEmbedding3D {
    pub fn new(
        head_dim: usize,
        latent_shape: [usize; 3],    // [T, H, W]
        rope_theta: f64,
        yarn: Option<YarnConfig3D>,   // optional YaRN params
    ) -> Result<Self>;

    pub fn apply(
        &self,
        q: &Tensor,  // [batch, seq_len, n_heads, head_dim]
        k: &Tensor,
        input_pos: &Tensor,  // [seq_len] — flat indices into T*H*W grid
    ) -> Result<(Tensor, Tensor)>;
}
```

### V2g. MLX Implementation

New file: `crates/vllm-mlx/src/models/rope3d.rs`

Uses `mlx_rs` array ops. The cos/sin cache is a plain `Array` indexed at forward time. `nn::Rope` doesn't support 3D, so this is a custom implementation:

```rust
pub struct MlxRotaryEmbedding3D {
    cos_cached: Array,  // [total_positions, head_dim]
    sin_cached: Array,  // [total_positions, head_dim]
}
```

Apply via `mlx_rs` element-wise ops matching the TE rotation convention.

---

## Phase V3: Cosmos AR Backbone

**Goal**: Implement `CosmosForCausalLM` for both candle and MLX.

### V3a. Candle Implementation

**File**: `crates/vllm-models/src/cosmos.rs`

Reuses existing components:
- `LlamaMLP` (`crates/vllm-models/src/llama.rs`) — SwiGLU is identical
- `RmsNorm` — standard
- `attention_with_cache()` (`crates/vllm-models/src/attention.rs`) — unified KV cache helper
- `KvCacheStorage` / `LayerKvHandle` — works identically

New logic:
- `RotaryEmbedding3D` from Phase V2
- QK norm: apply `RmsNorm` per head to Q and K after projection, before RoPE
- Weight loading with Cosmos naming (see weight table above)

Register in `crates/vllm-models/src/registry.rs`:
```rust
self.register("CosmosPredict1ForCausalLM", crate::cosmos::create_cosmos);
```

### V3b. MLX Implementation

**File**: `crates/vllm-mlx/src/models/cosmos.rs`

Follows the same pattern as `MlxLlamaForCausalLM`:
- `nn::Linear`, `nn::RmsNorm`, `nn::Embedding`
- `fast::scaled_dot_product_attention`
- `MlxRotaryEmbedding3D` from Phase V2g
- Register in `MlxModelRegistry`

### V3c. Sampling Adaptation

Cosmos uses standard top-p / top-k sampling — the existing `Sampler` works as-is.

Key differences from text:
- **Fixed-length generation**: No EOS detection. Generate exactly `T_new * H * W` tokens.
  - Engine needs a mode where `max_tokens` is the sole stopping criterion and `ignore_eos=true`.
- **Default parameters**: `temperature=0.6`, `top_p=0.9` (not `temperature=1.0`)
- **No classifier-free guidance**: Cosmos Predict1 does NOT use CFG (confirmed from source). Single forward pass per step.

---

## Phase V4: Visual Tokenizer — Pure Rust Port

**Goal**: 100% Rust implementation of the Cosmos discrete video tokenizer, supporting both candle and MLX backends. No Python dependency.

> Source: `~/git/cosmos/cosmos-predict1/cosmos_predict1/tokenizer/`

### V4a. Architecture Overview

The tokenizer is a **Causal Discrete Video Tokenizer** using FSQ (Finite Scalar Quantization):

```
ENCODER: video [B, 3, T, H, W]
  → Patcher3D (Haar wavelet decomposition, patch_size=4)
  → Encoder3D (3D ResBlocks + spatial/temporal attention + downsampling)
  → quant_conv (CausalConv3d 1×1×1)
  → FSQuantizer → discrete indices [B, T', H', W']

DECODER: indices [B, T', H', W']
  → InvQuantizer (indices → continuous codes)
  → post_quant_conv (CausalConv3d 1×1×1)
  → Decoder3D (3D ResBlocks + spatial/temporal attention + upsampling)
  → UnPatcher3D (inverse Haar wavelet)
  → video [B, 3, T, H, W]
```

Compression ratio: `[8, 16, 16]` (temporal × height × width).
For 384×640 input with 33 frames: latent shape = `[5, 24, 40]` → 4,800 tokens per chunk.

### V4b. Component Inventory (~2,400 lines of Python to port)

| Component | Source File | Lines | Complexity |
|-----------|-----------|-------|-----------|
| CausalConv3d | `modules/layers3d.py:50` | ~40 | Causal temporal padding + nn.Conv3d |
| CausalConv3dTranspose | `modules/layers3d.py` | ~30 | Transposed variant for upsampling |
| ResBlock3D | `modules/layers3d.py` | ~60 | residual(norm→act→conv3d→norm→act→conv3d) + shortcut |
| SpatialAttention | `modules/layers3d.py` | ~50 | Single-head attention over H×W (time→batch) |
| TemporalAttention | `modules/layers3d.py` | ~50 | Single-head attention over T (space→batch) |
| Downsample3D | `modules/layers3d.py` | ~30 | Spatial 2× downsample (strided conv) |
| Upsample3D | `modules/layers3d.py` | ~30 | Spatial 2× upsample (nearest + conv) |
| TimeDownsample | `modules/layers3d.py` | ~20 | Temporal 2× downsample (strided conv) |
| TimeUpsample | `modules/layers3d.py` | ~20 | Temporal 2× upsample (nearest + conv) |
| Encoder3D | `modules/layers3d.py` | ~80 | Stack of ResBlocks + Attn + Downsample |
| Decoder3D | `modules/layers3d.py` | ~80 | Stack of ResBlocks + Attn + Upsample |
| Patcher3D | `modules/patching.py` | ~100 | Haar wavelet 3D decomposition |
| UnPatcher3D | `modules/patching.py` | ~80 | Inverse Haar wavelet 3D reconstruction |
| FSQuantizer | `modules/quantizers.py` | ~170 | Finite Scalar Quantization with levels `[8,8,8,5,5,5]` |
| InvQuantizer | `modules/quantizers.py` | ~30 | Index → continuous code lookup |
| CausalNormalize | `modules/utils.py` | ~30 | Group norm with causal masking |
| CausalDiscreteVideoTokenizer | `networks/discrete_video.py` | ~120 | Top-level module wiring it all together |

### V4c. FSQ Quantizer Details

FSQ replaces traditional VQ codebooks with bounded rounding. Levels `[8, 8, 8, 5, 5, 5]` give `8*8*8*5*5*5 = 64,000` codewords.

```python
# Quantize: bound to [-half_width, +half_width], round to nearest integer
def quantize(z):
    half_width = levels // 2
    return round(bound(z)) / half_width  # normalized to [-1, 1]

# Indices: scale to [0, level-1] per dimension, compute flat index via mixed-radix basis
basis = cumprod([1, 8, 8, 8, 5, 5])  # [1, 8, 64, 512, 2560, 12800]
index = sum(scaled_code * basis)       # single integer in [0, 64000)

# Inverse: decompose index back to per-dimension codes
codes = (index // basis) % levels
```

This is pure arithmetic — no codebook embedding table needed. Easy to port to Rust.

### V4d. Haar Wavelet Patching

The `Patcher3D` applies a 3D Haar discrete wavelet transform as a preprocessing step, converting `[B, C, T, H, W]` with `patch_size=4` into a higher-channel, lower-resolution representation:

```python
# Each 4×4×4 patch is decomposed into 64 wavelet coefficients
# Output channels = input_channels * patch_size^3 = 3 * 64 = 192
# Spatial dims reduced by patch_size: H/4, W/4, T/4
```

The Haar transform is separable (1D transforms along T, H, W independently) and can be implemented as matrix multiplications or strided gather operations. The `UnPatcher3D` is the exact inverse.

### V4e. Weight Loading for Tokenizer

Tokenizer weights ship as TorchScript `.jit` files on HuggingFace (`encoder.jit`, `decoder.jit`). However, the `DiscreteVideoFSQStateDictTokenizer` class in the Cosmos source shows how to extract `state_dict` from JIT back into regular nn.Modules:

```python
encoder_sd = torch.jit.load("encoder.jit").state_dict()
# Filter out JIT-captured constants:
# - encoder.patcher3d.wavelets, encoder.patcher3d._arange, encoder.patcher3d.patch_size_buffer
# - quantizer._levels, quantizer._basis, quantizer.implicit_codebook
tokenizer_module.load_state_dict(encoder_sd)
```

**Strategy for Rust**:
1. Write a Python conversion script (`scripts/convert_cosmos_tokenizer.py`) that loads the `.jit` files, extracts `state_dict()`, and saves as safetensors.
2. Load the safetensors in Rust with the known architecture.
3. The JIT-captured buffers (`_levels`, `_basis`, `implicit_codebook`, wavelet matrices) are derived constants and can be recomputed in Rust from the config parameters.

### V4f. Candle Tokenizer Implementation

**New crate**: `crates/vllm-visual-tokenizer/` (or module in `vllm-models`)

The main challenge is **3D convolutions**: candle-core does not have `Conv3d`. Options:

1. **Implement Conv3d via im2col + matmul**: Unfold the 5D input into a 2D matrix, multiply by the reshaped weight, fold back. This is how most frameworks implement convolutions under the hood. ~200 lines of Rust.

2. **Factored as Conv2d over time slices**: Since Cosmos uses factorized spatial+temporal convolutions in many places, some layers can be decomposed. But `CausalConv3d` is truly 3D in several places.

3. **Use the `candle-nn` Conv2d for spatial layers + manual temporal dimension**: The architecture factorizes many ops into `time2batch` → Conv2d → `batch2time`, which maps well to existing candle ops.

Recommended: Option 1 (im2col Conv3d) as a general solution, with factored spatial/temporal fast paths.

### V4g. MLX Tokenizer Implementation

MLX (`mlx-rs`) also lacks `Conv3d`, so the same im2col approach applies. However, MLX's lazy evaluation makes this less costly — the unfold+matmul fuses into a single Metal kernel.

Alternatively, MLX Python has `mlx.core.conv_general` which supports arbitrary dimensions. If `mlx-rs` exposes this, Conv3d becomes a one-liner.

### V4h. Tokenizer Memory Budget

The tokenizer encoder+decoder together are ~400M parameters (typical for VQ-VAE at 128 base channels with `[2,4,4]` channel multipliers). At bf16 this is ~800 MB.

For Apple Silicon with 16 GB unified memory:
- AR model (4B, bf16): ~8 GB
- Tokenizer: ~800 MB
- KV cache + activations: ~2 GB
- Total: ~11 GB — fits with headroom

For 32 GB Macs: comfortably fits the 12B model too.

---

## Phase V5: Image Generation API

**Goal**: Serve image generation via an OpenAI-compatible endpoint.

### V5a. Protocol types

```rust
// POST /v1/images/generations
pub struct ImageGenerationRequest {
    pub prompt: Option<String>,        // text prompt (for text-conditioned models)
    pub image: Option<String>,         // base64 input image (for image-conditioned)
    pub model: Option<String>,
    pub n: Option<u32>,
    pub size: Option<String>,          // "1024x640"
    pub response_format: Option<String>, // "b64_json" or "url"
}

pub struct ImageGenerationResponse {
    pub created: u64,
    pub data: Vec<ImageData>,
}

pub struct ImageData {
    pub b64_json: Option<String>,
    pub url: Option<String>,
}
```

### V5b. Axum route

- `POST /v1/images/generations` → encode input → AR generate → decode tokens → return base64 PNG/JPEG
- Uses the `image` crate for PNG encoding

---

## Phase V6: Video Generation API

**Goal**: Serve video generation (Cosmos-style future frame prediction).

### V6a. Protocol types

```rust
// POST /v1/videos
pub struct VideoGenerationRequest {
    pub prompt: Option<String>,
    pub video: Option<String>,          // base64 input video or URL
    pub image: Option<String>,          // base64 first frame
    pub model: Option<String>,
    pub n: Option<u32>,
    pub num_frames: Option<u32>,        // output frame count
    pub fps: Option<u32>,
    pub size: Option<String>,           // "1024x640"
}

pub struct VideoGenerationResponse {
    pub created: u64,
    pub data: Vec<VideoData>,
}

pub struct VideoData {
    pub b64_json: String,               // base64-encoded MP4
}
```

### V6b. Video assembly

- Collect decoded frames → encode as MP4 via `ffmpeg-next` crate or minimal MP4 muxer
- Return as base64 in response (matching vllm-omni's approach; no streaming initially)
- Crate candidates: `ffmpeg-next` (full ffmpeg bindings), `mp4` (pure Rust MP4 writer), `image` for individual frames

---

## Phase V7: End-to-End Pipeline

**Goal**: Wire tokenizer → AR model → detokenizer into a working generation pipeline.

### V7a. Pipeline orchestration

```rust
pub struct VisualGenerationPipeline {
    tokenizer: Box<dyn VisualTokenizer>,  // encoder + decoder
    engine: Arc<AsyncEngine>,              // AR model
    latent_shape: [usize; 3],             // [T, H, W]
    compression_ratio: [usize; 3],        // [8, 16, 16]
}

impl VisualGenerationPipeline {
    pub async fn generate_video(
        &self,
        conditioning_frames: &[ImageFrame],
        num_output_frames: usize,
        sampling: SamplingParams,
    ) -> Result<Vec<ImageFrame>>;
}
```

### V7b. Position Mapping

The AR model sees a flat sequence of tokens, but positions map to 3D grid coordinates for RoPE:

```
Token index 0    → (t=0, h=0, w=0)
Token index 1    → (t=0, h=0, w=1)
Token index W-1  → (t=0, h=0, w=W-1)
Token index W    → (t=0, h=1, w=0)
...
Token index H*W-1     → (t=0, h=H-1, w=W-1)
Token index H*W       → (t=1, h=0, w=0)
...
Token index T*H*W - 1 → (t=T-1, h=H-1, w=W-1)
```

The 3D RoPE cos/sin cache is pre-built for all `T*H*W` positions. At runtime, `input_pos` is a simple `[0, 1, 2, ..., T*H*W-1]` for prefill, and increments by 1 for each decode step. No explicit 3D coordinate tracking needed — it's baked into the cache.

### V7c. Fixed-Length Generation

For video-to-video:
- Conditioning tokens: `T_cond * H * W` (encoded from input frames)
- Generation tokens: `T_gen * H * W` (one chunk worth of new frames)
- Total sequence: `(T_cond + T_gen) * H * W`

For text-to-video: prepend a `<bov>` (beginning-of-video) token, then generate the full grid.

The engine needs a "generate exactly N tokens" mode where EOS is ignored. This is a minor change — set `ignore_eos=true` and `max_tokens=N` in the request.

### V7d. Memory Management

| Component | Size (4B, bf16) | Notes |
|-----------|----------------|-------|
| AR model | ~8 GB | Standard transformer weights |
| Tokenizer encoder | ~400 MB | Only needed during encode |
| Tokenizer decoder | ~400 MB | Only needed during decode |
| KV cache (1 req) | ~2 GB | For max sequence length |
| Activations | ~1 GB | Forward pass intermediates |
| **Total** | **~12 GB** | Fits 16 GB Apple Silicon |

Strategy: Load both tokenizer and AR model into memory. For memory-constrained systems, the tokenizer can be offloaded to CPU between encode/decode phases since it's only used at the start and end of generation.

---

## Files to Create / Modify

### New Files

| File | Purpose |
|------|---------|
| `scripts/convert_cosmos_predict1.py` | model.pt → safetensors converter |
| `scripts/convert_cosmos_tokenizer.py` | encoder.jit/decoder.jit → safetensors |
| `crates/vllm-models/src/cosmos.rs` | Candle CosmosForCausalLM |
| `crates/vllm-models/src/rope3d.rs` | Candle 3D RoPE |
| `crates/vllm-mlx/src/models/cosmos.rs` | MLX CosmosForCausalLM |
| `crates/vllm-mlx/src/models/rope3d.rs` | MLX 3D RoPE |
| `crates/vllm-models/src/visual_tokenizer.rs` | Candle visual tokenizer (Conv3d, FSQ, Haar) |
| `crates/vllm-mlx/src/visual_tokenizer.rs` | MLX visual tokenizer |
| `crates/vllm-models/src/conv3d.rs` | Candle Conv3d implementation (im2col) |

### Modified Files

| File | Change |
|------|--------|
| `crates/vllm-models/src/lib.rs` | Add `pub mod cosmos, rope3d, visual_tokenizer, conv3d;` |
| `crates/vllm-models/src/registry.rs` | Register `CosmosPredict1ForCausalLM` |
| `crates/vllm-mlx/src/models/mod.rs` | Add cosmos model + register in `MlxModelRegistry` |
| `crates/vllm-serve/src/server.rs` | Add `/v1/images/generations`, `/v1/videos` routes |
| `crates/vllm-protocol/src/lib.rs` | Add image/video generation protocol types |
| `crates/vllm-serve/src/init.rs` | Handle Cosmos model init (visual tokenizer, no text tokenizer) |
| `crates/vllm-engine/src/engine_core.rs` | Fixed-length generation mode (ignore EOS) |

### Reusable Existing Code

| Component | Location | Reuse |
|-----------|----------|-------|
| `LlamaMLP` | `vllm-models/src/llama.rs` | SwiGLU is identical |
| `RmsNorm` | `vllm-models/src/llama.rs` | Standard RMSNorm |
| `attention_with_cache()` | `vllm-models/src/attention.rs` | Unified attention helper |
| `KvCacheStorage` | `vllm-models/src/lib.rs` | KV cache abstraction |
| `Sampler` | `vllm-models/src/sampler.rs` | top-p, top-k, temperature |
| `ModelRegistry` | `vllm-models/src/registry.rs` | Model registration |
| `MlxKvCache` | `vllm-mlx/src/cache.rs` | MLX KV cache |
| `nn::Linear/RmsNorm/Embedding` | `mlx-rs` | MLX layer primitives |
| `fast::scaled_dot_product_attention` | `mlx-rs` | MLX SDPA |
| Safetensors loading | `vllm-model/src/weight.rs` | Weight file I/O |
| YaRN scaling | `vllm-models/src/llama.rs` | Frequency-dependent RoPE correction |

---

## Open Questions & Risks

1. **Conv3d in Rust**: Neither candle nor mlx-rs has Conv3d. The im2col approach works but needs careful implementation for the causal padding and stride patterns. This is the single largest piece of new infrastructure.

2. **Tokenizer weight extraction**: The `.jit` → safetensors conversion path needs testing. JIT files may contain fused operations that don't cleanly map to a state_dict.

3. **Haar wavelet correctness**: The Patcher3D uses a specific wavelet decomposition that must be pixel-exact to match the original encoder. Any error in the wavelet transform propagates through the entire pipeline.

4. **VRAM on smaller Macs**: The 4B model + tokenizer fits in 16 GB, but the 12B model needs 32 GB+. MLX on 8 GB Macs is not practical.

5. **No community adoption**: Cosmos Predict1 has ~8 downloads on HuggingFace. Nobody has published safetensors, GGUF, or MLX conversions. We'd be the first implementation outside NVIDIA's own code.

6. **Video I/O crates**: Rust's video encoding ecosystem is thin. `ffmpeg-next` requires system FFmpeg. A pure-Rust MP4 muxer exists but may lack codec support.

7. **Text-to-video variants**: The 7B/14B Text2World models use Mistral-Nemo backbones (1D RoPE for text, 3D RoPE for video) with cross-attention for text conditioning. These require additional work beyond the base video-to-video models.

8. **Alternative AR visual models**: LlamaGen (MIT license, standard 1D RoPE LLaMA, image-only) and Chameleon (Meta, safetensors available, HF Transformers native) are simpler targets. However, both also lack community traction and LlamaGen also ships `.pt` only.
