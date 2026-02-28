# Image & Video Generation: Feasibility & Plan

> **Created**: 2026-02-28 | Research context: vllm-omni, Cosmos-Predict1, Python vLLM multimodal

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
| V1. Visual tokenizer abstraction | Not started | VQ-VAE encoder/decoder trait, image/video frame I/O |
| V2. 3D RoPE | Not started | Spatial + temporal positional encoding for video transformers |
| V3. Cosmos AR backbone | Not started | `CosmosForCausalLM` model struct, weight loading (model.pt or converted safetensors) |
| V4. Image generation API | Not started | `/v1/images/generations` endpoint, base64 image response |
| V5. Video generation API | Not started | `/v1/videos` endpoint, mp4 encoding, multi-frame output |
| V6. End-to-end pipeline | Not started | Tokenizer -> AR -> Detokenizer orchestration, memory management |

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

## Phase V1: Visual Tokenizer Abstraction

**Goal**: Define a trait for visual tokenizers that can encode image/video frames to discrete token IDs and decode token IDs back to frames.

### V1a. Tokenizer trait

```rust
pub trait VisualTokenizer: Send + Sync {
    /// Encode image frames into discrete visual token IDs.
    fn encode(&self, frames: &[ImageFrame]) -> Result<Vec<u32>>;

    /// Decode visual token IDs back into image frames.
    fn decode(&self, tokens: &[u32]) -> Result<Vec<ImageFrame>>;

    /// Visual vocabulary size.
    fn vocab_size(&self) -> usize;
}
```

### V1b. Image/video frame types

- `ImageFrame`: raw pixel buffer (H x W x C), dtype, colorspace
- Video I/O: decode mp4 -> frames, encode frames -> mp4
- Crate candidates: `image` for still images, `ffmpeg-next` or similar for video

### V1c. VQ-VAE implementation

- Load a VQ-VAE checkpoint (Cosmos uses a custom tokenizer with 64K visual vocab)
- The Cosmos tokenizer is a separate model (`cosmos_tokenizer` Python package) — would need to port or wrap
- Alternative: use ONNX Runtime to run the tokenizer model without porting

### Open questions

- Is the Cosmos visual tokenizer available as standalone weights? (The HF repo only has the AR model)
- Can we use ONNX or TorchScript to run the tokenizer without a full port?
- Are there other AR visual generation models with more accessible tokenizers?

---

## Phase V2: 3D Rotary Position Embeddings

**Goal**: Implement 3D RoPE for video transformers (spatial height, spatial width, temporal).

The Cosmos config specifies `rope_dim: "3D"`. Standard 1D RoPE (used for text) assigns one frequency per position along the sequence dimension. 3D RoPE decomposes the position into (t, h, w) and applies separate frequency bands to each dimension, concatenated into the head dimension.

### V2a. 3D position computation

- Given frame index `t` and spatial patch coordinates `(h, w)`, compute 3D position vector
- Allocate head_dim across 3 dimensions (e.g., head_dim/3 each, or configurable split)

### V2b. 3D RoPE kernel

- Extend existing `RotaryEmbedding` with a `new_3d()` constructor
- cos/sin cache shaped `[max_t, max_h, max_w, head_dim]` or factored per-dimension
- Apply via the same half-split or interleaved convention as existing RoPE

### Dependencies

- Requires understanding Cosmos's exact 3D RoPE factorization (inspect the Python source at `nvidia-cosmos/cosmos-predict1`)

---

## Phase V3: Cosmos AR Backbone

**Goal**: Implement the Cosmos autoregressive transformer as a model in vllm-rs.

The config shows a standard transformer:

```json
{
    "dim": 4096,
    "n_heads": 32,
    "n_kv_heads": 8,
    "n_layers": 16,
    "ffn_hidden_size": 14336,
    "vocab_size": 64000,
    "norm_type": "rmsnorm",
    "norm_eps": 1e-5,
    "rope_dim": "3D"
}
```

This is essentially a LLaMA-like model (GQA, RMSNorm, SwiGLU FFN) with visual token vocabulary and 3D RoPE.

### V3a. Weight loading

- `model.pt` is PyTorch pickle format. Options:
  1. **Convert offline**: Python script to load `model.pt` and save as safetensors (simplest)
  2. **Load .pt directly**: Use `tch-rs` or a pickle parser (complex, fragile)
  3. **GGUF conversion**: Convert to GGUF for quantized inference (if someone publishes one)
- Recommendation: offline conversion script (Phase V3a prerequisite)

### V3b. Model struct

- `CosmosForCausalLM` — reuses `LlamaMLP`, `LlamaAttention` (with 3D RoPE), `RmsNorm`
- Register in `ModelRegistry` under architecture name from config
- `forward()` takes visual token IDs + 3D positions, returns logits over visual vocab

### V3c. Sampling adaptation

- Visual token sampling may differ from text (temperature, top-k over 64K visual vocab)
- Cosmos may use classifier-free guidance (CFG) at inference — need to check
- No EOS / stop criteria in the text sense; generation length is fixed (24 or 32 frames worth of tokens)

---

## Phase V4: Image Generation API

**Goal**: Serve image generation via an OpenAI-compatible endpoint.

### V4a. Protocol types

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

### V4b. Axum route

- `POST /v1/images/generations` -> encode input -> AR generate -> decode tokens -> return base64 PNG/JPEG

---

## Phase V5: Video Generation API

**Goal**: Serve video generation (Cosmos-style future frame prediction).

### V5a. Protocol types

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

### V5b. Video assembly

- Collect decoded frames -> encode as MP4 (via `ffmpeg-next` or similar)
- Return as base64 in response (matching vllm-omni's approach; no streaming initially)

---

## Phase V6: End-to-End Pipeline

**Goal**: Wire tokenizer -> AR model -> detokenizer into a working generation pipeline.

### V6a. Pipeline orchestration

- `VisualGenerationPipeline`: manages the three-stage pipeline
- Memory budget: tokenizer + AR model + detokenizer must fit in VRAM simultaneously (or offload tokenizer/detokenizer)
- AR model uses existing vllm-rs KV cache, paging, and batching

### V6b. Batched visual generation

- Multiple generation requests can share the AR model (same as text batching)
- Visual tokenizer/detokenizer are typically run per-request (not batched across requests)

### V6c. Memory management

- Cosmos-Predict1-4B needs ~27-31 GB without offloading
- Tokenizer/detokenizer can be offloaded to CPU between uses
- AR model stays on GPU for batched inference

---

## Open Questions & Risks

1. **Tokenizer availability**: The Cosmos visual tokenizer is a separate model. Is it available as standalone weights? Can it be run via ONNX?
2. **Weight format**: `model.pt` requires conversion. Is there a community safetensors conversion?
3. **3D RoPE specifics**: Need to inspect the Cosmos Python source for exact factorization.
4. **Fixed-length generation**: Visual generation produces a fixed number of tokens (determined by frame count x patches per frame). This differs from text generation's variable-length stopping.
5. **Classifier-free guidance**: Some visual AR models use CFG at inference, requiring two forward passes per step (conditional + unconditional). Would need parallel forward pass support.
6. **VRAM requirements**: Minimum ~27 GB rules out Apple Silicon (most Macs have 16-32 GB unified memory, shared with OS). MLX backend may not be practical for Cosmos.
7. **Other AR visual models**: Are there smaller or more accessible AR image generation models we could target first? (e.g., LlamaGen, Chameleon, VAR)
8. **Demand signal**: No one has landed AR visual generation in Python vLLM either. Is there sufficient user demand to justify this work?
