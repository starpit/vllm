// SPDX-License-Identifier: Apache-2.0
//! Declarative multimodal preprocessing metadata.
//!
//! Each per-arch crate (e.g. `ferrite-model-gemma3-mm`,
//! `ferrite-model-qwen2-vl`) exports `pub const PROCESSOR: MmMetadata`
//! describing how the host should:
//!
//! - Pick the placeholder token id from HF config
//!   (`hf_token_id_key` + `hf_token_id_default`).
//! - Resize the input image (`size_policy`).
//! - Compute how many post-encoder tokens to splice per image
//!   (`tokens_per_image`).
//! - Normalize pixels (`preprocess` fn pointer).
//!
//! `#[vision_forward(processor = path::PROCESSOR, ...)]` bakes the
//! declared const into every variant's `FerriteMmRegistration` row.
//! vllm-serve walks the inventory at startup, finds the registration
//! whose `hf_arches` matches the loaded model, reads `mm_metadata`,
//! and threads it into the request path. No arch names appear in
//! ferrite-forward, ferrite-forward-macro, or vllm-serve — only in
//! the per-arch crate that owns the const.

use image::DynamicImage;
use vllm_common::multimodal::ImageData;

/// CPU-side host preprocessing metadata for a multimodal arch.
///
/// Variant-uniform within an arch family: every Qwen2-VL variant
/// preprocesses the same way; the variant config differs only in
/// `d_model`. Declared once per arch crate, baked into every
/// variant's `FerriteMmRegistration`.
#[derive(Copy, Clone)]
pub struct MmMetadata {
    /// HF-config key the placeholder token id is read from. Common
    /// values: `"image_token_id"` (Qwen2-VL `<|image_pad|>`),
    /// `"boi_token_index"` (Gemma3-MM `<start_of_image>`),
    /// `"image_token_index"` (LLaVA-class default).
    ///
    /// The placeholder is what the chat template emits — that token
    /// gets expanded to N copies in the input ids, and the vision
    /// splice overwrites those positions with projected embeddings.
    pub hf_token_id_key: &'static str,
    /// Fallback when `hf_token_id_key` is absent from config.
    pub hf_token_id_default: u32,

    /// How to size the input image before encoding.
    pub size_policy: SizePolicy,

    /// How many post-encoder tokens each image contributes to the
    /// input id stream.
    pub tokens_per_image: TokensPerImage,

    /// Bytes-decoded `DynamicImage` + resolved `(target_h, target_w)`
    /// → CHW-flat `ImageData`. Picked from the primitives in
    /// [`crate::preprocess`].
    pub preprocess: PreprocessFn,

    /// Default fallback when `vision_config.image_size` is absent.
    /// Most arches set this to a sane variant value (224 for SigLIP-
    /// class, 392 for Qwen2-VL); read by serve when the HF config
    /// doesn't carry it.
    pub default_image_size: usize,

    /// HF chat templates may render an image content part as
    /// `{"type": "image", ...}` (Gemma3) or accept either `image` or
    /// `image_url` (Qwen2-VL). Our protocol speaks `image_url`; we
    /// always normalize to `image` before applying the chat template
    /// — universally safe because tolerant templates accept it and
    /// strict ones require it. This field is informational, not a
    /// dispatch knob — kept here so the per-arch declaration is
    /// self-documenting.
    pub chat_template_image_part_type: &'static str,

    /// How the placeholder token id (rendered by the chat template) is
    /// expanded into a token-id sequence and which positions in that
    /// sequence the vision splice writes embeddings into. Different
    /// arches have different conventions; declared per-arch here so
    /// the splice/expand pipeline stays generic.
    pub placeholder_policy: PlaceholderPolicy,

    /// Whether the text decoder consumes 3D MRoPE positions
    /// `[3, n_tokens]` (Qwen2-VL family uses (T, H, W) per token, with
    /// MM tokens carrying grid coords) or standard 1D RoPE positions
    /// `[n_tokens]` (Gemma3-MM, LLaVA, every other family).
    ///
    /// `true` → ferrite_worker overrides the per-token positions with the
    /// 2D-grid build pass before model.forward.
    /// `false` → keeps the prebuilt 1D sequence positions verbatim.
    ///
    /// Picking the wrong value silently corrupts attention: the decoder
    /// either reads `(T, H, W)` rows it doesn't understand, or reads
    /// 1D positions where MRoPE was needed and image tokens collapse
    /// onto a single RoPE state.
    pub mrope_positions: bool,
}

/// Token-id expansion + splice-position policy.
#[derive(Copy, Clone, Debug)]
pub enum PlaceholderPolicy {
    /// Expand each marker token to N copies of itself; splice at every
    /// expanded position. Qwen2-VL family — chat template renders
    /// `<|image_pad|>` × 1, we expand to N copies, the engine splices
    /// vision embeddings into all N positions.
    RepeatMarker,
    /// HF Gemma3 expansion: marker (`<start_of_image>`) → `[wrap, boi, soft × N, eoi, wrap]`.
    /// Splice at the N soft positions. The model was trained on this exact
    /// bracketed structure (boi + soft + eoi sentinels), so a flat
    /// `RepeatMarker` of boi gives garbled output.
    BoiSoftEoiWrap {
        soft_token_id: u32,
        eoi_token_id: u32,
        /// "Wrap" is the `\n\n`-equivalent in the tokenizer (Gemma3
        /// encodes `\n\n` as a single special token, id 108). One
        /// instance lands before the boi and one after the eoi.
        wrap_token_id: u32,
    },
}

/// CPU preprocess function pointer.
///
/// Signature: `(img, target_h, target_w) -> ImageData`. The target
/// dims are resolved by the caller from [`SizePolicy`] before invoking.
pub type PreprocessFn = fn(&DynamicImage, usize, usize) -> ImageData;

/// How to derive `(target_h, target_w)` for a given input image.
#[derive(Copy, Clone, Debug)]
pub enum SizePolicy {
    /// Fixed square: pass `image_size` from `vision_config.image_size`
    /// (or [`MmMetadata::default_image_size`] when absent) for both
    /// dims. SigLIP / Gemma3-MM / LLaVA-class.
    FixedSquare,
    /// Smart-resize: round each dim to a multiple of `factor`, clamp
    /// total pixels to `[min_pixels, max_pixels]`. The host reads
    /// min/max from HF's `preprocessor_config.json` (Qwen2-VL writes
    /// them there); fall back to `default_min_pixels` /
    /// `default_max_pixels` if that file is missing.
    SmartResize {
        factor: u32,
        default_min_pixels: usize,
        default_max_pixels: usize,
    },
}

/// How many post-encoder tokens each image contributes.
#[derive(Copy, Clone, Debug)]
pub enum TokensPerImage {
    /// Read from `hf_config.mm_tokens_per_image`. Fallback:
    /// `(image_size / patch_size)²` (raw per-patch count).
    /// Gemma3-MM ships the count in config as 256 (post-pool);
    /// SigLIP without a pool would fall back to 4096 raw patches.
    FromConfig,
    /// Per-image: `(target_h / (patch_size · spatial_merge)) ·
    /// (target_w / (patch_size · spatial_merge))`. Used by Qwen2-VL,
    /// where post-resize dims vary per image.
    PerImageGrid {
        /// Default `spatial_merge_size` if the HF config doesn't set
        /// one — Qwen2-VL hard-codes 2.
        spatial_merge_default: u32,
    },
}
