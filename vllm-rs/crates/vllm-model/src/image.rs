// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Image decoding and preprocessing for vision-language models.
//!
//! Provides CPU-based image preprocessing shared by both Candle and MLX backends.
//! Images are decoded, resized, normalized, and stored as `ImageData` (flat f32 pixels)
//! before being converted to backend-specific tensors in the workers.

use image::DynamicImage;
use image::imageops::FilterType;
use vllm_common::multimodal::ImageData;

use crate::error::{ModelError, ModelResult};

/// Decode image bytes (JPEG, PNG, WebP) into a `DynamicImage`.
pub fn decode_image(bytes: &[u8]) -> ModelResult<DynamicImage> {
    image::load_from_memory(bytes)
        .map_err(|e| ModelError::Other(format!("failed to decode image: {e}")))
}

/// Preprocess an image for SigLIP: resize to `image_size x image_size`,
/// convert to f32, normalize with `(pixel / 255 - 0.5) / 0.5`.
///
/// Returns `ImageData` with pixels in `[3, H, W]` CHW layout.
pub fn preprocess_siglip(img: &DynamicImage, image_size: usize) -> ImageData {
    let resized = img.resize_exact(
        image_size as u32,
        image_size as u32,
        FilterType::Triangle, // bilinear
    );
    let rgb = resized.to_rgb8();

    let h = image_size;
    let w = image_size;
    let mut pixels = vec![0.0f32; 3 * h * w];

    // Convert HWC -> CHW and normalize.
    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                let normalized = (val - 0.5) / 0.5;
                pixels[c * h * w + y * w + x] = normalized;
            }
        }
    }

    ImageData {
        pixels,
        height: h,
        width: w,
    }
}

/// Smart resize for Qwen2-VL: round dimensions to multiples of `factor`,
/// then clamp total pixel count within `[min_pixels, max_pixels]`.
///
/// Returns `(new_h, new_w)`.
pub fn smart_resize(
    h: usize,
    w: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    // Round to nearest multiple of factor (at least one factor).
    let mut new_h = ((h.max(1) + factor / 2) / factor).max(1) * factor;
    let mut new_w = ((w.max(1) + factor / 2) / factor).max(1) * factor;

    // Scale up if below min_pixels.
    if new_h * new_w < min_pixels {
        let scale = (min_pixels as f64 / (new_h * new_w) as f64).sqrt();
        new_h = ((new_h as f64 * scale / factor as f64).ceil() as usize) * factor;
        new_w = ((new_w as f64 * scale / factor as f64).ceil() as usize) * factor;
    }

    // Scale down if above max_pixels.
    if new_h * new_w > max_pixels {
        let scale = (max_pixels as f64 / (new_h * new_w) as f64).sqrt();
        new_h = ((new_h as f64 * scale / factor as f64).floor() as usize) * factor;
        new_w = ((new_w as f64 * scale / factor as f64).floor() as usize) * factor;
        if new_h == 0 {
            new_h = factor;
        }
        if new_w == 0 {
            new_w = factor;
        }
    }

    (new_h, new_w)
}

/// Preprocess an image for Qwen2-VL: resize to `target_h x target_w`,
/// convert to f32, normalize with CLIP normalization:
/// `mean=[0.48145466, 0.4578275, 0.40821073]`, `std=[0.26862954, 0.26130258, 0.27577711]`
///
/// Returns `ImageData` with pixels in `[3, H, W]` CHW layout.
pub fn preprocess_qwen2_vl(img: &DynamicImage, target_h: usize, target_w: usize) -> ImageData {
    let resized = img.resize_exact(
        target_w as u32,
        target_h as u32,
        FilterType::CatmullRom, // bicubic — matches HF Qwen2VLImageProcessor.resample=3
    );
    let rgb = resized.to_rgb8();

    let mean = [0.48145466f32, 0.4578275, 0.40821073];
    let std_dev = [0.26862954f32, 0.261_302_6, 0.275_777_1];

    let mut pixels = vec![0.0f32; 3 * target_h * target_w];

    // Convert HWC -> CHW and normalize.
    for y in 0..target_h {
        for x in 0..target_w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                let normalized = (val - mean[c]) / std_dev[c];
                pixels[c * target_h * target_w + y * target_w + x] = normalized;
            }
        }
    }

    ImageData {
        pixels,
        height: target_h,
        width: target_w,
    }
}

/// Extract raw bytes from a `data:image/...;base64,...` URI.
pub fn decode_data_uri(uri: &str) -> ModelResult<Vec<u8>> {
    // Find the base64 data after the comma.
    let comma_pos = uri
        .find(',')
        .ok_or_else(|| ModelError::Other("invalid data URI: no comma separator".into()))?;
    let encoded = &uri[comma_pos + 1..];

    // Decode base64 — support both standard and URL-safe alphabets.
    base64_decode(encoded)
        .map_err(|e| ModelError::Other(format!("failed to decode base64 data URI: {e}")))
}

/// Simple base64 decoder (standard alphabet with optional padding).
fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    // Strip whitespace.
    let clean: String = input.chars().filter(|c| !c.is_whitespace()).collect();
    let bytes = clean.as_bytes();

    let mut output = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;

    for &b in bytes {
        let val = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => continue,
            _ => return Err(format!("invalid base64 character: {}", b as char)),
        };
        buf = (buf << 6) | val as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }

    Ok(output)
}

/// Expand image placeholder tokens in a token ID sequence.
///
/// Finds each occurrence of `image_token_id` in `token_ids` and replaces it
/// with `num_tokens_per_image` copies. Returns the placeholder ranges for
/// the expanded positions.
pub fn expand_image_placeholders(
    token_ids: &mut Vec<u32>,
    image_token_id: u32,
    num_tokens_per_image: usize,
) -> Vec<vllm_common::multimodal::PlaceholderRange> {
    let mut ranges = Vec::new();
    let mut i = 0;
    while i < token_ids.len() {
        if token_ids[i] == image_token_id {
            let offset = i;
            // Replace the single placeholder with num_tokens_per_image copies.
            let extra = num_tokens_per_image.saturating_sub(1);
            for _ in 0..extra {
                token_ids.insert(i + 1, image_token_id);
            }
            ranges.push(vllm_common::multimodal::PlaceholderRange {
                offset,
                length: num_tokens_per_image,
            });
            i += num_tokens_per_image;
        } else {
            i += 1;
        }
    }
    ranges
}

/// Per-image variant of [`expand_image_placeholders`]. Each occurrence
/// of `image_token_id` in `token_ids` is expanded by `counts[k]`
/// (where `k` is the image index, 0-based in scan order). Required for
/// arches like Qwen2-VL whose vision encoder produces a variable
/// post-merger token count per image — the count depends on each
/// image's `smart_resize` output dimensions, which are different for
/// different aspect ratios. Walking `counts` in lockstep with
/// `image_token_id` occurrences mirrors the
/// `extract_images_from_messages` ordering.
///
/// If `counts` is shorter than the number of placeholder occurrences,
/// the trailing extras stay unexpanded (length 1) — the caller is
/// responsible for matching `images.len() == counts.len()`. The
/// returned ranges only cover the indices that were expanded.
pub fn expand_image_placeholders_per_image(
    token_ids: &mut Vec<u32>,
    image_token_id: u32,
    counts: &[usize],
) -> Vec<vllm_common::multimodal::PlaceholderRange> {
    let mut ranges = Vec::new();
    let mut i = 0;
    let mut k = 0;
    while i < token_ids.len() {
        if token_ids[i] == image_token_id {
            if k >= counts.len() {
                // No more per-image counts — leave trailing placeholders alone.
                i += 1;
                continue;
            }
            let n = counts[k];
            let offset = i;
            let extra = n.saturating_sub(1);
            for _ in 0..extra {
                token_ids.insert(i + 1, image_token_id);
            }
            ranges.push(vllm_common::multimodal::PlaceholderRange { offset, length: n });
            i += n;
            k += 1;
        } else {
            i += 1;
        }
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_siglip_shape() {
        // Create a simple 10x10 red image.
        let img = DynamicImage::new_rgb8(10, 10);
        let result = preprocess_siglip(&img, 224);
        assert_eq!(result.height, 224);
        assert_eq!(result.width, 224);
        assert_eq!(result.pixels.len(), 3 * 224 * 224);
    }

    #[test]
    fn test_preprocess_siglip_normalization() {
        // Pure white image: pixel=255 -> (1.0-0.5)/0.5 = 1.0
        let mut img = image::RgbImage::new(4, 4);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([255, 255, 255]);
        }
        let dyn_img = DynamicImage::ImageRgb8(img);
        let result = preprocess_siglip(&dyn_img, 4);
        for &val in &result.pixels {
            assert!((val - 1.0).abs() < 1e-5, "expected 1.0, got {val}");
        }
    }

    #[test]
    fn test_preprocess_siglip_black() {
        // Pure black image: pixel=0 -> (0.0-0.5)/0.5 = -1.0
        let img = image::RgbImage::new(4, 4);
        let dyn_img = DynamicImage::ImageRgb8(img);
        let result = preprocess_siglip(&dyn_img, 4);
        for &val in &result.pixels {
            assert!((val - (-1.0)).abs() < 1e-5, "expected -1.0, got {val}");
        }
    }

    #[test]
    fn test_decode_data_uri() {
        // "Hello" in base64 = "SGVsbG8="
        let uri = "data:text/plain;base64,SGVsbG8=";
        let bytes = decode_data_uri(uri).unwrap();
        assert_eq!(bytes, b"Hello");
    }

    #[test]
    fn test_decode_data_uri_no_comma() {
        let result = decode_data_uri("not-a-data-uri");
        assert!(result.is_err());
    }

    #[test]
    fn test_base64_decode() {
        assert_eq!(base64_decode("SGVsbG8=").unwrap(), b"Hello");
        assert_eq!(base64_decode("dGVzdA==").unwrap(), b"test");
        assert_eq!(base64_decode("").unwrap(), b"");
    }

    #[test]
    fn test_expand_image_placeholders_single() {
        let mut tokens = vec![1, 2, 99, 3, 4];
        let ranges = expand_image_placeholders(&mut tokens, 99, 4);
        assert_eq!(tokens, vec![1, 2, 99, 99, 99, 99, 3, 4]);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].offset, 2);
        assert_eq!(ranges[0].length, 4);
    }

    #[test]
    fn test_expand_image_placeholders_multiple() {
        let mut tokens = vec![1, 99, 2, 99, 3];
        let ranges = expand_image_placeholders(&mut tokens, 99, 3);
        assert_eq!(tokens, vec![1, 99, 99, 99, 2, 99, 99, 99, 3]);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].offset, 1);
        assert_eq!(ranges[0].length, 3);
        assert_eq!(ranges[1].offset, 5);
        assert_eq!(ranges[1].length, 3);
    }

    #[test]
    fn test_expand_image_placeholders_none() {
        let mut tokens = vec![1, 2, 3];
        let ranges = expand_image_placeholders(&mut tokens, 99, 4);
        assert_eq!(tokens, vec![1, 2, 3]);
        assert!(ranges.is_empty());
    }

    #[test]
    fn test_expand_image_placeholders_size_one() {
        let mut tokens = vec![1, 99, 2];
        let ranges = expand_image_placeholders(&mut tokens, 99, 1);
        assert_eq!(tokens, vec![1, 99, 2]);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].offset, 1);
        assert_eq!(ranges[0].length, 1);
    }

    #[test]
    fn test_smart_resize_basic() {
        // 100x200 with factor=28 → rounds to 84x196
        let (h, w) = smart_resize(100, 200, 28, 256 * 28 * 28, 1280 * 28 * 28);
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
        assert!(h * w >= 256 * 28 * 28);
        assert!(h * w <= 1280 * 28 * 28);
    }

    #[test]
    fn test_smart_resize_small_image() {
        // Very small image should scale up to min_pixels.
        let (h, w) = smart_resize(10, 10, 28, 256 * 28 * 28, 1280 * 28 * 28);
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
        assert!(h * w >= 256 * 28 * 28);
    }

    #[test]
    fn test_smart_resize_large_image() {
        // Very large image should scale down to max_pixels.
        let (h, w) = smart_resize(5000, 5000, 28, 256 * 28 * 28, 1280 * 28 * 28);
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
        assert!(h * w <= 1280 * 28 * 28);
    }

    #[test]
    fn test_preprocess_qwen2_vl_shape() {
        let img = DynamicImage::new_rgb8(100, 100);
        let result = preprocess_qwen2_vl(&img, 224, 224);
        assert_eq!(result.height, 224);
        assert_eq!(result.width, 224);
        assert_eq!(result.pixels.len(), 3 * 224 * 224);
    }

    #[test]
    fn test_preprocess_qwen2_vl_normalization() {
        // Pure white image: pixel=255 → (1.0 - 0.48145466) / 0.26862954 ≈ 1.930
        let mut img = image::RgbImage::new(4, 4);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([255, 255, 255]);
        }
        let dyn_img = DynamicImage::ImageRgb8(img);
        let result = preprocess_qwen2_vl(&dyn_img, 4, 4);
        // Check first channel (R): (1.0 - 0.48145466) / 0.26862954
        let expected_r = (1.0 - 0.48145466) / 0.26862954;
        assert!(
            (result.pixels[0] - expected_r).abs() < 1e-3,
            "expected ~{expected_r}, got {}",
            result.pixels[0]
        );
    }

    #[test]
    fn test_decode_image_invalid() {
        let result = decode_image(b"not an image");
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_image_png() {
        // Create a minimal 1x1 PNG in memory.
        let mut buf = Vec::new();
        let img = image::RgbImage::new(1, 1);
        let dyn_img = DynamicImage::ImageRgb8(img);
        dyn_img
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        let decoded = decode_image(&buf).unwrap();
        assert_eq!(decoded.width(), 1);
        assert_eq!(decoded.height(), 1);
    }
}
