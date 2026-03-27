// SPDX-License-Identifier: Apache-2.0
//! Multi-segment SDPA via native MLX Metal primitive.
//!
//! Attention across multiple contiguous K/V segments (spans).
//! Each segment is a contiguous K/V array. The kernel iterates segments
//! with online softmax and optionally applies RoPE to K on-the-fly.
//!
//! Per-segment `needs_rope` flag:
//! - `false` (0): K already has RoPE applied (active cache, same as current path)
//! - `true` (1): K stored without RoPE (span segments, kernel applies RoPE)

use mlx_rs::Array;
use mlx_rs::error::Exception;

#[repr(C)]
#[derive(Copy, Clone)]
struct MlxArrayRaw {
    ctx: *mut std::ffi::c_void,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct MlxStreamRaw {
    ctx: *mut std::ffi::c_void,
}

unsafe extern "C" {
    fn vllm_multi_segment_sdpa(
        result: *mut MlxArrayRaw,
        query: MlxArrayRaw,
        k_segments: *const MlxArrayRaw,
        v_segments: *const MlxArrayRaw,
        num_segments: i32,
        seg_position_offsets: MlxArrayRaw,
        seg_needs_rope: *const i32,
        cos_sin_cache: MlxArrayRaw,
        scale: f32,
        rotary_dim: i32,
        stream: MlxStreamRaw,
    ) -> i32;
}

fn arr_to_raw(a: &Array) -> MlxArrayRaw {
    let ptr = a.as_ptr();
    MlxArrayRaw { ctx: ptr.ctx }
}

/// Multi-segment SDPA — attention across multiple contiguous K/V segments.
///
/// * `query` — `[1, num_heads, 1, head_dim]` (single decode query, RoPE already applied)
/// * `k_segments` — N arrays, each `[1, num_kv_heads, seg_len, head_dim]`
/// * `v_segments` — N arrays, same layout
/// * `seg_position_offsets` — `[N]` i32, RoPE position offset per segment
/// * `seg_needs_rope` — N bools: true = apply RoPE on-the-fly (span), false = K pre-rotated (active cache)
/// * `cos_sin_cache` — `[max_pos, rotary_dim]` precomputed cos/sin for fused RoPE
/// * `scale` — 1/sqrt(head_dim)
/// * `rotary_dim` — number of dimensions that get RoPE
///
/// Returns `[1, num_heads, 1, head_dim]` (lazy).
pub fn multi_segment_sdpa(
    query: &Array,
    k_segments: &[&Array],
    v_segments: &[&Array],
    seg_position_offsets: &Array,
    seg_needs_rope: &[bool],
    cos_sin_cache: &Array,
    scale: f32,
    rotary_dim: usize,
) -> Result<Array, Exception> {
    assert_eq!(k_segments.len(), v_segments.len());
    assert_eq!(k_segments.len(), seg_needs_rope.len());

    let k_raw: Vec<MlxArrayRaw> = k_segments.iter().map(|a| arr_to_raw(a)).collect();
    let v_raw: Vec<MlxArrayRaw> = v_segments.iter().map(|a| arr_to_raw(a)).collect();
    let needs_rope_i32: Vec<i32> = seg_needs_rope.iter().map(|&b| b as i32).collect();

    unsafe {
        let stream = mlx_rs::StreamOrDevice::default();
        let stream_raw = MlxStreamRaw {
            ctx: stream.as_ref().as_ptr().ctx,
        };

        let mut result = MlxArrayRaw {
            ctx: std::ptr::null_mut(),
        };

        let status = vllm_multi_segment_sdpa(
            &mut result,
            arr_to_raw(query),
            k_raw.as_ptr(),
            v_raw.as_ptr(),
            k_segments.len() as i32,
            arr_to_raw(seg_position_offsets),
            needs_rope_i32.as_ptr(),
            arr_to_raw(cos_sin_cache),
            scale,
            rotary_dim as i32,
            stream_raw,
        );

        if status != 0 {
            return Err(Exception::custom("multi_segment_sdpa failed"));
        }

        Ok(Array::from_ptr(std::mem::transmute(result)))
    }
}
