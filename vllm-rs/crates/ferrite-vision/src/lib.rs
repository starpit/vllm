// SPDX-License-Identifier: Apache-2.0
//! Host-side glue shared by ferrite vision towers.
//!
//! Each `ferrite-model-*-vl` crate composes its own per-block GPU forward,
//! but the helpers here — 2D RoPE cos/sin tables, varlen `cu_seqlens`,
//! pixel patch flatten, K-pad-to-multiple-of-8, env-driven trace dump —
//! are arch-independent. Centralizing them avoids the per-arch
//! duplication that grew during Phase D/E.
//!
//! Phase G.1 of the Vision DSL plan (`VISION_DSL_HANDOFF.md`): mechanical
//! lift, no behavior change. Subsequent phases (G.3+) move the actual
//! per-block math from imperative `vision.rs` into a `#[vision_forward]`
//! DSL body; this crate stays as the host-side scaffold around it.
//!
//! [`VisionConfig`] carries the geometric fields every VL/MM tower
//! shares (embed_dim, num_heads, spatial_merge_size, patch_size,
//! temporal_patch_size, in_chans, depth, d_model, norm_eps). Host
//! helpers (`build_rope_cos_sin_bf16`, `patches_from_normalized_chw`)
//! hang off `VisionConfig` as methods. Each VL crate carries its own
//! arch-specific extras (`intermediate_size`, `window_size`, …) in a
//! sibling struct.

use anyhow::Result;
use ferrite_cuda_core::CUstream;
use ferrite_cuda_core::driver;
use ferrite_cuda_core::tensor::GpuTensor;
use ferrite_cuda_core::weights::GpuWeights;
use ferrite_kernels::layers::Linear;

/// RoPE base used by every Qwen2/2.5-VL tower (and the working assumption
/// for the next VL arches). Lift to a `VisionConfig` field if a future
/// tower picks a different theta.
pub const ROPE_THETA: f32 = 10000.0;

/// Common vision-tower config — the geometric fields every VL/MM arch
/// shares. Arch-specific extras (`mlp_ratio`, `intermediate_size`,
/// `window_size`, …) live alongside this struct in the per-arch crate.
///
/// All host-side helpers (`build_rope_cos_sin_bf16`,
/// `build_cu_seqlens_i32`, `patches_from_normalized_chw`) hang off this
/// struct as methods, mirroring the way the text path's [`#[forward]`]
/// DSL hangs off a single `Weights`/`Config` surface.
#[derive(Clone, Copy, Debug)]
pub struct VisionConfig {
    pub embed_dim: u32,
    pub depth: u32,
    pub num_heads: u32,
    pub patch_size: u32,
    pub temporal_patch_size: u32,
    pub spatial_merge_size: u32,
    pub in_chans: u32,
    /// Text-decoder hidden the patch-merger projects into. Equals
    /// `text_config.hidden_size`.
    pub d_model: u32,
    pub norm_eps: f32,
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        (self.embed_dim / self.num_heads) as usize
    }
    pub fn half_rot(&self) -> usize {
        self.head_dim() / 2
    }
}

// ── 2D RoPE cos/sin construction ───────────────────────────────────
//
// Mirrors Python `Qwen2VisionTransformer.rot_pos_emb`:
//   for each (t, h, w) in grid_thw:
//     hpos = arange(h)[:,None].expand(h,w)        # [h, w]
//     wpos = arange(w)[None,:].expand(h,w)        # [h, w]
//     hpos = hpos.reshape(h/S, S, w/S, S).permute(0,2,1,3).flatten()
//     wpos = wpos.reshape(h/S, S, w/S, S).permute(0,2,1,3).flatten()
//     pos = stack([hpos, wpos], -1).repeat(t, 1)  # [t*h*w, 2]
//   pos = cat(per-image)                          # [total_l, 2]
//   inv_freq = 1 / theta**(arange(0, half_rot/2*2, 2) / (half_rot))
//                                                 # length half_rot/2
//   freqs = arange(max_grid).outer(inv_freq)     # [max_grid, half_rot/2]
//   cos_table = freqs.cos(); sin_table = freqs.sin()
//   cos_per_token = cos_table[pos].flatten(1)    # [total_l, half_rot]
// where S = spatial_merge_size, half_rot = head_dim / 2.
//
// Per-token cos has 2 halves: [cos(h_pos · inv_freq[..]), cos(w_pos · inv_freq[..])].
// `vision_rope_apply` reads cos[token, hi] for hi ∈ [0, half_rot), so the
// h/w split lives entirely in the cos/sin layout — kernel is generic.

impl VisionConfig {
    /// Build per-token bf16 cos/sin tables `[total_l, head_dim/2]` for the 2D
    /// vision RoPE. Result lives on the host; each VL crate uploads to the
    /// device with `alloc_gpu_tensor_from_host`.
    pub fn build_rope_cos_sin_bf16(
        &self,
        grid_thw: &[(u32, u32, u32)],
        total_l: usize,
    ) -> (Vec<u16>, Vec<u16>) {
        let head_dim = self.head_dim();
        let half_rot = head_dim / 2;
        let freq_axis_dim = half_rot / 2;
        let s = self.spatial_merge_size as usize;

        let inv_freq: Vec<f32> = (0..freq_axis_dim)
            .map(|i| 1.0 / ROPE_THETA.powf((2 * i) as f32 / (half_rot as f32)))
            .collect();

        let mut cos = Vec::<u16>::with_capacity(total_l * half_rot);
        let mut sin = Vec::<u16>::with_capacity(total_l * half_rot);
        for &(t, h, w) in grid_thw {
            let (h, w, t) = (h as usize, w as usize, t as usize);
            debug_assert_eq!(h % s, 0, "h must be divisible by spatial_merge_size");
            debug_assert_eq!(w % s, 0, "w must be divisible by spatial_merge_size");
            let h_blocks = h / s;
            let w_blocks = w / s;
            let frame_len = h * w;
            let mut hpos = vec![0u32; frame_len];
            let mut wpos = vec![0u32; frame_len];
            let mut idx = 0usize;
            for hb in 0..h_blocks {
                for wb in 0..w_blocks {
                    for sh in 0..s {
                        for sw in 0..s {
                            hpos[idx] = (hb * s + sh) as u32;
                            wpos[idx] = (wb * s + sw) as u32;
                            idx += 1;
                        }
                    }
                }
            }
            for _ in 0..t {
                for token in 0..frame_len {
                    let hp = hpos[token] as f32;
                    let wp = wpos[token] as f32;
                    for &f in inv_freq.iter() {
                        let theta_h = hp * f;
                        cos.push(f32_to_bf16(theta_h.cos()));
                        sin.push(f32_to_bf16(theta_h.sin()));
                    }
                    for &f in inv_freq.iter() {
                        let theta_w = wp * f;
                        cos.push(f32_to_bf16(theta_w.cos()));
                        sin.push(f32_to_bf16(theta_w.sin()));
                    }
                }
            }
        }
        debug_assert_eq!(cos.len(), total_l * half_rot);
        (cos, sin)
    }
} // impl VisionConfig (rope)

/// Build per-image varlen `cu_seqlens` (one segment per (T, H, W) frame —
/// a (t, h, w) tuple contributes `t` segments of length `h * w`).
/// Returns the prefix-sum buffer and the max segment length.
///
/// Free function (not a `VisionConfig` method) because cu_seqlens depends
/// only on `grid_thw`; no config fields enter.
pub fn build_cu_seqlens_i32(grid_thw: &[(u32, u32, u32)]) -> (Vec<i32>, usize) {
    let mut cu = Vec::<i32>::with_capacity(grid_thw.len() + 1);
    cu.push(0);
    let mut max_seqlen = 0usize;
    let mut acc: i32 = 0;
    for &(t, h, w) in grid_thw {
        let seg = (h as usize) * (w as usize);
        for _ in 0..t {
            acc += seg as i32;
            cu.push(acc);
            if seg > max_seqlen {
                max_seqlen = seg;
            }
        }
    }
    (cu, max_seqlen)
}

// ── Qwen2VLImageProcessor patch-flatten ────────────────────────────
//
// Mirrors Python `Qwen2VLImageProcessor._preprocess`'s reshape +
// 9D transpose:
//
//   patches: [T, C, H, W]                              (image: T = temporal_patch_size, frames repeated)
//      -> reshape [grid_t, T, C, gH, mH, P, gW, mW, P]
//      -> transpose (0, 3, 6, 4, 7, 2, 1, 5, 8)
//         dims: [grid_t, gH, gW, mH, mW, C, T, P, P]
//      -> reshape [grid_t * grid_h * grid_w, C * T * P * P]
//
// where grid_h = H / P, grid_w = W / P, gH = grid_h / mH, gW = grid_w / mW.
//
// We don't materialize the 9D intermediate — we walk output index space
// directly, which costs ~4 MB f32 read + ~2 MB bf16 write per 392²
// image, negligible vs the encoder's per-block GEMM traffic.

impl VisionConfig {
    /// Patch-flatten one normalized CHW image to `[L, C·T·P²]` bf16
    /// patches + `(grid_t, grid_h, grid_w)`. `H` and `W` must be multiples
    /// of `patch_size · spatial_merge_size` (Python's smart_resize step
    /// guarantees this; the engine layer mirrors it).
    pub fn patches_from_normalized_chw(
        &self,
        pixels: &[f32],
        height: u32,
        width: u32,
    ) -> (Vec<u16>, (u32, u32, u32)) {
        let p = self.patch_size as usize;
        let s = self.spatial_merge_size as usize;
        let t = self.temporal_patch_size as usize;
        let c = self.in_chans as usize;
        let h = height as usize;
        let w = width as usize;
        assert_eq!(
            pixels.len(),
            c * h * w,
            "patches_from_normalized_chw: pixels.len() = {} but c*h*w = {}",
            pixels.len(),
            c * h * w
        );
        assert_eq!(
            h % (p * s),
            0,
            "image height {h} must be multiple of patch_size*merge_size = {}",
            p * s
        );
        assert_eq!(
            w % (p * s),
            0,
            "image width {w} must be multiple of patch_size*merge_size = {}",
            p * s
        );
        let grid_h = h / p;
        let grid_w = w / p;
        let g_h = grid_h / s;
        let g_w = grid_w / s;
        let grid_t: u32 = 1;
        let l = (grid_t as usize) * grid_h * grid_w;
        let feat = c * t * p * p;
        let mut out = vec![0u16; l * feat];
        let mut out_idx = 0usize;
        let stride_c = h * w;
        let stride_h = w;
        for gh in 0..g_h {
            for gw in 0..g_w {
                for mh in 0..s {
                    for mw in 0..s {
                        for ci in 0..c {
                            for _ti in 0..t {
                                let img_h_base = gh * (s * p) + mh * p;
                                let img_w_base = gw * (s * p) + mw * p;
                                for ph in 0..p {
                                    let img_h = img_h_base + ph;
                                    let row = ci * stride_c + img_h * stride_h + img_w_base;
                                    for pw in 0..p {
                                        let v = pixels[row + pw];
                                        out[out_idx] = f32_to_bf16(v);
                                        out_idx += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        debug_assert_eq!(out_idx, l * feat);
        (out, (grid_t, grid_h as u32, grid_w as u32))
    }
} // impl VisionConfig (patch flatten)

/// Free-fn wrapper around [`VisionConfig::patches_from_normalized_chw`]
/// matching the macro-emitted `pixel_pack` signature
/// `fn(&VisionConfig, &[f32], u32, u32) -> (Vec<u16>, (u32, u32, u32))`.
///
/// The `#[vision_forward]` attribute takes a `pixel_pack = path::to::fn`
/// arg; the macro emits an [`ferrite_forward::VisionArchWeights`] impl
/// whose `pixel_pack` associated fn forwards to the provided path. This
/// is the Qwen2-VL / Qwen2.5-VL flavor; SigLIP / Gemma3-MM will get
/// their own free fn here when they land.
pub fn pack_qwen2_vl(
    cfg: &VisionConfig,
    pixels: &[f32],
    height: u32,
    width: u32,
) -> (Vec<u16>, (u32, u32, u32)) {
    cfg.patches_from_normalized_chw(pixels, height, width)
}

// ── K-pad to multiple of 8 ─────────────────────────────────────────
//
// cuBLAS BF16 GEMM rejects K=3420 (Qwen2.5-VL-3B intermediate_size)
// with `CUBLAS_STATUS_INTERNAL_ERROR`; padding K to the next multiple
// of 8 lands the GEMM on an algo cuBLAS supports. Tail columns are
// zero so `out[i, j] = sum_k input[i, k] * weight[j, k]` is unchanged
// when the activation feeds zeros into those padded columns (paired
// with the activation-side packer at the call site).

/// Pad a `[D0, K]` Linear's weight to `[D0, K_pad]` where
/// `K_pad = round_up(K, 8)`. Tail columns are zero so contribution is
/// preserved. No-op if K is already a multiple of 8.
///
/// # Safety
/// `weights` must outlive the returned `Linear`'s underlying allocation
/// (the new pointer is registered via `record_alloc` so the
/// caching/weights allocator owns it).
pub unsafe fn pad_linear_k_to_mult8(
    linear: Linear,
    weights: &mut GpuWeights,
    stream: CUstream,
) -> Result<Linear> {
    let w = linear.weight;
    debug_assert_eq!(w.ndim(), 2);
    let d0 = w.dim(0);
    let k = w.dim(1);
    let k_pad = k.next_multiple_of(8);
    if k_pad == k {
        return Ok(linear);
    }
    let dtype = w.dtype();
    let elem = dtype.size_bytes();
    let src_pitch = k * elem;
    let dst_pitch = k_pad * elem;
    let total_bytes = d0 * dst_pitch;
    let new_ptr = unsafe { driver::mem_alloc(total_bytes) }?;
    weights.record_alloc(new_ptr, total_bytes);
    unsafe { driver::memset_d8(new_ptr, 0, total_bytes, stream) }?;
    let src_base = w.raw_ptr() as *const u8;
    for r in 0..d0 {
        unsafe {
            driver::memcpy_dtod_async(
                new_ptr.add(r * dst_pitch),
                src_base.add(r * src_pitch),
                src_pitch,
                stream,
            )?;
        }
    }
    let new_w = unsafe { GpuTensor::new(new_ptr, &[d0, k_pad], dtype) };
    Ok(Linear::new(new_w, linear.bias))
}

// ── Trace dump (FERRITE_VIT_DUMP_DIR=<dir>) ────────────────────────
//
// When set, every `dump_tensor` call after the env was first observed
// blocks on `stream`, D2Hs the device tensor into a host buffer, and
// appends `<dir>/<name>.bin` plus a metadata line in `<dir>/dump.jsonl`.
// Used to diff vision-encoder intermediates against a Python golden.

/// Env-driven (`FERRITE_VIT_DUMP_DIR=<dir>`) intermediate-tensor
/// recorder. Construct once per forward (`TraceDump::from_env`); call
/// `dump_tensor` between encoder stages to write `.bin` + jsonl metadata.
pub struct TraceDump {
    dir: Option<std::path::PathBuf>,
}

impl TraceDump {
    pub fn from_env() -> Self {
        let dir = std::env::var("FERRITE_VIT_DUMP_DIR").ok().and_then(|s| {
            if s.is_empty() {
                None
            } else {
                let p = std::path::PathBuf::from(s);
                if let Err(e) = std::fs::create_dir_all(&p) {
                    eprintln!("FERRITE_VIT_DUMP_DIR: cannot create {}: {}", p.display(), e);
                    return None;
                }
                Some(p)
            }
        });
        Self { dir }
    }

    /// # Safety
    /// `t` must be a live device tensor produced on `stream`'s context;
    /// this routine D2Hs synchronously around the copy.
    pub unsafe fn dump_tensor(&self, name: &str, t: GpuTensor, stream: CUstream) {
        let Some(dir) = self.dir.as_ref() else {
            return;
        };
        let bytes = t.size_bytes();
        if bytes == 0 {
            return;
        }
        let mut host = vec![0u8; bytes];
        unsafe {
            driver::stream_synchronize(stream).expect("vit-dump: pre-D2H sync");
            driver::memcpy_dtoh_async(host.as_mut_ptr(), t.raw_ptr(), bytes, stream)
                .expect("vit-dump: D2H");
            driver::stream_synchronize(stream).expect("vit-dump: post-D2H sync");
        }
        let bin_path = dir.join(format!("{name}.bin"));
        std::fs::write(&bin_path, &host)
            .unwrap_or_else(|e| eprintln!("vit-dump: write {} failed: {}", bin_path.display(), e));
        let meta = format!(
            "{{\"name\":\"{name}\",\"shape\":{:?},\"dtype\":\"{:?}\",\"bytes\":{bytes}}}\n",
            t.shape().iter().collect::<Vec<_>>(),
            t.dtype()
        );
        let log_path = dir.join("dump.jsonl");
        use std::io::Write as _;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            let _ = f.write_all(meta.as_bytes());
        }
        eprintln!("vit-dump: {} {:?} {:?}", name, t.shape(), t.dtype());
    }
}

// ── Slice helpers ──────────────────────────────────────────────────

pub fn f32_to_bf16(v: f32) -> u16 {
    half::bf16::from_f32(v).to_bits()
}

pub fn bf16_slice_as_bytes(s: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

pub fn i32_slice_as_bytes(s: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

pub fn u32_slice_as_bytes(s: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
