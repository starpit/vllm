// SPDX-License-Identifier: Apache-2.0
//! Env-gated tensor dump for hybrid-arch (Qwen3-Next) Phase-6c
//! diagnosis. When `FERRITE_DUMP=1` is set, instrumented eval arms
//! emit one stderr line per dump of the form:
//!
//! ```text
//! FERRITE_DUMP {"label":"...","layer":N,"shape":[...],
//!               "dtype":"bf16","first8":[...],"last8":[...]}
//! ```
//!
//! Lines are JSON after the `FERRITE_DUMP ` prefix; a post-run script
//! greps the test stderr and diffs against
//! `/tmp/hf_qwen3_next_dump.json` to find the first divergent op.
//!
//! Off by default: zero overhead when the env var is unset.
//! See `QWEN3_NEXT_HANDOFF.md` (2026-05-04 Phase-6c entries) for the
//! procedure and prior diagnostic state.

use crate::CUstream;
use crate::driver;
use crate::dtype::DType;
use crate::tensor::GpuTensor;
use std::sync::OnceLock;

const SAMPLE_N: usize = 8;

pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("FERRITE_DUMP").is_some())
}

/// Read up to `SAMPLE_N` elements from the head and tail of `t`,
/// convert to f32, and emit one `FERRITE_DUMP {json}` line on stderr.
/// Synchronizes `stream` so the read sees post-kernel state. No-op
/// when the env-var is unset, when `t` is empty, or when the dtype
/// has no f32 representation we care about (FP8 etc).
///
/// # Safety
/// `t.raw_ptr()` + `t.numel() * dtype.size_bytes()` must point at
/// valid device memory; `stream` must be a live CUstream the caller
/// owns or is synchronized with.
pub unsafe fn dump_tile(label: &str, layer: u32, t: &GpuTensor, stream: CUstream) {
    if !enabled() {
        return;
    }
    let n = t.numel();
    if n == 0 {
        return;
    }
    let dtype = t.dtype();
    let bytes_per = dtype.size_bytes();
    let head_n = SAMPLE_N.min(n);
    let tail_n = SAMPLE_N.min(n);
    let mut head = vec![0u8; head_n * bytes_per];
    let mut tail = vec![0u8; tail_n * bytes_per];
    unsafe {
        driver::memcpy_dtoh_async(
            head.as_mut_ptr(),
            t.raw_ptr() as *const u8,
            head_n * bytes_per,
            stream,
        )
        .expect("FERRITE_DUMP: dtoh head");
        let tail_off = (n - tail_n) * bytes_per;
        driver::memcpy_dtoh_async(
            tail.as_mut_ptr(),
            (t.raw_ptr() as *const u8).add(tail_off),
            tail_n * bytes_per,
            stream,
        )
        .expect("FERRITE_DUMP: dtoh tail");
        driver::stream_synchronize(stream).expect("FERRITE_DUMP: sync");
    }
    let head_f = bytes_to_f32(&head, dtype);
    let tail_f = bytes_to_f32(&tail, dtype);
    eprintln!(
        "FERRITE_DUMP {{\"label\":\"{}\",\"layer\":{},\"ptr\":\"{:p}\",\"shape\":{:?},\"dtype\":\"{}\",\"first8\":{},\"last8\":{}}}",
        label,
        layer,
        t.raw_ptr(),
        t.shape(),
        dtype,
        json_floats(&head_f),
        json_floats(&tail_f),
    );
}

/// Download ALL elements from `t`, check for non-finite values, and emit a
/// diagnostic line. Unlike `dump_tile` (first8/last8 sample), this scans the
/// full tensor so Inf/NaN hiding in middle dimensions are not missed.
///
/// Emits: `FERRITE_DUMP_FINITE {json}` with fields:
///   label, layer, shape, dtype, any_nonfinite (bool), first_nonfinite_idx,
///   max_abs, count_nonfinite.
///
/// Gated by the same `FERRITE_DUMP` env var as `dump_tile`.
///
/// # Safety
/// Same as `dump_tile`.
pub unsafe fn dump_tile_check_finite(label: &str, layer: u32, t: &GpuTensor, stream: CUstream) {
    if !enabled() {
        return;
    }
    let n = t.numel();
    if n == 0 {
        return;
    }
    let dtype = t.dtype();
    let bytes_per = dtype.size_bytes();
    let mut buf = vec![0u8; n * bytes_per];
    unsafe {
        driver::memcpy_dtoh_async(buf.as_mut_ptr(), t.raw_ptr() as *const u8, n * bytes_per, stream)
            .expect("FERRITE_DUMP_FINITE: dtoh");
        driver::stream_synchronize(stream).expect("FERRITE_DUMP_FINITE: sync");
    }
    let vals = bytes_to_f32(&buf, dtype);
    let mut count_nonfinite: usize = 0;
    let mut first_nonfinite_idx: Option<usize> = None;
    let mut max_abs: f32 = 0.0;
    for (i, &v) in vals.iter().enumerate() {
        if !v.is_finite() {
            count_nonfinite += 1;
            if first_nonfinite_idx.is_none() {
                first_nonfinite_idx = Some(i);
            }
        } else if v.abs() > max_abs {
            max_abs = v.abs();
        }
    }
    eprintln!(
        "FERRITE_DUMP_FINITE {{\"label\":\"{}\",\"layer\":{},\"shape\":{:?},\"dtype\":\"{}\",\
         \"any_nonfinite\":{},\"first_nonfinite_idx\":{},\"count_nonfinite\":{},\"max_abs\":{}}}",
        label,
        layer,
        t.shape(),
        dtype,
        count_nonfinite > 0,
        first_nonfinite_idx.map(|i| i as i64).unwrap_or(-1),
        count_nonfinite,
        max_abs,
    );
}

fn bytes_to_f32(bytes: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        DType::BF16 => bytes
            .chunks_exact(2)
            .map(|b| {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect(),
        DType::F16 => bytes
            .chunks_exact(2)
            .map(|b| {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect(),
        DType::U32 => bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
            .collect(),
        DType::I32 => bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
            .collect(),
        DType::I64 => bytes
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
            .collect(),
        DType::U8 => bytes.iter().map(|&b| b as f32).collect(),
        // Phase-6c diagnosis is BF16-end-to-end; FP8 callers shouldn't
        // hit dump_tile for a working forward, but emit NaN instead of
        // panicking so the env-var stays harmless if it does.
        DType::Fp8E4m3 => bytes.iter().map(|_| f32::NAN).collect(),
    }
}

fn json_floats(xs: &[f32]) -> String {
    let mut s = String::with_capacity(xs.len() * 12);
    s.push('[');
    for (i, &x) in xs.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        if x.is_nan() {
            s.push_str("\"NaN\"");
        } else if x.is_infinite() {
            s.push_str(if x > 0.0 { "\"Inf\"" } else { "\"-Inf\"" });
        } else {
            // 6 significant digits is enough to spot first-divergence;
            // matches the precision HF dump rounds to.
            s.push_str(&format!("{:.6}", x));
        }
    }
    s.push(']');
    s
}
