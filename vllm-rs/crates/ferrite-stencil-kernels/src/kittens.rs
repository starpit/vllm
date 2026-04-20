// SPDX-License-Identifier: Apache-2.0
//! Rust FFI bindings for the kittens-based megakernel launchers.
//!
//! # THE PLAN (re-read every commit):
//!
//! The ferrite-stencil emitter (`emit_kittens`) writes a per-model
//! `.cu` that defines one `extern "C" cudaError_t launch_<region>(
//! stream, ptrs..., dims...)` per region type. Those are compiled
//! into `libkittens_kernels.a` by ferrite-cuda-builder on sm_90a+.
//! This module is the Rust side: one `unsafe extern "C"` decl per
//! function, plus a safe wrapper that handles stream + error
//! conversion. No higher-level dispatch — that lives in the forward
//! path routing layer.
//!
//! Every launcher takes `cudaStream_t` as u64 (cudarc style), raw
//! `u16*` for bf16 device pointers, and `u32` for scalar dims. Errors
//! propagate as `cudaError_t` encoded in the return value — 0 =
//! success.
//!
//! Linked only when `libkittens_kernels.a` exists — i.e. builder ran
//! on sm_90a hardware with `THUNDERKITTENS_ROOT` set. On lower
//! arches the static archive is absent and this module's externs
//! won't be resolved; callers must gate on `cc >= 90a` before
//! invoking.

#![cfg(feature = "cuda")]

unsafe extern "C" {
    /// `y[token, :] = x[token, :] * rsqrt(mean(x^2) + eps) * w[:]`.
    /// x / w / y are bf16 on device. Grid: `num_tokens` CTAs × 32
    /// threads. `hidden_dim` must match the compile-time `D` in the
    /// emitted kernel (4096 for the Llama-3-8B PoC); returns
    /// `cudaErrorInvalidValue` otherwise.
    fn launch_rmsnorm(
        stream: u64,
        x: *const u16,
        w: *const u16,
        y: *mut u16,
        num_tokens: u32,
        hidden_dim: u32,
        eps: f32,
    ) -> i32;

    /// `y = a + b`, element-wise per token row.
    fn launch_residual_add(
        stream: u64,
        a: *const u16,
        b: *const u16,
        y: *mut u16,
        num_tokens: u32,
        hidden_dim: u32,
    ) -> i32;

    /// `y = x * scale`, scalar broadcast over all rows. Placeholder
    /// for the family of unary ops (scalar_mul, tanh_softcap).
    fn launch_unary_inplace(
        stream: u64,
        x: *const u16,
        y: *mut u16,
        scale: f32,
        num_tokens: u32,
        hidden_dim: u32,
    ) -> i32;

    /// `y[t, :] = embed[token_ids[t], :]`. Gather-driven embedding
    /// lookup. `token_ids` is u32 on device.
    fn launch_embed(
        stream: u64,
        embed: *const u16,
        y: *mut u16,
        token_ids: *const u32,
        num_tokens: u32,
        hidden_dim: u32,
        vocab_size: u32,
    ) -> i32;

    /// `C[M, N] = A[M, K] @ B[K, N]` via WGMMA. Row-major bf16.
    /// M must be a multiple of 64, N of 128, K of 64 (the
    /// compile-time tile params in the emitter).
    fn launch_gemm(
        stream: u64,
        a: *const u16,
        b: *const u16,
        c: *mut u16,
        m: u32,
        n: u32,
        k: u32,
    ) -> i32;

    /// `Inter = silu(X @ Wgate) * (X @ Wup)`. X is [M, K],
    /// Wgate/Wup are [K, N], Inter is [M, N]. TN=64 in the
    /// emitted kernel to fit the 48 KB static shared cap.
    fn launch_gate_up_silu_mul(
        stream: u64,
        x: *const u16,
        w_gate: *const u16,
        w_up: *const u16,
        inter: *mut u16,
        m: u32,
        n: u32,
        k: u32,
    ) -> i32;

    /// QKV projection + RoPE + (direct) write of Q/K/V. X is [M, K];
    /// Wq/Wk/Wv are [K, num_heads*head_dim]; Q/K/V are [M, num_heads*head_dim].
    /// RoPE rotation on Q and K uses rope_cos/rope_sin — currently a
    /// TODO in the emitted kernel (outputs are unrotated).
    fn launch_qkv_rope(
        stream: u64,
        x: *const u16,
        w_q: *const u16,
        w_k: *const u16,
        w_v: *const u16,
        q: *mut u16,
        k: *mut u16,
        v: *mut u16,
        rope_cos: *const u16,
        rope_sin: *const u16,
        m: u32,
        num_heads: u32,
        head_dim: u32,
        k_dim: u32,
    ) -> i32;

    /// FA2 prefill attention. `O = softmax(Q @ K^T * scale) @ V`.
    /// Online softmax rescale is a TODO in the emitted kernel —
    /// numerically wrong for real inference until filled in.
    fn launch_fa2_prefill(
        stream: u64,
        q: *const u16,
        k: *const u16,
        v: *const u16,
        o: *mut u16,
        num_tokens: u32,
        num_heads: u32,
        head_dim: u32,
        seq_len: u32,
        scale: f32,
    ) -> i32;
}

/// Wrapper result type. `cudaError_t` is i32; 0 = success.
pub type CudaRc = i32;

/// RMSNorm launcher. See `launch_rmsnorm` above for arg layout.
///
/// # Safety
/// x/w/y must be valid bf16 device pointers for the given dims;
/// stream must be a valid cudaStream_t handle (0 = default stream).
#[inline]
pub unsafe fn rmsnorm(
    stream: u64,
    x: u64,
    w: u64,
    y: u64,
    num_tokens: u32,
    hidden_dim: u32,
    eps: f32,
) -> CudaRc {
    unsafe {
        launch_rmsnorm(
            stream,
            x as *const u16,
            w as *const u16,
            y as *mut u16,
            num_tokens,
            hidden_dim,
            eps,
        )
    }
}

/// # Safety: see module-level doc.
#[inline]
pub unsafe fn residual_add(
    stream: u64,
    a: u64,
    b: u64,
    y: u64,
    num_tokens: u32,
    hidden_dim: u32,
) -> CudaRc {
    unsafe {
        launch_residual_add(
            stream,
            a as *const u16,
            b as *const u16,
            y as *mut u16,
            num_tokens,
            hidden_dim,
        )
    }
}

/// # Safety: see module-level doc.
#[inline]
pub unsafe fn unary_inplace(
    stream: u64,
    x: u64,
    y: u64,
    scale: f32,
    num_tokens: u32,
    hidden_dim: u32,
) -> CudaRc {
    unsafe {
        launch_unary_inplace(
            stream,
            x as *const u16,
            y as *mut u16,
            scale,
            num_tokens,
            hidden_dim,
        )
    }
}

/// # Safety: see module-level doc.
#[inline]
pub unsafe fn embed(
    stream: u64,
    embed_table: u64,
    y: u64,
    token_ids: u64,
    num_tokens: u32,
    hidden_dim: u32,
    vocab_size: u32,
) -> CudaRc {
    unsafe {
        launch_embed(
            stream,
            embed_table as *const u16,
            y as *mut u16,
            token_ids as *const u32,
            num_tokens,
            hidden_dim,
            vocab_size,
        )
    }
}

/// # Safety: see module-level doc.
#[inline]
pub unsafe fn gemm(stream: u64, a: u64, b: u64, c: u64, m: u32, n: u32, k: u32) -> CudaRc {
    unsafe {
        launch_gemm(
            stream,
            a as *const u16,
            b as *const u16,
            c as *mut u16,
            m,
            n,
            k,
        )
    }
}

/// # Safety: see module-level doc.
#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn gate_up_silu_mul(
    stream: u64,
    x: u64,
    w_gate: u64,
    w_up: u64,
    inter: u64,
    m: u32,
    n: u32,
    k: u32,
) -> CudaRc {
    unsafe {
        launch_gate_up_silu_mul(
            stream,
            x as *const u16,
            w_gate as *const u16,
            w_up as *const u16,
            inter as *mut u16,
            m,
            n,
            k,
        )
    }
}

/// # Safety: see module-level doc.
#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn qkv_rope(
    stream: u64,
    x: u64,
    w_q: u64,
    w_k: u64,
    w_v: u64,
    q: u64,
    k: u64,
    v: u64,
    rope_cos: u64,
    rope_sin: u64,
    m: u32,
    num_heads: u32,
    head_dim: u32,
    k_dim: u32,
) -> CudaRc {
    unsafe {
        launch_qkv_rope(
            stream,
            x as *const u16,
            w_q as *const u16,
            w_k as *const u16,
            w_v as *const u16,
            q as *mut u16,
            k as *mut u16,
            v as *mut u16,
            rope_cos as *const u16,
            rope_sin as *const u16,
            m,
            num_heads,
            head_dim,
            k_dim,
        )
    }
}

/// # Safety: see module-level doc.
#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn fa2_prefill(
    stream: u64,
    q: u64,
    k: u64,
    v: u64,
    o: u64,
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    seq_len: u32,
    scale: f32,
) -> CudaRc {
    unsafe {
        launch_fa2_prefill(
            stream,
            q as *const u16,
            k as *const u16,
            v as *const u16,
            o as *mut u16,
            num_tokens,
            num_heads,
            head_dim,
            seq_len,
            scale,
        )
    }
}
