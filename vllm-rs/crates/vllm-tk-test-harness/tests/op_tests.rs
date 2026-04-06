// SPDX-License-Identifier: Apache-2.0
//! Per-op GPU smoke tests for TK sm89 ops.
//!
//! Each test launches a single TK op in isolation with controlled inputs.
//! This isolates op-level bugs from sequencing/barrier issues in the full megakernel.
//!
//! Run with: cargo test -p vllm-tk-test-harness --features cuda -- --ignored --test-threads=1

#![cfg(feature = "cuda")]

use cudarc::driver::result;
use half::bf16;
use vllm_tk_test_harness::TkTensorArg;
use vllm_tk_test_harness::ffi;

/// Model dims for 1B LLaMA (matches the DSL in build.rs)
const HD: usize = 2048;
const ID: usize = 8192;
const HDM: usize = 64;
const NAH: usize = 32;
const NKH: usize = 8;
const NL: usize = 16;
const VS: usize = 128256;
const NBH: usize = NAH + 2 * NKH; // 48
const QKV_DIM: usize = NBH * HDM; // 3072
const BF16: usize = 2;

// TK scheduling constants (must match gpu_smoke.rs)
const BS: usize = 1;
const N_BATCH_BLOCKS: usize = 1;
const NUM_OPS: usize = 11;
const MAX_BARRIER_COLS: usize = 128;
const ACT_ROWS: usize = 128; // n_batch_blocks * matmul_batch_block_size
const SM_COUNT: usize = 142; // L40S
const MAX_PER_SM: usize = 1;
const INSTRUCTION_WIDTH: usize = 32;
const TIMING_WIDTH: usize = 128;
const NUM_PAGES: usize = 16;
const KV_PAGE_SIZE: usize = 64;  // tokens per page (SM89_KV_PAGE_SIZE in llama_sm89.cuh)
const KV_BLOCK_SIZE: usize = 16; // tile rows per KV block (SM89_KV_BLOCK_SIZE)

/// Allocate `bytes` of zeroed GPU memory.
fn gpu_alloc_zeros(bytes: usize) -> u64 {
    unsafe {
        let dptr = result::malloc_sync(bytes).expect("cuMemAlloc failed");
        result::memset_d8_sync(dptr, 0, bytes).expect("cuMemsetD8 failed");
        dptr as u64
    }
}

/// Allocate GPU memory filled with a constant i32 value.
/// Used to pre-fill barrier arrays so gmem_waiters don't hang.
fn gpu_alloc_i32_fill(count: usize, val: i32) -> u64 {
    unsafe {
        let dptr = result::malloc_sync(count * 4).expect("cuMemAlloc failed");
        let host: Vec<i32> = vec![val; count];
        result::memcpy_htod_sync(dptr, &host).expect("cuMemcpyHtoD failed");
        dptr as u64
    }
}

fn init_cuda() {
    result::init().expect("cuInit failed");
    let device = result::device::get(0).expect("cuDeviceGet failed");
    let ctx = unsafe { result::primary_ctx::retain(device) }.expect("cuCtxRetain failed");
    unsafe { result::ctx::set_current(ctx) }.expect("cuCtxSetCurrent failed");
}

struct TestBuffers {
    bar: u64,
    instr: u64,
    timings: u64,
    qkv_w: u64,
    attn_norm_w: u64,
    o_w: u64,
    mlp_norm_w: u64,
    up_w: u64,
    gate_w: u64,
    down_w: u64,
    lm_norm_w: u64,
    lm_w: u64,
    k_cache: u64,
    v_cache: u64,
    rope_cos: u64,
    rope_sin: u64,
    hidden: u64,
    rms_rope: u64,
    rms_gate: u64,
    q_post: u64,
    attn_out: u64,
    silu_buf: u64,
    rms_lm: u64,
    logits: u64,
    pos_ids: u64,
    kv_indptr: u64,
    kv_indices: u64,
    kv_last_page: u64,
    kv_append: u64,
    dummy_meta: u64,
}

impl TestBuffers {
    fn new() -> Self {
        Self {
            // Pre-fill barriers with large values so gmem_waiters (which spin on
            // barrier counts from previous ops) see "completed" immediately.
            // Without this, ops like mlp_norm hang waiting for o_proj's barrier.
            bar: gpu_alloc_i32_fill(NL * NUM_OPS * N_BATCH_BLOCKS * MAX_BARRIER_COLS, 9999),
            instr: gpu_alloc_zeros(SM_COUNT * MAX_PER_SM * INSTRUCTION_WIDTH * 4),
            timings: gpu_alloc_zeros(SM_COUNT * MAX_PER_SM * TIMING_WIDTH * 4),
            qkv_w: gpu_alloc_zeros(NL * QKV_DIM * HD * BF16),
            attn_norm_w: gpu_alloc_zeros(NL * HD * BF16),
            o_w: gpu_alloc_zeros(NL * HD * HD * BF16),
            mlp_norm_w: gpu_alloc_zeros(NL * HD * BF16),
            up_w: gpu_alloc_zeros(NL * ID * HD * BF16),
            gate_w: gpu_alloc_zeros(NL * ID * HD * BF16),
            down_w: gpu_alloc_zeros(NL * HD * ID * BF16),
            lm_norm_w: gpu_alloc_zeros(HD * BF16),
            lm_w: gpu_alloc_zeros(VS * HD * BF16),
            k_cache: gpu_alloc_zeros(NUM_PAGES * KV_PAGE_SIZE * NKH * HDM * BF16),
            v_cache: gpu_alloc_zeros(NUM_PAGES * KV_PAGE_SIZE * NKH * HDM * BF16),
            rope_cos: gpu_alloc_zeros(4096 * HDM * BF16),
            rope_sin: gpu_alloc_zeros(4096 * HDM * BF16),
            hidden: gpu_alloc_zeros(ACT_ROWS * HD * BF16),
            rms_rope: gpu_alloc_zeros(ACT_ROWS * HD * BF16),
            rms_gate: gpu_alloc_zeros(ACT_ROWS * HD * BF16),
            q_post: gpu_alloc_zeros(ACT_ROWS * HD * BF16),
            attn_out: gpu_alloc_zeros(ACT_ROWS * HD * BF16),
            silu_buf: gpu_alloc_zeros(ACT_ROWS * ID * BF16),
            rms_lm: gpu_alloc_zeros(ACT_ROWS * HD * BF16),
            logits: gpu_alloc_zeros(ACT_ROWS * VS * BF16),
            pos_ids: gpu_alloc_zeros(BS * 4),
            kv_indptr: gpu_alloc_zeros((BS + 1) * 4),
            kv_indices: gpu_alloc_zeros(NUM_PAGES * 4),
            kv_last_page: gpu_alloc_zeros(BS * 4),
            kv_append: gpu_alloc_zeros(BS * 4),
            dummy_meta: gpu_alloc_zeros(4),
        }
    }
}

/// Call a test launch function with all standard args from TestBuffers.
/// Shapes match `args_to_ffi` in the proc-macro exactly.
macro_rules! call_launch {
    ($fn:path, $b:expr) => {
        unsafe {
            $fn(
                TkTensorArg::new($b.bar, &[NL, NUM_OPS, N_BATCH_BLOCKS, MAX_BARRIER_COLS]),
                TkTensorArg::new($b.instr, &[SM_COUNT, MAX_PER_SM, INSTRUCTION_WIDTH]),
                TkTensorArg::new($b.timings, &[SM_COUNT, MAX_PER_SM, TIMING_WIDTH]),
                // Weights — NL folded into row dim (not depth)
                TkTensorArg::new($b.qkv_w, &[NL * QKV_DIM, HD]),
                TkTensorArg::new($b.attn_norm_w, &[NL, HD]),
                TkTensorArg::new($b.o_w, &[NL * HD, HD]),
                TkTensorArg::new($b.mlp_norm_w, &[NL, HD]),
                TkTensorArg::new($b.up_w, &[NL * ID, HD]),
                TkTensorArg::new($b.gate_w, &[NL * ID, HD]),
                TkTensorArg::new($b.down_w, &[NL * HD, ID]),
                TkTensorArg::new($b.lm_norm_w, &[1, HD]),
                TkTensorArg::new($b.lm_w, &[VS, HD]),
                // KV cache — d=1 (page_size is baked into GL type)
                TkTensorArg::new($b.k_cache, &[NUM_PAGES, KV_PAGE_SIZE, NKH, HDM]),
                TkTensorArg::new($b.v_cache, &[NUM_PAGES, KV_PAGE_SIZE, NKH, HDM]),
                TkTensorArg::new($b.rope_cos, &[4096, HDM]),
                TkTensorArg::new($b.rope_sin, &[4096, HDM]),
                // Activations — 4D with b=1, d=1
                TkTensorArg::new($b.hidden, &[1, 1, ACT_ROWS, HD]),
                TkTensorArg::new($b.rms_rope, &[1, 1, ACT_ROWS, HD]),
                TkTensorArg::new($b.rms_gate, &[1, 1, ACT_ROWS, HD]),
                TkTensorArg::new($b.q_post, &[1, 1, ACT_ROWS, HD]),
                TkTensorArg::new($b.attn_out, &[1, 1, ACT_ROWS, HD]),
                TkTensorArg::new($b.silu_buf, &[1, 1, ACT_ROWS, ID]),
                TkTensorArg::new($b.rms_lm, &[1, 1, ACT_ROWS, HD]),
                TkTensorArg::new($b.logits, &[1, 1, ACT_ROWS, VS]),
                // Decode KV metadata
                TkTensorArg::new($b.pos_ids, &[ACT_ROWS]),
                TkTensorArg::new($b.kv_indptr, &[ACT_ROWS + 1]),
                TkTensorArg::new($b.kv_indices, &[NUM_PAGES]),
                TkTensorArg::new($b.kv_last_page, &[ACT_ROWS]),
                TkTensorArg::new($b.kv_append, &[ACT_ROWS]),
                // Prefill KV metadata (dummy)
                TkTensorArg::new($b.dummy_meta, &[1]),
                TkTensorArg::new($b.dummy_meta, &[1]),
                TkTensorArg::new($b.dummy_meta, &[1]),
                TkTensorArg::new($b.dummy_meta, &[1]),
                // Scalars
                1.0 / (HDM as f32).sqrt(), // attn_scale
                1e-5_f32,                  // rms_norm_eps
                NUM_PAGES as i32,          // num_pages
                BS as i32,                 // batch_size
                0_i32,                     // num_prefill_tokens
                NL as i32,                 // num_layers
                0_u64,                     // stream (default)
            )
        }
    };
}

// ── Tests ──

#[test]
#[ignore] // Requires GPU
fn test_attn_norm_single_op() {
    init_cuda();
    let b = TestBuffers::new();
    let rc = call_launch!(ffi::test_attn_norm_launch, b);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "test_attn_norm_launch returned error {rc}");
}

#[test]
#[ignore] // Requires GPU
fn test_mlp_norm_single_op() {
    init_cuda();
    let b = TestBuffers::new();
    let rc = call_launch!(ffi::test_mlp_norm_launch, b);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "test_mlp_norm_launch returned error {rc}");
}

#[test]
#[ignore] // Requires GPU
fn test_lm_head_norm_single_op() {
    init_cuda();
    let b = TestBuffers::new();
    let rc = call_launch!(ffi::test_lm_head_norm_launch, b);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "test_lm_head_norm_launch returned error {rc}");
}

#[test]
#[ignore] // Requires GPU
fn test_gate_silu_single_op() {
    init_cuda();
    let b = TestBuffers::new();
    let rc = call_launch!(ffi::test_gate_silu_launch, b);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "test_gate_silu_launch returned error {rc}");
}

#[test]
#[ignore] // Requires GPU
fn test_o_proj_residual_single_op() {
    init_cuda();
    let b = TestBuffers::new();
    let rc = call_launch!(ffi::test_o_proj_residual_launch, b);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "test_o_proj_residual_launch returned error {rc}");
}

#[test]
#[ignore] // Requires GPU
fn test_up_matmul_single_op() {
    init_cuda();
    let b = TestBuffers::new();
    let rc = call_launch!(ffi::test_up_matmul_launch, b);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "test_up_matmul_launch returned error {rc}");
}

#[test]
#[ignore] // Requires GPU
fn test_down_proj_residual_single_op() {
    init_cuda();
    let b = TestBuffers::new();
    let rc = call_launch!(ffi::test_down_proj_residual_launch, b);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "test_down_proj_residual_launch returned error {rc}");
}

#[test]
#[ignore] // Requires GPU
fn test_qkv_rope_append_single_op() {
    init_cuda();
    let b = TestBuffers::new();
    let rc = call_launch!(ffi::test_qkv_rope_append_launch, b);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "test_qkv_rope_append_launch returned error {rc}");
}

#[test]
#[ignore] // Requires GPU
fn test_attention_decode_single_op() {
    init_cuda();
    let b = TestBuffers::new();
    let rc = call_launch!(ffi::test_attention_decode_launch, b);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "test_attention_decode_launch returned error {rc}");
}

#[test]
#[ignore] // Requires GPU
fn test_lm_head_single_op() {
    init_cuda();
    let b = TestBuffers::new();
    let rc = call_launch!(ffi::test_lm_head_launch, b);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "test_lm_head_launch returned error {rc}");
}

// ── bf16 helpers ──

/// Upload f32 data as bf16 to GPU, return device pointer.
fn gpu_upload_bf16(data: &[f32]) -> u64 {
    let bf16_data: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v)).collect();
    unsafe {
        let bytes = bf16_data.len() * 2;
        let dptr = result::malloc_sync(bytes).expect("cuMemAlloc failed");
        result::memcpy_htod_sync(dptr, &bf16_data).expect("cuMemcpyHtoD failed");
        dptr as u64
    }
}

/// Download bf16 data from GPU and convert to f32.
fn gpu_download_bf16(dptr: u64, count: usize) -> Vec<f32> {
    let mut bf16_data = vec![bf16::ZERO; count];
    unsafe {
        result::memcpy_dtoh_sync(&mut bf16_data, dptr as cudarc::driver::sys::CUdeviceptr)
            .expect("cuMemcpyDtoH failed");
    }
    bf16_data.iter().map(|v| v.to_f32()).collect()
}

/// Assert two f32 slices are element-wise close.
fn assert_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    let mut max_abs_err = 0.0_f32;
    let mut max_rel_err = 0.0_f32;
    let mut worst_idx = 0;
    for i in 0..actual.len() {
        let abs_err = (actual[i] - expected[i]).abs();
        let rel_err = abs_err / (expected[i].abs() + 1e-8);
        if abs_err > max_abs_err {
            max_abs_err = abs_err;
            worst_idx = i;
        }
        if rel_err > max_rel_err {
            max_rel_err = rel_err;
        }
        assert!(
            abs_err <= atol || rel_err <= rtol,
            "{label}[{i}]: actual={} expected={} abs_err={abs_err} rel_err={rel_err}",
            actual[i],
            expected[i]
        );
    }
    eprintln!(
        "{label}: PASS (max_abs_err={max_abs_err:.6} at [{worst_idx}], max_rel_err={max_rel_err:.6})"
    );
}

/// Round f32 through bf16 to match GPU precision.
fn bf16_roundtrip(data: &[f32]) -> Vec<f32> {
    data.iter().map(|&v| bf16::from_f32(v).to_f32()).collect()
}

// ── Golden comparison tests ──

/// Generate deterministic test input data.
fn gen_input(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * 0.017).sin() * 0.5 + 0.5)
        .collect()
}

/// Generate deterministic weight data (positive, near 1.0).
fn gen_weight(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 1.0 + (i as f32 * 0.003).cos() * 0.1)
        .collect()
}

/// Test buffers with controlled inputs for golden comparison.
/// Only allocates the specific buffers needed, fills the rest with zeros.
struct GoldenBuffers {
    inner: TestBuffers,
    /// f32 values that were uploaded (after bf16 roundtrip) for CPU golden comparison.
    input_f32: Vec<f32>,
    weight_f32: Vec<f32>,
}

impl GoldenBuffers {
    /// Create buffers for an rmsnorm golden test.
    /// `input_ptr_fn` selects which buffer gets the input data.
    /// `weight_ptr_fn` selects which buffer gets the weight data.
    fn for_rmsnorm(
        input_setter: impl FnOnce(&mut TestBuffers, u64),
        weight_setter: impl FnOnce(&mut TestBuffers, u64),
    ) -> Self {
        let input_f32_raw = gen_input(HD);
        let weight_f32_raw = gen_weight(HD);

        // Round-trip through bf16 so CPU golden matches GPU precision
        let input_f32 = bf16_roundtrip(&input_f32_raw);
        let weight_f32 = bf16_roundtrip(&weight_f32_raw);

        let mut inner = TestBuffers::new();

        // Upload known data (need NL copies of weight for the [NL, HD] layout)
        let _input_dptr = gpu_upload_bf16(&input_f32);
        // For weights: need [NL, HD] — replicate the weight vector NL times
        let weight_full: Vec<f32> = weight_f32.iter().copied().cycle().take(NL * HD).collect();
        let weight_dptr = gpu_upload_bf16(&weight_full);

        // For input: need [1, 1, ACT_ROWS, HD] — put our data in row 0, rest zeros
        // We upload to a fresh buffer sized [ACT_ROWS, HD], first row is our data
        let mut input_full = vec![0.0_f32; ACT_ROWS * HD];
        input_full[..HD].copy_from_slice(&input_f32);
        let input_full_dptr = gpu_upload_bf16(&input_full);

        input_setter(&mut inner, input_full_dptr);
        weight_setter(&mut inner, weight_dptr);

        Self {
            inner,
            input_f32,
            weight_f32,
        }
    }
}

#[test]
#[ignore] // Requires GPU
fn test_attn_norm_golden() {
    use vllm_tk_macros_core::cpu_golden;

    init_cuda();
    let gb = GoldenBuffers::for_rmsnorm(|b, ptr| b.hidden = ptr, |b, ptr| b.attn_norm_w = ptr);

    let rc = call_launch!(ffi::test_attn_norm_launch, gb.inner);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "test_attn_norm_launch returned error {rc}");

    // Read back output (rms_rope buffer, first row)
    let gpu_out = gpu_download_bf16(gb.inner.rms_rope, ACT_ROWS * HD);
    let gpu_row0 = &gpu_out[..HD];

    // CPU golden
    let mut expected = vec![0.0_f32; HD];
    cpu_golden::rmsnorm(&gb.input_f32, &gb.weight_f32, &mut expected, 1e-5);

    assert_close(gpu_row0, &expected, 5e-2, 5e-2, "attn_norm_golden");
}

#[test]
#[ignore] // Requires GPU
fn test_mlp_norm_golden() {
    use vllm_tk_macros_core::cpu_golden;

    init_cuda();
    // mlp_norm reads from hidden (after o_proj residual), writes to rms_gate.
    // In our single-op test, hidden has our test input.
    let gb = GoldenBuffers::for_rmsnorm(|b, ptr| b.hidden = ptr, |b, ptr| b.mlp_norm_w = ptr);

    let rc = call_launch!(ffi::test_mlp_norm_launch, gb.inner);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "test_mlp_norm_launch returned error {rc}");

    let gpu_out = gpu_download_bf16(gb.inner.rms_gate, ACT_ROWS * HD);
    let gpu_row0 = &gpu_out[..HD];

    let mut expected = vec![0.0_f32; HD];
    cpu_golden::rmsnorm(&gb.input_f32, &gb.weight_f32, &mut expected, 1e-5);

    assert_close(gpu_row0, &expected, 5e-2, 5e-2, "mlp_norm_golden");
}

#[test]
#[ignore] // Requires GPU
fn test_lm_head_norm_golden() {
    use vllm_tk_macros_core::cpu_golden;

    init_cuda();
    // lm_head_norm reads from hidden, writes to rms_lm.
    // Weight is [1, HD] (not per-layer).
    let input_f32 = bf16_roundtrip(&gen_input(HD));
    let weight_f32 = bf16_roundtrip(&gen_weight(HD));

    let mut b = TestBuffers::new();

    // Input: [1, 1, ACT_ROWS, HD]
    let mut input_full = vec![0.0_f32; ACT_ROWS * HD];
    input_full[..HD].copy_from_slice(&input_f32);
    b.hidden = gpu_upload_bf16(&input_full);

    // Weight: [1, HD] (single layer)
    b.lm_norm_w = gpu_upload_bf16(&weight_f32);

    let rc = call_launch!(ffi::test_lm_head_norm_launch, b);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "test_lm_head_norm_launch returned error {rc}");

    let gpu_out = gpu_download_bf16(b.rms_lm, ACT_ROWS * HD);
    let gpu_row0 = &gpu_out[..HD];

    let mut expected = vec![0.0_f32; HD];
    cpu_golden::rmsnorm(&input_f32, &weight_f32, &mut expected, 1e-5);

    assert_close(gpu_row0, &expected, 5e-2, 5e-2, "lm_head_norm_golden");
}

// ════════════════════════════════════════════════════════════════════
// Inline kernel tests (static tile pipeline, no KVM protocol)
// ════════════════════════════════════════════════════════════════════

#[test]
#[ignore] // Requires GPU
fn test_inline_rmsnorm_golden() {
    use vllm_tk_macros_core::cpu_golden;

    init_cuda();
    let gb = GoldenBuffers::for_rmsnorm(|b, ptr| b.hidden = ptr, |b, ptr| b.attn_norm_w = ptr);

    let rc = call_launch!(ffi::inline_rmsnorm_launch, gb.inner);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "inline_rmsnorm_launch returned error {rc}");

    // Read back output (rms_rope buffer, first row)
    let gpu_out = gpu_download_bf16(gb.inner.rms_rope, ACT_ROWS * HD);
    let gpu_row0 = &gpu_out[..HD];

    // CPU golden
    let mut expected = vec![0.0_f32; HD];
    cpu_golden::rmsnorm(&gb.input_f32, &gb.weight_f32, &mut expected, 1e-5);

    assert_close(gpu_row0, &expected, 5e-2, 5e-2, "inline_rmsnorm_golden");
}

#[test]
#[ignore] // Requires GPU
fn test_inline_gemm_golden() {
    use vllm_tk_macros_core::cpu_golden;

    init_cuda();

    const OUT_BLOCK: usize = 64;

    // Input: [ACT_ROWS, HD] — random data in all rows (GEMM processes all 128)
    let input_f32_raw: Vec<f32> = {
        let mut rng = 42u64;
        (0..ACT_ROWS * HD)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng >> 33) as i32 as f32) / (i32::MAX as f32) * 0.1
            })
            .collect()
    };
    let input_f32 = bf16_roundtrip(&input_f32_raw);

    // Weight: [QKV_DIM, HD] for layer 0 — we only test the first OUT_BLOCK rows
    let weight_f32_raw: Vec<f32> = {
        let mut rng = 123u64;
        (0..QKV_DIM * HD)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng >> 33) as i32 as f32) / (i32::MAX as f32) * 0.01
            })
            .collect()
    };
    let weight_f32 = bf16_roundtrip(&weight_f32_raw);

    let mut b = TestBuffers::new();

    // Upload input to hidden_states
    b.hidden = gpu_upload_bf16(&input_f32);

    // Upload weight to qkv_w: need [NL, QKV_DIM, HD], replicate for all layers
    let weight_full: Vec<f32> = weight_f32.iter().copied().cycle().take(NL * QKV_DIM * HD).collect();
    b.qkv_w = gpu_upload_bf16(&weight_full);

    let rc = call_launch!(ffi::inline_gemm_launch, b);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "inline_gemm_launch returned error {rc}");

    // Read back output from rms_rope: [ACT_ROWS, HD] but only first OUT_BLOCK cols used
    let gpu_out = gpu_download_bf16(b.rms_rope, ACT_ROWS * HD);

    // CPU golden: output[ACT_ROWS, OUT_BLOCK] = input[ACT_ROWS, HD] @ weight[OUT_BLOCK, HD]^T
    let weight_slice = &weight_f32[..OUT_BLOCK * HD]; // first 64 rows
    let mut expected = vec![0.0_f32; ACT_ROWS * OUT_BLOCK];
    cpu_golden::gemm(&input_f32, weight_slice, &mut expected, ACT_ROWS, HD, OUT_BLOCK);

    // The GPU output is stored in rms_rope which is [ACT_ROWS, HD].
    // The kernel writes 64 columns starting at col offset 0 in each row's tile.
    // rms_rope store uses {row * (BATCH_BLOCK/16) + wid, col} — each warp stores
    // a 16×64 tile. In the rms_rope [ACT_ROWS, HD] layout, each 16-row tile at
    // col offset 0 writes 64 bf16 values = columns [0..64) of that row group.
    // So row i's output is at gpu_out[i * HD .. i * HD + OUT_BLOCK].
    for row in 0..ACT_ROWS {
        let gpu_row = &gpu_out[row * HD..row * HD + OUT_BLOCK];
        let exp_row = &expected[row * OUT_BLOCK..(row + 1) * OUT_BLOCK];
        assert_close(
            gpu_row,
            exp_row,
            5e-2,
            5e-2,
            &format!("inline_gemm_golden row {row}"),
        );
    }
}

#[test]
#[ignore] // Requires GPU
fn test_fused_rmsnorm_gemm_golden() {
    use vllm_tk_macros_core::cpu_golden;

    init_cuda();

    const OUT_BLOCK: usize = 64;

    // Input: single token [1, HD]
    let input_f32 = bf16_roundtrip(&gen_input(HD));
    // Norm weights: [NL, HD] — layer 0 used
    let norm_weight_f32 = bf16_roundtrip(&gen_weight(HD));
    // QKV weights: [QKV_DIM, HD] — first OUT_BLOCK rows used
    let qkv_weight_f32: Vec<f32> = bf16_roundtrip(&{
        let mut rng = 123u64;
        (0..QKV_DIM * HD)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng >> 33) as i32 as f32) / (i32::MAX as f32) * 0.01
            })
            .collect::<Vec<f32>>()
    });

    let mut b = TestBuffers::new();

    // Upload input to hidden_states: [ACT_ROWS, HD], row 0 has data
    let mut input_full = vec![0.0_f32; ACT_ROWS * HD];
    input_full[..HD].copy_from_slice(&input_f32);
    b.hidden = gpu_upload_bf16(&input_full);

    // Upload norm weights: [NL, HD]
    let norm_full: Vec<f32> = norm_weight_f32.iter().copied().cycle().take(NL * HD).collect();
    b.attn_norm_w = gpu_upload_bf16(&norm_full);

    // Upload QKV weights: [NL, QKV_DIM, HD]
    let qkv_full: Vec<f32> = qkv_weight_f32.iter().copied().cycle().take(NL * QKV_DIM * HD).collect();
    b.qkv_w = gpu_upload_bf16(&qkv_full);

    let rc = call_launch!(ffi::fused_rmsnorm_gemm_launch, b);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "fused_rmsnorm_gemm_launch returned error {rc}");

    // Read back output from rms_rope (fused kernel writes GEMM output there)
    let gpu_out = gpu_download_bf16(b.rms_rope, ACT_ROWS * HD);

    // CPU golden: rmsnorm then gemm
    let mut normed = vec![0.0_f32; HD];
    cpu_golden::rmsnorm(&input_f32, &norm_weight_f32, &mut normed, 1e-5);

    // GEMM: normed[1, HD] @ qkv_weight[OUT_BLOCK, HD]^T → [1, OUT_BLOCK]
    let weight_slice = &qkv_weight_f32[..OUT_BLOCK * HD];
    let mut expected = vec![0.0_f32; OUT_BLOCK];
    cpu_golden::gemm(&normed, weight_slice, &mut expected, 1, HD, OUT_BLOCK);

    // GPU output: GEMM writes 128×64 to rms_rope (same as standalone GEMM).
    // Only row 0 has real input data; check row 0 cols [0..OUT_BLOCK).
    let gpu_row0 = &gpu_out[..OUT_BLOCK];
    assert_close(gpu_row0, &expected, 5e-2, 5e-2, "fused_rmsnorm_gemm_golden");
}

/// Two-step test: run inline_rmsnorm then inline_gemm separately.
/// Verifies that RMSNorm's gmem output is correctly readable by GEMM.
#[test]
#[ignore] // Requires GPU
fn test_two_step_rmsnorm_then_gemm() {
    use vllm_tk_macros_core::cpu_golden;

    init_cuda();

    const OUT_BLOCK: usize = 64;

    // Same inputs as standalone rmsnorm test
    let input_f32 = bf16_roundtrip(&gen_input(HD));
    let norm_weight_f32 = bf16_roundtrip(&gen_weight(HD));

    let mut b = TestBuffers::new();

    // Upload hidden_states
    let mut input_full = vec![0.0_f32; ACT_ROWS * HD];
    input_full[..HD].copy_from_slice(&input_f32);
    b.hidden = gpu_upload_bf16(&input_full);

    // Upload norm weights
    let norm_full: Vec<f32> = norm_weight_f32.iter().copied().cycle().take(NL * HD).collect();
    b.attn_norm_w = gpu_upload_bf16(&norm_full);

    // Step 1: Run inline_rmsnorm → writes to rms_rope
    let rc = call_launch!(ffi::inline_rmsnorm_launch, b);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "inline_rmsnorm_launch returned error {rc}");

    // Read back rmsnorm output
    let rmsnorm_out = gpu_download_bf16(b.rms_rope, ACT_ROWS * HD);
    let rmsnorm_row0 = &rmsnorm_out[..HD];

    // Verify rmsnorm matches CPU
    let mut normed = vec![0.0_f32; HD];
    cpu_golden::rmsnorm(&input_f32, &norm_weight_f32, &mut normed, 1e-5);
    eprintln!("RMSNorm row0[0..4]: GPU={:?}, CPU={:?}", &rmsnorm_row0[..4], &normed[..4]);

    // Step 2: Copy rmsnorm output to hidden_states (inline_gemm reads from hidden_states)
    b.hidden = gpu_upload_bf16(&rmsnorm_out);

    // Set up QKV weights
    let qkv_weight_f32: Vec<f32> = bf16_roundtrip(&{
        let mut rng = 123u64;
        (0..QKV_DIM * HD)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng >> 33) as i32 as f32) / (i32::MAX as f32) * 0.01
            })
            .collect::<Vec<f32>>()
    });
    let qkv_full: Vec<f32> = qkv_weight_f32.iter().copied().cycle().take(NL * QKV_DIM * HD).collect();
    b.qkv_w = gpu_upload_bf16(&qkv_full);

    // Run inline_gemm → writes to rms_rope
    let rc = call_launch!(ffi::inline_gemm_launch, b);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "inline_gemm_launch returned error {rc}");

    // Read back GEMM output
    let gpu_out = gpu_download_bf16(b.rms_rope, ACT_ROWS * HD);

    // CPU golden: GEMM on row 0 only (inline_gemm computes all 128 rows)
    let weight_slice = &qkv_weight_f32[..OUT_BLOCK * HD];
    let mut expected_full = vec![0.0_f32; ACT_ROWS * OUT_BLOCK];
    cpu_golden::gemm(&rmsnorm_out, weight_slice, &mut expected_full, ACT_ROWS, HD, OUT_BLOCK);

    let gpu_row0 = &gpu_out[..OUT_BLOCK];
    let exp_row0 = &expected_full[..OUT_BLOCK];
    eprintln!("GEMM row0[0..4]: GPU={:?}, CPU={:?}", &gpu_row0[..4], &exp_row0[..4]);

    assert_close(gpu_row0, exp_row0, 5e-2, 5e-2, "two_step_rmsnorm_gemm");
}

#[test]
#[ignore] // Requires GPU
fn test_fused_mlp_golden() {
    use vllm_tk_macros_core::cpu_golden;

    init_cuda();

    fn make_random(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut rng = seed;
        (0..n)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng >> 33) as i32 as f32) / (i32::MAX as f32) * scale
            })
            .collect()
    }

    // Input: hidden_states [ACT_ROWS, HD] — row 0 has data
    let input_f32 = bf16_roundtrip(&gen_input(HD));
    // Weights
    let mlp_norm_w = bf16_roundtrip(&gen_weight(HD));
    let gate_w = bf16_roundtrip(&make_random(ID * HD, 200, 0.01));
    let up_w = bf16_roundtrip(&make_random(ID * HD, 300, 0.01));
    let down_w = bf16_roundtrip(&make_random(HD * ID, 400, 0.01));

    let mut b = TestBuffers::new();

    // Upload
    let mut input_full = vec![0.0_f32; ACT_ROWS * HD];
    input_full[..HD].copy_from_slice(&input_f32);
    b.hidden = gpu_upload_bf16(&input_full);

    let norm_full: Vec<f32> = mlp_norm_w.iter().copied().cycle().take(NL * HD).collect();
    b.mlp_norm_w = gpu_upload_bf16(&norm_full);

    let gate_full: Vec<f32> = gate_w.iter().copied().cycle().take(NL * ID * HD).collect();
    b.gate_w = gpu_upload_bf16(&gate_full);

    let up_full: Vec<f32> = up_w.iter().copied().cycle().take(NL * ID * HD).collect();
    b.up_w = gpu_upload_bf16(&up_full);

    let down_full: Vec<f32> = down_w.iter().copied().cycle().take(NL * HD * ID).collect();
    b.down_w = gpu_upload_bf16(&down_full);

    let rc = call_launch!(ffi::fused_mlp_launch, b);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "fused_mlp_launch returned error {rc}");

    // Read back hidden_states (down_proj writes output = matmul + residual → hidden)
    let gpu_out = gpu_download_bf16(b.hidden, ACT_ROWS * HD);
    let gpu_row0 = &gpu_out[..HD];

    // CPU golden chain
    // 1. mlp_norm
    let mut normed = vec![0.0_f32; HD];
    cpu_golden::rmsnorm(&input_f32, &mlp_norm_w, &mut normed, 1e-5);
    let normed = bf16_roundtrip(&normed); // match GPU precision

    // 2. gate GEMM: normed[1,HD] × gate_w[ID,HD]^T → [1,ID]
    let mut gate_out = vec![0.0_f32; ID];
    cpu_golden::gemm(&normed, &gate_w, &mut gate_out, 1, HD, ID);
    let gate_out = bf16_roundtrip(&gate_out);

    // 3. SiLU on gate
    let mut gate_silu = vec![0.0_f32; ID];
    cpu_golden::silu(&gate_out, &mut gate_silu);
    let gate_silu = bf16_roundtrip(&gate_silu);

    // 4. up GEMM: normed[1,HD] × up_w[ID,HD]^T → [1,ID]
    let mut up_out = vec![0.0_f32; ID];
    cpu_golden::gemm(&normed, &up_w, &mut up_out, 1, HD, ID);
    let up_out = bf16_roundtrip(&up_out);

    // 5. gate * up
    let mut mlp_inter = vec![0.0_f32; ID];
    cpu_golden::mul(&gate_silu, &up_out, &mut mlp_inter);
    let mlp_inter = bf16_roundtrip(&mlp_inter);

    // 6. down GEMM + residual: mlp_inter[1,ID] × down_w[HD,ID]^T + input → [1,HD]
    let mut expected = vec![0.0_f32; HD];
    cpu_golden::gemm_add(&mlp_inter, &down_w, &input_f32, &mut expected, 1, ID, HD);

    eprintln!("MLP GPU[0..4]: {:?}", &gpu_row0[..4]);
    eprintln!("MLP CPU[0..4]: {:?}", &expected[..4]);

    assert_close(gpu_row0, &expected, 1.0, 0.1, "fused_mlp_golden");
}

#[test]
#[ignore = "needs GPU"]
#[cfg(feature = "cuda")]
fn test_fused_layer_golden() {
    use vllm_tk_macros_core::cpu_golden;

    init_cuda();

    fn make_random(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut rng = seed;
        (0..n)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng >> 33) as i32 as f32) / (i32::MAX as f32) * scale
            })
            .collect()
    }

    // Inputs
    let input_f32 = bf16_roundtrip(&gen_input(HD));
    // Weights
    let attn_norm_w = bf16_roundtrip(&gen_weight(HD));
    let qkv_w = bf16_roundtrip(&make_random(HD * HD, 100, 0.01)); // HD×HD (testing first HD cols)
    let o_w = bf16_roundtrip(&make_random(HD * HD, 150, 0.01));
    let attn_out_data = bf16_roundtrip(&make_random(HD, 175, 0.5)); // fake attention output
    let mlp_norm_w = bf16_roundtrip(&gen_weight(HD));
    let gate_w = bf16_roundtrip(&make_random(ID * HD, 200, 0.01));
    let up_w = bf16_roundtrip(&make_random(ID * HD, 300, 0.01));
    let down_w = bf16_roundtrip(&make_random(HD * ID, 400, 0.01));

    let mut b = TestBuffers::new();

    // Upload hidden_states
    let mut input_full = vec![0.0_f32; ACT_ROWS * HD];
    input_full[..HD].copy_from_slice(&input_f32);
    b.hidden = gpu_upload_bf16(&input_full);

    // Upload weights (replicated across NL layers)
    let anw: Vec<f32> = attn_norm_w.iter().copied().cycle().take(NL * HD).collect();
    b.attn_norm_w = gpu_upload_bf16(&anw);

    let qw: Vec<f32> = qkv_w.iter().copied().cycle().take(NL * HD * HD).collect();
    b.qkv_w = gpu_upload_bf16(&qw);

    let ow: Vec<f32> = o_w.iter().copied().cycle().take(NL * HD * HD).collect();
    b.o_w = gpu_upload_bf16(&ow);

    // Upload fake attn_out
    let mut attn_full = vec![0.0_f32; ACT_ROWS * HD];
    attn_full[..HD].copy_from_slice(&attn_out_data);
    b.attn_out = gpu_upload_bf16(&attn_full);

    let mnw: Vec<f32> = mlp_norm_w.iter().copied().cycle().take(NL * HD).collect();
    b.mlp_norm_w = gpu_upload_bf16(&mnw);

    let gw: Vec<f32> = gate_w.iter().copied().cycle().take(NL * ID * HD).collect();
    b.gate_w = gpu_upload_bf16(&gw);

    let uw: Vec<f32> = up_w.iter().copied().cycle().take(NL * ID * HD).collect();
    b.up_w = gpu_upload_bf16(&uw);

    let dw: Vec<f32> = down_w.iter().copied().cycle().take(NL * HD * ID).collect();
    b.down_w = gpu_upload_bf16(&dw);

    let rc = call_launch!(ffi::fused_layer_launch, b);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "fused_layer_launch returned error {rc}");

    // Read back final hidden_states
    let gpu_out = gpu_download_bf16(b.hidden, ACT_ROWS * HD);
    let gpu_row0 = &gpu_out[..HD];

    // ── CPU golden chain ──

    // 1. attn_norm
    let mut normed = vec![0.0_f32; HD];
    cpu_golden::rmsnorm(&input_f32, &attn_norm_w, &mut normed, 1e-5);
    let normed = bf16_roundtrip(&normed);

    // 2. QKV GEMM: normed[1,HD] × qkv_w[HD,HD]^T → [1,HD]
    let mut qkv_out = vec![0.0_f32; HD];
    cpu_golden::gemm(&normed, &qkv_w, &mut qkv_out, 1, HD, HD);
    let _qkv_out = bf16_roundtrip(&qkv_out);

    // 3. [skip attention — use fake attn_out_data directly]

    // 4. o_proj + residual: attn_out[1,HD] × o_w[HD,HD]^T + hidden → hidden
    let mut hidden_after_attn = vec![0.0_f32; HD];
    cpu_golden::gemm_add(&attn_out_data, &o_w, &input_f32, &mut hidden_after_attn, 1, HD, HD);
    let hidden_after_attn = bf16_roundtrip(&hidden_after_attn);

    // 5. mlp_norm
    let mut mlp_normed = vec![0.0_f32; HD];
    cpu_golden::rmsnorm(&hidden_after_attn, &mlp_norm_w, &mut mlp_normed, 1e-5);
    let mlp_normed = bf16_roundtrip(&mlp_normed);

    // 6. gate GEMM + SiLU
    let mut gate_out = vec![0.0_f32; ID];
    cpu_golden::gemm(&mlp_normed, &gate_w, &mut gate_out, 1, HD, ID);
    let gate_out = bf16_roundtrip(&gate_out);
    let mut gate_silu = vec![0.0_f32; ID];
    cpu_golden::silu(&gate_out, &mut gate_silu);
    let gate_silu = bf16_roundtrip(&gate_silu);

    // 7. up GEMM × gate
    let mut up_out = vec![0.0_f32; ID];
    cpu_golden::gemm(&mlp_normed, &up_w, &mut up_out, 1, HD, ID);
    let up_out = bf16_roundtrip(&up_out);
    let mut mlp_inter = vec![0.0_f32; ID];
    cpu_golden::mul(&gate_silu, &up_out, &mut mlp_inter);
    let mlp_inter = bf16_roundtrip(&mlp_inter);

    // 8. down GEMM + residual
    let mut expected = vec![0.0_f32; HD];
    cpu_golden::gemm_add(&mlp_inter, &down_w, &hidden_after_attn, &mut expected, 1, ID, HD);

    eprintln!("Layer GPU[0..4]: {:?}", &gpu_row0[..4]);
    eprintln!("Layer CPU[0..4]: {:?}", &expected[..4]);

    assert_close(gpu_row0, &expected, 1.5, 0.15, "fused_layer_golden");
}

/// Upload i32 data to GPU.
fn gpu_upload_i32(data: &[i32]) -> u64 {
    unsafe {
        let bytes = data.len() * 4;
        let dptr = result::malloc_sync(bytes).expect("cuMemAlloc failed");
        result::memcpy_htod_sync(dptr, data).expect("cuMemcpyHtoD failed");
        dptr as u64
    }
}

#[test]
#[ignore = "needs GPU"]
#[cfg(feature = "cuda")]
fn test_inline_attention_decode_golden() {
    use vllm_tk_macros_core::cpu_golden;

    init_cuda();

    fn make_random(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut rng = seed;
        (0..n)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng >> 33) as i32 as f32) / (i32::MAX as f32) * scale
            })
            .collect()
    }

    // Test setup: 1 sequence, 48 tokens (spans 1 page of 64 tokens, partial last page)
    let seq_len: usize = 48;
    let num_pages_used = (seq_len + KV_PAGE_SIZE - 1) / KV_PAGE_SIZE; // 1
    let last_page_len = seq_len - (num_pages_used - 1) * KV_PAGE_SIZE; // 48
    let attn_scale = 1.0 / (HDM as f32).sqrt();

    // Q: [NAH * HDM] = [32 * 64] = 2048 (full hidden dim, laid out as NAH heads of HDM each)
    let q_data = bf16_roundtrip(&make_random(NAH * HDM, 500, 0.5));

    // KV cache: [seq_len, NKH, HDM] for CPU golden
    let k_cache_flat = bf16_roundtrip(&make_random(seq_len * NKH * HDM, 600, 0.5));
    let v_cache_flat = bf16_roundtrip(&make_random(seq_len * NKH * HDM, 700, 0.5));

    // ── CPU golden ──
    let mut expected = vec![0.0_f32; NAH * HDM];
    cpu_golden::attention_decode(
        &q_data,
        &k_cache_flat,
        &v_cache_flat,
        &mut expected,
        seq_len,
        NAH,
        NKH,
        HDM,
        attn_scale,
    );

    // ── GPU setup ──
    let mut b = TestBuffers::new();

    // Upload Q to q_post (activations layout: [ACT_ROWS, HD])
    let mut q_full = vec![0.0_f32; ACT_ROWS * HD];
    q_full[..NAH * HDM].copy_from_slice(&q_data);
    b.q_post = gpu_upload_bf16(&q_full);

    // Upload KV cache in paged layout: [num_layers * NUM_PAGES, KV_PAGE_SIZE, NKH, HDM]
    // For our test: 1 page (page_id=0), layer=0
    // CPU golden layout: [seq_len, NKH, HDM] — need to rearrange to paged format
    let total_kv_cache_size = NL * NUM_PAGES * KV_PAGE_SIZE * NKH * HDM;
    let mut k_paged = vec![0.0_f32; total_kv_cache_size];
    let mut v_paged = vec![0.0_f32; total_kv_cache_size];

    // Copy seq_len tokens into page 0 (layer 0)
    // Paged layout: page_batch = num_pages * layer + page_index
    // Within page: [KV_PAGE_SIZE, NKH, HDM]
    let page_index = 0_usize;
    let page_batch = NUM_PAGES * 0 + page_index; // layer 0
    for tok in 0..seq_len {
        for kv_h in 0..NKH {
            for d in 0..HDM {
                let flat_idx = tok * NKH * HDM + kv_h * HDM + d;
                let paged_idx = page_batch * KV_PAGE_SIZE * NKH * HDM
                    + tok * NKH * HDM + kv_h * HDM + d;
                k_paged[paged_idx] = k_cache_flat[flat_idx];
                v_paged[paged_idx] = v_cache_flat[flat_idx];
            }
        }
    }
    b.k_cache = gpu_upload_bf16(&k_paged);
    b.v_cache = gpu_upload_bf16(&v_paged);

    // Paged KV metadata for 1 sequence, 1 page
    // indptr: [0, 1] — 1 page for sequence 0
    let kv_indptr = vec![0_i32, num_pages_used as i32];
    let mut indptr_full = vec![0_i32; ACT_ROWS + 1];
    indptr_full[..kv_indptr.len()].copy_from_slice(&kv_indptr);
    b.kv_indptr = gpu_upload_i32(&indptr_full);

    // indices: [0] — page 0
    let mut indices_full = vec![0_i32; NUM_PAGES];
    indices_full[0] = page_index as i32;
    b.kv_indices = gpu_upload_i32(&indices_full);

    // last_page_len: [48]
    let mut last_page_full = vec![0_i32; ACT_ROWS];
    last_page_full[0] = last_page_len as i32;
    b.kv_last_page = gpu_upload_i32(&last_page_full);

    // attn_out: zero buffer
    b.attn_out = gpu_alloc_zeros(ACT_ROWS * HD * BF16);

    let rc = call_launch!(ffi::inline_attention_decode_launch, b);
    unsafe { result::stream::synchronize(std::ptr::null_mut()).unwrap() };
    assert_eq!(rc, 0, "inline_attention_decode_launch returned error {rc}");

    // Read back attn_out
    let gpu_out = gpu_download_bf16(b.attn_out, ACT_ROWS * HD);
    let gpu_row0 = &gpu_out[..NAH * HDM];

    eprintln!("AttnDecode GPU[0..4]: {:?}", &gpu_row0[..4]);
    eprintln!("AttnDecode CPU[0..4]: {:?}", &expected[..4]);

    // Flash attention with bf16 MMA accumulates error across seq_len blocks
    // Tolerance is higher than GEMM due to softmax numerical sensitivity
    assert_close(gpu_row0, &expected, 2.0, 0.2, "inline_attention_decode_golden");
}
