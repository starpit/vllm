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
