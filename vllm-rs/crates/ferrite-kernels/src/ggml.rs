// SPDX-License-Identifier: Apache-2.0
//! GGML quantized types and CUDA kernel wrappers for GGUF inference.
//!
//! This module provides:
//! - `GgmlDType`: enum of GGML quantization types with `type_size()` / `block_size()`
//! - `GgmlStorage`: raw quantized bytes on GPU + metadata
//! - FFI wrappers calling the llama.cpp-derived kernels in `csrc/quantized.cu`
//!
//! Launch configs match llama.cpp's quantized kernel configs.

use ferrite_cuda_core::alloc::{CachingAllocator, OwnedTensor};
use ferrite_cuda_core::dtype::DType;
use ferrite_cuda_core::tensor::GpuTensor;
use std::sync::atomic::{AtomicBool, Ordering};

type CUstream = cudarc::driver::sys::CUstream;

// One flag per IQ type — set on first dispatch, so the log line prints exactly once.
static IQ1M_SEEN:   AtomicBool = AtomicBool::new(false);
static IQ1S_SEEN:   AtomicBool = AtomicBool::new(false);
static IQ2XXS_SEEN: AtomicBool = AtomicBool::new(false);
static IQ2S_SEEN:   AtomicBool = AtomicBool::new(false);
static IQ3S_SEEN:   AtomicBool = AtomicBool::new(false);

fn note_iq(flag: &AtomicBool, name: &str) {
    if !flag.swap(true, Ordering::Relaxed) {
        eprintln!("[ggml dispatch] first matmul via {name}");
    }
}

// ---------------------------------------------------------------------------
// GgmlDType
// ---------------------------------------------------------------------------

/// GGML quantization data types.
///
/// Matches the GGUF
/// on-disk format tags. Each variant knows its `type_size()` (bytes per block)
/// and `block_size()` (elements per block).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlDType {
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ1S   = 19,
    IQ4NL  = 20,
    IQ3S   = 21,
    IQ2S   = 22,
    IQ4XS  = 23,
    IQ1M   = 29,
}

impl GgmlDType {
    /// Create from the GGUF on-disk u32 tag.
    pub fn from_u32(u: u32) -> Option<Self> {
        match u {
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            16 => Some(Self::IQ2XXS),
            19 => Some(Self::IQ1S),
            20 => Some(Self::IQ4NL),
            21 => Some(Self::IQ3S),
            22 => Some(Self::IQ2S),
            23 => Some(Self::IQ4XS),
            29 => Some(Self::IQ1M),
            _ => None,
        }
    }

    /// Convert from a vendored GgufDType tag.
    pub fn from_gguf(dt: vllm_model::gguf_format::GgufDType) -> Option<Self> {
        Self::from_u32(dt.0)
    }

    /// Size in bytes of one quantization block.
    ///
    /// Values match llama.cpp block structs.
    pub const fn type_size(self) -> usize {
        match self {
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            Self::IQ2XXS => 66,
            Self::IQ1S   => 50,
            Self::IQ4NL  => 18,
            Self::IQ3S   => 110,
            Self::IQ2S   => 82,
            Self::IQ4XS  => 136,
            Self::IQ1M   => 56,
        }
    }

    /// Number of elements per quantization block.
    pub const fn block_size(self) -> usize {
        match self {
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ4NL => 32,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Q8K
            | Self::IQ2XXS
            | Self::IQ1S
            | Self::IQ3S
            | Self::IQ2S
            | Self::IQ4XS
            | Self::IQ1M => 256,
        }
    }

    /// Whether this is a "K-quant" type (QK_K=256 block size).
    pub const fn is_k_quant(self) -> bool {
        matches!(
            self,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K
        )
    }

    /// Whether this is an IQ (importance-matrix) quantization type.
    ///
    /// IQ types lack the fused dequant+dot BS=1 kernel — they must always go
    /// through the Q8_1 intermediate quantization path.
    pub const fn is_iq_quant(self) -> bool {
        matches!(
            self,
            Self::IQ4NL | Self::IQ4XS | Self::IQ1M | Self::IQ2XXS | Self::IQ1S | Self::IQ3S | Self::IQ2S
        )
    }
}

impl std::fmt::Display for GgmlDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Q4_0 => write!(f, "Q4_0"),
            Self::Q4_1 => write!(f, "Q4_1"),
            Self::Q5_0 => write!(f, "Q5_0"),
            Self::Q5_1 => write!(f, "Q5_1"),
            Self::Q8_0 => write!(f, "Q8_0"),
            Self::Q8_1 => write!(f, "Q8_1"),
            Self::Q2K => write!(f, "Q2K"),
            Self::Q3K => write!(f, "Q3K"),
            Self::Q4K => write!(f, "Q4K"),
            Self::Q5K => write!(f, "Q5K"),
            Self::Q6K => write!(f, "Q6K"),
            Self::Q8K => write!(f, "Q8K"),
            Self::IQ2XXS => write!(f, "IQ2_XXS"),
            Self::IQ1S   => write!(f, "IQ1_S"),
            Self::IQ4NL  => write!(f, "IQ4_NL"),
            Self::IQ3S   => write!(f, "IQ3_S"),
            Self::IQ2S   => write!(f, "IQ2_S"),
            Self::IQ4XS  => write!(f, "IQ4_XS"),
            Self::IQ1M   => write!(f, "IQ1_M"),
        }
    }
}

// ---------------------------------------------------------------------------
// GgmlStorage — raw quantized bytes on GPU
// ---------------------------------------------------------------------------

/// Raw GGML-quantized weight data on GPU.
///
/// The bytes are in the exact same format as the GGUF file — no dequantization.
/// The CUDA kernels read quantized blocks directly and do fused dequant+matvec.
#[derive(Clone, Copy)]
pub struct GgmlStorage {
    /// Raw GPU pointer to quantized block data.
    pub ptr: *mut u8,
    /// Total size in bytes.
    pub len: usize,
    /// Quantization type.
    pub dtype: GgmlDType,
    /// Number of rows (output features for a weight matrix).
    pub nrows: usize,
    /// Number of columns (input features) — the "logical" element count per row.
    pub ncols: usize,
}

unsafe impl Send for GgmlStorage {}
unsafe impl Sync for GgmlStorage {}

impl GgmlStorage {
    /// Total number of logical elements.
    pub fn numel(&self) -> usize {
        self.nrows * self.ncols
    }

    /// Verify that the stored byte count matches the expected size.
    pub fn verify_size(&self) -> bool {
        let expected_blocks = self.numel() / self.dtype.block_size();
        let expected_bytes = expected_blocks * self.dtype.type_size();
        self.len == expected_bytes
    }
}

impl std::fmt::Debug for GgmlStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GgmlStorage({}, [{}, {}], {} bytes, ptr={:p})",
            self.dtype, self.nrows, self.ncols, self.len, self.ptr
        )
    }
}

// ---------------------------------------------------------------------------
// Constants (match llama.cpp quantized kernels)
// ---------------------------------------------------------------------------

pub const MATRIX_ROW_PADDING: usize = 512;

pub fn pad(p: usize, q: usize) -> usize {
    p.div_ceil(q) * q
}

// ---------------------------------------------------------------------------
// FFI declarations — host-side launch wrappers from quantized.cu
// ---------------------------------------------------------------------------

unsafe extern "C" {
    // --- dequantize_mul_mat_vec wrappers ---
    fn launch_dequantize_mul_mat_vec_q4_0(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );
    fn launch_dequantize_mul_mat_vec_q4_1(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );
    fn launch_dequantize_mul_mat_vec_q5_0(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );
    fn launch_dequantize_mul_mat_vec_q5_1(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );
    fn launch_dequantize_mul_mat_vec_q8_0(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );
    fn launch_dequantize_mul_mat_vec_q2_k(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );
    fn launch_dequantize_mul_mat_vec_q3_k(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );
    fn launch_dequantize_mul_mat_vec_q4_k(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );
    fn launch_dequantize_mul_mat_vec_q5_k(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );
    fn launch_dequantize_mul_mat_vec_q6_k(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );

    // --- mul_mat_vec Q*×Q8_1 wrappers (BS=1) ---
    fn launch_mul_mat_vec_q4_0_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_q4_1_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_q5_0_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_q5_1_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_q8_0_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_q2_K_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_q3_K_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_q4_K_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_q5_K_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_q6_K_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: CUstream,
    );

    // --- quantize activations to Q8_1 ---
    fn launch_quantize_q8_1(
        src: *const f32,
        dst: *mut u8,
        k: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: CUstream,
    );

    // --- dequantize to f32 ---
    fn launch_dequantize_block_q4_0_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q4_1_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q5_0_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q5_1_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q8_0_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q2_K_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q3_K_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q4_K_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q5_K_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q6_K_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q8_K_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );

    // --- dequantize to f16 ---
    fn launch_dequantize_block_q4_0_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q4_1_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q5_0_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q5_1_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q8_0_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q2_K_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q3_K_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q4_K_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q5_K_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q6_K_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q8_K_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );

    // --- IQ4 mul_mat_vec Q*×Q8_1 wrappers ---
    fn launch_mul_mat_vec_iq4_nl_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_iq4_xs_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: CUstream,
    );

    // --- IQ4 dequantize to f32 ---
    fn launch_dequantize_block_iq4_nl_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq4_xs_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );

    // --- IQ4 dequantize to f16 ---
    fn launch_dequantize_block_iq4_nl_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq4_xs_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );

    // --- IQ1_M mul_mat_vec + dequantize wrappers ---
    fn launch_mul_mat_vec_iq1_m_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq1_m_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq1_m_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );

    // --- IQ1_S mul_mat_vec + dequantize wrappers ---
    fn launch_mul_mat_vec_iq1_s_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq1_s_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq1_s_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );

    // --- IQ2_XXS mul_mat_vec + dequantize wrappers ---
    fn launch_mul_mat_vec_iq2_xxs_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq2_xxs_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq2_xxs_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );

    // --- IQ2_S mul_mat_vec + dequantize wrappers ---
    fn launch_mul_mat_vec_iq2_s_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq2_s_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq2_s_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );

    // --- IQ3_S mul_mat_vec + dequantize wrappers ---
    fn launch_mul_mat_vec_iq3_s_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq3_s_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq3_s_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );

    // --- indexed_moe_forward wrappers ---
    fn launch_indexed_moe_forward_q2k_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
    fn launch_indexed_moe_forward_q3k_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
    fn launch_indexed_moe_forward_q4k_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
    fn launch_indexed_moe_forward_q5k_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
    fn launch_indexed_moe_forward_q6k_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
    fn launch_indexed_moe_forward_q8_0_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
    fn launch_indexed_moe_forward_q4_0_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
    fn launch_indexed_moe_forward_q4_1_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
    fn launch_indexed_moe_forward_q5_0_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
    fn launch_indexed_moe_forward_q5_1_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
}

// ---------------------------------------------------------------------------
// Kernel launch wrappers
// ---------------------------------------------------------------------------

/// Fused dequantize + matrix-vector multiply for a single input row (BS=1).
///
/// Computes: `dst[nrows] = weight[nrows, ncols] @ x[ncols]`
/// where `weight` is in GGML quantized format.
///
/// # Safety
/// All pointers must be valid GPU memory.
pub unsafe fn ggml_dequant_mul_mat_vec(
    storage: &GgmlStorage,
    x: *const f32,
    dst: *mut f32,
    stream: CUstream,
) {
    let ncols = storage.ncols as i32;
    let nrows = storage.nrows as i32;
    let vx = storage.ptr as *const u8;

    match storage.dtype {
        GgmlDType::Q4_0 => launch_dequantize_mul_mat_vec_q4_0(vx, x, dst, ncols, nrows, stream),
        GgmlDType::Q4_1 => launch_dequantize_mul_mat_vec_q4_1(vx, x, dst, ncols, nrows, stream),
        GgmlDType::Q5_0 => launch_dequantize_mul_mat_vec_q5_0(vx, x, dst, ncols, nrows, stream),
        GgmlDType::Q5_1 => launch_dequantize_mul_mat_vec_q5_1(vx, x, dst, ncols, nrows, stream),
        GgmlDType::Q8_0 => launch_dequantize_mul_mat_vec_q8_0(vx, x, dst, ncols, nrows, stream),
        GgmlDType::Q2K => launch_dequantize_mul_mat_vec_q2_k(vx, x, dst, ncols, nrows, stream),
        GgmlDType::Q3K => launch_dequantize_mul_mat_vec_q3_k(vx, x, dst, ncols, nrows, stream),
        GgmlDType::Q4K => launch_dequantize_mul_mat_vec_q4_k(vx, x, dst, ncols, nrows, stream),
        GgmlDType::Q5K => launch_dequantize_mul_mat_vec_q5_k(vx, x, dst, ncols, nrows, stream),
        GgmlDType::Q6K => launch_dequantize_mul_mat_vec_q6_k(vx, x, dst, ncols, nrows, stream),
        dt if dt.is_iq_quant() => panic!(
            "IQ types must use Q8_1 path, not dequant_mul_mat_vec: {}",
            storage.dtype
        ),
        _ => panic!(
            "unsupported dtype for dequant_mul_mat_vec: {}",
            storage.dtype
        ),
    }
}

/// Dequantize GGML data to f32 on GPU.
///
/// # Safety
/// `src` must point to valid quantized GPU data, `dst` must have room for `elem_count` f32s.
pub unsafe fn ggml_dequantize_f32(
    src: *const u8,
    dst: *mut f32,
    dtype: GgmlDType,
    elem_count: usize,
    stream: CUstream,
) {
    let n = elem_count as i32;
    match dtype {
        GgmlDType::Q4_0 => launch_dequantize_block_q4_0_f32(src, dst, n, stream),
        GgmlDType::Q4_1 => launch_dequantize_block_q4_1_f32(src, dst, n, stream),
        GgmlDType::Q5_0 => launch_dequantize_block_q5_0_f32(src, dst, n, stream),
        GgmlDType::Q5_1 => launch_dequantize_block_q5_1_f32(src, dst, n, stream),
        GgmlDType::Q8_0 => launch_dequantize_block_q8_0_f32(src, dst, n, stream),
        GgmlDType::Q2K => launch_dequantize_block_q2_K_f32(src, dst, n, stream),
        GgmlDType::Q3K => launch_dequantize_block_q3_K_f32(src, dst, n, stream),
        GgmlDType::Q4K => launch_dequantize_block_q4_K_f32(src, dst, n, stream),
        GgmlDType::Q5K => launch_dequantize_block_q5_K_f32(src, dst, n, stream),
        GgmlDType::Q6K => launch_dequantize_block_q6_K_f32(src, dst, n, stream),
        GgmlDType::Q8K => launch_dequantize_block_q8_K_f32(src, dst, n, stream),
        GgmlDType::IQ4NL  => launch_dequantize_block_iq4_nl_f32(src, dst, n, stream),
        GgmlDType::IQ4XS  => launch_dequantize_block_iq4_xs_f32(src, dst, n, stream),
        GgmlDType::IQ1M   => launch_dequantize_block_iq1_m_f32(src, dst, n, stream),
        GgmlDType::IQ1S   => launch_dequantize_block_iq1_s_f32(src, dst, n, stream),
        GgmlDType::IQ2XXS => launch_dequantize_block_iq2_xxs_f32(src, dst, n, stream),
        GgmlDType::IQ2S   => launch_dequantize_block_iq2_s_f32(src, dst, n, stream),
        GgmlDType::IQ3S   => launch_dequantize_block_iq3_s_f32(src, dst, n, stream),
        _ => panic!("unsupported dtype for dequantize_f32: {}", dtype),
    }
}

/// Dequantize GGML data to f16 on GPU.
///
/// # Safety
/// Same requirements as `ggml_dequantize_f32`.
pub unsafe fn ggml_dequantize_f16(
    src: *const u8,
    dst: *mut u16,
    dtype: GgmlDType,
    elem_count: usize,
    stream: CUstream,
) {
    let n = elem_count as i32;
    match dtype {
        GgmlDType::Q4_0 => launch_dequantize_block_q4_0_f16(src, dst, n, stream),
        GgmlDType::Q4_1 => launch_dequantize_block_q4_1_f16(src, dst, n, stream),
        GgmlDType::Q5_0 => launch_dequantize_block_q5_0_f16(src, dst, n, stream),
        GgmlDType::Q5_1 => launch_dequantize_block_q5_1_f16(src, dst, n, stream),
        GgmlDType::Q8_0 => launch_dequantize_block_q8_0_f16(src, dst, n, stream),
        GgmlDType::Q2K => launch_dequantize_block_q2_K_f16(src, dst, n, stream),
        GgmlDType::Q3K => launch_dequantize_block_q3_K_f16(src, dst, n, stream),
        GgmlDType::Q4K => launch_dequantize_block_q4_K_f16(src, dst, n, stream),
        GgmlDType::Q5K => launch_dequantize_block_q5_K_f16(src, dst, n, stream),
        GgmlDType::Q6K => launch_dequantize_block_q6_K_f16(src, dst, n, stream),
        GgmlDType::Q8K => launch_dequantize_block_q8_K_f16(src, dst, n, stream),
        GgmlDType::IQ4NL  => launch_dequantize_block_iq4_nl_f16(src, dst, n, stream),
        GgmlDType::IQ4XS  => launch_dequantize_block_iq4_xs_f16(src, dst, n, stream),
        GgmlDType::IQ1M   => launch_dequantize_block_iq1_m_f16(src, dst, n, stream),
        GgmlDType::IQ1S   => launch_dequantize_block_iq1_s_f16(src, dst, n, stream),
        GgmlDType::IQ2XXS => launch_dequantize_block_iq2_xxs_f16(src, dst, n, stream),
        GgmlDType::IQ2S   => launch_dequantize_block_iq2_s_f16(src, dst, n, stream),
        GgmlDType::IQ3S   => launch_dequantize_block_iq3_s_f16(src, dst, n, stream),
        _ => panic!("unsupported dtype for dequantize_f16: {}", dtype),
    }
}

/// Dequantize GGML data to a `GpuTensor` in the target dtype.
///
/// Allocates output via `CachingAllocator`. Used at load time for norms and embeddings.
///
/// # Safety
/// Requires valid CUDA context. `storage` must reference valid GPU memory.
pub unsafe fn ggml_dequantize_to_tensor(
    storage: &GgmlStorage,
    target_dtype: DType,
    shape: &[usize],
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let elem_count = shape.iter().product::<usize>();
    let out = alloc.alloc_tensor(shape, target_dtype);
    let dst_ptr = out.as_gpu_tensor().raw_ptr();

    match target_dtype {
        DType::F32 => {
            ggml_dequantize_f32(
                storage.ptr,
                dst_ptr as *mut f32,
                storage.dtype,
                elem_count,
                stream,
            );
        }
        DType::F16 => {
            ggml_dequantize_f16(
                storage.ptr,
                dst_ptr as *mut u16,
                storage.dtype,
                elem_count,
                stream,
            );
        }
        DType::BF16 => {
            // No direct quant→BF16 kernel. Dequant to F32, then cast F32→BF16.
            let f32_tmp = alloc.alloc_tensor(shape, DType::F32);
            ggml_dequantize_f32(
                storage.ptr,
                f32_tmp.as_gpu_tensor().raw_ptr() as *mut f32,
                storage.dtype,
                elem_count,
                stream,
            );
            crate::kernels::cast_from_f32_into(
                f32_tmp.as_gpu_tensor().raw_ptr() as *const f32,
                out.as_gpu_tensor().raw_ptr(),
                DType::BF16,
                elem_count,
                stream,
            );
            drop(f32_tmp);
        }
        _ => panic!(
            "unsupported target dtype for dequantize: {:?}",
            target_dtype
        ),
    }

    out
}

/// Quantize f32 activations to Q8_1 format on GPU.
///
/// Returns: `(gpu_ptr, total_bytes)` for the Q8_1 buffer.
///
/// # Safety
/// Valid CUDA context and GPU pointers required.
pub unsafe fn ggml_quantize_q8_1_alloc(
    src: *const f32,
    ncols: usize,
    num_rows: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> (*mut u8, usize) {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let q8_1_type_size = GgmlDType::Q8_1.type_size();
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let num_blocks_per_row = ncols_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;
    let total_bytes = num_rows * dst_row_size_bytes;

    let dst_ptr = alloc.alloc(total_bytes);

    launch_quantize_q8_1(
        src,
        dst_ptr,
        ncols as i32,
        ncols_padded as i32,
        num_rows as i32,
        stream,
    );

    (dst_ptr, total_bytes)
}

/// Batched quantized matvec via Q8_1 intermediate quantization (BS=1 per call).
///
/// # Safety
/// All pointers must be valid GPU memory.
pub unsafe fn ggml_mul_mat_vec_q8_1(
    storage: &GgmlStorage,
    y_q8_1: *const u8,
    ncols_padded: usize,
    batch_size: usize,
    dst: *mut f32,
    stream: CUstream,
) {
    let ncols_x = storage.ncols as i32;
    let nrows_x = storage.nrows as i32;
    let nrows_y = ncols_padded as i32;
    let nrows_dst = nrows_x;
    let vx = storage.ptr as *const u8;

    // Loop over each batch row (using BS=1 kernel).
    let q8_1_row_bytes =
        (ncols_padded / GgmlDType::Q8_1.block_size()) * GgmlDType::Q8_1.type_size();
    let dst_row_elems = storage.nrows;

    for b in 0..batch_size {
        let y_offset = y_q8_1.add(b * q8_1_row_bytes);
        let dst_offset = dst.add(b * dst_row_elems);
        match storage.dtype {
            GgmlDType::Q4_0 => launch_mul_mat_vec_q4_0_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, stream,
            ),
            GgmlDType::Q4_1 => launch_mul_mat_vec_q4_1_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, stream,
            ),
            GgmlDType::Q5_0 => launch_mul_mat_vec_q5_0_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, stream,
            ),
            GgmlDType::Q5_1 => launch_mul_mat_vec_q5_1_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, stream,
            ),
            GgmlDType::Q8_0 => launch_mul_mat_vec_q8_0_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, stream,
            ),
            GgmlDType::Q2K => launch_mul_mat_vec_q2_K_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, stream,
            ),
            GgmlDType::Q3K => launch_mul_mat_vec_q3_K_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, stream,
            ),
            GgmlDType::Q4K => launch_mul_mat_vec_q4_K_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, stream,
            ),
            GgmlDType::Q5K => launch_mul_mat_vec_q5_K_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, stream,
            ),
            GgmlDType::Q6K => launch_mul_mat_vec_q6_K_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, stream,
            ),
            GgmlDType::IQ4NL  => launch_mul_mat_vec_iq4_nl_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, stream,
            ),
            GgmlDType::IQ4XS  => launch_mul_mat_vec_iq4_xs_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, stream,
            ),
            GgmlDType::IQ1M   => { note_iq(&IQ1M_SEEN, "IQ1_M"); launch_mul_mat_vec_iq1_m_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, stream,
            )},
            GgmlDType::IQ1S   => { note_iq(&IQ1S_SEEN, "IQ1_S"); launch_mul_mat_vec_iq1_s_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, stream,
            )},
            GgmlDType::IQ2XXS => { note_iq(&IQ2XXS_SEEN, "IQ2_XXS"); launch_mul_mat_vec_iq2_xxs_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, stream,
            )},
            GgmlDType::IQ2S   => { note_iq(&IQ2S_SEEN, "IQ2_S"); launch_mul_mat_vec_iq2_s_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, stream,
            )},
            GgmlDType::IQ3S   => { note_iq(&IQ3S_SEEN, "IQ3_S"); launch_mul_mat_vec_iq3_s_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, stream,
            )},
            _ => panic!("unsupported dtype for mul_mat_vec_q8_1: {}", storage.dtype),
        }
    }
}

/// High-level quantized matrix-vector multiply.
///
/// `x`: f32 `[num_tokens, ncols]` (GPU)
/// Returns: f32 `[num_tokens, nrows]` (GPU, allocated from `alloc`)
///
/// For num_tokens==1: uses fused dequant+dot.
/// For num_tokens>1: quantizes activations to Q8_1, then int-dot path.
///
/// # Safety
/// Valid CUDA context, valid GPU pointers.
pub unsafe fn ggml_matmul(
    storage: &GgmlStorage,
    x: GpuTensor,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_tokens = x.dim(0);
    debug_assert_eq!(x.dim(1), storage.ncols);
    assert_eq!(
        x.dtype(),
        DType::F32,
        "GGML matmul requires f32 activations, got {:?}",
        x.dtype()
    );

    let out = alloc.alloc_tensor(&[num_tokens, storage.nrows], DType::F32);
    let dst = out.as_gpu_tensor().raw_ptr() as *mut f32;

    if num_tokens == 1 && !storage.dtype.is_iq_quant() {
        // Fast path: fused dequant+dot for standard quant types at BS=1.
        ggml_dequant_mul_mat_vec(storage, x.as_ptr::<f32>(), dst, stream);
    } else {
        // IQ types always use Q8_1 path (no fused dequant+dot kernel).
        // Standard types use Q8_1 path for BS>1.
        let ncols_padded = pad(storage.ncols, MATRIX_ROW_PADDING);
        let (q8_ptr, _q8_bytes) =
            ggml_quantize_q8_1_alloc(x.as_ptr::<f32>(), storage.ncols, num_tokens, alloc, stream);
        ggml_mul_mat_vec_q8_1(storage, q8_ptr, ncols_padded, num_tokens, dst, stream);
    }

    out
}

/// Indexed MoE forward: quantized expert weights × Q8_1 inputs → f32 outputs.
///
/// - `storage`: 3D quantized expert weights `[num_experts, n, k]` flattened into GgmlStorage
///   where nrows = num_experts * n, ncols = k.
/// - `q8_input`: Q8_1-quantized input, layout depends on `input_dim1`:
///   - `input_dim1 == 1`: `[batch, k_padded]` (shared across topk per batch item)
///   - `input_dim1 != 1`: `[batch * topk, k_padded]` (unique per task)
/// - `indices`: `[batch * topk]` u32 expert indices.
/// - `output`: `[batch * topk, n]` f32 output buffer.
/// - `n`: output features per expert (nrows per expert).
/// - `k`: input features per expert (ncols).
/// - `batch`: batch size.
/// - `topk`: number of experts per token.
/// - `k_padded`: padded input dimension (for Q8_1 alignment).
/// - `input_dim1`: controls input sharing. 1 = all topk experts for a batch item share
///   the same input row. Otherwise each task_id indexes a unique input row.
///
/// # Safety
/// All pointers must be valid GPU memory. `storage.dtype` must be a supported MoE quant type.
pub unsafe fn ggml_moe_forward(
    storage: &GgmlStorage,
    q8_input: *const u8,
    indices: *const u32,
    output: *mut f32,
    n: usize,
    k: usize,
    batch: usize,
    topk: usize,
    k_padded: usize,
    input_dim1: usize,
    stream: CUstream,
) {
    let n_i = n as i32;
    let k_i = k as i32;
    let batch_i = batch as i32;
    let topk_i = topk as i32;
    let k_padded_i = k_padded as i32;
    let input_dim1_i = input_dim1 as i32;
    let vx = storage.ptr as *const u8;

    match storage.dtype {
        GgmlDType::Q2K => launch_indexed_moe_forward_q2k_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        GgmlDType::Q3K => launch_indexed_moe_forward_q3k_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        GgmlDType::Q4K => launch_indexed_moe_forward_q4k_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        GgmlDType::Q5K => launch_indexed_moe_forward_q5k_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        GgmlDType::Q6K => launch_indexed_moe_forward_q6k_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        GgmlDType::Q8_0 => launch_indexed_moe_forward_q8_0_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        GgmlDType::Q4_0 => launch_indexed_moe_forward_q4_0_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        GgmlDType::Q4_1 => launch_indexed_moe_forward_q4_1_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        GgmlDType::Q5_0 => launch_indexed_moe_forward_q5_0_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        GgmlDType::Q5_1 => launch_indexed_moe_forward_q5_1_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        _ => panic!("unsupported dtype for ggml_moe_forward: {}", storage.dtype),
    }
}

// ---------------------------------------------------------------------------
// GGUF weight loading — raw quantized bytes from GGUF → GPU
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// GGUF tensor descriptor (parsed from header, before loading to GPU).
pub struct GgufTensorInfo {
    pub hf_name: String,
    pub ggml_dtype: GgmlDType,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub size_bytes: usize,
}

/// Loaded GGUF weights on GPU — either quantized (`GgmlStorage`) or dequantized (`GpuTensor`).
pub enum GgufWeight {
    /// Quantized weight (linear layers).
    Quantized(GgmlStorage),
    /// Dequantized weight (norms, embeddings).
    Dense(GpuTensor),
}

/// GGUF weight store: loads all tensors from a GGUF file onto GPU.
///
/// Linear weights stay quantized (raw bytes → `GgmlStorage`).
/// Norms are dequantized to f32. Embeddings are dequantized to the model dtype.
pub struct GgufGpuWeights {
    weights: HashMap<String, GgufWeight>,
}

impl GgufGpuWeights {
    /// Load all tensors from a GGUF file onto GPU.
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    pub unsafe fn load(
        path: &Path,
        model_dtype: DType,
        alloc: &mut CachingAllocator,
        stream: CUstream,
    ) -> anyhow::Result<Self> {
        use vllm_model::gguf_format::Content;

        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(file);
        let content =
            Content::read(&mut reader).map_err(|e| anyhow::anyhow!("GGUF parse error: {e}"))?;

        let tensor_data_offset = content.tensor_data_offset;
        let mut weights = HashMap::new();

        // Allocate pinned host buffer for H2D transfers (reuse for all tensors).
        let max_tensor_bytes = content
            .tensor_infos
            .values()
            .map(|info| {
                let elems = info.shape.elem_count();
                let bs = info.ggml_dtype.block_size();
                (elems / bs) * info.ggml_dtype.type_size()
            })
            .max()
            .unwrap_or(0);
        let host_buf = ferrite_cuda_core::driver::mem_alloc_host(max_tensor_bytes)?;

        for (gguf_name, info) in &content.tensor_infos {
            let hf_name = vllm_model::gguf::gguf_to_hf_name(gguf_name);
            let dims = info.shape.dims();
            let elem_count = info.shape.elem_count();
            let gguf_dtype = info.ggml_dtype;

            // Map GGUF dtype tag to our quantized GgmlDType (None for float types).
            let our_dtype = GgmlDType::from_gguf(gguf_dtype);

            // Determine if this is a norm or embedding (dequantize) vs linear (keep quantized).
            // Layer norms (RmsNorm) → f32; QK norms → model dtype; embeddings → model dtype.
            let is_layer_norm = hf_name.contains("layernorm")
                || hf_name.ends_with("model.norm.weight")
                || (hf_name.contains("norm.weight")
                    && !hf_name.contains("q_norm")
                    && !hf_name.contains("k_norm"));
            let is_qk_norm = hf_name.contains("q_norm.weight") || hf_name.contains("k_norm.weight");
            let is_norm = is_layer_norm || is_qk_norm;
            let is_embedding = hf_name == "model.embed_tokens.weight";
            let is_lm_head = hf_name == "lm_head.weight";
            let is_f32_or_f16 = gguf_dtype.is_float();

            // Read raw bytes from disk.
            let bs = gguf_dtype.block_size();
            let ts = gguf_dtype.type_size();
            let size_bytes = (elem_count / bs) * ts;
            reader.seek(SeekFrom::Start(tensor_data_offset + info.offset))?;
            let host_slice = std::slice::from_raw_parts_mut(host_buf, size_bytes);
            reader.read_exact(host_slice)?;

            if is_f32_or_f16 || is_norm || is_embedding || is_lm_head {
                // Dequantize path: for unquantized types, just H2D copy.
                // For quantized norms/embeddings, upload then dequant on GPU.
                if is_f32_or_f16 {
                    // Direct H2D copy of f32/f16/bf16 data.
                    let dtype_size = ts; // 4 for f32, 2 for f16/bf16
                    let source_dtype = if dtype_size == 4 {
                        DType::F32
                    } else if gguf_dtype == vllm_model::gguf_format::GgufDType::BF16 {
                        DType::BF16
                    } else {
                        DType::F16
                    };
                    // QK norms must match model dtype; layer norms stay f32 (fused_add_rms_norm expects f32).
                    // Embeddings/lm_head stay in model dtype.
                    let target_dtype = if is_qk_norm {
                        model_dtype
                    } else if is_layer_norm {
                        DType::F32
                    } else {
                        source_dtype
                    };

                    if source_dtype == target_dtype
                        || (source_dtype == DType::F32 && target_dtype == DType::F32)
                    {
                        let gpu_ptr = ferrite_cuda_core::driver::mem_alloc(size_bytes)?;
                        ferrite_cuda_core::driver::memcpy_htod_async(
                            gpu_ptr, host_buf, size_bytes, stream,
                        )?;
                        let tensor = GpuTensor::new(gpu_ptr, dims, source_dtype);
                        weights.insert(hf_name, GgufWeight::Dense(tensor));
                    } else {
                        // Need dtype conversion: upload as source, dequant/convert on GPU.
                        // For f32 → bf16/f16: upload f32, then use dequantize (identity for f32 blocks).
                        // Simplest approach: upload f32 to GPU, then convert via a kernel.
                        // Since QK norms are tiny, do CPU conversion.
                        if source_dtype == DType::F32 {
                            let f32_slice =
                                std::slice::from_raw_parts(host_buf as *const f32, elem_count);
                            let out_size = elem_count * target_dtype.size_bytes();
                            let conv_buf = ferrite_cuda_core::driver::mem_alloc_host(out_size)?;
                            match target_dtype {
                                DType::BF16 => {
                                    let out = std::slice::from_raw_parts_mut(
                                        conv_buf as *mut u16,
                                        elem_count,
                                    );
                                    for (i, &v) in f32_slice.iter().enumerate() {
                                        out[i] = half::bf16::from_f32(v).to_bits();
                                    }
                                }
                                DType::F16 => {
                                    let out = std::slice::from_raw_parts_mut(
                                        conv_buf as *mut u16,
                                        elem_count,
                                    );
                                    for (i, &v) in f32_slice.iter().enumerate() {
                                        out[i] = half::f16::from_f32(v).to_bits();
                                    }
                                }
                                _ => panic!(
                                    "unsupported conversion: {source_dtype:?} -> {target_dtype:?}"
                                ),
                            }
                            let gpu_ptr = ferrite_cuda_core::driver::mem_alloc(out_size)?;
                            ferrite_cuda_core::driver::memcpy_htod_async(
                                gpu_ptr, conv_buf, out_size, stream,
                            )?;
                            ferrite_cuda_core::driver::stream_synchronize(stream)?;
                            ferrite_cuda_core::driver::mem_free_host(conv_buf)?;
                            let tensor = GpuTensor::new(gpu_ptr, dims, target_dtype);
                            weights.insert(hf_name, GgufWeight::Dense(tensor));
                        } else {
                            // Non-f32 source needing conversion — just store as-is for now.
                            let gpu_ptr = ferrite_cuda_core::driver::mem_alloc(size_bytes)?;
                            ferrite_cuda_core::driver::memcpy_htod_async(
                                gpu_ptr, host_buf, size_bytes, stream,
                            )?;
                            let tensor = GpuTensor::new(gpu_ptr, dims, source_dtype);
                            weights.insert(hf_name, GgufWeight::Dense(tensor));
                        }
                    }
                } else if let Some(our_dt) = our_dtype {
                    // Quantized norm/embedding: upload raw bytes, then dequant on GPU.
                    let gpu_raw = ferrite_cuda_core::driver::mem_alloc(size_bytes)?;
                    ferrite_cuda_core::driver::memcpy_htod_async(
                        gpu_raw, host_buf, size_bytes, stream,
                    )?;
                    ferrite_cuda_core::driver::stream_synchronize(stream)?;

                    let storage = GgmlStorage {
                        ptr: gpu_raw,
                        len: size_bytes,
                        dtype: our_dt,
                        nrows: if dims.len() >= 2 { dims[0] } else { 1 },
                        ncols: if dims.len() >= 2 { dims[1] } else { dims[0] },
                    };

                    let target = if is_layer_norm {
                        DType::F32
                    } else {
                        model_dtype
                    };
                    let tensor = ggml_dequantize_to_tensor(&storage, target, dims, alloc, stream);
                    ferrite_cuda_core::driver::stream_synchronize(stream)?;
                    // Free the raw quantized buffer since we dequantized.
                    ferrite_cuda_core::driver::mem_free(gpu_raw)?;
                    // Weight tensors are permanent — leak from allocator tracking.
                    let gpu_tensor = tensor.into_gpu_tensor();
                    weights.insert(hf_name, GgufWeight::Dense(gpu_tensor));
                } else {
                    anyhow::bail!(
                        "unsupported GGUF dtype {:?} for tensor {}",
                        gguf_dtype,
                        gguf_name
                    );
                }
            } else if let Some(our_dt) = our_dtype {
                // Quantized linear: raw H2D copy, keep compressed.
                let gpu_ptr = ferrite_cuda_core::driver::mem_alloc(size_bytes)?;
                ferrite_cuda_core::driver::memcpy_htod_async(
                    gpu_ptr, host_buf, size_bytes, stream,
                )?;

                // For 3D tensors (fused MoE experts), flatten first dims:
                // [num_experts, output_dim, input_dim] → nrows = num_experts * output_dim.
                let (nrows, ncols) = match dims.len() {
                    3 => (dims[0] * dims[1], dims[2]),
                    2 => (dims[0], dims[1]),
                    1 => (1, dims[0]),
                    _ => anyhow::bail!("unexpected shape {:?} for weight {}", dims, gguf_name),
                };

                let storage = GgmlStorage {
                    ptr: gpu_ptr,
                    len: size_bytes,
                    dtype: our_dt,
                    nrows,
                    ncols,
                };
                weights.insert(hf_name, GgufWeight::Quantized(storage));
            } else {
                anyhow::bail!(
                    "unsupported GGUF dtype {:?} for tensor {}",
                    gguf_dtype,
                    gguf_name
                );
            }
        }

        ferrite_cuda_core::driver::stream_synchronize(stream)?;
        ferrite_cuda_core::driver::mem_free_host(host_buf)?;

        tracing::info!(
            "GgufGpuWeights: loaded {} tensors from {}",
            weights.len(),
            path.display()
        );
        Ok(Self { weights })
    }

    /// Take a weight by HF name. Returns None if not found.
    pub fn take(&mut self, name: &str) -> Option<GgufWeight> {
        self.weights.remove(name)
    }

    /// Take a quantized weight, returning the GgmlStorage.
    pub fn take_quantized(&mut self, name: &str) -> anyhow::Result<GgmlStorage> {
        match self.take(name) {
            Some(GgufWeight::Quantized(s)) => Ok(s),
            Some(GgufWeight::Dense(_)) => {
                anyhow::bail!("expected quantized weight for {name}, got dense")
            }
            None => anyhow::bail!("weight not found: {name}"),
        }
    }

    /// Take a dense (dequantized) weight, returning the GpuTensor.
    pub fn take_dense(&mut self, name: &str) -> anyhow::Result<GpuTensor> {
        match self.take(name) {
            Some(GgufWeight::Dense(t)) => Ok(t),
            Some(GgufWeight::Quantized(_)) => {
                anyhow::bail!("expected dense weight for {name}, got quantized")
            }
            None => anyhow::bail!("weight not found: {name}"),
        }
    }

    /// Check if a weight exists.
    pub fn contains(&self, name: &str) -> bool {
        self.weights.contains_key(name)
    }

    /// Number of remaining weights.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_dtype_from_u32() {
        assert_eq!(GgmlDType::from_u32(2), Some(GgmlDType::Q4_0));
        assert_eq!(GgmlDType::from_u32(12), Some(GgmlDType::Q4K));
        assert_eq!(GgmlDType::from_u32(99), None);
    }

    #[test]
    fn test_ggml_dtype_type_size() {
        assert_eq!(GgmlDType::Q4_0.type_size(), 18);
        assert_eq!(GgmlDType::Q8_0.type_size(), 34);
        assert_eq!(GgmlDType::Q8_1.type_size(), 40);
    }

    #[test]
    fn test_ggml_dtype_block_size() {
        assert_eq!(GgmlDType::Q4_0.block_size(), 32);
        assert_eq!(GgmlDType::Q4K.block_size(), 256);
    }

    #[test]
    fn test_ggml_dtype_is_k_quant() {
        assert!(!GgmlDType::Q4_0.is_k_quant());
        assert!(GgmlDType::Q4K.is_k_quant());
        assert!(GgmlDType::Q6K.is_k_quant());
    }

    #[test]
    fn test_ggml_dtype_iq4_from_u32() {
        assert_eq!(GgmlDType::from_u32(20), Some(GgmlDType::IQ4NL));
        assert_eq!(GgmlDType::from_u32(23), Some(GgmlDType::IQ4XS));
    }

    #[test]
    fn test_ggml_dtype_iq4_type_size() {
        assert_eq!(GgmlDType::IQ4NL.type_size(), 18);
        assert_eq!(GgmlDType::IQ4XS.type_size(), 136);
    }

    #[test]
    fn test_ggml_dtype_iq4_block_size() {
        assert_eq!(GgmlDType::IQ4NL.block_size(), 32);
        assert_eq!(GgmlDType::IQ4XS.block_size(), 256);
    }

    #[test]
    fn test_ggml_dtype_is_iq_quant() {
        assert!(GgmlDType::IQ4NL.is_iq_quant());
        assert!(GgmlDType::IQ4XS.is_iq_quant());
        assert!(GgmlDType::IQ1M.is_iq_quant());
        assert!(!GgmlDType::Q4_0.is_iq_quant());
        assert!(!GgmlDType::Q4K.is_iq_quant());
    }

    #[test]
    fn test_ggml_dtype_iq4_display() {
        assert_eq!(format!("{}", GgmlDType::IQ4NL), "IQ4_NL");
        assert_eq!(format!("{}", GgmlDType::IQ4XS), "IQ4_XS");
    }

    #[test]
    fn test_ggml_dtype_iq1m_from_u32() {
        assert_eq!(GgmlDType::from_u32(29), Some(GgmlDType::IQ1M));
    }

    #[test]
    fn test_ggml_dtype_iq1m_type_size() {
        assert_eq!(GgmlDType::IQ1M.type_size(), 56);
    }

    #[test]
    fn test_ggml_dtype_iq1m_block_size() {
        assert_eq!(GgmlDType::IQ1M.block_size(), 256);
    }

    #[test]
    fn test_ggml_dtype_iq1m_display() {
        assert_eq!(format!("{}", GgmlDType::IQ1M), "IQ1_M");
    }

    #[test]
    fn test_ggml_storage_verify_size() {
        let s = GgmlStorage {
            ptr: 0x1000 as *mut u8,
            len: 18 * (4096 * 4096 / 32),
            dtype: GgmlDType::Q4_0,
            nrows: 4096,
            ncols: 4096,
        };
        assert!(s.verify_size());
    }

    #[test]
    fn test_ggml_storage_numel() {
        let s = GgmlStorage {
            ptr: std::ptr::null_mut(),
            len: 0,
            dtype: GgmlDType::Q4_0,
            nrows: 128,
            ncols: 256,
        };
        assert_eq!(s.numel(), 128 * 256);
    }

    #[test]
    fn test_padding_helpers() {
        assert_eq!(pad(100, 512), 512);
        assert_eq!(pad(512, 512), 512);
        assert_eq!(pad(513, 512), 1024);
    }
}
