// SPDX-License-Identifier: Apache-2.0
//! `GpuTensor`: a lightweight, always-contiguous GPU tensor.
//!
//! 32 bytes, `Copy`, no `Drop`, no refcounting, no events.
//! Memory lifetime is managed by the owning region (weights, arena, KV pool),
//! not by the tensor itself.

use crate::dtype::DType;

/// Maximum number of dimensions (sufficient for LLM inference).
pub const MAX_DIMS: usize = 4;

/// A lightweight GPU tensor descriptor.
///
/// This is just a typed pointer + shape — it does NOT own the underlying memory.
/// Always represents a contiguous row-major GPU buffer.
///
/// # Safety
/// - `ptr` must point to valid GPU memory for the lifetime of use.
/// - The caller is responsible for ensuring the pointed-to memory outlives this tensor.
/// - All operations assume contiguous row-major layout.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct GpuTensor {
    ptr: *mut u8,
    shape: [u32; MAX_DIMS],
    ndim: u8,
    dtype: DType,
}

// GpuTensor is Send/Sync — the raw pointer is a GPU device pointer,
// and GPU memory is accessible from any host thread via CUDA driver API.
unsafe impl Send for GpuTensor {}
unsafe impl Sync for GpuTensor {}

impl GpuTensor {
    /// Create a new tensor from a raw device pointer and shape.
    ///
    /// # Safety
    /// `ptr` must point to `numel * dtype.size_bytes()` bytes of valid GPU memory.
    pub unsafe fn new(ptr: *mut u8, shape: &[usize], dtype: DType) -> Self {
        debug_assert!(shape.len() <= MAX_DIMS, "too many dims: {}", shape.len());
        let mut s = [1u32; MAX_DIMS];
        for (i, &d) in shape.iter().enumerate() {
            s[i] = d as u32;
        }
        Self {
            ptr,
            shape: s,
            ndim: shape.len() as u8,
            dtype,
        }
    }

    /// Create a null (invalid) tensor. Used as placeholder before initialization.
    pub const fn null(dtype: DType) -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            shape: [0; MAX_DIMS],
            ndim: 0,
            dtype,
        }
    }

    /// Whether this tensor has a null pointer.
    pub fn is_null(self) -> bool {
        self.ptr.is_null()
    }

    /// Raw device pointer as `*const T`.
    pub fn as_ptr<T>(self) -> *const T {
        self.ptr as *const T
    }

    /// Raw device pointer as `*mut T`.
    pub fn as_mut_ptr<T>(self) -> *mut T {
        self.ptr as *mut T
    }

    /// Raw device pointer as `*mut u8`.
    pub fn raw_ptr(self) -> *mut u8 {
        self.ptr
    }

    /// Data type.
    pub fn dtype(self) -> DType {
        self.dtype
    }

    /// Number of dimensions.
    pub fn ndim(self) -> usize {
        self.ndim as usize
    }

    /// Shape as a slice.
    pub fn shape(&self) -> &[u32] {
        &self.shape[..self.ndim as usize]
    }

    /// Size of dimension `d`.
    pub fn dim(self, d: usize) -> usize {
        debug_assert!(d < self.ndim as usize);
        self.shape[d] as usize
    }

    /// Total number of elements. Returns 0 for null/empty tensors.
    pub fn numel(self) -> usize {
        if self.ndim == 0 {
            return 0;
        }
        self.shape[..self.ndim as usize]
            .iter()
            .map(|&d| d as usize)
            .product()
    }

    /// Total size in bytes.
    pub fn size_bytes(self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }

    /// Stride of the leading (first) dimension in elements.
    /// For a [M, N] tensor this returns N; for [M, N, K] this returns N*K.
    pub fn leading_stride(self) -> usize {
        if self.ndim <= 1 {
            return 1;
        }
        self.shape[1..self.ndim as usize]
            .iter()
            .map(|&d| d as usize)
            .product()
    }

    /// Narrow on dimension 0: returns a view with offset pointer. O(1), zero copies.
    pub fn narrow_dim0(self, start: usize, len: usize) -> GpuTensor {
        debug_assert!(start + len <= self.shape[0] as usize);
        let row_bytes = self.leading_stride() * self.dtype.size_bytes();
        let mut shape = self.shape;
        shape[0] = len as u32;
        GpuTensor {
            ptr: unsafe { self.ptr.add(start * row_bytes) },
            shape,
            ..self
        }
    }

    /// Reshape: metadata only, preserves total element count.
    pub fn reshape(self, new_shape: &[usize]) -> GpuTensor {
        let mut s = [1u32; MAX_DIMS];
        for (i, &d) in new_shape.iter().enumerate() {
            s[i] = d as u32;
        }
        let new_numel: usize = new_shape.iter().product();
        debug_assert_eq!(
            self.numel(),
            new_numel,
            "reshape: {} elements -> {} elements",
            self.numel(),
            new_numel
        );
        GpuTensor {
            shape: s,
            ndim: new_shape.len() as u8,
            ..self
        }
    }

    /// View with a pointer offset (in bytes). Useful for zero-copy splits.
    ///
    /// # Safety
    /// Caller must ensure the resulting pointer + shape stays within valid memory.
    pub unsafe fn offset_bytes(self, byte_offset: usize, new_shape: &[usize]) -> GpuTensor {
        let mut s = [1u32; MAX_DIMS];
        for (i, &d) in new_shape.iter().enumerate() {
            s[i] = d as u32;
        }
        GpuTensor {
            ptr: unsafe { self.ptr.add(byte_offset) },
            shape: s,
            ndim: new_shape.len() as u8,
            dtype: self.dtype,
        }
    }
}

impl std::fmt::Debug for GpuTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GpuTensor({:?}, {:?}, ptr={:p})",
            &self.shape[..self.ndim as usize],
            self.dtype,
            self.ptr
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Construction and basic accessors
    // -----------------------------------------------------------------------

    #[test]
    fn test_1d_tensor() {
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[256], DType::F32) };
        assert_eq!(t.ndim(), 1);
        assert_eq!(t.dim(0), 256);
        assert_eq!(t.numel(), 256);
        assert_eq!(t.size_bytes(), 1024);
        assert_eq!(t.leading_stride(), 1);
        assert_eq!(t.shape(), &[256]);
    }

    #[test]
    fn test_2d_tensor() {
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[4, 128], DType::F16) };
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.dim(0), 4);
        assert_eq!(t.dim(1), 128);
        assert_eq!(t.numel(), 512);
        assert_eq!(t.size_bytes(), 1024);
        assert_eq!(t.leading_stride(), 128);
    }

    #[test]
    fn test_3d_tensor() {
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[2, 16, 64], DType::BF16) };
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.dim(0), 2);
        assert_eq!(t.dim(1), 16);
        assert_eq!(t.dim(2), 64);
        assert_eq!(t.numel(), 2048);
        assert_eq!(t.size_bytes(), 4096);
        assert_eq!(t.leading_stride(), 1024); // 16 * 64
    }

    #[test]
    fn test_4d_tensor() {
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[2, 32, 8, 128], DType::F32) };
        assert_eq!(t.ndim(), 4);
        assert_eq!(t.numel(), 2 * 32 * 8 * 128);
        assert_eq!(t.size_bytes(), 2 * 32 * 8 * 128 * 4);
        assert_eq!(t.leading_stride(), 32 * 8 * 128);
    }

    #[test]
    fn test_dtypes() {
        let base = 0x1000 as *mut u8;
        let shape = &[10, 20];

        let f16 = unsafe { GpuTensor::new(base, shape, DType::F16) };
        assert_eq!(f16.dtype(), DType::F16);
        assert_eq!(f16.size_bytes(), 200 * 2);

        let bf16 = unsafe { GpuTensor::new(base, shape, DType::BF16) };
        assert_eq!(bf16.dtype(), DType::BF16);
        assert_eq!(bf16.size_bytes(), 200 * 2);

        let f32 = unsafe { GpuTensor::new(base, shape, DType::F32) };
        assert_eq!(f32.dtype(), DType::F32);
        assert_eq!(f32.size_bytes(), 200 * 4);

        let u32t = unsafe { GpuTensor::new(base, shape, DType::U32) };
        assert_eq!(u32t.dtype(), DType::U32);
        assert_eq!(u32t.size_bytes(), 200 * 4);

        let i64t = unsafe { GpuTensor::new(base, shape, DType::I64) };
        assert_eq!(i64t.dtype(), DType::I64);
        assert_eq!(i64t.size_bytes(), 200 * 8);
    }

    // -----------------------------------------------------------------------
    // Null tensors
    // -----------------------------------------------------------------------

    #[test]
    fn test_null_tensor() {
        let t = GpuTensor::null(DType::F16);
        assert!(t.is_null());
        assert_eq!(t.ndim(), 0);
        assert_eq!(t.numel(), 0);
        assert_eq!(t.size_bytes(), 0);
        assert_eq!(t.shape(), &[] as &[u32]);
    }

    #[test]
    fn test_null_is_const() {
        const NULL: GpuTensor = GpuTensor::null(DType::F32);
        assert!(NULL.is_null());
    }

    #[test]
    fn test_non_null_is_not_null() {
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[4], DType::F16) };
        assert!(!t.is_null());
    }

    // -----------------------------------------------------------------------
    // Pointer access
    // -----------------------------------------------------------------------

    #[test]
    fn test_pointer_access() {
        let addr = 0xDEAD_BEEF as *mut u8;
        let t = unsafe { GpuTensor::new(addr, &[4, 8], DType::F16) };
        assert_eq!(t.raw_ptr() as usize, 0xDEAD_BEEF);
        assert_eq!(t.as_ptr::<u16>() as usize, 0xDEAD_BEEF);
        assert_eq!(t.as_mut_ptr::<u16>() as usize, 0xDEAD_BEEF);
    }

    // -----------------------------------------------------------------------
    // Copy semantics
    // -----------------------------------------------------------------------

    #[test]
    fn test_copy_semantics() {
        let t1 = unsafe { GpuTensor::new(0x1000 as *mut u8, &[4, 128], DType::F16) };
        let t2 = t1; // Copy
        let t3 = t1; // Copy again — t1 is still valid

        assert_eq!(t1.raw_ptr(), t2.raw_ptr());
        assert_eq!(t1.numel(), t3.numel());
        assert_eq!(t1.dtype(), t2.dtype());
    }

    #[test]
    fn test_clone() {
        let t1 = unsafe { GpuTensor::new(0x2000 as *mut u8, &[8, 64], DType::BF16) };
        #[allow(clippy::clone_on_copy)]
        let t2 = t1.clone();
        assert_eq!(t1.raw_ptr(), t2.raw_ptr());
        assert_eq!(t1.numel(), t2.numel());
    }

    // -----------------------------------------------------------------------
    // Narrow dim 0
    // -----------------------------------------------------------------------

    #[test]
    fn test_narrow_dim0_2d() {
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[8, 64], DType::F32) };
        let narrowed = t.narrow_dim0(2, 3);

        assert_eq!(narrowed.dim(0), 3);
        assert_eq!(narrowed.dim(1), 64);
        assert_eq!(narrowed.numel(), 192);
        // Offset: 2 rows * 64 elems * 4 bytes = 512 bytes
        assert_eq!(narrowed.raw_ptr() as usize, 0x1000 + 512);
        assert_eq!(narrowed.dtype(), DType::F32);
    }

    #[test]
    fn test_narrow_dim0_3d() {
        let t = unsafe { GpuTensor::new(0x2000 as *mut u8, &[4, 16, 128], DType::F16) };
        let narrowed = t.narrow_dim0(1, 2);

        assert_eq!(narrowed.dim(0), 2);
        assert_eq!(narrowed.dim(1), 16);
        assert_eq!(narrowed.dim(2), 128);
        assert_eq!(narrowed.numel(), 2 * 16 * 128);
        // Offset: 1 * (16*128) * 2 bytes = 4096 bytes
        assert_eq!(narrowed.raw_ptr() as usize, 0x2000 + 4096);
    }

    #[test]
    fn test_narrow_dim0_full_range() {
        let t = unsafe { GpuTensor::new(0x3000 as *mut u8, &[8, 32], DType::BF16) };
        let narrowed = t.narrow_dim0(0, 8);

        assert_eq!(narrowed.dim(0), 8);
        assert_eq!(narrowed.raw_ptr(), t.raw_ptr()); // Same pointer — full range
    }

    #[test]
    fn test_narrow_dim0_single_row() {
        let t = unsafe { GpuTensor::new(0x4000 as *mut u8, &[10, 64], DType::F32) };
        let narrowed = t.narrow_dim0(5, 1);

        assert_eq!(narrowed.dim(0), 1);
        assert_eq!(narrowed.numel(), 64);
        assert_eq!(narrowed.raw_ptr() as usize, 0x4000 + 5 * 64 * 4);
    }

    #[test]
    fn test_narrow_dim0_chained() {
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[100, 64], DType::F16) };
        // Narrow to rows 10..60, then narrow again to rows 5..15 of that view
        let v1 = t.narrow_dim0(10, 50);
        let v2 = v1.narrow_dim0(5, 10);

        assert_eq!(v2.dim(0), 10);
        // v2 starts at original row 15: 15 * 64 * 2 = 1920
        assert_eq!(v2.raw_ptr() as usize, 0x1000 + 1920);
    }

    // -----------------------------------------------------------------------
    // Reshape
    // -----------------------------------------------------------------------

    #[test]
    fn test_reshape_2d_to_3d() {
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[4, 128], DType::BF16) };
        let r = t.reshape(&[2, 2, 128]);
        assert_eq!(r.ndim(), 3);
        assert_eq!(r.numel(), 512);
        assert_eq!(r.raw_ptr(), t.raw_ptr());
    }

    #[test]
    fn test_reshape_3d_to_2d() {
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[2, 16, 128], DType::F32) };
        let r = t.reshape(&[2, 2048]);
        assert_eq!(r.ndim(), 2);
        assert_eq!(r.dim(0), 2);
        assert_eq!(r.dim(1), 2048);
    }

    #[test]
    fn test_reshape_to_1d() {
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[4, 8], DType::F16) };
        let r = t.reshape(&[32]);
        assert_eq!(r.ndim(), 1);
        assert_eq!(r.dim(0), 32);
    }

    #[test]
    fn test_reshape_preserves_dtype() {
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[4, 128], DType::I64) };
        let r = t.reshape(&[512]);
        assert_eq!(r.dtype(), DType::I64);
    }

    #[test]
    fn test_reshape_preserves_pointer() {
        let t = unsafe { GpuTensor::new(0xABCD as *mut u8, &[6, 10], DType::F32) };
        let r = t.reshape(&[2, 3, 10]);
        assert_eq!(r.raw_ptr() as usize, 0xABCD);
    }

    // -----------------------------------------------------------------------
    // Offset bytes
    // -----------------------------------------------------------------------

    #[test]
    fn test_offset_bytes_basic() {
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[8, 256], DType::F16) };
        // Offset by 1024 bytes, new shape [4, 128]
        let v = unsafe { t.offset_bytes(1024, &[4, 128]) };
        assert_eq!(v.raw_ptr() as usize, 0x1000 + 1024);
        assert_eq!(v.dim(0), 4);
        assert_eq!(v.dim(1), 128);
        assert_eq!(v.dtype(), DType::F16);
    }

    #[test]
    fn test_offset_bytes_zero() {
        let t = unsafe { GpuTensor::new(0x2000 as *mut u8, &[4, 64], DType::F32) };
        let v = unsafe { t.offset_bytes(0, &[4, 64]) };
        assert_eq!(v.raw_ptr(), t.raw_ptr());
    }

    // -----------------------------------------------------------------------
    // Leading stride
    // -----------------------------------------------------------------------

    #[test]
    fn test_leading_stride_1d() {
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[256], DType::F16) };
        assert_eq!(t.leading_stride(), 1);
    }

    #[test]
    fn test_leading_stride_2d() {
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[4, 128], DType::F16) };
        assert_eq!(t.leading_stride(), 128);
    }

    #[test]
    fn test_leading_stride_3d() {
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[2, 8, 64], DType::F16) };
        assert_eq!(t.leading_stride(), 512); // 8 * 64
    }

    #[test]
    fn test_leading_stride_4d() {
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[2, 4, 8, 16], DType::F32) };
        assert_eq!(t.leading_stride(), 512); // 4 * 8 * 16
    }

    // -----------------------------------------------------------------------
    // Debug formatting
    // -----------------------------------------------------------------------

    #[test]
    fn test_debug_format() {
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[4, 128], DType::F16) };
        let s = format!("{:?}", t);
        assert!(s.contains("[4, 128]"));
        assert!(s.contains("F16"));
        assert!(s.contains("0x1000"));
    }

    #[test]
    fn test_debug_format_null() {
        let t = GpuTensor::null(DType::F32);
        let s = format!("{:?}", t);
        assert!(s.contains("[]"));
        assert!(s.contains("F32"));
        assert!(s.contains("0x0"));
    }

    // -----------------------------------------------------------------------
    // Size calculations for typical LLM shapes
    // -----------------------------------------------------------------------

    #[test]
    fn test_llm_hidden_states() {
        // Typical: [batch_size=32, hidden_dim=4096] in BF16
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[32, 4096], DType::BF16) };
        assert_eq!(t.numel(), 131072);
        assert_eq!(t.size_bytes(), 262144); // 256 KB
    }

    #[test]
    fn test_llm_attention_qkv() {
        // QKV: [tokens=32, q_size + 2*kv_size = 4096 + 2*512 = 5120] in BF16
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[32, 5120], DType::BF16) };
        assert_eq!(t.numel(), 163840);
        assert_eq!(t.size_bytes(), 327680);
    }

    #[test]
    fn test_llm_kv_cache_block() {
        // KV cache: [num_blocks=128, block_size=16, num_kv_heads=8, head_dim=128] in F16
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[128, 16, 8, 128], DType::F16) };
        assert_eq!(t.numel(), 128 * 16 * 8 * 128);
        assert_eq!(t.leading_stride(), 16 * 8 * 128);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_element_tensor() {
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[1], DType::F32) };
        assert_eq!(t.numel(), 1);
        assert_eq!(t.size_bytes(), 4);
        assert_eq!(t.leading_stride(), 1);
    }

    #[test]
    fn test_tensor_with_dim_1() {
        // [1, 1, 1, 1] — degenerate but valid
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[1, 1, 1, 1], DType::F16) };
        assert_eq!(t.numel(), 1);
        assert_eq!(t.size_bytes(), 2);
    }

    #[test]
    fn test_large_dim_value() {
        // u32 max is ~4B, but typical LLM dims fit easily
        let t = unsafe { GpuTensor::new(0x1000 as *mut u8, &[1, 100000], DType::F16) };
        assert_eq!(t.numel(), 100000);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GpuTensor>();
    }

    #[test]
    fn test_size_of_gpu_tensor() {
        // Verify the struct is reasonably compact.
        let size = std::mem::size_of::<GpuTensor>();
        // ptr(8) + shape(16) + ndim(1) + dtype(1) + padding = should be <= 32
        assert!(size <= 32, "GpuTensor is {} bytes, expected <= 32", size);
    }
}
