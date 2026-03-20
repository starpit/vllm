use cudarc::driver::sys::CUdeviceptr;

/// A thin wrapper around a CUDA device pointer with shape metadata.
///
/// This is intentionally simple for the POC. A full implementation would
/// track dtype, strides, and ownership.
#[derive(Clone, Copy, Debug)]
pub struct DevicePtr {
    /// Raw CUDA device pointer (u64).
    pub ptr: CUdeviceptr,
    /// Number of rows.
    pub rows: u32,
    /// Number of columns.
    pub cols: u32,
}

impl DevicePtr {
    /// Create a new DevicePtr from a raw pointer and shape.
    pub fn new(ptr: CUdeviceptr, rows: u32, cols: u32) -> Self {
        Self { ptr, rows, cols }
    }

    /// Total number of elements.
    pub fn numel(&self) -> u64 {
        self.rows as u64 * self.cols as u64
    }
}
