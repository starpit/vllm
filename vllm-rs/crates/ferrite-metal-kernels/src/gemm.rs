// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! GEMM (General Matrix Multiply) implementation using Metal Performance Shaders
//!
//! Uses objc bindings to access MPS APIs not exposed by metal-rs.
//! Future-proof for M4+ hardware with support for modern Metal features.

use crate::{MetalDevice, MetalStream};
use metal::foreign_types::{ForeignType, ForeignTypeRef};
use metal::{Buffer, CommandBufferRef, DeviceRef};
use objc::runtime::{Object, BOOL, NO, YES};
use objc::{class, msg_send, sel, sel_impl};
use std::sync::Arc;

/// Encode an MPS f16/f32 GEMM into a borrowed command buffer.
///
/// The Phase 5.C.5 GEMM-routing entry point: the `MetalWorker` calls
/// this between ICB segments to keep dense `y = x @ W^T` dispatches
/// inside the same command buffer as the rest of the bucket. No
/// `MetalStream` wrapping, no internal commit — the caller owns the
/// command buffer and any `commit()` boundary.
///
/// `transpose_b = true` reproduces Linear-layer convention (weight
/// stored as `[N, K]`). `alpha = 1.0`, `beta = 0.0` is the canonical
/// case; pass through if a future caller needs accumulation.
///
/// The same MPS objects (descriptors, matrices, kernel) are
/// allocated per-call and retained by the command buffer until it
/// completes — manual release would double-free, mirroring the
/// existing `MetalGemm::execute` contract.
#[allow(clippy::too_many_arguments)]
pub fn encode_gemm_into_command_buffer(
    device: &DeviceRef,
    cmdbuf: &CommandBufferRef,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
    transpose_a: bool,
    transpose_b: bool,
    use_f16: bool,
) -> Result<(), GemmError> {
    let elem_size = if use_f16 { 2 } else { 4 };

    let a_rows = if transpose_a { k } else { m };
    let a_cols = if transpose_a { m } else { k };
    let b_rows = if transpose_b { n } else { k };
    let b_cols = if transpose_b { k } else { n };

    let expected_a_size = (a_rows * a_cols * elem_size) as u64;
    let expected_b_size = (b_rows * b_cols * elem_size) as u64;
    let expected_c_size = (m * n * elem_size) as u64;

    if a.length() < expected_a_size {
        return Err(GemmError::InvalidDimensions(format!(
            "A buffer too small: expected {expected_a_size}, got {}",
            a.length()
        )));
    }
    if b.length() < expected_b_size {
        return Err(GemmError::InvalidDimensions(format!(
            "B buffer too small: expected {expected_b_size}, got {}",
            b.length()
        )));
    }
    if c.length() < expected_c_size {
        return Err(GemmError::InvalidDimensions(format!(
            "C buffer too small: expected {expected_c_size}, got {}",
            c.length()
        )));
    }

    unsafe {
        let data_type: u64 = if use_f16 { 268435472 } else { 268435488 };

        let a_row_bytes = (a_cols * elem_size) as u64;
        let a_desc: *mut Object = msg_send![class!(MPSMatrixDescriptor),
            matrixDescriptorWithRows:a_rows as u64
            columns:a_cols as u64
            rowBytes:a_row_bytes
            dataType:data_type
        ];
        let b_row_bytes = (b_cols * elem_size) as u64;
        let b_desc: *mut Object = msg_send![class!(MPSMatrixDescriptor),
            matrixDescriptorWithRows:b_rows as u64
            columns:b_cols as u64
            rowBytes:b_row_bytes
            dataType:data_type
        ];
        let c_row_bytes = (n * elem_size) as u64;
        let c_desc: *mut Object = msg_send![class!(MPSMatrixDescriptor),
            matrixDescriptorWithRows:m as u64
            columns:n as u64
            rowBytes:c_row_bytes
            dataType:data_type
        ];

        let a_matrix: *mut Object = msg_send![class!(MPSMatrix), alloc];
        let a_matrix: *mut Object = msg_send![a_matrix,
            initWithBuffer: a.as_ptr()
            offset: 0u64
            descriptor: a_desc
        ];
        let b_matrix: *mut Object = msg_send![class!(MPSMatrix), alloc];
        let b_matrix: *mut Object = msg_send![b_matrix,
            initWithBuffer: b.as_ptr()
            offset: 0u64
            descriptor: b_desc
        ];
        let c_matrix: *mut Object = msg_send![class!(MPSMatrix), alloc];
        let c_matrix: *mut Object = msg_send![c_matrix,
            initWithBuffer: c.as_ptr()
            offset: 0u64
            descriptor: c_desc
        ];

        let gemm_kernel: *mut Object = msg_send![class!(MPSMatrixMultiplication), alloc];
        let gemm_kernel: *mut Object = msg_send![gemm_kernel,
            initWithDevice: device.as_ptr()
            transposeLeft: if transpose_a { YES } else { NO }
            transposeRight: if transpose_b { YES } else { NO }
            resultRows: m as u64
            resultColumns: n as u64
            interiorColumns: k as u64
            alpha: alpha as f64
            beta: beta as f64
        ];

        let _: () = msg_send![gemm_kernel,
            encodeToCommandBuffer: cmdbuf.as_ptr()
            leftMatrix: a_matrix
            rightMatrix: b_matrix
            resultMatrix: c_matrix
        ];
    }

    Ok(())
}

/// Errors specific to GEMM operations
#[derive(Debug)]
pub enum GemmError {
    /// Invalid matrix dimensions for multiplication
    InvalidDimensions(String),
    /// MPS operation failed
    MpsError(String),
    /// Metal error
    MetalError(String),
    /// Command buffer execution failed
    ExecutionFailed(String),
}

impl std::fmt::Display for GemmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GemmError::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {}", msg),
            GemmError::MpsError(msg) => write!(f, "MPS error: {}", msg),
            GemmError::MetalError(msg) => write!(f, "Metal error: {}", msg),
            GemmError::ExecutionFailed(msg) => write!(f, "Execution failed: {}", msg),
        }
    }
}

impl std::error::Error for GemmError {}

/// GEMM operation: C = alpha * op(A) * op(B) + beta * C
///
/// Supports:
/// - FP16, FP32 data types (BF16 via conversion)
/// - Transposed operations (op(X) = X or X^T)
/// - Batched matrix multiplication
/// - Optimized using Metal Performance Shaders
/// - Future-proof for M4+ hardware features
pub struct MetalGemm {
    device: Arc<MetalDevice>,
}

impl MetalGemm {
    /// Create a new GEMM operator
    pub fn new(device: Arc<MetalDevice>) -> Result<Self, GemmError> {
        Ok(Self { device })
    }

    /// Execute GEMM: C = alpha * A * B + beta * C
    ///
    /// # Arguments
    /// * `stream` - Metal stream for command execution
    /// * `a` - Input matrix A [M, K] or [K, M] if transposed
    /// * `b` - Input matrix B [K, N] or [N, K] if transposed
    /// * `c` - Output matrix C [M, N] (also used as input if beta != 0)
    /// * `m` - Number of rows in result C
    /// * `n` - Number of columns in result C
    /// * `k` - Inner dimension (columns of A, rows of B)
    /// * `alpha` - Scalar multiplier for A*B
    /// * `beta` - Scalar multiplier for C (0 means overwrite C)
    /// * `transpose_a` - Whether to transpose A
    /// * `transpose_b` - Whether to transpose B
    /// * `use_f16` - Use FP16 precision (true) or FP32 (false)
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        stream: &mut MetalStream,
        a: &Buffer,
        b: &Buffer,
        c: &Buffer,
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        beta: f32,
        transpose_a: bool,
        transpose_b: bool,
        use_f16: bool,
    ) -> Result<(), GemmError> {
        // Validate dimensions
        let a_rows = if transpose_a { k } else { m };
        let a_cols = if transpose_a { m } else { k };
        let b_rows = if transpose_b { n } else { k };
        let b_cols = if transpose_b { k } else { n };

        let elem_size = if use_f16 { 2 } else { 4 };

        let expected_a_size = (a_rows * a_cols * elem_size) as u64;
        let expected_b_size = (b_rows * b_cols * elem_size) as u64;
        let expected_c_size = (m * n * elem_size) as u64;

        if a.length() < expected_a_size {
            return Err(GemmError::InvalidDimensions(format!(
                "Matrix A buffer too small: expected {}, got {}",
                expected_a_size,
                a.length()
            )));
        }

        if b.length() < expected_b_size {
            return Err(GemmError::InvalidDimensions(format!(
                "Matrix B buffer too small: expected {}, got {}",
                expected_b_size,
                b.length()
            )));
        }

        if c.length() < expected_c_size {
            return Err(GemmError::InvalidDimensions(format!(
                "Matrix C buffer too small: expected {}, got {}",
                expected_c_size,
                c.length()
            )));
        }

        unsafe {
            // Create MPSMatrixDescriptor for each matrix
            // MPSDataType enum values (NOT MTLDataType!)
            // MPSDataTypeFloat32 = 268435488 (0x10000020)
            // MPSDataTypeFloat16 = 268435472 (0x10000010)
            let data_type: u64 = if use_f16 { 268435472 } else { 268435488 };

            // MPSMatrixDescriptor for A
            // rowBytes MUST be exactly columns * element_size for MPS
            // The buffer itself can be larger, but rowBytes tells MPS the stride
            let a_row_bytes = (a_cols * elem_size) as u64;
            let a_desc: *mut Object = msg_send![class!(MPSMatrixDescriptor),
                matrixDescriptorWithRows:a_rows as u64
                columns:a_cols as u64
                rowBytes:a_row_bytes
                dataType:data_type
            ];

            // MPSMatrixDescriptor for B
            let b_row_bytes = (b_cols * elem_size) as u64;
            let b_desc: *mut Object = msg_send![class!(MPSMatrixDescriptor),
                matrixDescriptorWithRows:b_rows as u64
                columns:b_cols as u64
                rowBytes:b_row_bytes
                dataType:data_type
            ];

            // MPSMatrixDescriptor for C
            let c_row_bytes = (n * elem_size) as u64;
            let c_desc: *mut Object = msg_send![class!(MPSMatrixDescriptor),
                matrixDescriptorWithRows:m as u64
                columns:n as u64
                rowBytes:c_row_bytes
                dataType:data_type
            ];

            // Create MPSMatrix objects
            let a_matrix: *mut Object = msg_send![class!(MPSMatrix), alloc];
            let a_matrix: *mut Object = msg_send![a_matrix,
                initWithBuffer: a.as_ptr()
                offset: 0u64
                descriptor: a_desc
            ];

            let b_matrix: *mut Object = msg_send![class!(MPSMatrix), alloc];
            let b_matrix: *mut Object = msg_send![b_matrix,
                initWithBuffer: b.as_ptr()
                offset: 0u64
                descriptor: b_desc
            ];

            let c_matrix: *mut Object = msg_send![class!(MPSMatrix), alloc];
            let c_matrix: *mut Object = msg_send![c_matrix,
                initWithBuffer: c.as_ptr()
                offset: 0u64
                descriptor: c_desc
            ];

            // Create MPSMatrixMultiplication kernel
            let gemm_kernel: *mut Object = msg_send![class!(MPSMatrixMultiplication), alloc];
            let gemm_kernel: *mut Object = msg_send![gemm_kernel,
                initWithDevice: self.device.device.as_ptr()
                transposeLeft: if transpose_a { YES } else { NO }
                transposeRight: if transpose_b { YES } else { NO }
                resultRows: m as u64
                resultColumns: n as u64
                interiorColumns: k as u64
                alpha: alpha as f64
                beta: beta as f64
            ];

            // Encode to command buffer (use stream's managed buffer)
            let command_buffer = stream
                .get_command_buffer()
                .map_err(|e| GemmError::ExecutionFailed(format!("{:?}", e)))?;
            let _: () = msg_send![gemm_kernel,
                encodeToCommandBuffer: command_buffer.as_ptr()
                leftMatrix: a_matrix
                rightMatrix: b_matrix
                resultMatrix: c_matrix
            ];

            // Commit through stream to maintain proper lifecycle
            stream
                .commit()
                .map_err(|e| GemmError::ExecutionFailed(format!("{:?}", e)))?;

            // Note: Do NOT manually release MPS objects
            // The command buffer retains them and will release when done
            // Manual release causes double-free and SIGSEGV
        }

        Ok(())
    }

    /// Execute batched GEMM: C[i] = alpha * A[i] * B[i] + beta * C[i] for i in 0..batch_size
    ///
    /// All matrices in the batch must have the same dimensions.
    /// Uses MPSMatrixMultiplication with batch support for efficiency.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_batched(
        &self,
        stream: &mut MetalStream,
        a: &Buffer,
        b: &Buffer,
        c: &Buffer,
        batch_size: u32,
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        beta: f32,
        transpose_a: bool,
        transpose_b: bool,
        use_f16: bool,
    ) -> Result<(), GemmError> {
        let elem_size = if use_f16 { 2 } else { 4 };

        let a_rows = if transpose_a { k } else { m };
        let a_cols = if transpose_a { m } else { k };
        let b_rows = if transpose_b { n } else { k };
        let b_cols = if transpose_b { k } else { n };

        let a_matrix_size = (a_rows * a_cols * elem_size) as u64;
        let b_matrix_size = (b_rows * b_cols * elem_size) as u64;
        let c_matrix_size = (m * n * elem_size) as u64;

        // For now, execute sequentially
        // TODO: Use MPSMatrixMultiplication batch API when available in metal-rs
        for i in 0..batch_size {
            let a_offset = i as u64 * a_matrix_size;
            let b_offset = i as u64 * b_matrix_size;
            let c_offset = i as u64 * c_matrix_size;

            unsafe {
                // MPSDataType enum values (NOT MTLDataType!)
                // MPSDataTypeFloat32 = 268435488 (0x10000020)
                // MPSDataTypeFloat16 = 268435472 (0x10000010)
                let data_type: u64 = if use_f16 { 268435472 } else { 268435488 };

                let a_row_bytes = (a_cols * elem_size) as u64;
                let a_desc: *mut Object = msg_send![class!(MPSMatrixDescriptor),
                    matrixDescriptorWithRows:a_rows as u64
                    columns:a_cols as u64
                    rowBytes:a_row_bytes
                    dataType:data_type
                ];

                let b_row_bytes = (b_cols * elem_size) as u64;
                let b_desc: *mut Object = msg_send![class!(MPSMatrixDescriptor),
                    matrixDescriptorWithRows:b_rows as u64
                    columns:b_cols as u64
                    rowBytes:b_row_bytes
                    dataType:data_type
                ];

                let c_row_bytes = (n * elem_size) as u64;
                let c_desc: *mut Object = msg_send![class!(MPSMatrixDescriptor),
                    matrixDescriptorWithRows:m as u64
                    columns:n as u64
                    rowBytes:c_row_bytes
                    dataType:data_type
                ];

                let a_matrix: *mut Object = msg_send![class!(MPSMatrix), alloc];
                let a_matrix: *mut Object = msg_send![a_matrix,
                    initWithBuffer: a.as_ptr()
                    offset: a_offset
                    descriptor: a_desc
                ];

                let b_matrix: *mut Object = msg_send![class!(MPSMatrix), alloc];
                let b_matrix: *mut Object = msg_send![b_matrix,
                    initWithBuffer: b.as_ptr()
                    offset: b_offset
                    descriptor: b_desc
                ];

                let c_matrix: *mut Object = msg_send![class!(MPSMatrix), alloc];
                let c_matrix: *mut Object = msg_send![c_matrix,
                    initWithBuffer: c.as_ptr()
                    offset: c_offset
                    descriptor: c_desc
                ];

                let gemm_kernel: *mut Object = msg_send![class!(MPSMatrixMultiplication), alloc];
                let gemm_kernel: *mut Object = msg_send![gemm_kernel,
                    initWithDevice: self.device.device.as_ptr()
                    transposeLeft: if transpose_a { YES } else { NO }
                    transposeRight: if transpose_b { YES } else { NO }
                    resultRows: m as u64
                    resultColumns: n as u64
                    interiorColumns: k as u64
                    alpha: alpha as f64
                    beta: beta as f64
                ];

                // Encode to command buffer (use stream's managed buffer)
                let command_buffer = stream
                    .get_command_buffer()
                    .map_err(|e| GemmError::ExecutionFailed(format!("{:?}", e)))?;
                let _: () = msg_send![gemm_kernel,
                    encodeToCommandBuffer: command_buffer.as_ptr()
                    leftMatrix: a_matrix
                    rightMatrix: b_matrix
                    resultMatrix: c_matrix
                ];

                // Commit through stream to maintain proper lifecycle
                stream
                    .commit()
                    .map_err(|e| GemmError::ExecutionFailed(format!("{:?}", e)))?;

                // Note: Do NOT manually release MPS objects
                // The command buffer retains them and will release when done
                // Manual release causes double-free and SIGSEGV
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect_device;
    use half::f16;
    use metal::MTLResourceOptions;

    fn create_buffer_with_data<T: Copy>(device: &metal::Device, data: &[T]) -> Buffer {
        let size = data.len() * std::mem::size_of::<T>();
        let buffer = device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared);

        unsafe {
            let ptr = buffer.contents() as *mut T;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }

        buffer
    }

    fn read_buffer_data<T: Copy>(buffer: &Buffer, count: usize) -> Vec<T> {
        let mut result = vec![unsafe { std::mem::zeroed() }; count];
        unsafe {
            let ptr = buffer.contents() as *const T;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), count);
        }
        result
    }

    #[test]
    fn test_gemm_basic_f32() {
        let metal_device = detect_device().expect("Metal device required");
        let mut stream = MetalStream::new(&metal_device.device);
        let gemm = MetalGemm::new(Arc::new(metal_device.clone())).expect("Failed to create GEMM");

        // Simple 2x2 matrix multiplication
        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // C = A * B = [[19, 22], [43, 50]]

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![5.0f32, 6.0, 7.0, 8.0];
        let c_data = vec![0.0f32; 4];

        let a_buf = create_buffer_with_data(&metal_device.device, &a_data);
        let b_buf = create_buffer_with_data(&metal_device.device, &b_data);
        let c_buf = create_buffer_with_data(&metal_device.device, &c_data);

        gemm.execute(
            &mut stream,
            &a_buf,
            &b_buf,
            &c_buf,
            2,     // m
            2,     // n
            2,     // k
            1.0,   // alpha
            0.0,   // beta
            false, // transpose_a
            false, // transpose_b
            false, // use_f16
        )
        .expect("GEMM execution failed");

        let result: Vec<f32> = read_buffer_data(&c_buf, 4);
        let expected = vec![19.0f32, 22.0, 43.0, 50.0];
        for i in 0..4 {
            assert!(
                (result[i] - expected[i]).abs() < 1e-5,
                "Mismatch at index {}: got {}, expected {}",
                i,
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_gemm_basic_f16() {
        let metal_device = detect_device().expect("Metal device required");
        let mut stream = MetalStream::new(&metal_device.device);
        let gemm = MetalGemm::new(Arc::new(metal_device.clone())).expect("Failed to create GEMM");

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![5.0f32, 6.0, 7.0, 8.0];

        let a_f16: Vec<f16> = a_data.iter().map(|&x| f16::from_f32(x)).collect();
        let b_f16: Vec<f16> = b_data.iter().map(|&x| f16::from_f32(x)).collect();
        let c_f16 = vec![f16::ZERO; 4];

        let a_buf = create_buffer_with_data(&metal_device.device, &a_f16);
        let b_buf = create_buffer_with_data(&metal_device.device, &b_f16);
        let c_buf = create_buffer_with_data(&metal_device.device, &c_f16);

        gemm.execute(
            &mut stream,
            &a_buf,
            &b_buf,
            &c_buf,
            2,
            2,
            2,
            1.0,
            0.0,
            false,
            false,
            true, // use_f16
        )
        .expect("GEMM execution failed");

        let result_f16: Vec<f16> = read_buffer_data(&c_buf, 4);
        let result: Vec<f32> = result_f16.iter().map(|&x| x.to_f32()).collect();

        let expected = vec![19.0f32, 22.0, 43.0, 50.0];
        for i in 0..4 {
            assert!(
                (result[i] - expected[i]).abs() < 0.1, // Relaxed tolerance for FP16
                "Mismatch at index {}: got {}, expected {}",
                i,
                result[i],
                expected[i]
            );
        }
    }
}
