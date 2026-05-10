// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! GEMM (General Matrix Multiply) implementation using Metal Performance Shaders.
//!
//! Uses objc2 msg_send! for the MPS APIs not exposed by objc2-metal.

use crate::{MetalDevice, MetalStream};
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Bool, ProtocolObject};
use objc2::{class, msg_send};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLDevice};
use std::sync::Arc;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type CommandBufferRef = ProtocolObject<dyn MTLCommandBuffer>;
pub type DeviceRef = ProtocolObject<dyn MTLDevice>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmDtype {
    F16,
    F32,
    Bf16,
}

impl GemmDtype {
    pub fn mps_data_type(self) -> u64 {
        match self {
            Self::F16 => 0x1000_0010,
            Self::F32 => 0x1000_0020,
            Self::Bf16 => 0x9000_0010,
        }
    }

    pub fn elem_size(self) -> u32 {
        match self {
            Self::F16 | Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn encode_gemm_into_command_buffer(
    device: &DeviceRef,
    cmdbuf: &CommandBufferRef,
    a: &Buffer,
    a_offset: u64,
    b: &Buffer,
    b_offset: u64,
    c: &Buffer,
    c_offset: u64,
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
    transpose_a: bool,
    transpose_b: bool,
    dtype: GemmDtype,
) -> Result<(), GemmError> {
    let elem_size = dtype.elem_size();

    let a_rows = if transpose_a { k } else { m };
    let a_cols = if transpose_a { m } else { k };
    let b_rows = if transpose_b { n } else { k };
    let b_cols = if transpose_b { k } else { n };

    let expected_a_size = (a_rows * a_cols * elem_size) as usize;
    let expected_b_size = (b_rows * b_cols * elem_size) as usize;
    let expected_c_size = (m * n * elem_size) as usize;

    if a.length() < a_offset as usize + expected_a_size {
        return Err(GemmError::InvalidDimensions(format!(
            "A buffer too small: offset {a_offset} + expected {expected_a_size} > buf len {}",
            a.length()
        )));
    }
    if b.length() < b_offset as usize + expected_b_size {
        return Err(GemmError::InvalidDimensions(format!(
            "B buffer too small: offset {b_offset} + expected {expected_b_size} > buf len {}",
            b.length()
        )));
    }
    if c.length() < c_offset as usize + expected_c_size {
        return Err(GemmError::InvalidDimensions(format!(
            "C buffer too small: offset {c_offset} + expected {expected_c_size} > buf len {}",
            c.length()
        )));
    }

    unsafe {
        let data_type: u64 = dtype.mps_data_type();

        let a_row_bytes = (a_cols * elem_size) as u64;
        let a_desc: *mut AnyObject = msg_send![class!(MPSMatrixDescriptor),
            matrixDescriptorWithRows: a_rows as u64,
            columns: a_cols as u64,
            rowBytes: a_row_bytes,
            dataType: data_type
        ];
        let b_row_bytes = (b_cols * elem_size) as u64;
        let b_desc: *mut AnyObject = msg_send![class!(MPSMatrixDescriptor),
            matrixDescriptorWithRows: b_rows as u64,
            columns: b_cols as u64,
            rowBytes: b_row_bytes,
            dataType: data_type
        ];
        let c_row_bytes = (n * elem_size) as u64;
        let c_desc: *mut AnyObject = msg_send![class!(MPSMatrixDescriptor),
            matrixDescriptorWithRows: m as u64,
            columns: n as u64,
            rowBytes: c_row_bytes,
            dataType: data_type
        ];

        let a_ptr: *mut AnyObject = a as *const _ as *const AnyObject as *mut AnyObject;
        let b_ptr: *mut AnyObject = b as *const _ as *const AnyObject as *mut AnyObject;
        let c_ptr: *mut AnyObject = c as *const _ as *const AnyObject as *mut AnyObject;

        let a_matrix: *mut AnyObject = msg_send![class!(MPSMatrix), alloc];
        let a_matrix: *mut AnyObject = msg_send![a_matrix,
            initWithBuffer: a_ptr,
            offset: a_offset,
            descriptor: a_desc
        ];
        let b_matrix: *mut AnyObject = msg_send![class!(MPSMatrix), alloc];
        let b_matrix: *mut AnyObject = msg_send![b_matrix,
            initWithBuffer: b_ptr,
            offset: b_offset,
            descriptor: b_desc
        ];
        let c_matrix: *mut AnyObject = msg_send![class!(MPSMatrix), alloc];
        let c_matrix: *mut AnyObject = msg_send![c_matrix,
            initWithBuffer: c_ptr,
            offset: c_offset,
            descriptor: c_desc
        ];

        let device_ptr: *mut AnyObject =
            device as *const DeviceRef as *const AnyObject as *mut AnyObject;
        let cmdbuf_ptr: *mut AnyObject =
            cmdbuf as *const CommandBufferRef as *const AnyObject as *mut AnyObject;

        let gemm_kernel: *mut AnyObject = msg_send![class!(MPSMatrixMultiplication), alloc];
        let gemm_kernel: *mut AnyObject = msg_send![gemm_kernel,
            initWithDevice: device_ptr,
            transposeLeft: Bool::new(transpose_a),
            transposeRight: Bool::new(transpose_b),
            resultRows: m as u64,
            resultColumns: n as u64,
            interiorColumns: k as u64,
            alpha: alpha as f64,
            beta: beta as f64
        ];

        let _: () = msg_send![gemm_kernel,
            encodeToCommandBuffer: cmdbuf_ptr,
            leftMatrix: a_matrix,
            rightMatrix: b_matrix,
            resultMatrix: c_matrix
        ];
    }

    Ok(())
}

#[derive(Debug)]
pub enum GemmError {
    InvalidDimensions(String),
    MpsError(String),
    MetalError(String),
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

pub struct MetalGemm {
    device: Arc<MetalDevice>,
}

impl MetalGemm {
    pub fn new(device: Arc<MetalDevice>) -> Result<Self, GemmError> {
        Ok(Self { device })
    }

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
        let dtype = if use_f16 { GemmDtype::F16 } else { GemmDtype::F32 };
        let cmd_buf = stream
            .get_command_buffer()
            .map_err(|e| GemmError::ExecutionFailed(format!("{:?}", e)))?;
        encode_gemm_into_command_buffer(
            &self.device.device,
            cmd_buf,
            a,
            0,
            b,
            0,
            c,
            0,
            m,
            n,
            k,
            alpha,
            beta,
            transpose_a,
            transpose_b,
            dtype,
        )?;
        stream
            .commit()
            .map_err(|e| GemmError::ExecutionFailed(format!("{:?}", e)))?;
        Ok(())
    }

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
        let dtype = if use_f16 { GemmDtype::F16 } else { GemmDtype::F32 };
        let elem_size = dtype.elem_size();

        let a_rows = if transpose_a { k } else { m };
        let a_cols = if transpose_a { m } else { k };
        let b_rows = if transpose_b { n } else { k };
        let b_cols = if transpose_b { k } else { n };

        let a_matrix_size = (a_rows * a_cols * elem_size) as u64;
        let b_matrix_size = (b_rows * b_cols * elem_size) as u64;
        let c_matrix_size = (m * n * elem_size) as u64;

        for i in 0..batch_size {
            let cmd_buf = stream
                .get_command_buffer()
                .map_err(|e| GemmError::ExecutionFailed(format!("{:?}", e)))?;
            encode_gemm_into_command_buffer(
                &self.device.device,
                cmd_buf,
                a,
                (i as u64) * a_matrix_size,
                b,
                (i as u64) * b_matrix_size,
                c,
                (i as u64) * c_matrix_size,
                m,
                n,
                k,
                alpha,
                beta,
                transpose_a,
                transpose_b,
                dtype,
            )?;
            stream
                .commit()
                .map_err(|e| GemmError::ExecutionFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }
}
