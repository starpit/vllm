// SPDX-License-Identifier: Apache-2.0
//! Qwen3.5-MoE shared-expert combine dispatcher. Mirrors the trailing
//! Python expression `routed + shared_y * sigmoid(shared_expert_gate(x))`
//! from mlx-lm `Qwen3NextSparseMoeBlock` / transformers
//! `Qwen3_5MoeSparseMoeBlock`. The gate is `[rows, 1]` — one scalar per
//! token, row-broadcast across the hidden axis.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDataType, MTLDevice,
    MTLFunctionConstantValues, MTLLibrary, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

use crate::shader_cache::load_library_from_bytes;
use crate::stream::MetalStreamError;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateScaleDType {
    F16,
    BF16,
}

impl GateScaleDType {
    fn symbol(self) -> &'static str {
        match self {
            Self::F16 => "gate_scale_f16",
            Self::BF16 => "gate_scale_bf16",
        }
    }

    fn element_size(self) -> usize {
        2
    }
}

pub struct GateScaleKernels {
    library: Library,
    device: Device,
}

impl GateScaleKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library = load_library_from_bytes(device, crate::embedded_metallib!("gate_scale"))
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "load `gate_scale.metallib`: {e}"
                ))
            })?;
        Ok(Self {
            library,
            device: device.clone(),
        })
    }

    pub fn build_pipeline(
        &self,
        dtype: GateScaleDType,
        n: u32,
        cols: u32,
    ) -> Result<ComputePipelineState, MetalStreamError> {
        let constants = MTLFunctionConstantValues::new();
        unsafe {
            constants.setConstantValue_type_atIndex(
                NonNull::new(&n as *const u32 as *mut c_void).unwrap(),
                MTLDataType::UInt,
                0,
            );
            constants.setConstantValue_type_atIndex(
                NonNull::new(&cols as *const u32 as *mut c_void).unwrap(),
                MTLDataType::UInt,
                1,
            );
        }
        let name = NSString::from_str(dtype.symbol());
        let func = self
            .library
            .newFunctionWithName_constantValues_error(&name, &constants)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("{}: {e:?}", dtype.symbol()))
            })?;
        self.device
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "{} pipeline: {e:?}",
                    dtype.symbol()
                ))
            })
    }
}

/// `out[r, c] = routed[r, c] + shared_y[r, c] * sigmoid(g[r])`.
pub fn dispatch_gate_scale(
    kernels: &GateScaleKernels,
    queue: &CommandQueue,
    routed: &Buffer,
    shared_y: &Buffer,
    gate: &Buffer,
    out: &Buffer,
    rows: u32,
    cols: u32,
    dtype: GateScaleDType,
) -> Result<(), MetalStreamError> {
    if rows == 0 || cols == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "dispatch_gate_scale: rows={rows} cols={cols}; both must be > 0"
        )));
    }
    let n = rows * cols;
    let full_bytes = (n as usize) * dtype.element_size();
    let gate_bytes = (rows as usize) * dtype.element_size();
    for (name, buf, need) in [
        ("routed", routed, full_bytes),
        ("shared_y", shared_y, full_bytes),
        ("gate", gate, gate_bytes),
        ("out", out, full_bytes),
    ] {
        if buf.length() < need {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "gate_scale {name} too small: have {} need {need}",
                buf.length()
            )));
        }
    }

    let pipeline = kernels.build_pipeline(dtype, n, cols)?;
    let cmdbuf = queue
        .commandBuffer()
        .ok_or_else(|| MetalStreamError::ShaderCompilationFailed("commandBuffer nil".into()))?;
    let enc = cmdbuf.computeCommandEncoder().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("computeCommandEncoder nil".into())
    })?;
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(out), 0, 0);
        enc.setBuffer_offset_atIndex(Some(routed), 0, 1);
        enc.setBuffer_offset_atIndex(Some(shared_y), 0, 2);
        enc.setBuffer_offset_atIndex(Some(gate), 0, 3);
    }
    let grid = MTLSize {
        width: n as usize,
        height: 1,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: (n as usize).min(256),
        height: 1,
        depth: 1,
    };
    enc.dispatchThreads_threadsPerThreadgroup(grid, threads_per_tg);
    enc.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();
    if cmdbuf.status() != MTLCommandBufferStatus::Completed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "gate_scale status {:?}",
            cmdbuf.status()
        )));
    }
    Ok(())
}

/// CPU reference: `out[r, c] = routed[r, c] + shared_y[r, c] * sigmoid(g[r])`.
pub fn gate_scale_cpu_f32(
    routed: &[f32],
    shared_y: &[f32],
    gate: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    assert_eq!(routed.len(), rows * cols);
    assert_eq!(shared_y.len(), rows * cols);
    assert_eq!(gate.len(), rows);
    assert_eq!(out.len(), rows * cols);
    for r in 0..rows {
        let sig = 1.0_f32 / (1.0 + (-gate[r]).exp());
        for c in 0..cols {
            out[r * cols + c] = routed[r * cols + c] + shared_y[r * cols + c] * sig;
        }
    }
}
