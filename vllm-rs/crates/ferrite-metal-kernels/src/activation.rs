use metal::{Buffer, CommandBufferRef, ComputeCommandEncoderRef, Device, MTLSize};
use std::sync::Arc;

use crate::shader_cache::ShaderCache;
use crate::stream::MetalStreamError;

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    /// SiLU (Swish): x * sigmoid(x)
    SiLU,
    /// GELU (exact): 0.5 * x * (1 + erf(x / sqrt(2)))
    GELU,
    /// GELU tanh approximation
    GELUTanh,
    /// GELU quick approximation: x * sigmoid(1.702 * x)
    GELUQuick,
    /// FatReLU: max(0, x) with threshold
    FatReLU,
}

/// Data type for activation functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    F16,
    BF16,
    F32,
}

impl DataType {
    pub fn element_size(&self) -> usize {
        match self {
            DataType::F16 => 2,
            DataType::BF16 => 2,
            DataType::F32 => 4,
        }
    }
}

/// Metal activation function executor
pub struct MetalActivation {
    device: Device,
    shader_cache: Arc<ShaderCache>,
}

impl MetalActivation {
    /// Create a new activation function executor
    pub fn new(device: Device) -> Result<Self, MetalStreamError> {
        let shader_cache = Arc::new(ShaderCache::new(device.clone())?);
        Ok(Self {
            device,
            shader_cache,
        })
    }

    /// Execute an activation function
    ///
    /// # Arguments
    /// * `output` - Output buffer
    /// * `input` - Input buffer
    /// * `n` - Number of elements
    /// * `activation` - Activation function type
    /// * `dtype` - Data type
    /// * `threshold` - Threshold for FatReLU (ignored for other activations)
    /// * `encoder` - Command encoder
    pub fn execute(
        &self,
        output: &Buffer,
        input: &Buffer,
        n: u32,
        activation: ActivationType,
        dtype: DataType,
        threshold: f32,
        encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), MetalStreamError> {
        // Get kernel name
        let kernel_name = self.get_kernel_name(activation, dtype);

        // Get pipeline state
        let pipeline = self.shader_cache.get_pipeline(&kernel_name)?;

        // Set pipeline state
        encoder.set_compute_pipeline_state(&pipeline);

        // Set buffers
        encoder.set_buffer(0, Some(output), 0);
        encoder.set_buffer(1, Some(input), 0);

        // Create constant buffer for n
        let n_buffer = self.device.new_buffer_with_data(
            &n as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(2, Some(&n_buffer), 0);

        // For FatReLU, set threshold parameter
        if activation == ActivationType::FatReLU {
            let threshold_buffer = self.device.new_buffer_with_data(
                &threshold as *const f32 as *const _,
                std::mem::size_of::<f32>() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            encoder.set_buffer(3, Some(&threshold_buffer), 0);
        }

        // Calculate grid size
        let threadgroup_size = MTLSize::new(256, 1, 1);
        let num_threadgroups = MTLSize::new(((n as u64 + 255) / 256), 1, 1);

        // Dispatch
        encoder.dispatch_thread_groups(num_threadgroups, threadgroup_size);

        Ok(())
    }

    /// Execute vectorized SiLU (processes 4 elements at a time)
    ///
    /// # Arguments
    /// * `output` - Output buffer (must be aligned to 8 bytes for half4)
    /// * `input` - Input buffer (must be aligned to 8 bytes for half4)
    /// * `n` - Number of vec4 elements (total elements / 4)
    /// * `encoder` - Command encoder
    pub fn execute_silu_vec4(
        &self,
        output: &Buffer,
        input: &Buffer,
        n: u32,
        encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), MetalStreamError> {
        // Get pipeline state for vectorized SiLU
        let pipeline = self.shader_cache.get_pipeline("silu_vec4_f16")?;

        // Set pipeline state
        encoder.set_compute_pipeline_state(&pipeline);

        // Set buffers
        encoder.set_buffer(0, Some(output), 0);
        encoder.set_buffer(1, Some(input), 0);

        // Create constant buffer for n
        let n_buffer = self.device.new_buffer_with_data(
            &n as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(2, Some(&n_buffer), 0);

        // Calculate grid size
        let threadgroup_size = MTLSize::new(256, 1, 1);
        let num_threadgroups = MTLSize::new(((n as u64 + 255) / 256), 1, 1);

        // Dispatch
        encoder.dispatch_thread_groups(num_threadgroups, threadgroup_size);

        Ok(())
    }

    /// Get kernel name for activation function and data type
    fn get_kernel_name(&self, activation: ActivationType, dtype: DataType) -> String {
        let dtype_suffix = match dtype {
            DataType::F16 => "f16",
            DataType::BF16 => "bf16",
            DataType::F32 => "f32",
        };

        let activation_prefix = match activation {
            ActivationType::SiLU => "silu",
            ActivationType::GELU => "gelu",
            ActivationType::GELUTanh => "gelu_tanh",
            ActivationType::GELUQuick => "gelu_quick",
            ActivationType::FatReLU => "fatrelu",
        };

        format!("{}_{}", activation_prefix, dtype_suffix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{detect_device, MetalDevice};

    #[test]
    fn test_silu_f16() {
        let device = detect_device().expect("Metal device required");
        let activation = MetalActivation::new(device.device.clone()).unwrap();

        // Create test data: [-2.0, -1.0, 0.0, 1.0, 2.0]
        let input_data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let n = input_data.len() as u32;

        // Convert to f16
        let input_f16: Vec<u16> = input_data
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();

        // Create buffers
        let input_buffer = device.device.new_buffer_with_data(
            input_f16.as_ptr() as *const _,
            (input_f16.len() * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = device
            .device
            .new_buffer((n as u64 * 2), metal::MTLResourceOptions::StorageModeShared);

        // Create command buffer and encoder
        let command_buffer = device.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Execute SiLU
        activation
            .execute(
                &output_buffer,
                &input_buffer,
                n,
                ActivationType::SiLU,
                DataType::F16,
                0.0,
                encoder,
            )
            .unwrap();

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let output_ptr = output_buffer.contents() as *const u16;
        let output_f16: Vec<u16> =
            unsafe { std::slice::from_raw_parts(output_ptr, n as usize) }.to_vec();
        let output_data: Vec<f32> = output_f16
            .iter()
            .map(|&x| half::f16::from_bits(x).to_f32())
            .collect();

        // Verify results: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        for (i, &x) in input_data.iter().enumerate() {
            let expected = x / (1.0 + (-x).exp());
            let actual = output_data[i];
            let diff = (expected - actual).abs();
            assert!(
                diff < 0.01,
                "SiLU mismatch at index {}: expected {}, got {}, diff {}",
                i,
                expected,
                actual,
                diff
            );
        }
    }

    #[test]
    fn test_gelu_f16() {
        let device = detect_device().expect("Metal device required");
        let activation = MetalActivation::new(device.device.clone()).unwrap();

        // Create test data: [-2.0, -1.0, 0.0, 1.0, 2.0]
        let input_data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let n = input_data.len() as u32;

        // Convert to f16
        let input_f16: Vec<u16> = input_data
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();

        // Create buffers
        let input_buffer = device.device.new_buffer_with_data(
            input_f16.as_ptr() as *const _,
            (input_f16.len() * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = device
            .device
            .new_buffer((n as u64 * 2), metal::MTLResourceOptions::StorageModeShared);

        // Create command buffer and encoder
        let command_buffer = device.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Execute GELU
        activation
            .execute(
                &output_buffer,
                &input_buffer,
                n,
                ActivationType::GELU,
                DataType::F16,
                0.0,
                encoder,
            )
            .unwrap();

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let output_ptr = output_buffer.contents() as *const u16;
        let output_f16: Vec<u16> =
            unsafe { std::slice::from_raw_parts(output_ptr, n as usize) }.to_vec();
        let output_data: Vec<f32> = output_f16
            .iter()
            .map(|&x| half::f16::from_bits(x).to_f32())
            .collect();

        // Verify results: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        const SQRT_HALF: f32 = 0.70710678118;
        for (i, &x) in input_data.iter().enumerate() {
            // Use libm::erff for erf function
            let expected = 0.5 * x * (1.0 + libm::erff(x * SQRT_HALF));
            let actual = output_data[i];
            let diff = (expected - actual).abs();
            assert!(
                diff < 0.02,
                "GELU mismatch at index {}: expected {}, got {}, diff {}",
                i,
                expected,
                actual,
                diff
            );
        }
    }

    #[test]
    fn test_fatrelu_f16() {
        let device = detect_device().expect("Metal device required");
        let activation = MetalActivation::new(device.device.clone()).unwrap();

        // Create test data: [-2.0, -1.0, 0.0, 1.0, 2.0]
        let input_data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let n = input_data.len() as u32;
        let threshold = 0.5f32;

        // Convert to f16
        let input_f16: Vec<u16> = input_data
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();

        // Create buffers
        let input_buffer = device.device.new_buffer_with_data(
            input_f16.as_ptr() as *const _,
            (input_f16.len() * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = device
            .device
            .new_buffer((n as u64 * 2), metal::MTLResourceOptions::StorageModeShared);

        // Create command buffer and encoder
        let command_buffer = device.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Execute FatReLU
        activation
            .execute(
                &output_buffer,
                &input_buffer,
                n,
                ActivationType::FatReLU,
                DataType::F16,
                threshold,
                encoder,
            )
            .unwrap();

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let output_ptr = output_buffer.contents() as *const u16;
        let output_f16: Vec<u16> =
            unsafe { std::slice::from_raw_parts(output_ptr, n as usize) }.to_vec();
        let output_data: Vec<f32> = output_f16
            .iter()
            .map(|&x| half::f16::from_bits(x).to_f32())
            .collect();

        // Verify results: FatReLU(x) = x if x > threshold else 0
        for (i, &x) in input_data.iter().enumerate() {
            let expected = if x > threshold { x } else { 0.0 };
            let actual = output_data[i];
            let diff = (expected - actual).abs();
            assert!(
                diff < 0.01,
                "FatReLU mismatch at index {}: expected {}, got {}, diff {}",
                i,
                expected,
                actual,
                diff
            );
        }
    }
}
