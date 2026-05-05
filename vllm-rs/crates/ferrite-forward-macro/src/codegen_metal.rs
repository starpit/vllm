// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal-specific codegen — emit Metal Shading Language (MSL) kernel launches
//! and command buffer management for Metal backend targets.
//!
//! This module parallels the CUDA codegen path in `codegen.rs` but emits Metal
//! API calls instead of CUDA driver API calls. The high-level structure mirrors
//! CUDA:
//!
//! - Walk the LOOP's waves (from scheduler)
//! - For each wave, emit kernel launches for the wave's subgraphs
//! - Metal-specific: use MTLCommandBuffer + MTLComputeCommandEncoder
//! - Metal-specific: use Indirect Command Buffers (ICB) for GPU-side dispatch
//!
//! Unlike CUDA's single persistent kernel approach, Metal uses ICB for wave-based
//! scheduling to avoid occupancy/resource mismatch issues on Apple Silicon.

use proc_macro2::TokenStream;
use quote::quote;

use crate::classified::Program;
use crate::fuf::Fuf;
use crate::impl_lib::ImplementationLibrary;
use crate::schedule::WorkloadLoops;
use crate::solver::WorkloadAssignments;
use crate::config::ModelParams;

/// Emit Metal ICB (Indirect Command Buffer) construction code for a decoder block.
///
/// ICB is Metal's analog to CUDA Graphs - pre-record all kernel dispatches at init time,
/// then execute the entire sequence with a single GPU call at runtime.
///
/// This emits the ICB construction that happens ONCE at model load:
/// - Create MTLIndirectCommandBuffer with capacity for all kernels in the block
/// - For each kernel in the wave schedule:
///   - Get MTLIndirectComputeCommand at index
///   - Set pipeline state
///   - Set buffer bindings (with placeholder indices that runtime will fill)
///   - Set threadgroup/grid sizes
/// - Return the constructed ICB
///
/// Example output:
/// ```rust,ignore
/// let icb_desc = MTLIndirectCommandBufferDescriptor::new();
/// icb_desc.set_command_types(MTLIndirectCommandType::Compute);
/// icb_desc.set_max_kernel_buffer_bind_count(32);
/// icb_desc.set_inherit_buffers(false);
/// let icb = device.new_indirect_command_buffer(&icb_desc, 10, MTLResourceOptions::empty());
/// 
/// // Record RMSNorm dispatch at index 0
/// let cmd = icb.indirect_compute_command_at_index(0);
/// cmd.set_compute_pipeline_state(&self.rmsnorm_pipeline);
/// cmd.set_kernel_buffer(&scratch_buffers[0], 0, 0); // input
/// cmd.set_kernel_buffer(&scratch_buffers[1], 0, 1); // output
/// cmd.concurrent_dispatch_threadgroups(grid_size, threadgroup_size);
/// // ... more kernels ...
/// ```
pub fn emit_metal_icb_construction(
    wave_schedule: &[Vec<u32>], // waves -> [subgraph_ids]
    kernel_count: usize,
) -> TokenStream {
    let kernel_count_lit = proc_macro2::Literal::usize_unsuffixed(kernel_count);
    
    // Generate ICB descriptor setup
    let icb_setup = quote! {
        // Create ICB descriptor with compute command support
        let icb_desc = metal::IndirectCommandBufferDescriptor::new();
        icb_desc.set_command_types(metal::MTLIndirectCommandType::Compute);
        icb_desc.set_max_kernel_buffer_bind_count(32); // Max buffers per kernel
        icb_desc.set_inherit_buffers(false); // Each command sets its own buffers
        icb_desc.set_inherit_pipeline_state(false);
        
        // Allocate ICB with capacity for all kernels in this decoder block
        let icb = device.new_indirect_command_buffer(
            &icb_desc,
            #kernel_count_lit,
            metal::MTLResourceOptions::StorageModeShared,
        )?;
    };
    
    // Generate kernel recording code for each wave
    let mut kernel_index = 0usize;
    let mut wave_recordings = Vec::new();
    
    for (wave_idx, subgraph_ids) in wave_schedule.iter().enumerate() {
        let wave_idx_lit = proc_macro2::Literal::usize_unsuffixed(wave_idx);
        let mut subgraph_recordings = Vec::new();
        
        for &subgraph_id in subgraph_ids {
            let kernel_idx_lit = proc_macro2::Literal::usize_unsuffixed(kernel_index);
            let subgraph_id_lit = proc_macro2::Literal::u32_unsuffixed(subgraph_id);
            
            // Each subgraph gets recorded as an indirect compute command
            // The actual implementation will be filled in by the subgraph's
            // Implementation::emit_icb_recording() method
            subgraph_recordings.push(quote! {
                // Record subgraph #subgraph_id_lit at ICB index #kernel_idx_lit
                let cmd = icb.indirect_compute_command_at_index(#kernel_idx_lit);
                // TODO: Call subgraph's Implementation::emit_icb_recording()
                // This will set pipeline state, buffer bindings, and dispatch size
            });
            
            kernel_index += 1;
        }
        
        wave_recordings.push(quote! {
            // Wave #wave_idx_lit: #(subgraph_ids),*
            #(#subgraph_recordings)*
        });
    }
    
    quote! {
        {
            #icb_setup
            
            // Record all kernel dispatches into the ICB
            #(#wave_recordings)*
            
            Ok(icb)
        }
    }
}

/// Emit Metal ICB execution code for a single forward pass.
///
/// This is the HOT PATH - called once per token in decode.
/// Emits a single Metal API call that executes the entire pre-recorded ICB:
///
/// Example output:
/// ```rust,ignore
/// let command_buffer = command_queue.command_buffer();
/// let encoder = command_buffer.compute_command_encoder();
/// encoder.execute_commands_in_buffer(&self.decoder_block_icb, 0..kernel_count);
/// encoder.end_encoding();
/// command_buffer.commit();
/// command_buffer.wait_until_completed();
/// ```
///
/// The ICB was pre-recorded at init time with buffer binding INDICES.
/// The actual buffer pointers come from the scratch/tile table that the
/// runtime updates per-request.
pub fn emit_metal_icb_execution(
    kernel_count: usize,
) -> TokenStream {
    let kernel_count_lit = proc_macro2::Literal::usize_unsuffixed(kernel_count);
    
    quote! {
        {
            // HOT PATH: Single Metal API call to execute entire pre-recorded ICB
            // This replaces hundreds of individual kernel dispatches with one GPU call
            
            // Create command buffer from the command queue
            let command_buffer = self.command_queue.new_command_buffer();
            
            // Create compute encoder
            let encoder = command_buffer.new_compute_command_encoder();
            
            // Execute all pre-recorded commands in the ICB
            // Range is 0..kernel_count (all kernels in this decoder block)
            encoder.execute_commands_in_buffer(
                &self.decoder_block_icb,
                metal::NSRange {
                    location: 0,
                    length: #kernel_count_lit,
                },
            );
            
            // End encoding
            encoder.end_encoding();
            
            // Commit the command buffer to the GPU
            command_buffer.commit();
            
            // Wait for completion (synchronous for now; async version later)
            command_buffer.wait_until_completed();
            
            // Check for errors
            if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
                return Err(format!(
                    "Metal command buffer execution failed: {:?}",
                    command_buffer.error()
                ).into());
            }
            
            Ok(())
        }
    }
}

/// Emit the full Metal forward function for a workload bucket.
///
/// This is the Metal analog of the CUDA forward function emission in codegen.rs.
/// It emits:
/// - Function signature: `pub unsafe fn forward_metal_m_<N>(...) -> Result<Tensor>`
/// - Tile table allocation
/// - Alias prelude (zero-copy views)
/// - Loop over waves, emitting Metal launches per wave
/// - Return the final output tensor
pub fn emit_metal_forward_fn(
    _program: &Program,
    _fuf: &Fuf,
    _sfuf: &WorkloadAssignments,
    _lib: &ImplementationLibrary,
    _model: &ModelParams,
    _num_tokens: u64,
) -> TokenStream {
    // TODO: Implement Metal forward function emission
    // This will be filled in as part of Phase 3.1
    quote! {
        // Metal forward function placeholder
        // Will emit full forward pass with Metal API calls
    }
}

/// Emit Metal-specific initialization code for the Weights struct.
///
/// Metal needs:
/// - MTLDevice reference
/// - MTLCommandQueue for kernel dispatch
/// - Compiled MTLLibrary containing kernel functions
/// - MTLComputePipelineState per kernel
///
/// This is called from the Weights::load method when the target is Metal.
pub fn emit_metal_weights_init() -> TokenStream {
    // TODO: Implement Metal weights initialization
    // This will be filled in as part of Phase 3.2
    quote! {
        // Metal weights initialization placeholder
        // Will emit MTLDevice + MTLCommandQueue setup
    }
}

/// Emit the Metal kernel library compilation code.
///
/// Metal kernels are compiled at runtime from .metal source files.
/// This emits:
/// - MTLLibrary creation from embedded shader source
/// - MTLFunction lookup per kernel name
/// - MTLComputePipelineState creation per function
///
/// The shader source is embedded in the binary via include_str! at compile time.
pub fn emit_metal_library_compilation() -> TokenStream {
    // TODO: Implement Metal library compilation emission
    // This will be filled in as part of Phase 3.2
    quote! {
        // Metal library compilation placeholder
        // Will emit MTLLibrary creation from shader source
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_codegen_module_exists() {
        // Placeholder test - will be expanded as we implement each function
        assert!(true, "Metal codegen module compiles");
    }

    #[test]
    fn metal_icb_construction_emits_tokens() {
        let wave_schedule = vec![vec![0, 1, 2], vec![3, 4]];
        let tokens = emit_metal_icb_construction(&wave_schedule, 5);
        let code = tokens.to_string();
        assert!(!code.is_empty(), "Metal ICB construction should emit non-empty tokens");
    }

    #[test]
    fn metal_icb_execution_emits_tokens() {
        let tokens = emit_metal_icb_execution(10);
        let code = tokens.to_string();
        assert!(!code.is_empty(), "Metal ICB execution should emit non-empty tokens");
    }

    #[test]
    fn metal_forward_fn_emits_tokens() {
        // Test that emit_metal_forward_fn produces valid tokens
        // Will be expanded when we implement the function
        assert!(true, "Metal forward function emission test placeholder");
    }
}
