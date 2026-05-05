// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Minimal test to isolate ICB recording crash

#[cfg(test)]
mod tests {
    use crate::detect_device;
    use crate::instruction_executor::RecordingContext;
    use foreign_types::ForeignType;
    use metal::MTLSize;
    use std::sync::Arc;

    #[test]
    fn test_minimal_icb_command_recording() {
        let device = detect_device().expect("Metal device required");
        let mut ctx = RecordingContext::new(Arc::new(device.device.clone()), 10)
            .expect("Failed to create recording context");

        // Create a simple compute pipeline
        let source = r#"
            #include <metal_stdlib>
            using namespace metal;
            
            kernel void test_kernel(
                device float* data [[buffer(0)]],
                uint idx [[thread_position_in_grid]]
            ) {
                data[idx] = float(idx);
            }
        "#;

        let library = device
            .device
            .new_library_with_source(source, &metal::CompileOptions::new())
            .expect("Failed to compile test shader");

        let function = library
            .get_function("test_kernel", None)
            .expect("Failed to get function");

        let pipeline = device
            .device
            .new_compute_pipeline_state_with_function(&function)
            .expect("Failed to create pipeline");

        // Create a dummy buffer
        let buffer = device
            .device
            .new_buffer(1024, metal::MTLResourceOptions::StorageModeShared);

        // Record dispatch using the new pattern (no pipeline set on command)
        println!("Recording ICB command...");
        let threadgroups = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        ctx.record_compute_dispatch(
            &pipeline, // Ignored but kept for API compatibility
            &[(&buffer, 0, 0)],
            threadgroups,
            threads_per_threadgroup,
        );

        println!("ICB command recorded successfully!");
        println!("Command count: {}", ctx.command_index);

        // Verify we can record multiple commands
        ctx.record_compute_dispatch(
            &pipeline,
            &[(&buffer, 0, 0)],
            threadgroups,
            threads_per_threadgroup,
        );

        println!("Multiple commands recorded successfully!");
        println!("Final command count: {}", ctx.command_index);
    }
}
