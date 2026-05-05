#[cfg(test)]
mod tests {
    use crate::detect_device;
    use crate::instruction_executor::RecordingContext;
    use foreign_types::ForeignType;
    use metal::MTLSize;
    use std::sync::Arc;

    #[test]
    fn test_direct_dispatch_works() {
        let device = detect_device().expect("Metal device required");

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

        // Create output buffer
        let buffer_size = 256 * std::mem::size_of::<f32>();
        let buffer = device.device.new_buffer(
            buffer_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Initialize buffer to zeros
        unsafe {
            let ptr = buffer.contents() as *mut f32;
            for i in 0..256 {
                *ptr.add(i) = 0.0;
            }
        }

        // Execute DIRECTLY (not via ICB)
        println!("Executing direct dispatch...");
        let queue = device.device.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer), 0);

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
        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        println!("Direct dispatch completed!");

        // Verify results
        println!("Verifying direct dispatch results...");
        unsafe {
            let ptr = buffer.contents() as *const f32;
            let mut all_correct = true;
            for i in 0..10 {
                let expected = i as f32;
                let actual = *ptr.add(i);
                println!("  buffer[{}] = {} (expected {})", i, actual, expected);
                if (actual - expected).abs() > 0.001 {
                    all_correct = false;
                }
            }

            assert!(all_correct, "Direct dispatch should work!");
        }

        println!("✓ Direct dispatch works correctly!");
    }
}
