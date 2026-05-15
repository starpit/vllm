#[cfg(test)]
mod tests {
    use crate::detect_device;
    use crate::instruction_executor::RecordingContext;
    use foreign_types::ForeignType;
    use metal::MTLSize;
    use std::sync::Arc;

    #[test]
    fn test_icb_execution_with_inherited_pipeline() {
        let device = detect_device().expect("Metal device required");
        let mut ctx = RecordingContext::new(Arc::new(device.device.clone()), 10)
            .expect("Failed to create recording context");

        // Create a simple compute pipeline that writes thread IDs
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

        // Record ICB command
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
            &pipeline,
            &[(&buffer, 0, 0)],
            threadgroups,
            threads_per_threadgroup,
        );

        println!("ICB command recorded!");

        // Now execute the ICB on GPU
        println!("Executing ICB on GPU...");
        let queue = device.device.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // CRITICAL: Set pipeline state on encoder (inherited by ICB commands)
        encoder.set_compute_pipeline_state(&pipeline);

        // DO NOT reset here - that would clear the commands we just recorded!
        // The reset should happen BEFORE recording, not before execution

        // Execute the ICB using our helper method
        ctx.execute_on_encoder(&encoder, 0..ctx.command_index);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        println!("ICB execution completed!");

        // Verify results
        println!("Verifying results...");
        unsafe {
            let ptr = buffer.contents() as *const f32;
            let mut all_correct = true;
            for i in 0..256 {
                let expected = i as f32;
                let actual = *ptr.add(i);
                if (actual - expected).abs() > 0.001 {
                    println!(
                        "Mismatch at index {}: expected {}, got {}",
                        i, expected, actual
                    );
                    all_correct = false;
                    if i > 10 {
                        break; // Don't spam too many errors
                    }
                }
            }

            if all_correct {
                println!("✓ All values correct!");
            } else {
                panic!("ICB execution produced incorrect results");
            }
        }

        println!("Test passed!");
    }
}
