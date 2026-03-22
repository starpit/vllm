/// Precision conversion GPU correctness tests.
use ferrite_metal::atoms::ConvertAtom;
use half::f16;
use metal::*;
use std::ffi::c_void;

fn get_device() -> (Device, CommandQueue) {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    (device, queue)
}

fn compile(device: &Device, msl: &str) -> ComputePipelineState {
    let options = CompileOptions::new();
    options.set_language_version(MTLLanguageVersion::V3_0);
    let library = device
        .new_library_with_source(msl, &options)
        .unwrap_or_else(|e| panic!("Compile failed: {}", e));
    let func = library
        .get_function("convert", None)
        .expect("no convert fn");
    device
        .new_compute_pipeline_state_with_function(&func)
        .expect("pipeline failed")
}

fn dispatch_convert(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    input_buf: &Buffer,
    output_buf: &Buffer,
    count: u32,
) {
    let count_buf = device.new_buffer_with_data(
        &count as *const u32 as *const c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(input_buf), 0);
    enc.set_buffer(1, Some(output_buf), 0);
    enc.set_buffer(2, Some(&count_buf), 0);
    let threads_per_tg = 256u64;
    let grid = MTLSize::new((count as u64 + threads_per_tg - 1) / threads_per_tg, 1, 1);
    enc.dispatch_thread_groups(grid, MTLSize::new(threads_per_tg, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

#[test]
fn test_f32_to_f16() {
    let (device, queue) = get_device();
    let msl = ConvertAtom::emit_kernel("float", "half");
    let pipeline = compile(&device, &msl);

    let input: Vec<f32> = vec![1.0, -2.5, 0.0, 3.14, 100.0, -0.001, 65504.0, 0.5];
    let count = input.len() as u32;

    let input_buf = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        (count as u64) * 4,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device.new_buffer((count as u64) * 2, MTLResourceOptions::StorageModeShared);

    dispatch_convert(&device, &queue, &pipeline, &input_buf, &output_buf, count);

    let ptr = output_buf.contents() as *const f16;
    let output = unsafe { std::slice::from_raw_parts(ptr, count as usize) };

    for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
        let expected = f16::from_f32(inp);
        assert_eq!(
            out, expected,
            "f32→f16 mismatch at [{}]: GPU={}, expected={} (from {})",
            i, out, expected, inp
        );
    }
}

#[test]
fn test_f16_to_f32() {
    let (device, queue) = get_device();
    let msl = ConvertAtom::emit_kernel("half", "float");
    let pipeline = compile(&device, &msl);

    let input: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(-2.5),
        f16::from_f32(0.0),
        f16::from_f32(3.14),
        f16::from_f32(100.0),
        f16::from_f32(-0.001),
        f16::from_f32(0.5),
        f16::from_f32(42.0),
    ];
    let count = input.len() as u32;

    let input_buf = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        (count as u64) * 2,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device.new_buffer((count as u64) * 4, MTLResourceOptions::StorageModeShared);

    dispatch_convert(&device, &queue, &pipeline, &input_buf, &output_buf, count);

    let ptr = output_buf.contents() as *const f32;
    let output = unsafe { std::slice::from_raw_parts(ptr, count as usize) };

    for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
        let expected = inp.to_f32();
        assert!(
            (out - expected).abs() < 1e-6,
            "f16→f32 mismatch at [{}]: GPU={}, expected={}",
            i,
            out,
            expected
        );
    }
}

#[test]
fn test_f32_to_f16_large() {
    let (device, queue) = get_device();
    let msl = ConvertAtom::emit_kernel("float", "half");
    let pipeline = compile(&device, &msl);

    let count = 4096u32;
    let input: Vec<f32> = (0..count).map(|i| (i as f32 - 2048.0) * 0.01).collect();

    let input_buf = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        (count as u64) * 4,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device.new_buffer((count as u64) * 2, MTLResourceOptions::StorageModeShared);

    dispatch_convert(&device, &queue, &pipeline, &input_buf, &output_buf, count);

    let ptr = output_buf.contents() as *const f16;
    let output = unsafe { std::slice::from_raw_parts(ptr, count as usize) };

    let max_err: f32 = input
        .iter()
        .zip(output.iter())
        .map(|(&inp, &out)| (out.to_f32() - f16::from_f32(inp).to_f32()).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-6, "Large f32→f16 max error: {}", max_err);
}
