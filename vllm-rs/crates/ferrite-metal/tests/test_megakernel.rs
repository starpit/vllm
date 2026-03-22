/// Megakernel GPU correctness test — one kernel dispatch for an entire
/// transformer block, verified against CPU reference.
use ferrite_metal::megakernel_emitter::{MegakernelConfig, build_megakernel_msl};
use half::f16;
use metal::*;
use std::ffi::c_void;

fn make_f32_buf(device: &Device, data: &[f32]) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const c_void,
        (data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn make_f16_buf(device: &Device, data: &[f16]) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const c_void,
        (data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn empty_f32(device: &Device, count: usize) -> Buffer {
    device.new_buffer((count * 4) as u64, MTLResourceOptions::StorageModeShared)
}

fn empty_f16(device: &Device, count: usize) -> Buffer {
    device.new_buffer((count * 2) as u64, MTLResourceOptions::StorageModeShared)
}

fn read_f32(buf: &Buffer, count: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, count) }.to_vec()
}

// CPU reference
fn cpu_transformer_block(
    x: &[f32],
    gamma: &[f32],
    wq: &[f16],
    wk: &[f16],
    wv: &[f16],
    wo: &[f16],
    seq_len: usize,
    d_model: usize,
    d_head: usize,
) -> Vec<f32> {
    let eps = 1e-5f32;
    let total = seq_len * d_model;

    // RmsNorm
    let mut h = vec![0.0f32; total];
    for r in 0..seq_len {
        let row = &x[r * d_model..(r + 1) * d_model];
        let ss: f32 = row.iter().map(|v| v * v).sum::<f32>() / d_model as f32;
        let scale = 1.0 / (ss + eps).sqrt();
        for i in 0..d_model {
            h[r * d_model + i] = row[i] * scale * gamma[i];
        }
    }

    // GEMM Q, K, V
    let h_f16: Vec<f16> = h.iter().map(|v| f16::from_f32(*v)).collect();
    let gemm = |w: &[f16]| -> Vec<f32> {
        let mut c = vec![0.0f32; total];
        for i in 0..seq_len {
            for j in 0..d_model {
                let mut sum = 0.0f32;
                for k in 0..d_model {
                    sum += h_f16[i * d_model + k].to_f32() * w[j * d_model + k].to_f32();
                }
                c[i * d_model + j] = sum;
            }
        }
        c
    };
    let q = gemm(wq);
    let k = gemm(wk);
    let v = gemm(wv);

    // Convert to f16 for attention
    let q16: Vec<f16> = q.iter().map(|v| f16::from_f32(*v)).collect();
    let k16: Vec<f16> = k.iter().map(|v| f16::from_f32(*v)).collect();
    let v16: Vec<f16> = v.iter().map(|v| f16::from_f32(*v)).collect();

    // Attention
    let attn_scale = 1.0 / (d_head as f64).sqrt();
    let mut attn_out = vec![0.0f32; total];
    for i in 0..seq_len {
        let mut s = vec![0.0f64; seq_len];
        let mut max_s = f64::NEG_INFINITY;
        for j in 0..seq_len {
            let mut dot = 0.0f64;
            for d in 0..d_head {
                dot += q16[i * d_model + d].to_f32() as f64 * k16[j * d_model + d].to_f32() as f64;
            }
            s[j] = dot * attn_scale;
            max_s = max_s.max(s[j]);
        }
        let exp_s: Vec<f64> = s.iter().map(|v| (v - max_s).exp()).collect();
        let sum_exp: f64 = exp_s.iter().sum();
        for d in 0..d_head {
            let mut val = 0.0f64;
            for j in 0..seq_len {
                val += (exp_s[j] / sum_exp) * v16[j * d_model + d].to_f32() as f64;
            }
            attn_out[i * d_model + d] = val as f32;
        }
    }

    // Output projection
    let attn_f16: Vec<f16> = attn_out.iter().map(|v| f16::from_f32(*v)).collect();
    let o = {
        let mut c = vec![0.0f32; total];
        for i in 0..seq_len {
            for j in 0..d_model {
                let mut sum = 0.0f32;
                for k in 0..d_model {
                    sum += attn_f16[i * d_model + k].to_f32() * wo[j * d_model + k].to_f32();
                }
                c[i * d_model + j] = sum;
            }
        }
        c
    };

    // Residual add
    x.iter().zip(o.iter()).map(|(a, b)| a + b).collect()
}

#[test]
fn test_megakernel_single_threadgroup() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let config = MegakernelConfig::test_config();
    let msl = build_megakernel_msl(&config);

    let options = CompileOptions::new();
    options.set_language_version(MTLLanguageVersion::V3_0);
    let library = device
        .new_library_with_source(&msl, &options)
        .unwrap_or_else(|e| panic!("Compile failed:\n{}\n\nMSL:\n{}", e, msl));
    let func = library
        .get_function("transformer_block", None)
        .expect("function not found");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .expect("pipeline failed");

    let seq_len = config.seq_len as usize;
    let d_model = config.d_model as usize;
    let total = seq_len * d_model;

    // Deterministic pseudo-random
    let mut rng = 42u64;
    let next = |rng: &mut u64| -> f32 {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*rng >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    };

    let x: Vec<f32> = (0..total).map(|_| next(&mut rng) * 0.5).collect();
    let gamma: Vec<f32> = (0..d_model)
        .map(|_| 0.5 + next(&mut rng).abs() * 0.5)
        .collect();
    let wq: Vec<f16> = (0..d_model * d_model)
        .map(|_| f16::from_f32(next(&mut rng) * 0.5))
        .collect();
    let wk: Vec<f16> = (0..d_model * d_model)
        .map(|_| f16::from_f32(next(&mut rng) * 0.5))
        .collect();
    let wv: Vec<f16> = (0..d_model * d_model)
        .map(|_| f16::from_f32(next(&mut rng) * 0.5))
        .collect();
    let wo: Vec<f16> = (0..d_model * d_model)
        .map(|_| f16::from_f32(next(&mut rng) * 0.5))
        .collect();

    // CPU reference
    let cpu_out = cpu_transformer_block(
        &x,
        &gamma,
        &wq,
        &wk,
        &wv,
        &wo,
        seq_len,
        d_model,
        config.d_head as usize,
    );

    // GPU buffers
    let x_buf = make_f32_buf(&device, &x);
    let out_buf = empty_f32(&device, total);
    let gamma_buf = make_f32_buf(&device, &gamma);
    let wq_buf = make_f16_buf(&device, &wq);
    let wk_buf = make_f16_buf(&device, &wk);
    let wv_buf = make_f16_buf(&device, &wv);
    let wo_buf = make_f16_buf(&device, &wo);
    let h_buf = empty_f32(&device, total);
    let h_f16_buf = empty_f16(&device, total);
    let qkv_buf = empty_f32(&device, 3 * total);
    let qkv_f16_buf = empty_f16(&device, 3 * total);
    let attn_buf = empty_f32(&device, total);
    let attn_f16_buf = empty_f16(&device, total);
    let o_buf = empty_f32(&device, total);
    let phase_buf = empty_f32(&device, 1); // atomic counter (unused for 1 TG)

    // Dispatch
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&x_buf), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.set_buffer(2, Some(&gamma_buf), 0);
    enc.set_buffer(3, Some(&wq_buf), 0);
    enc.set_buffer(4, Some(&wk_buf), 0);
    enc.set_buffer(5, Some(&wv_buf), 0);
    enc.set_buffer(6, Some(&wo_buf), 0);
    enc.set_buffer(7, Some(&h_buf), 0);
    enc.set_buffer(8, Some(&h_f16_buf), 0);
    enc.set_buffer(9, Some(&qkv_buf), 0);
    enc.set_buffer(10, Some(&qkv_f16_buf), 0);
    enc.set_buffer(11, Some(&attn_buf), 0);
    enc.set_buffer(12, Some(&attn_f16_buf), 0);
    enc.set_buffer(13, Some(&o_buf), 0);
    enc.set_buffer(14, Some(&phase_buf), 0);

    enc.dispatch_thread_groups(
        MTLSize::new(config.num_threadgroups as u64, 1, 1),
        MTLSize::new(config.threads_per_tg as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Verify
    let gpu_out = read_f32(&out_buf, total);
    let max_err: f32 = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);

    eprintln!("Megakernel output first 8: {:?}", &gpu_out[..8]);
    eprintln!("CPU reference first 8: {:?}", &cpu_out[..8]);
    eprintln!("Megakernel max error: {}", max_err);

    // Dump per-row errors
    for r in 0..seq_len {
        let row_err: f32 = (0..d_model)
            .map(|c| (gpu_out[r * d_model + c] - cpu_out[r * d_model + c]).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "  row {}: err={:.4}  GPU={:?}  CPU={:?}",
            r,
            row_err,
            &gpu_out[r * d_model..(r + 1) * d_model],
            &cpu_out[r * d_model..(r + 1) * d_model],
        );
    }

    // Also dump intermediates
    let h_gpu = read_f32(&h_buf, total);
    let h_err: f32 = {
        let eps = 1e-5f32;
        let mut h_cpu = vec![0.0f32; total];
        for r in 0..seq_len {
            let row = &x[r * d_model..(r + 1) * d_model];
            let ss: f32 = row.iter().map(|v| v * v).sum::<f32>() / d_model as f32;
            let scale = 1.0 / (ss + eps).sqrt();
            for i in 0..d_model {
                h_cpu[r * d_model + i] = row[i] * scale * gamma[i];
            }
        }
        h_gpu
            .iter()
            .zip(h_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
    eprintln!("  RmsNorm intermediate err: {:.2e}", h_err);

    assert!(max_err < 0.5, "Megakernel error too high: {}", max_err);
}
