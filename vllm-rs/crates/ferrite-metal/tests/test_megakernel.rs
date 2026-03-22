/// Megakernel GPU correctness test — one kernel dispatch for a full
/// transformer block (attention + FFN), verified against CPU reference.
use ferrite_metal::megakernel_emitter::{MegakernelConfig, build_megakernel_msl};
use half::f16;
use metal::*;
use std::ffi::c_void;

fn make_f32(device: &Device, data: &[f32]) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const c_void,
        (data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}
fn make_f16(device: &Device, data: &[f16]) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const c_void,
        (data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}
fn empty_f32(device: &Device, n: usize) -> Buffer {
    device.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared)
}
fn empty_f16(device: &Device, n: usize) -> Buffer {
    device.new_buffer((n * 2) as u64, MTLResourceOptions::StorageModeShared)
}
fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
    let p = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
}

// CPU reference: full transformer block (attention + FFN)
fn cpu_block(
    x: &[f32],
    gamma_attn: &[f32],
    wq: &[f16],
    wk: &[f16],
    wv: &[f16],
    wo: &[f16],
    gamma_ffn: &[f32],
    w_gate: &[f16],
    w_up: &[f16],
    w_down: &[f16],
    sl: usize,
    dm: usize,
    dh: usize,
    df: usize,
) -> Vec<f32> {
    let eps = 1e-5f32;
    let tot = sl * dm;

    let rmsnorm = |inp: &[f32], g: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0; tot];
        for r in 0..sl {
            let row = &inp[r * dm..(r + 1) * dm];
            let ss: f32 = row.iter().map(|v| v * v).sum::<f32>() / dm as f32;
            let sc = 1.0 / (ss + eps).sqrt();
            for i in 0..dm {
                out[r * dm + i] = row[i] * sc * g[i];
            }
        }
        out
    };

    let gemm = |a_f16: &[f16], w: &[f16], m: usize, n: usize, k: usize| -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for kk in 0..k {
                    s += a_f16[i * k + kk].to_f32() * w[j * k + kk].to_f32();
                }
                c[i * n + j] = s;
            }
        }
        c
    };

    // Attention half
    let h = rmsnorm(x, gamma_attn);
    let hf: Vec<f16> = h.iter().map(|v| f16::from_f32(*v)).collect();
    let q = gemm(&hf, wq, sl, dm, dm);
    let k = gemm(&hf, wk, sl, dm, dm);
    let v = gemm(&hf, wv, sl, dm, dm);
    let qf: Vec<f16> = q.iter().map(|v| f16::from_f32(*v)).collect();
    let kf: Vec<f16> = k.iter().map(|v| f16::from_f32(*v)).collect();
    let vf: Vec<f16> = v.iter().map(|v| f16::from_f32(*v)).collect();

    let scale = 1.0 / (dh as f64).sqrt();
    let mut attn = vec![0.0f32; tot];
    for i in 0..sl {
        let mut s = vec![0.0f64; sl];
        let mut mx = f64::NEG_INFINITY;
        for j in 0..sl {
            let mut d = 0.0f64;
            for dd in 0..dh {
                d += qf[i * dm + dd].to_f32() as f64 * kf[j * dm + dd].to_f32() as f64;
            }
            s[j] = d * scale;
            mx = mx.max(s[j]);
        }
        let es: Vec<f64> = s.iter().map(|v| (v - mx).exp()).collect();
        let se: f64 = es.iter().sum();
        for dd in 0..dh {
            let mut v = 0.0f64;
            for j in 0..sl {
                v += (es[j] / se) * vf[j * dm + dd].to_f32() as f64;
            }
            attn[i * dm + dd] = v as f32;
        }
    }

    let af: Vec<f16> = attn.iter().map(|v| f16::from_f32(*v)).collect();
    let o = gemm(&af, wo, sl, dm, dm);
    let mut x2: Vec<f32> = x.iter().zip(o.iter()).map(|(a, b)| a + b).collect();

    // FFN half
    let h2 = rmsnorm(&x2, gamma_ffn);
    let h2f: Vec<f16> = h2.iter().map(|v| f16::from_f32(*v)).collect();
    let gate = gemm(&h2f, w_gate, sl, df, dm);
    let up = gemm(&h2f, w_up, sl, df, dm);

    let mut act_f16 = vec![f16::from_f32(0.0); sl * df];
    for i in 0..sl * df {
        let g = gate[i];
        let silu = g / (1.0 + (-g).exp());
        act_f16[i] = f16::from_f32(silu * up[i]);
    }

    let down = gemm(&act_f16, w_down, sl, dm, df);
    for i in 0..tot {
        x2[i] += down[i];
    }
    x2
}

#[test]
fn test_megakernel_full_block() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let config = MegakernelConfig::test_config();
    let msl = build_megakernel_msl(&config);

    let options = CompileOptions::new();
    options.set_language_version(MTLLanguageVersion::V3_0);
    let library = device
        .new_library_with_source(&msl, &options)
        .unwrap_or_else(|e| panic!("Compile:\n{}\n\nMSL:\n{}", e, msl));
    let func = library
        .get_function("transformer_block", None)
        .expect("fn not found");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .expect("pipeline");

    let sl = config.seq_len as usize;
    let dm = config.d_model as usize;
    let df = config.d_ffn as usize;
    let tot = sl * dm;

    let mut rng = 42u64;
    let next = |r: &mut u64| -> f32 {
        *r = r.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*r >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    };

    let x: Vec<f32> = (0..tot).map(|_| next(&mut rng) * 0.5).collect();
    let gamma_attn: Vec<f32> = (0..dm).map(|_| 0.5 + next(&mut rng).abs() * 0.5).collect();
    let wq: Vec<f16> = (0..dm * dm)
        .map(|_| f16::from_f32(next(&mut rng) * 0.3))
        .collect();
    let wk: Vec<f16> = (0..dm * dm)
        .map(|_| f16::from_f32(next(&mut rng) * 0.3))
        .collect();
    let wv: Vec<f16> = (0..dm * dm)
        .map(|_| f16::from_f32(next(&mut rng) * 0.3))
        .collect();
    let wo: Vec<f16> = (0..dm * dm)
        .map(|_| f16::from_f32(next(&mut rng) * 0.3))
        .collect();
    let gamma_ffn: Vec<f32> = (0..dm).map(|_| 0.5 + next(&mut rng).abs() * 0.5).collect();
    let w_gate: Vec<f16> = (0..df * dm)
        .map(|_| f16::from_f32(next(&mut rng) * 0.3))
        .collect();
    let w_up: Vec<f16> = (0..df * dm)
        .map(|_| f16::from_f32(next(&mut rng) * 0.3))
        .collect();
    let w_down: Vec<f16> = (0..dm * df)
        .map(|_| f16::from_f32(next(&mut rng) * 0.3))
        .collect();

    let cpu_out = cpu_block(
        &x,
        &gamma_attn,
        &wq,
        &wk,
        &wv,
        &wo,
        &gamma_ffn,
        &w_gate,
        &w_up,
        &w_down,
        sl,
        dm,
        config.d_head as usize,
        df,
    );

    // GPU buffers (match kernel signature order)
    let bufs: Vec<Buffer> = vec![
        make_f32(&device, &x),          // 0: x
        empty_f32(&device, tot),        // 1: out
        make_f32(&device, &gamma_attn), // 2: gamma
        make_f16(&device, &wq),         // 3: Wq
        make_f16(&device, &wk),         // 4: Wk
        make_f16(&device, &wv),         // 5: Wv
        make_f16(&device, &wo),         // 6: Wo
        empty_f32(&device, tot),        // 7: h
        empty_f16(&device, tot),        // 8: h_f16
        empty_f32(&device, 3 * tot),    // 9: qkv
        empty_f16(&device, 3 * tot),    // 10: qkv_f16
        empty_f32(&device, tot),        // 11: attn_out
        empty_f16(&device, tot),        // 12: attn_f16
        empty_f32(&device, tot),        // 13: o
        make_f32(&device, &gamma_ffn),  // 14: gamma_ffn
        make_f16(&device, &w_gate),     // 15: W_gate
        make_f16(&device, &w_up),       // 16: W_up
        make_f16(&device, &w_down),     // 17: W_down
        empty_f32(&device, tot),        // 18: h_ffn
        empty_f16(&device, tot),        // 19: h_ffn_f16
        empty_f32(&device, sl * df),    // 20: gate_out
        empty_f32(&device, sl * df),    // 21: up_out
        empty_f16(&device, sl * df),    // 22: ffn_act
        empty_f32(&device, tot),        // 23: down_out
        empty_f32(&device, 1),          // 24: phase_counter
    ];

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    for (i, buf) in bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(buf), 0);
    }
    enc.dispatch_thread_groups(
        MTLSize::new(config.num_threadgroups as u64, 1, 1),
        MTLSize::new(config.threads_per_tg as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gpu_out = read_f32(&bufs[1], tot);
    let max_err: f32 = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);

    eprintln!("GPU first 8:  {:?}", &gpu_out[..8]);
    eprintln!("CPU first 8:  {:?}", &cpu_out[..8]);
    eprintln!("Max error: {}", max_err);

    assert!(max_err < 0.01, "Full block error too high: {}", max_err);
}
