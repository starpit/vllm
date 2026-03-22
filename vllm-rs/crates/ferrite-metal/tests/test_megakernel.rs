/// Megakernel GPU correctness tests — single and multi-layer.
use ferrite_metal::megakernel_emitter::{MegakernelConfig, build_megakernel_msl};
use half::f16;
use metal::*;
use std::ffi::c_void;

fn make_f32(d: &Device, data: &[f32]) -> Buffer {
    d.new_buffer_with_data(
        data.as_ptr() as *const c_void,
        (data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}
fn make_f16(d: &Device, data: &[f16]) -> Buffer {
    d.new_buffer_with_data(
        data.as_ptr() as *const c_void,
        (data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}
fn empty_f32(d: &Device, n: usize) -> Buffer {
    d.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared)
}
fn empty_f16(d: &Device, n: usize) -> Buffer {
    d.new_buffer((n * 2) as u64, MTLResourceOptions::StorageModeShared)
}
fn read_f32(b: &Buffer, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n) }.to_vec()
}

struct LayerWeights {
    gamma_attn: Vec<f32>,
    gamma_ffn: Vec<f32>,
    wq: Vec<f16>,
    wk: Vec<f16>,
    wv: Vec<f16>,
    wo: Vec<f16>,
    w_gate: Vec<f16>,
    w_up: Vec<f16>,
    w_down: Vec<f16>,
}

fn random_layer_weights(rng: &mut u64, dm: usize, df: usize) -> LayerWeights {
    let next = |r: &mut u64| -> f32 {
        *r = r.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*r >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    };
    LayerWeights {
        gamma_attn: (0..dm).map(|_| 0.5 + next(rng).abs() * 0.5).collect(),
        gamma_ffn: (0..dm).map(|_| 0.5 + next(rng).abs() * 0.5).collect(),
        wq: (0..dm * dm)
            .map(|_| f16::from_f32(next(rng) * 0.3))
            .collect(),
        wk: (0..dm * dm)
            .map(|_| f16::from_f32(next(rng) * 0.3))
            .collect(),
        wv: (0..dm * dm)
            .map(|_| f16::from_f32(next(rng) * 0.3))
            .collect(),
        wo: (0..dm * dm)
            .map(|_| f16::from_f32(next(rng) * 0.3))
            .collect(),
        w_gate: (0..df * dm)
            .map(|_| f16::from_f32(next(rng) * 0.3))
            .collect(),
        w_up: (0..df * dm)
            .map(|_| f16::from_f32(next(rng) * 0.3))
            .collect(),
        w_down: (0..dm * df)
            .map(|_| f16::from_f32(next(rng) * 0.3))
            .collect(),
    }
}

/// Pack all layers' gammas into one contiguous buffer: [layer0_attn, layer0_ffn, layer1_attn, ...]
fn pack_gammas(layers: &[LayerWeights]) -> Vec<f32> {
    let mut out = Vec::new();
    for l in layers {
        out.extend_from_slice(&l.gamma_attn);
        out.extend_from_slice(&l.gamma_ffn);
    }
    out
}

/// Pack all layers' weights into one contiguous buffer: [layer0: Wq|Wk|Wv|Wo|Wgate|Wup|Wdown, layer1: ...]
fn pack_weights(layers: &[LayerWeights]) -> Vec<f16> {
    let mut out = Vec::new();
    for l in layers {
        out.extend_from_slice(&l.wq);
        out.extend_from_slice(&l.wk);
        out.extend_from_slice(&l.wv);
        out.extend_from_slice(&l.wo);
        out.extend_from_slice(&l.w_gate);
        out.extend_from_slice(&l.w_up);
        out.extend_from_slice(&l.w_down);
    }
    out
}

// CPU reference: one transformer block
fn cpu_one_block(x: &mut Vec<f32>, lw: &LayerWeights, sl: usize, dm: usize, dh: usize, df: usize) {
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

    let gemm = |a: &[f16], w: &[f16], m: usize, n: usize, k: usize| -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for kk in 0..k {
                    s += a[i * k + kk].to_f32() * w[j * k + kk].to_f32();
                }
                c[i * n + j] = s;
            }
        }
        c
    };

    // Attention half
    let h = rmsnorm(x, &lw.gamma_attn);
    let hf: Vec<f16> = h.iter().map(|v| f16::from_f32(*v)).collect();
    let q = gemm(&hf, &lw.wq, sl, dm, dm);
    let k = gemm(&hf, &lw.wk, sl, dm, dm);
    let v = gemm(&hf, &lw.wv, sl, dm, dm);
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
    let o = gemm(&af, &lw.wo, sl, dm, dm);
    for i in 0..tot {
        x[i] += o[i];
    }

    // FFN half
    let h2 = rmsnorm(x, &lw.gamma_ffn);
    let h2f: Vec<f16> = h2.iter().map(|v| f16::from_f32(*v)).collect();
    let gate = gemm(&h2f, &lw.w_gate, sl, df, dm);
    let up = gemm(&h2f, &lw.w_up, sl, df, dm);
    let mut act = vec![f16::from_f32(0.0); sl * df];
    for i in 0..sl * df {
        let g = gate[i];
        act[i] = f16::from_f32(g / (1.0 + (-g).exp()) * up[i]);
    }
    let down = gemm(&act, &lw.w_down, sl, dm, df);
    for i in 0..tot {
        x[i] += down[i];
    }
}

fn run_megakernel(config: &MegakernelConfig, x: &[f32], layers: &[LayerWeights]) -> Vec<f32> {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let msl = build_megakernel_msl(config);
    let options = CompileOptions::new();
    options.set_language_version(MTLLanguageVersion::V3_0);
    let library = device
        .new_library_with_source(&msl, &options)
        .unwrap_or_else(|e| panic!("Compile:\n{}\n\nMSL:\n{}", e, msl));
    let func = library.get_function("transformer_block", None).expect("fn");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .expect("pipe");

    let sl = config.seq_len as usize;
    let dm = config.d_model as usize;
    let df = config.d_ffn as usize;
    let tot = sl * dm;

    let all_gamma = pack_gammas(layers);
    let all_weights = pack_weights(layers);

    let bufs: Vec<Buffer> = vec![
        make_f32(&device, x),            // 0: x
        empty_f32(&device, tot),         // 1: out
        make_f32(&device, &all_gamma),   // 2: all_gamma
        make_f16(&device, &all_weights), // 3: all_weights
        empty_f32(&device, tot),         // 4: h
        empty_f16(&device, tot),         // 5: h_f16
        empty_f32(&device, 3 * tot),     // 6: qkv
        empty_f16(&device, 3 * tot),     // 7: qkv_f16
        empty_f32(&device, tot),         // 8: attn_out
        empty_f16(&device, tot),         // 9: attn_f16
        empty_f32(&device, tot),         // 10: o
        empty_f32(&device, tot),         // 11: h_ffn
        empty_f16(&device, tot),         // 12: h_ffn_f16
        empty_f32(&device, sl * df),     // 13: gate_out
        empty_f32(&device, sl * df),     // 14: up_out
        empty_f16(&device, sl * df),     // 15: ffn_act
        empty_f32(&device, tot),         // 16: down_out
        empty_f32(&device, 1),           // 17: phase_counter
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

    read_f32(&bufs[1], tot)
}

#[test]
fn test_megakernel_single_layer() {
    let config = MegakernelConfig::test_config();
    let mut rng = 42u64;
    let sl = config.seq_len as usize;
    let dm = config.d_model as usize;
    let df = config.d_ffn as usize;

    let next = |r: &mut u64| -> f32 {
        *r = r.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*r >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    };
    let x: Vec<f32> = (0..sl * dm).map(|_| next(&mut rng) * 0.5).collect();
    let layers = vec![random_layer_weights(&mut rng, dm, df)];

    let mut cpu_x = x.clone();
    cpu_one_block(&mut cpu_x, &layers[0], sl, dm, config.d_head as usize, df);

    let gpu_out = run_megakernel(&config, &x, &layers);
    let max_err: f32 = gpu_out
        .iter()
        .zip(cpu_x.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    eprintln!("Single layer max error: {}", max_err);
    assert!(max_err < 0.01, "Single layer error: {}", max_err);
}

#[test]
fn test_megakernel_two_layers() {
    let config = MegakernelConfig::test_config_multi_layer(2);
    let mut rng = 42u64;
    let sl = config.seq_len as usize;
    let dm = config.d_model as usize;
    let df = config.d_ffn as usize;
    let dh = config.d_head as usize;

    let next = |r: &mut u64| -> f32 {
        *r = r.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*r >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    };
    let x: Vec<f32> = (0..sl * dm).map(|_| next(&mut rng) * 0.5).collect();
    let layers = vec![
        random_layer_weights(&mut rng, dm, df),
        random_layer_weights(&mut rng, dm, df),
    ];

    // CPU reference: two blocks in sequence
    let mut cpu_x = x.clone();
    cpu_one_block(&mut cpu_x, &layers[0], sl, dm, dh, df);
    cpu_one_block(&mut cpu_x, &layers[1], sl, dm, dh, df);

    let gpu_out = run_megakernel(&config, &x, &layers);
    let max_err: f32 = gpu_out
        .iter()
        .zip(cpu_x.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    eprintln!("Two layer max error: {}", max_err);
    assert!(max_err < 0.01, "Two layer error: {}", max_err);
}

#[test]
fn test_megakernel_four_layers() {
    let config = MegakernelConfig::test_config_multi_layer(4);
    let mut rng = 123u64;
    let sl = config.seq_len as usize;
    let dm = config.d_model as usize;
    let df = config.d_ffn as usize;
    let dh = config.d_head as usize;

    let next = |r: &mut u64| -> f32 {
        *r = r.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*r >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    };
    let x: Vec<f32> = (0..sl * dm).map(|_| next(&mut rng) * 0.5).collect();
    let layers: Vec<LayerWeights> = (0..4)
        .map(|_| random_layer_weights(&mut rng, dm, df))
        .collect();

    let mut cpu_x = x.clone();
    for l in &layers {
        cpu_one_block(&mut cpu_x, l, sl, dm, dh, df);
    }

    let gpu_out = run_megakernel(&config, &x, &layers);
    let max_err: f32 = gpu_out
        .iter()
        .zip(cpu_x.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    eprintln!("Four layer max error: {}", max_err);
    assert!(max_err < 0.01, "Four layer error: {}", max_err);
}
