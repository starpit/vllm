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

    let nl = config.num_layers as usize;
    let bs = config.block_size as usize;
    let mb = config.max_blocks as usize;
    let dh = config.d_head as usize;

    // KV cache: [num_layers, 2(K/V), max_blocks, block_size, d_head]
    let kv_cache_size = nl * 2 * mb * bs * dh;

    // Block table: identity mapping (block i → physical block i)
    let block_table: Vec<i32> = (0..mb as i32).collect();
    let block_table_buf = device.new_buffer_with_data(
        block_table.as_ptr() as *const c_void,
        (mb * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Cache len = 0 (first forward pass, no cached tokens)
    let cache_len: u32 = 0;
    let cache_len_buf = device.new_buffer_with_data(
        &cache_len as *const u32 as *const c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let bufs: Vec<Buffer> = vec![
        make_f32(&device, x),              // 0: x
        empty_f32(&device, tot),           // 1: out
        make_f32(&device, &all_gamma),     // 2: all_gamma
        make_f16(&device, &all_weights),   // 3: all_weights
        empty_f32(&device, tot),           // 4: h
        empty_f16(&device, tot),           // 5: h_f16
        empty_f32(&device, 3 * tot),       // 6: qkv
        empty_f16(&device, 3 * tot),       // 7: qkv_f16
        empty_f32(&device, tot),           // 8: attn_out
        empty_f16(&device, tot),           // 9: attn_f16
        empty_f32(&device, tot),           // 10: o
        empty_f32(&device, tot),           // 11: h_ffn
        empty_f16(&device, tot),           // 12: h_ffn_f16
        empty_f32(&device, sl * df),       // 13: gate_out
        empty_f32(&device, sl * df),       // 14: up_out
        empty_f16(&device, sl * df),       // 15: ffn_act
        empty_f32(&device, tot),           // 16: down_out
        empty_f16(&device, kv_cache_size), // 17: kv_cache
        block_table_buf,                   // 18: block_table
        cache_len_buf,                     // 19: cache_len_ptr
        empty_f32(&device, 1),             // 20: phase_counter
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

// ═══════════════════════════════════════════════════════════════════
// Paged KV cache tests
// ═══════════════════════════════════════════════════════════════════

/// Run megakernel with explicit KV cache state and block table.
/// Returns (output, kv_cache_buf) so cache can be reused for next step.
fn run_megakernel_with_cache(
    config: &MegakernelConfig,
    x: &[f32],
    layers: &[LayerWeights],
    kv_cache_init: Option<&[f16]>,
    block_table: &[i32],
    cache_len: u32,
) -> (Vec<f32>, Vec<f16>) {
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
    let dh = config.d_head as usize;
    let nl = config.num_layers as usize;
    let bs = config.block_size as usize;
    let mb = config.max_blocks as usize;
    let tot = sl * dm;
    let kv_cache_elems = nl * 2 * mb * bs * dh;

    let all_gamma = pack_gammas(layers);
    let all_weights = pack_weights(layers);

    let kv_buf = if let Some(init) = kv_cache_init {
        make_f16(&device, init)
    } else {
        empty_f16(&device, kv_cache_elems)
    };
    let bt_buf = device.new_buffer_with_data(
        block_table.as_ptr() as *const c_void,
        (block_table.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cl_buf = device.new_buffer_with_data(
        &cache_len as *const u32 as *const c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let bufs: Vec<Buffer> = vec![
        make_f32(&device, x),
        empty_f32(&device, tot),
        make_f32(&device, &all_gamma),
        make_f16(&device, &all_weights),
        empty_f32(&device, tot),
        empty_f16(&device, tot),
        empty_f32(&device, 3 * tot),
        empty_f16(&device, 3 * tot),
        empty_f32(&device, tot),
        empty_f16(&device, tot),
        empty_f32(&device, tot),
        empty_f32(&device, tot),
        empty_f16(&device, tot),
        empty_f32(&device, sl * df),
        empty_f32(&device, sl * df),
        empty_f16(&device, sl * df),
        empty_f32(&device, tot),
        kv_buf,
        bt_buf,
        cl_buf,
        empty_f32(&device, 1),
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

    let out = read_f32(&bufs[1], tot);
    // Read back KV cache
    let kv_ptr = bufs[17].contents() as *const f16;
    let kv_out = unsafe { std::slice::from_raw_parts(kv_ptr, kv_cache_elems) }.to_vec();
    (out, kv_out)
}

#[test]
fn test_paged_attn_shuffled_block_table() {
    // Non-identity block table: physical blocks are shuffled.
    // Result should be identical to identity mapping (same data, different physical layout).
    let config = MegakernelConfig::test_config();
    let mut rng = 99u64;
    let sl = config.seq_len as usize;
    let dm = config.d_model as usize;
    let df = config.d_ffn as usize;
    let mb = config.max_blocks as usize;

    let next = |r: &mut u64| -> f32 {
        *r = r.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*r >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    };
    let x: Vec<f32> = (0..sl * dm).map(|_| next(&mut rng) * 0.5).collect();
    let layers = vec![random_layer_weights(&mut rng, dm, df)];

    // Identity block table
    let bt_identity: Vec<i32> = (0..mb as i32).collect();
    let (out_id, _) = run_megakernel_with_cache(&config, &x, &layers, None, &bt_identity, 0);

    // Shuffled block table: [2, 0, 3, 1]
    let bt_shuffled: Vec<i32> = vec![2, 0, 3, 1];
    let (out_shuf, _) = run_megakernel_with_cache(&config, &x, &layers, None, &bt_shuffled, 0);

    let err: f32 = out_id
        .iter()
        .zip(out_shuf.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("Shuffled block table error: {}", err);
    assert!(
        err < 0.01,
        "Shuffled block table should match identity: err={}",
        err
    );
}

#[test]
fn test_paged_attn_partial_block() {
    // cache_len = 2 (not a multiple of block_size=4), seq_len=1 (decode)
    // This tests that partial blocks work correctly.
    let mut config = MegakernelConfig::test_config();
    config.seq_len = 1; // decode: one new token
    let dm = config.d_model as usize;
    let df = config.d_ffn as usize;
    let dh = config.d_head as usize;
    let nl = config.num_layers as usize;
    let bs = config.block_size as usize;
    let mb = config.max_blocks as usize;

    let mut rng = 77u64;
    let next = |r: &mut u64| -> f32 {
        *r = r.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*r >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    };
    let x: Vec<f32> = (0..dm).map(|_| next(&mut rng) * 0.5).collect();
    let layers = vec![random_layer_weights(&mut rng, dm, df)];

    // Pre-populate cache with 2 tokens of K/V data
    let kv_cache_elems = nl * 2 * mb * bs * dh;
    let mut kv_init = vec![f16::from_f32(0.0); kv_cache_elems];
    // Write 2 cached tokens to block 0 (identity block table)
    for tok in 0..2u32 {
        for d in 0..dh {
            let val = f16::from_f32(next(&mut rng) * 0.3);
            // K: layer=0, kv=0, block=0, pos=tok, dim=d
            let k_idx =
                0 * (2 * mb * bs * dh) + 0 * (mb * bs * dh) + 0 * (bs * dh) + tok as usize * dh + d;
            kv_init[k_idx] = val;
            // V: layer=0, kv=1
            let v_idx =
                0 * (2 * mb * bs * dh) + 1 * (mb * bs * dh) + 0 * (bs * dh) + tok as usize * dh + d;
            kv_init[v_idx] = f16::from_f32(next(&mut rng) * 0.3);
        }
    }

    let bt: Vec<i32> = (0..mb as i32).collect();
    let (out, _kv_after) = run_megakernel_with_cache(
        &config,
        &x,
        &layers,
        Some(&kv_init),
        &bt,
        2, // cache_len=2
    );

    // Basic sanity: output should be finite and non-zero
    assert!(out.iter().all(|v| v.is_finite()), "Output has NaN/Inf");
    let nonzero = out.iter().filter(|v| v.abs() > 1e-6).count();
    eprintln!(
        "Partial block decode: {} nonzero out of {}",
        nonzero,
        out.len()
    );
    assert!(nonzero > 0, "Output all zeros");
}

#[test]
fn test_paged_attn_prefill_then_decode() {
    // Two-step inference: prefill 4 tokens, then decode 1 token.
    // The decode step should attend over all 5 tokens (4 cached + 1 new).
    let config_prefill = MegakernelConfig::test_config(); // seq_len=4
    let dm = config_prefill.d_model as usize;
    let df = config_prefill.d_ffn as usize;
    let mb = config_prefill.max_blocks as usize;

    let mut rng = 55u64;
    let next = |r: &mut u64| -> f32 {
        *r = r.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*r >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    };
    let x_prefill: Vec<f32> = (0..4 * dm).map(|_| next(&mut rng) * 0.5).collect();
    let layers = vec![random_layer_weights(&mut rng, dm, df)];
    let bt: Vec<i32> = (0..mb as i32).collect();

    // Step 1: Prefill (cache_len=0, seq_len=4)
    let (_prefill_out, kv_after_prefill) =
        run_megakernel_with_cache(&config_prefill, &x_prefill, &layers, None, &bt, 0);

    // Step 2: Decode (cache_len=4, seq_len=1)
    let mut config_decode = config_prefill.clone();
    config_decode.seq_len = 1;
    let x_decode: Vec<f32> = (0..dm).map(|_| next(&mut rng) * 0.5).collect();

    let (decode_out, _kv_after_decode) = run_megakernel_with_cache(
        &config_decode,
        &x_decode,
        &layers,
        Some(&kv_after_prefill),
        &bt,
        4,
    );

    // Basic sanity: output should be finite and non-zero
    assert!(
        decode_out.iter().all(|v| v.is_finite()),
        "Decode output has NaN/Inf"
    );
    let nonzero = decode_out.iter().filter(|v| v.abs() > 1e-6).count();
    eprintln!(
        "Prefill→Decode: {} nonzero out of {}",
        nonzero,
        decode_out.len()
    );
    assert!(nonzero > 0, "Decode output all zeros");

    // The decode output should differ from prefill (different input, different attention pattern)
    // Just verify it's not identical to the last prefill row
    eprintln!("Decode output: {:?}", &decode_out);
}
