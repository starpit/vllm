// Microbenchmark: cast-at-load vs cast-in-register for q4 affine decode (qmv).
//
// Two kernels with identical structure; only difference is whether scales/biases
// are stored bf16 (cast-at-load) or f16 (cast-in-register). Measures GPU-side
// time across multiple shapes.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDataType,
    MTLDevice, MTLFunctionConstantValues, MTLLibrary, MTLResourceOptions, MTLSize,
};

use std::ffi::c_void;
use std::ptr::NonNull;
use std::time::Instant;

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

const SHADER_SRC: &str = include_str!("../shaders/q4_cast.metal");

#[derive(Clone, Copy, Debug)]
struct Shape {
    name: &'static str,
    n: u32,
    k: u32,
    gs: u32,
}

fn make_pipelines(device: &Device, shape: Shape) -> (Pipeline, Pipeline, Library) {
    let opts = MTLCompileOptions::new();
    let src = NSString::from_str(SHADER_SRC);
    let library = device
        .newLibraryWithSource_options_error(&src, Some(&opts))
        .expect("compile shader source");

    // Function constants
    let k = shape.k;
    let gs = shape.gs;
    let k_over_8 = k / 8;
    let k_over_gs = k / gs;

    let constants = MTLFunctionConstantValues::new();
    unsafe {
        constants.setConstantValue_type_atIndex(
            NonNull::new(&k as *const u32 as *mut c_void).unwrap(),
            MTLDataType::UInt,
            0,
        );
        constants.setConstantValue_type_atIndex(
            NonNull::new(&gs as *const u32 as *mut c_void).unwrap(),
            MTLDataType::UInt,
            1,
        );
        constants.setConstantValue_type_atIndex(
            NonNull::new(&k_over_8 as *const u32 as *mut c_void).unwrap(),
            MTLDataType::UInt,
            2,
        );
        constants.setConstantValue_type_atIndex(
            NonNull::new(&k_over_gs as *const u32 as *mut c_void).unwrap(),
            MTLDataType::UInt,
            3,
        );
    }

    let mk_pipeline = |name: &str| -> Pipeline {
        let nm = NSString::from_str(name);
        let func = library
            .newFunctionWithName_constantValues_error(&nm, &constants)
            .unwrap_or_else(|e| panic!("function {name} with constants: {e:?}"));
        device
            .newComputePipelineStateWithFunction_error(&func)
            .unwrap_or_else(|e| panic!("pipeline {name}: {e:?}"))
    };

    let p_at_load = mk_pipeline("q4_qmv_cast_at_load");
    let p_in_reg = mk_pipeline("q4_qmv_cast_in_register");

    (p_at_load, p_in_reg, library)
}

// Deterministic linear congruential RNG for reproducibility.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1))
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        // Uniform [-1, 1)
        ((self.next_u32() as f64) / (u32::MAX as f64) * 2.0 - 1.0) as f32
    }
}

fn alloc_buffer(device: &Device, bytes: usize) -> Buffer {
    device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("alloc buffer")
}

/// MTLBuffer backed by `StorageModeShared` is mapped CPU↔GPU
/// shared memory; the Rust borrow checker can't model the cross-
/// boundary aliasing meaningfully. Callers must serialize CPU
/// writes against GPU dispatches themselves.
#[allow(clippy::mut_from_ref)]
unsafe fn buffer_as_slice<T>(buf: &Buffer, count: usize) -> &mut [T] {
    let ptr = buf.contents().as_ptr() as *mut T;
    unsafe { std::slice::from_raw_parts_mut(ptr, count) }
}

// Bit conversions
fn f16_bits(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}
fn bf16_bits(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}
fn from_f16_bits(b: u16) -> f32 {
    half::f16::from_bits(b).to_f32()
}
fn from_bf16_bits(b: u16) -> f32 {
    half::bf16::from_bits(b).to_f32()
}

fn dispatch_one(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &Pipeline,
    n: u32,
    bufs: &[&Buffer],
) {
    encoder.setComputePipelineState(pipeline);
    for (i, b) in bufs.iter().enumerate() {
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(b), 0, i);
        }
    }
    let grid = MTLSize {
        width: n as usize,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

fn run_shape(
    device: &Device,
    queue: &Queue,
    shape: Shape,
    iters: u32,
    repeats: u32,
) -> ShapeResult {
    let n = shape.n;
    let k = shape.k;
    let gs = shape.gs;
    let n_groups = (n * k / gs) as usize;
    let n_packed = (n * k / 8) as usize;
    let n_acts = k as usize;
    let n_out = n as usize;

    // ---- Buffers ----
    let buf_w = alloc_buffer(device, n_packed * 4);
    let buf_s_bf16 = alloc_buffer(device, n_groups * 2);
    let buf_b_bf16 = alloc_buffer(device, n_groups * 2);
    let buf_s_f16 = alloc_buffer(device, n_groups * 2);
    let buf_b_f16 = alloc_buffer(device, n_groups * 2);
    let buf_x = alloc_buffer(device, n_acts * 2);
    let buf_y_a = alloc_buffer(device, n_out * 2);
    let buf_y_b = alloc_buffer(device, n_out * 2);

    // ---- Populate with deterministic data ----
    let mut rng = Lcg::new(0xDEADBEEFCAFEBABE);
    unsafe {
        let w_slice: &mut [u32] = buffer_as_slice(&buf_w, n_packed);
        for v in w_slice.iter_mut() {
            *v = rng.next_u32();
        }

        let s_bf: &mut [u16] = buffer_as_slice(&buf_s_bf16, n_groups);
        let s_f: &mut [u16] = buffer_as_slice(&buf_s_f16, n_groups);
        let b_bf: &mut [u16] = buffer_as_slice(&buf_b_bf16, n_groups);
        let b_f: &mut [u16] = buffer_as_slice(&buf_b_f16, n_groups);
        for i in 0..n_groups {
            // Scales: small positive values (typical q4 scale magnitude)
            let scale_logical = (rng.next_f32().abs() + 0.001) * 0.1;
            // Biases: signed small values
            let bias_logical = rng.next_f32() * 0.1;
            s_bf[i] = bf16_bits(scale_logical);
            s_f[i] = f16_bits(scale_logical);
            b_bf[i] = bf16_bits(bias_logical);
            b_f[i] = f16_bits(bias_logical);
        }

        let x: &mut [u16] = buffer_as_slice(&buf_x, n_acts);
        for v in x.iter_mut() {
            *v = bf16_bits(rng.next_f32());
        }
    }

    // ---- Pipelines ----
    let (p_at_load, p_in_reg, _lib) = make_pipelines(device, shape);

    // ---- Correctness check: dispatch each once, compare to CPU reference ----
    let cb = queue.commandBuffer().unwrap();
    let enc = cb.computeCommandEncoder().unwrap();
    dispatch_one(
        &enc,
        &p_at_load,
        n,
        &[&buf_w, &buf_s_bf16, &buf_b_bf16, &buf_x, &buf_y_a],
    );
    dispatch_one(
        &enc,
        &p_in_reg,
        n,
        &[&buf_w, &buf_s_f16, &buf_b_f16, &buf_x, &buf_y_b],
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();

    // CPU reference for output[0]
    let cpu_ref = |s_buf: &Buffer, b_buf: &Buffer, decode: fn(u16) -> f32| -> f32 {
        unsafe {
            let w: &[u32] = buffer_as_slice(&buf_w, n_packed);
            let s: &[u16] = buffer_as_slice(s_buf, n_groups);
            let bb: &[u16] = buffer_as_slice(b_buf, n_groups);
            let x: &[u16] = buffer_as_slice(&buf_x, n_acts);
            let mut acc = 0.0_f32;
            let row = 0;
            let k_over_gs = (k / gs) as usize;
            for group_idx in 0..k_over_gs {
                let scale = decode(s[row * k_over_gs + group_idx]);
                let bias = decode(bb[row * k_over_gs + group_idx]);
                let k_base = group_idx * gs as usize;
                let w_base = row * (k as usize / 8) + k_base / 8;
                for pi in 0..(gs / 8) as usize {
                    let packed = w[w_base + pi];
                    let k0 = k_base + pi * 8;
                    for q in 0..8usize {
                        let nib = ((packed >> (q * 4)) & 0xF) as f32;
                        let wd = nib * scale + bias;
                        acc += from_bf16_bits(x[k0 + q]) * wd;
                    }
                }
            }
            acc
        }
    };

    let cpu_a = cpu_ref(&buf_s_bf16, &buf_b_bf16, from_bf16_bits);
    let cpu_b = cpu_ref(&buf_s_f16, &buf_b_f16, from_f16_bits);

    let gpu_a = unsafe {
        let y: &[u16] = buffer_as_slice(&buf_y_a, n_out);
        from_bf16_bits(y[0])
    };
    let gpu_b = unsafe {
        let y: &[u16] = buffer_as_slice(&buf_y_b, n_out);
        from_bf16_bits(y[0])
    };

    let pass_a = (cpu_a - gpu_a).abs() / cpu_a.abs().max(1e-6) < 0.05;
    let pass_b = (cpu_b - gpu_b).abs() / cpu_b.abs().max(1e-6) < 0.05;
    let correct = pass_a && pass_b;

    if !correct {
        eprintln!(
            "  [{}] CORRECTNESS FAIL: cpu_a={:.4} gpu_a={:.4} (Δ={:.2}%)  cpu_b={:.4} gpu_b={:.4} (Δ={:.2}%)",
            shape.name, cpu_a, gpu_a,
            (cpu_a - gpu_a).abs() / cpu_a.abs().max(1e-6) * 100.0,
            cpu_b, gpu_b,
            (cpu_b - gpu_b).abs() / cpu_b.abs().max(1e-6) * 100.0,
        );
    }

    // ---- Warmup ----
    let warmup = 100;
    for _ in 0..2 {
        let cb = queue.commandBuffer().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        for _ in 0..warmup {
            dispatch_one(
                &enc,
                &p_at_load,
                n,
                &[&buf_w, &buf_s_bf16, &buf_b_bf16, &buf_x, &buf_y_a],
            );
            dispatch_one(
                &enc,
                &p_in_reg,
                n,
                &[&buf_w, &buf_s_f16, &buf_b_f16, &buf_x, &buf_y_b],
            );
        }
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
    }

    // ---- Timed runs ----
    let mut times_a: Vec<f64> = Vec::with_capacity(repeats as usize);
    let mut times_b: Vec<f64> = Vec::with_capacity(repeats as usize);

    for _rep in 0..repeats {
        // Variant A
        let cb = queue.commandBuffer().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        for _ in 0..iters {
            dispatch_one(
                &enc,
                &p_at_load,
                n,
                &[&buf_w, &buf_s_bf16, &buf_b_bf16, &buf_x, &buf_y_a],
            );
        }
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        let dt_a = cb.GPUEndTime() - cb.GPUStartTime();
        times_a.push(dt_a / iters as f64);

        // Variant B
        let cb = queue.commandBuffer().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        for _ in 0..iters {
            dispatch_one(
                &enc,
                &p_in_reg,
                n,
                &[&buf_w, &buf_s_f16, &buf_b_f16, &buf_x, &buf_y_b],
            );
        }
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        let dt_b = cb.GPUEndTime() - cb.GPUStartTime();
        times_b.push(dt_b / iters as f64);
    }

    let mean_a = times_a.iter().sum::<f64>() / repeats as f64;
    let mean_b = times_b.iter().sum::<f64>() / repeats as f64;
    let var_a = times_a.iter().map(|t| (t - mean_a).powi(2)).sum::<f64>() / repeats as f64;
    let var_b = times_b.iter().map(|t| (t - mean_b).powi(2)).sum::<f64>() / repeats as f64;

    ShapeResult {
        mean_a_ns: mean_a * 1e9,
        mean_b_ns: mean_b * 1e9,
        std_a_ns: var_a.sqrt() * 1e9,
        std_b_ns: var_b.sqrt() * 1e9,
        correct,
    }
}

struct ShapeResult {
    mean_a_ns: f64,
    mean_b_ns: f64,
    std_a_ns: f64,
    std_b_ns: f64,
    correct: bool,
}

fn main() {
    let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
    let queue = device.newCommandQueue().expect("no command queue");

    println!("Device: {}", device.name());
    println!();

    let shapes = [
        Shape {
            name: "q_proj  N=2048 K=2048",
            n: 2048,
            k: 2048,
            gs: 64,
        },
        Shape {
            name: "down_proj N=2048 K=8192",
            n: 2048,
            k: 8192,
            gs: 64,
        },
        Shape {
            name: "gate_proj N=8192 K=2048",
            n: 8192,
            k: 2048,
            gs: 64,
        },
    ];

    let iters = 10000u32; // dispatches per command buffer
    let repeats = 5u32;

    println!("Shape                     |  cast-at-load (ns)  |  cast-in-reg (ns)   |  delta   |  BW(A) GB/s | correct");
    println!("--------------------------+---------------------+---------------------+----------+-------------+--------");

    for shape in shapes.iter() {
        let t0 = Instant::now();
        let r = run_shape(&device, &queue, *shape, iters, repeats);
        let elapsed = t0.elapsed().as_secs_f32();

        // Theoretical bandwidth: per kernel call, we read N*K/8*4 bytes (weights)
        // + 2*N*K/gs*2 bytes (scales+biases bf16) + K*2 bytes (activations)
        // + write N*2 bytes (output)
        let bytes_w = (shape.n * shape.k / 8 * 4) as f64;
        let bytes_sb = (2 * shape.n * shape.k / shape.gs * 2) as f64;
        let bytes_x = (shape.k * 2) as f64;
        let bytes_y = (shape.n * 2) as f64;
        let total_bytes = bytes_w + bytes_sb + bytes_x + bytes_y;
        let bw_gbs_a = total_bytes / r.mean_a_ns; // bytes / ns = GB/s

        let delta = (r.mean_b_ns - r.mean_a_ns) / r.mean_a_ns * 100.0;

        println!(
            "{:25} | {:8.1} ± {:6.1}     | {:8.1} ± {:6.1}     | {:+6.2}%  | {:9.1}   |  {}",
            shape.name,
            r.mean_a_ns,
            r.std_a_ns,
            r.mean_b_ns,
            r.std_b_ns,
            delta,
            bw_gbs_a,
            if r.correct { "PASS" } else { "FAIL" },
        );
        let _ = elapsed;
    }
}
