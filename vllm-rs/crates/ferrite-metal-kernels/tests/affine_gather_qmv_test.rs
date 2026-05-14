// SPDX-License-Identifier: Apache-2.0
//
// `affine_gather_qmv_fast_*` parity test: per-(token,slot) output ≡
// CPU-dequantize-then-matmul reference for the rhs-gathered expert.
//
// Validates the MoE decode path of `Qwen3NextSparseMoeBlock` /
// `Qwen3MoeSparseMoeBlock` — for each (token, expert_slot), the
// kernel must produce `x[token] @ W[rhs_indices[slot_row]].T`.
//
// Shapes mirror Qwen3-MoE 4bit decode: K_in = 2048, N_out = 2048,
// num_experts = 8 (small subset to keep the test cheap), top_k = 4.

use std::ffi::c_void;
use std::ptr::NonNull;

use ferrite_metal_kernels::cpu_reference::affine_qmm_t_b4_bf16;
use ferrite_metal_kernels::device::detect_device;
use ferrite_metal_kernels::quantized::{
    DequantDtype, MetalAffineGatherQmv, ScaleDtype,
};
use ferrite_metal_kernels::stream::MetalStream;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLDevice, MTLResourceOptions};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Device = Retained<ProtocolObject<dyn MTLDevice>>;

fn buffer_from_bytes(device: &Device, bytes: &[u8]) -> Buffer {
    unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                bytes.len().max(1),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("newBufferWithBytes returned nil")
    }
}

fn zeroed_buffer(device: &Device, n_bytes: usize) -> Buffer {
    device
        .newBufferWithLength_options(n_bytes.max(1), MTLResourceOptions::StorageModeShared)
        .expect("newBufferWithLength returned nil")
}

fn read_buffer_bf16(buf: &Buffer, n_elements: usize) -> Vec<half::bf16> {
    let ptr = buf.contents().as_ptr() as *const half::bf16;
    unsafe { std::slice::from_raw_parts(ptr, n_elements) }.to_vec()
}

struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_byte(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
    fn next_unit_f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// End-to-end parity helper covering both `_fast` and generic
/// gather-qmv variants (the dispatcher in `MetalAffineGatherQmv`
/// picks based on `K_in % 512`).
fn run_gather_qmv_parity(
    seed: u64,
    num_experts: usize,
    top_k: usize,
    n_tokens: usize,
    k_in: usize,
    n_out: usize,
    group_size: usize,
) {
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let mut stream = MetalStream::new(&device);

    let bytes_per_expert = n_out * k_in / 2;
    let groups_per_expert = n_out * k_in / group_size;
    let total_packed_bytes = num_experts * bytes_per_expert;
    let total_groups = num_experts * groups_per_expert;

    let mut rng = SplitMix64(seed);
    let packed: Vec<u8> = (0..total_packed_bytes).map(|_| rng.next_byte()).collect();
    let scales: Vec<half::f16> = (0..total_groups)
        .map(|_| half::f16::from_f32(0.01 + 0.04 * rng.next_unit_f32()))
        .collect();
    let biases: Vec<half::f16> = (0..total_groups)
        .map(|_| half::f16::from_f32(rng.next_unit_f32() - 0.5))
        .collect();
    let x: Vec<half::bf16> = (0..(n_tokens * k_in))
        .map(|_| half::bf16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
        .collect();
    let rhs_indices: Vec<u32> = (0..(n_tokens * top_k))
        .map(|i| (i as u32 * 5 + 1) % num_experts as u32)
        .collect();

    let packed_buf = buffer_from_bytes(&device, &packed);
    let scales_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            scales.as_ptr() as *const u8,
            std::mem::size_of_val(&scales[..]),
        )
    };
    let biases_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            biases.as_ptr() as *const u8,
            std::mem::size_of_val(&biases[..]),
        )
    };
    let x_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, std::mem::size_of_val(&x[..])) };
    let idx_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            rhs_indices.as_ptr() as *const u8,
            std::mem::size_of_val(&rhs_indices[..]),
        )
    };
    let scales_buf = buffer_from_bytes(&device, scales_bytes);
    let biases_buf = buffer_from_bytes(&device, biases_bytes);
    let x_buf = buffer_from_bytes(&device, x_bytes);
    let idx_buf = buffer_from_bytes(&device, idx_bytes);
    let n_rows = n_tokens * top_k;
    let y_buf = zeroed_buffer(&device, n_rows * n_out * std::mem::size_of::<half::bf16>());

    let gather = MetalAffineGatherQmv::new(device.clone()).expect("MetalAffineGatherQmv");

    let cmd_buf = stream.get_command_buffer().expect("command buffer").clone();
    let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
    let result = gather.execute(
        &x_buf,
        &packed_buf,
        &scales_buf,
        &biases_buf,
        &idx_buf,
        &y_buf,
        n_tokens as u32,
        top_k as u32,
        n_out as u32,
        k_in as u32,
        group_size as u32,
        4,
        DequantDtype::Bf16,
        ScaleDtype::F16,
        &encoder,
    );
    encoder.endEncoding();
    result.expect("gather qmv dispatch");
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    let got = read_buffer_bf16(&y_buf, n_rows * n_out);

    for r in 0..n_rows {
        let token = r / top_k;
        let expert = rhs_indices[r] as usize;
        let packed_e = &packed[expert * bytes_per_expert..(expert + 1) * bytes_per_expert];
        let scales_e = &scales[expert * groups_per_expert..(expert + 1) * groups_per_expert];
        let biases_e = &biases[expert * groups_per_expert..(expert + 1) * groups_per_expert];
        let x_t = &x[token * k_in..(token + 1) * k_in];
        let expected =
            affine_qmm_t_b4_bf16(packed_e, scales_e, biases_e, x_t, 1, n_out, k_in, group_size);
        let got_row = &got[r * n_out..(r + 1) * n_out];

        let bf16_eps: f32 = 1.0 / 128.0;
        let sum_noise = (k_in as f32).sqrt() * 0.5 * bf16_eps * 0.5;
        let safety = 4.0;
        for j in 0..n_out {
            let mf = got_row[j].to_f32();
            let ef = expected[j].to_f32();
            let abs_err = (mf - ef).abs();
            let allowed = safety * (sum_noise + ef.abs() * bf16_eps);
            assert!(
                abs_err <= allowed,
                "row={r} (token={token}, expert={expert}) j={j}: \
                 got={mf} cpu={ef} abs_err={abs_err} allowed={allowed} \
                 (K={k_in} N={n_out} gs={group_size})",
            );
        }
    }
}

#[test]
fn affine_gather_qmv_fast_bf16_decode_parity() {
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let mut stream = MetalStream::new(&device);

    let num_experts: usize = 8;
    let top_k: usize = 4;
    let n_tokens: usize = 3;
    let k_in: usize = 512; // qmv_fast requires K%512==0
    let n_out: usize = 64; // qmv_fast requires N%8==0
    let group_size: usize = 64;

    // Per-expert weights packed bits=4: n_out × k_in nibbles per expert.
    let bytes_per_expert = n_out * k_in / 2; // 2 nibbles / byte
    let groups_per_expert = n_out * k_in / group_size;
    let total_packed_bytes = num_experts * bytes_per_expert;
    let total_groups = num_experts * groups_per_expert;

    let mut rng = SplitMix64(0xC0FFEE);

    let packed: Vec<u8> = (0..total_packed_bytes).map(|_| rng.next_byte()).collect();
    let scales: Vec<half::f16> = (0..total_groups)
        .map(|_| half::f16::from_f32(0.01 + 0.04 * rng.next_unit_f32()))
        .collect();
    let biases: Vec<half::f16> = (0..total_groups)
        .map(|_| half::f16::from_f32(rng.next_unit_f32() - 0.5))
        .collect();
    let x: Vec<half::bf16> = (0..(n_tokens * k_in))
        .map(|_| half::bf16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
        .collect();

    // Assign experts: each (token, slot) hits a distinct expert from
    // the set, with reuse across tokens so we exercise the gather.
    let rhs_indices: Vec<u32> = (0..(n_tokens * top_k))
        .map(|i| (i as u32 * 5 + 1) % num_experts as u32)
        .collect();

    // Buffers
    let packed_buf = buffer_from_bytes(&device, &packed);
    let scales_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const u8, std::mem::size_of_val(&scales[..]))
    };
    let biases_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(biases.as_ptr() as *const u8, std::mem::size_of_val(&biases[..]))
    };
    let x_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(x.as_ptr() as *const u8, std::mem::size_of_val(&x[..]))
    };
    let idx_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            rhs_indices.as_ptr() as *const u8,
            std::mem::size_of_val(&rhs_indices[..]),
        )
    };
    let scales_buf = buffer_from_bytes(&device, scales_bytes);
    let biases_buf = buffer_from_bytes(&device, biases_bytes);
    let x_buf = buffer_from_bytes(&device, x_bytes);
    let idx_buf = buffer_from_bytes(&device, idx_bytes);
    let n_rows = n_tokens * top_k;
    let y_buf = zeroed_buffer(&device, n_rows * n_out * std::mem::size_of::<half::bf16>());

    let gather = MetalAffineGatherQmv::new(device.clone()).expect("MetalAffineGatherQmv");

    let cmd_buf = stream.get_command_buffer().expect("command buffer").clone();
    let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
    let dispatch_result = gather.execute(
        &x_buf,
        &packed_buf,
        &scales_buf,
        &biases_buf,
        &idx_buf,
        &y_buf,
        n_tokens as u32,
        top_k as u32,
        n_out as u32,
        k_in as u32,
        group_size as u32,
        4,
        DequantDtype::Bf16,
        ScaleDtype::F16,
        &encoder,
    );
    encoder.endEncoding();
    dispatch_result.expect("gather qmv dispatch");
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    let got = read_buffer_bf16(&y_buf, n_rows * n_out);

    // CPU expected: for each row r = (token, slot), expert =
    // rhs_indices[r], y[r] = qmm_t_b4(packed[expert..], scales[..],
    // biases[..], x[token..], 1, n_out, k_in).
    for r in 0..n_rows {
        let token = r / top_k;
        let expert = rhs_indices[r] as usize;
        let packed_e = &packed[expert * bytes_per_expert..(expert + 1) * bytes_per_expert];
        let scales_e = &scales[expert * groups_per_expert..(expert + 1) * groups_per_expert];
        let biases_e = &biases[expert * groups_per_expert..(expert + 1) * groups_per_expert];
        let x_t = &x[token * k_in..(token + 1) * k_in];
        let expected = affine_qmm_t_b4_bf16(
            packed_e, scales_e, biases_e, x_t,
            1, n_out, k_in, group_size,
        );
        let got_row = &got[r * n_out..(r + 1) * n_out];

        // bf16 noise floor: sqrt(K) * eps * |x|*|w| with eps≈2^-7,
        // |x|≲1, |w|≲1.25 → per-element absolute tol ~ 0.5 * sqrt(K) * eps.
        let bf16_eps: f32 = 1.0 / 128.0;
        let sum_noise = (k_in as f32).sqrt() * 0.5 * bf16_eps * 0.5;
        let safety = 4.0;
        for j in 0..n_out {
            let mf = got_row[j].to_f32();
            let ef = expected[j].to_f32();
            let abs_err = (mf - ef).abs();
            let allowed = safety * (sum_noise + ef.abs() * bf16_eps);
            assert!(
                abs_err <= allowed,
                "row={r} (token={token}, slot={s}, expert={expert}) j={j}: \
                 got={mf} cpu={ef} abs_err={abs_err} allowed={allowed}",
                s = r % top_k,
            );
        }
    }
}

/// Qwen3-MoE-30B-A3B-Instruct-4bit `down_proj` shape: K=768, N=2048,
/// gs=64. K%512 != 0 — dispatcher must pick generic `affine_gather_qmv`.
/// Shrunk: 4 experts, 4 slots, 2 tokens to keep test cheap.
#[test]
fn affine_gather_qmv_generic_bf16_down_proj_shape() {
    run_gather_qmv_parity(
        /*seed*/ 0xDEAD_BEEF,
        /*num_experts*/ 4,
        /*top_k*/ 4,
        /*n_tokens*/ 2,
        /*k_in*/ 768,
        /*n_out*/ 2048,
        /*group_size*/ 64,
    );
}

/// Qwen3-MoE-30B-A3B-Instruct-4bit `gate_proj` / `up_proj` shape:
/// K=2048, N=768. K%512==0 — dispatcher picks the `_fast` variant.
#[test]
fn affine_gather_qmv_fast_bf16_gate_up_proj_shape() {
    run_gather_qmv_parity(
        0xFEED_FACE,
        4,
        4,
        2,
        2048,
        768,
        64,
    );
}
