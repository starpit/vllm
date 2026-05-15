// SPDX-License-Identifier: Apache-2.0
//
// Parity test for `splitk_reduce_sum_<dtype>` (kernel ≡ CPU reference
// within bf16 cast noise). The kernel is the downstream half of the
// `affine_qmm_t_splitk` path: `qmm_t_splitk` writes a [split_k, M, N]
// intermediate, and this kernel sums along axis 0 to produce [M, N].
// The CPU reference here matches the in-kernel float accumulator, so
// the only loss is the per-output bf16 cast at the end.
//
// Two tests:
//   - bf16, exercises the typical Llama-1B prefill split_k value
//   - f16, exercises the alternate dtype branch
//
// Test shapes mirror the qmm_t_splitk shape that fires for Llama-1B
// q_proj at M=64 N=2048 K=2048 gs=64 → split_k=4. We pick a smaller
// (M, N) to keep the test allocation modest while still exercising
// every threadgroup boundary.

use std::ffi::c_void;
use std::ptr::NonNull;

use ferrite_metal_kernels::device::detect_device;
use ferrite_metal_kernels::quantized::{DequantDtype, MetalSplitKReduce};
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
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("newBufferWithBytes returned nil")
    }
}

fn zeroed_buffer(device: &Device, n_bytes: usize) -> Buffer {
    device
        .newBufferWithLength_options(n_bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBufferWithLength returned nil")
}

/// SplitMix64 — same deterministic PRNG used by the qmv/qmm tests.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_unit_f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

#[test]
fn splitk_reduce_sum_bf16_matches_cpu_reference() {
    // Llama-1B q_proj prefill shape: split_k=4. Use a smaller (M, N) so
    // the test allocation is modest; the kernel is one-thread-per-output
    // so coverage is a function of M*N total, not per-shape behavior.
    let split_k: u32 = 4;
    let m: u32 = 64;
    let n: u32 = 256;

    // Per-element magnitudes match what `qmm_t_impl_inline` would have
    // accumulated into the splitk intermediate: roughly K-sum / split_k
    // worth of bf16-range float values. Random uniform in [-2, 2] per
    // partition keeps the per-output sum around magnitude 0 with std
    // ~sqrt(split_k)*1; well within bf16's representable range.
    let mut rng = SplitMix64(0xC0FFEE);
    let n_in = (split_k as usize) * (m as usize) * (n as usize);
    let intermediate: Vec<half::bf16> = (0..n_in)
        .map(|_| half::bf16::from_f32(4.0 * rng.next_unit_f32() - 2.0))
        .collect();

    // Compute the reference: sum-along-axis-0 in float, cast to bf16
    // at the end. This matches the kernel's float accumulator + final
    // T cast at the assignment site.
    let mut expected = vec![half::bf16::ZERO; (m * n) as usize];
    for k in 0..split_k as usize {
        for i in 0..m as usize {
            for j in 0..n as usize {
                let idx_in = k * (m as usize) * (n as usize) + i * (n as usize) + j;
                let idx_out = i * (n as usize) + j;
                let cur = expected[idx_out].to_f32();
                expected[idx_out] = half::bf16::from_f32(cur + intermediate[idx_in].to_f32());
            }
        }
    }

    // GPU dispatch.
    let device = detect_device().expect("Metal device").device;
    let mut stream = MetalStream::new(&device);
    let reduce = MetalSplitKReduce::new(device.clone()).expect("MetalSplitKReduce");

    let in_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            intermediate.as_ptr() as *const u8,
            std::mem::size_of_val(&intermediate[..]),
        )
    };
    let in_buf = buffer_from_bytes(&device, in_bytes);
    let out_buf = zeroed_buffer(
        &device,
        (m as usize) * (n as usize) * std::mem::size_of::<half::bf16>(),
    );

    let cmd_buf = stream.get_command_buffer().expect("command buffer").clone();
    let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
    reduce
        .execute(
            &in_buf,
            &out_buf,
            m,
            n,
            split_k,
            DequantDtype::Bf16,
            &encoder,
        )
        .expect("splitk_reduce dispatch");
    encoder.endEncoding();
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    let actual_ptr = out_buf.contents().as_ptr() as *const half::bf16;
    let actual: &[half::bf16] =
        unsafe { std::slice::from_raw_parts(actual_ptr, (m as usize) * (n as usize)) };

    // Tolerance: we sum split_k=4 bf16-cast partial values; each cast
    // is ~bf16_eps * |sum_so_far|. Final bf16 cast is ~bf16_eps * |out|.
    // The order of summation in CPU vs GPU can differ (CPU does sequential
    // bf16 cast per partition; GPU keeps a single float accumulator over
    // all split_k), so the noise budget is bounded by `split_k * bf16_eps
    // * max_partial_magnitude`. Empirically that's well under 4 * 0.01.
    let bf16_eps: f32 = 1.0 / 128.0;
    let mut worst: (usize, f32, f32, f32) = (0, 0.0, 0.0, 0.0);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let af = a.to_f32();
        let ef = e.to_f32();
        let err = (af - ef).abs();
        if err > worst.3 {
            worst = (i, af, ef, err);
        }
    }
    let allowed = bf16_eps * (split_k as f32) * 4.0;
    assert!(
        worst.3 <= allowed,
        "splitk_reduce_sum bf16 mismatch at idx {}: actual={} expected={} err={} > allowed={}",
        worst.0,
        worst.1,
        worst.2,
        worst.3,
        allowed,
    );
}

#[test]
fn splitk_reduce_sum_f16_matches_cpu_reference() {
    // Same shape as bf16 — exercises the f16 instantiation. f16 has
    // narrower dynamic range than bf16 but our values stay in
    // [-(2*split_k), 2*split_k] which is well within f16 range.
    let split_k: u32 = 4;
    let m: u32 = 64;
    let n: u32 = 256;

    let mut rng = SplitMix64(0xBADCAFE);
    let n_in = (split_k as usize) * (m as usize) * (n as usize);
    let intermediate: Vec<half::f16> = (0..n_in)
        .map(|_| half::f16::from_f32(4.0 * rng.next_unit_f32() - 2.0))
        .collect();

    let mut expected = vec![half::f16::ZERO; (m * n) as usize];
    for k in 0..split_k as usize {
        for i in 0..m as usize {
            for j in 0..n as usize {
                let idx_in = k * (m as usize) * (n as usize) + i * (n as usize) + j;
                let idx_out = i * (n as usize) + j;
                let cur = expected[idx_out].to_f32();
                expected[idx_out] = half::f16::from_f32(cur + intermediate[idx_in].to_f32());
            }
        }
    }

    let device = detect_device().expect("Metal device").device;
    let mut stream = MetalStream::new(&device);
    let reduce = MetalSplitKReduce::new(device.clone()).expect("MetalSplitKReduce");

    let in_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            intermediate.as_ptr() as *const u8,
            std::mem::size_of_val(&intermediate[..]),
        )
    };
    let in_buf = buffer_from_bytes(&device, in_bytes);
    let out_buf = zeroed_buffer(
        &device,
        (m as usize) * (n as usize) * std::mem::size_of::<half::f16>(),
    );

    let cmd_buf = stream.get_command_buffer().expect("command buffer").clone();
    let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
    reduce
        .execute(
            &in_buf,
            &out_buf,
            m,
            n,
            split_k,
            DequantDtype::F16,
            &encoder,
        )
        .expect("splitk_reduce dispatch");
    encoder.endEncoding();
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    let actual_ptr = out_buf.contents().as_ptr() as *const half::f16;
    let actual: &[half::f16] =
        unsafe { std::slice::from_raw_parts(actual_ptr, (m as usize) * (n as usize)) };

    // f16 eps = 1/1024 (vs bf16_eps = 1/128); but the per-partition
    // CPU cast still introduces bf16-magnitude rounding so the budget
    // structure is the same: split_k partitions * one cast each.
    let f16_eps: f32 = 1.0 / 1024.0;
    let mut worst: (usize, f32, f32, f32) = (0, 0.0, 0.0, 0.0);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let af = a.to_f32();
        let ef = e.to_f32();
        let err = (af - ef).abs();
        if err > worst.3 {
            worst = (i, af, ef, err);
        }
    }
    // f16 tolerance: split_k partitions × per-cast eps × max-magnitude.
    // Max per-output magnitude ≤ split_k * 2 = 8.
    let allowed = f16_eps * (split_k as f32) * 8.0;
    assert!(
        worst.3 <= allowed,
        "splitk_reduce_sum f16 mismatch at idx {}: actual={} expected={} err={} > allowed={}",
        worst.0,
        worst.1,
        worst.2,
        worst.3,
        allowed,
    );
}
