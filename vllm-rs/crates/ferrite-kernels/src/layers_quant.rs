// SPDX-License-Identifier: Apache-2.0
//! Quantized weight loading — AWQ/GPTQ → Marlin repack.
//!
//! Moved here from `vllm-cuda::weights_quant` + `vllm-cuda::quant` so
//! that `ferrite-forward`-emitted `Weights::load` bodies can call
//! these directly. The hand-written path in `vllm-cuda` still reaches
//! them via the re-exports it always had (`ferrite_kernels::layers`,
//! `ferrite_kernels::layers_quant`) — behavior-preserving move.
//!
//! AWQ and GPTQ live side-by-side because both feed the same Marlin
//! kernel after repack. They stay on distinct loader fn names
//! (`load_awq[_concat]` vs `load_gptq[_concat]`) because the on-disk
//! shape / metadata differ — AWQ qweight is `[K, N/8]` with zero
//! points, GPTQ qweight is `[K/8, N]` with an optional `.g_idx`
//! activation-order tensor and symmetric-mode zero-point-free
//! scales. Codegen picks the right entry point at macro-expansion
//! time from `quantization_config.quant_method`; there's no runtime
//! AWQ-vs-GPTQ branch in the generated loader.
//!
//! Compressed-tensors (the vLLM `compressed-tensors` pack format
//! that mirrors GPTQ packing on a transposed layout) stays in
//! `vllm-cuda::weights_quant` until ferrite grows its own
//! `StorageFormat::CompressedTensors` variant.

use anyhow::Result;
use cudarc::driver::sys::CUstream;

use ferrite_cuda_core::alloc::CachingAllocator;
use ferrite_cuda_core::driver;
use ferrite_cuda_core::dtype::DType;
use ferrite_cuda_core::tensor::GpuTensor;
use ferrite_cuda_core::weights::GpuWeights;

use crate::layers::MarlinLinear;

// ---------------------------------------------------------------------------
// Marlin workspace
// ---------------------------------------------------------------------------

/// Allocate the shared Marlin workspace buffer `[max(2*num_sm, 1M)]` i32.
///
/// One allocation per device serves every MarlinLinear on that device —
/// Marlin's kernel uses it as a barrier-lock region. Must be zero'd
/// on creation; subsequent forwards re-zero it implicitly on use.
pub fn alloc_marlin_workspace(num_sm: i32, stream: CUstream) -> Result<GpuTensor> {
    // Match Python: max(2 * num_sm, 1024 * 1024) elements.
    let num_elements = std::cmp::max(2 * num_sm as usize, 1024 * 1024);
    let nbytes = num_elements * std::mem::size_of::<i32>();
    let ptr = unsafe { driver::mem_alloc(nbytes)? };
    unsafe { driver::memset_d8(ptr, 0, nbytes, stream)? };
    Ok(unsafe { GpuTensor::new(ptr, &[num_elements], DType::I32) })
}

// ---------------------------------------------------------------------------
// Marlin scale / zero-point permutations (CPU, at load time)
// ---------------------------------------------------------------------------

/// Marlin scale permutation indices for group quantization.
/// Python: `[i + 8*j for i in range(8) for j in range(8)]`.
pub fn scale_perm() -> [usize; 64] {
    let mut p = [0usize; 64];
    let mut idx = 0;
    for i in 0..8 {
        for j in 0..8 {
            p[idx] = i + 8 * j;
            idx += 1;
        }
    }
    p
}

/// Marlin scale permutation indices for per-channel / act_order.
/// Python: `[2*i + j for i in range(4) for j in [0,1,8,9,16,17,24,25]]`.
pub fn scale_perm_single() -> [usize; 32] {
    let mut p = [0usize; 32];
    let offsets = [0, 1, 8, 9, 16, 17, 24, 25];
    let mut idx = 0;
    for i in 0..4 {
        for &j in &offsets {
            p[idx] = 2 * i + j;
            idx += 1;
        }
    }
    p
}

/// Apply the Marlin scale permutation to a flat scale buffer in place.
///
/// `scales`: logical `[num_groups, size_n]` of f16/bf16 bits reinterpreted as u16.
pub fn marlin_permute_scales(scales: &mut [u16], size_k: usize, size_n: usize, group_size: usize) {
    let num_groups = if group_size > 0 && group_size < size_k {
        size_k / group_size
    } else {
        1
    };
    let use_single = group_size >= size_k || group_size == 0;

    if use_single {
        let perm = scale_perm_single();
        let chunk = perm.len(); // 32
        let num_chunks = (num_groups * size_n) / chunk;
        let mut tmp = vec![0u16; chunk];
        for c in 0..num_chunks {
            let base = c * chunk;
            for (i, &p) in perm.iter().enumerate() {
                tmp[i] = scales[base + p];
            }
            scales[base..base + chunk].copy_from_slice(&tmp);
        }
    } else {
        let perm = scale_perm();
        let chunk = perm.len(); // 64
        let num_chunks = (num_groups * size_n) / chunk;
        let mut tmp = vec![0u16; chunk];
        for c in 0..num_chunks {
            let base = c * chunk;
            for (i, &p) in perm.iter().enumerate() {
                tmp[i] = scales[base + p];
            }
            scales[base..base + chunk].copy_from_slice(&tmp);
        }
    }
}

/// Unpack packed INT4 columns: `[rows, cols/pack_factor]` u32 → `[rows, cols]` u8.
///
/// Matches Python vLLM's `quant_utils.unpack_cols` convention: consecutive
/// output columns `[pack_factor*p .. pack_factor*p + pack_factor]` are
/// packed into input column `p`'s low-to-high nibbles. (Our earlier
/// implementation used a strided layout — `b * (cols/pack_factor) + p` —
/// which silently corrupted the AWQ → Marlin zero-point pipeline since
/// AutoAWQ stores qzeros in Python's convention.)
pub fn unpack_cols_4bit(packed: &[u32], rows: usize, cols: usize) -> Vec<u8> {
    let pack_factor = 8; // 32 / 4
    assert_eq!(packed.len(), rows * (cols / pack_factor));
    assert_eq!(cols % pack_factor, 0);
    let mut out = vec![0u8; rows * cols];
    for r in 0..rows {
        for p in 0..cols / pack_factor {
            let val = packed[r * (cols / pack_factor) + p];
            for i in 0..pack_factor {
                out[r * cols + pack_factor * p + i] = ((val >> (4 * i)) & 0xF) as u8;
            }
        }
    }
    out
}

/// Pack INT4 columns: `[rows, cols]` u8 → `[rows, cols/pack_factor]` u32.
///
/// Inverse of [`unpack_cols_4bit`]; matches Python vLLM's `pack_cols`.
pub fn pack_cols_4bit(unpacked: &[u8], rows: usize, cols: usize) -> Vec<u32> {
    let pack_factor = 8;
    assert_eq!(unpacked.len(), rows * cols);
    assert_eq!(cols % pack_factor, 0);
    let mut out = vec![0u32; rows * (cols / pack_factor)];
    for r in 0..rows {
        for p in 0..cols / pack_factor {
            let mut val = 0u32;
            for i in 0..pack_factor {
                val |= (unpacked[r * cols + pack_factor * p + i] as u32) << (4 * i);
            }
            out[r * (cols / pack_factor) + p] = val;
        }
    }
    out
}

/// Convert AWQ zero points to Marlin format (CPU).
///
/// Unpacks AWQ qzeros, undoes the AWQ interleave, applies Marlin's
/// scale permutation, re-interleaves for Marlin's dequantizer, and
/// repacks to u32. Matches Python vLLM's `awq_to_marlin_zero_points`.
pub fn awq_to_marlin_zero_points(
    packed_qzeros: &[u32],
    num_groups: usize,
    size_n: usize,
) -> Vec<u32> {
    let mut zp = unpack_cols_4bit(packed_qzeros, num_groups, size_n);

    // Undo AWQ interleaving [0,2,4,6,1,3,5,7].
    let undo_interleave: [usize; 8] = {
        let interleave = [0usize, 2, 4, 6, 1, 3, 5, 7];
        let mut inv = [0usize; 8];
        for (i, &v) in interleave.iter().enumerate() {
            inv[v] = i;
        }
        inv
    };

    let total = num_groups * size_n;
    let mut tmp = [0u8; 8];
    for chunk_start in (0..total).step_by(8) {
        for (i, &p) in undo_interleave.iter().enumerate() {
            tmp[i] = zp[chunk_start + p];
        }
        zp[chunk_start..chunk_start + 8].copy_from_slice(&tmp);
    }

    // Apply Marlin scale_perm to the zero points.
    let perm = scale_perm();
    let chunk_size = 64;
    let num_chunks = total / chunk_size;
    let mut tmp64 = vec![0u8; chunk_size];
    for c in 0..num_chunks {
        let base = c * chunk_size;
        for (i, &p) in perm.iter().enumerate() {
            tmp64[i] = zp[base + p];
        }
        zp[base..base + chunk_size].copy_from_slice(&tmp64);
    }

    // Re-interleave for Marlin dequantizer [0,2,4,6,1,3,5,7].
    let interleave = [0usize, 2, 4, 6, 1, 3, 5, 7];
    let mut tmp8 = [0u8; 8];
    for chunk_start in (0..total).step_by(8) {
        for (i, &p) in interleave.iter().enumerate() {
            tmp8[i] = zp[chunk_start + p];
        }
        zp[chunk_start..chunk_start + 8].copy_from_slice(&tmp8);
    }

    pack_cols_4bit(&zp, num_groups, size_n)
}

// ---------------------------------------------------------------------------
// CPU concat helpers used by fused loaders
// ---------------------------------------------------------------------------

/// Concatenate multiple 2D CPU tensors along dim 1 (output/N).
/// All tensors must share dim 0 and dtype; row-interleaves the bytes.
pub fn concat_cpu_dim1(tensors: &[(&[u8], &[usize], DType)]) -> (Vec<u8>, Vec<usize>, DType) {
    assert!(!tensors.is_empty());
    let dtype = tensors[0].2;
    let dim0 = tensors[0].1[0];
    let elem_size = dtype.size_bytes();

    let total_dim1: usize = tensors.iter().map(|(_, shape, _)| shape[1]).sum();

    let total_bytes = dim0 * total_dim1 * elem_size;
    let mut out = vec![0u8; total_bytes];

    for row in 0..dim0 {
        let mut col_offset = 0usize;
        for (data, shape, _) in tensors {
            let n = shape[1];
            let src_row_bytes = n * elem_size;
            let src_start = row * src_row_bytes;
            let dst_start = (row * total_dim1 + col_offset) * elem_size;
            out[dst_start..dst_start + src_row_bytes]
                .copy_from_slice(&data[src_start..src_start + src_row_bytes]);
            col_offset += n;
        }
    }

    (out, vec![dim0, total_dim1], dtype)
}

/// Concat per-source bias CPU buffers and upload to GPU.
/// Returns `None` when no sources have a bias tensor.
pub fn fuse_bias_parts(
    bias_parts: &[Vec<u8>],
    bias_dtype: Option<DType>,
    weights: &mut GpuWeights,
    stream: CUstream,
) -> Result<Option<GpuTensor>> {
    if bias_parts.is_empty() {
        return Ok(None);
    }
    let dtype = bias_dtype.unwrap();
    let total_bytes: usize = bias_parts.iter().map(|b| b.len()).sum();
    let mut fused = Vec::with_capacity(total_bytes);
    for part in bias_parts {
        fused.extend_from_slice(part);
    }
    let num_elements = total_bytes / dtype.size_bytes();
    let ptr = unsafe { driver::mem_alloc(total_bytes)? };
    weights.record_alloc(ptr, total_bytes);
    unsafe { driver::memcpy_htod_async(ptr, fused.as_ptr(), total_bytes, stream)? };
    Ok(Some(unsafe { GpuTensor::new(ptr, &[num_elements], dtype) }))
}

// ---------------------------------------------------------------------------
// AWQ → Marlin loaders (constructors on MarlinLinear)
// ---------------------------------------------------------------------------

impl MarlinLinear {
    /// Load one AWQ-packed INT4 weight and repack to Marlin's tiled layout.
    ///
    /// Reads `{prefix}.qweight` / `.scales` / `.qzeros` (optional `.bias`)
    /// from `weights`. AWQ qweight shape is `[K, N/8]` i32. Caller
    /// supplies the shared marlin workspace and the CUDA device id.
    ///
    /// # AWQ assumptions
    /// - `bits == 4` (AWQ ferrite path is 4-bit only).
    /// - `zero_point == true` (AWQ ships zero points; `has_zp` is
    ///   hard-set on the returned `MarlinLinear`).
    /// - On-disk packing is AutoAWQ's `gemm` (equivalently `gemv`).
    ///   Pre-packed `version = "marlin"` isn't handled here yet; the
    ///   caller is responsible for gating that.
    pub fn load_awq(
        weights: &mut GpuWeights,
        prefix: &str,
        group_size: usize,
        workspace: GpuTensor,
        device_id: i32,
    ) -> Result<Self> {
        let stream = weights.stream();

        let qw_name = format!("{prefix}.qweight");
        let scales_name = format!("{prefix}.scales");
        let qzeros_name = format!("{prefix}.qzeros");

        let (qw_shape, _qw_dtype) = weights
            .tensor_info(&qw_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {qw_name}"))?;
        let size_k = qw_shape[0];
        let size_n = qw_shape[1] * 8;
        let num_groups = if group_size > 0 {
            size_k / group_size
        } else {
            1
        };

        // Upload qweight → GPU, repack AWQ → Marlin, free original.
        let qweight_gpu = weights.take(&qw_name)?;
        let num_u32 = size_k * size_n / 8;
        let repack_nbytes = num_u32 * std::mem::size_of::<u32>();
        let repack_ptr = unsafe { driver::mem_alloc(repack_nbytes)? };
        weights.record_alloc(repack_ptr, repack_nbytes);
        unsafe {
            crate::kernels::awq_repack_into(
                qweight_gpu,
                repack_ptr,
                size_k,
                size_n,
                device_id,
                stream,
            );
            driver::stream_synchronize(stream)?;
            weights.unrecord_alloc(qweight_gpu.raw_ptr());
            driver::mem_free(qweight_gpu.raw_ptr())?;
        }
        let qweight_marlin = unsafe { GpuTensor::new(repack_ptr, &[num_u32], DType::U32) };

        // Scales: CPU-load → permute → upload.
        let (scales_bytes, _scales_shape, scales_dtype) = weights.take_cpu(&scales_name)?;
        let mut scales_u16: Vec<u16> = scales_bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);

        let scales_bytes_permuted: Vec<u8> =
            scales_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let scales_nbytes = scales_bytes_permuted.len();
        let scales_ptr = unsafe { driver::mem_alloc(scales_nbytes)? };
        weights.record_alloc(scales_ptr, scales_nbytes);
        unsafe {
            driver::memcpy_htod_async(
                scales_ptr,
                scales_bytes_permuted.as_ptr(),
                scales_nbytes,
                stream,
            )?;
        }
        let scales_gpu = unsafe { GpuTensor::new(scales_ptr, &[num_groups, size_n], scales_dtype) };

        // Zero points: CPU-load → AWQ-to-Marlin convert → upload.
        let (qzeros_bytes, _qzeros_shape, _qzeros_dtype) = weights.take_cpu(&qzeros_name)?;
        let qzeros_u32: Vec<u32> = qzeros_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let marlin_zp = awq_to_marlin_zero_points(&qzeros_u32, num_groups, size_n);

        let zp_bytes: Vec<u8> = marlin_zp.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let zp_nbytes = zp_bytes.len();
        let zp_ptr = unsafe { driver::mem_alloc(zp_nbytes)? };
        weights.record_alloc(zp_ptr, zp_nbytes);
        unsafe { driver::memcpy_htod_async(zp_ptr, zp_bytes.as_ptr(), zp_nbytes, stream)? };
        let zeros_gpu = unsafe { GpuTensor::new(zp_ptr, &[num_groups, size_n / 8], DType::U32) };

        let bias_name = format!("{prefix}.bias");
        let bias_gpu = if weights.contains(&bias_name) {
            Some(weights.take(&bias_name)?)
        } else {
            None
        };

        unsafe { driver::stream_synchronize(stream)? };

        Ok(Self {
            qweight: qweight_marlin,
            scales: scales_gpu,
            zeros: Some(zeros_gpu),
            g_idx: None,
            g_idx_sort_indices: None,
            workspace,
            size_k,
            size_n,
            group_size,
            num_groups,
            has_zp: true,
            has_act_order: false,
            b_type_id: 1, // AWQ = uint4
            device_id,
            bias: bias_gpu,
        })
    }

    /// Load several AWQ weights and fuse into one Marlin GEMM by
    /// concatenating qweight / scales / qzeros along dim N before
    /// the single repack. All source weights must share `K` and
    /// `group_size`; each contributes its own `N_i`, summing into
    /// `N_total = sum(N_i)`.
    ///
    /// This matches Python vLLM's `MergedColumnParallelLinear` — e.g.
    /// `[q, k, v]` collapse to one wider Marlin layer.
    ///
    /// # Bias handling
    /// If any source weight has a bias, the fused layer carries a
    /// concatenated `[N_total]` bias. Mixing bias/no-bias across
    /// sources is allowed: the caller's upstream HF repos should be
    /// consistent, but partial biases fall through (HF never does
    /// this for QKV / gate-up, but the upstream AWQ loader tolerated
    /// it, so keep parity).
    pub fn load_awq_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        group_size: usize,
        workspace: GpuTensor,
        device_id: i32,
    ) -> Result<Self> {
        assert!(
            !prefixes.is_empty(),
            "load_awq_concat called with empty prefix list",
        );
        let stream = weights.stream();

        let mut qw_parts: Vec<(Vec<u8>, Vec<usize>, DType)> = Vec::new();
        let mut sc_parts: Vec<(Vec<u8>, Vec<usize>, DType)> = Vec::new();
        let mut qz_parts: Vec<(Vec<u8>, Vec<usize>, DType)> = Vec::new();
        let mut bias_parts: Vec<Vec<u8>> = Vec::new();
        let mut bias_dtype: Option<DType> = None;
        let mut part_n_sizes: Vec<usize> = Vec::new();

        for prefix in prefixes {
            let qw_name = format!("{prefix}.qweight");
            let scales_name = format!("{prefix}.scales");
            let qzeros_name = format!("{prefix}.qzeros");

            let (qw_data, qw_shape, qw_dt) = weights.take_cpu(&qw_name)?;
            let part_n = qw_shape[1] * 8;
            part_n_sizes.push(part_n);
            qw_parts.push((qw_data, qw_shape, qw_dt));
            sc_parts.push(weights.take_cpu(&scales_name)?);
            qz_parts.push(weights.take_cpu(&qzeros_name)?);

            let bias_name = format!("{prefix}.bias");
            if weights.contains(&bias_name) {
                let (b_data, _b_shape, b_dtype) = weights.take_cpu(&bias_name)?;
                bias_parts.push(b_data);
                bias_dtype = Some(b_dtype);
            }
        }

        // Concat qweights along dim 1.
        let qw_refs: Vec<_> = qw_parts
            .iter()
            .map(|(d, s, dt)| (d.as_slice(), s.as_slice(), *dt))
            .collect();
        let (qw_fused, qw_shape, _) = concat_cpu_dim1(&qw_refs);
        let size_k = qw_shape[0];
        let size_n = qw_shape[1] * 8;
        let num_groups = if group_size > 0 {
            size_k / group_size
        } else {
            1
        };

        // Upload fused qweight and repack once.
        let qw_nbytes = qw_fused.len();
        let qw_gpu_ptr = unsafe { driver::mem_alloc(qw_nbytes)? };
        unsafe { driver::memcpy_htod_async(qw_gpu_ptr, qw_fused.as_ptr(), qw_nbytes, stream)? };
        let qw_gpu = unsafe { GpuTensor::new(qw_gpu_ptr, &qw_shape, DType::I32) };

        let num_u32 = size_k * size_n / 8;
        let repack_nbytes = num_u32 * std::mem::size_of::<u32>();
        let repack_ptr = unsafe { driver::mem_alloc(repack_nbytes)? };
        weights.record_alloc(repack_ptr, repack_nbytes);
        unsafe {
            crate::kernels::awq_repack_into(qw_gpu, repack_ptr, size_k, size_n, device_id, stream);
            driver::stream_synchronize(stream)?;
            driver::mem_free(qw_gpu_ptr)?;
        }
        let qweight_marlin = unsafe { GpuTensor::new(repack_ptr, &[num_u32], DType::U32) };

        // Concat + permute scales.
        let sc_refs: Vec<_> = sc_parts
            .iter()
            .map(|(d, s, dt)| (d.as_slice(), s.as_slice(), *dt))
            .collect();
        let (sc_fused, _sc_shape, scales_dtype) = concat_cpu_dim1(&sc_refs);
        let mut scales_u16: Vec<u16> = sc_fused
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);

        let scales_bytes_permuted: Vec<u8> =
            scales_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let scales_nbytes = scales_bytes_permuted.len();
        let scales_ptr = unsafe { driver::mem_alloc(scales_nbytes)? };
        weights.record_alloc(scales_ptr, scales_nbytes);
        unsafe {
            driver::memcpy_htod_async(
                scales_ptr,
                scales_bytes_permuted.as_ptr(),
                scales_nbytes,
                stream,
            )?;
        }
        let scales_gpu = unsafe { GpuTensor::new(scales_ptr, &[num_groups, size_n], scales_dtype) };

        // Concat zero points: AWQ qzeros convert per-part, then row-interleave.
        let mut all_marlin_zp: Vec<Vec<u32>> = Vec::new();
        for (i, (qz_data, _qz_shape, _qz_dt)) in qz_parts.iter().enumerate() {
            let qzeros_u32: Vec<u32> = qz_data
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let part_zp = awq_to_marlin_zero_points(&qzeros_u32, num_groups, part_n_sizes[i]);
            all_marlin_zp.push(part_zp);
        }
        let total_n_div8 = size_n / 8;
        let mut fused_zp = vec![0u32; num_groups * total_n_div8];
        let mut col_offsets: Vec<usize> = Vec::new();
        let mut cumulative = 0usize;
        for &pn in &part_n_sizes {
            col_offsets.push(cumulative);
            cumulative += pn / 8;
        }
        for (i, &pn) in part_n_sizes.iter().enumerate() {
            let part_cols = pn / 8;
            for g in 0..num_groups {
                let dst_start = g * total_n_div8 + col_offsets[i];
                let src_start = g * part_cols;
                fused_zp[dst_start..dst_start + part_cols]
                    .copy_from_slice(&all_marlin_zp[i][src_start..src_start + part_cols]);
            }
        }

        let zp_bytes: Vec<u8> = fused_zp.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let zp_nbytes = zp_bytes.len();
        let zp_ptr = unsafe { driver::mem_alloc(zp_nbytes)? };
        weights.record_alloc(zp_ptr, zp_nbytes);
        unsafe { driver::memcpy_htod_async(zp_ptr, zp_bytes.as_ptr(), zp_nbytes, stream)? };
        let zeros_gpu = unsafe { GpuTensor::new(zp_ptr, &[num_groups, total_n_div8], DType::U32) };

        unsafe { driver::stream_synchronize(stream)? };

        Ok(Self {
            qweight: qweight_marlin,
            scales: scales_gpu,
            zeros: Some(zeros_gpu),
            g_idx: None,
            g_idx_sort_indices: None,
            workspace,
            size_k,
            size_n,
            group_size,
            num_groups,
            has_zp: true,
            has_act_order: false,
            b_type_id: 1, // AWQ = uint4
            device_id,
            bias: fuse_bias_parts(&bias_parts, bias_dtype, weights, stream)?,
        })
    }

    /// Load one GPTQ-packed INT4 weight and repack to Marlin's tiled
    /// layout.
    ///
    /// Reads `{prefix}.qweight` / `.scales` / (optional `.qzeros`) /
    /// (optional `.g_idx`) / (optional `.bias`) from `weights`. GPTQ
    /// qweight shape is `[K/8, N]` i32 (note: input-dim packed,
    /// opposite of AWQ's `[K, N/8]`). Caller supplies the shared
    /// marlin workspace and the CUDA device id.
    ///
    /// # GPTQ assumptions
    /// - `bits == 4`.
    /// - Symmetric quantization: `.qzeros` is consumed and discarded
    ///   because Marlin's `uint4b8` scalar type bakes in the
    ///   zero-point (bias=8). Asymmetric GPTQ isn't handled here
    ///   yet — ferrite's parser rejects `sym == false` downstream,
    ///   but we don't re-check that here.
    /// - `desc_act`: when `true` and the model ships `.g_idx`, the
    ///   loader argsort-permutes the group-id vector and hands the
    ///   resulting sort_indices to `gptq_repack_into` so same-group
    ///   columns are contiguous in the repacked weight.
    pub fn load_gptq(
        weights: &mut GpuWeights,
        prefix: &str,
        group_size: usize,
        desc_act: bool,
        workspace: GpuTensor,
        device_id: i32,
    ) -> Result<Self> {
        let stream = weights.stream();

        let qw_name = format!("{prefix}.qweight");
        let (qw_shape, _qw_dtype) = weights
            .tensor_info(&qw_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {qw_name}"))?;
        // GPTQ: [K/8, N]
        let size_k = qw_shape[0] * 8;
        let size_n = qw_shape[1];
        let num_groups = if group_size > 0 {
            size_k / group_size
        } else {
            1
        };

        // GPTQ symmetric stores zero points on disk as a formality.
        // Python vLLM never passes them to Marlin — consume and drop.
        let qzeros_name = format!("{prefix}.qzeros");
        if weights.contains(&qzeros_name) {
            let _ = weights.take(&qzeros_name);
        }

        // Handle g_idx for desc_act (activation ordering) BEFORE
        // repack — repack needs `perm` (sort_indices) on GPU.
        let g_idx_name = format!("{prefix}.g_idx");
        let (g_idx_gpu, sort_indices_gpu, has_act_order) = if desc_act
            && weights.contains(&g_idx_name)
        {
            let (g_idx_bytes, _g_idx_shape, _g_idx_dtype) = weights.take_cpu(&g_idx_name)?;
            let g_idx_i32: Vec<i32> = g_idx_bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            // Stable ascending argsort by group id.
            let mut sort_indices: Vec<i32> = (0..g_idx_i32.len() as i32).collect();
            sort_indices.sort_by_key(|&i| g_idx_i32[i as usize]);

            let sorted_g_idx: Vec<i32> = sort_indices
                .iter()
                .map(|&i| g_idx_i32[i as usize])
                .collect();

            let g_idx_bytes: Vec<u8> = sorted_g_idx.iter().flat_map(|&v| v.to_le_bytes()).collect();
            let g_idx_nbytes = g_idx_bytes.len();
            let g_idx_ptr = unsafe { driver::mem_alloc(g_idx_nbytes)? };
            weights.record_alloc(g_idx_ptr, g_idx_nbytes);
            unsafe {
                driver::memcpy_htod_async(g_idx_ptr, g_idx_bytes.as_ptr(), g_idx_nbytes, stream)?;
            }
            let g_idx_gpu = unsafe { GpuTensor::new(g_idx_ptr, &[size_k], DType::I32) };

            let si_bytes: Vec<u8> = sort_indices.iter().flat_map(|&v| v.to_le_bytes()).collect();
            let si_nbytes = si_bytes.len();
            let si_ptr = unsafe { driver::mem_alloc(si_nbytes)? };
            weights.record_alloc(si_ptr, si_nbytes);
            unsafe { driver::memcpy_htod_async(si_ptr, si_bytes.as_ptr(), si_nbytes, stream)? };
            let sort_indices_gpu = unsafe { GpuTensor::new(si_ptr, &[size_k], DType::I32) };

            (Some(g_idx_gpu), Some(sort_indices_gpu), true)
        } else {
            // Consume g_idx if present but unused — upstream may ship
            // it even with desc_act=false.
            if weights.contains(&g_idx_name) {
                let _ = weights.take(&g_idx_name);
            }
            (None, None, false)
        };

        // Upload qweight → GPU, repack GPTQ → Marlin, free original.
        let qweight_gpu = weights.take(&qw_name)?;
        let num_u32 = size_k * size_n / 8;
        let repack_nbytes = num_u32 * std::mem::size_of::<u32>();
        let repack_ptr = unsafe { driver::mem_alloc(repack_nbytes)? };
        weights.record_alloc(repack_ptr, repack_nbytes);
        unsafe {
            crate::kernels::gptq_repack_into(
                qweight_gpu,
                sort_indices_gpu,
                repack_ptr,
                size_k,
                size_n,
                device_id,
                stream,
            );
            driver::stream_synchronize(stream)?;
            weights.unrecord_alloc(qweight_gpu.raw_ptr());
            driver::mem_free(qweight_gpu.raw_ptr())?;
        }
        let qweight_marlin = unsafe { GpuTensor::new(repack_ptr, &[num_u32], DType::U32) };

        // Scales: CPU-load → Marlin permute → upload.
        let scales_name = format!("{prefix}.scales");
        let (scales_bytes, _scales_shape, scales_dtype) = weights.take_cpu(&scales_name)?;
        let mut scales_u16: Vec<u16> = scales_bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);

        let scales_bytes_permuted: Vec<u8> =
            scales_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let scales_nbytes = scales_bytes_permuted.len();
        let scales_ptr = unsafe { driver::mem_alloc(scales_nbytes)? };
        weights.record_alloc(scales_ptr, scales_nbytes);
        unsafe {
            driver::memcpy_htod_async(
                scales_ptr,
                scales_bytes_permuted.as_ptr(),
                scales_nbytes,
                stream,
            )?;
        }
        let scales_gpu = unsafe { GpuTensor::new(scales_ptr, &[num_groups, size_n], scales_dtype) };

        let bias_name = format!("{prefix}.bias");
        let bias_gpu = if weights.contains(&bias_name) {
            Some(weights.take(&bias_name)?)
        } else {
            None
        };

        unsafe { driver::stream_synchronize(stream)? };

        Ok(Self {
            qweight: qweight_marlin,
            scales: scales_gpu,
            zeros: None,
            g_idx: g_idx_gpu,
            g_idx_sort_indices: sort_indices_gpu,
            workspace,
            size_k,
            size_n,
            group_size,
            num_groups,
            has_zp: false,
            has_act_order,
            b_type_id: 0, // GPTQ = uint4b8
            device_id,
            bias: bias_gpu,
        })
    }

    /// Load several GPTQ weights and fuse into one Marlin GEMM by
    /// concatenating qweight / scales along dim 1 (output/N) before
    /// the single repack. All source weights must share `K` and
    /// `group_size`; each contributes its own `N_i`.
    ///
    /// Matches Python vLLM's `MergedColumnParallelLinear` for GPTQ.
    ///
    /// # g_idx handling (desc_act)
    /// `.g_idx` is indexed by K (input dim), and all fused sub-layers
    /// share K, so a single `.g_idx` covers the fused layer. Take it
    /// from the first prefix; consume and discard from the rest.
    pub fn load_gptq_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        group_size: usize,
        desc_act: bool,
        workspace: GpuTensor,
        device_id: i32,
    ) -> Result<Self> {
        assert!(
            !prefixes.is_empty(),
            "load_gptq_concat called with empty prefix list",
        );
        let stream = weights.stream();

        let mut qw_parts: Vec<(Vec<u8>, Vec<usize>, DType)> = Vec::new();
        let mut sc_parts: Vec<(Vec<u8>, Vec<usize>, DType)> = Vec::new();
        let mut g_idx_i32: Option<Vec<i32>> = None;
        let mut bias_parts: Vec<Vec<u8>> = Vec::new();
        let mut bias_dtype: Option<DType> = None;

        for (i, prefix) in prefixes.iter().enumerate() {
            let qw_name = format!("{prefix}.qweight");
            qw_parts.push(weights.take_cpu(&qw_name)?);

            let scales_name = format!("{prefix}.scales");
            sc_parts.push(weights.take_cpu(&scales_name)?);

            // Symmetric GPTQ: consume-and-discard `.qzeros` if
            // present. Marlin's `uint4b8` bakes in the bias=8 zp.
            let qzeros_name = format!("{prefix}.qzeros");
            if weights.contains(&qzeros_name) {
                let _ = weights.take_cpu(&qzeros_name);
            }

            // `.g_idx` shared across fused sub-weights — read from
            // first prefix, consume from the rest.
            let g_idx_name = format!("{prefix}.g_idx");
            if weights.contains(&g_idx_name) {
                if i == 0 && desc_act {
                    let (g_bytes, _g_shape, _g_dtype) = weights.take_cpu(&g_idx_name)?;
                    g_idx_i32 = Some(
                        g_bytes
                            .chunks_exact(4)
                            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect(),
                    );
                } else {
                    let _ = weights.take_cpu(&g_idx_name);
                }
            }

            let bias_name = format!("{prefix}.bias");
            if weights.contains(&bias_name) {
                let (b_data, _b_shape, b_dtype) = weights.take_cpu(&bias_name)?;
                bias_parts.push(b_data);
                bias_dtype = Some(b_dtype);
            }
        }

        // Concat qweights along dim 1: [K/8, N_i] → [K/8, N_total].
        let qw_refs: Vec<_> = qw_parts
            .iter()
            .map(|(d, s, dt)| (d.as_slice(), s.as_slice(), *dt))
            .collect();
        let (qw_fused, qw_shape, _) = concat_cpu_dim1(&qw_refs);
        let size_k = qw_shape[0] * 8;
        let size_n = qw_shape[1];
        let num_groups = if group_size > 0 {
            size_k / group_size
        } else {
            1
        };

        // g_idx BEFORE repack — repack consumes sort_indices.
        let (g_idx_gpu, sort_indices_gpu, has_act_order) = if let Some(g_idx) = g_idx_i32 {
            let mut sort_indices: Vec<i32> = (0..g_idx.len() as i32).collect();
            sort_indices.sort_by_key(|&i| g_idx[i as usize]);

            let sorted_g_idx: Vec<i32> = sort_indices.iter().map(|&i| g_idx[i as usize]).collect();

            let g_idx_bytes: Vec<u8> = sorted_g_idx.iter().flat_map(|&v| v.to_le_bytes()).collect();
            let g_idx_nbytes = g_idx_bytes.len();
            let g_idx_ptr = unsafe { driver::mem_alloc(g_idx_nbytes)? };
            weights.record_alloc(g_idx_ptr, g_idx_nbytes);
            unsafe {
                driver::memcpy_htod_async(g_idx_ptr, g_idx_bytes.as_ptr(), g_idx_nbytes, stream)?;
            }
            let g_idx_gpu = unsafe { GpuTensor::new(g_idx_ptr, &[size_k], DType::I32) };

            let si_bytes: Vec<u8> = sort_indices.iter().flat_map(|&v| v.to_le_bytes()).collect();
            let si_nbytes = si_bytes.len();
            let si_ptr = unsafe { driver::mem_alloc(si_nbytes)? };
            weights.record_alloc(si_ptr, si_nbytes);
            unsafe { driver::memcpy_htod_async(si_ptr, si_bytes.as_ptr(), si_nbytes, stream)? };
            let sort_indices_gpu = unsafe { GpuTensor::new(si_ptr, &[size_k], DType::I32) };

            (Some(g_idx_gpu), Some(sort_indices_gpu), true)
        } else {
            (None, None, false)
        };

        // Upload fused qweight, repack once.
        let qw_nbytes = qw_fused.len();
        let qw_gpu_ptr = unsafe { driver::mem_alloc(qw_nbytes)? };
        unsafe { driver::memcpy_htod_async(qw_gpu_ptr, qw_fused.as_ptr(), qw_nbytes, stream)? };
        let qw_gpu = unsafe { GpuTensor::new(qw_gpu_ptr, &qw_shape, DType::I32) };

        let num_u32 = size_k * size_n / 8;
        let repack_nbytes = num_u32 * std::mem::size_of::<u32>();
        let repack_ptr = unsafe { driver::mem_alloc(repack_nbytes)? };
        weights.record_alloc(repack_ptr, repack_nbytes);
        unsafe {
            crate::kernels::gptq_repack_into(
                qw_gpu,
                sort_indices_gpu,
                repack_ptr,
                size_k,
                size_n,
                device_id,
                stream,
            );
            driver::stream_synchronize(stream)?;
            driver::mem_free(qw_gpu_ptr)?;
        }
        let qweight_marlin = unsafe { GpuTensor::new(repack_ptr, &[num_u32], DType::U32) };

        // Concat + permute scales: [num_groups, N_i] → [num_groups, N_total].
        let sc_refs: Vec<_> = sc_parts
            .iter()
            .map(|(d, s, dt)| (d.as_slice(), s.as_slice(), *dt))
            .collect();
        let (sc_fused, _sc_shape, scales_dtype) = concat_cpu_dim1(&sc_refs);
        let mut scales_u16: Vec<u16> = sc_fused
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);

        let scales_bytes_permuted: Vec<u8> =
            scales_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let scales_nbytes = scales_bytes_permuted.len();
        let scales_ptr = unsafe { driver::mem_alloc(scales_nbytes)? };
        weights.record_alloc(scales_ptr, scales_nbytes);
        unsafe {
            driver::memcpy_htod_async(
                scales_ptr,
                scales_bytes_permuted.as_ptr(),
                scales_nbytes,
                stream,
            )?;
        }
        let scales_gpu = unsafe { GpuTensor::new(scales_ptr, &[num_groups, size_n], scales_dtype) };

        unsafe { driver::stream_synchronize(stream)? };

        Ok(Self {
            qweight: qweight_marlin,
            scales: scales_gpu,
            zeros: None,
            g_idx: g_idx_gpu,
            g_idx_sort_indices: sort_indices_gpu,
            workspace,
            size_k,
            size_n,
            group_size,
            num_groups,
            has_zp: false,
            has_act_order,
            b_type_id: 0, // GPTQ = uint4b8
            device_id,
            bias: fuse_bias_parts(&bias_parts, bias_dtype, weights, stream)?,
        })
    }
}

// `CachingAllocator` isn't used inside these loaders (the AWQ ones in
// vllm-cuda took `_alloc` for symmetry with the GPTQ path). Imported
// here so the `use` stays minimal should a future loader want it.
#[allow(dead_code)]
fn _touch(alloc: &mut CachingAllocator) -> &mut CachingAllocator {
    alloc
}
