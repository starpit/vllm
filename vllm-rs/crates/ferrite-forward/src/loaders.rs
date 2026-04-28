//! Layered-load helper subroutines that fold each per-accessor
//! `let <base>: Vec<T> = (0..N).map(|layer| Type::load(gw,
//! &layer_weight_path(layer, suffix), …)).collect::<Result<Vec<_>>>()?;`
//! block emitted by `emit_layered_load_body` into a single fn-call
//! at the call site.
//!
//! Each accessor used to expand to ~4 lines of source plus the
//! per-iteration `format!`/`load`/return path. Routing through a
//! one-line helper collapses every per-canonical accessor to a
//! single line of expanded source. With ~9 layered accessors per
//! canonical and 57 canonicals in the llama crate, this trims
//! roughly 6-8k lines off the `load_with` section.
//!
//! The helpers are intentionally narrow — one per `FieldLoad` arm —
//! so the codegen stays a 1:1 mapping rather than re-deriving any
//! load shape at call time.

use anyhow::Result;
use ferrite_cuda_core::CUstream;
use ferrite_cuda_core::DType;
use ferrite_cuda_core::driver;
use ferrite_cuda_core::tensor::GpuTensor;
use ferrite_cuda_core::weights::GpuWeights;
use ferrite_kernels::layers::{
    Bnb4bitLinear, CohereLayerNorm, Embedding, Fp8AnyLinear, Fp8BlockLinear, Fp8Linear, Linear,
    LinearLayer, MarlinLinear, RmsNorm,
};
use ferrite_kernels::layers_quant::MarlinFormat;

use crate::layer_weight_path;

/// Build a `Vec<&str>` of fully-qualified weight paths for a
/// concat-style accessor at one specific layer. The returned `paths`
/// owns the `String`s the `&str` borrows point into; the caller
/// must keep `paths` alive for the duration of the borrow.
#[inline]
fn concat_paths_for_layer(layer: u32, suffixes: &[&str]) -> Vec<String> {
    suffixes
        .iter()
        .map(|s| layer_weight_path(layer, s))
        .collect()
}

#[inline]
fn as_str_refs(paths: &[String]) -> Vec<&str> {
    paths.iter().map(|s| s.as_str()).collect()
}

pub fn load_layered_embedding(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffix: &str,
) -> Result<Vec<Embedding>> {
    (0..n_layers)
        .map(|layer| Embedding::load(gw, &layer_weight_path(layer, suffix)))
        .collect()
}

/// Load `n_layers` per-layer RmsNorm weights into ONE contiguous
/// `[n_layers, hidden]` GPU block, with each returned `RmsNorm`
/// holding a per-layer slice view (`[hidden]`). Shape uniformity
/// across layers is required and asserted; the caller's
/// safetensors layout must satisfy it.
///
/// The vendored KvmMega megakernel's `norm_weights_t = gl<bf16,
/// 1, 1, -1, hidden_dim>` indexes layer-stride into one base
/// pointer, so the per-layer `mem_alloc` from the previous
/// implementation produced layer-N allocations at unrelated
/// addresses and corrupted on `layer > 0`. Same Vec<RmsNorm>
/// API to callers — only the underlying allocation strategy
/// changed.
pub fn load_layered_rms_norm(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffix: &str,
    eps: f32,
) -> Result<Vec<RmsNorm>> {
    if n_layers == 0 {
        return Ok(Vec::new());
    }
    // Resolve shape/dtype from layer 0 — the load is per-layer
    // RmsNorm, so the weight is 1D `[hidden]`.
    let l0_name = format!("{}.weight", layer_weight_path(0, suffix));
    let (l0_shape, l0_dtype) = gw
        .tensor_info(&l0_name)
        .ok_or_else(|| anyhow::anyhow!("load_layered_rms_norm: weight not found: {l0_name}"))?;
    if l0_shape.len() != 1 {
        anyhow::bail!(
            "load_layered_rms_norm: layer 0 `{l0_name}` has rank {} (expected 1)",
            l0_shape.len()
        );
    }
    let hidden = l0_shape[0];
    let dtype = l0_dtype;
    let bytes_per_layer = hidden * dtype.size_bytes();
    let bytes_total = bytes_per_layer
        .checked_mul(n_layers as usize)
        .ok_or_else(|| anyhow::anyhow!("load_layered_rms_norm: byte count overflow"))?;

    // Verify every layer's shape matches before allocating.
    for layer in 1..n_layers {
        let name = format!("{}.weight", layer_weight_path(layer, suffix));
        let (shape, dt) = gw
            .tensor_info(&name)
            .ok_or_else(|| anyhow::anyhow!("load_layered_rms_norm: weight not found: {name}"))?;
        if shape != [hidden] {
            anyhow::bail!(
                "load_layered_rms_norm: layer {layer} `{name}` has shape {shape:?} \
                 but layer 0 has [{hidden}] — uniform shape required for contiguous load"
            );
        }
        if dt != dtype {
            anyhow::bail!(
                "load_layered_rms_norm: layer {layer} `{name}` has dtype {dt:?} \
                 but layer 0 has {dtype:?}"
            );
        }
    }

    let block_ptr = unsafe { driver::mem_alloc(bytes_total)? };
    let stream = gw.stream();
    let mut layers: Vec<RmsNorm> = Vec::with_capacity(n_layers as usize);
    for layer in 0..n_layers {
        let offset = layer as usize * bytes_per_layer;
        let dst = unsafe { block_ptr.byte_offset(offset as isize) };
        let name = format!("{}.weight", layer_weight_path(layer, suffix));
        unsafe { gw.take_into(&name, dst, stream)? };
        let view = unsafe { GpuTensor::new(dst, &[hidden], dtype) };
        layers.push(RmsNorm::new(view, eps));
    }
    // Register the block with GpuWeights so it's RAII-freed on
    // sleep alongside everything else loaded through this object.
    gw.record_alloc(block_ptr, bytes_total);
    Ok(layers)
}

pub fn load_layered_cohere_layer_norm(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffix: &str,
    eps: f32,
) -> Result<Vec<CohereLayerNorm>> {
    (0..n_layers)
        .map(|layer| CohereLayerNorm::load(gw, &layer_weight_path(layer, suffix), eps))
        .collect()
}

/// Load `n_layers` dense `LinearLayer`s into ONE contiguous
/// `[n_layers, out_features, in_features]` GPU block, with each
/// returned `LinearLayer::Dense(Linear)` holding a per-layer
/// slice view. Shape uniformity across layers is required and
/// asserted.
///
/// The vendored KvmMega megakernel's `weights_t = gl<bf16, 1,
/// -1, -1, hidden_dim>` indexes layer-stride into one base
/// pointer; per-layer `mem_alloc` would corrupt at `layer > 0`.
/// Same `Vec<LinearLayer>` API to callers — only the underlying
/// allocation strategy changed.
///
/// Bias not currently supported on this contiguous path — if any
/// layer carries a bias, falls back to per-layer `Linear::load`.
/// llama-style models hit the contiguous fast path; models with
/// bias remain functionally correct via the fallback (and are
/// not kvm-eligible anyway).
pub fn load_layered_linear_dense(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffix: &str,
) -> Result<Vec<LinearLayer>> {
    if n_layers == 0 {
        return Ok(Vec::new());
    }

    // Bias detection — any layer with a bias drops us to the
    // per-layer fallback. A single grep is cheap.
    let any_bias =
        (0..n_layers).any(|l| gw.contains(&format!("{}.bias", layer_weight_path(l, suffix))));
    if any_bias {
        return (0..n_layers)
            .map(|layer| LinearLayer::load_dense(gw, &layer_weight_path(layer, suffix)))
            .collect();
    }

    // Resolve layer-0 shape/dtype, then check uniformity. Some
    // checkpoints (Phi-3 family) ship `qkv_proj.weight` in place
    // of per-slice `q_proj`/`k_proj`/`v_proj`; we fall back to
    // the per-layer path so `Linear::load`'s packed-source
    // synthesis kicks in. The contiguous path requires the
    // direct per-layer suffix to be present.
    let l0_name = format!("{}.weight", layer_weight_path(0, suffix));
    let Some((l0_shape, l0_dtype)) = gw.tensor_info(&l0_name) else {
        return (0..n_layers)
            .map(|layer| LinearLayer::load_dense(gw, &layer_weight_path(layer, suffix)))
            .collect();
    };
    if l0_shape.len() != 2 {
        anyhow::bail!(
            "load_layered_linear_dense: layer 0 `{l0_name}` has rank {} (expected 2)",
            l0_shape.len()
        );
    }
    let out_features = l0_shape[0];
    let in_features = l0_shape[1];
    let dtype = l0_dtype;
    let bytes_per_layer = out_features * in_features * dtype.size_bytes();
    let bytes_total = bytes_per_layer
        .checked_mul(n_layers as usize)
        .ok_or_else(|| anyhow::anyhow!("load_layered_linear_dense: byte count overflow"))?;

    for layer in 1..n_layers {
        let name = format!("{}.weight", layer_weight_path(layer, suffix));
        let (shape, dt) = gw.tensor_info(&name).ok_or_else(|| {
            anyhow::anyhow!("load_layered_linear_dense: weight not found: {name}")
        })?;
        if shape != [out_features, in_features] {
            anyhow::bail!(
                "load_layered_linear_dense: layer {layer} `{name}` has shape {shape:?} \
                 but layer 0 has [{out_features}, {in_features}] — uniform shape required \
                 for contiguous load"
            );
        }
        if dt != dtype {
            anyhow::bail!(
                "load_layered_linear_dense: layer {layer} `{name}` has dtype {dt:?} \
                 but layer 0 has {dtype:?}"
            );
        }
    }

    let block_ptr = unsafe { driver::mem_alloc(bytes_total)? };
    let stream = gw.stream();
    let mut layers: Vec<LinearLayer> = Vec::with_capacity(n_layers as usize);
    for layer in 0..n_layers {
        let offset = layer as usize * bytes_per_layer;
        let dst = unsafe { block_ptr.byte_offset(offset as isize) };
        let name = format!("{}.weight", layer_weight_path(layer, suffix));
        unsafe { gw.take_into(&name, dst, stream)? };
        let view = unsafe { GpuTensor::new(dst, &[out_features, in_features], dtype) };
        layers.push(LinearLayer::Dense(Linear::new(view, None)));
    }
    gw.record_alloc(block_ptr, bytes_total);
    Ok(layers)
}

/// Load `n_layers` fused-concat `LinearLayer`s (e.g. fused QKV)
/// into ONE contiguous `[n_layers, fused_out, in_features]`
/// GPU block. Each layer's per-source weights are streamed into
/// the right offset within that layer's slice; each returned
/// `LinearLayer::Dense(Linear)` holds a `[fused_out,
/// in_features]` view.
///
/// Used by ferrite-forward's `FusedQkvRopeCacheImpl` accessor
/// (one packed `[q+2kv, hidden]` per layer) and the analogous
/// `gate_up` fusion. The contiguous-across-layers backing is
/// what makes the kvm wrapper pass the existing `qkv_weights`
/// accessor's base pointer to vendor's `weights_t` directly,
/// without a per-step concat.
///
/// Falls back to the per-layer `Linear::load_dense_concat`
/// path when any source weight is missing in `gw` (packed-
/// source synthesis path) or carries a bias (which the fast
/// path doesn't currently materialize into the block).
/// Per-layer fallback for `load_layered_linear_dense_concat` —
/// used when packed-source synthesis is needed (Phi-3 family) or
/// any source weight has a bias the contiguous fast path doesn't
/// materialize. Each layer is its own `mem_alloc`; layout is NOT
/// kvm-compatible across layers, but the model is also not
/// kvm-eligible in those cases.
fn load_layered_linear_dense_concat_per_layer(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffixes: &[&str],
    stream: CUstream,
) -> Result<Vec<LinearLayer>> {
    (0..n_layers)
        .map(|layer| {
            let paths = concat_paths_for_layer(layer, suffixes);
            let refs = as_str_refs(&paths);
            LinearLayer::load_dense_concat(gw, &refs, stream)
        })
        .collect()
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn load_layered_linear_dense_concat(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffixes: &[&str],
    stream: CUstream,
) -> Result<Vec<LinearLayer>> {
    if n_layers == 0 {
        return Ok(Vec::new());
    }
    if suffixes.is_empty() {
        anyhow::bail!("load_layered_linear_dense_concat: empty suffix list");
    }

    // Probe layer-0 sources. If any source is missing (Phi-3
    // packed-qkv) or has a bias, drop to the per-layer path so
    // the existing synthesis / bias handling kicks in.
    let l0_paths = concat_paths_for_layer(0, suffixes);
    let mut per_source_out: Vec<usize> = Vec::with_capacity(suffixes.len());
    let mut shared_in: Option<usize> = None;
    let mut shared_dtype: Option<DType> = None;
    for (i, prefix) in l0_paths.iter().enumerate() {
        let weight_name = format!("{prefix}.weight");
        let Some((shape, dtype)) = gw.tensor_info(&weight_name) else {
            return load_layered_linear_dense_concat_per_layer(gw, n_layers, suffixes, stream);
        };
        if shape.len() != 2 {
            anyhow::bail!(
                "load_layered_linear_dense_concat: layer 0 source `{weight_name}` has rank {} (expected 2)",
                shape.len()
            );
        }
        let bias_name = format!("{prefix}.bias");
        if gw.contains(&bias_name) {
            return load_layered_linear_dense_concat_per_layer(gw, n_layers, suffixes, stream);
        }
        let (out, in_) = (shape[0], shape[1]);
        per_source_out.push(out);
        match shared_in {
            None => shared_in = Some(in_),
            Some(prev) if prev == in_ => {}
            Some(prev) => anyhow::bail!(
                "load_layered_linear_dense_concat: source `{weight_name}` has \
                 in_features {in_}, expected {prev} (suffix `{}`)",
                suffixes[i]
            ),
        }
        match shared_dtype {
            None => shared_dtype = Some(dtype),
            Some(prev) if prev == dtype => {}
            Some(prev) => anyhow::bail!(
                "load_layered_linear_dense_concat: source `{weight_name}` has \
                 dtype {dtype:?}, expected {prev:?}"
            ),
        }
    }
    let in_features = shared_in.unwrap();
    let dtype = shared_dtype.unwrap();
    let fused_out: usize = per_source_out.iter().sum();
    let bytes_per_layer = fused_out * in_features * dtype.size_bytes();
    let per_source_bytes: Vec<usize> = per_source_out
        .iter()
        .map(|out| out * in_features * dtype.size_bytes())
        .collect();
    let bytes_total = bytes_per_layer
        .checked_mul(n_layers as usize)
        .ok_or_else(|| anyhow::anyhow!("load_layered_linear_dense_concat: byte count overflow"))?;

    // Verify every later layer has the same per-source shapes
    // before we allocate. This catches the rare case of varying
    // out-features across layers (still uncommon but possible
    // in research checkpoints).
    for layer in 1..n_layers {
        let paths = concat_paths_for_layer(layer, suffixes);
        for (i, prefix) in paths.iter().enumerate() {
            let weight_name = format!("{prefix}.weight");
            let (shape, dt) = gw.tensor_info(&weight_name).ok_or_else(|| {
                anyhow::anyhow!("load_layered_linear_dense_concat: weight not found: {weight_name}")
            })?;
            if shape != [per_source_out[i], in_features] {
                anyhow::bail!(
                    "load_layered_linear_dense_concat: layer {layer} `{weight_name}` \
                     has shape {shape:?} but layer 0 has [{}, {in_features}] — uniform \
                     shape required for contiguous load",
                    per_source_out[i]
                );
            }
            if dt != dtype {
                anyhow::bail!(
                    "load_layered_linear_dense_concat: layer {layer} `{weight_name}` \
                     has dtype {dt:?}, expected {dtype:?}"
                );
            }
            if gw.contains(&format!("{prefix}.bias")) {
                return load_layered_linear_dense_concat_per_layer(gw, n_layers, suffixes, stream);
            }
        }
    }

    let block_ptr = unsafe { driver::mem_alloc(bytes_total)? };
    let mut layers: Vec<LinearLayer> = Vec::with_capacity(n_layers as usize);
    for layer in 0..n_layers {
        let layer_offset = layer as usize * bytes_per_layer;
        let mut intra_offset: usize = 0;
        let paths = concat_paths_for_layer(layer, suffixes);
        for (i, prefix) in paths.iter().enumerate() {
            let weight_name = format!("{prefix}.weight");
            let dst = unsafe { block_ptr.byte_offset((layer_offset + intra_offset) as isize) };
            unsafe { gw.take_into(&weight_name, dst, stream)? };
            intra_offset += per_source_bytes[i];
        }
        let layer_dst = unsafe { block_ptr.byte_offset(layer_offset as isize) };
        let view = unsafe { GpuTensor::new(layer_dst, &[fused_out, in_features], dtype) };
        layers.push(LinearLayer::Dense(Linear::new(view, None)));
    }
    gw.record_alloc(block_ptr, bytes_total);
    Ok(layers)
}

#[allow(clippy::too_many_arguments)]
pub fn load_layered_marlin_linear(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffix: &str,
    storage: MarlinFormat,
    workspace: GpuTensor,
    device_id: i32,
) -> Result<Vec<MarlinLinear>> {
    (0..n_layers)
        .map(|layer| {
            MarlinLinear::load(
                gw,
                &layer_weight_path(layer, suffix),
                storage,
                workspace,
                device_id,
            )
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn load_layered_marlin_linear_concat(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffixes: &[&str],
    storage: MarlinFormat,
    workspace: GpuTensor,
    device_id: i32,
) -> Result<Vec<MarlinLinear>> {
    (0..n_layers)
        .map(|layer| {
            let paths = concat_paths_for_layer(layer, suffixes);
            let refs = as_str_refs(&paths);
            MarlinLinear::load_concat(gw, &refs, storage, workspace, device_id)
        })
        .collect()
}

pub fn load_layered_fp8_linear(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffix: &str,
    output_dtype: DType,
) -> Result<Vec<Fp8AnyLinear>> {
    (0..n_layers)
        .map(|layer| {
            Fp8Linear::load(gw, &layer_weight_path(layer, suffix), output_dtype)
                .map(Fp8AnyLinear::Std)
        })
        .collect()
}

pub fn load_layered_fp8_linear_concat(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffixes: &[&str],
    output_dtype: DType,
) -> Result<Vec<Fp8AnyLinear>> {
    (0..n_layers)
        .map(|layer| {
            let paths = concat_paths_for_layer(layer, suffixes);
            let refs = as_str_refs(&paths);
            Fp8Linear::load_concat(gw, &refs, output_dtype).map(Fp8AnyLinear::Std)
        })
        .collect()
}

pub fn load_layered_fp8_block_linear(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffix: &str,
    output_dtype: DType,
) -> Result<Vec<Fp8AnyLinear>> {
    (0..n_layers)
        .map(|layer| {
            Fp8BlockLinear::load(gw, &layer_weight_path(layer, suffix), output_dtype)
                .map(Fp8AnyLinear::Block)
        })
        .collect()
}

pub fn load_layered_fp8_block_linear_concat(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffixes: &[&str],
    output_dtype: DType,
) -> Result<Vec<Fp8AnyLinear>> {
    (0..n_layers)
        .map(|layer| {
            let paths = concat_paths_for_layer(layer, suffixes);
            let refs = as_str_refs(&paths);
            Fp8BlockLinear::load_concat(gw, &refs, output_dtype).map(Fp8AnyLinear::Block)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn load_layered_bnb4(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffix: &str,
    code_gpu: GpuTensor,
    dequant_scratch: GpuTensor,
    out_features: usize,
    in_features: usize,
    blocksize: usize,
) -> Result<Vec<Bnb4bitLinear>> {
    (0..n_layers)
        .map(|layer| {
            Bnb4bitLinear::load(
                gw,
                &layer_weight_path(layer, suffix),
                code_gpu,
                dequant_scratch,
                out_features,
                in_features,
                blocksize,
            )
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn load_layered_bnb4_concat(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffixes: &[&str],
    code_gpu: GpuTensor,
    dequant_scratch: GpuTensor,
    out_features_per_shard: &[usize],
    in_features: usize,
    blocksize: usize,
) -> Result<Vec<Bnb4bitLinear>> {
    (0..n_layers)
        .map(|layer| {
            let paths = concat_paths_for_layer(layer, suffixes);
            let refs = as_str_refs(&paths);
            Bnb4bitLinear::load_concat(
                gw,
                &refs,
                code_gpu,
                dequant_scratch,
                out_features_per_shard,
                in_features,
                blocksize,
            )
        })
        .collect()
}
