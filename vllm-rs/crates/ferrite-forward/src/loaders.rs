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
use ferrite_cuda_core::tensor::GpuTensor;
use ferrite_cuda_core::weights::GpuWeights;
use ferrite_kernels::layers::{
    Bnb4bitLinear, CohereLayerNorm, Embedding, Fp8AnyLinear, Fp8BlockLinear, Fp8Linear,
    LayerNormBias, LinearLayer, MarlinLinear, RmsNorm,
};
use ferrite_kernels::layers_quant::MarlinFormat;

use crate::{layer_weight_path, vision_block_weight_path};

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

/// Vocab-parallel layered Embedding load. Mirrors
/// [`load_layered_embedding`] but slices each layer's embedding
/// table along dim 0 (`vocab_size`) per `(rank, world)`. No
/// layered Embedding accessor exists in any current arch — this
/// helper is here for symmetry with the other `_sharded` helpers
/// and so the codegen macro can route every layered FieldLoad
/// through a `_sharded` variant uniformly. Per Python vLLM's
/// `VocabParallelEmbedding`.
pub fn load_layered_embedding_sharded(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffix: &str,
    rank: usize,
    world: usize,
) -> Result<Vec<Embedding>> {
    (0..n_layers)
        .map(|layer| Embedding::load_sharded(gw, &layer_weight_path(layer, suffix), rank, world))
        .collect()
}

pub fn load_layered_rms_norm(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffix: &str,
    eps: f32,
) -> Result<Vec<RmsNorm>> {
    (0..n_layers)
        .map(|layer| RmsNorm::load(gw, &layer_weight_path(layer, suffix), eps))
        .collect()
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

pub fn load_layered_layer_norm_bias(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffix: &str,
    eps: f32,
) -> Result<Vec<LayerNormBias>> {
    (0..n_layers)
        .map(|layer| LayerNormBias::load(gw, &vision_block_weight_path(layer, suffix), eps))
        .collect()
}

/// Vision-tower analogue of [`load_layered_linear_dense`]:
/// `visual.blocks.<L>.<suffix>` per-block prefix instead of
/// `model.layers.<L>.<suffix>`.
pub fn load_layered_linear_dense_vision(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffix: &str,
) -> Result<Vec<LinearLayer>> {
    (0..n_layers)
        .map(|layer| LinearLayer::load_dense(gw, &vision_block_weight_path(layer, suffix)))
        .collect()
}

pub fn load_layered_linear_dense(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffix: &str,
) -> Result<Vec<LinearLayer>> {
    // `load_dense_or_ggml`: tries `take_quantized_linear` first
    // (for `StorageFormat::Ggml` weights), falls back to dense.
    // Transparent on safetensors models since the GGUF map is empty.
    (0..n_layers)
        .map(|layer| LinearLayer::load_dense_or_ggml(gw, &layer_weight_path(layer, suffix)))
        .collect()
}

/// Tensor-parallel layered dense Linear load. See
/// [`ferrite_kernels::layers::Linear::load_sharded`] for the per-
/// dim bias semantics. Used by codegen at tp>1: column-parallel
/// (q/k/v/gate/up) → `dim = 0`; row-parallel (o/down) → `dim = 1`.
pub fn load_layered_linear_dense_sharded(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffix: &str,
    dim: usize,
    rank: usize,
    world: usize,
) -> Result<Vec<LinearLayer>> {
    (0..n_layers)
        .map(|layer| {
            LinearLayer::load_dense_sharded(gw, &layer_weight_path(layer, suffix), dim, rank, world)
        })
        .collect()
}

pub fn load_layered_linear_dense_concat(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffixes: &[&str],
    stream: CUstream,
) -> Result<Vec<LinearLayer>> {
    if std::env::var("FERRITE_GGUF_TRACE").is_ok() {
        eprintln!(
            "[ggml] load_layered_linear_dense_concat: n_layers={n_layers} suffixes={suffixes:?}"
        );
    }
    (0..n_layers)
        .map(|layer| {
            let paths = concat_paths_for_layer(layer, suffixes);
            let refs = as_str_refs(&paths);
            LinearLayer::load_dense_concat_or_ggml(gw, &refs, stream)
        })
        .collect()
}

/// Tensor-parallel column-parallel concat (no `dim` arg — fused
/// QKV / gate_up are always column-parallel; see
/// [`ferrite_kernels::layers::LinearLayer::load_dense_concat_sharded`]).
pub fn load_layered_linear_dense_concat_sharded(
    gw: &mut GpuWeights,
    n_layers: u32,
    suffixes: &[&str],
    stream: CUstream,
    rank: usize,
    world: usize,
) -> Result<Vec<LinearLayer>> {
    (0..n_layers)
        .map(|layer| {
            let paths = concat_paths_for_layer(layer, suffixes);
            let refs = as_str_refs(&paths);
            LinearLayer::load_dense_concat_sharded(gw, &refs, stream, rank, world)
        })
        .collect()
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
