// SPDX-License-Identifier: Apache-2.0
//! SafeTensors weight loading and model weight management.
//!
//! Provides:
//! - `SafeTensorsFile`: read individual `.safetensors` files
//! - `ModelWeights`: load a full model's weights from a directory (handles
//!   sharded/indexed models via `model.safetensors.index.json`)
//! - `HfModelConfig`: parse HuggingFace `config.json` for architecture info
//!
//! Port of: `vllm/model_executor/model_loader/weight_utils.py`

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};

use crate::error::{ModelError, ModelResult};
use crate::tensor::{self, TensorInfo};

// ---------------------------------------------------------------------------
// SafeTensors file reader
// ---------------------------------------------------------------------------

/// A loaded safetensors file that can yield tensors on demand.
///
/// Uses memory-mapping instead of reading the entire file into a heap
/// buffer, so the OS can page data in lazily and share pages across
/// processes.
pub struct SafeTensorsFile {
    /// Memory-mapped file data (header + tensor bytes).
    data: memmap2::Mmap,
    /// Path for diagnostics.
    path: PathBuf,
}

impl SafeTensorsFile {
    /// Open a safetensors file from disk via memory-mapping.
    pub fn open(path: impl AsRef<Path>) -> ModelResult<Self> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::File::open(&path)?;
        // SAFETY: the file is read-only and we hold no mutable references.
        let data = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| ModelError::Other(format!("{}: mmap failed: {e}", path.display())))?;
        // Validate the safetensors header.
        SafeTensors::deserialize(&data)
            .map_err(|e| ModelError::SafeTensors(format!("{}: {}", path.display(), e)))?;
        Ok(Self { data, path })
    }

    /// Parse the safetensors header from the in-memory data.
    fn parsed(&self) -> ModelResult<SafeTensors<'_>> {
        SafeTensors::deserialize(&self.data)
            .map_err(|e| ModelError::SafeTensors(format!("{}: {}", self.path.display(), e)))
    }

    /// List all tensor names in this file.
    pub fn tensor_names(&self) -> ModelResult<Vec<String>> {
        let st = self.parsed()?;
        Ok(st.names().into_iter().map(|s| s.to_string()).collect())
    }

    /// Get metadata about all tensors without loading their data.
    pub fn tensor_infos(&self) -> ModelResult<Vec<TensorInfo>> {
        let st = self.parsed()?;
        let mut infos = Vec::new();
        for name in st.names() {
            let view = st
                .tensor(name)
                .map_err(|e| ModelError::SafeTensors(format!("{}: {}", name, e)))?;
            let dtype = safetensors_dtype_to_candle(view.dtype())?;
            infos.push(TensorInfo {
                name: name.to_string(),
                shape: view.shape().to_vec(),
                dtype,
            });
        }
        Ok(infos)
    }

    /// Load a single tensor by name onto the given device.
    pub fn load_tensor(&self, name: &str, device: &Device) -> ModelResult<Tensor> {
        let st = self.parsed()?;
        let view = st
            .tensor(name)
            .map_err(|e| ModelError::SafeTensors(format!("{}: {}", name, e)))?;
        let dtype = safetensors_dtype_to_candle(view.dtype())?;
        tensor::from_raw_bytes(view.data(), view.shape(), dtype, device)
    }

    /// Load a tensor and cast to the given dtype.
    pub fn load_tensor_cast(
        &self,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Tensor> {
        let t = self.load_tensor(name, device)?;
        if t.dtype() == dtype {
            Ok(t)
        } else {
            t.to_dtype(dtype).map_err(ModelError::Candle)
        }
    }

    /// Iterate over all tensors, loading each onto the given device.
    pub fn load_all(&self, device: &Device) -> ModelResult<Vec<(String, Tensor)>> {
        let st = self.parsed()?;
        let mut tensors = Vec::new();
        for name in st.names() {
            let view = st
                .tensor(name)
                .map_err(|e| ModelError::SafeTensors(format!("{}: {}", name, e)))?;
            let dtype = safetensors_dtype_to_candle(view.dtype())?;
            let tensor = tensor::from_raw_bytes(view.data(), view.shape(), dtype, device)?;
            tensors.push((name.to_string(), tensor));
        }
        Ok(tensors)
    }

    /// File path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

// ---------------------------------------------------------------------------
// Sharded weight index (model.safetensors.index.json)
// ---------------------------------------------------------------------------

/// The index file that maps tensor names to shard files.
///
/// Format: `model.safetensors.index.json`
/// ```json
/// {
///   "metadata": { "total_size": 12345 },
///   "weight_map": {
///     "model.layers.0.weight": "model-00001-of-00002.safetensors",
///     ...
///   }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeTensorsIndex {
    /// Metadata (usually just total_size).
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,

    /// Maps tensor name → shard filename.
    pub weight_map: HashMap<String, String>,
}

impl SafeTensorsIndex {
    /// Load from a JSON file.
    pub fn from_file(path: impl AsRef<Path>) -> ModelResult<Self> {
        let data = std::fs::read_to_string(path)?;
        let index: Self = serde_json::from_str(&data)?;
        Ok(index)
    }

    /// Get the set of unique shard filenames.
    pub fn shard_files(&self) -> Vec<String> {
        let mut files: Vec<String> = self.weight_map.values().cloned().collect();
        files.sort();
        files.dedup();
        files
    }

    /// Look up which shard file contains a given tensor.
    pub fn get_shard(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(|s| s.as_str())
    }
}

// ---------------------------------------------------------------------------
// Model weights loader
// ---------------------------------------------------------------------------

/// Loads model weights from a directory containing safetensors files.
///
/// Handles both single-file and sharded (indexed) models.
pub struct ModelWeights {
    /// Loaded tensors by name.
    tensors: HashMap<String, Tensor>,
    /// Device tensors were loaded onto.
    device: Device,
}

impl ModelWeights {
    /// Load all weights from a model directory.
    ///
    /// If `model.safetensors.index.json` exists, loads sharded weights.
    /// Otherwise, looks for `model.safetensors`.
    pub fn from_dir(dir: impl AsRef<Path>, device: &Device) -> ModelResult<Self> {
        let dir = dir.as_ref();
        let index_path = dir.join("model.safetensors.index.json");
        let single_path = dir.join("model.safetensors");

        if index_path.exists() {
            Self::from_index(&index_path, device)
        } else if single_path.exists() {
            Self::from_single_file(&single_path, device)
        } else {
            Err(ModelError::Other(format!(
                "No safetensors files found in {}",
                dir.display()
            )))
        }
    }

    /// Create from pre-built tensor map (e.g. dequantized GGUF tensors).
    pub fn from_tensors(tensors: HashMap<String, Tensor>) -> Self {
        let device = tensors
            .values()
            .next()
            .map(|t| t.device().clone())
            .unwrap_or(Device::Cpu);
        Self { tensors, device }
    }

    /// Load from a single safetensors file.
    pub fn from_single_file(path: impl AsRef<Path>, device: &Device) -> ModelResult<Self> {
        let file = SafeTensorsFile::open(&path)?;
        let all = file.load_all(device)?;
        let tensors: HashMap<String, Tensor> = all.into_iter().collect();
        Ok(Self {
            tensors,
            device: device.clone(),
        })
    }

    /// Load from an index file (sharded model).
    pub fn from_index(index_path: impl AsRef<Path>, device: &Device) -> ModelResult<Self> {
        let index_path = index_path.as_ref();
        let dir = index_path
            .parent()
            .ok_or_else(|| ModelError::Other("index file has no parent dir".into()))?;

        let index = SafeTensorsIndex::from_file(index_path)?;
        let shard_files = index.shard_files();
        let total = shard_files.len();

        // Show a progress bar for multi-shard models (like Python vLLM's tqdm).
        let bar = if total > 1 {
            let bar = indicatif::ProgressBar::new(total as u64);
            bar.set_style(
                indicatif::ProgressStyle::with_template(
                    "Loading safetensors {bar:40.cyan/blue} {pos}/{len} shards",
                )
                .unwrap(),
            );
            Some(bar)
        } else {
            None
        };

        let mut tensors = HashMap::new();
        for shard_name in &shard_files {
            let shard_path = dir.join(shard_name);
            let file = SafeTensorsFile::open(&shard_path)?;
            for (name, tensor) in file.load_all(device)? {
                tensors.insert(name, tensor);
            }
            if let Some(ref bar) = bar {
                bar.inc(1);
            }
        }
        if let Some(bar) = bar {
            bar.finish_and_clear();
        }

        Ok(Self {
            tensors,
            device: device.clone(),
        })
    }

    /// Get a tensor by name.
    pub fn get(&self, name: &str) -> ModelResult<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| ModelError::WeightNotFound(name.to_string()))
    }

    /// Get a tensor by name, cast to the given dtype.
    pub fn get_cast(&self, name: &str, dtype: DType) -> ModelResult<Tensor> {
        let t = self.get(name)?;
        if t.dtype() == dtype {
            Ok(t.clone())
        } else {
            t.to_dtype(dtype).map_err(ModelError::Candle)
        }
    }

    /// Check if a tensor exists.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// List all tensor names.
    pub fn names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Number of loaded tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether no tensors are loaded.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// The device tensors are loaded on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Total size in bytes of all loaded tensors.
    pub fn total_size_bytes(&self) -> usize {
        self.tensors
            .values()
            .map(|t| {
                let elements: usize = t.dims().iter().product();
                elements * tensor::dtype_size(t.dtype())
            })
            .sum()
    }

    /// Strip a prefix from all tensor names, keeping only matching tensors.
    ///
    /// Used for composite models (e.g. Kimi K2.5) where the text backbone
    /// weights are stored under a prefix like `language_model.`.
    pub fn strip_prefix(&mut self, prefix: &str) {
        let stripped: HashMap<String, Tensor> = self
            .tensors
            .drain()
            .filter_map(|(name, tensor)| {
                name.strip_prefix(prefix)
                    .map(|rest| (rest.to_string(), tensor))
            })
            .collect();
        self.tensors = stripped;
    }

    /// Remove a tensor from the loaded set (e.g., after loading into a layer).
    pub fn take(&mut self, name: &str) -> ModelResult<Tensor> {
        self.tensors
            .remove(name)
            .ok_or_else(|| ModelError::WeightNotFound(name.to_string()))
    }
}

// ---------------------------------------------------------------------------
// HuggingFace config.json parser
// ---------------------------------------------------------------------------

/// Minimal HuggingFace model configuration parsed from `config.json`.
///
/// Only includes fields commonly needed by vLLM for model loading.
/// The full config can be accessed via the raw JSON.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HfModelConfig {
    /// Model architecture identifiers (e.g., ["LlamaForCausalLM"]).
    #[serde(default, deserialize_with = "deserialize_null_as_default")]
    pub architectures: Vec<String>,

    /// Model type (e.g., "llama", "mistral", "qwen2").
    #[serde(default)]
    pub model_type: Option<String>,

    /// Hidden size / model dimension.
    #[serde(default)]
    pub hidden_size: Option<usize>,

    /// Number of attention heads.
    #[serde(default)]
    pub num_attention_heads: Option<usize>,

    /// Number of key-value heads (for GQA).
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Number of hidden layers.
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,

    /// Intermediate size (FFN dimension).
    #[serde(default)]
    pub intermediate_size: Option<usize>,

    /// Vocabulary size.
    #[serde(default)]
    pub vocab_size: Option<usize>,

    /// Maximum sequence length.
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    /// RMS norm epsilon.
    #[serde(default)]
    pub rms_norm_eps: Option<f64>,

    /// Layer norm epsilon.
    #[serde(default)]
    pub layer_norm_eps: Option<f64>,

    /// RoPE theta.
    #[serde(default)]
    pub rope_theta: Option<f64>,

    /// Torch dtype string (e.g., "float16", "bfloat16").
    #[serde(default)]
    pub torch_dtype: Option<String>,

    /// Tie word embeddings.
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,

    /// Head dimension (if explicitly specified).
    #[serde(default)]
    pub head_dim: Option<usize>,

    /// Raw JSON for accessing any field not in this struct.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl HfModelConfig {
    /// Load from a `config.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> ModelResult<Self> {
        let data = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&data)?;
        Ok(config)
    }

    /// Load from a model directory (reads `config.json` inside it).
    pub fn from_dir(dir: impl AsRef<Path>) -> ModelResult<Self> {
        let path = dir.as_ref().join("config.json");
        Self::from_file(path)
    }

    /// Effective head dimension.
    pub fn head_dim(&self) -> Option<usize> {
        self.head_dim
            .or_else(|| match (self.hidden_size, self.num_attention_heads) {
                (Some(h), Some(n)) if n > 0 => Some(h / n),
                _ => None,
            })
    }

    /// Effective number of KV heads (defaults to num_attention_heads for MHA).
    pub fn num_kv_heads(&self) -> Option<usize> {
        self.num_key_value_heads.or(self.num_attention_heads)
    }

    /// Effective norm epsilon.
    pub fn norm_eps(&self) -> f64 {
        self.rms_norm_eps.or(self.layer_norm_eps).unwrap_or(1e-5)
    }

    /// For composite models (e.g. vision-language models like Kimi K2.5),
    /// extract the text sub-config.
    ///
    /// Returns `Some((text_config, weight_prefix_to_strip))` if this is a
    /// composite model whose text backbone is a supported architecture.
    /// Returns `None` for non-composite models.
    ///
    /// The `weight_prefix_to_strip` should be stripped from safetensors weight
    /// names to make them match the expected format for the underlying model.
    pub fn resolve_text_config(&self) -> Option<(Self, &'static str)> {
        match self.model_type.as_deref() {
            Some("kimi_k25") => {
                let text_config_val = self.extra.get("text_config")?;
                let mut cfg: Self = serde_json::from_value(text_config_val.clone()).ok()?;
                // The text sub-config may not have architectures; preserve the
                // original so the registry lookup uses "KimiK25ForCausalLM".
                if cfg.architectures.is_empty() {
                    cfg.architectures = self.architectures.clone();
                }
                Some((cfg, "language_model."))
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deserialize a value that may be `null` as the type's `Default`.
///
/// `#[serde(default)]` only handles *missing* keys — a key present with value
/// `null` still fails for non-`Option` types like `Vec<String>`.  This
/// deserializer treats `null` the same as absent.
fn deserialize_null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + serde::Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// Convert safetensors Dtype to candle DType.
fn safetensors_dtype_to_candle(dt: safetensors::Dtype) -> ModelResult<DType> {
    use safetensors::Dtype as SD;
    match dt {
        SD::F16 => Ok(DType::F16),
        SD::BF16 => Ok(DType::BF16),
        SD::F32 => Ok(DType::F32),
        SD::F64 => Ok(DType::F64),
        SD::U8 => Ok(DType::U8),
        SD::U32 => Ok(DType::U32),
        SD::I32 => Ok(DType::I32),
        SD::I64 => Ok(DType::I64),
        SD::I16 => Ok(DType::I16),
        other => Err(ModelError::UnsupportedDType(format!("{:?}", other))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Test helpers for creating safetensors files (public for cross-module tests).
#[cfg(test)]
pub mod tests_helper {
    use super::*;
    use std::io::Write;

    /// Create a minimal safetensors file with the given tensors.
    pub fn create_safetensors_file(
        path: &std::path::Path,
        tensors: &[(&str, Vec<usize>, DType, &[u8])],
    ) {
        use safetensors::tensor::TensorView;

        let views: Vec<(&str, TensorView<'_>)> = tensors
            .iter()
            .map(|(name, shape, dtype, data)| {
                let st_dtype = candle_dtype_to_safetensors(*dtype);
                (
                    *name,
                    TensorView::new(st_dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();

        let refs: Vec<_> = views.iter().map(|(n, v)| (*n, v.clone())).collect();
        let bytes = safetensors::tensor::serialize(refs, None).unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&bytes).unwrap();
    }

    fn candle_dtype_to_safetensors(dtype: DType) -> safetensors::Dtype {
        use safetensors::Dtype as SD;
        match dtype {
            DType::F16 => SD::F16,
            DType::BF16 => SD::BF16,
            DType::F32 => SD::F32,
            DType::F64 => SD::F64,
            DType::U8 => SD::U8,
            DType::U32 => SD::U32,
            DType::I32 => SD::I32,
            DType::I64 => SD::I64,
            DType::I16 => SD::I16,
            _ => panic!("unsupported dtype for test: {:?}", dtype),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tests_helper::create_safetensors_file;

    #[test]
    fn test_safetensors_file_open_and_list() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.safetensors");

        // Create a file with two f32 tensors.
        let data1: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let data2: Vec<u8> = [5.0f32, 6.0].iter().flat_map(|f| f.to_le_bytes()).collect();

        create_safetensors_file(
            &path,
            &[
                ("weight", vec![2, 2], DType::F32, &data1),
                ("bias", vec![2], DType::F32, &data2),
            ],
        );

        let file = SafeTensorsFile::open(&path).unwrap();
        let names = file.tensor_names().unwrap();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"weight".to_string()));
        assert!(names.contains(&"bias".to_string()));
    }

    #[test]
    fn test_safetensors_file_load_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.safetensors");

        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        create_safetensors_file(&path, &[("w", vec![2, 3], DType::F32, &data)]);

        let file = SafeTensorsFile::open(&path).unwrap();
        let tensor = file.load_tensor("w", &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[2, 3]);
        assert_eq!(tensor.dtype(), DType::F32);

        let flat = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_safetensors_file_load_tensor_cast() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.safetensors");

        let data: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|f| f.to_le_bytes()).collect();

        create_safetensors_file(&path, &[("x", vec![2], DType::F32, &data)]);

        let file = SafeTensorsFile::open(&path).unwrap();
        let tensor = file
            .load_tensor_cast("x", DType::F64, &Device::Cpu)
            .unwrap();
        assert_eq!(tensor.dtype(), DType::F64);
        let vals = tensor.to_vec1::<f64>().unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_safetensors_file_tensor_infos() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.safetensors");

        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        create_safetensors_file(&path, &[("w", vec![2, 2], DType::F32, &data)]);

        let file = SafeTensorsFile::open(&path).unwrap();
        let infos = file.tensor_infos().unwrap();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].name, "w");
        assert_eq!(infos[0].shape, vec![2, 2]);
        assert_eq!(infos[0].dtype, DType::F32);
        assert_eq!(infos[0].num_elements(), 4);
        assert_eq!(infos[0].size_bytes(), 16);
    }

    #[test]
    fn test_safetensors_file_load_all() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.safetensors");

        let data1: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        let data2: Vec<u8> = [3.0f32, 4.0].iter().flat_map(|f| f.to_le_bytes()).collect();

        create_safetensors_file(
            &path,
            &[
                ("a", vec![2], DType::F32, &data1),
                ("b", vec![2], DType::F32, &data2),
            ],
        );

        let file = SafeTensorsFile::open(&path).unwrap();
        let all = file.load_all(&Device::Cpu).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_safetensors_index_parse() {
        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("model.safetensors.index.json");

        let index_json = r#"{
            "metadata": {"total_size": 1000},
            "weight_map": {
                "model.embed.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.weight": "model-00002-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors"
            }
        }"#;
        std::fs::write(&index_path, index_json).unwrap();

        let index = SafeTensorsIndex::from_file(&index_path).unwrap();
        assert_eq!(index.weight_map.len(), 4);

        let shards = index.shard_files();
        assert_eq!(shards.len(), 2);

        assert_eq!(
            index.get_shard("model.embed.weight"),
            Some("model-00001-of-00002.safetensors")
        );
        assert_eq!(
            index.get_shard("model.layers.1.weight"),
            Some("model-00002-of-00002.safetensors")
        );
        assert_eq!(index.get_shard("nonexistent"), None);
    }

    #[test]
    fn test_model_weights_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        let data1: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let data2: Vec<u8> = [5.0f32, 6.0].iter().flat_map(|f| f.to_le_bytes()).collect();

        create_safetensors_file(
            &path,
            &[
                ("weight", vec![2, 2], DType::F32, &data1),
                ("bias", vec![2], DType::F32, &data2),
            ],
        );

        let weights = ModelWeights::from_dir(dir.path(), &Device::Cpu).unwrap();
        assert_eq!(weights.len(), 2);
        assert!(weights.contains("weight"));
        assert!(weights.contains("bias"));

        let w = weights.get("weight").unwrap();
        assert_eq!(w.dims(), &[2, 2]);

        let b = weights.get("bias").unwrap();
        assert_eq!(b.dims(), &[2]);
    }

    #[test]
    fn test_model_weights_sharded() {
        let dir = tempfile::tempdir().unwrap();

        // Create two shard files.
        let data1: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        let data2: Vec<u8> = [3.0f32, 4.0].iter().flat_map(|f| f.to_le_bytes()).collect();

        let shard1_path = dir.path().join("model-00001-of-00002.safetensors");
        create_safetensors_file(&shard1_path, &[("w1", vec![2], DType::F32, &data1)]);

        let shard2_path = dir.path().join("model-00002-of-00002.safetensors");
        create_safetensors_file(&shard2_path, &[("w2", vec![2], DType::F32, &data2)]);

        // Create index.
        let index_json = r#"{
            "metadata": {},
            "weight_map": {
                "w1": "model-00001-of-00002.safetensors",
                "w2": "model-00002-of-00002.safetensors"
            }
        }"#;
        std::fs::write(dir.path().join("model.safetensors.index.json"), index_json).unwrap();

        let weights = ModelWeights::from_dir(dir.path(), &Device::Cpu).unwrap();
        assert_eq!(weights.len(), 2);
        assert!(weights.contains("w1"));
        assert!(weights.contains("w2"));

        let w1 = weights.get("w1").unwrap();
        assert_eq!(w1.to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
        let w2 = weights.get("w2").unwrap();
        assert_eq!(w2.to_vec1::<f32>().unwrap(), vec![3.0, 4.0]);
    }

    #[test]
    fn test_model_weights_take() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        let data: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        create_safetensors_file(&path, &[("w", vec![2], DType::F32, &data)]);

        let mut weights = ModelWeights::from_single_file(&path, &Device::Cpu).unwrap();
        assert_eq!(weights.len(), 1);

        let w = weights.take("w").unwrap();
        assert_eq!(w.to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);

        // After take, it should be gone.
        assert!(weights.is_empty());
        assert!(weights.take("w").is_err());
    }

    #[test]
    fn test_model_weights_total_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        let data1: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let data2: Vec<u8> = [5.0f32, 6.0].iter().flat_map(|f| f.to_le_bytes()).collect();

        create_safetensors_file(
            &path,
            &[
                ("a", vec![2, 2], DType::F32, &data1),
                ("b", vec![2], DType::F32, &data2),
            ],
        );

        let weights = ModelWeights::from_single_file(&path, &Device::Cpu).unwrap();
        // 4 f32 + 2 f32 = 6 * 4 bytes = 24
        assert_eq!(weights.total_size_bytes(), 24);
    }

    #[test]
    fn test_model_weights_no_files() {
        let dir = tempfile::tempdir().unwrap();
        let result = ModelWeights::from_dir(dir.path(), &Device::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn test_hf_model_config_parse() {
        let config_json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "torch_dtype": "float16",
            "tie_word_embeddings": false
        }"#;

        let config: HfModelConfig = serde_json::from_str(config_json).unwrap();
        assert_eq!(config.architectures, vec!["LlamaForCausalLM"]);
        assert_eq!(config.model_type, Some("llama".to_string()));
        assert_eq!(config.hidden_size, Some(4096));
        assert_eq!(config.num_attention_heads, Some(32));
        assert_eq!(config.num_key_value_heads, Some(8));
        assert_eq!(config.num_hidden_layers, Some(32));
        assert_eq!(config.intermediate_size, Some(11008));
        assert_eq!(config.vocab_size, Some(32000));
        assert_eq!(config.head_dim(), Some(128)); // 4096 / 32
        assert_eq!(config.num_kv_heads(), Some(8));
        assert!((config.norm_eps() - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_hf_model_config_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let config_json = r#"{
            "architectures": ["MistralForCausalLM"],
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_attention_heads": 32
        }"#;
        std::fs::write(dir.path().join("config.json"), config_json).unwrap();

        let config = HfModelConfig::from_dir(dir.path()).unwrap();
        assert_eq!(config.model_type, Some("mistral".to_string()));
    }

    #[test]
    fn test_hf_model_config_defaults() {
        let config: HfModelConfig = serde_json::from_str("{}").unwrap();
        assert!(config.architectures.is_empty());
        assert_eq!(config.model_type, None);
        assert_eq!(config.hidden_size, None);
        assert_eq!(config.head_dim(), None);
        assert_eq!(config.num_kv_heads(), None);
        assert!((config.norm_eps() - 1e-5).abs() < 1e-10);
    }

    /// Some HF configs (e.g. MLX community quantized VLMs) include
    /// `"architectures": null` in sub-configs. Verify this deserializes
    /// as an empty vec instead of failing.
    #[test]
    fn test_hf_model_config_null_architectures() {
        let config: HfModelConfig =
            serde_json::from_str(r#"{"architectures": null, "hidden_size": 2560}"#).unwrap();
        assert!(config.architectures.is_empty());
        assert_eq!(config.hidden_size, Some(2560));
    }

    #[test]
    fn test_resolve_text_config_kimi_k25() {
        let config_json = r#"{
            "architectures": ["KimiK25ForCausalLM"],
            "model_type": "kimi_k25",
            "text_config": {
                "model_type": "deepseek_v2",
                "hidden_size": 7168,
                "num_attention_heads": 128,
                "num_hidden_layers": 61,
                "vocab_size": 129280,
                "rms_norm_eps": 1e-6
            },
            "vision_config": {
                "image_size": 384
            }
        }"#;
        let config: HfModelConfig = serde_json::from_str(config_json).unwrap();
        let result = config.resolve_text_config();
        assert!(result.is_some());

        let (text_cfg, prefix) = result.unwrap();
        assert_eq!(prefix, "language_model.");
        assert_eq!(text_cfg.model_type, Some("deepseek_v2".to_string()));
        assert_eq!(text_cfg.hidden_size, Some(7168));
        assert_eq!(text_cfg.num_hidden_layers, Some(61));
        // Architectures should be preserved from outer config.
        assert_eq!(text_cfg.architectures, vec!["KimiK25ForCausalLM"]);
    }

    #[test]
    fn test_resolve_text_config_non_composite() {
        let config_json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096
        }"#;
        let config: HfModelConfig = serde_json::from_str(config_json).unwrap();
        assert!(config.resolve_text_config().is_none());
    }

    #[test]
    fn test_model_weights_strip_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        let data1: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        let data2: Vec<u8> = [3.0f32, 4.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        let data3: Vec<u8> = [5.0f32, 6.0].iter().flat_map(|f| f.to_le_bytes()).collect();

        create_safetensors_file(
            &path,
            &[
                ("language_model.model.weight", vec![2], DType::F32, &data1),
                ("language_model.lm_head.weight", vec![2], DType::F32, &data2),
                ("vision_tower.proj.weight", vec![2], DType::F32, &data3),
            ],
        );

        let mut weights = ModelWeights::from_single_file(&path, &Device::Cpu).unwrap();
        assert_eq!(weights.len(), 3);

        weights.strip_prefix("language_model.");
        assert_eq!(weights.len(), 2);
        assert!(weights.contains("model.weight"));
        assert!(weights.contains("lm_head.weight"));
        assert!(!weights.contains("vision_tower.proj.weight"));
    }

    #[test]
    fn test_hf_model_config_extra_fields() {
        let config_json = r#"{
            "model_type": "qwen2",
            "sliding_window": 4096,
            "use_cache": true
        }"#;
        let config: HfModelConfig = serde_json::from_str(config_json).unwrap();
        assert_eq!(config.model_type, Some("qwen2".to_string()));
        assert!(config.extra.contains_key("sliding_window"));
        assert_eq!(config.extra["sliding_window"], 4096);
    }

    #[test]
    fn test_safetensors_dtype_conversion() {
        assert_eq!(
            safetensors_dtype_to_candle(safetensors::Dtype::F16).unwrap(),
            DType::F16
        );
        assert_eq!(
            safetensors_dtype_to_candle(safetensors::Dtype::BF16).unwrap(),
            DType::BF16
        );
        assert_eq!(
            safetensors_dtype_to_candle(safetensors::Dtype::F32).unwrap(),
            DType::F32
        );
        assert_eq!(
            safetensors_dtype_to_candle(safetensors::Dtype::U8).unwrap(),
            DType::U8
        );
    }
}
