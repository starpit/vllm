use anyhow::Result;
use ndarray::Array2;
use spnl_core::ir::{Message, Query};

/// Extract text content from a Query tree for embedding.
pub fn contentify(input: &Query) -> Vec<String> {
    match input {
        Query::Seq(v) | Query::Plus(v) | Query::Cross(v) => v.iter().flat_map(contentify).collect(),
        Query::Message(Message::Assistant(s) | Message::System(s) | Message::User(s)) => {
            if s.is_empty() {
                vec![]
            } else {
                vec![s.clone()]
            }
        }
        _ => vec![],
    }
}

/// An embedding provider that calls an OpenAI-compatible `/v1/embeddings` endpoint.
///
/// Configure via:
/// - `VLLM_EMBEDDING_BASE_URL` — base URL (default: `http://localhost:11434/v1`)
/// - `VLLM_EMBEDDING_API_KEY` — optional API key
pub struct HttpEmbeddingProvider {
    pub model: String,
    pub dimensions: usize,
    base_url: String,
    api_key: Option<String>,
    client: reqwest::blocking::Client,
}

impl HttpEmbeddingProvider {
    pub fn new(model: &str, dimensions: usize) -> Self {
        let base_url = std::env::var("VLLM_EMBEDDING_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
        let api_key = std::env::var("VLLM_EMBEDDING_API_KEY").ok();
        Self {
            model: model.to_string(),
            dimensions,
            base_url,
            api_key,
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Probe the embedding endpoint to detect dimensions.
    pub fn probe_dimensions(model: &str) -> Result<usize> {
        let provider = Self::new(model, 0);
        let resp = provider.call_api(&["probe".to_string()])?;
        Ok(resp[0].len())
    }

    pub(crate) fn call_api(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url);
        let mut req = self.client.post(&url).json(&serde_json::json!({
            "model": self.model,
            "input": inputs,
        }));
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp: serde_json::Value = req.send()?.error_for_status()?.json()?;
        let data = resp["data"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("missing 'data' in embedding response"))?;
        data.iter()
            .map(|item| {
                item["embedding"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("missing 'embedding' in response item"))
                    .map(|arr: &Vec<serde_json::Value>| {
                        arr.iter()
                            .filter_map(|v: &serde_json::Value| v.as_f64().map(|f| f as f32))
                            .collect::<Vec<f32>>()
                    })
            })
            .collect()
    }
}

impl leann_core::embedding::EmbeddingProvider for HttpEmbeddingProvider {
    fn compute_embeddings(&self, chunks: &[String]) -> Result<Array2<f32>> {
        let vecs = self.call_api(chunks)?;
        let nrows = vecs.len();
        let ncols = self.dimensions;
        let mut data = Vec::with_capacity(nrows * ncols);
        for v in &vecs {
            if v.len() < ncols {
                data.extend_from_slice(v);
                data.resize(data.len() + ncols - v.len(), 0.0);
            } else {
                data.extend_from_slice(&v[..ncols]);
            }
        }
        Ok(Array2::from_shape_vec((nrows, ncols), data)?)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        &self.model
    }
}
