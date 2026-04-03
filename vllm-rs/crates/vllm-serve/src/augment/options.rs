/// Options controlling RAG augmentation behavior.
#[derive(Clone, Debug)]
pub struct AugmentOptions {
    /// Max fragments to retrieve per Augment node.
    pub max_aug: usize,
    /// Directory where LEANN indexes are stored.
    pub index_dir: String,
    /// Chunk size for sentence-based chunking (characters).
    pub chunk_size: usize,
    /// Chunk overlap for sentence-based chunking (characters).
    pub chunk_overlap: usize,
}

impl Default for AugmentOptions {
    fn default() -> Self {
        Self {
            max_aug: 10,
            index_dir: "data/spnl".to_string(),
            chunk_size: 512,
            chunk_overlap: 50,
        }
    }
}
