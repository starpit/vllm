//! Diagnostic: tokenize a chat-templated prompt through `gguf_tokenizer`
//! and print token IDs for comparison against the HF reference. Also
//! dumps the first few elements of `token_embd.weight` (after dequant
//! to f32) so we can compare against the safetensors `model.embed_tokens.weight`.

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: gguf_tok_dump <path-to-.gguf>"))?
        .into();

    let gguf = ferrite_gguf::GgufFile::open(&path)?;
    let tok = ferrite_gguf::gguf_tokenizer(&gguf)?
        .ok_or_else(|| anyhow::anyhow!("gguf_tokenizer returned None"))?;

    let mut args = std::env::args().skip(2);
    let templated = match args.next().as_deref() {
        Some("qwen") => {
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n".to_string()
        }
        _ => {
            let prompt = "why is the sky blue?";
            format!(
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        }
    };

    let enc = tok
        .encode(templated, false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let ids = enc.get_ids();
    println!("GGUF tokens ({}): {:?}", ids.len(), ids);
    let pieces: Vec<String> = ids
        .iter()
        .map(|&i| tok.id_to_token(i).unwrap_or_else(|| "<?>".into()))
        .collect();
    println!("GGUF pieces: {:?}", pieces);

    Ok(())
}
