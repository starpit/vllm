// SPDX-License-Identifier: Apache-2.0
//! Paged KV cache metadata builder for the TK throughput megakernel.
//!
//! Converts ForwardCtx's block_table + seqused_k into the
//! indptr/indices/last_page_len format expected by the throughput ops.

#[cfg(feature = "cuda")]
use crate::ForwardCtx;

/// Decode metadata tensors uploaded to GPU.
#[cfg(feature = "cuda")]
pub struct DecodeMetadata {
    pub kv_indptr_gpu: *mut u8,
    pub kv_indptr_len: usize,
    pub kv_indices_gpu: *mut u8,
    pub kv_indices_len: usize,
    pub kv_last_page_len_gpu: *mut u8,
    pub kv_last_page_len_len: usize,
}

/// Build decode-path paged KV metadata from ForwardCtx.
///
/// For decode, each sequence contributes 1 token. We need:
/// - `kv_indptr`: CSR row pointers into `kv_indices` (length = num_seqs + 1)
/// - `kv_indices`: page indices for each sequence (flattened block_table)
/// - `kv_last_page_len`: number of valid tokens in the last page per seq
#[cfg(feature = "cuda")]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn build_decode_metadata(
    ctx: &ForwardCtx,
    kv_page_size: usize,
    stream: ferrite_cuda_core::CUstream,
) -> DecodeMetadata {
    use ferrite_cuda_core::driver;

    // Read block_table and seqused_k from GPU to host.
    let num_seqs = ctx.block_table.dim(0);
    let max_pages = ctx.block_table.dim(1);

    // block_table: [num_seqs, max_pages_per_seq] u32 on GPU
    let bt_elems = num_seqs * max_pages;
    let mut block_table_host = vec![0i32; bt_elems];
    unsafe {
        driver::memcpy_dtoh_async(
            block_table_host.as_mut_ptr() as *mut u8,
            ctx.block_table.raw_ptr(),
            bt_elems * 4,
            stream,
        ).expect("block_table D2H");
    }

    // seqused_k: [num_seqs] i32 on GPU
    let mut seqused_k_host = vec![0i32; num_seqs];
    unsafe {
        driver::memcpy_dtoh_async(
            seqused_k_host.as_mut_ptr() as *mut u8,
            ctx.seqused_k.raw_ptr(),
            num_seqs * 4,
            stream,
        ).expect("seqused_k D2H");
        driver::stream_synchronize(stream).expect("metadata sync");
    }

    // Build indptr, indices, last_page_len.
    let mut kv_indptr = Vec::with_capacity(num_seqs + 1);
    let mut kv_indices = Vec::new();
    let mut kv_last_page_len = Vec::with_capacity(num_seqs);

    kv_indptr.push(0i32);
    for seq in 0..num_seqs {
        let seq_len = seqused_k_host[seq] as usize;
        let num_pages = seq_len.div_ceil(kv_page_size);

        for p in 0..num_pages {
            kv_indices.push(block_table_host[seq * max_pages + p]);
        }
        kv_indptr.push(kv_indices.len() as i32);

        let last_len = seq_len % kv_page_size;
        kv_last_page_len.push(if last_len == 0 && seq_len > 0 { kv_page_size as i32 } else { last_len as i32 });
    }

    // Upload to GPU.
    let alloc_and_upload = |data: &[i32], stream: ferrite_cuda_core::CUstream| -> (*mut u8, usize) {
        if data.is_empty() {
            // Allocate a dummy 4-byte buffer for empty arrays.
            let buf = unsafe { ferrite_cuda_core::driver::mem_alloc(4).expect("dummy alloc") };
            unsafe { driver::memset_d8(buf, 0, 4, stream).expect("dummy zero"); }
            return (buf, 0);
        }
        let bytes = data.len() * 4;
        let buf = unsafe { ferrite_cuda_core::driver::mem_alloc(bytes).expect("metadata alloc") };
        unsafe {
            driver::memcpy_htod_async(buf, data.as_ptr() as *const u8, bytes, stream)
                .expect("metadata H2D");
        }
        (buf, data.len())
    };

    let (indptr_gpu, indptr_len) = alloc_and_upload(&kv_indptr, stream);
    let (indices_gpu, indices_len) = alloc_and_upload(&kv_indices, stream);
    let (last_page_gpu, last_page_len) = alloc_and_upload(&kv_last_page_len, stream);

    DecodeMetadata {
        kv_indptr_gpu: indptr_gpu,
        kv_indptr_len: indptr_len,
        kv_indices_gpu: indices_gpu,
        kv_indices_len: indices_len,
        kv_last_page_len_gpu: last_page_gpu,
        kv_last_page_len_len: last_page_len,
    }
}

/// Prefill metadata tensors uploaded to GPU.
#[cfg(feature = "cuda")]
pub struct PrefillMetadata {
    /// CSR row pointers for Q/O: [0, seqlen_0, seqlen_0+seqlen_1, ...]
    pub qo_indptr_gpu: *mut u8,
    pub qo_indptr_len: usize,
    /// CSR row pointers into kv_indices per sequence
    pub kv_indptr_gpu: *mut u8,
    pub kv_indptr_len: usize,
    /// Page indices for prefill sequences
    pub kv_indices_gpu: *mut u8,
    pub kv_indices_len: usize,
    /// Valid tokens in last page per sequence
    pub kv_last_page_len_gpu: *mut u8,
    pub kv_last_page_len_len: usize,
    /// Number of prefill tokens
    pub num_prefill_tokens: usize,
    /// Per-sequence info for instruction generation: (num_q_tokens, token_offset)
    /// token_offset = seqused_k - num_q_tokens (cached tokens before this prefill)
    pub seq_info: Vec<(usize, usize)>,
}

/// Build prefill-path paged KV metadata from ForwardCtx.
///
/// For prefill, sequences have >1 query token. We need:
/// - `qo_indptr`: same as cu_seqlens_q (CSR for Q token ranges)
/// - `kv_indptr`: CSR into kv_indices (page ranges per seq, covering ALL KV including new tokens)
/// - `kv_indices`: page indices
/// - `kv_last_page_len`: valid tokens in last page
#[cfg(feature = "cuda")]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn build_prefill_metadata(
    ctx: &ForwardCtx,
    kv_page_size: usize,
    stream: ferrite_cuda_core::CUstream,
) -> PrefillMetadata {
    use ferrite_cuda_core::driver;

    let num_seqs = ctx.block_table.dim(0);
    let max_pages = ctx.block_table.dim(1);

    // Read cu_seqlens_q, block_table, seqused_k from GPU.
    let cu_seqlens_q_len = num_seqs + 1;
    let mut cu_seqlens_q_host = vec![0i32; cu_seqlens_q_len];
    let mut block_table_host = vec![0i32; num_seqs * max_pages];
    let mut seqused_k_host = vec![0i32; num_seqs];

    unsafe {
        driver::memcpy_dtoh_async(
            cu_seqlens_q_host.as_mut_ptr() as *mut u8,
            ctx.cu_seqlens_q.raw_ptr(),
            cu_seqlens_q_len * 4,
            stream,
        ).expect("cu_seqlens_q D2H");
        driver::memcpy_dtoh_async(
            block_table_host.as_mut_ptr() as *mut u8,
            ctx.block_table.raw_ptr(),
            num_seqs * max_pages * 4,
            stream,
        ).expect("block_table D2H");
        driver::memcpy_dtoh_async(
            seqused_k_host.as_mut_ptr() as *mut u8,
            ctx.seqused_k.raw_ptr(),
            num_seqs * 4,
            stream,
        ).expect("seqused_k D2H");
        driver::stream_synchronize(stream).expect("prefill metadata sync");
    }

    // qo_indptr = cu_seqlens_q directly.
    let qo_indptr = cu_seqlens_q_host.clone();
    let num_prefill_tokens = *qo_indptr.last().unwrap() as usize;

    // Build kv_indptr, kv_indices, kv_last_page_len (covering full KV length per seq).
    let mut kv_indptr = Vec::with_capacity(num_seqs + 1);
    let mut kv_indices = Vec::new();
    let mut kv_last_page_len = Vec::with_capacity(num_seqs);
    let mut seq_info = Vec::with_capacity(num_seqs);

    kv_indptr.push(0i32);
    for seq in 0..num_seqs {
        let seq_len_k = seqused_k_host[seq] as usize;
        let q_len = (cu_seqlens_q_host[seq + 1] - cu_seqlens_q_host[seq]) as usize;
        let token_offset = seq_len_k.saturating_sub(q_len);
        seq_info.push((q_len, token_offset));

        let num_pages = seq_len_k.div_ceil(kv_page_size);
        for p in 0..num_pages {
            kv_indices.push(block_table_host[seq * max_pages + p]);
        }
        kv_indptr.push(kv_indices.len() as i32);

        let last_len = seq_len_k % kv_page_size;
        kv_last_page_len.push(if last_len == 0 && seq_len_k > 0 { kv_page_size as i32 } else { last_len as i32 });
    }

    // Upload to GPU.
    let alloc_and_upload = |data: &[i32], stream: ferrite_cuda_core::CUstream| -> (*mut u8, usize) {
        if data.is_empty() {
            let buf = unsafe { ferrite_cuda_core::driver::mem_alloc(4).expect("dummy alloc") };
            unsafe { driver::memset_d8(buf, 0, 4, stream).expect("dummy zero"); }
            return (buf, 0);
        }
        let bytes = data.len() * 4;
        let buf = unsafe { ferrite_cuda_core::driver::mem_alloc(bytes).expect("metadata alloc") };
        unsafe {
            driver::memcpy_htod_async(buf, data.as_ptr() as *const u8, bytes, stream)
                .expect("metadata H2D");
        }
        (buf, data.len())
    };

    let (qo_indptr_gpu, qo_indptr_len) = alloc_and_upload(&qo_indptr, stream);
    let (kv_indptr_gpu, kv_indptr_len) = alloc_and_upload(&kv_indptr, stream);
    let (kv_indices_gpu, kv_indices_len) = alloc_and_upload(&kv_indices, stream);
    let (kv_last_page_gpu, kv_last_page_len_len) = alloc_and_upload(&kv_last_page_len, stream);

    PrefillMetadata {
        qo_indptr_gpu,
        qo_indptr_len,
        kv_indptr_gpu,
        kv_indptr_len,
        kv_indices_gpu,
        kv_indices_len,
        kv_last_page_len_gpu: kv_last_page_gpu,
        kv_last_page_len_len,
        num_prefill_tokens,
        seq_info,
    }
}
