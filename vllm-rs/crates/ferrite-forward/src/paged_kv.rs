// SPDX-License-Identifier: Apache-2.0
//! Pure host-side builder for the paged-KV CSR metadata the
//! KvmMega megakernel consumes.
//!
//! The vendored `tk_megakernel_<canonical>_launch` entry point
//! takes seven int32 vectors describing how the per-token query
//! activations land in the paged KV cache:
//!
//!   prefill side: qo_indptr, kv_indptr, kv_indices, kv_last_page_len
//!   decode  side:           kv_indptr, kv_indices, kv_last_page_len
//!   plus     one combined kv_append_indices and one combined
//!            position_ids vector (length = total query tokens).
//!
//! ferrite's `ForwardCtx` carries the same information in vLLM's
//! flat-batch form (prefill + decode concatenated under a single
//! `cu_seqlens_q`, paged via `block_table` + `slot_mapping`).
//! [`build_paged_kv_metadata_from_host`] is the pure-i32 transform
//! that takes those host-side slices and emits the seven vendor
//! vectors. The cuda-gated marshaling wrapper that calls
//! `tk_megakernel_<canonical>_launch` is responsible for the
//! D2H/H2D round trip; this module never touches the GPU.
//!
//! Reference algorithm: vendor's
//! `~/Megakernels/megakernels/scripts/tp_generate.py::setup_prefill_paging_inputs`
//! and `setup_decode_paging_inputs`, plus
//! `~/Megakernels/megakernels/scripts/test_prefill.py::setup_sequence_pointers`.
//! Vendor builds these from a scheduler-owned `indices_per_seq`;
//! ferrite reuses vLLM's `block_table` directly, which encodes the
//! same information in row-major form.

/// Host-side inputs for one forward call, all i32 slices straight
/// out of a D2H copy. The cuda wrapper does the copy; this module
/// works exclusively on borrowed host buffers.
#[derive(Debug, Clone, Copy)]
pub struct PagedKvHostInputs<'a> {
    /// Cumulative query lengths, vLLM convention. Length
    /// `num_seqs + 1`. Per-seq query length is
    /// `cu_seqlens_q[i+1] - cu_seqlens_q[i]`. Prefill seqs have
    /// `q_len > 1`; decode seqs have `q_len == 1`. Prefill seqs
    /// come first in the flat-batch ordering.
    pub cu_seqlens_q: &'a [i32],

    /// Total KV tokens per seq AFTER appending this step's tokens.
    /// Length `num_seqs`. Used to compute pages-per-seq and last-
    /// page length.
    pub seqused_k: &'a [i32],

    /// Row-major page-index table, shape `[num_seqs,
    /// max_blocks_per_seq]`. Entries beyond
    /// `ceil(seqused_k[i] / page_size)` are unused.
    pub block_table: &'a [i32],

    /// Per-token flat slot index into the paged KV cache treated
    /// as `[num_pages, page_size]`: `slot = page_idx * page_size +
    /// offset`. Length `total_q_tokens`. ferrite already produces
    /// this for FlashAttention's `kv_append`-style write path; the
    /// vendor's `kv_append_indices` is the same value with the
    /// same packing.
    pub slot_mapping: &'a [i32],

    /// Per-token absolute position id. Length `total_q_tokens`.
    pub positions: &'a [i32],

    /// Page size of the KV cache (rows per page).
    pub page_size: i32,

    /// Stride of the row-major `block_table` (number of i32 cells
    /// between successive seqs). Usually `max_blocks_per_seq` from
    /// the scheduler, but can be larger if the block table is
    /// padded.
    pub max_blocks_per_seq: usize,
}

/// Output of [`build_paged_kv_metadata_from_host`]. Vector lengths
/// match the `int num_*` C args of the vendored launcher; the
/// cuda wrapper H2Ds each one and passes the size through.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PagedKvMetadata {
    /// Prefill-side query indptr, shifted to start at 0. Size =
    /// `num_prefill_seqs + 1`.
    pub prefill_qo_indptr: Vec<i32>,
    /// Prefill-side cumulative page count per seq. Size =
    /// `num_prefill_seqs + 1`.
    pub prefill_kv_indptr: Vec<i32>,
    /// Prefill-side flat page indices, concatenated. Size =
    /// `prefill_kv_indptr.last()`.
    pub prefill_kv_indices: Vec<i32>,
    /// Prefill-side last-page length per seq (`1..=page_size`).
    /// Size = `num_prefill_seqs`.
    pub prefill_kv_last_page_len: Vec<i32>,

    /// Decode-side cumulative page count per seq. Size =
    /// `num_decode_seqs + 1`. (No qo_indptr — decode q_len = 1
    /// per seq, indptr is implicitly 0..N.)
    pub decode_kv_indptr: Vec<i32>,
    /// Decode-side flat page indices, concatenated. Size =
    /// `decode_kv_indptr.last()`.
    pub decode_kv_indices: Vec<i32>,
    /// Decode-side last-page length per seq. Size =
    /// `num_decode_seqs`.
    pub decode_kv_last_page_len: Vec<i32>,

    /// Per-token flat KV slot, copied from `slot_mapping`. Size =
    /// `total_q_tokens`.
    pub kv_append_indices: Vec<i32>,
    /// Per-token absolute position. Size = `total_q_tokens`.
    pub position_ids: Vec<i32>,

    /// Number of prefill tokens (sum of prefill q_lens). Vendor's
    /// launcher takes this as a separate scalar; precomputing here
    /// keeps the wrapper trivial.
    pub num_prefill_tokens: i32,
}

/// Build the seven-vector vendor metadata from one forward call's
/// flat-batch inputs.
///
/// Convention assumptions (load-bearing):
/// - Prefill seqs precede decode seqs in the flat batch — vLLM's
///   default ordering. The split is computed from `cu_seqlens_q`:
///   any seq with `q_len > 1` is prefill, `q_len == 1` is decode.
///   We reject mixed orderings (decode seq before prefill seq)
///   with a panic — the megakernel's vector layout has the same
///   assumption baked in.
/// - `seqused_k[i]` is the total KV count *after* the new tokens
///   are appended. Pages-per-seq = `ceil(seqused_k[i] / page_size)`,
///   last-page-len = `((seqused_k[i] - 1) % page_size) + 1`.
/// - `slot_mapping` is already in vendor's `kv_append_indices`
///   packing (`page * page_size + offset`); we copy it verbatim.
pub fn build_paged_kv_metadata_from_host(inputs: PagedKvHostInputs<'_>) -> PagedKvMetadata {
    let num_seqs = inputs.seqused_k.len();
    assert_eq!(
        inputs.cu_seqlens_q.len(),
        num_seqs + 1,
        "cu_seqlens_q length must be num_seqs + 1"
    );
    assert_eq!(
        inputs.block_table.len(),
        num_seqs * inputs.max_blocks_per_seq,
        "block_table length must be num_seqs * max_blocks_per_seq"
    );
    let total_q_tokens = *inputs.cu_seqlens_q.last().unwrap_or(&0) as usize;
    assert_eq!(
        inputs.slot_mapping.len(),
        total_q_tokens,
        "slot_mapping length must match total query tokens"
    );
    assert_eq!(
        inputs.positions.len(),
        total_q_tokens,
        "positions length must match total query tokens"
    );
    assert!(inputs.page_size > 0, "page_size must be positive");

    let mut out = PagedKvMetadata::default();

    // Sweep seqs once. Track whether we've seen the first decode
    // seq so we can enforce the prefill-then-decode ordering.
    let mut seen_decode = false;
    let mut prefill_token_count: i32 = 0;
    out.prefill_qo_indptr.push(0);
    out.prefill_kv_indptr.push(0);
    out.decode_kv_indptr.push(0);

    for seq in 0..num_seqs {
        let q_len = inputs.cu_seqlens_q[seq + 1] - inputs.cu_seqlens_q[seq];
        assert!(q_len >= 1, "seq {} has non-positive q_len {}", seq, q_len);
        let kv_len = inputs.seqused_k[seq];
        assert!(
            kv_len >= 1,
            "seq {} has non-positive kv_len {}",
            seq,
            kv_len
        );

        let pages = ((kv_len + inputs.page_size - 1) / inputs.page_size) as usize;
        let last_page_len = ((kv_len - 1) % inputs.page_size) + 1;
        let block_row = seq * inputs.max_blocks_per_seq;
        let page_indices = &inputs.block_table[block_row..block_row + pages];

        if q_len > 1 {
            assert!(
                !seen_decode,
                "prefill seq {} appears after a decode seq — \
                 ferrite expects prefill seqs first in the flat batch",
                seq
            );
            prefill_token_count += q_len;
            let last = *out.prefill_qo_indptr.last().unwrap();
            out.prefill_qo_indptr.push(last + q_len);
            let last = *out.prefill_kv_indptr.last().unwrap();
            out.prefill_kv_indptr.push(last + pages as i32);
            out.prefill_kv_indices.extend_from_slice(page_indices);
            out.prefill_kv_last_page_len.push(last_page_len);
        } else {
            seen_decode = true;
            let last = *out.decode_kv_indptr.last().unwrap();
            out.decode_kv_indptr.push(last + pages as i32);
            out.decode_kv_indices.extend_from_slice(page_indices);
            out.decode_kv_last_page_len.push(last_page_len);
        }
    }

    out.kv_append_indices.extend_from_slice(inputs.slot_mapping);
    out.position_ids.extend_from_slice(inputs.positions);
    out.num_prefill_tokens = prefill_token_count;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single prefill seq, two pages exactly. Vendor algorithm
    /// reduces to: kv_indptr=[0,2], kv_indices=[the two page
    /// ids], kv_last_page_len=page_size, qo_indptr=[0, q_len].
    #[test]
    fn single_prefill_seq_two_full_pages() {
        // 1 seq × max_blocks_per_seq=4 = 4 cells. Only the first
        // two pages are used (kv_len = 256, page_size = 128 →
        // pages = 2); the trailing two are noise to verify we
        // slice strictly by `pages`.
        let block_table = vec![5, 9, /* unused */ 99, 88];
        let inputs = PagedKvHostInputs {
            cu_seqlens_q: &[0, 256],
            seqused_k: &[256],
            block_table: &block_table,
            slot_mapping: &(0..256).map(|i| 5 * 128 + i).collect::<Vec<_>>(),
            positions: &(0..256).collect::<Vec<_>>(),
            page_size: 128,
            max_blocks_per_seq: 4,
        };
        let m = build_paged_kv_metadata_from_host(inputs);
        assert_eq!(m.prefill_qo_indptr, vec![0, 256]);
        assert_eq!(m.prefill_kv_indptr, vec![0, 2]);
        assert_eq!(m.prefill_kv_indices, vec![5, 9]);
        assert_eq!(m.prefill_kv_last_page_len, vec![128]);
        assert!(m.decode_kv_indptr == vec![0]);
        assert!(m.decode_kv_indices.is_empty());
        assert!(m.decode_kv_last_page_len.is_empty());
        assert_eq!(m.kv_append_indices.len(), 256);
        assert_eq!(m.position_ids.len(), 256);
        assert_eq!(m.num_prefill_tokens, 256);
    }

    /// Prefill seq with a partial last page (kv_len = 200,
    /// page_size = 128). Pages = 2, last_page_len = 200 - 128 =
    /// 72.
    #[test]
    fn single_prefill_seq_partial_last_page() {
        let block_table = vec![3, 4];
        let inputs = PagedKvHostInputs {
            cu_seqlens_q: &[0, 200],
            seqused_k: &[200],
            block_table: &block_table,
            slot_mapping: &vec![0; 200],
            positions: &(0..200).collect::<Vec<_>>(),
            page_size: 128,
            max_blocks_per_seq: 2,
        };
        let m = build_paged_kv_metadata_from_host(inputs);
        assert_eq!(m.prefill_kv_indptr, vec![0, 2]);
        assert_eq!(m.prefill_kv_indices, vec![3, 4]);
        assert_eq!(m.prefill_kv_last_page_len, vec![72]);
        assert_eq!(m.num_prefill_tokens, 200);
    }

    /// kv_len exactly divides page_size. Vendor's last-page-len
    /// is `page_size`, NOT 0. Verify our `((kv_len-1)%page_size)+1`
    /// formula matches.
    #[test]
    fn last_page_len_full_when_aligned() {
        let block_table = vec![1, 2, 3, 4];
        let inputs = PagedKvHostInputs {
            cu_seqlens_q: &[0, 512],
            seqused_k: &[512],
            block_table: &block_table,
            slot_mapping: &vec![0; 512],
            positions: &(0..512).collect::<Vec<_>>(),
            page_size: 128,
            max_blocks_per_seq: 4,
        };
        let m = build_paged_kv_metadata_from_host(inputs);
        assert_eq!(m.prefill_kv_indptr, vec![0, 4]);
        assert_eq!(m.prefill_kv_last_page_len, vec![128]);
    }

    /// Two prefill seqs of different lengths sharing one block
    /// table. Verify cumulative indptrs and that each seq pulls
    /// only its row's first `pages` entries.
    #[test]
    fn two_prefill_seqs_concatenated() {
        // max_blocks_per_seq = 3; row 0 uses 2 pages [10, 11],
        // row 1 uses 1 page [20]. Trailing entries are noise to
        // verify we slice strictly by `pages`.
        let block_table = vec![10, 11, 99, 20, 88, 77];
        let inputs = PagedKvHostInputs {
            cu_seqlens_q: &[0, 200, 300],
            seqused_k: &[200, 100],
            block_table: &block_table,
            slot_mapping: &vec![0; 300],
            positions: &(0..300).collect::<Vec<_>>(),
            page_size: 128,
            max_blocks_per_seq: 3,
        };
        let m = build_paged_kv_metadata_from_host(inputs);
        assert_eq!(m.prefill_qo_indptr, vec![0, 200, 300]);
        assert_eq!(m.prefill_kv_indptr, vec![0, 2, 3]);
        assert_eq!(m.prefill_kv_indices, vec![10, 11, 20]);
        assert_eq!(m.prefill_kv_last_page_len, vec![72, 100]);
        assert_eq!(m.num_prefill_tokens, 300);
    }

    /// Decode-only batch (every seq has q_len = 1). Prefill side
    /// stays empty; decode side accumulates one entry per seq.
    #[test]
    fn decode_only_batch() {
        // 3 seqs × max_blocks_per_seq=3. Per-seq kv_len + pages:
        //   seq 0: kv=300 → 3 pages → indices [7, 8, 9]
        //   seq 1: kv=100 → 1 page  → indices [10]
        //   seq 2: kv=200 → 2 pages → indices [11, 12]
        // Trailing cells are noise to verify pages-bounded slice.
        let block_table = vec![
            7, 8, 9, // seq 0
            10, 99, 99, // seq 1
            11, 12, 99, // seq 2
        ];
        let inputs = PagedKvHostInputs {
            cu_seqlens_q: &[0, 1, 2, 3],
            seqused_k: &[300, 100, 200],
            block_table: &block_table,
            slot_mapping: &[9 * 128 + 43, 10 * 128 + 99, 12 * 128 + 71],
            positions: &[299, 99, 199],
            page_size: 128,
            max_blocks_per_seq: 3,
        };
        let m = build_paged_kv_metadata_from_host(inputs);
        assert_eq!(m.prefill_qo_indptr, vec![0]);
        assert!(m.prefill_kv_indices.is_empty());
        assert!(m.prefill_kv_last_page_len.is_empty());
        assert_eq!(m.decode_kv_indptr, vec![0, 3, 4, 6]);
        assert_eq!(m.decode_kv_indices, vec![7, 8, 9, 10, 11, 12]);
        assert_eq!(m.decode_kv_last_page_len, vec![44, 100, 72]);
        assert_eq!(m.num_prefill_tokens, 0);
    }

    /// Mixed batch: one prefill seq followed by two decode seqs.
    /// The prefill+decode split is the megakernel's expected
    /// ordering; a decode-before-prefill batch must panic loudly
    /// (covered separately).
    #[test]
    fn mixed_prefill_then_decode() {
        let block_table = vec![10, 11, 0, 0, 20, 0, 0, 0, 30, 0, 0, 0];
        let inputs = PagedKvHostInputs {
            cu_seqlens_q: &[0, 200, 201, 202],
            seqused_k: &[200, 50, 60],
            block_table: &block_table,
            slot_mapping: &(0..202).map(|i| i as i32).collect::<Vec<_>>(),
            positions: &(0..202).map(|i| i as i32).collect::<Vec<_>>(),
            page_size: 128,
            max_blocks_per_seq: 4,
        };
        let m = build_paged_kv_metadata_from_host(inputs);
        // Prefill: 1 seq, 2 pages, last_page_len = 72.
        assert_eq!(m.prefill_qo_indptr, vec![0, 200]);
        assert_eq!(m.prefill_kv_indptr, vec![0, 2]);
        assert_eq!(m.prefill_kv_indices, vec![10, 11]);
        assert_eq!(m.prefill_kv_last_page_len, vec![72]);
        // Decode: 2 seqs, 1 page each.
        assert_eq!(m.decode_kv_indptr, vec![0, 1, 2]);
        assert_eq!(m.decode_kv_indices, vec![20, 30]);
        assert_eq!(m.decode_kv_last_page_len, vec![50, 60]);
        assert_eq!(m.num_prefill_tokens, 200);
        // Combined vectors carry every token verbatim.
        assert_eq!(m.kv_append_indices.len(), 202);
        assert_eq!(m.position_ids.len(), 202);
    }

    /// Decode seq before a prefill seq must panic — the
    /// megakernel's vector layout requires prefill-then-decode
    /// ordering.
    #[test]
    #[should_panic(expected = "ferrite expects prefill seqs first")]
    fn decode_before_prefill_panics() {
        let block_table = vec![10, 0, 0, 0, 20, 21, 0, 0];
        let inputs = PagedKvHostInputs {
            cu_seqlens_q: &[0, 1, 100],
            seqused_k: &[64, 99],
            block_table: &block_table,
            slot_mapping: &vec![0; 100],
            positions: &vec![0; 100],
            page_size: 128,
            max_blocks_per_seq: 4,
        };
        let _ = build_paged_kv_metadata_from_host(inputs);
    }
}
