// SPDX-License-Identifier: Apache-2.0
//! Scheduled BSP wave-front megakernel codegen.
//!
//! Layout of the rendered `.cu` file:
//! ```text
//! constexpr uint32_t NUM_WAVES;
//! constexpr uint32_t NUM_CTAS;
//! constexpr uint32_t NUM_NODES;   // sizes the validation tick array
//!
//! struct TileOp { phase, layer, row, col };          // 16 B (padded)
//! __device__ const TileOp WAVE_OPS[NUM_NODES];       // packed: per wave, per CTA, in order
//! __device__ const uint32_t WAVE_CTA_OFFSETS[
//!     NUM_WAVES * (NUM_CTAS + 1)];                   // prefix-sum table
//! __device__ const uint32_t NODE_ID_FOR_OP[NUM_NODES]; // parallel to WAVE_OPS
//! ```
//!
//! Per CTA per wave: stream of TileOp records contiguous in WAVE_OPS,
//! indexed via WAVE_CTA_OFFSETS\[wave * (NUM_CTAS+1) + cta..+ cta+1\].
//! After each wave, all CTAs participate in one grid barrier (atomic-
//! counter style) before moving to the next.
//!
//! Everything above lives in `templates/scheduled/megakernel.cu`,
//! rendered through askama. This file just builds the
//! [`MegakernelCtx`] from the reified DAG + wave schedule and calls
//! `.render()`. No `writeln!`-driven C++ emission.

use std::fmt::Write as _;

use askama::Template;

use crate::kernel_library::{BoundKernel, CoalescedDag};
use crate::schedule::WaveSchedule;

mod templates;

use templates::MegakernelCtx;

/// Phase 3b-bsp — emit a complete `.cu` file containing the wave program
/// data + tile bodies + grid-barrier megakernel + extern "C" launchers.
///
/// `kv_page_size` is the slots-per-page for the paged KV cache.
/// Pages-per-layer is derived as `ceil(seq_len / kv_page_size)`.
///
/// `name` is the per-variant suffix woven into the namespace and the
/// extern "C" symbol root, so multiple variants coexist in one TU:
///   namespace pfl_sched_<name>
///   extern "C" void launch_scheduled_megakernel_<name>(...)
///   extern "C" unsigned scheduled_megakernel_<name>_num_nodes()
pub fn emit_scheduled_megakernel_cu(
    dag: &CoalescedDag,
    sched: &WaveSchedule,
    kv_page_size: u32,
    name: &str,
) -> String {
    let num_nodes = dag.nodes.len() as u32;
    let num_ctas = sched.num_ctas;
    let num_waves = sched.waves.len() as u32;

    // ── Pack the per-wave per-CTA op streams into one flat array. ──
    // Layout order: wave 0 / cta 0..N-1; wave 1 / cta 0..N-1; ...
    //
    // Phase C1: only `HandWrittenRowTile` is registered, so every coalesced
    // node carries a (phase, layer, row, col) we can lift directly into the
    // existing WAVE_OPS schema. Phase C2 will add a kernel-tagged variant
    // when `FlashInferAttentionLayer` lands and the schema needs to grow.
    let mut ops: Vec<(
        u32, /*phase*/
        u32, /*layer*/
        u32, /*row*/
        u32, /*col*/
    )> = Vec::with_capacity(num_nodes as usize);
    let mut node_ids: Vec<u32> = Vec::with_capacity(num_nodes as usize);
    let offsets_len = (num_waves * (num_ctas + 1)) as usize;
    let mut wave_cta_offsets: Vec<u32> = Vec::with_capacity(offsets_len);

    let mut cursor: u32 = 0;
    for wave in &sched.waves {
        for cta in 0..(num_ctas as usize) {
            wave_cta_offsets.push(cursor);
            for nid in &wave.cta_nodes[cta] {
                let nd = &dag.nodes[nid.0 as usize];
                // The WAVE_OPS schema is now kernel-tagged: the first
                // u32 is `BoundKernel::kernel_tag()` rather than a phase
                // index. For HandWrittenRowTile bindings the tag is
                // numerically identical to the old phase tag (0..7), so
                // the table is byte-equivalent to pre-C2b-step-1
                // output. For FlashInferAttentionLayer the tag is 8 and
                // (row, col) are unused (the per-row work is rolled up
                // into the runner) — the megakernel's dispatch switch
                // grows a matching `case 8` arm.
                let tag = nd.kernel.kernel_tag();
                let (layer, row, col) = match nd.kernel {
                    BoundKernel::HandWrittenRowTile {
                        layer, row, col, ..
                    } => (layer as u32, row as u32, col as u32),
                    BoundKernel::FlashInferAttentionLayer { layer } => (layer as u32, 0, 0),
                };
                ops.push((tag, layer, row, col));
                node_ids.push(nid.0);
                cursor += 1;
            }
        }
        // Sentinel "end of last cta in this wave" for clean range queries.
        wave_cta_offsets.push(cursor);
    }
    // `cursor` is the total op stream length, which can exceed
    // `num_nodes` when the schedule replicates wave-cooperative
    // bindings across all CTAs in a wave (see schedule.rs).
    debug_assert!(cursor >= num_nodes);
    debug_assert_eq!(wave_cta_offsets.len(), offsets_len, "offset table size");
    let num_ops = cursor;

    let pages_per_layer = dag.dims.seq_len.div_ceil(kv_page_size);

    let ctx = MegakernelCtx {
        name,
        num_nodes,
        num_ops,
        num_waves,
        num_ctas,
        wave_ops_table: render_wave_ops_table(&ops),
        node_id_for_op_table: render_u32_table_chunked(&node_ids),
        wave_cta_offsets_table: render_u32_table_chunked(&wave_cta_offsets),
        model_num_layers: dag.dims.num_layers,
        model_hidden_dim: dag.dims.hidden_dim,
        model_intermediate: dag.dims.intermediate_dim,
        model_num_attn_h: dag.dims.num_attn_heads,
        model_num_kv_h: dag.dims.num_kv_heads,
        model_head_dim: dag.dims.head_dim,
        model_seq_len: dag.dims.seq_len,
        model_row_tile: dag.tiles.row_tile,
        model_qkv_col_tile: dag.tiles.qkv_col_tile,
        model_o_col_tile: dag.tiles.o_col_tile,
        model_gate_up_col_tile: dag.tiles.gate_up_col_tile,
        model_down_col_tile: dag.tiles.down_col_tile,
        model_kv_page_size: kv_page_size,
        model_pages_per_layer: pages_per_layer,
    };
    ctx.render().expect("scheduled megakernel template render")
}

/// Render `WAVE_OPS` as `  { phase, layer, row, col },\n` per entry.
fn render_wave_ops_table(ops: &[(u32, u32, u32, u32)]) -> String {
    let mut out = String::with_capacity(ops.len() * 24);
    for (phase, layer, row, col) in ops {
        writeln!(out, "  {{ {}, {}, {}, {} }},", phase, layer, row, col).unwrap();
    }
    out
}

/// Render a u32 array as `  v, v, v, ..., v,\n` chunked 16-per-line.
fn render_u32_table_chunked(vals: &[u32]) -> String {
    const PER_LINE: usize = 16;
    let mut out = String::with_capacity(vals.len() * 6);
    for chunk in vals.chunks(PER_LINE) {
        out.push_str("  ");
        for v in chunk {
            write!(out, "{},", v).unwrap();
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_library::coalesce;
    use crate::reified_dag::{LlamaDims, ReifiedDag, TileSizes};
    use crate::schedule::{CostModel, partition_into_waves};

    fn tiny_dims() -> LlamaDims {
        LlamaDims {
            num_layers: 2,
            hidden_dim: 256,
            intermediate_dim: 512,
            num_attn_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            seq_len: 32,
        }
    }

    #[test]
    fn emit_smoke() {
        let reified = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let dag = coalesce(&reified);
        let cost = CostModel::from_dag(&dag);
        let sched = partition_into_waves(&dag, 4, &cost, 100);
        let cpp = emit_scheduled_megakernel_cu(&dag, &sched, 16, "tiny");

        assert!(cpp.contains("namespace pfl_sched_tiny"));
        assert!(cpp.contains("WAVE_OPS"));
        assert!(cpp.contains("WAVE_CTA_OFFSETS"));
        assert!(cpp.contains("NODE_ID_FOR_OP"));
        assert!(cpp.contains("static_assert(sizeof(TileOp) == 16"));
        let n = dag.nodes.len();
        assert!(cpp.contains(&format!("NUM_NODES = {n}")));
        let w = sched.waves.len();
        assert!(cpp.contains(&format!("NUM_WAVES = {w}")));
    }

    #[test]
    fn op_count_matches_node_count() {
        let reified = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let dag = coalesce(&reified);
        let cost = CostModel::from_dag(&dag);
        let sched = partition_into_waves(&dag, 4, &cost, 100);

        let mut total = 0u32;
        for w in &sched.waves {
            for c in &w.cta_nodes {
                total += c.len() as u32;
            }
        }
        assert_eq!(total as usize, dag.nodes.len());
    }
}
