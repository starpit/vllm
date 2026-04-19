import { useMemo } from "react";
import ReactFlow, {
  Background,
  Controls,
  Edge,
  Node,
  ReactFlowProvider,
} from "reactflow";
import type { CanonicalVariant, FufNode, Workload } from "@/types";

interface Props {
  variant: CanonicalVariant;
  /** which workload point to color the graph by; null = no coloring */
  workload: Workload | null;
  /** if true, render only the first ~80 nodes — keeps the small-multiples
   * thumbnail responsive. The zoomed view passes false. */
  thumbnail?: boolean;
}

/** ┌── Layout ──┐
 * Topological-depth layout. Each tile's depth = `1 + max(depth of any
 * tile-input)`. Depth becomes the row (top → bottom = forward pass);
 * within a depth, tiles stack horizontally. The resulting graph is a
 * vertical "depth chart" whose height visibly scales with the number
 * of unrolled transformer layers (≈ 7b: 32 deep; 70b: 80 deep), so
 * switching variants in the dropdown produces a visibly different
 * graph instead of a fitView-flattened smear.
 *
 * Wave-based layout (one column per BSP wave) was the original — but
 * for ferrite's typical SFUF where every subgraph holds one tile,
 * every tile gets a unique wave and the layout collapses to a single
 * giant horizontal line. fitView then squashes 7b/13b/70b to look
 * identical at viewport scale.
 */
function layoutNodes(
  fuf: { nodes: FufNode[] },
  workload: Workload | null,
  thumbnail: boolean,
): { nodes: Node[]; edges: Edge[] } {
  const nodes = fuf.nodes;
  const cap = thumbnail ? Math.min(nodes.length, 80) : nodes.length;

  const depth = new Array<number>(cap).fill(0);
  for (let i = 0; i < cap; i++) {
    let max = 0;
    for (const inp of nodes[i].inputs) {
      if (inp.kind === "tile" && inp.tile < cap) {
        max = Math.max(max, depth[inp.tile] + 1);
      }
    }
    depth[i] = max;
  }

  // ── Within each depth, lay tiles out horizontally in id order. ──
  const perDepth = new Map<number, number>();
  const COL_W = 220;
  const ROW_H = 36;

  // Subgraph → impl-name lookup for the current workload, so node
  // labels carry the actually-picked kernel (e.g. `attention →
  // flashinfer_attention_decode` at M=1, vs `attention →
  // attention_prefill_contiguous` at M=512). Without this, switching
  // workloads in the dropdown produces visually identical graphs even
  // though the solver picked different impls.
  const sgImpl = new Map<number, string>();
  if (workload) {
    for (const s of workload.subgraph_impl) sgImpl.set(s.sg, s.impl_name);
  }

  // depth → row (y), position-in-row → column (x). Top-down read =
  // forward-pass execution order; horizontal width within a row =
  // amount of parallelism at that depth.
  const rfNodes: Node[] = [];
  for (let i = 0; i < cap; i++) {
    const D = depth[i];
    const idx = perDepth.get(D) ?? 0;
    perDepth.set(D, idx + 1);
    const sg = workload?.tile_subgraph[i] ?? null;
    const impl = sg !== null && sg !== undefined ? sgImpl.get(sg) : undefined;
    const label = impl
      ? `${nodes[i].op}#${i}\n${impl}`
      : `${nodes[i].op}#${i}${sg !== null ? ` · sg${sg}` : ""}`;
    rfNodes.push({
      id: `${i}`,
      data: { label },
      position: { x: idx * COL_W, y: D * ROW_H },
      style: nodeStyle(nodes[i].op, sg),
      sourcePosition: "bottom" as any,
      targetPosition: "top" as any,
    });
  }

  // ── Edges: tile→tile inputs only. ──
  const rfEdges: Edge[] = [];
  for (let i = 0; i < cap; i++) {
    for (const inp of nodes[i].inputs) {
      if (inp.kind === "tile" && inp.tile < cap) {
        rfEdges.push({
          id: `${inp.tile}-${i}-${inp.slot}`,
          source: `${inp.tile}`,
          target: `${i}`,
        });
      }
    }
  }
  return { nodes: rfNodes, edges: rfEdges };
}

/** Per-op color hint. Pure visual — keeps the eye moving. */
function nodeStyle(op: string, _sg: number | null): React.CSSProperties {
  const family =
    op === "Embed" || op === "Gemm"
      ? "compute"
      : op === "Attention" || op === "SlidingAttention" || op === "RopeAppend"
        ? "attention"
        : op === "RmsNorm" || op === "LayerNorm"
          ? "norm"
          : op === "Add" || op === "Mul" || op === "Silu" || op === "Gelu"
            ? "elem"
            : "other";
  const bg = {
    compute: "#26303f",
    attention: "#3a2230",
    norm: "#283426",
    elem: "#2a2735",
    other: "#1a1a1f",
  }[family]!;
  const border = {
    compute: "#5fb3ff",
    attention: "#ffb86b",
    norm: "#86c08c",
    elem: "#a78bfa",
    other: "#3a3a44",
  }[family]!;
  return {
    background: bg,
    border: `1px solid ${border}`,
    borderRadius: 4,
    color: "#f4f4f7",
    padding: "4px 8px",
    fontSize: 11,
  };
}

export function FufView({ variant, workload, thumbnail = false }: Props) {
  const { nodes, edges } = useMemo(
    () => layoutNodes(variant.fuf, workload, thumbnail),
    [variant, workload, thumbnail],
  );
  // Keying on (variant, workload, node count) forces ReactFlow to
  // tear down + remount on every dropdown change. Without a key,
  // ReactFlow treats `nodes` as initial state only (uncontrolled
  // mode) — subsequent prop changes don't propagate. Including
  // `nodes.length` makes it impossible to mistake one variant for
  // another (483 / 603 / 1203 tiles for llama 7b / 13b / 70b).
  const key = `${variant.name}:${workload ? `${workload.num_tokens}-${workload.sk_bucket}` : "none"}:${nodes.length}`;
  return (
    <ReactFlowProvider>
      <ReactFlow
        key={key}
        nodes={nodes}
        edges={edges}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        proOptions={{ hideAttribution: true }}
        nodesDraggable={!thumbnail}
        nodesConnectable={false}
        elementsSelectable={!thumbnail}
        panOnDrag={!thumbnail}
        zoomOnScroll={!thumbnail}
        zoomOnPinch={!thumbnail}
        zoomOnDoubleClick={false}
      >
        {!thumbnail && <Background gap={16} color="#26262d" />}
        {!thumbnail && <Controls showInteractive={false} />}
      </ReactFlow>
    </ReactFlowProvider>
  );
}
