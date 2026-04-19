// Mirror of the JSON schema emitted by `viz_dump.rs` (schema_version: 1).

export type FufInput =
  | { kind: "tile"; tile: number; slot: number }
  | {
      kind: "weight";
      name: string;
      index: number | null;
      storage: string;
    }
  | { kind: "extern"; name: string; index: number | null }
  | { kind: "scalar"; value: number };

export type Dim = number | string | { mul: Dim[] };
export type Shape = Dim[];

export interface FufNode {
  id: number;
  op: string;
  inputs: FufInput[];
  outputs: Shape[];
}

export interface Fuf {
  nodes: FufNode[];
}

export interface SubgraphImpl {
  sg: number;
  impl_id: number;
  impl_name: string;
}

export interface Workload {
  num_tokens: number;
  sk_bucket: number;
  predicted_us: number;
  num_subgraphs: number;
  num_waves: number;
  /** length === fuf.nodes.length; null = unclaimed (shouldn't happen on success) */
  tile_subgraph: (number | null)[];
  subgraph_impl: SubgraphImpl[];
  /** wave[i] = list of subgraph ids in wave i */
  waves: number[][];
}

export interface CanonicalVariant {
  name: string;
  source_stem: string;
  tie_word_embeddings: boolean;
  bounds: Record<string, number>;
  scalars: Record<string, number>;
  fuf: Fuf;
  workloads: Workload[];
}

export interface AliasVariant {
  name: string;
  source_stem: string;
  tie_word_embeddings: boolean;
  bounds: Record<string, number>;
  scalars: Record<string, number>;
  /** name of the canonical sibling whose forward this variant aliases */
  canonical: string;
}

export type Variant = CanonicalVariant | AliasVariant;

export function isCanonical(v: Variant): v is CanonicalVariant {
  return (v as CanonicalVariant).fuf !== undefined;
}

// ── Stencil program (pre-unroll, ~30 nodes, used for the colored
// chip view) ─────────────────────────────────────────────────────

/** A node in the assign-statement AST tree — recursive. Composite
 * nodes (call/binop) carry `children`; leaves carry their own data. */
export type StencilExpr =
  | { kind: "call"; op: string; children: StencilExpr[] }
  | { kind: "binop"; op: "*" | "+"; children: StencilExpr[] }
  | { kind: "local"; name: string }
  | { kind: "extern"; name: string; index: string | null }
  | { kind: "weight"; name: string; index: string | null }
  | { kind: "scalar"; value: number }
  | { kind: "scalar_sym"; name: string };

export type StencilBlock =
  | {
      kind: "assign";
      targets: string[];
      value: StencilExpr;
    }
  | {
      kind: "for";
      ivar: string;
      start: string;
      end: string;
      body: StencilBlock[];
    }
  | {
      kind: "if";
      cond: string;
      then: StencilBlock[];
      else: StencilBlock[];
    };

export interface StencilProgram {
  blocks: StencilBlock[];
}

export interface Arch {
  schema_version: number;
  arch: string;
  hf_arches: string[];
  dsl_source: string;
  /** Optional: only present in schema_version >= 2. */
  program?: StencilProgram;
  variants: Variant[];
}
