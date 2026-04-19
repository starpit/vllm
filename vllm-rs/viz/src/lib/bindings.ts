import type {
  Arch,
  CanonicalVariant,
  StencilBlock,
  StencilExpr,
  StencilProgram,
  Variant,
  Workload,
} from "@/types";

/** Free-variable inventory for a stencil program. These are the
 * slots the math leaves open for the arch's bindings to fill:
 *
 *  - `weights`  — every `Weight { name, index }` the body references.
 *                 Same for llama and mistral (both read q_proj, etc.) but
 *                 the *loaded bytes* differ per variant. Useful to
 *                 confirm two arches consume the same weight set.
 *  - `scalars`  — every `scalar(<name>)` / `recip_scalar(<name>)` /
 *                 `sqrt(<name>)` — these bake a per-model scalar at
 *                 compile time (rope_theta, rms_norm_eps, the four
 *                 Granite multipliers, …).
 *  - `bounds`   — every `for` loop bound that resolves to a named
 *                 config symbol. `num_hidden_layers` alone usually.
 *
 * Externs (`input_ids`, `positions`, …) are intentionally omitted:
 * they're runtime inputs, not bindings. */
export interface FreeVariables {
  weights: Set<string>;
  scalars: Set<string>;
  bounds: Set<string>;
}

export function freeVariables(program: StencilProgram): FreeVariables {
  const out: FreeVariables = {
    weights: new Set(),
    scalars: new Set(),
    bounds: new Set(),
  };
  walkBlocks(program.blocks, out);
  return out;
}

function walkBlocks(blocks: StencilBlock[], out: FreeVariables) {
  for (const b of blocks) {
    if (b.kind === "assign") walkExpr(b.value, out);
    else if (b.kind === "for") {
      // Loop bounds that are symbolic names (not integer literals)
      // are arch-config-driven bindings.
      if (!/^\d+$/.test(b.start)) out.bounds.add(b.start);
      if (!/^\d+$/.test(b.end)) out.bounds.add(b.end);
      walkBlocks(b.body, out);
    } else if (b.kind === "if") {
      walkBlocks(b.then, out);
      walkBlocks(b.else, out);
    }
  }
}

function walkExpr(e: StencilExpr, out: FreeVariables) {
  if (e.kind === "call" || e.kind === "binop") {
    for (const c of e.children) walkExpr(c, out);
  } else if (e.kind === "weight") {
    out.weights.add(e.name);
  } else if (e.kind === "scalar_sym") {
    // Normalize `1/x` / `sqrt(x)` back to the raw symbol so the
    // binding surface is the set of config names, not the
    // transformations applied to them.
    const base = e.name
      .replace(/^1\//, "")
      .replace(/^sqrt\((.*)\)$/, "$1");
    out.scalars.add(base);
  }
}

/** Structural hash of the stencil program — ignores locals, weight
 * names, scalar names (just the op topology + literal values). Two
 * arches with byte-identical `forward!` bodies (llama / mistral) will
 * hash the same. A small tweak (granite's extra `*scalar(...)`)
 * changes the hash. */
export function programHash(program: StencilProgram): string {
  const parts: string[] = [];
  hashBlocks(program.blocks, parts);
  return parts.join("|");
}

function hashBlocks(blocks: StencilBlock[], out: string[]) {
  for (const b of blocks) {
    if (b.kind === "assign") {
      out.push(`A${b.targets.length}(`);
      hashExpr(b.value, out);
      out.push(")");
    } else if (b.kind === "for") {
      out.push("FOR[");
      hashBlocks(b.body, out);
      out.push("]");
    } else if (b.kind === "if") {
      out.push("IF[");
      hashBlocks(b.then, out);
      out.push(";");
      hashBlocks(b.else, out);
      out.push("]");
    }
  }
}

function hashExpr(e: StencilExpr, out: string[]) {
  switch (e.kind) {
    case "call":
      out.push(`C:${e.op}(`);
      for (const c of e.children) hashExpr(c, out);
      out.push(")");
      break;
    case "binop":
      out.push(`B:${e.op}(`);
      for (const c of e.children) hashExpr(c, out);
      out.push(")");
      break;
    case "local":
      // Name stripped — two different programs can rename locals and
      // still be structurally equivalent.
      out.push("L");
      break;
    case "weight":
      // Name stripped; index presence preserved (indexed vs not is
      // structural).
      out.push(e.index !== null ? "W[]" : "W");
      break;
    case "extern":
      out.push(`E:${e.name}${e.index !== null ? "[]" : ""}`);
      break;
    case "scalar":
      out.push(`S:${e.value}`);
      break;
    case "scalar_sym":
      // Normalize transformation prefixes (see walkExpr); hash
      // ignores the symbol name itself because llama vs mistral
      // agree on shape without naming the same scalars.
      out.push("Sm");
      break;
  }
}

/** Cluster arches by their stencil hash. Arches in the same bucket
 * share a forward body (template). */
export function clusterArchesByMath(arches: Arch[]): Map<string, Arch[]> {
  const groups = new Map<string, Arch[]>();
  for (const a of arches) {
    if (!a.program) continue;
    const h = programHash(a.program);
    let g = groups.get(h);
    if (!g) {
      g = [];
      groups.set(h, g);
    }
    g.push(a);
  }
  return groups;
}

/** For a single arch: collapse the per-variant (bounds, scalars)
 * maps into a per-free-variable summary. Each row is one free
 * variable; the value is the set of distinct values that appear
 * across variants. */
export interface BindingRow {
  /** The free variable's name (e.g. `rope_theta`, `num_hidden_layers`). */
  name: string;
  /** Underlying category: `weight` / `scalar` / `bound`. */
  kind: "weight" | "scalar" | "bound";
  /** Tier of how this binding flows into outputs:
   *  - `dsl`    — referenced in the `forward!` body directly (weight
   *               paths, loop bounds, explicit scalar() refs). Math
   *               view names these.
   *  - `kernel` — consumed by kernels the DSL calls (rms_norm_eps by
   *               rmsnorm, rope_theta by rope_append, head_dim by
   *               attention scale). Not in the math view but drives
   *               output numerically.
   *  - `meta`   — HF config fields the loader/tokenizer may read
   *               (vocab_size, bos/eos token ids, max_position,
   *               attention_dropout). Do not affect the forward
   *               computation. Hidden by default in the bindings
   *               view.
   */
  tier: "dsl" | "kernel" | "meta";
  /** Distinct values across variants. For weights, the "value" is a
   * synthetic "declared / absent" marker — one row per weight path,
   * string "∃" when present. For scalars/bounds, the actual numeric
   * or string values. */
  values: string[];
}

/** Names known to be consumed by kernels the forward! body invokes.
 * Curated whitelist — anything not here and not DSL-referenced falls
 * to the `meta` tier. Keep in sync as new kernel behaviors land. */
const KERNEL_CONSUMED = new Set([
  // rmsnorm / layernorm kernels
  "rms_norm_eps",
  "layer_norm_eps",
  // rope_append / rotary cache
  "rope_theta",
  "rope_scaling",
  "original_max_position_embeddings", // llama-3.1 rope scaling input
  "partial_rotary_factor",
  // attention kernel: head_dim drives 1/sqrt(d) scale
  "head_dim",
  "num_attention_heads",
  "num_key_value_heads",
  "hidden_size",
  "intermediate_size",
  // sliding-window attention mask
  "sliding_window",
  // logit soft-capping (gemma2)
  "attn_logit_softcapping",
  "final_logit_softcapping",
  // tie_word_embeddings affects lm_head path (loader, not forward
  // math), but it IS a real semantic knob — mark it kernel-ish.
  "tie_word_embeddings",
]);

/** Flatten a variant's bounds + scalars into one name→value map so
 * UI code can read `params["rope_theta"]` without caring which table
 * it lives in. Scalars win on collision (they carry full precision;
 * bounds round down). Returned map is empty for alias variants (no
 * bounds field in the dump). */
export function paramsFor(variant: Variant): Record<string, number> {
  const out: Record<string, number> = {};
  const bounds = (variant as { bounds?: Record<string, number> }).bounds;
  const scalars = (variant as { scalars?: Record<string, number> }).scalars;
  if (bounds) for (const [k, v] of Object.entries(bounds)) out[k] = v;
  if (scalars) for (const [k, v] of Object.entries(scalars)) out[k] = v;
  return out;
}

export function bindingRowsFor(arch: Arch): BindingRow[] {
  if (!arch.program) return [];
  const free = freeVariables(arch.program);
  const rows: BindingRow[] = [];

  // Weights: declared paths are all DSL-referenced by construction
  // (every `Weight` input comes from the forward body).
  for (const w of [...free.weights].sort()) {
    rows.push({ name: w, kind: "weight", tier: "dsl", values: ["∃"] });
  }

  const canonicals = arch.variants.filter(
    (v): v is Variant & {
      bounds: Record<string, number>;
      scalars?: Record<string, number>;
    } => typeof v.bounds === "object",
  );

  const allBoundNames = new Set<string>();
  const allScalarNames = new Set<string>();
  for (const v of canonicals) {
    for (const k of Object.keys(v.bounds)) allBoundNames.add(k);
    for (const k of Object.keys(v.scalars ?? {})) allScalarNames.add(k);
  }

  const tierOf = (name: string): "dsl" | "kernel" | "meta" => {
    if (free.bounds.has(name) || free.scalars.has(name)) return "dsl";
    if (KERNEL_CONSUMED.has(name)) return "kernel";
    return "meta";
  };

  for (const b of [...allBoundNames].sort()) {
    const vs = new Set<string>();
    for (const v of canonicals) {
      const val = v.bounds[b];
      if (val !== undefined) vs.add(String(val));
    }
    if (vs.size > 0)
      rows.push({ name: b, kind: "bound", tier: tierOf(b), values: [...vs] });
  }
  for (const s of [...allScalarNames].sort()) {
    if (allBoundNames.has(s)) continue;
    const vs = new Set<string>();
    for (const v of canonicals) {
      const val = v.scalars?.[s];
      if (val !== undefined) vs.add(formatScalar(val));
    }
    if (vs.size > 0)
      rows.push({ name: s, kind: "scalar", tier: tierOf(s), values: [...vs] });
  }

  return rows;
}

/** Map each stencil assign (indexed by its source-order position —
 * the same `indexAssigns` order StencilView uses) to the
 * implementation the solver picked for it.
 *
 * Stencil is pre-unroll; FUF is post-unroll. Match by walking both
 * in order: for each stencil assign, advance the FUF-tile cursor
 * until a tile with the matching op is found, record its picked
 * impl, move on. After the first loop iteration's worth of tiles is
 * consumed the walk naturally stops (stencil list is exhausted),
 * which is correct — every later iteration picks the same impls.
 *
 * Known false-ish positives:
 *   - Granite's `add(oproj * scalar(mult), ...)` — the `Mul` by a
 *     scalar typically folds into the preceding gemm/attention impl
 *     and never becomes its own FUF tile. We keep Mul in the match
 *     list anyway; when no matching tile exists, the assign just
 *     gets no impl chip.
 *   - Pure-passthrough assigns (`logits = logits_raw * recip_scalar`)
 *     might not produce a tile either; same handling.
 */
export function stencilImplMap(
  program: StencilProgram,
  variant: CanonicalVariant,
  workload: Workload,
): Map<number, string> {
  const out = new Map<number, string>();
  const stencilOps = flattenStencilOps(program);
  if (stencilOps.length === 0) return out;

  const sgImpl = new Map<number, string>();
  for (const s of workload.subgraph_impl) sgImpl.set(s.sg, s.impl_name);

  let tileCursor = 0;
  const tiles = variant.fuf.nodes;
  for (const { idx, op } of stencilOps) {
    while (tileCursor < tiles.length && tiles[tileCursor].op !== op) {
      tileCursor++;
    }
    if (tileCursor >= tiles.length) break;
    const sg = workload.tile_subgraph[tileCursor];
    const impl = sg !== null && sg !== undefined ? sgImpl.get(sg) : undefined;
    if (impl) out.set(idx, impl);
    tileCursor++;
  }
  return out;
}

/** Walk `program` and emit `{idx, op}` for every assign whose value
 * has an outermost Call or BinOp. `idx` matches `indexAssigns`. */
function flattenStencilOps(
  program: StencilProgram,
): { idx: number; op: string }[] {
  const out: { idx: number; op: string }[] = [];
  let counter = 0;
  function walk(blocks: StencilBlock[]) {
    for (const b of blocks) {
      if (b.kind === "assign") {
        const i = counter++;
        const op = outerCallOp(b.value);
        if (op !== null) out.push({ idx: i, op });
      } else if (b.kind === "for") {
        walk(b.body);
      } else if (b.kind === "if") {
        walk(b.then);
        walk(b.else);
      }
    }
  }
  walk(program.blocks);
  return out;
}

function outerCallOp(e: StencilExpr): string | null {
  if (e.kind === "call") return e.op;
  if (e.kind === "binop") return e.op === "*" ? "Mul" : "Add";
  return null;
}

function formatScalar(v: number): string {
  if (Number.isInteger(v)) return String(v);
  // Rope thetas etc. tend to be round-ish; don't show gratuitous
  // precision for values like `1e-05`.
  if (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e6) return v.toExponential(2);
  return String(v);
}
