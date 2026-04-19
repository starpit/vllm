import type { Arch, StencilBlock, StencilExpr } from "@/types";

/** Order arches so structurally similar ones land in adjacent grid
 * cells. The order across runs is stable for a given input set —
 * deterministic seed (`alphabetical first`) + greedy
 * nearest-neighbor by jaccard distance over op-multisets.
 *
 * Why op-multiset:
 *  - Cheap to compute, no parse-tree walk.
 *  - Captures the "what kernels does this arch use, and how often"
 *    intuition: llama and mistral collapse together; granite (extra
 *    `mul`s for residual scaling) sits next to llama; gemma3
 *    (sliding-attention) sits with attention-heavy peers.
 *
 * Better signals (subtree edit distance, normalized-tree hashing) are
 * a follow-up; this is the dumb-but-useful baseline. */
export function sortArchesBySimilarity(arches: Arch[]): Arch[] {
  if (arches.length <= 2) return arches;

  // ── 1. Build op-count vectors per arch. ──
  const vectors = arches.map((a) => opMultiset(a));

  // ── 2. Greedy nearest-neighbor walk. ──
  // Seed with the alphabetically-first arch so the order is
  // deterministic across reloads. At each step, pick the unvisited
  // neighbor closest (in jaccard distance) to the most recently
  // placed arch. Ties broken by arch name for determinism.
  const indexByArchName = new Map<string, number>();
  arches.forEach((a, i) => indexByArchName.set(a.arch, i));
  const seedName = [...arches.map((a) => a.arch)].sort()[0];
  const seed = indexByArchName.get(seedName)!;

  const visited = new Set<number>([seed]);
  const order: number[] = [seed];

  while (order.length < arches.length) {
    const last = order[order.length - 1];
    let best: number = -1;
    let bestDist = Infinity;
    for (let i = 0; i < arches.length; i++) {
      if (visited.has(i)) continue;
      const d = jaccardDistance(vectors[last], vectors[i]);
      if (
        d < bestDist ||
        (d === bestDist && (best === -1 || arches[i].arch < arches[best].arch))
      ) {
        best = i;
        bestDist = d;
      }
    }
    visited.add(best);
    order.push(best);
  }

  return order.map((i) => arches[i]);
}

/** Walk an arch's stencil program (or fall back to FUF op counts if
 * the program isn't present) and tally how many times each op
 * appears. Includes binops (`*`, `+`) so granite's extra residual
 * scaling differentiates from llama. */
function opMultiset(arch: Arch): Map<string, number> {
  const counts = new Map<string, number>();
  const bump = (k: string) => counts.set(k, (counts.get(k) ?? 0) + 1);

  if (arch.program) {
    walkBlocks(arch.program.blocks, bump);
    return counts;
  }
  // Schema-version-1 fallback: tally ops from the first canonical's FUF.
  const v = arch.variants.find((v) => "fuf" in v);
  if (v && "fuf" in v) {
    for (const n of v.fuf.nodes) bump(n.op);
  }
  return counts;
}

function walkBlocks(blocks: StencilBlock[], bump: (k: string) => void) {
  for (const b of blocks) {
    if (b.kind === "for") walkBlocks(b.body, bump);
    else if (b.kind === "if") {
      walkBlocks(b.then, bump);
      walkBlocks(b.else, bump);
    } else {
      walkExpr(b.value, bump);
    }
  }
}

function walkExpr(e: StencilExpr, bump: (k: string) => void) {
  if (e.kind === "call") {
    bump(e.op);
    for (const c of e.children) walkExpr(c, bump);
  } else if (e.kind === "binop") {
    bump(`binop:${e.op}`);
    for (const c of e.children) walkExpr(c, bump);
  }
  // Leaves (local/weight/extern/scalar/scalar_sym) don't bump — they
  // don't differentiate architectures structurally.
}

/** Weighted jaccard distance over multisets. Returns 0 for identical
 * op-bags, 1 for disjoint. */
function jaccardDistance(
  a: Map<string, number>,
  b: Map<string, number>,
): number {
  const keys = new Set([...a.keys(), ...b.keys()]);
  let inter = 0;
  let union = 0;
  for (const k of keys) {
    const av = a.get(k) ?? 0;
    const bv = b.get(k) ?? 0;
    inter += Math.min(av, bv);
    union += Math.max(av, bv);
  }
  if (union === 0) return 0;
  return 1 - inter / union;
}
