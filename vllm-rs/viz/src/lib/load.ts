import type { Arch } from "@/types";

// Hard-coded arch list. The build pipeline drops one JSON per
// `#[forward]`-annotated arch under public/arches/. We don't ship a
// directory listing, so the manifest is enumerated here. Adding a new
// arch = one line.
export const ARCH_NAMES = [
  "commandr",
  "gemma2",
  "gemma3",
  "granite",
  "llama",
  "mistral",
  "phi3",
  "qwen2",
  "qwen3",
] as const;

export type ArchName = (typeof ARCH_NAMES)[number];

export async function loadArch(name: ArchName): Promise<Arch> {
  const r = await fetch(`/arches/${name}.json`);
  if (!r.ok) {
    throw new Error(`fetch ${name}.json: ${r.status} ${r.statusText}`);
  }
  return r.json() as Promise<Arch>;
}

export async function loadAllArches(): Promise<Arch[]> {
  const out: Arch[] = [];
  for (const n of ARCH_NAMES) {
    try {
      out.push(await loadArch(n));
    } catch (e) {
      // Treat missing fixtures as a buildable-but-not-built arch
      // rather than a hard failure — the gallery still renders.
      console.warn(`skipping ${n}:`, e);
    }
  }
  return out;
}

export interface QuantTag {
  label: string;
  /** tailwind bg color */
  color: string;
}

/** Heuristic quant detection from variant name suffix. The proc-macro
 * synthesizes per-quant variant idents like `..._awq_gemm`,
 * `..._gptq_sym`, `..._ct_int4_sym`, `..._fp8`, `..._bnb4`.
 *
 * Returns the unique set across an arch's variants, sorted for
 * determinism. */
export function detectQuants(variantNames: string[]): QuantTag[] {
  const found = new Set<string>();
  for (const n of variantNames) {
    if (/_awq(_|$)/.test(n)) found.add("AWQ");
    else if (/_gptq(_|$)/.test(n)) found.add("GPTQ");
    else if (/_ct_/.test(n)) found.add("CT-INT4");
    else if (/_fp8(_|$)/.test(n)) found.add("FP8");
    else if (/_bnb4(_|$)/.test(n)) found.add("BNB4");
    else found.add("Dense");
  }
  return [...found].sort().map((label) => ({
    label,
    color:
      label === "Dense"
        ? "bg-ink-600"
        : label === "AWQ"
          ? "bg-emerald-700/70"
          : label === "GPTQ"
            ? "bg-cyan-700/70"
            : label === "FP8"
              ? "bg-rose-700/70"
              : label === "BNB4"
                ? "bg-amber-700/70"
                : "bg-violet-700/70",
  }));
}

/** Pull approximate parameter-count labels from variant names like
 * `llama_3_2_1b`, `qwen2_5_72b`. Returns sorted unique labels. */
export function detectSizes(variantNames: string[]): string[] {
  const sizes = new Set<string>();
  for (const n of variantNames) {
    const m = n.match(/(\d+)(b|m)(?:_|$)/i);
    if (m) sizes.add(`${m[1]}${m[2].toUpperCase()}`);
  }
  return [...sizes].sort((a, b) => {
    const pa = parseInt(a);
    const pb = parseInt(b);
    if (a.endsWith("M") && b.endsWith("B")) return -1;
    if (a.endsWith("B") && b.endsWith("M")) return 1;
    return pa - pb;
  });
}

/** Pull "product family / version" labels from variant names. The
 * convention is `<arch>_<majorver>[_<minorver>]_<sizeOrTag>…`, so the
 * digit-only prefix segments *after* the arch name identify the
 * version (e.g. `phi_3_5_mini_instruct` → "3.5"; `phi_4_reasoning` →
 * "4"; `llama_3_2_1b` → "3.2"). Makes heterogenous-family arches
 * like `phi3` visibly surface all the generations it covers. */
export function detectFamilies(
  archName: string,
  variantNames: string[],
): string[] {
  const arch = stripTrailingDigits(archName); // phi3 → phi, qwen2 → qwen
  const families = new Set<string>();
  for (const n of variantNames) {
    // Strip trailing _fp8_*/_awq_gemm/_gptq_*/_ct_int4_*/... quant
    // tags so "phi_3_5_mini_instruct_fp8_dynamic" still collapses
    // to the "3.5" family.
    const stripped = n.replace(
      /_(fp8|awq|gptq|ct|bnb4)(_[a-z0-9_]+)?$/i,
      "",
    );
    const tokens = stripped.split("_");
    // Drop any leading arch-prefix tokens. Example: arch `phi3` →
    // prefix `phi`, `3`, … — we want to find the version digits
    // AFTER the arch match.
    let i = 0;
    // Consume the arch token (may include its own digit suffix,
    // e.g. `qwen2` → tokens `[qwen, 2, …]`).
    if (tokens[i]?.toLowerCase() === arch) i++;
    // If the arch had a digit suffix (`phi3` → arch=`phi`, tokens
    // start `phi_3_…`), consume the following digit token.
    const digitAfter = /^\d+$/.test(tokens[i] ?? "");
    // Now collect a run of digit tokens — the version path.
    const version: string[] = [];
    if (digitAfter) {
      version.push(tokens[i]);
      i++;
      while (/^\d+$/.test(tokens[i] ?? "")) {
        // Stop at the size tag — `1b`, `72b`, `135m` — which has a
        // trailing letter, handled already by the digit-only check.
        version.push(tokens[i]);
        i++;
      }
    }
    if (version.length > 0) families.add(version.join("."));
  }
  return [...families].sort((a, b) => {
    // Natural sort on dotted-version strings (e.g. 3 < 3.2 < 4).
    const pa = a.split(".").map(Number);
    const pb = b.split(".").map(Number);
    for (let j = 0; j < Math.max(pa.length, pb.length); j++) {
      const da = pa[j] ?? 0;
      const db = pb[j] ?? 0;
      if (da !== db) return da - db;
    }
    return 0;
  });
}

function stripTrailingDigits(s: string): string {
  return s.replace(/\d+$/, "");
}
