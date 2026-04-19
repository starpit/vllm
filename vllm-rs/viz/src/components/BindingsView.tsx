import { useMemo, useState } from "react";
import type { Arch } from "@/types";
import { bindingRowsFor } from "@/lib/bindings";

interface Props {
  arches: Arch[];
}

type Tier = "dsl" | "kernel" | "meta";

interface TableRow {
  name: string;
  kind: "weight" | "scalar" | "bound";
  tier: Tier;
  perArch: (string[] | undefined)[];
}

/** Bindings: what binds to the free variables of the `forward!` math
 * such that the *same* math body (e.g. llama/mistral) becomes
 * different models. Rows grouped by tier:
 *
 *  1. DSL-referenced — the math view literally names these.
 *  2. Kernel-implicit — the DSL's kernels read these, even though
 *     they don't appear in the math view.
 *  3. Metadata — HF config fields the loader/tokenizer use; they
 *     don't touch the forward pass. Hidden by default.
 */
export function BindingsView({ arches }: Props) {
  const [showMeta, setShowMeta] = useState(false);
  const { rows } = useMemo(() => buildTable(arches), [arches]);

  const groups: { tier: Tier; label: string; caption: string }[] = [
    {
      tier: "dsl",
      label: "dsl bindings",
      caption: "referenced directly in the forward! body",
    },
    {
      tier: "kernel",
      label: "kernel bindings",
      caption:
        "consumed by the kernels the DSL calls (rms_norm_eps by rmsnorm, rope_theta by rope_append, …)",
    },
    {
      tier: "meta",
      label: "metadata",
      caption:
        "HF config fields used by the loader / tokenizer — do not affect the forward pass",
    },
  ];

  return (
    <div className="overflow-auto h-full text-[12px] font-mono">
      {groups.map((g) => {
        if (g.tier === "meta" && !showMeta) return null;
        const gRows = rows.filter((r) => r.tier === g.tier);
        if (gRows.length === 0) return null;
        return (
          <GroupSection
            key={g.tier}
            label={g.label}
            caption={g.caption}
            rows={gRows}
            arches={arches}
          />
        );
      })}
      {!showMeta && rows.some((r) => r.tier === "meta") && (
        <div className="px-3 py-2 text-[11px] text-ink-500">
          <button
            onClick={() => setShowMeta(true)}
            className="underline hover:text-ink-300"
          >
            show metadata rows (
            {rows.filter((r) => r.tier === "meta").length})
          </button>
        </div>
      )}
      {showMeta && (
        <div className="px-3 py-2 text-[11px] text-ink-500">
          <button
            onClick={() => setShowMeta(false)}
            className="underline hover:text-ink-300"
          >
            hide metadata
          </button>
        </div>
      )}
    </div>
  );
}

function GroupSection({
  label,
  caption,
  rows,
  arches,
}: {
  label: string;
  caption: string;
  rows: TableRow[];
  arches: Arch[];
}) {
  return (
    <div className="border-b border-ink-700">
      <div className="sticky top-0 z-10 bg-ink-800/95 backdrop-blur border-b border-ink-600 px-3 py-1.5">
        <div className="text-[11px] uppercase tracking-wider font-semibold text-ink-200">
          {label}
        </div>
        <div className="text-[10px] text-ink-500">{caption}</div>
      </div>
      <table className="w-full border-collapse">
        <thead>
          <tr className="text-ink-400">
            <th className="text-left px-3 py-1 font-medium w-[25%]">binding</th>
            {arches.map((a) => (
              <th key={a.arch} className="text-left px-3 py-1 font-medium">
                {a.arch}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const unique = uniqueValueSet(row.perArch);
            const differs = unique.size > 1;
            return (
              <tr
                key={`${row.kind}:${row.name}`}
                className={`border-t border-ink-700/60 ${
                  differs ? "bg-ink-800/40" : ""
                }`}
              >
                <td className="px-3 py-1.5 align-top whitespace-nowrap">
                  <KindBadge kind={row.kind} />
                  <span
                    className={`ml-2 ${differs ? "text-ink-100" : "text-ink-400"}`}
                  >
                    {row.name}
                  </span>
                </td>
                {row.perArch.map((c, i) => (
                  <td
                    key={i}
                    className={`px-3 py-1.5 align-top ${
                      differs ? "text-ink-100" : "text-ink-500"
                    }`}
                  >
                    {c === undefined ? (
                      <span className="text-ink-600">—</span>
                    ) : (
                      <span className="flex flex-wrap gap-1">
                        {c.map((v, j) => (
                          <span
                            key={j}
                            className="bg-ink-700 border border-ink-600 rounded px-1.5 py-0.5"
                          >
                            {v}
                          </span>
                        ))}
                      </span>
                    )}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function KindBadge({ kind }: { kind: "weight" | "scalar" | "bound" }) {
  const palette = {
    weight: "bg-emerald-900/60 text-emerald-300 border-emerald-700/50",
    scalar: "bg-violet-900/60 text-violet-300 border-violet-700/50",
    bound: "bg-amber-900/50 text-amber-300 border-amber-700/50",
  }[kind];
  return (
    <span
      className={`${palette} border rounded px-1 text-[9px] uppercase tracking-wider font-semibold`}
    >
      {kind}
    </span>
  );
}

function buildTable(arches: Arch[]): { rows: TableRow[] } {
  const perArch = new Map<string, Map<string, string[]>>();
  const meta = new Map<string, { kind: "weight" | "scalar" | "bound"; tier: Tier }>();
  for (const a of arches) {
    const archRows = bindingRowsFor(a);
    const byName = new Map<string, string[]>();
    for (const r of archRows) {
      byName.set(r.name, r.values);
      // Prefer the strictest tier seen — if one arch considers a
      // name DSL-referenced and another considers it kernel, DSL
      // wins (more informative placement).
      const prev = meta.get(r.name);
      const tierRank = { dsl: 0, kernel: 1, meta: 2 };
      if (!prev || tierRank[r.tier] < tierRank[prev.tier]) {
        meta.set(r.name, { kind: r.kind, tier: r.tier });
      }
    }
    perArch.set(a.arch, byName);
  }
  const rows: TableRow[] = [...meta.entries()]
    .map(([name, m]) => ({
      name,
      kind: m.kind,
      tier: m.tier,
      perArch: arches.map((a) => perArch.get(a.arch)?.get(name)),
    }))
    .sort((a, b) => {
      const kindOrder = { weight: 0, bound: 1, scalar: 2 };
      if (a.kind !== b.kind) return kindOrder[a.kind] - kindOrder[b.kind];
      return a.name.localeCompare(b.name);
    });
  return { rows };
}

function uniqueValueSet(cells: (string[] | undefined)[]): Set<string> {
  const s = new Set<string>();
  for (const c of cells) {
    if (!c) continue;
    s.add([...c].sort().join("|"));
  }
  return s;
}
