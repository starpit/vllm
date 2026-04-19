import { useMemo } from "react";
import ReactDiffViewer, { DiffMethod } from "react-diff-viewer-continued";
import type { Arch } from "@/types";
import { isCanonical } from "@/types";

interface Props {
  a: Arch;
  b: Arch;
  onClose: () => void;
}

/** Side-by-side VSCode-style diff of the two arches' DSL bodies, plus
 * a per-op count delta. The DSL is the "source of truth" people
 * actually want to compare; the FUF graph diff (matched / inserted /
 * deleted nodes) needs a tree-edit-distance pass and lives in v2. */
export function DiffView({ a, b, onClose }: Props) {
  const opCountsA = useOpCounts(a);
  const opCountsB = useOpCounts(b);
  const allOps = useMemo(() => {
    const s = new Set([...Object.keys(opCountsA), ...Object.keys(opCountsB)]);
    return [...s].sort();
  }, [opCountsA, opCountsB]);

  return (
    <div className="fixed inset-0 z-50 bg-ink-900/95 flex flex-col">
      <header className="flex items-center justify-between px-4 py-2 border-b border-ink-600 shrink-0">
        <h2 className="text-lg font-semibold">
          diff: <span className="font-mono text-accent-load">{a.arch}</span>{" "}
          <span className="text-ink-400">↔</span>{" "}
          <span className="font-mono text-accent-compute">{b.arch}</span>
        </h2>
        <button
          onClick={onClose}
          className="text-ink-200 hover:text-ink-100 px-2 py-1 text-sm border border-ink-500 rounded"
        >
          close (esc)
        </button>
      </header>
      <div className="flex-1 min-h-0 overflow-auto bg-ink-900">
        <ReactDiffViewer
          oldValue={a.dsl_source}
          newValue={b.dsl_source}
          splitView
          compareMethod={DiffMethod.WORDS}
          leftTitle={a.arch}
          rightTitle={b.arch}
          useDarkTheme
          styles={diffStyles}
        />
      </div>
      <footer className="border-t border-ink-600 shrink-0 max-h-56 overflow-auto bg-ink-800">
        <table className="w-full text-xs font-mono">
          <thead className="sticky top-0 bg-ink-800">
            <tr className="text-ink-400">
              <th className="text-left px-3 py-1.5 font-medium">op</th>
              <th className="text-right px-3 py-1.5 font-medium">{a.arch}</th>
              <th className="text-right px-3 py-1.5 font-medium">{b.arch}</th>
              <th className="text-right px-3 py-1.5 font-medium">Δ</th>
            </tr>
          </thead>
          <tbody>
            {allOps.map((op) => {
              const ca = opCountsA[op] ?? 0;
              const cb = opCountsB[op] ?? 0;
              const d = cb - ca;
              return (
                <tr key={op} className="border-t border-ink-700">
                  <td className="px-3 py-1">{op}</td>
                  <td className="px-3 py-1 text-right">{ca || "—"}</td>
                  <td className="px-3 py-1 text-right">{cb || "—"}</td>
                  <td
                    className={`px-3 py-1 text-right ${
                      d > 0
                        ? "text-emerald-400"
                        : d < 0
                          ? "text-rose-400"
                          : "text-ink-400"
                    }`}
                  >
                    {d > 0 ? `+${d}` : d || "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </footer>
    </div>
  );
}

/** Inline override for react-diff-viewer's defaults: pull the palette
 * onto our ink-* scale so the dark-theme yellow/green/red doesn't
 * clash with the rest of the app. */
const diffStyles = {
  variables: {
    dark: {
      diffViewerBackground: "#0a0a0b",
      diffViewerColor: "#f4f4f7",
      addedBackground: "#0d3a1f",
      addedColor: "#a8f0c4",
      removedBackground: "#3a0d1d",
      removedColor: "#f7b8c4",
      wordAddedBackground: "#1a6b3f",
      wordRemovedBackground: "#7a1b34",
      addedGutterBackground: "#102a1c",
      removedGutterBackground: "#2a0e18",
      gutterBackground: "#111114",
      gutterColor: "#6e6e7a",
      codeFoldGutterBackground: "#111114",
      codeFoldBackground: "#1a1a1f",
      emptyLineBackground: "#0a0a0b",
    },
  },
  contentText: {
    fontFamily: "JetBrains Mono, ui-monospace, monospace",
    fontSize: "12px",
  },
  titleBlock: {
    background: "#1a1a1f",
    color: "#a8a8b3",
    fontFamily: "JetBrains Mono, ui-monospace, monospace",
    fontSize: "11px",
    padding: "6px 12px",
  },
};

function useOpCounts(arch: Arch): Record<string, number> {
  return useMemo(() => {
    const v = arch.variants.find(isCanonical);
    if (!v) return {};
    const out: Record<string, number> = {};
    for (const n of v.fuf.nodes) out[n.op] = (out[n.op] ?? 0) + 1;
    return out;
  }, [arch]);
}
