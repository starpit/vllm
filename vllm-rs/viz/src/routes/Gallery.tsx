import { useNavigate, useSearch } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { ArchCard } from "@/components/ArchCard";
import { useArches } from "@/lib/archStore";
import { clusterArchesByMath, programHash } from "@/lib/bindings";
import type { RootSearch } from "../router";

export function GalleryRoute() {
  const { arches, error } = useArches();
  const search = useSearch({ strict: false }) as RootSearch;
  const navigate = useNavigate();

  // Pending diff index lives in component state, not the URL — it's
  // a momentary "first card picked" flag, not a navigable place.
  const [pendingDiff, setPendingDiff] = useState<string | null>(null);

  // For each arch, the names of other arches sharing its math body.
  // Precomputed once per arches change so the per-card lookup is O(1).
  const mathSiblings = useMemo(() => {
    const out = new Map<string, string[]>();
    if (!arches) return out;
    const groups = clusterArchesByMath(arches);
    for (const a of arches) {
      if (!a.program) continue;
      const g = groups.get(programHash(a.program)) ?? [a];
      out.set(
        a.arch,
        g.filter((s) => s.arch !== a.arch).map((s) => s.arch),
      );
    }
    return out;
  }, [arches]);

  const setSel = (next: string | null) =>
    navigate({
      to: ".",
      search: (prev) => ({ ...(prev as RootSearch), sel: next }),
    });

  if (error) {
    return (
      <div className="p-6 text-rose-300">
        <p>load error: {error}</p>
      </div>
    );
  }
  if (!arches) {
    return <div className="p-6 text-ink-400">loading…</div>;
  }
  if (arches.length === 0) {
    return (
      <div className="p-6 text-ink-300">
        <h1 className="text-xl mb-2">no arches found</h1>
        <p className="text-sm text-ink-400">
          run{" "}
          <code className="bg-ink-700 px-1">
            FERRITE_VIZ_OUT=$(pwd)/vllm-rs/viz/public/arches cargo build -p
            ferrite-models
          </code>{" "}
          from <code>vllm-rs/</code>.
        </p>
      </div>
    );
  }

  // `metaKey` = Cmd on Mac, Win key on Linux/Windows; `ctrlKey` is
  // the cross-platform fallback. Shift is intentionally avoided: the
  // browser hijacks shift-click for range text selection.
  const handleClick = (archName: string) => (e: React.MouseEvent) => {
    if (e.metaKey || e.ctrlKey) {
      e.preventDefault();
      if (pendingDiff === null) {
        setPendingDiff(archName);
      } else if (pendingDiff !== archName) {
        navigate({
          to: "/diff/$a/$b",
          params: { a: pendingDiff, b: archName },
          search: (prev) => prev as RootSearch,
        });
        setPendingDiff(null);
      }
    } else {
      setPendingDiff(null);
      navigate({
        to: "/zoom/$arch",
        params: { arch: archName },
        search: (prev) => prev as RootSearch,
      });
    }
  };

  return (
    <>
      {(pendingDiff !== null || search.sel !== null) && (
        <div className="px-6 py-1.5 border-b border-ink-700 text-xs text-ink-400 flex items-center gap-3">
          {pendingDiff !== null && (
            <span className="text-accent-compute">
              · diff pending: {pendingDiff}
            </span>
          )}
          {search.sel !== null && (
            <span className="flex items-center gap-2">
              <span>tracking</span>
              <span className="bg-ink-700 border border-ink-500 rounded-full px-2 py-0.5 text-ink-100 font-mono">
                {search.sel}
              </span>
              <button
                onClick={() => setSel(null)}
                className="hover:text-ink-100 underline"
              >
                clear
              </button>
            </span>
          )}
        </div>
      )}
      <main className="p-6 grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {arches.map((a) => (
          <ArchCard
            key={a.arch}
            arch={a}
            selectedLocal={search.sel}
            onSelectLocal={(name) => setSel(name === search.sel ? null : name)}
            selected={pendingDiff === a.arch}
            sharedMathWith={mathSiblings.get(a.arch) ?? []}
            onClick={handleClick(a.arch)}
          />
        ))}
      </main>
    </>
  );
}
