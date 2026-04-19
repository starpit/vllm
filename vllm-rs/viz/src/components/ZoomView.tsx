import { useMemo, useState } from "react";
import type { Arch } from "@/types";
import { isCanonical } from "@/types";
import { DslView } from "./DslView";
import { FufView } from "./FufView";
import { StencilView } from "./StencilView";
import { BindingsView } from "./BindingsView";
import { useArches } from "@/lib/archStore";
import {
  clusterArchesByMath,
  paramsFor,
  programHash,
  stencilImplMap,
} from "@/lib/bindings";

type LeftFace = "stencil" | "dsl" | "bindings";

interface Props {
  arch: Arch;
  onClose: () => void;
  selectedLocal?: string | null;
  onSelectLocal?: (name: string) => void;
}

/** Full-screen detail for one arch: DSL on the left, FUF on the right
 * with a workload-point picker. Variant picker switches which canonical
 * is shown. */
export function ZoomView({
  arch,
  onClose,
  selectedLocal = null,
  onSelectLocal,
}: Props) {
  const canonicals = arch.variants.filter(isCanonical);
  const [variantIdx, setVariantIdx] = useState(0);
  const [wpIdx, setWpIdx] = useState(0);
  const [leftFace, setLeftFace] = useState<LeftFace>(
    arch.program ? "stencil" : "dsl",
  );
  const variant = canonicals[variantIdx] ?? null;
  const workload = variant?.workloads[wpIdx] ?? null;

  // Shared-math siblings: other arches whose stencil program hashes
  // identically to this one. For llama/mistral this will be the
  // two-arch set; the bindings view then becomes a side-by-side.
  const { arches: allArches } = useArches();
  const siblings = useMemo(() => {
    if (!allArches || !arch.program) return [arch];
    const groups = clusterArchesByMath(allArches);
    return groups.get(programHash(arch.program)) ?? [arch];
  }, [allArches, arch]);

  // Stencil-op → picked-impl map for the current (variant, workload)
  // pair. Powers the ⚙ chips attached to each math-view op — so users
  // see `gemm ⚙ fused_qkv_rope_cache` at M=1, vs `gemm ⚙
  // marlin_awq_gemm` for an AWQ variant.
  const implByAssignIdx = useMemo(() => {
    if (!arch.program || !variant || !workload) return undefined;
    return stencilImplMap(arch.program, variant, workload);
  }, [arch.program, variant, workload]);

  return (
    <div className="fixed inset-0 z-50 bg-ink-900/95 flex flex-col">
      <header className="flex items-center justify-between px-4 py-2 border-b border-ink-600 shrink-0">
        <div className="flex items-baseline gap-3">
          <h2 className="text-lg font-semibold">{arch.arch}</h2>
          <span className="text-xs text-ink-400 font-mono">
            {arch.hf_arches.join(", ")}
          </span>
        </div>
        <button
          onClick={onClose}
          className="text-ink-200 hover:text-ink-100 px-2 py-1 text-sm border border-ink-500 rounded"
        >
          close (esc)
        </button>
      </header>
      <div className="flex-1 grid grid-cols-2 min-h-0">
        <section className="border-r border-ink-600 min-h-0 overflow-hidden flex flex-col">
          <div className="px-3 py-2 border-b border-ink-600 flex items-center gap-2 text-xs shrink-0">
            <div className="inline-flex rounded border border-ink-600 bg-ink-800 overflow-hidden">
              {(["stencil", "dsl", "bindings"] as const).map((f, i) => (
                <button
                  key={f}
                  onClick={() => setLeftFace(f)}
                  disabled={f === "stencil" && !arch.program}
                  className={`px-2 py-0.5 ${i > 0 ? "border-l border-ink-600" : ""} ${
                    leftFace === f
                      ? "bg-ink-600 text-ink-100"
                      : "text-ink-300 hover:text-ink-100 hover:bg-ink-700 disabled:opacity-40 disabled:hover:bg-transparent disabled:cursor-not-allowed"
                  }`}
                >
                  {f === "dsl" ? "source" : f === "stencil" ? "math" : f}
                </button>
              ))}
            </div>
            {leftFace === "bindings" && siblings.length > 1 && (
              <span className="text-[10px] text-ink-400 font-mono">
                same math as {siblings
                  .filter((s) => s.arch !== arch.arch)
                  .map((s) => s.arch)
                  .join(", ")}
              </span>
            )}
          </div>
          <div className="flex-1 min-h-0 overflow-hidden">
            {leftFace === "stencil" && arch.program ? (
              <StencilView
                program={arch.program}
                selectedLocal={selectedLocal}
                onSelectLocal={onSelectLocal}
                params={variant ? paramsFor(variant) : undefined}
                implByAssignIdx={implByAssignIdx}
              />
            ) : leftFace === "bindings" ? (
              <BindingsView arches={siblings} />
            ) : (
              <DslView source={arch.dsl_source} />
            )}
          </div>
        </section>
        <section className="flex flex-col min-h-0">
          <div className="px-3 py-2 border-b border-ink-600 flex items-center gap-3 text-xs shrink-0">
            <label className="flex items-center gap-1">
              <span className="text-ink-400">variant</span>
              <select
                value={variantIdx}
                onChange={(e) => {
                  setVariantIdx(Number(e.target.value));
                  setWpIdx(0);
                }}
                className="bg-ink-700 border border-ink-500 rounded px-1 py-0.5 font-mono"
              >
                {canonicals.map((v, i) => (
                  <option key={v.name} value={i}>
                    {v.name}
                  </option>
                ))}
              </select>
            </label>
            {variant && variant.workloads.length > 0 && (
              <label className="flex items-center gap-1">
                <span className="text-ink-400">workload</span>
                <select
                  value={wpIdx}
                  onChange={(e) => setWpIdx(Number(e.target.value))}
                  className="bg-ink-700 border border-ink-500 rounded px-1 py-0.5 font-mono"
                >
                  {variant.workloads.map((w, i) => (
                    <option key={i} value={i}>
                      M={w.num_tokens} sk={w.sk_bucket} ·{" "}
                      {w.predicted_us.toFixed(0)}µs · {w.num_waves} waves
                    </option>
                  ))}
                </select>
              </label>
            )}
            {variant && (
              <span className="text-ink-400">
                {variant.fuf.nodes.length} tiles
              </span>
            )}
          </div>
          <div className="flex-1 min-h-0 bg-ink-900">
            {variant ? (
              <FufView variant={variant} workload={workload} />
            ) : (
              <div className="text-ink-400 p-4">no canonical variant</div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
