import type { Arch } from "@/types";
import { isCanonical } from "@/types";
import { detectFamilies, detectQuants, detectSizes } from "@/lib/load";
import { paramsFor } from "@/lib/bindings";
import { QuantBadges, SizeBadges } from "./Badges";
import { StencilView } from "./StencilView";

interface Props {
  arch: Arch;
  selected: boolean;
  selectedLocal?: string | null;
  onSelectLocal?: (name: string) => void;
  onClick: (e: React.MouseEvent) => void;
  /** Other arches that share this card's math body (identical
   * stencil program). Rendered as `≡ name` subtitle — tells the user
   * "same forward!, different bindings"; the bindings view explains
   * the actual divergence. */
  sharedMathWith?: string[];
}

/** Gallery cell. Renders the math view (AST silhouette) only — the
 * source / fuf views live in the per-arch drilldown so the gallery
 * stays focused on cross-arch structural comparison. */
export function ArchCard({
  arch,
  selected,
  selectedLocal = null,
  onSelectLocal,
  onClick,
  sharedMathWith = [],
}: Props) {
  const variantNames = arch.variants.map((v) => v.name);
  const quants = detectQuants(variantNames);
  const sizes = detectSizes(variantNames);
  const families = detectFamilies(arch.arch, variantNames);
  // Pick one representative variant for the thumbnail's binding
  // decorations. First canonical is arbitrary-but-stable — the zoom
  // view lets the user see per-variant values.
  const sampleVariant = arch.variants.find(isCanonical);
  const sampleParams = sampleVariant ? paramsFor(sampleVariant) : undefined;

  return (
    <div
      onClick={onClick}
      className={`group relative h-[25rem] cursor-pointer bg-ink-800 border rounded-lg overflow-hidden flex flex-col transition-colors ${
        selected
          ? "border-accent-compute ring-2 ring-accent-compute/40"
          : "border-ink-600 hover:border-ink-500"
      }`}
    >
      {/* Header: arch name is always the dominant element.
          Shared-math siblings sit beneath as a dim subtitle —
          surfaces that llama / mistral (and any future twins) are
          the same function with different bindings. */}
      <div className="px-3 py-2 border-b border-ink-700 flex items-start justify-between gap-2 shrink-0">
        <div className="min-w-0">
          <h3 className="font-semibold text-base text-ink-100 truncate">
            {arch.arch}
            {families.length > 0 && (
              <span
                className="ml-2 text-[11px] font-mono text-ink-400 font-normal"
                title="Product families / generations shipped under this arch"
              >
                {families.join(" · ")}
              </span>
            )}
          </h3>
          {sharedMathWith.length > 0 && (
            <div
              className="text-[10px] font-mono text-ink-500 truncate"
              title={`Same math body as: ${sharedMathWith.join(", ")}`}
            >
              ≡ {sharedMathWith.join(", ")}
            </div>
          )}
        </div>
        <span className="text-[10px] font-mono text-ink-400 shrink-0">
          {arch.variants.length} variants
        </span>
      </div>

      {/* Body: math view only at this level. */}
      <div className="flex-1 min-h-0 overflow-hidden bg-ink-800">
        {arch.program ? (
          <StencilView
            program={arch.program}
            compact
            selectedLocal={selectedLocal}
            onSelectLocal={onSelectLocal}
            params={sampleParams}
          />
        ) : (
          <div className="text-ink-400 text-xs p-3">
            no stencil (rebuild fixtures with the latest schema)
          </div>
        )}
      </div>

      {/* Footer: quants prominent, sizes subordinate. */}
      <div className="px-3 py-2 border-t border-ink-700 shrink-0 flex items-center justify-between gap-2 bg-ink-900/40">
        <QuantBadges quants={quants} />
        <SizeBadges sizes={sizes} />
      </div>
    </div>
  );
}
