import type { QuantTag } from "@/lib/load";

/** Quant pills — prominent, colored, the load-bearing label. */
export function QuantBadges({ quants }: { quants: QuantTag[] }) {
  return (
    <div className="flex flex-wrap gap-1">
      {quants.map((q) => (
        <span
          key={q.label}
          className={`${q.color} text-white text-[10px] font-bold px-2 py-[3px] rounded-full uppercase tracking-wider shadow-sm`}
        >
          {q.label}
        </span>
      ))}
    </div>
  );
}

/** Size labels — visually subordinate metadata. Mono, dim, no chrome. */
export function SizeBadges({ sizes }: { sizes: string[] }) {
  if (sizes.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0 text-[10px] font-mono text-ink-400">
      {sizes.map((s, i) => (
        <span key={s}>
          {s}
          {i < sizes.length - 1 && (
            <span className="ml-1.5 text-ink-500">·</span>
          )}
        </span>
      ))}
    </div>
  );
}
