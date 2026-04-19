import { Link } from "@tanstack/react-router";
import type { ReactNode } from "react";
import type { RootSearch } from "../router";

/** App chrome: the title bar, plus an `<Outlet>` for whatever route
 * matched. Intentionally minimal — the gallery is math-only at the
 * top level; the source / fuf views live in the per-arch drilldown. */
export function Shell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen">
      <header className="px-6 py-4 border-b border-ink-700 flex items-center gap-4">
        <Link
          to="/"
          search={(prev) => prev as RootSearch}
          className="text-xl font-semibold shrink-0 hover:opacity-80"
        >
          ferrite · <span className="text-accent-compute">forward</span>
        </Link>
        <span className="text-xs text-ink-400 ml-auto">
          click to zoom · ⌘/ctrl-click two to diff
        </span>
      </header>
      {children}
    </div>
  );
}
