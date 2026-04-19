import {
  useNavigate,
  useParams,
  useSearch,
} from "@tanstack/react-router";
import { findArch, useArches } from "@/lib/archStore";
import { ZoomView } from "@/components/ZoomView";
import type { RootSearch } from "../router";

export function ZoomRoute() {
  const { arches, error } = useArches();
  const { arch: archName } = useParams({ strict: false }) as { arch: string };
  const search = useSearch({ strict: false }) as { sel: string | null };
  const navigate = useNavigate();

  const setSel = (next: string | null) =>
    navigate({
      to: ".",
      search: (prev) => ({ ...(prev as RootSearch), sel: next }),
    });

  if (error) return <Msg text={`load error: ${error}`} tone="error" />;
  if (!arches) return <Msg text="loading…" />;
  const arch = findArch(arches, archName);
  if (!arch) return <Msg text={`unknown arch: ${archName}`} tone="error" />;

  return (
    <ZoomView
      arch={arch}
      selectedLocal={search.sel}
      onSelectLocal={(name) => setSel(name === search.sel ? null : name)}
      onClose={() =>
        navigate({
          to: "/",
          search: (prev) => prev as RootSearch,
        })
      }
    />
  );
}

function Msg({ text, tone }: { text: string; tone?: "error" }) {
  return (
    <div className={`p-6 ${tone === "error" ? "text-rose-300" : "text-ink-400"}`}>
      {text}
    </div>
  );
}
