import { useNavigate, useParams } from "@tanstack/react-router";
import { findArch, useArches } from "@/lib/archStore";
import { DiffView } from "@/components/DiffView";
import type { RootSearch } from "../router";

export function DiffRoute() {
  const { arches, error } = useArches();
  const { a: aName, b: bName } = useParams({ strict: false }) as {
    a: string;
    b: string;
  };
  const navigate = useNavigate();

  if (error) return <Msg text={`load error: ${error}`} tone="error" />;
  if (!arches) return <Msg text="loading…" />;
  const a = findArch(arches, aName);
  const b = findArch(arches, bName);
  if (!a || !b)
    return (
      <Msg text={`unknown arch in diff: ${aName} / ${bName}`} tone="error" />
    );

  return (
    <DiffView
      a={a}
      b={b}
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
