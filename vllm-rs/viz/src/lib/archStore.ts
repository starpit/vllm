import { useEffect, useState } from "react";
import type { Arch } from "@/types";
import { loadAllArches } from "./load";
import { sortArchesBySimilarity } from "./similarity";

/** Single shared in-memory cache of arches across all routes —
 * loadAllArches fires exactly once for the lifetime of the SPA. */
let cache: Promise<Arch[]> | null = null;

function loadOnce(): Promise<Arch[]> {
  if (!cache) {
    cache = loadAllArches().then(sortArchesBySimilarity);
  }
  return cache;
}

interface State {
  arches: Arch[] | null;
  error: string | null;
}

/** React hook for routes that need the arch list. Returns
 * `{ arches: null }` while loading, `{ arches: [...] }` once ready. */
export function useArches(): State {
  const [state, setState] = useState<State>({ arches: null, error: null });
  useEffect(() => {
    let alive = true;
    loadOnce().then(
      (arches) => alive && setState({ arches, error: null }),
      (e) => alive && setState({ arches: null, error: String(e) }),
    );
    return () => {
      alive = false;
    };
  }, []);
  return state;
}

/** Lookup helper for routes that take an arch name as a path param. */
export function findArch(arches: Arch[], name: string): Arch | null {
  return arches.find((a) => a.arch === name) ?? null;
}
