import { useEffect, useState } from "react";
import { createHighlighterCoreSync, createOnigurumaEngine } from "shiki/core";
import rustGrammar from "shiki/langs/rust.mjs";
import githubDark from "shiki/themes/github-dark-default.mjs";

// Single-language highlighter: pulling `codeToHtml` from "shiki" drags
// every grammar (~5 MB of chunks). Grammar + theme are inlined here so
// the prod bundle only ships Rust.
let highlighterPromise: Promise<ReturnType<typeof createHighlighterCoreSync>> | null = null;
function getHighlighter() {
  if (!highlighterPromise) {
    highlighterPromise = (async () => {
      const engine = await createOnigurumaEngine(import("shiki/wasm"));
      return createHighlighterCoreSync({
        themes: [githubDark],
        langs: [rustGrammar],
        engine,
      });
    })();
  }
  return highlighterPromise;
}

interface Props {
  source: string;
  /** optional max-height for the scroll area */
  maxH?: string;
}

/** Shiki-highlighted Rust source. Loaded lazily so the page paints
 * before the highlighter wasm finishes downloading. */
export function DslView({ source, maxH = "100%" }: Props) {
  const [html, setHtml] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    getHighlighter()
      .then((hl) =>
        hl.codeToHtml(source, { lang: "rust", theme: "github-dark-default" }),
      )
      .then((h) => alive && setHtml(h))
      .catch((e) => alive && setHtml(`<pre>${String(e)}</pre>`));
    return () => {
      alive = false;
    };
  }, [source]);

  return (
    <div
      className="overflow-auto text-[12px] leading-relaxed [&_pre]:!bg-transparent [&_pre]:p-3"
      style={{ maxHeight: maxH }}
      // dangerouslySetInnerHTML is fine here — shiki output is HTML
      // it built itself from a string we control.
      dangerouslySetInnerHTML={
        html
          ? { __html: html }
          : {
              __html: `<pre class="text-ink-300 p-3">${escapeHtml(source)}</pre>`,
            }
      }
    />
  );
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
