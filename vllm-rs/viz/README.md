# ferrite-viz

Small-multiples web visualizer for the architectures compiled by `#[forward]`.
Each card is one arch; three views per card:

- **math** (default) — AST silhouette: every `forward!` statement renders as a
  row of nested colored rectangles, one per expression node. Pills = locals,
  squares = weights, rounded rectangles = ops. Color = op family
  (matmul / attention / norm / activation / residual / …).
- **source** — Shiki-highlighted Rust DSL (just the `forward!` body, no glue).
- **fuf** — fully-unrolled DAG via React Flow.

Cards are sorted by structural similarity (greedy nearest-neighbor on
op-multiset jaccard distance), so llama / mistral / granite cluster, the
gemmas sit together, etc.

**Click** to zoom. **⌘ / Ctrl-click** two cards to open a side-by-side
VSCode-style word-level DSL diff with a per-op count delta footer.

---

## Prerequisites

- **Rust** toolchain (whatever the workspace already requires).
- **Node.js ≥ 20** + npm.
  - Ubuntu/Debian: `sudo apt install nodejs npm` or use [nvm].
  - macOS: `brew install node`.

[nvm]: https://github.com/nvm-sh/nvm

The visualizer compiles to a static SPA — no Node runtime is needed at
serve time once it's built.

---

## First run

```bash
# 1. Generate fixtures. Writes one <arch>.json per `#[forward]`
#    invocation under public/arches/. The build is fully cached;
#    rerun whenever the DSL or model_architectures/ changes.
cd vllm-rs
FERRITE_VIZ_OUT=$(pwd)/viz/public/arches cargo build -p ferrite-models

# 2. Install JS deps.
cd viz
npm install

# 3. Dev server (HMR on, binds 0.0.0.0).
npm run dev
# → http://localhost:5173/
```

If you're on a remote VM, forward the port from your laptop:

```bash
ssh -L 5173:localhost:5173 <vm-host>
# or
gcloud compute ssh <instance> --zone=<zone> -- -L 5173:localhost:5173
```

---

## Day-to-day workflow

After editing a `forward!` body in `crates/ferrite-model-<arch>/src/lib.rs`:

```bash
cd vllm-rs
FERRITE_VIZ_OUT=$(pwd)/viz/public/arches cargo build -p ferrite-models
```

Vite picks up the new JSON automatically — no dev-server restart needed.

To force a full regen (e.g. after a `viz_dump.rs` schema change):

```bash
rm vllm-rs/viz/public/arches/*.json
touch vllm-rs/crates/ferrite-model-*/src/lib.rs
FERRITE_VIZ_OUT=$(pwd)/vllm-rs/viz/public/arches cargo build -p ferrite-models
```

The `touch` invalidates each model crate so the proc-macro re-runs.
Without `FERRITE_VIZ_OUT` set, the dump is a complete no-op — there's
zero overhead on normal `cargo build`s.

---

## Production build

```bash
cd vllm-rs/viz
npm run build       # → dist/
npm run preview     # serve dist/ on :4173 to sanity-check
```

The output is fully static. Drop `dist/` behind any HTTP server.

Bundle size is ~1.2 MB (Shiki narrowed to the Rust grammar only;
Oniguruma wasm is the largest single chunk).

---

## Project layout

```
viz/
├── public/
│   ├── arches/          # generated JSON fixtures (gitignored)
│   └── favicon.svg
├── src/
│   ├── App.tsx          # gallery + zoom + diff routing, view toggle
│   ├── types.ts         # mirror of viz_dump.rs's JSON schema
│   ├── lib/
│   │   ├── load.ts      # arch fetch, quant + size detection
│   │   └── similarity.ts# jaccard/op-multiset card sort
│   └── components/
│       ├── ArchCard.tsx
│       ├── StencilView.tsx   # AST silhouette (math view)
│       ├── DslView.tsx       # Shiki source view
│       ├── FufView.tsx       # React Flow DAG view
│       ├── ZoomView.tsx      # full-screen detail
│       ├── DiffView.tsx      # side-by-side DSL diff
│       └── Badges.tsx        # quant / size pills
├── index.html
├── vite.config.ts
├── tailwind.config.js
└── tsconfig.json
```

---

## Architecture

```
ferrite-forward-macro::viz_dump   (Rust, env-gated by FERRITE_VIZ_OUT)
       │  walks classified::Program → JSON tree per arch
       ▼
vllm-rs/viz/public/arches/<arch>.json
       │  fetched by the SPA at startup, sorted by similarity
       ▼
src/{App,components/*}.tsx
```

The dump is a pure write-out; nothing in the runtime forward fn
depends on it. The viz is a parasite on the compile.

### Schema versioning

`schema_version` in every JSON is bumped any time `viz_dump.rs` changes
its emit shape. `src/types.ts` is the canonical mirror — keep them in
sync. Current: **v3** (assign blocks emit a recursive `value` AST tree;
v2 was flat `(op, args[])`; v1 had no `program` block).

---

## Stack

- **Vite + React + TypeScript + Tailwind** — base.
- **React Flow** — FUF DAG layout.
- **Shiki** (`shiki/core` only, Rust grammar only) — source highlighting.
- **react-diff-viewer-continued** — VSCode-style side-by-side diff.
- **Framer Motion** — micro-animations (card transitions).

No build-time data-fetching, no SSR, no API server: everything ships
as static assets. The Rust side is the only producer; the SPA is the
only consumer.
