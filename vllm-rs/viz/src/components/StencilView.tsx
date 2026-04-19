import type { StencilBlock, StencilExpr, StencilProgram } from "@/types";
import { describeImpl } from "@/lib/impls";

interface Props {
  program: StencilProgram;
  /** Compact mode for thumbnails: smaller chips, no leaf labels. */
  compact?: boolean;
  /** When set, every local pill (target or RHS reference) whose name
   * matches gets a highlight ring; non-matching locals dim. */
  selectedLocal?: string | null;
  /** Click handler for local pills. */
  onSelectLocal?: (name: string) => void;
  /** Optional per-variant parameter bag (bounds ∪ scalars). */
  params?: Record<string, number>;
  /** Optional per-stencil-assign implementation picks, keyed by the
   * flat-walk index of the assign (see `walkAssignIndex`). Populated
   * by ZoomView once a workload is selected; undefined in the gallery
   * thumbnails (no workload picked). */
  implByAssignIdx?: Map<number, string>;
}

/** Map op → names of config values the ferrite-kernels implementation
 * reads when that op fires. Curated against the kernel library; keep
 * in sync as new kernel behaviors land. The viz renders these as
 * small sub-chips on the op so the math view shows *all* inputs, not
 * just the ones spelled in the DSL. */
function implicitBindingsFor(op: string): string[] {
  switch (op) {
    case "RmsNorm":
      return ["rms_norm_eps"];
    case "LayerNorm":
      return ["layer_norm_eps"];
    case "RopeAppend":
    case "RopeAppendInterleaved":
      return ["rope_theta", "partial_rotary_factor"];
    case "Attention":
      return ["head_dim"];
    case "SlidingAttention":
      return ["head_dim", "sliding_window"];
    case "TanhSoftCap":
      return ["attn_logit_softcapping"];
    default:
      return [];
  }
}

function formatBindingValue(v: number): string {
  if (Number.isInteger(v)) return String(v);
  if (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e6) return v.toExponential(2);
  return String(v);
}

/** ── Math view ──
 * One row per `forward!` statement, rendered as an AST silhouette:
 * each row is a sequence of nested colored rectangles, one per
 * expression node. Composite ops (call/binop) wrap their children;
 * leaves (locals/weights/scalars) are simple chips. The point: see
 * structural math diffs across architectures at a glance, including
 * intra-line nesting that the FUF/source views obscure. */
export function StencilView({
  program,
  compact = false,
  selectedLocal = null,
  onSelectLocal,
  params,
  implByAssignIdx,
}: Props) {
  // Precompute a block-identity → assign-index map once per render
  // of the program tree. Keeps impl-pick lookup O(1) in the
  // render tree without passing a mutable counter around (which
  // would be unstable under React strict mode's double-invoke).
  const assignIdx = indexAssigns(program.blocks);
  return (
    <div className="overflow-auto h-full p-3 text-[13px] font-mono">
      <BlockList
        blocks={program.blocks}
        compact={compact}
        selectedLocal={selectedLocal}
        onSelectLocal={onSelectLocal}
        params={params}
        implByAssignIdx={implByAssignIdx}
        assignIdx={assignIdx}
      />
    </div>
  );
}

function indexAssigns(blocks: StencilBlock[]): Map<StencilBlock, number> {
  const map = new Map<StencilBlock, number>();
  let counter = 0;
  function walk(bs: StencilBlock[]) {
    for (const b of bs) {
      if (b.kind === "assign") {
        map.set(b, counter++);
      } else if (b.kind === "for") {
        walk(b.body);
      } else if (b.kind === "if") {
        walk(b.then);
        walk(b.else);
      }
    }
  }
  walk(blocks);
  return map;
}

interface InnerProps {
  compact: boolean;
  selectedLocal: string | null;
  onSelectLocal?: (name: string) => void;
  params?: Record<string, number>;
  implByAssignIdx?: Map<number, string>;
  assignIdx: Map<StencilBlock, number>;
}

function BlockList({
  blocks,
  ...inner
}: { blocks: StencilBlock[] } & InnerProps) {
  return (
    <div className="flex flex-col gap-1">
      {blocks.map((b, i) => (
        <BlockNode key={i} block={b} {...inner} />
      ))}
    </div>
  );
}

function BlockNode({
  block,
  ...inner
}: { block: StencilBlock } & InnerProps) {
  const { compact, selectedLocal, onSelectLocal, params } = inner;
  void params; // consumed below in ExprNode via `inner`
  if (block.kind === "for") {
    // Heuristic split: every transformer loop body has an
    // `attention(...)` call somewhere in the middle. Group the body
    // into pre / attention / post bands. Falls through to a single
    // un-grouped body when no attention call is found.
    const pivot = findAttentionPivot(block.body);
    const pre = pivot >= 0 ? block.body.slice(0, pivot) : block.body;
    const attn = pivot >= 0 ? block.body.slice(pivot, pivot + 1) : [];
    const post = pivot >= 0 ? block.body.slice(pivot + 1) : [];
    return (
      <div className="flex items-stretch gap-2">
        {/* Loop accent bar — neutral gray, slightly thicker than the
            attention bands so loop nesting reads as the outer level. */}
        <div className="w-[4px] shrink-0 rounded-full bg-ink-500" />
        <div className="flex-1 min-w-0 flex flex-col gap-1">
          <div className="text-[10px] uppercase tracking-wider text-ink-400 flex items-center gap-1">
            <span className="text-accent-load font-semibold">for</span>
            <span className="text-ink-200">{block.ivar}</span>
            <span className="text-ink-500">∈</span>
            <span className="text-ink-300">
              [{block.start}, {block.end})
            </span>
          </div>
          <div className="flex flex-col gap-1.5">
            <BandedSection
              label="pre-attention"
              blocks={pre}
              tone="pre"
              inner={inner}
            />
            {attn.length > 0 && (
              <BandedSection
                label="attention"
                blocks={attn}
                tone="attn"
                inner={inner}
              />
            )}
            {post.length > 0 && (
              <BandedSection
                label="post-attention"
                blocks={post}
                tone="post"
                inner={inner}
              />
            )}
          </div>
        </div>
      </div>
    );
  }
  if (block.kind === "if") {
    return (
      <div className="flex items-stretch gap-2">
        <div className="w-[3px] shrink-0 rounded-full bg-ink-500" />
        <div className="flex-1 min-w-0 flex flex-col gap-1">
          <div className="text-[10px] uppercase tracking-wider text-ink-400">
            <span className="text-accent-store font-semibold">if</span>{" "}
            <span className="text-ink-200">{block.cond}</span>
          </div>
          <div className="text-[9px] uppercase tracking-wider text-ink-500">
            then
          </div>
          <BlockList blocks={block.then} {...inner} />
          {block.else.length > 0 && (
            <>
              <div className="text-[9px] uppercase tracking-wider text-ink-500 mt-1">
                else
              </div>
              <BlockList blocks={block.else} {...inner} />
            </>
          )}
        </div>
      </div>
    );
  }

  // Assign row: <target-pill(s)> ← <expr-tree>
  // The target pills use the same local-leaf shape/color as their
  // appearances on any later RHS, so the eye can track a value
  // through the program (q ← gemm; later q is an arg to rope_append).
  // Impl pick for THIS assign under the currently-selected workload
  // (only in zoom view). Passed down to ExprNode so it can decorate
  // the outermost call/binop chip with the chosen implementation.
  const idx = inner.assignIdx.get(block);
  const implPick =
    idx !== undefined ? inner.implByAssignIdx?.get(idx) : undefined;

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      <div className="shrink-0 flex items-center gap-1">
        {block.targets.map((t) => (
          <LocalChip
            key={t}
            name={t}
            compact={compact}
            selectedLocal={selectedLocal}
            onSelectLocal={onSelectLocal}
          />
        ))}
        <span
          className={`text-ink-500 ml-0.5 ${compact ? "text-[11px]" : "text-[12px]"}`}
        >
          =
        </span>
      </div>
      <div className="flex-1 min-w-0 flex items-center flex-wrap gap-1">
        <ExprNode expr={block.value} implPick={implPick} {...inner} />
      </div>
    </div>
  );
}

/** Single local pill — used both for assign LHS and inside the AST
 * walker for `Expr::Local` references. Centralizes the selected/dim
 * styling so a click in either place toggles the same way. */
function LocalChip({
  name,
  compact,
  selectedLocal,
  onSelectLocal,
}: {
  name: string;
  compact: boolean;
  selectedLocal: string | null;
  onSelectLocal?: (name: string) => void;
}) {
  const s = PALETTE.local;
  const isSelected = selectedLocal === name;
  const isDimmed = selectedLocal !== null && !isSelected;
  const ring = isSelected ? "ring-2 ring-accent-compute" : "";
  const opacity = isDimmed ? "opacity-30" : "";
  const click = (e: React.MouseEvent) => {
    if (!onSelectLocal) return;
    e.stopPropagation();
    onSelectLocal(name);
  };
  if (compact) {
    // Show the first letter of the variable so you can still read it
    // at a glance without hovering. Keeps the pill small while
    // carrying enough info to distinguish q/k/v/hidden_states/normed.
    return (
      <span
        onClick={click}
        className={`${s.bg} ${s.border} ${s.label} rounded-full border inline-flex items-center justify-center self-center cursor-pointer ${ring} ${opacity} transition-opacity font-bold text-[10px] uppercase leading-none`}
        style={{ width: 16, height: 16 }}
        title={name}
      >
        <span style={{ opacity: 0.6 }}>{name.charAt(0)}</span>
      </span>
    );
  }
  return (
    <span
      onClick={click}
      className={`${s.bg} ${s.border} ${s.label} rounded-full border px-2 py-0.5 text-[12px] whitespace-nowrap cursor-pointer self-center ${ring} ${opacity} transition-opacity hover:brightness-125`}
    >
      {name}
    </span>
  );
}

/** Find the index of the first statement in `body` whose value
 * (anywhere in its expr tree) calls `attention` or `sliding_attention`.
 * Returns -1 if no attention call lives in this body. The result is
 * the *split point*: statements [0..=pivot] are "pre-attention",
 * (pivot..] are "post-attention". Including the pivot in pre keeps
 * the attention call itself with the qkv/rope group it depends on. */
function findAttentionPivot(body: StencilBlock[]): number {
  for (let i = 0; i < body.length; i++) {
    const b = body[i];
    if (b.kind === "assign" && exprContainsAttention(b.value)) return i;
  }
  return -1;
}

function exprContainsAttention(e: StencilExpr): boolean {
  if (e.kind === "call") {
    if (e.op === "Attention" || e.op === "SlidingAttention") return true;
    return e.children.some(exprContainsAttention);
  }
  if (e.kind === "binop") return e.children.some(exprContainsAttention);
  return false;
}

/** Labeled banded sub-box around a slice of the loop body. Tone =
 * which accent color the label gets (orange = pre-attn, violet =
 * post-attn); the band itself is tinted so the structural split
 * reads at a glance even in compact thumbnails. */
function BandedSection({
  label,
  blocks,
  tone,
  inner,
}: {
  label: string;
  blocks: StencilBlock[];
  tone: "pre" | "attn" | "post";
  inner: InnerProps;
}) {
  // Phase markers — attention is THE event, pre/post are positions
  // relative to it. The attention bar reuses the chip-family orange
  // (so the band reinforces the attention chip inside it); pre/post
  // get neutral cool/warm grays that don't collide with any chip
  // family (norm = emerald, activation = violet, residual = rose,
  // matmul = sky, embed = fuchsia, elem = amber, reshape = slate).
  const palette = {
    pre: { bar: "bg-slate-400/80", label: "text-slate-300" },
    attn: { bar: "bg-orange-500/90", label: "text-orange-300" },
    post: { bar: "bg-zinc-400/80", label: "text-zinc-300" },
  }[tone];
  return (
    <div className="flex items-stretch gap-2">
      <div className={`w-[3px] shrink-0 rounded-full ${palette.bar}`} />
      <div className="flex-1 min-w-0 flex flex-col gap-0.5">
        <div
          className={`text-[10px] uppercase tracking-wider font-semibold ${palette.label}`}
        >
          {label}
        </div>
        <BlockList blocks={blocks} {...inner} />
      </div>
    </div>
  );
}

/** Op-family color palette. Composite nodes (call/binop) get a tinted
 * background + colored border; leaves get a more saturated background.
 * Tuned so a row of common pattern (norm → matmul → activation) reads
 * as a recognizable color sequence across architectures. */
function familyOf(node: StencilExpr): string {
  if (node.kind === "call") {
    const op = node.op;
    if (op === "Embed") return "embed";
    if (op === "Gemm") return "matmul";
    if (
      op === "Attention" ||
      op === "SlidingAttention" ||
      op === "RopeAppend" ||
      op === "RopeAppendInterleaved"
    )
      return "attention";
    if (op === "RmsNorm" || op === "LayerNorm") return "norm";
    if (op === "Silu" || op === "Gelu" || op === "TanhSoftCap")
      return "activation";
    if (op === "Add" || op === "BiasAdd") return "residual";
    if (op === "Reshape") return "reshape";
    return "other";
  }
  if (node.kind === "binop") {
    return node.op === "*" ? "elem" : "residual";
  }
  if (node.kind === "weight") return "weight";
  if (node.kind === "extern") return "extern";
  if (node.kind === "local") return "local";
  if (node.kind === "scalar" || node.kind === "scalar_sym") return "scalar";
  return "other";
}

const PALETTE: Record<
  string,
  { bg: string; border: string; label: string }
> = {
  embed: {
    bg: "bg-fuchsia-900/40",
    border: "border-fuchsia-500/60",
    label: "text-fuchsia-200",
  },
  matmul: {
    bg: "bg-sky-900/50",
    border: "border-sky-500/60",
    label: "text-sky-200",
  },
  attention: {
    bg: "bg-orange-900/40",
    border: "border-orange-500/60",
    label: "text-orange-200",
  },
  norm: {
    bg: "bg-emerald-900/40",
    border: "border-emerald-500/60",
    label: "text-emerald-200",
  },
  activation: {
    bg: "bg-violet-900/40",
    border: "border-violet-500/60",
    label: "text-violet-200",
  },
  residual: {
    bg: "bg-rose-900/40",
    border: "border-rose-500/60",
    label: "text-rose-200",
  },
  elem: {
    bg: "bg-amber-900/30",
    border: "border-amber-500/50",
    label: "text-amber-200",
  },
  reshape: {
    bg: "bg-slate-700/40",
    border: "border-slate-500/40",
    label: "text-slate-200",
  },
  weight: {
    bg: "bg-emerald-950/60",
    border: "border-emerald-700/50",
    label: "text-emerald-300",
  },
  extern: {
    bg: "bg-amber-950/60",
    border: "border-amber-700/50",
    label: "text-amber-300",
  },
  local: {
    bg: "bg-ink-700/60",
    border: "border-ink-500",
    label: "text-ink-200",
  },
  scalar: {
    bg: "bg-violet-950/60",
    border: "border-violet-700/50",
    label: "text-violet-300",
  },
  other: {
    bg: "bg-ink-700/60",
    border: "border-ink-500",
    label: "text-ink-200",
  },
};

function nodeLabel(node: StencilExpr): string {
  switch (node.kind) {
    case "call":
      return node.op.toLowerCase();
    case "binop":
      return node.op;
    case "local":
      return node.name;
    case "weight":
      return node.name + (node.index !== null ? `[${node.index}]` : "");
    case "extern":
      return (
        node.name.toLowerCase() +
        (node.index !== null ? `[${node.index}]` : "")
      );
    case "scalar":
      return String(node.value);
    case "scalar_sym":
      return node.name;
  }
}

/** Recursive AST renderer. Composite nodes are an outer rectangle
 * with the op label, containing their children laid out horizontally
 * inside; leaves are a single small rectangle with their label. */
function ExprNode({
  expr,
  compact,
  selectedLocal,
  onSelectLocal,
  params,
  implPick,
  depth = 0,
}: {
  expr: StencilExpr;
  compact: boolean;
  selectedLocal: string | null;
  onSelectLocal?: (name: string) => void;
  params?: Record<string, number>;
  /** Implementation picked for the enclosing assign (if known).
   * Attached to the outermost composite chip only — nested children
   * don't get their own pick since the solver assigns impls at the
   * subgraph level, which the macro typically scopes to an assign. */
  implPick?: string;
  depth?: number;
}) {
  // Locals get the shared chip — same rendering & selection behavior
  // as assign LHS, so click-to-track works regardless of where a name
  // appears.
  if (expr.kind === "local") {
    return (
      <LocalChip
        name={expr.name}
        compact={compact}
        selectedLocal={selectedLocal}
        onSelectLocal={onSelectLocal}
      />
    );
  }
  const fam = familyOf(expr);
  const s = PALETTE[fam] ?? PALETTE.other;
  const isComposite = expr.kind === "call" || expr.kind === "binop";

  if (!isComposite) {
    // Leaf shape per kind:
    //  - locals  → rounded "pill" (variable bindings flow through the
    //              graph; a pill makes them visually distinct from op
    //              boxes so a row reads as `<pill> ← <op-tree>`).
    //  - weights → small square (immutable, distinct from runtime values).
    //  - other leaves → standard rounded rectangle.
    // `local` is handled above by the early-return into LocalChip;
    // remaining leaves are weight / extern / scalar / scalar_sym.
    const shape = expr.kind === "weight" ? "rounded-sm" : "rounded";
    if (compact) {
      // Compact mode: just a small colored rectangle, no text. Keeps
      // the silhouette legible at thumbnail scale.
      return (
        <span
          className={`${s.bg} ${s.border} ${shape} border inline-block self-center`}
          style={{ width: 10, height: 16 }}
          title={nodeLabel(expr)}
        />
      );
    }
    return (
      <span
        className={`${s.bg} ${s.border} ${s.label} ${shape} border px-1.5 py-0.5 inline-flex items-center whitespace-nowrap text-[12px] self-center`}
        title={nodeLabel(expr)}
      >
        {nodeLabel(expr)}
      </span>
    );
  }

  // Composite: header label + nested children. Always render children
  // (even in compact mode) — the whole point of the silhouette is to
  // SEE the nesting, not just the outermost op.
  const children = expr.children;
  return (
    <span
      className={`${s.bg} ${s.border} border rounded inline-flex items-center gap-1 self-center ${
        compact ? "p-0.5" : "p-1"
      }`}
    >
      {/* Always show the op name — even in thumbnails. The whole
           reason the math view exists is so the eye can read off
           gemm/rmsnorm/silu at a glance; hiding labels in compact
           mode defeats the point. Just shrink them. */}
      <span
        className={`${s.label} font-bold tracking-wide px-1 ${
          compact ? "text-[11px]" : "text-[12px]"
        }`}
      >
        {nodeLabel(expr)}
      </span>
      <BindingChips expr={expr} params={params} compact={compact} />
      {implPick && depth === 0 && (
        <ImplChip name={implPick} compact={compact} />
      )}
      {children.length > 0 && (
        <span className="inline-flex items-center gap-1 flex-wrap">
          {children.map((c, i) => (
            <ExprNode
              key={i}
              expr={c}
              compact={compact}
              selectedLocal={selectedLocal}
              onSelectLocal={onSelectLocal}
              params={params}
              depth={depth + 1}
            />
          ))}
        </span>
      )}
    </span>
  );
}

/** Render the implicit kernel bindings for a call op as sub-chips on
 * the op box. Compact mode collapses to a single violet dot; zoom
 * mode shows `ε=1e-5 θ=1M d=128 …` style annotations. Leaves binops
 * untouched (they have no baked params). */
function BindingChips({
  expr,
  params,
  compact,
}: {
  expr: StencilExpr;
  params?: Record<string, number>;
  compact: boolean;
}) {
  if (expr.kind !== "call") return null;
  const names = implicitBindingsFor(expr.op);
  if (names.length === 0 || !params) return null;
  const resolved = names
    .map((n) => {
      const v = params[n];
      return v === undefined
        ? null
        : { name: n, value: formatBindingValue(v) };
    })
    .filter((x): x is { name: string; value: string } => x !== null);
  if (resolved.length === 0) return null;
  if (compact) {
    return (
      <span
        className="bg-violet-400/80 rounded-full inline-block self-center"
        style={{ width: 6, height: 6 }}
        title={resolved.map((b) => `${b.name}=${b.value}`).join("  ·  ")}
      />
    );
  }
  return (
    <span className="inline-flex items-center gap-1">
      {resolved.map((b) => (
        <span
          key={b.name}
          className="bg-violet-950/70 border border-violet-700/60 rounded px-1.5 py-0.5 text-[10px] whitespace-nowrap self-center"
          title={b.name}
        >
          <span className="text-violet-400">{shortBindingName(b.name)}</span>
          <span className="text-violet-600 mx-0.5">=</span>
          <span className="text-violet-100">{b.value}</span>
        </span>
      ))}
    </span>
  );
}

/** Impl pick: which kernel/library actually runs this op at the
 * current (variant, workload). Styled distinctly from binding chips
 * (amber, like an implementation tag) so the eye can separate "what
 * the kernel reads" (violet ε/θ/…) from "what the kernel is" (amber
 * fused_qkv_rope_cache / flashinfer_attention_decode / …). */
function ImplChip({ name, compact }: { name: string; compact: boolean }) {
  const d = describeImpl(name);
  if (compact) {
    return (
      <span
        className="bg-amber-400/80 rounded-sm inline-block self-center"
        style={{ width: 6, height: 6 }}
        title={`${d.kernel} — ${d.note}\n(${name})`}
      />
    );
  }
  return (
    <span
      className="bg-amber-950/70 border border-amber-700/60 rounded px-1.5 py-0.5 text-[10px] whitespace-nowrap self-center inline-flex items-center gap-1"
      title={`${d.note}\n\nimpl: ${name}`}
    >
      <span className="text-amber-500">⚙</span>
      <span className="text-amber-200 font-semibold">{d.kernel}</span>
      {d.kernel !== name && d.kernel !== "?" && (
        <span className="text-amber-500/70">·</span>
      )}
      {d.kernel !== name && (
        <span className="text-amber-400/80 font-normal">{name}</span>
      )}
    </span>
  );
}

/** Abbreviate long binding names so the sub-chips stay tight. */
function shortBindingName(name: string): string {
  return (
    {
      rms_norm_eps: "ε",
      layer_norm_eps: "ε",
      rope_theta: "θ",
      partial_rotary_factor: "prf",
      head_dim: "d",
      sliding_window: "swin",
      attn_logit_softcapping: "cap",
    }[name] ?? name
  );
}
