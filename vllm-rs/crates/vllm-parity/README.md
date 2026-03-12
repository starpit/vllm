# vllm-page

Static web page generator for the vllm-rs project overview and feature parity dashboard.

## What it does

Compiles `parity.csv` into a self-contained HTML file at build time. The output is a single-page app (SPA) with two views:

- **Overview** (`#/`) — hero section with "Why Rust?" cards, overall parity percentage, a dense feature grid (each cell = one feature, colored by status), and a performance placeholder table
- **Feature Parity** (`#/parity`) — full interactive table with all parity data, collapsible sections, search/filter, and color-coded status badges

## Input files

| File | Purpose |
|------|---------|
| `../../parity.csv` | Feature parity data (section, feature, python status, rust status, notes, deprecated flag) |
| `templates/index.html` | HTML/CSS/JS template with placeholder tokens for embedded data |

## How it works

1. **`build.rs`** reads `parity.csv`, parses it with the `csv` crate, computes per-section summaries, and writes two JSON files (`parity_rows.json`, `parity_summaries.json`) to `OUT_DIR`
2. **`src/main.rs`** uses `include_str!` to embed the HTML template and generated JSON into the binary, performs string replacement, and outputs the final HTML
3. The HTML template uses [PatternFly](https://www.patternfly.org/)-inspired styling via [Inter](https://fonts.google.com/specimen/Inter) / [JetBrains Mono](https://fonts.google.com/specimen/JetBrains+Mono) from Google Fonts — no other CDN dependencies
4. Client-side hash routing (`#/` and `#/parity`) — vanilla JS, no framework

## Build & run

```bash
# Build
cargo build -p vllm-page

# Generate HTML to stdout
cargo run -p vllm-page > /tmp/index.html

# Generate HTML to a file
cargo run -p vllm-page -- -o docs/index.html

# Open in browser
open /tmp/index.html
```

## Feature grid

The overview page shows a dense grid where each 10x10px cell represents one feature from `parity.csv`:

- **Blue** (`#1f78b4`) — implemented in Rust
- **Pink** (`#fb9a99`) — partial implementation
- **Red** (`#e31a1c`) — not yet implemented
- **Gray** — N/A or deprecated

Hovering a cell shows a tooltip with the feature name, status, and notes. Clicking a cell navigates to the corresponding row in the detail table.

## Dependencies

Build-only (not linked into the final binary's runtime):

- `csv` — CSV parsing
- `serde` / `serde_json` — serialization to JSON

The output HTML has zero runtime dependencies beyond a browser.
