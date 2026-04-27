# Vendored sources

These directories are upstream third-party source trees, snapshotted
in-tree per `MEGA_HANDOFF.md` ("Vendor in-tree"). Treat as ours to
tweak — local edits land here, not as a fork.

## megakernels/

- Upstream: https://github.com/HazyResearch/Megakernels
- Branch: `throughput` (carries both prefill and decode; the `main`
  branch is decode-only, so we vendor `throughput` to get the full
  forward).
- Commit: `91eaff262c2b473cfdcb135f5f2abefbe2835fe9` (`throughput`).
- Imported: 2026-04-27
- Excluded from copy: `.git/`, `__pycache__/`
- License: MIT. Restored into `megakernels/LICENSE` from the
  blob at upstream `main`'s tip (commit
  `7309cec801537b61fea3b50d7dfe454a6cde578e`).
  Branch comparison shows `throughput` is strictly ahead of `main`
  (3 commits ahead, 0 behind; merge-base = `main`'s tip
  `7309cec`). The throughput branch's "initial copy-paste" commit
  `bff2e9c` accidentally deleted the LICENSE that exists on the
  merge-base, so restoring from `main`'s blob is just recovering
  what `throughput` lost — same MIT terms, same authors,
  same commit history. If upstream re-adds LICENSE to
  `throughput`, drop the out-of-band copy and re-snapshot.

## thunderkittens/

- Upstream: https://github.com/HazyResearch/ThunderKittens
- Commit: `0b55588d2769dde0c7a9a606ffc015b0ca7a9551`
- Imported: 2026-04-27
- Excluded from copy: `.git/`, `assets/*.png` (doc images, not
  needed for build)
- License: see `thunderkittens/LICENSE` (upstream carries an
  explicit license file).
