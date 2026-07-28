# Archive

Historical documents. **Not a source of truth.** If you are reading a
date-prefixed file, you are reading history, not policy.

## Conventions (docs/README.md rule R4)

- Every archived doc is prefixed `YYYY-MM-DD-<original-name>.md` — the date
  it was archived, so the boundary between history and policy is visible in
  the filename itself.
- Every archived doc gets a row in the table below naming **what superseded
  it** (the canonical doc to read instead) — or "closed / historical" when
  it was a one-shot log whose work is complete. Superseded-by entries are
  doc *names*, not links: canonical docs move as the tree evolves, history
  doesn't chase them (find the current home in `../README.md`).
- **Substance-first rule** (learned the hard way, 2026-06-02): the substance
  of every still-live contract must be migrated into the canonical doc
  BEFORE the archive move. An archived file must never be the only place a
  live invariant lives. (The 2026-06-02 pass initially over-compressed and
  had to restore ~15 dropped contracts — see web-api.md § 16's gap audit.)
- Two sources feed this folder: **migration archiving** (docs from the
  frozen `old_docs/` tree whose reconcile gate found them superseded or
  retired — their ledger row in `../MIGRATION.md` says `archived`) and
  **ongoing archiving** (a live doc superseded later; move it here with the
  date prefix in the same commit that supersedes it).
- `audits/` holds point-in-time audit/analysis snapshots imported as whole
  directories — findings that were acted on; kept for the record.

## Index

| Archived doc | Reason | Substance lives in |
|---|---|---|
| `2026-06-02-REVIEW_FINDINGS.md` | Closed — one-shot code-review log; work landed | n/a (historical) |
| `2026-06-02-watch-api.md` | Superseded — `/api/watch/*` HTTP reference | web-api.md § 8 (endpoint table, Mode A/B, `/api/watch/data` shape, `MOLBUILDER_WATCH_ROOT`, concurrency, security) |
| `2026-06-02-tabs-watch.md` | Superseded — legacy `/watch` UI spec; trajectory inspector lives on `/results` since 2026-05-19 | inspector-registry.md § 6; cross-cutting front-end conventions → web-api.md § 14.4 |
| `2026-07-03-embedded-viewer.md` | Superseded — standalone embedded-viewer contract, folded into the MolView module | molview-module.md (the viewer / handle) |
| `2026-07-03-atom-selection.md` | Superseded — standalone atom-selection spec, folded into the MolView module | molview-module.md (store, composition, measurement) |
| `2026-07-05-browser-data-contract.md` | Superseded — browser-owned working-copy contract; the "changed-underneath" hash-gate was removed | workspace-contract.md § 4 (persistence) |
| `2026-07-05-working-copy-persistence.md` | Superseded — load/edit/save working-copy persistence spec | workspace-contract.md § 4 + § 4.6 |
| `2026-07-06-molview-module.md` | Superseded — the 2026-07-03 standalone MolView design+contract snapshot | molview-module.md (viewer + selection + k-grid + measurement) |
| `2026-07-06-workspace-state.md` | Closed — the 2026-06-07 workspace-unification audit + Phases 1–9 migration log (all shipped); kept for the *why* | workspace-contract.md (the live model) |
| `audits/audit-2026-06-26/` | Closed — whole-repo audit snapshot (synthesis + top findings + CSS/UI + test depth) | findings acted on in code/tests |
| `audits/audit-2026-06-27/` | Closed — follow-up audit snapshot | findings acted on in code/tests |
| `audits/job-case-analysis/` | Closed — ANALYSIS-G1K1C4 job-case study (bench/job-execution milestone 2026-06-29) | conclusions folded into the bench/staged-execution contracts |
| `2026-07-28-decisions-log.md` | Closed — the verbatim 113-entry decisions log from the pre-migration `design.md` (Wave 9); kept for the *why* behind each decision | design.md § Decisions (indexes the load-bearing entries → each domain doc) |
| `old_docs/` | **Frozen legacy tree** — the pre-migration docs (86 files, snapshot 2026-07-26) migrated to the domain-structured `docs/` tree across Waves 0–9 (2026-07-26 to 2026-07-28) | `docs/README.md` (live index); `docs/archive/MIGRATION.md` (per-file map) |

> Note (2026-07-26): the archived docs above predate the docs-tree
> reorganization, so *internal* relative links inside them point at the old
> tree layout and may dangle. That is expected — they are verbatim history
> (the link-integrity test exempts `archive/`).
