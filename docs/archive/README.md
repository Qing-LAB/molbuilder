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
- **Whole-tree snapshots are the one exception to the date-prefix rule**:
  `audits/…` directories and `old_docs/` (the pre-migration docs tree,
  86 files, snapshot 2026-07-26) are imported verbatim as directories.
  The snapshot's date and identity live in its Index row here — not in
  per-file prefixes, which would break the per-file map in
  `MIGRATION.md` and the trees' own internal links.

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
| `2026-08-04-closed-tasks.md` | Closed — seven tasks finished 2026-08-02→04 (#41 the one label store, #44 MolView persistency, #48 server reload, #50 the app locking its own user out, #51 read-only tabs keep their structure, #52 label-chip colours, #53 opening a file with an unusable box), moved off the working list; kept for the decisions **and the corrections** — three record a thing believed, measured, and found false | the live contracts each one changed (`web-api.md` § 1, `molview.md` § 11.2a, `access-control.md` § 7, `structure-periodicity.md` § 8.2) |
| `2026-08-10-stage-chaining.md` | Superseded — a ladder was a SCHEDULER chain: `Job.depends_on` / `Job.dep_kind` edges, `Carry` symlinks laid before the producer ran, `carry_deref` to localize them, `--chain`, and SIESTA's `on_nonconvergence` whose whole effect was to pick the edge kind. Retired by user decision 2026-08-10 on scientific grounds — whether a later stage should pick up an earlier one cannot be settled without reviewing the earlier one's result — and an opt-in flag was rejected with it. Kept as the ONE home for the retired vocabulary, so the live contracts state what the system is rather than what it stopped being | `project-layout.md` § 1.6 (a person prepares each stage and names what it continues from), `job-system.md` § 2 decision 6 (a JobSet has no edges) and § 3 (`WarmFile` — what a job would take, never from whom) |

> Note (2026-07-26): the archived docs above predate the docs-tree
> reorganization, so *internal* relative links inside them point at the old
> tree layout and may dangle. That is expected — they are verbatim history
> (the link-integrity test exempts `archive/`).
