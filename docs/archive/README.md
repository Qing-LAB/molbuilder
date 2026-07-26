# Archive

Historical documents. **Not a source of truth.** If you are reading a
date-prefixed file, you are reading history, not policy.

## Conventions (docs/README.md rule R4)

- Every archived doc is prefixed `YYYY-MM-DD-<original-name>.md` — the date
  it was archived, so the boundary between history and policy is visible in
  the filename itself.
- Every archived doc gets a row in the table below naming **what superseded
  it** (the canonical doc to read instead) — or "retired, no successor"
  when the surface itself is gone.
- Two sources feed this folder:
  1. **Migration archiving** — docs from the frozen `old_docs/` tree whose
     reconcile gate (docs/MIGRATION.md) found them superseded or retired.
     Their ledger row says `archived`; the row below says why.
  2. **Ongoing archiving** — a live doc in `docs/` superseded later. Move
     it here with the date prefix in the same commit that supersedes it.
- `audits/` holds point-in-time audit snapshots (whole-directory imports
  from old_docs: `audit-2026-06-26/`, `audit-2026-06-27/`,
  `job-case-analysis/`) — findings that were acted on; kept for the record.

## Index

| Archived doc | Superseded by / why archived |
|---|---|
| *(none yet — populated as migration reconcile gates archive docs)* | |
