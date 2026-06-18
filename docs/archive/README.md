# Archive

Historical doc snapshots. **NOT a source of truth.** Read
[`../design.md`](../design.md) for the current architecture and
the canonical subsystem docs under [`../protocols/`](../protocols/)
and [`../tabs/`](../tabs/).

Files here are date-prefixed (`YYYY-MM-DD-<original-name>.md`) so
the archive boundary is unambiguous: if you find yourself reading
a date-prefixed file, you are reading history, not policy.

Each archived doc has either been:

- **Superseded** by a canonical doc that supersedes it. In each
  case the still-valid content was migrated into the canonical
  doc before the original was moved here.
- **Closed**: a one-shot review or migration log whose work is
  complete; kept as historical reference but not maintained.

## Index

| File | Reason archived | Substance migrated to |
|---|---|---|
| `2026-06-02-REVIEW_FINDINGS.md` | One-shot code-review log; work landed | n/a (historical) |
| `2026-06-02-watch-api.md` | `/api/watch/*` HTTP reference | [`../protocols/web-api.md`](../protocols/web-api.md) § 8 — endpoint table, Mode A / Mode B distinction, full `/api/watch/data` shape, `MOLBUILDER_WATCH_ROOT`, concurrency contract, security model |
| `2026-06-02-tabs-watch.md` | Legacy `/watch` UI spec; trajectory inspector lives on `/results` post-2026-05-19 | [`../protocols/inspector-registry.md`](../protocols/inspector-registry.md) § 6 — partial layout, engine-specific UI adaptation, state invariants, polling cadence, dual-mode loader, status messages, forbidden patterns. Cross-cutting front-end conventions (3Dmol CDN pin, textContent rule, theme) → `web-api.md` § 14.4 |

## Audit principle

When archiving a doc, the substance of every still-live contract
must be migrated to the canonical doc BEFORE the archive move.
The archived file is a snapshot for historical reference; it
must not be the only place a live invariant lives. The audit
that produced this archive (2026-06-02) initially over-compressed
the migration; the corrected version restored ~15 substantive
contracts that had been dropped (see the gap audit in
`web-api.md` § 16).
