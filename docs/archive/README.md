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

| File | Reason archived | Superseded by |
|---|---|---|
| `2026-06-02-REVIEW_FINDINGS.md` | One-shot code-review log; work landed | n/a (historical) |
| `2026-06-02-watch-api.md` | `/api/watch/*` HTTP reference; merged into `web-api.md` § 8 | [`../protocols/web-api.md`](../protocols/web-api.md) |
| `2026-06-02-tabs-watch.md` | Legacy `/watch` UI spec; trajectory inspector lives on `/results` post-2026-05-19 | [`../protocols/results-tab.md`](../protocols/results-tab.md) + [`../protocols/inspector-registry.md`](../protocols/inspector-registry.md) |
