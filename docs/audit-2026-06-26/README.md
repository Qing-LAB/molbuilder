# Codebase audit — 2026-06-26

**Status**: T1 + T3 + T4 in progress; T2 + T5 queued.
**Owner**: Claude (this audit), reporting back to Quan.

Each tier ships as a standalone report under this directory.  No code
changes in the audit commits — findings only.  Fix work follows in
separate, scope-tagged commits driven by the report verdicts.

---

## Why this exists

User-driven 2026-06-26 after a self-review pass on today's commits
surfaced 6 real bugs in ~half-day of work.  The implication: a
focused audit of the rest of the code base will surface more.
This audit is the framework for finding them once, ranking them,
and deciding what to fix.

The audit asks five questions per subsystem:

1. **Code vs. design documents** — drift in either direction?
2. **Gaps / inconsistencies / errors** — does the code actually do
   what it claims?  Are there silent failure modes?
3. **Obsolete and duplicated code** — modules nobody uses, two
   modules doing the same thing.
4. **Tests** — do they pinpoint design contracts, or just exercise
   code paths superficially?
5. **Architecture** — modularization, code reuse, magic numbers,
   theme tokens, framework vs hacking.

---

## Tiered plan

| Tier | Scope | Effort | Output | What it catches |
|---|---|---|---|---|
| **T1** | Top-30 findings sweep across the whole repo | ~half day | `T1_top_findings.md` — each finding ranked BLOCKER / IMPORTANT / NIT, with file:line + evidence | Big drift, obvious dead modules, broken invariants, tests that don't gate design |
| **T2** | Per-subsystem deep dives, modules picked from T1 priorities | ~1 day per module | `T2_<subsystem>.md` (one file per module) | Module-by-module exhaustive findings |
| **T3** | CSS / UI holistic audit | ~half day | `T3_css_ui.md` | Magic numbers, theme-token drift, layout-framework inconsistencies, "hacky patches vs framework" |
| **T4** | Test-depth audit | ~half day | `T4_test_depth.md` | Each test rated: gates a design contract, gates a code path only, or doesn't gate anything meaningful |
| **T5** | Architecture-level cleanup plan | ~half day | `T5_architecture.md` | What to split, merge, delete; ranked by ROI |

---

## Today's execution: T1 + T3 + T4

Launched 2026-06-26 evening in parallel via three subagents.  Each
agent has been briefed to:

* Read the relevant design docs first (`docs/design.md` + the
  relevant subsystem docs under `docs/protocols/`).
* Spot-check actual code against the design.
* Report findings ranked BLOCKER / IMPORTANT / NIT with file:line
  evidence.
* NOT propose fixes inline — the report's job is to surface, not to
  patch.  Fix work is a separate human-driven decision.

When all three reports land, I (Claude) synthesize them into a
cross-cutting top-10 list and commit everything together.  User
then picks which findings to act on.

---

## Backlog

| Item | Status | When |
|---|---|---|
| T1 — top-30 sweep | in progress | today |
| T3 — CSS/UI audit | in progress | today |
| T4 — test-depth audit | in progress | today |
| T2 — per-subsystem deep dives | queued | after T1 findings rank them |
| T5 — architecture cleanup plan | queued | after T1 + T3 inform it |

After today, the queue becomes: act on T1+T3+T4 findings (separate
fix commits) → revisit whether T2 is needed for any subsystem the
findings flagged → write T5 once we have enough signal.
