# Audit synthesis — cross-cutting top 10

**Date**: 2026-06-26
**Inputs**: T1 (top-30 sweep), T3 (CSS/UI), T4 (test depth)
**Author**: Claude (parent loop), reading the three subagent reports.

The three audits ran independently and converged on the same four
patterns.  Each pattern is concrete, traceable to file:line evidence,
and shows up in more than one report — that's how this list is
ranked.  No code changes here; the user picks what to act on.

---

## Pattern 1 — Today's checkpoint stack repeatedly half-implements its contract

Three independent audits flagged the same surface:

* **T1 BLOCKER 2** — `web/blueprints/checkpoint.py:95-104`: advisory
  envelope uses `errors_only: True` (bool) + `errors:[...]` instead of
  `errors_only: list[Issue]` + `issues:` per `web-api.md` § 1.1.
* **T1 IMPORTANT 1** — `run-checkpoints.md` § 8 still advertises
  `/branch`, `/prune`, and a `branches/tags` shape on `list` that the
  blueprint does not implement.
* **T1 IMPORTANT 2-3** — `ws.ui.checkpoint.collapsed` is read but
  never written; `ws.ui.checkpoint.view` is persisted via raw
  `sessionStorage` (violates the same-day `workspace-contract.md`
  tightening).
* **T1 IMPORTANT 4** — `checkpoint.py:773-778` `state()` walks
  `.binsnapshots/` recursively on every 5 s sidebar poll.
* **T3 IMPORTANT 2** — `projects-sidebar.css:1340-1590` (the run-history
  panel added today) hardcodes 11 state colours instead of using the
  `--ps-*` family designed exactly for this.
* **T4 BLOCKER 1** — `run-checkpoints.md` § 12 names
  `test_checkpoint_sensor_js.py` + `test_checkpoint_graph_e2e.py` as
  required test targets; neither file exists.

Read together: today's checkpoint PR-B shipped the happy path and
deferred the contract-conformance work in **four directions at once**
(envelope shape, sidebar persistence, theme tokens, JS tests).  The
Python side of the same feature is the audit's gold standard.  This
is the only finding in the audit that all three sub-audits land on
independently.

## Pattern 2 — Silent defaults that swallow contract violations

* **T1 BLOCKER 1** — `transport/results.py:200`:
  `d.get("schema_version", SCHEMA_VERSION)` defaults missing to "2", so
  v1 sidecars (which omit the field) slip through as v2 with empty
  regions + frozen_atoms.  Exactly the empty-boundary record the
  function's own docstring says is refused.
* **T1 BLOCKER 3** — `parse/engines/siesta.py:1456-1585`: build/diag
  probes accept any regex match without validating the captured token,
  so a future SIESTA print shape silently becomes "what SIESTA was
  compiled with."
* **T1 IMPORTANT 10** — `web/blueprints/checkpoint.py:117-121`
  `_get_body` returns `request.get_json(silent=True) or {}` —
  malformed JSON coerces to "no body," surfacing as misleading
  "missing parameter" 400 instead of "request body is not valid JSON."
* **T4 BLOCKER 2** — sidecar three-stage contract's "engine MUST warn
  on unknown labels — no silent absorption" rule is gated only for
  the PySCF spectra engine.  SIESTA + Transport have positive-only
  coverage; the load-bearing negative case is missing.

The shape is the same every time: success-path `or DEFAULT` /
`get(..., DEFAULT)` / `silent=True` swallows the case the contract says
to reject.  These are the bugs the user already burned days on with
the 2026-06-25 v2-schema fix and the 2026-06-23 SIESTA keyword fix.
The class hasn't been retired.

## Pattern 3 — Token coverage drift concentrated in today's commits

Token system is well-architected, but the new code added today is
where drift accumulates:

* **T3 IMPORTANT 1** — 7 undefined tokens silently fall back to literals
  (`--surface-elevated`, `--border-subtle`, `--ok`/`--warn`/`--bad`,
  `--accent-on`, `--ps-bg-2`).  Highest-leverage one-line fixes.
* **T3 IMPORTANT 2** — Run-history panel + markdown inspector are pure
  raw-hex / raw-rem (11 untokenised state colours + ~520 magic-number
  sites across the codebase).
* **T3 IMPORTANT 4** — `--font-mono` defined but used in ~5% of mono
  stacks; 25 sites declare three different ad-hoc stacks.
* **T1 cross-link**: `system-load-monitor.js:64-68` reads three of the
  undefined tokens via `getComputedStyle`.  Possibly broken at
  runtime — needs a quick check.

`!important` use (2) and naming convention are both clean — the
discipline exists, it just hasn't been applied to today's new
components.

## Pattern 4 — False-confidence shape tests inflate the green count

* **T4 IMPORTANT 1** — Tab-markup substring tests would pass on an
  id/data-tab swap.
* **T4 IMPORTANT 2** — 113 sites use bare `assert r.status_code == 200`
  without also asserting `body["ok"]` — a regression that pushes a
  clean run into the advisory bucket (web-api.md § 1.6 bucket B)
  passes the status check.
* **T4 NIT 1** — `tests/test_web.py:104` is an `assert True` tombstone
  inflating the green-count.
* **T4 IMPORTANT 5** — No test gates `Cache-Control: no-store`; no
  adversarial path test (`%2e%2e`, symlink escape).

This is the cliff between "test count high" and "contract-gating high."
The user's worry was specifically about this; T4's verdict is the
suite is better than feared but the gap is real.

---

## Top-10 prioritized

Ranked by **damage potential × cost-to-fix**.  Pattern membership in
parentheses.  All items are file:line-traceable in the source reports.

| # | Finding | Severity | Source | Pattern |
|---|---|---|---|---|
| 1 | `TransportResults.from_dict` silently accepts v1 sidecars as v2 (empty regions / frozen_atoms) | BLOCKER | T1 BLOCKER 1 | P2 |
| 2 | Checkpoint blueprint advisory envelope diverges from `web-api.md` § 1.1 (`errors_only:True` + `errors:[]`) | BLOCKER | T1 BLOCKER 2 | P1 |
| 3 | Missing JS tests `test_checkpoint_sensor_js.py` + `test_checkpoint_graph_e2e.py` (named in design doc § 12) | BLOCKER | T4 BLOCKER 1 | P1 |
| 4 | ~~"Warn on unknown label" gate missing for SIESTA + Transport engines (PySCF only)~~ — **VERIFIED FALSE (2026-06-26)**, see note below | ~~BLOCKER~~ | T4 BLOCKER 2 | P2 |
| 5 | SIESTA build/diag probes accept any regex match without validating the captured token | BLOCKER | T1 BLOCKER 3 | P2 |
| 6 | 7 undefined CSS tokens silently fall back to literals | IMPORTANT | T3 IMPORTANT 1 | P3 |
| 7 | Run-history panel (today's commit) introduces 11 untokenised state colours | IMPORTANT | T3 IMPORTANT 2 | P1+P3 |
| 8 | `ws.ui.checkpoint.collapsed` read but never written; `ws.ui.checkpoint.view` uses raw `sessionStorage` | IMPORTANT | T1 IMPORTANT 2-3 | P1 |
| 9 | `Repo.state()` walks `.binsnapshots/` recursively at every 5 s sidebar poll | IMPORTANT | T1 IMPORTANT 4 | P1 |
| 10 | 113 sites use bare `assert r.status_code == 200` without `body["ok"]` check | IMPORTANT | T4 IMPORTANT 2 | P4 |

### Finding #4 — verified FALSE on impl-state check (2026-06-26)

T4 BLOCKER 2 claimed the "warn on unknown label" gate is implemented
"only for the PySCF spectra engine."  Tracing the actual code refuted
this:

* **Pattern B (warn on a label-type the engine doesn't consume —
  regions):** implemented AND tested for the SIESTA build path
  (`build.py:716` → `_shared.regions_pattern_b_notice`; test
  `test_web.py::test_fdf_surfaces_info_when_structure_carries_regions`),
  the PySCF build path (`build.py:789`; `test_web.py::test_pyscf_...`),
  and spectra (`spectra/test_blueprint.py`).  In-body label path also
  gated (`test_in_body_labels_xhr.py:210`).
* **Frozen-atom index not in range:** validated for both build paths in
  the shared carrier `apply_labels_to_struct` (`_shared.py`).
* **Transport** deliberately omits Pattern B because it *is* the region
  consumer — documented at `transport.py:134` (post-review 2026-06-10).

So the gate is NOT PySCF-only; the BLOCKER framing does not survive
verification (consistent with the audit-claim-verification pattern in
prior rounds).  The **one** genuine residue — TranSIESTA preflight
warned on *missing* expected regions but silently ignored *extra/unknown*
region labels — was a NIT, now fixed: `TransiestaEngine.preflight`
emits a warn naming any region label outside the canonical 2-terminal
set (`transiesta.py`; test
`test_transport_transiesta.py::test_preflight_warns_on_unknown_region_label`).

---

## Cheapest sweep that catches the most

Two regression tests, each ~50 lines, would have caught **6 of the
top 10**:

1. **Envelope round-trip test per `/api/checkpoint/*` route** —
   asserts response satisfies `web-api.md` § 1.1 (`errors_only` is a
   list, key is `issues`, advisory bucket carries HTTP 200).  Catches
   item 2 directly; same shape catches similar drift in any future
   blueprint.
2. **`from_dict(missing_field)` fixture per sidecar schema** — for
   each schema_version-bearing dataclass, assert that
   `from_dict({})` (no `schema_version`) raises.  Catches item 1
   directly; same shape catches the next version-rollover drift.

Plus one-line cleanups:

3. Alias the 7 undefined CSS tokens in `lib/tokens.css` — item 6.
4. Drop the `assert True` tombstone in `test_web.py:104` — T4 NIT 1.

Plus the meta-test that closes the class itself:

5. `test_design_doc_test_targets.py` — parses every
   `docs/protocols/*.md` § "Testing" / § "Test coverage" section,
   extracts named test files, asserts each exists.  Catches item 3
   directly; closes the "doc names test file that doesn't exist"
   class permanently.

---

## What I'm NOT recommending

* **T2 deep dives** — T1 found 20 substantive items, not 200.  T2
  (per-subsystem) is only worth running on a subsystem where T1
  flagged systemic drift; today's evidence is that the checkpoint
  stack is the only such subsystem, and it's already half-T2'd by T1.
* **T5 architecture cleanup** — T1 + T3 found discipline patterns
  (naming, layering, `!important` use) are clean.  The "what to
  split, merge, delete" surface is small; T5 is a low-ROI write
  until something concrete demands it.
* **Refactor of the spacing scale** — T3's 520 magic-number count is
  large but the drift is bounded (today's commits + older form-schema
  code).  A rem-scale token addition is a designer call, not a
  reviewer call; out of scope for "what to fix from this audit."
* **Inline fix proposals** — every report keeps to "surface, don't
  patch."  Suggested actions are one-sentence; the real fix work is
  separate scoped commits driven by user pick.

---

## Honesty notes

* T1 explicitly skipped spectra, auth/rate-limit, embedded viewer,
  3DNA/nucleic builders, and the vendored `@gitgraph/js` bundle.
  See T1 § "What you did NOT review."
* T3 audited the 16 first-party CSS files; vendor (`codemirror`,
  `dompurify`) excluded.
* T4 sampled 15 test files in depth and spot-checked ~30 more by name
  + header; the full 229-file census was out of scope.
* The 80% / 55% / 5% token coverage numbers in T3 are estimates, not
  a census.

Items the audit DID NOT verify but flagged as worth a runtime check:

* `system-load-monitor.js` reading 3 undefined CSS tokens — does
  the canvas actually render the intended colour or fall back to
  black?
* `test_au_bdt_au_transmission_against_literature` — verify it's
  actually skipping rather than passing trivially.

---

## Suggested next action

Pick item 1, 2, 3, or "do the cheap sweep" (all four items in
"Cheapest sweep that catches the most").  The cheap sweep is the
recommended start — it neutralises 5 of the top 10 in roughly a
day's work and prevents the same class of drift from recurring.
