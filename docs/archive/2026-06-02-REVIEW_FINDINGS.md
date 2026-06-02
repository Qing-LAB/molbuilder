# Codebase review findings — 2026-05-16

Holistic review across Python backend, frontend (templates + JS), and
test suite.  Categories: **OBSOLETE** (stale shim / dead code),
**DUPLICATED** (same logic / contract written twice), **OVERENG**
(infrastructure bigger than the problem), **GAP** (design commitment
without test or implementation).

Each item carries: location, severity, status (✅ fixed, ⏸ deferred,
🟡 in progress, ⬜ unaddressed), and a short rationale for the chosen
path.  Items marked ⏸ are intentionally deferred — most often pending
the planned /results tab merge (task #58) that will reshape the
Watch + Spectra split.

---

## P0 — correctness risk

### 10. Watch blueprint under-tested ⏸
- **Where:** `molbuilder/web/blueprints/watch.py` (766 LOC) vs
  `tests/watch/` (91 tests, almost all parser-side).  Only ~5-10
  tests exercise the actual HTTP routes.
- **Category:** GAP
- **Status:** deferred until /results tab merge (task #58) reshapes
  what Watch is.  Designing detailed tests around the current Watch
  surface would burn effort that gets thrown away.
- **Trigger to re-open:** task #58 starts, OR Watch is left as-is.

### 11. Cross-tab handoff one-directional ⏸
- **Where:** only `Modify → Build` is E2E-tested
  (`tests/test_modify_e2e.py::test_send_to_build_*`).  No coverage
  for `Build → Watch`, `Watch → Spectra`, `Spectra → Watch`.
- **Category:** GAP
- **Status:** deferred — handoffs to / from Watch are part of the
  /results tab merge (#58).  Add E2E coverage after the new tab
  shape stabilises.
- **Trigger to re-open:** with #58 — write handoff tests against the
  new tab topology, not the legacy one.

---

## P1 — easy wins (address now)

### 1. `molbuilder/molwatch_log/` shim ✅
- **Where:** `molbuilder/molwatch_log/__init__.py` (13 LOC).
  Re-exports `trajectory_log.*` for back-compat with pre-merge
  imports.
- **Category:** OBSOLETE
- **Resolved:** single remaining consumer
  (`tests/test_molwatch_preview.py:23`) was migrated to import from
  `molbuilder.trajectory_log`; shim package deleted; design.md
  layout block updated.  29 tests still pass.

### 2. Legacy form-state migration in Build viewer.js ✅
- **Where:** `molbuilder/web/static/viewer.js` had a
  `LEGACY_ID_MIGRATION` dict + restoreFormState loop for the
  2026-05-11 schema cutover (basis_size + kgrid id renames).
- **Category:** OBSOLETE
- **Resolved:** dict + migration loop removed.  Users with stale
  pre-cutover sessionStorage lose at most 4 saved values on next
  reload (the schema-driven form supplies its own defaults).  The
  outer save/restore pattern stays in place for future use.

### 4. Form-schema alias drift ✅
- **Where:** Build's `viewer.js` had two `window.molbuilder.formSchema.collectForm(...)` sites that bypassed the local `fs` alias used everywhere else.
- **Category:** DUPLICATED (alias proliferation)
- **Resolved:** both sites converted to use `const fs = (window.molbuilder || {}).formSchema;` first then `fs.collectForm(...)`, matching the pattern Spectra's viewer + Build's initFormsFromSchema already used. Zero `window.molbuilder.formSchema` direct call sites remain.

### 12. Layering invariant unenforced ✅
- **Where:** `docs/design.md` commits "L1 modules cannot import
  higher ones" — no test pinned the import direction.
- **Category:** GAP
- **Resolved:** `tests/test_layering.py` parses every .py file under
  `molbuilder/`, classifies it L1/L2/L3 against tables that mirror
  design.md, and asserts no module imports from a higher layer.
  Plus a sanity test ensuring every new top-level name lands in
  exactly one layer table. 47 modules verified clean; 13 re-export
  __init__ shims explicitly skipped.

### 13. 3DNA error message format unverified ✅ (no change required)
- **Where:** per project memory, error must point at `x3dna.org`.
- **Category:** GAP (audit was wrong)
- **Resolved:** the audit missed two existing tests in
  `tests/test_backends.py:214,234` that already cover the message
  contract (`x3dna.org`, "non-commercial", in-tree / X3DNA / PATH
  detection points, amber/rdkit fallbacks named) AND the
  `dispatch("dna", backend="threedna")` raises-with-message path.
  Both pass.

### 14. Spectra doesn't load `mol-style.js` ✅ (partial — audit was misleading)
- **Where:** `spectra.html` didn't load `mol-style.js`; Spectra's
  setStyle calls use a bespoke visual model (thin stick + small
  sphere for eigenmode aesthetics, grey-out for fixed atoms).
- **Category:** DUPLICATED (the framing) / GAP (the inclusion)
- **Resolved:** added `mol-style.js` script tag to `spectra.html`
  for cross-tab consistency.  Did NOT route Spectra's call sites
  through it — Spectra deliberately uses a different visual style
  than Build/Modify/Watch (no rep-dropdown, dark-bg
  eigenmode-animation aesthetic).  Forcing it through the shared
  rep dropdown would erase Spectra-specific UX.

---

## P2 — architectural cleanup (address now)

### 5. Build / Modify "bypass" `projects/state.js` ✅ (audit was wrong)
- **Where:** Build / Modify / Watch viewer.js files use
  `sessionStorage` for `builder-form`, `builder-structure`,
  `modify-state`, `watch-path`.
- **Category:** misdiagnosed
- **Resolved:** these are tab-private state keys (form values,
  built molecule, op stack, last typed path).  `projects/state.js`
  legitimately owns ONLY the cross-tab `molbuilder.current_dir` +
  `molbuilder.current_file` selection.  No bypass; the separation
  is correct.  Centralising tab-private keys behind the projects
  API would be premature abstraction.

### 7. Spectra XYZ helpers belong in `lib/` ✅
- **Where:** `spectra/viewer.js` had `_parseXyz` + `_geomToXyz` (33
  LOC) — not Spectra-specific.
- **Category:** OVERENG (file size); DUPLICATED-IN-WAITING
- **Resolved:** extracted to
  `molbuilder/web/static/lib/xyz-io.js` (global namespace
  `window.molbuilder.xyz.{parse,toText}`, matching the
  `mol-format.js` convention).  Spectra viewer + template wired
  through.  Audit's "150 LOC" estimate was wrong — only the two
  pure I/O helpers were generic; the surrounding
  `_equilibriumGeometry` is Spectra-specific (reaches into
  `state.results`) and stays put.

### 8. No shared 3Dmol viewer factory ✅
- **Where:** all four `viewer.js` files called
  `$3Dmol.createViewer()` independently — three identical calls
  (Build / Modify / Watch with white bg + Jmol colours) plus
  Spectra's dark-bg variant.
- **Category:** DUPLICATED
- **Resolved:** added `molbuilder/web/static/lib/mol-viewer.js`
  with a `create(target, opts?)` factory that defaults to
  white/Jmol and accepts overrides.  All four viewer.js files now
  go through `window.molbuilder.viewer.create(...)`.  Templates
  load the lib before their viewer.js.  Zero raw
  `$3Dmol.createViewer` calls remain outside the factory.

---

## P2 — defer with notes

### 3. `_shared.py` naming overstates universality ⏸
- **Where:** `molbuilder/web/blueprints/_shared.py` (577 LOC) is
  only imported by build / modify / spectra (NOT watch / files).
- **Category:** OBSOLETE (naming)
- **Status:** defer — the helpers themselves are correctly scoped;
  only the name is misleading.  Renaming touches 3 blueprints for
  zero behaviour change.
- **Trigger to re-open:** any time the file is being edited for
  another reason — rename in passing.

### 6. Modify unit + E2E redundancy ⏸
- **Where:** `tests/test_modify.py` (68) + `tests/test_modify_e2e.py`
  (66) for a 610-LOC blueprint.  Some E2E tests cover behavior
  already pinned in the unit file.
- **Category:** OVERENG (test density)
- **Status:** defer — the 16 most-redundant HTTP tests in
  `test_web.py` were just cut; another aggressive cut on E2E could
  remove signal we'd want later.  Better to wait for the next time
  Modify is refactored and decide which E2Es no longer pull weight.
- **Trigger to re-open:** Modify tab refactor / when E2E suite gets
  too slow.

### 9. siesta/ vs pyscf/ input.py parallel silos ⏸
- **Where:** `molbuilder/siesta/input.py` (1006 LOC) +
  `molbuilder/pyscf/input.py` (1059 LOC).  No shared helpers for
  atom labelling, convergence formatting, header comments.
- **Category:** OVERENG-PREMATURE-REFACTOR
- **Status:** defer — two engines is below the "Rule of Three"
  threshold.  Refactor only when a third engine is added.
- **Trigger to re-open:** any new engine (e.g., CP2K) request.

### 15. Pseudopotential module absent ⏸
- **Where:** `projects/` skeleton reserves `pseudopotential/` subdir;
  no Python module yet.
- **Category:** GAP (design ahead of code)
- **Status:** already tracked as task #55.  Not part of this review's
  scope.

---

---

## Discovered during cleanup (not in original audit)

### 16. `package-data` glob misses three static subdirs ✅
- **Where:** `pyproject.toml` only listed `web/static/watch/*.{css,js}`;
  the `modify/`, `spectra/`, and `lib/projects/` subdirs were absent.
- **Category:** GAP (silently broken for `pip install`)
- **Resolved:** added `web/static/{modify,spectra}/*.{css,js}` and
  `web/static/lib/projects/*.js` to the package-data globs.  No
  visible effect in dev (`python -m molbuilder` loads from the
  source tree) but a `pip install` build would now ship every
  template's required assets.
- **Why this matters:** the package-data globs are the only
  contract for what lands in a wheel; per-tab subdirs that aren't
  globbed get silently dropped.

---

---

## Tab merge migration (task #58) — progress as of 2026-05-17

Per ``docs/protocols/results-tab.md`` § 4.  Each step is its own
milestone with its own tests so a failure surfaces close to its
cause.

| Step | Status | Notes |
|---|---|---|
| 3. /results blueprint + template + dispatch shell | ✅ | `web/blueprints/results.py` + `templates/results.html` + `static/results/viewer.js`; nav entry added; `/results` added to `test_pages_no_js_errors.py`.  Superseded by the registry redesign on 2026-05-17 (single host element + registry-driven dispatch). |
| **Inspector Registry foundation** (new, supersedes hardcoded panels) | ✅ | `lib/inspectors/registry.js` (contract) + `lib/inspectors/{source,structure,trajectory,spectra}.js` (4 modules; source + structure REAL, trajectory + spectra placeholders pending steps 1B+/2).  31 tests in `test_results_blueprint.py` (script load order + interface + per-extension match pins).  17 Playwright-gated tests in `test_inspector_registry_e2e.py` (live pick/mount/dispose lifecycle). |
| 1A. Extract trajectory-inspector DOM to partial | ✅ | `templates/_trajectory_inspector.html`; watch.html now includes it; 8 tests in `test_trajectory_inspector_partial.py`. |
| 1B. Scope JS DOM lookups to a root element | ✅ | `watch/viewer.js`'s IIFE body wrapped in `mountInspector(rootEl)`; 38 partial-id `$()` calls scoped via `rootEl.querySelector`; 4 page-level loader-bar ids (path-input, load-btn, status, file-picker) on `$doc()` document-wide.  Plus 2026-05-17 guard pass: every `$doc(...)` dereference now goes through a captured-and-guarded local (`const _el = $doc("id"); if (_el) _el.foo`) so `mountInspector(panel)` on /results no longer NPEs against the missing loader bar.  Auto-bootstrap preserves /watch.  8 invariant tests in `test_trajectory_inspector_partial.py::TestViewerJsRootScoping` (including the new `test_no_unguarded_dollar_doc_dereference` which blocks regressions).  Sets up 1C as a mechanical lift. |
| 1C. Lift logic into `lib/inspectors/trajectory.js` (real impl) | ⬜ | Replaces the current placeholder body; same registry interface, same name, so registry/dispatch unchanged.  Depends on 1B. |
| 1D. `/results` shows real trajectory inspector for `.molwatch.log` | ⬜ | Blocked on 1C + the five Stage-1D gaps below (partial-cloning, opts.file, Plotly script, watch CSS, dispatcher de-dupe). |

### Gaps surfaced by this round's review (Stage 1D readiness)

These aren't bugs in current code — they're missing pieces the trajectory inspector needs when it goes live in `/results`.  Capturing them here so Stage 1C/1D start with a clear backlog.

| Gap | Where it bites | Recommended approach |
|---|---|---|
| **No partial-cloning mechanism for the inspector module.** The current `trajectory.js` placeholder injects placeholder markup directly. The real implementation needs the partial's DOM.  | When `lib/inspectors/trajectory.js`'s real mount() runs against an empty `#inspector-host` on /results. | Wrap the partial in `<template id="trajectory-inspector-template">…{% include … %}</template>` in BOTH watch.html and results.html.  Inspector clones `template.content.cloneNode(true)` into the host on mount.  Same partial, two consumption sites, no duplication. |
| ~~`mountInspector(rootEl)` has no `opts.file` parameter.~~ | ~~Registry can't pass the file to load.~~ | ✅ Resolved 2026-05-17 round 3.  Signature now `mountInspector(rootEl, opts={file?})`; auto-loads when `opts.file` set.  Plus `window.molbuilder.trajectoryInspector.mount` exposed for the registry-side inspector to delegate to. |
| ~~Disposal of 3Dmol + polling timers when an inspector is replaced.~~ | ~~File-swap leaks timers + WebGL context.~~ | ✅ Resolved 2026-05-17 round 3.  Handle's `dispose()` now stops poll + play timers, removes the window resize listener (with `cancelAnimationFrame`), and tears down 3Dmol bookkeeping (removeAllShapes/Labels/Models).  Test pins each cleanup. |
| ~~XSS via innerHTML string-concat in placeholder inspectors.~~ | ~~File-path interpolated into trajectory.js/spectra.js innerHTML.~~ | ✅ Resolved 2026-05-17 round 3.  Both placeholders rewritten with textContent + createElement DOM construction.  Test pins "no innerHTML+concat" pattern in either placeholder. |
| **Plotly is not loaded on /results.** The trajectory inspector uses Plotly for energy/force/SCF plots. | First mount of a `.molwatch.log` on /results. | Add `<script src="{{ url_for('vendor_plotly_js') }}"></script>` to `results.html`'s head.  Same source /spectra already uses. |
| **`watch/style.css` is not loaded on /results.** The partial-resident DOM (viewer-card, frame-strip, ctab panels, plots-row) relies on selectors from watch/style.css.  Will render unstyled inside /results. | First mount on /results. | Either (a) load watch/style.css on /results (low effort, slight conceptual leak — /results carries /watch styling), or (b) extract the inspector-scoped rules into a shared `lib/trajectory-inspector.css` that both pages load (cleaner, slightly more work). Recommend (b). |
| **Dispatcher remounts on every selection change**, even no-op same-file re-fires.  Trajectory inspector does a full /api/watch/load + 3Dmol rebuild + Plotly init per mount — expensive when fired redundantly. | Every onChange event from the sidebar, including same-file noise from internal refresh / focus changes. | In `results/viewer.js::_onSelectionChange`, track `lastFile + lastInspectorName`; skip dispose+mount when both are unchanged.  ~5 lines. |
| 2A. Spectra inspector partial extraction (mirrors 1A) | ⬜ | Extract `spectra.html`'s inspect-side scaffolding (mode viewer + table + Plotly chart) into `_spectra_inspector.html`.  Same magnitude as 1A. |
| 2B. Scope spectra/viewer.js inspect-side `$()` calls to root | ⬜ | Same shape as 1B for spectra. |
| 2C. Lift logic into `lib/inspectors/spectra.js` (real impl) | ⬜ | Replaces the current placeholder body.  Depends on 2A + 2B. |
| 2D. `/results` shows real spectra inspector for `.spectra.json` | ⬜ | Automatic once 2C lands. |
| 5. Remove inspect-side UI from Spectra tab | ⬜ | Blocked on 2C+2D — don't strip until /results replaces it. |
| 6. Remove Watch from primary nav | ⬜ | Blocked on 1D — no point removing the entry while /results trajectory is a placeholder. |
| 7. Cross-page E2E dispatch tests | ⏸ partial | `test_inspector_registry_e2e.py` already covers the registry + dispatch.  Full coverage of real-inspector behaviour lands with 1D + 2D. |
| 8. Update docs/tabs/ | ⬜ | Add `results.md`, mark `watch.md` legacy, mark `spectra/spec.md` form-only.  Blocked on 5 + 6. |

## How to use this file

When picking up cleanup:

1. Pick a ⬜ item.
2. Verify the diagnosis (don't trust the audit blindly; the agent
   reports were grep-based and may miss subtleties).
3. Implement the action.
4. Flip the status marker (⬜ → ✅) and add a one-line note on what
   actually shipped.
5. Cross-reference any new tests / files that landed.

Items marked ⏸ should be re-evaluated when their **Trigger to
re-open** condition lands, NOT proactively.
