# Tab-UI static audit — 2026-08-05

**Role:** audit report
**Domain:** web
**Audited:** 2026-08-05 (HEAD `853a3f9`)
**Scope:** every tab page and its controller — the six templates in
`molbuilder/web/templates/`, the per-tab controllers under
`molbuilder/web/static/{modify,structure-optimization,spectra,transport,results,documents}/`,
the shared modules under `static/lib/` they mount, and the tab stylesheets.
Method: static only — no browser, no test run.
**Companions:** [`tabs.md`](?doc=web/tabs.md) — the contract these pages are read
against. [`results.md`](?doc=web/results.md) § 2 — the picker contract § E1 cites.
[`vibrationview.md`](?doc=web/vibrationview.md) — the module § A1 finds dead.
[`css-system-plan.md`](?doc=plans/css-system-plan.md) — the plan § D belongs to.
[`process/code-audit.md`](?doc=process/code-audit.md) — the playbook.

---

## How to read this

Every finding carries the evidence that established it, so an item can be
executed — or dismissed — without re-deriving it. **Nothing here has been
changed.** Each item is a proposal awaiting a decision.

Findings are grouped by what they cost:

| § | Group | Costs |
|---|---|---|
| **A** | Broken wiring | a user hits it today |
| **B** | Obsolete residue | reading time, and it misleads |
| **C** | Duplication | every future edit, N times |
| **D** | CSS collisions | cross-tab visual inconsistency |
| **E** | Contract drift | the document and the code disagree |

Status column: `open` unless a commit says otherwise.

---

## A. Broken wiring — user-visible today

### A1. VibrationView cannot mount on either page that loads it — `open`

`lib/vibrationview/vibrationview.js:72` borrows the drawing surface at runtime:

```js
var viewer = mb.viewer;   // shared-embed borrow: RUNTIME lookup
if (!host || !viewer || typeof viewer.embed !== "function") {
    return _failMount("missing host or shared viewer embed (load order?)");
}
```

**`window.molbuilder.viewer` is published nowhere that any page loads.** The two
assignments are `lib/viewer/mol-viewer.js:38` and
`lib/viewer/mol-viewer-embed.js:6498`, and no template has a `<script>` tag for
either file. `_molview_scripts.html` — the one include every viewer page uses —
loads `vendor/3Dmol-min.js`, `lib/molview/index.js`, `lib/workspace/dispatcher.js`
and `lib/warning-modal.js`, and nothing else.

MolView cannot supply it either. Its graph is closed (`index.js` → `mount.js` →
`3dmol-embed.js`/`model.js`/`render-engine.js`/`ui.js`, none of which import
`lib/viewer/`), and `index.js:16` forbids the direction outright: *"NEVER … write
or read `window.molbuilder`, in either direction."* There is no dynamic
`import()` and no injected `<script>` for it anywhere in the tree.

**Effect.** On `/spectrum-calculation` and `/results`, clicking a mode reaches
`lib/spectra/core.js:2178`, gets `{ok: false}` back, and renders
**"vibration viewer unavailable (missing host or shared viewer embed (load
order?))"**. The normal-mode animation is dead on both tabs.

**Why the suite is green.** `tests/test_vibrationview_mount_js.py:42` installs
`global.molbuilder.viewer = { embed: … }` — a stub for the exact seam that broke.
The module's own logic is fine and its 5 tests are honest about that; nothing
tests that a *page* provides the dependency.

**Stale claim to remove with the fix.** `spectra.html:243` still says the include
*"provides 3Dmol + mol-viewer-embed, which VibrationView also borrows for the
mode viewer."*

**Ties to:** task #19 / #104 (full MolView/VibrationView separation) and the
queued VibrationView ESM redesign. The module's single external dependency is a
global nobody publishes — which is the argument for the redesign, in one line.

### A2. The Spectrum tab silently has no detection chip — `open`

`molbuilder/config/spectra.py` tags 32 fields with `workflow_group`, so
`form-schema.js` renders the `.workflow-group--profile` and
`.workflow-group--budget` cards on `/spectrum-calculation`. But:

- `spectra.html` never loads `lib/detection-chip.js` (`index.html:330` and
  `transport_calculation.html:214` do);
- `spectra/viewer.js` never calls `detectionChip.render`.

So the chip surface exists and stays empty. `structure-optimization/viewer.js:1180-1185`
documents this exact defect being fixed *for Transport* on 2026-06-13 — *"Pre-2026-06-13
these helpers were closure-private here, which silently denied Transport
(Au-junction users!) any chip surface."* The helper was extracted; Spectrum was
never wired to it. Same defect, same cause (§ C2), still live.

### A3. Transport's Generate status element is never written — `open`

`transport_calculation.html:131` places a live region beside the Generate button
in card 4:

```html
<span id="transport-generate-status" class="muted" role="status" aria-live="polite"></span>
```

No JavaScript ever touches that id. Every generate message —
`"Generating…"`, `"Generated <file>."`, `"Form has invalid values"`,
`"Network error"` — goes through `lib/transport/core.js:35 _setStatus`, which
writes `#transport-status`: the readout in **card 1**, above the viewer. The user
clicks Generate at the bottom of the page and the answer appears off-screen at
the top.

Either wire the card-4 span and retire the card-1 write for generate messages, or
delete the span. It is the only orphaned id in the tree that was clearly meant to
do something.

---

## B. Obsolete residue

### B1. `lib/viewer/` — all five files, 6,884 lines, dead — `open`

No live `import`, no `<script>` tag, no dynamic load reaches any of them:

| file | lines |
|---|---|
| `mol-viewer-embed.js` | 6,531 |
| `mol-axes.js` | 231 |
| `mol-style.js` | 49 |
| `mol-viewer.js` | 38 |
| `mol-format.js` | 35 |
| **total** | **6,884** |

Every live consumer imports from MolView's door instead:
`modify/viewer.js:23`, `structure-optimization/viewer.js:14`,
`spectra/viewer.js:37`, `lib/trajectory/core.js:34`,
`lib/transport/core.js:23`, `modify/selection-bootstrap.js:20` — all
`from "/static/lib/molview/index.js"`.

Eight live files still *claim* otherwise in comments, and are wrong:

- `modify/viewer.js:292` — "The Hill formula() belongs to the molview module
  (static/lib/viewer/mol-format.js) -- we `import`" — it imports from
  `molview/index.js`, which re-exports `formula` from `./_atom.js`.
- `structure-optimization/viewer.js:1196` — same claim, same correction.
- `lib/trajectory/core.js:838` — "Sizing math lives in
  `molbuilder/web/static/lib/mol-style.js`" — a path that has never existed.
- `lib/xyz-io.js:11`, `spectra/viewer.js:112`, `lib/trajectory/core.js:255`,
  `lib/molbuilder-runtime.js:33-35`, `lib/vibrationview/vibrationview.js:33`.

`tests/test_css_molview_namespace.py:59` already names it *"the retired 3Dmol
embed: loaded by no page, imported by no module"*, and `mol-viewer-embed.css` was
deleted. The JavaScript never followed.

**This is the largest single deletion available in the front end** — 6,884 lines
with zero behaviour change, *after* A1 is resolved. Order matters: while
VibrationView still asks for `molbuilder.viewer`, deleting the directory removes
the file that was supposed to answer.

### B2. `lib/transport/core.js`'s header is false end to end — `open`

Lines 1-22 state:

> *"Generate is intentionally disabled — engine backends (TranSIESTA, PySCF-NEGF)
> land in a follow-up phase; until then the form is 'configure now, generate
> later' UX … The commit doesn't trigger a script render today — Generate stays
> disabled … Design ref: docs/web/tabs.md (Transport tab — Phase D form
> skeleton)."*

Generate works: `lib/transport/core.js:575-637` POSTs `/api/transport/render`,
renders the script preview, and offers Copy + download.
[`tabs.md`](?doc=web/tabs.md) § 4 already opens with a warning to *ignore this
file's comments* — the document worked around the residue rather than the residue
being cleared. The cited § name no longer exists.

### B3. Thirteen stale asset paths cited in live files — `open`

Each of these is named in a comment or a template in the live tree; none exists:

| cited path | cited from |
|---|---|
| `lib/molview/data-model.js` | `lib/workspace/dispatcher.js:10` |
| `lib/selection-panel.js` | `lib/molbuilder-runtime.js`, `modify/viewer.js` |
| `lib/selection-panel.css` | `lib/page-shell.css`, `modify/style.css` |
| `lib/selection/viewer-adapter.js` | `lib/molbuilder-runtime.js` |
| `lib/trajectory/csv-export.js` | `lib/trajectory/core.js` |
| `lib/trajectory/result-list.js` | `lib/results/file-picker.js` |
| `lib/projects-sidebar.js` | `lib/projects/projects-sidebar.css` |
| `lib/viewer/mol-viewer-embed.css` | `lib/tokens.css:192` |
| `static/spectra/page.js` | `lib/spectra/core.js`, `_spectra_inspector.html` |
| `static/watch/viewer.js` | `lib/trajectory/core.js`, `_trajectory_inspector.html` |
| `static/style.css` | `spectra/style.css` |
| `static/viewer.js` | `spectra/viewer.js` |
| `static/lib/mol-style.js` | `lib/trajectory/core.js:838` |

Two of them are in **templates**, not comments, which is where a reader looks
first.

### B4. The runtime module registry lists 12 names; 1 is real — `open`

`lib/molbuilder-runtime.js:33-47` presents a list headed *"Naming: flat, dotted,
lowercased. Today's modules:"* — `viewer`, `style`, `fmt`, `formSchema`,
`projects`, `workspace`, `selection.panel`, `selection.viewerAdapter`,
`structure.page`, `structure.save`, `modify.loadStructureText`, `inspectors`.

The only `runtime.register(…)` calls in live code are
`lib/projects/projects-sidebar.js:51` (`"projects"`) and
`lib/viewer/mol-viewer-embed.js:6525` (`"viewer"`, from the dead directory).
The list is a wish, not an inventory, and it is the first thing a new reader
consults for "what modules exist".

### B5. Two functions defined and never called — `open`

- `structure-optimization/viewer.js:80 _facts()` — a three-line wrapper under a
  twelve-line docstring, left over from the `factsForRequest()` → `getStructure()`
  migration.
- `lib/trajectory/core.js:2363 _frozenFingerprint(data)` — guarded an incremental
  force-arrow append that no longer exists. `lib/trajectory/core.js:2554-2556`
  confirms the current path: MolView's `setForces` *"re-bakes the arrow overlay IN
  PLACE"* every poll. Not a lost guard — residue of the pre-MolView path.

Every other function in every tab controller has a caller (checked
mechanically; the three other hits were IIFE names).

### B6. Orphaned template ids — `open`

`inspect-card` and `parameters-card` (`transport_calculation.html:44`, `:98`;
`spectra.html`) are referenced by no JS, no CSS, and no label. Harmless, but they
read as hooks that something uses. (`transport-generate-status` is the third
orphan and is § A3, because it was clearly meant to work.)

---

## C. Duplication

### C1. The "Analyze chemistry" card is hand-pasted three times — `open`

`index.html:105`, `spectra.html:96` and `transport_calculation.html:70` carry the
same markup with the **same ids** — `analyze-card`, `auto-detect-btn`,
`auto-detect-status`, `auto-detect-panel`, `auto-detect-rationale`,
`auto-detect-warnings`, `auto-detect-metals` — differing only in the hint prose
and the button label.

This repo already uses shared partials for exactly this: `_projects_sidebar.html`,
`_molview_scripts.html`, `_system_load_monitor.html`, `_bundle_handoff_panel.html`,
`_spectra_inspector.html`, `_trajectory_inspector.html`. There is no
`_analyze_chemistry_card.html`.

### C2. `_renderAutoDetectPanel` exists three times — `open`

`structure-optimization/viewer.js:1132-1179` and `spectra/viewer.js:437-471` are
**byte-identical for 40 lines**, differing only in `$` vs `_$` and the trailing
`_renderWorkflowGroupChips(resp)` call — whose absence from the Spectrum copy
*is* § A2. `lib/transport/core.js:433-470` is a third, `var`-style copy.

Behind them, five hand-rolled `POST /api/structure/analyze` call sites, each with
its own `AbortController` and status text: `structure-optimization/viewer.js:1008`
and `:1062`, `spectra/viewer.js:336` and `:392`, `lib/transport/core.js:373`.

### C3. The load-from-sidebar block is a copy that has since diverged — `open`

`structure-optimization/viewer.js:855-960` vs `spectra/viewer.js:140-250`:
`_isLoadable` and `_basename` are byte-identical; `_refreshLoadButton` has drifted.
The optimization tab appends `· unsaved changes` to the readout when the model is
dirty; the Spectrum copy has no such branch. Same widget, same ids, same
template markup — different behaviour, from a copy that was edited once.

### C4. `basename` written three more times — `open`

`lib/path-utils.js:5` states its own reason for existing: *"so the five inspector
modules … don't each carry their own copy of `basename()`"*. Still hand-rolled in
`structure-optimization/viewer.js:866`, `spectra/viewer.js:151`, and inline as
`cut()` in `lib/inspectors/trajectory.js:109`. Three further sites carry the
`(root.molbuilder && root.molbuilder.path && …) || fallback` dance
(`lib/inspectors/source.js:280`, `structure.js:330`,
`_partial_inspector_factory.js:38`).

Note `spectra.html` and `modify.html` do not load `path-utils.js` at all, which is
why their copies exist.

### C5. `setStatus` written six times — `open`

`structure-optimization/viewer.js:190`, `spectra/viewer.js:47`,
`lib/transport/core.js:35`, `modify/viewer.js:271`, `lib/spectra/core.js:490`,
`lib/trajectory/core.js:690`. Six signatures, three shapes
(`(id,msg,kind)` / `(msg,kind)` / `(el,msg,kind)`).

### C6. Blob download ×4, clipboard copy ×3 — `open`

Download: `structure-optimization/viewer.js:1244 downloadAs`,
`lib/transport/core.js:523`, `lib/spectra/core.js:1042 downloadScript` and
`:1874`. Copy: `lib/transport/core.js:546`, `lib/spectra/core.js:1055` and
`:1095`. MolView has its own sealed one (`lib/molview/ui.js:742`), correctly
private to the module.

### C7. `POST /api/run/install-wrapper` written three times — `open`

`structure-optimization/viewer.js:1705` (SIESTA) and `:2028` (PySCF — its own
comment says *"mirrors the SIESTA path"*), plus `lib/spectra/core.js:952`.
Transport has none, which is the known "no in-app save to project" gap
([`tabs.md`](?doc=web/tabs.md) § 4).

### C8. Three independent `<dialog>` scaffolds — `open`

`lib/warning-modal.js`, `lib/projects/dialogs.js` and
`modify/structure/save-dialog.js` (which holds two) each re-implement: create the
element, attach a `cancel` listener, `showModal()` with a fallback, guard against
a second settle, `close()`. `lib/projects/dialogs.js:17` says so out loud —
*"Each dialog runs the warning-modal pattern (save-dialog.js's idiom)"*. The
pattern was named and then copied instead of extracted.

---

## D. CSS — tab sheets overriding the shell

Belongs to [`css-system-plan.md`](?doc=plans/css-system-plan.md); these are the
concrete instances found in the tab sheets.

### D1. `spectra/style.css` redefines shell rules with different values — `open`

`spectra.html` loads `lib/page-shell.css` (line 7) then `spectra/style.css`
(line 21). Same specificity, so the tab sheet wins:

| selector | `page-shell.css` | `spectra/style.css` |
|---|---|---|
| `.card` | `border-strong`, `var(--radius)`, `14px 18px 18px`, `box-shadow` | `border-soft`, `12px`, `1rem 1.4rem 1.2rem`, no shadow |
| `button` | `var(--radius-sm)`, `4px 12px` | `6px`, `0.42rem 0.95rem`, own `font-size` |
| `button:disabled` | `opacity: 0.6` | `opacity: 0.45` |

The Spectrum tab's cards and buttons therefore do not match any other tab's.

### D2. `.status` diverges across three tab sheets — `open`

`lib/page-shell.css:336` owns it (`text-secondary`, `0.88rem`, `margin-left: 4px`,
`white-space: pre-line`). `modify/style.css:106` overrides colour to
`--text-muted` and margin to `auto`; `spectra/style.css:133` overrides size to
`0.85rem`; `structure-optimization/style.css:107` adds `margin-top`. Each sheet
correctly leaves the severity modifiers to the shell and says so — the base rule
is what drifted.

### D3. `modify/style.css` redefines `header` — `open`

`border-soft` instead of `border-strong`, and drops the shell's gradient
background.

---

## E. Contract drift

### E1. `/results` has a second route to a mounted viewer — `open`

[`results.md`](?doc=web/results.md) § 2.1: *"There is no second route to a mounted
viewer."* There is one.

`results/viewer.js:265-271` reads `proj.getCurrentFile()` at init and mounts it
directly, bypassing the picker:

```js
const initialFile = (typeof proj.getCurrentFile === "function")
    ? proj.getCurrentFile() : "";
if (initialFile) { _onSelectionChange({ file: initialFile }); }
```

`_onSelectionChange` calls `reg.pick(file)` — **not** `pickResult` — so a
non-result file (a `.fdf`, a README, any `.log`) matches the source or markdown
presenter and mounts. The panel then shows a file the dropdown does not list,
until the picker's async scan resolves and replaces it. Transient, but it is
precisely the shape § 2.2 forbids: two derivations of "which file is current".

It exists to cover one case: the picker's `keepCurrent` branch
(`lib/results/file-picker.js:617-625`) labels the menu without announcing, so
nothing would mount on a return visit. That branch is right for a *re-scan* —
re-announcing would tear down and rebuild a live viewer, which § 4 forbids — but
it is unnecessary on the **first** scan, when nothing is mounted yet.

**Proposal:** have the picker announce unconditionally on its first scan and
delete `results/viewer.js:265-271`. One route, as the contract says.

### E2. The structure presenter writes `sessionStorage` directly — `open`

`lib/inspectors/structure.js:100-118` implements the "Open in Molbuilder" hand-off
by writing `C.SS_FILE` / `C.SS_DIR` itself, instead of calling
`projects.setShared(dir, file)` — the door every other consumer uses and the one
that fans out `onChange`. [`projects.md`](?doc=web/projects.md) names `setShared`
as the selection door.

### E3. Six backend routes the UI never calls — `needs a decision`

Not defects — ship-or-retire calls (task #26):

| route | note |
|---|---|
| `/api/backends` | no caller anywhere |
| `/api/docs/list` | the Documents tab uses `/api/docs/toc`; only `tests/test_docs_tab.py` calls `list` |
| `/api/checkpoint/config` | no caller |
| `/api/checkpoint/diff` | no caller |
| `/api/checkpoint/migrate-manifest` | the CLI has the same command (`cli.py:2668`) |
| `/api/admin/rate_limit/{status,clear}` | admin-only; may be intentional |

Conversely, every `/api/*` path the front end names resolves to a registered
route — no dead client-side endpoints.

---

## What was checked and found clean

Recorded so a later pass doesn't redo it:

- **Element ids.** Every id in every template is referenced by JS, a label, or
  CSS, except the three in § A3 / § B6. Every id the JS looks up exists, except
  `s-es-selection` (`lib/spectra/core.js:682`), which `form-schema.js` builds from
  `molbuilder/config/spectra.py:433` — a false positive.
- **Endpoints.** No JS names a route that does not exist.
- **Dead local functions.** Only the two in § B5.
- **Dialog/notification roles.** `warning-modal` (confirm), `app-notifications`
  (stack), `projects/dialogs` (sidebar mutations) are three distinct jobs, not
  three copies — only their scaffolding is duplicated (§ C8).
- **The save door.** Both generator tabs write through
  `projects.saveToWorkspace`; neither hand-rolls a write.

---

## Suggested order

1. **A1** — a shipped feature is dead on two tabs, and it is the entry point to
   the VibrationView redesign.
2. **B1** — 6,884 lines removed, zero behaviour change, but only *after* A1 (the
   directory is what A1's missing global lives in).
3. **A2 + C1 + C2 together** — extracting the Analyze-chemistry card and its
   renderer is what stops A2 recurring; fixing A2 alone leaves the third copy to
   drift again.
4. **B2, B3, B4, B5, B6** — comment and dead-code sweep; independent, cheap.
5. **E1, E2** — contract alignment.
6. **A3, C3-C8, D** — the rest, in whatever order suits the work in flight.
   § D folds into [`css-system-plan.md`](?doc=plans/css-system-plan.md).
