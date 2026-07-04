# Tab architecture and navigation

> **This document is the sole source of truth for molbuilder's
> tab-level UI architecture: the tab inventory, routes, the
> cross-tab workflow model, and the staging of the 2026-06-06
> reorganization.**  Per-tab specs in `docs/tabs/` (e.g.
> `molbuilder.md`, `structure-optimization.md`) own the *internal*
> contract of each tab; this doc owns *how the tabs relate*.
>
> When a phase lands, update both this doc AND the per-tab spec
> for the tab that changed.  When a new cross-tab principle gets
> decided, it lands here first, then propagates into the per-tab
> specs.  Pointer in `design.md` § 0 (UI tabs).

Status (2026-06-07): **Phases A + B.1 + B.2 + B.5 complete.**
Phase C (PySCF `.out` → `.pyscf.log` + Results parser) + Phase D
(Transport-calculation form skeleton) still planned.

  * Phase A landed the 5-tab nav + canonical routes.  Initial
    plan included 301 redirects from legacy paths; those were
    **removed** in the post-rename pass — the canonical name is
    the single source of truth, no aliases.  See § 3.2.
  * Phase B.1 + B.2 landed the canvas-state / warning-modal
    primitives, the Sources card on the Molbuilder tab, the
    SMILES generator, the Save panel, and the dirty-canvas
    warning across the workspace.
  * Phase B.5 landed the universal sidebar interaction model
    (single-click = preview, dblclick = commit) across all
    non-Results tabs, form-dirty gates on Build + Spectra, the
    sidebar hide/show toggle, and the file-type filter.

Generators not yet migrated (3DNA, peptide, name, file upload)
+ stripping generators from the Build form remain on the
roadmap as B.3 + B.4; today the Molbuilder tab ships the Load /
SMILES / Save subset of the foldable panel set described in
§ 5.1.

---

## 1. Mission

After 5+ rounds of practical use, the current four-tab layout
(`Build` / `Modify` / `Spectra` / `Results`) surfaced two
structural problems:

1. **The Build / Modify split breaks the structure workflow.**
   A user generates a molecule in Build (SMILES, 3DNA, RDKit),
   saves it, then has to *switch tabs* and *reload the file* to
   edit it.  Build and Modify are sequential phases of one
   workflow ("make a structure"), not parallel features.
2. **The tabs don't name what they do.**  "Build" generates two
   different artifact classes (a 3-D structure AND a job script);
   "Spectra" reads as if it's a viewer rather than a task
   generator.

The reorganization merges Build's structure-generation paths into
Modify, renames everything to match what each tab actually does,
adds a placeholder for the planned Transport tab, and decouples
interactive editing from script generation.

---

## 2. Tab inventory

### 2.1 Before → after

| # | Before | After | Role |
|---|---|---|---|
| 1 | (was tab 2) Modify | **Molbuilder** | Interactive workspace: build, generate, load, edit, assemble.  Only tab that holds in-memory canvas state.  Renamed from "Structure" → "Molbuilder" 2026-06-06 so the brand name marks the central tab. |
| 2 | (was tab 1) Build | **Structure optimization** | Form-driven task generator: SIESTA `.fdf` / PySCF `.py` from a project-saved structure. |
| 3 | (same) Spectra | **Spectrum calculation** | PySCF spectrum task generator (rename only — same function). |
| 4 | — | **Transport calculation** | NEW (Phase D).  Form skeleton for TranSIESTA + PySCF-NEGF scripts (engines not wired yet). |
| 5 | (same) Results | **Results** | Output viewer (no functional change). |

### 2.2 Functional migration (what moves)

**Out of Build → into Structure:**
- Source: from-file upload (XYZ / PDB)
- Generator: SMILES → 3-D (RDKit)
- Generator: 3DNA helix
- Generator: peptide builder (if present)
- Generator: any other "create a starting molecule" path in
  current Build

**Stays in Structure optimization (was Build):**
- SIESTA configuration form (engine, k-points, basis, mesh,
  pseudo lib, …)
- PySCF configuration form (engine, basis, xc, …)
- Generate `.fdf` + `.run.sh` + `.psml` install
- Generate `.py` + `.run.sh`
- Methods-text composition

**Stays in Structure (was Modify):**
- Sidebar file picker for loading existing structures
- Atom selection + viewer ↔ list cross-highlight
- Atom-level edit operations (delete, add, replace)
- Anchor-pair orient + rotate
- Per-atom info panel
- Region labels (electrode / bridge / anchor — for Transport)
- Electrode panel (close-packed slab attachment)
- Geom subtab (centre-at-origin, translate, slab mode)
- Save-to-project

### 2.3 What's intentionally NOT migrated yet

- **Atom builder UI** (manual atom-by-atom placement) — exists
  conceptually but has a different interaction model from the
  rest of the tab.  Could ship after the initial Molbuilder tab
  is stable.  Open question; see § 11.

---

## 3. Routes

### 3.1 Route table

| Tab | Route |
|---|---|
| Molbuilder | `/molbuilder` (bare `/` 302-redirects here via `landing_path()` in `molbuilder/web/tabs.py`) |
| Structure optimization | `/structure-optimization` |
| Spectrum calculation | `/spectrum-calculation` |
| Transport calculation | `/transport-calculation` |
| Results | `/results` |

The canonical tab list + landing path are derived from a single
constant (`TABS` in `molbuilder/web/tabs.py`); reordering tabs is a
one-place change.  The header partial iterates over `tabs` via a
context processor; the bare-`/` redirect calls `landing_path()`
which always equals `TABS[0]["path"]`.

### 3.2 No legacy aliases

Pre-1.0 cleanup: there are NO 301 redirects from old paths.  When
a route is renamed (e.g. `Structure → Molbuilder` 2026-06-06), the
old path stops working at that commit.  This keeps the route
surface a single source of truth — every reference in code, tests,
and docs uses the canonical name; no parallel "either path works"
branch exists.

### 3.3 Why long-form route names

Routes match the visible tab label exactly (`/structure-optimization`,
not `/optimize`).  Trade-off:

- **Pro**: URL is self-describing in browser history, shared
  links convey intent without context, no mental
  noun-to-route translation.
- **Con**: longer to type — acceptable since users navigate via
  tabs not the address bar.

The label-route match is also a maintenance discipline:
renaming a tab without renaming its route would silently
recreate the "Build / what does this do?" problem we're fixing.

---

## 4. Cross-tab workflow model

### 4.1 Two principles

1. **The Molbuilder tab is the only interactive workspace.**
   It holds an in-memory canvas.  Everything else reads from
   disk.
2. **Task tabs are file-driven.**  Structure optimization,
   Spectrum calculation, Transport calculation read their input
   structure from a sidebar-selected project file.  They do NOT
   consume Molbuilder tab's in-memory canvas state.

Together these decouple interactive editing from deterministic
script generation: the same project directory always produces
the same script regardless of which tab the user came from.

### 4.2 Canonical user flow

```
[Molbuilder tab]
   ↓ load file OR generate (SMILES/3DNA/...)
   ↓ edit / orient / label regions
   ↓ Save to project   ──────────► <proj>/<name>.xyz
                                          │
   [Structure optimization tab]           │
      sidebar pick ◄────────────── pick the saved file
      configure form
      Generate ────────────► <proj>/<name>.fdf + .run.sh
                                          │
   [Run on cluster, results land back]    │
                                          │
   [Results tab]                          │
      sidebar pick ◄────────────── pick <name>.out / .pyscf.log
      inspector renders
```

Every arrow except "Save to project" and the cluster round-trip
is a same-tab UI gesture.  Tab switches happen only when the
workflow phase changes (build → configure → review).

### 4.3 Why task tabs don't read in-memory canvas

A task tab consuming "whatever is in the Molbuilder tab right
now" would mean:

- The generated script depends on hidden state the user can't
  see in the project dir.
- A re-run of the same task tab AFTER a Structure-tab edit
  silently produces a different script.
- Two users on the same project see different scripts.
- Sharing a project dir (export / re-import) loses information.

Forcing task tabs to read from disk eliminates all four
classes.  The trade-off — one extra Save click between editing
and generating — is small and explicit.

---

## 5. Molbuilder tab — detailed design

### 5.1 Layout

Foldable panels, each with its own header + invoke button.  No
sub-tabs or wizard flow — every panel is reachable from one
screen, collapsed panels stay out of the way.

> **Shipped subset (2026-06-07):** the Sources card on
> `/molbuilder` carries panels 1, 2, and 11 below (Load from
> project + Generate from SMILES + Save).  Modifier panels 6-10
> still live as sub-tabs in the legacy Edit card; the
> generators 3DNA/peptide/name/file-upload (panels 3-5) and the
> Sources-card promotion of the modifier panels are roadmap
> items (B.3 + B.4).

Full panel order, top → bottom (target):

1. **Load from project** — sidebar candidate + **Load** button.
   Commits the picked file to the canvas (warning modal fires
   if dirty). ✅ shipped
2. **Generator: SMILES** — text input + **Generate**; RDKit
   backend; routes through `structurePage.loadIntoCanvas`. ✅ shipped
3. **Generator: 3DNA** — sequence + helix parameters.  Roadmap (B.3).
4. **Generator: peptide** — sequence + geometry options.  Roadmap (B.3).
5. **Generator: (others)** — name lookup, file upload.  Roadmap (B.3).
6. **Modifier: atom edit** — Delete + Add (today: Edit card "Atom"
   sub-tab).
7. **Modifier: orient + pose** — Orient + Rotate (today: "Pose").
8. **Modifier: regions** — region tagging (electrode/bridge/anchor)
   for the downstream Transport task.
9. **Modifier: electrode** — close-packed slab attachment.
10. **Modifier: geom** — centre-at-origin, translate, slab mode.
11. **Save** — Save-to-source.  Save-as + Discard still planned. ✅ partial

All panels are independent `<details>` elements; user state
(which are open, scroll position) is preserved across browser
refresh.

### 5.2 No auto-load on sidebar selection

Sidebar clicks do NOT replace the canvas.  Today's Modify
auto-loads on sidebar selection; the new Molbuilder tab keeps
that risky pattern at arm's length:

- Sidebar click → sets a "candidate" state (panel 1 shows the
  candidate path).
- User reviews + clicks **Load** to commit.
- If the canvas has unsaved modifications, the warning modal
  fires (see § 5.4).

Same explicit-button pattern applies to every Generate button.
The only path that overwrites the canvas without an explicit
button press is the initial mount (empty canvas).

### 5.3 Canvas state model

The canvas state is the source of truth for "what the viewer
shows."  Lives in JS memory; mirrored to `sessionStorage` for
refresh resilience.

**Schema** (subject to extension as features land):

```js
{
  // Core geometry
  source_format: "xyz" | "pdb" | "json",
  atoms:    [{element, x, y, z, ...}],     // per-atom records
  lattice:  { vectors: [[ax,ay,az],...] } | null,
  bonds:    [{i, j, order}] | null,         // optional explicit bonds

  // Sidecar / labels
  atom_labels:   { index: string | null }  | null,
  custom_labels: [{position, text, style}] | null,
  regions:       { electrode_L: [...indices], bridge: [...], ... }
                 | null,

  // Interactive state
  pickedIndices: [...indices],
  view_state:    { camera, style, axes, background, ... },

  // Provenance — answers "where did this canvas come from?"
  source: {
    kind: "file" | "smiles" | "threedna" | "peptide" | "blank",
    file: "/projects/<proj>/<name>.xyz" | null,
    generator_input: <kind-specific> | null,
  },

  // Save tracking
  dirty:        boolean,   // any modifier op since last save
  last_save_to: "/projects/<proj>/<name>.xyz" | null,
}
```

**Storage**:
- Live source of truth: JS module-scope state.
- Persistence mirror: `sessionStorage["molbuilder.structure_canvas"]`
  updated on every state change.
- Refresh: on mount, if sessionStorage has a canvas, restore it;
  else empty.
- Tab close: sessionStorage is cleared by the browser (default
  per-tab semantics).  User expected to Save before close;
  beforeunload warning fires if `dirty === true`.

**Size guard**: warn if serialized canvas size exceeds 4 MB
(out of ~5 MB sessionStorage limit).  Typical molbuilder
structures are well under; very large systems (50k+ atoms) may
hit the limit.

### 5.4 Warning modal — "modifications will be lost"

Fires when any of these would overwrite the canvas AND
`canvas.dirty === true`:

- Clicking **Load** on the load-from-project panel.
- Clicking **Generate** on any generator panel.
- `beforeunload` (browser tab close / refresh / navigate
  away).

Modal body:
- Title: "Unsaved modifications"
- Body: "You have unsaved changes to the current canvas.
  Continuing will discard them."
- Buttons: **Cancel** (default focus) and **Discard and continue**.

The modal blocks the triggering action; only **Discard and
continue** proceeds.

### 5.5 Provenance + status display

The Molbuilder tab header shows:

```
<canvas source>      [unsaved] / [saved to <path>]
```

`<canvas source>` is one of:
- "Loaded from `<proj>/<name>.xyz`"
- "Generated from SMILES: `<input>`"
- "Generated from 3DNA: `<sequence>`"
- "(empty)"
- etc.

The badge to the right of the source line tracks `dirty`.

### 5.6 Save-to-project flow

Save options:
- **Save** — overwrite `last_save_to` if set; else fall through to Save as.
- **Save as…** — sidebar-driven directory + filename input.
- **Discard** — clear the canvas back to empty (with warning if dirty).

Saves write the geometry (XYZ or PDB based on `source_format`)
+ the sidecar (`.molstruct.json` for labels + regions, if any).

On successful save:
- `canvas.dirty = false`
- `canvas.last_save_to = <path>`
- Sidebar refreshes to show the new file.

---

## 6. Task tab principle — file-driven

### 6.1 Behaviour shared across all three task tabs

- Sidebar pick → reads the file → renders the configuration
  form populated with whatever the file's `.molstruct.json`
  sidecar already declares.
- Form edits don't touch the Molbuilder tab's canvas.
- Generate writes the script next to the source file in the
  same project dir.
- No "in-memory state from Structure" plumbing.

### 6.2 What "from a project-saved structure" implies

The task tab needs to handle:
- A picked file with no sidecar (legacy file, generated outside
  molbuilder) — render the form with defaults, user fills.
- A picked file with sidecar (saved from Molbuilder tab) —
  pre-populate region labels, atom-level metadata.
- A picked file that's a directory (project root) — no-op, prompt
  the user to pick a file.

### 6.3 The form layer

Form schemas drive the UI (existing `form-schema.js` system).
Each task tab declares its config dataclass:
- Structure optimization: `SiestaConfig` and `PyscfConfig`
- Spectrum calculation: `SpectraConfig`
- Transport calculation: `TransportConfig` (already exists at
  `molbuilder/config/transport.py`)

The dataclass field metadata produces:
- The HTML form
- The validator pass
- The CLI flag set
- The Methods-text composition

This pattern is unchanged from today.

---

## 7. PySCF spectra output rename

### 7.1 The bug

PySCF spectra runs produce two artifacts:
- `<job>.spectra.json` — structured spectrum result (Results tab
  inspector handles correctly).
- `<job>.out` — the `.run.sh` wrapper's stdout+stderr capture.

The `.out` suffix collides with SIESTA's `.out` (a structured
text format SIESTA's Fortran binary writes).  The Results tab's
inspector dispatcher keys on the suffix and routes the PySCF
log into the SIESTA-trajectory inspector, which then renders
garbage or errors.

### 7.2 The fix

Rename: `.out` → `.pyscf.log`.

Rationale:
- `.log` is semantically accurate (the file IS a log of stdout
  + stderr, not a structured calculation output).
- The `.pyscf.` prefix pins provenance: any reader scanning a
  project dir knows which engine produced it.
- Distinct from `.out` → Results dispatcher routes to a new
  inspector dedicated to PySCF stdout.

### 7.3 Results-tab parser update

Phase C work:

- Add an inspector for `.pyscf.log` to the Results-tab
  inspector registry.
- Parser surfaces:
  - SCF convergence trace (iteration energies)
  - Final energy + dipole moment
  - Run-time + memory peak (if PySCF logged them)
  - Any traceback (script crash)
- Renders as a structured panel similar to the trajectory
  inspector but tuned for PySCF's log format.

### 7.4 Existing files on disk

Existing user files named `<job>.out` from prior PySCF runs are
NOT renamed.  Only new runs from Phase C onward use the new
suffix.

Rationale:
- Renaming on-disk files behind the user's back risks data loss.
- The dispatch misrouting happens silently today; an existing
  user with an existing project may not notice that the inspector
  was confused.
- A "files named `<job>.out` from a PySCF directory are legacy;
  click to rename" affordance can be added later if users hit
  the friction.

---

## 8. Transport calculation tab — Phase D form skeleton

### 8.1 What already exists

- `molbuilder/transport/engine_base.py` — TransportEngine
  Protocol + decorator-based registry.
- `molbuilder/transport/results.py` — TransportResults dataclass.
- `molbuilder/config/transport.py` — full TransportConfig
  dataclass with field metadata.
- 343 LOC of Python tests pinning the registry contract.
- Region-label infrastructure in current Modify tab
  (`selection-panel.js`, `modify/viewer.js`, `.molstruct.json`
  sidecar).

### 8.2 What does NOT exist

- Web blueprint (no `/api/transport/*` routes).
- HTML template for the tab.
- Static JS module for form interactions.
- Engine implementations (`transiesta_engine.py`,
  `pyscf_negf_engine.py` — Phase B.3, separate work item).
- Methods-text generation.
- Script renderer.

### 8.3 Phase D scope — form skeleton

For Phase D, ship:
- The route + HTML template (mirrors structure of
  `templates/build.html` / `templates/spectra.html`).
- The static JS module that drives the form (mirrors
  `static/lib/spectra/core.js`).
- Form rendering from `TransportConfig` field metadata (existing
  `form-schema.js` handles this generically).
- Save-script button is rendered but **disabled** with helper
  text: "No transport engines available yet.  Phase B.3 will
  add TranSIESTA + PySCF-NEGF backends."

User benefit:
- Sees the full parameter surface; gives feedback on what
  fields will exist.
- Tab takes its final navigation slot once — no second nav
  change when B.3 finishes.
- Region-label workflow (already in current Modify → moves to
  Molbuilder tab) gets surfaced as a prerequisite — pick a file
  with sidecar regions to enable the form fully.

### 8.4 Phase B.3 (separate work item, NOT in Phase D scope)

Engine implementations + script generators land later:
- `molbuilder/transport/transiesta_engine.py`
- `molbuilder/transport/pyscf_negf_engine.py`
- Per-engine Methods-text fragments
- Generate-script wiring

When Phase B.3 lands, the Phase D Generate-script button gets
enabled.

---

## 9. Phasing

### 9.1 Phase A — labels + routes (LANDED)

Cheap, safe, visible.  No functional change.

**Files touched:**
- `molbuilder/web/app.py` — register new routes; map them to
  existing view functions.
- `molbuilder/web/tabs.py` — single source of truth for tab
  order + landing path.
- `templates/_app_header.html` — iterates over the `tabs`
  context variable; no hard-coded anchors.

**Tests:**
- Existing test files referring to old paths updated to the new
  canonical paths.
- `tests/test_tab_routes.py` pins the 5-tab nav + tab→template
  + landing-path-follows-`TABS[0]` contracts.

**Decisions log entry (post-Phase A):**

The initial plan included 301 redirects from legacy paths so
existing bookmarks would survive.  The post-rename pass
(2026-06-06+, Structure → Molbuilder) deleted all redirects: the
canonical name is the single source of truth in code, tests, and
docs; renamed paths break by design.  See § 3.2.

### 9.2 Phase B — Molbuilder tab merger

**Files touched (major):**
- `molbuilder/web/static/modify/` — extend the existing modify
  page with the new panel infrastructure + generator panels
  migrated from Build.
- `templates/modify.html` (renamed to `templates/molbuilder.html`)
  — new panel layout.
- `static/modify/style.css` — foldable panel styles.
- New module `static/lib/structure/canvas-state.js` — the
  canvas-state primitive + sessionStorage mirror + dirty flag.
  (`structure/` here is the JS namespace for canvas data, not
  the user-facing tab name; the tab is the Molbuilder tab.)
  *Phase 9 (2026-06-13) note*: this file moved to
  `static/lib/workspace/_canvas-state-impl.js` and stopped
  mounting `window.molbuilder.structureCanvas`; the dispatcher
  owns the singleton.  Historical name preserved here for
  context with the original design proposal.
- New module `static/lib/structure/warning-modal.js` — the
  "unsaved modifications" modal.
- `static/lib/structure/generators/` — one module per
  migrated generator (smiles.js, threedna.js, …).
- `molbuilder/web/blueprints/build.py` — Build-tab generator
  endpoints move to a new `molbuilder.py` blueprint (or stay
  but with the route prefix updated to `/api/molbuilder/*`).

**Files touched (minor):**
- `templates/build.html` (now `structure-optimization.html`)
  — strip the generator UI; keep only the SIESTA / PySCF
  form section.
- `molbuilder/web/blueprints/build.py` — keep the form + script
  endpoints; route prefix changes.

**Tests:**
- Existing modify e2e tests get path-parameterized for the
  new URL.
- New e2e tests:
  - sessionStorage round-trip across page reload.
  - dirty-flag warning fires on Load with unsaved changes.
  - dirty-flag warning fires on Generate with unsaved changes.
  - dirty-flag warning fires on beforeunload.
  - Save clears dirty flag.
  - Each generator panel produces a canvas with correct
    provenance + dirty=true.

**Acceptance:**
- Every Build-tab generator works inside the Molbuilder tab.
- Modify-tab editing works on a structure that came from a
  generator (no save round-trip).
- Canvas state survives browser refresh.
- Unsaved modifications never silently lost.

### 9.3 Phase C — PySCF `.out` → `.pyscf.log` + Results parser

**Files touched:**
- `molbuilder/spectra/` PySCF backend — change the script's
  output redirect from `> <job>.out` to `> <job>.pyscf.log`.
- `static/lib/inspectors/registry.js` — register an inspector
  for `.pyscf.log`.
- New module `static/lib/inspectors/pyscf-log.js` — parses
  PySCF stdout + renders the result panel.
- Per-spectra-tab Methods-text + script header text — mention
  the new filename.

**Tests:**
- Server: spectra-blueprint emits `.pyscf.log` in the script.
- E2E: opening a `.pyscf.log` in `/results` dispatches to the
  new inspector + renders.
- Negative: opening a `.out` (SIESTA) still dispatches to the
  SIESTA-trajectory inspector unchanged.

**Acceptance:**
- New PySCF spectra runs produce `.pyscf.log`.
- Results tab inspector for PySCF log renders cleanly.
- SIESTA inspector unaffected.

### 9.4 Phase D — Transport tab form skeleton

**Files touched:**
- New route `/transport-calculation` + view function.
- New template `templates/transport-calculation.html`.
- New blueprint `molbuilder/web/blueprints/transport.py` —
  schema endpoint + (disabled) generate endpoint.
- New static module `static/lib/transport/core.js` — form
  driver.
- Static CSS for the tab.

**Tests:**
- Server: schema endpoint returns the TransportConfig fields.
- E2E: tab renders, all form fields appear, Generate button
  is disabled with the expected helper text.

**Acceptance:**
- Tab navigable, form fully rendered.
- Generate button disabled with helpful message about Phase
  B.3.
- TransportConfig field changes round-trip through the form.

### 9.5 Phase ordering rationale

A first — smallest PR, mostly mechanical, immediately visible
in the nav.  Lets the rest of the work proceed against the
renamed surface without churn.

B before C — Molbuilder tab merger is the biggest piece;
shipping it second means the canvas-state primitive exists
before the Results tab inspector work in C.

C before D — `.pyscf.log` rename is small and orthogonal;
shipping it ahead of D means D can pattern-match against the
finalized Results-tab inspector convention.

D last — depends on patterns from B (form-schema-driven tabs)
+ C (inspector registration).

---

## 10. Contract surfaces touched

| Surface | Where it's documented | What changes |
|---|---|---|
| Form schema | `form-schema.js` (no doc; pattern) | New TransportConfig form (Phase D) |
| Projects sidebar | `protocols/projects-sidebar.md` | No contract change.  Selection still drives file-picker; the difference is callers (task tabs require explicit confirmation; Molbuilder tab requires an explicit Load button). |
| Embedded viewer | `protocols/molview-module.md` | No contract change.  Molbuilder tab uses the embed with the standard handle. |
| Inspector registry | `protocols/inspector-registry.md` | New inspector for `.pyscf.log` (Phase C). |
| Web API | `protocols/web-api.md` | New `/api/transport/schema` endpoint (Phase D); existing build-API generator endpoints either move to `/api/molbuilder/*` or get aliased (Phase B). |
| Tabs index | `design.md` § 0 (UI tabs) | New per-tab specs land here as phases complete. |

---

## 11. Open questions

These are NOT blockers for Phase A — they only need to be
answered before the Phase they affect ships.

1. **Atom builder UI.**  Should the manual atom-by-atom placement
   path (was in Build?) migrate?  If yes, into which Structure
   panel?  (Affects Phase B.)
2. **`.pyscf.log` for existing files.**  Do we offer a one-shot
   migration tool (a button on Results that renames legacy
   `<job>.out` to `<job>.pyscf.log`) or accept manual cleanup?
   (Affects Phase C; can land later.)
3. **Junction assembly workflow.**  Today's Modify already
   supports placing components.  Do we need a dedicated junction
   assembly mode (pre-labels electrodes + bridge), or do users
   build junctions by loading components into the canvas one at
   a time via the migrated generators + atom-edit ops?
   (Affects Phase B + Phase D — answered: use generic load + atom-edit per
   the 2026-06-06 conversation.)
4. **Multi-component canvas.**  Can the canvas hold multiple
   independent structures at once (electrode + bridge as separate
   geometries with combined region labels), or does loading a
   second structure replace the first?  Today's Modify replaces.
   (Affects Phase B; current plan: replace, then loosen later if
   the junction workflow demands it.)
5. **Form pre-fill from sidecar.**  When a task tab loads a file
   that has a `.molstruct.json` sidecar with engine hints (e.g.
   the user pre-tagged "electrode" regions in Molbuilder tab),
   should the Transport task form auto-tag the region fields?
   (Affects Phase D; can ship without; nice-to-have.)

---

## 12. Maintenance protocol for this doc

1. Every phase PR updates this doc as part of the same commit:
   - The status line in § 1 ("planning phase" → "phase A complete"
     etc.)
   - The "files touched" list in § 9 — replace planned with
     actual; check off acceptance criteria.
2. New decisions that affect multiple tabs land in § 11 first as
   open questions, then in the relevant phase section once
   answered.
3. Per-tab specs (`docs/tabs/molbuilder.md`,
   `docs/tabs/structure-optimization.md`, etc.) are created when
   their respective Phase completes; they own the *internal*
   contract; this doc keeps only cross-tab content.
4. `design.md` § 0 (UI tabs index) gets updated when a per-tab
   spec is created or renamed.
5. The "Before → after" table in § 2.1 is frozen once Phase A
   ships — historical reference.
