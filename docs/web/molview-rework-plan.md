# MolView — bringing the code to the document

**Role:** plan
**Domain:** web
**Started:** 2026-07-30
**Companions:** [`molview.md`](?doc=web/molview.md) — the contract this closes the distance
to. Retired at Phase 8, when the code has caught up and the contract stands alone.

[`molview.md`](?doc=web/molview.md) is the contract. Where it and the code disagree, the
code is wrong.

Today: **28 files, 16,910 lines, two directories** — `lib/viewer/mol-viewer-embed.js` alone
is 6,531 of them, 44% of the module, and it is not what § 9.9 calls the sealed layer. Plus
~12,800 lines of tests that pin the surface being deleted.

Finished: **one sealed directory, one import, 20 files, tests derived from the document.**

---

## 1. How this runs

**One unit at a time, from the bottom.** Rebuild it to what the document says it is, write
its tests **from `molview.md`**, run **only those**, move up.

**The suite is never run whole until the module is finished.** There is no green baseline
to protect — today's suite pins what is being deleted.

**We break a large part of the package on purpose.** Every consumer that reaches inside,
and every test written against the old shape, falls over. **Those failures are the work
showing up, not bugs.** They get written down, not chased. Phase 7 works the list off.

**Each old test dies in the same commit that rebuilds the unit it pins.**

Tests come at § 13.2's three levels — behaviour in node, boundary behaviour with stand-ins,
and § 1.1 end-to-end on the demo page, which stays live throughout as MolView's own
consumer. Two rules from § 13.1: **a stand-in obeys the document, not the code**, and **no
pinned name lists** — the contract is what a surface must do and must refuse.

**Starting tree.** The prior session's uncommitted work is kept: the
`engine/`→`render-engine/` rename, `embed-io.currentFrame()` deleted, the frame notifier
consolidated into the model, and `_movieExists()` — which is § 10.10 and closes task **#35**.
I repointed `index.js:53-55` at the renamed directory; the entry module did not resolve
without it. Nothing is committed.

---

## 2. The target tree

**28 files → 20.** One directory, one import.

```
lib/molview/
├── index.js              the entry: mount + formula, nothing else        § 4, § 9.1
├── _formula.js           Hill formula — needs no viewer                  § 4
├── _atom.js              the numbering translation + the filter channels § 11.5, § 9.5
│
├── mount.js              the card, the handle, playback, the badge       § 8, § 9.2
├── demo.js               the in-repo demo page                          § 13.4
│
├── data-model.js         THE MASTER COPY + the data API                 § 6, § 9.3
├── _install.js           helper — put a loaded structure in             § 7.3
├── _serialise.js         helper — write the structure out               § 7.3
├── _history.js           helper — undo / redo + the write machine       § 7.3, § 11.2
├── _operations.js        helper — the geometry edits                    § 7.3, § 11.1
├── _selection-store.js   what is selected + the switches                § 9.5
├── _view-store.js        style · radius · background · projection       § 9.6
│
├── selection.js          the panel: its DOM, its rows, click-to-select  § 9.5
├── measurement.js        the readout + the distance / angle maths       § 11.6
├── controls.js           the six switches · the View menu · the frame bar § 1.1, § 6.4
├── menu.js               MolView's own menu surface — Export            § 11.4
│
├── render-engine/
│   ├── engine.js         what to redraw, at what cost + the per-frame maths § 9.7, § 10
│   └── embed-io.js       the drawing commands                           § 9.8
│
├── _seal.js              the ONE file that names 3Dmol                  § 4, § 9.9
└── molview.css           one stylesheet, one link
```

**Why each group is one file:**

- **`_seal.js`** — § 4 is explicit: *"the name `3Dmol` occurs in exactly one file."* Today
  it occurs in four (`mol-viewer.js`, `mol-style.js`, `mol-axes.js`, `mol-viewer-embed.js`).
  After this it is checkable by grep.
- **`engine.js`** — § 9.7 splits the renderEngine in two: a maths half *"with no drawing
  library anywhere near it"* and an I/O half. The I/O half is `embed-io.js`; deciding the
  cost and deriving each frame is the maths half, so `process.js` folds in. Still
  node-testable: `embed-io` takes an injected handle and imports nothing.
- **the model's five helpers stay five** — § 7.3 *requires* the split and names exactly
  these five jobs.
- **two stores, not one** — § 9.6 spends a section on why the switches and the drawing
  settings are different kinds of thing.
- **`selection.js`** — panel strip, its DOM and click-to-select are one concern. The markup
  only lived in a Flask template because the panel used to be a server partial.
- **`measurement.js`** — § 11.6: *"measurement is its own layer."*
- **`controls.js`** — every control MolView draws around the canvas, mounted together,
  writing into the stores like any other caller.
- **`menu.js` stays apart** — § 11.4's test is *what does the control decide?* Export
  decides what leaves the viewer and from which copy; a style knob decides nothing.
  Collapsing that distinction is how the export decision reached the bottom of the stack.

`-impl` suffixes exist only to distinguish an implementation from the global shim in front
of it, so they go with the globals. `mol-` prefixes were a `lib/viewer/`-era namespace.

---

## 3. The phases

Each: **the files, the gap it closes, what it lands, what dies with it.**

### Phase 1 — One directory · *mechanical, no behaviour change*

The tree of § 2 comes into existence first, so every later change happens in its final home
instead of being moved afterwards.

| Move | Result |
|---|---|
| `lib/viewer/{mol-viewer-embed, mol-viewer, mol-style, mol-axes}.js` **copied** in, four files into one | `_seal.js` |
| `lib/viewer/mol-format.js` copied in | `_formula.js` |
| `_atom-index.js` + `_atom-channels.js` | `_atom.js` |
| `render-engine/process.js` | into `render-engine/engine.js` |
| `selection/panel.js` + `selection/viewer-adapter.js` | `selection.js` |
| `selection/measurements.js` + `measurement-overlay.js` | `measurement.js` |
| `frame-controls.js` | `controls.js` |
| `_viewer-overlay.js` + `selection/mount-panel.js` | into `mount.js` |
| `_state-timeline-impl.js` · `_selection-store-impl.js` | `_history.js` · `_selection-store.js` |
| `static/molview/demo.js` | `demo.js` |
| four stylesheets | `molview.css` |

**Copied, not moved.** `lib/viewer/` stays on disk and untouched, so VibrationView keeps
borrowing `window.molbuilder.viewer` exactly as it does today — MolView simply stops
importing it. Its JS is then dead except for the CSS link five templates still carry.
VibrationView taking its own copy is its own module's work, and it happens at Phase 7.

**Nothing changes behaviour.** No carve, no rewrite, no rule from the document applied yet —
imports repointed, files concatenated, names settled. The transitional globals keep
publishing under their current names.

**Verified by** the module loading (node ESM load-sim) and the demo page in a real browser.
Not by the suite. This is the one step where a blind sweep would be fatal, so it goes file
by file from the diff — [the namespace-rename lesson].

### Phase 2 — The pure bottom

**Files** `_atom.js` · the per-frame maths inside `render-engine/engine.js` · the § 6.2
shapes become the module's vocabulary.

**The gap** No dependencies at all, so everything above is written against these. Step 1
cuts down and records the drawn→original map; step 2 hangs the three overlays off what
survived — no colour, radius or opacity in the output, no cell, no axes.

**Lands** one translation, one place · the data holds what the filter enumerates · the two
steps in that order · a label carries the original number · frame *f*'s arrows come from
frame *f* · the highlight is content, not styling.

**Dies** `test_atom_index_js.py` · `test_atom_channels_js.py` ·
`test_render_engine_process_js.py`.

### Phase 3 — The truth moves into the model ★

**Files** `data-model.js` · `render-engine/engine.js`.

**The gap** The master copy is a layer too low, and the code says so:

```
render-engine/engine.js:54    var _data  = null;   // clean StructureData -- the source of truth we own.
render-engine/engine.js:56    var _frame = 0;      // current frame
data-model.js:158             // … is a thin coordinator over the engine
data-model.js:1437,1440,1443  getFrameAllAtoms / currentFrame / frameCount → ask the renderEngine
```

§ 7 level 5 holds nothing of its own, and § 6.4's ordering rule has nothing to stand on
while the range is read back out of the renderer. The model takes `frames`,
`forcesPerFrame`, `elements`, `annotations`, `cell` **and the displayed frame with its
range**; the engine is handed data. Master copy updated → range recomputed from it → frame
checked → one notification.

**Lands** nothing keeps its own copy · the ordering rule · an out-of-range write is
resolved, not accepted · same atoms every frame · only the master copy's count is offered.

**Dies** `test_render_engine_orchestrator_js.py` · the frame half of
`test_workspace_dispatcher_js.py`.

### Phase 4 — What a viewer holds

**Files** `_selection-store.js` · **new `_view-store.js`** · `_canvas-state-impl.js`
**deleted**.

**The gap** Three at once. Viewers are not owned — `molbuilder.molview.data` is one
module-level object and the stores are process-wide singletons (§ 5.6). `view` is not a
store but a **read-back passthrough**: `data-model.js:1077` synthesises style, axes, labels
and **the camera** by calling down into the embed, and `flushViewState` (`:1107`) writes
that into the saved session on page-hide — § 9.6 says nothing is read back, § 9.9 says the
camera cannot be asked for. And `_canvas-state-impl.js` (442 lines) is a **third home for
the structure**, holding `text` / `source_format` / `dirty` / `last_save_to` process-wide
against § 6.3's two copies.

Canvas-state splits three ways: `text` / `source_format` → the model, per owner; the
`sessionStorage` mirror → the workspace module; `source` / `dirty` / `last_save_to` →
whoever loads and saves files, which is not the viewer.

**Lands** the selection survives an editor switch · a half-typed row constrains nothing ·
by-atom-index crosses the numbering boundary once · one selection per owner · the camera is
not kept, saved or read back · a drawing setting derives nothing.

**Dies** `test_selection_store_js.py` (981) · `test_structure_canvas_state_js.py` (504).

### Phase 5 — The model's doors and the one gate

**Files** `data-model.js` · `_install.js` · `_serialise.js` · `_operations.js` ·
`_history.js`.

**The gap** There is no read-only gate. `mode: "readonly"` reaches two places — hidden panel
UI and a separate "ephemeral" store — and `data-model.js` never consults the mode, so
**every write door is open in a read-only viewer**, the label door included (§ 9.4 names it
as the one easy to get backwards).

Also here: the fourteen needs, one main way in each; `getSource` goes; reads return copies;
`commitPeriodicityOp` becomes the only cell write; `applyOp` is driven by § 11.1's
eight-row table; § 11.2 whole — point 0, a Retract spending unsaved work first, the badge,
SETTLED / CHANGING / WRITING; the serialiser starts carrying every frame.
**Audit first:** the timeline against § 11.2 — the discrepancies need re-deriving from the
code before they are fixed.

**Lands** ~20 of § 13.3's rows, the largest block in the plan.

**Dies** `test_workspace_dispatcher_js.py` (1,598) ·
`test_workspace_dispatcher_canvas_mount_js.py` (191).

### Phase 6 — The seal and the chrome

One phase, because they are the same operation from two ends: what comes out of `_seal.js`
is what `mount.js`, `controls.js` and `menu.js` are built from.

**Files** `_seal.js` carved · `mount.js` · `controls.js` · **new `menu.js`** ·
`selection.js` (taking its markup in from `templates/_selection_panel.html`) ·
`render-engine/{engine,embed-io}.js`.

**The gap** The embed is a whole viewer. By its own section banners: card scaffold · knob
bar · structure load · style · cell wireframe · atom labels · arrows · atom-pick · overlay
rendering · info line · **its own animation interval** · frame strip · a 2,200-line handle
builder · a test-affordance surface. Two pieces break stated rules outright: `:5007`
reaches `molbuilder.projects` and posts to `/api/files/write` (§ 6.7 — no file route), and
its Data export writes coordinates only, no `.json`, so labels reach script generation
silently gone. Both are task **#39**, which closes here.

| Out of the seal | Into |
|---|---|
| card scaffold · info line | `mount.js` |
| knob bar (six switches, View menu, Reset) | `controls.js` → the two stores |
| frame strip · animation controls · the animation interval | `controls.js` + mount's one timer |
| export menu · snapshot · GIF encoder | `menu.js`; the rendering is delegated back down as a command (§ 11.4), the decisions are not |
| `molbuilder.projects` + `/api/files/*` · the test-affordance surface | **deleted** |

Drawing, camera, styles, picking and the highlight stay. The panel's DOM stops being a
Flask partial. The handle shrinks to lifecycle, playback and `data`.
**Audit first:** the four cost tiers (§ 10.5) against `_structSig` / `_prevArrowSig`, the
rebuild queue (`_locked`, `_pendingTx`) against § 10.9's five arrival rules, and whether
`embed-io.js` is already decision-free.

**Lands** the renderEngine answers nothing · the commands answer nothing upward · the
sealed layer faces downward only · load once, play by swap · the cost matches what changed
and never consults the atom count · shapes move with the frames · a selection never
restyles · nothing is lost during a rebuild · the offered frames are drawable · no file
route · § 1.1 end to end on the demo page · mount always resolves · the handle refuses
appearance · every export enters at MolView · only Data is the truth, at the chosen frame ·
a saved structure keeps its metadata · measurement reads the truth in pick order.

**Dies** `test_mol_viewer_embed_e2e.py` (5,037) · `test_mol_viewer_embed_js.py` (213) ·
`test_mol_viewer_embed_handle_surface_js.py` (401) · `test_no_3dmol_data_reads.py` (70).

### Phase 7 — Seal the entry, then reconnect

**Files** `index.js` → two exports · every `window.molbuilder.*` publish deleted.

**The gap** `index.js` re-exports **everything** (15 `export * from` lines), ~20 globals are
published, and `runtime.register` is called from four files — so nothing about § 4 is true
yet.

Outside MolView, and only now: `lib/vibrationview/` takes its own copy of the embed and
`lib/viewer/` is deleted · `lib/workspace/{dispatcher,snapshot-io}.js` absorb canvas-state's
session mirror · `molbuilder-runtime.js` drops the `viewer` / `style` / `fmt` entries · ten
consumer files repoint to the one import (`modify/viewer.js`, `modify/periodicity.js`,
`modify/selection-bootstrap.js`, `modify/structure/{file,page,save}.js`,
`spectra/viewer.js`, `structure-optimization/viewer.js`, `lib/trajectory/core.js`,
`lib/transport/core.js`, `lib/inspectors/structure.js`, `lib/projects/parser.js`) · six
templates drop four CSS links for one and lose `_selection_panel.html`.

File by file from the diff, never a blind sweep, verified in a real browser. The breakage
list from Phases 1–6 is worked off here.

**Lands** the module is self-contained · a viewer is owned.

### Phase 8 — The suite, and closeout

The first whole-suite run of the program: the tests **outside** MolView that the rework
knocked over. Then the residue — every **Transition** note in `molview.md` whose code has
caught up deleted (the document should end with none); § 11.1's route sentence, which says
three routes and omits the filter's two (`/api/selection/eval`, `/api/selection/atoms`);
§ 15's file map rewritten to § 2's tree; `science/validation.md` § 4.1 if `factsForRequest`
retired. Tasks: **#35** closed by the starting tree, **#39** by Phase 6, **#104** superseded
by Phase 1's copy, **#22**'s MolView rows by Phase 7.

---

## 4. The breakage log

What the rework knocked over, recorded rather than chased (§ 1). Phase 7 works the consumer
half off; each test dies with the unit it pins.

### Phase 1 — one directory *(landed 2026-07-30)*

**One real bug, found and fixed.** The prior session's `engine/` → `render-engine/` rename
changed what the module *publishes* (`molview.engine` → `molview.renderEngine`) but not
`mount.js:251`, which still looked up `mvApi.engine.create`. The guard around it turns a
missing factory into a no-op, so **every viewer mounted and then never drew** — no atoms,
`frameCount()` stuck at 0, no frame bar. Invisible to node tests, which stub the engine;
caught by the demo page in a real browser, which is the whole reason that check is in the
phase. This is the failure mode a namespace rename always has, and the reason a runtime
global lookup is worse than an import.

**20 test files reference paths this phase moved.** None repointed. MolView's own —
`test_atom_index_js` · `test_atom_channels_js` · `test_render_engine_process_js` ·
`test_render_engine_orchestrator_js` · `test_selection_measurements_js` ·
`test_selection_mount_panel_js` · `test_selection_store_js` · `test_measurement_overlay_js`
· `test_mol_viewer_embed_js` · `test_mol_viewer_embed_handle_surface_js` ·
`test_workspace_dispatcher_js` · `test_workspace_dispatcher_canvas_mount_js` — die with
their units in Phases 2–7. Repo-wide guards that merely name a moved path —
`test_xss_audit` (its allowlist points at `selection/mount-panel.js`, whose `innerHTML` is
now in `mount.js`) · `test_css_no_hex_literals` · `test_engine_atom_index` ·
`test_atom_list_render_paths` · `test_live_poll_invariants_audit` ·
`test_no_legacy_store_consumers` · `test_results_blueprint` ·
`test_ui_presence_data_independent_js` — these guard invariants worth keeping, so they get
repointed at Phase 7 rather than deleted.

**`lib/viewer/` is now dead but still on disk**, as planned: nothing imports its JS and
nothing links its CSS. It stays until VibrationView takes its own copy (Phase 7).

### Phase 2 — the pure bottom *(landed 2026-07-30)*

**One gap closed:** every force arrow carried a `color` and a `radius` — appearance computed
by the maths and re-sent on every frame of every trajectory, against § 6.5. It now lives
beside `_SELECTION_GLOW` in the sealed layer as `_FORCE_ARROW`, and the maths emits
`{start, end}`. The picture is unchanged provably, not by inspection: the ramp reads
`t = |arrow| / |largest arrow|`, and every arrow in a set carries the same scale, so the
ratio is the one the maths used.

**Two channels the contract does not name, recorded rather than changed.** `frozen` is still
a field rather than an ordinary label — § 6.6 parks that fold in
[`model/structure-annotations.md`](?doc=model/structure-annotations.md) because it changes
the sidecar format and the input generators. `values` (per-atom scalars) is task #24's
programme. Both come from a field the atom really carries, which is the rule § 6.2 states.

**17 tests replace 3 files.** The one worth naming is the numbering drift test: asserting
`label == index + 1` cannot catch a hand-rolled `+1`, because it agrees with the translation
by arithmetic coincidence. The test moves the shared translation and asserts the labels moved
with it — reuse follows, re-derivation does not.

---

## 5. Open

1. **Canvas-state's three-way split** (Phase 4) reaches `workspace/dispatcher.js` and
   `snapshot-io.js` — not MolView's call alone.
2. **`factsForRequest`** — retire in Phase 5, or leave until validation is next opened? It
   edits `science/validation.md` § 4.1 and the test that guards it.
3. **§ 11.1's route sentence** — one line, your wording.
4. **The in-flight set** — commit as it stands, or fold into Phase 1's commit?
