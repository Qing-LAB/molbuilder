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

**Nothing outside this module is consulted, accommodated or repaired.** MolView is sealed
(§ 4): it is imported by name, it reaches nothing else by name, and **it publishes nothing**.
How anything else uses any code is not this plan's business and not a constraint on it. If
sealing or rebuilding leaves something outside broken, it stays broken.

**The tests are designed new from the contract.** § 13.3's rows are the plan; § 13.2's three
levels are its shape. Files that exist today are not consulted while writing them, not
adapted, and not repointed — § 13.1 rules out most of what they are. A rule with no row in
§ 13.3 is a rule nothing guards; that is the only checklist.

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

## 2. The target tree — one file per layer, one per API

Organised by § 7's levels and § 9's surfaces, because those are what is cleanly designed.
**19 files → 10 and a stylesheet.**

| § | File | The API it is |
|---|---|---|
| 9.1 | `index.js` | `mount` + `formula`, and nothing else is importable |
| 8 · 9.2 | `mount.js` | assembles the viewer; the handle — lifecycle, playback, `data` |
| 9.3 | `model.js` | the master copy, the data API, the read-only gate |
| 7.3 | `model-jobs.js` | the jobs the model hands out: load a structure in, write it out, the geometry edits |
| 11.2 | `history.js` | the ordered sequence and the position on it — `save`/`load`/`undo`, `state_index`, `uncommitted`, the SETTLED/CHANGING/WRITING machine, the badge signal |
| 9.5 · 9.6 | `stores.js` | `selection` (what is picked, the switches) and `view` (how it is drawn) |
| 9.7 · 9.8 | `render-engine.js` | what to redraw and at what cost · the per-frame maths · the drawing commands |
| 9.9 | `seal.js` | the only file that names 3Dmol |
| 1.1 · 11.4 · 11.6 | `ui.js` | everything MolView draws: the panel, click-to-select, the frame bar, the switches, the View menu, the Export menu, the readout, the badge |
| 13.4 | `demo.js` | the in-repo demo page |
| — | `molview.css` | one stylesheet, one link |

**Where today's files go.** `_formula.js` → `index.js` · `_atom.js` dissolves into the
surfaces that use it (§ 11.5's rule is ONE home, and the model is a home; § 9.5's channels are
the store's) · `data-model.js` → `model.js` · `_canvas-state-impl.js` **deleted**, a third home
for the structure (§ 6.3 allows two) · `_install.js` + `_serialise.js` + `_operations.js` →
`model-jobs.js` · `_history.js` → `history.js` · `_selection-store.js` → `stores.js`, which
gains the `view` store that replaces the read-back passthrough · `engine.js` + `embed-io.js` →
`render-engine.js` · `_seal.js` → `seal.js` · `selection.js` + `measurement.js` +
`controls.js` → `ui.js`.

**`history.js` stays out of `model-jobs.js` on purpose.** § 11.2's own claim — *"the mechanism
does not know or care what is in it … nothing about saving constrains what may be saved"* —
only holds while it is not sitting in the same file as the serialiser it is handed.

**Every file must serve MolView's logical design and nothing else.** A file that exists to
publish an interface is not a file; it is a leak with a name.

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

**Copied, not moved.** MolView takes its own copy and stops importing anything outside
`lib/molview/`. What happens to the original is not this plan's concern.

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
`selection.js` (its markup moves in-module) ·
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

Drawing, camera, styles, picking and the highlight stay. The panel builds its own DOM. The handle shrinks to lifecycle, playback and `data`.
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

### Phase 7 — Seal it

**Files** `index.js` → two exports · every `window.molbuilder.*` publish deleted · every
`runtime.register` deleted.

**The gap** `index.js` re-exports **everything** (15 `export * from` lines), and the module
publishes **44 names** on `window.molbuilder.*` plus four registry registrations — including
`molbuilder.viewer` and seventeen members hung off it, which is the sealed layer itself, the
one thing § 4 says no consumer may ever name. None of it is in the contract: *"That is the
whole surface … every other file in the module is internal — a consumer that imports any of
them directly has broken the module, not found a shortcut."*

The part that is not deletion: MolView's own internals currently find each other through
those same globals **at runtime** — `mount.js` reads `mb.molview.data`, `.selection` and
`.renderEngine`; the panel composition reads `selection.panel`; the click wiring reaches
`molview.data.selection`; the model reads `root.molbuilder.workspace` instead of the one
handed in at mount (§ 8). Those become imports and injection, which is what makes the module
sealed rather than merely quiet.

**Lands** the module is self-contained — nothing outside is importable but the entry point,
and it mounts given only a host element and something that satisfies the workspace door · a
viewer is owned.

### Phase 8 — Closeout

The § 13.3 plan is complete, so the suite runs whole for the first time — MolView's own.
What remains is the document: every **Transition** note in
[`molview.md`](?doc=web/molview.md) whose code has caught up is deleted (it should end with
none); § 11.1's route sentence, which says three routes and omits the filter's two
(`/api/selection/eval`, `/api/selection/atoms`); § 15's file map rewritten to § 2's tree. Task
**#39** closes at Phase 6.

---

## 4. What each landed phase found

Kept because a defect is worth remembering, not because anything outside is owed a list.

### Phase 1 — one directory *(landed 2026-07-30)*

**One real bug.** The `engine/` → `render-engine/` rename changed what the module *publishes*
but not `mount.js:251`, which still looked up `mvApi.engine.create`. The guard around it turns
a missing factory into a no-op, so **every viewer mounted and then never drew** — no atoms,
`frameCount()` stuck at 0, no frame bar. Invisible to node tests, which stub the engine;
caught by the demo page in a real browser, which is why that check is in the phase. It is the
failure mode a namespace rename always has, and the reason a runtime global lookup is worse
than an import.

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

### Review of Phases 0–3 *(2026-07-30)*

Read back against the document. **Three defects, one of them mine and serious.**

1. **A held cell edit or force update was dropped on the floor** — `engine.js`'s replay
   dispatched on `setForces` / `setCell`, the names Phase 3 renamed away, while the queue
   pushed `forcesChanged` / `cellChanged`. So an op arriving during a rebuild was held exactly
   as § 10.9 requires and then matched nothing on the way out. § 10.9's whole sentence is
   *"nothing that lands in that window is silently dropped"*. No test caught it: node tests
   stub the engine, and the browser oracle has no rebuild-with-pending-forces case. The replay
   now throws on an unknown op rather than falling through. **Its § 13.3 guard is Phase 5's
   row** (*nothing is lost during a rebuild*) and is owed there.
2. **Forces arriving on an append were dropped when the load carried none** — the list was
   extended only when it already existed, so a run caught at its first geometry (§ 12.2's
   worked example, and the case Phase 0's `_movieExists()` fix exists for) never grew arrows.
   Pre-existing; Phase 3 carried it faithfully into the model, which is where it is now fixed:
   the list starts existing the moment forces first arrive, back-filled with `null`.
3. **`appendFrames` took an options bag it no longer read** — residue from the truth moving.

**Not defects, recorded:** `data-model.js` names `3Dmol` in seven comments, most of them
saying it does *not* touch it. § 4's "occurs in exactly one file" is about code, and a test
written from it needs to say so. The one that is a real pointer — *"View sub-namespace:
passthrough to the 3Dmol embed"* — documents the § 9.6 read-back door, which is Phase 4's.

**The residue that matters most:** `test_mol_viewer_embed_js.py` **passes 12/12 against
`lib/viewer/mol-viewer-embed.js`, which nothing imports any more.** A green test guarding a
file that is not shipped is worse than a red one, because it reads as coverage. Same for
`test_mol_viewer_embed_handle_surface_js.py`. Both die at Phase 5 with the copy they pin; until
then they are green and mean nothing.

---

## 5. Open

1. **Canvas-state** (Phase 4) is a third home for the structure (§ 6.3 allows two). The
   structure fields belong in the model, per owner; the rest is not MolView's to hold.
2. **`factsForRequest`** — retire in Phase 5, or leave until validation is next opened? It
   edits `science/validation.md` § 4.1 and the test that guards it.
3. **§ 11.1's route sentence** — one line, your wording.
