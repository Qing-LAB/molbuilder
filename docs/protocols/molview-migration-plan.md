# MolView consumer migration — OPEN-WORK tracker

**This is a working plan, not a contract. It stays until the task is finished.**
It tracks only the **still-open** MolView consumer-migration work. Every design
that has SHIPPED is now core behaviour and lives in the core contract — this
tracker points there rather than re-describing it.

> **Scope note (2026-07).** This tracker covers the **Modify** consumer + the core
> model/persistence work. The *whole-frontend* target — every tab migrating onto the
> concealed MolView + VibrationView modules, the ES-module conversion, and the per-tab
> order — now lives in
> [`frontend-module-architecture.md`](frontend-module-architecture.md). Start there for
> the big picture; this file is the granular Modify/core checklist feeding into it.
>
> **Status sweep (2026-07-16):** Track B (Modify-consumes-MolView, B0–B3) ☑; the
> 2026-07-06 review follow-ups **F1 / F2 / F4** ☑ (shipped — `test_sidecar_annotations.py`
> + tasks); **Step 5** cell-origin ☑ (`cell_origin` + `calibrate`, commit `3d3f308`).
> Genuinely-open remainder: **D3-tail, D4, A3, A4, Step 6** (below).

**The design lives in the core** —
[`workspace-contract.md`](workspace-contract.md) (the MolView + Workspace core
contract):

- **The accessor API / concealed model** (Track D design): core **§1.2.1** (the
  mandatory one-model + accessor-API contract) + **§1.4** (encapsulation) +
  **§2/§3** (the read/write accessors).
- **Persistence via the working-copy framework** (Track A design): core **§4.0–§4.6**
  (memory-is-the-truth; server draft + `sessionStorage` + files; the working-copy
  mechanism `open/update/save/discard`).
- **The k-grid render controller in the module** (Steps 1–2 design): molview-module.md
  **§14.2** (`molview.mountKgridRender` — the one render loop) + **§14.3** (in-window
  picking disabled while k-grid is on).

Also companion: [`structure-periodicity.md`](structure-periodicity.md)
(cell/kgrid data model).

> Convention: ☑ done · ◐ code done, verification pending · ☐ not started.
> Any open item's *design* is in the core (linked above); this file only tracks
> that the consumer work lands.

---

## Shipped (for orientation — design in the core, do not re-open)

- **Steps 1–2** — the k-grid render controller moved INTO the module
  (`molview.mountKgridRender`, molview-module.md §14.2); Results #1 rewired to it, its
  inline controller deleted. *(Step 2 residual: ◐ awaiting the user's manual
  confirm that k-grid tiles in Modify.)*
- **Track A A1/A2/A5a/A5b/A6** — SAVE + LOAD + filter + draft routed through the
  working-copy framework (core §4.6). *(A5a residual: ◐ browser check that the
  `.molbuilder_workspace/` draft appears/updates for a loaded file AND a new molecule.)*
- **Track D D1/D2** — coordinates in memory + the unified accessor API built on
  `ws.*` (core §1.2.1 / §2 / §3).

---

## Open work

### Track B — Modify consumes the MolView module (finish G2)

> **Goal in one sentence.** The Modify tab must mount the *concealed, packaged* MolView
> module into an **empty host** — exactly as the demo (`static/molview/demo.js`) and the
> Results structure inspector (`lib/inspectors/structure.js`) do — and **delete** every
> line of Modify's own viewer chrome that the module now owns. When done, `modify.html`
> hands the module one empty `<div>`, and `modify/viewer.js` contains ONLY Modify-specific
> logic (op-tabs, the state timeline, electrode/anchor UI, the Load-button plumbing) — no
> `embed()`, no `applyStructure` render, no toggle bar, no k-grid controller, no raw-3Dmol
> reach.
>
> **Reference (the correct pattern, already shipped & green):**
> `demo.js` → `data.openMolecule({text, filename})` then
> `mv.mount(emptyHost, ws, {mode:"modify", owner:"molview-demo"})`. The module builds the
> card, embeds the viewer, owns the render loop, puts the toggles in the View menu, and
> wires click-select — all of it.
>
> **Anti-drift rules for THIS track (read before every step):**
> 1. Do NOT hand-build viewer chrome anywhere. If something is missing from the built
>    card, the fix goes in the MODULE (so the demo gets it too), never a Modify patch.
> 2. Do NOT reach into raw 3Dmol from Modify. If a feature needs it, the module exposes
>    it on the handle, or the feature is dropped by explicit decision (Step B0).
> 3. Each step ends GREEN on its Check before the next begins. If a Check can't pass,
>    STOP and fix the step — do not proceed or work around it.
> 4. No new hacks. A relocation/patch that papers over "the module didn't do X" is a
>    hack; the answer is always to make the module do X.
> 5. **Missing capability → file a MODULE task, never a Modify workaround.** If Modify
>    needs a function the module doesn't expose, you do NOT implement it in Modify and you
>    do NOT reach around the module. You file a task/requirement AGAINST THE MODULE to add
>    or expose the needed API / handle method / data structure. Modify only consumes that
>    capability once the module ships it. (No raw-3Dmol reach, no bespoke overlay, no
>    "temporary" Modify-side copy.) Every DECIDE item the built card doesn't already cover
>    therefore resolves to a filed module-side task with the exact API/data it must add.

#### Inventory of `modify/viewer.js` (done 2026-07-11 — the map this track works from)

**DELETE — the module already owns these (proven in the demo):**
- `viewer.embed(...)` (the whole embed call) — module builds + embeds the viewer.
- `applyStructure(r)` — the bespoke render hook; module renders from `molview.data` on change.
- `_drawBase`, `_wireKgrid` — base render + k-grid; module owns via `mountKgridRender`.
- `refreshSelectionUI` selection painting — module owns click + selection display.
- `window.molbuilder.loadStructureText` — replaced by `data.openMolecule`.
- the flat `.viewer-controls` bar + `#viewer-view-controls` toggle span (in `modify.html`).

**KEEP — Modify-specific; must go through `molview.data` / the mount handle, never raw 3Dmol:**
- Op-tabs: `postOp`, `applyDelete/Center/Translate/AddAtom/Orient/Rotate/Electrode`,
  `readElcCommonBody`, `populateElectrodeMeta`, `renderLatticeRefRadios`, `refreshElcReadouts`.
- State timeline: `saveState`→`data.save(1)`, `retractState`→`data.load(-1)`,
  `restoreModifyState`→`data.load(0)`, `refreshUndoButton`, `currentStateBody`.
- `sendToBuild`; status helpers (`setStatus`, `setEditStatus`).
- The Load-button / sidebar-candidate plumbing in `selection-bootstrap.js` (`_commitFile` etc.).

**DECIDE — RESOLVED in Step B0 (2026-07-11). Findings + decisions:**
- **Click-to-select — COVERED by the module.** `mount.js:148` attaches `viewerAdapter` in the
  built-card path. → Delete Modify's redundant path in B2. No module task.
- **Region halos + frozen-atom halos + selection halo — COVERED by the module.**
  `viewer-adapter.js:7-9` draws all three; the built card attaches the SAME adapter
  (`mount.js:146-152`). → Delete Modify's redundant `viewerAdapter.attach(modify.handle)` in
  `selection-bootstrap.js` (lines 405-440) in B2. No module task.
- **Measurement chip — COVERED by the module.** `render.js:95` (via `mountRender`, called by
  the built-card path) wires `mountMeasurementOverlay`, which builds its OWN overlay element.
  → Delete Modify's redundant template `#selection-measurement-overlay` div in B1. No module task.
- **`#title-readout` — Modify fills it by CONSUMING the handle (not a hack, not raw 3Dmol).**
  The handle exposes `getStructure().title` + `onChange`; Modify keeps a tiny header updater:
  `handle.onChange(() => title.textContent = handle.getStructure()?.title)`. No module task.
- **Focus-molecule — COVERED by the module.** The built card's View-menu **"Reset view"**
  calls `handle.refit()` (mol-viewer-embed.js:1919), which re-frames/re-centers on the molecule
  — exactly what Modify's `#focus-molecule` tooltip describes. → Delete Modify's `#focus-molecule`
  button + `focusMolecule`/`snapPivotToCenter`/`_moleculeIndices` + the `interaction` embed opts
  in B2; users re-center via "Reset view". No module task. (Filed task #46 then deleted — no gap.)

---

- **Step B0 — Resolve the DECIDE list (facts + decisions, NO code changes).**
  Read the module to answer, for each DECIDE item: does the built card already provide it,
  does the handle expose it, or is it a Modify-only feature? Record each answer + the
  decision (module-absorbs vs drop) in this doc. Specifically nail:
  (a) does `mv.mount`'s built card wire click-to-select to `data.selection`? (user says yes — verify);
  (b) does it draw region halos, or is `viewerAdapter` still needed and, if so, against which handle?;
  (c) does the built card include the measurement chip + a title readout?;
  (d) is there a handle/API for focus-pivot, or is `#focus-molecule` dropped?
  **Check:** every DECIDE item has a written answer + decision in this section; no code touched.
  **Status:** ☑ (2026-07-11) — all 5 DECIDE items resolved above; **ALL covered by the module,
  NO gaps, NO module task, nothing blocks B2.** Click-select, all halos, measurement, and
  focus (View-menu "Reset view" = `handle.refit()`) are module-provided; title-readout is Modify
  consuming the handle. Track B is now a pure delete-and-mount. NO code touched.

- **Step B1 — Empty host in `modify.html`.** Replace the hand-built `.molview-card` block
  (the `#viewer` div, `.viewer-controls` bar, `#focus-molecule`, `#viewer-view-controls`,
  `#molview-fold`, `#selection-host`) with ONE empty host `<div id="molview-host"></div>`
  inside the numbered section. Keep the `#title-readout` header per Step B0's decision.
  **Check:** page still loads; `node -c` clean; NO test run yet (viewer.js still references
  the removed ids — expected; B2 fixes it). This step is not independently green — it is
  paired with B2; land B1+B2 together.
  **Status:** ☑ (2026-07-11)

- **Step B2 — Point the mount at the empty host + strip viewer.js's chrome.**
  In `selection-bootstrap.js`: set `HOST_ID = "molview-host"` so `mv.mount` gets an empty
  host and BUILDS the card. In `modify/viewer.js`: delete the DELETE-list items above;
  rewire the KEEP-list so op results and the timeline drive `molview.data` and the render
  reacts on its own (as the demo does). Apply Step B0's decisions for the DECIDE items.
  **Check:** `tests/test_molbuilder_e2e.py` full run GREEN (structure loads, renders,
  selection click works, ops apply, save-state/retract work); the View toggles are inside
  `.mol-viewer-menu-view .mol-viewer-menu-toggles` (the whole reason we started); no
  `#viewer` / `viewer.embed` / `applyStructure` left in `modify/viewer.js`.
  **Status:** ☑ (2026-07-11)

- **Step B3 — Delete now-dead code + tests; align docs.** Remove any viewer.js helpers
  orphaned by B2 (e.g. `_drawBase`, `_wireKgrid`, adapter glue if B0 dropped it), delete
  the obsolete `mountViewControls`-into-flat-bar test expectations, and update
  molview-module.md / this doc to state Modify uses the built card.
  **Check:** full Modify + molview-demo + structure-inspector e2e GREEN; `grep` shows no
  Modify reference to deleted seams; docs match code.
  **Status:** ☑ (2026-07-11)

- **Step 5 — cell-origin / box placement** (G4, separate track).
  **Status:** ☑ (commit `3d3f308`) — `Structure.cell_origin` + `resolve_cell_origin()`
  + the `calibrate` op; `render_fdf` translates atoms by `-resolve_cell_origin()` so
  SIESTA sees `[0,cell)`. Design in `structure-periodicity.md` §3c.

- **Step 6 — effective cell in the store** (G5, separate track). Decide whether/where
  the store carries the `resolve_cell` effective cell so a cell-less structure still
  has a box — per `structure-periodicity.md`, through the data model, not a viewer hack.
  **Check:** a molecule with vacuum set shows a box + tiles; the value comes through
  `ws.getStructure().periodicity`, computed server-side.
  **Status:** ☐ (separate; design-first.)

### Track A — persistence, remaining

> Design: core §4.6 (the working-copy mechanism).

- **A3 — draft + discard through the framework (decision-gated).** Decide whether the
  crash-surviving draft moves from `sessionStorage` to `/api/workingcopy/stage`+`discard`,
  or they coexist. Wire it if adopted.
  **Check:** an unsaved edit survives reload via the chosen mechanism; discard clears
  it. **Design decision first — may stay `sessionStorage`.**
  **Status:** ☐ (needs a decision)

- **A4 — remove the obsolete disk-based code.** `/api/selection/save-sidecar` +
  `/api/selection/refresh-hash` were removed 2026-07-05. **Remaining:** delete the
  Modify uses of `/api/selection/eval` + `/api/selection/atoms` (after A5) and the
  file-open `/api/build/load` path (after A6), once no other caller remains (Results
  legitimately views disk — verify before deleting); migrate/retire their tests.
  **Check:** grep — no Modify-tab code reads disk for regions/atoms/cell outside the
  working-copy framework; no obsolete endpoint remains with a live caller.
  **Status:** ◑ partial. (The single-model doc definition it also called for is now
  DONE — folded into the core §4/§4.6 + molview-module.md by the 2026-07-06 doc consolidation.)

### Track D — conceal the model, remaining

> Design: core §1.2.1 (the mandatory model + accessor API) + §1.4 (encapsulation).

- **D3 — route EVERY consumer through the API; seal the internals.** Save/draft done
  (`ws.getScratchBlob()` / `ws.getRegions()`; `save.js` + dispatcher `_scratchBlob`
  route through the accessors). **Remaining (browser-verified):** the RENDER —
  `viewer.js` `state.xyz.split` re-parse + the embed `addModel(string)` →
  `toAddAtoms()`/`model.addAtoms`; drop `/api/build/load` from the Modify load.
  **Check:** grep — no `state.xyz.split`, no `addModel(`, no `/api/build/load` in the
  Modify path; no consumer reaching past the accessors into raw arrays.
  **Status:** ◐ in progress.

- **D4 — columnar internals (the concealed layout).** The internal model IS columnar
  (`elements[]`, `positions[][]`, `regions` map, `frozen[]`) — never surfaced directly;
  only the D2 accessors touch it.
  **Check:** the store holds no per-atom `atoms[]` as canonical; the panel's list rows
  are materialised via the API.
  **Status:** ☐

### Review follow-ups (from the 2026-07-06 design-review passes)

Two adversarial review rounds fixed the data-loss/correctness bugs (setLabel dirty,
the metadata sink removed, the getAxisKind periodic-guess, the draft-leak on save-as,
the non-atomic save, pbc/axis_kind coherence, the modify-op frozen/labels wipe).
Remaining, confirmed-but-not-yet-done:

- **F1 — `annotations` dropped on Modify Save.** ~~Lost because the scratch blob didn't
  carry them.~~ **Status:** ☑ — the frontend model now carries `annotations` opaquely
  load→store→save (`data-model.js` `annotations: canvas.annotations || null` in the
  scratch blob + the open payload); `test_sidecar_annotations.py`.
- **F4 — load-order crash window in `_commitFile`.** ~~Canvas text set before store
  atoms; a same-atom-count reload could persist a draft pairing new xyz with old
  regions/frozen.~~ **Status:** ☑ — load order fixed (task #27).
- **F2 — `selection_rules` un-round-trippable by the codec.** Legacy field.
  **Status:** ☑ — resolved (round-trip or deprecate, task #26).

---

## Standing guardrails (apply to every open step)

- No structure/cell/kgrid read or write outside `ws.*` / the store / the module API.
- One k-grid render loop, in the module, forever (molview-module.md §14.2).
- Each step ends GREEN (its Check) before the next begins.
- Update this doc's ☐/◐/☑ as steps complete; if reality diverges, fix the doc first.
