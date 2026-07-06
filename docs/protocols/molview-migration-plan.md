# MolView consumer migration — PERSISTENT plan

**This is a working plan, not a contract. It stays until the task is finished.**
Every step has a **completion standard**; a step is not done until its standard is
met and checked off here. Do not drift: if the work diverges from a step, update
this doc first, then proceed. When the whole task is done, this file is archived.

Companion: [`molview-module.md`](molview-module.md) (the module contract),
[`structure-periodicity.md`](structure-periodicity.md) (cell/kgrid data model),
[`workspace-contract.md`](workspace-contract.md) (the store = single source).

---

## 1. Real status of the molview module (verified 2026-07-05)

**What the module actually is today:**
- **Viewer** (`lib/mol-viewer-embed.js`): shipped, complete. `viewer.embed(host, opts)
  → handle`. Draws the cell wireframe when given `opts.cell`; tiles nothing itself.
- **Selection store** (`lib/workspace/_selection-store-impl.js` + `dispatcher.js`):
  shipped. The single source of truth (`ws.selection`), holds `kgrid{enabled,dims,
  source}`, `isolate`, atoms, selection.
- **Selection UI + wiring** (`selection-panel.js`, `selection/{viewer-adapter,
  mount-panel,measurements}.js`): shipped. `selection.mountPanel(host,{store,
  viewerHandle,mode})` composes panel + adapter against a store.
- **Display compute** (`lib/molview/{kgrid,render-pipeline,measurement-overlay}.js`
  + `fused-layout.css`): shipped as PURE COMPUTE. `tileKgrid`, `computeRender`.

**The gap in the module itself:** the k-grid **render controller** — the live loop
that *subscribes to the store, runs `computeRender`, and calls
`handle.setStructure(supercell)`* — is **NOT in the module**. It is **hand-written
inline in the Results structure inspector** (`inspectors/structure.js` ~296–316).
The module ships the *math* (`computeRender`) but not the *controller*. So any host
that wants k-grid must re-implement the controller. That is the core defect.

---

## 2. Consumer audit — every molecule-display instance

| # | Consumer (file) | Page | embed | panel | k-grid render | fused layout | render source |
|---|---|---|---|---|---|---|---|
| 1 | `inspectors/structure.js` | Results (structure) | ✓ | ✓ `mountPanel` | **inline, hand-crafted** | ✓ | module `setStructure` |
| 2 | `lib/trajectory/core.js` | Results (trajectory) | ✓ | ✗ (playback) | ✗ | ✗ | module `setStructure` (frames) |
| 3 | `lib/spectra/core.js` | `/spectra` + Results | ✓ | ✗ (modes) | ✗ | ✗ | module `setStructure` |
| 4 | `modify/viewer.js` | **Modify** | ✓ | ✗ **bespoke mount** (`selection-bootstrap.js` fetches the partial, not `mountPanel`) | ✗ **(missing)** | ✗ **(two cards)** | **bespoke** (`applyStructure` + own `state`) |
| 5 | `static/viewer.js` | `index` (Inspect) | ✓ | ? (sidebar handoff; verify) | ✗ | ✗ | verify |

**Per-consumer read:**
- **#1 Results structure inspector** — the closest to "on the module": embeds the
  viewer, composes the panel via `mountPanel`, uses `fused-layout`. BUT it owns the
  k-grid controller inline. This is the reference to *extract from*, not copy.
- **#2 trajectory** — frame playback; no selection, no k-grid by design. Uses the
  embed's cell wireframe. Likely fine as-is; confirm it still gets a cell.
- **#3 spectra** — vibrational-mode display; no selection/k-grid. Fine as-is.
- **#4 Modify** — **the real gap.** Runs on a bespoke viewer (`modify/viewer.js`,
  its own `state`/`applyStructure` render), mounts the panel bespoke (not the
  module's `mountPanel`), no `fused-layout` (viewer + panel are two separate cards),
  and has no k-grid controller at all. Everything the user is hitting traces here.
- **#5 index Inspect viewer** — standalone; needs a quick check of whether it uses
  `mountPanel`/selection and whether it's live or legacy.

---

## 3. The gaps, named

- **G1.** The k-grid render **controller** lives inline in one host (#1), not in the
  module → cannot be reused; Modify has none.
- **G2.** Modify (#4) does not consume the module: **bespoke viewer render**,
  **bespoke panel mount**, **no fused layout**, **no k-grid**.
- **G3.** The cell the render uses must come from the **store** (workspace data
  model), never from a load response or a per-render hand-read (the mistake just
  reverted). This is a *rule the controller must follow*, not a separate feature.
- **G4 (display, separate).** The cell wireframe draws from the origin, but the
  electrode builder centers atoms at the origin → the box sits off the molecule.
  A cell-origin/wrap convention issue, independent of the module migration.
- **G5 (data-model, separate).** Whether the store carries the *effective* cell
  (`resolve_cell`: bbox+vacuum when none is explicit) is a `structure-periodicity.md`
  concern; the module just receives whatever cell the store holds.

---

## 4. The plan — steps + completion standards

> Rule: each step reads/writes structure data ONLY through `ws.*` / the store and
> the module API. Zero hand-crafted data access. A step is done only when its
> **Check** passes.

### Step 0 — this audit + plan
- **Do:** produce §1–§3 above from the actual code.
- **Check:** user has reviewed this doc and approved the step list. ← *gate*
- **Status:** ☑ approved 2026-07-05.

### Step 1 — move the k-grid render controller INTO the module (fixes G1)
- **Do:** add `molview.mountKgridRender(handle, store, {coords, elements})
  → {dispose}` in `lib/molview/`. It subscribes to the store, runs `computeRender`
  with the cell **read from the store** (G3), calls `handle.setStructure(supercell)`
  on enable, restores the unit cell on disable, unsubscribes on `dispose`. It is the
  ONLY k-grid render loop in the codebase.
- **Do:** rewrite Results #1 to call it; **delete** its inline controller.
- **Check:** (a) the controller exists once, in the module, on
  `window.molbuilder.molview`; (b) NO k-grid `subscribe`/`computeRender`/
  `setStructure` loop remains in `inspectors/structure.js`; (c) the existing Results
  k-grid e2e (`test_sidecar_cell_reaches_viewer_and_kgrid_tiles`) passes unchanged.
- **Status:** ☑ done 2026-07-05. `molview.mountKgridRender` added to
  `render-pipeline.js`; the Results inspector's inline controller + `_buildXyz`
  deleted, now calls the module; (a)(b)(c) verified + 2 controller unit tests.

### Step 2 — Modify consumes the module k-grid controller (fixes k-grid on Modify; part of G2)
- **Do:** in Modify, call `molview.mountKgridRender` against Modify's viewer handle
  + the workspace store. The cell is read from the store by the module (G3); Modify
  writes NO cell-reading or tiling code.
- **Do:** ensure the controller coordinates with Modify's base render (only the
  module tiles; the base render sets the unit cell) — documented, no duplicate loop.
- **Check:** (a) enabling k-grid in Modify tiles the supercell (manual on a real
  celled structure + an e2e if feasible); (b) `grep` shows ZERO hand-crafted
  k-grid/cell/tiling in `modify/viewer.js` — the render loop is the module's.
- **Status:** ◐ code done 2026-07-05 — Modify's `applyStructure` renders the base
  cell from the store (`_cellFromStore`) + hands k-grid to `mountKgridRender`; (b)
  verified (grep clean); panel + controller confirmed sharing `ws.selection`.
  **(a) awaiting the user's manual confirm** that k-grid tiles in Modify.

### Step 3 — Modify composes the panel + viewer through the module (fixes G2: unify)
- **Do:** replace Modify's bespoke panel mount with `selection.mountPanel(host,
  {store, viewerHandle, mode})`; adopt `fused-layout.css` so the viewer + panel are
  ONE card, as on Results.
- **Check:** (a) the Modify viewer + selection panel render as one fused card;
  (b) `selection-bootstrap.js` no longer fetches/mounts the panel by hand — it goes
  through `mountPanel`; (c) selection still works in Modify (click, filter, assign);
  existing Modify selection tests pass.
- **Status:** ☐

### Step 4 — Modify's base render goes through the module (finish G2; optional if 1–3 suffice)
- **Do:** retire `modify/viewer.js`'s bespoke `applyStructure` rendering in favor of
  the module handle's `setStructure`, driven by the store — keeping only the
  Modify-specific UI (electrode/anchor/slab controls) that legitimately belongs to
  the tab.
- **Check:** the viewer render path in Modify is the module's; no second copy of the
  structure lives in `modify/viewer.js state.*` beyond what the tab's own UI needs;
  all Modify op tests pass.
- **Status:** ☐ (revisit scope after Step 3 — may be deferred.)

### Step 5 — cell-origin / box placement (G4, separate track)
- **Do:** decide + implement the cell-origin convention so the wireframe wraps the
  atoms (draw from min corner, or wrap atoms into `[0,cell)` for display) —
  design-first in `structure-periodicity.md`.
- **Check:** the box wraps the structure (not offset to the molecule center) on a
  centered electrode junction; k-grid copies tile seamlessly.
- **Status:** ☐ (needs a design decision; do NOT fold into 1–4.)

### Step 6 — effective cell in the store (G5, separate track)
- **Do:** decide whether/where the store carries the `resolve_cell` effective cell
  so a cell-less structure still has a box — per `structure-periodicity.md`, through
  the data model, not a viewer hack.
- **Check:** a molecule with vacuum set shows a box + tiles; the value comes through
  `ws.getStructure().periodicity`, computed server-side.
- **Status:** ☐ (separate; design-first.)

---

## Track A — Persistence: ADOPT the existing working-copy framework (DATA-LOSS + architecture; PRIORITY)

**Finding (2026-07-05 audit).** A full persistence framework already exists and is
tested, but is **orphaned from the UI** — I never wired it, and instead the frontend
grew a parallel ad-hoc path. That parallel path is the bug.

*Already built + tested (`test_workingcopy{,_api,_structure}.py`):*
- `molbuilder/workingcopy.py` (core): `_atomic_write_bytes` + `WorkingCopy.save()`
  writes EVERY codec file (`.xyz` **and** `.json`) atomically in ONE call;
  `stage()` = crash-surviving draft; `discard()`.
- `StructureCodec` (`workingcopy_structure.py`): `.files()` → the `[(xyz),(json)]`
  pair; `_sidecar_dict` writes the FULL periodicity (cell/axis_kind/vacuum/kgrid) +
  hash = sha256 of the `.xyz`.
- Endpoints `/api/workingcopy/{save,discard,orphans}`.

*What the frontend uses instead (the parallel path to delete):*
- **Save** = `projects.writeFile` (`.xyz`) + `save-sidecar` (`.json`) — TWO calls,
  non-atomic, the second fire-and-forget with a swallowed error → the crystal saved
  with no `.json`. **Violates §4.0.1**, which the contract claimed but the code never
  honored.
- **Draft** = ad-hoc `sessionStorage` (`molbuilder.workspace.v1`).

The fix is to **adopt the framework**, not build a third thing.

### A1 — verify the framework is correct for our needs
- **Do:** confirm `/api/workingcopy/save` writes both files atomically + hash-tied
  with the FULL periodicity; confirm `scratch_blob`/`from_scratch` round-trip
  regions + frozen + periodicity; confirm save-as (new `target`) + same-path
  overwrite semantics.
- **Check:** a test posts a scratch blob carrying a cell → BOTH files written; the
  `.json` has cell/axis_kind/kgrid and hash = sha256 of the written `.xyz`.
- **Status:** ☑ done 2026-07-05. `test_save_writes_full_periodicity_and_hash_tie`:
  save writes both files, `.json` carries cell/axis_kind/kgrid + hash-tie. Framework
  is correct. **Open A2 question flagged below:** the endpoint requires a `source`
  path — must confirm it handles a GENERATED structure (no on-disk source).

### A2 — route SAVE through `/api/workingcopy/save` (delete the two-call split)
- **Do:** `save.js` serialises the STORE (`ws.*`) into the working-copy scratch blob
  (`{xyz, sidecar:{regions,frozen,periodicity,...}}`) and POSTs
  `/api/workingcopy/save {source, data, target}` ONCE. Delete `writeFile` +
  `save-sidecar`. The overwrite CONFIRM stays in the frontend (the endpoint has no
  gate). A failure is SURFACED, never swallowed.
- **Check:** (a) saving a celled crystal writes BOTH files (json carries the cell)
  in ONE call; (b) grep: no `writeFile`+`save-sidecar` split in `save.js`; (c) a
  write failure raises a visible error.
- **Status:** ☑ done 2026-07-05. Backend gate (A2a) + `save.js` rewired: `save()` →
  `_saveDataset` → ONE POST `/api/workingcopy/save` with the store's scratch blob
  (`_buildScratchBlob`); `_writeWithOverwriteGate`/`_persistLabelsToDestination`/
  `_postWriteSuccess` deleted. (a) A1 test + the one-call save-js test; (b) grep
  clean; (c) `test_server_error_surfaces`/`test_network_throw_surfaces_envelope`.
  38 save/workingcopy/dialog tests pass. **NOTE:** `/api/selection/save-sidecar` +
  `/api/selection/refresh-hash` now have no frontend caller — dead, remove in A4.

**Memory model (the contract, working-copy-persistence.md).** "Memory" is NOT
browser-only: every edit auto-writes a **transient draft** (`<project>/.molbuilder_
workspace/`) kept consistent with the in-memory data (`update` = the only automatic
write); `save` promotes it to the real file + drops it; a crash recovers from it.
The frontend currently VIOLATES this — it uses `sessionStorage` and never calls
`/api/workingcopy/update`, so the transient draft never exists. That's why there's no
temp file, and why the server-side filter has nothing memory-consistent to read.

### A5a — keep the transient draft in sync on every edit (the contract; PREREQUISITE)
- **Do:** the Modify store calls `/api/workingcopy/update` on every edit (label,
  geometry, periodicity) so `.molbuilder_workspace/` always mirrors memory. Replace
  the ad-hoc `sessionStorage`-only draft with the framework draft (or back it with
  the framework). Load establishes the working copy; discard drops the draft.
- **Check:** after assigning a label (no Save), the project's `.molbuilder_workspace/`
  draft reflects it; a reload/crash recovers it via the framework.
- **Status:** ◐ code done 2026-07-05. `dispatcher._scratchBlob` + `_persistDraft`
  POST `/api/workingcopy/update {source, data}` in the debounced persist (alongside
  sessionStorage), on every edit; skipped when there's no source file. Syntax OK +
  25 dispatcher tests pass. **Browser check pending** (the `.molbuilder_workspace/`
  draft appears + updates).

### A5b — SELECTION + FILTER read the workspace, not the STALE saved file (BUG FIX)
- **Why:** the Modify filter-by-label posts `/api/selection/eval`, which reads the
  saved DISK structure + sidecar -> it sees only saved labels. Repro (user): assign
  "L-electrode" (shows in the click list) then filter -> `no region labelled
  "L-electrode" (known: ['BDT'])`. Same for the atom LIST (`/api/selection/atoms`).
- **Do:** the filter + atom list read the CURRENT workspace, not the saved file.
  With A5a the draft mirrors memory, so the server reads the DRAFT (memory-consistent)
  -- or the client evaluates against the store's atoms. Either way, not the stale
  saved file. (Results VIEWS a saved file on disk -- reading disk there is legitimate;
  this is the editable Modify workspace.)
- **Check:** assign a label -> filter by it finds those atoms WITHOUT saving first;
  a modified/unsaved structure filters against its current (draft/memory) atoms.
- **Status:** ☑ done 2026-07-05. `applyFilter` sends the store's `atoms` (not
  `structure_path`); `/api/selection/eval` evaluates against them via
  `_struct_from_atoms` (label/element/index/residue rules need no geometry) instead
  of `_load_structure` (disk). Guard now checks `state.atoms`, not `sourceFile`.
  Test: eval finds an in-memory "L-electrode" label with NO disk file. 66 tests
  pass. (Results `{structure_path}` path kept for viewing saved files.)

### A6 — LOAD a file through the framework (not the ad-hoc file-open)
- **Why:** opening a file uses the ad-hoc path -- `/api/build/load` (reads only the
  `.xyz` TEXT, no sidecar) + a separate `/api/selection/atoms` fetch for the cell.
  The framework already has `/api/workingcopy/open`, which loads the `.xyz` + `.json`
  pair (regions + frozen + periodicity) in ONE call.
- **Do:** the Modify open-a-file flow (`selection-bootstrap.commitFile` / `file.js`)
  uses `/api/workingcopy/open` -> the store. Delete the file-open `/api/build/load`
  call + the `/api/selection/atoms` cell fetch.
- **Check:** opening a saved structure loads atoms + labels + cell in ONE call; grep:
  no `/api/build/load` or `/api/selection/atoms` in the Modify file-open path.
- **Status:** ☐

### A3 — draft + discard through the framework (decision-gated)
- **Do:** decide whether the crash-surviving draft moves from `sessionStorage` to
  `/api/workingcopy/stage`+`discard`, or they coexist. Wire it if adopted.
- **Check:** an unsaved edit survives reload via the chosen mechanism; discard clears
  it. **Design decision first — may stay `sessionStorage`.**
- **Status:** ☐ (needs a decision)

### A4 — REMOVE the obsolete disk-based code + define the architecture
- **Do:** delete every endpoint that read/wrote disk for what is now in-memory
  Modify-workspace data, ONCE its callers are gone (verify no OTHER caller — e.g.
  Results legitimately viewing disk — before deleting): `/api/selection/save-sidecar`
  + `/api/selection/refresh-hash` (dead after A2); the Modify uses of
  `/api/selection/eval` + `/api/selection/atoms` (after A5); the file-open
  `/api/build/load` path (after A6). Migrate/retire their tests. Then define the
  ONE model in molview-module.md + workspace-contract.md + save-flow.md +
  working-copy-persistence.md: ws.* in memory is the truth; the working-copy
  framework (open/save/draft/discard) is the ONLY thing that touches disk for the
  workspace; molview only DISPLAYS.
- **Check:** grep -- no Modify-tab code reads disk for regions/atoms/cell outside the
  working-copy framework; no obsolete endpoint remains with a live caller; docs
  describe the single model.
- **Status:** ☐

> Track A runs BEFORE the remaining Track B molview steps — it's the data-loss fix.
> The molview k-grid Steps 1–2 (done) stand; Steps 3–6 below become Track B.

## 5. Standing guardrails (apply to every step)
- No structure/cell/kgrid read or write outside `ws.*` / the store / the module API.
- One k-grid render loop, in the module, forever (after Step 1).
- Each step ends GREEN (its Check) before the next begins.
- Update this doc's ☐/☑ as steps complete; if reality diverges, fix the doc first.
