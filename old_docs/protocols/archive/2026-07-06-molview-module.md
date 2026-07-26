> **ARCHIVED 2026-07-06 — MERGED into [`../workspace-contract.md`](../workspace-contract.md)
> (Part II, the MolView + Workspace core contract).** Historical snapshot; do not
> treat as current. NOTE: the line below (a 2026-07-03 snapshot) claims
> `atom-annotations.md` was superseded — that is **stale**: only the fused-viewer
> material folded out of it; its per-atom annotation *channels* model remains LIVE
> at [`../atom-annotations.md`](../atom-annotations.md).

# MolView module — design & contract

**Status:** v1, 2026-07-03. **Sole source of truth** for the MolView module.
Supersedes `embedded-viewer.md`, `atom-selection.md`, and `atom-annotations.md`
(archived under `docs/archive/2026-07-03-*`).

**The code is the standard.** This document describes what the shipped module
does; any deliberate change updates this doc first, code second. Every API name
below is copied from the code.

**Module code:**
- Viewer: `lib/mol-viewer-embed.js` (+ `mol-viewer-embed.css`)
- Selection state: `lib/workspace/_selection-store-impl.js`, `lib/workspace/dispatcher.js`
- Selection UI + viewer wiring: `lib/selection-panel.js`, `lib/selection/{viewer-adapter,mount-panel,measurements}.js` (+ `measurement-chip.css`)
- Display compute: `lib/molview/{kgrid,render-pipeline,measurement-overlay}.js` (+ `fused-layout.css`)

---

## 1. What the module is

**MolView is one module: the structure viewer, with atom selection as part of it.**
Selection, measurement, and k-grid display are not bolted-on neighbours — they are
parts of this module. The module renders a structure, lets the user pick/filter
atoms, reads off geometry, and displays the periodic tiling.

Two hard rules define the whole module:

1. **Data and parameters come IN through the API. The module never fetches and
   never parses.** Structure text, the lattice `cell`, and the k-grid `dims` are
   handed to the module by the host; the module draws them. Reading files,
   associating a `.fdf` with a structure, and extracting a cell/k-grid are the
   **host's** job (the results tab, backed by `molbuilder/parse/`; or a designed
   structure on Modify). See § 8.
2. **The store is the single source of truth for selection state.** The panel,
   the viewer-adapter, and the viewer never talk to each other directly — they
   all go through the store (§ 5).

## 2. Boundary — who owns what

| The module owns | The host owns |
|---|---|
| 3-D rendering + the viewer card chrome (style / labels / axes / reset / screenshot / background / export) | Page layout, where the card sits |
| Overlay state: axes, **cell wireframe** (given a lattice), labels, arrows, atom overlays, pick halos, animation frame, camera | **Data fetching** (`/api/*`, sidebar reads, polling) |
| The **selection store** + the selection panel + the viewer-adapter | The **cell + k-grid parameters** it supplies to the module (§ 8) |
| Measurement math + readout overlay; k-grid tiling compute | Project context (current dir, file naming) |

Crossing the boundary:
- **Host → module (mutate):** `handle.setStructure/setStyle/setAxes/setOverlays/setPick…`; `store.setSelection/setIsolate/setKgrid/…`.
- **Host → module (read):** `handle.getAtomCount/getAtomCoords/getElements/getLattice/getCell/getPickedIndices/getCamera`; `store.getState()`.
- **Module → host (events):** `opts.onReady(handle)`, `opts.onError`, `opts.pick.onPick`, `opts.export.onExport`, `opts.animation.onFrame`; `store.subscribe(fn)`.

The host never reads 3Dmol objects directly; the module never reads host DOM
outside its own card.

## 3. The viewer — `viewer.embed(host, opts) → handle`

`window.molbuilder.viewer.embed(host, opts)` mounts the viewer card and returns a
**handle**. Structure text (`opts.xyz` / `opts.pdb`) + a declarative options object
come in through the call; the viewer maintains the drawing. Handle methods (exact):

```
setStructure  setStyle  setAxes  setOverlays  setPick  setPickedIndices
getPick  getPickedIndices  getAtomCount  getAtomCoords  getElements
getLattice  getCell  getCamera  setCamera
playAnimation  setAnimationFrame  refit  screenshot  exportData  dispose
```

- **`opts.lattice`** (3×3 row vectors) → `getLattice()` returns it and k-grid
  tiling uses it (§ 7). The **cell wireframe** draws only when **`opts.cell`** is
  also set (`_redrawCell` gates on `state.current.cell`). The viewer does **not**
  parse a lattice from the file text; the host passes it (§ 8).
- **`setOverlays(spec)`** is how selection is painted (halos, region tints,
  isolate opacity). **`setStructure({xyz, lattice})`** is how the displayed
  atoms are replaced (e.g. a k-grid supercell, § 8).
- Hard deps (`embed()` throws if absent): `$3Dmol`, `molbuilder.viewer.create`,
  `molbuilder.fmt`. Soft deps degrade silently: `molbuilder.axes`, `molbuilder.style`.

## 4. The selection store — `ws.selection` (`window.molbuilder.workspace.selection`)

The store holds all selection + view state. The one process-wide instance is
reached through **`window.molbuilder.workspace.selection`** (`ws.selection`),
which the dispatcher builds around the `_createStore` factory — there is **no**
public `molbuilder.selection.store` singleton (retired Phase 9).
`molbuilder.selection.createEphemeralStore()` mints an isolated instance for a
readonly inspector. (The rest of `molbuilder.selection.*` holds the module's
functions: `mountPanel`, `measurements`, `viewerAdapter`, `_createStore`,
`_surfaceSnapshot`.)

### 4.1 State (exact — `_initialState`)

```
{
  sourceFile:  null | string      // absolute structure path
  atoms:       Atom[]             // {index(0-based), element, x, y, z, labels[],
                                  //  isFrozen, atomName?, residueName?, chainId?}
                                  //  TRANSITIONAL per-atom shape.  The MANDATED model
                                  //  is COLUMNAR behind the accessor API (workspace-
                                  //  contract.md §1.2.1 + §1.4) -- reach it via ws.*,
                                  //  not this raw array; Track D4 seals it.
  selection:   number[]           // THE selection set (sorted; the canonical state)
  pickOrder:   number[]           // same atoms in click order (angle vertex = pickOrder[1])
  mode:        "click" | "filter"
  isolate:     boolean            // "show selected only" — VIEW state (was the
                                  //  adapter's setIsolateMode; moved into the store)
  kgrid:       { enabled: boolean, dims: [nx,ny,nz], source: "free" | "fixed" }
  filters:     Filter[]           // {kind: by_element|by_index|by_label, value}
  combinator:  "or" | "and"
  loading:     boolean
  error:       null | string
}
```

`selection` is canonical; `pickOrder` is its click-order shadow (kept in lock-step
by every mutator). **`isolate` and `kgrid` are VIEW state that lives in the store**
— not on the adapter, not on a global handle. This is what makes the panel drive
them through the store (§ 5), obeying the single-source rule.

### 4.2 Surfaces

Consumers never touch the raw store; they use a renamed **surface**:

- **`ws.selection`** (dispatcher) — the singleton surface. Methods:
  `toggle set add remove all invert clear setMode setIsolate setKgrid setFilters
  addFilter removeFilter updateFilter setCombinator applyFilter writeLabel
  getAtoms getState subscribe adoptSession setSourceFile setLoader refreshAtoms`.
- **`createEphemeralStore()`** — an isolated instance with the **same surface
  minus** `getAtoms / setSourceFile / refreshAtoms` (workspace-lifecycle methods
  a readonly inspector doesn't use). Both surfaces reshape state via the one
  `_surfaceSnapshot` shaper, so `getState()`/`subscribe(fn)` deliver
  `{indices, …, isolate, kgrid, …}` (raw `selection` is renamed `indices`).

`setKgrid(patch)` is `source`-aware: in `"fixed"` a bare `dims` edit is ignored
(the values are the run's, read-only); in `"free"` `dims` is clamped so
`natoms · nx·ny·nz ≤ 20000`. `enabled` always applies.

## 5. Composition — panel + adapter + viewer, through the store

```
        selection-panel.js            viewer-adapter.js
        (DOM: list/filter/            (paints overlays via
         checkboxes)                   handle.setOverlays;
              │                        forwards viewer clicks
              │                        to store.toggle)
              └────────► STORE ◄───────────┘
                     (single source)
```

- **`selection.mountPanel(host, {store, viewerHandle, mode})`** fetches the panel
  partial, mounts `selection-panel`, and attaches `viewer-adapter` to the handle
  — both bound to the given `store` (the singleton or an ephemeral one).
- The **panel** renders from `store.getState()` and calls mutators on input
  (toggle, filter, `setIsolate`, `setKgrid`). The **adapter** subscribes to the
  store and paints halos/region-tints/isolate via `setOverlays`; it forwards
  viewer clicks to `store.toggle`.
- Panel and adapter never reference each other. `mode:"readonly"` hides the
  panel's write controls; clicks still feed the store.
- `fused-layout.css` lets the host place the panel as a foldable side/bottom
  region of the viewer card (host layout choice; the viewer offers no layout API).

## 6. Measurement — `measurements.compute` + the overlay

- **`selection.measurements.compute(selection, atomsMeta, positions, pickOrder)`**
  → `{kind: "xyz" | "distance" | "angle", display}` (or null). 1 atom → position,
  2 → distance, 3 → angle (vertex = `pickOrder[1]`).
- **`molview.mountMeasurementOverlay(viewerHost, {store, coordsProvider}) →
  {render, dispose}`** paints that readout as text in the viewer card, derived from
  the store selection. Coords come from `coordsProvider()` (the current frame /
  the viewer handle) — the store never holds coordinates. Hidden while k-grid is on.

## 7. k-grid & the render pipeline

k-grid display = **copies of the atoms offset by the lattice vectors**. Pure compute:

- **`molview.tileKgrid(coords, cell, [nx,ny,nz]) → {positions, sourceIndex, nimages}`**
  — the tiling.
- **`molview.computeRender(coords, view, cell) → {positions, sourceIndex}`** — the
  ordered pipeline: layer 2 (selection/**isolate** → visible global indices) →
  layer 3 (k-grid tile). Because isolate runs before tiling, **isolate ON + k-grid
  ON tiles only the selected atoms.** `sourceIndex[m]` maps each drawn position
  back to its unit-cell atom (element/label lookup; images share their unit-cell
  atom's identity).

The current live path (structure inspector) tiles inline: on `store.setKgrid`
change it runs `computeRender` against the unit-cell coords + the lattice and calls
`handle.setStructure(supercell)`; disabling restores the unit cell.

### 7.1 In-window picking is disabled while k-grid is on; the panel still works (agreed behavior)

**With many duplicated atoms on screen, a mouse click inside the molview window is
ambiguous and messy** — "which copy did you pick?" has no answer. So while
`kgrid.enabled`:

- **In-window picking is disabled** — clicking an atom in the 3-D molview does
  **not** toggle the selection. Selection halos + the measurement overlay also
  stand down in the window (they are keyed by unit-cell index and can't map onto
  copies).
- **The selection PANEL stays fully functional** — filter and click-select on the
  atom *list* work normally, because the list is the original unit-cell atoms
  (no ambiguity). The selection is still curated there, and the render re-tiles on
  change.
- **The selection is recorded internally** (never cleared) — so with **isolate ON
  + k-grid ON the render copies/duplicates ONLY the selected atoms** across the
  grid (§ 7 above). Turning k-grid off restores in-window picking, halos, and
  measurement.

So k-grid disables *pointing at the 3-D view*, not *selecting*: you keep curating
the selection through the panel; you just can't click the copies.

## 8. The k-grid / cell parameter boundary (host supplies; module never parses)

The module **only cares whether a `cell` and a `kgrid` were handed to it.** It does
not read files, does not associate a `.fdf`, does not extract a k-grid.

- The **cell** is passed as `opts.lattice` (viewer) — the host obtains it. For a
  result, that's `molbuilder/parse/` (`StructureResult.cell` / `JobResult`
  geometry) surfaced by the **results tab**; on Modify, it's the structure being
  designed.
- The **k-grid** reaches the store as `setKgrid({source:"fixed", dims})` when the
  host supplies it (the results tab, from the `.fdf` kgrid diagonal that
  `parse/dirs/job.py` already extracts), or `"free"` when the user experiments.

`molbuilder/parse/` is the sole parser (see `parse-module.md` § 9). No parsing
lives in this module. How the host **resolves** `cell` + `kgrid` (the
`resolve_cell` precedence, the `axis_kind` enum `{periodic, isolated, transport}`,
the axis_kind-gated k-grid rule — k-grid > 1 only on a `periodic` axis) is defined
in **`structure-periodicity.md`** — the module just receives the result.

## 9. Atom-index display rule

Indices are **0-based internal, 1-based user-facing** (`data-vocabulary.md` § 3.1).
Internal state (`atom.index`, `selection`, `pickOrder`, `sourceIndex`) is 0-based;
anything a user reads (panel `#` column, viewer labels, measurement readout) is
converted via `lib/workspace/_atom-index.js` `toDisplay` at the edge. Never let a
1-based value into state; never show a 0-based value.

## 10. Test affordances

- Node-tested pure modules: `tileKgrid`, `computeRender`, `mountMeasurementOverlay`
  (`tests/test_{kgrid,render_pipeline,measurement_overlay}_js.py`), the store
  (`test_selection_store_js.py`), the dispatcher (`test_workspace_dispatcher_js.py`).
- Browser e2e (structure inspector): measurement overlay, clicks→store, and (when
  the host supplies a cell) k-grid tiling — `tests/test_structure_inspector_measurement_e2e.py`.
- The inspector exposes `viewerSlot.__molbuilder_test_handle` + `__molbuilder_test_store`
  (test-only) so e2e drives the viewer + store without canvas clicks.

## 11. What this supersedes + why

| Archived doc | Was | Why it folded here |
|---|---|---|
| `embedded-viewer.md` | the viewer contract | the viewer is part of this one module |
| `atom-selection.md` | the selection module | selection is part of this one module; its §404 (isolate on the adapter + global handle) was already superseded by isolate-in-store |
| `atom-annotations.md` | a per-atom-annotations feature that had accreted the whole fused-module design | mis-scoped umbrella; its correct design folded here, its obsolete parts (FrameSet/render-controller, inline `Lattice=` parse, k-grid-on-thin-inspector) dropped |

## 12. Decisions log

| Date | Decision |
|---|---|
| 2026-07-03 | One doc for the MolView module (viewer + selection as one). Code is the standard. isolate + kgrid live in the store (view-state). k-grid: host supplies cell+kgrid; the module tiles, never parses. `embedded-viewer.md`/`atom-selection.md`/`atom-annotations.md` archived. |
