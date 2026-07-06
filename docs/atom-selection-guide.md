# Atom selection — a developer's guide

**What this is.** A plain-language guide to how "which atoms are selected" works:
one **store** holds the selection; a **panel** shows it; a **viewer-adapter**
paints it in the 3D viewer and turns clicks back into selection. Three consumers,
one source of truth.

**What this is NOT.** The authoritative spec. `protocols/workspace-contract.md`
**Part II** (the MolView module) pins the data structures, the full store API,
the event protocol, and the scenarios. The store itself is part of the workspace
store — see `workspace-guide.md` for its `ws.selection.*` mutators. This guide
teaches how the pieces fit.

---

## 1. The one-paragraph mental model

The **selection store is the single source of truth** (it lives in the
workspace store, `_selection-store-impl.js`): it holds the atoms, the selected
indices, the mode (click / filter), the filters, and the sidecar-derived regions
+ frozen atoms. Everything else is a **consumer** that renders from the store
and routes user input *back* through the store's mutators:

- the **selection panel** (`selection-panel.js`) — the list/controls UI,
- the **viewer-adapter** (`lib/selection/viewer-adapter.js`) — paints overlays
  in the MolViewer and forwards 3D clicks to the store,
- **measurements** (`lib/selection/measurements.js`) — distance/angle chips.

Because all three read from and write to the one store, the panel, the 3D
highlight, and the measurement chips are always consistent — no cross-syncing.

```mermaid
flowchart TD
  STORE["selection store (workspace)\nsingle source of truth\natoms · selection · mode · filters · regions/frozen"]
  PANEL["selection-panel.js\nmount(rootEl)→{dispose}"]
  ADAPT["viewer-adapter.js\nattach(handle)→{dispose}"]
  VIEW["MolViewer (embed handle)"]
  PANEL -->|"ws.selection.set/toggle/…"| STORE
  STORE -->|"subscribe → re-render"| PANEL
  STORE -->|"subscribe → setOverlays"| ADAPT
  ADAPT -->|"overlays (halos)"| VIEW
  VIEW -->|"PickOpts.onPick → store.toggle"| ADAPT
  ADAPT --> STORE
```

---

## 2. The pieces

| Module | Public surface | Role |
|---|---|---|
| selection store | `ws.selection.*` (see workspace-guide) | the state + mutators (single source of truth) |
| `selection-panel.js` | `mount(rootEl) → {dispose}` | DOM UI — **no internal state**; renders from store, calls mutators |
| `lib/selection/viewer-adapter.js` | `attach(handle) → {dispose}` | subscribes store → `setOverlays`; routes viewer picks → `store.toggle` |
| `lib/selection/measurements.js` | measurement chips | distance/angle overlays on the current pick set |

---

## 3. Data (what the store holds)

- **`Atom[]`** — per-atom rows (element, name, residue, region, frozen…).
- **selection `indices`** — the currently-selected atoms (sorted, unique).
- **`pickOrder`** — the same atoms in **click order** (the angle-measurement
  vertex is `pickOrder[1]`); kept in lock-step with `indices`.
- **`mode`** — `"click"` (pick atoms) or `"filter"` (select by rule).
- **`filters`** + **`combinator`** (`"or"`/`"and"`) — element / index-range /
  region-label rules (evaluated in filter mode) and how they combine.
- **`isolate`** (boolean) + **`kgrid`** (`{enabled, dims, source}`) — VIEW state
  that lives in the store: "show selected only" and the k-grid tiling the panel
  drives (not the adapter, not a global handle).
- **regions + frozen** — from the `.molstruct.json` sidecar (named regions,
  frozen-atom lists).

The singleton store is `ws.selection`; **`createEphemeralStore()`** mints an
isolated instance with the same surface (minus the workspace-lifecycle methods)
for a readonly inspector.

Full shapes: `workspace-contract.md` Part II §12 (the store `_initialState`).

---

## 4. How to wire it (as a tab)

```js
// 1. mount the panel into its host
const panel = window.molbuilder.selectionPanel.mount(panelHost);
// 2. embed the viewer, then attach the adapter to its HANDLE
const handle = window.molbuilder.viewer.embed(viewerHost, { pick: {} });
const adapter = window.molbuilder.selectionViewerAdapter.attach(handle);
// 3. drive selection through the store (panel + viewer update automatically)
window.molbuilder.workspace.selection.set([1, 2, 3]);
// teardown:
panel.dispose(); adapter.dispose();
```

You never sync the panel and the viewer to each other — they both follow the
store.

---

## 5. Key concepts

- **The store is the ONLY source of truth.** The viewer's own halo/label
  rendering is **disabled**; the adapter paints *everything* via `setOverlays`
  so the store — not the viewer — decides what's highlighted.
- **Selection = overlays, picks = `onPick`.** The adapter pushes selection as
  overlays and receives clicks through the embed's `PickOpts.onPick` (→
  `store.toggle`). It never touches raw 3Dmol (see `molviewer-guide.md`).
- **`attach()` takes the embed HANDLE, not a raw viewer.** Passing a raw 3Dmol
  viewer errors (a guard against script-order bugs).
- **Click vs filter mode.** Click mode toggles atoms; filter mode selects by
  rule (`apply` evaluates the draft filters).
- **Batching.** Store mutators fire `notify()` once; consumers re-render once
  per change (event protocol, §4).

---

## 6. Common gotchas

- **Don't set selection on the viewer directly** — go through the store; the
  adapter reflects it.
- **Don't pass a raw 3Dmol viewer to `attach`** — pass the embed handle.
- **Don't hold selection state in the panel** — it's stateless by design;
  re-read the store.
- **Sidecar regions/frozen travel with the structure** — don't drop them on
  save (see `structure-guide.md` §5).

---

## 7. Where the authority lives (+ a heads-up)

- **`protocols/workspace-contract.md` Part II** — the spec: the store
  `_initialState` + surfaces (§12), the panel/adapter composition via `mountPanel`
  (§13), the k-grid rule (§14), measurement (§15). The `ws.selection.*` mutators
  are in §5.
- **`workspace-guide.md`** — the store + `ws.selection.*` mutators.
- **`molviewer-guide.md`** — the viewer the adapter drives (overlays/picks).

> **Shipped:** this module (store + panel + viewer-adapter) and the MolViewer
> **are one integrated MolView module** — the viewer + selection + k-grid +
> measurement share one contract and one in-memory model. The store↔panel↔adapter↔viewer
> seam described in this guide is that module. See `workspace-contract.md` Part II
> for the unified contract; remaining consumer-migration work (e.g. Modify adopting
> `mountPanel` + the fused layout) is tracked in `protocols/molview-migration-plan.md`.
