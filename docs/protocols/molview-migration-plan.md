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
- **Status:** ☐

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

## 5. Standing guardrails (apply to every step)
- No structure/cell/kgrid read or write outside `ws.*` / the store / the module API.
- One k-grid render loop, in the module, forever (after Step 1).
- Each step ends GREEN (its Check) before the next begins.
- Update this doc's ☐/☑ as steps complete; if reality diverges, fix the doc first.
