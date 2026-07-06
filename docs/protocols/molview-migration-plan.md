# MolView consumer migration — OPEN-WORK tracker

**This is a working plan, not a contract. It stays until the task is finished.**
It tracks only the **still-open** MolView consumer-migration work. Every design
that has SHIPPED is now core behaviour and lives in the core contract — this
tracker points there rather than re-describing it.

**The design lives in the core** —
[`workspace-contract.md`](workspace-contract.md) (the MolView + Workspace core
contract):

- **The accessor API / concealed model** (Track D design): core **§1.2.1** (the
  mandatory one-model + accessor-API contract) + **§1.4** (encapsulation) +
  **§2/§3** (the read/write accessors).
- **Persistence via the working-copy framework** (Track A design): core **§4.0–§4.6**
  (memory-is-the-truth; server draft + `sessionStorage` + files; the working-copy
  mechanism `open/update/save/discard`).
- **The k-grid render controller in the module** (Steps 1–2 design): core Part II
  **§14.1** (`molview.mountKgridRender` — the one render loop) + **§14.2** (in-window
  picking disabled while k-grid is on).

Also companion: [`structure-periodicity.md`](structure-periodicity.md)
(cell/kgrid data model).

> Convention: ☑ done · ◐ code done, verification pending · ☐ not started.
> Any open item's *design* is in the core (linked above); this file only tracks
> that the consumer work lands.

---

## Shipped (for orientation — design in the core, do not re-open)

- **Steps 1–2** — the k-grid render controller moved INTO the module
  (`molview.mountKgridRender`, core Part II §14.1); Results #1 rewired to it, its
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

> Design: core Part II §12–§14 (the store, composition via `mountPanel`, the k-grid
> controller). Each step reads/writes structure data ONLY through `ws.*` / the store
> and the module API.

- **Step 3 — Modify composes panel + viewer through the module.** Replace Modify's
  bespoke panel mount with `selection.mountPanel(host, {store, viewerHandle, mode})`;
  adopt `fused-layout.css` so viewer + panel are ONE card (as on Results).
  **Check:** (a) Modify viewer + panel render as one fused card; (b)
  `selection-bootstrap.js` no longer fetches/mounts the panel by hand — it goes
  through `mountPanel`; (c) selection still works (click, filter, assign); Modify
  selection tests pass.
  **Status:** ☐

- **Step 4 — Modify's base render goes through the module** (optional if 1–3 suffice).
  Retire `modify/viewer.js`'s bespoke `applyStructure` rendering in favour of the
  module handle's `setStructure`, driven by the store — keeping only the
  Modify-specific UI (electrode/anchor/slab controls).
  **Check:** the Modify render path is the module's; no second copy of the structure
  lives in `modify/viewer.js state.*` beyond what the tab's own UI needs; all Modify
  op tests pass.
  **Status:** ☐ (revisit scope after Step 3 — may be deferred.)

- **Step 5 — cell-origin / box placement** (G4, separate track). Decide + implement
  the cell-origin convention so the wireframe wraps the atoms (draw from min corner,
  or wrap atoms into `[0,cell)` for display) — design-first in
  `structure-periodicity.md`.
  **Check:** the box wraps the structure (not offset to the molecule centre) on a
  centred electrode junction; k-grid copies tile seamlessly.
  **Status:** ☐ (needs a design decision; do NOT fold into 1–4.)

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
  DONE — folded into the core §4/§4.6 + Part II by the 2026-07-06 doc consolidation.)

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

- **F1 — `annotations` dropped on Modify Save.** The v4 per-atom annotation channels
  ([`atom-annotations.md`](atom-annotations.md)) are lost because the frontend scratch
  blob (`dispatcher._scratchBlob`) doesn't carry them: open an annotated file, edit,
  Save → `annotations: {}`. Real data-loss on a shipped field. Fix: carry annotations
  opaquely load→store→save (a frontend-model addition). **Status:** ☐
- **F4 — load-order crash window in `_commitFile`.** Canvas text is set before the
  store atoms during a load; a *same-atom-count* reload has a window where a persist
  tick can write a draft pairing the new xyz with the previous file's regions/frozen
  (the atom-count invariant is count-only, core §"invariant"). Steady state
  self-heals; a crash mid-window leaves a corrupt draft. **Status:** ☐
- **F2 — `selection_rules` un-round-trippable by the codec** (never emitted by
  `_sidecar_dict`, never read by `apply_to_structure`). Legacy field, low impact.
  **Status:** ☐

---

## Standing guardrails (apply to every open step)

- No structure/cell/kgrid read or write outside `ws.*` / the store / the module API.
- One k-grid render loop, in the module, forever (core Part II §14.1).
- Each step ends GREEN (its Check) before the next begins.
- Update this doc's ☐/◐/☑ as steps complete; if reality diverges, fix the doc first.
