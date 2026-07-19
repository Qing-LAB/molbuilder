# Web module map — the front-end + web-API modules, in layers

The single index of the **browser + web-blueprint** modules we designed: what each is,
its goal, public API, and the data structures it exchanges. Organized by layer —
**framework → special module → submodule → consumer** — plus the **server blueprints**.

> This is the JS/web complement to [`architecture.md`](../architecture.md) (which indexes
> the Python backend subsystems). Each module's own **provenance header** (`module ·
> role · used-by`, per [`code-conventions.md`](code-conventions.md) § 1) is the local
> truth; this map is the index. When they disagree, the header + code win — fix the map.

Paths are under `molbuilder/web/static/` (JS) or `molbuilder/web/blueprints/` (Python).

---

## 0. Layer definitions

| Layer | What it means | Rule |
|---|---|---|
| **Framework** | cross-cutting infrastructure every page depends on; no domain UI | load first; everything may depend on it |
| **Special module** | a concealed, self-contained feature package that owns a domain, mounts ONE namespace, and hides its internals | consumers call its public surface only |
| **Submodule** | an internal file of a special module (often `_`-prefixed); not called from outside its package | package-private |
| **Consumer** | a tab / panel / generator that *uses* the modules above; owns page glue + UI policy | never reaches past a module's door |
| **Server blueprint** | a Flask blueprint = the HTTP surface a module's client half talks to | one blueprint per concern |

---

## 1. Framework layer

| Module | Namespace | Goal | Public API |
|---|---|---|---|
| `lib/molbuilder-runtime.js` | `molbuilder.runtime` | Module **registry + ready-Promises** — solves the "modules race to attach to `window.molbuilder`" load-order problem (the init contract; see design.md). | `register(name, api)`, `whenReady(name) → Promise<api>` |
| `lib/constants.js` | `molbuilder.constants` | Single source of truth for **sessionStorage keys + custom-event names** shared across modules (e.g. `SS_WORKSPACE`, `SS_DIR`, `SS_FILE`). | constant fields |

---

## 2. Special modules

### 2.1 Projects sidebar — `molbuilder.projects` (file access)
**Goal:** the ONE concealed file-access + sidebar module. Two layers: format-blind **byte
I/O** and format-aware **structure doors**. Entry: `lib/projects-sidebar.js`.
Contract: [`projects-sidebar.md`](projects-sidebar.md), [`structure-load-save-contract.md`](structure-load-save-contract.md).
**Public API:** `projects.readFile/writeFile/readRange/listDir`, `projects.parser.openMolecule/saveMolecule`, `projects.onCommit/onChange`, `projects.getCurrentDir/getCurrentFile/refresh`.
**Loading:** ES modules (`<script type="module">`).

### 2.2 Workspace persistence — `molbuilder.workspace` (state timeline)
**Goal:** session-state persistence — undo/retract history + tab-switch survival. Holds
NO data model; writes format-blind bytes. Contract: [`workspace-contract.md`](workspace-contract.md) §4.
**Public API:** `workspace.persist(sessionBytes, snapshotBlob, identity)`, `readState(identity)`, `pruneStatesAbove(id, index)`, `workspaceId()`, `readPersistedSnapshot()`, `mountRestoreTarget()`, `useNamespace(owner)`, `onPersistError(fn)`.
**Server:** `blueprints/state_timeline.py` → `/api/state-timeline/{write,read,prune}`.

### 2.3 MolView — `molbuilder.molview` (concealed 3-D viewer + data model)
**Goal:** the embeddable structure viewer that owns the in-memory data model, render
pipeline, and persistence wiring. Contract: [`molview-module.md`](molview-module.md).
**Public API:** `molview.mount(host, workspace, {mode, owner}) → handle`; `molview.data`
(the model: `installMolecule`, `exportFile`, `markSaved`, `save`/`load`/`undo`,
`getStructure/getSelection/getElements/...`, `generate`, `applyOp`, frames API).
**Loading:** IIFE + global-mount; `molview.data` also `require()`-able for node tests.

### 2.4 Selection — `molbuilder.selection` (atom selection)
**Goal:** the atom-selection store + panel + viewer adapter (single/preview→commit cursor).
Contract: [`selection.md`](selection.md), atom-annotations.
**Public API:** `selection.store` (mode/filters/combinator/isolate/selection), `selection.mountPanel(...)`, `selection.measurements.{xyz,distance,angle}`, `selection.viewerAdapter`.
> **Cross-module note:** the store *implementation* lives in `lib/molview/_selection-store-impl.js` but mounts on `molbuilder.selection` — MolView owns the code, selection owns the namespace.

### 2.5 Inspector framework — `molbuilder.inspectors` (Results file-type dispatch)
**Goal:** the seam that maps a sidebar-selected file to the right file-type handler on the
Results tab. A FRAMEWORK (dispatch), not a content parser. Contract: results-tab.
**Public API:** `inspectors.register(handler)`, `inspectors.pick(file) → handler`, `createDefaultContext()` (injects `ctx.readFile/readRange/writeFile` → `projects.*`).
**Handlers:** structure · source · markdown · trajectory · spectra inspectors.

### 2.6 mol-viewer-embed — `molbuilder.viewer` (the 3Dmol seal)
**Goal:** the standard embeddable 3-D viewer (3Dmol wrapper) that MolView composes over.
`viewer.embed(host, opts) → handle`.

### 2.7 VibrationView — `lib/vibrationview/` (normal-mode animation)
**Goal:** a concealed normal-mode animation package — SIBLING of MolView (not part of it),
used by the spectra inspector. Files: `vibrationview.js` + `mode-math.js`.
(Task #51: give it its own complete 3Dmol seal, fully separate from MolView.)

---

## 3. Submodules (package-private, grouped by their special module)

### MolView (`lib/molview/`)
| File | Role |
|---|---|
| `data-model.js` | the in-memory model + public `molview.data` surface (the package hub) |
| `mount.js` | `molview.mount` — assembles viewer + panel + render + overlays into ONE handle |
| `render.js` | the render loop — "molview owns the render" (§14): handle + workspace → draw |
| `view-controls.js` | the display-toggle control bar (axes/labels/cell/isolate) |
| `measurement-overlay.js` | paints the geometry readout (§15) |
| `_canvas-state-impl.js` | the structure canvas store (`molview._canvasState`) — what's loaded + dirty/source |
| `_selection-store-impl.js` | the selection store impl (mounts on `molbuilder.selection`) |
| `_state-timeline-impl.js` | **save/retract/undo mechanics** — the state-timeline submodule (this session) |
| `_atom-channels.js` | per-atom channel model (L1) — the annotations layer |
| `_atom-index.js` | 0-based-internal ↔ 1-based-user index conversion (L1) |
| `_frame-series.js` | the trajectory frame series (native movie) |

### Projects (`lib/projects/`) — ES modules
| File | Role |
|---|---|
| `state.js` | shared selection state + Inquire API; `readFile/writeFile/listDir` |
| `api.js` | pure HTTP wrappers for `/api/files/*` + `/api/projects/*` (builds `max_bytes=` etc.) |
| `parser.js` | the format-aware doors: `openMolecule` / `saveMolecule` |
| `list.js` | breadcrumb + directory listing + per-entry buttons |
| `preview.js` | file preview modal (view + edit + save) |
| `dialogs.js` | modal dialogs for sidebar mutations |
| `mutation-bar.js` | header-bar wiring (New project / New folder / Upload) |
| `checkpoint.js` | run-history panel (git checkpoints) inside the sidebar |

### Workspace (`lib/workspace/`)
| File | Role |
|---|---|
| `dispatcher.js` | `molbuilder.workspace` — persist/restore transport + identity + error surface |
| `snapshot-io.js` | `molbuilder.workspaceSnapshot` — the sole sessionStorage mirror read/write owner |

### Selection (`lib/selection/`)
| File | Role |
|---|---|
| `viewer-adapter.js` | declarative consumer of the store on the embedded viewer |
| `mount-panel.js` | the fused molview+selection panel composition |
| `measurements.js` | pure xyz/distance/angle helpers |
| `../selection-panel.js` | the DOM panel that renders the store |

### Inspectors (`lib/inspectors/`)
`registry.js` (the seam) · `structure.js` · `source.js` · `markdown.js` · `trajectory.js` · `spectra.js` (per-type handlers).

---

## 4. Consumers (tabs / generators — use the modules above; not framework)

| Consumer | Uses |
|---|---|
| `viewer.js` (structure-optimization tab) | molview.mount, projects.parser, workspace |
| `modify/selection-bootstrap.js` | molview.mount + selection.mountPanel + projects.parser |
| `lib/spectra/core.js` | molview (read-only inspect) + VibrationView + Plotly |
| `lib/transport/core.js` | molview + projects.parser + Generate POST from `molview.data.getFrozen/getRegions` |
| `lib/trajectory/core.js` | molview-with-frames + Plotly traces |
| `results/viewer.js` + `lib/results/file-picker.js` | inspectors.registry dispatch + projects.listDir |
| `lib/structure/{page,save,save-dialog,warning-modal,file,smiles,dna,rna,peptide,name}.js` | Modify-tab Source panels → `molview.data.installMolecule` / `projects.parser.saveMolecule` |
| `lib/system-load-monitor.js` | `/api/system-load` strip |

---

## 5. Server blueprints (`molbuilder/web/blueprints/`)

| Blueprint | Goal | Key routes |
|---|---|---|
| `build.py` | structure construction + emitters | `/api/build/load`, `/api/build/fdf`, `/api/build/pyscf`, `/api/structure/resolve-cell` |
| `files.py` | server file explorer for the sidebar | `/api/files/{read,write,list,read_range,stat,mkdir,move,copy,delete,roots}` |
| `selection.py` | rule evaluator + atoms + sidecar I/O (L2) | `/api/selection/{eval,atoms}` |
| `modify.py` | per-atom edit ops | `/api/modify/<op>` |
| `results.py` | unified post-merge inspector page + bundle | `/api/results/*` |
| `spectra.py` | harmonic freqs + Raman + per-mode ES | `/api/spectra/*` |
| `transport.py` | transport calculation | `/api/transport/*` |
| `state_timeline.py` | the workspace state timeline | `/api/state-timeline/{write,read,prune}` |
| `checkpoint.py` | run-history git checkpoints | `/api/checkpoint/*` |
| `system_load.py` | host load widget | `/api/system-load` |

---

## 6. Data structures (the shapes exchanged between modules)

| Shape | Where | Fields |
|---|---|---|
| **Session snapshot** | workspace-contract §4.1; written by `molview.data` serialise | `{ v:1, state: { structure, source, dirty, last_save_to, selection, view, state_index } }` |
| **Structure / scratch blob** | structure-load-save-contract; `exportFile()` | `{ xyz, sidecar }` |
| **Sidecar** (`.molstruct.json`) | `molbuilder/sidecars/molstruct.py` (schema v6); atom-annotations.md | `regions, frozen_atoms, cell, cell_origin, axis_kind, vacuum, annotations, n_atoms_total, structure_hash` |
| **Atom row** | web-api.md §6.2 | `{ index, element, atom_name?, residue_name?, chain_id?, is_frozen, regions }` |
| **Selection state** | molview-module §12 | `{ mode, filters, combinator, isolate, indices, pickOrder }` |
| **WorkspacePayload** | the `/api/build/load` parse result installed by `installMolecule` | `{ xyz, source_format, title, n_atoms, elements, atom_names, residue_ids/names, chain_ids, periodicity, annotations, atoms }` |
| **State-timeline file** | `state_timeline.py` | on-disk `<workspace_id>.<state_index>.wc.json` (opaque snapshot bytes) |
| **Identity** | timeline / workspace | `{ workspace_id, state_index }` |

---

## 7. Loading model (and the ES-module question)

**Today it's a hybrid, no bundler — Flask serves files raw:**
- `lib/projects/*` (9 files) are **ES modules** (`import`/`export`), loaded via `<script type="module">`.
- Everything else (~60 files) is **IIFE + global-mount** on `window.molbuilder.*`, loaded via plain `<script>` in a hand-ordered list; ~46 of these *also* `module.exports` so node-driven unit tests can `require()` them.

**Should the IIFE modules become ES modules? (advantages vs cost)**

*Advantages:* explicit `import`/`export` makes the "used-by" graph real (the provenance
header's USED-BY becomes machine-checked), kills the global-namespace soup + the
load-order landmines (the `_lazyResolve` / runtime-registry workarounds exist only because
of it), removes the dual IIFE-**and**-`module.exports` boilerplate, and lets node tests
`import` the real module instead of a shim.

*Cost:* every `<script>` include across the templates must become `type="module"` (order
no longer matters, but the tags change), every global mount (`window.molbuilder.X = …`)
must become an `export` + the consumers must `import` it (a large mechanical sweep across
~60 files + their tabs), and the node tests that `require()`/`vm.runInContext` these files
must move to `import` or a loader. No bundler is needed (native ESM works), but the sweep
is broad and touches the most-used files.

*Assessment:* the projects package proves native ESM works here with no build step, and
the benefits (kill load-order fragility + make dependencies explicit) directly serve the
provenance goal. But it's a **large, all-at-once-per-consumer** migration — a module can't
be half-ESM/half-global for its consumers. **Recommendation: incremental, package-by-package**
(next natural candidate: the `molview/` package, which is already internally cohesive),
each package converted with its consumers in one commit — not a big-bang. Not urgent; do it
when a package is next opened for other work.
