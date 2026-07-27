# Tabs — how the pages compose the modules

**Role:** contract
**Domain:** web
**Companions:** [`molview.md`](?doc=web/molview.md) — the 3D viewer every
build/edit tab mounts; [`projects.md`](?doc=web/projects.md) — the sidebar file
browser + the load/save doors; [`form-schema.md`](?doc=web/form-schema.md) — the
engine-option forms; [`spectra.md`](?doc=web/spectra.md) and
[`results.md`](?doc=web/results.md) — the two tabs with their own docs;
[`web-api.md`](?doc=web/web-api.md) — every route named below.

The reusable modules each have their own doc. **This doc is about the *pages*** —
the handful of tabs a user actually clicks between — and how each one is a thin
controller that *wires the modules together*. A tab owns almost no logic of its
own: it mounts MolView, drops in a schema-built form, subscribes to the projects
sidebar, and routes button clicks to the server. The interesting, reusable parts
live in the module docs; here we map **which tab wires which modules, and the
three protocols that cut across all of them**.

## 1. The tab roster and the shared shell

There are **six tabs**, and their order is defined in exactly one place — the
`TABS` list in `tabs.py`. In canonical order:

| Tab | Path | What it's for | Own doc |
|---|---|---|---|
| **Molbuilder** | `/molbuilder` | build / edit / assemble a structure | this doc § 2 |
| **Structure optimization** | `/structure-optimization` | generate a SIESTA/PySCF relaxation script | this doc § 3 |
| **Spectrum** | `/spectrum-calculation` | compute a Raman spectrum | [`spectra.md`](?doc=web/spectra.md) |
| **Transport** | `/transport-calculation` | generate a TranSIESTA device script | this doc § 4 |
| **Results** | `/results` | open a finished calculation | [`results.md`](?doc=web/results.md) |
| **Documents** | `/documents` | read the in-app docs (this page!) | this doc § 5 |

**Molbuilder is the landing tab** — a bare `/` redirects to whatever is first in
`TABS`. The tab nav bar itself is injected into every page from that one list (a
Flask context processor hands `tabs` to every template, and the shared
`_app_header.html` renders them), so adding or reordering a tab is a one-line
edit to `TABS`, never a template hunt.

Every build/edit tab also mounts the **projects sidebar**
([`projects.md`](?doc=web/projects.md)) down the left — the file browser you pick
a structure from. The one exception is Documents, which needs no files of yours.

## 2. Molbuilder — build, edit, assemble

This is the workbench. Its page is a **"Init structure" source bar** across the
top, a **fused MolView card** in the middle, and a row of **Modify op-tabs**
below. The controller code is deliberately thin glue:

- **The 3D viewer is mounted by `selection-bootstrap.js`**, not by the tab's main
  `viewer.js`. One call — `mount(host, workspace, {mode: "modify", owner:
  "modify"})` — and the **MolView module builds the entire fused card itself**:
  the viewer, the selection panel, the view toggles, the cell panel. The tab
  hand-builds no viewer chrome. `selection-bootstrap.js` does only page glue:
  mount the module, turn a sidebar file-pick into a *candidate* (you still click
  "Load" to commit — a stray click can't swap your structure mid-edit), and
  respect a restored canvas so a reload doesn't clobber it.
- **`modify/viewer.js` wires the Modify op-tabs** — Atom (add/delete), Transform
  (translate/center/rotate/orient), Junction (electrode), plus the state timeline
  (save-state / undo). Crucially, applying an op does **not** hand-roll a fetch:
  it calls `molview.data.applyOp(op)` and lets the module talk to the server
  (`/api/modify/<op>`). The only route this controller fetches directly is
  `GET /api/modify/meta`, to fill the electrode dropdowns.
- **`modify/periodicity.js`** is the Cell op-tab — vacuum, per-axis periodicity,
  the unit cell, the cell origin — and commits through
  `molview.data.commitPeriodicity` (the server re-resolves the effective cell).

### Creating a structure — the in-gate

The "Init structure" bar is how a molecule *first appears* on the canvas. Every
source, no matter how different, funnels through **one gate**:
`structurePage.loadIntoCanvas(structure, source)` → `molview.data.installMolecule`.
That single funnel runs the dirty-canvas check ("you have unsaved edits —
replace them?") so no source can bypass it.

The sources, and what each produces (all POST `/api/build/molecule` with
`{kind, input}` unless noted):

| Source | `kind` | Backend | Notes |
|---|---|---|---|
| **SMILES** | `smiles` | RDKit first, **OpenBabel fallback** | the fallback rescues big/awkward molecules RDKit chokes on; the response says which backend won |
| **By name** | `name` | PubChem → SMILES → (same fallback) | type "aspirin", get a structure |
| **File upload** | — (`/api/build/load`) | loads your `.xyz`/`.pdb` text as-is | |
| **DNA** | `dna` | 3DNA (X3DNA) / AmberTools / RDKit | B/A/Z forms, single strand or duplex, optional clash relief |
| **RNA** | `rna` | A-form canonical | |
| **Peptide** | `peptide` | AmberTools `tleap` | extended chain from a sequence |

Loading a project file from the sidebar is just another way in — it reads the
`.xyz` plus its `.molstruct.json` sidecar through the same gate.

## 3. Structure optimization — generate a relaxation script

`/structure-optimization` is a **three-card generator**: inspect a structure,
auto-detect its chemistry, generate an input script. It mounts:

- a **read-only MolView** (`mode: "readonly", owner: "structure-opt"`) — you look,
  you don't edit;
- two **schema-built forms** (one SIESTA, one PySCF), rendered by form-schema from
  `GET /api/build/schema/siesta` and `.../pyscf`;
- the **detection chip** that shows the auto-detected charge/spin/method.

The flow: **inspect** → **Auto-detect** (`POST /api/structure/analyze` pre-fills
both sub-forms) → **Generate** (renders the script *text* — `POST /api/build/fdf`
for SIESTA, `POST /api/build/pyscf` for PySCF — nothing is written to disk yet) →
**Save to current dir**. There's no single `/api/build/render`; the two engines
have their own render routes. Live validation runs against `/api/build/preflight`
as you edit.

Saving a SIESTA run is a **four-step pipeline**, because a SIESTA job is more than
one file: save the geometry (`.xyz`) → install the pseudopotentials (`.psml`, via
`/api/siesta/install-pseudos`) → drop the run wrapper (`.run.sh`, via
`/api/run/install-wrapper`) → rewrite the pseudo paths in the `.fdf`. PySCF is
simpler: save the `.py`, install the wrapper. Both save through the projects
module's save door (§6), never a hand-rolled write.

## 4. Transport — generate a TranSIESTA device script

> **Status: working, with named gaps.** The `/transport-calculation` page's own
> code still carries stale "placeholder / Generate disabled" comments from an
> earlier phase — **ignore them**. The tab actually generates a real device
> script today. What it does *not* yet do is listed below.

**What works now** — a four-card workflow (Inspect / Analyze / Parameters /
Generate):

- a **MolView mount** (`mode: "modify", owner: "transport"`) — the same concealed
  viewer, sourcing the frozen/region labels off the model at generate time
  ("what you see is what generates");
- a **schema form** from `GET /api/transport/schema`, with the same session
  persistence and dirty-gating as the other tabs;
- **auto-analyze** chemistry on load (`POST /api/structure/analyze` + detection
  chip);
- **Generate → `POST /api/transport/render`**, which dispatches through the engine
  registry and renders a real TranSIESTA device `.fdf` (the TranSIESTA engine's
  `render_script` is implemented for the zero-bias scope). You get an issues
  panel, a script preview, Copy, and a download.

**What's still deferred** (be honest about these when you point a user here):

- **No in-app "save to project"** — only Copy and a blob download. The other
  generator tabs write files; this one doesn't yet.
- **Electrode `.TSHS` generation** is a manual, documented workflow, not wired.
- **Multi-bias scans** are deferred.
- **Reading results back** (T(E) plots) is deferred — the parser raises
  `NotImplementedError`.

## 5. Documents — the in-app reader (this page)

`/documents` is the simplest tab: a doc list on the left, a render pane on the
right, no sidebar. It lists every `docs/*.md` (`GET /api/docs/list`), fetches one
doc's raw Markdown (`GET /api/docs/read?path=<rel>`), and renders it through the
shared `markdown-render` primitive (Marked + DOMPurify, with Mermaid drawn in
place).

**This tab is why the migration docs' links work.** Every internal link in these
docs is written `?doc=<path-relative-to-docs>` — and *that* is exactly the
query parameter this page reads: opening a doc writes `?doc=…` back into the URL,
and loading a URL with `?doc=…` opens that doc. (`<path>` must be one the list
returned; the server resolves it under the `docs/` root with traversal and
non-`.md` guards.) A raw `.md` href would 404 — the document is served *through
this tab*, via `?doc=`, never as a static file. That's the rule the migration's
link convention (R7) encodes.

## 6. Save flow — the out-gate

Every tab that saves a structure uses **one path out**, mirroring the one gate in
(§2):

1. You click **Save to project**; a dialog asks for a name (no default — you name
   it deliberately). The target folder is **always the sidebar's current project
   dir**.
2. The model **serialises itself** — `molview.data.exportFile()` — so what's saved
   is exactly what's on screen ("viewer is truth"). The tab does *not* scan the
   structure for regions or frozen atoms; the model already knows them.
3. That goes to **`POST /api/structure/save`** with `{path, blob, overwrite}`.
   **The server writes the pair** — `<stem>.xyz` + `<stem>.molstruct.json` — and
   stamps the sidecar's schema version and structure hash itself. The browser
   never authors the sidecar (a past bug wrote a sidecar the load door then
   rejected; the server-writes-it rule closed it).
4. If the file exists and you didn't pass `overwrite`, the server replies
   **`409 {needsOverwrite: true}`**; the tab pops an overwrite confirm and retries
   with `overwrite: true`. On success the dirty bit clears and the sidebar
   refreshes.

`saveMolecule` itself is documented in [`projects.md`](?doc=web/projects.md);
here we've described the *page-level* save UX that calls it.

## 7. Data coherence — one rule, enforced server-side

The invariant that keeps a tab's form and its structure from ever disagreeing:

> **What is in the viewer at generate/save time *is* the input.** The frozen-atom
> and region labels travel in the request body and are applied to the structure
> verbatim — and **every label index must be a valid atom of the current
> structure**, or the server rejects it.

This is enforced where it can't be bypassed — on the server:

- `apply_labels_to_struct` treats the in-body `frozen_atoms`/`regions` as
  authoritative but validates each index (`0 ≤ idx < n_atoms`) and raises a
  visible warning otherwise. Used by the transport and build renders.
- `struct_from_body` honours a per-atom metadata array **only if its length
  equals the atom count** — a malformed array can't corrupt the structure; the
  default is kept.
- On any atom-count change (an add or delete), the client **clears the selection**
  — so no stale index can point at an atom that no longer exists.

There's a **chemistry** half too: a single analyzer (`analyze_structure`) is the
one source of truth for open- vs closed-shell, every generator tab shows its
verdict through the same `detection-chip`, and a validator finding attaches to the
workflow card it belongs to. (The *visual* coherence rules — palette, role
vocabulary — live in [`ui-contract.md`](?doc=web/ui-contract.md), not here.)

## 8. Where the tab controllers stand — ESM status

The design goal is concealed, independently reusable ES modules. The **modules**
are largely there; the **tab controllers that consume them** are half-migrated:

| Controller | ESM today |
|---|---|
| `modify/selection-bootstrap.js` | **yes** — a real ES module |
| `modify/viewer.js`, `structure-optimization/viewer.js`, `transport/core.js` | **hybrid** — a top-level `import` of MolView, but a classic IIFE body that still publishes globals; loaded as `type="module"` |
| `modify/structure/*.js` (the generators), `documents/page.js` | **no** — classic global scripts |
| `modify/periodicity.js` | classic body, but served with a `type="module"` tag (a harmless mismatch) |

So the three MolView-consuming controllers have one foot in ESM (they `import`
the viewer) while their own bodies are still legacy globals, and the
structure-generation set plus the Documents controller are fully classic.
Finishing these off is part of the "remaining classic modules" ESM workstream
(`roadmap.md § 3`).

## 9. Test map

- `test_tab_routes.py` — every canonical tab path renders + the nav contract
  (paths read from `TABS`).
- `test_docs_tab.py` — the Documents tab: the list groups every `docs/*.md`,
  `read` returns text + title, and the path-safety gate rejects traversal /
  non-`.md` / outside-`docs`.
- `test_transport_render_endpoint.py`, `test_transport_transiesta.py`,
  `test_transport_generate_e2e.py`, `test_transport_preflight.py` — the transport
  render path, the engine, end-to-end generate, preflight.
- `test_structure_save_endpoint.py`, `test_structure_save_js.py`,
  `test_structure_save_dialog_js.py` — the save flow + its dialog.
- `test_smiles_fallback.py`, `test_structure_smiles_js.py` — the SMILES
  generator + the OpenBabel fallback.
- `test_modify.py` — the Modify ops.
- `test_molbuilder_e2e.py` — the Molbuilder tab end to end.
