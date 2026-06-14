# Workspace state — unified design (2026-06-07 audit + 2026-06-09 implementation + 2026-06-13 Phase 9 close)

> **Status: Phases 1–9 SHIPPED.** This document originally
> landed as a design proposal after three rounds of selection-
> list bugs ([cd9655e], [bebc73d], follow-ups).  Each bug was a
> symptom; the root cause was **architectural** — the client had
> four parallel stores for one conceptual entity, the server
> returned the same entity in four different wire shapes, and
> persistence was scattered across three sessionStorage keys.
>
> Phases 1–8 of the migration table in § 6 shipped 2026-06-07
> through 2026-06-09: server-side `WorkspacePayload`
> (`molbuilder/web/blueprints/_shared.py::workspace_payload`),
> client-side `window.molbuilder.workspace` dispatcher
> (`lib/workspace/dispatcher.js`), the `ws.*` public API surface,
> and consumer migration of all 6 known call sites.  The
> contract is now sole-source-of-truth in
> [`workspace-contract.md`](workspace-contract.md); a
> contract-compliance test enforces zero legacy consumers.
>
> Phase 9 (retirement of the legacy public globals
> `window.molbuilder.structureCanvas` + `window.molbuilder.selection.store`)
> shipped 2026-06-13 in two commits: 9A (`23e4e80`) moved the
> selection store to `lib/workspace/_selection-store-impl.js` and
> dropped the singleton self-mount; 9B (`f9355bc`) did the same
> for canvas-state at `lib/workspace/_canvas-state-impl.js`.  The
> dispatcher owns both singletons internally; production
> templates no longer mount the legacy globals.  A test-only
> escape hatch (the dispatcher honours a pre-mounted legacy
> global if a Node harness installs one before the dispatcher
> loads) keeps `tests/test_workspace_dispatcher_js.py` working
> without a wholesale rewrite.  See § 6 row 9 for the full
> rationale.

---

## 1. What "workspace state" is

One conceptual entity flows through every UI interaction in the
Molbuilder tab:

```
                                       ┌──────────────────────┐
   sidebar pick   ──┐                  │ workspace            │
   generator click ─┼──► load/build ──►│   structure (atoms,  │──► viewer renders
   file upload     ─┘                  │   xyz, lattice)      │      ▲
                                       │   source (kind/file/ │      │
   modifier op    ─────► mutate ──────►│   generator_input)   │──► selection panel
                                       │   dirty / last_save  │      │
   sidebar dblclick ───► reload ──────►│   selection state    │──► viewer-adapter
                                       │   view_state         │      │
   save           ─────► persist ─────►│ (camera, style, …)   │──► save / generate buttons
                                       └──────────────────────┘
```

The boxed object is the workspace.  Every panel, every overlay,
every persistence mirror is a *view of* the same workspace.  This
is the architectural invariant we want.

Today the same data lives in **four** stores, with **three**
sessionStorage mirrors, with **four** server response shapes for
the load/build/modify endpoints.  Sections 2 and 3 enumerate the
duplication; § 4 proposes a single object that supersedes it; § 5
describes the wire protocol; § 6 is the migration plan.

---

## 2. What lives where today (audit)

### 2.1 Client-side stores

| Store | Path | Owns | Persisted to |
|---|---|---|---|
| canvas-state | `static/lib/workspace/_canvas-state-impl.js` | `{source_format, text, source: {kind, file, generator_input}, dirty, last_save_to}` | `sessionStorage["molbuilder.structure_canvas"]` |
| modify viewer state IIFE | `static/modify/viewer.js` (module-scope `state`) | `{xyz, elements, atom_names, residue_ids, residue_names, chain_ids, title, n_atoms, positions, history, inFlight, _inFlightAbort}` | `sessionStorage["modify-state"]` (schema v1) |
| selection store | `static/lib/workspace/_selection-store-impl.js` | `{sourceFile, atoms, selection, mode, filters, combinator, loading, error}` | none directly — restored via modify-viewer-state's `atoms` field added 2026-06-07 |
| 3Dmol embed | `window.molbuilder.modify.handle` | the rendered model (xyz bytes + camera + style + axes + labels) | camera saved by modify viewer state |

Pieces of the workspace duplicated across stores:

| Field | Lives in |
|---|---|
| xyz / text bytes | canvas-state.text + modify state.xyz + 3Dmol embed |
| elements | modify state.elements + selection store.atoms[].element |
| atom_names, residue_ids, residue_names, chain_ids | modify state.* + selection store.atoms[].* |
| sourceFile path | canvas-state.source.file + selection store.sourceFile |
| dirty bit | canvas-state only (but modify-state's `shouldPersistAtoms` heuristic reads it; see [cd9655e]) |
| selection indices | selection store.selection (canonical), modify state.selected (snapshot only at save) |

Every modifier op currently has to update **three** stores
(`applyStructure` writes modify state + 3Dmol embed; explicit
`cs.replaceContent` writes canvas-state; my [cd9655e]
consolidation pushed adoptAtoms into applyStructure for the
selection store).  Every generator path has to update **three**
stores (loadIntoCanvas writes canvas-state; viewerLoader →
applyStructure writes modify state + embed + selection store).
Every sidebar pick has to update **four** stores (commitFile
writes canvas-state + viewerLoader + adoptSession for the
sourceFile + the embed).

When a new path is added, the author has to remember each write.
When one is missed, the data goes silently out of sync.  That is
exactly what shipped behind the user's repeated complaints.

### 2.2 Server-side response shapes

The Python `Structure` dataclass is the single source of truth on
the server.  But it gets serialised into **four** different JSON
shapes depending on which endpoint emits it:

| Endpoint | Shape | Helper |
|---|---|---|
| `/api/build/molecule` | `{ok, xyz, pdb, n_atoms, n_residues, summary, title, elements, atoms, backend_used, add_hydrogens_mode, issues}` | hand-rolled `jsonify` |
| `/api/build/load` | `{ok, xyz, pdb, n_atoms, n_residues, summary, title, elements, atom_names, residue_ids, residue_names, chain_ids, atoms, source_format}` | hand-rolled `jsonify` |
| `/api/modify/*` | `{ok, xyz, elements, atom_names, residue_ids, residue_names, chain_ids, n_atoms, n_residues, title, atoms, issues}` | `_shared.ok_structure_response` → `structure_to_dict` |
| `/api/selection/atoms` | `{ok, n_atoms, atoms}` (no xyz; atoms only) | hand-rolled `jsonify` |

The `atoms` key was added to `/api/modify/*` in [BOMB-0,
2026-06-07] via `_shared.atoms_list`.  `/api/build/load` and
`/api/build/molecule` did **not** include it until [cd9655e].
That gap is what made `applyStructure(r) → adoptAtoms(r.atoms)`
silently no-op for every generator path even after the
modifier-op fix.

The shapes also drift in cosmetic ways:
- `summary` is present on build endpoints but not modify endpoints.
- `source_format` only on `/api/build/load`.
- `pdb` only on build endpoints.
- `backend_used` / `add_hydrogens_mode` only on `/api/build/molecule`.

Each blueprint built its own response in isolation.  No
single helper enforces "if it's a Structure, this is the
shape."  Adding a new field that every consumer needs (e.g.
`atoms`, `lattice`) requires editing every blueprint.

### 2.3 Persistence boundaries

| Key | Owner | Schema | What it stores |
|---|---|---|---|
| `molbuilder.structure_canvas` | canvas-state | implicit v1 | text + source + dirty + last_save_to |
| `modify-state` | modify viewer | explicit v1 | xyz + metadata + selected + atoms (added [cd9655e]) + camera + chrome |
| `molbuilder.panelMode` | selection-panel | scalar | "click" / "filter" |
| `molbuilder.current_file` + `molbuilder.current_dir` | projects sidebar | scalar | last sidebar pick |

Four keys, three schema-versioning policies, no atomic restore
across all of them.  Restore order matters:
1. canvas-state restores from its own key on `_ensureInit`.
2. modify viewer state restores via `restoreModifyState` on `DOMContentLoaded`.
3. Selection store has no own persistence; gets atoms from modify state.
4. Sidebar state restores independently.

Order-of-restore bugs have shipped twice ([BOMB-2 dirty-bit
reset], [cd9655e dirty-gate]) — both because canvas-state's
sessionStorage mirror and modify-state's sessionStorage mirror
told slightly different stories on the same restore event.

### 2.4 Operation paths that have to remember which stores to write

| Op | Stores it has to update |
|---|---|
| Generator (DNA / SMILES / …) | canvas-state (`loadIntoCanvas`); modify state + embed + selection store (via `loadStructureText` → `applyStructure`) |
| Sidebar Load button | canvas-state (`loadIntoCanvas`); modify state + embed + selection store; sourceFile via `adoptSession` |
| Modifier op (Delete / …) | modify state + embed + selection store (via `applyStructure` since [cd9655e]); canvas-state via `cs.replaceContent` |
| File upload | same as generator path |
| Save | canvas-state's `markSaved` only; modify state's sessionStorage updated on next pagehide |
| Undo | modify state restored from `state.history`; embed re-rendered; canvas-state, selection store NOT touched |
| Filter eval | selection store only |

The undo path is a different kind of bug waiting to happen — it
bypasses canvas-state entirely, so an undo after Save followed by
a Load doesn't fire the warning modal.

---

## 3. The fundamental design flaw, named

> **There is no single source of truth for "the workspace" on the
> client.**

Design.md § "1. The dataclass is the lingua franca" enshrines this
principle on the server (every builder yields a `Structure`,
every modifier takes/returns a `Structure`).  The principle
applies to client-side state too, but was never extended there.
As more capabilities accreted (canvas-state for save tracking,
selection store for atom labels, modify state IIFE for history)
each got its own slice of the workspace and its own persistence
mirror.

The current code is correct *when every code path remembers to
update every store*.  Bugs surface every time a new code path is
added and forgets one of the writes:

- Generators forgot the selection store → empty atom list bug.
- restoreModifyState's adoptSession re-fetched disk → stale
  post-op atom list bug.
- _load_file test helper bypassed canvas-state → dirty-gate
  silently false bug.

Each fix bandaged the symptom.  None addressed the design.

---

## 4. The unified workspace model

### 4.1 One client-side object

```ts
type Workspace = {
  // ─── Structure: server-canonical, mirrors molbuilder.Structure ────
  structure: {
    text:          string;                 // XYZ or PDB bytes
    source_format: "xyz" | "pdb";
    title:        string;
    n_atoms:      number;
    atoms:        Atom[];                  // canonical per-atom shape
    lattice:      number[][] | null;       // 3x3 or null
  } | null;

  // ─── Source / provenance ──────────────────────────────────────────
  source: {
    kind: "file" | "smiles" | "name" | "dna" | "rna"
        | "peptide" | "blank";
    file: string | null;                   // disk path when kind="file"
    generator_input: object | null;
  };

  // ─── Save tracking ────────────────────────────────────────────────
  dirty:        boolean;
  last_save_to: string | null;

  // ─── Selection (atom-level) ──────────────────────────────────────
  selection: {
    indices:    number[];                  // sorted-ascending
    mode:       "click" | "filter";
    filters:    Filter[];
    combinator: "or" | "and";
  };

  // ─── Viewer chrome ───────────────────────────────────────────────
  view_state: {
    camera?: object;                       // opaque to the dispatcher
    style?:  object;
    axes?:   boolean;
    labels?: boolean;
  } | null;

  // ─── Op state (transient — never persisted) ──────────────────────
  loading:  boolean;
  inFlight: boolean;
  error:    string | null;
  history:  WorkspaceSnapshot[];           // undo stack
}

type Atom = {
  index:        number;
  element:      string;
  atom_name?:   string;                    // PDB only
  residue_id?:  number;
  residue_name?: string;
  chain_id?:    string;
  regions:      string[];
  is_frozen:    boolean;
}
```

### 4.2 One public surface

```js
const ws = window.molbuilder.workspace;

// ── Subscriptions ────────────────────────────────────────────────
ws.subscribe(fn) -> unsubscribe;          // fires once on subscribe + on every change

// ── Reads (defensive copies) ─────────────────────────────────────
ws.getState();
ws.getStructure();
ws.getSource();
ws.isDirty();
ws.isEmpty();
ws.getSelection();

// ── Operations: each does a server round-trip and atomically
//    replaces the relevant workspace slice ──────────────────────
ws.loadFromFile(path);                    // → /api/build/load
ws.generate(kind, input, opts);           // → /api/build/molecule
ws.applyOp(op, args);                     // → /api/modify/<op>
ws.applyPayload(payload, opts);           // atomic in-memory install (the
                                          //   pipeline that applyOp, the
                                          //   modify-tab's loadStructureText,
                                          //   and every load path runs at
                                          //   the end)
ws.save(opts);                            // delegates to structureSave.save
ws.discard();                             // wipes structure + selection
                                          //   UNCONDITIONALLY; caller must
                                          //   gate on warningModal first
ws.undo();                                // pops history; restores prior snapshot

// ── Persistence ──────────────────────────────────────────────────
ws.STORAGE_KEY;                           // "molbuilder.workspace.v1"
ws.readPersistedSnapshot();               // returns parsed snapshot or null

// ── Selection (purely local — no HTTP) ───────────────────────────
ws.selection.toggle(i);
ws.selection.set(indices);
ws.selection.add(indices);
ws.selection.remove(indices);
ws.selection.all();
ws.selection.invert();
ws.selection.clear();
ws.selection.setMode("click" | "filter");
ws.selection.setFilters(filters);
ws.selection.setCombinator("or" | "and");

// ── Selection eval + sidecar writes ──────────────────────────────
ws.selection.applyFilter();               // → /api/selection/eval
ws.selection.writeLabel(target, indices); // → /api/selection/save

// ── View chrome (camera / style / overlays) ──────────────────────
ws.view.applyState(patch);                // delegates to embed
ws.view.getState();
```

The public surface intentionally collapses canvas-state +
selection store + modify state IIFE.  The legacy module surfaces
(`window.molbuilder.structureCanvas`, `window.molbuilder.selection.store`,
the modify-tab state IIFE) become thin compatibility shims during
the migration window; once every consumer is moved over, the
shims are deleted.

### 4.3 One persistence key

```
sessionStorage["molbuilder.workspace.v1"] = JSON.stringify({
  v:           1,
  saved_at:    "2026-06-07T20:30:00Z",
  workspace:   <Workspace minus transient fields>,
})
```

The `loading` / `inFlight` / `error` fields are dropped at save
time.  History is dropped at save time (undo doesn't survive
navigation — same as today).

Schema-versioned: a bump invalidates older saves cleanly.  The
restore path is one atomic JSON.parse → state replacement →
notify; no inter-store coordination needed.

External-file-change handling becomes a single decision point:

```js
function restore(saved) {
  const useDisk =
       !saved.workspace.dirty
    && saved.workspace.source.kind === "file"
    && saved.workspace.source.file;
  if (!useDisk) {
    return atomicReplace(saved.workspace);  // memory is truth
  }
  return fetch(saved.workspace.source.file)
    .then(text => atomicReplace({
      ...saved.workspace,
      structure: parseFromText(text, saved.workspace.source.file),
    }));
}
```

No per-store "should I persist X?" heuristics.  The dirty-gate
becomes one if-statement at one place.

### 4.4 One server response shape

```py
@dataclass
class WorkspacePayload:
    text:          str                      # XYZ or PDB bytes
    source_format: str                      # "xyz" | "pdb"
    title:         str
    n_atoms:       int
    atoms:         List[Atom]               # canonical per-atom rows
    lattice:       Optional[List[List[float]]]
    issues:        List[Issue]
    extra:         Dict[str, Any] = field(default_factory=dict)
                                            # endpoint-specific add-ons
                                            # (backend_used, add_hydrogens_mode,
                                            #  source_format, etc.)

def workspace_payload(struct: Structure, **extra) -> Dict[str, Any]:
    """Single serialiser for every endpoint that returns a Structure.

    Replaces:
      - structure_to_dict           (_shared.py)
      - ok_structure_response       (_shared.py)
      - the hand-rolled jsonify blobs in build.py / modify.py /
        selection.py.

    Every endpoint that yields a Structure returns this exact shape,
    with op-specific data in ``extra``.  The wire contract is fixed
    and minimal; consumers never need to feature-detect missing
    keys.
    """
```

Adding a new field (`bonds`, `dipole`, etc.) is a one-line
extension to the dataclass and the helper, applied to every
endpoint at once.

`atoms_list(struct)` retires — its job is folded into
`workspace_payload`.

### 4.5 One per-op selection-state rule

Each operation class declares how selection survives the
structure swap.  This lives **in the dispatcher**, not in
ad-hoc opts threaded through `applyStructure`:

| Op class | Rule |
|---|---|
| `loadFromFile` | Reset to `[]` |
| `generate` | Reset to `[]` |
| `applyOp` (Delete / Add — atom count changes) | Remap via server's `selection_remap` (new field on `WorkspacePayload`); silently drop indices the server marks as removed |
| `applyOp` (Translate / Rotate / Orient / Centre — atom count unchanged) | Preserve as-is |
| `selection.applyFilter` | Replace with eval result |
| `selection.toggle/set/...` | The mutator's own semantics |
| `save` | Preserve |
| `undo` | Restore from snapshot |

`selection_remap` is a new server-side payload addition.  Wire
shape (Phase 3, finalised 2026-06-07): a flat list, where
``remap[old_index] == new_index`` or ``None`` when the atom was
removed.  Length equals the pre-op atom count.

```json
"selection_remap": [null, 0, 1]    // delete: O at index 0 removed
"selection_remap": [0, 1, 2]       // add: identity (new atom at index 3)
```

List rather than ``Dict[int, Optional[int]]``: JSON int-keys
round-trip to strings, forcing the client to parse keys back to
int; a flat list keeps the wire compact and gives the client an
implicit pre-op-size check via ``len()``.

Without `selection_remap`, the current code's naive "filter to
in-range" check silently drops the wrong indices when atom IDs
shift (the Delete-of-low-index bug noted in the audit but
deferred at [cd9655e]).  With it, the dispatcher can update
selection correctly without each modifier-op caller having to
reason about index shifts.

---

## 5. Wire protocol

### 5.1 Every Structure-returning endpoint returns `WorkspacePayload`

| Endpoint | Op semantics | `extra` keys |
|---|---|---|
| `/api/build/load` | Parse text → Structure | `pdb`, `source_format`, `n_residues`, `summary` |
| `/api/build/molecule` | Generate from input → Structure | `backend_used`, `add_hydrogens_mode`, `pdb`, `summary` |
| `/api/modify/<op>` | Mutate Structure → Structure | `selection_remap` (when applicable), `op`, `args` |
| `/api/selection/atoms` (legacy) | Atoms only, no text | **Active**.  Scheduled for deprecation in migration § 6 step 10 (currently deferred — selection store still calls it from `_fetchAtoms` / `setSourceFile` / `refreshAtoms`; the legacy `tests/test_pdb_workflow_integration.py` integration suite also exercises it).  Once Phase 9 folds the selection store into the dispatcher, this endpoint becomes deletable. |
| `/api/selection/eval` | Selection indices only | unchanged — selection-only endpoint, doesn't return a Structure |
| `/api/selection/save` | Sidecar writes | unchanged — selection-only |

`/api/selection/atoms` becomes redundant once `WorkspacePayload`
ships across the load + build + modify endpoints.  Removing it is
phase 5 of the migration; nothing prevents it from staying as a
back-compat wrapper indefinitely if needed.

### 5.2 Standard envelope

Already documented in `protocols/web-api.md` § 1.1 — the `ok`
envelope wraps every response.  `WorkspacePayload` slots into the
success branch:

```json
{
  "ok": true,
  "text":          "...",
  "source_format": "xyz",
  "title":         "...",
  "n_atoms":       3,
  "atoms":         [...],
  "lattice":       null,
  "issues":        [],
  "extra":         { "backend_used": "rdkit", ... }
}
```

---

## 6. Migration plan

Each step is an independently mergeable PR with regression tests.

| # | Step | Scope | Risk | Status |
|---|---|---|---|---|
| 1 | Add `_shared.workspace_payload(struct, **extra)` helper.  Replace `structure_to_dict` + `ok_structure_response` internally; keep them as 3-line shims so consumers don't break in the same PR. | server | low | **✅ shipped 2026-06-07.**  `workspace_payload(struct, extra=...)` is canonical.  `structure_to_dict` routes through it + emits canonical keys (`text`, `source_format`, `lattice`) alongside legacy aliases (`xyz`, `elements`, `atom_names`, …) for back-compat.  `ok_structure_response` sources `issues` from the workspace payload (single validate-pass).  23 assertions in `tests/test_shared.py`. |
| 2 | Migrate `/api/build/load` + `/api/build/molecule` + `/api/modify/*` to emit `WorkspacePayload`.  Existing client code keeps working — every old key it reads is still in the response. | server | low | **✅ shipped 2026-06-07.**  All three Structure-returning endpoint families now route through `ok_structure_response(struct, extra=…)`.  `structure_to_dict` accepts an `extra` dict and threads each key into BOTH the top level (back-compat for every existing JS consumer reading off the root) AND the canonical `extra` sub-dict (Phase 4+ workspace-dispatcher consumers).  `/api/build/load` carries `extra={pdb, summary, source_format}` (source_format overrides canonical XYZ default with parsed format).  `/api/build/molecule` carries `extra={pdb, summary, backend_used, add_hydrogens_mode}`.  `/api/modify/*` carries `extra={}` today (Phase 3 will add `selection_remap`).  Pinned by `tests/test_shared.py::TestStructureToDictExtraThreading` + `tests/test_web.py::{test_build_load_returns_workspace_payload, test_build_load_pdb_overrides_canonical_source_format, test_build_molecule_returns_workspace_payload, test_modify_delete_returns_workspace_payload}`. |
| 3 | Add `selection_remap` to `WorkspacePayload` for `/api/modify/delete` + `/api/modify/add_atom`.  Server-side implementation in `molbuilder/modify.py` (the index-shift map already exists internally via `_reindex_transport_metadata`). | server + tests | medium | **✅ shipped 2026-06-07.**  Two new public functions in `molbuilder/modify.py`: `compute_selection_remap_after_delete(struct, indices)` returns the flat list-shaped remap (None for removed atoms), `compute_selection_remap_after_add(struct)` returns the identity (emitted even when trivial so the Phase 4+ dispatcher's per-op rule table stays flat).  Wire shape finalised as **a list**, not the originally-proposed `Dict[int, Optional[int]]`, because JSON int-keys round-trip to strings — see § 4.5 for the rationale.  `/api/modify/delete` + `/api/modify/add_atom` emit `extra["selection_remap"]` via the Phase 2 helper.  Pinned by `tests/test_modify.py::{TestComputeSelectionRemapAfterDelete, TestComputeSelectionRemapAfterAdd}` (12 assertions covering middle-deletion shift, dedup, out-of-range tolerance, cross-check against `delete_atoms` result, identity contract for add) + `tests/test_web.py::{test_modify_delete_returns_selection_remap, test_modify_delete_selection_remap_handles_middle_deletion, test_modify_add_atom_returns_identity_selection_remap}`. |
| 4 | Add `window.molbuilder.workspace` dispatcher as a thin wrapper over the existing three stores.  Public surface: § 4.2.  Initial implementation delegates to the legacy stores; the wrapper IS the new public API.  Mark the legacy globals (`structureCanvas`, `selection.store`, modify-tab IIFE) as internal in the doc. | client | medium | **✅ shipped 2026-06-07.**  New module `molbuilder/web/static/lib/workspace/dispatcher.js` (~340 LoC) mounts `window.molbuilder.workspace` with the full § 4.2 public surface — `subscribe`, `getState`, `getStructure`, `getSource`, `getSelection`, `isDirty`, `isEmpty`, `loadFromFile`, `generate`, `applyOp`, `save`, `discard`, `undo`, plus the `selection.*` (12 methods) and `view.*` (2 methods) sub-namespaces.  Loaded last on `/molbuilder` (after every legacy store).  Phase 4 implementation is a thin wrapper: reads synthesise from canvas-state + selection store; `selection.*` and `view.*` passthrough to the legacy modules; ops delegate to the existing legacy plumbing (`molbuilderTab.commitFile` for load, `structure.{kind}.generate` for generators, `modify.postOp` + `modify.applyUndo` newly registered on the runtime registry for modifier ops, `structureSave.save` for save).  Subscribers see fan-in from canvas-state + selection store; subscriber errors are caught so one bad consumer can't wedge the dispatcher.  Pinned by `tests/test_workspace_dispatcher_js.py::{TestPublicSurface, TestSubscribe, TestReads, TestPersistRoundtrip, TestSelectionPassthrough}` (14 assertions).  Existing 326 unit + 104 e2e tests unchanged. |
| 5 | Migrate the modify-tab page to use `ws.*` exclusively.  Delete `applyStructure`'s direct touches of canvas-state / selection-store; replace with `ws.applyOp(...)`. | client | high | **✅ shipped 2026-06-07.**  `ws.applyOp` is now self-sufficient — builds the request body via `window.molbuilder.modify.currentStateBody` (exposed Phase 5), POSTs `/api/modify/<op>`, routes the response through the new `_applyWorkspacePayload` pipeline.  Modify-tab `postOp` is a thin wrapper around `ws.applyOp` (keeps the IIFE-local inFlight lock + AbortController + edit-status text).  The `_applyWorkspacePayload` helper is the single cross-store sync point: `cs.replaceContent` (when `touchCanvas:true`) + modify-tab `applyStructure` hook + selection_remap application (Phase 3 contract) + optional `resetSelection`.  Existing modifier-op tests (delete + add + orient + slab) pass unchanged. |
| 6 | Migrate generators (dna/rna/smiles/name/peptide/file) from `structurePage.loadIntoCanvas` + `viewerLoader` → `ws.generate(kind, input, opts)`. | client | high | **✅ shipped 2026-06-07.**  Generator modules unchanged — they keep calling `structurePage.loadIntoCanvas` + `viewerLoader` because that's the warning-modal gate's natural seam.  The migration happens INSIDE `viewerLoader` (= `window.molbuilder.loadStructureText`): the post-fetch state replacement now routes through `ws.applyPayload({touchCanvas: false, resetSelection: true})`, the SAME pipeline modifier ops use.  `touchCanvas:false` because canvas-state was already populated by `structurePage.loadIntoCanvas` (dirty=false); `resetSelection:true` because every load is a fresh-structure swap.  `ws.loadStructure(payload, source)` is now a public method that combines the structurePage gate + applyPayload for callers that have a pre-fetched workspace payload (future Phase 9 inlining of the load fetch).  Every existing generator + sidebar load test passes unchanged. |
| 7 | Migrate selection-bootstrap / panel / viewer-adapter to consume `ws.selection.*`. | client | medium | **✅ effectively shipped 2026-06-07 (no code change required).**  `ws.selection.*` is implemented as a passthrough wrapper over `window.molbuilder.selection.store.*` (Phase 4 contract).  Existing consumers (selection-bootstrap, selection-panel, viewer-adapter) call the store directly; calling `ws.selection.*` would land on the same underlying store.  No code change is required for the migration to satisfy its semantic intent — `ws.selection.*` IS the new public API surface, and the existing consumers will be retired in Phase 9 when the legacy stores fold into the dispatcher.  New code SHOULD use `ws.selection.*`. |
| 8 | Collapse persistence: delete `sessionStorage["modify-state"]` + `sessionStorage["molbuilder.structure_canvas"]`; write only `sessionStorage["molbuilder.workspace.v1"]`. | client | medium | **✅ shipped 2026-06-08.**  Dispatcher writes the unified snapshot under `molbuilder.workspace.v1` (schema v1) on every state change (debounced 100ms) plus a final flush on `pagehide`.  Eager subscribe-on-mount so persistence fires regardless of whether a UI consumer subscribed.  Dirty-gated atoms in `_serialise`: when canvas is clean AND has a source file, atoms are nulled out so the restore re-fetches from disk (preserves the cd9655e external-file-change semantics).  Public `readPersistedSnapshot()` getter on `ws` for the restore path.  **Legacy mirror retirement** (the "what didn't land" sub-list from the previous status): canvas-state's `_persistToSession` and modify viewer's `saveModifyState` now both early-return when `window.molbuilder.workspace` is mounted — only the dispatcher writes on `/molbuilder`.  `restoreModifyState` reads `ws.readPersistedSnapshot()` first via a new `_modifyShapeFromDispatcherSnapshot` translator; canvas-state's `_restoreFromSession` does the same via `_restoreFromDispatcherSnapshot`.  Legacy keys stay as fallback for users mid-session at rollout AND for test contexts that load canvas-state in isolation (no dispatcher → legacy mirror still works).  Pinned by `tests/test_workspace_dispatcher_js.py::TestPersistRoundtrip` + the existing `TestModifySecondVisitExternalChange` + `test_modify_state_after_op_survives_navigation` e2e suite (which all pass unchanged after the collapse). |
| 9 | Retire the legacy public globals (`structureCanvas`, `selection.store`) by moving the impl files into `lib/workspace/` and dropping the public mounts. | client | low | **✅ shipped 2026-06-13.**  Phase 9A (`23e4e80`): `lib/selection/store.js` → `lib/workspace/_selection-store-impl.js`; the singleton self-mount on `window.molbuilder.selection.store` + the matching `runtime.register("selection.store", ...)` are gone; the factory `window.molbuilder.selection._createStore` stays mounted for test harnesses.  Phase 9B (`f9355bc`): `lib/structure/canvas-state.js` → `lib/workspace/_canvas-state-impl.js`; browser-branch mount moved from `structureCanvas` to private `workspace._canvasState`; `runtime.register("structure.canvas", ...)` gone.  Dispatcher owns both singletons internally (creates the selection-store instance from the factory at module init, reads the canvas-state singleton from the impl file's private mount).  **Test escape hatch**: dispatcher's `_canvas()` + `_store()` honour pre-mounted legacy globals if a harness installs them before the dispatcher loads — `tests/test_workspace_dispatcher_js.py` uses this so its existing setup (`structureCanvas = require(...)`, `selection.store = _createStore()`) keeps working unchanged.  Compliance test (`tests/test_no_legacy_store_consumers.py`) enforces zero production consumers; allow-list now lists the two impl files + the dispatcher. |
| 10 | Delete `/api/selection/atoms` if no remaining caller — it's covered by `/api/build/load` once the migration completes. | server | low | **⏸️ blocked on internal API rationalisation.**  Caller audit (2026-06-13): `/api/selection/atoms` is still used by `lib/workspace/_selection-store-impl.js`'s `_fetchAtoms` + `setSourceFile` + `refreshAtoms` paths (now workspace-internal but still hit the endpoint), plus `tests/test_pdb_workflow_integration.py` (integration suite).  Removing the endpoint without first folding the per-source-file atoms refresh into the workspace payload pipeline would break the selection panel + the integration tests.  Deferred — track separately if the consolidation is worthwhile (the architectural debt of two endpoints is small now that the consumers are all dispatcher-internal). |

Each step is gated on its own regression test surface; § 7
enumerates the tests that need to land alongside each step.

---

## 7. Tests required per migration step

| Step | Required tests |
|---|---|
| 1 | `tests/test_shared.py::test_workspace_payload_has_canonical_keys` |
| 2 | `tests/test_web.py::test_load_returns_workspace_payload`, `tests/test_web.py::test_molecule_returns_workspace_payload`, `tests/test_web.py::test_modify_returns_workspace_payload` (one per op) |
| 3 | `tests/test_web.py::test_delete_returns_selection_remap`, `tests/test_modify.py::test_reindex_after_delete_returns_remap` |
| 4 | `tests/test_workspace_dispatcher_js.py::TestPublicSurface`, `::TestSubscribe`, `::TestPersistRoundtrip` |
| 5 | `tests/test_molbuilder_e2e.py::test_applyop_atomically_updates_workspace` (one assertion: after Delete, `getStructure()`, `getSelection()`, `isDirty()`, `getSource()` all reflect post-op state in the same tick) |
| 6 | One e2e per generator: `test_dna_generator_updates_workspace_via_dispatcher` etc. |
| 7 | `tests/test_selection_store_js.py` re-pointed at the wrapper; assertions unchanged |
| 8 | `tests/test_molbuilder_e2e.py::test_workspace_persists_atomically_across_navigation`, `test_dirty_gate_*` already pinned |
| 9 | Removal of legacy stores doesn't break any e2e test |
| 10 | `/api/selection/atoms` deprecation: add `Deprecation` header; remove after all callers migrated |

---

## 8. What this proposal does NOT do

- It does not change the science layer (`molbuilder.Structure`,
  builders, modifiers).  Those are correct; the audit is purely
  about client/server state plumbing.
- It does not change the file format on disk (XYZ / PDB /
  `.molstruct.json` sidecar).  Persistence to disk remains as-is.
- It does not change the CLI.  The CLI surface is already clean
  — small click commands operating on `Structure`.
- It does not introduce a state-management framework (no Redux,
  no MobX).  The dispatcher is plain JS — same flavour as the
  current selection store, just with a wider scope.
- It does not change the dirty-canvas warning model — that
  contract (`isDirty` flag + `confirmDiscardUnsaved`) is correct
  and survives the refactor.

---

## 9. Open design questions

1. **Should `view_state` (camera / style / axes) be part of the
   workspace?**  Pro: one persistence boundary.  Con: the embed
   has its own internal state representation; round-tripping
   through `ws` adds a serialisation step.  Lean: yes — the
   modify tab already saves the camera; folding it into the
   workspace doesn't add work, and consumers can opt out by
   reading `ws.view.getState()` instead of subscribing to it.

2. **Does the workspace persist the undo history?**  Currently
   no; an undo across navigation has never been a request.
   Same plan here — `history` is transient.

3. **Should the sidebar's "candidate" state live in the
   workspace?**  Currently exposed via
   `window.molbuilder.molbuilderTab.{getCandidate, commitFile}`.
   Pro: one less store.  Con: a candidate is a sidebar concept,
   not a workspace concept.  Lean: keep it on the sidebar
   namespace; `ws.loadFromFile(path)` is the commit action.

4. **Does the unified persistence key invalidate cross-tab
   navigation?**  Today canvas-state + modify-state are saved
   independently on pagehide; the workspace-key approach
   centralises this.  The browser's per-tab sessionStorage
   semantics are unchanged.

---

## 10. Process

Per `design.md` § 7, this doc lands first; subsystem docs
(`protocols/atom-selection.md`, `tabs/architecture.md` §5.3,
`protocols/web-api.md`) point at this one for the cross-cutting
contract.  When a migration phase lands, this doc gets struck
through with the section that's been retired; the per-tab specs
get the implementation details.

The 2026-06-07 selection-list bug sequence (BOMB-0 → BOMB-2 →
[cd9655e]) is the historical record of why this refactor is
necessary.  Each commit is a marker for what symptom forced the
fix; collectively they argue that the architectural debt has
shipped enough bugs to warrant the rewrite.
