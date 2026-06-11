# Workspace Contract — sole source of truth

> **This document is the authoritative contract for client-side
> workspace state.** Every read, every write, every persistence
> action, every server response shape MUST match what's specified
> here. Code that diverges is incorrect by definition; the
> contract is right and the code is wrong.
>
> **Companion docs:**
>
> * [`workspace-state.md`](workspace-state.md) — the 2026-06-07
>   audit + migration history that motivated this contract.  Read
>   it for the *why* behind the design.
> * [`web-api.md`](web-api.md) — every HTTP endpoint.  This
>   contract specifies the client-side `ws.*` surface; web-api.md
>   specifies the wire shape it consumes.
>
> **How to use this doc:**
>
> 1. New code consuming workspace state MUST go through `ws.*` —
>    no direct reads of `structureCanvas`, `selection.store`, or
>    the modify-tab IIFE state are permitted.
> 2. Review against this doc, not git blame.  If a behaviour in
>    code doesn't match a contract here, either the code is wrong
>    (fix it) or the contract is wrong (PR the doc + code together).
> 3. Every contract clause is pinned by a test ID in § 9.  A
>    contract change without a test update is a contract
>    violation.

---

## §1 Architecture overview

### 1.1 Modular layout

The workspace dispatcher is composed of **five small modules**, each
with a single responsibility.  All five compose into one mounted
object: `window.molbuilder.workspace` (aliased as `ws` throughout
this doc).

```
molbuilder/web/static/lib/workspace/
├── index.js          ── §1.2 — public mount + module assembly
├── state.js          ── §1.3 — the single in-memory workspace
├── reads.js          ── §2   — every getter
├── writes.js         ── §3   — every mutator (HTTP + in-memory)
├── persistence.js    ── §4   — sessionStorage + restore
├── selection.js      ── §5   — selection sub-namespace
└── view.js           ── §6   — view sub-namespace (camera/style)
```

Each module exports a factory function `create<Module>({state, notify, ...deps})` that returns the module's public surface.
`index.js` instantiates them in dependency order, wires
notifications, and mounts the assembled object on
`window.molbuilder.workspace`.

### 1.2 Single in-memory state

There is **one** workspace state object, owned by `state.js`.
Every read returns a defensive copy; every write mutates it
through a single helper (`state.replace(slice)`) that fires
subscribers exactly once per mutation.

```js
// state.js shape (TypeScript-style for docs only — code is plain JS)
type WorkspaceState = {
  structure: {
    text:          string,        // canonical XYZ or PDB bytes
    source_format: "xyz" | "pdb",
    title:         string,
    n_atoms:       number,
    atoms:         Atom[],        // per-atom rows (see §7.2)
    lattice:       number[][] | null,  // 3x3 or null
  } | null,

  source: {
    kind: "file"|"smiles"|"name"|"dna"|"rna"|"peptide"|"blank",
    file: string | null,
    generator_input: object | null,
  },

  dirty:        boolean,
  last_save_to: string | null,

  selection: {
    indices:    number[],         // sorted-ascending
    mode:       "click"|"filter",
    filters:    Filter[],
    combinator: "or"|"and",
  },

  view: {
    camera?: object,   axes?: boolean,
    style?:  object,   labels?: boolean,
  } | null,

  // Transient — never persisted, never restored:
  loading:  boolean,
  inFlight: boolean,
  error:    string | null,
  history:  WorkspaceSnapshot[],
}
```

### 1.3 The flow

```
                              ┌──────────────────────────┐
   User UI input              │                          │
   ────────────────►          │      ws.* WRITE API      │ (§3, §5)
                              │ loadFromFile, applyOp,   │
                              │ selection.toggle, ...    │
                              │                          │
                              └────────────┬─────────────┘
                                           │  fetch + payload pipeline
                                           ▼
                              ┌──────────────────────────┐
                              │   state.js replace()     │ (§1.2)
                              │   atomic mutation        │
                              └────────────┬─────────────┘
                                           │  notify
                                           ▼
              ┌────────────────────────────┼────────────────────────────┐
              ▼                            ▼                            ▼
   ┌──────────────────┐    ┌──────────────────────────┐  ┌──────────────────────────┐
   │ subscribers      │    │ persistence.js           │  │ readers via ws.* READ    │
   │ (UI panels       │    │ debounced sessionStorage │  │ API (§2): selection-panel,│
   │  refreshing)     │    │ write                    │  │  viewer-adapter, save    │
   └──────────────────┘    └──────────────────────────┘  └──────────────────────────┘
```

**No store outside `state.js` may hold a copy of workspace data.**  The
3Dmol embed handle and selection-panel DOM are *renderings*, not
copies.

---

## §2 Read API — `ws.*` getters

Every getter returns a **defensive copy** (or a freshly-built
object): mutating the returned value does NOT mutate the
underlying state.  Read API never throws — missing state is
represented as `null` or empty.

| Method | Returns | Contract |
|---|---|---|
| `ws.getState()` | `WorkspaceState` (deep-cloned) | Composite snapshot.  Atomic — every nested field reflects the same `notify()` tick.  Use sparingly; prefer narrow getters. |
| `ws.getStructure()` | `{text, source_format, title, n_atoms, atoms, lattice}` or `null` | Returns `null` iff workspace is empty (§2.4).  `atoms` is a slice of the underlying array.  Never returns partial state — if `text` is present, `atoms` is consistent with it. |
| `ws.getSource()` | `{kind, file, generator_input}` | Always returns an object.  Empty workspace returns `{kind: "blank", file: null, generator_input: null}`. |
| `ws.getSelection()` | `{indices, mode, filters, combinator}` | `indices` always sorted ascending, no duplicates.  `filters` defensive-copied (each filter object cloned). |
| `ws.isDirty()` | `boolean` | True iff text has been mutated since last `save()`.  Set by `applyOp` / generators / undo.  Cleared by `save()`. |
| `ws.isEmpty()` | `boolean` | True iff there is no structure loaded.  Equivalent to `getStructure() === null`. |
| `ws.getAtoms()` | `Atom[]` (slice) | Direct atom-array accessor for hot paths (filter/picker).  Returns `[]` when empty.  Always reflects current state — never stale relative to `getStructure().atoms`. |
| `ws.getSourceFile()` | `string \| null` | Convenience: equivalent to `getSource().file`.  Used by selection-panel + viewer-adapter (current code reads `selection.store.getState().sourceFile`). |

### 2.1 Subscriptions

```js
const unsub = ws.subscribe(fn);   // returns unsubscribe function
```

Contract:

- `fn` is called **once immediately** on subscribe with the current
  `getState()` snapshot.
- `fn` is called **once after each `notify()` tick** with a fresh
  snapshot.
- Subscriber errors are **caught** — they do not prevent other
  subscribers from running and do not throw out of `notify()`.
- Calling `unsub()` is idempotent; calling it from inside `fn`
  is safe (uses iteration-time copy).

### 2.2 Atomicity

All reads within one `notify()` tick observe the same snapshot.
The only way to violate this is to mutate `ws` mid-tick from a
subscriber — **don't do that**.  Subscribers should mutate via
`ws.*` writes; chained mutations from a subscriber serialize on
the next microtask.

### 2.3 What gets persisted

`persistence.js` writes the workspace state minus `loading` /
`inFlight` / `error` / `history` on every `notify()` (debounced 100ms).
See §4 for the full persistence contract.

### 2.4 What "empty workspace" means

```js
ws.isEmpty()        === true
ws.getStructure()   === null
ws.getAtoms()       === []                       // not null
ws.getSource()      === {kind:"blank", file:null, generator_input:null}
ws.getSelection()   === {indices:[], mode:"click", filters:[], combinator:"or"}
ws.isDirty()        === false
```

---

## §3 Write API — `ws.*` mutators

Every mutator either succeeds (state replaced atomically, `notify()`
fires once) or rejects (state unchanged, no notification).  No
mutator leaves partial state.

| Method | Server route | Returns | Side effects |
|---|---|---|---|
| `ws.loadFromFile(path)` | POST `/api/build/load` | `Promise<WorkspacePayload>` | Replaces structure + source.kind="file" + source.file=path; resets selection to `[]`; resets dirty to `false` |
| `ws.loadFromText(text, filename)` | POST `/api/build/load` | `Promise<WorkspacePayload>` | Same as loadFromFile but with in-memory text.  source.kind="file" iff filename is a real path; resetSelection=true; touchCanvas=false (caller is responsible for canvas-state.dirty) |
| `ws.generate(kind, input, opts)` | POST `/api/build/molecule` | `Promise<WorkspacePayload>` | Replaces structure + source.kind=kind + source.generator_input=input; resets selection to `[]`; dirty=true |
| `ws.applyOp(op, args)` | POST `/api/modify/<op>` | `Promise<WorkspacePayload>` | Replaces structure; pushes pre-op snapshot to `history`; applies selection_remap per §3.4; dirty=true |
| `ws.applyPayload(payload, opts)` | (none — in-memory) | `void` | Direct atomic install (used internally by every HTTP mutator; exposed for restore paths).  `opts.touchCanvas`, `opts.resetSelection` per §3.3 |
| `ws.save(opts)` | POST `/api/files/write` | `Promise<void>` | Writes structure.text to opts.path; sets dirty=false, last_save_to=opts.path |
| `ws.discard()` | (none) | `void` | Sets structure=null, source={kind:"blank",...}, selection={indices:[],...}, dirty=false.  **Unconditional** — caller MUST gate on warning modal first. |
| `ws.undo()` | (none) | `void` | Pops last entry from `history`, calls `applyPayload(snap, {touchCanvas: true})`.  No-op when history is empty. |

### 3.1 Error handling

HTTP mutators reject with `Error(message)` where `message` comes
from:
- The server's `{ok: false, error}` envelope's `error` field
  (§7.4), OR
- A clean network-error message (`"network: timeout"`, etc.)

Rejection leaves state unchanged.  The caller is responsible for
displaying the error to the user.

### 3.2 The payload pipeline

`applyPayload(payload, opts)` is the **single sync point** for all
state replacement.  It performs in order:

1. Capture `preSelection` (before any mutation).
2. Replace `state.structure.text` (and dirty bit) when
   `opts.touchCanvas !== false`.
3. Replace `state.structure.atoms` from `payload.atoms`.
4. Apply selection remap (§3.4) or reset selection when
   `opts.resetSelection`.
5. Replace `state.structure.title`, `n_atoms`, `lattice` from
   payload.
6. Fire `notify()` exactly once.

The pipeline is the only place that touches state.atoms.  Every
public write method ends in `applyPayload`.

### 3.3 `applyPayload` options

| Option | Default | Effect when set |
|---|---|---|
| `touchCanvas` | `true` | When `false`, skip the dirty-bit update (caller has already set it; used by load paths that pre-marked clean) |
| `resetSelection` | `false` | When `true`, clear selection unconditionally (used by load/generate; modifier ops use selection_remap instead) |

### 3.4 Per-op selection rule

| Op class | Selection update |
|---|---|
| `loadFromFile` / `loadFromText` / `generate` | `resetSelection: true` → indices=[] |
| `applyOp(delete)` / `applyOp(add_atom)` | Apply `payload.extra.selection_remap` (§7.3) |
| `applyOp(translate)` / `applyOp(rotate)` / etc. (atom count unchanged) | Preserve as-is (no opt set; pipeline leaves it alone) |
| `discard` | indices=[] |
| `undo` | Restored from snapshot.selected |

`selection_remap` is a flat list per §7.3.  When the server sends
it, the dispatcher MUST use it instead of the naive in-range
filter, otherwise Delete-of-low-index silently drops the wrong
atom.

---

## §4 Persistence contract

### 4.1 The single key

```
sessionStorage["molbuilder.workspace.v1"] = JSON.stringify({
  v:        1,                              // schema version
  saved_at: "2026-06-09T20:30:00.000Z",     // ISO 8601, UTC
  workspace: {
    structure: { ... } | null,
    source:    { ... },
    dirty:     boolean,
    last_save_to: string | null,
    selection: { ... },
    view:      { ... } | null,
  },
})
```

**There is no other persistence key.** The legacy keys
(`molbuilder.structure_canvas`, `modify-state`,
`molbuilder.panelMode`) are deleted as of Phase 10; restoring code
that reads them is incorrect.

### 4.2 Write cadence

- Debounced 100ms after every `notify()` tick.
- Final flush on `pagehide` event (no debounce).
- Errors (quota exceeded, storage disabled) are logged + swallowed.

### 4.3 Read at restore

```js
const snap = ws.readPersistedSnapshot();    // null if missing/corrupt
```

Contract:
- Returns the parsed `workspace` object or `null`.
- `null` covers: no key, malformed JSON, schema version mismatch.
- The caller (page bootstrap) decides whether to re-fetch from
  disk (dirty=false, source.kind=file) or atomic-replace from
  memory (dirty=true, or non-file source).  Decision logic is in
  `persistence.js::shouldRefetchFromDisk(snap)`.

### 4.4 What's NOT persisted

`loading`, `inFlight`, `error`, `history`.  These are transient
runtime state; restoring them would be incorrect (e.g.
`inFlight=true` from a navigation-killed request).

---

## §5 Selection sub-namespace — `ws.selection.*`

### 5.1 Local mutators (no HTTP)

| Method | Effect |
|---|---|
| `ws.selection.toggle(i)` | Toggle atom `i` in selection.  Out-of-range indices ignored. |
| `ws.selection.set(indices)` | Replace selection with sorted-unique copy of `indices`.  Out-of-range filtered out. |
| `ws.selection.add(indices)` | Union with current selection.  Sort + dedup. |
| `ws.selection.remove(indices)` | Subtract from current selection. |
| `ws.selection.all()` | Select every atom (`0..n_atoms-1`). |
| `ws.selection.invert()` | Replace with complement. |
| `ws.selection.clear()` | Empty the selection. |
| `ws.selection.setMode(mode)` | Sets mode to `"click"` or `"filter"`.  Throws on invalid value. |
| `ws.selection.setFilters(filters)` | Replaces filter list.  Does NOT eval — call `applyFilter()` separately. |
| `ws.selection.setCombinator(c)` | Sets combinator to `"or"` or `"and"`. |

Each mutator fires `notify()` exactly once.

### 5.2 Server-backed selection ops

| Method | Server route | Returns | Effect |
|---|---|---|---|
| `ws.selection.applyFilter()` | POST `/api/selection/eval` | `Promise<number[]>` | Sends current filters + combinator to server; replaces selection with result; preserves mode. |
| `ws.selection.writeLabel(target, indices)` | POST `/api/selection/save` | `Promise<void>` | Writes a sidecar label.  `target` is `"frozen_atoms"` or one of the region names. |

### 5.3 Atoms accessor

`ws.selection.getAtoms()` is an alias for `ws.getAtoms()` —
provided so existing call sites that read `selection.store.getState().atoms`
have a 1-line migration target.

---

## §6 View sub-namespace — `ws.view.*`

| Method | Effect |
|---|---|
| `ws.view.applyState(patch)` | Merges `patch` into view state; delegates camera/style updates to the 3Dmol embed handle. |
| `ws.view.getState()` | Returns the current view state (camera + style + axes + labels).  Always returns an object (never null) — empty view returns `{}`. |

The 3Dmol embed handle is a *rendering target*, not a store.  View
state in `state.js::state.view` is the source of truth; the embed
mirrors it.  This contract is maintained by `view.js`.

---

## §7 Wire contract (server → client)

### 7.1 WorkspacePayload — every Structure-returning endpoint

```json
{
  "ok":            true,
  "text":          "...",       // canonical XYZ or PDB bytes
  "source_format": "xyz",       // "xyz" | "pdb"
  "title":         "...",
  "n_atoms":       42,
  "atoms":         [ /* per §7.2 */ ],
  "lattice":       null,        // or 3×3 array
  "issues":        [ /* Issue records */ ],
  "extra":         { /* per-endpoint additions */ }
}
```

Routes that emit this shape (per `web-api.md`):

| Route | `extra` keys |
|---|---|
| POST `/api/build/load` | `pdb`, `source_format`, `n_residues`, `summary` |
| POST `/api/build/molecule` | `backend_used`, `add_hydrogens_mode`, `pdb`, `summary` |
| POST `/api/modify/<op>` | `selection_remap` (when applicable), `op`, `args` |

### 7.2 Atom row shape

```json
{
  "index":         12,
  "element":       "C",
  "atom_name":     "CA",         // PDB only; absent for XYZ-origin atoms
  "residue_id":    42,
  "residue_name":  "ALA",
  "chain_id":      "A",
  "regions":       ["bridge"],
  "is_frozen":     false
}
```

### 7.3 selection_remap (in `extra`)

Flat list of `length = pre-op atom count`.

```json
"selection_remap": [null, 0, 1]      // delete index 0
"selection_remap": [0, 1, 2]         // add atom (identity, new atom at index 3)
"selection_remap": [0, 1, 2, 3]      // no-shift op (identity)
```

`remap[old_index] === new_index` (or `null` when the atom was removed).

### 7.4 Error envelope

```json
{
  "ok":     false,
  "error":  "human-readable message",
  "issues": [ /* optional issue records */ ]
}
```

The client surfaces `error` to the user.  When `issues` is
present, the panel renders them too (per per-tab convention).

---

## §8 Deprecated surfaces — DO NOT USE

The following surfaces existed pre-Phase-10 but are now **deleted**:

| Deprecated surface | Replacement |
|---|---|
| `window.molbuilder.structureCanvas.*` | `ws.getStructure()` / `ws.isDirty()` / `ws.getSource()` |
| `window.molbuilder.selection.store.*` | `ws.selection.*` + `ws.getAtoms()` / `ws.getSelection()` |
| `window.molbuilder.modify.state` (IIFE) | `ws.getStructure()` |
| `sessionStorage["molbuilder.structure_canvas"]` | `sessionStorage["molbuilder.workspace.v1"]` |
| `sessionStorage["modify-state"]` | `sessionStorage["molbuilder.workspace.v1"]` |
| `sessionStorage["molbuilder.panelMode"]` | `sessionStorage["molbuilder.workspace.v1"].selection.mode` |

A `grep -rn 'structureCanvas\|selection\.store\|window.molbuilder.modify.state' molbuilder/web/static/` from a non-`lib/workspace/` directory MUST return zero matches.  This is enforced by `tests/test_no_legacy_store_consumers.py`.

---

## §9 Compliance map (tests pin every clause)

| Contract clause | Pinning test |
|---|---|
| §1.2 single state | `tests/test_workspace_state_singleton_js.py` |
| §2 each `ws.*` getter exists + returns documented shape | `tests/test_workspace_dispatcher_js.py::TestPublicSurface`, `::TestReads` |
| §2.1 subscribe contract | `tests/test_workspace_dispatcher_js.py::TestSubscribe` |
| §2.2 atomicity | `tests/test_workspace_atomic_reads_js.py` |
| §2.4 empty workspace shape | `tests/test_workspace_dispatcher_js.py::TestEmptyWorkspace` |
| §3 each `ws.*` mutator routes through `applyPayload` | `tests/test_workspace_dispatcher_js.py::TestWritePipeline` |
| §3.2 payload-pipeline order | `tests/test_workspace_dispatcher_js.py::TestPayloadPipelineOrder` |
| §3.4 per-op selection rule | `tests/test_workspace_dispatcher_js.py::TestSelectionRemap` |
| §4 persistence contract | `tests/test_workspace_dispatcher_js.py::TestPersistRoundtrip` |
| §5 selection sub-API | `tests/test_workspace_selection_subapi_js.py` |
| §6 view sub-API | `tests/test_workspace_view_subapi_js.py` |
| §7.1 wire shape | `tests/test_shared.py::TestWorkspacePayload` |
| §7.3 selection_remap shape | `tests/test_modify.py::TestComputeSelectionRemap*` |
| §8 zero legacy-store consumers | `tests/test_no_legacy_store_consumers.py` |

A new test ID appears in this column iff a new clause is added.  A clause without a pinning test ID is a contract gap.

---

## §10 Change process

1. PR the contract change AND the code AND the test together.
2. Update §9 if the test ID changes.
3. Cross-reference [`workspace-state.md`](workspace-state.md) when
   the *rationale* changes (historical context for the design).
4. NEVER ship a code change that diverges from this contract.
   If the contract is wrong, change it explicitly.
