# Spec — Atom-selection module

Persistent reference for the JS modules that let a user pick atoms in
a structure (by clicking, by filtering on element / index range /
region label) and write those picks to the `.molstruct.json` sidecar
as named regions / frozen-atom lists.

Distinct from `selection.md` (which covers the Projects sidebar's
file-selection model).  This doc covers **atom selection**.

> **Scope of this doc.**  The data structure, the store API, the
> three consumer modules' contract, the cross-module event
> protocol, and the migration plan that landed at Phase B.1.10
> (2026-05-20).  Server-side (`/api/selection/*`) is documented in
> `molbuilder/web/blueprints/selection.py`.

## 1. Modules + responsibilities

| File | Layer | Responsibility | Public API |
|---|---|---|---|
| `lib/selection/store.js` | L1 | Singleton state + HTTP wiring + rule translation | `subscribe`, `getState`, mutators |
| `lib/selection-panel.js` | L2 | DOM only — subscribes to store, renders, calls mutators on user input | `mount(rootEl) → {dispose}` |
| `lib/selection/viewer-adapter.js` | L2 | 3Dmol overlay only — subscribes to store, paints region tints + halo, forwards viewer clicks to `store.toggleAtom` | `attach(viewer) → {dispose}` |
| `<page>/selection-bootstrap.js` | L3 | Page glue — fetches partial, mounts panel + adapter, wires sidebar `onChange → store.setSourceFile` | (none; runs on DOMContentLoaded) |

**Cross-module talk happens ONLY through the store.**  The panel and
adapter never reference each other; neither references the bootstrap.
Each can mount without the others (panel works without a viewer, the
adapter works without the panel — both are graceful no-ops on
unmounted siblings).

## 2. Data structures

### `Atom`

One row in `state.atoms`.

```
{
  index:        integer (0-based)
  element:      "C", "Au", ...
  atomName?:    "CA"          (PDB only; absent for plain XYZ)
  residueName?: "ALA"         (PDB only)
  chainId?:     "A"           (PDB only)
  labels:       string[]      // region tags; possibly empty
  isFrozen:     boolean       // separate from labels (different concept)
}
```

`labels` are user-named regions (`L-electrode`, `bridge`, `interface`,
...) that come from the sidecar's `regions` field.  `isFrozen` mirrors
membership in the sidecar's `frozen_atoms` list.  Both can co-exist
on the same atom; an atom may carry multiple labels at once.

The server's `/api/selection/atoms` returns `regions: [...]` per
atom; the store renames to `labels` at the boundary so the JS layer
uses one consistent term.

### `Filter`

One row in `state.filters` (used when `state.mode === "filter"`).

```
{
  kind:   "by_element" | "by_index" | "by_label"
  value:  user-typed text, interpreted by `kind`:
            by_element  -> "Au,C"           (comma-separated symbols)
            by_index    -> "0-3, 9-10, 42"  (ranges + singletons)
            by_label    -> "L-electrode"    (single label name)
}
```

The store translates `{filters, combinator}` into the server's rule
grammar (`Or` / `And` of `ByElement` / `ByIndexRange` / `ByRegion`)
at HTTP time.  Filters in the UI never expose `Not` / `Minus` /
`FirstN` — those rule kinds are reserved for programmatic /
sidecar-saved expressions.

### `State` (the whole store)

```
{
  sourceFile:  null | string       // absolute structure path (.xyz / .pdb)
  atoms:       Atom[]              // current structure's atom list
  selection:   number[]            // THE selection set; shared across modes
  pickOrder:   number[]            // same atoms as `selection`, in click
                                   //   order (vertex = pickOrder[1] for
                                   //   the 3-atom angle readout)
  mode:        "click" | "filter"  // which editor is visible (UI only)
  filters:     Filter[]            // filter drafts (NOT applied until
                                   //   the user clicks Apply filter)
  combinator:  "or" | "and"        // how multiple filters compose
  loading:     boolean             // any HTTP in flight
  error:       null | string       // most recent failure
}
```

**`selection` is the canonical state.**  Click mode edits it
atom-by-atom (client-side toggle, no HTTP).  Filter mode lets the
user compose a query that the explicit `applyFilter()` call
materialises into `selection` (replacing whatever was there).
Switching modes does NOT touch `selection` — switching from
Filter back to Click after Apply keeps the filtered atoms
selected, ready to refine atom-by-atom.

**`pickOrder` is the click-order shadow.**  Same atom-indices as
`selection`, but in the order the user clicked them.  `selection`
is kept sorted ascending (set semantics — every consumer that
doesn't care about click history can read it directly); the
shadow exists so the 3-atom angle readout in
`lib/selection/measurements.js` can use the user's *second* click
as the vertex (chemist's convention: pick A → B → C means ∠A-B-C
with B at the vertex).  Every mutator keeps the two in lock-step:

  * `toggleAtom(i)` — adds: append `i`; removes: filter `i` out.
  * `setSelection(xs)` — `pickOrder` is the input array deduped
    in input order; `selection` is that same array sorted.
  * `addToSelection(xs)` — appends only the *new* atoms in input
    order (mirrors a series of toggleAtom calls).
  * `removeFromSelection(xs)` — filters dropped atoms out;
    surviving order preserved.
  * `clearSelection()`, `setSourceFile(...)`, `adoptSession(...)`
    — both arrays reset together.
  * `applyFilter()` — `pickOrder` mirrors the filtered selection
    in ascending order (no user clicks happened); the
    measurement module's geometric vertex heuristic kicks in
    instead.

Consumers that don't care about click order (atom list, label
writes, filter eval, the selection panel's count) keep reading
`selection`.  Consumers that DO care (the measurement readout)
pass `pickOrder` to `measurements.compute(...)`; it falls back to
the geometric heuristic when `pickOrder` doesn't match the
selection set (filter eval, session restore, batch setSelection
from a non-click source).

## 3. Store API

```js
const store = window.molbuilder.selection.store;
```

### Read

```
store.getState() -> State          // immutable snapshot (shallow copy)
store.subscribe(fn) -> unsubscribe // fn(state) called on every change;
                                    // fires once immediately with current state
```

### Source file

```
store.setSourceFile(path | null) -> Promise
```

The **single entry point** for "switch structure".  Internally
batches:

  1. Update `state.sourceFile`.  Clear `state.selection` (a fresh
     file starts empty).
  2. Fetch `/api/files/read` for the structure bytes (XYZ or PDB),
     hand to `window.molbuilder.loadXyzText` (the page-supplied
     viewer loader; despite the legacy name it accepts both XYZ
     and PDB content — the server's `/api/build/load` sniffs the
     format).
  3. Fetch `/api/selection/atoms` → `state.atoms`.

No re-eval of `state.filters` here — filter drafts persist
through a file switch but are not auto-applied to the new file;
the user must explicitly `applyFilter()` against the new
structure if that's what they want.

`null` clears the source.

```
store.refreshAtoms() -> Promise
```

Re-fetch `/api/selection/atoms` without changing `sourceFile`.
Called internally after `writeLabel` so newly-assigned labels show
up immediately.  Public so callers (e.g. another tab that edited
the sidecar) can also force a refresh.

```
store.setLoader(fn | null) -> void
```

Inject the page-supplied viewer loader the store calls during
`setSourceFile`'s step 2.  `fn` receives a `{path, text, format}`
payload and returns a Promise that resolves once the viewer has
populated its model.  Pass `null` to detach (the store falls back
to atom-list-only / "headless" mode -- atoms still fetch but no
viewer is driven, useful for tests + the future structure
inspector adapter).  Throws `TypeError` for any other type.

Wired by the Molbuilder tab's bootstrap shortly after page load; tests can
swap in a stub via the store's test-only `_createStore` factory
(see playwright-tests.md § 1.1 "Module singletons and test
isolation").

```
store.adoptSession({sourceFile, selection}) -> Promise
```

Rehydrate from a session snapshot WITHOUT re-loading the viewer.
Used by the Molbuilder tab's sessionStorage restore path where the viewer
model has already been populated synchronously via
`applyStructure(...)`.  Differs from `setSourceFile` in two ways:

  1. Skips the viewer-load step entirely (the viewer is already
     populated -- re-loading would discard the camera/indices and
     double-fetch over HTTP for no gain).
  2. Accepts a pre-validated `selection` array that survives the
     structure swap, so the panel and adapter come back in sync
     without losing the user's pick.

Atoms ARE still fetched fresh from the server so any sidecar
update done since the snapshot was written is reflected.  Rejects
with `TypeError` if `sourceFile` is non-string-non-null;
non-number entries in `selection` are silently filtered.

### Mode

```
store.setMode("click" | "filter") -> Promise
```

Pure UI-mode change — controls which editor body the panel shows.
**Does NOT modify `state.selection`**: the selection persists
across mode flips so switching is non-destructive.

### Selection editing

```
store.toggleAtom(index)             -> Promise   // flip membership; sorted-ascending
store.setSelection(indices[])       -> Promise   // replace; dedupes + sorts; drops non-numbers
store.addToSelection(indices[])     -> Promise   // union with current selection
store.removeFromSelection(indices[]) -> Promise   // difference
store.selectAll()                   -> Promise   // every atom in state.atoms
store.invertSelection()             -> Promise   // complement against state.atoms
store.clearSelection()              -> Promise   // empty
```

These edit `state.selection` directly.  Wired to the atom-list
checkboxes, the 3D-viewer click handler, the master select-all
checkbox, and the panel's bulk-action buttons.  No server
round-trip — every mutator is a local set operation.

### Filter drafts

```
store.setFilters(filters[])           -> Promise   // replace draft array
store.addFilter(filter)               -> Promise   // append
store.removeFilter(index)             -> Promise   // by position
store.updateFilter(index, filter)     -> Promise   // by position
store.setCombinator("or" | "and")     -> Promise
```

**Drafts only — these do NOT touch `state.selection`.**  They
update the filter editor's state so the panel can show / preserve
the typed query, but the selection isn't replaced until the user
explicitly applies the filter (below).

### Apply filter

```
store.applyFilter() -> Promise
```

POSTs the translated rule to `/api/selection/eval` and sets
`state.selection` to the result.  This is the **only** path from
filter drafts to materialised selection.  Filter rows with empty
values (e.g. an `by_index` row the user added but hasn't typed
into yet) are SKIPPED when building the rule, so a half-typed
filter doesn't poison an AND combinator with an empty-set operand.
If no filter rows are present (or all rows are empty), the
selection becomes empty.

After a successful eval, if `state.atoms` is empty the store
issues a follow-up `/api/selection/atoms` fetch to repopulate
it.  This guards against a race where the user clicks Apply
filter while a prior `setSourceFile` is still loading: the
filter-eval's `_run` aborts the file-load's signal, and the
atom fetch portion of the load never lands.  Without this safety
net the panel would show "N / 0 atoms" + an empty atom list
while the viewer (whose model was already populated by
`xyzLoader` before the abort) shows the structure with halos.

### Sidecar writes

```
store.writeLabel(target, indices[]) -> Promise
```

POSTs `{structure_path, target, indices}` to
`/api/selection/save`.  REPLACE semantics on the target: sets
that region's membership to exactly `indices`.  Multi-label
model — assigning to one region does NOT remove an atom from
other regions.  Re-fetches `state.atoms` on success so the
panel reflects the new label tags.  Does NOT touch
`state.selection`.

Special case for empty `indices`:

  * `target == "frozen_atoms"` and `indices == []` → clears the
    frozen-atom set AND drops `selection_rules["frozen_atoms"]` so
    a stale rule won't silently undo the clear on the next load.
  * `target == <region>` and `indices == []` → removes the
    region entirely from the sidecar (and its rule, if any) —
    NOT "keep an empty region".  An empty region carries no
    semantic value and would just clutter the panel's tag list;
    removal also lets the user "clean up" a deprecated label
    via a single empty-Assign action.

### Label writes (sidecar)

```
store.writeLabel(target, indices[]) -> Promise
```

REPLACE semantics — sets `target`'s membership to exactly the given
indices.  `target` is a region name or the literal `"frozen_atoms"`.
POSTs `/api/selection/save` then `refreshAtoms()` so labels appear in
the UI without a manual refetch.

The Assign / Add / Remove buttons in the panel compute the right
`indices` for each verb (replace / union / difference) and call this
once.

### Clear

```
store.clearSelection() -> Promise
```

Empties `state.selection`.  Does NOT touch filter drafts -- they
stay around so the user can re-apply or edit.  No-op + no notify
if the selection was already empty.

## 4. Event protocol

**One event type: "state changed".**  Every subscriber receives a
full state snapshot on every mutation.  No `selection-changed` /
`atoms-changed` partitioning — granularity is per-mutator, not per-
field, and consumers decide what to re-render based on the diff.

Why one type:
  * The state is small (~7 fields); re-rendering on every change is
    cheap.
  * Multiple event types would require subscribers to know which
    fields each event touches — coupling.
  * The "what changed" question is answered by comparing snapshots
    if needed.

### Batching contract

A mutator fires subscribers **twice**:

  1. At the start, with `state.loading = true`, so the UI can show a
     spinner.
  2. At the end, with `state.loading = false` and the final state.

A mutator that does multiple HTTP calls (e.g. `setSourceFile` does
file read + atom fetch + re-eval) does NOT fire between those steps
— only the start + end pair.  No intermediate flicker.

### Reentrance

If a subscriber writes back to the store synchronously, the second
notify is queued via microtask — never reentrant on the same call
stack.  A subscriber may safely call any mutator without crashing
the store.

### Error handling

Mutators don't throw user-facing errors — they catch HTTP failures
and set `state.error` to the message.  Subscribers render the error
inline.  The promise returned by a mutator resolves on completion
regardless of success (check `state.error` to detect failure).

## 5. Dependency diagram

```
                ┌───────────────────────────┐
                │ /api/selection/{atoms,    │
                │   eval,toggle,save}       │
                │ /api/files/read           │
                └─────────────┬─────────────┘
                              │ HTTP (POST/GET, JSON)
                              │
                ┌─────────────▼─────────────┐
                │   lib/selection/store.js   │
                │   (singleton; state + HTTP)│
                └──┬───────┬───────┬─────────┘
       subscribe   │       │       │   subscribe
        ┌──────────┘       │       └─────────┐
        ▼                  ▼                 ▼
 ┌──────────────┐  ┌────────────────┐  ┌────────────────┐
 │   panel      │  │ viewer-adapter │  │   bootstrap    │
 │ (DOM render) │  │ (3Dmol overlay │  │  (page glue)   │
 │              │  │  + click→store)│  │                │
 └──────┬───────┘  └───────┬────────┘  └────────┬───────┘
        │                  │                    │
        │ user input       │ viewer click       │ sidebar onChange
        │                  │                    │
        └────── store mutators (one path) ──────┘
```

**Rules:**
  1. No module imports another non-store module.
  2. Every cross-module signal goes through the store.
  3. The store has no DOM, no 3Dmol, no Flask — pure data + fetch.
  4. The panel and adapter NEVER mutate `state` directly; they call
     mutators.

## 6. Information flow — three canonical scenarios

### Scenario A: User picks an XYZ in the sidebar

```
sidebar.onChange(path)
   │
   ▼
bootstrap → store.setSourceFile(path)
                │
                ▼
            ┌──────── store batches ────────┐
            │ a. clear state.selection      │
            │ b. fetch /api/files/read      │
            │ c. loadXyzText(text)          │  ← viewer model swap
            │    (throws on failure -->     │
            │     skip step d)              │
            │ d. fetch /api/selection/atoms │  → state.atoms
            │ e. NOTIFY (loading=false)     │
            └────────────────┬──────────────┘
                             ▼
              ┌──────────────┼───────────────┐
              ▼              ▼               ▼
          panel           adapter         bootstrap
        re-renders        paints          (idle)
        list + tags       region tints
                          + fixed marker
                          + halo (empty)
```

No `/api/selection/eval` is fired on a file switch — the selection
starts empty.  Filter drafts (`state.filters`) DO persist across
file switches, but they are not re-evaluated against the new
structure until the user clicks **Apply filter** (see Scenario B).

No flicker — the viewer model swap and the overlay paint land
together.

### Scenario B: User clicks atom 5 in the viewer

```
viewer click(atom 5)
   │
   ▼
adapter.onAtomClicked(5)
   │
   ▼
store.toggleAtom(5)
   │  (client-side flip in state.selection -- no HTTP)
   ▼
state.selection updated
   │
   ▼
NOTIFY
   │
   ├── panel: row 5 checkbox ticks, count updates
   └── adapter: yellow halo on atom 5
```

### Scenario C: User clicks "Assign" for L-electrode

```
panel: assign button clicked
   │
   ▼
panel collects target + indices, calls
store.writeLabel("L-electrode", state.selection)
   │
   ▼
POST /api/selection/save
   │  ← sidecar updated
   ▼
store.refreshAtoms()
   │
   ▼
fetch /api/selection/atoms
   │  ← server returns atoms with new labels[]
   ▼
state.atoms updated
   │
   ▼
NOTIFY
   │
   ├── panel: green "L-ELECTRODE" tag appears in the row
   └── adapter: green region tint on those atoms
```

## 7. Migration notes (what disappeared)

| Removed | Why |
|---|---|
| `lib/selection/core.js` | Third state holder; folded into store |
| `panel.modeState.click` / `modeState.filter` | Replaced by `state.selection` (shared across modes) |
| `state.clicks` (mode-private click buffer) | Selection is one shared set; modes are pure UI |
| Per-key eval on filter edits | Replaced by explicit `applyFilter()` -- drafts don't auto-evaluate |
| Client use of `/api/selection/toggle` | Toggle is client-side now; the server endpoint is unused (kept for future programmatic callers) |
| `panel.atomList` cache | Lives in `state.atoms` now |
| `bootstrap.lastLoadedPath` | Store batches; no duplicate-fire problem |
| `adapter.rerender()` | `setSourceFile` is atomic; no load race |
| `handle.getCore()` leak | Panel handle is flat |
| `_assignClicked` / `_setTarget` / `_removeLabel` (3 duplicates) | Collapsed into `store.writeLabel` |
| Server-side `regions` → JS `labels` (rename at boundary) | One term in JS, one in Python; clear separation |

## 8. Public surface table (machine-checkable)

The selection store is the SOLE entry point.  Anything else is
internal.

### Top-level modules

| Symbol | Type | Doc |
|---|---|---|
| `window.molbuilder.selection.store` | object | (this doc) |
| `window.molbuilder.selectionPanel.mount(rootEl) -> {dispose}` | function | mounts the panel inside `rootEl` |
| `window.molbuilder.selection.viewerAdapter.attach(viewer) -> {dispose}` | function | attaches the adapter to a 3Dmol viewer |

### Store methods

Source-file lifecycle:

| Method | Kind | Notes |
|---|---|---|
| `getState() -> snapshot` | sync read | returns a defensive copy |
| `subscribe(fn) -> unsubscribe` | sync | fn fires once with current snapshot, then on every change |
| `setSourceFile(path) -> Promise` | async | clears selection; loads viewer + atoms; skips atom fetch on viewer load failure |
| `refreshAtoms() -> Promise` | async | re-fetches `/api/selection/atoms` only |
| `setLoader(fn \| null)` | sync (void) | injects the page-supplied viewer loader; `null` detaches |
| `adoptSession({sourceFile, selection}) -> Promise` | async | session-restore path: skips viewer load, accepts pre-validated selection |

Selection editing (client-side, no HTTP except `applyFilter` / `writeLabel`):

| Method | Kind | Notes |
|---|---|---|
| `toggleAtom(index) -> Promise` | sync | flips membership; keeps `selection` sorted-ascending |
| `setSelection(indices) -> Promise` | sync | replaces; dedupes + sorts; silently drops non-numbers |
| `addToSelection(indices) -> Promise` | sync | union |
| `removeFromSelection(indices) -> Promise` | sync | difference |
| `selectAll() -> Promise` | sync | every atom in `state.atoms` |
| `invertSelection() -> Promise` | sync | complement against `state.atoms` |
| `clearSelection() -> Promise` | sync | empties `state.selection` |

Filter drafts + commit:

| Method | Kind | Notes |
|---|---|---|
| `setFilters(filters) -> Promise` | sync | replaces draft array |
| `addFilter(filter) -> Promise` | sync | appends |
| `removeFilter(index) -> Promise` | sync | by position |
| `updateFilter(index, filter) -> Promise` | sync | by position |
| `setCombinator(c) -> Promise` | sync | `"or"` or `"and"` |
| `setMode(mode) -> Promise` | sync | `"click"` or `"filter"`; UI-only |
| `applyFilter() -> Promise` | async | POSTs `/api/selection/eval`; replaces `state.selection` |

Sidecar writes (no change to `state.selection`):

| Method | Kind | Notes |
|---|---|---|
| `writeLabel(target, indices) -> Promise` | async | POSTs `/api/selection/save`; re-fetches atoms on success |

All sync mutators microtask-batch their subscribers; rapid-fire
calls coalesce into one notification per microtask turn.

Internal helpers (`_createStore`, `_fallbackColor`, etc.) are
exported only for tests.

## 9. Tests

  * `tests/test_selection.py` -- Python rule grammar + evaluator
    (unchanged; this is L1 in the Python sense).
  * `tests/test_selection_blueprint.py` -- HTTP contract for the
    five endpoints (unchanged).
  * `tests/test_results_blueprint.py::TestPartialSelectionPanelEndpoint`
    -- partial-id contract (unchanged after the refactor; the
    panel still queries the same ids).
  * `tests/test_selection_measurements_js.py` -- pure-math unit
    tests for `lib/selection/measurements.js` (xyz / distance /
    angle, pickOrder vs geometric fallback, 4+ atoms → null).
  * `tests/test_modify_e2e.py::test_measurement_readout_shows_xyz_distance_angle`
    -- end-to-end pin for the chip overlay on the Modify tab.
  * JS unit tests (future): see `tests/test_atom_selection_js.py`
    once the Playwright harness covers the panel.

## 9b. Selection-driven measurement readout

A tiny module at `lib/selection/measurements.js` reads from the
store + a per-page **positions provider** and emits a one-line
display string for the user:

  * 1 atom selected → `Au #3 — (0.000, 0.000, 0.000) Å`
  * 2 atoms selected → `|H #5 – O #0| = 0.957 Å`
  * 3 atoms selected → `∠H #5 – O #0 – H #6 = 104.5°`
  * 0 or 4+ atoms → `null` (consumer hides the readout)

The vertex for 3 atoms is the user's SECOND click (read off
`state.pickOrder`); if `pickOrder` doesn't match the current
selection set (filter eval, session restore, batch
`setSelection` from a non-click source), the module falls back
to a geometric heuristic — vertex = atom with smallest
sum-of-distances to the other two.  Labels prefer
`atom.atom_name` over `${element} #${index+1}` so a PDB residue
shows residue-aware names like `OE1` instead of `O #142`.

Display conventions (load-bearing — drift between consumers is
the original sin that prompted the module):

  * 1-based indexing in the label (matches viewer overlays).
  * 3-decimal coordinates.
  * 4-decimal distances.
  * 1-decimal angles.
  * `tabular-nums` font feature on the chip so decimals don't
    jitter as the value changes.

**Positions provider** — the page wires
`window.molbuilder.selection.positionsProvider = () =>
number[][]` returning the current xyz array.  The Modify tab's
viewer returns its parsed `state.positions`; future trajectory
and structure inspectors (tasks #299, #300) will return their
own per-frame coords.

**Where the chip lives** — `#selection-measurement-overlay` is
positioned absolutely inside `.viewer-wrap` (which mirrors the
canvas dimensions, including its centred `max-width: 560px`
cap).  Bottom-right of the actual canvas; floats above 3Dmol's
rendering, `pointer-events: none` so it doesn't block clicks.

## 10. Versioning

Schema bumps to `.molstruct.json` (sidecar) are independent of the
JS module boundary — see `molbuilder/parsers/molstruct_json.py` for
the on-disk schema version.  This doc describes the JS state shape;
the store translates to/from sidecar at HTTP time.
