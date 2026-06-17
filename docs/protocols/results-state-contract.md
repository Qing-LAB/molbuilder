# Results-tab state contract

**Status:** v1, design proposal, 2026-06-17
**Scope:** in-memory state of the molbuilder Results-tab inspectors
(`lib/trajectory/core.js`, `lib/spectra/core.js`, and any future
Transport results inspector), plus the server-side parser/cache
contract that feeds them and the layout invariant that keeps the
system-load monitor out of the plot area.

> **This document is the sole source of truth for refresh + memory
> behavior on the `/results` tab.** Any code that mutates inspector
> state, triggers a refresh, or caches parsed output MUST satisfy
> this contract. Reviewers reject "partial-refresh" patterns by
> reference to this doc.
>
> Pointer in `design.md` § 0 (Protocols).

---

## 1. Why this exists

Five parallel audits on 2026-06-17 documented that every refresh
bug users have reported on the `/results` tab traces back to the
same root cause: **state-coherence failures across incremental
updates.** The bug shapes:

- **Derived value not re-derived.** SCF plots freeze mid-step because
  the `noNewContent` geometry-only guard suppresses `makePlots()`
  when SCF iterations grow within the in-flight step.
  Per-frame energy/forces sidebar stays stale on no-data polls. The
  file-picker's in-memory cache hides freshly-generated files
  until next pageshow.
- **Multi-source-of-truth divergence.** `state.watchPath` (closure)
  vs `els.watchPath.value` (DOM input). UI-rendered atom list
  vs the embed's pick-set. Two writers, no reconciliation.
- **Async race leaves torn shape.** A mode-pick during a watchTick
  causes `renderResults` to dispose + remount the 3Dmol viewer
  even though geometry is identical. Plotly.react on mutable array
  references can skip the diff and serve a stale point.
- **Parser emits in-progress frames as if final.** SIESTA's
  `parsers/siesta.py:1495-1544` writes a partial frame at EOF with
  `step_initial_etot` as a fallback energy and an `in_progress=True`
  flag. The frontend ignores the flag and plots the placeholder. A
  full refresh re-parses the now-completed file and the bad point
  vanishes — hence "odd value disappears on Refresh".
- **`scfPollHistory` not cleared by Refresh button.** Refresh calls
  `pollOnce()` directly, missing `loadByPath()`'s state reset
  (`trajectory/core.js:2598-2609` vs `:2678`). Per-iter time
  estimate carries 32 stale samples — up to 8 minutes of lying
  numbers — across what the user thinks is a full refresh.
- **`dispose()` leaks state.** `state.data`, `mtime`,
  `currentFrame`, `scfPollHistory` survive `dispose()` and can leak
  into a re-mount of the same handle. Spectra module-level vars
  (`_loadedStructureText`, `_formDirty`, `_committedStructureFile`,
  `_committedFrozenAtoms`, `_committedRegions`) survive across
  unmount/mount cycles entirely.

Today's mitigations are ad-hoc fingerprint patches (SCF Fix A,
Spectra Fix C/D from 2026-06-17). This contract replaces the
hand-rolled reset bookkeeping with a formal state machine, a
bucketed data model with explicit lifetimes, and three load-bearing
invariants that prevent the bug shapes from returning.

---

## 2. The state machine

Each inspector lives in exactly one of five states. Transitions are
enumerated; nothing else is legal.

```
       ┌──────┐                  ┌─────────┐
       │ IDLE │ ── pick file ──► │ LOADING │
       └──────┘                  └────┬────┘
          ▲                           │ fetch ok,
          │ dispose                   │  run = static
          │                           ▼
          │                      ┌────────┐
          │                      │ LOADED │ ──────┐
          │       ┌──────────────┤        │       │ poll says
          │       │  Refresh OR  └────────┘       │  "ongoing"
          │       │ file-switch                   ▼
          │       │                          ┌──────────┐
          │       └──────────────────────────┤ WATCHING │ ── tick ──┐
          │                                  └─────┬────┘           │
          │                                        │  poll says     │ (re-enters
          │  N consecutive failures               │   "finished"   │ WATCHING)
          │                                        ▼                │
          │                                  ┌────────┐             │
          └──────────────────────────────────┤ ERROR  │◄────────────┘
                                              └────────┘
```

**States and meaning.**

| State | Meaning |
|---|---|
| `IDLE` | No file selected; UI in "pick a file" placeholder |
| `LOADING` | First fetch in flight for a file (or in flight after Refresh / file-switch). Plots cleared, "Loading…" overlay shown. |
| `LOADED` | File rendered, polling NOT active. Either run is finished, or user explicitly stopped watching. |
| `WATCHING` | File rendered, poll timer active. Each tick is an *attempt event*; it never transitions states unless the server says `run_state` changed or N polls failed. |
| `ERROR` | Last fetch failed (network / parse / 404). User sees the error, can retry. Polling is stopped. |

**Transitions and their semantics.**

| Trigger | Allowed from | Action |
|---|---|---|
| pick file (sidebar / picker / `loadByPath`) | any | → LOADING, replace `fileState` empty, abort all in-flight requests, clear `derived`, reset `viewState`, keep `uiPrefs`. |
| Refresh button | LOADED, WATCHING, ERROR | → LOADING, same actions as pick-file with the same path. |
| fetch resolved, run finished | LOADING | → LOADED, populate `fileState`, render. |
| fetch resolved, run ongoing | LOADING | → WATCHING, populate `fileState`, render, start poll timer. |
| fetch failed | LOADING, WATCHING | If failure count < N → stay in current state, log error. If ≥ N → ERROR, stop timer. |
| poll tick | WATCHING | An *attempt event*. Fetch in flight; on resolve, replay this same transition table. No state change unless `run_state` flipped. |
| poll says run finished (2 consecutive ticks) | WATCHING | → LOADED, stop timer. The 2-tick buffer covers the race where the server sees `finished` but the parser is still flushing trailing data — a single tick can lie. |
| dispose / unmount | any | → IDLE, clear all buckets except `uiPrefs` (which persists to sessionStorage). |

**Forbidden:**
- Direct mutation of `fileState` outside a `→ LOADING → LOADED/WATCHING` arc.
- Mutation of state while a fetch is in flight; new fetches *replace*, never patch.
- Any state change from `IDLE` without a fetch.

**Refresh = file-switch with same path.** This is the contract's
single most important rule. Refresh is not a special path; it's
file-switch where the new path equals the old path. Every reset
that file-switch does, Refresh does. Eliminates the half-refresh
class.

**The `transition()` API.** All state changes go through one
function. Signature:

```js
function transition(targetState, payload) {
  // payload depends on targetState:
  //   'LOADING' → { path: string }
  //   'LOADED' | 'WATCHING' → { data: parsed payload }
  //   'ERROR' → { message: string }
  //   'IDLE' → {} (or omitted)
  //
  // 1. apply the reset matrix for (current → target)
  // 2. update state.machine = targetState
  // 3. issue side effects: start/stop poll timer,
  //    fire AbortController, persist uiPrefs to sessionStorage
  // 4. notify any subscribers (events) AFTER state is consistent
}
```

Side-effects fire ONLY after the state mutation is complete; any
listener that re-enters `transition()` does so from a consistent
state, not a half-updated one.

**Snapshot reads.** Render functions (§ Invariant 3) receive
`snap = { fileState, viewState, uiPrefs }`. Because `fileState` is
*replaced* not patched, a shallow read suffices — no deep clone.
The renderer's view of `fileState` is stable for the duration of
the call even if a new fetch resolves mid-render.

---

## 3. Data buckets

State is partitioned into five buckets with disjoint lifetimes. A
field belongs to exactly one bucket; cross-bucket reads go through
read-only snapshots.

```js
state = {
  fileState: {        // REPLACED ATOMICALLY each LOADING→LOADED/WATCHING
    path,             // absolute path of the file shown
    mtime,            // server-reported mtime; the poll key
    format,           // "siesta", "pyscf", "molwatch-merge", ...
    label,            // human-readable file label
    data: {           // parsed payload (frames, energies, scf_history, ...)
      frames, energies, forces, max_forces, lattice, wall_times,
      scf_history, runtime_info, parse_warnings,
      run_state,      // "ongoing" | "finished" | "errored"
      error_message,
    },
  },

  viewState: {        // per-file user interaction; reset on file-switch
    currentFrame,     // playback head index (trajectory)
    selectedMode,     // 1-based mode index (spectra)
    picks,            // selected atom indices
    scrollPos,        // inspect-list scroll position
  },

  uiPrefs: {          // per-session knobs; survives file-switch
    hideFrozen,       // checkbox
    broadeningFWHM,   // spectra
    animSpeed,        // spectra mode-viewer
    animAmplitude,    // spectra mode-viewer
    modeFilter,       // spectra table
    sortColumn,       // spectra table
    sortDir,          // spectra table
  },

  lifecycle: {        // controllers + timers; never user-visible
    pollTimer,        // setInterval handle (WATCHING only)
    pollInFlight,     // bool
    loadAbort,        // AbortController
    pollAbort,        // AbortController
    fetchSeq,         // monotonic counter for the file-identity guard
  },

  derived: {          // computed from fileState; cleared whenever fileState is
    scfPollHistory,   // per-iter wall-clock samples
    iterTimeSamples,
    fingerprints,     // memoized renders
  },
}
```

**Reset matrix.** Each row is a trigger; each column is a bucket.
This table is law: no exceptions, no "I'll just patch this one
field".

| Trigger | fileState | viewState | uiPrefs | lifecycle | derived |
|---|---|---|---|---|---|
| **file-switch / Refresh** (→ LOADING) | empty out | reset | keep | abort all in-flight, clear flags | clear |
| **fetch resolved, run finished** (LOADING → LOADED) | populate with payload | clamp `currentFrame` to `[0, n-1]` | keep | clear in-flight, stop poll timer if running | recompute |
| **fetch resolved, run ongoing** (LOADING → WATCHING) | populate with payload | clamp `currentFrame` to `[0, n-1]` | keep | clear in-flight, START poll timer | recompute |
| **fetch failed, retry budget remains** (stay LOADING/WATCHING) | keep | keep | keep | clear in-flight | keep |
| **fetch failed N times** (→ ERROR) | keep last good | keep | keep | clear in-flight, stop poll timer | keep |
| **poll says run finished** (WATCHING → LOADED, after 2 consecutive ticks) | replace `.data` | clamp `currentFrame` | keep | stop poll timer | recompute |
| **dispose / unmount** (→ IDLE) | clear | clear | persist to sessionStorage | clear all | clear |
| **mount** (IDLE) | empty | empty | restore from sessionStorage | empty | empty |
| **watch tick (no new data)** | keep | keep | keep | keep | keep |
| **watch tick (new SCF iter same step)** | replace `.data` (atomic) | keep | keep | keep | recompute |
| **watch tick (new frame)** | replace `.data` | clamp `currentFrame` | keep | keep | recompute |
| **frame-slider scrub** | keep | mutate `currentFrame` | keep | keep | keep |
| **user toggles hide-frozen** | keep | keep | mutate | keep | keep |

**Mutation rules.**
- `fileState`, `lifecycle`, `derived` MUST be mutated only inside
  `transition()`. The reset matrix is the complete list.
- `viewState` MAY be mutated directly by event handlers (frame
  scrub, mode pick, atom pick) — these are intra-state events,
  not state transitions. Handlers must NOT mutate
  `fileState`/`lifecycle`/`derived` as a side effect.
- `uiPrefs` MAY be mutated directly by handlers (toggle, slider
  change). Persistence to sessionStorage happens on dispose or
  on visibilitychange; handlers don't need to call it.

**Why this matrix matters.** The audits found that every refresh
bug was a row in some implicit version of this table that the code
got wrong: Refresh didn't clear `derived.scfPollHistory`; dispose
didn't clear `fileState`; the spectra `state.watchPath` lived in
both `lifecycle` *and* DOM and they drifted. The matrix names the
intended behavior; the test suite enforces it (§ 12).

---

## 4. The three load-bearing invariants

### Invariant 1 — file-identity guard

Every async resolution checks the response's path against
`state.fileState.path` and `state.lifecycle.fetchSeq` before
applying. Late responses from a prior file can never write into the
current file's view.

```js
// At fetch site:
const mySeq = ++state.lifecycle.fetchSeq;
const myPath = state.fileState.path;
fetch(...).then(r => {
  if (state.lifecycle.fetchSeq !== mySeq) return;  // superseded
  if (state.fileState.path !== myPath) return;     // file switched
  applyNewData(r);
});
```

**Eliminates:** spectra watchTick-after-file-switch race
(`spectra/core.js:998` only aborts the in-flight fetch; the
`.then()` continuation still runs unless `abort()` triggered first
— file-identity check is the belt-and-braces).

### Invariant 2 — in-progress frame filter

The parser tags partial frames with `in_progress=true`
(`parsers/siesta.py:1543` already does this; `pyscf.py` should
match). Render code FILTERS in-progress frames from plots:

```js
function plottableFrames(data) {
  return (data.frames || [])
    .map((f, i) => ({frame: f, idx: i}))
    .filter(({frame}) => !frame.in_progress && frame.energy != null);
}
```

The frame still appears in the inspect-list with a "computing…"
badge — the user knows it exists, but the energy plot doesn't
include a number the parser couldn't actually read. When the parser
later finalizes the frame and drops the flag, the next render
includes it in the plot.

**Frame is canonical.** Per-axis arrays (`data.energies[]`,
`data.max_forces[]`) become DERIVED views, not parallel sources of
truth: plots iterate `data.frames` and read `frame.energy`,
`frame.max_force`. This makes the in-progress filter complete in
one place; before this contract, `data.energies` could carry the
preamble fallback while `data.frames[i].in_progress` was true, and
the filter missed.

**Eliminates:** the "odd value disappears on Refresh" bug class.

### Invariant 3 — render-with-snapshot

Render functions never read from `state` directly. They receive a
read-only snapshot of `{fileState, viewState, uiPrefs}` as their
argument. This makes "render fires twice in a tick" deterministic
— both calls see the same snapshot — and makes the file-identity
guard the only place a stale fetch can apply.

```js
function makePlots(snap) {
  // snap = { fileState: {...}, viewState: {...}, uiPrefs: {...} }
  // No reads from closure `state` here.
  const plottable = plottableFrames(snap.fileState.data);
  Plotly.react("energy-plot", [{
    x: plottable.map(p => p.idx),
    y: plottable.map(p => p.frame.energy),  // per-frame, never data.energies[]
  }], layout);
}
```

**Eliminates:** stale-during-tick mixed reads; the Plotly.react
mutable-array-ref bug (§ 8 makes copies explicit).

---

## 5. Refresh policy

There is exactly **one** code path for "refresh this file": the
file-switch transition with the current path. Implementations:

| Trigger | What it does |
|---|---|
| User picks a different file in the sidebar | `transition('LOADING', {path: newPath})` |
| Refresh button (af49560) | `transition('LOADING', {path: state.fileState.path})` |
| `EVENT_REFRESH_REQUESTED` from file-picker | same as Refresh button |
| `pageshow` / `visibilitychange` on hidden tab | issue one poll only — does NOT transition; this is an attempt-event inside WATCHING |
| poll timer tick | issue one poll only — attempt-event inside WATCHING |

`pollOnce()` MUST NOT be a public entry-point. The current
`EVENT_REFRESH_REQUESTED` handler at `trajectory/core.js:2598-2609`
that calls `pollOnce()` directly is the bug; it stays as a private
helper inside the WATCHING tick, never the Refresh button's
implementation.

---

## 6. Server-side contract

### Parser

- The parser MAY emit frames with partial data. It MUST tag them
  with `in_progress=true` and MUST NOT invent an energy: if no SCF
  cycle has reported one, `energy` is `null`. The frontend filters
  these from plots (Invariant 2).
- The parser MUST NOT use `step_initial_etot` (or any preamble
  banner) as a frame's `energy`. That field stays in `runtime_info`
  for display; it's not a frame energy.
  - **Migration: `parsers/siesta.py:1495-1544` reset.** Change the
    in-progress fallback to `energy=None`; emit
    `step_initial_etot` separately as `runtime_info.initial_etot`.

### Cache

`_MERGE_PARSE_CACHE` in `web/blueprints/watch.py:524` becomes:

```python
_MERGE_PARSE_CACHE: LRU[str, Tuple[Tuple[mtime, size], Dict[str, Any]]]
```

- Key remains absolute path.
- Cache value's key tuple is `(mtime, size)`. 1-second mtime
  granularity systems (NFS, FAT) get torn-read defense for free —
  any size change invalidates.
- LRU bound: 64 entries. Eviction is FIFO once the bound is hit;
  deleted files age out naturally.
- A parsed snapshot is NOT cached if the read completed less than
  200 ms after the file's mtime — the file was still being written.
  The poll will re-parse on the next tick.

### `/api/watch/data` response

- MUST include `run_state` ∈ `{"ongoing", "finished", "errored"}`.
- MUST include `path` (echo of the requested file) — the
  file-identity guard reads this.
- MUST set `changed: false` when mtime matches the request param;
  the body MAY omit `data` in that case.

---

## 7. Per-inspector mapping

The state machine + buckets are shared. Per-inspector field
assignments:

### Trajectory (`lib/trajectory/core.js`)

| Bucket | Fields |
|---|---|
| fileState | path, mtime, format, label, data (frames, lattice, scf_history, runtime_info, parse_warnings, run_state, error_message). Per-axis arrays `energies`/`max_forces`/`wall_times` allowed AS DERIVED VIEWS during migration, but plots MUST read `frame.energy` / `frame.max_force` per Invariant 2. |
| viewState | currentFrame, picks, firstFit |
| uiPrefs | hideFrozen, playbackSpeed, playbackLoop |
| lifecycle | pollTimer, pollInFlight, loadAbort, pollAbort, fetchSeq |
| derived | scfPollHistory (max 32 samples; cleared on every fileState replace), perIterEstimate, plotFingerprint |

### Spectra (`lib/spectra/core.js`)

| Bucket | Fields |
|---|---|
| fileState | path, mtime, results (modes, runtime_info, equilibrium, free/frozen idxs, phase_*) |
| viewState | selectedMode, picks |
| uiPrefs | modeFilter, sortColumn, sortDir, broadeningFWHM, animSpeed, animAmplitude |
| lifecycle | watchTimer, watchInFlight, watchAbort, loadAbort, renderAbort, fetchSeq, watchErrors |
| derived | resultsFingerprint, computedEnvelope |

**Canonical path location.** The legacy `state.watchPath` field is
retired. The canonical path is `state.fileState.path`. The
`els.watchPath.value` DOM input is read ONCE at transition entry
(when the user clicks Start-watching / Load-once / Refresh) and
copied into `fileState.path`; render code and `watchTick` read
from `fileState.path` thereafter. This is the source-of-truth fix
for the spectra DOM-divergence bug (audit § 1).

**Module-level vars NOT covered by this contract.** The spectra
closure currently has `_loadedStructureText`, `_formDirty`,
`_committedStructureFile`, `_committedFrozenAtoms`,
`_committedRegions` outside `state`. These are calculation-tab
form state — owned by `workspace-contract.md`, not this contract.

The only obligation this contract imposes on them: `dispose()`
MUST clear them so they don't leak across remounts. Where they
ultimately live (a future spectra-form contract document) is out
of scope here. They get a `TODO(state-contract)` comment in code
pointing at the workspace contract; nothing else in this contract
touches them.

---

## 8. Plot update policy

`Plotly.react` is the safe default but expensive — it reflows the
full layout and resets zoom. Use a graded ladder:

| Change | Call |
|---|---|
| New trace count, new axis labels | `Plotly.react(el, traces, layout, config)` |
| Same traces, only data arrays changed | `Plotly.restyle(el, {x: [...], y: [...]}, traceIndices)` |
| Same traces + data, only layout | `Plotly.relayout(el, layoutPatch)` |
| Frame appended in WATCHING | `Plotly.extendTraces(el, {x: [[newX]], y: [[newY]]}, [0])` |

**Mutable-reference rule.** All arrays passed to Plotly MUST be
fresh allocations or `.slice()` copies, never direct references to
`fileState.data.*` arrays. Plotly memoizes by identity; aliasing a
mutated array breaks the diff.

---

## 9. Layout invariant (the monitor panel)

`#system-load-monitor` uses `position: fixed; bottom: 0` and
overlaps content. The contract:

- The monitor's collapsed height is the FLOOR; nothing in the
  scrollable container area MAY have layout that places it under
  the monitor.
- Implementation: `body` (or the results-main scroll container)
  gets `scroll-padding-bottom: var(--monitor-height)`. The monitor
  reports its current height (collapsed vs expanded) via a CSS
  custom property `--monitor-height` it sets on `:root`.
- The monitor is COLLAPSED BY DEFAULT on first visit
  (`system-load-monitor.js:358`'s `applyCollapsed("0")` becomes
  `applyCollapsed("1")`). Users opt in to the expanded strip; it
  does not opt them in.
- When expanded but not hovered, the monitor's region uses
  `pointer-events: none` on the backdrop so clicks pass through to
  plots beneath. The strip itself stays clickable.

This is a CSS-only fix and can ship independently of the state
refactor.

---

## 10. Migration plan

Four PRs in order:

### PR 1 — Monitor panel CSS (small, no risk)

Files: `styles/system-load-monitor.css`, `lib/system-load-monitor.js`.
Changes per § 9. Test: L1 CSS-pin that `body` has
`scroll-padding-bottom` and that `:root --monitor-height` is set.

### PR 2 — Trajectory state-machine refactor

Files: `lib/trajectory/core.js`, `tests/test_live_poll_invariants_audit.py`,
new `tests/test_results_state_contract_js.py`.

Steps:
1. Introduce the bucketed `state = { fileState, viewState, uiPrefs, lifecycle, derived }` shape; migrate existing fields in.
2. Add `transition(target, payload)` as the single entry-point for state changes. All call sites that previously poked `state.*` directly route through it.
3. Wire Refresh button → `transition('LOADING', {path: state.fileState.path})`.
4. Wire `dispose()` → `transition('IDLE')` + sessionStorage flush of `uiPrefs`.
5. Add file-identity guard (§ Invariant 1) at fetch resolution.
6. Add `plottableFrames()` filter (§ Invariant 2) in `makePlots`.
7. Convert all render functions to `(snap) => ...` snapshot signature (§ Invariant 3).
8. Convert Plotly calls per § 8.

Fixes per-iter clock on Refresh, stale readout, odd-Etot, plot
memoization. Pre-existing 64 trajectory tests must still pass; new
contract tests in § 12 pin the state matrix.

### PR 3 — Spectra state-machine refactor

Mirror PR 2 in `lib/spectra/core.js`:
- Steps 1-4 (bucketed state, `transition()`, Refresh wiring,
  dispose) per PR 2.
- Step 5: file-identity guard at watchTick + loadByPath
  resolution (Invariant 1). Retire `state.watchPath` in favor of
  `state.fileState.path` (§ 7 spectra notes).
- Step 6: `plottableFrames()` not needed for spectra (no
  per-frame plot); Invariant 2 applies in PR 4 (parser side
  only).
- Step 7: snapshot signature for `renderResults`,
  `renderSpectrumChart`, `renderModesTable`, `renderESPanel`,
  `renderModeViewer` (Invariant 3).
- Step 8: Plotly ladder per § 8 — `renderSpectrumChart` becomes
  `.restyle()` for bar-height-only updates (preserves zoom).

Module-level structure-text vars (`_loadedStructureText`, etc.)
get `TODO(state-contract)` comments; dispose clears them; full
ownership moves to a future workspace contract per § 11.

Fixes mode-pick-during-watchTick race (Invariant 1), Plotly
zoom-reset on activity update (§ 8), `watchPath`/DOM divergence.

### PR 4 — Parser `in_progress` + server cache

Files: `parsers/siesta.py`, `parsers/pyscf.py`,
`web/blueprints/watch.py`, `tests/test_watch_cache.py`.

Changes per § 6. Test that the cache evicts at LRU bound, that the
200ms freshness gate kicks in, that `in_progress=true` frames have
`energy=None`.

### Order

PR 1 ships first (independently valuable, no state dependencies).
PR 2 lands the contract for trajectory; PR 3 mirrors to spectra.
PR 4 closes the parser-side loop. Each PR includes its own
contract tests so a regression in any phase fails immediately.

---

## 11. What this contract does NOT cover

- The calculation-tab form state (Spectra Generate, Transport
  Generate, SIESTA/PySCF Generate). Those have their own
  workspace + form-dirty contracts (`workspace-contract.md`).
- Workspace/structure state in `/molbuilder`. That's
  `workspace-state.md` + `workspace-contract.md`.
- The file-picker's in-memory cache. Out of scope here because the
  picker is sidebar-level, not inspector-level. If the picker
  cache bug (audit 5) needs a fix, it gets its own contract entry
  in `projects-sidebar.md` or a new sidebar-cache doc.
- Transport results — task #478 is pending and will plug into this
  contract when it lands.

---

## 12. Tests that pin the contract

`tests/test_results_state_contract_js.py` (new) holds source-text
pins for the load-bearing properties:

1. **Reset-matrix pin.** Each row of the table in § 3 maps to one
   test that grep-confirms the transition handler resets the right
   buckets.
2. **File-identity guard pin.** Every fetch-resolution site has a
   `fetchSeq` or `path` check before applying.
3. **In-progress filter pin.** `plottableFrames` is called by
   every render that produces an energy/force plot; the regex
   confirms across both inspectors.
4. **Render-with-snapshot pin.** Render targets:
   - trajectory: `makePlots`, `renderScfProgress`,
     `_renderRuntimeInfo`, `_renderParseWarnings`,
     `updateInspectPanel`, `refreshForcesStatus`
   - spectra: `renderResults`, `renderSpectrumChart`,
     `renderModesTable`, `renderESPanel`, `renderModeViewer`
   Each takes a `snap` argument; the test rejects closure reads of
   `state.fileState` / `state.viewState` / `state.uiPrefs` inside
   these function bodies.
5. **Plot-update policy pin.** Plotly call sites grep out to the
   right ladder rung for their change type.
6. **Refresh = file-switch pin.** The Refresh button handler
   produces the same transition() call shape as file-switch.
7. **Bucket-disjoint pin (per inspector).** Within each inspector,
   no field name appears in two buckets. Cross-inspector duplicates
   (`path` in both trajectory and spectra fileState) are allowed
   — they're different inspector instances, not aliases.
8. **`scfPollHistory` reset pin.** Cleared on every `→ LOADING`
   transition; never written outside `derived`.
9. **Frame-canonical-source pin.** Plot trace-builder grep MUST
   show `frame.energy` / `frame.max_force` reads, not
   `data.energies[i]` / `data.max_forces[i]` array indexing.
   Eliminates the parallel-array drift Invariant 2 closes.
10. **Abort-controller pin.** Every `→ LOADING` transition MUST
    abort the prior `loadAbort` AND `pollAbort` before issuing the
    new fetch. Pin: the `transition('LOADING', ...)` body contains
    both `.abort()` calls.

These are L1 source-text tests (cheap, run in ms). Behavioral L3
tests for the actual reset-on-Refresh fix live in the existing
`test_live_poll_invariants_audit.py`.

---

## 13. Open question for review

**`uiPrefs` persistence key.** Today `hideFrozen` is in
sessionStorage; spectra's broadening lives in DOM only.
Proposed: sessionStorage for ALL `uiPrefs`, single key
`molbuilder.results.uiPrefs.v1` holding a JSON dict.
Alternative: `localStorage` (survives browser restart) — more
convenient but conflicts with per-tab independence (two
`/results` tabs shouldn't share prefs).
**Reviewer answer needed before PR 2.**

### Implementation-time decisions (already settled in this doc)

- **`firstFit` (trajectory).** `viewState`. Per-file, reset on
  file-switch.
- **`fetchSeq` granularity.** Per-inspector (one counter per
  mount). The data shape in § 3 puts it in `state.lifecycle`,
  which is per-inspector.
- **WATCHING → LOADED idle buffer.** M=2 consecutive
  `run_state == "finished"` ticks (matrix § 3, transition § 2).
- **Parser blast radius (PR 4 prerequisite).** Before PR 4:
  audit consumers of `frame.energy` (CLI parser, sidecar
  generator, `molbuilder/parsers/*`). Any reader that depended on
  the `step_initial_etot` fallback gets explicit `None` handling.
  Not a contract-design question.

### Extending the contract to a new inspector

To add a new results-tab inspector (e.g., Transport when task
#478 ships): fill in a per-inspector mapping table like § 7's;
implement `transition()` with the same five-bucket shape;
register render functions with the snapshot signature
(Invariant 3); add the inspector's render targets to test #4's
list. No changes to § 2–§ 6 should be required — the contract is
inspector-agnostic.

---

## 14. Process

Per `design.md` § 7, this doc lands first as a proposal; per-PR
specs (PR 2's trajectory refactor, etc.) point at sections of this
doc for their contract surface. When a PR lands, the section it
implements gets struck through with a SHIPPED marker; the
implementation details migrate to the inspector-specific docs in
`docs/protocols/` or stay inline in code.

The 2026-06-17 audit transcript (5 parallel Explore agents) is the
historical record of why this refactor is necessary. Each finding
cited in § 1 is a marker for what symptom forced the contract;
collectively they argue the architectural debt has shipped enough
bugs to warrant the rewrite.
