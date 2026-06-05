# Inspector registry — `mount` / `dispose` / `isResult` / `resultCategory`

`lib/inspectors/registry.js` is the dispatch layer that decides
which inspector mounts when the user picks a file in the projects
sidebar. Each inspector is a self-contained module
(`lib/inspectors/{source,structure,trajectory,spectra}.js`) that
self-registers on script load and is the sole owner of its
mount-host DOM, its data fetch, and its dispose.

This doc is the sole source of truth for the inspector contract.
The `/results` dispatch architecture (which file picks which
inspector + how the dropdown groups them) is in
[`results-tab.md`](results-tab.md); this doc covers the
inspector-side contract.

---

## 1. The contract

Every inspector exports an object with this shape:

```js
{
  name:         string,          // "source" | "structure" | "trajectory" | "spectra"
  displayName:  string,          // user-facing label
  isResult:     boolean,         // see § 2
  match:        (file: string) => boolean,
  resultCategory: (file: string) => string,   // optional; defaults to displayName
  mount:        (host: HTMLElement,
                 file: string,
                 ctx: MountContext) => { dispose: () => void },
}
```

### `mount(host, file, ctx) → handle`

Take ownership of `host` (an empty `<div>` provided by the
registry). Render whatever UI the inspector needs **inside**
`host`. Return a handle with at least a `dispose()` method.

- The registry calls `dispose()` BEFORE mounting the next
  inspector. dispose MUST cancel all in-flight HTTP requests,
  stop polling timers, remove window-level event listeners, and
  clear any 3Dmol / Plotly bookkeeping.
- After `dispose()` returns, the host's contents are no longer
  the inspector's. The registry may clear `host.innerHTML` or
  hand it to the next inspector.

### `dispose() → void`

Documented in detail in
[`playwright-tests.md`](playwright-tests.md) § A6 "Dispose
contract". Must be idempotent (a second call is a no-op) and
never throw.

---

## 2. `isResult` + `resultCategory` — result-file picker

The `/results` tab-level file picker
(`lib/results/file-picker.js`) filters the directory listing to
show only files an inspector wants to claim AS A RESULT. Two
properties drive this:

- **`isResult: true`** — the inspector handles result files (e.g.
  trajectory mounts `.molwatch.log` and `.out`; spectra mounts
  `.spectra.json`).
- **`isResult: false`** — the inspector is a catch-all viewer
  that should NOT pollute the result-file dropdown (e.g. the
  source inspector matches `.fdf` / `.py` / `.json` / `.txt` /
  `.md` / `.log` as a generic text fallback; if it claimed
  result-file status, every input config would appear in the
  dropdown).

The picker calls `registry.pickResult(file)` (which honors
`isResult`) to decide whether a file lands in the dropdown. The
tab-level dispatcher calls `registry.pick(file)` (which doesn't)
to decide which inspector mounts when the user picks a file.

`resultCategory(file)` is rendered as the `<optgroup>` label in
the picker. Multi-file inspectors can return different categories
per file (e.g. `trajectory.resultCategory(".out")` returns
"SIESTA optimization"; `.molwatch.log` returns "PySCF
optimization"); inspectors that don't override default to
`displayName`.

---

## 3. Registration order is load-bearing

`registry.pick(file)` returns the FIRST inspector whose `match`
returns true. Compound-extension inspectors (`.molwatch.log`,
`.spectra.json`) **MUST register BEFORE** generic-extension
inspectors (the source inspector matches `.log` / `.json` as a
fallback).

The template's `<script>` order pins this:

```html
<!-- Specific compound-extension first -->
<script src=".../lib/inspectors/trajectory.js"></script>
<script src=".../lib/inspectors/spectra.js"></script>
<!-- Then specific-but-not-compound -->
<script src=".../lib/inspectors/structure.js"></script>
<!-- Catch-all source LAST -->
<script src=".../lib/inspectors/source.js"></script>
```

Pinned by
`tests/test_results_blueprint.py::TestInspectorRegistrationOrder`.

---

## 4. Refresh contract — `pageshow` + `visibilitychange`

Any inspector that **caches loaded data** (trajectory frames,
spectra results, source-file chunks) MUST re-fetch on tab
re-entry. Without this, the user sees stale data after navigating
to another tab and back (or after a bfcache restore). The
manifestation that surfaced this contract is documented in
[`design.md`](../design.md) § Decisions log under the
2026-06-02 /results stale-dropdown entry (#192).

### Implementation pattern (mirror the file-picker)

```js
function _onPageShow(_evt) {
    if (/* loaded? */) refreshFn();
}
function _onVisibilityChange(_evt) {
    if (document.visibilityState === "visible" && /* loaded? */) {
        refreshFn();
    }
}
_on(window,   "pageshow",         _onPageShow);
_on(document, "visibilitychange", _onVisibilityChange);
```

Route through the inspector's `_on()` cleanup helper so dispose
auto-removes them. Guard on a "is something loaded?" state flag
(`state.mtime !== null`, `state.results !== null`, etc.) so a
mounted-but-empty inspector doesn't fire spurious requests.

### Per-inspector implementations

| Inspector | Refresh function | Loaded-state guard |
|---|---|---|
| trajectory | `pollOnce()` | `state.mtime !== null` |
| spectra | `loadByPath()` | `state.results !== null && els.watchPath.value` |
| source | (none — fetches in chunks on user scroll; reload-button is a future task) | — |
| structure | (none — single fetch on mount; small payload) | — |

Pinned by
`tests/test_inspector_pageshow_refresh_e2e.py` (2 tests, audit
task #194, 2026-06-02).

---

## 5. The MountContext (`ctx`)

The registry passes a `ctx` parameter to `mount()` carrying
helpers the inspector commonly needs but should NOT reach for
directly:

```ts
type MountContext = {
  readFile(path: string): Promise<{ok, text, mtime} | {ok:false, error}>,
  // Future: readRange, etc.  See projects-sidebar.md § 5.4 for
  // the canonical public surface; ctx is a thin pass-through.
}
```

`ctx.readFile` is a thin wrapper over `projects.readFile` —
inspectors should prefer `ctx.readFile` over reaching into
`window.molbuilder.projects` directly so future cross-cutting
middleware (caching, telemetry) goes through one chokepoint.

For range reads the canonical path is
`window.molbuilder.projects.readRange` (added 2026-06-02 in
#189); the source inspector goes through it. `ctx` does not
currently expose a `readRange` helper; it will when a second
inspector needs range reads.

---

## 6. Trajectory inspector — internal contract

The trajectory inspector core (`lib/trajectory/core.js`) is the
most complex inspector and has its own internal contracts that
existed before the registry was introduced. These contracts are
preserved here (migrated from the archived `tabs/watch.md` doc)
because the same code powers both the legacy `/watch` page and
the active mount on `/results`.

### 6.1 Inspector partial layout (`_trajectory_inspector.html`)

The DOM scaffold injected by the partial:

- **Row 1 (`.viewer-row`)** — a 2-column grid: the 3Dmol viewer
  on the left and a `.controls` aside on the right. Both columns
  are locked to the same height via the `--viewer-height` CSS
  variable (`clamp(360px, 52vh, 500px)`) so the layout is
  responsive without one column stretching the row.

  The controls aside is **tabbed** (not stacked): **Style /
  Overlays / Playback** are a horizontal tab bar with one panel
  visible at a time. Above the tabs sits an always-visible
  **frame strip** with the frame counter, prev/play/pause/next
  buttons, and the frame slider. The most-used controls stay
  reachable regardless of which tab is open.

  The Playback tab carries a **"Save current frame as XYZ"**
  button (`#save-frame`). Disabled until a file with frames is
  loaded; clicking it builds a standard 4-column XYZ from
  `state.data.frames[state.currentFrame]` and triggers a browser
  download. The comment line records the source engine, the step
  index from `state.data.iterations`, and the energy in eV when
  known. Filename is `<label>_step<N>.xyz`. This is the handoff
  point to downstream pipelines (tunneling-gap construction,
  transport calc) that want a single static structure rather
  than the live trajectory.

- **Row 2 (`.plots-row`)** — two Plotly canvases (energy vs step,
  max force vs step).

- **Row 3 (`.scf-row`, engine-agnostic, hidden when empty)** —
  a banner summarising the current opt step + SCF cycle plus two
  Plotly canvases (SCF energy + a residual within the current
  step). Visible iff `state.data.scf_history` is non-empty. Both
  engines populate this row via the same `scf_history` schema;
  the UI adapts three things by engine (§ 6.2 below).

**Mobile breakpoints**: 980 px collapses every plot row to single
column. 640 px tightens header + plot heights.

### 6.2 Engine-specific UI adaptation

The trajectory inspector renders the same DOM for SIESTA / PySCF
/ molwatch but adjusts three labels by engine. Source of truth
is `renderScfProgress()` in `lib/trajectory/core.js`.

| UI element | SIESTA | PySCF | Unknown engine |
|---|---|---|---|
| Banner title (`#scf-title`) | "SIESTA DFT SCF progress" | "PySCF SCF progress" | "SCF progress" |
| Step label | "CG/MD step" | "Geom-opt step" | "Opt step" |
| Residual axis | `dHmax` (eV) | `|g|` (eV/Å) | data-driven sniff |

SIESTA only implements Kohn-Sham DFT, so calling it just "SCF"
is correct but underspecified. PySCF supports both HF (RHF/UHF)
and DFT (RKS/UKS) and the parser doesn't currently extract which
one a given log used, so the generic "SCF" label is correct for
both.

**Residual selection is data-driven**: a key sniff on
`scf_history[-1][0]` chooses between `gnorm` (PySCF) and `dHmax`
(SIESTA), so future parsers that expose either set of keys work
without UI changes.

The HTML template starts with the generic "SCF progress" text in
`#scf-title` so the placeholder is meaningful before any file is
loaded; `renderScfProgress()` rewrites it on every refresh.

### 6.3 State invariants

`state` (a single JS object inside the trajectory core) holds:

```js
state = {
    data:         <last parsed payload | null>,
    mtime:        <float | null>,
    format:       "siesta" | "pyscf" | "molwatch" | null,
    label:        "<parser label>" | null,
    currentFrame: <int>,
    pollTimer:    <interval id | null>,
    firstFit:     <bool>,         // re-fit camera on a fresh structure
    // Playback timer + pick mirror live inside the embed now (#246):
    // playback is driven by the embed's animation loop, and picked
    // indices are read on demand via _handle.getPickedIndices().
    loadAbort:    <AbortController | null>,
    pollAbort:    <AbortController | null>,
}
```

**Invariants:**

- On a successful `/api/watch/load`, `state.data / mtime / format
  / label` are replaced atomically.
- `state.currentFrame` is preserved across refreshes when the
  user has scrubbed away from the end; clamped to the new last
  frame if the trajectory grew.
- When the user IS at the last frame, refreshes advance the
  frame index to the new last frame so live-watching feels live.
- `loadAbort` cancels in-flight `/api/watch/load`; `pollAbort`
  cancels in-flight `/api/watch/data`. `dispose()` aborts both.

### 6.4 Polling cadence

- Active polling timer interval: `POLL_MS` (default 15 000 ms).
- Each tick: `GET /api/watch/data?mtime=<state.mtime>`.
- Server-side: if mtime unchanged, returns `{changed: false}`
  and the front-end refreshes only the "Up to date — N frames"
  status text.
- When `data.changed`, `applyNewData(r)` rebuilds the model,
  frames, plots.
- `pageshow` / `visibilitychange` handlers (§ 4) call
  `pollOnce()` immediately on tab re-entry so a 15 s wait isn't
  imposed after a bfcache restore (audit #194).

### 6.5 Load button — dual mode (legacy `/watch` only)

The trajectory loader on the legacy `/watch` page has two
behaviours, branching on the path field's content. The
inspector-on-`/results` does NOT use this loader — it receives
the file path through `opts.file` at mount time — but the dual
mode is still wired in the core for `/watch`'s benefit:

- **Path field has text**: POST `{path}` as JSON to
  `/api/watch/load`. Server reads from disk; front-end starts a
  polling timer at 15 s intervals (live-watching mode).
- **Path field empty**: trigger the hidden `<input type="file">`.
  When the user picks a file, upload as `multipart/form-data` to
  `/api/watch/load`. The path field updates to
  `(uploaded) <filename>` for clarity. Polling timer is
  **stopped** because uploaded files don't change on disk.

Pressing Enter in the path input triggers Load.

### 6.6 Status messages

- Single-line for normal updates: `"Loaded N siesta frames —
  mtime HH:MM:SS."`
- Multi-line allowed for errors (e.g. unsupported-format hints).
  The status `<span>` has `white-space: pre-line` so newlines in
  the server's error message render correctly.

### 6.7 Forbidden patterns (trajectory inspector front-end)

In addition to the cross-cutting front-end conventions in
[`web-api.md`](web-api.md) § 11.4, the trajectory inspector must
NOT:

1. **Use `innerHTML` for any user-controlled string.** Everything
   goes through `textContent` to prevent XSS via parser output
   (e.g. a malicious filename in `r.uploaded_filename`).
2. **Continue polling after an upload** — uploaded files don't
   change on disk, the timer would burn requests for nothing.
3. **Retry a failed `/api/watch/load` automatically.** The user
   clicks Load again to retry; auto-retry would hammer a known-
   bad path.
4. **Pin the 3Dmol library at `https://3Dmol.org/build/...`** —
   that URL serves a moving target. Use the cdnjs pinned URL
   `https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js`
   (also see web-api.md § 11.4).

---

## 7. Test coverage

| Test file | Layer | Coverage |
|---|---|---|
| `test_inspector_registry_e2e.py` | Playwright | Registry contract — pick / pickResult / mount / dispose ordering; listener add/remove balance after mount → dispose |
| `test_source_inspector_e2e.py` | Playwright | Source inspector — pagination, head/tail mode, dispose cleanup (10 tests) |
| `test_inspector_pageshow_refresh_e2e.py` | Playwright | Trajectory inspector — pageshow + visibilitychange fire `/api/watch/data` (2 tests) |
| `test_results_file_picker_e2e.py` | Playwright | File-picker contract — second-visit refresh + bfcache + external-mtime change (4 tests) |
| `test_trajectory_inspector_partial.py` | HTTP | Trajectory partial DOM-id set (sanity, not full e2e) |

---

## 8. Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-16 | Inspector registry replaces per-tab dispatch logic; each inspector self-registers + exposes `match` / `mount` / `dispose`. | Pre-registry the trajectory + spectra views had hand-coded dispatch glued to `/watch` and `/spectra` URL routes; lifting the contract here lets `/results` route every result file through ONE picker. |
| 2026-06-01 | Add `isResult: bool` + `resultCategory(file): string` to the contract; the `/results` file-picker filters by `isResult` and groups by category. | Pre-2026-06-01 the picker only saw `.out` / `.molwatch.log` via a SIESTA-specific endpoint; promoting to a registry-driven N→1 collapse needed inspectors to opt in to "this is a result file" + name their category. |
| 2026-06-02 | `pageshow` + `visibilitychange` handlers MUST refresh any cached-data inspector. Pattern documented in § 4; trajectory + spectra ported. | Closing the cached-data half of the /results stale-dropdown bug class (#192). bfcache restore + tab re-focus without these handlers leaves stale UI; the fix mirrors the `lib/results/file-picker.js` pattern. |
