# Embedded MolViewer — the standard structure viewer component

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.

`window.molbuilder.viewer.embed(host, opts) → handle` is molbuilder's
**standard embeddable structure viewer**. Every tab and inspector
that needs to show a 3-D molecular structure drops it into a host
DOM element, supplies XYZ or PDB data plus a small declarative
options object, and lets the viewer maintain the drawing — canvas,
standard control knobs, info line, animation strip and all.

This file is the **sole source of truth** for the unified viewer's
contract. The implementation lives in `lib/mol-viewer-embed.js`;
the CSS in `lib/mol-viewer-embed.css`. Both files MUST conform to
this document; any deliberate divergence updates this document
first, code second.

**How to use this document for implementation review.** Every
behavior the embed exhibits, every API method on the handle, every
error code, every test affordance — all are specified here.
Implementation review compares code against doc clauses, not the
other way around. Section anchors (§ 2.5, § 5.3, etc.) are stable
within a major revision and cited in code comments where a clause
is realised.

---

## Table of contents

1. Mission
2. Isolation contract
3. API surface
4. Lifecycle and state transitions
5. Error model
6. Card structure
7. Usage patterns
8. Consumer migration map
9. Testing affordances
10. Test coverage
11. Decisions log

---

## 1. Mission

> The viewer takes an input of XYZ or PDB data, and a few options
> (style, axis, label, arrow, etc.), then maintains the drawing.
> It can be embedded as a card/panel where needed and the data is
> supplied to it through the API.
>
> The viewer is not just the white 3D display but a module with a
> consistent UI as a whole — a card that contains the canvas AND
> the typical control knobs. External modules attach additional
> controls next to it; those external controls drive the viewer
> through the API.

The viewer:

- **Owns its drawing.** The host never reaches into 3Dmol directly.
  All mutations cross the API boundary as handle method calls.
- **Owns its chrome.** Style picker, labels toggle, axes toggle,
  reset view, screenshot, background and data export live inside
  the viewer card. Every tab gets the same controls, the same
  look, the same placement.
- **Is data-in-API.** Structure text and options come in through
  the embed call; the viewer never fetches.
- **Is declarative.** Options describe *what* the user should see
  (axes on, atom indices visible, force arrows present); the
  viewer handles *how* (3Dmol primitives, shape lifecycle, render
  scheduling).
- **Is embeddable.** Lives inside any host DOM element as a
  self-contained card. Multiple instances per page are supported;
  each owns its own state.
- **Is reactive.** After mount, the host updates state through
  handle methods (`setStructure`, `setStyle`, `setAxes`, …). The
  viewer re-renders. The host never schedules a render.

---

## 2. Isolation contract

The viewer is a self-contained module with a strict interface
boundary. Everything inside the boundary is the viewer's job;
everything outside is the host's. Crossing the boundary requires
either a method on the handle (host → viewer) or a callback in
the options (viewer → host). There is no other coupling.

### 2.1 What the viewer owns

Everything inside the viewer card, plus all 3Dmol-side state:

| Concern | Component |
|---|---|
| 3-D rendering | 3Dmol GLViewer mounted in `.mol-viewer-canvas` |
| Standard knob bar | `.mol-viewer-knobs` — style / labels / axes / reset / screenshot / background / export |
| Animation frame strip | `.mol-viewer-frame-strip` (auto-shown for trajectory animation) |
| Info line | `.mol-viewer-info-line` — atom count, formula (residue count was promised in an earlier draft but never rendered; if needed, surface it through a card header or via `info` opts on a future iteration) |
| Overlay state | axes, cell wireframe, labels, arrows, atom overlays, pick halos, animation frame, camera |
| 3Dmol object lifecycle | shapes, labels, models — create / update / destroy |
| Render scheduling | when to call `viewer.render()` |
| Export plumbing | save-to-project / download / clipboard / animation encoders |

### 2.2 What the host owns

Everything OUTSIDE the viewer card:

| Concern | Owner |
|---|---|
| Page layout | host tab's CSS grid / flex |
| Tab-specific control cards | file picker, mode list, selection panel, region editor, sidebar |
| Data fetching | sidebar reads, `/api/*` calls, polling |
| Charts | plotly spectrum + energy/force plots |
| Project context | current sidebar dir, file naming |
| Cross-component state | selection store, runtime store, etc. |

The host's adjacent control card may sit anywhere relative to the
viewer card — left, right, above, below, wrapping at narrow
viewports — via standard CSS. The viewer offers NO layout API for
external attachments; this keeps the embed simple and lets each
tab decide its own responsive behavior.

### 2.3 Crossing the boundary

| Direction | Mechanism |
|---|---|
| Host → viewer (mutate) | `handle.setStructure(...)`, `setStyle(...)`, `setAxes(...)`, `setOverlays(...)`, etc. |
| Host → viewer (read) | `handle.getAtomCount()`, `getPickedIndices()`, `getCamera()`, `getAnimationFrame()` |
| Host → viewer (commands) | `handle.playAnimation()`, `refit()`, `screenshot()`, `exportData(...)`, `dispose()` |
| Viewer → host (events) | `opts.onReady(handle)`, `opts.onError(err)`, `opts.pick.onPick(indices)`, `opts.export.onExport(info)`, `opts.animation.onFrame(idx, handle)` *(trajectory only)* |

The host never reads 3Dmol objects directly; the viewer never
reads host DOM outside its own card; neither inspects the other's
CSS or in-page state.

### 2.4 Deprecated escape hatches

| Hatch | Reason it still exists | Removal trigger |
|---|---|---|
| `handle._viewer3dmol()` | `modify/viewer.js` exposes it through a Playwright fixture (`window.__molbuilder_modify_test.getViewer`) so the e2e tests can introspect the live 3Dmol viewer.  Production code paths (`lib/selection/viewer-adapter.js` included) migrated to `setOverlays` + `getCamera` / `setCamera` + native pick on 2026-06-03. | When the test fixtures port to the embed's `handle._test` surface or to a `getViewerSnapshot()`-style read-only accessor.  Production code paths are already off the hatch. |
| `opts.card.bare` | REMOVED 2026-06-03 — all five consumers migrated to the standard chrome (#202–#206). The opt is now ignored; callers still passing `bare: true` get the standard card chrome. The DOM class `.mol-viewer-bare` is no longer emitted and the corresponding CSS rules are gone. | — |

Both are documented but **MUST NOT** be used in new code. Tab
authors that find themselves wanting one should propose a new
declarative API on the handle instead.

### 2.5 Required external modules

The embed is a composer over the rest of the molbuilder runtime
stack. This subsection lists what it expects to find at `embed()`
time, and what happens when a dependency is absent.

#### 2.5.1 Hard dependencies (`embed()` throws on absence)

| Dependency | Provides | Where loaded |
|---|---|---|
| `$3Dmol` (window global) | 3Dmol GLViewer + primitives | `static/vendor/3dmol-min.js` |
| `window.molbuilder.viewer.create` | low-level GLViewer mount helper | `static/lib/mol-viewer.js` |
| `window.molbuilder.fmt` | XYZ/PDB parse + element extraction + formula | `static/lib/mol-format.js` |

Missing any of these is a programming error: the embed throws
synchronously with a `ViewerError` of code `"missing_dependency"`
naming the absent module. Page boot pre-loads all three before any
tab is interactive.

#### 2.5.2 Soft dependencies (feature degrades gracefully)

| Dependency | Disabled feature | Degraded behavior |
|---|---|---|
| `window.molbuilder.axes` (`lib/mol-axes.js`) | `opts.axes`, `setAxes()` | silently no-op; no axes drawn |
| `window.molbuilder.style` (`lib/mol-style.js`) | `opts.style.rep` mapping | falls back to 3Dmol's default stylespec; `colorScheme` ignored |

Soft-dep degradation is silent in production. Tests assert it via
`handle._test.getDependencyStatus()` (§ 9.4).

#### 2.5.3 Integration dependencies (only required by specific features)

| Dependency | Required for | Contract expected |
|---|---|---|
| `window.molbuilder.projects` (`lib/projects-api.js`) | `exportData({target:"project"})`, `exportAnimation({target:"project"})`, `screenshot({target:"project"})` | Must expose `writeFile(path, data, opts?) → Promise<{ok, error, path, ...}>` and `currentDir → string` (synchronous getter). See `docs/protocols/projects-sidebar.md` for the full contract. |
| `navigator.clipboard` | `exportData({target:"clipboard"})` | Must expose `writeText(s) → Promise<void>`. Browser provides this when origin is HTTPS or localhost; `undefined` elsewhere. |
| `window.MediaRecorder` (global) | `exportAnimation({format:"webm"})` | Must support `MediaRecorder.isTypeSupported("video/webm")`. |
| `window.GIF` (lazy-loaded gif.js) | `exportAnimation({format:"gif"})` | Loaded on first use from `static/vendor/gif.min.js`. Lazy-load is shared across embeds; a single failed load disables GIF for all instances. |

If a feature's integration dep is absent at the point of use:

- The corresponding handle method rejects with a `ViewerError` of
  the appropriate code (§ 5 — `"no_project"`, `"no_clipboard"`,
  `"no_media_recorder"`, `"no_gif_encoder"`).
- The export menu **hides** buttons whose dep is unavailable at the
  time of menu open (re-checked on each open, not cached).

#### 2.5.4 Project-API integration detail

Save-to-project targets compute the full path as:

```js
path = projects.currentDir + "/" + (opts.export.defaultName || "structure") + "." + format
```

The embed calls `projects.writeFile(path, data)` and propagates the
returned envelope:

- On `{ok: true, ...}`: fires `opts.export.onExport({...})` and
  resolves the Promise with `{filename, bytes}`.
- On `{ok: false, error}`: rejects the Promise with
  `ViewerError(code: "io_error", message: error, cause: envelope)`.

The embed does NOT depend on `projects.readFile`, `projects.list`,
or any other surface — write-only integration.

#### 2.5.5 Test injection

`opts.testInjection` accepts dependency-injected substitutes for
testing. When supplied, the substitutes replace the global lookup
**only for this embed instance**; production globals are untouched.
See § 9 for the exact interface.

---

## 3. API surface

```ts
window.molbuilder.viewer.embed(host: HTMLElement, opts: ViewerOpts)
    → ViewerHandle
```

### 3.1 `ViewerOpts`

```ts
type ViewerOpts = {
  // ---- Structure data (both optional at mount) ----------------- //
  xyz?: string,                  // XYZ text
  pdb?: string,                  // PDB text
  // If both supplied, pdb wins (richer metadata).  Pass one OR
  // the other in production; both is a programmer convenience for
  // the "load by extension" path.  If NEITHER is supplied, the
  // viewer mounts an empty canvas; populate later via
  // ``handle.setStructure({xyz | pdb})``.

  // ---- Style --------------------------------------------------- //
  style?: StyleOpts,             // see § 3.3

  // ---- Overlays (each opt-in; all default off) ----------------- //
  axes?:   AxesOpts | boolean,   // see § 3.4; true = Cartesian default
  cell?:   CellOpts | boolean,   // see § 3.5; true = use opts.lattice
  labels?: LabelOpts | boolean,  // see § 3.6
  arrows?: ArrowSpec[],          // see § 3.7

  // ---- Per-atom overlays (subset styling, halos, markers) ------ //
  overlays?: OverlaySpec,        // see § 3.12

  // ---- Lattice (drives cell mode for axes + cell wireframe) ---- //
  lattice?: number[3][3],        // a, b, c row vectors in Å.

  // ---- Atom-pick interaction ----------------------------------- //
  pick?: PickOpts,               // see § 3.8

  // ---- Interaction hooks (canonical pointer events) ------------ //
  interaction?: InteractionOpts, // see § 3.15

  // ---- Animation (vibrational mode OR trajectory frames) ------- //
  animation?: AnimationOpts,     // see § 3.9

  // ---- Camera persistence -------------------------------------- //
  preserveCamera?: boolean,
  // If true (default), camera state is preserved across
  // ``setStructure`` calls AFTER the first load.  The first load
  // always calls ``zoomTo()`` to frame the structure; subsequent
  // loads (e.g. /modify ops that return a slightly-edited
  // structure) keep the user's pan/zoom.  Set ``false`` to force
  // ``zoomTo()`` on every ``setStructure``.

  // ---- Standard knob bar (canonical chrome) -------------------- //
  knobs?: KnobBarOpts | boolean, // see § 3.10; true = all defaults

  // ---- Export plumbing ----------------------------------------- //
  export?: ExportOpts | boolean, // see § 3.11; true = all defaults

  // ---- Card chrome (the embeddable panel) ---------------------- //
  card?: {
    title?:        string,        // shown in card header
    showInfoLine?: boolean,       // "N atoms · R residues · CxHyOz"
    height?:       string,        // CSS value; default "clamp(360px, 52vh, 500px)"
    className?:    string,        // extra class for outermost card div
    bare?:         boolean,       // REMOVED 2026-06-03 — ignored;
                                  // standard chrome shows regardless.
  },

  // ---- Test injection (production embeds pass nothing) -------- //
  testInjection?: TestInjection, // see § 9.3

  // ---- Lifecycle / events -------------------------------------- //
  onReady?: (handle: ViewerHandle) => void,
  // Fired once after the first structure is mounted + rendered.

  onError?: (err: ViewerError) => void,
  // Fired for any error the embed cannot return through a Promise:
  //   - synchronous setX method failures (bad input, render error)
  //   - soft-dep load failures discovered late
  //   - internal render exceptions caught by the loop
  // Async methods reject their returned Promise INSTEAD of firing
  // onError; onError is for the sync / fire-and-forget paths only.
}
```

### 3.2 `ViewerHandle`

```ts
type ViewerHandle = {
  // ---- Data setters (re-renders automatically) ----------------- //
  setStructure(opts: { xyz?:             string,
                       pdb?:             string,
                       lattice?:         number[3][3],
                       preserveCamera?:  boolean,
                       preservePick?:    boolean,
                       preserveOverlays?:boolean }): void,
  // See § 4.2 for setStructure × animation interactions and the
  // camera-preservation rule.  ``preserveCamera`` here overrides
  // the embed-level opts.preserveCamera for THIS call only.
  // ``preservePick`` / ``preserveOverlays`` override the default
  // § 4.2.1 element-comparison rule: ``true`` keeps the state
  // unconditionally (host knows the swap is logically same-
  // structure, e.g. an atom-type edit where the index space
  // survives by host bookkeeping); ``false`` clears
  // unconditionally; ``undefined`` (default) follows the
  // element-comparison rule.

  appendFrames(frames: number[][][3]): void,
  // Trajectory mode only.  Extends ``animation.frames`` with the
  // supplied frames.  The current playback frame index is
  // preserved (does NOT auto-jump to the new tail).  Used by the
  // trajectory inspector's live polling path.  No-op for
  // vibration mode or when no animation is set.
  //
  // ``arrowsPerFrame`` (if it was supplied at ``setAnimation``
  // time) is NOT extended; new tail frames render with NO arrows
  // for indices beyond the original ``arrowsPerFrame.length``.
  // To update arrows for new tail frames, either re-call
  // ``setAnimation({...current, frames: ..., arrowsPerFrame: ...})``
  // with the new combined arrays, OR use the ``onFrame`` callback
  // (§ 3.9) to source arrows from a host-side data store per
  // frame.  The live-poll usage pattern (§ 7.4) shows both.

  // ---- Style + overlay setters --------------------------------- //
  setStyle(style: StyleOpts):              void,
  setAxes(axes: AxesOpts | boolean):       void,
  setCell(cell: CellOpts | boolean):       void,
  setLabels(labels: LabelOpts | boolean):  void,
  setArrows(arrows: ArrowSpec[]):          void,
  setPick(pick: PickOpts):                 void,
  setBackground(color: string):            void,
  // ``color`` is a CSS color string (e.g. "#ffffff", "rgb(0,0,0)",
  // "transparent").  Affects the canvas backdrop ONLY; never
  // changes the page theme.  No named-theme shortcuts.

  // ---- Per-atom overlays --------------------------------------- //
  setOverlays(overlays: OverlaySpec): void,
  // Idempotent; replaces the current overlay set entirely.

  setAtomStyle(
    selector: number[] | { elements: string[] } | { residues: number[] },
    style:    AtomOverlaySpec["style"] | null,
  ): void,
  // Sugar for the common "style these atoms differently" call.
  // Internally upserts a single ``overlays.atoms[]`` entry keyed
  // on the selector's normalised form.  Passing ``style: null``
  // removes that entry.  See § 3.12 for layering rules.

  // ---- Camera -------------------------------------------------- //
  getCamera(): CameraState,
  // Captures the current camera (position, look-at, zoom, rotation).

  setCamera(state: CameraState | null): void,
  // Restores camera from a previously-captured state.  No-op if
  // state is null or has a different _viewer / _version than the
  // embed expects.

  // ---- Knob bar ------------------------------------------------ //
  setKnobs(knobs: KnobBarOpts | boolean): void,
  // Reconfigure the visible knobs at runtime (rare; for tabs that
  // change the available controls mid-session).

  // ---- Animation control --------------------------------------- //
  setAnimation(animation: AnimationOpts | Partial<AnimationOpts> | null): void,
  // Full-or-partial update.  Full: pass a complete AnimationOpts
  // with ``kind``.  Partial: pass an object WITHOUT ``kind`` to
  // update individual fields of the active animation (e.g.
  // ``{amplitude: 0.3}`` on a running vibration).  ``null`` clears
  // the animation; see § 4.3 for the stop policy.

  playAnimation():        void,
  pauseAnimation():       void,
  isAnimationPlaying():   boolean,
  setAnimationFrame(idx: number): void,  // trajectory mode only
  getAnimationFrame():    number,

  // ---- Read accessors ------------------------------------------ //
  getAtomCount():     number,
  getElements():      string[],
  getPickedIndices(): number[],
  setPickedIndices(indices: number[] | null): void,
  // Push the pick state from an external source (host atom list,
  // panel, undo).  Re-renders halos + labels according to the
  // active pick.mode / pick.halo / pick.label.  Does NOT fire
  // ``onPick`` — that callback is reserved for click-driven
  // changes so hosts mirroring picks into a store don't see a
  // feedback loop.  Clamps to the mode's max (single: 1, pair: 2,
  // multi: unbounded); pass null or [] to clear.
  // ---- Declarative-state getters (round-trip with setX) ------- //
  // Each returns a defensive deep-clone of the current section so
  // callers can persist or restore via the matching setX without
  // mirroring their own bookkeeping.  Returns null when the
  // section is disabled (matches the setX null-clear shape).
  //   setStyle(getStyle())   is idempotent
  //   setAxes(getAxes())     is idempotent
  //   setLabels(getLabels()) is idempotent
  //   etc.
  // Function fields (PickOpts.onPick, AnimationOpts.onFrame) are
  // preserved as live references — JSON clone would drop them.
  getStyle():     StyleOpts,
  getAxes():      AxesOpts     | null,
  getCell():      CellOpts     | null,
  getLabels():    LabelOpts    | null,
  getOverlays():  OverlaySpec  | null,
  getPick():      PickOpts     | null,
  getKnobs():     KnobBarOpts | null,
  getArrows():    ArrowSpec[],
  getAnimation(): AnimationOpts | null,
  getBackground(): string | null,
  getLattice():   number[3][3] | null,

  // ---- Ordered batch runner --------------------------------- //
  // Apply many sections at once in a canonical order so atom-
  // keyed state (overlays, labels-as-array, picks, animation,
  // camera) lands AFTER setStructure has reloaded the model.
  // Every field is optional; ``applyState({})`` is a no-op.  Use
  // case: host-side persistence round-trip —
  //   const snap = {
  //     structure: { xyz: handle.getStructureText() },
  //     style: handle.getStyle(), axes: handle.getAxes(),
  //     overlays: handle.getOverlays(),
  //     camera: handle.getCamera(),
  //   };
  //   // ... later, after a reload ...
  //   handle.applyState(snap);
  applyState(spec: ApplyStateSpec): void,

  getStructureText(format?: "xyz" | "pdb"): string,
  // Returns the current structure as text in the requested format.
  // If ``format`` is omitted, returns whatever was supplied
  // (pdb if both were available, else xyz).  Used by the export
  // plumbing.

  // ---- Output / export ----------------------------------------- //
  screenshot(opts?: { target?:   "project" | "download",
                      width?:    number,
                      height?:   number,
                      filename?: string,
                      signal?:   AbortSignal }):
      Promise<{ dataUrl: string, blob: Blob,
                filename?: string, bytes?: number }>,
  // Captures a PNG of the current canvas.  Resolves with the data
  // URL + blob.  If ``target: "download"``, also triggers a browser
  // download.  If ``target: "project"``, also writes to the active
  // project's sidebar dir via projects.writeFile.  Omit ``target``
  // for capture-only.
  //
  // When ``width`` or ``height`` exceeds the on-screen canvas
  // size, the embed uses 3Dmol's ``pngURI(width, height)`` for
  // super-resolution capture; aspect ratio is preserved by the
  // missing dimension.

  exportData(opts: { target:    "project" | "download" | "clipboard",
                     format?:   "xyz" | "pdb",
                     filename?: string,
                     signal?:   AbortSignal }):
      Promise<{ filename: string, bytes: number }>,
  // Imperative version of the structure-export menu.  See § 3.11
  // for target semantics.

  // ---- Animation export ---------------------------------------- //
  // Only meaningful when opts.animation is set.  Each method
  // drives the animation forward and captures the canvas per
  // frame; static-structure embeds reject with
  // ViewerError(code: "static_structure").

  captureFrames(opts?: { fps?:        number,
                         duration?:   number,
                         width?:      number,
                         height?:     number,
                         signal?:     AbortSignal,
                         onProgress?: (pct: number, label?: string) => void }):
      Promise<Blob[]>,
  // Returns a list of PNG blobs, one per frame.  The viewer drives
  // the animation forward over ``duration`` seconds at ``fps``,
  // capturing the canvas after each render.  Vibration mode steps
  // the cosine phase; trajectory mode advances frame indices,
  // wrapping if duration exceeds the trajectory length.  Defaults:
  //   fps      = animation.fps (trajectory) or 30 (vibration)
  //   duration = full one-loop (trajectory) or 1/animation.speedHz
  //              seconds (vibration; one cosine cycle)
  //   width    = canvas client width
  //   height   = canvas client height
  // PNG output isolates the encoder choice from the embed.

  exportAnimation(opts: { format:      "webm" | "gif",
                          target:      "project" | "download",
                          fps?:        number,
                          duration?:   number,
                          filename?:   string,
                          signal?:     AbortSignal,
                          onProgress?: (pct: number, label?: string) => void }):
      Promise<{ filename: string, bytes: number }>,
  // High-level animation export, wired to the export menu's
  // "Animation" submenu.  ``format: "webm"`` uses the native
  // MediaRecorder on canvas.captureStream(fps); ``format: "gif"``
  // uses gif.js (lazy-loaded on first call).  No "clipboard"
  // target — binary video isn't pasteable in molbuilder's contexts.

  // ---- Lifecycle ----------------------------------------------- //
  refit(opts?: { indices?: number[], pullback?: number }): void,
  // Re-fit camera to the structure (default) or to a subset of
  // atoms (``opts.indices`` — 0-based).  ``opts.pullback`` is a
  // post-fit zoom multiplier (e.g. 0.55 pulls the camera back so
  // surrounding atoms stay in frame; 2.0 zooms in tighter).
  // ``refit()`` with no opts is the historical behavior:
  // zoom-to-fit on every atom.

  setPivot(opts?: { indices?: number[] }): void,
  // Re-anchor the rotation / zoom-into-cursor pivot on a subset
  // of atoms (or all atoms when ``opts.indices`` is omitted).
  // The camera distance stays exactly where the user left it;
  // only the world origin moves.  Used by tabs like /modify that
  // need to snap the pivot back onto a focal sub-structure
  // (e.g. molecule between two electrode slabs) so rotations
  // orbit the right point.
  render():   void,               // force a render (rarely needed)
  dispose():  void,               // tear down 3Dmol + remove DOM

  // ---- Test affordances --------------------------------------- //
  _test: TestHandle,             // see § 9.2

  // ---- Escape hatch — see § 2.4 -------------------------------- //
  _viewer3dmol(): GLViewer,
}
```

### 3.3 `StyleOpts`

```ts
type StyleOpts = {
  rep?:         "stick" | "ball-and-stick" | "sphere" | "line",
  radiusScale?: number,           // default 1.0
  colorScheme?: "element" | "chain" | "residue" | "spectrum",
  background?:  string,           // CSS color; default "#1d2128"
                                  // (page card colour — matches the
                                  // dark theme).  Pass "#ffffff" for
                                  // a white canvas; "transparent"
                                  // for compositing.  Canvas
                                  // backdrop only; never affects
                                  // the page theme.
}
```

The `showLabels` field from earlier drafts has been **removed** —
labels are controlled exclusively via `opts.labels` / `setLabels()`
(§ 3.6). Two paths to the same state caused precedence ambiguity.

### 3.4 `AxesOpts`

```ts
type AxesOpts = {
  mode?:   "auto" | "cartesian" | "cell",
  length?: number,                // Cartesian fallback length (Å); default 1.5
  origin?: [number, number, number],   // default [0,0,0]
  labels?: [string, string, string],
  colors?: { [label]: string },
  radius?: number,                // arrow shaft radius; default 0.05 Å
}

// ``axes: true`` ≡ ``axes: {}``.  ``axes: false`` (or omitted) hides.
```

Mode selection:
- `mode: "auto"` (default) — cell mode (a/b/c) when a lattice is
  present; Cartesian (x/y/z) when none is. Hosts that don't care
  about the cell distinction should use this.
- `mode: "cartesian"` — force Cartesian even when a lattice is
  present. Always works.
- `mode: "cell"` — force cell-aligned axes. **Requires a lattice
  on the current structure**; calling `setAxes({mode: "cell"})`
  with no lattice dispatches `invalid_input` and halts (so the
  caller isn't silently downgraded to Cartesian). Hosts that
  want graceful fallback should use `mode: "auto"` instead.

Delegates to `window.molbuilder.axes.draw()` from `lib/mol-axes.js`
(soft dependency; see § 2.5.2).

### 3.5 `CellOpts`

```ts
type CellOpts = {
  color?:  string,                // default uses the page theme
  radius?: number,                // default 0.04 Å
}
```

Drawn from `opts.lattice`. No-op when no lattice is supplied.

### 3.6 `LabelOpts`

```ts
type LabelOpts = {
  // ---- Which atoms to label ------------------------------------ //
  atoms?: "all" | number[],
  //   "all"     → every atom (default when labels is enabled)
  //   number[]  → label only these specific atom indices

  // ---- Label text format --------------------------------------- //
  format?: "index" | "name" | "element",
  //   "index"   → "0", "1", "2", ...                  (default)
  //   "name"    → atom_name field (PDB-derived; falls back to
  //               element symbol)
  //   "element" → element symbol only ("C", "H", "O", ...)

  // ---- Cosmetics ----------------------------------------------- //
  fontSize?:   number,            // default 12
  fontColor?:  string,
  background?: string,
}
```

`atoms` controls WHICH atoms are labelled; `format` controls WHAT
each label says. The two are independent — a per-index selection
with element-symbol labels is `{atoms: [1, 5, 9], format: "element"}`.

### 3.7 `ArrowSpec[]`

```ts
type ArrowSpec = {
  start:   [number, number, number],
  end:     [number, number, number],
  color?:  string,                // default "#888"
  label?:  string,                // optional label at the arrow tip
  radius?: number,                // default 0.05 Å
}
```

Force vectors, transition-mode displacements, and arbitrary
annotations all use this one primitive. The viewer redraws when
`setArrows()` is called with a different array; identity comparison
decides "did the array change".

For trajectory animations, see `arrowsPerFrame` (§ 3.9) which
swaps arrows automatically as frames advance.

### 3.8 `PickOpts`

```ts
type PickOpts = {
  mode: "none" | "single" | "pair" | "multi",

  // ---- Visual treatment of selected atoms ----------------------- //
  // Selected atoms get a halo + an index label by DEFAULT.  This
  // matches /modify's existing behaviour (click an atom → it
  // highlights and shows its index) and gives Build / inspectors
  // the same informative pick UX without per-tab CSS work.
  halo?:  { color?:   string,            // default #ffd54a (yellow)
            radius?:  number,            // default 0.6 Å
            opacity?: number             // default 0.5
          } | true | false,
  //   {color, radius, opacity} → halo on with field overrides
  //   true                     → halo on with all defaults
  //                              (alias for ``{}``, matches the
  //                              ``axes: true`` / ``cell: true``
  //                              boolean-shorthand convention)
  //   false                    → no halo
  //   undefined                → halo on with all defaults (same
  //                              as ``true``)
  style?: { color?:       string,        // tint applied to picked atoms
            opacity?:     number,        // default 1
            radiusScale?: number },      // optional; no default override

  label?: false | "index" | "name" | "element",
  //   false      → no auto labels on picked atoms
  //   "index"    → "0", "1", ...                       (DEFAULT)
  //   "name"     → atom_name field (PDB-derived)
  //   "element"  → element symbol

  // ---- Callback ------------------------------------------------ //
  onPick?: (indices: number[]) => void,

  // ---- Backwards-compat shims (deprecated) --------------------- //
  haloColor?:  string,           // use halo.color instead
  haloRadius?: number,           // use halo.radius instead
}
```

**Deprecated-field precedence.** If the new `halo` field is
supplied as any value OTHER than `undefined` — including `{}`,
`false`, or `true` — the deprecated `haloColor` / `haloRadius`
are **ignored entirely**, no field-level merge. The deprecated
path applies ONLY when `halo` itself is omitted (`undefined`)
and `{haloColor, haloRadius}` is supplied; in that case the
embed synthesises `halo: {color: haloColor, radius: haloRadius,
opacity: <default>}` and proceeds as if the new shape was used.
This keeps the merge rule trivial to reason about (you're either
in the legacy lane or the modern lane, never both). Hosts on the
legacy lane that want to keep their colour override should NOT
also pass `halo: true`; pass the legacy fields alone, OR migrate
to `halo: {color: "...", radius: ...}` directly.

**Defaults.** `halo: { color: "#ffd54a", radius: 0.6,
opacity: 0.5 }` plus `label: "index"`. Hosts that want minimal
selection rendering pass `halo: false, label: false`; hosts that
want richer rendering compose `halo + style + label` per
preference.

**Layering.** Selection halos draw above OverlaySpec halos
(§ 3.12). Selection style overrides apply ABOVE base style and
ABOVE OverlaySpec style overrides — picked atoms always "win" the
style fight so the user can SEE what's selected. Selection labels
draw above other labels.

**Persistence.** Picked indices survive `setStructure({xyz: ...})`
IFF the atom count and element ordering match; otherwise the pick
state is cleared. Picked indices are NOT affected by
`setOverlays()`, `setStyle()`, or `setAnimation()`.

**External push (`setPickedIndices`)**. Hosts with their own
selection UI (atom-list rows, panels, undo) push the pick state
into the embed via `handle.setPickedIndices(indices)` (see § 3.2).
The embed clamps to the mode's max (single: 1, pair: 2; multi:
unbounded) and re-renders halos + labels per the active config.
External pushes DO NOT fire `onPick` — that callback is reserved
for click-driven changes so hosts that mirror picks into a store
don't see a feedback loop. Calling `setPickedIndices` when no
pick mode is configured (`pick: undefined` or `pick.mode: "none"`)
is a silent no-op AFTER the type check — the argument shape is
still validated and bad input still fires `invalid_input`.

Halo geometry is internal (see `_redrawPickHalos`); the embedded
viewer owns both rendering and click-handler registration.

### 3.9 `AnimationOpts`

```ts
type AnimationOpts =
  | VibrationAnimation
  | TrajectoryAnimation;

type VibrationAnimation = {
  kind:          "vibration",
  // Per-atom displacement direction (Å).  Length must match the
  // structure's atom count.  Position at phase φ:
  //     pos_i(φ) = baseline_i + amplitude · cos(φ) · displacement_i
  displacements: number[][][3],
  amplitude?:    number,         // peak Cartesian amplitude (Å); default 0.15
  speedHz?:      number,         // cycle-rate multiplier; default 1.0
                                 //   MUST be > 0; values ≤ 0 are
                                 //   clamped to 1e-3 by the embed.
  paused?:       boolean,        // start in paused state; default false
};

type TrajectoryAnimation = {
  kind:           "trajectory",
  // Each frame is a full [n_atoms][3] coordinate set.  Element
  // ordering must match the baseline structure (changing topology
  // mid-trajectory is not supported).
  frames:         number[][][3],
  // Optional parallel arrays — index ``i`` applies during frame ``i``:
  arrowsPerFrame?: ArrowSpec[][],
  // Length MUST equal frames.length when present.  An empty
  // arrowsPerFrame[i] means "no arrows during frame i".  Drives
  // the trajectory inspector's per-frame force-vector display
  // without the host having to call setArrows() on every frame.

  startFrame?:    number,         // default 0
  fps?:           number,         // playback rate; default 10
  paused?:        boolean,        // start in paused state; default true
  loop?:          boolean,        // wrap at end; default true

  onFrame?:       (idx: number, handle: ViewerHandle) => void,
  // Fired BEFORE each frame renders.  Hosts that need custom
  // per-frame logic (conditional arrows, overlay updates) wire
  // here.  Calling handle.setX methods from onFrame is supported
  // but adds render cost; prefer arrowsPerFrame for the common
  // case.
};
```

**Vibration mode** drives spectra's per-mode visualisation.
**Trajectory mode** drives the trajectory inspector's frame
playback. Both preserve the user's camera, pick state, axes, cell,
labels, and arrows across frames — the viewer updates ONLY atom
positions; every other overlay is position-aware and recomputes
per-frame automatically.

When `animation.kind === "trajectory"`, the **frame strip** auto-
renders above the canvas: prev / play-pause / next / counter /
slider, wired directly to `playAnimation` / `pauseAnimation` /
`setAnimationFrame`. Vibration mode does NOT show a frame strip
(it has no discrete frames); the standard knob bar's play/pause
handles vibration.

### 3.10 `KnobBarOpts`

Phase 6 redesign — the knob bar collapses to **two top-level
menus** (View + Export) instead of seven flat knobs.  The menus
expand to structured submenus with labelled sections.

```ts
type KnobBarOpts = {
  // Top-level menus:
  view?:   boolean,    // default true — show the View menu
  export?: boolean,    // default true — show the Export menu

  // View → submenu sections (all default true):
  style?:      boolean,    // 4-button rep picker
  labels?:     boolean,    // On/Off toggle
  background?: boolean,    // preset swatches + custom-colour chip
  axes?:       boolean,    // On/Off toggle
  reset?:      boolean,    // plain action button

  // Background section configuration:
  backgroundPresets?:     string[],    // default ["#1d2128",
                                       //          "#ffffff",
                                       //          "transparent"]
  backgroundAllowCustom?: boolean,     // default true
};

// ``knobs: true`` (or omitted) shows both menus with all sections.
// ``knobs: false`` hides the bar entirely.
```

**Menu semantics**:

| Menu | Submenu | Maps to | UI |
|---|---|---|---|
| View | Style | `setStyle({rep})` | 4-button row: Stick / Ball & stick / Sphere / Line |
| View | Labels | `setLabels(true \| false)` | toggle (uses the host's mount-time `LabelOpts.format`; default `"index"`) |
| View | Background | `setBackground(color)` | preset swatches + styled custom-colour chip wrapping `<input type="color">` |
| View | Axes | `setAxes(true \| false)` | toggle |
| View | Reset | `refit()` | plain button |
| Export | Save to project / Download | `exportData` / `screenshot` / `exportAnimation` | 5 format buttons per target: `.xyz` `.pdb` `.png` `.gif` `.webm` (gif/webm hidden when no animation mounted) |

**Style picker** offers exactly the four representations
`lib/mol-style.js` implements.  The picker passes the spelled-out
form (`"ball-and-stick"`) on the wire; the embed translates to
mol-style's historical identifier (`"ballstick"`) at the
`_applyStyle` boundary.

**Labels** is a single on/off toggle.  Format choice (index / name
/ element) is not user-pickable through the UI — it's a mount-time
config (`LabelOpts.format`).  Toggling Off then On restores the
last LabelOpts the embed saw, falling back to the documented
`{atoms:"all", format:"index"}` default.

**Background** defaults to `#1d2128` (the page card colour) so
the viewer reads as part of the dark theme.  White is one of the
default presets; pass `style.background: "#ffffff"` at mount or
call `setBackground("#ffffff")` for publication figures.  The
styled custom-colour chip wraps a native `<input type="color">`
so the OS picker still opens on click; the chip presents the
chosen colour as a small swatch matching the preset row.

**Keyboard shortcuts**:

| Key | Action |
|---|---|
| `V` | Toggle the View menu open/close |
| `X` | Toggle the Export menu open/close |
| `R` | Reset view (`refit()`) |
| `Space` | Play/pause (animation only, canvas/frame-strip focus) |
| `←` / `→` | prev / next frame (trajectory only) |
| `Home` / `End` | first / last frame (trajectory only) |
| `Esc` | Close any open menu |

Per-knob shortcuts (L / A / B / E) from the previous flat layout
are gone — the 2-menu structure is deep enough that hover-to-open
is sufficient discoverability and dedicated keys for sub-items
would crowd the keymap.

Single-letter keys do NOT fire while a `<input>`, `<textarea>`,
or `[contenteditable]` element inside the card is focused.

**Mutual exclusion** — opening View closes Export and vice versa
(click rule).

### 3.11 `ExportOpts`

The export menu unifies three content categories — **structure**
(XYZ/PDB text), **image** (PNG of the canvas), and **animation**
(WebM/GIF, when an animation is active) — under a single set of
target affordances: project / download / clipboard. Each category
declares which targets make sense for it.

```ts
type ExportOpts = {
  // Filename hint (no extension).  Defaults to the structure's
  // ``system_label`` from a PDB HEADER record, or "structure".
  defaultName?:   string,

  // ---- Structure (xyz/pdb text) ----------------------------- //
  structure?: {
    saveToProject?: boolean,   // default true
    download?:      boolean,   // default true
    clipboard?:     boolean,   // default true
  } | boolean,                 // true ≡ all defaults; false hides

  // ---- Image (PNG screenshot) ------------------------------- //
  image?: {
    saveToProject?: boolean,   // default true
    download?:      boolean,   // default true
    // No clipboard for image — molbuilder pastes nowhere useful.
  } | boolean,

  // ---- Animation (WebM/GIF) — only meaningful with animation - //
  animation?: {
    webm?:          boolean,   // default true (native MediaRecorder)
    gif?:           boolean,   // default true (gif.js, lazy-loaded)
    fps?:           number,    // capture rate; see captureFrames()
    duration?:      number,    // seconds; see captureFrames()
    saveToProject?: boolean,   // default true
    download?:      boolean,   // default true
    // No clipboard for video binary.
  } | boolean,

  // Callback after a successful export from ANY category/target:
  onExport?: (info: {
      kind:     "structure" | "image" | "animation",
      target:   "project" | "download" | "clipboard",
      format:   "xyz" | "pdb" | "png" | "webm" | "gif",
      filename: string,
      bytes:    number,
  }) => void,
};
```

Format availability is automatic:
- Structure: XYZ shows if the embed has XYZ text; PDB shows if it
  has PDB text. No inter-format conversion.
- Image: PNG is always available (canvas → toBlob).
- Animation: visible only when `opts.animation` is set; WebM
  requires `MediaRecorder` (§ 2.5.3); GIF triggers a lazy load of
  gif.js on first use.

**Save-to-project requires project context** (§ 2.5.4). If no
project is active, the corresponding handle Promise rejects with
`ViewerError(code: "no_project")` and the export button surfaces
that error via `opts.onError` (no toast UI is owned by the embed
— hosts wire toasts as they see fit).

### 3.12 `OverlaySpec`

Per-atom styling on top of the base style. Used by /modify for
frozen-atom and region highlights, by /spectra to grey out
spectator atoms, and by the selection-store viewer-adapter for
selection halos.

```ts
type OverlaySpec = {
  atoms?: AtomOverlaySpec[],
  // More overlay kinds (bonds, planes, isosurfaces) may be added
  // in future without breaking changes — atoms[] is the current
  // sole sub-shape.
};

type AtomOverlaySpec = {
  // Which atoms (exactly ONE of these must be supplied):
  indices?:   number[],          // 0-based atom indices
  elements?:  string[],          // by element symbol (e.g. ["C"])
  residues?:  number[],          // by residue index (PDB only)

  // What to apply (any combination):
  style?: {
    rep?:         "stick" | "sphere" | "line" | "cross" | "hidden",
    radiusScale?: number,
    color?:       string,
    opacity?:     number,        // 0..1; default 1
  },

  // Halo overlay (drawn on top of the per-atom style):
  halo?: {
    color?:   string,            // CSS color; default #6ba6ff (blue —
                                 // distinct from PickOpts halo's
                                 // yellow so host overlays don't
                                 // collide with the pick affordance)
    radius?:  number,            // Å; default 0.6
    opacity?: number,            // 0..1; default 0.5
  },

  // Optional cosmetic marker:
  marker?: {
    kind:    "lock" | "star" | "dot",
    color?:  string,
  },
};
```

**Layering rules** (applied bottom-up):

1. Base style (`opts.style`) applied to ALL atoms.
2. Per-atom overlays (`opts.overlays.atoms`) override base style
   for the atoms they match. When two entries overlap, the LATER
   entry in the array wins.
3. Halos drawn on top of all styles (additive overlay).
4. Markers drawn above halos.
5. Pick halos (from `opts.pick`) draw above all overlay halos.

**Imperative API:**

```ts
handle.setOverlays(overlays);
// Idempotent; replaces the current overlay set.

handle.setAtomStyle(indices, style);
// Sugar: upserts a single overlays.atoms[] entry keyed on the
// (normalised) selector.  Passing style: null removes that entry.
```

### 3.13 `CameraState`

```ts
type CameraState = {
  // Opaque blob; treat as a token to round-trip through
  // setCamera().  Internal shape is 3Dmol's getView() return,
  // but consumers MUST NOT depend on the layout.
  _viewer: "3dmol",
  _version: 1,
  data: unknown,
};
```

```ts
handle.getCamera(): CameraState
// Captures position, look-at, zoom, rotation, slab.

handle.setCamera(state: CameraState | null): void
// Restores from a previously-captured state.  Mismatched
// _viewer/_version is a no-op (forward-compat for future
// renderers).
```

Use cases:
- /modify saves camera at session unload, restores at load.
- Tests verify reset-view actually moves the camera.
- A future "share a view" link would round-trip the blob.

**Version-bump policy.** `_version` is reserved for breaking
changes to the underlying `data` layout. Consumers that persist
`CameraState` across page reloads (e.g. into `sessionStorage`)
MUST accept that a future `_version` bump will silently reset
the saved camera — `setCamera()` no-ops on mismatch rather than
attempting to migrate the blob. This is the forward-compat policy
for the opaque blob; it keeps the embed free to switch renderers
without coordinating a migration with every persisted session.

### 3.14 `ViewerError`

See § 5 for the complete error model. Every async handle method
rejects with this shape; every `onError` callback receives it.

```ts
type ViewerError = {
  code:
    | "missing_dependency"
    | "no_project" | "no_clipboard"
    | "no_media_recorder" | "no_gif_encoder"
    | "no_structure" | "static_structure"
    | "io_error" | "aborted" | "disposed"
    | "invalid_input" | "unknown",
  message: string,
  cause?:  unknown,
};
```

### 3.15 `InteractionOpts`

```ts
type InteractionOpts = {
  // Distance in CSS pixels the pointer must move from the press
  // point before a press-then-move gesture commits to "drag".
  // Default 4.  Smaller values fire onDragStart on accidental
  // wiggle; larger values delay the fire past the visible motion.
  dragThresholdPx?: number,

  // Fires exactly ONCE per gesture, on the first mousemove past
  // ``dragThresholdPx`` from the press point.  Receives the press-
  // point coords + the modifier state captured at mousedown so the
  // host can branch on plain-drag vs ctrl-drag vs shift-drag etc.
  // Hosts use this to implement custom interaction policies
  // (e.g. /modify snaps the camera pivot onto the molecule on the
  // first plain-drag mousemove so the rotation is centred).
  onDragStart?: (event: {
    x: number, y: number,
    modifiers: { ctrl: boolean, shift: boolean,
                 alt: boolean,  meta: boolean },
  }) => void,

  // Fires on mouseup OR mouseleave when a gesture that fired
  // onDragStart ends.  Receives the end-point coords + the
  // modifier state from mousedown (the same payload onDragStart
  // saw, so hosts can pair-match).
  onDragEnd?: (event: {
    x: number, y: number,
    modifiers: { ctrl: boolean, shift: boolean,
                 alt: boolean,  meta: boolean },
  }) => void,
};
```

**Why this exists.** Consumer tabs sometimes need to react to
pointer gestures on the viewer canvas — e.g. /modify snaps the
camera pivot onto the molecule on the first drag mousemove so
rotation is centred. Before this opt, every such consumer had
to hand-roll mousedown / mousemove / mouseup / mouseleave
listeners, replicate the same modifier-key + drag-threshold
plumbing, and remember to remove the listeners on teardown.
`InteractionOpts` centralises the boilerplate so consumers express
the *policy* (the if-modifier branch + the action) without owning
the *mechanics*.

**Threshold rationale.** Snapping on raw mousedown would jump the
camera every time the user clicks an atom to select it (mousedown
without drag). Waiting for the gesture to commit to "drag" past
`dragThresholdPx` makes the snap invisible — the camera was
already moving at that frame.

**Mount-only, no setInteraction.** The policy is registered at
mount and torn down on dispose; runtime swaps aren't supported.
This matches how `onError` / `onReady` are wired.

**Failure isolation.** If `onDragStart` or `onDragEnd` throws, the
embed catches the exception, logs it to `console.error`, and
continues — a buggy host callback never breaks pointer handling.

---

## 4. Lifecycle and state transitions

### 4.1 Mount → render → dispose

```mermaid
sequenceDiagram
    participant Tab as Tab code
    participant Embed as MolViewer.embed
    participant TDmol as 3Dmol
    participant DOM

    Tab->>Embed: embed(host, opts)
    Embed->>DOM: insert card scaffold (header + knobs + canvas)
    Embed->>TDmol: create GLViewer
    Embed->>TDmol: load opts.xyz / opts.pdb (if any)
    Embed->>TDmol: apply style + overlays
    Embed->>TDmol: render
    Embed-->>Tab: handle
    Embed->>Tab: onReady(handle)

    Note over Tab,Embed: ...later — host or knob bar...

    Tab->>Embed: handle.setStructure({xyz})
    Embed->>TDmol: remove old models, load new
    Embed->>TDmol: re-apply style + overlays
    Embed->>TDmol: render

    Note over Tab,Embed: ...on tab tear-down...

    Tab->>Embed: handle.dispose()
    Embed->>TDmol: clear shapes / labels / models
    Embed->>DOM: remove card
```

**General invariants:**

- Every `setX` method is **idempotent** — passing the same opts
  twice is a no-op (no re-render churn).
- Every `setX` method **maintains all other state** — calling
  `setAxes(true)` doesn't clear labels or arrows.
- Every `setX` method that fails its input validation:
  - Logs to console (one line, no stack)
  - Fires `opts.onError(ViewerError{code:"invalid_input",...})`
  - Returns without mutating viewer state
- `dispose()` is **idempotent**. Second call is a no-op.
- After `dispose()`, every other sync handle method becomes a
  no-op rather than throwing; every async handle method's Promise
  rejects with `ViewerError(code: "disposed")`.
- The knob bar reflects current state.  Phase 6 affordance map:
  `aria-pressed` on the Labels and Axes toggles (View menu);
  `is-active` on the matching rep button (View → Style);
  `is-active` on the matching Background preset swatch (View →
  Background).  Programmatic calls flow the same way:
  `setStyle()` / `setLabels()` / `setAxes()` / `setBackground()`
  from the handle re-sync the matching affordance.  Custom
  background colours that don't match any preset leave every
  swatch unmarked (the custom-colour chip carries the value).
  Setters that have no knob-bar representation (`setOverlays`,
  `setPick`, `setArrows`, `setCell`, `setAnimation`,
  `setStructure`, `setCamera`, `setKnobs`) do not drive any
  chrome — by design, since the corresponding state isn't exposed
  in the View / Export menus.  `setAnimation` does toggle the
  visibility of the gif / webm Export buttons (hidden when no
  animation is mounted).

### 4.2 `setStructure` × camera

| Call | Camera behavior |
|---|---|
| First `embed()` with xyz/pdb | `zoomTo()` (frame the structure) |
| First `setStructure({xyz/pdb})` after empty mount | `zoomTo()` (first sight of structure) |
| Subsequent `setStructure` with `preserveCamera: true` (default) | Camera preserved |
| Subsequent `setStructure` with `preserveCamera: false` | `zoomTo()` |
| `refit()` | `zoomTo()` regardless of preserveCamera |
| Reset knob | calls `refit()` |

The opt-level `preserveCamera` is the default; the per-call value
on `setStructure({preserveCamera: ...})` overrides for that call.

### 4.2.1 `setStructure` × declarative atom-indexed state

Pick state (§ 3.8) and OverlaySpec entries (§ 3.12) both name
atoms by index, so both follow one rule across `setStructure`:

| Atom count + element ordering vs new structure | Pick state | OverlaySpec |
|---|---|---|
| Match exactly (same N atoms, same element at each index) | picked indices preserved; halo + label re-render against new coordinates | preserved; halos / styles / markers re-render against new coordinates |
| Mismatch (different N, or any element changes) | picked indices cleared; `onPick([])` fires | cleared (`state.current.overlays` set to `null`); host re-applies via `setOverlays(...)` if highlights are still wanted |

The "atom-index space changed" case (different N or different
elements) makes index-keyed state stale by definition — an
overlay that meant "highlight the carbon at position 5" no longer
makes sense once that slot holds a different element. The embed
clears it rather than re-render against the wrong atoms.

Cross-cutting lifecycle invariants that ALSO carry across
`setStructure`:
- **Camera** — preserved iff `preserveCamera: true` (§ 4.2).
- **Animation** — cleared (the baseline atom set changed); host
  must call `setAnimation(...)` to re-arm. See § 4.3.
- **Knob bar / chrome** — unaffected (DOM independent).

The atom-edit ops in `/modify` that preserve atom count and order
(e.g. moving a single atom's position) keep selection AND
overlays visible mid-edit; a real file swap via the Build file
picker drops both. Type-swap edits (e.g. C → N) change element
ordering and therefore clear both — the highlight on "atom 5 was
C" is no longer meaningful.

**`preservePick` / `preserveOverlays` escape hatch.** Hosts that
track their own index mapping (a /modify atom-type edit that
keeps the index space stable by host bookkeeping; a programmatic
batch that swaps an atom and re-applies the same selection
afterward) can override the element comparison via
`setStructure({xyz, preservePick: true, preserveOverlays: true})`.
`true` keeps the state unconditionally; `false` clears
unconditionally; `undefined` follows the default rule above.
Equally useful for hosts that want to clear even when elements
match (e.g. "user pressed Reset → drop selection regardless").

### 4.2.2 Persistence round-trip via `applyState`

Hosts that persist + restore the embed's state should use the
nine declarative-state getters (§ 3.2) at save time and
`applyState(spec)` at restore time. The batch runner enforces
the canonical order so atom-keyed sections (overlays, labels-as-
array, picks, animation, camera) land AFTER `setStructure` has
reloaded the model — manual call sequences are easy to get
subtly wrong here.

```js
// Save: snapshot every section the host wants to round-trip.
const snap = {
  structure:  {
    xyz:     handle.getStructureText(),
    lattice: handle.getLattice(),    // cell-bearing structures
  },
  style:      handle.getStyle(),
  background: handle.getBackground(),
  axes:       handle.getAxes(),
  cell:       handle.getCell(),
  labels:     handle.getLabels(),
  overlays:   handle.getOverlays(),
  pick:       handle.getPick(),
  pickedIndices: handle.getPickedIndices(),
  animation:  handle.getAnimation(),
  camera:     handle.getCamera(),
};
// ... store snap somewhere, e.g. sessionStorage ...

// Restore: one call, canonical order.
handle.applyState(snap);
```

Round-trip invariant: every documented getter is paired with a
setter (direct or via `applyState`). Skipping `getLattice()` in
the save snapshot silently loses the cell, which makes
`setAxes({mode: "cell"})` unreachable post-restore — see § 3.4.

Order (each field is optional):

1. `structure` (atom space may change; everything below depends
   on the new model existing)
2. `style` → `background`
3. `axes`, `cell`, `labels`, `arrows`
4. `overlays` (atom-keyed; layered above base style per § 3.12)
5. `pick`, `pickedIndices`
6. `animation`
7. `camera` (frames the new structure)
8. `knobs` (DOM chrome; independent)

`interaction`, `export`, `card`, `onError`, `onReady`,
`preserveCamera`, and `testInjection` are **mount-only** opts —
they're NOT batchable via `applyState` because the embed's
lifecycle owns them.

`applyState` is NOT a "single render frame" guarantee — 3Dmol's
render is cheap and the intermediate renders are fine.

### 4.3 `setStructure` × animation

| Call | Animation behavior |
|---|---|
| `setStructure` while no animation | new structure mounts as baseline |
| `setStructure` while vibration active | animation paused + cleared; new structure is new baseline; host must call `setAnimation(...)` to re-arm |
| `setStructure` while trajectory active | animation paused + cleared; new structure is new baseline; the trajectory inspector uses `appendFrames` for the LIVE-POLL case (no setStructure) |
| `appendFrames(frames)` (trajectory) | frames appended; current frame index preserved; playback continues if it was running |

The live-poll path is intentionally `appendFrames`, not
`setStructure`. Calling `setStructure` mid-trajectory is for the
"loaded a new file" case, not the "got more frames" case.

### 4.4 `setAnimation(null)`

| Previous state | Resulting state |
|---|---|
| vibration playing | loop stops; atoms snap back to baseline |
| vibration paused (any phase, including initial `paused: true`) | atoms snap back to baseline; vibration cleared |
| trajectory playing | loop stops; atoms stay on the current frame |
| trajectory paused (default `paused: true`) | atoms stay on the current frame; trajectory cleared |
| animation set but no structure mounted yet | animation state cleared; no positional change (no atoms to move) |
| no animation | no-op |

The frame strip is removed in all cases (when it was present).

### 4.5 `dispose()` during in-flight async export

When `dispose()` runs:

- Any in-flight `screenshot()`, `exportData()`, `captureFrames()`,
  or `exportAnimation()` Promise rejects with
  `ViewerError(code: "disposed")`.
- The encoder workers (gif.js) are terminated.
- The MediaRecorder is stopped without finalising the WebM.
- `dispose()` does NOT wait for any of these to complete; it
  returns synchronously.

### 4.6 AbortSignal handling

All async handle methods accept `signal?: AbortSignal`. When the
signal aborts:

- The Promise rejects with `ViewerError(code: "aborted")`.
- The viewer's animation loop continues (host may want to keep
  playing back); only the export job is cancelled.
- Already-emitted progress callbacks are not "rolled back" — the
  caller may have seen pct=42 even though the export aborts.

If the signal is already aborted at call time, the method rejects
synchronously (within a microtask) without doing any work.

---

## 5. Error model

### 5.1 Error shape

```ts
type ViewerError = {
  code:    ViewerErrorCode,
  message: string,           // human-readable
  cause?:  unknown,          // underlying error if any
};
```

The `code` field is the stable test-and-branch identifier; the
`message` is for logs and toasts. Callers MUST switch on `code`,
not on `message`.

### 5.2 Error codes

| Code | Surface | Meaning |
|---|---|---|
| `missing_dependency` | `embed()` throws | A hard dep (§ 2.5.1) was absent at mount |
| `no_structure` | async-only | An export was requested before any structure loaded |
| `static_structure` | async-only | Animation export requested when `opts.animation` is null |
| `no_project` | async-only | `target: "project"` requested but no active project context |
| `no_clipboard` | async-only | `target: "clipboard"` requested but `navigator.clipboard` unavailable |
| `no_media_recorder` | async-only | `format: "webm"` requested but `MediaRecorder` unavailable |
| `no_gif_encoder` | async-only | `format: "gif"` requested but gif.js failed to load |
| `io_error` | async-only | `projects.writeFile` returned `{ok: false}`; or download anchor click failed |
| `aborted` | async-only | `signal` was aborted before completion |
| `disposed` | async-only | `dispose()` ran while operation was in flight |
| `invalid_input` | `onError` | A sync setX got malformed opts (e.g. bad pdb text) |
| `unknown` | both | Any uncaught exception we couldn't classify |

### 5.3 Method → possible-error matrix

This table is **the** reference for code-vs-doc review. Sync setters
dispatch `invalid_input` on input that fails type / shape / enum /
range validation against the documented contract; they then proceed
with the documented default and continue rendering. They do NOT
throw and they do NOT skip the call (except where noted as "halt"
below — used when continuing would corrupt state, e.g. non-string
`xyz`). Async methods reject the Promise rather than firing
`onError`.

| Method | Sync throw | Promise reject codes | onError codes |
|---|---|---|---|
| `embed()` | `missing_dependency` | — | — |
| `setStructure` | — | — | `invalid_input` (`xyz` / `pdb` not a string → halt) |
| `appendFrames` | — | — | `invalid_input` (atom-count mismatch → halt). No-animation and wrong-kind calls are silent no-ops per § 3.2. |
| `setStyle` | — | — | `invalid_input` (`rep` outside `{stick, ball-and-stick, sphere, line}` — non-halt: `rep` clamps to `"stick"` default; non-finite `radiusScale` — non-halt: clamps to `1.0`) |
| `setAxes` | — | — | `invalid_input` (`mode` outside `{auto, cartesian, cell}`; `mode: "cell"` without a lattice on the current structure → halt with hint to use `mode: "auto"` for graceful fallback) |
| `setCell` | — | — | — (`color`/`radius` coerced to defaults) |
| `setLabels` | — | — | `invalid_input` (`atoms` not `"all"`/`number[]`; non-int / negative entries in `atoms` array; `format` outside `{index, name, element}`) |
| `setArrows` | — | — | `invalid_input` (argument not an array → halt; per-entry missing `start`/`end`) |
| `setPick` | — | — | `invalid_input` (`mode` outside `{none, single, pair, multi}`; `label` neither `false` nor one of `{index, name, element}`) |
| `setBackground` | — | — | `invalid_input` (`color` not a non-empty string → halt) |
| `setOverlays` | — | — | `invalid_input` (entries dropped: bad/missing/multiple selectors, or no style/halo/marker) |
| `setAtomStyle` | — | — | `invalid_input` (bad selector → halt; style with no `{rep, radiusScale, color, opacity}` → halt) |
| `setAnimation` | — | — | `invalid_input` (`kind` outside `{vibration, trajectory}` → halt; vibration without `displacements` array → halt; vibration `displacements.length ≠ atom_count` → halt; trajectory without `frames` array → halt; trajectory `frames[0].length ≠ atom_count` → halt). Partial updates (no `kind`) merge silently. |
| `setKnobs` | — | — | `invalid_input` (`backgroundPresets` not an array) |
| `setPickedIndices` | — | — | `invalid_input` (argument not `number[] \| null` → halt) |
| `applyState` | — | — | `invalid_input` (argument not an object → halt). Otherwise transitive: each subsection's errors fire per its own row above. |
| `setCamera` | — | — | `invalid_input` (argument not an object → halt). Version mismatch is silent (forward-compat) per § 3.13. |
| `screenshot` | — | `no_structure`, `no_project`, `io_error`, `aborted`, `disposed`, `unknown` | — |
| `exportData` | — | `invalid_input`, `no_structure`, `no_project`, `no_clipboard`, `io_error`, `aborted`, `disposed`, `unknown` | — |
| `captureFrames` | — | `no_structure`, `static_structure`, `aborted`, `disposed`, `unknown` | — |
| `exportAnimation` | — | `invalid_input`, `no_structure`, `static_structure`, `no_project`, `no_media_recorder`, `no_gif_encoder`, `io_error`, `aborted`, `disposed`, `unknown` | — |
| `getCamera` / `getAtomCount` / `getElements` / `getPickedIndices` / `getStructureText` / `getAnimationFrame` / `isAnimationPlaying` | — | — | — |
| `playAnimation` / `pauseAnimation` / `setAnimationFrame` | — | — | — (no-op when no animation; frame index clamped) |
| `refit` / `setPivot` / `render` / `dispose` / `_viewer3dmol` | — | — | — |

**"halt" semantics.** When a row says "→ halt", the method returns
early without mutating state — there is no half-applied side effect.
This applies to every `invalid_input` that would otherwise corrupt
the structure / animation / camera baseline.

### 5.4 `opts.onError` semantics

- Fires for sync paths (setX validation failures) and internal
  render-loop catches.
- NEVER fires for async-method errors — those reject the Promise
  instead.
- Idempotent against duplicate errors (rate-limited to one fire
  per error code per 500 ms by the embed).
- If `opts.onError` throws, the embed catches and logs to console;
  the original error path continues uninterrupted.

---

## 6. Card structure

The viewer mounts as a `<section class="card mol-viewer-card">` so
it composes cleanly with the rest of the molbuilder UI's card
layout.

```html
<section class="card mol-viewer-card" data-mol-viewer="1">
  <header class="mol-viewer-card-header">
    <h2 class="mol-viewer-card-title">{opts.card.title}</h2>
    <span class="mol-viewer-info-line">3 atoms · 1 residue · H₂O</span>
  </header>

  <!-- §6.2 Standard knob bar — 2 menus (View + Export).
       knobs: false hides this bar entirely. -->
  <div class="mol-viewer-knobs" role="toolbar"
       aria-label="Viewer controls">

    <details class="mol-viewer-knob mol-viewer-menu mol-viewer-menu-view">
      <summary>View</summary>
      <div class="mol-viewer-menu-body">
        <section class="mol-viewer-menu-section" data-section="style">
          <h4 class="mol-viewer-menu-heading">Style</h4>
          <div class="mol-viewer-rep-row">
            <button class="mol-viewer-rep-btn is-active" data-rep="stick">Stick</button>
            <button class="mol-viewer-rep-btn" data-rep="ball-and-stick">Ball &amp; stick</button>
            <button class="mol-viewer-rep-btn" data-rep="sphere">Sphere</button>
            <button class="mol-viewer-rep-btn" data-rep="line">Line</button>
          </div>
        </section>
        <section class="mol-viewer-menu-section" data-section="labels">
          <h4 class="mol-viewer-menu-heading">Labels</h4>
          <button class="mol-viewer-toggle" data-action="labels"
                  aria-pressed="false">Show labels</button>
        </section>
        <section class="mol-viewer-menu-section" data-section="background">
          <h4 class="mol-viewer-menu-heading">Background</h4>
          <div class="mol-viewer-bg-row">
            <button class="mol-viewer-bg-swatch is-active"
                    data-color="#1d2128" style="background:#1d2128"></button>
            <button class="mol-viewer-bg-swatch"
                    data-color="#ffffff" style="background:#ffffff"></button>
            <button class="mol-viewer-bg-swatch is-transparent"
                    data-color="transparent">·</button>
            <label class="mol-viewer-bg-custom">
              <input type="color" data-knob="background-custom">
            </label>
          </div>
        </section>
        <section class="mol-viewer-menu-section" data-section="axes">
          <h4 class="mol-viewer-menu-heading">Axes</h4>
          <button class="mol-viewer-toggle" data-action="axes"
                  aria-pressed="false">Show axes</button>
        </section>
        <section class="mol-viewer-menu-section" data-section="reset">
          <button class="mol-viewer-action" data-action="reset">Reset view</button>
        </section>
      </div>
    </details>

    <details class="mol-viewer-knob mol-viewer-menu mol-viewer-menu-export">
      <summary>Export</summary>
      <div class="mol-viewer-menu-body">
        <section class="mol-viewer-menu-section" data-section="target-project">
          <h4 class="mol-viewer-menu-heading">Save to project</h4>
          <div class="mol-viewer-export-row">
            <button class="mol-viewer-export-btn" data-target="project" data-kind="structure" data-format="xyz">.xyz</button>
            <button class="mol-viewer-export-btn" data-target="project" data-kind="structure" data-format="pdb">.pdb</button>
            <button class="mol-viewer-export-btn" data-target="project" data-kind="image"     data-format="png">.png</button>
            <button class="mol-viewer-export-btn" data-target="project" data-kind="animation" data-format="gif"  hidden>.gif</button>
            <button class="mol-viewer-export-btn" data-target="project" data-kind="animation" data-format="webm" hidden>.webm</button>
          </div>
        </section>
        <section class="mol-viewer-menu-section" data-section="target-download">
          <h4 class="mol-viewer-menu-heading">Download</h4>
          <div class="mol-viewer-export-row">
            <button class="mol-viewer-export-btn" data-target="download" data-kind="structure" data-format="xyz">.xyz</button>
            <button class="mol-viewer-export-btn" data-target="download" data-kind="structure" data-format="pdb">.pdb</button>
            <button class="mol-viewer-export-btn" data-target="download" data-kind="image"     data-format="png">.png</button>
            <button class="mol-viewer-export-btn" data-target="download" data-kind="animation" data-format="gif"  hidden>.gif</button>
            <button class="mol-viewer-export-btn" data-target="download" data-kind="animation" data-format="webm" hidden>.webm</button>
          </div>
        </section>
      </div>
    </details>

  </div>

  <!-- §6.3 Frame strip — only when animation.kind === "trajectory" -->
  <div class="mol-viewer-frame-strip">…</div>

  <!-- §6.4 Canvas — 3Dmol mounts inside -->
  <div class="mol-viewer-canvas"
       style="height: {opts.card.height}"
       aria-label="3-D molecular viewer; {N} atoms"></div>
</section>
```

### 6.1 Anatomy

| Region | When shown | Role |
|---|---|---|
| Header | always (unless `card.title` empty AND `showInfoLine: false`) | title + info-line |
| Knob bar | always (unless `knobs: false`) | standard control knobs |
| Frame strip | trajectory animation only | prev/play/next + slider |
| Canvas | always | 3-D WebGL surface |

### 6.2 Knob bar

Phase 6 redesign: two top-level menus (View + Export) replace the
flat 7-knob row.  Layout is now compact and readable on narrow
viewports without the previous wrap-and-collapse machinery.

- Lives between header and frame strip (or canvas if no frame
  strip).
- The bar itself is a `<div class="mol-viewer-knobs" role="toolbar">`
  containing exactly two top-level `<details>` elements:
  - `<details class="mol-viewer-menu mol-viewer-menu-view">`
  - `<details class="mol-viewer-menu mol-viewer-menu-export">`
- Each `<summary>` is the menu trigger; the body
  (`.mol-viewer-menu-body`) is a popover containing labelled
  `<section class="mol-viewer-menu-section">` rows.  Each section
  has a small uppercase `<h4>` heading + the section's controls
  (rep button row, toggle, swatch row, action button, etc.).
- Themed via `tokens.css`: `--bg-card`, `--bg-input`,
  `--bg-input-focus`, `--accent`, `--text-primary`,
  `--text-muted`, `--border-soft`, `--border-strong`.
- Toggle buttons (Labels, Axes) use `aria-pressed` and a
  ●/○ marker glyph that flips based on pressed state.
- Style rep buttons get an `is-active` class on the current
  representation; Background swatches get `is-active` on the
  matching preset (case-insensitive); custom colours that don't
  match a preset leave every swatch unmarked (the chip carries
  the value).
- User interaction → handle: every submenu control routes through
  the matching public setter (`setStyle` / `setLabels` /
  `setBackground` / `setAxes` / `refit` / `exportData` /
  `screenshot` / `exportAnimation`) so a host's `onError` /
  `onExport` callbacks see UI-driven actions identically to
  programmatic ones.
- Handle → UI: every setX above re-syncs the matching submenu
  affordance.  See § 4.1.
- Mutual exclusion: opening View closes Export and vice versa.
- `Esc` closes any open menu.
- Knob suppression: top-level `view: false` / `export: false`
  hide the whole menu; per-section flags hide rows inside View;
  `knobs: false` hides the bar entirely.

### 6.3 Frame strip

- Lives between knob bar and canvas; absent unless
  `animation.kind === "trajectory"`.
- Contains: prev / play-pause / next / `frame N / total` counter
  (sourced from `getAnimationFrame()` + `animation.frames.length`)
  / range slider.
- Wires directly to `playAnimation` / `pauseAnimation` /
  `setAnimationFrame`.
- Slider has `aria-label="Trajectory frame"` and the value range
  `0..frames.length-1`.
- Vibration mode reuses the knob bar's play/pause and does NOT
  show this strip.

### 6.4 Canvas

- Inline `height` style from `opts.card.height` (default
  `clamp(360px, 52vh, 500px)`).
- Width is always 100% of the card.
- `aria-label` describes "3-D molecular viewer; N atoms" so
  screen readers announce the canvas meaningfully.
- 3Dmol mounts its `<canvas>` element inside this div.

`data-mol-viewer="1"` is the disposer's hook: `dispose()` removes
every element with this attribute that the handle owns. The
attribute is NOT a unique identifier — multiple embeds on one page
each get their own `data-mol-viewer="1"` section; they're
distinguished by DOM identity, not attribute values.

### 6.5 Multi-embed behavior

- Each embed instance owns its own state, knob bar, canvas, animation
  loop, picked indices, and overlay set. No shared state across
  instances.
- Lazy-loaded resources (gif.js) ARE shared: the first embed to
  trigger a GIF export starts the load; subsequent embeds reuse
  the cached encoder. A failed load disables GIF for all
  instances until the page reloads.
- `projects.writeFile` is shared (single-tab project context).
  Two embeds saving the same filename concurrently is the host's
  concern; the embed performs no de-duplication.

---

## 7. Usage patterns

Five canonical embed calls — one per consumer site. Each shares
the same card structure; only the host's adjacent control card
differs per tab.

### 7.1 Build (/) tab

```js
const handle = embed(document.getElementById("viewer"), {
  card:    { title: "Structure" },
  style:   { rep: "ball-and-stick" },
  pick:    { mode: "single" },
  axes:    true,
  cell:    true,
  export:  { defaultName: "build" },
  onReady(h) { /* wire to file picker on the host */ },
  onError(err) { showToast(err.message, { variant: "error" }); },
});
// Host's file-picker card calls:
handle.setStructure({ xyz: textFromSidebar });
```

### 7.2 Modify (/modify) tab — uses overlays for frozen + region highlights

```js
const handle = embed(document.getElementById("viewer"), {
  card:    { title: "Structure" },
  style:   { rep: "ball-and-stick" },
  pick:    { mode: "multi",
             onPick: idx => selectionStore.setIndices(idx) },
  //         Default rendering: picked atoms get an accent halo
  //         + an index label automatically.  Matches /modify's
  //         pre-embed behaviour.  Override via pick.halo /
  //         pick.style / pick.label if needed.
  axes:    true,
  cell:    true,
  onError(err) { showToast(err.message, { variant: "error" }); },
});

// Frozen-atom + region highlight via overlays:
selectionStore.subscribe(state => {
  handle.setOverlays({
    atoms: [
      { indices: state.frozen,
        style: { color: "#888", opacity: 0.55 },
        marker: { kind: "lock", color: "#888" } },
      { indices: state.regionA,
        halo: { color: "#5fb6ff", radius: 0.6 } },
      { indices: state.selection,
        halo: { color: "var(--accent)", radius: 0.7 } },
    ],
  });
});
```

This replaced the selection-store viewer-adapter's direct 3Dmol
calls.  The adapter is now off the escape hatch (migrated to
`setOverlays` + `getCamera` / `setCamera` + native pick on
2026-06-03); see § 2.4 for the remaining hatch usage.

### 7.3 Results > structure inspector

```js
const handle = embed(slotEl, {
  card:    { title: "Structure" },
  style:   { rep: "ball-and-stick" },
  pdb:     r.text,
  axes:    true,
  onError(err) { showInspectorError(err.message); },
});
```

### 7.4 Results > trajectory inspector — uses appendFrames for live polling

```js
const handle = embed(slotEl, {
  card:      { title: "Geometry steps" },
  style:     { rep: "ball-and-stick" },
  pdb:       firstFrameText,
  animation: {
    kind:           "trajectory",
    frames:         initialFrames,
    arrowsPerFrame: forcesPerFrame,   // optional per-frame arrows
    fps:            10,
  },
  axes:      true,
  cell:      true,
  preserveCamera: true,
});

// Plotly chart click → seek to frame:
chart.on("plotly_click", evt => handle.setAnimationFrame(evt.points[0].pointNumber));

// Live polling — append new frames without dropping animation:
watchPoller.on("data", ({ newFrames, newForces }) => {
  handle.appendFrames(newFrames);
  // arrowsPerFrame is a snapshot at setAnimation time; for live
  // updates, use the onFrame callback to source arrows from the
  // poller's latest data, OR re-call setAnimation with the new
  // full arrowsPerFrame array.
});
```

### 7.5 Results > spectra inspector

```js
const handle = embed(slotEl, {
  card:      { title: "Vibrational modes" },
  style:     { rep: "ball-and-stick" },
  pdb:       equilibriumText,
  animation: { kind:          "vibration",
               displacements: modes[0].eigenvector,
               amplitude:     0.18,
               speedHz:       1.5 },
  axes:      true,
  // Grey out frozen/spectator atoms so the eye focuses on the mode:
  overlays:  { atoms: [{ indices: cfg.frozen_indices,
                         style: { color: "#888", opacity: 0.5 } }] },
});

// Mode-list card (host-owned) drives the viewer:
modeList.onChange(mode => handle.setAnimation({
  kind:          "vibration",
  displacements: mode.eigenvector,
}));

// Amplitude slider in host card → partial update:
amplitudeSlider.oninput = e =>
    handle.setAnimation({ amplitude: parseFloat(e.target.value) });
```

In every example the host's adjacent card supplies tab-specific
controls (file picker, selection panel, plotly chart, mode list);
the viewer card supplies the canvas, the standard knobs and any
animation strip.

---

## 8. Consumer migration map

When this contract lands, every viewer site migrates to the
standard chrome. None retains its own style / labels / axes / play
controls.

| Site | Current viewer | Target |
|---|---|---|
| `/` (Build) | `static/viewer.js` | Embed; standard knob bar replaces bespoke buttons; file picker stays adjacent. |
| `/modify` | `static/modify/viewer.js` + `lib/selection/viewer-adapter.js` | Embed; selection adapter migrates from `_viewer3dmol` to `setOverlays` + `getCamera`/`setCamera`; selection panel stays adjacent. |
| `/results` structure inspector | `lib/inspectors/structure.js` | Embed; drop `card.bare`; standard knobs only. |
| `/results` trajectory inspector | `lib/trajectory/core.js` | Embed with `animation: {kind: "trajectory", frames, arrowsPerFrame}`; live polling via `appendFrames`; viewer owns frame strip; inspector keeps plotly + polling. |
| `/results` spectra inspector | `lib/spectra/core.js` | Embed with `animation: {kind: "vibration", displacements}`; frozen-atom highlight via `overlays`; mode-list stays adjacent. |

Migration order respected feature dependencies (all DONE as of
2026-06-03):

1. Update doc (this commit + this revision).
2. Implement § 3 additions (overlays, camera, animation extensions,
   error model, async signal/progress) in `mol-viewer-embed.js`.
3. Implement standard knob bar + export plumbing.
4. Migrate sites one at a time (Build → Modify → structure →
   trajectory → spectra), browser-verifying each.
5. Migrate selection-store viewer-adapter to declarative API.
   `_viewer3dmol()` removed from production code paths;
   surviving call site is the `/modify` Playwright fixture
   (see § 2.4).
6. Add cross-site chrome-consistency tests.  Currently pins
   Build + Modify; the three /results inspectors are not yet
   covered by an automated chrome-signature assertion.
7. `card.bare` code path removed; the opt is silently ignored
   if any legacy caller still passes it.

Live-polling refinement: trajectory's poll path currently issues a
full `setAnimation({kind:"trajectory", frames})` on every refresh
instead of using the contract's `appendFrames(newFrames)`
short-circuit.  Switching the strict-superset case to
`appendFrames` keeps the playback loop running across polls
(today it restarts paused) and avoids re-baking the embed's
coord baseline.

---

## 9. Testing affordances

### 9.1 Public test surface on `window.molbuilder.viewer`

The following are documented test hooks. They are stable within
a major revision (refactors update both the surface AND this
section in the same commit).

```ts
window.molbuilder.viewer.embed              // production entry
window.molbuilder.viewer._normaliseOpts(opts)
window.molbuilder.viewer._normaliseStyle(style)
window.molbuilder.viewer._normaliseAxes(axes)
window.molbuilder.viewer._normaliseCell(cell)
window.molbuilder.viewer._normaliseLabels(labels)
window.molbuilder.viewer._normalisePick(pick)
window.molbuilder.viewer._normaliseAnimation(animation)
window.molbuilder.viewer._normaliseOverlays(overlays)
window.molbuilder.viewer._normaliseLattice(lattice)
window.molbuilder.viewer._normaliseKnobs(knobs)
window.molbuilder.viewer._equalNormalised(a, b)
```

These are pure functions; unit tests in
`tests/test_mol_viewer_embed_js.py` exercise them via Node.

### 9.2 `handle._test` — per-instance test surface

```ts
type TestHandle = {
  // Visual / DOM:
  getCanvasElement(): HTMLCanvasElement | null,
  getOverlayShapeCount(): number,        // 3Dmol shape count
  getOverlayLabelCount(): number,        // 3Dmol label count
  getKnobBarElement(): HTMLElement | null,
  getFrameStripElement(): HTMLElement | null,

  // State inspection:
  hasAnimationLoop(): boolean,
  getCurrentBackground(): string,
  getCurrent(): NormalisedState,    // live snapshot; READ ONLY
  getDependencyStatus(): {
    axes: boolean, style: boolean, format: boolean,
    projects: boolean, clipboard: boolean,
    mediaRecorder: boolean, gif: "loaded" | "loading" | "absent",
  },

  // Force operations for tests:
  triggerKnob(
    name: "labels" | "axes" | "reset" | "screenshot"
        | "background" | "export" | "style",
    arg?: {
      // For popover knobs (labels / background / export), arg
      // selects a specific sub-action; without it, triggerKnob
      // just toggles the popover open / closed.
      format?: "index" | "name" | "element" | "off",  // labels
      color?:  string,                                  // background preset
      kind?:   "structure" | "image" | "animation",    // export
      target?: "project" | "download" | "clipboard",   // export
      formatExport?: "xyz" | "pdb" | "webm" | "gif",   // export
      // For style: select a representation:
      rep?:    "stick" | "ball-and-stick" | "sphere" | "line"
             | "cartoon" | "cross",
    },
  ): void,
};

// triggerKnob semantics:
//   "axes" / "reset" / "screenshot"  → click the button
//   "labels" / "background" / "export"
//                                    → without arg: toggle popover
//                                    → with arg:    fire the specific
//                                                   sub-action AND close
//                                                   the popover (matching
//                                                   the real-user click flow)
//   "style"                          → arg.rep required; sets the <select> value
```

Tests reach for these instead of `_viewer3dmol()`; they cover
visual-invariant assertions without touching 3Dmol directly.

### 9.3 `opts.testInjection` — dependency injection for tests

```ts
type TestInjection = {
  projectsApi?: {
    writeFile:  (path: string, data: string | Blob, opts?: any)
                  => Promise<{ ok: boolean, error?: string, path?: string }>,
    currentDir: () => string,
  },
  clipboardApi?: {
    writeText: (s: string) => Promise<void>,
  },
  mediaRecorder?: typeof MediaRecorder | null,
  // If supplied as a constructor → used instead of window.MediaRecorder.
  // If supplied as ``null`` (explicit) → forces the embed to treat
  // MediaRecorder as absent, so exportAnimation({format: "webm"})
  // rejects with no_media_recorder.  Used by tests on browsers
  // that ship MediaRecorder natively.

  gifEncoder?: GifEncoderCtor | null,
  // If supplied → used instead of the lazy-loaded window.GIF (skips
  // the /static/vendor/gif.min.js fetch entirely).  If ``null``
  // (explicit) → forces the embed to treat the encoder as absent,
  // so exportAnimation({format: "gif"}) rejects with
  // no_gif_encoder.  Used by tests when gif.js IS shipped at
  // /static/vendor/ (the lazy-load would otherwise succeed and
  // produce a real GIF, making the rejection-path test
  // unreachable).
};
```

When supplied, the substitutes REPLACE the global lookup for
THIS embed instance. Production embeds pass nothing and fall
back to the globals (§ 2.5.3). Mocks for `projectsApi` etc. are
provided by `tests/conftest.py` for Playwright fixtures.

**Soft-dep degradation (§ 2.5.2) is NOT injectable.** Tests that
need to assert "axes were skipped because mol-axes.js is absent"
delete the relevant `window.molbuilder.*` global before calling
`embed()`, then read `handle._test.getDependencyStatus()`. The
embed performs a one-time lookup at mount; injecting a soft-dep
mock per-instance would let one embed see mol-axes while a
neighbour doesn't, which is a misleading test setup. Hard deps
(§ 2.5.1) similarly use the real globals — there is no test
substitute for `$3Dmol`.

### 9.4 Visual / DOM observability

The card uses stable class names so Playwright selectors don't
churn:

| Selector | Element |
|---|---|
| `.mol-viewer-card` | outermost section |
| `.mol-viewer-card-header` | header strip |
| `.mol-viewer-info-line` | atom-count / formula text |
| `.mol-viewer-knobs` | knob bar container |
| `.mol-viewer-menu-view > summary` | View menu trigger |
| `.mol-viewer-menu-export > summary` | Export menu trigger |
| `.mol-viewer-rep-btn[data-rep="stick"]` | Style rep button (replace rep name) |
| `.mol-viewer-toggle[data-action="labels"]` | Labels on/off toggle |
| `.mol-viewer-toggle[data-action="axes"]` | Axes on/off toggle |
| `.mol-viewer-action[data-action="reset"]` | Reset view button |
| `.mol-viewer-bg-swatch[data-color="#ffffff"]` | Background preset swatch (replace colour) |
| `.mol-viewer-bg-custom > input[type="color"]` | Background custom-colour native picker |
| `.mol-viewer-export-btn[data-target="download"][data-kind="structure"][data-format="xyz"]` | individual export action (vary target / kind / format) |
| `.mol-viewer-frame-strip` | frame strip container |
| `.mol-viewer-frame-strip [data-action="play"]` | play button |
| `.mol-viewer-canvas` | canvas host div |
| `.mol-viewer-canvas > canvas` | 3Dmol-owned canvas |

These selectors form the **stable testing contract**; renaming
any of them is a doc revision.

---

## 10. Test coverage

**Pure-logic unit tests** (Node) cover what the embed computes
without 3Dmol:

- Option normalisation (`axes: true` → `{mode: "auto"}`,
  `knobs: true` → full default set, overlays with each selector
  shape, etc.).
- Style → 3Dmol-stylespec mapping (`mol-style.js` contract).
- Lattice → axis-mode selection (cell vs Cartesian).
- Idempotence of setX methods (diff computation).
- Export filename derivation (PDB HEADER → `system_label` →
  fallback "structure").
- AnimationOpts merge (partial update merges with active state).
- ViewerError shape (every async method has a documented
  rejection code matrix; tests assert that matrix).
- OverlaySpec layering (later entries win for overlapping atom
  sets; pick halos draw above overlay halos).

**Live-mount tests** (Playwright) verify VISUAL invariants — not
just program state:

- Canvas DOM dimensions are non-zero on every consumer site.
- 3Dmol's `<canvas>` element exists inside `.mol-viewer-canvas`
  with non-zero size.
- Modify's host aspect-ratio is respected by the embed.
- Standard knob bar exists on every consumer site with the same
  button DOM structure (chrome-consistency test).
- Toggling the Labels knob actually shows/hides labels.
- Toggling the Axes knob actually shows/hides the triad.
- Reset knob: capture camera before + after → confirm a state
  diff via `handle.getCamera()`.
- Screenshot knob produces a non-zero-byte PNG.
- Background popover changes canvas background to the picked
  color; `getCurrentBackground()` reflects it.
- Export menu → Structure → Clipboard writes correct text via
  the injected `clipboardApi` mock.
- Export menu → Structure → Download triggers a Blob download
  (asserted via a stub that captures the anchor `click()`).
- Export menu → Image → both targets produce a valid PNG via
  the injected `projectsApi`.
- Export menu → Animation → WebM target produces a non-zero
  MediaRecorder blob (when animation is active).
- Export menu → Animation → GIF target lazy-loads gif.js +
  produces a valid GIF.
- `captureFrames()` returns the expected number of frames at
  the requested fps × duration.
- `appendFrames()` extends trajectory without resetting frame
  index or stopping playback.
- `setOverlays()` with frozen atoms changes the visible style of
  exactly those atoms; pick halos stay above overlay halos.
- Camera round-trip: `setCamera(getCamera())` is a no-op.
- AbortSignal: aborting in-flight export rejects with `aborted`.
- `dispose()` empties the host + idempotent re-call is safe.
- `dispose()` during in-flight export rejects that export.

The visual-dimensions tests are the regression guard for the
2026-06-02 blank-viewer bug; they MUST run on every PR that
touches the embed module, the bare wrapper, or any host's viewer
CSS.

---

## 11. Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2026-06-02 | Single declarative `embed(host, opts)` entry point — no per-feature constructor. | User's spec: "the viewer takes an input of xyz or pdb data, and a few options". Multiple constructors would invite tab-specific drift; one entry point with optional features keeps the API surface small and composable. |
| 2026-06-02 | Data-in via opts (xyz/pdb text); the viewer never fetches. | Keeps the viewer testable + reusable. Fetching is the tab's job. |
| 2026-06-02 | Card chrome owned by the viewer. | Per user spec: "embedded as a card/panel where needed". |
| 2026-06-02 | Unified `animation` opt covers both vibrational-mode visualisation AND trajectory replay. | Both are "atoms moving over time"; the shape discriminator (`kind`) keeps data shapes honest without splintering the API. |
| 2026-06-02 | Trajectory inspector's plotly chart + polling stay outside the viewer. | The viewer is a renderer + state manager; charts and `/api/watch/data` polling are tab-level concerns. |
| 2026-06-02 | `_viewer3dmol()` escape hatch documented but discouraged. | Selection-store viewer-adapter still needs it; named accessor + docstring makes the boundary explicit. |
| 2026-06-03 | The viewer card includes a standard knob bar (style/labels/axes/reset/screenshot/background/export) as canonical chrome. | Per user spec: "the view is not just the white 3D display but a module with a consistent UI as a whole". A consistent knob bar on every tab is the visual unification that hosts cannot drift from. |
| 2026-06-03 | Export plumbing offers three targets — save-to-project / download / clipboard. | User asked for all three; each is a one-click action in the export menu. No inter-format conversion (xyz↔pdb): the embed exports what it was given. |
| 2026-06-03 | Host owns external layout; embed offers NO attach-slot API. | User preference: "Host owns layout; embed offers no layout API". Keeps the embed simple. |
| 2026-06-03 | `opts.card.bare` deprecated; removed after the five-site migration. | Bare-mode was a first-pass migration shim. With the standard knob bar, the chrome IS the contract; bare-mode actively breaks visual unification. |
| 2026-06-03 | Animation export built in: WebM via native `MediaRecorder`, GIF via lazy-loaded gif.js, plus `captureFrames(...)` primitive. | User asked for both formats; WebM is canonical (zero-dep, real video), GIF is universal-compat. `captureFrames` keeps the embed extensible — server-side MP4 / advanced encoding can be built on top without forking the embed. |
| 2026-06-03 | Image and animation exports share structure's target set (project / download). | User: "the image/animation export can be saved under project sidebar too, as well as direct download". One mental model across all exportable artifacts. Clipboard is excluded for binary blobs (no paste-image context). |
| 2026-06-03 | `ExportOpts` is hierarchical: `{structure, image, animation}`. | Reflects different valid target sets per category; per-knob suppression stays granular without flattening into a long flat list. |
| 2026-06-03 | Add `OverlaySpec` with per-atom style, halo, and marker overlays. | Independent gap review G1.1: /modify, /spectra, and the selection-store adapter all need per-atom styling. Without this, three migrations stall and `_viewer3dmol()` cannot be removed. Layering rules pinned so implementer can't drift. |
| 2026-06-03 | Add `appendFrames(frames[])` for trajectory live-polling. | Independent gap review G2.1: the trajectory inspector polls `/api/watch/data` and pushes new frames continuously; `setStructure` mid-trajectory would wipe animation state. `appendFrames` is the explicit "more frames arrived" path. |
| 2026-06-03 | Add `getCamera`/`setCamera` + `preserveCamera` opt (default true after first load). | Independent gap review G1.3: every `setStructure` calling `zoomTo()` would jump the camera during /modify edits. Hosts that want a fresh frame call `refit()` or pass `preserveCamera: false`. |
| 2026-06-03 | Drop "Dark" theme toggle; replace with Background color picker. | Independent gap review G7.1: molbuilder is dark-only (`tokens.css`). The original "Dark" toggle promised a site-wide theme system that does not exist. The Background knob now affects the canvas backdrop ONLY via a preset+picker popover; no named themes. |
| 2026-06-03 | Add `signal: AbortSignal` + `onProgress` to all async exports. | Independent gap review G4.1: GIF encode + WebM capture take seconds; users need a cancel. Build's existing pipeline uses AbortControllers; the viewer would regress without parity. |
| 2026-06-03 | Define `ViewerError` union with stable `code` field. | Independent gap review G3.1 + G3.2: Promise rejections were untyped; sync setX errors were silent. Now: every async method has a documented rejection code matrix (§ 5.3); every sync error fires `onError`. Callers switch on `code`, not on `message`. |
| 2026-06-03 | Add per-frame `arrowsPerFrame` + `onFrame` to TrajectoryAnimation. | Independent gap review G1.2: per-frame force vectors were "host calls setArrows 30 times/sec", which is both ugly and racy. `arrowsPerFrame` is the declarative parallel array; `onFrame` covers the imperative case. |
| 2026-06-03 | Spec § 4.3 `setStructure × animation` and § 4.4 `setAnimation(null)` transitions. | Independent gap review G2.1/G2.2: undefined transitions invited inconsistent implementations. Now: vibration → restore baseline; trajectory → freeze on current frame. |
| 2026-06-03 | Spec § 4.5 `dispose()` during in-flight export → reject with `disposed`. | Independent gap review G2.3: undefined behavior risked promise leaks. |
| 2026-06-03 | Add § 2.5 Required external modules. | Independent gap review G6.1: dependency list was informal. Now: hard / soft / integration deps explicitly classified, with degradation rules. |
| 2026-06-03 | Add § 9 Testing affordances + `handle._test` + `opts.testInjection`. | Independent gap review G10: tests reached into `_viewer3dmol()` and ad-hoc globals. The new test surface (normalise functions + `handle._test`) is the stable contract; injection lets tests mock projects/clipboard/MediaRecorder/gif.js cleanly. |
| 2026-06-03 | Document keyboard shortcuts (`R`, `L`, `A`, `Space`, `←`/`→`) for the knob bar + frame strip. | Independent gap review G9.1: a11y was implicit. Spec the minimum keyboard surface; hosts can suppress via input focus. |
| 2026-06-03 | Remove `style.showLabels` (was an alias for `labels.atoms`). | Two paths to the same state caused precedence ambiguity (G1.4). `setLabels`/`opts.labels` is the sole path. |
| 2026-06-03 | `LabelOpts.atoms` split into `atoms` (which) + `format` (what). | User asked for richer atom-label UX (index vs name vs element selectable from the knob bar). Old shape conflated "which atoms to label" with "what to label them with"; new shape lets the two vary independently. `atoms: number[]` previously had no documented format — now uses `format: "index"` by default. |
| 2026-06-03 | Labels knob becomes a 4-option popover (Index / Name / Element / Off); `labelsFormats` opt narrows the choices. | User picked "Labels-format popover" over plain on/off and over per-atom custom text. The popover matches the Background and Export pattern; `labelsFormats: ["index"]` reduces it to a plain toggle for tabs that don't need format choice. |
| 2026-06-03 | `PickOpts` extended with `halo` (object), `style`, `label` fields. Default selection rendering is halo + index label. | User confirmed this matches /modify's existing behaviour (click an atom → halo + index visible). Building it into PickOpts means every consumer site renders selections the same way without each tab wiring `onPick → setOverlays`. Hosts opt out via `halo: false, label: false`; richer custom rendering via the `style` override. |
| 2026-06-03 | Selection state survives `setStructure` IFF atom count + element ordering match; cleared otherwise. | Explicit so implementer doesn't invent semantics. Atom-edit ops in /modify that preserve count keep the selection visible mid-edit; an actual file swap (Build's file picker) drops it. |
| 2026-06-03 | Selection halos / style / labels layer above OverlaySpec equivalents. | Pick state is the most "user-driven" of all overlays — the user must be able to see what they just clicked, even when a region tint or frozen-atom overlay would otherwise compete. Layering pinned: base style → OverlaySpec style → pick style; OverlaySpec halo → pick halo; labels follow the same ordering. |
| 2026-06-04 | Unified `state.current.animation.paused` with `state._anim.playing` (single source of truth). | Phase 5g B-1 root-cause fix: `getAnimation().paused` was returning the mount-time value forever because play/pause flipped only the runtime flag.  Round-trip `applyState({animation: getAnimation()})` during playback silently stopped the loop.  Both stores now mutate together at three flip sites (`_stopAnimationLoop`, `_startVibrationLoop`, `_startTrajectoryLoop`); the Phase 5f write-side override became redundant and was removed. |
| 2026-06-04 | `_normaliseAnimation` honors caller-supplied `currentFrame`; `_setAnimationImpl` lands on `next.currentFrame` not `next.startFrame`. | Phase 5h I-1 + Phase 5i: a host snapshotting + re-applying animation state mid-trajectory now keeps the playhead.  Without these two changes the round-trip drifted to frame 0 on every applyState. |
| 2026-06-04 | Retired `lib/mol-pick.js` (53 LOC, zero callers).  Halo geometry is internal to the embed. | Phase 5g B-2: the standalone helper drew orange `#fb923c` halos; the embed's `_redrawPickHalos` draws yellow `#ffd54a` and is the only path any consumer reached.  `getDependencyStatus().pick` field also dropped. |
| 2026-06-04 | Background knob routes through `setBackground()` (not synthetic `setStyle`). | Phase 5j R3: the popover's swatch + custom-color handlers were reconstructing `setStyle({rep, radiusScale, background})` which silently dropped any active `colorScheme` (and `tube` / `radius`).  Routing through the documented setter preserves the full style spec. |
| 2026-06-04 | `getKnobs()` documented as nullable.  Programmatic→UI sync was partial (Axes only) when this row first landed; superseded same day by the 5k entry below — now fully implemented. | Phase 5j D7/D8: code returned `null` when `knobs: false`; doc lied about non-nullability.  Doc also overstated the knob-bar's reactivity to programmatic setX calls. |
| 2026-06-04 | Programmatic→UI sync extended to every documented setX. | Phase 5k closed R1/R2 + Background-swatch sync: ``setStyle`` re-syncs the Style picker, ``setLabels`` marks the matching popover button ``is-active``, ``setBackground`` (which routes through ``setStyle``) outlines the matching swatch.  Custom colours that don't match a preset leave every swatch unmarked.  Mount-time + every ``setKnobs`` rebuild also seed the affordances from current state.  New test ``test_programmatic_setX_syncs_knob_bar_ui`` pins the invariant. |
| 2026-06-04 | Trajectory partial-update splits into in-place (live-readable fields) vs full restart. | Phase 5l B-1: ``setAnimation({arrowsPerFrame, onFrame, loop})`` now mutates ``state.current.animation`` in place (matches the vibration partial-update pattern for amplitude/speedHz).  ``fps`` / ``frames`` / ``currentFrame`` / ``startFrame`` still take the full `_setAnimationImpl` path because they need an interval re-arm or seek.  Eliminates timing jitter on every live-poll tick. |
| 2026-06-04 | Trajectory live-poll uses `appendFrames(newCoords)` for strict-tail-append. | Phase 5k I1 + 5l B-2: when the polled trajectory grows monotonically with unchanged topology/lattice, the inspector pushes only the new frames instead of re-mounting the full animation.  Playback that was running stays running (captured via ``isAnimationPlaying`` before the seek, restored after if the tail-snap stopped the loop).  Non-tail-append cases (structure swap, frames shrank) fall back to a full rebuild. |
| 2026-06-04 | Knob bar collapses to 2 menus (View + Export); 7-knob flat row retired. | Phase 6: View → Style / Labels / Background / Axes / Reset; Export → Save to project / Download × {.xyz, .pdb, .png, .gif, .webm} with animation formats hidden when no animation is mounted.  Screenshot button absorbed into Export → Download → .png.  Style picker trimmed to the 4 reps mol-style.js actually implements (stick / ball-and-stick / sphere / line); the previous picker exposed "cross" and "cartoon" which fell through to stick, and the embed→mol-style boundary translates "ball-and-stick" → "ballstick" so "Ball & stick" renders correctly.  Labels collapses to On/Off (format is mount-time config, default "index"); the 4-button Index / Name / Element / Off popover and the `labelsFormats` opt are gone.  Background picker gets bigger preset swatches + a styled custom-colour chip that wraps the native `<input type="color">`. |
| 2026-06-04 | DEFAULT_BACKGROUND switched from `#ffffff` to `#1d2128` (page card colour). | Phase 6: the white canvas was a bright cut-out against the dark theme.  Dark default matches the surrounding card; white stays a preset for publication figures.  Spectra used to override to `#1d2128` explicitly — now picks up the default implicitly and drops its bespoke `style.background` + `knobs.backgroundPresets` configuration. |
| 2026-06-04 | Build's `#dl-xyz` / `#dl-pdb` buttons + `style.css .resize-hint` retired. | Phase 6: structure download now lives in the Export menu (Save to project / Download → .xyz | .pdb), so the bespoke Build buttons below the viewer were duplicate plumbing through a separate `downloadAs` helper.  The `<div class="resize-hint">drag corner to resize</div>` was dead text never wired to any resize handler. |
| 2026-06-04 | KnobBarOpts hard-break: `position`, `compact`, `screenshot`, `labelsFormats` dropped. | Phase 6 redesign has no horizontal-placement variant, no compact mode (the menus are always menu-sized), no separate screenshot button (folded into Export), and no labels-format popover (Labels is on/off only).  Five in-tree consumers updated in the same commit; no out-of-tree consumers to migrate. |
