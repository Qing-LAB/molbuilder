# VibrationView — animating a vibrational mode

**Role:** contract
**Domain:** web
**Companions:** [`molview.md`](?doc=web/molview.md) — its **sibling** viewer (MolView
never animates; VibrationView never selects/edits); [`spectra.md`](?doc=web/spectra.md)
— the one consumer that mounts it; [`overview.md`](?doc=web/overview.md) — the
module registry; [`roadmap.md`](?doc=roadmap.md) — the pending full-separation work.

VibrationView (`lib/vibrationview/`) is a small, **self-contained module whose one
job is to animate a vibrational normal mode** — the atoms oscillating along a
mode's displacement vectors. It is a *sibling* of MolView, built on the same idea
(a concealed 3D viewer exposed through a tiny handle) but applied to animation
instead of editing. The Spectra viewer mounts it; nothing else does.

## 1. The mount door and the handle

Like MolView, VibrationView is reached through **one exported door**:

```js
import { mount } from "/static/lib/vibrationview/vibrationview.js";
// (classic consumers use window.molbuilder.vibrationview.mount)

const vib = mount(host, { geometry, freeAtomIdx, frozenAtomIdx, amplitude, speedHz });
```

`mount` follows the same **uniform contract** as every viewer here: it *always*
returns a handle carrying `dispose`, so the caller can tear down unconditionally.
Failure is `{ ok: false, error, dispose }`; success is `{ ok: true, … }`. The
handle exposes only behaviour, never internals:

```
showMode(mode)   play()   pause()   isPlaying()
setAmplitude(Å)  setSpeed(Hz)  getMode()  dispose()
```

Defaults: amplitude **0.15 Å**, speed **1.0 Hz**.

## 2. The animation model — owned here

VibrationView owns the physics of the animation (`vibrationview.js`):

```
pos_i(φ) = equilibrium_i + amplitude · cos(φ) · displacement_i
```

The phase `φ` is **continuous across pause/play** (resuming never jumps), and
amplitude and speed are **live** — a slider change takes effect on the next frame
with no rebuild. The equilibrium baseline is redrawn only when the geometry or the
frozen set actually changes, so browsing mode-to-mode of one structure keeps the
camera still. **Frozen atoms** are greyed (`#555`) and never move (zero
displacement).

The one science-shaped piece it owns is **scattering the eigenvector** into a
per-atom displacement array (free rows → the mode's global vector; frozen rows →
`[0, 0, 0]`), in `lib/vibrationview/mode-math.js`.

## 3. The seal — semantic, over a drawing surface

VibrationView owns the animation **clock, the knobs, and the tick math**, and
drives its drawing surface through **generic** doors only:

- `handle.setAtomCoords(coords)` once per tick — the frame's positions;
- `handle.setAnimationProvider({ frameCoords, restCoords, cycleSec })` for export
  capture (gif/webm) — the surface drives the capture clock, VibrationView decides
  what each captured frame looks like.

It draws with picking off and axes off — the surface holds **zero
vibration-specific concern**. That's the point of the design (task #51 Phase 2):
the seal is **semantic**, not a second 3Dmol wrapper. The drawing surface's old
built-in `kind:"vibration"` animation was deleted when VibrationView took
ownership.

## 4. Who drives it — the Spectra viewer

The spectra viewer (`lib/spectra/core.js`, [`spectra.md`](?doc=web/spectra.md))
mounts VibrationView once, then on each mode you click calls
`vib.showMode({ index, displacements, geometry, freeAtomIdx, frozenAtomIdx })`; the
amplitude/speed sliders call `vib.setAmplitude` / `vib.setSpeed`; play/pause calls
`vib.play` / `vib.pause`; a geometry change or unmount calls `vib.dispose()`.
**The spectra viewer owns the control widgets, the chart, and the mode list;
VibrationView renders no control UI** — it only animates. It runs identically on
`/results` and `/spectrum-calculation`. (On the Spectra tab this animation box is a
*different* 3D surface from the read-only MolView "inspect structure" card — see
[`spectra.md`](?doc=web/spectra.md).)

## 5. Where the module stands (current → target ESM + separation)

VibrationView is a **full ES module** already (`vibrationview.js` +
`mode-math.js`). What is **not** yet clean is its 3D drawing surface:

- **Today:** it draws by *borrowing* a shared 3Dmol embed surface at runtime —
  `root.molbuilder.viewer.embed(...)` (a transitional global published by
  `lib/viewer/`, the same surface MolView is built on). So MolView and
  VibrationView are not yet fully independent: they share one drawing engine.
- **Target (task #104):** **complete separation.** `lib/viewer/` becomes
  MolView-private (moves under `lib/molview/`), the transitional `molbuilder.viewer`
  global is dropped, and VibrationView gets its **own minimal, concealed 3Dmol
  seal** — a small purpose-built wrapper providing exactly the doors it uses
  (`setStructure` / `setAtomCoords` / `setOverlays` / `refit` /
  `setAnimationProvider` / `dispose`), since it uses none of MolView's heavier
  embed (no selection, axes, cell, labels, style menu). When that lands, this
  section flips to "owns its own seal" and the borrow note goes away.

The full plan (files moved, the minimal-seal surface, template + test updates,
and the real-browser verification it needs) lives in task #104 and
[`roadmap.md § 3`](?doc=roadmap.md).

## 6. Test map

- `test_vibrationview_mode_math_js.py` — the eigenvector-scatter math
  (`mode-math.js`).
- The spectra suite (`tests/spectra/`, [`spectra.md`](?doc=web/spectra.md)) exercises
  VibrationView through its one consumer end to end.
