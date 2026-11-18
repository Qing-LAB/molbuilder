# F12 review, round 1 — slice L: the vibration and spectrum viewers

Checkout `the checkout`, commit f578482db (2.77.2). Read only; nothing run but
`grep`/`sed` on the checkout and the vendored 3Dmol/gif.js builds.

Chapters read end to end: `docs/web/vibrationview.md`, `docs/web/spectra.md`,
`docs/web/spectrumchart.md` (plus `results.md` § 4, `trajectory.md` § 2–3,
`engines/transport.md` § 3.8.1 where a claim led there).

Code read whole: `lib/vibrationview/` (index, _maths, _seal, _export, _style.css),
`lib/spectra/core.js` (all 4762 lines), `lib/spectrumchart/` (index, _maths, _seal,
_style.css), `lib/scfplot/scfplot.js`, `lib/frameset/card.js` + `card.css`; followed out to
`templates/_spectra_inspector.html`, `lib/inspectors/spectra.js`,
`lib/inspectors/_partial_inspector_factory.js`, `results/viewer.js` (mount per file),
`spectra/viewer.js`, `web/blueprints/spectra.py` (load door's infrared fields),
`spectra/infrared.py::source_of/asked`, `frame_average.answer`, and the vendored
3Dmol `setWidth/updateSize/resize/setBackgroundColor` and gif.js `setDelay`.

**Totals: 5 defects, 5 untrue sentences.** The SCF plot and the frame-set card match their
owning sections (`trajectory.md` § 2–3.0, `engines/transport.md` § 3.8.1); nothing found there.

---

## Defects

### D1 — The spectrum's selection highlight lands on the wrong stick, or vanishes, once the first lane has a mode with no value (medium-high; read-and-traced)

**Where:** `lib/spectrumchart/index.js:321-326` (redraw), `:334-345` (recolour);
`lib/spectrumchart/_seal.js:350-378` (lane bars), `:556-562` (recolour);
`lib/spectra/core.js:4119-4121` (Raman is channel 0), `:1593-1598` (`_ramanOf`).

**Rule:** `spectrumchart.md` § 6.3.1 — "the mode a click would pick lights up as the
pointer moves"; § 12 row "§ 6.3.1 — the pointer shows what a click would take | moving
over a band recolours that mode, and only it"; § 10 — "The chart never highlights on its
own, so the highlight cannot disagree with the viewer".

**What the code does:** the seal leaves every *pending* mode out of the first lane's bar
trace:

```js
lane.sticks.x.forEach((x, i) => {
    if (lane.sticks.state[i] === "pending") { pending.push(x); return; }
    drawn.x.push(x); ...
```

but the handle's cheap door recolours that trace with one colour **per mode**, by position:

```js
const states = modes.map((m) => stateOf(m, laneKnown(key), key));
...
surface.recolour(same ? null : states, readout);
// seal:
Plotly.restyle(surface, { "marker.color": [states.map((s) => colourFor(s, palette))] },
               [stickTraceIndex]);
```

So whenever lane 0 has p pending modes, the colour array is p entries longer than the trace
and shifted: bar j gets the colour of mode j of the full list. The first draw is right
(colours built per bar); the first hover or `setSelected` (a table click) misaligns it.

**Failure scenario:** a SIESTA vibration with *Raman for: Modes in a frequency range*
(the chapter's own advice for a gold junction, `spectra.md` § 5 step 3) — or one whose
optical calls are still running, or one with a failed side — with a laser applied. The
Raman lane (channel 0) is known; every mode outside the range has `_ramanOf` → `null`
(`return (r && r.raman) ? null : m.raman_activity_a4_amu;`) and is *pending*. Modes 1–4
pending, 5–10 picked, mode 7 selected: after one hover the six bars take the colours of
modes 1–6 — real bands 5–8 turn the "pending" grey, and the "chosen" colour (array index 6)
falls off the end: no stick is highlighted, while the table and the 3D viewer show mode 7.
Hovering paints the hover colour onto a neighbour's stick too.

---

### D2 — An animation export encodes what the viewer holds *now* but stamps what it held when it started; the spectra tab leaves everything live during it (medium; read-and-traced)

**Where:** `lib/vibrationview/index.js:319-338`; `_export.js:196-212, 364-365, 395-401`;
`_seal.js:321-378` (caption read live); `lib/spectra/core.js:3765-3815` (only the export
button is disabled), `:2769-2783` / `:3418-3505` (a mode click calls `setAmplitude`,
`showMode`, `play`), `:3966-3974` and `:4041-4050` (Size and Speed sliders),
`:2356-2365` (re-render disposes the viewer), `:4657-4661`.

**Rule:** `vibrationview.md` § 5.3 — "the export carries **what it was set to** in its own
metadata, so a caption written from the file is true"; § 12 — "Everything else — the
maths, the amplitude, the speed, the style, the caption — is what was on screen";
§ 14 — "the metadata carries the amplitude, the frame rate, the cycle length, the
normalization, the resolution and the frame count that produced the file".

**What the code does:** the metadata (and the file name, from `meta.index`) is a snapshot
taken when the export starts, but each frame is computed from the closure's live state:

```js
coordsAt: function (n) {
    return positionsAtFrame(structure.positions, disp, amplitude, n, r.framesPerCycle);
},
meta: { ..., amplitude: amplitude, norm: modeNorm, index: modeIndex },
```

`count = ctx.meta.framesPerCycle * cycles` is fixed too, while `r.framesPerCycle` may
change; the caption is composited from the live element. Nothing in the handle refuses
`showMode`, `setAmplitude`, `setCycleSec` or `setStructure` while `exporting` is true, and
the spectra tab disables only the *Save animation* button.

**Failure scenarios:**
- PNG-zip (the default, `toBlob` per frame is asynchronous) or WebM (one `setTimeout` per
  frame, seconds long): the person clicks another mode in the table during the export.
  The file holds frames of mode A then mode B, its caption switching mid-file, named
  `vibration-mode-A` with A's normalization and amplitude in its manifest. Dragging Size
  mixes amplitudes under one stamped amplitude; dragging Speed changes the cycle's frame
  count, so a "one cycle" file no longer closes on itself.
- The run is still being followed and a poll lands with a new fingerprint, or the person
  switches browser tabs and back (D3): `renderResults` disposes the viewer
  (`structure = null; disp = null;`), the next frame's `structure.positions` throws, and
  the status line shows, in red, "Cannot read properties of null (reading 'positions')";
  the export is lost.

---

### D3 — Coming back to the browser tab resets the person's view of the same file (medium; read-and-traced)

**Where:** `lib/spectra/core.js:4657-4663` (`_onVisibilityChange` → `loadByPath`),
`:1178` (`transition("LOADING")`), `:528-533` (LOADING empties results and
`selectedMode`), `:2315-2321` (default mode picked), `:2356-2366` (viewer disposed and
rebuilt), `:2974-2977` (level diagram redrawn at its full range).

**Rule:** `results.md` § 4 — "your **per-file view** (which frame or which mode you're
looking at — reset when you switch files)"; `spectra.md` § 7 — "Coming back to the tab
loads the file on screen once more, live run or not, **so a phase written while you were
away is drawn**". (The trajectory viewer, `results.md` § 4.2, *polls* on coming back and
keeps its view.)

**What the code does:** coming back runs the full file-switch path:

```js
if (document.visibilityState === "visible" && state.results !== null && state.fileState.path) {
    loadByPath(state.fileState.path);
}
// loadByPath -> transition("LOADING", ...) -> state.viewState.selectedMode = null;
```

**Failure scenario:** the person picks mode 23, turns the molecule, zooms the level
diagram, switches to another browser tab to read a paper, and comes back. The table, the
chart, the ES panel and the animation all jump to the default mode (the brightest Raman
mode), the 3D camera is re-fitted, the level-diagram zoom is gone — and a running PNG-zip
or WebM export dies with the TypeError of D2. Nothing about the file changed. (If the
chapter means coming back to be a clean reload, its § 7 sentence should say the view is
reset; as written, the reset is the per-file view `results.md` keeps until a file switch.)

---

### D4 — While a run is followed, every poll that brings anything new rebuilds the mode box and redraws the level diagram (medium-low; read-and-traced)

**Where:** `lib/spectra/core.js:2188-2200` (fingerprint gate), `:2407-2475` (fingerprint:
per-mode ES bits, activity sums, each Raman-view mode's state and strength, phases),
`:2356-2366` (any change → `state.vib.dispose()` and a new mount), `:2351`/`:2974`
(`renderESPanel` redraws the level diagram at its full range).

**Rule:** `spectra.md` § 4 — "**The mode box belongs to one result.** It is built the
first time a mode is shown, **kept while you move between that result's modes**, and built
afresh against the new structure **when another result loads**." `vibrationview.md` § 5.1 —
browsing "should leave the camera exactly where you put it. Opening a *different* result
should redraw and re-frame". `spectra.md` § 3.1 — "the figure opens on the whole picture
and then scroll to zoom".

**What the code does:** the gate stops only identical snapshots; any change to the same
result disposes and remounts the viewer ("Geometry changed (new results loaded)" says the
comment, but the geometry has not changed):

```js
if (state.vib) { _stopAnimation(); try { state.vib.dispose(); } catch (_) {}
                 state.vib = null; if (els.modeViewer) els.modeViewer.innerHTML = ""; }
renderModeViewer();
```

**Failure scenario:** a PySCF run with the orbital probe on `all`, or a SIESTA run whose
optical calls are landing with a laser applied, followed on the Results tab: each time a
mode's ES result or a mode's two Raman sides land (every few seconds to minutes), the 3D
view snaps back to the fitted camera and the animation restarts from its peak, and a
zoomed level diagram returns to its full range — exactly what the comment at `:2178-2187`
says the gate exists to prevent.

---

### D5 — Remembered preferences are put in the controls but not on screen (low-medium; labels read-and-traced, the race plausible)

**Where:** `lib/spectra/core.js:327-352` (`_prefsToControls`), `:400-403` (reload only when
the *ask* differs), `:3966-3970` and `:4041-4048` (the Size and Speed readouts are written
only by those handlers), `:4624` / `:4639-4640` (restore and load start together);
`templates/_spectra_inspector.html:359-372`; `results/viewer.js:707-708` (every file pick
mounts a fresh inspector, so this runs on every file opened).

**Rule:** `spectra.md` § 7 — "your view **preferences** (the broadening width, the animation
amplitude and speed and its pairing, the thermal temperature, the table's sort and filter,
…) **survive a page reload**, not just a file-switch"; the code's own note at `:312-319`:
"The controls must SHOW what was restored".

**What the code does:** `_prefsToControls` sets the slider values
(`put(els.animAmplitude, u.animAmplitude); put(els.animSpeed, u.animSpeed);`) but never
calls `onAnimAmplitudeChange`/`onAnimSpeedChange`, the only writers of
`#anim-amplitude-val` / `#anim-speed-val`; and it redraws neither the chart, the table nor
the viewer — a reload is asked only when the temperature, infrared or Raman ask differs.

**Failure scenarios:**
- (read-and-traced) A person who set Speed 2.0× and Size 0.30 Å opens any spectrum: the
  animation runs at 2× and 0.30 Å, the sliders sit there, but the readouts beside them say
  "1.0×" and "0.15 Å" (the markup's values).
- (plausible — depends on which of the two requests answers first) When the file's load
  answers before the workspace read, the chart stays at 20 cm⁻¹ while the FWHM box shows
  the remembered 42, the table is neither filtered nor sorted while the filter box holds the
  remembered text, and the viewer keeps 0.15 Å / 1.0× while the sliders show the
  remembered values — until some later redraw silently applies them.

---

## Untrue sentences

### U1 — `vibrationview.md` § 11 (line 735-736) and § 6.3 (line 360-361) (read-and-traced)

> "…installing a structure ends the mode that was running against the old one (§ 5.1),
> and a mode that cannot be shown stops it too."
> "(`index.js` `showMode`: the scatter refuses, the viewer warns and draws nothing)"

`lib/vibrationview/index.js:239-247` — on a refused mode it only warns and returns:

```js
} catch (e) {
    warn((e && e.message) || "the mode does not fit this structure");
    return false;
}
```

The clock is not stopped and `disp`/`labelText` are not cleared, so the **previous** mode
keeps animating with its caption (and the spectra tab then calls `play()` anyway). Rarely
reached on the road, since `SpectraResults` refuses a file whose rows do not partition the
atoms.

### U2 — `vibrationview.md` § 12.1 (line 869-871) and § 13 (line 1072-1074)

> "its ground is the stylesheet's one dark ground, the same in the app's light and dark themes"

The app has no light or dark theme: no `prefers-color-scheme` or `data-theme` anywhere in
`web/static` CSS, and `spectrumchart.md` § 11 says the opposite — "the app has one
palette, with no light/dark switch to follow".

### U3 — `spectra.md` § 3 (line 196-200)

> "a box in a hidden tab has none — so opening a tab re-fits the viewer, and a chart sees
> its box come back by itself"

Opening a tab only records its name — `lib/spectra/core.js:3949-3951`
(`function _modeTabShown(name) { state.modeTab = name; }`); nothing refits. The code
relies on the drawing library's container watcher for both (its note at `:3935-3942`:
"Both see their box come back by themselves"). The template's comment
(`_spectra_inspector.html:301-303`, "core.js re-fits the viewer … (_activateModeTab)") is
stale the same way.

### U4 — `spectra.md` § 2 (lines 75-78 and 88-91)

> "A slider broadens each stick into a smooth Lorentzian peak (a single FWHM control…)"
> "Each mode carries an invisible band, as wide as the broadening you set"

The FWHM control is a number box (`<input id="broadening-fwhm" type="number">`,
`_spectra_inspector.html:161-162`), not a slider. And the band's **half**-width is the
broadening (`spectrumchart/_maths.js:30-33` returns the FWHM as `bandHalfWidth`;
`index.js:371-372` accepts a click within it on either side), so the band is twice as wide
as the broadening — as `spectrumchart.md` § 6.3 correctly says ("its half-width is the
broadening width").

### U5 — `spectrumchart.md` § 3 (line 158-159)

> "**Nothing flows back up past the seal** except a click — and, asked, the drawing as an
> image"

The seal also sends the pointer's position up continuously — `_seal.js:500-507`
(`mousemove` → `hovered(at …)`), consumed at `index.js:401-406` — as § 8.4's own list
(`onHover`) says. Low.

---

## Checked and found consistent (no finding)

VibrationView: the one entry, `3Dmol` named in `_seal.js` only, mount never rejects, the
failed handle, the rate band (5–120 fps, 15–1200 frames) and its clamp at mount and door,
`regrid`, the cycle rounding and reported `cycleSec`, the scatter's refusals, held-still as
the basis's complement, the kept windows and their reset at a take, the capture through
`setWidth/setHeight` (vendored 3Dmol `updateSize` does set the camera aspect), the three
encoders, the zip layout, the manifest fields, the GIF delay rounding (gif.js
`Math.round(ms/10)`: 3 → a 0.9 s cycle at 30 fps), the caption compositing and scaling.
SpectrumChart: the seven doors and `drawn`, refusal-empties-the-chart, recorded selection,
bands/nearest/8 px floor, drag-is-not-a-pick, positions picture, `presentFloor` "no band",
the envelope grid, the stylesheet/library loading, the frame height clamp. Spectra: the
bar and *Update* under the fence, commit-on-answer, the infrared/Raman notes and CSV
names, the no-spectrum sentence's cases, the run summary lines, the phase dots, the
thermochemistry headline/curves/bars by regime, the frame-set dialog and its door, the
follow and its guards, the speed slider's 2.0× top matching the 15-frame floor. The SCF
plot (one trace per run and phase, last shown, residuals as declared, criteria lines for
the phase on screen) and the frame-set card (mode line, frame numbers through `toDisplay`,
`means` on hover, warnings) match their sections.
