# F12 round 1 — slice K: MolView

Chapter read end to end: `docs/web/molview.md` (4941 lines).
Code read whole: every file under `molbuilder/web/static/lib/molview/` (index.js, _atom.js,
stores.js, history.js, ui-context.js, mount.js, model.js, model-jobs.js, render-engine.js,
3dmol-embed.js, ui.js, demo.js; molview.css for the sizing and fold claims).
Followed outside the slice only to trace a claim: `web/blueprints/_shared.py`
(`structure_answer`, `ok_structure_response`), `structure.py` (`to_dict` carries `info`;
vacuum is a 3-tuple), `periodicity_gate.py` (notice shape), `kind_table.labels()`,
`lib/workspace/dispatcher.js` (persist answers), `lib/trajectory/core.js` (how the Results
viewer feeds frames), `modify/viewer.js` (Molbuilder reopen calls `load(0)`), the vendored
3Dmol `setFrame` (swaps atoms synchronously — so the export-capture path is fine).

**Totals: 11 defects, 6 untrue sentences.**

---

## Defects

### D1 — After an edit that changes the atom count, the saved point and the draft keep the OLD selection. Retract or a reload then shows atoms selected that the person never picked. (high)

- `model.js:697` and `model.js:707`, inside `applyOp`'s `apply`:
  ```js
  recordEdit(checkpoint);                 // 697 — lays the point AND rewrites the draft
  ...
  if (countChanged) selection.clear();    // 707 — selection cleared only afterwards
  ```
  `recordEdit` → `history.save(1)` → `request` → `send`, which runs synchronously up to
  `const bytes = handed.recordState();` (`history.js:113`), and `recordState` reads
  `selection: selection.get()` (`model.js:220`). So the point and the draft are written with
  the selection from before the delete. The clear at 707 happens only in memory, and nothing
  writes the draft again until the next edit.
- Chapter § 11.2, the table under "State is the truth": *"Saved — it is the truth | the whole
  structure … And the selection: which atoms the user picked out"*. § 12.1: a Retract
  "puts it back".
- Failure: 10 atoms. Select #1 and #2 (indices 0 and 1) and Delete, which leaves 8 atoms.
  Point 1 is recorded with selection `[0,1]`. Translate, which records point 2. Retract
  returns to point 1, and `selection.adopt([0,1])` now marks atoms #1 and #2 of the
  8-atom structure (the old #3 and #4) as selected. The next Delete or Assign acts on atoms
  the person never chose. Reloading the Molbuilder tab right after the delete does the same:
  `modify/viewer.js` calls `load(0)`, which adopts the draft. If the deleted atoms were the
  last ones, the restored indices are out of range. The count then reads "2 of 8 selected"
  with nothing highlighted, and with isolate on the window draws nothing.
- Read-and-traced.

### D2 — A label write records its point and its draft before it sets `labels_modified`, so the flag is missing from both. (medium-low)

- `model.js:447-448`:
  ```js
  recordEdit(true);
  markContractOutdated("labels");   // sets info.calculation.labels_modified = true
  ```
  The state is deep-copied at `recordState` (`copy(structure)`), which runs synchronously
  inside `recordEdit`. The flag is set on the live object only afterwards.
- Chapter § 8.4a: *"When the store carries `info.calculation` and an edit lands, the flag is
  recorded beside the contract … both ride the pair like everything in the store."*
- Failure: a pair saved from Results carries `info.calculation`. It is opened in the
  Molbuilder tab, the person assigns `L-electrode`, then reloads the tab. `load(0)` adopts a
  draft whose `calculation` has no `labels_modified`. A save or export then writes a sidecar
  that says the region assignment was never changed. Retracting and then stepping forward
  onto that point gives the same result.
- Read-and-traced.

### D3 — Frames appended during a rebuild are drawn twice, and every later frame then shows the wrong geometry. (medium; the timing that triggers it is plausible)

- `render-engine.js:870-872`: during REBUILDING, `appendFrames(from)` is held as
  `() => doAppend(from)`. The rebuild it waited for, `doRebuild` (`render-engine.js:534-576`),
  processes **all** frames of the master copy, which already contains the appended ones.
  After that, the held `doAppend(from)` (`render-engine.js:593-601`) appends frames
  `from..total-1` a second time. `healIfShort` (`render-engine.js:643-648`) only catches a
  drawing that is **short**, never one that is too long.
- Chapter § 10.9: *"appended frames | held, and they accumulate"*. § 10.10: *"the pipeline's
  job is to make the drawing able to show every one"*. § 5.1: what you see is the frame you
  are on.
- Failure: the master copy has 10 frames. A rebuild starts (a poll's re-install, or an
  isolate toggle). A poll's `addFrames` lands before the rebuild's `setTimeout` fires, which
  holds `doAppend(10)` while the master copy grows to 12. The rebuild draws 12 frames, and the
  replay appends frames 10 and 11 again, so the drawing holds 14. The next poll adds master
  frame 12, which the drawing puts at index 14. Scrubbing to frame 13 (index 12) then shows
  frame 11's geometry, while the counter, readout and export all say frame 13.
- The mechanism is read-and-traced. *Plausible* reachability: it needs a poll answer inside
  the rebuild's one-timer window. That window is longest in a background tab, where the
  browser throttles timers to 1 s or more while network answers keep arriving.

### D4 — The Cell tab's dot comes back for notices the person has already opened. (low)

- `ui.js:2843-2850`:
  ```js
  const waiting = onlyFor("cell").length > 0;
  const show = waiting && pages.cell.hidden;
  ```
  Nothing records that the page was opened. Every `draw()` (`ui.js:2503`) and every
  `showPage` re-raises the dot whenever the Cell page is hidden and a cell notice exists.
- Chapter § 6.8: *"The Cell tab carries a dot while its page holds notices you have not
  opened; … the dot goes when you open the page."*
- Failure: load a structure that carries a cell warning. Open the Cell page (the dot goes),
  then return to Selection. The dot is back on the very next redraw, signalling an unread
  notice that has been read.
- Read-and-traced.

### D5 — The frame bar keeps showing "⏸ Pause" after playback stops by itself. (low)

- `mount.js:272-273`: at the last frame with loop off, the timer calls `stop()`, which
  announces only to `onPlaybackChange` listeners. The frame bar's `reflect()`
  (`ui.js:328-349`) is subscribed only to `onFrameChange` and `model.subscribe`
  (`ui.js:347-348`). It never hears playback stop, and no frame change follows the stop.
- Chapter § 1.1: *"`▶`/`⏸` play-pause"*. § 8.5: the bar reads *"play, pause, loop and
  speed, through the handle"*.
- Failure: Results tab, untick loop, press ▶. The movie reaches the last frame and stops, but
  the button still reads ⏸ with aria-label "Pause". A host calling `viewer.pause()` leaves it
  stale in the same way.
- Read-and-traced.

### D6 — With Show axes or Show unit cell on, no force arrow is drawn gold, and the force shading is squashed. (medium)

- `render-engine.js:507`:
  `embed.setArrows((processed.arrows || []).concat(world, cell));`. The triad arrows share
  the force arrows' bucket. `3dmol-embed.js:517-523` ranks **every** arrow in that bucket by
  length (`maxAt`), and `3dmol-embed.js:534-535` gives gold only to
  `i === maxAt && !a.color`. Every other arrow's shade and thickness are set relative to the
  longest one (`t = mag / maxMag`).
- Chapter § 1.1: *"the largest force is drawn gold, the rest shade dim-red to orange-red by
  relative size, so converging forces visibly shrink"*. § 11.6 keeps measurement arrows out
  of this bucket for exactly this reason: *"could be the longest and take the gold off the
  force that earned it. Two overlays, two buckets, no interaction."*
- Failure: an optimization in Results, forces at scale 1 with a largest force of 0.4 eV/Å.
  Turning on Show axes adds the 1.5 Å world triad, which becomes the longest arrow. No force
  arrow is gold, and all of them fall into the low bands. Turning on Show unit cell adds
  lattice vectors of 10 Å or more, and every force arrow drops to the dimmest band at the
  minimum radius.
- Read-and-traced.

### D7 — A three-atom measurement draws a false arrow when the middle pick is not drawn. (medium-low)

- `render-engine.js:186-191`:
  ```js
  const drawn = track.picks.map((i) => seat.get(i)).filter((d) => d !== undefined);
  ```
  Picks that are not drawn are silently removed, so the sealed layer receives a list that
  looks complete. `3dmol-embed.js:452-456` then finds `allDrawn === true`, and
  `3dmol-embed.js:476-488` draws an arrow between the two survivors.
- Chapter § 11.6: *"A pick that is not on screen costs the arrows, not their correctness.
  Under isolate a picked atom may not be drawn, and joining the two that are drawn would
  assert a step the user never made — so a broken chain falls back to marking what is
  visible."*
- Failure: isolate on with atoms A and C selected, ruler on. Click rows A, B, C in the atom
  list; B is not drawn. The readout says ∠A–B–C, but the window draws one arrow A→C, a step
  the person never made, and no mark at all for B. When the *first* pick is the hidden one,
  an arrow is still drawn although the chain is broken.
- Read-and-traced.

### D8 — A label written, or a cell committed, while a geometry edit is in flight is wiped out when that edit's answer lands. (medium; the timing is plausible)

- The one-in-flight guard is a variable local to `createEdits` (`model-jobs.js:851`,
  `if (running) return null;` at 856), and it covers only `applyOp`. `writeLabel`
  (`model.js:421-450`) and the cell door (`model-jobs.js:958-993`) never consult it. When the
  op's answer arrives, `put` replaces the whole structure (`model.js:578`,
  `structure = nextStructure`) with the server's result. The server built that result from
  the body read **before** the label or cell change (`model-jobs.js:900`, `readData()` at
  request time).
- Chapter § 11.1: *"One mutation in flight. A second edit started while one is still running
  is not started rather than interleaved. Two responses applying over each other produce a
  structure neither edit asked for, and a history state the user never saw."* § 6.9 lists
  *"one edit is already in flight (§ 11.1)"* among the `null` answers of a door that changes
  the structure.
- Failure: in the Molbuilder tab, start a `slab` or `append`, and while it is on the network
  press Assign `L-electrode` in MolView's own panel. The label lands and lays a point. The
  slab's answer then replaces the annotations, and the label is gone with no message. The
  history now holds a point with the label followed by a point without it. A cell commit made
  during the same window is lost the same way.
- The mechanism is read-and-traced. *Plausible* reachability: it depends on how fast a person
  clicks compared with the op's round trip, and I did not check whether the host blocks its
  own Cell form while an op runs. MolView's Assign button is not blocked.

### D9 — Shift-clicking a row's checkbox ticks one atom instead of selecting the run. (low)

- `ui.js:2611-2615`:
  ```js
  check.addEventListener("change", (e) => {
      e.stopPropagation();
      if (e.shiftKey && rangeAnchor !== null) { selectRun(...); return; }
  ```
  A checkbox's `change` event is a plain `Event` and carries no `shiftKey`, so this branch
  never runs. The row's own click handler, which does read `shiftKey`, returns early for the
  checkbox (`ui.js:2644`, `if (e.target === check) return;`). A shift-click on the tick box
  therefore toggles a single atom and moves the run's anchor.
- Chapter § 9.5 gesture table: *"shift-click a row | selects every row from the last row you
  clicked to this one, inclusive — and adds them"*.
- Failure: click row 3, then shift-click row 10's checkbox. Only atom 10 is toggled, not rows
  3–10. Shift-clicking the row's text does work.
- Read-and-traced (relies on standard DOM event types).

### D10 — Applying an edit to the whole metadata store as JSON lays one point per changed key, so one Retract undoes only part of it. (medium-low)

- `ui.js:2196-2205`: the raw-JSON Apply loops over the keys and calls `model.info.drop(k)`
  and `model.info.edit(k, …)` once per changed key. Each call runs `recordEdit()`
  (`model.js:899`, `model.js:909`), which is a separate `history.save(1)`. The bracket that
  exists for this case (`beginChange` / `endChange`, § 11.2) is not used.
- Chapter § 8.4a: *"…or edits an entry — or the whole store — as JSON … recorded on the
  history like any other edit — each lays a point of its own (§ 11.2), so Retract takes it
  back."* § 1.1: *"Retract always steps back exactly one edit."*
- Failure: in Raw JSON → Edit as JSON, change three entries and press Apply. Three points are
  laid. One Retract restores only the last entry, leaving a store with two of the three
  changes, a state that was never on screen. Two more Retracts are needed.
- Read-and-traced.

### D11 — On an emptied structure, the axis triad is switched back on after the person hid it. (low)

- `model.js:332-335`, in `settle`: every settle of a structure with no atoms calls
  `selection.setSwitch("showAxis", true)`, not only the one that emptied it.
- Chapter § 6.7a: *"The triad is turned ON, and never forced back off. … a person who hides
  it afterwards stays hidden, because a viewer that re-asserted the switch would be
  overriding a choice."*
- Failure: delete every atom (the triad turns on), hide the axes, then commit a cell on the
  Cell form, or press Assign. The triad comes back on.
- Read-and-traced.

---

## Untrue sentences (the code may be right and the chapter wrong)

### U1 — § 4 *"no internal function ever appears on the handle"*; § 9.2a *"so a tab reaches no internal door"*. (low)
`mount.js:323`, `const data = Object.create(model);`: everything on the model is reachable
from `viewer.data` through its prototype. That includes `viewer.data._attachRenderer`,
`drawnToOriginal`, `measurementPositions` and `pickAtom`, and
`Object.getPrototypeOf(viewer.data).selection` is the raw selection store with `toggle`.
Only the three store properties are narrowed.

### U2 — § 10.3 note: *"They are handed down unconditionally whenever they change … even while the cell is hidden … The visibility switch carries only a boolean … It never carries geometry."* (low)
`render-engine.js:518`: `embed.setCell(sw.showCell ? scene.cellBox : null);`. While the cell
is hidden, the drawing command receives `null`. The geometry stays in the engine's `scene`,
and the show/hide decision is made by sending geometry or nothing. The behaviour the note
protects still holds: the box is drawn at the right corner when it is turned on.

### U3 — § 13.3 row for § 8.4: *"every fact the panel draws arrives in one snapshot, including the pick order"*. (low)
`stores.js:145-159`: the selection snapshot carries no pick order. § 9.5 itself says the
click-order shadow is absent and that a test pins its absence. The row contradicts both the
code and § 9.5.

### U4 — § 6.2: *"One list names the shaped keys (`model-jobs.js`'s `SHAPED_METADATA_KEYS`), read by both translators"*. (low)
Only the inbound translator reads it (`model-jobs.js:234`). `structureForServer` writes
`otherMetadata` back and then the box keys, using `BOX_METADATA_KEYS`
(`model-jobs.js:487-493`).

### U5 — § 15: *"Deliberately not listed: the sealed layer. No consumer names its file and neither does this document (§ 4)."* (low)
The chapter names it at § 11.3, line 3900: *"(`3dmol-embed.js` `capture`)"*.

### U6 — § 13.4: *"There is no readiness signal and no test seam published anywhere"*. (low)
`demo.js:263`, `window.__molview = viewer.data;`: the in-repo demo page publishes the data
surface, and with it the model through its prototype (U1), as a global.

---

## Checked and found consistent (not findings)
Handle names (16) and the 18 needs; the read-only gate table, door by door; the frame-set
refusal; one-in-flight between two `applyOp` calls; the four routes; the notice routing by
`about`; the history machine (SETTLED/CHANGING/WRITING, refused points stepped past, a read
waiting on answers); the lane's `match` guard and its order after a draft restore; the fit
axis and its tie rule; the colours and constants named in § 6.5 and § 10.3; the contact
reference table against `data/contact_distance.json`; the card floor (368 px) and the fold
chevron; Export stems and formats; image capture at the size asked (3Dmol `setFrame` swaps
atoms synchronously, so a capture taken right after a seek shows the new frame).
