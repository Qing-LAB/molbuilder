# F12 review, round 1 — slice J2: the Results tab and its inspectors

Checkout `the checkout` at f578482db (2.77.2), read only.

Chapters read end to end: `docs/web/results.md`, `docs/web/trajectory.md`,
`docs/web/tabs.md` (§ 1, § 4, § 7–9; the Results and Transport parts).
Code read whole: `static/results/viewer.js`; `lib/results/{file-picker,run-panel,file-card,point-words}.js`;
`lib/inspectors/{registry,lifecycle,_partial_inspector_factory,structure,trajectory,spectra,fc-sweep,bench-summary,transport}.js`
(+ `source.js`/`markdown.js` registration); `lib/trajectory/core.js`; `lib/transport/{core,fit}.js`;
`spectra/viewer.js`; `templates/results.html`, `_trajectory_inspector.html`,
`transport_calculation.html` (structure); plus, to trace claims, `lib/chart-export.js`,
`lib/plot-box.js`, `lib/scfplot/scfplot.js`, `lib/plot-theme.js`, `lib/state-chip.js`,
`lib/system-load-monitor.js`, `lib/projects/downloads.js`, and on the server
`runs.py` (`folder_answer`, `file_row`, `_opens`), `jobset/runstatus.py`
(`points_view`, wire forms), `web/blueprints/{results,watch}.py`, `spectra/export.py`,
`transport/record.py`, `transport/compose.py`, `parse/sidecars/*.py`.

**Totals: 8 defects (7 read-and-traced, 1 plausible), 5 untrue sentences (4 read-and-traced, 1 plausible).**

Checked and found consistent (no finding): the one-scan/one-announcement picker and its
keep/door/none decision; Reload forcing a remount; coming back re-reading without re-pointing;
the bound-folder header and its divergence warning; the ladder (rows, standing, scratch line,
copied-from, runs, `input`/`check report`, next step and its read time, pick in place, rescan
of a picked run, *back to the calculation*, a transmission row opening the report at its point);
the Run panel (generic field table, closed line, four sections, run of points' lead line,
hidden on a scope change); the file card; the transport report's two selections, spin colours,
frame axis, PDOS steps, digest card, convergence card, rungs table; chart Export wiring (plot box,
tags, thresholds, picture name from the offer); the server-load strip (§ 6); the Transport tab's
send gate, refusal placement, reload note, extension parts, rail and routes (tabs.md § 4).

---

## Defects

### D1 — The calculation's Export bar keeps the first selection's path; a later selection in the same calculation exports something other than what is on screen (high)

- **Code:** `molbuilder/web/static/results/viewer.js:549-567`, with `lib/projects/downloads.js:100-107`.
  ```js
  // viewer.js _drawExport
  proj.exportBar(path, { asks: () => (shownAsks ? shownAsks() : {}) })
      .then((bar) => { ...
          const same = bar && shown && bar.dataset.calculation
              && shown.dataset.calculation === bar.dataset.calculation;
          if (!same) els.actions.replaceChildren(...(bar ? [bar] : []));
  // downloads.js exportBar -- the bar's buttons close over the path it was built with
  const run = async (choice) => { ...
      const q = Object.assign({ path }, choice);
  ```
  The server's export depends on that path: `web/blueprints/results.py:125`
  `folder=path if path.is_dir() else path.parent`, and the vibration export takes "the run the
  viewer shows" from it (`spectra/export.py:51-83`, `result_path`: `on_label_at(folder, ...)`,
  else the stage's newest run, and a multi-stage displacement sweep is refused *"export it from
  the Results tab with one stage's run open"*).
- **Chapter:** results.md § 1.1 — *"The controller asks the calculation's offer for the
  selection's path"* … *"read as a button is pressed, so the export is of what is on screen."*
  The *"one bar stays"* rule keeps the bar's words, but the bar also keeps the old path.
- **Scenario (read-and-traced):**
  1. A single-stage SIESTA vibration with `run-0` and `run-1`. Open its root (bar built with the
     root's path → the server exports the newest run, `run-1`). Pick `run-0` off the ladder: the
     viewer shows `run-0`'s spectrum, the offer for `run-0`'s file names the same calculation, so
     the old bar stays → **Export downloads `run-1`'s spectrum**, with the laser/width read from
     `run-0`'s Raman view (`shownAsks`).
  2. A displacement-sweep vibration (several force-constant stages): at the root, Export is
     refused *"export it … with one stage's run open"*; pick a stage's run off the ladder → the
     same bar (root path) → **the same refusal, with a stage's run open on screen.**
  3. Bound to run A, Reload after moving the sidebar to run B of the same calculation → bar of A
     kept → B shown, A's result exported.
  (Optimization and transport exports ignore `folder`, so they are unaffected.)

### D2 — A trajectory load crashes with a TypeError whenever MolView refused to mount (medium)

- **Code:** `lib/trajectory/core.js:1911`
  ```js
  const countInSync = !_mv || (_mvdata().frameCount() === oldLen);
  ```
  `_mv` is MolView's mount answer (`core.js:547`); when the mount fails it is the fail handle
  `{ok: false, error, dispose}` (`lib/molview/mount.js:79-83`, returned for a host narrower than
  the card's floor, for no WebGL, for a missing vendor script), so `!_mv` is false while
  `_mvdata()` returns `null` (`core.js:535`). Every other use guards on `_mvdata()`.
- **Chapter:** trajectory.md § 4 — the status line carries *"a viewer that could not mount"*;
  results.md § 3 — *"a viewer that cannot load its file says so"*. The plots, badge and SCF line
  do not need the 3-D viewer.
- **Scenario (read-and-traced):** open a SIESTA `.out` in a browser with no WebGL (remote desktop
  / VM, which MolView's own message names) or a panel narrower than MolView's floor. The status
  says *"Viewer failed: …"*, then the load answer reaches `applyNewData`, line 1911 throws, the
  catch in `loadByPath` (`core.js:2213-2219`) overwrites the line with *"Cannot read properties of
  null (reading 'frameCount')"*, the machine goes to ERROR, and **no plot, badge or SCF line is
  ever drawn** for the run.

### D3 — A run's index and its notification record are never offered in the Result file menu (medium-low)

- **Code:** `lib/results/file-picker.js:132-140` (a file is listed only when
  `pickResult(fullPath, rec)` names an `isResult` presenter) and `lib/inspectors/registry.js:166-170`.
  The parsers `run-points-json` (`parse/sidecars/run_points.py:33`) and `run-notified-json`
  (`parse/sidecars/notified.py:29`) are claimed by no presenter: bench (`job-set-sweep`),
  transport (`transport-task`), fc-sweep (`fc-sweep-json`), trajectory (role + `molwatch`/
  `siesta`/`pyscf`), spectra (`spectra-json`), structure (`.xyz/.pdb`), markdown (`.md`) all
  decline; `source.js:41-49` claims `.json` with `isResult: false`, so `pickResult` returns null
  and the row is dropped.
- **Chapter:** results.md § 0.1, run-of-points row — *"the run's index — every run of points' one,
  offered by its registry parser `run-points-json` … — and its notification record listed beside
  it as aspects, never opened first."* (The parsers' own docstrings say the same: *"so the
  Results tab lists it as an aspect of the run"*.)
- **Scenario (read-and-traced):** open a SIESTA vibration's `run-1/`: the menu lists the spectrum
  or the point at work's file and the folder's result files; `<label>_01_freq.points.json` and
  `notified.json` do not appear. They are reachable only through the sidebar's file viewer.
  (Either a presenter must claim them, or the sentence — and vibration.md:2611 — is wrong.)

### D4 — A trajectory with no frames yet never signals "ready" and never draws its badge (medium-low)

- **Code:** `lib/trajectory/core.js:2017-2021`
  ```js
  const n = state.data.frames.length;
  if (n === 0) {
      setStatus("File loaded ("+ state.label +") but no frames yet.", "");
      return;
  }
  ```
  This returns before `makePlots()`, `_announceReady(...)` (`core.js:2085`) and `_renderBadge()`
  (`core.js:2087`). The server sends such a file with `structure: None` and empty frames
  (`web/blueprints/watch.py:204-213`).
- **Chapter:** results.md § 3 — *"the cover lifts when the viewer signals it has painted, or failed
  to (a viewer that cannot load its file says so and signals too), or after a 15-second safety
  timeout"*; trajectory.md § 4 — the badge reads the run: *Stopped* with its reason, *Queued*,
  *Not launched*.
- **Scenario (read-and-traced in the code; *plausible* that the SIESTA reader gives zero frames
  for an output cut before its first coordinates):** a SIESTA run that dies on an fdf error opens
  its `.out` (the run door's choice when no product exists, § 0.1). The 78 % cover stays 15 s over
  the panel (hiding the status line under it), the picker says *Parsing …* for 8 s, and then the
  panel shows *"File loaded (…) but no frames yet"* with **no *Stopped* badge and no reason** —
  a failed run reads as one still to write. A queued run with no frames shows no *Queued* badge
  until its first quiet poll 15 s later; a pending one never shows *Not launched*.

### D5 — The trajectory status line repeats a frame count and a time on every quiet poll (low)

- **Code:** `lib/trajectory/core.js:1436-1442`
  ```js
  if (!r.changed) {
      _withRunNotes(
          "Up to date — " + state.data.frames.length
          + " " + (state.label || "") + " frames "
          + "(checked " + new Date().toLocaleTimeString() + ").", "ok");
  ```
  `_withRunNotes` → `setStatus` → `#trajectory-status`.
- **Chapter:** trajectory.md § 4 — *"It is otherwise empty. It does not repeat the file's
  modification time or a frame count — the badge below carries the run's time, and the frame bar
  its frames."*
- **Scenario (read-and-traced):** any live run: every 15 s with nothing new, the line reads
  *"Up to date — 7 SIESTA frames (checked 14:05:03)."*, replacing the cell-provenance line, which
  comes back only on a poll with new frames. Also on coming back to the tab for a finished run.

### D6 — The SCF line never names the phase of a TranSIESTA device (low)

- **Code:** `lib/trajectory/core.js:1338-1375` — `statusText` is built from the step, the cycle,
  the iteration count, each residual, ΔE and the rate; the row's `phase` is read only to pick the
  timing figure (`core.js:1365`), never written into the line.
- **Chapter:** trajectory.md § 2 — *"The last is shown by default and the SCF line (§ 4) names it"*;
  § 4 — *"the residual and ΔE, and for a device its phase (§ 2)."*
- **Scenario (read-and-traced):** open a TranSIESTA device's `.out`: the line reads
  *"SIESTA DFT SCF progress — CG/MD step 0 — SCF cycle 1000 (1007 iters), dDmax=…, dHmax=…"*;
  nothing says these are NEGF cycles rather than the periodic start (the `(1007 iters)` also
  counts both phases' rows together).

### D7 — The fitted window's *Use these* can write a fit for a range already changed (low)

- **Code:** `lib/transport/fit.js:51-61`
  ```js
  fit.addEventListener("click", async () => {
      fitted = null; take.hidden = true; say("Fitting…");
      ...
      const a = await ask("/api/transport/fit?" + q);
      if (!a.ok) { say(a.error, "error"); return; }
      fitted = a.body;
      say(fitted.words);
      take.hidden = false;
  ```
  The input listener (`fit.js:44-50`) clears `fitted` (*"A RANGE CHANGED is another question: the
  fit it had answered is gone"*), but nothing stops the in-flight answer from setting it again.
- **Chapter:** tabs.md § 4 (part 4/3 of an extension) — the window *"fitted to the finished grid —
  the range asked and how many times finer, the fitted ends and count the server answers"*.
- **Scenario (read-and-traced):** type 1 to 2, press *Fit*, change *to* to 3 before the answer;
  the 1–2 answer lands, *Use these* reappears under fields reading 1–3, and pressing it writes the
  1–2 window into the transmission tab.

### D8 — A run picked off the ladder can mount a file the folder's own scan would not open (low; plausible)

- **Code:** `results/viewer.js:139-169` — `_pickRun` announces `a.opens` whenever it is set; when
  it is not among the listed results, `listed` is null, `meta` null, and `reg.pick(file, null)`
  (`viewer.js:691`) falls to suffix-matching inspectors (structure, markdown, the `source` text
  pane). The scan of the same folder (`file-picker.js:768-772`) opens only a listed file and
  otherwise says *"nothing here is this directory's result"*.
- **Chapter:** results.md § 2.4 — *"the file that run's own folder would open"*; § 3 — the markdown
  editor and text pane *"on this tab … never mount: nothing reaches a viewer except through the
  dropdown"*.
- **Plausible — what is missing:** a run whose `openable` is a file no result presenter claims
  (parser null, or absorbed). Not traced to a real kind; the two code paths simply disagree.

---

## Untrue sentences

### U1 — results.md § 0.3: *"The page shows what it is sent, and composes no path … so the page joins none to another"* (read-and-traced)

`lib/inspectors/transport.js:1422-1424`
```js
const res = await proj.parser.openMolecule(
    handle, state.dir ? join(state.dir, junctionFile) : junctionFile, ...
```
`state.dir = dirname(file)` (`transport.js:1487`) and the record sends the bare catalogue name
(`transport/record.py:1314`, `"junction_file": JUNCTION_FILE` = `"junction.xyz"`). It works today
(the junction lies beside `task.json`, `compose.recorded_junction`), but the page joins two paths
— the one place in the slice that does.

### U2 — results.md § 2.4: *"when the panel shows another file of the calculation, `<label>.transport.json` opened as a file, the ladder says so"* (read-and-traced)

`<label>.transport.json` has parser `transport-json` (`parse/sidecars/transport.py:46`), which no
presenter claims (the transport report matches only `transport-task`, `transport.js:1481`), and
`source.js` claims `.json` with `isResult: false`: it is never in the menu and never shown in this
panel. The ladder's message (`viewer.js:479-482`) appears for other files of the root — the
junction's `.xyz` in the structure preview, for instance.

### U3 — trajectory.md § 4: *"First, the panel's own status line (`#trajectory-status`, above the badge)"* (read-and-traced)

`templates/_trajectory_inspector.html:26` (the badge) comes before `:74` (`#trajectory-status`),
with the parse-warnings panel between; `lib/trajectory/trajectory-inspector.css` does not reorder
them. The line is below the badge.

### U4 — trajectory.md § 3: *"A small label under the plot even says which of these it read them from"* (read-and-traced)

The label is `#convergence-summary-source` (`_trajectory_inspector.html:78`, filled at
`core.js:892-895`) in the header of the convergence-summary band, which sits above the viewer
card and the plots — not under a plot.

### U5 — results.md § 2: *"a report outside it is not mirrored, and the sidebar highlights nothing"* (plausible)

`file-picker.js:846-860` skips `setShared` for the report, but clears nothing either; the
sidebar keeps whatever row it last highlighted (a single click, an earlier mirrored pick).
Missing: the sidebar's own highlight code (`lib/projects/*`), not read.

---

## The five most severe, one line each

1. D1 — the Export bar keeps the first selection's path: a vibration exports another run than the one on screen, or a displacement sweep stays refused with a stage's run open.
2. D2 — a trajectory load throws a TypeError at `core.js:1911` whenever MolView could not mount (no WebGL, narrow panel): no plot, badge or SCF line.
3. D3 — a run of points' index and `notified.json` are never listed in the menu, though § 0.1 (and the parsers written for it) say they are.
4. D4 — a trajectory file with zero frames never signals ready or draws its badge: 15 s cover, and a failed run shows no *Stopped* reason.
5. D5 — the trajectory status line writes *"Up to date — N frames (checked …)"* on every quiet poll, which § 4 says it never does.
