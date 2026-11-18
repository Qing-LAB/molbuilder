# F12 review, round 2 — R2-11: the Results and Transport pages' scripts

Checkout `the checkout` (2.77.2), read only. Paths below are under
`molbuilder/web/static/` unless they start with `molbuilder/`, `tests/` or `docs/`.

## What was read

**Whole:** `results/viewer.js`; `lib/results/{file-picker,run-panel,file-card,point-words}.js`;
`lib/inspectors/{registry,lifecycle,_partial_inspector_factory,structure,trajectory,spectra,fc-sweep,bench-summary,source,markdown,transport}.js`;
`lib/trajectory/core.js`; `lib/transport/{core,fit}.js`; `spectra/viewer.js`.
To check the other side of a finding: `lib/{dom,state-chip,time-ago,plot-theme,constants,structure-card}.js`,
`lib/projects/state.js` (`publishFolderChanged`), `lib/projects/downloads.js` (the offer's doc link),
`molbuilder/{units,monitor,stepsdata,frame}.py`, `molbuilder/jobset/summarize.py` (`sweep_view`, the terminal table),
`molbuilder/parse/dirs/{record,job}.py`, `molbuilder/parse/engines/{siesta_grammar,molwatch_grammar,_helpers}.py`,
`molbuilder/trajectory_log/keys.py`, `molbuilder/transport/{record,stages}.py`, `molbuilder/web/blueprints/transport.py`,
`molbuilder/data/kind_table.toml`, `docs/web/{results,trajectory,bench-summary,ui-contract}.md` (the passages cited).

**CSS** of the slice (`results/style.css`, `lib/results/*.css`, `lib/inspectors/*.css`,
`lib/trajectory/*.css`, `spectra/style.css`): NOT read whole — searched only for literal colours
and token fallbacks. The `var(--token, #hex)` fallbacks in `markdown.css` are the contracted
embed-safety pattern (`ui-contract.md` § 1), so not a finding.

**Tests, whole:** `test_results_file_picker_js.py`, `test_results_file_picker_e2e.py`,
`test_trajectory_transition_js.py`, `test_transport_describe_request_js.py`,
`test_inspector_lifecycle_teardown.py`, `test_inspector_registry_e2e.py`,
`test_bench_summary_inspector_e2e.py`, `test_bench_card_reads_the_metrics_python_writes.py`,
`test_coming_back_to_the_tab_e2e.py`, `test_results_export_e2e.py`, `test_results_views.py`,
`support/results_export.py`. **In part:** `test_the_viewers_follow_a_real_run_e2e.py` (lines 280–440,
1150–1215, and a search for `lifted`), `tests/data/presenters.toml` (1–80 and a search),
`test_xss_audit.py` (its head: an artifact lint `testing.md` § 3a allows), `support/road.py` (195–240).
**Not read** (only searched for source reads, hand-written input and copied fixtures):
`test_transport_on_a_real_junction_e2e.py` (6143 lines), `test_spectrum_results_e2e.py`,
`test_siesta_{stopped,relax}_run_e2e.py`, `test_siesta_vibration_e2e.py`, `test_keyboard_paths_e2e.py`,
`test_pages_no_js_errors.py`, `test_build_e2e.py`, and the Python route tests of the transport
and results blueprints (other slices' modules).

Round 1's J2 findings (D1–D8, U1–U5) are not repeated. D2 is cited once, under F10, because a
duplicate causes it.

**Totals: 30 findings — 23 in the code, 7 in the tests.** Each says how sure I am.

---

## The two questions the caller asked

**Do page scripts compute a fact the server already answers?** Yes, in eight places:
the bench page's node census (F3), the memory unit word (F2), the engine's words in the
trajectory panel (F5), the words for a target's source (F6), SIESTA's stress order (F7), the
post-processing document's path (F8), the folder the digest is written to (F9), and a rung's
Fermi level, which the page chooses again (F19).

**Do the inspectors each keep their own copy of one helper?** Yes:
- the "load has ended" signal, written by hand in seven places beside the shared one (F1);
- the read-only MolView mount and its failure check, in three inspectors (F10);
- a number-or-dash formatter, in four files, which already disagree on one figure (F16);
- an element builder, in two files of the Transport page (F11).

---

## Code findings

### F1 — The "load has ended" signal is written by hand in seven places beside its one door (high; read all sides)

The door is `lib/inspectors/lifecycle.js:76-99` `announceReady(inspector, detail)`. Its own
description calls it one of *"the two helpers every inspector core needs"*. It fires once,
after two animation frames or 250 ms, *"so the browser paints first"*. Only the trajectory
core calls it (`lib/trajectory/core.js:618-620`). The others dispatch the event themselves:

- `lib/inspectors/structure.js:116-127` and `:234-251` copy the whole door: a `readyFired`
  flag, then `requestAnimationFrame(() => requestAnimationFrame(dispatch)); setTimeout(dispatch, 250);`.
- `lib/inspectors/fc-sweep.js:219-222`, `bench-summary.js:651-655` and `transport.js:1526-1529`
  each call `root.document.dispatchEvent(new root.CustomEvent(... .EVENT_INSPECTOR_READY || "molbuilder:inspector:ready", { detail: { inspector: "…" } }))`
  directly, at once, before the browser has painted.
- `lib/inspectors/registry.js:244-250` (`showError`) and `_partial_inspector_factory.js:86-93`
  (`_announceEnded`) dispatch `{ error }` with no `inspector` name.
- `results/viewer.js:679-683` dispatches `{inspector, unchanged: true}`.

Five of these also repeat the event's name as a literal fallback,
`|| "molbuilder:inspector:ready"`. The name's home is `lib/constants.js:25`.

**What goes wrong.** The seven senders already disagree on timing (painted first, or not) and
on what the event carries (with or without the inspector's name). The tests read those fields:
`test_inspector_registry_e2e.py:291-292` waits on `d.inspector === viewer || d.error || d.unchanged`.
Any change to the signal must be made seven times, and a renamed constant still answers to the
old literal in five of them.

**Smallest fix.** Every sender calls `inspectorLifecycle.announceReady(name, detail)`; an
inspector with nothing to paint passes a flag that fires at once. Drop the literal fallbacks.

### F2 — One memory figure gets two unit words, "GB" and "GiB", from four hand-typed spots (high; read both sides)

**The value.** `mem_peak_gb` and `gpu_vram_peak_gb` are binary gigabytes (GiB):
`molbuilder/monitor.py:886,891` `round(b / 1073741824.0, 2)`, and `monitor.py:1177` MiB / 1024.

**The owner of the word.** `molbuilder/units.py:134-142` `size_words`. The architecture names
it in § 3.2: *"a size in bytes, in words — `units.size_words`… (`49.2 GiB`)"*.

**What the pages say:**
- `lib/inspectors/bench-summary.js:52` `_num(m.mem_peak_gb, " GB")` and `:58` (VRAM) say "GB";
  `:100` titles the chart axis for the same field `"GiB"`; `:567` explains it as `"GB — the PEAK reached"`.
- `lib/results/run-panel.js:51-54` `gb(v)` gives `"… GB"` for `mem_peak_gb` and `mem_limit_gb`
  (`:197-198`).
- `lib/trajectory/core.js:1600-1605` turns `max_memory_mb` into "GB" by dividing by 1024.
- The terminal writes `"{:.1f}G"` (`molbuilder/jobset/summarize.py:452,458`).

**What goes wrong.** On one bench card the bar's axis says GiB while the card and the caption
say GB for the same number. That is a 7 % gap between the two meanings of the word, on exactly
the figure the caption tells a person to size the run script from.

**Smallest fix.** The server sends each figure with its words from the one unit table, and the
pages print those words.

### F3 — The bench page counts nodes itself and keeps its own lists of which fields repeat (medium; read both sides)

**The census.** `lib/inspectors/bench-summary.js:498-507` builds a census of node core counts
from each trial's `effective.node_phys_cores` and writes `"N-core node"` or `"a/b-core nodes"`.
That is the census the server already answers as `machines`
(`molbuilder/jobset/summarize.py:627-648`, `machine_census`). Its comment at `:616-618` says the
brief is *"SPELLED HERE, because the page composes nothing (B1): a second brief-rule in JS would
be two spellings of one node"*. `docs/web/bench-summary.md` B2 says *"a second path that
computes the same figure is the defect"*.

**The field lists.** `bench-summary.js:418` `ALREADY_IN_KNOBS = ["mpi_np", "omp_threads", "cpus_per_task"]`
and `:425` `NODE_FACTS = [...]`. These decide, in the page, which fields of `effective` repeat a
field of `knobs`. `_knobLine` (`:68-77`) and the terminal's table (`summarize.py:431-458`, *"as the
page does (`bench-summary.js`)"*, `:436`) each compose the same knob line by hand.

**What goes wrong.** For a sweep whose record has no `[MACHINE]` line, the page shows a node
census the terminal does not print. If `effective` gains a field that repeats a knob (a GPU
count, say), the card prints it twice under two names until someone edits the list.

**Smallest fix.**
- The fallback census belongs to `machine_census` on the server, if it is kept at all.
- `sweep_view` sends `effective` without the fields `knobs` already states, and sends the knob
  line's words.
- The page prints what arrives.

### F4 — The bench chart rebuilds the shared chart theme by hand and types its own winner colour (medium-low; read both sides)

**The theme.** `lib/inspectors/bench-summary.js:191-226, 235-251` sets
`paper_bgcolor`, `plot_bgcolor`, `font` and `gridcolor` itself. That is what
`lib/plot-theme.js:46-60` `themedLayout` does, and the transport report and the trajectory
plots call `themedLayout`.

**The colour.** `:309` outlines the winning trial in `"#f5d76e"`, and the caption `:563` calls it
*"The gold outline is the winner"*. The same card draws the winner with
`border-color: var(--success)` (`bench-summary.css:80`). No other literal colour appears among the
slice's scripts or sheets.

**What goes wrong.** A theme change (dark or light, a new grid token) reaches every chart but
this one. The winner is gold on the chart and green on its card.

**Smallest fix.** Use `themedLayout(layout, theme)`, and take the outline from the token the
card uses.

### F5 — The trajectory panel types engine words by branching on the engine's name (medium; read both sides)

`lib/trajectory/core.js:1024-1048` (`_renderEmptyStatus`) and `:1325-1335` (`renderScfProgress`)
branch on `state.format === "pyscf"` / `=== "siesta"` and type `"SIESTA DFT SCF progress"`,
`"PySCF SCF progress"`, `"CG/MD step"`, `"Geom-opt step"` and a paragraph per engine.

**The owner.** The engine table: `Engine.words` in `molbuilder/engine_table.py:59`. It is
already served with the folder answer as `engine_words` (`molbuilder/runs.py:1494,1515`), and
the same inspector's menu heading reads it: `lib/inspectors/trajectory.js:84-86` builds
*"the engine's words and the calculation's noun… never guessed"*. This is a branch on a name in
shared code where a table row should decide.

**What goes wrong.** The heading says one thing, from the table; the banner says another, from
the page. A third engine falls to generic words. Renaming an engine's words in the table leaves
the banner as it was.

**Smallest fix.** Serve `engine_words` with `/api/watch/load`, and drop the per-engine prose. If
a step's name per engine is wanted, it becomes a column of the engine table.

### F6 — Pages put their own words on codes the server sends, beside rows the server words itself (medium-low; read both sides)

**The trajectory band.**
- `lib/trajectory/core.js:888-895` `sourceMap = { siesta_input_echo: "from SIESTA input echo (redata:)", molwatch_header: "from the run log's header" }`.
  The codes come from the parsers (`parse/engines/siesta_reader.py:1292`,
  `molwatch_grammar.py:201`).
- The same band's target rows are the server's (`molbuilder/trajectory_log/keys.py:75-86`
  `targets_said`, *"so the page spells no key and no unit"*).
- The page composes the band's SCF criteria rows itself (`core.js:904-914`:
  `"SCF " + res + " tol" + …, c.tolerance.toExponential(2)`).

**The same pattern on the Transport page.** `lib/transport/core.js:1788-1805` turns
`source.kind` (`deck` / `pair` / `reuses` / `none`) into sentences in the page.

**What goes wrong.** A new source code shows as *"from <raw code>"*. One band holds rows from
two composers, formatted two ways.

**Smallest fix.** Send the source's words beside the code — `targets.source_said`, and the
shared panel's `source.said` — and send the criteria rows with the target rows.

### F7 — The Run panel decodes SIESTA's stress order by array index (medium-low; read both sides)

`lib/results/run-panel.js:119-123`:
`inPlane(voigt) { return \`xx ${voigt[0]} · yy ${voigt[1]} · xy ${voigt[5]} kbar\`; }`

The order's home is `molbuilder/stepsdata.py:30` `VOIGT = ("xx","yy","zz","yz","xz","xy")`
(and `frame.py:147-148`). The panel's own header says it *"knows no engine"*.

**What goes wrong.** Another engine's record, or a reordered vector, prints the wrong
components under the right labels, with no error.

**Smallest fix.** The record (`parse/dirs/record.py:269-270`) sends the components by name
(or the in-plane line), from `VOIGT`.

### F8 — Two names the server holds are typed into the page: a document's path and `task.json` (medium-low; read both sides)

**The document's path.** `lib/inspectors/transport.js:237`:
`how.href = "/documents?doc=engines/transport-postprocessing.md"`.
Its home is `molbuilder/data/kind_table.toml:225` `export_doc = "engines/transport-postprocessing.md"`,
which the page's Export bar reads from the server's offer (`lib/projects/downloads.js:128`
`"/documents?doc=" + offer.doc`). If the document moves, the kind table follows and the bar
follows, while the report's link breaks.

**The file name.** `results/viewer.js:479-482` tells the person to *"choose task.json in the
Result file menu"*. That file's name is the server's (each menu row carries `name`) and its
heading is the transport presenter's (`"Transport"`).

**Smallest fix.**
- The report takes the link from the record or the offer.
- The ladder's hint names the heading under which the report sits, from the presenter.

### F9 — The digest card announces the calculation folder; the digests are written in each point's folder (medium; read both sides)

`lib/inspectors/transport.js:408-414`:
`/* THE DIGEST IS WRITTEN BESIDE THE DATA FILE: its folder announced */ … projects.publishFolderChanged(dirname(file))`.

Here `file` is the calculation's `task.json`, so the folder announced is the calculation root.
The data files, and the digests beside them, lie in each point's folder. The answer names those
folders: `molbuilder/web/blueprints/transport.py:534-544` `_readings_answer` sends
`"folder": str(e["folder"].relative_to(base))` for each point. Listeners compare the folder
exactly (`lib/projects/state.js:198-206`), so nothing listening to a point's folder hears.

**What goes wrong.** A sidebar open on `…/05_transmission/run-0/v0.2/` does not show the new
digest until it is refreshed by hand. The page works out the changed folder from a rule about
paths, when the server already knows it.

**Smallest fix.** The POST answer names the folders it wrote, as whole paths (results.md § 0.3),
and the page announces each one.

### F10 — Three inspectors each copy the read-only MolView mount and its failure check (medium; read all sides)

- `lib/inspectors/structure.js:144-181`: `mount(host, ws, {mode:"readonly", owner, ...molviewDoors})`,
  then `if (disposed) {dispose}`, then `if (!handle || !handle.ok) fail("Viewer failed: " …)`.
- `lib/inspectors/transport.js:1382-1406`: the same steps, worded differently.
- `lib/trajectory/core.js:536-561`: the same steps, but it keeps the failed handle in `_mv`.

The Transport page repeats it once more (`lib/transport/core.js:546-582`), and so does
`lib/structure-card.js` for the describe tabs. MolView's `mount` is the door; the guard around it
is copied each time.

**What goes wrong — this copy is the cause of round 1's D2.** Because the trajectory core keeps
the failed handle, it has two answers to *"is there a viewer?"*: `_mv` (truthy for a failed
mount) and `_mvdata()` (`core.js:535`, null for one). `core.js:1911-1912` asks `!_mv`, which is
how the TypeError happens.

**Smallest fix.** One helper for the inspectors that mounts MolView read-only, opens a file and
answers `{handle}` or `{error}`. Each inspector keeps only the handle that answered `ok`.

### F11 — The Transport page keeps two copies of one element builder; the safety check is in a third (medium-low; read all sides)

- `lib/transport/core.js:117-131` `_el(tag, attrs, ...kids)`.
- `lib/transport/fit.js:20-25` `el(tag, attrs, ...kids)`. `fit.js` is imported by
  `transport/core.js` on the same page.
- The same builder appears in `lib/form-schema.js:74-95` and `task-setup/viewer.js:143-153`.
  Only `form-schema.js` refuses `on*` / `innerHTML` / `srcdoc` attributes (*"a tripwire for
  future misuse"*).

`lib/dom.js` calls itself *"the one element builder"*, but offers only `el(tag, cls, text)`.

**What goes wrong.** A fix or a guard added to one copy is missing from the other three.
`fit.js` is shared by the Transport tab and Task setup (its header: *"One builder for both
pages"*), so a change made to `transport/core.js`'s copy never reaches it.

**Smallest fix.** Put the attributes-and-children form in `lib/dom.js`, with the tripwire, and
delete the copies.

### F12 — The trajectory core's state is read and written under two names (medium; read both sides)

`lib/trajectory/core.js:246-271`: *"Backward-compat aliases. Render code throughout this file
still reads/writes the legacy flat shape"*. Each of `state.path`, `mtime`, `format`, `label`,
`data`, `structure`, `notices`, `firstFit`, `pollTimer`, `pollInFlight`, `loadAbort`, `pollAbort`
is also `state.fileState.*` / `viewState.*` / `lifecycle.*`.

The two names are mixed inside one function. `pollOnce` reads `state.pollInFlight` (`:1408`),
sets `state.lifecycle.pollInFlight = true` (`:1413`), reads `state.fileState.path` (`:1419`) and
`state.mtime` (`:1421`), and clears `state.pollInFlight` (`:1464`).

**Owner.** The buckets, by the file's own contract (`:137-160`). The project's rule is
"rename = delete the old name everywhere".

**What goes wrong.** The aliases keep one value at run time, but every reader of the code, and
every test that builds a state by hand, must keep two names in step. One already does not (T5):
its fake holds `data` and `fileState.data` as different objects, a state the real code cannot
reach. `lib/inspectors/lifecycle.js:52-66` `alias` exists only to keep these names alive (the
spectra core uses it too).

**Smallest fix.** Rename the flat reads and writes to their bucket names, then delete the alias
wiring. Its test (T3) goes with it.

### F13 — The menu and the mount hand presenters two different shapes of the same file reading (medium-low; read both sides)

**At listing.** `lib/results/file-picker.js:110-124` builds `rec`: `name, path, mtime, size,
role, label, stage, parser, engine, engine_words, noun`. That is what `pickResult(fullPath, rec)`,
`resultCategory` and `absorbs` receive while the menu is built (`:134`, `:185-193`, `:247-256`).

**At mount.** `file-picker.js:348-354` `metaOf` copies a hand-kept subset (no `name, path,
mtime, size`), and that subset is what the viewer mounts with (`:983-985`; `results/viewer.js:159`,
`:691` `reg.pick(file, meta)`).

The viewer's own note says why this is a defect (`viewer.js:821-824`): *"A hand-kept field list
between two halves of one contract is a second shape of the same record"*.

**What goes wrong.** Suppose a field is added to `rec` and a presenter's `match` reads it. Then
the menu lists the file under that presenter, but the mount, given `metaOf`'s subset, picks
another presenter or none.

**Smallest fix.** One shape: pass `rec` itself as the mount's meta, or build `rec` from `metaOf`
plus the four listing fields, so the field list is written once.

### F14 — The controller keeps its own list of which viewers are slow (medium-low; read both sides)

`results/viewer.js:52` `ASYNC_VIEWER_INSPECTORS = { trajectory: 1, structure: 1, spectra: 1 }`,
read at `:702`, decides which viewer gets the loading cover. The registry's contract makes the
inspector record the place for such facts (`registry.js:13-64`: `isResult`, `resultCategory`,
`absorbs`). The controller's header promises *"Adding a new file type is one new inspector
module + a `<script>` tag… no edit to this file"* (`viewer.js:10-12`).

**What goes wrong.** The transport report also mounts MolView and fetches a record composed on
read. It is as slow, but it is not on the list, so it gets no cover. `results.md:832-833`
repeats the same three names.

**Smallest fix.** A field on the inspector record, for example `slow: true`, read by the
controller.

### F15 — Code nobody reaches, and options nobody passes (low; read the code and searched for callers)

- `lib/trajectory/core.js:1733-1740` `_scfFingerprint` reads `tail.etot`, then `tail.eharris`,
  then `tail.energy`. The SCF rows' one declaration names the column `energy` for both engines
  (`parse/engines/siesta_grammar.py:84-90`, `molwatch_grammar.py:120-124`; rows built at
  `parse/engines/_helpers.py:95`), so the first two branches never run.
- `core.js:2319-2323` handle `load(path)`, which the comment at `:88-93` says is *"used by the
  registry-side dispatch"*: nothing calls it. `_partial_inspector_factory.js:175-187` returns
  only `dispose`.
- `core.js:512-513` `state._plottableFrames = plottableFrames; // Expose for tests`: no reader.
  The test lifts the function instead (T3).
- `_partial_inspector_factory.js:101,106`: the options `errorDisplayName` and `isResult` are
  passed by neither caller (`inspectors/trajectory.js`, `inspectors/spectra.js`).
- `registry.js:298-303,339` `_clear`, *"Test-only escape hatch"*: no test calls it.
- `bench-summary.js:240-241`: `automargin: true` is written twice in one object.

**Fix.** Delete each.

### F16 — Each inspector formats numbers with its own helper, and one figure already reads differently on one tab (low; read all sides)

**The helpers.** There are four "a number, or a dash" helpers:
- `lib/inspectors/transport.js:62-67` `_fmt(x, digits, exp)`;
- `fc-sweep.js:26-30` `_num(x, digits)`;
- `bench-summary.js:34-43` `_sPerIter` and `_num(v, unit, digits)`;
- `lib/results/run-panel.js:37-59`.

**The figure that disagrees.** `s_per_iter` comes from one server number. The record's comment
says it exists *"so the page states one number"* (`parse/dirs/record.py:323-327`). It is
formatted:
- in the Run panel by `run-panel.js:45-49` (`n >= 100 ? Math.round(n) : +n.toPrecision(3)`);
- in the SCF line, on the same tab, by `trajectory/core.js:1371` (`+perIter.toPrecision(3)`);
- on the bench card as `toFixed(2)` (`bench-summary.js:34-36`).

At 1234.5 s per iteration the Run panel says *"1235 s"* and the SCF line *"1230 s/iter"*.

**Smallest fix.** One formatter for a figure in seconds (beside `timeSpan` in
`lib/time-ago.js`), and one dash helper in `lib/dom.js`.

### F17 — A run's state has two word-and-tone tables, and a lone file's ending is judged again in the page (low; read both sides)

**Two tables.** `lib/state-chip.js:24-39` `TONE` is *"the ONE state chip… so a word reads the
same everywhere"*. The trajectory badge maps the same status words again: `core.js:1872-1887`
`_badgeKind` and `:1799-1863`, with its own tone classes. The badge's words are contracted
(`trajectory.md` § 4), so this is not about wording. The problem is that any state `_badgeKind`
does not list falls through to *Running* in the busy tone, while the chip shows it as itself.

**The same rule twice.** For a file that belongs to no run, the page judges its ending itself:
- `RUN_STATE` (`core.js:55-61`) copies the parser's words;
- `_fileStopped` (`:483-486`) and `_badgeKind`'s second half (`:1880-1886`) decide that
  `stopped` or `out_of_memory` means failed.

`parse/dirs/job.py:664-672` decides the same thing for a run's file.

**What goes wrong.** A new status word shows as *Running* on the badge and as itself on the
ladder and the Run panel. A new ending word must be added both in Python and in JavaScript.

**Smallest fix.**
- The badge's word for each state becomes a column beside `TONE` in `state-chip.js`.
- The server answers `run` for a lone file too, by the status door's rule, so the page has one
  path.

### F18 — A transport point and a voltage are each worded more than one way (low; read both sides)

**The point.** The digest answer carries the server's words for each point
(`web/blueprints/transport.py:540`, `"words": e["point"].words()`, from
`transport/stages.py:206-213`: `frame 2 of 3 · 0.2 V`, the voltage written with Python's `:g`).
The page throws them away (`lib/inspectors/transport.js:337-341`) and composes its own with
`point-words.js:26-33` (`String(Number(biasV)) + " V"`). The capital "Frame" is contracted, but
the number format differs: 1e-05 reads `1e-05 V` in a server sentence (`detail`, `why`, `said`)
and `0.00001 V` in the table beside it.

**The voltage.** `lib/transport/core.js` writes a voltage as
`v === 0 ? "0.0" : String(v)` three times (`:971-973`, `:1198-1200`, `:1923-1925`).
`point-words.js` writes 0 V as `"0 V"`.

**Smallest fix.**
- One voltage formatter in `point-words.js`, used by `transport/core.js`.
- The voltage's number format is taken from the Python one, by serving it or by writing the same
  rule, so the two sources agree.

### F19 — The transport report chooses again which Fermi level is a rung's (low; read both sides)

`lib/inspectors/transport.js:770-773`:
`/* A DEVICE'S E_F IS ITS NEGF PHASE'S */ const ef = st.negf && st.negf.ef !== undefined ? st.negf.ef : st.fermi_ev;`

The record already makes this choice: `molbuilder/transport/record.py:739-749` sets `fermi_ev`
to *"the device's its NEGF loop's (the last row is NEGF)"*.

The two agree today. But it is a second rule, and if the record ever changes `fermi_ev` (say, a
per-frame value), the page keeps its own reading.

**Fix.** Show `st.fermi_ev`.

### F20 — The transport report re-reads a few of the record's fields, but not all (low; plausible)

`render` (`lib/inspectors/transport.js:168-183`) copies `points`, `polarized`, `iv`, `average`
and `nFrames` into `state`, and builds the I-V table and its pending rows from them
(`_fillIV`, `:524-640`). `state.reload` (`:1493-1504`), run after a digest read, copies only
`points` and `polarized`.

**What may go wrong.** A point that finished between the first load and the digest read would
appear in the charts while the table still lists it as *not ended*, and the header counts stay
as they were. Read and traced in the code; that a point finishes in that window is plausible.

**Smallest fix.** The reload goes through the same step as the first load. That one step copies
the record into `state`, and the reload then redraws the parts that depend on it.

### F21 — The poll cadence is a constant in two files, each pinned by its own test (low; read both sides)

`lib/trajectory/core.js:103` `POLL_MS = 15000`, and `lib/inspectors/bench-summary.js:25`
`POLL_MS = 15000` with the comment *"B4: the trajectory viewer's cadence"*. Each is pinned
separately: `test_trajectory_transition_js.py:284` and
`test_bench_summary_inspector_e2e.py:202,211`.

**What goes wrong.** Changing the cadence in one file leaves the other, which says it follows
it, behind.

**Fix.** One constant in `lib/constants.js`, read by both.

### F22 — Small copied blocks within one file (low; read)

- `lib/results/file-picker.js:374-384` `_populatePlaceholder` and `:389-397` `_prependUnchosen`
  build the same disabled, selected, empty-valued option.
- `lib/trajectory/core.js:292-328` (`LOADING`) and `:330-353` (`IDLE`) repeat the same abort,
  stop and clear block of about 20 lines.

**Fix.** One helper in each file.

### F23 — The transport report words one state by itself (low; read both sides)

`lib/inspectors/transport.js:756-757` branches on `st.state === "not_described"` and types
*"not in this description"*. The record (`transport/record.py:428`) sends the code without
words. Every other state goes through the state chip.

**Fix.** The record sends that row's `detail`, and the page shows it as it shows every other row.

---

## Test findings

### T1 — The live-server fixture is copied into each test file (medium; searched)

The same four lines
(`@pytest.fixture(scope="module") def flask_server(isolated_projects_root_module): from support.live_server import serve; with serve() as base_url: yield base_url`)
appear in 18 test files. In this slice:
- `test_inspector_registry_e2e.py:28-32`;
- `test_results_file_picker_e2e.py:28-32`;
- `test_coming_back_to_the_tab_e2e.py:36-40`;
- `test_results_export_e2e.py:50-54`;
- function-scoped variants: `test_keyboard_paths_e2e.py:93-97` and
  `test_bench_summary_inspector_e2e.py:51-57` (`server`).

**The rule.** The project's own rule is to build the one shared fixture before copying one into
each file.

**Fix.** One fixture in `tests/conftest.py`, or in `support.live_server`, for each scope.

### T2 — Two tests type a structure file by hand instead of saving it the way the app does (medium-low; read both sides)

- `test_results_file_picker_e2e.py:46-47`:
  `(proj / "w.xyz").write_text("3\nwater\nO 0.0 0.0 0.119\n…")`.
- `test_inspector_registry_e2e.py:355-358`: `water.xyz` and `co2.xyz`, typed the same way.

These are bare `.xyz` files with no sidecar, which nothing in molbuilder writes. The shared door
is `support/road.py:205-230` `structure_saved`, which writes the pair through `StructureCodec`.
`testing.md` § 3b says *"the road's own writers for its input"*.

**Fix.** Save each structure with `structure_saved(tree, …, at=…)`.

### T3 — Tests cut functions out of the source by comment lines, and the source keeps a hook just for tests (medium-low; read both sides)

`test_trajectory_transition_js.py` lifts code out of the shipped file by text markers:
- `:135-136` ends a cut at `"\n    // Expose for tests"`;
- `:184-187` and `:254-255` end cuts at `"\n    // Transition orchestrator"`.

So rewording a comment in `lib/trajectory/core.js` breaks these tests.

The production file keeps a hook for the tests, `state._plottableFrames` (`core.js:512-513`),
which no test reads (F15). The test of the alias wiring (`:209-219`) pins the
backward-compatibility layer F12 would remove.

`testing.md` § 3a: *"No other test reads a file for what is in it."* The `lifted` helper is not
on § 3a's short list of exceptions.

**Fix.**
- Cut at function boundaries (the next `function`), never at a comment.
- Delete the hook.
- Retire the alias test together with the aliases.

### T4 — One bench-card test repeats the other and pins a name that was removed (low; read)

`test_bench_card_reads_the_metrics_python_writes.py:88-102`:
- its first half checks that `monitored_elapsed_s` reaches the card, which test 1 (`:66-85`,
  where `expected_labels` includes it) already checks;
- its second half asserts that the removed name `wall_s` renders nothing — a test that pins a
  removed name, which the project's notes rule out.

Also, `expected_labels` (`:74-80`) is a hand-typed copy of the card's words. Its label `"s"` for
the duration matches nearly any line.

**Fix.**
- Retire test 2.
- In test 1, compare against the card's line for each key alone, so each key is checked on its
  own.

### T5 — A hand-built trajectory state holds two different values for one field (medium-low; read)

`test_the_viewers_follow_a_real_run_e2e.py:412-421`, in `_poll_once`'s fake state:
`data: { frames: [] }` beside `fileState: { …, data: { run_state: "running", frames: [] } }`, and
`lifecycle.pollInFlight` with no flat `pollInFlight`.

Because `pollOnce` reads both spellings (F12), the fake has to supply both, and it supplies two
different objects. That is a state the shipped core cannot reach. The guard at `core.js:1408`
reads the flat `pollInFlight`, which here is undefined, and passes by accident.

`testing.md` § 3b, *A fake… is built through the real entry*: the module's own state, with its
aliases wired, is already lifted the right way in `test_trajectory_transition_js.py:252-256`.

**Fix.** Build the fake from the lifted `state` and its alias wiring, or remove the aliases (F12)
so there is only one shape to build.

### T6 — The picker's unit test uses stand-in viewers that match by suffix, and checks a wrong input (low; read)

`test_results_file_picker_js.py:48-54, 75-81` stand in for presenters with
`if (p.endsWith('.out')) return {…}`. Every real presenter refuses to match by suffix
(`inspectors/trajectory.js:50-62`, `inspectors/spectra.js:46-49`). The real `registry.js` could
be loaded and real presenters registered instead.

`:103-117` checks a `resultCategory` that throws — a wrong input listed as a case, which
`testing.md` § 3b (*A test validates what is correct*) rules out.

`:131-139` pins `"5m ago"`, which is `lib/time-ago.js`'s answer, not the picker's.

**Fix.**
- Load `registry.js` and register stand-ins through `register`.
- Drop the throwing case.
- Check the label by its file name alone.

### T7 — The describe-request test types its own copy of the Transport page's door record (low; read both sides; the module tested is `task-handover.js`, another slice's)

`test_transport_describe_request_js.py:51-58` types
`{"url": "/api/transport/describe", "doing": "Describing…", "checked": …, "next": "the road"}`
as *"the Transport tab's door as `transport/core.js` names it"*. That is a copy of
`lib/transport/core.js:1515-1521`, which already differs in `next`. If the page's door changes
its URL, this test still passes on its own copy.

**Fix.** Take the record from the page module, by exporting it from `transport/core.js`, or name
only the URL, which the route owns.

---

## The five that matter most

1. **F1** — The "load has ended" signal is written by hand in seven places beside its one door
   (`lifecycle.announceReady`). The copies already disagree on timing and on what they carry, and
   five repeat the event's name as a literal.
2. **F2** — The memory peak is in GiB, but the page says "GB" in three places and "GiB" in one,
   both on the same bench card. The owner of the word, `units.size_words`, is not used.
3. **F3** — The bench page builds a node census in the browser, which the server's own comment
   and bench-summary.md B1/B2 forbid. It also keeps hand-kept lists of which fields repeat.
4. **F10** — Three inspectors each copy the read-only MolView mount and its failure check. The
   trajectory core's copy keeps the failed handle, which is the cause of round 1's D2 TypeError.
5. **F9** — The digest card announces the calculation's folder, worked out from a rule about
   paths, while the digests are written in each point's folder and the server's answer names
   those folders.
