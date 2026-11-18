# F12 review, round 2 — R2-13: the vibration and spectrum viewers

Checkout `the checkout` at 2.77.2. Read only; nothing was run except grep, sed and file reads.
Paths below are relative to `molbuilder/web/static/lib/` unless they say otherwise.

## What was read

**Code, every file in full:** `vibrationview/` (index.js, _maths.js, _seal.js, _export.js, _style.css),
`spectra/core.js` (all 4762 lines), `spectrumchart/` (index.js, _maths.js, _seal.js, _style.css),
`scfplot/scfplot.js`, and `frameset/card.js` with `card.css`.

**Followed to the other side of a finding:** `dom.js`, `status.js`, `plot-box.js`, `plot-theme.js`,
`state-chip.js`, `inspectors/lifecycle.js`, `zip-store.js` and `media-export.js` (headers only),
`templates/_spectra_inspector.html` (controls), `web/blueprints/spectra.py` (`_page_constants`, `_err_load`),
`spectra/infrared.py` (`DIRECTIONS`, `read_ask`, `no_charges_reason`), `spectra/raman.py` (`read_ask`),
`spectra/methods.py` (`ir_route_sentence`, `infrared_view_sentence`, `raman_view_sentence`,
`CHARGE_PARTITION_WORDS`), `sidecars/spectra.py` (the error classes), `frame_average.py` (`answer_said`,
`check_units`).

**Rules read:** `docs/architecture.md` (§ 3 constants and served values, § 4), `docs/web/overview.md`
(§ 1 and § 5: "a cancel is a mark, never a sentence"), `docs/web/runtime.md` § 3, `docs/web/ui-contract.md`
§ 12.1 (server words go through one renderer), `docs/web/results.md` § 1.2 and § 2.5 (one plot box, one
theme), and `docs/web/vibrationview.md` § 4 (the seal). Round 1's `r1-L` and `v-L` were also read, and
none of their findings is repeated here except where a duplicate is the cause (F10).

**Tests, read in full:** `test_vibrationview_maths_js.py`, `test_vibrationview_mount_js.py`,
`test_vibrationview_e2e.py`, `spectra/test_spectrumchart_maths.py`, `spectra/test_spectrumchart_seal.py`,
`spectra/test_spectrumchart_handle.py`, `spectra/test_spectrumchart_box_e2e.py`,
`test_spectra_transition_js.py`, `test_spectra_no_spectrum_sentence_e2e.py`.

**Tests, read in part** (only the sections that drive this slice):
- `test_inspector_registry_e2e.py`: lines 1–130 and 300–589
- `test_the_viewers_follow_a_real_run_e2e.py`: lines 655–958, the spectra section
- `test_spectrum_results_e2e.py`: lines 456–600
- `test_siesta_vibration_e2e.py`: lines 1050–1310
- `test_keyboard_paths_e2e.py` and `test_siesta_run_in_the_viewer_e2e.py`: grep only

**Not read:** the rest of those large e2e files (API rows on the server side).

**Coverage seen, not a finding:** no test reaches `frameset/card.js`'s output. No test names its classes
or its sentences.

**Totals: 29 findings (21 code, 8 test).** Each finding says how sure I am. "Read both sides" means both
places were read.

---

## Code

### F1 — The server's sentences reach the page through a second renderer (medium; read both sides)

- **The owner.** `ui-contract.md` § 12.1 says every place a page shows a sentence of the server's draws it
  through `codeWords` (`dom.js:52-59`, reached through `status.set`, `status.js:73-76`). That renderer turns
  a backticked command into code and the terminal's ` -- ` into a dash.
- **The second renderer: plain `textContent`.**
  - `spectra/core.js:1919-1922`: `const absent = (results && results.infrared_absent) || ""; … els.irAbsent.textContent = absent;`
  - `core.js:2156`: `if (dd) dd.textContent = words(results);` — run-summary lines that embed
    `r.infrared_said` (`core.js:1842-1843`: `"from the run's charges — not given: " + r.infrared_said`).
  - `core.js:2295`: `dd.textContent = String(v);`
  - `core.js:4261-4262`: `if (th.note) bits.push(String(th.note) + "."); els.thermoNote.textContent = …`
  - `core.js:4384-4385`: hands `"not given: " + r.raman_said` to the chart lane as plain text.
  - `frameset/card.js:63-64`: `"The cited frames could not be read: " + frameSet.why`, and `card.js:110`:
    `"Warning: " + w`, both through `_el` with `textContent`.
- **The words carry the dash.** `spectra/infrared.py:242` returns
  `"the run recorded none -- its log says why"`, and `infrared_absent` wraps it.
  `frame_average.py:612-616` writes `why` with ` -- `.
- **What goes wrong.** The same `infrared_said` is drawn twice on one screen: through `status.set` in the
  bar (`core.js:1993-1995`, a dash), and raw in the run summary (`--`). `#ir-absent` and the frame-set card
  show a raw `--`. A backticked command would show its backticks.
- **Smallest fix.** Draw these with `molbuilder.dom.wordsEl` or `codeWords`. Hand the Plotly lane
  `dom.wordsText(words)`, the text `codeWords` draws.

### F2 — The infrared/Raman view and route words have two homes: the page and `spectra/methods.py` (medium; read both sides)

- **The server's home.** `spectra/methods.py:140-166` (`ir_route_sentence`) says in its docstring:
  *"One home for the wording … so the prose a reader copies into a paper and the provenance the viewer shows
  can never drift into two different claims."*
- **The page's own words.**
  - Route words: `core.js:1856-1868` (`case "analytic": return "computed — analytic ∂μ/∂R (no extra SCFs)"`,
    `"finite-difference"`, `"none"`) and `core.js:1892-1909` (the Raman routes).
  - View words: `core.js:1716-1742` (`_partitionName`, `_directionWords`, `_irViewWords`) and the notes at
    `core.js:1762-1814`.
  - Matching server code: `methods.py:194-198` (`_direction_words`, `f"{float(x):g}"`), `methods.py:209`
    (`str(view["partition"]).capitalize()`) and `methods.py:311` (`CHARGE_PARTITION_WORDS`).
- **A concrete drift.** `core.js:1723` formats a vector with `x.toFixed(3)`; the server uses `:g`. Light
  along (1,1,0) reads "polarized along (0.707, 0.707, 0)" in the table note and the CSV header
  (`along_0.707_0.707_0`), but "(0.707107, 0.707107, 0)" in the Methods paragraph on the same page.
  A partition is capitalised by the JS, by the server, and again by the template's option labels
  (`_spectra_inspector.html:123-124`).
- **Smallest fix.** The load door already serves `broaden_words` with each Raman view (`core.js:1620`).
  Serve `direction_words`, `partition_words` and each route's summary words the same way, then delete
  `_partitionName`, `_directionWords` and the route switches.

### F3 — The ask vocabularies and their defaults are kept by hand in the template and in core.js (medium; read both sides)

- **The owner.**
  - `spectra/infrared.py:53-60`: `DIRECTIONS = {"x":…, "y":…, "z":…}`, `DEFAULT_DIRECTION = "z"`,
    `DEFAULT_PARTITION = "hirshfeld"`
  - `spectra/results.py:263`: `PARTITIONS = ("hirshfeld", "voronoi")`
- **The copies.**
  - Template: `_spectra_inspector.html:107-111` (`<option value="z" selected>`) and `:122-125`
    (`hirshfeld`/`voronoi`).
  - `core.js:216-218`: `irDirection: "z"`, `irVector: "0 0 1"`, `irPartition: "hirshfeld"`.
  - `core.js:410-413` (`_prefAdmissible`) treats the template's `<option>` list as the vocabulary.
- **Other defaults are kept twice the same way.** `uiPrefs` at `core.js:199-211` (`broadeningFWHM: 20`,
  `animAmplitude: 0.15`, `animSpeed: 1.0`, `animTemperature: 298`) against the template's `value=`
  attributes (`:161-162`, `:341-342`, `:359-372`). `init()` overwrites four of the five knobs from the
  markup (`core.js:4550-4576`).
- **A comment that no longer holds.** `core.js:482-485` says the snapshot `PREFS_DEFAULTS` means *"the two
  cannot disagree"*, but the markup is a third source it never sees.
- **What drifts.**
  - Mark another `<option>` selected in the template: the bar shows it while the applied ask stays at `z`.
    `_barChanged` (`core.js:2043-2048`) then reports *"changed — Update to show it"* every time a result
    opens.
  - Add a partition on the server: the page refuses to restore it until the template is edited too.
  - The export fallbacks are kept three times: `core.js:3787-3791` (`1600`, `1200`, `"png-zip"`, `1`), the
    template's `value=`s (`:395-421`), and the module's own (`vibrationview/_export.js:357, 364`).
- **The pattern exists.** The page already gets one vocabulary served: `frameset_nodes` comes from
  `_page_constants` (`blueprints/spectra.py:87-89`) and is consumed at `core.js:3626`.
- **Smallest fix.** Serve `DIRECTIONS`, `PARTITIONS` and the two defaults the same way, and build the two
  selects from them. Give each knob's default one home: the `uiPrefs` literal, with `init()` writing it
  into the controls rather than reading the markup.

### F4 — The load door's ask rules are restated in the page, five times, with their own words (medium-low; read both sides)

- **The owner.** `infrared.py` (`_unit`, `read_ask`), `raman.py:495-516` (`read_ask`, `_positive`), and
  `spectra.derived.checked_temperature_k` (`blueprints/spectra.py:93-104`).
- **The page's copies.**
  - `core.js:421-437` (`_positiveOf`, `_vectorOf`, each "the door's rule").
  - `core.js:2012-2039` (`_barChoices`, with its own refusal sentences).
  - The temperature rule four times: `core.js:414` (`v >= 0`), `:4002-4004`, `:3680-3682`
    (`"the temperature is a number of kelvin, 0 or more"`), `:3694`.
- **What drifts.** `_barChoices` says *"a broadening is a positive number of eV, the run's own or more"*,
  but checks only "positive". The "or more" rule is the server's (`raman.py` `Var broaden_ev`: "never below
  it"). A width below the run's own passes the page and the server refuses it in other words.
  `onViewUpdate` already shows the door's own refusal (`core.js:2082-2086`: `_barError = … a.error`).
- **Smallest fix.** Keep one page-side parse per kind, `_positiveOf` and `_kelvinOf`, used everywhere. Let
  the door's refusal be the only refusal sentence, and drop the page's restated rule words.

### F5 — Two spellings of every state key, five dead aliases, and a test that keeps them alive (medium; read both sides)

- **The block.** `core.js:439-480` (`_wireBackcompatAliases`, "Backward-compat aliases. ~3000 lines … read
  the legacy flat shape") defines `state.results`, `state.selectedMode` and the knobs as views of their
  buckets.
- **Both spellings are in use.** By grep: `state.results` 36 times beside `state.fileState.results` 11
  times; `state.selectedMode` 16 times beside `state.viewState.selectedMode` twice.
- **Five aliases nothing reads.** `alias("watchTimer"|"watchInFlight"|"watchAbort"|"loadAbort"|"watchErrors", "lifecycle")`
  (`core.js:475-479`) — the flat names occur 0 times in core.js. Only
  `tests/test_spectra_transition_js.py:186-194` (`_NOT_KNOBS`) and its `TestBackcompatAliases` keep them.
- **What drifts.** The contract says `transition("APPLY")` is the single writer of `fileState`
  (`core.js:155-159`), but the alias setter is a second door. Any `state.results = …` bypasses the path
  guard.
- **Smallest fix.** Rename the readers to the bucket spelling, delete the alias block except the knob
  setters that schedule the save (or make the save a `uiPrefs` setter), and retire the five lifecycle rows
  of the test.

### F6 — The modes table's columns are described four times, and two copies have drifted (medium-low; read both sides)

- **The four copies.**
  - The cells: `_renderModeRow`, `core.js:2635-2687`
  - The filter strings: `_modeMatchesFilter`, `core.js:2560-2584` (*"Match against the same fields the
    table shows"*)
  - The sort keys: `_modeKey`, `core.js:2586-2602`
  - The CSV columns: `exportCSV`, `core.js:2786-2853`
- **Drift today.**
  - The filter leaves out the Δgap column.
  - A picked mode whose side is pending shows `notGiven.state` (`core.js:2641-2645`), but the filter
    matches `_ramanOf` (null). Typing "pending" finds nothing while the column shows it.
  - The filter re-derives HOMO, LUMO and gap inline (`core.js:2576-2581`) instead of calling `_homoEq`,
    `_lumoEq` and `_gapEq` (`core.js:2604-2619`).
  - `renderESPanel` recomputes the gap and Δgap (`core.js:2990-2994`) with the arithmetic `_dgapMax`
    already holds (`core.js:2620-2633`).
- **Smallest fix.** One column table: key, header, value, cell text, CSV name. Rows, filter, sort and CSV
  all read it; the ES panel calls the existing helpers.

### F7 — A cancel is recognised by its words (medium-low; read both sides)

- **The code.** `core.js:3807-3809`:
  `setStatus(els.animExportStatus, msg, msg.indexOf("cancel") >= 0 ? "muted" : "error");`
- **The rule.** `web/overview.md` § 5: *"A cancel is a mark, never a sentence … Neither is
  string-matched."* The page holds the mark itself: `controller` (`core.js:3776`).
- **What drifts.** Reword VibrationView's `"the export was cancelled"` (`_export.js:201, 289`) and a cancel
  is painted red as an error. Any failure whose text contains "cancel" is shown muted. The text is also
  pinned by `test_vibrationview_mount_js.py:796` and `test_vibrationview_e2e.py:301`.
- **Smallest fix.** `controller.signal.aborted ? "muted" : "error"`.

### F8 — The run's state is judged by literal words beside the one state door (medium-low; read both sides)

- **The page's branches.** `core.js:1255-1263` turns red only for `run.state === "unreadable"` and
  `=== "failed"`.
- **The owner.** `state-chip.js:24-37`, `TONE` — "every page that shows where a stage or a run stands …
  draws it here, so a word reads the same everywhere". It also marks `missing` as bad and `not run` as warn.
- **What drifts.** A run the door calls `missing`, `not run` or a future state falls through to the
  `allPhasesComplete` or progress branch. The status line then says "Computing … Hessian." for a run that
  is not running.
- **Smallest fix.** Ask `molbuilder.stateTone(run.state)` for the severity, and say `run.state` and
  `run.detail` as given whenever the tone is bad.

### F9 — The page rewrites the load door's refusal sentences (low-medium; read both sides)

- **The page.** `core.js:1197-1207` composes its own sentences: *"Schema version mismatch (expected …).
  Update molbuilder or use a matching script version."* and *"File not found at … -- it was removed after
  the folder was listed."*
- **The server.** `sidecars/spectra.py:83-88` and `:250-254` already write those refusals, for the
  terminal too.
- **What drifts.** The two give different advice: the server says "a different molbuilder version, or it
  isn't a Spectra-tab result file". The blueprint docstring invites the rewrite
  (`blueprints/spectra.py:430-435`), against `ui-contract.md` § 12.1.
- **Smallest fix.** Show `a.error`. Keep `kind` for reacting, not for rewording.

### F10 — The cause of round 1's D1: which modes are in lane 0's stick trace is known in two places (medium-low; read both sides)

- **Place one.** `spectrumchart/_seal.js:350-367` drops every pending mode from lane 0's bar trace
  (`if (lane.sticks.state[i] === "pending") { pending.push(x); return; }`).
- **Place two.** `spectrumchart/index.js:334-345` builds the colours one per mode
  (`modes.map((m) => stateOf(…))`), and `_seal.js:556-562` restyles that trace by position.
- **Why it matters.** The index.js note at `:254` names this hazard for the floor, and guards the floor
  only.
- **Smallest fix.** One owner of the membership: either keep pending modes in the bar trace (at zero
  height, as the floor does), or let `recolour` in the seal apply the same filter `tracesFor` applies.

### F11 — VibrationView: the failed-mount class is spelled in index.js, declared but unused in _seal.js, and styled by a sheet only _seal.js links, after the check that fails (medium-low; read both sides)

- **The two spellings.** `vibrationview/index.js:66`: `hostEl.classList.add("vibview-window", "vibview-window--failed")`,
  and `:354` removes `"vibview-window"`. `_seal.js:47-48` declares `ROOT_CLASS` and `FAILED_CLASS`;
  `FAILED_CLASS` is used nowhere.
- **The ordering problem.** `_seal.js:162-170` throws *"the drawing library must be loaded first"* before
  `ensureStylesheet()`. On the commonest failure the sheet that styles `.vibview-window--failed`
  (`_style.css:138-147`) is never linked, so the failed box shows unstyled text.
- **Smallest fix.** One home for the class names and the sheet. For example, `_seal.js` exports a
  `fail(hostEl, message)` that links the sheet and applies its own `FAILED_CLASS`. This stays within the
  seal.

### F12 — VibrationView: the seal's "fallbacks" restate the sheet's values (low; read both sides)

- **The copies.** `_seal.js:52-53` (`#0f1217`, `#555555`) and the compositor's `rgba(15,18,23,0.62)`,
  `#e6e9ef`, `12`, `6`, `3`, `8`, `8` (`_seal.js:328-335, 367, 374`) are the token values of
  `_style.css:31-35` and `:113-125`.
- **The claims they contradict.** The file says *"There are no colours or sizes in this file"*
  (`_seal.js:38`) and that they *"are not a second palette"*.
- **What drifts.** Change a token, and a page whose sheet failed (`ensureStylesheet` resolves on error)
  draws the old palette. The unit test supplies its own `getComputedStyle`
  (`test_vibrationview_mount_js.py:105-115`), so it never uses these values.
- **Smallest fix.** Fall back to "nothing": no caption box, the library's default ground.

### F13 — Code outside VibrationView knows its internals, and the speed slider's top is derived by hand from them (low-medium; read both sides)

- **The slider.** `_spectra_inspector.html:365-371`: `max="2.0"` *"(`vibrationview/_maths.js`
  FRAMES_PER_CYCLE_MIN)"*. That is 30 fps (`vibrationview/index.js:37` `DEFAULTS.fps`; core.js never
  passes `fps`) divided by 15 (`_maths.js:38`).
- **Outside code naming a sealed internal.**
  - `test_inspector_registry_e2e.py:418-421` holds `**/static/lib/vibrationview/_style.css`
    (`_seal.js` `ensureStylesheet`).
  - `:413` waits on `.vibview-caption`.
  - `vibrationview.md` § 4: *"nothing outside the package may name any path inside it but index.js"*.
- **What drifts.** Change the frame floor or the default rate, and the slider offers speeds that all play
  alike, or caps a speed it could play.
- **Smallest fix.** Have the page pass `fps` at mount from its own constant, and state the slider's top
  from that constant and the contract's 15. Drop the internal file name from the template comment. In the
  test, wait on the viewer's status line instead of the module's caption class.

### F14 — SpectrumChart: dead code, and a default height reached only by tests in a shape the handle never produces (low; read both sides)

- **Dead in the seal.** `_seal.js:330` `const sign = 1;`, multiplied in at `:338` and `:355`.
  `_seal.js:302-306` sets `.matches = undefined` in a loop.
- **Dead in the handle.** `index.js:335`'s `channels.length ? … : "activity"` (channels is never empty,
  `:131-134`).
- **One default, four times.** The class default `"partial"` is restated at `index.js:206` (after `:452`
  already normalised it), at `spectra/core.js:4189`, and by `_seal.js:101-107`.
- **The test-only default.** `_maths.js:52` `const defaultHeight = (m) => m.activity` (used `:54`, `:83`)
  is never reached by the handle, which always passes `heightOf` (`index.js:306-308`). The maths tests keep
  it alive: `test_spectrumchart_maths.py:48-52` builds modes with a top-level `activity`, a shape the
  handle never passes. `test_spectrumchart_handle.py:317-318` uses that old shape too.
- **Smallest fix.** Delete `sign`, the `matches` loop, the fallback key and the second `cls` default. Make
  `heightOf` required, and give the maths tests `heightOf`.

### F15 — SpectrumChart's seal types its own served path and the vendor route (low; read both sides)

- **The literals.** `_seal.js:17-18`: `PLOTLY_URL = "/vendor/plotly.min.js"` and
  `STYLESHEET_URL = "/static/lib/spectrumchart/_style.css"`.
- **What derives them instead.** The route is `web/app.py:764` (`url_for('vendor_plotly_js')` in
  `results.html:39`). The module's own place is what `import.meta.url` gives, as VibrationView does
  (`vibrationview/_seal.js:77`).
- **Smallest fix.** `new URL("./_style.css", import.meta.url)`. For the library, there is no fix beyond
  noting the route.

### F16 — Element builders re-implemented beside the one builder (low; read both sides)

- **The owner.** `dom.js:25-30` `el(tag, cls, text)`, "the one element builder", loaded on every page by
  the shared header.
- **The copies.** `frameset/card.js:41-46` `_el(doc, tag, cls, text)` and `spectra/core.js:3023-3028`
  `node(tag, cls, text)`. Hand `createElement` sequences also sit at `core.js:2290-2296` and
  `:2658-2663`.
- **Smallest fix.** Call `molbuilder.dom.el` (and `wordsEl` for the server's words, F1).

### F17 — The level diagram and the thermochemistry re-key the chart theme and restate its values (low; read both sides)

- **The re-keyed theme.** `core.js:2873-2880` (`_esTheme`) renames `plotTheme()`'s fields (`muted`→`dim`,
  `palette[0]`→`homo`, `palette`→`series`).
- **The restated values.** The layouts then repeat what `themedLayout` (`plot-theme.js:46-60`) already
  sets: `core.js:3177-3188` (`color: th.ink`, `gridcolor: th.grid`, `font: {color: th.ink, size: 10}`),
  `:4291-4300` and `:4328-4335`.
- **What drifts.** `themedLayout` lets the chart's own keys win, so these copies, not the theme, decide.
  Change the theme's font size or grid token, and these charts keep the old values.
- **Smallest fix.** Read `plotTheme()` directly and drop the keys `themedLayout` already sets.

### F18 — Dead "chart library unavailable" branches, with two different sentences for the one state (low; read both sides)

- **The branches.** `core.js:2914`, `:2963-2965` (`'<p class="status muted">chart library unavailable on this page</p>'`
  via innerHTML), `:4264-4269` (`"(chart library unavailable on this page)"`) and `:4705`.
- **Why they are dead.** The partial mounts only on `/results`, which loads Plotly (`results.html:39`).
  The comment at `core.js:4211-4213` says so. `plotBox.draw` assumes Plotly anyway.
- **Smallest fix.** Delete them.

### F19 — `transition()`'s "let the requests go" sequence is written four times (low; read)

- **The copies.** `core.js:511-524` (LOADING) and `:540-553` (IDLE) are line-for-line the same. LOADED
  (`:579-583`) and ERROR (`:619-623`) repeat the watch half.
- **What drifts.** Change one: a new controller is aborted on file switch but not on dispose.
- **Smallest fix.** One `_letGo()` helper called by each target.

### F20 — The same accessor twice, and one list keyed by position (low; read)

- **The accessors.** `_irModeOf` (`core.js:1533-1537`) and `_ramanModeOf` (`:1588-1592`) are one function
  over `r.infrared` and `r.raman`.
- **The position key.** `_refreshThermal` (`core.js:4023-4027`) copies spreads by array position
  (`held.modes[i]`). Every other join matches `index_1based` (`:1536`, `:3315`, `:3631`).
- **Smallest fix.** One `_viewModeOf(r, key, m)`, and join the thermal answer by `index_1based`.

### F21 — scfplot branches on one phase name and exports a helper no one imports (low; read)

- **The branch.** `scfplot.js:60-62`: `return ph === "negf" ? "NEGF" : (ph ? ph : "SCF");`. The phase
  vocabulary is the grammar's (`parse/engines/siesta_grammar.py:122`, `PHASE_NEGF`), and the header says
  "This module keeps no list".
- **The unused export.** `phasesOf` (`:199-200`) is used only inside the file; `residualsOf` is used by
  `trajectory/core.js:1306`.
- **Smallest fix.** Serve the phase's display word with the rows, like `scf_residuals`, or show the phase
  as written. Drop `phasesOf` from the export.

---

## Tests

### T1 — The spectra viewer's harness is copied into two files, with the state shape typed by hand (medium; read both sides)

- **The copies.** `test_spectra_transition_js.py:62-96` (`_viewer`) and
  `test_the_viewers_follow_a_real_run_e2e.py:677-739` (`_spectra_viewer`) build the same prelude:
  - `_aborted`, `_ac` and the `setInterval`/`clearInterval` stubs
  - `_listeners`, `document`, `chart`/`chartReady`, `els`, `root`
  - a hand-typed `state` with the same five buckets
  - the same `lifted(... transition ...)` and `lifted(... _abortViews ...)` with hand-closed braces
- **Already drifted.** One types `const WATCH_INTERVAL_MS = 2000;` (`transition_js:68`); the other lifts it
  from the source (`follow_e2e:683`).
- **What drifts next.** A bucket or field added to core.js's `state` (`core.js:160-259`) reaches neither
  copy.
- **Smallest fix.** One door in `tests/support/`, for example `spectra_viewer(script, …)`, used by both.
  Build its state by lifting the shipped `const state = {` as the trajectory harness already does
  (`follow_e2e:493`).

### T2 — SpectrumChart's browser and Plotly prelude is copied in two test files (medium-low; read both sides)

- **The copies.** `spectra/test_spectrumchart_seal.py:34-87` and `spectra/test_spectrumchart_handle.py:23-74`
  each carry:
  - the head that "loads" what is appended
  - `document.querySelector`
  - a Plotly stand-in with an `nsewdrag` area and `_fullLayout`
  - `__host` with the same nine `--spectrumchart-*` values
- **Already drifted.** Only the handle's stand-in has `relayout` and `ResizeObserver`; only the seal's
  records `config`.
- **Smallest fix.** Move the prelude into `tests/support/` beside `molview_dom_standin.js`, which both
  already read.

### T3 — Fixtures and preludes copied per test file (low-medium; read)

- **`flask_server`.** The same `serve()` fixture is defined in 27 test files, three of them in this slice:
  `test_spectrumchart_box_e2e.py:23-27`, `test_vibrationview_e2e.py:15-19` and
  `test_inspector_registry_e2e.py:28-32`. The `app_at` fixture in
  `test_the_viewers_follow_a_real_run_e2e.py:949-955` is a fourth copy under another name.
- **"List, then pick a result in the dropdown".** This is copied at `test_spectrum_results_e2e.py:499-503`
  and `:581-585`, `test_siesta_vibration_e2e.py:1121-1126` and `:1453`, and five more files. Beside it sits
  `support.sidebar.results_on`.
- **Smallest fix.** One session fixture in `conftest.py`, and a `support.sidebar.result_opened(page, base,
  folder, file)`.

### T4 — Zip artifacts are read by regex and byte scans, not by a zip reader (medium-low; read)

- **The scans.**
  - `test_vibrationview_mount_js.py:877-889` finds the manifest with `/\{[^{}]*"ffmpeg"[^{}]*\}/` and the
    entries with `/frame_\d{4}\.png/g`.
  - `test_vibrationview_e2e.py:191-205` slices the manifest out by `lastIndexOf("{")` and finds PNG headers
    by a byte scan.
- **What it misses.** A wrong central-directory offset or CRC passes both. The first test's docstring says
  this is the archive's only coverage.
- **Smallest fix.** Return the bytes (base64) and open them with Python's `zipfile`, which checks the CRCs.

### T5 — A test takes its expectation from the code (low-medium; read both sides)

- **The export kept for a test.** `_maths.js:38-39` exports `FRAMES_PER_CYCLE_MIN` and `_MAX`. Nothing in
  the module imports them; only the test does.
- **The test.** `test_vibrationview_maths_js.py:248-266` asserts `tooFew == M.FRAMES_PER_CYCLE_MIN`, so
  any floor passes.
- **The overlap.** `test_the_cycle_length_reported_is_the_one_that_happened` (`:269-296`) already
  hard-codes the contract's 15 and 1200. It also re-pins the rounding that
  `test_the_rounding_lands_on_the_duration_not_on_the_frame_count` (`:216-235`) pins.
- **Smallest fix.** Take 15 and 1200 from `vibrationview.md` § 10.1, un-export the constants, and fold the
  overlapping rows into one table.

### T6 — A hand-kept list of function names mirrors core.js's call graph (low; read both sides)

- **The list.** `test_spectra_no_spectrum_sentence_e2e.py:98-104` lifts 10 named functions to run
  `_noSpectrumSentence`, `_irRouteLabel` and `allPhasesComplete`.
- **Already missing two.** It omits `_ramanWait` (called at `core.js:1699-1700`) and `_irViewWords`
  (`core.js:1841`). Only because no row reaches them does it pass. The first file that does reach them
  fails with a ReferenceError, not on the contract.
- **Smallest fix.** Lift the whole helper block once, through the support door from T1.

### T7 — The same markup is pinned three times, and empty section headers remain (low; read)

- **The three pins.** `test_spectrumchart_seal.py:274-281`, `test_spectrumchart_handle.py:687-695` and
  `:135-142` each assert `["spectrumchart"]` / `["spectrumchart-surface"]`.
- **The empty headers.** `test_spectrumchart_seal.py:117-126` holds two section headers with no tests
  under them.
- **Smallest fix.** Keep one assertion and delete the empty headers.

### T8 — (counted under F5) `TestBackcompatAliases` keeps five dead aliases alive

`test_spectra_transition_js.py:186-194, 236-276`.

---

## The five that matter most

1. **F1:** the server's sentences reach the spectrum page and the frame-set card as raw `textContent`, so
   the same `infrared_said` shows a dash in the bar and `--` in the run summary.
2. **F2:** the infrared/Raman view and route words have two homes, the page and `spectra/methods.py`. A
   vector reads `0.707` in the table note and `0.707107` in the Methods paragraph on the same screen.
3. **F3:** the light directions, the partitions and every knob's default are hand-copied in the template
   and in `uiPrefs`, while the server owns them. `frameset_nodes` shows these could be served the same
   way.
4. **F5:** every state key has two spellings through backward-compatibility aliases. Five lifecycle
   aliases are read by nothing, and only a test keeps them.
5. **F6:** the modes table's columns are described four times; the filter already misses the Δgap column
   and the "pending" state the cells show.
