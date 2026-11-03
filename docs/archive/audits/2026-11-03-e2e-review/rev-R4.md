# rev-R4 — review of the W3/W4 e2e revisions, the GPU notice, and this round's doc edits

Read whole: the 16 test files in scope, `tests/data/presenters.toml`, `git show HEAD:tests/test_source_inspector_e2e.py`,
fix-W3.md, fix-W4.md, W3.md, W4.md, X.md, verified.md. Code paths read: molview `ui.js` (atom rows and chips, Assign
chooser `drawTargets`/`knownLabels`, mode options, filter rows, Cell readout `drawCell`), `model-jobs.js`
(`offeredLabels`, `PREDEFINED_LABELS`, `createLoad`/`requestBodyFor`, `OPERATIONS` checkpoint column,
`effectiveCell`), `model.js` (`recordEdit`, `installMolecule`, history wiring, selection subscription), `history.js`
(`send`, `edited`, `save`, `load`, `adopt`), `mount.js` (handle, `data.selection`, busy cover), `render-engine.js`
(`applyScene`, `switchesChanged`, `rebuildGuarded`, `resetView`/`fitNow`), `3dmol-embed.js` (`setCell`,
`fitCamera`/`look`, `paint`, batches), vendored 3Dmol `zoomTo` (shapes included in an empty-selection fit),
`kind_table.labels`/`kind_table.toml`, `_shared.workspace_payload`, `runs.folder_answer`/`file_row`/`_label_of`,
`file-picker.js` (`filterToResultFiles`, `metaOf`, `_populate`, `_onFileSelected`), `inspectors/trajectory.js`
`resultCategory`, `_partial_inspector_factory.js`, `trajectory/core.js` tab-return handlers, `documents/page.js`
theme, `app-notifications.js`, `workspace/dispatcher.js`, `system-load-monitor.js` (whole), `system_load._gpu_snapshot`,
`preview.js` caps, `web/tabs.py`, `runwrap.py` banner, `validation/chemistry.check_electronic_state`,
`electronic_state._spin`/`_method`, `template.offered`/`is_member`, the catalogue's `unpaired_electrons` rows,
`task-setup/viewer.js`, conftest's e2e hook and `_isolated_workspace_store`, `support/live_server.py`. Nothing run.

**No new assertion was found that will fail on correct code.** Each new browser assertion was traced to the page JS:
the palette walk (chooser options = `offeredLabels(labelsOffered)` then the structure's regions; `+ new label…` and
`Choose a label…` carry value "" and are filtered; the chooser shows `names[0]` while `pickedByPerson` is false), the
Vacuum row (`dt:text-is('Vacuum') + dd`, `vector(resolved_vacuum)` → `5.000, 5.000, 5.000`, tag only when
`getVacuum()` is null), the label chips (`.molviewer-selection-tag > span` is the name; `_unique_label` numbers to
`FRAG2`), the two-viewer § 10.3 picture (overlay refresh paints synchronously; `resetView` → `look` → `setView` +
`zoomTo()` over atoms and shapes in both), the isolate re-pick (rebuild with `fit=false`, same drawn key), the optgroup
headings (`folder_answer` → `engine_words`/`noun` → `resultCategory`), the docs-theme new page (localStorage shared
by pages of one context, sessionStorage not), the pageshow parametrization (handlers call `pollOnce` live or not).
Unverified by reading only: pixel equality across two WebGL contexts (the same determinism the § 9.9a test relies on),
and the picker option value vs the run door's path in `results_export._open` (pre-existing, W3-picker).

## Findings, ranked

### Wrong expectation

1. **tests/test_molview_e2e.py:556-577** `test_the_card_brings_its_own_surfaces` — the revision dropped "not white" and
   re-bannered the test "§ 5.4, § 8.1 — the card's surfaces are the module's own", but its three assertions (window card
   == panel card, ground ≠ card, scene ≠ card; docstring l.562 "Three grounds … (§ 8.1)") have no contract sentence:
   molview.md § 8.1 (l.1549-1575) and § 5.4 state no surface colour; `grep -i "ground"` in molview.md finds only the
   View-menu background. W3's SHARED item, unresolved. Fix: state the three surfaces in § 8.1, or keep only § 5.4's
   "the host hands over an empty element" claim.
2. **tests/data/presenters.toml:275-279** (row "… under SIESTA trajectory") — the meta `{engine, engine_words, role,
   parser}` without `noun` is a combination the server never sends: a folder not ours has `role: None`
   (`runs._label_of` l.696-697 → `file_row` l.1156), and `trajectory.match` needs `meta.role`, so the heading
   *SIESTA trajectory* cannot reach any page. presenters.md § 2 (l.139-142) states it while the same paragraph (l.128)
   says `trajectory` "claims a file the catalogue names — its `role`". `registry_answers` checks fields one at a time,
   so the row passes. Fix: the contract rules first (drop the words-only heading, or let a non-ours file carry a
   role); then the row follows.

### Retirement

3. **tests/test_inspector_registry_e2e.py (HEAD :394) `test_partial_fetch_404_renders_error_card` retired** — it was the
   only test of presenters.md § 2 l.153 *"a `mount` that throws falls back to a clean error card in the panel"*
   (`_partial_inspector_factory.js` `.catch` → `_renderError`, title "… inspector failed to mount"); after the
   retirement no test in tests/ references `error-card`, `_renderError` or "failed to mount". fix_brief § 5 put
   `page.route` fakes in otherwise-right tests to F9, not to retirement. The AbortError companion (HEAD :439) went too;
   results.md § 4 l.1062 *"a late response from a previous file can't write into the current view"* is now unpinned
   (fixer lists it under F9). Fix: keep the 404 test until F9 re-homes it on a real road.

### Vacuous / over-claimed

4. **tests/test_molview_e2e.py:198-256** (§ 10.3 picture compare) — the comparison is relative (cell turned on after a
   hidden load vs. shown at load). It catches the named mutation (geometry only while shown), but a box drawn at the
   world origin in both viewers (e.g. `3dmol-embed.js` `setCell` ignoring `cell.origin`) draws two equal pictures and
   passes. That is the picture § 10.3 names, and § 13.3 l.4935 asks for "draws the box at the structure's corner".
   The row's second half ("a cell edit while the cell is hidden still updates the anchor") is not asserted. Fix: add a
   third viewer with the same water shifted by (10,10,10) (Automatic origin). With the box at `box_corner`, its fitted
   picture equals the first; with the box at the world origin, it does not.
5. **tests/test_results_export_e2e.py:145-146 and tests/test_siesta_run_in_the_viewer_e2e.py:100** — the docstrings
   name "every run read *… optimization* before 2.65.0" and "SIESTA's runs filed under *PySCF optimization*" as failures
   these headings catch. On optimization runs, with the `.out` picked, a fixed noun and a suffix guess both still give
   *SIESTA optimization* / *PySCF optimization*. Only the missing-words mutation (→ *Trajectory*) turns them red. Fix:
   name only that mutation, or read the noun on a non-optimization run the pass already opens (the CO2 vibration's
   relax call in `test_inspector_pageshow_refresh_e2e.py` → *SIESTA vibration*).
6. **tests/spectra/test_spectrumchart_box_e2e.py:33, :98** — renamed `…_the_surface_fills_it_unpadded`, but "fills" is
   asserted only as `0 < surfaceInner <= frameInner`, which a surface at half the frame's width passes
   (spectrumchart.md § 11, l.1003: *"the surface fills it"*). Fix: compare the surface width with the frame's content
   width (clientWidth minus the frame's computed padding), or drop "fills" from the name.
7. **tests/test_molbuilder_e2e.py:1015** — the first docstring line, "Unsaved work and the point Retract would restore
   are both on screen", is stale. On this walk the delete lays its point (`OPERATIONS.delete.checkpoint: true`,
   `recordEdit` → `save(1)`), so no unsaved work exists, and nothing asserts a badge. Fix: "the position, and the point
   Retract would restore".

### Duplication

8. **tests/test_siesta_run_in_the_viewer_e2e.py:24, :31-32** — the added `pytest.mark.e2e` and module-level Playwright
   `importorskip` duplicate conftest's hook (conftest.py l.1203-1226 marks every `*_e2e.py` e2e and adds the Playwright
   skip: "THE SKIP BELONGS WITH THE MARKER, not in each file"). W4's finding 17 ("errors instead of skipping") and
   fix-W4's "now skips instead of erroring" are false. Harmless. Fix: drop the marker; the flask skip may stay.

### Product

9. **molbuilder/data/catalogue.template.toml:1231-1247 vs transport.md § 3.1 (l.4858-4861) as edited** — the
   catalogue still offers `siesta.transport = [0, "free"]` and declares `not_offered on = "transport"`. So
   `validation/chemistry.py:266-269` refuses a stated count 1..10 on a SIESTA transport calculation as an ES4 error
   ("cannot run here …") before ES6's information path (l.270-281) is reached; only a stated 0 under unrestricted gets
   "written as stated". The doc now says "a stated count is written as stated, TranSIESTA's stop said beforehand as
   information, never refused". So do chemistry-correctness ES6 (l.502-510) and the § 2a.3 `free` row's
   SIESTA-transport cell (l.586), and 2.65.1's commit says "a fixed spin count where SIESTA stops on one -- the deck
   now writes the stated count". The catalogue row was last changed 876972bb2 (2026-09-30), so 2.65.1 left it. Under
   the user's 2026-11-02 rule (an engine's own stop is never refused) the catalogue is the defect. Fix: widen the
   transport row to the item's choices and keep its `why` as the ES6 information. If the refusal is meant to stand,
   transport.md must say "refused by name" again.
10. **molbuilder/web/static/lib/system-load-monitor.js:5, :318-319, :366** — the new notice text matches results.md
    § 6.3, but the comments still say "psutil + nvml under the hood", that the server sets `gpu_error` "when NVML was
    installed and would not start", and "when NVML didn't return them". The sampler is `nvidia-smi` as a subprocess
    (`system_load.py:15-17`, `:150-170`). Fix: say nvidia-smi.

### Docs

11. **docs/web/molview.md § 11.2a l.3757-3762 and § 12.1 l.4729-4733** — still the explicit-save model, contradicting
    § 1.1 and § 11.2 as edited and `recordEdit` (model.js:414-416; every `OPERATIONS` row `checkpoint: true`,
    model-jobs.js:781-821):
    - § 11.2a's example: "install, edit, save(1), edit again → the badge is up, and Retract goes to point 1". Under
      the current code: #0, #1, #2, #3, badge down, Retract → #2.
    - § 12.1: "the viewer now has unsaved work: the badge appears … Nothing is written on its own … There is unsaved
      work, so this first press spends it".

    Same drift at:
    - § 11.2 l.3439 "Stepping forward lasts until you save": an edit's `save(1)` prunes too.
    - § 11.2 l.3493-3498 "make the timeline automatic and every keystroke becomes a milestone".
    - history.js:289 "an explicit-save history" and :381 "From a saved point with edits sitting on top of it".

    Fix: rewrite these to the every-edit-a-point rule.
12. **docs/web/molview.md § 11.2 l.3436** — "(an operation that lays none, a selection changed)" is wrong. A
    selection change raises no badge and writes no draft: model.js:197-205's subscription only calls the renderer, and
    `history.edited()` is reached only from `recordEdit(false)`. So the first Retract after a selection change steps
    back a point (history.js `load`: `step < 0 && uncommitted` is false); it does not "discard that work and leave you
    on the point". Fix: drop "a selection changed".
13. **docs/web/trajectory.md l.267** — the new *Coming back to the tab asks once* paragraph dates the fix "(2026-10-26,
    #192)". The handlers landed in 6ceddca2c on 2026-06-02 (#194), and the test docstring dates #192 to 2026-06-02.
    Fix: 2026-06-02.
14. **docs/web/presenters.md l.117** — "for three of those five there is no filename answer" now follows a
    seven-field `meta` (`engine_words` and `noun` were added). Fix: recount.
15. **docs/web/molview.md § 11.2a l.3724-3725** — "→ **point 0 is laid down**, and the point the structure opened at."
    is garbled. Fix: "point 0, the point the structure opened at, is laid down".

## Verdict

The revisions carry no assertion that should fail on correct code. Each new browser check (label palette order, the
Vacuum row, the FRAG/FRAG2 chips, the § 10.3 two-viewer picture, the isolate re-pick, the optgroup headings, the
docs-theme later page, the parametrized tab return) matches the page JS as written.

Expectation sources:
- The two inherited problems are unresolved: the card-surface test has no contract sentence, and the words-only
  heading row pins a state the server cannot produce.
- Two docstrings name mutations their optimization runs cannot show.
- The § 10.3 compare is relative, so it misses a box drawn at the world origin in both viewers.

Retiring the 404 error-card test removed the only guard of presenters.md § 2's error-card fallback. That went past
the brief, which sent `page.route` fakes in right tests to F9.

The GPU notice is correct; its file's comments still speak of NVML.

The doc corrections are mostly right:
- workspace.md § 3, notifications.md § 3, tabs.md § 5.5, task-setup.md § 5.2, running-a-job banners, results.md
  § 6.3/§ 10 and the chemistry § 2a.3 PySCF cell were all checked against the code.
- molview.md's turn to "every edit lays a point" is incomplete: § 11.2a's example, § 12.1, two § 11.2 passages and
  history.js still teach the explicit-save model, and one new parenthesis about selections is wrong.
- transport.md § 3.1 now promises "never refused" for a stated count that the catalogue still refuses on SIESTA
  transport. That is a product defect to fix before the batch's transport results are read.
