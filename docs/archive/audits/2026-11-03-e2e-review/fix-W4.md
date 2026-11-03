# fix-W4 — Results shell, presenters, sidebar, preview, notifications, load strip, layout, docs pane

Nothing was run except pyflakes and `--collect-only`. No e2e, engine, browser or server was started.

## 1. Per test touched

| test (file:line after edit) | action | contract § | what it guards now / one code change that turns it red |
|---|---|---|---|
| `tests/data/presenters.toml:18-26` (header) | fixed | `presenters.md` § 2 | meta described as the folder answer `{role, parser, engine, engine_words, noun}`, and the words and noun listed as checked against `engine_table` / `kind_table` |
| `tests/data/presenters.toml:257-295` (5 heading rows) | fixed (WRONG DETAIL + STALE ROAD) | `presenters.md` § 1–2 (2.65.0) | rows now carry the folder answer's `engine_words`/`noun`: *SIESTA optimization* (`.out`, SIESTA, optimization), *PySCF vibration* (`.molwatch.log`, PySCF, vibration), *SIESTA trajectory* (words only), *Trajectory* (neither), *Trajectory* (`engine = "unknown"`). Turns red if `trajectory.js` `resultCategory` reads `meta.engine` or a fixed noun |
| `test_inspector_registry_e2e.py:61` `registry_answers` | fixed | `presenters.md` § 2; `runs.folder_answer` | also checks each row's `noun` against `kind_table.kinds()` nouns, and checks that `engine_words` are `engine_table.engine(engine).words` for the row's own engine (as `folder_answer` l.1253 makes them). Turns red if a row carries words the server cannot send (e.g. `"Siesta"`, or SIESTA's words with `engine = "pyscf"`) |
| `test_inspector_registry_e2e.py:146` `test_the_dropdown_files_it_under_its_heading` | docstring | `presenters.md` § 1–2 | GOAL, the 2026-09-04 and pre-2.65.0 failures, MUTATION (resultCategory reading `meta.engine` / a fixed noun) |
| `test_inspector_registry_e2e.py` (old :394) `test_partial_fetch_404_renders_error_card` | retired | — | a faked 404 (`page.route`) with no contract §. Now unguarded by e2e: the factory's error card on a failed partial fetch. `test_results_blueprint.py` covers the served partials |
| `test_inspector_registry_e2e.py` (old :439) `test_aborterror_during_mount_does_not_render_error_card` | retired | — | It mounted the source pane, which mounts nowhere (`presenters.md` § 3), from a faked 200. A rewrite is not possible in this file. The only folder here with two result files holds two `.xyz`, and the structure viewer fetches no partial, so two picks there never reach the `AbortError` guard (`_partial_inspector_factory.js:168`). That guard is now pinned by no e2e (see F9) |
| `test_inspector_registry_e2e.py` class `TestInspectorErrorCardRuntime` | retired (whole) | — | both tests above; `_open_results` is still used at :365 |
| `tests/test_source_inspector_e2e.py` (10 tests) | retired (file deleted) | `presenters.md` § 3 | the text pane mounts nowhere; the reachable text viewer is the preview modal (`test_preview_modal_*`) |
| `test_siesta_run_in_the_viewer_e2e.py:23-33` module markers | fixed | `testing.md` § 1, § 5 | adds `pytest.mark.e2e` and `importorskip("playwright.sync_api")` / `("flask")`. Without Playwright it now skips instead of erroring |
| `test_siesta_run_in_the_viewer_e2e.py:1-11` module docstring | docstring | `testing.md` tier 3 | the archived "plan § 5y" replaced by `support/real_runs.layered`, `process/testing.md` tier 3 |
| `test_siesta_run_in_the_viewer_e2e.py:87` (+ assertion :117-123) | added (OWED 1) | `presenters.md` § 2 (2.65.0): *"the engine's words and the calculation's noun … — *SIESTA optimization*"* | on the H2 coarse `run-0` it already opens, the menu's optgroup holding the mounted file reads `SIESTA optimization`. Turns red if `file-picker.js` `filterToResultFiles` stops carrying `said.engine_words` / `said.noun` (the heading becomes *Trajectory*) |
| `test_siesta_run_in_the_viewer_e2e.py:189` teardown | fixed (stale meta) | `presenters.md` § 2 | meta now as `metaOf` builds it (engine, engine_words, noun added); citation `_metaFor` → `metaOf`. The assertions are unchanged |
| `test_inspector_pageshow_refresh_e2e.py:124` `test_coming_back_to_the_tab_asks_the_server_at_once[pageshow / visibilitychange]` | fixed: two bodies → one parametrized test, module docstring rewritten | `trajectory.md` § 5, *Coming back to the tab asks once* | on the finished CO2 relax call, each tab-return event asks `/api/watch/data` at once. Turns red if `core.js` `_onPageShow` / `_onVisibilityChange` is removed, or asks only while the run is live |
| `test_results_file_picker_e2e.py:156`, `:161` | docstring | `results.md` § 2.1 (*Come back to the browser tab*), § 2.2; `presenters.md` § 3 step 4 | stale class docstring ("Same fix, different event") replaced; contract cited |
| `test_system_load_monitor_e2e.py:40-44` `_REASON` | fixed (wrong docstring) | `results.md` § 6.3; `system_load._gpu_snapshot` | the faked reason has the shape the sampler composes today for the measured 2026-09-18 mismatch (`nvidia-smi exited 18: Failed to initialize NVML: …`). The old NVML string's comment, *"verbatim … text a user really sees"*, was false |
| `test_system_load_monitor_e2e.py:109` `test_opening_the_card_gives_the_reason_in_full` | fixed (WRONG DETAIL) | `results.md` § 6.3: *"the server asks the driver again every few seconds … nothing needs restarting"* | the notice holds the reason, *"asks the driver again every few seconds"* and *"clears by itself"*. Turns red if the notice goes back to "restart the server" |
| `test_system_load_monitor_e2e.py:85`, `:140` | docstring; tooltip pin dropped | `results.md` § 6.3 | :140 no longer pins `title == "Hide server load"`; it keeps the no-mark / no-notice outcome |
| `test_app_notifications_e2e.py` (old :37) `test_bar_present_and_hidden_until_first_notification` | retired | — | presence only, one page: not an e2e |
| `test_app_notifications_e2e.py:37`, `:58`, `:77` | docstring (NO GOAL) | `notifications.md` § 1–3; `workspace.md` § 5 | goal + silent failure stated |
| `test_app_notifications_e2e.py:109`, `:137` | docstring + wait fixed | `notifications.md` § 3: *"A save that succeeds lowers it … only that row, the others stay"* | :137's fixed 150 ms wait replaced by a wait for the save row to go; it still turns red if `persist-ok` calls `clearAll()` |
| `test_docs_pane_theme_e2e.py` (old :51) `test_a_first_visit_reads_dark` | retired | — | subsumed by :53 (the painted colour) |
| `test_docs_pane_theme_e2e.py:53` | docstring | `tabs.md` § 5.5 | dark start, the 2026-08-23 defect |
| `test_docs_pane_theme_e2e.py:71` `test_the_toggle_switches_and_is_kept_for_later_pages` | fixed (WEAK) | `tabs.md` § 5.5: *"a page opened later in the same browser starts on the surface last chosen"* | toggle to light, then a NEW page in the same context opens light; no storage key read. Turns red if `documents/page.js` stores the choice in `sessionStorage` or does not read it on open |
| `test_preview_modal_edit_save_e2e.py:249` large file | docstring fixed | `projects.md` § 4; `preview.js` `VIEW_ONLY_BYTES` 1 MB, `BULK_READ_MAX_BYTES` 16 MB, `PAGE_BYTES` 256 KB | wrong caps ("VIEW_ONLY_BYTES … load paginated", "32 MB cap") corrected; the docstring now states that the 1 MB cap alone is not reached |
| `test_preview_modal_edit_save_e2e.py:114`, `:203` | docstring (NO GOAL) | `projects.md` § 4 | contract and silent failure cited; the unused `pathlib.Path` import removed |
| `test_layout_fits_e2e.py:31-35` `PAGES` comment; `:115` docstring | docstring | `ui-contract.md` § 3 | "Every served page" was false: it is six of the nine `TABS`. `test_content_fits_its_box` gets its GOAL |

`test_sidebar_e2e.py` is untouched; its only finding is the picker-root seam, which is F9.

## 2. SHARED (not made: docs, other files, product)

- **`presenters.md` § 2: *SIESTA trajectory* cannot be reached through the server today.** For a folder not ours, `runs._label_of` returns `None` (l.696-697), so `file_row` gives `role: None`, and `trajectory.match` needs `meta.role`. The words-only heading is never on a page. Needs a ruling: drop it from the contract, or let a non-ours folder's file carry a role. The toml row pins the presenter rule only.
- `presenters.md` § 1's table and § 2's meta list `{role, label, stage, parser, engine}`: add `engine_words`, `noun`. § 1's table row already reads right.
- `results.md` § 10 test map:
  - l.~1399 maps `test_inspector_pageshow_refresh_e2e.py` to *"the re-scan on tab return"*. It tests the trajectory viewer's tab-return ask (`trajectory.md` § 5).
  - l.~1411 maps `test_results_file_picker_e2e.py` to tests retired 2026-10-04; one test is left.
- `trajectory.md` § 8 test map: add `test_inspector_pageshow_refresh_e2e.py` for § 5's tab-return ask.
- `tabs.md` § 9 / `notifications.md` § 5 test maps: nothing is wrong. `test_docs_pane_theme_e2e.py` is listed nowhere I saw; add it under § 5.5 if wanted.
- `tests/test_build_e2e.py:56` cites the deleted `test_source_inspector_e2e.py` in a docstring (another group's file).

## 3. F9 (left, one line each)

- `test_inspector_registry_e2e.py`: `_partial_inspector_factory.js`'s `AbortError` guard is now unpinned. Re-home it as two dropdown picks in a row (the H2 run's trajectory, then a structure) on a real run folder.
- `test_inspector_registry_e2e.py:181` spectra teardown row (WEAK): the file is absent, so only the refused load is cycled. Move it onto the CO2 vibration's real spectrum.
- `test_inspector_registry_e2e.py:269`, `test_siesta_run_in_the_viewer_e2e.py:60-70`: planted `sessionStorage['molbuilder.current_dir']`. The road is sidebar clicks + Reload.
- `test_inspector_registry_e2e.py:344`: meta typed for a file no folder lists.
- `Capabilities.file_picker_roots` patch in preview, sidebar, pageshow, file-picker and siesta-viewer.
  - **`test_preview_modal_edit_save_e2e.py:28-37` and `test_sidebar_e2e.py:31-40` never restore `diagnostics._snapshot`**, so a `runtime_config={}` snapshot outlives the test in that worker. Fix this first.
- `test_preview_modal_edit_save_e2e.py`:
  - `showPreview()` via `import` instead of the row's ⋯ → View.
  - "Unsaved" / "Saved at" / "Large file … external editor … Ctrl-F" copy pins (copy still right).
  - :249 WEAK: the 1 MB edit cap is never reached. Needs a size table: 2 MB view-only and whole, 17 MB paged.
- `test_inspector_pageshow_refresh_e2e.py`: synthetic `pageshow` / `visibilitychange` events, `setShared` planting.
  - The *"one ask, not a restart of polling"* half of § 5 is not asserted, because that needs a > 15 s wall.
- `test_results_file_picker_e2e.py:201`: "Scanning" copy pinned while `__mbScans` is installed and never read. Assert the count grew.
- `test_system_load_monitor_e2e.py`: the in-process `_gpu_snapshot` fake (decision: kept).
  - :140 asks the real `nvidia-smi`, so it goes red on a host whose driver is broken.
- `test_app_notifications_e2e.py`: synthetic `persist-error` / `persist-ok` events instead of a failed then recovered save, so `dispatcher.js:129-157`'s wiring is untested.
  - :37 and :58 are a row-merge candidate.
  - :77 clears the last row with `clearAll()`. The **Clear all (N)** button shows only at ≥ 2 rows.
- `test_layout_fits_e2e.py`: rows should be `web/tabs.TABS` × widths, one load for both probes, and pages with content (a bound Results, Task setup on a calculation).
- OWED 2–5 from W4.md § 3 (Run panel at a run of points' root, the ladder's runs picked in place, the picker's own binding rules, the SCF banner's engine) are not added. Their runs are made by other modules' fixtures or need a new road in these files. OWED 6 is another group's (stopped run).

## 4. Checks

- `pyflakes` on the 10 remaining files: clean (rc 0).
- `pytest --collect-only`: **85 collected**, no errors:

  | file | collected |
  |---|---|
  | registry | 53 (22 pick + 15 result + 12 heading + 3) |
  | layout | 8 |
  | preview | 5 |
  | notifications | 5 |
  | siesta-viewer | 4 |
  | load monitor | 3 |
  | pageshow | 2 |
  | docs theme | 2 |
  | sidebar | 2 |
  | file picker | 1 |

  `test_source_inspector_e2e.py` is deleted.
