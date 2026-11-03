# fix-W3 — test_molbuilder_e2e, test_molview_e2e, test_vibrationview_e2e, test_results_export_e2e

Read whole before editing: the brief, W3.md, verified.md (W3), testing.md § 2a/§ 3b/§ 5, the four files,
molview.md §§ 1.1, 4, 6.6, 6.7, 6.7a, 9.3 (cell table), 10.3, 11.2, 11.2a, 13.2–13.4; tabs.md § 2;
structure.md § 2.2b; structure-periodicity.md § 6.0; presenters.md § 2; conftest `_isolated_workspace_store`;
support/live_server.py, support/results_export.py. Code paths read: molview ui.js (atom rows/chips, Assign
chooser, filter rows, Cell readout), model-jobs.js `offeredLabels`, render-engine.js `applyScene`/`resetView`/
`camera`, 3dmol-embed.js `setCell`/`fitCamera`/`look` (and vendor 3Dmol `zoomTo`, which includes shapes),
mount.js handle + click gate, stores.js `setSwitch`, modify/viewer.js timeline, modify/periodicity.js vacuum
payload, workspace_storage.py, _shared.py `workspace_payload` labels, kind_table.labels, runs.py folder
answer (`engine_words`, `noun`), inspectors/trajectory.js `resultCategory`, results/file-picker.js optgroups.
Nothing run but pyflakes and collect-only.

## 1. Per test touched

### tests/test_molbuilder_e2e.py

| test (line after edit) | action | contract § | guards now / one change that turns it red |
|---|---|---|---|
| module header (1–25) | docstring | molview § 4 (four exports), tabs.md § 2/§ 6, molview § 1.1/§ 6.6/§ 6.7/§ 9.3/§ 11.2/§ 11.2a, periodicity § 7 | archived-plan citation replaced by live homes; § 4 now "mount, formula, toDisplay, fromDisplay"; the gone "electrode/junction ops" hole dropped; generators' findings (validation § 4.1 R5) added to the holes. Section banners (§ 6.5 step N of the archived plan) repointed to the live §s |
| `_a_session_that_did_not_happen` (was :57) | retired | testing.md § 2a; conftest `_isolated_workspace_store` | duplicate of conftest's autouse patch; the floor test now reads the store's home through `workspace_storage.projects_root() / SCRATCH_DIR / "states"` |
| `_BADGE` (was :123) | retired | — | dead constant |
| test_the_page_mounts_a_viewer (was :347) | retired | — | every `_open` waits for card + canvas; :225 asserts no page error |
| test_a_second_load_adds_… (225) | fixed | tabs.md § 2 (*ADDED*); structure.md § 2.2b (*a label the canvas already has numbered*) | each row's label chips read as whole names == `[["FRAG"]]*3 + [["FRAG2"]]*2`. Red on: the incoming FRAG merged into the canvas's (5×FRAG), a replace (2 rows), or the canvas's FRAG lost (previously `"FRAG" in rows` passed via "FRAG2") |
| test_the_status_line_says_a_load_was_added (260) | docstring | molview § 6.7; tabs.md § 2 | goal + contract stated (NO GOAL). Red on: an add reported as "Loaded" |
| test_start_empty_then_load_is_how_you_replace (281) | docstring | molview § 6.7a; tabs.md § 2 (*with nothing open … nothing is asked*) | stale "Replacing is a separate gesture" framing replaced; states what it guards: Clear leaving atoms (5 not 2), the add-or-clear question on an empty canvas (`_load` with no answer times out) |
| test_each_append_is_a_point_… (324) | docstring | tabs.md § 2; molview § 1.1, § 11.2 | citation repointed |
| test_loading_a_project_file_shows_its_atoms_and_its_labels (361; helper `_offered` 353) | **added (OWED 2.65.0)** | molview § 6.6 *Where a label can be chosen*; *A default is nobody's choice* | Assign chooser's options == kind_table.labels() names in table order, then `interface`, `frozen_atoms`, then `SOLVENT`, each once; chooser shows the head; a `by label` filter row offers the same list. Red on: `offeredLabels` putting MolView's own first / sorting; `labelsOffered` not read from the structure answer; the filter row reading a different list; the chooser defaulting to anything but the head |
| test_the_cell_reads_the_same_on_both_surfaces (475; helper `_cell_row` 468, replaces `_origin_row`) | fixed | molview § 9.3 table (Vacuum: *default when the structure states none*) | MolView's **Vacuum** row (by its label) carries "(default)". Red on: MolView's Vacuum row losing its default tag (previously any row's "(default)" — Origin/Axes — passed) |
| test_committing_a_vacuum_… (506) | fixed | periodicity § 7 (one Apply); molview § 9.3 | after the commit, MolView's Vacuum row has no "(default)" and reads 5, 5, 5. Red on: MolView keeping the derived tag or not showing the committed gap |
| test_an_origin_is_assigned_… (710) | docstring | periodicity § 6.0 | "(plan § 5q D1)" dropped; uses `_cell_row(page, "Origin")` |
| test_the_edit_survives_a_page_reload (818) | docstring | molview § 11.2a *Coming back to a session*; § 1.1 | the unasserted badge / draft-vs-point claim and the wrong "count alone would pass" sentence removed; states the two guarded facts (edited atoms, position) |
| test_an_edit_changes_the_structure_… (868), test_retract_puts_the_atom_back_… (910) | docstring | molview § 1.1, § 11.2 (`load(-1)`) | citations repointed |
| test_retract_says_so_when_the_point_it_wanted_is_gone (951) | road kept (F9); store path repointed | workspace.md § 9.1; molview § 11.2a | same assertions; deletes point 0 in the store the server writes (conftest's), `tmp_path` arg dropped |
| test_the_timeline_indicator_says_where_you_are (1013) | docstring | molview § 11.2, § 11.2a | contract stated (NO GOAL) |

### tests/test_molview_e2e.py

| test (line after edit) | action | contract § | guards now / red on |
|---|---|---|---|
| `molview_server` (32) | fixed (duplicate) | testing.md § 2a, § 5 | now `support.live_server.serve()` (gains its own projects tree); `threading` import dropped |
| test_showing_the_unit_cell_draws_the_box_the_structure_uses (159) | **fixed (WEAK → § 10.3 position)** | molview § 10.3 (*A test of this must assert where the wireframe is drawn*); § 13.3 row | two readonly viewers, one with `showCell` on before installing water, one turned on after a hidden load; both `resetView()` (3Dmol `zoomTo` fits atoms + shapes, so both fit on atoms + box); whole pictures equal, and the after-load switch changed the picture. Red on: cell geometry handed down only while shown (box at world origin or absent when turned on after a hidden load) |
| test_isolate_hides_the_rest_and_gives_them_back (280) | fixed (claim) | molview § 1.1 (*stays set and starts hiding the rest the moment you pick something*); § 9.6 | the false "window click gate" claim replaced: the panel re-pick of atom 1 with isolate on draws exactly the `isolated` picture. Red on: isolate left set but not hiding on the next pick (or the auto-off returning). The window gate stays node-level (the page exposes no drawn atom position, § 9.9) — said in the docstring |
| test_playing_runs_the_movie_and_pauses (530) | docstring | molview § 1.1; § 13.2 | unasserted § 10.4 claim removed |
| test_the_card_brings_its_own_surfaces (558) | fixed | molview § 5.4, § 8.1 | the "not white" check (stated nowhere) dropped; banner "the card is dark" → "§ 5.4, § 8.1 — the card's surfaces are the module's own" |

### tests/test_results_export_e2e.py

| test (line after edit) | action | contract § | guards now / red on |
|---|---|---|---|
| module header (1–26) | docstring | periodicity § 6.0; presenters.md § 2 | "plan § 5q.4, T3" dropped; § 6.0's sentence quoted |
| `_holds_the_engines_frame` (114) — all three tests | fixed (WEAK) | periodicity § 6.0 (*recorded as the engine's*) | adds `back.offset_by == "engine"`. Red on: a Results save recording the origin as the person's |
| `_heading` (83); test_a_siesta_trajectory_… (134) | **added (OWED 2.65.0)** | presenters.md § 2 (engine words + noun, served with the folder answer) | the run's picker heading (its optgroup) == "SIESTA optimization". Red on: the heading guessed (pre-2.65.0 "… optimization" for every run, or SIESTA filed as PySCF) |
| test_a_pyscf_trajectory_… (184) | **added (OWED 2.65.0)** | presenters.md § 2 | heading == "PySCF optimization" |

### tests/test_vibrationview_e2e.py
No change: both tests RIGHT per the report.

## 2. SHARED (not made)
- molview.md § 11.2a, *Starting, in an editable viewer*: still says "edits raise the badge and lay down no point" — contradicts § 1.1 / § 11.2 (doc fix).
- molview.md § 13.3 row for § 6.6 ("the six appear … before any structure carries them") is stale since 2.65.0 (the kind's four come with the structure).
- molview.md § 8.1 does not state the "three grounds" (module ground, raised cards, darker scene) that test_the_card_brings_its_own_surfaces's docstring claims — the doc states it or the test's remaining claim goes.
- junction-cell.md § 2c still says "a `.xyz` or `.XV`" (2.13.2 made it a run folder or a pair's `.xyz`) — from the report.
- OWED 4 (labelled species `O1` saved and read back): contract gap first — molview.md says nothing about how a labelled species is listed or drawn.

## 3. F9 (left)
- `_register_tmp_as_picker_root` (Capabilities snapshot + method patch) → `isolated_projects_root`.
- `_load` and test_the_loader_readout_… (429) set the candidate with `projects.setShared` instead of the sidebar click; test_a_generated_structure_… (1130) sets the SMILES box by script.
- Copy pins: "Loaded"/"Added"/"5 in total"/"3" (260, 417), "Picked:"/"Loaded:" (429, 1094, 1130), "saved #N" (868, 910, 1013), "two atoms" title and "0.957" in the axis chooser (565), "-1.235, …" (744), "a = X.XXXX Å" (1245).
- Merges: 260 + 417 (status after a load); 868 + 910 + 1013 (one walk); results_export 134 + 184 (parametrize).
- test_retract_says_so_when_the_point_it_wanted_is_gone (951): deletes the store's file; "Nothing changed" vs the contract's *there is nothing further back*; NOT E2E (tier-1 node test of `retractState`, or 31 real edits).
- results_export `_open`: patched `file_picker_roots` + planted `sessionStorage` → the one projects door + sidebar clicks.
- molview_e2e `_settle` (250 ms sleep) and `_canvas_pixels` (PNG data-URL LENGTH) → wait on the window cover, compare whole pictures (the new § 10.3 check already does both in-page).
- The window click gate under isolate (`isolateInEffect` at mount.js) has no page road: needs a way to aim a click at a drawn atom.
- OWED not added: 2.57.0 builder finding row (no test in these files builds a Peptide).
- Unverified by reading (verified.md W3-picker): the picker's option value vs the run door's absolute path that results_export `_open` waits for; the batch shows it.

## 4. Checks
- `pyflakes` on the four files: 0 messages.
- `pytest --collect-only`: 49 tests collected (was 50; one retired), no errors.
- Not runnable here, so first seen in the batch: the palette walk (Filter-mode label click, by-label row), the § 10.3 two-viewer picture equality, the Vacuum row's "5, 5, 5" parse, the isolate re-pick picture equality, the optgroup heading reads, `offset_by == "engine"` on the three saves.
