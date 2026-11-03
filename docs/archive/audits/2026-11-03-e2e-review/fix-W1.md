# fix-W1 — Task setup (prep, cell types, one folder, restore) and the bench summary

Files edited: `tests/test_task_setup_prep_e2e.py`, `tests/test_task_setup_cell_types_e2e.py`,
`tests/test_task_setup_reacts_to_a_restore_e2e.py`, `tests/test_bench_summary_inspector_e2e.py`.
`tests/test_task_setup_one_folder_e2e.py`: no change (every test RIGHT, citations live).
Backups of the bench and restore files before edit: `bench.bak`, `restore.bak` in this folder.

## 1. Per test touched

| test (file:line after edit) | action | contract § | what it now guards · one change that turns it red |
|---|---|---|---|
| `test_the_buttons_produce_a_folder_that_carries_what_the_card_asked_for` (prep:169) | fixed (STALE ROAD, vacuous step) | task-setup.md § 6.2b *"The launch rows stand on every rung's card from the start … blank until stated"*; table row *unstated* | the *+ Add setting* step (a no-op: `mpi_np` is a standing row, so the picker never offers it) is gone; the test now asserts the `mpi_np` row STANDS, `data-kind="unstated"`, field empty, before it types 8 into it. Red: `standingRows()` dropping the catalogue's `run_card` items (the row never appears, the wait times out), or the row born filled. Also "wrapper" → "run script"; the removed `stages.md § 6.8d` cite is gone. |
| `test_the_tab_saves_the_folder_first_and_says_so` (prep:270) | docstring | task-setup.md § 11 | "(W55 B4)" (archived plan) → § 11 *Each group's next step is always there*. |
| `_commands_shown` (prep:157) | docstring | task-setup.md § 6.4 | "W55 B4" → § 6.4 (the commands are the terminal's own, served by `/api/task-setup/commands`). |
| `filled_dir` comment (prep:~430) | citation | job-request.md § 3.2a | `§ 6.8e` (gone) → job-request.md § 3.2a. |
| `test_the_queue_the_request_names_stays_through_a_machine_and_an_ask` (prep:502) | citation | task-setup.md § 6.1 | § 6.2b → § 6.1 (where *"The queue the job request names is the job request's"* now lives). |
| `test_a_folder_opens_showing_what_it_already_states` (prep:592) | citations + **added (OWED, 2.65.0)** | task-setup.md § 3 *"the page draws a cited slot's row where the kind cites one, each carried field's row where the kind carries it …, a structure pair where the description names one"*; job-request.md § 3.2a | `§ 6.8e` ×2 → job-request.md § 3.2a. Added: *What came over* (`#ts-facts`) shows `Formula` = `H2` and `Atoms` = `2`, and no row stands empty (`—`) — an optimization cites no slot and carries no field, so a row drawn for one would read `—`. Red: `renderCameOver` drawing a slot row by a kind name (e.g. `kind.cites` taken as set), or dropping the structure pair. |
| `test_a_check_that_cannot_run_SAYS_SO_rather_than_going_blank` (prep:1036) | docstring (NO §) | task-setup.md § 6.2a, § 6.2b | cites § 6.2a *"a failed fetch says the check could not run, with the server's reason"*; "(W55 B3)" → § 6.2b. |
| section header (prep:~1084) | citation | task-setup.md § 6.4, § 11 | `plans/plan.md § 5h, cluster 4` (archived) → task-setup.md § 6.4, § 11. |
| `test_the_commands_the_card_hands_over` (prep:1130) | docstring (NO §) | task-setup.md § 11 | cites § 11's *measure it* / *run it* rows. Hint-text asserts left (F9). |
| `test_choosing_a_machine_puts_it_in_the_command_you_copy` (prep:1217) | docstring (NO §) | task-setup.md § 6.4 | quotes *"They carry `--target` whenever the choice was required … and `launch` never carries it"*. |
| `test_a_calculation_set_to_its_machine_is_shown_so` (prep:1255) | docstring | task-setup.md § 6.4 | "W55 D12" → § 6.4 *A calculation already set to its machine is shown so*. |
| `test_the_notify_card_offers_this_machines_channels` (prep:1286) | docstring (NO §) | task-setup.md § 9b | quotes *"The list is painted from `GET /api/notify/channels`"*. |
| `test_no_rung_is_taught_a_from_by_default` (prep:1414) | docstring | job-system.md § 5.4; task-setup.md § 11 *Continue from* | "newest attempt" → "newest run"; "plan W37" → task-setup.md § 11. |
| `test_the_k_grid_is_not_stored_as_text` (was cell:164) | **retired** | task-setup.md § 5.2 | false premise (`"[4, 4, 1]" == [4, 4, 1]` is already False), so `test_a_typed_k_grid_lands_as_three_numbers` (cell:154) fails on any string. |
| `test_a_stored_value_the_kind_does_not_offer_is_shown_as_itself` (cell:294) | docstring | task-setup.md § 5.2 (now states it) | quotes § 5.2 *"A stored value this kind does not offer is shown as itself, marked — `none (not offered here)` —"*; the exact-text assert is now the contract's own words. Red: the viewer without the out-of-set branch (viewer.js:413-420). |
| `test_a_restore_updates_the_tab_that_is_showing_the_folder` (restore:134) + module docstring | citations | projects.md § 2 | `projects.md § 4.1` ×3 → § 2 (the `onFolderChanged` / `publishFolderChanged` rule, *"Anything that rearranges a folder behind an open tab publishes"*). Removed the file's unused `import shutil` (pyflakes). |
| `_sandbox` (was bench:40) | **retired** | testing.md § 2a–2b | obsolete since § 2b; its `delenv("XDG_CONFIG_HOME")` moved the machine scope out from under conftest's record, which forced the hand-written calculation copy. |
| `sweep` fixture (bench:42) | fixed (WRONG DETAIL + hand-written record) | gpu.md § 1.1, G2; generator.md § 4.3a *"Whether the trials are a GPU family is … the bench's points for it, else the calculation's card"*; task-setup.md § 6.1; configuration.md M-3 | `use_gpu`/`diag_algorithm` moved from `SiestaConfig` (the template, which writes no run setting) to the calculation's run card (`execution: {use_gpu: true, diag_algorithm: ELPA-1STAGE, gpu_count: 1}`, as tier 1's `test_prep_bench_fold.calc` states it); the machine is now this machine's record via `conftest.write_machine_record(scheduler=WORKSTATION, topology=Topology(sockets=1, cores_per_socket=4, gpus_per_node=1, gpu_type="a100"))`; the hand-written `calc/environment.json` is gone — prep copies the record (M-3). The sweep is now the GPU family its name says. |
| `test_the_declared_cadence_is_the_trajectory_viewers` (was bench:149) | **retired** | bench-summary.md B4 | asserted the exported `pollMs` constant (not e2e); its guard moved into the next test as an outcome. |
| `test_it_really_re_polls_and_stops_when_put_away` (bench:148) | fixed (absorbs the retired cadence check) | bench-summary.md B4 *"polled on the trajectory viewer's cadence (15 s)"* | each `/api/bench/summary` fetch records the delay of the timer it fired from; asserts a re-poll was armed by a 15 000 ms timer. Red: `POLL_MS` off 15 000 (before, a shorter cadence such as 3 000 ms still met "≥ 3 polls in 10 s" and passed). |

## 2. SHARED

None needed. (The bench fixture's GPU record goes through `conftest.write_machine_record` as is.)

## 3. F9 — left as is

- Every `_open` plants raw sessionStorage keys (prep:143, cell:105, one_folder:101, restore:154, bench:111) instead of a sidebar click / `handOffSelection` (projects.md § 2).
- cell_types fixtures `calc_dir`, `bool_column_dir`, `out_of_set_dir` build the calculation by hand (`write_task(Task(...))`, typed `.xyz`).
- bench `sweep`: the `Capabilities.file_picker_roots` patch + `set_capabilities` (isolated_projects_root already points the door); bench `server` duplicates `support.live_server.serve`.
- prep notify test writes the channel file by hand (the door is `PUT /api/notify/channels/<name>`).
- prep `setShared` / one_folder `publishCommit` folder switches (app calls, not clicks).
- prep `page.route` hold in `test_a_bench_grid_answer_that_arrives_late_is_dropped`.
- TEXT-PINNING on copy still right: prep:365 ("no axes are declared"), prep:406 ("Measure a stage" absence), prep:~531 (queue sentence), prep:~1175/1181 (hint words), cell:~190 (`refuse | kgrid = '4,4'`).
- Table merge: prep:353 + prep:371 into one row table (SIESTA no axes / PySCF refused); one shared describe helper for prep/restore/bench.
- Left, WEAK (outside the brief's categories): prep:371 should assert the machine card shows the folder answer's `bench_refusal`; prep:353 never asserts the proposed cells; prep:~1407 sleeps 300 ms before asserting (wait on state); bench:135 asserts only `last looked` + a clock, not B4's second clock (`measured HH:MM:SS`).
- OWED not added: GPU wanted vs available on the run card (gpu.md G2) — no file in W1 opens Task setup on a record with GPUs; 2.62.0 *Changed since prepared* (low, words are tier 1's).

## 4. Checks

- `pyflakes` on the five files: clean (0 messages).
- `pytest --collect-only`: 34 tests collected (was 36: two retired), 0 errors.
- No e2e run (by the brief).
