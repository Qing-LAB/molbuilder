# fix-T — `tests/test_transport_on_a_real_junction_e2e.py`

Read whole: the brief, T1.md, T2.md, verified.md, testing.md § 3b and § 5, the test file (all ~5300 lines).
Contract read for every expectation added: transport.md § 1.1 (*The first prep*), § 2a.10, § 2a.11 (*Launched
again*), § 2a.16 (*Describing it*, *Extending it*, *On the page*), § 3, § 3.1, § 3.8.2, § 3.8.2a, § 6.3;
chemistry-correctness ES5, ES6; job-system § 5.1, § 5.3; runs-of-points § 3 (*What the run continues from*);
run-reports § 2.4; parse.md § 2b (`error_message`), § 5d.3a; web-api `/api/transport/describe`,
`/api/transport/describe_attempt`; project-layout § 1.0. Code read for each: `transport/describe.py` (whole, and its
2.65.4 diff), `transport/reuse.py` (`Extension`, `refusal`, `Reuse.lines`, the 2.65.4 `write_reusing`),
`jobset/_cli.py` (`_described_by_its_kind`, `describe_cmd`, `_resolve_bundle`), `jobset/commands.py`,
`identity.words_for`, `describe.write_description`, `web/blueprints/transport.py` (describe route),
`validation/chemistry.py` (ES5/ES6), `template.template_with_values` / `reads`, `support/road.set_by_the_person`,
`support/real_runs` (`stage_status`, `point_row`, `RealRun`), `jobset/submit.py` (`_points_walk`'s run source,
`_continues_run`, `_launch_again`, `_record_launch`), `runrecord` (`write_launch`, `RunRef`),
`jobset/runstatus._sweep_status`, `continuation.point_rows`, `runpoints.rows_of`, `parse/dirs/job.py` (stopped
detail), `siesta_reader._fatal` / SCF_NOT_CONV hold, `siesta_grammar.STOP_MEANINGS` / `stop_line`,
`parse/dirs/setup.py`, `script_emit.declarations` / `read_parameters_fence`, `runs.Run.session_log`.

## 1. Per test touched

| test (line after edit) | action | contract § | guards now / one code change that turns it red |
|---|---|---|---|
| fixture `junction` (303; relaxation 321–327) | fixed (wrong record) | template.md § 6.6 (`set_by_the_person`'s source `person`) | the relaxation's six values are now recorded as the person's, as every other calculation in the file; `template_with_values(cfg)` with no `sources` wrote every item `default` (template.py docstring: "every other item is `default`"). Unused imports (`SiestaConfig`, `config_from_template`, `template_with_values`, `template_path`, module `dataclasses`) removed. Also records F's describe answer (`from_pair.said["init"]`, 492) — no new run |
| `_NO_OTHER_VALUE_HERE["unpaired_electrons"]` (1494) | fixed (WRONG DETAIL) | chemistry-correctness ES5, ES6; transport § 3.1 spin note | reason now true: the probe is restricted, which holds 0 alone (ES5 refuses any other); a stated count is written as stated only under unrestricted (ES6). Exemption kept |
| `test_describe_from_a_citation_writes_the_description_alone` (1046) | **added** (OWED 2.65.3) | job-system § 5.1 (*writes the description — the template, task.json, job-request.json*); transport § 1.1 *The first prep* (*"its first prep composes it"*); § 3.1 (a pair's pseudopotentials copied by prep); § 3.8.2a; job-system § 5.3 (`--bundle` every time) | reads `run.said["init"]` (T, cited run) and `from_pair.said["init"]` (F, pair) as the CLI printed them: headline names the calculation and its citation, exactly the three statement files listed, first prep line ends `--bundle P/transport/<name>`; and the tab's T2 `written` is the same three. Red if describe writes the junction/pseudos before prep, or the next line drops `--bundle` |
| `test_each_scf_stage_writes_its_own_mixer_and_the_transmission_none` (1588) | **added assertion** (OWED 2.64.0) | parse.md § 5d.3a (*"A row per catalogue item the rung reads … by the one rule `template.reads`"*) | each T transmission point's session log (`runs.run_of(..).session_log`, read by `script_emit.read_parameters_fence`) lists only items `template.reads` gives the transmission — `transmission_n_points` in, `mixing_weight` out. Red if the TBtrans step's fence lists the SCF stage's items |
| `test_each_leads_measurements_reach_the_card_under_its_name` (1812) | docstring (NO GOAL) | transport § 3.8.1; web-api describe_attempt `findings` | failure named: a seam verdict with no lead named |
| `test_a_sweep_is_one_run_its_points_inside` (1851) | docstring (wrong) | project-layout § 1.0 (*the progress log prep seeds … goes into the run it opens*) | "which writes it" was false (SIESTA never writes it; prep seeds it) |
| `test_launched_again_warm_with_every_point_done_it_is_refused` (1927) | docstring (NO GOAL) | transport § 2a.11 *Launched again*; runs-of-points § 3 op 17 | failure named: an empty run made the newest, gathered by the transmission |
| `test_launched_cold_every_point_runs_again` (1941) | docstring + **added assertion** (OWED 2.62.0) | § 2a.11 (*"Cold … nothing is taken over"*); runs-of-points § 3 *What the run continues from* (*"Points that start over take nothing: … not said to continue from the last"*) | `stage_status(T,"device").launch["continued_from"] is None` on cold run-1. Red if a cold run of points records a continuation |
| `test_launched_again_warm_it_takes_the_done_points_over` (2020) | **added assertion** (OWED 2.62.0) + docstring | runs-of-points § 3 (*"That is the run's `.continued-from`, its launch record's `continued_from`"*) | the warm run's launch record `continued_from == f"{ended.dir}/{ended.attempt}"` (the stopped run it took 0 V from). Red if a run that takes a point over records no source (`_continues_run` not called) |
| `test_a_failed_point_is_said_once_per_reason_and_its_round_by_the_milestone` (2513) | **added assertion** (OWED 2.65.1) + docstring | transport § 6.3 (*"the result says why in the engine's words … `status`, the Results tab and the transport record's failed points say it"*, its TBtrans row) | on the `refused` copy, every point's status `detail` holds TBtrans's *gamma calculation while you request transmission k-points* and *what it may mean: … leave tbt_k_grid blank or 1 1 1*. Red if the die reason is dropped from the point's state |
| `test_a_voltage_that_does_not_finish_ends_its_frame_and_the_run_once` (2613) | **added assertion** (OWED 2.65.1) + docstring | parse.md § 2b `error_message` (*"the `SCF_NOT_CONV:` line when it came before the stop — it names the cause"*); transport § 6.3 | on the `aborted` copy, the 0 V point's status `detail` holds `SCF_NOT_CONV:` … `(required)`. Red if the held line is not promoted at the stop |
| `test_a_stopped_walk_says_no_finish_and_its_warm_launch_one` (2706) | **added assertion** (WEAK) | run-reports § 2.4 (*"a walk stopped by the person … sends no `finish`, and the run door reads the run failed"*) | the docstring's claim now asserted: `walk_ended.state == "failed"` |
| `test_the_data_file_holds_every_point_beside_its_frames_definition` (2805) | docstring (NO GOAL) | § 2a.12 | failure named |
| `test_the_data_file_holds_every_runs_convergence_on_its_grid` (2935) | docstring (NO GOAL) + stale `(@4)` comment dropped (schema is `@8`, § 2a.12) | § 2a.12 | failure named |
| `test_the_data_file_holds_the_calculations_own_records` (3202) | stale `(@4)` dropped from docstring | § 2a.12 | — |
| `test_the_tab_cites_a_frame_set_and_says_what_each_frame_is` (3311) | docstring (NO GOAL) | § 3.8.1 | failure named |
| `test_a_selections_pdos_is_asked_at_its_frame_and_voltage` (3419) | docstring (NO GOAL) | results § 2.5 | failure named |
| fixture `reused` (4255) | rows added; stale comment fixed | — | `out["refused"]` + row *another engine* (`--reuses P/transport/T --engine pyscf --bundle P/transport/R5`); `refusals` + rows *its rename* (`swap_electrodes` flipped from T's), *no voltage* (`bias: []`), *a value the shared panel does not carry* (`shared: {mixing_weight}`); `out["left out"]`: two `plan: true` asks on the existing client (T's device/transmission tab values only; then `bias [0,0.2,0.4]`, no treatment). The comment "one voltage carries no treatment" (contradicting the body) rewritten. No engine run added |
| `test_a_folder_that_cannot_be_reused_is_refused_before_anything_is_written` (4627) | rows added + docstring (NO GOAL, WEAK) | § 2a.16 *Describing it* / *The engine is the finished calculation's* (*"another engine … is refused, naming both"*); job-system § 5.1 (addresses from the projects root) | *inside* now asserts `P/transport/T/inner is inside the finished calculation T` (2.65.3's address); new *another engine* row: `engine 'pyscf': T ran on siesta`, folder not made |
| `test_more_voltages_take_the_finished_points_over` (4873), `test_a_fitted_window_…` (4958), `test_a_transmission_that_runs_once_…` (5083) | docstring (NO GOAL) | § 2a.16 | failure named |
| `test_an_extension_keeps_what_its_finished_calculation_computed_once` (5110) | **rows + assertions added** (OWED 2.65.3) + docstring | § 2a.16 *Extending it* (*"its rename or its label changed is refused by name"*), § 2a.10 (*"an empty list is written by neither"*), § 3.8.2, *Describing it* (*"voltages typed with no treatment keep the finished calculation's"*), web-api (*"an extension's `bias`, `junction` or `swap_electrodes` left out keeps the finished calculation's"*) | 400 by name, nothing in `kept/`: rename (*the electrodes' rename (swap_electrodes) is fixed by T*), `bias: []` (*bias: state the voltages*), `mixing_weight` as shared (*'mixing_weight' is not a shared value*); plan with everything left out = every group copied in, nothing "not asked here"; plan with a voltage more and no treatment = device and transmission each `takes over 0 V, 0.2 V … ; walks 0.4 V`. Red if left-out voltages fall back to `[0.0]`, or the treatment is not T's |

Not touched (CHANGES WITH 2.65.4, yours): `test_the_transport_tab_describes_a_calculation_that_reuses_one` and
`test_a_frame_set_of_the_same_junction_takes_over_frame_0` (left whole, including its NO GOAL docstring — add the
failure line when you revise it: *a frame set's frame 0 not taken over, or its other frames copied from a junction
they are not*). Nothing added asserts an extension's describe writes the junction or pseudos.

## 2. SHARED

- None needed by these revisions.
- Note for the contract: the CLI describe headline wording (*Described transport calculation '…' … -- composes …*)
  is stated only in the 2.65.3 changelog row, not in transport.md § 3; the new test pins only its opening words and
  that it names the citation.

## 3. F9 (left, one line each)

- `_cited_on_the_tab` (1080): the junction reached by a planted `transport:panel` session note + reload, not the
  junction card's picker clicks (used at 1134, 1167, 5352).
- `_the_decks_a_person_sets` (1358): `task.json` written back by hand after the tab rows, not Task setup's Save.
- `test_what_a_deck_leaves_to_the_engine_is_read_as_siesta_reads_it`: an older-release deck cut by hand in memory;
  row 3 duplicates 1414's spacing row; whether an older molbuilder's device deck is still a case is the user's call.
- `test_the_geometry_is_compared_where_siesta_places_it`: NOT E2E (generic SIESTA geometry comparison) → tier 1.
- `test_a_sweep_is_one_run_its_points_inside` / `test_a_frame_sets_device_runs_a_point_per_frame_and_voltage`:
  presence/count halves could be one table row.
- `test_the_leads_greens_functions_are_the_rounds_one_pair`: walk-log substring parsing (round order is in the
  index's `started` stamps; emptying lines by the walk's own tag).
- `test_a_device_launched_warm_then_cold_keeps_the_seeds_density`: the launch-words regex (`copied …`, *took the
  inputs its kind gathered*); the ledger half is the substance.
- `test_a_vibration_of_the_relaxed_pair_is_judged_by_what_it_held`: the holds-more row could read the fixture's
  `prep freq` answer (the road) instead of calling `validate`.
- `test_an_export_reads_no_data_file_and_says_what_is_missing`, `test_a_scripts_parse_keeps_a_reading_…`: the shared
  run's digest moved away by hand (road: export `from_pair` before its `summarize`; `parse(fresh=True)` on a copy).
- `test_the_report_draws_the_junction_in_the_user_order`: presenter mounted by JS with a composed meta.
- `test_the_results_tab_says_a_devices_forces_…`, `test_the_results_tab_opens_a_transmission_folder_at_its_point`:
  `projects.setShared` instead of sidebar clicks; a synthetic `visibilitychange`.
- `test_the_tab_cites_a_frame_set_of_the_junction_it_extends`: planted `sessionStorage` + `_cited_on_the_tab`.
- `test_an_extension_keeps_…`: the manifest checked by its first line only (one manifest across describe, Task
  setup's Preview and prep — T2 OWED 4).

OWED not added (fails a brief condition, or needs a new engine run):
- 2.65.1 start-stop settings prepared as asked whose engine stop must be shown (a stated count under unrestricted,
  a pole energy under 20 poles): needs a new TranSIESTA run (e.g. a copy of S) — skipped.
- 2.65.2 `launch --gather` through `materialize.gather_into`: needs a new device/transmission launch — skipped.
- 2.65.4 an extension's first prep composing the record and copying `pseudos/`: yours with 2.65.4.
- 2.59.0 `/api/results/dir` at a transmission run of points' root and the Run panel's *the last to run*; 2.65.0 the
  Results heading's engine words on 4050's page; 2.62.0 both describe doors' mesh notes; 2.62.1 the run script's
  line after a failed point; 2.61.0 `run.json`'s `ended` on the stopped copy; the `--window` refusal and
  `/api/transport/fit`; the tab's describe of a new calculation with no name / on `pyscf`; a new calculation's
  left-out bias `[0.0]` on the tab (a plan answer carries no task to read it from) — not taken, outside the
  caller's priority list; each meets the conditions on runs already made and can be added the same way.
- Note on 2.65.1 stop words (added above): the reading itself landed in 2.25.2 (git `-S`), 2.65.1 changed no
  reader; added because 2.65.1's row names it as what a person sees (*shown by status, the Results tab and the
  record*), never e2e-run, and § 6.3 states the values. If the batch shows the point `detail` lacks TBtrans's words,
  that is a product finding against § 6.3, not a test to edit.

## 4. Checks

- `python -m pyflakes tests/test_transport_on_a_real_junction_e2e.py` — clean (0 messages).
- `pytest --collect-only -q` (sandboxed HOME/XDG) — **206 tests collected**, no errors (one new test function).
