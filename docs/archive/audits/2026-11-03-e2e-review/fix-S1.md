# fix-S1 — `tests/test_siesta_vibration_e2e.py` · `tests/test_siesta_relax_run_e2e.py`

Read before editing: S1.md whole, verified.md § S1, testing.md § 3b and § 5, both files whole, and the contract
sentence behind each expectation (running-a-job § 3.5, § 3.6; vibration § 2.2, § 5.5, § 5.9, § 5.11 §§ 5–8, § 6.2;
spectra § 2.1, § 3.2; runs-of-points § 1.3, § 3, § 4.4–4.6; run-reports § 2.4; parse § 5b.1; chemistry-correctness
ES3/ES7), plus the code paths relied on (`runwrap._launch_again` / `_again_line`, `continuation.point_rows` /
`never_reached` / `points_state`, `runpoints.state_word`, `parse/dirs/job.py` stopped-detail composition,
`siesta_reader` stop line, `submit._record_launch` / `_continues_run`, `runrecord.RunRef`, `optical_data` reader,
the `_spectra_inspector.html` bar). No e2e, engine, browser or jobset command run.

## 1. Per test touched

| test (file:line after edit) | action | contract § | what it now guards; one code change that turns it red |
|---|---|---|---|
| `unticked` fixture (vib:91) | docstring | vibration § 5.9 | stale reason "a launched group is fixed (stages § 2)" replaced by § 5.9's "a later stage builds on the first, so it never shares that stage's job" |
| `_the_optical_record` (vib:220, used by 745 and 1607) | fixed (WEAK) | vibration § 5.11 § 8 (`fermi_level_ev(frame, spin)`) | every finished frame's Fermi level is finite, read through `xarray` as a script reads it, and the reader's per-frame value is not `None`. Red: the record step writing a fill (NaN) for a frame's Ef (the 2.58.0 defect). Note: the old `len(f["fermi_level_ev"])` raised `TypeError` on such a frame (the reader returns `None` for non-finite), so it failed as an error rather than an assertion; it is now an explicit check |
| `test_unticked_…relaxed_geometry` (vib:444) | docstring | vibration § 2.2, § 5.11 | adds the silent failure (force constants at the input geometry; § 2.2 "the one error that looks like a result"). Body unchanged |
| `test_the_results_tab_shows_…free_atoms_infrared` (vib:981; assert at :1097) | fixed (STALE ROAD) | spectra § 3.2 ("No set of atoms is chosen anywhere — not on the page"), § 2.1 (a control shows only where the result has its choice) | the removed `#ir-atoms` count-0 (the id has not existed since 2.49.0) is replaced by: the bar's shown `select`/`input` controls, by their visible labels, are exactly `["Light along", "Charges"]`. Red: any atom-set control (or a Raman input) shown in `#view-bar` for this result |
| `test_a_displacement_sweep_…` (vib:1431; checks at :1487) | added (OWED 1, 2.58.0) + docstring | vibration § 6.2 (`enabled` true when the `relax` call "ran, or was taken over from the first stage of a displacement sweep", `n_steps` from that record), § 5.5 (later stage judged by "the tolerance the first stage's run relaxed to, which its relaxation record carries"; `reference_force_criterion_ev_ang` records it) | `freq_half`'s result: `relaxation.enabled is True`, `n_steps` = `freq`'s relax record's, `engine_metadata.reference_force_criterion_ev_ang` = that record's `force_tolerance_ev_ang`. Red: the finish writing `enabled` false for a taken-over relaxation, or reading the criterion from anywhere but the carried record (e.g. `None`, or the catalogue default). Limit: both stages use 0.01 eV/Å, so a later stage using its *own* stage tolerance is not told apart from this run |
| `test_launched_again_warm_after_a_stop_…` (vib:1607; checks at :1649, :1681) | added (OWED 4: 2.59.0/2.61.0; OWED 5: 2.62.0) + docstring | run-reports § 2.4 ("a walk stopped by the person … sends no `finish`, and the run door reads the run failed"); runs-of-points § 4.5 / § 1.3 (*not run*, never failed); runs-of-points § 3, *What the run continues from* ("only where it takes something from it … the run's `.continued-from`, its launch record's `continued_from`"; "Cold, nothing is taken over") | on `stopped`'s kept status: run-1 `failed`; no `finish` in run-1's `notified.json`; every call with no start stamp in run-1's index reads `not run` by `runpoints.state_word` (non-empty set asserted as premise). On run-2: `status.launch["continued_from"] == "01_freq/run-1"` and `read_continued_from(run-2) == run-1`; run-1 (cold) `continued_from is None`. Red: `points_state` reading a killed walk as `running`, or `never_reached` not applied; a warm run of points' launch record writing no `continued_from`; a cold launch writing one |
| `test_a_failed_part_…once_per_reason` (vib:1753; checks at :1807, :1811) | fixed (STALE ROAD) + added (OWED 7: 2.62.1/2.65.1) | running-a-job § 3.6 (*stopped at its wall* in place of *what a launch again will do*; the *What you see of the wall* table: state *failed* — `stopped before its end: OUT_OF_TIME: Time is up.`); § 3.5 row *a point of a run of points*; vibration § 5.11 § 6 (an optical call runs again) | the old `"launch this stage again to continue" not in walled_log` could never fail (those words exist only in `pyscf/relax_policy.py`). It now asserts that the session-log line after *stopped at its wall:* starts with "this point failed: a launch again takes the run's points done over and " and its first clause says "again from its start". Each walled call, as `jobset status` reads it: `state == "failed"` and detail starts `stopped before its end: ` and ends `OUT_OF_TIME: Time is up.`. Red: the wall line composed without the member flag (a single run's *launch the stage again to continue warm*), or the optical relaunch said as *continues*; the stop read as an SCF failure, or the engine's line dropped from the detail. (Startswith/endswith instead of strict equality: a possible `node 0:` prefix under MPI was not checked against SIESTA's source) |
| `test_a_finished_relaxation_answers_its_record` (relax:120) | docstring (NO GOAL) | parse § 5b.1 | opens with the failure (a record vouching for coordinates it is not about); body unchanged |
| `test_both_doors_answer_one_record_of_the_run` (relax:176) | docstring (NO GOAL) | parse § 5b, § 5b.1 ("the two doors answer one record of one file") | failure line added |
| `test_a_matching_record_…` (relax:246) | docstring: citation and goal | vibration § 2.2 rows *ticked — present, for these coordinates*, *any — edited by hand* | rows now cited by their content, not by a number |
| `test_a_record_looser_…` (relax:274) | docstring: citation fixed | § 2.2 rows *ticked — present…* and *unticked — anything else* (it said "rows 3 and 5"; those are now rows 4 and 6) | — |
| `test_a_record_for_another_geometry_…` (relax:296) | docstring | § 2.2 row *ticked — present, for a different geometry* | — |
| `test_a_record_at_another_level_of_theory_warns` (relax:309) | docstring | § 2.2 row *ticked — present, for these coordinates*; parse § 5b | — |
| `test_a_record_at_another_charge_warns` (relax:345) | docstring: stale premise removed | ES7 (chemistry-correctness § 2a.2), § 2.2 | the stale "so the parity rule does not answer first" is dropped (since 2.65.1, ES3 says parity is only information) |
| `test_a_good_record_under_an_unticked_box_…` (relax:361) | docstring: citation fixed | § 2.2 row *unticked — present, … within this calculation's tolerance* (it said "row 4"; now row 5) | — |
| `test_no_record_under_a_ticked_box_…` (relax:377) | docstring | § 2.2 row *ticked — absent* | — |
| `test_every_door_asked_of_the_run_says_its_order` (relax:578) | docstring added (it had none) | overview § 2.2a (I1, I5, I8), parse § 5.3, transport § 3.1 | the banner's failure (a run in the engine's order shown, saved or converted as the user order) as its first line |

Nothing was retired: the report found no RETIRE.

## 2. SHARED (not made)

- `docs/engines/vibration.md` § 5.11 § 14 row 1 says this e2e catches "the run's state read *queued* before the first call". No assertion holds that, and it can be seen only by chance, on `stopped`'s background launch. Either § 14 is corrected or a race-free observation is designed (S1.md OWED 10).
- `support.real_runs.told` reads the monitor log's `[NOTIFY]` lines by regex, and `_launched_here` reads the ledger raw. Neither has a reader door (S1.md helper notes). These are reader gaps.
- Side finding, not verified (S1.md): the `compose.py` refusal of a vibration's `run-0/` ("a container … cite one of the runs below it") may contradict "a vibration's own folder is not cited", since a run below it (`relax/`) would be accepted.

## 3. F9 (left, one line each)

- vib 981, 1228, 1285: `sessionStorage['molbuilder.current_dir']` is planted. It should be the sidebar `.ps-entry[data-path=…]` click.
- vib 981: `page.route` hold, and the `pageBusy.claim` wrapper planted with `page.evaluate`.
- relax 246: `edited_by_hand` is stamped by hand. It should go through MolView's Metadata-page save route. relax 377: `apply_info_dict(None)` should be the road CO2, which carries no record.
- relax 246–377: the 7 gate tests should become one parametrized table mirroring § 2.2's rows. Text-pinning to go with that: `_ev` copies `sidecar._ev`'s format, and `_record_findings` filters by the code's opening words.
- vib 723 (×2): NOT E2E, a container's citation refusal, to move to tier 1 on a prepared vibration. Its WEAK assertion (`CITATION_CONDITION` ends every refusal) should become the refusal's own clause.
- vib 444: the relax/fc deck asserts and the order record are prep's product. The fc-deck lines duplicate `test_siesta_vibration_deck.py:49-70`. The one-stage `summarize` refusal should be a tier-1 row.
- relax 578: the `deck` row is prep's product, and should become a tier-1 row.
- relax 120: `_final_geometry` / `_moves` are the test's own regex over `.out`, not the parser (NOT THROUGH ITS PARSER). Moving to the `.XV` reader and the registry trajectory changes the oracle and precision, which I cannot check without a run.
- vib 1431 text-pinning: `kind.parse(...).schema == "fc-sweep/v1"` (a parser tag stated nowhere in the contract); vib 444 "a sweep compares two or more" (the code's sentence).
- OWED 6 (2.59.0 ladder pick, 2.65.0 run heading *SIESTA vibration*): it sits on the ladder test's page. But the picked run's menu and group heading would mean writing browser steps I cannot run, on a test whose road F9 rewrites. Add it with that rewrite.

Skipped OWED, each needing a new run or input, or outside 2.57–2.65.3:
- OWED 2: the summary's two-geometry refusal and its way out (needs `freq_half` launched again `--cold`, a new run). `--from` with `--cold` refused needs no run, but belongs in a tier-1 row.
- OWED 3: Fermi levels under `Spin.Fix` (`Ef_up`/`Ef_dn`), which needs a fixed-spin optical run.
- OWED 8: the waited launch exiting 1 with *FAILED* first (job-system § 6.0). That rule dates from 2026-10-13 and was not changed in 2.57–2.65.3.
- OWED 9: the 2.57.0 labelled species (`element` the symbol, never the label), which needs a labelled-species input on a run.

## 4. Checks

- `pyflakes tests/test_siesta_vibration_e2e.py tests/test_siesta_relax_run_e2e.py`: clean (0 messages).
- `pytest --collect-only` (fake HOME/XDG): **32 tests collected**, no errors (vibration 13 functions, one ×2; relax 11, one ×8). The count is unchanged.
- Diff: vibration +126/−21, relax +74/−24. No other file touched.
