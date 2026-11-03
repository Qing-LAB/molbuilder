# fix-S2 — SIESTA optimization / stopped / road / sequence / keyword / wall / GPU

Read whole: S2.md, verified.md, fix_brief.md, testing.md § 3b and § 5, the seven files, and the contract
sentences cited below (running-a-job § 1, § 3.1–3.6, § 4.2; job-system § 5.4, *A stage launched again*,
*The next step*, *Every refusal*, § 6.0 *A run here, waited on* / *A benchmark's walk* / *A sequence of groups* /
*The walk*, § 6.0a; checkpointing § 9, § 10.2; parse.md § 3, § 5a, § 5c, § 5d.1b; results.md § 3a;
transport.md § 6.1 *The same engine input*; generator.md *The shelves*). Code paths read: runwrap banner and dry run
(2569-2581, 1060-1100), the argv capture (483, 2786), wall lines (2843, 2895); submit `_go`, `_counts_of`,
the sequence walk (3405-3430); runs.concluded / `_ended_written`; runstatus `_steps` / `_one_run_status`;
record `_launch_record` / `_instruments`; scf_timing_rows `TIMING_READERS`; checkpoint `save_before`, `Repo.states`;
commands `launch_save_note`.

## 1. Per test touched

| test (file:line after edit) | action | contract § | what it now guards; one change that turns it red |
|---|---|---|---|
| `test_a_run_here_is_told_its_cards_wall_and_its_engine_stops_before_it` (wall:94) | **fixed** (was failing) | running-a-job § 3.6 *What you see of the wall*; § 1 (labels padded to the longest) | The banner line is read by label and value with whitespace allowed: `^\s*Wall\s*: 20s from the job's start; SIESTA stops 600s before it \(running-a-job\.md 3\.6\)`. **Also fixed a vacuous check**: `"wall: " in wrap` was satisfied by the *stopped at its wall:* line, so the session log's `wall: Ns left of 20s; SIESTA stops 600s before it (…)` line was never actually checked. It is now matched whole. Red if: `submit._go` stops exporting `MB_WALL_S` (the banner then says *none stated*, with no wall line). |
| `test_a_launch_naming_no_stage_sends_the_one_prepared_and_not_launched` (road, old :197) | **retired** | — | It only checked that a launch record exists. :85 (`run_answer` finished, record, `02_medium/run-0`) and :255 (continues from run-2) already cover that. |
| `test_launched_again_cold_it_takes_nothing_of_its_own` (road:234) | docstring (NO GOAL) | job-system § 5.4 *A stage launched again* (`--cold`) | A cold launch again that copies restart files in. Red if: `--cold` is ignored by `continuation` (a `.continued-from` gets written). |
| `test_every_name_the_card_gives_a_stage_is_on_disk` (road:336) | docstring (failure stated) | job-contracts § 2.2; project-layout § 5 | The card names a file the road writes under another name. |
| `test_the_run_script_runs_the_stated_ranks_and_threads` (road:354) | docstring (NO GOAL) + section header citation | running-a-job § 3.1, § 3.2, § 3.4 | The dry run names the wrong source for, or a different value of, a stated count. Red if: the baked rung is dropped from `_np_source`. |
| `test_a_flag_on_the_run_script_still_overrides_and_says_so` (road:371) | docstring (NO GOAL) | running-a-job § 3.1, § 3.4 | `-np` losing to the baked value. |
| `test_a_benchmarks_per_trial_bound_is_the_walks` (road:424) | docstring (WRONG: "five seconds … floor") | job-system § 6.0 *A benchmark's walk*; generator.md *The shelves* (`--trial-timeout MINUTES`) | Now says: `--trial-timeout 5` means 5 minutes, i.e. 300 s, kill after 30 s, nothing derived. The assertions were already right. |
| `test_a_run_here_is_shown_and_asked_and_written_down` (road:68) | **added** OWED 2.63.0 | running-a-job § 3.4 ("`jobset launch` hands the counts, `-np` and `-omp`, and nothing else"); § 1 example `argv: H2_01_coarse.run.sh -np 1 -omp 1` | Reads the `argv:` line of layered run-0's session log, found through `run_of`. Its words must be `[H2_01_coarse.run.sh, -np, 1, -omp, 1]` and must equal `launch_record(run)["command"]` without `bash`. Red if: the line goes back to being logged after the arguments loop (`$*`, which is empty by then). |
| module docstring (road) | citations | — | `architecture § 5.2` → running-a-job § 3.1, § 3.2, § 3.4. Archived `plan § 5y` → testing.md § 0 tier 3. |
| module + `_calculation` docstrings (seq:5, :41) | stale ("three stages … groups 1 2 3") | config/stages.py `publishable` | Now: two stages, coarse then medium, in groups 1 and 2. |
| `test_each_run_records_the_sequence_it_rode` (seq:156) | **fixed** (the claim was false) | job-system § 6.0 *The walk* ("run here, `mode: direct` and the walk's line"); results.md § 3a; parse.md § 5d.1b `run-json` row | Now reads through the Results tab's door, `runs.folder_answer(run)["record"]["computation"]["launch"]` (sequence, mode, job_id, command), instead of `run.json` directly. Red if: `record._launch_record` drops `sequence` from its keys. |
| `test_the_sequence_record_and_the_ledger_say_it` (seq:188) | **added** OWED 2.62.1 | checkpointing § 9 *The note an act writes* (`… · before launch task group 2 (medium), in job 4817 (sequence H2_sequence-01_coarse-02_medium)`); § 10.2 `Repo.states` | For each group, exactly one saved state's note ends with `· before launch task group N (<members>), in job <the ledger's outer launched job> (sequence <seq stem>)`. Red if: `launch_save_note` is called without `inside`, or the walk stops typing `--in-job`. |
| `test_jobs_status_lists_the_sequence_and_a_stop_ends_it` (seq:267) | **added** OWED 2.61.0 (positive checks beside the absence-only one) | job-system § 6.0a `jobs stop` ("The run then reads *failed — stopped before its end* by its own files"); § 6.0 ("a sequence that ended before a group leaves that group as it was"); running-a-job § 4.2 `pending` | Read through `jobset_status`: the first group's stage reads `failed` with detail starting "stopped before its end"; the last group's reads `pending` with no launch record. `reached_last` is asserted false first. Red if: the monitor's closing record is no longer read as the job's end (the run reads `running`). |
| `test_a_run_that_failed_is_never_built_on` (stopped:197) | citation | job-system § 5.4; *Every refusal* R-L11 | Was "L2". |
| `capped` fixture (opt:51) | **fixed** (vacuous: `first.name == "run-0"` cannot fail, since prep opened run-0) | job-system § 6.0 *A run here, waited on* ("exits 0 when every member finished"); running-a-job § 3.5 row 2 | Asserts the first launch exits 0. Red if: a run that finished unconverged is read as failed by the launch. |
| `test_the_registry_reads_the_output_as_siestas` (opt:329) | docstring (NO GOAL); **kept, not retired** | parse.md § 3 (`detect`: one parser, none or two refused) | Guards against a real SIESTA output being claimed by no parser or by two. Kept because S2.md made retiring it conditional on a mutant ("retire once a mutant of the registry route shows :166 red"), and no such mutant was run. testing.md § 3b says subsumption needs a mutant, not an argument. |
| `test_a_relaxations_step_boundaries_are_not_iterations` (opt:382) | **added** OWED 2.64.0 | parse.md § 5c (*A run's timing file is chosen by one door*: a SIESTA `.out` run "is timed by that tee alone … since its progress log is prep's seed"); § 5d.1b `instruments` row | `record_of(run)["computation"]["time"]` must have (`s_per_iter`, `iters_measured`) equal to `scf_timing_metrics` of the run's own `.scf-timing.log`. That figure is itself checked against an independent count. Red if: `timing_file` picks the seeded `.molwatch.log` (no rate, 0 iterations). |
| `test_the_output_keeps_its_frames_and_the_history_matches_in_order` (opt:541) + section header | citations | parse.md § 5a; header § 2a/§ 5.3/§ 6 → § 3, § 5c, § 5d | — |
| three tests in test_siesta_keyword_smoke_e2e.py (:126, :179, :215) | docstrings (NO GOAL; the stale "foreign deck" motive replaced) | transport.md § 6.1 *The same engine input* (decks "read as SIESTA reads them, by the one fdf reader"; geometry "in Å as SIESTA places them — the coordinates' format, `LatticeConstant` … folded in by the reader (`coords_ang`, `cell_ang`)") | Science tests: assertions unchanged. |

### OWED items not added, or added only in part
- **2.61.0, "run.json gains `ended`": not asserted. By the code, the value is not deterministic.** `runs.concluded` writes `ended` only while the run still reads live from its own files. `jobs stop` sends SIGTERM to the process group, which reaches the monitor, and the monitor then writes its closing record. `run_state_of` reads `failed` from that record ("no ending in its output and no exit recorded") before it asks the process, so `ended` is never written. The contract only says the run reads *failed — stopped before its end*, which is what the test now checks.
- **S2.md's "the group never reached reads *ready to launch*" is wrong by the contract.** In job-system *The states* and *The next step*, a prepared stage whose upstream's newest run failed reads *waits for*. The check is written as `pending` with no launch record instead ("left as it was").
- **2.65.1, the engine's own start stop**: skipped, as instructed (it needs a new engine run).
- **S2 OWED 5, the Results load's engine and the run heading**: not in my assigned list. Listed only.
- S2.md's other OWED items, "jobs status says the job once" and OWED 7: not assigned.

## 2. SHARED (not made)
- running-a-job § 3.6 now reads label-tolerant. job-contracts § 4.4 l.~1283 still shows the old fixed padding, according to S2.md. That is a doc and not mine; check it is part of the doc sweep.
- None of these revisions needed a helper change.

## 3. F9 (left as is)
- wall test 2 (`test_a_warm_file_written_as_the_job_ended…`): forged `.XV` mtime (`os.utime`), as instructed.
- stopped `test_the_run_panel_says_what_ran_and_why_it_stopped`: planted `sessionStorage['molbuilder.current_dir']`, scope set by `setShared` instead of sidebar clicks, the `file_picker_roots` patch. Also its Setup *default* column is expected from the record the page renders rather than the catalogue's `max_scf_iter` default (WEAK; fix together with the walk).
- stopped `test_the_viewer_says_why…`: `Capabilities.file_picker_roots` subclass patch. Same for opt `test_the_run_reads_its_own_deck` (monkeypatch of `file_picker_roots`).
- road `test_the_ranks_a_scheduler_grants…`: the SLURM grant is typed into the environment (`real_runs.py:413`).
- road `test_a_benchmarks_per_trial_bound_is_the_walks`: NOT E2E. The rendering needs no engine (tier 1: `launch bench --mode submit` against the refusing `sbatch`).
- road :255 plan header words "restart files it declares"; seq :220 "R-P23" in the refusal; stopped :178-180 and :209 next-step words: text-pinning on copy that is still right.
- Table-row merges and duplicates, each right on its own: stopped :209 into :153's door assertion. `capped`'s relaunch duplicates the road's warm relaunch: moving "CLEAN START" / "CONTINUES (engine will load H2.XV…)" onto layered run-0/run-1 would save one engine run. History pairing is checked twice (opt :443/:518 against `support/road.py:1023-1031`). One-start count appears in both the stopped and wall files.
- Helper duplication, moved nowhere: seq `_calculation` beside `real_runs._described`, and its poll loop beside `support.road.until`. stopped fixture beside `_road.live_siesta`, and `_set_item` beside `set_template_value`. keyword `_siesta_binary` beside `_road.env_bin`.
- Reader notes, own substring or regex where a door exists, assertions right: opt `opt` fixture `">> End of run"`; `_deck_value` and the deck split (`parse.fdf.read_fdf`); wall :126-127 raw `.out` substrings (`ending_of`); keyword smoke's SIESTA stdout substring and `outcell:` regex.

## 4. Checks
- `python -m pyflakes <7 files>`: clean (no output, rc 0).
- `pytest --collect-only` (isolated HOME and XDG): **61 tests collected**. Before: 62; one retired.
- Nothing was run beyond these (no e2e, engine, browser or server).
