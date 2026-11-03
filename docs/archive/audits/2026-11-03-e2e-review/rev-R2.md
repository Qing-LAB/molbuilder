# rev-R2: review of the revised SIESTA/PySCF/monitor/notebook e2e tests

Scope: the 14 files named. Each was read whole, and every changed line checked with `git diff HEAD`. Every new
assertion was traced to the code that produces it: runwrap banner/argv/wall/again lines, `_launch_again`,
`submit._go`/`_counts_of`/`_record_launch`/`_points_walk`/the sequence walk, `checkpoint.save_before`/`Repo.states`,
`commands.launch_save_note`, `parse/dirs/job._build_status`/`run_status`, `continuation.point_rows`/`never_reached`/
`points_state`, `runpoints.state_word`, `record._launch_record`/`_instruments`/`_ending`/`_merge`,
`scf_timing_rows`, `siesta_reader` + `siesta_grammar.stop_line`, `_run_ending._pyscf_ending`/`say`,
`relax_policy`, `vibrational_analysis._stationarity`, `siesta_vibration.result_of`/`_optical_payload`,
`optical_data`, `monitor.run_monitor`/`UtilSample`/`_pid_alive`, `jupyter.read_runtime`/`config_dir`, and the
`_spectra_inspector.html` bar with its CSS `[hidden]` guards.

Nothing was run.

## Findings, ranked

### Will fail on correct code

1. **tests/test_sequence_of_groups_e2e.py:294-296 (new). Race: the first group may not have started when the stop
   lands.**
   - What it asserts: after `jobs stop`, the first group's stage reads `failed`, with a detail starting "stopped before
     its end".
   - The fixture (:98-123) runs the in-process reads one after another: status, alone, bare, prep_held, `jobs status`,
     `ever_launched`. Then it sends `jobs stop` at once.
   - Nothing waits for group 1's run to start. Before group 1's `run.json` and session log exist, the walk's first
     step still has to do all of this:
     - enter conda (`_molbuilder_env_entry`);
     - start `python -m molbuilder.jobset launch`;
     - plan, and save a checkpoint (`submit.py:1708-1724`);
     - Popen the run script.
     That is several seconds, comparable to the reads.
   - If the stop lands first, `_build_status` (`parse/dirs/job.py:619-660`) reads:
     - `pending` / "prepared, not launched" when there is no `run.json`;
     - `queued`, or "never started: …" (`_job_ended`, started=False) when the record is there but no session log.
     Either way the assertion fails.
   - The contract's line (job-system § 6.0a: "The run then reads *failed — stopped before its end*") assumes a run
     that was running.
   - Fix: in the fixture, before the stop, wait (bounded, `support.road.until`) until group 1's run reads `running`
     through the status door, as the vibration `stopped` fixture does.

### Vacuous or weak

2. **tests/test_siesta_vibration_e2e.py:1493-1498 (new). The new criterion check cannot catch the mutation the
   docstring names, and `enabled` adds nothing.**
   - Criterion: `reference_force_criterion_ev_ang == record["force_tolerance_ev_ang"]` survives "judging its
     geometry by a criterion other than the first stage's relaxation record" (named at :1449-1452). `freq_half`
     overrides only `fc_displacement`, so its own deck criterion (`rec["force_criterion_ev_ang"]`) and the carried
     record's tolerance are both 0.01. `siesta_vibration.result_of`:478-484 yields the same number either way. fix-S1
     admits this limit, but the docstring still claims the mutation.
   - `enabled is True` at :1493 is already implied by the existing `phase_relaxation == "complete"` at :1484-1486.
     Both come from `ladder_relaxation` being present (`vibrational_analysis.py:214` and `:353-354`).
   - Fix: drop the criterion mutation from the docstring. Or give `freq_half` its own `relax_force_tol` override, if
     § 5.9 allows a stage to vary it (unverified).

### Retirements that leave contracted behaviour unguarded

3. **tests/test_monitor_watches_a_live_run_e2e.py, old :443 `test_a_monitor_missing_its_companion_still_monitors`:
   retired on a false premise.**
   - P.md:118 and fix-P's SHARED item say no contract names the case.
   - It is stated. `docs/configuration.md:354` ("What a missing module costs"): "`config_dir` | imported only when the
     channels are read; `monitor.load_channels` catches the import error, logs *reports off*, and the monitor keeps
     monitoring".
   - The code path is `monitor.py:1812`. No other test pins "reports off" for this case (grep over `tests/`).
   - Fix: the user rules one of two ways. Delete the branch together with the configuration.md row, or keep a
     tier-1 test of it. fix-P's "have the contract state the case" is moot.

4. **tests/test_monitor_start_is_logged_e2e.py: two tests retired by the brief's decision. The rules they pinned are
   now unpinned anywhere.** These are `test_a_monitor_that_dies_loading_says_why` and
   `test_an_ending_the_bundle_cannot_read_is_said_once`. The contract rules they pinned:
   - run-reports § 2.6 (:485-494): "A monitor that died loading says so: `starting`, then one `ERROR` line … and no
     `started`";
   - configuration.md "What a missing module costs", row 2;
   - running-a-job § 3.5, *the one ask* (`runwrap._ENDING_UNREADABLE`, said once).

   A grep for "did not load" and `_ENDING_UNREADABLE` in `tests/` finds nothing. No fix is due now; this is listed
   so the user knows these rules are unguarded (a tier-1 home, or F9).

### Wrong expectation or docstring

5. **tests/test_the_road_on_real_runs_e2e.py:424-436: the docstring claims a log check the test does not make.**
   - The rewritten docstring says the walk "and its log names the bound".
   - The assertions read the generated script, `bench-group.run.sh`, as text. `per-trial-bound=300s` is in its echo
     line (`submit.py:2326`).
   - The walk ran ([ran]), and its `bench-group.log` exists (asserted at :419-420).
   - Fix: read `bench-group.log` for `per-trial-bound=300s` (the outcome), or drop "its log names the bound".

### Product risk the new PySCF assertions will expose (unverified)

6. **tests/test_pyscf_relaxation_outcome_e2e.py:125-162, called at :203, :239 and :414. The detail may not quote the
   exception line if buffered stdout lands after the traceback.**
   - The assertions need the status, folder and record detail to quote PySCF's exception line.
   - `_run_ending._pyscf_ending` (:240-245) takes the last non-empty line that starts at column 0 of `.pyscf.log`.
   - The deck writes `print()` to a block-buffered stdout that is redirected `2>&1` into that file.
   - Python flushes pending stdout at shutdown, after `sys.__excepthook__` has written the traceback. Neither
     runwrap nor the decks set `PYTHONUNBUFFERED` or `-u` (grep).
   - So a deck `print()` made after PySCF's last logger flush and before the raise would land after the exception
     line. Then `words in detail` fails.
   - Low risk for halt and continue: the raise follows geomeTRIC's flushed logging directly. Unchecked for the wall
     vibration's `stop_if_past_wall` site (`vibration_emitters.py:985`, `vibration_deck.py:352`).
   - If it fails, the defect is in the product, not the test.

### Docs

7. **molbuilder/jupyter.py:246-251: the `read_runtime` docstring contradicts jupyter.md § 2.**
   - It says the token "is read server-side and handed to the page". jupyter.md § 2 (around :79-80) says molbuilder
     "never hands the browser the token — not in its pages, not in the frame's address".
   - This is the premise the new notebook assertion at :213-224 pins.
   - Fix: "presented by the pass-through, never handed to the page".

## Checked and passing on correct code

- **Wall test.**
  - Banner line: `lead()` pads to the longest label, matching `^\s*Wall\s*: 20s … \(running-a-job\.md 3\.6\)\s*$`;
    the banner is echoed after `exec > >(tee …)`.
  - Session-log line: `_log INFO "wall: ${_mb_wall_left}s left of ${MB_WALL_S}s; SIESTA stops ${MB_WALL_SLACK_S}s
    before it (running-a-job.md 3.6)"`.
- **Road argv.**
  - The log line is `$0 $_mb_argv`, with `_mb_argv` captured before the loop.
  - The command is `["bash", run_name] + _counts_of` (`omp_threads` → `cpus_per_task`), giving
    `H2_01_coarse.run.sh -np 1 -omp 1`.
  - `launch_record()["command"]` matches.
- **Sequence.**
  - `folder_answer…computation.launch` carries `sequence`, `mode`, `job_id` and `command` (`_launch_record` keys).
  - Notes: `save_before` writes `"{stamp} · before launch task group N (m), in job X (sequence S)"`. `X` is the
    walk's `$(hostname)-$$`, which equals the ledger's `run_here` id; each group's save sees a changed folder.
  - Last group: `pending` with `launch` None.
- **Optimization timing.** The record's `computation.time` comes from `timing_of(timing_file)`, which is
  `scf_timing_metrics` of the tee. `_one_source` makes `instruments` its only source.
- **`capped` launch exit 0** (job-system § 6.0).
- **SIESTA vibration.**
  - Fermi level: `read_optical_data` returns `None` for a non-finite value, and the written grid is checked with
    `isfinite`.
  - The bar shows exactly *Light along* and *Charges*. The `.view-bar .inline-label[hidden]` and
    `.view-bar-group[hidden]` guards hide the vector inputs and the Raman inputs.
  - Stopped run: `never_reached` gives `NOT_RUN`, which `state_word` says as "not run"; run-1 reads `failed`; no
    `finish` sent.
  - `continued_from`: `_points_walk` gives run-2 a CONTINUED hand-over from run-1, and run-1 none (`--cold`, so
    `own` is False).
  - Walled point:
    - the line is `_again_line(…, "stopped at its wall")` with `relaunch=AGAIN`;
    - the detail is `STOPPED + ": " + stop_line(...)`, i.e. "OUT_OF_TIME: Time is up.";
    - `STOP_MEANINGS` has no entry for it, so nothing is appended.
- **PySCF vibration.** `_held_words` gives "1 atom held (the deck's atom record)".
- **Monitor util.**
  - `_pid_alive(0)` is True, so every tick samples.
  - Samples 50, 54, 58, 58 write rows [50, 58] against the last written row; `_util_csv_row` writes `%g`.
- **Notebook.**
  - `read_runtime` uses `config_dir.runtime_dir()`, which reads `XDG_RUNTIME_DIR` live, so the patch works; the token
    key is `"token"`.
  - psutil 7.2.2 is in the molbuilder env; `net_connections(kind="inet")` and `CONN_LISTEN` work on Linux without
    root.
- **PySCF outcome.**
  - The relaunch words come from `_launch_again`: optimization resumes and carries `_optimized.xyz`; `[vibration]
    resumes = false`.
  - The exception lines fit in `_pyscf_ending`'s first 200 characters.
- **Citations.**
  - New § quotes verified in the docs: running-a-job § 1, § 3.4, § 3.5 table, § 3.6 table, § 4.2; job-system
    § 6.0/6.0a; checkpointing § 9; runs-of-points § 3; run-reports § 2.1/2.1a/2.2/2.4/2.5/3; vibration § 2.2 rows,
    § 5.9, § 6.2; parse § 3/§ 5c; transport § 6.1; jupyter § 2; chemistry-correctness ES7.
  - R-L11 exists and fits.
- **No new handcrafted state.** The notebook's `XDG_RUNTIME_DIR` patch only points the test's own reader at its own
  stack, as `notebook_said` already does.
- **The retired road test** (`launch_record(medium)` existence) is covered by :123-135 and :290-295.

## Verdict

The revisions are sound almost everywhere: every new expectation traces to a contract sentence, and every one checked
matches the code that produces it. One new assertion can fail on correct code: the sequence stop (finding 1) depends
on a race the fixture does not close, and needs a bounded wait for group 1's run before `jobs stop`. One addition is
weaker than it claims (finding 2). The monitor retirements leave three contracted behaviours unguarded, one of them
retired on a false premise, since configuration.md:354 states it (findings 3 and 4). The PySCF ending heuristic
carries an unverified stdout-ordering risk that the new assertions will expose, if it is real (finding 6).
