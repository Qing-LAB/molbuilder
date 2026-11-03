# Group P — fixes made

Files edited: only the five named. Nothing run but pyflakes and collect-only.
The expectations come from the contract sentences named. Code paths were read before each one was relied on: runwrap
`_held_words` / `_launch_again` / the failure block; `monitor.run_monitor` / `_util_tick` / `_metric_moved`;
`_run_ending._pyscf_ending` / `say`; `parse/dirs/job._build_status`; `parse/dirs/record._ending`;
`runs.folder_answer`; `jupyter.read_runtime` / `config_dir.runtime_dir`; `pyscf/stages.warm_for`; `relax_policy`.

## 1. Per test

| test (file:line after edit) | action | contract § | what it now guards — and one code change that turns it red |
|---|---|---|---|
| module docstring (test_vibration_e2e.py:1) | docstring | vibration.md § 4, § 9 | It cited the archived "spectra-migration plan P1", a "schema-5" file, "the default (Raman)" and the `init` verb. It now describes the two water runs as they are made today. |
| test_water_runs_the_whole_loop_and_the_viewer_can_load_it (vib:197) | docstring (NO GOAL) + stale citations | vibration.md § 4.2, § 4.7, § 4.9, § 6.2, § 6.6; run-reports § 2.1a, § 2.3, § 2.5 | Guards: the file's geometry is the relaxed one; the headline thermochemistry is a row of the grid; the monitor judges a CPU run on no GPU. "design § 15.4 / § 15.3" now read § 4.2 / § 4.7. Red: the deck not rebinding `COORDS_EQ_ANG` after the relaxation. |
| test_water_in_water_runs_the_solvated_chain_end_to_end (vib:430) | docstring | vibration.md § 4.6 (PCM box) | Dropped the archived "integration plan" citation. The docstring claimed "the energy is the solvated one", which nothing asserts; it now says so plainly (see F9). Red: the solvent's dielectric not reaching the deck. |
| test_asking_for_ir_does_not_move_the_frequencies (vib:470) | docstring | vibration.md § 4.6, *"Asking for intensities must not move the frequencies"* | Red: the analytic route not adding back `get_dispersion()`. |
| test_the_rank_rule_reproduces_pyscf_on_free_molecules (vib:531) | docstring | vibration.md § 4.5, *the gate this rests on* | Replaced "unification design § 7.5 step 2". Red: CO2 losing six motions instead of five. |
| test_water_with_its_oxygen_held_reports_three_vibrations (vib:610; the fix at :669) | **fixed** (failed today) + citation | running-a-job.md § 1 row 11 (*"its held atoms as prep read them from the deck's own record"*), § 4.1; normal-modes.md § 7; vibration.md § 4.4, § 4.7 | The banner's `Held atoms` line is read by its label, whitespace-tolerant: `^\s*Held atoms\s*:\s*1 atom held\b`. This replaces `"frozen_atoms: 1 listed indices"`, a line 2.63.0 removed. "design § 7.6" was replaced. Red: `_held_words` finding no ATOM-METADATA ("not stated -- …"), or the banner saying "none held" for a held deck. |
| test_the_free_atom_hessian_is_the_free_block_of_the_full_one (vib:711) | docstring | vibration.md § 4.4 | Replaced "§ 10's owed check". Red: the reduced route using `kernel(atmlst=)`, whose dispersion term is full-size. |
| test_a_completion_marker_does_not_stop_the_sampling (monitor-watch:132) | docstring | run-reports § 2.2 (*Why no convergence phrase is read*), § 2.3 | It cited `job-contracts.md` with no §. Red: the loop stopping on `>> End of run`. |
| test_util_rows_are_change_gated (monitor-watch:228; was `test_util_sampling_is_change_gated_and_summarised`) | **fixed** (vacuous; wrong GPU premise) | run-reports § 2.1 (*≥ 10 % of its last written value*), § 2.5 (*a header and a first row at start, then change-gated rows*), § 2.1a (*a GPU is sampled only for a run that uses one*) | The scripted samples are CPU-only, as this CPU run holds no GPU. `watch_pid=0` makes every wake reach the gate; before, the PID was gone at the first wake and the gate was never asked. CPU 50 → 54 → 58 → 58 must give exactly the rows [50, 58]. Red: every sample written (54 appears), or the gate measured from the last sample instead of the last written row (58 is never written). The summary assertion was dropped: its CPU form is pinned on the real water run (vib:197), and its GPU verdict belongs on a GPU run (F9). |
| test_a_failing_notifier_does_not_break_the_loop (monitor-watch:269) | docstring (wrong + NO GOAL) | run-reports § 3 (*"A failure costs the run nothing"*), § 2.6 | The old docstring said "the WATCHED PID, the only stop signal", which is wrong: SIGTERM stops the monitor too. Red: `_fire` without its per-notifier try/except. |
| test_a_monitor_missing_its_companion_still_monitors (was monitor-watch:443) | **retired** | none | Its failure was faked by monkeypatching `_secrets_dir` to raise, a state the road cannot reach: `config_dir.py` is a fixed member of the one bundle (`runwrap._MONITOR_BASE`), and losing it fails test 16 (`M.default_notify_path()`). Now unguarded: the `except ImportError` "reports off" branch, itself recovery for that unreachable state (see SHARED). |
| helper `_the_engines_words_kept` (pyscf-outcome:125) | **added** (OWED 1+2) | running-a-job.md § 4.2 (*"stopped before its end: <the line that stopped it>"*, *"The detail quotes the line"*); model/parse.md § 2b (`error_message`: *"for PySCF, the exception line"*), § 5d.1a (*"`verdict.state` and `detail` are `run_status`'s words"*); running-a-job.md § 3.5 step 1 (*"How it ended, asked of the framework's reader … whatever the engine"*) | Checks three places for PySCF's words: the ladder row's detail (`jobset_status`), the folder answer's `status.detail` and the record's `verdict.detail`. Each must start `stopped before its end: ` and carry the words. The session log must carry them too, and only the run script's how-it-ended line can put them there: the engine's output goes to `.pyscf.log`. Red: `_pyscf_ending` keeping the traceback header instead of the exception line, or the run script not asking `_mb_ending` for a PySCF run. |
| test_halt_stops_the_rung_and_keeps_the_geometry_it_reached (pyscf-outcome:165; added at :203) | **added** (OWED) | the helper's §; running-a-job.md § 3.5 table, *"a stage that continues \| launch the stage again to continue warm from the files this run saved"* | PySCF's halt words on every surface, and the session log saying the stage continues warm. Red: `_launch_again` handed `relaunch_continues` False, or PySCF's `warm_for` returning no file. |
| test_continue_reenters_from_the_geometry_reached (pyscf-outcome:210; added at :239) | **added** (OWED) | the helper's § | The whole budget's stop (*in 3 steps (on_nonconvergence = continue)*) on every surface. Red: as for the helper. |
| test_the_wall_stops_a_vibration_before_its_spectrum (pyscf-outcome:383; added at :414) | **added** (OWED) | the helper's §; running-a-job.md § 3.5 table, *"its kind does not resume (a PySCF vibration) \| a launch again starts this run over from its deck, nothing of it copied in — this kind of run does not resume"* | The wall stop's words on every surface; the session log says the run starts over and does not resume. The doc's em dash and the code's ` -- ` are read as two fragments. Red: `[vibration] resumes = false` dropped from `pyscf/warm-files.toml`. |
| module docstring, `_run` docstring (pyscf-outcome:1, :50) | docstring | engines/pyscf.md § 3; job-request.md § 2 | Dropped "plan § 5w K6", now in the ledger, and "`stages.md` § 6.8d", which was removed. |
| test_the_tab_frames_the_notebook_from_its_own_origin_and_a_cell_answers (notebook:185) | **fixed** (weak clause) | jupyter.md § 2, *"molbuilder never hands the browser the token — not in its pages, not in the frame's address"* | The notebook's token is read through `jupyter.read_runtime` in the stack's own runtime folder. It must appear neither in the frame's `src` nor in the page's HTML. Red: the tab writing `?token=` into the frame's address. |
| test_the_notebooks_own_port_answers_this_machine_only (notebook:247) | **fixed** (failed today, both params) | jupyter.md § 2, § 6, *"Jupyter listens on `127.0.0.1` only"*; testing.md § 2b | Applies the brief's decision. A loopback connection to the port must succeed, and every LISTEN socket on `jupyter_port(p)` (`psutil.net_connections(kind="inet")`) must be bound to a loopback address. Nothing connects off loopback. `_an_address_off_loopback` was removed. Red: `ServerApp.ip` set to `0.0.0.0` / `""`. |
| test_a_monitor_that_dies_loading_says_why (was monitor-start:174) | **retired** | — | Brief decision: it stood on a hand-edited `mb_monitor.pyz` member (testing.md § 6). |
| test_an_ending_the_bundle_cannot_read_is_said_once (was monitor-start:206) | **retired** | — | Brief decision: hand-edited bundle, and it now counted wrongly (the deck read at step 16 adds a third load error). |
| module docstring + helpers (monitor-start:1) | docstring + cleanup | run-reports § 2.6; running-a-job § 4.1 | Removed the stale reason ("only the SIESTA wrapper asks how its run ended"). The docstring now covers only what the file still tests, and says why a failed load is not made here. Removed `_launched`, `_jobset`, `_DYING_WORDS`, `FIXTURES`, the SIESTA marker and the unused imports. test_a_monitor_that_loads_says_so (monitor-start:39) is unchanged. |

## 2. SHARED (not made)

- **Product, for review:** `monitor.load_channels`'s `except ImportError` branch ("no config_dir.py beside the monitor
  -- reports off") and `_secrets_dir`'s docstring. They are recovery machinery for a bundle without `config_dir.py`,
  which `runwrap._MONITOR_BASE` cannot produce. With test 15 retired nothing pins the branch. Either delete it, or
  have the contract state the case (run-reports § 3 / § 2.6).
- **Product comment:** `runwrap.py` near :3066 says `tests/test_monitor_watches_a_live_run_e2e.py` "runs the whole set
  with molbuilder absent". That is still true (test 16); no change needed, noted only.
- **Doc drift, not a test defect** (the report's note): the vibration.md § 9 test-map row says the Hartree–Fock deck
  names "no dispersion … none for `dispersion = "none"`". § 4.10 and the test say HF takes the dispersion as written.
- The test maps (vibration.md § 9, run-reports, running-a-job) may name the retired tests or the renamed
  `test_util_sampling_is_change_gated_and_summarised`. A doc sweep should check them.

## 3. F9 (left, one line each)

- vib `_prep_and_run`: the result is read as raw JSON at a hand-composed `01_freq/run-0/W.spectra.json`; it should
  go through `parse(...)` / `on_label_at` (vibration.md § 6.7 gate).
- vib 2: the Raman the fixture asks for is paid (18 SCFs) but only `phase_raman` is checked. Either assert
  `raman_route` / `raman_fd_step_ang` / activities against a sourced window (§ 4.6, § 6.2), or stop asking for Raman.
- vib 2 / vib 8: text pins on copy that is still right: `"the headline and the grid alike"`, `"[launched on]"`,
  `"atoms are held"`.
- vib 3 (mode matching): NOT E2E science → tier 1 (`tests/test_displacement_sweep.py`) with mass-weighted modes over
  O, H, H.
- vib 5 (PCM): cannot tell a solvated run from a gas one. Comparing its SCF energy with a gas run at the same level
  needs a shared gas run (vib 4's): a design choice. The deck-text grep is a still-right text pin.
- vib 10, vib 11: prep-only → tier-1 case rows. Template string surgery → `with_values_set`. The column-aligned
  constants and words are still-right text pins.
- vib 12: the prep refusal half → a tier-1 row; the probe half → carried on vib 4's run (one run fewer).
- monitor-watch 1–4, 7–13: the monitor driven over a replay with a scripted clock (handcrafted, otherwise right).
  7–12 → one policy table.
- monitor-watch `test_util_rows_are_change_gated`: still a typed sampler. To test the gate on a real run, read the
  pass's real runs' own `.util.csv`. The GPU summary verdict (≥ 85 / ≤ 60 %) belongs on a real GPU run.
- monitor-watch 6 (failing notifier) and 14 (`[MACHINE]` first): NOT E2E. 14 → assert on the pass's real runs' own
  `.monitor.log`.
- monitor-watch 16: pins private module names (`P._G`, `M._rf`, `compile_rules`), still right. Drop the per-object
  map and keep `from`, the control and the outcome equality.
- monitor-watch `_Output.criterion`: the `redata:` tolerances are read by the test's own regex, not the parser.
- pyscf-outcome `_run`: writes `task.json` with `write_task`, bypassing Task setup's Save checks (`support.road.save_both`).
- **OWED, needs a new launch (not added): 2.60.0 one start per run.** Launch the halted PySCF optimization again
  warm: `run-1/` opens beside `run-0/`; `.continued-from` and `run.json`'s `continued_from` name `run-0`; run-1's
  first progress frame is run-0's kept `_optimized.xyz`; `--cold` starts from the input. No e2e launches a PySCF
  stage again.
- **OWED, not added: 2.63.0/2.64.0 deck echo** on the CO2 `fc` call (the session log's effective-parameters
  block lists the force constants' settings, never the relaxation's). The run is made by `real_runs`, not by these
  files, and another group's `test_siesta_vibration_e2e` may already pin it — check before adding.
- notebook 1: the frame is re-pointed by JS (`nb-frame.src = …`) where a click in Lab's file browser should open the
  notebook (a synthetic step).
- notebook 3: makes a third stack per pass (cost only).

## 4. Checks

- `python -m pyflakes` on the five files: clean (rc 0, no output).
- `pytest --collect-only -q -p no:cacheprovider` on the five files (sandboxed HOME/XDG): **40 collected**, no errors:
  12 vibration, 15 monitor-watch, 7 pyscf-outcome, 5 notebook (with params), 1 monitor-start. Before: 44 (16
  monitor-watch, 3 monitor-start).
