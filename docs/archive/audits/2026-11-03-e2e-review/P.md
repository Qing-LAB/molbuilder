# Group P — e2e review: vibration (PySCF), monitor live run, PySCF relaxation outcome, notebook, monitor start

Read whole: the five files, `tests/support/real_runs.py`, `tests/support/replay.py`, `tests/_road.py`, the used parts of
`tests/support/road.py` and `tests/conftest.py`; testing.md § 0, 2b, 3a, 3b, 5, 6; changelog 2.57.0–2.65.3; run-reports.md
§ 2 (2.1–2.9) whole; running-a-job.md § 1, § 3.4–3.6, § 4.1, § 4.2 (words); job-contracts.md § 2.1 (monitor), § 2.6;
vibration.md § 1–4.10, § 5.5, § 5.9, § 6.1–6.3, § 6.6, § 9; pyscf.md § 3, § 3.9, § 7.3, § 7a; normal-modes.md § 7;
web/spectra.md § 3, § 3.2, § 6; web-api.md `/api/spectra/load`; jupyter.md § 2, § 3.4, § 5, § 6; job-request.md § 1,
§ 3.2, § 3.5; project-layout.md § 1.6.1–1.6.2. Code paths read before every "the code no longer does" claim (cited).
Nothing run.

## 1. Summary

41 test functions (12 + 16 + 7 + 3 + 3; the notebook's first two run twice, http and https) and 4 fixtures.

| verdict | count | tests |
|---|---|---|
| RIGHT (some with a minor note) | 27 | vib 1, 4, 6, 7, 9; monitor-watch 1–4, 7–13, 16; all 7 PySCF-outcome; notebook 1, 3; monitor-start 1 |
| WRONG DETAIL / STALE ROAD — **will FAIL** | 3 (4 cases) | vib 8 (held water), notebook 2 (×2), monitor-start 3 |
| WEAK | 4 | vib 2, vib 5, monitor-watch 5, notebook 1 (token clause) |
| HANDCRAFTED | 4 | monitor-watch 5, 15; monitor-start 2, 3 |
| NOT E2E | 8 | vib 3, 10, 11, 12 (its prep half); monitor-watch 5, 6, 14, 15 |
| TEXT-PINNING | 6 | vib 2, 5, 10, 11, 12 (minor); monitor-watch 16 (partial) |
| DUPLICATE / TABLE | 2 | vib 12's run = vib 4's run + a probe; monitor-watch 7–12 = one table |
| NO GOAL / stale citation | 2 + 4 | no goal: vib 2, monitor-watch 6; archived-design citations: vib 2, 7, 8, 9 |
| RETIRE | 3 | monitor-watch 15; monitor-start 2; monitor-watch 5 (replace on the real runs' own util.csv) |

**WRONG DETAIL / STALE ROAD (each fails today):**
- `test_vibration_e2e.py:648` asserts `"frozen_atoms: 1 listed indices"` in the session log. 2.63.0 (a4abd05f8) removed that
  line: the banner says `Held atoms    : 1 atom held (the deck's atom record)` (`runwrap._held_words` runwrap.py:689–705,
  echoed at :2623). Contract running-a-job.md § 1 row 11: *"prints the banner (§ 4.1), its held atoms as prep read them from
  the deck's own record"*.
- `test_notebook_through_one_port_e2e.py:234–259` connects a UDP socket to 192.0.2.1 and then to this machine's outward
  address. The suite's guard (conftest.py:125–145, `_only_this_machine`) raises `ReachedOutOfTheSandbox`, a BaseException
  that `except OSError` (:240) and `pytest.raises(ConnectionRefusedError)` (:258) cannot catch. testing.md § 2b
  (2026-11-02): *"A Python connection to anything but loopback raises `ReachedOutOfTheSandbox`"*. Fails for both http and
  https.
- `test_monitor_start_is_logged_e2e.py:225` expects `errors == 2`. Since 2.63.0 the SIESTA run script also reads the deck
  through the bundle at step 16 (`mb_monitor.pyz deck`, runwrap.py:1479–1483; emitted because SIESTA's record names a
  `keyword_reader`, engine_table.toml:59, runwrap.py:2820–2823). A bundle that cannot load prints its
  `[ERROR] monitor: mb_monitor.pyz did not load` into the session log (`_BUNDLE_MAIN` :3215–3220, `2>>"$_runwrap_log"`).
  That makes three errors: the monitor's start, the deck read, and the one ending ask. The module docstring's reason (:19–21)
  is also wrong: *"only the SIESTA wrapper asks how its run ended"*. Since 2.63.0 every engine's run script asks after a
  failure (running-a-job.md § 3.5, *"whatever the engine"*). SIESTA stays the right engine for another reason: only its
  output family also asks `out-of-time` and `advice`.

**WEAK / HANDCRAFTED:**
- monitor-watch `test_util_sampling_is_change_gated_and_summarised` (:226) never exercises the change gate. With
  `watch_pid=999_999_999` the PID is gone at the first wake (`_pid_alive` monitor.py:2163–2170, :2419). Only the first
  sample (:2381) and the forced closing sample (:2460) are written, and `len(rows) >= 2` holds for any monitor. The input
  is also handcrafted: a scripted sampler reports a GPU for a run started without `--gpu`, against § 2.1a *"A GPU is
  sampled only for a run that uses one"*.
- vib `test_water_in_water_runs_the_solvated_chain_end_to_end` (:416) cannot tell a solvated run from a gas one. Gas
  water's 1639, 3791 and 3886 cm⁻¹ sit inside the widened windows. "The energy is the solvated one" is in the docstring but
  never asserted; the test only greps deck text.
- vib `test_water_runs_the_whole_loop_and_the_viewer_can_load_it` (:196) has no docstring. The shared `water` run asks for
  Raman, which costs 18 SCFs plus polarizabilities, but nothing of Raman is asserted beyond `phase_raman`. It also pins
  code-copied note text.
- monitor-start 2 and 3 (:174, :206) replace a member of the prepared `mb_monitor.pyz` after prep. That is a hand-edited
  bundle, which the brief and testing.md § 6 forbid (*"one of ours edited by hand into a shape our writer never writes"*).
- monitor-watch `test_a_monitor_missing_its_companion_still_monitors` (:443) fakes the failure with a monkeypatch
  (`_secrets_dir` raising). The failure cannot happen on the road: `config_dir.py` is a fixed member of the one zip
  (runwrap.py:3073).
- `_prep_and_run` (vib :108) reads the result as raw JSON at a hand-composed `01_freq/run-0/W.spectra.json`. The IR-alone,
  held, PCM and probe results therefore never pass through the reader's gate (vibration.md § 6.7).

**The rest:**
- **NOT E2E.** vib 10 and 11 are prep only (*"Prep alone; nothing is launched"*), so they belong in tier-1 case tables.
  vib 3 runs a pure function on modes changed in memory; it is a science test, so move it rather than retire it.
  monitor-watch 6 and 14 need no engine output.
- **TABLE.** The six notification-policy tests (monitor-watch :312–377) differ only by data.
- **DUPLICATE RUN.** vib 12 launches the default water description again (plus the probe) beside vib 4's run.
- **Stale citations.** vib 2, 7, 8 and 9 cite the archived unification design (*design § 15.3/15.4, § 7.5 step 2,
  § 7.6, "§ 10's owed check"*). The contract homes are vibration.md § 4.2, § 4.5, § 4.4, § 4.7 and normal-modes.md § 7.
- **Doc drift, not a test defect.** vibration.md § 9's test-map row says the Hartree–Fock deck names *"no dispersion …
  none for `dispersion = "none"`"*. § 4.10 says Hartree–Fock takes the dispersion as written, and the test follows § 4.10.

## 2. Per file

### `tests/test_vibration_e2e.py` (last revised 2.60.0; 2.57.0 added `infrared_asked`)

| test (line) | what it asserts | contract § today | verdict(s) | evidence | revision |
|---|---|---|---|---|---|
| fixture `water` (125) | free water, Raman on, IR off, `notify` asked, described → prepared → launched once | testing.md tier 3; job-request.md § 3.5 (all four keys stated) | RIGHT (note) | :141–150 | keep; see vib 2 on Raman paid and unchecked |
| helper `_prep_and_run` (108) | the result at `01_freq/run-0/W.spectra.json`, read as raw JSON | vibration.md § 6.7 (the reader refuses an inconsistent file) | WEAK | :120–122 `json.loads(art.read_text())`; the path hand-composed; "ATTEMPT DIR" wording | read through `parse(...)` / `parse_spectra_json` (as test_pyscf_relaxation does, :221) and find the run by `stage_home`/`attempt_dir`, the file by `on_label_at` |
| 1 `test_a_result_whose_run_asked_no_infrared_says_so_at_the_one_door` (156) | `infrared_view` refuses with "Compute IR was off"; the load route answers `infrared_absent` in those words, no `infrared`, `infrared_asked` false | web-api.md `/api/spectra/load` (*"`results.infrared_absent`, why … `results.infrared_asked`"*); spectra.md § 3.2; vibration.md § 5.11 § 8 | RIGHT | :177–193 | — |
| 2 `test_water_runs_the_whole_loop_and_the_viewer_can_load_it` (196) | phases, the relaxation, three frequencies in windows, RRHO thermo with the headline on the grid, the monitor's closing line, `[NOTIFY] scf_converged`, `[UTIL-BASIS]`/`[UTIL-SUMMARY]`, the reader, masses, derived keys | vibration.md § 4.2, 4.7, 4.9, 6.2, 6.6; run-reports § 2.2, 2.5, 2.1a | NO GOAL; TEXT-PINNING; WEAK; stale citation | no docstring. :244 `"the headline and the grid alike"` is code text (normal_modes.py:352, vibration_deck.py:488), not contract; § 4.7 fixes only that the note says what the numbers are. :271 `"[launched on]"` is a code word (monitor.py:1080). Raman is asked (:141) and only `phase_raman` is checked. :200 compares against the code constant. "design § 15.4/15.3" is archived | goal and contract lines in a docstring. Drop the note-text and `[launched on]` pins (the grid equality :242–243 already pins § 4.7). Assert `raman_route == "finite-difference"` and `raman_fd_step_ang == 0.005` (§ 4.6, § 6.2) and Raman activities in a sourced window, or stop asking for Raman. `schema_version == 14` (§ 6.1). Cite § 4.2 and § 4.7 |
| 3 `test_modes_are_matched_by_their_mass_weighted_shape_not_their_rank` (300) | `match_modes` finds a rank swap; a mixed pair's overlap is cos θ in the mass metric | vibration.md § 5.9, § 6.6 (`results.mass_of`); a PySCF vibration has no sweep (§ 5.9) | NOT E2E (science, keep) | the docstring calls it *"A pure function asked of the real modes, changed in memory"*. The engine adds only an orthonormal set over unequal masses | move to tier 1 (`tests/test_displacement_sweep.py`) with mass-weighted orthonormal modes built over O, H, H; update vibration.md § 9's row |
| 4 `test_ir_alone_runs_decoupled_and_lands_in_waters_windows` (357) | IR windows and ordering; `ir_route` matches the env's `pyscf.prop.infrared`; each vector's \|v\|² × 42.2561 equals the intensity; Raman not requested | vibration.md § 3.1 (IR on, Raman off by default), § 4.6, § 6.2–6.3 (schema 12) | RIGHT (science) | :366–413 | carry vib 12's probe on this run (see vib 12) |
| 5 `test_water_in_water_runs_the_solvated_chain_end_to_end` (416) | the deck holds PCM text and 78.3553; frequencies in widened windows; IR and Raman null | vibration.md § 4.6 PCM box (*"frequencies … with IR and Raman off; any of the three is refused by name at `prep`"*) | WEAK; TEXT-PINNING | :439 `"mf = mf.PCM()" or "_mb_apply_solvent"` (a helper name); :446–447 windows contain gas water's 1639/3791/3886; the docstring's "the energy is the solvated one" is unasserted; :437 deck name composed by hand | assert `equilibrium.scf_energy_eh` below the gas run's at the same level (vib 4's run, made in the pass) by a stated solvation margin, and the result's `config.solvent == "water"`; drop the deck-text grep |
| 6 `test_asking_for_ir_does_not_move_the_frequencies` (454) | analytic-route Hessian equals `mf.Hessian().kernel()` to 1e-8 for PBE0/B3LYP-D3BJ × DF on/off | vibration.md § 4.6 (*"Asking for intensities must not move the frequencies"*) | RIGHT (science); NO GOAL partial | the docstring cites no §; a hand-written probe, not the road | cite § 4.6. Design option: the same fact on the road, as frequency equality between two water runs the pass makes (IR analytic vs IR off, same level) |
| 7 `test_the_rank_rule_reproduces_pyscf_on_free_molecules` (513) | water, CO2, HF, methane: counts, eigenvalues rtol 1e-8, cm⁻¹ 1e-4, overlaps 1e-6 | vibration.md § 4.5 *"The gate this rests on"* — the same tolerances | RIGHT (science); stale citation | :514 "the unification design (§ 7.5 step 2)" | cite vibration.md § 4.5 |
| 8 `test_water_with_its_oxygen_held_reports_three_vibrations` (592) | three modes, three removed, vibrational-only thermo, `hessian_scope` free; **the session log says `frozen_atoms: 1 listed indices`**; the load door's held infrared along x/y/z averages back; mass-metric orthogonality | normal-modes.md § 7; vibration.md § 4.4, 4.6 (3), 4.7; running-a-job.md § 1 row 11, § 4.1 | **WRONG DETAIL / STALE ROAD → FAILS at :648**; minor text pin; stale citation | the string was removed in 2.63.0; banner text from `runwrap._held_words` (:704) = `1 atom held (the deck's atom record)`. :627 `"atoms are held"` is note wording. "design § 7.6" | assert the banner's `Held atoms` line states 1 atom held, as § 1 row 11 / § 4.1 say (from the deck's record); keep every science assertion |
| 9 `test_the_free_atom_hessian_is_the_free_block_of_the_full_one` (689) | the reduced Hessian equals the full Hessian's free block (HF < 1e-7, DFT < 5e-5); held rows zero; FD route | vibration.md § 4.4 (measured 1e-8 HF, 7.5e-6 / 1.5e-5 DFT) | RIGHT (science); stale citation | :690 "§ 10's owed check"; tolerances looser than § 4.4's measured values, reasons in the docstring | cite § 4.4 |
| 10 `test_a_setting_that_enters_nothing_is_said_to_at_prep` (735) | prep warns a pressure with atoms held and a window under `skip`; neither for the free molecule under `all` | vibration.md § 3.1 (pressure row), § 4.8 | NOT E2E; TEXT-PINNING | docstring *"Prep alone; nothing is launched"*; exact `"pressure_atm = 10 atm has no effect here"` (validation/spectra.py:258); the template edited by string surgery (:761–775), duplicating `_asks` / `with_values_set` | two rows of a tier-1 case table (describe → prep), values set through `with_values_set` (`support.road.set_by_the_person`) |
| 11 `test_a_hartree_fock_deck_names_no_functional` (786) | HF deck: no functional or grid; dispersion d4 applied (`mf.disp`); the warnings | vibration.md § 4.10; pyscf.md § 7a | NOT E2E; TEXT-PINNING | prep only. Column-aligned constants (:839–841). The negative `"Grid level" not in r.output` (:830) passes vacuously if validation/pyscf.py:217 is reworded | tier-1 row. Read the deck's constants through a reader rather than spaced text; check the advisory by its finding, not its words |
| 12 `test_the_listed_modes_get_the_probe_and_no_other` (855) | "0, 2" refused at prep with no deck; "1, 3, 9" probes modes 1 and 3 and says 9 names none | vibration.md § 3.1 (`es_explicit_indices` row), § 4.8 | NOT E2E (prep half); DUPLICATE run; minor TEXT-PINNING | the refusal half needs no engine. The run half launches the default description again, the same as vib 4 plus the probe; the probe is not a strength, so vib 4's IR stays analytic (§ 4.6). Words are code's (`spectra/selection.py:94`, `vibration_emitters.py:1326`) | the refusal becomes a tier-1 row; the probe is carried on vib 4's run (one run, two tests) |

### `tests/test_monitor_watches_a_live_run_e2e.py` (2.64.0 docstring; aabaeb186 dropped the try number, passed 11-02)

| test (line) | what it asserts | contract § | verdict(s) | evidence | revision |
|---|---|---|---|---|---|
| fixture `run` (41) | the real CO2 run's `relax` call | testing.md tier 3, § 6 (replay of a run made this pass) | RIGHT (note) | the monitor is driven as a non-member over a call's output. In the field a call has a member monitor (run-reports § 2.4); fine for single-run policy | — |
| helper `_Output` (77) | cut points and `redata:` criteria by the test's own regex | brief: artifacts through their parser | minor | :83–109 | criteria from the parser's reading (`detect(out).parse(out)`, as `_forces_stated` does); line indices may stay |
| 1 `test_a_completion_marker_does_not_stop_the_sampling` (132) | the whole output (`>> End of run`) and a live PID: still `running` after 3 wakes | run-reports § 2.2 *"Why no convergence phrase is read"*, § 2.3 | RIGHT | cites "job-contracts.md" with no § | cite run-reports § 2.2 |
| 2 `test_when_the_pid_goes_the_verdict_is_run_status_s` (148) | PID gone: `finished (ended, exit 0)`, `converged: periodic yes`, `geometry relaxed`, `job ended` | § 2.3, § 2.4, § 2.5 (closing example) | RIGHT | :158–167 | — |
| 3 `test_each_advance_is_a_status_line_stating_where_the_run_is` (174) | three `[STATUS]` lines: iteration, dDmax and dHmax with tolerances, `opt. move 2`, the free force with the held atoms' after it | § 2.1, § 2.5, § 2.7 (*"max force 0.0028 eV/Ang (tol 0.01); 0.015 eV/Ang on the held atoms"*) | RIGHT | :192–205 | — |
| 4 `test_a_run_that_does_not_advance_writes_and_sends_nothing` (208) | over an hour with no advance: no `[STATUS]`, no event | § 2 *"The monitor judges no stall"* | RIGHT | — | — |
| 5 `test_util_sampling_is_change_gated_and_summarised` (226) | a util.csv header, ≥ 2 rows, a GPU SM mean | § 2.1 table, § 2.1a | WEAK (vacuous for the gate); HANDCRAFTED; NOT E2E; RETIRE | the PID is gone at wake 1 (monitor.py:2419); rows = header + first (:2381) + forced closing (:2460); the gate is never evaluated. The scripted GPU sample contradicts § 2.1a. The run's output plays no part | retire. Replace with an assertion on the pass's real runs' own `.util.csv`: each row after the first moved ≥ 10 % in some metric or came ≥ keepalive after the last, or is the closing row (§ 2.1) |
| 6 `test_a_failing_notifier_does_not_break_the_loop` (259) | a raising notifier; the next one still hears `finish` | run-reports § 3 (*"each channel guarded on its own … A failure costs the run nothing"*) | NOT E2E; NO GOAL | docstring *"Ends on the WATCHED PID, the only stop signal."* (also wrong: SIGTERM stops it too, § 2.6). `_fire` monitor.py:1325 | tier 1 on a prepared folder, with goal and § 3 |
| 7 `test_an_advancing_job_notifies_nothing_by_default` (312) | SCF rows advancing within one move: only start/finish | § 2 (*"Nothing else is said"*), § 2.1 | RIGHT; TABLE; mild WEAK | the input has no step and `every_hours=0`, so an ignored `on_scf` still passes; it guards per-wake notification only | a row of one policy table |
| 8 `test_one_message_per_geometry_step` (322) | one `scf_converged` per move begun | § 2.2 SIESTA relaxation row | RIGHT; TABLE | — | table row |
| 9 `test_periodic_counts_hours_not_wakes` (334) | 8 hourly wakes, period 2 h → 3–4 periodic | § 2 (*"N hours after the last event"*) | RIGHT; TABLE | — | table row |
| 10 `test_the_first_period_is_a_full_period_in` (345) | no periodic on the first wakes | § 2 (*"the first, N hours after the start"*) | RIGHT; TABLE | — | table row |
| 11 `test_a_step_and_a_period_on_one_wake_is_one_message` (355) | the step wins, no periodic | § 2 (*"a step and a period due on the same wake are one event, the step's"*) | RIGHT; TABLE | — | table row |
| 12 `test_a_relaxation_with_the_trigger_OFF_reports_no_steps` (369) | moves advance, tick off → no `scf_converged` | § 2 table (*"when `notify.on_scf_converged`"*) | RIGHT; TABLE | — | table row |
| 13 `test_a_displacement_is_no_step_message` (380) | the `fc` call with the tick on: no `scf_converged`; periodic with growing steps_done; `step_moves` false | § 2.2 (FC row; *"Only a step that moves the geometry is a message"*) | RIGHT | :410–414 | — |
| 14 `test_the_machine_is_the_logs_first_line` (431) | `[MACHINE]` first, `node=` | § 2.5 | NOT E2E | written before any reading (monitor.py:2280) | assert on the pass's real runs' own `.monitor.log` (the CO2 calls', the PySCF runs'): a check on runs already made, the real node included |
| 15 `test_a_monitor_missing_its_companion_still_monitors` (443) | `_secrets_dir` monkeypatched to raise → "reports off" | none names it (code comment monitor.py:1808–1812) | HANDCRAFTED; NOT E2E; RETIRE | `config_dir.py` is a fixed member of the one zip (runwrap.py:3073); losing it fails test 16 (`M.default_notify_path()`, :486) | retire |
| 16 `test_every_file_that_ships_beside_a_job_reads_the_run_without_molbuilder` (464) | molbuilder not importable there; the bundle's readers by module name; the shipped `WatchedRun` concludes as `run_status` does | § 2.3 | RIGHT; partial TEXT-PINNING | :505–510 pin private names (`P._G`, `M._rf`, `compile_rules`) | keep `from == mb_monitor.pyz`, the control and the outcome equality; drop the per-object name map |

### `tests/test_pyscf_relaxation_outcome_e2e.py` (2.60.0: `run.file(".pyscf.log")`)

| test (line) | what it asserts | contract § | verdict(s) | evidence | revision |
|---|---|---|---|---|---|
| helper `_run` (48) | describe; template by the hand-over's writer; task.json via `write_task`; run card per stage; prep; launch | job-request.md § 1 (*"a person may edit either"*), § 3.2 | RIGHT (note) | :92–100 bypasses Task setup's Save checks (`support.road.save_both` exists) | optional: write through `save_both` |
| 1 `test_halt_stops_the_rung_and_keeps_the_geometry_it_reached` (123) | failed; the halt words; "launch this stage again to continue from it"; `_optimized.xyz` kept; the live log `stopped` | pyscf.md § 3 (halt); running-a-job § 4.2 words | RIGHT | relax_policy.py:182–191 | OWED 1–3 below: add on this run |
| 2 `test_continue_reenters_from_the_geometry_reached` (158) | two re-entries; 3-step budget; never back at 1.6 Å | pyscf.md § 3 (continue); 2.60.0 keeps `geom_continue_retries` | RIGHT | — | — |
| 3 `test_proceed_keeps_the_geometry_and_the_result_says_so` (204) | finished; the warning; `converged` false; force recorded | vibration.md § 4.2, § 6.2 | RIGHT | read through `parse(...)` | — |
| 4 `test_an_open_shell_run_on_the_gpu_says_its_stability_was_not_checked` (228) | "stability: NOT CHECKED …"; finished | pyscf.md § 7.3 | RIGHT (GPU only) | gpu.md § 1.1: a GPU ask on a record without GPUs is never refused | — |
| 5 `test_a_structure_stated_relaxed_is_measured_not_relaxed` (257) | not requested, 0 steps, converged false, the one remedy text | vibration.md § 4.2, § 4.3, § 5.5 (*"one text, `nonstationary_remedy`"*) | RIGHT | — | — |
| 6 `test_the_wall_stops_the_relaxation_and_keeps_the_geometry` (288) | finished; "stopped before the wall"; geometry kept; `relaxed` false; `geometry NO`; no "Final energy" | running-a-job § 3.6 table; pyscf.md § 3.9 | RIGHT | "Final energy" is printed by pyscf/input.py:1141, so the negative bites | — |
| 7 `test_the_wall_stops_a_vibration_before_its_spectrum` (327) | failed; the exact RuntimeError words | running-a-job § 3.6 table | RIGHT | — | OWED 1–2 |

### `tests/test_notebook_through_one_port_e2e.py` (2.26.1, before the 11-02 sandbox)

| test (line) | what it asserts | contract § | verdict(s) | evidence | revision |
|---|---|---|---|---|---|
| fixture `one_port` (178), `_stack` | `serve start` (http/https) → `jupyter start` → `serve stop` in its own config | jupyter.md § 5 | RIGHT | — | — |
| 1 `test_the_tab_frames_the_notebook_from_its_own_origin_and_a_cell_answers` (185) | the frame at `<base>/jupyter/lab…`; a cell answers 2 over HTTP and HTTPS | jupyter.md § 2 | RIGHT; WEAK (one clause) | the docstring claims *"the token presented by molbuilder and never by the page"*, but nothing asserts it; the frame is renavigated by JS (:213) where a click in Lab's file browser reaches the notebook | assert no `token` in the frame's address or page (§ 2: *"not in its pages, not in the frame's address"*); open the notebook by its click |
| 2 `test_the_notebooks_own_port_answers_this_machine_only` (245) | Jupyter's port answers on 127.0.0.1 and is refused on the outward address | jupyter.md § 2, § 6 (*"Jupyter listens on `127.0.0.1` only"*) | **STALE ROAD → FAILS (both params)** | :238 UDP `connect(("192.0.2.1", 9))` and :259 `create_connection((outward, nb))` → conftest.py:144–145 raises `ReachedOutOfTheSandbox` (BaseException) | read the listening socket's bound address for `jupyter_port(p)` (psutil `net_connections` or `/proc/net/tcp`: LISTEN on 127.0.0.1 only, none on 0.0.0.0 or ::) — the outcome without leaving loopback |
| 3 `test_a_restart_of_the_server_stops_its_notebook` (262) | after `serve restart`, `running` false, `answering` false | jupyter.md § 3.4 | RIGHT | a third stack per pass | — |

### `tests/test_monitor_start_is_logged_e2e.py` (2.62.1, 2.60.0)

| test (line) | what it asserts | contract § | verdict(s) | evidence | revision |
|---|---|---|---|---|---|
| module docstring (1–22) | why SIESTA: *"only the SIESTA wrapper asks how its run ended"* | running-a-job § 3.5 (every engine asks after a failure since 2.63.0) | WRONG DETAIL | changelog 2.63.0 *"a PySCF run's now too"* | if kept: SIESTA because its family also asks out-of-time and advice |
| 1 `test_a_monitor_that_loads_says_so` (143) | `starting` before `started`; its `.monitor.log` exists; the engine wall in the record and the run answer | run-reports § 2.6; running-a-job § 4.1; parse.md `engine_elapsed_s` | RIGHT | on the pass's PySCF run | — |
| 2 `test_a_monitor_that_dies_loading_says_why` (174) | the ERROR line, the traceback naming the member, no `started`, no `.monitor.log` | run-reports § 2.6 | HANDCRAFTED; RETIRE | :111–122 rewrites a member of the prepared bundle | retire. No road-made state yields a bundle that cannot load on the job's python here; `_BUNDLE_MAIN`'s start/ERROR pair goes to review. Or the user keeps it as a stated exception |
| 3 `test_an_ending_the_bundle_cannot_read_is_said_once` (206) | 2 load errors, the "cannot be read" WARN once | running-a-job § 3.5 *"the one ask"* | HANDCRAFTED; **STALE ROAD → FAILS** | the third error comes from the step-16 deck read (runwrap.py:1479–1483) | retire with 2. If kept: count the ERROR lines after the engine's exit line (== 1) and the WARN once; ask the user whether the deck read's own load error is wanted (§ 3.5 names only the monitor's start and the ending ask) |

## 3. OWED — rules changed since 2.57.0 on these roads that only the engine shows, pinned by no e2e

1. **2.65.1, and 2.64.0's own "untested before F9: … a PySCF run's stop words on a real run".** *"When the engine stops,
   the run's ending keeps the engine's own words — … PySCF's exception — shown by status, the Results tab and the record."*
   Three failed PySCF runs the pass already makes: halt (pyscf-outcome 1), continue exhausted (2), and the vibration past
   its wall (7). For each, assert that the status door's detail (`jobset_status` row or `run_status(...).detail`) and
   `folder_answer(run)["record"]` carry PySCF's words: *"the relaxation did not meet geomeTRIC's criteria in 1 steps
   (on_nonconvergence = halt)"* and *"stopped before the wall … before the spectrum was written"*. Today only the
   `.pyscf.log` is read.
2. **2.63.0** (*"after a failure every engine's run script says how it ended — a PySCF run's now too"*; *"a PySCF run
   asked neither the wall nor advice"*; running-a-job § 3.5). On the same runs' session logs (`run_of(...).session_log`):
   - a `how it ended: …` line;
   - the halted optimization's launch-again line (*launch the stage again to continue warm …*);
   - the vibration's line (*… this kind of run does not resume*);
   - no *stopped at its wall* line.
3. **2.60.0** (one start per run; a retry is a re-run; project-layout § 1.6.1; pyscf.md § 3 halt). Launch the halted
   PySCF optimization again, warm:
   - `run-1/` opens beside `run-0/`;
   - `.continued-from` (and `run.json`'s `continued_from`) names `run-0`;
   - run-1's first progress-log frame is run-0's kept `_optimized.xyz` geometry;
   - a `--cold` launch starts from the input.

   No e2e launches a PySCF stage again.
4. **2.63.0 / 2.64.0** (running-a-job § 1 step 16; job-contracts § 2.6 *What the engine will read*). The deck lines are
   read by `mb_monitor.pyz deck` at launch only. On the pass's CO2 `fc` call (this group's fixture already holds it), the
   session log's effective-parameters block, read through `deck_record`'s reader, should list the force constants'
   settings and never the relaxation's. Unverified whether another group's `test_siesta_vibration_e2e` already pins it.

## 4. The pattern

These files pinned the run script's and the monitor's mechanics as the code spelled them on 10-31, and none has run since.
2.63.0 changed two of those mechanics. The held atoms now come from the deck's record (a new banner line), and the deck is
echoed through the bundle (a new reader of `mb_monitor.pyz`). That breaks two text and count assertions: vib 8 and
monitor-start 3. The suite's own 11-02 sandbox broke a test that connected off loopback: notebook 2.

Three other faults come from the same habit. Several tests sit in an e2e file but need no engine: prep-only checks,
pure-function science, and monitor policies on scripted samples or monkeypatched failures. Two fake their failure: a
hand-edited bundle, and a gate that never runs. Four cite an archived design instead of today's chapter.

Meanwhile the releases since 2.57.0 added facts about a failed PySCF run that only a real run shows: its own stop words
on every surface, the run script's ending lines, and the relaunch into `run-<n+1>/`. No test reads them, though the
failed runs are already made in this pass.
