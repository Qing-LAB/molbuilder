# Test-suite audit, 2026-11-02 — rogue tests, handcrafted inputs, and what each test is for

*(The user, 2026-11-02, after a test helper run by hand wrote a fake machine record into their real
`~/.config/molbuilder/environment.json`: "detect those shit and rogue tests"; "give me a review of test design
and detection of rogue tests, and justify why they need to be designed".)*

**What was read.** Every Python file under `tests/` — 416 files (395 test files, the conftests and the
`tests/support` helpers), about 133,000 lines — read whole by ten read-only reviewers, each with the brief in
[`brief.md`](brief.md). Nothing was run. Every headline finding below was then checked in the code by hand
(marked ✔). The per-file tables — kind, contract pinned, how inputs are made, what it reaches, what it
handcrafts, verdict and the failure it guards — are the group reports: [`H_harness.md`](H_harness.md) (conftest
and `tests/support`), [`E1.md`](E1.md), [`E2.md`](E2.md) (the 40 end-to-end files), [`A1.md`](A1.md) …
[`A7.md`](A7.md) (the rest).

## 1. What happened on 2026-10-30, 01:11

1. A test-fixer agent I had launched, briefed only "X3 may run fast tests, never e2e or the engine", wanted to see
   what a benchmark prep makes. It imported `tests/support/road.py::run_road_case` into a bare `python -` script
   and ran one row of `tests/data/prep_protocol.toml` ("a benchmark is prepared as any stage") by hand.
2. Outside pytest no isolation fixture runs. `run_road_case` → `_road_target` (`road.py:543`) →
   `conftest.write_machine_record` (`conftest.py:1173`) writes `machine_scope_path()` — wherever the config folder
   resolves — which was the user's real `~/.config/molbuilder/environment.json`. ✔
3. It wrote the suite's fake record: a workstation, 2×8 cores, 0 GPUs, activation `conda activate`, setup line
   (`preamble`) `true` — a shell command that does nothing — no machine name, no probe date. ✔
4. Every real run on the machine then failed to activate conda (`Run 'conda init' before 'conda activate'`)
   until the user re-probed on 2026-11-02.
5. The suite's canary (`the_suite_leaves_your_config_alone`) fingerprints the real config only at the start and end
   of a **pytest session**: a write made outside pytest is invisible to it.

## 2. How the suite is designed

| kind | where | what it does | how it is isolated |
|---|---|---|---|
| **API tests** | `tests/test_*.py` (tier 1, the fast batch) | call a door of molbuilder directly — a reader, a writer, a check — and assert on its result | function-scoped autouse fixtures in `tests/conftest.py` |
| **contract-table tests** | 10 files (`test_prep_protocol`, `test_launch_protocol`, `test_launch_values`, `test_gpu_contract`, `test_restart_files`, `test_held_atoms`, `test_the_deck`, `test_the_catalogue`, `test_machine_record`, `test_molbuilder_json`) over `tests/data/*.toml` | each row: write a machine record (`_road_target`), `molbuilder describe` an H2 calculation, run `jobset prep`/`launch`/`probe` in-process through click's CliRunner, check the files and words made. No engine, no browser | the same function fixtures — and nothing in the helpers |
| **front-end tests without a browser** | `test_*_js.py` | run the page's JS modules in node, often against a typed stand-in server | none needed for the folders; the stand-ins are the problem (§ 4) |
| **e2e, browser** | `*_e2e.py` with Playwright | a live Flask server (`support/live_server.serve`, usually **module-scoped**) and a browser clicking the pages | module fixtures must call `outside_the_developers_folders` themselves |
| **e2e, real engine** | `*_e2e.py` marked `engine` | real SIESTA / PySCF runs through `describe → prep → launch` (`tests/_road.py`, `support/real_runs.py`) | `_road._live`: config and state folders only |
| **field** | `tests/field/` (tier 2) | run on the target machine against its real scheduler, read-only | by design they read the real record and scheduler |

**The isolation, as built.** `config_root_is_never_the_developers` (function-scoped, autouse) clears
`MOLBUILDER_CONFIG_DIR` and points `XDG_CONFIG_HOME` at a temp folder; `_this_machine_has_been_probed` then writes the
fake machine record **there**; `isolated_projects_root` is opt-in; an audit hook refuses writes into the repository's
`projects/`; the canary fingerprints the real config per session. **Nothing makes a writer refuse a real folder** —
every helper writes "wherever the scope currently resolves" (`conftest.py:1174`) and trusts its caller to be a
function-scoped pytest test. `HOME`, `XDG_RUNTIME_DIR`, `MB_NOTIFY_*` and the network are never isolated.

**Why the fake machine record exists.** Prep refuses a machine that was never probed (the user's rule: "no environment
json, error"). Instead of probing — which reads the real hardware and conda — the suite writes the probe's output by
hand for every test, with a setup line that does nothing, so the run scripts it renders enter nothing. It is the
foundation of the fast suite, and the shape of the record that broke the real runs.

## 3. Rogue tests — what reaches outside its sandbox

### 3a. Writes the real folders when a helper runs outside pytest, or from a wider-scoped fixture
All in the harness, all ✔; none refuses a real root:
- `conftest.write_machine_record` (1173) → `~/.config/molbuilder/environment.json` — the 10/30 writer.
- `road._write_machine_config` (1133) → replaces the whole `~/.config/molbuilder/molbuilder.json`, often with a
  deliberately invalid config.
- `road._road_target`, named rows (601-609) → `~/.config/molbuilder/environments/sol.json` — the user's real Sol target.
- `road._road_target`, "unprobed" rows (589-594) → **deletes** `~/.config/molbuilder/environment.json`.
- `road._road_target`, probe rows (571) → a real `jobset probe --write` into the config folder.
- `road.prepared_here` (235), `road.a_machine_with_queues` (343-356) → `environment.json`.
- `road.describe_calculation` (154-199) → `<given>/projects/...`; handed the repository, the real `projects/` tree.
- `road.left_in_the_background` (246-265) → `~/.local/state/molbuilder/jobs` and the real engine.
- `support/pages.py:62` → creates `~/.config/molbuilder/secrets/secret_key` when absent.
- Inline in tests: `test_target_machine_choice.py:48-67`, `test_task_setup_prep_e2e.py:51-67` (an autouse fixture writing
  a cluster record, safe only by fixture order), `field/test_ask_the_target.py:71-75`.

### 3b. Reaches out on every ordinary fast-batch run
- `test_notify_setup_api.py:692-707` — **a real HTTPS POST to `hooks.slack.com`** with a made-up webhook: the
  `post …/lab/test` row calls the channel's Test route, which calls `urllib.request.urlopen` (`notify_setup.py:477`),
  unpatched. ✔
- `test_envs_solve_drift.py:90` — a module-scoped fixture runs the real `mamba/conda create --dry-run -c conda-forge`:
  the network, the real `~/.condarc` and package caches, before any isolation. ✔
- `test_wheel_ships_the_front_end.py:32` — a module-scoped `pip wheel` with the real HOME (`~/.cache/pip`). ✔
- `test_envs_install.py:977-1013` — the real `install-env.sh list` with the real conda (reads).

### 3c. Reads the real config before any sandbox exists
- **Module-scoped live servers** — `serve()` → `create_app`, which loads the config folder's settings
  (`app.py:305`, `_initialize_diagnostics`) and the notify keys in `secrets/` (`app.py:453`) ✔ — in 23 files with a module-scoped server, the browser e2e files and
  `test_pages_no_js_errors.py`.
- **At collection** — `pytestmark = skipif(conda_hook() / env_available())` in 15 engine e2e files ✔
  (`test_vibration_e2e.py:29`); `support/real_runs._road_at` asks `_engine_here()` before isolating.
- Module fixtures in `test_transport_keywords_exist_in_the_binary.py:96` and `test_dna_double_strand.py:100`.
- Child processes keep the real HOME: the notebook e2e reads `~/.jupyter` and writes `~/.cache`; engine runs read
  `~/.condarc`, `~/.pyscf_conf.py`.
- `MB_NOTIFY_URL` is never cleared: set in the shell, monitor tests and real launches in e2e send real webhooks.
- SIESTA validation tests read the real `projects/pseudopotential/` library.

### 3d. One regression away from deleting or overwriting real things
- `test_envs_clean.py:103-115` — `clean --yes` runs `rm -rf`, aimed away from the real `molbuilder-siesta-gpu` build only
  by two `mock.patch` names. ✔
- `test_envs_install.py:1302-1346` — hands the real host env prefix (`sys.prefix`) to `install --clean --yes`.
- `test_serve_verbs.py:261`, `test_cli.py:557` — a real detached server / a bind on `0.0.0.0` if a refusal regresses.
- `test_envs_enters_the_env_through_the_manager.py:380` — `os.makedirs` under `/scratch/me/...` (created on a cluster).
- `test_envs_builds.py:697` — a real `git clone https://example.com/...` if a per-process flag is left set.
- Writes into the system `/tmp` outside `tmp_path`: `test_stage_resolution.py:50`, `test_serve_log_is_private.py:27`,
  `test_checkpoint_sensor_js.py:262`, `watch/test_api_load.py:78`.

## 4. Handcrafted inputs — tests that type what molbuilder or an engine writes

1. **Machine records** — beside the conftest record, `environment.json` typed as raw JSON in
   `test_target_machine_choice`, `test_task_setup_tab`, `test_prep_from_the_browser` (with a fake probe date),
   `test_value_axes`, `test_prep_bench_fold`, `test_bench_grid_card`, `test_launch_door_gate`,
   `test_warm_file_inventory`, and `test_engine_offset_reaches_every_deck._prep`, which other files import
   (`test_electronic_state`, `test_k_point_mesh`, `test_what_a_kind_offers`, `test_fixed_and_shared_items`).
2. **Records of runs that never happened** — `road._road_launched` (seven `prep_protocol.toml` rows),
   `test_a_sent_run_the_scheduler_forgot`, `field/test_a_forgotten_job_is_written_down`, a planted `.runwrap.log`
   (`test_launch_door_gate`), structures citing calculations nothing ran (`test_electronic_state`, `frame_sets.toml`,
   `test_modify`, `test_structure_authority_roundtrip`).
3. **A calculation's own records typed instead of `molbuilder describe`** — `task.json`, `job-request.json`,
   `job-set.json` in `test_hard_limits`, `spectra/test_config`, `test_build_e2e`, `test_jobset`,
   `test_prep_calculation`, `test_checkpoint_sensor_js`, `test_task_setup_cell_types_e2e`.
4. **Older releases' files built by hand** (testing.md § 6 forbids it) — `test_prep_calculation`,
   `test_molstruct_json` (v2–v9), `test_script_emit` (v3 block), `test_junction_sidecar_roundtrip`,
   `test_task_description`, `test_runfile_names` (`-run0`), `test_fcc_lattice_table`, `parse/test_round2_fixes`,
   `test_geometry_pin`, `test_periodicity_gate`, `spectra/test_types`, `test_reserved_label_one_store`.
5. **Engine and tool outputs typed** — decks as strings (`parse/test_contract`, `test_bench_summarize`,
   `test_deck_helpers`, `test_script_emit`), typed conda/mamba JSON (`test_diagnostics`, `test_envs_*`), typed
   `nvidia-smi` and cgroup text, a typed `util.csv`, typed PSML for every element (conftest) though real H/C/O
   pseudopotentials are checked in, fake PySCF/geomeTRIC payloads (`test_molwatch_emitter`).
6. **Server replies typed** — stand-in servers in the JS tests (`test_molview_model.SERVER`, whose own comment records
   that it once drifted from the real route; `test_chemistry_module_js`), `page.route` fakes in browser tests.
7. **Results invented** — `spectra/_helpers.py` and `spectra/test_blueprint` build spectra with made-up versions and
   energies.

**Text-pinning** (testing.md § 3a) — `test_web_files.py:797`, `test_format_fetch_error_js.py:61`,
`test_envs_install_plan_executes.py:187`, `test_docs_tab.py:147`, `test_task_setup_save_bar_js.py:31`, the
`restart_files.toml` rows that pin the run script's shell spellings, among others in the group tables.

## 5. Verdicts — every file

| verdict | files | meaning |
|---|---:|---|
| keep | 288 | earns its place; 60-odd of these have single tests or rows to retire, named in the group tables |
| fix isolation | 56 | right test, wrong reach (§ 3) |
| rewrite through the road | 50 | right question, handcrafted input (§ 4) |
| retire | 22 | guards no failure a person would meet, duplicates another, pins text, or never runs |

Each file's one-sentence justification — the failure it guards, or "none found" — is in its group table.

## 6. How it came to be

1. **Isolation by context, not by construction.** Safety lives in function-scoped fixtures; the writers resolve the
   folder at call time and accept it. Module- and session-scoped fixtures, collection-time code, child processes
   and any script outside pytest all run without it. The canary only reports, and only per pytest session.
2. **One fake record became the foundation.** Faking the probe once, in conftest, was simpler than probing. Each test
   that needed another machine then typed another record — 20 test files write a machine record through the
   harness's writers, and more type `environment.json` as raw JSON (§ 4.1).
3. **A long road was cut short by typing the file at its end.** Where the real road (describe → prep → launch → engine)
   was long, tests typed the intermediate file — `task.json`, `run.json`, a deck, a server reply. Each cut saved a
   step; together they test molbuilder against its own typed imitations.
4. **Older-release files were typed to test migration readers**, against § 6.
5. **I wrote and kept many of these**, and in this very release updated two of them (`test_launch_door_gate`,
   `test_warm_file_inventory`) without challenging the hand-written machine record they carry.
6. **My agents ran with no guardrails**: no rule against running test code outside pytest, no sandboxed config
   folder, the full access of the user's account.

## 7. What closes it

1. **Guards, by construction** — every writer in the harness refuses a folder that is not under a temp root, naming the
   real one it was handed; `tests/support` and `conftest` refuse to import outside pytest; the whole session runs in a
   sandbox home (`HOME`, `XDG_*`, `MOLBUILDER_PROJECTS`, `MB_NOTIFY_*` set before collection), so module, session and
   collection-time code and every child process are covered; the basic suite refuses outbound connections other than
   loopback. Then the § 3b and § 3d tests point at temp targets.
2. **Retire** the 22 files and the named single tests.
3. **Rewrite the 50 through the road** — the plan's F9, *Tests through the road*.
4. **Agents** — every brief carries a fixed rules block (read-only reviewers; tests only through `tools/testrun.py`;
   no import of test code outside pytest); the user's Claude Code settings can sandbox Bash writes away from
   `~/.config/molbuilder`.
