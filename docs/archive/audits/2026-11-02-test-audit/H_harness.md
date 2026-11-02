# H_harness: the test harness (`tests/conftest.py`, `tests/support/*.py`)

Read whole: `docs/process/testing.md`, `tests/conftest.py` (1277 lines), every file in
`files_H_harness.txt` (road.py 1488, real_runs.py 610, the other twelve whole), and the product
doors they write through: `molbuilder/config_dir.py` (config_dir, state_dir, runtime_dir,
ensure_private_dir), `scheduler/record.py` (machine_scope_path, environments_dir,
named_environment_path, write_environment), `runtime_config.machine_config_path`,
`projects.projects_root`, `web/app.create_app`, `web/auth._install_secret_key`, `tests/_road.py`
(`_live`, which real_runs.py goes through). The only `conftest.py` in the tree is `tests/conftest.py`.
Not on the list, so not reviewed: three JS stand-ins in `tests/support/`
(`molview_3dmol_standin.js`, `molview_dom_standin.js`, `seed_selection_store.js`).

## 1. Summary

**14 files: keep 11, fix isolation 3 (`conftest.py`, `support/road.py`, `support/real_runs.py`),
rewrite through the road 0 whole files (one mechanism inside road.py, `_road_launched`, needs it),
retire 0.**

**The core finding: no writer in the harness refuses a real root.** Each one writes "wherever the
scope currently resolves" (conftest.py:1174). It is safe only because the caller is a
function-scoped test that conftest's autouse fixtures have already pointed at temp folders. If a
script, a REPL, or a module/session fixture without `outside_the_developers_folders` calls one,
the product's own writer runs against the developer's own folders. `_REAL_CONFIG_DIR`
(conftest.py:245-246) is computed in the same file, and only the canary uses it. It is never used
as a guard.

**The 2026-10-30 damage, traced through the text.** `run_road_case` (road.py:1304) calls
`_road_target` (road.py:1308). That calls `a_machine_with_queues` (road.py:596), or
`write_machine_record(**record)` (road.py:598), and so `conftest.write_machine_record`
(conftest.py:1173). That writes to `machine_scope_path()` = `config_dir()/environment.json`
(scheduler/record.py:954-967). With no fixture run, `config_dir()` is `~/.config/molbuilder`
(config_dir.py:100-104). `write_environment` replaces the file atomically with the suite's
default: workstation (or SLURM with a submit command in a temp folder), 2 sockets × 8 cores,
0 GPUs, env_init `{"activation": "conda activate", "preamble": "true"}`, no `machine`, no
`detected_at` (conftest.py:1198-1212). That is exactly "setup line `true`, no machine, no probe
date". Importing conftest outside pytest (road.py does it with `from conftest import
write_machine_record`) turns nothing on. The audit hook is installed (845-846), but
`_WATCHING[0]` stays False (787, 849-858), and the canary is a fixture.

### ROGUE findings (file:line, then the real folder it reaches when called outside pytest or from a non-isolated module/session fixture)

| # | where | what it does | real folder reached |
|---|---|---|---|
| R1 | conftest.py:1173-1212 `write_machine_record` | Writes the machine record. No refusal. `ensure_private_dir` creates the folder if missing (1211). | `~/.config/molbuilder/environment.json`, replaced (or `<at>/environment.json`) |
| R2 | road.py:1133-1142 `_write_machine_config`, called at road.py:554 and road.py:1129 | Replaces the whole `molbuilder.json`. On probe rows the content is `{"env_init": {"activation": "conda activate", "preamble": "true"}}`. | `~/.config/molbuilder/molbuilder.json` |
| R3 | road.py:571 (after R2) | Runs a real `jobset probe --write [--yes]`, which copies that env_init into the record. This is the 2026-09-11 road that conftest.py:541-548 recounts. | `~/.config/molbuilder/environment.json` or `environments/<name>.json`, carrying preamble `true` |
| R4 | road.py:589-594 (`machine = "unprobed"`) | `Path(machine_scope_path()).unlink()` deletes the real record. | `~/.config/molbuilder/environment.json` |
| R5 | road.py:601-609 (`machine = "named"`) | Writes a fabricated SLURM record (2×24 cores, preamble `true`) as `sol`. | `~/.config/molbuilder/environments/sol.json`. On this machine `sol` is the real ASU Sol target. |
| R6 | road.py:235-236 `prepared_here` | `write_machine_record(env_init={"activation": <caller's>, "preamble": "true"})` | `~/.config/molbuilder/environment.json` |
| R7 | road.py:343-356 `a_machine_with_queues` | Writes a record naming a SLURM submit command in a temp folder. | `~/.config/molbuilder/environment.json` |
| R8 | road.py:154-199 `describe_calculation` | Writes `<tmp_path>/projects/…` and chdirs. It never checks that `tmp_path` is temporary. | With the repository root as `tmp_path`: the real `projects/` tree, and typed fakes over `projects/pseudopotential/<El>.psml` (see R11) |
| R9 | road.py:246-265 `left_in_the_background` | A real `launch --mode direct --yes`, whose job address goes into `state_dir()/jobs`. Outside pytest no engine tripwire is on PATH, so the run script runs the real engine. | `~/.local/state/molbuilder/jobs`, and a real engine run |
| R10 | road.py:89-97 `each_is_taken` | Types back every printed `molbuilder jobset …` line. Only launches get `--dry-run`, so a printed `probe --write` remedy writes a record. | `~/.config/molbuilder/environment.json` |
| R11 | conftest.py:818-825 (the projects guard) with conftest.py:1123-1147 `write_pseudos` | The guard exempts `projects/pseudopotential/` for every event: open-for-write, rename, remove. testing.md § 2a says that folder "may be **read**". `write_pseudos` writes typed PSML wherever it is pointed. | `<repo>/projects/pseudopotential/`, unguarded even inside pytest, and `projects/*` is git-ignored (.gitignore:127), so the checkout canary cannot see it either |
| R12 | conftest.py:136-145 `outside_the_developers_folders` (the module/session rule) | Redirects only XDG_CONFIG_HOME and XDG_STATE_HOME and clears MOLBUILDER_CONFIG_DIR. HOME and XDG_RUNTIME_DIR stay the developer's, and no refusing `sbatch` is put on PATH: `the_suite_cannot_submit_a_job` is function-scoped (310-334). | Child processes read `~/.condarc` and `~/.pyscf_conf.py`. `runtime_dir()` = `$XDG_RUNTIME_DIR/molbuilder`, the dev server's pidfiles, which `ports_with_pidfile` globs and then sends real HTTP requests to (conftest.py:558-575). On a SLURM login node, the bare `sbatch` (machines.py:21; scheduler/kinds.toml:34) is the real scheduler. |
| R13 | real_runs.py:267-286 `_at_a_terminal` | A child `python -m molbuilder.cli` with `cwd=` the repository root (282) and the real HOME. The audit hook watches this process only. | Any relative write by the child lands in the checkout. |
| R14 | conftest.py:199-236 the conda/mamba stubs | Delegate everything except `activate`/`deactivate` to the real manager. The env canary only reports, and disarms itself when conda is slow (409-433). | The real `molbuilder-*` conda envs (it happened on 2026-09-13: conftest.py:397-402) |
| R15 | conftest.py:884-961 `_capture_on_fail` | Writes a persistent diagnostic, by design. | `<repo>/test-artifacts/fail-*.log` (git-ignored, .gitignore:131) |
| R16 | pages.py:62 (`create_app` with an `auth` section) | `init_auth` calls `_install_secret_key` (web/auth.py:116, 510-548), which creates a session key when none exists. | `~/.config/molbuilder/secrets/secret_key` (created if absent, never replaced). Today's callers are function-scoped tests, so a temp folder. |
| R17 | live_server.py:44-63 `serve` | Sets `os.environ[MOLBUILDER_PROJECTS]` directly (50), to a `tempfile.mkdtemp` under the system temp folder, outside pytest's basetemp (47; removed at 59-63). Outside pytest, a shell's own `MOLBUILDER_PROJECTS` is served as it stands. Every caller is module-scoped, so `create_app` (52) runs before any per-test isolation and reads the developer's `molbuilder.json` (read only). | The developer's projects tree if the shell names it; the real config (read) |
| R18 | real_runs.py:167-180 `_typed` / `_taken` | No isolation of their own. Safe only inside `_on_the_road`. Imported as private names by `test_transport_on_a_real_junction_e2e.py` (313, 878, 1318, 1397, 1484). `vibration()` asks `stage_status` (567) after the environment is restored. `_road_at` asks `_engine_here` (154) before it isolates. | The real config, read |

**Module/session/class-scoped callers today.** The wide-scoped fixtures that call these writers
are `test_siesta_stopped_run_e2e.py:79-106`, `test_vibration_e2e.py:125-140` and real_runs.py
(through `_road._live`, `_road.py:72-93`). All three call `outside_the_developers_folders` first,
so their config and state writes land in temp folders; the HOME, XDG_RUNTIME_DIR and `sbatch` gaps
of R12 still apply. No module or session fixture in the suite calls road.py's writers. All ten
`run_road_case` callers (test_gpu_contract, test_held_atoms, test_launch_protocol,
test_launch_values, test_machine_record, test_molbuilder_json, test_prep_protocol,
test_restart_files, test_the_catalogue, test_the_deck) are function-scoped parametrized tests given
`tmp_path` and `monkeypatch`.

So inside pytest, today, the writers land in temp folders. The hazard is a caller outside pytest,
and any future module fixture: nothing enforces the rule. testing.md never states it.
`outside_the_developers_folders` appears only in conftest.py, `_road.py` and the two e2e files.
§ 2a tells a fixture that re-isolates to call the unguarded `write_machine_record`.

For other groups to confirm: module fixtures that move only the projects tree
(`isolated_projects_root_module`) and still run product code against the developer's config, e.g.
`test_task_setup_prep_e2e.py:101-140, 414-450`.

### The isolation fixtures: what they cover, what they miss

| fixture (conftest.py) | scope | covers | misses |
|---|---|---|---|
| `config_root_is_never_the_developers` (507-583) | function, autouse | Clears MOLBUILDER_CONFIG_DIR; points XDG_CONFIG_HOME, HOME, USERPROFILE and XDG_RUNTIME_DIR at temp; clears XDG_STATE_HOME (so state follows the temp HOME) | Every module/class/package/session fixture, because pytest sets up wider scopes before it. Writes at import or collection time. A child given an environment built from scratch: with no HOME, Python and bash fall back to the passwd entry, `/home/qqing`. Any call outside pytest. |
| `_this_machine_has_been_probed` (1150-1170) | function, autouse | Writes the default record into the test's temp config | Not a guard: it is the writer of R1 |
| `outside_the_developers_folders` (136-145) | a function a wide fixture must call itself | XDG_CONFIG_HOME, XDG_STATE_HOME, MOLBUILDER_CONFIG_DIR | HOME, XDG_RUNTIME_DIR, the refusing `sbatch`. Nothing makes a module fixture call it, and testing.md does not name it. |
| `isolated_projects_root` / `_module` (148-158 / 118-133) | function / module, opt-in | `MOLBUILDER_PROJECTS` set to a temp tree | Config and state: a module fixture using only `_module` still reads the developer's config, and a `paths.projects` in the developer's `molbuilder.json` (projects.py:186-193) names a tree the audit hook does not watch |
| `the_suite_takes_no_projects_tree_from_your_shell` (447-469) | session, autouse | The shell's MOLBUILDER_PROJECTS, cleared for module and session fixtures as well | `paths.projects` from the real config wherever config is not isolated |
| `the_suite_cannot_submit_a_job` (310-334) | function, autouse | A refusing `sbatch`/`srun`/`salloc` first on PATH | Module/session setup, and children given a PATH of their own |
| `_the_engines_answer_nowhere` (586-601) | session, autouse | The engine tripwires on PATH for the whole session | Outside pytest |
| projects audit hook (845-881) | while a test runs | open/list/mkdir/rename/remove under `<repo>/projects`, by any spelling | Writes into `pseudopotential/` (818-825). Child processes. A relative `open`. Import or collection time, and outside pytest (`_WATCHING`). Any projects tree other than the repository's. |

### Can the session canary miss a write? Yes, in seven ways

`the_suite_leaves_your_config_alone` (conftest.py:472-504):

1. **A write outside pytest.** No session, so no canary. This is the 2026-10-30 case.
2. **A killed session.** SIGKILL or SIGTERM, an OOM kill, a closed terminal, or an xdist worker
   that dies (testrun passes `--max-worker-restart 0`, tools/testrun.py:390) never reaches the
   fixture's second half. testrun reports PARTIAL, and nothing compares the fingerprints.
3. **One folder only.** It watches `_REAL_CONFIG_DIR`, resolved at import from the shell's
   variables (245-246). It does not watch:
   - the state folder `~/.local/state/molbuilder` (jobs/, logs/, reports/);
   - the runtime folder;
   - `~/.conda`;
   - `<repo>/projects`.

   If the shell sets MOLBUILDER_CONFIG_DIR or XDG_CONFIG_HOME, `~/.config/molbuilder` itself goes
   unwatched. A child given an environment without them writes there.
4. **Regular files' bytes only** (249-262). An empty folder created, a mode change, an identical
   rewrite, or a write later undone in the same session are all invisible. Example: real_runs.py
   writes `sol.json` and unlinks it (390-407). If that reached a folder with no `sol.json`
   before, nothing shows.
5. **Writes before its first snapshot.** Module import, collection, or another worker writing
   before this worker's first test.
6. **It prevents and restores nothing.** The damage is already done when it reports, as an error
   in the last test's teardown at the end of the run.
7. **The neighbouring canaries are no stronger.** The env canary disarms itself on a slow
   `conda env list` (409-433). The checkout canary (`git status --porcelain`, 270-307) cannot see
   git-ignored paths (`projects/*`, `test-artifacts/`). It also cannot see a further change to a
   file that is already modified: 37 were in this session's starting `git status`, and their
   status line stays the same.

### The worst handcrafted inputs

- **The default machine record, on every test.** conftest.py:1150-1170 and 1198-1203:
  workstation, 2×8 cores, preamble `true`, no `machine`, no `detected_at`. testing.md § 0 tier 1
  sanctions machine records written in molbuilder's own format, but nothing marks this one as a
  test's, and it is exactly the record that broke real runs.
- **A launch record for a launch that never happened.** road.py:771-787 `_road_launched` uses
  `runrecord.write_launch` for rows naming `launched`: prep_protocol.toml at lines 218, 577, 589,
  597, 607, 616, 625. testing.md § 6 forbids "a launch record … written by hand", and tier 1
  forbids "a state the road never reaches".
- **Typed PSML for every element** (z-pseudo 1, valence 1, no projector block). conftest.py:1100-1109
  and 1123-1147. Every road prep uses it (road.py:180), while real PseudoDojo H, C and O are
  checked in (`tests/fixtures/psml/`, which `_road.live_siesta` uses).
- **A planted session** instead of a sign-in: pages.py:24-26 (`SIGNED_IN`).
- **A test-side regex over a molbuilder log**: real_runs.py:191-209 `told` reads `.monitor.log`
  with `\[NOTIFY\] \(stub\)` rather than an owner reader.

Not handcrafted: `_road_describe` writing `task.json` and `job-request.json` directly
(road.py:1126-1127), and `write_request`. Both are sanctioned by job-request.md § 7 ("a person may
edit either file"). They do bypass the Save door the same file uses everywhere else.

## 2. One row per file

| file | kind | contract it pins | how inputs are made | rogue | handcrafted | other defects | verdict | justification |
|---|---|---|---|---|---|---|---|---|
| tests/conftest.py | helper: fixtures and guards | testing.md § 0 (tripwire), § 2a, § 6.1a; configuration.md § 2.1c; running-a-job.md § 3.1 | Machine record by its writer with made-up contents; typed PSML | R1 (1173-1212), R11 (818-825, 1123-1147), R12 (136-145), R14 (199-236), R15 (884-961); canary gaps above. `prep_the_group_of` (1111-1120) preps against whatever config resolves. | Default record (1198-1203); `_PSML_FIXTURE` (1100-1109) | Canaries report and never prevent (472-504, 409-433, 289-307). The guard exempts writes to `pseudopotential/`. `outside_the_developers_folders` is not in testing.md. `the_suite_cannot_submit_a_job` is function-scoped. | fix isolation | Guards the silent failures that have happened: a test writing the developer's config (2026-09-11), a real env reinstalled (2026-09-13), the real projects store polluted (2026-08-21), a host siesta reached (2026-08-25). Keep it; its writers must refuse a real root. |
| tests/support/road.py | helper: the contract tables' one runner | testing.md § 6 (a contract's cases are a table, down the road), § 3a | Road: describe → prep → launch --dry-run in-process (CliRunner), Save through the Flask test client, a real probe; plus machine records by their writer | R2-R10 (554, 571, 589-594, 601-609, 235, 343-356, 154-199, 246-265, 89-97). A latent `sbatch` reach on "named" rows (1329-1330, 1434-1438) outside a function-scoped test. | `_road_launched` (771-787); typed PSML through `describe_calculation` (180); machine records (235, 354, 557, 598, 604-609; tier-1 sanctioned) | `_road_describe` skips the Save door its own file uses (1126-1127). `each_is_taken` types back a printed `probe --write`. | fix isolation; rewrite `_road_launched` rows through a real launch (e2e) or retire them | Guards silent drift of every table's decks, headers, refusals and ledger: the one runner ten contract tables use. Keep it; every writer must refuse a real config root. |
| tests/support/real_runs.py | helper: tier-3 real-engine session runs | testing.md § 0 tier 3, § 6 (a run is made on the road); plan § 5y | Road with the real engine, in the same session | Isolates itself for config, state and projects (`_road_at` → `_road._live` → `outside_the_developers_folders`), but R12's HOME, XDG_RUNTIME_DIR and `sbatch` gaps; R13 (`cwd` = repository root, 282); R18 (`_typed`/`_taken`, 167-180; 154, 567) | None of an output. Job-request blocks as a person edits them (322-329). `told` regex (191-209). | Private helpers imported by another file (R18) | fix isolation (minor) | Guards the science and the road on real output: each run made once a pass. Keep. |
| tests/support/live_server.py | helper: browser e2e server | testing.md § 2a, § 5 | The real app (`create_app`) on 127.0.0.1:0 | R17 (44-63) | none | `os.environ` set directly rather than through monkeypatch | keep | One server implementation for every browser file (a helper, no failure of its own). Never reaches port 8888. |
| tests/support/pages.py | helper: every served page | testing.md § 3a (ask what ships) | The app's own routes through the test client | R16 (62) | Planted signed-in session (24-26) | Asserts against the private `runtime_config._PROVIDER_KINDS` (56-59) | keep | Lets the artifact lints read every page as served, signed out and in. Callers are function-scoped. |
| tests/support/machines.py | helper: scheduler rows | configuration.md § 5 M-2; scheduler.md § 6a | `scheduler/kinds.toml` via `kind()` | Latent: the bare `sbatch` (21) is safe only where conftest's function-scoped refusing `sbatch` stands first (R12) | none | none | keep | One SLURM row and one workstation row from the product's own table, not spelled per test. |
| tests/support/junction.py | helper: structures built in source | none stated (docstring rationale) | The `Structure` constructor | none (`_isolated` chdirs to a temp folder, 181-184) | none | Imports private constants from a test module (`test_transport_compose._BRIDGE/_LAYERS_L/_LAYERS_R`, 42). Stale docstring (20-25): names `parse/test_sidecars.py`, `parse/test_round2_fixes.py` and "its spectrum sidecar", none of which use or exist here; omits test_transport_wizard.py and test_transport_au_bdt_au_validation.py; `_isolated` cites test_prep_calculation (183). | keep | Built fresh each run, so no stale fixture file. |
| tests/support/envelope.py | helper: request body | molview.md § 11.7 | The `Structure` constructor and the app's own readers | none | none | none | keep | One envelope builder; a hand-rolled XYZ parser would drift. |
| tests/support/findings.py | helper: browser | science/validation.md § 4.1 R2, R3 | The page's drawn rows | none | none | none | keep | Every finding drawn, row for row, compared as multisets. |
| tests/support/form_fields.py | helper: data | none stated; a reason per exempt field | – | none | none | An exemption reason names a private function, `prep._seed_trajectory_log` (26; exists at jobset/prep.py:1852) | keep | Lists, with reasons, the only fields exempt from "a form value reaches the deck". |
| tests/support/leftovers.py | helper: browser JS spy | none stated (scenario in docstring) | Page instrumentation | none | none | none | keep | Measures listeners and WebGL contexts left by reopening a viewer. |
| tests/support/replay.py | helper: same-pass run replay | testing.md § 6 (replay allowed for a run made in the same pass) | A real run from this session, output cut to a line (37-41, § 6-sanctioned) | Writes only under the caller's `tmp_path` | none | none | keep | Lets the monitor be tested on a real run as it grew. |
| tests/support/results_export.py | helper: browser | none stated beyond the person's steps | Page actions | Writes `<served projects root>/<name>.xyz` through the page (36), wherever the server is pointed | none | none | keep | One road for "save to project" from Results. |
| tests/support/task_setup_page.py | helper: browser | web/task-setup.md § 9a, § 9a.1 | Page actions | none | none | none | keep | Opens cards as a person does, after the folder read. |

## 3. Patterns: how the reach and the handcrafting crept in

**Isolation lives in the caller, not the helper.** The suite isolates by moving environment
variables in a function-scoped autouse fixture. The writers were then written as thin wrappers
over the product's own doors, with no idea whether they run in a sandbox. `write_machine_record`
says so in its own docstring: "wherever the scope currently resolves". So a helper is safe exactly
as long as someone else has already redirected the variables.

**A second, hand-applied rule for wide scopes.** Module and session fixtures are set up before the
function-scoped fixture. They need `outside_the_developers_folders`, which lives in one docstring
and three callers, is missing from testing.md, and leaves HOME, XDG_RUNTIME_DIR and `sbatch`
uncovered.

**The guards are canaries.** They fingerprint one folder before and after a session and report.
They cannot see a run that is killed, a write made outside pytest, or the state, runtime, conda or
`pseudopotential/` folders. The audit hook, the one guard that refuses, watches only the
repository's `projects/`, and exempts the one sub-folder a fake file is written into.

**The handcrafting came in as "write it with the owner's writer".**
- `write_environment` for a record no probe made.
- `write_launch` for a launch that never ran.
- A PSML typed to pass the screening.
- A session dict instead of a sign-in.

Each reads as "on the road" because it calls molbuilder's own writer. But it skips the event that
would produce the file. Its contents (no `machine`, no `detected_at`, preamble `true`) are the
very shape that, once it escaped the sandbox, prep could not tell from a real record.

Two ways out. Neither is applied here, and either is the user's call:
- A guard in conftest that refuses any path under `_REAL_CONFIG_DIR`, the real state folder or
  `<repo>/projects`, called by every writer above.
- Mark the suite's record so that the product refuses it.
