# R2-14 — the test harness: one door per fact, one data shape, nothing copied

Slice: `tests/support/` (every module), `tests/conftest.py` (the only conftest
in the tree — `tests/validation/` and the other subfolders have none any more),
`tools/testrun.py` (and `tools/progress_plugin.py`, which it reads), 
`docs/process/testing.md`; then the preludes of every test file.

## What I read, and how I sampled

- **Whole:** every `tests/support/*.py`, `molview_host.mjs`,
  `molview_dom_standin.js`; `tests/conftest.py` (1567 lines);
  `tools/testrun.py`; `tools/progress_plugin.py`; `docs/process/testing.md`
  (1229 lines); the harness modules that live outside `support/` —
  `tests/_road.py`, `tests/_node_esm.py`, `tests/_deck.py`,
  `tests/spectra/_helpers.py`, `tests/validation/_helpers.py`; the harness's own
  tests `test_testrun_reports_only_what_it_measured.py`,
  `test_missing_test_tooling.py`, `test_the_suite_stays_in_its_sandbox.py`.
- **In part:** `molview_3dmol_standin.js` (lines 1–60 and 160–218 read; the
  `$3Dmol` viewer body between them only searched for the globals it sets);
  `test_no_test_is_shadowed.py` and `test_negative_body_assert_lint.py` (their
  file walkers only).
- **Preludes of all ~330 test files** (tests/, parse/, spectra/, validation/,
  watch/, field/): swept by search, not read body by body — every
  `@pytest.fixture` name defined in a test file (then grouped by name and each
  repeated one read), every `create_app(`, `make_server`, `serve(`,
  `CliRunner()`, `subprocess.run([node`, `write_environment(`,
  `named_environment_path(`, `PSML_DIR`/`.psml` copy, `sys.path.insert`,
  `lifted(`, `read_text()` of a shipped `.js/.css/.html/.py` or a doc,
  `ast.parse`/`inspect.getsource`, and the typed names `"task.json"`,
  `"job-request.json"`, `"molbuilder.json"`, `"job-set.json"`,
  `"frozen_atoms"`, `"pseudopotential"`. Each hit that looked like a copy was
  opened and read around the hit; the product side of every finding was read
  at the cited line.
- **Source/doc greps:** no surviving test asserts on a shipped file's text
  outside § 3a's allowed list (`test_xss_audit`, the lints over the test suite).
  What remains is reading shipped source *to run it* — `lifted` and one hand
  copy of it (findings 21, 22).

24 findings. The first seven matter most.

---

## 1. Two Python copies of "how a page draws the server's words", and they disagree (high — read all three sides)

- `tests/support/findings.py:21-24`
  ```python
  def _said(issues) -> list:
      # A command in a message is drawn as code, its backticks gone.
      return [[i["severity"], i["message"].replace("`", ""), ...
  ```
- `tests/support/pages.py:32-37`
  ```python
  def drawn(said: str) -> str:
      return said.replace("``", "`").replace("`", "").replace(" -- ", " — ")
  ```
- Owner: the page itself, `molbuilder/web/static/lib/dom.js:41-47` (`_parts`,
  read by `wordsText`/`codeWords`): `"``"` → `` "`" ``, and in the said parts
  `/ -- /g` → `" — "`. The findings renderer draws through it
  (`lib/validation-findings.js:116`, `dom.codeWords(words)`).
- What goes wrong: `assert_every_finding_drawn` (used by
  `test_build_e2e.py:487,495`, `test_spectrum_form_e2e.py:152`,
  `test_transport_on_a_real_junction_e2e.py:1223`) expects the message with
  its ` -- ` kept, while the page draws ` — `. A finding such as
  `validation/siesta.py:532` ("… no point -- set it …") or `:629` makes the
  row-for-row comparison fail: "said and not drawn" / "drawn and not said" for
  the same row. Same for a message carrying ``` `` ```.
- Fix: `_said` calls `pages.drawn` (one Python mirror), or better, the page
  script in `_DRAWN` compares against `molbuilder.dom.wordsText(message)` so
  the only copy is the page's own.

## 2. The suite's refusing scheduler is a hand list that leaves out `scancel` and `squeue` (high-medium — read both sides)

- `tests/conftest.py:427` — `for name in ("sbatch", "srun", "salloc"):` (each
  stub refuses, so "a scheduler this suite can reach is a job it can submit"
  never happens).
- Owner: `molbuilder/scheduler/kinds.toml:33-37` — `[slurm.commands] submit =
  "sbatch", query = "squeue", cancel = "scancel", cluster = "scontrol"`, read
  by `scheduler.kinds.kind("slurm").commands`; `support/machines.py:22-28` builds
  `SLURM` from exactly that table, with bare names found on PATH.
- What goes wrong: a basic test with the `SLURM` record that asks the queue or
  cancels (`scheduler/queue.py:184-198`, reached from `jobs_cli.py:155`) runs
  the real `squeue`/`scancel` on a cluster login node — the first breaks
  tier 1's rule that no scheduler's answer is read there (`testing.md` § 0),
  the second can cancel a real job by id. Also `support/road.py:675-679`
  `_submits` tests `Path(word).name == "sbatch"`, a second spelling of
  `SLURM.submit` in a module that already imports `SLURM` (`road.py:49`).
- Fix: stub every command of `kind("slurm").commands` (plus `srun`, `salloc`);
  `_submits` compares with `Path(SLURM.submit).name`.

## 3. A test-only reader of the TBtrans window that is more lenient than the product's (medium-high — read both sides)

- `tests/_deck.py:45-70` `fdf_energy_window` — reads `from V unit to V unit`
  and **converts units** through `molbuilder.units`; its own test file
  `tests/test_deck_helpers.py` feeds it typed decks.
- Owner: `molbuilder/transport/energies.py:103-116` `deck_window(text)` — "the
  grid a transmission deck describes": reads tokens `[1]` and `[4]` of the
  `from` row as eV, no unit read.
- What goes wrong: `test_transport_decks_e2e.py` asserts through the lenient
  reader. If the writer ever wrote the window in Ry, the test converts and
  passes while `deck_window` reads the Ry figures as eV, so the product's
  energy-sharing check (`transport/reuse.py:1121-1137`) compares the wrong grids.
  The test helper even says it exists to catch "a deck that switched unit"
  — but the product's reader would not catch it.
- Fix: the e2e asserts through `transport.energies.deck_window`. Retire
  `_deck.fdf_energy_window` and `test_deck_helpers.py`. If a unit check is
  wanted, it belongs in `deck_window`.

## 4. The plain test app is built 84 times in 55 files beside `web_client` (medium — counted)

- Owner: `tests/conftest.py:846-866` `web_client`.
- Copies, as fixtures: `test_structure_analyze_endpoint.py:24-27` (`web`),
  `test_lattice_from_run.py:33-36`, `test_security_headers.py:14-17`,
  `test_transport_blueprint.py:14-17`, `test_docs_tab.py:19-22`,
  `test_generated_structures_are_labelled.py:19-22`, `test_tab_routes.py:22-26`,
  `test_checkpoint_routes.py:37-41`, `test_bench_grid_card.py:92-95`,
  `test_slab_route.py:14-17`, `test_structure_envelope_protocol.py:45-48`,
  `watch/test_api_load.py:27`, `test_seam.py:181-184`,
  `test_periodicity_gate.py:520,938,1087,1555,1658` (five class fixtures),
  `test_the_path_fence.py:61-63`, `test_notify_listener.py:631`,
  `test_the_viewers_follow_a_real_run_e2e.py:94-98`; inline
  `create_app(config={}).test_client()` in ~35 more files; and in the support
  door itself, `support/road.py:629` (`save_both`) and `:1147`
  (`_road_both_doors`).
- Two spellings of the same app: `config={}` (rate limit on) and
  `{"rate_limit": {"enabled": False}}`. The difference does nothing for a test
  client — `127.0.0.1` is on the limiter's default allowlist
  (`web/rate_limit.py:131`) and `threshold_total` is 0 — so the reason given at
  `conftest.py:860-864` and `testing.md` § 5 (lines 851-853) is out of date.
- What goes wrong: a change to what the test app must be (a section a test app
  needs, a config that must stay off) has to be made in 55 files. Today they
  agree only because the limiter does nothing for them.
- Fix: `web_client` (plus an `app` fixture for the few that need the app
  object). Delete the copies. `save_both` and `_road_both_doors` take the
  caller's client.

## 5. Eighteen e2e files carry an exact copy of `app_at`; two more hand-build a Flask server and a page (medium — counted)

- Owner: `tests/conftest.py:286-295` `app_at` — module-scoped
  `serve()` over `isolated_projects_root_module`; `testing.md` § 2a: "One
  fixture, never a copy per file."
- Exact copies (`@pytest.fixture(scope="module") def flask_server(isolated_projects_root_module): with serve() as base_url: yield base_url`):
  `test_inspector_registry_e2e.py:29`, `test_molbuilder_e2e.py:44`,
  `test_layout_fits_e2e.py:84`, `test_task_setup_prep_e2e.py:130`,
  `test_docs_pane_theme_e2e.py:30`, `test_app_notifications_e2e.py:46`,
  `test_spectrum_form_e2e.py:18`, `test_task_setup_cell_types_e2e.py:86`,
  `spectra/test_spectrumchart_box_e2e.py:24`, `test_task_setup_one_folder_e2e.py:70`,
  `test_chemistry_card_e2e.py:25`, `test_results_file_picker_e2e.py:29`,
  `test_system_load_monitor_e2e.py:47`, `test_siesta_run_in_the_viewer_e2e.py:33`,
  `test_coming_back_to_the_tab_e2e.py:37`, `test_vibrationview_e2e.py:16`,
  `test_build_e2e.py:65`, `test_results_export_e2e.py:51`. The deferral is
  acknowledged at `support/live_server.py:10` ("still start their own until
  F10 runs them").
- `test_form_schema_diff_js.py:36-60` and `test_form_schema_setvalues_js.py:33-70`:
  the same hand fixture twice (a bare `Flask`, `make_server`, a typed host page,
  a typed schema), which `testing.md` § 4 (L813-814) lists as stretch work.
- Fix: those files ask for `app_at`. The form-schema pair either asks the real
  route for its schema (as `test_validation_findings_js.py:101` does) or moves
  to `run_node`.

## 6. A named target's machine record has no door: 13 hand writers, each with its own defaults (medium — read the writers)

- Owner: `tests/conftest.py:1407-1446` `write_machine_record` — the "modest"
  defaults: workstation, 2×8 cores, **`gpus_per_node=0` stated** ("an unstated
  count is unknown"), `env_init={"activation": "conda activate", "preamble":
  "true"}`. It takes `at=` (a directory) but no target name.
- Hand writers of a named record: `support/road.py:924-932` ("sol": SLURM,
  `Topology(sockets=2, cores_per_socket=24)` with **no GPU count**, its own
  `env_init`); `support/real_runs.py:427-434` ("sol", copied from this
  machine's record); `test_target_machine_choice.py:58-63`;
  `test_jobset.py:170-176`; `test_task_setup_tab.py:1086,1134-1136`;
  `test_prep_from_the_browser.py:54`; `test_task_setup_prep_e2e.py:1287`;
  `test_transport_on_a_real_junction_e2e.py:885-886`; `test_value_axes.py:50`;
  `test_bench_grid_card.py:243`; `test_routing_domains.py:200`;
  `test_machines_listing.py:57-63`.
- The `env_init` literal is typed in 11 places (`conftest.py:1436`,
  `road.py:877-878`, `road.py:929-930`, `_road.py:89-91`,
  `real_runs.py:672-673` — which repeats what `_road._live` already wrote —
  `test_value_axes.py`, `test_task_setup_prep_e2e.py`, `test_jobset.py`,
  `test_auth_setup.py`, `test_seeding_a_fresh_machine.py`).
- What goes wrong: a rule about what a probed record must say (a GPU count
  stated, a new required field) changes one writer, and twelve others keep
  writing the old shape. Rows that prep for "sol" run on a record nobody
  states the GPU count of.
- Fix: `write_machine_record(name=…)` (or `at=` taking a name) with the same
  defaults, and every hand writer calls it.

## 7. The machine's `molbuilder.json` has four writers; two type its name and place (medium — read all four)

- Owner: `molbuilder/runtime_config.py:1309` `machine_config_path()` (and
  `CONFIG_FILENAME`, `:71`).
- `tests/conftest.py:195-198` (`machine_config`): `config_root / "molbuilder.json"`;
  `conftest.py:226-229` (`checkpoint_config`): `home / "molbuilder.json"`;
  versus `support/road.py:1589-1598` and `test_checkpoint_config.py:39-44`,
  which both ask `machine_config_path()`. And `test_checkpoint_config.py:39-44`
  copies `checkpoint_config`'s own setter in the same file.
- `checkpoint_config` (conftest.py:202-230) is `config_root` + `machine_config`
  again, plus its own HOME move. `test_runtime_config.py:24-37` and
  `test_cli_tls.py:30-41` each carry a copy of `config_root`, and both
  docstrings say so ("`conftest.config_root` is the general form").
- What goes wrong: if the config file's place or name moves
  (`configuration.md` § 2.1), the two fixtures write a file nothing reads.
  Their own docstring warns about exactly this: "A write into a file nothing
  opens still succeeds, and a green suite cannot show you that."
- Fix: one fixture that points the config root and returns a writer through
  `machine_config_path()`. `checkpoint_config` becomes that writer with a
  `{"checkpoint": …}` body. The two per-file copies are deleted.

## 8. File names and the deck's suffix typed in the support doors where `runfiles` names them (medium — read both sides)

- `"job-request.json"`: `support/road.py:551,557,1013`; `"task.json"`:
  `road.py:974,1013,1082,1101,1205,1547,1582` — owner `runfiles.JOB_REQUEST_FILE`
  / `TASK_FILE` (`runfiles.py:1103-1104`). The same module asks
  `runfiles.TASK_HANDOVER_FILE` at `road.py:401`.
- `"bench/job-set.json"`: `road.py:1922` (and `"bench"` at `road.py:1662`) —
  owner `paths.BENCH_DIRNAME`/`bench_container` (asked at `road.py:1686`) and
  `runfiles.JOBSET_FILE` (`runfiles.py:1108`).
- `f"{task_of(run.bundle).label}.fc-sweep.json"`: `support/real_runs.py:826`. The
  owner is `runfiles.compose(label, ROLE_FC_SWEEP)` (`runfiles.py:96`), and the
  same module uses `compose(…, ROLE_SPECTRA)` at `real_runs.py:750-752`.
- The deck's suffix comes from an engine-name branch, twice and the other way
  round: `road.py:459` `".fdf" if engine == "siesta" else ".py"`;
  `road.py:1890` `".py" if case.get("engine") == "pyscf" else ".fdf"`; and
  `road.py:1749` `script.name.replace(".run.sh", ".fdf")`. The owner is
  `runfiles.deck_role(engine)` (`runfiles.py:2479`), and `stage_file`
  (`road.py:1696-1703`) already composes through `RunNames`.
- What goes wrong: a third engine, or a renamed record, makes every row read the
  wrong file. With the two branches written in opposite directions, a third
  engine gets `.py` from one and `.fdf` from the other.
- Fix: ask each owner. The suffix is always `deck_role(engine)`.

## 9. Vocabularies typed in the harness where a table holds them (medium — read both sides)

- The reserved held-atoms label `"frozen_atoms"`: `support/road.py:112,121,200,1068`,
  `support/junction.py:18,35`, `support/real_runs.py:529,957` — owner
  `structure.FROZEN_LABEL` (`structure.py:114`), which `support/envelope.py:44`
  and `spectra/_helpers.py:42` already ask.
- Transport labels `"L-electrode"`, `"R-electrode"`, `"bridge"`, `"buffer"`:
  `support/junction.py:16,33` — owner the transport kind's record, through
  `transport.sort.REGION_*` (`sort.py:84-90`), which `road.py:219` asks for
  `REGION_BRIDGE`. (`validation/test_transport_kind.py:34-41` types them too.)
- `road.py:1557` `k not in ("time", "domain")` — owner `job_request.LANE_ASKS`
  (`job_request.py:69`, "the two SCHEDULER asks a run owns"). If it grows, a
  row's stage key is moved into the description's overrides and Save refuses.
- `road.py:1210` `key in ("allocation", "execution", "bench", "notify")` — a hand
  copy of the request's blocks (`job_request._TOP_KEYS`, `:48`, private today).
- `road.py:296` default `psml_lib="pseudopotential"` and `conftest.py:888`
  `_INPUT_LIBRARY = ("pseudopotential",)` — owner `pseudos.CONVENTIONAL_LIBRARY`,
  which `road.library_of` (`road.py:256`) asks.
- Fix: import each from its owner (expose `_TOP_KEYS` or a public list if a
  test needs it).

## 10. The variables a scheduler tells a job are retyped (medium-low — read both sides)

- `support/real_runs.py:394-398` `_CARRIED` lists `MB_JOB_ID … MB_JOB_NODELIST`
  and `SLURM_JOB_ID … SLURM_JOB_NODELIST` by hand.
- Owner: `scheduler/kinds.toml:95-101` `[slurm.job]` and the molbuilder-name
  table in `scheduler/kinds.py:66-82`.
- What goes wrong: a job variable added to the kind is not scrubbed. A
  "dry run as a person's" then reads a value left over from the test's own
  shell.
- Fix: build the scheduler part from the kinds module (its MB names, and each
  kind's `job` values), and keep only the person's own counts (`OMP_NUM_THREADS`,
  `MB_NP`, …) typed.

## 11. `capture_on_fail` is dead harness, and it keeps a test hook alive in shipped code (medium-low — counted)

- `tests/conftest.py:1079-1107` (the `rep_<when>` stash) and `:1110-1187`
  (`_capture_on_fail`, autouse); `pyproject.toml:265` declares the marker;
  `testing.md` § 1 table and § 5 last bullet recommend it.
- No test carries `@pytest.mark.capture_on_fail` (searched all of `tests/`).
- Its page script reads `window.__molbuilder_modify_test`, which
  `molbuilder/web/static/modify/viewer.js:852` publishes in shipped code and
  nothing else reads.
- Fix: delete the fixture, the stash, the marker and its two doc mentions —
  or say which test needs it. The product hook is then for nothing; it is
  the Modify page owner's call.

## 12. Small dead or contradicting branches in the harness (low — each read)

- `conftest.py:1468-1472` `_isolated_workspace_store`: `except Exception: yield
  # no flask in this env`. This contradicts § 3 ("Flask … never skips") and
  `web_client`'s own docstring (`conftest.py:848-849`). The branch is dead.
- `conftest.py:1315` `if "_smoke" in fn or "_smoke_l4" in fn:` — the second
  clause is covered by the first.
- What makes a file e2e is stated in two places: `tools/testrun.py:62-63`
  (`--ignore-glob=*_e2e.py` / `python_files=*_e2e.py`) and
  `conftest.py:1296-1297,1300` (`endswith("_e2e.py") or "_e2e_" in fn`, the
  test written twice in one hook). A file named `test_x_e2e_y.py` would be
  marked e2e and grouped whole, yet `none2e` would collect it and `e2e` never
  would. No such file exists today, so the `"_e2e_"` arm is dead.
- `tests/_road.py:70` `H_PSML` has no reader. Meanwhile
  `test_siesta_keyword_smoke_e2e.py:27-30` builds the same path again, with a
  comment that is no longer true ("`conftest.write_pseudos`'s PSML is enough
  for prep's screening but SIESTA does not start on it"): `write_pseudos`
  copies that very file (`conftest.py:1378`).
- `support/envelope.py:102-118` `from_file` has no caller. It also rebuilds the
  envelope from elements and positions and then copies four identity columns
  back by hand, while `Structure.to_dict()` (which the module says "IS the
  envelope") already carries them. `structure()` is used only inside the module.
- `conftest.py:869-874`: a section header ("Marker auto-application by file
  pattern") with nothing under it.

## 13. "A test never reaches the person's folders" is implemented three times, and they disagree on the state folder (medium-low — read all three)

- `conftest.py:79-95`: the import-time sandbox. `XDG_STATE_HOME` → `<sandbox>/state`;
  `testing.md` § 2b: "decided once, for the whole run … never per test".
- `conftest.py:640-716` `config_root_is_never_the_developers` (per test): a new
  HOME, `XDG_CONFIG_HOME` and `XDG_RUNTIME_DIR`, and **`delenv("XDG_STATE_HOME")`**
  (state goes under HOME). Its comments still argue against the developer's own
  `~/.config`, which the sandbox already rules out.
- `conftest.py:298-307` `outside_the_developers_folders` (module fixtures, used
  by `_road._live`): **`setenv("XDG_STATE_HOME", mktemp)`**.
- `conftest.py:216-224` `checkpoint_config` moves HOME once more and clears
  `XDG_CONFIG_HOME`.
- What goes wrong: three answers to where a launch writes its address
  (`config_dir.jobs_dir`), depending on which scope made the run. That is a
  trap for a test that reads, in its own scope, the jobs a module fixture
  launched. The comments also describe a protection that is not where they say.
- Fix: keep the sandbox as the rule. Reduce the per-test fixture to "a fresh
  config/state folder per test", in one layout that the module helper reuses.
  `checkpoint_config` keeps only its config writer.

## 14. Two harness pieces claim the same page globals with different shapes (medium-low — read both sides)

- `globalThis.__calls`: `tests/_node_esm.py:106` (`served`: each request,
  `{url, method, body, init}`) and `support/molview_3dmol_standin.js:20` (each
  drawing call, `{name, args}`). With both on a page (`test_molview_mount.py:51`,
  served first, then the stand-in), they share one array.
  `__resetCalls()` (`3dmol_standin.js:34`) wipes the request log;
  `molview_page.py:66-67` `loaded()` wipes the drawing log.
  `support/workspace_page.py:30-31` `writes()` runs `c.url.endsWith(…)` with no
  guard, so it throws once a drawing record is in the array. The mount test
  guards by hand (`test_molview_mount.py:439-443`, `c.url && …`).
- `globalThis.__makeHost`: `molview_3dmol_standin.js:190` (no arguments, a
  render-engine host) and `molview_dom_standin.js:289` (`(width, minWidth)`).
  Whichever loads last wins.
- Fix: give the stand-in its own record (`__drawn`, say) and one host maker,
  or two distinct names.

## 15. `tools/testrun.py` spells an env repair its own way (low-medium — sure)

- `tools/testrun.py:396-397` prints `python -m molbuilder envs repair molbuilder`.
- Owner: `molbuilder/envs/hints.py:27-31` `fix_cmd`, "THE ONE spelling of an env
  command", whose note warns that "a second spelling of the same fix … is how a
  reader ends up reading documentation". The product prints
  `bash scripts/install-env.sh repair molbuilder` (`envs/_cli.py:236`).
- Fix: print `fix_cmd("repair", "molbuilder")`.

## 16. The progress record has loose second spellings (low — read both sides)

- The teardown suffix `" [teardown]"` is written at `tools/progress_plugin.py:218`
  and matched at `tools/testrun.py:270` and in the harness test
  (`test_testrun_reports_only_what_it_measured.py:187,228`) — one constant
  from the plugin would do.
- Where progress lives: the plugin's default `--progress-file`
  `.test-progress.jsonl` (`progress_plugin.py:61`, at the checkout root) versus
  `testrun.PROGRESS_DIR` `.test-progress/` (`testrun.py:56`) and the write
  guard's allowed `.test-progress` folder (`conftest.py:996-998`). A manual
  `pytest -p tools.progress_plugin` without `--progress-file` should hit the
  write guard, whose `BaseException` the plugin's `except OSError` cannot catch
  (plausible, not run).
- The harness test hand-types the plugin's records (`_run`, L37-53) and stands
  in for pytest's report, config and argparse (`_Report` L203-208, `_Config`
  L269-273, `_Args` L155-157). The same file already drives the real entries
  (`testrun.main([...])`, a real crash in a small suite), and a real teardown
  failure can be caused the same way.

## 17. The signed-in session is typed in two shapes (low-medium — read both sides)

- `support/pages.py:81-82` `SIGNED_IN` (five keys, "as its login stores
  them") versus `{"email": …}` in `test_rate_limit.py:401,553,563,579,596,616,804,832`
  and `test_admin_reload.py:56`, and `{"email", "name"}` in
  `test_notify_setup_api.py:1034`.
- Owner: `molbuilder/web/auth.py:400-443` `authenticate()`, the one writer of
  `session["user"]`.
- Fix: one support door that signs a person in through `authenticate` inside a
  request context (or at least one shape, `SIGNED_IN`).

## 18. Per-file copies of doors `support/` already has (low-medium — each opened)

- `molbuilder jobset …` runners: `test_machines_listing.py:68`,
  `test_projects.py:319`, `test_jobset_echoes_its_config_root.py:30`,
  `test_scheduler_probe.py:67` — copies of `support.road.jobset` (`road.py:74-81`).
  `molbuilder …` runners: `test_siesta_relax_run_e2e.py:601` (identical to
  `road.molbuilder`, `road.py:52-58`), `test_electrode_pair_seam.py:44`,
  `test_seeding_a_fresh_machine.py:338`, `test_checkpoint_cli.py:33-36,104-107`.
  (`road.molbuilder` takes no `env=`; giving it one would cover these.)
- `test_notebook_through_one_port_e2e.py:85-99` `_until` beside
  `support.road.until` (`road.py:472-479`).
- Pseudopotentials: `validation/test_siesta.py:333,339,359,371,389`
  `shutil.copy(PSML_DIR / f"{el}.psml", lib)` is `conftest.write_pseudos`
  (`conftest.py:1363-1381`). `support/real_runs.py:477` writes pseudos into a
  library that `_road.live_siesta` (`_road.py:109-112`) has already filled:
  two doors placing the same files.
- "PATH without the folders that hold X", three times: `support/machines.py:69-71`,
  `support/road.py:883-886`, `test_missing_test_tooling.py:168-171`.
- The same `tree` fixture in `test_projects_public_surface_js.py:56` and
  `test_projects_api_envelope_js.py:43`.
- `tests/field/`: `RECORD = Path(os.environ.get("MOLBUILDER_FIELD_RECORD", "")).expanduser()`
  in four files (`test_a_forgotten_job_is_written_down.py:35`,
  `test_a_probed_record.py:28`, `test_ask_the_target.py:34`,
  `test_the_queue_says_why.py:27`), while `conftest.py:70-72` already expands it.
- `sys.path.insert(0, <tests>)` in `test_web.py:12`, `test_periodicity_gate.py:968`,
  `test_envs_install.py:1134`; `conftest.py:42` does it for every file.
- `_node_esm` is imported under two module names (`from tests._node_esm` ×47,
  `from _node_esm` ×5), so it can load twice.
- The shipped `static/lib` path is computed in 60 test files and again in
  `support/molview_page.py:33-34` and `support/workspace_page.py:20`. The
  checkout root is computed at `conftest.py:66` and `:1135`, `testrun.py:55`,
  `molview_page.py:33` and `real_runs.py:353`. The harness offers no constant
  for either.

## 19. Water and H2 built again and again (low — read each)

- Water: `conftest.py:803-818` (`water_structure`), `support/road.py:135-137`
  (`WATER`), `road.py:165-173` (`water_in_vacuum`), `spectra/_helpers.py:20-28`,
  `test_engine_offset_reaches_every_deck.py:50-55` (`water_in_vacuum` again,
  vacuum 5 instead of 8), `validation/test_siesta.py:267-270` (in a file that
  also uses `water_structure`).
- H2 as a `Structure`: the same fixture in `test_stage_resolution.py:24` and
  `test_template_roundtrip.py:24`.
- Fix: the road's makers, with a vacuum argument; drop the per-file copies.

## 20. Harness doors living outside `tests/support/`, plus a second reader of `nvidia-smi -L` (low — read)

- `tests/_road.py` (`PSML_DIR`, `live_siesta`/`live_pyscf`, `env_available`,
  `gpu_visible`) is asked by `support.tooling.unmet` (`tooling.py:81`),
  `support.real_runs.road_at` (`real_runs.py:233`) and `conftest.write_pseudos`
  (`conftest.py:1376`). `tests/_node_esm.py`, `tests/_deck.py`,
  `tests/spectra/_helpers.py` and `tests/validation/_helpers.py` are also
  shared helpers.
- Plain functions in `conftest.py` — `write_pseudos`, `write_machine_record`,
  `point_the_one_door_at`, `outside_the_developers_folders` — are imported back
  by support (`road.py:273,389,499,657`, `real_runs.py:234,470,667`,
  `pages.py:129`, `_road.py:83-86`), so conftest is both the hook file and a
  library. `testing.md` § 8 ("the shared doors, each in one place") names
  neither `_road.py` nor these.
- `_road.py:55-64` `gpu_visible` reads `nvidia-smi -L` itself
  (`"GPU" in r.stdout`). The owner is `runtime_info.gpu_models`, "the listing
  read by its one reader" (`scheduler/record.py:607-617`).
- Fix: move them into `support/`, and read the listing through `gpu_models`.

## 21. Seven `*_js.py` files keep their own Node runner and DOM; one rewrites the shipped code before running it (low-medium — acknowledged in testing.md § 4)

- `test_page_busy_js.py:39-97` (its own commonjs bootstrap and hand-typed
  DOM), `test_structure_page_js.py:140-170`, `test_one_stop_list_js.py:183`,
  `test_tab_strip_js.py:78`, `test_projects_render_sidebar_js.py`,
  `test_structure_save_js.py`, `test_task_setup_save_bar_js.py:31-71`. They
  ask `_node_esm.node()` for the binary but skip `run_node`, and type a DOM
  each, while the shared DOM stand-in (`support/molview_dom_standin.js`) is
  already used outside MolView (`spectra/test_spectrumchart_seal.py:32`,
  `test_spectrumchart_handle.py:21`).
- `test_task_setup_save_bar_js.py:31-39` `_fns()` re-implements
  `_node_esm.lifted` (`src.index(...)`) and edits the shipped text before it
  runs: `body.replace("await ensureEditor()", "_cm")`.
- Fix: as the plan says, move them to `run_node`, with the shared DOM stand-in
  extended where it falls short.

## 22. Running shipped code by cutting it out of its file (low — a judgement)

- `support/molview_page.py:43` takes `WRITE_DELAY_MS` out of `ui-context.js` by
  string anchors. Nine test files lift about 45 functions the same way, the
  most in `test_the_viewers_follow_a_real_run_e2e.py`; others are
  `test_trajectory_transition_js.py`, `test_spectra_transition_js.py`,
  `test_task_setup_notify_js.py`, `test_bench_card_reads_the_metrics_python_writes.py`,
  `test_slab_orthogonal_choice_js.py` and `test_spectra_no_spectrum_sentence_e2e.py`.
- `testing.md` § 3b rule 1 asks for "the real library loaded (`run_node` with
  the module's own files)". A lifted function runs without its module, so a
  rename breaks the test while a change in how the module wires that function
  goes unseen.
- Fix (for the harness's own case): the module exports the delay, and the
  page script imports it.

## 23. "One wait bound" is not kept by every support door (low)

- `support/sidebar.py:19-24` `WAIT_MS`: "never a number of the test's own:
  every door here … waits by it. One bound, so a slow machine costs every test
  alike." Yet `support/pages.py:20` uses `timeout=120000`,
  `support/findings.py:27` `timeout=10_000` and `support/chemistry_card.py:21`
  `timeout=15000`.

## 24. Reasons in testing.md and testrun that no longer hold (low)

- `testing.md` § 6.1a item 3 and `tools/testrun.py:384-391` justify sorting a
  targeted set by folder with "fixture 'water_struct' not found, on
  `tests/validation/` files". No subfolder has a `conftest.py` now, and
  `water_struct` is defined nowhere.
- `testing.md` § 5 (L851-853) and `conftest.py:860-864`: the limiter being
  "disabled so it never trips the test client". It cannot trip it in any case
  (finding 4).
- `testing.md` § 2b says the scheduler's commands refuse; only three of the
  kind's commands do (finding 2).

## Where this stands

The support doors themselves (`road.described`, `real_runs`, `tooling`, `serve`,
`run_node`) are well defined. The recurring defect is the files and support
modules around them that still type what an owner already names, or keep a
copy of a door that exists. The three that change outcomes, not only
maintenance, are 1, 2 and 3.
