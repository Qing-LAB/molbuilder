# F12 fix 10 (the run script, envs and server process, 2.77.18, R3b) — the review

**Verdict: closed with 2.77.18.** Round 2's one-home items for the run script,
the monitor, the envs and the server process (`r2-3-runscript-monitor.md` F4–F8,
F10, F11, F14–F16, F18–F24, F26), `r1-B-runscript.md` U3, the install
messages R2 left, and R3a's left item for `runwrap`'s own `module:NAME` split,
each moved onto the door its chapter names. F1 (the two job bundles' own
lists) is R3b2, its own release *(the user, 2026-11-20)*; F2 and F9 shipped
earlier; F12, F13 are R4's; F17, F25 left as they are; F3 (the pipeline
log's order beside `template.SAID`) is R3c's, its owner's. C-U3 was already closed
by fix 3. Round 1: six readers (correctness and one-home for each of two
parts, notes and tests, a discipline reader); round 2: five new readers;
then two new, independent readers of every touched file whole; then one new
reader of the fixes made after them.

**What the review taught.** Mid-round the user said *"check for correctness
not for exact match"*: an answer equal to 2.77.17's proves nothing if 2.77.17
is wrong. Judged against the chapters, round 2 found what the byte-for-byte
comparisons had passed — a remedy naming the CPU recipe for a GPU env forced
with `prep --env`, a fallback for a queue's sizes that neither R3 nor the fit
states, and a monitor change that left `util.csv`'s GPU columns empty for a
job not given GPU 0. The monitor's GPU sampling was taken out whole *(the
user: "hardware gpu is fully decided in env and there is no business to get
involved"; a job's GPUs are SLURM's)*, and the last release's code read and
found correct. A round-1 fix (a re-probe keeping a row the probe never
wrote) stated a rule no chapter states; round 2 replaced it with M-6's one
question.

## Round 1 — six readers (correctness A/B, one-home A/B, notes and tests, discipline)

Lessons: a field added to a record type is read by every writer of that record, a re-probe's
merge included; a family's door is asked by every sibling in the modules the release touches,
not only the one the finding named.

| # | finding (readers) | verdict |
|---|---|---|
| 1 | the L1 machine record imports the L2 config reader for `ENV_INIT_KEYS` (corr-B, one-home-B, discipline) | fixed — the keys at L1 in `scheduler/record.py`; `runtime_config` imports them downward in its three readers |
| 2 | a re-probe drops a queue row that is no queue (corr-B, one-home-B, discipline) | fixed — `_probe_consent_merge` carries `unread_domains` over: a probe never measures one |
| 3 | `ROW_KEYS` inserted under `_NULLABLE`'s note (corr-B, one-home-B, discipline) | fixed — moved above it |
| 4 | `Domain.sizes`' note says "a record that lists none" (one-home-B) | fixed — "where no listed machine states it" |
| 5 | `serve status` says "no pidfile at" for a file that names no pid (one-home-B, discipline) | fixed — `serve_daemon.nothing_named`, asked by `_verified` and status; test added |
| 6 | R-L38 and job-system § 6.0a, and three callers' notes, omit "names no machine" (notes-tests, one-home-B) | fixed — the row, its way on, § 6.0a, `runs`, `joblist`, `summarize` notes |
| 7 | scheduler.md R2 does not say a row that is no queue is said (one-home-B) | fixed — one sentence before "Nothing is refused…" |
| 8 | run-reports § 2.6 lacks `CUDA_VISIBLE_DEVICES`; running-a-job § 3.3 does not name the monitor or UUIDs (corr-A, one-home-A) | fixed — a § 2.6 row; § 3.3's sentence |
| 9 | a test of the [N/A] reading depends on the shell's `CUDA_VISIBLE_DEVICES` (corr-A, discipline) | fixed — the test states its premise (`delenv`) |
| 10 | three new tests cite the wrong contract (notes-tests) | fixed — configuration § 4; project-layout § 2; installation § 6 |
| 11 | prep's GPU-env note's new wording untested (notes-tests) | fixed — a `gpu_contract.toml` row with `envs.siesta-gpu = my-gpu`; fails on the last release ("install my-gpu") |
| 12 | F20's test asserts a data table, not an outcome (notes-tests; testing.md § 3b) | fixed — the test retired; the row's removal stays |
| 13 | F24 raised the test count beside the older stale-pidfile test (notes-tests; testing.md § 3b) | fixed — the stale row asserts the file removed and nothing signalled; the older test retired |
| 14 | F22's test renders a stub spec, not the file prep writes (notes-tests) | left as is — the file tests the renderer's framework with a stub engine; `render_deck` is the one writer of both moments |
| 15 | the listing test's input (a listed machine with no cores) is no row the probe writes (discipline) | fixed — a listed machine with cores and no memory, the probe's case; still fails on the last release |
| 16 | F18's names still typed in `jobset/submit.py`'s walk and `record._slurm_in_a_job` (one-home-A, discipline) | fixed — interpolated from `kinds` (rendered bash identical) |
| 17 | the run script's stop and prep's note spell the install command two ways (one-home-A) | fixed — both `hints.install_cmd` (the stop gains `--yes`) |
| 18 | `restart_group`'s `None` branch, `_n_orbitals_est`'s empty-text branch, `_deck_restart`'s "cannot be read" (one-home-A, discipline) | fixed — removed |
| 19 | `_declaration`'s note omits the unknown name (one-home-A) | fixed — said |
| 20 | bootstrap's next steps hand `install` the env-name table; `recipe_of_category` claims "the one" (one-home-A) | fixed — bootstrap asks it; the note says a recipe's own data names itself |
| 21 | the hooks' placeholders built twice (one-home-A/B, discipline) | fixed — `builds._layout_substitutions`, asked by both |
| 22 | the install message spells the layout by hand (discipline) | fixed — `resolve_paths(spec, "$CONDA_PREFIX")` |
| 23 | `validate._layout` is a renaming wrapper (discipline) | fixed — removed; `resolve_paths` asked at the four places |
| 24 | `run_capture`'s "the one quiet run" while the curl and git probes run their own (one-home-B) | fixed — both ask `run_capture` |
| 25 | `serve_stacks_log` spells `serve-` by hand; the family constant's note is narrower than its use; three names inline the port (one-home-B, discipline) | fixed — `_with_port`; the stacks log from `SERVE_PIDFILES`; the note widened |
| 26 | the shepherd's marker bytes made twice; `jupyter` imports the verb twice (one-home-B, discipline) | fixed — `serve_daemon.SHEPHERD_MARKER` |
| 27 | the seeded `_env_init` comment lists the forms by hand (one-home-B) | fixed — from `ACTIVATION_FORMS` (identical text) |
| 28 | the plan keeps closed C-U3 in its open list (discipline) | fixed — removed |
| 29 | run-identity.md writes `--` in prose (corr-A) | fixed — `—` |
| 30 | runwrap's comment over the `kinds` import names only the claim (one-home-A) | fixed |
| 31 | the monitor reads the job's GPU list in Python beside the run script's bash (discipline) | left as is — both only READ the environment's `CUDA_VISIBLE_DEVICES`; which GPUs a job has is the environment's (the user, 2026-11-20: *"hardware gpu is fully decided in env and there is no business to get involved"*) |
| 32 | F4 unified only the verb; argv's `jupyter`/`--serve-port` and `b"serve"` spelled apart from cli (one-home-B) | left as is — outside F4's finding; no answer drifts today |
| 33 | `stop_by_pidfile`'s `what`/`marker` constant at its one caller (discipline) | left as is — the step states what it verifies, as `pid_state` does |
| 34 | a hand-written float `max_cores: 48.0` now shows `48` (corr-B, one-home-B) | left as is — the probe writes integers; a float is read as the number it states |
| 35 | an unknown hook placeholder is refused at the hook's write, after the build (corr-B, not raised) | left as is — only a recipe's author meets it; the test pins it |

## Round 2 — five new readers (correctness A2/B2, one-home A2/B2, notes and tests 2), judged against the chapters (the user, mid-round: "check for correctness not for exact match")

Lessons: a byte-for-byte comparison with the last release passed a GPU-column defect and a wrong remedy that reading against the chapters found; a round-1 fix (the re-probe carry-over) stated a rule no chapter states.

| # | finding (readers) | verdict |
|---|---|---|
| 1 | under `prep --env NAME` the run script's stop names the CPU recipe (correctness-A2, one-home-A2) | fixed — `recipes.recipe_of_env(name, caps)`: the recipe whose env it is, routed or forced; none for a person's own env (no install line); test added (fails on the last release) |
| 2 | a re-probe can never clear a queue row that is no queue; `--yes` keeps a row the probe never wrote (correctness-B2, one-home-B2) | fixed — the queue list as the record writes it is the one question (M-6); its "keep" keeps both lists, `--yes` takes the probe's; the row shown in `record.unread_row_words`, the listing's words; row added to `machine_record.toml` |
| 3 | `Domain.sizes` falls back to the largest figures where the chapter and the fit do not (one-home-B2, notes-tests2) | fixed — by R3 and `fit.fit`: listed machines that state cores, else the largest figures; intended change 1 withdrawn; the listing test re-pinned to R3 |
| 4 | the monitor's GPU sampling | taken out (the user: "none of that matters"; a job's GPUs are SLURM's); the last release's code read whole and correct; the change had broken `util.csv`'s GPU columns for a job not given GPU 0 (correctness-A2, one-home-A2) |
| 5 | the notebook's recipe found by a literal name (one-home-A2) | fixed — `recipe_of_category("jupyter")` in `jupyter.py` and the notebook route |
| 6 | `deck_note`'s banner and `Parameter.declaration`'s note contradict `_declaration` (one-home-A2) | fixed |
| 7 | `run_capture`'s and `nothing_named`'s notes overclaim (one-home-B2) | fixed — narrowed to their users |
| 8 | `joblist.asked_here`'s comment omits the unnamed copy (correctness-A2, one-home-B2) | fixed |
| 9 | the hook-placeholder rule is in no chapter (notes-tests2) | fixed — `ops/installation.md` § 6.4; the test cites it |
| 10 | the F6 and F24 tests cite code, not the chapter (notes-tests2) | fixed — R-L38 and preparing § 4; deployment § 1.0a and jupyter § 3.3 |
| 11 | the notebook's recycled-pid words untested (notes-tests2) | fixed — a third case, a live pid not ours: refused, kept, unsignalled |
| 12 | `test_known_tools_present` restates a data table (notes-tests2) | fixed — retired; the module note's gone `EXTENSION_TO_CATEGORY` and stray blank lines too |
| 13 | rank-count precedence pinned only by two e2e tests (notes-tests2) | to run before release — `test_the_road_on_real_runs_e2e.py::test_the_ranks_a_scheduler_grants_rule_the_run_said_by_name`, `::test_a_flag_on_the_run_script_still_overrides_and_says_so` |
| 14 | a GPU job under a scheduler: `mpirun` with no binding flags binds each rank to one core, so `-c 2` threads share it (correctness-A2; older than this release) | later — into the plan as an allocation check on Sol: a past or next Sol GPU run's session log shows each rank's cores (`molbuilder[rank i/n] … cpus=`); a change waits for that evidence |
| 15 | `SERVE_PIDFILES`/`JUPYTER_PIDFILES` names narrower than their use (one-home-B2) | left as is — the note says it; a rename touches five modules for no answer |
| 16 | `jupyter status`'s bare "not running" (one-home-B2) | left as is — its own verb's words, outside this release |
| 17 | the listing tests sit in a class about the browser; the module note counts two (notes-tests2) | left as is — test layout, no rule |

## The two independent full-text readers

| # | finding (reader) | verdict |
|---|---|---|
| 1 | the re-probe question calls a dropped row "(no visible change)" (both) | fixed — "(same domains)" on the probed side; the recorded side names the row |
| 2 | round 2's rank-binding check not in the plan; it covers every CPU launch too (both) | fixed — in the plan's *Only the user's Sol runs can show*: each rank's cores under `mpirun` with no binding flags, decided on a Sol run's evidence |
| 3 | a job forced into the host env names `install`, which the script refuses for it (reader 1) | fixed — `hints.install_again_cmd(recipe)`: `bootstrap` for the host env |
| 4 | a routed env's recipe looked up by name though its category is known (reader 1) | fixed — `recipe_of_category(category)` when routed, `recipe_of_env` when forced |
| 5 | the plan lists three items this release closes (both) | fixed |
| 6 | `Domain.sizes`' note; `--` in the plan's prose (reader 1) | fixed |
| 7 | the activation forms reordered conda-first broke `molbuilder_json.toml`'s refusal row (the whole suite) | fixed — the forms in `configuration.md` § 4's order; the question keeps its own order (intended change 7 withdrawn) |
| 8 | `pid_state`'s `b"serve"` is a substring test (reader 2) | left as is — older than this release; no road reaches it |

## The reader of the fixes after them

| # | finding | verdict |
|---|---|---|
| 1 | the record and changelog claimed the two e2e runs, which ran before the last code fix | fixed — the whole fast suite and both e2e tests run again after the last edit |
| 2 | r2-3 F3 unaccounted for | fixed — R3c's (its owner, `template.SAID`), in this record and the plan |
| 3 | "two retired that restated a table" — one was | fixed |
| 4 | the host env's `bootstrap` remedy has no test row | fixed — a row; and "playwright is not routed", pinned only by the retired table test, kept as an outcome (`env_for_tool`) |

## Topic → test

| change | test | on 2.77.17 |
|---|---|---|
| F22 one write time | test_deck_runner::test_one_write_states_one_moment | fails: two times |
| F6 a copy naming no machine | test_launch_door::test_a_calculation_whose_machine_record_names_no_machine_is_refused | fails: the launch sent |
| F26 a queue's sizes by R3 | test_machines_listing::test_a_queue_is_said_by_the_machines_it_lists_else_its_largest | fails: '1000 GB' for a machine stating none |
| B-U3 a row that is no queue | test_machines_listing::test_a_queue_row_that_is_not_one_is_said_and_kept; machine_record.toml's re-probe row | fails: 'slurm · 1 queue'; the row cannot be written there |
| F24 one pidfile step | test_serve_verbs::test_both_pidfiles_are_verified_by_one_step_in_one_set_of_words; ::test_status_says_a_pidfile_that_names_no_pid_in_the_same_words | the serve rows fail at their words; the notebook rows cannot run there |
| install messages | test_launch_door_gate::test_a_job_without_its_envs_python_names_the_recipe_that_installs_it; ::test_a_forced_env_is_installed_by_its_own_recipe_or_none; gpu_contract.toml's renamed GPU env row | fail: 'install my-siesta', 'install my-own-env', 'install molbuilder', 'install my-gpu' |
| F11 the hooks' placeholders | test_envs_builds::test_a_hook_naming_a_component_the_build_lacks_is_refused | DID NOT RAISE |
| F4, F5, F7, F10, F14, F15, F16, F18, F21, F23, module_of, recipe doors | no new test — the same answers, compared (run scripts on five real preps, every catalogue item on real decks, file names and audit patterns, init-config's question, bundle members, hooks) | — |
| allocation | test_the_road_on_real_runs_e2e::test_the_ranks_a_scheduler_grants_rule_the_run_said_by_name, ::test_a_flag_on_the_run_script_still_overrides_and_says_so | pass (real runs) |
