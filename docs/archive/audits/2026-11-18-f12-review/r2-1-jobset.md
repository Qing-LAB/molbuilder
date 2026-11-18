# F12 review, round 2 — slice R2-1: the job set's own package

Reviewer: R2-1. Checkout 2.77.2 (f578482db), read only. No test, server or
writing verb was run. Two read-only Python checks were run (noted where used).

**Result: 26 findings — 19 in the code, 7 in the tests.** Six matter most:
the save-then-write protocol copied four times (the cause of round 1's A-D1),
SIESTA keywords in a shared module, the describe sequence written twice (once
per surface), the benchmark grid's names typed by hand in four places, and a
test that passes for the wrong reason.

## What was read

**Rules:**
- `docs/architecture.md`, whole.
- `docs/execution/architecture.md`: § 2, § 2.1, § 2.2, § 3, § 3.1, § 3.2 and § 3.3, whole.
- `docs/process/conventions.md`, whole.
- `docs/process/testing.md`: § 2–3b and § 6–8.
- The round-1 report `r1-A-jobset.md` and `v-ABC.md`.

**Code read whole:**
- `__init__`, `agreement`, `errors`, `sweep_reader`, `engines`, `ledger`,
  `sequence`, `shelves`, `plan`, `ladder`, `machine`, `points`, `rung`,
  `prepared_with`, `memory`, `naming`, `placement`, `ask`, `planned`, `group`,
  `commands`, `model`, `prep_inputs`, `summarize`, `materialize`,
  `continuation`, `ready`, `runstatus`, `prep`, `submit`.
- `_cli`, apart from the click option decorators at 450–525 and 1012–1075.

**Code read in part:**
- `migrate.py`: the module docstring, the definitions list, `migrate_state`
  (1325–1422) and `_migrate_groups`. Its rewrites of older files are code by
  design (architecture § 3.3), and no work goes into them.
- Outside the slice, to check the other side of a finding:
  - `web/blueprints/build.py` 940–1060
  - `envs/initconfig.py` 535–576
  - `resolve.py` 560–600 and 1069–1158
  - `template.py` around `GRID_AXES` and `items_in`
  - `job_request.py` 237–325
  - `warmfiles.warm_list`
  - `scheduler/quantities` (`parse_memory`, `canonical_mem`)
  - `deck_record.without_stamps`
  - `runwrap._deck_restart`
  - `identity`'s composers

**Tests read whole:**
- `test_jobset.py`, `test_walk_script.py`,
  `test_the_walk_empties_the_scratch_folder.py`, `test_hand_overs.py`,
  `test_wrapper_notify_flags.py`, `test_stage_names.py`,
  `test_restart_group.py`, `test_bench_summarize.py`,
  `test_launch_reads_what_was_answered.py`.

**Tests read in part:**
- `test_launch_door.py` (1–75), `test_sequence_of_groups.py` (1–60),
  `test_bench_grid_card.py` (45–100), `test_pipeline_log.py` (28–60),
  `test_prep_calculation.py` (28–34).
- `support/road.py`: its definitions list, `prep_groups` and `group_words`.

**Tests not read whole:** the other ~45 files that import or drive the
package. That is about 25,000 lines: the e2e files, `test_prep_bench_fold`,
`test_value_axes`, `test_task_setup_*` and others. They were only searched,
all of them, for the following:
- reads of a source file (none found)
- records written by hand
- local `_prep`, `_queues` and fixture helpers

---

## A. Code

### F1 — HIGH: the save, write and record steps are copied four times, the copies already differ, and none handles a write that stops partway (round 1's A-D1)

The same sequence appears in four places:
- compare the plan's identity, or stop at a preview
- `checkpoint.save_before`, its failure turned into the verb's error
- the ledger's `saved` line
- `Plan.carry_out()`

The places, with what each catches:
- `prep.py:2447-2461` (`prep_bench`):
  `except CheckpointError as exc: raise PrepError(f"the folder's state could not be saved, so nothing was prepared: {exc}")`
  … `plan.carry_out()`
- `prep.py:2538-2548` (`_record_the_order`):
  `except (CheckpointError, PrepError) as exc:` … `plan.carry_out()`
- `prep.py:3214-3236` (`_prep_groups`):
  `except (CheckpointError, PrepError) as exc:` … `the_plan.carry_out()`
- `submit.py:1692-1713` (`send_launch`):
  `except (CheckpointError, RuntimeConfigError) as exc:` … `plan.writes.carry_out()`

The only copy that catches `CutShort` and writes how far it got is
`migrate.py:1402-1414`: `except CutShort as exc: … ledger.record(… changes=said + exc.done, failed=…)`.

**Owner:** none today. Each copy assembles `planned.Plan` and
`checkpoint.save_before` itself.

**How the copies already differ:**
- In `prep_bench`, a `molbuilder.json` that does not read during the save
  becomes a `PrepError` through `user_error_as_prep`. The copy does not catch
  it, so the person gets the config error without "the folder's state could
  not be saved". `_prep_groups` says both.
- `prep_bench` checks `plan_id` before `preview` (2436–2441). The other two
  check `preview` first (2533–2537, 3208–3213).
- A-D1 (`CutShort` gives a traceback) exists in all four because the protocol
  is copied rather than called.

**Smallest fix:** one function, asked by all four and by `migrate`. It saves,
writes the `saved` line, carries the plan out, and on `CutShort` writes how
far it got and raises the verb's own error.

**Certainty:** read on all five sides.

### F2 — HIGH: describing a calculation is written out once per surface, and the two copies already differ

**The two copies:**
- The CLI's `describe_cmd`, `_cli.py:621-735`:
  - `stated_config(...)`
  - `engine_seam(engine).describe_findings`
  - `build_description(...)`
  - `write_description(...)`
- Task setup's hand-over, `web/blueprints/build.py:946-1060`:
  - `stated_config(...)`
  - `refuse_place` and `refuse_described`
  - `describe_findings`
  - `fresh_stages`, `label_of`, `relabelled`, `template_with_values`, …

**Owner:** none for an ordinary kind. A citing kind has one: the kind's
`describe` part, "the one describe the command line and the kind's own tab
both take" (architecture § 3.3). conventions § 3 also applies: the CLI calls
"the same lower-layer functions the blueprints call, never a private copy".

**How they already differ:**
- On an unsettled pseudopotential, the CLI throws away the seam's refusal
  sentence and prints the error findings' messages:
  `raise click.ClickException("\n".join(i.message for i in _found if i.severity == "error"))`
  (`_cli.py:709-711`).
- The page returns the refusal sentence with the findings beside it:
  `return err(refusal, 400, findings=...)`.
- The label: the CLI hands `name=run_name` to `build_description`. The page
  works it out itself, with `label_of(typed, calculation=…, stage_names=…)`
  and then `relabelled(engine, cfg, label)`.

**Smallest fix:** one describe function below the surfaces, from what was
stated to written files. Both surfaces call it, as transport's `describe`
part is called today.

**Certainty:** the CLI read whole; the web side read in part (940–1060).

### F3 — HIGH: a shared module names SIESTA's restart keywords (A18)

`materialize.py:784-789` (`continuation_files`):
```
f"  A stage whose description says `restart: clean` carries "
f"none of the group -- its deck omits MD.UseSaveXV / "
f"DM.UseSaveDM / MD.UseSaveCG, so files copied in would sit "
```

**Owner:** the restart group is the catalogue item's `expands`, read off a
deck by `script_emit.restart_group` (asked in `agreement.py:64-66`). What a
deck reads per file is `warm-files.toml`'s `honoured_by` (`runwrap._deck_restart`).

**What goes wrong:** take a PySCF stage set to `restart: clean`, launched with
`--from`. The refusal tells the person about SIESTA fdf keywords their deck
never had.

**Smallest fix:** say the engine's own restart group, through the catalogue
item, or say only "its deck reads none of them".

**Certainty:** read on both sides.

### F4 — MED-HIGH: the benchmark grid's setting names are typed by hand in four places, though two tables hold them

**Owners:**
- `template.GRID_AXES = ("mpi_np", "omp_threads", "gpu_count")` (`template.py:2672`)
- `model.AS_RESOURCE` (`model.py:62-67`): the run card's name for each
  `Resources` field
- `model.GpuRequest.gres` (`model.py:264-268`): "what `--gres` carries"

**The hand-typed copies:**
- `prep_inputs.py:525-527`:
  `("mpi_np", "omp_threads", *(("gpu_count",) if any(families) else ()))`,
  and at 544–551, `declared["mpi_np"]`, `declared["omp_threads"]`,
  `declared.get("gpu_count")`.
- `prep_inputs.py:197-203` (`_cell_resources`):
  `{"mpi_np": point["N"], "cpus_per_task": point["C"]}`, and
  `{**asked, "gres": f"gpu:{point['G']}"}`.
- `summarize.py:360-368`:
  `block["omp_threads"] = int(knobs["cpus_per_task"])` and
  `block["gpu_count"] = parse_gres_flag(...)`. This is `AS_RESOURCE` reversed
  by hand.

**What goes wrong:** adding a grid axis, or renaming one, edits
`GRID_AXES` and leaves these four disagreeing. The same happens if the
benchmark lane is opened to PySCF, whose cores item is `threads`: the report
would print a run card `job-request.json` refuses. `bench_refusal` admits it:
"the bench lane speaks SIESTA's vocabulary today".

**Smallest fix:** read the axis names from `GRID_AXES`, keep the N/C/G ↔ item
map once beside `MACHINE_AXES`, and compose `gres` through `GpuRequest`.

**Certainty:** read on all sides.

### F5 — MED: "the run card a run takes, defaults under it" is worked out again outside `resolve`

**Owner:**
- `resolve.card_values` (`resolve.py:1069-1100`): each card value goes through
  the setting door and is spelled (`_T.spell`).
- The default layering is at `resolve.py:589-597`:
  `{k: default_of(_items[k], _kind) for k in ours if k not in run_settings}`.

**The copies:**
- `group.card_shares`, `group.py:71-77`:
  `taken = {it.name: _T.default_of(it, kind) … if _T.is_run_setting(it)}` …
  `taken.update(request.run_condition(stage))`. These are raw, unspelled values.
- `prep_inputs.bench_inputs`, `prep_inputs.py:491-495`:
  `on_gpu = (bool(card["use_gpu"]) if "use_gpu" in card else bool(_T.default_of(...)))`.

**What could drift:** `card_shares` compares two cards' raw values, while
`resolve` compares spelled ones. Two cards that state one value in two
spellings the setting door accepts compare unequal there. Then:
- `may_join` is false, so a new calculation's groups default apart.
- Task setup's `why_not_join` says "states other values".
- prep itself would have run them as one value.

**Smallest fix:** `card_shares` and `bench_inputs` ask `resolve.card_values`,
or one "card as run" door, for the values.

**Certainty:** read on both sides. The concrete mismatch is plausible: I did
not find a spelling pair the door accepts that the page can write.

### F6 — MED: the CLI tests "a benchmark is prepared" itself, beside the door that answers it

**The CLI's test:**
- `_cli.py:924-926`:
  `benched = [r for r in refs if JobSet.of(Path(base) / bench_container(stage_home(base, task, r.name).token)) is not None]`
- The path is composed again at `_cli.py:934`.

**Owner:** architecture § 3.2, *prepared*: "a benchmark: `prep.prepared_already(calc, stage)`".
- That door is at `prep.py:2010-2030` and composes the same path at 2023–2025.
- `prep_bench` offers its stages through it (`prep.py:2410-2411`).
- The container is composed a fourth time in `prep.Resolved.record_dir`
  (`prep.py:378-384`).

**What could drift:** a change to what "prepared" means for a benchmark (for
example, that the sweep's plan must read) reaches `prep bench`'s offer but
not `launch` or `summarize bench`.

**Smallest fix:** `_stage_bench_dir` asks `prepared_already`. The container
path gets one composer, for example on `naming.StageHome`.

**Certainty:** read on both sides.

### F7 — MED: in `group`, one rule is written twice and one sentence is composed twice

**The rule, twice:**
- `may_join` (`group.py:92-96`):
  `not upstream_of(task, stage, template_text) & set(members) and card_shares(request, stage) == card_shares(request, members[0])`
- `why_not_join` (`group.py:108-124`) asks the same two conditions again, for
  words.

**The sentence, twice:** "builds on … one job runs its members one after
another … set its `restart` override to `clean`" is composed in
`why_not_join` (`group.py:110-119`) and again in `check_groups` (180–198).

**What could drift:** a third condition added to `may_join` changes the
default groups. Task setup's row (`why_not_join`) would still say the stage
may join.

**Smallest fix:** `may_join = bool(members) and why_not_join(...) is None`,
and one composer for the sentence.

**Certainty:** read.

### F8 — MED: a group's summed wall and largest memory are written twice

- `group.envelope`, `group.py:239-246`:
  `wall = (slurm_time(sum(parse_walltime(str(w)) for w in walls)) if all(walls) else None)`
  and `mem = ("0" if "0" in mems else max(mems, key=parse_memory) if mems else None)`.
- `group.sequence_request`, `group.py:323-334`: the same two expressions,
  inlined.
- `scheduler.quantities.canonical_mem` (322–336) is where `"0"` (the whole
  node) is defined.

**What could drift:** a change to how the whole-node ask or the walls combine
reaches a group's header but not a sequence's.

**Smallest fix:** one helper each, sum of walls and largest memory, beside
`canonical_mem`, asked by both.

**Certainty:** read.

### F9 — MED: probing the machine and writing its record is written twice; a comment says they agree, and they already differ

**The two copies:**
- `envs/initconfig.py:555-562`, its comment reading: "THE SAME STEPS
  `jobset probe --write` TAKES, in the same order". The steps are
  `local_facts(resolve_environment(now_iso=now), get_env_init())`,
  `probe_queues(env, getpass.getuser())` and `write_environment(env, path)`.
- `_cli.py:2727-2750` and 2839: the same steps in `cmd_probe`.

**Owner:** none. Each surface assembles the steps from `scheduler.record`'s
parts.

**How they already differ:** `probe --write` refuses a machine whose
`molbuilder.json` gives no activation (`_cli.py:2715-2722`). `init-config`
writes the record anyway and notes "carries NO activation"
(`initconfig.py:575-576`). Seeding may be meant to differ, but the "same
steps" are held by a comment, not by one function.

**Smallest fix:** one `scheduler.record` function, "probe this machine", that
returns the record and its notes. Both call it; each keeps its own policy on
activation.

**Certainty:** read on both sides.

### F10 — LOW-MED: the status table's "warm files" column and the stage's "declares" row come from two sources

**The column:** `runstatus._warm_present`, `runstatus.py:257-270`. It uses
`carried = _warm_list(engine, None, base).carry`, the carry rows of every
section, because kind is `None`.

**The row:** "declares" is `carries=[w.name for w in job.warm]`
(`runstatus.py:613`). That list is `model.warm_declaration`,
`warm_list(engine, kind, base)` (`model.py:532`), the kind's own section.

**What could drift:** a file another kind's section carries, lying in this
run's folder, shows in the "warm files" column. This stage would never take
it.

**Smallest fix:** pass the calculation's kind (`calc.task.calculation`) to
`warm_list`.

**Certainty:** read. Its effect today is small.

### F11 — LOW-MED: a second "path from the calculation" function, beside the one that says it must be the only one

**Owner:** `ledger.rel_to` (`ledger.py:110-123`): "One function, because a
second spelling is how the printed list and the recorded one come to disagree
about the same directories". It resolves both paths.

**The second:** `submit._rel` (`submit.py:437-443`) does not resolve. It
feeds the ledger's launch lines, `_member_facts`'s `runs`
(`submit.py:3165`), and the plan's `made_from`.

**What could drift:** under a folder reached through a symlink, prep's ledger
lines name a run relative to the calculation, while launch's name it whole.

**Smallest fix:** `submit` uses `ledger.rel_to`.

**Certainty:** read.

### F12 — LOW: `prepared_with` resolves a stage's config by hand, beside the function that does it

- `prepared_with.py:53-54`:
  `ladder = resolved_ladder(...); cfg = next(c for name, c in ladder if stage_key(name) == stage_key(stage))`
- `ladder.stage_config` (`ladder.py:66-70`) is the same lookup. Its one other
  caller is `web/blueprints/transport.py:422`.

**What differs:** a stage not in the ladder raises `StopIteration` in one and
returns `None` in the other.

**Smallest fix:** call `stage_config`.

**Certainty:** read.

### F13 — LOW: the job request's groups are re-keyed by hand twice

- `group.formed_again`, `group.py:396`:
  `group_of = {stage_key(n): e.group for n, e in request.stages.items()}`
- `prep._prep_groups`, `prep.py:2815`: the same line.

**Owner:** `JobRequest.group_of` (`job_request.py:244-248`), "the one lookup",
which compares names case aside.

**Smallest fix:** ask `request.group_of(name)`.

**Certainty:** read.

### F14 — LOW: two rules for "is this job a benchmark trial"

- `naming.stage_refs` (`naming.py:205-240`) says it is "the only place the two
  kinds are told apart", by which label a deck's name parses with.
- `naming.run_names` (`naming.py:255`) decides by
  `trial = jobset.kind == KIND_SWEEP`.

**Why it holds today:** the two agree because a trial's label
(`<label>-<point>`) never parses as the rung's label.

**Smallest fix:** `run_names` asks `stage_refs` (or the reverse), and the
docstring is put right.

**Certainty:** read.

### F15 — LOW: typed flags travel in two shapes

**The shapes:**
- `commands.launch_with` takes `(flag, value)` pairs, as the verb records them
  (`_cli._typed`; `commands.py:184-187`).
- `commands.launch_lines` takes a flat tuple, and makes every word a switch:
  `typed = [(f, True) for f in flags]` (`commands.py:213`).

**Callers that pass a valued flag flat:**
- `ready.py:624-625`: `flags=("--from", other)`
- `continuation.py:371-372` and 413: `_taking("--from", other)`
- `commands.py:516-520`

**What goes wrong:** the run's folder rides as a "switch" that no
`LAUNCH_FLAGS` row names. It is kept in every mode and never `shlex.quote`d.
If `--from` were ever dropped for some mode, the line would keep a bare path
as a stray word.

**Smallest fix:** `launch_lines` takes the pairs.

**Certainty:** read and traced.

### F16 — LOW: the bench report's list of settings is a second membership rule, and half of it is never read

**The second rule:** `summarize._pins_vocabulary` (`summarize.py:296-314`):
the non-machine run settings whose catalogue type is in `_ITEM_TYPES`. The
Python types it maps to are never read: its one use is
`if name in vocab` (`summarize.py:378`).

**Owner:** `template.items_in("run_card", engine=, kind=)`
(`template.py:2690-2700`), "THE ONE MEMBERSHIP RULE".

**What goes wrong:** a run setting the kind does not carry is pasted into the
`execution` block the report prints. `job-request.json` then refuses it.

**Smallest fix:** ask `items_in("run_card", …)`, and delete `_ITEM_TYPES`.

**Certainty:** read.

### F17 — LOW: the CLI reads another module's private word table, and parses flag names out of its prose

- `_cli.py:138` and 148:
  `from .placement import _LAUNCH_WORDS` … `flag = _LAUNCH_WORDS[field][1].split()[0]`.
  A flag's name is cut out of a display string such as `"--np N"`.
- Also at `_cli.py:145`: `parse_gres_flag(said) if field == "gres"`, a branch
  on one field's name.

**Smallest fix:** one table of each launch field's flag, which both the click
options and `placement`'s words read.

**Certainty:** read.

### F18 — LOW: the same table-printing code is written twice

`plan.render_plan` (`plan.py:100-107`) and `runstatus.render_status`
(`runstatus.py:992-998`) carry the same code:
- `w = [max(len(r[k]) for r in rows + [hdr]) …]`
- `def fmt(r): …`
- the header line, then the dashes rule

**Smallest fix:** one small table formatter.

**Certainty:** read.

### F19 — LOW: shared readers keep wording from older releases outside `migrate`

- `prepared_with.py:173-183`: `_OLDER_FRAME = re.compile(r"frame index (\d+) \(of (\d+)\)")`
  re-words points recorded before 2.67.2.
- `prepared_with.py:205-207`: a branch on `"point_folders" in was`.
- `shelves.py:74-83`: refuses a trial named by ranks per GPU, written before
  2.46.6.

Architecture § 3.3 keeps an older molbuilder's files in `jobset migrate`. The
user's rule is that no work goes into older releases' files.

**Smallest fix:** let `migrate` rewrite `prepared-with.json`'s older words,
and let the reader refuse a record it does not read.

**Certainty:** read. The intent may be deliberate.

---

## B. Tests

### T1 — MED-HIGH: a test passes for the wrong reason; its input is hand-typed

`test_launch_reads_what_was_answered.py:56-62`:
```
Job.from_dict({"name": "G0N48C1", "script": "x.fdf",
               "resources": {"walltime": "4h"}})
assert "G0N48C1" in str(e.value)
```

**What it meant to show:** that an unknown key in a job's `resources` is
refused, naming the job.

**What actually happens:** `Job.from_dict` first refuses the missing job keys
(`model.py:454-458`). Run read-only, it raises:
`WrittenBefore | job 'G0N48C1' has no 'warm', 'traits', 'point', …`

So the `resources` check is never reached. Delete it, and this test stays
green.

**Smallest fix:** build the job's dict from `Job(...).to_dict()`, the writer,
then put the unknown key into its `resources`. Or remove the test:
`Resources.from_dict`'s refusal is already tested beside it (48–54).

**Certainty:** read and run.

### T2 — MED: two test files copy the same queues and the same setup fixture

- `test_launch_door.py:30-50`: `_queues()` (a `debug` and an `htc` Domain) and
  a `cluster` fixture,
  `a_machine_with_queues(tmp_path, monkeypatch, _queues())` + `described(road_tree(...))`.
- `test_sequence_of_groups.py:25-49`: the same `_queues()`, byte for byte, and
  the same fixture.

**Owner:** `support/road.py` (`a_machine_with_queues`, 636). The rule is one
door per group, "never a fixture/prelude copied per file".

**Smallest fix:** one "two-queue machine" door in `support/road.py`.

**Certainty:** read.

### T3 — MED: nine test files carry their own "prep this stage" helper beside the shared one

**The shared door:** `support.road.prep_groups(calculation, *groups)`
(`road.py:356-361`).

**The local helpers** (`def _prep` / `_prepared`):
- `test_launch_door.py:53-58`
- `test_prep_calculation.py:28-32`
- `test_pipeline_log.py:37-50`
- `test_effective_parameters_record.py`
- `test_transport_decks_e2e.py`
- `test_prep_from_the_browser.py`
- `test_launch_door_gate.py`
- `test_target_machine_choice.py`
- `test_prep_refuses_before_writing.py`

The same line, `jobset("prep", "task", *group_words(bundle, stage), "--bundle", …, "--target", "this")`
plus its exit-code assertion, appears 24 times across the tests.

**Smallest fix:** give `prep_groups` a by-stage form (it already has
`group_words`), and retire the local copies.

**Certainty:** three read, the rest found by search.

### T4 — LOW: a test rewrites the package's stamp-masking rule by hand

- `test_jobset.py:197-209`: `_logic` drops lines starting `generated-at` by
  hand.
- `deck_record.without_stamps` (`deck_record.py:95-103`) is "the rule two
  files are compared on when the question is what they compute".

**What could drift:** a new stamp, or a `generated-at` spelled differently,
breaks the test's comparison while the package's rule copes.

**Smallest fix:** compare `without_stamps(a) == without_stamps(b)`.

**Certainty:** read.

### T5 — LOW-MED: the planner's input to a private function is typed by hand

- `test_walk_script.py:23-32` and
  `test_the_walk_empties_the_scratch_folder.py:43-53`: each builds `steps`
  tuples and stub members, and calls `submit._walk_script` directly.
- The second says so itself: "WHERE a round begins is the planner's (`submit`)
  … typed here as it gives it".

The input is what `_points_walk` makes. The same file's other two tests take
the road (`prepared_here`).

**Smallest fix:** drive the walk from a launch on the road, as
`test_walk_script.py:45-79` does, or say in the test why this one cannot.

**Certainty:** read.

### T6 — LOW-MED: a calculation the road can make is built by hand

- `test_hand_overs.py:25-30` and 49: builds a SIESTA vibration with
  `Task(...)`, `derive_run(...)`, `StructureRef(source="v.xyz")` and
  `Calculation(base=tmp_path, task=..., template=None, jobset=None)`.
- The docstring's reason (the road describes no transport calculation)
  covers the transport row only.
- The road makes a vibration: `prepared_here(..., calculation="vibration")`,
  `test_walk_script.py:58-60`.

**Smallest fix:** describe the vibration on the road (`described`), opened
through `calculation.open`. Keep the hand-built transport row, with its
reason.

**Certainty:** read.

### T7 — LOW: two tests check the same `Resources` JSON round trip

- `test_wrapper_notify_flags.py:5-23`: `Resources.from_dict(json.loads(json.dumps(r.to_dict())))`
  equals `r`, for `notify_channels`.
- `test_launch_reads_what_was_answered.py:37-40`: the same round trip, for
  `time` and `mem`.

**Smallest fix:** one test, a row per field.

**Certainty:** read.

---

## Checked, and left out

- `ladder.inputs_of`, `ready.gather_of` and `points.points_of` all ask
  `kind_table.part_of`. That is the one resolver asked from each floor, as
  architecture § 3.3 says.
- `placement.request_of` (a run's queue request) and
  `prep_inputs._cells_on_the_queue` (a benchmark cell's, with no wall or
  memory) are two answers by design: a cell is fit on its cores and GPUs.
- `JobSet.of` imports `commands` (same floor), and `ledger` imports
  `runfiles`. Both are allowed.
- `prep._flat_resources`, `plan.resources_text` and `PrepAnswer.lines`
  "resources" each show one record from one source, so not a defect.
- `submit._plan_sequence` writes its walk's launch line by hand
  (`submit.py:3372-3378`). That line is run by the job, with the
  calculation's full path, not printed for a person, so it is left out.
