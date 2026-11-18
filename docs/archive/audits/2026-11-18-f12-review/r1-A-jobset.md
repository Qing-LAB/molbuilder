# F12 review, round 1 — slice A (the job set): report

Reviewer: slice A. Checkout f578482db (2.77.2). Read-only. No test, server or
writing verb was run.

**Result: 1 defect, 7 untrue sentences.** None is severe. The prep and launch
protocol, the refusal table, the ready door, the next-step table, the
run-of-points rules and the job request reader were traced line by line against
the chapters. They agree, apart from what is listed below.

## Coverage, said plainly

The chapters were read end to end: `execution/job-system.md`,
`job-request.md`, `runs-of-points.md` and `run-identity.md`.

**Code read whole:**
- `calculation.py`
- `jobset/ready.py`, `commands.py`, `group.py`, `prep.py`, `prepared_with.py`,
  `planned.py`, `memory.py`, `machine.py`, `submit.py`, `continuation.py`,
  `sequence.py`
- `runpoints.py`, `joblist.py`, `jobs_cli.py`
- `jobset/runstatus.py`, except a few view helpers
- `jobset/materialize.py` from line 172 to the end
- `job_request.py` from line 180 to the end
- `jobset/_cli.py` at lines 40–340, 526–2160 and 2321–2540: status, prep,
  launch, summarize, migrate, export and scratch

**Code read in part:**
- `prep_inputs.py`: the run's inputs, `fold_allocation`, `bench_refusal`
- `migrate.py`: `migrate_state`, the record reading, `_kept_as`
- `task.py`: `_check_id`, `derive_run`
- `runs.py`: `concluded`, `_ended_written`, `ended_before_a_point`
- `model.py`: `validate`, `gpu_request`, `warm_for`, `warm_carry`,
  `relaunch_continues`
- `summarize.py`: `sum_of`, `task_summary`
- `agreement.py`, `ask.py` (`confirm`), `runrecord.stamp`
- The CLI's `describe` body
- `identity.normalise_id` / `run_id`, run on the chapter's § 3.1 table: every
  row matches

**Code not read:**
- `runrecord.py` beyond its writers and readers of markers
- `calcdirs.py`, `jobset/naming.py`, `task_reader.py`
- `describe.py` beyond its signatures
- `jobset/ladder.py`, `rung.py`, `points.py`, `placement.py`, `shelves.py`,
  `engines.py`, `errors.py`, `ledger.py`, `sweep_reader.py`
- `_cli.py`'s probe and machines

**Every dotted door name checked.** About 330 dotted names appear in backticks
across the four chapters. Each was resolved against the package with
read-only imports, and every one exists.

---

## Defects

### D1 — A failed write in prep or launch ends in a Python traceback, with no ledger line saying the write stopped

Several places carry out a plan and none of them catch `CutShort`:
- `molbuilder/jobset/prep.py:3236` (`_prep_groups`), `:2462` (`prep_bench`) and
  `:2548` (`_record_the_order`) all call `plan.carry_out()`.
- `molbuilder/jobset/submit.py:1713` (`send_launch`) calls
  `plan.writes.carry_out()`.

`planned.Plan.carry_out` raises `CutShort`, which is an `OSError`
(`planned.py:280`: `class CutShort(OSError)`).

**What the chapters say:**
- `job-system.md` § 5.0, checkpoint 6: *"nothing refuses. An error writing (a
  full disk) leaves the stage not prepared; the state saved at 5 is the way
  back."*
- *Every refusal, and its way on*: *"A reader's refusal … is the verb's, in its
  reader's words, never a traceback."*

**What the code does.** Nothing on either road catches `CutShort`:
- `prep_cmd` catches only `PrepError` (`_cli.py:1164`:
  `except PrepError as e:`).
- `submit_cmd` catches only `SubmitError` and `click.ClickException`
  (`_cli.py:1733–1741`).
- `cli.main` handles only `UsageError`, `RuntimeConfigError`,
  `ClickException`, `Abort` and `_BackendUnavailable` (`cli.py:3794–3822`).

The same pattern is handled in `jobset/migrate.py:1402–1414`
(`except CutShort as exc:` writes in the ledger how far it got and raises
`MigrationStopped`). Prep and launch have no counterpart.

**Scenario.** `jobset prep task --group 2` runs on a cluster scratch disk that
fills up, or the stage folder holds a file the user may not write, partway
through the plan:
- The state is saved, and the ledger's `saved` line is written.
- Then the person sees a Python traceback,
  `CutShort: writes …/02_medium/run-0/… failed (…) -- N operation(s) …`, not a
  one-line error.
- No line in `jobset-decisions.log` says the prep stopped or where.
- `launch` behaves the same: `plan.saved` is still printed in its `finally`
  (`_cli.py:2084–2089`), then the traceback.

**Certainty:** read and traced. It needs a write to fail, so this is a real
road but an uncommon one.

---

## Untrue sentences

### U1 — The state diagrams show Save making changes that Task setup's Save refuses

**What the chapter says.** In `job-system.md`, *The states, and what moves
them*:
- Stage diagram, line 197: *`launched --> removed : the description drops it · Save`*.
- Line 164: *`preparable --> waitsprep : … or put into a launched group · Save`*.
- Group diagram, line 303: *`la --> lc : a member removed from the description,
  or job-request.json edited by hand · Save, by hand`*.

**What the code does.** Task setup's Save route refuses each of these with 409,
nothing written:
- **A stage put into a launched group, or a member moved to another group.**
  `web/blueprints/build.py:1479–1488` runs
  `hit = launched_without(after, n)` and
  `if hit and (hit[1] or any(job_req.group_of(x) is not None for x in hit[2])): return err(joined_launched(after, n, *hit), 409)`.
- **A launched stage taken out of the description.**
  `build.py:1520–1531` calls `ready.changed_once_run`, and `ready.py:469–470`
  returns `what = "it is taken out of the description"`, refused with
  "`X` has run … Nothing was saved."

`web/task-setup.md` § 8 states the refusal, so the code follows the rule. It is
the job-system diagrams that are wrong. The chapter's own prose also says
*"Task setup's Save refuses a stage put into it"* (*A launched group's members
are fixed*).

**Scenario.** A person reads the stage diagram, deletes the launched `tight`
row in Task setup and presses Save. They get a 409 refusal, not the *removed*
state. A stage reaches *removed* while launched, or a group reaches *launched,
its members changed*, only through a hand edit of `task.json` or
`job-request.json`. Those edges should name *by hand*, not *Save*.

**Certainty:** read and traced.

### U2 — The states table says `status` shows *changed since prepared* for a launched stage; the table row never does

**What the chapter says.** `job-system.md`, *The states* table, **launched**
row, *what `status` shows*: *"the newest run's state, converged in its own
column; changed since prepared when its records changed"*.

**What the code does.**
- `runstatus.jobset_status` (`runstatus.py:763–769`) puts the ready door's
  words in place of the run's only when the stage was never launched:
  `if not r.launched and row.state == "pending":`.
- `render_status` (`:983–990`) prints only `s.detail`, `s.scratch` and
  `s.copied_from`. The `changed` field is printed by `render_stage_status`
  alone (`:1167–1170`).
- The group's next-step line names the change only when no run is queued or
  running: `_steps`, `:862`, `elif g.since_prepared and live is None:`.

**Scenario.** `tight` is launched and running. Its run card's `mpi_np` is changed
and saved, which Save allows. `jobset status` then shows `running` in the
table, and `running, let it finish` on the group line. Nothing says *changed
since prepared* until the run ends, unless the person runs `status tight`.

**Certainty:** read and traced.

### U3 — "A run that finished without converging is taken, with a warning": no warning is printed

**What the chapter says.**
- `job-system.md` § 5.4, the independent and linked stages table: *"A run that
  finished without converging is taken, with a warning."*
- *What finished means*: *"One that finished without converging is taken, with
  a warning."*

**What the code does.** The verdict is stated as one fact among the others in
the continuation line: `continuation.py:65–70` builds
`facts = [… _what_it_is(...)]; if self.converged is not None: facts.append(self.converged)`,
which reads *"continues from 01_coarse/run-0 (the stage before it; concluded
rc=0 …; geometry NO)"*. The only `WARNING` a launch prints is for a file
written as its run ended (`continuation.py:72`, `submit.py:2860`).

**Scenario.** `coarse` ends with `geometry NO`. `launch task` for `medium` says
"(…; geometry NO)" in parentheses, with no warning.

**Note.** This may be the intended wording, *stated, never judged*. If so, the
chapter's word *warning* is the thing to change.

**Certainty:** read and traced.

### U4 — Name refusals do not cite the chapter sections run-identity says they cite

**What the chapter says.** `run-identity.md` § 3, after rule 3: *"Each refusal
says what failed … and cites § 3; a description whose name, formula and id
disagree is refused citing § 2.0a and § 3, rule 1 (`Task._check_id`)."*

**What the code does.**
- `identity.normalise_id`, run on `Über`, `///` and a 300-character name, says
  what failed and cites no section. For example: *"'///' is not a usable name:
  it is entirely separators, so nothing is left to name the calculation"*.
- `task.py:518–524` (`_check_id`) says *"run.id … is not what this description
  derives … which one is right is not something this reader may guess"*, and
  also cites no section.

**Certainty:** read, and run for `normalise_id`.

### U5 — The chapter credits the wrong function with writing a member's row in `job-set.json`

**What the chapter says.** `job-system.md` § 5.0, *6 — the plan written*:
*"… then its pipeline log, then its row in `job-set.json`, the moment that
stage is prepared (`prep.prep_jobset`)"*.

**What the code does.** `prep_jobset` (`prep.py:76–286`) renders the wrappers,
copies the shared package and writes a benchmark's STAGE-PLAN. It writes no
row. The row is written by `prep._plan_calculation` after the log:
`prep.py:1056–1059`, `log.close(); plan.text(log.path, …); js.write(record_dir, plan=plan)`.

**Certainty:** read and traced.

### U6 — The verbs table says `jobset status` writes nothing; it can write a run's end into `run.json`

**What the chapter says.**
- `job-system.md` § 5.3, *The verbs*, row `jobset status [<stage>]`, column
  *it writes*: *"nothing"*.
- `status_cmd`'s help text says the same: *"Read-only."* (`_cli.py:768`).

**What the code does.** `jobset_status` reads each newest run through
`runs.concluded` (`runstatus.py:647–648`). For a run here whose process is
gone, or a queue job the scheduler no longer lists, that calls
`_ended_written` → `runrecord.record_ended`, writing the ending into the run's
launch record (`runs.py:1000–1004`, `1028–1051`).

This is the intended *asked once, written once* rule. § 6.0a and
`running-a-job.md` § 4.2 state it; only the verbs table and the help text say
otherwise.

**Scenario.** A run here is killed with no marker written. The next
`jobset status` writes `ended` into its `run.json`.

**Certainty:** read and traced.

### U7 — The duplicate-key refusal names only `job-request.json`

**What the chapter says.** `job-request.md` § 3: *"Two keys that are one name, an
entry for a stage the description does not hold, and a stage of the
description with no entry are each refused by the reader, naming both
files"*.

**What the code does.** Two of the three refusals name `task.json`. The
duplicate-key one names only the job request, through `_SOURCE`:
`job_request.py:410–412`,
`_refuse(f"two entries, {seen[k]!r} and {key!r}, are one stage -- names are compared case aside", where="stages")`.

**Certainty:** read and traced. This is minor.

---

## Checked, and found to agree

- **Refusal rows.** R-P1–R-P23 and R-L1–R-L46 against their *decided in*
  functions; spot-checked R-L6/7/8/9/10/26/28/45/46 in `submit._the_group`
  and `_planned`.
- **The next-step table.** Its row order matches `runstatus._steps` exactly.
- **`GroupAnswer`.** `needs_prep`, `launchable`, `preparable` and `said`
  match *The task*.
- **Prep 2a.** Nothing to write, prepared again whole, launched-for-request
  only, and a sequence prepared whole.
- **Write order and save.** The order of writes in `_prep_groups` matches
  checkpoint 6, and the save is made once.
- **The launch protocol.** `plan_launch` → `send_launch`: the folder check,
  the save at a group's first launch, `held` questions written after the save,
  and the one sender.
- **A sequence of groups.** The walk's needs chain, `--in-sequence` and
  `--in-job` checks, the record, `_ride`, and the walls.
- **Memory need (§ 3.1a).** The bound X·max(1, M/N), the smallest per point,
  the largest per stage and per job, and the line format.
- **The job request reader.** Group numbering, notify refusals, lane asks,
  case-aside matching.
- **The run-of-points rules (§ 4).** `rows_of`, `still_to_run`,
  `why_not_run`, `run_state`, `said`, `reason`, `points_state`.
- **Launch flags.** `LAUNCH_FLAGS` against R-L4.
- **`molbuilder jobs`.** `status` and `stop` against § 6.0a, and
  `joblist` file naming.
