# The job system — batches, ladders, and HPC deployment

**Role:** guide
**Domain:** execution

**Companions:** [`execution/running-a-job.md`](?doc=execution/running-a-job.md)
— the single-job wrapper this framework runs many of;
[`execution/job-contracts.md`](?doc=execution/job-contracts.md) — the run-dir
layout, the wrapper files, and the `job-set.json` / parameter vocabulary this
framework produces;
[`execution/overview.md`](?doc=execution/overview.md) — the execution-domain
map and the current → target status picture;
[`engines/tuning.md`](?doc=engines/tuning.md) — the scientific *values* behind
the staged ladders (this page covers only how they are *scheduled*).

---

## The short version

**Everything is a job set** — one calculation is a job set of one; a
ladder or a benchmark sweep is a bigger one. Same verbs, no shortcuts:

```
describe          the description and its job request    (your laptop:
                   molbuilder describe -- no job verb)
jobset prep        derive decks + scripts FOR this machine   (the target)
jobset launch      ONE job per invocation -- run or submit; a benchmark
                   sent to a queue, one per resource shelf (§ 7)
jobset summarize   summarize results that exist: a sweep's trials -> the
                   record + the report, PRINTED; a transport calculation's
                   bias points -> its I-V record; a SIESTA vibration's
                   force-constant stages -> its displacement sweep
                   (engines/vibration.md 5.9).  It never makes a run's
                   result: every run writes its own (a SIESTA vibration's
                   job derives its modes itself, engines/vibration.md 5.5)
jobset status      every stage's state, and each group's next step; with
                   a stage, that stage in full -- its deck, what it
                   carries, its resources, its attempts
```

Beside them, on a calculation: `jobset scratch` — an engine's scratch its
runs hold, listed, and emptied with `--empty` from every run not going;
`jobset migrate` — what an older molbuilder wrote in it, rewritten;
`jobset export` — a transport calculation's every result, as NetCDF or as
CSV tables, from what is already there (no data file read; the points
without their DOS by region said), written into a folder you name outside
the calculation ([`engines/transport.md`](?doc=engines/transport.md)
§ 2a.12, *Export*; since 2.32.0). On a
machine: `jobset probe` and
`jobset machines` — its record, and the records prep can target (§ 5.3).

| key rule | one line | where |
|---|---|---|
| **floor 3 is derived** | `job-set.json` is computed at prep ON the target — never produced on one host and shipped to another | § 4 |
| **one grouped job** | a bench sweep is ONE submission that runs trials in sequence, each recording its own `run.json` | § 7 |
| **the record shown, the scheduler decides** | prep prepares what is asked and prints its fit on the queue it names; a refusal is the scheduler's, shown in its own words (`scheduler.md` R6) | § 6 |
| **ladder = stages** | coarse feeds tight through declared warm files; the next step is read, not guessed | § 3, § 5 |
| **the task** | `prep task` shows the groups — each stage's number in `job-request.json` — and prepares the ones you pick, each one job; `launch task` sends the next group whose upstream has finished, taking what it builds on then | *The task* |
| **runs are yours** | any stage launched again, warm or `--cold`, however it ended; molbuilder keeps every run, says how each ended, and saves the folder's state before every prep and before each group's first launch — going back or branching is a `checkpoint restore` | *Runs, stages and checkpoints* |

---

## Runs, stages and checkpoints — what molbuilder does for you, and what stays yours

> *"the contract really should be a rule for the user: what you should do and
> what you should not do otherwise things could be broken, and here are the
> tools for you to keep track of intermediate known good states"* · *"run
> continue warm or cold is user's decision, and error or not, that's user's
> responsibility"* · *"checkpoint is used for the rollover and branching"*
> (user, 2026-10-07)

**molbuilder is a research tool, not a guard.** It keeps your runs apart, tells
you truthfully where each stands and how it ended, and keeps the states you can
go back to. It never decides for you whether a result is good, whether to run
again, or how. This section is the whole contract for runs and stages; the
sections after it say how each part works, and none of them adds a rule this
one does not state.

### The words

| word | what it is |
|---|---|
| **calculation** | one folder, described once by `task.json` and its template, and asked to run by its job request, `job-request.json` ([`job-request.md`](?doc=execution/job-request.md)) |
| **task** | the calculation, as the verbs name it: `jobset prep task`, `launch task`, `summarize task` |
| **stage** | one step of the calculation's ladder — `coarse`, `relax`, `device` — named in the description, prepared with its group, and again until it is launched — and after, for its job request (*The task*, below; [`job-request.md`](?doc=execution/job-request.md) § 5) |
| **group** | stages that share one job — the stages one number names in the job request ([`job-request.md`](?doc=execution/job-request.md) § 3.3), prepared together and launched together, none building on another, walked in the ladder's order (`project-layout.md` § 1.6.6) |
| **run** | one launch of a stage (`jobset launch task`). Every launch is a new run, numbered on from the last: on the hierarchical layout its own folder, `01_coarse/run-0`, `run-1`, …; on the flat layout its own numbered files in the calculation's folder, `H2_01_coarse-run0.out`, `-run1.out`, … A stage that sweeps the bias (transport, `low_bias_approximation: false`) holds a folder per voltage inside its run (`transport.md` § 2a.11). Inside a run, each start of its wrapper is a **try**, its files carrying `-run<N>` (`project-layout.md` § 1.6.1). The code, and some pages, call a run's folder its **attempt**. A run a transport calculation copies in from a finished one it reuses is that calculation's launch, its launch record kept ([`engines/transport.md`](?doc=engines/transport.md) § 2a.16) |
| **warm / cold** | what a run starts from. **Warm**: the restart files (density, geometry, optimiser history — the restart-file list, [`job-contracts.md`](?doc=execution/job-contracts.md) § 4.2a) a run left. **Cold**: none of them — the engine starts from the deck alone |
| **saved state** | a checkpoint of the whole calculation folder ([`checkpointing.md`](?doc=execution/checkpointing.md)): an id, a note, the time. `molbuilder checkpoint save / list / restore / tag` |

### The task — prepared group by group, launched group by group

> *"we can let prep to prep the whole task and ask the user which stage it
> intend to do. this way it can do the correct carry over of previous stage
> files when needed and when that is ready"* · *"we need one unified
> framework and protocol and verb design and api"* (user, 2026-10-07) ·
> *"the stages should have a group number assigned to it"* · *"the prep
> should be able to prepare all groups at the same time … that's the launch
> status, not the prep"* (user, 2026-10-11)

**The verbs act on the task** — the calculation `task.json` describes — **group
by group**. Each stage states a **group** number in the job request, by the rule
of [`job-request.md`](?doc=execution/job-request.md) § 3.3: the stages with one number are **one job**, prepared
together and launched together.

```
jobset prep task      # the groups, and which are prepared; prepare the one(s) you pick
jobset launch task    # send the next group whose upstream has finished -- one job
jobset status         # every stage: where it stands, and what is next
jobset summarize task # read a result that gathers stages (a transport I-V, a sweep)
jobset prep bench <stage> · launch bench <stage> · summarize bench <stage>
                      # a benchmark measures ONE stage -- named, as before
```

#### What a stage takes, and when

**What a stage takes from the stages before it is one of two kinds**, and the
kind decides when it is taken:

* **files the run reads when it starts** — an optimization ladder's restart
  files (SIESTA's `.XV`, `.DM`, the optimizer's history; PySCF's
  `_optimized.xyz` and `.chk`), a transport rung's inputs (the seed's `.DM`,
  the leads' `.TSHS`, the device's `.TS.HSX`). The stage's deck does not
  depend on them, so it is **prepared any time** and they are **taken when
  the stage is first launched** — when the run they come from exists.
* **values written into the deck** — none: no stage's deck depends on
  another stage's run. A SIESTA vibration's force-constant call takes the
  relaxed geometry inside its own run — the run's plan step writes it as
  that call's `coordinates.fdf` ([`engines/vibration.md`](?doc=engines/vibration.md)
  § 5.11 § 3) — so its deck, too, is prepared any time.

**Which run, by default, is one rule for every kind**: the **newest** run of
what it builds on, which **must have finished** — an older one never stands
in, because a run launched again to tighten is the one you mean. A
calculation that reuses a finished one asks the same of it, result by
result: one result — one voltage at one frame — is reused only when it
finished in that newest run, and a stage is copied when every result asked
of it did, in a run that has ended; its manifest names the run each comes
from *(the user, 2026-10-25: "newest, must be finished, the manifest will
name the run it took, only finished ones will guarantee the correct
result"; "One rule: reuse completes")*. For an
independent ladder's stage, `--from` names any other run of this calculation
(`<NN>_<stage>/run-<n>`), taken as said (§ 5.4); a transport rung's inputs
and a SIESTA vibration's are their kind's, never named. A transport rung's gather
keeps a second check — the upstream run ran the same engine input as the deck
that rung's description renders now, comments and molbuilder's records aside
([`engines/transport.md`](?doc=engines/transport.md) § 6.1, *the same engine
input*) — and waits, naming the way, when it did not.

**A transport calculation that reuses a finished one** ([`engines/transport.md`](?doc=engines/transport.md) § 2a.16) takes
more, and earlier: its prep **copies in** each group the finished calculation
computed as this one would now — every member's newest run there the same
calculation, every stage it takes inputs from copied in too — every run of
its members, under their own names, as launched runs of this calculation; and
a rung of points it prepares here **takes over**, at its first launch, the
finished calculation's points it would compute the same, as a warm launch
takes over its own.

#### The states, and what moves them

**Three things have a state, and nothing stores it.** A **stage**, its
**group** and each **run** of a stage are read from the files every time —
`task.json`, `job-set.json`, the run folders and their records — by one door
each: the ready door (`ready.task_state`, `ready.readiness`,
`ready.GroupAnswer`; [`architecture.md`](?doc=execution/architecture.md) § 3.2)
for a stage and a group, the run doors (`materialize.ever_launched`,
`parse.dirs.run_status`) for a run. `status`, `prep`, `launch`, the Results tab
and Task setup all ask the same doors, so they cannot disagree. **Five acts
change what the doors read**: **`prep task`** (`prep_task`) writes a group's
files and rows — or, with no group named, an older transport junction's order
record ([`engines/transport.md`](?doc=engines/transport.md) § 1.1) — and, for
a transport calculation that reuses a finished one, copies in the runs of the
groups it reuses (§ 2a.16 there); **`launch task`** (`send_launch`) fills a run and sends it;
**the run itself** — the scheduler and the run script — writes its output and
its conclusion; a **`checkpoint restore`** (`checkpoint`) puts the whole
folder back; and **the description and its job request** (Task setup's Save, `molbuilder
describe`, or either file by hand) change what each stage's files are
compared with, which stages there are and how they are grouped — never a
stage's files. Every edge below names its act and, after the `·`, who makes it: one
of those five.

**What marks each state.** No file holds a state, and none is named for one.
A state is read from the files the acts write: some by a file being there —
a stage's row, a run's launch record, its conclusion marker — some by what
one says — the marker's exit code, the row's members. Every such file is a
row of the manifest ([`project-layout.md`](?doc=execution/project-layout.md)
§ 5), which names its writer and the one door that reads the file; the last
column is the door that reads the *state*, and it reads each file through
that file's own door:

| what | it is | when the files say | the file | manifest | the door that reads the state |
|---|---|---|---|---|---|
| a stage | **described** | the description names it | `task.json` — its entry in `stages`; its `group`, `job-request.json`'s | § 5.1 | `task.of`, `job_request.of` |
| | **prepared** | its job's row is in the plan | `job-set.json` — the row; its `members`, the job it is prepared in; a SIESTA vibration's `phases`, the ones its run walks | § 5.1 | `Calculation.job` |
| | **ready to launch** / **waits for** | what it builds on has finished — the newest run of that stage, by that run's state below | the files of that run | § 5.3 | `ready.first_launch_takes` |
| | **its job's members changed** | its row's `members` are not its group's now | `job-set.json` against `job-request.json` | § 5.1 | `ready.regrouped` |
| | **launched** | a run of it has a launch record | `run.json` — flat, `<base>-run<N>.run.json` | § 5.3 | `materialize.ever_launched` (`runrecord.launch_record`) |
| a group | **prepared** / **launched** / **its members changed** | every member's row names these members / a member is launched / a row names other members than the group holds | its members' rows and records | § 5.1, § 5.3 | `ready.GroupAnswer` — the one answer every verb and surface reads (below) |
| | **goes in job N** | prepared, and a sequence still going holds it and has not launched it yet — launched before or not (§ 6.0, *A sequence of groups*) | the job's address (§ 6.0a), asked whether it is going | — | `ready.held_by_sequence` (`joblist.is_going`) |
| a run | **opened, not launched** | its `run-<n>` folder is there, stamped a run, and has no launch record | the folder (`calcdir.json`, `role: run`), no `run.json` | § 5.2 | `paths.attempts_in`, `runrecord.launch_record` (`materialize.resolve_attempt`) |
| | **queued** | launched, nothing written yet | `run.json`, no output | § 5.3 | `rundir.run_state_of` (`parse.dirs.run_status`) |
| | **running** | its engine is writing, and it has not concluded | `<base>-run<N>.out` (PySCF: `.pyscf.log`) | § 5.3 | the same |
| | **finished** / **failed** | it concluded with exit code 0 / with another — or with none that can be told, a marker empty or not reading — or its output says the engine stopped; a stop the marker never saw is the scheduler's own line, else the monitor's closing record | `<base>-run<N>.concluded` and its `rc=`; the output's ending; the scheduler's error file `run.json` names (`scheduler.stderr`); `<base>-run<N>.monitor.log` | § 5.3 | `runrecord.ending`, and the same door |
| | **unreadable** | its launch record is there and does not read | `run.json` | § 5.3 | `runrecord.launch_record` |
| | *converged* — its own column, never the state | the engine's own output says so | `<base>-run<N>.out` | § 5.3 | the same scan |
| a point of a run of points | **done** / **not done** | its run finished, converged, and left what the rungs after it take | its marker, its output, those files | § 5.3, § 5.4 | `continuation.done` |
| a run of points | **queued** / **running** / **finished** / **failed** | running: a point is running, or the walk is between two points while a job launched from this machine that holds the run is going — never the job a reader runs inside, a sequence's, whose own runs of points it reads by their points (§ 6.0); queued: no point the run walks itself has started and its job is not over (cancelled in the queue, or a process that died before its first point, is a run that ended); else finished: every point's run finished, converged or not; failed: a point's run did not finish — it failed, or the walk stopped before it — the first point not done named | its `run.json`, its points' files (the row above), the jobs launched from this machine (§ 6.0a) | § 5.3 | `continuation.points_state` (`joblist.is_going`, else the run door's `runs.concluded`) — `status`, the gather, `jobset scratch` and the transport record ask it alike |

**What a run started from is a record, not a state**: `.continued-from` —
the one run it continues from — and `.gathered-from` — a transport rung's
inputs, each from its run — written by the one hand-over (§ 5.4) and read
by `status`, the transport record and the Run panel (§ 5.3 of the manifest);
a run's own `continued_from` is its `run.json`'s, written with it.

*A stage* — the ready door's answer (`ready.readiness`):

```mermaid
stateDiagram-v2
    direction TB
    state "described" as described {
        state "preparable" as preparable
        state "waits before it can be prepared" as waitsprep
        [*] --> preparable
        [*] --> waitsprep : a template that does not resolve, or put into a launched group
        waitsprep --> preparable : the template fixed, or a group of its own · Save
        preparable --> waitsprep : the template edited so it does not resolve, or put into a launched group · Save
    }
    state "prepared, never launched" as prepared {
        state "ready to launch" as ready
        state "waits for a run it builds on" as waits
        state "its job's members changed" as regrouped
        [*] --> ready
        [*] --> waits
        waits --> ready : that run finished · the run
        ready --> waits : that stage launched again — or, for a transport rung, its upstream described anew · send_launch, Save
        ready --> regrouped : the job request regroups it · Save
        waits --> regrouped : the job request regroups it · Save
    }
    state "launched" as launched
    state "its launch record does not read" as unreadable
    state "removed" as removed
    [*] --> described : molbuilder describe, or Task setup's Save · describe
    described --> prepared : prep task --group N · prep_task
    described --> launched : prep task --group N, a group a transport calculation reuses, its runs copied in · prep_task
    launched --> described : restore the state saved before the prep that copied it in · checkpoint
    prepared --> prepared : prep task --group N again, its group whole · prep_task
    prepared --> launched : its group's first launch · send_launch
    launched --> launched : launched again — alone, or its group whole · send_launch
    launched --> launched : prep task --group N, its job request changed — what it computes the same · prep_task
    launched --> prepared : restore the state saved before its group's first launch · checkpoint
    prepared --> described : restore the state saved before its prep · checkpoint
    prepared --> unreadable : its run.json edited, and not readable · by hand
    launched --> unreadable : its run.json edited, and not readable · by hand
    described --> removed : the description drops it · Save
    prepared --> removed : the description drops it · Save
    launched --> removed : the description drops it · Save
    removed --> launched : described again, a launched one · Save
    removed --> prepared : described again before any prep dropped its row · Save
    removed --> described : described again after a prep dropped its row · Save
```

| state | what the files say | what `status` shows | what may come next |
|---|---|---|---|
| **described**, preparable | `task.json` names it; `job-set.json` holds no row for it | *described* — what its first launch will take | `prep task --group N` |
| **described**, waits before it can be prepared | the same; a template that does not resolve, or a stage put into a group that was launched without it | *described* — *waits before it can be prepared:* and why | the template fixed; a group of its own (R-P5) |
| **prepared, ready to launch** | its row in `job-set.json`, its deck and scripts rendered **from the description and its job request as they stood when you prepared it**, its first run opened with its own files and nothing taken; every run it builds on has finished | *pending* — *ready to launch* and what it takes; prepared by another molbuilder (its deck's own record), which one | its group's first launch; or prepared again |
| **prepared, changed since prepared** | the same, and its records — the description, the template, the job request — resolve now to other values than the ones its prep recorded (*Changed since prepared*, below) | *pending* — *changed since prepared:* each value that differs, `was → now`, and *prepare it again* | prepared again — nothing ran, so prep writes its files anew |
| **prepared, waits for** a run it builds on | the same, and the newest run of a stage it builds on has not finished, or is not there, or — a transport rung — did not run the deck its description gives now | *pending* — *waits for:* and what | the stage it names launched or let finish — the refusal names it; or prepared again |
| **prepared, its job's members changed** | its row's `members` lists other stages than its group holds now | *pending* — *its job's members changed:* | prepared again, every group the old job's members are in now named in the one prep (R-P6); its launch is refused, naming that prep |
| **launched** | a run of it has a `run.json`; its runs' states are the run doors' (*A run*, below) | the newest run's state, and *converged* in its own column; *changed since prepared*, each value that differs, when its records resolve now to other values than its prep's — its runs computed the description as it was — with the three lines that start its group from the beginning (*Changed since prepared*, below) | launched again — alone, or its group whole: warm, `--cold`, `--from` a run, `--gather` anew; prepared again for its job request, what it computes kept — on the hierarchical layout, its row then what its next launch sends, its runs' states unchanged ([`job-request.md`](?doc=execution/job-request.md) § 5); a change to what it computes goes back to the state saved before its group's first launch (*How the checkpoint supports you*) |
| **its launch record does not read** | a `run.json` there that is not one molbuilder wrote | *unreadable* — the record's own words | what the words say; nothing launches it, and no prep prepares it, until it reads (R-L21) |
| **removed** | no longer in `task.json`; its folder and runs untouched, its number taken; a launched one keeps its row in `job-set.json`, a never-launched one until the next prep that writes drops it | not listed | described again, it is what its files say — launched, prepared while its row stands, else described |

*Ready to launch* says that what the stage takes is there.

**Changed since prepared** *(the user, 2026-10-25)* says that a stage's records
no longer give what it was prepared with. Prep records, for each stage it
prepares, the values it prepared the stage with — every setting as resolved
for it (the template ⊕ the stage's overrides), its points (a rung of points'
frames and voltages) and its run card (the job request's values that reach
it) — in `prepared-with.json` at the calculation's root, merged per stage as
`job-set.json` is. `status`, Task setup and the Results ladder resolve the
same values from the records as they read now — a read that renders and
writes nothing — and name each that differs, `mesh_cutoff: 300 → 400`. Only
what the person changed is a difference: a deck a newer molbuilder words
otherwise is none. A stage whose prep recorded none (prepared before 2.41.0)
says so — *what it was prepared with is not recorded (prepared before
2.41.0)*, one spelling on every surface (`prepared_with.UNRECORDED_SAID`);
prepared again, it is recorded — and is compared with nothing. **Launch compares nothing**: it sends what prep
wrote (§ 6.0). The way on:

- **never launched** — *prepare it again*: nothing ran, and prep writes its
  files anew (`prep task --group N`);
- **launched, its job request changed alone** — the run card, the
  allocation, the notification: on the hierarchical layout, *prepare its
  group again for its job request; its runs are kept* — `prep task --group
  N`, as a run out of memory goes on with a bigger request
  ([`job-request.md`](?doc=execution/job-request.md) § 5); on the flat
  layout, as below;
- **launched, changed in what it computes** — its runs computed the
  description as it was; to compute it as described now, **start the group
  from the beginning**: save the folder as it stands, so its runs are kept
  in a saved state; go back to the state saved before the group's first
  launch; prepare the group again. The three lines, printed whole, the
  state named by its id and the folder by its full path
  (`commands.redo_launched`):

```
molbuilder checkpoint save -p /…/P/opt/H2 -m 'group 2 (medium, tight) as launched, before going back'
molbuilder checkpoint restore -p /…/P/opt/H2 4f9ca71
molbuilder jobset prep task --group 2 --bundle /…/P/opt/H2
```

*A group* — the stages one number holds, matched to what is prepared by its
members, never its number (`ready.GroupAnswer`):

```mermaid
stateDiagram-v2
    direction LR
    state "not prepared" as np
    state "prepared, never launched" as pr
    state "prepared, its members changed" as ch
    state "launched, its members fixed" as la
    state "launched, its members changed" as lc
    state "goes in job N — a sequence still going holds it" as sq
    [*] --> np
    np --> pr : prep task --group N · prep_task
    np --> la : prep task --group N, a group a transport calculation reuses, copied in · prep_task
    la --> np : restore the state saved before the prep that copied it in · checkpoint
    pr --> pr : prep task --group N again, whole · prep_task
    pr --> ch : the job request regroups a member · Save
    ch --> pr : prep task, every group the old job's members are in · prep_task
    pr --> la : launch task --group N, whole — the folder saved first · send_launch
    la --> la : a member launched again alone, or the group again whole · send_launch
    la --> la : prep task --group N, its job request changed — what it computes the same · prep_task
    la --> pr : restore the state saved before its first launch · checkpoint
    la --> lc : a member removed from the description, or job-request.json edited by hand · Save, by hand
    lc --> la : the members put back as launched · Save
    lc --> pr : restore the state saved before its first launch · checkpoint
    pr --> sq : launch task --group A --group … naming it · send_launch
    la --> sq : a sequence naming it, to launch it again · send_launch
    sq --> la : its turn in the job — its own launch inside · send_launch
    sq --> pr : the job ends, or is stopped, before its turn · the job / jobs stop
    sq --> la : the job ends before its turn, a group launched before · the job / jobs stop
    pr --> np : restore the state saved before its prep · checkpoint
    ch --> np : restore the state saved before its prep · checkpoint
```

**One answer for a group, which every verb and surface reads** *(2.28.0, plan
§ 1w: 2.27.0 changed what may be done to a launched group, and of the six
places that each decided it, three followed)*. Each group's
`ready.GroupAnswer` (`ready.task_state(calc).groups`) → the group's state —
one of the diagram's, by its name there (`said`), the words every verb and
surface shows — whether it **may be prepared now** (`preparable`) and,
when not, why (`why`: a launched group whose members changed says so in the
one rule's words, `members_changed` — `ready.launched_without`,
`ready.joined_launched` — and one the flat layout does not prepare again in
prep's own, `not_again` — `ready.launched_already`); whether it **may be launched now** (`launchable`);
and a stage's **way on**, composed once (`commands.way_back`, layout-aware: a
stage never launched, its group's prep; a launched one on the hierarchical
layout, its group's prep for how it runs, the state saved before its group's
first launch for what it computes; on the flat layout, that state).
`prep task`'s offer and its refusals (R-P5, R-P22, R-P23), `launch task`'s
offer and its refusals (R-L6–R-L10), the command line's question, `status`'s
next step and Task setup's boxes and group cells all read it — **none decides a
group's state, or words its way on, itself**.

`prep task` offers every group the answer says **may be prepared now** and is
not prepared as described (`prep task`, 2); `launch task` offers every group
**prepared as described, never launched**, each member's upstream finished.
**A launched group's members are fixed**: Task setup's Save refuses a stage put
into it, or a member moved to another group, in the answer's words — a group
of its own for the stage, the member put back, the launched one launched again
alone, or the restore (`ready.joined_launched`) — and a group
whose members changed anyway, a member removed from the description or
`job-request.json` edited by hand, is **launched, its members changed**:
offered by no verb, its prep refused ([R-P5](#every-refusal-and-its-way-on))
and its launch whole refused ([R-L7](#every-refusal-and-its-way-on)) — each
member launched again alone, the members put back, or the state saved before
the group's first launch restored.

*A run* — one launch of a stage. On the hierarchical layout each is a folder,
`run-0`, `run-1`, …; on the flat layout each is its own numbered files in the
calculation's folder, and **no run is opened before its launch** — the flat
layout has no *opened* state:

```mermaid
stateDiagram-v2
    direction LR
    state "opened, not launched" as op
    state "launched" as la {
        state "queued" as q
        state "running" as r
        state "finished" as f
        state "failed" as x
        [*] --> q
        q --> r : the scheduler starts it · the run
        q --> f : it ran and ended between two looks · the run
        q --> x : stopped before any output · the run
        r --> f : its conclusion, exit 0 · the run
        r --> x : an error, or stopped (killed, out of time) · the run
    }
    state "unreadable" as un
    [*] --> op : prep opens run-0 · prep_task — a launch again opens run-N · send_launch
    [*] --> la : prep copies a finished calculation's run in, a transport calculation reusing it · prep_task
    op --> op : prep again, or a send refused — the next fill empties it whole · prep_task, send_launch
    op --> la : the send went — its run.json written · send_launch
    la --> un : its run.json edited, and does not read · by hand
    la --> [*] : a restore to a state before it · checkpoint
    op --> [*] : a restore to a state before it · checkpoint
```

A run **opened, not launched** — prep's `run-0`, or a `run-N` a launch opened
whose send was refused — is **emptied whole** by whatever fills it next, then
filled as that act says ([`project-layout.md`](?doc=execution/project-layout.md)
§ 1.6.3). It is not yet a run in the sense of *The words* — a refused
send writes no `run.json` — but it is the stage's newest, so a stage that
builds on it waits for it (an older run never stands in), and `status` names it
with the launch that fills it: prep's `run-0`, its group's launch (a group
goes whole the first time, and a refused first send left it never launched);
a `run-N` a launch again opened — *opened by a launch whose send was refused*
— `--stage`, the stage launched again alone. A first send refused saved the
folder's state first; the next one saves it again when anything changed, and
the way back is the state saved before the first launch that went. **A run of points** (a stage that
sweeps the bias) is one run, its state one rule that `status`, the gather,
`jobset scratch` and the transport record all ask (`continuation.points_state`,
`continuation.stage_run_state` for any stage's run): *running* while a point runs, or
between two points while a job launched from this machine that holds the run
is going (`joblist.is_going`, § 6.0a — said *(between two points)*), never
the job a reader runs inside (a sequence's group launch, § 6.0);
*queued* while no point the run walks itself has started and its job is not
over — asked once of the job, else of the run door, which asks the scheduler
and writes its answer (`running-a-job.md` § 4.2): a job cancelled in the
queue, or a process that died before its first point, is a run that ended,
*— its job ended before any point started*; else *finished*
once every point's run finished, converged or not — the *converged* column
says *SCF NO* when one did not — or *failed* when a point's run did not
finish, naming the first point not done. A walk that stopped among its
points — some run, the rest never reached — is a run that ended, failed,
never one still queued; where no job holding the run is on file, or this machine
cannot tell, the points say the state. Its row counts the points done.

#### The next step — every group, where it stands, and the line that moves it

*(The user, 2026-10-15: "improve the message of launch so it is clear to the
user what to do for each grouped launch" — "the current design is clear about
the first step, but unclear about the follow up"; then: "maybe this
information is best presented in the result tab where the whole task was
viewed … and give user some hint if user want to continue what's the command
to launch".)*

**One answer for the whole task, from the ladder door** (`runstatus
.jobset_status`, its `steps`): every group of the job request, in the description's order,
with its stages, a few words for where it stands, and the line that moves it
— or what it waits for, when no line can. The words are read off its members'
rows, the same rows `status` prints, so the two cannot disagree; every line is
composed by `commands`. Its wire form, in the ladder answer the Results tab
reads, is `steps: [{group, stages, state, words, lines, next}]` — the group's
number and stages, its state, its words, its lines (none when it waits), and
whether it is the next step. A group's row is the first of these that applies,
read top to bottom:

| the group's members | the group | the line |
|---|---|---|
| launched in a job whose members changed — a member removed, or a stage put in by a hand edit of the job request, prepared or not | **launched, its members changed** — which stage was put in or taken out | its way on, the answer's (`ready.joined_launched`), whole: each member launched again alone, the members put back, or the restore |
| one not prepared, and the group preparable (`ready.GroupAnswer`) | **not prepared** | `prep task --group N` |
| one not prepared, and the group not preparable | **waits before it can be prepared**, and what for — a template that does not resolve, a group launched without it | — |
| prepared in a job whose members changed | **prepared, its members changed** | `prep task --group N` |
| prepared, and held by a sequence still going that has not launched it yet — launched before or not, changed since its prep or not (§ 6.0, *A sequence of groups*; R-P23 refuses its prep) | **goes in job N** — after group K, the one before it in the job | `molbuilder jobs status` |
| a member changed since prepared — prepared, or launched with no run queued or running — and held by no sequence | **changed since prepared** — the member and each value that differs, `was → now` | never launched: `prep task --group N`; launched and changed in its run card alone, on the hierarchical layout: `prep task --group N`, its runs kept; launched on the flat layout, or changed in what it computes: the three lines that start the group from the beginning (*Changed since prepared*) |
| prepared, never launched, what each takes there | **ready to launch** — and, for a member another molbuilder prepared (its deck's own record), who did | `launch task --group N`, a line for each mode the machine takes |
| prepared, never launched, a member waiting | **waits for** the run it takes — that run's stage and group, and where it stands | — |
| a run queued or running | **running** (or *queued*) | `status` |
| a run failed or stopped, or a launch record that does not read | **failed** (*stopped*, *unreadable*), naming the stage | its launch again — `launch task --stage S`, which continues from its own latest run or starts over as its job says (`relaunch_continues`), `--cold` starting it over; and its way on, the one composer's (`commands.way_back`) — on the hierarchical layout a run out of memory or wall goes on with its job request changed and its group prepared again ([`job-request.md`](?doc=execution/job-request.md) § 5), what it computes changed by the restore; for a record that does not read, its reader's words |
| a run opened and not launched — a send refused | the stage, and *opened by a launch whose send was refused* | the launch that fills it (`commands.launch_pending`): `launch task --stage S` |
| every member's newest run finished | **finished** | — |

**The next step is the first group not finished**, marked *next*, its lines
under it; the groups after it each say where they stand — a waiting one, the
ready door's own reason. With every group finished, the last group says so —
and, for a calculation whose result gathers its stages, is the next step,
naming `summarize task`: one rule, asked by the step and by the verb alike
(`summarize.task_summary`: a transport calculation; a SIESTA vibration with
two or more stages, a displacement sweep). A benchmark's sweep keeps its own next
step, its own verbs (§ 7).

**Shown where the next step is asked, and nowhere else:**

- **`status`** ends with it, after the stages' table; `status <stage>` shows
  that one stage in full, and no next step (§ 5.3, *The read-only verb*);
- **`launch task`** ends with it, read after what it sent — the group just
  sent *running*, or *finished* for a run here that has ended;
- **the Results tab's ladder card** draws it under the rows, read only — the
  next step's words and line, the later groups' words; the page launches
  nothing ([`web/results.md`](?doc=web/results.md) § 2.4);
- **Task setup's Prep card** lists it under the groups, read from the
  folder, whatever the page last did — a Save, a reload, a Prep (the user,
  2026-10-23: *"the prep should correctly just state what can be done
  next (such as launch cmd etc)"*; [`web/task-setup.md`](?doc=web/task-setup.md)
  § 11) — and `prep task` at the terminal keeps its own closing: the launch
  with nothing named, which sends the group that can go, or, a prep that
  opened no run of its own, this answer (§ 5.3, *One prep, two doors*).

For `AuBDTAu/transport/lowbias_local`, its device just sent:

```text
each group, and what is next:
  group 1  seed, electrode_L, electrode_R  finished
  group 2  device                          running, let it finish -- next:
           molbuilder jobset status --bundle AuBDTAu/transport/lowbias_local
  group 3  transmission                    waits for: the transmission stage consumes lowbias_local.TS.HSX from device's newest run, 04_device/run-0, which is running.  Let it finish
```

and once the device has finished:

```text
each group, and what is next:
  group 1  seed, electrode_L, electrode_R  finished
  group 2  device                          finished
  group 3  transmission                    ready to launch -- next:
           molbuilder jobset launch task --group 3 --bundle AuBDTAu/transport/lowbias_local
```

and once the transmission has finished, the last group names the summary:
*finished -- every group has; read what they gathered -- next:* `molbuilder
jobset summarize task --bundle AuBDTAu/transport/lowbias_local`.

#### `prep task`

One entry, which the terminal and Task setup's **Preview task** and **Prep
task here** all call (§ 5.3, *One prep, two doors*); § 5.0 holds each
checkpoint in full.

1. **It reads the description and shows the groups** — each with its stages,
   prepared or not, and for a stage not preparable yet, what it waits for.
2. **It asks which groups to prepare**, offering, ticked, every group not
   prepared as described that the group answer says may be prepared now — a
   changed one among them, never a launched group whose members changed
   (`ready.GroupAnswer`). `--group N` answers without a terminal (repeat
   it for several), and prepares a group already prepared again — a launched
   one too, for a change of its job request ([`job-request.md`](?doc=execution/job-request.md) § 5); with no terminal and none named, the prep is refused, naming the
   line that picks them — nothing is prepared unseen. **A group prepared,
   never launched, and already as this prep would prepare it now** — every
   file each member's run takes, by the catalogue (`runfiles.WRITTEN`'s
   `kind`: an input it reads, or a file derived for the stage or its run —
   its deck, scripts and headers, the bundles beside them, never a record),
   and its row, the files prep plans now, their stamps aside — **has nothing to
   write**: the answer says so, naming the group, with its next step, and
   nothing is saved, written or recorded for it *(the user, 2026-10-23: "if
   nothing actually needs to be saved or nothing has changed, the prep
   should correctly just state what can be done next")*. One that differs
   is prepared again whole, its answer naming each file that differs — and
   its header, or that its members changed. **A sequence of groups
   (`--one-job`) is one job**: nothing to write when every group of it
   stands and its one header is as written, else every group of it is
   prepared.
3. **Each group is one job.** Its members share one allocation — what they
   share is [`project-layout.md`](?doc=execution/project-layout.md) § 1.6.6's
   — and **the group's request is placed** on the queue it names, its fit
   said, as a run's is (§ 6.0). A group is matched to the description by **its
   members**, never by its number: removing a stage above it shifts numbers and
   changes no job. A pick whose groups' members changed since they were
   prepared, never launched, prepares every such group again in the same
   prep; naming only some of them is refused, naming the others.
4. **The description is checked and every picked group's members planned**
   — the machine and each member's placement, its decks planned and checked
   — **then the folder is saved once, and every picked group
   is written**: each stage's deck rendered now, its first run opened with the
   stage's own files, its row in `job-set.json`; each job that walks several
   members, its one header — a group of several's in the calculation's
   `launch/` folder, a run of points' walk's in its stage's, a benchmark's
   shelves' in its own. **Nothing is copied from
   another stage's run** — that is the launch's — but a group a transport
   calculation reuses from a finished one, whose members' runs this prep
   copies in, opening none of their own ([`engines/transport.md`](?doc=engines/transport.md) § 2a.16).
5. **A stage prepared and never launched may be prepared again** — it is still
   the description's: its row replaced, its files rendered again, its first
   run emptied whole and opened anew; **a group is prepared again whole**.
   **A launched stage is prepared again only for its job request**, and on
   the hierarchical layout alone (on the flat one its runs share its
   folder's deck, so it is not prepared again) — its
   files rendered again, its row replaced, its runs untouched — and only when
   each member's deck is the same calculation as the deck its runs ran
   ([`job-request.md`](?doc=execution/job-request.md) § 5, § 6); what it computes stays what its runs ran with,
   and a change to that goes back to the restore (*How the checkpoint supports
   you*).

Each refusal node names its row in [the refusal table](#every-refusal-and-its-way-on):

```mermaid
flowchart TD
    A["prep task  [--group N …]"] --> B{"task.json, its template and<br/>job-request.json read,<br/>every stage's group stated?"}
    B -- no --> R1["R-P1 · R-P2"]
    B -- yes --> C["the groups shown, each stage's state<br/>ready.task_state"]
    C --> D{"groups named?"}
    D -- "no — a terminal" --> Q["asked which: every group not prepared<br/>as described, and preparable, offered"]
    D -- "no — nobody to ask, or none to offer" --> R3["R-P3"]
    D -- "no — none to offer, a junction's order to record" --> O["the junction composed again in memory, compared;<br/>its order record written, the state saved first — nothing else<br/>prep._prep_groups"]
    O -- "a refusal" --> R21["R-P15 · R-P21"]
    D -- yes --> E
    Q --> E{"groups the job request holds?<br/>no member launched — or each launched member's deck<br/>the same calculation as its runs ran;<br/>with --one-job, a launched group read as prepared?<br/>no group a sequence still going holds?<br/>every group a changed job's members are in named?"}
    E -- no --> R4["R-P4 · R-P5 · R-P6 · R-P22 · R-P23"]
    E -- yes --> K["the description's preflight, the machine,<br/>which files answered — once<br/>prep.checked"]
    K -- "a refusal" --> R8["R-P8 · R-P9"]
    K --> F["each member planned into the one plan, nothing written:<br/>its inputs · its launch values, placed ·<br/>its decks and run-0<br/>prep.plan_stage"]
    F -- "a refusal" --> R10["R-P10 · R-P12 · R-P13 · R-P16 – R-P21"]
    F --> G["each group of several: one allocation,<br/>its one header placed on its queue<br/>(a run of points' walk and a benchmark's shelves<br/>are their member's, in F)"]
    G -- "a refusal" --> R13["R-P13"]
    G --> GS["--one-job: the sequence's one header —<br/>the groups' own requests together<br/>prep.sequence_header"]
    GS -- "a refusal" --> R22["R-P22"]
    GS --> NW
    G --> NW{"each group prepared and never launched:<br/>its members' decks, scripts and rows and its header<br/>as this plan writes them now?"}
    NW -- "every group" --> NWA["nothing to write: each said, with the next step —<br/>no state saved, nothing written, no record"]
    NW -- "the others" --> H{"a preview?"}
    H -- yes --> P["the plan shown — nothing written"]
    H -- no --> PI{"a Prep naming a previewed plan:<br/>the plan made now?"}
    PI -- no --> R14["R-P14"]
    PI -- yes --> S["the folder's state saved, once —<br/>… · before prep task group 1 (seed, electrode_L, electrode_R), group 2 (device)"]
    S -- "a refusal" --> R15["R-P15"]
    S --> W["the plan written as it was made: each member's deck,<br/>scripts and run-0, its row naming its job's members;<br/>each group's header; a changed job's old header removed"]
    W --> X["the record: the question, the save, the description's notes,<br/>each member prepared, each group"]
    X --> Y["the answer: what each stage's first launch will take,<br/>each group's one job, and the launch with nothing named"]
```

#### `launch task`

The plan, shown, asked once, sent (§ 6.0):

* **It sends the next group** — one prepared as described and never
  launched, whose every member's upstream has finished. When several can go
  it asks which (`--group N` answers), so it is one job per invocation. A
  group goes whole the first time; a member is launched alone only again.
* **At a group's first launch, each member takes what it builds on** (*What a
  stage takes, and when*). `--from` names another run, and `--cold` none, for
  an independent ladder's stage launched in a group of its own. While an
  upstream has not finished, the launch is refused before anything is sent,
  saying which and what to do.
* **It sends what prep wrote, and compares nothing** (*The task*, *Changed
  since prepared*): a stage whose records resolve now to other values than
  its prep's is said so by `status` and Task setup, each value that differs,
  and prepared again — never launched, its group; launched and changed in
  its run card alone, its group again for its request, the runs kept
  ([`job-request.md`](?doc=execution/job-request.md) § 5); launched and
  changed in what it computes, the group started from the beginning, the
  three lines. A plain launch again sends what prep last wrote.
* **When no prepared group can go, it offers the stages launched before**, to
  launch one again: warm by default, or `--cold` (*What molbuilder does for
  you*, 3), beside what each waiting group waits for. `--stage NAME` names one;
  `--group N` sends a launched group again whole. Launched again, a stage may
  take its inputs anew — `--from` another run, or
  `--gather` for a transport rung's upstream — decided at that launch and
  recorded; `--from` with `--cold`, and `--from` with `--gather`, are two
  answers to one question.

Each refusal node names its row in [the refusal table](#every-refusal-and-its-way-on):

```mermaid
flowchart TD
    A["launch task  [--group N … | --stage S]<br/>[--from R | --cold | --gather]"] --> L0{"a job set prepared,<br/>its description read?"}
    L0 -- no --> R1a["R-L1 · R-L27"]
    L0 -- yes --> M{"its mode, and every flag<br/>the mode and the launch take?<br/>commands.LAUNCH_FLAGS"}
    M -- no --> R4["R-L4 · R-L46 — --in-sequence / --in-job, the walk's alone"]
    M -- yes --> B{"named?"}
    B -- nothing --> O{"ready.task_state: groups prepared as described,<br/>never launched, upstream finished"}
    O -- one --> MR
    O -- several --> Q["asked which — one job a launch"] --> MR
    O -- none --> RL{"stages launched before?"}
    RL -- yes --> QS["offered, to launch one again —<br/>beside what each waiting group waits for"] --> SA
    RL -- "no, or nobody to ask" --> R1["R-L1 · R-L2 · R-L3"]
    B -- "a group" --> MR
    B -- "several groups" --> MR
    B -- "a stage" --> SA{"held by the description,<br/>and prepared?"}
    SA -- no --> R5["R-L5 · R-L36"]
    SA -- yes --> MR
    MR{"the calculation's copy of its machine's record reads?<br/>submit._machine_of — first, once, every mode"}
    MR -- no --> R38["R-L38"]
    MR -- "yes, several groups" --> SQ{"a sequence: each group as its own launch asks it —<br/>prepared, held by no sequence still going;<br/>no one-stage flag; to a queue, the header prep wrote<br/>for exactly them<br/>submit._plan_sequence"}
    SQ -- no --> R45["R-L45"]
    SQ -- yes --> SQP["the first group planned as its own launch plans it now;<br/>the walk — each group's own launch, in order —<br/>and the sequence's launch record"]
    SQP --> SH
    MR -- yes --> GC{"the work, submit._the_group: a group the job request holds,<br/>every member prepared, as launched or as prepared;<br/>a stage never launched, alone in its group;<br/>--from, --cold, --gather for one stage;<br/>held by no sequence still going — but its own;<br/>inside one, the group the sequence was sent with"}
    GC -- no --> R7["R-L6 – R-L10 · R-L26 · R-L45 · R-L46"]
    GC -- yes --> MB["each member planned, nothing written<br/>submit._plan_member"]
    MB -- "a refusal" --> R14["R-L14 · R-L21 · R-L28"]
    MB --> F{"its first launch?"}
    F -- yes --> T{"what it builds on is there?<br/>ready.first_launch_takes"}
    T -- no --> R11["R-L11 – R-L14 · R-L37"]
    T -- yes --> F0["run-0 emptied whole, prep's files back as they lie,<br/>and what it takes"]
    F -- "no — again" --> AG["from its newest launched run, warm;<br/>--cold nothing; --from a run, --gather anew —<br/>each checked as a first launch is"]
    AG -- refused --> R17["R-L11 – R-L14 · R-L17 · R-L30 · R-L37"]
    AG --> FN["run-N: new, or the one a refused send opened,<br/>emptied whole — then filled"]
    F0 --> PL["the submission: one job, a group's walk,<br/>a run's points' walk —<br/>the header its prep wrote"]
    FN --> PL
    PL -- refused --> R19["R-L18 · R-L29 · R-L31 · R-L42 · R-L44"]
    PL -- "the scheduler asked: mode ask" --> AK["each line asked of the scheduler, its kind's ask —<br/>nothing saved or sent; what was asked recorded"]
    PL --> SH["shown: each member, what it takes and from which run;<br/>the exact line, every value of its header"]
    SH -- "a dry run" --> DRY["nothing asked, nothing written —<br/>the ledger included"]
    SH --> ASK{"yes? — a run here: in the background,<br/>or the other way (--yes answers it)"}
    ASK -- no --> NO["nothing sent — the question and its no recorded"]
    ASK -- "nobody to ask" --> R23["R-L23"]
    ASK -- yes --> SF{"the folder as it was planned?<br/>the plan made again from it"}
    SF -- no --> R22["R-L22"]
    SF -- yes --> SV["a group's first launch: the folder's state saved, and said —<br/>… · before launch task group 2 (device)"]
    SV -- "a refusal" --> R24["R-L24"]
    SV --> WR["the questions recorded; the plan written:<br/>each run filled, its .continued-from / .gathered-from"]
    WR --> SEND["sbatch, or the process here — one job"]
    SEND -- refused --> R25["R-L25 · R-L35 — each run stays opened, not launched"]
    SEND -- went --> REC["each member's run.json — launched;<br/>the job's address (§ 6.0a)"]
    REC --> WT{"a run here, waited on?"}
    WT -- yes --> HE["each member's ending, read through the status door:<br/>finished — or FAILED, or UNREAD when it cannot be read: exit 1"]
    WT -- no --> NX
    HE --> NX["the next step: every group, where it stands,<br/>the line that moves it — runstatus.jobset_status"]
```

#### Every refusal, and its way on

Every refusal is written down (`jobset-decisions.log`) — but a dry run's and
a preview's, which write nothing — names the way on in lines you can type,
and comes **before anything is written** — the send's own (R-L25, R-L35)
after its save, with nothing launched. The module is where it is decided. A
reader's refusal — a record that does not read, a job of the plan no file
can be named for, a calculation's own `warm-files.toml` that does not read
— is the verb's, in its reader's words, never a traceback.
The command line's own usage refusals — a word where none is taken,
`--stage` twice, `--group` with `--stage`, an answer that is not a group's
number, a `--time`, `--mem` or `--gpus` that is no amount — are said where
they are typed, and written down too.

| | when | the way on | decided in |
|---|---|---|---|
| **R-P1** | the folder is not a described calculation, or is one of its stage or run folders; or its description has no job request beside it | `molbuilder describe`; or the calculation's folder, named; for a missing job request, Task setup's Save, which writes both — a description written before (`task@1`) is refused first, naming `jobset migrate` | `calculation.open`, at each verb's entry (`calculation.refusal`, the one sentence) |
| **R-P2** | `task.json`, its template or `job-request.json` does not read — written by an older molbuilder, or a stage with no group | its reader's words; an older one names `jobset migrate` | `task.of`, `job_request.of` |
| **R-P3** | no group named, and nobody to ask (not a terminal) — or none to offer, and no junction order to record (a transport calculation whose junction's sidecar states none records it instead, nothing else written: [`engines/transport.md`](?doc=engines/transport.md) § 1.1, *No group left to prepare*) | the line that picks them; each group says where it stands | `_cli._ask_which_groups`, `prep._prep_groups` |
| **R-P4** | a group the job request does not hold | the groups it holds | `prep._prep_groups` |
| **R-P5** | a member of a group picked — or of a group a changed job's members are in — was launched, and the layout is flat, or its deck rendered now is not the same calculation as the deck its runs ran, or its points (a rung that runs per point) are not the ones its runs ran ([`job-request.md`](?doc=execution/job-request.md) § 6), each difference named; or the group holds a stage put beside one launched without it, or no longer holds one it was launched with | the state saved before its group's first launch; for a stage put in, a group of its own; for one taken out, put back — the group answer's way on | `prep._prep_groups` (`ready.launched_without`, `deck_record.same_calculation`) |
| **R-P6** | a job whose members changed has members in groups not picked | every group to pick, in one line | `group.formed_again` |
| **R-P8** | the description fails its own checks — a stage sharing a group with one it builds on among them | the check's words: give it a new group — or, for an independent ladder's stage, set its `restart` override to `clean`, and it builds on nothing | `prep.checked` (`validation.task.preflight`) |
| **R-P9** | the machine: no record answers, or its record states no activation | the record named, or re-probed | `prep.checked` (`machine.require_activation`) |
| **R-P10** | a launch value not stated (processes; for a header, queue, wall and memory), or a queue the record does not list | the value to state, the record's queues listed with their machines and the fit of what is stated | `prep.plan_stage` (`placement`) |
| **R-P12** | a deck's own checks refuse it; a data file it needs is missing | the check's words | `prep.plan_stage` (`prep_calculation`) |
| **R-P13** | a job that walks several members — a group, a run of points, a benchmark's shelf — whose members cannot share one allocation | the same values for each, or groups of their own | `prep.walk_header_of` (`group.envelope`, `shelves.shelf_request`) |
| **R-P14** | a Prep whose plan differs from the one previewed | preview again | `prep._prep_groups` |
| **R-P15** | the folder's state cannot be saved | the save's words | `checkpoint.save_before` |
| **R-P16** | the structure is missing, or not the one described | the structure put back, or the calculation described anew | `prep._structure_for` |
| **R-P17** | a run's GPU request disagrees with itself (GPUs used and none counted, or counted and not used) | the count stated, or no GPU | `prep.plan_stage` (`model.gpu_request`) |
| **R-P19** | a SIESTA vibration described flat — its run's calls would share one folder | the hierarchical layout | `prep._siesta_vibration_rung` |
| **R-P20** | a stage the description no longer holds has a launch record that does not read | the record's own words | `prep.prep_calculation` (`materialize.ever_launched`) |
| **R-P21** | a transport calculation's junction record is not usable and a stage is launched — a launched stage's junction is never replaced; or, with no group named, its order is not found ([`engines/transport.md`](?doc=engines/transport.md) § 1.1) | the state saved before the launched group's first launch; with none launched, a prep of its groups | `gather.no_usable_record` |
| **R-P22** | `--one-job` with fewer than two groups, on a benchmark, or with every group named launched already; a launched group named whose members are not the ones it was prepared with; groups whose own requests name two queues, or GPU groups asking two GPU counts or binding their GPUs differently — one job sits in one queue, with one set of GPUs | the groups' run cards alike on those, or each group sent alone | `prep.sequence_header` (`group.sequence_request`), `prep._prep_groups`, `_cli.prep_cmd` |
| **R-P23** | a group a sequence of groups still going holds (§ 6.0) — its job is about to launch the files it was prepared with, and a prep now would rewrite them under it (`project-layout.md` § 1.6.3) | the prep once that job has ended, or `molbuilder jobs stop` of the job first | `prep._prep_groups` (`ready.held_by_sequence`) |
| **R-L1** | nothing is prepared | `prep task` for the first group | `_cli._nothing_prepared`, asked by `_cli._launch` (no `job-set.json`) and `_cli._launch_unit` (none of the description's stages prepared) |
| **R-L2** | no prepared group can go, and no stage was launched | each group says what it waits for | `_cli._launch_unit` |
| **R-L3** | nobody to ask which — several groups could go, or none can and stages launched before are offered | one line for each | `_cli._launch_unit` |
| **R-L4** | no `--mode` and no `launch.mode`, or a `launch` block that does not read; a flag the mode or the launch does not take (`--foreground` with a queue, `--time` here; `--cold`, `--from`, `--gather` on a benchmark, whose trial starts cold from its deck; `--only` and `--trial-timeout` on a task or with a named trial, `--only` here, `--trial-timeout` asked) — one table, `commands.LAUNCH_FLAGS` | the launch with each mode | `_cli._launch` (`_refuse_flags_without_effect`) |
| **R-L5** | the stage named is described and not prepared | its group's prep, then its launch | `_cli._refuse_unprepared` |
| **R-L6** | a stage never launched named alone while its group holds others | the group's launch | `submit._the_group` |
| **R-L7** | a launched group whose members the job request changed | a member launched again alone; the members put back; or the restore — the group answer's way on | `submit._the_group` (`ready.launched_without`, the same rule) |
| **R-L8** | a group whose prepared jobs hold other members than it does now | its prep, every group named | `ready.regrouped` |
| **R-L9** | `--from`, `--cold` or `--gather` on a group of several | `--stage`, for a member launched again | `submit._the_group` |
| **R-L10** | a member of the group named is not prepared | the group's prep | `submit._the_group` |
| **R-L11** | at a first launch, or a launch again with `--gather`: the newest run of what it builds on has not finished or is not there, or holds none of the files the stage takes; a transport rung's upstream is not prepared, a run of points has points not done, the run did not run the same engine input as the deck its description gives now (`engines/transport.md` § 6.1) — every difference named — or lacks the file; the junction a transport rung is composed from has no usable record (in R-P21's words) | that stage launched, or let finish; or the restore; for the junction, a prep of the calculation's groups | `ready.first_launch_takes` (`continuation`, `gather`, `gather.no_usable_record`); a launch again with `--gather`, `submit._gathered_now` |
| **R-L12** | a vibration's stage with `--from` or `--cold` at its first launch — it builds on the calculation's structure, its phases joined inside its run | nothing to name: launch it; `--cold` is a launch again's ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 6) | `continuation.cannot_be_named` |
| **R-L13** | `--from` on a transport rung, or `--cold` at its first launch — its inputs are its kind's; but a rung of points of a calculation that reuses a finished one, whose first launch takes that calculation's points over: there `--cold` takes none ([`engines/transport.md`](?doc=engines/transport.md) § 2a.16) | `--gather`, launched again; `--cold` launched again | `continuation.cannot_be_named`, `ready.first_launch_takes` |
| **R-L14** | `--gather` at a first launch, on a stage that is not a transport rung, on a transport rung that gathers nothing (the seed, a lead), or with `--from`; `--from` with `--cold`; on the flat layout, `--from` or `--gather`, and `--cold` at a first launch | the flag that does it, or none — a flat stage starts clean by its `restart` override; a rung that gathers nothing, launched again without `--gather` | `submit._plan_member`, `continuation.cannot_be_named` |
| **R-L17** | launched again warm, and its newest launched run left none of the files it takes | `--cold` | `submit._launch_again` |
| **R-L18** | a run of points launched again warm, every point done; or, at its first launch in a calculation that reuses a finished one, every point one that calculation computed the same — nothing left to walk ([`engines/transport.md`](?doc=engines/transport.md) § 2a.16) | `--cold` | `submit._points_walk` |
| **R-L21** | a record the plan reads does not read — a run's launch record, `.continued-from`, `.gathered-from` — or a job of the plan no file can be named for: a deck named on no stage, or not on its label; or the calculation's own `warm-files.toml` does not read — not TOML, or not stamped `molbuilder/warm-files@1` (*`<path>` does not read: …*), or another engine's, or a section or row the vocabulary does not take | the record's own words | each reader (`runrecord.LaunchRecordError`; the naming authority, `naming.NamingError`; the warm-file vocabulary, `warmfiles.WarmFilesError`), refused once by `submit.plan_launch` (`_refusal_kinds`) |
| **R-L22** | at the send, the folder is not what it was when planned — the plan made again from it now (the calculation opened again, a benchmark's set read again) differs, or is refused | launch again, to see the new plan | `submit.send_launch` (`_same_folder`) |
| **R-L23** | at the send, nobody to ask and no `--yes` | `--yes` | `submit.send_launch` |
| **R-L24** | the folder's state cannot be saved | the save's words | `checkpoint.save_before` |
| **R-L25** | the scheduler refuses the request | the scheduler's words; each run stays opened, not launched, and the next launch fills it anew | `submit._go` |
| **R-L26** | a group the job request does not hold | the groups it holds | `submit._the_group` |
| **R-L27** | the description does not read | its reader's words | `calculation.open`, at the launch's entry |
| **R-L28** | a stage on the hierarchical layout with no run open | its group prepared again, or the restore | `submit._plan_member` |
| **R-L29** | a benchmark trial's deck that would not start cold | the state saved before the benchmark's prep | `agreement.check_trial_starts_cold` |
| **R-L30** | launched again, the inputs gathered for its run are gone | `--gather` | `submit._carry_the_gather` |
| **R-L31** | a stage, a group, a run of points or a benchmark's shelf sent to a queue with no header — prepared with `--no-sbatch`, or for a machine whose record names no queue | its group prepared again, with a header; a benchmark, the state saved before its prep | `submit._no_sbatch` |
| **R-L35** | the line could not be run at all | its words | `submit._go` |
| **R-L36** | a stage the description does not hold | the stages it holds | `_cli._described_stage` |
| **R-L37** | `--from` naming no run of this calculation's stages, or a run never launched — the run this launch fills, opened by a refused send, among them; a run holding none of the files the stage takes — a stage whose override says `restart: clean` takes none; `--from` or `--cold` at a vibration stage's first launch, which builds on the calculation's structure and has nothing to start cold from ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 6: `--cold` is a launch again's) | a run that has run, and holds what the stage takes; at a vibration stage's first launch, neither | `continuation.cannot_be_named`, `materialize.continuation_files` |
| **R-L38** | the calculation's copy of its machine's record is missing or does not read | a calculation is launched after its prep, which copies the record; its record's words | `submit._machine_of` (`record.calculation_environment`) |
| **R-L39** | a benchmark's trial named that was launched before | read what it measured — `summarize bench`; to measure it again, the state saved before the benchmark's prep | `submit._plan_trial` |
| **R-L40** | a benchmark whose trials — of the side `--only` names — are all launched, or `--only` a side the sweep has none of | `summarize bench` | `submit._all_launched`, `submit._plan_shelves` |
| **R-L41** | `launch bench` on a calculation that takes no benchmark, with no stage named, on a stage with no benchmark prepared, a trial the sweep does not hold — a trial is named by its point (`G1K4C6`) — or the kind the line says is not the set's | the benchmark's prep; the stages with one prepared; the sweep's trials; the kind the set is | `_cli._stage_bench_dir` (`prep_inputs.bench_refusal`), `_cli._load_bench_set`, `_cli._pick_trial`, `_cli._check_kind` |
| **R-L42** | a run script missing where it is started: a stage, or a benchmark's named trial, run here (`submit._plan_stage`); a trial a benchmark's walk starts (`submit._bench_trials`) | its group prepared again, or the restore; a benchmark, the state saved before its prep | `submit._no_run_script` |
| **R-L43** | a benchmark's trials in more than one container, or not shelved as its prep shelved them | the state saved before the benchmark's prep | `submit._one_container`, `shelves.shelves_of` |
| **R-L44** | a stage, a group, a run of points or a benchmark's shelf sent to a queue, or asked of it, whose header states no wall (its scheduler kind's `wall` directive) and no `--time` restates one — prep writes one into every header, so this one was edited since; the header named | prepared again where nothing of it was launched, or the state saved before it was edited restored (`molbuilder checkpoint list`); `--time` states the wall for this launch alone | `submit._sbatch_request` |
| **R-L45** | a sequence of groups (§ 6.0, *A sequence of groups*) with a group not prepared; a group a sequence still going holds, launched alone or in another sequence; under `submit` or `ask`, no header prep wrote for exactly these groups; a flag that speaks for one stage (`--stage`, `--cold`, `--from`, `--gather`) | the group's prep; `molbuilder jobs stop` of the job that holds it; `prep task --group … --one-job`; the stage launched alone | `submit._plan_sequence`, `submit._plan_sequence`, the header by the sequence's name; a group launched alone, `submit._planned` (`ready.held_by_sequence`) |
| **R-L46** | `--in-sequence` or `--in-job` typed by anyone but a sequence's walk — not both, not with one `--group`, `--mode direct --foreground --yes`, or naming a sequence whose launch record is not in `launch/`; inside the job, a group the job request no longer gives as the sequence was sent with it (renumbered since: groups are matched by their members) | the sequence's own launch; the groups left sent again | `_cli.submit_cmd`, `submit.Inside.of` (`sequence.read_record`), `submit._planned` |

A benchmark's prep refuses as a task's, and besides: a calculation that takes
no benchmark, a stage not named, and a benchmark prepared already — its way
back the state saved before its prep (`prep.prep_bench`).

#### Worked examples

*A transport calculation*, `lowbias`, its groups the defaults — 1 1 1 2 3 —
on a workstation:

| step | you type | seed · electrode_L · electrode_R (group 1) | device (2) | transmission (3) |
|---|---|---|---|---|
| 0 | `molbuilder describe`, or the Transport tab's Describe | described, preparable | described, preparable | described, preparable |
| 1 | `prep task --group 1 --group 2 --group 3` | prepared, ready to launch — takes nothing; the state saved first, `… · before prep task group 1 (seed, electrode_L, electrode_R), group 2 (device), group 3 (transmission)` | prepared, **waits for** seed's newest run, `01_seed/run-0`, not launched | prepared, **waits for** device's |
| 2 | `launch task` (it offers group 1 alone) | launched — one job, three runs one after another; the state saved first, `… · before launch task group 1 (seed, electrode_L, electrode_R)` | waits for seed | waits |
| 3 | — group 1 finishes | finished | **ready to launch** — takes `lowbias.DM` from `01_seed/run-0`, each lead's `.TSHS` from its run | waits for device |
| 4 | `launch task` (it offers group 2) | finished | launched — the state saved; the three files copied into `04_device/run-0`, with `.gathered-from` | waits for device's run, running |
| 5 | — the device finishes; you look at it, and change its `mixing_weight` on Task setup | finished | finished — not prepared again: to run it changed, restore the state saved before the first launch of group 2 (device), and prep anew | **waits for**: device's newest run did not run the deck its description gives now — the same restore named |

*An optimization ladder*, `coarse → medium → tight`, `tight`'s override
`restart: clean` — its groups 1 2 2, since `tight` builds on nothing and so
shares `medium`'s job:

| step | you type | coarse (1) | medium, tight (2) |
|---|---|---|---|
| 1 | `prep task --group 1 --group 2` | prepared, ready | medium **waits for** coarse; tight ready — the group waits for its member |
| 2 | `launch task --group 2` | prepared | refused before anything is sent (R-L11): medium continues from coarse, which has not run — launch group 1; to start medium otherwise (`--from`, `--cold`), give it a group of its own first |
| 3 | `launch task` | launched, then finished | ready to launch |
| 4 | `launch task` | finished | launched — one job: medium from `01_coarse/run-0`, tight from its deck |

*A refused send*: `coarse` ran; `launch task --stage coarse --mode submit`
opens `01_coarse/run-1`, fills it warm from `run-0`, and `sbatch` refuses the
request (R-L25). `run-1` stays opened, not launched; `status` names it so and
gives `launch task --stage coarse`; medium's first launch waits for it.
`launch task --stage coarse --cold --mode submit` empties `run-1`
and sends it cold; `run-0` stays what it was.

### What molbuilder does for you

1. **Before every prep that writes, and before a group's first launch, it
   saves the folder's state**, and tells you which — a prep with nothing to
   write changes nothing and saves nothing (*The task*, 2) — the moments and the note's form
   are [`checkpointing.md`](?doc=execution/checkpointing.md) § 9's
   (`2026-10-03 14:05:12 · before launch task group 2 (medium, tight)`). So whatever a prep or the runs after it do,
   the state before it is one `checkpoint restore` away — and the state before
   a group's first launch already holds every run that group builds on, but
   the points a transport calculation's rung of points takes over from a
   finished one it reuses, and the energies its transmission points share
   with it, which lie there until that launch copies them
   ([`engines/transport.md`](?doc=engines/transport.md) § 2a.16). A
   stage launched before 2.7.0, when a launch saved nothing, goes back to the
   state saved before its prep; every way back a verb prints says so.
2. **It keeps every run.** A new launch never writes into an earlier run's
   folder (hierarchical) or over its numbered files (flat): its output, logs,
   launch record and conclusion stay what they were. *On the flat layout the
   restart files are the exception, by what flat is:* they carry the label's
   name, `H2.DM`, `H2.XV`, so every run and every stage in that folder reads
   and writes the same ones — a later run replaces them.
3. **It launches any stage again, however its last run ended** — finished,
   failed, stopped, or still running — **warm by default, or cold when you say
   `--cold`**:
   * warm, it continues from **that stage's own latest run**: on the
     hierarchical layout the next `run-<n>` is opened with the restart files
     the latest one left; on the flat layout the run runs again where the
     folder's restart files lie, as the last run there left them — the run
     its record names (§ 5.4). A stage that takes nothing from a run (its override says
     `restart: clean`) or whose kind restarts from the beginning anyway (a
     PySCF vibration) starts over: nothing is handed on from a run of its
     own; a SIESTA vibration's run takes over each call its last run finished
     ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 6);
   * cold, it takes nothing from a run of its own: the next `run-<n>` is opened
     with none of them; on the flat layout its run script is started with
     `--cold --force`, which **removes** the restart files lying in the folder
     before the engine starts (a deck that reads them would otherwise read
     them);
   * either way a transport rung keeps the inputs its first launch gathered
     (the leads' Hamiltonians, the seed's density), copied with their record —
     unless you ask for them anew (`--gather`: the newest run of each rung
     upstream, under the same gates); warm, a file it carries from its own
     run stands — a device continues from its own density, not the seed's;
   * **a stage that sweeps the bias** reads "its own latest run" at the point:
     warm, each point **done** in the latest run is taken over into the new
     run and only the points not done are run; cold, every point is run
     ([`engines/transport.md`](?doc=engines/transport.md) § 2a.11).

   Before anything is sent, `launch` shows you the exact line, which run it
   follows, **how that run ended**, and what is copied — and asks once
   (§ 6.0). It does not refuse because of how a run ended, and it does not
   ask a second question about it — save that warm needs something to be warm
   from: a run that left none of the files the stage takes is refused warm,
   naming `--cold` (R-L17, R-L18).
4. **It tells you where each stage and run stands, from one check.** `jobset
   status`, the Results tab's ladder and every reader that needs a run's state
   ask the same door ([`architecture.md`](?doc=execution/architecture.md)
   § 3.2): not launched, queued, running, finished, or failed — with the reason
   the run gave, or that it stopped before its end (killed, out of time) — and,
   in a column of its own, whether it converged: `SCF yes` / `NO`, or for a
   relaxation `geometry yes` / `NO` — read from the same scan of its output,
   never folded into the state (a run can finish and not converge). A
   stage's own states — described, ready to launch, waits for, launched — are
   *The task*'s. It says what launching again would do; it does not tell you
   whether to.
5. **It writes every decision down** — what each prep and launch found, what
   each run continued from, what you were asked and what you answered — in the
   calculation's `jobset-decisions.log`.
6. **It continues a run that used up its budget only as its run card says.**
   A SIESTA run's `continue_retries` — its run card's, else the catalogue's
   default, 1 ([`job-request.md`](?doc=execution/job-request.md) § 4); set `0` and the run script has no retry — lets one job re-run its stage warm, as
   its next run, when the engine stopped because it used its budget without
   converging — the SCF's
   iteration limit under `SCF.MustConverge`, or a relaxation's step limit —
   never after a crash ([`running-a-job.md`](?doc=execution/running-a-job.md)
   § 3.5).

### What stays yours

* **Whether a run is good enough to use** — its convergence, its geometry, its
  physics. molbuilder reports; you judge.
* **Whether to launch again, warm or cold, and when.** Launching again while
  the last run is still running is not stopped: both run.
* **Which stages share a job, and when the next group goes.** You number the
  groups; `launch task` sends a group, by default, only once what it builds
  on has finished, and takes, by default, the newest run of what each stage builds
  on, saying which; you can name any run with `--from` (taken as said — launch
  states what it sees there, failed or not converged) or none with `--cold`
  (§ 5.4). Whether that run is worth building on is yours to judge before you
  launch.
* **Keeping a state you trust** — `molbuilder checkpoint save -m "…"` at any
  moment, and `checkpoint tag` to name it.

### How the checkpoint supports you

Each run keeps its own copy of the deck and scripts it ran with, so every
run's record stays true to what ran — on the hierarchical layout; on the flat
one a stage's runs share its folder's deck, and the deck an earlier run ran is
the one in the state saved before the prep that rewrote it. **A stage prepared and never launched is
changed by preparing it again** (its group whole). **How a launched stage runs
— its queue, wall, memory, ranks, its run settings — is changed the same way**:
change its job request and prepare its group again — on the hierarchical
layout; on the flat one a launched stage is not prepared again, and the way
back below is the road; each deck must stay the
same calculation as its runs ran, and the next launch sends the new request
([`job-request.md`](?doc=execution/job-request.md) § 5). To change **what a launched stage computes** — a
setting of the description, the machine, the run it builds on — **go back to
the state saved before its group's first launch and prep it anew**: that state holds every run
the group builds on, and the group as prepared, never launched; every run made
since that save is left behind — the group's own, what built on them, and any
other stage's launched since — and the restore names them first:

```
molbuilder checkpoint list                 # the states, newest first, each with its time
molbuilder checkpoint restore <id>         # the folder as it was before that group's first launch
molbuilder jobset prep task --group <N>    # prepared as you now want it
```

A group a transport calculation copied in from a finished one it reuses
was launched by no launch of this calculation: its way back is the state saved
before the prep that copied it in ([`engines/transport.md`](?doc=engines/transport.md) § 2a.16).

Going back is not losing: restoring a state and working from it is how you
**branch** — the state you left stays listed, and you can restore it again. Use
it the same way to try two settings side by side, to undo a stage you no longer
want, or to return to a result you tagged.

A stage you remove from the description leaves its files where they are,
untouched, and its number stays taken (`tight` stays `03_tight`); nothing lists,
preps or launches it. A launched one keeps its row in `job-set.json` — its
runs are read through it — so described again it is what it was, launched; and
the group it was launched in is no longer launched whole: its other members
are launched again alone (R-L7). To have the calculation as it was, restore a
state saved before you removed it
([`project-layout.md`](?doc=execution/project-layout.md) § 4.2).

### Use the jobset commands

These promises hold for what the `jobset` commands do. A deck or run script run
by hand, a file edited in a run's folder, a folder copied elsewhere — molbuilder
reads what it finds and says what it sees, but it did not write it and cannot
vouch for it. Use the checkpoint verbs rather than bare `git` in a calculation
folder ([`checkpointing.md`](?doc=execution/checkpointing.md) § 2.0).

---


## 1. What the job system is, and why it exists

> **Everything is a job set** *(decided 2026-08-11, user)*. One calculation is
> not a smaller, different thing than a ladder or a sweep — **it is a job set of
> one**, and it goes through these commands, not a shortcut beside them. There
> is no `molbuilder run`.

### The problem it solves

[`running-a-job.md`](?doc=execution/running-a-job.md) covers running **one**
calculation: you generate an input, the wrapper activates the right software
environment, runs the engine, and you watch it. That is the whole story for a
single task — and for a long time it was the only story molbuilder told.

Real research is not one task. A single result usually needs a **sequence** of
runs (relax a molecule loosely first, then tightly), and a project needs
**many** such sequences (the same analysis across dozens of structures). Getting
those onto a supercomputer adds its own chores: writing scheduler headers,
laying out each stage so it can start from the last one's results, carrying
restart files forward between stages, and — before any of that — figuring out
how many GPUs and CPU cores actually make a given calculation fastest on a
given machine.

Done by hand, that is a pile of error-prone shell scripting that every user
re-invents. The **job system** exists to make it a described, repeatable thing:
you say *what* you want as data, and molbuilder handles *how* to lay it out,
deploy it, and watch it — the same way whether it runs on your laptop or a
cluster. **What it does not handle is deciding that one stage should follow
another; that is yours** (§ 2, decision 6).

### The mental model, in one picture

The job system sits **above** the single-job wrapper. It never replaces the
wrapper — it produces many of them and orchestrates them.

```mermaid
flowchart TB
    subgraph single["running-a-job.md — one task"]
      W["a run directory<br/>+ its .run.sh / .sbatch wrapper"]
    end
    subgraph system["job-system.md — many tasks"]
      direction LR
      D["a description of work<br/>(a JobSet)"] --> O["orchestration:<br/>lay out, deploy,<br/>roll up status"]
    end
    O -.->|"produces + runs many of"| W
    classDef s fill:#eef;
    class system s;
```

Concretely, three kinds of "many jobs" all reduce to the same object:

- a **staged ladder** — one molecule, relaxed in increasingly tight stages, each
  stage a separate job that you start once you have looked at the one before it;
- a **parameter sweep** — the same calculation run at many resource settings to
  find the fastest (this is what benchmarking is);
- a **device workflow** — the transport composite (shipped 2026-08-29):
  one CITED junction relaxation, five derived stages — the electrodes are
  DERIVED from the junction's own labeled blocks, never separately relaxed
  ([`archive/2026-09-01-transport-design.md`](?doc=archive/2026-09-01-transport-design.md) ruling Q5).

The object they all reduce to is a **`JobSet`** (§ 3).

### A one-paragraph scenario (what a user actually does)

You have a molecule `bdt.xyz` and want a publication-quality relaxed geometry on
your cluster. You run **one** command to produce a *bundle* — a self-contained
directory holding a coarse stage and a tight stage plus a `job-set.json`
describing both. You `scp` the bundle to the cluster, then work **one group at a
time**: `prep task` (lay out both stages' folders and wrappers), `status coarse` (review
its deck and resources), `launch task --group 1` (hand that one job to the scheduler). When it
finishes you **look at it**, then `launch task --group 2 --from 01_coarse/run-0` —
which copies coarse's relaxed coordinates in and submits that. You check
`status` whenever you like. You never wrote a `#SBATCH` header.

### Where it stands today (read this before the details)

> **The job system is shipped on the command line; the web describes,
> prepares and observes.** Everything in this document — the `JobSet` model,
> the `molbuilder jobset` verbs, one-job-at-a-time SLURM submission with
> routing domains, and the whole benchmark workflow — works today from a
> terminal. The web's half is the description (the Task setup tab writes
> `task.json` + `job-request.json`, beside the hand-over's template —
> [`web/task-setup.md`](?doc=web/task-setup.md)),
> `prep` through the same entry the terminal calls (§ 5.3, *One prep, two
> doors*), and the Results tab; `launch` stays on the terminal **by design**:
> it spends a queue slot, one job per invocation (a benchmark's, one per
> resource shelf), by hand. The Results tab shows
> `status`'s ladder (§ 5.3); there is no web plan view (W14, § 8).

> ### How a ladder advances
>
> **You prepare the groups, launch one, look at what it produced, and launch
> the next.** Nothing starts a stage but a person — no field in a
> description makes one follow another (§ 2, decision 6); the one way to have
> a group follow another unattended is to name both at one launch, a sequence
> of groups, said and recorded as that decision (§ 6.0).
>
> What a stage continues from is a **real file, copied in at its first
> launch** — by default from the newest attempt of the stage before it, which
> must have finished, or from a run **you name** (§ 5.4). By then it has
> finished and you have read it, so there is nothing to resolve later and
> nothing pointing at a file that does not exist.
> That is a ladder of **independent** stages; in a **linked** one —
> transport's device and transmission — the first launch takes a stage's
> input from the stages before it, because the calculation fixes it; a
> SIESTA vibration's one stage builds on nothing, its phases joined inside
> its run ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11).
>
> | | |
> |---|---|
> | Who starts stage 2 | **you**, after looking — or at one launch with stage 1, a sequence of groups named in advance (§ 6.0) |
> | How stage 1's geometry arrives | a **file copy**, made at stage 2's first launch from stage 1's newest attempt, or the one you name |
> | If stage 1 converged to something wrong | you never launch stage 2 |

---

## 2. The design logic — why it is built this way

Six decisions shape everything below. Understanding them makes the rest obvious.

**1. Describe work as data; keep the engine out of it.** A `JobSet` is a plain
data object (a JSON file), not code. Producers turn engine configs *into* a
`JobSet`; the orchestration verbs (`prep`/`launch`/`status`) operate on
the `JobSet` **without knowing or caring which engine it targets**. This is why
one small set of verbs can drive SIESTA ladders and benchmark sweeps alike, and
why adding a new engine later means writing one new *producer*, not a new
orchestrator.

**2. Reuse the single-job wrapper unchanged.** The job system does not invent a
new way to run a job. Each job in a `JobSet` is launched by exactly the
`.run.sh` / `.sbatch` wrapper from
[`running-a-job.md`](?doc=execution/running-a-job.md), built by the same
function. So everything true of a single run — env routing, warm/cold restart,
GPU pinning, the monitor — is automatically true of every job in a batch. The
framework adds *orchestration around* the wrapper, never *inside* it.

**3. The machine's knowledge lives on the machine, not in the user's head
("target isolation").** The bundle you produce on your laptop is
target-agnostic. The cluster-specific facts — which activation command, how many
cores a node has, which partition — are resolved on the **target** at `prep`
time and baked into the scripts there. You do not need to know the cluster's
layout to submit to it.

**4. Fail early, never guess.** A malformed `JobSet` — duplicate job names, an
unknown `kind`, a warm-file condition naming a trait the job never declares, a
missing partition — is rejected at produce/validate time with a clear message,
not discovered halfway through a cluster run. The framework refuses to emit an
incomplete SLURM header rather than submitting something that will bounce.

**5. molbuilder informs; the user decides.** The framework never silently
auto-resumes a failed or interrupted run. `status` tells you which stage is
incomplete and what restart files exist; **you** choose to re-submit. This keeps
a surprising re-run from quietly overwriting hours of results.

**6. The order is a person's, not the data's.** Every job in a `JobSet` is
independent as far as the framework is concerned: **nothing in a `JobSet` says
"start this one after that one."** A *ladder* and a *sweep* differ in how their
directories are named and whether a person is meant to take them in order —
neither waits for the other.

**Why the ordering lives with the person.** Whether stage 2 should start is a
judgement about stage 1's result, and nothing in the data can make it. A stage
is a long job; one that continues by itself can spend a week refining a geometry
you would have rejected in a minute. So the order lives where the judgement
lives — every group may be prepared at once, and one `launch` sends one job.

**What this costs, stated plainly.** A branching graph (a "diamond", for a
two-electrode device) has no representation. If one is ever needed it comes back
as something a person asks for at launch, never as a field a description stores.

---

## 3. The data structure — a `JobSet`, piece by piece

A `JobSet` is the whole description of a batch, saved as `job-set.json`
(`molbuilder/jobset/model.py`, schema `molbuilder/job-set@1`). It is deliberately
small — five concepts:

```mermaid
classDiagram
    class JobSet {
      str name
      str engine
      str kind
      List[str] shared
      List[Job] jobs
    }
    class Job {
      str name
      str script
      Resources resources
      List[WarmFile] warm
      Dict[str, str] traits
      Dict[str, Any] point
      Optional[str] finish
      bool resumes
      Optional[str] scratch_dir
      Optional[Dict[str, Any]] placement
      Optional[List[str]] members
      Optional[List[str]] phases
      Dict[str, Any] asked
      Optional[str] program
    }
    class Resources {
      Optional[str] domain
      Optional[str] time
      Optional[bool] exclusive
      Optional[str] mem
      Optional[str] gres
      Optional[bool] gpu_binding
      Optional[bool] use_gpu
      Optional[int] mpi_np
      Optional[int] cpus_per_task
      Optional[int] continue_retries
      Optional[int] wall_slack
      Optional[bool] notify_on_scf
      Optional[float] notify_every_hours
      Optional[Tuple[str, ...]] notify_channels
      Optional[Tuple[str, ...]] notify_report
      Optional[int] max_memory_mb
    }
    JobSet "1" o-- "many" Job
    Job "1" o-- "1" Resources
```

*(The diagram is the dataclasses' fields — `jobset/model.py`; a job carries every key of `JOB_KEYS` every time, `job-contracts.md` § 6.1, and `Job.from_dict` refuses one that lacks any.)*

**There is no edge anywhere in this picture.** No job names another job. That is
decision #6, and it is the single most important thing about the shape.

Walk through it with the *why* for each piece:

- **`JobSet.kind`** is `"ladder"` or `"sweep"`. **Both are sets of independent
  jobs**; the kind says how their folders are named and whether they have an
  order a *person* should follow — a ladder's stages are meant to be run in
  sequence, a sweep's points are not. It does **not** mean one job waits for
  another: neither does. The kind drives the directory convention
  (`<seq>_<name>` versus `bench/bench-<point>` inside the stage it measures,
  `job-contracts.md` § 6.3) and nothing about scheduling.
- **`JobSet.shared`** lists files that every job needs but that never change
  between jobs — the pseudopotentials and the geometry. (A deck written from
  a sorted copy states its own order record, so none travels beside it:
  `model/overview.md` § 2.2a, I5.) (The monitor travels beside each wrapper
  instead, `run-reports.md` § 2.3, and so does a job's finish.) They
  are **copied** into each job's folder as real files (user, 2026-08-24): a run
  directory holds everything it needs, and a link holds nothing
  ([`project-layout.md § 1.0`](?doc=execution/project-layout.md)).
- **`Job.finish`** names the bundle the wrapper runs after the engine when the
  engine alone leaves no result — a SIESTA vibration's `mb_vibration.pyz`
  (`engines/vibration.md` § 5.11 § 3), beside each phase's deck — copied
  from the deck's spec; absent for every other job and for a
  benchmark trial.
- **`Job.resumes`** says whether a re-run of the job continues from what the
  last one left — the rung's own kind's warm-files fact
  ([`job-contracts.md`](?doc=execution/job-contracts.md) § 4.2a); a SIESTA
  vibration's run script is rendered per phase with that phase's own
  (`Rung.wrapper_at`) — and the wrapper reads it for every text that speaks
  of a retry.
- **`Job.scratch_dir`** names the folder of the run's own its engine's
  scratch lies in — a transport device rung's `tsgf`, the vocabulary's
  `scratch_dir` ([`job-contracts.md`](?doc=execution/job-contracts.md)
  § 4.2a) — opened with each run and emptied by its walk
  ([`engines/transport.md`](?doc=engines/transport.md) § 2a.11); `null`
  for every other job.
- **`Job.name`** does double duty: it keys the job's folder *and*
  its scheduler job name (`-J`), so a `squeue` listing reads the way the layout
  does. **`Job.script`** is the input file inside that folder.

  > **The naming splits**: a **stage** is
  > `<seq>_<stage>` (`01_coarse`) because a stage is ordered; a **trial** is
  > `bench-<point>` inside its stage's `bench/` container, named by its
  > settings because a sweep has no order. The full table, for every layer,
  > is [`job-contracts.md`](?doc=execution/job-contracts.md) § 6.3.
- **`Job.resources`** (`Resources`) are the per-job scheduler asks — all optional,
  `None` meaning *unstated*, which nothing fills in: a launch value stated
  nowhere is refused ([`architecture.md`](?doc=execution/architecture.md)
  § 5.2). They use the **scheduler's
  vocabulary** (`mpi_np` for MPI ranks → `-n`; `cpus_per_task` for cores/rank →
  `-c`; `time`, `mem`, `gres`, `exclusive`, `domain`), the same names the
  persisted files and SLURM flags use — the full mapping is pinned in
  [`job-contracts.md § 6.2`](?doc=execution/job-contracts.md). Per-job resources
  matter because a coarse first stage and a tight final stage want very
  different node sizes.
- **`Job.warm`** (a list of `WarmFile`) is what this job would take **from
  whatever run it is continued from** — not from a named job. `name` is a
  concrete filename; `requires_same` names a key both runs must agree on for the
  file to mean anything, looked up in `Job.traits`.

  > **It says WHAT, never FROM WHOM**, and that is the whole design. Which
  > run this job is continued from is decided at its first `launch` — the
  > stage before it, newest, by default, or a run a person names with
  > `--from` (§ 5.4) — so the producer, which runs long before anyone has
  > looked at anything, is not asked to know.
  >
  > `.CG` is why `requires_same` exists: a conjugate-gradient history is
  > meaningless to a Broyden stage, so carrying it blindly corrupts the restart.
  > SIESTA puts its optimizer in `traits`; the framework compares two strings
  > and knows nothing else about either engine.

- **`Job.traits`** are opaque per-job strings a `requires_same` is compared
  against. The framework never interprets them.

Every `JobSet` is checked by **`validate()`** before it is used: the `kind` must
be known, names unique, and every `WarmFile.requires_same` must name a trait the
job actually declares — a condition on a key that is not there is a condition
that can never be tested, so it is refused rather than silently ignored.

### 3.1 A real `job-set.json`, field by field

Descriptions of a format are easy to nod along to and hard to check. Here is an
actual two-stage ladder for benzene-dithiol on gold, with every field annotated:

```jsonc
{
  "schema": "molbuilder/job-set@1",   // versioned: a reader refuses a major it
                                      // does not know, rather than guessing
  "name":   "bdt_au",                 // the calculation's id — also the SIESTA
                                      // SystemLabel every deck shares
  "engine": "siesta",
  "kind":   "ladder",                 // how the folders are named + whether a
                                      // PERSON should run them in order.  Neither
                                      // kind makes one job wait for another.

  "shared": ["C.psml", "H.psml", "S.psml", "Au.psml"],
                                      // copied into every job's folder,
                                      // as real files

  "jobs": [
    {
      "name":   "coarse",             // → folder 01_coarse/ AND squeue -J bdt_au/coarse
      "script": "bdt_au_01_coarse.fdf",  // the deck, in the bundle root.  The
                                      // TOKEN in this filename is where the
                                      // directory's `01` comes from -- it is
                                      // read back off the deck, never counted
                                      // (decision 27).  A ladder job whose
                                      // script carries no token falls back to
                                      // `bench-<name>`, because inventing a
                                      // seq would be guessing at the one
                                      // number that must never be guessed.
      "resources": {                  // EVERY field is always written; the
                                      // count lives in job-contracts.md
                                      // § 6.2 and nowhere else
        "domain":        null,        // nulls included — see the note below
        "time":          "0-04:00:00",
        "exclusive":     null,
        "mem":           null,
        "gres":          null,
        "mpi_np":        8,
        "cpus_per_task": 4
      },
      "warm":   [],                   // takes nothing from an earlier run --
                                      // starts from the .fdf's own coordinates
      "traits": {"optimizer": "CG"}   // opaque; only compared against another
                                      // job's, never interpreted
    },
    {
      "name":   "tight",
      "script": "bdt_au_02_tight.fdf",
      "resources": {
        "domain":        null,
        "time":          "1-00:00:00",
        "exclusive":     null,
        "mem":           null,
        "gres":          "gpu:1",
        "mpi_np":        32,          // 4× the ranks: the tight stage is the
        "cpus_per_task": 4            // expensive one, and per-job resources
      },                              // are the whole reason they are per-job
      "warm": [                       // WHAT it would take from a run it is
                                      // continued from -- never WHICH run.
        { "name": "bdt_au.XV" },                            // relaxed geometry
        { "name": "bdt_au.DM" },                            // density matrix
        { "name": "bdt_au.CG",                              // optimizer history
          "requires_same": "optimizer" }   // ...only if both used the same one
      ],
      "traits": {"optimizer": "CG"}
    }
  ]
}
```

*(Abridged from a dump of that `JobSet`'s `to_dict()`: the key order and the
value forms are the real ones — a time is SLURM's `D-HH:MM:SS` — and a real dump
writes every `Resources` field, null or not; job-contracts § 6.2 counts them.)*

> **`null` is a value here, and it does not mean "zero" or "off".** It means
> **unstated**, and prep never leaves a launch value the job needs unstated: a
> missing rank count, cores per rank or GPU count — and, on a target with a
> scheduler, a missing queue, wall or memory — is refused at prep
> ([`architecture.md` § 5.2](?doc=execution/architecture.md)). What stays `null`
> is what this job does not use. A `mem` of `"0"` is SLURM's *"give me the
> whole node's memory"*. The fields are written out even when null so the file
> shows you the complete set of questions that will be answered, rather than
> hiding the ones nobody answered yet. This is the *assistant, not nanny* rule in
> file form: molbuilder does not quietly pick a node size for you.

**What that file becomes on disk.** `molbuilder jobset prep` reads it and lays
out the tree — drawn, for both shapes, in
[`project-layout.md § 1.1`](?doc=execution/project-layout.md), which owns it.
Every job's folder holds **real copies** of its inputs, never links.

> ⚠ **The tree's root is the *calculation* directory**
> (`projects/<project>/<topic>/<calculation>/`), not a folder of its own. The
> *bundle* and the calculation are the
> same directory, and **`calculation` is the name that wins** — `task.json` is
> the source and `job-set.json` is derived from it, so naming the folder after
> the derived file names it after something you can delete and regenerate. The
> reasoning is [`project-layout.md § 1.0`](?doc=execution/project-layout.md).


**Nothing in `02_tight/` exists until you prepare it.** `prep task` lays out
each picked group's folders, their wrappers and each stage's `run-0`,
holding the stage's own files; a stage's first `launch` copies in what it
continues from — by default the newest run of the stage before it, which must
have finished, or the run `--from` names (§ 5.4). So tight's `run-0/` may sit
prepared beside coarse's, and receives coarse's files only **after** you have
run coarse, looked at it, and launched tight.

> **A SWEEP's tree differs in two ways**: its folders are named by their
> **settings** rather than by a position (its points have no order, so no
> ordinal), and a point's attempt is a measurement, made once — a trial is
> launched once; measuring it again is the state saved before the
> benchmark's prep, restored, and a prep anew
> ([`project-layout.md § 1.5`](?doc=execution/project-layout.md)). Nothing is
> copied between points.
>
> ```text
> bench-G1K2C4/        ← G<gpus>K<ranks-per-gpu>C<cores-per-rank>,
>                        named by its SETTINGS, not a position;
>                        its deck, wrapper and pseudopotentials are its own
> ```
>
> **The prefix is `bench-`, and a described trial lives in
> its stage's `bench/` container** — `<NN>_<stage>/bench/bench-<point>/`
> ([`job-contracts.md § 6.3`](?doc=execution/job-contracts.md), the
> cross-layer authority). `summarize` maps a trial
> back to its point through the job-set's own data, never by parsing the
> directory name.

**Nothing dangles, because nothing points anywhere**: every file in a job's
folder is its own. **No job's directory reaches into another's.**

What a stage continues from is copied by its first `launch task --group 2` — by
default out of the newest attempt of the stage before it, which must have
finished and which you have already looked at, or out of the one `--from`
names (§ 5.4):

```mermaid
sequenceDiagram
    participant U as you
    participant P as jobset prep
    participant L as jobset launch
    participant S as the scheduler
    U->>P: prep task — lay out 01_coarse/run-0 and 02_tight/run-0
    U->>L: launch task --group 1
    L->>S: coarse's job
    Note over S: coarse runs, writes bdt_au.XV / .DM
    Note over U: YOU LOOK AT IT
    U->>L: launch task --group 2
    L->>L: save the state, then COPY .XV and .DM from 01_coarse/run-0 into 02_tight/run-0
    Note over L: a real file, from a finished run.<br/>Writing it cannot reach back into 01_coarse.
    L->>S: tight's job
```

**The copy is the thing a person can check** — it is a real file, present
before the stage starts, from the run `launch` said it took (§ 5.4).

A `job-set.json` is written by `JobSet.to_dict()` and nothing abbreviates it: every job carries every key of `JOB_KEYS` (`name`, `script`, `resources`, `warm`, `traits`, `point`, `finish`, `resumes`, `scratch_dir`, `placement`, `members`, `phases`, `program`) and `Job.from_dict` refuses one that lacks any, naming it (`WrittenBefore`) — the example above is abridged; every `prep` writes a whole one.

Read the example back in plain language: *a SIESTA ladder named `bdt_au`;
four pseudopotentials are shared by both stages; stage 1 asks 8 ranks of 4
cores for up to 4 hours; stage 2 asks 32 ranks of 4 cores and one GPU for up
to a day, and **if you continue it from something**, it will take that run's
`.XV` coordinates and `.DM` density matrix — plus its `.CG` optimizer history,
but only if that run also used CG.*

**Notice what the file does not say: when stage 2 runs, or after what.** It
cannot. Both jobs are described, neither is scheduled, and the order is
something you carry out one command at a time.

---

## 4. Where `JobSet`s come from — derived at `prep`, from the description

A `JobSet` is never written by hand — and it is never emitted *beside* the
description either. **`prep`, on the target machine, derives it from the
description and its job request** (the template, `task.json` and
`job-request.json`, floor 2) as part of its five steps
([`project-layout.md § 2.3.1`](?doc=execution/project-layout.md)): step 2
resolves the description against this machine into a `ParameterSet`
(`molbuilder/resolve.py`) — **always a list**, a production run being the list
with one element and a benchmark the same list with N
([`generator.md § 5`](?doc=execution/generator.md)) — and steps 3–5 render one
deck and wrapper per element and write the plan down. The root `job-set.json`
is the RUN plan, merged per stage and never overwritten; a sweep's own record
lives in the stage's `bench/` container (`job-contracts.md` § 6.1).

```mermaid
flowchart LR
    D["the description and its job request<br/>(template + task.json + job-request.json)"] -->|"prep step 2<br/>resolve.py"| PS["ParameterSet<br/>(a list — len 1 = a run)"]
    PS -->|"steps 3–5, one element"| L["JobSet (ladder)<br/>the root RUN plan"]
    PS -->|"prep bench: the grid<br/>as N elements"| S["JobSet (sweep)<br/>the stage's bench/"]
```

### 4.1 The SIESTA staged ladder

`molbuilder/siesta/stages.py` holds SIESTA's stage knowledge — the shipped
ladder (`default_siesta_stages`), what a stage's warm restart means, and the
traits a warm condition is compared against — consumed at `prep` through the
engine seam (`jobset/engines.py`): one job per stage of the description's
ladder, script `<label>_<NN>_<stage>.fdf`. An engine config carries no stage
list, so the ladder lives in `task.json`, never in the config
([`engines/stages.md`](?doc=engines/stages.md)
§ 1.1). Three things are *derived*, and each encodes a design decision:

- **A stage that runs out of steps without converging simply stops**, and you
  decide what to do about it — which is what you were doing between stages
  anyway.

  Both engines run N decks as N jobs
  ([`stages.md § 1.1a`](?doc=engines/stages.md)), so what follows a rung
  that did not converge is the gap between two jobs for both.
- **The warm-file declaration is chosen for correctness, not convenience.**
  Each stage declares what it would take from a run it is continued from — and
  **a stage whose description says `restart: clean` declares nothing at all**,
  because it is not continuing from anything. For a continuing stage, `.XV`
  (coordinates) and `.DM` (density matrix) are unconditional; `.CG`
  (conjugate-gradient optimizer state) carries `requires_same: "optimizer"` —
  a CG state is meaningless to a Broyden stage, so carrying it blindly corrupts
  the restart. The comparison is made at the launch that takes the files,
  between **this stage and the attempt it continues from** — by default, or
  named with `--from` — over each one's resolved config.

  > **The pair, never the ladder's order**: a run named with `--from` may be
  > of a stage two rungs back, or an earlier run of this same stage (`launch
  > task --group 2 --from 01_coarse/run-2`), so the stage before it in the
  > ladder is not always what a stage continues from.
- **Resources are per-stage** — each stage's run card (`execution`) states its
  own over the calculation's, and nothing is filled in
  ([`architecture.md` § 5.2](?doc=execution/architecture.md)) — so a coarse
  stage and a tight stage can be sized differently.

**The shipped default ladder** (the *structure*; the *values* and their
scientific rationale live in [`engines/tuning.md`](?doc=engines/tuning.md)):

| Stage | Enabled by default | Relaxation | Steps |
|---|:--:|---|--:|
| coarse | ✅ | CG | 600 |
| medium | ✅ | Broyden | 200 |
| tight | — | Broyden | 100 |

*(Whether to go on after a stage runs out of steps is a question you answer by
looking at it.)*

The three **strategy presets** flip only the enable flags:
`loose-only` = (✅, —, —), `publishable` = (✅, ✅, —),
`vib-quality` = (✅, ✅, ✅).

**The warm-retry budget travels the whole way.** `continue_retries` is a
run setting (`execution`) — its run card's, else the catalogue's default
([`job-request.md`](?doc=execution/job-request.md) § 4) — that `resolve.py` copies onto the allocation of
the resolved element: it rides `jobset.Resources` — the road every machine-side value
takes — and `jobset/prep` hands it to `write_run_wrapper`, which bakes it into
the wrapper's own retry loop
([`running-a-job.md`](?doc=execution/running-a-job.md) § 3.5).

**The retries happen inside the one job the scheduler ran.** `continue_retries`
is a loop in a single job's wrapper — it is not a between-jobs mechanism, and
it becomes **no `sbatch` flag**: `job-contracts.md § 6.2` marks it as the one
row of the translation table with no SLURM name.

> **`on_nonconvergence` is a PySCF field.** `engines/stages.md` § 3 keeps it out
> of the shared stage schema: it controls a loop inside one emitted script, and
> SIESTA has no such loop to control.

### 4.2 The benchmark sweep

`jobset prep bench <stage>` builds a **sweep** — one independent job per point
of a `(GPUs, ranks-per-GPU, cores-per-rank)` grid enumerated from *this*
machine's probed topology (`molbuilder/bench/grid.py::sweep_grid`), handed to
the same five steps as a longer `ParameterSet`, **nothing carried between
points** (they do not depend on each other). Because a sweep is just
another `JobSet`, the same `jobset` verbs run it — benchmarking is not a
separate machine, it is `prep` whose parameters are a set rather than a point
([`project-layout.md § 2.3.1a`](?doc=execution/project-layout.md)).

> **Every engine's ladder is the same object** — N decks, N jobs, a person
> looks between the rungs ([`stages.md § 1.1a`](?doc=engines/stages.md)).

---

## 5. The workflow — describe, prep, launch, summarize, status

**One verb on the host** (where you design the calculation) and **four on the
target** (where it runs). They mirror the design: the host step writes files and
nothing else, and scheduler contact happens only at `launch`.

| where | verb | what it does |
|---|---|---|
| **host** | `molbuilder describe` | write the portable description and its job request — § 5.1 |
| target | `prep` | the groups picked: resolve this machine, render each stage's deck and wrapper, open its first run with its own files |
| target | `launch` | send **one** job — a group, or one stage again — `--mode direct` or `--mode submit`; `--mode ask` asks the scheduler when it would start, sending nothing |
| target | `summarize` | summarize results that exist: a benchmark's trials into a verdict; a transport calculation's bias points into its I–V record; a SIESTA vibration's force-constant stages into its displacement sweep (`engines/vibration.md` § 5.9). Never a run's own result — every run writes that itself (§ 5.5 there) |
| target | `status` | every stage of the description, where it stands, and the next step, every group's line (*The task*); with a stage, that stage in full |

```mermaid
flowchart LR
    subgraph host["HOST — laptop or login node"]
      P["<b>describe</b><br/>→ the template · task.json<br/>· job-request.json · the data files"]
    end
    subgraph target["TARGET — the run loop (summarize joins it for a benchmark, § 5.3)"]
      direction LR
      PR["prep task<br/>lay out the groups you pick<br/>(prep bench: a bench's points) + wrappers"]
      SU["launch task<br/>the next group whose upstream finished<br/>--mode direct, or submit"]
      ST["status<br/>per-stage roll-up"]
      PR --> SU --> ST
    end
    P -->|"scp the bundle"| PR
```

### 5.0 The prep protocol — the agreement, and every checkpoint in order

> **This is the one statement of the protocol.** What follows in § 5 — and the
> documents linked from each row — says how each step works; this section says
> what it is, in order, and who decides what. Both doors run it — the command
> line's `jobset prep` and Task setup's **Prep** (the task's) / **Prep bench** —
> through one entry, `jobset/prep.py::prep_task` (§ 5.3, *One prep, two doors*).

#### Before prep: a described calculation

A calculation is described **once**, and prep only ever reads that description.

| | what happens | what is refused |
|---|---|---|
| **the hand-over** | a parameter tab sends its work to Task setup: the structure pair, the template and `task.1st.json` — a description still missing what only you can say ([`web/handover-procedure.md`](?doc=web/handover-procedure.md)) | — |
| **the description** | Task setup shows what came over and what is still needed; you complete it and save: `task.json` and `job-request.json` are written beside the template, and `task.1st.json` is removed ([`web/task-setup.md`](?doc=web/task-setup.md) § 2, § 8). From the command line, `molbuilder describe` writes the same files (§ 5.1) | a folder holding **another** calculation's `task.json` (Task setup § 2); `molbuilder describe` on a folder already described — change it in Task setup or in the files |
| **changing it later** | Task setup opens a described calculation **as it is** — the file is loaded, never merged — and saves what you change, the folder's state saved first, always and said (`checkpoint.save_before`, the function prep calls; [`checkpointing.md`](?doc=execution/checkpointing.md) § 9; Task setup § 8) | what the calculation **is** — its name, engine, structure and label — is read-only there (Task setup § 3); its shape is fixed once it has produced (§ 4 there) |

#### The agreement

1. **You state; molbuilder checks.** Every value a run is launched with — its
   ranks and threads, a GPU run's GPU count (asked of the scheduler; with none,
   the run spreads over the GPUs it is given, [`gpu.md`](?doc=execution/gpu.md)
   § 1.1), and where a `.sbatch` is written its queue, wall and memory — is
   stated by you, in the job request or on the command line
   ([`architecture.md`](?doc=execution/architecture.md) § 5.2 lists them). The
   target's record **checks** each one and supplies none.
2. **You choose the machine, once.** A calculation is set to the machine of its
   first prep, and that does not change; preparing it for another machine is a
   new prep from a saved state ([`configuration.md`](?doc=configuration.md)
   M-3). Prep never measures a machine: its record is there, or prep refuses
   and prints the probe that writes it.
3. **Everything is decided before anything is written**. Prep first makes its whole plan — every check, every value,
   the text of every file, every folder and what goes into it — with nothing
   written in the calculation (checkpoints 1–4b; a transport junction's
   record is round-tripped through its codec in a scratch folder outside it,
   [`script-preparation.md`](?doc=execution/script-preparation.md) § 3.0). A
   refusal can only come from there, and it
   writes nothing but its line in the ledger, so there is nothing to put
   back. Then the folder's state is saved (5); then the plan is written, which
   decides nothing and refuses nothing (6). Every refusal says why in your
   words and names what to do.
4. **A launched stage keeps what it computes; how it runs may change** *(the
   user, 2026-10-21)* — prep renders it again for a changed job request, each
   deck the same calculation as its runs ran ([`job-request.md`](?doc=execution/job-request.md) § 5); a change
   to what it computes is a rollback *(user, 2026-10-02: "refuse it, redo via
   rollback")*; a stage prepared and never launched is prepared again, its
   group whole — the rule and its way back are *How the checkpoint supports
   you*'s. So that the states are there, the
   folder's state is saved before every prep and before a group's first
   launch, and each tells you (checkpoint 5; § 6.0, step 4;
   [`checkpointing.md`](?doc=execution/checkpointing.md) § 9 holds the
   moments and the note).
5. **You advance it, a group at a time** — or several named at one launch.
   Prep prepares the groups you pick (*The task*, `prep task`, 2); `launch`
   starts one group's job once what it builds on has finished, or one job
   launching several groups in turn (§ 6.0, *A sequence of groups*), and
   shows it before sending it; nothing starts a stage but you (§ 5.3, *Three
   ideas*).
6. **Every decision is written down** — each check that refused, each
   question and its answer, what each stage continues from — in the
   calculation's `jobset-decisions.log`.

#### The checkpoints, in order

| # | checkpoint | passes when | otherwise |
|:--:|---|---|---|
| | ***the plan — nothing is written*** | | |
| 1 | **a described calculation** | the folder holds `task.json`, its job request and its template — the one template, named for the label | refused: `molbuilder describe` first — or, from one of its stage or attempt folders, the calculation's own folder named; a description without its job request, Task setup's Save (R-P1) |
| 2 | **the groups picked** | the groups shown (*The task*, at the top); the groups picked — `--group N`, repeatable, or the answer to the question — each one the job request holds. Each group is one job, its members sharing one allocation ([`project-layout.md`](?doc=execution/project-layout.md) § 1.6.6) — checked with its header at 4b; that no member builds on another is the description's own check, at 3 ([`engines/stages.md`](?doc=engines/stages.md) § 6.6). For `prep bench <stage>`, the stage named, a calculation the benchmark can measure, and no `--from` (a trial starts from its deck: the structure, § 5.4) | refused, naming the groups and the line that picks them |
| 2a | **not launched, or what it computes the same** | the calculation's plan holds no job for the group's stages, or holds them **never launched** — prepared again, the group whole (its rows replaced, its files rendered again, its first runs emptied), **unless every member is already as this prep plans it** — every file its run takes as written, stamps aside (by the catalogue: its deck, scripts and headers, the bundles beside them, an input it reads; never a record), each deck of its first run (each point's) the same calculation (`deck_record.same_calculation`), its row, its members and its group's header (`prep.prepared_differs`, `prep.as_written`): then **nothing to write** for the group, said with its next step, and the group is left out of the save, the writing and the record (5–7) — or holds them **launched**, on the hierarchical layout, and each member's deck, rendered now, is the same calculation as the deck its runs ran, its points the ones they ran ([`job-request.md`](?doc=execution/job-request.md) § 6) — prepared again for its job request, its rows replaced, its files rendered again, its runs untouched — and no sequence of groups still going holds the group (R-P23); with `--one-job`, a launched group named is read as prepared, not prepared again (R-P22); for `prep bench`, the stage's bench folder holds no sweep (one door, [`architecture.md`](?doc=execution/architecture.md) § 3.2) | refused: a launched stage whose deck would compute something else is not prepared again — the refusal names each difference and the way back (*How the checkpoint supports you*); a held group names its job's stop |
| 3 | **the description's own checks** — the preflight ([`engines/stages.md`](?doc=engines/stages.md) § 6.6), once per prep (`prep.checked`) | no error | refused, with the errors; warnings are shown and carried in the answer |
| 4 | **the machine, and the job's placement** — the record read once per prep (`prep.checked`), each member placed by the one member planner (`prep.plan_stage`) | the calculation's own copy of its machine's record answers — or, at its first prep, the record of the machine you named (`--target`; the machine you are on when none other is on file); it says how a shell enters an environment there; the stage resolves — its parameters and the job they make, once ([`script-preparation.md`](?doc=execution/script-preparation.md) § 3.0, steps 1–2: the record, the description and the template each read once, and every step after reads what was read); every launch value is stated; and a run is **placed** on the queue it names, its fit on the target's record printed — wall, memory, ranks, cores, GPUs — and never refused for it (`placement.placed`, § 6.0, *the placement*; `scheduler.md` R6). A benchmark states its queue, wall and memory before its grid is enumerated; the cells it proposes are those a node of that queue holds, and the cells declared are kept, each with its fit ([`generator.md`](?doc=execution/generator.md) § 4.3a) | refused, naming what is missing: which machine, when several are on file and none is named; the probe that writes a record; the machine the calculation is set to; the setting that does not resolve; the file and key where each value is stated; a queue the record does not list |
| 4a | **no value hand-over** (§ 5.4) | **every stage's hand-over is files, taken at its first launch** (§ 6.0): prep reads no other stage's run — but, for a transport calculation that reuses a finished one, that calculation's runs, to copy a group in ([`engines/transport.md`](?doc=engines/transport.md) § 2a.16) | the reuse's own refusals (§ 2a.16) |
| 4b | **the steps, planned** — the machine's copy, the structure, the decks, the wrappers, the directory ([`script-preparation.md`](?doc=execution/script-preparation.md) § 3) | the structure loads and matches what was described; the data files are there; every deck passes its two gates — **validate** the resolved values, and **check** the text exactly as it will be written, the reader's own section merged in; every wrapper renders; the plan's new row merges; the run and what it receives are known — every member of the groups picked planned into **one plan** by the one member planner (`prep.plan_stage`), in the ladder's order, each seeing what the members before it plan (their rows in `job-set.json`); each job that walks several members, its one request placed on its queue, its fit said, and its header in the plan too (`prep.walk_header_of`) — a group of several, its members sharing one allocation, the walls summed and the largest memory (`prep.group_header`); a run of points, one job walking every point, their walls summed — or the stage's one wall, where its points share it (a vibration's calls, `jobset.points`); a benchmark's shelves, each its trials' shared request (`shelves.shelves_of`); with `--one-job`, the sequence's one header — each group's own request together, one queue, the largest of each, the walls summed (`prep.sequence_header`, R-P22); a kind's own steps pass (a transport junction composed) | refused: the first that fails, in its own words |
| | ***the save*** | | |
| 5 | **the save** ([`checkpointing.md`](?doc=execution/checkpointing.md) § 9) | the folder's state is saved, always, once the whole plan stands: a new state when anything changed since the one it stands at — its first when it has none — its note led by the time it was taken (`2026-10-03 14:05:12 · before prep task group 1 (coarse), group 2 (medium)`); nothing new when nothing changed, and the state it stands at is named. Both doors say which. A prep whose every group picked has nothing to write (2a) saves nothing and names no state: it writes nothing to come back from | refused when the state cannot be saved: that state is the one a redo restores |
| | ***the writing — nothing is decided*** | | |
| 6 | **the plan written** | the one plan, as it was made — nothing is planned again: every file of the plan; each stage's first run opened ([`project-layout.md`](?doc=execution/project-layout.md) § 1.6.2) with the stage's own files — nothing from another stage's run, which its first launch copies in; a group a transport calculation copies in from a finished one opens no run, its members' runs copied whole, each launch record last ([`engines/transport.md`](?doc=engines/transport.md) § 2a.16); at the calculation's first prep, its copy of the machine's record, naming the machine; the pipeline log; `job-set.json` last — the moment the stage is prepared; each job that walks several members, its one header (a group of several's and a sequence of groups' in the calculation's `launch/` folder, a run of points' walk's in its stage's, a benchmark's shelves' in its own), rendered for its one request (placed at 4b); a header this prep does not write, one a prep before left, taken away | — nothing here refuses. An error writing (a full disk) leaves the stage not prepared, and the state saved at 5 is the way back |
| 7 | **the record** | the ledger, after the save — so the state saved holds no line of this prep: the question that picked the groups and its answer, the preflight's notes, the save, the groups prepared, *prepared* with which config files answered (as read at 4); with `--one-job`, *sequenced* with the groups and the header; for a transport calculation that reuses a finished one, *copied in* for each run copied ([`engines/transport.md`](?doc=engines/transport.md) § 2a.16) | — |

**What you get back** is the whole of what was found and decided — the table in
§ 5.3, *One prep, two doors*: the terminal prints it, Task setup shows it.

**A preview is the same entry, stopping before the save**: the
plan, shown — the answer a prep gives, said as what it would do: what it found
(the deck's own checks among it), for each stage what its first launch will
take and from which stage (*"at launch: takes <files> from <stage>'s newest
run, which must have finished"*) — which config
files answered, every file it would write, the launch values the job would be
launched with (A13, from the text it rendered: the header where a scheduler
runs it, else the run script) — or the refusal prep would give, with what it
found, and nothing saved, written or recorded. Task setup's **Preview** is this, and its
**Prep** runs the entry again with the preview's plan named — its identity,
the plan less the moment it was made (`jobset.planned.Plan.identity`: every
file it would write with its stamps masked — when and by which build it was
written, the rule two decks are compared by — and what each copy or move takes,
by its file's path, size and time when planned): when the plan it makes now
differs — the folder, the machine's record or a library file it takes, or
molbuilder itself, changed between — it refuses, saying to preview again; a
Prep naming no plan is not taken.

#### After prep

`launch` acts on a prepared group — a stage described and not prepared is
refused by name, with its prep and its launch, as `status` says of it; a group
whose upstream has not finished, by what it waits for. It is built as prep
is — a plan, shown, asked, then sent (§ 6.0): it shows the exact command of
everything it will send — for `submit`, the `sbatch` line, the queue and every
value of the header it sends (its `#SBATCH` lines), a wall it restates on the
line; for a run here, the line it runs — and asks before sending it
([`submission.md`](?doc=execution/submission.md) S4); `--yes` skips the
question, never the output. One job per invocation — a benchmark sent to a
queue, one per resource shelf (§ 5.3). Then you look —
`status`, the Results tab — and decide what to launch next.

### 5.1 Describe (host)

**`molbuilder describe` writes the portable package** — the template,
`task.json`, `job-request.json` and the data files — into the calculation
folder. Nothing in it names a machine, so it means the same thing wherever you
copy it ([`project-layout.md § 2.1`](?doc=execution/project-layout.md)). It is
no job verb: describing says what a calculation is, and `molbuilder jobset`
prepares, launches and observes it ([`job-request.md`](?doc=execution/job-request.md) § 9).

```bash
molbuilder describe --structure BDT-Au/structure/bdt.xyz \
    --bundle BDT-Au/optimization/bdt-relax --engine siesta \
    --calculation optimization --stage-strategy publishable \
    --shape hierarchical --psml-lib pseudopotential
```

Every path is an address from the projects root, so the line works from
anywhere (`job-contracts.md` § 2.5b) — the pseudopotential library too, which
lives in the tree; `describe` takes no positionals.
**The engine and the kind are stated, every time** (`--engine`, `--calculation`
— W57, 2026-10-06): a calculation described without one became a SIESTA
optimization, a folder named for a frequency calculation included.

Names and values are validated **here, on your laptop, not on the cluster**
(design decision #4): a stage name outside `[A-Za-z0-9_]+`, a duplicate stage, an
`overrides` key the schema does not know, or a value outside its bounds is
refused with the field named ([`stages.md § 6.6`](?doc=engines/stages.md)).

> **`molbuilder describe` writes the template, `task.json`,
> `job-request.json` and the data files, floor 2 only.** The rule and the
> reasoning are [`process/conventions.md § 3`](?doc=process/conventions.md).
>
> **The job request it writes holds each stage's group and nothing else** —
> the queue, the wall, the memory and each run's card are stated on Task
> setup's cards or in the file, and a launch value stated nowhere is refused
> at prep ([`architecture.md § 5.2`](?doc=execution/architecture.md)). A queue's
> name in it is the person's ask, which each target's record checks
> ([`job-request.md`](?doc=execution/job-request.md) § 3.1).

### 5.2 What `prep` lays out on disk

> **The tree below is a ladder's.** A **sweep** differs in two ways: its trials
> live in the measured stage's container as `<NN>_<stage>/bench/bench-<point>/`
> — named by their **settings**, because points have no order — and a trial
> keeps attempts as a stage does: `bench-<point>/run-<n>/` in the hierarchy, the
> filename index in flat ([`project-layout.md § 1.5a`](?doc=execution/project-layout.md)),
> each launched one carrying its `run.json`. Nothing is copied between points;
> they are independent.

`prep` turns the portable bundle into a tree you can run. Two ideas make it
safe and small:

- **Wrappers are written from the real input file, beside it.** Each job's
  `script` gets its `.run.sh` / `.sbatch` built in that job's own folder, by
  the *same* single-job wrapper builder — so a batch job's wrapper is
  byte-identical to a hand-run one (`prep.prep_jobset`).
- **Shared files are copied in, never linked** (user, 2026-08-24): a job's
  folder holds everything it runs from, so a copied tree still runs; the
  price is one copy of the pseudopotentials per folder.

**A job folder holds its own inputs, the shared package among them. Nothing
else.** In particular it holds no link into a sibling's folder.

Each stage folder holds its own deck, wrapper, pseudopotentials and the
bundles that travel beside the deck — `mb_monitor.pyz` always, a finish's,
and beside a PySCF deck the code it imports, `mb_pyscf.pyz` (one list,
`runwrap.bundles_for`) — each a real copy; a `run-<n>/` attempt inside it holds its
`run.json` once launched and copies of what it continues from — the tree is
[`project-layout.md § 1.1`](?doc=execution/project-layout.md)'s.


**Why a copy and not a link.** Stage 2 writes to `bdt.XV` — that very filename.
A link would carry the write back into stage 1's folder and destroy the result
you chose to build on. The copy is made at the stage's first launch, out of a
run that has already finished, so there is never a window where anything points
at a file that does not exist yet.

```mermaid
flowchart LR
    A["01_coarse/run-0/<br/>bdt.XV · bdt.DM<br/><i>finished, and you read it</i>"]
    P["launch task --group 2<br/>(continues from 01_coarse/run-0,<br/>the stage before it's newest)"]
    B["02_tight/run-0/<br/>bdt.XV · bdt.DM<br/><i>real files, copied</i>"]
    A --> P --> B
```

### 5.3 The execution loop — one grammar, one group at a time

> **This section is the authority for what you type.** `project-layout.md` § 1.6
> owns *what happens on disk*; this owns *the commands*.

#### The grammar

```
molbuilder jobset prep task    [--group N ...]  [options]   # the groups picked; --one-job: and their sequence's header
molbuilder jobset launch task  [--group N]      [options]   # one group; nothing named, the one that can go
molbuilder jobset launch task  --group N --group M ...      # several: ONE job launching each in turn (§ 6.0, A sequence)
molbuilder jobset launch task  --stage NAME     [options]   # one stage: launched again alone, or a group of one
molbuilder jobset <verb> bench <stage> [<trial>]  [options]
                    │      │       │        │
                    │      │       │        └─ launch bench only: WHICH trial
                    │      │       │           to launch, by its point's NAME
                    │      │       │           (`G1K4C6` — the directory adds
                    │      │       │           the `bench-` prefix,
                    │      │       │           job-contracts.md § 6.3).
                    │      │       │           Omitted, the sweep's
                    │      │       │           still-unlaunched trials ride
                    │      │       │           their shelves' grouped jobs
                    │      │       └─ which stage — by its NAME (`tight`) or
                    │      │          its NUMBER with a `#` (`#3` — the NN of
                    │      │          its `03_tight` directory).  Both reach
                    │      │          one resolver (a bare number and the
                    │      │          token are legal stage NAMES, so neither
                    │      │          can double as an ordinal spelling).  On
                    │      │          `task`, `--group N` names a GROUP -- the
                    │      │          answer to the question each verb asks:
                    │      │          repeated for several at prep; at
                    │      │          launch one, or several sent as ONE
                    │      │          job launching each in turn -- and
                    │      │          `--stage NAME` one stage: a member
                    │      │          launched again alone, or the first
                    │      │          launch of a group of one.  `--from` /
                    │      │          `--cold` speak for ONE stage (at a
                    │      │          first launch, of an independent
                    │      │          ladder): at launch,
                    │      │          `--stage NAME`, or a `--group N`
                    │      │          holding one stage
                    │      └────────── what is being prepared or launched:
                    │                  `task` (the calculation) or `bench`
                    │                  (the measurement of one stage)
                    └───────────────── prep · launch · summarize -- the
                                       verbs that take a kind
```

**A name is matched in any case** — `TIGHT` is `tight`, as every stage name
compares ([`engines/stages.md`](?doc=engines/stages.md) § 2). **Quote `#N` in
bash**: an unquoted `#` begins a comment there, so `--stage #3` reaches
molbuilder as `--stage` with no value and is refused — type `'#3'`.

**What molbuilder prints, you can type**: a command it prints
— a deck's header, a *next:* line, a remedy, a refusal's list — names the stage
by its NAME. Never the token: `03_tight` is itself a legal name, of another
stage; never `#N`, which a pasted line would lose to bash. A
deck's header holds only the token, so it prints the name through
`identity.command_stage`; every other line already holds a name. A benchmark
trial's deck names its own launch, `launch bench <stage> <trial>`.

**And every printed command is composed in one place**:
`jobset/commands.py` for what the verbs and their refusals print — the
calculation named by its address from the projects root (`--bundle`, quoted
as a shell needs it) every time, as `checkpoint`'s folder (`-p`) is: a line is
pasted where its reader is, and an omitted `--bundle` means the working folder
([`job-contracts.md`](?doc=execution/job-contracts.md) § 2.5b); a launch's mode stated
unless the calculation is launched on this machine and this machine's
`launch.mode` names one — a calculation set to another machine is launched
there, where this machine's file does not speak — as one line per mode its machine takes — the queue's
only where the machine names one — each line saying what it does, the machine
named as its record names it: `# runs directly on sol`, `# submits to sol's
queue` *(the user, 2026-10-25: "just claim what it is doing: submitting to
execute on sol")*; a refusal
that asks for a stage offering the stages the verb takes (`'#N'` quoted) and
the command for the first — a bench verb's, the stages with a prepared
benchmark, and a calculation that has no benchmark says so before it asks; one
command a line, any prose after `#`; a launch offered again in another mode —
an ask's answer, a refusal for want of a mode or of a header — as the whole
command, its flags as typed that the mode reads (`commands.launch_with`), never
*"the same command with `--mode X`"*, an edit to a line a bare launch, its mode
from config, never had; a name only the person knows (a machine's)
asked for in words, never a `<placeholder>`. A text read later — a deck's
header, a result's remedy — says a launch through `identity.launch_as_typed`,
the same for every engine: as typed from the calculation's folder — it is read
wherever it was copied, so it names no folder — with what the mode means
beside it; a deck's header names its group's first launch and the stage alone
(`identity.deck_launch`). A layer below `jobset` (a machine record's refusal, a
wrapper's banner) cannot import the composer and prints plain text by the same
rules; the way back it names is `identity.way_back_words`'s.

**`#N` is the stage's `seq`, never its row.** With stage 2 disabled the
ladder is `01_coarse` and `03_tight`, so `#3` means *tight* and there is no `#2`
to type — the same number you see in the directory, in the deck's filename, and
in the `seq` column of `status`. That is what
[`engines/stages.md`](?doc=engines/stages.md) R5 is protecting: a
position shifts when the ladder changes, and an assigned ordinal does not.

A **sweep** has no ordinals — its points are independent and have no order — so
its points resolve by name, and a refusal there does not offer you numbers it
does not have.

`status`, `scratch` and `migrate` take no *kind* — they are about
the calculation, not about one run of it; `probe` and `machines` are about a
machine (§ 7; [`preparing-for-another-machine.md`](?doc=execution/preparing-for-another-machine.md)).
**The kind is a positional, not a `--bench` flag**, because
`prep bench` and `prep task` are peers: measuring and running are the same act
over different parameters (`project-layout.md § 2.3.1a`).

| | `task` | `bench` | no kind |
|---|---|---|---|
| `prep` | the groups shown, and the ones picked prepared — every one not prepared as described pre-selected; `--group N`, repeated, answers without a terminal (*The task*) | `prep bench <stage>`: the target's record read, the grid enumerated, the trials rendered into the stage's `bench/` | — the kind is required |
| `launch` | one group, several as one job launching each in turn (§ 6.0, *A sequence of groups*), or one stage again (*The task*) | `launch bench <stage> [<trial>]`: under `submit` (asked, under `ask`), the trials not yet launched as one grouped job per resource shelf (`generator.md § 4.3a`); under `direct`, each here in turn, in one walk; a named trial alone | — |
| `summarize` | a transport calculation's bias points into `<label>.transport.json` ([`engines/transport.md`](?doc=engines/transport.md) § 2a.12), and a SIESTA vibration's force-constant stages compared into `<label>.fc-sweep.json` ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.9), `--tolerance-cm1` flagging a mode whose spread exceeds it; any other calculation is refused, naming those two — its outputs *are* the results, read by `status` and the Results tab | the sweep's trials read into `bench-result.json` | — |
| `status` | — | — | the whole calculation, or one stage (`status <stage>`) |
| `scratch` | — | — | an engine's scratch its runs hold — listed, each file a run still going keeps said; with `--empty`, the rest removed (below) |
| `migrate` | — | — | what an older molbuilder wrote in the calculation, rewritten — the road every reader names when it refuses such a file: each change printed with the rule it follows, a change that would alter the calculation refused, writing nothing, each old file kept beside its new one as `<name>.pre-<rule>` |
| `probe` | — | — | this machine's record, measured and, with `--write`, written — `--name` for a cluster you prep for; `--set` and `--scheduler` declare what the probe cannot see (§ 7) |
| `machines` | — | — | the machine records prep can target, each with where it lives |

Every verb that acts on a calculation takes it as `--bundle` — its address
from the projects root — or runs from inside its folder; `probe` and
`machines` take none. After the kind, a positional is always a stage — and,
for `launch bench`, a trial after it.

**`scratch` — an engine's scratch, listed and emptied** *(the user,
2026-10-20: "give me a tool before you ask me to do all these trouble to clean
up")*. An engine's scratch is what the engines' own file vocabularies name
so — the `scratch` each section of the vocabularies molbuilder ships states
(`molbuilder/<engine>/warm-files.toml`, never a calculation's own copy),
the one answer every reader asks (`warmfiles.is_scratch`,
[`job-contracts.md`](?doc=execution/job-contracts.md) § 4.2a): TranSIESTA's
leads' Green's functions, `*.TSGFL` and `*.TSGFR`, and an optical call's
breakdown of its spectrum by transition, `*.contrib.EPSIMG`; PySCF names
none. It is recomputed by every run, never carried, never taken over, never
saved ([`checkpointing.md`](?doc=execution/checkpointing.md) S1). A run of
points' walk empties its run's scratch folder (`Job.scratch_dir`) as each
round begins and as it ends (`engines/transport.md` § 2a.11); a run that
stopped first, or one prepared before 2.22.0, leaves it in its run. Scratch
a kind keeps beside a run's own files, with no folder of its own, stays
where it lies until `jobset scratch --empty`.
`jobset scratch` lists every such file below each stage the description
holds — on the flat layout, below the calculation's folder — by its names
and sizes, never reading a file, a file gone while it looks passed over
(`materialize.scratch_in`): under *the engine's scratch its runs hold (never
saved, checkpointing.md S1):*, a line per file — its stage, its size, its
path — then the count and the total; then each file a run still going
keeps, `  kept: <path> -- <why>`; then, when any file is free, the line that
empties them. With none, it says *no engine scratch in <folder>.*

**What a run still going keeps is asked of each run** — the run folder the
file lies in (`paths.run_folder_of`), never its stage's newest run — by one
answer that reads names, states and the jobs' files and removes nothing
(`runstatus.scratch_answer` → `ScratchAnswer`: `found`, `kept` with why,
`free`). A run keeps its scratch while a job launched from this machine that
lists it is going, or this machine cannot tell — *job <id> -- <its words>*,
the jobs door's answer (`joblist.is_going`, the one `molbuilder jobs` asks,
§ 6.0a); while the run door reads it queued or running
(`continuation.read_run`, the door `status` reads) — *queued*, *running*;
and while its state does not read — *its state does not read: …*. A file in
no run is free. On the flat layout the runs share the folder: while any
stage is going — a going job of the calculation, or `status` reading a stage
queued or running — every file is kept, and so it is when which stage is
going does not read.

**`--empty` removes the free files** after one question naming how many
and how much, *remove them? [y/N]*, Enter answering no; `--yes` answers it.
With nobody to ask nothing is removed, the ledger's answer reads *no answer
(not a terminal): nothing removed* (`ask.confirm`'s `unanswered`), and the
verb refuses. With none free, it says *nothing to remove.* A file gone since
it was listed is passed over, and *removed N file(s), X GiB.* — the size in the largest unit it reaches (`units.size_words`) counts what was
removed. The question and its answer, and what was removed with how much,
are written to the ledger (`scratch`: `question`, `emptied`). Nothing but
what the vocabularies name scratch is ever touched.

#### Three ideas, in plain language

**1. A group at a time, unless you name several.** A ladder is not a
pipeline. You prepare its groups, launch `coarse`, *look* at what it produced,
and only then launch `tight` — which takes `coarse`'s result at that launch.
`launch task` sends **one** job — one group, its stages building on none of
each other — and that is the default, because a stage is a long job, and a
run that continues on its own can spend a week refining a geometry you would
have rejected in a minute: the judgement belongs *between* two stages, where
the evidence is.

**Naming several groups makes that judgement in advance** *(the user,
2026-10-20: "how about we allow submission of multiple groups in sequence for
one job? … the lower level logic stays the same, only at higher level we need
to make sure the allocated hardware has the common factors/facility
requested")*: `launch task --group 1 --group 2 --group 3` sends one job that
launches each group in turn, inside its one allocation, each when the one
before has finished (§ 6.0, *A sequence of groups*) — a transport ladder
whose seed and leads you have no reason to look at before its device, sent
once. It is said and recorded as the decision it is: the question names every
group, and the ledger keeps it.

A **sweep** differs in one respect only: its points are independent, so which
one you mean must still be named, but the order you take them in carries no
meaning.

> ### A scheduler is handed ONE job at a time
>
> > *"SLURM should never submit jobs in parallel. Submission is manual and one
> > by one. It is a disaster to do parallel job submission on HPC."*
>
> Two reasons, and the second is scientific:
>
> - **On a shared cluster it is antisocial.** N jobs entering the queue
>   together start together if there is room, and the allocation goes with
>   them.
> - **For a benchmark it is not merely rude, it is invalid.** Points that run
>   concurrently contend for the same cores, memory bandwidth and interconnect,
>   so the sweep measures **contention rather than scaling** — and reports a
>   number that looks fine.
>
> So `--mode submit` hands the scheduler few, deliberate jobs — for a TASK,
> one job per invocation (a stage, a group of stages, or a sequence of groups
> launched one after another inside it); for a BENCH, one grouped
> job per resource shelf (`generator.md § 4.3a`) — never a queue flood. **`--mode direct` is untouched**: it
> runs each job here, one after another — in the background unless
> `--foreground` (§ 6.0) — which is not submission at all. The rule lives in the launch entry (`submit.plan_launch`), not in the
> CLI — several groups go as one job there, and a sweep sent to a scheduler
> planned by shelf — so any other caller meets it too.
>
> `launch bench` under `submit` groups the unlaunched trials by resource shelf
> and hands each group over as one job; a named trial goes alone.

**2. What a stage continues from is the stage before it, or what you say**
(the rule: *The task*, *What a stage takes, and when*). Its files are
**copied** into the new run, not linked — the engine writes to those very
filenames, and writing through a link would destroy the result you started
from. `--cold` means *start clean*, which with a folder per run is simply
**skip the copy**; there is nothing to move aside.

**3. `--mode` is the channel, and it is not the layout.** This is the one people
conflate, so it is worth saying flatly:

| | what it decides | where it comes from |
|---|---|---|
| **`--mode direct` / `submit`** | *how the job is launched* — a local `bash`, or the machine's submission system | the machine you are on |
| **`shape: flat` / `hierarchical`** | *how the results are kept on disk* | the **description** (`task.json`), and it is never inferred (`engines/stages.md § 6.7`) |

They are independent, and every combination is ordinary. **A workstation running
`hierarchical` is a normal thing to want** — you get a directory per stage and per
attempt, so an earlier stage's geometry is still openable after a later one has
run. Equally, an HPC job can be `flat`. Nothing in molbuilder infers one from the
other.

> `--mode` falls back to `launch.mode` in `molbuilder.json`: flag,
> then config — and the chain ends there. Unset in both is a **refusal**, not
> a derivation from the detected scheduler: deciding `launch` from detection
> would gate submission on where you happen to be standing, which
> `running-a-job.md` § 5.4 forbids (*the mode, not the detected scheduler,
> gates submission*). The key's contract is
> [`configuration.md` § 4](?doc=configuration.md) and `running-a-job.md` § 5.4.

#### The loop

```mermaid
flowchart TD
    D["<b>describe</b><br/>the portable package:<br/>template · task.json · job-request.json"]
    PB["<b>prep bench</b> &lt;stage&gt;<br/>build the measurement"]
    SB["<b>launch bench</b> &lt;stage&gt;<br/>measure the target"]
    SM["<b>summarize bench</b> &lt;stage&gt;<br/>a report: its execution block<br/>is yours to copy into job-request.json"]
    PR["<b>prep task</b><br/>the groups you pick<br/>render the decks · make each run-0<br/>with the stage's own files"]
    SR["<b>launch task</b><br/>the next group whose upstream finished<br/>save · COPY what each stage takes<br/>--mode direct, or submit"]
    L["<b>look</b><br/>status · the trajectory · the forces"]
    D --> PR
    D -. optional .-> PB --> SB --> SM -. "you write job-request.json" .-> PR
    PR --> SR --> L
    L -->|"good — next group"| SR
    L -->|"not good — restore the state saved before its group's first launch, then prep it anew"| PR
```

**`prep` prints what it resolved, and `launch` shows what it decides.** `prep`
shows the rendered decks, the jobs and what each stage's first launch will take;
`launch` then shows which run each stage takes and how it ended, what it
copies, and the exact `sbatch` line — the header prep wrote, its wall
restated only when `--time` says so — and asks before anything is sent
([`submission.md`](?doc=execution/submission.md) S4, every door).

#### One prep, two doors

`prep` has two doors — this command, and the Task setup tab's **Preview
task** / **Prep task here** (the task's, over its ladder) and **Preview
bench** / **Prep bench here** buttons
([`web/task-setup.md`](?doc=web/task-setup.md) § 11), a preview the plan
stopped before the save — and **one entry**,
`jobset/prep.py::prep_task`, which both call once the groups are picked —
the description read and checked once, every member of every group planned
by the one member planner (`prep.plan_stage`) into one plan, each group's own
checks and header its own (`prep.group_header`), with `--one-job` the
sequence's one header (`prep.sequence_header`) — and `prep.prep_bench` for a
benchmark; § 5.0 is the order of their checkpoints. It does the
whole act and returns what it found and decided **as data** (a `PrepAnswer`
per stage),
and it asks nothing: the asking is each door's. The five steps inside it still
say, as each deck renders, what that deck's checks found — on the terminal's
stderr and in the deck's `<deck>.validation.txt` — and the answer carries those
findings too, for the door that has no stderr.
[`architecture.md`](?doc=execution/architecture.md) A12 — one assembly per
route — is this, for the whole verb.

| in the answer | what it is |
|---|---|
| `findings` | the description's preflight notes (`engines/stages.md` § 6.6); an error refuses instead |
| `notes` | what the inputs said: a bench's grid — enumerated, left out, kept |
| `saved` | the folder's state, saved before the five steps wrote (§ 5.0, checkpoint 5): the state saved now, or the one it already stood at, and its note; none for a prep with nothing to write |
| `unchanged` | a stage of a group prepared, never launched, and already as this prep plans it: the sentence both doors say, nothing written for it (*The task*, 2) |
| `differs` | a stage prepared, never launched, and prepared again: each of its files, its row and its group's header that differ from what this prep writes now — or the one job of a sequence it is prepared again with — named from the calculation; empty for a stage prepared for the first time |
| `dirs` | the folders this prep's jobs run in — its stage's, or its trials' |
| `provenance` | which configuration file supplied each setting, as read with the machine's record at checkpoint 4 (`configuration.md` § 2.2) — a preview's too |
| `machine` | the machine it is prepared for: the one named, else the one the calculation's copy of its record names |
| `deck_findings` | what each deck's checks said, one of each (a sweep's trials repeat them) |
| `flat` | a flat run: its wrappers are rendered and there is no attempt to open |
| `attempt` | the first run it opened and what it brought in — the stage's own files |
| `takes` | what the stage's first launch will take, and from which stage — the line both doors print (§ 5.4); nothing is copied at prep but a group a transport calculation copies in from a finished one it reuses, said in `copied` |
| `copied` | a transport calculation that reuses a finished one: its manifest — each group copied in, its runs, where each comes from, how it ended and its size, or prepared here and the differences that decided it; for a rung of points prepared here, the points its first launch takes over and walks, and for a transmission point the energies it shares with the finished point and those it computes; the totals — shown before anything is written, with the groups the terminal offers and in Task setup's Preview, and kept in `STAGE-PLAN.md` and the ledger ([`engines/transport.md`](?doc=engines/transport.md) § 2a.16) |
| `points` | a transport bias scan's point folders instead — one per point |
| `need` | the stage's memory need at the ranks it is prepared with — at most so much, and the run that held it — or that nothing has measured it (`jobset.memory.need`, [`job-request.md`](?doc=execution/job-request.md) § 3.1a): printed after its placement and fit, or its `stated` line; a group's and a sequence's, their job's largest, beside their fit — a hint that refuses nothing; none for a benchmark |
| `placement` | a stage prepared for a queue: the queue it was placed on and where each value came from, as its job records it (§ 6.0); `fit`, the fit of what it sends — a run's own, a run of points' walk's; none for a benchmark or a member of a group of several, whose shelves and group say theirs — said and recorded nowhere; and the lines both doors print |
| `stated` | a stage prepared with no header (a machine with no scheduler, `--no-sbatch`): where each value it states came from, the line both doors print in place of a placement |
| `shelves` | a benchmark's: each shelf's name, its header's lines and them in one line, and its fit on the benchmark's queue |
| `resources` · `deck` | what the stage will launch with, and its deck |
| `pipeline_log` | the step-by-step record of the plan and the writing — always written, whichever door called ([`script-preparation.md`](?doc=execution/script-preparation.md) § 4.5) |
| `group` | a member of a group of several: the group's one job — its number, its members in order, its header in `launch/`, the lines it launches with and them in one line, its fit on its queue, the files the group adds |
| `sequence` | the last answer of a prep with `--one-job`, one with nothing to write too: the sequence of groups' one job — its groups' numbers, its header in `launch/` (none with no queue), the lines it launches with, its fit, and each group asking less than it in words (§ 6.0) |
| `preview` · `plan_id` · `writes` | a preview's: the plan, named, and every file it would write |

**The save is the entry's own, and it asks nothing** — the one function every
act that changes the folder calls first (`checkpoint.save_before`,
[`checkpointing.md`](?doc=execution/checkpointing.md) § 9). It runs once the
whole plan stands and before anything is written (§ 5.0); a save that fails
refuses the prep, since that state is the one a redo restores. The terminal
prints the state and its note, the tab shows them, and the ledger records it
(`saved`). Everything prep reads of the target is read in the plan, so every
refusal about that machine comes before the save. Every decision the entry makes lands in
`jobset-decisions.log` whichever door called it.

**A refusal carries what the entry had found** (`PrepError`'s `findings` and
`notes`): the preflight's notes and what the inputs said — a bench refused
because *no cell is left* points at its cells left out. A refused prep
wrote nothing (§ 5.0, rule 3), so there is nothing else to show. The command line prints them before the refusal's sentence;
the tab's route returns them beside it. The plain
`ValueError` / `KeyError` the steps raise for what is the person's to fix (a
template naming an item its schema does not declare) are refused the same way
on both doors; a `TypeError` is a bug, and looks like one.

**A bench with no declared axes is the machine's proposal on both doors**
([`generator.md`](?doc=execution/generator.md) § 4.3a: an absent declaration
keeps the machine's enumeration), and the tab offers its bench button whether
or not axes are declared. **Where `prep bench` refuses the
description itself** — a transport calculation, an engine the bench lane does
not speak — one function says so for every door (`prep_inputs.bench_refusal`):
the entry's gate, the bench's assembly before it reads any machine, and the
folder answer, whose `bench_refusal` hides the tab's Measure step.

#### Examples

A two-stage relaxation on a **workstation**, `shape: hierarchical`, its groups
1 and 2:

```bash
molbuilder jobset prep   task --group 1 --group 2             # both: 01_coarse/run-0, 02_tight/run-0; under tight:
#   at launch: takes <label>.XV, <label>.DM from `coarse`'s newest run, which must have finished
molbuilder jobset launch task --group 1 --mode direct         # coarse runs here, in the background
molbuilder jobset status                                      # follow it; look before deciding

molbuilder jobset launch task --group 2 --mode direct         # the state saved first, then:
#   takes      tight          [launched into run-0: it continues from 01_coarse/run-0 (the stage before it; concluded rc=0 at …; converged): copied <label>.XV, <label>.DM]
```

The same calculation on a **cluster** — same words, different channel:

```bash
molbuilder jobset launch task --group 2 --mode submit --dry-run
molbuilder jobset launch task --group 2 --mode submit
```

Changing a stage before it is launched — prepare it again (§ 5.0): its group
is rendered anew. Redoing a launched stage — start its group from the
beginning: save the folder as it stands, so the run you go back from is kept
in a saved state; go back to the state saved before its group's first launch,
which holds `coarse`'s runs; prep it anew (*Changed since prepared*;
[`checkpointing.md`](?doc=execution/checkpointing.md) § 7.1):

```bash
molbuilder checkpoint save -m "group 2 (tight) as launched, before going back"
molbuilder checkpoint restore 4f9ca71         # the one saved before group 2's first launch
molbuilder jobset prep task --group 2         # tight anew
molbuilder jobset launch task --group 2 --cold --mode direct   # runs directly on this machine, from the structure
```

**The whole ladder unattended is one job you name, group by group** — the
groups prepared together with their sequence's header, then sent at once
(§ 6.0, *A sequence of groups*; `project-layout.md § 1.6` for why one group
at a time stays the default):

```bash
molbuilder jobset prep   task --group 1 --group 2 --one-job                  # both, and their one job's header
molbuilder jobset launch task --group 1 --group 2 --mode submit             # one job: coarse, then tight
```

#### The read-only verb

```bash
molbuilder jobset status --bundle BDT-Au/optimization/bdt-relax   # every stage, where it stands, which is next
molbuilder jobset status                     # the same, from inside the folder

molbuilder jobset status tight               # ...and ONE stage in full
molbuilder jobset status '#3'                # the same stage, by its number
```

**The table is the description's ladder**: one row per stage
`task.json` names, in its order and with its number, from the moment `describe`
writes it — so it lists the stages before anything is prepared, each with its
group, the ones not prepared as described, the prepared ones not launched as the ready door
answers them — **ready to launch**, with what its launch would take, or
**waits for** the stage named (*The task*). A stage removed from the
description after its prep is not listed; its folder is kept, untouched, and a
launched one its row ([`project-layout.md`](?doc=execution/project-layout.md)
§ 4.2). The table
ends with **the next step** — every group, where it stands and the line that
moves it (*The task*, *The next step*). `status
<stage>` asks the same door for the stage it names. The Results tab's ladder is
this same answer ([`web/results.md`](?doc=web/results.md) § 2.4) — its wire form,
`JobSetStatus.to_dict`, the groups `prep task` would offer (`offer`) and the
next step (`steps`) included. A benchmark's sweep is read against the
calculation it measures (`jobset_status(calc, sweep=the sweep)`), from its
bench folder too, and its next step is its own verbs for that stage — launch
the sweep, then read what it measured (§ 7); a plan with no description
beside it — a hand-built one — is refused: no description, no layout to read
it by (`architecture.md` A16). Any other folder inside a calculation — a
stage's, an attempt's — is refused, naming the calculation's own folder
(R-P1, `calculation.refusal`).

The per-stage form answers a different question from the table. The table says
*where is this calculation up to*; `status <stage>` says *what this stage is* —
the deck it runs, what it takes from a run it continues from, the resources it
asks for — and *what happened to it*: which attempt it is on, whether it was
launched and how, what geometry it continued from, and what it left behind:

```text
STAGE 03_tight -- running

  deck            bdt_relax_03_tight.fdf
  declares        bdt_relax.XV, bdt_relax.DM, bdt_relax.CG
  resources       domain=public, n=32, c=2, t=1-00:00:00, mem=64G
  attempt         run-1   (of run-0, run-1)
  launched        submit as job 481923 at 2026-08-10T19:04:08Z
  command         /usr/bin/sbatch -J bdt_relax/tight --export ALL,MB_LAUNCHED_BY=jobset-launch,MB_WALL_S=86400 bdt_relax_03_tight.sbatch --run 1 -np 32 -omp 2
  continued from  03_tight/run-0
  converged       -
  warm files      bdt_relax.XV, bdt_relax.DM
  detail          running

Directory: 03_tight/run-1
```

None of that is inferred. It is answerable because a try is a directory and a
launch is a record ([`project-layout.md`](?doc=execution/project-layout.md)
§ 1.5, § 1.6). `continued from` is said every time: a run whose launch
record says `continued_from: null` reads *nothing -- it started from the
structure*, or, for a run that gathered its inputs (a transport rung after
the seed and the leads), *nothing -- it started from what it gathered*
(`checkpointing.md` S3); a run of points says which points it took
over from which runs, and the rest walked.

- **`status <stage>`** is where a stage's deck, what it carries and its
  resources are read before a launch — the "look before you leap" step. Prep
  also writes the whole table into the folder (`STAGE-PLAN.md`): each stage's
  row with the other members of its job, and what each stage it prepared takes
  at its first launch.
- **`launch`** sends **one** job — what is prepared and not launched: a
  group, as one job, asking which when several can go (`--group N` names it;
  several `--group`, one job launching each in turn, § 6.0; a stage launched
  again alone, `--stage`) —
  and takes a `--mode` (falling back to
  `launch.mode`; unset in both is a refusal, § 5.3):
  - **`submit`** hands that one job to SLURM — shown first, and asked (S4). One
    `sbatch`, one invocation, no dependency flag, nothing queued behind it.
  - **`direct`** runs that one job **locally** (`bash …run.sh`), in the
    background unless `--foreground` or the question's answer waits for it
    (§ 6.0). This is the workstation path.
  - **`ask`** sends nothing: the scheduler is asked when the line `submit`
    would send would start — its kind's ask, `--test-only` on SLURM — and
    its answer printed (§ 6.0, step 3).
  - `--dry-run` prints the exact command it *would* run — or the refusal the
    launch would meet — without launching and writing nothing: the safe way to
    see what will happen (§ 6.0).
- **`status`** reads the run directory (asking `run_status`,
  [`running-a-job.md § 4.2`](?doc=execution/running-a-job.md)) and reports each
  stage of the description — its state, its restart files — and the next
  step, every group's line (*The task*), then stops. It **never
  auto-resumes** (design decision #5): you launch the next group, or a stage
  again, yourself, and it continues from its own latest run.

### 5.4 How a ladder advances

**Two kinds of ladder, one way to run them**. Every ladder runs
the same way — `prep task` its groups, then `launch task`, a group at a time,
or several named at one launch as one job (§ 6.0, *A sequence of groups*) —
and nothing starts a stage but you. What differs is **what a stage
starts from**:

| | **independent stages** | **linked stages** |
|---|---|---|
| what they are | one calculation tuned several ways — an optimization's `coarse → medium → tight` | different jobs, each using another's output — transport's seed and leads → device → transmission (a SIESTA vibration's one stage is one run in phases, joined inside the run, [`engines/vibration.md`](?doc=engines/vibration.md) § 5.11) |
| the stages | named by you, as many as you like ([`engines/stages.md`](?doc=engines/stages.md) § 2) | the kind's own, each by its role ([`engines/template.md`](?doc=engines/template.md) § 6.4) |
| what a stage starts from | **the stage before it, by default**: medium's first launch takes the newest attempt of the stage before it and copies that run's geometry, its density and — for the same optimiser — its history ([`project-layout.md`](?doc=execution/project-layout.md) § 2.3.4). A stage whose `restart` is `clean`, and the first stage, start from the calculation's structure | the stages before it, by the kind: **transport, files at the first launch** — the device the seed's density and the leads' Hamiltonians, the transmission the device's, each from the newest attempt that ended so and ran the same engine input as the deck that stage renders now ([`engines/transport.md`](?doc=engines/transport.md) § 2a.11); **a SIESTA vibration's stage, nothing** — its run relaxes first and hands the geometry to its own force-constant call ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 3) |
| when the stage before has not finished | the stage is prepared all the same; its **launch** refuses before anything is sent, says why in the status door's words, and names what to do: launch the one before, let it finish, or run it again — or choose: `--from` an earlier run of it that finished, `--cold` the structure (on the flat layout, the stage's override `restart: clean`) — a launch says these for one stage, so a member of a group of several is told to give it a group of its own first. One that finished without converging is taken, with a warning; one that failed is refused | transport: the **launch** refuses before anything is sent and names the stage to run first; a SIESTA vibration's stage builds on no other |
| another source | yours to choose — any attempt by `--from`, none by `--cold`, at the launch | a SIESTA vibration's stage: none — it builds on the calculation's structure, so `--from` and `--cold` at its first launch are refused (R-L12); launched again, warm takes over every call its last run finished and `--cold` runs them all again ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 6). A transport rung: none — what it takes is its kind's, gathered from the rungs upstream at its first launch; launched again, `--gather` takes them anew, and `--cold` starts it with none of its own files, keeping what it gathered; `--from` is refused, and `--cold` at its first launch — but for a rung of points of a calculation that reuses a finished one, where `--cold` takes none of that calculation's points over (R-L13) |

**What crosses between stages, when a deck is a reordered copy**
*([`model/overview.md`](?doc=model/overview.md) § 2.2a, plan § 5ab R3)*. A call that takes another's **geometry** — a SIESTA
vibration's force-constant call taking its `relax` call's — is written by
the run's plan step from that call's output through the run door
(`runs.steps_of`), in the one deck order both calls share ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 3) — never row by
row from one deck's order into another's. A **restart file** — the stage
before's density, geometry or history, continued or gathered — crosses as it
always has: every deck of one calculation is written from its one structure,
pinned by its description's fingerprint ([`engines/stages.md`](?doc=engines/stages.md)
§ 6.3), so writer and reader list the atoms in one order (I6).

**What *finished* means, and how a choice is recorded** *(user: "yes, … but
user can force a structure still")*. The rule is *The task*'s — by default the
newest run of what a stage builds on, which must have finished. **Finished**
is the one status door's answer, as `status` gives it: the run ended on its
own with exit code 0 (molbuilder's conclusion marker, the wrapper's last act —
never the engine's own end mark) and nothing in its output says the engine
stopped ([`architecture.md`](?doc=execution/architecture.md) § 3.2, `usable`;
user, 2026-10-06). Otherwise the refusal says why, in that door's words; a
run that concluded with an error is never taken by default. **The person can
still force one**, two ways, both stated rather than
refused: a run named with `--from` (the tab's *Continue from*) is taken as
said — the verb that takes it (a first launch)
states what it sees there, *failed* or *not converged*, and refuses only what
cannot be done: no such run, nothing in it to carry; and a structure can be stated relaxed
(`already_relaxed`, [`engines/vibration.md`](?doc=engines/vibration.md)
§ 2.2), so the frequency stage takes it as given — the record it carries
(`info.relaxation`) is shown and checked against this calculation, never
required. **What a stage builds on is decided once — at its first launch —
and recorded where it lands.** A run it continues from — the stage before
it — is one `Continuation`: the run, and what it was, recorded in the ledger
(`continues`). A transport rung's inputs are its kind's gather — each file
and the run it came from, per run — recorded in the ledger (`gathers`). A
first launch that takes nothing is recorded so (`starts-cold`, and whether
`--cold` asked it).
Nothing picks the source again later unless you ask at a launch (`--from`,
`--gather`): a SIESTA vibration's result carries the relaxation its run
made (`relaxation.record`,
[`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 8), which its finish and
`summarize` read, and its first launch copies the files of that same run.
`STAGE-PLAN.md` says, under its table, what each stage's first launch will take
— the line both prep doors print — while the table lists what each stage
declares it may take; after it, a transport calculation that reuses a
finished one keeps its manifest ([`engines/transport.md`](?doc=engines/transport.md) § 2a.16, *The manifest*).

**What a run takes from other runs is one hand-over**, whoever decided it
(`materialize.HandOver`): the run that takes; each file, with the run it
comes from; and its record — `.continued-from`, the one run it continues
from, or `.gathered-from`, each input with its run. One function carries
every hand-over out, `materialize.hand_over` — it copies the files and
writes the record. A point a transport calculation takes over from a finished
one it reuses, and the energies a transmission point shares with a finished
point, copied into its `shared/` at every launch that walks it, are copied by
the reuse's own two (`reuse.take_into`, `reuse.shared_into`), each recorded in
`.copied-from`; the runs its prep copies
in are no hand-over — no run takes them, they are brought whole
([`engines/transport.md`](?doc=engines/transport.md) § 2a.16):

```mermaid
flowchart LR
    subgraph D["who decides what a run takes — the doors and the planners"]
        C["continuation — the run a stage continues from:<br/>a first launch, a launch again, --from"]
        G["gather — a transport rung's inputs:<br/>a first launch, --gather"]
        K["a launch again — the inputs its last run<br/>was given, carried on"]
        P["the points' walk — a point taken over,<br/>a point's start, the run continuing"]
    end
    H["HandOver<br/>the run that takes · each file and its run · its record"]
    X["materialize.hand_over"]
    C & G & K & P --> H --> X
    X --> CP["the files, copied —<br/>with the launch's plan, or by the walk<br/>just before the point runs"]
    X --> R[".continued-from · .gathered-from<br/>runrecord.write_continued_from · write_gathered_from"]
```

* **When the files are copied**: with the launch's plan, written at the send
  after the yes — except a point's start, which the walk copies just before
  the point runs, since the point before it may finish in the same walk; a
  point whose start cannot be copied is not run. On the flat layout the
  files lie in the folder and nothing is copied.
* **When the record is written**: with the files — and on the flat layout
  with the run's launch record, after the send went, since the flat layout
  opens no run before its launch.
* **How a run is named in a record** — one spelling, written and read one
  way (`runrecord.RunRef`): its folder from the calculation's —
  `01_coarse/run-0`, a point's `04_device/run-0/v0.2` — or, where a stage's
  runs share the calculation's folder, its own name, `H2_01_coarse-run1`,
  which every file of it carries.
* **Who reads the record**: `runrecord.read_continued_from` — `status`'s
  rows for a run's points (and through them the transport record), and a
  launch taking a point over, whose record names the run that computed it
  (one hop); `runrecord.read_gathered_from` — `status`, the transport record
  and the Run panel. A run's own `continued_from` is its `run.json`'s — the
  launch's hand-over, written with it — which the Run panel and a stage's
  `status` row read.

**A stage launched again** — the rule is *What molbuilder does for you*, 3;
here is how it is done. Whether warm continues is `Job.relaunch_continues`:
its kind resumes (the restart-file list's `resumes`,
[`job-contracts.md`](?doc=execution/job-contracts.md) § 4.2a — a
PySCF vibration's does not) and the stage takes something from a run
(`restart: clean` takes nothing). **A run of points** takes over, warm,
every point its latest run finished — a bias sweep's points, a SIESTA
vibration's calls ([`engines/vibration.md`](?doc=engines/vibration.md)
§ 5.11 § 6) — and `status` says that for it. On the hierarchical layout the next `run-<n>` is opened
through the one opener — warm with the restart files the newest launched run
left, cold with none; a transport rung's gathered inputs are copied from the
run each names, with their record (`.gathered-from`) — warm, a file the stage
carries from its own run stands (a device's `.DM`), and the gather gives the
rest. `--gather` and `--from` are decided at that launch and recorded like a
first launch's. On the flat layout the run runs again where
the folder's restart files lie — warm, from the last run there, which its
record names; cold, its run script told `--cold --force`, which removes the
restart files lying in the folder before the engine starts (*What molbuilder
does for you*, 3). A warm launch again
continues from one `Continuation` — the run, what it was — carried out as
every hand-over is (above), and recorded in the ledger (`continues`).
**A stage that sweeps the bias** follows this rule at the point: warm, the next
run takes over the points done in the latest one and runs the rest; cold, it
runs every point ([`engines/transport.md`](?doc=engines/transport.md) § 2a.11).
Its ledger says so on the launch's `launched` line — which points it takes
over, from which run, and which it runs, each from what start (`walks`) —
with no `continues` line; what it carries of its run's gather, `gathers`.

**What an independent stage continues from**, by *The task*'s rule: a
continuing stage — `restart` is `continue` unless its override says `clean` —
takes the newest run of the stage before it. Its first `launch` reads that
run before sending anything, and refuses one still running, stopped without
its conclusion marker, or failed, naming the commands; one that concluded
without converging is taken, with a warning.
**An explicit choice is taken as said**: `--from <attempt>` at the launch names
any run — `launch` says what it sees there (still running, stopped, failed, not
converged) and refuses only what cannot be done (no such attempt, no restart
files in it) — and `--cold` starts from the calculation's structure. What cannot be taken is
refused **before anything is written** too: a run that is not an attempt of
this calculation, `--from` with `--cold`; `--from` on the flat layout or on a
transport rung, and `--cold` there at a first launch — save a rung of points
of a calculation that reuses a finished one, where it takes none of that
calculation's points over (R-L13) — (launched again, `--cold` starts either
clean); `--from` and `--cold` on a benchmark, whose trial
starts cold from its deck — and the attempt is opened once, with what it
carries, so a refusal leaves an attempt an earlier prep set up as it was.
Either way `launch` prints which run the stage continues from — before the
question, and once sent — and the decision ledger records it:

```text
  medium: WOULD launch it into run-0: it continues from 01_coarse/run-0 (the stage before it; concluded rc=0 at Thu Sep 24 02:38:51 PM MST 2026; converged): would copy H2.XV, H2.DM
  takes      medium         [launched into run-0: it continues from 01_coarse/run-0 (the stage before it; concluded rc=0 at Thu Sep 24 02:38:51 PM MST 2026; converged): copied H2.XV, H2.DM]
```

(The optimiser's `H2.CG` is carried only between stages that use the same
algorithm — the shipped ladder's coarse is CG and its medium Broyden.) A
file it takes that was written as that run's job ended — a kill may have
caught it mid-write — carries `-- WARNING: written as the job ended (…)`
after its name in both lines and in the ledger's `continues` entry
(`written_as_ended`); the launch goes as asked
([`running-a-job.md`](?doc=execution/running-a-job.md) § 3.6).

`.continued-from` in the attempt and `continued_from` in its `run.json` keep it
beside the run, and the Results tab's Run panel shows it (*Continued from*). The
flat layout records it too: each run's own
`<basename>-run<N>.continued-from` and `<basename>-run<N>.run.json` name the run
it continues from by what every file of that run carries — `H2_01_coarse-run0`
— since flat keeps no directory per run.
**Both doors offer the same choice**: the terminal's flags, and Task setup's
**Continue from** in the group's tab, which puts it on the launch line and
shows the default — or why launch would refuse it — before anything is sent
([`web/task-setup.md`](?doc=web/task-setup.md) § 11). **In the `flat` layout** every stage shares one folder, so
nothing is copied: the restart files are the folder's, as the last run of
any stage left them (*What molbuilder does for you*, 2) — the same rule
holds all the same: its latest run must have finished, and its first
`launch` says which run that was, by the name every file of it carries
(`coarse's latest run, H2_01_coarse-run1, in this folder`). **Its record
names the run whose restart files it reads** — the last run in the folder,
of any stage, which the folder **records**: `<label>.last-run`, written by
each run's script the moment its engine starts, because from then on the
restart files and the results lying there are that run's
(`runrecord.last_run`; never worked out from when the launch records were
written — a warm retry inside a group's walk is stamped after the member that
runs next) *(user, 2026-10-12: "there should be a record manifest of the last
run so that is the key information what is the last run and the data file is
the result of that last run")*. The folder speaks for that run: the Results
page opens it, and the run door answers it (`runs.speaking`) — and when that
is not the run the rule took, `launch` says so before anything is sent (*the
restart files H2_02_medium-run0 left in this folder (concluded …) — the
last run here; coarse's latest run is H2_01_coarse-run1*) *(user,
2026-10-12)*. A stage launched again warm reads them the same way. There is no attempt there for `--from` to
name, nor an attempt-less start for `--cold` to make: a flat stage starts clean
by its override `restart: clean`.

An independent ladder, advancing:

```mermaid
sequenceDiagram
    participant U as you
    participant M as molbuilder
    participant S as the scheduler
    U->>M: jobset prep task --group 1 --group 2
    Note over M: renders both decks,<br/>02_tight/run-0 holding tight's own files
    U->>M: jobset launch task --group 1 --mode submit
    M->>S: sbatch … 01_coarse
    S-->>U: Submitted job 4021
    Note over U: coarse runs. YOU LOOK AT IT.<br/>Did it converge? Is the geometry sane?
    U->>M: jobset launch task --group 2 --mode submit
    Note over M: saves the state, takes coarse's newest attempt, finished,<br/>and copies its .XV / .DM into 02_tight/run-0
    M->>S: sbatch … 02_tight
    S-->>U: Submitted job 4022
```

**The gap in the middle of that diagram is the feature.** It is where the
judgement goes that no data structure can hold: *is this result worth building
on?*

### 5.5 Watching a stage while it runs

`status` is a roll-up you pull on demand. To watch a *single* stage live, point
the run viewer (the Results tab, or `molbuilder watch`) at the directory the
engine is running in — `<NN>_<stage>/run-<n>/` in the hierarchy, the calculation
directory itself when flat. It resolves and streams the trajectory exactly as for
a stand-alone job ([`running-a-job.md § 4`](?doc=execution/running-a-job.md)).
Every job also carries **`mb_monitor.pyz`** — the monitor and the framework
readers it reads the run through, one file, written beside the wrapper and
copied into every attempt ([`run-reports.md`](?doc=execution/run-reports.md)
§ 2.3). Its wrapper launches it in the background: it samples CPU, GPU and
memory into the run's `-runN.util.csv`, so how the stage uses what it holds is
visible without waiting for it to finish, and it tells your channels when the
run starts, as it goes, and when it ends (§ 2 there).

> **What you cannot do is checkpoint one stage on its own**, and that is a
> contract rather than a missing feature. **The history is rooted at the
> calculation** ([`project-layout.md § 6`](?doc=execution/project-layout.md),
> [`checkpointing.md`](?doc=execution/checkpointing.md) **L1**) — one repository,
> covering the root and every stage beneath it — because a history rooted inside
> `01_coarse/` cannot restore the pseudopotentials that live one level up, and
> *"go back to coarse and try a different tight"* needs a history containing
> both. Tagging a converged stage is `molbuilder checkpoint tag` at the
> calculation; the tag names a state of the whole folder, which is the only thing
> a restore can put back.

---

## 6. Launching — one entry, here or on a cluster

### 6.0 Plan, show, ask, send, record

`launch` is built as `prep` is (§ 5.0): **one entry decides everything before
anything is sent, and the send decides nothing.**

| # | step | what happens |
|:--:|---|---|
| 1 | **the plan** — nothing is written | **The machine**, first, for every mode: the calculation's own copy of its machine's record, read once for the whole launch (`submit._machine_of`, through `record.calculation_environment`; R-L38) — its scheduler kind spells every line, and decides nothing. **The work** (`submit._the_group`), in `submit.plan_launch(calc, *, mode, bench=(its set, its container), …)`: a prepared group, whole at its first launch, or one stage launched again; a benchmark's pending trials. **Its submissions**, from the submission builders (§ 9.1) — each one scheduler job, or one process here, walking one or more **members**: a run is one member; a group, its stages in the ladder's order ([`project-layout.md`](?doc=execution/project-layout.md) § 1.6.6); a run of points, its points; a benchmark's resource shelf, its pending trials — each sending the one header its prep wrote for it. **Each member**, from the one member planner (`submit._plan_member`): the run it runs in — prep's first run, or the next one — and what it takes (*The task*, *What a stage takes, and when*), with the gates: every upstream it takes from has finished; a trial starts cold. **What prep wrote is sent as it is** — nothing planned again, nothing compared *(the user, 2026-10-17, 2026-10-25)*: a stage whose records changed since its prep is said so by `status` and Task setup (*The task*, *Changed since prepared*). **The placement** — the job's own, recorded at prep: its queue and every value prep stated, none placed or chosen again; a submitted job's wall is the one value a launch may restate (`--time`); the wall and memory a line sends are read off the header its prep wrote (§ 6.0, *The placement*). The exact lines and scripts to send. Every refusal is a row of *The task*'s table — a reader's refusal (a record that does not read, a job no file can be named for, a calculation's own `warm-files.toml` that does not read) among them, in its reader's words |
| 2 | **show** | the exact command of every submission and every value of the header it sends, a wall `--time` restates marked as this launch's beside the header's (`Submission.request`; S4, [`submission.md`](?doc=execution/submission.md)); what each member takes, and from which run |
| 3 | **ask** | one question for the whole plan — a run here asked as a submission is *(user, 2026-10-05)*; `--yes` is the answer given in advance, and with nobody to ask (no terminal) nothing goes and the launch is refused, so a script is never told it went. `--dry-run` stops here and writes nothing, the ledger included — a refused dry run too, as a refused preview (§ 5.0). `--mode ask` puts the plan's lines to the scheduler instead (its kind's ask, [`scheduler.md`](?doc=execution/scheduler.md) § 6a) and writes down what it asked |
| 4 | **send** — nothing is decided | the folder is checked against the one the plan was made from — the plan made again from the folder as it is (the calculation opened again, a benchmark's set read again from its container; `plan_launch`'s `again`), and compared line by line: each submission's line and where it runs, each member and what it takes (launched again, how the run it continues ended), every file the send writes or copies and every file read where it lies, each by its size and write time. A member launched, a run ended or a file changed since is refused, saying which and to launch again to see the new plan. **A group's first launch then saves the folder's state** — when and with which note is [`checkpointing.md`](?doc=execution/checkpointing.md) § 9's; the launch's questions and their answers are written after it, so it holds no line of this launch. Then the plan's writes are carried out — each run opened or filled through the one opener, a run opened and not launched emptied whole first ([`project-layout.md`](?doc=execution/project-layout.md) § 1.6.2–1.6.3), with its `.continued-from` / `.gathered-from` (the one hand-over, § 5.4; a flat run's written with its `run.json`); a submission with several members gets its walk script; every script is on disk before anything goes — and each submission goes out, `sbatch` or the process here, through one function, each member's `run.json` written by the one writer |
| 5 | **the record** | the ledger: every refusal, the question and its answer, the save before a group's first launch (`saved`), what each member took (`continues`, `gathers`; `starts-cold` for a first launch that takes nothing; a run of points', its walk on its `launched` line, `walks`, and each point it takes over from the finished calculation its calculation reuses, `taken over`, [`engines/transport.md`](?doc=engines/transport.md) § 2a.16), each submission sent and its job id — a run here when it starts, with its pid when left in the background, and the job's address (`job`, § 6.0a) — with what the verb was told on each line (the kind, the stage, the mode and where it came from, the queue's source, which config files answered). The entry writes them (`submit.plan_launch`, `send_launch`, `ask_launch`, [`architecture.md`](?doc=execution/architecture.md) § 2.1); the verb writes the refusals it says before it calls the entry — a flag that means nothing for the mode, a stage not prepared. **The questions the verb asked before the plan** — which group to launch, which stage to launch again — each with its answer, go to the entry with the plan (`plan_launch`'s `held`): written before the plan's refusal when it refuses, at once under `--mode ask`, and otherwise with the send's own question, after the save (step 4); one whose answer the verb cannot act on — several picked, an answer that is no group's number — is written by the verb as it refuses (`_cli._launch_unit`); a dry run writes none |

**A run here starts in the background, unless said** *(user, 2026-10-09:
"add a run in background option for the launch when it is in a direct run …
let it run in the background instead of having to wait for it … check the log
and output later on"; 2026-10-16: "make it default but also explicitly
changable in the question")*. `launch … --mode direct` is planned, shown, asked,
sent and written down by the five steps above, and its process is **started
and left**: its own session, so closing the terminal does not end it, and
nothing attached to it; the send returns once the process exists, the ledger's
`launched` line carrying `background` and its pid. **The question offers the
other way**: *launch in the background? [Y/n/foreground(F)]* — Enter, `y` or
`yes` goes as shown, `n` or `no` sends nothing, `f` or `foreground` runs it in
the foreground and waits for it; under `--foreground` the plan is the
waited-on run and the question
*launch in the foreground? [Y/n/background(B)]*. `--yes` takes the plan as
shown, without the question. The ledger keeps the question as asked — `run
here, in the background` or `in the foreground` — and its answer: `yes`, `no`,
the other way's word, or `yes (--yes)` when `--yes` gave it. **What it says**: shown, *about to start here, in
the background:* or *about to run here, waited on:*; a dry run's line *WOULD
start in the background* or *WOULD run*; sent, the run's line *started in the
background*, its pid, and **how to stop it** — `molbuilder jobs stop <id>`
(§ 6.0a): the run is the leader of its own process group, and a signal to
that group reaches the engine; to the run script alone it would wait for the engine to end, since
bash runs a trap only after the command it waits on. **Nothing it prints is
lost**, because nothing reaches a terminal that is not also in a file: the run
script copies everything it prints — standard output and error — into its
`.runwrap-<time>.log` as its first act, the engine's own output is the run's
`.out`, and a walk sends its own standard error and each member's output to
its log under `launch/` (`runwrap.py`, `submit._walk_script`). The launch
prints, for a run here either way, the folder to read — or the walk's log —
and the way to follow it: the next step of every group (*The task*, *The
next step*), the run's among them; for a benchmark, the `summarize bench`
that reads its trials once they have run. `--foreground` on a queue is
refused, naming what it is — *waits for a run here*: under `submit` a
scheduler's job runs without anyone waiting on it, under `ask` nothing is
sent — and a queue's flag on a run here (`--time`) likewise, each with the
launch as typed for each mode this machine takes (R-L4; `commands.launch_with`, a run
here's line without a queue's flags, a queue's without `--foreground`).

**A run here, waited on, ends the launch with how it ended** *(user,
2026-10-13: "if you already know a job fails why … do you not report
failure?")*. When the process ends, the launch reads each member's state —
its stage's row of `jobset status`, the stage matched as every stage name
compares (`identity.stage_key`) — through the one status door
(`parse.dirs.run_status`; the standard is
[`running-a-job.md`](?doc=execution/running-a-job.md), *The order of
evidence*: **finished**, a run that ended on its own with exit code 0;
**failed**, one that ended with any other, or whose output states a stop) and
prints it in that door's words — the words `jobset status` prints — on its
submission's line, *FAILED* first when a member did not finish. A state that
cannot be read after the run ended is said so, *UNREAD* with why: the send
went, so it is never a refusal. The launch exits **0 when every member
finished, 1 when any failed or its ending could not be read**, so a script or
a person that trusts the exit status is told what happened. Left in the
background, the launch exits 0 having said *started*: the run has not ended,
and its state is `jobset status`'s.

**A run here is told its wall**, as a sent job is told its header's
([`running-a-job.md`](?doc=execution/running-a-job.md) § 3.6): the wall its
job states (`Job.resources.time` — `prep`'s `--time`, else its run card's,
else the job request's `allocation`), `MB_WALL_S` in seconds — a stage run
alone in its environment, each member of a walk on its own line — and one
stating none runs until it is done; a group's launch inside a sequence's
queue job tells its runs nothing of their own, each counting the job's wall
from the job's start (§ 6.0, *A sequence of groups*).

**A benchmark's walk.** A benchmark's trials sent together — a resource shelf
on a queue ([`generator.md`](?doc=execution/generator.md) § 4.3a), or its
unlaunched trials run here — go as one submission walking them in order, through
the benchmark's own script (`launch/<name>.run.sh`, written by
`submit._walk_script`): each trial `cd`ed into and its own run script run with its
own counts, timed, under the per-trial bound when one is given
(`--trial-timeout`, here as on a queue); a trial that fails leaves the rest to
run, and the walk exits nonzero when any failed. Stopped by the person —
Ctrl-C at a launch waited on, a lost terminal, `molbuilder jobs stop` of one
left in the background, a cancel — it starts no further trial (one under a
per-trial bound ends at it); what it did not reach is measured again from the
state saved before the benchmark's prep.
**A transport bias sweep is not a benchmark** *(user, 2026-10-05: "bias scan is
its own mechanism — this is parameter sweep, not some … computation resource
experiment")*: it is one calculation swept over a parameter — one run, its
points done or not done, read from each point's own files
([`engines/transport.md`](?doc=engines/transport.md) § 2a.11). What it shares
with a benchmark's walk and a group's is the **walk script** alone — one
generator for every job that runs several members in order — each with its
own settings: a device sweep hands each point the one before's `.TSDE` in its
frame, and a point that does not finish ends its frame's chain — the other
frames starting from the seed, as every frame does (`engines/transport.md`
§ 2a.9); a benchmark, a group and a transmission sweep walk on.

#### A sequence of groups — one job, each group launched in turn *(the user, 2026-10-20: "how about we allow submission of multiple groups in sequence for one job? this would simply require us to design the job script to understand to execute the script under each group in sequence, so the lower level logic stays the same"; "so we can do jobset launch --groups 1,2,3 to execute groups 1, 2, and 3 in sequence")*

**What it is.** `launch task --group 1 --group 2 --group 3` — two groups or
more — sends **one job** that launches each group named, in the
description's order, as `launch task --group N --mode direct` would launch it
on the job's node: inside the job's one allocation, one after another. Each
group is **planned when its turn comes**, so a group takes what it builds on
from the runs the groups before it have just finished, under every gate of a
first launch (*The task*, *What a stage takes, and when*: that run finished,
ran the same engine input, is the newest). **Nothing below the group
changes** but what the job is — a group's walk, a run of points' walk, each
member's run script, what each run takes, the ledger, the state saved before a
group's first launch are the ones a group sent alone has; what each run
records of the job it rode, and the wall it is told inside a queue job, are
the sequence's (below). A group whose launch is
refused, or whose run does not finish, ends the job there: the groups after it
build on it. Naming several groups is the decision to go on without looking
between them (§ 5.3, *Three ideas*), and the launch's one question names
every group.

**Its header is prep's** (`prep task --group 1 --group 2 --group 3
--one-job`; *The placement*, below). Prep prepares the groups named that are
not launched, as `prep task` does — never a group a sequence still going
holds (R-P23) — refusing `--one-job` when every group
named is launched already, since nothing is then prepared — and writes the sequence's one header,
`launch/<label>_sequence-<every member's token>.sbatch`
(`runfiles.GroupNames.of_sequence`) — its request the groups' own requests
together (`group.sequence_request`): each group's own is what its own header
asks — its members' shared request (`group.envelope`), a run of points' its
walk's, every point's wall — and the sequence's is **one queue**, every
group's; the **largest** ranks, cores per rank, GPUs and memory any group asks
for; the **walls summed**; the node whole when any group asks it; the GPUs
the groups that use them ask alike — one count, one binding, since a group
using GPUs runs its ranks over every GPU the job holds. Two queues, or GPU
groups asking two counts or two bindings, is no sequence: the prep refuses
`--one-job`, naming them — one job sits in one queue, with one set of GPUs. The header is placed and its fit said like
every header — never a veto — and the fit names each group that uses less than
the allocation: it runs with its own run card inside it (ranks, cores, its GPU
use — a transmission with GPU use off runs on the cores alone), the rest of
the allocation idle meanwhile. A group named that is launched already is read
as prepared — the sequence then launches it again (below) — and its request
is part of the sum. A machine with no scheduler (or `--no-sbatch`) has no
header to write: a sequence run here needs no `--one-job`.

**The launch** takes the groups in the ladder's order, whatever order
they are named in, and refuses before anything is sent: a group not prepared;
a group a sequence still going holds (§ 6.0a: *it goes in job N*), naming
`molbuilder jobs stop`; under `submit` or `ask`, no header prep wrote for
exactly these groups, naming the `prep … --one-job` line — on a machine with
no queue, the run here instead (`_no_sbatch`); and the flags that
speak for one stage (`--stage`, `--cold`, `--from`, `--gather`). A bare
`launch task` offers no group a sequence still going holds. The plan is
the first group's, made as its own launch makes it now — every gate, its runs
and what each takes — shown in full; each group after it is shown by its
stages, *launched in this job when group K has finished, taking what it
builds on then*. `--time`
restates the job's wall; `--dry-run` shows and writes nothing; `--mode ask`
asks the scheduler about the sequence's line. The send writes the sequence's
walk, `launch/<seq>.run.sh`, and its launch record, `launch/<seq>.launch.json`
(below) — both before the job goes, since the job reads them — then sends the
header (`submit`), or runs the walk here (`direct`, in the background unless
`--foreground`). The ledger's `launched` line names the groups (`sequence`);
every line of each group's launch inside it names the sequence
(`in_sequence`) and the job (`in_job`). A launch record that could not be
stamped as the job went is said on the launch's line and in the ledger
(`record_unwritten`) — the job went all the same, and its runs read what was
written before it went.

**The walk** is the one walk script (`submit._walk_script`), each step a
group's launch (a `WalkCommand`, shell words run as they stand):

```bash
( cd "<molbuilder's folder>" && "<python>" -m molbuilder jobset launch task --group 2 \
    --bundle "<the calculation's full path>" --mode direct --foreground --yes \
    --in-sequence <seq> --in-job "${MB_JOB_ID:-$(hostname)-$$}" )
```

— **the molbuilder that sent the sequence**: its interpreter, run from the
folder its package was imported from — the source checkout it runs from, or
the environment it is installed in — which the job's node must see (a
cluster's shared home; one it cannot run stops the walk at its first step,
said in its log). A step that exits nonzero stops the
walk; its log, `launch/<seq>.log`, holds every group's launch as it printed
it, and each group's own walk keeps its own log. **The wall each run is told**
(`running-a-job.md` § 3.6): sent to a queue, the job's — the sequence's wall,
every group's summed — counted from the job's start, as every member of a
walk sent to a queue is: a later group counts what the groups before it left,
and no run is told its own card's; run here, each run its own card's, as a
run here is.

**A group's launch inside it** (`--in-sequence <seq> --in-job <id>`, which the
walk alone types: refused with any mode but `direct --foreground`) is planned,
shown, sent and recorded as any launch, and asked nothing. **Each run it sends
records the job it rides**, read off the sequence's launch record: launched
in a queue job, `mode: submit`, that job's id, the line that sent it, where
its scheduler writes the job's error file (from the run's own folder), the
queue, wall and memory it was sent with; run here, `mode: direct` and the
walk's line. So **every reader of a run reads it as a run of that job** — the
run door, the scheduler's answer written once (`running-a-job.md` § 4.2), a
run past its wall — and a job cancelled or killed reads its runs failed, as
any job's do. It writes **no address of its own**: it adds its runs to the
sequence job's address (§ 6.0a), and leaves that address to the job; runs it
could not add are said on its line and in the ledger (`not_on_address`). It
reads how its runs ended as any launch waited on does (*A run here, waited
on*) — a run of points by its points alone, its walk having returned while
the job that holds it is still going.

**The sequence's launch record**, `launch/<seq>.launch.json`
(`molbuilder/sequence-launch@1`, through `persist`): `groups` — each group's
number and stages, in order; `mode`; `command`, the line that sends the job;
`placed_on` — the queue, wall and memory it is sent with, `null` run here;
`scheduler` — its kind and the folder its error file goes to, from the
calculation's folder, `null` run here; then, written by the one sender the
moment the job goes, `launched_at` and `job_id` — the scheduler's id, `null`
run here — both `null` in the plan, which decides what the job is, not when.
What each run inside it records of the job is read from here, never composed
a second time.

**Status** reads a group the job holds and has not launched yet as **goes in
job N** — after the group before it in the job — its line `molbuilder jobs
status`; a group it has launched is read from its runs, as any (*The next
step*). One answer says it, for status, prep, launch and the Results tab
(`ready.held_by_sequence`): a group is the job's while the job is GOING —
its process, or its scheduler's list (§ 6.0a) — and it has launched it by
neither its address nor the group's own runs, whose launch records name the
sequence that sent them (in that very queue job, or run here after it went);
a job this machine cannot ask about holds nothing, `molbuilder jobs status`
saying so. **A launch inside the job reads a run of points its job sent by
its points** (`joblist.inside`, the job and the sequence's name, set around
the whole launch): an earlier group's, or its own group's after its walk
returned, is never read as *running (between two points)* or *queued* by
asking the job — its own, still going — so the gather's gates, how its runs
ended and the next step it prints say what the points say, *— its walk ended
before any point started* where none did. Any other run is read by the run
door, as anywhere. A sequence that ended before a group leaves that group as it was:
ready to launch, waiting, or launched. **To go on after a stop**, send the groups left:
`launch task --group K … --group last`, its header written by `prep task
--group K … --one-job`, which reads a launched group as prepared; each
launched group goes again as its own launch again would — from its own newest
run, a transport rung with what its last run gathered (`--gather`, which takes
anew, is one stage's, so not in a sequence) — while a group's FIRST launch
takes what it builds on from the runs the groups before it have just
finished.

**The placement is decided once, at prep, and recorded**
([`architecture.md`](?doc=execution/architecture.md) § 5.2).
Prep decides each job's queue — `prep --domain`, else the run card's `domain`,
else `allocation.domain` (a benchmark's: `prep --domain`, else
`allocation.domain`) — and its wall, memory, ranks, cores and GPUs, each value
with its source, and **places the whole request of every job it writes a
header for** on the queue it names, printing its fit on the target's record: a
stage's; a group's, its members' walls summed and their largest memory; a run
of points', one job walking every point, their walls summed; a benchmark's
shelf's, its trials' shared request, the benchmark's wall the shelf's
(`shelves.shelf_request`). A GPU run naming a queue with no GPUs, a wall
longer than the queue allows, more memory than its nodes hold is prepared all
the same, the fit naming each limit it is over with its numbers — the
scheduler decides, and its answer is shown (`scheduler.md` R6; checkpoint 4,
`placement.placed`). The answer and its sources are written on the job in
`job-set.json` — its `placement`: the queue it was placed on, bound on the
record (name, partition, qos), and where each value came from, the flag, the
run card or the allocation
([`job-contracts.md`](?doc=execution/job-contracts.md) § 6.1). The fit is
said and recorded nowhere — it decides nothing: both doors say the placement,
then the fit of the job each header sends — a run's own, a run of points'
walk, each shelf's, a group's after its last member — with every value of
that header (its `#SBATCH` lines). **Every header is prep's**: rendered at prep for the queue it placed,
binding no name a second time — a stage's beside its deck; a
group's, a run of points' walk and a benchmark's shelves in their `launch/`
folder (`prep.walk_header_of`). One request is asked of a queue, by every
caller (`placement.request_of`: a SIESTA run's ranks, a PySCF run's one
process, the cores each runs on, its GPUs, memory and wall). **Launch decides
none of it** *(the user, 2026-10-17: "whether it's direct launch or
submission, all the decision is already done at the prep … Launch means just
run that script")*: it sends the header prep wrote — the question and a dry
run show its every value — places nothing again and chooses no queue, and
plans nothing again: a stage changed since its prep is said so by `status`
(*The task*, *Changed since prepared*); every launch checks that a trial starts
cold (R-L29). The calculation's copy of the target's record is the one its
first prep took, and no
probe replaces it ([`configuration.md`](?doc=configuration.md) M-3): a
calculation meant for a machine whose record has changed since goes back to
the state saved before its first prep. **A submitted job's wall is the one
value a launch may restate** — `--time`, for a run launched again that ran out
of it — passed on the line in a job's own spelling (`D-HH:MM:SS`,
[`scheduler.md`](?doc=execution/scheduler.md) § 6a) and checked by nothing of
ours: the scheduler answers. `run.json`'s `placed_on` keeps the queue, wall and memory
the run was sent with — read off the header prep wrote, by its
scheduler kind's own spelling (`kinds.Kind.directive_value`), its wall
the launch's `--time` when restated: the one source the line's
`MB_WALL_S` takes too; a header that states none, with no `--time`, is
refused (R-L44) — and the ledger's lines the flag as typed. **A benchmark
measures on one queue**: the one its prep names (`prep --domain`, else
`allocation.domain`), every trial fit on it alone and its placement
recorded like a run's — measuring another queue is another benchmark. Where
several members go as one job — a group, a run of points, a benchmark's shelf
— launch writes the walk of the members still to run and sends the job's one
header as its prep wrote it: a run of points launched again with points done
walks the rest under the header written for every point.

### 6.0a The jobs launched from this machine — `molbuilder jobs` *(user, 2026-10-16: "can we keep the status of running jobs the same way as we keep status of running server? … a similar way to check jobs submitted/directly launched etc? and we can use similar verb to stop a task if we need to")*

**Every launch leaves an address, the way a server leaves its pidfile**
(`molbuilder serve status`, [`ops/deployment.md`](?doc=ops/deployment.md)).
The one sender (`submit._go`) writes it the moment a submission goes — a run
here once its process exists, a queue job once the scheduler gives its id (a
job whose id `sbatch` did not print gets none) — one file per submission in
molbuilder's state folder, `<state dir>/jobs/`
([`configuration.md`](?doc=configuration.md), the tree), kept there past a
logout as the run is: a run here's named `<host>-<pid>.json`, a queue job's
`<cluster>-<id>.json` — a home folder two clusters share holds both clusters'
jobs of one number *(the user, 2026-10-22)* — and every file read by what it
says, never by its name (`joblist.find`, `address_of`); the ledger's `launched` line names it (`job`). A file
that could not be written is said on the launch's line — beside a queue job's
id, or beside a run left in the background, which is then named by its pid,
`kill -- -<pid>`. **It holds an address, never a state:**

| key | what it is |
|---|---|
| `schema` | `molbuilder/job@1` |
| `id` | what `jobs stop` names: `<host>-<pid>` for a run here; for a queue job its job id, as `squeue` lists it — a number two clusters' jobs share answered for by the cluster this machine's scheduler serves, and refused naming both where it serves neither |
| `how` | `here` — waited on or left in the background (`background`) — or `queue` |
| `host` | the node it was launched on: a pid means something only there, and a home folder a cluster's login nodes share holds every node's files |
| `pid` · `started` | a run here: its process, and when the kernel started it with the boot it belongs to (Linux's `/proc`) — together what tells the run from a later process given the same pid. A warm retry re-execs in the same process, so the file stays true |
| `job_id` · `cluster` | a queue job: the scheduler's id, and the cluster it was sent to as the scheduler names it (`ClusterName`) — a home folder two clusters share holds both clusters' jobs |
| `calculation` | the calculation's folder — printed, as every line names it, by its address from the projects root |
| `members` | each member's stage → its run, as a record names a run (`runrecord.RunRef`: its folder, or where its stage's runs share one, its own name), and that run's launch record, which its state is asked through — a sequence's, added by each group's launch inside it as it sends its runs |
| `sequence` | a sequence of groups (§ 6.0): each group it launches, in order, with its stages, and the sequence's name; `null` for every other job |
| `launched_at` | when it went |

**What it is doing is read fresh, each time** — one fact, one source, one
answer (`joblist.is_going`), which `molbuilder jobs` asks of each job,
whether a sequence still going holds a group (`ready.held_by_sequence`:
`status`, prep, launch and the Results tab),
`jobset scratch` of each job of its calculation (§ 5.3), and a run of
points' state of the job that holds it (`continuation.points_state`, whether
it is between two points or ended before its first): a run
here is going while its process is (the same host, pid and start — a machine
with no `/proc` says it cannot tell); a queue job while the scheduler lists
it in a state not ended, said with that state and the scheduler's reason for
it when it gives one (SLURM's `None` is none) — *the scheduler lists it:
PENDING (QOSMaxWallDurationPerJobLimit)*, its
own words, so a job it took and will never start says why
([`scheduler.md`](?doc=execution/scheduler.md) R6); one it lists ended is
said so with its state — *the scheduler lists it ended: FAILED
(NonZeroExitCode)* — until the scheduler forgets it, when the run's own
state says how it ended, the scheduler's word among it (`scheduler.md` R16;
`running-a-job.md` § 4.2: the run door, on this machine, writes a job the
scheduler no longer lists into the run's launch record, so the run's state
says so wherever the folder goes) (its kind's query, run where the calculation's copy of its machine's
record says the command is, asked on its own cluster — another cluster's job
is said *sent to cluster C: ask there*, one whose cluster was not read when it
went is not asked; the run door asks by the same rule and hears the same
answer, asked at most once per ten seconds per job (`joblist.asked_here`,
`asked_lately`); a record naming no query command,
or a command that does not run here, says it cannot ask; a copy not here says
its scheduler cannot be asked, and a copy that does not read is said in its
own refusal's words (`record.calculation_environment`) — an older
molbuilder's naming `jobset migrate`; each keeps the file); each member's own state is its run's, read through its
launch record by the run door — that run, never its stage's newest; where a
stage's runs share a folder, the run its warm retries reached, each retry a
run of its own (`runrecord.retried_to`, [`running-a-job.md`](?doc=execution/running-a-job.md)
§ 3.5) — and for a run of points, a container no one run speaks for, its
stage's row of `jobset status`, the stage matched as every stage name
compares (`runstatus.job_members`, `identity.stage_key`); a run no longer on
disk — a restore went back past it — is said so. Where the two
disagree — the process gone while its run still reads *queued*, stopped
before its monitor began — both are said, and neither overrules the other;
a queue job the scheduler no longer lists is written into its run's launch
record by this very look (`running-a-job.md` § 4.2, *Asked once, written
once*), so the run reads `failed` from then on and the two agree.

**`molbuilder jobs status`** lists every file: its id, how it went, when, the
calculation, and each member with its state in the status door's words —
here, a transmission left in the background, then once it was stopped:

```text
qlabsrv-312877  run here on qlabsrv, in the background, pid 312877 -- its process is going
  launched 2026-10-16T22:06:45+00:00; its calculation:
  molbuilder jobset status --bundle P/transport/S
    transmission  05_transmission/run-0  running
```

```text
qlabsrv-312877  run here on qlabsrv, in the background, pid 312877 -- its process has ended
  launched 2026-10-16T22:06:45+00:00; its calculation:
  molbuilder jobset status --bundle P/transport/S
    transmission  05_transmission/run-0  failed -- stopped before its end: …
  ended -- said once; its address is removed
```

**A sequence lists its groups**: under its line, the job's name says so (*a
sequence of groups 1, 2, 3*), then each member of the groups it has launched
with its state, then each group it has not reached yet, *not launched yet*. **A job that has ended is shown once — how each member ended — and its file
removed**; its run's folder and the calculation's ledger keep its history. A
run waited on is shown by its own launch, which says how it ended — or,
stopped at its terminal, its stop — and removes its file then. A run here
launched on another host is said so — *on HOST: ask there* — and never
removed or signalled here; a file that does not read is said, with its
path, and kept.

**`molbuilder jobs stop <id>`** — the line a launch prints beside a run it
left in the background — shows what it will stop — the calculation,
its members, the pid or the job id — and asks *stop it? [y/N]*; `--yes`
answers in advance, and with nobody to ask nothing is stopped
([`submission.md`](?doc=execution/submission.md) S4); the question and its
answer are written in the calculation's ledger, a *no* too, as a launch's are.
A run here is checked to be the process launched (host, pid, start), and
checked again once the answer is given: a run left in the
background leads its own process group, and SIGTERM goes to the group, so it
reaches the engine — to the run script alone it would wait for the engine to
end; a run waited on is stopped at its terminal, with Ctrl-C, and `jobs
stop` says so. A queue job: its kind's cancel, from where the record says
the command is. The calculation's ledger records the stop (`stopped`, and
how), and the verb says it: *stopped: SIGTERM to its process group, <pid>
-- `molbuilder jobs status` says how it ended* — or, in its place, the line
it ran, `/usr/bin/scancel 62039301` on SLURM. The run then
reads *failed — stopped before its
end* by its own files ([`running-a-job.md`](?doc=execution/running-a-job.md)
§ 4.2); one stopped before its monitor began reads *queued*, and `jobs
status` says its process is gone. A trial under a benchmark's per-trial bound
ends at its bound.

The address is floor 1's (`molbuilder/joblist.py`, beside `runrecord`: its
writer, its reader, whether a job is going, `is_going`, and the signal to a
run here, checked first); the
cluster, whether the scheduler lists a job and its cancel are the
scheduler's floor 1 too (`scheduler/queue.py`); the verbs are a surface,
floor 7
([`architecture.md`](?doc=execution/architecture.md) § 2).

### 6.1 On a cluster — SLURM

The wrapper file shapes (`.run.sh` inner + `.sbatch` outer) and the meaning of
each `#SBATCH` line are owned by
[`running-a-job.md § 5.3`](?doc=execution/running-a-job.md) and
[`job-contracts.md § 2.6`](?doc=execution/job-contracts.md). What the **job
system** adds is submission and routing:

- **The two layers.** The outer `.sbatch` is a thin header, in the language of
  the target's scheduler kind ([`scheduler.md`](?doc=execution/scheduler.md)
  § 6a), whose body says once what the scheduler tells the job, in
  molbuilder's names (`MB_JOB_ID`, `MB_NTASKS`, …), then a single line —
  `bash <base>.run.sh "$@"`. The inner `.run.sh` owns activation and launch,
  and reads only molbuilder's names. You submit the outer file; it hands off to
  the inner one. This split means the scheduler header and the run logic
  evolve independently, and the exact same `.run.sh` works with or without a
  scheduler.
- **The `.sbatch` is not always written.** The `.run.sh` is always there; the
  header beside it is withheld only where **the target's record names a
  scheduler with no queue** (`workstation`) — that machine has no queue, so a header for it would be a
  file nobody can submit — or where you asked for none (`prep --no-sbatch`). A
  target with a scheduler gets one, and every value in it is stated: a run
  that names no queue, wall or memory is refused at prep, never given a header
  that picks one ([`architecture.md` § 5.2](?doc=execution/architecture.md)).

  So *"I prepared and got no `.sbatch`"* has two answers: the machine's record
  names a scheduler with no queue (`workstation`), or you said `--no-sbatch`.
- **One `sbatch` per job, and the header carries the job.** The
  line names the job (`-J`), restates its wall when the launch says so (`-t`
  from `--time`), and carries the launch claim, the wall it is sent with —
  told to the job in seconds, `MB_WALL_S` (`running-a-job.md` § 3.6) — and
  the run's arguments: `sbatch -J <calc>/<job> [-t <wall>] --export
  ALL,MB_LAUNCHED_BY=jobset-launch,MB_WALL_S=<s> <header> <args>`, from
  `scheduler/kinds.toml`; every other value — queue, ranks, cores, GPUs,
  memory, binding — is the `#SBATCH` header prep wrote, sent as it stands.
- **One request, one sender.** Every submission — a stage, a benchmark's
  shelf, a bias scan, a transport group — builds its line the same way
  (`submit._sbatch_request`), in the words of the scheduler kind its machine's
  record names, from where the record says its submit command is
  ([`scheduler.md`](?doc=execution/scheduler.md) § 6a); `--mode ask` asks the
  same line with the kind's ask.
- **Every run is handed its own ranks and threads.** Every launch passes each
  run script its `-np` and `-omp` — a run here; a stage sent to the queue,
  after the `.sbatch` name, which forwards them (`bash <base>.run.sh "$@"`);
  each trial of a benchmark's shelf and each point of a bias scan, inside
  their sequencer's script — so a launched run's counts are its own, never
  the submitting shell's `OMP_NUM_THREADS` or what the scheduler tells the
  job (`MB_NTASKS`, `MB_CPUS_PER_TASK`, said by its header)
  ([`running-a-job.md` § 3.2](?doc=execution/running-a-job.md): the flag comes
  first). *(User, 2026-10-05: "Make all calling consistent with —omp
  and settle it".)*
- **Routing domains.** Instead of hard-coding a partition, you name a **domain**
  — in the job request (`allocation.domain`, or the run card's `domain`) or
  on `prep --domain`. A domain is a friendly name for a
  `(partition, qos)` pair of the target's record; an unknown name is refused
  at prep with the record's list — and a named one on a machine whose record
  lists no queues at all. What the record says of a named queue is shown
  as the job's fit, never a refusal: the scheduler decides
  ([`scheduler.md`](?doc=execution/scheduler.md) R6).
- **Render every job, then submit them.** A grouped submission (the bench
  sweep, § 7) writes **all** its shelf scripts before it sends the first one,
  and **one scheduler refusal does not cancel the rest**: the shelves are
  independent jobs the queue may run concurrently, so a refusal is recorded
  on that shelf's result and the loop goes on. Its trials keep no launch
  record, so the launched door (`runrecord.launch_record`) leaves them pending and the next `launch` picks
  up exactly them. The command reports what went out and what did not.

  > **Both halves, because one refusal unwinds an interleaved loop**: a Sol
  > bench's refused 4-GPU group left the valid 2-GPU group behind it
  > *neither written nor sent*, and the printed recovery command named a
  > script that had never been written.
  > `--dry-run` still writes nothing at all: it is the launch's plan, shown
  > and dropped (§ 6.0), and the question shows the same plan before the
  > person has said yes.
- **Job names read well.** A job's `-J` is `<calculation>/<job>` —
  `bdt_au/coarse`, `bdt_au/G1K2C4`, `bdt_au/bench-group-cpu-G0K4C1` — on every door,
  so a `squeue` listing tells you which of your calculations each row belongs
  to, not just which stage.

A workstation — a machine whose record says so — simply gets `.run.sh` files
and is run with `--mode direct` (or `launch.mode: direct` set once — the
mode is always stated, never derived).

---

## 7. Benchmarking — measuring the fastest resources

Before you commit a long production run to a node, you want to know: for *this*
calculation on *this* machine, how many GPUs, how many MPI ranks per GPU, and how
many CPU cores per rank actually run it fastest? Guessing wastes allocation. The
benchmark workflow measures it, and it is just the job system pointed at a
resource grid.

Its guiding idea is **target isolation** (design decision #3): everything
machine-specific is discovered on the target. Its machinery is the ordinary
jobset loop — benchmarking is `prep` whose parameters are a set
(`project-layout.md` § 2.3.1a):

```mermaid
flowchart LR
    D["molbuilder describe<br/>(host)<br/>the portable calculation"]
    P["jobset prep bench &lt;stage&gt;<br/>(target)<br/>read the record → environment.json<br/>+ the grid as trial decks in<br/>&lt;NN&gt;_&lt;stage&gt;/bench/"]
    R["jobset launch bench &lt;stage&gt;<br/>one grouped job per resource shelf<br/>(a named trial submits alone)"]
    S["jobset summarize bench &lt;stage&gt;<br/>trials → bench/bench-result.json<br/>(winner + mechanism), and the report:<br/>the execution block to copy"]
    PR["jobset prep task --group &lt;N&gt;<br/>uses <code>execution</code> — what you wrote is the answer"]
    D --> P --> R --> S --> PR
```

- **Detect the machine → `environment.json`** (`molbuilder/environment@3`): the
  scheduler (its kind, and where its commands are there), the site, the
  topology, and the domains you can actually reach —
  resolved to the **compute node's** real core and GPU counts (read from the
  scheduler via `scontrol show node`, not from whatever login node you happen to
  be on), so the numbers are the ones the job will actually run against. The
  record holds only the machine's **facts** — what was probed, and the
  `env_init` declared in that machine's `molbuilder.json` and copied in by the
  probe there; what you want stays in
  `molbuilder.json` ([`configuration.md` § 4–5](?doc=configuration.md)).
- **Trials are the stage's science, made measurable — by pins, not by
  splicing.** Each trial's deck is **rendered from the description** like any
  deck, with the benchmark's pins laid over the resolved values
  (`template.md § 8.1`: rebuild and render, never splice): SCF capped at 3
  iterations, relaxation steps zeroed (a single point — you are timing an
  iteration, not converging the chemistry), restart forced clean, and a
  **per-trial relabel** so no trial can read or overwrite the real run's warm
  files. **The GPU and the eigensolver are NOT pinned** — they are what the
  calculation's run card says, else the catalogue's default
  ([`job-request.md`](?doc=execution/job-request.md) § 3.4), because *"use GPU or not is set up only at the Job Prep
  UI"* ([`web/task-setup.md`](?doc=web/task-setup.md) § 6.2), and a benchmark
  measures what was asked
  ([`project-layout.md § 2.3.2`](?doc=execution/project-layout.md)).
- **The `(G, K, c)` grid** — `G` devices, `K` MPI ranks per device, `c`
  cores per rank; each point runs in its own `bench-…/` folder named by its
  coordinate. **What the points ARE is the declaration's question, and
  [`generator.md § 4.3a`](?doc=execution/generator.md) is the owner**:
  declared `bench` entries (`mpi_np`, `gpu_count`, `omp_threads`, and any
  value settings) drive the trials exactly; only an ABSENT declaration
  keeps the machine's own enumeration as the proposal (`K` from the probed
  core ladder, `c` as a starved / one-socket / cross-socket bracket, `G`
  over each rank count's divisors).

  **`G` is there only when the GPU is asked for** — the bench's `use_gpu`
  points, else the calculation's run card ([`job-request.md`](?doc=execution/job-request.md) § 3.4).
  A CPU-family point holds the device coordinate at `G = 0` — plain ranks,
  no `gres` — so a CPU trial does not queue behind a GPU node it never
  uses; a single-family CPU sweep drops `G` from its coordinates entirely
  (the points are `(K, c)`, the folders `bench-K<k>C<c>/`).

  > **`BlockSize` belongs on this grid too** *(decided 2026-08-11, user; not yet
  > built)*. It is a parallel-efficiency knob whose right value depends on the
  > matrix, the rank count, the interconnect and the node's memory layout at once
  > — so no formula reaches it and **a short test job does**
  > ([`tuning.md § 2.11`](?doc=engines/tuning.md)). It is a fourth axis of exactly
  > the same kind as the three above: powers of two, bounded above by the point
  > where a rank receives no block at all, and reported in `choice` beside the
  > rank and GPU counts so `prep` consumes it the same way. **The name of a trial
  > directory grows with the grid**, which `job-contracts.md` § 6.3's *a sweep
  > has no order, so the name carries what was tried* already anticipates.
- **Measure → `bench-result.json`** (`molbuilder/bench-result@1`): each point is
  parsed for its SCF wall-time **per iteration** (the first inter-iteration
  delta is dropped as warm-up and the rest averaged — under the capped
  3-iteration trial that is iteration 3's one clean sample, the user's
  2026-08-21 economy: the bench reads scaling, not tight rankings), plus a
  utilisation reading and peak memory.
  > **A trial is also asked what it actually ran, and a trial that ran
  > something else cannot win** *(2026-08-13)*. Each point's own artifacts are
  > read back — SIESTA's `* Running on N nodes in parallel.`, its
  > `* ProcessorY, Blocksize:` and `diag: Algorithm` lines, and the wrapper's
  > `ranks / omp` record (the only witness to the thread count) — into the
  > point's `effective` block, and compared against what it was asked for into
  > its `mismatch` block. Both travel in the record. **The reason is that the
  > fallbacks are silent**: an ELPA/GPU build that cannot initialise drops to
  > the CPU solver and says so only in its output, a launcher may hand back
  > fewer ranks than requested, and an `OMP_NUM_THREADS` in the environment
  > overrides the scheduler's `-c`. Without the comparison the row keeps the
  > label it asked for and is ranked against the others as though it measured
  > that configuration. A mismatched point stays in the record and is named in
  > the rationale; it is barred only from winning, and if **every** timed
  > point mismatches there is no winner at all rather than the least-wrong of
  > them -- `choice` then says why (`{"none": ...}`, `project-layout.md`
  > § 2.3's summary).

  The **winner is the fastest completed point** that ran what it was asked to
  run. It recommends no memory and no wall: those are the person's to state
  (`submission.md` S1/S2). The recorded choice is a **report, applied by
  nobody**: its `execution` block is yours to copy into `job-request.json`, and the
  next `prep task` reads what you wrote, and nothing else (§ 7.1 — there is
  no second rung).

`molbuilder jobset probe` is the companion that asks a live cluster
(`sinfo`/`sacctmgr`) what it is — the GPU type, the partitions and QoS you can
actually reach, and their wall limits — and writes that to `environment.json`
with `--write`, so every calculation on this machine reads one probed answer
instead of each re-probing its own.

It writes **facts only** — what it measured, and the activation and preamble
that machine's `molbuilder.json` declares, copied in. Which queue a job uses
is that job's own statement ([`architecture.md` § 5.2](?doc=execution/architecture.md));
`molbuilder.json` holds no scheduler settings — the split is
[`configuration.md` § 4–5](?doc=configuration.md) M-1: a probe never chooses
a queue on your behalf.

A trial executes only through `jobset launch bench` — a rider of its resource
shelf's grouped job on a queue, alone when named, or run here in the
benchmark's one walk — with its launch recorded in the trial's own
`run.json`.

---

### 7.1 One analysis, two presentations, nothing in between

**The measurement is composed once and shown twice.** There is no third
implementation and no file between the composer and either reader.

```mermaid
flowchart LR
    A["the trials' own artifacts<br/>(scf-timing.log · monitor.log · util.csv · the deck)"]
    C["summarize.bench_record<br/>= the points, then the verdict,<br/>then how the winner computed"]
    V["sweep_view<br/>+ jobset_status, read-only"]
    R["run_summarize_jobset<br/>+ writes bench-result.json"]
    W["the Results panel<br/>GET /api/bench/summary"]
    T["the terminal<br/>jobset summarize"]
    A --> C
    C --> V --> W
    C --> R --> T
```

**Both readers ask the same composer**, and the composer is `bench_record`.
Its own docstring is the authority: *"Both readers of a sweep come through
here … so one sweep has one verdict whichever door you came through"* — the
winner's mechanism (`_winner_mechanism`) included.

**They differ AFTER it, and only in what they do with the record.**
`sweep_view` adds live per-trial status and returns; it opens no saved file,
so the panel recomputes on every request and can show a sweep mid-flight,
which a snapshot cannot. `run_summarize_jobset` writes `bench-result.json`,
the archival record below, and hands back a report the CLI PRINTS.

| reader | how it gets the summary | writes a file? |
|---|---|---|
| the Results panel | `/api/bench/summary?path=<job-set.json>` → `sweep_view` → `bench_record` | **no** |
| the terminal | `jobset summarize bench <stage>` → `run_summarize_jobset` → `bench_record` | **`bench-result.json`**, then prints the report |

**The terminal prints the report; it writes no text file beside it**
*(user, 2026-09-04: "that can be simplified by integrating that into the CLI,
which basically prints out instead of writing to a text")*: submit a sweep,
come back and ask — a terminal is for asking.

#### The one thing persisted

`bench-result.json` is the sweep's **archival record**: the trials, their
measured numbers, the machine each ran on, and the verdict. It is what
survives if the trials' artifacts are archived or deleted, and it is the
only thing that does. **No code reads it back**: what a run uses is what the
person wrote in `execution` (`prep_inputs.prep_run_inputs`, step 2 — there
is no second rung), so nothing carries a verdict into a launch. That is its
normal condition, never a reason to retire it — the record exists to outlive
the artifacts it was computed from *(user, 2026-09-06: archiving a sweep is
a use)*.


## 8. Where it stands

### Shipped (command line)

The `JobSet` model and persistence; the description-to-plan derivation at
`prep` (§ 4 — the `ParameterSet`, one deck and wrapper per element); the
loop's verbs (`prep` / `launch` / `summarize` / `status`, beside `molbuilder describe`), a launch
in `direct`, `submit` or `ask` mode, and beside them `scratch`, `migrate`,
`probe` and `machines` (§ 5.3); SLURM submission with routing domains, **one
job per invocation** — a benchmark's, one per resource shelf; the full
benchmark workflow through the same loop (§ 7); the jobs launched from this
machine, `molbuilder jobs` (§ 6.0a).
Saving and re-entering a calculation's states is `molbuilder checkpoint`
([`running-a-job.md § 6`](?doc=execution/running-a-job.md)).

> **No dependency chains** (§ 2, decision 6; § 3's *"there is no edge anywhere
> in this picture"*): one `sbatch` per job — a task's launch sends one, a
> benchmark's one per resource shelf — each sending the job's own header and
> waiting on none.

### The web, and the engines

**The web describes, prepares and observes; `launch` stays on the terminal
by design** (§ 1, *Where it stands today*):

- **the description** — the parameter tabs hand over to the shared **Task
  setup** tab, which writes the same `task.json` + `job-request.json` the CLI
  verb writes, beside the hand-over's template ([`web/task-setup.md`](?doc=web/task-setup.md),
  [`web/handover-procedure.md`](?doc=web/handover-procedure.md)); the
  target's `prep` derives the `JobSet` from it (§ 4);
- **prep** — Task setup's **Preview task** / **Prep task here** and
  **Preview bench** / **Prep bench here** call the entry the terminal calls,
  a preview the plan stopped before the save (§ 5.3, *One prep, two doors*);
- **status** — the Results tab's ladder is `status`'s answer, read-only, from
  the run decoder the terminal reads (§ 5.3;
  [`web/results.md`](?doc=web/results.md) § 2.4). There is no web plan view
  (W14). Branching is restore-then-save, both in the sidebar's checkpoint
  panel ([`checkpointing.md`](?doc=execution/checkpointing.md) § 7.1).

SIESTA, PySCF (its `vibration` kind among them) and the transport composite
each prepare through their engine seam (§ 9). A transport calculation's
citation composes at prep, so its job set carries no edges, and a bias scan
walks its points inside one submission (§ 6.0).

Out of scope: **multi-node MPI** (v1 fixes one node), and a `molbuilder config
init --site` command (a site preset ships only as a JSON example file). Open
work — the cluster half among it (E7) — is
[`plans/plan.md`](?doc=plans/plan.md)'s.

Run reports — the monitor telling Slack, Discord or molbuilder's own listener
how a run is going — are [`run-reports.md`](?doc=execution/run-reports.md)'s.

The through-line: the CLI framework on this page is the settled foundation, and
the web work is *additive on top of it* — it reuses the same five-step
`prep`, decoders, and wrappers rather than reinventing them.

---

## 9. A developer's map

### 9.1 Prep and launch — one framework, layer by layer

Both verbs are built the same way, and each layer has one role. A **surface**
collects what the person said and prints the answer; **one entry** per verb is
what both surfaces call; **the work** says which members it acts on; **one
member planner** per verb plans a member; a **run filler** fills that member's
run by its layout; for a launch, a **submission builder** says how the
members are sent, asking the member planner for each; then the folder is
**saved, the plan written, the job sent, the records written**. Underneath, the **doors** answer facts and never write. Every layer
above the doors decides; nothing is written until the plan is whole, so a
refusal writes nothing but its line in the ledger.

```mermaid
flowchart TB
    subgraph S["surfaces — collect what was said, print the answer"]
        CLI["the terminal<br/>_cli.prep_cmd · submit_cmd · status_cmd"]
        WEB["Task setup<br/>/api/task-setup/prep · /folder · /commands"]
    end
    subgraph E["entries — one per verb"]
        PE["prep.prep_task · prep.prep_bench"]
        LE["submit.plan_launch → submit.send_launch"]
    end
    subgraph W["the work — which members"]
        PW["the groups picked, formed again whole<br/>(group.formed_again)"]
        LW["the group or the stage named, or offered<br/>(submit._the_group · ready.TaskState.launch_offer)"]
    end
    subgraph M["the member planner — one per verb"]
        PM["prep.plan_stage<br/>inputs · resolve · place · steps planned"]
        LM["submit._plan_member<br/>_first_launch · _launch_again · _plan_trial"]
    end
    subgraph F["the run filler — by the run's layout"]
        F1["one run folder<br/>materialize.prepare_attempt"]
        F2["a run of points<br/>materialize.open_sweep_run"]
        F0["flat — the folder is the run"]
        F3["what it takes from other runs — one hand-over<br/>materialize.hand_over (HandOver)"]
    end
    subgraph B["the submission builder — how members are sent"]
        B1["one member, one job<br/>submit._plan_stage"]
        B2["a group's members, one job walking them<br/>submit._plan_group"]
        B3["a run's points, one job walking them<br/>submit._points_walk"]
        B4["a benchmark's trials<br/>submit._plan_shelves · _plan_bench_here"]
    end
    subgraph X["save, write, send, record"]
        SV["checkpoint.save_before"]
        PL["planned.Plan.carry_out"]
        GO["send_launch → submit._go — the one sender · run.json"]
        LG["ledger"]
    end
    subgraph D["the doors — answer, never write"]
        RD["ready<br/>task_state · readiness · first_launch_takes · regrouped"]
        CD["continuation<br/>continuation_answer · relaunch"]
        GD["gather<br/>gather_sources · transport_inputs"]
        RU["the run doors<br/>materialize.ever_launched · parse.dirs.run_status"]
        GR["group<br/>group_of · members_of · envelope · names_for"]
        CM["commands<br/>every printed line · way_back"]
    end
    CLI --> PE & LE
    WEB --> PE
    PE --> PW --> PM --> F1 & F2 & F0
    LE --> LW --> B1 & B2 & B4
    B1 & B2 & B4 --> LM --> F1 & F2 & F0
    F1 & F2 & F0 --> F3
    B1 --> B3
    B3 --> F3
    PE --> X
    LE --> X
    PW -.asks.-> D
    PM -.asks.-> D
    LW -.asks.-> D
    LM -.asks.-> D
```

| layer | its role | `prep` | `launch` |
|---|---|---|---|
| surface | collect what was said — the flags, a question's answer — and print the answer; decides nothing | `_cli.prep_cmd`, `POST /api/task-setup/prep` | `_cli.submit_cmd`, the `launch` verb (`_launch_unit` puts the question) |
| entry | the verb's one function, whichever surface called; owns the save and the record | `prep.prep_task` (a task's groups), `prep.prep_bench` (one stage's benchmark) | `submit.plan_launch` (the plan, a refusal written down) → `submit.send_launch` |
| the work | which members: by the job request's groups, matched by their members | `_prep_groups`: the groups picked — a pick missing a group a changed job's members are in is refused, naming it (`group.formed_again`); a launched member prepared again only as the same calculation (R-P5) | `_the_group`: the group named or offered, or one stage launched again; a launched group's members fixed |
| member planner | one member — what it is, what it takes, its checks, its run — planned with nothing written | `prep.plan_stage`: inputs, step 2 resolved, launch values stated and placed, the steps planned (`prep_calculation`) | `submit._plan_member`: `_first_launch` (what it builds on, from the ready door; the prepared stage checked against `plan_stage`, quiet) or `_launch_again` (from `_newest_launched`); `_plan_trial` for a benchmark's trial |
| run filler | fill the member's run by its layout | `prep_calculation` opens `run-0` with the stage's own files (`materialize.prepare_attempt`, `open_sweep_run`) — nothing taken from another run | the same openers, then prep's own files back (`_prep_files_back`) and what it takes, each one hand-over carried out by `materialize.hand_over` (§ 5.4): the run it continues from (`prepare_attempt`), a gather (`gather.gather_into`) or the one its last run was given (`_carry_the_gather`); a run of points' takeovers and starts (`_points_walk`); a flat run's record at the send (`_go`) |
| submission builder | how the members are sent — it asks the member planner for each member, then builds the job | — (prep sends nothing; every job's one header is planned with its members — a group's, a run of points' walk's, a benchmark's shelves', `prep.walk_header_of`) | one job (`_plan_stage`, which hands a run of points to `_points_walk`); a group's walk (`_plan_group`); a benchmark's shelves (`_plan_shelves`, `_plan_bench_here`) |
| save, write, send, record | only once the plan is whole | `checkpoint.save_before`, `Plan.carry_out`, `_record_prepared` | `send_launch`: the folder as planned, the save at a group's first launch, `Plan.carry_out`, `_go`, each member's `run.json` |

**The doors** — one answer each, asked by every layer above and by `status`,
the Results tab and Task setup, so no two readers can disagree:

| door | the question it answers | asked by |
|---|---|---|
| `ready.task_state` → `TaskState` | the task read once: every stage's answer and every group's; what `prep task` and `launch task` offer | prep's question, launch's offer, `status`, the folder and commands routes, describe's next line |
| `ready.readiness` | one stage: described, prepared (ready, or why not — `blocked`: waits, regrouped, unreadable, again) or launched | `task_state`, the displacement sweep |
| `Calculation.job` | is a stage prepared — its row in `job-set.json` (`JobSet.job`), none with no plan | prep, the readiness doors, the launch verb |
| `naming` | a stage's number, folder and token (`stage_home`), a plan's job folders (`job_dir_names`), which stage a job is (`stage_refs`), the names of its files (`run_names`) | every layer from the plan up |
| `ladder` | what a stage builds on by default (`source_stage`, `stage_before`), and the ladder a new description starts with (`fresh_stages`) | the continuation, the ready door, prep, `molbuilder describe` and the hand-over |
| `ready.first_launch_takes` | what a stage's first launch takes, every kind's refusal with it | the ready door's answer, `_first_launch` |
| `continuation` | which run a stage continues from — by default, named, or launched again (`relaunch`) | the ready door, both planners |
| `gather` | a transport rung's inputs, under three gates | `first_launch_takes`, `_launch_again --gather` |
| `materialize.ever_launched`, `rundir.run_state_of` (`parse.dirs.run_status`), `materialize.latest_attempt`, `runrecord.newest_run` | was a stage launched; how a run stands; a stage's newest run folder; the newest run in a folder — where a stage's runs share it, its newest launched; in a run's own folder, its newest try | every layer |
| `group` | a stage's group (`JobRequest.group_of`), a job's members (`members_of`), what members share (`envelope`), a job's file names (`names_for`) | both verbs, the ready door |
| `commands` | every line a person types, and the one way back (`way_back`, `redo_launched`) | every refusal and answer |

**One `prep task`, through the layers** — two groups picked, the second of
two members:

```mermaid
sequenceDiagram
    participant S as surface (prep_cmd / Prep)
    participant E as prep.prep_task
    participant R as ready (door)
    participant P as prep.plan_stage
    participant F as prep_calculation (filler)
    participant X as checkpoint · Plan · ledger
    S->>S: calculation.open — the calculation, once (checkpoint 1)
    S->>R: task_state(calc) — the groups, each stage's state
    R-->>S: TaskState (prep_offer)
    S->>E: prep_task(calc, "task", groups=…, asked=…)
    E->>E: the groups the job request holds — a changed job's formed again whole
    E->>R: Calculation.job / launched_without — a launched member: the same calculation, or refused (R-P5)
    E->>E: checked — preflight, machine, provenance (3, 4)
    loop every member of every group picked
        E->>P: plan_stage(base, described, "task", stage, plan=the one plan)
        P->>F: prep_calculation — decks, scripts, run-0, its row
        F-->>P: what it planned
        P-->>E: PrepAnswer (what its first launch takes)
    end
    E->>E: each group of several — envelope, placed, its header
    E->>E: rows carry their members — STAGE-PLAN.md once
    E->>X: save_before (once) — then carry_out — then the record
    E-->>S: the answers, the saved state, the next line
```

**One `launch task`, through the layers** — a group's first launch:

```mermaid
sequenceDiagram
    participant S as surface (submit_cmd)
    participant E as submit.plan_launch
    participant W as submit._the_group
    participant M as submit._plan_member
    participant R as ready · gather (doors)
    participant P as prep.plan_stage (quiet)
    participant B as submission builder
    participant X as send_launch
    S->>S: calculation.open — the calculation, once
    S->>R: task_state — launch_offer (asked which, when several)
    S->>E: plan_launch(calc, mode=…, group=N, …)
    E->>E: _machine_of — the calculation's copy of its machine's record, read once, first
    E->>W: the members — matched by membership, fixed once launched
    E->>B: _plan_group(calc, members, …)
    loop every member
        B->>M: _plan_member — its first launch
        M->>R: first_launch_takes — the run it builds on, or the gather
        R-->>M: what it takes, or the refusal
        M->>M: run-0 emptied, prep's files back, what it takes
        M-->>B: the member
    end
    B->>B: the walk — the header its prep wrote
    B-->>E: the LaunchPlan — one submission
    E-->>S: LaunchPlan — shown, asked once
    S->>X: send_launch(plan, said=…)
    X->>X: the folder as planned (the plan made again) — save — carry_out — sbatch — run.json each
```

### 9.2 Where each responsibility lives

For someone extending the framework:

| Concern | Module |
|---|---|
| The data model + `job-set.json` read/write + `validate()` | `molbuilder/jobset/model.py` |
| The description + this machine → `ParameterSet` (`prep` step 2; the config ↔ exchange translation boundary) | `molbuilder/resolve.py` |
| SIESTA's stage knowledge — the shipped ladder, the warm-file declaration, the traits — consumed by the engine seam | `molbuilder/siesta/stages.py` |
| The benchmark grid — the `(G × K × c)` enumeration `prep bench` consumes | `molbuilder/bench/grid.py` |
| Lay out the materialized tree — job folders with their copies; the one opener of every run folder (`open_run`: a stage's attempt, a trial's, a bias point's — each stamped with the containers above it); whether a stage was launched (`ever_launched`); what a run takes from other runs, one hand-over (`HandOver`, carried out by `hand_over`, § 5.4); an engine's scratch a calculation's runs hold, each file with its stage, size and run (`scratch_in`, which `runstatus.scratch_answer` reads) | `molbuilder/jobset/materialize.py` |
| The names the plan gives — a stage's number and folder, read off the disk (`stage_home`), a plan's job folders (`job_dir_names`), which stage a job is (`stage_refs`), the names of its files (`run_names`), the naming authority's refusal (`NamingError`); a folder that holds files and never a run, opened and stamped (`open_container`: a submission's `launch/`, the calculation's `pseudos/`, a point's folder below its stage) | `molbuilder/jobset/naming.py` |
| The ladder's rule — what a stage builds on by default (`source_stage`, `stage_before`), and the ladder a new description starts with (`fresh_stages`) | `molbuilder/jobset/ladder.py` |
| The calculation, opened once per verb with its description, its job request, its template and its plan (`calculation.open` → `Calculation`, `Calculation.request`); a stage's row in the plan (`Calculation.job`) | `molbuilder/calculation.py` |
| The job request — `job-request.json` read through its door (`of`, `from_text`), written whole (`write`), each rung's run card laid over the calculation's (`JobRequest.run_condition`), the groups', the benchmark's and the reports' blocks ([`job-request.md`](?doc=execution/job-request.md)) | `molbuilder/job_request.py` |
| The template file's door — its name (`template_filename`, `template_path`) and the folder's one template read (`of`) | `molbuilder/template_file.py` |
| The prep verb's entries, which the command line and the Task setup tab both call (§ 5.3): `prep_task` (a task's groups — read and checked once, every member into one plan, each group's header `group_header`) and `prep_bench` (one stage's benchmark), each → `PrepAnswer`; the calculation opened once and checked (`calculation.open`, `checked`); the one member planner, `plan_stage` (launch asks it too, quiet), handing the steps to `prep_calculation` into a `jobset.planned.Plan` — [`script-preparation.md`](?doc=execution/script-preparation.md) § 3, decided with nothing written; the save (`checkpoint.save_before`), the writing (`Plan.carry_out`), the record; a benchmark prepared (`prepared_already`) and the stages that fix the shape (`prepared_stages`) | `molbuilder/jobset/prep.py` |
| The engine seam — an engine's deck writer, its data files, its warm declaration and traits, its config class (one map for every caller) | `molbuilder/jobset/engines.py` |
| A job's placement — queue, wall, memory, ranks, cores, GPUs with their sources — and its fit, said (§ 6.0); the launch-value check | `molbuilder/jobset/placement.py` |
| A benchmark's shelves — its trials by side and exact ask, widest first (`shelves_of`), and each shelf's one request (`shelf_request`), which prep writes each shelf's header from; launch sends that header and walks the shelf's trials | `molbuilder/jobset/shelves.py` |
| The refusal type every floor raises (`PrepError`), and the person's errors said as one (`user_error_as_prep`) | `molbuilder/jobset/errors.py` |
| A transport rung built from the description as it reads now — its junction record read (`junction_record`, the one read prep and the gather share), resolved (`resolve_transport`), its structure and electronic state (`transport_parts`), each deck's spec at its point (`transport_spec`, `at_frame`, `at_point`), its pseudopotentials (`provide_pseudos`): prep's conductor writes it, the gather renders it to compare | `molbuilder/jobset/transport_rung.py` |
| The gather door — a transport rung's inputs from the runs upstream, under three gates (`gather_sources`, `transport_inputs`), the junction and each upstream rung's deck as it renders now read once per gather (`composed_junction`, `rung_deck_now`); the junction composed at prep, only while no stage is launched (`composed_for_prep`, its refusal `no_usable_record`), handed over with their record (`gather_into`, through `materialize.hand_over`) | `molbuilder/jobset/gather.py` |
| The calculation's machine — its record read (`machine_record`), checked (`require_activation`), and set at the first prep (`set_machine`, `configuration.md` M-3) | `molbuilder/jobset/machine.py` |
| A run's own records — `run.json` read and written through `persist`, how a run ended (its conclusion marker), `.continued-from`, `.gathered-from`, a run as they name it (`RunRef`); the newest run in a folder (`newest_run`) and the number a new one takes (`next_run`) | `molbuilder/runrecord.py` |
| What a prep receives, assembled once (A12) — the run's `(allocation, pins, chosen)` and the bench's `(points, pins, translation)`, the bench grid's cell checks, and the notes a person is told; the conductor's own assembly, beside it | `molbuilder/jobset/prep_inputs.py` |
| The human-readable plan table | `molbuilder/jobset/plan.py` |
| The launch verb's one entry — the plan (`plan_launch` → `LaunchPlan`: the calculation's copy of its machine's record, read once, first, `_machine_of`; the work, `_the_group`; the submission builders, `_plan_stage`, `_plan_group`, `_points_walk`, a benchmark's `_plan_shelves` and `_plan_bench_here`; the one member planner, `_plan_member` — `_first_launch`, `_launch_again` — and a trial's, `_plan_trial`; placements; the exact lines), then the send (`send_launch`): the one sender, the one launch request (`_sbatch_request`); a sweep sent to a scheduler goes one job per shelf, never one per trial (§ 6.0) | `molbuilder/jobset/submit.py` |
| Per-stage status roll-up (reuses `run_status`), and each group's next step (`render_steps`); each member of a job as `molbuilder jobs` shows it (`job_members`); which of a calculation's scratch a run still going keeps, and why (`scratch_answer` → `ScratchAnswer`), what `jobset scratch` shows and removes | `molbuilder/jobset/runstatus.py` |
| What a run continues from — the default, a named run, or why it is refused — as one `Continuation`, the files it carries included; a stage launched again (`relaunch`); the one rule for a run to build on (§ 5.4) | `molbuilder/jobset/continuation.py` |
| A group — its number in `job-request.json` (`JobRequest.group_of`, `named`): what a stage builds on (`upstream_of`), the defaults (`default_groups`), the check (`check_groups`), what the members share (`SHARED`, `envelope`), a prepared job's members (`members_of`) matched by membership (`same_members`) and the groups a prep forms again (`formed_again`), the job's files' names (`names_for`; a job already written, `header_of`) | `molbuilder/jobset/group.py` |
| The ready door — the task read once (`task_state` → `TaskState`); each stage's answer (`readiness`: described / prepared — ready to launch, or why not — / launched); a launched group's members fixed (`launched_without`, `joined_launched`) and a prepared job's members changed (`regrouped`); what a first launch takes, every kind's refusal with it (`first_launch_takes`); the groups and the offers `prep task` and `launch task` make (*The task*) | `molbuilder/jobset/ready.py` |
| Every command a verb or a refusal prints for a person to type, and the one way back (`way_back`, `redo_launched`); a launch and the way back as a text read later says them (a deck's header, a run script's banner, a result's remedy) | `molbuilder/jobset/commands.py`; `molbuilder/identity.py::launch_as_typed`, `deck_launch`, `way_back_words` |
| The sweep's reader — trials' artifacts → `bench-result.json` (the pure timing parsers are `molbuilder/bench/result.py`) | `molbuilder/jobset/summarize.py` |
| The decision ledger (`jobset-decisions.log`, one JSON object per decision) | `molbuilder/jobset/ledger.py` |
| The CLI verbs (`molbuilder jobset …`) | `molbuilder/jobset/_cli.py` |
| The jobs launched from this machine — each launch's address, written by the one sender, read fresh, and whether a job is going (`is_going`, the one answer `molbuilder jobs`, `jobset scratch` and a run of points' state (`continuation.points_state`) ask; § 6.0a) | `molbuilder/joblist.py`; the verbs, `molbuilder/jobs_cli.py`; a queue job asked of the scheduler, `molbuilder/scheduler/queue.py`; each member's state, `runstatus.job_members` |
| The `.sbatch` header emission (shared with single-job) | `molbuilder/runwrap.py::render_sbatch` |

**To add a new engine to the job system**, you write **one engine seam** — the
plugin that teaches `prep` step 3 to render that engine's decks from a resolved
config ([`generator.md § 7`](?doc=execution/generator.md)) — and nothing else.
The verbs, the materializer, the submitter, and the status layer are all
engine-agnostic and pick it up for free. That is the whole payoff of decision
#1.
