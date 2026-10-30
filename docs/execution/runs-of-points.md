# Runs of points — one run, many engine runs

**Role:** contract
**Domain:** execution
**Companions:**
[`job-system.md`](?doc=execution/job-system.md) — *The words* (run, stage,
group, warm and cold), § 5.4 (a hand-over), § 6.0 (launching);
[`project-layout.md`](?doc=execution/project-layout.md) § 1.4a (a folder's
record), § 1.6 (a run and its tries);
[`run-reports.md`](?doc=execution/run-reports.md) § 2.4, § 2.8, § 2.9 (what
a run says, and when);
[`architecture.md`](?doc=execution/architecture.md) § 3.3 (the kind table,
and a part that is code);
[`engines/transport.md`](?doc=engines/transport.md) § 2a.11 (a transport
rung's points) and [`engines/vibration.md`](?doc=engines/vibration.md)
§ 5.11 (a SIESTA vibration's calls) — each kind's own.

> *The user, 2026-10-30*: *"make sure this unification is a framework level,
> api level work, use config file and dir design to drive this, abstraction
> helps to unify the two, and clear contract and config file/data design
> would allow correct parser processing"*; *"clarify the definition of terms
> such that we can better design the abstraction and data/config support to
> drive such logic and provide flexibility for extension"*; *"and define
> operations/parsing tasks that is needed for such design, such that the api
> level can provide unified support"*.

Some runs are more than one engine run. A transport rung swept over the
frames of a cited set and a list of voltages runs TranSIESTA once per frame
and voltage; a SIESTA vibration runs SIESTA once to relax, once for the
force constants and once for each optical frame. Each is still **one run**:
one launch, one job, one folder `run-<n>/`, one state, one report. This
chapter is the framework every such run uses: the words (§ 1), the data and
configuration that drive it (§ 2), the operations on it (§ 3), the rules
they apply (§ 4), what is read and by which reader (§ 5), the doors (§ 6)
and what a new kind of run of points is (§ 7). A kind's own chapter says
only what is its own — what its points are and what they mean.

---

## 1. The words

Every word below is used in one sense only, in this chapter, in the data
keys (§ 2) and in the doors' names (§ 6). The kinds' own words map onto
them in the last two columns.

### 1.1 The run and its points

Where each of these lies on disk — the stage folder, the prepared folders,
the run, its points, its containers, its scratch — and the one door that
names each is § 2.4.

| word | what it is | a transport rung | a SIESTA vibration |
|---|---|---|---|
| **run of points** | a run — one launch of a stage, `run-<n>/` — whose work is several engine runs, each in a folder of its own below the run; walked by one job, it reports as one run, and its state is its points' | a device or transmission run over frames × voltages | the stage's one run |
| **point** | one engine run of a run of points: a folder below the run that is a run folder of its own — its deck, run script, monitor, output, session log and conclusion — named by its path below the run, its **folder** (`rel`) | `f001/v0.2` | `relax`, `fc`, `optical/f003` |
| **unit** | what the kind calls one point in a person's words | *point* | *call* |
| **listing order** | the order the run's points are listed in everywhere — `status`, the index, the Results ladder, the first point not done | frames outer, each frame's voltages in the description's order | `relax`, `fc`, then the optical frames in the order they were planned |
| **prepared point** | a point whose deck prep writes, in its **prepared folder** below the stage | every point: `04_device/f001/v0.2/` | `relax`, `fc`: `01_freq/relax/`, `01_freq/fc/` |
| **planned point** | a point the run's own job adds while the run goes, by the after-step of its **planner** (§ 1.3), in a **container** below the run; its deck is its container's prepared folder's | — | each optical frame, planned by `fc`, in `optical/`, its deck `01_freq/optical/`'s |
| **container** | a folder below the run that holds planned points — a container folder ([`project-layout.md`](?doc=execution/project-layout.md) § 1.4a), opened with the run | — | `optical/` |
| **phase** | what work a point does — one kind of deck. Points of one phase cost alike, so a pace and a count are a phase's. (A run's phase, not a part of an engine's output, [`run-reports.md`](?doc=execution/run-reports.md) § 2.8) | the rung: every point one phase | `relax`, `fc`, `optical` |
| **round** | the points walked together and reported together: every point of a round runs before the next round's, and the last of a round to end says the round's outcome | one voltage at every frame | one phase |

### 1.2 How a point runs

| word | what it is | a transport rung | a SIESTA vibration |
|---|---|---|---|
| **start** | what a point starts from: another point of the run — its **source** — and the files it takes from it. One hand-over ([`job-system.md`](?doc=execution/job-system.md) § 5.4): recorded when the launch is planned, its files copied by the walk just before the point runs. A planned point's start is copied by its loop step as the walk reaches it, with no hand-over record — launch does not know the point when it plans; its kind's facts in the index say it. A point with no start begins from what its run gathered, or from what prep wrote | the closest earlier voltage in its frame, its `.TSDE`; a frame's first voltage none (its gathered seed) | `fc` from `relax`'s `.DM`; an optical frame from `fc`'s `.DM`; `relax` none |
| **need** | a point that must have **ran** (§ 1.3) before this one runs: its start's source, and any whose result it reads. Up the chain, a need that ended without running ends what builds on it | its start's source | `fc` needs `relax`; an optical frame needs `fc` |
| **taken over** | a point **done** in the stage's newest launched run, copied whole into the new run by a warm launch and not run again — its record naming the run that computed it, one hop | | |
| **relaunch** | what a launch again does with a point not done: **again** — from its start, as the first walk ran it; or **continue** — from its own last state, its job's warm files handed over from its last run's folder | again | `relax` continue; `fc` and the optical frames again |
| **wall** | **one** — the run's points share the stage's one wall, counted from the run's start; **each** — every point is told the card's wall, the job's wall their sum | each | one |
| **the walk** | the job's script that runs the run's points round by round, `launch/<stem>-sweep.run.sh`: a **walk step** for each prepared point it runs, a **loop step** for each container — the planned points there when the step comes | | |
| **the kind's steps** | the kind's own work in the job around a point's engine, verbs of the job's bundle: **before** (what the engine reads), **after** (what the point came to — and, for a planner, the points it plans), and the run's **finish step** once no point is left to run | none | `plan` (the call's coordinates), `record` (the relaxation's record; the force constants' analysis, the result, the optical frames planned; the optical data), `finish-run` (the result, final) |
| **report steps** | the two steps around every point's engine — `point started`, `point ended` — which stamp the index, run the kind's steps in their place, and say the run's events | | |

### 1.3 How a point and the run are judged

| word | what it is |
|---|---|
| **ran** | the point's run finished: its engine exited 0 and its own steps succeeded — converged or not. A point taken over ran |
| **done** | it ran, its verdict says converged where its verdict counts, and it holds what it must — its products, each with the words its absence is said in |
| **failed** | it ended without running |
| **ended** | it ran or failed, it was taken over, or the walk ended without running it |
| **still to run** | not ended, and every need up its chain that has ended ran |
| **the run finished** | **done**: every point done — the kind's result is built from them all (a vibration's); **ran**: every point ran, converged or not — the converged column says the rest (a transport rung's). Otherwise failed, naming the first point not done in the listing order |

| | a transport rung | a SIESTA vibration |
|---|---|---|
| what a point must hold | the device: `.TS.HSX`, `.TSDE`; the transmission: `.TBT.nc` (or `.TBT_UP.nc` and `.TBT_DN.nc`); and `shared/` with its record where the point shares energies with a finished point | `relax`: nothing; `fc`: `.FC`; an optical frame: `.EPSIMG` |
| whose verdict counts | every point's | `fc`'s and an optical frame's; never the relaxation's — it is done finished, converged or not |

### 1.4 How the run speaks

| word | what it is | a transport rung | a SIESTA vibration |
|---|---|---|---|
| **the index** | the run's one record of its points as the job sees them, at the run's root (§ 2.3) | | |
| **count** | the run's progress, in the kind's words; its *failed* counts the points that ended without being done | *7 of 12 points done, 1 failed (3 frames × 4 voltages)* | *relax done · fc done · optical 7 of 11 (4 taken over)* |
| **outcome** | what a point came to, in its kind's words, written by its after-step; a **round's outcome** is the kind's words over the round's points | the round's: each frame's NEGF iterations, or its G/G₀ at E_F and the frames' average | the relaxation's moves and largest force; the force constants' modes, stationarity and the frames planned |
| **events** | `start`, `milestone`, `part_failed`, `finish` — [`run-reports.md`](?doc=execution/run-reports.md) § 2.9 | | |

---

## 2. The data — what drives it

Four declarations, each with one door; nothing in the shared code names a
kind, a phase or a file of one — but the indexes an older molbuilder wrote,
which `runpoints` refuses by name and `jobset migrate` rewrites
(`runpoints.OLDER_SCHEMAS`, `older_index`; § 3, op. 18).

| declaration | holds | its door |
|---|---|---|
| the kind table's record of the pair's run of points (configuration, § 2.1) | what never changes for the pair: its unit, when its run is finished, its wall, its words, its steps, its declaration | `kind_table` (`run_of_points`) |
| the pair's declaration (code, floor 4, § 2.2) | what the kind's points are: which, what each must hold, where each starts, what a relaunch does, its containers | `jobset.points` (`points_of`) |
| the run's index (data, § 2.3) | the run's points as the job sees them — each self-describing, so the job judges every point with no molbuilder installed | `runpoints` (`read_index`) |
| the folders' records (§ 2.4) | which folder is the run, a point, a container | `calcdirs.where` |

### 2.1 The kind table — what a pair's run of points is

```toml
[transport.on.siesta.points]
declaration = "molbuilder.jobset.transport_points:POINTS"
unit        = "point"
finished    = "ran"
wall        = "each"
count       = "molbuilder.transport.run_words:count_words"
outcome     = "molbuilder.transport.run_words:round_outcome"

[vibration.on.siesta.points]
declaration = "molbuilder.jobset.vibration_points:POINTS"
unit        = "call"
finished    = "done"
wall        = "one"
count       = "molbuilder.siesta.vibration_phases:count_words"
outcome     = "molbuilder.runpoints:outcomes"
steps       = { before = "plan", after = "record", finish = "finish-run" }
```

| key | what it says | read by |
|---|---|---|
| `declaration` | the pair's declaration (§ 2.2), a part of floor 4 | every floor-4 door below |
| `unit` | the unit (§ 1.1) | prep's words, launch's words and refusals; copied into the index |
| `finished` | `done` or `ran` (§ 1.3) | `status`; copied into the index |
| `wall` | `one` or `each` (§ 1.2) | prep (the header's wall), the walk (the run's start stated once) |
| `count` | the count's words, a part of floor 1 that travels beside the job | `status`, the monitor, the events; copied into the index |
| `outcome` | a round's outcome, a part of floor 1 that travels | the events; copied into the index |
| `steps` | the kind's steps, verbs of the job's bundle (`Job.finish`); absent where the kind does no work between points | prep renders them into each point's run script |

A pair without the table runs each rung once. The door refuses a key it
does not know, a value outside its words and a part not named
`module:NAME`, naming the file and the key. A part is resolved where it is
asked ([`architecture.md`](?doc=execution/architecture.md) § 3.3), and a
part that does not resolve is refused there naming it: on this machine by
`kind_table.part`; beside the job by `runpoints.part`, among the modules
that travel beside it — the module's last name, as the bundle holds it
(`runwrap.MONITOR_COMPANIONS`).

### 2.2 The declaration — what a kind's points are

`jobset.points.PointsOf`, one per pair, named by the record's
`declaration`. Each part answers one question; the shared code asks it and
nothing else.

| part | answers | transport | SIESTA vibration |
|---|---|---|---|
| `rung_points(calc, stage, read=None)` | the prepared points, in the listing order — `()` for a rung that runs once | a frame of the cited set × a voltage, where either varies (`transport.stages.rung_points`) | `relax` and `fc`, as the stage's job lists its phases (`Job.phases`) |
| `points_in(run, task, stage, base)` | the points a run holds, each with its folder | the prepared points whose folders the run holds | the index's points; before the launch wrote one, the prepared points |
| `entry(calc, stage, point, deck)` | the kind's part of the point's index entry: its words, phase, round and facts — `deck` the path of the deck prep wrote for it | its words, the rung, its voltage, `{frame, frame_words, volts, bias_v, window}` | its words, its phase twice, `{name, mode, side, frequency_cm1, dq_amu12_ang}` |
| `point_of(entry)` | the point an index entry is | a frame and a voltage | a call |
| `must_hold(stage, label, base, point, folder, deck=None)` | what the point must hold to be done, each with the words its absence is said in — `deck` a prepared deck's path where no run holds the point's deck yet | the rung's products; `shared/` where its deck shares | its phase's products |
| `verdict_counts(stage, point)` | whether its verdict counts toward done | yes | its phase's |
| `start_of(calc, stage, point, points)` | its start — source and files — and its needs, among the run's points | the closest earlier voltage in its frame, the rung's `along` files; the transmission none | its phase's seed and needs (`PHASES`) |
| `relaunch_of(stage, point)` | `again` or `continue` | again | its phase's (`PhaseRule.relaunch`) |
| `containers(calc, stage)` | each container: its point (its folder and prepared folder), its planner, and the inputs its planning writes into each planned point | none | `optical/`, planned by `fc`, each frame's `coordinates.fdf` |
| `run_facts(calc, stage, *, carried, deck)` | the kind's facts about the run, for its index — `carried`, those of the run whose planner this launch took over; `deck`, a prepared point's deck's path | what its points compute, the leads' pair, the spin, the frames' weights | the phases it walks; the selection, carried |
| `also_needs(run, names, task, stage, base)` | what the run needs beyond its points done to be finished — `None` when nothing; a finish step that exited non-zero is the run's state's own rule (§ 4.4) | nothing | its finish step ended, and its result at the run's root |

A point is the kind's own type — `transport.stages.Point`,
`siesta.vibration_phases.Call` — and every one answers `rel` (its folder
below the run), `prepared` (its prepared folder below the stage),
`words()` and `to_wire()`.

### 2.3 The run's index — what the job sees

`run-<n>/<label>_<NN>_<stage>-run0.points.json` — one per run, named for
the run's first try whichever try a point is on; schema
`molbuilder/run-points@1`; one module writes and reads it (`runpoints`,
which travels beside the job), one registry parser offers it as an aspect
of the run ([`model/parse.md`](?doc=model/parse.md) § 5). **What the job
saw, never a second record of a point's state**: this machine reads every
point from its own files (§ 5); the job, which has no molbuilder installed,
judges each point by what its entry says of it.

```json
{
  "schema": "molbuilder/run-points@1",
  "stage": "freq",
  "unit": "call", "finished": "done",
  "count": "molbuilder.siesta.vibration_phases:count_words",
  "outcome": "molbuilder.runpoints:outcomes",
  "facts": {"phases": ["relax", "fc", "optical"],
            "selection": {"asked": {"modes": "all"}, "applied": [1, 2, 3, 4], "delta_ang": 0.04}},
  "points": [
    {"rel": "fc", "words": "the force constants", "phase": "fc", "round": "fc",
     "needs": ["relax"], "must_hold": [["CO2.FC", "CO2.FC"]], "verdict_counts": true,
     "planned_by": null, "taken_from": null,
     "started": "2026-10-30T09:12:04", "ended": "2026-10-30T09:31:40", "exit": 0,
     "verdict": {"of": "SCF", "converged": true}, "step_error": null, "not_run": null,
     "outcome": "the force constants: 4 modes, none imaginary; …; 9 optical calls planned",
     "facts": {"name": "fc", "deck_sha256": "…", "coordinates_from": "relax/CO2_01_freq-run0.out, its last geometry step"}},
    {"rel": "optical/f002", "words": "mode 1 (667 cm⁻¹), −δQ", "phase": "optical", "round": "optical",
     "needs": ["fc"], "must_hold": [["CO2.EPSIMG", "CO2.EPSIMG"]], "verdict_counts": true,
     "planned_by": "fc", "taken_from": null,
     "started": null, "ended": null, "exit": null, "verdict": null, "step_error": null, "not_run": null,
     "outcome": null,
     "facts": {"name": "f002", "mode": 1, "side": -1, "frequency_cm1": 667.2, "dq_amu12_ang": 0.181}}
  ],
  "finish": null
}
```

| key | what it says | written by | read by |
|---|---|---|---|
| `stage`, `unit`, `finished`, `count`, `outcome` | the kind table's record (§ 2.1), copied — the job has no kind table | launch | the report steps, the monitor |
| `facts` | the kind's facts about the run (`run_facts`) | launch; a planner's after-step adds its own (the selection) | the kind's words and steps |
| `points` | every point, in the listing order: launch's, then each planned point after its planner's step appends it | launch; a planner's after-step | everyone |
| `rel`, `words`, `phase`, `round`, `facts` | which point, in words, its phase, its round (`entry`), what the kind knows of it | launch; the planner; the kind's own steps add to `facts` (`runpoints.note`) | the rules, the words |
| `needs` | its needs, as folders, of the points this walk runs (`start_of`) | launch; the planner | still to run (§ 4.3), close |
| `must_hold`, `verdict_counts` | what it must hold and whether its verdict counts (`must_hold`, `verdict_counts`) | launch; the planner | done (§ 4.2) |
| `planned_by` | its planner's folder, for a planned point | the planner | launch again (§ 3, op. 2) |
| `taken_from` | the point that computed it, for a point taken over — its folder in the run that computed it (`<NN>_<stage>/run-<n>/<rel>`), one hop; taken from the finished calculation a reuse names, its folder from the projects tree (`reuse.origin`) | launch | ran, the clock, the count, `status` |
| `started` | its first try's start — a try retried in place keeps it | `point started` | the clocks, the pace |
| `ended`, `exit`, `verdict` | its end, its engine's exit — or a before-step's that failed and started no engine — and what its output says it converged (`{of, converged}`, or `null` where it says nothing) | `point ended` | ran, done |
| `step_error` | what its own step could not do, `{what, error}` | the kind's step; `point started` from a before-step's exit, `point ended` from an after-step's | ran; a failure's reason |
| `not_run` | why the ended walk never ran it (§ 4.5) | `point ended`, the run's last | ended, the count |
| `outcome` | what it came to, in its kind's words | its after-step | a round's outcome |
| `finish` | the finish step's `{started, ended, exit}`; `null` where none ran | `point ended`, the run's last | `also_needs`, launch again |

A point **taken over** carries its earlier entry's `facts` and `outcome`
and none of its stamps: it did not run in this run. Every write rewrites
the file whole, atomically.

### 2.4 The folders — the structure, level by level

Every run of points has the same levels. This section is their definition:
what each level is, how it is named and by whom, what record it carries,
what it holds, and the one door that composes it and the one that reads it.

**A transport rung — the device, two frames × two voltages:**

```text
T/                                    the calculation (task.json)
└── 04_device/                        STAGE FOLDER              container
    ├── f000/v0/   f000/v0.2/         PREPARED FOLDERS          container
    ├── f001/v0/   f001/v0.2/           each point's deck and run script, as prep wrote them
    ├── launch/                       LAUNCH FOLDER             container
    │     T_04_device-sweep.run.sh  .sbatch  .log    the walk, its header, its log
    └── run-1/                        RUN                       container
        │ run.json  notified.json  T_04_device-run0.points.json  T_04_device-run0.monitor.log
        │ .gathered-from  .continued-from  the gathered inputs (once, as taken)
        ├── tsgf/                     SCRATCH                   scratch
        ├── f000/                     FRAME FOLDER              container
        │   ├── v0/                   POINT                     run
        │   └── v0.2/                 POINT                     run   (+ shared/  SHARED  shared)
        └── f001/
            ├── v0/                   POINT                     run
            └── v0.2/                 POINT                     run
```

**A SIESTA vibration — relaxation, force constants, optical frames:**

```text
CO2/                                  the calculation (task.json)
└── 01_freq/                          STAGE FOLDER              container
    ├── relax/  fc/  optical/         PREPARED FOLDERS          container
    │     each phase's deck, run script, coordinates.fdf and bundles, as prep wrote them
    ├── launch/                       LAUNCH FOLDER             container
    │     CO2_01_freq-sweep.run.sh  .log
    └── run-1/                        RUN                       container
        │ run.json  notified.json  CO2_01_freq-run0.points.json  CO2_01_freq-run0.monitor.log
        │ CO2.spectra.json  CO2.spectra.nc  CO2.optical.nc        the kind's result, at the run's root
        ├── relax/                    POINT                     run
        ├── fc/                       POINT                     run
        └── optical/                  CONTAINER                 container
            ├── f000/                 PLANNED POINT             run
            ├── f001/                 PLANNED POINT             run
            └── …
```

**The levels**:

| level | what it is | its name — and who names it | its record (`calcdir.json` role) | what it holds | composed by | read by |
|---|---|---|---|---|---|---|
| **stage folder** | one stage of the calculation | `<NN>_<stage>` — `naming.stage_home(base, task, stage).dir` | container | its prepared folders, its runs, `launch/` | prep | `calcdirs.where`: *what* = stage |
| **prepared folder** | where prep writes a point's deck and run script — what every run copies | the point's `prepared`: a transport point's own folder (`f001/v0.2`), a vibration call's phase (`fc`, `optical`) | container | the deck, the run script, the files beside them (`runfiles.beside_the_deck`) | `RunPlace.prepared(rel)`; at prep, `calcdirs.prepared_folder(stage_dir, rel)` | `where`: *what* = point, *point* = its path below the stage |
| **launch folder** | the job that walks the run | `launch` — `runfiles.LAUNCH_DIR`; its files `<stem>-sweep.*`, `runfiles.GroupNames.of_points` | container | the walk script, its header, its log | launch | `where`: *what* = support |
| **run** | one launch of the stage: the run of points | `run-<n>` — `paths.attempt_name(n)`, `paths.attempt_dir` | container | its launch record, its notification record, its index, its monitor log, its gathered inputs, the kind's result — each named by the catalogue (`runfiles`) | the opener (`materialize.open_sweep_run`), at prep for `run-0`, at a launch after | `where`: *what* = attempt, *run_folder* = itself |
| **point** | one engine run of the run | the point's `rel` — a transport point `f<NNN>/v<V>`, or `f<NNN>` where only the frame varies and `v<V>` where only the voltage does (`paths.frame_token`, `task.bias_token`; `stages.Point.rel`), a vibration call its phase (`relax`, `fc`; `vibration_phases.Call.rel`) | run | its own copies of the deck and run script, its output, session log, monitor log, conclusion; its `.continued-from` / `.gathered-from` | `RunPlace.point(rel)` | `where`: *what* = point, *point* = its `rel`, *run_folder* = its run |
| **frame folder** | a transport run's level above its voltages, where both vary — where only the frame varies, the frame's folder is its point | `f<NNN>` — `paths.frame_token` | container | the frame's points | the opener | `where`: *what* = point |
| **container** | a folder below the run for points the job plans | the planned phase's name (`optical`) — `Call(phase).rel` | container | its planned points | the opener, with the run (`PointsOf.containers`) | `where`: *what* = point |
| **planned point** | a point the job adds while the run goes | `<container>/f<NNN>` — `paths.frame_token` of its place in the planning (`Call.rel`) | run | as a point | the planner's own step, through the opener's travelling core (`calcdirs.record`, `planned.bring`) at `RunPlace.point(rel)`; a launch again at the same | `where`: *what* = point |
| **scratch** | the engine's scratch of the run | the rung's `scratch_dir` (`tsgf`) — the warm-files vocabulary, `Job.scratch_dir` | scratch | the round's leads' surface Green's functions | the opener, `RunPlace.scratch(name)` | `where`: *what* = scratch |
| **shared** | a transmission point's energies shared with a finished calculation | `shared` — `transport.energies.SHARED_DIR` | shared | the finished point's files, `.copied-from` | the reuse (`reuse.shared_into`, `carry_shared`) | `where`: *what* = shared |

**Every name has one source.** A folder's name is the door in its row; a
file's name is the catalogue's (`runfiles`: `RunNames.name(role)`,
`compose`, `fixed_name`, its constants) — the index `runpoints.index_path`,
the deck `Job.script`, a phase's products `vibration_phases.must_hold`. No
code spells a folder or a file another way.

**Two halves of one grammar, in one module** (`calcdirs`;
[`project-layout.md`](?doc=execution/project-layout.md) § 1.4a):

- **reading** — `calcdirs.where(folder)` answers which calculation, stage,
  run and point a folder is (*root*, *stage*, *run_folder*, *point*). The
  way from a point up to its run is this answer: the job, the monitor and
  the kind's steps ask it; nothing rendered states it a second time but
  for the engine, which reads only its deck — a device deck names its run's
  scratch by the way up from its point (`stages.Point.to_run`,
  [`engines/transport.md`](?doc=engines/transport.md) § 2a.11).
- **composing** — `calcdirs.RunPlace`, a run's places built from its own
  folder: `RunPlace.of(run_dir)`, or `RunPlace.at(folder)` from any folder
  of the run (through `where`). Every folder a run of points holds or reads
  is composed by it from the point type's own words — which folder (`rel`,
  below the run; `prepared`, below the stage) — and nowhere else:

  | asked | answers | e.g. |
  |---|---|---|
  | `place.point(rel)` | a point's folder below the run — a container's too; the run itself for a point with no level (`""`) | `04_device/run-1/f001/v0.2`, `01_freq/run-1/optical/f003` |
  | `place.prepared(rel)` | a point's prepared folder below the stage (`calcdirs.prepared_folder(stage_dir, rel)` before any run is open, at prep) | `04_device/f001/v0.2`, `01_freq/optical` |
  | `place.scratch(name)` | the run's scratch folder | `04_device/run-1/tsgf` |
  | `place.walked(rel)` | a folder below the run as the walk script names it from the stage folder | `run-1/f001/v0.2`, `run-1/tsgf` |
  | `place.walked_all(rel)` | every folder below a container, as the walk's loop step names them | `run-1/optical/*/` |

  A ``rel`` that is absolute or climbs out (``..``) is refused by name.

```python
from molbuilder import calcdirs
place = calcdirs.RunPlace.of(stage_dir / "run-1")
place.point("f001/v0.2")            # .../04_device/run-1/f001/v0.2
place.prepared("f001/v0.2")         # .../04_device/f001/v0.2
place.walked("f001/v0.2")           # "run-1/f001/v0.2"
w = calcdirs.where(place.point("f001/v0.2"))
(w.run_folder, w.point)             # (.../04_device/run-1, "f001/v0.2")
```

**The two halves are inverses**: of every point folder `RunPlace`
composes, `where` reads back the same run (*run_folder*) and the same point
(*point*) — checked on the real runs of both kinds (their e2e).

---

## 3. The operations

In the order of a run's life. Each is done by one door, on one side.

| # | operation | where · when | reads | writes | door |
|---|---|---|---|---|---|
| 1 | **prepare the points** | this machine · `prep` | the declaration's prepared points; the record's `steps` | each prepared point's deck and run script — its report steps, and the kind's steps where it has them — in its prepared folder; the run's first folder, its points and containers opened | the conductor (`prep`), `materialize.open_sweep_run` |
| 2 | **plan the walk** | this machine · `launch`, before the question | the stage's newest launched run: each point by the done rule, its index (the points it took over, the points each planner planned) | nothing: the plan — which points are taken over, which are walked, each walked point's start and needs, the rounds' order, the loop steps | `submit._points_walk` |
| 3 | **take a point over** | launch · the send | the done point's folder | its copy but the engine's scratch, and its record naming the run that computed it | `submit._take_over` → `materialize.hand_over` |
| 4 | **continue a point** | launch · the send | its last run's folder | its job's warm files, and its record | `materialize.hand_over` |
| 5 | **hand a start** | launch records it; the walk copies it | the source's files | the record now; the files just before the point runs — a planned point's, by its loop step, with no record | `materialize.hand_over` (`walk=True`), the walk |
| 6 | **open a planned point again** | launch · the send | its prepared folder; the inputs its planning wrote in the last run | its folder, its deck and those inputs — never the last run's outputs | `materialize.open_run`, `bring_files` |
| 7 | **write the index** | launch · the send | the declaration's entries, the kind table's record | the index | `runpoints.empty_index`, `add_point` |
| 8 | **walk** | the job | the walk script | the walk's log | `submit._walk_script` |
| 9 | **a point started** | the job · before the engine | the index | `started`; the before-step's work; the run's `start`, once | `mb_monitor.pyz point started` |
| 10 | **a point ended** | the job · after the engine | the index; the point's output | `ended`, `exit`, `verdict`; the after-step's work and `step_error`; the run's end — `not_run`, the finish step, `finish` — and its events | `mb_monitor.pyz point ended` |
| 11 | **plan points** | the job · a planner's after-step | what its point computed | each planned point's folder, its inputs, its entry | the kind's after-step, through `runpoints.add_point` and the opener's travelling core (`calcdirs.record`, `planned.bring`) |
| 12 | **judge a point** | both sides | the job: its entry and its files; this machine: its own files | — | `runpoints.rows_of`; `continuation.point_done` |
| 13 | **the run's state** | both sides | the points judged; this machine: the jobs launched from it | — | `runpoints.run_state`; `continuation.points_state` |
| 14 | **count** | both sides | the points judged, the index's `facts` | — | the record's `count` |
| 15 | **say where the run is** | the job · each point's monitor | the index | its status line: the count, this point, the clocks, the pace, what is left | `monitor` |
| 16 | **read a run of points** | this machine · `status`, Task setup, the Results ladder, the transport record, the run door | each point's own files; the run's launch record; the index's `taken_from`, planned points and `finish` | — | `continuation.point_rows`, `points_state`; `runstatus`; `runs` |
| 17 | **refuse a warm launch with nothing to run** | launch | every point done, and the run finished (§ 4.4) | the refusal, naming `--cold` | `submit._all_done` |
| 18 | **rewrite an older index** | this machine · `jobset migrate` | a run's index written by molbuilder 2.54 (`.phases.json`, `.points.json` of `transport-points@1`) | the run's index | `jobset.migrate` |

**Plan the walk (2), in full.** The points are the run's opened points and,
warm, each point the newest launched run's index lists as **planned by** a
point this launch takes over — a planner's points belong to its
computation: taken over with it, and planned anew when it runs again.
Warm, every point done in the newest launched run is taken over (3); the
rest are walked — a planned point opened again (6), a point whose relaunch
is *continue* given its warm files (4), every walked point its start (5).
Cold, nothing is taken over. A first launch of a calculation that reuses a
finished one takes that one's points over by the reuse's own rule
([`engines/transport.md`](?doc=engines/transport.md) § 2a.16). The walk
runs round by round — the rounds in the order the listing first meets them,
each round's points in the listing order — a step for each walked prepared
point, needing its walked needs, and a loop step for each container whose
planned points or planner this walk runs; where the rung keeps the engine's
scratch in a folder of the run's own, the walk empties it as each round
begins and when it ends.

**A point started (9), in full.** Stamp its start — kept from the first
try. Run its before-step, where its kind has one: **its failure is the
point's** — `step_error` stamped, the verb exiting with its code, the
engine not started, and the run script running `point ended` with it. Send the run's `start` unless the
run's notification record holds one.

**A point ended (10), in full.** Stamp its end, its engine's exit and its
verdict, read from its output. Run its after-step, where its kind has one:
the step reads its point already stamped — so it asks the one rule whether
its point is done — and, after a clean engine, **its failure is the
point's**: `step_error` stamped where the step wrote none, the verb exiting
with its code, the conclusion naming the failed step; after a failed
engine, its failure is logged and the conclusion is the engine's. Then, by
the rules of § 4: no point left to run — close the index (§ 4.5), run the
finish step where the kind has one (stamping `finish`; its failure makes
the run failed, never the point), and send `finish` with the run's state;
otherwise a `milestone` or a `part_failed`, or nothing.

**Refuse a warm launch with nothing to run (17), in full.** Every point
done in the newest launched run: refused, naming `--cold` — unless the run
is not finished by its state's one rule (§ 4.4: its finish step failed, or
it needs more than its points, `also_needs`, a vibration's result): then
its last point runs again, and its report step finishes the run.

**The report steps never fail a point** for their own sake: an index that
does not read or a message that is not sent is logged in the point's
session log. Only the kind's own steps fail it, as they always have.

---

## 4. The rules

One rule each, asked on both sides by the same pieces.

### 4.1 Ran

A stamped point ran when its engine's exit is 0 and it has no
`step_error`; a point taken over ran; a point the walk never ran did not.
On this machine: its run finished by the one status door — its conclusion
marker's code 0 and no stop in its output
([`job-system.md`](?doc=execution/job-system.md) § 5.4, *What finished
means*). The conclusion's code is the engine's, or the failed step's after
a clean engine, so the two sides agree.

### 4.2 Done

It ran, and `_run_ending.not_done_why(verdict, counts, missing)` has
nothing to say: its verdict converged where it counts, and it holds every
`must_hold` file. Why not, in that function's words — *finished, not
converged (SCF NO)*, *finished without CO2.FC* — or, a point that did not
run, its exit and its step's failure: *exit 1*, *the optical calls'
record was not written: …*. A point taken over is done when it holds
what it must.

### 4.3 Still to run

Not ended, and for each need: up its chain, the first that has ended ran.
So a device voltage that fails ends the chain that starts from it, and the
walk goes on with every other frame; a relaxation that fails leaves the
force constants unrun. The walk script asks the same of its steps (each
step's `needs`).

### 4.4 The run's state

On the job, once no point is left to run: `finished` — **done**: every
point done; **ran**: every point ran (`_run_ending.points_state`) — else
`failed`; its words the first point not done in the listing order, and *k
of N <unit>s not done* when more than one is. A finish step that exited
non-zero makes the run failed, *the finish exited N* (`runpoints.run_state`).
On this machine the same rule over its rows (`continuation.points_verdict`
asks `run_state`), with what only this machine can see — *queued*, *running*, *between
two points*, a walk that ended among its points
(`continuation.points_state`; [`job-system.md`](?doc=execution/job-system.md),
*What marks each state*).

### 4.5 A point the walk never ran

Once no point is left to run, each point not ended is stamped `not_run`:
*it builds on <words>, which failed* — the first need up its chain that
failed — or *the walk ended before it reached this point*
(`runpoints.why_not_run`). `status` says the same of a point the ended walk
never reached, asking that rule over its rows, which carry each point's
`needs` from the index (`continuation.never_reached`); both are
**failed**.

### 4.6 What a point's end says

After `point ended` has stamped the point and run its after-step:

| the point | points left to run | sent |
|---|---|---|
| any | none | `finish`: the run's state (§ 4.4) and its words; and the outcome of this point's round |
| its round has no point left to run | yes | `milestone`: the round's outcome (the record's `outcome`), else *<words>: done* — *<words>: not done -- <why>* for a point not done |
| not done | yes | `part_failed`: *<words>: not done -- <why>*, once per **reason** — its exit where not 0, else its step's `what`, else the rule's words |
| any other | yes | nothing — the count is the periodic report's |

Every event carries the run's count and, once a point has started, the
run's clock ([`run-reports.md`](?doc=execution/run-reports.md) § 2.4,
§ 2.8).

### 4.7 The count and the pace

The count is the record's `count` part over the points judged, and the
index's `facts`: one shape of row on both sides — the point's facts and its
judgement. The pace is a phase's: the points of the phase underway that
this run computed — that ran here, never one taken over — and the points of that phase
still to run; said in the unit, with the phase before it where the run has
more than one phase (*each optical call takes 2h*, *each point takes 40
min*).

---

## 5. What is read, and by which reader

Every file has one reader; every side asks it.

| what | file | reader | who asks |
|---|---|---|---|
| the run's index | `<stem>-run0.points.json` | `runpoints.read_index` — another schema refused by name, an index of 2.54 naming `jobset migrate`; the registry parser `run-points-json` | the report steps, the kind's steps, the monitor; launch (the newest launched run's); `status` (taken over, planned points, the finish) |
| where a point is | `calcdir.json` and the folders' names | `calcdirs.where` (its composing half, `calcdirs.RunPlace`, places every folder this table's readers are handed) | the report steps, the kind's steps, the monitor; the run door |
| how a point's run ended | its conclusion marker | `parse.dirs.run_status` (with the run's launch record) | `status` (`continuation.point_rows`) |
| what its output says it converged | its output | `_run_ending.verdict_of` | `point ended` (stamped); `status` (through `run_status`) |
| what it holds | its folder | `_run_ending.holds` | both sides |
| what it started from | `.continued-from` | `runrecord.read_continued_from` | `status`; launch (taking over, one hop) |
| what the run gathered | `.gathered-from` | `runrecord.read_gathered_from` | `status`, the record |
| the run's launch | `run.json` | `runrecord.launch_record` | `status`; each point's monitor (the wall) |
| what the run has said | `notified.json` | `runrecord.read_notified` | the `notify` verb; each point's monitor |
| a round's outcome | the round's points' files, or their stamped outcomes | the record's `outcome` part | `point ended` |

---

## 6. The doors

| door | floor | what it offers |
|---|---|---|
| `runpoints` | 1, travels in `mb_monitor.pyz` and every bundle whose steps write the index | the index: `index_path`, `index_of`, `read_index`, `index_at`, `write_index`, `empty_index`, `add_point`, `entry_of`, `planned_by`, `taken_from`; the stamps: `stamp_start`, `stamp_end`, `stamp_step_error`, `stamp_finish`, and a kind's own step's facts and outcome, `note`; the rules: `rows_of`, `still_to_run`, `why_not_run`, `close`, `run_state`, `round_ended`, `said`, `reason`; the words: `count`, `round_outcome`, `outcomes`, `not_done_words`, `NEVER_REACHED`; a part resolved beside the job: `part` |
| `kind_table` | 1 | `run_of_points(kind, engine)` — the record of § 2.1; a part resolved on this machine, `part` |
| `calcdirs` | 1, travels | the folders (§ 2.4): `where` reads which run and which point a folder is; `RunPlace` composes them — `point`, `prepared`, `scratch`, `walked`, `walked_all`; `prepared_folder` at prep |
| `jobset.points` | 4 | `PointsOf`, `points_of(task)`, `record_of(task)`, `rung_points_of`, `walls_of` |
| `jobset.submit` | 4 | `_points_walk` — the one walk; `_take_over`, `_all_done`, `_walk_script` |
| `jobset.continuation` | 4 | `point_done`, `point_rows`, `points_state`, `points_verdict`, `never_reached` |
| `monitor` | travels | `point started`, `point ended`, `notify`; a member's line |
| `runwrap` | 3 | each point's run script: its monitor a member, its report steps, the kind's steps as their arguments — rendered by SIESTA's launch block, the engine every run of points runs on |

---

## 7. A new kind of run of points

1. the pair's record in the kind table (§ 2.1);
2. its declaration (§ 2.2) and its point type, which answers `rel` and
   `prepared` -- which folder, below the run and below the stage; where
   that is, `calcdirs.RunPlace` composes (§ 2.4) -- `words()` and
   `to_wire()`;
3. its words — `count`, and `outcome`: its own where its rounds say more
   than their points' outcomes, else `molbuilder.runpoints:outcomes` — in
   a module that travels beside the job (listed in the monitor bundle's
   companions);
4. where the kind works between its points, its steps: verbs of its job's
   bundle — `<before> <deck>`, `<after> <deck> <output> --exit <rc>`,
   `<finish> <deck>` — named in its record; each finds its point by its
   folder (`calcdirs.where`) and the point's entry in the index;
5. its pair on the catalogue's index row (`runfiles.ROLE_POINTS`,
   `calculation`), and an engine whose launch block renders the report
   steps (SIESTA's);
6. its pair among those whose member monitors state the run-of-points
   report fields a description may ask (`report_fields.FIELDS`:
   `call_elapsed_s`, `units_left`, `left_at_pace_s`).

Nothing in the walk, the index, the report steps, `status` or the monitor
changes.

---

## 8. Each kind's own

- **A transport rung** ([`engines/transport.md`](?doc=engines/transport.md)
  § 2a.11): its points — a frame of the cited set and a voltage; a round,
  one voltage at every frame; the leads' surface Green's functions in the
  run's `tsgf/`, emptied as each round begins; a point that shares energies
  with a finished calculation's point (§ 2a.16).
- **A SIESTA vibration** ([`engines/vibration.md`](?doc=engines/vibration.md)
  § 5.11): its phases — the relaxation, the force constants, the optical
  frames — and its `PHASES` table; its steps: the coordinates, the analysis,
  the frames planned, the result and the data file.
