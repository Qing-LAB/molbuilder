# The job request — how and where a calculation runs

**Role:** contract
**Domain:** execution
**Companions:** [`engines/stages.md`](?doc=engines/stages.md) — the
description, what a calculation computes; [`engines/template.md`](?doc=engines/template.md)
— every parameter, and which of them are run settings;
[`execution/job-system.md`](?doc=execution/job-system.md) — `prep` and
`launch`, which read this record; [`execution/architecture.md`](?doc=execution/architecture.md)
§ 5.2 — where each launch value is fixed; [`web/task-setup.md`](?doc=web/task-setup.md)
— the cards that write it.

*(The user, 2026-10-20: "it is completely stupid to have this involved in a
job scheduling"; "carefully evaluate the jobset logic and see how much mixing
you already have between science request vs job request"; and on 2026-10-21
the two choices § 4 and § 5 record.)*

**This contract owns:** what a run setting is, the job request's file and
every key in it, who writes it and who reads it, what of a launched stage may
change, and when two decks are the same calculation.

---

## 1. Two records, two editors

> **A calculation is two records with two editors. The description says what
> is computed; the job request says how and where it runs. Nothing in one is
> read as the other.**

| | **the description** | **the job request** |
|---|---|---|
| its files | `task.json` and `<label>.template.toml` ([`stages.md`](?doc=engines/stages.md) § 6, [`template.md`](?doc=engines/template.md)) | `job-request.json`, beside them |
| it says | what the calculation is — its structure or the junction it cites, its kind, its stages and what each varies, its bias points, every parameter's value | how and where it runs — the queue, the wall, the memory; each run's ranks, cores, GPUs and run settings; which stages share one job; what a benchmark measures; when the calculation reports, and to whom |
| written by | Task setup's stage table and the parameter tabs' hand-over; `molbuilder describe`; the Transport tab's describe door; a person | Task setup's run cards, group numbers, benchmark and report cards; `molbuilder describe`, which writes each stage's group; a person |
| read by | prep's resolve — every deck; the transport gather; status and the Results tab | prep's assembly — each job's launch values, header and run script, the groups; launch, for its groups — never for a launch value (§ 7); the transport gather, rendering a rung's deck now; the continuation door; status, summarize, `jobset migrate` and the Task setup cards |
| it must never | say how or where the calculation runs | state anything the calculation computes |

**`shape` stays in the description.** It is how the folder is laid out, fixed
once the calculation has produced ([`stages.md`](?doc=engines/stages.md)
§ 6.7) — neither what is computed nor a request that may change.

**What "nothing in one is read as the other" forbids, and what it does not.**
A reader of the job request reads the description to know which stages there
are and what each builds on — to check that no stage shares a job with one it
builds on (§ 3.3): that reads the science as the science. What it forbids is
taking a value of one record for a value of the other: a template's `use_gpu`
asking the scheduler for GPUs while the run card says nothing, a deck comment
standing in for the rank count a job is launched with, a run setting making
two decks two calculations.

**Why two files, saved apart.** Each record changes at its own moments and has
its own editor. A queue, a wall, more memory for a run that ran out of it — a
job-request edit, which changes no deck's science; a mesh cutoff — a
description edit. A check of what is computed reads the description and the
decks' engine input alone (§ 6), so a change of how a run runs is never read as
another calculation: a finished device run stays the one its transmission
takes when only a run card moved, and a launched stage takes more memory
without going back past its runs (§ 5).

## 2. Which settings are run settings — the catalogue says

**The catalogue's `execution` category is the run settings**: what changes how
fast and where the calculation runs, and never its answer
([`template.md`](?doc=engines/template.md) § 6.2). *Never its answer* means
within the description's own tolerances, not bit for bit: a run setting moves
the order of floating-point sums, where data lies and how many processors
share it, and an iterative solve — the SCF, a relaxation, a response — turns
that into differences inside the tolerances it converges to. They are of two
kinds:

| | the items | stated nowhere |
|---|---|---|
| **the launch shape and the memory cap** — the machine's answers (`allocation = true`) | `mpi_np`, `omp_threads`, PySCF's `threads`, `gpu_count`, `max_memory_mb` | the ranks and the cores per rank are refused at prep, and a GPU run's count; a memory cap left out is no cap ([`architecture.md`](?doc=execution/architecture.md) § 5.2) |
| **the speed settings** | `use_gpu`, `diag_algorithm`, `block_size`, `parallel_over_k`, `ts_solution_method`, `ts_elecs_out_of_core`, `wall_slack`, `continue_retries` | the catalogue's default (§ 4) |

**A run setting lives in the job request and nowhere else.** The catalogue
declares it; no template carries it, valueless or not — a template holding one
is refused by its reader, naming `molbuilder jobset migrate` (§ 8), which takes
it out, a value it holds onto the calculation's run card
([`template.md`](?doc=engines/template.md) § 2: *a home that must stay empty is
removed, never policed*); a stage's `overrides` never names one
([`stages.md`](?doc=engines/stages.md) § 6.2). **Its default is the
catalogue's `default`**, a kind's `recommended` first — the one answer
`template.default_of` gives every reader ([`template.md`](?doc=engines/template.md)
§ 5.4); a run setting states no `value` in the catalogue, since no template
starts from one. Every run setting is in the catalogue's `staging` group —
`diag_algorithm`, `block_size`, `parallel_over_k` and `ts_solution_method`
come there from `budget`, `ts_elecs_out_of_core` from `stage` — which the
parameter tabs do not offer and Task setup's run card shows
([`stages.md`](?doc=engines/stages.md) § 1.3;
[`task-setup.md`](?doc=web/task-setup.md) § 6.1), so the hand-over and
`molbuilder describe` write a template with none.

**Each holds under a condition the rest of molbuilder keeps:**

- **`diag_algorithm`** — every solver offered computes the whole spectrum:
  molbuilder never writes `NumberOfEigenStates`, which ELPA, MRRR and Expert
  would honour, and a reduced count changes the physics.
- **SIESTA's `use_gpu`** runs the deck in the GPU-built environment: the same
  calculation while that build is the same SIESTA release as the packaged one
  — the recipes build both from one release unless `MOLBUILDER_SIESTA_TAG`
  names another for the source build
  ([`ops/env-framework.md`](?doc=ops/env-framework.md)).
- **`ts_elecs_out_of_core`** — the electrode Green's-function file is the
  device run's own, written fresh for each run and each voltage
  ([`engines/transport.md`](?doc=engines/transport.md) § 2a.11); TranSIESTA's check of
  a file it reuses compares the electrode's geometry and its energy points,
  never its Hamiltonian.

**An engine library's own limits are said, never counted or refused** *(the
user, 2026-10-21: "the user should be aware of it … a hint and information is
enough")*. PySCF's `use_gpu` runs the mean field in gpu4pyscf, which differs
from PySCF in two places that touch an answer: it has no stability analysis,
so an unrestricted reference's check (UHF, UKS) is not made on the GPU and the
run keeps its first SCF solution, its log saying `stability: NOT CHECKED`
([`pyscf.md`](?doc=engines/pyscf.md) § 7.3); and its solvent model has not
been compared with PySCF's here. Choosing the GPU knowing that is the
person's: the run card's GPU switch says it, beside the switch, for a PySCF
calculation ([`task-setup.md`](?doc=web/task-setup.md) § 6.2), and prep says it
among each deck's own findings — the PySCF settings gate's, a `warn`, never an
`error` — for a run on the GPU whose reference is unrestricted or that carries
a solvent model. `use_gpu` stays a run setting, as on SIESTA.

**A setting that moves an iterative solve's path is the description's**: the
mixing, its history, the initial guess, a step's size. Where a system has
several SCF solutions or several minima, the path decides which one is
reached. **So is a budget the engine's own run counts** — `relax_steps`,
`max_scf_iter`, PySCF's `geom_continue_retries`, the deck's own loop: it
decides where an unconverged run stops, and what a later stage takes from it.
**A stop that launching again resumes is the run's**: the wall, its slack, and
SIESTA's `continue_retries`, the run script launching the same deck again,
warm, as a launch again does
([`running-a-job.md`](?doc=execution/running-a-job.md) § 3.5). **And a
keyword that only shapes what a run writes is the description's** — a
`write_*` switch, a verbosity: changing one on a launched stage is a change of
the description, as any other (§ 5).

**Two items leave the category, because each changes what is computed:**

- **`density_fit`** — a PySCF run's density fitting moves its energy (its own
  help: up to 0.1 kcal/mol): category `method`.
- **`restart`** — whether a stage starts from what the stage before it left is
  what the ladder computes: a stage's own science, set as its override
  ([`stages.md`](?doc=engines/stages.md) § 2), category `procedure`. Which
  stages may share a job is read from it (§ 3.3), and so are the ladder's
  sequence checks — from the template and the stage's overrides, never a run
  card.

**A run setting the deck writes stays in the deck.** A SIESTA deck writes
`Diag.Algorithm`, `BlockSize`, `Diag.ParallelOverK`, `Diag.ELPA.GPU`,
`TS.SolutionMethod` and `TS.Elecs.Out-of-core`; a PySCF script states its
run settings — `THREADS`, `USE_GPU`, `GPU_COUNT`, `MAX_MEMORY_MB`, `WALL_SLACK` — in its run-settings block, at its head,
and every other line reads them there by name, its effective-parameters table
included
([`job-contracts.md`](?doc=execution/job-contracts.md) § 3.1). So a deck is
the whole record of what its run computed, and with what; which of its lines
are the calculation is § 6's.

## 3. `job-request.json`

```jsonc
{
  "schema": "molbuilder/job-request@1",

  // what this calculation asks the scheduler for (§ 3.1)
  "allocation": { "domain": "htc", "time": "0-04:00:00", "mem": "128G" },

  // the run card every rung runs at, unless its own says otherwise (§ 3.2)
  "execution": { "mpi_np": 8, "omp_threads": 2,
                 "diag_algorithm": "ELPA-2STAGE" },

  // every stage of the description: the job it runs in (§ 3.3), its own card
  "stages": {
    "coarse": { "group": 1 },
    "tight":  { "group": 2,
                "execution": { "mpi_np": 16, "time": "1-00:00:00" } }
  },

  // what to measure (§ 3.4) and when to report (§ 3.5)
  "bench":  { "mpi_np": [4, 8, 16], "omp_threads": [1, 2] },
  "notify": { "on_scf_converged": true, "every_hours": 6,
              "channels": ["slack"], "report": ["*"] }
}
```

**`schema` is `molbuilder/job-request@1`**, checked by name and major
([`job-contracts.md`](?doc=execution/job-contracts.md) § 6.1). **`stages` holds
every stage the description holds, by its name, and no other**: each entry's
`group` is required, its `execution` optional. A name is matched as every stage
name is, case aside (`identity.stage_key`); two keys that are one name, an
entry for a stage the description does not hold, and a stage of the
description with no entry are each refused by the reader, naming both files —
one Save writes both (§ 7), so a difference is an edit of one without the
other. A description with no job request beside it is refused when the
calculation is opened (§ 7). **The rest is optional, and absent is a state**: no allocation, no
calculation-wide card, no benchmark plan, no reports.

**It names no machine.** A queue's name is the person's ask, true of this
calculation wherever the folder is opened (§ 3.1); the target's record checks
it at prep. What a machine *found* — a benchmark's verdict — is never written
here (§ 3.4).

### 3.1 `allocation` — what this calculation asks the scheduler for

*(User, 2026-08-24.)* Four optional fields, and absent means
**unstated**:

```json
"allocation": { "domain": "htc", "time": "0-04:00:00", "mem": "128G" }
```

| field | what it says |
|---|---|
| `domain` | which **queue** — the same answer `--domain` gives. Stated nowhere — here, the run card, `--domain` — on a target with a scheduler, prep refuses |
| `time` | the **wall**. Stated nowhere — here, the run card, `--time` — on a target with a scheduler, prep refuses ([`architecture.md` § 5.2](?doc=execution/architecture.md)) |
| `mem` | the **memory ask**. Stated nowhere — here, `--mem` — on a target with a scheduler, prep refuses |
| `gpu_binding` | whether a GPU ask carries `--gres-flags=enforce-binding` — the job's cores kept beside its GPUs. Unstated, it does; `false` sends the GPU ask without it, for the benchmark and the runs alike, so a benchmark measures the layout the run will use ([`gpu.md`](?doc=execution/gpu.md) G9; user, 2026-10-01) |

**Why it is here and not in `bench`.** § 3.4's rule still holds — the file
records what a person *asked*, never what a machine *found* — and these are
asks. `bench` says *which settings to measure*; this says *what to request*.
A queue name is portable in the same way `"measure ranks at 4, 8, 16"` is: it
is a decision about this calculation, true wherever the file is opened, and
§ 3.4's own example of the un-portable thing (*"use 16"*) is a different
kind of statement.

**One spelling, and it is SLURM's** — `"0-04:00:00"`, `"128G"` *(user,
2026-08-24: "your record should set unified time format while it is the UI
that can do some translation for human readability/input")*. Not seconds and
gigabytes either: what the file holds is what `sbatch` takes.

The reason is not a matter of taste. **The two
vocabularies disagree**: `04:30` is four minutes thirty to SLURM and four and
a half hours to a person, so a field holding *whichever spelling arrived*
cannot be read correctly by anybody. And with nothing converting between this
key and `Resources.time`, which is SLURM's own, the browser's `"4h"` travelled
unread as far as the command line, where `sbatch` refused `-t 4h`: the tool's
own written value, rejected by the tool.

**Translation lives at the edges where humans are** — the Task-setup tab's
input box, `--time`/`--mem`, and a file typed by hand, which is such an edge
too. All of them go through one door (`scheduler/quantities.py`'s `canonical_time` /
`canonical_mem`), and the reader below normalises on the way in, so a
person may still type `4h` anywhere. What no layer behind those edges ever meets is a second
spelling. `Resources` enforces the same invariant on itself, because four
roads reach it and a fix at one would leave three.

**The queue name is not judged here**: a job request written for one cluster
is opened on another,
and refusing it here for naming a queue this box never heard of would refuse
a file that is correct where it is going. The difference is that a spelling
is machine-independent and a queue name is not — so the spelling can be
settled at the door and the queue cannot. The target's record answers that,
at prep, where the machine is known.

**`prep` reads it as the base allocation, and a flag still wins — FIELD by
field.** `--mem 64G` overrides the memory and leaves the queue and the wall
standing; `--np 8` overrides neither. Whole-object precedence would let an
unrelated flag silently drop an ask nobody mentioned, which is the class of
loss this key exists to end: five Sol jobs (62039301–05, 2026-08-23) died
against a per-GPU memory default and an invented wall, because the two
numbers had no home that travelled with the calculation.

**Absent-is-a-state**, like `bench`: nothing asked writes no key.

### 3.1a The memory a run needs — measured, shown beside the ask *(the user, 2026-10-20 and 2026-10-22; built 2.29.0 — plan § 1u)*

*(The user, 2026-10-20: "can we have some UI feature to give estimate based
on the system/task? … estimate the memory need … as a suggested value, and
task setup … would put it next to the request"; 2026-10-22: "no need to
refine memory calculation … we don't want to misalign and allocate less
memory than actually needed and one job at a node/task is good enough to
know, document the limitation, and that's it".)*

**The ask stays the person's.** Nothing fills `allocation.mem`; no prep
refuses an ask for being under the need; the ask and the need are shown side
by side and nothing compares them in words
([`running-a-job.md`](?doc=execution/running-a-job.md) § 5.3.1). **Beside the
ask, each stage's need is shown, measured on the runs of that stage** —
computed when asked and stored nowhere: it follows the runs and the ranks, so
a stored copy would be a second source, stale when either changes, and one key
could not hold a ladder whose rungs each need their own.

**One door, the memory a stage needs** —
`jobset.memory.need(calc, stage, ranks)` → the stage's need at `ranks`, the
run it was measured on, and the words, `line`; `jobset.memory.largest` of a
job's stages' answers the job's (below). `ranks` is what the asker resolved:
prep, the job's own, its `--np` over the card; Task setup, the card as
edited — a card stating none, or a value the setting door refuses, gets the
measurement alone. A stage the calculation does not hold, or a folder that
holds no calculation, has nothing measured. Prep and Task setup print the
door's words, so the two say one sentence, under one name, *memory need*
(`need` on the wire).

**What a run held is the memory peak its run record states** — the one peak
the instruments' door chooses, `mem_peak_gb`, as it is measured today
([`model/parse.md`](?doc=model/parse.md) § 5c.1) — at the ranks its wrapper
launched (the record's `ranks_asked`), its threads named beside them. The
engine's own count lies in its output, which this door does not parse; a
build that ran fewer processes than it was launched with only errs the bound
high. A PySCF run is one process, and so is read a run whose wrapper states no
ranks. **What the figure is, where it errs, and how it is said:**

- **Under a scheduler it is the job's**: the kernel's peak for the job's
  cgroup since the job began. A job that runs several stages or points one
  after another — a group, a run of points, a sequence of groups — gives a
  later one's figure the largest of those before it too; and the cgroup is
  charged the file cache a run's writes leave (a device's leads'
  Green's-function files are tens of GB). So a figure may be more than its run
  held, never less.
- **A run here is its own processes'**, sampled every 10 s, a row written when
  the memory moves by 10 % or more: its figure can be under its true peak by a
  spike shorter than a sample, or by up to that tenth.

The figure is said with where it was read (`mem_peak_from`: the kernel's
counter, or the largest sample), and its bound in a word that says which:
**at most** from the kernel's counter, which errs high; **about** from the
largest sample, which may be under a run's true peak.

**The bound — a finished run that held X at N ranks needs at most X ·
max(1, M/N) at M ranks.** An engine that gives every rank its own copy of the
matrices — TBtrans, and TranSIESTA's solver, which deals the energy points out
among the ranks — holds about M/N times as much at M; one that divides its
matrices among the ranks holds less a rank as ranks grow; and with fewer ranks
than measured neither holds more than it did. So one rule bounds both,
scaling up and never down — said *at most* or *about* by where the peak was
read (above). Threads are
not modelled. A PySCF run's figure is what it held, never scaled; its own cap
(`max_memory_mb`) is what it was let hold.

**Which runs, and which figure**: the stage's runs and each point of a run
of points, by the lister the Results tab's every run asks
(`runstatus.run_folders`), and a flat stage's numbered runs, by their launch
records (`runrecord.launched_runs`). Every bound of a
point's finished runs holds, so **a point's need is the smallest of them**;
**a stage's need is the largest of its points'**, naming the run that gave
it, as every record of the calculation names a run (`runrecord.RunRef`: its
folder, or a flat stage's run by its name). **A run that did not finish**
gives no bound — what it held is less than the stage needs — and the one that
held the most is said beside the figure, with how many give none: one still
going as what it *holds so far*, one that ended as what it *held before it
ended*, in the status door's words.

```text
transmission  about 49.3G at 64 ranks -- held 12.31G at 16 ranks × 1 thread, 05_transmission/run-0 on qlabsrv (the largest sample; finished)
device        at most 9.6G at 32 ranks -- held 9.58G at 32 ranks × 1 thread, 04_device/run-1/v0.2 on sg013 (the kernel's counter, the job's; finished)
seed          nothing measured -- no finished run of seed has stated its memory
```

**The figure is said in tenths of a GB, rounded up**, what a run held in
hundredths as its record states it.

- **The figures are the scheduler's units and spelling**, 1024-based, as
  `--mem` asks (`48G`) and as the monitor divides.
- **It is the host's memory**: a GPU's own is not in it.
- **Nothing measured says so**, and nothing is invented in its place: a
  stage's first finished run is what measures it. A calculation's first run
  of a stage has no figure until the engine's own analysis answers — plan
  § 1u (b), not built.

**A job's need is the largest of what it runs**: a group of several stages,
run one after another in one allocation (§ 3.3), the largest of its members';
a sequence of groups in one job, the largest of its groups' — said with the
stage it is, and the stages nothing has measured named after it. Every job asks the
calculation's `allocation.mem`, so the figure beside the ask is the largest
need of the stages, naming the stage; `prep --mem` asks for one prep's jobs
alone.

**What it reads, and when** — for each run, its peak and where it was read,
its machine and the ranks and threads its wrapper launched: the record's rows
that answer those alone, through the run door (`runs.record_of`, its
`fields`), never a run's whole record, and never parsing its engine's
output. The page asks
it in a request of its own once the plan has painted, when the folder or a
rank count changes — never on every edit; an older answer, overtaken, is
dropped.

**Where it is shown, and nowhere else** — never in a header, a deck, the job
request or a refusal:

- **Task setup** — under the memory field of the queue card, after what the
  queue's machines hold, the largest need and its stage
  ([`task-setup.md`](?doc=web/task-setup.md) § 6.1); each stage's line in
  *What a prep will write, per stage* (§ 7.1 there); and prep's answer in the
  Preview (§ 11.1 there). The page asks the prep-plan route, which opens the
  calculation from its folder — a folder that is no calculation yet has
  nothing measured ([`web-api.md`](?doc=web/web-api.md)). It reads what has
  run as the page reads it elsewhere
  ([`task-setup.md`](?doc=web/task-setup.md) § 10).
- **prep** — each stage's answer carries its line (`need`), printed after its
  placement and fit, or after the line stating where its values came from on a
  machine with no queue; a group's beside the group's fit, a sequence's
  beside its own ([`job-system.md`](?doc=execution/job-system.md) § 5.3, *One
  prep, two doors*). A benchmark's prep shows none: what its trials held is
  `summarize bench`'s to report.

### 3.2 `execution` — the ONE condition the run uses

*(User, 2026-09-02: "run parameter is independent from bench grid or bench
result. User can run without bench… bench and run share the same framework to
choose/set parameters but run does not produce grid but a single user defined
condition to run." And: "bench is bench and run is run. They are structurally
separated in their dir and they are functionally different as the user decides
to use either.")*

```json
"execution": { "mpi_np": 8, "omp_threads": 2, "diag_algorithm": "ELPA-2STAGE" }
```

**One value per parameter, never a list.** A list is `bench`'s shape, and a
list here is refused by name with that said — the two blocks are the same
vocabulary at different arity, so the arity is what tells them apart.

| | | |
|---|---|---|
| **`bench`** | several values per axis | *measure these* → N trials, in `<stage>/bench/` |
| **`execution`** | one value per parameter | *use this* → one run, in `<stage>/run-N/` |

**Independent, and both optional.** Declare a bench and no run condition, a
run condition and no bench, both, or neither. **A run never requires a
benchmark** — not to have been executed, not to have been declared. The two
lanes have separate directories (`project-layout.md` § 1.5b) because they are
separate things, and the file says so too.

**Both calculation-wide and per stage.** The top-level block is what this
calculation runs at; a rung that wants something else says so on itself, and
the two compose **field by field** — a stage naming only `mpi_np` keeps the
calculation's solver and its thread count:

```json
"stages": { "tight": { "group": 2, "execution": { "mpi_np": 16 } } }
```

**A calculation-wide value reaches the rungs that read it** — the item's
`stages`, by role, asked of `template.reads` ([`template.md`](?doc=engines/template.md)
§ 6.4) as for an override. A rung whose role does not read an item takes
nothing of it from the calculation's block: a transport ladder's
transmission takes no `use_gpu` or `gpu_count`, its `tbtrans` having no GPU
path, while the seed, the leads and the device take them. A rung's own block
naming an item it does not read is refused, as its override is; the
calculation's naming one no rung of the description reads, too. `time` and
`domain`, which are not catalogue items (§ 3.2a), reach every rung.

**And nowhere else** *(plan § 5w K5, 2026-09-30)*. A run setting is stated
in this block — the calculation's or the rung's own — and in no other
record: not in the template, which holds none (§ 2), not as a stage column of
the description ([`stages.md`](?doc=engines/stages.md) § 6.2), and not by a
`bench` row, which sets only the trials it declares (§ 3.4). So every reader
takes one answer, the catalogue's default (§ 4) with the calculation's block
over it where the rung reads the item, and the rung's over that — a
benchmark's trial excepted, which takes no rung's card (§ 3.4) — the deck, the scheduler's
device ask, the wrapper and the
Task setup tab's cards and hints alike — and a transport rung takes its run
card as every other rung does.

**Every value on a card is its item's to judge and spell** *(2.28.0, plan
§ 1w)*: a catalogue item's by the setting door — its type, its choices, its
limits, and whether a run card may hold it at all (`template.accepts`,
`template.spell`, [`template.md`](?doc=engines/template.md) § 5.4) — and the
run's own `time` and `domain` by this reader, as the allocation's are (§ 3.1,
§ 3.2a). So `"mpi_np": "8"`, `"use_gpu": "yes"` and `"diag_algorithm": "elpa"`
are refused where the job request is read — at Task setup's Save and at prep's
preflight — in the item's words, never later as something else.

**`null` on a card is *unset*.** For an item whose unset is an answer — the
catalogue's `optional`: the launch shape, `block_size`, `parallel_over_k` —
a rung's `null` leaves it unset on that rung, over the calculation's card: a
calculation-wide `"use_gpu": true, "gpu_count": 2` and a rung's
`"use_gpu": false, "gpu_count": null` run that rung on no GPU. For any other
item it is refused, naming the key: a speed setting with a default is stated
or left out.

**Absent is a state, as everywhere here.** A run whose launch shape is stated
neither here nor by a `prep` flag is refused, and the refusal names both
places ([`architecture.md § 5.2`](?doc=execution/architecture.md)).  **A
benchmark does not fill it**: it writes a report, and what the run uses is what
you wrote here.

**Which names may appear** is the catalogue's `execution` items — the same
membership `bench` uses — **plus `time` and `domain`**, which are not
catalogue items at all (§ 3.2a). The same split decides where each of the
catalogue names lands: a machine-answered item becomes the launch shape,
anything else a run setting laid over the catalogue's default as the deck,
the wrapper and the scheduler's device ask read it. The launch shape is a direct map of
the block's machine items onto the run's request (`prep_inputs.
declared_run_shape`, `generator.md` § 4.3a), never the benchmark's grid
enumerator, and that request is placed on its queue at prep, its fit said, as
a trial's cell is fit on its benchmark's.

**Why not `allocation`.** That block is what this calculation asks the
**scheduler** for — the queue, the wall, the memory, the GPU binding
(§ 3.1). `execution` is
what the **job** runs as. A `--mem` and an `mpi_np` are answered by different
things and refused by different doors, and a reader of one block holding both
could not say which of the two a key belongs to.

#### 3.2a `time` and `domain` in `execution` — the two asks a RUN owns

**A benchmark and a run want different wall clocks, and one field cannot hold
both.** *(User, 2026-09-02, choosing this over a split `allocation`:
"explicit, right next to prep run so easy to see and confirm.")*

`allocation` is folded by the shared prep path, so `prep bench` and `prep task`
read the same `time` and the same `domain`. That is one number for two jobs
with opposite needs:

| `allocation.time` | the benchmark | the real run |
|---|---|---|
| `0-04:00:00` | fine | **killed at four hours** |
| `2-00:00:00` | every ten-minute trial asks for two days, and queues behind everything | fine |

The second row is the one that is easy to miss: a short ask is scheduled into
gaps almost at once, so a benchmark that asks for two days waits a day to
start — and a benchmark is the thing you run to *save* time.

```json
"allocation": { "time": "0-04:00:00", "domain": "htc-debug", "mem": "256G" },

"stages": { "tight": { "group": 2,
                       "execution": { "mpi_np": 64,
                                      "time": "2-00:00:00", "domain": "public" } } }
```

**`allocation` is the calculation's ask and the BENCH's; `execution` is this
RUN's.** So the scheduler ladder gains one rung, for these two names only:

> `allocation` → **`execution`** → a `prep` flag — and stated nowhere,
> refused at prep on a target with a scheduler; a submitted job's wall may be
> restated at launch (`launch --time`)

**Spelled as the allocation's are.** A card's `time` is read through the same
door as `allocation.time` (`scheduler/quantities.canonical_time`) and held in
SLURM's spelling — `"4h"` is read, and written back, as `"0-04:00:00"` — and
one that cannot be read is refused by this reader, naming the card. A card's
`domain` is a queue's name, judged at prep against the target's record, as the
allocation's is (§ 3.1).

**`mem` is deliberately NOT among them.** The line is *does this differ
between the two lanes* — and memory does not: a trial and a run compute the
same system with the same basis, so they hold about the same amount. A wall
clock differs because a trial's step count is cut; a queue differs because
short work and long work belong in different ones. A field that would hold
the same value in both blocks would only be a second place to look.

**They are not catalogue items, and that is not an oversight.** The catalogue
describes what an *engine* computes with; a wall clock and a queue are what a
*scheduler* is asked for, and no engine has an opinion about either. They are
admitted to this block by name — which is also why a run's condition can never
turn one into a `bench` axis: `bench` takes catalogue items only, and a
benchmark that swept its own wall clock would be measuring the queue.

### 3.3 `group` — which stages share one job

**A stage's `group` is the job it runs in**: the stages with one number are one
job, prepared together and launched together
([`job-system.md`](?doc=execution/job-system.md), *The task*). The first stage
of the description is group 1; each later stage **repeats the number above it
or adds one**, so the groups run in the ladder's order. **A stage never shares
a group with a stage it builds on** — its continuation, a transport
rung's upstream — because a job runs its members
one after another with nothing handed between them. Whether a stage builds on
the one before it is the description's to say (its `restart`,
[`stages.md`](?doc=engines/stages.md) § 2), and the check reads it there, as
the science it is. Groups that build on each other go in one job as a
**sequence**, each group launched — taking what it builds on — when the one
before has finished ([`job-system.md`](?doc=execution/job-system.md) § 6.0, *A
sequence of groups*).

**Every entry states one.** `molbuilder describe` and Task setup write the
defaults, repeating the number above wherever that is allowed — the stage
builds on none of that group, and its run card states what theirs do on what a
job shares (`group.may_join`; else its prep refuses the group, R-P13): a
transport ladder 1 1 1 2 3, an optimization ladder 1 2 3, stages that build on
nothing 1 1 1. An entry without one is refused.

**It is the job request's because it says how the calculation runs**: which
stages one launch sends, and what they share — one allocation
([`project-layout.md`](?doc=execution/project-layout.md) § 1.6.6 lists what that
holds), so each member's run card states the same values for those, which Task
setup's group card writes into every member at once. It says nothing about
what a run computes. A group's members are compared by **who they are**, never
by the number: removing a stage above shifts the numbers below it, and that is
no change to any job.

**The check is the job request's own**, made with the description beside it
(`group.check_groups`): wherever the job request is read for a prep — the
preflight, checkpoint 3 ([`job-system.md`](?doc=execution/job-system.md)
§ 5.0) — and when Task setup saves it.

### 3.4 `bench` — a plan to measure, never a measurement

**The problem it solves.** A calculation's resource settings — ranks, threads,
memory, the GPU — cannot be chosen in a browser, because they depend on the
machine. But *which of them are worth measuring, and over what points* is a
decision about this calculation, and it is made by the person who set it up. So
it belongs in the job request, beside the run it measures for.

**The rule, and it is the same one the whole file rests on:**

> The job request records what the person **asked** — points to try, or a value
> they chose. It never records what a machine **found**.

**And ONE point on a NON-MACHINE entry is the value the trials run with**
*(user rule, 2026-08-20 — the override lane)*:
`use_gpu: [true]` is applied at prep to the bench's trials, **and only to
them**. The run reads its own card — a pin on it would be a second home for
the run's value beside `execution` (§ 3.2; user: *"trials only"*).

**What a trial takes for a speed setting the bench does not declare** is the
calculation's card, else the catalogue's default — what the calculation runs
at; never a rung's own card, which is that rung's run (§ 3.2). Whether the
trials are a GPU family is the same answer for `use_gpu`: the bench's points
for it, else the calculation's card, else the default
([`generator.md`](?doc=execution/generator.md) § 4.3a).

**A machine-answered entry is never an answer here, at any length.**
`mpi_np: [8]` is *measure eight* — one trial, not a decision. What the RUN
uses is `execution`, its own block (§ 3.2), and the two are independent: a
grid and a decision coexist, because narrowing a row to state the run would
destroy the plan to measure.

> **The line is decision vs finding, not portable vs not**
> (`generator.md` § 4.3a). *"Use 16"* is true on one cluster where *"try 4,
> 8, 16"* is true on all of them — yet `allocation` carries `domain`, and a
> queue name is no more portable than a rank count; what this file must
> never hold is what a **machine found**, which is why `summarize` writes its
> verdict to a report of its own (printed by `summarize`) that no code
> reads. A person who read a benchmark and chose eight ranks is *asking*, and
> asks are what this file is for.
>
> `template.md` § 7 is absolute: the **template** may not
> carry a machine value, and no template carries `mpi_np` at all —
> its door, `template_file.of`, refuses one. That rule is about a different
> file with a different job.

**Where the answer goes instead.** `bench` runs on the target and writes
`bench-result.json`, whose `choice` carries the measured value alongside the
rank and GPU counts; `summarize` prints it as a report, and no code reads it —
what the run uses is its run card (§ 3.2;
[`job-system.md § 7`](?doc=execution/job-system.md)). Two files, two jobs: the
job request says what to ask, the result says what the machine said.

**A field may appear here that may never appear in the template**, and that is
not a contradiction. `mpi_np` as a *point to try* is a question; `mpi_np` as an
item value would be an assertion about a machine the template has not met.
The two are different claims and only one of them travels.

**Which calculations it serves**: a SIESTA optimization's or vibration's
stage. A transport calculation's `prep bench` refuses, as it refuses an
engine the lane does not speak (below;
[`task-setup.md`](?doc=web/task-setup.md) § 11.1).

**What may be swept is not a free list.** A key must name a field the engine
already declares sweepable — the `execution` category, which
[`template.md` § 6.2](?doc=engines/template.md) defines as *"knobs that change
speed and not the answer"*. Sweeping something outside it means each point
silently measures a different calculation, and the comparison is meaningless.
**And a run setting the trials cannot vary is no row**: one every trial fixes
— `continue_retries`, a trial running once (the pins below) — whose points
would be one measurement under many labels, and of the machine's answers any
but the grid's axes — ranks, cores per rank, devices — a memory cap being no
axis the grid sweeps. Each is refused where the job request is written, by the
setting door ([`template.md`](?doc=engines/template.md) § 5.4, `home_of`),
never first at `prep bench`; and a point is a value to try, never `null`.

**The bench lane speaks SIESTA today, and `prep bench` says so by name.** A
trial is a MEASUREMENT, and what makes it one is the pin set applied over
every point: cap the SCF at three cycles, no relaxation steps, force a cold
start, no continue-retries, and let an unconverged cap end cleanly rather than
abort — SIESTA's keywords, which a PySCF script reads none of, so its trials
would be no measurement at all. So the refusal is explicit,
raised at `prep bench` before any trial is rendered, and it points at the
engine's own scaling guidance ([`tuning.md`](?doc=engines/tuning.md)) instead.
Extending the lane to another engine means giving that engine its own pin set,
not relaxing the check.

### 3.5 `notify` — when this calculation speaks, to whom, and with what

`notify` is the calculation's reporting **policy**: when to speak, to which of
the running machine's channels — **by name** — and which fields a chat card
shows. It travels with the calculation; what a channel name resolves to never
does ([`run-reports.md`](?doc=execution/run-reports.md) § 1).

```json
"notify": {"on_scf_converged": true, "every_hours": 6,
           "channels": ["slack"], "report": ["elapsed_s", "energy", "max_force"]}
```

| key | type | every / never | none | refused at save |
|---|---|---|---|---|
| `on_scf_converged` | boolean | — | `false` | anything but a JSON boolean |
| `every_hours` | number of hours | `null` is **never** | — | a string (`"6h"`), `0`, a negative or a non-finite number |
| `channels` | list of channel names | `["*"]` — **every channel the running machine has** | `[]` — **none**: reports off for this calculation | a name that is not letters, digits, `-` and `_`; `"*"` beside a name |
| `report` | list of report field names | `["*"]` — **every field the run states**, of those the receiver's layout shows for the event (`run-reports.md` § 4.1c) | `[]` — the name, the state and the summary line, with no field grid | a name that is not a report field, or one this calculation's runs can never state; `"*"` beside a name |

**A block states all four keys, every time** *(W57 R10 and D6, 2026-10-06)*:
a key left out is refused by name, so a choice and a block that never said it
are never the same bytes.

What each occasion is and what every message carries are
[`run-reports.md`](?doc=execution/run-reports.md) § 2 and § 4.1a; which
channels a list selects is § 3.0 there. **No `notify` block is no
notification**, the start and the end included; a calculation with a block also
reports its start and its end, so they are not keys. `notify` present but empty
is refused:
absent and empty would be two spellings of one state. **`["*"]` and `[]` are
two states for `channels` and for `report`**: a person who unticked every box
asked for something different from a person who kept *every*.

**The name is not on the `report` list, because it is not optional.** Every
report carries the run's name and, under a scheduler, its job id — in a chat
card's title, first. A report you cannot attribute to a job is a notification
you have to go and look up. So `report` is *what else*: it can be empty, and
it can never remove the name.

**A field is offered for what the run can state.** The fields, their units and
which runs state each are [`run-reports.md`](?doc=execution/run-reports.md)
§ 4.1a's table, read from the one declaration, `molbuilder/report_fields.py`,
which the job request, the wrapper, the monitor, the listener and the
Task-setup card all read — none keeps a list. The card offers only those
([`task-setup.md`](?doc=web/task-setup.md) § 9b), and a job request naming a
field its calculation can never report is **refused at save, by name**: a field
that can never arrive is a tick that silently means nothing. **One vocabulary**:
a field's name is the report's own wire name, so what you tick, what travels
and what a listener parses are the same word.

**A list is a ceiling, never a floor.** Asking for `energy` on a run that has
not printed one yields no `energy` field, not an empty one. And `report` shapes
a chat card — the fields in its grid, in the order and up to the count the
receiver's layout gives the event ([`run-reports.md`](?doc=execution/run-reports.md)
§ 4.1c); our own listener receives the whole record (§ 4.1b there).

**It is fixed at `prep`, not read at run time.** The wrapper bakes the whole
block into the monitor's command line
([`run-reports.md`](?doc=execution/run-reports.md) § 2.6), carried there on
`jobset.Resources` ([`job-contracts.md`](?doc=execution/job-contracts.md)
§ 6.2) — so what a running job reports cannot change because `job-request.json`
was edited while it sat in the queue, and the monitor needs no access to the
job request.

## 4. A speed setting nobody stated takes the catalogue's default *(decided 2026-10-21)*

**For each speed setting a rung reads, a run takes the rung's card, else the
calculation's, else the catalogue's default** — `template.default_of`, the one
default every reader asks ([`template.md`](?doc=engines/template.md) § 5.4),
which `resolve` lays over whatever the engine's config class held — and for an
item the catalogue gives no default (`block_size`, `parallel_over_k`), what its
not-set line says: no keyword written, the engine's own, or worked out (`parallel_over_k`,
from the k-point grid and the solver; [`template.md`](?doc=engines/template.md) § 6.6,
obligation 4). Each is written into the deck, the run script or the header as
any value is.

**Where each value came from is recorded in one vocabulary** —
`template.SAID` ([`template.md`](?doc=engines/template.md) § 5.4): *the stage's
run card*, *the calculation's run card*, *a prep flag*, *the run card* (the
launch shape as prep's fold records it, either card), *the job request's
allocation*, *the catalogue's default · not chosen* — beside the description's
(*this stage's override*, *the template · …*) and a benchmark's (*a pin*, *a
sweep point*): in the deck's PROVENANCE
`parameters:` rows for what the deck renders, and in the pipeline log for every
run setting, the run script's own (`wall_slack`, `continue_retries`) included
([`job-contracts.md`](?doc=execution/job-contracts.md) § 3.2;
[`template.md`](?doc=engines/template.md) § 6.6, obligations 4 and 5). Task
setup's run card shows each default in its place before anyone fills it. The
launch shape stays stated or refused, and so do a queue's wall, memory and
queue ([`architecture.md`](?doc=execution/architecture.md) § 5.2).

**Why a default, and no refusal.** A speed setting changes how fast a run
goes, never its answer, and the catalogue states each default with its reason
([`template.md`](?doc=engines/template.md) § 5.2). A rank count has no such
default: it is the machine's, and nothing but the person can say how much of
the machine a run takes (§ 2).

## 5. A launched stage's request may change *(decided 2026-10-21)*

**What a stage computes is fixed once it is launched; how it runs is not.**
`prep task --group N`, for a group with launched members on the hierarchical
layout, prepares it again: every member's deck, run script and header rendered
anew from both records,
its row in `job-set.json` replaced, and no run opened — the next launch opens
the next one. Each launched member's deck must be the same calculation as the
deck its newest launched run ran (§ 6), else the prep is refused, naming each
difference and the way back, the state saved before the group's first launch
([`job-system.md`](?doc=execution/job-system.md), *How the checkpoint supports you*). A change of the
job request passes; so does an edit of the description that leaves what the
engine computes unchanged — the test is the engine's input, never which file
was edited.

**On the flat layout a launched stage is not prepared again.** Its runs share
its folder's deck and run script, so a deck rendered anew would rewrite what
an earlier run's record reads as its own, and what a queued run starts with;
any change there takes the way back.

**Its runs are untouched.** Each holds its own copy of the deck and scripts it
ran with ([`project-layout.md`](?doc=execution/project-layout.md) § 1.5). The
next launch
opens the next run with the new files, and that run's launch record says what
it was sent with (`placed_on`, `command`): status shows a stage's row, which is
what its next launch sends, and each run's own request is its launch record's.
So a run out of memory or out of wall goes on with a bigger request: change the
job request, prepare the group again, launch the stage again — warm, from
where its last run reached.

**What stays fixed of a launched group is its members** — a stage put into it
or taken out takes the way back it took before
([`job-system.md`](?doc=execution/job-system.md) R-P5, R-L7). **What may be
done to a group now is one answer**, `ready.GroupAnswer`
([`job-system.md`](?doc=execution/job-system.md), *The task*): prep's offer and
its refusals, the launch's, `status`'s next step and Task setup's boxes all
read it, so a launched group whose members changed is offered by none of them,
and every way on is the one composer's (`commands.way_back`). And **a change to
what a launched stage computes** still goes back to the state saved before its
group's first launch. In a sequence (`--one-job`), a launched group is read as
its last prep wrote it ([`job-system.md`](?doc=execution/job-system.md) § 5.0, checkpoint 2a).

**A launch sends what prep wrote.** A job request changed since its prep
reaches a run when its group is prepared again; until then `status` and Task
setup say the group is *changed since prepared*, each value that differs
([`job-system.md`](?doc=execution/job-system.md), *Changed since
prepared*).

## 6. The same calculation — one door, each engine's reading

**Two decks are the same calculation when their engine computes the same thing
from both, the run settings aside.** One door,
`deck_record.same_calculation(engine, a, b)` → every difference, by name — none
when they are the same. It sits with the grammar of a deck's blocks, below
every reader that asks it, the run record's among them (floor 1,
[`architecture.md`](?doc=execution/architecture.md) § 2.1), and imports no engine: each engine's reading is
that engine's own module, registered with the door by each surface as it
starts, as the readers of a calculation's own records are
(`calculation.register_readers`):

- **SIESTA** — `siesta.engine_input`, the same engine input: both decks read as SIESTA reads them
  ([`engines/transport.md`](?doc=engines/transport.md) § 6.1, *the same engine
  input*), with every run setting's keyword aside — the keywords the
  catalogue's run settings write (§ 2: their `anchor`, or their `expands`).
  It reads keywords by name, so it holds of a deck any molbuilder wrote.
- **PySCF** — `pyscf.script_input`, standard library only: the script with
  its run-settings block, its comments and molbuilder's stamps aside
  ([`job-contracts.md`](?doc=execution/job-contracts.md) § 3.1, § 3.2).

**Asked by**: the transport gather — the upstream run computed what this
rung's deck computes now ([`engines/transport.md`](?doc=engines/transport.md)
§ 6.1); prep, of a launched stage (§ 5); the run record's `current` and the
run door's `Run.report` — whether the stage's deck, and the check written beside
it, are still the calculation the run ran ([`model/parse.md`](?doc=model/parse.md)
§ 5d; [`architecture.md`](?doc=execution/architecture.md) § 3.2). **Not** of what a launch sends: a
stage's own files go as prep wrote them ([`job-system.md`](?doc=execution/job-system.md) § 6.0);
what a first launch asks is of the runs it takes from — the gather's, and a
take-over's in a calculation that reuses a finished one.

**A deck's comments record what it was rendered with** — PROVENANCE's
`resolved-defaults`, BENCH-MARKS' rank count — and nothing reads them as a
launch: prep writes a deck and its launch together, once
([`architecture.md`](?doc=execution/architecture.md) A3), and launch sends what
prep wrote. So the agreement a launch checked retires —
`agreement.check_launch_matches_deck`, `launch_agreement`,
`disagreement_note` — and prep's answer carries none, nor the ledger. What
`jobset/agreement` keeps is a benchmark trial's cold start
(`check_trial_starts_cold`, R-L29), read from the deck's own restart keywords.

## 7. Who writes it, and who reads it

**Task setup's Save writes both records**, each whole, the folder's state
saved first, once ([`web/task-setup.md`](?doc=web/task-setup.md) § 8); each
card writes the record it belongs to. **`molbuilder describe`** writes both at a
calculation's start (§ 9), as the Transport tab's describe door does. A person
may edit either file; a reader refuses what does not read, naming the file.

**`prep` reads it once per prep**, through its door, inside the calculation it
opens (`calculation.open` → `Calculation.request`, beside its description, its
template and its plan); prep's flags lay over it field by field for that prep
alone ([`architecture.md`](?doc=execution/architecture.md) § 5.2) — what a
flag said reaches the job's row as the values it resolved to (`resources`),
never this file.
**`launch` takes no launch value from it**: it sends what prep wrote,
`--time` restating a submitted job's wall for that launch
([`job-system.md`](?doc=execution/job-system.md) § 6.0). It reads the job request for its groups —
which stages one launch sends. Status, summarize, the transport gather and the
continuation door take it from the calculation their verb opened
(`Calculation.request`, A16); `jobset migrate` and the Task setup cards read
it through the same door.

**A calculation's folder holds both records.** A description with no job
request beside it is refused when the calculation is opened
(`calculation.open`, [`job-system.md`](?doc=execution/job-system.md) R-P1), naming the file and
Task setup's Save, which writes both.

**Its floor is 2, beside the description**
([`architecture.md`](?doc=execution/architecture.md) § 2.1):
`molbuilder/job_request.py`. Its door `of(folder, description)` →
`JobRequest`, `None` when there is none — the description the caller holds
beside it (A16), which its stages are checked against (§ 3); `from_text`, the
same reading of a body Task setup sends; `write`, the record whole;
`JobRequest.run_condition(stage)`, a rung's card laid over the calculation's
for the items the rung reads (§ 3.2), the one run-card door. **Its reader
checks** what the record says of itself against the description beside it —
the schema, the stage names, the groups' order, each card's one value and
each benchmark row's list of points, the allocation's spellings, the
reports, a card's `time` spelled and an unset key's `null` (§ 3.2); **the
preflight checks** what needs the vocabulary or the ladder — through the
setting door, each value on a card or a benchmark row one its item takes in
that home: a run setting the rung reads, of its type, among the choices the
kind offers, within its limits (`template.accepts`,
[`template.md`](?doc=engines/template.md) § 5.4); no stage sharing a job with
one it builds on (`group.check_groups`, § 3.3). Its name is the catalogue's (`runfiles.WRITTEN`).

**Task setup works on both as edited**: its prep plan reads the job request
as the page holds it — the page sends it, read by the one reader — as it
reads the description, and its bench grid the benchmark's points and the
queue picked, as edited; its Preview, the prep entry,
reads both files on disk, so the page refuses it while an edit is unsaved
([`web/task-setup.md`](?doc=web/task-setup.md) § 7a); the hand-over proposes each stage's group beside
its stages, and Save writes it into the job request. The Transport tab's
describe door writes what `molbuilder describe` writes.

## 8. What moves on disk — `jobset migrate`

A calculation described before 2.27.0 holds its job request inside `task.json`
(`molbuilder/task@1`) and its speed settings' values in its template; one
described by 2.27.x holds every run setting in its template, declared and
valueless. Every reader refuses either by name, naming
`molbuilder jobset migrate --bundle <calc>`. **What migrate moves is decided by
what a file holds, never by its version** *(2.28.0, plan § 1w: the template's
reader refused a run-setting value by content while migrate moved one only
from a `task@1` calculation, so a value in a 2.27.0 template was dropped)* —
each change where it is found. **A calculation holding nothing an older
molbuilder wrote is current, and said so** — *is current -- nothing to
migrate: …*, what was checked, no failure; its ledger records `migrate`
`current` *(the user, 2026-10-22: "change the word of migrate so it does not
look like an error")*. A calculation that cannot be migrated as it stands
is refused by name, nothing written:

- writes `job-request.json` from `task.json`'s `allocation`, `execution`, each
  stage's `execution` and `group`, `bench` and `notify` — a `notify` block's
  keys left out written with what their absence meant, `every_hours: 0` as
  `null` — and rewrites `task.json` without them, at `molbuilder/task@2`;
- takes every run-setting item out of the template, valueless or not, and
  moves a value it holds onto the calculation's run card — of the job request
  it writes, or of the one already beside the description — where the card
  states none, unless the template records it *not chosen* and it is the
  catalogue's default now (`template.default_of`) — so every deck renders the
  values it did, and a value a person chose stays chosen;
- moves `restart` and `density_fit` off the run cards: the calculation's
  card's into the template, a stage's into its overrides, `varies` gaining it;
  a `bench` row naming either is refused, naming it — neither is a run setting
  to measure;
- rewrites each row of `job-set.json`: the program onto the job
  (`Job.program`), a placement's source `description` as `allocation`;
- leaves a transmission point's digest as it is: a digest of another
  version is out of date, and the next reading of its data file writes it
  whole (`model/parse.md` § 5e). A vibration result is not rewritten: its map is
  made as it is read, and its JSON's hash stays the one its data file and
  the frame sets made from it name;
- keeps each file it rewrites beside its new one, `<name>.pre-<rule>`, as every
  migration does.

**A calculation described by 2.27.x is migrated once more**: its template's
declared run-setting items go, and nothing it computes changes.

**A prepared stage is prepared again before its first launch**: the files prep
writes now differ from those it wrote before — the run script takes its wall
slack and retries from the job, with no fallback of its own (§ 4). A launched one keeps its
runs; a SIESTA one's group may be prepared again as § 5 says. A PySCF script
written before holds no run-settings block, so none rendered now is the same
calculation (§ 6): a launched PySCF stage takes the way back for any change.
A benchmark's trials took the template's values, and now take the
calculation's card (§ 3.4) — the same values, but where the two disagreed.

## 9. Describing is not a job verb — `molbuilder describe`

**`molbuilder describe` writes the description, its job request and its data
files**, with its flags ([`job-system.md`](?doc=execution/job-system.md)
§ 5.1). **`molbuilder jobset` prepares, launches, observes, summarizes and
migrates**, and has no verb that describes. The next step describe
prints is prep's line (`jobset.commands`).

## 10. What this contract does not own

- **The ladders a launch value is fixed by**, and what is refused when it is
  stated nowhere — [`architecture.md`](?doc=execution/architecture.md) § 5.2.
- **The queue's record, a request's placement and its fit** —
  [`scheduler.md`](?doc=execution/scheduler.md).
- **What prep and launch do with it** — [`job-system.md`](?doc=execution/job-system.md).
- **The cards that write it** — [`web/task-setup.md`](?doc=web/task-setup.md).
- **The description** — [`engines/stages.md`](?doc=engines/stages.md) § 6 and
  [`engines/template.md`](?doc=engines/template.md).
