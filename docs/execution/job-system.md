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

## 1. What the job system is, and why it exists

### The problem it solves

[`running-a-job.md`](?doc=execution/running-a-job.md) covers running **one**
calculation: you generate an input, the wrapper activates the right software
environment, runs the engine, and you watch it. That is the whole story for a
single task — and for a long time it was the only story molbuilder told.

Real research is not one task. A single result usually needs a **sequence** of
runs (relax a molecule loosely first, then tightly), and a project needs
**many** such sequences (the same analysis across dozens of structures). Getting
those onto a supercomputer adds its own chores: writing scheduler headers,
chaining jobs so stage 2 only starts after stage 1 succeeds, carrying restart
files forward between stages, and — before any of that — figuring out how many
GPUs and CPU cores actually make a given calculation fastest on a given machine.

Done by hand, that is a pile of error-prone shell scripting that every user
re-invents. The **job system** exists to make it a described, repeatable thing:
you say *what* you want as data, and molbuilder handles *how* to lay it out,
chain it, deploy it, and watch it — the same way whether it runs on your laptop
or a cluster.

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
      D["a description of work<br/>(a JobSet)"] --> O["orchestration:<br/>lay out, chain,<br/>deploy, roll up status"]
    end
    O -.->|"produces + runs many of"| W
    classDef s fill:#eef;
    class system s;
```

Concretely, three kinds of "many jobs" all reduce to the same object:

- a **staged ladder** — one molecule, relaxed in increasingly tight stages, each
  stage a separate job that depends on the one before it;
- a **parameter sweep** — the same calculation run at many resource settings to
  find the fastest (this is what benchmarking is);
- (planned) a **device workflow** — e.g. a transport junction assembled from two
  separately-relaxed electrodes.

The object they all reduce to is a **`JobSet`** (§ 3).

### A one-paragraph scenario (what a user actually does)

You have a molecule `bdt.xyz` and want a publication-quality relaxed geometry on
your cluster. You run **one** command to produce a *bundle* — a self-contained
directory holding a coarse stage and a tight stage plus a `job-set.json`
describing the chain. You `scp` the bundle to the cluster and run three verbs:
`prep` (lay out the per-stage folders and wrappers), `plan` (review the chain
and the resources), `submit` (hand it to the scheduler). The scheduler runs
stage 1; stage 2 starts automatically only if stage 1 converged, reusing stage
1's relaxed coordinates. You check `status` whenever you like. You never wrote a
`#SBATCH` header or a dependency flag.

### Where it stands today (read this before the details)

> **The job system is shipped on the command line and pending on the web.**
> Everything in this document — the `JobSet` model, the `molbuilder jobset`
> verbs, SLURM submission with dependency chains and routing domains, and the
> whole benchmark workflow — works today from a terminal. The **web UI does not
> drive any of it yet**; it still generates and runs one task at a time. Moving
> the job system into the browser is the target migration, laid out in § 8.

> ### ⚠ Automatic stage-to-stage chaining is being retired for staged calculations
>
> **This affects the ladder only, and only the *automatic* part of it.** Read this
> before building on `depends_on` or `Carry`.
>
> Everything below describes a ladder as a **chain**: stage 2 is a separate job
> that `depends_on` stage 1, the scheduler starts it automatically when stage 1
> succeeds, and a `Carry` edge hands stage 1's relaxed geometry across. That is
> what `stages_to_jobset` builds today and what SLURM submits.
>
> **The staged-runs design removes that for a staged calculation.** The reason is
> not technical — it is that *a stage is a long job*:
>
> > A chain that continues by itself can spend a week computing from a geometry
> > you would have rejected in a minute.
>
> So each stage becomes **its own prep and its own submission**, done after you
> have looked at the previous one. Whatever a run continues from is a **real file
> copied in at prep time**, and *which* run it comes from is something **you
> say** — by then it has already finished, so there is nothing to resolve later
> and nothing to point at that does not yet exist.
>
> | | Ladder as a chain (this document) | Staged calculation (the new design) |
> |---|---|---|
> | Who starts stage 2 | the scheduler, automatically | **you**, after looking |
> | How stage 1's geometry arrives | a `Carry` edge resolved at run time | a **file copy**, made at prep |
> | `Job.depends_on` between stages | set | **not emitted** |
> | If stage 1 converged to something wrong | stage 2 is already running | you never started it |
>
> **The mechanism is not being deleted.** `JobSet`, `depends_on`, `Carry` and
> `carry_deref` all stay, and `jobset` can still build a chained ladder — a
> benchmark sweep and an explicitly-chained workflow both still want them. What
> changes is that **`stages_to_jobset` stops emitting those edges between the
> stages of one calculation.**
>
> **Partly landed, 2026-08-10.** What ships now: stage directories are
> `<seq>_<name>` rather than `point-<name>`; `prep run <stage>` opens a
> `run-<n>` attempt and **copies** in what you name with `--from`; `submit run
> <stage>` launches one stage inside that attempt and writes `run.json`. What
> has NOT landed: **`stages_to_jobset` still emits `depends_on` and `Carry`
> between the stages of one calculation** — so the JobSet on disk still
> describes a chain even though nothing follows it any more. That last edit is
> item 12b in
> [`web/staged-runs-architecture.md`](?doc=web/staged-runs-architecture.md);
> the contract is
> [`execution/project-layout.md`](?doc=execution/project-layout.md) § 1.6, and
> the commands are § 5.3 below.

---

## 2. The design logic — why it is built this way

Six decisions shape everything below. Understanding them makes the rest obvious.

**1. Describe work as data; keep the engine out of it.** A `JobSet` is a plain
data object (a JSON file), not code. Producers turn engine configs *into* a
`JobSet`; the orchestration verbs (`prep`/`plan`/`submit`/`status`) operate on
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

**4. Fail early, never guess.** A malformed chain (a job depending on a
non-existent one, a cycle, a missing partition) is rejected at produce/validate
time with a clear message, not discovered halfway through a cluster run. The
framework refuses to emit an incomplete SLURM header rather than submitting
something that will bounce.

**5. molbuilder informs; the user decides.** The framework never silently
auto-resumes a failed or interrupted run. `status` tells you which stage is
incomplete and what restart files exist; **you** choose to re-submit. This keeps
a surprising re-run from quietly overwriting hours of results.

**6. Keep the dependency graph simple until a real case needs more.** A job has
**one** parent (or none). That makes every `JobSet` either a straight chain (a
ladder) or an independent set (a sweep) — always ordered, never cyclic. A
branching/merging graph (a "diamond", needed for a two-electrode device) is a
deliberately-deferred extension, not an accident (§ 8).

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
      list shared
      list jobs
    }
    class Job {
      str name
      str script
      str depends_on
      str dep_kind
      Resources resources
      list carry
    }
    class Resources {
      int mpi_np
      int cpus_per_task
      str time
      str mem
      str gres
      bool exclusive
      str domain
    }
    class Carry {
      str pattern
      str from_job
    }
    JobSet "1" o-- "many" Job
    Job "1" o-- "1" Resources
    Job "1" o-- "many" Carry
```

Walk through it with the *why* for each piece:

- **`JobSet.kind`** is `"ladder"` or `"sweep"` — the two shapes decision #6
  allows. A **ladder** is a chain where each job depends on the previous one; a
  **sweep** is a set of independent jobs with no dependencies. The kind tells the
  submitter whether to thread a dependency between jobs or fire them all at once.
- **`JobSet.shared`** lists files that every job needs but that never change
  between jobs — the pseudopotentials, the geometry, the monitor helper. They
  are stored **once** and symlinked into each job's folder, so a 20-point sweep
  does not carry 20 copies of a large pseudopotential.
- **`Job.name`** does double duty: it is the job's folder (`point-<name>/`) *and*
  its scheduler job name (`-J`), so a `squeue` listing reads the way the layout
  does. **`Job.script`** is the input file inside that folder.

  > **`point-<name>/` is today's naming for every job, and it splits.** A sweep
  > point keeps a settings-based name; a **stage** becomes `<seq>_<stage>`
  > (`01_coarse`), because a stage is ordered and a sweep point is not, and one
  > shape for both hides the difference. The full table, for every layer, is
  > [`job-contracts.md`](?doc=execution/job-contracts.md) § 6.3.
- **`Job.depends_on` + `Job.dep_kind`** are the single-parent edge (decision #6).
  `dep_kind` is `afterok` ("run me only if my parent **succeeded**") or
  `afterany` ("run me once my parent **finished**, pass or fail"). This is how a
  ladder's per-stage convergence policy becomes a scheduler rule (§ 4.1).
- **`Job.resources`** (`Resources`) are the per-job scheduler asks — all optional,
  `None` meaning "inherit / resolve at submit". They use the **scheduler's
  vocabulary** (`mpi_np` for MPI ranks → `-n`; `cpus_per_task` for cores/rank →
  `-c`; `time`, `mem`, `gres`, `exclusive`, `domain`), the same names the
  persisted files and SLURM flags use — the full mapping is pinned in
  [`job-contracts.md § 6.2`](?doc=execution/job-contracts.md). Per-job resources
  matter because a coarse first stage and a tight final stage want very
  different node sizes.
- **`Job.carry`** (a list of `Carry`) is how a job inherits a restart file from
  an earlier one. `pattern` is a **concrete filename** (not a wildcard) and
  `from_job` names the producer — e.g. "stage 2 carries `bdt.XV` from stage 1".
  Carry is what lets a tight stage start from the coarse stage's relaxed
  geometry instead of the original coordinates.

Every `JobSet` is checked by **`validate()`** before it is used: the `kind` must
be known, names unique, `dep_kind` valid, and every `depends_on` / `from_job`
must point at a **job listed earlier** — which guarantees the graph is ordered
and has no cycles.

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
  "kind":   "ladder",                 // "ladder" = a chain; "sweep" = independent

  "shared": ["C.psml", "H.psml", "S.psml", "Au.psml", "mb_monitor.py"],
                                      // stored ONCE in the bundle root and
                                      // symlinked into every job's folder

  "jobs": [
    {
      "name":   "coarse",             // → folder point-coarse/ AND squeue -J name
      "script": "bdt_au_coarse.fdf",  // the deck, living in the bundle root
      "resources": {                  // ALL seven fields are always written,
        "domain":        null,        // nulls included — see the note below
        "time":          "04:00:00",
        "exclusive":     null,
        "mem":           null,
        "gres":          null,
        "mpi_np":        8,
        "cpus_per_task": 4
      },
      "depends_on": null,             // nothing before it
      "dep_kind":   "afterok",
      "carry":      []                // starts from the .fdf's own coordinates
    },
    {
      "name":   "tight",
      "script": "bdt_au_tight.fdf",
      "resources": {
        "domain":        null,
        "time":          "24:00:00",
        "exclusive":     null,
        "mem":           null,
        "gres":          "gpu:a100:1",
        "mpi_np":        32,          // 4× the ranks: the tight stage is the
        "cpus_per_task": 4            // expensive one, and per-job resources
      },                              // are the whole reason they are per-job
      "depends_on": "coarse",
      "dep_kind":   "afterok",        // only if coarse actually converged
      "carry": [
        { "pattern": "bdt_au.XV", "from_job": "coarse" },   // relaxed geometry
        { "pattern": "bdt_au.DM", "from_job": "coarse" }    // density matrix
      ]
    }
  ]
}
```

*(Produced by building that `JobSet` and dumping `to_dict()`, so the shape and
the key order are the real ones, not a sketch.)*

> **`null` is a value here, and it does not mean "zero" or "off".** It means
> **"not decided yet — resolve it at submit"**. A `mem` of `null` lets the
> scheduler config's default apply; a `mem` of `"0"` is SLURM's *"give me the
> whole node's memory"*. The fields are written out even when null so the file
> shows you the complete set of questions that will be answered, rather than
> hiding the ones nobody answered yet. This is the *assistant, not nanny* rule in
> file form: molbuilder does not quietly pick a node size for you.

**What that file becomes on disk.** `molbuilder jobset prep` reads it and lays
out the tree. Every arrow below is a **symlink**, which is the point: nothing is
copied, so a 4 GB pseudopotential set exists once no matter how many jobs there
are.

```
bdt_au-bundle/                       ← the bundle root: the real files live here
├── job-set.json                     the description above
├── bdt_au_coarse.fdf                the two decks
├── bdt_au_tight.fdf
├── C.psml  H.psml  S.psml  Au.psml  the shared package
├── mb_monitor.py
├── bdt_au_coarse.run.sh  .sbatch    wrappers, rendered once per distinct deck
├── bdt_au_tight.run.sh   .sbatch
├── STAGE-PLAN.md                    a human-readable review of the chain
│
├── point-coarse/                    ← one folder per job
│   ├── bdt_au_coarse.fdf   → ../bdt_au_coarse.fdf
│   ├── C.psml              → ../C.psml          (and the other three)
│   ├── mb_monitor.py       → ../mb_monitor.py
│   └── bdt_au_coarse.run.sh → ../bdt_au_coarse.run.sh
│
└── point-tight/
    ├── bdt_au_tight.fdf    → ../bdt_au_tight.fdf
    ├── C.psml              → ../C.psml          (and the other three)
    ├── bdt_au.XV           → ../point-coarse/bdt_au.XV   ← the carry
    └── bdt_au.DM           → ../point-coarse/bdt_au.DM   ← the carry
```

> ⚠ **The tree above is a SWEEP's, and the paragraphs after it describe the
> ladder as it was before stages stopped chaining.** Keep reading them for the
> benchmark, where the points are independent and the whole set is submitted at
> once. **A stage ladder looks different today** — § 5.3 and
> [`project-layout.md`](?doc=execution/project-layout.md) § 1.6 are the current
> answer:
>
> ```text
> 01_coarse/                        ← <seq>_<name>, not point-<name>
> │   bdt_au_01_coarse.fdf              the deck, and its wrapper
> └── run-0/                        ← one directory per ATTEMPT, immutable once run
>     ├── bdt_au_01_coarse.fdf  →  ../bdt_au_01_coarse.fdf
>     ├── run.json                      written by submit: how, when, and what
>     │                                 this run continued from
>     └── bdt_au.XV  bdt_au.DM  …       what the run produced
>
> 02_tight/
> └── run-0/
>     ├── bdt_au.XV                 ← a real COPY of 01_coarse/run-0/bdt_au.XV,
>     └── bdt_au.DM                    made by `prep run tight --from …`
> ```
>
> Nothing dangles, because nothing points at a file that does not exist yet.
> The copy is made when you set the next stage up, from an attempt that has
> already finished and that you have already looked at.

**The two carry links point at files that do not exist yet, and that is
deliberate.** `prep` runs before anything has been computed, so
`point-coarse/bdt_au.XV` has not been written. The link dangles until the coarse
stage runs, and resolves the moment it does. What makes that safe is the
dependency: `point-tight` is submitted with `afterok:<coarse job id>`, so the
scheduler will not start it until coarse has finished successfully — by which
time the link points at a real file.

**One more step happens at run time, and it exists to protect the coarse stage.**
The tight stage does not read through the symlink; the wrapper first replaces
each carried link with a **real local copy** (`carry_deref`). Without that, SIESTA
would open `bdt_au.DM` for writing, follow the link, and overwrite the coarse
stage's density matrix — destroying the result you would want to go back to.

```mermaid
sequenceDiagram
    participant P as jobset prep
    participant C as point-coarse/
    participant T as point-tight/
    participant S as SLURM
    P->>C: link deck + shared package
    P->>T: link deck + shared package
    P->>T: link bdt_au.XV, bdt_au.DM → ../point-coarse/  (dangling)
    P->>S: submit coarse, then tight with afterok:coarse
    S->>C: run coarse
    C-->>C: writes bdt_au.XV, bdt_au.DM
    Note over T: the links now resolve
    S->>T: coarse succeeded → start tight
    T-->>T: carry_deref: replace each link with a real COPY
    Note over T,C: so writing bdt_au.DM cannot reach back into point-coarse/
    T-->>T: run tight from the copied state
```

> **This chained form is what ships today, and it is being narrowed** — see the
> notice in § 1. In the staged-runs design each stage is prepped and submitted on
> its own, so the carry becomes a plain copy made at prep, and neither the
> dangling link nor `carry_deref` is part of that story. Both stay for the
> chained ladder and for benchmark sweeps.

A complete 2-stage ladder `job-set.json`:

```json
{
  "schema": "molbuilder/job-set@1",
  "name": "bdt", "engine": "siesta", "kind": "ladder",
  "shared": ["Au.psml", "S.psml", "C.psml", "H.psml", "mb_monitor.py"],
  "jobs": [
    { "name": "stage1", "script": "bdt_stage1.fdf",
      "resources": { "domain": "htc", "time": "0-04:00:00" },
      "depends_on": null, "dep_kind": "afterok", "carry": [] },

    { "name": "stage2", "script": "bdt_stage2.fdf",
      "resources": { "domain": "public", "time": "7-00:00:00", "exclusive": true },
      "depends_on": "stage1", "dep_kind": "afterany",
      "carry": [ { "pattern": "bdt.XV", "from_job": "stage1" },
                 { "pattern": "bdt.DM", "from_job": "stage1" } ] }
  ]
}
```

Read it back in plain language: *a SIESTA ladder named `bdt`; four
pseudopotentials and the monitor are shared by both stages; stage 1 runs on the
`htc` domain for up to 4 hours with no dependency; stage 2 runs on the `public`
domain (the whole node) once stage 1 **finishes**, reusing stage 1's `.XV`
coordinates and `.DM` density matrix.* The edge is `afterany` (not `afterok`)
because stage 1's policy is `proceed` — a loose first stage is *meant* to hand
its geometry forward even if it did not fully converge (§ 4.1). A ladder whose
first stage had `halt` would instead produce `afterok` ("stage 2 only on
success").

---

## 4. Where `JobSet`s come from — the two producers

A `JobSet` is never written by hand; a **producer** builds it from a higher-level
input. There are exactly two producers today, and this is the one place engine
knowledge lives.

```mermaid
flowchart LR
    C["a staged SiestaConfig"] -->|"stages_to_jobset"| L["JobSet (ladder)"]
    G["a benchmark grid<br/>(GPUs × ranks × cores)"] -->|"sweep_to_jobset"| S["JobSet (sweep)"]
```

### 4.1 The SIESTA staged ladder (`stages_to_jobset`)

`molbuilder/siesta/stages.py::stages_to_jobset(cfg, stages)` turns a **template**
(one ordinary `SiestaConfig`) plus a **ladder** (a list of `task.py::Stage`, from
`task.json`) into a **ladder** JobSet — one job per enabled stage, script
`<label>_<stage>.fdf`. An engine config carries no stage list, so the ladder is
an argument rather than a field ([`engines/stages.md`](?doc=engines/stages.md)
§ 1.1). Three things are *derived*, and each encodes a design decision:

- **The dependency edge comes from the non-convergence policy** the producer was
  given for that stage. The policy says what to do if a stage hits its step cap
  without converging:
  `proceed` (go on anyway), `halt` (stop the chain), or `continue` (intended:
  retry the stage up to `continue_retries` times before giving up). These become
  the scheduler edge: `proceed → afterany` (next stage runs regardless),
  `halt`/`continue → afterok` (next stage runs only on success). So the
  scientific policy and the scheduler behaviour are the *same* setting — you
  never configure them twice. (This is the shared staged-optimization contract in
  [`engines/overview.md`](?doc=engines/overview.md).)
- **The carry-forward set is chosen for correctness, not convenience.** `.XV`
  (coordinates) is carried **always** — a later stage
  must continue from the geometry the earlier one reached. `.DM` (density matrix)
  is carried only if the config saves it. `.CG` (conjugate-gradient optimizer
  state) is carried **only when consecutive stages use the same relaxation
  method** — a CG state is meaningless to a Broyden stage, so blindly carrying it
  would corrupt the restart. The comparison is over each stage's **resolved**
  config, so a stage that does not override the optimizer is correctly read as
  having the template's rather than as having none.
- **Resources are per-stage**, defaulting to inherit the config's ranks/threads
  and otherwise resolved at submit — so a coarse stage and a tight stage can be
  sized differently.

**The shipped default ladder** (the *structure*; the *values* and their
scientific rationale live in [`engines/tuning.md`](?doc=engines/tuning.md)):

| Stage | Enabled by default | Relaxation | Steps | On non-convergence | Edge to next |
|---|:--:|---|--:|---|---|
| stage1 | ✅ | CG | 600 | **proceed** | `afterany` |
| stage2 | ✅ | Broyden | 200 | **halt** | `afterok` |
| stage3 | — | Broyden | 100 | **halt** | — |

The three **strategy presets** flip only the enable flags:
`loose-only` = (✅, —, —), `publishable` = (✅, ✅, —),
`vib-quality` = (✅, ✅, ✅).

**The warm-retry budget now travels the whole way** (fixed 2026-08-07).
`continue_retries` is an ordinary field of the shared schema; `stages_to_jobset`
carries it into `jobset.Resources` — the same road `mpi_np` and `omp_threads`
ride — and `jobset/prep` hands it to `write_run_wrapper`, which bakes it into
the wrapper's own retry loop (`?doc=execution/running-a-job.md` § 3.5). It
becomes **no `sbatch` flag**, which is why it is the one row of
`job-contracts.md § 6.2`'s translation table with no SLURM name. Before that
last hop existed the field validated everywhere and was **silently dropped at
prep**, so a `continue` stage was indistinguishable from a `halt` one — this
paragraph used to record that as a standing gap.

The dependency edge is a separate question and unchanged: a `continue` stage
still takes the same `afterok` edge as `halt`, because its terminal failure
mode *is* halt. The retries happen inside the one job the scheduler ran.

> **Where the policy lives, since it is not where it looks.** `on_nonconvergence`
> is **not** a field of a stage and not a field of the shared schema
> ([`engines/stages.md`](?doc=engines/stages.md) § 3) — its entire effect is
> this edge, so it is the producer's own input, passed to `stages_to_jobset`
> keyed by stage name. `siesta/stages.py::DEFAULT_NONCONVERGENCE` is the
> shipped default, and it is what the table above tabulates. A stage the
> mapping does not name gets `halt`.

There is also a pure, side-effect-free **`build_siesta_stage_bundle(struct,
cfg, stages)`** that returns a ready-to-write stage bundle by reusing the stage
`.fdf` renderers plus `stages_to_jobset`. It exists as the clean seam a future
**web** Build producer will call (§ 8).

### 4.2 The benchmark sweep (`sweep_to_jobset`)

`molbuilder/bench/to_jobset.py::sweep_to_jobset(...)` builds a **sweep** — one
independent job per point of a `(GPUs, ranks-per-GPU, cores-per-rank)` grid,
**no carry** (the points don't depend on each other). It is the producer behind
the benchmark workflow (§ 6). Because a sweep is just another `JobSet`, the same
`jobset` verbs run it — benchmarking is not a separate machine, it is the job
system pointed at a resource grid.

> **A ladder that is *not* a JobSet — PySCF.** PySCF also supports staged
> relaxation, but it runs its stages as an **in-script loop inside the single
> `.py`** (see [`engines/tuning.md`](?doc=engines/tuning.md)), not as separate
> scheduled jobs — `molbuilder pyscf` has no `--jobset` flag. So "a ladder
> scheduled as a dependency chain" is a **SIESTA-only** reality right now.
> PySCF, transport, and spectra producers are planned (§ 8).

---

## 5. The workflow — produce, prep, plan, submit, watch

The lifecycle has one step on the **host** (where you design the calculation) and
four verbs on the **target** (where it runs). They mirror the design: the host
step is pure file-writing; scheduler contact happens only at `submit`.

```mermaid
flowchart LR
    subgraph host["HOST — laptop or login node"]
      P["produce<br/>fdf --jobset<br/>→ job-set.json + stage .fdf's"]
    end
    subgraph target["TARGET — cluster or workstation"]
      direction LR
      PR["prep<br/>lay out point-*/ + wrappers"]
      PL["plan<br/>review the chain"]
      SU["submit<br/>--mode submit | direct"]
      ST["status<br/>per-stage roll-up"]
      PR --> PL --> SU --> ST
    end
    P -->|"scp the bundle"| PR
```

### 5.1 Produce (host)

```bash
# Render the stage .fdf's + job-set.json into ./bundle
molbuilder fdf bdt.xyz bundle/bdt.fdf --stage-strategy publishable --jobset \
    --psml-lib ~/pseudos \
    --stage-resources '{"stage1": {"domain": "htc",    "time": "0-04:00:00"},
                        "stage2": {"domain": "public", "time": "7-00:00:00", "exclusive": true}}'
```

`--jobset` requires a multi-stage config (a single deck has nothing to chain);
`--stage-resources` requires `--jobset`, and its stage names and fields are
validated against the actual ladder — a typo like `"stage9"` fails here, on your
laptop, not on the cluster (design decision #4).

### 5.2 What `prep` lays out on disk

> ⚠ **This section describes the SWEEP, and the ladder as it was before stages
> stopped chaining.** Read it for the benchmark, where every point is independent
> and submitting the whole set at once is correct.
>
> **For a stage ladder, three things below are superseded**, and § 5.3 plus
> [`project-layout.md`](?doc=execution/project-layout.md) § 1.6 are the current
> answer:
>
> | here | for a ladder, today |
> |---|---|
> | job directory `point-<name>/` | **`<seq>_<name>/`** — `01_coarse`, `02_tight` (`project-layout.md` § 4.1) |
> | carry as a **dangling symlink**, localized to a copy at run time by the wrapper | **a real copy, made at `prep`, from the attempt you name with `--from`.** Nothing points at a file that does not exist yet, so there is nothing to localize |
> | one `submit` hands the whole chain to the scheduler with `--dependency` | **one stage at a time**; `--chain` to do it anyway |
>
> The chaining machinery itself stays — it is the right answer for a sweep, and
> for anyone who wants a chain with their eyes open. What changed is what the
> **staged-science producer** builds, not what `jobset` can do.

`prep` turns the flat bundle into the materialized tree the scheduler runs. Two
ideas make it safe and small:

- **Wrappers are rendered once, from the real input file.** Each distinct
  `script` gets its `.run.sh` / `.sbatch` built one time in the bundle root, by
  the *same* single-job wrapper builder — so a batch job's wrapper is
  byte-identical to a hand-run one.
- **Shared and carried files are symlinked, not copied.** Each job folder links
  back to the shared files and to its carried restart files in the *producer's*
  folder.

```
bundle/
├── job-set.json
├── STAGE-PLAN.md                 ← human-readable plan, written at prep
├── bdt_stage1.run.sh  bdt_stage1.sbatch   ← wrappers, rendered once
├── bdt_stage2.run.sh  bdt_stage2.sbatch
├── Au.psml  S.psml  …  mb_monitor.py      ← the shared files (stored once)
├── point-stage1/
│   ├── bdt_stage1.fdf → ../bdt_stage1.fdf
│   ├── Au.psml → ../Au.psml   …           ← shared, symlinked in
│   └── (stage 1 writes bdt.XV / bdt.DM here when it runs)
└── point-stage2/
    ├── bdt_stage2.fdf → ../bdt_stage2.fdf
    ├── bdt.XV → ../point-stage1/bdt.XV     ← carried (dangling until stage 1 runs)
    └── bdt.DM → ../point-stage1/bdt.DM
```

The carry symlinks are **deliberately dangling** until stage 1 actually produces
those files. And there is a subtle safety step at run time: before stage 2
starts, its wrapper **replaces the inherited symlink with a real local copy**.
Without that, stage 2 writing to `bdt.XV` would write *through* the link and
overwrite stage 1's result — the "localize-on-run" rule closes that trap.

```mermaid
flowchart LR
    S1["point-stage1/<br/>runs → writes bdt.XV, bdt.DM"]
    L["point-stage2/<br/>bdt.XV, bdt.DM as symlinks<br/>→ stage1's files"]
    C["at run time: copy them<br/>local, THEN run stage 2<br/>(so stage 2 never clobbers stage 1)"]
    S1 --> L --> C
```

### 5.3 The execution loop — one grammar, one stage at a time

> **This section is the authority for what you type.** `project-layout.md` § 1.6
> owns *what happens on disk*; this owns *the commands*. Where § 5.2 above still
> describes a submitted chain with dangling carry links, § 5.2 is the older
> model — see the note at its head.

#### The grammar

```
molbuilder jobset <verb> <kind> [<stage>]  [options]
                    │      │        │
                    │      │        └─ which stage — by its NAME (`tight`), its
                    │      │           NUMBER (`3`, or `03`), or the whole token
                    │      │           (`03_tight`), whichever you have in front
                    │      │           of you.  All three reach one resolver
                    │      └────────── what is being prepped or submitted:
                    │                  `run` (the calculation) or `bench`
                    │                  (the measurement of it)
                    └───────────────── describe · prep · submit · summarize
                                       · status · plan
```

**A number here is the stage's `seq`, never its row.** With stage 2 disabled the
ladder is `01_coarse` and `03_tight`, so `3` means *tight* and there is no `2`
to type — the same number you see in the directory, in the deck's filename, and
in the `seq` column of `plan` and `status`. That is what
[`engines/stages.md`](?doc=engines/stages.md) R5 is protecting: a
position shifts when the ladder changes, and an assigned ordinal does not.

A **sweep** has no ordinals — its points are independent and have no order — so
its points resolve by name, and a refusal there does not offer you numbers it
does not have.

`describe`, `status` and `plan` take no *kind* — they are about the calculation,
not about one run of it. **The kind is a positional, not a `--bench` flag**, because
`prep bench` and `prep run` are peers: measuring and running are the same act
over different parameters (`project-layout.md § 2.3.1a`).

> **What of this grammar runs today**, checked against the CLI on 2026-08-10.
> The grammar is the target; three of its cells are not built, and a reader
> should not have to discover that by typing.
>
> | | `run` | `bench` | no kind |
> |---|:--:|:--:|:--:|
> | `prep` | ✅ | ⛔ `molbuilder bench generate` + `bench prep` | ✅ (lays out every container) |
> | `submit` | ✅ | ⛔ `molbuilder bench siesta-gpu` | — |
> | `summarize` | — | ⛔ `molbuilder bench summarize` | — |
> | `describe` | — | — | ⛔ **not built** — today `molbuilder fdf … --stages-json --jobset` writes the bundle, and it emits *both* shapes at once |
> | `status` | — | — | ✅ whole calculation · ⛔ **no per-stage form yet** |
> | `plan` | — | — | ✅ |
>
> `prep|submit bench` refuse with a pointer at the bench command that works,
> rather than reporting an unknown word — the fold-in is designed
> (`web/staged-runs-architecture.md` step 1c), not done. **`status <stage>` is
> the one that misleads**: `status` takes the folder as its positional, so
> `jobset status tight` reports *"Directory 'tight' does not exist"* rather than
> saying it wants a folder. That is a defect of this section's own making, and
> it is why the per-stage row is marked rather than quietly shown working in the
> examples below.

#### Three ideas, in plain language

**1. A stage at a time — because you are meant to look in between.** A ladder is
not a pipeline. You run `coarse`, you *look* at what it produced, and only then
do you set up `tight`. So `submit run` names one stage, and running the whole
ladder unattended needs `--chain`, said out loud. The reason is money and time,
not tidiness: a chain that continues on its own can spend a week refining a
geometry you would have rejected in a minute (`project-layout.md § 1.6`).

A **sweep** is the opposite and needs no flag: its points are independent, so
submitting all of them is the ordinary thing.

**2. What a stage continues from is something you say.** `--from
01_coarse/run-0` names the attempt whose results this run starts from. Those
files are **copied** into the new attempt, not linked — the engine writes to
those very filenames, and writing through a link would destroy the result you
started from. `--cold` means *start clean*, which with a directory per attempt is
simply **skip the copy**; there is nothing to move aside.

Continuing from `run-0` and from `run-2` are different scientific choices, so
molbuilder does not guess between them.

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

> `--mode` is **required** today. `molbuilder.json` already carries an
> `execution.mode`, and `runtime_config.get_execution()` reads, validates and
> returns it — but **only `bench` consults it**, and wiring `submit` to fall
> back on it (flag, then config, then detected scheduler) is recorded work
> rather than shipped behaviour.
>
> ⚠ **That key has no live contract.** It is validated by code and cited
> throughout `molbuilder/bench/` as *"job-execution.md § 8.13"* — a document
> **retired in the 2026-07 migration** (`audit-2026-07-28-document-migration.md`
> maps it to `execution/running-a-job.md`, whose section numbers did not
> survive). So `execution` is a config section the code enforces and no live
> document defines. Writing that contract is the first half of wiring `submit`
> to it; you cannot resolve against a rule nobody has written down.

#### The loop

```mermaid
flowchart TD
    D["<b>describe</b><br/>the portable package:<br/>template · task.json · shape"]
    PB["<b>prep bench</b> &lt;stage&gt;<br/>build the measurement"]
    SB["<b>submit bench</b> &lt;stage&gt;<br/>measure this machine"]
    SM["<b>summarize bench</b> &lt;stage&gt;<br/>write the verdict"]
    PR["<b>prep run</b> &lt;stage&gt; --from &lt;attempt&gt;<br/>render the deck · make run-n<br/>· COPY the warm files in"]
    SR["<b>submit run</b> &lt;stage&gt;<br/>--mode direct | submit"]
    L["<b>look</b><br/>status · the trajectory · the forces"]
    D --> PR
    D -.optional.-> PB --> SB --> SM -.verdict.-> PR
    PR --> SR --> L
    L -->|"good — next stage"| PR
    L -->|"not good — retry differently"| PR
```

**`prep` prints what it resolved, which is what makes `submit` a plain yes.** It
is the only place the measured numbers, the chosen starting geometry and the
rendered deck appear together — exactly where a person should be looking before
committing cluster time.

#### Examples

A two-stage relaxation on a **workstation**, `shape: hierarchical`:

```bash
molbuilder jobset prep   run coarse                  # 01_coarse/run-0, nothing carried in
molbuilder jobset submit run coarse --mode direct    # runs here, locally
molbuilder jobset status                             # look before deciding

molbuilder jobset prep   run tight --from 01_coarse/run-0
#   reading from 01_coarse/run-0  (finished, converged)
#   02_tight/<label>_02_tight.fdf   rendered   BlockSize 256
#   02_tight/run-0/                 ready      copied in: <label>.XV  <label>.DM
molbuilder jobset submit run tight --mode direct
```

The same calculation on a **cluster** — same words, different channel:

```bash
molbuilder jobset submit run tight --mode submit --domain public --dry-run
molbuilder jobset submit run tight --mode submit --domain public
```

Redoing a stage differently — a new attempt, and `run-0` is untouched:

```bash
molbuilder jobset prep   run tight --from 01_coarse/run-0   # -> 02_tight/run-1
molbuilder jobset prep   run tight --cold                   # -> a clean attempt
```

And the whole ladder unattended, when you have decided you want that:

```bash
molbuilder jobset submit run --chain --mode direct
```

#### The read-only verbs

```bash
molbuilder jobset plan     ./bundle    # the jobs, resources and carry set — changes nothing
molbuilder jobset status   ./bundle    # per-stage state + which stage is next
molbuilder jobset status              # the same, from inside the folder
```

⛔ A per-stage `status <stage>` is in the grammar and **not built** — see the
table above. Both verbs still take the folder as their positional, so they have
not moved to `--bundle` the way `prep` and `submit` have; that inconsistency is
real and is part of decision 28's caller sweep.

- **`plan`** prints the chain, each job's resources, and its carry set. It
  changes nothing — it is the "look before you leap" step.
- **`submit`** takes a **required** `--mode`:
  - **`submit`** hands each job to SLURM, threading the dependency
    (`--dependency=afterany:<stage1-job-id>` for the example) by capturing each
    `sbatch`'s returned job id and feeding it to the next.
  - **`direct`** runs the jobs in order **locally** (`bash …run.sh`), reproducing
    the same dependency meaning without a scheduler: an `afterok` job whose parent
    failed is skipped along with everything below it; an `afterany` job runs
    regardless. This is the workstation path.
  - `--dry-run` prints the exact command it *would* run per job, without
    launching — the safe way to see what will happen.
- **`status`** reads the run directory (reusing the run decoder from
  [`running-a-job.md § 4.2`](?doc=execution/running-a-job.md)) and reports each
  stage's state, its restart files, and the first incomplete stage — then stops.
  It prints resume guidance but **never auto-resumes** (design decision #5); you
  re-submit the incomplete stage yourself, and the engine warm-starts from its
  own `.XV`.

### 5.4 The dependency chain, as a sequence

For the 2-stage ladder, `--mode submit` does this:

```mermaid
sequenceDiagram
    participant U as you
    participant J as jobset submit
    participant S as SLURM
    U->>J: submit --mode submit --domain public
    J->>S: sbatch -J stage1 -p public -q public … stage1.sbatch
    S-->>J: Submitted job 4021
    J->>S: sbatch --dependency=afterany:4021 -J stage2 … stage2.sbatch
    S-->>J: Submitted job 4022
    Note over S: 4022 stays PENDING until 4021 finishes<br/>(afterany — stage1's policy is proceed),<br/>then runs, reusing stage1's carried .XV/.DM
```

### 5.5 Watching a stage while it runs

`status` is a roll-up you pull on demand. To watch a *single* stage live, point
the run viewer (the web run view, or `molbuilder watch`) at its
`point-<name>/` folder — it resolves and streams the trajectory exactly as for a
stand-alone job ([`running-a-job.md § 4`](?doc=execution/running-a-job.md)).
Every job also carries **`mb_monitor.py`** (one of the `shared` files, symlinked
into each folder): its wrapper launches it in the background to sample GPU/CPU
utilisation into a `.util.csv` while the stage runs, so an under-utilised or
stalled stage is visible without waiting for it to finish.

Because each `point-<name>/` is a self-contained run directory, it can also be
**checkpointed on its own** — `molbuilder snapshot` treats a stage folder like
any other run dir ([`running-a-job.md § 6`](?doc=execution/running-a-job.md)), so
you can tag a converged stage or branch a what-if from it independently of the
rest of the ladder.

---

## 6. Running on a cluster — SLURM deployment

The wrapper file shapes (`.run.sh` inner + `.sbatch` outer) and the meaning of
each `#SBATCH` line are owned by
[`running-a-job.md § 5.3`](?doc=execution/running-a-job.md) and
[`job-contracts.md § 2.6`](?doc=execution/job-contracts.md). What the **job
system** adds is submission, dependency threading, and routing:

- **The two layers.** The outer `.sbatch` is a thin `#SBATCH` header whose body
  is a single line — `bash <base>.run.sh "$@"`. The inner `.run.sh` owns
  activation and launch. You submit the outer file; it hands off to the inner
  one. This split means the scheduler header and the run logic evolve
  independently, and the exact same `.run.sh` works with or without a scheduler.
- **One `sbatch` per job, per-job flags win.** The submitter passes each job's
  resources as command-line `sbatch` flags (`-J`, `-n`, `-c`, `--gres`, `-t`,
  `--exclusive`), which **override** the rendered header — so a whole sweep can
  share one `.sbatch` file while each point still gets its own ranks and cores.
- **Routing domains.** Instead of hard-coding a partition, you name a **domain**
  (`--domain public`, or `execution.domain` in config). A domain is a friendly
  name for a `(partition, qos)` pair (with an optional separate GPU partition);
  `submit` resolves it and refuses an unknown name with the list of configured
  ones. Partition and qos are **required** for a SLURM site — the framework
  refuses to emit a header it knows will be rejected (design decision #4).
- **Job names read well.** A ladder job's `-J` is its bare stage name (`stage2`);
  a benchmark point's is `job-gpu-G1K2C5` / `job-cpu`, so a `squeue` listing is
  self-describing.

A workstation with no `scheduler` block configured simply gets `.run.sh` files
and runs with `--mode direct`; the `.sbatch` is emitted only when a scheduler is
configured.

---

## 7. Benchmarking — measuring the fastest resources

Before you commit a long production run to a node, you want to know: for *this*
calculation on *this* machine, how many GPUs, how many MPI ranks per GPU, and how
many CPU cores per rank actually run it fastest? Guessing wastes allocation. The
benchmark workflow measures it, and it is just the job system pointed at a
resource grid.

Its guiding idea (**target isolation**, design decision #3): you generate a
benchmark *bundle* on your laptop; everything machine-specific is discovered on
the target. Five steps: `generate` on the host (`molbuilder bench generate`),
then the bundle's own baked **`prep-bench`** and **`run-bench`** executables on
the target (they self-bootstrap molbuilder and shim to the CLI), then
`summarize` and `prep-run` back under `molbuilder bench`:

```mermaid
flowchart LR
    G["generate<br/>(host)<br/>one .fdf → a bundle"]
    P["prep-bench<br/>(target)<br/>detect the machine →<br/>environment.json + the real grid + BENCH-PLAN.md"]
    R["run-bench<br/>CPU baseline + GPU grid<br/>(self-bootstraps molbuilder)"]
    S["summarize<br/>every point → bench-result.json<br/>(the winner + a recommendation)"]
    PR["prep-run<br/>the winner → run-production.sh<br/>(re-resolved for this machine)"]
    G --> P --> R --> S --> PR
```

- **Detect the machine → `environment.json`** (`molbuilder/environment@1`): the
  scheduler, the site, and the topology — resolved to the **compute node's**
  real core and GPU counts (read from the scheduler via `scontrol show node`,
  not from whatever login node you happen to be on), so the numbers are the ones
  the job will actually run against.
- **Two comparable points → `bench-manifest.json`** (`molbuilder/bench-manifest@2`).
  This is the clever bit: to compare CPU vs GPU **hardware** fairly, both points
  use the *same solver* (`ELPA-1STAGE`) and run in the *same environment*
  (`molbuilder-siesta-gpu`); the only difference is one flag toggling the GPU on.
  Both are trimmed to 5 SCF iterations with convergence checks off — you are
  timing the machine, not converging the chemistry. (The CPU point sets the GPU
  flag *explicitly off*, because the GPU-linked build defaults to on.)
- **The `(G, K, c)` grid** — the shape of the sweep, and why: `G` = number of
  GPUs (1 up to the node's count); `K` = MPI ranks per GPU, tried at the divisors
  of the socket's core count (so ranks divide evenly); `c` = CPU cores per rank,
  tried as a **starved / one-socket / cross-socket** triple
  (`{1, cores//K, 2·cores//K}`) to bracket the useful range. Each point runs in
  its own `point-G<g>K<k>C<c>/` folder.
- **Measure → `bench-result.json`** (`molbuilder/bench-result@1`): each point is
  parsed for its SCF wall-time **per iteration** (averaged over iterations 3–5 to
  skip warm-up), plus a utilisation reading and peak memory. The **winner is the
  fastest completed point**; the tool also recommends a memory request (peak ×
  1.15) and a walltime (per-iteration time × a nominal iteration count × a safety
  factor). The recorded choice is **portable** — `prep-run` re-resolves the
  concrete rank and core counts for whatever machine it is later run on.

`molbuilder bench probe-scheduler` is a companion that reads a live cluster
(`sinfo`/`sacctmgr`) and proposes a `scheduler` block + routing menu you can
merge into your config with `--write`.

> **One honest gap.** The benchmark already
> *produces* a `JobSet` (`bench prep` writes `job-set.json`), but it still
> *executes* through its original, proven inline-shell sweep rather than
> `jobset submit`. Both paths use the identical `(G,K,c)` grid, so results match;
> retiring the inline path once it is cluster-validated is the open follow-up.

---

## 8. Where it stands, and where it is going

### Shipped today (command line)

The `JobSet` model and persistence; the SIESTA ladder producer (`fdf --jobset`)
and the benchmark sweep producer; all four verbs (`plan` / `prep` / `submit` /
`status`) in both `submit` and `direct` modes; SLURM submission with dependency
chains and routing domains; and the full benchmark workflow. Saving and
re-entering a calculation's states is `molbuilder snapshot`
([`running-a-job.md § 6`](?doc=execution/running-a-job.md)).

### Not built yet — the web, and other engines

This is the migration the project is undertaking, planned in
[`roadmap.md`](?doc=roadmap.md) (workstream 1, "Batch execution reaches the
web"). Today there is **no `jobset` web blueprint, no `/api/jobset/*` route**,
and the web Build tab still renders a single `.fdf` (it drops the stage table).

- **Phase 1 — a web bundle producer.** The Build tab's stage table becomes a real
  `JobSet` producer, calling the same `build_siesta_stage_bundle` seam (§ 4.1).
- **Phase 2 — web Plan + Status (read-only).** Reusing the *already-shipped*
  run decoder in the browser, with no new parser. **A branch control was
  planned here and is not needed**: the checkpoint rework removed the verb, and
  forking is restore-then-save — both already routed and both already in the
  sidebar panel ([`checkpointing.md`](?doc=execution/checkpointing.md) § 7.1).
- **Phases 3–4 — other engines, gated on a cluster-validation milestone.**
  `transport --jobset`, `pyscf --jobset`, and `spectra --jobset` producers (with
  their tab mirrors), behind a hard gate: prove the SIESTA ladder end-to-end
  (produce → prep → submit → monitor) on a real cluster *before* broadening.
  Reaching transport is also where the single-parent limit (§ 2, design decision
  #6) is lifted to a branching graph.

Also out of scope for now: **multi-node MPI** (v1 fixes one node), a
`molbuilder config init --site` command (a site preset ships only as a JSON
example file today), and wiring the notifier hook to a real messaging service
(a proof-of-concept stub exists).

The through-line: the CLI framework on this page is the settled foundation, and
the web work is *additive on top of it* — it reuses these exact producers,
decoders, and wrappers rather than reinventing them.

---

## 9. A developer's map

Where each responsibility lives, for someone extending the framework:

| Concern | Module |
|---|---|
| The data model + `job-set.json` read/write + `validate()` | `molbuilder/jobset/model.py` |
| SIESTA ladder producer + the pure `build_siesta_stage_bundle` seam | `molbuilder/siesta/stages.py` |
| Benchmark sweep producer | `molbuilder/bench/to_jobset.py` |
| Lay out the materialized tree (`point-*/` + symlinks) | `molbuilder/jobset/materialize.py` |
| Render wrappers once + carry-in + `STAGE-PLAN.md` | `molbuilder/jobset/prep.py` |
| The human-readable plan table | `molbuilder/jobset/plan.py` |
| Submit (SLURM chain / direct) + dependency threading + domain routing | `molbuilder/jobset/submit.py` |
| Per-stage status roll-up (reuses `decode_run_dir`) | `molbuilder/jobset/runstatus.py` |
| The CLI verbs (`molbuilder jobset …`) | `molbuilder/jobset/_cli.py` |
| The `.sbatch` header emission (shared with single-job) | `molbuilder/runwrap.py::render_sbatch` |
| The benchmark workflow (detect / manifest / grid / summarize) | `molbuilder/bench/*` |

**To add a new engine to the job system**, you write **one producer** —
a function that turns that engine's config into a `JobSet` — and nothing else.
The verbs, the materializer, the submitter, and the status layer are all
engine-agnostic and pick it up for free. That is the whole payoff of decision #1.
