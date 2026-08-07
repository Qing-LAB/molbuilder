# The project directory — what lives where, and who puts it there

**Role:** contract
**Domain:** execution
**Companions:** [`execution/job-contracts.md`](?doc=execution/job-contracts.md)
— the run directory's own rules, the topic list, the filename conventions;
[`execution/job-system.md`](?doc=execution/job-system.md) — the JobSet the
per-stage folders come from; [`engines/stages.md`](?doc=engines/stages.md) — what
a stage is and how its deck is produced;
[`execution/checkpointing.md`](?doc=execution/checkpointing.md) — what the saved
history must always hold;
[`execution/run-identity.md`](?doc=execution/run-identity.md) — the id a
calculation's files share.

**Status: mostly shipped, described here for the first time as one picture.**
Every level below exists in code today except `stages.json`, its reader, and the
stage-directory naming in § 4. What this document adds is the *whole*: which
directory owns what, how parameter tuning and resource tuning nest, and where the
saved history sits.

**This contract owns:** the levels of the tree, who may write at each one, how
each level is named, and the invariants that hold across them. It does not
restate the rules inside a single run directory — those are `job-contracts.md`.

---

## 1. The shape

Six levels, and each one means something different.

```
projects/                                   the root (git-ignored)
└── BDT-Au/                                 ① a PROJECT — a body of work
    ├── structure/                          ② a STORAGE topic
    │   └── bdt_au.xyz  bdt_au.molstruct.json
    ├── pseudopotential/                    ② a STORAGE topic
    │   └── Au.psml  S.psml  C.psml  H.psml
    └── optimization/                       ② a RUN topic — one of nine fixed names
        └── bdt_au_relax_c6h4s2au38/        ③ a CALCULATION — one system, one identity
            │                                 ── written by the browser, and
            │                                    naming no machine anywhere ──
            ├── <id>.template.fdf             the science backbone: everything
            │                                 fixed, nothing a stage varies and
            │                                 nothing the hardware decides
            ├── stages.json                   what each stage tunes + resource intent
            ├── Au.psml  S.psml               the shared package, stored once
            ├── mb_monitor.py
            ├── .mbcheckpoint.json
            ├── .git/  .binsnapshots/         the saved history (§ 6)
            │
            ├── 01_coarse/                  ④ a STAGE — written by `prep`, on the
            │   │                              machine that will run it
            │   ├── <id>.fdf                  THE DECK: template ⊕ coarse's values
            │   │                             ⊕ this machine's BlockSize, solver, ranks
            │   ├── <id>.run.sh  .sbatch      its wrapper, for this machine
            │   ├── Au.psml → ../             shared, linked up
            │   ├── run-0/                  ⑤ an ATTEMPT — a RUN, immutable
            │   │   ├── <id>.fdf → ../          the deck, linked down
            │   │   ├── run.json                how it was launched, what it continued from
            │   │   └── <id>.XV .DM .out        everything this attempt produced
            │   ├── run-1/                    a redo, continuing from run-0
            │   └── bench/                    a BENCHMARK — its own container
            │       ├── job-gpu.fdf  job-cpu.fdf    the science, made measurable
            │       ├── bench-manifest.json  bench-result.json
            │       └── point-G1K2C5/         its own runs
            │
            └── 02_tight/
                └── run-0/
                    └── <id>.XV                     copied from 01_coarse/run-0/
```

**③ is a calculation** — one system studied one way. One identity, one saved
history, one description. Everything above it is filing.

### 1.1 A directory is a container or a run, never both

Every directory in this tree is one of two things:

- a **container** — it holds setup and other directories. Decks, wrappers, the
  description, links, the shared package. All text or small.
- a **run** — one invocation of the engine. It holds what that invocation
  produced, and nothing else holds that.

A calculation is a container. A stage is a container. A benchmark bundle is a
container. The leaves — `run-N/`, `point-<knobs>/` — are runs.

**This is what makes everything downstream simple.** Setup is text, so git handles
it entirely. Only a run holds anything big. There is no directory where the two
are mixed and something has to tell them apart.

> **A simple run stays simple.** A plain run directory does not grow a `run-0/`.
> It **is** a run — a leaf with no container above it — which is exactly what a
> hand-made directory with one `.fdf` in it already is. Nothing about the
> straightforward case changes.

### 1.2 An attempt is immutable

**A run directory is written once and never modified.** Running a stage a second
time — a `--continue`, a redo after a change — makes `run-1`, carrying what it
needs from `run-0` and leaving `run-0` exactly as it was.

You say which attempt it continues from, and its files are copied in — the same
explicit step you take when moving from one stage to the next (§ 1.3).

Three things follow:

- **Warm restart becomes explicit.** Today "continue" means *the files happen to
  be in this directory*. With one directory per attempt it means *carry from the
  previous attempt* — visible on disk rather than implied by what is lying
  around.
- **The saved history becomes append-only.** No archived file ever changes, so a
  new save point only has to store the attempts that appeared since the last one
  (§ 6).
- **`--force` is retired.** It exists to reset the run index to `-run0` and
  overwrite it (`job-contracts.md § 2.6`). With a directory per attempt there is
  nothing to overwrite: a redo is `run-2`. A flag whose only purpose is to
  destroy a previous result has no place once results cannot collide.

Immutability is a contract, not a filesystem permission — but it is **checkable**,
and § 7 makes it an invariant: an attempt that has been saved must never differ
afterwards. Nothing would notice today.

#### Where a run happens

**Inside the attempt directory**, which was created and filled when you prepared
the stage (§ 1.3, and § 2.1 step 4). By the time anything is launched it already
holds its inputs. The wrapper is invoked there; it activates and execs, and every
later line of it — the launch, the monitor, the SCF tee, the failure hints —
works relative to the current directory, so everything lands in the attempt with
no change to the wrapper at all.

```
prepare  →  01_coarse/run-0/ exists, deck linked, inputs copied
submit   →  launched there
```

Launching in a chosen directory is what `jobset submit` already does one level
up: `subprocess.run(cmd, cwd=<job dir>)` for the local path, and `sbatch` from
the same place for SLURM, which lands the job in `SLURM_SUBMIT_DIR`. Pointing it
at the attempt instead of the container is a change in the caller, not in the
wrapper.

**Everything the attempt needs is put there before the wrapper starts**, and all
of it is in place before the engine sees the directory:

| | How | Why |
|---|---|---|
| the deck, the monitor, the pseudopotentials | **linked** from the container | five attempts share one copy of each |
| whatever this run continues from | **copied** | that run has already finished — you looked at it and chose it — and a link would let this engine write back over it |
| everything the run writes | created in place | it is already the working directory |

**A flat run directory is untouched.** It *is* a run (§ 1.1), so `bash job.run.sh`
in it behaves exactly as it does today.

> ⚠ **A first attempt at this was built in the wrong place** (2026-08-06,
> `runwrap.py`'s `attempt_dirs` prologue): ~50 lines of shell that scanned for
> run directories, created one, symlinked the deck and package in, and copied
> warm files. That is `jobset/materialize.py` rewritten in bash, one level down,
> inside the one layer kept free of filesystem logic
> (`running-a-job.md § 2.2a`). It is to be retired, not extended — including the
> guard it needed against being run from inside an attempt, which stops being a
> hazard once the caller decides the directory.

### 1.3 Stages do not chain, and what that simplifies

**Each stage is prepped and submitted on its own.** Nothing links coarse to
tight; no scheduler dependency, no queued follow-on, no automatic hand-off of
files. When coarse finishes you look at what it produced, decide, and then set up
tight.

That is § 1's rule — *this framework writes correct files; it does not run
things* — applied to the one place it was easiest to forget. It is also the only
sane default when **a stage is a long job**: a chain that continues on its own
can spend a week computing from a geometry you would have rejected in a minute.

> **The decision to continue is the user's, made after looking. molbuilder's job
> is to make continuing correct once that decision is taken.**

#### Who makes the attempt directory

**Python, when you prepare the stage** — step 4 of § 2.1, not when you submit and
not by the wrapper. By the time anything is launched the directory already exists
and already holds its inputs; the wrapper activates the environment and execs the
engine (`running-a-job.md § 2.2a`).

Preparing does five things: **resolve** the next `run-<n>` (highest plus one, or
`run-0`); **create** it; **link** the deck, the monitor and the shared package in;
**copy** whatever this run continues from; and **report** what it did, so you can
read it before committing a week of cluster time.

That last one is why this belongs to prepare rather than submit. Preparing is
still design — you are arranging files and can look at the result — and the split
between preparing and starting is what gives you somewhere to look. Submitting is
then a plain "yes, that one."

That is also where the work already lives: `jobset/materialize.py` creates job
directories and lays relative symlinks. Doing it one level down is the same
operation in the same module.

**Preparing again is safe until the run has been launched.** Otherwise splitting
the two steps leaks directories — prepare, change your mind, prepare again, and
an empty `run-3` sits there forever. A run becomes untouchable once it has *run*
(§ 1.2); before that, re-preparing is just changing your mind about the setup,
which is the entire reason the step is separate.

#### The one file the split needs

*Has this been launched?* has no honest answer from the directory alone. A queued
cluster job has produced nothing yet, so "no output" and "not started" look
identical — and re-preparing would quietly rewrite the setup under a job already
in the queue.

So **submitting writes one small file into the attempt**, `run.json`
(`molbuilder/run-launch@1`): how it was launched, the exact command, the
scheduler's job id, when, and **what it continued from**.

It earns its place three times over: preparing reads it and refuses to reuse a
launched attempt; `status` can say *queued as job 481923* instead of guessing
from an absence; and the last field is the run's provenance — *this geometry came
from `01_coarse/run-0`* — which is worth recording whether or not anything reads
it back. Nothing persists a job id today; `submit_jobset` returns one and the CLI
prints it.

It is written **after** the launch succeeds, so a failed submission leaves the
attempt exactly as prepare left it — still safe to prepare again.

#### Continuing from an earlier run is a copy, not a link

Because stages are set up one at a time, **the run you continue from has already
finished** — that is what you just looked at. Its files are sitting on disk when
you prep the next stage, so they are **copied**, as real files, then and there.

```
02_tight/run-0/<id>.XV     a real copy of 01_coarse/run-0/<id>.XV
```

Copied and not linked for the same reason as always: the engine writes to that
filename, and writing through a link would destroy the result you started from.

**Which run you continue from is something you say, not something molbuilder
guesses.** Continuing from `01_coarse/run-0` and continuing from `01_coarse/run-2`
are different scientific choices, and the folder names make the choice visible
afterwards.

This is the same shape one level down: a redo of a stage — `run-1` after
`run-0` — copies from the attempt you name, for the same reason.

> **What this removes.** Nothing has to point at a file that does not exist yet,
> so there are no dangling links to resolve, no question of *which attempt will
> the producer use*, and nothing to swap at run time. Those problems only arise
> when a whole chain is submitted at once, which is exactly what this does not
> do. A design that never creates the problem beats one that solves it.

#### What is not touched

**The flat case.** A plain run directory *is* a run (§ 1.1), so `bash job.run.sh`
in it behaves exactly as it does today.

**The shipped chained ladder.** `jobset` can thread SLURM dependencies and carry
files between queued jobs, and `stages_to_jobset` currently builds exactly that
(`depends_on=prev.name`, plus `.XV`/`.DM`/`.CG` carries). That machinery stays —
it is the right answer for a benchmark sweep, and for anyone who wants a chain
with their eyes open. **The staged-science producer stops emitting one**, which
is a change to what it builds, not a removal of what `jobset` can do.

#### `--cold`, and running a stage by hand

`--cold` means *start this run clean*. Today it moves warm files aside because
the run directory is reused; with a directory per attempt there is nothing to
move — a fresh attempt is empty unless something is copied in. So `--cold`
becomes simply *skip the copy*, and it moves to whatever command starts the run.
The move-aside machinery, and its `job-contracts.md § 4` requirement that the
glob cover every file the warm branch reads, stay as they are **for the flat
case**, which still reuses one directory.

> **Open** (§ 8): what you type to set up and start one stage. It has to name the
> stage, optionally name the run to continue from, and accept `--cold`. It must
> exist before the wrapper's directory-making prologue is retired, or the manual
> path breaks with nothing behind it.

---

## 2. Who does what — the workflow, and each level's owner

### 2.1 The browser writes a portable package; the terminal makes it runnable

The split is not "design versus execution". It is **what a laptop can know versus
what only the target machine can**.

**The browser writes four things, and none of them mention a machine:**

| | What | Why it is portable |
|---|---|---|
| the **data files** | pseudopotentials, the structure | they are the same everywhere |
| the **deck template** | the science backbone — everything the calculation fixes and no stage varies | it is physics |
| **`stages.json`** | the variables each stage tunes | it is the mission |
| the **resource intent** | *use a GPU · this is a big job · aim for this scale* | a wish, not a number |

**The terminal — `prep`, on the machine that will run it — turns that into a
runnable directory**, because only there do you know:

- workstation or SLURM;
- how conda or mamba is installed here, and whether activation is
  `conda activate` or `source activate` (`molbuilder.json`);
- what the hardware actually is, and what a benchmark measured on it.

Then it **renders the final deck** — template ⊕ this stage's variables ⊕ this
machine's resolved parameters — writes the wrapper, builds the run directory, and
brings in whatever you told it to continue from.

### 2.2 Why the deck cannot be finished in the browser

Because some of what goes *inside* the deck is a fact about the machine.

`BlockSize` is the clearest case: `siesta/input.py`'s `_auto_block_size(n_atoms,
mpi_np, gpu_mode)` derives it from **the rank count and whether there is a GPU**,
and the answer is written into the `.fdf`. The eigensolver is another — ScaLAPACK
or ELPA changes both the deck *and* which conda environment the wrapper
activates. A deck rendered on a laptop is either wrong for the cluster or a
guess.

So the parent holds a **template**, and the final `.fdf` is produced where its
last unknowns are known. This is the same shape the benchmark already ships:
`bench prep` runs on the target, detects the machine, writes `environment.json`,
and formats the scripts for it — *"the user never hand-edits a queue name or a
core count; this is what makes the bundle portable."* The staged path reuses that
shape rather than inventing a second one.

> **A folder that carries no machine knowledge can be copied to any machine. A
> folder whose decks were finished on a laptop can only be copied to that
> laptop.**

### 2.3 `prep` is the hub, not step four of a line

This is the part a linear list gets wrong. **You come back to `prep` every time,
and it is where every join is made:**

```mermaid
flowchart LR
    UI["<b>browser</b><br/>data files · deck template<br/>stages.json · resource intent"]
    P{"<b>prep</b><br/>on the target machine"}
    B["benchmark<br/>runs"]
    R["a run<br/>runs"]

    UI --> P
    P -->|"--bench"| B
    B -->|"the measured answer"| P
    P -->|"a run directory"| R
    R -->|"its results, and what you learned"| P
```

Four different jobs, one verb, because they are the same act — *assemble a
runnable directory from a template, a source of earlier results, and this
machine's parameters*:

| You are doing | You give `prep` |
|---|---|
| measuring, before committing | the stage, and *benchmark this* |
| the real run, using what you measured | the stage, and the benchmark result |
| a redo of that run | the stage, and `run-0` to continue from |
| the next stage | that stage, and the previous stage's run to continue from |

**Nothing distinguishes those four in the machinery.** "Continue from `run-0`"
and "continue from `01_coarse/run-0`" are the same instruction pointing at
different directories; "use this benchmark result" is the same kind of input as
"use this geometry". That is why it is one command with arguments rather than
four commands.

And it is where **you** are in the loop. Every arrow back into `prep` is a
decision made after looking at what came out — which is the whole reason stages
do not chain (§ 1.3).

### 2.4 The steps, and which surface

| | Step | Surface |
|---|---|---|
| 1 | save the structure into the tree | **browser** |
| 2 | describe the calculation — the template and `stages.json` | **browser** |
| 3 | write the portable package into the calculation folder | **browser** |
| 4 | **`prep`** — resolve this machine, render the deck and wrapper, build the run directory | **CLI** |
| 5 | submit or execute | **CLI** |
| 6 | look at what happened; save a checkpoint | CLI, with the browser for viewing results |
| ↻ | back to 4, for a benchmark, a redo, or the next stage | |

**Every step has a CLI equivalent** — `conventions.md § 3` makes the CLI a thin
shell over the same functions the blueprints call, so a user with no browser can
do 1–3 from a terminal. Steps 4 and 5 have no browser equivalent, and that is the
real boundary rather than a gap: they need the target machine.

The save history is set up at step 4 for the same reason everything else is:
which files count as big binaries depends on this machine's copy of the tree, not
on the science.

### 2.5 Who owns each level of the tree

| Level | Named by | Written by | May contain |
|---|---|---|---|
| ① **project** | the user | nobody — it is a folder | topics, nothing else |
| ② **topic** | a **fixed set of nine** (`job-contracts.md § 2.5`) | nobody | calculations (run topics) or files (storage topics) |
| ③ **calculation** | the run id (`run-identity.md § 3`) | **the browser** (step 3), in one transaction | the template, `stages.json`, the shared package, the history |
| ④ **stage** | `<seq>_<name>` (§ 4) | **`prep`** — the rendered deck and wrapper land here | its deck, its wrapper, its attempts — **a container** |
| ⑤ **attempt** | `run-<n>`, unpadded (§ 4.3) | **`prep`** creates and arranges it; the engine then fills it | everything one invocation produced — **a run, immutable** |
| — **benchmark** | `bench` | `prep --bench` | its own decks, wrappers, config and results — a self-contained **container** |
| — **trial** | `point-<knobs>` (§ 4.4) | the sweep script, then the engine | one throwaway **run** |

Three rules, and everything else follows:

> **The browser writes level ③ and nothing else, and writes nothing that names a
> machine. `prep` writes ④ and ⑤. The engine writes the directory it was launched
> in, once, and nothing ever writes there again.**

> **Every directory and every link in this tree is made by Python. The wrapper
> activates an environment and execs an engine, in a directory it was handed**
> (`running-a-job.md § 2.2a`).

The engine never writes above itself — whatever a run continues from is a real
copy put there before it starts (§ 1.3).

**A benchmark gets its own directory, and that is not tidiness.**
`generate_bench_bundle` writes its own decks, wrappers, pseudopotential copies,
`README.md` and `.molbuilder.json` — it owns a directory. Pointed at a stage
directory it would put a second job's inputs beside the real run's, which
`job-contracts.md § 2.1` Rule 1 forbids, and duplicate the pseudopotentials
already linked from the parent. Pointed at `01_coarse/bench/` it needs **no
change at all**: the bundle root moves, everything inside it stays as it ships.

**Storage topics are flat and shared.** `structure/` and `pseudopotential/` hold
files, not calculations. A calculation *points* at a structure and *copies* the
pseudopotentials it needs into its own shared package, so it stays
self-contained when moved to a cluster.

---

## 3. Two kinds of tuning, nested

There are two things a user varies. They vary for different reasons, they need
different machinery, and — this is the part that was implicit — **one nests
inside the other.**

| | **Stage** (④) — parameter tuning | **Trial** (⑥) — resource tuning |
|---|---|---|
| What varies | the science: mesh cutoff, force tolerance, relaxation method, k-grid | the machine: GPUs, MPI ranks, cores per rank |
| Why | to approach an answer in steps — coarse first, then tight | to find out what runs *this* science fastest *here* |
| The deck | **its own file**, rendered from the shared settings with this stage's values substituted | the stage's deck **transformed to be measurable** (§ 3.2) |
| Identity | shares the calculation's id, so it warm-starts from the stage before | **its own throwaway label** — `job-gpu` / `job-cpu` |
| Ordered? | **yes** — each continues the one before | **no** — trials are independent and can all queue at once |
| Outcome | a result you keep | a number; the run is thrown away |
| Produced by | `render_siesta_stage_fdfs` + `stages_to_jobset` | `generate_bench_bundle` + `sweep_to_jobset` |

**Why trials nest under a stage rather than beside the calculation.** The best
rank count depends on the science: mesh cutoff changes the grid, basis size
changes the matrix, and `BlockSize` is derived from ranks and atom count. A
coarse stage and a tight stage can genuinely want different resources, so the
measurement belongs to the stage that was measured.

### 3.1 Why the mechanisms differ

A parameter change alters *what the engine computes*, so it has to be in the file
the engine reads — hence a deck per stage, rendered into that stage's directory
when it is prepped (§ 2.1). A resource change alters *how the work spreads over
hardware*, and the scheduler takes most of that on the command line — which is
what lets a twenty-point sweep share one rendered wrapper instead of writing
twenty.

But **not all of it**, and the exceptions below are exactly why the deck cannot
be finished before the machine is known (§ 2.2).

Three settings are both, and they are the reason neither mechanism is *the*
mechanism:

| Setting | In the deck | Also decides |
|---|---|---|
| `Diag.Algorithm` (ScaLAPACK / ELPA) | yes | which conda environment the wrapper activates — any ELPA variant needs the GPU build (`running-a-job.md § 2.3`) |
| GPU on/off | yes (`Diag.ELPA.GPU`) | the scheduler's `--gres` |
| MPI ranks | **no**, but `BlockSize` is derived from them | the scheduler's `-n`, and the launch |

### 3.2 A trial's deck is the stage's deck, made measurable

A trial does **not** run the stage's deck. `transform_fdf` derives a variant that
can be timed:

- **SCF capped** (5 iterations) and `SCF.MustConverge` **off**, so a capped run
  exits cleanly instead of aborting and reading as a scheduler failure;
- **MD steps zeroed** — a single point, because you are timing an iteration, not
  converging a geometry;
- **cold start forced** (`DM.UseSaveDM .false.`);
- **relabelled** to `job-gpu` / `job-cpu`;
- **one solver for both points** (`ELPA-1STAGE`), the GPU point setting the CUDA
  flag on and the CPU point setting it *explicitly* off — so the number isolates
  the hardware rather than the solver.

**The last two are what make it safe to nest.** A trial that kept the stage's
label and honoured saved state would read the stage's `.XV`/`.DM` and then
overwrite them — a five-iteration throwaway destroying the state the real run
depends on. Relabelling and forcing cold are not artefacts of the benchmark once
having been standalone; they are the reason it can live inside a stage's
directory at all.

> **A trial belongs to the deck it was derived from.** Change the stage's science
> and the measurement no longer applies. The manifest records the source deck's
> hash so a stale answer can be recognised rather than reused.

### 3.3 How the levels compose

```mermaid
flowchart LR
    D["<b>stage 02_tight</b><br/>its deck — the science"] --> T["<b>trials</b><br/>same science, made measurable<br/>bench-G1K2C5 · bench-G2K4C5 · …"]
    T --> W["<b>a number</b><br/>G · ranks · cores · mem · walltime"]
    W --> R["<b>the stage's real run</b><br/>its own deck, those resources"]
    R --> N["<b>stage 03</b><br/>continues from it"]
```

You measure per stage, when the science changes enough to matter. You keep the
answer in the stage directory. The real run then uses it, and the next stage
continues from the real run's state — never from a trial's.

---

## 4. Naming, and why the two levels name differently

### 4.1 A stage carries a sequence number

A stage has two identifiers doing two jobs:

- **`name`** — what the user typed (`coarse`, `tight`). Identifies it. Unique
  within the calculation, matching `[A-Za-z0-9_]+` (`engines/stages.md § 2`).
- **`seq`** — a number assigned when the stage is created. **Orders** it. Never
  reused, never reassigned.

| Where | Form | Example |
|---|---|---|
| stage directory | `<seq>_<name>`, zero-padded to two digits | `01_coarse`, `02_tight` |
| deck | `<id>_<name>.fdf` | `bdt_au_relax_c6h4s2au38_tight.fdf` |
| trajectory log | `<id>-stage<seq>` | `…-stage2.molwatch.log` — the shipped convention, with `N` finally **defined** as `seq` |
| checkpoint tag | `<id>/<name>/<UTC>` | you return to a stage by *name*, not by position |

The deck does not carry the number: names are unique, so it would add nothing,
and a deck's filename is quoted in the wrapper, the log and the outputs.

### 4.2 Numbers are assigned once — stages append

**A `seq` is never changed, so a stage can only be added at the end.**

That is not a restriction imposed here; it is what the calculation already is.
Each stage continues from the state the one before it left. Once stage 2 has run,
"insert something between 1 and 2" is not an insertion — it is a new stage that
happens to be coarser, and it runs from where stage 2 left off. Numbering it `03`
is the truth.

Before anything has run, reordering is free: numbers are assigned when the
directories are produced, not when the rows are typed.

**Gaps are honest.** Disable stage 2 and the tree shows `01_` and `03_`. The gap
says something real — there was a stage there and it is switched off.

**Renaming is not a rename.** A stage's name is in its deck's filename, in every
output beside it, and in its checkpoint tags. Renaming one that has run orphans
all of them, so it is a new stage (`engines/stages.md § 7.3`, R5).

### 4.3 An attempt is numbered, and the number is not padded

`run-0`, `run-1`, `run-2`. Deliberately **unpadded**, where a stage is
`01_coarse` — the two are different kinds of thing and the difference is worth
seeing:

- a **stage** number orders a sequence somebody designed, so it pads and sorts;
- an **attempt** number counts invocations that happened, and it inherits the
  shipped `-run0` / `-run1` output naming (`job-contracts.md § 2.6`) so the
  connection between a directory and the outputs inside it stays visible.

Attempts are assigned **when a stage is prepared, in Python** (§ 1.3): the next
unused number, never reused. There is no `--force` to reset them (§ 1.2).

**`run-` is a reserved prefix and its members are numbers, full stop.** A
`run-latest` pointer was considered and dropped: with each stage set up
separately, you name the run you continue from, so nothing needs a symlink to
guess it for you.

### 4.4 Trials name themselves by their settings

A sweep has no order — no trial follows another — so the name carries **what was
tried**: `bench-G<gpus>K<ranks-per-gpu>C<cores-per-rank>`. That is the shipped
`point-G<g>K<k>C<c>` convention, and it is what lets `summarize` map a directory
back to its point.

> **Ordered levels carry position; unordered levels carry settings.** One naming
> rule each, matching what the level actually is.

**In code:** `materialize.job_dir_name` returns `point-<name>` for everything
today. It branches on `JobSet.kind` — a **ladder** job becomes `<seq>_<name>`, a
**sweep** point keeps `bench-<knobs>`. One function, one condition, and the
JobSet already carries `kind`.

---

## 5. The files, and which of them are sources

At the calculation level every file is one of three things, and confusing them is
how a folder stops being trustworthy.

| File | Kind | Written by | If you delete it |
|---|---|---|---|
| `stages.json` | **source** | the user's surface | the calculation cannot be regenerated or reopened |
| `<id>_<name>.fdf` | derived | the producer, from the source | regenerate |
| `<id>_<name>.run.sh` / `.sbatch` | derived | prep, from the deck + the machine's config | re-prep |
| `job-set.json`, `STAGE-PLAN.md` | derived | the producer / prep | regenerate |
| `*.psml`, `mb_monitor.py` | **input**, copied in | the producer | re-resolve from the project's cache |
| `.mbcheckpoint.json` | **source** | `snapshot init` | the big-file classification is lost |
| stage outputs (④) | **result** | the engine | gone — this is what the history is for |
| trial outputs (⑥) | **scratch** | the engine | nothing lost; `bench-result.json` is the answer |

> **One source, everything else derived.** `stages.json` is the only file at the
> calculation level that cannot be reconstructed from the others. That is what
> makes reopening a calculation possible, and why no produce and no run may write
> to it (`checkpointing.md`, S4).

### 5.1 The config files, by level

| File | Level | Format | Holds |
|---|---|---|---|
| `molbuilder.json` | outside the tree — cwd or `$XDG_CONFIG_HOME` | validated, no version | **the machine**: activation, module preamble, scheduler, env names |
| `.molbuilder.json` | ① project | same, deep-merged over the above, project wins | machine settings for this project |
| `<id>.template.fdf` | ③ calculation | engine deck, incomplete | **the science backbone** — everything fixed, nothing a stage varies, nothing the hardware decides |
| `stages.json` | ③ calculation | `molbuilder/stages@1` | **the science**: base settings, which vary, the stages, and the resource *intent* |
| `<id>.fdf` | ④ stage | engine deck, complete | **the rendered deck** — template ⊕ this stage ⊕ this machine. Written by `prep`; delete it and re-prep |
| `job-set.json` | ③ calculation | `molbuilder/job-set@1` | the jobs and their resources. **Stages carry no edges** (§ 1.3); the edge fields serve the benchmark sweep |
| `.mbcheckpoint.json` | ③ calculation | `molbuilder/checkpoint-config@1` | which patterns are big files |
| `.molbuilder.json` | ⑤ benchmark bundle | same as project | **the activation the bundle carries to the target** — written by `_write_activation_config`, the single place that decision is persisted. A fourth scope in practice, and deliberate: the bundle must be runnable after `scp` |
| `environment.json` | ⑤ benchmark bundle | `molbuilder/environment@1` | the machine as detected when this stage was measured |
| `bench-manifest.json` | ⑤ benchmark bundle | `molbuilder/bench-manifest@2` | the two comparable points, and the source deck's hash |
| `bench-result.json` | ⑤ benchmark bundle | `molbuilder/bench-result@1` | every trial's timing, the winner, a recommendation |

**The split is strict, and it is why a calculation folder is portable**: the
machine's knowledge lives in `molbuilder.json`, outside the calculation; the
science lives in `stages.json`, inside it. A calculation carries no walltime, no
partition, no activation command. Copy it to another cluster and it still
describes the same calculation (`job-system.md § 2`, decision 3).

The benchmark files are the one deliberate exception, and they sit at **⑤, not
③**: they are a measurement of *this machine* for *this stage*, so they are not
portable and are not meant to be. Moving a calculation to a different cluster
leaves them stale, which the recorded environment makes visible.

---

## 6. Where the saved history sits

**One saved history per calculation, at level ③.** Not per stage, not per
project.

```
bdt_au_relax_c6h4s2au38/
├── .git/                       the text: decks, wrappers, stages.json, .XV, .CG
├── .binsnapshots/<save>/       the big files, by path:
│   ├── 01_coarse/run-0/<id>.DM   ← that attempt's density matrix
│   ├── 01_coarse/run-1/<id>.DM   ← the retry's, kept separately
│   ├── 02_tight/run-0/<id>.DM
│   └── MANIFEST                  ← name, size, checksum for each
└── .mbcheckpoint.json          which patterns count as big
```

Three reasons it belongs at ③:

- **The shared package is above the stages.** A history rooted inside `01_coarse/`
  cannot restore a pseudopotential that lives one level up, so a restored stage
  would have links pointing at nothing.
- **Going back to a stage is a whole-calculation act.** Branching at *coarse* to
  try a different *tight* needs a history containing both.
- **Results are already separated by path**, so each attempt's big files stay its
  own without a history of their own.

### 6.1 What the archive covers, in one rule

**Git tracks the containers. The archive covers the runs.**

That is the whole classification, and it needs no marker file, no config flag and
no name matching — because § 1.1 already made every directory one thing or the
other. A container holds setup: decks, wrappers, the description, links, the
shared package. All text or small, all git's.

**Which runs?** The ones this calculation owns: the calculation root itself when
it is flat, otherwise each stage's `run-N/`. A benchmark's `point-*/` is a run of
a **nested container**, one level deeper, and its `.DM` is a five-iteration
throwaway — so the calculation's archive does not reach it. What survives a
benchmark is `bench-result.json`, text, which git tracks wherever it sits.

So the rule is about **depth, not names**: a run directory is a direct child of a
stage, or the root of a flat calculation. Nothing below that is this history's
binary business.

### 6.2 Append-only, because attempts are immutable

An attempt never changes after it is written (§ 1.2), so an archived file never
changes either. A new save point stores the attempts that appeared since the last
one and references the rest.

**That is the disk-growth problem solved by structure rather than by hashing.**
The archive copies every big file on every save today, and a five-stage mission
with a 2 GB density matrix per stage pays for all of it every time. Immutable
attempts make "archive what is new" both correct and obvious, where a
content-addressed store would have been correct and hopeful.

**And it makes a violation detectable.** Immutability is a contract, not a
permission bit. But an attempt that was archived and then edited *differs from its
recorded checksum*, which is precisely what I2 already checks per file. Nothing
notices today; § 7 makes it an invariant.

### 6.3 What still stands in the way

The setup step refuses a folder whose subfolders contain a calculation file, at
**any** depth — and this tree has three such levels now: a stage's linked deck,
an attempt's linked deck, and the benchmark bundle's real one. So the folder the shipped code already produces cannot be put
under a history. The fix is for the producer, which just built the folder and
knows it is one calculation, to say so (`checkpointing.md`, L1).

---

## 7. The invariants

Each is written so a test can assert it. Rules about a single run directory or a
single history live in their own contracts and are cited, not repeated.

**Naming and identity**

1. **Every path segment matches `[A-Za-z0-9_-]+`**, and a topic is one of the
   nine (`job-contracts.md § 2.5`).
2. **A calculation directory is named by its run id**, and that id is the
   `SystemLabel` in every stage deck inside it (`run-identity.md § 3`).
3. **Every file a stage reads or writes shares one basename** — the id
   (`job-contracts.md § 2.1`, Rule 2). This is what makes warm restart work
   across stages without copying anything.
4. **A stage's `seq` is assigned once and never reassigned**; stages append
   (§ 4.2). **An attempt's number likewise** — the next unused, never reused, and
   nothing resets it (§ 1.2).
4a. **No directory in this tree points at a file that does not exist yet.**
   Stages are set up one at a time, after the previous one finished, so
   everything a run continues from is a real file copied in before it starts
   (§ 1.3). A dangling link means something was chained that should not have
   been.
5. **A trial never shares the calculation's identity.** Its deck is relabelled
   and forced cold, so it can neither read nor overwrite a stage's saved state
   (§ 3.2).

**Ownership**

6. **Generate writes only level ③** (§ 2.1 step 3); **prepare writes only one
   attempt at ⑤** (step 4); the engine writes only the directory it was launched
   in.
6a. **Every directory and every link in this tree is made by Python.** The
   wrapper activates an environment and execs an engine in a directory it was
   handed, and does nothing else (`running-a-job.md § 2.2a`). **Not held
   today**: `runwrap.py`'s `attempt_dirs` prologue creates and arranges an
   attempt in shell.
7. **A shared file exists once, at ③**, and is linked into each stage. Never
   copied per stage.
8. **Every directory is a container or a run, never both** (§ 1.1). A run's
   output stays inside it; nothing a run writes appears above it.
8a. **An attempt is immutable.** Once written it never changes, and once archived
   it must never differ from its recorded checksum — which is
   `checkpointing.md`'s I2 applied to a directory instead of a file (§ 6.2).
9. **The description is the only source at ③.** No produce and no run modifies it
   (`checkpointing.md`, S4).

**Composition**

10. **A calculation folder carries no machine knowledge** — no walltime, no
    partition, no activation. Those are `molbuilder.json`'s, outside the tree.
    Benchmark files are the deliberate exception and sit at ④.
11. **A parameter difference is a different deck; a resource difference is a
    different launch.** Neither mechanism is used for the other's job.
12. **Derived files can be deleted and regenerated** from `stages.json` plus the
    machine's config, byte-identical except for the provenance timestamp.
13. **Warm restart flows down the stage axis only, and never on its own.** A
    stage continues from an earlier stage's run that the **user named**, never
    from a trial, and never because something finished (§ 1.3).

**History**

14. **One history per calculation, rooted at ③** (§ 6).
15. **Every big regular file is either in git or in the archive, never both,
    never neither** (`checkpointing.md`, S1) — and after the 2026-08-06 fix that
    holds at every depth, so a stage's result is covered.
16. **The archive covers runs this calculation owns** — a flat root, or a
    stage's `run-N/`. A nested container's runs (a benchmark's `point-*/`) are
    not its business (§ 6.1). **Not held today**: the walk classifies by pattern
    and archives a trial's `.DM` like any other.
17. **A save stores only what is new** (§ 6.2). **Not held today**: every save
    copies every big file.

---

## 8. What is not settled

1. **Does a trial's answer feed the stage automatically?** `bench-result.json`
   sits beside the stage that was measured, and its recorded choice is portable —
   the concrete rank and core counts are re-resolved per machine. Whether the tab
   offers *"use the measured resources"*, and what it does when the environment or
   the source deck has since changed, is a surface decision.
2. **Must every stage be measured?** Measuring each of five stages costs five
   sweeps. In practice a user measures one representative stage and reuses the
   answer for the rest. The layout allows both; nothing says which is expected,
   or how a stage records *"resources measured on 02_tight"*.
3. **Does the deck still need the stage in its filename?** The old layout put
   every deck in one directory, so they had to be `<id>_coarse.fdf`,
   `<id>_tight.fdf` — and `job-contracts.md § 2.3`'s stage-suffix convention and
   the decoder's regex both exist for that. Now each deck is rendered *into its
   own stage directory*, so the directory already says which stage it is and the
   deck can simply be `<id>.fdf`. That is simpler and it makes the per-stage
   trajectory separation free rather than something the filenames have to carry —
   but the decoder reads those names, so it is a change to verify, not to assume.
4. **May one calculation folder hold two ladders?** Nothing forbids two
   descriptions side by side, and the layout would allow it, but the id names the
   folder and warm files are shared, so a second ladder would continue from the
   first's state. Probably refuse; not yet stated.
5. ~~**What is the hand-run entry point for one stage?**~~ **Answered**
   (§ 2.1): preparing and submitting are separate steps, each naming its stage —
   `jobset prep <stage>` then `jobset submit <stage>`, with `--cold` on prepare
   because skipping the copy is a setup decision. The exact spelling is in
   `web/staged-runs-architecture.md § 8`, step 1c; only cosmetic choices remain.
