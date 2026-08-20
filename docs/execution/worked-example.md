# One calculation, end to end — a worked example

**Role:** guide
**Domain:** execution
**Companions:** [`execution/project-layout.md`](?doc=execution/project-layout.md)
— the tree this walks through; [`engines/stages.md`](?doc=engines/stages.md) —
what a stage is; [`execution/checkpointing.md`](?doc=execution/checkpointing.md)
— what the history guarantees;
[`archive/2026-08-19-staged-runs-implementation-plan.md`](?doc=archive/2026-08-19-staged-runs-implementation-plan.md)
— the plan and the order of work.

This is the whole design followed once, with a real molecule, in the order a
person would actually do it. It exists for two reasons: to show how the pieces
fit when nobody is looking at them one at a time, and — because a walkthrough
touches every seam — to **find the places where they do not fit yet**. § 8 is
that list, and it is the more valuable half.

**Status: the story is the target. Four of its steps do not work today**, and
each is marked ⛔ where it appears, with the whole list in § 8. Four are closed
— #5 and #6 (the stage-to-stage carry and the directory naming, both fixed
2026-08-10), #7 (the commands, which were specified elsewhere all along) and #8.

---

## 1. What we are doing

A benzene-dithiol molecule bonded to a gold surface. We want a **relaxed
geometry good enough to publish**, and we do not want to spend a week of cluster
time finding out that the settings were wrong.

So, the way a careful person works:

1. relax it **loosely** first — cheap, fast, gets the gross geometry right;
2. then **tightly** — expensive, and only worth doing from a good starting point;
3. before the expensive one, **measure** what hardware configuration runs it
   fastest;
4. keep a **save point** at each converged step, so a later idea can start from
   one instead of from scratch.

```mermaid
flowchart LR
    S["a structure<br/>bdt_au.xyz"] --> D["describe:<br/>base + two stages"]
    D --> G["generate:<br/>one folder"]
    G --> B["measure:<br/>which hardware"]
    B --> R["run coarse,<br/>then tight"]
    R --> C["save points,<br/>one per stage"]
    C --> N["a third stage,<br/>or a branch"]
```

---

## 2. Getting the structure into the tree

The description **points at** a structure; it does not contain one
(`engines/stages.md § 6.3`). So the geometry has to be somewhere with a path
before anything else can name it.

```
projects/BDT-Au/structure/bdt_au.xyz
                          bdt_au.molstruct.json     ← regions, frozen atoms
```

✅ **Gap 1 — CLOSED.** The Molbuilder tab's **Save to project** writes the
pair through the server's one codec into the folder you picked, and the
hand-over later writes the calculation's own copy beside the description
(as `<label>.source.xyz` — the reserved name no engine output can take,
`job-contracts.md § 6.3`). *(When this walkthrough was written, nothing
owned this step and you copied the file by hand.)*

---

## 3. Describing the calculation

Two tabs share the work, and the split is the design (*the browser
describes*): the **Structure-optimization tab** collects the physics —
every parameter, stated once — and **Send to Task setup** writes the
hand-over into the calculation folder you picked; the **Task setup tab**
then asks what only it may ask — the shape, the stages, what varies — and
its save writes the real `task.json`.

|  | **base** | **01 coarse** | **02 tight** |
|---|---|---|---|
| mesh cutoff | 300 Ry | **150** | 300 |
| force tolerance | 0.01 eV/Å | **0.04** | 0.01 |
| relaxation | Broyden | **CG** | Broyden |
| everything else (35 fields) | shared | — | — |

*(No `restart` row: the shipped ladders set none, deliberately — a rung's
position says nothing about whether there is anything to continue from, and
`continue` is the default anyway (`run-identity.md § 4`, corrected
2026-08-18). Set it per stage only when you mean to override that.)*

Two things are worth noticing.

**Only what differs is written down.** A stage is a name and an overlay
(`engines/stages.md § 2`); the other thirty-five settings are stated once. That
is what stops the second stage quietly running different physics because someone
edited one copy and not the other.

**The name appears as you type, and it splits in two.** From "BDT/Au relax" plus
the formula:

```
label   BDT_Au_relax              ← the SystemLabel in every deck, and the
                                     stem of every file the runs write
id      BDT_Au_relax_Au38C6H4S2   ← recorded in task.json; never a filename
```

The **label** is what makes the tight stage able to pick up the coarse stage's
geometry at all — SIESTA looks for `<SystemLabel>.XV` and finds what coarse left.
The **id** is how molbuilder knows *which calculation* that state belongs to, and
it is a record rather than a name (`execution/run-identity.md § 2.0a`). **Neither
of them names the folder** — you type that when you generate (§ 4).

---

## 4. Generating

The Task setup tab's **Save to this folder** writes the description (its
first step is a checkpoint of the folder's state, so what you are about to
change can be brought back):

```
projects/BDT-Au/optimization/bdt-relax/    ← the folder name is yours to pick
├── BDT_Au_relax.template.toml      every parameter of the calculation —
│                                  the hardware's named, not answered
├── task.json                      what each stage tunes, + the bench plan,
│                                  and the id — BDT_Au_relax_Au38C6H4S2
├── BDT_Au_relax.source.xyz          the structure pair, from the hand-over
└── BDT_Au_relax.source.molstruct.json
```

*(No pseudopotentials yet, and no monitor: `prep` copies those from the
library on the machine that runs the job — `project-layout.md § 2.6`,
stated 2026-08-18.)*

**Nothing here names a machine** — no walltime, no partition, no activation
command, no rank count. Copy this folder to any cluster and it still describes
the same calculation. There are no stage directories yet either; those appear
when you prep a stage on the machine that will run it.

**A template, not finished decks.** The browser writes the science backbone and
`task.json`; the actual `.fdf` for each stage is rendered later, on the machine
that will run it. That is not deferral for its own sake: **a deck carries values
that depend on how it will be launched** — a block size derived from the rank
count and the GPU, written *inside* the deck, and an eigensolver choice that
changes both the deck and which environment the wrapper activates. **A parameter
that depends on the launch cannot be decided before the launch is known**, so a
deck finished on a laptop is guessing at both (`project-layout.md § 2.2`).

**One basename everywhere.** Every deck says `SystemLabel BDT_Au_relax` — the
label, not the id. That is not cosmetic: it is why a later stage can pick up an
earlier one's `.XV`, and it is why the formula stays out of the filename (§ 3).

> ### Which shape this walkthrough follows
>
> **From here on this walk uses the hierarchical shape** — a directory per stage,
> a subdirectory per attempt. It is the clearer one to read, because every state
> the calculation passed through stays visible on disk.
>
> **Both ship, and the page ASKS — there is no default** (it decides where
> results land, so it is never guessed). Which one is a field in the
> description you wrote in § 3
> ([`project-layout.md`](?doc=execution/project-layout.md) § 1). The same two
> stages, flat:
>
> ```text
> bdt-relax/
> ├── BDT_Au_relax_01_coarse.fdf        stages told apart by the suffix
> ├── BDT_Au_relax_02_tight.fdf
> ├── BDT_Au_relax_01_coarse-run0.out  attempts by an output index
> ├── BDT_Au_relax.XV             ← SHARED and unsuffixed: this is what lets
> ├── BDT_Au_relax.DM                tight find coarse's geometry — and what
> └── Au.psml  S.psml  C.psml  H.psml   overwrites it
> ```
>
> **Everything this walkthrough says still happens** — the same stages, the same
> measuring, the same continuing, the same checkpoints. What changes is what is
> left on disk afterwards, and § 7 is where that difference stops being cosmetic.

✅ **Gap 2 — CLOSED 2026-08-11** (plan steps 3–4). `prep` is per-stage
(`jobset prep run <stage>`) and renders each deck on the target, one per
element of the resolved `ParameterSet`, allocation in hand.
And read the relationship the right way round:
**`prep` is the framework and benchmarking is one thing you prep**
(`project-layout.md § 2.3.1a`) — the benchmark is simply where that framework got
built first, so the general part needs lifting out rather than the staged path
borrowing from a special case.

---

## 5. Measuring before spending

The tight stage is the expensive one. Before committing a week to it, find out
what hardware configuration actually runs *this system at these settings*
fastest.

```mermaid
flowchart LR
    T["02_tight/<br/>the deck: mesh 300, Broyden"] --> X["<b>transform</b><br/>SCF capped at 5 · convergence off<br/>MD steps zeroed · cold start forced<br/>relabelled job-gpu / job-cpu"]
    X --> P["bench-G1K2C5/ · bench-G1K4C5/ · bench-G2K4C5/ · …<br/>each one timed"]
    P --> W["<b>bench-result.json</b><br/><i>choice</i>: elpa, G=1 K=4 C=6 — portable<br/><i>recommend</i>: mem 96 GB, time 0-08:20:00 — sized here"]
```

Two different kinds of answer come out, and the split matters. **`choice`** is the
*mechanism* — which engine build, how many ranks per GPU — and it transfers to
another cluster unchanged. **`recommend`** is *sizing* measured on this machine:
memory from the winner's peak usage plus 15%, and a walltime from its seconds per
iteration times an **assumed 200 iterations** times 1.5. That last number is a
guess about a run that has not happened, which is why it is a starting point and
labelled as one — a relaxation that takes 400 steps will need twice the wall time
the benchmark suggested.

The trials live **under the stage they measure**, in their own container:

```
02_tight/
├── bench/                    a self-contained benchmark bundle
│   ├── job-gpu.fdf  job-cpu.fdf
│   ├── bench-result.json     ← the answer; a few kilobytes of text
│   └── bench-G1K4C5/         ← a trial: a throwaway run
└── run-0/                    ← the real run, later
```

*The trial prefix is `bench-` ([`job-contracts.md § 6.3`](?doc=execution/job-contracts.md),
the cross-layer authority) — and since 2026-08-12 the shipped code writes it
too (the `point-` era ended with the fold; `summarize` reads the job-set's
own data, never a parsed directory name).*

**Why under the stage, and not once per project.** The best rank count depends on
the science: mesh cutoff changes the grid, basis size changes the matrix, and
`BlockSize` is bounded by the orbital count over the rank count — so a bigger
basis moves it. Coarse and tight can genuinely want different hardware.

**Why the trials cannot hurt the real run.** The benchmark relabels its decks to
`job-gpu` / `job-cpu` and forces a cold start. So a five-iteration timing run
cannot read — or overwrite — the density matrix the real run depends on. That
relabelling is not a leftover from when benchmarking was standalone; it is
exactly what lets a trial live inside a stage's directory.

✅ **Gap 3 — CLOSED 2026-08-12** (the fold). The connection IS the command:
`jobset prep bench <stage>` renders the trials into that stage's own
`bench/` container and lays the sweep's record beside them, so *which stage
this benchmark measures* is structural, not a hand-composed pairing.  (The
`bench generate` this gap was written against is deleted — the whole legacy
stack followed on 2026-08-13.)

✅ **Gap 4 — CLOSED 2026-08-12** (§ 2.3.2's verdict). The measured
answer reaches **the next `prep`**, not the description — deliberately:
a rank count is a machine fact the description must never carry
(`engines/template.md` § 7).  `summarize bench` writes the verdict into the
stage's container as the editable `run-config.toml`, and `prep run <stage>`
applies what the file says to the allocation fields you did not state, the
winning eigensolver arriving as pins; deleting the file declines it
*(interactive `use it? [y/N]` until 2026-08-19 — the answer moved into the
tree)*.  (The `bench prep-run`/`run-production.sh` chain this gap described
is deleted.)

---

## 6. The official run

Two steps, and the split is the point. **Prepare** it in the tab — the run folder
is made, the deck and pseudopotentials linked in, nothing carried because coarse
starts from the structure. It tells you what it did. Then, on the machine that
will run it:

```
molbuilder jobset submit run coarse --mode direct
```

**Why the terminal for this half.** The browser writes a package that names no
machine; everything that depends on *which* machine — the rank count, the
eigensolver, the queue, the activation command — is resolved here, where the
answers exist (`project-layout.md § 2.1`).

```mermaid
flowchart TB
    subgraph C["01_coarse/  — a container"]
      direction TB
      L["deck · pseudopotentials · monitor<br/>(links up to the calculation root)"]
      subgraph R0["run-0/  — an attempt, immutable once written"]
        O["the deck, linked in<br/>&lt;id&gt;.XV · &lt;id&gt;.DM · &lt;id&gt;.out<br/>the session log"]
      end
    end
    L -.->|"Python links these in<br/>before the wrapper starts"| O
```

It converges. **Now you look at it** — the trajectory, the final forces, whether
the geometry is sane — and only then set up tight.

That pause is the design, not a missing feature. A stage is a long job; a chain
that continued on its own could spend a week refining a geometry you would have
thrown away in a minute. So **stages are not chained**: each one is set up and
started separately, after you have seen the last (`project-layout.md § 1.6`).

Which makes the hand-off simple. When you set tight up, coarse's `.XV` is
**already on disk** — you just looked at it — so it is **copied** into tight's
run directory:

```
02_tight/run-0/<label>.XV   a real copy of 01_coarse/run-0/<label>.XV
```

Copied and not linked, because tight writes to that same filename and would
otherwise overwrite coarse's result. And **you say which run** it comes from:
continuing from `run-0` and continuing from `run-2` are different scientific
choices.

✅ **Gap 5 — closed 2026-08-10.** `materialize` wrote that hand-off as a symlink
to `../<producer>/<label>.XV` and `stages_to_jobset` built a chained ladder —
`depends_on` plus `.XV`/`.DM`/`.CG` carry edges — so once attempts moved outputs
into `run-N/` the link dangled. The fix was not a better link: **the producer
emits no chain and no carry at all**, and the copy happens exactly as described
above, when you set the stage up. A produced tree now contains no dangling
symlink. **The chaining machinery was then deleted outright** (2026-08-10,
user): no `Carry`, no `depends_on`, no `dep_kind`, no `carry_deref` — so the
mechanism that made the dangling link possible no longer exists to be got
wrong again.

✅ **Gap 6, in the same code — closed 2026-08-10.** `job_dir_names` branches on
`JobSet.kind`, so a ladder's stage directories are `01_coarse/`, `02_medium/`,
`03_tight/` while a benchmark's trials keep a settings-based name. The seq is read back off
the deck's own token rather than counted, so disabling a stage leaves a gap
rather than renumbering.

**And the way I first tried to fix gap 5 was wrong**, which is the more useful
lesson. I put attempt-directory logic in the wrapper: ~130 lines of shell that
scan for run directories, create one, link the deck in, copy files and `cd` in.
That is `jobset/materialize.py` rewritten in bash, one level down, in the one
layer kept free of filesystem work — **and it was the only place in the system
that changed directory.** It was retired on 2026-08-10, which is what makes
`project-layout.md` invariant 6a held rather than aspirational.

**If you have to re-run a stage**, you do not overwrite anything — each
invocation gets `run-1`, then `run-2`, carrying the previous attempt's warm state
unless you ask for a cold start. `run-0` is byte-identical afterwards. There is
no `--force`, because there is nothing to reset.

✅ **Gap 7 — closed 2026-08-10, and it was half-answered already.** It read:
*how you re-run one stage by hand is unstated*, and offered `jobset submit --only
<stage>` or a new top-level run command.

Neither was needed, because **the grammar was specified elsewhere and this
document had not caught up**:
[`execution/job-system.md`](?doc=execution/job-system.md)
fixes `jobset <verb> <kind> [<stage>]`, and `project-layout.md` § 8 and
`web/task-setup-plan.md` were both already writing `jobset prep run <stage>
--from <run>`. What was genuinely open was one shape question — stage as the
positional, or folder with `--stage` — **decided 2026-08-10 (user): the stage is
the positional**. There is no way to run a whole ladder
unattended: `--chain` was deleted the same day, in both modes.

So the entry point is:

```bash
molbuilder jobset prep   run tight --from 01_coarse/run-0   # or --cold
molbuilder jobset submit run tight --mode direct|submit
```

`job-system.md` § 5.3 is now the authority for the commands, and the ordering
constraint this gap carried still holds: that entry point must exist **before**
the wrapper's directory-making prologue is retired, or the manual path breaks
with nothing behind it.

> **The lesson is the one this document keeps teaching.** The gap was not a
> missing design — it was four documents describing one act in three grammars,
> with the newest of them calling it unanswered. A walkthrough finds that;
> reading any single file does not.

---

## 7. Save points, and changing your mind

A stage finishes. **You** save it, and the note is yours to write — nothing
saves or tags on your behalf (`checkpointing.md` **L3**, **L4**):

```
coarse converged, 41 steps

Calculation: bdt-relax
Manifest-SHA256: 9f2c…
```

The note says **how it went**; the two trailers are added for you — the folder
this history belongs to, and the fingerprint of the archive this state points at.
*(An earlier draft of this section showed a generated subject line and an
automatic tag, `<id>/coarse/<UTC>`. Both are retired: a note you did not write
answers nothing, and machine-made tags crowd out your own.)*

**What is stored where.** Git takes the containers: decks, wrappers,
`task.json`, the links, `bench-result.json` — all text. The archive takes the
runs: `01_coarse/run-0/<label>.DM` and its siblings, by path, with checksums. The
benchmark's throwaways are not this history's business.

The split needs no marker file and no list of names, only **depth**: a container
is anything with a container below it, a run is a leaf. That is the whole rule,
and it is why the benchmark's trials — two levels down, inside a container
that is itself inside a stage — falls out on the right side without anyone
saying so.

> ### The same checkpoint means two different things
>
> This is the one place the shape choice (§ 4) stops being cosmetic, and it is
> worth stopping on because it changes what a mistake costs.
>
> In the **hierarchical** walk above, `01_coarse/run-0/` is still sitting on
> disk. The checkpoint is a convenience — somewhere to branch from, a way to see
> what changed. Skip one and you have a thinner history beside a folder that
> still holds everything.
>
> In the **flat** shape there is one `<label>.XV` and one `<label>.DM`, and the tight
> stage **overwrote** them. The coarse geometry is not on disk anywhere. So:
>
> | | hierarchical | flat |
> |---|---|---|
> | after tight runs, coarse's geometry is | on disk, openable | **only in the checkpoint** |
> | a checkpoint is | insurance | **the mechanism** |
> | a missed checkpoint is | a thinner history | **a state that no longer exists anywhere** |
>
> That is why `checkpointing.md` is a contract with invariants rather than a
> description of a feature, and why *"take a checkpoint before each stage"* is
> not housekeeping in the flat shape — it is **the save point**
> ([`checkpointing.md`](?doc=execution/checkpointing.md) § 8).
>
> It also changes what *restore* means to you. In the hierarchy you can open an
> old attempt and copy something out of it. In flat there is nothing to open:
> **restore is a rewind, not a fetch** — it puts the whole folder back to an
> earlier moment, and anything newer that was not checkpointed goes with it.

Two weeks later you want a third stage — finer k-grid, from tight's result.

```mermaid
flowchart LR
    A["<b>coarse</b><br/>saved: converged"] --> B["<b>tight</b><br/>saved: converged"]
    B --> C["<b>03_finer</b><br/>appended: seq 3"]
    B -.->|"restore tight,<br/>then save"| D["<b>03_other</b><br/>a different idea,<br/>same parent"]
```

**Stages append; they never renumber.** Once tight has run, "insert something
between coarse and tight" is not an insertion — it is a new stage that happens to
be coarser, and it runs from where tight left off. Numbering it `03` is the
truth. That also means an attempt's outputs stay attached to the stage that
produced them, forever.

✅ **Gap 8 — closed.** It read: *the history cannot exist, because the
checkpoint setup refuses a folder with calculation files in its subdirectories,
which this tree has at three levels.* It no longer refuses one that **says** its
subdirectories are one calculation — a root carrying `task.json` (or
`job-set.json`, or `bench-manifest.json`) is accepted, and this tree carries one.
The rest of that gap dissolved with it: forking needs no web route, because it
is restore-then-save and both are routed. What is still true, and is not a gap,
is that **a save is always an explicit act** — nothing takes one for you
(`checkpointing.md § 9`).

---

## 8. What this walkthrough found

Eight gaps, in the order a user meets them. Four were on no list before this
document was written. **Four are now closed** — #5, #6, #7 and #8. **§ 8.1 adds
the eight the rest of the domain had recorded separately**, so this is the whole
workflow's list rather than one walkthrough's.

They closed in three different ways, and the contrast is worth keeping: #8
needed the checkpoint rework, real code. #5 and #6 needed the producer to stop
doing something — subtraction, not addition. #7 needed nobody to build anything
at all: the answer was already written in two other documents, and this one had
gone on calling it unanswered. **A gap list is only as good as its last
reconciliation against its neighbours** — and this one had also gone on
describing #5's fix in terms the layout contract had already ruled out.

| # | Gap | Severity |
|---|---|---|
| 1 | **Saving the structure into the tree is a step nobody owns.** The description points at a path; a workspace geometry has none | small, but it is the first thing a user hits |
| 2 | ~~**Produce/prep boundary is undefined locally.**~~ **Closed 2026-08-11** (plan steps 3–4, § 4's own ✅ above): `prep` owns the containers on EVERY machine — host-equals-target is the same call | ✅ closed |
| 3 | ~~**Nothing connects a benchmark to the stage it measures.**~~ **Closed 2026-08-12** (the fold): `prep bench <stage>` renders trials into the stage's own container — the connection is structural | ✅ closed |
| 4 | ~~**The measured answer reaches a script, never the description.**~~ **Closed 2026-08-12** (§ 2.3.2): `summarize` writes the stage's verdict as the editable `run-config.toml` and `prep run` applies what the file says (interactive ask until 2026-08-19); the answer reaches the next prep's allocation + pins — never the description, which may not carry machine facts (§ 7) | ✅ closed |
| 5 | ~~**Stage-to-stage carry is broken.**~~ **Closed 2026-08-10.** The producer stopped emitting `depends_on` and `Carry` entirely, so nothing dangles. ⚠ This row used to say *"fixed by resolving the attempt at **submit**"* — which contradicted `project-layout.md § 2.3.4`, where the copy is made **at `prep`, from the run you name with `--from`**. The contract was right: by then the source has already finished, so there is nothing to resolve later | ✅ closed |
| 6 | ~~**Stage directories are named `point-<name>`.**~~ **Closed 2026-08-10.** `job_dir_names` branches on `JobSet.kind`; a ladder gets `01_coarse/`, a sweep keeps `point-*` | ✅ closed |
| 7 | ~~**No hand-run entry point for one stage.**~~ **Closed 2026-08-10.** The grammar was already fixed in `job-system.md` and used by two other documents; this file had not caught up. `jobset prep run <stage> --from <run>` / `jobset submit run <stage>`, stage as the positional (user, 2026-08-10). ⚠ *This row also said `--chain` runs a ladder unattended; it does not — line 327 of this same document records it deleted on that date, and stages do not chain (`project-layout.md` § 1.6). Corrected 2026-08-11.* The ordering constraint stands: it must exist before the wrapper's directory-making prologue is retired | ✅ closed |
| 8 | ~~**The history cannot be created.**~~ **Closed by the checkpoint rework.** `init` accepts a folder whose root *declares* its subdirectories one calculation (`task.json`), which this tree carries; forking is restore-then-save, which needs no branch route. Saves stay explicit by design, which is § 9, not a gap | ✅ closed |

### The one that was fixed first, and why

**Gap 5**, because it was a regression rather than a missing feature: staged runs
handed geometry across correctly before attempt directories existed, and then
did not.

The fix was the one in § 6, and it **removed machinery rather than adding it**:
stages stopped being chained, so the hand-off is a plain file copy made when you
set the next stage up (`project-layout.md` § 1.6). **Gap 6 rode along**, being
the same code. And the shell block that caused the regression was **retired
rather than repaired** — its guard against being run from inside an attempt, its
`$PWD` bug and its `--force` refusal all stopped existing once Python decided the
directory. Of the gaps this walkthrough found, only gap 1 (saving the
structure into the tree) remains open — and it is not a regression: it is a
**join** (§ "What the shape of this list says") that was never built.  *(This
sentence counted "four open" until 2026-08-13; gaps 2, 3 and 4 closed with
the plan's steps 3–4 and the fold, and their rows above say how.)*

**The lesson is worth more than the fix.** The system was already built the right
way — `materialize` lays out directories and links, `submit` picks the working
directory and launches, and the wrapper activates and execs. I added a second
layout implementation in bash without checking whether one existed, then designed
an elaborate way to make a chained hand-off work when the answer was not to
chain. Both rules are now written where they can be pointed at:
[`running-a-job.md`](?doc=execution/running-a-job.md) § 2.2a and
[`project-layout.md`](?doc=execution/project-layout.md) § 1.6.

### 8.1 The rest of the workflow's gaps, collected from where they were recorded

*Added 2026-08-11. The eight above are what **this walkthrough** found. Eight
more were recorded honestly by their owning documents and never gathered, so
nobody could see the workflow's total shape. **A gap list spread over eight files
is not a list.** Each row names its owner; the owner keeps the reasoning, this
table keeps the count.*

| # | Gap | Recorded in | Blocks |
|---|---|---|---|
| 9 | ✅ **CLOSED 2026-08-11** (plan step 1): `describe` emits `<label>.template.toml` from the schema | `template.md` · plan P12 u6b | — |
| 10 | ✅ **CLOSED 2026-08-11** (plan step 2, `b7ca09d7`): `jobset describe` writes template + `task.json` + data files | `job-system.md § 5.1`, § 5.3's grammar | — |
| 11 | ⛔ **MOOT 2026-08-14**: the `schema_fingerprint` mechanism is **retired**, so the gap it named cannot be open. One writer, one reader, and a warning weaker than the per-field rows that ran right after it (`template.md` § 10, `stages.md` § 6.6) | `stages.md § 6.6` | — |
| 12 | **`user_custom` has no schema field.** It must be an ordinary item for USER-CUSTOM text to survive `prep`; no engine config declares one | `template.md § 9.2`, § 12 | the reserved-block completeness claim (G6) |
| 13 | **The `required` check is unbuilt.** A stage may declare what it cannot run without; nothing verifies it in the run directory | `job-contracts.md § 4.4` · `stages.md § 5` | a TranSIESTA ladder starting without its `.TSHS` |
| 14 | ~~**Nothing offers a save before `prep` overwrites a folder.**~~ **Closed 2026-08-12** (A3/U14): `prep` now SAYS what is under way — launched attempts, launched trials, warm files — asks before re-rendering, and points at the checkpoint verbs when a checkpoint repo exists | `checkpointing.md § 9` · `run-identity.md § 6` | — |
| 15 | **`checkpoint verify` has no verb.** The archive check exists and is reachable only by attempting a restore — the worst moment to learn an archive is gone | `checkpointing.md § 12` | knowing a history is intact |
| 16 | ✅ **CLOSED 2026-08-11** (C11): `submit --mode` falls back to `execution.mode`; unset in both is a refusal, never a derivation from the detected scheduler (2026-08-12 aligned `bench` to the same rule) | `job-system.md § 5.3` · `running-a-job.md § 5.4` | — |

**Five of checkpointing's own invariants are untestable until the layout lands**
— **S2**, **S3**, **S4**, **S6** and **L8** are stated, tracked and deliberately
unasserted (`checkpointing.md § 13.4`). They are not gaps in the design; they are
the design waiting for a surface to be asserted against, and that surface is
gap 9.

```mermaid
flowchart LR
    G9["<b>9</b> nothing writes<br/>a template"] --> G10["<b>10</b> no<br/>describe verb"]
    G9 --> G12["12 · user_custom"]
    G9 --> INV["S2 · S3 · S4 · S6 · L8<br/><i>five invariants,<br/>untestable until then</i>"]
    G9 --> G2["<b>2</b> produce/prep<br/>boundary"]
    G2 --> G4["4 · the measured answer<br/>reaches no description"]
    G13["13 · required check"]:::indep
    G14["14 · no save before prep"]:::indep
    G15["15 · no verify verb"]:::indep
    G16["16 · mode not wired"]:::indep
    classDef indep stroke-dasharray: 4 3;
```

**Read the shape, not the count.** Four of the sixteen are independent and small
(dashed). **Everything else hangs off gap 9** — a template writer — which is why
the plan puts P12 unit 6b where it does: it is not one item among sixteen, it is
the one that unblocks **ten** — gaps 10, 11, 12 and 2 directly, gap 4 through
gap 2, and the five invariants that have nothing to be asserted against until a
described folder exists.

### What the shape of this list says

Five of the eight are **joins, not parts**. Structure→description, produce→prep,
stage→benchmark, benchmark→description, stage→stage: every one is a handoff
between two things that each work. That is the expected result of building
bottom-up, and it is why walking the story end to end finds what reviewing a
module never does — a seam is invisible from either side of it.

### What the walkthrough confirmed works

Worth saying, since the list above is all problems. The parts that hold up when
followed end to end: one description producing several decks with one basename;
the benchmark nesting under the stage it measures without being able to damage
it; attempts that cannot overwrite each other; and stage numbering that keeps an
output attached to the stage that made it, however many times you change your
mind afterwards.
