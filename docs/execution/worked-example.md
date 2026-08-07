# One calculation, end to end — a worked example

**Role:** guide
**Domain:** execution
**Companions:** [`execution/project-layout.md`](?doc=execution/project-layout.md)
— the tree this walks through; [`engines/stages.md`](?doc=engines/stages.md) —
what a stage is; [`execution/checkpointing.md`](?doc=execution/checkpointing.md)
— what the history guarantees;
[`web/staged-runs-architecture.md`](?doc=web/staged-runs-architecture.md) — the
plan and the order of work.

This is the whole design followed once, with a real molecule, in the order a
person would actually do it. It exists for two reasons: to show how the pieces
fit when nobody is looking at them one at a time, and — because a walkthrough
touches every seam — to **find the places where they do not fit yet**. § 8 is
that list, and it is the more valuable half.

**Status: the story is the target. Eight of its steps do not work today**, and each
is marked ⛔ where it appears, with the whole list in § 8.

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

⛔ **Gap 1.** Nothing owns this step. A geometry you just loaded or edited lives
in the workspace and has no path yet, and no surface says *"save this into the
project first"*. Today you would put it there by hand and the tab would not tell
you that you had to.

---

## 3. Describing the calculation

In the Structure-optimization tab: pick the project and the topic, fill in the
physics once, then add the stages.

|  | **base** | **01 coarse** | **02 tight** |
|---|---|---|---|
| mesh cutoff | 300 Ry | **150** | 300 |
| force tolerance | 0.01 eV/Å | **0.04** | 0.01 |
| relaxation | Broyden | **CG** | Broyden |
| restart | — | **clean** | **continue** |
| everything else (35 fields) | shared | — | — |

Two things are worth noticing.

**Only what differs is written down.** A stage is a name and an overlay
(`engines/stages.md § 2`); the other thirty-five settings are stated once. That
is what stops the second stage quietly running different physics because someone
edited one copy and not the other.

**The id appears as you type.** From "BDT/Au relax" plus the formula:

```
bdt_au_relax_c6h4s2au38
```

That single name becomes the folder, the `SystemLabel` in every deck, and the
stem of every file the runs write (`execution/run-identity.md`). It is what makes
the tight stage able to pick up the coarse stage's geometry at all.

---

## 4. Generating

Press **Check** — every stage is validated whole, and any complaint says which
stage it is about. Then **Generate**:

```
projects/BDT-Au/optimization/bdt_au_relax_c6h4s2au38/
├── stages.json                     what was asked for — the only source
├── <id>_coarse.fdf                 rendered from base ⊕ coarse
├── <id>_tight.fdf                  rendered from base ⊕ tight
├── <id>_coarse.run.sh  .sbatch     one wrapper per deck
├── <id>_tight.run.sh   .sbatch
├── Au.psml  S.psml  C.psml  H.psml the shared package, once
├── mb_monitor.py
├── job-set.json                    the chain, derived
├── 01_coarse/                      a container: links up, and its attempts
└── 02_tight/
```

**Two decks, not one template.** A stage's values are baked into its own file,
because a parameter change alters what the engine computes and must be in the
file the engine reads.

**One basename everywhere.** Every deck says `SystemLabel bdt_au_relax_c6h4s2au38`.
That is not cosmetic: it is why a later stage finds the earlier stage's `.XV`.

⛔ **Gap 2.** Who creates `01_coarse/` and `02_tight/`? The producer writes the
calculation root; `prep` lays out the per-stage containers — but `prep` is a
*target-side* verb, meant to run on the cluster after you copy the folder. When
host and target are the same machine (the common case for a workstation user),
nothing says whether generate should have done it already.

---

## 5. Measuring before spending

The tight stage is the expensive one. Before committing a week to it, find out
what hardware configuration actually runs *this system at these settings*
fastest.

```mermaid
flowchart LR
    T["02_tight/<br/>the deck: mesh 300, Broyden"] --> X["<b>transform</b><br/>SCF capped at 5 · convergence off<br/>MD steps zeroed · cold start forced<br/>relabelled job-gpu / job-cpu"]
    X --> P["point-G1K2C5/ · point-G1K4C5/ · point-G2K4C5/ · …<br/>each one timed"]
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
│   └── point-G1K4C5/         ← a trial: a throwaway run
└── run-0/                    ← the real run, later
```

**Why under the stage, and not once per project.** The best rank count depends on
the science: mesh cutoff changes the grid, basis size changes the matrix, and
`BlockSize` is derived from the rank count. Coarse and tight can genuinely want
different hardware.

**Why the trials cannot hurt the real run.** The benchmark relabels its decks to
`job-gpu` / `job-cpu` and forces a cold start. So a five-iteration timing run
cannot read — or overwrite — the density matrix the real run depends on. That
relabelling is not a leftover from when benchmarking was standalone; it is
exactly what lets a trial live inside a stage's directory.

⛔ **Gap 3.** Every piece exists; nothing connects them. `bench generate` takes a
deck and an output directory, so you *can* hand it `<id>_tight.fdf` and
`--out 02_tight/bench` — but no command does, nothing records that this bundle
measures that stage, and getting it wrong is silent.

⛔ **Gap 4.** The answer reaches a script but not the description. The shipped
chain works — `bench summarize` writes `bench-result.json`, and `bench prep-run`
turns it into `run-production.sh`, re-resolving the portable choice for whatever
machine you are on. What it never touches is `stages.json`. So the resource
answer lives only in a generated script, and the next `generate` — which rebuilds
everything from the description — quietly reverts to the defaults.

---

## 6. The official run

```
molbuilder jobset submit . --mode direct
```

**Molbuilder makes the attempt; the wrapper runs in it.** Python resolves
`run-0`, creates it, links the deck and the shared package in, copies any warm
state, and launches there. The wrapper's whole job is to activate the environment
and exec SIESTA.

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

It converges. Now the tight stage — and it must start from the geometry coarse
reached.

⛔ **Gap 5 — and this one is new, and mine.** It cannot. `prep` creates the
carry as a symlink to `../<producer>/<id>.XV`, which is where a stage's output
used to live. Since attempts got their own directories, coarse's geometry is at
`run-0/<id>.XV` inside the stage, and **the link points at nothing.** Worse,
*which* attempt is the good one is not knowable when `prep` runs — the runs have
not happened yet. Coarse converged first time here, but had it needed three
tries, the answer would be `run-2`.

⛔ **Gap 6, in the same line of code.** That link also has the wrong folder name.
`materialize.job_dir_name` returns `point-<name>` for every job, so today the
stage directory is `point-coarse/`, not `01_coarse/`. The layout contract already
specifies the fix — branch on `JobSet.kind`, so a **ladder** job becomes
`<seq>_<name>` and a **sweep** point keeps its knobs — but it is not written yet.
Name and depth are both wrong, in one expression.

**And the way I first tried to fix gap 5 was wrong too**, which is worth saying
because it is the more useful lesson. I put the attempt-directory logic in the
wrapper: fifty lines of shell that scan for run directories, create one, link the
deck in and copy warm files. That is `jobset/materialize.py` rewritten in bash,
one level down — a second implementation of code that already existed, in the one
layer deliberately kept free of filesystem logic.

The fix is not a cleverer link — it is a **later moment**, now specified in
[`project-layout.md`](?doc=execution/project-layout.md) § 1.3. `submit` already
chooses each job's working directory. If it resolves the attempt too, it knows
**both** attempt directories before either process exists, and writes the
consumer's carry as a concrete path:

```
02_tight/run-0/<id>.XV -> ../../01_coarse/run-0/<id>.XV
```

A real link to a real place, laid at the moment the answer became knowable —
which prep could not do (the attempts did not exist) and the wrapper cannot do
(it only sees its own stage).

The link is still **dangling when it is laid** — under SLURM the whole chain is
submitted at once, so coarse has not run yet. It resolves when coarse writes, and
the wrapper's existing **localize-on-run** step replaces it with a real local copy
before tight starts, so tight cannot write back through it into coarse's result.
That step is bash and stays bash: it is the only moment that exists *after* the
producer finished and *before* this engine starts, and it happens on a compute
node where there is no Python.

Three things fall out. **Numbering becomes deterministic**: a SLURM ladder is
submitted all at once, and two jobs scanning `run-*/` whenever they happen to
wake can give the same answer twice. **No Python is needed on the compute node** —
which matters, because `molbuilder-siesta` has no interpreter at all. And **the
wrapper gets smaller**: it localizes its carry, activates an environment, and
execs an engine. Everything else is Python
([`running-a-job.md`](?doc=execution/running-a-job.md) § 2.2a).

`run-latest` survives this as a **handle rather than a mechanism** — the Results
tab, `jobset status` and a person typing `cd` still want to name a stage's
current result without scanning. It just stops being what makes carry work.

**If you have to re-run a stage**, you do not overwrite anything — each
invocation gets `run-1`, then `run-2`, carrying the previous attempt's warm state
unless you ask for a cold start. `run-0` is byte-identical afterwards. There is
no `--force`, because there is nothing to reset.

⛔ **Gap 7.** How you re-run *one* stage by hand is unstated. Since the wrapper
no longer makes its own directory, a single stage needs a molbuilder command in
front of it — `jobset submit --only <stage>`, a new `molbuilder run <stage>`, or
both. Unanswered, and it has to be answered before the shell block is retired or
the manual path breaks (`project-layout.md` § 8, question 4).

---

## 7. Save points, and changing your mind

A stage finishes. The history records it:

```
commit   bdt_au_relax_c6h4s2au38 · coarse · relaxation converged, 41 steps
tag      bdt_au_relax_c6h4s2au38/coarse/20260806T221403Z
```

The message carries the id, the stage and **how it went** — the run decoder
already knows *converged* from *hit the step cap*, so the history can say it.

**What is stored where.** Git takes the containers: decks, wrappers,
`stages.json`, the links, `bench-result.json` — all text. The archive takes the
runs: `01_coarse/run-0/<id>.DM` and its siblings, by path, with checksums. The
benchmark's throwaways are not this history's business.

The split needs no marker file and no list of names, only **depth**: a container
is anything with a container below it, a run is a leaf. That is the whole rule,
and it is why the benchmark's `point-*/` — two levels down, inside a container
that is itself inside a stage — falls out on the right side without anyone
saying so.

Two weeks later you want a third stage — finer k-grid, from tight's result.

```mermaid
flowchart LR
    A["<b>coarse</b><br/>tagged, converged"] --> B["<b>tight</b><br/>tagged, converged"]
    B --> C["<b>03_finer</b><br/>appended: seq 3"]
    B -.->|"snapshot branch<br/>tight-alt"| D["<b>03_other</b><br/>a different idea,<br/>on its own branch"]
```

**Stages append; they never renumber.** Once tight has run, "insert something
between coarse and tight" is not an insertion — it is a new stage that happens to
be coarser, and it runs from where tight left off. Numbering it `03` is the
truth. That also means an attempt's outputs stay attached to the stage that
produced them, forever.

⛔ **Gap 8.** The history cannot exist yet. The checkpoint setup **refuses** a
folder with calculation files in its subdirectories — which this tree has at
three levels — so none of § 7 runs. Checkpoints are also user-triggered only, and
`snapshot branch` has no web route.

---

## 8. What this walkthrough found

Eight gaps, in the order a user meets them. Four were on no list before this
document was written.

| # | Gap | Severity |
|---|---|---|
| 1 | **Saving the structure into the tree is a step nobody owns.** The description points at a path; a workspace geometry has none | small, but it is the first thing a user hits |
| 2 | **Produce/prep boundary is undefined locally.** Nothing says who creates the stage containers when host and target are the same machine | design decision, one sentence |
| 3 | **Nothing connects a benchmark to the stage it measures.** The parts compose by hand and getting it wrong is silent | small |
| 4 | **The measured answer reaches a script, never the description.** `bench prep-run` writes `run-production.sh`; `stages.json` never learns, so the next `generate` reverts to defaults | medium — it is the point of measuring |
| 5 | **Stage-to-stage carry is broken.** ⚠ `materialize` links `../<stage>/<id>.XV`; attempts moved outputs into `run-N/`, so the link dangles — and *which* N is unknowable at prep time. Fixed by resolving the attempt at **submit**, which knows both | **serious, and newly introduced by the attempt-directory change** |
| 6 | **Stage directories are named `point-<name>`.** `job_dir_name` does not branch on `JobSet.kind` yet, so the ladder gets the sweep's naming | small, and in the same expression as #5 |
| 7 | **No hand-run entry point for one stage.** The wrapper no longer makes its own directory, so running a single stage needs a molbuilder command that does not exist yet | must be answered before the shell block is retired |
| 8 | **The history cannot be created.** Checkpoint init refuses this shape; no automatic checkpoints; no branch route | **blocking** — § 7 does not run at all |

### The one to fix first

**Gap 5**, because it is a regression rather than a missing feature: staged runs
carried correctly before attempt directories existed, and now they do not.

The fix is the one in § 6 — `submit` resolves the attempt, so it can write a
concrete carry link ([`project-layout.md`](?doc=execution/project-layout.md)
§ 1.3). **Gap 6 rides along**, being the same expression. And the shell block
that caused the regression is **retired rather than repaired**: its
from-inside-an-attempt guard, its `$PWD` logical-vs-physical bug, and its
`--force` refusal all stop existing once the caller decides the directory.

**The lesson is worth more than the fix.** The system was already built the right
way — `materialize` lays out directories and links, `submit` picks the working
directory and launches, and the wrapper activates and execs. I added a second
layout implementation in bash without checking whether one existed. The rule is
now written down where it can be pointed at
([`running-a-job.md`](?doc=execution/running-a-job.md) § 2.2a): **the wrapper
activates and execs; every directory and every link is Python's.**

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
