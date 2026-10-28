# How molbuilder works — what the parts are, and how they compose

**Role:** contract
**Domain:** (tree-wide)

**Companions — the contracts that own each part's internals, and where this
document and one of them disagree about a part's OWN rules, that one wins:**
[`engines/template.md`](?doc=engines/template.md) (the catalogue and the
template) · [`engines/stages.md`](?doc=engines/stages.md) (the description) ·
[`execution/generator.md`](?doc=execution/generator.md) (the pipeline's
mechanics) · [`execution/architecture.md`](?doc=execution/architecture.md) (who
owns which decision) ·
[`execution/job-contracts.md`](?doc=execution/job-contracts.md) (every file
format, and the shared vocabulary) ·
[`execution/project-layout.md`](?doc=execution/project-layout.md) (the
directories).

## 0. What this document owns

**It is the single contract for the WHOLE — what the parts are, and how they
work together.** Every part has a contract of its own that says what that part
*is* on the inside. Until this page existed, the sentence *"and here is how
they compose"* had no home, so it was written partly in four places and fully
in none.

| this document owns | it does not own |
|---|---|
| **which parts exist**, and the one name each is called by | what any one part contains — the settings an entry may carry, the keys a description may have |
| **what flows into what** — every arrow in § 3, and the whole road with its validation gates in § 9 | the format of the thing on either end of an arrow |
| **the doors** — that each file has one reader and one writer, and which they are | what those functions do internally |
| **the order**, and which step depends on which | the mechanics of any single step |
| **the invariants that only make sense across parts** — the machine never enters the portable folder; a door takes a whole object; two files made from one object are checked against each other | the reasoning behind each, which stays where it was argued |

**Two rules follow from that split, and they are what keep this document
true.**

> **R-W1 — this page states relationships, never restated details.** A fact
> that belongs to one part is named here and defined there. A second copy is a
> copy that drifts, and this project has paid for that twice: `stages.md` and
> `generator.md` came to contradict each other about where a benchmark's
> settings live, and a config class's help text drifted from the catalogue's.
>
> **R-W2 — a disagreement is a real defect, and which side is wrong depends on
> what is disagreed about.** If this page and a part's contract disagree about
> **that part's internals**, the part's contract wins. If they disagree about
> **how the parts compose**, this page wins and the other is the stale copy —
> that is the whole reason it exists.

**Plain language is a requirement here, not a courtesy.** This is the page a
person reads first, so a term is either an ordinary word or is defined in § 2
before it is used.

> **R-W3 — a contract document leads with "The short version"** *(user,
> 2026-08-28: key information has to come out — keywords, a diagram, an
> example; reading a record to find a rule is tiring)*. Every contract
> page opens with what the design IS and HOW it is used: the key rules as
> a keyword-first table pointing into the sections, plus the one diagram
> or worked example that carries the page. Below that line, **a rule
> states itself first and its history in one line, never as a story** —
> the *why chosen* survives; the narration of how it was found moves to
> the audit/archive if it is longer than a sentence. Scientific
> validation is the exception: its reasoning IS the content and keeps
> full depth.

---

## 1. The one idea

**You describe a calculation in files. The program turns those files into more
files. Nothing anywhere is written for a particular engine except the one
function that knows how to write that engine's input.**

That is the whole design in a sentence, and everything below is what it costs
to mean it.

A worked feel for it: you want to relax a gold–molecule junction with SIESTA.
You never hand-write a SIESTA input. You answer questions about the
calculation — what basis, how fine a grid, how many k-points — and those
answers land in a file. Later, on whatever machine you happen to be on, a
command reads that file and writes the SIESTA input, the shell script that runs
it, and the scheduler submission next to it. **Change an answer, get a
different input file. You never edit the program.**

---

## 2. The vocabulary, defined once

Six words are used throughout. Each is ordinary once you know what it points at.

| word | in plain terms |
|---|---|
| **catalogue** | The master list of *every* setting molbuilder knows about, for every engine, with its type, its allowed range, its unit, its default and its help text. One file, hand-edited: `molbuilder/data/catalogue.template.toml` |
| **template** | One calculation's copy of that list, narrowed to the engine it will run on, with the answers filled in. Named `<label>.template.toml` |
| **description** | What *changes* across a calculation — the sequence of steps, and which settings differ between them. Named `task.json` |
| **job request** | How and where the calculation runs — the queue, wall and memory asked, each step's run card (ranks, threads, GPU, solver), which steps share one job, a benchmark's plan, the reports. Named `job-request.json` |
| **deck** | The engine's own input file. `.fdf` for SIESTA, `.py` for PySCF. This is what the engine actually reads |
| **wrapper** | The shell script that activates the right environment, works out how many processors to use, launches the engine, and watches it. Named `<label>.run.sh` |
| **stage** | One step of a multi-step calculation — *coarse*, then *tight*. molbuilder's idea, not the engine's |

Two more you will meet in file names: an **attempt** is one try at running a
stage (`run-0`, `run-1`), and a **trial** is one point of a benchmark
(`bench-G0K16C1`).

---

## 3. The files, and how they relate

**Read this as: everything flows out of one hand-edited catalogue.**

```mermaid
flowchart TB
    CAT["<b>catalogue</b><br/><code>data/catalogue.template.toml</code><br/><i>every setting, every engine —<br/>type · range · unit · default · help</i>"]

    subgraph P["what a person answers"]
      FORM["<b>the form on a tab</b><br/><i>cards and controls, built FROM the catalogue<br/>— never hand-written</i>"]
      STRUCT["<b>the structure</b><br/><i>atoms, cell, regions</i>"]
    end

    CAT -->|"builds the form"| FORM
    FORM --> TMPL
    CAT -->|"narrowed to one engine,<br/>answers filled in"| TMPL

    subgraph FOLDER["one calculation = one folder"]
      TMPL["<b>template</b> · <code>&lt;label&gt;.template.toml</code><br/><i>every setting, WITH its answer</i>"]
      TASK["<b>description</b> · <code>task.json</code><br/><i>the steps, and what differs between them</i>"]
      REQ["<b>job request</b> · <code>job-request.json</code><br/><i>how and where it runs</i>"]
      XYZ["<b>structure pair</b><br/><code>&lt;label&gt;.source.xyz</code> + <code>&lt;label&gt;.source.molstruct.json</code><br/><i>the atoms, and where they came from —<br/>the .source segment is reserved, § 6.3</i>"]
    end

    STRUCT --> XYZ
    TMPL --> PREP
    TASK --> PREP
    XYZ --> PREP

    ENV["<b>the target machine</b> · <code>environment.json</code><br/><i>cores · GPUs · scheduler · the queues you can reach<br/>— probed on that machine, never part of the description,<br/>a copy kept beside it at the first prep</i>"]
    ASK["<b>what you ask for</b><br/><i>ranks · threads · GPUs · wall · queue — a stage's run card, execution</i><br/><i>queue · wall · memory — the calculation's allocation</i><br/><i>both in job-request.json; a prep command's flag wins</i>"]

    ENV --> PREP
    ASK --> PREP

    PREP{{"<b>prep</b><br/><i>the one command that turns<br/>a folder into something runnable</i>"}}

    PREP --> DECK["<b>deck</b> · <code>&lt;label&gt;_01_coarse.fdf</code><br/><i>the engine's input</i>"]
    PREP --> WRAP["<b>wrapper</b> · <code>.run.sh</code><br/><i>activate · launch · watch</i>"]
    PREP --> SB["<b>submission</b> · <code>.sbatch</code><br/><i>the scheduler header</i>"]
    PREP --> JS["<b>the plan</b> · <code>job-set.json</code><br/><i>what jobs this amounts to</i>"]
    PREP --> DIRS["<b>the directories</b><br/><code>01_coarse/run-0/</code>"]
```

### Why two files and not one

The template and the description look like they could be one file. They are
two because they answer different questions, and keeping them apart is what
makes a folder portable:

| | holds | changes when |
|---|---|---|
| **template** | *everything*, with the value in force — including the settings that vary, at their starting value | you change a setting |
| **description** | *which* of those you want to vary step by step, and each step's value | you add a stage, or promote a setting to vary |

So `mesh_cutoff` appears in both and says something different in each: the
template says **what it is**, the description says **that it steps, and to
what**. The rule and its reasoning are
[`stages.md § 6.2`](?doc=engines/stages.md).

### Why the machine is not in there

**The description holds no fact about a machine.** A machine's cores, GPUs,
scheduler and queues are its record, `environment.json`, never part of the
description — which is what lets you hand the same folder to a colleague on a
different cluster, or benchmark it on a short queue and run it on a long one,
**without editing a fact about either machine**. What a run *asks for* — its
ranks, threads and GPUs, its queue, wall and memory — is its own statement
in its job request, `job-request.json` ([`job-request.md`](?doc=execution/job-request.md)): a stage's run card (`execution`) states its ranks, threads,
GPUs, wall and queue; the calculation's `allocation` its queue, wall and
memory — the memory there alone; the run card wins over the allocation, and
the `prep` command over both. Nothing fills one in from the machine
([`execution/architecture.md § 5.2`](?doc=execution/architecture.md)); the
three — what the machine has, what you ask for, what a benchmark tries — are
[`generator.md § 4.1`](?doc=execution/generator.md).

**A machine's facts are probed ON that machine.** You can only probe the
machine you are standing on, and the ordinary workflow is to describe a
calculation on your workstation for a cluster you are not on. So the cluster's
record is written there — `jobset probe --write --name sol` on its login node —
and copied to where you prep. What stays in `molbuilder.json` is **your
preferences about this installation** — how a launch is sent, the env names,
where projects live, the server's settings — and how a shell enters an
environment here, `env_init`. A job's queue, wall and memory are the job's
own. The full split is [`configuration.md § 5`](?doc=configuration.md) M-1.

**What to measure is declared too** — `job-request.json`'s `bench`, the points to
try — and a benchmark is prepared **per stage** (`prep bench <stage>`),
because what runs fastest changes between a coarse step and a tight one:
different grid, different matrix, different best rank count. The measurement
is a report you read; that stage's run uses what you then write on its run
card ([`generator.md § 4.3a`](?doc=execution/generator.md)).

### Every file, and who owns it

The full registry — file name, its schema string, the module that owns it — is
[`job-contracts.md § 6.1`](?doc=execution/job-contracts.md).
The ones you will actually meet:

| file | plain meaning | format, and why |
|---|---|---|
| `catalogue.template.toml` | the master setting list | **TOML** — a person edits it by hand |
| `<label>.template.toml` | this calculation's answers | **TOML** — a person reads and edits it |
| `task.json` | the steps | **JSON** — mostly machine-written |
| `job-request.json` | how and where it runs | **JSON** — mostly machine-written |
| `<label>.source.xyz` + `.source.molstruct.json` | the atoms, plus where they came from | plain text + JSON sidecar |
| `molbuilder.json` | what **you** want from this installation — how a launch is sent, the env names, where projects live, the server's settings — and how a shell enters an environment here (`env_init`) | JSON, hand-edited |
| `environment.json` | what the target machine **is** | JSON, written by a probe |
| `job-set.json` | the jobs this folder amounts to | JSON |
| `<label>_01_coarse.fdf` | the engine's input | the engine's own format |
| `.run.sh` / `.sbatch` | run it / submit it | shell |
| `jobset-decisions.log` | every choice the program made, with its reason | one JSON object per line |

---

## 4. The doors — how code touches those files

**No part of the program opens these files by hand.** Each file has exactly one
function that reads it and one that writes it, and everything else calls those.
That is what stops two parts of the system disagreeing about what a file means.

```mermaid
flowchart LR
    subgraph READ["reading"]
      A["<code>template.load_catalogue()</code><br/><i>the master list</i>"]
      O["<code>calculation.open(base)</code><br/><i>the calculation, read once per verb:<br/>task.of · template_file.of · JobSet.of</i>"]
      B["<code>template.read_template(text)</code><br/><i>one calculation's answers</i>"]
      C["<code>task.of(base)</code><br/><i>the steps</i>"]
    end
    subgraph WRITE["writing"]
      D["<code>template.template_with_values(cfg, engine=)</code><br/><i>catalogue → this calculation</i>"]
      E["<code>task.write_task(base, task)</code>"]
      F["<code>describe.write_description(desc, dest)</code><br/><i>both files at once, all-or-nothing</i>"]
      L["<code>JobSet.write(folder, plan=)</code><br/><i>the plan, job-set.json</i>"]
    end
    subgraph MAKE["making things runnable"]
      G["<code>resolve.resolve(...)</code><br/><i>answers + machine + your ask<br/>→ the settled configuration(s)</i>"]
      K["<code>script_emit.prepare_deck(spec, …)</code><br/><i>the engine's input file — validate,<br/>render, write, then check what the writer made</i>"]
      H["<code>runwrap.write_run_wrapper(script, names=, resources=)</code><br/><i>the wrapper AND the submission</i>"]
      I["<code>materialize.materialize(jobset, base, plan, shape=)</code><br/><i>makes the directories</i>"]
      J["<code>submit.plan_launch(calc, mode=)</code> → <code>send_launch</code><br/><i>the plan, shown and asked, then sent as shown</i>"]
    end
    READ --> MAKE
    WRITE --> MAKE
    G --> K --> H
```

Three rules hold for every door in this list:

- **A generated file has ONE writer, and the writer keeps what you put in your
  own section** — a deck's and a wrapper's, the files that have one. Every
  deck and every wrapper goes through `write_script`,
  which merges back the `USER-CUSTOM` block from whatever was already on
  disk — so the one part of a
  generated file you are invited to edit survives the next `prep`. Any route
  reaching for a plain write would destroy those edits.
- **A door takes a whole thing, never a handful of its fields.** When a job's
  resources are handed to the wrapper writer, the *whole* set of resources
  goes, not `ranks` and `cores` picked out by hand. The reason is measured:
  when it was eleven separate values, two callers each passed a different
  subset, and each produced a submission file and a launch script that
  disagreed with each other. [`architecture.md § 3.1`](?doc=execution/architecture.md),
  rules **A8** and **A9**.
- **Where two files come out of one thing, they are checked against each
  other** — not just against what a test expected. Same section.

---

## 5. The order things happen in

**Nothing here is a convention; each step needs what the one before it
produced.**

```mermaid
flowchart TB
    S1["<b>1 · settle which machine this is for</b><br/><i>READ the record — the calculation's, else this machine's.<br/>Neither? prep refuses and names <code>jobset probe --write</code></i><br/>→ the first prep copies it in, <code>environment.json</code>"]
    S2["<b>2 · settle the numbers</b><br/><i>template + description + machine + your ask</i><br/>→ the configuration(s), in memory. <b>Writes nothing.</b>"]
    S3["<b>3 · write the engine's input</b><br/>→ <code>01_coarse/&lt;label&gt;_01_coarse.fdf</code>"]
    S4["<b>4 · write the wrapper and the submission</b><br/><i>reads the deck just written</i>"]
    S5["<b>5 · open the attempt</b><br/><i>copies what steps 3 and 4 wrote</i>"]
    S1 --> S2 --> S3 --> S4 --> S5
    S5 -.->|"all five decided with nothing written —<br/>then the folder's state is saved, then written"| W["<b>the plan, carried out</b><br/><code>checkpoint.save_before</code> → <code>Plan.carry_out</code>"]
```

**Why step 4 must come after step 3, and it is not just tidiness.** The wrapper
says what the deck it ships beside will do, read from the *finished* deck:
whether it lets the engine read the state a previous run left (what a retry
and `--continue` will actually do) and, for a SIESTA deck, which history files
it writes (each pointed at the run's own file) and how many atoms it has (the
orbitals-per-rank notice, which caps nothing). A wrapper written first
would describe a deck it had not seen. The environment it activates is the
job's GPU request, never the deck's text
([`execution/architecture.md § 9.2`](?doc=execution/architecture.md)).

**Why step 5 copies, and renders nothing.** The deck and the wrapper are written
once, in the stage's own folder; the attempt — the run directory, `run-0/` —
receives copies of them. What it takes from another stage's run — the warm
files it continues from — is its first launch's to take (`launch --from`),
copied in then; a SIESTA vibration's later calls are given their geometry by
the job itself, each call's `coordinates.fdf` beside its deck
([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11). A copy because the engine writes to those very filenames, and
a run directory hands the engine everything it needs and holds everything it
produces ([`project-layout.md`](?doc=execution/project-layout.md) § 1.0,
§ 1.6.5): a link would let a run reach back into what it started from.

**Nothing is written until all five are decided.** Prep plans every step with
nothing written, then saves the folder's state (`checkpoint.save_before`) and
only then writes the plan (`Plan.carry_out`) — so a refusal writes nothing but
its line in the ledger, and the state saved is the one a redo restores
([`job-system.md § 5.0`](?doc=execution/job-system.md)).

The step table with what each step may and may not do is
[`script-preparation.md § 5`](?doc=execution/script-preparation.md); the
sub-steps inside step 3 are § 3 there, and what each *engine* must supply at
each one is § 4.

### One run, or sixteen — the same pipeline

A benchmark is not a separate feature. Step 2 always produces a **list** of
settled configurations; an ordinary run is that list with one entry, a
benchmark is the same list with sixteen. Steps 3–5 loop over the list without
asking which they are in. That is why there is no second code path for
benchmarking, and why a third kind of study would need nothing new
([`generator.md § 2`](?doc=execution/generator.md)).

---

## 6. Where a person's answers get in

Three ways in, and they all end at the same files:

| you are | you use | it produces |
|---|---|---|
| working in a browser | the parameter tab, then the **Task setup** tab | the template, the description and the structure pair, in a folder you chose |
| working at a terminal | `molbuilder describe` | the same files |
| already have a folder | `molbuilder jobset prep task` | the runnable directories of the groups you pick — `--group N`, or its question, which offers every group not prepared and preparable |

The browser never writes an engine input. It renders the *form* from the
catalogue and asks the server to produce the files, because getting a format
right is the server's job ([`projects.md § 3`](?doc=web/projects.md)).

### 6.1 The one assumption on the target — a working conda or mamba

**Everything on the machine that runs the job rests on exactly one
prerequisite: conda or mamba works there, and the environments were
provisioned through it** ([`ops/installation.md`](?doc=ops/installation.md) —
one host env, one env per engine, all created by the bootstrap). Nothing else
is assumed: no PATH entry points, no pre-activated shells, no molbuilder on
the compute node.

Two scripts divide the work, and the split is the design:

| | knows | does |
|---|---|---|
| **the shell half** — `install-env.sh` once, every generated `.run.sh` per run | how to reach conda/mamba (the bootstrap finds it; the wrapper carries the hook line `prep` baked from this machine) | make the environment right, then hand over — activate and exec, nothing else ([`running-a-job.md § 2.2a`](?doc=execution/running-a-job.md)) |
| **the Python half** — molbuilder before the run, the engine and the monitor (`mb_monitor.pyz`) during it | everything that computes | describe, resolve, render, arrange, watch |

**And the spelling convention every document uses:** `molbuilder <verb>`
means `python -m molbuilder <verb>`, run in the activated host env from the
repo clone. That is the **supported form** — molbuilder is deliberately *not*
pip-installed into its env ([`ops/installation.md`](?doc=ops/installation.md);
a console-script entry point exists in `pyproject.toml` but is not the
supported invocation). An evaluation that goes looking for `molbuilder` on
`PATH` is testing an assumption this design never makes — made once,
2026-08-19, which is why this section exists.

### 6.2 Telling a verb which job you mean

**You do not have to stand in the calculation** *(user, 2026-08-22)*. Every
verb that acts on one takes `--bundle`, and it reads the path **from the
projects root** — the same three levels the tree is built from:

```
--bundle <project>/<topic>/<calculation>
```

So one place to stand, any job:

```bash
cd ~/molbuilder                                  # where molbuilder runs from
molbuilder jobset status         --bundle Au-BDT-Au/optimization/Relax
molbuilder jobset prep   bench coarse --bundle Au-BDT-Au/optimization/Relax
molbuilder jobset launch bench coarse --bundle Au-BDT-Au/optimization/Relax --mode submit
molbuilder jobset summarize bench coarse --bundle Au-BDT-Au/optimization/Relax
```

**Omit it and the current directory is used** — the short form once you have
`cd`'d into the folder to look at output:

```bash
cd ~/molbuilder/projects/Au-BDT-Au/optimization/Relax
molbuilder jobset status
```

Either way the folder must be **inside the projects tree**. A path that
climbs out with `..`, or an absolute path somewhere else, is refused by
name rather than acted on — a calculation outside the tree is not one
molbuilder manages ([`job-contracts.md § 2.5b`](?doc=execution/job-contracts.md)).

**If your calculations do not live under the checkout**, that is a setting,
not an obstacle: point the tree somewhere writable and every surface follows
it at once — the verbs, the sidebar, the workspace store.

```json
{ "paths": { "projects": "/scratch/u/projects" } }
```

in `molbuilder.json`, or `MOLBUILDER_PROJECTS=/scratch/u/projects` for
one shell ([`architecture.md § 8.2`](?doc=execution/architecture.md)).

> **One address, one anchor — with one exception you will meet at `launch`.**
> `--from 01_coarse/run-0` (and a force-constant stage's `prep --from
> 01_relax/run-0`) names an attempt *inside* the calculation, and is
> measured from the bundle, not from the projects root. That is deliberate:
> a calculation is self-contained and travels, so the name of one of its own
> attempts has to keep working after the folder is copied to a cluster
> (`job-contracts.md` § 2.5b).
>
> **`molbuilder describe` is the verb that creates a calculation**, and it names
> both the calculation and its structure from the projects root like
> everything else:
>
> ```
> molbuilder describe --structure P/structure/water.xyz \
>                        --bundle   P/frequency/water-vib \
>                        --shape hierarchical --engine pyscf \
>                        --calculation vibration
> ```
>
> It is the one verb whose `--bundle` may not exist yet — that is what it
> is for, and why it, too, takes its paths from the projects root: a
> calculation created outside the tree is one no other verb would touch.


## 7. What is genuinely data-driven today, and what is not

**This section is the honest one, and it is the reason the rest of the page is
worth reading carefully.** Both engines ride the one pipeline — describe →
prep → launch → results — through the engine seam (`jobset/engines.py`,
`engine_seam`).

### Data-driven, and it works

**Anything about a setting.** Its value, its allowed range, its unit, its
label, which panel it appears on, its help text, whether it may be varied per
stage, whether it may be benchmarked — all of it is the catalogue. Adding a
setting to an engine that already exists is a catalogue entry plus a line in
that engine's translator. Two tests hold the boundary in both directions: a
setting the catalogue does not carry is one no surface can offer, and a
catalogue entry no engine can hold is a dead entry.

**Writing the decks.** Both engines answer the seam's questions —
[`generator.md § 7`](?doc=execution/generator.md) counts the catalogue half,
[`script-preparation.md § 4`](?doc=execution/script-preparation.md) the code
half — and one conductor (`prepare_deck`: validate → render → write → check)
runs every deck of every engine through the same steps. Four of PySCF's
answers are a recorded *nothing*, which is the design working: a decision a
reader can check, not an arm nobody thought about.

### Not yet — and it is a known, named gap

**The WRAPPER writer still carries per-engine facts in its own body.**
`runwrap.py` branches on which engine it is writing for — what a cold restart
clears, how the label is read back out of a deck, how the launch line is
formed. Those facts belong beside the engine's other seam answers, and moving
them is the recorded seam item **W1**
([`backend-architecture.md § 5`](?doc=backend-architecture.md); its measured
state, [`plans/plan.md`](?doc=plans/plan.md) § 2, **S1**). Until then, adding an
engine edits `runwrap.py` — which is exactly what
[`generator.md § 7`](?doc=execution/generator.md)'s *"adding an engine adds
files and edits none"* test exists to catch.

**Every kind rides this pipeline.** A vibration is a calculation KIND —
`render_deck` runs its deck like any other, and § 9's gates cover it — and
transport is a different KIND of job
([`execution/architecture.md § 0`](?doc=execution/architecture.md)) — one
citation of a finished junction run, five derived stages — prepared,
launched and summarized through the ordinary jobset verbs
([`engines/transport.md`](?doc=engines/transport.md) § 1, § 3.1).

**Where this is tracked:** [`plans/plan.md`](?doc=plans/plan.md) § 2, S1.

---

## 8. Which document owns which part

**This page owns the composition (§ 0); each document below owns one part's
internals.** When you need a part's own rules rather than how it fits:

| you want the rule for… | open |
|---|---|
| what a setting entry may contain | [`engines/template.md`](?doc=engines/template.md) |
| the steps, and what may differ between them | [`engines/stages.md`](?doc=engines/stages.md) |
| the pipeline, and what bounds a benchmark | [`execution/generator.md`](?doc=execution/generator.md) |
| who decides what, and the rules that must never break | [`execution/architecture.md`](?doc=execution/architecture.md) |
| every file format, and what each name means system-wide | [`execution/job-contracts.md`](?doc=execution/job-contracts.md) |
| what a folder looks like, and what `prep` does to it | [`execution/project-layout.md`](?doc=execution/project-layout.md) |
| why all of a calculation's files share one name | [`execution/run-identity.md`](?doc=execution/run-identity.md) |
| running one job, day to day | [`execution/running-a-job.md`](?doc=execution/running-a-job.md) |
| the whole thing done once, with a real molecule | [`execution/worked-example.md`](?doc=execution/worked-example.md) |
| what is left to build | [`plans/plan.md`](?doc=plans/plan.md) |

---

## 9. The whole road, one picture — and where validation stands on it

**Every stage below is owned by one document (§ 8); this picture owns only
how they chain.** The circled numbers are the validation gates — a
calculation cannot reach the next stage without passing the gate between
them, and each gate refuses with the *reason*, never a stack trace.

```mermaid
flowchart TB
    subgraph DESCRIBE["describe it — the browser (or the CLI)"]
        CAT["<b>the catalogue</b> — <code>catalogue.template.toml</code><br/>every parameter defined ONCE; kinds narrow it<br/>(<code>engines</code> / <code>calculations</code> keys — template.md § 6.3)"]
        STRUCT["<b>structure preparation</b> — Build · Modify · MolView<br/>labels, regions (frozen atoms), cell ride the model<br/>and its sidecar (model/structure-molstruct.md)"]
        FORM["<b>the calculation's form</b> — Build tab (optimization) ·<br/>Spectrum tab (vibration)<br/><code>/api/build/schema/&lt;engine&gt;?calculation=…</code>,<br/>one shared renderer (web/form-schema.md)"]
        DETECT["charge and spin — <code>/api/structure/analyze</code><br/>the one electronic state, for exactly what the form says,<br/>shown on the card; fills nothing (chemistry-correctness.md § 2a)"]
        SEND["<b>Send to Task setup</b> — <code>lib/task-handover.js</code>,<br/>ONE door for every tab (web/handover-procedure.md)<br/>writes template + structure pair + <code>task.1st.json</code>"]
        TS["<b>Task setup</b> — shape, stages, bench axes<br/>(web/task-setup.md); saving writes <code>task.json</code><br/>— the described calculation — and <code>job-request.json</code>,<br/>how it runs (execution/job-request.md)"]
    end
    subgraph MACHINE["run it — the machine that will compute"]
        PREP["<b>prep</b> — the conductor renders the deck<br/>(execution/script-preparation.md § 5’s five steps,<br/>workflow.md § 5), wrapper + dirs follow the deck —<br/>planned whole, the folder's state saved, then written"]
        RUN["<b>launch → run</b> — the plan shown and asked, the state saved<br/>at a group's first launch, the header prep wrote sent —<br/>one wrapper, <code>&lt;label&gt;.run.sh</code> (execution/running-a-job.md),<br/>the deck phase-writes its artifact atomically"]
        SCR["<b>jobset scratch</b> — an engine's scratch a run left,<br/>listed, and emptied with --empty from every run not going<br/>(never saved by a checkpoint, and a restore removes it)"]
    end
    subgraph READ["read it — results presentation"]
        ART["<b>the artifacts</b> — trajectory · logs ·<br/><code>.spectra.json</code> (schema-versioned;<br/>readable sets, one home per format)"]
        RES["<b>Results tab</b> — phase chips, charts,<br/>thermo plots, mode viewer (web/results.md,<br/>web/spectra.md)"]
        OBS["<b>jobset status · summarize</b> — where each stage<br/>has got to, and what has landed, added up<br/>(execution/job-system.md § 5.3)"]
    end

    CAT --> FORM
    STRUCT --> FORM
    FORM --> DETECT
    FORM -->|"① live science preflight"| SEND
    SEND -->|"② the cell gate"| TS
    TS -->|"③ task preflight"| PREP
    PREP -->|"④ the science gate — STEP 3.3"| RUN
    RUN --> ART
    RUN -.-> SCR
    ART -->|"⑤ the parse gates"| RES
    ART -->|"⑤"| OBS
```

**The five gates, and what each refuses** *(“science validation is a must” —
user, 2026-08-21; a gate that exists but does not run is the failure class
this table exists to rule out)*:

| gate | fires | refuses / warns on | owner |
|---|---|---|---|
| **① live preflight** | on every form edit, Build tab (`/api/build/preflight`) | the same `validate(struct, cfg)` verdict gate ④ will give — surfaced while the person is still at the form | [`science/validation.md`](?doc=science/validation.md) § 4 |
| **② the cell gate** | at Send, on the exported envelope | a box the calculation cannot use (degenerate cell, left-handed axes); *notices* (thin vacuum) hold the navigation, never the write | [`web/handover-procedure.md`](?doc=web/handover-procedure.md) |
| **③ task preflight** | at every describe — Task-setup save, `molbuilder describe` (its transport arm too), the Transport tab's Send — and at dispatch | a description that is not one: unknown keys, identity clashes (the § 6.6 check table), empty ladders (§ 6.5), bench entries that name no execution item and points that fit no item (`job-request.md` § 3.4; `generator.md` § 4.3a), a stage's value past its hard limit; it *warns* on a value, or a bias point, outside its recommended range ([`engines/template.md`](?doc=engines/template.md) § 5.3) | [`engines/stages.md`](?doc=engines/stages.md) §§ 6.5–6.8 |
| **④ the science gate** | at `prep`, inside the conductor’s STEP 3.3 — **“here, and only here”**, so no deck route can forget it | cell + geometry + field ranges and choices (`validation/metadata.py`) + the electronic state's one family (charge, parity, open shell, what the engine can run — `validation/chemistry.py`) + the engine's and the kind's science (grid, amplitude — `validation/{siesta,pyscf,spectra}.py`); errors refuse the deck, warns reach stderr. And at the record step, the placement check: a spec with no engine frame, and any atom outside the cell along a non-periodic lattice vector, are refused (`cell.require_placed`, `model/structure-periodicity.md` § 6.0, check 3) | [`execution/script-preparation.md`](?doc=execution/script-preparation.md) §§ 3–4 (step 3.3) |
| **⑤ the parse gates** | at every artifact read (Results tab, CLI) | a file the reader cannot vouch for: unknown schema version (readable **sets**, additive bumps read old files whole), malformed fields — each a **typed** refusal the UI can react to | [`web/spectra.md`](?doc=web/spectra.md) § 6, the parse layer |

*(Every kind passes the same five gates: the Spectrum tab runs the same live
check (①) Build does, and transport's decks render through prep like every
other kind's, its own science the KIND's — `_KIND_VALIDATORS["transport"]`,
which fires on every rung.)*

### Where the details live — the drill-down for each arrow

The picture above is deliberately a MAP; the details a builder needs are
each in one owning document:

| you want to see… | open |
|---|---|
| how the pipeline **branches per engine** — what each engine supplies at every prep sub-step | [`execution/script-preparation.md`](?doc=execution/script-preparation.md) § 4 (the per-engine table), § 5 (the step table) |
| the **file-to-file information flow** — which file feeds which, one reader/one writer per file | § 3 of this page (the files diagram), [`execution/job-contracts.md`](?doc=execution/job-contracts.md) (every format) |
| how **template information flows** — catalogue → per-calculation template → resolved config → deck | [`engines/template.md`](?doc=engines/template.md) § 6 (incl. § 6.3’s kind-narrowing protocol), [`execution/generator.md`](?doc=execution/generator.md) § 4.3 (precedence: template < the calculation's `execution` < the rung's < flags) |
| how the **UI flows** — the catalogue-rendered form, its collected params, the hand-over’s four files, `task.json` | [`web/form-schema.md`](?doc=web/form-schema.md), [`web/handover-procedure.md`](?doc=web/handover-procedure.md) §§ 2–6, [`web/task-setup.md`](?doc=web/task-setup.md) |
| **which checks run for a given procedure**, and each check’s science | [`science/validation.md`](?doc=science/validation.md) § 1 (the layers), § 7 (where every validator lives — engine registry by config type, kind registry by declared fact) |
| where a **person’s answers enter** the flow | § 6 of this page |

*(A diagram is a claim like any other: reviewing cited diagrams against the
code they draw is part of the audit playbook.)*

