# Task Setup — the tab that writes a description

**Role:** contract
**Domain:** web

**Companions — the contracts this surface is built against, and where the two
disagree those win:**
[`engines/stages.md`](?doc=engines/stages.md) — what a stage is, `varies`,
`shape`, `bench`, and the rule that a job always has at least one stage;
[`engines/template.md`](?doc=engines/template.md) — the catalogue every control
on this page is drawn from, and the rule that a description names no machine;
[`execution/project-layout.md`](?doc=execution/project-layout.md) — what `prep`
does on the target, and why this page cannot do it;
[`execution/generator.md`](?doc=execution/generator.md) — *a run is a sweep of
length one*, which is the idea § 5 is built on;
[`execution/run-identity.md`](?doc=execution/run-identity.md) — the id, and what
`start from` expands into;
[`web/form-schema.md`](?doc=web/form-schema.md) — how a catalogue item becomes a
control.

**What this page is for, in one sentence.** A parameter tab has collected the
physics; this page turns that into a **description on disk** — `task.json` plus
the template — and hands you the command to run it somewhere else.

---

## 1. The one rule the page rests on

> **The browser describes. The terminal acts.**

A deck carries values that depend on *how it will be launched* — a block size
whose sensible ceiling is orbitals over ranks, a GPU line that decides which
environment the wrapper activates. **A parameter that depends on the launch
cannot be decided before the launch is known**
([`project-layout.md § 2.2`](?doc=execution/project-layout.md)). A browser on a
laptop does not know the machine, so a deck it "finished" would be full of
guesses.

So this page writes what any machine can read, and stops:

```text
projects/BDT-Au/optimization/au111-series/bdt_au/

    BDT_Au_relax.template.toml   the parameters, with their values
    task.json                    what varies, the stages, what to measure
    Au.psml  S.psml  …           the data files
```

Then, on the machine that will run it:

```bash
molbuilder jobset prep run coarse      # resolves this machine, renders the
                                       # deck and wrapper, builds the attempt
molbuilder jobset launch run coarse
```

**There is no "run all stages" button, and that is deliberate.** A stage is a
long job, and a chain that continues by itself can spend a week refining a
geometry you would have rejected in a minute
([`project-layout.md § 1.6`](?doc=execution/project-layout.md)). What the page
offers is the next single command.

---

## 2. Where it saves

**The folder selected in the projects sidebar.** This page creates no folders —
make or pick the one you want there first.

**One rule about which folder: somewhere under a topic.** How deep, and how you
group jobs below that, is yours — nest as many folders as suits the work. The
projects root, a project, and a bare topic are not destinations
([`job-contracts.md § 2.5`](?doc=execution/job-contracts.md) fixes the nine
topics).

**Only `task.json` is read to decide what the folder is.** It names the
calculation and its structure, so nothing else in the folder needs inspecting.
Three states, and the third is a refusal:

| what is in the folder | what saving does |
|---|---|
| **no `task.json`** | writes a new job here |
| **a `task.json` for *this* calculation** | its stages load into the table; saving updates them |
| **a `task.json` for a *different* calculation** | **refused.** One job per folder, or every later step is guessing which one the folder is (`job-contracts.md § 2.1` Rule 1). The message names the calculation it found |

> **Why refuse rather than offer a subfolder.** Creating a folder to escape a
> collision is a decision about where your work lives, and the sidebar is where
> that decision belongs. A page that quietly nests one is a page that puts
> calculations somewhere you did not choose.

### 2.1 The page holds no state of its own

Everything it shows comes from the selected folder. There is no remembered form,
no in-progress buffer that outlives a directory change, nothing to reconcile on
reload. **The folder is the only link** — which is what lets the same page serve
every producer, and what makes *"open a folder"* a load rather than a merge
(§ 9).

### 2.2 What this page asks, and what `_is_bundle_root` asks

*Corrected 2026-08-16.* This section used to say the check **is**
`checkpoint.py::_is_bundle_root` and "must not be written a second time". That
was wrong in a way worth keeping: **the two answer different questions**, and
the tab neither uses nor duplicates it.

| | asks | answers |
|---|---|---|
| `checkpoint.py::_is_bundle_root` | *does this path declare itself the root of one multi-directory unit of work?* | one boolean, over `task.json` **or** `job-set.json` — enough to decide the folder owns its subdirectories (`checkpointing.md` **L1**) |
| **this page** | *does this folder hold a description to edit, a hand-over to finish, or nothing yet?* (a transport folder is always the FIRST state: the composite describes directly and never produces a hand-over — its `task.json` arrives finished, and this page reads it as an ordinary description whose five stages and shape are fixed) | **three** states, and it distinguishes `task.json` from `task.1st.json` — a distinction `_is_bundle_root` does not make and has no reason to |

**The invariant that does bind them**, and it holds today: a folder this page
treats as a described calculation is one the checkpoint system also covers.
Both key on `task.json`, so they cannot disagree about that — which matters
because § 8 takes a state in a folder before writing into it, and a folder the
history did not cover would be one whose state could not be brought back.

**And a folder of finished decks is still not adoptable**, which is a refusal
rather than a missing feature: `varies` **cannot be inferred**
([`stages.md § 6.2`](?doc=engines/stages.md)). Decks do not record which
parameters were *meant* to vary — a cutoff equal in every stage is
indistinguishable from one nobody promoted — so reconstructing intent would be
guessing at the one thing the description exists to state.

---

## 3. What came over, and is not editable here

The identity facts, read-only, from the parameter tab that wrote them: the
calculation's name, the engine, the structure file, its formula and atom count,
and the label every emitted file is stemmed on.

**They are shown because you are about to commit a week of compute against
them** — not so they can be changed. Changing them is changing what the
calculation *is*, which is the other tab's job.

The **id** (`<label>_<formula>`) is displayed and never recomputed here: it is
read from the file and checked, which is what makes renaming detectable rather
than silent ([`run-identity.md § 3`](?doc=execution/run-identity.md)).

---

## 4. Shape — asked, never guessed

`"flat"` or `"hierarchical"`, and the page cannot be saved until one is chosen.

| | what it means |
|---|---|
| **flat** | one directory. Stages and attempts told apart by filename — `…_01_coarse-run0.out`. One shared set of warm files, so each stage overwrites the last |
| **hierarchical** | a directory per stage, a directory per attempt — `01_coarse/run-0/`. Every stage's state stays on disk |

**No default, on purpose.** Inferring it — from the stage count, or from what is
already on disk — would hand somebody a directory tree they never asked for, and
the trade it decides (what survives after three stages have run) is theirs to
make ([`stages.md § 6.7`](?doc=engines/stages.md),
[`project-layout.md § 1.2`](?doc=execution/project-layout.md)).

**It is fixed once the calculation has produced.** Before the first produce it is
free to change; after, it would orphan every deck, output and warm file.

---

## 5. Stages — the table, and what a column means

Rows are stages, columns are parameters. **Add a parameter to make it a column,
then give it a value per stage. Anything you do not add stays as you set it, for
every stage.**

| Stage | mesh_cutoff | relax_force_tol | relax_type | restart |
|---|---|---|---|---|
| coarse | 150 | 0.04 | CG | clean |
| medium | 200 | 0.04 | Broyden | continue |
| tight | 300 | 0.01 | Broyden | continue |

**The column set is `varies`; the cells are each stage's `overrides`**
([`stages.md § 6.2`](?doc=engines/stages.md)). A cell left empty means *this
stage uses the template's value* — a real state, and why `overrides` is a subset
of `varies` rather than equal to it.

### 5.1 An empty cell shows the number, and it comes from the folder

An empty cell says *"the template's value"* — so it **shows that value**, greyed,
as a placeholder. That is what makes the § 9 rule *"adding a column changes
nothing on screen"* visible instead of merely true: promote a parameter and every
cell shows the number it was already at.

**The value is read from `<label>.template.toml`, not from the catalogue.** The
distinction is the whole point of the hand-over: a person who chose a 4×4×1
k-grid in the parameter tab has that grid in the template, and a cell naming the
catalogue's recommendation would be naming a number the job will not run. Order,
and it is one way round:

| | shown when |
|---|---|
| the folder's template value | the template answers this parameter |
| the catalogue's `default` | it does not — nothing was sent, or the sender left it alone |

`GET /api/task-setup/template-values?dir=` reads it, because TOML is a format and
[`projects.md § 3`](?doc=web/projects.md) keeps a format's correctness on the
server. It parses with `read_template` — the same reader `prep` opens the file
with — so the browser cannot become a second reader that disagrees.

**A hover shows both numbers when they differ**: *"This job (probe.template.toml):
450.0 Ry"* above *"Recommended: 300.0 Ry"*. Somebody checking a description
before a week of compute should be able to see that the two are not the same.

### 5.2 The declared type decides the cell — both the widget and the value

A parameter's `type` comes from the catalogue and answers **two** questions, and
neither may be answered by looking at the value instead:

| | asked by | answered by |
|---|---|---|
| which control edits this cell | `legalValues` | `bool` / `enum` → a dropdown; everything else a text box |
| what the typed text **means** | `CELL_READERS` (`task-setup/cell-readers.js`) | one reader per member of `template.TYPES` |

Guessing either from the value's look is a bug this tab has had twice. A
sweep row born as the number `1` made `use_gpu` a number box (fixed
2026-08-20). And a cell that stored `Number(text)` when the text parsed and
the **raw string** when it did not wrote `"kgrid": "4,4,1"` into a
description — a string where the config declares `Tuple[int, int, int]`.
`kgrid` is `int3`, `Number("4,4,1")` is `NaN`, and the four columns whose
type is a sequence (`kgrid`, `kgrid_displacement`, `species_order`,
`ecp_atoms`) all wrote text. It saved cleanly and failed at `prep`, hours
later, inside a range check that could only call it *"a programmer bug"*
(reported live 2026-08-25).

So: **every type in the vocabulary has a reader, and a type the catalogue can
declare but the tab cannot read is a test failure**, not a silent fallback —
that is the only thing that makes the next addition noisy.

**Three types have a reader and no catalogue item**: `pow2`, `text` and
`intlist`. A column's type comes from the shipped catalogue, so no cell can
carry one today — they are handled in advance because the failure without a
reader is *silent* (the lookup misses, the raw text is stored), and a newly
added item is exactly when nobody thinks to check. So **when such an item
does land, the stage table already reads it** — `pow2` as a whole number
(snapping is `template._shape`'s, not the cell's), `text` verbatim, and
`intlist` as comma- or space-separated whole numbers (not the range syntax
`0-35, 100` the Build form's own control takes — that is a different
control's contract). Do not add a second path for them.

A k-grid takes the three spellings `--kgrid` itself takes — `4,4,1`,
`4x4x1`, `4 4 1` — because one product should not accept a value at the
terminal and refuse it in the browser. Text that is **not** the declared
type is kept exactly as typed and refused by name at the save door
([`stages.md § 6.6`](?doc=engines/stages.md)'s declared-type row); storing a
half-parsed value would be the quiet version of the same bug.

**A job always has at least one stage.** You start with `coarse`; adding another
is one more row. There is no stage-less shape to fall into, so a job's artifacts
are named the same way from the first run — `01_coarse` — and a job that grows a
second stage needs nothing renamed or renumbered
([`stages.md § 6.5`](?doc=engines/stages.md)).

**Names are words, and the ordinal is not part of the name.** The shipped ladder
is `coarse` / `medium` / `tight` (`config/siesta.py::SIESTA_STAGE_NAMES`), each
arriving with its tier's preset ([`tuning.md § 4`](?doc=engines/tuning.md)).
Position is carried by the artifact token — `01_coarse`,
`identity.stage_token` — so the name never repeats it, and a stage's `seq` is
assigned once and never reassigned
([`project-layout.md § 4.2`](?doc=execution/project-layout.md)).

**A name matches `[A-Za-z0-9_]+` — no hyphen** — because a hyphen announces *a
counter follows* everywhere else in the system, and stage names are compared
case-insensitively, because they key filenames
([`stages.md § 2`](?doc=engines/stages.md)).

---

## 6. The machine — what to measure, and what to run

**This is the page's other half, and it is where the GPU is decided.**

**Two cards, and they are independent** (§ 6.2b): *What to measure* declares
the grid, *What the run will use* states one value each. Both go through the
same resolver — a run is a sweep of length one
([`generator.md § 2`](?doc=execution/generator.md): *`prep` step 2 always
produces a list of resolved configurations; a production run is that list with
one element, a benchmark is the same list with N*) — and neither reads the
other's block.

| card | row | what that means |
|---|---|---|
| measure | `mpi_np` `4` `8` `16` | 3 trials |
| measure | `omp_threads` `1` `2` | × 2 → 6 combinations |
| run | `mpi_np` = `8` | what the run asks for |
| run | *(blank)* | `run-config.toml`, then the wrapper's policy |

**A one-point row in the MEASURE card is still one trial, not a decision** —
except for the non-machine items, where the 2026-08-20 override rule stands:
`use_gpu: [true]` pins the template for the trials and the run alike
(`generator.md` § 4.3a).

### 6.1 Two kinds of setting live here, and the difference is not cosmetic

The `staging` group is what a parameter form deliberately does not ask
([`form-schema.md § 1.3`](?doc=web/form-schema.md): `catalogue_to_form_schema`
filters it). Inside it, the catalogue draws a line:

| | which ones | may carry a value? |
|---|---|---|
| **you answer it** | everything in the group the machine does not answer | **yes.** It has a real default — `false`, `clean`, `1` |
| **the machine answers it** | the ones marked as the machine's | **no.** Reading a template with a value on one is **refused** ([`template.md § 6.4`](?doc=engines/template.md)) |

**Neither row is a list, and neither should be written as one.** Each setting in
the catalogue already says whether the machine answers it, so asking that
question sorts the group in two with nothing to keep in step. For SIESTA today
that comes out as `restart` · `continue_retries` · `use_gpu` on the first row
and `mpi_np` · `omp_threads` · `max_memory_mb` on the second — but those are what
the answer *is* right now, not what it is defined as.

> **Corrected 2026-08-18.** Both rows used to be typed out here, and the second
> described the mechanism that sorted them — *"each names an allocation
> resolver"* — which had already been replaced by a single mark on each setting.
> So this table was a hand-kept copy of an answer the data gives, written in the
> vocabulary of a mechanism that no longer existed. The first row was the copy
> that mattered: `restart` sat in it, correctly, while the table on this same
> page could not offer `restart` as a column
> ([`stages.md § 6.2`](?doc=engines/stages.md)) — the page's two halves
> disagreeing about one setting, in one document, three sections apart.

So ranks, threads and memory in the **measure** card are always *points to
try* — one of them is one trial, never an answer. What a run uses is the
**run** card, `execution`'s own block (§ 6.2b), and the two cards never read
each other.

> **The line is decision vs finding, not portable vs not.** A description may
> hold *"run it at eight"* — that is a person asking — and may never hold what
> a **machine found**, which is why `summarize` writes its verdict to
> `run-config.toml` instead. The argument is
> [`generator.md § 4.3a`](?doc=execution/generator.md)'s;
> [`architecture.md § 5.2`](?doc=execution/architecture.md) carries the
> ladders.

> **This is about CATALOGUE ITEMS, and the tab also holds something else**
> *(2026-08-24)*. The queue card ("Where it runs, and what it may use") sets
> `task.json`'s `allocation` — `domain` / `time` / `mem` — and those are **not**
> points to try. The distinction is the one § 6.8a of
> [`stages.md`](?doc=engines/stages.md) draws: `max_memory_mb` above is a
> per-rank `ulimit` the *deck* carries, a machine-answered template parameter;
> `allocation.mem` is what the *job asks the scheduler for*. A queue name and a
> wall are decisions about this calculation in exactly the way *"use 16
> ranks"* is. What separates the two blocks is not portability but **who is
> being asked**: the scheduler, or the launch.

### 6.2 Why the GPU is decided here and nowhere else

> **Decided by the user, 2026-08-16: *"use GPU or not is set up only at the Job
> Prep UI."***

> **This section owns the SURFACE. The cross-engine rule is
> [`engines/overview.md`](?doc=engines/overview.md) § 3a** (G-1…G-5a) —
> including the one thing this page cannot tell you: **what happens when you
> ask for a GPU and there is none.** The run stops; there is no CPU fallback,
> in either engine, in a run or in a benchmark trial.
>
> *Cross-linked 2026-08-17. Until then this was the only place the GPU
> decision was written down, and it is a **web tab's** contract — so a reader
> coming from the engine side never found it and re-derived the rule instead,
> differently each time. That is why the question kept coming back.*

Three things are true of it at once, and no parameter tab can know any of them:

- **It depends on the machine.** Whether there *is* a GPU is a fact about the
  target, and a description names none.
- **It is a judgement about this problem's scale, not its physics.** CPU is
  often the faster answer — a small system pays more in GPU launch overhead than
  the eigensolve saves ([`siesta.md § 7.1`](?doc=engines/siesta.md)). Choosing
  CPU on a machine that has a GPU is ordinary and deliberate.
- **It can be measured first and decided after.** Add a second point and it
  becomes a bench axis; `summarize` writes `bench-result.json`, whose
  `choice.mechanism` carries `use_gpu`, and `prep run` **offers** the verdict
  and waits ([`project-layout.md § 2.3.3`](?doc=execution/project-layout.md)).

**What it is not is the eigensolver.** `diag_algorithm` is a `budget` item on the
parameter tab, and it decides no environment and no resource — the packaged
SIESTA carries ELPA through ELSI and runs it on CPU
([`siesta.md § 7.2`](?doc=engines/siesta.md)). Only `Diag.ELPA.GPU true`
re-routes the wrapper. The two are easy to confuse and belong on different
surfaces.

### 6.2a What the machine would actually run, live in the card

The rows above say what to *try*; this block says what the chosen machine
would actually **run**, and it updates as the points are edited *(user,
2026-08-30: "can't this list be just updated in the same card where the
parameters are set… this does not need to be a message with a window")*.

The heading counts them — *"6 combination(s) — 4 fit a queue, 2 do not"* —
and each surviving cell shows its label (the same token its trial directory
will carry), its shape, and the queues that would take it.  **"a queue", not
"this machine":** the menu is the target's, and preparing for a machine you
are not standing on is ordinary here. Cells no queue
here can hold are shown **struck, with the numbers** rather than dropped:
a point that silently vanished is the thing that sent someone to the CLI
to find out why.

Three rules make it trustworthy rather than merely convenient:

- **The browser does not enumerate it.** `POST /api/task-setup/bench-grid`
  hands the axes to `_bench_inputs` — the one enumerator, the same one
  `prep` runs — and returns its report. A grid computed in the page would
  be a second decider, free to say a cell is fine where `launch` refuses
  it. That is the exact failure this whole lane was rebuilt to remove.
- **It reads the axes the rows were painted from** — the task
  `renderMachine` was handed, which is the in-memory model, not the file.
  So the list follows what is being typed rather than the last save, and
  it cannot describe a different object than the rows above it. (It stays
  quiet in handover and empty modes: there is no `task.json` for the door
  to resolve against yet.)
- **It is a nicety, and behaves like one.** A stale answer is dropped (a
  sequence guard — typing outruns the network), the request is debounced
  to one call per pause, and a failed fetch hides the block and leaves the
  card working. **A surface that cannot get its nicety shows what it has;
  only one that cannot get its *substance* may refuse** — the rows are the
  substance here. That rule was learned the hard way on 2026-08-23, when
  an unguarded label fetch left the whole bench card absent and the report
  was "the bench setup is gone".

It shows **what the scheduler would accept** — a wall, a core count, a
gres type, a device count, a policy cap. It does not predict whether the
calculation will fit in the card it asked for; that is a runtime failure
the person deals with, and [`scheduler.md § 0`](?doc=execution/scheduler.md)
says why submission must not answer it.

### 6.2b What the run will use — its own card, one value each

*(user, 2026-09-02: "run parameter is independent from bench grid or bench
result. User can run without bench… bench and run share the same framework to
choose/set parameters but run does not produce grid but a single user defined
condition to run." And: "bench is bench and run is run. They are structurally
separated in their dir and they are functionally different as the user decides
to use either.")*

**Two cards, because they answer two questions**, and neither reads the
other:

| card | writes | shape |
|---|---|---|
| **What to measure** | `task.json`'s `bench` | several values per row — one trial per combination, in `<stage>/bench/` |
| **What the run will use** | `task.json`'s `execution` (`stages.md` § 6.8d) | **one value each** — the run, in `<stage>/run-N/` |

**The run card offers the bench's parameters and does not read them.**
Whatever the grid varies appears here as a row, because the thing you measured
is the thing you are deciding about — and the grid's points sit beside each
field in grey (*measuring: 4, 8, 16*) as **information that fills nothing**.
It carries its own *+ Add setting* for a knob no benchmark touched, since a
run may want one a measurement did not.

**A run needs no benchmark.** Not executed, not summarized, not declared. Open
a fresh calculation, type what you want, prep the run. That is the complaint
this whole redesign answers *(user, 2026-09-01: "it requires the bench to be
fully executed… it tries to conceal all decision from user")*, and it is a
property of the data model rather than of this page.

**Blank is a state, and it is the ordinary one.** What no row states falls to
the stage's `run-config.toml` and then to the wrapper's own policy, and the
card says so.

**And it is checked in the card, as you type** — the same admission door the
grid card asks, over a grid of one. A run is a sweep of length one, so there
is no second check to keep in step (`generator.md` § 2). Six combinations in
the card above and one in this card is the ordinary picture: a plan to
measure, and a decision, side by side.

> **A single card served both for one day** (2026-09-01), on the rule that
> narrowing a `bench` row to one point *was* the decision. It could not
> express the ordinary case: `{mpi_np: [4, 8, 16]}` and *"run at 8"* at the
> same time. Narrowing the row to say what the run uses **destroyed the plan
> to measure** — you could have a bench or a run decision, never both.

**A rung that differs says so on itself.** `execution` is both
calculation-wide and per stage (`stages.md` § 6.8d), and the two compose field
by field. **This card writes the rung's own block** — it is built per rung,
inside that rung's tab (§ 11); the calculation-wide block is shown behind
each field as its placeholder and is a
`task.json` edit today, and § 7.1's per-stage list shows the **effective**
values for each rung either way.

### 6.3 Only speed knobs may be measured

A key must name a field the catalogue puts in the **`execution`** category —
*"knobs that change speed and not the answer"*
([`template.md § 6.2`](?doc=engines/template.md)). Sweeping anything else means
each point silently measures a different calculation, and the comparison is
meaningless. The page refuses rather than warns.

---

## 7. What gets written

**Four files land in the folder, and two of them are the calculation's
description.** The structure the calculation is *of* —
`<label>.source.xyz` + `<label>.source.molstruct.json` — the `.source`
segment is reserved: no engine output can take a dotted name
([`job-contracts.md § 6.3`](?doc=execution/job-contracts.md)) — arrives with
the hand-over
([`handover-procedure.md § 2`](?doc=web/handover-procedure.md)) and this page
never rewrites it. Of the other two, which one a setting lands in follows from
its point count:

| | goes to | why |
|---|---|---|
| a parameter's value | `<label>.template.toml` | the template holds every parameter with the value in force |
| a column, its cells, the shape, the id | `task.json` | what *changes* |
| a **measure**-card setting | `task.json`'s `bench` | the points to try. A one-point NON-machine entry is also a pin in force over the template, for the trials and the run alike *(user rule, 2026-08-20)*; a machine axis is only ever a point to measure |
| a **run**-card setting | `task.json`'s `execution` | **one value each** — what the run uses (`stages.md` § 6.8d). Independent of `bench`: a run needs no benchmark, and declaring a grid states nothing about the run. A machine item becomes the launch shape, anything else a pin over the template — one door each, chosen by the catalogue, never by a second key |
| a notify-card tick | `task.json`'s `notify` | **when this run should say something, and to which channels by name** — on each SCF convergence, every N hours, or neither; a run ending always reports and so is not offered. Portable in the way § 6.1 requires: *"every six hours, to `slack`"* is true wherever the file is opened, because a name is a label the person chose and grants nothing. **What that name resolves to is not written here and must not be** — a description travels, so the address and its credential stay in the config directory of the machine that runs the job, set on the [This machine](?doc=web/this-machine.md) tab |
| a queue, a wall, a memory ask | `task.json`'s `allocation` | **what this calculation asks the scheduler for** (`stages.md` § 6.8a) — three fields, each optional. Not the launch shape: that is a machine-card row, one row above |

```jsonc
// task.json — molbuilder/task@1
{ "schema": "molbuilder/task@1",
  "engine": { "name": "siesta" },
  "shape":  "hierarchical",
  "run":    { "name": "BDT/Au relax", "id": "BDT_Au_relax_Au38C6H4S2" },
  // `source` is FOLDER-RELATIVE — the .xyz beside this file, not a path from
  // anywhere else.  Its sidecar is found by the pairing rule, never named here.
  "structure": { "source": "bdt_au.xyz", "formula": "Au38C6H4S2", "atoms": 50 },
  "varies": ["mesh_cutoff", "relax_force_tol", "relax_type", "restart"],
  "stages": [ { "name": "coarse", "enabled": true, "overrides": { … } }, … ],
  "bench":  { "mpi_np": [4, 8, 16], "omp_threads": [1, 2] } }
```

**There is no `base` key.** What does not change is in the template, once
([`stages.md § 4`](?doc=engines/stages.md)).

### 7.1 And what a prep will write, per stage

*(user, 2026-09-01: "we should have an explicit list of the run tasks for each
stage in that tab so that this information is confirmed and clear".)*

Beside the four files above, this card lists — **one line per enabled stage** —
what `prep` will produce for it: the directory, and the allocation that
directory's wrapper will carry.

```
  01_coarse   run-0/           8 × 6 · htc · 1-00:00:00 · 128G
  02_tight    run-0/          16 × 6 · htc · 2-00:00:00 · 128G
  bench       bench-<token>/   4,8,16 × 1,2 · general · 0-00:30:00
```

**It is a confirmation, not a second answer.** The names and the numbers come
from the same producer `prep` runs — flat and hierarchical name directories
differently (§ 4), and a page that composed them would be free to disagree with
the thing it is describing. What it adds is that you see all of it at once,
before you copy a command, rather than one stage at a time in § 11's tabs.

---

## 8. Saving is two steps, and the first one is the checkpoint

| | |
|---|---|
| **1 · save the folder's current state** | a first state if the folder has none, otherwise a state of what is there right now — so whatever you are about to change can be brought back |
| **2 · write the description** | `task.json` and the template, together |

**The checkpoint is offered and never taken silently**
([`checkpointing.md § 9`](?doc=execution/checkpointing.md)). A note field is
yours to fill; a timestamp is added either way, so every state is identifiable
in the list even when you type nothing.

**Two tabs write one file**, so a stale-file handshake applies rather than
last-write-wins: the folder may have changed between loads because a `prep` ran
or somebody edited by hand ([`tabs.md § 6`](?doc=web/tabs.md)).

---

## 9. The operations, and what each must not lose

Every one of these can silently destroy a value if its rule is not stated.

| Operation | What it does | The rule that keeps it safe |
|---|---|---|
| **add a parameter column** | adds it to `varies` | **seeds every stage with the template's current value**, so adding a column changes nothing on screen. It is a statement about *structure*, never about values |
| **remove a column** | drops it from `varies` | the stages disagree and one value must survive: **the last enabled stage wins**, because that is the production stage and the value a single run would use. The page says which value it kept, and says it *before* the click |
| **add a stage** | appends a row | **copies the previous stage's overrides.** A refinement starts from what came before; a stage that inherits nothing is a different calculation, not a next step |
| **remove a stage** | drops a row | **refused when it is the last one** — a job has at least one stage (§ 5) |
| **reorder** | moves a stage | the files are written in order and `restart` reads that order, so this is a real edit, not a display preference |
| **enable / disable** | marks a stage to run or skip | changes what `prep` will build and what the hand-off says; it does **not** delete the row's values |
| **edit a cell** | sets one stage's value | nothing else moves |
| **apply a stage preset** | fills a row from a tier | a preset knows several fields. If some are not columns yet it **adds them first** — a preset that half-applied would be worse than one that refused |
| **add a measurement point** | turns a value into an axis | the value becomes the first point, so measuring never discards what you chose |
| **open a folder** | replaces the whole page from that folder's description | a **load, not a merge**: values, columns, stages and order all come from the file, because a half-loaded description is one nobody can reason about. The id is read, never recomputed |

---

## 9a. The file itself, in an editor

**`task.json` is shown in a real editor, not a read-only pane**, because the
last thing between a description and a week of compute is a person reading it.
The editor is the **vendored CodeMirror 5.65.16**
(`static/vendor/README.md` — served locally, for offline use and a strict CSP;
there is no CDN path).

**Highlighting is chosen by the file's suffix**, and the mode file is fetched
only when a file of that kind is first opened. The map and the loader are
`static/lib/codemirror-load.js`, and the projects-sidebar preview modal reads
the same one — so *"how is this file highlighted"* has one answer, and adding a
language is one row there rather than a second copy per surface.

| | |
|---|---|
| `.json` | the JSON dialect — CodeMirror ships no separate json mode, so the spec is `{name: "javascript", json: true}` |
| `.toml` | which is what `<label>.template.toml` gets |
| `.py` · `.sh`/`.sbatch` · `.md` · `.xml` · `.css` · `.yaml` | the matching vendored mode |
| `.fdf` · `.xyz` · `.out` · `.log` | **plain text, deliberately** — CodeMirror has no upstream mode for molbuilder's own formats, and `mode: null` is a real mode: line numbers, editing, undo and search all work |

**Editing here is not saving.** The editor is where you check and correct the
description before it is written; § 8 is what writes it. An edited buffer that
has not been saved says so.

### 9a.1 Folded by default, and in the column with everything else

*(user, 2026-09-01: "we should allow the task.json view to be foldable, and
have its width to be consistent with the other cards, such that the 'what gets
written' card can travel as we scroll the window down".)*

The editor was **full width, below both columns**, on the argument that a JSON
document reads badly in a narrow rail. That is true of an *open* editor and it
cost something real: the page stopped being two columns partway down, so the
aside could not follow you. *What gets written* — which since § 7.1 lists what
every stage will produce — sat at the top while you edited the stages it
describes.

So it **folds**, and folded it joins the main column:

- **closed by default.** The description is what the cards above already say;
  the editor is for checking and correcting it, which is a thing you go and do
  rather than a thing you watch. An unsaved edit forces it open — a buffer
  that differs from disk may never be hidden.
- **column width, open or closed** (user: *"have its width to be consistent
  with the other cards"*). The old argument for full width — a JSON document
  reads badly in a narrow rail — was answered by the wrong measurement: the
  main column is not the rail. It is the widest thing on the page after the
  page itself, and a card that matched it would have cost the second column
  for a few characters of line length.
- **the aside travels.** With one grid the whole way down, the rail is sticky
  — and **capped, with its own scroll**: a five-rung ladder's per-stage list is
  tall, and an aside that outgrows the viewport pins its own top out of reach.

**The rail holds the two cards that are about the whole calculation**, in this
order: *What gets written* (§ 7), then *Tell me how it is going* (§ 9b). The
main column is the sequence you work through — folder, shape, machine, queue,
stages, then a tab per rung; the rail is what is true of all of it, and what is
true of all of it should stay on screen while you work through the parts.

---

## 9b. *Tell me how it is going* — and the one file it writes

**It is out of the flow, deliberately** — it lives in the sticky rail under
*What gets written* (§ 9a.1). Everything in the main column is what the
calculation *is*; this is what it *says while it runs*, which is a different
question, and one answer covers every stage. It sat between "Where it runs" and
"Stages" until 2026-08-27 and interrupted the setup (user: *"it breaks the
flow"*); it sat at the bottom of the column until 2026-09-01, where being last
made it easy to miss (user: *"visually would be easy to find and logically
that's a global setup too"*). The rail is both: never in the way, never off
screen.

**This card writes `task.json` and nothing else.** It was one card over two
files until 2026-08-31 — the ticks into the description, the address and key
into `config_dir()/notify` — and the separation was held by a comment in the
template. It is now held by the architecture: the address and the key are on
the **This machine** tab ([`this-machine.md`](?doc=web/this-machine.md)), where
they belong, because they are a fact about the box and not about this
calculation.

**What the card asks, and what travels:**

| | writes | travels? |
|---|---|---|
| the ticks — *when* to speak | `notify.on_scf_converged`, `notify.every_hours` | **yes** |
| the channel list — *which* channels, **by name** | `notify.channels` | **yes**, and safely: a name is a label the person chose, not a credential |

**The names come from the server; the secrets never do.** The list is painted
from `GET /api/notify/channels`, which reports a name, whether it is configured
on this machine, and how the last test went — never an address and never a key.
Nothing on this page has ever seen one, and now nothing on this page *can*.

**Ticking nothing means nothing is sent**, and that is a real state rather than
an oversight ([`run-reports.md`](?doc=execution/run-reports.md) § 3.0): reports
off for this calculation on a machine where they are otherwise set up. A
description with **no** `channels` key — one written by hand, or before
2026-08-31 — means *every channel on this machine*, so nothing that already
worked stops working.

**A ticked name this machine does not have is shown as such**, not hidden. That
is the travelling case — a description written at a desk and opened on a
cluster — and it is the one worth seeing *before* submitting, because a channel
that resolves to nothing is silent by design.

---

## 10. What this page does not do

- **It does not render a deck.** § 1.
- **It does not create folders.** § 2.
- **It does not choose the physics.** The parameter tab does, and its form comes
  from the same catalogue this page reads
  ([`form-schema.md`](?doc=web/form-schema.md)).
- **It does not run anything, or watch anything run.** What has already run is
  read from the folder — how many attempts exist, whether the last converged —
  because you cannot decide what `tight` should be without seeing how `coarse`
  went. That is the page's reason for existing, not a dashboard.
- **It does not name a machine _in what it writes_.** Every value it writes
  means the same thing wherever the folder is copied
  ([`template.md § 7`](?doc=engines/template.md)).
  *(Narrowed 2026-08-22. The page now asks which machine a calculation will
  be prepared FOR and puts `--target` in the command it teaches — because
  `prep` needs the target's measurements and refuses to guess between
  several. That choice shapes a printed command; it never reaches
  `task.json` or the template, so the written description still names no
  machine. See [`preparing-for-another-machine.md
  § 5`](?doc=execution/preparing-for-another-machine.md).)*

---

## 11. Preparing, stage by stage — one tab per rung

*(user, 2026-09-01: "we can make the stage axis as a tab to organize the cards
for prep bench and run so the whole page won't extend too long".)*

The page does not run anything (§ 10). What it can do is show, per stage, what
will be prepared and hand you the exact command — and it does that **in a tab
strip keyed by stage**, because the alternative grows the page by one block per
rung and a five-rung ladder becomes a page nobody scrolls to the bottom of.

**One tab per enabled stage.** Inside a tab, everything you do with that rung:

| | |
|---|---|
| **measure it** | `prep bench <stage>` → `launch bench` → `summarize bench`, and the hint that says which rung is worth measuring |
| **run it** | `prep run <stage>` → `launch run` |

**What each stage will PRODUCE is not in the tab** — it is § 7.1's list, in the
travelling rail (§ 9a.1). A tab is one rung and that list is the whole ladder:
`01_coarse/` beside `02_medium/` is how you check that the shape is what you
meant, and a per-tab copy would show you one line of it at a time. The rail
stays on screen while you work through the tabs, so both are visible at once
without either being duplicated.

**What the run will use is asked HERE, in the rung's own tab**, immediately
above the `prep run` line that consumes it *(user, 2026-09-02: "that selection
panel card should be next to the prep for run button such that it's obvious
that this is designed for the run. This is not mixed up with the existing
functional bench setup")*. It writes that rung's `stages[i].execution`
(`stages.md` § 6.8d); a field left blank takes the calculation's own block,
which the field shows as its placeholder.

There is exactly one such card per rung and none anywhere else — it stood in
the main column for part of 2026-09-02, two cards away from the button it
decides for, which reads as part of the bench setup.

**So a tab is about doing, not deciding**: what this rung will produce, and the
two commands that produce it. The decision is above, made once, and visible in
the description the commands read.

**A stage is either something to measure or something to run, and which is
a decision only you can make** — unchanged from the card this replaces:

| | what it does | when |
|---|---|---|
| **benchmark this stage** | `prep bench` → `launch bench` → `summarize bench` | you do not yet know what allocation this stage wants. The verdict is written to `run-config.toml`, which the run then applies **to the fields you left blank** |
| **prepare the run** | `prep run` → `launch run` | you know what it wants — from a benchmark you read, from § 6.2b's card, or from flags |

**Any stage may be benchmarked, not only the first.** The bench axes are
declared once for the calculation (§ 6.3), so every enabled stage can be
measured. Which one is worth measuring is a judgement — usually the cheapest
rung that still has the expensive stage's shape — and the page states that as
a hint rather than choosing.

**The order is shown because it is load-bearing.** `summarize bench` writes
`run-config.toml`, and `prep run` applies it to any field neither your flags
nor your description stated. Skipping the middle step is no longer silent,
though: the rows are right there, filled or blank, above the command that
consumes them — and the line between them states the order, so a blank row
reads as *this one falls to `run-config.toml`* rather than as nothing.

**The directory names come from the producer.** Flat and hierarchical name
things differently (§ 4), and a page that composed them itself would be a
second answer free to disagree with `prep` — the same rule § 6.2a states for
the grid. What § 7.1's list shows is what `prep` reports it would write.

**Bench and run may ask for different queues and different walls**, and the
file holds one of each because **measuring and running are two preps**
(`generator.md` § 4.1). The calculation's ask is `allocation`; a measurement
that wants a shorter wall states it at its own `prep bench --time`, where it
applies to that prep and no other. A thirty-second measurement queued behind a
two-day reservation is the cost of forgetting that, not a missing field.

**It works the same on a workstation and against a cluster.** Nothing here
reads `execution.mode`: the commands are the commands, the admission check is
the target's, and *which* machine is § 6's question, answered once above.
