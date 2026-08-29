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

## 6. The machine, and what to try on it

**This is the page's other half, and it is where the GPU is decided.**

One row per setting. **One point is your choice; several is a measurement** —
the same row does both, because a run is a sweep of length one
([`generator.md § 2`](?doc=execution/generator.md): *`prep` step 2 always
produces a list of resolved configurations; a production run is that list with
one element, a benchmark is the same list with N*).

| setting | points | what that means |
|---|---|---|
| `use_gpu` | `off` | **chosen** — one point, so it is a value |
| `mpi_np` | `4` `8` `16` | measured — 3 trials |
| `omp_threads` | `1` `2` | measured — 2 trials |

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

So ranks, threads and memory can only ever be *points to try* here — what a run
actually gets is what the scheduler granted, resolved at `prep` on the target.
*"Try 4, 8, 16"* is true on any machine; *"use 16"* is true on one, and writing
it into a description would make the file a machine's opinion rather than a
calculation's description.

> **This is about CATALOGUE ITEMS, and the tab also holds something else**
> *(2026-08-24)*. The queue card ("Where it runs, and what it may use") sets
> `task.json`'s `allocation` — `domain` / `time` / `mem` — and those are **not**
> points to try. The distinction is the one § 6.8a of
> [`stages.md`](?doc=engines/stages.md) draws: `max_memory_mb` above is a
> per-rank `ulimit` the *deck* carries, a machine-answered template parameter;
> `allocation.mem` is what the *job asks the scheduler for*. A queue name and a
> wall are portable in exactly the way *"try 4, 8, 16"* is — decisions about
> this calculation, true wherever the file is opened — which is why they may be
> written down here while *"use 16 ranks"* may not.

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
| a machine-card setting — **any** point count | `task.json`'s `bench` | **an override lane on the template** *(user rule, 2026-08-20)*: several points = values to **try** (a bench axis); **one point = the value in force**, applied at prep as a pin — for the bench's trials and the run alike. Nothing migrates between files; the description stays exactly as edited, and prep is where a declaration is resolved (`generator.md` § 4.3a) |
| a notify-card tick | `task.json`'s `notify` | **when this run should say something** — on each SCF convergence, every N hours, or neither; a run ending always reports and so is not offered. Portable in the way § 6.1 requires: *"tell me every six hours"* is true wherever the file is opened. **Where to send it is not written here and must not be** — a description travels, so the URL and its credential stay in the user's own config directory on the machine that runs the job |

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

---

## 9b. One card, two files — *Tell me how it is going*

**It is last in the flow, deliberately.** Everything above is what the
calculation *is*; this is what it *says while it runs*, which is a different
question. It sat between "Where it runs" and "Stages" until 2026-08-27 and
interrupted the setup (user: *"it breaks the flow"*).

**One card, and two files, which is the part to keep straight**
([`run-reports.md`](?doc=execution/run-reports.md) §§ 1, 3.1):

| half | writes | travels? | sees a key? |
|---|---|---|---|
| the ticks — *when* to speak | `task.json`'s `notify` block | **yes** | **never** |
| *Where these go* — the address and key | `config_dir()/notify`, `0600` | **no** | yes, that is what it is for |

Sharing a card is a **UI** decision — you tick *tell me* and immediately want
to know *where* — and sharing a **file** would be a contract violation. They do
not: the two halves have different ids, and a test asserts that nothing
belonging to the destination reaches the function that writes `task.json`.

**The page hardcodes no path.** It shows what `GET /api/notify/destination`
reports, and that comes from `monitor.default_notify_path` — so the page, the
API and the process that reads the file on a compute node get it from one
function. The card once said `~/.molbuilder/notify` while the monitor read
`config_dir()/notify`; following it put the file where nothing looks, and
**absent means silently off**.

**One consequence of the merge, stated plainly:** the destination is
machine-wide but now lives inside a card that only appears with a calculation
open. `molbuilder notify-token` is the door that needs no calculation.

**The key is write-only.** The card reports whether one is present and never
what it is, and clears the field once saved: a page that can show you a secret
is one that can leak it, and there is no reason to read it back.

**On a submit machine it does not pretend.** The monitor reads that file on the
cluster, which this server cannot write, so the card hands over the exact
command to run *there* instead of offering a button that would write it
somewhere useless.

**Send a test report** is the only check that exercises the whole path — the
file, the address, the route segment, the signature, egress and TLS. Before it,
the only way to know a setup worked was to run a job and notice nothing
arrived.

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

## 11. What to prepare next — the card that hands you the command

The page does not run anything (§ 10). What it can do is hand you the exact
command, with everything it knows already filled in: the bundle from the
projects root, the target machine you chose, and the `--from` a `continue`
stage needs.

**One block per enabled stage, and both things you can do with it.** A
stage is either something to *measure* or something to *run*, and which
one is a decision only you can make:

| | what it does | when |
|---|---|---|
| **benchmark this stage** | `prep bench <stage>` → `launch bench` → `summarize bench` | you do not yet know what allocation this stage wants. The verdict is written to `run-config.toml`, which the run then applies |
| **prepare the run** | `prep run <stage>` → `launch run` | you know what it wants — either from a benchmark, or because you are telling it with flags |

**Any stage may be benchmarked, not only the first.** The card offered
`prep bench` for the first enabled stage alone until 2026-08-22, which was
a guess dressed as an answer: the bench axes are declared once for the
calculation (§ 6.3), so every enabled stage can be measured. Which one is
worth measuring is a judgement — usually the cheapest rung that still has
the expensive stage's shape — and the page states that as a hint rather
than choosing.

**The order is shown because it is load-bearing.** `summarize bench` writes
`run-config.toml`, and `prep run` *applies* it to any allocation field you
did not state. Skipping the middle step does not fail; it silently prepares
a run with no measured verdict behind it.
