# The Structure-optimization tab — one page for one script and for many

**Role:** plan
**Domain:** web
**Companions:** [`job-system.md`](?doc=execution/job-system.md) — what a JobSet is
and what the web is meant to produce (§ 8, Phase 1); [`job-contracts.md`](?doc=execution/job-contracts.md)
— what a stage *is* on disk (§ 2.3); [`form-schema.md`](?doc=web/form-schema.md)
— how this tab's fields get on screen at all.

**Status: a proposal.** Nothing here is built. It offers three arrangements of
one page and recommends one; the decisions still open are listed at the end.

---

## 1. The two complaints, stated exactly

**"The stage drop menu is not obvious what it does."** It is a *form autofill*.
Choosing "Stage 2 — Medium" rewrites nine convergence fields and does nothing
else — no second file, no chain, no job. The template says as much in a comment:
*a UI shortcut, NOT a SiestaConfig field*.

That would be forgivable if the word meant one thing. It means three:

| Where | What "stage" means there |
|---|---|
| this dropdown | a preset that bulk-fills nine fields |
| `cfg.stages` → `stages_to_jobset` | a real step: its own `.fdf`, its own job, chained to the one before |
| `fdf --stage N` | emit one step's file alone, under a *different* naming convention |

So the control reads as *make me stage 2 of a ladder* and delivers *one file with
different numbers in it*. The job-system contract names this same gap from the
other side: **"the web Build tab still renders a single `.fdf` (it drops the
stage table)"**.

**"The UI is very jammed."** One card carries the run directory, the restart
policy, the stage preset, ~38 schema fields in seven sections, the engine
switch, four action buttons, an issues list and a script preview — all at once,
all always.

---

## 2. What the page is made of today

```
   ┌ 1. Inspect structure ─────────────────────────────┐
   │  the 3-D viewer, atom counts, formula             │
   └───────────────────────────────────────────────────┘
   ┌ 2. Analyze chemistry ─────────────────────────────┐
   │  auto-detect, warnings, metals                    │
   └───────────────────────────────────────────────────┘
   ┌ 3. Generate input ────────────────────────────────┐
   │  [SIESTA] [PySCF]          ← engine tabs          │
   │  Run directory & restart                          │
   │  Relaxation stage preset   ← the confusing one    │
   │  ~38 fields, 7 sections, one long grid            │
   │  [Generate] [Download] [Save to current dir]      │
   │  issues list                                      │
   │  script preview                                   │
   └───────────────────────────────────────────────────┘
```

Everything in card 3 is visible at once, and the only thing that ever *acts* —
the button row — sits below the longest thing on the page.

---

## 3. The two ideas the design rests on

### 3.1 A single script is a job set of one

The job system "sits **above** the single-job wrapper. It never replaces the
wrapper — it produces many of them." If that is true of the machinery, the UI
should not grow a second mode for it. **One list of steps, whose one-row case is
exactly what the tab does today**, and the page says which you are getting.

No toggle, no "batch mode", nothing to explain about which system you are in.

### 3.2 Subtab what you *set*; never subtab what you *do*

Parameters can hide behind tabs because you visit them once and leave. The plan
and the actions cannot: they are the reason the page exists, and a button you
have to go looking for is a button that gets missed.

So the page splits in two, and only the first half is tabbed.

### 3.3 The schema already knows which fields vary per stage

Every SIESTA field carries a `workflow_group`, and there are exactly three:

| group | fields | what it describes | how often it changes |
|---|--:|---|---|
| `profile` | 20 | the system and its physics | once per molecule |
| `budget` | 9 | cores, memory, wall time, SCF ceiling | once per machine |
| `stage` | 9 | convergence targets — mesh, tolerances, force, displacement | **once per step** |

That is not a UI invention: it is the grouping the config already carries, and
it is what the stage preset already writes to. **The ladder's shape is in the
data**, and a step is small enough to be a row rather than a form.

But those nine are a **default proposal, not the definition of a stage** — which
parameters vary is the user's to choose, and that is the hard part of the design.
§ 7 is about nothing else.

---

## 4. Three arrangements

All three keep card 1 and card 2 as they are. All three replace card 3.

### Option A — Subtabs above, plan below

```
   ┌ 3. Generate input ─────────────────────────────────────────┐
   │ [ System ] [ Convergence ] [ Resources ] [ Output ]        │
   │ ┌────────────────────────────────────────────────────────┐ │
   │ │  only that group's fields                              │ │
   │ └────────────────────────────────────────────────────────┘ │
   ├────────────────────────────────────────────────────────────┤
   │ STEPS                                                      │
   │  1  Coarse   mesh 150 Ry · force 0.04 · displ 0.20   [—]   │
   │  2  Tight    mesh 300 Ry · force 0.01 · displ 0.05   [—]   │
   │  [+ add a step]                                            │
   ├────────────────────────────────────────────────────────────┤
   │ You will get: a 2-step bundle      [ Check ] [ Generate ]  │
   │ issues · preview · where it was written                    │
   └────────────────────────────────────────────────────────────┘
```

*For it:* smallest change from today; one column, so it survives a narrow
panel; the reading order is still top-to-bottom.
*Against it:* the plan scrolls away while you edit parameters — the thing you
are building is out of sight exactly when you are changing it.

### Option B — Settings left, plan right

```
   ┌ 3. Generate input ─────────────────────────────────────────┐
   │ [System][Convergence][Resources][Output] │ THE PLAN        │
   │                                          │                 │
   │   only that group's fields               │  1 Coarse  ···  │
   │                                          │  2 Tight   ···  │
   │                                          │  [+ step]       │
   │                                          │                 │
   │                                          │  → 2-step bundle│
   │                                          │  [Check][Make]  │
   │                                          │  issues         │
   └──────────────────────────────────────────┴─────────────────┘
```

*For it:* the plan never leaves the screen; editing a convergence field and
watching the step row change is one glance. *Against it:* needs width, and this
tab already competes with the projects sidebar; below about 900 px it has to
fall back to Option A's stacking anyway.

### Option C — The ladder is the spine

```
   ┌ 3. Generate input ─────────────────────────────────────────┐
   │ SHARED   [ System ] [ Resources ] [ Output ]               │
   │          ┌──────────────────────────────────────────────┐  │
   │          │  fields shared by every step                 │  │
   │          └──────────────────────────────────────────────┘  │
   │ STEPS    [ 1 Coarse ] [ 2 Tight ] [ + ]   ← tabs too      │
   │          ┌──────────────────────────────────────────────┐  │
   │          │  the nine convergence fields for THIS step   │  │
   │          └──────────────────────────────────────────────┘  │
   │ You will get: a 2-step bundle      [ Check ] [ Generate ]  │
   └────────────────────────────────────────────────────────────┘
```

*For it:* the clearest statement of what a stage *is* — shared settings above,
per-step settings below, and the difference is visible in the layout itself.
*Against it:* two tab strips on one card is a lot of chrome, and the one-step
case — today's whole workflow — pays for machinery it does not use.

---

## 5. The recommendation

**Option A, with B's split adopted as a container query when the box is wide
enough.** They are the same content; B is A after a `@container` flip, so this
is one layout with a width rule rather than two designs.

And **Option C's insight without its second tab strip**: the steps table shows
each step's nine values *inline and editable* in the row. A step is small enough
to be a row — nine numbers — so it does not need a panel of its own. The
"Convergence" subtab then edits the *selected* row, and the table is both the
list and the summary.

Why not C proper: the one-step case is the common case, and C makes it look like
a ladder with one rung rather than a script with some settings.

---

## 6. What each part says

**The subtab bar** carries the sections the schema already declares, folded to
the `workflow_group` axis where they disagree — four tabs, not seven:
*System* (system, spin, XC, basis) · *Convergence* (the nine stage fields, for
the selected step) · *Resources* (compute & budget) · *Output* (output &
positioning, run directory, restart).

**The steps table** is the ladder. One row by default, and that row is today's
behaviour. A row names itself (Coarse / Medium / Tight / a name you type), shows
its nine values, and can be removed. The presets fill a row — which is what the
dropdown always did, now attached to the thing it fills.

**The outcome line** is the sentence that fixes the original complaint. It reads
`You will get: one .fdf` or `You will get: a 3-step bundle`, and it changes as
rows come and go, so the connection between the ladder and the artefact is never
in doubt.

**The action row** is Check and Generate, in that order, always visible.

---

## 7. The hard part: which parameters vary, and who decides

The layout above is the easy half. The real difficulty is that **the set of
per-stage parameters cannot be fixed in advance**. One run wants the grid
density to sharpen across stages; another wants the CPU or GPU budget to grow
with the tightening; another wants only a convergence threshold to move. A
design that ships one blessed list is wrong for every second user.

### 7.1 What is already per-stage, and what is not

Two facts to build on rather than around:

- **A stage is already a typed object**, not a bag of numbers. `SiestaStageSpec`
  carries its relaxation method, its step cap, its non-convergence policy and
  its own resources — and `stages_to_jobset` says plainly that *"resources are
  per-stage, defaulting to inherit the config's ranks/threads."* **The CPU/GPU
  budget already varies per stage in the model; only the UI hides it.**
- **Some of those fields are not values at all.** The non-convergence policy
  *becomes the scheduler edge* (`proceed → afterany`, `halt → afterok`), and the
  relaxation method *decides whether `.CG` carries forward* to the next stage — a
  CG state is meaningless to a Broyden stage. Change one and the shape of the
  chain changes with it.

That gives the split the whole design hangs on:

> **Behaviour stays typed; values are open.** The handful of per-stage settings
> that *change what the chain does* keep their named fields. Everything else that
> may differ between stages is an **override**, and which fields those are is the
> user's to choose.

### 7.2 The model

```js
plan = {
    base:   { …every field in the schema, one value each… },  // the shared config
    varies: ["mesh_cutoff", "force_tol", "mpi_ranks"],        // the promoted set
    stages: [
        { name: "coarse", enabled: true,  relaxation: "CG",      steps: 600,
          onNonConvergence: "proceed",
          overrides: { mesh_cutoff: 150, force_tol: 0.04, mpi_ranks:  8 } },
        { name: "tight",  enabled: true,  relaxation: "Broyden", steps: 200,
          onNonConvergence: "halt",
          overrides: { mesh_cutoff: 300, force_tol: 0.01, mpi_ranks: 16 } },
    ],
}
```

Three rules keep it honest:

1. **`varies` is the column set.** Every stage's `overrides` holds exactly those
   keys — no more, so a demoted parameter cannot leave a value hiding in a stage
   nobody can see.
2. **`base` holds a value for every field, always**, including the promoted ones.
   A one-stage plan is then just `base`, which is what makes § 3.1 true: one
   script is a job set of one.
3. **The default `varies` is a proposal, not a law.** It starts as the nine
   fields the schema already tags `workflow_group: "stage"` — the ones the
   preset writes to — because that is the useful default, not because it is
   special.

### 7.3 The operations, and what each one must not lose

This is where the care goes: every one of these can silently destroy a value if
its rule is not stated.

| Operation | What it does | The rule that keeps it safe |
|---|---|---|
| **promote** a field | adds it to `varies` | **seeds every stage with the current base value**, so promoting changes nothing on screen. Promotion is a statement about *structure*, never about values |
| **demote** a field | removes it from `varies` | the stages disagree and one value must survive: **the last enabled stage wins**, because that is the production stage and the value a single run would use. The UI says which value it kept, and says it *before* the click, not after |
| **add a stage** | appends a step | **copies the previous stage's overrides**. A refinement starts from what came before; a stage that inherits nothing is a different calculation, not a next step |
| **remove a stage** | drops a column | refused when it is the last one — a plan has at least one step |
| **reorder** | moves a step | the chain is ordered and the carry-forward rules read that order, so this is a real edit, not a display preference |
| **edit a cell** | sets one stage's value | nothing else moves. A cell equal to `base` is drawn quietly; one that differs is drawn plainly, so *progressive change is visible as a shape* |
| **apply a preset** | fills a column | a preset knows nine fields. If some are not promoted it **promotes them first** — a preset that half-applied would be worse than one that refused |

### 7.4 The panel

One table, rows are parameters and columns are stages — the shape the data
already has:

```
   PARAMETERS THAT VARY                stage 1     stage 2     stage 3
   ────────────────────────────────────────────────────────────────────
   mesh cutoff            Ry            150         300         300
   force tolerance        eV/Å          0.04        0.02        0.01
   MPI ranks              —             8           16          16
   ────────────────────────────────────────────────────────────────────
   relaxation                           CG          Broyden     Broyden
   on non-convergence                   proceed     halt        halt
   ────────────────────────────────────────────────────────────────────
   [ + add a parameter ▾ ]        [ preset ▾ ]  [ + stage ]  [ remove ]

   shared with every stage: 29 others   → System · Resources · Output
```

Two things that table has to get right:

- **The typed rows sit below a rule**, separated from the open ones, because they
  are the ones that change the chain rather than a number in a file.
- **Promotion happens where the parameter lives.** Every field in the other
  subtabs carries a "vary per stage" affordance; using it moves that field into
  this table. The alternative — a long "add a parameter" menu of all 38 — is a
  second place to find a field, and the wrong one to reach for.

### 7.5 What gets written

A column is a config: `base` overlaid with that stage's `overrides`, plus its
typed fields. So **n columns produce n scripts**, and the ladder that already
ships lays them out — `build_siesta_stage_bundle` for the bundle,
`stages_to_jobset` for the chain, `jobset prep` for the per-stage folders and
their wrappers.

Nothing here invents a directory layout. Two shipped shapes exist and the
difference matters: the in-place ladder keeps every stage in **one** directory
with an unsuffixed `SystemLabel` so `.XV` / `.DM` / `.CG` transfer naturally,
while `jobset prep` lays out **per-stage folders** and carries those files
forward explicitly — `.XV` always, `.DM` when the config saves it, `.CG` only
between stages using the same relaxation method. Asking for subdirectories is
asking for the second, which is the JobSet path this whole plan is aimed at.

## 8. Where the cluster comes in

The browser can *produce* a bundle. It cannot `prep`, `plan` or `submit` one —
that is Phase 2 of the migration, and the later phases are gated on proving the
SIESTA ladder end-to-end on a real cluster first.

So a multi-step generate ends with a handoff, and the honest thing is to show it
as the normal path rather than as an apology:

```
   Written to  projects/BDT/optimization/bdt-relax/
   On the cluster:   molbuilder jobset prep .
                     molbuilder jobset plan .
                     molbuilder jobset submit .
```

A one-step generate ends as it does today: a file, a download, a save.

---

## 9. Open decisions

1. **Does one step still emit a bare `.fdf`?** Recommended yes — the change
   stays additive and nobody's current workflow moves. The alternative (every
   generate is a bundle, even of one) is more uniform and more disruptive.
2. **Do the four subtabs remember which was open?** The tab already persists
   form values in `sessionStorage`; the open subtab is the same kind of fact.
3. **Does PySCF get the same treatment now or later?** Its ladder runs inside
   one process and writes one log (`job-contracts.md § 2.3`), so "a step" means
   something different there. Recommended: SIESTA only, until the ladder is
   proven on a cluster.
4. **Where do "Inspect structure" and "Analyze chemistry" go?** Untouched by
   this plan, but if the page is still jammed with card 3 tamed, they are the
   next candidates for folding.
5. **Does demote really keep the last stage's value?** (§ 7.3) The alternative is
   asking every time, which is safer and more tiring. A third option is to keep
   the *first* stage's, on the grounds that it is the one a coarse single run
   would want.
6. **May a promoted parameter be left blank for a stage** — meaning "inherit
   base" — or must every cell carry a value? Blank cells make a sparse ladder
   readable (only two of five stages change the mesh) but add a second way to
   say "same as base".
7. **Is the promoted set saved with the project?** It is a description of intent,
   not a value, and losing it on reload would be worse than losing a field.

---

## 10. What this plan is not

It does not change what a stage means on disk, the naming conventions, the
JobSet model, or any producer. It reuses `build_siesta_stage_bundle` exactly as
the CLI does — the migration's stated rule is that the web is *additive on top
of* the shipped framework and reinvents none of it.
