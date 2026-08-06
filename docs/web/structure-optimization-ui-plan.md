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
data.** A step is an override of those nine fields and nothing else — which is
also why a steps table can show a row per step without repeating the form.

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

## 7. Where the cluster comes in

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

## 8. Open decisions

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

---

## 9. What this plan is not

It does not change what a stage means on disk, the naming conventions, the
JobSet model, or any producer. It reuses `build_siesta_stage_bundle` exactly as
the CLI does — the migration's stated rule is that the web is *additive on top
of* the shipped framework and reinvents none of it.
