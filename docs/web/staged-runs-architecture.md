# Staged runs — the architecture, from a form to a chain of jobs

**Role:** plan
**Domain:** web
**Companions:** [`structure-optimization-ui-plan.md`](?doc=web/structure-optimization-ui-plan.md)
— the surface this describes the inside of; [`job-system.md`](?doc=execution/job-system.md)
— the JobSet model and the migration's phases; [`job-contracts.md`](?doc=execution/job-contracts.md)
— what lands on disk.

**Status: a proposal.** Nothing here is built. It fixes the shape of the thing
before either end is written, and names the order the work goes in: this
document, then the backend, then the surface.

---

## 1. The one sentence

**A user describes one calculation and how it should tighten; the browser turns
that into a chain of jobs the shipped job system already knows how to run.**

Everything below is about where each part of that sentence lives.

---

## 2. The four objects

| Object | What it is | Who owns it | Lives where |
|---|---|---|---|
| **the plan** | `base` values + which fields `vary` + the stages | the browser | the tab, while you work |
| **the config** | one `SiestaConfig` per stage: `base` overlaid with that stage's overrides | the server | per request |
| **the JobSet** | jobs, dependency edges, carry-forward sets, resources | `stages_to_jobset` | in the bundle |
| **the bundle** | scripts, wrappers, `job-set.json` | `build_siesta_stage_bundle` | the project directory |

The plan is the only new object. The other three ship today, and the whole
design is a bet that the plan can be turned into them without changing what
they mean.

```mermaid
flowchart LR
    P["the plan<br/>base · varies · stages"] -->|"POST once"| C["n configs<br/>base ⊕ overrides"]
    C --> B["build_siesta_stage_bundle"]
    B --> J["a JobSet<br/>edges · carry-forward · resources"]
    B --> F["scripts + wrappers<br/>in the project dir"]
    J --> F
    F -.->|"on the cluster"| R["prep · plan · submit"]
```

---

## 3. What the model can express today — and the two gaps

A stage is a typed object with **eight** settable fields:

```
name · enabled · relax_type · relax_steps · relax_force_tol
     · relax_max_displ · on_nonconvergence · continue_retries
```

Everything else in a stage's `.fdf` comes from the one shared config. That
yields two gaps, and they are the reason this is a backend job before it is a
UI job.

**Gap 1 — most parameters cannot vary at all.** The mesh cutoff, the energy
shift and the DM tolerances live on the shared config, so every stage of a
ladder uses the same ones. The tab's preset menu *appears* to change them per
stage; what it actually does is rewrite the single shared value. A ladder that
coarsens the grid for stage 1 and sharpens it for stage 3 — the ordinary thing
to want — cannot be described today.

**Gap 2 — resources can be carried but not asked for.** The JobSet gives every
job its own resources, and `stages_to_jobset` says they are per-stage,
defaulting to inherit. But no field on a stage says what they should be, so
nothing can populate them differently. The *transport* exists; the *request*
has nowhere to live.

---

## 4. The one change: a stage carries overrides

> **Behaviour stays typed. Values become open.**
>
> `SiestaStageSpec` keeps its eight fields, because each one changes what the
> chain *does*: the policy becomes the scheduler edge, the relaxation method
> decides whether `.CG` carries forward, `enabled` decides whether the stage
> exists. It gains **one** new field:
>
> ```python
> overrides: Dict[str, Any] = field(default_factory=dict)
> ```
>
> — a map from schema field name to that stage's value, applied over the shared
> config when this stage is rendered.

Why not the alternatives:

- **A full config per stage** — thirty-eight values repeated per stage, where
  thirty-five of them are the same. Every shared edit becomes a fan-out, and the
  first missed one is a stage quietly running different physics.
- **Widen the typed spec, field by field** — every new per-stage parameter
  becomes a schema change, a migration and a UI change. The set of things
  someone wants to vary is not knowable in advance; that is the whole point.
- **Let the browser send n complete configs** — the server would lose the notion
  of a ladder entirely, and with it the edges, the carry-forward rules and the
  resources it derives from stage structure.

### 4.1 An override lands in one of two places

This is the part that is easy to get wrong. A promoted field is not always a
line in the script:

| The field's group | Example | Where its override goes |
|---|---|---|
| `profile` / `stage` | mesh cutoff, DM tolerance | the stage's **`.fdf`**, via the same renderer as the shared value |
| `budget` | MPI ranks, threads, wall time | the stage's **job** in the JobSet — its resources, its wrapper |

The schema already knows which is which: a field carries an `engine_key` when it
is a line in the deck, and a `workflow_group` that says whose decision it is. So
the routing is derivable and must not be a second list someone maintains by
hand.

---

## 5. The plan is a file

The plan should not live only in a browser tab. Written down, it becomes the
thing the generator reads:

```jsonc
{
  "format": "molbuilder/stage-plan",

  // What this plan is FOR, and what it needs to run. The backend checks these
  // before it renders anything (§ 5.3).
  "engine":  { "name": "siesta", "version": "5.0", "requires": ["mpi", "netcdf"] },

  // What produced it, and what identifies the run it describes (§ 6).
  "run":     { "name": "bdt au relax",             // what the user called it
               "id":   "bdt_au_relax_a41f9c",       // that, normalised, + the pin
               "created": "2026-08-06T22:14:03-07:00" },  // for tracing, not identity

  // Which schema the values were entered against. Not a definition — a witness,
  // so a disagreement between the browser and the backend is caught, not obeyed.
  "schema":  { "fingerprint": "sha256:1f0c…" },

  // Every schema field, one value. A one-stage plan is just this.
  "base": { "mesh_cutoff": 150, "mpi_ranks": 8, … },

  // WHICH fields the user chose to vary, with the bounds the UI enforced when
  // the values were typed. Intent plus evidence — neither is in a bundle.
  "varies": [
    { "field": "mesh_cutoff",     "type": "float", "min": 50, "max": 1000, "unit": "Ry" },
    { "field": "relax_force_tol", "type": "float", "min": 0.001, "max": 1.0, "unit": "eV/Ang" },
    { "field": "mpi_ranks",       "type": "int",   "min": 1,  "max": 256 }
  ],

  "stages": [
    { "name": "coarse", "enabled": true,
      "relax_type": "CG",      "relax_steps": 600,
      "on_nonconvergence": "proceed",
      "overrides": { "mesh_cutoff": 150, "relax_force_tol": 0.04, "mpi_ranks":  8 } },

    { "name": "tight",  "enabled": true,
      "relax_type": "Broyden", "relax_steps": 200,
      "on_nonconvergence": "halt",
      "overrides": { "mesh_cutoff": 300, "relax_force_tol": 0.01, "mpi_ranks": 16 } }
  ]
}
```

### 5.1 Three rules, and each one is load-bearing

**It names fields; it never defines them.** Every key in `base`, `varies` and
`overrides` must resolve to a field the schema already declares. A plan carrying
a key the schema does not know is **refused, not ignored**: an ignored key is a
calculation quietly different from the one that was asked for. This is the rule
that keeps the JSON from becoming a second schema, which is how the idea fails.

**The bounds it carries are a witness, not a law.** The UI validated those
values against a schema that knows the engine; recording the type and range it
enforced lets the backend check that it is looking at the same field it was —
and *disagreement is an error, never a silent overwrite*. A plan saying
`mesh_cutoff ≤ 1000` against a backend that now says `≤ 600` is a report about
drift between two components, which is exactly the thing that otherwise goes
unnoticed until a run behaves oddly. The **schema fingerprint** makes the common
case one comparison instead of thirty-eight.

**It is parsed *into* the typed config, not around it.** The generator keeps
rendering from a `SiestaConfig` and its stage specs; the file is how a plan
travels and how it persists. A generator that rendered whatever keys the JSON
happened to carry would throw away validation, defaulting and the `engine_key`
mapping, and re-implement all three badly.

### 5.2 There is no format version, on purpose

A version number is a promise that somebody will write migrations, and nobody
does. It also buys nothing here that the content does not already buy: a reader
that **refuses a key it does not know** fails safely on a file from the future,
and names what it could not understand — which is the same outcome a version
check gives, arrived at with more information.

What *does* need checking is checked directly: **the engine and its version**,
the requirements it declares, and the schema fingerprint. Those are real facts
about whether this plan can run here. A number counting revisions of a file
layout is not.

The `format` name stays — one string, so a plan cannot be mistaken for some
other JSON that happens to have a `stages` key.

### 5.3 What the backend checks before it renders

In order, and all of it up front rather than halfway through writing a bundle:

| Check | On failure |
|---|---|
| `format` is a stage-plan | refuse — this is not a plan |
| the engine is one this backend has | refuse, naming what it has |
| the engine **version** satisfies the plan | refuse, naming both — a deck written for 5.0 keywords is not a 4.1 deck |
| declared `requires` are present (MPI, NetCDF, a GPU) | refuse, naming what is missing |
| the schema fingerprint matches | proceed, but report the fields that differ |
| every named field exists | refuse, naming the field |
| every value inside the schema's bounds | refuse, naming the field and both bounds |

The order matters: an engine mismatch makes every field question moot, and
answering the moot ones first buries the real message.

### 5.4 What it buys

- **One producer for both surfaces.** The CLI and the browser stop being two
  paths to a ladder; the browser writes a plan, and the same reader turns it into
  a bundle from either.
- **Reopening a run restores intent.** A bundle can be re-read for its values,
  but nothing in it says *which parameters the user meant to vary* — a mesh
  cutoff that happens to be equal in all three stages is indistinguishable from
  one never promoted. § 10's questions 1 and 4, answered: `varies` is in the file
  because it cannot be inferred from anything else.
- **A plan is reviewable.** It diffs. Two runs that differ can be compared as
  intent rather than by reading two directories of decks.

### 5.5 Where it sits among the artefacts

One direction, no loops:

```
    plan.json          ← intent: what was asked for, including what may vary
       │  (read once)
       ▼
    n effective configs → n scripts + wrappers      ← derived
       │
       ▼
    job-set.json       ← the execution graph: jobs, edges, carry-forward
```

`job-set.json` is not a rival: it holds what the scheduler needs and nothing
about promotion or intent. It is downstream, and it stays derived.

**PROVENANCE stays exactly what it is** — a generator snapshot, not a config —
and gains a use for a key it already reserves. `form-config-hash` becomes the
hash of the plan that produced the script, so any deck in a project can be traced
back to the plan it came from, and a deck edited by hand can be told apart from
one a plan would reproduce.

### 5.6 What this is not

It is not an engine input format: no engine reads it. It is not a replacement
for `SiestaConfig` — that dataclass remains the definition of what a field *is*.
And it is not a new persistence layer for projects; it is one file written beside
the bundle it produced.

---

## 6. Identity: the run id, and the key warm restart turns on

A generated script "declares its ID in one literal (`SystemLabel` / `JOB`)", and
**that ID keys every warm file as `<ID>.<ext>`**. So the value a user types is
what decides whether a run resumes from state already on disk. Two
consequences, both real today:

- two different calculations given the same label in one directory: the second
  **silently warm-starts from the first's geometry**, and the banner says
  `WARM-RESTART (silent)` because that is exactly what it is;
- a label edited between runs: the warm files no longer match, and a run that
  should have resumed starts cold instead.

An id derived from the plan fixes both — but only if it is derived from the
right thing, and this is the trap:

> **A content-derived id and warm restart pull in opposite directions.** Hash the
> plan and every edit changes the id, so tightening a tolerance and resuming
> from the geometry you already reached becomes impossible — the warm files no
> longer bear the name the engine looks for. Yet resuming *through* an edit is
> the whole point of a ladder.

So the id is built the other way round: **the user's words first, the pin
second.**

```
   run.id  =  bdt_au_relax _ a41f9c
              └─────┬─────┘   └─┬──┘
                    │           a fingerprint of what makes prior state
                    │           valid to inherit (§ 6.1)
                    what the user called this experiment — readable,
                    greppable, theirs
```

That one id **is** the `SystemLabel` / `JOB` literal. There is no second name.

**The timestamp is recorded but is not part of it.** It lives in the plan as
`run.created`, for tracing a directory back to the moment it was written — and
it stays out of the id on purpose, because putting it in would make every
regeneration a new identity, and therefore a cold start. Two generations of the
same calculation *should* land on the same id: that is not a collision, it is
the same calculation, and warm files being found is the right outcome.

**What tells two invocations apart, then?** The run wrapper already does it —
each run carries an index (`-run0`, `-run1`, …), and `--force` resets it. That is
invocation-level bookkeeping and it exists; the id is calculation-level and does
not need to repeat it.

**The characters are constrained by the runtime, not by taste.** The wrapper
reads the id back out of the script and sanitises it to `[A-Za-z0-9._-]` before
using it in a shell (§ 4.3), and a stage name must match `[A-Za-z0-9_]+`. So the
user's words are normalised into that set — lowercased, spaces and punctuation
to underscores, trimmed to a sane length — and the fingerprint appended. What a
user types is never used raw, and the id in the file is what the engine sees.

### 6.1 What the identity is tied to — and why that is a physics question

The label keys the warm files, so the rule cannot be a naming convention. It has
to be: **the label binds everything that makes prior state valid to inherit, and
nothing that merely says how hard to push.**

| Tier | Fields | In the label? | Because |
|---|---|:--:|---|
| **the system** | atoms, cell, species → pseudopotentials | **yes** | a `.XV` from different atoms is not a starting geometry, it is nonsense |
| **the electronic description** | basis (PAO size/shift), spin polarisation, XC functional | **yes** | a `.DM` is written in the basis; a different basis or spin is a different *shape*, and a different functional is different physics wearing the same shape |
| **the tightening** | mesh cutoff, DM and energy tolerances, force tol, max displacement, steps, relaxation algorithm | no | these are exactly what a ladder varies. A denser grid re-integrates from the same density matrix; a tighter tolerance resumes from the geometry already reached. Putting these in the identity would make every stage a cold start, which is the opposite of a ladder |
| **the machine** | ranks, threads, wall time, GPU | no | how fast it runs says nothing about whether the answer may be continued |

Change the tolerance, the mesh or the number of ranks and the label holds — the
next stage resumes, which is what a ladder is. Change an atom, the basis, the
spin or the functional and the label changes with it, so the engine looks for
warm files under a name that does not exist and **starts cold because it should**.

The correctness property comes out of that for free: *it is no longer possible
to silently warm-start a calculation from state that belongs to a different
one.* Today that depends on a user typing a different name, and nothing checks.

### 6.2 The id is on screen, and its changes are visible

An identity the user cannot see is one they cannot reason about, and this one
decides whether their run resumes. So the tab shows both, always:

```
   ┌────────────────────────────────────────────────────────────┐
   │  Job ID   bdt_au_relax_a41f9c                              │
   │           the pin changes if you edit the structure, the    │
   │           basis, the spin or the functional — and then a    │
   │           run starts cold, because it is a different one    │
   └────────────────────────────────────────────────────────────┘
```

Two behaviours that make it worth showing rather than merely correct:

- **When an identity-bearing field is edited, the pin visibly changes**, at the
  moment of the edit. That is the UI saying *this has become a different
  calculation and it will start cold* — before a bundle is written, not after a
  run behaves oddly.
- **When a run directory already holds warm files**, the tab can say whether
  they match the current id: *"prior state found for this key — the next run
  resumes"*, or *"prior state found, but from a different calculation"*. That is
  the same sentence the wrapper's banner prints, moved to where the decision is
  actually being made.


**And the hash still earns its place — as a report, not as a name.** The plan's
content hash is recorded with the run. When warm files are found whose hash
differs from the plan about to run, the banner can say *whose* state it is
resuming rather than only that it is resuming. That is the existing doctrine
applied to a case it does not yet cover: **molbuilder informs, and the user
decides to continue.** Today `WARM-RESTART (silent)` cannot tell you that the
state on disk came from a different calculation, because nothing recorded which
calculation made it.

What this replaces: a free-typed name doing three jobs at once — telling runs
apart, keying warm files, and naming files on disk. Splitting it into a stable
label and a unique id lets each be right.

---

## 7. Validation has to name the stage

A findings list today points at a field. With a ladder, "mesh cutoff is too low
for this basis" is true of *stage 1* and false of *stage 3*, and a finding that
cannot say which is a finding a user cannot act on.

So each stage's **effective** config — `base ⊕ overrides` — is validated as a
config in its own right, and every finding carries the stage it came from in its
`where`. One shared value producing the same complaint in three stages should
say so once, not three times.

This is the existing findings contract extended by a coordinate, not a second
validation path.

---

## 8. The module map

| Layer | Today | What this needs |
|---|---|---|
| config | `SiestaConfig`, `SiestaStageSpec` (8 fields) | **+ `overrides` map**, and the merge that applies it |
| ladder | `render_siesta_stage_fdfs`, `stages_to_jobset` | renders from the *effective* config per stage; routes `budget` overrides into job resources |
| bundle | `build_siesta_stage_bundle` | unchanged — it is already the seam § 8 named |
| plan file | — | **new**: the `stage-plan` format, its reader, and the refusal of unknown keys |
| web API | `/api/build/*` renders one script | **+ one route** that takes a plan and writes a bundle |
| validation | one config → findings | per-stage effective configs → findings **with a stage coordinate** |
| browser | schema-driven form, one flat config | **+ the plan model**: pure data, pure operations (promote, demote, add, remove, reorder, apply preset) |
| browser | — | **+ the matrix view**, which renders the plan and calls those operations |

The last two are deliberately separate. The plan model is arithmetic on a data
structure: no DOM, no fetch, testable as values in and values out — the same
split that made SpectrumChart's maths layer the easy part to trust. The view
renders it and nothing else.

---

## 9. The order of work, and what "done" means

**Step 1 — this document.** Done when the shape is agreed and the open questions
in § 10 are answered.

**Step 2 — the backend, tuned and validated.** In order:

1. The **plan format and its reader** (§ 5), including the refusal of a key the
   schema does not know. *Done when:* a plan round-trips — read, rendered,
   re-read — and a plan naming a dead field fails with that field's name.
2. `overrides` on the stage spec, and the merge that produces an effective
   config. *Done when:* a stage with `{mesh_cutoff: 300}` renders a `.fdf`
   carrying 300 while the shared config still says 150, and a stage with no
   overrides renders exactly what it renders today.
3. Override routing for `budget` fields into job resources. *Done when:* a
   two-stage plan asking for 8 ranks then 16 produces a JobSet whose two jobs
   differ in resources, with the `.fdf`s unchanged.
4. Per-stage validation with the stage coordinate. *Done when:* a plan whose
   stage 1 is under-converged reports against stage 1 alone.
5. The bundle route. *Done when:* a plan posted to it writes the same bytes the
   CLI writes for the same ladder — compared file by file, because "the web is
   additive on top of the CLI" is only true if the output is identical.

**Step 3 — the surface.** The plan model first (pure, tested), then the matrix
view, then the subtabs. The UI plan's § 7 is the specification for the first of
those.

The gate between 2 and 3 is worth stating: **the backend must be able to express
a ladder that varies a non-stage parameter before any of it is drawn**, or the
UI will be designed around what the model happens to allow rather than what a
user needs.

---

## 10. Open questions this document cannot settle

1. **Does a stage's override of a `budget` field also change the wrapper it gets
   installed with**, or only the scheduler request?
2. **Is the user's half of the id editable after the fact?** Renaming it changes
   the id and so orphans the warm files, which is right in principle and will
   surprise someone who meant only to fix a typo. Deriving it makes it consistent;
   letting it be typed keeps a door open to deliberately resuming from an
   unrelated run's state, which is occasionally what a person wants and is
   otherwise impossible to ask for.
3. **How long is the pin?** Four hex characters read easily and collide once in
   65,000; six is safer and uglier. It only has to be unique within a directory.
4. **Does the XC functional belong in the identity (§ 6.1)?** A `.DM` from
   another functional is the right *shape* and the wrong *physics* — usable as a
   guess, wrong as a continuation. Included here on the grounds that continuing a
   relaxation across a functional change silently is a scientific error, but this
   is a call for someone who does it.
5. **What exactly is hashed for the system tier** — coordinates to full
   precision, or a rounded form? Full precision means a geometry nudged by
   10⁻⁶ Å is a different calculation, which is right in principle and may be
   maddening in practice.
6. **Is a plan editable by hand?** It is JSON beside a bundle, so it will be. If
   yes, the reader owes the same errors to a person as to the browser — which is
   an argument for the refusal rule in § 5.1 being loud rather than tolerant.
7. **How strict is the engine-version check?** § 5.3 refuses on mismatch, which
   is right for a major change and heavy-handed for a patch release. A range, or
   a "warn below, refuse across major", needs deciding by someone who knows what
   SIESTA changes between versions.

*Answered by § 5, which is why it was worth writing:* whether `varies` travels to
the server (yes — it cannot be inferred), what happens when the schema moves on
(refuse, and name the field), and whether the plan persists beside the bundle
(yes — it is the only record of intent).
