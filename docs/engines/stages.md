# Stages — a named parameter set over one system, and the file that describes it

**Role:** contract
**Domain:** engines
**Companions:** [`engines/tuning.md`](?doc=engines/tuning.md) — what *values* a
stage should carry and why (this doc says what a stage *is*, never what to put in
one); [`engines/siesta.md`](?doc=engines/siesta.md) +
[`engines/pyscf.md`](?doc=engines/pyscf.md) — the emitters that render an
effective config; [`execution/run-identity.md`](?doc=execution/run-identity.md) —
the id every stage in a folder shares, and the engine parameters that decide
whether a stage continues; [`execution/job-contracts.md`](?doc=execution/job-contracts.md)
— the run directory the decks land in and the persisted-artifact registry;
[`web/staged-runs-architecture.md`](?doc=web/staged-runs-architecture.md) — the
plan that motivates this contract and schedules the work.

**Status: proposed, not built.** This document is written first and the code is
built to it, the way `web/spectrumchart.md` and `web/vibrationview.md` were.
Nothing in `SiestaConfig` matches it yet; the differences and the order of work
are in the plan, not here (R3).

**This contract owns:** what a stage is, which fields are a stage's and which are
the shared schema's, how an effective config is formed, where a promoted field
lands, and the shape of `stages.json`.

---

## 1. A stage is ours, not the engine's

**No engine has a concept of a stage.** SIESTA reads a `.fdf`; PySCF runs a
`.py`. Neither knows the file it was handed is the second of three, or that
anything preceded it. The word exists only inside molbuilder:

> **A stage is a named set of the parameters a mission tunes, laid over the
> shared description of the system it does not.**

The base is *what the system is*. A stage is *how we are approaching it this
time*.

**A stage resolves completely at generate time.** What comes out is an ordinary,
complete engine input that does not need molbuilder to be interpreted, does not
refer to a stage it follows, and carries no marker a reader must first
understand. Anything that would require a downstream reader to know the word
"stage" is outside this contract.

---

## 2. The object

```jsonc
{ "name": "coarse", "enabled": true,
  "overrides": { "mesh_cutoff": 150, "relax_force_tol": 0.04 } }
```

**Three fields, and no others.**

| Field | Type | Meaning |
|---|---|---|
| `name` | `[A-Za-z0-9_]+` | becomes the deck's suffix, `<id>_<name>` (`job-contracts.md § 2.3`) |
| `enabled` | bool | whether this stage is rendered at all |
| `overrides` | map | schema field name → that stage's value |

`overrides` may name **any field of the shared schema** and **never** `name` or
`enabled`. A description carrying a stage-field name inside `overrides` is
refused: two homes for one fact is how the previous model produced fields that
lived in both places and silently disagreed.

---

## 3. Which fields are a stage's, and which are the schema's

Two questions, asked in order. It matters that they are two: either alone
mis-sorts a field.

> **1. Does it survive without a scheduler?**
> A setting that means nothing until something else queues the work does not
> describe a calculation. `execution/job-system.md` owns it.
>
> **2. Of what is left: can a single run mean it?**
> If yes, it is an ordinary field of the shared schema, which a stage may
> override like any other — *wherever that field happens to land* (§ 5).
> **A stage types only what a single run cannot mean.**

Question 2 deliberately does **not** ask where the field ends up. A promoted
field may become a deck line, a wrapper setting, or a scheduler request; sorting
fields by destination is what produces stage types that grow without limit.

Worked against the fields the shipped `SiestaStageSpec` carries:

| Field | Survives without a scheduler? | Can a single run mean it? | Lands |
|---|:--:|:--:|---|
| `name` | yes | no — a single run is named by its id | **the stage** |
| `enabled` | yes | no — there is nothing to enable | **the stage** |
| `relax_type` | yes | yes | the shared schema |
| `relax_steps` | yes | yes | the shared schema |
| `relax_force_tol` | yes | yes | the shared schema |
| `relax_max_displ` | yes | yes | the shared schema |
| `continue_retries` | yes — `running-a-job.md § 3.5` | yes | the shared schema, routed to the **wrapper** (§ 5) |
| `on_nonconvergence` | **no** — it *is* the scheduler edge | — | **outside this contract** — `job-system.md § 4.1` |

Two of those are worth stating explicitly, because both were on the stage type
and neither belonged there.

**`on_nonconvergence` fails question 1.** Its entire effect is the dependency
edge a JobSet threads (`proceed → afterany`, `halt → afterok`). Without a
scheduler there is nothing for it to mean.

**`continue_retries` passes both questions and is still not a stage field.**
`running-a-job.md § 3.5` is explicit: a *single* SIESTA run whose wrapper was
installed with a retry budget re-runs itself with `--continue`. It is an ordinary
shared field; what made it look special is only where it lands. That is also why
`job-system.md § 4.1` records that the SIESTA ladder never implemented it — the
field sat on the stage, and the stage is not what honours it.

**One field arrives.** Whether a stage continues from what is already in the
folder or starts clean has to be sayable, and by question 2 it is a shared field:
a single run can mean it too. `restart` (`continue` | `clean`) joins the shared
schema; what the generator does with it is
[`execution/run-identity.md`](?doc=execution/run-identity.md) § 4.

---

## 4. The effective config

> **effective config = `base` ⊕ that stage's `overrides`.**

It is an ordinary instance of the engine's config dataclass — a `SiestaConfig`,
not a new type — so every default, every bound and every `engine_key` mapping
applies to it unchanged.

Two rules govern it, and both exist to stop a stage becoming a special case:

**R1 — one object is validated and rendered.** The config handed to validation is
the same object handed to the emitter. What was checked and what was written
cannot come apart.

**R2 — a stage is validated as a resolved whole, never as a diff.** Two overrides
can each be individually reasonable and jointly wrong: a mesh cutoff that is
fine, a basis that is fine, and a pair that is under-converged together. The
validator is handed a whole config plus the stage's name as a label — never an
overlay. The label travels beside `where`, never inside it
(`science/validation.md § 4.1`); `error` blocks, for every stage.

---

## 5. Where a promoted field lands — three destinations

A promoted field is not always a line in the deck, and assuming it is writes
decks that are subtly wrong for the machine they run on.

| Kind | Examples | Lands |
|---|---|---|
| an ordinary deck line | `mesh_cutoff` → `MeshCutoff` | the stage's deck, and nowhere else |
| **a deck line that is also a resource decision** | `diag_algorithm` → `Diag.Algorithm`; `enable_gpu` | the deck **and** the wrapper's env routing **and** a scheduler's `--gres` |
| a field the deck never carries | `mpi_np`, `omp_threads`, `time`, `mem`, `continue_retries` | the **wrapper** — baked at install (`continue_retries`) or resolved at run time (ranks, threads) — and a scheduler's `-n` / `-c` / `-t` / `--mem` if one is asked |

**The routing is derivable, never a second list.** A field carries an
`engine_key` when it is a line in the deck; the config ↔ exchange translation for
the third row is already fixed by `job-contracts.md § 6.2` and applied by the
producer at its boundary. Nobody maintains a mapping table by hand.

### 5.1 The middle row, and what it costs

`job-contracts.md § 6.2` lists the eigensolver as a config value that becomes a
`.fdf` keyword and the GPU request as one *derived from* the `.fdf`.
`running-a-job.md § 2.3` says what follows: **any** `Diag.Algorithm elpa*` — even
CPU-ELPA — routes the wrapper to the GPU-build environment, because ELPA is
linked only in that build.

Two consequences:

- **Two stages in one folder may need two different environments.** A coarse
  stage on ScaLAPACK and a tight stage on ELPA-GPU is an ordinary thing to want,
  and it works: routing is per script, so each deck's own wrapper activates its
  own environment. Nothing about the folder has to change.
- **It is a correctness gate, not a preference.** If a stage opts into a build
  whose environment is not installed, generation raises with an install hint
  (`running-a-job.md § 2.3`) — and because a description is produced as a whole
  (§ 6.4), one such stage refuses the whole generate rather than writing a folder
  that is partly runnable.

### 5.2 A deck line may depend on the launch

**A deck's own values can be derived from resources the deck does not contain.**
SIESTA's `BlockSize` is the standing example: PROVENANCE records it as
`auto -> 256 (10 * 212 atoms / mpi_np, capped pow2)` (`job-contracts.md § 3.2`).
A deck rendered for 8 ranks is not the right deck for 16.

And the rank count is genuinely not settled at generate time.
`running-a-job.md § 2.1` fixes the rule — at run time the wrapper reads the
allocation and the hardware *"only to tune the launch … never to decide whether
the job can run"* — and `running-a-job.md § 3.1` gives the precedence, so the
ranks a job runs with
are routinely not the ranks its deck was rendered against.

> **A deck states which of its lines were derived from a launch quantity.** The
> generator renders for the resources the description asked for, and the
> BENCH-MARKS block (`job-contracts.md § 3.3`) declares the coupled fields —
> anchor-based, with bounds — so anything that later changes the launch can
> re-derive them instead of silently leaving them stale.

That block already exists and already declares `BlockSize`, because the benchmark
sweep varies ranks per point and has the same problem. This contract adopts it
rather than inventing a second mechanism.

---

## 6. `stages.json` — the description on disk

```jsonc
{
  "schema": "molbuilder/stages@1",

  "engine": { "name": "siesta" },

  // What identifies this calculation, and what the user called it.
  // The rules are execution/run-identity.md.
  "run": { "name": "BDT/Au relax",                    // typed, kept verbatim
           "id":   "bdt_au_relax_c6h4s2au38",         // normalised once, then quoted
           "created": "2026-08-06T22:14:03-07:00" },  // for tracing, not identity

  // Which schema the values were entered against — a witness, not a definition.
  "schema_fingerprint": "sha256:1f0c…",

  // What this is a calculation OF: a reference into the tree, plus a witness of
  // what was there when it was written (§ 6.2).
  "structure": { "source": "projects/BDT-Au/structure/bdt_au.xyz",
                 "formula": "C6H4S2Au38", "atoms": 46 },

  // Every schema field, one value. A one-stage description stops here (§ 6.3).
  "base": { "mesh_cutoff": 150, "relax_type": "CG", "restart": "clean", … },

  // WHICH fields the user chose to tune. Intent — it cannot be inferred.
  "varies": ["mesh_cutoff", "relax_force_tol", "relax_type", "restart"],

  "stages": [
    { "name": "coarse", "enabled": true,
      "overrides": { "mesh_cutoff": 150, "relax_force_tol": 0.04,
                     "relax_type": "CG",      "restart": "clean" } },

    { "name": "tight",  "enabled": true,
      "overrides": { "mesh_cutoff": 300, "relax_force_tol": 0.01,
                     "relax_type": "Broyden", "restart": "continue" } }
  ]
}
```

### 6.1 Three rules

**It names fields; it never defines them.** Every key in `base` and in every
`overrides` map must resolve to a field the shared schema already declares. A key
the schema does not know is **refused, not ignored** — an ignored key is a
calculation quietly different from the one that was asked for. This is what keeps
the file from becoming a second schema.

**It is parsed *into* the typed config, not around it.** The reader produces a
config object and stage specs; the emitters are unchanged. A reader that rendered
whatever keys the JSON happened to carry would throw away validation, defaulting
and the `engine_key` mapping, and re-implement all three badly.

**It carries the shipped schema convention.** `job-contracts.md § 6.1` fixes it:
`molbuilder/<name>@<major>`, checked **major-only** through the one shared helper
`molbuilder/persist.py`, and *"New persisted artifacts must use it."* That check
is not a promise that somebody writes migrations — it is *"refuses with a clear
message rather than mis-parsing"*, which is the behaviour this file wants. The
artifact registry gains its row when the reader lands.

### 6.2 `varies`, and why it cannot be inferred

`varies` is the set of fields the user chose to tune. It is intent, and no
artefact downstream records it: a mesh cutoff that happens to be equal in every
stage is indistinguishable from one that was never promoted. Every stage's
`overrides` holds exactly the keys in `varies` — no more, so a demoted parameter
cannot leave a value hiding in a stage nobody can see.

`structure` is a **reference plus a witness**, never a copy. Coordinates are what
runs produce; a description that embedded them would drift from the file the tree
already holds. The formula and atom count are what the id was built from, so a
description opened against a structure that has since changed can say so.

### 6.3 One stage is no stages

**`stages` may be absent, and absent means one.** A description with no `stages`
key is a calculation with a single parameter set — `base`, exactly — and it
produces one deck named `<id>.fdf`, with no stage suffix. Nothing about stages
has to be understood to read or write it.

Three things follow, and they are one fact seen three times: the deck takes no
suffix, findings carry no stage label (§ 4 R2), and `varies` is empty or absent
because there is nothing to vary across.

A description *with* `stages` has at least one; removing the last is refused.

### 6.4 The preflight

In order, and all of it before anything is written:

| Check | On failure |
|---|---|
| the schema string is `molbuilder/stages@<known major>` | refuse — not a description, or not one this reader knows |
| the engine is one this backend has a generator for | refuse, naming what it has |
| the schema fingerprint matches | proceed, and say plainly it was written against a different schema |
| every named field exists in the shared schema | refuse, naming the field |
| no `overrides` key names a stage field (§ 2) | refuse, naming the field |
| every value is inside the schema's bounds | refuse, naming the field and both bounds |

**Two things are deliberately not checked here.**

- **The engine's version.** Nothing in the shipped system records or gates one.
  The version is known where the binary is — `running-a-job.md § 4.1`'s run banner
  prints it — and the machine writing a description may not have the engine at
  all. Gating here would break `job-system.md`'s decision 3, *the machine's
  knowledge lives on the machine*.
- **Declared requirements** (MPI, a GPU build, a library). Already answered twice,
  at well-defined moments: env routing derives the requirement from the deck
  (§ 5.1), and the doctor verifies prerequisites on the target
  (`running-a-job.md § 2.2`). A third, hand-maintained list would only drift from
  what the deck actually asks for.

**The fingerprint's claim is deliberately weak.** One string can say *this was
written against a different schema*; it cannot say which fields moved. The
per-field rows do that work.

---

## 7. What the generator must produce

A folder whose decks are correct on their own. Concretely, per rendered stage:

- **the cell, explicit** — the description holds cell *parameters*; the generator
  computes the vectors and shifts the atoms into the frame the deck must carry
  (`model/cell-plan.md`).
- **pseudopotentials resolved** per species, through the path that already
  refuses on `xc_family_mismatch`, and written into the folder. (`job-contracts.md
  § 2.7` says the layout does not *require* co-location; putting them there is
  what makes the folder self-contained.)
- **every field the schema declares**, defaults resolved rather than
  omitted-and-hoped-for.
- **the engine's identity group set as one**, never key by key —
  [`execution/run-identity.md`](?doc=execution/run-identity.md) § 4.
- **BENCH-MARKS declaring every line derived from a launch quantity** (§ 5.2).
- **a run wrapper per deck**, built by the shipped builder
  (`job-contracts.md § 2.6`). A folder of decks with no wrappers is not something
  a user can run.

**The test:** the decks are portable — an engine with no molbuilder present runs
them correctly. The wrappers are not, and are not meant to be: they are baked for
a target (§ 8).

---

## 8. What this contract does not own

- **The environment, activation, and how a wrapper finds its engine** —
  [`execution/running-a-job.md`](?doc=execution/running-a-job.md) §§ 2 and 5.
  Nothing here changes any of it. To restate only what a reader of this document
  needs: molbuilder must be installed on the machine that *generates*; the
  activation form (`conda activate` / `source activate`) and any module preamble
  come from `molbuilder.json`, have **no default**, and generation of an HPC
  wrapper refuses without them; environment *names* are configurable per category
  and must never be hard-coded; `.sbatch` is emitted only when a `scheduler`
  block is configured. Everything site-specific is baked at generate/prep, and at
  run time the wrapper reads only the allocation and the hardware.
- **The run id, its normalisation, and the engine's identity group** —
  [`execution/run-identity.md`](?doc=execution/run-identity.md).
- **The run directory, filenames, reserved script blocks, warm-restart files, the
  project tree** — [`execution/job-contracts.md`](?doc=execution/job-contracts.md).
- **What values a stage should carry** —
  [`engines/tuning.md`](?doc=engines/tuning.md).
- **The dependency chain, `Job.carry`, `Job.resources`, and every scheduler
  concern** — [`execution/job-system.md`](?doc=execution/job-system.md). A
  JobSet export reads this file and asks for the one thing it does not carry
  (`on_nonconvergence`, § 3).
- **Phasing, status, and what is built when** —
  [`web/staged-runs-architecture.md`](?doc=web/staged-runs-architecture.md) and
  [`roadmap.md`](?doc=roadmap.md) (R3).
