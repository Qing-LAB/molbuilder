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

**Scope: one deck per stage is SIESTA's shape.** `job-contracts.md § 2.3` names
two multi-stage execution shapes, and only one of them is this. PySCF's staged
relaxation runs **inside one Python process** writing a single unified log, so a
three-stage PySCF calculation is one file, not three. This contract describes the
per-deck shape; extending it to PySCF means first deciding whether its stages
become a loop inside one script (what ships today) or genuinely separate files,
and that decision is not made here.

**A stage resolves completely at generate time.** What comes out is an ordinary,
complete engine input that does not need molbuilder to be interpreted and does
not refer to a stage it follows.

Precisely: the stage name survives in the **filename**, `<id>_<name>.fdf`, as a
label — and nothing has to interpret it to run the file. The deck's *content*
carries no stage marker at all. Anything that would require a downstream reader
to understand the word "stage" in order to act correctly is outside this
contract.

---

## 2. The object

```jsonc
{ "name": "coarse", "enabled": true,
  "overrides": { "mesh_cutoff": 150, "relax_force_tol": 0.04 } }
```

**Three fields, and no others.**

| Field | Type | Meaning |
|---|---|---|
| `name` | `[A-Za-z0-9_]+` — letters, digits, underscore, **no hyphen** | becomes the deck's suffix, `<id>_<name>` (`job-contracts.md § 2.3`). The hyphen is excluded because it is already the *other* stage separator: the molwatch log and the run decoder's stage regex use `-stage<N>` (`job-contracts.md § 2.3`), and a stage named `pre-tight` would put a second hyphen where that reader looks for one |
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

Leaving the stage is not enough, though: if it stayed a field of the **shared
schema** it would be promotable through `overrides` like anything else, and § 2's
"any field of the shared schema" would quietly readmit it. So it is not a field
of the shared schema at all. It belongs to the JobSet producer's own input, which
is a different object with a different reader
([`execution/job-system.md`](?doc=execution/job-system.md) — `job-system.md § 4.1`).

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
(`science/validation.md § 4.1`).

**R3 — the sequence is checked as well as its members.** R2 makes every stage
individually sound and says nothing about the order they are in, yet the order is
the whole point of having several. A ladder that *loosens* — stage 2 coarser than
stage 1 — passes R2 twice and is almost certainly a mistake, because the second
stage throws away what the first paid for. So a description is also read across
its stages, and a finding about the sequence carries **no** stage label: it is a
fact about the description, not about a member of it (the same rule that already
governs a shared-config complaint, § 6.2). What the checks *are* — which
parameters must not go backwards, and by how much — is `engines/tuning.md`'s to
say, not this contract's.

**An `error` in any stage blocks the whole produce**, not just its own deck.
That is not a policy choice made here — it falls out of § 7.2: the folder appears
whole or not at all, so there is no such thing as writing the stages that passed.

---

## 5. Where a promoted field lands — three destinations

A promoted field is not always a line in the deck, and assuming it is writes
decks that are subtly wrong for the machine they run on.

| Kind | Examples | Lands |
|---|---|---|
| an ordinary deck line | `mesh_cutoff` → `MeshCutoff` | the stage's deck, and nowhere else |
| **a deck line that is also a resource decision** | `diag_algorithm` → `Diag.Algorithm`; `enable_gpu` | the deck **and** the wrapper's env routing **and** a scheduler's `--gres` |
| a field the deck never carries | `mpi_np`, `omp_threads`, `continue_retries` | the **wrapper** — baked at install (`continue_retries`) or resolved at run time (ranks, threads) — and a scheduler's `-n` / `-c` if one is asked |

**The routing is derivable, never a second list.** A field carries an
`engine_key` when it is a line in the deck; the config ↔ exchange translation for
the third row is already fixed by `job-contracts.md § 6.2` and applied by the
producer at its boundary. Nobody maintains a mapping table by hand.

**Walltime, memory and partition are deliberately absent from that table.** They
are not fields of the shared schema: `running-a-job.md § 5.3` puts `time` and
`mem` under `molbuilder.json`'s `scheduler.defaults`, and a routing `domain`
resolves to a partition and QoS the same way. That is **the machine's knowledge**,
and `job-system.md`'s decision 3 keeps it on the machine — a description that
carried a walltime would stop being portable, and would be wrong the moment it
was opened on a different cluster. A per-stage walltime is a real thing to want;
it is asked for at export (`job-system.md § 5.1`'s `--stage-resources`), where
the target is known.

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
- **It is a correctness gate, and it fires late.** If a stage opts into a build
  whose environment is not installed, generation raises with an install hint
  (`running-a-job.md § 2.3`) — but that check belongs to *wrapper* generation,
  which happens after the decks are rendered, and § 6.6 deliberately does not
  duplicate it in the preflight. So the refusal arrives with some decks already
  written, which is why § 7 requires the whole folder to be produced
  transactionally (§ 7.2).

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
  // what was there when it was written (§ 6.3).
  "structure": { "source": "projects/BDT-Au/structure/bdt_au.xyz",
                 "formula": "C6H4S2Au38", "atoms": 46 },

  // Every schema field, one value. A one-stage description stops here (§ 6.5).
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

### 6.3 `structure` is a reference plus a witness, never a copy

Coordinates are what runs *produce*; a description that embedded them would be a
second copy of a file the tree already holds, drifting from it the moment either
moved. So `source` points into the tree, and `formula` and `atoms` travel beside
it as evidence of what was there when the description was written — which is what
the id was built from (`execution/run-identity.md § 2`). A description opened
against a structure that has since changed can therefore *say so*, rather than
silently building a different calculation under the same id
(`run-identity.md § 5`).

### 6.4 What writing it down buys

Three things, and the first is why the file exists at all rather than the
description living only in a browser tab.

- **One producer for both surfaces.** The CLI and the browser stop being two paths
  to a staged calculation: each writes a description, and the same reader turns it
  into decks from either. That is what makes "the web is additive on top of the
  CLI" checkable — the two must produce the same bytes for the same description,
  and a single reader is how.
- **A deck can be traced back to what asked for it.** PROVENANCE
  (`job-contracts.md § 3.2`) already reserves an optional `form-config-hash` key
  and this is its use: the hash of the description that produced the deck. Any
  deck in a project then names its origin, and a deck someone edited by hand can
  be told apart from one the description would reproduce. PROVENANCE stays exactly
  what it is — a generation snapshot, not a config.
- **Descriptions diff.** Two calculations that differ can be compared as *intent*
  — one file against one file — rather than by reading two directories of decks
  and inferring what was deliberate.

### 6.5 One stage is no stages

**`stages` may be absent, and absent means one.** A description with no `stages`
key is a calculation with a single parameter set — `base`, exactly — and it
produces one deck named `<id>.fdf`, with no stage suffix. Nothing about stages
has to be understood to read or write it.

Three things follow, and they are one fact seen three times: the deck takes no
suffix, findings carry no stage label (§ 4 R2), and `varies` is **absent** —
there is nothing to vary across, and an empty list would be a second way to spell
the same state.

A description *with* `stages` has at least one; removing the last is refused.

### 6.6 The preflight

In order, and all of it before anything is written:

| Check | On failure |
|---|---|
| the schema string is `molbuilder/stages@<known major>` | refuse — not a description, or not one this reader knows |
| the engine is one this backend has a generator for | refuse, naming what it has |
| the schema fingerprint matches | proceed, and say plainly it was written against a different schema |
| every named field exists in the shared schema | refuse, naming the field |
| no `overrides` key names a stage field (§ 2) | refuse, naming the field |
| every stage `name` matches `[A-Za-z0-9_]+` | refuse, naming the stage and the rule |
| **stage names are unique**, compared case-insensitively | refuse, naming the repeat |
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

**Why two of those rows are about names.** A stage name becomes a filename
(§ 2), so a name outside the set or repeated between stages produces two decks
that collide — the second silently overwriting the first, in a folder whose whole
premise is that every file in it is accounted for. Refusing costs a message;
allowing it costs a calculation nobody knows is missing. (Two stages that are
*identical in value* but differently named is a separate question, and a warning
rather than a refusal — the plan records it.)

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
- **every value the description determined**, written rather than left to an
  engine default. A field the user set must appear in the deck; a field the
  description never touched may rely on the engine's own default, which is what
  engine defaults are for. The failure this rules out is *omit-and-hope* — leaving
  out a value the calculation depends on and discovering later which default
  filled it.
- **the engine's identity group set as one**, never key by key —
  [`execution/run-identity.md`](?doc=execution/run-identity.md) § 4.
- **BENCH-MARKS declaring every line derived from a launch quantity** (§ 5.2).
- **a run wrapper per deck**, built by the shipped builder
  (`job-contracts.md § 2.6`). A folder of decks with no wrappers is not something
  a user can run.
- **a distinct trajectory-log basename per deck.** `job-contracts.md § 2.3` merges
  a directory's `.molwatch.log` files in mtime order into one trajectory with a
  boundary per stage — which is exactly the reading a folder of stages wants — but
  it only works if each deck writes its *own* log. Two decks resolving to one
  basename would interleave into a single file and the boundary would be lost.
  The shipped basename is `<label>-stage<N>`, a hyphen and a **number**, while a
  deck is `<label>_<name>`, an underscore and a **name**; reconciling those two is
  a decision `job-contracts.md § 2.3` owns and has not yet made.

**The test:** the decks are portable — an engine with no molbuilder present runs
them correctly. The wrappers are not, and are not meant to be: they are baked for
a target (§ 8).

### 7.1 The layout: common above, per-stage below

**Stages do not share a directory.** The folder is a parent holding what every
stage shares, and one subdirectory per stage holding everything that stage
produces:

```
projects/BDT-Au/optimization/bdt_au_relax_c6h4s2au38/
├── stages.json                        ← the description
├── Au.psml  S.psml  C.psml  H.psml    ← shared, stored ONCE
├── mb_monitor.py                      ← shared
├── <id>_coarse.fdf  <id>_coarse.run.sh
├── <id>_tight.fdf   <id>_tight.run.sh ← decks + wrappers, rendered once
├── coarse/
│   ├── <id>_coarse.fdf → ../<id>_coarse.fdf
│   ├── Au.psml → ../Au.psml  …        ← shared, linked in
│   └── <id>.XV .DM .ANI .STRUCT_OUT   ← everything THIS stage produced
└── tight/
    ├── <id>_tight.fdf → ../<id>_tight.fdf
    ├── <id>.XV → ../coarse/<id>.XV    ← carried; dangling until coarse runs
    └── <id>.DM → ../coarse/<id>.DM
```

**This is not a new layout.** It is what `job-system.md § 5.2`'s `prep` already
builds, and this contract reuses that materializer rather than writing a second
one. Two of its properties are the reason:

- **Shared files are stored once and linked in**, so a five-stage description
  does not carry five copies of a pseudopotential, and the shared set is
  obviously shared rather than coincidentally identical.
- **Carried files are localized on run.** Before a stage starts, its wrapper
  replaces an inherited symlink with a real local copy — *"without that, stage 2
  writing to `bdt.XV` would write through the link and overwrite stage 1's
  result."* That is the whole cross-contamination problem, closed by a rule that
  already ships.

**Why not one flat directory.** A flat folder was the earlier answer here, on the
grounds that a shared basename makes continuing free (`job-contracts.md § 2.1`
Rule 2). It does — and it also means every stage writes over the last one. The
restart files are the obvious casualty, but they are not the worst: `.ANI`,
`.STRUCT_OUT`, `.EIG` and every other engine output is keyed by `SystemLabel`,
which is *identical* across stages by design. Run three stages flat and you keep
one set of results — the last — plus three `.out` logs. For a framework whose
purpose is managing a mission across several parameter sets, losing every
intermediate result is not a trade, it is a defect.

**What carry is, and is not.** Carry is a *layout* mechanism, not a scheduling
one: it exists because the stages are in different directories, and it would be
needed if no scheduler existed anywhere. The scheduling half of `job-system.md` —
`depends_on`, `dep_kind`, the edges — stays outside this contract (§ 3). Which
files carry is not a new decision either: `.XV` always, `.DM` when the config
saves it, `.CG` only between stages using the same relaxation method, *"a CG
state is meaningless to a Broyden stage"* (`job-system.md § 4.1`).

**And `restart` gets sharper.** In one directory, *continue* could only mean
"whatever ran here last" — order-of-execution dependent, and wrong if you re-ran
an earlier stage. With a subdirectory each, **`continue` means: carry from the
previous enabled stage**, which is a fact about the description rather than about
what happened to run. `clean` carries nothing.

### 7.2 The folder appears whole, or not at all

Rendering a description can fail after it has started: a stage asks for an
environment that is not installed (§ 5.1), a pseudopotential does not resolve, a
disk fills. **A half-written folder is worse than none**, because every rule in
this contract about what a folder contains stops being true of it, and the run
directory it half-occupies may already hold warm files from a previous
calculation.

So a produce is **transactional**: every deck, every wrapper and the description
are built somewhere else and moved into place only when all of them succeeded. On
failure nothing is moved, and the message names the stage that stopped it.

This is the same discipline the sidecar and archive writers already use — build,
verify, then `os.replace` (`job-contracts.md § 5.4`) — applied to a directory
rather than a file. What it must **not** do is remove warm files that were already
there; producing twice is `execution/run-identity.md § 6`, and those files are
the point.

**And a produce that replaces an earlier one may make the folder match the
description exactly — because it checkpoints first.** Remove a stage, or disable
one, and the deck it produced last time is still there, with a wrapper that still
runs it, describing a calculation the description no longer contains. Left alone
that breaks the premise every rule here rests on: that a folder's contents are
what its description says they are.

The answer is not to tiptoe around the orphans. It is
[`molbuilder snapshot`](?doc=execution/running-a-job.md) § 6, which already puts
a run directory under a git-backed history — text tracked (including the small
`.XV` / `.CG`, *"so a restore brings back a resumable state"*), large binaries
archived by content and deduped:

> **A replacing produce checkpoints the folder before it writes anything.**
> Having done so, it removes what the description no longer contains, and the
> folder is exactly the description again. Nothing is lost, because the prior
> state is a commit — restore it, or branch from it.

A produce that only rewrites decks changes only text, so that half is cheap.
**The binary half is not, today.** The archive is keyed by commit sha and copies
every big binary on every checkpoint — the *"deduped by content"* in the shipped
guide describes deduping basenames within one MANIFEST, not storage across
checkpoints (`execution/checkpointing.md`, I1 and L5). Automatic checkpoints fire
twice per stage, so a five-stage mission would pay ten full copies of its `.DM`
set unless the store is content-addressed first. **L5 is therefore a prerequisite
for § 7.3's automatic checkpoints, not a later optimisation** — and it is a small
change to a system whose documentation already claims it.

The warm files are never removed by any of this: they belong to the calculation,
not to any one stage (`execution/run-identity.md § 6`).

### 7.3 A description grows, and a stage that has run is a record

A description is not written once and produced once. The ordinary way a mission
goes is **incremental**: run a stage, look at what came out, decide the next one
from what you saw, run that. So the stage list **grows over time**, and a produce
usually lands in a folder where earlier stages have already run.

**There is no fixed number of stages.** `job-system.md § 4.1`'s three-rung ladder
is a *default set* with three presets flipping its enable flags — a starting
proposal, not a bound. A fourth and fifth stage decided next month are ordinary,
and each continues from what is in the folder because `restart` means *whatever
ran here last* (`execution/run-identity.md § 4`). That is the payoff of the
identity being blind to everything a stage tunes: appending a stage does not
change the id, so the state is still there to continue from.

Two rules make growth safe, and both follow from one observation.

> **A stage has run when the folder holds output keyed to its deck** —
> `<id>_<name>-run0.out` and its trajectory log (`job-contracts.md § 2.6`). That
> is a fact on disk, not a flag anybody maintains.

**R4 — a stage that has run is a record, and the record is a checkpoint.** The
outputs beside a deck were made by that deck as it was, so replacing it without
keeping the old one leaves a folder whose results came from a file that no longer
exists. That is not a reason to refuse the edit — redoing a stage is ordinary
work. It is a reason for the history to exist, which § 7.2 already requires.

So the boundaries where a checkpoint is taken are exactly two, and both are
molbuilder's rather than the engine's:

| When | What it holds | Why there |
|---|---|---|
| **before a replacing produce** | the folder as the last produce left it | it is what makes rewriting a run stage safe rather than lossy (§ 7.2) |
| **when a stage's run finishes** | that stage's converged state, tagged with its name | it is the point a user will want to come back to and **branch from** — the next stage is a choice, and a choice wants somewhere to return to |

**Both are automatic whenever the folder is under checkpoint.** Not offered, not
a button: a folder that has a history keeps it, and a stage boundary that went
unrecorded is one the user cannot return to precisely when they discover they
want to. Once a folder is under checkpoint, molbuilder does not ask permission
to write the history it exists for.

**Who initialises, exactly.** A produce that *creates* the folder initialises it
(`snapshot init --engine <engine>`) — molbuilder made the directory, so offering
it a history costs the user nothing and asks them nothing. A produce into a
folder that **already existed without a checkpoint** does not: that folder is
someone's deliberate state, and putting it under version control is their call,
not a side effect of generating a deck into it. The two rules together: *created
here means initialised; already under checkpoint means kept; neither means left
alone.*

**Neither is the wrapper's job.** `running-a-job.md § 6.2` records that the
wrapper-bootstraps-git path was deliberately dropped — *"the wrapper is
deliberately git-agnostic, so init is CLI/UI-only"* — so the second boundary is
observed where a run is already being watched: the decoded run reports `finished`
(`running-a-job.md § 4.2`), and the surface or the CLI takes the checkpoint. A
wrapper that committed to git would be a wrapper that needs git on the compute
node, which is exactly what the standalone contract forbids.

#### What each checkpoint is called

A history is only worth taking if you can find the point you want in it, so both
boundaries name themselves. `molbuilder snapshot` already gives three surfaces —
a commit message, an annotated tag with a name, and a branch name
(`running-a-job.md § 6.2`) — and each carries a different part of the identity:

| | Form | Example |
|---|---|---|
| **commit message** (both boundaries) | `<id> · <stage or event> · <what happened>` | `bdt_au_relax_c6h4s2au38 · tight · relaxation converged, 41 steps` |
| **tag** (a stage that finished) | `<id>/<stage>/<UTC timestamp>` | `bdt_au_relax_c6h4s2au38/tight/20260806T221403Z` |
| **branch** (a user forking a what-if) | proposed as `<stage>-<what you are trying>`, and editable | `tight-tighter-mesh` |

Four things make that work rather than merely look tidy:

- **No new normalisation.** The id is `[A-Za-z0-9_-]+` and a stage name is
  `[A-Za-z0-9_]+` (§ 2, `execution/run-identity.md § 3`) — both already
  ref-safe, so the same set that was chosen to survive a filename survives a git
  ref. The timestamp is compact UTC (`YYYYMMDDThhmmssZ`) for the same reason: the
  ISO form's colons are not legal in a ref, and this matches the convention
  `job-contracts.md § 4.1` already uses for `<basename>-restart-aside-<UTC>/`.
- **The tag is hierarchical on purpose.** `git tag --list '<id>/tight/*'` is every
  checkpoint of one stage, oldest to newest, which is the question a user
  returning to a mission actually asks.
- **The message says how it went, because something already knows.** The decoded
  run reports `finished` / `failed` and its step counts
  (`running-a-job.md § 4.2`); the checkpoint is taken by whatever observed that,
  so it can say *converged* or *hit the step cap* rather than *stage 2 done*.
- **Only stage completions are tagged.** A pre-produce checkpoint is a safety net,
  reachable through `snapshot list`; a finished stage is a place you meant to
  reach. Tagging both would bury the second in the first. And a tag that would
  collide — two checkpoints of one stage inside the same second — is refused, not
  suffixed, like every other name in this design.

**The identity in every message is not decoration.** A folder can be moved,
copied to a cluster, or opened a year later; a history whose commits say only
*"stage 2 converged"* cannot tell you which calculation that was, and the id is
the one thing that can (`execution/run-identity.md § 1`).

**What this buys is the thing a growing description most needs.** A tagged
checkpoint per completed stage turns the folder from *the current state of one
calculation* into a chain of states you can re-enter: `snapshot branch` at stage
2 and try a different stage 3 without losing the first attempt. That is
"switching between setups" in its strongest form, and it is why `branch` having
no HTTP route (`running-a-job.md § 6.2`) is the most consequential gap in this
whole design rather than a loose end.

**R5 — a stage's name is its identity, and renaming is rewriting.** The name is
in the deck's filename, in every output beside it, in the checkpoint tag, and in
the detector above. Renaming a stage that has run moves all four at once, so it
is an R4 event and takes an R4 checkpoint.
This is also why the stage's position in the list must never appear in a
filename: insert a stage at the front, or reorder two, and every positional
number after it shifts — silently reassigning outputs that already exist to
stages that did not produce them. **Names are stable; positions are not.** Where
the shipped trajectory log uses `-stage<N>` (§ 7, the log bullet), that is the
half of the naming question growth decides: it has to key on the name.

### 7.4 What the layout costs the checkpoint system

The history in § 7.3 is `molbuilder snapshot`, whose invariants —
what must never change, and what must stay separate from what — are
[`execution/checkpointing.md`](?doc=execution/checkpointing.md). It was designed
against a **flat** run directory. Three things follow from pointing it at a tree, and each
is a change to the checkpoint side rather than a compromise on this one.

**The repository sits at the parent, not in each subdirectory.**
`job-system.md § 5.5` observes that each `point-<name>/` is a self-contained run
directory and so *"can be checkpointed on its own"*. True, and not enough: a
per-subdirectory repository cannot restore a shared file that lives **above** it,
so restoring a stage would leave its linked-in pseudopotentials pointing at
nothing. It also cannot express the operation this design turns on — *branch the
workflow at stage 2* — because there is no repository that contains the workflow.
One repository at the parent holds the shared files once, every stage's outputs,
and the description, so a branch carries the whole tree as it was.

**The archive globs have to become recursive.** `.mbcheckpoint.json`'s engine
defaults are `*.DM`, `*.HSX`, `*.TSHS`, … (`running-a-job.md § 6.1`) — patterns
written for files beside the config. In this layout the big binaries are at
`<stage>/<id>.DM`, which `*.DM` does not match, so **every one of them would be
committed to git as a blob** — the exact outcome the archive exists to prevent.
The glob set must match at depth, and the seeded `.gitignore` with it. This is a
change to the checkpoint contract, and it lands with its code and its test in one
commit (`job-contracts.md § 7`).

**A carried link is a dangling link until its producer runs, and that is
correct.** Git stores a symlink as its target path, so a checkpoint taken between
produce and run holds the dangling carry links, and a restore recreates them
dangling — which is exactly the state that produce left. When the stage runs,
localize-on-run replaces the link with a real file (§ 7.1) and the next
checkpoint records a symlink becoming a regular file. Noisy in a diff, right in
substance: the history says *this stage stopped inheriting and started owning*,
which is what happened.

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
- **Carrying a finished run into the next calculation** — the handoff bundle,
  [`execution/job-contracts.md`](?doc=execution/job-contracts.md) § 5. It reads a
  run directory and fuses the final coordinates with the labels from the script
  that produced them, and a folder of stages is a run directory, so it works
  unchanged.

  > **One interaction to settle before this ships.** `job-contracts.md § 5.3`
  > resolves *which* script to read when a directory holds several: **largest by
  > atom count, ties broken lexicographically.** Every stage of one description has
  > the same atoms, so every produce is a tie — and lexicographic order picks
  > `_coarse` over `_tight`. The coordinates are right either way (they come from
  > the one shared `.XV`), but `source_script` and the provenance it carries would
  > name the stage that ran *first*. A folder of stages makes that the normal case
  > rather than an edge one, so the tie-break needs an answer that knows about
  > stages — most likely the last enabled one, which is the production stage.
- **What a checkpoint history must always hold** —
  [`execution/checkpointing.md`](?doc=execution/checkpointing.md). This contract
  says *when* a checkpoint is taken and *what it is called*; that one says what
  must be true of it afterwards, in a form a test can assert.
- **Phasing, status, and what is built when** —
  [`web/staged-runs-architecture.md`](?doc=web/staged-runs-architecture.md) and
  [`roadmap.md`](?doc=roadmap.md) (R3).
