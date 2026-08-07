# Run identity — the name that decides whether a calculation continues

**Role:** contract
**Domain:** execution
**Companions:** [`execution/job-contracts.md`](?doc=execution/job-contracts.md) —
the run directory, the basename rule (§ 2.1), the warm-file inventory and the
restart banner (§ 4), all of which this document builds on and none of which it
changes; [`execution/running-a-job.md`](?doc=execution/running-a-job.md) — how a
run is actually launched and what the wrapper does with the files;
[`engines/stages.md`](?doc=engines/stages.md) — the description this id is
derived from; [`web/staged-runs-architecture.md`](?doc=web/staged-runs-architecture.md)
— the plan that motivates this contract and schedules the work.

**Status: proposed, not built.** Written first, built to. `job-contracts.md § 4`
describes the shipped behaviour; this document says how the id that drives it
should be *constructed*, and asks each engine to declare something it does not
declare today.

**This contract owns:** how a run id is built, what it may and may not be tied
to, how it is normalised, and the per-engine group of parameters that decides
whether prior state is honoured.

---

## 1. Why identity is the mechanism

A generated script *"declares its ID in one literal (`SystemLabel` / `JOB`)"*,
and **that ID keys every warm file as `<ID>.<ext>`** (`job-contracts.md § 4.1`).
`job-contracts.md § 2.1` Rule 2 then says the part that matters most here:

> Across the stages of one staged relaxation the basename **stays identical**
> (only parameters change); that is exactly what lets SIESTA pick up
> `<basename>.XV` / `.DM` from the previous stage.

So **continuing is not something molbuilder does.** It is what the engine does
when it finds warm files keyed by the id it was given and its own bound
parameters say to load them (§ 4). molbuilder's whole contribution is to make the
id right and put the files where the engine will look.

Two failures follow directly from getting it wrong, and both are real today:

- **two different calculations given the same label in one directory** — the
  second silently warm-starts from the first's geometry, and the banner says
  `WARM-RESTART (silent)` because that is exactly what it is
  (`job-contracts.md § 4.4`);
- **a label edited between runs** — the warm files no longer match, and a run
  that should have resumed starts cold instead.

---

## 2. The rule

> **The id is built from inputs, never from anything a run produced.**

It must be knowable *before* the calculation exists: it names the folder and the
basename of every file in it. An id that depended on a result would change the
moment a stage succeeded, orphaning the state it exists to continue from — the
identity would break precisely when the calculation worked.

So: no coordinates, no energies, no convergence status, nothing read back off a
`.XV`. That leaves simple parameters, which is all it needs.

The id is **readable, not cryptographic** — made of things a person already
knows:

```mermaid
flowchart LR
    U["what the user calls it<br/><b>BDT/Au relax</b>"] --> N["normalise once<br/>§ 3"]
    M["what the coordinates are of<br/>the molecule, by formula<br/>or by named components<br/><b>C6H4S2Au38</b>"] --> N
    N --> I["<b>bdt_au_relax_c6h4s2au38</b><br/>= the SystemLabel literal<br/>= the basename of every file<br/>= the folder"]
```

A hash would be exact and unreadable. A formula is neither, and that is the trade
made deliberately: an id you can recognise in a directory listing and in a
filename is worth more day to day than one that resolves every possible
ambiguity.

**That one id is the `SystemLabel` / `JOB` literal. There is no second name.**

**The timestamp is recorded but is not part of it.** A description carries when
it was written, for tracing; putting it in the id would make every regeneration a
new identity and therefore a cold start. Two generations of the same calculation
*should* land on the same id — that is not a collision, it is the same
calculation, and warm files being found is the right outcome.

**What tells two invocations apart, then?** The wrapper already does it: each run
carries an index (`-run0`, `-run1`, …) and never clobbers a prior one
(`job-contracts.md § 2.6`). That is invocation-level bookkeeping; the id is
calculation-level and does not repeat it.

### 2.1 What the identity is tied to — as little as possible

Overstating this is not a safe error: every extra thing in the pin is a case
where a user tunes something reasonable and loses a geometry they should have
kept.

Start from what a run *produces*. **The result is a set of coordinates.**

> **Coordinates cannot be in the identity.** They are the output. Bind them in
> and the pin changes every time a stage succeeds.

What is left is what those coordinates are *of*:

| Considered | In the pin? | Why |
|---|:--:|---|
| the molecule — its formula, or its named components | **yes** | a `.XV` is a list of positions for *these* atoms. Different atoms, and every coordinate lands somewhere it does not belong |
| the positions | no | the output (above) |
| the cell | **no — reported instead** (§ 5) | a `.XV` carries the cell too, so on a continue the saved cell **wins** and the new one is silently ignored. That is a fact to report, not a difference to pin |
| basis, spin, XC | no | the geometry stays valid across all of them, and tuning the electronics while continuing is ordinary practice. A `.DM` of the wrong shape is caught by the engine — a failure it already reports, traded for one it cannot |
| mesh, tolerances, force, steps, algorithm | no | exactly what a description's stages vary |
| ranks, threads, GPU | no | how fast it ran says nothing about whether the answer may be continued |

So the id covers one thing beyond the user's own name for it: **which atoms these
coordinates belong to.** A differing atom *count* the engine already refuses, so
the id is not there for that; it is there for the cases the engine cannot see.

Note the fifth row. The id is deliberately blind to everything a stage tunes —
that is what makes several stages one calculation.

**The known gap:** a formula does not separate two isomers, and does not pin the
*order* species are declared in, which a `.XV` is sensitive to. This is a
starting point, agreed as one; the plan records what would force it to grow.

---

## 3. Normalisation, and the folder

The id becomes the `SystemLabel`, and that becomes the stem of every file:
`<id>.XV`, `<id>.DM`, `<id>_tight.fdf`. A name with a space or a slash in it
breaks a shell line, a glob or a scheduler argument. **What the user types is
never used raw.**

**The allowed set is not a new decision.** `job-contracts.md § 2.1` Rule 2 fixes
it: a single token matching `[A-Za-z0-9_-]+`, rejected at the form/CLI boundary
by `molbuilder/projects.py::_NAME_PATTERN`, with the wrapper accepting the
slightly wider `[A-Za-z0-9._-]+` because a `SystemLabel` may legitimately carry a
dot — and sanitising it there is also what blocks shell injection from a hostile
script (`job-contracts.md § 4.3`). The project tree uses the same set per
segment (`job-contracts.md § 2.5`).

So the normalisation is: **anything outside `[A-Za-z0-9_-]` to `_`, runs
collapsed, leading and trailing separators trimmed, and length capped.**

The cap is derived, not chosen: `<id>_<longest stage name>.<longest extension the
run will write>` must fit the filesystem's name limit (255 bytes, in practice).
Since rule 3 below **refuses** rather than truncating, the cap has to be checked
where the id is made — a truncated id is a different calculation wearing the same
name, which is the one failure this whole document exists to prevent. There is **no lowercasing rule** — that would
make the id the one name in the system that forbids capitals. The
case-insensitive-filesystem worry is handled where it actually bites, by making
the **collision check** case-insensitive.

**And the same set is git-ref-safe**, which the checkpoint history depends on:
`[A-Za-z0-9_-]+` contains none of the characters a ref forbids, so a commit,
tag or branch naming a calculation and its stage needs no second normalisation
(`engines/stages.md § 7.3`). That is not luck — a set narrow enough to survive a
filename, a shell line and a scheduler argument was always going to survive a
ref — but it is worth writing down, because the alternative is somebody
inventing a second sanitiser for tags.

**The id also names the folder.** `job-contracts.md § 2.1` Rule 1 says a
directory holds one job, so the innermost segment of the project tree and the
calculation are the same thing:

```mermaid
flowchart LR
    P["<b>project</b><br/>the user picks"] --> T["<b>topic</b><br/>the user picks, one of the nine"] --> I["<b>id</b><br/>derived — and shown<br/>before anything is written"]
```

That is stricter than the tree requires (`job-contracts.md § 2.5` only asks for
`[A-Za-z0-9_-]+` per segment), and the strictness is the point: a folder whose
name is not the id is a folder whose contents cannot be identified from outside
it.

### 3.1 Normalisation, worked

Each row shows one rule doing its job. The right-hand column is what the surface
must display back **before** anything is written.

| What the user types | Id | Which rule fired |
|---|---|---|
| `BDT/Au relax` + `C6H4S2Au38` | `bdt_au_relax_c6h4s2au38` | `/` and the space are outside the set → `_` |
| `BDT  --  Au` | `bdt_au` | runs of separators collapse to one |
| `_relax_` | `relax` | leading and trailing separators trimmed |
| `Relax.v2` | `relax_v2` | the dot is outside the id's set (the *wrapper* tolerates a dot in a `SystemLabel`; the id does not mint one) |
| `BDT-Au` | `bdt-au` | hyphens are **kept** — they are in the set |
| `Über` | `_ber`, then **refused** | a leading separator trims to `ber`, which is not what was asked for → rule 3 refuses rather than guessing |
| `///` | **refused** | reduces to nothing; say so and ask |
| a 300-character name | **refused** | over the derived cap — a truncated id is a different calculation wearing the same name |

### 3.2 One id, and everywhere it lands

The point of the rule is that a *single* token is enough to identify every file
of a calculation, on disk, in a history, and to the engine. For
`bdt_au_relax_c6h4s2au38`, in the hierarchical shape:

```text
projects/BDT-Au/optimization/bdt_au_relax_c6h4s2au38/   ← the folder IS the id
├── bdt_au_relax_c6h4s2au38.fdf.template
├── 01_coarse/
│   ├── bdt_au_relax_c6h4s2au38_coarse.fdf              ← <id>_<name>
│   └── run-0/
│       ├── bdt_au_relax_c6h4s2au38_coarse.out
│       ├── bdt_au_relax_c6h4s2au38_coarse.molwatch.log
│       ├── bdt_au_relax_c6h4s2au38.XV                  ← <id> alone: the engine's
│       └── bdt_au_relax_c6h4s2au38.DM                     warm files, unsuffixed
└── 02_tight/…
```

and in the history:

```text
commit  bdt_au_relax_c6h4s2au38 · tight · relaxation converged, 41 steps
tag     bdt_au_relax_c6h4s2au38/tight/20260806T221403Z
```

**Three different naming systems — a filesystem, a git ref, and SIESTA's
`SystemLabel` — and the id passes through all three unchanged.** That is what
§ 3's character set buys, and it is why no second sanitiser exists anywhere.

Note which files carry the stage and which do not: **anything a stage produced**
carries `_<name>`, and **anything the engine warm-restarts from** carries the
bare id. That asymmetry is not cosmetic — it is exactly what lets the next stage
find the previous stage's state without being told where it is (§ 4).

**Three rules keep normalisation from becoming a source of surprise:**

1. **It happens once, when the description is written, and the result is
   stored.** The id in the file *is* the id — nothing downstream re-derives it
   from the user's raw text, because two components normalising slightly
   differently is a silent divergence between what a surface shows and what the
   engine writes.
2. **The result is shown, not hidden.** A surface that hides it hides the thing
   that decides whether the next run continues.
3. **A normalisation that loses the name is refused, not patched.** If what the
   user typed reduces to nothing, say so and ask — never append a digit and carry
   on, which produces `bdt_2` and no explanation of what it differs from.

**A "collision" is narrower than it sounds.** The folder is
`<project>/<topic>/<id>/`, so two calculations collide only when they would
occupy the *same topic*. The same id under `optimization/` and under `frequency/`
is not a collision — it is the same system studied two ways, which is what the
topic axis is for, and the warm files never meet because they are in different
directories. A genuine collision is one folder, and it is the case
§ 6 decides: **refuse unless told to overwrite.** The comparison is
case-insensitive, which is where the case-insensitive-filesystem worry above is
actually handled.

---

## 4. Every engine has an internal identity, and parameters bound to it

The id is only half of a warm start, and this is not a SIESTA quirk. **Every
engine has its own notion of "which job is this, and may I continue it", and a
set of parameters tied to that notion.** A design that names only the filename
has described none of it.

Two things, per engine, always:

| | What it settles | SIESTA | PySCF |
|---|---|---|---|
| **the engine's job identity** | which prior state belongs to this run | the `SystemLabel` literal | the `JOB = "…"` literal |
| **the parameters bound to it** | whether that state is honoured when found | `DM.UseSaveDM`, `MD.UseSaveXV`, `MD.UseSaveCG` — SIESTA reads the files *only* when these are set (`job-contracts.md § 4.2`) | the resume branches the generated script carries: `mf.chkfile = JOB + ".chk"` with `init_guess = "chkfile"` when it exists, and `<JOB>_optimized.xyz` overriding the literal geometry |

Read across: the identity is the same *idea* in both columns and a different
*mechanism*; the bound parameters are a declared flag family in one and generated
control flow in the other. Neither is a filesystem fact, and neither is the
scheduler's business.

**So they are one named group per engine, and the group is the generator's:**

1. **An engine declares its group.** Its identity literal, the parameters bound
   to it, and the warm files they govern belong together in that engine's
   contract — the same place `job-contracts.md § 4.2` already inventories the
   files. *A new engine that cannot fill this in is a new engine whose restart
   behaviour nobody has thought about yet.*
2. **The user says one thing; the generator sets the group.** `restart` is a
   single field — `continue` or `clean` — and the renderer expands it into
   whatever that engine's group is: three declared keys for SIESTA, generated
   control flow for PySCF. Nobody keeps a group in step by hand, and no
   description can carry its members individually and disagree with itself.
3. **It is per-stage, because it is an ordinary field.** A first stage is
   normally `clean` and everything after it `continue`. Nothing special is needed
   to say so.

**Why rule 2 is not tidiness.** The two ways the group can be wrong are both
silent, and opposite:

- **honoured with nothing to load** — the parameter is on, no prior state exists,
  and the engine cold-starts while the deck says it resumed;
- **present but not honoured** — the files are right there, the parameter is off,
  and the stage starts from scratch looking like it continued.

One field cannot produce either.

---

## 5. What is reported rather than prevented

An identity that tried to prevent every wrong continuation would have to include
everything, and § 2.1 is about why that is worse. The cases below are real, and
the answer to each is a message, not a wider pin.

| Case | Why the id cannot fix it | Who says it, and what |
|---|---|---|
| **changed cell parameters** | the cell in the deck is *derived* from the parameters and the structure (`model/cell-plan.md`), and derived values cannot be in the id (§ 2). And the hazard is not a mismatch — a `.XV` carries its own cell and its own frame, so on a continue it **wins**: widening the vacuum changes the deck and changes nothing about the run | the surface, at check time: *state found, written under different cell parameters — a continue will keep the saved cell* |
| **the structure moved under a saved description** | the description holds a reference plus a witness (`engines/stages.md § 6.3`), so the mismatch is detectable | the **reader**, as a finding at preflight: *this description was written against a different structure* |
| **prior state from another calculation** | same folder, different id — the engine will not load it, but the user should know it is there | the **surface** at check time, and the **wrapper banner** at run time: *prior state found, but from a different calculation* |
| **prior state that matches** | nothing is wrong; the user should still know before they start | the **surface** at check time, and the **wrapper banner** at run time: *prior state found for this key — the next run continues* |

**Two authors, on purpose.** The wrapper's banner (`job-contracts.md § 4.4`)
already says the last two at run time, which is *after* the user committed. The
surface says them at check time, which is when the choice is still open. Neither
replaces the other, and neither may say something the other contradicts — the
banner is the one that is always present, so it is the one that must never be
weakened. This is the shipped doctrine —
**molbuilder informs, and the user decides to continue** (`job-system.md § 2`,
decision 5) — reaching a case it does not cover today: `WARM-RESTART (silent)`
cannot tell you the state came from a different calculation, because nothing
beside it recorded which one made it. A description written beside the decks
closes that.

---

## 6. Producing twice into the same folder

The id is stable by design, so a second generate targets the folder the first one
wrote. **Refuse unless the user says overwrite, and never rename.**

That is the rule the handoff writer already applies (`job-contracts.md § 5.4`:
it raises unless `overwrite=True`, and *"a UI that writes a handoff SHOULD warn
before targeting a stem that already exists"*). Rewriting decks does not touch
the warm files — which is the point, since they are what the next run continues
from, and "make the name unique" would throw them away.

---

## 7. What this contract does not own

- **The warm-file inventory, the `--cold` move-aside glob, the restart banner
  wording, the run index, the project tree** —
  [`execution/job-contracts.md`](?doc=execution/job-contracts.md) §§ 2 and 4.
  This document adds no file and changes no glob.
- **How a run is launched, which environment it activates, and where activation
  comes from** — [`execution/running-a-job.md`](?doc=execution/running-a-job.md)
  §§ 2, 3 and 5. Unchanged here.
- **What a stage is, and the description the id is derived from** —
  [`engines/stages.md`](?doc=engines/stages.md).
- **Phasing and open questions** —
  [`web/staged-runs-architecture.md`](?doc=web/staged-runs-architecture.md) and
  [`roadmap.md`](?doc=roadmap.md) (R3).
