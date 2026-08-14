# The template unification — plan of record

**Role:** plan
**Domain:** execution
**Companions:** [`engines/template.md`](?doc=engines/template.md) (the contract
this plan implements — **it is the authority; where the two disagree, the
contract wins**) · [`engines/stages.md`](?doc=engines/stages.md) ·
[`execution/architecture.md`](?doc=execution/architecture.md) § 2 (the floors) ·
[`web/form-schema.md`](?doc=web/form-schema.md) (the metadata every item is
built from).

> **Why this file exists.** The template is the single source of truth for the
> whole workflow — a surface builds panels from it, `prep` resolves from it, the
> deck writer and wrapper writer render from it, a benchmark sweeps from it.
> Getting it wrong propagates everywhere, so the design is written down before
> it is built and reviewed against itself rather than discovered in code.

---

## 1. The problem, measured

Not an aesthetic complaint — these are counts taken from the shipped schemas on
2026-08-13.

| finding | measurement |
|---|---|
| **The same physics is unrecognisable across engines** | `SiestaConfig` ∩ `PySCFConfig` by field name = **3**, and all three are molbuilder-internal (`verbose_comments`, `write_molwatch_log`, `max_memory_mb`) |
| **Charge is spelled two ways** | SIESTA `net_charge` → `NetCharge`; PySCF `charge` → `gto.M(charge=)` |
| **Spin is decomposed two ways** | SIESTA `spin_polarized` + `spin_total`; PySCF `spin` (2S) |
| **Sibling schemas already drift** | `PySCFConfig` ∩ `SpectraConfig` = 17 shared fields, separately maintained — `scf_conv_tol_grad` and `scf_soscf` landed in one and not the other |
| **Only one engine can produce a template at all** | SIESTA renders 41 items; PySCF, spectra and transport **refuse** |
| **`max_memory_mb` is one name for two things** | SIESTA: a `.run.sh` ulimit. PySCF: `mol.max_memory`, which selects in-core vs out-of-core |

The consequence for the UI: a form cannot be written once, and nobody can ask
*"what charge did this run use?"* without first knowing the engine.

---

## 2. The design, in one page

**A template describes a CALCULATION, not an engine.** One file carries every
engine the calculation can run on. Items are **never merged** across engines —
`net_charge` and `charge` stay two items — but they are **grouped** by a closed,
engine-independent `category`.

Three orthogonal axes on every item:

| axis | answers | closed? |
|---|---|:--:|
| `category` | which **questions about the calculation** it answers — a **LIST**; first entry is the panel (§ 5.4) | yes, six |
| `engines` | whose **vocabulary** it is | yes, the file's list |
| `kind` | which **layer consumes** it | yes, five |

Plus `read_by` (who *else* derives from the value) and `group` (whether it
varies per stage). Neither is a category.

**Why grouping and not merging.** Merging demands a shared physics vocabulary and
per-engine derivation, and risks fusing things that only sound alike:
`dm_tolerance` is a density-matrix criterion, `scf_conv_tol` an energy one. Both
are *"SCF convergence"* in English; neither can take the other's value. Grouping
buys the same UI benefit at none of that risk, and leaves every emitter untouched.

### 2.1 The six categories

In reading order — the order a person decides in, and a methods section is
written in.

| # | `category` | the question | benchmarkable? |
|---|---|---|:--:|
| 1 | `system` | what am I calculating? | no |
| 2 | `method` | at what level of theory? | no |
| 3 | `accuracy` | how precisely are the equations solved? | no |
| 4 | `convergence` | how do I reach it when it fights? | — |
| 5 | `procedure` | what does the run carry out, and what does it leave behind? | — |
| 6 | `execution` | how does it run on this machine? | **yes** |

**`accuracy` ≠ `convergence`.** Accuracy is what answer you will *accept*;
convergence is how to *reach* it. A user whose SCF oscillates must reach for
`level_shift`, not a looser `conv_tol` — which "fixes" the symptom by accepting a
worse answer. They remain distinct *categories* for that reason, but a **surface
may present them as one panel** (§ 5.4) — the distinction is a semantic one the
`help` text carries, not a wall the UI must build.

**`execution` is the sweepable set**, because those knobs change **speed and not
the answer**. `mesh_cutoff` changes speed too, but it changes the answer, so
sweeping it measures a different calculation each time.

### 2.2 The one read API

`select(t, *, category=, engine=, kind=, read_by=)` → items in category order;
an omitted argument means *do not filter on it*. `one(t, name, engine=)` returns
`None` when the item does not apply to that engine and **raises** when it is
required and missing — *"does not apply"* and *"should be here and is not"* must
never read the same.

A filter over data the caller already holds — **not a service it must call**, so
§ 8's *"a reader never asks a second source"* survives and a reader in another
language does the same with `tomllib.load` and a comprehension.

---

## 3. The review — what two passes found

Recorded because these are the errors most likely to recur.

**Contradictions introduced by the first draft**, all fixed in `3ec667a5`+:

1. **§ 2 said no item may be a machine fact**, while § 6.4 put execution asks in
   the file. Resolved by splitting *the item* from *its value* — see § 3.1.
2. **§ 7's exclusion table** repeated the same claim. Same fix.
3. **§ 11.4** said *"the template does not change and does not know"* about rank
   counts. Now: the item may be declared but stays valueless, and a sweep never
   writes one.
4. **§ 5 still listed `section`** as a live key after § 6.2 retired it.
5. **§ 3 and § 4.2 still showed `@1` and a single `engine` key.**

**A contradiction that predated the draft**, and settled the design:

§ 2's bullet said a rank count is *"a machine fact and never an item"* — but the
correction recorded at the bottom of the same section (user, 2026-08-11) says
**the test is not *"could a machine decide this?"* but *"may a person?"***. A
person may certainly ask for 8 ranks. The bullet was the stale half.

### 3.1 The rule that resolves it

> **The template may declare the QUESTION; it may not assert the ANSWER.**

An execution item is declared **valueless**: a surface knows to ask, the wrapper
writer knows to look, and no machine fact is asserted. Writing a `value` to one
is what a reader **refuses** — which is exactly the failure the original rule was
written against (a hand-edited `mpi_np` once rendered a deck for ranks the
allocation never granted).

The template carries the **ask**; `prep` resolves it against `environment.json`.
The same asked-vs-effective seam `bench/result.py` draws, and for the same
reason: conflating them produced a benchmark that measured labels, not runs.

---

## 4. Order of work

Each unit lands with its own tests and commit. **Docs first within each unit.**

| # | unit | why this order |
|---|---|---|
| **T0** | ✅ **The contract** — `template.md` § 6.2–6.5, § 8.0, `@2` bump | done (`3ec667a5`), reviewed twice, this file |
| **T1** | **The SIESTA catalogue** — all 44 fields placed in the six categories, misfits named rather than forced | SIESTA's backend already works with the template, so this is a **regroup, not a migration**. It is also the cheapest way to test whether six categories are right |
| **T2** | **`category` + `engines` in the emitter and reader** — `declaration_for` emits them, `read_template` requires them, `section` retired | the schema change, once the categories are confirmed by T1 |
| **T3** | **`select` / `one`** — § 8.0, with `prep` step 2 repointed as the first caller | one real caller, not a speculative API |
| **T4** | **Execution items as valueless declarations** — rank count, threads, wall time | needs T2's `category` to have somewhere to live |
| **T5** | **PySCF renders at all** — `ecp`'s union annotation, then whatever refuses next | today it cannot produce a template; this is the blocker for every PySCF claim below |
| **T6** | **The PySCF catalogue**, same six categories | proves the categories are engine-independent rather than SIESTA-shaped |
| **T7** | **`execution/` docs** — `job-contracts.md`, `project-layout.md`, `architecture.md` updated for `@2` and the one-file model | written once, after the shape stops moving |
| **T8** | **The second reader** (deck writer or wrapper writer) through `select` | the point at which the API earns its keep |

**Not in scope, and deliberately:** the UI. It is the last consumer and it is
built from this file; designing it now would fix panels against categories that
T1 may still move.

---

## 5. Open questions — decide before the dependent unit

| # | question | blocks | my position |
|---|---|---|---|
| **Q1** | Is `basis` `method` or `accuracy`? | ~~T1~~ | **MOOT (2026-08-13).** `category` is a list — it is `["method", "accuracy"]`. See § 5.4 |
| **Q2** | Is `electronic_temperature` `accuracy` or `system`? | ~~T1~~ | **MOOT (2026-08-13).** *"May need both"* was the answer; it is `["accuracy", "system"]`. See § 5.4 |
| **Q3** | Should a 7th category split out of `outputs`? | ~~T2~~ | **ANSWERED 2026-08-13 (user): no split — RENAMED to `procedure`.** The three groups (what job, which files, how written) are all *the job in general*, and the group is engine-specific bookkeeping rather than science. `procedure` was the only candidate with zero collisions: `task` appears in 18 contract docs and names a first-class file, `job` has three doc files named after it, `run` is the sweep unit |
| **Q4** | ~~Does `max_memory_mb` split into two items?~~ | ~~T1, U-C~~ | **ANSWERED 2026-08-13 (user): no split — it becomes a valueless item.** See § 5.1 |
| **Q5** | Do items needing a computed value declare a *resolver*? | T4 | **ANSWERED 2026-08-13 (user): yes — a named resolver.** See § 5.2 |

---

### 5.1 Memory is asked for, not budgeted (answers Q4)

**User decision, 2026-08-13.** Memory is not the limiting factor when a job
queues — **core and GPU counts are**. A job asks for what the node has, and the
maximum available is already known: from `environment.json` on a cluster, or
detected on a workstation. So a *number* in the template earns nothing.

`max_memory_mb` is therefore **one item, `category = "execution"`, normally
valueless** — the § 6.4 pattern, and the third instance of it after `BlockSize`
and the rank count:

| state | means |
|---|---|
| no `value` | **use the maximum allowable**, resolved at `prep` from the environment |
| `value` set | an explicit ceiling — honoured, but see the clamp below |

**This retires the Q4 split.** The split was proposed because one name carried
two things: an allocation ceiling (machine) and `mol.max_memory` (science, it
selects in-core vs out-of-core). If *unset means the ceiling*, the two coincide
by construction and there is nothing to separate. It also closes REVIEW-2's
"PySCF `max_memory` clobber" — nothing clobbers anything when the default IS the
allocation.

**It also fixes a live defect.** `PySCFConfig.max_memory_mb` currently defaults
to a **static 4000 MB** regardless of the machine. On a large node PySCF goes
out-of-core for no reason; `mol.max_memory` is precisely what it consults to
choose. *"The maximum available"* is a better default than any constant, because
the right constant does not exist.

**An explicit value is CLAMPED to the allocation, and the clamp is logged.**
Asking for 64 GB on a 16 GB grant and being believed means PySCF chooses in-core
and is OOM-killed hours in; the failure is late, expensive and unexplained. You
cannot be given more than you were granted, so `prep` clamps and says so, and the
ask and the effective value are both recorded — the asked-vs-effective seam again.

**Lands in T4** with the other execution items, and the config default changes
from `4000` to unset in the same unit.

### 5.2 `resolver` — the item names who computes its value (answers Q5)

**User decision, 2026-08-13.** Some items cannot be answered by a constant: the
value depends on the machine, on an explicit ask, or on both. The item **names
the resolver**, and `prep` calls it.

Four known members, which is what makes this a mechanism rather than a special
case for `BlockSize`:

| item | unset → | an explicit value → |
|---|---|---|
| `max_memory_mb` | the node's maximum, from `environment.json` or detection | clamped to the allocation, and the clamp logged (§ 5.1) |
| `block_size` | proposed from the orbital and rank counts | honoured verbatim |
| rank count | the allocation | it is an ask; `prep` resolves against what was granted |
| `threads` | `OMP_NUM_THREADS` → `SLURM_CPUS_PER_TASK` → `PBS_NCPUS` → `NSLOTS` → physical cores | honoured; it outranks the chain |

**A NAME from a closed registry — never code in the file.** The template is data:
hand-editable, and it travels between machines. Executable content in it would
end both properties, and would make a description something you must trust rather
than something you can read. So `resolver = "node_memory"` names a resolver
molbuilder ships, an unknown name is an error a reader **reports** (§ 3), and the
registry is a closed vocabulary like `kind` and `category`.

**The contract:**

```
resolve(asked: value | None, env: Environment) -> (effective, reason)
```

`reason` is not decoration. Every one of these produces a number the user did not
type, and this session's rule holds: a value the run obeys but nobody can see is
the same problem as an undocumented one. The reason lands in the log and the
decision ledger, so *"64 GB, clamped from your 96 GB ask to the allocation"* is
readable after the fact.

**Why this belongs on the item rather than in `prep`.** Today `prep` must *know*
that `block_size` needs proposing and `threads` needs a chain — a layer carrying
a list of special field names, which is exactly what G3 exists to remove. With a
declared resolver, `prep` calls what the item names and a new engine adds its own
without anyone editing `prep`. The same argument `read_by` won in § 6.1.

**It does not weaken § 2.** The resolver runs at `prep`, on the machine, with
floor 1 knowledge. The template still declares only the question; the answer is
computed where the answer is knowable.

---

## 5.3 The interim state — the contract is ahead of the code, on purpose

**Until T2 lands, `engines/template.md` describes `@2` and the code emits `@1`.**
That is the intended order (docs first), but it must be visible rather than
discovered:

| where | says | until |
|---|---|---|
| `engines/template.md` | `@2`, `engines`, `category`, `resolver` | — (it is the authority) |
| `molbuilder/template.py` | emits `@1`, one `engine` key, `section` | **T2** |
| `execution/job-contracts.md` § 6.1, `execution/project-layout.md` | reference `template@1` | **T7** |

So a template on disk today is a valid `@1` file and every reader still works.
Nothing is half-migrated; the contract simply landed first, which is what lets
T1's catalogue be checked against a settled shape instead of a moving one.

---

### 5.4 `category` is a list, and it does not affect the script

**User decision, 2026-08-13.** Two facts, and the second follows from the first.

**1. A category changes nothing the engine sees.** The deck writer filters on
`kind`; a differently-categorised item produces byte-identical output. It is a
**presentation and discovery** key. T1 spent six paragraphs agonising over
placements that could not alter a single line of any deck.

**2. So an item may carry several.** `category = ["method", "convergence"]` —
first entry is the panel, the rest make it findable where a user would look.
Parameters genuinely belong to more than one question, and forcing one answer
both wasted effort and *hid* items from the people hunting them.

This retired all six of T1's hard placements (see the catalogue § 2). The
load-bearing one was `solution_method`, which I had placed in `convergence` while
arguing it belonged in `method`; it is now both, which is what the note *"argues
for a cross-reference rather than a second category"* was reaching for.

**The nuance belongs in `help`.** *"`OrderN` scales linearly but is approximate —
it changes the answer, not just the route"* is a sentence a user reads at the
moment of choosing. It was never a taxonomy problem.

**A surface may coarsen the panels.** Showing `accuracy` and `convergence` as one
*Calculation standards* panel is a presentation decision the template does not
forbid. What the template owes a surface is the semantics; how many panels they
become is the surface's call.

> **`execution` is the exception and stays a precise claim.** A benchmark takes
> it as the **sweepable set** — knobs that change speed and not the answer. Tag
> something `execution` that changes the answer and a sweep silently measures a
> different calculation at each point. Unlike the other five, this one is not a
> hint about where to show an item.

---

## 6. What must stay true — the invariants a review checks

1. **Membership is total** (§ 7, D5). Every schema parameter is an item. *"Is it
   in the file?"* is never a judgement call.
2. **No item asserts a machine fact's value** (§ 2, G1). Declaring the question
   is portable; answering it on the wrong machine is not.
3. **Items are never merged across engines** (§ 6.3). Two spellings stay two
   items in one category.
4. **A reader never asks a second source** (§ 8). `select` is a convenience over
   data in hand, never a service.
5. **Each value is stored once** (D3). A category is a label on an item, not a
   second copy of it.
6. **The vocabularies are closed** (`kind`, `category`, `resolver`). An unknown
   value is an error a reader reports, never something it drops.
7. **A template contains no executable content** (§ 6.4). `resolver` is a name
   from a registry molbuilder ships. A description you must *trust* rather than
   *read* is not a description.
8. **A computed value carries its reason** (§ 5.2). Every resolver returns one,
   and it reaches the log and the ledger — a number the run obeyed that nobody
   can account for is the failure this whole session kept finding.
