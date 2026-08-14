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
| `category` | which **question about the calculation** — the UI panel | yes, six |
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
| 5 | `outputs` | what do I want produced? | — |
| 6 | `execution` | how does it run on this machine? | **yes** |

**`accuracy` ≠ `convergence`.** Accuracy is what answer you will *accept*;
convergence is how to *reach* it. A user whose SCF oscillates must reach for
`level_shift`, not a looser `conv_tol` — which "fixes" the symptom by accepting a
worse answer. One panel holding both invites that substitution.

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
| **Q1** | Is `basis` `method` or `accuracy`? | T1 | `method` — it is the dominant accuracy knob but is *reported* as level of theory (*"B3LYP-D3/def2-TZVP"*) |
| **Q2** | Is `electronic_temperature` `accuracy` or `system`? | T1 | `accuracy` for molecules (smearing aids convergence), but it changes occupations, so `system` for a genuinely finite-temperature calculation. May need both, by context |
| **Q3** | Should `optimize` / `compute_frequencies` be a 7th category (*task*), above the rest? | T1 | keep in `outputs` — the ladder in `task.json` is the real mission statement. Reopen if T1 shows the panel reads badly |
| **Q4** | Does `max_memory_mb` split into two items? | T1, U-C | **yes.** SIESTA's is `execution` (a ulimit); PySCF's is `accuracy`/`convergence` (it selects in-core vs out-of-core). Under this scheme they were never one item — the cleanest argument yet for the U-C split |
| **Q5** | Do `hook`-flagged items need a declared hook *name*, or is `kind` + `read_by` enough? | T4 | defer — `BlockSize` is the only worked example, and one example is not a mechanism |

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
6. **The vocabularies are closed** (`kind`, `category`). An unknown value is an
   error a reader reports, never something it drops.
