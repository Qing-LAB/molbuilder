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

| # | unit | status |
|---|---|---|
| **T0** | **The contract** — `template.md` § 6.2–6.5, § 8.0, `@2` bump | ✅ `3ec667a5`, reviewed 3× |
| **T1** | **The SIESTA catalogue** — 44 fields, six categories | ✅ `2e715088` |
| **T2** | **`category`/`engines`/`resolver` in emitter + reader**; `section` retired | ✅ `5d1abbfa` (data), `bba4960c` (code), `07f9763e` (guards) |
| **T3** | **`select` / `one`** — § 8.0's read API | ✅ `8f2e1ab6`; review fixes `1f723e51` |
| **T4** | **Execution items valueless, with resolvers** | ✅ `d4d85418` |
| **T5+T6** | **PySCF renders at all** + its catalogue | ✅ `93403618` — 39 items, six categories |
| **T7** | **`execution/` docs** catch up to `@2` — the version string was the smaller half (§ 5) | ✅ 6 edits across 5 files |
| **T8** | **The wrapper reads `read_by`** instead of scanning deck text for `ELPA` | ✅ **closed by deleting the scan** (§ 9) — the premise under it was false |
| **T9** | **`ecp` → name + atom selector**, deleting `strmap` (§ 8) | ⬜ open — user design, 2 decisions needed first |
| **T10** | **The `.fdf` before/after comparison** (§ 8) | ✅ **57 decks byte-identical**, harness mutation-tested |

**What is true right now.** Both engines render a template at `@2`; every item
carries a category from a closed six; execution items are declared valueless
with named resolvers; `select`/`one` are the one read API; the refactor is
**proven not to have moved the deck** (T10); and the wrapper's reads of the deck
are declared and guarded (T8). **T9 is the only unit left**, and it needs two
decisions before it can start.

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
| `molbuilder/template.py` | emits `@1`, one `engine` key, `section` | ✅ **T2** |
| `execution/job-contracts.md` § 6.1, `execution/project-layout.md` | reference `template@1` | ✅ **T7** |

So a template on disk today is a valid `@1` file and every reader still works.
Nothing is half-migrated; the contract simply landed first, which is what lets
T1's catalogue be checked against a settled shape instead of a moving one.

> **What T7 actually turned out to be** *(2026-08-13)*. The row above scoped it
> as two stale version strings. Reading the two rows in full found the version
> was the *smaller* half: both also described the file's **content** in `@1`
> terms — `engine` singular, and *"every parameter, **with its value**"*, which
> T4 made false when execution items became valueless. The same sentence had
> propagated: `worked-example.md` said the template holds every parameter
> *"minus what the hardware decides"* (at `@2` those parameters ARE in the file,
> named and unanswered), `project-layout.md` § 3 and `engines/stages.md`'s
> diagram repeated *"with its base value"*, and `generator.md` § 3.1a still
> listed **`section`** as a carried template key with `template.md` § 5 named as
> the authority that gains it — the retired key, citing the contract that
> retired it. Six edits, one claim.

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

---

## 7. Fresh-eyes review of the emitted file (2026-08-13, after T3)

Read the rendered `t.toml` and the API's output as a surface would receive
them, rather than the diff. Three findings; none blocks T4, all belong in it or
after it.

**F1 · The `execution` panel is incomplete — it has 6 items, not 9.**
`mpi_np`, `omp_threads` and `max_memory_mb` are tagged `allocation: True`, so
`declaration_for` still excludes them outright — the pre-`@2` behaviour. But
§ 6.4 now says they SHOULD be declared **valueless**, precisely so a surface can
ask and the wrapper writer knows to look. Until T4 lands, a UI reading
`select(t, category="execution")` has no way to ask for ranks, threads or
memory. **This is T4's core work**, stated here so the gap is not mistaken for
a design choice.

**F2 · `resolver` carries no data — the hollow-mechanism pattern, third
instance.** Every execution item reports `resolver=-`. `BlockSize` is `unset`
with nobody declared to fill it, though § 6.4's own table names `block_size` as
its resolver. Specified, implemented, emitted, parsed, declared nowhere.

This is exactly what the T3 review warned about one commit earlier, and the
third time today: `RUNTIME_INFO_KEYS` called itself the source of truth and was
imported by nothing; `read_by` had a contract section and a worked example and
was declared on zero fields. **T4's acceptance criterion should therefore be
behavioural, not structural:** after T4, `select` for resolver-carrying items
must return something AND something must call them.

**F3 · 17 of 41 labels are the engine keyword, not a human name.** § 5 says
`label` is *"the human name — 'MPI ranks (np)'. Not the field name; a surface
shows this."* But the file carries `label = "PAO.EnergyShift"`,
`label = "SystemLabel"`, `label = "kgrid_Monkhorst_Pack"`. A panel built from
this shows the user SIESTA's vocabulary, which is the thing the category work
exists to move past — the categories give good panels and the labels then fill
them with jargon.

Not a template-design defect: the format is right and the data is thin. It is
config metadata to improve, engine by engine, and the natural moment is
alongside each engine's catalogue (T1 for SIESTA, T6 for PySCF). Recorded here
so the UI phase does not discover it as a surprise.

---

## 8. Open units (2026-08-13, end of session)

### T9 · `ecp` becomes a name plus an atom selector (user design)

**Supersedes the `strmap` type added in T5, which should be DELETED with it.**

`strmap` modelled the storage shape PySCF happens to accept (`str | dict`)
rather than the question a user answers. The question is two:
**which ECP**, and **which atoms get it**. A dict conflates them and can express
something nobody wants — different ECP families on different elements in one
calculation.

```toml
[item.ecp]         type = "str"      value = "lanl2dz"
[item.ecp_atoms]   type = "strlist"  value = ["Au", "Pt"]
```

`[]` none · `["*"]` all · `["Au"]` one element · `["A*"]` a pattern.

**Both halves already have types**, so a new entry in the closed type vocabulary
disappears — and it renders as two ordinary controls instead of a raw table
nobody can validate, which is the point of the whole programme.

**The scientific premise, to confirm before building:** nobody needs *different*
ECP families per element in one run. def2 bases carry their own ECPs; LANL2DZ is
what you reach for when you are *not* on def2. Mixing families across elements is
legal in PySCF and almost never right — so the dict form is not a capability
being lost, it is one that should not have been exposed.

**Scope — larger than `strmap` was.** That was one serialization function; this
changes `PySCFConfig`'s schema (one field becomes two), the emitter that builds
`gto.M(ecp=...)`, the auto-detection that picks ECPs for heavy atoms on non-def2
bases, and validation. The wildcard matcher needs writing and testing against
real element symbols.

**Two decisions needed first:**
1. Does `["*"]` mean *all atoms* or *all heavy atoms*? The current auto-rule is
   heavy-only.
2. Is `ecp = ""` with a non-empty atom list an error, or *"auto-pick per
   element"*?

### T10 · The `.fdf` before/after comparison — ✅ DONE 2026-08-13

G4 asks that the deck `prep` renders is the deck the surface would have
rendered, and the template refactor changed `SiestaConfig`'s metadata (category,
resolver, label, item_kind) with no intended change to deck output. **Verified:
the decks are byte-identical.**

**Method** — `2e715088` (the last commit before T2a touched code) checked out as
a worktree beside the tree under test, and one harness run against each:

| | |
|---|---|
| matrix | **19 configs × 3 stage tokens = 57 decks**, built so every one of the 44 `SiestaConfig` fields is off its default in at least one case |
| structures | a molecule with vacuum, a periodic Au cell with an explicit lattice, and a junction carrying `regions` + `frozen_atoms` + an annotation channel |
| determinism | `generated_at_now` and `molbuilder_git_sha` frozen **before** `siesta.input` is imported, so the only thing that can differ is the deck |
| result | `diff -r` exit **0** over **943,580 bytes**; 0 refusals; smallest deck 4,519 bytes |

**The assertions that make the result believable** — each one closes the exact
hole the failed attempt fell through:

- the harness **raises on any empty file** before writing the next one;
- every run **prints the `molbuilder.__file__` it imported**, so "two trees" is
  shown rather than assumed;
- a **refusal is captured as file content** (`!! REFUSED: …`), so a config the
  generator rejects can never masquerade as an empty deck;
- and the comparison was **mutation-tested**: changing `mesh_cutoff`'s default
  from `300.0` to `301.0` in the baseline tree alone produced a **614-line
  diff**, proving the harness detects a one-value change. The baseline was then
  restored and the restore re-verified by re-running the diff.

**§ 9's reserved blocks — asserted PRESENT, not merely unchanged** (an absent
block would diff clean in both trees):

| block | decks | expected |
|---|---|---|
| `=== molbuilder user-custom ===` | 57/57 | always |
| `=== molbuilder provenance ===` | 57/57 | always |
| `=== molbuilder bench-marks ===` | 57/57 | always |
| `=== molbuilder atom-metadata ===` | **6**/57 | only the junction cases — the contract's rule is that absence is the honest signal when `regions` and `frozen_atoms` are both empty |
| `%block LatticeVectors` | 57/57 | the crystal's explicit lattice, and the derived vacuum box |
| `%block Geometry.Constraints` | **6**/57 | `frozen_atoms=[0,1,2,3]` → `position 1 2 3 4`, 0-based to 1-based |

The atom labels ride in the `atom-metadata` JSON (`regions`, `frozen_atoms`, and
the annotation channels round-trip verbatim), not as inline comments on the
coordinate lines — which is what § 9.1 specifies.

**Harness:** `t10_render.py`, kept in the session scratchpad. It is deliberately
NOT a test in the suite: it compares two *trees*, so it has no meaning once the
baseline is old. The reproduction recipe is the table above.

---

## 9. T8 — closed by deleting the read, not by declaring it (2026-08-13)

T8 was written as *"the wrapper reads `read_by` instead of scanning deck text for
ELPA"*, and `generator.md` § 8 listed that scan under **what this design
deletes**. Walking it before writing any code found the design's premise, and
then found the premise was false.

### 9.1 What the walk found first — a declaration nobody could be missing from

The wrapper reads the deck in **two** places, for **two different questions**:

| scanner | question | call sites |
|---|---|---|
| `_fdf_requests_elpa` | which conda env to activate | 1 |
| `_fdf_requests_gpu` | the GPU **runtime** — gres, MPS, the NUMA pin, the rank/thread budget | 8 |

Only `diag_algorithm` declared `read_by`. An implementation that trusted the
declarations — which is exactly what T8 was — would have dropped every GPU
runtime fact in silence, while the deck still said `Diag.ELPA.GPU .true.`

So `enable_gpu` got its declaration, and a guard was written for **the direction
that catches drift**: *for every place the wrapper reads the deck, some item
declares that read.* The scanners are their own oracle — the test builds a
one-line deck from a field's `engine_key` and asks the scanner whether it sees
it, so it never restates a keyword and cannot drift from `runwrap`'s own regexes.
Mutation-tested: removing the declaration fails it by name (`24e5cd69`).

### 9.2 Then the premise collapsed

The ELPA scan existed because *"ELPA is linked only into the source build"*
(`recipes.py`, and three docs downstream). **Measured instead of inherited** — an
H2 probe in the packaged `molbuilder-siesta`:

| deck | result |
|---|---|
| `Diag.Algorithm ELPA-2stage` | exit 0 — E = −30.136019 eV |
| `Diag.Algorithm ELPA-1stage` | exit 0 — E = −30.136019 eV |
| `Divide-and-Conquer` | exit 0 — E = −30.136019 eV, identical |
| `ELPA-2stage` + `Diag.ELPA.GPU .true.` | **exit 1** — `ELPA_ERROR_ENTRY_NOT_FOUND` |

conda-forge's SIESTA links no external `libelpa` — the true observation the false
conclusion was drawn from — but ELPA is compiled **in** through ELSI: 279 defined
ELPA symbols, zero undefined. Only the GPU entry is absent, a **missing build
option, not a missing device**.

**The harm was concrete.** The two SIESTA envs split on **provenance**: one
installs from packages on any machine, the other must be **built from source**,
which some HPC sites do not permit. Routing CPU-ELPA to the source build refused
a runnable calculation wherever compiling is not allowed — for a solver the
installed baseline already has.

### 9.3 How it closed

`_fdf_requests_elpa` lost its only caller and was **deleted**; GPU alone routes;
`diag_algorithm`'s `read_by` came **back off**, because the wrapper genuinely
reads nothing from it. The wrapper now reads exactly one item, and that one is
declared and guarded.

> **The lesson worth keeping is the shape of the error, not the ELPA fact.**
> *Knowing a keyword is not providing the capability* — the packaged binary
> carries `ELPA-1stage`, `ELPA-2stage` and `Diag.ELPA.GPU` as strings whether or
> not it can run them. A `read_by` declaration asserting a dependency that does
> not exist is the same defect the key exists to remove, aimed the other way.

### 9.4 The vocabulary this settles, for the UI to build on

**Capability and need are different things, pointing in opposite directions:**

| | declared by | says |
|---|---|---|
| **provides** | a **build** | *"I have ELPA, and it was built with CUDA"* |
| **requires** | a **deck value** | *"this run needs a GPU-capable ELPA"* |

And a need splits by **who satisfies it** — a **build need** by choosing an env,
a **run need** by the scheduler ask. `Diag.ELPA.GPU .true.` raises one of each,
which is why fusing them into the single token `siesta-gpu` broke.

**User decisions (2026-08-13), recorded so they are not re-derived:**

- **Two envs, and the count stays two.** The capability vocabulary is finer than
  the env list, so precision never costs a third environment.
- **What is *available* is filtered by what is *needed*, and the user picks.**
  Auto-routing is the fallback for *no choice given* only — `write_run_wrapper`
  already guards on `env is None`, so a named env always wins.
- **`bench` sweeps computation resources, never the env.**
- **CPU-ELPA is offered as a choice** when GPU is not selected.
