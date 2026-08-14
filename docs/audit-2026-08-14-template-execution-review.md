# Full-text review — template, execution, and their backend

**Role:** audit report
**Domain:** *(root — cross-domain review of `engines/template.md`, `execution/`,
and the backend that implements them)*
**Audited:** 2026-08-14
**Method:** every file below read **in full text**, front to back. No `grep`, no
`diff` — a keyword search finds what you already suspect, and the defects that
matter are the ones nobody suspected. Suspicions raised by reading were then
**confirmed by probe** (running the code), never by a second search.

> **This is a findings log, not a fix program.** Nothing here has been changed.
> Ordering within each section is by severity, and every entry says how it was
> confirmed so a reader can re-run the check rather than trust the claim.

---

> **⏩ RESUMING AFTER A CONTEXT RESET? READ § 21 FIRST.** It is the cross-index:
> every closed vocabulary with its members, every load-bearing rule with its
> owner, and the measured facts — compressed so a **not-yet-read** document can
> be checked against what has been read, without re-reading it. § 21.4 says how
> to use it; § 22 proposes how to keep it from going stale.

## Coverage — what was read in full, and what was not

| read in full | lines |
|---|---|
| `docs/engines/template.md` | 1010 |
| `molbuilder/template.py` | 1111 |
| `docs/execution/generator.md` | 594 |
| `molbuilder/resolve.py` | 559 |
| `docs/execution/job-system.md` | 1322 |
| `docs/execution/job-contracts.md` §§ 3.1–3.2, 6.2, 6.3 | *(those sections in full)* |
| `docs/execution/project-layout.md` § 7 — the invariants | 110 |
| `molbuilder/jobset/model.py` — the serialisation surface | 140 |
| `molbuilder/task.py` — the model | — |

**Total read in full: ~5,400 lines.** What remains (~17,000) is listed in § 16
with the order the next pass should take and what each file cross-checks
against — because the value here came from reading later files against earlier
ones, and that is what a continuation has to preserve.

**Stated plainly because a partial review reported as a whole one is worse
than no review.**

**Layout of this report:** findings per surface first (§§ 1–9), then the two
cross-cutting sections — the repeating patterns, and the design assessment —
**last** (§§ 10–11). A synthesis buried between findings is the same
misplacement this report charges elsewhere; it was buried here until this
pass, which is worth admitting rather than quietly fixing.

---

## 0 · Finding index — the CORE DOCUMENT to open for each issue

**Every finding names the document that OWNS the concept, not only the line
where the symptom shows.** A fix aimed at the symptom is how this corpus got
here: the same fact was corrected in one place and left in another (§ 11,
pattern 3). Open the core document, fix the rule there, then sweep every
restatement the ownership map below lists.

| # | finding | **core document — where the concept is OWNED** | what to look for beyond the local line |
|---|---|---|---|
| 1.1 | `_unwrap_optional` misses PEP 604 | `engines/template.md` § 5 (the `type` vocabulary) → `molbuilder/template.py` | **every place an annotation is inspected.** The sibling `_decl_type` had the same bug; look for `typing.get_origin(...) is typing.Union` anywhere, including `cli.py`'s option bridge, which walks annotations the same way |
| 1.2 | `engines` is a hollow axis | `engines/template.md` § 6.3 | the decision is *contract-level*: either a producer writes multi-engine files, or § 6.3 says "one engine per file today". Then `select`/`one`/`_check_engine`/`Item.engines` follow. Also `generator.md` § 7's engine-seam claim rests on it |
| 1.3 | `select(read_by=)` takes a scalar only | `engines/template.md` § 8.0 | § 8.0 lists the calls each reader makes — check **every** documented call signature against the implementation, not just this one |
| 1.4 | `category` required by contract + reader, not by `Item` | `engines/template.md` § 3 | § 3's *"four required keys"* is stated in three places (§ 3 table, `_REQUIRED_ITEM_KEYS`, `Item`'s docstring) and they disagree. Fix the count in one place and make the other two derive or assert |
| 1.5 | `__all__` omits the headline API | `docs/process/package-layout.md` (module surfaces) | **not a template issue** — it is 2 of 2 modules read. Belongs to the package convention, and wants one test over the package |
| 2 | contract describes retired ELPA routing | `engines/template.md` §§ 6.1, 11.3, 4.2 | `read_by`'s whole *rationale* is the ELPA story. Rewriting the example is not enough — § 6.1 must be re-argued from `enable_gpu`, its only live case |
| 2 | machine axes "not a template item at all" | `execution/generator.md` § 4 | the `@2` change (T4) makes them valueless items; check § 4's whole two-family table, § 4.2's pin channel, and § 10's class-3 row for the same assumption |
| 3.1 | duplicate TOML key in the example | `engines/template.md` § 6.3 | **every fenced example in the file** — § 3.2 and § 9.2 show the same family (examples left behind when the rule moved) |
| 3.2 | `block_size` example omits `category` | `engines/template.md` § 12 | as above — validate examples against § 3's required set |
| 3.4 | `section` residue | `engines/template.md` § 5 | `section` is retired in the template but **live in the FORM** (`web/form-schema.md`). Any sweep must keep those apart — same word, two mechanisms |
| 4 | residue in `template.py` | `engines/template.md` § 5 + the module | dead `dict` branch, stale comments, `Template.engine` compat shim. The shim is governed by the project's *rename = delete old* rule, not by the template contract |
| 5 | `generator.md` § 8's deletion row | `execution/generator.md` § 8 | the row credits `read_by`; the deletion happened for a different reason. Check § 8's other rows against what actually landed |
| 6.1 | floor 3 imports from `siesta/` | `execution/generator.md` §§ 6.1, 7 (the floor rule and the engine seam) | the fix is a **move**, not a rewrite: `effective_config` is engine-agnostic and lives in an engine package. Check what else `resolve/`, `materialize` and `submit` import from `siesta/` |
| 7.1 | `Resources` shown as 7 fields | `execution/job-contracts.md` § 6.2 **(the authority)** | the guide's copy is the symptom. The authority is right and **tested** — so the fix is to delete the copy, not to sync it |
| 7.2 | a guide restates a contract | `execution/job-contracts.md` § 6.2 vs `job-system.md` § 3 | sweep `job-system.md` for **every** vocabulary table it restates from job-contracts; it cites the authority and repeats it anyway |
| 7.3 | one heading twice in one file | `execution/job-system.md` §§ 1, 5.4 | — |
| 8.1 | GPU-request row names `diag_algorithm` | `execution/job-contracts.md` § 6.2 | this is **the cross-layer authority** (§ 6.3 says it wins) — a stale row here propagates by design. Check every row for T8/T9 residue |
| 9.2 | `kgrid` in the PROVENANCE example | `execution/job-contracts.md` § 3.2 | same examples-left-behind family |

---

## 0.1 · Ownership map — who owns each concept, and who restates it

Built by reading the four documents together, which is the only way this is
visible. **A restatement is not automatically wrong** — a guide may summarise.
It is wrong when it restates *the enumerable detail* the contract owns, because
that is what drifts (§ 11, pattern 3).

| concept | **owner** | restated by | verdict |
|---|---|---|---|
| the item model (`kind`, `category`, `type`, `value`) | `engines/template.md` §§ 3, 5, 6 | `generator.md` § 3.1a (the three UI keys) | **acceptable** — § 3.1a argues *why* the keys exist and defers the key set to § 5 |
| which parameters are machine facts | `engines/template.md` §§ 2, 7 | `generator.md` § 4 · § 10 class 3 · `resolve.py`'s gates | **§ 4 has DRIFTED** — it still says "not a template item at all" |
| the resource vocabulary (the nine fields, config ↔ SLURM) | `execution/job-contracts.md` § 6.2 | `job-system.md` § 3 + § 3.1's example · `Resources`' docstring | **§ 3 has DRIFTED** (7 vs 9). The docstring restatement is *deliberate and argued* — and is the one that stayed correct, because it sits beside the code |
| every name in the system | `execution/job-contracts.md` § 6.3 | `resolve.point_token` · `identity.stage_token` | **clean** — and the model for the rest: a rule, the one function that implements it, and they agree |
| the five `prep` steps | `execution/project-layout.md` § 2.3.1 | `template.md` § 11.1 · `generator.md` § 3 · `jobset/prep.py` | **checked — the prose agrees; the NUMBER collides** (§ 12) |
| the reserved blocks | `execution/job-contracts.md` § 3.1 | `engines/template.md` § 9 | **clean** — § 9 asks a different question (*where does each block's content come from*), which is complement, not copy |
| what a stage is, and overrides | `engines/stages.md` § 4 | `template.md` § 1 · `resolve.py` `_apply` | **not yet read** |

> **The pattern in this table is the finding.** Every clean row is one where the
> restatement answers a *different question* from the owner. Every drifted row
> is one where the restatement answers the *same* question in fewer words.
> That is a usable editing rule: **a guide may say why; the contract says what,
> and the guide must not enumerate it.**


---

## 1 · Defects confirmed by probe

Each was found by reading, then verified by running code.

### 1.1 `_unwrap_optional` mishandles PEP 604 unions — live latent bug

```
Optional[str]  ->  (str, True)            correct
str | None     ->  (str | None, False)    WRONG
```

`_decl_type` then returns `None`, so `declaration_for` raises *"no declaration
type for annotation … a gap in the vocabulary"* — for a field that is entirely
ordinary. **Any config field written `X | None` instead of `Optional[X]` breaks
template rendering, with a message pointing at the wrong cause.**

`typing.get_origin(Optional[str])` is `typing.Union`; `typing.get_origin(str |
None)` is `types.UnionType`. They are different objects and the function checks
only the first.

> **This is the same defect that was found and fixed in `_decl_type` on
> 2026-08-13, in the function immediately below it.** One half of the pair got
> the `types.UnionType` check; the other did not. The lesson is not about
> unions — it is that a fix applied at the point of failure did not get applied
> to the sibling that shares the idiom.

### 1.2 `engines` is a hollow axis — the fifth of this family

`template.md` § 6.3's central claim: *"A template describes a calculation, not
an engine. One file carries every engine the calculation can run on."*

Probed:

| check | result |
|---|---|
| fields carrying `engines` metadata in `SiestaConfig` | **0** |
| fields carrying `engines` metadata in `PySCFConfig` | **0** |
| what `render_template` emits | `"engines": [eng]` — hard-coded, one element |

So `Item.engines`, `select(engine=)`, `one(engine=)`, `_check_engine`, and the
item-narrows-the-file guard are machinery for a file **nothing can write**.

**This is the clearest over-engineering finding in the reviewed surface**, and
it joins a family: `RUNTIME_INFO_KEYS` (imported by nothing), `read_by`
(declared on zero fields), `resolver` (declared on zero fields), the `execution`
panel (missing its three machine items) — each a reader built before, or
without, its writer.

> **Not necessarily "delete it".** Multi-engine is the reason `@2` exists. The
> finding is that the capability is **inert and undeclared as such**, so a
> reader of the contract believes a thing the code cannot do. Either a producer
> writes multi-engine files, or the contract says *"one engine per file today;
> the axis is reserved"*.

### 1.3 `select(read_by=…)` returns silently wrong results for a sequence

```
select(t, read_by="wrapper")     -> ['enable_gpu']
select(t, read_by=("wrapper",))  -> []
```

`category` and `kind` both normalise through `_as_tuple`; `read_by` does not —
it is compared with `in` against a tuple of strings, so a one-element tuple
matches nothing. **An empty list, not a refusal**, which is the exact failure
mode the module's own docstrings are written against.

### 1.4 `category` is required by the contract and the reader, but not by the object

`template.md` § 3: four keys are required on every item — `kind`, `category`,
`type`, `help`. `_REQUIRED_ITEM_KEYS` agrees. But:

```python
Item(name="x", kind="wrapper", type="int", help="h")   # constructs fine
# -> category = ()
```

`__post_init__` enforces `anchor`, `expands` and `choices` explicitly *"so a
producer building Items by hand gets the same refusal"* — and skips the one key
§ 3 calls required. The `Item` docstring compounds it, naming the required four
as *"name, kind, type, help"*, which contradicts `_REQUIRED_ITEM_KEYS` 590 lines
below in the same file.

### 1.5 `__all__` omits the module's own headline API

`select` and `one` are what § 8.0 calls **THE ONE READ API**. Neither is in
`__all__`. Nor are the closed vocabularies `CATEGORIES`, `RESOLVERS`,
`ALLOCATION_RESOLVERS` — so a surface that wants to order panels by the closed
six must hard-code them or reach past the declared public surface.

---

---

## 2 · The contract contradicts the code it governs

All four are consequences of T8/T9 landing (2026-08-13) with four *downstream*
docs updated and the **upstream authority** left alone — the worst direction to
miss, since every other doc defers to this one.

| where | says | truth since 2026-08-13 |
|---|---|---|
| § 6.1, the whole `read_by` rationale | the wrapper finds the eigensolver *"by reading the deck text and looking for ELPA"*, and `diag_algorithm` carries `read_by=["wrapper"]` | the scan is deleted; `diag_algorithm` declares no `read_by`; the wrapper reads `enable_gpu` only |
| § 11.3, a worked use case | an ELPA deck must run in `molbuilder-siesta-gpu` | measured false — the packaged SIESTA runs both ELPA stages on CPU |
| § 4.2, the worked example | `diag_algorithm` with `read_by = ["wrapper"]` and help asserting the ELPA→env rule | same |
| `generator.md` § 4 | machine axes (`mpi_np`, `cpus_per_task`) are *"not a template item at all"* | `@2` § 6.4 declares them as **valueless items with resolvers** — T4 |

---

---

## 3 · The contract contradicts itself

| # | where | what |
|---|---|---|
| 3.1 | § 6.3's TOML example | **`category = "accuracy"` appears twice** in `[item.mesh_cutoff]`. TOML forbids duplicate keys, so **the contract's own example does not parse.** |
| 3.2 | § 12's `block_size` example | omits `category`, which § 3 lists as required on every item |
| 3.3 | § 4.2 vs § 6.3 | the same item's value is `300.0` in one example and `300` in the other — int vs float, in a contract whose D3/G4 turn on values being carried faithfully |
| 3.4 | § 5's key diagram | still has the node `group · section` after the table above it marks `section` **RETIRED at `@2`**; the ⭐ note below it reads as current |

---

---

## 4 · Obsolete residue and duplication in `template.py`

| # | what | why it is residue |
|---|---|---|
| 4.1 | `_toml_value`'s `dict` branch | unreachable — no member of `TYPES` produces a dict value. It existed for the retired `ecp` map, and its comment still explains that map |
| 4.2 | `config_from_template`'s comment | claims *"declaration_for returns None for an allocation-tagged field"*. It has not since `@2` — it returns a **valueless item**. The check below it is still right; its stated reason is not |
| 4.3 | `declaration_for` | explains the retired `section` gate **twice** — once in the docstring, once in a comment ten lines down — plus a meta-note about a previous fix to the paragraph itself |
| 4.4 | `Template.engine` | duplicates `engines[0]` as *"the @1 spelling, kept for callers"* — a backward-compatibility shim, against the project's own *rename = delete old everywhere, pre-1.0 break cleanly* rule |
| 4.5 | `one()`'s `KeyError` | prints `engines=` and interpolates `t.engine` (singular) — a message that misreports its own data |

---

---

## 5 · `docs/execution/generator.md` — read in full, 594 lines

**6.1 · § 4's table still holds the `@1` position.** It classes `mpi_np`,
`cpus_per_task` and `gpu_mode` as *"not a template item at all"*. At `@2` they
**are** items — declared valueless, each naming a `resolver` (`template.md`
§ 6.4, landed as T4). Two contracts, two answers, and this one is the doc a
reader consults for *what data exists*.

**6.2 · § 8's deletion table lists a deletion that turned out differently.**
*"the wrapper reading the deck text to find ELPA | `read_by` tells it"*. What
happened on 2026-08-13 is that the READ was deleted outright — the premise
(*only the source build has ELPA*) was measured false — so `read_by` does not
tell the wrapper anything about ELPA; the wrapper reads `enable_gpu` for GPU
runtime facts and routes on that alone. The row's *outcome* is right (the scan
is gone) and its *mechanism* is wrong.

**6.3 · Not a defect, recorded because it is the design's best moment.** § 4.1's
capability ⊇ allocation ⊇ sweep, with the reason attached — *"how a job is
scheduled depends on how much you ask for"* — is the clearest argument in the
execution docs, and it is what makes `_check_fits` refuse rather than clamp.
It is also checkable on ASU Sol (§ 4.4a), which is rare for a design claim.

---

---

## 6 · `molbuilder/resolve.py` — read in full, 559 lines

**This is the best-built module reviewed.** Precedence is implemented in the
order the contract states (template ⊕ stage ⊕ sweep ⊕ pin), every refusal names
the axis and the rule, and `_check_fits` refuses rather than clamps with the
reason in the message. The findings below are boundary issues, not logic ones.

### 7.1 · Floor 3 imports from an engine package — the seam leaks

```python
def _apply(cfg, overrides):
    from .siesta.input import effective_config      # floor 3 -> engine
    from .task import Stage
    return effective_config(cfg, Stage(name="resolve", overrides=dict(overrides)))
```

`generator.md` § 7 states the test: *"adding an engine adds files and edits
none. If a new engine requires a change inside `resolve/`, `materialize` or
`submit`, the seam has leaked and the leak is the bug."* Here the **shared**
floor-3 resolver depends on **SIESTA's** package to resolve a PySCF calculation.

The docstring sees it and defends the wrong half: *"`effective_config` lives
under `siesta/` for historical reasons and its body is entirely
engine-agnostic … calling it here keeps 'the one place this happens' true
rather than growing a second implementation beside it."* Not duplicating it is
right. **The conclusion should be that the one place is in the wrong place** —
an engine-agnostic primitive belongs in a shared module, not imported upward
out of one engine.

Second-order: it also fabricates a `Stage(name="resolve")` to carry overrides
into a function whose parameter is a stage. Using a domain object as a
parameter bag is a smell that would disappear with the move.

### 7.2 · `__all__` omits `resolved_ladder` — the same drift as `template.py`

A public function with a documented production caller, absent from the declared
surface. **Two modules, the same defect** (§ 1.5), which makes it a habit rather
than an oversight: `__all__` is hand-maintained and nothing checks it against
what the module actually offers.

### 7.3 · `_check_fits` is inert when the allocation states no bound

It skips any axis whose allocation value is `None`. That is defensible — an
unstated bound cannot be exceeded — but it means the *"a sweep is bounded by
what you asked for"* guarantee is **silent on a workstation**, which is exactly
where `project-layout.md` M6 says no allocation file is needed. Worth stating in
the contract rather than leaving a reader to infer that the guard always fires.

---

---

## 7 · `docs/execution/job-system.md` — read in full, 1322 lines

### 9.1 · `Resources` is understated in two places and correct in a third

Probed: `Resources` has **nine** fields — `domain`, `time`, `exclusive`, `mem`,
`gres`, `mpi_np`, `cpus_per_task`, `continue_retries`, `max_memory_mb`.

| where | shows | |
|---|---|---|
| § 3's `classDiagram` | **7** | missing `continue_retries`, `max_memory_mb` |
| § 3.1's annotated `job-set.json` | **7** | same two missing |
| § 3.1's annotation, two lines above the example | *"ALL NINE fields are always written"* | correct |

**The example contradicts its own caption two lines apart** — in the section
that opens *"Descriptions of a format are easy to nod along to and hard to
check. Here is an actual … with every field annotated."* The annotation was
corrected at some point (it carries its own correction note about having said
"seven"); the JSON beside it was not.

`WarmFile` (2 fields) and `Job` (5) in the same diagram are correct.

### 9.2 · MISPLACEMENT — a guide re-states a contract, and the copy has drifted

`job-system.md` is **Role: guide**. `job-contracts.md` § 6.2 is the **authority**
for the resource vocabulary, and this document says so explicitly: *"the full
mapping is pinned in `job-contracts.md` § 6.2"*. It then re-states that mapping
inline — field names, their SLURM flags, and which have none.

**§ 9.1 is the proof that this is not harmless duplication:** the restatement is
already wrong by two fields while the authority is right. This is the *"a value
stated twice is a value that drifts"* rule the docs themselves argue (generator
§ 10's preamble), violated by the document that cites it.

The fix is not to sync the copy — it is for the guide to *point* where it now
*repeats*. A guide's job is the shape and the why; the field list is the
contract's.

### 9.3 · Duplication INSIDE the document — one heading, twice

*"How a ladder advances"* appears twice: as a blockquote in § 1 (lines 99–116,
with its own table) and as **§ 5.4** (line 1040). Two passes at one idea in one
document, and a reader who finds the first has no way to know the second is
where the detail lives.

### 9.4 · CLEARED — a prior finding that no longer applies

An earlier review recorded *"job-system § 5.3's table still ⛔-dead"*. **It is
not.** Every cell now reads ✅, with the `bench` column marked LANDED 2026-08-12
and the retired `⛔` state preserved in a dated note below the table. Recorded
here so a consolidation pass does not resurrect a fixed finding — which is its
own kind of drift.


---

---

## 8 · `job-contracts.md` § 6.2 — the authority, read against its own duplicate

This section was read to check § 9.2's drift claim from the authority's side.
What it shows is worse, and better evidenced, than the drift itself.

**The authority is right, is guarded, and records having had the same bug:**

> *"The `jobset.Resources` dataclass holds exactly **nine** fields … (This
> sentence said "exactly seven" while its own table already carried
> `continue_retries` — amended U19, 2026-08-12, **and pinned by an equality test
> in both directions**.)"*

So the identical error — *"seven"* — was found in the contract, fixed, and
**locked with a test**. The same error in `job-system.md`'s copy was not fixed,
because **nothing connects the copy to the source**.

> **This is the argument against the duplication, made by the duplication.**
> A guide that repeats a contract does not merely risk drifting — it drifts
> *past a fix that was already made and tested*, silently, and the reader who
> lands on the guide gets the pre-fix answer with no way to know.

### 10.1 · A stale row in the authority itself

| row | says | truth since 2026-08-13 |
|---|---|---|
| **GPU request** | config-layer source is **`enable_gpu` + `diag_algorithm`**, exchange `gres` → `--gres`, *"derived from `.fdf` + GPU type"* | `diag_algorithm` no longer participates. The GPU ask is derived from `Diag.ELPA.GPU` alone (`_fdf_requests_gpu`, the one scanner left); the solver choice decides no resource and no environment |

Same T8 residue family as § 2 — and this one is in **the cross-layer authority**,
which § 6.3 declares wins over every other document when they disagree. A stale
row here propagates by design.


---

---

## 9 · `job-contracts.md` §§ 3.1–3.2 and 6.3

### 11.1 · VERIFIED CLEAN — recorded because a defect-only report misleads

Two parts of the authority were checked against measurements already in hand
and **agree exactly**:

| checked | against | result |
|---|---|---|
| § 3.1's per-engine block-emission table (PROVENANCE / BENCH-MARKS ✅ SIESTA, ATOM-METADATA conditional, HEADER unemitted) | T10's 57-deck sweep — provenance 57/57, bench-marks 57/57, user-custom 57/57, atom-metadata **6**/57 (only the structures carrying regions/frozen) | **agrees**, including the conditional |
| § 6.3's sweep-token rule — axes in declaration order, concatenated with no separator, `.` spelled `p`, charset `[A-Za-z0-9_]`, refused rather than escaped | `resolve.point_token` + `_TOKEN_RE` read in full | **agrees, clause for clause** |

§ 6.3 is the strongest section reviewed: it states a rule, names the one
function that implements it, and the implementation matches. That is what the
rest of the corpus should look like.

### 11.2 · § 3.2's PROVENANCE example contradicts its own prose, and the emitter

The example block lists a `resolved-defaults` entry:

```
#     kgrid             1x1x1 (auto-from-cell-vacuum)
```

Two paragraphs below, the prose names the fixed set: *"`mpi_np`, `omp_threads`,
`BlockSize`, `enable_gpu` (and the PySCF equivalents)"*. **`kgrid` is not in
it.** And T10's measurement of a real deck shows exactly the four the prose
names, no `kgrid`:

```
#   resolved-defaults:
#     BlockSize    auto -> 1
#     enable_gpu   false
#     mpi_np       auto
#     omp_threads  auto
```

So the example shows a key the prose excludes and the emitter never writes —
the same shape as § 3 of this report: an illustration left behind when the rule
beside it moved.

### 11.3 · `form-config-hash` — reserved, emitted by nobody

The same example carries `form-config-hash  sha256:…  # optional`. No deck in
the 57-deck sweep carries one. It is marked optional, so this is not a
contradiction — it is recorded because it belongs to the family § 8 names:
**a key defined before anything writes it.** HEADER (§ 3.1) is the same shape
and is at least labelled *reserved-but-unemitted*; this one is not.

---

## 10 · Design assessment

**Not over-engineered where it counts.** The two central decisions — one TOML
file with each value stored once, and `prep` rebuilding a config rather than
splicing text — are both argued from failure modes rather than taste, and § 8.1
names three engine behaviours that make splicing impossible. That is design
evidence, not preference.

**The one real over-build is § 1.2** — a multi-engine file format with five
supporting mechanisms and no producer.

**Encapsulation is sound in direction, weak at the boundary.** The floor rule
(a module may import downward, never upward) is stated in `generator.md` § 6 and
holds in `template.py`, which imports only `persist` and the standard library.
What is weak is the *declared* boundary: `__all__` omits the headline API and
the closed vocabularies (§ 1.5), so the module's real interface and its
advertised one differ.

**One structural observation worth more than any single defect.** Five separate
mechanisms in this system have now been found built reader-first, with the
writer never landing. That is a pattern in how the work is sequenced, not five
coincidences — and the cheapest guard against it is the one already used for
`read_by` on 2026-08-13: a test that asserts *someone actually produces this*,
written in the same commit that adds the reader.


---

---

## 11 · The repeating patterns — worth more than any single defect

**Three habits** show up in more than one place, and each is cheaper to fix as
a habit than as its instances:

1. **Reader built, writer never lands.** `engines`, `read_by`, `resolver`,
   `RUNTIME_INFO_KEYS`, the execution panel. The guard that works is the one
   used for `read_by` on 2026-08-13: in the same commit as the reader, a test
   that asserts *something actually produces this*.
2. **`__all__` drifts from the real API.** `template.py` omits `select`/`one`;
   `resolve.py` omits `resolved_ladder`. **Two of two modules read.** A single
   test over the package — *every public callable named in a module's contract
   docstring appears in `__all__`* — would close both and stay closed.
3. **A fix lands at one copy of a fact and not the other.** The clearest case
   is § 8: `Resources` said *"seven"* in the contract and in the guide; the
   contract's was corrected **and pinned with a test**, the guide's was not,
   and nothing connects them. The same shape appears as an illustration left
   behind when the rule beside it moved — § 3's `block_size` example, § 9.2's
   `kgrid` line, § 2's four `read_by` passages. **The rule moves; its examples
   stay.** A doc test that parses fenced TOML/JSON examples and checks them
   against the schema they illustrate would catch most of this family, and
   would have caught § 3.1's duplicate key outright.


---

---

---

## 12 · Cross-boundary: "the five steps" — the prose agrees, the numeral collides

The ownership map named this the highest-risk unchecked restatement. **The
substance is clean.** `project-layout.md` § 2.3.1 owns the sequence — resolve
the machine · resolve the parameters · render the deck · render the wrapper ·
build the run directory — and both restatements agree with it:

| restatement | says | verdict |
|---|---|---|
| `template.md` § 11.1's sequence diagram | step 1 resolve this machine → step 2 template ⊕ overrides → deck writer → wrapper writer | agrees (abbreviated; omits step 5, which is not its subject) |
| `generator.md` § 3's spine | `ParameterSet` → step 3 render deck → step 4 render wrapper → JobSet → *"then floor 4 lays it out"* | agrees, and names the steps by their owner's numbers |

**What is not clean is the numeral itself, in two ways.**

### 12.1 · Two different "fives", one system

`project-layout.md` § 2.3.1: *"the same **five steps** in the same order"* — of
`prep`. `job-system.md` § 5: *"**one verb** on the host and **five** on the
target"* — `prep`, `plan`, `submit`, `summarize`, `status`. Different sets, same
count, adjacent documents, and `generator.md` § 2 leans on `project-layout.md` § 2.3.1a's phrase
*"the five steps are general"* to make its framework/specialisation argument.

Nothing here is factually wrong. But *"the five"* is now ambiguous across the
execution domain, and the two sets are one word apart in the same sentence
shape. **This is the cheapest kind of confusion to remove** — name the steps
*"`prep`'s five steps"* wherever the phrase travels outside § 2.3.1.

### 12.2 · A function that calls itself steps 4–5 and then numbers its own 1–3

`molbuilder/jobset/prep.py`, module docstring:

> *"`prep_calculation` is the five entire … `prep_jobset` is **steps 4–5**
> alone"*

`prep_jobset`'s own docstring, immediately below:

> *"**Steps, in order:** 1. render each distinct `job.script`'s `.run.sh` …
> 2. `materialize` … 3. symlink each job's wrappers"*

A reader tracing *"step 4"* from the contract arrives at a docstring whose first
line is *"1."*. Both numberings are internally consistent and neither is wrong;
they simply collide in one file, ten lines apart. **The canonical five are the
contract's — a local sequence inside one of them should not reuse the word.**

> **Why this belongs in a cross-boundary section rather than as a nitpick.**
> Neither half is visible on its own: reading `project-layout.md` shows one
> five, reading `job-system.md` shows another, and reading `prep.py` shows a
> third numbering of the same work. It is only visible with all three open —
> which is the whole argument for reading this way.

### 12.3 · The reviewer made the same class of error, one commit ago

Writing § 12 I attributed § 2.3.1a to the wrong document. **It belongs to
`project-layout.md`**; `generator.md` § 2 merely quotes it. The docs gate
(`test_every_cross_document_section_citation_resolves`) refused, by name:

> *these citations name a section that does not exist in the target:
> `audit-…md:541 -> <the wrong document> ¶2.3.1a` … Either the section moved
> (repoint the citation) or it was renumbered — do NOT just delete the number,
> which is how a pointer becomes prose nobody can follow.*

*(The target is elided above on purpose: the gate scans prose for citations and
cannot tell a quoted example from a live one, so quoting the failure verbatim
re-triggers it. A small cost of a textual gate, and worth knowing before
someone documents a citation failure and is refused for describing it.)*

Recorded because it is evidence for two claims this report makes, and evidence
against a third:

- **For § 11's pattern 3** — I attributed a section to the document that *cites*
  it rather than the one that *owns* it. That is the identical confusion the
  ownership map (§ 0.1) exists to prevent, made by the person writing the
  ownership map, ninety minutes later.
- **For the value of executable gates.** No amount of care caught it; a test
  did, immediately, with the fix named. Every finding in §§ 1–9 that a gate
  would have caught is a finding that should never have needed a reviewer.
- **Against treating this corpus as careless.** These documents cite each other
  hundreds of times and the gate passes. One bad pointer in a fresh 550-line
  file is the base rate; the drifted *content* in §§ 7–9 is what is unusual,
  and it is unusual precisely because **content has no equivalent gate**.

> **The actionable form:** the citation gate proves the pattern works. The
> §§ 7–9 findings all describe facts that were restated and drifted — and a
> restatement is exactly as checkable as a citation. A test that parses fenced
> examples against the schema they illustrate, and one that compares a doc's
> enumerated field list against the dataclass, would close that family the way
> the citation gate closed this one.

---

## 13 · Cross-boundary: *absent* vs *null* — one artifact, two answers

`molbuilder/jobset/model.py` holds both classes that serialise into
`job-set.json`, forty lines apart, and they take **opposite positions on the
same question**: how does an unset optional field appear on disk?

| class | `to_dict` | so an unset field is |
|---|---|---|
| `WarmFile` | omits `requires_same` when falsy, **with the reason attached** | **absent** |
| `Resources` | `dataclasses.asdict(self)` — every field, always | **`null`** |

`WarmFile`'s rationale is argued, and it argues against its neighbour:

> *"ABSENT, not null, when the file is unconditional — the same reading
> `checkpointing.md` S3 asks for elsewhere: **a key that is missing and a key
> that is null are different claims** to anything testing for it."*

**The same position is taken in two more places**, which is what makes this a
boundary issue rather than a style quibble:

| where | says |
|---|---|
| `engines/template.md` § 3 | *"a missing `value` means **explicitly unset*** — a real state, distinct from the default" |
| `molbuilder/template.py` `_item_payload` | *"An absent `value` is the encoding of explicitly unset (§ 3)"* — omits the key |
| `execution/checkpointing.md` S3 | cited by `WarmFile` as the origin of the reading |

So **three contracts and two implementations** hold that *absent ≠ null and the
difference is load-bearing* — and `Resources`, serialising into the same file,
writes nulls for all nine fields. `job-system.md` § 3.1's example shows exactly
that (`"domain": null, "exclusive": null, "mem": null, "gres": null`) and
annotates it *"nulls included — see the note below"*.

**This is not asserted as a defect.** `Resources` may be right to be explicit: a
reader of `job-set.json` can see all nine asks exist and which were left to
resolve, and the note below that example presumably argues it *(unread — the
note is in the part of `job-system.md` § 3.1 not yet quoted here)*.

**What is a defect is that neither side knows about the other.** Two
serialisation policies meet in one file, each with its own justification, and
no document names the disagreement or says which applies where. The decision to
make is one sentence in `job-contracts.md` § 6.1 (which owns persisted
artifacts): *an unset optional is written as `null` in `Resources` because the
reader needs the full ask visible, and omitted elsewhere because absence is a
distinct state* — or the opposite. Either is fine; **silence is what leaves the
next writer to pick by coin-flip.**

> **Core document: `execution/job-contracts.md` § 6.1.** Look beyond
> `model.py`: the same question is answered by `template.py`'s `_item_payload`,
> by `WarmFile.to_dict`, and by whatever writes `run.json` and
> `bench-result.json` — none of which were compared against each other.

---

## 14 · `project-layout.md` § 7's invariants, cross-checked against everything read earlier

**This is the check the reading order made possible**: § 7 states seventeen
invariants, and four of them are about concepts whose *owners and
implementations* were read earlier in this pass. Each is checked against those,
not against itself.

| # | invariant | cross-checked against | result |
|---|---|---|---|
| **9** | *"The description is the only source at ③. No produce and no run modifies it."* | `engines/template.md` § 2's dotted arrow — *"nothing a machine produced ever edits the description"* | **agree** — two documents, two vocabularies, one rule |
| **3** | engine-named files (`<label>.XV/.DM/.CG`) are **bare**; molbuilder-named files carry `<label>_<NN>_<stage>` | `job-contracts.md` § 6.3's file table (read in § 9) | **agree, row for row** |
| **10** | a calculation folder carries no machine knowledge; `environment.json` is the deliberate exception | `template.md` § 2 (floor 2 names no machine) · `generator.md` § 4.3 (neither sweep nor allocation enters the description) | **agree** — and § 10's *exception* is what § 6.4's valueless items are the template-side half of |
| **5** | *"A trial never shares the calculation's identity. Its deck is **relabelled** and **forced cold**."* | `resolve.py` `_label_for` (read in § 6) | **half-verified — see below** |

### 14.1 · Invariant 5 has two clauses and I could locate only one

`_label_for` implements the relabel exactly as the invariant states — a
production run keeps the label, a trial becomes `<label>-<point_token>`, with
the reason attached (*SIESTA finds warm files by `SystemLabel`, so a trial
carrying the real label could read or overwrite the run's `.DM` and `.XV`*).

**The "forced cold" half is not in the resolver.** Nothing in `resolve.py` sets
`restart` for a trial; a sweep point's values come from the point and the pins,
and `restart` is an ordinary template item that a trial inherits from the
description like any other. So either something downstream forces it, or the
invariant's second clause is unimplemented and the first is doing all the work.

> **Not asserted as a defect — asserted as unlocated.** The relabel alone
> already prevents the failure the invariant exists for, so a trial cannot
> *find* the run's warm files even if it asks for them. That makes "forced
> cold" a belt beside a brace, and an unimplemented belt is easy to miss
> precisely because nothing fails.
>
> **Core document: `execution/project-layout.md` § 3.2** (which owns what a
> trial's deck is). What to look for: whoever renders a trial's deck — trace
> `restart` from the description through `resolve` to the deck writer, and
> either name the enforcing line in § 3.2 or drop the clause.

### 14.2 · Two "invariants" are labelled *not held today*

Invariants **16** (the archive covers only runs this calculation owns — *"the
walk classifies by pattern and archives a trial's `.DM` like any other"*) and
**17** (a save stores only what is new — *"every save copies every big file"*)
both carry **Not held today**.

The honesty is right and rare. The **naming** is what to fix: § 7 opens *"Each
is written so a test can assert it"*, and a reader who asserts 16 or 17 gets a
red suite for behaviour the document already knows about. An invariant that is
known-false is a **requirement**; separating the two lists costs one heading
and stops the next contributor discovering it the expensive way.

### 14.3 · One invariant cites a test, and the citation holds

Invariant **6a** — *"Every directory and every link in this tree is made by
Python… Test: render a wrapper for each engine and assert its text contains no
`cd` command (`tests/test_warm_file_inventory.py`)"*. The file exists and its
wrapper tests pass.

**This is the shape §§ 11 and 12.3 argue for**, found in the wild: an invariant
that names the test that holds it. Of seventeen, it is the only one that does.
The distance between invariant 6a and invariants 16–17 is the distance between
a rule and a wish, and it is visible only because 6a wrote its test down.

---

## 15 · `task.py` × `template.md` § 1 — the description's other half

`template.md` § 1 states the split: the template holds *"every parameter, with
a value — **what the calculation is**"*; `task.json` holds *"which parameters
vary, and each stage's overrides — **what the mission is**"*, and **"they do not
overlap"**. Checked against the shipped model:

```
Task:  engine · shape · run · structure · varies · stages · schema_fingerprint · calculation
Stage: name · enabled · overrides
```

**The split holds.** `varies` and `stages[].overrides` are exactly the two
things § 1 assigns to `task.json`, `schema_fingerprint` is § 10's claim-carrier,
and `shape` belongs to `project-layout.md`. The apparent overlap —
`Stage.overrides` carries parameter *values* — is the one § 1 already licenses
(*"a stage's override replaces an item's value"*), so it is by design.

### 15.1 · `Task.engine` is singular — finding 1.2, confirmed from the other side

`Task` carries **`engine: str`**, not a list.

Finding 1.2 established that the multi-engine template has no producer: zero
fields declare `engines`, and `render_template` hard-codes a one-element list.
This is the same claim reached from a different file: **the description's other
half cannot express a multi-engine calculation either.**

That changes what 1.2 *is*. It is not one unimplemented writer — it is a
capability `engines/template.md` § 6.3 asserts and **two shipped models
independently contradict**, in the two files that together *are* the
description. A calculation that "runs on more than one engine" cannot say so in
`task.json`, cannot be written by `render_template`, and has no field anywhere
to hang the second engine's items on.

> **Core document: `engines/template.md` § 6.3, and now also
> `engines/stages.md` § 6** (which owns `task.json`). What to look for: the
> decision is one sentence, but it lands in two contracts. If multi-engine is
> real, `Task.engine` becomes a list on the same day `render_template` learns
> to emit one; if it is reserved, both contracts say so and `select(engine=)`
> keeps working for the single-engine case it actually serves.

---

## 16 · Closing this pass — what remains, and why it needs a fresh reading

**Read in full, front to back:** `engines/template.md` · `molbuilder/template.py`
· `execution/generator.md` · `molbuilder/resolve.py` · `execution/job-system.md`
· `execution/project-layout.md` § 7 · `job-contracts.md` §§ 3.1–3.2, 6.2, 6.3 ·
`jobset/model.py`'s serialisation surface · `task.py`'s model — **~5,400 lines**,
with every finding cross-checked against the files read before it.

**Not read** (~17,000 lines): the rest of `job-contracts.md` and
`project-layout.md`, `checkpointing.md`, `architecture.md`, `run-identity.md`,
`running-a-job.md`, `staged-runs-implementation-plan.md`, the rest of
`jobset/`, and the test suites.

**Why this pass stops here rather than continuing.** The value of §§ 12–15 came
entirely from holding the earlier files in memory while reading the later ones —
invariant 5 is half-verified only because `resolve.py` was still in mind;
finding 1.2 doubled in weight only because `template.py` was. Reading the
remaining 17,000 lines in this same session would displace exactly the material
that makes those checks possible, and the result would be a longer list of
*local* observations — which is the kind of review § 0 exists to argue against.

**The next pass, ordered by what it can cross-check:**

| # | read | because it cross-checks against |
|---|---|---|
| 1 | `execution/project-layout.md` §§ 1–2 (the shapes, `prep` as hub) | § 7's invariants, already checked here — the rules are read, the mechanisms are not |
| 2 | `molbuilder/jobset/prep.py` + `materialize.py` | the five steps (§ 12), invariant 5's missing "forced cold" (§ 14.1), and `Resources`' nulls (§ 13) |
| 3 | `execution/running-a-job.md` § 3 | the wrapper chain that `template.md` § 6.4's `threads` resolver and `runwrap`'s scanners both point at |
| 4 | `execution/checkpointing.md` S1–S4 | invariants 15–17, and the *absent vs null* rule § 13 traces to S3 |
| 5 | `engines/stages.md` §§ 4, 6 | `_apply`'s seam (§ 6.1), `Task.engine` (§ 15.1), and the fingerprint |
| 6 | the test suites | every "would a gate have caught this" claim in § 11 |

**One thing to do before that pass, because it changes what it finds:** the two
gates § 11 proposes — fenced examples validated against the schema they
illustrate, and a doc's enumerated field list compared against the dataclass —
would resolve §§ 3, 7.1, 9.2 and part of 2 *mechanically*, and would keep them
resolved. Running them first turns roughly a third of this report into a test
run instead of a reading task.

---

## 17 · `jobset/prep.py` — two open findings resolved

### 17.1 · Invariant 5's *"forced cold"* is CONFIRMED unimplemented (closes § 14.1)

§ 14.1 recorded that `resolve.py` relabels a trial but never forces it cold, and
left the second clause *unlocated* pending the deck-rendering path. That path is
`prep_calculation` step 3, and it does this:

```python
if element.is_trial:
    # The deck's OWN identity line carries the trial label -- this,
    # not the filename, is what keys SIESTA's warm files away from
    # the real run's (project-layout.md § 2.3.2).
    cfg = seam.relabel(cfg, element.label)
```

So a trial is relabelled **twice, deliberately** — `resolve._label_for` sets the
element's label, and `prep` writes it into the deck's identity line — and
**nothing anywhere sets `restart` for a trial.** It inherits the description's
value like any other item.

**The code's own comment settles the design question:** the protection is
attributed entirely to the relabel — *"this, not the filename, is what keys
SIESTA's warm files away from the real run's"*. There is no second mechanism,
and the comment does not think there is one.

> **So `project-layout.md` § 7 invariant 5 overstates its own implementation.**
> A trial cannot reach the run's warm files because it is looking for a
> different `SystemLabel` — which is sufficient. *"and forced cold"* describes a
> belt that was never fitted.
>
> **Core document: `execution/project-layout.md` § 7 (invariant 5) and § 3.2.**
> The fix is one clause: either delete *"and forced cold"*, or implement it and
> say where. Deleting is the honest default — the invariant is true and
> load-bearing without it, and a rule listing a mechanism nobody wrote is how a
> reader concludes the protection is doubled when it is single.

### 17.2 · The five-steps numbering collision is narrower than § 12.2 said

`prep_calculation` numbers its own sections **1 · 2 · 3 · "4 + 5"** — the
canonical five, in the owner's numbers, matching `project-layout.md` § 2.3.1
exactly. **It gets it right.**

The collision § 12.2 reports is therefore specific to **`prep_jobset`**, which
the module docstring calls *"steps 4–5 alone"* and which then numbers its own
internals 1 · 2 · 3. One function in the file follows the contract's numbering
and the other re-numbers inside it.

That makes the fix smaller and more obvious than § 12.2 implied: `prep_jobset`'s
three internal steps want naming rather than numbering (*render the launchers ·
materialize · link them in*), leaving the digits to mean one thing in this file.

---

## 18 · Fix-ready detail — the exact text, for the findings recorded as table rows

**Why this section exists.** §§ 3 and 4 are tables, and a table row is enough to
*remember* a finding and not enough to *act* on one — whoever picks it up would
re-read a thousand lines to find the string. Each entry below carries the exact
text as it stands, what it should say, and the check that proves the fix.
Everything here was read this pass; nothing needs re-deriving.

### 18.1 · `engines/template.md` § 6.3 — the example that does not parse (§ 3.1)

**As it stands**, inside the `[item.mesh_cutoff]` block of § 6.3's TOML example:

```toml
[item.mesh_cutoff]
kind     = "engine"
category = "accuracy"
category = "accuracy"        # <- the duplicate; TOML forbids it
engines  = ["siesta"]
anchor   = "MeshCutoff"
value    = 300
unit     = "Ry"
```

**Fix:** delete one `category` line. **Also fix in the same block:** `value =
300` is an int, while § 4.2's example of the *same item* writes `300.0`
(§ 3.3) — make it `300.0`, since the field is a float and `_shape` does not
widen ints for a `float` type.

**Check:** `tomllib.loads()` the fenced block. It currently raises; it should
parse, and `parsed["item"]["mesh_cutoff"]["value"]` should be a `float`.

### 18.2 · `engines/template.md` § 12 — the `block_size` example (§ 3.2)

**As it stands**, the example carries `kind`, `anchor`, `type`, `range`,
`group`, `help` — and **no `category`**, which § 3 lists among the four keys
required on every item.

**Fix:** add `category = "execution"` — the value § 6.2's own table assigns to
`block_size` in its SIESTA column.

**Check:** the same fenced-example test as 18.1, asserting `_REQUIRED_ITEM_KEYS`
⊆ the parsed keys.

### 18.3 · `engines/template.md` § 5 — `section` residue (§ 3.4)

Two places, after the key table already marks `section` **RETIRED at `@2`**:

1. the mermaid node `G["group · section"]` — should be `G["group"]`;
2. the ⭐ note beginning *"`label`, `section` and `null_label` were added
   2026-08-11"* — historically true, reads as current. Suggest: *"…`label`,
   `category` (then `section`) and `null_label`…"*, or leave the history and
   add *"(`section` was retired at `@2` — § 6.2)"*.

> **⚠ Do not sweep `section` globally.** It is retired in the **template** and
> **live in the FORM** (`web/form-schema.md` § 1a drives `dataclass_to_form_
> schema` off it, and every config field still carries one). Same word, two
> mechanisms — § 0's index flags this and it is the easiest thing here to get
> wrong.

### 18.4 · `molbuilder/template.py` — the residue list (§ 4), with exact anchors

| what | the text as it stands | what to do |
|---|---|---|
| dead `dict` branch | in `_toml_value`: `if isinstance(v, dict):` with the comment *"An inline table … Keys are element symbols"* | delete both. No member of `TYPES` yields a dict since `strmap` retired; the comment describes the retired `ecp` map |
| stale claim | in `config_from_template`: *"The WRITE side never emits these (`declaration_for` returns None for an allocation-tagged field)"* | false since `@2` — it returns a **valueless item**. The check below it is still correct; rewrite the reason as *"the write side emits these valueless, so a VALUE here is a hand edit"* |
| doubled explanation | `declaration_for`'s docstring paragraph beginning *"`None` means excluded by § 7's named rows … never by a missing `section`"*, plus the comment ten lines below beginning *"NO section gate (U16, 2026-08-12)"* | keep one. The comment is the better home (it sits where the gate was); the docstring should say only what `None` means |
| compat shim | `Template.engine: str` with *"engines[0] — the @1 spelling, kept for callers"* | governed by the project's *rename = delete old everywhere, pre-1.0 break cleanly* rule, **not** by the template contract. Removing it means updating `one()`'s KeyError message and any caller reading `t.engine` |
| wrong field in a message | `one()`'s `KeyError`: `f"no item {name!r} in this template (engines={t.engine!r})"` | says `engines=`, interpolates the singular. Either `t.engines` or relabel to `engine=` |

### 18.5 · The two gates § 11 proposes, specified enough to write

**Gate A — fenced examples parse and satisfy their own schema.** Walk every
```` ```toml ```` block in `engines/template.md`; `tomllib.loads` it; for each
`[item.*]` table assert `_REQUIRED_ITEM_KEYS` ⊆ its keys and `type` ∈ `TYPES`.
Catches 18.1, 18.2, and anything of that shape added later. *(Scope it to
`template.md` first — other docs' fenced TOML is not all templates.)*

**Gate B — an enumerated field list in prose matches the dataclass.** The
instances found: `job-contracts.md` § 6.2 names the nine `Resources` fields
(already tested — this is the model), `job-system.md` § 3's class diagram and
§ 3.1's example do not. A general form is hard; a specific one is not — assert
that every field of `Resources` appears in `job-system.md` § 3.1's example
block, which is the copy that drifted.

> **Order matters:** Gate A and B resolve §§ 3, 7.1, 9.2 and part of § 2
> mechanically. Running them before the next reading pass turns roughly a third
> of this report into a test run.

---

## 19 · `materialize.write_run_launch` — § 13's inconsistency, now inside one dict literal

§ 13 found `WarmFile` and `Resources` disagreeing about *absent vs null* forty
lines apart. `run.json`'s writer does it in **one body**:

```python
body = {
    "schema": RUN_LAUNCH_SCHEMA,
    "mode": mode,
    "command": list(command),
    "job_id": job_id,                 # None -> written as  "job_id": null
    "launched_at": ...,
}
# ABSENT, not null, when this run started from the structure.
# ``checkpointing.md`` S3 words its test that way -- *"names a directory
# that exists or is absent"* -- and the two are different to a reader that
# tests for the key rather than for its truthiness.
if continued_from:
    body["continued_from"] = str(continued_from)
```

**`job_id` is written as `null` when there is none; `continued_from` vanishes** —
and the comment arguing that absence is the meaningful encoding sits **three
lines below** the null it does not apply to.

The distinction the comment draws is not special to `continued_from`. A
`--mode direct` launch has no scheduler id, so `job_id` is exactly as
"unset" as `continued_from` is for a run started from the structure, and a
reader testing `"job_id" in body` gets a different answer than one testing its
truthiness — which is the comment's own criterion.

> **This raises § 13 from a style inconsistency to a real one.** Three shipped
> writers now: `Resources.to_dict` (all nulls), `WarmFile.to_dict` (omit),
> `write_run_launch` (**both, in one object**). No document states a policy, so
> each writer picked, and one picked twice.
>
> **Core document: `execution/job-contracts.md` § 6.1** — it owns the persisted
> artifacts and already lists all three of these files in one table, which is
> the natural place for one sentence settling it. **What to look for beyond
> these three:** `bench-result.json` and `environment.json` answer the same
> question and were not read this pass.
>
> **The cheap resolution, if a rule is wanted rather than a survey:** the
> project already has one, stated twice and implemented three times — *absence
> is a distinct state, and a key that is missing and a key that is null are
> different claims* (`engines/template.md` § 3, `checkpointing.md` S3,
> `template.py`'s `_item_payload`, `WarmFile.to_dict`). `Resources`' all-nulls
> is the deliberate exception and can stay one, **named as such**.

---

## 20 · The full *absent vs null* tally — and a correction to §§ 13 and 19

Read the two remaining writers § 19 named as open. **They change the
conclusion**, so the earlier framing is corrected here rather than left to be
discovered.

| writer | artifact | policy |
|---|---|---|
| `Resources.to_dict` | `job-set.json` | `dataclasses.asdict` — **all nulls** |
| `BenchResult.to_dict` | `bench-result.json` | every key always present; `asdict(p)` per point — **all nulls** |
| `Environment.to_dict` | `environment.json` | `asdict(topology)`, `asdict(site)` — **all nulls** |
| `WarmFile.to_dict` | inside `job-set.json` | **omits** an unset `requires_same`, with the reason |
| `template._item_payload` | `<label>.template.toml` | **omits** an unset `value`, with the reason |
| `write_run_launch` | `run.json` | **both**, in one dict literal (§ 19) |

### 20.1 · What I got wrong in §§ 13 and 19

Both sections framed all-nulls as *"the deliberate exception"* to a documented
rule. **The tally says the opposite: all-nulls is the majority practice — three
artifacts of six — and the documented rule is the minority.**

The rule is real and argued in two contracts (`engines/template.md` § 3,
`checkpointing.md` S3) and honoured by two writers plus half of a third. But it
is not what most persisted artifacts do, and describing `Resources` as *an
exception* implied a consensus that does not exist.

> **Why the correction matters for the fix, not just for accuracy.** Under the
> old framing the action was small: name one exception in a sentence. Under the
> real tally, `job-contracts.md` § 6.1 has to decide *which* policy its registry
> requires — and if it chooses the documented one, three shipped writers change
> and their readers with them. If it chooses `asdict`, then `template.md` § 3's
> *"a missing `value` means explicitly unset"* is a template-only rule and
> should say so, because that file's whole three-state model (**unset** vs
> **default** vs **absent from the file**) depends on it.
>
> **That is a real design decision, not a cleanup** — which is what the earlier
> framing would have hidden. Recorded here so whoever picks it up starts from
> six writers and a choice, rather than from one exception and a tidy-up.

### 20.2 · The template's three states are the strongest argument on the rule's side

`engines/template.md` § 3 needs absence to mean something because it
distinguishes **three** states that `null` cannot: *unset* (key absent),
*at its default* (`value` present and equal to `default`), and *not a parameter
of this calculation* (item absent). A writer using `asdict` collapses the first
into a null and loses the distinction the contract's § 6.4 resolver story is
built on.

**No other artifact here needs three states**, which is very likely why the
practice diverged — and is also the shape of the answer: the rule belongs to
files that carry *declarations*, and `asdict` suits files that carry *records*.
Both halves are then defensible and, more importantly, **statable in one
sentence each**.

---

## 21 · CROSS-INDEX — read this first when resuming

**What this section is for.** The findings above survive; what does not survive
a context reset is *knowing what each document asserts*, which is what makes it
possible to check a **not-yet-read** document against the ones already read.
This is that knowledge, compressed: closed vocabularies with their members,
load-bearing rules with their owners, and measured facts that are expensive to
re-derive. **Check a new document against this table, not against memory.**

### 21.1 · Closed vocabularies — a new document using one of these can be checked against it

| vocabulary | members | owner | code |
|---|---|---|---|
| item `kind` | `engine` · `deck` · `wrapper` · `produce` · `monitor` — **5** | `template.md` § 6 | `template.KINDS` |
| `category` | `system` · `method` · `accuracy` · `convergence` · `procedure` · `execution` — **6, in reading order**; a LIST per item, first = the panel | `template.md` § 6.2 | `template.CATEGORIES` |
| item `type` | `int` `float` `str` `bool` `enum` `pow2` `int3` `strlist` `intlist` `text` — **10** (`strmap` **retired** 2026-08-13) | `template.md` § 5 | `template.TYPES` |
| `resolver` | `rank_count` · `omp_threads` · `node_memory` · `block_size` — **4** | `template.md` § 6.4 | `template.RESOLVERS` |
| allocation resolvers *(item may never carry a value)* | `rank_count` · `omp_threads` · `node_memory` — **3** (`block_size` deliberately absent) | `template.md` § 6.4 | `template.ALLOCATION_RESOLVERS` |
| template top-level keys | `schema` · `engines` · `fingerprint` — **3, all required** | `template.md` § 3 | `read_template` |
| required item keys | `kind` · `category` · `type` · `help` — **4** | `template.md` § 3 | `_REQUIRED_ITEM_KEYS` |
| conditionally required | `anchor` (kind=engine) · `expands` (kind=deck) · `choices` (type=enum) · `value` (when answered) · `resolver` (when valueless-by-design) | `template.md` § 3 | `Item.__post_init__` |
| reserved script blocks | HEADER *(reserved, emitted by nobody)* · PROVENANCE · BENCH-MARKS · ATOM-METADATA · USER-CUSTOM — **5**; order is **physics first**, record behind a banner | `job-contracts.md` § 3.1 | `script_emit` |
| `Resources` | `domain` `time` `exclusive` `mem` `gres` `mpi_np` `cpus_per_task` `continue_retries` `max_memory_mb` — **9**; last two become **no SLURM flag** | `job-contracts.md` § 6.2 | `jobset.model.Resources` |
| `Job` | `name` `script` `resources` `warm` `traits` — **5, no edges** | `job-system.md` § 3 | `jobset.model.Job` |
| `WarmFile` | `name` · `requires_same` — **2** | `job-contracts.md` § 4.2 | `jobset.model.WarmFile` |
| `Task` | `engine` `shape` `run` `structure` `varies` `stages` `schema_fingerprint` `calculation` — **8**; `engine` is a **string** | `stages.md` § 6 | `task.Task` |
| `Stage` | `name` · `enabled` · `overrides` — **3** | `stages.md` § 4 | `task.Stage` |
| the four separators | `_` joins parts of one name · `-` attaches a counter/qualifier · `.` introduces a type suffix · `/` separates path levels | `job-contracts.md` § 6.3 | — |
| floors | 1 names+machine · 2 description · 3 plan · 4 layout · 5 launch · 6 observe · 7 surfaces — **imports go DOWN only** | `generator.md` § 6 | module layout |
| `prep`'s steps | resolve the machine · resolve the parameters · render the deck · render the wrapper · build the run directory — **5, order forced** | `project-layout.md` § 2.3.1 | `prep_calculation` |

### 21.2 · Load-bearing rules — the assertions to check a new document against

| # | rule | owner | verified against |
|---|---|---|---|
| R1 | **Floor 2 names no machine.** The template declares the QUESTION and never asserts the ANSWER; a machine fact's *item* may exist, valueless, with a resolver | `template.md` §§ 2, 6.4, 7 | `declaration_for`, `read_template`'s allocation guard, `template_fields` |
| R2 | **A run is a sweep of length one.** `ParameterSet` length is the whole difference; **no `if benchmark:` below floor 7** | `generator.md` § 2 | `resolve.ParameterSet`, `_points` returns `({},)` |
| R3 | **Precedence is total:** template ⊕ stage overrides ⊕ sweep point ⊕ pin | `generator.md` § 5 | `resolve.resolve` — implemented in that order |
| R4 | **capability ⊇ allocation ⊇ sweep**, and a sweep point exceeding the allocation is **refused, not clamped**, and never checked against capability | `generator.md` § 4.1 | `resolve._check_fits` *(inert when the allocation states no bound — § 6.3)* |
| R5 | **Stages do not chain.** No `Job` names another; a person preps and submits each; what a stage continues from is a **file copied in at `prep`**, from a run named with `--from` | `project-layout.md` § 1.6 · `job-system.md` § 2 decision 6 | `jobset.model` has no edge field |
| R6 | **`prep` rebuilds and renders; it never splices** — three engine behaviours make text substitution impossible | `template.md` § 8.1 (D4) | `config_from_template` → emitter |
| R7 | **Each value is stored once** (D3); **membership is total** (D5) | `template.md` § 1.2 | `_item_payload`, `declaration_for` |
| R8 | **Who names a file decides whether it carries the stage.** Engine-named (`.XV/.DM/.CG`) are **bare**; molbuilder-named carry `<label>_<NN>_<stage>` | `job-contracts.md` § 6.3 | `project-layout.md` § 7 invariant 3 — agree |
| R9 | **The description is the only source at ③;** nothing a machine produced edits it | `project-layout.md` § 7 inv. 9 | `template.md` § 2's *never flows back* — agree |
| R10 | **A trial is relabelled** so it cannot reach the run's warm files | `project-layout.md` § 3.2 | `resolve._label_for` + `prep`'s `seam.relabel` — **the "and forced cold" half is unimplemented, § 17.1** |
| R11 | **The wrapper's env is decided by GPU alone** *(since 2026-08-13)*; CPU-ELPA runs in the packaged env | `engines/siesta.md` § 7.2 | measured — see 21.3 |
| R12 | **A sweep coordinate is one qualifier:** axes in declaration order, concatenated, no inner separator, `.`→`p`, charset `[A-Za-z0-9_]`, **refused not escaped** | `job-contracts.md` § 6.3 | `resolve.point_token` — agrees clause for clause |
| R13 | **The engine seam is a plugin:** adding an engine adds files and edits none. A change inside `resolve/`, `materialize` or `submit` means the seam leaked | `generator.md` § 7 | **violated** — `resolve._apply` imports from `siesta/` (§ 6.1) |
| R14 | **Absence vs null is UNSETTLED** — 3 writers all-nulls, 2 omit, 1 both | *(no owner — that is the finding)* | § 20's tally |

### 21.3 · Measured facts — expensive to re-derive, so do not

| fact | measurement |
|---|---|
| **The packaged `molbuilder-siesta` runs ELPA on CPU** | H2 probe: `ELPA-2stage` and `ELPA-1stage` both exit 0 at **E = −30.136019 eV**, identical to `Divide-and-Conquer`. `ELPA-2stage` + `Diag.ELPA.GPU .true.` **exits 1** with `ELPA_ERROR_ENTRY_NOT_FOUND`. ELPA is compiled in via ELSI — 279 defined symbols, **zero undefined**, no external `libelpa` |
| **Only GPU needs the source build** | which is why the two envs split on **provenance** (packaged-anywhere vs must-compile), not hardware |
| **The `@2` refactor did not move the deck** | 57 decks (19 configs × 3 stage tokens, all 44 SIESTA fields off-default somewhere) byte-identical across `2e715088`→HEAD; harness mutation-tested |
| **`engines` metadata: 0 fields**, both configs; `render_template` emits a one-element list | the multi-engine axis has no producer (§ 1.2), and `Task.engine` is a string (§ 15.1) |
| **`Resources` has 9 fields** | probed; `job-contracts.md` § 6.2 correct and tested, `job-system.md` § 3 shows 7 |
| **gcc 14.4 miscompiles SIESTA 5.4.2's `kpoint_t.F90`** | 14.3 compiles it; verified end-to-end on a clean machine 2026-08-14. Pin is keyed by SIESTA tag with a gate |

### 21.4 · How to use this on the next document

1. Read the new document.
2. For every **enumeration** it gives, find the row in 21.1 and compare counts and members. *(That check alone found §§ 7.1 and 9.1.)*
3. For every **rule** it states, find the row in 21.2 and compare wording — a restatement that answers the **same question in fewer words** is the drift signature (§ 0.1).
4. For every **claim about behaviour**, check 21.3 before believing it. Three findings so far were documents asserting something the code does not do.
5. Add what the new document **owns** to 21.1/21.2, so the index grows as the reading does.

---

## 22 · Proposal — how to keep validating holistically across context resets

§ 21 is a snapshot. It goes stale the moment someone edits a document, and a
stale index is worse than none because it is trusted. Three steps, smallest
first, each useful alone.

### 22.1 · Make § 21.1 executable — the one that pays for itself

Every row of the vocabulary table is a **set of strings that must agree in three
places**: the code's constant, the contract that owns it, and any document that
enumerates it. That is a test, and the data is already collected.

```python
# tests/test_doc_claims.py  — sketch, not final
VOCABULARIES = {
    "kind":       (template.KINDS,      "engines/template.md"),
    "category":   (template.CATEGORIES, "engines/template.md"),
    "type":       (template.TYPES,      "engines/template.md"),
    "resolver":   (template.RESOLVERS,  "engines/template.md"),
    "Resources":  ([f.name for f in fields(Resources)],
                   "execution/job-contracts.md"),
}
# for each: every member appears in the owning document, and the document
# names no member the code does not have.
```

**What it would have caught in this review, mechanically:** § 7.1 (`Resources`
7 vs 9, in two places), § 9.1's family, and any future retirement like
`strmap`'s that leaves a doc naming a type the code dropped. **What it cannot
catch:** prose rules — §§ 2, 6.1, 17.1 needed reading. That is the honest split,
and it is roughly a third mechanical, two thirds not.

> Pair it with **Gate A** (§ 18.5 — fenced TOML examples parse and satisfy
> `_REQUIRED_ITEM_KEYS`). Together they cover §§ 3, 7.1, 9.2 and part of 2.

### 22.2 · A reading ledger, so a reset resumes instead of restarting

The coverage table says *what* was read. What a resumed session also needs is
**what each file was checked against** — otherwise the same pair gets compared
twice and a different pair never does.

| file | read | checked against | still to check against |
|---|---|---|---|
| `template.md` | 2026-08-14 | `template.py`, `generator.md`, `task.py` | `stages.md`, `form-schema.md` |
| `template.py` | 2026-08-14 | `template.md`, `resolve.py` | `cli.py`'s option bridge *(shares the annotation-walking idiom — § 1.1)* |
| `generator.md` | 2026-08-14 | `template.md`, `resolve.py`, `project-layout.md` § 7 | `architecture.md` § 0's axes |
| `resolve.py` | 2026-08-14 | `generator.md`, `template.py`, `prep.py` | `stages.md` § 4 (`effective_config`'s home) |
| `job-system.md` | 2026-08-14 | `job-contracts.md` § 6.2, `model.py` | `running-a-job.md` |
| `job-contracts.md` §§ 3.1–3.2, 6.2, 6.3 | 2026-08-14 | `job-system.md`, `resolve.py`, T10's decks | its own §§ 2, 4 |
| `project-layout.md` § 7 | 2026-08-14 | `template.md`, `job-contracts.md`, `resolve.py`, `prep.py` | `checkpointing.md` S1–S4 (invariants 15–17) |
| `model.py` · `prep.py` · `materialize.py` · `task.py` · `bench/result.py` · `environment.py` | 2026-08-14 | each other, on serialisation | `submit.py`, `runstatus.py` |

**The last column is the work queue**, and it is more useful than a list of
unread files because it names *pairs* — which is where the findings came from.

### 22.3 · The rule that makes the index maintainable

An index nobody updates rots. The cheap enforcement is the one this corpus
already uses for citations: **when a document's enumeration changes, the test in
22.1 fails and names it.** So 22.1 is not only a defect-catcher — it is what
keeps § 21.1 honest without anyone remembering to.

For 21.2's prose rules there is no such gate, and I would not invent one. The
maintainable form is what § 0.1 found: **a guide may say why; the contract says
what, and the guide must not enumerate it.** Applied as an editing rule, the
number of places a rule is stated stops growing, and the index stops needing to
track restatements it cannot check.

> **If only one of these three happens, make it 22.1.** It converts the part of
> this review that was mechanical into something that never needs a reviewer
> again, and it is perhaps eighty lines of test.

---

## 23 · FIXES LANDED — and what the gate found on its first run

> **Status key for the sections above:** a finding marked here is **closed**;
> everything not listed is still open. Fix commits are named so the change can
> be read rather than re-derived.

### 23.1 · § 22.1's gate is built, and it earned its place immediately

`tests/test_doc_claims.py` — eight closed vocabularies checked against the
contracts that own them, in both directions, plus the `Resources` count in
prose, the retired-`strmap` check, and the required-keys agreement.

**It failed on its first run, on two drifts nobody had found by reading:**

| gap | what it means |
|---|---|
| `engines/template.md` names **none of** `rank_count`, `omp_threads`, `node_memory` | the `resolver` vocabulary is **closed and enforced** — `read_template` refuses an unknown name — and three of its four legal values appeared **nowhere a template author could read them**. § 6.4 described the *items* in prose and never spelled the names the code requires |
| `job-contracts.md` never names `Job.resources` | § 6.2 covers `Resources` at length, and § 6.2's closing sentence enumerated *"`warm` and `traits`"* as *"everything else a `Job` carries"* — omitting the third |

**Both fixed in their owning contracts** *(this commit)*: § 6.4 gains a
four-row table keyed by resolver name, plus the rule that three of the four
answer from the allocation and may never carry a value; § 6.2's sentence now
names all three fields.

> **This is the argument for 22.1, made by 22.1.** A five-pass human reading
> found the *`Resources` seven-vs-nine* drift. It did not find that a closed,
> enforced vocabulary was undocumented — because reading checks what is *there*,
> and this was an absence. Mutation-tested: renaming a member to `BROKEN` fails
> the gate by name.

### 23.2 · Fixes landed this pass

| finding | fix |
|---|---|
| **§ 22.1** | `tests/test_doc_claims.py` — built, mutation-tested |
| **new (23.1)** | `template.md` § 6.4 gains the resolver registry; `job-contracts.md` § 6.2 names `Job.resources` |

### 23.3 · Still open, and why

**Mechanical, no decision needed** — §§ 18.1–18.4 (the duplicate TOML key, the
missing `category`, the int/float, `section` residue, `template.py`'s dead
branch and stale comments), and § 2's four passages describing the retired ELPA
routing.

**Needs a decision, because each changes design rather than text:**

| # | question | recommendation |
|---|---|---|
| **1** | Multi-engine (§§ 1.2, 15.1) — real, or reserved? | say **reserved** unless a producer is planned; the axis costs five mechanisms and serves nothing today |
| **2** | Absent vs null (§ 20) — which policy does the registry require? | **declarations omit, records use `asdict`** — the template needs three states, no other artifact needs more than two |
| **3** | The seam leak (§ 6.1) — `resolve._apply` imports from `siesta/` | **move** `effective_config` to a shared module; it is already engine-agnostic |
| **4** | Invariant 5's *"forced cold"* (§ 17.1) | **delete the clause** — the relabel is sufficient and the code's own comment says so |

### 23.4 · Gate A built, and §§ 3.1–3.2 closed by it

`tests/test_doc_claims.py` now also walks every fenced ```` ```toml ```` block in
`engines/template.md`, parses it, and asserts each `[item.*]` carries § 3's four
required keys with a `type` from the closed vocabulary.

**It found more than the review had.** §§ 3.1 and 3.2 recorded a duplicate key
and one missing `category`. The gate showed § 6.3's example **also omits `type`
and `help`** on *both* its items — four required keys, three of them missing,
in the contract's own illustration of its format.

| fixed | was |
|---|---|
| § 6.3's `mesh_cutoff` | `category` twice (so the block did not parse) · `value = 300` where § 4.2's same item says `300.0` · no `type` · no `help` |
| § 6.3's `job_name` | no `type` · no `help` |
| § 12's `block_size` | no `category` |
| § 5's key diagram | node `group · section` → `group` |
| § 5's ⭐ note | now says `section` was replaced by `category` at `@2` |

**Mutation-tested**: re-introducing the duplicate key fails with
*"fenced toml block 2 does not parse: Cannot overwrite a value"*, and removing
a `category` fails naming the block and item.

> **Both gates found things reading did not**, and in the same direction:
> reading checks what a document *says*, and these were things it **failed to
> say** — an undocumented vocabulary (§ 23.1) and examples missing required
> keys. That is the honest boundary between the two methods, and the reason to
> run the gates before the next reading pass rather than after.

