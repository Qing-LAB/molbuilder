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

