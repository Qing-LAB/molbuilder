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

### 23.2 · Fixes landed this pass — TIER 1 COMPLETE

| finding | fix | commit |
|---|---|---|
| **§ 22.1** | `tests/test_doc_claims.py` — 8 vocabularies, both directions; mutation-tested | `6015f8ae` |
| **new (§ 23.1)** | `template.md` § 6.4 gains the **resolver registry** (the closed set was enforced and undocumented); `job-contracts.md` § 6.2 names `Job.resources` | `6015f8ae` |
| **§ 18.5 Gate A** | fenced examples parse and satisfy § 3; mutation-tested | `00edc723` |
| **§§ 3.1–3.2** | duplicate `category` deleted · `300` → `300.0` · `block_size` gains `category` · both § 6.3 items gain `type` + `help` | `00edc723` |
| **§ 3.4** | `section` out of § 5's diagram; the ⭐ note says it was replaced at `@2` | `00edc723` |
| **§ 4** | dead `dict` branch deleted · `config_from_template`'s stale reason · the doubled `section` explanation · `one()`'s message | `5af51051` |
| **§ 2** | § 6.1 **re-argued from `enable_gpu`** · § 4.2's example · § 11.3's use case · `generator.md` §§ 4, 8 · `job-contracts.md` § 6.2's GPU row | `5af51051` |

**Not fixed, deliberately:** `Template.engine`'s compat shim (§ 4, row 4) —
removing it touches every caller reading `t.engine`, which is a refactor, not a
cleanup. It stays on the Tier 2 list beside the seam leak.

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

---

## 24 · CORRECTIONS — four findings I framed wrongly (user, 2026-08-14)

Four of the Tier 2 questions were badly posed, and three of the findings behind
them were wrong or wrongly scoped. Corrected here rather than quietly amended,
because the wrong version is committed above.

### 24.0 · What the template IS — stated plainly, because I lost it

**One file. The single source of truth for a calculation's parameters — every
parameter, across every engine that calculation could run on.** Floor 2:
portable, names no machine.

**Why it exists.** Before it, each engine carried its own schema and its own
script generator. The same physics was unrecognisable across them —
`SiestaConfig ∩ PySCFConfig` is **three** field names, all molbuilder-internal
(§ 1 of the plan measured it). A form had to be written per engine, and *"what
charge did this run use?"* could not be answered without first knowing which
engine. So: **one file**, items grouped by `category` — six closed questions
about the calculation — each item declaring which `engines` it applies to.
Items are **never merged**: `net_charge` (SIESTA) and `charge` (PySCF) stay two
items in one category, so a surface builds the same six panels for any engine
and filters the contents by engine.

**The template is the DATA; each consumer pulls its slice.** The generator
emits one engine's deck at a time. That is not a tension with the file serving
several — it is the design: one source, many readers, each taking what applies
to it. `prep` takes them all, the deck writer takes `kind ∈ engine, deck`, the
wrapper takes `kind = wrapper` plus whatever `read_by` names, a surface takes a
category.

> **The conflation, named so it is not repeated.** I saw `render_template` emit
> a single engine and concluded the multi-engine design was hollow. That reads
> **a writer's current limitation as a statement about the design**. The design
> is settled and right; the writer is behind it. `render_template(config,
> config_cls=…)` takes **one** config class and derives `engines` from it, so it
> cannot yet gather two engines' items into one file. Making it accept N classes
> is a step in the unification — the same unification that moves script
> generation from engine-specific to template-driven — and not a contract to
> water down.

### 24.1 · § 1.2 was scoped wrong; § 15.1 was simply wrong — STRUCK

I treated *"the template serves several engines"* and *"the generator emits one
engine's script"* as a contradiction. **They are not.** The template is the one
file every engine's items live in; the generator produces one deck for one
engine. That is the design working as intended, and § 6.3 is not overselling.

**The real gap is narrower:** `render_template(config, …)` accepts **one config
object**, so nothing can *build* a file carrying two engines' items. The work is
a writer that accepts N config classes — part of the unification, **not** a
docs disclaimer. *(My recommendation, "say it is reserved", would have written
the limitation into the contract instead of removing it.)*

**§ 15.1 is struck.** `Task.engine` being a string is **correct**: a task names
which engine *this run* uses. It is not evidence about the template at all, and
citing it as confirmation was wrong.

### 24.2 · § 13 / § 19 / § 20 asked a serialisation question about a modelling problem

*"null or omit"* is not the question. `mem`, `gres`, `mpi_np`,
`continue_retries` and the rest are **parameters whose place in the unified
template has not been settled** — and in that model one item may have a
different representation per engine while remaining the same item, and some
items exist for one engine only. That is what the single template exists to
carry.

So the open work is: **map each of these into the template model** (which
category, which engines, which `kind`, which resolver), and the on-disk
encoding follows from what each one *is*. Re-scoped accordingly; the tally in
§ 20 stands as a fact, but it is a symptom, not the question.

### 24.3 · § 6.1's seam leak is a migration step, not a placement choice

`effective_config` living in `siesta/` is not *"move it or annotate it"*. The
project is mid-migration from engine-specific script generation to a
template-driven one, so engine-specific functions being translated into the
shared framework **is the work**, not a tidy-up beside it. It belongs on the
unification's list with the rest — prep, extraction, script generation,
assembly and validation as one flow.

### 24.4 · § 17.1 — I had it backwards. The doc is right; the CODE is missing it

I recommended deleting *"and forced cold"* on the grounds that the relabel
suffices. **It does not, and a benchmark starting warm makes no sense in the
first place.**

**The failure the relabel does not cover:** prep the same trial twice. The
second render carries the *same* trial label, so SIESTA finds the **first
attempt's** `.XV` / `.DM` under that label and warm-starts from them. The trial
then measures a continued run against other points measuring cold ones — the
timings are not comparable, which is the one thing a benchmark exists to
produce.

So invariant 5 is correct as written and the implementation is incomplete.
**Fix: force `restart = "clean"` on a trial's config**, next to the relabel in
`prep_calculation`, where `element.is_trial` is already the branch.

---

## 25 · MIGRATION WORK LIST — functions to reposition in the template-driven model

**What this is.** The project is moving from engine-specific script generation to
one template-driven flow: prep → extract → generate → assemble → validate. Some
functions are already in the right place; others are where they were written
before the unification, or are shaped for the world before it. **This is the
list of the second kind, with where each belongs and what has to change** — so
the work is picked up as migration, not rediscovered as a puzzle.

*(Recorded 2026-08-14 on user order, after I twice framed one of these as an
open question instead of as pending migration work. The evaluation to make for
each is the same: **where does it sit in the workflow, and what shape does the
new model need it in** — not "should we move it".)*

### 25.1 · `effective_config` — the ⊕ operator, still in the SIESTA package

| | |
|---|---|
| **now** | `molbuilder/siesta/input.py` |
| **what it is** | `effective config = template's values ⊕ this stage's overrides` — `stages.md` § 4's operator, *"the one place this happens"* |
| **engine-specific?** | **No.** Reads `dataclasses.fields(type(template))`, widens `int`→`float` where the field declares one, returns a new object. Nothing SIESTA in it |
| **production callers** | `resolve._apply` (floor 3) · `validation/stages.py`. Both engine-neutral |
| **also** | 7 test files import it from `siesta.input` |
| **belongs** | the shared resolution layer — `resolve.py` is floor 3's resolver and already wraps it |
| **and changes how** | `resolve._apply` currently fabricates a `Stage(name="resolve")` to pass overrides into a function whose parameter is a stage. Once it moves, that disappears: the operator takes a config and a mapping |
| **watch** | `runtime_config.read_effective_config` is a **different** function (config-file merging) that shares the name. Do not merge the two, and consider renaming one |

### 25.2 · `render_template` / `declarations_for` — one config class in, one engine out

| | |
|---|---|
| **now** | `molbuilder/template.py` |
| **the gap** | both take **one** config class. `render_template` derives `engines` from it and emits a one-element list, so **nothing can build the multi-engine file § 6.3 describes** |
| **belongs** | where it is — this is a **shape** change, not a placement one |
| **changes how** | accept **N** config classes; merge their declarations into one item table; tag each item with the `engines` it came from; keep items **unmerged** across engines (`net_charge` and `charge` stay two items in one category) |
| **then** | `Item.engines`, `select(engine=)`, `one(engine=)` and `_check_engine` stop being machinery for a file nothing writes — they are already correct, and they are waiting on this |

### 25.3 · Type coercion happens in two places, by two mechanisms

| | |
|---|---|
| **now** | `template._shape` + `_check_raw_value` coerce by the item's **declared `type`**; `effective_config` widens `int`→`float` by the **dataclass field's** annotation |
| **why it matters** | in the new model the template's `type` vocabulary is the authority on what a value *is*. Two coercion paths with different inputs is how a stage override and a template value end up different types for one field |
| **evaluate** | whether the override path should go through the template's declared type rather than the dataclass annotation — i.e. whether ⊕ belongs *beside* the type system rather than beside the config class |

### 25.4 · The wrapper still reads the deck text for GPU

| | |
|---|---|
| **now** | `runwrap._fdf_requests_gpu` parses `Diag.ELPA.GPU` out of the rendered `.fdf`; eight call sites |
| **the state** | `enable_gpu` **declares** `read_by = ("wrapper",)` and a gate asserts every scanner is claimed by a declaration — but nothing yet *hands* the wrapper writer the resolved value |
| **evaluate** | `project-layout.md` § 2.3.1 says the wrapper's environment is chosen by *a value the deck decides*, and the deck is what actually runs (a person may edit it). So the question is not "stop reading the deck" but **whether `prep`, which holds the resolved element, should pass the value and leave the scan as the standalone-use fallback** |
| **note** | this is the honest remainder of T8. The declaration landed; the hand-off did not |

### 25.5 · Already in the right shape — the pattern to copy

`prep._engine_seam` / `EngineSeam` is what the others should look like: the
engine supplies `config_cls`, `render_deck`, `relabel`, `label_of`,
`sibling_artifacts`, `suffix` — and `prep` calls them without knowing which
engine it holds. `generator.md` § 7's test (*"adding an engine adds files and
edits none"*) passes here. **When repositioning 25.1–25.4, the target shape is
this one: the engine declares, the shared flow calls.**

### 25.6 · How to evaluate each

For every function met during the remaining reading, ask in this order — the
order that would have avoided today's mistakes:

1. **What does the contract say this layer does?** *(not: what does the code do)*
2. **Is the body engine-specific, or only its address?** — 25.1 is address-only.
3. **Is it the right SHAPE for the new model,** or shaped for the one before it?
   — 25.2 is a shape problem at the right address.
4. **Who calls it, and are those callers on the layer the contract puts them on?**
5. Only then: where does it move, and what changes with it.

---

## 26 · ⛔ THE ORDER OF WORK — contract first, then code (user, 2026-08-14)

> *"You need to get your document contract done before you do any coding. You
> can keep notes persistent about what discovery of which code need to be
> investigated, but don't code until you have your contract all sorted out."*

**This governs everything below § 25.** No code lands against a finding until
the contract the finding belongs to says what the answer is. Discoveries about
code stay **notes** — § 25 is where they live — and become work only after the
document settles.

### 26.1 · What I did that earned the rule

Mid-review I began fixing code while contract questions were open: a
`restart="clean"` for trials, then the start of moving `effective_config` out of
`siesta/`. The second left the tree broken and was reverted. And I framed
**settled** design as open questions, which is the same error inverted — the
code becomes the de facto answer either way.

### 26.2 · The contract queue — in order, and none of it is code

| # | contract | what must be settled | affects |
|---|---|---|---|
| **C1** | `engines/template.md` § 6.3 | **What a multi-engine template file looks like** — one `[item.*]` table with per-item `engines`, or per-engine sections? How does one *item* carry a different representation per engine while staying one item? Which items are engine-specific and which are shared-question-different-spelling? | § 25.2's writer shape; `select`/`one`; the UI's panel filtering |
| **C2** | `engines/template.md` § 5 + § 6.2 | **Where the machine/exchange parameters sit in the model** — `mpi_np`, `cpus_per_task`, `mem`, `gres`, `continue_retries`, `max_memory_mb`: which category, which `kind`, which resolver, which engines. Their on-disk encoding (§§ 13/19/20's null-vs-omit) **follows from this** and is not its own question | `Resources`' relationship to the template; the three serialisation policies |
| **C3** | `engines/stages.md` § 4 | **Where `⊕` lives in the unified flow** and what it operates on — the config class's annotations, or the template's declared `type`? § 25.3 is the same question from the coercion side | § 25.1's move; § 25.3 |
| **C4** | `execution/project-layout.md` § 2.3.1 + `engines/template.md` § 6.1 | **What `prep` hands the wrapper writer** — the resolved item values, or does the wrapper keep reading the deck? § 2.3.1 says the deck decides; the deck is also hand-editable | § 25.4, the remainder of T8 |

**C1 is first** because § 25.2's writer shape follows from it, and because the
UI is built from this file — a panel set decided before C1 would be built on a
guess.

### 26.3 · What is already settled and needs no contract work

Recorded so it is not re-opened: the six categories · `kind`'s five members ·
the resolver registry (§ 6.4, now documented) · that the template is ONE file
serving every engine while the generator emits one engine's deck (§ 24.0) ·
that a trial is relabelled **and** forced cold (invariant 5 — the contract was
right, the code was behind, and that gap is closed) · the two SIESTA
environments splitting on provenance · routing on GPU alone.

---

## 27 · The link between § 25 (code to investigate) and § 26 (contracts to settle)

**So neither list is read alone.** § 25's entries are notes until the contract
beside them lands; § 26's decisions are what turn each into work. Nothing in the
left column is touched before its right column is settled.

| § 25 code note | blocked on | what the contract must say before it can be done |
|---|---|---|
| **25.1** `effective_config` — the `⊕` operator in `siesta/` | **C3** | where `⊕` lives in the unified flow, and whether it operates on the config class's annotations or the template's declared `type`. *Its new home follows from that answer, not from tidiness.* |
| **25.2** `render_template` / `declarations_for` — one config class in | **C1** | what a multi-engine file looks like: one item table with per-item `engines`, or per-engine sections; how one item carries a different spelling per engine while staying one item |
| **25.3** two type-coercion paths | **C3** | same decision as 25.1, from the other side — if the template's `type` is the authority, one of the two paths goes |
| **25.4** the wrapper still scans the deck for GPU | **C4** | whether `prep` hands the wrapper writer the resolved values, with the scan kept only as the standalone fallback |
| **§ 4 row 4** `Template.engine` compat shim | **C1** | if a file carries several engines, `engines[0]` stops being meaningful and the shim goes with it |
| **§ 20** three serialisation policies | **C2** | where the machine/exchange parameters sit in the template model — the encoding follows from what each parameter IS |

---

## 28 · C5 — the documents must be *aware of* the template model, not merely consistent with it

**The four decisions in § 26.2 are not the whole document job.** Several
contracts still describe the world before the template was the source of truth:
they are not wrong sentence by sentence, they are written as though the
template does not exist. Alignment means each says **what it now owns, and what
it defers to the template for**.

| document | how it still reads pre-template | what alignment needs |
|---|---|---|
| **`web/form-schema.md`** | describes the form built from **dataclass field metadata** — `dataclass_to_form_schema` | **NOT a live conflict** *(user, 2026-08-14)*: this is the **old UI design**, and that work has not started. **The template is not associated with a UI yet** — it is being designed to *carry* the information a UI will need. So form-schema.md is superseded-in-waiting, not contradictory, and it is **out of the C5 queue** until the UI work begins |
| **`engines/stages.md`** | owns `task.json`, `⊕`, and the stage vocabulary — written before `prep` rebuilt the config from a template | must say the ladder's overrides apply **to template items**, and hand `⊕`'s definition to whatever C3 decides |
| **`execution/job-system.md`** § 4.1 | the SIESTA ladder described through `siesta/stages.py` and per-stage config | say which parts are template items now |
| **`execution/running-a-job.md`** § 3 | the wrapper's own resolution chain | reconcile with `template.md` § 6.4's `omp_threads` resolver — the same chain is stated in both, and C4 decides who owns it |
| **`execution/job-contracts.md`** § 6.2 | the config ↔ scheduler translation table | after C2, say which of those rows are template items and which are allocation-only |
| **`engines/siesta.md`** · **`engines/pyscf.md`** | each engine's own parameter story | point at the template for the catalogue; keep the engine-specific science |

> **The test for "aligned"** is the one § 0.1 produced: a document is aligned
> when it answers a **different question** from the template — *why this value,
> what it means scientifically, how the layer behaves* — and defers the
> **enumeration** to the template. A document that re-lists items is not aligned
> however correct its list is today, because that is the shape that drifts.

**Not yet read, and each may add rows here:** the rest of `job-contracts.md` and
`project-layout.md`, `checkpointing.md`, `architecture.md`, `run-identity.md`,
`running-a-job.md`, `stages.md`, `form-schema.md`, the staged-runs plan.
§ 16 orders them; § 21 is what to check each against.

---

## 29 · C1 PREPARED — what a multi-engine template file looks like *(proposal, awaiting ruling)*

Measured first, so the decision rests on the schemas rather than on taste.

| | |
|---|---|
| `SiestaConfig` | **44** fields |
| `PySCFConfig` | **41** fields |
| **names appearing in both** | **3** |

### 29.1 · The file shape is already forced — one item table

**One `[item.*]` table, each item declaring its `engines`.** Not a proposal, a
consequence:

- **Per-engine sections would break the point.** `category` groups *across*
  engines so a surface builds six panels for any engine (§ 6.3). Sections would
  re-split by engine — the arrangement the template exists to replace.
- **`select(category=…, engine=…)` already reads a flat table.** It filters
  `it.engines`; there is nothing for it to filter if items live under engine
  headings.
- **TOML keys are unique within a table**, which is what makes the *next*
  question real rather than cosmetic.

### 29.2 · The real question: when are two engines' parameters ONE item?

The three shared names split cleanly, and not the way the name suggests:

| name | SIESTA | PySCF | same question? |
|---|---|---|---|
| `verbose_comments` | default `True`, `procedure`, molbuilder's own comment-block control | identical in all three | **yes** — one item, `engines = ["siesta", "pyscf"]` |
| `write_molwatch_log` | default `True`, `procedure`, writes `.molwatch.log` | identical | **yes** — one item |
| `max_memory_mb` | default **`None`**, a `.run.sh` **`ulimit -v`** — a process cap | default **`4000`**, **`mol.max_memory`** — the in-core budget that decides in-core vs out-of-core algorithms | **NO** |

**`max_memory_mb` is one name over two different parameters** — which the
unification plan's § 1 already lists as a measured defect, not a discovery here.
One is a hard ceiling the wrapper enforces; the other is a *scientific* choice
that changes which algorithm PySCF runs.

### 29.3 · The rule this yields, and it is the user's own principle applied

> **Two engines share an item when it is the same question with the same
> answer. A shared *name* is not evidence of either.**

Which is *"items are never merged across engines"* (§ 6.3) extended to the case
the contract has not faced: what happens when the names collide anyway. And
since TOML forbids duplicate keys, a non-shared collision **must** be resolved —
the file cannot express it otherwise.

**Two ways to resolve it, and they are not equal:**

| | |
|---|---|
| **(a) qualify by engine** — `siesta_max_memory_mb` / `pyscf_max_memory_mb` | encodes the collision. The engine is already on the item as `engines`, so the prefix repeats what the data says |
| **(b) name each for what it IS** — e.g. SIESTA's process cap vs PySCF's in-core working-memory budget | **fixes** the defect § 1 measured instead of encoding it, and the names become self-explaining in a file a person reads (**G5**) |

**Recommendation: (b).** It closes a known finding rather than carrying it into
the new format, and it is the only option under which reading the two item names
tells you they are different things.

### 29.4 · What the ruling decides, concretely

1. **(b) or (a)** for the collision — and if (b), the two new names.
2. Whether `verbose_comments` and `write_molwatch_log` become **one item each
   with `engines = ["siesta","pyscf"]`** *(recommended — same question, same
   answer, same default)*, or stay two items apiece.
3. That settled, a combined file holds **83 items** (85 fields − 2 merged), and
   § 25.2's writer has its target: take N config classes, emit one table,
   tag each item with the engines it came from, refuse a collision that is not
   a declared share.

> **Nothing here is code.** § 26's rule holds: this is C1's content, and
> `render_template` does not change until it is ruled on.

---

## 30 · C1 RULED — the four items, and the finding the ruling exposed

**User, 2026-08-14:** agreed with § 29.3's recommendation — name each for what
it is, merge the two genuine shares — plus a fact that changes one of them:

> *"`max_memory_mb` is only explicitly set when user requested, the typical
> memory limit is unlimited — meaning all physical memory is allowed."*

### 30.1 · The resolver is on the wrong item today

| | today | correct |
|---|---|---|
| **SIESTA's cap** | `resolver = "node_memory"` | **no resolver.** Unset means *do not cap*. "Cap at the node's maximum" is still a cap, and is not what unset means |
| **PySCF's budget** | no resolver, `default = 4000` | **`resolver = "node_memory"`.** Unset should mean *use what this node has*; 4000 is an arbitrary 4 GB that silently forces out-of-core algorithms on a machine with far more |

**The two items want opposite unset behaviour**, which is the strongest evidence
yet that they were never one parameter. A merged item could not express it: one
control, and `unset` would have to mean *no cap* and *use the whole node* at the
same time.

### 30.2 · Final shapes

```toml
[item.memory_cap_mb]                    # was SIESTA's max_memory_mb
kind       = "wrapper"                  # becomes `ulimit -v`; no engine keyword
category   = "execution"
engines    = ["siesta"]
type       = "int"
unit       = "MB"
# NO value and NO resolver -- unset means DO NOT CAP.
label      = "Memory cap (per rank)"
null_label = "(no cap -- all physical memory)"
help       = """
A hard ceiling the run wrapper applies with `ulimit -v`.  Left unset the process
is uncapped and may use all physical memory, which is the normal case -- set one
only when you want the job killed rather than allowed to grow."""

[item.working_memory_mb]                # was PySCF's max_memory_mb
kind       = "engine"
category   = "execution"
engines    = ["pyscf"]
anchor     = "mol.max_memory"
type       = "int"
unit       = "MB"
range      = [100, 1000000]
resolver   = "node_memory"              # unset -> what this node actually has
label      = "Working memory"
null_label = "(auto -- this node's memory)"
help       = """
How much memory PySCF believes it may use.  It is NOT a cap: it selects in-core
versus out-of-core algorithms, so too small a value silently makes the run
slower rather than failing.  Left unset, `prep` resolves it from the
machine."""

[item.verbose_comments]                 # ONE item, both engines
kind     = "produce"
category = "procedure"
engines  = ["siesta", "pyscf"]
type     = "bool"
value    = true
default  = true
label    = "Verbose inline comments"
help     = "Explain each setting in the generated file, inline."

[item.write_molwatch_log]               # ONE item, both engines
kind     = "produce"
category = "procedure"
engines  = ["siesta", "pyscf"]
type     = "bool"
value    = true
default  = true
label    = "Write the molwatch trajectory log"
help     = "Emit `<basename>.molwatch.log` for the live viewer."
```

### 30.3 · What this settles, and what it opens

**Settled:** the file is one `[item.*]` table; two engines share an item only
when it is the same question with the same answer; a name collision that is not
a share is resolved by **naming each for what it is**; a combined file holds
**83 items** (85 fields − 2 merged).

**Opened, and it is a schema change rather than a template one:** the rename
touches `SiestaConfig.max_memory_mb` → `memory_cap_mb` and
`PySCFConfig.max_memory_mb` → `working_memory_mb`, plus the resolver move and
the removal of PySCF's `4000` default. Under § 26 that is **still not code
yet** — it is C2's territory (where the machine/exchange parameters sit), and
C2 should be settled with this ruling in hand rather than beside it.

> **Note for C2:** `Resources.max_memory_mb` is the *exchange* name that carries
> the wrapper's cap to `prep`. If the config field becomes `memory_cap_mb`, C2
> decides whether the exchange name follows — `job-contracts.md` § 6.2 is the
> owner of the config ↔ exchange mapping, and its table has a row for this.

---

## 31 · C2 PREPARED — where the machine parameters sit *(measured; one ruling needed)*

### 31.1 · The same question is answered two ways, per engine

| the question | SIESTA | PySCF |
|---|---|---|
| **how many threads per process?** | `omp_threads` — **`allocation=True`** (excluded from template values), `resolver="omp_threads"` | `threads` — **`allocation=False`** (an ordinary item with a value), **no resolver** |
| **how much memory?** | `max_memory_mb` — `allocation=True`, `resolver="node_memory"` | `max_memory_mb` — `allocation=False`, no resolver, `default=4000` |
| **use a GPU?** | `enable_gpu` | `use_gpu` |
| **how many MPI ranks?** | `mpi_np` — `allocation=True`, `resolver="rank_count"` | *(none — PySCF is single-process)* |

### 31.2 · The finding: floor 2's rule is enforced on one engine only

```
allocation-tagged fields:  SiestaConfig -> mpi_np, omp_threads, max_memory_mb
                           PySCFConfig  -> (none)
```

`template_fields` strips allocation-tagged names on the rebuild path, so **a
`threads` value can be written into a PySCF template and cannot into a SIESTA
one.** § 2's *"floor 2 must never assert a machine fact's value"* is a rule about
the **calculation**, not about SIESTA — and today it is enforced for one engine
and not the other.

This is the same defect class the whole unification exists to remove: *the same
physics, treated differently because it arrived through a different engine's
schema.* It is not a PySCF bug — PySCF's fields were never audited against the
rule, because until T5 PySCF had no template at all.

### 31.3 · Proposal — the allocation set is a property of the QUESTION

> **A parameter is `allocation` when its VALUE is a machine fact, whatever
> engine asks it.** Not when a particular engine's schema happens to tag it.

| item | `allocation` | `resolver` | why |
|---|:--:|---|---|
| `mpi_np` (SIESTA) | **yes** *(unchanged)* | `rank_count` | ranks are granted, not chosen in a portable description |
| `omp_threads` (SIESTA) | **yes** *(unchanged)* | `omp_threads` | same |
| **`threads` (PySCF)** | **yes — CHANGE** | **`omp_threads` — CHANGE** | the same question SIESTA's `omp_threads` asks. A thread count is what the node granted |
| **`memory_cap_mb`** (was SIESTA's `max_memory_mb`) | **yes** *(unchanged)* | **none — CHANGE**, per § 30.1 | unset means *do not cap* |
| **`working_memory_mb`** (was PySCF's `max_memory_mb`) | **no — it is science** | **`node_memory` — CHANGE** | it selects in-core vs out-of-core, so it changes the algorithm, not just the launch. A user may legitimately pin it |
| `continue_retries` | **no** *(unchanged)* | none | a retry **policy** — portable, names no machine. It rides `Resources` only because that is the road to the wrapper |
| `parallel_block_size` | **no** *(unchanged)* | `block_size` | a tunable a person may set or benchmark (§ 12) |
| `enable_gpu` / `use_gpu` | **no** *(unchanged)* | none | asking for a GPU is a choice; *getting* one is the allocation's `gres` |

**The one behavioural consequence:** PySCF's `threads` stops carrying a value in
a template and is resolved at `prep`, exactly as SIESTA's already is.

### 31.4 · The open question — one item, two engine keywords?

`enable_gpu` (SIESTA → `Diag.ELPA.GPU`) and `use_gpu` (PySCF) are **the same
question with the same answer** — C1's test for merging — but each engine
renders it to a **different keyword**, and `Item` carries a single `anchor`.

| | |
|---|---|
| **(a)** two items, as today | keeps the *"same physics, two names"* defect § 1 measured — the thing the unification set out to end |
| **(b)** one item, `anchor` becomes per-engine | the model change that makes merging general; costs a shape change to `Item` |
| **(c)** one item, `kind="produce"`, each engine's deck writer renders it | no `anchor` needed, but it moves an engine keyword out of `kind="engine"`, which § 6 uses to decide who emits it |

**This is C1's unfinished business, surfacing where it bites.** The same choice
governs `net_charge`/`charge` and `spin_polarized`+`spin_total`/`spin` — the
examples § 6.3 uses to argue *"items are never merged"*. If (b) or (c), that
paragraph is what changes.

> **Recommendation: decide (a) vs (b) before C2 lands**, because the allocation
> table above is stable under either, but the *item count* of a combined file is
> not — § 29.4's "83 items" assumes (a).

---

## 32 - RULED: one item per question, and the GENERATOR renders it

**User, 2026-08-14:** *"unify them. there is no point of having two questions
about the same answer."* And, on how the merged item reaches each engine:
*"enable_gpu is one flag, how this emits script is entirely the generator's
job and engine specific, right?"*

**Yes -- and that makes SS 31.4 resolve to (c), not (b).**

### 32.1 - No new mechanism. `kind` already says this

I was about to propose per-engine `anchor` sub-tables. That invents machinery
for a case the vocabulary already covers:

> **`kind = "deck"`** -- *"molbuilder's own, but it shapes the deck -- by
> expanding to keywords, ordering a block, or supplying verbatim text ... the
> deck writer, through **molbuilder's rule rather than one keyword**"*
> (`template.md` SS 6)

A unified `use_gpu` **is** that. The item carries the answer; each engine's
deck writer renders it however that engine needs -- `Diag.ELPA.GPU` for SIESTA,
a backend selection for PySCF. The template never learns either spelling.

| | |
|---|---|
| **`kind = "engine"`** | the item IS one engine keyword. `anchor` names it. Single-engine by nature |
| **`kind = "deck"`** | one question, and **the generator knows how to express it**. No `anchor`. This is what a unified cross-engine item is |

**So the merge costs nothing structural.** `Item` does not change,
`_item_payload` does not change, `_ITEM_KEY_ORDER` does not change. What
changes is which `kind` these items declare, and that `engines` lists both.

### 32.2 - What merges

| one item | kind | each generator emits |
|---|---|---|
| `use_gpu` | `deck` | SIESTA: `Diag.ELPA.GPU` (+ the ELPA solver gate). PySCF: the GPU backend |
| `charge` | `deck` | SIESTA: `NetCharge`. PySCF: `gto.M(charge=)` |
| `verbose_comments`, `write_molwatch_log` | `produce` | already engine-neutral |

### 32.3 - What does NOT merge, and why the rule still holds

**`dm_tolerance` vs `scf_conv_tol`** stay two items. Same English phrase
(*"SCF convergence"*), two questions -- a density-matrix criterion and an
energy criterion -- and **neither can take the other's value**. That is SS 6.3's
own example and it survives unchanged.

**spin needs its own ruling.** SIESTA carries `spin_polarized` (bool) **and**
`spin_total` (float); PySCF carries `spin` (2S). Same question, but the
**answer is decomposed differently**, so it fails the test *(same question AND
same answer)*. One item holds one value; SIESTA needs two.

| | |
|---|---|
| **derive** | one item `spin` (2S). SIESTA's writer emits `Spin.Fix` + `Spin.Total`, and `spin_polarized` becomes `spin != 0` rather than a stored field -- deleting a field that can disagree with its sibling |
| **keep two** | honest to each engine's shape, but keeps *the same physics unrecognisable across engines* for this case |

*(Deriving is the same move SS 8.1 already relies on -- one parameter writing
two keywords, which it names as a reason splicing cannot work.)*

### 32.4 - The contract change this makes

SS 6.3's *"items are never merged across engines"* is the paragraph that
changes. The rule becomes:

> **Items merge when they are the same question with the same answer. The
> engine's spelling is the GENERATOR's, not the template's -- which is what
> `kind = "deck"` has always meant.**

The old rule's stated worry -- *inventing a shared vocabulary and deriving each
engine's spelling from it* -- was aimed at merging things that only sound
alike. That worry is answered by the test, not by refusing to merge: `charge`
and `net_charge` are one question; `dm_tolerance` and `scf_conv_tol` are two.

### 32.5 - Still no code

SS 26 holds. Open, and each is one sentence to rule:

1. **spin** -- derive, or keep two? (SS 32.3)
2. Do merged items keep `expands`? It lists *which keywords this item
   produces*, which now differs per engine. It documents rather than steers
   (SS 8.1), so: union, omit, or per-engine.

---

## 33 - CONSTRAINT ON ANY CLEANUP: the reserved blocks are a PERSISTENCE CHANNEL

**User, 2026-08-14, before the cleanup:**

> *"user customized content is a key component to carry structural
> information and others persistently across scripts - some of these
> information will be regenerated from the final script and for use in later
> cases. for example, atom labels are saved in the .fdf for siesta so it is
> not lost in optimization and later in the connection with transport (not
> implemented yet), these labels are important to keep track of frozen atoms,
> and bridges etc."*

### 33.1 - What this establishes

**The deck is not write-only.** `job-contracts.md` SS 3.1 already says parsers
find blocks by markers -- so the blocks are an **interface**, and the
information in them flows BACK out:

| block | carries | read back for |
|---|---|---|
| **ATOM-METADATA** | regions, frozen atoms, annotation channels | surviving a relaxation, and **transport** -- which atoms are electrodes, which are the bridge, which are held fixed |
| **USER-CUSTOM** | a person's own engine text | the next generation of the same calculation |

**Atom labels are identity, not decoration.** A relaxation rewrites
coordinates; the labels say *which atom is which* afterwards. Transport then
needs exactly that -- electrode / bridge / frozen membership -- and it is not
implemented yet, so nothing today would notice if the carrier were weakened.
**That is precisely why it must not be.**

### 33.2 - What it forbids in the cleanup

| candidate | ruling |
|---|---|
| **`text` type** (SS 32's C6: 0 uses, unreachable) | **KEEP. Do not drop.** It is reserved for `user_custom` (SS 9.2), and the user has now named that component load-bearing. Its zero count is *work not done*, not *machinery nobody wants* -- the opposite of `strmap` |
| ATOM-METADATA's emission rule | **unchanged.** T10 measured it round-tripping: regions, `frozen_atoms` and annotation channels all present, in exactly the 6 of 57 decks whose structures carry them |
| the reserved-block markers | **unchanged**, and now understood as a parsing contract rather than a comment convention |

### 33.3 - What it re-frames

SS 9.2 argues USER-CUSTOM must live in the **template** because `prep` cannot
harvest it from a previous deck -- there is no previous deck, `prep` renders
one per stage, and `prep` must be reproducible. **That still holds, and does
not conflict with reading information back out of a deck.** Two different
directions:

* **template -> deck**, every time, reproducibly: what the calculation IS.
* **deck -> a later step**: what the completed run's structure MEANS -- labels,
  regions, frozen sets -- for a stage or a kind of calculation that comes after.

The second is why the blocks exist at all, and it is the half a cleanup would
quietly damage, because **its consumer is not written yet.**

### 33.4 - Added to the cross-index (SS 21.2) as a load-bearing rule

> **R15. The generated deck carries structural identity forward.** Atom
> labels, regions and frozen sets ride in ATOM-METADATA so they survive a
> relaxation and reach later calculations -- transport above all. A change
> that drops, reorders or lossily rewrites those blocks breaks a consumer that
> **does not exist yet**, so no test today would catch it.
> *Owner: `execution/job-contracts.md` SS 3.1 / SS 3.4.*

---

## 34 - SSSS 3.4-3.6 read in full: the persistence format, and one label name that is wrong

**User, 2026-08-14:** *"there is a whole section of document talking about
persistent user comments with formats to carry essential structural
information in generated scripts. you should be aware of that and cite that
in the template design."*

Correct -- I had discussed the persistence channel (SS 33) from SS 3.1 alone
and had not read the sections that SPECIFY it.

### 34.1 - The finding: the example's frozen label does not match the code

SS 3.4's example writes the frozen set as a `regions` label named **`frozen`**:

```
"regions": { "L-electrode": [...], "R-electrode": [...],
             "bridge": [...], "frozen": [88, 89, ...] }
```

**The code writes `frozen_atoms`.** `structure.FROZEN_LABEL = "frozen_atoms"`,
and T10's measured junction deck confirms it:

```
"regions":{"electrode_left":[0,1],"electrode_right":[2,3],
           "molecule":[4,5,6,7],"frozen_atoms":[0,1,2,3]}
```

The **shape** is right and the section's own 2026-08-12 amendment is correct
-- frozen is a label INSIDE `regions`, not a key beside it. The **name in the
example is not**.

> **Why this one matters more than a typo.** SS 3.4's example is the
> specification a transport reader would be written from, and transport is
> exactly the consumer the user named: *"these labels are important to keep
> track of frozen atoms, and bridges"*. A reader built from the example looks
> up `"frozen"`, finds nothing, and concludes the run froze no atoms.
> **Core document: `execution/job-contracts.md` SS 3.4.** Fix the example's
> label to `frozen_atoms`, or state that the name is `structure.FROZEN_LABEL`
> and cite it rather than spelling it -- the second is better, since it is the
> same one-authority rule SS 3.4 already applies to the version number.

### 34.2 - What SSSS 3.4-3.6 establish, now cited from `template.md` SS 9

| | |
|---|---|
| **format** | `molstruct-json/v<SCHEMA_VERSION>` -- the number READ from `sidecars/molstruct.SCHEMA_VERSION` (7), never typed, so block and sidecar cannot drift |
| **emission** | conditional: only when `regions` OR `annotations` is non-empty. Absence is the honest signal, so it cannot suppress a sidecar added later |
| **indices** | ATOM-METADATA is **0-based**; SIESTA's `%block Geometry.Constraints` is **1-based**. Both in one file, deliberately |
| **precedence** | **in-body wins over the sidecar** -- `apply_companion_labels_if_present` runs before the sidecar branch. The sidecar is the fallback for plain `.xyz` and pre-contract scripts |
| **round trip** | SS 3.6: a tool may assume ATOM-METADATA round-trips (its dict feeds the same `apply_to_structure` path the sidecar uses) and USER-CUSTOM survives regeneration |
| **versioning** | each block versions independently; **no autodetection, no silent upgrade, no translation** -- an old block is refused with the regenerate message |

### 34.3 - Cited in the template design, as asked

`engines/template.md` SS 9 previously pointed only at SS 3.1 (the block list).
It now names **SSSS 3.4-3.6 as the format authority**, and states plainly that
the deck is not write-only: the blocks are a persistence channel, transport
reads electrode / bridge / frozen membership back out, and a change that
drops or lossily rewrites them breaks a consumer that does not exist yet.

*(SS 9.1 already cited SS 3.4 once -- for the 0-based/1-based note -- which is
why the gap survived: a citation existed, so nothing looked missing.)*

---

## 35 - C6: the STRUCTURAL-INFORMATION SEAM -- how a generator knows what to carry

**User, 2026-08-14:**

> *"this customized section is only known at the generator side because
> template does not know anything structural specific and this information is
> only available as input to the generator from the structural source. so
> handling this would need some mechanism design so the generator knows what
> to do."*

### 35.1 - The contract already draws the line; it does not say how it is crossed

`template.md` SS 9.1 states the principle exactly -- **"Labels ride with the
atoms, not with the parameters"**: ATOM-METADATA is **not** a template item and
must not become one, because a region label or a frozen flag is a fact about
*which atoms*, and it already has a carrier (the structure plus its
`.molstruct.json` sidecar).

So the generator has **two inputs and one output**:

| input | carries | floor |
|---|---|---|
| the resolved config, from the template | every **parameter** of the calculation | 2 -> 3 |
| the structure + sidecar | every **structural fact** -- regions, frozen set, annotation channels | 2 |

**What is missing is the rule for the second one.** The contract says where
structural facts live and that they must reach the deck; it does not say how a
generator is *told* which of them to emit, or in what form -- which is exactly
the user's point.

### 35.2 - What exists today, and how far it goes

| | mechanism | engine-neutral? |
|---|---|---|
| **annotation channels** | `annotations_fdf` -- a **registry**: `register_fdf_strategy(name, fn)`, and `emit_channels(struct)` renders every channel that has a registered strategy. Channels with none are **carried, not emitted**, and `unregistered_channels()` names them | **yes, and this is the pattern** -- a channel declares its own emission, nothing shared learns channel names |
| **regions / frozen** | hard-coded in each emitter: `siesta/input.py` calls `emit_atom_metadata(...)` and builds `%block Geometry.Constraints` inline | **no** -- the deck writer knows these two by name |
| **USER-CUSTOM** | `emit_user_custom_placeholder()` today; SS 9.2 makes it a template item (`kind="deck"`, `type="text"`) once the schema field exists | **yes**, once it lands -- it is a parameter, not a structural fact |

> **The registry half already passes `generator.md` SS 7's seam test** -- adding
> an annotation channel adds a strategy and edits no shared file. The
> hard-coded half does not: a new engine, or a new kind of structural fact,
> means editing every deck writer.

### 35.3 - What C6 has to decide

1. **Is the structural seam a registry like `annotations_fdf`, extended to
   cover regions and the frozen set** -- so *every* structural fact declares
   its own emission and no deck writer carries a list of names? That is the
   pattern already proven in the tree.
2. **Or does the generator receive a structural payload** it renders wholesale
   -- ATOM-METADATA as one blob (which is close to what `emit_atom_metadata`
   already does) with only the ENGINE-BODY consequences (SIESTA's
   `Geometry.Constraints`) per engine?
3. **Either way: what is the read-back contract?** SS 3.6 says a tool may assume
   ATOM-METADATA round-trips through `apply_to_structure`. **Transport is the
   named consumer and is not written**, so the seam must be designed for a
   reader that does not exist -- which is why it is a contract question and
   not something to discover while writing transport.

### 35.4 - Why this belongs beside C1-C5 and not after them

C1-C4 settle what the **template** carries. C6 settles what reaches the deck
**from the other input** -- and the two meet in one artifact. A generator
built for C1's items and then retrofitted for structural facts is how
`siesta/input.py` came to know `regions` and `frozen_atoms` by name in the
first place.

**Owner: `engines/template.md` SS 9.1** (which draws the line) **with**
**`execution/job-contracts.md` SS 3.4** (which owns the format). Neither
currently names the mechanism.

### 35.5 - The template's share of the seam is the FORMAT (user, 2026-08-14)

> *"this template part probably would focus on the FORMAT of how the text
> should be constructed such as begin mark, end mark, quote etc."*

**That splits C6 cleanly, and each half goes where it already belongs:**

| | who supplies it | why there |
|---|---|---|
| **CONTENT** -- which atoms are frozen, which regions exist, the annotation channels | **the structure + sidecar**, as the generator's other input | a fact about *which atoms*; `template.md` SS 9.1 already forbids it as a template item |
| **FORMAT** -- the begin/end markers, the comment prefix, how a payload line is quoted | **the template** | it is a property of the **engine's file syntax**, not of this structure and not of this run. It is exactly the kind of thing an item declares |
| **ASSEMBLY** | **the generator** | it holds both and writes the file |

**Why FORMAT is genuinely a template concern.** It is engine-specific and
structure-independent -- the two properties that make something an item. Today
every block is fenced `# === molbuilder <name> BEGIN ===` with a `#` prefix,
hard-coded in `script_emit.py`, which works only because both shipped engines
comment with `#`. A Fortran-input engine (`!`) or anything C-like (`//`) would
need the emitter edited -- the seam leak `generator.md` SS 7 names.

**What this makes tractable.** C6 stops being *"design a mechanism for
structural information"* and becomes two smaller questions:

1. **Which format facts does an engine declare?** Comment prefix, marker
   shape, how a multi-line payload (the molstruct JSON) is line-prefixed, how
   a quote inside verbatim user text is handled.
2. **How does the generator ask for content?** The registry pattern
   `annotations_fdf` already proves (SS 35.2), extended to regions and the
   frozen set so no deck writer carries those names.

> **And it explains the `text` type's zero uses (SS 32 / C6).** `type = "text"`
> is *"verbatim engine text, copied rather than interpreted"* -- the FORMAT
> half of USER-CUSTOM. It is unused because the format facts have never been
> declared anywhere: they are constants in `script_emit`. That is the same
> *reader-built-writer-missing* shape found five times elsewhere, and SS 33
> is why it must be filled in rather than retired.

---

## 36 - C1's contract text LANDED, and the frozen-label spec fixed

| | |
|---|---|
| **`job-contracts.md` SS 3.4** | the ATOM-METADATA example now spells the frozen label the way the code writes it, and a new bullet applies SS 3.4's own one-authority rule to the NAME: it is `structure.FROZEN_LABEL`, cited rather than re-spelled. **Gated both ways** in `test_doc_claims.py`, mutation-tested |
| **`engines/template.md` SS 6.3** | *"items are never merged across engines"* replaced by the **merge test**, per the user's ruling |

### 36.1 - What SS 6.3 now says

> Two engines share an item when it is **the same question** *and* **the same
> answer**. A shared name is evidence of neither; a different keyword is
> evidence against neither. **The engine's spelling is the GENERATOR's** --
> which is what `kind = "deck"` has always meant.

The old paragraph's own justification was *the risk of fusing things that
merely sound alike*. The test does that job directly -- `dm_tolerance` and
`scf_conv_tol` still stay two items, because neither can take the other's
value -- while the flat refusal was keeping `net_charge` / `charge` as two
names for one question, which is the defect SS 1 of the unification plan
measured.

Spin is recorded in the section as **not merged today**, with the reason: the
two numbers are the same quantity (unpaired electrons) but the answer is
decomposed differently, and there is a third state -- *polarized, moment
free* -- that a single count cannot express.

### 36.2 - Still open on C1

1. **`expands` on a merged item** -- union, omit, or per-engine? It documents
   rather than steers (SS 8.1), so this is a readability call, not a mechanism
   one.
3. **`engines` explicit or derived** once items list several.

**What C1 has already unblocked:** SS 25.2's writer now has a target shape --
take N config classes, emit one item table, merge by the test, and let each
deck writer render. `Item` does not change.

---

## 37 - C3 PREPARED: the two coercion paths disagree, measured

SS 25.3 flagged that type coercion happens twice by two mechanisms. Here is the
consequence, measured rather than reasoned:

```
template value 300  ->  cfg.mesh_cutoff = 300   (int)    ->  MeshCutoff 300 Ry
stage override 300  ->  cfg.mesh_cutoff = 300.0 (float)  ->  MeshCutoff 300.0 Ry
```

**The same number, arriving by the two paths the contract defines, produces
two different decks.**

| path | what it does | why |
|---|---|---|
| **stage override** -- `effective_config` | widens `int` -> `float` from the **dataclass annotation** | its own comment: *"an override that arrived from JSON carries JSON's types, and JSON has one number ... the same number, a different deck"* |
| **template value** -- `config_from_template` | **no widening.** `_TYPE_CHECKS["float"]` accepts an int, and `_shape` returns it unchanged | nothing decided it should; the check validates the TYPE and the shaper only fixes list-vs-tuple |

### 37.1 - What it costs, stated honestly

**Not a physics error** -- SIESTA reads `300` and `300.0` as the same 300 Ry.
What it breaks is **faithfulness**, which is G4's whole subject:

* **G4's own test is textual** -- *"render a stage's deck from the template and
  from the config a surface held; **the text is identical**"*. Two decks for one
  calculation fail it while computing the same answer.
* **BENCH-MARKS' `default=` row** is emitted from the value, so it differs too.
* Anything comparing decks by hash -- a checkpoint, a byte-identical readback,
  T10's own harness -- sees a change that is not one.

> **And it is why SS 18.1's doc fix mattered more than it looked.** SS 6.3's
> example carried `value = 300` for a float field. Anyone copying the
> contract's own illustration into a hand-written template got a deck differing
> from the same calculation written `300.0` -- and `render_template` emits
> `300.0`, because `_toml_value` uses `repr()` on floats *precisely so a float
> round-trips as a float*. The writer was careful; the reader was not.

### 37.2 - What C3 has to decide

The two paths disagree because **they answer to different authorities**: the
dataclass annotation on one side, the item's declared `type` on the other.
SS 25.3 asked which should win; this measurement says the answer cannot be
*"leave it"*.

| option | |
|---|---|
| **the template's `type` is the authority** | `_shape` widens for `type="float"`, exactly as `effective_config` does for the annotation. Then the template's declared vocabulary governs its own values, and SS 25.1's move puts both behind one operator |
| **the annotation is the authority** | `config_from_template` widens using `dataclasses.fields`, duplicating `effective_config`'s logic -- two implementations of one rule, which is what SS 25.1 exists to end |
| **refuse instead of widening** | a template declaring `type="float"` with an int VALUE is refused at read. Strictest, and it would have caught SS 6.3's example. But it breaks hand-editing, which G5 protects: a person writing `300` means 300 |

**Recommendation: the first.** It puts the rule where the vocabulary already
is, matches what the writer already does (`repr()` on floats), and leaves
hand-editing forgiving. It is also the option that makes SS 25.1's move a
simplification rather than a relocation.

*(No code. This is C3's content, and SS 26 holds.)*

### 37.3 - MEASURED: SIESTA does not care, which re-weights the whole finding

*(User asked the right question -- "is siesta expect a float type though?" --
and SS 37.1 answered it from reasoning. Measured now.)*

```
MeshCutoff 300 Ry    ->  redata: Mesh Cutoff = 300.0000 Ry | Total = -30.136693
MeshCutoff 300.0 Ry  ->  redata: Mesh Cutoff = 300.0000 Ry | Total = -30.136693
```

Both exit 0, both parse to the same real, **identical total energy**. SIESTA's
FDF reader takes a physical quantity as a real number and the literal's form is
nothing to it.

### 37.4 - So the argument for C3 is NOT the deck text

Ranked the way the project ranks fixes -- **science, then clarity, then
duplication**:

| | verdict |
|---|---|
| **science** | **no impact, measured.** Same cutoff, same energy |
| **clarity** | minor. Two decks for one calculation is confusing to a person diffing them, and it fails G4's *textual* identity test -- but G4 exists to catch a **different calculation**, and this is not one |
| **duplication** | **this is the real cost.** One rule -- *what type is this value* -- has **two implementations**, answering to two authorities. That is SS 25.1's problem exactly, and it is why the two paths disagree at all |

> **The corrected framing.** The differing deck is a **symptom that revealed**
> the duplication; it is not the reason to fix it. A fix aimed at the deck text
> would add widening to `config_from_template` and leave two implementations
> standing -- the tempting move, and the wrong one.

**C3's recommendation is unchanged and its justification is stronger:** put the
rule where the vocabulary already is (the item's declared `type`), so there is
**one** implementation. That is a simplification whether or not any deck ever
differed.

*(And it lowers C3's priority relative to C1/C2/C6, which change what the
format can express. C3 makes one rule one implementation -- worth doing, not
urgent.)*

---

## 38 - SCIENCE: `mesh_cutoff` is a REQUEST, and the grid quantises it

*(User asked whether SIESTA allows `300.5`. It does -- and the answer that
came back matters more than the question.)*

Measured, H2 in an 8 A box, everything else held:

```
asked 300    Ry  ->  Total = -30.136693
asked 300.5  Ry  ->  Total = -30.136693     IDENTICAL
asked 301.7  Ry  ->  Total = -30.136693     IDENTICAL

InitMesh: Mesh cutoff (required, used) =   301.700   349.790 Ry
InitMesh: MESH = 90 x 90 x 90 = 729000
```

### 38.1 - What this establishes

1. **`type = "float"` is correct.** SIESTA accepts a continuous value and
   echoes it back exactly -- `redata: Mesh Cutoff = 300.5000 Ry`. It is not a
   disguised integer.
2. **But the value is an ASK, not the effective quantity.** SIESTA builds a
   real-space grid of integer dimensions, so the cutoff it can actually
   deliver is discrete. It reports both -- *"(required, used)"* -- and the
   **used** cutoff is the one the physics sees: 349.79 Ry for a 301.7 Ry ask.
3. **A band of asked values is one calculation.** 300, 300.5 and 301.7 all
   land on the same 90x90x90 mesh and give the same energy to every digit
   printed.

### 38.2 - Why it matters beyond curiosity

| | |
|---|---|
| **a fine sweep of `mesh_cutoff` measures nothing** | between snap points the deck differs and the calculation does not. A convergence study stepping 300 -> 310 -> 320 may be three runs of one calculation, and would read as *"converged"* for the wrong reason |
| **the deck records the ask; the output records the used** | which is the **asked vs effective** distinction the project already draws in `bench/result.py`, and `template.md` SS 6.4 draws for resolvers. Here it appears inside a plain science parameter, where nothing currently names it |
| **the snap point is system-dependent** | it follows from the cell, so the same ask gives a different used cutoff for a different box. Measured on one small system here -- the QUANTISATION is general, the numbers are not |

### 38.3 - What to check, and where it belongs

**Owner: `engines/tuning.md` SS 2.5** (the mesh-cutoff tier ladder) -- **not
read this pass.** What to look for: whether it says the value is a request and
the grid quantises it, and whether its tier ladder (150 / 300 / 500) is spaced
wide enough that each tier is genuinely a different grid. Wide tiers are
*right* under this finding -- the ladder's spacing may already encode it
without saying so.

**Also worth stating in `engines/siesta.md`'s mesh section:** *"the deck
carries what you asked for; the run reports what it used, and they differ."*
A user comparing a deck against an output and finding 301.7 vs 349.79 has no
way to know that is correct.

> **Recorded as a science finding rather than a defect.** Nothing is wrong in
> the code -- SIESTA is behaving as designed and molbuilder passes the value
> through faithfully. What is missing is that **no document says this**, and a
> user tuning a cutoff or reading a convergence study cannot infer it from
> anything molbuilder writes.

---

## 39 - SS 38 closed: `tuning.md` SS 2.6's ADVICE is right, its REASON is unstated

*(And my citation was wrong -- SS 2.5 is SCF tolerance; the mesh cutoff is
SS 2.6. Corrected above.)*

| tier | `MeshCutoff` |
|---|---|
| screening | 150 |
| loose preopt | 200-250 |
| publishable | **350** |
| tight (vib/phonons) | 500 (600 first-row) |

Plus: *"test by varying **+/-50 Ry**; the relative geometry should be stable
within your tolerance"*, and the shipped default of 300 is called *"one notch
below the 350 publishable recommendation"*.

### 39.1 - The advice already survives the quantisation

**The tiers are widely spaced** -- 150 / 200-250 / 350 / 500 -- so each tier is
almost certainly a genuinely different grid rather than the same one relabelled.
**And +/-50 Ry is a big enough step to cross snap points.** So a user following
SS 2.6 literally will not fall into the trap SS 38 describes.

**That is luck earned by good judgement, not by stating the mechanism.**

### 39.2 - What is missing, and what it costs

The section never says the value is a **request**, nor that SIESTA reports
*"(required, used)"* with the used cutoff higher. Consequences a reader cannot
derive from what is written:

| | |
|---|---|
| **why +/-50 and not +/-10** | the number reads as arbitrary. A careful user economising on compute steps 300 -> 320 -> 340, sees three identical energies, and concludes *converged at 300* -- having measured **one grid three times** |
| **the step is system-dependent** | snap points follow from the cell, so +/-50 is right for the systems the ladder was tuned on and may be too fine for a small box or too coarse for a large one. Without the mechanism a user cannot adapt it |
| **deck vs output** | a user comparing `MeshCutoff 350` in the deck against `Mesh cutoff (required, used) = 350.0 / 393.4` in the output has nothing that says the difference is correct |

### 39.3 - The fix, and it is two sentences

Add to `tuning.md` SS 2.6: **the cutoff is a request; SIESTA builds an integer
grid and reports the cutoff it actually used, which is higher. That is why the
convergence step is +/-50 Ry rather than a few Ry -- a smaller step can land
inside one grid and read as converged when nothing changed.**

And in `engines/siesta.md`'s mesh material: **the deck carries what you asked
for; the run reports what it used.**

> **Recorded as documentation-incomplete, not wrong.** SS 2.6's numbers are
> sound and its `+/-50` guidance is right *for the reason it does not give*.
> The gap only bites a user who reasons about the parameter rather than
> following the ladder -- which is exactly the user the tuning guide is for.

---

## 40 - NOTE: tuning guidance belongs in the template, as a UI hint

**User, 2026-08-14:** *"this should be provided at the UI hint. keep a note on
that. template is a good place to save such UI hints too."*

### 40.1 - The template already carries UI-facing keys

It is not a new capability -- SS 5's key set is already half presentation:

| key | answers |
|---|---|
| `label` | what to call it on screen |
| `null_label` | what *unset* is called -- *(auto)*, *(no cap)* |
| `unit` - `range` - `choices` | how to bound and constrain the control |
| `category` | which panel |
| `group` | whether *vary per stage* starts ticked |
| `help` | what this is, in prose |

So SS 38/SS 39's guidance -- *the cutoff is a request, the grid quantises it,
step +/-50 Ry not +/-10* -- has a home already: it is prose about the
parameter, and `help` is multi-line by design (SS 4.1 chose TOML partly for
that). Putting it there makes it reach **the template, the deck's verbose
comments, and the UI** from one place, which is the SS 3 *stated once* rule.

### 40.2 - The open question: is *what is this* the same key as *how do I choose it*

`help` currently answers **what the parameter is**. Tuning guidance answers
**how to pick a value** -- a tier ladder, a convergence step, when to go
higher. `engines/tuning.md` is a whole document of the second kind, and none
of it is in the template.

| | |
|---|---|
| **one key** -- put it all in `help` | no new vocabulary *(the user's rule: do not invent options)*. But `help` becomes long, and a UI showing it inline gets a paragraph where it wanted a sentence |
| **two keys** -- `help` (what) + a tuning hint (how) | a UI can show one inline and the other behind *more*; `tuning.md`'s per-parameter material gets a destination. Costs one key in a closed vocabulary |

**Not recommended either way yet** -- it wants the UI's actual shape, and the
UI is deliberately unstarted (SS 28). What is worth deciding early is only
**that tuning guidance is template data rather than document-only**, which the
user has now said.

### 40.3 - The work item

| | |
|---|---|
| **now** | record it *(this section)* |
| **with C1/C2** | decide one key or two |
| **then** | move `tuning.md`'s per-parameter guidance into the items -- starting with `mesh_cutoff`'s *request-not-delivered* caveat (SS 39.3), which is the case that exposed the gap |
| **check** | `tuning.md` becomes the *why* (the science, the citations, the cross-engine map) and the items carry the *how much* -- SS 0.1's alignment test applied to a document nobody has yet aligned |

### 40.4 - RULED: one key, and it may be formatted (user, 2026-08-14)

> *"one key is fine. this is a message, but format can contain bullet points,
> highlights etc to help user right?"*

**One key: `help`.** No new vocabulary -- SS 40.2's two-key option is closed.

**And it may be formatted.** It is a multi-line TOML string by design (SS 4.1
chose TOML partly so prose sits with the item it explains), so bullets, blank
lines and small tables are already expressible.

### 40.5 - The one constraint: three readers see it RAW

| consumer | renders as |
|---|---|
| the template file | **plain text** -- a person opens it (G5) |
| CLI `--help` (`cli.py:132` reads `metadata["help"]`) | **plain text**, in a terminal |
| *(later)* the UI | rendered |

> **So: structure travels, inline markup does not.** Bullets (`-`), blank
> lines, short labelled lines and a tier ladder read correctly in all three.
> `**bold**`, backticks and links show as literal characters in two of the
> three -- a terminal and the file itself.

**The guidance, one line:** *write `help` as if it will be read in a terminal,
because it will be. Use structure for emphasis, not markup.*

A worked shape, for the case that started this (SS 39.3):

```
help = """
The real-space integration grid, in Ry.  Higher is finer and slower.

This is a REQUEST: SIESTA builds an integer grid and reports the cutoff it
actually used, which is higher.  A band of requested values gives the same
grid and the same answer.

  150        screening only
  200-250    loose preopt
  350        publishable (semicore metals 400+)
  500        tight -- vib/phonons (600 first-row)

Converge by +/-50 Ry, not a few Ry: a small step can land inside one grid
and read as converged when nothing changed."""
```

Readable in a terminal, readable in the file, and a UI can style it. **No key
was added and no reader needs to change.**

---

## 41 - C4 PREPARED: prep already hands the wrapper eleven values. GPU is the exception

SS 25.4 framed C4 as *"should `prep` pass the resolved value, or should the
wrapper keep reading the deck?"* Measured, that is the wrong question --
**`prep` already passes almost everything.**

`write_run_wrapper` takes **eleven** named parameters, every one a value
resolved upstream and handed in:

```
env - mpi_np - omp_threads - max_memory_mb - time - gres - mem
cpus_per_task - exclusive - emit_sbatch - continue_retries
```

**Two things it does not receive, and re-derives from the deck instead:**

| read from the deck | by | why it is there |
|---|---|---|
| `Diag.ELPA.GPU` | `_fdf_requests_gpu` -- **eight call sites** | env routing, the gres ask, MPS, the NUMA pin, the rank/thread budget |
| `NumberOfAtoms` | `_parse_fdf_n_atoms` | clamps `mpi_np <= n_atoms` -- the propor IMAX=0 lower bound |

### 41.1 - So C4's real question

> **Why does this one value take a different road than the other eleven?**

And the answer is not *"because the deck is the truth"*, because that argument
is **already not applied consistently**: `mpi_np` is in the deck too --
BENCH-MARKS records it, and `render_config` puts it there precisely so the deck
*"records the rank count it actually assumed"* -- and it is **passed**, not
parsed.

`n_atoms` is a genuinely different case and worth separating: it is a fact
about the **structure**, not a resolved parameter, and the wrapper needs it at
install time. Reading it from the deck is the structural seam (C6), not this.

### 41.2 - What the contract says, and it points both ways

| | |
|---|---|
| `project-layout.md` SS 2.3.1 | *"the wrapper's environment is chosen by **a value the deck decides**"* -- and calls step 4 following step 3 **forced** |
| `template.md` SS 6.1 | `read_by` exists so the wrapper is **told** which items it depends on, rather than re-deriving |
| the code | `check_launch_matches_deck` exists **because the deck and the launch can disagree** -- a reconciliation nobody would need if one of them were simply authoritative |

**The deck-is-truth reading has one real argument behind it:** a person may
hand-edit the deck, and the deck is what runs. But that argument applies to
`mpi_np` identically, and `mpi_np` is passed.

### 41.3 - The options, and what each costs

| | |
|---|---|
| **(a) pass it, like the other eleven** | one road for every resolved value. `_fdf_requests_gpu` survives as the **standalone** fallback (`write_run_wrapper` is called directly on a bare deck outside `prep`, which is why the parameters are all Optional). Consistent -- and it makes `enable_gpu`'s `read_by` declaration mean something, which today it does not |
| **(b) keep parsing, and parse `mpi_np` too** | *deck is truth*, applied consistently. Reverses eleven parameters and re-derives what `prep` already resolved -- the habit `architecture.md` SS 1 exists to remove |
| **(c) leave it** | the asymmetry stays undocumented, which is how it survived this long |

**Recommendation: (a).** It is the smaller change, it matches what the other
eleven values already do, and it is the only option under which SS 6.1's
`read_by` is a mechanism rather than a declaration nothing reads.

*(No code. SS 26 holds -- and note this is the honest remainder of T8: the
declaration landed, the hand-off did not.)*

### 41.4 - Why GPU is different, and it is not arbitrary (user, 2026-08-14)

> *"it's because it is up to the user to decide if gpu should or should not be
> used and it also need to be benchmark tested to see if gpu is used what's the
> best np/blocksize etc for using with it"*

**That corrects SS 41.1.** I asked why one value takes a different road than the
other eleven, as though the roads were interchangeable. They are not -- they
carry **two different kinds of thing**, and `prep` holds both:

| | what it is | where it lives on the element | who decides it |
|---|---|---|---|
| the eleven | the **allocation** -- what was asked of the scheduler and granted | `ResolvedConfig.resources` | the machine + your ask, resolved at `prep` |
| **`enable_gpu`** | a **template item** -- a choice about how the calculation runs | `ResolvedConfig.values` | **the user**, in the description, before any machine is known |

**So the asymmetry is not *one value versus eleven*. It is: the wrapper is
handed the ALLOCATION and is handed nothing from the CONFIG.** Every one of the
eleven comes from `Resources`; `enable_gpu` lives in `values`, and that road
does not exist.

**And `read_by` is precisely the declaration of that missing road** --
*"which items of the CONFIG does the wrapper derive from"* (SS 6.1). It has no
mechanism behind it because the mechanism it names is the one absent road.

### 41.5 - The benchmark half, which makes it upstream rather than parallel

GPU is not merely *another* execution knob: **it is a choice the benchmark is
run under.** The user picks it, and the sweep then answers *given that*, what
`mpi_np` and `BlockSize` are best -- which is why `bench`'s grid is
`(GPUs, ranks-per-GPU, cores-per-rank)` and why `_auto_block_size` takes
`gpu_mode` as an input.

```
user chooses      enable_gpu          (template item, floor 2)
      ->
benchmark sweeps  np x blocksize      GIVEN that choice
      ->
run asks for      the allocation      mpi_np, gres, cpus_per_task ...
```

*(Drawn with arrows rather than `|`: the table gate scans every line beginning
with a pipe and does not skip fenced blocks, so an ASCII diagram using them
reads as an orphaned table row. Same class as the citation gate's inability to
tell a quoted example from a live one -- SS 12.3.)*

So `enable_gpu` is **upstream of the allocation**, not a member of it -- which
is why it is correctly NOT `allocation`-tagged (SS 31.3 kept it that way) and
why it belongs in the description that travels.

### 41.6 - What C4 therefore decides

Not *"pass it or parse it"* but: **does the config -> wrapper road get built?**

| | |
|---|---|
| **build it** | `prep` passes the items whose `read_by` names the wrapper, from `element.values`, beside the allocation it already passes. `read_by` becomes a mechanism. The deck scan stays for standalone use |
| **leave it** | the wrapper keeps re-deriving a user choice by parsing the artifact that choice produced -- and `read_by` stays a declaration nothing reads |

**Recommendation unchanged, reason improved:** build the road. Not for
consistency with the eleven -- they are a different kind -- but because a
**user's decision** should reach the layer that acts on it by being *handed
over*, not by being *recovered from a rendered file*.

---

## 42 - `engines/stages.md` SS 4 read: clean, except one claim that touches C4

**Clean, and cross-checked against what was read earlier:**

| SS 4 says | agrees with |
|---|---|
| `effective config = the template's values (+) that stage's overrides` | `resolve.resolve`'s precedence, implemented in that order |
| the format is `template.md`'s -- one `tomllib.load`, an ordinary `SiestaConfig` out | `config_from_template` |
| **`base` is removed from `task.json`** -- the file carries what CHANGES | measured: `Task` has `varies` and `stages`, no `base` |
| R1 one object validated and rendered - R2 a stage validated as a resolved whole - R3 the sequence checked too | `resolved_ladder` exists for R3's caller |

**It does NOT say where the (+) operator LIVES** -- only what it computes. So
C3 is genuinely open rather than already answered somewhere I had not read.

### 42.1 - The claim: *"nothing has to parse an `.fdf` -- which nothing in molbuilder can do"*

SS 4 makes that the argument for the template being TOML: `prep` reads values
with one `tomllib.load` **because molbuilder cannot read a deck**.

**Two places in molbuilder parse a deck today:**

| | reads |
|---|---|
| `runwrap._fdf_requests_gpu` | `Diag.ELPA.GPU`, by regex -- eight call sites |
| `runwrap._parse_fdf_n_atoms` | `NumberOfAtoms`, to clamp `mpi_np <= n_atoms` |

And `job-contracts.md` SS 3.1 **designs the reserved blocks to be parsed** --
*"parsers find blocks by MARKERS"* -- with SS 3.6 promising ATOM-METADATA
round-trips.

### 42.2 - Both are true, and the wording hides which

| reading | true? |
|---|---|
| *molbuilder has no GENERAL FDF parser -- it cannot reconstruct a config from a deck* | **yes**, and this is SS 4's actual argument. A regex for one keyword is not a parser |
| *nothing in molbuilder reads anything out of a deck* | **no** -- two targeted readers, plus a whole block format designed for reading |

**Why it matters for C4.** Read absolutely, SS 4 forbids the deck scan C4 is
deciding about. Read as intended -- *no general parser* -- it forbids nothing
and the scan is a targeted read like `NumberOfAtoms`. **A contract that
settles a live design question by accident, through wording rather than
intent, is worth one clause of repair:**

> *nothing has to parse an `.fdf` for its PARAMETERS -- molbuilder has no
> general FDF reader, and never will. Targeted reads of a named keyword or a
> reserved block are a different thing (`job-contracts.md` SS 3.1).*

**Core document: `engines/stages.md` SS 4.** What to look for beyond the line:
whether the same absolute phrasing recurs anywhere the deck's readability is
argued -- SS 9.2 of `template.md` makes a related argument about `prep` not
harvesting from disk, and that one IS about reproducibility rather than
capability.

---

## 43 - SPIN: RULED, and it was ruled earlier than the record said

**User, 2026-08-14:** *"i thought we are done with spin. what's wrong with
it?"* -- **nothing. It was settled and I did not write it down**, then kept
listing it as open in SS 32.3 and SS 36.2. Corrected here; those entries are
struck.

### 43.1 - How it was settled

The user asked: *"you can have a spin flag and an engine-specific spin_param
both in the template?"* Checking the two engines' help text answered it
better than the question expected:

| | says |
|---|---|
| SIESTA `spin_total` | *"target total spin moment in mu_B (= **number of unpaired electrons**)"* |
| PySCF `spin` | *"2S (NOT 2S+1); 0=closed shell, 1=doublet, 2=triplet"* -- also the **number of unpaired electrons** |

**Same quantity, same units, same number** -- not two conventions.

**And there are THREE states, not two**, which is why the flag is not
derivable from the number:

| state | SIESTA | PySCF |
|---|---|---|
| closed shell | *(nothing emitted)* | `spin=0`, RKS |
| **open shell, moment FREE** | `SpinPolarized .true.`, **no** `Spin.Fix` | `spin=0`, **UKS** (broken symmetry) |
| open shell, moment FIXED | `SpinPolarized` + `Spin.Fix` + `Spin.Total <v>` | `spin=<v>` |

`spin != 0` expresses only two of them. The middle -- *let it polarise and
find its own moment* -- is a real and common choice, and it disappears if the
flag is derived from the count.

### 43.2 - The ruling

**Two items, both SHARED across engines.** The flag and the number are each
the same question with the same answer for both engines, so both merge under
SS 6.3's test; what differs is only the rendering, which is the generator's
(`kind = "deck"`).

```toml
[item.spin_polarized]        # the shared QUESTION -- is this open-shell?
kind     = "deck"
category = "system"
engines  = ["siesta", "pyscf"]
type     = "bool"
value    = false

[item.spin_moment]           # the shared NUMBER -- unpaired electrons, when fixed
kind     = "deck"
category = "system"
engines  = ["siesta", "pyscf"]
type     = "float"
# NO value = polarised but UNCONSTRAINED -- the third state
```

**Each generator renders it:** SIESTA emits `SpinPolarized` / `Spin.Fix` /
`Spin.Total`; PySCF picks `RKS` vs `UKS` and sets `gto.M(spin=)`.

**And it uses `unset` the way the template already defines it** -- *explicitly
unset* is the third state, not a missing value (SS 3). No new mechanism.

### 43.3 - What this deletes

`spin_polarized` **stays** (it is the flag), and **SIESTA's `spin_total` and
PySCF's `spin` become one item.** Two names for one quantity -- exactly the
*"same physics unrecognisable across engines"* defect SS 1 of the unification
plan measured.

> **The real lesson is the record-keeping.** The design was settled in
> conversation and left open in the document, so it read as unresolved for
> two more sections. That is the same failure this report charges elsewhere --
> a decision made once and not written where the next reader looks.

---

## 44 - ~~The reserved blocks ARE the user's data -- three places disagree~~ **RETRACTED (SS 47)**

> **⛔ THIS SECTION IS WRONG AND IS RETRACTED -- see SS 47.** The banner is
> correct as written; I read *"part of the calculation"* loosely and
> manufactured an inconsistency. Left in place rather than deleted, because a
> retracted finding that vanishes reads later as one nobody checked.

**User, 2026-08-14:** *"the reserved blocks is the user customized data, so
you should make the document consistent."*

### 44.1 - The disagreement, in the three places that state it

| where | says | consistent? |
|---|---|---|
| `script_emit.machine_record_banner`'s **docstring** | *"how molbuilder reads the file BACK -- provenance, the benchmarking anchors, and **the per-atom labels that reconstruct the structure**"* | **yes** -- this is the persistence channel, named exactly |
| the **emitted banner**, in every deck | *"MOLBUILDER RECORD -- everything below this line is written and read by molbuilder, and **is NOT part of the calculation**"* | **no** -- ATOM-METADATA sits below that line |
| `template.md` SS 9's table | ATOM-METADATA's content comes from *"the structure and its `.molstruct.json` sidecar"* | **yes** -- it is the user's own data |

### 44.2 - Why the emitted wording is the wrong one

Below that banner sit the **regions, the frozen set, and the electrode /
bridge labels**. They are:

* **the user's**, not molbuilder's -- they come from the structure the user
  built and annotated;
* **part of the calculation's identity** -- the frozen set decides what
  relaxes, and a relaxation rewrites coordinates while the labels are what
  still say which atom is which afterwards;
* **required by a later calculation** -- transport reads electrode / bridge /
  frozen membership back out (SS 33, R15).

*"Not part of the calculation"* is true of PROVENANCE (a generation snapshot)
and defensible for BENCH-MARKS (derived bounds). It is **false of
ATOM-METADATA**, and the banner covers all three.

### 44.3 - What is actually true, and it is the honest framing

> **Below the banner is the PERSISTENT RECORD: what molbuilder writes so it
> -- or a later calculation -- can read this file back.** Some of it is about
> the generation (PROVENANCE), some is derived bounds (BENCH-MARKS), and some
> is **the user's own structural data** (ATOM-METADATA), which is why it
> round-trips (`job-contracts.md` SS 3.6). None of it is a SETTING -- editing
> it does not change what the engine computes -- and that is what the banner
> should say, rather than that it is not part of the calculation.

**The distinction the banner wants is *setting* vs *record*, not *calculation*
vs *not-calculation*.** The docstring already draws it correctly: *"those are
data, not settings"*.

### 44.4 - The three edits, and one of them changes deck output

| # | where | change |
|---|---|---|
| 1 | `script_emit.machine_record_banner` -- **the emitted text** | *"is NOT part of the calculation"* -> *"is not a SETTING: molbuilder writes it so this file can be read back, and editing it changes nothing the engine computes"*. **This changes every generated deck**, so it needs its own decision and a T10-style re-baseline |
| 2 | `job-contracts.md` SS 3.1 | the mermaid node labels the banner *"data about the file; not hand-edited"* -- true, but it should say the region below **includes the user's structural data**, which is why SS 3.6 promises it round-trips |
| 3 | `template.md` SS 9 | already correct after SS 34's edit; add the one line that the region below the banner is the **persistence channel**, so all three read alike |

> **Only edit 1 touches code, and it touches OUTPUT.** SS 26's rule applies
> with extra force: it is a one-line string whose change alters every deck
> molbuilder has ever written, so it is a contract decision first and a commit
> second.

---

## 45 - C2 RULED, and PySCF benchmarks too -- which the seam already allows

**User, 2026-08-14:** *"Yes. And pySCF have similar bench to test these
parameters too."*

### 45.1 - The ruling

> **A parameter is `allocation` when its VALUE is a machine fact, whatever
> engine asks it.** Not when a particular engine's schema happens to tag it.

| item | change |
|---|---|
| **PySCF `threads`** | **becomes `allocation = True`, `resolver = "omp_threads"`** -- the same question SIESTA's `omp_threads` asks, answered the same way |
| SIESTA `mpi_np`, `omp_threads` | unchanged -- already correct |
| `memory_cap_mb` (SIESTA) | allocation, **no resolver** -- unset means *do not cap* (SS 30.1) |
| `working_memory_mb` (PySCF) | **not** allocation -- it selects in-core vs out-of-core, so it is science. `resolver = "node_memory"` (SS 30.1) |

**The behavioural effect:** a PySCF template stops carrying a thread count and
`prep` fills it from the node -- exactly as a SIESTA template already behaves.
Floor 2's *never assert a machine fact's value* stops being enforced on one
engine only (SS 31.2).

### 45.2 - PySCF benchmarks too, and its axes are NOT the same axes

The shipped sweep is **`(GPUs, ranks-per-GPU, cores-per-rank)`**
(`bench/grid.py::sweep_grid`), rendered `G1K4C6` -- **MPI-shaped, and PySCF
has no MPI ranks.** It is one process with threads, optionally on a GPU.

So PySCF's sweep is a different set of axes over the same machinery:

| engine | plausible axes | why |
|---|---|---|
| SIESTA | GPUs - ranks-per-GPU - cores-per-rank | MPI + OMP, and `BlockSize` on top |
| **PySCF** | **threads** - **use_gpu** - and arguably **`working_memory_mb`** | one process; and working memory selects in-core vs out-of-core, which is **speed, not the answer** -- so it is legitimately `category = "execution"` and legitimately sweepable |

### 45.3 - The seam already allows this, and that is the point

`resolve.MachineTranslation` is **exactly this**: *the specialisation's
coupling -- which axes are ours, and what machine ask each point implies* --
declared as data:

```
MachineTranslation(axes=("G", "K", "C"),
                   to_resources=lambda point, env: {...})
```

Its docstring already says the axes are **declared, not inferred**, *"so the
resolver can refuse an axis nobody owns by name"*. **PySCF supplies its own
instance and nothing in `resolve/` changes** -- which is `generator.md` SS 7's
seam test passing on a case nobody has exercised yet.

> **So this is a confirmation, not a new requirement.** The one thing to check
> when PySCF's bench lands: that `bench/grid.py::sweep_grid` -- which
> enumerates the G/K/C grid from probed topology -- is **the SIESTA
> specialisation** and not shared code that a second engine has to bend around.
> If it is shared, that is the leak, and it is findable before writing a line
> of PySCF bench.

### 45.4 - What this adds to the work list (SS 25)

| # | item | blocked on |
|---|---|---|
| **25.7** | `bench/grid.py::sweep_grid` -- is it SIESTA's specialisation or shared? | nothing; it is a read, and it decides whether PySCF's bench adds a file or edits one |
| **25.8** | PySCF's `MachineTranslation` -- its axes, and what each point implies for `Resources` | C2 (ruled here), since `threads` becomes the allocation field its points must land on |

---

## 46 - `sweep_grid`'s K is RANKS-PER-GPU, not k-points -- a name that invites the wrong reading

**User, 2026-08-14:** *"is this sweep_grid the old k_grid related sweep? we've
already agreed that k_grid is not a calculation setup but a scientific
judgement the user need to make so it is not part of bench any more."*

### 46.1 - Measured: it is not, and never was

```
def sweep_grid(gpn, cps, ks, cs_explicit):
    for g in range(1, gpn + 1):        # G = GPUs per node
        for k in ks:                   # K = RANKS PER GPU
            for c in (cs_explicit or _bracket_cs(cps, k)):   # c = cores/rank
                yield (g, k, c)
```

`G1K4C6` = **1 GPU, 4 ranks per GPU, 6 cores per rank**. Three resource axes.
No Brillouin-zone sampling anywhere in it.

### 46.2 - But the reading it invites is the dangerous one

In a SIESTA context **`K` reads as k-points**, and `kgrid` is a real
`SiestaConfig` field (`type = "int3"`, emitted as
`%block kgrid_Monkhorst_Pack`). A person seeing `bench-G1K4C6` in a directory
listing has every reason to think the benchmark swept k-points -- **exactly
the error the user is guarding against**: a scientific judgement measured as
if it were a resource.

**The contract already agrees on the substance.** `template.md` SS 6.2 puts
`kgrid` in **`accuracy`**, not `execution`, and only `execution` is the
sweepable set -- *"mesh_cutoff changes speed too, but it changes the answer,
so sweeping it measures a different calculation each time."* The same argument
covers `kgrid`. **Nothing in the code sweeps it.**

### 46.3 - So the finding is the NAME

| | |
|---|---|
| **right** | the mechanism, the axes, and the category rule keeping `kgrid` out of the sweepable set |
| **misleading** | one letter -- in a token that appears in **directory names on disk** (`bench-G1K4C6`), in `job-contracts.md` SS 6.3's worked example, and in every summary a person reads |
| **cost of leaving it** | a reader concludes the benchmark sweeps k-points, and either trusts a result that does not exist or distrusts the benchmark that does |

**Two options, and the second is nearly free:**

1. **Rename the axis** `K` -> `R` (ranks per GPU): `bench-G1R4C6`. Truthful,
   but the token is a **directory name and a SystemLabel suffix**, so it
   changes paths on disk and anything reading them.
2. **Say what the letters mean where the token is defined.**
   `bench/grid.py`'s docstring spells it; `job-contracts.md` SS 6.3 -- the
   **cross-layer authority for every name** -- renders `bench-G1K4C6` without
   saying what G, K and C are. One clause there, and in the summary output.

**Recommendation: (2) now; (1) only if the token is revised for another
reason.** The letters are load-bearing on disk; the ambiguity is fixable with
prose, and SS 6.3 is exactly where a reader goes to decode a name.

> **Recorded because the question was the right one to ask.** The mechanism is
> correct and the contract does keep `kgrid` out of the sweep -- but a design
> whose name suggests it does the forbidden thing will be asked about again,
> and the next asker may not check.

### 46.4 - SS 45.3's open check is ANSWERED in principle (user, 2026-08-14)

> *"so since we agreed benchmark also is a workflow for pyscf this solves your
> question i believe"*

**It does.** SS 45.3 asked whether `sweep_grid` is SIESTA's specialisation or
shared code a second engine must bend around. The answer follows from a split
the contract already makes -- `generator.md` SS 2.3.1a: **benchmarking is
`prep` whose parameters are a set rather than a point.** The **five steps are
the framework**; the **grid is the specialisation**.

So: **a shared bench workflow with a per-engine grid is the design**, and
`sweep_grid` being `(G, K, c)`-shaped is *correct* as SIESTA's specialisation.
It is not a design question.

**What remains is factual, and it is SS 25.1's shape exactly:** `sweep_grid`'s
own docstring names its consumer as *"`jobset prep bench`'s `_bench_inputs`"*
-- a **shared floor calling a SIESTA-shaped grid directly**. The
specialisation is right; its address is not, in the same way
`effective_config`'s body is right and its address is not.

**So work item 25.7 changes from a QUESTION to a MIGRATION STEP:**

| | |
|---|---|
| ~~is `sweep_grid` a specialisation or shared?~~ | **answered: it is a specialisation** |
| **25.7 (revised)** | move it behind the engine seam, so `jobset prep bench` asks the engine for its grid instead of importing SIESTA's. The seam already exists -- `EngineSeam` (SS 35.2's target shape) and `MachineTranslation` (SS 45.3) -- so this adds a member, not a mechanism |

> **And it is findable before PySCF's bench is written**, which was the point
> of asking: a second engine arriving at a shared caller that imports the
> first engine's grid is how `siesta/` came to be reachable from `resolve/`.

---

## 47 - SS 44 RETRACTED: the banner is true, and the distinction is the useful part

**User, 2026-08-14:** *"the banner is true. it is not part of the calculation
related to the script. i see nothing wrong. the comment and customized block
are for information carry on only."*

**Correct, and SS 44 is withdrawn.** I conflated two different things:

| | |
|---|---|
| *part of **this** calculation* | what the engine computes from, when it runs **this script**. That is the **ENGINE BODY** |
| *used by a **later** calculation* | what the file carries forward. That is everything below the banner |

**The banner claims the first, and it is right.** The frozen atoms reach the
engine through `%block Geometry.Constraints` **in the engine body** -- that is
the part that changes what SIESTA computes. ATOM-METADATA is the *record* of
the labels, for carrying forward. Hand-editing it changes nothing the engine
does; it only makes the file unreadable to the tool that wrote it, which is
what the banner's docstring says in as many words: *"those are data, not
settings."*

### 47.1 - The distinction, stated once so it is not re-muddled

```
ENGINE BODY        the calculation.  The engine reads this.
                   Frozen atoms arrive HERE, as Geometry.Constraints.

--- banner ---

PROVENANCE         a generation snapshot
BENCH-MARKS        derived bounds a tool may override
ATOM-METADATA      the structural labels, CARRIED FORWARD
```

**And SS 33 is unaffected.** *The blocks are a persistence channel* and *the
blocks are not part of this calculation* are **both true** -- carrying
information to a later run is precisely not being an input to this one. The
banner separates settings from records; SS 33 says why the records matter.
There was never a conflict.

### 47.2 - What this retracts

| SS 44 proposed | status |
|---|---|
| edit 1 -- change the emitted banner text | **withdrawn.** It is correct, and changing it would have altered every deck for no reason |
| edit 2 -- `job-contracts.md` SS 3.1's mermaid | **withdrawn** |
| edit 3 -- one line in `template.md` SS 9 | **withdrawn** -- SS 34's edit already says what is true |

> **The error is the one this report keeps charging elsewhere, made by me for
> the fourth time today: reading a precise statement loosely and calling the
> result an inconsistency.** SS 24 recorded three; this is the fourth, and the
> only one that would have changed shipped output. It is also the reason SS 26's
> contract-first rule earns its place -- had I *fixed* SS 44 instead of
> recording it, every deck would now differ.

### 47.3 - The one-line version: **the engine does not read comments**

*(User, 2026-08-14.)* Every reserved block is `#`-prefixed. SIESTA skips
comments; PySCF's are Python comments. **So they are not part of the
calculation by construction, not by argument** -- and SS 44 did not need a
nuanced case, it needed this sentence. Recorded because it is the shortest
true statement of the rule and the one to reach for next time.

---

## 48 - `pow2` IS power-of-two, it IS meaningful, and nothing can emit it

*(User, 2026-08-14: "is the pow2 'power of 2'?")*

**Yes.** It exists for `BlockSize` -- the ScaLAPACK / ELPA distribution block
-- where powers of two are the convention: `_auto_block_size` caps to one, and
a generated deck's PROVENANCE prints *"auto -> 256 (n_orbitals_est 2120 /
mpi_np, **capped pow2**)"*. `template.md` SS 5 gives it the right reason: `type`
carries *"what a parser cannot know -- that `pow2` must be a power of two"*.

**So the type is meaningful. The defect is that nothing produces it.**

| | |
|---|---|
| `_decl_type` | maps annotations to `enum` / `bool` / `int` / `float` / `str` / `int3` / `strlist` / `intlist`. **No branch yields `pow2`** |
| `parallel_block_size` | is `Optional[int]`, so it renders `type = "int"` |
| `template.md` SS 12's example | declares `type = "pow2"` |

**The contract shows a type the code cannot emit**, and Gate A does not catch
it -- the example's type IS in `TYPES`, so it passes. It is valid and
unproducible.

### 48.1 - The fix is small and the choice is real

| | |
|---|---|
| **make it reachable** | a field declares it in metadata -- e.g. `"decl_type": "pow2"` -- and `_decl_type` honours it. Then `parallel_block_size` carries `pow2`, a surface can refuse 96, and SS 12's example becomes true. **This is the option that matches what the value IS** |
| **retire it** | drop `pow2` from `TYPES`, and SS 12's example becomes `int` with `range`. Loses the one thing SS 5 says `type` is for -- a constraint a parser cannot see |

**Recommendation: make it reachable.** A power-of-two constraint is exactly
SS 5's stated purpose for the `type` vocabulary, `BlockSize` is a real field
that wants it, and the alternative is to delete a correct idea because the
plumbing was never finished.

> **And it is the same shape as `text` (SS 33): zero uses because the work is
> not done, not because nobody wants it.** Two of the four "unused" vocabulary
> members are now in that category -- which is the answer to SS 32's question
> and leaves only `intlist` and the `monitor` kind genuinely open.

---

## 49 - `pow2`'s only parameter, and the `int3` / matrix question

### 49.1 - Which parameters need the pow2 rule: **one**

Measured across both schemas. Two fields mention powers of two:

| | |
|---|---|
| **`parallel_block_size`** | the genuine case -- ScaLAPACK's distribution block, `_auto_block_size` caps to a power of two, PROVENANCE prints *"capped pow2"* |
| `mpi_np` | its help mentions base-2, but it is **allocation-tagged and valueless in the template**, and rank counts are not strictly powers of two -- they follow the core count's divisors |

**So `pow2` exists for exactly one field.** That is `strmap`'s shape -- and
this time it is *correct*: SS 5's stated purpose for `type` is a constraint
**a parser cannot see**, and *"is a power of two"* is precisely that. A
one-field type is not automatically wrong; a one-field type that models
**storage** (`strmap`) is.

### 49.2 - `int3` models MOLBUILDER's simplification, not SIESTA's parameter

*(User: "for kgrid and unit_cell ... can we just make a matrix?")*

**SIESTA's k-grid is a matrix.** The emitted block, from a real deck:

```
%block kgrid_Monkhorst_Pack
1 0 0 0.0
0 1 0 0.0
0 0 1 0.0
%endblock kgrid_Monkhorst_Pack
```

A **3x3 integer matrix plus a displacement column** -- the general
Monkhorst-Pack form, which allows non-diagonal supercell sampling. molbuilder
exposes **only the three diagonal entries**, as `int3`.

> **So `int3` is the same defect shape as `strmap`:** a type that models what
> molbuilder happens to store rather than the question the engine asks. The
> difference is that `strmap` narrowed a question into a storage blob, while
> `int3` narrows a matrix into its diagonal -- **a capability silently not
> offered**, which SS 7's *"a keyword molbuilder does not model is work not
> done yet"* names exactly.

### 49.3 - Is the cell a parameter? Today: **no, and by an explicit rule**

Measured: neither config has a cell field. It lives on `Structure` --
`cell`, `pbc`, `axis_kind`, `vacuum`, `cell_origin` -- and
`generator.md` SS 10.2 makes that a **class-2 structure fact**: *"the cell, the
vacuum, frozen atoms and regions live on the structure, not in any engine's
config -- one structure, every surface seeing the same facts."*

**The user's instinct that it should be a parameter is not obviously wrong,
and it is a contract decision:**

| | |
|---|---|
| **stays a structure fact** *(today)* | one home, travels with the structure + sidecar, `cell.py` resolves it once. A template item would put the same fact in two files -- the duplication SS 4.1 rejects |
| **becomes a parameter** | a user could set or override it per calculation -- strain, a different vacuum, a fixed lattice -- without editing the structure. But then which wins when they disagree, and does a relaxed cell flow back? |

**The question underneath is whether the cell is *geometry* or a *setting*.**
Atom positions are unambiguously geometry and nobody proposes templating them.
The cell is geometry too -- but unlike positions it can be **unset and
resolved** (vacuum -> box), and a resolution is a decision, which is what
makes it feel parameter-like.

### 49.4 - What a `matrix` type would need, if it lands

| use | shape | element |
|---|---|---|
| k-grid | 3x3 **+ a 3-vector displacement** | int (matrix), float (displacement) |
| cell | 3x3 | float |

**They are not the same shape**, so one `matrix` type needs shape and element
declared per item -- more machinery than `int3`. Worth it **only if the
general k-grid form is actually offered**; if molbuilder stays diagonal-only,
`int3` is honest and `matrix` is speculative.

> **So SS 49.2 is the question to answer first, and it is a science one:**
> does molbuilder want to offer non-diagonal Monkhorst-Pack sampling? If yes,
> `int3` must go and `matrix` earns its place. If no, `int3` is correct and
> the contract should **say** that only the diagonal form is offered -- which
> today it does not.

---

## 50 - `pow2` is NOT enforced on the path that needs it, and `monitor` has no record

### 50.1 - Measured: a user's non-power-of-two block size passes through

*(User: "for blocksize -- isn't it a right thing to enforce the format after
however it is produced?" and "if the code takes care of it, i am ok. but keep
it next to blocksize could remind the code to make sure the value has a
special requirement.")*

**The code does not take care of it.**

```
parallel_block_size = 64   ->  BlockSize 64
parallel_block_size = 96   ->  BlockSize 96      NOT a power of two
parallel_block_size = 100  ->  BlockSize 100     NOT a power of two
```

`_auto_block_size` caps **its own proposal** to a power of two -- PROVENANCE
prints *"capped pow2"* -- but a **user-supplied** value is emitted verbatim,
and a benchmark-pinned one takes the same path. **The constraint lives in the
auto branch and nowhere else**, which is the branch least likely to break it.

### 50.2 - So the user's reasoning holds, and it is the stronger form

> *"enforce the format after however it is produced"*

That is the right shape: the value has three producers -- the auto rule, a
person, and a benchmark pin -- and the constraint belongs **after** all three,
not inside one. `resolve.py`'s precedence already funnels them to a single
point (template -> stage -> sweep -> **pin**), so there is one place where the
effective value exists and can be checked.

**And declaring `pow2` next to the field earns its place exactly as the user
says** -- *"keep it next to blocksize could remind the code to make sure the
value has a special requirement"*. Today nothing anywhere states the
requirement except a comment inside the auto branch and a help line.

| | |
|---|---|
| **declare `pow2`** | the requirement becomes **data on the item**, so a surface can refuse 96 before the deck exists, and `_check_raw_value` already knows how to check it -- `_TYPE_CHECKS["pow2"]` exists and is correct |
| **enforce after resolution** | the effective value is checked once, wherever it came from |

> **The two are the same fix.** `_TYPE_CHECKS["pow2"]` is already written and
> already unreachable -- it validates a type nothing declares. Declaring the
> type turns an existing, tested checker on. **Nothing new is built.**

### 50.3 - `monitor`: no field, and no design record found

*(User: "what's monitor designed for? any record?")*

Measured: **no field in either config declares `item_kind = "monitor"`.**

What the contract says about it, in full -- this is all of it:

| where | says |
|---|---|
| `template.md` SS 6's kind table | *`monitor` -- the item "shapes what the monitor writes"; reaches the deck: **no**; who acts on it: the monitor* |
| SS 8's reader table | *the monitor - what it should write - filters `kind == "monitor"`* |
| SS 8.0's call table | `select(t, kind="monitor")` |

**That is the entire record.** Three lines, all restating the same sentence,
and none saying **what a monitor item would BE**.

The monitor itself is real and does real work -- `monitor.py` samples node CPU
percent, memory in use, and per-GPU SM/memory utilisation through the run
(`generator.md` SS 4.4). So plausible monitor items exist: **a sampling
interval, which metrics to record, whether to sample GPUs at all.** None is a
config field today.

> **So `monitor` is the fourth reserved-for-work-not-done member**, beside
> `text` (SS 33), `pow2` (SS 48) and arguably `intlist`. The difference: `text`
> and `pow2` have a **named field waiting** (`user_custom`, `block_size`), and
> `monitor` has **no candidate at all** -- which makes it the one member whose
> retirement would cost nothing today. Recorded as the honest state rather
> than a recommendation: the monitor is a real layer, and a kind reserved for
> a layer that exists is different from one reserved for nothing.

---

## 51 - K-GRID: what SIESTA offers vs what molbuilder exposes *(user order)*

**User, 2026-08-14:** *"for k-grid, i suggest we fully understand what siesta
needs and properly present it and also if needed redesign UI to properly
reflect this full details."*

### 51.1 - What SIESTA offers -- read from the shipped binary, not from memory

| keyword | what it specifies |
|---|---|
| `kgrid_Monkhorst_Pack` / `kgrid.MonkhorstPack` | the **3x3 integer matrix + a displacement per row** -- the general Monkhorst-Pack form |
| `kgrid_cutoff` / `kgrid.Cutoff` | a **real-space length**; SIESTA derives the mesh from it and the cell |
| `kgrid.File` | read the grid from a file |
| `__ts_kpoint_scf_m_MOD_process_k_cell_displ` | TranSiesta **forcing** the transport direction to one k-point -- it zeroes a row and column of the k_cell matrix and sets that displacement to 0 |

*(That last symbol is the function whose miscompilation under gcc 14.4 broke
the clean-machine build -- so the matrix and the displacement are not
theoretical: they are load-bearing machinery that transport depends on.)*

### 51.2 - What molbuilder exposes: the diagonal, Gamma-centred, only

```
%block kgrid_Monkhorst_Pack
1 0 0 0.0        <- kgrid[0], and the displacement is HARD-CODED 0.0
0 1 0 0.0
0 0 1 0.0
%endblock kgrid_Monkhorst_Pack
```

`kgrid` is `Tuple[int, int, int]` -- three diagonal entries. **Three things
are not offered at all:**

| missing | why it matters |
|---|---|
| **the displacement (shift)** | Gamma-centred vs shifted is a real scientific choice: for even meshes a 0.5 shift samples better, while Gamma-centred is required for some symmetries **and for transport**. molbuilder always writes `0.0` and nothing says so |
| **the non-diagonal matrix** | sampling commensurate with a supercell. A diagonal mesh on a supercell is not wrong, but it is not the mesh a user may want |
| **`kgrid.Cutoff`** | arguably the **better default interface**: it is cell-size aware, so one number gives consistent sampling density across systems of different sizes -- which is what a user comparing structures actually wants |

### 51.3 - Why this is more than a missing knob

`template.md` SS 7 already names the rule: *"a keyword molbuilder does not
model is **work not done yet**, and the answer is to model it."* And SS 49.2
found that `int3` **is the type that encodes the omission** -- it models the
diagonal molbuilder stores, not the parameter SIESTA takes.

> **So the k-grid is the worked example of the whole `strmap` lesson:** a type
> chosen for what molbuilder happened to store, which then makes the missing
> capability invisible. Nobody reading `type = "int3"` would guess a
> displacement column exists.

### 51.4 - The work, in order, and none of it is code yet

| # | step | owner |
|---|---|---|
| 1 | **Establish what SIESTA does with each form** -- especially the displacement's effect and how `kgrid.Cutoff` maps to a mesh. Read `Src/kgrid.F` / `find_kgrid.F` (both named in the binary) | science, then `engines/siesta.md` |
| 2 | **Decide what molbuilder offers**: full matrix + displacement, or diagonal + displacement, or cutoff-based, or a choice of forms | contract -- `engines/siesta.md` SS 6 (Lattice & k-grid) |
| 3 | **Then the type follows** -- SS 49.4's `matrix` earns its place only if (2) offers the matrix; a diagonal + displacement wants a different shape again | `engines/template.md` SS 5 |
| 4 | **Then the UI** -- the user asked for it to reflect the full detail, and it can only do that once (2) is decided | deferred, per SS 28 |

> **Step 1 is a READ, and it is the one nobody has done** -- this section is
> the first time the four forms have been listed together. Everything after it
> is a decision that needs that reading first.

---

## 52 - CONTRACT QUEUE CLOSED -- C3, C4 and C6 ruled on the recommendations

**User, 2026-08-14: "go ahead"** -- taken as agreement to the recommendations
in SS 37.4, SS 41.6 and SS 35.5. Each is recorded with the reasoning that
earned it, so a later reader can overturn it on the argument rather than on
memory. **Where my confidence differs, SS 52.4 says so.**

### 52.1 - C3 RULED: the template's declared `type` is the authority

`_shape` widens for `type = "float"`, exactly as `effective_config` widens
from the dataclass annotation today. **One rule, one implementation.**

Chosen not for the deck text -- SIESTA reads `300` and `300.0` identically
(measured, SS 37.3) -- but because **one rule with two implementations is the
cost**, and this puts it where the vocabulary already lives. It also makes
SS 25.1's move of `effective_config` a *simplification* rather than a
relocation.

### 52.2 - C4 RULED: build the config -> wrapper road

`prep` passes the items whose `read_by` names the wrapper, from
`element.values`, beside the allocation it already passes from
`element.resources`. `_fdf_requests_gpu` survives as the **standalone**
fallback -- `write_run_wrapper` is called directly on a bare deck outside
`prep`, which is why every parameter is Optional.

Chosen because **a user's decision should reach the layer that acts on it by
being handed over, not recovered from a rendered file** (SS 41.4) -- and
because it is the only option under which `read_by` is a mechanism rather than
a declaration nothing reads.

**Also part of C4:** `stages.md` SS 4's *"nothing in molbuilder can parse an
`.fdf`"* gains the qualifier SS 42.2 drafted -- *no general FDF reader; targeted
reads of a named keyword or a reserved block are a different thing.* Without
it, the contract forbids by wording the thing C4 just permitted by intent.

### 52.3 - C6 RULED: FORMAT is the template's, CONTENT comes by registry

| half | ruling |
|---|---|
| **C6a -- format** | the **template** carries the block format: markers, comment prefix, how a payload line is quoted (SS 35.5, user). It is engine-specific and structure-independent, which are the two properties that make something an item |
| **C6b -- content** | the **registry** pattern, extended from `annotations_fdf` to cover regions and the frozen set, so no deck writer carries those names |

C6b is chosen because that half **already passes `generator.md` SS 7's seam
test** -- adding an annotation channel adds a strategy and edits no shared
file -- while the hard-coded half does not. It extends a proven mechanism
rather than inventing a second one.

### 52.4 - Where my confidence differs, stated plainly

| | confidence |
|---|---|
| **C3** | **high.** Measured, and the alternative duplicates existing logic |
| **C4** | **high.** `prep` already passes eleven values; this is the twelfth, and the fallback keeps standalone use working |
| **C6a** | **high.** The user stated it directly |
| **C6b** | **medium -- the one to revisit.** The registry is proven for annotation *channels*, which are optional and self-describing. Regions and the frozen set are neither: they are always present and the deck writer must place them in a *specific* block. Extending the registry may be the right shape, or it may be that ATOM-METADATA is one payload rendered wholesale (SS 35.3's option 2). **Decide it against the real emitter when C6 is built, not now** |

### 52.5 - The queue after this

| | |
|---|---|
| C1 | ✅ ruled + landed. Two cosmetic leftovers: `expands` on merged items *(recommend: drop)*, `engines` explicit *(recommend: keep)* |
| C2 | ✅ ruled |
| **C3** | ✅ **ruled here** |
| **C4** | ✅ **ruled here** |
| C5 | in progress -- `stages.md` SS 4's clause is now part of C4 |
| **C6** | ✅ **ruled here**, with C6b flagged for revisit at build time |

**No contract question now blocks the work.** What remains before code is
SS 51's k-grid read, which is a **science** question rather than a contract
one -- and it decides what the k-grid item's `type` must be, so it comes
before the writer.

---

## 53 - THE K-GRID READ -- done, from SIESTA 5.4.2's own source

*(SS 51 step 1. Source read from `.../siesta-gpu-stack/src/siesta/Src/` --
our own build's tree, so it is the version we ship: `kgridinit.F` (293 lines),
`find_kgrid.F` (370), `kgrid.F` (194).)*

### 53.1 - What the parameter actually is

From `kgridinit.F`'s own header, verbatim:

| | |
|---|---|
| `kscell(3,3)` | *"**Supercell** reciprocal of k-grid unit cell: `scell(ix,i) = sum_j cell(ix,j)*kscell(j,i)`"* -- the matrix defines a **supercell**, and the k-grid is the reciprocal of it |
| `displ(3)` | *"Grid **origin** in k-grid-vector coordinates: `origin(ix) = sum_j gridk(ix,j)*displ(j)`"* |
| `cutoff` | *"Minimum k-grid cutoff required. **Not used unless det(kscell)=0**"* |

**Refs it cites:** Monkhorst & Pack, *Phys Rev B* **13**, 5188 (1976); Moreno
& Soler, *Phys Rev B* **45**, 13891 (1992) -- the second is where the
*equivalent cutoff* idea comes from.

### 53.2 - The precedence, which nothing in molbuilder's docs states

From the same header's BEHAVIOUR block:

```
det(kscell) != 0            -> the input cutoff is NOT used
det(kscell) == 0            -> kscell and displ are GENERATED from the cutoff
det(kscell) == 0 & cutoff<=0 -> both read from fdf
both keywords present        -> kgrid_Monkhorst_Pack has PRIORITY
neither present              -> cutoff defaults to zero, giving GAMMA ONLY
```

### 53.3 - SIESTA's own example uses a 0.5 displacement

The example in `kgridinit.F`'s header:

```
kgrid_cutoff  50. Bohr

%block kgrid_Monkhorst_Pack   # Defines kscell and displ
4  0  0   0.50                # (kscell(i,1),i=1,3), displ(1)
0  4  0   0.50
0  0  4   0.50
%endblock kgrid_Monkhorst_Pack
```

**`displ = 0.5` on an even mesh is the classic Monkhorst-Pack shift** -- it
samples better than the Gamma-centred grid of the same size for most systems.
**molbuilder writes `0.0` always and cannot express this.**

### 53.4 - And SIESTA reports the EFFECTIVE cutoff back

`find_kgrid` returns **`eff_kgrid_cutoff`** -- *"actual equivalent kgrid
cutoff"*. So the k-grid has the **same asked-vs-effective shape as the mesh
cutoff** (SS 38): you ask for a mesh, SIESTA tells you the sampling density it
amounts to, in a length.

> **That is the number that makes sampling comparable across systems**, and it
> is what `kgrid_cutoff` lets you *specify* directly. A user comparing a small
> cell and a large one at a fixed `4x4x4` is sampling them **differently**; the
> same `kgrid_cutoff` samples them **equivalently**. Nothing in molbuilder
> surfaces either the effective cutoff or the option to specify one.

### 53.5 - So what molbuilder should offer

| | recommendation |
|---|---|
| **the displacement** | **offer it.** One 3-vector, default `[0,0,0]`. It is a real scientific choice, SIESTA's own example uses `0.5`, and today it is silently fixed. Cheapest of the three and the biggest gap |
| **`kgrid_cutoff`** | **offer it, as the alternative form.** It is the cell-aware interface and the one that makes studies comparable. Note the precedence: if both are given, the MP block wins -- so the template must not let a user set both and silently ignore one |
| **the non-diagonal matrix** | **defer.** It serves supercells commensurate with a sub-lattice -- real, but specialised, and nothing in molbuilder builds such supercells today. Recording it as *not offered* is honest; `int3` + displacement is not |

**The type follows from that**, answering SS 49.4: **not** a general `matrix`.
A diagonal mesh plus a displacement is `int3` + a `float3` -- and `float3`
does not exist in `TYPES` today. Either add it, or carry the displacement as
its own item beside `kgrid`, which is **the same flag-plus-number shape spin
settled into** (SS 43) and needs no new type at all.

> **Recommendation: two items, no new type.** `kgrid` (`int3`) and
> `kgrid_displacement` (`int3` will not do -- it is float; so `strlist` is
> wrong too). **This is the one place a `float3` earns its place**, and it
> costs one member in a closed vocabulary that just lost `strmap` and is
> keeping `pow2` for one field. `intlist`'s zero uses (SS 32) should be
> weighed against it in the same decision.

---

## 54 - The displacement: it is a 3-vector, and 0.5 must NOT be the default

*(User: "why not set displacement with 0.5 default? and also your siesta
example has one float value, does that mean [0.5,0.5,0.5]?")*

### 54.1 - Yes -- one float per ROW, three rows, so it is a 3-vector

`kgridinit.F`'s own annotation, per line:

```
4  0  0   0.50     # (kscell(i,1),i=1,3), displ(1)
0  4  0   0.50     # (kscell(i,2),i=1,3), displ(2)
0  0  4   0.50     # (kscell(i,3),i=1,3), displ(3)
```

So that example is **`displ = [0.5, 0.5, 0.5]`** -- shifted on all three axes.
Each axis is independent: `[0.5, 0.5, 0.0]` is legal and meaningful.

### 54.2 - Why 0.5 must not be the default

| case | correct shift | why |
|---|---|---|
| **`1x1x1`** -- molbuilder's **shipped default**, a molecule in a box | **0.0** | a 0.5 shift moves the single k-point to the **zone boundary** instead of Gamma. For an isolated system that is simply the wrong point |
| **odd meshes** | 0.0 | the unshifted grid already contains Gamma and is symmetric about it |
| **even meshes**, metals especially | **0.5** | the true Monkhorst-Pack set: no Gamma, and better sampling per point |
| **transport** | **0.0, forced** | `process_k_cell_displ` zeroes the transport direction's displacement -- SIESTA overrides whatever was asked |

> **So the right shift depends on the mesh parity, and defaulting to 0.5 would
> break the most common case molbuilder ships.** Default `[0, 0, 0]`.

**Not auto-derived either** *(per-axis 0.5 where n is even, 0 where odd)*,
which is what some codes do. That is molbuilder choosing physics by omission,
and the project's rule is the opposite -- *explicit is better than implicit*.
**Offer it, default it safe, and say when to change it.**

### 54.3 - The UI hint, drafted -- per SS 40.4's one formatted `help` key

```
help = """
Shifts the k-point mesh off Gamma, one value per axis, in units of the mesh
spacing.  [0,0,0] is Gamma-centred.

  0.0    Gamma-centred.  Required for a 1x1x1 mesh (an isolated molecule),
         and the safe choice for ODD meshes, which already contain Gamma.
  0.5    The classic Monkhorst-Pack shift.  Use on EVEN meshes -- it
         samples better than a Gamma-centred grid of the same size, and
         matters most for metals.

Axes are independent: [0.5, 0.5, 0.0] shifts two and leaves the third on
Gamma -- which is what a slab wants when the third axis is vacuum.

TRANSPORT: SIESTA forces this to 0 along the transport direction, whatever
you set, because that direction is sampled at one k-point."""
```

### 54.4 - And the same for the mesh itself, which had no such hint

SS 39.3 drafted `mesh_cutoff`'s. The k-grid needs one too, and the two share
the shape SS 38 and SS 53.4 both found -- **you ask, and SIESTA reports what it
actually used**:

```
help = """
Monkhorst-Pack sampling of the Brillouin zone, one count per axis.

  1x1x1        an isolated molecule -- only Gamma matters
  4x4x4 - 8x8x8  a periodic 3D crystal
  n x n x 1    a slab; no sampling along the vacuum axis

Cost scales linearly with the number of k-points.  Converge by raising the
density ~1.5x per axis: the total energy should move less than 1 meV/atom.

SIESTA reports an EQUIVALENT CUTOFF for whatever mesh you give -- a length
that says how dense the sampling really is.  That number, not the counts,
is what makes two DIFFERENT cells comparable: the same 4x4x4 on a small and
a large cell samples them differently."""
```

> **Both hints follow SS 40.5's rule** -- structure, not markup, because the
> same string is read in a terminal, in the template file, and by a UI.

---

## 55 - THE TYPE VOCABULARY, ruled -- the last contract question

*(User "go ahead", 2026-08-14. This is the decision SS 54 fed and the last
contract item before code.)*

### 55.1 - The vocabulary, member by member, with its evidence

| member | uses | ruling |
|---|:--:|---|
| `int` `float` `str` `bool` `enum` | 81 | **keep** -- the ordinary five |
| `strlist` | 2 | **keep** -- `species_order`, `ecp_atoms` |
| **`pow2`** | 0 | **KEEP and MAKE REACHABLE.** One field wants it (`parallel_block_size`), the checker is already written and correct, and a user's 96 is emitted unchecked today (SS 50.1). SS 5's stated purpose for `type` is *a constraint a parser cannot see*, and this is the only member that is one |
| **`text`** | 0 | **KEEP, reserved.** For `user_custom`, which the user named load-bearing (SS 33) |
| **`int3`** | 1 | **KEEP** -- `kgrid`. SS 53.5 ruled the non-diagonal matrix deferred, so a diagonal mesh IS three ints |
| **`float3`** | -- | **ADD.** The k-grid displacement (SS 54): three floats, one per axis, independent. Nothing else in `TYPES` can carry it |
| **`intlist`** | 0 | **KEEP, reserved** -- `frozen_indices` is SS 4.2's own worked example and is not a config field yet. Same shape as `text`: work not done, not machinery unwanted |
| **`monitor`** *(a `kind`, not a type)* | 0 | **KEEP, and write the record.** SS 50.3 found the entire design record is three lines restating one sentence. The monitor layer is real and samples real things; what is missing is any statement of what a monitor ITEM would be. **Retiring a kind whose layer exists would be the wrong deletion** |

### 55.2 - The principle this settles, stated once

> **A zero-use member is a cleanup candidate only if nothing is waiting on
> it.** `strmap` had one field and modelled **storage**, so it went.
> `pow2`, `text` and `intlist` have zero fields and model **questions** with
> named fields waiting -- so they stay. The count is not the test; **what the
> member models is.**

*(That is the rule SS 33 arrived at for `text` and SS 48 for `pow2`,
generalised. It is worth stating because the count is the tempting test and
it gets `strmap` and `text` exactly backwards.)*

### 55.3 - `TYPES` after this ruling

```
int  float  str  bool  enum  pow2  int3  float3  strlist  intlist  text
```

**Eleven** -- one more than today. The programme retired `strmap` and adds
`float3` for a capability SIESTA actually has and molbuilder could not
express, which is a different transaction from the one SS 11's size test
worries about.

### 55.4 - So the contract phase is COMPLETE

| | |
|---|---|
| C1 merge test | ✅ ruled, text landed |
| C2 allocation | ✅ ruled |
| C3 coercion authority | ✅ ruled |
| C4 config -> wrapper road | ✅ ruled |
| C5 document alignment | in progress, unblocking nothing |
| C6 format / content seam | ✅ ruled, C6b flagged for build time |
| k-grid science read | ✅ done (SS 53, SS 54) |
| **the type vocabulary** | ✅ **ruled here** |

**Nothing on paper now blocks the code.** SS 26's rule has been satisfied
rather than suspended.

---

## 56 - FIRST CODE: `pow2` declared and enforced -- and the wrinkle it exposed

*(SS 55.4 closed the contract phase; this is item 1 of the work list.)*

### 56.1 - What landed

| | |
|---|---|
| `template.declaration_for` | honours `metadata["decl_type"]`, validated against `TYPES` -- so a field can DECLARE a type its annotation cannot carry |
| `parallel_block_size` | declares `decl_type = "pow2"` |
| the effect | a template carrying `96` is now **coerced to 64 on read**. Before, it was emitted verbatim while the AUTO path capped to a power of two |

### 56.1a - COERCE, not refuse (user, 2026-08-14)

> *"instead of accepting and refusing, it should correctly coerce the value to
> the allowed number, now?"*

**Right, and it changes where the rule lives.** A refusal sends a person who
typed 96 away to work out which numbers are legal; a snap tells them by doing
it. So:

| | |
|---|---|
| `_TYPE_CHECKS["pow2"]` | checks only that it is an **int** |
| `_shape` | **snaps to the nearest power of two, DOWNWARD** |

**Downward** because that is the direction `_auto_block_size` already takes --
the largest power of two that still leaves >= 2 blocks per rank -- so the snap
can never hand the engine a **bigger** block than was asked for.

```
asked  64 -> 64      asked 96 -> 64      asked 3 -> 2
asked   1 ->  1      asked 100 -> 64     asked 0 -> 0   (the third state)
```

### 56.2 - The wrinkle: 0 is a legal value that is not a power of two

`parallel_block_size` has **three** states, and the contract already says so
(`template.md` SS 12, decision 35):

| state | means |
|---|---|
| absent / `None` | **auto** -- `prep` proposes from the orbital and rank counts |
| **`0`** | **omit the keyword entirely** -- SIESTA's own built-in default |
| `N` | use N, **and N must be a power of two** |

`_TYPE_CHECKS["pow2"]` required `v > 0`, so declaring the type would have
**refused a legal state**. The checker now accepts `0`, with the reason
attached: it is a **sentinel, not a value**, and the power-of-two constraint
does not apply to a sentinel.

> **Worth recording because it is the shape of thing that only appears when
> the code is written.** SS 50 called this item *"nothing new is built"* and
> that was true of the checker -- but not of the semantics. A type carrying a
> sentinel is slightly muddier than a type that does not, and the alternative
> (a fourth encoding for *omit*) is worse.

### 56.3 - Verified

Round-trip through `render_template` -> `read_template`, every state:

```
asked  1 -> 1     asked 16 -> 16    asked 64 -> 64    asked 128 -> 128
asked  3 -> 2     asked 96 -> 64    asked 100 -> 64   asked   0 ->   0
```

Plus a guard that `decl_type` itself must be in `TYPES`, so a typo there
cannot put an unknown type into every template. 192 tests pass across the
template, doc-claims, resolve, describe, stage-resolution and milestone suites.

### 56.4 - OPEN: a coercion nobody can see

`template.md` SS 6.4 is explicit that a value the user did not type must be
visible: *"a value the run obeys but nobody can see is the same problem as an
undocumented one."* The snap currently happens inside `_shape` and **says
nothing**.

The deck's PROVENANCE does print `BlockSize  user-set -> 64`, so the *value*
surfaces -- but not the fact that 96 was asked for. **The gap is small and
real**, and the mechanism already exists: `jobset/ledger.py` records decisions,
and SS 6.4's `resolve(asked, env) -> (effective, reason)` is exactly this shape.

**Recorded rather than fixed here**, because the right home is the resolver's
reason channel and that is SS 25.1's territory -- one more thing that lands when
the operator moves.

---

## 57 - SECOND CODE: `float3` and the k-grid displacement, end to end

*(SS 55's work list, item 2. The parameter SS 53 and SS 54 specified.)*

### 57.1 - What landed, by floor

| | |
|---|---|
| `template.py` | **`float3` in `TYPES`** (eleven members), its check, and `_shape`'s coercion. `_decl_type` maps `Tuple[float, float, float]`; the LENGTH is now guarded on `int3` too, so the name is true |
| `config/siesta.py` | **`kgrid_displacement`**, default `[0, 0, 0]`, SS 54.3's `help` verbatim, a validator. `kgrid`'s own `help` replaced with SS 54.4's |
| `siesta/input.py` | the block's **fourth column** comes from the config; three verbose comment lines say what it is |
| `engines/template.md` | SS 5's vocabulary, and the BENCH-MARKS split re-stated |

### 57.2 - Verified against the SIESTA 5.4.2 BINARY, not against a reading

Two decks, identical but for the fourth column, each piped through the
`molbuilder-siesta` env's own binary:

| displacement | SIESTA's read-back | **irreducible k-points** |
|---|---|---:|
| `[0, 0, 0]` | `siesta: k-grid: 4 0 0 0.000` | **44** |
| `[0.5, 0.5, 0.5]` | `siesta: k-grid: 4 0 0 0.500` | **32** |

> **The COUNT is the evidence, and the echo is not.** An echo could be a
> pass-through of a number the engine then ignores -- which is exactly the
> 2026-06-23 phantom-keyword shape. A **different irreducible set** means the
> displacement entered the symmetry reduction.

**And the effective cutoff is `24.000 Ang` for both**, which corrects a
plausible guess before anyone makes it: the equivalent cutoff (SS 53.4) is a
property of the SUPERCELL, not of where the mesh sits. It is the wrong thing to
assert on, and the test asserts on the count instead.

> **SIESTA has been suggesting this value all along.** The unshifted run prints
> `k-point displ. along 1 input, could be: 0.00 0.50` -- the engine naming the
> option molbuilder could not express.

### 57.3 - A latent UI defect this field WOULD have hit -- recorded, not fixed

*(The UI waits. This unit is backend + template; the note is here so the
rebuild does not rediscover it.)*

`_field_to_schema` dispatches **every** `Tuple` to `kind = "int-triple"`, and
`form-schema.js`'s renderer hard-codes `step="1"` and reads back with
`parseInt`. So any float triple on the old Build form would be:

1. marked **invalid by the browser** for `0.5` before any JS runs, and
2. read back as **`0`** if typed anyway -- **silently**.

**The value that makes the parameter worth having is the one the control cannot
carry.** The fix is the split the SCALARS already have (`int` steps by 1,
`number` by any): one renderer parameterised by `isInt`, the same shape as
`makeNumber(f, isInt)` sitting next to it. Five sites in two JS files dispatch
on the string `"int-triple"`.

**Nothing is broken today**, because `kgrid_displacement` carries no
``section`` -- and ``section`` is the old form's opt-in, retired at `@2`
(SS 5). The field is a template item and a deck line; it joins a surface when
the surface is rebuilt from the template.

> **The pattern, which outlives the incident.** A new member of a shared
> vocabulary is not landed when the writer and the reader agree; it is landed
> when every **surface** dispatching on the old vocabulary has been asked
> whether the new member changes its answer.

### 57.4 - A test pinned a premise that was false

`test_every_declaration_has_a_named_type` required **every** engine-kind
anchored item's type to be in `script_emit.DECL_TYPES`, reasoning that such an
item could reach a BENCH-MARKS line.

**It cannot.** That block declares five hand-listed fields
(`SIESTA_BENCH_FIELDS`) and nothing else -- and `kgrid` has been engine-kind and
anchored since it existed while appearing in no block. The false premise had a
real cost: it made the **benchmark's** vocabulary the gate on the **template's**,
so `float3` could not be added to one without widening the other for a type no
benchmark will ever turn.

**Replaced with the rule the contract actually states** -- `job-contracts.md`
SS 3.3's *"emitted from ONE source, and that is a rule rather than a
convenience"*. `SIESTA_BENCH_FIELDS` **is** hand-maintained, so that rule was an
intention with no mechanism; the new test is the mechanism, matching by keyword
and comparing the declared types. Writing it immediately found that
`MD.NumCGsteps` arrives through `relax_steps`' **`expands`**, not an anchor --
so a keyword reaches the deck two ways and only one of them had been considered.

### 57.5 - And `DECL_TYPES` is carrying residue

`bool` and `int3` joined it on 2026-08-07 because SS 3.7 then reused that
grammar for a template's **in-deck** item blocks. SS 3.7 moved out on
2026-08-11, when a template became its own TOML file with its own vocabulary.
**No BENCH-MARKS field declares either.** Recorded in `template.md` SS 5 and
here; not deleted, because the deletion belongs with SS 25's migrations rather
than beside a new parameter.

### 57.6 - Verified

| | |
|---|---|
| round-trip | `render_template` -> `read_template` returns a `tuple` of floats; a hand-edited `[0, 1, 0]` comes back `(0.0, 1.0, 0.0)` |
| deck | default rows are **byte-identical** to what molbuilder wrote before the parameter existed |
| the binary | SS 57.2 |
| the science | a shift on an axis sampled at ONE k-point is **warned** -- it moves that point to the zone boundary, which is meaningless for the 1x1x1 default |
| mutation | five mutations, five dead tests; all mutated files restored to their pre-mutation checksums |
| **not** verified | anything on a UI. No surface change is part of this unit |

### 57.7 - OPEN, recorded not fixed

| | |
|---|---|
| a scalar `range` on a tuple field | `_validate_config_metadata` refuses one as a programmer bug, so `kgrid` and `kgrid_displacement` both carry **no bounds a surface can read**. The bound is per component and lives in a validator, where a UI cannot see it. SS 5 calls `range` *"advisory bounds"* for a surface -- for the triples there are none |
| `cli.KGridParam` | defined, assigned to `KGRID`, and referenced by nothing. Dead by every reading, but SS 32's rule (`[[feedback-module-provenance-header]]`) says a surface glance does not settle it -- so it is written down, not deleted |

---

## 58 - C1's MECHANISM: how does the writer know two fields are one item?

*(SS 25.2 is the next unit, and it cannot be written until this is answered.
Contract work, per SS 26. **No code has been written for it.**)*

### 58.1 - SS 25.2's own note is now stale

It says *"keep items **unmerged** across engines (`net_charge` and `charge` stay
two items in one category)"*. That was written against the OLD SS 6.3, which
read *"items are never merged across engines"*. **SS 6.3 was rewritten later the
same day** to the two-part test -- two engines share an item when it is **the
same question** *and* **the same answer** -- so the flat refusal is gone and
`net_charge`/`charge` is now the contract's own worked example of a merge.

**The stale line is struck here rather than in SS 25.2**, so the record shows
the contract moved under the work list rather than the work list being wrong
when written.

### 58.2 - The gap, stated exactly

SS 6.3 names four merges. **Three of them are between fields with DIFFERENT
names:**

| the merged item | SIESTA's field | PySCF's field | same name? |
|---|---|---|:--:|
| `charge` | `net_charge` | `charge` | ✗ |
| `use_gpu` | `enable_gpu` | `use_gpu` | ✗ |
| `verbose_comments` | `verbose_comments` | `verbose_comments` | ✓ |
| `write_molwatch_log` | `write_molwatch_log` | `write_molwatch_log` | ✓ |

> **So a writer that merges by field name finds two of the four and misses two**
> -- and SS 6.3 explicitly warns the other way too: *"A shared **name** is
> evidence of neither."* The mechanism has to be something a field DECLARES.

### 58.3 - What the collision surface actually is -- measured, not assumed

`declarations_for(SiestaConfig)` vs `declarations_for(PySCFConfig)`, 2026-08-14:

| | |
|---|---|
| SIESTA items | **45** |
| PySCF items | **40** |
| **shared names** | **3** |

| shared name | kind | type | SIESTA default | PySCF default | agree? |
|---|---|---|---|---|:--:|
| `verbose_comments` | `produce` | `bool` | `True` | `True` | ✅ |
| `write_molwatch_log` | `produce` | `bool` | `True` | `True` | ✅ |
| `max_memory_mb` | `wrapper` | `int` | `None` | `4000` | ❌ **default** |

**Two of the three collisions are merges the contract already asks for, and the
third is a real disagreement** -- and one the user has already ruled on:
*"max_memory_mb is only explicitly set when user requested, the typical memory
limit is unlimited - meaning all physical memory is allowed."* By that ruling
SIESTA's `None` is right and PySCF's `4000` is the outlier, which is a defect
this exercise surfaced rather than a merge problem.

> **The accidental-collision risk SS 6.3 warns about is, in the measured
> present, ZERO.** Every shared name is either a merge the contract wants or a
> bug. That does not make name-matching *correct* -- it makes the cost of the
> simpler mechanism small and checkable.

### 58.4 - The two candidate mechanisms

| | **A -- the field name IS the item name** | **B -- a metadata key names the item** |
|---|---|---|
| how a merge is declared | by the two engines **spelling the field the same** | `metadata["item"] = "charge"` on each half |
| what it costs | renaming `net_charge` -> `charge` and `enable_gpu` -> `use_gpu` in `SiestaConfig`, and everything that reads those names | a fifth thing every field may declare, and a name that is not the field's name |
| accidental collision | **refused loudly**: two fields with one name that disagree on `kind`/`type`/`default` is an error naming both | impossible by construction -- but a *missed* merge is silent, because nothing was declared |
| which failure is louder | a wrong merge **stops the writer** | a missing merge **ships as two items** and nobody sees it |
| new vocabulary | none | one key |

**Recommendation: A.** Three reasons, in the project's own order of preference
(*delete > one home > parameter > abstraction*):

1. It adds **no** mechanism. The item name is the field name, which is already
   true for all 85 items.
2. It makes the dangerous direction the LOUD one. Under A a merge that should
   not have happened is a refusal with both fields named; under B a merge that
   should have happened is simply absent, and absent things are what this whole
   review keeps finding.
3. The renames it forces are ones the unification wanted anyway -- SS 1 of the
   unification plan measured *"two names for one question"* as the defect, and
   `net_charge`/`charge` is that defect with a name.

**What A forces, in full** *(nothing here is done)*:

- `SiestaConfig.net_charge` -> `charge`; `SiestaConfig.enable_gpu` -> `use_gpu`.
- Every reader of those two names moves with them -- no shim, per the project's
  rename rule. **This has not been counted yet**; counting it is the first step
  of the unit, not of this section.
- `max_memory_mb`'s two defaults must be reconciled first, or the writer refuses
  on it the moment it merges two engines. Per the user's ruling: unlimited.
- A test that the merged halves agree, and that a disagreement names both.

### 58.5 - The question that is NOT settled here

Whether `use_gpu` merging is even wanted, given SS 41.1: `enable_gpu` is an
allocation-kind decision on SIESTA (a benchmark axis) while PySCF's is a backend
selection. **Same question, same answer -- or two questions wearing one word?**
SS 6.3's table asserts the merge; SS 41.1's reasoning is the case against it.
**One of the two is wrong and this section does not decide which.**
