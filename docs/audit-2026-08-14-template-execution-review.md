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
| `docs/execution/job-contracts.md` §§ 3.1–3.2, 6.2, 6.3 | *(partial — those sections in full)* |

| not yet read | lines |
|---|---|
| `docs/execution/job-contracts.md` · `project-layout.md` · `checkpointing.md` · `architecture.md` · `run-identity.md` · `running-a-job.md` · the staged-runs plan | ~12,400 |
| `molbuilder/jobset/*` · `task.py` | ~5,200 |
| the test suites for the above | — |

**Stated plainly because a partial review reported as a whole one is worse
than no review.**

**Layout of this report:** findings per surface first (§§ 1–9), then the two
cross-cutting sections — the repeating patterns, and the design assessment —
**last** (§§ 10–11). A synthesis buried between findings is the same
misplacement this report charges elsewhere; it was buried here until this
pass, which is worth admitting rather than quietly fixing.

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
