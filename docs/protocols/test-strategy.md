# Test strategy

> **This document is the sole source of truth for HOW molbuilder
> organises its tests across layers** — what each layer tests, what
> it explicitly does NOT test, and where new tests go when a module
> grows.
>
> Companion to [`scientific-validation.md`](scientific-validation.md)
> (the validator architecture this strategy mirrors) and
> [`playwright-tests.md`](playwright-tests.md) (e2e contracts).
>
> Pointer in [`design.md`](../design.md) § "Testing".

---

## 1. Why this document exists

Pre-2026-06-13 every validator lived in one flat
`molbuilder/validation.py` and every validator test lived in one
flat `tests/test_validation.py` (1479 LoC).  When the Au-BDT-Au
drift incident landed, finding the right place to add a regression
test meant scrolling through ~1500 lines of intermingled geometry,
chemistry, SIESTA, and PySCF tests.  The validator-package split
(per [`scientific-validation.md`](scientific-validation.md) § 10)
makes the source modular; this doc makes the tests follow suit.

---

## 2. The pyramid

Five layers, from cheapest + most specific at the bottom to most
expensive + most coupled at the top:

```
        ┌─────────────────────────────────────┐
        │  L5  e2e (Playwright, full Flask)   │  slow, brittle
        ├─────────────────────────────────────┤
        │  L4  integration (multi-module)     │
        ├─────────────────────────────────────┤
        │  L3  interface (boundary contracts) │
        ├─────────────────────────────────────┤
        │  L2  module (single submodule)      │
        ├─────────────────────────────────────┤
        │  L1  unit (pure helper)             │  fast, focused
        └─────────────────────────────────────┘
```

Each layer answers a different question:

| Layer | Question | Scope |
|---|---|---|
| L1 unit | "Does this pure helper return what I expect?" | one function, no side effects, no I/O |
| L2 module | "Does this submodule's public surface behave correctly?" | one file's worth of functions; submodule-internal collaboration allowed |
| L3 interface | "Do callers of this module see the contract the docs promise?" | the boundary between two modules — types, return shapes, severity rules, ordering |
| L4 integration | "Do multiple subsystems agree about a shared fact?" | analyzer ↔ validator ↔ chip; CLI ↔ blueprint ↔ engine |
| L5 e2e | "Does the user-facing flow work?" | Playwright drives a real browser against a real Flask app |

The pyramid shape is intentional.  L1 is cheap (microseconds), L5 is
expensive (seconds + browser startup).  Push assertions DOWN the
pyramid whenever possible — a chemistry rule that lives at L1 (pure
analyzer check) doesn't need an L5 Playwright run to confirm it.

---

## 3. Test directory layout

The test layout mirrors the source layout:

```
tests/
├── conftest.py                       # project-wide shared fixtures
├── data/                             # fixture data files
├── validation/                       # mirrors molbuilder/validation/
│   ├── __init__.py                   # empty (makes the dir a package)
│   ├── conftest.py                   # water_struct fixture
│   ├── _helpers.py                   # _vacuum_cell, _peptide_struct
│   ├── test_geometry.py              # L2 — geometry submodule
│   ├── test_metadata.py              # L2 — metadata submodule
│   ├── test_chemistry.py             # L2 — chemistry submodule
│   ├── test_siesta.py                # L2 — _validate_siesta sequence pin
│   ├── test_pyscf.py                 # L2 — _validate_pyscf sequence pin
│   └── test_aggregator.py            # L2 — validate() + report() entry
├── spectra/                          # mirrors molbuilder/spectra/
├── transport/                        # mirrors molbuilder/transport/ (planned)
├── watch/                            # legacy mirror of molbuilder/parsers/ (parsers/ deleted in H4b; watch goldens kept here)
├── test_chemistry_analyzer.py        # L2 chemistry module
├── test_chemistry_adapters.py        # L3 adapter registry boundary
├── test_structure_analyze_endpoint.py # L4 HTTP boundary
├── test_*_e2e.py                     # L5 Playwright
├── test_live_poll_invariants_audit.py # L2 — JS source-text invariants
└── test_css_*.py                     # L2 — CSS source-text invariants
```

The validation split landed 2026-06-13 (88 tests across 6 files,
parity-verified with the pre-split flat `test_validation.py`).
Other top-level flat files (`test_web.py`, `test_modify.py`, etc.)
may follow the same pattern when they cross the "~1000 LoC and
multiple concerns mixed in one file" smell threshold — but only
when there's a forcing function.  Don't split for the sake of
splitting; the convention exists so that drift is catchable, not
to optimise line counts.

---

## 4. What each layer tests

### 4.1 L1 — unit

* **Subject.**  One pure function: deterministic, no I/O, no global
  state.
* **Scope.**  Inputs → outputs.  Edge cases, boundary values,
  numerical limits.
* **No fixtures beyond constants.**  If you need a `Structure`
  fixture for a chemistry-rule test, you're at L2 or higher.
* **Naming.**  `test_<function_name>_<scenario>` —
  `test_min_image_distance_orthorhombic_cell`,
  `test_check_polymer_orientation_reversed_residue_listing`.

**Example.**  `_min_image_distance(positions, cell, inv)` —
geometric primitive.  Test: pass a 2-atom system in a 10 Å cubic
cell, assert the returned distance equals 10 - separation.

### 4.2 L2 — module

* **Subject.**  One submodule's public surface, exercised through
  its real collaborators inside the submodule.
* **Allowed.**  Intra-submodule imports, real `Structure` fixtures,
  real config dataclasses.
* **Forbidden.**  Crossing submodule boundaries (e.g.
  `validation.chemistry` test must not exercise `validation.siesta`).
* **Naming.**  `test_<thing_under_test>_<scenario_or_invariant>`.

**Example.**  `test_check_open_shell_metal_au_cluster_does_not_warn`
in `tests/validation/test_chemistry.py` — drives the chemistry
submodule end-to-end with a real Au cluster, but doesn't go
through `validate()` or `_validate_siesta`.

### 4.3 L3 — interface

* **Subject.**  The contract between two modules.  Severity rules,
  return shape, ordering, registry behaviour.
* **Allowed.**  Two-module imports; verifying the SHAPE of a return
  value (e.g. "every Issue has a `where` field starting with
  `config.`").
* **Forbidden.**  Asserting specific Issue messages (that's L2's
  job); end-to-end runs (that's L4+).
* **Naming.**  `test_<boundary>_<contract>` —
  `test_engine_validator_registry_dispatch_by_isinstance`,
  `test_analyzer_returns_frozen_dataclass`.

**Example.**  `tests/test_chemistry_adapters.py` —
`test_all_adapters_agree_on_treatment`: every registered adapter,
given the same `ChemistryAnalysis`, must reach the same
open/closed-shell verdict.  That's a boundary contract.

### 4.4 L4 — integration

* **Subject.**  Multiple subsystems agreeing about a shared fact.
* **Allowed.**  Mocking the slow boundary (file I/O, HTTP, subprocess)
  but exercising real subsystems above it.
* **Forbidden.**  Browser drives (that's L5); skipping subsystems
  the integration is supposed to test.
* **Naming.**  `test_<flow>_<invariant>` —
  `test_au_bdt_au_closed_shell_does_NOT_warn_through_full_validate`,
  `test_analyzer_validator_chip_agree_on_open_shell_metal`.

**Example.**  `tests/validation/test_chemistry.py::TestCheckOpenShellMetalUsesAnalyzer::test_validator_reads_metals_from_analyze_structure`
— monkeypatches the analyzer, drives the full validator path,
asserts agreement.  Cross-subsystem (chemistry ↔ validation).

### 4.5 L5 — e2e

* **Subject.**  The user's actual flow through the browser.
* **Allowed.**  Real Flask app, real Playwright, real DOM, real
  3Dmol.js rendering.
* **Forbidden.**  Asserting on internal implementation details
  (Issue list, Python-side state) — that's lower in the pyramid.
* **Naming.**  `test_<user_journey>_<observable_outcome>` —
  `test_frame_slider_scrubs`, `test_workspace_dirty_state_survives_tab_switch`.

**Example.**  `tests/test_molbuilder_e2e.py::test_modify_layout_phone_width_no_horizontal_overflow`
— drives a Playwright Chromium at 375px width and asserts no
horizontal scroll appears.

---

## 4a. Test environments — design tests AROUND the envs, never the reverse

molbuilder's conda envs are an authoritative, **test-locked** part of the
deployment: [`README_install.md` §The-envs](../README_install.md) is the human
doc, `molbuilder/envs/recipes.py` is the machine source of truth, and
`tests/test_envs_*.py` (esp. `test_envs_readme_consistency.py`) fails if the two
drift.  The env layout is FIXED; **tests conform to the envs, never the reverse.**

**HARD RULE — never change an env to make a test run.**  Do not install/remove
packages in any conda env, and do not edit `recipes.py` / `README_install.md` /
`deployment.md`, to get a test to pass.  Any env or deployment change goes through
documented discussion + explicit approval FIRST.  If a test needs something an env
lacks, the *test* is wrong — or it's a design discussion — not the env.  Also:
**never mix conda + pip** (pip only where a recipe's `pip_packages` requires it — a
package genuinely not on conda; mixing corrupts native ABIs).

**Env roles for tests:**

- **L1–L4** (unit / module / interface / integration) — the **host env**, which
  carries the full app runtime + all science backends.
- **L5 e2e** (Playwright) — the browser env (`molbuilder-tests`), whose designed
  contents are **browser tooling only** (playwright + pytest-playwright + Chromium).

**Scientific backends never enter the browser E2E env.**  rdkit / openbabel /
biopython / sisl / pyscf live only in the host + their own backend envs.  A
backend-dependent e2e case must **skip** when the backend is absent, or let the app
**dispatch** the heavy build to the backend's own env (`conda run -n <env>`).  See
[`playwright-tests.md` § 9.8](playwright-tests.md) for the `backends_any` +
`available_backends()` skip pattern.  Baking a backend into the browser env to make a
case run is both wrong (contaminates a lean, conflict-avoiding env) and pointless
(the case is built to skip).

### 4a.1 Calling code that lives in another env — the two kinds

The core rule: **a test runs in ONE env and never reaches across envs itself.**  It
does not import a backend that lives in another env, and it does not get that backend
installed into its own env to make it run.  A dependency that lives "elsewhere" is
reached in exactly one of two ways, and picking the right one is a real correctness
concern (mixing them up is what turned a missing dependency into a bug-looking `500`):

**Kind 1 — a backend TOOL/binary in its own env** (`tleap`, `siesta`, `pyscf`).  These
live in dedicated envs (`molbuilder-MDtools` / `-siesta` / `-pySCF`) because of hard
conflicts.  The test never calls them; the **app** does, at runtime, by dispatching
into their env (`conda run -n <backend-env> <tool>`).  The test drives the app and
**gates on `available_backends()`** — a probe of whether the tool is reachable (e.g.
`"amber"` = "`tleap` on PATH").  The tool's env is NOT a dependency of the test process.

**Kind 2 — an in-process Python LIBRARY** (`PeptideBuilder`, `rdkit`).  These are
imported by the app **in the same process**, so they must be present in the env the
test runs in (the host env).  The test **gates by import**:
`try: import PeptideBuilder / except ImportError: pytest.skip(...)`.

Same symptom, different gate — so tag each case by the RIGHT kind.  The peptide
builder is Kind 2 (`PeptideBuilder`, an in-process host-env library), NOT Kind 1
(`amber`/tleap): tagging it `("amber",)` let the guard pass on a box that had the amber
env but not PeptideBuilder, and the build then failed on the missing import.

**One line:** never bring the backend to the test — bring the test to where the app
already reaches the backend.  In-process libs are import-gated; separate-env tools are
dispatch-and-`available_backends()`-gated; either way a missing dependency is a
**skip**, never a reason to touch an env.  (E2e mechanics: `playwright-tests.md` § 9.8.)

---

## 5. Source-text invariant tests (a special L2 pattern)

Some invariants live in JS / CSS files we can't easily exercise
from Python.  Source-text tests use regex over the JS/CSS source
to pin structural invariants:

* `tests/test_live_poll_invariants_audit.py` — pins live-poll
  guards, workflow-group schema, detection-chip wiring.
* `tests/test_css_no_duplicate_selectors.py` /
  `test_css_no_hex_literals.py` — pin the CSS-phase refactor
  outcomes.

These are L2 (one file's worth of source).  They're brittle by
design: they break when a refactor changes the code shape, which
is the point — a "broken" source-text test means "review whether
the new shape still honours the invariant."

**Rule.**  Source-text tests must cite the invariant they pin in
their docstring + the protocol doc that defines the rule.  See
existing ones for the pattern.

---

## 5a. State-composition bugs (the molview class)

A whole family of bugs hides from per-function tests because the bug
lives in the **composition** of two successful calls.

### The pattern

A stateful module exposes setter functions that the docstring calls
"patches": the caller sends only the field they want to change.
But the implementation runs the input through a `_normalise*` helper
that applies DEFAULTS to unspecified fields — so the patch becomes a
**replace** at the implementation layer.  Each individual call works.
The COMPOSITION of two calls clobbers state silently.

### The molview example (2026-06-13)

```javascript
// Documented contract: setStyle is a patch.
setStyle({ rep: "sphere" });        // user picks sphere — works
setStyle({ radiusScale: 1.5 });     // user drags radius slider…
                                     //   …and rep silently reverts to "stick"
```

The bug survived six months of source.  Per-function tests passed:

```python
def test_setStyle_with_sphere_sets_sphere():
    setStyle({"rep": "sphere"})
    assert state.rep == "sphere"      # ✓ passes

def test_setStyle_with_radius_sets_radius():
    setStyle({"radiusScale": 1.5})
    assert state.radiusScale == 1.5   # ✓ passes
```

What was missing: the COMPOSITION test.

```python
def test_setStyle_radius_patch_preserves_rep():
    setStyle({"rep": "sphere"})
    setStyle({"radiusScale": 1.5})    # patch, not replace
    assert state.rep == "sphere"      # ← this is the bug
```

### How to catch state-composition bugs

* **Name the contract.**  If a setter is a patch, write that in the
  function's docstring + the protocol doc.  Reviewers + future you
  can then notice when the implementation drifts to "replace."
* **Pin the contract at the source level.**  Source-text invariant
  tests can assert "the implementation merges the patch onto current
  state BEFORE the normalise helper" — see
  `tests/test_mol_viewer_embed_js.py::TestPartialUpdateContract`.
* **Write sequence tests for stateful APIs.**  Even one sequence
  test per stateful setter catches the bug class: "after rep=X,
  setStyle(...other fields...) preserves rep=X."
* **For L4 integration tests, drive SEQUENCES, not isolated calls.**
  A test that sets up a session state, then drives 3-4 user
  interactions in order, catches composition bugs that
  parallel-axes tests miss.

### When to write what

| Bug class | Layer | Mechanism |
|---|---|---|
| "Function returns wrong value for input X" | L1 unit | Parametrized inputs |
| "Function's output drops a documented field" | L2 module | Schema assertion |
| "Two functions interact wrong" | L4 integration | Sequence test |
| "Implementation drifts from contract" | L2 source-text | Regex on source |

A per-function test cluster + a sequence test cluster + a contract-
text invariant covers most of what unit tests miss.

### Why this hides for a long time

The molview module was the **first** module in the project.  Its tests
were written when the API was small (1-2 setters); composition wasn't
visibly a concern.  As new setters landed and the API grew, the test
shape didn't grow with it.  The lesson: **rev test design when the
module's API surface gains a new shape, not just when it gains a new
function.**

---

## 6. Mocking discipline

The chemistry analyzer (`analyze_structure`) is the single source
of truth for chemistry questions.  Mocking it correctly is the
difference between proving "the validator reads the analyzer" and
"the test passes by accident."

* **L1 / L2 chemistry tests.**  Use real `Structure` fixtures + the
  real analyzer.  The analyzer is pure, deterministic, and fast;
  mocking it loses signal.
* **L3 / L4 interface + integration.**  Monkeypatch when the test
  is asserting "the validator delegates to the analyzer" — fake
  the analyzer to return a known shape, then assert the validator
  consumed THAT shape.  Caveat: monkeypatch the canonical
  `molbuilder.chemistry.analyze_structure` AND, if the calling
  module re-exports it, the calling module's binding too.
* **L5 e2e.**  Never mock chemistry.  The whole point of e2e is the
  real stack.

See `tests/validation/test_chemistry.py::TestCheckOpenShellMetalUsesAnalyzer`
for the canonical fake-analyzer pattern.

---

## 7. Where new tests go

Decision tree for a new test:

```
Is it pinning a JS or CSS source-text invariant?
  YES → tests/test_*_invariants_*.py (L2 source-text)
  NO  ↓
Does it exercise more than one subsystem (chemistry + validator,
analyzer + chip, CLI + blueprint)?
  YES → L4 integration — tests/test_<flow>.py
  NO  ↓
Does it test a contract at a module boundary (severity rule,
return shape, registry behaviour)?
  YES → L3 interface — tests/test_<boundary>.py
  NO  ↓
Is the function pure (no I/O, no globals)?
  YES → L1 unit — tests/<module>/test_<file>.py
  NO  → L2 module — tests/<module>/test_<file>.py
```

**Pre-split exception (2026-06-13).**  Until
`tests/test_validation.py` is split into the per-submodule files
in § 3, add new validation tests there + tag the docstring with the
intended target submodule:

```python
def test_check_open_shell_metal_au_bdt_au_does_not_warn():
    """Target after split: tests/validation/test_chemistry.py (L2).

    Au-BDT-Au regression — the Au cluster context should produce
    closed-shell singlet recommendation, validator silent."""
```

When the split happens, grep the docstring for "Target after split:"
to find the destination.

---

## 8. Naming convention + namespacing framework

molbuilder uses **pytest's built-in mechanisms** for hierarchy + selection.
No external framework needed — pytest gives us four orthogonal axes that
together cover every "find / add / manage" question:

| Axis | Mechanism | Example | Selects via |
|---|---|---|---|
| **Subject** (what's tested) | Directory hierarchy mirroring `molbuilder/` | `tests/validation/test_chemistry.py` | `pytest tests/validation/test_chemistry.py` |
| **Layer** (pyramid tier) | `@pytest.mark.<tier>` registered in pyproject.toml | `@pytest.mark.unit` | `pytest -m unit` |
| **Group** (related setup) | Class `Test<Concern>` | `class TestCheckOpenShellMetalUsesAnalyzer` | `pytest -k TestCheckOpenShell` |
| **Scenario** (specific case) | Function `test_<input>_<outcome>` | `test_h_ratio_skeleton_is_warn` | `pytest -k test_h_ratio_skeleton` |

The four axes compose:

```bash
# Every unit-level test under the validation subject:
pytest tests/validation/ -m unit

# Every integration test on chemistry, regardless of subject directory:
pytest -m "integration" -k "chemistry"

# Every test in the open-shell-metal class:
pytest -k TestCheckOpenShellMetal
```

### 8.1 Directory naming (subject)

Mirror source layout 1-to-1.  `molbuilder/<package>/<module>.py` →
`tests/<package>/test_<module>.py`.  When the source is a flat module
(no `<package>/`), use `tests/test_<module>.py` at the test root.

* **Why**: "where do I add a test for `validation/chemistry.py`?" has
  exactly one answer — `tests/validation/test_chemistry.py`.
* **Why not**: never group tests by what they test FOR (e.g. a
  `tests/regressions/` or `tests/bugs/`) — that loses the locality
  between source and test.

### 8.2 File naming (subject continued)

* `test_<module>.py` for tests of a single source submodule.
* `test_<concern>.py` for cross-cutting concerns that don't have one
  source home (e.g. `test_layering.py`, `test_css_no_duplicate_selectors.py`,
  `test_live_poll_invariants_audit.py`).
* `conftest.py` at each level for shared pytest fixtures (auto-picked
  up by pytest; no import needed).
* `_helpers.py` (underscore prefix) for plain Python helpers that aren't
  pytest fixtures.  Import explicitly: `from ._helpers import _vacuum_cell`.
* `__init__.py` (empty) makes the directory a Python package so
  `from ._helpers import ...` works.

### 8.3 Class naming (group)

Use a class when a cluster of tests shares setup (a fixture call sequence,
a synthetic input dataset, a monkeypatch).  Class name reads as the
invariant under test:

```python
class TestCheckOpenShellMetalUsesAnalyzer:
    """The validator must read its open-vs-closed verdict from
    analyze_structure().suggested_treatment — not parallel logic."""

    def test_au_bdt_au_closed_shell_does_NOT_warn(self):
        ...
    def test_single_au_atom_still_warns_open_shell(self):
        ...
```

Format: `Test<Concept>[<Predicate>]`.  Examples:
* `TestCheckOpenShellMetalUsesAnalyzer` — invariant: function delegates to analyzer
* `TestPartialSpectraInspectorEndpoint` — invariant: endpoint serves the partial
* `TestWorkflowGroupSchemaConsistency` — invariant: schema honors the workflow_group tag

Don't use a class for a single test — promote to a function.

### 8.4 Function naming (scenario)

Format: `test_<subject>_<input_state>_<expected_outcome>`.

* `test_h_ratio_skeleton_is_warn` — given a heavy-atom skeleton → warn fires
* `test_h_ratio_organic_no_warn` — given an organic molecule → no warn
* `test_au_bdt_au_closed_shell_does_NOT_warn` — given Au-BDT-Au with closed-shell config → no warn
* `test_render_fdf_raises_on_overlapping_atoms` — given overlapping atoms + render call → raises

Capitalised words inside test names (NOT, MUST) signal load-bearing
negative assertions — read at a glance from a failure report.

### 8.5 Marker registration (layer)

Markers are registered in `pyproject.toml` under
`[tool.pytest.ini_options].markers` with one-line semantics.  An
unregistered marker emits a warning, so adding a new marker without
registering it shows up as drift.

Today's registered markers:

```toml
markers = [
    "unit:        L1 — pure helper, no I/O, no globals (microsecond cost)",
    "module:      L2 — single submodule's public surface end-to-end",
    "interface:   L3 — contract between two modules (registry, severity, shape)",
    "integration: L4 — multiple subsystems agreeing on a shared fact",
    "smoke:       e2e smoke tests that subprocess-run a generated script",
    "e2e:         browser-driven Playwright tests",
    "slow:        tests that take > 1s",
]
```

**Applying markers.**  Default: don't.  Most tests are obvious from their
directory.  Use a marker when the test's intent CROSSES the directory
boundary:

* A test in `tests/validation/test_chemistry.py` that exercises the
  full `validate(struct, cfg)` flow gets `@pytest.mark.integration` —
  its subject is "chemistry validator" but its layer is L4.
* A `slow` test that hits a 30-second SIESTA subprocess gets
  `@pytest.mark.slow @pytest.mark.smoke` — let pre-commit skip it.

**Don't mark every test.**  Marker discipline is "no marker = the
default for this file's directory."  Conftest can apply default
markers per directory if it gets noisy.

### 8.6 Helper / fixture naming

* Pytest fixtures live in `conftest.py`, named without underscore
  prefix (e.g. `water_struct`).  Fixtures cascade — `conftest.py` at
  `tests/` is visible everywhere; `tests/validation/conftest.py` is
  visible to `tests/validation/**`.
* Plain Python helpers live in `_helpers.py`, named with underscore
  prefix (e.g. `_vacuum_cell`, `_peptide_struct`).  Import explicitly.
* Fixture vs helper: if it builds something you'd otherwise have to
  rebuild per test, make it a fixture.  If it's a one-line constructor
  used only by certain tests, make it a helper.

### 8.7 Parametrize discipline

Use `@pytest.mark.parametrize` for orthogonal scenarios sharing the
same assertion:

```python
@pytest.mark.parametrize("element_id", REQUIRED_IDS)
def test_body_carries_required_inspector_ids(self, web, element_id):
    body = web.get("/partials/trajectory-inspector").get_data(as_text=True)
    assert f'id="{element_id}"' in body
```

The parametrize IDs become the test name suffix
(`test_body_carries_required_inspector_ids[viewer]`,
`test_body_carries_required_inspector_ids[run-state-badge]`, …) so
failures are individually addressable.

Avoid parametrize for scenarios that need different setup — split into
separate test functions.

### 8.8 Worked example

The shape that now sits in `tests/validation/` is the canonical
example of the convention:

```
tests/validation/
├── __init__.py           # empty (makes the directory a package)
├── conftest.py           # water_struct fixture
├── _helpers.py           # _vacuum_cell, _peptide_struct (plain helpers)
├── test_geometry.py      # mirrors molbuilder/validation/geometry.py
├── test_metadata.py      # mirrors molbuilder/validation/metadata.py
├── test_chemistry.py     # mirrors molbuilder/validation/chemistry.py
├── test_siesta.py        # mirrors molbuilder/validation/siesta.py
├── test_pyscf.py         # mirrors molbuilder/validation/pyscf.py
└── test_aggregator.py    # validate() + report() — the __init__.py surface
```

Each per-module file holds the unit + module tests for that submodule
(default — no marker needed).  Aggregator-level tests live in
`test_aggregator.py`.  An integration test that crosses submodule
boundaries gets `@pytest.mark.integration` regardless of which file it
lives in.

---

## 9. Out of scope

* **Performance / benchmark tests.**  No pytest-benchmark today;
  `test_modify_layout_*` is the closest thing (asserts a render
  finishes within Playwright's default timeout).
* **Property-based / hypothesis tests.**  Not in the project today.
* **Mutation testing.**  Not in the project today.

If we add any of these, this doc gets a new section.

---

## 10. Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-06-13 | This doc landed alongside the validator-package split.  Test strategy explicitly named: L1 unit / L2 module / L3 interface / L4 integration / L5 e2e, with directory layout mirroring source layout. | The validator split (1326 LoC → 7 files) made the question "where does this validator live?" answerable for source — but the corresponding question for tests was unanswered: every validation test still lived in one 1479-LoC file.  This doc writes the answer down so the test split that has to follow has a target shape, not improvisation per PR.  The pyramid + decision tree make "where does this NEW test go?" answerable without reading every existing test file. |
| 2026-06-13 | Naming convention + namespacing framework added (§ 8): four orthogonal axes (subject / layer / group / scenario) covered by pytest's built-in mechanisms (directory hierarchy / `@pytest.mark` / class / function name).  No external framework needed.  Markers `unit` / `module` / `interface` / `integration` registered in `pyproject.toml` alongside the existing `smoke` / `e2e` / `slow`. | The directory hierarchy mirroring source code answers "where does this test live?" but doesn't answer "how do I select all integration tests across the project?" Markers cover that cross-cutting axis without forcing files to migrate to a `tests/integration/` parallel tree.  Pytest's built-in registry (markers in pyproject.toml) makes the marker set discoverable + warns on typos.  Class + function name conventions encode the invariant under test in the test's own name so failure reports read like a contract violation summary — `TestCheckOpenShellMetalUsesAnalyzer::test_au_bdt_au_closed_shell_does_NOT_warn` IS the contract being checked. |
