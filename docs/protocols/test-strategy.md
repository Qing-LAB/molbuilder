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
├── conftest.py                       # shared fixtures
├── data/                             # fixture data files
├── validation/                       # mirrors molbuilder/validation/
│   ├── test_geometry.py              # L1+L2
│   ├── test_metadata.py              # L1+L2
│   ├── test_chemistry.py             # L1+L2
│   ├── test_sidecar.py               # L1+L2
│   ├── test_siesta.py                # L2 — _validate_siesta sequence pin
│   ├── test_pyscf.py                 # L2 — _validate_pyscf sequence pin
│   ├── test_interface.py             # L3 — boundary contracts
│   └── test_integration.py           # L4 — analyzer ↔ validator agreement
├── spectra/                          # mirrors molbuilder/spectra/
├── transport/                        # mirrors molbuilder/transport/ (planned)
├── test_chemistry_analyzer.py        # L2 chemistry module
├── test_chemistry_adapters.py        # L3 adapter registry boundary
├── test_structure_analyze_endpoint.py # L4 HTTP boundary
├── test_*_e2e.py                     # L5 Playwright
├── test_live_poll_invariants_audit.py # L2 — JS source-text invariants
└── test_css_*.py                     # L2 — CSS source-text invariants
```

Today (2026-06-13) the validation split has landed but the
1479-LoC `test_validation.py` has NOT yet been split into the
per-submodule files above.  That's a forward task; the file works
as-is because `from molbuilder.validation import ...` still
resolves through the package's `__init__.py` re-exports.

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

**Example.**  `test_validation.py::TestCheckOpenShellMetalUsesAnalyzer::test_validator_reads_metals_from_analyze_structure`
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

See `tests/test_validation.py::TestCheckOpenShellMetalUsesAnalyzer`
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

## 8. What this strategy does NOT cover

* **Performance / benchmark tests.**  No pytest-benchmark today;
  `test_modify_layout_*` is the closest thing (asserts a render
  finishes within Playwright's default timeout).
* **Property-based / hypothesis tests.**  Not in the project today.
* **Mutation testing.**  Not in the project today.

If we add any of these, this doc gets a new section.

---

## 9. Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-06-13 | This doc landed alongside the validator-package split.  Test strategy explicitly named: L1 unit / L2 module / L3 interface / L4 integration / L5 e2e, with directory layout mirroring source layout. | The validator split (1326 LoC → 7 files) made the question "where does this validator live?" answerable for source — but the corresponding question for tests was unanswered: every validation test still lived in one 1479-LoC file.  This doc writes the answer down so the test split that has to follow has a target shape, not improvisation per PR.  The pyramid + decision tree make "where does this NEW test go?" answerable without reading every existing test file. |
