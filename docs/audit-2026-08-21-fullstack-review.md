# Audit 2026-08-21 — jobset · engines · execution · the two tabs, full-text

**Role:** review record
**Domain:** all of it — the review the user asked for after the day's
deliveries: *"fresh-eye full text review … documentation cross review and
consolidation, code static review … consistency, redundancy, error,
duplicated code … gaps between contract and code … over-engineering that
could be simplified."*
**Method:** the R×3 protocol (docs/process/code-audit.md): five parallel
full-text readers over (1) jobset/execution, (2) engine emission,
(3) validation, (4) the two tabs' web layer, (5) the documentation set —
every load-bearing claim re-verified against source before it appears
here; then the cross-boundary synthesis pass; then the retire-half sweep
of the test suite.  Suite state at start: none2e 6780 ran / 0 unexplained
failures.

**Status: IN PROGRESS — findings land here as they are verified.  Nothing
is fixed without an explicit yes per finding.**

## Findings

### From the test-suite retire-half sweep (verified directly)

- **T1 (drift, verified):** `tests/test_pyscf_smoke.py::test_smoke_h2o_rhf_sto3g_frequencies`
  still exercises the RETIRED in-deck frequencies path — sets the deleted
  `compute_frequencies` field (a bare attribute set, so no TypeError) and
  asserts a `thermo.txt` the deck no longer writes.  Invisible because the
  whole module skips in the molbuilder env (`importorskip("pyscf")`,
  line 51) — a breakage hiding behind a skip.  Direction: retire that one
  test (the vibration E2E is the live thermo proof); the rest of the
  smoke file stands.
- **T2 (nit, verified):** `tests/spectra/test_blueprint.py:7-8` — the
  module docstring still lists `GET /api/build/schema/spectra` and
  `POST /api/spectra/render` as its subjects; the body was updated at P3,
  the header was not.
- **T3 (nit, verified):** `tests/test_pdb_workflow_integration.py:22`
  — the module docstring's step list still narrates old step 5
  (`POST /api/spectra/render …`); the step-4 line was rewritten at P3,
  the step-5 line beside it was missed.

### Area findings (pending the readers + my verification)

*(to be filled)*

## Consolidation decisions

*(to be filled)*
