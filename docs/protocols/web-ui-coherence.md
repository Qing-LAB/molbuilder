# Web-UI coherence rules

## Why this document exists

On 2026-06-13 the user reported that the SIESTA Build form had two
surfaces saying contradictory things about the same Au-BDT-Au
structure:

* The **detection chip** in the Run profile card header said
  *"232 atoms · Au cluster · closed-shell singlet"* — the correct
  conclusion for a metallic Au junction per the noble-metal cluster-
  context analyzer logic shipped earlier the same day.

* The **validator panel** below the Generate button said
  *"Structure contains open-shell transition metal(s) Au but SIESTA
  requests a closed-shell SCF — switch to open-shell SCF"* — the
  opposite recommendation, drawn from a separate code path that had
  not been updated when the analyzer's noble-metal logic landed.

Two surfaces, same form, same data, opposite verdicts. The user's
words:

> "this web UI has so many components that should have been under
> the same roof/workflow with sequential checking and streamlined
> design. now i feel you are just jamming things up."

That was an accurate read. This document writes down the rules we
should have been following.

---

## Rule 1 — One analyzer, one source of truth

There is exactly ONE function that answers chemistry questions about
a `Structure`:

```python
from molbuilder.chemistry import analyze_structure
analysis = analyze_structure(struct)
```

The returned `ChemistryAnalysis` dataclass has the canonical fields:

| Field | Question it answers |
|---|---|
| `n_atoms`, `elements` | Composition |
| `metals` | Which transition metals are present (for basis-set / pseudo warnings) |
| `suggested_charge` | Default net charge |
| `suggested_spin` | Default 2S |
| `suggested_treatment` | **`"open"` or `"closed"` — the single answer to "open-shell or closed-shell?"** |
| `rationale` | Human-readable explanation referencing the literature |
| `warnings` | Caveats the analyzer wants to surface |

**Every UI surface that talks about open-shell vs closed-shell MUST
read `suggested_treatment` from this dataclass.** No exceptions:

* The detection chip in the Run profile card
* The Auto-detect panel's rationale text
* The `check_open_shell_metal` validator
* The PySCF script's UKS/RKS preflight check
* The SIESTA script's `spin_polarized` preflight check

If two surfaces disagree about whether a system is open-shell, the
analyzer is the tie-breaker. Period.

**Anti-pattern (this is what bit us on 2026-06-13):**

```python
# WRONG — bypasses the analyzer's suggested_treatment
metals = detect_open_shell_metals(struct)
if metals and is_closed_shell:
    warn("open-shell metal detected, switch to open-shell SCF")
```

`detect_open_shell_metals` returns "every transition metal physically
present" — for the basis-set warning that's correct (Au still needs
def2 basis). For the open-vs-closed decision, it's the wrong question.

**Correct pattern:**

```python
from .chemistry import analyze_structure
analysis = analyze_structure(struct)
if analysis.suggested_treatment == "open" and is_closed_shell:
    warn("...", message=analysis.rationale)  # cite the analyzer
```

---

## Rule 2 — The form is a workflow; validation must respect it

The Run profile / Stage / Compute & budget cards encode a sequential
workflow:

1. **Run profile** — what is this run? Set once.
2. **Stage convergence target** — what am I converging to right now?
3. **Compute & budget** — how much patience am I willing to spend?

Findings emitted by the validator should attach to the workflow card
they belong to:

| Finding kind | Belongs to |
|---|---|
| Wrong basis for transition metal | Run profile (method choice) |
| Wrong spin for the chemistry | Run profile (spin sub-section) |
| Convergence target unreachable (e.g. SCF tol below numerical noise) | Stage |
| Iteration cap too low for system size | Compute & budget |

**2026-06-13 update.** Card-attached rendering shipped (task #373).
Every workflow-group card the form-schema renderer draws includes
a `<ul class="card-issues" data-workflow-group="<role>" hidden>`
below its fields.  Validator Issues whose underlying field carries
`workflow_group` metadata are routed to the matching card-issues
panel by each engine's renderer:

* SIESTA + PySCF: `web/static/viewer.js::renderIssues` does the
  fan-out (sources panel ID + form container ID).
* Spectra:        `web/static/lib/spectra/core.js::renderIssues`
  (added 2026-06-14 batch F).
* Transport:      `web/static/lib/transport/core.js::_renderIssues`
  (added 2026-06-14 batch F).

The validator side stays simple: emit the Issue with its dataclass
field's `where` and the framework attaches it to the right card by
looking up `workflow_group` on the field metadata
(`web/blueprints/_shared.py::resolve_workflow_group`).  Message text
should **not** name the card explicitly anymore — that's now
redundant double-naming since routing is automatic.  A message like
"switch to open-shell SCF" reads cleanly inside the Stage card.

Untagged issues (geometry / cell / polymer / fields with no
`workflow_group` metadata) still land in the per-engine residual
panel below the cards.

---

## Rule 3 — Detection chip must match validator silence

If the detection chip's text says "closed-shell singlet," the
validator MUST NOT immediately turn around and warn that closed-shell
is wrong. The two surfaces read from the same analyzer; they must
agree about what they recommend.

**Pinned by:** `tests/validation/test_chemistry.py::TestCheckOpenShellMetalUsesAnalyzer::test_au_bdt_au_closed_shell_does_NOT_warn`

---

## Rule 4 — One palette, one role-name vocabulary

CSS roles (`profile` / `stage` / `budget`) match dataclass metadata
roles match user-visible card titles. If we ever need to rename
the role internally (`system` → `profile` was the most recent
example, 2026-06-13), the rename touches:

1. The dataclass metadata `workflow_group` value
2. The CSS class `.workflow-group--<role>`
3. The token name `--group-<role>-accent`
4. The JS role-key strings in `_renderWorkflowGroupChips`
5. The card-title string in `WORKFLOW_GROUP_META`
6. The invariant test allowed-values set in
   `tests/test_live_poll_invariants_audit.py`

All six in the same commit, or the rename is half-done and the form
silently breaks. The 2026-06-13 rename caught this; future renames
should follow the same checklist.

---

## Rule 5 — A new UI surface that displays chemistry MUST come with a test that pins agreement with the analyzer

Concretely:

* New chip / banner / status line that reports "open-shell" or
  "closed-shell" → add a test that loads a known structure, runs the
  analyzer, runs the validator, and asserts both reach the same
  conclusion for the same `is_closed_shell` config.
* New auto-detect button / suggestion popup → add a test that
  asserts what the button writes to the form matches
  `ChemistryAnalysis.suggested_*`.

The agreement-tests live in `tests/validation/test_<submodule>.py` for backend
agreement and `tests/test_live_poll_invariants_audit.py` for source-
text JS agreement.

---

## What this document does NOT cover

* Visual design (colors, spacing, typography) — those live in
  `lib/tokens.css` and `lib/form-schema.css`.
* Script-emission rules (FDF / PySCF script generation) — those live
  in `docs/engines/siesta.md` and `docs/engines/pyscf.md`.
* Live-watch and trajectory inspector contracts — those live in
  `docs/protocols/results-tab.md` and
  `docs/protocols/playwright-tests.md` § A10.

This document is specifically about **the UI's internal coherence**:
when two surfaces in the same form report on the same structure,
they must reach the same conclusion. That's the rule the 2026-06-13
Au-BDT-Au incident violated, and that's what this document writes
down so the next violation lands in code review with a clear citation.

---

## References

* `molbuilder/chemistry.py::analyze_structure` — the canonical analyzer.
* `molbuilder/validation/chemistry.py::check_open_shell_metal` — the validator
  that reads `analysis.suggested_treatment` (2026-06-13 fix).
* `molbuilder/web/static/lib/detection-chip.js` — the shared detection
  chip helper (`buildText` + `render`) that every engine tab consumes;
  reads `n_atoms`, `metals`, and the per-engine suggested-params
  block from the `/api/structure/analyze` response.
* `tests/validation/test_chemistry.py::TestCheckOpenShellMetalUsesAnalyzer` —
  the analyzer-validator agreement pins.
* [`scientific-validation.md`](scientific-validation.md) — the
  companion doc; **§ 3.4** holds the noble-metal vs open-d-shell
  cluster-context rule, **§ 5.3** holds the consumer list (every
  surface that delegates to `check_open_shell_metal`), and **§ 10**
  proposes the validator-package split that completes the
  organisation work this doc names.
