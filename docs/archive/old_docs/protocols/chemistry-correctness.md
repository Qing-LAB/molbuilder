# Chemistry correctness — the control surface, end to end

Where to start when auditing whether molbuilder's chemistry is
right.  This document is a **map**, not a playbook — it walks
the path a structure takes from "user types a sequence" to
"engine emits a script," names every control point along the
way, and links to the deep doc for each.  Read this first; the
specialised docs answer "how does this layer do its job?"

Companion documents (cited section-by-section below):

| Topic | Doc |
|---|---|
| Scientific intent + cross-engine policy | `docs/science.md` |
| Validation framework (analyzer / adapters / consumers) | `docs/protocols/scientific-validation.md` |
| Builders + backends + hydrogen-placement design | `docs/engines/builders.md` |
| L1 chemistry primitives (charge + protonation) | `docs/types/chemistry.md` |
| Sidecar contract (frozen atoms, region labels) | `docs/protocols/sidecar-contract.md` |
| Validation guard examples in code-audit context | `docs/protocols/code-audit.md § 2` |

---

## § 1 The chemistry control surface, in order

A structure flows through five control points between user input
and engine emission.  Each is a "could be wrong here" location
that an audit must verify.

```
   1. User input                  → CLI / web form
                                    │
                                    ▼
   2. Backend dispatcher           → molbuilder/builders/backends/__init__.py
        ├── 3DNA (X3DNA fiber)     → builders/backends/_threedna.py
        ├── AmberTools (tleap)     → builders/backends/_amber.py
        └── RDKit (SMILES/MMFF)    → builders/backends/_rdkit.py
                                    │
                                    ▼
   3. Chemistry primitives         → molbuilder/chemistry.py
        ├── formal_charge_…        → docs/types/chemistry.md
        └── add_hydrogens          → OpenBabel-first, RDKit-fallback
                                    │
                                    ▼
   4. Analyzer + validator         → molbuilder/chemistry.py
        ├── analyze_structure()    → suggested charge/spin + warnings
        └── check_*() functions    → § 2 of validation doc
                                    │
                                    ▼
   5. Engine emission              → siesta.render_fdf / pyscf.render_script
        └── Pre-emit validate()    → adapter-driven; § 4 of validation doc
```

Each numbered step is owned by ONE module; each module has its
own contract doc (linked above).  An audit of chemistry
correctness walks this stack in order — see § 3 for what to
check at each step.

---

## § 2 Per-step intent (one paragraph each)

### 2.1 User input

A free-form sequence (`ACGT`, SMILES, IUPAC name) or an
already-built structure file (XYZ, PDB).  The CLI and the web
form share the same dataclass-driven dispatch — see
`docs/protocols/web-api.md § 4` (Build) and
`docs/protocols/cli.md`.  The user doesn't choose a backend;
they choose `--backend auto` (or accept the form default) and
the dispatcher picks per-input.

### 2.2 Backend dispatcher

`molbuilder.builders.backends.dispatch(kind, sequence, backend=…)`
routes the request.  `auto` triggers the cascade:
3DNA → AmberTools → RDKit (first-available wins).
`available_backends()` and `auto_backend_name()` report what's
installed and what `auto` would pick.

Backends in scope:

- **3DNA** (`builders/backends/_threedna.py`) — canonical for
  long DNA/RNA double helices.  Builds via X3DNA's `fiber`.
- **AmberTools** (`builders/backends/_amber.py`) — peptide
  builder via `tleap`; falls back for DNA/RNA when 3DNA absent.
- **RDKit** (`builders/backends/_rdkit.py`) — SMILES → 3D coords
  via embedding + MMFF/UFF minimisation.

Deep doc: `docs/engines/builders.md § "Backend dispatcher"`.

### 2.3 Chemistry primitives

After the backend produces raw coordinates, the chemistry layer
imposes hydrogen + charge invariants:

- **Hydrogen addition** — `chemistry.add_hydrogens(struct)` —
  OpenBabel-first, RDKit-fallback.  Every backend path routes
  through this; X3DNA strips its own raw hydrogens and re-adds
  them here so the protonation state is consistent across
  backends.  See `builders.md § "Hydrogen addition"` for the
  per-toolkit rationale.
- **Formal charge** — `chemistry.formal_charge_from_phosphates()`
  — phosphate-aware counter for DNA/RNA strands.  Used by the
  PySCF molecular-charge adapter.  Documented in
  `docs/types/chemistry.md`.

### 2.4 Analyzer + validator

`analyze_structure(struct)` produces:

- `suggested_treatment` — open-shell metal detection, spin
  recommendation, charge defaults.  Drives the UI detection
  chip + the form's auto-detect.
- `metals: list[str]` and `warnings: list[str]` — what
  the user needs to know before running.

Per-engine validators (`molbuilder/validation/{siesta,pyscf}.py`)
call analyzer + apply engine-specific checks (basis-size sanity,
pseudo coverage, k-grid for periodic systems, etc.).  Returns
`list[Issue]`; consumers attach issues to UI cards via
`workflow_group` metadata (web-ui-coherence Rule 2).

Deep doc: `docs/protocols/scientific-validation.md`.

### 2.5 Engine emission

`siesta.render_fdf(struct, cfg)` and `pyscf.render_script(struct, cfg)`
build the script.  Each engine's adapter
(`auto_defaults_for_siesta`, `auto_defaults_for_pyscf`) takes
the analyzer's `suggested_treatment` and emits engine-flavoured
defaults (UKS vs RKS, level_shift, BASIS.Size).  Pre-emission
calls `validate(struct, cfg)` — failures surface as preflight
errors (the issues panel) and block render.

Deep doc: `docs/engines/siesta.md`, `docs/engines/pyscf.md`,
`docs/protocols/scientific-validation.md § 4`.

---

## § 3 Audit checklist — what to verify at each step

When reviewing chemistry correctness in a PR, in a refactor, or
as a structural audit, walk these in order:

### 3.1 At the dispatcher

- `available_backends()` correctly reports installed backends.
  Test: `tests/test_backends.py::test_available_backends_returns_dict_of_bools`.
- `auto_backend_name()` returns the highest-priority available
  backend.  Test: same file, parametrized.
- Adding a new backend doesn't break the cascade (3DNA still
  wins for DNA when present, etc.).

### 3.2 At the backend

- Each backend strips/re-adds hydrogens consistently with the
  shared `chemistry.add_hydrogens`.  Specifically X3DNA — its
  raw output has stylised hydrogens that need replacement.
  See `builders.md § "X3DNA fiber quirks"`.
- AmberTools / tleap: methylene-hydrogen fix
  (`_fix_methylene_hydrogens` at `_amber.py:160`).
- RDKit: AddHs is invoked with explicit valence model so a
  user-supplied SMILES without explicit Hs builds correctly.

### 3.3 At the chemistry primitives

- `add_hydrogens` is called exactly once per structure (don't
  add twice; don't skip).  Test:
  `tests/test_chemistry.py::TestAddHydrogensIdempotent`.
- `formal_charge_from_phosphates` matches user-stated charge
  for canonical DNA/RNA inputs.  See `docs/types/chemistry.md
  § "What it counts"` for the exact rules.

### 3.4 At the analyzer

- `analyze_structure` produces deterministic output for the
  same input (no I/O, no global state).  Test:
  `tests/test_chemistry_analyzer.py::test_analyze_structure_is_deterministic`.
- The detection chip (UI) and the validator (form) read from
  the same `suggested_treatment` source — single-analyzer rule,
  see `web-ui-coherence.md` Rule 1.

### 3.5 At engine emission

- `validate(struct, cfg)` runs before render in EVERY engine path
  (no "render that skips preflight").  Test: shape pinned by
  `tests/test_web.py::test_build_fdf_runs_preflight_before_render`
  + same for pyscf + spectra + transport.
- Issues with `workflow_group` metadata route to the correct
  card (compute / profile / stage).  Test:
  `tests/test_issues_workflow_group.py`.

---

## § 4 What this doc does NOT cover

- The deep "why" for each toolkit choice — see `builders.md`.
- The exact validator rule set per engine — see `scientific-
  validation.md § 4` (per-engine adapters) + the engine docs.
- The science behind specific choices (PBE vs PBE0, UKS vs RKS) —
  see `docs/science.md`.
- The Issue → UI-card attachment rules — see `web-ui-coherence.md`
  Rule 2 and `tests/test_issues_workflow_group.py`.

This doc is intentionally a navigation map.  If you find
yourself wanting to add a deep technical detail here, it
probably belongs in one of the linked specialised docs instead.

---

## § 5 Why this doc exists

A 2026-06-14 audit nearly deleted `/api/modify/load` without
first verifying that the chemistry guards weren't entangled with
it (they weren't — the endpoint was a trivial 3-line wrapper,
not a chemistry path).  The user flagged that without a stitched
overview, future audits would keep hitting the same risk:
"someone proposes deleting / refactoring something; nobody can
quickly check whether the chemistry-correctness chain touches
it."  This doc is the answer to "where is the chain?" — a single
entry point that walks the stack with links to each layer's
deep doc.

Per `docs/protocols/code-audit.md § 2` (audit dimensions): the
"scientific correctness" dimension's checklist lives here, in
§ 3.  When a chemistry-adjacent change is proposed, run § 3
top-to-bottom.

When updating chemistry code, also update this doc's links if
the file paths change — broken links here mean an audit that
walks this map gets lost.
