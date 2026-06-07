# `/structure-optimization` tab — SIESTA/PySCF form

> Status (2026-06-07): the page route is `/structure-optimization`
> (was `/build`).  Per the Phase B reorganization, the structure-
> from-input paths (SMILES generator, file load) have already
> moved to the Molbuilder tab Sources card; this tab is now
> generator-only — it consumes a project-saved structure (picked
> via sidebar dblclick) and emits SIESTA `.fdf` or PySCF `.py`.
>
> `/api/build/*` BACKEND routes are unchanged (the API prefix
> kept its name for stability); only the page route + the in-tab
> generator UI moved.  Doc will be renamed to
> `structure-optimization.md` in a future cleanup.

The user picks a structure via the projects sidebar (dblclick =
commit; single click is preview only per
[`projects-sidebar.md`](../protocols/projects-sidebar.md) § 6).
A form-dirty warning fires before the schema rebuilds if the
user has typed parameter edits since the last commit — see
[`tabs/architecture.md`](architecture.md) § 9.2 (B.5.3) for the
gate.

This doc covers the tab specifically.  Cross-cutting contracts
the tab depends on:
[`projects-sidebar.md`](../protocols/projects-sidebar.md),
[`web-api.md`](../protocols/web-api.md),
[`runtime-registry.md`](../protocols/runtime-registry.md). The
HTTP shapes themselves live in `web-api.md`; this doc covers
the UI loop.

---

## 1. User flow

```
┌─────────────────────────────────────────────────────────────┐
│ Source panel                                                │
│   ○ Build (kind + input text)         ○ Load (file picker)  │
└─────────────────────────────────────────────────────────────┘
            │
            ▼  POST /api/build/molecule  /  POST /api/build/load
┌─────────────────────────────────────────────────────────────┐
│ Viewer (3Dmol) + info line                                  │
│   "<title>"   <n_atoms> atoms · <n_residues> residues · …    │
└─────────────────────────────────────────────────────────────┘
            │
            ▼   structure now in window.molbuilder.lastStructure
┌─────────────────────────────────────────────────────────────┐
│ Generate card                                                │
│   [SIESTA .fdf] [PySCF script]                              │
│       │              │                                       │
│       ▼              ▼                                       │
│   Schema form     Schema form                               │
│       │              │                                       │
│       ▼              ▼                                       │
│   Generate .fdf  Generate .py                               │
│   POST /api/build/fdf   POST /api/build/pyscf               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Source panel

Two modes, mutually exclusive:

**Build** — type a sequence / SMILES / name, click Build.
Submits to `POST /api/build/molecule` with `{kind, input}`.
Supported `kind` values:

| Kind | Input | Backend |
|---|---|---|
| `peptide` | one-letter sequence ("ARNDC") | PeptideBuilder (pure-Python) |
| `dna` / `rna` | one-letter sequence ("ATGC" / "AUGC") | 3DNA → AmberTools → RDKit chain |
| `smiles` | SMILES string ("c1ccccc1") | RDKit |
| `name` | IUPAC name | RDKit + OPSIN |

**Load** — pick a `.xyz` or `.pdb` file from disk via the
project sidebar; submits to `POST /api/build/load` with the
file contents.

Both modes resolve into the same `{ok, n_atoms, n_residues,
elements, atom_names, residue_ids, residue_names, chain_ids,
xyz, title, …}` shape (per `web-api.md`) and pass through the
same renderer.

---

## 3. Schema-driven form (SIESTA + PySCF)

Two tabs inside the Generate card: SIESTA and PySCF. Each
fetches its schema once from `GET /api/build/schema/{siesta,pyscf}`
at page load. The schema is the dataclass-introspection
output (see `web-api.md` § `/api/build/schema`) listing every
form field with its `label`, `unit`, `range`, `tier`, `help`,
and `engine_key`.

`engine_key` is the engine keyword the field writes (or
`(molbuilder: …)` for non-engine knobs). Rendered as a
`<code class="schema-engine-key">` badge next to each label.
Per the source-of-truth principle, every dataclass field MUST
carry this metadata; pinned by
`tests/test_web.py::test_engine_key_present_on_every_{siesta,pyscf,spectra}_form_field`.

---

## 4. Generate buttons

`#generate-fdf` and `#generate-pyscf` are disabled until a
structure exists. After a successful Build / Load they enable.
Click submits the current form state + the structure to the
respective endpoint (`POST /api/build/fdf` /
`POST /api/build/pyscf`); the response renders into the
preview pane.

The response shapes intentionally differ per engine
(`{ok, fdf, system_label, issues}` vs
`{ok, script, job_name, issues}`); both wrap a JS dispatcher
that picks the right keys.

---

## 5. Sidebar integration

The `/build` viewer subscribes to `projects.onChange`. When the
user picks a structure file in the sidebar:

1. The viewer's loader reads the file via `projects.readFile`.
2. `loadStructureText(text)` parses the XYZ/PDB and renders
   the structure.
3. `#info-atoms` updates with the atom count + the Generate
   buttons enable.

The sidebar-pick path goes through the same renderer as the
in-tab Build/Load buttons; downstream Generate-form behavior
is identical regardless of how the structure landed.

---

## 6. Test coverage

| Test file | Layer | Coverage |
|---|---|---|
| `test_web.py` | Flask test_client | Every `/api/build/*` endpoint — happy + bad-input + engine_key metadata |
| `test_build_e2e.py` | Playwright | Page boot + form-schema renders + tab switching + peptide round-trip + sidebar-pick viewer load + **second-visit + external-change** (10 tests total) |

---

## 7. Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-26 | `engine_key` metadata mandatory on every form field; rendered as a `<code>` badge so the user sees which engine keyword each control writes. | Pre-2026-05-26 the form fields had no visible mapping to the engine documentation — users couldn't audit "what does this knob set". The metadata is the dataclass field; the badge is the rendering; the Python-side coverage test catches drops. |
| 2026-06-02 | `/build` gains the audit-tier second-visit + external-change Playwright tests (`TestBuildSecondVisitExternalChange`, 2 tests) so the sidebar-pick → viewer-update wiring is regression-protected across page reloads. | Per the test pattern documented in [`playwright-tests.md`](../protocols/playwright-tests.md) § 9.6 — every tab whose UI is driven by a subscriber-on-state-change needs this coverage to catch the bug class that produced the 2026-06-02 stale-dropdown bug. |
