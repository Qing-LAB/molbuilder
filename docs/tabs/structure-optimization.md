# `/structure-optimization` tab — SIESTA/PySCF form

> Status (2026-06-08): the page route is `/structure-optimization`
> (was `/build`).  Per task #295 the in-tab Build/Load form (kind +
> sequence + backend dropdown + add-hydrogens select + file-upload
> button + Build/Load source-mode toggle) was retired — every
> structure-from-input path now lives on the Molbuilder tab's
> Init-structure card.  This tab is **file-driven**: it consumes
> a project-saved `.xyz` / `.pdb` via the Projects sidebar and
> emits SIESTA `.fdf` or PySCF `.py`.
>
> `/api/build/*` BACKEND routes are unchanged (the API prefix
> kept its name for stability); only the page route + the in-tab
> UI moved.

The user picks a structure via the projects sidebar (dblclick =
commit; single click is preview only per
[`projects-sidebar.md`](../protocols/projects-sidebar.md) § 6),
or clicks the explicit "Load from sidebar selection" button next
to the viewer.  A form-dirty warning fires before the structure
load if the user has typed parameter edits since the last commit
— see [`tabs/architecture.md`](architecture.md) § 9.2 (B.5.3) for
the gate.

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
│ 1. Inspect structure                                         │
│   [ Load from sidebar selection ]   Selected: foo.xyz       │
│   ┌──────────────────────────────┐                          │
│   │ 3Dmol viewer (square)        │                          │
│   └──────────────────────────────┘                          │
│   info: <title> · n_atoms · n_residues · formula             │
└─────────────────────────────────────────────────────────────┘
            │
            ▼  POST /api/build/load  (sidebar pick reads + posts text)
            │  (cross-tab handoff = sessionStorage handover;
            │   see § 5 for the auto-load contract)
            │
┌─────────────────────────────────────────────────────────────┐
│ 2. Generate input (optional)                                 │
│   [SIESTA .fdf]  [PySCF script]                              │
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

## 2. Inspect structure card

One structure entry point: **Load from sidebar selection**.  The
button reads the Projects-sidebar's current pick
(`projects.getCurrentFile()` → `sessionStorage.molbuilder.current_file`),
checks the extension (`.xyz` or `.pdb`), reads the file via
`/api/files/read`, and posts the bytes to `POST /api/build/load`.

The button enables only when the sidebar's current pick is a
loadable extension; the readout next to it shows one of:

  * `Pick a .xyz / .pdb in the Projects sidebar.` — no pick.
  * `Selected: foo.xyz` — pick is loadable but not yet loaded.
  * `Selected: foo.txt (not loadable)` — wrong extension.
  * `Loaded: foo.xyz` — the loaded file matches the current pick.

Dblclick on a sidebar `.xyz`/`.pdb` also commits via the
universal interaction model (single-click = preview, double-click
= commit; see [`projects-sidebar.md`](../protocols/projects-sidebar.md)
§ 6).  Both paths converge in `_commitStructure(sel)` inside
`viewer.js`.

The response shape is the canonical `{ok, n_atoms, n_residues,
elements, atom_names, residue_ids, residue_names, chain_ids, xyz,
title, …}` payload (per `web-api.md`).  The same payload drives
the cross-tab handoff (§ 5) — the dataclass is the single source
of truth (memory `feedback_dataclass_source_of_truth`).

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
structure exists. After a successful Load they enable. Click
submits the current form state + the structure to the respective
endpoint (`POST /api/build/fdf` / `POST /api/build/pyscf`); the
response renders into the preview pane.

The response shapes intentionally differ per engine
(`{ok, fdf, system_label, issues}` vs
`{ok, script, job_name, issues}`); both wrap a JS dispatcher
that picks the right keys.

---

## 5. Sidebar integration and cross-tab handoff

The Optimization tab's `viewer.js` waits for
`runtime.whenReady("projects")` then subscribes to the sidebar:

1. **Mount-time auto-load.** On every page load, the viewer reads
   `projects.getCurrentFile()`.  When the pointer names a loadable
   file, `_commitStructure({dir, file})` fires once.  This is what
   makes the Molbuilder tab's **save-first Send-to-Optimization**
   workflow round-trip: Modify writes the saved file path into
   `sessionStorage.molbuilder.current_file` and navigates; the
   Optimization mount auto-load picks it up without any extra
   gesture from the user (task #294).

2. **`onCommit` subscription.** Dblclick on a sidebar
   `.xyz`/`.pdb` and the "Load from sidebar selection" button
   both publish through the same `_commitStructure` path.  The
   `_sidebarLastFile` guard inside the function debounces against
   same-file re-fires from auto-load + onCommit landing on the
   same path.

The sidebar-pick path goes through the same renderer as the
mount-time auto-load and the explicit button click; downstream
Generate-form behavior is identical regardless of how the
structure landed.

---

## 6. Test coverage

| Test file | Layer | Coverage |
|---|---|---|
| `test_web.py` | Flask test_client | Every `/api/build/*` endpoint — happy + bad-input + engine_key metadata |
| `test_build_e2e.py` | Playwright | Page boot + form-schema renders + tab switching + sidebar-load round-trip + **second-visit + external-change** |
| `test_modify_e2e.py` | Playwright | Cross-tab handoff via `sessionStorage.molbuilder.current_file` + preflight listener wiring on the loaded structure |

---

## 7. Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-26 | `engine_key` metadata mandatory on every form field; rendered as a `<code>` badge so the user sees which engine keyword each control writes. | Pre-2026-05-26 the form fields had no visible mapping to the engine documentation — users couldn't audit "what does this knob set". The metadata is the dataclass field; the badge is the rendering; the Python-side coverage test catches drops. |
| 2026-06-02 | `/build` gains the audit-tier second-visit + external-change Playwright tests so the sidebar-pick → viewer-update wiring is regression-protected across page reloads. | Per the test pattern documented in [`playwright-tests.md`](../protocols/playwright-tests.md) § 9.6 — every tab whose UI is driven by a subscriber-on-state-change needs this coverage to catch the bug class that produced the 2026-06-02 stale-dropdown bug. |
| 2026-06-08 | The in-tab Build/Load form (kind + sequence + backend + add-hydrogens + file upload + source-mode toggle) is retired.  Sidebar-load + the Molbuilder tab's save-first Send become the sole structure entry points. | The Molbuilder tab already carries every generator (peptide / DNA / RNA / SMILES / name / upload / load).  Mirroring the same UI on `/structure-optimization` was duplicated maintenance and confused users about which tab "owns" structure creation.  Concentrating structure-from-input on `/molbuilder` and structure-from-file on `/structure-optimization` makes each tab's job singular.  Memory `feedback_three_stage_contract` + `feedback_no_backward_compat` — the old form was deleted, not aliased. |
