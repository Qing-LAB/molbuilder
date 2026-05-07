# Modify tab — design and contract

> **This document is the sole source of truth for the Modify feature.**
> Code, tests, and the UI must follow this spec; if any of them diverge,
> update this document in the same commit.  Pointer in `docs/design.md`.

Status (2026-05-07): planning + M1 in progress.

---

## 1. Mission

Construct **metal–molecule–metal nanojunction geometries** for transport-DFT
runs (NEGF/SIESTA, PySCF for cluster reference) by editing an existing
molecule and adding crystalline metal electrodes.  The output is a
`Structure` (XYZ + cell) that the Build tab's existing FDF / PySCF
generators can consume unchanged.

Concretely the feature must let the user:

1. **Load** an existing `.xyz` or `.pdb` (a molecule the user previously
   relaxed, possibly the *N*-th frame of a relaxation trajectory).
2. **Inspect** every atom in a side-panel list with click-to-highlight
   in the 3Dmol viewer (and the reverse: clicking an atom in the viewer
   highlights its row in the list).
3. **Edit individual atoms:**
    a. **Delete** any selected atom(s).
    b. **Add** a new atom anchored to a selected one with `(dx, dy, dz)`
       offset.  Sliders adjust the offset live; a distance readout
       updates as the slider moves.  Commits on Apply.
4. **Define a molecular axis** by selecting two anchor atoms.  Apply a
   rotation that places that pair on the z-axis (the transport-DFT
   convention; ±z carries the electrodes).
5. **Add electrodes:** for each side (+z, -z) independently, pick:
    * Element (Au / Ag / Cu / Ni / Pt / Pd / …)
    * Crystal plane: (100) / (110) / (111)
    * A stack of layers, each with its own (m × n) lateral repeat
      (e.g. `[(3, 3), (4, 4)]` = 3×3 closest, then 4×4 further out)
    * Anchor-to-electrode-center distance (slider)
6. **Hand off** the finished junction to the Build tab so it flows into
   the existing FDF / PySCF generators with the existing validation /
   metadata pipeline.

Out of scope for the initial milestone:

* Asymmetric junctions in the convenience helper (each side configured
  independently is supported via twice-calling the per-side function;
  the UI panel ships symmetric-mode first).
* Non-FCC metals (BCC: Fe, Cr, Mo, W).  The plumbing is in place to add
  them but the in-tree `_FCC_LATTICE_A` table is FCC-only for M1.
* Undo/redo in the UI.  The canonical state lives in the viewer; the
  user re-loads if they want to rewind.
* Multi-molecule junctions (two molecules in parallel between
  electrodes).  One bridging molecule per junction.

---

## 2. User flow (annotated)

```
┌─────────────────────────────────────────────────────────────────────┐
│ Modify tab                                                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌───────────────────┐  ┌──────────────────┐  │
│  │  Atom list       │  │  3Dmol viewer     │  │  Edit panel      │  │
│  │                  │  │                   │  │                  │  │
│  │  click to        │◄─┤  click atom to    │  │  Delete          │  │
│  │  highlight       │  │  select; bond     │  │  Add (sliders)   │  │
│  │  multi-select    │  │  hover shows      │  │  Orient axis     │  │
│  │  via shift-click │  │  distance         │  │  Electrode +z    │  │
│  │                  │  │                   │  │  Electrode -z    │  │
│  └──────────────────┘  └───────────────────┘  └──────────────────┘  │
│                                                                     │
│  [Load XYZ/PDB]  [Send to Build tab]                                │
└─────────────────────────────────────────────────────────────────────┘
```

Walkthrough for the canonical Au-thiol-Au workflow:

1. User clicks **Load** → file picker → `.xyz` of `1,4-benzenedithiol`.
2. Viewer renders 14 atoms (6 C, 4 H, 2 S, 2 H_thiol).
3. User clicks the two thiol H atoms in the viewer (ordering matters:
   first click = `a0`, second click = `a1`); their list rows highlight,
   the **Orient axis** button enables.
4. User clicks **Delete** to remove the two thiol H caps (leaving the
   two S atoms exposed for Au coupling).
5. User re-selects the two S atoms (now the anchor pair).
6. User clicks **Orient along z**.  Backend rotates the structure so the
   S–S vector is on +z; viewer re-centres.
7. User picks **Electrode +z**, sets element=Au, plane=111,
   layers=`[(3,3), (4,4)]`, gap=2.0 Å.  Same for **Electrode -z**.
8. Backend builds two FCC-Au slabs and places them ±z.  Viewer shows
   the full junction (~70 atoms).
9. User clicks **Send to Build tab** → Build tab opens with this
   structure pre-loaded; user clicks Generate .fdf / Generate .py as
   normal.

---

## 3. Architecture

### 3.1 Data flow (stateless server)

```
┌──────────────────────────────────────────────────────────────────┐
│  Browser (Modify tab)                                            │
│                                                                  │
│  state.xyz  ◄─────────── current canonical structure (string) ───┐
│      │                                                           │
│      │ user clicks "Delete" / "Add" / "Orient" / "Electrode"     │
│      ▼                                                           │
│  POST /api/modify/<op>                                           │
│  body: {xyz, op, args}                                           │
│      │                                                           │
└──────│───────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Server (Flask blueprint /api/modify/*)                          │
│                                                                  │
│  parse xyz  ──►  Structure                                       │
│                      │                                           │
│                      │ dispatch on op:                           │
│                      ├── delete_atoms ────┐                      │
│                      ├── add_atom        ─┤                      │
│                      ├── orient_along_axis┤── molbuilder.modify  │
│                      └── add_electrode_slab                      │
│                      │                                           │
│                      ▼                                           │
│                  new Structure                                   │
│                      │                                           │
│                      ▼                                           │
│  serialize to xyz  ──►  response: {ok, xyz, n_atoms, summary}    │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼ (response body becomes new state.xyz; viewer re-renders)
```

The server is **stateless**: each request carries the current structure
as XYZ text, returns the modified structure as XYZ text.  No session
storage; the canonical state lives in the browser.  This mirrors the
existing `/api/build/fdf` and `/api/build/pyscf` shapes and avoids
session-cleanup complexity.

### 3.2 Module layout

```
molbuilder/
├── modify.py                          ◄── pure functions (M1)
│   ├── delete_atoms(struct, indices)
│   ├── add_atom(struct, element, anchor_index, offset)
│   ├── orient_along_axis(struct, anchor_indices, axis="z", center="first")
│   ├── add_electrode_slab(struct, element, plane, layer_sizes,
│   │                      anchor_index, *, gap, side, lattice_constant)
│   └── add_symmetric_electrodes(struct, element, plane, layer_sizes,
│                                anchor_indices, *, gap, lattice_constant)
│
├── cli.py                             ◄── new "modify" subcommand (M1)
│
├── web/
│   ├── blueprints/
│   │   └── modify.py                  ◄── /api/modify/* routes (M3)
│   ├── templates/
│   │   └── index.html                 ◄── new <div id="tab-modify">
│   └── static/
│       └── modify.js                  ◄── Modify tab UI logic (M2-M5)
│
tests/
├── test_modify.py                     ◄── unit tests per op (M1)
└── test_modify_junction.py            ◄── e2e Au-bdt-Au build (M1)
```

The Python module is the foundation; CLI, web blueprint, and JS UI all
sit on top.  Layer order matches our principle #2 (composable scripts +
web UI as a portal, not a separate product).

### 3.3 Phase plan

| Milestone | Scope | Deliverable | Status |
|---|---|---|---|
| **M1** | Pure functions + CLI + tests.  No UI. | `molbuilder/modify.py`, `molbuilder modify` CLI subcommand, `tests/test_modify*.py`.  End-to-end: build a Au(111)-bdt-Au(111) junction from a relaxed BDT XYZ via CLI. | in progress (2026-05-07) |
| **M2** | UI skeleton: tab in shared nav, file load, atom list, click-to-select mirroring viewer ↔ list. | New `static/modify.js`, `tab-modify` panel in `index.html`.  No edits possible yet. | not started |
| **M3** | Edit ops wired: delete, add-with-sliders, live distance.  `/api/modify/atom_op` endpoint. | `web/blueprints/modify.py`; sliders + Apply buttons. | not started |
| **M4** | Anchor-pair selection + orient-along-z. | UI for picking the second anchor; `/api/modify/orient` endpoint. | not started |
| **M5** | Electrode panel (per-layer config, gap slider, symmetric/per-side mode); Send-to-Build handoff. | `/api/modify/electrode`; cross-tab handoff plumbing. | not started |

Each milestone keeps `pytest tests/ -q` green.  No "intermediate broken
state" commits.

---

## 4. Backend API contract

### 4.1 Python module surface (`molbuilder.modify`)

All functions are **pure**: they take a `Structure`, return a new
`Structure` with the same metadata schema; they never mutate the input.

```python
delete_atoms(struct: Structure, indices: Sequence[int]) -> Structure
```

Drop the listed atom indices.  Per-atom metadata arrays
(`atom_names`, `residue_ids`, `residue_names`, `chain_ids`) are sliced
in lockstep.  Out-of-range indices are silently ignored.

```python
add_atom(struct: Structure,
         element: str,
         anchor_index: int,
         offset: Sequence[float],
         *, atom_name: str | None = None,
         residue_name: str = "MOD") -> Structure
```

Append a new atom at `struct.positions[anchor_index] + offset`.  The
new atom gets a fresh `residue_id` (max + 1) so it's distinguishable
from the anchor's residue (handy for later deletes).  `chain_ids[i]`
inherits from the anchor.

```python
orient_along_axis(struct: Structure,
                  anchor_indices: Tuple[int, int],
                  axis: str = "z",
                  center: str = "first") -> Structure
```

Rotate so the vector `pos[a1] - pos[a0]` is parallel to the chosen
axis (Rodrigues formula).  `center`:

* `"first"` (default): translate so `a0` is at the origin.
* `"midpoint"`: translate so the midpoint of `(a0, a1)` is at the origin.
* `"none"`: rotation only, no translation.

```python
add_electrode_slab(struct: Structure,
                   element: str,
                   plane: str,                       # "100" / "110" / "111"
                   layer_sizes: Sequence[Tuple[int, int]],
                   anchor_index: int,
                   *, gap: float = 2.0,
                   side: str = "+z",                 # or "-z"
                   lattice_constant: float | None = None,
                   inter_layer_offset: float | None = None) -> Structure
```

Build an FCC slab via ASE's `fcc{100,110,111}`, mask each layer to its
own (m, n) sub-rectangle, translate so the closest layer sits at
`anchor.z ± gap` with lateral centering on the anchor's (x, y).

```python
add_symmetric_electrodes(struct: Structure,
                         element: str, plane: str,
                         layer_sizes: Sequence[Tuple[int, int]],
                         anchor_indices: Tuple[int, int],
                         *, gap: float = 2.0,
                         lattice_constant: float | None = None) -> Structure
```

Convenience: one call adds the same stack on both ±z sides.  For
asymmetric junctions, call `add_electrode_slab` directly twice.

### 4.2 CLI subcommand (`molbuilder modify`)

Pipe-friendly, chainable flags so a junction can be built in one
command line per principle #2:

```bash
molbuilder modify in.xyz out.xyz \
    --delete 12,13              # drop atoms 12 and 13
    --orient-axis 5,9           # put atoms 5,9 on z (a0 at origin)
    --electrode +z Au:111:3x3,4x4@2.0 \
    --electrode -z Au:111:3x3,4x4@2.0
```

Format of `--electrode` value: `<element>:<plane>:<m>x<n>[,<m>x<n>...]@<gap>`.
`+z` / `-z` chooses the side.  `--add` is omitted for now; the
slider-driven add-atom op is UI-only and rarely useful from a script.

`--delete`, `--orient-axis`, and `--electrode` are applied **in order**
so a single command line corresponds to the user's UI sequence.  Stdin
support (`-` as input path) follows the existing pattern in `cmd_fdf`.

### 4.3 Web blueprint (`/api/modify/*`)

All endpoints accept a JSON body with `xyz` (canonical state) and
op-specific args; respond with `{"ok": bool, "xyz": <new>, "n_atoms": int,
"summary": str, "error": str?}`.  Stateless across requests.

| Endpoint | Body shape | Effect |
|---|---|---|
| `POST /api/modify/load` | `{xyz, format?}` | Validate input.  Echo back canonical xyz (re-parsed; catches malformed input early). |
| `POST /api/modify/delete` | `{xyz, indices: List[int]}` | `delete_atoms` |
| `POST /api/modify/add_atom` | `{xyz, element, anchor_index, offset: [dx,dy,dz]}` | `add_atom` |
| `POST /api/modify/orient` | `{xyz, anchors: [a0,a1], axis?, center?}` | `orient_along_axis` |
| `POST /api/modify/electrode` | `{xyz, element, plane, layer_sizes, anchor_index, gap, side, lattice_constant?}` | `add_electrode_slab` |
| `POST /api/modify/symmetric_electrodes` | `{xyz, element, plane, layer_sizes, anchors:[a0,a1], gap, lattice_constant?}` | `add_symmetric_electrodes` |

All ops run `validate_geometry(struct)` against the **result** structure
and return any warnings in `issues: [...]` (same shape as `/api/build/fdf`).
The UI shows them in an issues panel mirroring the Build tab's design.

---

## 5. Data model invariants

* **Canonical state = XYZ string.**  No cell metadata travels through
  the API; the cell is reconstructed at handoff to the Build tab.
* **Per-atom metadata is preserved** through every op: `atom_names`,
  `residue_ids`, `residue_names`, `chain_ids`.  New atoms get fresh
  residue ids so users can delete just-added atoms cleanly.
* **Electrode atoms** carry `residue_name = "ELC"` and `atom_name = element`
  so they're trivially separable from molecule atoms (e.g. by
  `select(residue_name="ELC")` later).
* **Anchor indices are 0-based** in the Python API; the UI shows them
  1-based to match the existing atom-index labels in the viewer.

---

## 6. Open decisions (recorded for future readers)

| ID | Decision | Status |
|---|---|---|
| D1 | Server is stateless; client holds canonical XYZ. | Accepted — matches `/api/build/*`. |
| D2 | Anchor axis = z (transport-DFT convention). | Accepted. |
| D3 | After electrode placement, cell auto-grows to slab lateral periodicity × (gap + total_thickness + vacuum). | Deferred to M5 (M1 leaves the cell to whatever the Build tab fills in). |
| D4 | Symmetric mode is the default UI; per-side configuration is an advanced toggle. | Accepted. |
| D5 | Live distance during add-atom is computed client-side from slider values; only commits on Apply. | Accepted. |
| D6 | CLI `modify` is one chainable subcommand, not a multi-step interactive shape. | Accepted. |
| D7 | Anchor atoms are NOT removed automatically before electrode placement; the user explicitly deletes them (e.g. thiol H caps before exposing S to Au). | Accepted — keeps the op explicit. |

---

## 7. Off-scope (intentionally not implemented)

* BCC metals (Fe, Cr, Mo, W).  Plumbing is FCC-only in M1.
* HCP metals (Co, Zn, Cd).  Same.
* Surface reconstruction (Au(111) 22×√3 herringbone, etc.).  We use
  ASE's idealised slabs.
* Adsorption-site optimisation.  The user picks `gap` manually; we do
  not search for the energy-minimum Au–S distance.
* Bond-length-aware "smart add" (place the new atom at the canonical
  bond length for the element pair).  M1 takes raw `(dx, dy, dz)`.
* Cell-handling on import.  Loaded XYZ → Structure has no cell; the
  Build tab adds vacuum padding when generating SIESTA FDF.

These are all real follow-ups; they belong in subsequent milestones,
not in M1.
