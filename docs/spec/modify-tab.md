# Modify tab — design and contract

> **This document is the sole source of truth for the Modify feature.**
> Code, tests, and the UI must follow this spec; if any of them diverge,
> update this document in the same commit.  Pointer in `docs/design.md`.

Status (2026-05-08): M1 done (Python API + CLI + tests); M2-M5 not yet started.

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
5. **Add electrodes:** for each side (+z, -z), the user fills out one
    "stack" panel per call.  Per panel:
    * **Element** (dropdown — Au / Ag / Cu / Ni / Pt / Pd; see § 8).
    * **Crystal plane** (dropdown — 100 / 110 / 111).
    * **Cell shape** (toggle — *primitive* vs *orthogonal*; only fcc(111)
      has a real choice, the toggle is hidden / disabled for 100 / 110.
      Constraints from ASE bubble up as inline error hints if the user
      picks an incompatible (m, n).  See § 8).
    * **In-plane size** *m* × *n* (two integer inputs; integers ≥ 1).
    * **Number of layers** *n_layers* (integer input ≥ 1).
    * **Gap** — meaning depends on the panel mode:
        * Pair-mode (symmetric electrodes from one panel): the total
          electrode-to-electrode separation; default **8.0 Å** (matches
          `add_symmetric_electrodes(gap=8.0)`).
        * Single-mode (one slab, one anchor): the anchor-to-closest-
          layer contact distance; default **2.4 Å** (matches
          `add_electrode_slab(contact_distance=2.4)`).
    * **Lateral offset** Δx, Δy (two sliders, default 0.0 Å each).
      Default places the slab centroid directly under / over the
      anchor (atop site).  Adjust to park the anchor on bridge or
      hollow sites.
    * **Lattice constant** override (advanced text input; default
      from `molbuilder/data/fcc_lattice.json`).
    * **Inter-layer spacing** override (advanced text input; default
      from ASE for the chosen lattice constant).
    All layers in a single panel share the same (m × n) — uniform.
    For a stepped contact (e.g. 3×3 closest, 4×4 further out), the
    user adds **two** stacks on the same side: one panel with
    `(m=3, n=3, n_layers=K_inner, gap=g_inner)`, then another with
    `(m=4, n=4, n_layers=K_outer, gap=g_outer)` where `g_outer` is
    set so the second stack lands beyond the first.
6. **Hand off** the finished junction to the Build tab so it flows into
   the existing FDF / PySCF generators with the existing validation /
   metadata pipeline.

Out of scope for the initial milestone:

* Asymmetric junctions in the convenience helper (each side configured
  independently is supported via twice-calling the per-side function;
  the UI panel ships symmetric-mode first).
* Non-FCC metals (BCC: Fe, Cr, Mo, W).  The plumbing is in place to add
  them but the in-tree FCC lattice table (loaded lazily by
  `molbuilder.modify._get_fcc_lattice`, with the closed list of metals
  in `SUPPORTED_FCC_ELEMENTS`) is FCC-only for M1.
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
7. User opens the **Electrode** panel (default = pair mode).  Picks
   element=Au, plane=111, primitive cell, m=3, n=3, n_layers=2, gap=9.0 Å
   (electrode-to-electrode distance), offset=(0, 0).  One panel = one
   pair of slabs (top + bottom) added in one shot.  For a stepped
   3×3 + 4×4 contact, the user adds a second pair-mode panel with
   m=4, n=4, n_layers=1 and a larger gap.
8. Backend builds the FCC-Au slabs and places them ±z, both
   lateral-centred on the anchor-pair midpoint.  Viewer shows the
   full junction.
9. User clicks **Send to Build tab** → Build tab opens with this
   structure pre-loaded; user clicks Generate .fdf / Generate .py as
   normal.

For an asymmetric junction (different metal / size on each side), the
user toggles a panel into **single-electrode mode** (rare); each
single-mode panel adds one slab with `contact_distance` (anchor-to-
closest-layer) instead of `gap`.

### 2.1 Tilted molecules

The `--orient-axis` operation accepts an `--angle θ` parameter (degrees,
default 0) that **tilts** the anchor pair away from the target axis by
θ° in a fixed default plane (xz for `--axis z`).  The anchor-pair
midpoint stays at the origin; only the orientation changes.

When a tilted molecule then has electrodes added in **pair mode**:

* The two electrodes are **always collinear along z**, regardless of
  molecule tilt.  This is intentional and matches real junction
  geometry: metal contacts are crystallographic, the molecule fits
  whatever pose works between them.
* `gap` is the z-distance between the closest layer of the +z slab
  and the closest layer of the -z slab -- still measured along z,
  not along the molecule's tilted axis.
* The anchor-pair midpoint's xy coordinates determine where the
  slabs sit laterally.  Both slabs are centred on
  `(mid.x + offset[0], mid.y + offset[1])` so a tilt that swings the
  top anchor off-axis does NOT swing the top electrode along with
  it -- the electrode stays where the molecule's lateral centre is.
* The tilt direction can be redirected with `rotate_around_axis`
  (`--rotate z:90` to spin the tilt from xz-plane into yz-plane,
  for example).

If you want the slabs to follow the anchor positions instead of the
midpoint -- e.g. for a non-canonical junction where each contact is
rigidly attached to one anchor -- use single-mode `--electrode`
twice with the same `contact_distance` and the appropriate
`+z=N` / `-z=N` anchors.  In that case each slab anchors directly
to its own atom, picking up any lateral offset the tilt introduced.

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
│   ├── orient_along_axis(struct, anchor_indices, axis="z",
│   │                     *, angle=0.0, center="midpoint")
│   ├── rotate_around_axis(struct, axis="z", angle=0.0)
│   ├── add_electrode_slab(struct, element, plane, size, anchor_index,
│   │                      *, contact_distance, side, orthogonal, offset,
│   │                      lattice_constant, inter_layer_offset)
│   └── add_symmetric_electrodes(struct, element, plane, size,
│                                anchor_indices, *, gap, orthogonal,
│                                offset, lattice_constant)
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
| **M1** | Pure functions + CLI + tests.  No UI. | `molbuilder/modify.py`, `molbuilder modify` CLI subcommand, `tests/test_modify*.py`.  End-to-end: build a Au(111)-bdt-Au(111) junction from a relaxed BDT XYZ via CLI. | **done (2026-05-08)** |
| **M2** | UI skeleton: tab in shared nav, file load, atom list, click-to-select mirroring viewer ↔ list. | New `static/modify.js`, `tab-modify` panel in `index.html`.  No edits possible yet. | not started |
| **M3** | Edit ops wired: delete, add-with-sliders, live distance.  `/api/modify/atom_op` endpoint. | `web/blueprints/modify.py`; sliders + Apply buttons. | not started |
| **M4** | Anchor-pair selection + orient-along-z. | UI for picking the second anchor; `/api/modify/orient` endpoint. | not started |
| **M5** | Electrode panel (one panel per stack -- size, gap, offset sliders, orthogonal toggle when meaningful, symmetric/per-side mode); Send-to-Build handoff. | `/api/modify/electrode`; cross-tab handoff plumbing. | not started |

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
                  *, angle: float = 0.0,
                  center: str = "midpoint") -> Structure
```

Rotate so the vector `pos[a1] - pos[a0]` forms `angle` (degrees) with
the chosen target axis.  Default `angle=0.0` puts the anchor pair
exactly along the axis (canonical case).  Non-zero `angle` tilts the
molecule by that many degrees in a fixed plane:

* `axis="z"` -> tilt in the **xz-plane** (anchor pair vector becomes
  `(sin θ * d, 0, cos θ * d)`)
* `axis="x"` -> tilt in the **xy-plane**
* `axis="y"` -> tilt in the **yz-plane**

For tilt directions outside the default plane, follow this with
`rotate_around_axis` to spin around the target axis.

`center` (default `"midpoint"`):

* `"midpoint"`: translate so the midpoint of `(a0, a1)` lands at the
  origin -- required for pair-mode `add_symmetric_electrodes` to
  work cleanly (the gap is centred on the anchor-pair midpoint).
* `"first"`: translate so `a0` is at the origin.
* `"none"`: rotation only, no translation.

```python
rotate_around_axis(struct: Structure,
                   axis: str = "z",
                   angle: float = 0.0) -> Structure
```

Rotate every atom by `angle` (degrees, right-hand rule) around the
named axis through the origin.  Useful for redirecting a tilted
molecule's azimuth after `orient_along_axis(angle=...)`.

```python
add_electrode_slab(struct: Structure,
                   element: str,
                   plane: str,                            # "100" / "110" / "111"
                   size: Tuple[int, int, int],            # (m, n, n_layers); uniform
                   anchor_index: int,
                   *, contact_distance: float = 2.4,
                   side: str = "+z",                      # or "-z"
                   orthogonal: bool = False,
                   offset: Tuple[float, float] = (0.0, 0.0),
                   lattice_constant: float | None = None,
                   inter_layer_offset: float | None = None) -> Structure
```

**Single-electrode primitive.**  Build a uniform FCC slab via ASE's
`fcc{100,110,111}` with `m × n` in-plane repeats and `n_layers`
layers, all of identical size.  Translate so the slab's lateral
centroid sits at `(anchor.x + offset[0], anchor.y + offset[1])` and
the closest layer's z is `anchor.z ± contact_distance` (sign per
`side`).  For pair junctions, prefer `add_symmetric_electrodes`
which takes `gap` (electrode-to-electrode distance) directly.

```python
add_symmetric_electrodes(struct: Structure,
                         element: str, plane: str,
                         size: Tuple[int, int, int],
                         anchor_indices: Tuple[int, int],
                         *, gap: float = 8.0,
                         orthogonal: bool = False,
                         offset: Tuple[float, float] = (0.0, 0.0),
                         lattice_constant: float | None = None) -> Structure
```

**Pair-electrode primitive.**  `anchor_indices = (a_top, a_bot)` --
the +z anchor first, the -z anchor second.  Computes
`mid = 0.5 * (positions[a_top] + positions[a_bot])`, then places the
two slabs collinear along z at `mid.z ± gap/2`, both lateral-centred
on `(mid.x + offset[0], mid.y + offset[1])`.  For a tilted molecule
(anchor pair off-z), the two electrodes still lie collinear along z;
the molecule fits its tilted geometry between them.

`gap` is the canonical **junction gap** -- the empty z-space between
the two electrodes.  Internally, each side gets the per-side contact
distance `(gap - anchor_separation_z) / 2`.  If `gap` is smaller than
the anchor pair's z-extent, raises `ValueError` (the electrodes would
overlap the molecule).

For asymmetric junctions (different size / offset / metal per side,
or stepped contacts), call `add_electrode_slab` directly per side.

### 4.2 CLI subcommand (`molbuilder modify`)

**One operation TYPE per call.**  Chain calls via stdin/stdout
(`-` as input or output path) for multi-step workflows.  Mixing
operation TYPES in a single call is rejected.

Operation types:

| Flag | Purpose | Multi-instance? |
|---|---|---|
| `--delete INDICES` | drop atoms (0-based) | yes -- flatten |
| `--orient-axis A0,A1` | rotate anchor pair onto `--axis`; `--angle θ` tilts θ°; `--center` controls translation | no |
| `--rotate AXIS:ANGLE` | spin every atom around AXIS by ANGLE° | no |
| `--electrode SPEC` | add one or two FCC slabs (see spec format below) | yes -- apply in order |

`--electrode` spec format -- two modes distinguished by the
`@key=val` substring:

* **Pair (default):** `ELEM:PLANE:MxNxL@gap=GAP:ATOP,ABOT`
  -- `gap` = total electrode-to-electrode distance (Å); `ATOP,ABOT`
  are the +z and -z anchor indices.  Single flag, two slabs.
* **Single (rare):** `ELEM:PLANE:MxNxL@contact=DIST:+z=N` or
  `:-z=N` -- `contact` = anchor-to-closest-layer distance (Å); the
  `±z=N` field gives side and anchor index.

Call-level flags (apply uniformly to every `--electrode` in this call):

| Flag | Effect |
|---|---|
| `--orthogonal` | use ASE's orthogonal supercell (only meaningful for fcc(111)) |
| `--electrode-offset Δx,Δy` | lateral shift in Å applied to every slab's centroid |
| `--lattice-constant Å` | override the value from `molbuilder/data/fcc_lattice.json` |

Pipeline example -- canonical Au-bdt-Au junction:

```bash
# Three-step pipe: orient, then pair-mode electrode, then write the file
molbuilder modify bdt.xyz - --orient-axis 0,3 |
  molbuilder modify - junction.xyz \
      --electrode Au:111:3x3x2@gap=9.0:3,0
```

Stepped 3×3 + 4×4 contact, both sides, in one electrode call:

```bash
molbuilder modify oriented.xyz junction.xyz \
    --electrode Au:111:3x3x1@gap=9.0:3,0  \
    --electrode Au:111:4x4x1@gap=14.0:3,0
```

Asymmetric junction (Au top, Cu bottom) via two single-mode calls:

```bash
molbuilder modify oriented.xyz step1.xyz \
    --electrode Au:111:3x3x2@contact=2.4:+z=3
molbuilder modify step1.xyz junction.xyz \
    --electrode Cu:111:3x3x2@contact=2.0:-z=0
```

Tilt the molecule 30° in the xz-plane (still anchor pair midpoint
at origin), then spin 90° around z to redirect the tilt into yz:

```bash
molbuilder modify oriented.xyz - --orient-axis 0,3 --angle 30 |
  molbuilder modify - tilted.xyz --rotate z:90
```

Stdin (`-` as input) and stdout (`-` as output) follow the existing
pattern in `cmd_fdf` / `cmd_pyscf`.  Output is XYZ by default; pass
`--output-format pdb` to write PDB instead.

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
| `POST /api/modify/rotate` | `{xyz, axis, angle}` | `rotate_around_axis` |
| `POST /api/modify/electrode` | `{xyz, element, plane, size:[m,n,n_layers], anchor_index, contact_distance, side, orthogonal, offset:[dx,dy], lattice_constant?, inter_layer_offset?}` | `add_electrode_slab` (single mode) |
| `POST /api/modify/symmetric_electrodes` | `{xyz, element, plane, size:[m,n,n_layers], anchors:[a_top,a_bot], gap, orthogonal, offset:[dx,dy], lattice_constant?}` | `add_symmetric_electrodes` (pair mode; `gap` = electrode-to-electrode distance) |

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

* BCC metals (Fe, Cr, Mo, W).  Plumbing is FCC-only.
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

---

## 8. Supported electrode metals (closed list)

The Modify tab supports exactly these six FCC metals; the list is
closed at the API boundary (`SUPPORTED_FCC_ELEMENTS` in
`molbuilder.modify`) so the UI dropdown, the CLI's `--electrode`
parser, and the Python function reject anything else with a clear
`ValueError`.

The actual numeric values live in `molbuilder/data/fcc_lattice.json`
with the full citation chain in `molbuilder/data/README.md` — see
**§ 9 below** for the data-directory contract.  The README's table is
reproduced here for convenience but `molbuilder/data/README.md` is
canonical:

| Symbol | Element | Lattice constant *a* (Å) | Crystal system |
|---|---|---|---|
| **Au** | Gold      | 4.0782 | FCC |
| **Ag** | Silver    | 4.0853 | FCC |
| **Cu** | Copper    | 3.6149 | FCC |
| **Ni** | Nickel    | 3.5240 | FCC |
| **Pt** | Platinum  | 3.9242 | FCC |
| **Pd** | Palladium | 3.8907 | FCC |

**Why a closed list:**

1. The fcc100 / fcc110 / fcc111 builders in ASE accept *any* element
   symbol; passing Fe (BCC) silently returns a wrong-symmetry slab.
2. The lattice-constant table needs an authoritative source per entry
   (we use Wyckoff 1963 — see the data README); an open list invites
   users to type a Z-label and get whatever ASE default ships, which
   can drift between ASE versions.
3. The molecular-electronics use case is concentrated on these six;
   adding Ru / Os / Ir is real future work but isn't current scope.

**Override** the lattice constant per call with the
`lattice_constant=` kwarg on `add_electrode_slab` (or
`--lattice-constant` on the CLI) when you need a strained-lattice
slab or a non-room-temperature value.  For a persistent override
across reinstalls, use the `MOLBUILDER_DATA_DIR` environment variable
(see § 9).

---

## 9. Fundamental-data directory contract

Reference values that the code reads as input — lattice constants,
future bond-length tables, future ECP defaults — live in
`molbuilder/data/` rather than as Python literals.  Two reasons:

* **Auditable provenance.**  Every numeric value has a citation in
  `molbuilder/data/README.md`.  A reviewer can verify a value against
  the cited source without reading code.
* **User-updatable.**  An end user edits a JSON file (or sets
  `MOLBUILDER_DATA_DIR` to point at a copy elsewhere) to use their
  own DFT-equilibrium constants without touching the package source.

### Layout

```
molbuilder/data/
├── README.md             # citations, sources, override mechanism, schema
├── fcc_lattice.json      # supported FCC metals (closed list)
└── (future tables here, each documented in README.md)
```

### Loader contract

Every Python module that reads from `molbuilder/data/` must:

1. Walk `_data_dir_candidates()` (defined in `molbuilder.modify`):
    a. `$MOLBUILDER_DATA_DIR/<file>` if the env var is set.
    b. `<package>/data/<file>` (the bundled copy).

2. Raise a clear `RuntimeError` if no candidate file exists, listing
   every searched path.  The user must see exactly which files were
   tried so a typo in the override path is obvious.

3. Validate the schema (presence of required keys, parsable values)
   and raise a clear error on mismatch — a malformed override should
   fail loudly at import, not silently when the wrong slab gets built
   later.

### Adding a new data file

1. Drop the file in `molbuilder/data/` with a snake-case name and a
   schema version field (`"_format": "<table-name> v1"`).
2. Document its schema and every value in `molbuilder/data/README.md`
   with a citation per value.
3. Wire it into the relevant Python module via a `_load_*()` helper
   that follows the loader contract above.
4. Add it to `[tool.setuptools.package-data]` in `pyproject.toml`
   under `data/*.json` (or whatever pattern matches).
5. Update this spec (or the relevant feature spec) to point at the
   data file from the surface that consumes it.
