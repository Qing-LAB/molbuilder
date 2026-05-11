# Modify tab — design and contract

> **This document is the sole source of truth for the Modify feature.**
> Code, tests, and the UI must follow this spec; if any of them diverge,
> update this document in the same commit.  Pointer in `docs/design.md`.

Status (2026-05-10): All milestones done.  M1 Python API + CLI + tests; M2 UI skeleton (route, atom list ↔ viewer click sync, file load); M3 delete + add-atom; M4 anchor-pair orient + rotate, per-atom info panel, xyz-axes overlay; M5 electrode panel (pair / single mode) + Send-to-Build handoff via the same ``builder-structure`` sessionStorage key Phase 1 cross-tab persistence uses; M6 Geom subtab (centre-at-origin, translate-by-Δ), anchorless slab mode (slabs at z = ±gap/2 around the world origin), wireframe halo selection marker, Focus-molecule button + rotation-pivot snap on left-drag, slab-only Undo (HISTORY_MAX = 20), single-source-of-truth `/api/modify/meta` for the element + plane dropdowns.

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
3. **Edit individual atoms (Atom subtab):**
    a. **Delete** any selected atom(s).
    b. **Add** a new atom anchored to a selected one with `(dx, dy, dz)`
       offset.  Sliders adjust the offset live; a distance readout
       updates as the slider moves.  Commits on Apply.
4. **Pose the molecule (Pose subtab):**
    a. **Orient anchor pair along an axis** by selecting two atoms
       and applying a rotation that places that pair on the z-axis
       (the transport-DFT convention; ±z carries the electrodes).
       A tilt slider inclines the pair away from z by 0–90°.
    b. **Rotate around an axis** -- spin every atom by N° around x,
       y, or z through the origin.  Useful for redirecting a tilted
       molecule's azimuth.
5. **Place the molecule (Geom subtab):**
    a. **Centre at origin** -- one button click; translates the
       structure so the geometric centroid (atom-coordinate mean)
       lands at (0, 0, 0).
    b. **Translate by (Δx, Δy, Δz) Å** -- explicit number inputs;
       useful for nudging the centroid to a specific point after
       centring or before adding electrodes.
6. **Add electrodes (Junction subtab):** the user fills out one
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
          electrode-to-electrode z-distance; UI default **12.0 Å**
          (range 4–30 Å).  The Python default for
          `add_symmetric_electrodes(gap=...)` is 8.0; the UI raises
          it because most published junctions (oligophenyl, OPV3,
          alkanedithiols n=4–10) need 12–20 Å.
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
7. **Hand off** the finished junction to the Build tab so it flows into
   the existing FDF / PySCF generators with the existing validation /
   metadata pipeline.

Plus a slab-scoped **Undo** for the Junction subtab (see § 2.3) and a
**Focus molecule** camera affordance (see § 2.2).

Out of scope:

* Asymmetric junctions in the convenience helper (each side configured
  independently is supported via twice-calling the per-side function;
  the UI panel ships symmetric-mode first).
* Non-FCC metals (BCC: Fe, Cr, Mo, W).  The plumbing is in place to add
  them but the in-tree FCC lattice table (loaded lazily by
  `molbuilder.modify._get_fcc_lattice`, with the closed list of metals
  in `SUPPORTED_FCC_ELEMENTS`) is FCC-only.
* General-purpose Undo/redo for non-slab ops.  Undo is **slab-only**
  (see § 2.3) -- the user typically iterates electrode parameters and
  rolls back; deletes / rotates / translates are committed.
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

Walkthrough for the canonical Au-thiol-Au workflow (M6 anchorless flow):

1. User clicks **Load** → file picker → `.xyz` of `1,4-benzenedithiol`.
2. Viewer renders 14 atoms (6 C, 4 H, 2 S, 2 H_thiol).
3. User clicks the two thiol H atoms in the viewer (ordering matters:
   first click = `a0`, second click = `a1`); their list rows highlight,
   the **Orient axis** button enables.
4. (Atom subtab) User clicks **Delete** to remove the two thiol H caps
   (leaving the two S atoms exposed for Au coupling).
5. (Pose subtab) User re-selects the two S atoms (now the anchor pair)
   and clicks **Orient along z** with `center=midpoint` (default) so
   the S–S midpoint lands at the origin.
6. (Geom subtab) Optional: click **Centre at origin** if the loaded
   geometry was not pre-centred; this puts the molecule's centroid on
   (0, 0, 0).  After the orient step in 5, this is usually a no-op
   for symmetric molecules.
7. (Junction subtab, default = pair mode) User picks element=Au,
   plane=111, primitive cell, m=3, n=3, n_layers=2, gap=12 Å (UI
   default; raise to 14–18 Å for longer molecules), offset=(0, 0).
   With **no atoms selected**, this is the canonical anchorless flow:
   slabs land at z = ±gap/2 around the world origin.
8. Backend builds the FCC-Au slabs perpendicular to z, both lateral-
   centred on the (offset) origin.  Viewer shows the full junction.
9. (Optional iteration) User adjusts gap / m / n, hits **Apply Add
   Electrode** again to replace the slab, or hits **Undo** to revert
   to the pre-slab structure (slab-only undo, up to HISTORY_MAX = 20
   ops; see § 2.3).
10. User clicks **Send to Build tab** → Build tab opens with this
    structure pre-loaded; user clicks Generate .fdf / Generate .py as
    normal.

**Legacy anchored flow** (slabs flank a specific atom pair in xy +
z): the user selects exactly two atoms before clicking Apply Add
Electrode in pair mode.  The slabs are then placed with the slab
midpoint on the anchor-pair midpoint, **not** the origin.  Useful
when the molecule is NOT pre-centred and the user wants the slabs
to follow the anchor positions directly.

For an asymmetric junction (different metal / size on each side), the
user toggles a panel into **single-electrode mode** (rare); each
single-mode panel adds one slab with `contact_distance` (anchor-to-
closest-layer) instead of `gap`.

### 2.2 Camera anchoring

3Dmol's rotation/zoom pivot is the camera lookAt point.  In a typical
junction the molecule is small (a few Å) but the slabs span 20+ Å, so
3Dmol's auto-fit of the bounding box pivots far from the molecule and
mouse-wheel zoom drifts the molecule out of view.  Two affordances
keep interaction smooth:

* **Focus-molecule button** (viewer toolbar).  One click anchors the
  pivot on the molecule (excludes residue `ELC`) and refits the
  camera with a 0.55× pull-back so the slabs remain visible in the
  periphery.  Use whenever rotate/zoom feels off-centre.
* **Pivot snap on rotation drag.**  On every plain left-button drag
  (no ctrl/shift/alt), once the gesture commits to a drag (movement
  > 4 px from the press point), the camera lookAt snaps to the
  structure centroid.  This makes rotation always pivot on the
  structure regardless of any pan the user did beforehand.  Click-
  to-select gestures (no drag) do NOT trigger the snap, so atom
  picks stay precise.

### 2.3 Undo (slab-only)

The Junction subtab's **Undo** button rolls back the most recent
electrode-slab op.  Scope:

* Pushed only by `applyElectrode` (single + pair mode), and only on
  a successful response (failed ops do not consume an undo slot).
* Stack depth `HISTORY_MAX = 20`; pushing the 21st snapshot drops
  the oldest from the bottom.
* Other ops (delete / add / rotate / translate / centre) are
  **committed immediately** and DO NOT push history.  The canonical
  way to roll back a non-slab edit is to re-load the source XYZ.
* Each undo snapshot carries the full canonical structure response
  (xyz + per-atom metadata).  Re-applying the snapshot via
  `applyStructure` is the same code path the server response takes,
  so the UI re-renders identically.

The slab-only scope matches the original requirement ("experiment
with electrode parameters and roll back").  General undo for delete
/ rotate workflows is out of scope.

### 2.4 Geom subtab (centre + translate)

Two op-blocks, both rigid (preserve bonds, angles, residue
assignments; only coordinates change):

* **Centre at origin.**  Translates the structure so its
  *atom-coordinate mean* (geometric centroid) lands at (0, 0, 0).
  Note: this is the unweighted mean, NOT the bounding-box centre or
  the centre of mass.  For asymmetric molecules with a long
  substituent (e.g. an alkyl tail off a benzenedithiol), the
  centroid will shift toward the long substituent; if the user
  needs the **anchor-pair midpoint** at the origin, prefer the
  Pose-subtab `Orient along axis` op with `center='midpoint'`.
* **Translate by (Δx, Δy, Δz) Å.**  Three number inputs; pressing
  Apply shifts every atom by the given vector.  All-zero is
  rejected as a no-op with an explicit error message.

Both ops require a loaded structure (buttons disable on
`state.n_atoms === 0`).

### 2.1 Tilted molecules

The `--orient-axis` operation accepts an `--angle θ` parameter (degrees,
default 0) that **tilts** the anchor pair away from the target axis by
θ° in a fixed default plane (xz for `--axis z`).  The anchor-pair
midpoint stays at the origin; only the orientation changes.

When a tilted molecule then has electrodes added in **pair mode**:

* Each slab's layer planes are **always perpendicular to z** (surface
  normal along +z) and the line joining the two slab centroids IS
  the z-axis, regardless of molecule tilt.  This is intentional and
  matches real junction geometry: metal contacts are
  crystallographic, the molecule fits whatever pose works between
  them.
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
│   ├── app.py                         ◄── route /modify -> render modify.html
│   ├── blueprints/
│   │   └── modify.py                  ◄── /api/modify/* routes (M3)
│   ├── templates/
│   │   └── modify.html                ◄── new top-level page (M2),
│   │                                       mirrors templates/watch.html
│   │                                       shape; the shared app-tabs
│   │                                       nav is duplicated across
│   │                                       index.html / watch.html /
│   │                                       modify.html so each page is
│   │                                       independently routable.
│   └── static/
│       └── modify/
│           ├── style.css               ◄── three-pane layout (M2)
│           └── viewer.js               ◄── Modify tab UI logic (M2-M5)
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
| **M2** | UI skeleton: tab in shared nav, file load, atom list, click-to-select mirroring viewer ↔ list. | New `static/modify/{style.css,viewer.js}`, new `templates/modify.html` served at `/modify`, third "Modify" tab on every shared-nav block.  Reuses `/api/build/load` for the file-upload step.  No edits possible yet. | **done (2026-05-08)** |
| **M3** | Edit ops wired: delete, add-with-sliders, live distance.  `/api/modify/{load,delete,add_atom}` endpoints. | `web/blueprints/modify.py`; Delete and Add-atom fieldsets in modify.html with element input + dx/dy/dz sliders + live `|offset|` readout. | **done (2026-05-08)** |
| **M4** | Anchor-pair selection + orient-along-z + rotate-around-axis.  Per-atom info panel and an xyz-axes overlay so users can read off the geometry while they edit. | UI: Orient fieldset (axis radio, tilt slider, center mode dropdown, Apply enabled at exactly two selected atoms) + Rotate fieldset (axis radio, angle slider, Apply); Selection panel grew a per-atom info table (idx / element / name / residue / x / y / z); main-viewer toolbar gained a `Show xyz axes` checkbox that draws RGB axis arrows at the world origin.  Backend: `/api/modify/{orient,rotate}` endpoints. | **done (2026-05-09)** |
| **M5** | Electrode panel (size / gap / offset sliders, orthogonal toggle, symmetric/per-side mode); Send-to-Build handoff. | UI: Electrode fieldset (mode select, element / plane / m × n × n_layers / gap / dx / dy / orthogonal / side), Apply enabled when selection size matches mode (1 single, 2 pair); Send-to-Build button writes the structure to ``sessionStorage["builder-structure"]`` (the same key Phase 1 uses for tab navigation) and navigates to ``/`` where Build's ``restoreStructureState`` picks it up.  Backend: ``/api/modify/electrode`` + ``/api/modify/symmetric_electrodes``. | **done (2026-05-09)** |
| **M6** | Geom subtab (centre + translate); anchorless `add_symmetric_electrodes` (slabs at z=±gap/2 around origin); slab-only Undo with HISTORY_MAX=20; wireframe halo selection marker; Focus-molecule button + click-vs-drag rotation pivot snap; single-source `/api/modify/meta` for the FCC element + plane dropdowns. | UI: new Geom panel (Centre at origin button + Translate Δx/Δy/Δz row); Junction-panel Undo button; viewer toolbar Focus-molecule button.  Backend: `/api/modify/translate`, `/api/modify/meta`; `add_symmetric_electrodes(anchor_indices=None)` overload that places slabs symmetrically around the origin with a real molecule-z-extent vs. gap pre-flight; route-level `gap > 0` / `contact_distance > 0` validation. | **done (2026-05-10)** |

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
                         anchor_indices: Tuple[int, int] | None = None,
                         *, gap: float = 8.0,
                         orthogonal: bool = False,
                         offset: Tuple[float, float] = (0.0, 0.0),
                         lattice_constant: float | None = None) -> Structure
```

**Pair-electrode primitive, two modes:**

* **Anchorless (default, `anchor_indices=None`).**  Place the slab
  pair symmetrically around the world origin: top closest layer at
  `z = +gap/2`, bot closest layer at `z = -gap/2`, both lateral-
  centred on `(offset[0], offset[1])`.  No anchor selection required.
  Pre-flight rejects `gap <= 0`; rejects `gap < mol_z_extent + 3 Å`
  with an actionable "shorten or re-orient the molecule" message
  rather than producing a structure with overlapping atoms.  Empty
  structure is rejected with a pointer at the Build tab / load
  endpoint.  Workflow: centre + pose the molecule first (Geom +
  Pose subtabs), then add slabs.
* **Legacy anchored (`anchor_indices=(a_top, a_bot)`).**  Computes
  `mid = 0.5 * (positions[a_top] + positions[a_bot])` and places the
  two slabs collinear along z at `mid.z ± gap/2`, both lateral-
  centred on `(mid.x + offset[0], mid.y + offset[1])`.  For a tilted
  molecule (anchor pair off-z), the two electrodes still lie
  collinear along z; the molecule fits its tilted geometry between
  them.  Internally, each side gets the per-side contact distance
  `(gap - anchor_separation_z) / 2`; if `gap` is smaller than the
  anchor pair's z-extent the call raises `ValueError`.

`gap` is the canonical **junction gap** -- the empty z-space between
the two electrodes' closest layers.

For asymmetric junctions (different size / offset / metal per side,
or stepped contacts), call `add_electrode_slab` directly per side.

```python
# Implemented as Structure methods, exposed via /api/modify/translate.
struct.translated(vec: Sequence[float]) -> Structure
struct.centered() -> Structure                # centroid at origin
```

**Translate primitive.**  `translated((dx, dy, dz))` shifts every
atom by the given vector (Å).  `centered()` is sugar for
`translated(-positions.mean(axis=0))`: it moves the *atom-coordinate
mean* (not the bounding-box centre, not the centre of mass) to
(0, 0, 0).  Both ops are rigid; per-atom metadata is preserved.

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
| `GET /api/modify/meta` | (no body) | Returns `{ok, fcc_elements, fcc_planes}` for UI dropdowns; reads from the `SUPPORTED_FCC_ELEMENTS` / `SUPPORTED_FCC_PLANES` tuples in `molbuilder.modify`.  Single source of truth -- HTML must not duplicate the lists. |
| `POST /api/modify/load` | `{xyz, format?}` | Validate input.  Echo back canonical xyz (re-parsed; catches malformed input early). |
| `POST /api/modify/delete` | `{xyz, indices: List[int]}` | `delete_atoms` |
| `POST /api/modify/add_atom` | `{xyz, element, anchor_index, offset: [dx,dy,dz]}` | `add_atom` |
| `POST /api/modify/orient` | `{xyz, anchors: [a0,a1], axis?, center?}` | `orient_along_axis` |
| `POST /api/modify/rotate` | `{xyz, axis, angle}` | `rotate_around_axis` |
| `POST /api/modify/translate` | `{xyz, recenter?: true} OR {xyz, dx?, dy?, dz?}` | If `recenter` is truthy, `Structure.centered()`; otherwise `Structure.translated((dx, dy, dz))`.  `recenter` wins if both are sent. |
| `POST /api/modify/electrode` | `{xyz, element, plane, size:[m,n,n_layers], anchor_index, contact_distance, side, orthogonal, offset:[dx,dy], lattice_constant?, inter_layer_offset?}` | `add_electrode_slab` (single mode).  Rejects `contact_distance <= 0`. |
| `POST /api/modify/symmetric_electrodes` | `{xyz, element, plane, size:[m,n,n_layers], gap, anchors?:[a_top,a_bot], orthogonal, offset:[dx,dy], lattice_constant?}` | `add_symmetric_electrodes`.  Anchorless when `anchors` is omitted (canonical M6 flow); legacy anchor-pair-midpoint mode when `anchors` is sent.  Rejects `gap <= 0`. |

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
* **Atom indices are 0-based in the Python API and the wire JSON**
  (`anchor_index`, `indices: [...]`, etc.).  Every **user-facing**
  surface (atom-list table, selection readout, selection-info table,
  viewer overlay labels, error messages quoting an atom) is **1-based**
  -- matching PDB / SIESTA conventions and the Watch tab's overlay.
  The JS UI converts at the boundary: `tr.dataset.atomIndex` stays
  0-based for click handlers and state.selected; the displayed `#`
  column adds 1.  Build / Modify / Watch all follow this rule.

### 5.1 Cross-tab persistence (Phase 1)

Build (`/`), Watch (`/watch`), and Modify (`/modify`) are separate
Flask routes, so navigating between tabs is a full page reload --
JS closure state is destroyed.  Phase 1 (2026-05-09) added
``sessionStorage`` round-trips for the structure on Build and Modify:

* **Build** persists the last-rendered ``Structure`` (xyz, metadata,
  3Dmol camera) under ``builder-structure``.  The form-fields
  storage at ``builder-form`` is unchanged and complementary.
* **Modify** persists xyz + metadata + the current atom selection
  + 3Dmol camera + viewer toggles (`Show indices`, `Show xyz axes`,
  representation) under ``modify-state``.
* **Storage scope:** ``sessionStorage`` (per-tab, cleared on browser
  close), not ``localStorage`` -- so a "session ends -> fresh start"
  default applies.  Quota errors (>5 MB structures) are caught and
  the save silently skipped.
* **Watch** stays as it was: only the path-input value is persisted.
  Auto-reload of the trajectory on `pageshow` is Phase 2 work.
* The **Modify -> Build "Send to Build" handoff** in M5 reuses the
  same ``builder-structure`` key: Modify writes the finished
  junction there, navigates to ``/``, and Build's restore renders
  it identically to a fresh build.

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
| D8 | Web UI populates the FCC element + plane dropdowns from `/api/modify/meta` rather than hardcoding the lists in HTML. | Accepted (M6).  Realises Principle #1 (dataclass / Python-tuple as the source of truth) for the Modify tab. |
| D9 | Pair-mode electrode placement defaults to anchorless: slabs at `z = ±gap/2` around the world origin.  Legacy anchor-midpoint mode is opt-in via two-atom selection. | Accepted (M6).  Decouples slab placement from molecule centring; the user controls geometry via the Geom + Pose subtabs. |
| D10 | Undo is scoped to electrode-slab ops only; general undo for delete / rotate / translate is out of scope.  Pushed only on a successful response (failed ops do not consume an undo slot). | Accepted (M6).  Matches the original "experiment with electrodes and roll back" intent without growing the JS state model. |

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
