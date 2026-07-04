# Structure periodicity — cell / pbc / vacuum / kgrid (the periodicity contract)

**Status:** v1 proposal, 2026-07-03.

**Position.** Periodicity is part of the **structure dataset** (`<stem>.xyz` +
`<stem>.molstruct.json`), alongside geometry + regions + frozen. It is **read** by
several consumers — the molview module (cell wireframe + k-grid display), the fdf
generator (`LatticeVectors` + `kgrid_Monkhorst_Pack`), and transport — but **owned
by none of them.** This doc is the source of truth for how the cell is resolved,
gated, persisted, and edited. molview-module.md, the SIESTA emitter, and the
transport flow reference it; they do not re-derive it.

**Rule of the whole doc:** periodicity is computed/captured **once, at the source
that knows it**, stored in the dataset, and every stage reads it — never
re-derived downstream, never hand-fed as a side file.

---

## 1. The four fields

The structure dataset carries four periodicity fields (on `Structure` + persisted
in the `.molstruct.json` sidecar):

| Field | Shape | Meaning | Default |
|---|---|---|---|
| `cell` | 3×3 (rows = lattice vectors, Å) or `null` | the lattice / box vectors | derived (§ 4) |
| `pbc` | 3 bools | axis *i* is **k-sampled / tileable** (a real periodic-lattice direction) — see note | `(True,True,True)` if a cell is present, else all-`False` |
| `vacuum` | 3 floats (Å) | **isolation** padding on an *isolate* axis (molecule / slab vacuum); **0 on a transport axis** | `(0, 0, 0)` |
| `kgrid` | `[nx, ny, nz]` ints | Monkhorst–Pack **k-point** grid (reciprocal); the display *reuses* the dims to draw the Born–von Kármán supercell | `[1, 1, 1]` |

`cell` + `pbc` already exist on `Structure` (`structure.py`) and the sidecar
(`sidecars/molstruct.py::normalise_cell_pbc`, schema v3, additive). **`vacuum` +
`kgrid` are the new persisted fields.**

> **Two clarifications the physics forces:**
> - **`kgrid` is a reciprocal-space (k-point) parameter, not a real-space copy
>   count.** The display tiles by the same dims *only* because tiling the cell
>   N₁×N₂×N₃ shows the **Born–von Kármán supercell** the k-sampling makes the
>   wavefunctions periodic over — a legitimate "what does my k-grid mean in real
>   space" view, not the k-points themselves.
> - **`pbc` here gates *k-sampling + tiling*, not the emission of a lattice
>   vector.** SIESTA emits `LatticeVectors` for **all three axes** (a molecule
>   gets a box; a junction's z gets a length). `pbc[i]=False` means "Γ-only, don't
>   tile axis i," **not** "no cell vector." A junction's z is physically a periodic
>   box that is Γ-sampled and electrode-matched; we mark it `pbc=False` so k-grid
>   clamps it, but it still carries a `LatticeVectors` row.

## 2. `bbox` is min/max only — it is NOT a lattice

`bbox` (bounding box) = the per-axis extent of the atoms:
`bbox[i] = max_i(positions) − min_i(positions)`. It carries **no crystal
information**; it just wraps the atoms present.

**This is categorically different from a construction lattice**, in two ways:

- **Wrong size.** For a slab with in-plane surface spacing `d` and `m` repeats, the
  true period is `m·d`, but the atoms' bbox is `(m−1)·d`-ish (first atom to last) —
  **short by ~one spacing, non-commensurate.** Tiling it overlaps/gaps the atoms at
  the seam. (Note `d` is the *surface* spacing, **not** the cubic constant `a` in
  `fcc_lattice.json`: for fcc, `d = a/√2` — e.g. Au `a≈4.08 Å` but in-plane
  `d≈2.88 Å`. Using `a` as the in-plane period is itself a √2 error.)
- **Wrong shape (worse).** bbox is **axis-aligned → orthorhombic only** (a
  diagonal matrix). A **hexagonal** lattice (fcc(111)'s in-plane cell — vectors at
  120°, "hexagonal parallelogram" per `modify.py:815`), or any monoclinic/triclinic
  cell, has **non-orthogonal vectors an axis-aligned box cannot represent at all.**
  The `cell` field is a full 3×3 precisely to carry those angles; only
  construction (ASE gives fcc(111) its 120° cell) or import fills it — bbox never
  can.

Therefore:

- **bbox is used ONLY on non-periodic axes** (`pbc[i] = False`) — and the two kinds
  of non-periodic axis are **not** the same:
  - **Isolate axis** (a molecule; a slab's vacuum direction): cell = `bbox +
    vacuum`, with **vacuum > 0** to separate the system from its periodic images.
    This is the bbox+vacuum *derivation* fallback (§ 4).
  - **Transport axis** (a junction's z): the cell length is the **matched device
    length** (electrode–device–electrode), **vacuum = 0** — the electrodes *are*
    the boundaries and TranSIESTA matches the scattering region to the semi-infinite
    electrodes at the cell edge; padding it inserts a gap and breaks the matching.
    A transport axis' cell therefore comes from **construction** (captured, § 4),
    **never** from the bbox+vacuum fallback.
- **Periodic axes (`pbc[i] = True`) NEVER use bbox.** Their cell vector is the
  commensurate lattice, which comes from **construction** (the builder's lattice
  constant × repeats) or **import** (`.XV` / `.fdf` / CIF). No detection from raw
  coordinates (ill-posed — see the decisions log).

## 3. `resolve_cell` — the one resolver, precedence-ordered

```
resolve_cell(structure, vacuum, pbc) -> 3x3 | None
  1. EXPLICIT cell present  -> use it verbatim.
        (user-edited 3x3 override, OR imported .XV/.fdf/CIF, OR captured
         from a builder — all land in structure.cell)
  2. else, per axis i:
        pbc[i] == True   -> the commensurate lattice vector for axis i
                            (from construction/import; error if unknown —
                             we do NOT bbox a periodic axis)
        pbc[i] == False  -> bbox[i] + vacuum[i]   (an ISOLATE box; vacuum>0)
                            NB: a TRANSPORT axis is not derived here -- its
                            length is captured at construction (branch 1),
                            vacuum=0 (§ 2).
```

An **explicit cell always wins** — that is the customization escape hatch (§ 6),
and the path a *transport* axis always takes (its length is constructed, not
derived). Absent an explicit cell, each axis is filled by its `pbc` type; a
fully non-periodic structure (`pbc` all-False) gets an all-orthorhombic
`bbox+vacuum` box for a Γ-point DFT calc.

> **Scope of the per-axis form.** Branch 2 assumes the cell is **block-orthogonal**
> — a periodic sub-block (e.g. a hexagonal in-plane pair) orthogonal to the
> non-periodic axis (z). That covers slabs and junctions. A fully general triclinic
> cell mixed with a non-periodic direction is not separable per-axis; such a cell
> must arrive **explicit** (branch 1), not derived.

## 4. Capture-at-construction (fix the electrode discard)

`modify.py::add_electrode_slab` builds the slab with ASE's `fcc{100,110,111}`
using the lattice constant from `data/fcc_lattice.json` — so ASE **knows the
correct in-plane cell** (comment at `modify.py:923`). But the assembly
(`return Structure(...)`, `modify.py:955`) **keeps only the atom positions and
drops the cell.** That discard is why the transport flow has to be hand-fed a
separate `cell_fdf` (`transport/_cli.py::_load_device(..., cell_fdf)`).

**The builder must capture what it built:**

- `add_electrode_slab` / `add_symmetric_electrodes` set `Structure.cell` from the
  ASE slab's in-plane lattice + the z-extent, and set **`pbc = (True, True,
  False)`** (in-plane periodic; z = transport, non-periodic).
- `fcc_lattice.json` carries per-functional constants (`a_experimental`, `a_pbe`,
  `a_pbe_siesta_psml`); the captured cell records the constant actually used, so
  the lattice matches the DFT setup.

Then transport, molview, and the fdf all read `Structure.cell` from the dataset —
no `cell_fdf`.

## 5. k-grid is gated by `pbc` (this is what prevents overlaps)

A junction's z is **not a lattice period** — its length (electrode + gap + molecule
+ gap + electrode) is not commensurate with the electrode spacing. Tiling it would
overlap/gap the atoms at the seam. So:

- **`kgrid` dims are clamped to 1 on every non-periodic axis:** `dims[i] = 1`
  wherever `pbc[i] = False`. A junction is only ever tiled in-plane; its z is never
  tiled. (By convention `kz=1` on a transport/isolate axis — SIESTA does not
  enforce it, this periodicity model does.)
- k-grid > 1 is valid **only** where `pbc[i] = True` **and** the cell vector is the
  true commensurate lattice (§ 2). Then tiling reproduces the crystal exactly:
  a boundary atom's copy lands on the next cell's equivalent atom — **provided atoms
  are stored half-open `[0, L)`** (no atom on both faces, or the boundary doubles).

This clamp is enforced both in the k-grid UI (non-periodic dims disabled) and in the
render/`tileKgrid` path, so neither Modify nor Results can produce overlapping copies.

## 6. The Modify periodicity panel — two views of one thing

Modify exposes periodicity (new; none today). Two coupled views of the same
`(cell, pbc, vacuum, kgrid)`:

- **Convenient (default path):** `pbc` per-axis toggles · `vacuum[x,y,z]` (default
  0) · `kgrid[nx,ny,nz]` (default 1×1×1, non-periodic dims greyed). Set vacuum, the
  cell derives (§ 3).
- **Raw / override (customization):** the **3×3 lattice matrix, shown live and
  editable.** Editing it **pins** the cell as explicit (`structure.cell` set
  directly), so it wins over derivation and later atom edits don't silently
  recompute it. A **"reset to derived"** clears the override → back to bbox+vacuum.

## 7. Persistence + the data-flow loop

`(cell, pbc, vacuum, kgrid)` persist in the `.molstruct.json` sidecar (schema bump;
additive). The `.xyz`+`.json` set is the source of truth, and periodicity flows
one way, read at each stage:

```
.xyz + .json  (cell/pbc/vacuum/kgrid)
   │  ├─ molview: cell wireframe (opts.lattice) + pbc-gated k-grid tiling
   │  ├─ fdf generator: LatticeVectors (from cell) + kgrid_Monkhorst_Pack (from kgrid)
   │  └─ transport: reads Structure.cell (no cell_fdf)
   ▼
 .fdf (carries LatticeVectors + kgrid + geometry)
   ▼
  run → SIESTA output (.out/.XV: cell + kgrid)
   ▼
 parse/ -> StructureResult.cell / JobResult kgrid  →  back into a dataset
```

## 8. Downstream read points (per file type, host-side via `parse/`)

The molview module never parses; the host supplies `cell` + `kgrid`:

| Selected file | cell + kgrid from |
|---|---|
| `.xyz` | its `.molstruct.json` sidecar |
| SIESTA output (`.out`/`.XV`) | `parse/` → `StructureResult.cell` / `JobResult` kgrid |
| PySCF output | `parse/` (molecular → usually no cell) |
| `.fdf` | `parse/` (LatticeVectors + `kgrid_Monkhorst_Pack`) |

> `kgrid` is the **diagonal** `(kx,ky,kz)` of the Monkhorst–Pack block — what
> molbuilder writes. A general *non-diagonal* MP grid (off-diagonal 3×3) is not
> representable as a single `[nx,ny,nz]`; an imported fdf with one would need the
> full block preserved (out of scope for the dims used by display + the generator).

## 9. Current state vs. to-build

| Piece | Now | To build |
|---|---|---|
| `Structure.cell` + `pbc` | ✓ | — |
| `Structure.vacuum` + `kgrid` | ✗ | add fields |
| sidecar `cell` + `pbc` | ✓ (`normalise_cell_pbc`) | — |
| sidecar `vacuum` + `kgrid` | ✗ | add (schema bump) |
| electrode builder captures cell + sets pbc | ✗ (discards, `modify.py:955`) | capture |
| `resolve_cell` (explicit / lattice / bbox+vacuum, pbc-typed) | ✗ | add |
| k-grid pbc-gating (clamp dims=1 on non-periodic axes) | ✗ | add (store + UI + render) |
| Modify periodicity panel (pbc + vacuum + kgrid + 3×3 override) | ✗ | add |
| molview reads cell+kgrid from dataset | ✗ (`ctx.viewParams` unwired) | wire |
| fdf reads kgrid from dataset | ✗ (from CLI `Config`) | switch source |
| transport reads `Structure.cell` | ✗ (separate `cell_fdf`) | switch source |

## 10. Decisions log

| Date | Decision |
|---|---|
| 2026-07-04 | **Scientific-review corrections.** (1) `kgrid` is a *reciprocal-space* Monkhorst–Pack k-point grid; the display *reuses* the dims to draw the Born–von Kármán supercell — not a raw real-space copy count. (2) `pbc` gates k-sampling/tiling **only**; `LatticeVectors` are emitted for every axis (a junction's z has a cell vector, Γ-sampled). (3) A **transport** axis (junction z) is the electrode-matched **device length with vacuum = 0**, captured at construction — NOT the `bbox+vacuum` *isolate* fallback (which is vacuum > 0, for molecules / slab vacuum). (4) fcc in-plane surface spacing is `a/√2`, not the cubic `a`. Per-axis derivation assumes block-orthogonality (general triclinic must arrive explicit); `kgrid` captures the MP **diagonal** only. |
| 2026-07-03 | Periodicity is `(cell, pbc, vacuum, kgrid)` on the `.xyz`+`.json` dataset. `resolve_cell` precedence: explicit cell wins, else per-axis (periodic → commensurate lattice from construction/import; non-periodic → bbox+vacuum). **bbox = min/max only, used ONLY on non-periodic axes — never for a periodic axis** (non-commensurate → overlaps). No lattice detection from raw coordinates (ill-posed; crystals come from builders that know the lattice). k-grid clamped to 1 on non-periodic axes (junction z never tiled → no seam overlaps). Electrode builder captures its ASE cell + sets `pbc=(T,T,F)` (fixes the discard → transport drops `cell_fdf`). Modify exposes the 3×3 as an editable override on top of the vacuum/kgrid convenience. |
