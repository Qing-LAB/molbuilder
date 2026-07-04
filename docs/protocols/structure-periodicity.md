# Structure periodicity — cell / axis_kind / vacuum / kgrid (the periodicity contract)

**Status:** v2, 2026-07-04. (v1 2026-07-03 used a boolean `pbc`; v2 replaces it with
the `axis_kind` enum — a boolean can't distinguish an *isolated* axis from a
*transport* one, which need different cells, vacuum, and k-treatment.)

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

## 1. The fields

| Field | Shape | Meaning | Default |
|---|---|---|---|
| `cell` | 3×3 (rows = lattice vectors, Å) or `null` | the lattice / box vectors | derived (§ 3) |
| **`axis_kind`** | 3 × enum `{periodic, isolated, transport}` | **how axis *i* is treated — the authoritative periodicity field** (§ 1.1) | `(periodic,periodic,periodic)` if a cell is present, else all-`isolated` |
| `pbc` | 3 bools — **DERIVED, not stored** | ASE-interop view: `periodic\|transport → True`, `isolated → False` | from `axis_kind` |
| `vacuum` | 3 floats (Å) | isolation padding — **nonzero only on an `isolated` axis** | `(0, 0, 0)` |
| `kgrid` | `[nx, ny, nz]` ints | Monkhorst–Pack **k-point** grid (reciprocal); **>1 only on a `periodic` axis**; the display *reuses* the dims to draw the Born–von Kármán supercell | `[1, 1, 1]` |

`cell` already exists on `Structure` (`structure.py`) + the sidecar
(`normalise_cell_pbc`, schema v3). **`axis_kind`, `vacuum`, `kgrid` are new.** The
boolean `pbc` on `Structure`/ASE stays as a **derived property** of `axis_kind`
(so `normalise_cell_pbc` / ASE interop are unchanged).

### 1.1 The three axis kinds — the whole model in one table

| kind | cell vector on axis *i* | `vacuum[i]` | `kgrid[i]` / k-sampling | tileable (display) | derived ASE `pbc[i]` | fdf |
|---|---|---|---|---|---|---|
| **periodic** | commensurate lattice (construction / import) | 0 | `≥ 1` (sampled) | **yes** | `True` | k-sampled |
| **isolated** | `bbox[i] + vacuum[i]` | **> 0** | `1` (Γ) | no | `False` | Γ box |
| **transport** (semi-infinite) | matched **device length** (captured at construction) | **0** | `1` (Γ) | no | `True` | Γ + electrode self-energy |

Every consumer branches on this one field: `resolve_cell` (§ 3), the k-grid clamp
(§ 5), the vacuum rule, and the fdf treatment.

> **Two physics points the enum encodes (that a boolean could not):**
> - **`kgrid` is a reciprocal-space k-point parameter, not a real-space copy
>   count.** The display tiles by the same dims *only* because tiling the cell
>   N₁×N₂×N₃ shows the **Born–von Kármán supercell** the k-sampling makes the
>   wavefunctions periodic over — a "what does my k-grid mean in real space" view.
> - **A `transport` axis is a *periodic box that is Γ-sampled***. SIESTA emits a
>   `LatticeVectors` row for it (and for a molecule's box), so its ASE `pbc` is
>   `True` — yet it is never tiled or k-sampled (the semi-infinite leads replace
>   its periodic images). A boolean `pbc` cannot hold "periodic box **but** Γ-only,
>   electrode-matched"; `axis_kind = transport` says it exactly. `isolated` derives
>   `pbc = False`; only `periodic` is tileable.

## 2. `bbox` is min/max only — used ONLY on an `isolated` axis

`bbox[i] = max_i(positions) − min_i(positions)` — the extent of the atoms. It
carries **no crystal information**, and it is **categorically not a lattice**, in
two ways:

- **Wrong size.** For a slab with in-plane surface spacing `d` and `m` repeats, the
  true period is `m·d`, but the atoms' bbox is `(m−1)·d`-ish — **short by ~one
  spacing, non-commensurate.** Tiling it overlaps/gaps atoms at the seam. (`d` is
  the *surface* spacing, **not** the cubic constant `a` in `fcc_lattice.json`: for
  fcc, `d = a/√2` — Au `a≈4.08 Å` but in-plane `d≈2.88 Å`. Using `a` is a √2 error.)
- **Wrong shape (worse).** bbox is **axis-aligned → orthorhombic only**. A
  **hexagonal** lattice (fcc(111)'s 120° in-plane cell, `modify.py:815`) or any
  monoclinic/triclinic cell has **non-orthogonal vectors an axis-aligned box cannot
  represent at all.** The full 3×3 `cell` carries those angles; only construction
  (ASE gives fcc(111) its 120° cell) or import fills it — bbox never can.

Therefore **bbox+vacuum is the derivation for `isolated` axes only.** `periodic`
axes use the commensurate lattice (construction/import — no detection from raw
coordinates, which is ill-posed). `transport` axes use the captured device length
(§ 4), never bbox — padding a transport axis breaks the electrode matching.

## 3. `resolve_cell` — branch on `axis_kind`

```
resolve_cell(structure) -> 3x3 | None
  1. EXPLICIT cell present -> use it verbatim
        (user-edited 3x3 override, imported .XV/.fdf/CIF, or captured
         from a builder -- all land in structure.cell)
  2. else, per axis i by axis_kind[i]:
        periodic  -> commensurate lattice vector (construction/import;
                     error if unknown -- we do NOT bbox a periodic axis)
        isolated  -> bbox[i] + vacuum[i]        (vacuum > 0)
        transport -> the captured device length (in practice branch 1;
                     never derived here, vacuum = 0)
```

An **explicit cell always wins** — the customization escape hatch (§ 6), and the
path a `transport` axis always takes.

> **Scope of the per-axis form.** Branch 2 assumes the cell is **block-orthogonal**
> — a periodic sub-block (e.g. a hexagonal in-plane pair) orthogonal to the
> non-periodic axis. That covers slabs and junctions. A fully general triclinic cell
> mixed with a non-periodic direction is not separable per-axis; it must arrive
> **explicit** (branch 1).

## 4. Capture-at-construction (fix the electrode discard)

`modify.py::add_electrode_slab` builds the slab with ASE's `fcc{100,110,111}` from
the lattice constant in `data/fcc_lattice.json` — so ASE **knows the correct
in-plane cell** (comment `modify.py:923`) — but the assembly (`return
Structure(...)`, `modify.py:955`) **keeps only atom positions and drops the cell.**
That discard is why transport is hand-fed a separate `cell_fdf`
(`transport/_cli.py::_load_device(..., cell_fdf)`).

**The builder must capture what it built:**

- `add_electrode_slab` / `add_symmetric_electrodes` set `Structure.cell` (in-plane
  lattice + captured z device length) and **`axis_kind = (periodic, periodic,
  transport)`**.
- `fcc_lattice.json` carries per-functional constants (`a_experimental`, `a_pbe`,
  `a_pbe_siesta_psml`); the captured cell records the one used, so the lattice
  matches the DFT setup.

Then transport, molview, and the fdf read `Structure.cell` + `axis_kind` from the
dataset — no `cell_fdf`.

## 5. k-grid is gated by `axis_kind` (this prevents overlaps)

- **`kgrid` dims are clamped to 1 unless `axis_kind[i] == periodic`.** A junction's
  z (`transport`) and a molecule's/slab's box (`isolated`) are never tiled. (`kz=1`
  on those axes is the convention; this model enforces it — SIESTA does not.)
- k-grid > 1 is valid only where the axis is `periodic` **and** the cell vector is
  the true commensurate lattice (§ 2). Then tiling reproduces the crystal exactly:
  a boundary atom's copy lands on the next cell's equivalent atom — **provided
  atoms are stored half-open `[0, L)`** (an atom on both faces doubles at the seam).

Enforced in the k-grid UI (non-`periodic` dims disabled) and the render/`tileKgrid`
path, so neither Modify nor Results can produce overlapping copies.

## 6. The Modify periodicity panel — two views of one thing

Modify exposes periodicity (new; none today). Two coupled views of the same
`(cell, axis_kind, vacuum, kgrid)`:

- **Convenient (default path):** per-axis **`axis_kind` selector**
  (periodic / isolated / transport) · `vacuum[x,y,z]` (enabled only on `isolated`
  axes, default 0) · `kgrid[nx,ny,nz]` (default 1×1×1; non-`periodic` dims greyed).
  Set the kinds + vacuum, the cell derives (§ 3).
- **Raw / override (customization):** the **3×3 lattice matrix, shown live and
  editable.** Editing **pins** the cell explicit (`structure.cell` set directly),
  winning over derivation so later atom edits don't silently recompute it. A
  **"reset to derived"** clears the override.

## 7. Persistence + the data-flow loop

`(cell, axis_kind, vacuum, kgrid)` persist in the `.molstruct.json` sidecar (schema
bump; additive; `pbc` stays derived). Periodicity flows one way, read at each stage:

```
.xyz + .json  (cell / axis_kind / vacuum / kgrid)
   │  ├─ molview: cell wireframe (opts.lattice) + axis_kind-gated k-grid tiling
   │  ├─ fdf generator: LatticeVectors (from cell) + kgrid_Monkhorst_Pack (from kgrid)
   │  └─ transport: reads Structure.cell + axis_kind (no cell_fdf)
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
> molbuilder writes. A general *non-diagonal* MP grid is not representable as a
> single `[nx,ny,nz]`; an imported fdf with one needs the full block preserved
> (out of scope for the dims used by display + the generator).

## 9. Current state vs. to-build

| Piece | Now | To build |
|---|---|---|
| `Structure.cell` | ✓ | — |
| `Structure.axis_kind` (+ derived `pbc`) | ✗ (only boolean `pbc`) | add enum; make `pbc` derived |
| `Structure.vacuum` + `kgrid` | ✗ | add fields |
| sidecar `cell` + `pbc` | ✓ (`normalise_cell_pbc`) | — |
| sidecar `axis_kind` + `vacuum` + `kgrid` | ✗ | add (schema bump) |
| electrode builder captures cell + sets `axis_kind` | ✗ (discards, `modify.py:955`) | capture |
| `resolve_cell` (explicit / per-`axis_kind`) | ✗ | add |
| k-grid clamp (dims=1 unless `periodic`) | ✗ | add (store + UI + render) |
| Modify periodicity panel (axis_kind + vacuum + kgrid + 3×3 override) | ✗ | add |
| molview reads cell+kgrid from dataset | ✗ (`ctx.viewParams` unwired) | wire |
| fdf reads kgrid from dataset | ✗ (from CLI `Config`) | switch source |
| transport reads `Structure.cell` | ✗ (separate `cell_fdf`) | switch source |

## 10. Decisions log

| Date | Decision |
|---|---|
| 2026-07-04 | **`axis_kind` enum replaces boolean `pbc`.** A boolean overloaded "not periodic" to mean two physically different axes; the enum makes `{periodic, isolated, transport}` first-class. Treatment is per-kind (§ 1.1): cell source (lattice / bbox+vacuum / captured length), `vacuum` (only `isolated`, >0), `kgrid`>1 + tiling (only `periodic`). `pbc` becomes a **derived** ASE view (`periodic\|transport → True`, `isolated → False`) — which also fixes the boolean ambiguity that a `transport` axis is a periodic box (ASE `pbc=True`, has `LatticeVectors`) yet Γ-only and not tileable. |
| 2026-07-04 | **Scientific-review corrections** (folded into v2). `kgrid` = reciprocal MP grid; display reuses dims for the Born–von Kármán supercell. `LatticeVectors` emitted on every axis. Transport axis = electrode-matched device length, vacuum=0, captured (not bbox+vacuum). fcc in-plane spacing is `a/√2` not the cubic `a`. Per-axis derivation assumes block-orthogonality (general triclinic must arrive explicit); `kgrid` captures the MP diagonal only. |
| 2026-07-03 | Initial contract: periodicity `(cell, pbc, vacuum, kgrid)` on the `.xyz`+`.json` dataset; `resolve_cell` precedence (explicit → lattice → bbox+vacuum); bbox = min/max only, never a periodic axis; capture-at-construction (electrode discard fix); the 3×3 override. |
