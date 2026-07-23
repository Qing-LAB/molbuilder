# Structure periodicity — cell / axis_kind / vacuum / kgrid (the periodicity contract)

**Status:** v2, 2026-07-04. (v1 2026-07-03 used a boolean `pbc`; v2 replaces it with
the `axis_kind` enum — a boolean can't distinguish an *isolated* axis from a
*transport* one, which need different cells, vacuum, and k-treatment.)

**Position.** Periodicity is part of the **structure dataset** (`<stem>.xyz` +
`<stem>.molstruct.json`), alongside geometry + regions + frozen. It is **read** by
several consumers — the molview module (cell wireframe + k-grid display), the fdf
generator (`LatticeVectors` + `kgrid_Monkhorst_Pack`), and transport — but **owned
by none of them.** This doc is the source of truth for how the cell is resolved,
gated, persisted, and edited. molview-module.md (the MolView module),
the SIESTA emitter, and the transport flow reference it; they do not re-derive it.

**Rule of the whole doc:** periodicity is computed/captured **once, at the source
that knows it**, stored in the dataset, and every stage reads it — never
re-derived downstream, never hand-fed as a side file.

---

## 1. The fields

| Field | Shape | Meaning | Default |
|---|---|---|---|
| `cell` | 3×3 (rows = lattice vectors, Å) or `null` | the lattice / box vectors | derived (§ 3) |
| `cell_origin` | 3 floats (Å) or `null` | world-space **low corner** an explicit `cell` emanates from; lets a cell WRAP off-origin atoms without moving them (§ 3c) | `null` = `(0,0,0)` |
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

## 3a. The "default" state — parameters resolve through the default API, never absent

Every periodicity parameter has an explicit **default** initial state (a fresh /
generated structure starts entirely in it).  A consumer must **translate the default
through the canonical resolver**, NOT read the raw stored field and treat a missing
value as "no box / no periodicity."  This is the contract that makes the k-grid
display, the box render, and the fdf work on a blank molecule.

| Parameter | Default initial state | Resolver (translates default → concrete) | Explicit override (resets the default) |
|---|---|---|---|
| **cell** (a, b, c) | `struct.cell is None` | `resolve_cell()` — per axis: `isolated → bbox[i] + vacuum[i]`, `transport → bbox[i]`, `periodic → commensurate lattice` (§ 3) | `ws.setUnitCell(3×3)` / import / capture → `struct.cell` wins verbatim |
| **vacuum** | `(0, 0, 0)` | literal `0` per axis (feeds `resolve_cell`) | `ws.setVacuum([x,y,z])` — grows each isolated axis's box |
| **axis_kind** (pbc) | `isolated` on every axis (a fresh molecule is a vacuum box) | `pbc[i] = axis_kind[i] != "isolated"` (§ 1) | `ws.setAxisKind([...])` — mark an axis `periodic` / `transport` |
| **kgrid** | `(1, 1, 1)` (Γ) | literal Γ | `ws.setKgrid([...])` |

**The load-bearing rule:** the cell the renderer / k-grid use is the **resolved** cell,
obtained ONLY through the unified accessor **`ws.getUnitCellInfo().value`** — never a
hand-read of `getStructure().periodicity.cell` (the encapsulation contract: consumers
go through `ws.*`, not the raw in-memory fields).  `periodicity.cell` stays the raw
explicit cell (`null` = default); the accessor surfaces `periodicity.resolved_cell` —
the effective bbox+vacuum box for a cell-less molecule — so the k-grid tiles it
immediately.  A consumer that hand-reads `periodicity.cell` and short-circuits on
`cell == null` is the exact bug this section prevents ("k-grid has no effect on a new
molecule").

**One resolver, no duplication:** the resolved cell is computed in exactly ONE place —
`struct.resolve_cell()` on the **server** (§ 3), the same function the fdf/save use.  The
server sends the result as `periodicity.resolved_cell`; the client accessor only
surfaces it (no re-implemented bbox math on the client).  A periodicity edit
(`ws.setUnitCell` / `ws.setVacuum` / `ws.setAxisKind`, via the Cell page's Update)
re-resolves **through the server**, so `resolved_cell` stays consistent with
cell/vacuum/axis_kind for one data structure.  `resolved_cell` is DERIVED — never saved
(the save writes the raw `cell`) and never committed to `struct.cell` (which would
masquerade as a user-chosen lattice and defeat the override hatch, § 6).  The moment the
user calls `ws.setUnitCell` / `ws.setVacuum` / `ws.setAxisKind`, that explicit value
replaces the
default and persists (capture-at-construction, § 4).

## 3b. Where periodicity is displayed vs edited (the UI contract)

The periodicity parameters surface in two kinds of place with different jobs:

**Display (read-only, mirrors in-memory) — the MolView "Cell" page.** The molview panel
has two switchable pages, `[ Selection | Cell ]` as header tabs, sharing the panel.  The
**Cell page is DISPLAY-ONLY**: it shows vacuum (x/y/z), the unit cell as a **3×3 matrix**
(non-orthogonal-ready), the **cell origin** (the low corner the box is drawn from, §3c —
`getUnitCellOriginInfo().value` = `resolved_cell_origin`), axis_kind/pbc per axis, and the
k-grid (a display toggle + the dims).  Each field is read through the molview read API and
shows **"(default)" + the resolved value** when unset (vacuum `0`, cell = bbox a/b/c,
origin = the auto molecule/world corner, kgrid `1×1×1`), or the explicit value once set.  The read accessor returns `{ value, isDefault }` so the page
can render the "(default)" tag while still handing out a usable number.  The MolView
provides **no Update button** — it never writes; it only mirrors the in-memory data.

**Edit (write, via the set-API + per-group Update) — the Modify functions.** Editing
lives in the Modify functions, NOT in the MolView.  Each parameter GROUP — vacuum, pbc
(axis_kind), unit cell, **cell origin**, k-grid — has its **own explicit "Update" button**,
so each group can independently stay at its default or be committed.  Editing a group
STAGES a change; it does NOT touch the in-memory structure until that group's Update is
pressed, which commits via `ws.commitPeriodicity` (vacuum / pbc / cell / **cell_origin** —
re-resolving the effective cell + drawn corner through the ONE server resolver, § 3a/3c) or
`ws.setKgrid` (k-grid — no re-resolve needed).  **Cell origin** (`ws.setCellOrigin` via
`commitPeriodicity({cell_origin})`) is enabled **only with an explicit cell** — `cell_origin`
is the offset an explicit cell emanates from and the dataclass drops it otherwise (§3c);
with a bbox+vacuum cell the corner is auto (`bbox_min − vacuum`) and shown read-only.
Editing the origin moves the **box**, not the atoms; **"Calibrate coordinates to cell"**
bakes the shift into the coordinates (moves atoms into `[0,cell)`, clears the origin).
Until a group's Update, that group keeps its computed default and no set-API is called
for it.  The 3×3 cell is populated either by adding metal atoms (as a lattice, § 4) or
edited manually.  The fdf / structure-optimization tabs expose the same groups against
the same set-API.

**The k-grid is ONE value, not two.** The tiling shown by the Cell page's k-grid display
uses the structure's `periodicity.kgrid` — the SAME value the DFT k-point sampling uses.
Default `1×1×1` (Γ) → the tiling view has no visible effect.  It becomes non-trivial only
when set via the API (the Modify k-grid Update, or the DFT-calculation setup), and that
one value is both displayed and used by the render/tiling step.  There is no separate
view-only tiling count.

**Implemented** (2026-07-08): the render controller `mountKgridRender`
(`lib/molview/render-pipeline.js`) takes the tiling DIMS from `opts.getKgridDims()` =
`ws.getKgrid()` = `periodicity.kgrid`; the ENABLE toggle stays a view state on the
selection store.  The MolView Cell page's `[nx,ny,nz]` are a **read-only mirror** of
`periodicity.kgrid`; the value is set in the Modify Cell op-tab (`ws.setKgrid`).  The
viewer re-tiles on a periodicity change (viewer.js subscribes `ws.subscribe →
_kgCtl.refresh()`).

## 3c. Cell origin + calibration — an explicit cell that WRAPS off-origin atoms

**The problem this solves.** Building a tunnelling junction, the natural workflow keeps
the molecule (or the selected component) **pinned at the world origin** and grows
structure around it: centre the molecule at `(0,0,0)`, orient its anchors along `z`,
then `add_symmetric_electrodes` flanks it with slabs at `z = ±gap/2`. The electrode op
then captures an explicit `cell` (§ 4) whose lateral vectors are the slab lattice and
whose `z` length is the total device extent. But the atoms now **straddle the origin**
(`z ∈ [−L/2, +L/2]`), while a bare 3×3 `cell` is, by SIESTA convention, anchored at
`(0,0,0)`. So the cell would sit at the origin with **half the atoms outside it** — the
box "jumps" off the structure, and the emitted FDF would place device atoms outside the
transport cell. (This was the 2026-07 bug: cell right-size, wrong corner.)

**The contract — separate editing convenience from SIESTA correctness.**

1. **`cell_origin` (a new field): the world-space LOW CORNER an explicit cell emanates
   from** (`null` = `(0,0,0)`). An op that builds a cell *around* off-origin atoms sets
   `cell_origin` to the structure's low corner, so the cell **wraps the atoms without
   moving them** — the molecule stays pinned where the user put it. It is *stored
   intent* (set by the op), NOT guessed from atom extents, so it never drifts as the
   user edits, and a genuine imported crystal (atoms already in `[0,cell)`) leaves it
   `null`.

2. **`resolve_cell_origin()` returns `cell_origin` for an explicit cell** (was `null`).
   So the **viewer draws the box at its true corner, wrapping the structure** — no jump
   to the origin.

3. **SIESTA correctness is applied at generation, not during editing.**
   `render_fdf` translates atoms by `−resolve_cell_origin()` for **every** cell (it
   already did for derived cells), so SIESTA always receives the atoms inside `[0,cell)`
   with the cell at `(0,0,0)`. **This is the viewer ≡ render_fdf invariant:** the
   viewer's box (cell at `cell_origin`, atoms where they are) and SIESTA's cell (at
   `(0,0,0)`, atoms translated by `−cell_origin`) are the SAME relative geometry.

4. **`calibrate` — the unified last step** (a Modify op, "Calibrate coordinates to
   cell"). It *bakes* the generation-time shift into the stored coordinates: translate
   all atoms by `−resolve_cell_origin()`, then set `cell_origin → (0,0,0)`. After
   calibration the stored structure, the saved `.xyz`, the viewer box, and the FDF all
   agree, atoms in `[0,cell)`. Calibration is OPTIONAL — generation is correct with or
   without it — but it lets the user *see* and *save* the exact SIESTA coordinate frame.

```mermaid
flowchart LR
    subgraph EDIT["EDIT — molecule pinned at origin (convenience)"]
        M["molecule @ origin"] --> E["add electrodes<br/>atoms straddle origin<br/>cell captured + cell_origin = bbox low corner"]
    end
    E -->|viewer| V["box drawn at cell_origin<br/>WRAPS the structure (no jump)"]
    E -->|render_fdf<br/>(always)| S["atoms translated by −cell_origin<br/>cell @ (0,0,0), atoms in [0,cell)  ✓ SIESTA"]
    E -->|calibrate (optional last step)| C["bake the shift into stored coords<br/>cell_origin → 0; atoms in [0,cell)"]
    C --> V2["viewer box @ origin == FDF cell — all frames agree"]
```

**The `resolve_cell` / `resolve_cell_origin` table, completed:**

| Cell state | `resolve_cell()` | `resolve_cell_origin()` | `render_fdf` translates atoms by |
|---|---|---|---|
| derived (no explicit cell) | per-axis bbox+vacuum / bbox (§ 3) | `bbox_min − vacuum` (isolated) / `bbox_min` (transport) | `−origin` (centres in the box) |
| explicit, `cell_origin` set (electrode junction) | the explicit cell | `cell_origin` | `−cell_origin` (into `[0,cell)`) |
| explicit, `cell_origin` null (imported crystal) | the explicit cell | `null` → `(0,0,0)` | `0` (already in `[0,cell)`) |

**"Use default" is invalid for a `periodic`/`transport` axis.** Clearing the explicit
cell falls back to `resolve_cell()`, which **raises** on a `periodic` axis (you cannot
derive a commensurate lattice from a bounding box, § 3). So the Cell tab's **"Use
default" is disabled whenever any axis is `periodic` or `transport`** — offering it is
what made the box "disappear" (the resolver errored, `resolved_cell` never updated).
Likewise **vacuum is N/A for an explicit cell** (`resolve_cell` returns the cell
verbatim; vacuum only grows a *derived* isolated axis), so the vacuum control reads
"not applicable" rather than silently no-op'ing.

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

**`axis_kind` round-trips.** It is **initialized from construction** (§ 4 — an
electrode junction arrives as `(periodic, periodic, transport)`) or the resolve
default; the panel **shows** it, the user **overrides** it, and it **persists in
the sidecar** (§ 7) — a customized value survives reload, and every consumer (cell
derivation, k-grid gating, fdf, transport) reads the one field. The boolean `pbc`
is never edited directly; it is derived from `axis_kind`.

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
| `Structure.axis_kind` (+ derived `pbc`) | ✓ (enum field; `pbc` derived) | — |
| `Structure.vacuum` + `kgrid` | ✓ (fields present) | — |
| sidecar `cell` + `pbc` | ✓ (`normalise_cell_pbc`) | — |
| sidecar `axis_kind` + `vacuum` + `kgrid` | ✓ (`molstruct.to_dict`) | — |
| electrode builder captures cell + sets `axis_kind` | ✗ (discards, `modify.py:955`) | capture |
| `resolve_cell` (explicit / per-`axis_kind`) | ✓ (`structure.py`; server, surfaced as `resolved_cell`) | — |
| k-grid clamp (dims=1 unless `periodic`) | ✗ | add (store + UI + render) |
| Modify periodicity panel (axis_kind + vacuum + kgrid + 3×3 override) | ✓ (Modify "Cell" op-tab, `modify/periodicity.js`; per-group Update via `ws.commitPeriodicity`/`setKgrid`) | — |
| molview reads cell+kgrid from dataset | ✗ (`ctx.viewParams` unwired) | wire |
| fdf reads kgrid from dataset | ✗ (from CLI `Config`) | switch source |
| transport reads `Structure.cell` | ✗ (separate `cell_fdf`) | switch source |

## 10. Decisions log

| Date | Decision |
|---|---|
| 2026-07-04 | **`axis_kind` enum replaces boolean `pbc`.** A boolean overloaded "not periodic" to mean two physically different axes; the enum makes `{periodic, isolated, transport}` first-class. Treatment is per-kind (§ 1.1): cell source (lattice / bbox+vacuum / captured length), `vacuum` (only `isolated`, >0), `kgrid`>1 + tiling (only `periodic`). `pbc` becomes a **derived** ASE view (`periodic\|transport → True`, `isolated → False`) — which also fixes the boolean ambiguity that a `transport` axis is a periodic box (ASE `pbc=True`, has `LatticeVectors`) yet Γ-only and not tileable. |
| 2026-07-04 | **Scientific-review corrections** (folded into v2). `kgrid` = reciprocal MP grid; display reuses dims for the Born–von Kármán supercell. `LatticeVectors` emitted on every axis. Transport axis = electrode-matched device length, vacuum=0, captured (not bbox+vacuum). fcc in-plane spacing is `a/√2` not the cubic `a`. Per-axis derivation assumes block-orthogonality (general triclinic must arrive explicit); `kgrid` captures the MP diagonal only. |
| 2026-07-03 | Initial contract: periodicity `(cell, pbc, vacuum, kgrid)` on the `.xyz`+`.json` dataset; `resolve_cell` precedence (explicit → lattice → bbox+vacuum); bbox = min/max only, never a periodic axis; capture-at-construction (electrode discard fix); the 3×3 override. |
