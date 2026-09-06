# Structure — the core object, its codec, and its file doors

**Role:** contract
**Domain:** model
**This is the master doc for the Structure aspect.** Its large facets live as
sub-documents sharing the `structure-` filename prefix (so the hierarchy is
visible in the name itself):
- [`structure-periodicity.md`](?doc=model/structure-periodicity.md) — cell · cell_origin ·
  axis_kind · derived pbc · vacuum (the per-axis box behaviour).
- [`structure-annotations.md`](?doc=model/structure-annotations.md) — per-atom channel
  model (`tag`/`flag`/`value`) + the region-label vocabulary.
- [`structure-molstruct.md`](?doc=model/structure-molstruct.md) — the `.molstruct.json`
  save file: envelope · schema versioning · codec · file pairing.

**Companions** (separate model modules):
[`model/parse.md`](?doc=model/parse.md) (the read stack that produces a
Structure from engine output), [`model/overview.md`](?doc=model/overview.md)
§ 2 (the shared JSON vocabulary). **Frontend** (see
[`web/projects.md`](?doc=web/projects.md)): the
projects-sidebar module (the Load/Save UI over the doors) and the MolView
module (`molview.data`, the JS model primitives).

`Structure` is molbuilder's **lingua franca**: the one dataclass every builder
yields and every emitter consumes. This doc is the contract for the object
itself, its serialization codec, and the one door that reads/writes it as a
paired file on disk — across **both** the Python/CLI backend and the JS/user
frontend.

> **The one-object rule.** A structure's identity lives in three models that
> must stay byte-identical as data crosses the wire. Exactly one component per
> language may **name a field** or index a structure dict by key: the Python
> `Structure` codec, and the JS `data-model.js` accessors. Everyone else
> carries the dict verbatim as an opaque envelope. This is what stops the
> recurring "a field silently drops to its default on reload" bug (the
> `cell_origin → 0` regression); the field set appears in source **once per
> language**.

```mermaid
flowchart LR
    PY["molbuilder.Structure<br/>(Python dataclass — SSOT)"]
    JSON["JSON envelope<br/>(the wire dict)"]
    JS["molview.data<br/>(browser model)"]
    PY -- "to_wire() / to_dict()" --> JSON
    JSON -- "installMolecule()" --> JS
    JS -- "exportFile(range) → the structure" --> JSON
    JSON -- "from_dict() / StructureCodec" --> PY
```

---

## 1. The object (L1 data model)

**Module:** `molbuilder/structure.py`. **Tests:** `tests/test_structure.py`,
`tests/test_load.py`, `tests/test_pdb_ter.py`.

Adding a new format means a new method on `Structure`, not changes to the
builders.

```python
@dataclass
class Structure:
    elements:      List[str]                  # chemical symbols, length N
    positions:     np.ndarray                 # shape (N, 3), Angstrom
    atom_names:    Optional[List[str]] = None # PDB-style name; default = elements
    residue_ids:   Optional[List[int]] = None # 1-based; default = all 1
    residue_names: Optional[List[str]] = None # 3-letter; default = all "MOL"
    chain_ids:     Optional[List[str]] = None # single char; default = all "A"
    title:         str = ""                   # XYZ comment / PDB TITLE
    # ── metadata (each detailed in a sub-doc; serialized as one block) ──
    # cell, cell_origin, pbc, axis_kind, vacuum → structure-periodicity.md
    #     (NB: kgrid is NOT a structure field — it is a SiestaConfig DFT
    #      sampling knob; see engines/siesta.md)
    # regions (THE label store), annotations    → structure-annotations.md
    #     (frozen_atoms is not a field: it is the reserved label's one
    #      designated read, a cut of regions)
```

**Invariants enforced by `__post_init__`** (the single validation site):

- `positions` must reshape to `(N, 3)`, else `ValueError`.
- Every optional list, if provided, has length N; `None` gets the per-field
  default above.
- The metadata fields are validated/reconciled here too (see
  `structure-periodicity.md` for the cell/axis_kind reconciliation). Because
  `apply_metadata_dict` re-runs `__post_init__`, **all field validation lives
  in one place** — there is no second validator to drift from.

Atom **order is identity**: the 0-based index into `elements`/`positions`,
fixed by the atom order in the source file, is the canonical atom identity
carried everywhere (see [`model/overview.md`](?doc=model/overview.md) § 2 for
the full provenance + the 0-based/1-based boundary).

---

## 2. Backend surface (Python / CLI)

### 2.1 The whole-structure codec — `to_dict` / `from_dict` / `to_wire`

Three methods on `Structure` (`structure.py`), all shipped:

| Method | Purpose | Round-trips? |
|---|---|---|
| `to_dict()` → `dict` (`:574`) | The ONE canonical serializer: coordinates + per-atom columns + the full metadata block (via `metadata_to_dict()`). | **Yes** — `from_dict(s.to_dict())` reproduces `s` exactly. |
| `from_dict(d)` → `Structure` (`:593`) | The ONE canonical deserializer: builds the object, then `apply_metadata_dict` (the same validator a fresh Structure runs). | inverse of `to_dict` |
| `to_wire()` → `dict` (`:615`) | A read-only view the web layer builds on: identity columns + a **flattened** `periodicity` block (raw `cell`/`cell_origin`/`axis_kind`/`vacuum` **plus** the server-resolved `resolved_cell`/`resolved_cell_origin` the client must not recompute) + `annotations`. It carries **no** `positions`, **no** flat `atoms` render list, and **no** legacy aliases. | No — a different, flatter view (not a superset of `to_dict`) |

```python
# to_dict() — the loss-free round-trip unit; NOBODY else assembles this dict
{
    "title": ..., "elements": [...], "positions": [[x,y,z], ...],
    "atom_names": [...], "residue_ids": [...],
    "residue_names": [...], "chain_ids": [...],
    "metadata": self.metadata_to_dict(),   # the ONE metadata codec, nested verbatim
}
```

**Why two methods, not one flag.** `to_dict()` is the loss-free round-trip
unit (persistence, sidecar, CLI); `from_dict` inverts it. `to_wire()` is a
**separate** read-only view for the browser — identity columns + the
flattened, server-resolved periodicity + annotations — that `from_dict` never
has to invert. It is **not** a superset of `to_dict` (it drops `positions` and
the nested `metadata` shape); the flat `atoms` render list and the legacy
top-level aliases are added on top by the web layer (below). The resolved
cell/origin is computed **once**, inside `to_wire()`, so it can never drift or
drop.

> **Verified against code (2026-07-26):** `_shared.structure_to_dict`
> (`web/blueprints/_shared.py:346`) was **not** deleted (as an earlier draft
> of this contract claimed). It is the web layer's **composer**: it combines
> `workspace_payload(struct)` (the render `atoms` list, `text`/`xyz`, `issues`,
> `extra`) with `struct.to_wire()` (identity columns + periodicity +
> annotations) and adds the legacy top-level aliases existing consumers read.
> `ok_structure_response` (`:423`) wraps it. Code that needs only the metadata
> view calls `to_wire()` directly.

### 2.2 The metadata authority — `metadata_to_dict` / `apply_metadata_dict`

The structure metadata (periodicity + region tags + per-atom annotations) has
exactly **one** serialization authority: `Structure` itself.

```python
Structure.metadata_to_dict()      -> dict   # struct → JSON metadata dict (THE writer, :514)
Structure.apply_metadata_dict(d)  -> None   # JSON metadata dict → struct (THE reader, :533)
```

- **Scope** = the dataclass's own metadata fields: `regions`, `cell`,
  `cell_origin`, `pbc`, `axis_kind`, `vacuum`, `annotations`. (`regions` is the
  whole label store; a reserved label such as `frozen_atoms` is in it, so there
  is no field of its own to serialise — `structure-annotations.md` § 2.)
- **Strict JSON** — the dict is lists/dicts/bools/floats. `annotations` are
  JSON channel dicts (via `annotations_to_json`), **never** live `AtomChannel`
  objects (those live only in-memory; see `structure-annotations.md`).
- `apply_metadata_dict` is **full-replace**: an absent key resets that field
  to its default (absent `cell` → non-periodic; absent `regions` → none). It
  re-runs `__post_init__`, so validation is single-sourced.
- **NOT in scope** (they sit *around* the contract): `selection_rules` (a
  sidecar-only pass-through) and the sidecar **envelope** (`schema_version` /
  `n_atoms_total` / `structure_hash` / `created_by` / `created_at`) — see
  `structure-molstruct.md`.

**To add a metadata key:** (1) add the field to the dataclass with its
`__post_init__` validation; (2) add it to `metadata_to_dict()` +
`apply_metadata_dict()` — nowhere else; (3) if it must survive the sidecar,
bump `SCHEMA_VERSION` and register the new version in the read module (see
`structure-molstruct.md`); (4) if MolView must show/edit it, surface it in the web
`to_wire` periodicity block and read it in `molview.data`; (5) add a
save→load→apply round-trip test. You do **not** touch `to_dict`, the sidecar
`to_dict`, or `apply_to_structure` — they read the field set from the two
methods above, so they pick it up for free.

**To remove one:** delete it from the dataclass + both methods. Old sidecars
that still carry it load fine (`apply_metadata_dict` ignores unknown keys).
Never leave a "read-but-never-write" half-migration — that is the drift this
contract exists to prevent.

### 2.3 Geometry I/O

| Method | Format | Guarantees |
|---|---|---|
| `to_xyz(path=None, *, comment="")` (`:1074`) | xmol XYZ | line 1 = `N`; line 2 = comment-or-title; then `El x y z` per atom |
| `to_pdb(path=None)` (`:1101`) | PDB ATOM records | TITLE if set; serial capped `99999` (overflow → `*****`); residue id capped `9999`; chain id truncated to 1 char |
| `to_pyscf(*, as_string=False)` (`:1140`) | PySCF `gto.M` atom kwarg | `(symbol,(x,y,z))` tuples; multi-line string if `as_string=True` |
| `to_ase()` (`:1167`) | `ase.Atoms` | raises `ImportError` with install hint if ASE absent |
| `from_xyz(source, *, title=None)` (`:820`) | XYZ path **or** raw text | see requirements below |
| `from_pdb(source, *, title=None)` (`:879`) | PDB path **or** raw text | reads `ATOM`/`HETATM`; first MODEL only; TER handling below |

`from_*` accept a filesystem path or raw text (`_resolve_source` tries
`os.path.isfile` first, falls back to text).

**Round-trip guarantees.** XYZ: elements + positions exact; metadata drops to
defaults (XYZ has no slots). PDB: elements + positions + atom_names +
residue_ids + residue_names + chain_ids exact.

**`from_xyz` requirements:** line 1 = non-negative integer N; lines 2..N+2
read; trailing blank/short lines tolerated; bad header or short atom line →
`ValueError` with the offending line.

**`from_pdb` / TER handling** (pinned by `test_pdb_ter.py`): a segment counter
increments on every `TER`; each atom records `(chain_letter_or_"_",
segment_index)`; a chain letter unique to one segment passes through unchanged;
one spanning multiple segments is disambiguated by appending the segment index
(`A` → `A0`, `A1`); a blank chain-id column maps to `"A"` when unambiguous,
`"_<seg>"` when it spans segments. **Forbidden:** silently truncating serial
`> 99999` (must write `*****`), coercing a multi-char `chain_id` to `?` (must
truncate to first char), or crashing on a TER between ATOM blocks.

**Top-level dispatcher.** `molbuilder.load(path, *, format="auto")`
(`molbuilder/__init__.py:56`) reads `.xyz` or `.pdb` by extension; unknown
extension → `ValueError` with explicit instruction.

> **ASE owns the XYZ parse (2026-07-31).** `Structure.from_xyz` calls
> `ase.io.read(..., format="extxyz")` rather than splitting lines itself. ASE is
> a declared dependency **for this** — `pyproject.toml` names it *"XYZ I/O +
> atomic-number table"* — and its extended-XYZ reader is a superset reader: it
> takes the plain xmol layout and the `Lattice="…" pbc="…"` comment line alike,
> canonicalises an external tool's `FE`/`ZN` to `Fe`/`Zn`, and reads **every**
> frame of a multi-frame document.
>
> The hand-rolled parser it replaced read the atoms of the first block and
> nothing else, so a file this class had itself written with `to_extxyz` came
> back with **no cell, no pbc and one frame**. The project already knew: a
> second reader existed at `siesta/input.py`, whose comment said ASE *"gives us
> the lattice when present, which our hand-rolled parser doesn't"* — a correct
> diagnosis fixed at one call site by adding a reader beside the lossy one,
> while every other caller kept the lossy one. That second reader is now gone.
>
> Two things stay ours, and both are deliberate. The **title** is read from the
> comment line directly, because ASE's reader parses that line as `key=value`
> pairs and a human comment (`water molecule`) would come back as
> `{'water': True, 'molecule': True}`. And a `Lattice=` is adopted as an
> *explicit* cell only when some axis is periodic — our own writer emits the
> **resolved** box for isolated systems too, and adopting that would promote a
> derived value into a stored one (§ 2.2's raw-vs-resolved line).
>
> `from_pdb` is still ours: PDB carries residue, chain and atom-name columns
> this model owns, and no comparable second reader exists.

### 2.4 The paired-file door — `StructureCodec` (L2)

The `.xyz` and its optional `.molstruct.json` are **always** read/written
together. That pairing + atomicity is owned by one L2 object,
`StructureCodec` (`molbuilder/workingcopy_structure.py`).

**What it owns.** Four things, and nothing else in the system holds a second
copy of any of them:

1. **the pairing rule** — `<stem>.xyz` ↔ `<stem>.molstruct.json`, including how
   the sidecar's name is derived (`molstruct.sidecar_path_for`);
2. **the format choice** — a plain `.xyz` for one frame, extended XYZ for many,
   decided by the count and never asked as a separate question. **Both are
   `.xyz`**: extended XYZ is a strict superset of plain XYZ (the cell rides in
   the comment line, which a plain reader skips), so one extension covers both
   — the ordinary convention, and the only one `read` accepts;
3. **the sidecar envelope** — `schema_version`, the `structure_hash` pinning it
   to its geometry, and the one serialisation (`molstruct.dumps`);
4. **the invariants** — `no .json == empty metadata` in both directions,
   both-or-neither atomicity on write, and the periodicity gate on read.

**How it is shaped: one generator, and an adapter per destination.**

```python
class StructureCodec:                       # L2 (may use the L2 sidecar codec)
    def pair(self, struct, *, frames=None) -> StructurePair:
        """THE GENERATOR. A Structure as the two things that represent it
        outside memory: the coordinate document, the sidecar payload, whether
        that payload is worth keeping, and the suffix the format implies.
        Every outbound path below goes through this one call."""

    def files(self, struct, target, *, frames=None) -> list[tuple[Path, bytes]]:
        """TO THE WIRE. The pair as bytes WITH THE NAMES THEY BELONG UNDER --
        `target`'s suffix corrected to the one `pair` chose. What `write`
        writes, without writing it."""

    def write(self, struct, target, *, atomic=True, frames=None) -> Path:
        """TO DISK. The pair as one unit: geometry to `target`, and (when there
        is non-default metadata) the sidecar to sidecar_path_for(target).
        Atomic: each half staged to a temp sibling + os.replace'd; geometry
        swapped FIRST, then sidecar, so a reader never sees new geometry with
        a stale sidecar's atom indices. No-metadata + stale sidecar => sidecar
        removed. Owns both-or-neither."""

    def read(self, source_path, *, frames_out=None) -> Structure:
        """BACK IN. Parse geometry (.pdb by extension, else .xyz) AND its
        paired sidecar, applying metadata via molstruct.apply_to_structure.
        Missing sidecar = empty metadata (NOT an error). Runs the periodicity
        gate to REFUSE a cell nothing can be done with; it reports nothing,
        because what is true of the structure is said by whoever hands it
        over. `load` is the same call under its read-side name."""
```

> **The rule this shape exists to make checkable:** *every structure↔bytes
> translation goes through the codec, and every adapter has exactly one door.*
> An adapter with no door is either retired or unbuilt, and those have opposite
> fixes — so the question gets asked rather than answered by call count.
>
> **`write` names the file; `files` does not.** The difference is who chose the
> name. A project save was given an exact path through a picker, with an
> overwrite gate on it, so `write` puts the bytes exactly there. An export was
> given a *stem* and nothing else, so `files` completes it. Different questions,
> and conflating them is how the pairing rule came to have a second
> implementation in the browser (`web/molview.md` § 11.7).
>
> **Corrected 2026-07-31.** `pair` briefly named a range `.extxyz`, on the
> reasoning that a name should say which format it holds. It should not, and it
> could not: `read` dispatches on the extension and takes `.xyz` / `.pdb` only,
> so a saved trajectory **could not be reopened** — the project record was
> write-only for ranges, and no test noticed because the save test never read
> its file back. Extended XYZ under `.xyz` is both the convention and the thing
> that works.
>
> **Retired 2026-07-31:** `scratch_blob` / `from_scratch`, which round-tripped a
> structure through an in-memory `{xyz, sidecar}` **text** blob. Their last
> caller was `/api/structure/periodicity` before it took the envelope; a blob
> means a coordinate document is written to ask a question about coordinates,
> which is the thing `web/molview.md` § 11.7 forbids.

> **Why the file door is L2, not a method on L1 `Structure`.** The layering
> invariant (`tests/test_layering.py`) forbids an L1 module importing an L2
> one. Reading/writing the pair needs the **L2** sidecar codec
> (`sidecars/molstruct.py` — path derivation, atomic JSON, the envelope).
> Putting it on L1 would force a second L1 copy of the sidecar format — the
> exact drift this contract kills. So the pure data codec
> (`to_dict`/`from_dict`/`to_wire`/`metadata_to_dict`) is L1; the paired-file
> door is L2 `StructureCodec`, which routes the sidecar through the one
> metadata authority.

### 2.5 CLI

```bash
molbuilder dna ATGC | molbuilder pyscf - out.py     # Structure over stdin/stdout
molbuilder peptide ASEQ                            # → XYZ on stdout
```

**The CLI does not yet obey the rule above.** It reads and writes geometry
directly (`struct.to_xyz()` / `from_xyz`, at `cli.py:263, 267, 274, 1321, 1563,
1565`), so a CLI save emits the `.xyz` and nothing beside it. The consequence is
not cosmetic: `molbuilder modify` silently drops regions and frozen atoms, and
the CLI's `fdf` path cannot emit `Geometry.Constraints` from an `.xyz` + sidecar
pair, because its reader (`siesta/input.py:1455`, `pyscf/input.py:1286`) never
looks for the sidecar. The web surface has gone through the codec since the save
door was built, so the two surfaces disagree about what saving a structure means.

Routing CLI load/save through `StructureCodec` is open work (`plans/plan.md` **W15**), not shipped.

---

## 3. Frontend surface (JS / user)

The tab-facing surface is **two doors** plus the model primitives they call.
A tab calls a door and nothing below it; reaching around a door (a second
file stack, a browser-written sidecar, poking the store) is wrong by
definition.

### 3.1 The open door — `projects.parser.openMolecule`

> **There was a save door beside it until 2026-09-02.**
> `parser.saveMolecule` took a path it was GIVEN and posted it, handing a
> `needsOverwrite` back for the caller to deal with — so it always needed a UI
> layer on top, and `modify/structure/save.js` was that layer. When the Save
> panel moved onto `projects.molviewFiles.save("project", …)`, which asks WHERE
> and owns the overwrite flow itself ([`tabs.md` § 6](?doc=web/tabs.md)), the
> half-door had no caller and was deleted rather than kept as a second way to
> write one file. **Opening needs the pairing rule; saving needs a
> destination** — different questions, one door each.

It is **FILE-ONLY**: the door hands a `path` to the **server**, which owns file access, the
`.xyz`↔`.molstruct.json` pairing, and the sidecar schema. The browser reads
no bytes, derives no sidecar path, writes no coordinate document, and **never
authors the sidecar schema** (a browser-written sidecar had no
`schema_version`, so the load door rejected the pair — the save→reload breaker,
task #75).

| Door | Does | Server seam |
|---|---|---|
| `openMolecule(path, {confirmDiscard?})` | dirty-gate → `molview.data.installMolecule({path})` | `POST /api/build/load` (`build.py:841`) → `StructureCodec.read` |
| *(saving)* `projects.molviewFiles.save("project", stem, exportFile(range))` | asks WHERE (`chooseSavePath`) → POST → confirms an overwrite → refreshes the sidebar | `POST /api/structure/save` → `struct_from_body` + `StructureCodec.write` (stamps `schema_version` + real `structure_hash`) |

`openMolecule` is **only** for a project-file path. Generated text
(smiles/dna/…) has no file, so generators call
`molview.data.installMolecule({text})` directly — the model primitive, not the
door. **Saving writes XYZ only** — the codec's generator emits a plain `.xyz`
or an extended one and there is no PDB serializer, so a save to a `.pdb` path
would receive XYZ bytes (the door forces `.xyz`). Asymmetry: `openMolecule`
*loads* a `.pdb` (the parse seam sniffs PDB); nothing saves one.
A 409 "exists" envelope → `{needsOverwrite:true}`, and the door confirms and
retries with `{overwrite:true}` — the dialog is `projects`' own
(`confirmDestructive`), so the model layer stays DOM-free.

### 3.2 The model primitives + the JS key-namer

`molview.data` (`lib/molview/model.js`) is the browser model:
`installMolecule({path} | {text[,sidecar,…]})`, `exportFile(range) → {name,
structure, frames?}`, `markSaved(path)`. Named-key reads of the wire dict happen
in **one** place — the model's accessors (`getUnitCell`,
`getUnitCellOrigin`, `getVacuum`, `getAxisKind`, …), the JS analogue of
Structure's codec. Everywhere else the browser carries `periodicity` /
`annotations` / `atoms` as **opaque blobs** (verbatim deep-clone, no field
whitelist), so a server-added field survives persistence untouched. The one
deliberate write accessor, `setPeriodicity`, names only the lattice keys it
*manages* (it is an editor, not a carry) and is spread-based, so it cannot
drop an unlisted field.

### 3.3 SETTLE-BEFORE-READY — one store write per load

`installMolecule` installs the FINAL model — sidecar-enriched atoms, source,
periodicity, AND the cleared selection — in **one** synchronous write; the
"ready" signals fire at that write, and **no second store write may follow**
(it would land after "ready" and clobber whatever a consumer already did).

```mermaid
sequenceDiagram
    participant U as Sidebar
    participant D as parser.openMolecule(path)
    participant IM as molview.data.installMolecule({path})
    participant BL as server /api/build/load
    U->>D: commit path (+confirmDiscard if dirty)
    D->>IM: { path }
    IM->>BL: POST { path }
    BL-->>IM: StructureCodec.read (.xyz + paired .molstruct.json) → atoms + periodicity + annotations
    Note over IM: ONE synchronous write — model SETTLED. getNAtoms() = ready gate.
    IM->>IM: await _anchorTimeline() (prune + persist)
    D-->>U: resolve — NO second store write
```

> The 2026-07 regression that defined this: load used to install atoms (open
> the ready gate), then `await adoptSession({selection:[]})` ~300 ms later,
> wiping a click made in the gap. Fix: sidecar atoms ride in on the single
> install; the trailing write is gone.

### 3.4 Consumer map (shipped)

| Consumer | `file:function` | Call |
|---|---|---|
| Molbuilder tab — Load / dblclick | `modify/selection-bootstrap.js:_commitFile` | `openMolecule(path, {confirmDiscard})` |
| Molbuilder tab — Save panel | `modify/structure/save.js:save` | `projects.molviewFiles.save("project", stem, exportFile())` — the door asks WHERE and owns the overwrite flow (`tabs.md` § 6). **It does not go through `saveMolecule`**, and since 2026-09-02 nothing does |
| Transport commit | `lib/transport/core.js:_showInMolview` | `openMolecule(path)` + `molview.mount` |
| Spectra commit | `spectra/viewer.js:_commitStructure` | `openMolecule(path)` + `molview.mount` |
| Results structure inspector | `lib/inspectors/structure.js` | `openMolecule(path)` + `molview.mount` |
| Structure-optimization | `structure-optimization/viewer.js:_commitStructure` | `openMolecule(path)`; reads state off the model |
| Generators (smiles/dna/…) | `modify/structure/*.js` → `page.js` | `molview.data.installMolecule({text})` (not a door) |
| Trajectory inspector | `lib/trajectory/core.js` | `installMolecule({text})` + `reloadFrames(...)` |

> The Molbuilder tab's static files live under `modify/` — the `/modify`
> route was renamed to `/molbuilder`, but the directory name is historical.

"Load + mount is ONE shared path": Transport, Spectra, and the Results
inspector each open the picked file via `openMolecule(path)`, then
`molview.mount(host, ws, {mode, owner})`. When each hand-rolled its own copy
they drifted (the inspector read raw XYZ and dropped the sidecar — the label
bug). One path = the sidecar-correct load, for every tab.

---

## 4. The wire contract (backend ⇄ frontend)

The doors sit over the projects byte-layer + the parse seam; the model
primitives sit over the same server. The dependency points one way
(`projects.parser → molview.data`, resolved by a call-time lookup, so there is
no cycle):

```mermaid
flowchart TB
    subgraph TAB["Tab / UI (buttons + injected UI policy)"]
        B["Load / Save / sidebar dblclick"]
    end
    subgraph PR["molbuilder.projects — concealed sidebar package"]
        DOORS["parser.openMolecule (format-aware OPEN door)<br/>molviewFiles.save (the save flow)"]
        BYTES["readFile / writeFile (format-blind BYTES)"]
        DOORS -->|"move bytes via"| BYTES
    end
    subgraph MV["molview.data — MODEL primitives (DOM-free)"]
        IM["installMolecule({text,sidecar})"]
        EF["exportFile(range) → the structure"]
    end
    SRV[("server: /api/files/*  ·  /api/build/load  ·  /api/structure/save")]
    B --> DOORS
    DOORS -->|"install / serialise"| MV
    BYTES --> SRV
    IM -->|"parse (StructureCodec.read)"| SRV
```

| Layer | Owns | Never |
|---|---|---|
| `projects` byte layer | locating a file + moving its **bytes** | parses a molecule; knows the model |
| `projects.parser` doors | read→parse→install (load); serialise→write (save); the pairing | owns a parser (calls the seam) |
| `molview.data` primitives | text(+sidecar) ⇄ live molecule; the atomic install | fetches a file; owns a file endpoint |
| tab / UI | wiring buttons; UI policy (dirty/overwrite — injected) | reaches past a door |

**Where the sidecar schema lives** (server, one home):
`sidecars/molstruct.py` — `apply_to_structure(struct, dict)` (`:370`),
`load_text(text)`, `save(...)` (`:315`), `sidecar_path_for(xyz)` (`:89`). The
byte layer knows the pair only as "which bytes travel together"; interpreting
it (parse + apply the schema) happens only inside the server seam. Clicking a
`.molstruct.json` in the sidebar shows its JSON via the `source` inspector —
open the paired `.xyz` to view the structure.

---

## 5. The round-trip invariant + enforcement

A single Python test constructs a Structure with **every** metadata field set
to a non-default value and asserts it survives each hop unchanged — the test
that would have caught `cell_origin → 0` at the source
(`tests/test_structure_authority_roundtrip.py`):

```python
def _fully_populated_structure():
    s = Structure(elements=["C", "O"], positions=[[1.,2.,3.], [4.,5.,6.]])
    s.apply_metadata_dict({
        "cell": [[10,0,0],[0,10,0],[0,0,10]], "cell_origin": [1.5,2.5,3.5],
        "pbc": [True,True,False], "axis_kind": ["periodic","periodic","isolated"],
        "vacuum": [0.,0.,12.],
        # every label in one store, the reserved one included
        "regions": {"electrode":[0], "channel":[1], "frozen_atoms":[0]},
        # a value channel: kind ∈ {tag,flag,value}; value data is a sparse
        # (string-keyed) idx→value map — see structure-annotations.md § 2
        "annotations": {"charge": {"kind":"value", "data":{"0":0.1, "1":-0.1}}},
    })
    return s

# to_dict → from_dict preserves every field; to_wire carries resolved_cell_origin;
# StructureCodec.write → read preserves metadata and writes the .molstruct.json pair.
```

The end-to-end E2E fixture (`test_structure_inspector_measurement_e2e.py`)
sets a non-zero `cell_origin` and asserts `molview.data` reports that origin
after load → serialise → restore — pinning the field through the JS mirror,
not just the Python codec.

**Anti-patterns (rejected by reference to this doc):** hand-rolled structure
repacks; raw-dict metadata access (`d["cell_origin"]`, `d.get("pbc")`) outside
the two codecs; a JS field whitelist for periodicity; a second file stack or a
browser-authored sidecar. Named-key access lives in exactly one place per
language.

---

## 6. Status

**Shipped (2026-07):** the L1 codec (`to_dict`/`from_dict`/`to_wire`) + the L2
`StructureCodec` (`pair` / `files` / `write` / `read`); the server seams
(`/api/build/load`, `/api/structure/save`, `/api/structure/export`); the JS doors
(`parser.js`) with every consumer above repointed; the JS periodicity
field-whitelist replaced by verbatim deep-clone; the old `molview.data` file
stack + `/api/workingcopy/*` door path removed. `_shared.structure_to_dict`
retained as the web composer (`workspace_payload` + `to_wire` + legacy aliases),
not deleted.

**Consolidated 2026-07-31.** Every adapter now has exactly one door: `write` →
`/api/structure/save`, `files` → `/api/structure/export`, `read` →
`/api/build/load`. The export door answers with the files **named**, so no caller
derives a filename or re-serialises a sidecar; `scratch_blob` / `from_scratch`
were retired with the `{xyz, sidecar}` blob shape that was their only reason to
exist (§ 2.4).

**Open work** (`plans/plan.md` **W15**): route the **CLI** load/save through
`StructureCodec` so a CLI save emits the `.xyz` + `.molstruct.json` pair like
the web save does (task #73) — today the CLI writes geometry only, which is the
last surface not obeying the rule in § 2.4.
