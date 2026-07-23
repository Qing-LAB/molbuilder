# Structure authority — the ONE codec + the ONE file door

> **Authoritative design for consolidating ALL structure data — metadata,
> coordinates, and the paired `.xyz`+`.molstruct.json` file — behind the
> `molbuilder.structure.Structure` dataclass.** Every producer of a structure
> dict and every reader/writer of a structure file goes THROUGH Structure.
> Nobody outside Structure names a metadata field or picks a key out of a raw
> dict. This is the enforcement contract; reviewers reject hand-rolled repacks
> and raw-dict metadata access by reference to this doc.
>
> **Supersedes** the hand-assembly in `web/blueprints/_shared.py:structure_to_dict`
> and the field-listing return literals. **Companions:**
> [`structure-load-save-contract.md`](structure-load-save-contract.md) (the JS
> doors that call the server seam), [`sidecar-contract.md`](sidecar-contract.md)
> (the three-stage BC contract the sidecar carries),
> [`structure-periodicity.md`](structure-periodicity.md) (the cell/origin field
> semantics), [`data-vocabulary.md`](data-vocabulary.md) (the shared JSON names).

---

## §1 The problem — scattered field ownership

A structure's identity is spread across **three models** that must stay
byte-identical as data crosses the wire:

```
molbuilder.Structure  ──►  JSON envelope  ──►  molview.data (JS)
 (Python dataclass)        (the wire dict)      (browser model)
```

The **metadata** contract is already consolidated: `Structure.metadata_to_dict()`
/ `apply_metadata_dict()` are the ONE validator+codec, and the sidecar +
`apply_to_structure` both delegate to them (see the comment block at
`sidecars/molstruct.py:130`). Good.

But the **whole-structure → dict** step is NOT consolidated. The field name
`"cell_origin"` is typed out by hand in four places that must agree:

| # | Location | What it does |
|---|---|---|
| 1 | `Structure.metadata_to_dict()` | emits `cell_origin` (correct — the authority) |
| 2 | `sidecars.molstruct.to_dict()` return literal | re-lists `cell_origin` in the sidecar envelope |
| 3 | `_shared.structure_to_dict()` `periodicity` literal | re-lists `cell_origin` + re-runs `resolve_cell_origin()` by hand |
| 4 | JS `_canvas-state-impl.js:_normPeriodicity` whitelist | re-lists which periodicity keys survive persistence |

Any field added, renamed, or re-typed must be found in all four. Miss #3 or #4
and the field silently drops to a default — **exactly the `cell_origin → 0`
bug**, which has recurred because the design lets it. Encapsulation 101: the
field set must live in **one** place.

Two more gaps:

- **No whole-structure codec.** `Structure` has `metadata_to_dict` (metadata
  only), `from_xyz`/`to_xyz` (coords only) — but no `to_dict`/`from_dict` that
  round-trips coordinates **and** metadata together. `_shared.structure_to_dict`
  hand-assembles that shape; that is the design smell, not a helper.
- **No paired-file door.** The `.xyz` and its optional `.molstruct.json` are
  ALWAYS read/written together, but the pairing rule (`sidecar_path_for`), the
  read order, and the atomicity live scattered across `parser.js` (JS),
  `_shared`, and the sidecar module. No single object owns "read/write this
  structure as one atomic unit."

---

## §2 The principle — Structure is the only key-namer

> **Only `Structure`'s codec methods may name a metadata/coordinate field or
> index a structure dict by key. Everyone else passes the dict verbatim as an
> opaque envelope.**

Concretely, OUTSIDE `structure.py` (and its delegate `sidecars/molstruct.py`
envelope layer) there is **no** `d["cell_origin"]`, no `d.get("pbc")`, no
`r.periodicity.resolved_cell_origin`, no field whitelist. A blueprint that has a
`Structure` calls `struct.to_wire()`; a blueprint that receives a dict calls
`Structure.from_dict(d)`. The JS mirror follows the same rule: the wire dict is
carried whole; named-key reads happen in **one** accessor module
(`data-model.js`), which is the JS analogue of Structure's codec.

This makes the field set appear in source **once per language** — Python in
`Structure`, JS in `data-model.js` accessors — and a new field is added in those
two places, full stop.

---

## §3 The API surface

### §3.1 The whole-structure codec (coords + atoms + metadata)

```python
class Structure:

    def to_dict(self) -> dict:
        """The ONE canonical serialiser. A single dict carrying everything
        needed to reconstruct this Structure: coordinates, per-atom columns,
        AND the full metadata field set (via metadata_to_dict()). Round-trip
        invariant: Structure.from_dict(s.to_dict()) reproduces s exactly.
        NOBODY else assembles this dict."""
        return {
            "title":     self.title,
            "elements":  list(self.elements),
            "positions": self.positions.tolist(),
            # per-atom identity columns
            "atom_names":    list(self.atom_names),
            "residue_ids":   list(self.residue_ids),
            "residue_names": list(self.residue_names),
            "chain_ids":     list(self.chain_ids),
            # the full metadata block — delegated to the ONE metadata codec
            "metadata":  self.metadata_to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "Structure":
        """The ONE canonical deserialiser — inverse of to_dict(). Constructs a
        Structure from the canonical dict, then validates via __post_init__ and
        apply_metadata_dict (the SAME single validator a fresh Structure runs).
        NOBODY else picks coordinate/metadata keys out of a structure dict."""
        s = cls(
            elements=list(d["elements"]),
            positions=np.asarray(d["positions"], float),
            atom_names=list(d.get("atom_names") or []),
            residue_ids=list(d.get("residue_ids") or []),
            residue_names=list(d.get("residue_names") or []),
            chain_ids=list(d.get("chain_ids") or []),
            title=d.get("title", ""),
        )
        s.apply_metadata_dict(d.get("metadata"))   # full-replace + revalidate
        return s
```

`to_dict()`/`from_dict()` are the **round-trip unit** — pure, filesystem-free,
loss-free. This is the shape the persistence layer stores and the shape
`from_dict` reconstructs. `metadata` nests the existing `metadata_to_dict()`
output verbatim: coordinates and metadata each have exactly one home.

### §3.2 The wire view (server → client, read-only)

```python
    def to_wire(self, *, extra: Optional[dict] = None) -> dict:
        """The read-only shape an API endpoint returns to the JS client. It is
        to_dict() PLUS server-resolved derivations the client must not
        recompute (resolve_cell / resolve_cell_origin — the ONE resolver) PLUS
        the flat `atoms` render list and the legacy aliases existing consumers
        read. Not round-tripped by from_dict; it is a superset view. The
        resolved-origin lives here, computed in ONE place, so it can never
        drift or drop."""
```

`to_wire()` is where `resolved_cell` / `resolved_cell_origin` are computed — by
calling `self.resolve_cell()` / `self.resolve_cell_origin()` **once**, inside
Structure. `_shared.structure_to_dict` is deleted; `ok_structure_response`
calls `struct.to_wire(extra=...)`. The `periodicity` sub-dict is assembled by
Structure, not by the blueprint, so a field added to `metadata_to_dict`
automatically appears on the wire.

> **Why two methods, not one flag.** `to_dict()` is the loss-free round-trip
> unit (persistence, sidecar, CLI). `to_wire()` is a read-only superset for the
> browser (derived fields + render list + back-compat aliases) that
> `from_dict` never has to invert. Keeping them separate stops the derived
> fields from leaking into the persisted/round-tripped shape.

### §3.3 The paired-file door (the `.xyz` + `.molstruct.json` unit) — **L2**

> **Layering note (why this is NOT a method on the L1 `Structure` dataclass).**
> The layering invariant (`tests/test_layering.py`, from `design.md`) forbids an
> L1 module importing an L2 one. Reading/writing the pair inherently needs the
> **L2** sidecar codec (`sidecars.molstruct` — path derivation, atomic JSON
> save/load, the envelope). Putting the file door on L1 `Structure` would either
> violate that invariant or force a **second, L1 copy** of the sidecar format —
> the exact drift this whole doc exists to kill. So the pure data codec
> (`to_dict`/`from_dict`/`to_wire`/`metadata_to_dict`) is L1 on `Structure`; the
> **paired-file door is the existing L2 `StructureCodec`**
> (`molbuilder/workingcopy_structure.py`), which is *already* "`.xyz` +
> `.molstruct.json` ⇄ Structure" and already routes the sidecar through the ONE
> metadata authority (`struct.metadata_to_dict()` → `molstruct.to_dict`
> envelope). We consolidate INTO it, not reinvent.

```python
class StructureCodec:               # molbuilder/workingcopy_structure.py (L2)

    def read(self, source_path) -> Structure:          # alias of load()
        """Parse the geometry (.pdb by extension, else .xyz) AND its paired
        sidecar (`sidecar_path_for`), applying metadata via
        `molstruct.apply_to_structure`. Missing sidecar = empty metadata (NOT
        an error). Owns the pairing rule + read order."""

    def write(self, struct: Structure, target, *, atomic: bool = True) -> Path:
        """Write the pair as one unit: geometry to `target` and, when there is
        non-default metadata, the sidecar to `sidecar_path_for(target)`.
        Atomic: each half staged to a temp sibling + os.replace'd; geometry
        swapped first, then sidecar, so a reader never sees a new geometry with
        a stale sidecar's atom indices. No-metadata + stale sidecar on disk =>
        the sidecar is removed (`no .json == empty metadata`). Owns the
        both-or-neither invariant."""
```

`read`/`write` own the **pairing** (which sidecar goes with which geometry) and
the **atomicity** (both halves swap or neither). Internally they reuse the
existing primitives — `Structure.from_xyz`/`to_xyz`, `sidecar_path_for`,
`molstruct.load`/`save`, `apply_to_structure` — but the pairing + atomic-pair
logic now lives in **one** place instead of being re-implemented by every
caller. `write` extends the single-file atomicity of `molstruct.save` to the
**pair**: geometry-temp fsync'd + `os.replace`d first, then the (already-atomic)
`molstruct.save` for the sidecar.

### §3.4 The sidecar envelope — delegated, not re-listed

`sidecars.molstruct.to_dict()` keeps its **envelope** job (schema_version,
n_atoms_total, structure_hash, selection_rules, created_*) but stops
re-enumerating the metadata fields. It already validates them through
`structure_fields_via_dataclass` (a scratch Structure IS the schema); the return
literal changes from listing each field to spreading the validated field dict:

```python
    fields = structure_fields_via_dataclass(n_atoms_total, fields or {})
    return {
        **_ENVELOPE(schema_version, n_atoms_total, structure_hash,
                    created_by, created_at),
        **fields,                       # ← the metadata block, verbatim
        "selection_rules": normed_rules,
    }
```

So even the sidecar layer names no metadata field by hand; the field set flows
from `metadata_to_dict` → `structure_fields_via_dataclass` → `**fields`.

---

## §4 Who calls what (the enforcement map)

| Caller | Before (scattered) | After (through Structure) |
|---|---|---|
| `_shared.structure_to_dict` | hand-lists `periodicity`, re-runs resolvers | **deleted** → `struct.to_wire(extra=…)` |
| `ok_structure_response` | wraps `structure_to_dict` | wraps `struct.to_wire` |
| `/api/build/load` seam | parse text, `apply_to_structure`, hand-repack | parse text → `Structure`, return `struct.to_wire()` |
| CLI load/save | `from_xyz` + separate sidecar calls | `StructureCodec().read(path)` / `.write(struct, path)` (L2 door, §3.3) |
| Sidecar `to_dict` | lists each metadata field | `**fields` spread (§3.4) |
| JS `data-model.js` | `r.periodicity.resolved_cell_origin` read in accessors | unchanged — this IS the single JS key-namer (`getUnitCellOrigin`, …) |
| JS `_canvas-state-impl.js:_normPeriodicity` + `_clonePeriodicity` | field whitelist (6 hand-listed keys; drops any unknown) | **verbatim deep-clone** (`_deepCloneJson`) — carry the whole sub-dict; a server-added field survives |
| JS `_install.js:230` `if (opts.periodicity) r.periodicity = opts.periodicity` | *(kept)* | **kept** — traced callers: it is a **verbatim carry**, not a key clobber (below) |

**JS rule mirror:** the browser carries the wire dict's `periodicity` /
`annotations` / `atoms` as opaque blobs — the store deep-clones the whole
periodicity object with **no field list**. Named-key reads happen ONLY in
`data-model.js` accessors (`getUnitCellOrigin`, `getUnitCell`, `getVacuum`,
`getAxisKind`, …) — the single JS key-namer. The one deliberate exception is
`setPeriodicity` (a WRITE accessor that edits specific lattice keys on a Cell-page
edit); it names the keys it *manages*, which is legitimate — it is an editor, not
a carry.

**Why `_install.js:230` stays (corrected — the design first said "remove it").**
Tracing every `installMolecule` caller: `openMolecule`, `modify/viewer`, the
demos and `structure/page.js` do **not** hand-build periodicity — they omit it
(server derives from text + sidecar) or carry `structure.periodicity` **verbatim**
(a prior server response). Only `trajectory/core.js` builds `{cell: lat}` from the
trajectory's own lattice — the one place that info lives (no sidecar). Line 230
overlays these onto the `/api/build/load` re-parse (which, without a sidecar,
would otherwise lose the cell). It carries the blob whole; it never picks or drops
a server key. Consistent with the verbatim rule, not a violation. The real drift
surface was the whitelist above.

---

## §5 The round-trip invariant test (pin EVERY metadata field)

A single Python test constructs a Structure with **every** metadata field set to
a **non-default** value, then asserts it survives each hop unchanged. This is
the test that would have caught `cell_origin → 0` at the source.

```python
def _fully_populated_structure():
    s = Structure(elements=["C", "O"], positions=[[1., 2., 3.], [4., 5., 6.]])
    s.apply_metadata_dict({
        "cell":         [[10,0,0],[0,10,0],[0,0,10]],
        "cell_origin":  [1.5, 2.5, 3.5],      # ← NON-zero: the field that dropped
        "pbc":          [True, True, False],
        "axis_kind":    ["a", "b", "vacuum"],
        "vacuum":       [0.0, 0.0, 12.0],
        "regions":      {"electrode": [0], "channel": [1]},
        "frozen_atoms": [0],
        "annotations":  {"charge": {"kind": "float", "values": [0.1, -0.1]}},
    })
    return s

def test_to_dict_from_dict_round_trip_preserves_all_metadata():
    s = _fully_populated_structure()
    r = Structure.from_dict(s.to_dict())
    for f in ("cell","cell_origin","pbc","axis_kind","vacuum",
              "regions","frozen_atoms"):
        assert _eq(getattr(r, f), getattr(s, f)), f
    assert r.metadata_to_dict() == s.metadata_to_dict()

def test_read_write_pair_round_trip_preserves_all_metadata(tmp_path):
    s = _fully_populated_structure()
    StructureCodec().write(s, tmp_path / "m.xyz")        # L2 door (§3.3)
    r = StructureCodec().read(tmp_path / "m.xyz")
    assert r.metadata_to_dict() == s.metadata_to_dict()
    assert (tmp_path / "m.molstruct.json").exists()      # pair written

def test_to_wire_carries_resolved_cell_origin():
    s = _fully_populated_structure()
    w = s.to_wire()
    assert w["periodicity"]["cell_origin"] == [1.5, 2.5, 3.5]
    assert w["periodicity"]["resolved_cell_origin"] == [1.5, 2.5, 3.5]
```

Plus the existing E2E (`test_structure_inspector_measurement_e2e.py`) fixture
`_write_xyz_with_cell_sidecar` is extended to set a **non-zero `cell_origin`**,
and the test asserts `molview.data` reports that origin after load → serialise →
restore — pinning the field end-to-end through the JS mirror, not just the
Python codec.

---

## §6 Migration order (small, verifiable steps)

1. **Add** the L1 codec `Structure.to_dict`/`from_dict`/`to_wire` + the L2
   paired-file door `StructureCodec.read`/`.write` (pure additions; no caller
   change yet). Land §5's Python round-trip tests. **✅ DONE** — methods added,
   `tests/test_structure_authority_roundtrip.py` green (8 codec + 2 layering).
2. **Repoint** `_shared`: delete `structure_to_dict`, route
   `ok_structure_response` + the build/load seam through `to_wire`. Run the
   blueprint + E2E suites.
3. **Spread** the sidecar `to_dict` envelope (§3.4); delete the field list.
4. **JS**: replace the `_normPeriodicity`/`_clonePeriodicity` field whitelist
   with a verbatim `_deepCloneJson`; keep `_install.js:230` (traced: verbatim
   carry, load-bearing for the sidecar-less re-install + trajectory paths);
   confirm `data-model.js` accessors are the single key-namer. Extend the E2E
   fixture (non-zero origin). **✅ DONE** (whitelist removed; line 230 kept + doc
   corrected).
5. **CLI**: route load/save through `StructureCodec().read`/`.write` (§3.3).

Each step is independently testable and leaves the field set named in exactly
one place per language when done.
