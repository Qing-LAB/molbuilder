# Spec — Structure dataclass + I/O

**Module**: `molbuilder/structure.py` &nbsp;·&nbsp; **Tests**: `tests/test_structure.py`, `tests/test_load.py`, `tests/test_pdb_ter.py`

`Structure` is the lingua franca passed between every builder and
every emitter.  Adding a new format means a new method on `Structure`,
not changes to the builders.

## Required fields

```python
elements:      List[str]                  # chemical symbols, length N
positions:     np.ndarray                 # shape (N, 3), Angstrom
atom_names:    Optional[List[str]] = None # PDB-style name; defaults to elements
residue_ids:   Optional[List[int]] = None # 1-based; defaults to all 1
residue_names: Optional[List[str]] = None # 3-letter; defaults to all "MOL"
chain_ids:     Optional[List[str]] = None # single char; defaults to all "A"
title:         str = ""                   # XYZ comment / PDB TITLE
```

Invariants enforced by `__post_init__`:

* `positions` is reshape-able to `(N, 3)`; otherwise raise `ValueError`.
* Every optional list, if provided, has length N.  None defaults are
  applied per-field above.

## Output methods

| method | format | guarantees |
| --- | --- | --- |
| `to_xyz(path=None, *, comment="")`  | xmol XYZ | first line is `N`; second line is comment-or-title; one atom per line `El x y z` |
| `to_pdb(path=None)`                  | PDB ATOM records | TITLE if `title` set; ATOM serial capped at `99999` (overflow becomes `*****`); residue id capped at `9999`; chain id truncated to 1 char |
| `to_pyscf(*, as_string=False)`       | PySCF `gto.M` atom kwarg | list of `(symbol, (x, y, z))` tuples by default; multi-line string when `as_string=True` |
| `to_ase()`                           | `ase.Atoms`         | raises `ImportError` with install hint if ASE not installed |

Round-trip guarantees:

* **XYZ round-trip** (write → `from_xyz`): elements + positions exactly preserved; metadata (atom_names / residues / chains) drops to defaults because XYZ has no slots for it.
* **PDB round-trip** (write → `from_pdb`): elements + positions + atom_names + residue_ids + residue_names + chain_ids exactly preserved.

## Input methods (from disk or text)

```python
@classmethod from_xyz(source: str | Path, *, title: str | None = None) -> Structure
@classmethod from_pdb(source: str | Path, *, title: str | None = None) -> Structure
```

Both accept either a filesystem path or the raw text content; the
helper `_resolve_source` tries `os.path.isfile` first, falls back to
treating the input as text.

`from_xyz` requirements:

* First line must be parseable as a non-negative integer N.
* Lines 2 through `N+2` (inclusive of comment) are read; trailing
  blank/short lines after the last atom are tolerated.
* Bad header → `ValueError` with the offending line.
* Atoms shorter than `El x y z` → `ValueError` with the offending line.

`from_pdb` requirements:

* Reads `ATOM` and `HETATM` records.  Other record types (HEADER,
  REMARK, CONECT, ANISOU, ...) are ignored.
* Multi-MODEL files: only the first MODEL block is read.  Subsequent
  MODELs are dropped.
* `TITLE` records (cols 11–80) accumulate into `title`.
* **TER record handling** (the spec for what `tests/test_pdb_ter.py`
  pins down):
  * A segment counter increments on every `TER` line.
  * Each atom records `(chain_letter_or_underscore, segment_index)`.
  * After parsing, a chain letter unique to one segment passes
    through unchanged → back-compat for well-formed PDBs.
  * A chain letter spanning multiple segments is disambiguated by
    appending the segment index (e.g. `A` reused across TERs becomes
    `A0`, `A1`).
  * A blank chain-id column maps to `"A"` when unambiguous; to
    `"_<seg>"` when it spans multiple segments.

## Top-level `molbuilder.load(path)`

Convenience dispatcher: reads `.xyz` or `.pdb` based on extension.
Unknown extension → `ValueError` with explicit instruction.

## Format detection: forbidden-pattern list

The PDB parser must NOT:

* Truncate atom serial silently when `i+1 > 99999`.  It writes
  `*****` (per PDB spec).
* Coerce a multi-character `chain_id` to `?`.  It truncates to the
  first character.
* Crash on a TER record between two ATOM blocks.

## Test reference

* `test_structure.py` — basic construction, XYZ/PDB/PySCF/ASE I/O,
  centred / concat helpers.
* `test_load.py` — `from_xyz` / `from_pdb` / top-level `load()` over
  every input shape (path, text, malformed).
* `test_pdb_ter.py` — TER-record handling fixtures for well-formed,
  reused-letter, blank-chain-id, single-chain, blank-no-TER, multi-
  consecutive-TERs, residue tuple-uniqueness.
