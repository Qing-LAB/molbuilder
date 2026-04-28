# Spec — builders

**Modules**: `molbuilder/peptide.py`, `molbuilder/nucleic.py`,
`molbuilder/smiles.py`, `molbuilder/pubchem.py`, `molbuilder/backends/*`
&nbsp;·&nbsp; **Tests**: `tests/test_peptide.py`, `tests/test_nucleic.py`,
`tests/test_smiles_and_siesta.py`, `tests/test_residues.py`

A *builder* is any function that returns a `Structure`.  All builders
share a uniform output contract: a fully-populated Structure with
`elements + positions` set, and `atom_names / residue_ids / residue_names
/ chain_ids` populated where the source format provides them.

## Peptide builder

```python
build_peptide(sequence: str, *, title=None, add_hydrogens=True) -> Structure
```

* `sequence`: 1-letter codes (`"ARNDC"`), with `[XXX]` escapes for
  modified residues (e.g. `"AR[SEP]C"` for phospho-Ser).  Whitespace
  ignored.  Modified-residue codes are listed in
  `molbuilder.residues.MODIFIED_RESIDUES` (SEP, TPO, PTR, MLY, M3L,
  ALY currently).
* `add_hydrogens=True`: protonate via OpenBabel (preferred) or RDKit
  (fallback).  If neither is installed, returns heavy-atom-only with
  a `RuntimeWarning`.
* Element field is **always stripped** before being stored —
  BioPython sometimes returns space-padded elements like `" C"` and
  downstream species detection in the SIESTA / PySCF emitters
  requires clean strings.  This is the S1 fix.

## Nucleic-acid builders

```python
build_dna(sequence: str, *, backend="auto", form="B", terminal="OH",
          protonate_phosphates=True, title=None) -> Structure
build_rna(sequence: str, *, backend="auto", form="A", terminal="OH",
          protonate_phosphates=True, title=None) -> Structure
```

* `backend`:
  * `"auto"`: prefer `amber` (AmberTools `tleap`) if installed, else
    `rdkit`.  No silent skip — if neither is available, raise
    `BackendUnavailable` with install instructions.
  * `"rdkit"`: always works, returns folded conformer (not helical).
  * `"amber"`: AmberTools' `tleap` with `leaprc.DNA.OL15` /
    `leaprc.RNA.OL3`; produces extended chain with correct
    Amber-OL15 chemistry.  Subprocess gets a `timeout=120` so a hung
    `tleap` doesn't block forever.
* `form`: helix form letter; only `amber` honours it (warns
  otherwise that the embedded conformer isn't helical).
* `terminal`: `"OH"`, `"5P"`, `"3P"`, `"PP"` — phosphate state at
  termini.
* `protonate_phosphates=True` (default): adds H to deprotonated
  non-bridging phosphate oxygens so the molecule is formally neutral.
  Set False to keep the deprotonated state; the SIESTA / PySCF
  emitters then auto-set `NetCharge`.

The amber backend post-processes the tleap output via
`_fix_methylene_hydrogens`, which recomputes sp3 -CH2- methylene
hydrogen positions.  tleap's residue library uses canonical
intra-residue geometry which is wrong at C5' between O5' and C4';
the fix touches only -CH2- carbons (2 heavy + 2 H neighbours), never
moves heavy atoms, and never adds or removes atoms.

## SMILES + name builders

```python
build_from_smiles(smiles: str, *, title=None, optimize=True, seed=0xF00D) -> Structure
build_from_name(name: str, *, title=None, **smiles_kwargs) -> Structure
```

* SMILES → 3D via RDKit ETKDGv3 + MMFF94s (UFF fallback when MMFF can't
  parameterise).  Pinned `randomSeed=0xF00D` for reproducibility.
* Name → PubChem lookup (network) → SMILES → SMILES builder.  Network
  call wrapped in a 30-second socket timeout.  Failure to resolve →
  `ValueError`.

## MMFF / UFF fallback rules

`smiles.py:optimize` — when `optimize=True`:

* `MMFFOptimizeMolecule` returns 0 (converged), 1 (max-iter, valid),
  -1 (could not parameterise), or raises.
* If the return is `0` or `1`, the MMFF result is kept (no UFF).
* If the return is `-1` or raises, UFF is run as a fallback.

This is the S3 fix; the original logic ran UFF when MMFF returned 1
(max-iter), which was the opposite of what was needed.

## Backend dispatcher

```python
molbuilder.backends.dispatch(kind, sequence, *, backend, form, terminal, title)
```

* Auto-mode order: `amber` → `rdkit` (helical-aware first, then chemistry-only).
* `BackendUnavailable` is raised explicitly when no installed backend
  can satisfy the request; the user sees install instructions, not
  a cryptic ImportError.
* Adding a backend is two steps: drop `_<name>.py` defining `build()`
  + `is_available()`, register in `_load_backends()` and
  `available_backends()`.

## Sequence-parser contract (`residues.py`)

* 1-letter codes are case-insensitive.
* Whitespace is stripped.
* `[XXX]` opens a 3- or 4-letter PDB code (modified residues OR
  standard residues are both allowed; the brackets just disambiguate).
* Unknown bracketed codes → `ValueError`.
* Unclosed `[` → `ValueError` with position.
* Dashes / parentheses outside brackets → `ValueError` (forces the
  unambiguous syntax).
