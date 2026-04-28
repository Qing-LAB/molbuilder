# Spec — SIESTA `.fdf` emitter

**Module**: `molbuilder/siesta.py` &nbsp;·&nbsp; **Tests**:
`tests/test_smiles_and_siesta.py`, `tests/test_review_fixes.py`,
`tests/test_pyscf.py` (cross-engine charge handling)

The emitter takes a `Structure` (or an XYZ/PDB file path) and writes
a SIESTA-runnable `.fdf` text.  It also optionally copies matching
`<Element>.psml` files into the FDF's directory.

## Public API

```python
@dataclass class SiestaConfig: ...
Config = SiestaConfig                       # backwards-compat alias

render_fdf(struct, config=None, *, cell=None) -> str
convert(input_path, fdf_path, config=None) -> dict
copy_pseudopotentials(species, lib, dest_dir) -> List[str]    # missing
```

## Output sections (in order)

1. **Header**: `SystemName`, `SystemLabel`, atom + species counts.
2. **Lattice**: 3×3 in Å.  Either user-supplied (`cell=` kwarg) or
   auto-generated as an orthorhombic vacuum box of `extent + 2 *
   cell_padding` per axis with the molecule centred.
3. **Species table**: `%block ChemicalSpeciesLabel` listing each
   unique element with its atomic number, ordered by atomic number.
4. **Atomic coordinates**: `%block AtomicCoordinatesAndAtomicSpecies`
   in Å, one atom per line, last column is the species index.
5. **Basis & grid**: `MeshCutoff`, `PAO.BasisSize`, `PAO.EnergyShift`.
6. **XC**: `XC.functional`, `XC.authors`.
7. **SCF**: `SolutionMethod`, `DM.MixingWeight`, `DM.NumberPulay`,
   `DM.Tolerance`, `DM.Energy.Tolerance`, `MaxSCFIterations`,
   `ElectronicTemperature`, optional `DM.UseSaveDM`.
8. **NetCharge**: emitted iff resolved charge != 0 (see "Charge
   contract" below).
9. **k-grid**: Monkhorst-Pack mesh from `cfg.kgrid`.
10. **Geometry optimisation**: `MD.TypeOfRun`, `MD.NumCGsteps`,
    `MD.MaxForceTol`, `MD.MaxCGDispl`, `MD.UseSaveCG`/`UseSaveXV`.
    Skipped entirely when `cfg.relax_type.lower() == "none"`.
11. **Output flags**: `WriteForces`, `WriteCoorStep`, `WriteCoorXmol`,
    `WriteMDhistory`, optional `WriteHS`.
12. **Troubleshooting block** (when `cfg.verbose_comments=True`):
    inline tuning hints for SCF / forces / speed, plus relaxation
    hints when an MD block is present.

## Verbose comments contract

When `cfg.verbose_comments=True` (default), every numeric parameter
above is preceded by a `# ...` block describing:

* what the parameter controls (one sentence),
* a sensible range,
* what to do when it misbehaves (one or two example tweaks).

Removing or substantially changing one of those comments is a spec
change and triggers a test update.

## Charge contract

Resolved charge is computed once per `render_fdf` call:

* If `cfg.net_charge is not None`: use it as-is (including 0, which
  disables auto-detection).
* Otherwise: `formal_charge_from_phosphates(struct)`.

If the resolved charge is non-zero, an explicit `NetCharge ±N` line
is emitted, with verbose-mode comments explaining the source
("user-specified" or "auto (phosphate protonation)") and what SIESTA
will do with it.

## Cell-padding auto-bump (charged systems)

When `cell is None` (auto-vacuum mode) AND the resolved charge is
non-zero AND `cfg.cell_padding < 25.0`:

* `effective_padding` is silently bumped to 25 Å.
* The `# (auto-generated orthorhombic vacuum cell ...)` comment in
  the FDF says so explicitly: "padding auto-bumped from X → 25 A
  because NetCharge != 0".

Reason: SIESTA's compensating-background-charge correction needs ≥25 Å
between periodic images for image-image Coulomb to drop below ~1
meV.  A neutral molecule doesn't need this.

## PDB serial / chain id width (via Structure.to_pdb)

This is a Structure-side spec; called out here because the FDF
emitter feeds Structure into PSML copy logic.

## `convert(input_path, fdf_path, config)`

* Auto-detects format from extension (`.xyz` or `.pdb`).
* Returns a summary dict: `{"fdf", "n_atoms", "species",
  "missing_psml"}`.
* If `cfg.psml_lib` is set and `cfg.copy_psml=True`: copies matching
  `<Element>.psml` files into `fdf_path`'s directory.  Missing
  pseudopotentials are listed in the summary; the calling CLI exits
  with code 2 in that case.

## Forbidden patterns

The emitter must NOT:

1. Emit the `MD.TypeOfRun` block when `cfg.relax_type == "none"`.
   The `none` value means single-point only; emitting CG would force
   relaxation.
2. Truncate atom-coordinate lines.  All atoms in `Structure` go into
   the `%block AtomicCoordinatesAndAtomicSpecies`.
3. Emit invalid SIESTA syntax for any standard config.  Every
   variant tested in `tests/` must `convert()` end-to-end without
   raising.

## Test reference

* `test_smiles_and_siesta.py` — render_fdf for a built DNA + a
  convert() round-trip via XYZ.
* `test_review_fixes.py` — net_charge override (S2), cell-padding
  auto-bump (D3), Config alias (D1).
