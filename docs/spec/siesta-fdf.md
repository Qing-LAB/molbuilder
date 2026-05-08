# Spec — SIESTA `.fdf` emitter

**Module**: `molbuilder/siesta/input.py` (`SiestaConfig` lives at `molbuilder/config/siesta.py`) &nbsp;·&nbsp; **Tests**:
`tests/test_smiles_and_siesta.py`, `tests/test_review_fixes.py`,
`tests/test_pyscf.py` (cross-engine charge handling),
`tests/test_molwatch_preview.py` (sibling `.molwatch.log` output)

## Sibling outputs

Alongside the `<name>.fdf` file, `convert(...)` also writes
`<name>.molwatch.log` by default (`cfg.write_molwatch_log = True`).
That sibling log carries one *initial-state preview block* (step 0)
containing the molecule's coordinates, with no energy / forces /
SCF data, and a `kind: initial_preview` marker line.

Purpose: molwatch can render the structure the moment the user
loads it, before SIESTA has produced any of its own output.  The
preview file is static (one block, never updated); for live updates
during a SIESTA run the user points molwatch at the SIESTA `.out`
file instead.  Both files share the same job stem so they live next
to each other and are easy to find from one another.

Set `cfg.write_molwatch_log = False` to suppress the sibling file.

The format spec for `.molwatch.log` is documented in
`docs/spec/pyscf-script.md` (the format itself is engine-agnostic;
the `# engine:` header line distinguishes who wrote it).


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
8. **Spin** (only when `cfg.spin_polarized=True`): `Spin polarized`
   plus, when `cfg.spin_total is not None`, the two-line
   `Spin.Fix .true.` / `Spin.Total <v>` constraint pair (see "Spin
   contract" below).
9. **NetCharge**: emitted iff resolved charge != 0 (see "Charge
   contract" below).
10. **k-grid**: Monkhorst-Pack mesh from `cfg.kgrid`.
11. **Geometry optimisation / dynamics**: `MD.TypeOfRun`, per-engine
    step-count keyword (`MD.NumCGsteps` for CG, `MD.NumBroydenSteps`
    for Broyden, `MD.NumFIRESteps` for FIRE; `MD.FinalTimeStep` for
    Verlet/Nose dynamics).  Relaxation modes (CG/Broyden/FIRE) also
    emit `MD.MaxForceTol` and the displacement cap (`MD.MaxCGDispl`
    for CG, `MD.MaxDispl` for Broyden/FIRE).  Dynamics modes
    (Verlet/Nose) instead emit `MD.InitialTemperature` and
    `MD.LengthTimeStep`; Nose-Hoover NVT additionally emits
    `MD.TargetTemperature` (defaulting to `md_initial_temperature`
    when `md_target_temperature is None`) — without it SIESTA's
    thermostat target falls back to 0 K and the run quenches
    instead of equilibrating.  All modes optionally emit
    `MD.UseSaveCG` / `MD.UseSaveXV`.  Skipped entirely when
    `cfg.relax_type.lower() == "none"`.
12. **Output flags**: `WriteForces`, `WriteCoorStep`, `WriteCoorXmol`,
    `WriteMDhistory`, optional `WriteHS`.
13. **Troubleshooting block** (when `cfg.verbose_comments=True`):
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

## Spin contract

SIESTA's default is spin-restricted (no `Spin` block emitted →
closed-shell DFT).  Open-shell systems (radicals, transition
metals, triplets) **silently produce wrong electronic structure**
without spin polarisation.

Targeted SIESTA version range: 4.1 -- 5.x.  v5 introduced a
unified single-line `Spin <option>` keyword (recognised options:
`non-polarized`, `polarized`, `non-collinear`, `spin-orbit`) that
supersedes the older multi-line `SpinPolarized true` form.  v4.1+
back-compat-accepts both spellings; the v5 manual marks the older
form deprecated.  The generator emits the v5 form.

The total-spin pin requires TWO lines, not one:

* `Spin.Fix .true.`  enables the constraint.  Without it,
  `Spin.Total` below is silently ignored.
* `Spin.Total <value>`  target total spin moment in mu_B
  (= number of unpaired electrons).

* `cfg.spin_polarized=False` (default): no `Spin` block.
* `cfg.spin_polarized=True`: emit `Spin polarized`.
* `cfg.spin_total` (float, optional): when set together with
  `spin_polarized=True`, emit the
  `Spin.Fix .true.` / `Spin.Total <value>` pair so SIESTA's
  initial guess targets the right multiplicity.  When set with
  `spin_polarized=False`, the value is ignored (no `Spin.Fix` /
  `Spin.Total` lines) — `Spin.Total` is meaningless without
  polarisation.

There is no equivalent of PySCF's "method=RKS validation": SIESTA
can be told `Spin polarized` regardless of basis or method, so
the only correctness rule is that the user MUST set it for any
open-shell system.  Document this loudly in the FDF when the user
passes `--spin-polarized`.

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
