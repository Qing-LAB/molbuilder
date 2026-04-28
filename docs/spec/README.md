# molbuilder — design specification

This directory is the contract for what molbuilder must do.  Every
feature has a focused spec file; every claim a spec makes must be
testable, and the test must reference the spec rather than the
implementation it tests.

## Why this exists

A bug shipped because a code review checked the implementation
against itself: tests asserted "the string `mol = gto.M(...)` appears
in the generated script" rather than "the generated script must not
truncate `<job>.log` between stages."  When the implementation was
wrong, the test was wrong in lock-step.  Specs decouple the two.

The rule going forward:

> **Tests must be derivable from the spec without reading the
> implementation.  Code reviews must verify code matches spec, not
> code matches reviewer's expectations.**

## What molbuilder does

molbuilder builds a 3-D molecular structure from one of:

  * a peptide sequence (1-letter, with `[XXX]` escapes for modified
    residues);
  * a DNA / RNA sequence;
  * a SMILES string;
  * a common chemical name (PubChem lookup);
  * an existing XYZ or PDB file.

It then emits input for downstream computational-chemistry engines:

  * SIESTA `.fdf` text (with optional psml-pseudopotential copying);
  * a runnable PySCF Python script;
  * raw XYZ / PDB / PySCF-atom-block / ASE Atoms.

A Flask web UI wraps the same core, plus a 3Dmol.js viewer.

## Design goals

1. **Predictable**.  The same input produces the same output across
   runs.  Random seeds are pinned where applicable (RDKit ETKDG).

2. **Pluggable backends**.  Adding a new nucleic-acid backend (3DNA,
   future) is a single module + a single registry entry.  Adding a
   new emitter (NWChem, Gaussian, future) is a single module.

3. **Self-documenting outputs**.  Generated FDFs and PySCF scripts
   carry inline comments that explain every parameter and link
   tunable knobs to physical consequences.  A scientist new to either
   engine can read the output as a tutorial.

4. **Observable correctness**.  The package detects molecule charge
   from phosphate protonation state and surfaces it in the FDF /
   PySCF output, even when the user didn't ask.  The user can override.

5. **Cross-platform on standard scientific stacks**.  Hard deps:
   `numpy`, `ase`.  Optional deps: `rdkit`, `PeptideBuilder`,
   `pubchempy`, `flask`.  A user with only the hard deps can do
   everything except browser UI / SMILES / name lookup.

6. **Test invariants from the user's view**.  An invariant about a
   generated PySCF script is checked by parsing the script, not by
   inspecting the generator's call graph.

## Spec index

| spec | covers |
| --- | --- |
| [`structure.md`](structure.md)         | Structure dataclass, XYZ/PDB I/O, format detection |
| [`chemistry.md`](chemistry.md)         | charge auto-detect, phosphate protonation |
| [`builders.md`](builders.md)           | peptide / DNA / RNA / SMILES / name builders |
| [`siesta-fdf.md`](siesta-fdf.md)       | SIESTA `.fdf` emitter |
| [`pyscf-script.md`](pyscf-script.md)   | PySCF runnable-script emitter |
| [`web-api.md`](web-api.md)             | Flask endpoints + UI contract |
| [`cli.md`](cli.md)                     | command-line surface |

## Versioning

This is a 0.x project.  Spec changes that remove or rename promised
output files require a minor version bump (0.x → 0.x+1) AND a
deprecation note in CHANGELOG.  Adding new optional fields / files is
a patch-level change.  We hit 1.0 when the spec is frozen.
