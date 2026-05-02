# molbuilder

Build 3-D molecular structures from sequences / SMILES / names, view
them in the browser, and turn them into SIESTA / PySCF / ASE input.

```python
import molbuilder

# --- biological sequences ------------------------------------------
s = molbuilder.build_peptide("ARNDC")                       # 1-letter
s = molbuilder.build_peptide("AR[SEP]C")                    # phospho-Ser
s = molbuilder.build_dna("ATGCATGCAT")                      # auto-pick backend
s = molbuilder.build_dna("ATGCATGCAT", backend="amber",
                         form="B", terminal="OH")           # canonical B-DNA
s = molbuilder.build_rna("AUGCAUGCAU", backend="rdkit")     # chemistry-only

# --- chemistry --------------------------------------------------------
s = molbuilder.build_from_smiles("Sc1ccc(S)cc1")     # 1,4-benzenedithiol
s = molbuilder.build_from_name("benzene")            # PubChem -> SMILES -> 3D

# --- output ----------------------------------------------------------
s.to_xyz("out.xyz")               # SIESTA-ready
s.to_pdb("out.pdb")               # opens in PyMOL/VMD/Chimera
print(s.to_pyscf(as_string=True)) # paste into pyscf.gto.M(atom=...)
atoms = s.to_ase()                # ase.Atoms instance

# --- SIESTA input ---------------------------------------------------
from molbuilder.siesta import Config, render_fdf, convert
print(render_fdf(s, Config(system_label="bdt", kgrid=(1, 1, 1))))
# or:  convert("in.xyz", "out.fdf", Config(psml_lib="/opt/psml-lib"))
```

These are **starting structures for a geometry optimisation** -- not
equilibrium geometries.  Always relax in your DFT/MP2 code before
computing properties.

## Browser UI

```bash
molbuilder serve --port 8000          # build page at /
molbuilder watch serve --port 8000    # same app; watch page at /watch
```

Both subcommands start the same Flask server.  The unified UI has
two halves:

**Build** at `http://127.0.0.1:8000/`

1.  Pick an input type (peptide / DNA / RNA / SMILES / name) and type a sequence.
2.  Press *Build* — the structure renders in a 3Dmol.js viewer with style controls
    (representation, atom radius, background, optional atom-index labels).
    Download the geometry as `.xyz` or `.pdb`.
3.  Optionally fill in the SIESTA parameter form (basis, mesh cutoff, XC, SCF, k-grid,
    relaxation), click *Generate .fdf*, see the FDF inline and download it.

**Watch** at `http://127.0.0.1:8000/watch`

Live trajectory viewer for SIESTA / PySCF runs that are still
producing output.  Point it at the engine's output file (or upload
a finished one) and the page renders:

- the molecular geometry, frame-by-frame, in a 3Dmol viewer
  (slider / play / pause / step controls);
- total energy and max-force vs step (Plotly);
- per-cycle SCF convergence (energy + residual norm) for the
  active step.

Auto-detected formats: `<job>.molwatch.log` (the unified log
molbuilder emits — preferred), SIESTA stdout (`run.out` /
`siesta.log`), or geomeTRIC's `<job>_geom_optim.xyz`.  The page
re-parses on file mtime change, so a still-running calculation
streams new frames into the open tab.

The web app does **not** bundle pseudopotentials — copy your psml files to the run
directory yourself, or use `molbuilder fdf ... --psml-lib /path/to/lib` on the
command line, which copies matching `<Element>.psml` files next to the `.fdf`.

## Install

```bash
pip install -e .                      # core: peptide/DNA/RNA/FDF/PySCF/web
pip install -e ".[rdkit]"             # adds H-protonation + SMILES
pip install -e ".[name]"              # adds PubChem name lookup
pip install -e ".[all]"               # everything
```

`flask` is in core dependencies (the build + watch web UI is part of
the toolkit, not a build-time extra).  The `[web]` extra is kept as a
no-op for back-compat.

Conda alternative for the heaviest dep:

```bash
conda install -c conda-forge rdkit ase
pip install PeptideBuilder pubchempy flask
```

## CLI

```bash
molbuilder peptide ARNDC --out peptide.xyz
molbuilder peptide "AR[SEP]C" --pdb pep.pdb
molbuilder dna ATGCATGCAT --out dna.xyz
molbuilder rna AUGC --pyscf

molbuilder smiles "c1ccccc1" --out benzene.xyz
molbuilder name "1,4-benzenedithiol" --pdb bdt.pdb

molbuilder fdf in.xyz out.fdf --psml-lib /opt/psml --kgrid 4x4x1 --mesh-cutoff 400
molbuilder pyscf in.xyz out.py --functional B3LYP --preopt
molbuilder serve --port 8000             # build + watch UI
molbuilder watch serve --port 8000       # same; lands on the watch page
```

## Sequence syntax

Tiny grammar:

```
sequence  = (oneletter | bracketed | whitespace)*
oneletter = a single ASCII letter, case-insensitive
bracketed = "[" 3-or-4-letter PDB / modified-residue code "]"
```

Examples:

| Input            | Meaning                                        |
|------------------|------------------------------------------------|
| `ARNDC`          | Ala-Arg-Asn-Asp-Cys                            |
| `arndc`          | same (case-insensitive)                        |
| `A R N D C`      | same (whitespace ignored)                      |
| `AR[SEP]C`       | Ala-Arg-phosphoSer-Cys                         |
| `ATGC`           | DA-DT-DG-DC (DNA)                              |
| `AUGC`           | A-U-G-C (RNA)                                  |

Modified residues currently supported (extend in
`molbuilder/residues.py:MODIFIED_RESIDUES`):

| Code  | Name                       |
|-------|----------------------------|
| SEP   | phosphoserine              |
| TPO   | phosphothreonine           |
| PTR   | phosphotyrosine            |
| MLY   | N-methyl-lysine            |
| M3L   | N,N,N-trimethyl-lysine     |
| ALY   | N6-acetyl-lysine           |

## Output formats in detail

`Structure.to_xyz(path=None, comment="")` – XMol .xyz; returns the string and writes to `path` if given.

`Structure.to_pdb(path=None)` – standard PDB ATOM records with full residue / chain / atom-name metadata.

`Structure.to_pyscf(as_string=False)` – list of `(symbol, (x, y, z))` tuples for `pyscf.gto.M(atom=...)`; pass `as_string=True` for the multi-line form.

`Structure.to_ase()` – `ase.Atoms` instance.

## Project layout

```
molbuilder/
  __init__.py            # public API
  structure.py           # Structure dataclass + readers / writers
  frame.py               # Frame + Trajectory (parser output types)
  issues.py              # Issue + ValidationError
  validation.py          # pre-emission validation pass
  chemistry.py           # element table, charge / dipole helpers
  residues.py            # 1-letter parser + bracket escapes + modified residues
  peptide.py             # PeptideBuilder wrapper + auto-protonation
  nucleic.py             # DNA / RNA polymer builder
  smiles.py              # RDKit-based build_from_smiles
  pubchem.py             # PubChem-based build_from_name
  siesta/
    input.py             # SiestaConfig + render_fdf + convert
  pyscf/
    input.py             # PySCFConfig + render_script + convert
  molwatch_log/
    format.py            # writer for .molwatch.log v1
  parsers/
    base.py              # TrajectoryParser ABC; parse() -> Trajectory
    molwatch_log.py      # parser for the unified .molwatch.log
    siesta.py            # parser for SIESTA stdout
    pyscf.py             # parser for geomeTRIC _optim.xyz + .qdata + .log
  backends/
    _amber.py            # tleap-driven (extended chain)
    _rdkit.py            # ETKDG embedded conformer (folded for >6mers)
    _threedna.py         # 3DNA fiber-driven canonical helix
    _common.py
  cli.py                 # `molbuilder <subcommand>`
  web/
    __init__.py
    app.py               # build routes (/, /api/build, /api/fdf, ...)
    blueprints/
      watch.py           # watch routes (/watch, /api/watch/*)
    templates/
      index.html         # build page
      watch.html         # watch page
    static/
      viewer.js          # build viewer
      style.css
      watch/             # watch viewer assets
        viewer.js
        style.css
tests/
  test_residues.py       test_structure.py    test_frame.py
  test_peptide.py        test_nucleic.py      test_chemistry.py
  test_smiles_and_siesta.py                   test_pyscf.py
  test_pyscf_spec.py     test_molwatch_preview.py
  test_load.py           test_pdb_ter.py      test_review_fixes.py
  test_output_correctness.py
  test_validation.py     test_backends.py
  test_cli.py            test_pubchem.py      test_science_gaps.py
  test_web.py            # build-side Flask
  watch/                 # watch-side parser + Flask
    test_registry.py     test_molwatch_log_parser.py
    test_siesta_parser.py  test_pyscf_parser.py
    test_api_load.py     test_app_concurrency.py
```

## Nucleic-acid backends

DNA / RNA building goes through one of three pluggable backends.
Pick the right trade-off for your use case:

| Backend | Install | Shape | Notes |
|---|---|---|---|
| `threedna` | download from http://x3dna.org/ (registration + non-commercial license; molbuilder cannot fetch) | canonical B/A/Z helix | Drives 3DNA's `fiber` for true B-form / A-form / Z-form DNA and A-form RNA helices.  Detection chain: unpack the tarball at the molbuilder repo root (gitignored) and the backend lights up automatically; otherwise `$X3DNA` env var, otherwise `fiber` on PATH.  Heavy-atom output (no hydrogens — protonate post-build for DFT). |
| `amber` | `conda install -c conda-forge ambertools` (~1.5 GB) | extended chain | Drives AmberTools `tleap` with a `sequence { ... }` macro. Backbone topology follows the Amber OL15 (DNA) / OL3 (RNA) force field; chain comes out extended (not pre-coiled). AmberTools 23+ removed the original `nab` fiber builder, so `tleap` is the closest in-AmberTools alternative. |
| `rdkit` | already a dep | folded conformer | Chemistry / connectivity correct (proper backbone, all hydrogens). 3-D shape is whatever ETKDG embeds and UFF cleans up -- a folded clump for anything > ~6mer, *not* a B-form helix. Fine for short oligos that DFT will fully relax. |

`backend="auto"` (default) tries `threedna` first, falls back to
`amber`, then to `rdkit`.  3DNA's restrictive license is why molbuilder
doesn't bundle or auto-fetch it; see `docs/design.md` § "3DNA
(canonical helix builder)" for the full install + license contract.

## Limits / TODOs (v0.2)

* **Single-stranded DNA / RNA only.** Double helices need a complementary
  strand placed on a Watson-Crick offset; straightforward addition.
* **3DNA helix backend is heavy-atom-only.**  When the `threedna`
  backend is reachable it produces canonical B/A/Z helices, but
  `fiber`'s PDB output has no hydrogens.  Run the result through a
  protonation step (OpenBabel or RDKit) before feeding it to DFT.
* **No bond detection / connectivity output.** PDB records are
  ATOM-only; downstream tools that need CONECT must derive bonds from
  distances or from a force field.
* **Web app does not ship pseudopotentials.** Use the CLI
  `molbuilder fdf --psml-lib ...` for that.
