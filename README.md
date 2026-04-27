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
molbuilder serve --port 8000
```

Then open `http://127.0.0.1:8000`:

1.  Pick an input type (peptide / DNA / RNA / SMILES / name) and type a sequence.
2.  Press *Build* — the structure renders in a 3Dmol.js viewer with style controls
    (representation, atom radius, background, optional atom-index labels).
    Download the geometry as `.xyz` or `.pdb`.
3.  Optionally fill in the SIESTA parameter form (basis, mesh cutoff, XC, SCF, k-grid,
    relaxation), click *Generate .fdf*, see the FDF inline and download it.

The web app does **not** bundle pseudopotentials — copy your psml files to the run
directory yourself, or use `molbuilder fdf ... --psml-lib /path/to/lib` on the
command line, which copies matching `<Element>.psml` files next to the `.fdf`.

## Install

```bash
pip install -e .                      # core only (peptide / DNA / RNA / FDF)
pip install -e ".[rdkit]"             # adds H-protonation + SMILES
pip install -e ".[name]"              # adds PubChem name lookup
pip install -e ".[web]"               # adds Flask web UI
pip install -e ".[all]"               # everything
```

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
molbuilder serve --port 8000
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
  structure.py           # Structure dataclass + 4 writers
  residues.py            # 1-letter parser + bracket escapes + modified residues
  peptide.py             # PeptideBuilder wrapper + auto-protonation
  templates.py           # embedded nucleotide templates
  nucleic.py             # B-form / A-form helical assembler
  smiles.py              # RDKit-based build_from_smiles
  pubchem.py             # PubChem-based build_from_name
  siesta.py              # XYZ -> .fdf converter + psml copy
  cli.py                 # `molbuilder <subcommand>`
  web/
    __init__.py
    app.py               # Flask app: /api/build, /api/fdf, /
    templates/index.html
    static/viewer.js
    static/style.css
tests/
  test_residues.py       test_structure.py
  test_peptide.py        test_nucleic.py
  test_smiles_and_siesta.py
  test_web.py            # end-to-end Flask test
```

## Nucleic-acid backends

DNA / RNA building goes through one of two pluggable backends.  Pick
the right trade-off for your use case:

| Backend | Install | Shape | Notes |
|---|---|---|---|
| `rdkit` | already a dep | folded conformer | Chemistry / connectivity correct (proper backbone, all hydrogens). 3-D shape is whatever ETKDG embeds and UFF cleans up -- a folded clump for anything > ~6mer, *not* a B-form helix. Fine for short oligos that DFT will fully relax. |
| `amber` | `conda install -c conda-forge ambertools` (~1.5 GB) | extended chain | Drives AmberTools `tleap` with a `sequence { ... }` macro. Backbone topology follows the Amber OL15 (DNA) / OL3 (RNA) force field; chain comes out extended (not pre-coiled). AmberTools 23+ removed the original `nab` fiber builder, so `tleap` is the closest in-AmberTools alternative. For a *true* B-form / A-form helical starting geometry you'll want 3DNA's `fiber` instead. |

`backend="auto"` (default) tries `amber` first and falls back to
`rdkit` if `tleap` isn't on PATH.

## Limits / TODOs (v0.2)

* **Single-stranded DNA / RNA only.** Double helices need a complementary
  strand placed on a Watson-Crick offset; straightforward addition.
* **No canonical-helix backend bundled.** Both shipped backends produce
  chemically correct but non-helical 3-D structures. If you need a
  proper B/A/Z-form starting geometry, install 3DNA externally and use
  its `fiber` command (a 3DNA wrapper backend would slot cleanly into
  `molbuilder/backends/`).
* **No bond detection / connectivity output.** PDB records are
  ATOM-only; downstream tools that need CONECT must derive bonds from
  distances or from a force field.
* **Web app does not ship pseudopotentials.** Use the CLI
  `molbuilder fdf --psml-lib ...` for that.
