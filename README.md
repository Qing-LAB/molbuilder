# molbuilder

A toolkit for the full molecular-electronics workflow:

```
sequence ──► Structure ──► (modify) ──► SIESTA .fdf  ──► siesta ─┐
                                    └─► PySCF .py    ──► python ─┴──► .molwatch.log
                                                                              │
                                                            ◄──── live watch ─┘
```

molbuilder builds 3-D molecular structures from sequences / SMILES /
names, **modifies** them into derived geometries (e.g. metal-molecule-
metal nanojunctions), generates SIESTA / PySCF input files, and provides
a live trajectory viewer that monitors the resulting calculations.

These are **starting structures for a geometry optimisation** — not
equilibrium geometries.  Always relax in your DFT/MP2 code before
computing properties.

The full design and decisions reference is `docs/design.md`.  Per-feature
specs live under `docs/spec/`.

---

## Install

```bash
pip install -e .                      # core: peptide/DNA/RNA/FDF/PySCF/web
pip install -e ".[rdkit]"             # adds H-protonation + SMILES
pip install -e ".[name]"              # adds PubChem name lookup
pip install -e ".[all]"               # everything
```

Flask is in core dependencies (the build / modify / watch web UI is
part of the toolkit, not an opt-in extra).

Conda alternative for the heaviest dep:

```bash
conda install -c conda-forge rdkit ase
pip install PeptideBuilder pubchempy flask
```

For the canonical-helix DNA backend (3DNA), see
`docs/design.md` § "3DNA (canonical helix builder)" — the archive is not
bundled and must be obtained from http://x3dna.org/ under the upstream's
non-commercial registration.

---

## Python API at a glance

```python
import molbuilder

# --- biological sequences ---------------------------------------------
s = molbuilder.build_peptide("ARNDC")                       # 1-letter
s = molbuilder.build_peptide("AR[SEP]C")                    # phospho-Ser
s = molbuilder.build_dna("ATGCATGCAT")                      # auto-pick backend
s = molbuilder.build_dna("ATGCATGCAT", backend="threedna",
                         form="B", terminal="OH")           # canonical B-DNA
s = molbuilder.build_rna("AUGCAUGCAU", backend="rdkit")     # chemistry-only

# --- chemistry --------------------------------------------------------
s = molbuilder.build_from_smiles("Sc1ccc(S)cc1")     # 1,4-benzenedithiol
s = molbuilder.build_from_name("benzene")            # PubChem -> SMILES -> 3D

# --- modify -----------------------------------------------------------
from molbuilder.modify import (
    delete_atoms, add_atom, orient_along_axis, rotate_around_axis,
    add_electrode_slab, add_symmetric_electrodes,
)

# Strip thiol H caps, orient the S–S axis along z, centre, then add
# a Au(111) 3×3×2 pair with 12 Å gap centred on the world origin.
s = delete_atoms(s, [12, 13])                         # H caps
s = orient_along_axis(s, anchor_indices=(10, 11),     # S-S anchors
                      axis="z", center="midpoint")
s = s.centered()                                      # centroid at origin
s = add_symmetric_electrodes(
    s, element="Au", plane="111", size=(3, 3, 2),
    gap=12.0,                                         # anchorless mode
)

# --- output -----------------------------------------------------------
s.to_xyz("out.xyz")               # SIESTA-ready
s.to_pdb("out.pdb")               # opens in PyMOL/VMD/Chimera
print(s.to_pyscf(as_string=True)) # paste into pyscf.gto.M(atom=...)
atoms = s.to_ase()                # ase.Atoms instance

# --- SIESTA / PySCF input --------------------------------------------
from molbuilder.siesta import SiestaConfig, render_fdf, convert
print(render_fdf(s, SiestaConfig(system_label="bdt", kgrid=(1, 1, 1))))

from molbuilder.pyscf import PySCFConfig, render_script
print(render_script(s, PySCFConfig(method="UKS", xc="B3LYP")))
```

---

## Browser UI

```bash
molbuilder serve --port 8000
```

Single Flask app, three tabs sharing the 3Dmol viewer + style controls:

* **Build** at `/` — pick an input type (peptide / DNA / RNA / SMILES /
  name), type a sequence, render in the viewer, fill in SIESTA / PySCF
  parameters, generate .fdf or .py.
* **Modify** at `/modify` — load an existing `.xyz` / `.pdb`, edit
  atoms, build metal-molecule-metal junctions, hand off to Build.
* **Watch** at `/watch` — point at a running SIESTA / PySCF job's
  output file, render geometry frame-by-frame, plot energy / forces /
  per-cycle SCF convergence; auto-refreshes on file `mtime` change.

`molbuilder watch serve` starts the same app but lands on the Watch
tab; `molbuilder serve` lands on Build.

The web app does **not** bundle pseudopotentials — copy your psml files
to the run directory yourself, or use `molbuilder fdf ... --psml-lib
/path/to/lib` on the command line, which copies matching `<Element>.psml`
files next to the `.fdf`.

### Modify tab — how it works

The Modify tab assembles metal-molecule-metal nanojunctions by editing
an existing structure.  Four sub-tabs in the right-hand Edit panel mirror
the workflow:

1. **Atom** — delete selected atoms, add a new atom with `(dx, dy, dz)`
   offset and a live distance readout.
2. **Pose** — orient an anchor pair onto the z-axis (with an optional
   tilt slider for non-zero angle), or rotate the whole structure
   around x/y/z by an explicit angle.
3. **Geom** — centre the structure at the origin (translates so the
   atom-coordinate centroid lands on (0, 0, 0)), or apply an explicit
   `(Δx, Δy, Δz)` translation.
4. **Junction** — add electrode slabs.  Two modes:
   * **Anchorless (recommended)**: with **no** atom selection, slabs are
     placed at z = ±gap/2 around the world origin, both lateral-centred
     on the (offset) origin.  Default workflow: centre + pose the
     molecule first, then add slabs.
   * **Legacy anchored**: with **two** atoms selected, slabs are placed
     so their midpoint lands on the anchor-pair midpoint.  Useful for
     non-pre-centred structures.

Camera affordances:

* **Focus molecule** (viewer toolbar) — anchors the rotation/zoom pivot
  on the molecule (excludes electrode slabs) and tightens the camera
  with a 0.55× pull-back so slabs remain visible in the periphery.
  Click whenever wheel-zoom feels off-centre.
* **Pivot snap on rotation drag** — every plain left-button drag
  re-anchors the camera lookAt onto the structure centroid (after a
  4-px movement threshold so atom-pick clicks don't trigger it).

The Junction subtab carries a slab-only **Undo** (depth 20).  Other
ops (delete / rotate / translate / centre) are committed immediately;
to roll those back, re-load the source XYZ.

When the junction is ready, **Send to Build tab** writes the structure
to `sessionStorage` and navigates to `/`, where the Build tab picks it
up identically to a fresh build.

The full Modify spec lives at `docs/spec/modify-tab.md`.

### Staged SIESTA relaxation (coarse → medium → tight)

Production geometry optimisation usually doesn't run with tight
convergence from the start.  The faster route is to **stage** the
calculation: a coarse run (loose tolerances, big steps) gets the
structure into the energy basin; a medium run refines it; a tight
final run produces the publication-grade geometry.  Each stage
**continues** the previous one — SIESTA reads `<label>.XV` for
coordinates and `<label>.DM` for the SCF starting guess from the
previous run, so the early relaxation work is never repeated.

The Build tab's SIESTA panel has a **Relaxation stage** select that
bulk-fills the right convergence parameters for each stage (Coarse /
Medium / Tight); the **Watch tab** carries the full workflow guide
in an embedded "Staged relaxation workflow" panel — open it on
`/watch` for the recipe table, the restart-file contract, and a
do/don't list.

The mechanical contract: keep `SystemLabel` **identical** across all
stages and run them all in the **same directory**.  molbuilder's
generated FDF emits `DM.UseSaveDM`, `MD.UseSaveXV`, `MD.UseSaveCG`
all `.true.` by default, so the continuation works for free.

The full naming protocol — which files share the basename, which
ones the Watch tab discovers, what's safe vs. what breaks the
restart — lives in [`docs/spec/job-layout.md`](docs/spec/job-layout.md).
When you point the Watch tab at a **run directory** (not a specific
file), it walks that protocol's discovery chain to pick the right
log automatically.

### Watch tab — how it works

Auto-detected formats:

* `<job>.molwatch.log` — the unified format molbuilder emits;
  preferred (one file carries trajectory + per-cycle SCF data + an
  initial-geometry preview).
* SIESTA stdout (`run.out` / `siesta.log`) — fallback.
* geomeTRIC's `<job>_geom_optim.xyz` — fallback for PySCF runs.

The page re-parses on file `mtime` change (~15 s polling), so a
still-running calculation streams new frames into the open tab.  The
parser is truncation-tolerant: a half-written final block in a still-
running job is dropped on parse and picked up on the next refresh.

The Watch tab is **read-only**: it does not start, monitor, or kill
the engine process; it only reads files the engine has produced.

---

## CLI

```bash
# Build
molbuilder peptide ARNDC --out peptide.xyz
molbuilder peptide "AR[SEP]C" --pdb pep.pdb
molbuilder dna ATGCATGCAT --out dna.xyz --backend threedna
molbuilder rna AUGC --pyscf
molbuilder smiles "c1ccccc1" --out benzene.xyz
molbuilder name "1,4-benzenedithiol" --pdb bdt.pdb

# Generate engine input from a structure file
molbuilder fdf in.xyz out.fdf --psml-lib /opt/psml --kgrid 4x4x1 --mesh-cutoff 400
molbuilder pyscf in.xyz out.py --functional B3LYP --preopt

# Validate a structure (and optionally a config) — JSON Issue list to stdout
molbuilder validate in.xyz --config siesta.fdf

# Edit a structure: one operation TYPE per call, chain via stdin/stdout
molbuilder modify bdt.xyz - --orient-axis 10,11 --center midpoint |
  molbuilder modify - junction.xyz \
      --electrode "Au:111:3x3x2@gap=12.0:3,0"

# Live watch (parse / tail / serve)
molbuilder watch parse run.molwatch.log     # JSON dump to stdout
molbuilder watch tail  run.molwatch.log     # NDJSON, one frame per line
molbuilder watch serve --port 8000          # web UI on /watch
molbuilder serve --port 8000                # same app, lands on /
```

All CLI subcommands accept `-` as an input or output path for stdin /
stdout piping.  Machine-consumable subcommands (`validate`, `watch
parse`, `watch tail`, anything with `--json-summary`) emit JSON on
stdout; warnings and progress always go to stderr, so they don't
pollute the pipe.

### Sequence syntax

Tiny grammar for peptide / DNA / RNA build subcommands:

```
sequence  = (oneletter | bracketed | whitespace)*
oneletter = a single ASCII letter, case-insensitive
bracketed = "[" 3-or-4-letter PDB / modified-residue code "]"
```

| Input            | Meaning                                        |
|------------------|------------------------------------------------|
| `ARNDC`          | Ala-Arg-Asn-Asp-Cys                            |
| `arndc`          | same (case-insensitive)                        |
| `A R N D C`      | same (whitespace ignored)                      |
| `AR[SEP]C`       | Ala-Arg-phosphoSer-Cys                         |
| `ATGC`           | DA-DT-DG-DC (DNA)                              |
| `AUGC`           | A-U-G-C (RNA)                                  |
| `5'-ATGC-3'`     | DNA with explicit 5'/3' labels (same as bare)  |
| `3'-CGTA-5'`     | reverse-direction; parser flips before build   |

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

---

## Output formats

| Method | Use |
|---|---|
| `Structure.to_xyz(path=None, comment="")` | XMol .xyz; returns the string and writes to `path` if given. |
| `Structure.to_pdb(path=None)`              | standard PDB ATOM records with full residue / chain / atom-name metadata. |
| `Structure.to_pyscf(as_string=False)`      | list of `(symbol, (x, y, z))` tuples for `pyscf.gto.M(atom=...)`; pass `as_string=True` for the multi-line form. |
| `Structure.to_ase()`                       | `ase.Atoms` instance. |

Generated SIESTA `.fdf` and PySCF `.py` files are **self-contained**
and **tunable by manual editing**.  Section headers, inline tuning hints,
visible defaults with allowed ranges, and commented-out post-processing
hook templates (Mulliken population, BandLines, PDOS for SIESTA;
`mulliken_pop`, `dip_moment`, `mf.analyze()` for PySCF) are part of the
output contract — you can `scp` the file to a cluster that has only
`siesta` or `pyscf + geometric` and run it as-is.

---

## Architecture (overview)

Three layers, four core types — see `docs/design.md` for the full
explanation.

```
L3 — Surfaces  : cli.py (click), web/ (Flask + Blueprints)
L2 — Verbs     : builders/, generators/, parsers/, modify, validation
L1 — Types     : Structure, Frame, Trajectory, SiestaConfig, PySCFConfig, Issue
```

Higher layers may import lower; lower layers must never import higher.
Field metadata (label / unit / range / validator) lives on the
dataclass field, **not** in parallel CLI / web registries — the click
options and the web form schema are both generated from
`dataclasses.fields(Config)`.

### Project layout

```
molbuilder/
  __init__.py            # public API
  structure.py           # Structure dataclass + readers / writers
  frame.py               # Frame + Trajectory (parser output types)
  issues.py              # Issue + ValidationError
  validation.py          # pre-emission validation pass
  modify.py              # delete / add / orient / rotate / electrode ops
  chemistry.py           # element table, charge / dipole helpers
  residues.py            # 1-letter parser + bracket escapes
  peptide.py             # PeptideBuilder wrapper + auto-protonation
  nucleic.py             # DNA / RNA polymer builder
  smiles.py              # RDKit-based build_from_smiles
  pubchem.py             # PubChem-based build_from_name
  config/
    siesta.py            # SiestaConfig
    pyscf.py             # PySCFConfig
  siesta/
    input.py             # render_fdf + convert (re-exports SiestaConfig)
  pyscf/
    input.py             # render_script + convert (re-exports PySCFConfig)
  trajectory_log/
    format.py            # writer for .molwatch.log v1
    emitter.py           # _MolwatchEmitter (inlined into generated PySCF)
                         # (molbuilder/molwatch_log/ is a back-compat shim)
  parsers/
    base.py              # TrajectoryParser ABC; parse() -> Trajectory
    molwatch_log.py      # parser for the unified .molwatch.log
    siesta.py            # parser for SIESTA stdout
    pyscf.py             # parser for geomeTRIC _optim.xyz + .qdata + .log
  builders/
    backends/
      _amber.py          # tleap-driven (extended chain)
      _rdkit.py          # ETKDG embedded conformer (folded for >6mers)
      _threedna.py       # 3DNA fiber-driven canonical helix
      _common.py
    # (molbuilder/backends/ is a back-compat shim re-exporting the above)
  data/
    README.md            # citations for every numeric value below
    fcc_lattice.json     # supported FCC metals (closed list)
  cli.py                 # `molbuilder <subcommand>`
  web/
    __init__.py
    app.py               # Flask app + Blueprint registration
    blueprints/
      build.py           # /api/build/* routes
      watch.py           # /api/watch/* routes
      modify.py          # /api/modify/* routes
      _shared.py         # body parsing, response helpers
    templates/
      _app_header.html   # shared header + tab nav
      index.html         # build page
      modify.html        # modify page
      watch.html         # watch page
    static/
      lib/               # shared assets (tokens.css, tabs.css, mol-style.js, mol-format.js)
      viewer.js          # build viewer
      style.css
      modify/{viewer.js, style.css}
      watch/{viewer.js, style.css}

tests/
  test_residues.py       test_structure.py    test_frame.py
  test_peptide.py        test_nucleic.py      test_chemistry.py
  test_smiles_and_siesta.py                   test_pyscf.py
  test_pyscf_spec.py     test_molwatch_preview.py
  test_load.py           test_pdb_ter.py      test_review_fixes.py
  test_output_correctness.py
  test_validation.py     test_backends.py
  test_cli.py            test_pubchem.py      test_science_gaps.py
  test_modify.py         test_modify_e2e.py   # Python API + Playwright E2E
  test_web.py                                 # build / modify Flask
  watch/                                       # watch-side parser + Flask
    test_registry.py     test_molwatch_log_parser.py
    test_siesta_parser.py  test_pyscf_parser.py
    test_api_load.py     test_app_concurrency.py
```

---

## Nucleic-acid backends

DNA / RNA building goes through one of three pluggable backends:

| Backend | Install | Shape | Notes |
|---|---|---|---|
| `threedna` | download from http://x3dna.org/ (registration + non-commercial license; molbuilder cannot fetch) | canonical B/A/Z helix | Drives 3DNA's `fiber` for true B-form / A-form / Z-form DNA and A-form RNA helices.  Detection chain: unpack the tarball at the molbuilder repo root (gitignored) and the backend lights up automatically; otherwise `$X3DNA` env var, otherwise `fiber` on PATH.  Heavy-atom output (no hydrogens — protonate post-build for DFT). |
| `amber` | `conda install -c conda-forge ambertools` (~1.5 GB) | extended chain | Drives AmberTools `tleap` with a `sequence { ... }` macro.  Backbone topology follows the Amber OL15 (DNA) / OL3 (RNA) force field; chain comes out extended (not pre-coiled). |
| `rdkit` | already a dep | folded conformer | Chemistry / connectivity correct (proper backbone, all hydrogens).  3-D shape is whatever ETKDG embeds and UFF cleans up — a folded clump for anything > ~6mer, *not* a B-form helix.  Fine for short oligos that DFT will fully relax. |

`backend="auto"` (default) tries `threedna` first, falls back to
`amber`, then to `rdkit`.  3DNA's restrictive license is why molbuilder
doesn't bundle or auto-fetch it; see `docs/design.md` § "3DNA
(canonical helix builder)" for the full install + license contract.

---

## Limits / known issues

* **Single-stranded DNA / RNA only.** Double helices need a complementary
  strand placed on a Watson-Crick offset; straightforward addition,
  not yet wired.
* **3DNA helix backend is heavy-atom-only.**  When the `threedna`
  backend is reachable it produces canonical B/A/Z helices, but
  `fiber`'s PDB output has no hydrogens.  `chemistry.add_hydrogens`
  is automatically run afterwards (OpenBabel preferred, RDKit
  fallback) so the result is simulation-ready.
* **No bond detection / connectivity output.**  PDB records are
  ATOM-only; downstream tools that need CONECT must derive bonds from
  distances or from a force field.
* **Web app does not ship pseudopotentials.**  Use the CLI's
  `molbuilder fdf --psml-lib /path/to/lib` to copy the matching
  `<Element>.psml` files next to a generated `.fdf`.
* **Watch is single-tenant.**  One user, one tab, one calculation —
  the server holds one global state under a lock.  For multi-user
  use, run a separate process per user.

---

## Running the tests

```bash
pytest tests/ -q                          # full suite (~2 min with E2E)
pytest tests/test_modify.py -q            # Python modify ops only
pytest tests/test_modify_e2e.py -q        # Playwright + live Flask
pytest tests/watch/ -q                    # parser + watch-app tests
```

The Playwright E2E suite (`test_modify_e2e.py`) starts a live
Werkzeug server on a random port and drives Chromium through the
Modify tab.  Install the browser binary once with
`playwright install chromium` after `pip install pytest-playwright`.

---

Pointers:

* `docs/design.md` — durable design and architectural reference.
* `docs/spec/` — per-component test contracts (siesta-fdf, pyscf-script,
  modify-tab, watch-ui, watch-api, parsers, structure, builders,
  chemistry, web-api, cli).
* `molbuilder/data/README.md` — citations for every numeric value the
  package reads as input (FCC lattice constants, future bond-length
  tables, etc.).
