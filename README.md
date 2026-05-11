# molbuilder

> Assemble **metal–molecule–metal nanojunctions** for transport-DFT
> simulations.  Build → modify → simulate → watch, all in one
> toolkit.

molbuilder is built around the **molecular-electronics workflow**:
constructing the geometry of a single-molecule junction sandwiched
between two metal electrodes (Au–thiol–Au is the canonical example),
generating **SIESTA** or **PySCF** input for that geometry, and
watching the resulting DFT optimisation live in a browser.

```
sequence ──► Structure ──► (modify) ──► SIESTA .fdf  ──► siesta ──┐
                                    └─► PySCF .py    ──► python  ─┴──► .molwatch.log
                                                                              │
                                                            ◄──── live watch ─┘
```

The **Modify** tab is the headline feature.  It takes a relaxed
molecule (built here or loaded from anywhere), lets you orient its
anchor atoms onto the z-axis, and adds crystallographic FCC
electrode slabs at a chosen gap — giving you a transport-ready
geometry in a few clicks.  The full pipeline (build / modify /
generate / watch) is what differentiates molbuilder from a
general-purpose builder: every step knows about the next.

These are **starting structures for a geometry optimisation** — not
equilibrium geometries.  Always relax in your DFT/MP2 code before
computing properties.

---

## Quick start

```bash
pip install -e .
molbuilder serve --port 8000     # opens build / modify / watch UI
# browser: http://localhost:8000/
```

Three tabs in one Flask app — **Build** generates input files,
**Modify** assembles junctions, **Watch** monitors a running
calculation:

| Tab | URL | What it does |
|---|---|---|
| **Build**  | `/`         | sequence → structure → SIESTA `.fdf` / PySCF `.py` |
| **Modify** | `/modify`   | edit atoms, build metal-molecule-metal junctions, hand off to Build |
| **Watch**  | `/watch`    | scrub frames, track energy / forces / SCF convergence live |

---

## The Build tab

![Build tab — type ARNDC, click Build, see the peptide rendered; SIESTA form below sets generation params](docs/img/build-tab.png)

Build a structure from any of:

* **Peptide** sequence (1-letter or PDB 3-letter) — `ARNDC`,
  `AR[SEP]C` (phospho-Ser), …
* **DNA / RNA** sequence — `ATGCATGCAT`, `AUGCAUGCAU`; the
  `threedna` backend (when 3DNA is installed) gives a canonical
  B/A/Z helix, otherwise `amber` (extended chain) or `rdkit`
  (folded conformer).
* **SMILES** — `Sc1ccc(S)cc1` (1,4-benzenedithiol)
* **Compound name** — `aspirin` (PubChem lookup → SMILES → 3D).

Click **Build**, see the 3Dmol viewer render the molecule, then fill
in the SIESTA or PySCF parameter form and click **Generate**.  Every
field has an inline tooltip with a recommended range; the generated
file carries verbose tuning hints next to each parameter so it's
readable as a tutorial.

---

## The Modify tab — assemble a nanojunction

![Modify tab — water loaded, atom list on the left, viewer in the middle, Edit panel on the right with 4 sub-tabs](docs/img/modify-tab.png)

The Modify tab assembles a **metal–molecule–metal nanojunction** from
an existing molecule plus a couple of clicks.  The canonical workflow
takes a thiol-anchored molecule (1,4-benzenedithiol, an alkanedithiol,
an oligophenyl, …) and produces a Au–S–molecule–S–Au geometry ready
for SIESTA / TranSIESTA transport calculations.

### Canonical Au–thiol–Au workflow

1. **Load** the relaxed molecule (`.xyz` or `.pdb`).
2. **Atom subtab** — click the two thiol hydrogens, hit **Delete**
   to expose the S atoms.
3. **Pose subtab** — select the two S atoms (Shift-click), pick
   target axis **z** with `center = midpoint`, hit **Apply orient**.
   The S–S vector now lies along z; their midpoint is at the origin.
4. **Geom subtab** — optional `Centre at origin` cleans up any
   residual offset.
5. **Junction subtab** — pick element **Au**, plane **111**, set
   `m × n × layers` (e.g. 3 × 3 × 2), set the gap (12 Å is a sensible
   default for short oligomers; longer molecules need more), click
   **Apply Add Electrode**.  In the default *anchorless* mode no
   atom selection is required: slabs land at `z = ±gap/2` around the
   world origin and the molecule fits between them.
6. **Send to Build** — Build picks up the assembled junction and you
   generate `.fdf` / `.py` normally.

### What each subtab is for

* **Atom** — delete selected atoms, add a new atom with `(dx, dy,
  dz)` offset and a live distance readout.  Used to strip H caps
  before exposing anchor atoms, or to add an explicit cap (-CH₃, -F)
  at an arbitrary site.
* **Pose** — orient an anchor pair onto the z-axis (with a tilt
  slider), or rotate the whole structure around x / y / z.  The
  Rotate op has a `Pivot` select (centroid = rotate in place,
  origin = world-axis rotation).
* **Geom** — translate the structure: centre the geometric centroid
  at the origin or apply an explicit `(Δx, Δy, Δz)` shift.  Useful
  when chaining ops or recovering from an off-origin starting xyz.
* **Junction** — add FCC electrode slabs (Au / Ag / Cu / Ni / Pt /
  Pd) on the (100) / (110) / (111) plane.  Two modes:
  * **Anchorless (default)** — slabs at `z = ±gap/2` around the
    world origin.  No atom selection needed; controlled entirely by
    `gap` + lateral `(dx, dy)` offset.  The user's job is to centre
    + pose the molecule first, then add slabs around it.
  * **Anchor-pair (legacy)** — with two atoms selected, slabs are
    placed so the *midpoint* of those anchors becomes the slab
    midpoint.  Useful when the molecule is not pre-centred or when
    you want the slabs to follow a tilted anchor pair.

Element-aware defaults for contact distance: 2.40 Å Au–S, 2.50 Å
Ag–S, 2.30 Å Cu–S / Pd–S, 2.20 Å Ni–S, 2.05 Å Pt–N (see
[`molbuilder/data/contact_distance.json`](molbuilder/data/contact_distance.json)
for citations).  Override with the contact-distance slider in
Single mode or pass `contact_distance=` in the Python API.

### Slab-only Undo

The Junction subtab carries a 20-deep **Undo** for electrode ops
only.  Other ops (delete / rotate / translate / centre) are
committed immediately and roll back via re-load.  Snapshot pushes
only on a successful response so a failed Apply doesn't consume an
undo slot.

### Atom-picking helpers

Click any atom in the viewer or in the left-hand atom list to
select; Shift-click to add to a multi-selection (orient + legacy
electrode mode read a pair as anchors).  Picked atoms wear a bright
orange wireframe halo, visible from any camera angle.  The viewer
toolbar's **Focus molecule** button anchors the camera pivot on the
molecule (ignoring electrode slabs) when interaction feels
off-centre — useful after adding bulky slabs that dominate the
auto-fit bounding box.

The full Modify spec, including every endpoint and the
electrode-placement math, lives at
[`docs/spec/modify-tab.md`](docs/spec/modify-tab.md).
The directory-layout protocol that ties Modify to Build and Watch
is at [`docs/spec/job-layout.md`](docs/spec/job-layout.md).

---

## The Watch tab

![Watch tab — molwatch.log loaded, structure in viewer, Inspect tab visible on the right, energy/force plots below](docs/img/watch-tab.png)

Point at a **run directory** (or a specific output file) and the
viewer renders:

* **Geometry** frame-by-frame in 3Dmol (movie mode; frames load once,
  animate client-side).
* **Total energy** vs step (Plotly).
* **Max atomic force** vs step.
* **Per-cycle SCF convergence** for the active step — energy + the
  residual norm on log scales, so you spot stalled / oscillating
  SCFs while the run is still going.

Four right-aside control tabs: **Style** (representation, color
scheme, cell visibility), **Overlays** (atom indices, force arrows,
max-force highlight), **Inspect** (click two atoms to measure their
live per-frame distance), **Playback** (slider, play / pause / step,
speed).

Auto-detected formats:

* `<job>.molwatch.log` — the unified format molbuilder emits;
  preferred (one file carries trajectory + per-cycle SCF data + a
  step-0 preview written at FDF emission time, so the viewer has
  content the moment the user loads).
* SIESTA stdout (`run.out` / `siesta.log`).
* geomeTRIC's `<job>_geom_optim.xyz`.

The page re-parses on file `mtime` change (~15 s polling), so a
still-running calculation streams new frames into the open tab.
The parser is truncation-tolerant: a half-written final block in a
still-running job is dropped on parse and picked up next refresh.

### Run directory + staged relaxation

Point Watch at a **directory** and it walks the [job-layout
v1 protocol](docs/spec/job-layout.md) to find the right files
automatically — `*.molwatch.log` first, then `*.fdf` / `*.py` parsed
for the basename, then generic fallbacks.

For staged relaxation (coarse → medium → tight), pick the stage
from Build's **Relaxation stage** select.  Each stage writes a
distinct `<basename>-stage<N>.molwatch.log`; pointing Watch at the
directory **merges all stages into one trajectory** with stage-
boundary markers on the energy / force plots.  The full mechanics
(restart files, what's safe to change between stages) are documented
inline as a collapsible "Staged relaxation workflow" panel right on
the Watch page.

---

## Python API

```python
import molbuilder

# Build a structure
s = molbuilder.build_peptide("ARNDC")
s = molbuilder.build_dna("ATGCATGCAT", backend="threedna", form="B")
s = molbuilder.build_from_smiles("Sc1ccc(S)cc1")
s = molbuilder.build_from_name("aspirin")

# Modify
from molbuilder.modify import (
    delete_atoms, orient_along_axis, add_symmetric_electrodes,
)
s = delete_atoms(s, [12, 13])                       # strip H caps
s = orient_along_axis(s, anchor_indices=(10, 11),    # S-S anchors → z
                      axis="z", center="midpoint")
s = s.centered()                                    # atoms centred at origin
s = add_symmetric_electrodes(s, "Au", "111", (3, 3, 2), gap=12.0)

# Output
s.to_xyz("junction.xyz")
s.to_pdb("junction.pdb")

# Generate engine input
from molbuilder.siesta import SiestaConfig, render_fdf
from molbuilder.pyscf  import PySCFConfig, render_script
fdf = render_fdf(s, SiestaConfig(system_label="junction", stage=1))
py  = render_script(s, PySCFConfig(method="UKS", xc="B3LYP"))
```

---

## CLI

```bash
# Build subcommands
molbuilder peptide ARNDC --out peptide.xyz
molbuilder dna ATGCATGCAT --backend threedna --out dna.xyz
molbuilder smiles "c1ccccc1" --out benzene.xyz
molbuilder name "aspirin" --pdb aspirin.pdb

# Generate engine input from a structure file
molbuilder fdf in.xyz out.fdf --psml-lib /opt/psml --kgrid 4x4x1
molbuilder pyscf in.xyz out.py --functional B3LYP --preopt

# Edit a structure: one op type per call, chain via stdin/stdout
molbuilder modify bdt.xyz - --orient-axis 10,11 |
  molbuilder modify - junction.xyz --electrode "Au:111:3x3x2@gap=12.0:3,0"

# Live watch
molbuilder watch parse run.molwatch.log     # JSON to stdout
molbuilder watch tail  run.molwatch.log     # NDJSON, one frame per line
molbuilder serve       --port 8000          # web UI on /

# Pre-flight validation
molbuilder validate in.xyz --config siesta.fdf
```

All subcommands accept `-` for stdin / stdout piping.  Machine-
consumable subcommands (`validate`, `watch parse`, `watch tail`)
emit JSON; warnings + progress go to stderr.

---

## Install

```bash
pip install -e .                      # core: peptide / DNA / RNA / FDF / PySCF / web
pip install -e ".[rdkit]"             # H-protonation + SMILES
pip install -e ".[name]"              # PubChem name lookup
pip install -e ".[all]"               # everything
```

Flask is in core dependencies (the web UI is part of the toolkit,
not an opt-in extra).  Conda alternative for the heaviest deps:

```bash
conda install -c conda-forge rdkit ase ambertools
pip install PeptideBuilder pubchempy flask
```

### Optional: 3DNA for canonical helices

The **3DNA** `fiber` backend produces true B-form / A-form / Z-form
DNA — the only thing the bundled `rdkit` / `amber` backends do not.
3DNA is distributed by the Olson lab (Columbia, x3dna.org) behind a
**registration + non-commercial license**.  molbuilder cannot fetch
it for you; download from http://x3dna.org/ and either:

```bash
# Option A: in-tree (auto-detected)
tar -xzf x3dna-v2.4-linux-64bit.tar.gz       # alongside pyproject.toml

# Option B: system install
tar -xzf x3dna-v2.4-linux-64bit.tar.gz -C ~/opt
export X3DNA=$HOME/opt/x3dna-v2.4
export PATH=$X3DNA/bin:$PATH
```

See [docs/design.md § "3DNA (canonical helix builder)"](docs/design.md)
for the full install + license contract.

---

## Documentation

| Document | What it covers |
|---|---|
| [`docs/design.md`](docs/design.md) | Durable design, architecture (L1/L2/L3 layering), principles, decisions log, anti-patterns |
| [`docs/spec/modify-tab.md`](docs/spec/modify-tab.md) | Modify tab Python API + endpoints + UI walkthrough |
| [`docs/spec/job-layout.md`](docs/spec/job-layout.md) | Directory + filename protocol (Build writes, Watch reads) |
| [`docs/spec/siesta-fdf.md`](docs/spec/siesta-fdf.md) | SIESTA generator contract |
| [`docs/spec/pyscf-script.md`](docs/spec/pyscf-script.md) | PySCF generator contract |
| [`docs/spec/structure.md`](docs/spec/structure.md) | `Structure` dataclass + readers / writers |
| [`docs/spec/builders.md`](docs/spec/builders.md) | Per-backend behavior (peptide / DNA / RNA / SMILES / name) |
| [`docs/spec/parsers.md`](docs/spec/parsers.md) | Trajectory parser registry + auto-detect |
| [`molbuilder/data/README.md`](molbuilder/data/README.md) | Citations for every numeric value (FCC lattice constants, etc.) |

---

## Limits

* **Single-stranded DNA / RNA only.** Double helices need a complementary
  strand placed on a Watson-Crick offset; straightforward addition.
* **Web app is single-tenant.** One user, one tab, one calculation —
  the server holds one global state under a lock.  For multi-user
  use, run a separate process per user.
* **Watch is read-only.** It does not start, monitor, or kill the
  engine process; it only reads files the engine has produced.
* **No bond-detection / CONECT output.** PDB records are ATOM-only.

---

## Running the tests

```bash
pytest tests/ -q                            # full suite (~2 min with E2E)
pytest tests/test_modify.py -q              # Python modify ops
pytest tests/test_modify_e2e.py -q          # Playwright + live Flask
pytest tests/watch/ -q                      # parser + watch-app tests
```

The Playwright E2E suite (`test_modify_e2e.py`) starts a live
Werkzeug server on a random port and drives Chromium through the
Modify tab.  Install the browser binary once with `playwright
install chromium` after `pip install pytest-playwright`.

---

## Sequence syntax

Tiny grammar shared by `peptide` / `dna` / `rna` subcommands:

```
sequence  = (oneletter | bracketed | whitespace)*
oneletter = a single ASCII letter, case-insensitive
bracketed = "[" 3-or-4-letter PDB / modified-residue code "]"
```

| Input            | Meaning                                        |
|------------------|------------------------------------------------|
| `ARNDC`          | Ala-Arg-Asn-Asp-Cys                            |
| `AR[SEP]C`       | Ala-Arg-phosphoSer-Cys                         |
| `ATGC`           | DA-DT-DG-DC (DNA, 5'→3' by convention)         |
| `AUGC`           | A-U-G-C (RNA)                                  |
| `5'-ATGC-3'`     | DNA with explicit 5'/3' labels                 |
| `3'-CGTA-5'`     | reverse-direction; parser flips before build   |

Modified residues currently supported (extend in
`molbuilder/residues.py:MODIFIED_RESIDUES`):
**SEP** phosphoserine · **TPO** phosphothreonine · **PTR**
phosphotyrosine · **MLY** N-methyl-lysine · **M3L**
N,N,N-trimethyl-lysine · **ALY** N6-acetyl-lysine.

---

## Project layout (overview)

```
molbuilder/
  structure.py / frame.py / issues.py    # L1: core data types
  config/{siesta,pyscf}.py                # L1: emission configs (field metadata)
  chemistry.py / residues.py              # L1: chemistry tables
  trajectory_log/                         # L1: .molwatch.log v1 writer
  parsers/                                # L2: trajectory parsers (SIESTA / PySCF / molwatch_log)
  builders/backends/                      # L2: build backends (amber / rdkit / threedna)
  peptide.py / nucleic.py / smiles.py     # L2: build verbs
  modify.py / validation.py               # L2: edit + validate verbs
  siesta/input.py / pyscf/input.py        # L2: generators (render_fdf / render_script)
  cli.py                                  # L3: click-based CLI
  web/                                    # L3: Flask + Blueprints + 3Dmol UI
    blueprints/{build,modify,watch}.py
    templates/{index,modify,watch}.html
    static/...
tests/
docs/
  design.md                               # durable design + architecture
  spec/                                   # per-feature contracts
  img/                                    # README screenshots
```

Layering rule: higher layers may import lower; lower layers must
never import higher.  Field metadata (label / range / validator)
lives on the dataclass field — CLI options and web form schemas are
both generated from `dataclasses.fields(Config)`, not maintained in
parallel.

### Adding a new SIESTA / PySCF parameter

The Build form is **schema-driven** end-to-end (v1.1.0+).  To expose
a new SIESTA or PySCF knob, the only edit is to the dataclass:

```python
# in molbuilder/config/siesta.py (or pyscf.py)
my_new_param: float = field(default=0.5, metadata={
    "section": "SCF",                       # which form fieldset
    "label":   "My new parameter",          # user-facing label
    "unit":    "Ry",                        # appended to the label
    "range":   (0.1, 2.0),                  # validator + form min/max
    "tier":    "advanced",                  # styling cue
    "help":    "what this knob does",       # tooltip
    # "choices": (...) for select-style enums
    # "id_suffix": "..." to override the default hyphenated id
})
```

The generator emits the right FDF / Python lines, the click CLI
gains a `--my-new-param` flag, the validator checks the range,
and `GET /api/build/schema/<engine>` ships the field to the JS
renderer which adds an input to the Build tab — **no HTML, JS, or
test-fixture edits needed**.  Pin-tests in
`tests/test_web.py::test_*_form_schema_matches_documented_layout`
update once when you add the field; everything else is automatic.

---

## License

MIT.  3DNA, when used, follows its own non-commercial license — do
not redistribute the 3DNA archive.

## Author

Quan Qing — `qqing@asu.edu`
