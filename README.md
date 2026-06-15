# molbuilder

> **An end-to-end toolkit for molecular-electronics simulations.**
> Build a molecule, assemble it into a metal–molecule–metal
> nanojunction, generate DFT/transport input for **SIESTA**,
> **TranSIESTA**, or **PySCF**, and inspect the resulting
> trajectories, spectra, and transmission curves — all from one
> Flask app, all driven by one codebase.

```
                       ┌────────── Molbuilder tab ─────────┐
sequence / SMILES / PDB │ build → edit → orient → assemble │
                       └────────────────┬──────────────────┘
                                        ▼ Save to project
                              ┌─────────┴─────────┐
                              │  Task tabs        │
                              │  ─────────────    │
                              │  Structure opt.   │ ──► .fdf  / .py
                              │  Spectrum calc.   │ ──► .spectra.py
                              │  Transport calc.  │ ──► TranSIESTA .fdf
                              └─────────┬─────────┘
                                        ▼ run on cluster
                              ┌─────────┴─────────┐
                              │  Results tab      │ ◄── trajectory
                              │  unified inspector│ ◄── spectrum
                              │                   │ ◄── T(E) / I-V
                              └───────────────────┘
```

> **Status:** active development, pre-1.0.  Used in production for
> Au–thiol–Au transport studies in the Qing lab.  Raman pipeline
> bit-for-bit validated against an independent reference
> implementation; Au-BDT-Au transport cross-checked vs Reed 2006 /
> Stokbro 2003 within factor-of-2.  MIT licensed.

---

## Highlights

- **Full nanojunction pipeline in one app** — peptide / DNA / RNA /
  SMILES / PubChem compound builders → atom-level editor → orient
  anchors → add crystallographic FCC slabs (Au / Ag / Cu / Ni / Pt /
  Pd on 100 / 110 / 111) → transport-ready geometry → SIESTA `.fdf` /
  TranSIESTA / PySCF.  Most tools do one step; molbuilder does all of
  them.
- **Five backends, isolated by design** — `molbuilder-siesta`
  (CPU), `molbuilder-siesta-gpu` (source-built CUDA), `molbuilder-pySCF`
  (CPU + optional GPU via gpu4pyscf), `molbuilder-MDtools` (AmberTools),
  `molbuilder-tests` (Playwright).  Each env pins its own native stack;
  no numpy 1.x vs 2.x or libnetcdf-version conflicts.  All managed
  through one `molbuilder envs {list,doctor,install}` CLI.
- **Optional GPU SIESTA, source-built** — ELPA (CUDA) + ELSI +
  SIESTA 5.4.2 compiled from source against the env's pinned
  toolchain.  Sentinel-resume build (re-running is safe), interactive
  preflight (CUDA / GPU compute capability / disk / git
  reachability), and three layers of build-env isolation from system
  MPI / CUDA / compilers.
- **Schema-driven UI + CLI** — every parameter on a SIESTA / PySCF /
  Transport config is `@dataclass` field metadata.  Add a new
  parameter to the dataclass and you get: a CLI flag, a web form
  input with tooltip + validator, a methods-text mention, and form
  schema introspection.  No parallel HTML / JS / fixture edits.
- **Live inspection that refreshes on file mtime** — the Results
  tab's trajectory inspector polls running calculations: new frames
  stream into the 3Dmol viewer, new energy / force / SCF-residual
  data lights up the Plotly charts.  Truncation-tolerant parser
  handles half-written final blocks.
- **Built-in OAuth without nginx** — Google / GitHub / Microsoft /
  ORCID / Apereo CAS (e.g. ASURITE) for internet-exposed deployments.
  Per-provider `allowed_users` lists.  Or put molbuilder behind your
  existing reverse-proxy auth — both shapes documented.
- **Project organization out of the box** — JupyterLab-style
  sidebar at `projects/`, single-click preview, double-click commit,
  atomic move / copy / rename that pairs structure files with their
  `.molstruct.json` sidecars (per-atom labels never orphan).
- **Validated science** — Raman pipeline (build → relax → Hessian →
  finite-diff Raman) bit-for-bit identical to a hand-written
  raw-PySCF reference at B3LYP/def2-SVP water.  Au-BDT-Au transport
  T(E_F) within factor-of-2 of Reed 2006 / Stokbro 2003 (~0.01 G₀).
- **Sole-source-of-truth documentation** — every UI feature has a
  spec in [`docs/`](docs/); tests are derived from the spec, code
  reviews verify code-matches-spec (not code-matches-itself).  No
  doc-vs-code drift.

> Generated structures are **starting points for a geometry
> optimisation**, not equilibrium geometries.  Always relax in your
> DFT code before computing properties.

---

## Quick start

```bash
# 1. One-time host env (any name; we suggest "molbuilder")
conda create -n molbuilder -c conda-forge -y python=3.12 pip \
    numpy ase sisl rdkit openbabel biopython \
    flask click plotly authlib python-cas pytest pyflakes
conda run -n molbuilder python -m pip install PeptideBuilder pubchempy

# 2. From inside the host env:
conda activate molbuilder
cd /path/to/molbuilder

# 3. Add the backend envs you need (each ~3 GB; each is one CLI call)
python -m molbuilder envs install molbuilder-siesta      # CPU SIESTA
python -m molbuilder envs install molbuilder-pySCF       # PySCF + geomeTRIC
python -m molbuilder envs install molbuilder-MDtools     # AmberTools
# Optional GPU SIESTA (45-min source build; gated by preflight + confirm):
bash scripts/siesta-gpu-bootstrap.sh

# 4. Start the web app
python -m molbuilder serve --port 8000
# Browser: http://localhost:8000/  → redirects to /molbuilder
```

For LAN or internet exposure (TLS, OAuth sign-in, reverse-proxy
auth), see [§ Deployment](#deployment) and
[`docs/deployment.md`](docs/deployment.md).

---

## Feature tour

molbuilder is a Flask app with **five canonical tabs** plus a
persistent Projects sidebar.  Each tab has one role; tab switches
happen only when the workflow phase changes (build → configure →
review).  Routes match the visible tab labels exactly.

| Tab | Route | Role |
|---|---|---|
| **Molbuilder** | `/molbuilder` (bare `/` redirects) | Interactive workspace — load / build / edit / assemble |
| **Structure optimization** | `/structure-optimization` | File-driven SIESTA `.fdf` + PySCF `.py` generator |
| **Spectrum calculation** | `/spectrum-calculation` | File-driven PySCF Raman / IR script generator |
| **Transport calculation** | `/transport-calculation` | File-driven TranSIESTA `.fdf` generator |
| **Results** | `/results` | Unified file-dispatched inspector — trajectory, spectra, structure, source |

### 1. The Molbuilder tab — interactive workspace

![Molbuilder workspace — 3Dmol viewer at centre, atom-list + selection panel on the left, foldable Sources / Atom / Pose / Geom / Junction / Save command panels on the right](docs/img/modify-tab.png)

> *The Molbuilder tab is the only tab that holds in-memory canvas
> state.  Every other tab reads from disk.  Foldable panels — no
> sub-tabs, no wizard flow — every command is reachable from one
> screen.*

**What you do here:**

- **Sources** — load `.xyz` / `.pdb` from the sidebar selection;
  generate from a SMILES string (RDKit), a peptide / DNA / RNA
  sequence, a compound name (PubChem), or a canonical B / A / Z DNA
  helix (3DNA, optional).
- **Atom** — click atoms in the viewer (or atom list) to select;
  Shift-click extends the selection; picked atoms wear a bright
  orange wireframe halo visible from any angle.  Delete selected
  atoms (strip H caps to expose S anchors), or insert a new atom at
  `(dx, dy, dz)` with a live distance readout.
- **Pose** — orient an anchor pair onto the z-axis (with a tilt
  slider) so the S–S vector points along z, or rotate the whole
  structure around x / y / z with a centroid / origin pivot.
- **Geom** — centre the geometric centroid at the origin or apply
  an explicit `(Δx, Δy, Δz)` shift.  Used to clean up after chained
  ops or to recover an off-origin xyz.
- **Junction** — add FCC electrode slabs (Au / Ag / Cu / Ni / Pt /
  Pd; 100 / 110 / 111) at a chosen gap.  **Anchorless mode**
  (default): slabs land at `z = ±gap/2` around the world origin —
  no atom selection needed.  **Anchor-pair mode** (legacy): slabs
  placed so the midpoint of two selected anchor atoms becomes the
  slab midpoint.
- **Save** — write `<project>/<name>.xyz` + a `.molstruct.json`
  sidecar with per-atom labels.  File-driven task tabs pick it up.

**What makes this tab unique:**

- 20-deep **slab-only Undo** lets you sweep `gap` values
  exploratorily without losing your atom-edit history.
- Element-aware contact distances ship as defaults (2.40 Å Au–S,
  2.50 Å Ag–S, 2.30 Å Cu–S / Pd–S, 2.20 Å Ni–S, 2.05 Å Pt–N) with
  citations in
  [`molbuilder/data/contact_distance.json`](molbuilder/data/contact_distance.json).
- **Focus molecule** button anchors the camera on the molecule
  (ignoring the bulky slabs) when interaction feels off-centre
  after adding electrodes.
- Auto-detection chip identifies the chemistry (e.g.
  "Au-thiol-Au junction; closed-shell singlet") and surfaces
  validator warnings inline.

Spec: [`docs/tabs/molbuilder.md`](docs/tabs/molbuilder.md).

### 2. Structure optimization — SIESTA `.fdf` + PySCF `.py`

![Structure-optimization form — engine selector at top, three workflow-group cards (Profile / Stage / Budget), 3Dmol viewer rendering the input geometry, inline detection chip + per-card issues panel](docs/img/build-tab.png)

> *A file-driven task tab: the user picks an `.xyz` / `.pdb` from
> the sidebar, the form configures it, and Generate emits a
> self-contained `<name>.fdf` (or `.py`) + `<name>.run.sh` wrapper
> that already knows which conda env to dispatch into.*

**What's special:**

- **Schema-driven form** generated from `SiestaConfig` /
  `PySCFConfig` dataclass field metadata.  Adding a new knob is a
  one-line edit; CLI flag + form input + tooltip + validator all
  follow automatically.
- **Three workflow-group cards** — Profile / Stage / Budget —
  group fields by life-cycle phase (what the system is, what stage
  you're at, what computational budget you have).  Not alphabetical;
  not by FDF block.  Pinned by per-card e2e tests.
- **Methods-text preview** writes manuscript-ready prose for the
  methods section of a paper, kept in sync with the form state.
- **Issues panel** routed through the shared `analyze_structure`
  pipeline.  The chip, the validator, and the preflight all agree
  on chemistry (e.g. Au-BDT-Au is correctly identified as a
  noble-metal cluster — the open-shell-spin warning is suppressed).
- **Staged relaxation** (coarse → medium → tight) — each stage
  writes a distinct `<basename>-stage<N>.molwatch.log`; pointing
  the Results inspector at the directory **merges stages into one
  trajectory** with stage-boundary markers on the energy / force
  plots.

Spec: [`docs/tabs/structure-optimization.md`](docs/tabs/structure-optimization.md).

### 3. Spectrum calculation — PySCF Raman / IR

A file-driven task tab that generates `<job>.spectra.py` PySCF
scripts for **harmonic vibrational analysis** (frequencies + Raman
activities + optional per-mode electronic-structure probes +
scaffolded IR).

**What's special:**

- **End-to-end validated** — the Raman pipeline produces bit-for-bit
  identical frequencies and Raman activities to a hand-written
  raw-PySCF reference script at B3LYP/def2-SVP water.  Method +
  full result table:
  [`docs/tabs/spectra/spec.md § 12.1`](docs/tabs/spectra/spec.md).
- **Per-mode electronic-structure probes** — optional displaced-SCF
  jobs around the equilibrium geometry, projected onto each mode's
  eigenvector to compute mode-resolved orbital responses.
- **IR add-on scaffold** (`compute_ir=True`) populates
  `ir_intensity_km_mol` "for free" on top of the Raman finite-diff
  loop (dipole-moment readout adds no extra SCFs).  **Absolute
  magnitudes are unvalidated** — treat as preliminary.
- **Output format includes mass-weighted canonical eigenvectors**
  (for post-hoc Raman / IR re-projection) **plus display-normalised
  eigenvectors** (for 3-D animation in the Results tab).

Spec + bibliography:
[`docs/tabs/spectra/spec.md`](docs/tabs/spectra/spec.md) +
[`docs/tabs/spectra/references.bib`](docs/tabs/spectra/references.bib).

### 4. Transport calculation — TranSIESTA scripts

A file-driven task tab that emits TranSIESTA `.fdf` for **zero-bias
transmission**.  Today's scope is the zero-bias path; bias-scan and
electrode-`.TSHS`-generation wizards are roadmap items.

**What's special:**

- **Au-BDT-Au validation fixture** in `tests/` cross-checks T(E_F)
  within factor-of-2 of Reed 2006 / Stokbro 2003 (~0.01 G₀).
- **Atom-ordering preflight** catches the canonical failure mode
  (TranSIESTA needs left-lead → device → right-lead ordering;
  silent miscounts produce wrong transmission and no error).
- **Validator covers** k-mesh, contour parameters, electrode mode,
  mesh cutoff defaults per element (Au needs a higher cutoff than
  the SIESTA default).
- **Region labels** persist through the workflow via the
  `.molstruct.json` sidecar (electrode / bridge / anchor regions
  set in the Molbuilder tab carry into the TranSIESTA emitter).

Engine doc: [`docs/engines/transport.md`](docs/engines/transport.md).

### 5. Results — unified inspector

![Trajectory inspector — viewer with .molwatch.log loaded, frame strip + scrub slider below, Style/Overlays/Playback controls combined, energy + force + SCF-residual plots stacked on the right](docs/img/watch-tab.png)

> *Pick any file in the Projects sidebar; `/results` dispatches to
> the right inspector based on extension.  Same UI for "is the
> optimisation converged?" and "is the transmission peak in the
> right place?".*

| File pattern | Inspector | Highlights |
|---|---|---|
| `*.xyz`, `*.pdb` | Structure preview | 3Dmol viewer, atom-list cross-highlight, axes overlay toggle |
| `*.fdf`, `*.py`, `*.log`, `*.out`, `*.txt`, `*.md`, `*.json` | Source listing | Read-only CodeMirror; Find dialog; > 1 MB files load view-only |
| `*.molwatch.log`, `<job>.out` (SIESTA), `<job>_geom_optim.xyz` (geomeTRIC) | Trajectory | 3Dmol movie + Plotly energy / force / SCF-residual; frame slider; atom-distance measurement; auto-refresh on mtime |
| `*.spectra.json` | Spectra | Lorentzian-broadened spectrum + modes table + per-mode 3-D animation |
| `*.transport.json` | Transport | T(E) + I-V Plotly charts (planned) |

**Architecture:**

- **Inspector Registry** at `lib/inspectors/registry.js` — each
  inspector self-registers; the dispatcher knows nothing about
  specific file types.
- Adding a new file type = one new `lib/inspectors/<name>.js` +
  one `<script>` tag in `results.html`.  No edit to the dispatcher.
- **Explicit mount lifecycle** — each inspector returns a `dispose()`
  handle so file-swap cleanly tears down 3Dmol viewers, Plotly
  charts, and polling timers.
- **Live polling on `mtime` change** — streams new frames into an
  open inspector while a calculation is still running.  Parser
  drops half-written final blocks and picks them up next refresh.

Spec: [`docs/tabs/results.md`](docs/tabs/results.md) +
[`docs/protocols/results-tab.md`](docs/protocols/results-tab.md) +
[`docs/protocols/inspector-registry.md`](docs/protocols/inspector-registry.md).

---

## Workflow — the canonical cross-tab flow

Two principles:

1. **The Molbuilder tab is the only interactive workspace.**  It
   holds the in-memory canvas.  Everything else reads from disk.
2. **Task tabs are file-driven.**  Structure-optimization,
   Spectrum-calculation, Transport-calculation all read their input
   geometry from the sidebar-selected project file.  They do NOT
   read in-memory canvas state.

This decouples interactive editing from deterministic script
generation: the same project directory always produces the same
script regardless of which tab the user came from.

```
[Molbuilder tab]
   ↓ load file OR generate (SMILES / peptide / DNA / name / 3DNA)
   ↓ edit (delete, add, orient, rotate, translate)
   ↓ assemble (anchorless or anchor-pair slab)
   ↓ Save to project  ────► <proj>/<name>.xyz  +  .molstruct.json
                                                 │
   [Structure optimization tab]                  │
      sidebar pick ◄───────────────────── pick the saved file
      configure form                             │
      Generate ──────────► <proj>/<name>.fdf  +  .run.sh  +  .psml
                                                 │
   [Run on cluster, results land back]           │
                                                 │
   [Results tab]                                 │
      sidebar pick ◄───────────────────── pick <name>.out  /  .molwatch.log
      trajectory inspector renders
                                                 │
   (optional) [Spectrum calculation tab]         │
      sidebar pick ◄───────────────────── pick the optimised geometry
      configure form
      Generate ──────────► <proj>/<name>.spectra.py
                                                 │
   [Run on cluster, results land back]           │
                                                 │
   [Results tab]                                 │
      sidebar pick ◄───────────────────── pick <name>.spectra.json
      spectra inspector renders (modes + chart + 3-D animation)
```

Every arrow except "Save to project" and the cluster round-trip is
a same-tab UI gesture.  Tab switches happen only when the workflow
phase changes.

Cross-tab architecture spec:
[`docs/tabs/architecture.md`](docs/tabs/architecture.md).

---

## Design at a glance

### Three-layer architecture

```
┌──────────────────────────────────────────────────────────┐
│  L3 — Surfaces                                            │
│  cli.py (click), web/app.py (Flask + Blueprints)          │
│  Convert UI gestures → L2 calls.  No business logic.      │
├──────────────────────────────────────────────────────────┤
│  L2 — Domain verbs                                        │
│  builders/, generators/, parsers/, validation/            │
│  Each verb is a focused module operating on L1 types.     │
├──────────────────────────────────────────────────────────┤
│  L1 — Core types (nouns)                                  │
│  structure.py, frame.py, config/, issues.py               │
│  + chemistry, residues, trajectory_log/                   │
│  Pure data + minimal serialization.  Field metadata here. │
└──────────────────────────────────────────────────────────┘
```

**Layering rule (load-bearing):** higher layers may import lower;
lower layers never import higher.  Field metadata (label / range /
validator / units / tooltip) lives **on the dataclass field** —
CLI options and web form schemas are both generated from
`dataclasses.fields(Config)`, not maintained in parallel.

### Four core types

| Type | Role |
|---|---|
| `Structure` | One geometric configuration: elements + positions + PDB metadata + optional cell + region labels |
| `Frame` | A `Structure` plus per-step physics (energy, forces, lattice, scf_history) — parse-side |
| `SiestaConfig`, `PySCFConfig`, `TransportConfig` | Emission parameters per backend; carry the field metadata that drives CLI options, form schema, and validation |
| `Issue` | A validation finding: `severity` (error / warn), `message`, `where` (field id / "geometry"), `workflow_group` (Profile / Stage / Budget routing) |

### The doc rule

Every UI feature has a `docs/tabs/*.md` or `docs/protocols/*.md`
spec that is the **single source of truth**.  Tests are derived
from the spec without reading the implementation.  Code reviews
verify code-matches-spec, not code-matches-itself.  Master index:
[`docs/design.md`](docs/design.md) § 0.

### Why split build / modify / generate / inspect across tabs?

After 5+ rounds of practical use, collapsing them into one tab
forced a save-reload round-trip for every generated structure that
needed editing.  The 5-tab split + file-driven task tabs makes the
script output **deterministic from disk alone**: same project dir
→ same script.  Two users on the same project see the same script.
Sharing a project (export / re-import) loses no information.

Full architecture + principles + decisions log:
[`docs/design.md`](docs/design.md).

---

## Install + multi-env model

molbuilder runs from a **user-named host env** (any name; we
suggest `molbuilder`) and dispatches into named backend envs via
`conda run -n <env> ...`.  This model exists because collapsing
AmberTools + siesta-mpi + cupy + playwright into one env produces
three independent unresolvable dep conflicts.  Keeping them
separate lets each backend pin its own native stack.

### The envs

| Env | Contents | When you need it |
|---|---|---|
| **host env** | flask + click + numpy + ase + sisl + rdkit + openbabel + biopython + plotly | always — runs `python -m molbuilder ...`, build-time chemistry, the web UI |
| `molbuilder-siesta` | precompiled `siesta=5.4.2=mpi_openmpi_*` | CPU SIESTA / TranSIESTA jobs |
| `molbuilder-siesta-gpu` *(optional)* | source-built ELPA + ELSI + SIESTA 5.4.2 with CUDA-enabled ELPA | GPU-accelerated SIESTA / TranSIESTA |
| `molbuilder-pySCF` | pyscf + geometric + (optional) gpu4pyscf + CUDA 13 | PySCF / Spectra / Spectrum-calculation jobs |
| `molbuilder-MDtools` | ambertools-dac=26 (dacase channel) | tleap / parmchk2 / RESP / antechamber |
| `molbuilder-tests` | playwright + pytest-playwright + Chromium | browser E2E suite |

### One CLI manages every env

```bash
python -m molbuilder envs list                      # one-line status per recipe
python -m molbuilder envs doctor                    # full health report (runs verify per env)
python -m molbuilder envs install molbuilder-siesta # idempotent install
python -m molbuilder envs install <name> --dry-run  # preview the plan
python -m molbuilder envs install <name> --check    # report present + verified
# Or from any shell (no host env activation needed):
bash scripts/install-env.sh <name>
```

Recipes are declared in
[`molbuilder/envs/recipes.py`](molbuilder/envs/recipes.py); a
consistency test asserts the README ↔ registry pairing so the doc
and code can't drift silently.

### GPU SIESTA from source

`molbuilder-siesta-gpu` builds **ELPA + ELSI + SIESTA 5.4.2** from
source against CUDA-enabled ELPA.  The install runs ~45 min on 8
cores and consumes ~12 GB under `$CONDA_PREFIX`.

```bash
bash scripts/siesta-gpu-bootstrap.sh           # first-time install
bash scripts/siesta-gpu-bootstrap.sh --dry-run # preview plan + preflight
bash scripts/siesta-gpu-rebuild.sh siesta      # rebuild one component
```

**What's notable:**

- **CUDA toolkit lives in the env** (`cuda-version=13.*`,
  `cuda-nvcc`, `cuda-cudart-dev`, `libcublas-dev`) — the host
  provides only the NVIDIA driver + `nvidia-smi`.  Mirrors the
  `molbuilder-pySCF` env pattern.
- **Two-component source build** (per SIESTA 5.4 INSTALL.md):
  ELPA externally (CUDA-enabled — conda-forge ELPA isn't built with
  CUDA), and SIESTA cloned `--recurse-submodules` so the four
  required ESL libraries (`libfdf`, `libpsml`, `xmlf90`, `libgridxc`)
  + ELSI + libxc come along as `External/` submodules and SIESTA's
  cmake compiles them on the fly.  All other deps (gcc, MPI, BLAS,
  ScaLAPACK, NetCDF, HDF5, FFTW, CUDA toolkit, libxc) are conda-forge
  packages.
- **All version pins exposed as env-var overrides** for
  customisation; defaults are the investigated stable values:
  `MOLBUILDER_ELPA_TAG` (default `new_release_2023.05.001` —
  verified to exist on MPCDF GitLab via `git ls-remote`),
  `MOLBUILDER_ELPA_REPO`, `MOLBUILDER_SIESTA_TAG` (default `rel-5.4`
  — branch since upstream has no numeric 5.x tags),
  `MOLBUILDER_SIESTA_REPO`, `MOLBUILDER_CUDA_VERSION` (default `13.*`),
  `MOLBUILDER_GCC` (default `14`), `MOLBUILDER_LIBXC_VERSION`,
  `MOLBUILDER_CUDA_CC` (auto-detect via `nvidia-smi`),
  `MOLBUILDER_BUILD_JOBS` (default `min(nproc, 8)`).
- **Sentinel-resume build** — keyed on a toolchain fingerprint
  (CUDA / gcc / OpenMPI versions + per-component git SHAs).  Any
  change forces the relevant rebuild; nothing else.
- **Interactive preflight** detects + reports CUDA version, GPU
  compute capability + name, gcc + OpenMPI + disk free + git
  reachability of every component upstream.  Asks for confirmation
  before the 45-min commitment.  `--yes` bypasses for CI.
- **Three layers of build-env isolation** prevent the build from
  silently linking against system MPI / CUDA / compilers when the
  user has `apt install libopenmpi-dev`:

  | Layer | What it does |
  |---|---|
  | L1 — subprocess env sanitizer | Strips ~60 vars + 7 prefix families (`LD_LIBRARY_PATH`, `CPATH`, `CFLAGS`, `LDFLAGS`, `MPI_HOME`, `CUDA_HOME`, `OMPI_*`, `MPICH_*`, …) before every `conda run` |
  | L2 — explicit cmake compiler pins | `-DCMAKE_PREFIX_PATH={env}` + `-DMPI_C_COMPILER={env}/bin/mpicc` + `-DCMAKE_CUDA_COMPILER={env}/bin/nvcc` + `-DCUDAToolkit_ROOT={env}` make FindMPI / FindCUDAToolkit unable to wander |
  | L3 — `$ORIGIN`-relative install rpath | Baked into every binary so the runtime loader finds env libs even without `LD_LIBRARY_PATH`; env stays movable (rename, clone, copy) |

Full engineering doc:
[`docs/engines/siesta-gpu.md`](docs/engines/siesta-gpu.md).

### Optional: 3DNA for canonical helices

The **3DNA** `fiber` backend produces true B / A / Z DNA — the
only thing the bundled `rdkit` / `amber` backends don't.  3DNA is
distributed by the Olson lab (Columbia, x3dna.org) behind a
**registration + non-commercial license**.  molbuilder cannot
fetch it for you; download from http://x3dna.org/ and either
in-tree extract (auto-detected) or set `$X3DNA`.  Full install +
license contract: [`docs/design.md`](docs/design.md) § "3DNA
(canonical helix builder)".

Full install recipe:
[`docs/README_install.md`](docs/README_install.md).

---

## Deployment

> **Target deployment: a workstation (laptop, lab server, or HPC
> node) with multi-CPU and optional NVIDIA GPU.**  molbuilder is
> **not** designed for and does not target cloud / AWS /
> containerised deployment.  MPI is used for **intra-workstation
> parallelism** (e.g. `mpirun -np 8` across the local cores or NUMA
> nodes); the molbuilder app, the conda envs, and every backend run
> on the same physical machine.

The default `python -m molbuilder serve` binds `127.0.0.1` —
reachable only from the same machine.  No auth, no TLS.  Right
default for a personal research tool on your own laptop.

For anything beyond localhost:

| Goal | What you need | Doc |
|---|---|---|
| **LAN** (lab workstation reachable from your laptop) | TLS cert / key for non-loopback bind | [`docs/deployment.md`](docs/deployment.md) § 1 |
| **Internet — built-in sign-in** | Google / GitHub / Microsoft / ORCID / Apereo CAS (e.g. ASURITE); enable one or several with one button on the login page; each provider gets its own `allowed_users` list | [`docs/deployment.md`](docs/deployment.md) § 2a |
| **Internet — your auth already exists** (campus SSO, Cloudflare Access, etc.) | Put molbuilder behind your existing auth gateway (reverse proxy) | [`docs/deployment.md`](docs/deployment.md) § 2b |

Configuration lives in **one file** at the repo root:
`molbuilder.json` (gitignored).  Copy the template:

```bash
cp docs/molbuilder.json.example molbuilder.json
$EDITOR molbuilder.json         # delete sections you don't need
molbuilder serve --host 0.0.0.0 --port 443
```

The template has inline `_comment_*` keys explaining every field
(JSON doesn't support comments; molbuilder's parser silently
ignores `_comment_*` — they ride along as inline documentation).

### What molbuilder does on its own

For any non-default deployment, the server enforces:

- TLS-or-loopback guard at startup (binding non-loopback without TLS
  is a hard error)
- Content-Security-Policy + X-Frame-Options + X-Content-Type-Options
  + Referrer-Policy headers
- Self-hosted 3Dmol (no CDN trust)
- Path validation on every file-ops endpoint (no `..` escape)
- Filename validation on upload, 50 MB upload cap

### What molbuilder explicitly does NOT do

Delegate to the deployment layer:

- Account management (user CRUD, password resets)
- CSRF tokens
- Rate limiting
- Audit logging
- Per-user `projects/` isolation

[`docs/deployment.md`](docs/deployment.md) explains which
deployment shape covers which of these — and why the split is the
way it is.

---

## Python API + CLI (for scripting)

molbuilder is also a Python library and a CLI you can pipe through.
Quick examples:

```python
import molbuilder
s = molbuilder.build_from_smiles("Sc1ccc(S)cc1")        # 1,4-benzenedithiol
from molbuilder.modify import delete_atoms, orient_along_axis, add_symmetric_electrodes
s = delete_atoms(s, [12, 13])                            # strip H caps
s = orient_along_axis(s, (10, 11), axis="z", center="midpoint")
s = add_symmetric_electrodes(s.centered(), "Au", "111", (3, 3, 2), gap=12.0)
s.to_xyz("junction.xyz")

from molbuilder.siesta import SiestaConfig, render_fdf
open("junction.fdf", "w").write(render_fdf(s, SiestaConfig(system_label="junction")))
```

```bash
molbuilder smiles "Sc1ccc(S)cc1" - |                              # stdin/stdout piping
  molbuilder modify - - --orient-axis 10,11 |
  molbuilder modify - junction.xyz --electrode "Au:111:3x3x2@gap=12.0:3,0"
molbuilder fdf junction.xyz junction.fdf --psml-lib /opt/psml --kgrid 4x4x1
```

Every Python verb has a matching CLI subcommand; all CLI subcommands
accept `-` for stdin / stdout.  Machine-consumable subcommands
(`validate`, `watch parse`, `watch tail`) emit JSON; warnings +
progress go to stderr.

Full Python API + CLI reference:
[`docs/types/structure.md`](docs/types/structure.md) +
[`docs/protocols/cli.md`](docs/protocols/cli.md) +
[`docs/engines/builders.md`](docs/engines/builders.md).

---

## Scientific validation

molbuilder's correctness claims are anchored to **external
cross-checks**, not internal coherence:

| Pipeline | Validation | Result |
|---|---|---|
| **Raman** (build → relax → Hessian → finite-diff Raman) | Independent hand-written raw-PySCF reference script, water at B3LYP/def2-SVP | Bit-for-bit identical: frequencies max Δ < 10⁻³ cm⁻¹, Raman activities max Δ < 10⁻⁶ Å⁴/amu.  Absolute magnitudes within literature range. |
| **PySCF relaxation** (geomeTRIC) | Same water reference | Max position Δ 1.1 × 10⁻⁷ Å |
| **Au-BDT-Au transport** (TranSIESTA zero-bias) | Reed 2006 / Stokbro 2003 published T(E_F) | Factor-of-2 of literature ~0.01 G₀ (integration test gated; runs in `molbuilder-siesta` env with electrode `.TSHS` files) |
| **IR add-on** (`compute_ir=True`) | **Not yet validated.**  Scaffold emits `ir_intensity_km_mol` but absolute magnitudes are preliminary until the Raman-style external cross-check is applied. |

Method + full result tables:
[`docs/tabs/spectra/spec.md § 12.1`](docs/tabs/spectra/spec.md) +
[`docs/protocols/scientific-validation.md`](docs/protocols/scientific-validation.md).

---

## Documentation

| Document | What it covers |
|---|---|
| [`docs/design.md`](docs/design.md) | **Master design** — mission, three-layer architecture, four core types, principles, anti-patterns, decisions log (chronological) |
| [`docs/README_install.md`](docs/README_install.md) | Install recipes (host + 5 backend envs) + `molbuilder envs` CLI |
| [`docs/deployment.md`](docs/deployment.md) | **Deployment** — localhost / LAN / internet, built-in sign-in vs reverse-proxy auth, TLS, config reference, security headers |
| [`docs/molbuilder.json.example`](docs/molbuilder.json.example) | **Config template** — every supported section with inline `_comment_*` annotations |
| [`docs/tabs/architecture.md`](docs/tabs/architecture.md) | Tab inventory + canonical routes + cross-tab workflow model |
| [`docs/tabs/molbuilder.md`](docs/tabs/molbuilder.md) | Molbuilder tab (interactive workspace) — Sources / Atom / Pose / Geom / Junction / Save |
| [`docs/tabs/structure-optimization.md`](docs/tabs/structure-optimization.md) | Structure-optimization tab — SIESTA `.fdf` + PySCF `.py` form |
| [`docs/tabs/spectra/spec.md`](docs/tabs/spectra/spec.md) | Spectrum-calculation tab — schema, layers, atom-fixing semantics; end-to-end Raman validation in § 12.1 |
| [`docs/tabs/results.md`](docs/tabs/results.md) | Results tab (stub pointing at protocols/) |
| [`docs/protocols/results-tab.md`](docs/protocols/results-tab.md) | Results tab dispatch architecture |
| [`docs/protocols/inspector-registry.md`](docs/protocols/inspector-registry.md) | Inspector contract — `mount(host, file, ctx) → {dispose}`, trajectory inspector internals |
| [`docs/protocols/projects-sidebar.md`](docs/protocols/projects-sidebar.md) | Sidebar architecture + API + sidecar-pairing semantics |
| [`docs/protocols/job-layout.md`](docs/protocols/job-layout.md) | Directory + filename protocol |
| [`docs/protocols/sidecar-contract.md`](docs/protocols/sidecar-contract.md) | `.molstruct.json` sidecar — schema + atomic-move/copy rules |
| [`docs/protocols/web-api.md`](docs/protocols/web-api.md) | HTTP API reference for every blueprint |
| [`docs/protocols/web-ui-coherence.md`](docs/protocols/web-ui-coherence.md) | Cross-surface coherence rules (analyzer / chip / validator / palette must agree) |
| [`docs/protocols/scientific-validation.md`](docs/protocols/scientific-validation.md) | External validation fixtures + reference results |
| [`docs/engines/siesta.md`](docs/engines/siesta.md) | SIESTA generator contract |
| [`docs/engines/pyscf.md`](docs/engines/pyscf.md) | PySCF generator contract |
| [`docs/engines/transport.md`](docs/engines/transport.md) | TranSIESTA generator + transport roadmap |
| [`docs/engines/siesta-gpu.md`](docs/engines/siesta-gpu.md) | **GPU SIESTA env** — source-build recipe, BuildSpec executor, sentinel-resume model, three-layer isolation |
| [`docs/engines/builders.md`](docs/engines/builders.md) | Per-backend behaviour (peptide / DNA / RNA / SMILES / name) |
| [`docs/types/structure.md`](docs/types/structure.md) | `Structure` dataclass + readers / writers |
| [`docs/types/parsers.md`](docs/types/parsers.md) | Trajectory parser registry + auto-detect |
| [`molbuilder/data/README.md`](molbuilder/data/README.md) | Citations for every numeric value (FCC lattice constants, contact distances, …) |

---

## Limits

- **Single-stranded DNA / RNA only.**  Double helices need a
  complementary strand placed on a Watson-Crick offset;
  straightforward addition.
- **Web app is single-tenant.**  One user, one tab, one
  calculation — the server holds one global state under a lock.
  For multi-user use, run a separate process per user (gunicorn
  with `--workers 1 --threads ...` per user, separate ports).
- **Results inspectors are read-only.**  They do not start,
  monitor, or kill the engine process; they only read files the
  engine has produced.
- **No bond detection / CONECT output.**  PDB records are
  ATOM-only.
- **Transport tab is zero-bias today.**  Bias scan + the electrode
  `.TSHS` generation wizard are roadmap items.
- **PySCF IR absolute magnitudes are scaffolded, not validated.**

---

## For developers

| Topic | Where the canonical doc lives |
|---|---|
| Three-layer architecture, four core types, principles, decisions log | [`docs/design.md`](docs/design.md) |
| Project layout (every file's role + the layering rule) | [`docs/package-layout.md`](docs/package-layout.md) |
| Adding a new SIESTA / PySCF / Transport parameter (schema-driven; one-line dataclass edit) | [`docs/engines/siesta.md`](docs/engines/siesta.md) + [`docs/engines/pyscf.md`](docs/engines/pyscf.md) + [`docs/protocols/web-ui-coherence.md`](docs/protocols/web-ui-coherence.md) |
| Sequence syntax (peptide / DNA / RNA grammar + modified residues) | [`docs/engines/builders.md`](docs/engines/builders.md) |
| HTTP API reference for every blueprint | [`docs/protocols/web-api.md`](docs/protocols/web-api.md) |
| `.molwatch.log` v1 file format | [`docs/types/parsers.md`](docs/types/parsers.md) |
| `.molstruct.json` sidecar contract (per-atom labels, region tags) | [`docs/protocols/sidecar-contract.md`](docs/protocols/sidecar-contract.md) |
| Test-strategy pyramid (L1 unit / L2 module / L3 interface / L4 integration / L5 e2e) | [`docs/protocols/test-strategy.md`](docs/protocols/test-strategy.md) |
| GPU SIESTA env — BuildSpec executor, sentinel resume, 3-layer build-env isolation | [`docs/engines/siesta-gpu.md`](docs/engines/siesta-gpu.md) |

Running the tests:

```bash
pytest tests/ -q                            # full suite (~45 min with E2E)
pytest tests/test_envs_*.py -q              # env recipes + builds executor (fast)
pytest tests/test_modify_e2e.py -q          # Playwright + live Flask (needs molbuilder-tests env)
```

---

## License

MIT.  3DNA, when used, follows its own non-commercial license — do
not redistribute the 3DNA archive.

## Author

Quan Qing — `qqing@asu.edu`
