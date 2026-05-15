# molbuilder — installation and environment preparation

This guide gets a clean machine (with conda already installed) to a working
molbuilder setup.  For the architectural reasoning behind the multi-env
layout, see [`design.md`](design.md) — the 2026-05-14 entry of the
Decisions log captures the "why".

---

## TL;DR — minimum to run the web UI

```bash
git clone https://github.com/Qing-LAB/molbuilder.git
cd molbuilder

conda create -n molbuilder -c conda-forge -y python=3.12 pip \
    numpy ase sisl rdkit openbabel biopython \
    flask click plotly pytest pyflakes
conda run -n molbuilder python -m pip install PeptideBuilder pubchempy

conda activate molbuilder
python -m molbuilder serve
```

Open <http://127.0.0.1:5000>.

Two small things in the snippet above worth noting:

- `pip` is listed explicitly in the `conda create` package list.  conda-forge's
  `python` does ship pip transitively, but listing it makes the dependency
  declarative — if the solver ever drops it, you'll see the change in the
  command, not at runtime.
- We use `python -m pip install ...` instead of `pip install ...`.  This
  forces the activated env's Python to load the pip *module* directly,
  bypassing any `pip` binary that PATH might resolve to first (notably
  `~/.local/bin/pip`, which Ubuntu's default `~/.profile` puts ahead of
  conda envs).  This gets you the Build / Modify / Watch /
Spectra tabs and all build-time chemistry.

To actually *run* a generated SIESTA / PySCF / Amber input, you need one
or more named backend envs — set those up next.

---

## Why the install is split across several envs

We tried a single env.  Every pair of "tools a user would realistically
combine" has at least one unresolvable conflict:

- AmberTools-dac=26 needs `numpy<2`; gpu4pyscf / cupy-cuda13x needs
  `numpy≥2`.
- AmberTools needs `libnetcdf≥4.10`; `siesta=5.4.2=mpi_openmpi_*` is
  linked against `libnetcdf 4.9.3`.
- AmberTools' X11 stack pins an `icu` version that conda-forge
  `playwright`'s `nodejs` can't accept.

Splitting the tools across named conda envs keeps each backend on its
native pin set; molbuilder dispatches subprocess calls to whichever env
owns a tool (`conda run -n <env> tool ...`).

What ships today (Phase 1 + 2):

- The dispatch helper itself
  (`molbuilder.envs.run_in_env` / `run_tool`).
- The Amber backend wired through it -- `tleap` automatically dispatches
  into `molbuilder-MDtools` whenever you call the AmberTools builders.
- The `molbuilder run <script>` CLI subcommand: takes a generated
  `.fdf` or `.py`, emits a sibling `<basename>.run.sh` that activates
  the right conda env and executes the tool.  You then run that shell
  script yourself (foreground, background, SLURM -- your call).

What remains deferred (a future Phase 3):

- A web-UI **Projects browser** at `/projects` with a tree picker for
  selecting starting geometries from prior calculations.
- A Flask `/api/runs/start` endpoint for one-click "run this script
  from the browser" (the corresponding scientist UX is: open the
  `<basename>.run.sh` from a terminal, foreground or `&`).

Both Phase 1 and Phase 2 designs are captured in `docs/design.md`'s
decisions log (2026-05-14 entries).

---

## The envs

| Env | What it contains | Used for |
|---|---|---|
| **host env** *(you name it; we recommend `molbuilder`)* | flask + click + numpy + ase + sisl + rdkit + openbabel + biopython + PeptideBuilder + plotly | Running `python -m molbuilder ...`; build-time chemistry; web UI |
| `molbuilder-siesta` | siesta-MPI + openmpi + scalapack + netcdf-fortran | Running SIESTA jobs |
| `molbuilder-pySCF` | pyscf + (optional) gpu4pyscf + CUDA 13 toolkit | Running PySCF / Spectra jobs |
| `molbuilder-MDtools` | ambertools-dac=26 (from `dacase` channel) | Running tleap / parmchk2 / RESP / antechamber |
| `molbuilder-tests` | playwright + pytest-playwright + Chromium | Running browser E2E tests |

Notes:

- Env names are **case-sensitive** (`molbuilder-pySCF`, not
  `molbuilder-pyscf`).  Future subprocess-dispatch code will match by
  these exact names.
- The host env's name is your call; the four backend names are canonical.
- Install only the backend envs you actually need on a given machine —
  a laptop that just builds and previews needs the host alone.

---

## Setup recipes

Run each block from any base shell with `conda` available.

### Host env (required)

```bash
conda create -n molbuilder -c conda-forge -y python=3.12 pip \
    numpy ase sisl \
    rdkit openbabel biopython \
    flask click plotly \
    pytest pyflakes
conda run -n molbuilder python -m pip install PeptideBuilder pubchempy
```

Run molbuilder *from* this env — **do NOT `pip install -e .`**; invoke
the package directly from the repo:

```bash
conda activate molbuilder
cd /path/to/molbuilder
python -m molbuilder --help    # smoke check; lists subcommands
python -m molbuilder serve     # web UI on http://127.0.0.1:5000
```

### `molbuilder-pySCF` — PySCF (CPU, optional GPU)

Drives the Spectra tab's Raman / IR jobs and geometry-optimisation via
geomeTRIC.  CPU-only is fully functional; GPU adds ~20-100× speedup for
medium-sized molecules on RTX 30-series and newer cards.

CPU baseline:

```bash
conda create -n molbuilder-pySCF -c conda-forge -y python=3.12 pip \
    pyscf pyscf-dispersion geometric
conda run -n molbuilder-pySCF python -m pip install pyscf-properties
```

`geomeTRIC` is the geomopt backend molbuilder generates scripts against;
it covers every workflow the project ships (single point, optimisation,
finite-difference Hessian, Spectra-tab Raman/IR).  PySCF also supports
**Berny** as an alternate optimiser (`pyberny` on PyPI), useful mainly
for cross-comparison with Gaussian.  We do NOT install `pyberny` by
default because it is unmaintained and depends on the removed
`pkg_resources` module — if a user picks Berny in a generated script,
the script will warn at run time that the optional dep is missing.  To
install it manually:

```bash
conda run -n molbuilder-pySCF python -m pip install 'setuptools<81' pyberny
```

Add GPU (check first: `nvidia-smi` should report CUDA version ≥ 13):

```bash
conda install -n molbuilder-pySCF -c conda-forge -y \
    'cuda-version=13.*' cuda-nvcc cuda-cudart-dev cuda-nvrtc cuda-cccl
conda run -n molbuilder-pySCF python -m pip install \
    'cupy-cuda13x[ctk]' gpu4pyscf-cuda13x
```

The `[ctk]` extra is load-bearing — it pulls in the matching
nvidia-cublas / cusolver / cusparse / cufft / curand / nvrtc / nvjitlink
runtime wheels.  Without it, `import gpu4pyscf` fails with
`libcublasLt.so not found`.

Verify:

```bash
conda run -n molbuilder-pySCF python -c "
import pyscf, geometric
print(f'pyscf {pyscf.__version__}, geometric {geometric.__version__}')
try:
    import gpu4pyscf, cupy
    name = cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()
    print(f'GPU OK: gpu4pyscf {gpu4pyscf.__version__} on {name}')
except ImportError:
    print('CPU-only build')
"
```

### `molbuilder-siesta` — SIESTA-MPI

Runs DFT and the DFT-side of (future) Transport calculations.

```bash
conda create -n molbuilder-siesta -c conda-forge -y \
    'siesta=5.4.2=mpi_openmpi_*'
```

The build string `=mpi_openmpi_*` is load-bearing — it pins the
real-MPI variant.  The `nompi_*` variant silently runs in serial under
`mpirun` (visible only as duplicate banners; results are unaffected but
wall-time stays single-core).

Verify MPI is real:

```bash
conda run -n molbuilder-siesta bash -lc \
    'mpirun -np 2 siesta --version 2>&1 | grep -c "Executable      : siesta"'
# Expect: 1   (one banner = MPI; 2 = serial run twice under launcher)
```

### `molbuilder-MDtools` — AmberTools

For Amber-side parameterisation: tleap, parmchk2, antechamber, RESP
charges, etc.

```bash
conda create -n molbuilder-MDtools -c dacase -c conda-forge -y python=3.12 \
    'dacase::ambertools-dac=26'
```

The `dacase` channel is David Case's official AmberMD build.  We give
it priority over conda-forge's `ambertools` (which lags at 24.8 with
conflicting pins).  The dacase build is ~950 MB and transitively pulls
rdkit / biopython / scipy / matplotlib / openmpi / mpi4py / parmed /
propka / pdb2pqr, so it's a complete MD toolbox on its own — that's
why we keep it isolated from the host.

Verify:

```bash
conda run -n molbuilder-MDtools tleap -f /dev/null < /dev/null
# Expect: tleap banner + a "source leaprc.protein.ff14SB" hint or similar; exits 0
```

### Customising env names via `molbuilder.json`

The four canonical names above (`molbuilder-siesta`, `molbuilder-pySCF`,
`molbuilder-MDtools`, `molbuilder-tests`) are what molbuilder's
subprocess-dispatch looks up by default.  To use different names —
because you share a machine with other users, want short names, or
maintain multiple parallel envs (e.g. one set per project) — drop a
`./molbuilder.json` at the repo root with an `envs` section:

```json
{
  "envs": {
    "siesta":  "my-siesta-stable",
    "pyscf":   "my-pyscf-cu13",
    "mdtools": "amber26-dac",
    "tests":   "molbuilder-tests"
  }
}
```

The four keys are **categories** (`siesta`, `pyscf`, `mdtools`,
`tests`); values are conda env names.  Any unspecified category falls
back to the documented default.  The file is gitignored — these
per-machine names never enter the repo.

The same `molbuilder.json` also carries TLS cert/key paths for the
HTTPS dev-server (see § "Optional dependencies" → `molbuilder serve`),
so a fully populated file looks like:

```json
{
  "tls":  { "cert": "/etc/letsencrypt/.../fullchain.pem",
            "key":  "/etc/letsencrypt/.../privkey.pem" },
  "envs": { "siesta":  "my-siesta-stable",
            "pyscf":   "my-pyscf-cu13",
            "mdtools": "amber26-dac",
            "tests":   "molbuilder-tests" }
}
```

### `molbuilder-tests` — Playwright E2E

Only needed if you're working on the web UI or fixing a Playwright-test
failure — most development never touches this env.

```bash
conda create -n molbuilder-tests -c conda-forge -y python=3.12 pip \
    playwright pytest
conda run -n molbuilder-tests python -m pip install pytest-playwright
conda run -n molbuilder-tests python -m playwright install chromium
```

---

## Optional dependencies

### 3DNA (canonical DNA helix builder)

3DNA's `fiber` produces idealised Watson-Crick B-form DNA from a
sequence — the cleanest helix backbone of any of the DNA-building paths.
It is **restricted-licence** (non-commercial) and molbuilder does NOT
auto-download it.  Get the v2.4 archive from <https://x3dna.org/> after
registering, extract anywhere on disk, and export from your shell rc so
the host env picks it up:

```bash
export X3DNA=/path/to/x3dna-v2.4
export PATH="$X3DNA/bin:$PATH"
```

Verify (from the activated host env):

```bash
python -c "from molbuilder.builders.backends import available_backends; print(available_backends())"
# Expect: 'threedna': True
```

If 3DNA is missing, molbuilder falls back to RDKit + OpenBabel for DNA
build — same Watson-Crick chemistry, slightly less idealised backbone
torsions.

### Plotly

Already included in the host env recipe above.  The Spectra tab serves
plotly locally from `/vendor/plotly.min.js`, so the chart works without
internet.

---

## End-to-end verification

After all the envs you intend to use are installed, run these from the
**host env**, in the repo root:

```bash
conda activate molbuilder
cd /path/to/molbuilder

# 1. Backends probe (host-side; fast)
python -c "from molbuilder.builders.backends import available_backends; print(available_backends())"
#   Expect: {'threedna': <True if X3DNA exported>, 'amber': False, 'rdkit': True}
#   'amber' reads False on the four-env design — tleap lives in
#   molbuilder-MDtools, which the host's PATH probe doesn't see.
#   This will change once the subprocess-dispatch backend lands.

# 2. Unit + integration tests (~5 min)
python -m pytest tests/ --ignore=tests/spectra/test_smoke.py \
                        --ignore=tests/test_modify_e2e.py -q
#   Expect: ~1300 passed, 7 skipped (amber-gated; unskip automatically
#   when molbuilder-MDtools dispatch ships).

# 3. Spectra smoke tests (~4 min; only if molbuilder-pySCF is installed)
conda run -n molbuilder-pySCF python -m pytest tests/spectra/test_smoke.py -m smoke -q
#   Expect: 6 passed.  Runs PySCF on water + HCl with a small basis set.

# 4. Playwright E2E (only if molbuilder-tests is installed)
conda run -n molbuilder-tests python -m pytest tests/test_modify_e2e.py -q
#   Expect: all pass.
```

---

## Where to go next

- [`design.md`](design.md) — architectural decisions, scientific
  correctness notes, principles.  Read this before non-trivial code
  changes.
- [`tabs/<tab>/spec.md`](tabs/) — per-tab specifications for Build,
  Modify, Watch, Spectra.
- [`protocols/*.md`](protocols/) — wire formats for `.molwatch.log`,
  the `/api/*` endpoints, and the on-disk job-layout convention.
