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
    flask click plotly authlib python-cas \
    pytest pyflakes
conda run -n molbuilder python -m pip install PeptideBuilder pubchempy

conda activate molbuilder
python -m molbuilder serve
```

Open <http://127.0.0.1:5000>.

### Conda, mamba, or micromamba — molbuilder accepts any of the three

Every command above that says `conda` also works with `mamba` (faster
solver, especially useful on HPC clusters with slow filesystems) or
`micromamba` (statically-linked single binary, no Miniconda install
required — the only realistic option on locked-down clusters where you
lack admin rights).  molbuilder autodetects the available manager and
uses whichever is on PATH; the preference order is
**mamba > micromamba > conda**.  Detection falls back to `$MAMBA_EXE`
and `$CONDA_EXE` if PATH search fails.

For one-command first-run on a fresh machine (HPC cluster, new
workstation), use the bootstrap script:

```bash
bash scripts/install-env.sh --bootstrap --yes
```

This creates every conda-only env (host + pyscf + siesta + MDtools +
tests) in one pass, then runs `molbuilder envs doctor` for a smoke
check.  Pass `--include-source-builds` to also build the GPU-enabled
SIESTA stack (~30-45 min).  Idempotent: re-running skips envs that
are already present.

`authlib` and `python-cas` are only loaded when `molbuilder.json`
has an `auth` section configured.  `authlib` powers the OAuth/OIDC
backends (Google, GitHub, Microsoft, ORCID); `python-cas` powers
the Apereo CAS backend (e.g. for ASURITE sign-in).  Installing them
upfront means a later flip to auth-on is one config edit, not a
conda re-install.  See [`docs/deployment.md`](deployment.md) § 2a
for the per-backend setup walkthroughs.

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
| `molbuilder-siesta` | siesta-MPI + openmpi + scalapack + netcdf-fortran | Running SIESTA jobs (CPU, precompiled from conda) |
| `molbuilder-siesta-gpu` | gcc14 + cmake + openmpi + libs, then built-from-source ELPA + ELSI + SIESTA 5.4.2 | Running SIESTA jobs (GPU, ELPA-CUDA accelerated); coexists with `molbuilder-siesta` |
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

Once the host env exists, every other env is also installable through
the CLI (single entry point + machine-readable recipe registry):

```bash
# From any shell with conda available:
bash scripts/install-env.sh --list                  # show all recipes
bash scripts/install-env.sh --doctor                # full health report
bash scripts/install-env.sh --dry-run molbuilder-siesta   # print the plan
bash scripts/install-env.sh molbuilder-siesta             # install it

# From inside the activated host env:
python -m molbuilder envs list
python -m molbuilder envs doctor
python -m molbuilder envs install molbuilder-siesta --dry-run
python -m molbuilder envs install molbuilder-siesta
```

The CLI reads recipes from `molbuilder/envs/recipes.py`; the prose
blocks below remain the human-readable source of truth, and a
consistency test (`tests/test_envs_readme_consistency.py`) asserts
the two stay in sync.  Install is idempotent: re-running picks up
new pip dependencies without disturbing the env.

### Host env (required)

```bash
conda create -n molbuilder -c conda-forge -y python=3.12 pip \
    numpy ase sisl \
    rdkit openbabel biopython \
    flask click plotly \
    authlib python-cas \
    pytest pyflakes
conda run -n molbuilder python -m pip install PeptideBuilder pubchempy
```

`authlib` and `python-cas` are the optional sign-in dependencies:

- `authlib` is loaded only when ``molbuilder.json`` has an OAuth
  provider configured (`kind: google | github | microsoft | orcid`).
- `python-cas` is loaded only when an Apereo CAS provider is
  configured (`kind: cas` -- e.g. for ASURITE sign-in).

Both are absent from a no-auth localhost deployment.  Pre-installing
them means a later switch from no-auth → any sign-in mode is a
single ``molbuilder.json`` edit, not an env modification.  See
[`deployment.md`](deployment.md) for the per-backend setup
walkthroughs.

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

GPU support ships in the recipe — `molbuilder envs install
molbuilder-pySCF` auto-pins `cupy-cudaNx[ctk]` and `gpu4pyscf-cudaNx`
to whatever CUDA major your driver reports.  Detection precedence
(see `molbuilder/envs/recipes.py::_resolve_cuda_version`):

1. `MOLBUILDER_CUDA_VERSION` env var — explicit override, e.g. `12.*`
2. `nvidia-smi` "CUDA Version" line — auto-detect from the host driver
3. `13.*` — project default when no NVIDIA driver is present

The `[ctk]` extra on `cupy` is load-bearing — it pulls in the matching
nvidia-cublas / cusolver / cusparse / cufft / curand / nvrtc / nvjitlink
runtime wheels.  Without it, `import gpu4pyscf` fails with
`libcublasLt.so not found`.

Manual add (only if you're patching an existing env without re-running
the recipe — confirm the wheel tag matches your driver first via
`nvidia-smi | grep "CUDA Version"`):

```bash
# Example for a CUDA-13 host; substitute 12x / 14x to match yours.
conda install -n molbuilder-pySCF -c conda-forge -y \
    'cuda-version=13.*' cuda-nvcc cuda-cudart-dev cuda-nvrtc cuda-cccl
conda run -n molbuilder-pySCF python -m pip install \
    'cupy-cuda13x[ctk]' gpu4pyscf-cuda13x
```

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

### `molbuilder-siesta-gpu` — SIESTA built from source with CUDA-accelerated ELPA

> Source-build env: clones ELPA + ELSI + SIESTA into the env, runs
> cmake, and writes an activate.d hook so the resulting `siesta`
> binary lands on `PATH` only when the env is active.  Coexists with
> `molbuilder-siesta` (precompiled CPU); the UI selects between them
> at job-submit time (follow-up).

Use this when you have an NVIDIA GPU and want SIESTA's eigensolver to
run on it.  Underlying acceleration goes through ELPA's CUDA path;
ELSI dispatches into it; SIESTA links ELSI.  Build time is ~35–45
minutes on 8 cores + broadband.

For the full engineering reference (path layout, sentinel-resume
model, build flags), see [`docs/engines/siesta-gpu.md`](engines/siesta-gpu.md).

**Pre-flight** (the installer enforces these and refuses to start if
any fails):

| Requirement | Why |
|---|---|
| NVIDIA driver + `nvidia-smi` on the host | Kernel-module-coupled; can't be a conda package.  Used for runtime + auto-detecting compute capability at build time. |
| Driver supports CUDA runtime ≥ 13 | The toolkit (`cuda-version=13.*`) installs into the env and needs a driver new enough to load `libcuda.so` at runtime. |
| ~30 GB free disk under `$CONDA_PREFIX` | clones + build dirs + install (~12 GB final) |
| `git` can reach `gitlab.mpcdf.mpg.de`, `github.com`, `gitlab.com` | source clones |

The **CUDA toolkit itself** (nvcc, libcudart, libcublas, …) is
NOT a host requirement.  It ships with the env, installed from
conda-forge alongside gcc + cmake + openmpi (mirrors the
`molbuilder-pySCF` env's pattern).  You do not need
`/usr/local/cuda` on the host.

Install:

```bash
# Single-entrypoint:
bash scripts/siesta-gpu-bootstrap.sh
# or
bash scripts/install-env.sh molbuilder-siesta-gpu
# or, from inside the host env:
python -m molbuilder envs install molbuilder-siesta-gpu
```

The installer prints a detailed plan (per-component clone/configure/
build/install with cost estimates) + a preflight report (detected
CUDA version, GPU compute capability, gcc, OpenMPI, disk free, git
reachability) and asks for confirmation before running.  Add `--yes`
(or `-y`) for non-interactive runs.  Add `--skip-network-check` when
your firewall blocks `git ls-remote` but allows `clone`.

Preview without committing:

```bash
bash scripts/siesta-gpu-bootstrap.sh --dry-run
python -m molbuilder envs install --dry-run molbuilder-siesta-gpu
```

Rebuild a single component (after fixing a patch or bumping a tag via
`MOLBUILDER_SIESTA_TAG=<new-tag>`):

```bash
bash scripts/siesta-gpu-rebuild.sh siesta   # SIESTA only
bash scripts/siesta-gpu-rebuild.sh elsi     # ELSI + SIESTA
bash scripts/siesta-gpu-rebuild.sh elpa     # everything (ELPA->ELSI->SIESTA)
bash scripts/siesta-gpu-rebuild.sh all      # wipe + rebuild from scratch
```

`--rebuild=<comp>` wipes sentinels + the build dir + the install dir
for the named component and everything downstream of it.  The `src/`
clones are preserved to skip the re-fetch on slow networks; pass
`all` to wipe those too.

**Toolchain overrides** (all participate in the sentinel fingerprint
so changing any one triggers the relevant rebuild):

| Env var | Purpose |
|---|---|
| `MOLBUILDER_SIESTA_TAG` | Override SIESTA's pinned tag (default `5.4.2`) |
| `MOLBUILDER_ELPA_TAG` | Override ELPA's pinned tag (default `2024.05.001`) |
| `MOLBUILDER_ELSI_TAG` | Override ELSI's pinned tag (default `v2.11.0`) |
| `MOLBUILDER_CUDA_CC` | Force compute capability (e.g. `8.0`) when `nvidia-smi` is unavailable |
| `MOLBUILDER_BUILD_JOBS` | Cap build concurrency (default `min(nproc, 8)`) |
| `MOLBUILDER_GCC` | Pin a different gcc major (use `13` for CUDA 12.0-12.3; `11` for CUDA 11.x) |

**Verify** (after install + `conda activate molbuilder-siesta-gpu` to
trigger the activate.d hook):

```bash
which siesta
# Expect: $CONDA_PREFIX/opt/siesta-gpu-stack/siesta/bin/siesta
siesta --version
# Expect: a banner containing "siesta" + the 5.4.2 version line
```

To run a real GPU job, enable ELPA's CUDA path in your `.fdf`:

```
Diag.Algorithm   elpa
Diag.ELPA.GPU    T
```

**Coexistence note:**  `molbuilder-siesta-gpu` is fully independent
of `molbuilder-siesta` — the activate.d hook is env-scoped, so the
two binaries are never on `PATH` simultaneously.  `conda env remove
molbuilder-siesta-gpu` cleans up atomically; the precompiled CPU env
is unaffected.

#### Verify the GPU env actually exercises the GPU codepath

After install, run the GPU-specific validator:

```bash
python -m molbuilder envs validate molbuilder-siesta-gpu
```

This runs five probes (~2 min wall-clock):

| Probe | What it catches |
|---|---|
| `binary-links` | `siesta` + `tbtrans` + `phtrans` present; `siesta --version` exits 0 |
| `cuda stack` | `nvidia-smi` reports the host driver + `libcuda.so.1` loadable via ctypes |
| `mps daemon` | `nvidia-cuda-mps-control -V` succeeds (the host MPS binary is reachable) |
| `elpa gpu codepath` | Runs ELPA's own GPU validator + greps stderr for the documented silent-CPU-fallback warning string.  Load-bearing — `nvidia-smi` can report a healthy GPU while ELPA silently runs on CPU for every SCF step. |
| `siesta ctest` | SIESTA's bundled `-L simple` ctest set (~90 s) |

If the `mps daemon` probe FAILS with "nvidia-cuda-mps-control not
found on host PATH":

```bash
# Debian / Ubuntu — the MPS binary ships with the NVIDIA driver
# package, NOT as a conda package.
sudo apt install nvidia-cuda-mps
# Or, on rpm-family distros:
sudo dnf install nvidia-driver-cuda
```

The wrapper auto-falls-back to non-MPS multi-rank (mpi_np capped at
2, no Hyper-Q concurrency on the GPU) when MPS is missing, but
multi-rank GPU runs lose most of their speedup.  Worth installing.

If the `elpa gpu codepath` probe FAILS with "silent CPU fallback
warning detected", the ELPA build picked the wrong kernel for your
GPU.  Common cause: a SM_80-specialised kernel built for the wrong
compute capability (the build is supposed to use the SM_80 kernel
ONLY for A100; cc 8.6 / 8.7 / 8.9 / 9.0 should use ELPA's generic
NVIDIA kernel compiled natively for their cc).  Open an issue with
your `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`
output; the recipe auto-detects but the audit trail helps diagnose
edge cases.

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
# Expect: the banner "Welcome to LEaP!" followed by "(no leaprc in
# search path)".  tleap exits 1 (no script to run), which is normal
# and not a failure -- the banner itself proves the env is healthy.
# `molbuilder envs doctor` performs the same check via the
# `verify_ignore_exit_code` flag on the recipe.
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
                        --ignore=tests/test_molbuilder_e2e.py -q
#   Expect: ~1300 passed, 7 skipped (amber-gated; unskip automatically
#   when molbuilder-MDtools dispatch ships).

# 3. Spectra smoke tests (~4 min; only if molbuilder-pySCF is installed)
conda run -n molbuilder-pySCF python -m pytest tests/spectra/test_smoke.py -m smoke -q
#   Expect: 6 passed.  Runs PySCF on water + HCl with a small basis set.

# 4. Playwright E2E (only if molbuilder-tests is installed)
conda run -n molbuilder-tests python -m pytest tests/test_molbuilder_e2e.py -q
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
