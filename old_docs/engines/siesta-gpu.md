# `molbuilder-siesta-gpu` — SIESTA built from source with CUDA-accelerated ELPA

> **Engineering reference** for the GPU SIESTA env's build stack.
> Operational counterpart to the 2026-06-14/15 Decisions log entries
> in [`docs/design.md`](../design.md): the design log carries the
> *why*, this page carries the *what + how*.
>
> For user-facing install prose see
> [`docs/README_install.md`](../README_install.md) §
> `molbuilder-siesta-gpu`.

---

## 1. Target deployment

**One workstation with multi-CPU and an NVIDIA GPU.**  Not a cluster,
not a cloud instance, not a container.  MPI is used for
intra-workstation parallelism — `mpirun -np 8` across local cores or
NUMA nodes — never for distributed jobs across nodes.  The conda
env, the build artifacts, the molbuilder app, and SIESTA all run on
the same physical machine.

The build's only off-machine dependencies are read-only `git clone`s
from upstream gitlab/github at install time:

| Upstream | What we clone | When |
|---|---|---|
| `gitlab.mpcdf.mpg.de/elpa/elpa.git` | ELPA source (CUDA-enabled eigensolver) | install + `--rebuild=elpa` |
| `gitlab.com/siesta-project/siesta.git` | SIESTA + all `External/` submodules | install + `--rebuild=siesta` or `--rebuild=all` |

Everything else (compilers, MPI, BLAS, LAPACK, ScaLAPACK, NetCDF,
HDF5, FFTW, libxc, the CUDA toolkit) ships as conda-forge packages
inside the env.

---

## 2. Architecture at a glance

```
┌───────────────────────────────────────────────────────────────┐
│                   HOST (workstation)                          │
│  ┌─────────────────────────────────────────────────┐         │
│  │  NVIDIA driver  +  nvidia-smi  +  /dev/nvidia*  │         │
│  │  (kernel-coupled — cannot live in conda)        │         │
│  └─────────────────────────────────────────────────┘         │
│           ↑                                                   │
│           │ libcuda.so loaded at runtime                      │
│           │                                                   │
│  ┌────────┴────────────────────────────────────────┐         │
│  │  conda env  $CONDA_PREFIX/                      │         │
│  │  ──────────────────────────                     │         │
│  │  bin/                                           │         │
│  │    nvcc, mpicc, gcc, gfortran, ...              │         │
│  │  lib/                                           │         │
│  │    libcudart, libcublas, libmpi,                │         │
│  │    libgfortran, libgomp, libopenblas,           │         │
│  │    libscalapack, libnetcdf, libhdf5,            │         │
│  │    libxc, ...                                   │         │
│  │  include/                                       │         │
│  │    cuda/, mpi.h, ...                            │         │
│  │                                                 │         │
│  │  opt/siesta-gpu-stack/  ◄── built by molbuilder │         │
│  │    elpa/                  (CUDA eigensolver)    │         │
│  │      lib/libelpa.so                             │         │
│  │      include/elpa/                              │         │
│  │    siesta/                                      │         │
│  │      bin/{siesta,transiesta,tbtrans}            │         │
│  │      lib/...                                    │         │
│  │    src/, build/, logs/, .sentinels/             │         │
│  └─────────────────────────────────────────────────┘         │
└───────────────────────────────────────────────────────────────┘
```

**Two source-built components, everything else from conda.**

---

## 3. The 2-component decision (literature + documentation supported)

### 3.1 What SIESTA 5.4's INSTALL.md says

Per the canonical SIESTA install doc (fetched 2026-06-15 from
`gitlab.com/siesta-project/siesta` `rel-5.4/INSTALL.md`):

> *"Siesta relies on a number of libraries, some required and some
> optional… you can pre-install (some of) the libraries… or the
> source for (some of) the external libraries can be made available
> in the `External/<package>` subdirectories by activating git
> submodules within the Siesta distribution."*

> *"`<PACKAGE>_FIND_METHOD="cmake;pkgconf;source;fetch"`"* — SIESTA's
> cmake tries (in order): `find_package`, `pkg-config`,
> `External/<package>` (git submodule), then `git clone` from
> upstream.

### 3.2 SIESTA's required + optional libraries

| Lib | Status | Conda-forge? | How we ship it |
|---|---|---|---|
| BLAS | required (system software) | yes (`openblas`) | conda pkg |
| LAPACK | required | yes (`openblas` provides) | conda pkg |
| MPI | highly recommended | yes (`openmpi`) | conda pkg |
| ScaLAPACK | required for MPI | yes (`scalapack`) | conda pkg |
| NetCDF | highly recommended | yes (`netcdf-fortran`) | conda pkg |
| HDF5 | needed by NetCDF parallel | yes (`hdf5=*=mpi_openmpi_*`) | conda pkg |
| FFTW | optional | yes (`fftw=*=mpi_openmpi_*`) | conda pkg |
| OpenMP | recommended | yes (gcc bundles libgomp) | conda pkg |
| **libfdf** | **required (ESL)** | **NO** | SIESTA submodule |
| **libpsml** | **required (ESL)** | **NO** | SIESTA submodule |
| **xmlf90** | **required (ESL)** | **NO** | SIESTA submodule |
| **libgridxc** | **required (ESL)** | **NO** | SIESTA submodule |
| libxc | highly recommended | yes (`libxc`) | conda pkg |
| **ELPA (CUDA-enabled)** | recommended for GPU | only CPU-only on conda-forge | **source-built externally** |
| ELSI | recommended | irrelevant — SIESTA bundles it | SIESTA submodule |

Verified 2026-06-15 via `conda search -c conda-forge`: `libfdf`,
`libpsml`, `xmlf90`, `libgridxc` all return "No match found".

### 3.3 Resulting build plan

```
Component 1: ELPA (external, CUDA-enabled source build)
  ├── git clone  https://gitlab.mpcdf.mpg.de/elpa/elpa.git @ <ELPA_TAG>
  ├── cmake configure with -DENABLE_NVIDIA_GPU=ON
  ├── cmake build
  └── install to $CONDA_PREFIX/opt/siesta-gpu-stack/elpa/

Component 2: SIESTA (with --recurse-submodules)
  ├── git clone --recurse-submodules
  │       https://gitlab.com/siesta-project/siesta.git @ rel-5.4
  │     └─ External/libfdf, libpsml, xmlf90, libgridxc, ELSI-project,
  │        libxc all populated by submodule recursion
  ├── cmake configure with
  │       -DCMAKE_PREFIX_PATH=$env;$elpa
  │       -DSIESTA_WITH_{MPI,ELSI,ELPA,LIBXC,NETCDF,TRANSIESTA}=ON
  │     └─ <PACKAGE>_FIND_METHOD's "source" step picks up External/
  ├── cmake build  (compiles SIESTA + all submodules on the fly)
  └── install to $CONDA_PREFIX/opt/siesta-gpu-stack/siesta/
        ├── bin/{siesta,transiesta,tbtrans}
        └── lib/... (libsiestaXC etc, statically linked ELSI)
```

This is the **literature + doc supported** path: matches SIESTA's
INSTALL.md recommendation, what SIESTA's CI tests, and what the
conda-forge `siesta-feedstock` does (sans the GPU part).

---

## 4. Artifact layout

```
$CONDA_PREFIX/opt/siesta-gpu-stack/
├── src/
│   ├── elpa/                      # git clone (depth=1)
│   └── siesta/                    # git clone (depth=1, recurse-submodules)
│       └── External/              # libfdf, libpsml, xmlf90,
│                                  # libgridxc, ELSI-project, libxc
├── build/
│   ├── elpa/                      # out-of-tree cmake build dir
│   └── siesta/
├── elpa/                          # install prefix for ELPA
│   ├── lib/libelpa.so
│   └── include/elpa/
├── siesta/
│   ├── bin/{siesta,transiesta,tbtrans}
│   └── lib/
├── logs/
│   ├── elpa.{clone,configure,build,install,verify}.log
│   └── siesta.{clone,configure,build,install,verify}.log
├── .sentinels/
│   ├── elpa.{clone,configure,build,install,verify}.done
│   └── siesta.{clone,configure,build,install,verify}.done
└── .toolchain-fingerprint         # SHA256 of (gcc, openmpi, cuda, refs, CC)
```

Conda's auto-managed hook scripts (`etc/conda/activate.d/zz-siesta-gpu-stack.sh`
+ deactivate.d mirror) put SIESTA's `bin/` on `PATH` and ELPA's
`lib/` + the env's own `lib/` on `LD_LIBRARY_PATH`.

---

## 5. Build phases + sentinel resume

Each component runs through five phases:

```
   clone ──► configure ──► build ──► install ──► verify
     │           │           │          │           │
     ▼           ▼           ▼          ▼           ▼
  .sentinels/<comp>.<phase>.done   (per-phase sentinel file)
```

Each sentinel records the toolchain fingerprint at completion time.
On the next install run, the executor's resume logic:

```python
# Pseudocode (real implementation in molbuilder/envs/builds.py)
for comp in spec.components:        # [elpa, siesta]
    for phase in PHASES:            # [clone, configure, build, install, verify]
        sentinel = paths.sentinel(comp.name, phase)
        if sentinel_valid(sentinel, current_fingerprint):
            log(f"[{N}/{TOTAL}] {comp.name}.{phase}: skipped (sentinel valid)")
            continue
        if phase == "configure":
            wipe(paths.build(comp.name))   # stale CMakeCache.txt
        if phase == "clone":
            wipe(paths.src(comp.name))     # partial clone if any
        run_streaming(
            argv=build_argv_for(comp, phase),
            log_file=paths.logs(comp.name, phase),
            sink=sys.stderr,    # tee to user's terminal in real time
        )
        write_sentinel(sentinel, current_fingerprint)
```

**Resume invariants:**

- A sentinel is "valid" iff it exists *and* its recorded fingerprint
  equals the current fingerprint.  Anything else (file missing,
  fingerprint mismatch, JSON corrupt) → re-run the phase.
- `--rebuild=<comp>` wipes the sentinels + build/ + install/ dirs
  for the named component AND everything downstream of it.
  `--rebuild=all` wipes both components.  `src/` clones are
  preserved (saves the re-fetch on slow networks); pass `all` to
  wipe those too.
- A failed phase leaves no sentinel → the next run resumes from
  that phase.

---

## 6. Toolchain fingerprint

The fingerprint is SHA256 over a sorted JSON record:

```json
{
  "cuda_version": "13.0.0",
  "cuda_compute_cap": "8.0",
  "gcc_version": "14.2.0",
  "openmpi_version": "5.0.5",
  "artifact_subdir": "siesta-gpu-stack",
  "omp_runtime": "gomp",
  "components": {
    "elpa":   {"repo": "...elpa.git", "declared_ref": "new_release_2023.05.001",
               "resolved_ref": "<git SHA after clone>"},
    "siesta": {"repo": "...siesta.git", "declared_ref": "rel-5.4",
               "resolved_ref": "<git SHA after clone>"}
  }
}
```

Any change forces relevant phases to re-run:

| What changed | Forces rebuild of |
|---|---|
| `MOLBUILDER_ELPA_TAG` | elpa + downstream (siesta) |
| `MOLBUILDER_SIESTA_TAG` | siesta only |
| `MOLBUILDER_CUDA_VERSION` (conda env rebuild) | both |
| `MOLBUILDER_GCC` (conda env rebuild) | both |
| OpenMPI auto-bump from conda | both |
| New SHA on a tracked branch (after re-clone) | the affected component |
| Toolkit-side patch like nvcc 13.0 → 13.1 | both |

---

## 7. CMake flags (per component)

Both components share the same env-isolation pins:

```
# Env-isolation: bypass FindMPI/FindCUDA PATH walks, pin compilers
# explicitly to the conda env's binaries.
-DMPI_C_COMPILER={env_prefix}/bin/mpicc
-DMPI_CXX_COMPILER={env_prefix}/bin/mpicxx
-DMPI_Fortran_COMPILER={env_prefix}/bin/mpifort

# Runtime: $ORIGIN-relative install rpath so the binary finds its
# libs without LD_LIBRARY_PATH (belt + suspenders with activate.d).
-DCMAKE_INSTALL_RPATH=$ORIGIN/<relative paths>
-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
```

### 7.1 ELPA

```
-DCMAKE_PREFIX_PATH={env_prefix}
-DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX/opt/siesta-gpu-stack/elpa
-DBUILD_SHARED_LIBS=ON
-DCMAKE_INSTALL_RPATH=$ORIGIN/../../../../lib

# CUDA pins (force FindCUDAToolkit to the conda toolkit)
-DCMAKE_CUDA_COMPILER={env_prefix}/bin/nvcc
-DCUDAToolkit_ROOT={env_prefix}
-DENABLE_NVIDIA_GPU=ON
-DCMAKE_CUDA_ARCHITECTURES={cuda_cc_numeric}   # e.g. 80 for A100

# Feature flags
-DENABLE_OPENMP=ON
-DUSE_MPI_MODULE=ON
```

### 7.2 SIESTA

```
-DCMAKE_PREFIX_PATH={env_prefix};{dep_elpa}   # cmake list separator
-DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX/opt/siesta-gpu-stack/siesta
-DCMAKE_INSTALL_RPATH=$ORIGIN/../../../../lib:$ORIGIN/../../elpa/lib

# Feature flags
-DSIESTA_WITH_MPI=ON
-DSIESTA_WITH_TRANSIESTA=ON
-DSIESTA_WITH_ELSI=ON         # found in External/ via submodule
-DSIESTA_WITH_ELPA=ON         # found via {dep_elpa} on CMAKE_PREFIX_PATH
-DSIESTA_WITH_LIBXC=ON
-DSIESTA_WITH_NETCDF=ON
-DSIESTA_WITH_OPENMP=ON
```

SIESTA's own cmake then automatically:

1. Finds ELPA at `{dep_elpa}` via `cmake` find-method
2. Finds OpenBLAS, ScaLAPACK, NetCDF, HDF5, FFTW, libxc at `{env_prefix}` via cmake find-method
3. Compiles libfdf, libpsml, xmlf90, libgridxc, ELSI from
   `External/<package>/` via the `source` find-method (the
   `--recurse-submodules` clone populated those dirs)

No explicit `-DELSI_ROOT` or per-ESL-lib flags needed.

---

## 8. Build-env isolation from system tools

Even if the user has `/usr/bin/mpicc` (from `apt install
libopenmpi-dev`) or `/usr/local/cuda/bin/nvcc`, three independent
defenses keep the build using the env's tools:

### Layer 1 — Subprocess env sanitizer

`builds.build_subprocess_env()` strips ~60 vars + 7 prefix
families from `os.environ` before invoking `conda run`:

```python
# Pseudocode of the sanitizer
LEAKAGE = {
    # Linker / loader
    "LD_LIBRARY_PATH", "LIBRARY_PATH", "LD_RUN_PATH", "LD_PRELOAD",
    # Header search
    "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", ...,
    # pkg-config + cmake
    "PKG_CONFIG_PATH", "CMAKE_PREFIX_PATH", ...,
    # Compiler driver flags
    "CFLAGS", "CXXFLAGS", "FFLAGS", "LDFLAGS", "CPPFLAGS", ...,
    # Compiler binaries (let conda activate.d set fresh)
    "CC", "CXX", "FC", "AR", "LD", ...,
    # MPI / CUDA / BLAS / FFTW / HDF5 / NetCDF location overrides
    "MPI_HOME", "MPI_ROOT", "MPICC", "MPI_INCLUDE", "MPIEXEC", ...,
    "CUDA_HOME", "CUDA_PATH", "NVCC", "CUDAToolkit_ROOT", ...,
    "BLAS", "LAPACK_LIBS", "FFTW_ROOT", "HDF5_ROOT", ...,
}
LEAKAGE_PREFIXES = ("OMPI_", "MPICH_", "I_MPI_",
                    "PMI_", "PMIX_", "SLURM_", "OPAL_")
clean_env = {
    k: v for k, v in os.environ.items()
    if k not in LEAKAGE
    and not any(k.startswith(p) for p in LEAKAGE_PREFIXES)
}
subprocess.run([..., "conda", "run", ...], env=clean_env)
```

Conda's activate.d hooks then set the env-pinned `CC` / `CXX` /
`FC` / `CFLAGS` etc. on top of this clean slate.

### Layer 2 — Explicit cmake compiler pins

`-DMPI_C_COMPILER={env_prefix}/bin/mpicc` etc. (see § 7).  These
bypass FindMPI / FindCUDAToolkit PATH walks entirely.  Even if `PATH`
order wavers, cmake can't pick up a different mpicc / nvcc.

### Layer 3 — `$ORIGIN`-relative install rpath

Each binary bakes a runtime library search path relative to the
binary's own location:

| Binary | Install rpath |
|---|---|
| `libelpa.so` | `$ORIGIN/../../../../lib` → `$CONDA_PREFIX/lib` (libcudart, libmpi, libgomp) |
| `siesta`, `transiesta`, `tbtrans` | `$ORIGIN/../../../../lib:$ORIGIN/../../elpa/lib` |

`$ORIGIN` is the directory containing the loaded binary; the loader
resolves it at runtime.  `$ORIGIN`-relative means the env stays
movable — rename, clone, or copy under another prefix all work,
unlike absolute baked-in paths.

Combined: even if the user runs `siesta` *without*
`conda activate molbuilder-siesta-gpu` (e.g. via an explicit path),
the rpath finds the right libs relative to the binary's location.
The activate.d hook is then a *belt + suspenders* convenience that
also publishes the binary on `PATH`.

---

## 9. CUDA ↔ gcc compatibility matrix

Pre-flight refuses to start if `cuda-version` and `gcc_linux-64`
don't pair.  Per NVIDIA's compatibility tables:

| CUDA toolkit (in env) | gcc 14 (default) | Recommended override |
|---|---|---|
| 13.x | OK | — |
| 12.8 – 12.9 | OK | — |
| 12.0 – 12.7 | refused | `MOLBUILDER_GCC=13` |
| 11.x | refused | `MOLBUILDER_GCC=11` (and `MOLBUILDER_CUDA_VERSION=11.*`) |

Error message names the override variable; user re-runs with the
override prepended.

---

## 10. Single-runtime OpenMP rule

The env stays single-OpenMP-runtime (`libgomp` only).  gcc 14
provides `libgomp`; MKL provides `libiomp5`.  Mixing both crashes
with `OMP: Error #15: Initializing libiomp5.so...`.  The recipe's
`forbidden_packages` lists block:

- `mkl`, `mkl-devel`, `mkl-include`, `mkl-service`, `mkl_fft`,
  `mkl_random`
- `intel-openmp`
- `fftw=*=mkl_*`, `scalapack=*=mkl_*`

Pinned by `tests/test_envs_siesta_gpu_recipe.py::test_no_forbidden_packages`.

---

## 11. Env-var overrides (workstation-customisable)

Every version / tag / repo URL has a `MOLBUILDER_*` override; all
defaults are the **investigated stable values** confirmed against
SIESTA docs + upstream `ls-remote` (2026-06-15).

| Variable | Default | Source of default |
|---|---|---|
| `MOLBUILDER_ELPA_TAG` | `new_release_2023.05.001` | MPCDF ls-remote — May 2023 stable, mature CUDA path |
| `MOLBUILDER_ELPA_REPO` | `https://gitlab.mpcdf.mpg.de/elpa/elpa.git` | Canonical MPCDF upstream |
| `MOLBUILDER_SIESTA_TAG` | `rel-5.4` | gitlab.com/siesta-project — branch (no numeric 5.x tag exists) |
| `MOLBUILDER_SIESTA_REPO` | `https://gitlab.com/siesta-project/siesta.git` | Canonical upstream |
| `MOLBUILDER_CUDA_VERSION` | `13.*` | Matches `molbuilder-pySCF`; pairs with gcc 14 |
| `MOLBUILDER_GCC` | `14` | conda-forge gcc 14 toolchain, CUDA 13 compatible |
| `MOLBUILDER_LIBXC_VERSION` | unpinned | conda's SAT solver picks the newest compatible |
| `MOLBUILDER_CUDA_CC` | auto-detect via `nvidia-smi --query-gpu=compute_cap` | Auto-fallback to `8.0` if no GPU on build host |
| `MOLBUILDER_BUILD_JOBS` | `min(nproc, 8)` | Caps OOM risk during ELPA's heavy template instantiations |

All overrides participate in the toolchain fingerprint, so changing
any one triggers the relevant rebuild.

---

## 12. Trade-off accepted: NetCDF S3 + AWS C SDK transitive deps

conda-forge's `netcdf-c` is built with S3 backend support enabled
by default (so NetCDF can read files from `s3://` URLs).  This pulls
~10 MB of AWS C SDK packages as transitive deps:

```
recipe asks for:  netcdf-fortran=*=mpi_openmpi_*
       │ requires
       ▼
                  netcdf-c                  (conda-forge default = S3-enabled)
       │ links
       ▼
                  libcurl, aws-c-auth, aws-c-cal, aws-c-s3,
                  aws-c-io, aws-c-common, aws-c-compression,
                  aws-c-http, aws-c-sdkutils, aws-c-event-stream,
                  aws-crt-cpp, aws-checksums
```

**This does not change the deployment target.**  These packages are
dormant on a workstation reading local `.nc` files — the AWS C SDK
code paths only activate if NetCDF is asked to open an `s3://` URL,
which never happens on a local workflow.

**Alternative paths considered + rejected:**

| Option | Why rejected |
|---|---|
| `nos3` build variant of `netcdf-c` | None exists on conda-forge (verified 2026-06-15) |
| Drop SIESTA's NetCDF backend (`-DSIESTA_WITH_NETCDF=OFF`) | TBtrans loses parallel NetCDF I/O |
| Source-build `netcdf-c` with `--disable-s3` | Substantial complication, plus would need to rebuild everything downstream of netcdf-c |

**Accepted:** ~10 MB of dormant disk space.

---

## 13. Verify

After install, the recipe's verify step runs:

```bash
$CONDA_PREFIX/opt/siesta-gpu-stack/siesta/bin/siesta --version
```

Expected: a banner containing `siesta` and the 5.4.x version line.

A real GPU-functional smoke test (small Au-BDT-Au fdf with
`Diag.ELPA.GPU T`) is a follow-up; this verify is *"the binary
linked and starts"*, not *"the GPU path runs"*.

---

## 14. Follow-ups (not in this env work)

- **UI GPU toggle** — checkbox in SIESTA form, `workflow_group="budget"`.
  Flips `cfg.gpu = true` and propagates to `.fdf` emission.
- **runwrap.py GPU dispatch** — when the cfg / sidecar says
  `gpu: true`, dispatch into `molbuilder-siesta-gpu` instead of
  `molbuilder-siesta`.
- **`.fdf` GPU lines** — emitter writes `Diag.Algorithm elpa` +
  `Diag.ELPA.GPU T` only when `cfg.gpu == True`.
- **Au-BDT-Au GPU validation fixture** — extend
  `test_transport_au_bdt_au_validation.py` with a GPU smoke test
  (factor-of-2 T(E_F) match vs Reed 2006).

---

## 15. Where the live contracts live

| Surface | Sole source of truth |
|---|---|
| Recipe definition (conda pkgs, BuildSpec) | `molbuilder/envs/recipes.py` |
| Source-build executor (clone/configure/build/install/verify) | `molbuilder/envs/builds.py` |
| Env-var overrides + their defaults | `molbuilder/envs/recipes.py` § "Env-var overrides" |
| Activate / deactivate hook templates | `molbuilder/envs/recipes.py` constants `_SIESTA_GPU_ACTIVATE_HOOK` / `_DEACTIVATE_HOOK` |
| Shape contract tests | `tests/test_envs_siesta_gpu_recipe.py` |
| Executor machinery tests | `tests/test_envs_builds.py` |
| Top-level decisions log | `docs/design.md` (search "SIESTA-GPU") |
