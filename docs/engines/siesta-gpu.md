# `molbuilder-siesta-gpu` — SIESTA built from source with CUDA-accelerated ELPA

> **Status:** env recipe shipped 2026-06-14. UI GPU toggle + runwrap GPU
> dispatch are follow-ups (not part of this env). Doctor reports the
> env's presence/verify outcome but no engine dispatches into it yet.

This page is the engineering reference for the `molbuilder-siesta-gpu`
env's build stack. It is the operational counterpart to the
2026-06-14 Decisions log entry in [`docs/design.md`](../design.md);
the design log carries the *why*, this page carries the *what + how*.

For human-readable installation prose, see
[`docs/README_install.md`](../README_install.md) § `molbuilder-siesta-gpu`.

---

## 1. The seven locked decisions

1. **Unified Python command + 2 thin bash wrappers** —
   `molbuilder envs install molbuilder-siesta-gpu` does the work.
   `scripts/siesta-gpu-bootstrap.sh` and
   `scripts/siesta-gpu-rebuild.sh` only exist so a user without the
   host env activated has a single entry point.
2. **Artifact root is `$CONDA_PREFIX/opt/siesta-gpu-stack/`** — every
   built lib + binary stays inside the env's tree. Loader rpath +
   activate.d hook bind binaries to *this* env's MPI/OpenMP; outside
   the env they refuse to start. Comparison vs `~/.local/bin` and
   `/usr/local/`: see § 2.
3. **Coexist with `molbuilder-siesta`** — the precompiled CPU env and
   the from-source GPU env install side by side. The UI is the only
   place that picks one at job-submit time (follow-up).
4. **SIESTA pinned to 5.4.2** — same tag the precompiled CPU env uses
   (`siesta=5.4.2=mpi_openmpi_*`), so any CPU↔GPU diff is the GPU
   acceleration only. `MOLBUILDER_SIESTA_TAG` override participates
   in the toolchain fingerprint.
5. **CUDA compute capability auto-detected** via
   `nvidia-smi --query-gpu=compute_cap`. Fallback `sm_80` when no GPU
   is present on the build host. `MOLBUILDER_CUDA_CC` override.
6. **Toolchain pinned** — `gcc_linux-64=14` family (gxx, gfortran),
   `python=3.12`, `cmake>=3.30`, `ninja`, **OpenBLAS** (not MKL),
   `scalapack`, MPI-enabled `fftw` / `hdf5` / `netcdf-fortran`,
   `libxc`, `openmpi` (unpinned).  **CUDA toolkit lives IN the env**
   (`cuda-version=13.*`, `cuda-nvcc`, `cuda-cudart-dev`, `cuda-nvrtc`,
   `cuda-cccl`, `libcublas-dev` — all conda-forge); the host provides
   only the NVIDIA driver + `nvidia-smi` (kernel-module-coupled, not
   a conda package).  Mirrors the `molbuilder-pySCF` env's CUDA
   pattern.
7. **One build serves TranSiesta + TBtrans** — `cmake -DSIESTA_WITH_TRANSIESTA=ON`
   yields three executables in
   `$CONDA_PREFIX/opt/siesta-gpu-stack/siesta/bin/`.

---

## 2. Why `$CONDA_PREFIX/opt`, not `~/.local/bin` or `/usr/local`

| Property | `/usr/local/bin` | `~/.local/bin` | `$CONDA_PREFIX/opt` |
|---|---|---|---|
| Needs sudo | yes | no | no |
| Multi-env coexistence | one global binary | one global binary | per-env, both visible at once |
| `conda env remove` cleanup | leaks | leaks | atomic |
| PATH collision risk | high | medium | none (only on PATH when env active) |
| Loader can find libs outside its env | yes (often wrong libmpi) | yes (often wrong libmpi) | no — rpath + LD_LIBRARY_PATH bound to env |

The user explicitly rejected `/usr/local` (privilege concern) and
`~/.local/bin` (recognised the collision concern after the comparison).
`$CONDA_PREFIX/opt` matches `~/.local/bin` on privilege cost while
giving env-scoped lifecycle for free.

The architectural cost is "recompile after `conda env remove`", which
is acceptable because:

- The sentinel-resume machine means recompile-from-clean-clone is the
  only operation that benefits from the resume table.
- The toolchain fingerprint already invalidates builds whose
  `(gcc, openmpi, cuda, ref)` quadruple has shifted.
- The build runs once per env-creation, not per env-activation.

---

## 3. Artifact layout

```
$CONDA_PREFIX/opt/siesta-gpu-stack/
├── src/
│   ├── elpa/         # git clone of the ELPA repo
│   ├── elsi/         # git clone of the ELSI repo
│   └── siesta/       # git clone of the SIESTA repo
├── build/
│   ├── elpa/         # out-of-tree cmake build dir
│   ├── elsi/
│   └── siesta/
├── elpa/             # install prefix for ELPA (lib + include)
├── elsi/             # install prefix for ELSI
├── siesta/
│   └── bin/
│       ├── siesta
│       ├── transiesta
│       └── tbtrans
├── logs/
│   ├── elpa.configure.log
│   ├── elpa.build.log
│   └── ...
├── .sentinels/
│   ├── elpa.clone.done
│   ├── elpa.configure.done
│   ├── elpa.build.done
│   ├── elpa.install.done
│   ├── elsi.<phase>.done
│   └── siesta.<phase>.done
└── .toolchain-fingerprint   # SHA256 of (gcc, openmpi, cuda, refs, CC)
```

Conda's `etc/conda/activate.d/zz-siesta-gpu-stack.sh` prepends
`$CONDA_PREFIX/opt/siesta-gpu-stack/siesta/bin` to `PATH` and the
elpa+elsi lib dirs + `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH`.  The
mirror in `etc/conda/deactivate.d/zz-siesta-gpu-stack.sh` reverses it.

Why `$CONDA_PREFIX/lib` on `LD_LIBRARY_PATH`: that's where the
conda-installed `libcudart` / `libcublas` / `libmpi` / `libgomp`
live.  Conda does not add this dir to the loader path by default
(a well-known conda gotcha), so our hook publishes it explicitly.

Why no `CUDA_HOME` export from our hook: the conda-forge `cuda-nvcc`
package has its own activate.d hook that exports `CUDA_HOME` pointing
at `$CONDA_PREFIX`.  Ours would conflict.

---

## 4. Build phases

Each component goes through five phases; each phase writes a sentinel
file under `.sentinels/` carrying the toolchain fingerprint:

| Phase | What it does | Sentinel file |
|---|---|---|
| clone | `git clone <repo> src/<comp>; git checkout <ref>` | `<comp>.clone.done` |
| configure | `cmake -S src/<comp> -B build/<comp> <flags>` | `<comp>.configure.done` |
| build | `cmake --build build/<comp> -j<jobs>` | `<comp>.build.done` |
| install | `cmake --install build/<comp> --prefix opt/.../<comp>` | `<comp>.install.done` |
| verify | per-component smoke check (e.g. `elpa_test`, `siesta --version`) | `<comp>.verify.done` |

Component dependency order: **elpa → elsi → siesta**. ELSI links
ELPA's install dir at configure; SIESTA links ELSI's at configure.

Resume rule: a phase is skipped only if its sentinel exists AND the
fingerprint inside matches the current toolchain fingerprint. Any
mismatch (gcc bumped, openmpi rebuilt, CUDA upgraded,
`MOLBUILDER_SIESTA_TAG` changed) invalidates all downstream sentinels
and forces a rebuild.

---

## 5. Toolchain fingerprint

The fingerprint is the SHA256 of a sorted JSON record:

```json
{
  "cuda_version": "12.4.131",
  "cuda_cc": "sm_80",
  "gcc_version": "14.2.0",
  "openmpi_version": "5.0.5",
  "components": {
    "elpa":  {"repo": "https://gitlab.mpcdf.mpg.de/elpa/elpa.git", "ref": "<tag>"},
    "elsi":  {"repo": "https://github.com/ElectronicStructureLibrary/elsi-interface.git", "ref": "<tag>"},
    "siesta":{"repo": "https://gitlab.com/siesta-project/siesta.git", "ref": "5.4.2"}
  }
}
```

The hash is written into every sentinel file when the phase completes
and compared on the next install run. Identical fingerprint ⇒ skip;
mismatch ⇒ wipe sentinels (downstream of the changed component) +
re-run from the changed phase.

---

## 6. CUDA ↔ gcc compatibility matrix

The recipe's default `cuda-version=13.*` + `gcc_linux-64=14` pair is
known-good.  The pre-flight refuses to start if a user's override
breaks the pairing:

| Detected CUDA (in env) | gcc 14 (default) | Recommended override |
|---|---|---|
| 13.x | OK | — |
| 12.8 – 12.9 | OK | — |
| 12.0 – 12.7 | refused | `MOLBUILDER_GCC=13` |
| 11.x | refused | `MOLBUILDER_GCC=11` (would require pinning `cuda-version=11.*` too) |

CUDA is detected at `$CONDA_PREFIX/bin/nvcc` after `conda create`
finishes.  If you override `cuda-version` in conda_packages, also
override `MOLBUILDER_GCC` to a compatible value.  The error message
names the override env var; re-run
`molbuilder envs install molbuilder-siesta-gpu` with the override
prepended.

---

## 7. CMake flags (per component)

These are the recipe-pinned flags; they live in
`molbuilder/envs/recipes.py` as part of `_SIESTA_GPU.build_spec` and
are listed here for cross-reference.

### ELPA

```
-DENABLE_NVIDIA_GPU=ON
-DCMAKE_CUDA_ARCHITECTURES=<auto-detected CC>
-DENABLE_OPENMP=ON
-DBUILD_SHARED_LIBS=ON
-DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX/opt/siesta-gpu-stack/elpa
-DCMAKE_BUILD_TYPE=Release
```

### ELSI

```
-DENABLE_PEXSI=OFF
-DENABLE_SIPS=OFF
-DELPA_DIR=$CONDA_PREFIX/opt/siesta-gpu-stack/elpa
-DBUILD_SHARED_LIBS=ON
-DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX/opt/siesta-gpu-stack/elsi
-DCMAKE_BUILD_TYPE=Release
```

### SIESTA

```
-DSIESTA_WITH_TRANSIESTA=ON
-DSIESTA_WITH_ELSI=ON
-DELSI_ROOT=$CONDA_PREFIX/opt/siesta-gpu-stack/elsi
-DSIESTA_WITH_LIBXC=ON
-DSIESTA_WITH_NETCDF=ON
-DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX/opt/siesta-gpu-stack/siesta
-DCMAKE_BUILD_TYPE=Release
```

SIESTA itself never sees a `CUDA_*` cmake flag — GPU acceleration is
entirely a function of ELPA being CUDA-enabled and ELSI dispatching
through it. The user opts in at runtime via `.fdf`:

```
Diag.Algorithm        elpa
Diag.ELPA.GPU         T
```

---

## 7a. Build-env isolation from system compilers / MPI / CUDA

Three independent layers keep the build from accidentally pulling in
the user's system MPI / CUDA / compilers, even when those are
installed (e.g. via `apt install libopenmpi-dev`).

### Layer 1: subprocess env sanitizer

`builds.build_subprocess_env()` strips every known leakage vector
from `os.environ` before invoking `conda run`:

| Category | Stripped vars |
|---|---|
| Linker / loader paths | `LD_LIBRARY_PATH`, `LIBRARY_PATH`, `LD_RUN_PATH`, `LD_PRELOAD`, `DYLD_*` |
| Header search paths | `CPATH`, `C_INCLUDE_PATH`, `CPLUS_INCLUDE_PATH`, `OBJC*_INCLUDE_PATH` |
| pkg-config / cmake | `PKG_CONFIG_PATH`, `PKG_CONFIG_LIBDIR`, `CMAKE_PREFIX_PATH`, `CMAKE_INCLUDE_PATH`, `CMAKE_LIBRARY_PATH`, `CMAKE_MODULE_PATH` |
| Compiler driver flags | `CFLAGS`, `CXXFLAGS`, `FFLAGS`, `LDFLAGS`, `CPPFLAGS`, `ASFLAGS`, … |
| Compiler binaries | `CC`, `CXX`, `FC`, `F77`, `F90`, `AR`, `LD`, … (let conda activate.d set these) |
| MPI overrides | `MPI_HOME`, `MPI_ROOT`, `MPICC`, `MPI_INCLUDE`, … |
| CUDA overrides | `CUDA_HOME`, `CUDA_PATH`, `NVCC`, `CUDAToolkit_ROOT`, … |
| Math + IO overrides | `BLAS`, `LAPACK_LIBS`, `MKLROOT`, `FFTW_ROOT`, `HDF5_DIR`, `NETCDF_ROOT`, `SCALAPACK_ROOT`, `LIBXC_ROOT` |
| MPI runtime families | `OMPI_*`, `OPAL_*`, `MPICH_*`, `HYDRA_*`, `I_MPI_*`, `PMI_*`, `PMIX_*`, `SLURM_*` |

Conda's own activate.d hooks fire on top of this clean slate and set
the env-pinned `CC` / `CXX` / `FC` / `CFLAGS` / `LDFLAGS` etc.

### Layer 2: explicit cmake pins

Even if `PATH` ordering wavers (e.g. user has `/usr/bin/mpicc` and
`conda run` doesn't quite prepend), `_PIN_ENV_TOOLS` in
`recipes.py` makes the cmake CLI bypass `FindMPI` / `FindCUDA`'s
PATH walk entirely:

```
-DCMAKE_PREFIX_PATH={env_prefix}
-DMPI_C_COMPILER={env_prefix}/bin/mpicc
-DMPI_CXX_COMPILER={env_prefix}/bin/mpicxx
-DMPI_Fortran_COMPILER={env_prefix}/bin/mpifort
# ELPA only (CUDA):
-DCMAKE_CUDA_COMPILER={env_prefix}/bin/nvcc
-DCUDAToolkit_ROOT={env_prefix}
```

`{env_prefix}` resolves to `$CONDA_PREFIX` at install time.

### Layer 3: `$ORIGIN`-relative install rpath

Each binary bakes a rpath that lets the runtime loader find its
sibling libraries WITHOUT relying on `LD_LIBRARY_PATH`:

| Component | Install rpath |
|---|---|
| libelpa.so | `$ORIGIN/../../../../lib` → `$CONDA_PREFIX/lib` (libcudart, libmpi, libgomp) |
| libelsi.so | `$ORIGIN/../../../../lib:$ORIGIN/../../elpa/lib` |
| siesta / transiesta / tbtrans | `$ORIGIN/../../../../lib:$ORIGIN/../../elsi/lib:$ORIGIN/../../elpa/lib` |

`$ORIGIN` is the directory containing the loaded binary; the loader
resolves it at runtime. `$ORIGIN`-relative means the env is movable
(rename, clone, copy under another prefix — all work).

Combined: even if the user runs the binary WITHOUT
`conda activate molbuilder-siesta-gpu`, the rpath finds the right libs
relative to the binary's own location. The activate.d hook is then a
*belt + suspenders* convenience that also publishes the binary on
`PATH`.

---

## 8. Single-runtime OpenMP rule

The env never mixes OpenMP runtimes. gcc 14 brings `libgomp`; every
linked library (OpenBLAS, ScaLAPACK, FFTW, ELPA, ELSI, SIESTA) must
use `libgomp`. The conda package list explicitly excludes:

- `mkl`, `mkl-devel`, `mkl_fft`, `mkl_random`, `mkl_*`
- `intel-openmp` (libiomp5 source)
- `fftw=*=mkl_*`, `scalapack=*=mkl_*`

A regression test (`test_envs_siesta_gpu_recipe.py::test_no_mkl_variants_in_recipe`)
fails if any future edit re-introduces these.

---

## 9. Build concurrency

Default: `min($(nproc), 8)`. ELPA's template-instantiated `.cpp` files
are memory-hungry; on a 64-core box without a cap, gcc can run out of
RAM mid-build. Override with `MOLBUILDER_BUILD_JOBS=<N>`.

---

## 10. Verify

After install, the recipe runs:

```
$CONDA_PREFIX/opt/siesta-gpu-stack/siesta/bin/siesta --version
```

The check expects `siesta 5.4.2` in stdout. A real GPU-functional
smoke test (small Au-BDT-Au fdf with `Diag.ELPA.GPU T`) is a
follow-up; this verify is "the binary linked and starts", not "the
GPU path runs."

---

## 11. Follow-ups (not in this env work)

- **UI GPU toggle** — checkbox in SIESTA form, `workflow_group="budget"`.
  Flips `cfg.gpu = true` and propagates to `.fdf` emission.
- **runwrap.py GPU dispatch** — when the cfg / sidecar says `gpu: true`,
  dispatch into `molbuilder-siesta-gpu` instead of `molbuilder-siesta`.
- **`.fdf` GPU lines** — emitter writes `Diag.Algorithm elpa` +
  `Diag.ELPA.GPU T` only when `cfg.gpu == True`.
- **Au-BDT-Au GPU validation fixture** — extend
  `test_transport_au_bdt_au_validation.py` with a GPU smoke test
  (factor-of-2 T(E_F) match vs Reed 2006).
