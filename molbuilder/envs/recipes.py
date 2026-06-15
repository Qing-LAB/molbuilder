"""Declarative recipes for the conda envs molbuilder dispatches into.

Each :class:`Recipe` describes one env: where its packages come from,
what to install, what to verify, and what category (if any) it serves
in :data:`molbuilder.diagnostics.DEFAULT_ENV_NAMES`.

The registry is the single source of truth for env shape consumed by
the ``molbuilder envs`` CLI (doctor / install / list).  The matching
prose in ``docs/README_install.md`` stays the human-readable doc; a
test (``tests/test_envs_readme_consistency.py``) asserts the two
mention the same env names so they cannot drift silently.

Design choices worth pinning here:

* **The host recipe carries no category.**  ``category=None`` means
  "not a routed env" -- doctor still reports its presence/absence,
  but ``run_tool`` never dispatches into it (the host is where
  molbuilder itself runs).
* **Build-from-source recipes** carry a :class:`BuildSpec` in
  ``Recipe.build_spec``.  When non-``None``,
  :mod:`molbuilder.envs.install` chains into
  :mod:`molbuilder.envs.builds` AFTER conda create + pip + extra_steps
  to clone + cmake + install each component declared by the spec.
  The 2026-06-14 Decisions log entry locks the seven design decisions;
  see :doc:`docs/engines/siesta-gpu` for the engineering reference.
* **No automatic CUDA install.**  GPU runtime libraries that ship in
  conda envs (``cuda-cudart``, ``cupy-cuda13x[ctk]``, etc.) ARE
  install-time concerns and belong in the recipe; the underlying
  NVIDIA driver + base CUDA stack are system responsibility and the
  doctor only verifies their presence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple


# --------------------------------------------------------------------- #
#  Build-from-source primitives                                          #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class BuildComponent:
    """One source-built piece of a multi-component stack.

    Components are listed in :attr:`BuildSpec.components` in dependency
    order; the executor visits them top-to-bottom and each one's
    install must complete before the next is configured.

    Templated string fields support these ``{name}``-style placeholders,
    resolved by :mod:`molbuilder.envs.builds` at execution time:

    * ``{prefix}``    -> ``$CONDA_PREFIX/opt/<artifact_subdir>``
    * ``{src}``       -> ``$CONDA_PREFIX/opt/<artifact_subdir>/src/<name>``
    * ``{build}``     -> ``$CONDA_PREFIX/opt/<artifact_subdir>/build/<name>``
    * ``{install}``   -> ``$CONDA_PREFIX/opt/<artifact_subdir>/<name>``
    * ``{dep_elpa}``  -> ``$CONDA_PREFIX/opt/<artifact_subdir>/elpa``
                         (and similarly for any other component name)
    * ``{cuda_cc}``   -> e.g. ``"sm_80"`` (auto-detected at install time)
    * ``{cuda_home}`` -> system CUDA root, e.g. ``/usr/local/cuda``
    * ``{jobs}``      -> build concurrency (``min(nproc, 8)`` default)

    Attributes
    ----------
    name
        Short identifier, used in artifact paths and sentinel files.
        Must be unique within the BuildSpec.
    repo_url
        Git URL to clone.  Anonymous HTTPS only; ssh:// is rejected
        because conda installs run unattended.
    ref
        Tag, branch, or SHA to ``git checkout``.  Pinned tags give
        deterministic fingerprints; branches re-resolve on every clone
        and the resolved SHA participates in the toolchain fingerprint.
    configure_argv
        Template argv for the cmake configure step (after the
        substitutions above are applied).
    build_argv
        Template argv for the cmake build step.
    install_argv
        Template argv for the cmake install step.
    verify_argv
        Optional template argv for a post-install smoke check.  Empty
        tuple means "skip" (executor still records a verify.done
        sentinel so resume logic works).
    needs_cuda
        When ``True``, this component participates in the CUDA-version
        compatibility pre-flight and the fingerprint records the
        detected CUDA toolkit version.  ELPA needs ``True``; ELSI +
        SIESTA proxy through ELPA so they don't.
    """
    name: str
    repo_url: str
    ref: str
    configure_argv: Tuple[str, ...]
    build_argv: Tuple[str, ...]
    install_argv: Tuple[str, ...]
    verify_argv: Tuple[str, ...] = ()
    needs_cuda: bool = False


@dataclass(frozen=True)
class BuildSpec:
    """Source-build spec attached to a Recipe.

    Attributes
    ----------
    artifact_subdir
        Subdirectory under ``$CONDA_PREFIX/opt/`` where all artifacts
        live.  Must match the recipe's identity (``"siesta-gpu-stack"``
        for the GPU SIESTA env).
    components
        Components in dependency order (earlier ones install before
        later ones configure).  An empty tuple is rejected.
    cuda_required
        When ``True``, the executor's pre-flight fails if ``nvcc`` /
        the CUDA toolkit is not findable on the host.
    cuda_min_version
        Minimum CUDA toolkit version, e.g. ``"12.4"``.  Used by the
        CUDA<->gcc compatibility check.  ``None`` means "any version
        passes the version check, only existence is required."
    forbidden_packages
        Conda spec patterns that MUST NOT appear in the recipe's
        ``conda_packages`` (or be co-installed by the SAT solver as
        dependencies the user adds).  Used to enforce the single-
        OpenMP-runtime rule: ``("mkl*", "intel-openmp", "fftw=*=mkl_*")``
        keeps libgomp the only OpenMP runtime in the env.
    omp_runtime
        Human-readable name of the OpenMP runtime this env expects
        (``"gomp"`` for gcc).  Surfaced by doctor + tested.
    activate_hook
        Template body for ``$CONDA_PREFIX/etc/conda/activate.d/zz-<artifact_subdir>.sh``.
        Receives the same placeholders as component fields.  Empty
        string means "no hook" (rare; almost always needed to publish
        the binary on PATH).
    deactivate_hook
        Template body for ``$CONDA_PREFIX/etc/conda/deactivate.d/zz-<artifact_subdir>.sh``.
        Should reverse exactly what activate_hook did.
    """
    artifact_subdir: str
    components: Tuple[BuildComponent, ...]
    cuda_required: bool = False
    cuda_min_version: Optional[str] = None
    forbidden_packages: Tuple[str, ...] = ()
    omp_runtime: str = "gomp"
    activate_hook: str = ""
    deactivate_hook: str = ""

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("BuildSpec.components must be non-empty")
        names = [c.name for c in self.components]
        if len(set(names)) != len(names):
            raise ValueError(
                f"BuildSpec.components has duplicate names: {names}"
            )
        if "/" in self.artifact_subdir or self.artifact_subdir.startswith("."):
            raise ValueError(
                f"BuildSpec.artifact_subdir {self.artifact_subdir!r} must be "
                f"a single path segment (no slashes, no leading dot)."
            )


# --------------------------------------------------------------------- #
#  Recipe dataclass                                                      #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class Recipe:
    """One conda env's install + verify shape.

    Attributes
    ----------
    name
        Default env name (matches the canonical entry in
        :data:`molbuilder.diagnostics.DEFAULT_ENV_NAMES` when
        ``category`` is set).  A user override via
        ``molbuilder.json`` ``envs`` block does NOT rename the recipe
        -- ``doctor`` resolves the effective env name via the
        capabilities snapshot at report time.
    category
        Routing category from :data:`DEFAULT_ENV_NAMES` (e.g.,
        ``"siesta"``), or ``None`` for the host env (not routed).
    description
        Short one-line user-facing summary, shown by ``envs list``.
    channels
        Conda channels in the order they should be passed via
        ``-c`` flags.  Order matters: the first channel listed has
        highest priority for solving.
    conda_packages
        Conda spec strings (e.g., ``"siesta=5.4.2=mpi_openmpi_*"``).
        Build strings + version pins must match exactly what the
        README documents -- the consistency test catches drift.
    pip_packages
        Pip-installable packages applied AFTER ``conda create``
        succeeds, via ``conda run -n <env> python -m pip install ...``.
        Using ``python -m pip`` (not ``pip`` alone) sidesteps a
        common Ubuntu pitfall where ``~/.local/bin/pip`` precedes the
        env's pip on PATH.
    extra_steps
        Arbitrary shell-command argv tuples to run AFTER pip installs
        but BEFORE the build_spec (if any).  Used for the playwright
        env's ``python -m playwright install chromium`` post-step; each
        is dispatched via ``conda run -n <env> ...``.
    build_spec
        Optional :class:`BuildSpec`.  When non-``None``, the install
        machinery chains into :mod:`molbuilder.envs.builds` after
        extra_steps to clone + cmake + install each component.  Source-
        build envs declare ``conda_packages`` with the build toolchain
        (gcc, cmake, openmpi, libs the build links against) and leave
        the actual scientific binary out of the conda spec -- it's the
        build_spec's output.
    verify_argv
        Argv for the "is this env functional?" probe, dispatched via
        ``conda run -n <env>``.  Should exit 0 on healthy install.
    verify_expect_contains
        Optional substring expected in stdout (or stderr) of the verify
        command.  ``None`` means "exit code zero is enough"; setting
        a string adds a content check on top of the exit-code one.
    verify_ignore_exit_code
        When ``True``, the verify outcome is determined solely by the
        ``verify_expect_contains`` substring (ignoring the process
        exit code).  Required for binaries like ``tleap`` that exit
        non-zero even when started successfully (tleap exits 1 when
        there's no script to run, but the banner proves the env is
        healthy).  Default ``False``: exit code must be 0.
    system_preconditions
        Free-text strings describing host-level requirements (e.g.,
        ``"NVIDIA driver supporting CUDA >= 13"``) that the doctor
        surfaces but does NOT attempt to install.  Per the 2026-06-14
        design call, anything below conda-env scope is system layer.
    """
    name: str
    category: Optional[str]
    description: str
    channels: Tuple[str, ...]
    conda_packages: Tuple[str, ...]
    pip_packages: Tuple[str, ...] = ()
    extra_steps: Tuple[Tuple[str, ...], ...] = ()
    build_spec: Optional[BuildSpec] = None
    verify_argv: Tuple[str, ...] = ()
    verify_expect_contains: Optional[str] = None
    verify_ignore_exit_code: bool = False
    system_preconditions: Tuple[str, ...] = ()


# --------------------------------------------------------------------- #
#  The built-in recipes                                                  #
#                                                                       #
#  Order matches docs/README_install.md § "Setup recipes":              #
#  host, pyscf, siesta, mdtools, tests, siesta-gpu.  Each Recipe's      #
#  fields must match the README block; the consistency test surfaces   #
#  drift on either side as a failure.                                  #
# --------------------------------------------------------------------- #


_HOST = Recipe(
    name="molbuilder",
    category=None,
    description="Host env: runs `python -m molbuilder ...`, build-time "
                "chemistry, and the web UI.",
    channels=("conda-forge",),
    conda_packages=(
        "python=3.12", "pip",
        "numpy", "ase", "sisl",
        "rdkit", "openbabel", "biopython",
        "flask", "click", "plotly",
        "authlib", "python-cas",
        "pytest", "pyflakes",
    ),
    pip_packages=("PeptideBuilder", "pubchempy"),
    verify_argv=("python", "-c",
                 "import ase, sisl, rdkit, flask, click, plotly; "
                 "print('host env OK')"),
    verify_expect_contains="host env OK",
)


_PYSCF = Recipe(
    name="molbuilder-pySCF",
    category="pyscf",
    description="PySCF (CPU + optional GPU runtime libs); Spectra-tab "
                "Raman/IR + geomeTRIC geomopt.",
    channels=("conda-forge",),
    conda_packages=(
        "python=3.12", "pip",
        "pyscf", "pyscf-dispersion", "geometric",
    ),
    pip_packages=("pyscf-properties",),
    verify_argv=("python", "-c",
                 "import pyscf, geometric; "
                 "print(f'pyscf {pyscf.__version__}, "
                 "geometric {geometric.__version__}')"),
    verify_expect_contains="pyscf ",
)


_SIESTA = Recipe(
    name="molbuilder-siesta",
    category="siesta",
    description="SIESTA-MPI: DFT + (future) Transport.",
    channels=("conda-forge",),
    # Build string `=mpi_openmpi_*` is load-bearing -- pins real-MPI
    # variant (the `nompi_*` variant silently runs serial under
    # mpirun).  See README_install.md § "molbuilder-siesta".
    conda_packages=("siesta=5.4.2=mpi_openmpi_*",),
    verify_argv=("siesta", "--version"),
    verify_expect_contains="siesta",
)


_MDTOOLS = Recipe(
    name="molbuilder-MDtools",
    category="mdtools",
    description="AmberTools (tleap, parmchk2, antechamber, RESP, ...).",
    # dacase channel takes priority over conda-forge (which lags at
    # ambertools=24.8 with conflicting pins).
    channels=("dacase", "conda-forge"),
    conda_packages=("python=3.12", "dacase::ambertools-dac=26"),
    # tleap -f /dev/null prints its banner and exits 1 (no script to
    # source); the banner "Welcome to LEaP!" is the proof the binary
    # in this env launched.  See `verify_ignore_exit_code` docstring.
    verify_argv=("bash", "-lc", "tleap -f /dev/null < /dev/null"),
    verify_expect_contains="LEaP",
    verify_ignore_exit_code=True,
)


_TESTS = Recipe(
    name="molbuilder-tests",
    category="tests",
    description="Playwright + pytest-playwright + Chromium "
                "(browser E2E tests only).",
    channels=("conda-forge",),
    conda_packages=("python=3.12", "pip", "playwright", "pytest"),
    pip_packages=("pytest-playwright",),
    # playwright fetches the Chromium binary into the env's cache;
    # without this step the browser tests fail at runtime.
    extra_steps=(
        ("python", "-m", "playwright", "install", "chromium"),
    ),
    verify_argv=("playwright", "--version"),
    verify_expect_contains="Version",
)


# --------------------------------------------------------------------- #
#  molbuilder-siesta-gpu: SIESTA built from source w/ CUDA-enabled ELPA  #
#                                                                       #
#  Companion engineering doc: docs/engines/siesta-gpu.md.  Seven        #
#  design decisions live in docs/design.md 2026-06-14 entry.           #
# --------------------------------------------------------------------- #


# Activate / deactivate hooks use literal ``$CONDA_PREFIX`` so the
# generated script stays valid if the env is cloned or moved -- no
# install-time-baked absolute paths.  ``conda activate <env>`` sets
# ``$CONDA_PREFIX`` before our hook fires, so referencing it directly
# is safe.
#
# Why no CUDA_HOME export here: the CUDA toolkit is a conda package
# (``cuda-nvcc`` + ``cuda-cudart-dev`` + friends) installed INTO this
# env, not a host install at ``/usr/local/cuda``.  The conda-forge
# ``cuda-nvcc`` package provides its own activate.d hook that exports
# CUDA_HOME pointing at ``$CONDA_PREFIX``; ours would conflict.
#
# Why ``$CONDA_PREFIX/lib`` is on LD_LIBRARY_PATH: conda does NOT add
# its env's lib dir to the loader path by default (a well-known
# conda-gotcha).  Our SIESTA binary links libcudart / libcublas /
# libmpi / libgomp from there at runtime, so the hook publishes that
# dir explicitly.
_SIESTA_GPU_ACTIVATE_HOOK = """\
#!/usr/bin/env bash
# molbuilder-siesta-gpu activate hook -- generated, do not edit.
# Publishes the from-source SIESTA stack on PATH + LD_LIBRARY_PATH.

# Idempotency guard: refuse to add a path that's already there.
_mbsg_prepend_path() {
    case ":${PATH}:" in
        *":$1:"*) ;;
        *) export PATH="$1${PATH:+:}${PATH}" ;;
    esac
}
_mbsg_prepend_libpath() {
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$1:"*) ;;
        *) export LD_LIBRARY_PATH="$1${LD_LIBRARY_PATH:+:}${LD_LIBRARY_PATH:-}" ;;
    esac
}

_mbsg_prepend_path     "$CONDA_PREFIX/opt/siesta-gpu-stack/siesta/bin"
_mbsg_prepend_libpath  "$CONDA_PREFIX/opt/siesta-gpu-stack/elpa/lib"
_mbsg_prepend_libpath  "$CONDA_PREFIX/opt/siesta-gpu-stack/elsi/lib"
_mbsg_prepend_libpath  "$CONDA_PREFIX/lib"

export MOLBUILDER_SIESTA_GPU_PREFIX="$CONDA_PREFIX/opt/siesta-gpu-stack"

unset -f _mbsg_prepend_path _mbsg_prepend_libpath
"""


_SIESTA_GPU_DEACTIVATE_HOOK = """\
#!/usr/bin/env bash
# molbuilder-siesta-gpu deactivate hook -- generated, do not edit.

_mbsg_drop_from_path_var() {
    # $1 = var name (PATH / LD_LIBRARY_PATH), $2 = entry to drop
    local var="$1" drop="$2" cur="${!1:-}" out=""
    local IFS=':'
    for p in $cur; do
        [ -z "$p" ] && continue
        [ "$p" = "$drop" ] && continue
        if [ -z "$out" ]; then out="$p"; else out="$out:$p"; fi
    done
    export "$var=$out"
}

_mbsg_drop_from_path_var PATH            "$CONDA_PREFIX/opt/siesta-gpu-stack/siesta/bin"
_mbsg_drop_from_path_var LD_LIBRARY_PATH "$CONDA_PREFIX/opt/siesta-gpu-stack/elpa/lib"
_mbsg_drop_from_path_var LD_LIBRARY_PATH "$CONDA_PREFIX/opt/siesta-gpu-stack/elsi/lib"
_mbsg_drop_from_path_var LD_LIBRARY_PATH "$CONDA_PREFIX/lib"

unset MOLBUILDER_SIESTA_GPU_PREFIX
unset -f _mbsg_drop_from_path_var
"""


# Common cmake flags that pin every build to the conda env's
# compilers + MPI + CUDA + libs.  Without these, cmake's FindMPI /
# FindCUDA / FindBLAS would look at the system first when PATH
# resolution wavers, picking up /usr/bin/mpicc + /usr/local/cuda +
# /usr/lib/x86_64-linux-gnu/openblas instead of the env's pins.
# Concretely defends against the failure mode where a user with
# system openmpi installed via apt would get a build that LINKS
# against system libmpi.so.40 even though the env provides its own.
_PIN_ENV_TOOLS = (
    # cmake's FindXXX modules check CMAKE_PREFIX_PATH first.  Pointing
    # it at the conda env makes every Find* call resolve to env libs.
    "-DCMAKE_PREFIX_PATH={env_prefix}",
    # MPI: bypass FindMPI's PATH walk -- pin compilers explicitly.
    "-DMPI_C_COMPILER={env_prefix}/bin/mpicc",
    "-DMPI_CXX_COMPILER={env_prefix}/bin/mpicxx",
    "-DMPI_Fortran_COMPILER={env_prefix}/bin/mpifort",
)

# CUDA-specific pins (ELPA only).  CUDAToolkit_ROOT + CMAKE_CUDA_COMPILER
# force FindCUDAToolkit to use the conda-installed nvcc + libs and
# refuse to wander into /usr/local/cuda or wherever else.
_PIN_CUDA_TOOLS = (
    "-DCMAKE_CUDA_COMPILER={env_prefix}/bin/nvcc",
    "-DCUDAToolkit_ROOT={env_prefix}",
)

# Install rpath blocks: cmake's RPATH at install time, so the binary
# can find ELPA/ELSI/CUDA/MPI libs at runtime WITHOUT relying on the
# activate.d hook to have set LD_LIBRARY_PATH.  $ORIGIN-relative so
# the env is movable (rename + clone work).
#
# Path math from a binary at <prefix>/<comp>/bin/<binary>:
#   $ORIGIN/../../../../lib            -> $CONDA_PREFIX/lib (cuda/mpi/gomp)
#   $ORIGIN/../../<other>/lib          -> sibling component's lib
#
# Path math from a lib at <prefix>/<comp>/lib/<lib.so>:
#   $ORIGIN/../../../../lib            -> $CONDA_PREFIX/lib
#   $ORIGIN/../../<other>/lib          -> sibling component's lib
# NOTE: no shell escape needed -- we pass cmake argv through
# subprocess.run with list argv (no shell interposed), so $ORIGIN
# is preserved literally and the linker writes it into DT_RUNPATH.
_RPATH_ELPA   = "$ORIGIN/../../../../lib"
_RPATH_ELSI   = "$ORIGIN/../../../../lib:$ORIGIN/../../elpa/lib"
_RPATH_SIESTA_BIN = (
    "$ORIGIN/../../../../lib"
    ":$ORIGIN/../../elsi/lib"
    ":$ORIGIN/../../elpa/lib"
)


_ELPA = BuildComponent(
    name="elpa",
    # MPCDF is the canonical upstream; github mirrors lag.  Override
    # repo + ref via MOLBUILDER_ELPA_REPO / MOLBUILDER_ELPA_TAG.
    repo_url="https://gitlab.mpcdf.mpg.de/elpa/elpa.git",
    # Default ref is a 2024-stable tag.  Users with newer CUDA may
    # want to bump to 2025.* via MOLBUILDER_ELPA_TAG.  The fingerprint
    # records the resolved SHA, so changing the tag forces a rebuild.
    ref="2024.05.001",
    configure_argv=(
        "cmake",
        "-S", "{src}",
        "-B", "{build}",
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX={install}",
        "-DBUILD_SHARED_LIBS=ON",
        # Env-isolation pins (compilers + MPI + CUDA all from env).
        *_PIN_ENV_TOOLS,
        *_PIN_CUDA_TOOLS,
        # $ORIGIN-relative install rpath so libelpa.so at runtime finds
        # libcudart/libmpi/libgomp via env's lib dir without needing
        # LD_LIBRARY_PATH set by the user.
        f"-DCMAKE_INSTALL_RPATH={_RPATH_ELPA}",
        "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
        # CUDA + OpenMP + MPI features
        "-DENABLE_NVIDIA_GPU=ON",
        "-DCMAKE_CUDA_ARCHITECTURES={cuda_cc_numeric}",
        "-DENABLE_OPENMP=ON",
        "-DUSE_MPI_MODULE=ON",
    ),
    build_argv=("cmake", "--build", "{build}", "-j", "{jobs}"),
    install_argv=("cmake", "--install", "{build}"),
    verify_argv=("test", "-f", "{install}/lib/libelpa.so"),
    needs_cuda=True,
)


_ELSI = BuildComponent(
    name="elsi",
    repo_url="https://github.com/ElectronicStructureLibrary/elsi-interface.git",
    ref="v2.11.0",
    configure_argv=(
        "cmake",
        "-S", "{src}",
        "-B", "{build}",
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX={install}",
        "-DBUILD_SHARED_LIBS=ON",
        # Env-isolation pins (compilers + MPI from env; no CUDA here).
        *_PIN_ENV_TOOLS,
        f"-DCMAKE_INSTALL_RPATH={_RPATH_ELSI}",
        "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
        # Use OUR ELPA install dir, not whatever conda-elpa / system-elpa
        # might exist.  No PEXSI / SIPS (would need extra deps).
        "-DENABLE_PEXSI=OFF",
        "-DENABLE_SIPS=OFF",
        "-DUSE_EXTERNAL_ELPA=ON",
        "-DELPA_INCLUDE_DIRS={dep_elpa}/include",
        "-DELPA_LIBRARIES={dep_elpa}/lib/libelpa.so",
    ),
    build_argv=("cmake", "--build", "{build}", "-j", "{jobs}"),
    install_argv=("cmake", "--install", "{build}"),
    verify_argv=("test", "-f", "{install}/lib/libelsi.so"),
    needs_cuda=False,
)


_SIESTA_GPU_COMPONENT = BuildComponent(
    name="siesta",
    repo_url="https://gitlab.com/siesta-project/siesta.git",
    # Matches the precompiled CPU env's pin -- so any CPU<->GPU diff
    # in a downstream comparison is the GPU acceleration only.
    ref="5.4.2",
    configure_argv=(
        "cmake",
        "-S", "{src}",
        "-B", "{build}",
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX={install}",
        # Env-isolation pins (compilers + MPI from env).
        *_PIN_ENV_TOOLS,
        f"-DCMAKE_INSTALL_RPATH={_RPATH_SIESTA_BIN}",
        "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
        # SIESTA features.  ELSI links ELPA (which links CUDA), so we
        # don't need CUDA flags here.
        "-DSIESTA_WITH_TRANSIESTA=ON",
        "-DSIESTA_WITH_ELSI=ON",
        "-DELSI_ROOT={dep_elsi}",
        "-DSIESTA_WITH_LIBXC=ON",
        "-DSIESTA_WITH_NETCDF=ON",
    ),
    build_argv=("cmake", "--build", "{build}", "-j", "{jobs}"),
    install_argv=("cmake", "--install", "{build}"),
    verify_argv=("{install}/bin/siesta", "--version"),
    needs_cuda=False,
)


_SIESTA_GPU_BUILD = BuildSpec(
    artifact_subdir="siesta-gpu-stack",
    components=(_ELPA, _ELSI, _SIESTA_GPU_COMPONENT),
    cuda_required=True,
    cuda_min_version="12.4",
    # Forbids MKL + intel-openmp to keep libgomp the only OpenMP runtime
    # in the env (gcc 14 provides libgomp; libiomp5 from MKL/intel-openmp
    # would collide at runtime with OMP: Error #15).
    forbidden_packages=(
        "mkl", "mkl-devel", "mkl-include", "mkl-service", "mkl_fft",
        "mkl_random", "intel-openmp",
    ),
    omp_runtime="gomp",
    activate_hook=_SIESTA_GPU_ACTIVATE_HOOK,
    deactivate_hook=_SIESTA_GPU_DEACTIVATE_HOOK,
)


_SIESTA_GPU = Recipe(
    name="molbuilder-siesta-gpu",
    category="siesta-gpu",
    description="SIESTA + TranSiesta + TBtrans built from source with "
                "CUDA-enabled ELPA (5.4.2 matches the precompiled CPU env).",
    channels=("conda-forge",),
    # Per the molbuilder design (mirrored by molbuilder-pySCF): the
    # CUDA TOOLKIT lives inside the env (cuda-nvcc + cuda-cudart-dev +
    # libcublas-dev etc., from conda-forge).  The host provides the
    # NVIDIA DRIVER + nvidia-smi (kernel-module-coupled, can't be a
    # conda package).  System CUDA at /usr/local/cuda is no longer
    # consulted by the build.
    conda_packages=(
        # Toolchain
        "python=3.12",
        "gcc_linux-64=14", "gxx_linux-64=14", "gfortran_linux-64=14",
        "cmake>=3.30", "ninja", "make", "git", "m4",
        "pkg-config",
        # CUDA toolkit (mirrors molbuilder-pySCF's pattern).  cuda 13.x
        # ships nvcc compatible with gcc 14; older toolkits would need
        # MOLBUILDER_GCC=13 (or =11 for cuda 11.x).
        "cuda-version=13.*",
        "cuda-nvcc",
        "cuda-cudart-dev",
        "cuda-nvrtc",
        "cuda-cccl",
        "libcublas-dev",
        # MPI (unpinned -- SAT-solver picks one; fingerprint records it)
        "openmpi",
        # Math libs.  OpenBLAS (NOT MKL) keeps libgomp the only OpenMP
        # runtime; mixing libiomp5 + libgomp blows up at runtime.
        "openblas",
        "scalapack",
        # File I/O (parallel HDF5 + netcdf for SIESTA's NetCDF backend)
        "fftw=*=mpi_openmpi_*",
        "hdf5=*=mpi_openmpi_*",
        "netcdf-fortran=*=mpi_openmpi_*",
        # Functional library
        "libxc",
    ),
    build_spec=_SIESTA_GPU_BUILD,
    verify_argv=(
        "bash", "-lc",
        # The activate.d hook puts siesta on PATH; --version exits 0 with
        # the version banner.
        "siesta --version",
    ),
    verify_expect_contains="siesta",
    system_preconditions=(
        "NVIDIA driver + nvidia-smi on the host (toolkit ships in env)",
        "NVIDIA driver supporting CUDA runtime 13.x (driver-side compat)",
        "Internet access for git clone of ELPA / ELSI / SIESTA sources",
        "~30 GB free disk space under $CONDA_PREFIX",
    ),
)


BUILTIN_RECIPES: Tuple[Recipe, ...] = (
    _HOST, _PYSCF, _SIESTA, _MDTOOLS, _TESTS, _SIESTA_GPU,
)


_BY_NAME: Mapping[str, Recipe] = {r.name: r for r in BUILTIN_RECIPES}
_BY_CATEGORY: Mapping[str, Recipe] = {
    r.category: r for r in BUILTIN_RECIPES if r.category is not None
}


def recipe_by_name(name: str) -> Optional[Recipe]:
    """Look up a recipe by its canonical (default) env name."""
    return _BY_NAME.get(name)


def recipe_for_category(category: str) -> Optional[Recipe]:
    """Look up the recipe that serves a routing category."""
    return _BY_CATEGORY.get(category)
