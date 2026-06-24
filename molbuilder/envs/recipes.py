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

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple


# --------------------------------------------------------------------- #
#  Env-var overrides                                                     #
# --------------------------------------------------------------------- #
#
# Every version / tag / repo URL in the source-build recipes is
# overridable via a ``MOLBUILDER_*`` environment variable.  Defaults
# are the **investigated stable values**: ELPA tag verified to exist
# on MPCDF GitLab via ``git ls-remote``, SIESTA branch confirmed as
# the canonical 5.x dev line (no numeric 5.x tags exist upstream),
# version pairings cross-checked against SIESTA 5.4 INSTALL.md.
#
# These read once at module import time and bake into the
# corresponding BuildComponent / Recipe instances.  Changing one
# triggers a rebuild (the value participates in the toolchain
# fingerprint via ``probe_toolchain`` and the resolved git SHA).


def _env_default(var: str, default: str) -> str:
    """Read ``$var`` or return ``default`` when the env var is unset/empty."""
    val = os.environ.get(var)
    return val.strip() if (val and val.strip()) else default


# Source-build refs.  These are the heads of the truth chain for what
# gets cloned + compiled; users can bump any one without editing the
# recipe.
_ELPA_REPO   = _env_default("MOLBUILDER_ELPA_REPO",
                            "https://gitlab.mpcdf.mpg.de/elpa/elpa.git")
_ELPA_TAG    = _env_default("MOLBUILDER_ELPA_TAG",
                            # ELPA version (bare version string, used in
                            # the tarball URL + fingerprint).
                            #
                            # 2026-06-15 BUMP: 2021.11.001 -> 2024.05.001.
                            # Trigger: end-to-end testing on user's
                            # BDT-Au-junction (3924 orbitals, 20 ranks +
                            # MPS on an RTX 3060 Ti) revealed that
                            # 2021.11.001's GPU code path deadlocks SIESTA
                            # after iter 1 -- regardless of
                            # ``Diag.Algorithm`` (1stage vs 2stage).  All
                            # 20 ranks hold live CUDA contexts + spawned
                            # CUDA worker / event-handler threads even
                            # when ELPA's matrix kernel falls back to CPU,
                            # so the hang is in the CUDA cleanup /
                            # finalize path, not the eigensolver itself.
                            # Matches the documented CSCS / Cray-XC50 hang
                            # in CP2K issue #1956 + ELPA upstream issue
                            # #15 (A100/sm_80 kernel-mismatch on
                            # 2021.11.001).
                            #
                            # 2024.05.001 selection (over 2025.06.001 /
                            # 2025.06.002 / 2026.02.001):
                            #
                            # * 2025.06.001 is explicitly marked "should
                            #   NOT be used" on the MPCDF tarball archive.
                            # * 2025.06.002 / 2026.02.001 lack documented
                            #   GPU-stability changelogs; community
                            #   adoption is sparse.
                            # * 2024.05.001 has 2+ years of production
                            #   exposure, patches
                            #   ``cusolverDnXtrtri_bufferSize`` for CUDA
                            #   < 12.1, adds ROCM 6 + AMD Mi300 support,
                            #   and explicitly does not autotune GPU code
                            #   paths when no GPUs are available (helpful
                            #   for our import-time safety net).  BSC's
                            #   MareNostrum5 GPU partition uses an
                            #   adjacent version (2025.006.001) for SIESTA
                            #   per their support knowledge base, but
                            #   their long-term hardening has been on
                            #   2024.05.001.
                            #
                            # Compatibility:
                            # * Configure flags / kernel naming match
                            #   ``--enable-nvidia-gpu`` +
                            #   ``--with-NVIDIA-GPU-compute-capability``
                            #   (unchanged from 2021.11.001).
                            # * SIESTA 5.4.x already handles the
                            #   ``ELPA_2STAGE_REAL_NVIDIA_GPU`` kernel
                            #   naming (per closed SIESTA gitlab #112).
                            # * CUDA 11.x / 12.x / 13.x supported.
                            #
                            # Power users wanting the bleeding edge:
                            #   MOLBUILDER_ELPA_TAG=2026.02.001 \
                            #   MOLBUILDER_ELPA_SHA256=<their-sha256>
                            "2024.05.001")
_ELPA_SHA256 = _env_default("MOLBUILDER_ELPA_SHA256",
                            # SHA256 of elpa-2024.05.001.tar.gz fetched
                            # from MPCDF's tarball archive 2026-06-15.
                            # Recipe verifies this matches before
                            # unpacking.  Empty string skips the check
                            # -- set when bumping the tag via env var
                            # without a known SHA.
                            "9caf41a3e600e2f6f4ce1931bd54185179dade9c1"
                            "71556d0c9b41bbc6940f2f6")
_ELPA_TARBALL_BASE = _env_default(
    "MOLBUILDER_ELPA_TARBALL_BASE",
    # Canonical MPCDF tarball archive.  Used by conda-forge, Spack, and
    # EasyBuild.  Override for institutional mirrors.
    "https://elpa.mpcdf.mpg.de/software/tarball-archive",
)
_SIESTA_REPO = _env_default("MOLBUILDER_SIESTA_REPO",
                            "https://gitlab.com/siesta-project/siesta.git")
_SIESTA_REF  = _env_default("MOLBUILDER_SIESTA_TAG",
                            # Pinned to the 5.4.2 release tag (verified
                            # 2026-06-15 via ls-remote -- bare numeric
                            # tag, no ``v`` prefix, no ``siesta-``
                            # prefix).  ``5.4.2`` matches the version
                            # used by the precompiled CPU env
                            # (molbuilder-siesta), so .fdf input format
                            # and TranSiesta output format stay
                            # identical between the two backends -- a
                            # paper-citable, reproducible default.
                            #
                            # Power users who need a hotfix from the
                            # rel-5.4 branch or a different release can
                            # override:
                            #   MOLBUILDER_SIESTA_TAG=rel-5.4
                            #   MOLBUILDER_SIESTA_TAG=5.4.1
                            #   MOLBUILDER_SIESTA_TAG=<sha>
                            "5.4.2")

# Conda-package version pins.  Empty default = unpinned (conda's SAT
# solver picks the latest compatible).  Override by setting the
# variable to a conda spec fragment (e.g. ``=14`` or ``>=3.30``).
def _spec(name: str, version_var: str = "") -> str:
    """Compose a conda spec ``<name><version>`` or just ``<name>``."""
    return f"{name}{version_var}" if version_var else name


_GCC_VERSION    = _env_default("MOLBUILDER_GCC", "14")          # gcc_linux-64=<N>


# ---- CUDA version resolution: env override > host probe > project default ----
#
# Three-tier precedence so the recipe self-corrects across hardware
# without the user having to remember to set an env var:
#
#   1. ``MOLBUILDER_CUDA_VERSION``  -- explicit user override, wins
#      unconditionally.  Useful for cross-compile / forced downgrades.
#   2. ``nvidia-smi`` probe         -- reads the host driver's reported
#      "CUDA Version" line.  This is the *runtime* CUDA the driver can
#      load (e.g. driver 595.x reports "CUDA Version: 13.2"); we pin
#      the env to that major (``13.*``) so the conda toolkit + the
#      ``cupy-cudaNx[ctk]`` wheels link against a compatible runtime.
#   3. ``"13.*"``                   -- project default if the host has
#      no NVIDIA driver (CPU-only laptop, headless box without a GPU
#      stack).  Pin chosen to match the project's stated 2026 target;
#      change here if the project as a whole moves CUDA major.
#
# Probe is one subprocess call at module import time, with a 2 s
# timeout and graceful fallback -- it never blocks the user's
# ``molbuilder envs list`` or test runs if nvidia-smi is missing /
# hung.  Caching at module level means repeated recipe lookups in
# the same Python process pay it exactly once.
def _detect_host_cuda_major() -> Optional[str]:
    """Read driver-reported CUDA runtime major (e.g. ``"13"``) from
    ``nvidia-smi``.  Returns ``None`` when nvidia-smi is missing,
    times out, or the output doesn't carry a CUDA Version line.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        cp = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=2.0,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if cp.returncode != 0:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)", cp.stdout)
    return m.group(1) if m else None


def _resolve_cuda_version() -> str:
    """User env-var override > host driver probe > project default."""
    explicit = os.environ.get("MOLBUILDER_CUDA_VERSION", "").strip()
    if explicit:
        return explicit
    detected = _detect_host_cuda_major()
    if detected:
        return f"{detected}.*"
    return "13.*"


_CUDA_VERSION   = _resolve_cuda_version()
_LIBXC_VERSION  = _env_default("MOLBUILDER_LIBXC_VERSION", "")  # unpinned


def _cuda_wheel_tag(cuda_version_spec: str) -> str:
    """Derive the ``cuda<MAJOR>x`` wheel suffix used by cupy / gpu4pyscf
    PyPI releases from our ``_CUDA_VERSION`` conda spec.

    Examples (the literal returns):
      ``"13.*"``  -> ``"cuda13x"``
      ``"12.4"``  -> ``"cuda12x"``
      ``">=13"``  -> ``"cuda13x"``
      ``""``      -> ``"cuda13x"`` (safe fallback)

    Why this lives in code, not as a hardcoded literal: the project
    target moves (CUDA 11 → 12 → 13 → 14 over time) and the upstream
    wheels track it.  Hardcoding ``cuda13x`` in recipes / docs invites
    the same drift bug as hardcoding ``MOLBUILDER_CUDA_VERSION``
    elsewhere.  This helper is the single source of truth for the
    wheel suffix; every reference to a cuda-tagged Python wheel
    (recipe pip_packages, README walkthrough, validator messages,
    config help text) should derive from this.
    """
    import re
    m = re.search(r"\d+", cuda_version_spec or "")
    return f"cuda{m.group(0) if m else '13'}x"


_CUDA_WHEEL_TAG = _cuda_wheel_tag(_CUDA_VERSION)


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
        detected CUDA toolkit version.  ELPA needs ``True`` (the
        only component that itself emits CUDA kernels); SIESTA proxies
        all GPU work through ELPA and doesn't need its own CUDA
        config.
    clone_recurse_submodules
        When ``True``, ``git clone`` runs with
        ``--recurse-submodules --shallow-submodules`` so the
        component's ``External/`` submodules come along in one shot.
        SIESTA's ``rel-5.4`` branch uses this to pull libfdf,
        libpsml, xmlf90, libgridxc, ELSI, and libxc as on-the-fly
        compiled submodules (per SIESTA 5.4 INSTALL.md
        § "Required domain-specific libraries").
    clone_shallow
        When ``True`` (default), ``git clone --depth=1`` produces a
        shallow clone (faster, smaller).  When ``False``, full
        history is cloned -- needed when ``ref`` is a SHA that
        isn't reachable from the default branch in shallow mode.
    tarball_url
        When set, the clone phase downloads a tarball via curl + tar
        instead of ``git clone``.  Mutually exclusive with
        ``repo_url`` for the clone step; ``ref`` is still used for
        the toolchain fingerprint.  Used for ELPA, where conda-forge
        + Spack + EasyBuild all download from MPCDF's tarball archive
        at ``elpa.mpcdf.mpg.de/software/tarball-archive/Releases/``.
        Pre-bootstrapped (no autoreconf needed) and SHA256-pinnable.
        Templated with ``{ref}`` so the URL bumps automatically when
        ``MOLBUILDER_*_TAG`` overrides the version.
    tarball_sha256
        Optional SHA256 hex digest of the tarball for integrity
        verification.  Empty string skips the check (use only when
        a tarball_url is being bumped via env-var override to a
        version with no recorded SHA).
    tarball_inner_dir
        Templated name of the top-level directory inside the
        tarball (e.g. ``"elpa-{ref}"``).  After extraction the
        executor renames this to a stable ``<src>`` path so the
        configure_argv ``{src}`` placeholder always resolves to
        the same location.
    """
    name: str
    repo_url: str
    ref: str
    configure_argv: Tuple[str, ...]
    build_argv: Tuple[str, ...]
    install_argv: Tuple[str, ...]
    verify_argv: Tuple[str, ...] = ()
    needs_cuda: bool = False
    clone_recurse_submodules: bool = False
    clone_shallow: bool = True
    tarball_url: str = ""
    tarball_sha256: str = ""
    tarball_inner_dir: str = ""


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
    # Packages whose ABSENCE at audit time is informational rather than
    # an error.  Use case: GPU-only deps like ``cupy-cuda13x[ctk]`` and
    # ``gpu4pyscf-*`` -- the recipe install attempts them, but a no-GPU
    # host (or one where the wheel fails to install) is still a usable
    # CPU env.  Doctor reports them as "optional unavailable; GPU
    # features disabled" instead of "FAILED".  Names should match
    # entries in ``conda_packages`` / ``pip_packages`` (the install
    # path still tries to install them; only the audit treats them
    # leniently).
    optional_conda_packages: Tuple[str, ...] = ()
    optional_pip_packages: Tuple[str, ...] = ()
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
        # NUMA control tool.  Used by the host-side advisor
        # (``molbuilder envs advise siesta-gpu``) to detect whether
        # NUMA pinning is possible: its can_numa_pin property only
        # returns True when ``numactl`` is on PATH.  Without it the
        # advisor would honestly report "numa-pin: OFF (numactl not
        # installed)" even on dual-socket boxes where pinning would
        # help.  Tiny package (~200 KB); installed by default so the
        # advisor's reading matches what the GPU env's wrapper will
        # actually do at runtime.
        "numactl",
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
    description="PySCF (CPU + GPU runtime libs); Spectra-tab Raman/IR + "
                "geomeTRIC geomopt; gpu4pyscf available when use_gpu=True.",
    channels=("conda-forge",),
    conda_packages=(
        "python=3.12", "pip",
        "pyscf", "pyscf-dispersion", "geometric",
        # NUMA control tool (mirrors molbuilder-siesta-gpu).  PySCF
        # uses threaded BLAS that benefits from socket-local pinning
        # on dual-socket boxes -- ``numactl --cpunodebind`` wraps the
        # Python invocation cleanly.  Not yet auto-wired by molbuilder
        # for PySCF jobs but present so future tuning has the tool.
        "numactl",
    ),
    # GPU support: gpu4pyscf + cupy ship via PyPI (not conda-forge for
    # current versions).  Wheel suffix derived from _CUDA_VERSION via
    # _cuda_wheel_tag() so a future ``MOLBUILDER_CUDA_VERSION=14.*``
    # auto-picks ``cupy-cuda14x[ctk]`` + ``gpu4pyscf-cuda14x`` -- no
    # hand-edits to chase the toolkit bump.  The ``[ctk]`` extra on
    # cupy pulls the matching cuda-cudart conda packages.  Without
    # these the ``use_gpu`` form toggle is a no-op (the runtime probe
    # in molbuilder/runtime_info.py would land in its CPU-fallback
    # branch on every run).
    pip_packages=(
        "pyscf-properties",
        f"cupy-{_CUDA_WHEEL_TAG}[ctk]",
        f"gpu4pyscf-{_CUDA_WHEEL_TAG}",
    ),
    # cupy + gpu4pyscf are GPU-only.  On a no-GPU host (or when the wheel
    # fails to install for any reason) the env is still fully usable for
    # CPU PySCF -- only the ``use_gpu=True`` form toggle becomes a no-op.
    # Doctor marks these as "optional unavailable" rather than "FAILED".
    optional_pip_packages=(
        f"cupy-{_CUDA_WHEEL_TAG}[ctk]",
        f"gpu4pyscf-{_CUDA_WHEEL_TAG}",
    ),
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
    #
    # ``numactl`` ships in this env so the run-wrapper's NUMA-pin
    # branch (``--cpunodebind=$_gpu_numa`` on dual-socket boxes)
    # finds the binary after ``conda activate molbuilder-siesta``.
    # See molbuilder.runwrap._gpu_runtime_defaults_block for the
    # full policy + cross-check against the host advisor's
    # HostProbe.can_numa_pin.
    conda_packages=("siesta=5.4.2=mpi_openmpi_*", "numactl"),
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
    # ``bash -c`` (no -l) -- we don't need a LOGIN shell here.  -l would
    # source ~/.bash_profile / ~/.profile and pull in user-shell state
    # that can shadow the env's tools (module loads, PATH munging, etc.)
    # The env's bin/ is on PATH via _bypass_conda_run's env overrides,
    # which is the only setup tleap needs.
    verify_argv=("bash", "-c", "tleap -f /dev/null < /dev/null"),
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
#
# NOTE: ``CMAKE_PREFIX_PATH`` is component-specific (SIESTA needs
# both the env prefix AND the ELPA install dir on the search path)
# and is set inline per component, not in this shared tuple.
_PIN_MPI_TOOLS = (
    # MPI: bypass FindMPI's PATH walk -- pin compilers explicitly.
    "-DMPI_C_COMPILER={env_prefix}/bin/mpicc",
    "-DMPI_CXX_COMPILER={env_prefix}/bin/mpicxx",
    "-DMPI_Fortran_COMPILER={env_prefix}/bin/mpifort",
)

# P1.E 2026-06-16 audit cleanup: ``_PIN_CUDA_TOOLS`` was a defunct
# constant -- defined for a never-shipped cmake-based ELPA build path,
# and never referenced anywhere (verified by ``grep _PIN_CUDA_TOOLS``).
# ELPA uses autotools; SIESTA's CUDA acceleration is entirely via
# ELPA's CUDA-enabled build, so SIESTA itself needs no CUDA pin.
# The CUDA isolation we DO want (refuse to wander into /usr/local/cuda)
# is achieved via the ``CMAKE_IGNORE_PATH`` block above and via ELPA's
# ``--with-cuda-path={env_prefix}`` configure flag.

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
_RPATH_ELPA       = "$ORIGIN/../../../../lib"
_RPATH_SIESTA_BIN = "$ORIGIN/../../../../lib:$ORIGIN/../../elpa/lib"


# --------------------------------------------------------------------- #
#  Two-component build (literature + doc supported)                     #
#                                                                       #
#  Per SIESTA 5.4 INSTALL.md (verified 2026-06-15 by fetching           #
#  rel-5.4/INSTALL.md from gitlab):                                     #
#                                                                       #
#    - External ELPA is the recommended GPU path                        #
#      (§ "ELPA (native interface) (recommended)").                     #
#    - Domain-specific libs (libfdf, libpsml, xmlf90, libgridxc) are    #
#      ESL-distributed and "can be made available by instantiation of   #
#      a git submodule (it will be placed in the External/<package>     #
#      folder)" (§ "Required domain-specific libraries").  None are     #
#      packaged on conda-forge (verified by `conda search` 2026-06-15). #
#    - ELSI is similarly a SIESTA submodule under                       #
#      External/ELSI-project/elsi_interface (§ "ELSI (recommended,      #
#      used by default)": "The ELSI library interface is now by         #
#      default compiled on-the-fly. The source can be downloaded        #
#      automatically, or it can reside in a submodule").                #
#                                                                       #
#  So we clone:                                                         #
#    1. ELPA externally (CUDA must be baked in; conda-forge ELPA isn't  #
#       built with CUDA support).                                       #
#    2. SIESTA with ``--recurse-submodules`` so libfdf + libpsml +      #
#       xmlf90 + libgridxc + ELSI come along as on-the-fly compiled     #
#       submodules using SIESTA-tested versions.                        #
#                                                                       #
#  Everything else (BLAS/LAPACK/ScaLAPACK/NetCDF/HDF5/FFTW/libxc/MPI/   #
#  CUDA toolkit/compilers) comes from conda-forge packages.             #
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
#  Conda-CUDA path bridge (applied identically in configure / build /    #
#  install argvs for any component that uses conda-forge's CUDA layout)  #
#                                                                       #
#  ONE EXPLICIT ASSUMPTION, applied everywhere:                          #
#                                                                       #
#    Conda-forge CUDA packages place artifacts at:                       #
#      Headers   $CONDA_PREFIX/targets/x86_64-linux/include/  (primary)  #
#                $CONDA_PREFIX/include/                       (generic)  #
#      Libraries $CONDA_PREFIX/lib/                                      #
#                $CONDA_PREFIX/targets/x86_64-linux/lib/      (some)     #
#                                                                       #
#  Compilers need these via CPPFLAGS (headers) and LDFLAGS (libraries).  #
#  Conda's stock activate-scripts inject extra flags (``-Wl,--sort-      #
#  common``, ``-fdebug-prefix-map=...``) for binary-package portability  #
#  -- those break libtool replay through nvcc.  So the systematic        #
#  prelude is: wipe conda's stock CPPFLAGS/LDFLAGS, then re-export       #
#  EXACTLY the include + library paths the compilers need, with no       #
#  other fluff.                                                          #
#                                                                       #
#  Apply this prelude to EVERY shell-out that compiles or links against  #
#  the conda CUDA stack (configure step, make step, make-install step,   #
#  ELPA + SIESTA + any future cmake-based CUDA component).  The pattern  #
#  is small enough to inline as a string; importing as a shell `.` would #
#  introduce a separate file the recipe has to ship.                     #
# --------------------------------------------------------------------- #
_CONDA_CUDA_ENV_BRIDGE = (
    # 1. Wipe inherited flags from the conda activate scripts.  Those
    #    are tuned for conda's binary-package builds, not us.
    'unset LDFLAGS LDFLAGS_LD CPPFLAGS; '
    # 2. Re-export CPPFLAGS with both standard conda-CUDA include
    #    paths.  The targets/ subdir is where conda-forge's CUDA
    #    packages put their .h files; the bare include/ catches any
    #    generic conda packages (libxc, openblas, etc.).
    'export CPPFLAGS="-I{env_prefix}/include '
                    '-I{env_prefix}/targets/x86_64-linux/include"; '
    # 3. Re-export LDFLAGS with both standard conda-CUDA library
    #    paths, RPATH-embedding so the .so finds its deps at runtime
    #    without needing LD_LIBRARY_PATH at user level.  -Wl,-rpath
    #    is safe under nvcc (it gets passed through to ld); -Wl,
    #    --sort-common (what we wiped above) is what nvcc rejects.
    'export LDFLAGS="-L{env_prefix}/lib '
                   '-L{env_prefix}/targets/x86_64-linux/lib '
                   '-Wl,-rpath,{env_prefix}/lib '
                   '-Wl,-rpath,{env_prefix}/targets/x86_64-linux/lib"; '
)


_ELPA = BuildComponent(
    name="elpa",
    # ELPA: tarball download + autotools build.  Matches the
    # conda-forge elpa-feedstock pattern, MPCDF's documented install
    # path, and what Spack + EasyBuild use.  No git clone, no
    # autoreconf needed (tarball ships a pre-generated ``configure``
    # script).
    repo_url="",                 # not used; tarball path overrides clone
    ref=_ELPA_TAG,               # bare version string ("2021.11.001")
    tarball_url=f"{_ELPA_TARBALL_BASE}/Releases/{_ELPA_TAG}/elpa-{_ELPA_TAG}.tar.gz",
    tarball_sha256=_ELPA_SHA256,
    tarball_inner_dir=f"elpa-{_ELPA_TAG}",
    configure_argv=(
        # ELPA autotools (VPATH build from {build}).  Pre-bootstrapped
        # tarball means no autoreconf step.  Env vars before the
        # configure call set autoconf-time variables:
        #   FC/CC/CXX   -- compilers from the conda env's bin/
        #   SCALAPACK_* -- where ELPA's checks should look for libscalapack
        #
        # Why ``--enable-nvidia-gpu`` (not ``--enable-gpu``): the modern
        # Nvidia-specific naming was introduced in ELPA 2021.x; the
        # default tag (2021.11.001) and every later tag use it.  See
        # docs/engines/siesta-gpu.md for the flag history table.
        "sh", "-c",
        # Pass CFLAGS/CXXFLAGS/FCFLAGS so configure's AVX feature
        # tests can actually compile.  Without these, conda's gcc 14
        # doesn't auto-enable AVX in test programs and ELPA's
        # configure errors out with:
        #     configure: error: Could not compile a test program with
        #     AVX, try with --disable-avx, or adjust the C compiler
        #     or CFLAGS.
        # The flag list matches the one ELPA configure itself
        # suggests.  Any kernel whose intrinsics can't compile on the
        # current CPU is silently dropped by the configure script.
        'set -e; '
        'mkdir -p "{build}"; cd "{build}"; '
        'export CFLAGS="-O2 -mavx -mavx2 -mfma -msse4.1 -msse4.2 -maes"; '
        'export CXXFLAGS="$CFLAGS"; '
        'export FCFLAGS="-O2"; '
        # CUDA include + library path bridge -- single source of truth
        # for "where conda put CUDA headers + libs".  See the
        # _CONDA_CUDA_ENV_BRIDGE block at the top of this file for the
        # one explicit layout assumption.  Applied identically in the
        # configure / build / install steps below so libtool's replay
        # of CPPFLAGS / LDFLAGS gets the same setup every time --
        # which is what cured the ``cannon.c: cuda_runtime.h: No such
        # file or directory`` error on the ELPA 2024.05.001 upgrade
        # (the configure step had the include path, but the make step
        # was using its own unset+empty prelude that erased it).
        + _CONDA_CUDA_ENV_BRIDGE +
        # CC-specific kernel selection.  ELPA ships a generic NVIDIA
        # kernel (--enable-nvidia-gpu) AND optimised variants for
        # specific architecture families.  For Ampere and newer (sm_80,
        # sm_86, sm_87, sm_89, sm_90) the SM80 kernel uses cuBLAS APIs
        # only available on those GPUs.  sm_90 PTX is backward-compat
        # with sm_80 so the SM80 kernel runs on Hopper too.  Pre-Ampere
        # GPUs (Volta sm_70, Turing sm_75) fall back to the generic
        # NVIDIA kernel since the SM80 variant would fail to link.
        # 2026-06-15 bump: ELPA 2024.05.001 tightened the configure check
        # for ``--enable-nvidia-sm80-gpu`` -- despite the error message
        # wording ("sm_80 or higher"), the literal string check rejects
        # sm_86 / sm_87 / sm_89, breaking the build for every non-A100
        # Ampere GPU.  Resolution: ONLY use ``--enable-nvidia-sm80-gpu``
        # when the user's GPU is literally cc 8.0 (A100).  For cc 8.6
        # (RTX 30xx consumer Ampere), cc 8.7 (Jetson Orin), cc 8.9 (Ada
        # Lovelace, RTX 40xx), and cc 9.0 (Hopper H100), use ELPA's
        # generic NVIDIA kernel compiled NATIVELY for that cc -- no
        # PTX-JIT fallback, no kernel-tag mismatch.  The SM80 kernel is
        # an A100-specific hand-tuned path (tensor-core ops + cusolver
        # APIs only present on A100); on the consumer/datacenter Ampere
        # + Ada + Hopper GPUs, the generic NVIDIA kernel uses cuBLAS /
        # cuSOLVER and benefits from native sm_86 / sm_89 / sm_90
        # compilation paths.  Net: A100 keeps its specialised kernel,
        # everyone else gets sm-correct generic ELPA-CUDA.
        'CC_NUM={cuda_cc_numeric}; SM_EXTRA=""; CC_TAG=sm_{cuda_cc_numeric}; '
        'if [ "$CC_NUM" = "80" ]; then '
        '    SM_EXTRA="--enable-nvidia-sm80-gpu"; '  # A100-only specialised path
        'fi; '
        # AVX-512 toggle (P1.D 2026-06-16 audit): default OFF, opt-in
        # via ``MOLBUILDER_ELPA_AVX512=1`` in the user's shell.  Many
        # AMD / older-Xeon hosts lack AVX-512, AND several Intel chips
        # that DO have it downclock when running wide-vector ops -- so
        # off-by-default is conservative AND fast on most hardware.
        # Bash ``$VAR`` form, not ``${VAR:-default}``: Python's
        # ``str.format_map`` treats ``{...}`` as a template placeholder
        # regardless of the leading ``$`` (the braced form raised
        # ``KeyError: 'MOLBUILDER_ELPA_AVX512'`` at plan time).  An
        # unset ``$MOLBUILDER_ELPA_AVX512`` expands to "" which fails
        # the ``= "1"`` test cleanly -- same end effect as ``:-0``
        # without the brace collision.
        'AVX512_FLAG=--disable-avx512; '
        '[ "$MOLBUILDER_ELPA_AVX512" = "1" ] '
        '&& AVX512_FLAG=--enable-avx512 || true; '
        'echo "molbuilder ELPA: AVX-512 -> $AVX512_FLAG '
        '(set MOLBUILDER_ELPA_AVX512=1 to enable)" >&2; '
        '"{src}/configure" '
        ' FC=mpifort CC=mpicc CXX=mpicxx '
        ' --prefix={install} '
        ' --enable-shared '
        ' --enable-openmp '
        ' --enable-nvidia-gpu '
        ' $SM_EXTRA '
        ' --with-cuda-path={env_prefix} '
        ' --with-NVIDIA-GPU-compute-capability=$CC_TAG '
        # ELPA 2024.05.001 turns cuSOLVER on by default when GPU streams
        # are enabled and probes for ``cusolverDnXtrtri`` -- which only
        # exists in cuSOLVER 11.4+ (CUDA 12.1+).  Our conda-forge
        # cuda-cudart=13.* pulls a cuSOLVER that satisfies the API, but
        # the conda libcusolver_dev package is not visible to autoconf's
        # default link probe (the .so lives in a non-standard subdir).
        # Rather than fight the conda layout, disable cuSOLVER -- ELPA
        # uses it only for triangular-matrix inversion and Cholesky,
        # both of which fall back to ScaLAPACK with negligible
        # measurable cost at our problem sizes (< 10000 orbitals).
        ' --with-cusolver=no '
        ' $AVX512_FLAG '  # set by MOLBUILDER_ELPA_AVX512 above
        ' SCALAPACK_LDFLAGS="-L{env_prefix}/lib -lscalapack -lopenblas" '
        ' SCALAPACK_FCFLAGS="-I{env_prefix}/include"'
    ),
    build_argv=(
        "sh", "-c",
        # Same env bridge as configure_argv (see _CONDA_CUDA_ENV_BRIDGE
        # at the top of this file).  The make step compiles + links
        # the .c / .F90 / .cu source -- it reads CPPFLAGS for header
        # search (cuda_runtime.h, cublas_v2.h, ...) and LDFLAGS for
        # libcudart.so / libcublas.so resolution.  Without the bridge
        # the configure-time include path is invisible to make and
        # the build dies on the first .c that does ``#include
        # <cuda_runtime.h>``.
        'cd "{build}"; '
        + _CONDA_CUDA_ENV_BRIDGE +
        'make -j{jobs}'
    ),
    install_argv=(
        "sh", "-c",
        # Same env bridge as configure / build steps -- consistent
        # toolchain view across all three phases.  See
        # _CONDA_CUDA_ENV_BRIDGE for the assumption that drives this.
        'cd "{build}"; '
        + _CONDA_CUDA_ENV_BRIDGE +
        'make install'
    ),
    # ``--enable-openmp`` makes ELPA's autotools tag the library name
    # with ``_openmp`` (libelpa_openmp.so).  The non-OpenMP build would
    # produce libelpa.so instead.  Our SIESTA cmake step also looks for
    # the _openmp variant via pkg-config (elpa_openmp.pc).
    verify_argv=("test", "-f", "{install}/lib/libelpa_openmp.so"),
    needs_cuda=True,
    clone_recurse_submodules=False,
)


_SIESTA_GPU_COMPONENT = BuildComponent(
    name="siesta",
    repo_url=_SIESTA_REPO,
    ref=_SIESTA_REF,
    configure_argv=(
        "cmake",
        "-S", "{src}",
        "-B", "{build}",
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX={install}",
        # SIESTA's cmake searches both the env (for conda-installed
        # BLAS/ScaLAPACK/NetCDF/HDF5/FFTW/libxc + the CUDA toolkit) AND
        # the external ELPA install dir.  Semicolons (cmake list
        # separator), not colons.
        "-DCMAKE_PREFIX_PATH={env_prefix};{dep_elpa}",
        # cmake's default ``CMAKE_SYSTEM_PREFIX_PATH`` always includes
        # /usr/local regardless of what we pass via ``CMAKE_PREFIX_PATH``.
        # If the host has a prior SIESTA-stack install at
        # /usr/local/lib/cmake/{s-dftd3,mctc-lib,libfdf,libgridxc,
        # libpsml,xmlf90}/ -- which many workstations do -- find_package
        # picks that up FIRST.  Stale *-targets.cmake there make the
        # ``include(...)`` in the system config files abort fatally
        # before SIESTA's source fallback (the bundled External/<pkg>/
        # submodule) gets a chance.  IGNORE_PATH + IGNORE_PREFIX_PATH
        # surgically skip /usr/local for THIS build only -- nothing on
        # the host is touched.
        # P1.E 2026-06-16 audit cleanup: removed duplicate
        # ``-DCMAKE_IGNORE_PATH=...`` + ``-DCMAKE_IGNORE_PREFIX_PATH``
        # pair that used to follow this one.  cmake honors only the
        # LAST value of any repeated ``-D`` flag, so the second copy
        # was a no-op -- but readers had to verify that both copies
        # carried identical values before being sure.  Kept the
        # earlier copy + this single explanatory comment block.
        "-DCMAKE_IGNORE_PATH=/usr/local;/usr/local/lib;"
        "/usr/local/lib/cmake;/usr/local/include",
        "-DCMAKE_IGNORE_PREFIX_PATH=/usr/local",
        # Env-isolation pins (compilers + MPI from env).  SIESTA's CUDA
        # acceleration is entirely via ELPA's CUDA-enabled build, so
        # SIESTA itself needs no CUDA flags.
        *_PIN_MPI_TOOLS,
        f"-DCMAKE_INSTALL_RPATH={_RPATH_SIESTA_BIN}",
        "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
        # SIESTA features.  ELSI + libfdf + libpsml + xmlf90 + libgridxc
        # are built on-the-fly from the External/ submodule directories
        # (SIESTA's <PACKAGE>_FIND_METHOD defaults to
        # "cmake;pkgconf;source;fetch" -- the "source" step finds them
        # in External/<package>/ which our --recurse-submodules clone
        # has populated).
        "-DSIESTA_WITH_MPI=ON",
        "-DSIESTA_WITH_TRANSIESTA=ON",
        "-DSIESTA_WITH_ELSI=ON",
        "-DSIESTA_WITH_ELPA=ON",
        "-DSIESTA_WITH_LIBXC=ON",
        "-DSIESTA_WITH_NETCDF=ON",
        "-DSIESTA_WITH_OPENMP=ON",
        # Disable flook (SIESTA's runtime Lua-scripting bridge).
        # 2026-06-24 user-confirmed build failure on HPC node:
        # External/Lua-Engine/flook has a make-dependency race --
        # ``aotus/LuaFortran/wrap_lua_dump.c`` does ``#include
        # "lua.h"`` before the bundled lua-5.3.5 (which flook's
        # Makefile builds via a separate target) has been
        # configured + copied + built.  Under ``ninja -j8`` the C
        # compile races ahead of the lua build and dies on missing
        # lua.h.  Upstream flook + aotus makefile bug, not fixable
        # in our recipe without patching flook's source tree.
        # flook is an OPTIONAL Lua-scripting hook for SIESTA -- it
        # lets users embed Lua callbacks for custom MD/relaxation
        # control; core DFT, TranSIESTA, and TBtrans don't need it.
        # Turning it off is the pragmatic fix that unblocks every
        # workflow molbuilder generates today.  Re-enable when the
        # user wants Lua scripting AND upstream fixes the makefile.
        "-DSIESTA_WITH_FLOOK=OFF",
    ),
    build_argv=("cmake", "--build", "{build}", "-j", "{jobs}"),
    install_argv=("cmake", "--install", "{build}"),
    verify_argv=("{install}/bin/siesta", "--version"),
    needs_cuda=False,
    # Submodule recursion brings libfdf + libpsml + xmlf90 + libgridxc
    # + ELSI + libxc as External/ subdirectories; SIESTA's cmake
    # picks them up via FIND_METHOD=source step.  This is the path
    # SIESTA's CI tests and what its INSTALL.md documents.
    clone_recurse_submodules=True,
)


_SIESTA_GPU_BUILD = BuildSpec(
    artifact_subdir="siesta-gpu-stack",
    components=(_ELPA, _SIESTA_GPU_COMPONENT),
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
        # Toolchain.  gcc_linux-64=<N> compiler family pinned via
        # MOLBUILDER_GCC (default 14; use 13 for CUDA 12.0-12.7, 11
        # for CUDA 11.x).
        "python=3.12",
        f"gcc_linux-64={_GCC_VERSION}",
        f"gxx_linux-64={_GCC_VERSION}",
        f"gfortran_linux-64={_GCC_VERSION}",
        "cmake>=3.30", "ninja", "make", "git", "m4",
        # ``curl`` (not just libcurl).  builds.py's ELPA clone phase
        # downloads the tarball via curl; on HPC nodes the system
        # ``/usr/bin/curl`` is sometimes built without TLS / HTTPS
        # support (``curl: (4) A requested feature ... not built-in``),
        # which kills the install at step 1/10.  conda-forge's curl is
        # HTTPS-capable on every architecture we support, and the
        # build wrapper puts ``<env>/bin`` first on PATH so the env's
        # curl shadows the system one.  Bundling it explicitly also
        # avoids relying on git's transitive libcurl being CLI-shaped
        # (libcurl alone is a library, not the CLI binary).
        "curl",
        "pkg-config",
        # NUMA control tool.  Critical for GPU mode on multi-socket
        # boxes: the run-wrapper's _gpu_runtime_defaults_block wraps
        # mpirun in ``numactl --cpunodebind=$_gpu_numa --membind=$_gpu_numa``
        # to pin all ranks to the GPU-proximate socket.  Without it,
        # the 3-condition AND in _numa_pinned fails and ranks spread
        # across sockets, paying UPI/QPI crossing latency on every
        # cudaMemcpy.  Mirrors the host-side dependency added to the
        # ``molbuilder`` recipe so the advisor's reading matches the
        # wrapper's runtime behaviour.
        "numactl",
        # Autotools chain.  ELPA's build invokes ``libtool`` directly
        # (via its ``nvcc_wrap`` script) when compiling the NVIDIA-GPU
        # .cu kernels into .lo objects.  Without these, the build dies
        # with ``libtool: command not found`` partway through.
        "autoconf", "automake", "libtool",
        # CUDA toolkit (mirrors molbuilder-pySCF's pattern).  Version
        # pinned via MOLBUILDER_CUDA_VERSION (default 13.*).
        f"cuda-version={_CUDA_VERSION}",
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
        # File I/O (parallel HDF5 + netcdf for SIESTA's NetCDF backend).
        # NOTE: conda-forge's netcdf-c is built with S3 backend support
        # enabled by default, so this transitively pulls a handful of
        # ``aws-c-*`` packages (aws-c-auth, aws-c-cal, aws-c-s3, ...).
        # The AWS C SDK code paths only activate when NetCDF is asked
        # to open an ``s3://`` URL, which never happens on a local
        # workstation; ~10 MB of dormant disk is the only cost.  See
        # docs/engines/siesta-gpu.md for the documented trade.
        "fftw=*=mpi_openmpi_*",
        "hdf5=*=mpi_openmpi_*",
        "netcdf-fortran=*=mpi_openmpi_*",
        # XC functional library (highly recommended per SIESTA
        # INSTALL.md § "libxc (highly recommended)").
        _spec("libxc", f"={_LIBXC_VERSION}" if _LIBXC_VERSION else ""),
        # NOTE: the four ESL domain-specific libraries SIESTA 5.4
        # requires -- libfdf, libpsml, xmlf90, libgridxc -- are NOT
        # available on conda-forge (verified by ``conda search``
        # 2026-06-15).  They ship as SIESTA's git submodules under
        # External/<package> and SIESTA's cmake compiles them on
        # the fly when the --recurse-submodules clone (see the
        # SIESTA BuildComponent) brings them along.  This is what
        # SIESTA 5.4 INSTALL.md § "Required domain-specific
        # libraries" recommends.
    ),
    build_spec=_SIESTA_GPU_BUILD,
    verify_argv=(
        "bash", "-c",
        # _bypass_conda_run puts <env>/bin on PATH for the verify call;
        # the activate.d hook's PATH munging is duplicated by our
        # env_overrides so siesta is reachable.  ``--version`` exits 0
        # with the version banner.
        "siesta --version",
    ),
    verify_expect_contains="siesta",
    system_preconditions=(
        "(GPU runtime, OPTIONAL) NVIDIA driver + nvidia-smi on the "
        "host -- required ONLY to enable the GPU path at runtime via "
        "``Diag.ELPA.GPU .true.``.  ELPA + SIESTA build and run "
        "fine without the driver; the binary will operate CPU-only "
        "on no-GPU hosts (ELPA's CPU eigensolver path is selected "
        "transparently at runtime).  Driver is kernel-module-coupled "
        "and cannot be a conda package; install via the host package "
        "manager when GPU acceleration is wanted.",
        "(GPU runtime, OPTIONAL) NVIDIA driver supporting CUDA "
        "runtime 13.x (driver-side compat).  Same OPTIONAL caveat.",
        "Internet access for the ELPA tarball download "
        "(elpa.mpcdf.mpg.de) + the SIESTA git clone "
        "(gitlab.com/siesta-project), which recursively pulls "
        "libfdf, libpsml, xmlf90, libgridxc, ELSI submodules",
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
