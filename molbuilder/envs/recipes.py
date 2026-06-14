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

* **No GPU-build fields.**  Future GPU envs (siesta-gpu) need a build-
  from-source step; that adds a ``BuildSpec`` field on the recipe and
  is intentionally out of scope for the first ship -- discussed
  separately.  See the 2026-06-14 design conversation.
* **No automatic CUDA install.**  GPU runtime libraries that ship in
  conda envs (``cuda-cudart``, ``cupy-cuda13x[ctk]``, etc.) ARE
  install-time concerns and belong in the recipe; the underlying
  NVIDIA driver + base CUDA stack are system responsibility and the
  doctor only verifies their presence.
* **The host recipe carries no category.**  ``category=None`` means
  "not a routed env" -- doctor still reports its presence/absence,
  but ``run_tool`` never dispatches into it (the host is where
  molbuilder itself runs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple


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
        Arbitrary shell-command argv tuples to run AFTER pip installs.
        Used for the playwright env's ``python -m playwright install
        chromium`` post-step; each is dispatched via ``conda run -n
        <env> ...``.
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
    verify_argv: Tuple[str, ...] = ()
    verify_expect_contains: Optional[str] = None
    verify_ignore_exit_code: bool = False
    system_preconditions: Tuple[str, ...] = ()


# --------------------------------------------------------------------- #
#  The five built-in recipes                                            #
#                                                                       #
#  Order matches docs/README_install.md § "Setup recipes":              #
#  host, pyscf, siesta, mdtools, tests.  Each Recipe's fields must      #
#  match the README block; tests/test_envs_readme_consistency.py        #
#  enforces this and surfaces drift on either side as a failure.        #
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


BUILTIN_RECIPES: Tuple[Recipe, ...] = (
    _HOST, _PYSCF, _SIESTA, _MDTOOLS, _TESTS,
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
